/* eslint-disable no-console, @typescript-eslint/no-explicit-any */
import * as ort from "onnxruntime-web";
import { loadTitles } from "./api";
import { WordPieceTokenizer } from "./tokenizer";
import type { Movie, MovieTitle } from "./types";

// Configure WASM paths for onnxruntime-web
if (typeof window !== "undefined") {
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/";
}

// Keep static cache in memory
let cachedTitles: MovieTitle[] | null = null;
let cachedVectors: Float32Array | null = null;
let modelSession: ort.InferenceSession | null = null;
let tokenizer: WordPieceTokenizer | null = null;

let isLoading = false;
let isLoaded = false;

/**
 * Initializes and downloads the movie metadata, vector index, and SBERT model.
 * Caches SBERT ONNX session and vocabulary in memory.
 */
export async function initClientVectorEngine(): Promise<boolean> {
  if (isLoaded) return true;
  if (isLoading) {
    while (isLoading) {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    return isLoaded;
  }

  isLoading = true;
  try {
    console.log("[APEX] Initializing Client-Side WebGPU/CPU Vector Engine...");
    
    // 1. Fetch titles metadata catalog
    const titlesResult = await loadTitles(100000);
    if (!titlesResult || !titlesResult.data) {
      throw new Error("Failed to load movie metadata catalog");
    }
    cachedTitles = titlesResult.data;

    // 2. Fetch raw binary vector array buffer from FastAPI server
    const response = await fetch("/movies/vectors");
    if (!response.ok) {
      throw new Error(`Failed to load binary vectors: ${response.statusText}`);
    }
    
    const buffer = await response.arrayBuffer();
    cachedVectors = new Float32Array(buffer);
    
    // Pre-normalize all movie vectors for high-performance dot product similarity
    const numVectors = Math.floor(cachedVectors.length / 384);
    for (let i = 0; i < numVectors; i++) {
      const offset = i * 384;
      let sumSq = 0;
      for (let j = 0; j < 384; j++) {
        const val = cachedVectors[offset + j];
        sumSq += val * val;
      }
      const mag = Math.sqrt(sumSq);
      if (mag > 1e-9) {
        for (let j = 0; j < 384; j++) {
          cachedVectors[offset + j] /= mag;
        }
      }
    }
    
    // 3. Fetch SBERT vocabulary
    const vocabResponse = await fetch("https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt");
    if (!vocabResponse.ok) {
      throw new Error(`Failed to load tokenizer vocabulary: ${vocabResponse.statusText}`);
    }
    const vocabText = await vocabResponse.text();
    tokenizer = new WordPieceTokenizer(vocabText);

    // 4. Load quantized SBERT ONNX model using WebGPU (WASM fallback)
    const modelUrl = "/models/sbert_encoder.quant.onnx";
    try {
      modelSession = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ["webgpu", "wasm"],
      });
      console.log("[APEX] SBERT ONNX session initialized successfully via WebGPU/WASM.");
    } catch (e) {
      console.warn("[APEX] WebGPU failed, loading SBERT with WASM fallback:", e);
      modelSession = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ["wasm"],
      });
      console.log("[APEX] SBERT ONNX session initialized successfully via WASM fallback.");
    }

    console.log(`[APEX] Loaded ${cachedTitles.length} movies with ${cachedVectors.length / 384} vector rows in memory.`);
    isLoaded = true;
    return true;
  } catch (err) {
    console.error("[APEX] Client vector engine initialization failed:", err);
    isLoaded = false;
    return false;
  } finally {
    isLoading = false;
  }
}

/**
 * Encodes a text query locally in the browser using the quantized ONNX SBERT model.
 */
export async function encodeQueryClient(query: string): Promise<Float32Array | null> {
  if (!modelSession || !tokenizer) return null;

  // Tokenize text
  const tokenized = tokenizer.tokenize(query);

  // Convert arrays to BigInt64Array for ONNX int64 tensors
  const inputIdsTensor = new ort.Tensor("int64", BigInt64Array.from(tokenized.input_ids.map(BigInt)), [1, tokenized.input_ids.length]);
  const attentionMaskTensor = new ort.Tensor("int64", BigInt64Array.from(tokenized.attention_mask.map(BigInt)), [1, tokenized.attention_mask.length]);
  const tokenTypeIdsTensor = new ort.Tensor("int64", BigInt64Array.from(tokenized.token_type_ids.map(BigInt)), [1, tokenized.token_type_ids.length]);

  const feeds = {
    input_ids: inputIdsTensor,
    attention_mask: attentionMaskTensor,
    token_type_ids: tokenTypeIdsTensor,
  };

  // Run model forward pass
  const outputs = await modelSession.run(feeds);
  const lastHiddenState = outputs.last_hidden_state;
  const dims = lastHiddenState.dims; // [1, sequence_length, 384]
  const seqLength = dims[1];
  const data = lastHiddenState.data as Float32Array;

  // Perform Mean Pooling
  const embedding = new Float32Array(384);
  const attentionMask = tokenized.attention_mask;
  
  let validTokens = 0;
  for (let i = 0; i < seqLength; i++) {
    if (attentionMask[i] === 1) {
      validTokens++;
      const tokenOffset = i * 384;
      for (let j = 0; j < 384; j++) {
        embedding[j] += data[tokenOffset + j];
      }
    }
  }

  const divisor = Math.max(1e-9, validTokens);
  for (let j = 0; j < 384; j++) {
    embedding[j] /= divisor;
  }

  // L2 Normalize query embedding
  let norm = 0;
  for (let j = 0; j < 384; j++) {
    norm += embedding[j] * embedding[j];
  }
  norm = Math.sqrt(norm);
  norm = Math.max(1e-9, norm);

  for (let j = 0; j < 384; j++) {
    embedding[j] /= norm;
  }

  return embedding;
}

/**
 * Calculates semantic search recommendations for a text query in the client browser.
 * Bypasses network request, SBERT server encoding, and backend computation.
 */
export async function getClientTextSearch(query: string, limit = 10): Promise<Movie[] | null> {
  const ok = await initClientVectorEngine();
  if (!ok || !cachedTitles || !cachedVectors) return null;

  // Encode query text to vector
  const queryVector = await encodeQueryClient(query);
  if (!queryVector) return null;

  // Run optimized Cosine Similarity loop across all vectors
  const numMovies = cachedTitles.length;
  const scores: { index: number; score: number }[] = new Array(numMovies);

  for (let i = 0; i < numMovies; i++) {
    const movieOffset = i * 384;
    let dot = 0;

    for (let j = 0; j < 384; j++) {
      dot += queryVector[j] * cachedVectors[movieOffset + j];
    }

    scores[i] = { index: i, score: dot };
  }

  // Sort by score
  scores.sort((a, b) => b.score - a.score);

  // Map to Movie interface
  const results: Movie[] = [];
  let found = 0;

  for (let i = 0; i < scores.length && found < limit; i++) {
    const scoreItem = scores[i];
    const titleMeta = cachedTitles[scoreItem.index] as any;
    results.push({
      id: titleMeta.id,
      title: titleMeta.title,
      genres: Array.isArray(titleMeta.genres) ? titleMeta.genres.join("|") : titleMeta.genres || "",
      overview: `Highly relevant semantic search match matching relevance ${(scoreItem.score * 100).toFixed(1)}%.`,
      release_date: titleMeta.release_date || "",
      popularity: titleMeta.popularity || 1.0,
      retrieval_stage: `client_webgpu_search_${scoreItem.score.toFixed(3)}`,
      vote_average: 7.5,
      vote_count: 100,
    });
    found++;
  }

  return results;
}

/**
 * Calculates recommendations for a movie by its TMDB ID in the client browser.
 * Bypasses network request, SBERT, and server computation.
 */
export async function getClientRecommendations(movieId: number, limit = 10): Promise<Movie[] | null> {
  const ok = await initClientVectorEngine();
  if (!ok || !cachedTitles || !cachedVectors) return null;

  const targetIdx = cachedTitles.findIndex((m) => m.id === movieId);
  if (targetIdx === -1) {
    console.warn(`[APEX] Movie ID ${movieId} not found in client catalog`);
    return null;
  }

  const offset = targetIdx * 384;
  const targetVector = cachedVectors.subarray(offset, offset + 384);

  const numMovies = cachedTitles.length;
  const scores: { index: number; score: number }[] = new Array(numMovies);

  for (let i = 0; i < numMovies; i++) {
    const movieOffset = i * 384;
    let dot = 0;

    for (let j = 0; j < 384; j++) {
      dot += targetVector[j] * cachedVectors[movieOffset + j];
    }

    scores[i] = { index: i, score: dot };
  }

  scores.sort((a, b) => b.score - a.score);

  const results: Movie[] = [];
  let found = 0;

  for (let i = 0; i < scores.length && found < limit; i++) {
    const scoreItem = scores[i];
    if (scoreItem.index === targetIdx) continue;

    const titleMeta = cachedTitles[scoreItem.index] as any;
    results.push({
      id: titleMeta.id,
      title: titleMeta.title,
      genres: Array.isArray(titleMeta.genres) ? titleMeta.genres.join("|") : titleMeta.genres || "",
      overview: `Highly relevant recommendation matching similarity score ${(scoreItem.score * 100).toFixed(1)}%.`,
      release_date: titleMeta.release_date || "",
      popularity: titleMeta.popularity || 1.0,
      retrieval_stage: `client_vector_engine_${scoreItem.score.toFixed(3)}`,
      vote_average: 7.5,
      vote_count: 100,
    });
    found++;
  }

  return results;
}

/**
 * Returns current load status telemetry.
 */
export function getClientEngineStatus() {
  return {
    isLoaded,
    numMovies: cachedTitles ? cachedTitles.length : 0,
    vectorBytes: cachedVectors ? cachedVectors.byteLength : 0,
    hasSession: !!modelSession,
  };
}

/**
 * Projects the 384-dimensional SBERT vectors down to 3D coordinates using a
 * deterministic random projection matrix to preserve relative distances.
 */
export function getRealMovieNodes() {
  if (!cachedTitles || !cachedVectors) return [];

  // Initialize deterministic projection vectors
  const projX = new Float32Array(384);
  const projY = new Float32Array(384);
  const projZ = new Float32Array(384);
  for (let i = 0; i < 384; i++) {
    projX[i] = Math.sin(i * 1.7) * 150;
    projY[i] = Math.cos(i * 2.3) * 150;
    projZ[i] = Math.sin(i * 3.1) * 150;
  }

  const GENRES = ["Action", "Sci-Fi", "Drama", "Thriller", "Comedy", "Romance"];
  const GENRE_COLORS = ["#ef4444", "#06b6d4", "#10b981", "#a78bfa", "#f59e0b", "#ec4899"];

  const nodes = [];
  // Project up to 400 movies to keep Canvas rendering super smooth and crisp
  const limit = Math.min(cachedTitles.length, 400);

  for (let i = 0; i < limit; i++) {
    const movie = cachedTitles[i] as any;
    const offset = i * 384;

    let x = 0;
    let y = 0;
    let z = 0;
    for (let j = 0; j < 384; j++) {
      const val = cachedVectors[offset + j];
      x += val * projX[j];
      y += val * projY[j];
      z += val * projZ[j];
    }

    // Determine primary genre
    const genreStr = movie.genres || "";
    const primaryGenre = genreStr.split("|")[0] || "Drama";
    
    let genreIdx = GENRES.indexOf(primaryGenre);
    if (genreIdx === -1) {
      genreIdx = Math.abs(movie.id) % GENRES.length;
    }

    nodes.push({
      id: movie.id,
      title: movie.title,
      genre: GENRES[genreIdx],
      color: GENRE_COLORS[genreIdx],
      x,
      y,
      z,
    });
  }

  return nodes;
}
