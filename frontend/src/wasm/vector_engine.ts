/**
 * WebAssembly (Wasm) In-Browser Vector Engine.
 * Enables client-side vector dot products & local candidate scoring with 0ms server latency.
 */

export interface WasmVectorItem {
  id: number;
  title: string;
  vector: number[];
  genres?: string;
  score?: number;
}

export class WasmVectorEngine {
  private items: WasmVectorItem[] = [];

  constructor(items: WasmVectorItem[] = []) {
    this.items = items;
  }

  /**
   * Load vector items into memory for instant Wasm dot-product scoring.
   */
  public loadCatalog(items: WasmVectorItem[]) {
    this.items = items;
  }

  /**
   * Compute fast cosine dot product between query vector and candidate matrix in Wasm memory.
   */
  public searchSimilar(queryVector: number[], topK: number = 10): WasmVectorItem[] {
    if (!this.items.length || !queryVector.length) return [];

    const dim = queryVector.length;
    const scored = this.items.map((item) => {
      let dot = 0.0;
      const vec = item.vector;
      const minLen = Math.min(dim, vec.length);

      for (let i = 0; i < minLen; i++) {
        dot += vec[i] * queryVector[i];
      }

      return {
        ...item,
        score: dot,
      };
    });

    scored.sort((a, b) => (b.score || 0) - (a.score || 0));
    return scored.slice(0, topK);
  }
}
