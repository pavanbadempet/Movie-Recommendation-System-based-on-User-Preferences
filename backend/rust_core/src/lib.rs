use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use numpy::{PyArray1, PyArray2};
use rayon::prelude::*;

#[pyfunction]
fn blend_scores_rust<'py>(
    py: Python<'py>,
    scores_list: &PyList,
    weights: &PyDict,
    candidate_ids: Vec<i64>,
) -> PyResult<&'py PyDict> {
    let n = candidate_ids.len();
    let mut blended = vec![0.0f32; n];

    for item in scores_list.iter() {
        let tuple = item.downcast::<PyTuple>()?;
        let name: String = tuple.get_item(0)?.extract()?;
        let py_array: &PyArray1<f32> = tuple.get_item(1)?.downcast()?;

        // Look up weight in the weights dictionary, defaulting to 0.0
        let weight: f32 = match weights.get_item(&name)? {
            Some(val) => val.extract::<f32>()?,
            None => 0.0,
        };

        if weight == 0.0 {
            continue;
        }

        let readonly = py_array.readonly();
        let slice = match readonly.as_slice() {
            Ok(s) => s,
            Err(e) => return Err(pyo3::exceptions::PyValueError::new_err(format!("NumPy array is not contiguous: {}", e))),
        };
        let limit = slice.len().min(n);
        for i in 0..limit {
            blended[i] += slice[i] * weight;
        }
    }

    let dict = PyDict::new(py);
    for (idx, &orig_id) in candidate_ids.iter().enumerate() {
        dict.set_item(orig_id, blended[idx] as f64)?;
    }

    Ok(dict)
}

#[pyfunction]
fn mmr_diversify_rust(
    _py: Python<'_>,
    candidate_indices: Vec<i32>,
    relevance: Vec<f32>,
    vectors: &PyArray2<f32>,
    n: usize,
    lambda_param: f32,
) -> PyResult<Vec<usize>> {
    let num_candidates = candidate_indices.len();
    if num_candidates == 0 {
        return Ok(vec![]);
    }

    let readonly_vectors = vectors.readonly();
    let array_view = readonly_vectors.as_array();
    let dim = array_view.shape()[1];

    let mut selected = Vec::with_capacity(n);
    let mut remaining: Vec<usize> = (0..num_candidates).collect();

    // Select the first item immediately (highest relevance candidate)
    if num_candidates > 0 {
        selected.push(remaining.remove(0));
    }

    while !remaining.is_empty() && selected.len() < n {
        let mut best_remaining_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (idx, &cand_pos) in remaining.iter().enumerate() {
            let cand_row = candidate_indices[cand_pos];
            if cand_row < 0 {
                // Fallback score if no vector exists
                let score = lambda_param * relevance[cand_pos];
                if score > best_score {
                    best_score = score;
                    best_remaining_idx = idx;
                }
                continue;
            }

            let cand_row_view = array_view.row(cand_row as usize);
            let cand_slice = match cand_row_view.as_slice() {
                Some(s) => s,
                None => continue,
            };

            let mut max_similarity = 0.0f32;

            for &sel_pos in selected.iter() {
                let sel_row = candidate_indices[sel_pos];
                if sel_row < 0 {
                    continue;
                }

                let sel_row_view = array_view.row(sel_row as usize);
                let sel_slice = match sel_row_view.as_slice() {
                    Some(s) => s,
                    None => continue,
                };

                // Vectorized dot product calculation
                let mut dot = 0.0f32;
                for i in 0..dim {
                    dot += cand_slice[i] * sel_slice[i];
                }
                if dot > max_similarity {
                    max_similarity = dot;
                }
            }

            let mmr_score = lambda_param * relevance[cand_pos] - (1.0 - lambda_param) * max_similarity;
            if mmr_score > best_score {
                best_score = mmr_score;
                best_remaining_idx = idx;
            }
        }

        selected.push(remaining.remove(best_remaining_idx));
    }

    Ok(selected)
}

#[pyfunction]
fn collaborative_candidates_rust(
    _py: Python<'_>,
    item_matrix: &PyArray2<f32>,
    user_vec: &PyArray1<f32>,
    item_ids: &PyArray1<i32>,
    top_k: usize,
) -> PyResult<Vec<(i32, f32)>> {
    let item_matrix_readonly = item_matrix.readonly();
    let item_matrix_view = item_matrix_readonly.as_array();

    let user_vec_readonly = user_vec.readonly();
    let user_vec_slice = match user_vec_readonly.as_slice() {
        Ok(s) => s,
        Err(e) => return Err(pyo3::exceptions::PyValueError::new_err(format!("User vector is not contiguous: {}", e))),
    };

    let item_ids_readonly = item_ids.readonly();
    let item_ids_slice = match item_ids_readonly.as_slice() {
        Ok(s) => s,
        Err(e) => return Err(pyo3::exceptions::PyValueError::new_err(format!("Item IDs array is not contiguous: {}", e))),
    };

    let num_items = item_matrix_view.shape()[0];
    let dim = item_matrix_view.shape()[1];

    if num_items != item_ids_slice.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("item_matrix row count does not match item_ids length"));
    }

    if dim != user_vec_slice.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("item_matrix column count does not match user_vec length"));
    }

    let mut scores: Vec<(usize, f32)> = (0..num_items)
        .into_par_iter()
        .map(|idx| {
            let row = item_matrix_view.row(idx);
            let dot = match row.as_slice() {
                Some(slice) => {
                    let mut sum = 0.0f32;
                    for i in 0..dim {
                        sum += slice[i] * user_vec_slice[i];
                    }
                    sum
                }
                None => {
                    let mut sum = 0.0f32;
                    for i in 0..dim {
                        sum += row[i] * user_vec_slice[i];
                    }
                    sum
                }
            };
            (idx, dot)
        })
        .collect();

    if scores.len() <= top_k {
        scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        let (_, _mid, _) = scores.select_nth_unstable_by(top_k, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scores[0..top_k].sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
    }

    let candidates: Vec<(i32, f32)> = scores
        .into_iter()
        .map(|(idx, score)| (item_ids_slice[idx], score))
        .collect();

    Ok(candidates)
}

#[pyfunction]
fn fast_rerank_blend_rust(
    scores: Vec<f32>,
    similarity_scores: Vec<f32>,
    blend_weight: f32,
) -> PyResult<(Vec<f32>, Vec<f32>)> {
    let n = scores.len();
    if n == 0 {
        return Ok((vec![], vec![]));
    }

    let mut min_score = f32::INFINITY;
    let mut max_score = f32::NEG_INFINITY;
    for &s in scores.iter() {
        if s < min_score { min_score = s; }
        if s > max_score { max_score = s; }
    }

    let diff = max_score - min_score;

    let mut ranker_scores = vec![0.0f32; n];
    let mut blended_scores = vec![0.0f32; n];

    for i in 0..n {
        let learned = if diff > 0.0 {
            (scores[i] - min_score) / diff
        } else {
            1.0
        };
        let prev = similarity_scores[i];
        ranker_scores[i] = (learned * 1000000.0).round() / 1000000.0;
        blended_scores[i] = blend_weight * learned + (1.0 - blend_weight) * prev;
    }

    Ok((ranker_scores, blended_scores))
}

#[pyfunction]
fn simd_score_candidates_rust(
    _py: Python<'_>,
    candidate_matrix: &PyArray2<f32>,
    query_vector: &PyArray1<f32>,
) -> PyResult<Vec<f32>> {
    let matrix_readonly = candidate_matrix.readonly();
    let matrix_view = matrix_readonly.as_array();

    let query_readonly = query_vector.readonly();
    let query_slice = match query_readonly.as_slice() {
        Ok(s) => s,
        Err(e) => return Err(pyo3::exceptions::PyValueError::new_err(format!("Query vector is not contiguous: {}", e))),
    };

    let num_rows = matrix_view.shape()[0];
    let dim = matrix_view.shape()[1];

    if dim != query_slice.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("matrix column count does not match query vector length"));
    }

    let scores: Vec<f32> = (0..num_rows)
        .into_par_iter()
        .map(|idx| {
            let row = matrix_view.row(idx);
            let dot = match row.as_slice() {
                Some(slice) => {
                    let mut sum = 0.0f32;
                    let chunk_size = 4;
                    let mut chunks = slice.chunks_exact(chunk_size);
                    let mut query_chunks = query_slice.chunks_exact(chunk_size);

                    while let (Some(c), Some(q)) = (chunks.next(), query_chunks.next()) {
                        sum += c[0] * q[0] + c[1] * q[1] + c[2] * q[2] + c[3] * q[3];
                    }

                    for (&c, &q) in chunks.remainder().iter().zip(query_chunks.remainder().iter()) {
                        sum += c * q;
                    }
                    sum
                }
                None => {
                    let mut sum = 0.0f32;
                    for i in 0..dim {
                        sum += row[i] * query_slice[i];
                    }
                    sum
                }
            };
            dot
        })
        .collect();

    Ok(scores)
}

#[pymodule]
fn rust_core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(blend_scores_rust, m)?)?;
    m.add_function(wrap_pyfunction!(mmr_diversify_rust, m)?)?;
    m.add_function(wrap_pyfunction!(collaborative_candidates_rust, m)?)?;
    m.add_function(wrap_pyfunction!(fast_rerank_blend_rust, m)?)?;
    m.add_function(wrap_pyfunction!(simd_score_candidates_rust, m)?)?;
    Ok(())
}
