use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use numpy::PyArray1;

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

#[pymodule]
fn rust_core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(blend_scores_rust, m)?)?;
    Ok(())
}
