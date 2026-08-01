use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use rayon::prelude::*;

/// Simple genetic breeding math PoC.
/// Input: population as Vec<Vec<f64>> where each individual is a vector of trait values.
/// Returns (best_individual, best_score) where score is sum of values (simple fitness).
pub fn breed(population: &Vec<Vec<f64>>, generations: usize) -> (Vec<f64>, f64) {
    if population.is_empty() {
        return (vec![], 0.0);
    }

    // Simple fitness: sum of traits
    let mut pop = population.clone();
    let mut best = pop[0].clone();
    let mut best_score = fitness(&best);

    for _ in 0..generations.max(1) {
        // Score in parallel
        let scores: Vec<f64> = pop.par_iter().map(|ind| fitness(ind)).collect();

        // Keep best
        if let Some((i, &s)) = scores.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
            if s > best_score {
                best_score = s;
                best = pop[i].clone();
            }
        }

        // Simple next generation: take top 50% and crossover
        let mut pairs: Vec<Vec<f64>> = Vec::new();
        let mut sorted_idx: Vec<usize> = (0..pop.len()).collect();
        sorted_idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());

        let cutoff = (pop.len() / 2).max(1);
        let elites: Vec<Vec<f64>> = sorted_idx.iter().take(cutoff).map(|&i| pop[i].clone()).collect();

        // Crossover + small mutation
        while pairs.len() < pop.len() {
            let a = &elites[rand_idx(elites.len())];
            let b = &elites[rand_idx(elites.len())];
            let child = crossover(a, b);
            pairs.push(mutate(child, 0.01));
        }

        pop = pairs;
    }

    (best, best_score)
}

fn fitness(ind: &Vec<f64>) -> f64 {
    ind.iter().sum()
}

fn rand_idx(len: usize) -> usize {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos();
    (nanos as usize) % len
}

fn crossover(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    let n = a.len().min(b.len());
    let mut child = vec![0.0; n];
    for i in 0..n {
        child[i] = if i % 2 == 0 { a[i] } else { b[i] };
    }
    child
}

fn mutate(mut v: Vec<f64>, rate: f64) -> Vec<f64> {
    for x in v.iter_mut() {
        // tiny deterministic mutation using simple RNG
        let delta = ((rand_idx(1000) as f64) / 1000.0 - 0.5) * rate;
        *x += delta;
    }
    v
}

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pyfunction]
fn breed_py(population: Vec<Vec<f64>>, generations: Option<usize>) -> PyResult<(Vec<f64>, f64)> {
    Ok(breed(&population, generations.unwrap_or(1)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_breed_basic() {
        let population = vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0], vec![0.0, 1.0, 0.0]];
        let (best, score) = breed(&population, 2);
        assert!(score > 0.0);
        assert_eq!(best.len(), 3);
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn engine_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(breed_py, m)?)?;
    Ok(())
}
