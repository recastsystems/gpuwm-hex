// SPDX-License-Identifier: Apache-2.0
//! Host scoring on every core, fixed chunk order (deterministic sums).
pub fn threads() -> usize { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).min(24) }

pub fn par_map_sum<F: Fn(usize) -> f64 + Sync>(n: usize, f: F) -> f64 {
    let t = threads(); let chunk = n.div_ceil(t).max(1);
    let parts: Vec<f64> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..t).map(|k| { let f = &f; s.spawn(move || {
            let (lo, hi) = (k * chunk, ((k + 1) * chunk).min(n));
            let mut acc = 0.0; for i in lo..hi { acc += f(i); } acc }) }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect() });
    parts.iter().sum()
}
pub fn dot(a: &[f64], b: &[f64]) -> f64 { par_map_sum(a.len(), |i| a[i] * b[i]) }
pub fn norm2(a: &[f64]) -> f64 { par_map_sum(a.len(), |i| a[i] * a[i]) }
pub fn diff2(a: &[f64], b: &[f64]) -> f64 { par_map_sum(a.len(), |i| { let d = a[i] - b[i]; d * d }) }

/// SplitMix64 uniform in [-1, 1), one stream per (seed, index): order-free.
pub fn uniform(seed: u64, i: u64) -> f64 {
    let mut z = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(i.wrapping_mul(0xD1B54A32D192ED03)).wrapping_add(0x632BE59BD9B4E019);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}
pub fn random(seed: u64, n: usize) -> Vec<f64> {
    let t = threads(); let chunk = n.div_ceil(t).max(1);
    let mut v = vec![0.0; n];
    std::thread::scope(|s| { for (k, part) in v.chunks_mut(chunk).enumerate() {
        s.spawn(move || for (j, x) in part.iter_mut().enumerate() { *x = uniform(seed, (k * chunk + j) as u64); }); } });
    v
}

/// Elementwise map into a new vector on every core, fixed chunks (identical to the serial map).
/// Held lane rlw_rtrn_prol_gpoints: the harness spent ~60 s on one core converting and perturbing
/// 90 M-word launches (sampler: top thread 92-94 %), so every per-word host loop goes through this.
pub fn par_vec<T: Send + Default + Clone, F: Fn(usize) -> T + Sync>(n: usize, f: F) -> Vec<T> {
    let t = threads(); let chunk = n.div_ceil(t).max(1);
    let mut v = vec![T::default(); n];
    std::thread::scope(|s| { for (k, part) in v.chunks_mut(chunk).enumerate() { let f = &f;
        s.spawn(move || for (j, x) in part.iter_mut().enumerate() { *x = f(k * chunk + j); }); } });
    v
}
