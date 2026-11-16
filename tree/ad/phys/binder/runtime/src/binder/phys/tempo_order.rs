// SPDX-License-Identifier: Apache-2.0
//! Per-launch derivative entry for TEMPO's column-order scatter (held-kernel lane tempo_order_scatter).
//!
//! Forward (hex pin hexcore/mpasphys/rrfs/tempo_v303.py:4031-4046, pkg18 module 477158923b00908c):
//!   tempo_order_scatter(const unsigned* cost, int n, unsigned* start, int* order):
//!     order[atomicAdd(&start[tempo_order_bucket(cost[i])], 1u)] = i      (i < n)
//! Every argument is an integer word. cost is the previous call's per-column cycle count (clock-derived),
//! start is the scanned 128-bucket histogram (tempo_order_hist + tempo_order_scan), order is the column schedule
//! tempo_warp_steps walks. Columns are independent, so the schedule changes no float word (COMMON-AD 87.1); the
//! pkg18 residue witness shows two forwards with 85,631 of 184,464 order bytes different and every other word
//! of all 19 boundaries equal.
//!
//! Derivative (this entry). Active float words written: none. Active float words read: none.
//!   TL: the tangent of each integer output is zero: d_start[0..128) = 0, d_order[0..n) = 0; d_cost passes (read only).
//!   AD: the adjoints of the overwritten integer outputs are consumed and zeroed; their contribution to every
//!       input adjoint is zero (no float path). a_cost passes. No limiter or clamp exists, so no frozen coefficient.
//! The composed window binds the same launch through wrules.rs (Identity "inactive"): the outputs are declared
//! residue (cost / hist / order), which --stable-dir gives zero direction weight, and no float kernel writes their
//! shadow words, so the window's identity and this zero map agree on every word the window weighs.
//!
//! What the launch row checks at runtime, from the live pre- and post-launch words of the replay:
//!   - the signature: four arguments (device, 4-byte n, device, device), n inside both allocations;
//!   - the forward is a schedule: start[pre] equals the exclusive scan of bucket counts of cost, start[post] =
//!     start[pre] + count, order is a permutation of 0..n, and order sorted by bucket (range hazard at :4045:
//!     an index from start[] at or past n writes outside order);
//!   - no word outside order[0..n) and start[0..128) moves (a 4 KiB guard band past each extent, cost unchanged);
//!   - the dot <TL x, y> = <x, AD y> on float32 x, y (float64 sums), the zero direction, a second reverse run
//!     bitwise, and the gain ||TL x|| / ||x|| over the argument extents.
use crate::cuda::Cuda;
use crate::{dev, submit};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

pub const FUNCTION: &str = "tempo_order_scatter";
const BUCKETS: usize = 128;
const GUARD: usize = 4096;

/// tempo_v303.py:4031-4036, bit for bit.
pub fn bucket(c: u32) -> usize {
    if c == 0 { return 127; }
    let e = 31 - c.leading_zeros() as i32;
    let m = (if e >= 2 { c >> (e - 2) } else { c << (2 - e) }) & 3;
    let key = (e << 2) | m as i32;
    (127 - key.min(127)) as usize
}

pub struct Snap {
    n: usize,
    cost_at: u64, start_at: u64, order_at: u64,
    cost: Vec<u32>, start: Vec<u32>,
    guards: Vec<(u64, Vec<u8>)>,
    order_pre_sha256: String,
}

fn ptr(live: &HashMap<u64, submit::Allocation>, a: &submit::Argument, i: usize, need: usize) -> Result<(u64, usize), String> {
    match a {
        submit::Argument::Device(s) => {
            let al = live.get(&s.allocation).ok_or(format!("argument {i}: allocation {} not live", s.allocation))?;
            if s.offset + need > al.bytes { return Err(format!("argument {i}: {need} bytes at offset {} exceed allocation {} ({} bytes)", s.offset, s.allocation, al.bytes)); }
            Ok((al.address + s.offset as u64, (al.bytes - s.offset - need).min(GUARD)))
        }
        _ => Err(format!("argument {i}: not a device pointer")),
    }
}
fn words(b: &[u8]) -> Vec<u32> { b.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect() }
fn sha(b: &[u8]) -> String { format!("{:x}", Sha256::digest(b)) }
fn bytes_u32(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() }
fn bytes(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() }

pub unsafe fn snapshot(cuda: &Cuda, live: &HashMap<u64, submit::Allocation>, args: &[submit::Argument]) -> Result<Snap, String> {
    if args.len() != 4 { return Err(format!("{FUNCTION} with {} arguments (4 expected: cost, n, start, order)", args.len())); }
    let n = match &args[1] { submit::Argument::Bytes(b) if b.len() == 4 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]), _ => return Err("argument 1: not a 4-byte int".into()) };
    if n <= 0 { return Err(format!("n = {n}")); }
    let n = n as usize;
    let (cost_at, _) = ptr(live, &args[0], 0, 4 * n)?;
    let (start_at, sg) = ptr(live, &args[2], 2, 4 * BUCKETS)?;
    let (order_at, og) = ptr(live, &args[3], 3, 4 * n)?;
    dev::sync(cuda)?;
    let cost = words(&dev::download(cuda, cost_at, 4 * n)?);
    let start = words(&dev::download(cuda, start_at, 4 * BUCKETS)?);
    let order_pre = dev::download(cuda, order_at, 4 * n)?;
    let mut guards = Vec::new();
    for (at, g) in [(start_at + 4 * BUCKETS as u64, sg), (order_at + 4 * n as u64, og)] {
        if g > 0 { guards.push((at, dev::download(cuda, at, g)?)); }
    }
    Ok(Snap { n, cost_at, start_at, order_at, cost, start, guards, order_pre_sha256: sha(&order_pre) })
}

/// xorshift64* stream mapped to float32 in [-1, 1).
fn draws(seed: u64, k: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..k).map(|_| { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; let r = s.wrapping_mul(0x2545f4914f6cdd1d);
                     ((r >> 40) as f32) * (1.0 / 8388608.0) - 1.0 }).collect()
}

/// Shadow layout over the argument extents: [cost n | start 128 | order n], float32 words.
struct Layout { n: usize }
impl Layout {
    fn len(&self) -> usize { 2 * self.n + BUCKETS }
    /// TL of the launch: integer outputs get a zero tangent, the read-only cost passes.
    fn tl(&self, x: &[f32]) -> Vec<f32> { let mut t = x.to_vec(); for w in self.n..self.len() { t[w] = 0.0; } t }
    /// AD of the launch: output adjoints consumed and zeroed (no float path into any input), cost adjoint passes.
    fn ad(&self, y: &[f32]) -> Vec<f32> { let mut a = y.to_vec(); for w in self.n..self.len() { a[w] = 0.0; } a }
}
fn dot64(a: &[f32], b: &[f32]) -> f64 { a.iter().zip(b).map(|(&p, &q)| p as f64 * q as f64).sum() }

pub unsafe fn launch_dot(cuda: &Cuda, s: &Snap, seed: u64, tol: f64, checks: bool) -> Result<Value, String> {
    let t0 = std::time::Instant::now();
    dev::sync(cuda)?;
    let n = s.n;
    let cost_post = words(&dev::download(cuda, s.cost_at, 4 * n)?);
    let start_post = words(&dev::download(cuda, s.start_at, 4 * BUCKETS)?);
    let order_raw = dev::download(cuda, s.order_at, 4 * n)?;
    let order: Vec<i32> = order_raw.chunks_exact(4).map(|w| i32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect();
    // forward witness: the launch is a schedule
    let mut count = [0u32; BUCKETS];
    for &c in &s.cost { count[bucket(c)] += 1; }
    let (mut run, mut scan_ok, mut post_ok) = (0u32, true, true);
    for b in 0..BUCKETS {
        if s.start[b] != run { scan_ok = false; }
        if start_post[b] != s.start[b].wrapping_add(count[b]) { post_ok = false; }
        run += count[b];
    }
    let mut seen = vec![false; n];
    let (mut out_of_range, mut duplicates) = (0u64, 0u64);
    for &o in &order { if o < 0 || o as usize >= n { out_of_range += 1; } else if seen[o as usize] { duplicates += 1; } else { seen[o as usize] = true; } }
    let mut misplaced = 0u64;
    for b in 0..BUCKETS {
        for p in (s.start[b] as usize).min(n)..(start_post[b] as usize).min(n) {
            let o = order[p];
            if o >= 0 && (o as usize) < n && bucket(s.cost[o as usize]) != b { misplaced += 1; }
        }
    }
    let cost_moved = s.cost.iter().zip(&cost_post).filter(|(a, b)| a != b).count();
    let mut guard_moved = 0usize;
    for (at, g) in &s.guards { let now = dev::download(cuda, *at, g.len())?; guard_moved += now.iter().zip(g).filter(|(a, b)| a != b).count(); }
    let permutation = out_of_range == 0 && duplicates == 0 && seen.iter().all(|&v| v);
    let schedule_ok = scan_ok && post_ok && permutation && misplaced == 0 && cost_moved == 0 && guard_moved == 0;
    // derivative
    let l = Layout { n };
    let x = draws(seed, l.len());
    let y = draws(seed ^ 0x9e3779b97f4a7c15, l.len());
    let t_tl = std::time::Instant::now();
    let tlx = l.tl(&x);
    let tl_ms = t_tl.elapsed().as_secs_f64() * 1e3;
    let t_ad = std::time::Instant::now();
    let ady = l.ad(&y);
    let ad_ms = t_ad.elapsed().as_secs_f64() * 1e3;
    let (lhs, rhs) = (dot64(&tlx, &y), dot64(&x, &ady));
    let abs = (lhs - rhs).abs();
    let rel = abs / lhs.abs().max(rhs.abs()).max(f64::MIN_POSITIVE);
    let gain = (dot64(&tlx, &tlx) / dot64(&x, &x)).sqrt();
    let tl_out_norm = tlx[n..].iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
    let (mut zero_tl, mut zero_ad, mut reverse_diff) = (0usize, 0usize, 0usize);
    if checks {
        let z = vec![0f32; l.len()];
        zero_tl = l.tl(&z).iter().filter(|v| v.to_bits() != 0).count();
        zero_ad = l.ad(&z).iter().filter(|v| v.to_bits() != 0).count();
        let again = l.ad(&y);
        reverse_diff = again.iter().zip(&ady).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    }
    let pass = schedule_ok && rel <= tol && zero_tl == 0 && zero_ad == 0 && reverse_diff == 0 && tl_out_norm == 0.0;
    Ok(json!({"status": if pass { "PASS" } else { "FAIL" }, "row": "ad.held.tempo_order/v1", "unit": "TEMPO order",
        "derivative": "zero map on integer outputs (start, order), read-only cost passes; no active float word",
        "lhs": lhs, "rhs": rhs, "abs": abs, "rel": rel, "tol": tol, "checks": checks,
        "gain_tl_over_x": gain, "tl_output_norm": tl_out_norm, "active_float_words": 0, "shadow_words": l.len(),
        "zero_tl_nonzero_words": zero_tl, "zero_ad_nonzero_words": zero_ad, "reverse_differing_words": reverse_diff,
        "result_sha256": sha(&[bytes(&tlx), bytes(&ady)].concat()),
        "schedule": {"ok": schedule_ok, "n": n, "scan_ok": scan_ok, "start_post_ok": post_ok, "permutation": permutation,
                     "out_of_range": out_of_range, "duplicates": duplicates, "misplaced": misplaced,
                     "cost_words_moved": cost_moved, "guard_bytes_moved": guard_moved,
                     "guard_bytes": s.guards.iter().map(|g| g.1.len()).sum::<usize>(),
                     "buckets_used": count.iter().filter(|&&c| c > 0).count(), "start_total": start_post[BUCKETS - 1],
                     "order_pre_sha256": s.order_pre_sha256, "order_post_sha256": sha(&order_raw), "cost_sha256": sha(&bytes_u32(&s.cost))},
        "tl_ms": tl_ms, "ad_ms": ad_ms, "dot_ms": t0.elapsed().as_secs_f64() * 1e3}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn bucket_matches_source() {
        assert_eq!(bucket(0), 127); assert_eq!(bucket(1), 127); assert_eq!(bucket(u32::MAX), 0);
        assert_eq!(bucket(4), 127 - 8); assert_eq!(bucket(7), 127 - 11);
    }
}
