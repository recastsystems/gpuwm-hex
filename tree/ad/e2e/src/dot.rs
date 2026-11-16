// SPDX-License-Identifier: Apache-2.0
//! Per-launch dot test on the real trajectory. The derivative of one launch is
//! taken as a map on the whole derivative memory it binds (every shadow span
//! plus scalar-parameter cells): untouched words are the identity, overwritten
//! words must be consumed and cleared by the adjoint. With x, y random there,
//!     <y, M x - x>  ==  <M^T y - y, x>   (+ scalar-parameter terms),
//! i.e. <y, Mx> == <M^T y, x> with the identity part cancelled exactly
//! (unchanged words contribute an exact zero). On selected launches also: the
//! zero direction gives exact zeros, and two reverse runs are bitwise equal.
//! All reductions of a phase are queued and read back once.
use crate::cuda::Cuda;
use crate::deriv::{Bound, KernelDeriv, LaunchInfo};
use crate::dev::{self, Arena, Support};
use crate::submit;
use serde_json::{json, Value};
use std::collections::HashMap;

pub fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}
pub fn unit(seed: u64, i: u64) -> f64 { (mix(seed ^ mix(i)) >> 11) as f64 * (2.0 / 9007199254740992.0) - 1.0 }

pub unsafe fn put_cell(cuda: &Cuda, (a, w): (u64, usize), v: f64) -> Result<(), String> {
    if w == 8 { dev::upload(cuda, a, &v.to_le_bytes()) } else { dev::upload(cuda, a, &(v as f32).to_le_bytes()) }
}
pub unsafe fn get_cell(cuda: &Cuda, (a, w): (u64, usize)) -> Result<(f64, Vec<u8>), String> {
    let raw = dev::download(cuda, a, w)?;
    let v = if w == 8 { f64::from_le_bytes(raw[..8].try_into().unwrap()) } else { f32::from_le_bytes(raw[..4].try_into().unwrap()) as f64 };
    Ok((v, raw))
}

pub struct Run<'a> {
    pub cuda: &'a Cuda,
    pub support: &'a Support,
    pub live: &'a HashMap<u64, submit::Allocation>,
}

impl Run<'_> {
    unsafe fn phase(&self, kd: &KernelDeriv, arena: &mut Arena, b: &mut Bound, l: &LaunchInfo, entries: &[String]) -> Result<(), String> {
        for e in entries { kd.launch(self.cuda, arena, b, l, self.live, e)?; }
        Ok(())
    }
    unsafe fn clear_shadows(&self, b: &Bound) -> Result<(), String> {
        for s in b.shadows.values() { dev::zero(self.cuda, s.address, s.bytes)?; }
        Ok(())
    }
    pub(crate) unsafe fn tangent(&self, kd: &KernelDeriv, arena: &mut Arena, b: &mut Bound, l: &LaunchInfo) -> Result<(), String> {
        kd.zero_workspaces(self.cuda, b)?;
        kd.begin_tl(self.cuda, self.support, b, l)?;
        self.phase(kd, arena, b, l, &kd.tl_order)?;
        kd.finish_tl(self.cuda, self.support, b, l)
    }
    unsafe fn reverse(&self, kd: &KernelDeriv, arena: &mut Arena, b: &mut Bound, l: &LaunchInfo) -> Result<(), String> {
        for cell in b.ad_cells.values() { dev::zero(self.cuda, cell.0, 8)?; }
        kd.snapshot_seeds(self.cuda, b, l)?;
        kd.begin_ad(self.cuda, self.support, b, l)?;
        kd.zero_workspaces(self.cuda, b)?;
        self.phase(kd, arena, b, l, &kd.ad_order)?;
        kd.finish_ad(self.cuda, b, l, Some(self.support))
    }

    /// Called after the forward launch has been issued. `checks` adds the zero
    /// direction and the second reverse run.
    pub unsafe fn launch_dot(&self, kd: &KernelDeriv, arena: &mut Arena, b: &mut Bound, l: &LaunchInfo, seed: u64,
                             tol: f64, checks: bool) -> Result<Value, String> {
        let (cuda, sup) = (self.cuda, self.support);
        let shadows: Vec<_> = b.shadows.values().copied().collect();
        let mut x0 = Vec::new(); let mut y = Vec::new(); let mut r = Vec::new();
        for s in &shadows {
            x0.push(arena.take(s.bytes)?); y.push(arena.take(s.bytes)?);
            if checks { r.push(arena.take(s.bytes)?); }
        }
        let t0 = std::time::Instant::now();
        // The aug entry (if any) records its tape from the pre-launch primal once.
        if kd.aug { self.phase(kd, arena, b, l, &["aug".to_owned()])?; }
        sup.reset_words(cuda)?;
        let n = shadows.len();
        let mut zero_cells = [0u64; 2];
        if checks {
            self.clear_shadows(b)?;
            for cell in b.tl_cells.values() { dev::zero(cuda, cell.0, 8)?; }
            self.tangent(kd, arena, b, l)?;
            for (i, s) in shadows.iter().enumerate() { sup.words_into(cuda, i, s.address, 0, s.bytes)?; }
            self.clear_shadows(b)?;
            self.reverse(kd, arena, b, l)?;
            for (i, s) in shadows.iter().enumerate() { sup.words_into(cuda, n + i, s.address, 0, s.bytes)?; }
            for cell in b.ad_cells.values() { if get_cell(cuda, *cell)?.1.iter().any(|&x| x != 0) { zero_cells[1] += 1; } }
        }
        // Tangent along a random direction over all bound derivative memory.
        let mut scalar_x = Vec::new();
        for (i, s) in shadows.iter().enumerate() {
            sup.fill(cuda, x0[i], s.bytes, s.width, mix(seed ^ (2 * i as u64 + 1)))?;
            dev::copy(cuda, s.address, x0[i], s.bytes)?;
        }
        // AD_DOT_SCALARS=0 (diagnostic): scalar-parameter directions stay zero, so the dot covers array memory only
        // (separates a float32 scalar-cotangent reduction from the array map).
        let scalars_on = std::env::var("AD_DOT_SCALARS").as_deref() != Ok("0");
        for (k, (field, cell)) in b.tl_cells.iter().enumerate() {
            let v = if scalars_on { unit(seed ^ 0x5ca1_ab1e, k as u64) } else { 0.0 };
            put_cell(cuda, *cell, v)?;
            scalar_x.push((*field, v));
        }
        let t1 = std::time::Instant::now();
        self.tangent(kd, arena, b, l)?;
        dev::sync(cuda)?;
        let tl_ms = t1.elapsed().as_secs_f64() * 1e3;
        // slots: 3i lhs, 3i+1 |Mx-x|^2, 3i+2 |y|^2
        for (i, s) in shadows.iter().enumerate() {
            sup.fill(cuda, y[i], s.bytes, s.width, mix(seed ^ (2 * i as u64 + 2)))?;
            sup.dotdiff_into(cuda, 3 * i, y[i], s.address, x0[i], s.bytes, s.width)?;
            sup.sqdiff_into(cuda, 3 * i + 1, s.address, x0[i], s.bytes, s.width)?;
            sup.sqdiff_into(cuda, 3 * i + 2, y[i], 0, s.bytes, s.width)?;
        }
        // Adjoint of the same launch from y (the reductions above are queued first on the same stream).
        for (i, s) in shadows.iter().enumerate() { dev::copy(cuda, s.address, y[i], s.bytes)?; }
        let t2 = std::time::Instant::now();
        self.reverse(kd, arena, b, l)?;
        dev::sync(cuda)?;
        let ad_ms = t2.elapsed().as_secs_f64() * 1e3;
        let base = 3 * n;
        for (i, s) in shadows.iter().enumerate() {
            sup.dotdiff_into(cuda, base + 3 * i, x0[i], s.address, y[i], s.bytes, s.width)?;
            sup.sqdiff_into(cuda, base + 3 * i + 1, s.address, y[i], s.bytes, s.width)?;
            sup.sqdiff_into(cuda, base + 3 * i + 2, x0[i], 0, s.bytes, s.width)?;
            if checks { dev::copy(cuda, r[i], s.address, s.bytes)?; }
        }
        let mut cells_first = Vec::new();
        let mut scalar = (0.0, 0.0, 0.0);
        for (field, v) in &scalar_x {
            let (g, raw) = get_cell(cuda, b.ad_cells[field])?;
            scalar.0 += g * v; scalar.1 += v * v; scalar.2 += g * g;
            cells_first.push(raw);
        }
        let mut differing = 0u64;
        if checks {
            for (i, s) in shadows.iter().enumerate() { dev::copy(cuda, s.address, y[i], s.bytes)?; }
            self.reverse(kd, arena, b, l)?;
            for (i, s) in shadows.iter().enumerate() { sup.words_into(cuda, 2 * n + i, s.address, r[i], s.bytes)?; }
            for (k, (field, _)) in scalar_x.iter().enumerate() {
                if get_cell(cuda, b.ad_cells[field])?.1 != cells_first[k] { differing += 1; }
            }
        }
        let fired = kd.check_failures(cuda, b)?;
        let (sums, words) = sup.collect(cuda, 6 * n, if checks { 3 * n } else { 0 })?;
        let (mut lhs, mut dtl2, mut ny2, mut rhs, mut dad2, mut nx2) = (0.0, 0.0, 0.0, scalar.0, scalar.2, scalar.1);
        for i in 0..n {
            lhs += sums[3 * i]; dtl2 += sums[3 * i + 1]; ny2 += sums[3 * i + 2];
            rhs += sums[base + 3 * i]; dad2 += sums[base + 3 * i + 1]; nx2 += sums[base + 3 * i + 2];
        }
        let (mut tl_zero, mut ad_zero) = (0u64, zero_cells[1]);
        if checks {
            for i in 0..n { tl_zero += words[i][1]; ad_zero += words[n + i][1]; differing += words[2 * n + i][0]; }
        }
        let abs = (lhs - rhs).abs();
        let scale = f64::max(lhs.abs(), rhs.abs());
        let rel = if scale == 0.0 { if abs == 0.0 { 0.0 } else { f64::INFINITY } } else { abs / scale };
        let denominator = (ny2 * dtl2).sqrt() + (nx2 * dad2).sqrt();
        let ns = if denominator == 0.0 { if abs == 0.0 { 0.0 } else { f64::INFINITY } } else { abs / denominator };
        let finite = lhs.is_finite() && rhs.is_finite();
        // Physics-on window L3: name where a nonfinite result sits (allocation, first word, count), only on a
        // NONFINITE row, after every phase (the shadows then hold the second reverse run or the first).
        let mut nonfinite_at = Vec::new();
        if !finite {
            for (alloc, sh) in b.shadows.iter() {
                let raw = dev::download(cuda, sh.address, sh.bytes)?;
                let bad: Vec<usize> = if sh.width == 8 {
                    raw.chunks_exact(8).enumerate().filter(|(_, c)| !f64::from_le_bytes((*c).try_into().unwrap()).is_finite()).map(|(i, _)| i).collect()
                } else {
                    raw.chunks_exact(4).enumerate().filter(|(_, c)| !f32::from_le_bytes((*c).try_into().unwrap()).is_finite()).map(|(i, _)| i).collect()
                };
                if !bad.is_empty() { nonfinite_at.push(json!({"allocation": alloc, "first_word": bad[0], "last_word": bad[bad.len() - 1], "words": bad.len(), "shadow_words": sh.bytes / sh.width})); }
            }
            for (field, cell) in b.ad_cells.iter() {
                if !get_cell(cuda, *cell)?.0.is_finite() { nonfinite_at.push(json!({"scalar_field": field})); }
            }
        }
        let status = if !fired.is_empty() { "REFUSED" }
            else if !finite { "NONFINITE" }
            else if tl_zero != 0 || ad_zero != 0 { "ZERO-FAIL" }
            else if differing != 0 { "NONDETERMINISTIC" }
            else if rel <= tol { "PASS" }
            else if ns <= tol { "NS-ONLY" }
            else { "FAIL" };
        // Diagnostic (AD_DOT_PAIRS=<function>): the dot split per (direction allocation, seed allocation) pair.
        let pairs = if std::env::var("AD_DOT_PAIRS").ok().as_deref() == Some(kd.function.as_str()) {
            Some(self.pairs(kd, arena, b, l, seed, &x0, &y)?) } else { None };
        // Diagnostic (AD_DOT_GAIN=<function>, ad/held-tempo_to): one more tangent along the first direction x, read
        // back: ||M x|| / ||x|| over the bound memory, and over the words the tangent changed (M x != x bitwise, the
        // words the launch writes and that carry a direction) ||(M x)_W|| / ||x_W||, plus ||(M x - x)|| / ||x||.
        let gain = if std::env::var("AD_DOT_GAIN").ok().as_deref() == Some(kd.function.as_str()) {
            self.clear_shadows(b)?;
            for c in b.tl_cells.values() { dev::zero(cuda, c.0, 8)?; }
            for (i, s) in shadows.iter().enumerate() { dev::copy(cuda, s.address, x0[i], s.bytes)?; }
            for (k, (_, cell)) in b.tl_cells.iter().enumerate() { put_cell(cuda, *cell, if scalars_on { unit(seed ^ 0x5ca1_ab1e, k as u64) } else { 0.0 })?; }
            self.tangent(kd, arena, b, l)?;
            dev::sync(cuda)?;
            let (mut xx, mut mm, mut wx, mut wm, mut dd, mut nw) = (0f64, 0f64, 0f64, 0f64, 0f64, 0u64);
            let mut per = Vec::new();
            for (i, s) in shadows.iter().enumerate() {
                let a = dev::download(cuda, x0[i], s.bytes)?; let m = dev::download(cuda, s.address, s.bytes)?;
                let (mut pwx, mut pwm, mut pnw) = (0f64, 0f64, 0u64);
                for (u, v) in a.chunks_exact(s.width).zip(m.chunks_exact(s.width)) {
                    let (x, y) = if s.width == 8 { (f64::from_le_bytes(u.try_into().unwrap()), f64::from_le_bytes(v.try_into().unwrap())) }
                                 else { (f32::from_le_bytes(u.try_into().unwrap()) as f64, f32::from_le_bytes(v.try_into().unwrap()) as f64) };
                    xx += x * x; mm += y * y; dd += (y - x) * (y - x);
                    if u != v { pwx += x * x; pwm += y * y; pnw += 1; }
                }
                wx += pwx; wm += pwm; nw += pnw;
                if pnw > 0 { per.push(json!({"shadow_index": i, "changed_words": pnw, "x_norm": pwx.sqrt(), "tl_norm": pwm.sqrt(),
                                             "ratio": if pwx > 0.0 { (pwm / pwx).sqrt() } else { f64::INFINITY }})); }
            }
            let ids: Vec<u64> = b.shadows.keys().copied().collect();
            for p in per.iter_mut() { let i = p["shadow_index"].as_u64().unwrap() as usize; p["allocation"] = json!(ids.get(i)); }
            Some(json!({"x_norm": xx.sqrt(), "tl_norm": mm.sqrt(), "tl_over_x": (mm / xx.max(1e-300)).sqrt(),
                        "change_over_x": (dd / xx.max(1e-300)).sqrt(), "changed_words": nw, "changed_x_norm": wx.sqrt(),
                        "changed_tl_norm": wm.sqrt(), "changed_tl_over_x": if wx > 0.0 { (wm / wx).sqrt() } else { f64::INFINITY },
                        "per_allocation": per}))
        } else { None };
        Ok(json!({"status":status,"lhs":lhs,"rhs":rhs,"abs":abs,"rel":rel,"norm_scaled":ns,"pairs":pairs,"gain":gain,
                  "tl_change_norm":dtl2.sqrt(),"ad_change_norm":dad2.sqrt(),"checks":checks,
                  "zero_tl_nonzero_words":tl_zero,"zero_ad_nonzero_words":ad_zero,
                  "reverse_differing_words":differing,"refusals":fired,
                  "shadow_bytes":shadows.iter().map(|s| s.bytes).sum::<usize>(),
                  "scalar_parameters":scalar_x.len(),
                  "nonfinite_at":nonfinite_at,"scalar_term":scalar.0,
                  "tl_ms":tl_ms,"ad_ms":ad_ms,"dot_ms":t0.elapsed().as_secs_f64()*1e3}))
    }
    /// Diagnostic (AD_DOT_PAIRS=<function>, physics-on window integration): the launch dot split per pair of
    /// shadow allocations. lhs[i][j] = y_j . (M x_i - x_i)_j with the direction only on allocation i, rhs[i][j] =
    /// x_i . (M^T y_j - y_j)_i with the seed only on allocation j; a transposed pair has lhs = rhs up to rounding.
    /// Rows whose difference exceeds 1e-9 of the pair scale name where a TL / AD disagreement sits.
    unsafe fn pairs(&self, kd: &KernelDeriv, arena: &mut Arena, b: &mut Bound, l: &LaunchInfo, seed: u64,
                    x0: &[u64], y: &[u64]) -> Result<Value, String> {
        let (cuda, sup) = (self.cuda, self.support);
        let ids: Vec<u64> = b.shadows.keys().copied().collect();
        let sh: Vec<_> = b.shadows.values().copied().collect();
        let n = sh.len();
        let mut lhs = vec![vec![0.0f64; n]; n];
        let mut rhs = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            self.clear_shadows(b)?;
            for c in b.tl_cells.values() { dev::zero(cuda, c.0, 8)?; }
            sup.fill(cuda, x0[i], sh[i].bytes, sh[i].width, mix(seed ^ (2 * i as u64 + 1)))?;
            dev::copy(cuda, sh[i].address, x0[i], sh[i].bytes)?;
            self.tangent(kd, arena, b, l)?;
            for j in 0..n {
                sup.fill(cuda, y[j], sh[j].bytes, sh[j].width, mix(seed ^ (2 * j as u64 + 2)))?;
                sup.dotdiff_into(cuda, j, y[j], sh[j].address, if j == i { x0[i] } else { 0 }, sh[j].bytes, sh[j].width)?;
            }
            let (sums, _) = sup.collect(cuda, n, 0)?;
            for j in 0..n { lhs[i][j] = sums[j]; }
        }
        for j in 0..n {
            self.clear_shadows(b)?;
            sup.fill(cuda, y[j], sh[j].bytes, sh[j].width, mix(seed ^ (2 * j as u64 + 2)))?;
            dev::copy(cuda, sh[j].address, y[j], sh[j].bytes)?;
            self.reverse(kd, arena, b, l)?;
            for i in 0..n {
                sup.fill(cuda, x0[i], sh[i].bytes, sh[i].width, mix(seed ^ (2 * i as u64 + 1)))?;
                sup.dotdiff_into(cuda, i, x0[i], sh[i].address, if i == j { y[j] } else { 0 }, sh[i].bytes, sh[i].width)?;
            }
            let (sums, _) = sup.collect(cuda, n, 0)?;
            for i in 0..n { rhs[i][j] = sums[i]; }
        }
        let mut rows = Vec::new();
        for i in 0..n { for j in 0..n {
            let (a, c) = (lhs[i][j], rhs[i][j]);
            if a == 0.0 && c == 0.0 { continue; }
            rows.push(json!({"x_alloc":ids[i],"y_alloc":ids[j],"lhs":a,"rhs":c,"diff":a - c,
                             "rel":(a - c).abs() / f64::max(a.abs(), c.abs())}));
        } }
        Ok(json!(rows))
    }
}
