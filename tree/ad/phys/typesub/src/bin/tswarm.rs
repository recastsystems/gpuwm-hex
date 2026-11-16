// SPDX-License-Identifier: Apache-2.0
//! tswarm: P2 proof of the generated TEMPO warm_rain tangent and adjoint on captured hrrrv5 calls.
//!   identity  : generated primal (tangent and tape) vs the original captured function, every word
//!   hand      : generated TL/AD vs the 11-02 hand pair (warm-rain.ad.cu) on the same directions
//!   dot       : <TL dx, by> vs <dx, AD by>, 6 seeds, per output field, zero directions
//!   repeat    : two reverse runs bitwise equal
//!   cost      : forward, tangent, adjoint times; tape nodes and bytes per call; tile memory
#[path = "../gpu.rs"]
mod gpu;
use gpu::{Arg, Cuda, Nvrtc};
use std::fs;
use std::time::Instant;

const CALL: usize = 96;
const OUT: usize = 48;
const BARD: usize = 112;
const OUTS: [&str; 6] = ["pnr_rcr", "prr_wau", "pnr_wau", "pnc_wau", "prr_rcw", "pnc_rcw"];
const INS: [&str; 14] = ["rhof", "rc", "nc", "mvd_c", "rr", "nr", "mvd_r", "ilamc", "pnr_rcr", "prr_wau", "pnr_wau", "pnc_wau", "prr_rcw", "pnc_rcw"];
// Offsets of the 14 active inputs in the call record (WrIn) and the shadow (WrGrad); f = float.
const IN_OFF: [(usize, bool); 14] = [(0, true), (4, true), (8, true), (12, true), (16, true), (20, true), (24, true),
    (32, false), (40, false), (48, false), (56, false), (64, false), (72, false), (80, false)];
const GR_OFF: [(usize, bool); 14] = [(0, true), (4, true), (8, true), (12, true), (16, true), (20, true), (24, false),
    (32, false), (40, false), (48, false), (56, false), (64, false), (72, false), (80, false)];

fn rf32(b: &[u8], o: usize) -> f32 { f32::from_le_bytes(b[o..o + 4].try_into().unwrap()) }
fn rf64(b: &[u8], o: usize) -> f64 { f64::from_le_bytes(b[o..o + 8].try_into().unwrap()) }
fn wf32(b: &mut [u8], o: usize, v: f32) { b[o..o + 4].copy_from_slice(&v.to_le_bytes()); }
fn wf64(b: &mut [u8], o: usize, v: f64) { b[o..o + 8].copy_from_slice(&v.to_le_bytes()); }
fn in_val(c: &[u8], k: usize) -> f64 { let (o, f) = IN_OFF[k]; if f { rf32(c, o) as f64 } else { rf64(c, o) } }

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0 }
}

fn arg(a: &[String], k: &str) -> Option<String> { a.iter().position(|x| x == k).and_then(|i| a.get(i + 1).cloned()) }

fn rel(a: f64, b: f64) -> f64 { let d = (a - b).abs(); let s = a.abs().max(b.abs()); if d == 0.0 { 0.0 } else { d / s } }

fn main() {
    if let Err(e) = run() { eprintln!("tswarm: {e}"); std::process::exit(1); }
}

fn run() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let need = |k: &str| arg(&a, k).ok_or(format!("missing {k}"));
    let calls_path = need("--calls")?;
    let params_path = need("--params")?;
    let blob_list = need("--blob-list")?;
    let gen_dir = need("--gen-dir")?;
    let hand_src = need("--hand-src")?;
    let nvrtc_path = need("--nvrtc")?;
    let out_path = need("--out")?;
    let replicate: usize = arg(&a, "--replicate").map(|s| s.parse().unwrap()).unwrap_or(16);
    let tile: usize = arg(&a, "--tile").map(|s| s.parse().unwrap()).unwrap_or(4096);
    let label = format!("{}{}", arg(&a, "--label").unwrap_or_default(), arg(&a, "--define").map(|d| format!(" -D{d}")).unwrap_or_default());

    let t0 = Instant::now();
    let cuda = Cuda::new()?;
    cuda.stack_limit(16384)?;
    let nv = Nvrtc::new(&nvrtc_path)?;
    let arch = format!("-arch=sm_{}{}", cuda.arch.0, cuda.arch.1);
    // Options of the captured image (compile-record.json of the hrrrv5 TEMPO capture), arch of this card.
    let base_opts: Vec<String> = ["--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true", &arch,
        "--device-as-default-execution-space"].iter().map(|s| s.to_string()).collect();

    let calls = fs::read(&calls_path).map_err(|e| format!("{calls_path}: {e}"))?;
    if calls.len() % CALL != 0 { return Err("call file size".into()); }
    let n = calls.len() / CALL;
    let params = fs::read(&params_path).map_err(|e| format!("{params_path}: {e}"))?;
    let inv = rf32(&params, 0);

    // Builds.
    let tb = Instant::now();
    let proof_src = fs::read_to_string(format!("{gen_dir}/warm_rain_proof.cu")).map_err(|e| e.to_string())?;
    let mut gopts = base_opts.clone();
    gopts.push(format!("--include-path={gen_dir}"));
    if let Some(d) = arg(&a, "--define") { gopts.push(format!("-D{d}")); }
    let (gen_cubin, glog) = nv.build(&proof_src, "warm_rain_proof.cu", &gopts)?;
    let gen_build_s = tb.elapsed().as_secs_f64();
    let tb = Instant::now();
    let mut hsrc = fs::read_to_string(&hand_src).map_err(|e| format!("{hand_src}: {e}"))?;
    hsrc.push_str("\nextern \"C\" __global__ void hand_sizes(unsigned long long* o){o[0]=sizeof(WrActualMathTape);o[1]=sizeof(WarmRainFailure);o[2]=sizeof(WarmRainWarmParams);o[3]=sizeof(WarmRainWarmInput);o[4]=sizeof(WarmRainWarmGrad);o[5]=sizeof(WarmRainWarmPartial);o[6]=sizeof(WarmRainWarmOutput);}\n");
    let hopts: Vec<String> = base_opts.iter().filter(|o| !o.starts_with("--device-as-default") && !o.starts_with("--std")).cloned().collect();
    let (hand_cubin, _hlog) = nv.build(&hsrc, "warm-rain.ad.cu", &hopts)?;
    let hand_build_s = tb.elapsed().as_secs_f64();
    if !glog.is_empty() { eprintln!("generated build log:\n{}", &glog[..glog.len().min(4000)]); }
    let gm = cuda.load(&gen_cubin)?;
    let hm = cuda.load(&hand_cubin)?;
    let f = |m, s: &str| cuda.func(m, s);
    let (k_orig, k_tl, k_ad, k_arena, k_sizes) = (f(gm, "wr_orig")?, f(gm, "wr_tl")?, f(gm, "wr_ad")?, f(gm, "tsad_set_arena")?, f(gm, "wr_sizes")?);
    let (h_aug, h_tl, h_ad, h_sizes) = (f(hm, "warm_rain__aug")?, f(hm, "warm_rain__tl")?, f(hm, "warm_rain__ad")?, f(hm, "hand_sizes")?);

    let sz = cuda.alloc(64)?;
    cuda.run(k_sizes, 1, 1, &mut [Arg::P(sz)])?;
    let s: Vec<u64> = cuda.get(sz, 5)?;
    let (t_orig, t_tl, t_tp, node_b) = (s[0] as usize, s[1] as usize, s[2] as usize, s[3] as usize);
    cuda.run(h_sizes, 1, 1, &mut [Arg::P(sz)])?;
    let hs: Vec<u64> = cuda.get(sz, 7)?;
    let (h_tape_b, h_fail_b, h_par_b, h_partial_b) = (hs[0] as usize, hs[1] as usize, hs[2] as usize, hs[5] as usize);
    if hs[2] as usize != params.len() || hs[3] as usize != CALL || hs[4] as usize != CALL || hs[6] as usize != OUT {
        return Err(format!("hand layout mismatch {hs:?}"));
    }

    // Captured blob (TempoParams.L of the hrrrv5 TEMPO launch), chunk by chunk.
    let list = fs::read_to_string(&blob_list).map_err(|e| e.to_string())?;
    let chunks: Vec<&str> = list.lines().filter(|l| !l.trim().is_empty()).collect();
    let mut total = 0usize;
    let mut sizes = Vec::new();
    for c in &chunks { let m = fs::metadata(c).map_err(|e| format!("{c}: {e}"))?.len() as usize; sizes.push(m); total += m; }
    let blob = cuda.alloc(total)?;
    let mut off = 0;
    for (c, m) in chunks.iter().zip(&sizes) { let d = fs::read(c).map_err(|e| e.to_string())?; cuda.put(blob + off as u64, &d)?; off += m; }

    let nr = n * replicate;
    let mut rep = Vec::with_capacity(nr * CALL);
    for _ in 0..replicate { rep.extend_from_slice(&calls); }
    let x = cuda.alloc(nr * CALL)?;
    cuda.put(x, &rep)?;
    let p = cuda.alloc(params.len())?;
    cuda.put(p, &params)?;
    let dp = cuda.alloc(params.len())?;
    let tb_orig = cuda.alloc(nr * t_orig)?;
    let tb_tl = cuda.alloc(nr * t_tl)?;
    let y_orig = cuda.alloc(nr * OUT)?;
    let y_tl = cuda.alloc(nr * OUT)?;
    let dy = cuda.alloc(nr * OUT)?;
    let y_tp = cuda.alloc(nr * OUT)?;
    let by = cuda.alloc(nr * OUT)?;
    let bx = cuda.alloc(nr * BARD)?;
    let dx = cuda.alloc(nr * CALL)?;
    let blocks = |m: usize| ((m + 127) / 128) as u32;

    // --- original forward
    let orig = |m: usize| -> Result<(), String> {
        cuda.run(k_orig, blocks(m), 128, &mut [Arg::I(m as i32), Arg::P(x), Arg::P(blob), Arg::F(inv), Arg::P(tb_orig), Arg::P(y_orig)])
    };
    let tl = |m: usize| -> Result<(), String> {
        cuda.run(k_tl, blocks(m), 128, &mut [Arg::I(m as i32), Arg::P(x), Arg::P(dx), Arg::P(blob), Arg::F(inv), Arg::P(tb_tl), Arg::P(y_tl), Arg::P(dy)])
    };
    // --- tape sizing (count-only), then tiles
    let used = cuda.alloc(tile.max(nr) * 4)?;
    let ovf = cuda.alloc(tile.max(nr) * 4)?;
    let tb_tp = cuda.alloc(tile.max(n) * t_tp)?;
    cuda.run(k_arena, 1, 1, &mut [Arg::P(0), Arg::P(0), Arg::P(used), Arg::P(ovf), Arg::I(i32::MAX)])?;
    // count pass over the first tile-sized chunks of the base calls only
    let mut cap = 0i32;
    let mut first = 0;
    let mut counts: Vec<i32> = Vec::new();
    while first < n {
        let m = (n - first).min(tile);
        cuda.run(k_ad, blocks(m), 128, &mut [Arg::I(m as i32), Arg::I(first as i32), Arg::I(1), Arg::P(x), Arg::P(by), Arg::P(blob), Arg::F(inv), Arg::P(tb_tp), Arg::P(y_tp), Arg::P(bx)])?;
        let u: Vec<i32> = cuda.get(used, m)?;
        cap = cap.max(*u.iter().max().unwrap());
        counts.extend(u);
        first += m;
    }
    let cap = cap.max(1) as usize;
    let nodes = cuda.alloc(tile * cap * node_b)?;
    let bars = cuda.alloc(tile * cap * 8)?;
    cuda.run(k_arena, 1, 1, &mut [Arg::P(nodes), Arg::P(bars), Arg::P(used), Arg::P(ovf), Arg::I(cap as i32)])?;
    let ad = |m: usize| -> Result<(), String> {
        let mut first = 0;
        while first < m {
            let k = (m - first).min(tile);
            cuda.run(k_ad, blocks(k), 128, &mut [Arg::I(k as i32), Arg::I(first as i32), Arg::I(0), Arg::P(x), Arg::P(by), Arg::P(blob), Arg::F(inv), Arg::P(tb_tp), Arg::P(y_tp), Arg::P(bx)])?;
            first += k;
        }
        Ok(())
    };
    let check_ovf = |m: usize| -> Result<(), String> {
        let o: Vec<i32> = cuda.get(ovf, m.min(tile))?;
        if o.iter().any(|&v| v != 0) { Err("tape overflow".into()) } else { Ok(()) }
    };

    // --- scales for directions
    let mut rms_in = [0f64; 14];
    for c in calls.chunks(CALL) { for k in 0..14 { let v = in_val(c, k); rms_in[k] += v * v; } }
    for v in rms_in.iter_mut() { *v = (*v / n as f64).sqrt(); }
    let make_dx = |seed: u64, zero: bool| -> Vec<u8> {
        let mut r = Rng(seed);
        let mut g = vec![0u8; n * CALL];
        for i in 0..n {
            let c = &calls[i * CALL..(i + 1) * CALL];
            let gi = &mut g[i * CALL..(i + 1) * CALL];
            for k in 0..14 {
                let sc = { let v = in_val(c, k).abs(); if v > 0.0 { v } else if rms_in[k] > 0.0 { rms_in[k] } else { 1.0 } };
                let v = if zero { 0.0 } else { r.next() * sc };
                let (o, fl) = GR_OFF[k];
                if fl { wf32(gi, o, v as f32) } else { wf64(gi, o, v) }
            }
            gi[88..96].copy_from_slice(&c[88..96]);
        }
        g
    };
    let dx_val = |g: &[u8], i: usize, k: usize| -> f64 { let (o, fl) = GR_OFF[k]; let c = &g[i * CALL..]; if fl { rf32(c, o) as f64 } else { rf64(c, o) } };

    // ===== identity =====
    orig(n)?;
    let yo: Vec<f64> = cuda.get(y_orig, n * 6)?;
    cuda.put(dx, &make_dx(1, true))?;
    tl(n)?;
    let yt: Vec<f64> = cuda.get(y_tl, n * 6)?;
    let yp: Vec<f64> = cuda.get(y_tp, n * 6)?; // from the count pass (same primal path as the taped run)
    let diff = |a: &[f64], b: &[f64]| a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    let id_tl = diff(&yo, &yt);
    let id_tp_count = diff(&yo, &yp);
    // hand augmented forward on the same calls (its own native replica)
    let saved = cuda.alloc(n * h_tape_b)?;
    let fail = cuda.alloc(h_fail_b.max(64))?;
    let records = cuda.alloc(1 << 16)?;
    let yh = cuda.alloc(n * OUT)?;
    cuda.run(h_aug, blocks(n), 128, &mut [Arg::I(n as i32), Arg::P(x), Arg::P(p), Arg::P(yh), Arg::P(saved), Arg::P(fail), Arg::P(records)])?;
    let yhv: Vec<f64> = cuda.get(yh, n * 6)?;
    let hf: Vec<u8> = cuda.get_bytes(fail, h_fail_b)?;
    let id_hand = diff(&yo, &yhv);
    let hand_fail = hf.iter().any(|&b| b != 0);
    let mut out_rms = [0f64; 6];
    for i in 0..n { for j in 0..6 { out_rms[j] += yo[i * 6 + j].powi(2); } }
    for v in out_rms.iter_mut() { *v = (*v / n as f64).sqrt(); }
    let active_calls = (0..n).filter(|&i| (0..6).any(|j| yo[i * 6 + j] != in_val(&calls[i * CALL..], 8 + j))).count();

    // ===== generated TL/AD vs hand pair, seeds 1..6 =====
    let make_by = |seed: u64, only: Option<usize>| -> Vec<f64> {
        let mut r = Rng(seed ^ 0x9e3779b97f4a7c15);
        let mut v = vec![0f64; n * 6];
        for i in 0..n { for j in 0..6 {
            let s = if out_rms[j] > 0.0 { 1.0 / out_rms[j] } else { 1.0 };
            let x = r.next() * s;
            if only.map(|o| o == j).unwrap_or(true) { v[i * 6 + j] = x; }
        } }
        v
    };
    let mut tl_hand_worst = [0f64; 6];
    let mut tl_hand_norm = [0f64; 6];
    let mut ad_hand_worst = [0f64; 14];
    let mut ad_hand_norm = [0f64; 14];
    let mut dot_worst = 0f64;
    let mut dot_rows = Vec::new();
    let hand_partial = cuda.alloc(n * h_partial_b)?;
    let hbx = cuda.alloc(n * CALL)?;
    let hdy = cuda.alloc(n * OUT)?;
    let hby = cuda.alloc(n * OUT)?;
    let mut flips_note = 0usize;
    for seed in 1..=6u64 {
        let g = make_dx(seed, false);
        cuda.put(dx, &g)?;
        tl(n)?;
        let d: Vec<f64> = cuda.get(dy, n * 6)?;
        let b = make_by(seed, None);
        cuda.put(by, &b)?;
        ad(n)?;
        check_ovf(n)?;
        let r: Vec<f64> = cuda.get(bx, n * 14)?;
        // dot, f64
        let lhs: f64 = (0..n * 6).map(|q| d[q] * b[q]).sum();
        let mut rhs = 0f64;
        for i in 0..n { for k in 0..14 { rhs += dx_val(&g, i, k) * r[i * 14 + k]; } }
        let e = rel(lhs, rhs);
        dot_worst = dot_worst.max(e);
        dot_rows.push(format!("{{\"seed\":{seed},\"tl_dot\":{lhs:e},\"ad_dot\":{rhs:e},\"rel\":{e:e}}}"));
        // hand TL
        cuda.zero(dp, params.len())?;
        cuda.zero(fail, h_fail_b.max(64))?;
        cuda.run(h_tl, blocks(n), 128, &mut [Arg::I(n as i32), Arg::P(x), Arg::P(dx), Arg::P(p), Arg::P(dp), Arg::P(hdy), Arg::P(saved), Arg::P(fail), Arg::P(records)])?;
        let hd: Vec<f64> = cuda.get(hdy, n * 6)?;
        if cuda.get_bytes(fail, h_fail_b)?.iter().any(|&q| q != 0) { flips_note += 1; }
        for j in 0..6 {
            let m = (0..n).map(|i| hd[i * 6 + j].abs()).fold(0.0, f64::max);
            let mut worst = 0f64; let mut dmax = 0f64;
            for i in 0..n {
                let (u, h) = (d[i * 6 + j], hd[i * 6 + j]);
                dmax = dmax.max((u - h).abs());
                if h.abs() > 1e-12 * m || u.abs() > 1e-12 * m { worst = worst.max(rel(u, h)); }
            }
            tl_hand_worst[j] = tl_hand_worst[j].max(worst);
            tl_hand_norm[j] = tl_hand_norm[j].max(if m > 0.0 { dmax / m } else { dmax });
        }
        // hand AD
        cuda.put(hby, &b)?;
        cuda.zero(hbx, n * CALL)?;
        cuda.run(h_ad, blocks(n), 128, &mut [Arg::I(n as i32), Arg::P(x), Arg::P(p), Arg::P(hby), Arg::P(hbx), Arg::P(hand_partial), Arg::P(saved), Arg::P(fail), Arg::P(records)])?;
        let hb = cuda.get_bytes(hbx, n * CALL)?;
        for k in 0..14 {
            let hv: Vec<f64> = (0..n).map(|i| dx_val(&hb, i, k)).collect();
            let m = hv.iter().fold(0.0f64, |q, v| q.max(v.abs()));
            let mut worst = 0f64; let mut dmax = 0f64;
            for i in 0..n {
                let (u, h) = (r[i * 14 + k], hv[i]);
                // the hand stores the float-input cotangents in float: compare at that precision
                let u = if GR_OFF[k].1 { (u as f32) as f64 } else { u };
                dmax = dmax.max((u - h).abs());
                if h.abs() > 1e-12 * m || u.abs() > 1e-12 * m { worst = worst.max(rel(u, h)); }
            }
            ad_hand_worst[k] = ad_hand_worst[k].max(worst);
            ad_hand_norm[k] = ad_hand_norm[k].max(if m > 0.0 { dmax / m } else { dmax });
        }
    }
    // per-output-field dot (seed 1) and zero directions
    let g1 = make_dx(1, false);
    cuda.put(dx, &g1)?;
    tl(n)?;
    let d1: Vec<f64> = cuda.get(dy, n * 6)?;
    let mut field_dot = Vec::new();
    for j in 0..6 {
        let b = make_by(1, Some(j));
        cuda.put(by, &b)?;
        ad(n)?;
        let r: Vec<f64> = cuda.get(bx, n * 14)?;
        let lhs: f64 = (0..n).map(|i| d1[i * 6 + j] * b[i * 6 + j]).sum();
        let mut rhs = 0f64;
        for i in 0..n { for k in 0..14 { rhs += dx_val(&g1, i, k) * r[i * 14 + k]; } }
        let e = rel(lhs, rhs);
        dot_worst = dot_worst.max(e);
        field_dot.push(format!("{{\"field\":\"{}\",\"tl_dot\":{lhs:e},\"ad_dot\":{rhs:e},\"rel\":{e:e}}}", OUTS[j]));
    }
    cuda.put(dx, &make_dx(1, true))?;
    tl(n)?;
    let zero_tl = cuda.get::<f64>(dy, n * 6)?.iter().all(|&v| v.to_bits() == 0);
    cuda.put(by, &vec![0f64; n * 6])?;
    ad(n)?;
    let zero_ad = cuda.get::<f64>(bx, n * 14)?.iter().all(|&v| v.to_bits() == 0);
    // identity of the taped (non-count) primal and repeat determinism
    let b = make_by(3, None);
    cuda.put(by, &b)?;
    ad(n)?;
    let r1 = cuda.get_bytes(bx, n * BARD)?;
    let yp2: Vec<f64> = cuda.get(y_tp, n * 6)?;
    let id_tp = diff(&yo, &yp2);
    ad(n)?;
    let r2 = cuda.get_bytes(bx, n * BARD)?;
    let repeat_diff = r1.iter().zip(&r2).filter(|(a, b)| a != b).count();

    // ===== cost, replicated batch =====
    cuda.put(dx, &{ let g = make_dx(2, false); let mut v = Vec::with_capacity(nr * CALL); for _ in 0..replicate { v.extend_from_slice(&g); } v })?;
    cuda.put(by, &{ let b = make_by(2, None); let mut v = Vec::with_capacity(nr * 6); for _ in 0..replicate { v.extend_from_slice(&b); } v })?;
    let time = |f: &dyn Fn() -> Result<(), String>| -> Result<f64, String> {
        f()?; cuda.synchronize()?;
        let mut v = Vec::new();
        for _ in 0..7 { let t = Instant::now(); f()?; cuda.synchronize()?; v.push(t.elapsed().as_secs_f64() * 1e3); }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(v[3])
    };
    let ms_orig = time(&|| orig(nr))?;
    let ms_tl = time(&|| tl(nr))?;
    let ms_ad = time(&|| ad(nr))?;
    check_ovf(nr)?;
    let (free, totalm) = cuda.free_mem();

    let nodes_max = cap;
    let nodes_mean = counts.iter().map(|&c| c as f64).sum::<f64>() / counts.len() as f64;
    let per_node = node_b + 8;
    let fmt6 = |v: &[f64]| v.iter().map(|x| format!("{x:e}")).collect::<Vec<_>>().join(",");
    let pass = id_tl == 0 && id_tp == 0 && id_tp_count == 0 && repeat_diff == 0 && zero_tl && zero_ad
        && dot_worst <= 1e-10 && tl_hand_worst.iter().chain(ad_hand_worst.iter()).all(|&e| e <= 1e-6);
    let receipt = format!(r#"{{
 "schema": "ad.p2.warm_rain/v1",
 "label": "{label}",
 "pass": {pass},
 "device_arch": "sm_{}{}",
 "calls": {n}, "calls_file": "{calls_path}", "calls_with_any_tendency_change": {active_calls},
 "compile_options": [{}],
 "build_seconds": {{"generated_tu": {gen_build_s:.1}, "hand_pair": {hand_build_s:.1}}},
 "identity": {{"words": {}, "generated_tangent_primal_differ": {id_tl}, "generated_tape_primal_differ": {id_tp}, "generated_count_pass_primal_differ": {id_tp_count}, "hand_aug_vs_original_differ": {id_hand}, "hand_aug_failure": {hand_fail}}},
 "vs_hand_pair": {{"tl_worst_rel_by_output": {{{}}}, "tl_norm_rel_by_output": [{}], "ad_worst_rel_by_input": {{{}}}, "ad_norm_rel_by_input": [{}], "hand_tl_refusals": {flips_note}}},
 "dot": {{"worst_rel": {dot_worst:e}, "seeds": [{}], "per_output_field_seed1": [{}], "zero_direction_tl_exact_zero": {zero_tl}, "zero_cotangent_ad_exact_zero": {zero_ad}}},
 "repeat_reverse_bytes_differ": {repeat_diff},
 "tape": {{"nodes_per_call_max": {nodes_max}, "nodes_per_call_mean": {nodes_mean:.1}, "bytes_per_node": {per_node}, "bytes_per_call_max": {}, "tile_calls": {tile}, "tile_arena_bytes": {}}},
 "cost_ms_median_of_7": {{"batch_calls": {nr}, "forward_original": {ms_orig:.3}, "tangent": {ms_tl:.3}, "adjoint_record_and_reverse": {ms_ad:.3}, "tangent_over_forward": {:.2}, "adjoint_over_forward": {:.2}}},
 "gpu_memory_free_after_mib": {}, "gpu_memory_total_mib": {},
 "wall_seconds": {:.1}
}}
"#, cuda.arch.0, cuda.arch.1,
        base_opts.iter().map(|o| format!("\"{o}\"")).collect::<Vec<_>>().join(","),
        n * 6,
        OUTS.iter().zip(tl_hand_worst.iter()).map(|(k, v)| format!("\"{k}\":{v:e}")).collect::<Vec<_>>().join(","), fmt6(&tl_hand_norm),
        INS.iter().zip(ad_hand_worst.iter()).map(|(k, v)| format!("\"{k}\":{v:e}")).collect::<Vec<_>>().join(","), fmt6(&ad_hand_norm),
        dot_rows.join(","), field_dot.join(","),
        nodes_max * per_node, tile * cap * per_node,
        ms_tl / ms_orig, ms_ad / ms_orig, free >> 20, totalm >> 20, t0.elapsed().as_secs_f64());
    fs::write(&out_path, &receipt).map_err(|e| e.to_string())?;
    print!("{receipt}");
    Ok(())
}
