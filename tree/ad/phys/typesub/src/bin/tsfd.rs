// SPDX-License-Identifier: Apache-2.0
//! tsfd: P2 independent check. Finite-difference referee for the generated TEMPO warm_rain tangent.
//!   - builds `tshp` (the generated tangent namespace with TF = TD = double) from the generated file
//!   - full 6x14 Jacobian per call from the generated tangent, default partials and TSAD_HAND_PARTIALS
//!   - centred differences of tshp at h and h/4; an element is refereed only where the two agree
//!     (no branch, table cell or kink straddled) and the promoted primal matches the forward's
//!   - reports, per input/output pair, how far each tangent is from the referee, and which of the
//!     two partial modes is closer where they disagree.
#[path = "../gpu.rs"]
mod gpu;
use gpu::{Arg, Cuda, Nvrtc};
use std::fs;
use std::time::Instant;

const CALL: usize = 96;
const OUTS: [&str; 6] = ["pnr_rcr", "prr_wau", "pnr_wau", "pnc_wau", "prr_rcw", "pnc_rcw"];
const INS: [&str; 14] = ["rhof", "rc", "nc", "mvd_c", "rr", "nr", "mvd_r", "ilamc", "pnr_rcr", "prr_wau", "pnr_wau", "pnc_wau", "prr_rcw", "pnc_rcw"];
const IN_OFF: [(usize, bool); 14] = [(0, true), (4, true), (8, true), (12, true), (16, true), (20, true), (24, true),
    (32, false), (40, false), (48, false), (56, false), (64, false), (72, false), (80, false)];

fn arg(a: &[String], k: &str) -> Option<String> { a.iter().position(|x| x == k).and_then(|i| a.get(i + 1).cloned()) }
fn rf32(b: &[u8], o: usize) -> f32 { f32::from_le_bytes(b[o..o + 4].try_into().unwrap()) }
fn rf64(b: &[u8], o: usize) -> f64 { f64::from_le_bytes(b[o..o + 8].try_into().unwrap()) }
fn in_val(c: &[u8], k: usize) -> f64 { let (o, f) = IN_OFF[k]; if f { rf32(c, o) as f64 } else { rf64(c, o) } }

/// tshp: copy of `namespace tsad_tl { ... }` with plain double TF/TD, and the float `tnint`
/// overload dropped (both overloads collapse to tnint(double)).
fn make_hp(gen: &str) -> Result<String, String> {
    let a = gen.find("namespace tsad_tl {").ok_or("no tsad_tl")?;
    let end_tag = "}  // namespace tsad_tl";
    let b = gen[a..].find(end_tag).ok_or("no tsad_tl end")? + a + end_tag.len();
    let mut out = String::new();
    for line in gen[a..b].lines() {
        let l = line.replace("namespace tsad_tl {", "namespace tshp {").replace("}  // namespace tsad_tl", "}  // namespace tshp")
            .replace("typedef tsad::VF<tsad::Tan> TF;", "typedef double TF;").replace("typedef tsad::VD<tsad::Tan> TD;", "typedef double TD;");
        if l.contains("int tnint(TF x)") { continue; }
        out.push_str(&l);
        out.push('\n');
    }
    if out.contains("tsad::") { return Err("tshp still names tsad:: types".into()); }
    Ok(format!("{gen}\n// ---- tshp: double-promoted referee copy (P2 independent check) ----\n{out}"))
}

fn main() { if let Err(e) = run() { eprintln!("tsfd: {e}"); std::process::exit(1); } }

fn run() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let need = |k: &str| arg(&a, k).ok_or(format!("missing {k}"));
    let calls_path = need("--calls")?;
    let params_path = need("--params")?;
    let blob_list = need("--blob-list")?;
    let gen_dir = need("--gen-dir")?;
    let nvrtc_path = need("--nvrtc")?;
    let out_path = need("--out")?;
    let label = arg(&a, "--label").unwrap_or_default();
    let rel_h: f64 = arg(&a, "--rel").map(|s| s.parse().unwrap()).unwrap_or(1e-5);
    let agree: f64 = arg(&a, "--agree").map(|s| s.parse().unwrap()).unwrap_or(1e-7);
    let t0 = Instant::now();
    let cuda = Cuda::new()?;
    cuda.stack_limit(16384)?;
    let nv = Nvrtc::new(&nvrtc_path)?;
    let arch = format!("-arch=sm_{}{}", cuda.arch.0, cuda.arch.1);
    let base: Vec<String> = ["--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true", &arch,
        "--device-as-default-execution-space"].iter().map(|s| s.to_string()).collect();
    let gen = fs::read_to_string(format!("{gen_dir}/warm_rain.gen.cu")).map_err(|e| e.to_string())?;
    fs::write(format!("{gen_dir}/warm_rain_hp.gen.cu"), make_hp(&gen)?).map_err(|e| e.to_string())?;
    let src = fs::read_to_string(format!("{gen_dir}/warm_rain_fd.cu")).map_err(|e| e.to_string())?;
    let mut o1 = base.clone();
    o1.push(format!("--include-path={gen_dir}"));
    let mut o2 = o1.clone();
    o2.push("-DTSAD_HAND_PARTIALS".into());
    let (c1, l1) = nv.build(&src, "warm_rain_fd.cu", &o1)?;
    let (c2, _) = nv.build(&src, "warm_rain_fd.cu", &o2)?;
    if !l1.is_empty() { eprintln!("build log:\n{}", &l1[..l1.len().min(3000)]); }
    let (m1, m2) = (cuda.load(&c1)?, cuda.load(&c2)?);

    let calls = fs::read(&calls_path).map_err(|e| format!("{calls_path}: {e}"))?;
    if calls.len() % CALL != 0 { return Err("call file size".into()); }
    let n = calls.len() / CALL;
    let params = fs::read(&params_path).map_err(|e| e.to_string())?;
    let inv = rf32(&params, 0);
    let list = fs::read_to_string(&blob_list).map_err(|e| e.to_string())?;
    let chunks: Vec<&str> = list.lines().filter(|l| !l.trim().is_empty()).collect();
    let mut sizes = Vec::new();
    let mut total = 0usize;
    for c in &chunks { let m = fs::metadata(c).map_err(|e| format!("{c}: {e}"))?.len() as usize; sizes.push(m); total += m; }
    let blob = cuda.alloc(total)?;
    let mut off = 0;
    for (c, m) in chunks.iter().zip(&sizes) { cuda.put(blob + off as u64, &fs::read(c).map_err(|e| e.to_string())?)?; off += m; }

    let x = cuda.alloc(n * CALL)?;
    cuda.put(x, &calls)?;
    let nj = n * 14 * 6;
    let (j1, j2, f1, f2) = (cuda.alloc(nj * 8)?, cuda.alloc(nj * 8)?, cuda.alloc(nj * 8)?, cuda.alloc(nj * 8)?);
    let (y0, yo) = (cuda.alloc(n * 48)?, cuda.alloc(n * 48)?);
    let sz = cuda.alloc(8)?;
    cuda.run(cuda.func(m1, "fd_sizes")?, 1, 1, &mut [Arg::P(sz)])?;
    let tsz = cuda.get::<u64>(sz, 1)?[0] as usize;
    let tb = cuda.alloc(n * tsz)?;
    let mut scale = [0f64; 14];
    for c in calls.chunks(CALL) { for k in 0..14 { scale[k] = scale[k].max(in_val(c, k).abs()); } }
    for v in scale.iter_mut() { if *v == 0.0 { *v = 1.0; } }
    let sc = cuda.alloc(14 * 8)?;
    cuda.put(sc, &scale)?;
    let relb = cuda.alloc(8)?; // Arg carries no f64, so the step goes through memory
    cuda.put(relb, &[rel_h])?;
    let g = |t: usize| ((t + 127) / 128) as u32;
    cuda.run(cuda.func(m1, "wr_orig")?, g(n), 128, &mut [Arg::I(n as i32), Arg::P(x), Arg::P(blob), Arg::F(inv), Arg::P(tb), Arg::P(yo)])?;
    cuda.run(cuda.func(m1, "wr_jtl")?, g(n * 14), 128, &mut [Arg::I(n as i32), Arg::P(x), Arg::P(blob), Arg::F(inv), Arg::P(j1)])?;
    cuda.run(cuda.func(m2, "wr_jtl")?, g(n * 14), 128, &mut [Arg::I(n as i32), Arg::P(x), Arg::P(blob), Arg::F(inv), Arg::P(j2)])?;
    cuda.run(cuda.func(m1, "wr_fdhp_m")?, g(n * 14), 128, &mut [Arg::I(n as i32), Arg::P(x), Arg::P(blob), Arg::F(inv), Arg::P(relb), Arg::P(sc), Arg::P(f1), Arg::P(f2), Arg::P(y0)])?;
    cuda.synchronize()?;
    let jd: Vec<f64> = cuda.get(j1, nj)?;
    let jh: Vec<f64> = cuda.get(j2, nj)?;
    let fa: Vec<f64> = cuda.get(f1, nj)?;
    let fb: Vec<f64> = cuda.get(f2, nj)?;
    let yh: Vec<f64> = cuda.get(y0, n * 6)?;
    let yf: Vec<f64> = cuda.get(yo, n * 6)?;
    let gpu_s = t0.elapsed().as_secs_f64();

    // Promoted primal vs the forward: a call whose promoted outputs differ by more than 1e-4 took a
    // different branch or table cell in double and is not refereed.
    let mut call_ok = vec![true; n];
    let mut prim_worst = 0f64;
    for i in 0..n { for j in 0..6 {
        let (p, q) = (yh[i * 6 + j], yf[i * 6 + j]);
        let r = if p == q { 0.0 } else { (p - q).abs() / p.abs().max(q.abs()) };
        if r > 1e-4 { call_ok[i] = false; } else { prim_worst = prim_worst.max(r); }
    } }
    let mut colmax = vec![0f64; 84];
    for i in 0..n { for c in 0..84 { let v = fb[i * 84 + c]; if v.is_finite() { colmax[c] = colmax[c].max(v.abs()); } } }
    let thr = [1e-6, 1e-4, 1e-3, 1e-2];
    let mut pair_rows = Vec::new();
    let (mut tot, mut refd, mut unstable, mut skip_call, mut nonfinite) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut ge_d = [0usize; 4];
    let mut ge_h = [0usize; 4];
    let (mut wd, mut wh) = (0f64, 0f64);
    let (mut disagree, mut def_closer, mut hand_closer, mut tie) = (0usize, 0usize, 0usize, 0usize);
    let mut zero_mismatch = 0usize;
    let mut worst_rows: Vec<(f64, usize, usize, usize, f64, f64, f64, f64)> = Vec::new();
    for k in 0..14 { for j in 0..6 {
        let c = k * 6 + j;
        let (mut pd, mut ph, mut cnt, mut gd6) = (0f64, 0f64, 0usize, 0usize);
        for i in 0..n {
            let q = i * 84 + c;
            let (d, h, r1, r2) = (jd[q], jh[q], fa[q], fb[q]);
            if d == 0.0 && h == 0.0 && r1 == 0.0 && r2 == 0.0 { continue; }
            tot += 1;
            if !call_ok[i] { skip_call += 1; continue; }
            if !(d.is_finite() && h.is_finite() && r1.is_finite() && r2.is_finite()) { nonfinite += 1; continue; }
            let floor = 1e-10 * colmax[c];
            if (r1 - r2).abs() > agree * r1.abs().max(r2.abs()).max(floor) { unstable += 1; continue; }
            let r = r2;
            refd += 1;
            cnt += 1;
            let den = r.abs().max(floor).max(1e-300);
            let (ed, eh) = ((d - r).abs() / den, (h - r).abs() / den);
            if r.abs() <= floor && (d.abs() > 1e-6 * colmax[c] || h.abs() > 1e-6 * colmax[c]) { zero_mismatch += 1; }
            for t in 0..4 { if ed > thr[t] { ge_d[t] += 1; } if eh > thr[t] { ge_h[t] += 1; } }
            if ed > 1e-6 { gd6 += 1; }
            pd = pd.max(ed);
            ph = ph.max(eh);
            if (d - h).abs() > 1e-6 * d.abs().max(h.abs()) {
                disagree += 1;
                if ed < eh { def_closer += 1 } else if eh < ed { hand_closer += 1 } else { tie += 1 }
            }
            if ed > 1e-4 || eh > 1e-4 { worst_rows.push((ed.max(eh), i, k, j, d, h, r, r1)); }
        }
        wd = wd.max(pd);
        wh = wh.max(ph);
        if cnt > 0 {
            pair_rows.push(format!("{{\"in\":\"{}\",\"out\":\"{}\",\"refereed\":{cnt},\"default_worst\":{pd:.3e},\"hand_worst\":{ph:.3e},\"default_over_1e-6\":{gd6}}}", INS[k], OUTS[j]));
        }
    } }
    worst_rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let wr: Vec<String> = worst_rows.iter().take(12).map(|w| format!(
        "{{\"call\":{},\"in\":\"{}\",\"out\":\"{}\",\"tl_default\":{:e},\"tl_hand\":{:e},\"fd_h4\":{:e},\"fd_h\":{:e}}}",
        w.1, INS[w.2], OUTS[w.3], w.4, w.5, w.6, w.7)).collect();
    let ok_calls = call_ok.iter().filter(|&&b| b).count();
    let receipt = format!(r#"{{
 "schema": "ad.p2.warm_rain.fd-check/v1",
 "label": "{label}",
 "calls": {n}, "calls_refereed": {ok_calls}, "calls_file": "{calls_path}",
 "referee": "centred differences of tshp (generated tangent statements, TF=TD=double, float table constants) at h and h/4, h = {rel_h:e}*|x_k|; element kept when the two agree to {agree:e}",
 "promoted_primal_vs_forward_worst_rel": {prim_worst:.3e},
 "elements": {{"nonzero": {tot}, "refereed": {refd}, "fd_unstable_kink_or_table_cell": {unstable}, "promoted_branch_differs": {skip_call}, "nonfinite": {nonfinite}}},
 "default_partials": {{"worst_rel": {wd:.3e}, "over_1e-6": {}, "over_1e-4": {}, "over_1e-3": {}, "over_1e-2": {}}},
 "hand_partials": {{"worst_rel": {wh:.3e}, "over_1e-6": {}, "over_1e-4": {}, "over_1e-3": {}, "over_1e-2": {}}},
 "where_modes_disagree_over_1e-6": {{"elements": {disagree}, "default_closer_to_fd": {def_closer}, "hand_closer_to_fd": {hand_closer}, "equal": {tie}}},
 "fd_zero_but_tangent_nonzero": {zero_mismatch},
 "worst_elements": [{}],
 "pairs": [{}],
 "gpu_seconds": {gpu_s:.2}, "wall_seconds": {:.2}
}}
"#, ge_d[0], ge_d[1], ge_d[2], ge_d[3], ge_h[0], ge_h[1], ge_h[2], ge_h[3], wr.join(","), pair_rows.join(","), t0.elapsed().as_secs_f64());
    fs::write(&out_path, &receipt).map_err(|e| e.to_string())?;
    print!("{receipt}");
    Ok(())
}
