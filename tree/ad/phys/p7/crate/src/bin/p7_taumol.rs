// SPDX-License-Identifier: Apache-2.0
//! P7 proof of the generated whole-launch rlw_taumol_batched tangent/adjoint on a
//! real captured hrrrv5 launch: capture replay, transform self-check, augmented
//! primal identity, zero seed, dot tests (float64 derivatives), two reverse runs,
//! TL vs centred finite differences of the unchanged captured image, and cost.
#[path = "../gpu.rs"] mod gpu;
#[path = "../capture.rs"] mod capture;
#[path = "../num.rs"] mod num;
use capture::*; use gpu::*; use serde_json::{json, Value};
use std::path::PathBuf; use std::time::Instant;

struct Case { ncol: i32, nl: i32, oneminus: f32, laytrop: u64, fs: u64, isv: u64, wx: u64, chi: u64,
              bandptrs: u64, taug: u64, fracs: u64, nfs: usize, nwx: usize, nout: usize,
              grid: [u32; 3], block: [u32; 3] }

fn fwd_args(c: &Case, fs: u64, wx: u64) -> Vec<Arg> {
    vec![Arg::I32(c.ncol), Arg::I32(c.nl), Arg::Ptr(c.laytrop), Arg::Ptr(fs), Arg::Ptr(c.isv), Arg::Ptr(wx),
         Arg::Ptr(c.chi), Arg::F32(c.oneminus), Arg::Ptr(c.bandptrs), Arg::Ptr(c.taug), Arg::Ptr(c.fracs)]
}

fn run() -> Result<Value, String> {
    let a: Vec<String> = std::env::args().collect();
    let (cap, tables, gen, out, nvrtc) = (PathBuf::from(&a[1]), PathBuf::from(&a[2]), PathBuf::from(&a[3]), PathBuf::from(&a[4]), a[5].clone());
    let t0 = Instant::now();
    let src = std::fs::read_to_string(&gen).map_err(|e| e.to_string())?;
    // L2 (ad/window-L2-bindings): P7_ARCH picks the card (default sm_120, the 5090). On another arch the captured
    // original image (an sm_120 cubin) cannot load, so the capture's own source is recompiled with these options,
    // and the capture replay below still grades that forward against the captured post state word for word.
    let arch = std::env::var("P7_ARCH").unwrap_or_else(|_| "sm_120".into());
    let arch_opt = format!("-arch={arch}");
    let opts = ["-std=c++17", "--ftz=false", "--fmad=false", "--prec-div=true", "--prec-sqrt=true", arch_opt.as_str()];
    let tc = Instant::now();
    let cubin = nvrtc_compile(&nvrtc, &src, "p7_taumol_ad.cu", &opts)?;
    let compile_s = tc.elapsed().as_secs_f64();
    std::fs::write(out.join("p7_taumol_ad.cubin"), &cubin).map_err(|e| e.to_string())?;
    if std::env::var_os("P7_COMPILE_ONLY").is_some() { return Ok(json!({"compiled": cubin.len(), "compile_s": compile_s})); }
    let l = Launch::load(&cap)?;
    let mut g = Gpu::new()?;
    let orig_image = if arch == "sm_120" { l.image.clone() } else {
        let fsrc = std::fs::read_to_string(l.dir.join(l.manifest["source"]["path"].as_str().ok_or("source path")?)).map_err(|e| e.to_string())?;
        nvrtc_compile(&nvrtc, &fsrc, "forward.cu", &opts)?
    };
    let orig = g.module(&orig_image)?; let genm = g.module(&cubin)?;
    let f_orig = orig.func("rlw_taumol_batched")?; let f_self = genm.func("p7_taumol_fwd")?;
    let f_tl = genm.func("rlw_taumol_batched__tl")?; let f_ad = genm.func("rlw_taumol_batched__ad")?;

    // device state from the capture
    let mut dev = Vec::new();
    for b in &l.pre { dev.push(g.put(b)?); }
    // band coefficient tables rebuilt by hex's own loader; captured pointer tables re-pointed at them
    let index: Value = serde_json::from_slice(&std::fs::read(tables.join("index.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let mut band_arrays = Vec::new();
    for band in 1..=16 {
        let mut ptrs: Vec<u64> = Vec::new();
        for row in index["bands"][band.to_string()].as_array().unwrap() {
            ptrs.push(g.put(&std::fs::read(tables.join(row["file"].as_str().unwrap())).map_err(|e| e.to_string())?)?);
        }
        band_arrays.push(g.put(&ptrs.iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>())?);
    }
    let bandptrs = g.put(&band_arrays.iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>())?;
    let chi_file = std::fs::read(tables.join("chi.bin")).map_err(|e| e.to_string())?;
    let chi_alloc = l.alloc_of("chi_mls")?;
    let chi_match = l.pre[chi_alloc][..chi_file.len()] == chi_file[..];
    let (grid, block) = l.geometry();
    let ia = |n: &str| -> Result<usize, String> { l.alloc_of(n) };
    let c = Case { ncol: l.i32("ncol")?, nl: l.i32("nl")?, oneminus: l.f32("oneminus")?, laytrop: dev[ia("laytrop_v")?],
        fs: dev[ia("fs")?], isv: dev[ia("isv")?], wx: dev[ia("wx")?], chi: dev[chi_alloc], bandptrs,
        taug: dev[ia("taug")?], fracs: dev[ia("fracs")?], nfs: 33 * (l.i32("ncol")? * l.i32("nl")?) as usize,
        nwx: 4 * (l.i32("ncol")? * l.i32("nl")?) as usize, nout: (l.i32("ncol")? * l.i32("nl")?) as usize * 140, grid, block };
    let (ta, fa) = (ia("taug")?, ia("fracs")?);
    let post_t = &l.post[ta][..c.nout * 4]; let post_f = &l.post[fa][..c.nout * 4];
    let fs0 = f32s(&l.pre[ia("fs")?][..c.nfs * 4]); let wx0 = f32s(&l.pre[ia("wx")?][..c.nwx * 4]);

    // 1. capture replay with the rebuilt tables, unchanged original image
    let check = |g: &Gpu| -> Result<(usize, usize), String> {
        g.sync()?;
        Ok((words_differ(&g.download(c.taug, c.nout * 4)?, post_t), words_differ(&g.download(c.fracs, c.nout * 4)?, post_f)))
    };
    g.clear(c.taug, c.nout * 4)?; g.clear(c.fracs, c.nout * 4)?;
    let tf = Instant::now(); f_orig.launch(c.grid, c.block, &fwd_args(&c, c.fs, c.wx))?; g.sync()?; let fwd_ms0 = tf.elapsed().as_secs_f64() * 1e3;
    let replay = check(&g)?;
    // 2. transform self-check
    g.clear(c.taug, c.nout * 4)?; g.clear(c.fracs, c.nout * 4)?;
    f_self.launch(c.grid, c.block, &fwd_args(&c, c.fs, c.wx))?;
    let selfcheck = check(&g)?;

    // derivative buffers (float64)
    let dfs = g.alloc(c.nfs * 8)?; let dwx = g.alloc(c.nwx * 8)?;
    let dt = g.alloc(c.nout * 8)?; let df = g.alloc(c.nout * 8)?;
    let bfs = g.alloc(c.nfs * 8)?; let bwx = g.alloc(c.nwx * 8)?;
    let bt = g.alloc(c.nout * 8)?; let bf = g.alloc(c.nout * 8)?; let fail = g.alloc(4)?;
    let tl = |g: &Gpu, xf: &[f64], xw: &[f64]| -> Result<(Vec<f64>, Vec<f64>, f64), String> {
        g.upload(dfs, &bytes_f64(xf))?; g.upload(dwx, &bytes_f64(xw))?; g.clear(dt, c.nout * 8)?; g.clear(df, c.nout * 8)?;
        let mut args = fwd_args(&c, c.fs, c.wx); args.extend([Arg::Ptr(dfs), Arg::Ptr(dwx), Arg::Ptr(dt), Arg::Ptr(df)]);
        g.sync()?; let t = Instant::now(); f_tl.launch(c.grid, c.block, &args)?; g.sync()?; let ms = t.elapsed().as_secs_f64() * 1e3;
        Ok((f64s(&g.download(dt, c.nout * 8)?), f64s(&g.download(df, c.nout * 8)?), ms))
    };
    let adg = [((c.ncol * c.nl) as u32).div_ceil(64), 1, 1];
    let ad = |g: &Gpu, wt: &[f64], wf: &[f64]| -> Result<(Vec<f64>, Vec<f64>, f64, bool, i32), String> {
        g.upload(bt, &bytes_f64(wt))?; g.upload(bf, &bytes_f64(wf))?; g.clear(bfs, c.nfs * 8)?; g.clear(bwx, c.nwx * 8)?; g.clear(fail, 4)?;
        let args = vec![Arg::I32(c.ncol), Arg::I32(c.nl), Arg::Ptr(c.laytrop), Arg::Ptr(c.fs), Arg::Ptr(c.isv), Arg::Ptr(c.wx),
            Arg::Ptr(c.chi), Arg::F32(c.oneminus), Arg::Ptr(c.bandptrs), Arg::Ptr(bfs), Arg::Ptr(bwx), Arg::Ptr(bt), Arg::Ptr(bf), Arg::Ptr(fail)];
        g.sync()?; let t = Instant::now(); f_ad.launch(adg, [64, 1, 1], &args)?; g.sync()?; let ms = t.elapsed().as_secs_f64() * 1e3;
        let cleared = g.download(bt, c.nout * 8)?.iter().all(|&b| b == 0) && g.download(bf, c.nout * 8)?.iter().all(|&b| b == 0);
        let fl = i32::from_le_bytes(g.download(fail, 4)?.try_into().unwrap());
        Ok((f64s(&g.download(bfs, c.nfs * 8)?), f64s(&g.download(bwx, c.nwx * 8)?), ms, cleared, fl))
    };

    // 3. augmented primal identity and zero seed
    g.clear(c.taug, c.nout * 4)?; g.clear(c.fracs, c.nout * 4)?;
    let (zt, zf, tl_ms) = tl(&g, &vec![0.0; c.nfs], &vec![0.0; c.nwx])?;
    let aug = check(&g)?;
    let zero_nonzero = zt.iter().chain(zf.iter()).filter(|v| **v != 0.0).count();

    // 4. dot tests: 3 seeds over (fs, wx) -> (taug, fracs), plus per-input and per-output splits
    let mut dots = Vec::new(); let mut ad_ms = 0.0; let mut cleared_all = true; let mut fail_flag = 0;
    let rel = |l: f64, r: f64| (l - r).abs() / l.abs().max(r.abs()).max(1e-300);
    for seed in 1..=3u64 {
        let xf = num::random(seed * 11, c.nfs); let xw = num::random(seed * 11 + 1, c.nwx);
        let wt = num::random(seed * 11 + 2, c.nout); let wf = num::random(seed * 11 + 3, c.nout);
        let (yt, yf, _) = tl(&g, &xf, &xw)?;
        let (bxf, bxw, ms, cl, fl) = ad(&g, &wt, &wf)?; ad_ms = ms; cleared_all &= cl; fail_flag = fail_flag.max(fl);
        let lt = num::dot(&yt, &wt); let lf = num::dot(&yf, &wf);
        let rf = num::dot(&xf, &bxf); let rw = num::dot(&xw, &bxw);
        // split checks: fs only -> taug only, wx only -> taug only
        let (yt_f, _, _) = tl(&g, &xf, &vec![0.0; c.nwx])?; let (yt_w, _, _) = tl(&g, &vec![0.0; c.nfs], &xw)?;
        let (bxf_t, bxw_t, _, _, _) = ad(&g, &wt, &vec![0.0; c.nout])?;
        dots.push(json!({"seed": seed, "all": {"lhs": lt + lf, "rhs": rf + rw, "rel": rel(lt + lf, rf + rw)},
            "fs_to_taug": {"lhs": num::dot(&yt_f, &wt), "rhs": num::dot(&xf, &bxf_t), "rel": rel(num::dot(&yt_f, &wt), num::dot(&xf, &bxf_t))},
            "wx_to_taug": {"lhs": num::dot(&yt_w, &wt), "rhs": num::dot(&xw, &bxw_t), "rel": rel(num::dot(&yt_w, &wt), num::dot(&xw, &bxw_t))}}));
    }
    // two reverse runs bitwise
    let wt = num::random(901, c.nout); let wf = num::random(902, c.nout);
    let (r1f, r1w, _, _, _) = ad(&g, &wt, &wf)?; let (r2f, r2w, _, _, _) = ad(&g, &wt, &wf)?;
    let reverse_bitwise = bytes_f64(&r1f) == bytes_f64(&r2f) && bytes_f64(&r1w) == bytes_f64(&r2w);

    // 5. TL vs centred FD of the unchanged captured image, relative full-state direction
    let mut fd = Vec::new();
    for seed in [5u64, 6] {
        let uf = num::random(seed * 101, c.nfs); let uw = num::random(seed * 101 + 1, c.nwx);
        for eps in [0.1f64, 0.03, 0.01, 0.003, 0.001, 3e-4, 1e-4] {
            let pf = |s: f64| -> Vec<f32> { fs0.iter().zip(&uf).map(|(x, u)| (*x as f64 + s * eps * (*x as f64) * u) as f32).collect() };
            let pw = |s: f64| -> Vec<f32> { wx0.iter().zip(&uw).map(|(x, u)| (*x as f64 + s * eps * (*x as f64) * u) as f32).collect() };
            let (xpf, xmf, xpw, xmw) = (pf(1.0), pf(-1.0), pw(1.0), pw(-1.0));
            let run_at = |g: &Gpu, xf: &[f32], xw: &[f32]| -> Result<(Vec<f32>, Vec<f32>), String> {
                g.upload(c.fs, &bytes_f32(xf))?; g.upload(c.wx, &bytes_f32(xw))?;
                f_orig.launch(c.grid, c.block, &fwd_args(&c, c.fs, c.wx))?; g.sync()?;
                Ok((f32s(&g.download(c.taug, c.nout * 4)?), f32s(&g.download(c.fracs, c.nout * 4)?)))
            };
            let (ypt, ypf) = run_at(&g, &xpf, &xpw)?; let (ymt, ymf) = run_at(&g, &xmf, &xmw)?;
            g.upload(c.fs, &bytes_f32(&fs0))?; g.upload(c.wx, &bytes_f32(&wx0))?;
            let df_: Vec<f64> = xpf.iter().zip(&xmf).map(|(p, m)| (*p as f64 - *m as f64) / (2.0 * eps)).collect();
            let dw_: Vec<f64> = xpw.iter().zip(&xmw).map(|(p, m)| (*p as f64 - *m as f64) / (2.0 * eps)).collect();
            let (yt, yf, _) = tl(&g, &df_, &dw_)?;
            let fdt: Vec<f64> = ypt.iter().zip(&ymt).map(|(p, m)| (*p as f64 - *m as f64) / (2.0 * eps)).collect();
            let fdf: Vec<f64> = ypf.iter().zip(&ymf).map(|(p, m)| (*p as f64 - *m as f64) / (2.0 * eps)).collect();
            let et = (num::diff2(&fdt, &yt) / num::norm2(&yt)).sqrt(); let ef = (num::diff2(&fdf, &yf) / num::norm2(&yf).max(1e-300)).sqrt();
            let eall = ((num::diff2(&fdt, &yt) + num::diff2(&fdf, &yf)) / (num::norm2(&yt) + num::norm2(&yf))).sqrt();
            // rows whose taug jumps far from the tangent: a switch crossed (branch flip) at this size
            let flips = (0..c.nout).filter(|&i| (fdt[i] - yt[i]).abs() > 0.1 * yt[i].abs() + 1e-6 * (ypt[i].abs() as f64) / eps).count();
            // same test for the Planck fractions, and both graded again with the far words left out: if the
            // far count falls in proportion to eps and the rest converge, the misses are kinks crossed, not a
            // wrong derivative (a wrong rule would keep a fixed set of far words at every eps)
            let far_frac: f64 = std::env::var("P7_FAR_FRAC").ok().and_then(|s| s.parse().ok()).unwrap_or(0.1);
            let far = |fdv: &Vec<f64>, tl: &Vec<f64>, y: &Vec<f32>, i: usize| (fdv[i] - tl[i]).abs() > far_frac * tl[i].abs() + 1e-6 * (y[i].abs() as f64) / eps;
            let flips_f = (0..c.nout).filter(|&i| far(&fdf, &yf, &ypf, i)).count();
            let keep = |fdv: &Vec<f64>, tl: &Vec<f64>, y: &Vec<f32>| -> f64 {
                let e = num::par_map_sum(c.nout, |i| if far(fdv, tl, y, i) { 0.0 } else { (fdv[i] - tl[i]).powi(2) });
                let d = num::par_map_sum(c.nout, |i| if far(fdv, tl, y, i) { 0.0 } else { tl[i].powi(2) });
                (e / d.max(1e-300)).sqrt() };
            let (et_k, ef_k) = (keep(&fdt, &yt, &ypt), keep(&fdf, &yf, &ypf));
            fd.push(json!({"seed": seed, "eps": eps, "rel_taug": et, "rel_fracs": ef, "rel_all": eall, "taug_words_far_from_tl": flips,
                "fracs_words_far_from_tl": flips_f, "rel_taug_without_far": et_k, "rel_fracs_without_far": ef_k}));
            eprintln!("FD seed {seed} eps {eps:e}: fracs far {flips_f} taug-kept {et_k:.3e} fracs-kept {ef_k:.3e}");
            eprintln!("FD seed {seed} eps {eps:e}: taug {et:.3e} fracs {ef:.3e} all {eall:.3e} far {flips}");
        }
    }
    // cost: three timed repeats each
    let mut fwd_ms = Vec::new(); let mut tlv = Vec::new(); let mut adv = Vec::new();
    for _ in 0..3 {
        g.sync()?; let t = Instant::now(); f_orig.launch(c.grid, c.block, &fwd_args(&c, c.fs, c.wx))?; g.sync()?; fwd_ms.push(t.elapsed().as_secs_f64() * 1e3);
        tlv.push(tl(&g, &vec![0.0; c.nfs], &vec![0.0; c.nwx])?.2);
        adv.push(ad(&g, &wt, &wf)?.2);
    }
    let med = |v: &mut Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
    let (fm, tm, am) = (med(&mut fwd_ms), med(&mut tlv), med(&mut adv));
    let _ = (fwd_ms0, tl_ms, ad_ms);
    g.free_all();
    Ok(json!({"unit": "P7", "launch": "rlw_taumol_batched", "capture": cap.display().to_string(),
        "ncol": c.ncol, "nl": c.nl, "nvrtc_options": opts, "compile_s": compile_s,
        "tables_chi_match": chi_match,
        "capture_replay_differing_words": {"taug": replay.0, "fracs": replay.1, "of": c.nout},
        "transform_selfcheck_differing_words": {"taug": selfcheck.0, "fracs": selfcheck.1},
        "augmented_primal_differing_words": {"taug": aug.0, "fracs": aug.1},
        "zero_seed_nonzero_tangent_words": zero_nonzero,
        "dot": dots, "adjoint_tape_overflow": fail_flag, "adjoint_clears_overwritten_cotangents": cleared_all,
        "reverse_runs_bitwise": reverse_bitwise, "fd": fd,
        "cost_ms": {"forward": fm, "tangent": tm, "adjoint": am, "tl_over_fwd": tm / fm, "ad_over_fwd": am / fm},
        "wall_s": t0.elapsed().as_secs_f64()}))
}

fn main() {
    match run() {
        Ok(v) => { let out = PathBuf::from(std::env::args().nth(4).unwrap());
            let s = serde_json::to_string_pretty(&v).unwrap();
            std::fs::write(out.join("p7-taumol-receipt.json"), &s).unwrap(); println!("{s}"); }
        Err(e) => { eprintln!("FAILED: {e}"); std::process::exit(1); }
    }
}
