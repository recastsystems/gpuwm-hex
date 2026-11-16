// SPDX-License-Identifier: Apache-2.0
//! P7 generic proof of a gen_kernel.py whole-launch tangent/adjoint on one real
//! captured hrrrv5 launch: capture replay (every allocation word), generated
//! forward self-check, augmented-primal identity, zero seed, float64 dot tests,
//! two reverse runs bitwise, TL vs centred FD of the unchanged captured image, cost.
#[path = "../gpu.rs"] mod gpu;
#[path = "../capture.rs"] mod capture;
#[path = "../num.rs"] mod num;
use capture::*; use gpu::*; use serde_json::{json, Value};
use std::path::PathBuf; use std::time::Instant;

struct ArgSlot { name: String, scalar: Option<(String, [u8; 8])>, alloc: usize, offset: usize, words: usize }

fn main() {
    match run() {
        Ok(v) => { let out = PathBuf::from(std::env::args().nth(4).unwrap());
            let s = serde_json::to_string_pretty(&v).unwrap();
            std::fs::write(out.join("receipt.json"), &s).unwrap(); println!("{s}"); }
        Err(e) => { eprintln!("FAILED: {e}"); std::process::exit(1); }
    }
}

fn run() -> Result<Value, String> {
    let a: Vec<String> = std::env::args().collect();
    let (spec_p, cap, gen, out, nvrtc) = (PathBuf::from(&a[1]), PathBuf::from(&a[2]), PathBuf::from(&a[3]), PathBuf::from(&a[4]), a[5].clone());
    let spec: Value = serde_json::from_slice(&std::fs::read(&spec_p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let kname = spec["kernel"].as_str().unwrap().to_string();
    let ain: Vec<String> = spec["active_in"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    let aout: Vec<String> = spec["active_out"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    let arrays: Vec<String> = ain.iter().chain(aout.iter()).cloned().collect();
    let t0 = Instant::now();
    let src = std::fs::read_to_string(&gen).map_err(|e| e.to_string())?;
    // L2 (ad/window-L2-bindings): P7_ARCH picks the card (default sm_120, the 5090). On another arch the captured
    // original image (an sm_120 cubin) cannot load, so the capture's own source is recompiled with these options,
    // and the capture replay below still grades that forward against the captured post state word for word.
    let arch = std::env::var("P7_ARCH").unwrap_or_else(|_| "sm_120".into());
    let arch_opt = format!("-arch={arch}");
    let opts = ["-std=c++17", "--ftz=false", "--fmad=false", "--prec-div=true", "--prec-sqrt=true", arch_opt.as_str()];
    let tc = Instant::now();
    let cubin = nvrtc_compile(&nvrtc, &src, "g7.cu", &opts)?;
    let compile_s = tc.elapsed().as_secs_f64();
    if std::env::var_os("P7_COMPILE_ONLY").is_some() { return Ok(json!({"compiled": cubin.len(), "compile_s": compile_s})); }
    let l = Launch::load(&cap)?;
    let mut g = Gpu::new()?;
    let orig_image = if arch == "sm_120" { l.image.clone() } else {
        let fsrc = std::fs::read_to_string(l.dir.join(l.manifest["source"]["path"].as_str().ok_or("source path")?)).map_err(|e| e.to_string())?;
        nvrtc_compile(&nvrtc, &fsrc, "forward.cu", &opts)?
    };
    let orig = g.module(&orig_image)?; let genm = g.module(&cubin)?;
    let f_orig = orig.func(&kname)?; let f_self = genm.func(&format!("g7_{kname}_fwd"))?;
    let f_tl = genm.func(&format!("{kname}__tl"))?; let f_ad = genm.func(&format!("{kname}__ad"))?;
    let indirect = l.manifest["indirect_pointers"].as_array().map(|v| v.len()).unwrap_or(0);

    let mut dev = Vec::new();
    for b in &l.pre { dev.push(g.put(b)?); }
    let restore = |g: &Gpu| -> Result<(), String> { for (i, b) in l.pre.iter().enumerate() { g.upload(dev[i], b)?; } Ok(()) };
    let mut slots = Vec::new();
    for arg in l.manifest["arguments"].as_array().unwrap() {
        let name = arg["name"].as_str().unwrap().to_string();
        if arg["kind"] == "scalar" {
            let h = arg["value_hex"].as_str().unwrap(); let mut b = [0u8; 8];
            for i in 0..h.len() / 2 { b[i] = u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap(); }
            slots.push(ArgSlot { name, scalar: Some((arg["dtype"].as_str().unwrap().to_string(), b)), alloc: 0, offset: 0, words: 0 });
        } else {
            let words: usize = arg["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).product();
            slots.push(ArgSlot { name, scalar: None, alloc: arg["allocation"].as_u64().unwrap() as usize,
                                 offset: arg["offset"].as_u64().unwrap_or(0) as usize, words });
        }
    }
    let ptr_of = |s: &ArgSlot| dev[s.alloc] + s.offset as u64;
    let base_args = || -> Vec<Arg> { slots.iter().map(|s| match &s.scalar {
        Some((dt, b)) => match dt.as_str() { "<f4" => Arg::F32(f32::from_le_bytes(b[..4].try_into().unwrap())),
            "<i4" | "<u4" => Arg::I32(i32::from_le_bytes(b[..4].try_into().unwrap())), _ => Arg::Ptr(u64::from_le_bytes(*b)) },
        None => Arg::Ptr(ptr_of(s)) }).collect() };
    let slot = |n: &str| slots.iter().find(|s| s.name == n).unwrap();
    // refuse overlapping active ranges (each array gets its own derivative buffer)
    let ranges: Vec<(usize, usize, usize)> = arrays.iter().map(|n| { let s = slot(n); (s.alloc, s.offset, s.offset + 4 * s.words) }).collect();
    for i in 0..ranges.len() { for j in i + 1..ranges.len() {
        if ranges[i].0 == ranges[j].0 && ranges[i].1 < ranges[j].2 && ranges[j].1 < ranges[i].2 {
            return Err(format!("active arrays {} and {} overlap", arrays[i], arrays[j])); } } }
    let (grid, block) = l.geometry();
    let threads = (grid[0] as usize) * (grid[1] as usize) * (block[0] as usize) * (block[1] as usize);

    let compare = |g: &Gpu| -> Result<Vec<usize>, String> { g.sync()?; let mut v = Vec::new();
        for (i, p) in l.post.iter().enumerate() { v.push(words_differ(&g.download(dev[i], p.len())?, p)); } Ok(v) };
    let tf = Instant::now(); f_orig.launch(grid, block, &base_args())?; g.sync()?; let _ = tf;
    let replay = compare(&g)?;
    restore(&g)?; f_self.launch(grid, block, &base_args())?;
    let selfcheck = compare(&g)?;

    // derivative buffers
    let mut dbuf = Vec::new(); let mut bbuf = Vec::new();
    for n in &arrays { let w = slot(n).words; dbuf.push(g.alloc(w * 8)?); bbuf.push(g.alloc(w * 8)?); }
    let tcap: usize = spec["tape"].as_u64().unwrap() as usize; let wcap: usize = spec["writes"].as_u64().unwrap() as usize;
    let s_a = g.alloc(threads * tcap * 4)?; let s_b = g.alloc(threads * tcap * 4)?;
    let s_pa = g.alloc(threads * tcap * 8)?; let s_pb = g.alloc(threads * tcap * 8)?; let s_adj = g.alloc(threads * tcap * 8)?;
    let s_leaf = g.alloc(threads * tcap * 2)?; let s_li = g.alloc(threads * tcap * 8)?;
    let s_warr = g.alloc(threads * wcap * 2)?; let s_widx = g.alloc(threads * wcap * 8)?; let s_wt = g.alloc(threads * wcap * 4)?;
    let mut ti = Vec::new(); let mut sv = Vec::new();
    for n in &aout { let w = slot(n).words; ti.push(g.alloc(w * 4)?); sv.push(g.alloc(w * 4)?); }
    let mut sc: Vec<u8> = Vec::new();
    for v in [tcap as i32, wcap as i32, ain.len() as i32, 0] { sc.extend(v.to_le_bytes()); }
    for p in [s_a, s_b, s_pa, s_pb, s_adj, s_leaf, s_li, s_warr, s_widx, s_wt] { sc.extend(p.to_le_bytes()); }
    for k in 0..64 { sc.extend(ti.get(k).copied().unwrap_or(0).to_le_bytes()); }
    for k in 0..64 { sc.extend(sv.get(k).copied().unwrap_or(0).to_le_bytes()); }
    let scratch = g.put(&sc)?; let status = g.alloc(16)?;

    let nin = ain.len();
    let tl = |g: &Gpu, x: &[Vec<f64>]| -> Result<(Vec<Vec<f64>>, f64), String> {
        restore(g)?;
        for k in 0..arrays.len() { g.upload(dbuf[k], &bytes_f64(&x[k]))?; }
        let mut args = base_args(); for d in &dbuf { args.push(Arg::Ptr(*d)); }
        g.sync()?; let t = Instant::now(); f_tl.launch(grid, block, &args)?; g.sync()?; let ms = t.elapsed().as_secs_f64() * 1e3;
        let mut y = Vec::new(); for k in nin..arrays.len() { y.push(f64s(&g.download(dbuf[k], slot(&arrays[k]).words * 8)?)); }
        Ok((y, ms))
    };
    let ad = |g: &Gpu, w: &[Vec<f64>]| -> Result<(Vec<Vec<f64>>, f64, [i32; 3]), String> {
        restore(g)?;
        for (k, n) in arrays.iter().enumerate() { if k < nin { g.clear(bbuf[k], slot(n).words * 8)?; } else { g.upload(bbuf[k], &bytes_f64(&w[k - nin]))?; } }
        for (k, n) in aout.iter().enumerate() { g.upload(ti[k], &vec![0xFFu8; slot(n).words * 4])?; }
        g.clear(status, 16)?;
        let mut args = base_args(); for b in &bbuf { args.push(Arg::Ptr(*b)); } args.push(Arg::Ptr(scratch)); args.push(Arg::Ptr(status));
        g.sync()?; let t = Instant::now(); f_ad.launch(grid, block, &args)?; g.sync()?; let ms = t.elapsed().as_secs_f64() * 1e3;
        let st = g.download(status, 12)?; let st = [0, 1, 2].map(|i| i32::from_le_bytes(st[4 * i..4 * i + 4].try_into().unwrap()));
        let mut bx = Vec::new(); for k in 0..arrays.len() { bx.push(f64s(&g.download(bbuf[k], slot(&arrays[k]).words * 8)?)); }
        Ok((bx, ms, st))
    };

    // augmented primal identity + zero seed
    let zeros: Vec<Vec<f64>> = arrays.iter().map(|n| vec![0.0; slot(n).words]).collect();
    let (zy, _) = tl(&g, &zeros)?;
    let aug = compare(&g)?;
    let zero_nonzero: usize = zy.iter().map(|v| v.iter().filter(|x| **x != 0.0).count()).sum();
    // dot tests
    let rel = |l: f64, r: f64| (l - r).abs() / l.abs().max(r.abs()).max(1e-300);
    // optional weather-regime column classes (P7_CLASSES=file from gen/colclass.py): per-regime dot and FD
    let ncl: usize = spec["columns_arg"].as_str().map(|n| { let s = slot(n); i32::from_le_bytes(s.scalar.as_ref().unwrap().1[..4].try_into().unwrap()) as usize }).unwrap_or(1);
    let classes: Vec<(String, Vec<usize>)> = match std::env::var("P7_CLASSES") {
        Ok(p) => {
            let v: Value = serde_json::from_slice(&std::fs::read(&p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
            if v["ncol"].as_u64() != Some(ncl as u64) { return Err(format!("class file has {} columns, launch has {ncl}", v["ncol"])); }
            for n in &arrays { if slot(n).words % ncl != 0 { return Err(format!("{n} is not column-leading")); } }
            v["classes"].as_object().unwrap().iter().map(|(k, c)| (k.clone(), c.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect())).collect()
        }
        Err(_) => Vec::new() };
    // per-column sum of f(i) over the words of each column (columns are the leading dimension), every core
    let colsum = |len: usize, f: &(dyn Fn(usize) -> f64 + Sync)| -> Vec<f64> {
        let pc = len / ncl; let mut out = vec![0.0; ncl]; let chunk = ncl.div_ceil(num::threads()).max(1);
        std::thread::scope(|s| { for (k, part) in out.chunks_mut(chunk).enumerate() {
            s.spawn(move || for (j, o) in part.iter_mut().enumerate() { let c = k * chunk + j; let mut acc = 0.0; for i in c * pc..(c + 1) * pc { acc += f(i); } *o = acc; }); } });
        out };
    let mut class_dots = Vec::new();
    let mut dots = Vec::new(); let mut stat = [0; 3];
    for seed in 1..=3u64 {
        // every array's incoming tangent is seeded: written words drop it, untouched or in-place words carry it
        let x: Vec<Vec<f64>> = arrays.iter().enumerate().map(|(k, n)| num::random(seed * 1000 + k as u64, slot(n).words)).collect();
        let w: Vec<Vec<f64>> = aout.iter().enumerate().map(|(k, n)| num::random(seed * 1000 + 500 + k as u64, slot(n).words)).collect();
        let (y, _) = tl(&g, &x)?; let (bx, _, st) = ad(&g, &w)?; stat = st;
        let lhs: f64 = y.iter().zip(&w).map(|(a, b)| num::dot(a, b)).sum();
        let rhs: f64 = x.iter().zip(&bx).map(|(a, b)| num::dot(a, b)).sum();
        dots.push(json!({"seed": seed, "lhs": lhs, "rhs": rhs, "rel": rel(lhs, rhs)}));
        if !classes.is_empty() {
            let mut lc = vec![0.0; ncl]; let mut rc = vec![0.0; ncl];
            for (a, b) in y.iter().zip(&w) { for (o, v) in lc.iter_mut().zip(colsum(a.len(), &|i| a[i] * b[i])) { *o += v; } }
            for (a, b) in x.iter().zip(&bx) { for (o, v) in rc.iter_mut().zip(colsum(a.len(), &|i| a[i] * b[i])) { *o += v; } }
            // worst single column too: a per-column adjoint error cannot hide in a class sum
            let worst_col = (0..ncl).map(|c| rel(lc[c], rc[c])).fold(0.0f64, f64::max);
            let mut per = serde_json::Map::new();
            for (name, cols) in &classes {
                let (l2, r2): (f64, f64) = cols.iter().fold((0.0, 0.0), |(p, q), &c| (p + lc[c], q + rc[c]));
                let wc = cols.iter().map(|&c| rel(lc[c], rc[c])).fold(0.0f64, f64::max);
                per.insert(name.clone(), json!({"columns": cols.len(), "rel": rel(l2, r2), "worst_column_rel": wc}));
            }
            eprintln!("DOT seed {seed} worst single column rel {worst_col:.3e}");
            class_dots.push(json!({"seed": seed, "worst_column_rel": worst_col, "classes": per}));
        }
        eprintln!("DOT seed {seed}: {lhs:.15e} {rhs:.15e} rel {:.3e} tape {:?}", rel(lhs, rhs), st);
    }
    // a full tape drops nodes and the adjoint is silently wrong (seen: spec tape 9000 on the 65-layer call needing
    // 9,550 entries gave dot misses of 6e-2 to 7e-1 with rc 0); refuse the receipt instead of grading a broken run
    if stat[0] != 0 {
        return Err(format!("adjoint tape overflow (status {:?}; spec tape {} / writes {}): size the tape for this launch's layer count", stat, tcap, wcap));
    }
    // P7_GAIN=1 (held lane rlw_rtrn_prol_gpoints): the tangent's gain ||TL(x)|| / ||x||, with the outputs'
    // incoming tangent 0 so y is the launch Jacobian applied to x. x on all active inputs together and on
    // each alone; two direction kinds: iid (num::random words) and relative (x = x0 * u, the FD's kind).
    // Per output field: TL norm, nonzero words and the norm of y / y0 over words with y0 != 0 (y0 = the
    // captured post value), so the relative kind also gives the relative-to-relative gain.
    let mut gains = Vec::new();
    if std::env::var_os("P7_GAIN").is_some() {
        let y0: Vec<Vec<f32>> = aout.iter().map(|n| { let s = slot(n); f32s(&l.post[s.alloc][s.offset..s.offset + 4 * s.words]) }).collect();
        for kind in ["iid", "relative"] { for sel in std::iter::once(None).chain((0..nin).map(Some)) {
            let on = |k: usize| k < nin && sel.map_or(true, |s| s == k);
            let u: Vec<Vec<f64>> = arrays.iter().enumerate().map(|(k, n)| if on(k) { num::random(91000 + k as u64, slot(n).words) } else { vec![0.0; slot(n).words] }).collect();
            let x0: Vec<Vec<f32>> = arrays.iter().map(|n| { let s = slot(n); f32s(&l.pre[s.alloc][s.offset..s.offset + 4 * s.words]) }).collect();
            let x: Vec<Vec<f64>> = if kind == "iid" { u.clone() } else {
                u.iter().zip(&x0).map(|(v, x0)| v.iter().zip(x0).map(|(a, b)| a * *b as f64).collect()).collect() };
            let nu_eff: f64 = u.iter().zip(&x0).map(|(v, x0)| num::par_map_sum(v.len(), |i| if x0[i] != 0.0 { v[i] * v[i] } else { 0.0 })).sum::<f64>().sqrt();
            let (y, _) = tl(&g, &x)?;
            let nx = x.iter().map(|v| num::norm2(v)).sum::<f64>().sqrt();
            let ny = y.iter().map(|v| num::norm2(v)).sum::<f64>().sqrt();
            let mut rr_all = 0.0;
            let per: Vec<Value> = aout.iter().enumerate().map(|(k, n)| { let (yk, pk) = (&y[k], &y0[k]);
                let nz = num::par_map_sum(yk.len(), |i| if yk[i] != 0.0 { 1.0 } else { 0.0 });
                let rr = num::par_map_sum(yk.len(), |i| if pk[i] != 0.0 { (yk[i] / pk[i] as f64).powi(2) } else { 0.0 });
                rr_all += rr;
                json!({"field": n, "tl_norm": num::norm2(yk).sqrt(), "nonzero_words": nz as u64, "rel_tangent_norm": rr.sqrt()}) }).collect();
            let name = sel.map_or("all".to_string(), |s| arrays[s].clone());
            eprintln!("GAIN {kind} x on {name}: ||x|| {nx:.6e} ||TL x|| {ny:.6e} gain {:.6e}", ny / nx.max(1e-300));
            gains.push(json!({"kind": kind, "x_on": name, "x_norm": nx, "tl_norm": ny, "gain": ny / nx.max(1e-300),
                "u_norm_where_x0_nonzero": nu_eff, "rel_to_rel_gain": if kind == "relative" { json!(rr_all.sqrt() / nu_eff.max(1e-300)) } else { json!(null) },
                "per_field": per}));
        } }
    }
    let w: Vec<Vec<f64>> = aout.iter().enumerate().map(|(k, n)| num::random(77000 + k as u64, slot(n).words)).collect();
    let (r1, _, _) = ad(&g, &w)?; let (r2, _, _) = ad(&g, &w)?;
    let reverse_bitwise = r1.iter().zip(&r2).all(|(a, b)| bytes_f64(a) == bytes_f64(b));

    // Held lane rlw_setcoef (ad/held-rlw_setcoef): the launch gain ||TL x|| / ||x|| with the tangent seeded on the active
    // inputs only (incoming output tangents zero), on two directions: "normal" (x uniform in [-1, 1) per word) and
    // "relative" (x_q = u_q |x0_q|, graded as (||TL x|| / ||y0||) / (||x|| / ||x0||), y0 the captured post outputs).
    // Each gives per-field shares of ||TL x||^2 and the largest |TL x| word (field, column). sha256 of the gain tangent and
    // of the first reverse result let two processes be compared bitwise from their receipts alone.
    let sha = |v: &[Vec<f64>]| -> String { use sha2::{Digest, Sha256}; let mut h = Sha256::new(); for a in v { h.update(bytes_f64(a)); } format!("{:x}", h.finalize()) };
    let reverse_sha256 = sha(&r1);
    let x0in: Vec<Vec<f64>> = ain.iter().map(|n| { let s = slot(n); f32s(&l.pre[s.alloc][s.offset..s.offset + 4 * s.words]).into_iter().map(f64::from).collect() }).collect();
    let y0: Vec<Vec<f64>> = aout.iter().map(|n| { let s = slot(n); f32s(&l.post[s.alloc][s.offset..s.offset + 4 * s.words]).into_iter().map(f64::from).collect() }).collect();
    let nx0 = x0in.iter().map(|v| num::norm2(v)).sum::<f64>().sqrt();
    let ny0 = y0.iter().map(|v| num::norm2(v)).sum::<f64>().sqrt();
    let mut gains = serde_json::Map::new();
    for (label, rel_dir) in [("normal", false), ("relative", true)] {
        let u: Vec<Vec<f64>> = arrays.iter().enumerate().map(|(k, n)| if k < nin {
            let r = num::random(91000 + k as u64, slot(n).words);
            if rel_dir { r.iter().zip(&x0in[k]).map(|(u, x)| u * x.abs()).collect() } else { r }
        } else { vec![0.0; slot(n).words] }).collect();
        let (y, _) = tl(&g, &u)?;
        let nx = u[..nin].iter().map(|v| num::norm2(v)).sum::<f64>().sqrt();
        let fy: Vec<f64> = y.iter().map(|v| num::norm2(v)).collect();
        let ny = fy.iter().sum::<f64>().sqrt();
        let nonfinite: usize = y.iter().map(|v| v.iter().filter(|a| !a.is_finite()).count()).sum();
        let (mut big, mut bf, mut bi) = (0.0f64, 0usize, 0usize);
        for (k, v) in y.iter().enumerate() { for (i, a) in v.iter().enumerate() { if a.abs() > big { big = a.abs(); bf = k; bi = i; } } }
        let per_col = slot(&aout[bf]).words / ncl.max(1);
        let mut share: Vec<(usize, f64)> = fy.iter().copied().enumerate().collect();
        share.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let tot = fy.iter().sum::<f64>().max(1e-300);
        let top: Vec<Value> = share.iter().take(6).map(|(k, v)| json!({"field": aout[*k], "share": v / tot,
            "gain": if rel_dir { (v.sqrt() / ny0.max(1e-300)) / (nx / nx0.max(1e-300)) } else { v.sqrt() / nx.max(1e-300) }})).collect();
        let gv = if rel_dir { (ny / ny0.max(1e-300)) / (nx / nx0.max(1e-300)) } else { ny / nx.max(1e-300) };
        eprintln!("GAIN {label}: {gv:.4e} (largest |TL x| {big:.3e} in {} column {}; {nonfinite} nonfinite)", aout[bf], bi / per_col.max(1));
        gains.insert(label.into(), json!({"gain": gv, "norm_x": nx, "norm_tl_x": ny, "norm_x0": nx0, "norm_y0": ny0,
            "largest_abs_tl_x": big, "largest_field": aout[bf], "largest_column": bi / per_col.max(1), "nonfinite_words": nonfinite,
            "top_fields": top, "tl_sha256": sha(&y)}));
    }

    // TL vs centred FD of the unchanged captured image (relative full-state direction)
    let x0: Vec<Vec<f32>> = arrays.iter().map(|n| { let s = slot(n); f32s(&l.pre[s.alloc][s.offset..s.offset + 4 * s.words]) }).collect();
    let mask_ints: Vec<String> = spec["mask_int_outputs"].as_array().map(|v| v.iter().map(|x| x.as_str().unwrap().to_string()).collect()).unwrap_or_default();
    let ncolm: usize = spec["columns_arg"].as_str().map(|n| { let s = slot(n); i32::from_le_bytes(s.scalar.as_ref().unwrap().1[..4].try_into().unwrap()) as usize }).unwrap_or(1);
    // P7_FD_GAP=name: perturb that level array by a fraction of its smallest neighbouring level gap, so
    // the FD direction keeps level order (a relative per-level step turns thin top layers inside out)
    let fd_gap: Vec<String> = std::env::var("P7_FD_GAP").map(|s| s.split(',').map(String::from).collect()).unwrap_or_default();
    let fd_fixed: Vec<String> = spec["fd_fixed"].as_array().map(|v| v.iter().map(|x| x.as_str().unwrap().to_string()).filter(|n| !fd_gap.contains(n)).collect()).unwrap_or_default();
    // P7_FLOOR_MASKS=[{"out","in","offset","inner"}]: an output word read from a table at (int)(in - offset);
    // graded separately on words whose table index is the same in the + and - runs (no 1 K kink crossed)
    let floor_masks: Vec<Value> = std::env::var("P7_FLOOR_MASKS").ok().map(|s| serde_json::from_str::<Value>(&s).unwrap().as_array().unwrap().clone()).unwrap_or_default();
    let mut fd = Vec::new();
    for seed in [5u64, 6] {
        let u: Vec<Vec<f64>> = arrays.iter().enumerate().map(|(k, n)| if fd_fixed.contains(n) { vec![0.0; slot(n).words] }
            else if fd_gap.contains(n) {
                let r = num::random(seed * 7919 + k as u64, slot(n).words); let x = &x0[k]; let pc = x.len() / ncl;
                (0..x.len()).map(|i| { let l = i % pc;
                    let g1 = if l > 0 { (x[i] as f64 - x[i - 1] as f64).abs() } else { f64::INFINITY };
                    let g2 = if l + 1 < pc { (x[i + 1] as f64 - x[i] as f64).abs() } else { f64::INFINITY };
                    let g = g1.min(g2); if x[i] == 0.0 || !g.is_finite() { 0.0 } else { r[i] * 0.4 * g / (x[i] as f64).abs() } }).collect()
            } else { num::random(seed * 7919 + k as u64, slot(n).words) }).collect();
        for eps in [0.1f64, 0.03, 0.01, 0.003, 0.001, 3e-4, 1e-4] {
            let pert = |sg: f64| -> Vec<Vec<f32>> { x0.iter().zip(&u).map(|(x, u)| num::par_vec(x.len(), |i| (x[i] as f64 + sg * eps * (x[i] as f64) * u[i]) as f32)).collect() };
            let (xp, xm) = (pert(1.0), pert(-1.0));
            let run_at = |g: &Gpu, x: &[Vec<f32>]| -> Result<(Vec<Vec<f32>>, Vec<Vec<u8>>), String> {
                restore(g)?;
                for (k, n) in arrays.iter().enumerate() { g.upload(ptr_of(slot(n)), &bytes_f32(&x[k]))?; }
                f_orig.launch(grid, block, &base_args())?; g.sync()?;
                let mut y = Vec::new(); for n in &aout { let s = slot(n); y.push(f32s(&g.download(ptr_of(s), 4 * s.words)?)); }
                let mut iv = Vec::new(); for n in &mask_ints { let s = slot(n); iv.push(g.download(ptr_of(s), 4 * s.words)?); }
                Ok((y, iv)) };
            let ((yp, ip), (ym, im)) = (run_at(&g, &xp)?, run_at(&g, &xm)?);
            // columns whose recorded index selections differ between the two runs crossed a switch
            let mut stable = vec![true; ncolm];
            for (k, n) in mask_ints.iter().enumerate() {
                let per_col = slot(n).words / ncolm;
                for (w, (a, b)) in ip[k].chunks_exact(4).zip(im[k].chunks_exact(4)).enumerate() { if a != b { stable[w / per_col] = false; } }
            }
            let nstable = stable.iter().filter(|s| **s).count();
            let dx: Vec<Vec<f64>> = xp.iter().zip(&xm).map(|(p, m)| num::par_vec(p.len(), |i| (p[i] as f64 - m[i] as f64) / (2.0 * eps))).collect();
            let (ty, _) = tl(&g, &dx)?;
            let mut num_ = 0.0; let mut den = 0.0; let mut per = Vec::new(); let mut mnum = 0.0; let mut mden = 0.0;
            let mut class_fd = serde_json::Map::new();
            let mut cfields: Vec<(String, Vec<f64>, Vec<f64>)> = Vec::new();
            for (k, n) in aout.iter().enumerate() {
                let f: Vec<f64> = { let (p, m) = (&yp[k], &ym[k]); num::par_vec(p.len(), |i| (p[i] as f64 - m[i] as f64) / (2.0 * eps)) };
                if !classes.is_empty() {
                    let (fr, tr) = (&f, &ty[k]);
                    cfields.push((n.clone(), colsum(fr.len(), &|i| (fr[i] - tr[i]).powi(2)), colsum(fr.len(), &|i| tr[i].powi(2))));
                }
                let (e, d) = (num::diff2(&f, &ty[k]), num::norm2(&ty[k]));
                let per_col = slot(n).words / ncolm.max(1);
                let me = num::par_map_sum(f.len(), |i| if stable[i / per_col] { (f[i] - ty[k][i]).powi(2) } else { 0.0 });
                let md = num::par_map_sum(f.len(), |i| if stable[i / per_col] { ty[k][i].powi(2) } else { 0.0 });
                num_ += e; den += d; mnum += me; mden += md;
                let mut fl = json!(null);
                if let Some(fm) = floor_masks.iter().find(|m| m["out"] == n.as_str()) {
                    let ki = arrays.iter().position(|a| fm["in"] == a.as_str()).unwrap();
                    let (off, inner) = (fm["offset"].as_f64().unwrap() as f32, fm["inner"].as_u64().unwrap() as usize);
                    let (pco, pci) = (slot(n).words / ncl, slot(&arrays[ki]).words / ncl);
                    let (xpi, xmi, tr) = (&xp[ki], &xm[ki], &ty[k]);
                    let src = |i: usize| (i / pco) * pci + (i % pco) / inner;
                    let same = |i: usize| { let j = src(i); ((xpi[j] - off) as i32) == ((xmi[j] - off) as i32) };
                    let fe = num::par_map_sum(f.len(), |i| if same(i) { (f[i] - tr[i]).powi(2) } else { 0.0 });
                    let fdn = num::par_map_sum(f.len(), |i| if same(i) { tr[i].powi(2) } else { 0.0 });
                    let crossed = num::par_map_sum(f.len(), |i| if same(i) { 0.0 } else { 1.0 });
                    fl = json!({"rel": (fe / fdn.max(1e-300)).sqrt(), "words_crossing_a_table_index": crossed as u64, "words": f.len()});
                }
                per.push(json!({"field": n, "rel": (e / d.max(1e-300)).sqrt(), "rel_stable_columns": (me / md.max(1e-300)).sqrt(), "same_table_index": fl}));
            }
            for (name, cols) in &classes {
                // per regime, branch-stable columns only: aggregate and worst field (fields with nonzero tangent)
                let sc: Vec<usize> = cols.iter().copied().filter(|&c| stable[c]).collect();
                let (mut an, mut ad_) = (0.0, 0.0); let mut worst = (0.0f64, String::new());
                for (fname, e, d) in &cfields {
                    let (fe, fdn): (f64, f64) = sc.iter().fold((0.0, 0.0), |(p, q), &c| (p + e[c], q + d[c]));
                    an += fe; ad_ += fdn;
                    if fdn > 0.0 { let r = (fe / fdn).sqrt(); if r > worst.0 { worst = (r, fname.clone()); } }
                }
                class_fd.insert(name.clone(), json!({"columns": cols.len(), "stable_columns": sc.len(),
                    "rel": (an / ad_.max(1e-300)).sqrt(), "worst_field_rel": worst.0, "worst_field": worst.1}));
            }
            let r = (num_ / den).sqrt(); let rm = (mnum / mden.max(1e-300)).sqrt();
            eprintln!("FD seed {seed} eps {eps:e}: all {r:.3e} stable-columns {rm:.3e} ({nstable}/{ncolm} columns branch-stable)");
            fd.push(json!({"seed": seed, "eps": eps, "rel_all": r, "rel_stable_columns": rm, "stable_columns": nstable, "columns": ncolm, "per_field": per, "classes": class_fd}));
        }
    }
    // cost
    let mut fm = Vec::new(); let mut tm = Vec::new(); let mut am = Vec::new();
    for _ in 0..3 {
        restore(&g)?; g.sync()?; let t = Instant::now(); f_orig.launch(grid, block, &base_args())?; g.sync()?; fm.push(t.elapsed().as_secs_f64() * 1e3);
        tm.push(tl(&g, &zeros)?.1); am.push(ad(&g, &w)?.1);
    }
    let med = |v: &mut Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
    let (fmm, tmm, amm) = (med(&mut fm), med(&mut tm), med(&mut am));
    let words: usize = l.post.iter().map(|p| p.len() / 4).sum();
    g.free_all();
    Ok(json!({"unit": "P7", "launch": kname, "capture": cap.display().to_string(), "grid": grid, "block": block,
        "indirect_pointer_tables": indirect, "compile_s": compile_s, "nvrtc_options": opts,
        "active_in": ain, "active_out": aout, "allocation_words": words,
        "capture_replay_differing_words": replay.iter().sum::<usize>(), "selfcheck_differing_words": selfcheck.iter().sum::<usize>(),
        "augmented_primal_differing_words": aug.iter().sum::<usize>(), "zero_seed_nonzero_tangent_words": zero_nonzero,
        "dot": dots, "dot_by_class": class_dots, "tape_status_fail_maxentries_maxwrites": stat, "reverse_runs_bitwise": reverse_bitwise, "reverse_sha256": reverse_sha256, "gain": gains, "fd": fd,
        "cost_ms": {"forward": fmm, "tangent": tmm, "adjoint": amm, "tl_over_fwd": tmm / fmm, "ad_over_fwd": amm / fmm},
        "fd_fixed": fd_fixed, "wall_s": t0.elapsed().as_secs_f64()}))
}
