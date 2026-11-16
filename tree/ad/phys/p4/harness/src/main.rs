// SPDX-License-Identifier: Apache-2.0
//! p4-proof: unit proof of the P4 whole-launch derivatives (MYNN surface layer, GSL orographic drag)
//! on captured hrrrv5 launches: forward identity, augmented identity, dot (f32 and f64 entries,
//! full-state, per-field, zero), TL vs centred FD of the unchanged captured forward image with
//! branch-flip counts, tangent budgets, reverse determinism, cost. No Python; libcuda + NVRTC only.
mod cap;
mod check;
mod cuda;
mod nvrtc;
use cap::*;
use cuda::*;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

const EPS: [f64; 7] = [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001];
const BR_WORDS: usize = 512; // 16,384 recorded switches per column

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Rng { Rng(s.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xD1B54A32D192ED03) }
    fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn n(&mut self) -> f64 { let a = self.u().max(1e-300); let b = self.u(); (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos() }
}

fn i32b(v: i32) -> KArg { KArg::Bytes(v.to_le_bytes().to_vec()) }
fn i64b(v: i64) -> KArg { KArg::Bytes(v.to_le_bytes().to_vec()) }
fn f32b(v: f32) -> KArg { KArg::Bytes(v.to_le_bytes().to_vec()) }
fn p(b: &DBuf) -> KArg { KArg::Ptr(b.ptr) }
fn null() -> KArg { KArg::Ptr(0) }
fn dbuf_f64(v: &[f64]) -> DBuf { let b = DBuf::new(8 * v.len()); b.upload_at(0, &bytes_f64(v)); b }
fn dbuf_f32(v: &[f32]) -> DBuf { let b = DBuf::new(4 * v.len()); b.upload_at(0, &bytes_f32(v)); b }
fn read_f64(b: &DBuf, n: usize) -> Vec<f64> { f64s(&b.download_at(0, 8 * n), n) }
fn read_f32(b: &DBuf, n: usize) -> Vec<f32> { f32s(&b.download_at(0, 4 * n), n) }
fn read_i32(b: &DBuf, n: usize) -> Vec<i32> { i32s(&b.download_at(0, 4 * n), n) }
fn read_u32(b: &DBuf, n: usize) -> Vec<u32> { read_i32(b, n).into_iter().map(|x| x as u32).collect() }
fn grid1(n: usize, b: usize) -> [u32; 3] { [((n + b - 1) / b) as u32, 1, 1] }
fn ndiff(a: &[f32], b: &[f32]) -> usize { a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count() }
fn idiff(a: &[i32], b: &[i32]) -> usize { a.iter().zip(b).filter(|(x, y)| x != y).count() }
fn dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() }
fn nrm(a: &[f64]) -> f64 { a.iter().map(|x| x * x).sum::<f64>().sqrt() }
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); if v.is_empty() { f64::NAN } else { v[v.len() / 2] } }

/// std over the n entries of each of the nf planes (finite entries only).
fn plane_std(x: &[f32], nf: usize, n: usize) -> Vec<f64> {
    (0..nf).map(|j| {
        let v: Vec<f64> = x[j * n..(j + 1) * n].iter().map(|&a| a as f64).filter(|a| a.is_finite()).collect();
        if v.len() < 2 { return 0.0; }
        let m = v.iter().sum::<f64>() / v.len() as f64;
        (v.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / v.len() as f64).sqrt()
    }).collect()
}

/// Flip count between two branch records of one column.
fn flips(a: &[u32], ca: i32, b: &[u32], cb: i32, n: usize, c: usize) -> (u64, bool) {
    let m = (ca.min(cb) as usize).min(BR_WORDS * 32);
    let mut f = 0u64;
    for k in 0..(m + 31) / 32 {
        let mut x = a[k * n + c] ^ b[k * n + c];
        let hi = (m - 32 * k).min(32);
        if hi < 32 { x &= (1u32 << hi) - 1; }
        f += x.count_ones() as u64;
    }
    f += (ca - cb).unsigned_abs() as u64;
    (f, ca as usize > BR_WORDS * 32 || cb as usize > BR_WORDS * 32)
}

struct Mods { tl: Module, rev: Module }
fn load_mods(dir: &Path, fam: &str, arch: &str) -> Mods {
    let rd = |m: &str| std::fs::read(dir.join(format!("{fam}.{m}.{arch}.cubin"))).unwrap_or_else(|e| panic!("cubin {fam}.{m}: {e}"));
    let m = Mods { tl: Module::load(&rd("tl")), rev: Module::load(&rd("rev")) };
    if fam == "mynnsfc" { for md in [&m.tl, &m.rev] { tables(md); } }
    // G4: the ugwp entries keep their column arrays in a frame arena (p4_frame_base, p4_frame_bytes per launch thread),
    // sized here for up to 8,192 launch threads (the harness launches cover one capture, nc <= 8,192).
    if fam == "ugwp" { for md in [&m.tl, &m.rev] { md.p4_frame(8192); } }
    m
}
fn tables(m: &Module) {
    let e = DBuf::new(4);
    launch(m.func("mynnsfc_tables"), [1, 1, 1], [1, 1, 1], 0, &[p(&e)]);
    sync();
    assert_eq!(read_i32(&e, 1)[0], 0, "mynnsfc_tables error");
}

// ------------------------------------------------------------------------------------------ build
fn build(a: &[String]) {
    let (gen, out, lib, arch) = (PathBuf::from(&a[0]), PathBuf::from(&a[1]), a[2].clone(), a[3].clone());
    std::fs::create_dir_all(&out).unwrap();
    let mut hs = vec![];
    for fam in ["mynnsfc", "ugwp"] {
        for (mode, def) in [("tl", "P4_MODE_TL"), ("rev", "P4_MODE_REV")] {
            let (gen, out, lib, arch) = (gen.clone(), out.clone(), lib.clone(), arch.clone());
            hs.push(std::thread::spawn(move || {
                let src = std::fs::read_to_string(gen.join(format!("{fam}.p4.cu"))).unwrap();
                let opts: Vec<String> = ["--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"]
                    .iter().map(|s| s.to_string()).chain([format!("-arch={arch}"), format!("-D{def}")]).collect();
                let t = std::time::Instant::now();
                let r = nvrtc::compile(&lib, &src, &format!("{fam}.p4.cu"), &opts);
                let secs = t.elapsed().as_secs_f64();
                match r {
                    Ok((cub, log)) => {
                        let f = out.join(format!("{fam}.{mode}.{arch}.cubin"));
                        std::fs::write(&f, &cub).unwrap();
                        json!({"family": fam, "mode": mode, "arch": arch, "options": opts, "seconds": secs, "cubin": f, "sha256": sha(&cub), "bytes": cub.len(), "source_sha256": sha(src.as_bytes()), "log": log})
                    }
                    Err(e) => json!({"family": fam, "mode": mode, "error": e, "seconds": secs}),
                }
            }));
        }
    }
    let r: Vec<Value> = hs.into_iter().map(|h| h.join().unwrap()).collect();
    let bad = r.iter().any(|x| x.get("error").is_some());
    std::fs::write(out.join(format!("build.{arch}.json")), serde_json::to_string_pretty(&r).unwrap()).unwrap();
    for x in &r {
        let errs = x.get("error").map(|e| { let l: Vec<&str> = e.as_str().unwrap().lines().collect();
            l.iter().enumerate().filter(|(_, s)| s.contains("error")).flat_map(|(i, _)| l[i..(i + 3).min(l.len())].to_vec()).take(60).collect::<Vec<_>>().join("\n") }).unwrap_or_default();
        eprintln!("{} {} {:.1}s {}", x["family"], x["mode"], x["seconds"].as_f64().unwrap(), errs);
    }
    if bad { std::process::exit(1); }
}

// ------------------------------------------------------------------------------------- MYNN-SFC
const SFC_FIELDS: [&str; 68] = ["u3d2", "v3d2", "dz8w2", "u3d", "v3d", "t3d", "qv3d", "p3d", "dz8w", "th3d", "rho3d", "sigmaf", "shdmax", "z0pert", "ztpert", "psfcpa", "chs", "chs2", "cqs", "cqs2", "cpm", "znt", "ust", "ustm", "pblh", "mavail", "zol", "mol", "rmol", "psim", "psih", "xland", "qgh", "hfx", "qfx", "lh", "tsk", "flhc", "flqc", "qsfc", "u10", "v10", "th2", "t2", "lakemask", "q2", "snowh", "gz1oz0", "wspd", "br", "dx", "ch", "ck", "cka", "cd", "cda", "stress", "hflx", "qflx", "cm", "fm", "fh", "fm10", "fh2", "tsurf", "water_depth", "xice", "pattern_spp_pbl"];
// categorical or geometry rows that are never perturbed: xland, lakemask, dx (from meshDensity)
const SFC_FIXED: [usize; 3] = [31, 44, 50];
const NF: usize = 68;
// bounded rows: a value sitting on its bound is not perturbed (a centred difference there straddles the bound)
const SFC_UNIT_INTERVAL: [usize; 4] = [11, 12, 25, 66];   // sigmaf, shdmax, mavail, xice
const SFC_NONNEG: [usize; 3] = [6, 46, 65];              // qv3d, snowh, water_depth
fn sfc_interior(j: usize, x: f32) -> bool {
    if SFC_UNIT_INTERVAL.contains(&j) { return x > 0.0 && x < 1.0; }
    if SFC_NONNEG.contains(&j) { return x > 0.0; }
    true
}

struct Sfc<'a> { c: &'a Cap, m: &'a Mods, img: &'a Module, dev: Dev, n: usize }
impl<'a> Sfc<'a> {
    fn head(&self, f: u64) -> Vec<KArg> {
        let c = self.c;
        vec![KArg::Ptr(f), KArg::Ptr(self.dev.ptr(c, "options")), KArg::Ptr(self.dev.ptr(c, "vegtype")), KArg::Bytes(c.scalar("xice_threshold")),
             KArg::Bytes(c.scalar("present")), KArg::Bytes(c.scalar("n")), KArg::Bytes(c.scalar("step_override")), KArg::Bytes(c.scalar("init_override"))]
    }
    fn orig(&self, x: &[f32], img: bool) -> (Vec<f32>, Vec<i32>) {
        self.dev.reset(self.c);
        self.dev.put_f32(self.c, "f", x);
        let m = if img { self.img } else { &self.m.tl };
        launch(m.func("mynnsfc_columns"), self.c.grid(), self.c.block(), 0, &self.dev.fwd_args(self.c));
        sync();
        (self.dev.get_f32(self.c, "f"), self.dev.get_i32(self.c, "errors"))
    }
    fn aug(&self, x: &[f32]) -> (Vec<f32>, Vec<i32>, Vec<u32>, Vec<i32>) {
        let n = self.n;
        self.dev.reset(self.c);
        let fin = dbuf_f32(x); let fout = DBuf::new(4 * NF * n); let e = DBuf::new(4 * n); let br = DBuf::new(4 * BR_WORDS * n); let bc = DBuf::new(4 * n);
        let mut a = vec![p(&fin), p(&fout), p(&e)];
        a.extend(self.head(fin.ptr).into_iter().skip(1));
        a.extend([p(&br), i32b(BR_WORDS as i32), p(&bc)]);
        launch(self.m.tl.func("mynnsfc_columns__aug"), self.c.grid(), self.c.block(), 0, &a);
        sync();
        (read_f32(&fout, NF * n), read_i32(&e, n), read_u32(&br, BR_WORDS * n), read_i32(&bc, n))
    }
    fn tl64(&self, x: &[f32], d: &[f64]) -> (Vec<f64>, i32) {
        let fin = dbuf_f32(x); let db = dbuf_f64(d); let s = DBuf::new(4);
        let mut a = self.head(fin.ptr); a.extend([p(&db), p(&s)]);
        launch(self.m.tl.func("mynnsfc_columns__tl64"), self.c.grid(), self.c.block(), 0, &a);
        sync();
        (read_f64(&db, d.len()), read_i32(&s, 1)[0])
    }
    fn tl32(&self, x: &[f32], d: &[f32]) -> Vec<f32> {
        let fin = dbuf_f32(x); let db = dbuf_f32(d); let s = DBuf::new(4);
        let mut a = self.head(fin.ptr); a.extend([p(&db), p(&s)]);
        launch(self.m.tl.func("mynnsfc_columns__tl"), self.c.grid(), self.c.block(), 0, &a);
        sync();
        read_f32(&db, d.len())
    }
    /// adjoint: count pass, then tape + sweep in column chunks that fit `budget` bytes.
    fn ad(&self, x: &[f32], b: &DBuf, wide: bool, budget: usize) -> (i32, usize) { let r = self.ad_t(x, b, wide, budget); (r.0, r.1) }
    /// returns (singular, tape nodes per column max, count-pass ms, tape+sweep ms) with kernel-only event timing
    fn ad_t(&self, x: &[f32], b: &DBuf, wide: bool, budget: usize) -> (i32, usize, f64, f64) {
        let n = self.n;
        let fin = dbuf_f32(x); let counts = DBuf::new(4 * n); let ov = DBuf::new(4); let s = DBuf::new(4);
        let name = if wide { "mynnsfc_columns__ad64" } else { "mynnsfc_columns__ad" };
        let mut a = self.head(fin.ptr);
        a.extend([p(b), null(), null(), null(), i32b(0), i64b(n as i64), i32b(0), i32b(n as i32), p(&counts), p(&ov), p(&s)]);
        let tm = Timer::new(); tm.start();
        launch(self.m.rev.func(name), grid1(n, 128), [128, 1, 1], 0, &a);
        let ms_count = tm.stop_ms();
        sync();
        let cap = *read_i32(&counts, n).iter().max().unwrap() as usize;
        let chunk = ((budget / (32 * cap.max(1))) / 128 * 128).clamp(128, n.div_ceil(128) * 128);
        let ab = DBuf::new(8 * cap * chunk); let w = DBuf::new(16 * cap * chunk); let adj = DBuf::new(8 * cap * chunk);
        let mut c0 = 0;
        tm.start();
        while c0 < n {
            let nc = chunk.min(n - c0);
            let mut a = self.head(fin.ptr);
            a.extend([p(b), p(&ab), p(&w), p(&adj), i32b(cap as i32), i64b(chunk as i64), i32b(c0 as i32), i32b(nc as i32), null(), p(&ov), null()]);
            launch(self.m.rev.func(name), grid1(nc, 128), [128, 1, 1], 0, &a);
            c0 += nc;
        }
        let ms_tape = tm.stop_ms();
        sync();
        assert_eq!(read_i32(&ov, 1)[0], 0, "tape overflow");
        (read_i32(&s, 1)[0], cap, ms_count, ms_tape)
    }
    fn ad64(&self, x: &[f32], ybar: &[f64]) -> Vec<f64> { let b = dbuf_f64(ybar); self.ad(x, &b, true, 2 << 30); read_f64(&b, ybar.len()) }
    fn ad32(&self, x: &[f32], ybar: &[f32]) -> Vec<f32> { let b = dbuf_f32(ybar); self.ad(x, &b, false, 2 << 30); read_f32(&b, ybar.len()) }
}

/// Field-scaled relative error over a set of columns: sqrt(sum ((tl-fd)/s_j)^2) / sqrt(sum (fd/s_j)^2).
fn scaled_err(tl: &[f64], fd: &[f64], scale: &[f64], nf: usize, n: usize, cols: &[usize]) -> (f64, f64, f64) {
    let (mut num, mut den) = (0.0, 0.0);
    for j in 0..nf {
        if !(scale[j] > 0.0) { continue; }
        for &c in cols {
            let (t, f) = (tl[j * n + c], fd[j * n + c]);
            if !t.is_finite() || !f.is_finite() { continue; }
            num += ((t - f) / scale[j]).powi(2); den += (f / scale[j]).powi(2);
        }
    }
    (if den > 0.0 { (num / den).sqrt() } else { f64::NAN }, num.sqrt(), den.sqrt())
}

fn sfc_unit(dir: &Path, m: &Mods, seeds: u64) -> Value {
    let c = Cap::load(dir);
    let img = Module::load(&c.image());
    tables(&img);
    let n = c.scalar_i32("n") as usize;
    let s = Sfc { c: &c, m, img: &img, dev: Dev::new(&c), n };
    let x0 = c.pre_f32("f");
    let post = c.post_f32("f");
    let post_e = c.post_i32("errors");
    // identity
    let (o_img, e_img) = s.orig(&x0, true);
    let (o_mod, e_mod) = s.orig(&x0, false);
    let (o_aug, e_aug, br0, bc0) = s.aug(&x0);
    let ident = json!({"words": NF * n, "captured_image_vs_post": ndiff(&o_img, &post) + idiff(&e_img, &post_e),
        "derivative_module_original_kernel_vs_post": ndiff(&o_mod, &post) + idiff(&e_mod, &post_e),
        "aug_vs_post": ndiff(&o_aug, &post), "aug_errors_vs_post": idiff(&e_aug, &post_e),
        "branch_records_per_column_max": bc0.iter().max(), "branch_records_truncated_columns": bc0.iter().filter(|&&k| k as usize > BR_WORDS * 32).count()});
    // classes: ice where xice >= threshold, else land (xland < 1.5) or water
    let thr = c.scalar_f32("xice_threshold");
    let (xl, xi) = (&x0[31 * n..32 * n], &x0[66 * n..67 * n]);
    let ice: Vec<usize> = (0..n).filter(|&i| xi[i] >= thr).collect();
    let land: Vec<usize> = (0..n).filter(|&i| xi[i] < thr && xl[i] < 1.5).collect();
    let water: Vec<usize> = (0..n).filter(|&i| xi[i] < thr && xl[i] >= 1.5).collect();
    let all: Vec<usize> = (0..n).collect();
    let sx = plane_std(&x0, NF, n);
    let sy = plane_std(&post, NF, n);
    let active = |j: usize| !SFC_FIXED.contains(&j) && sx[j] > 0.0;
    let dir_full = |seed: u64| -> Vec<f64> { let mut r = Rng::new(seed); let mut d = vec![0.0; NF * n]; for j in 0..NF { if active(j) { for i in 0..n { let v = r.n() * sx[j]; if sfc_interior(j, x0[j * n + i]) { d[j * n + i] = v; } } } } d };
    let ybar_full = |seed: u64| -> Vec<f64> { let mut r = Rng::new(seed ^ 0xABCD); let mut y = vec![0.0; NF * n]; for j in 0..NF { if sy[j] > 0.0 { for i in 0..n { y[j * n + i] = r.n() / sy[j]; } } } y };
    // dot tests
    let mut rows = vec![];
    let mut worst64 = 0f64; let mut worst32 = 0f64; let mut worst32ns = 0f64;
    let mut dot_case = |label: String, d: Vec<f64>, y: Vec<f64>, rows: &mut Vec<Value>| {
        let (jd, _) = s.tl64(&x0, &d); let jty = s.ad64(&x0, &y);
        let (l, r) = (dot(&jd, &y), dot(&d, &jty));
        let rel64 = (l - r).abs() / l.abs().max(r.abs()).max(1e-300);
        let d32: Vec<f32> = d.iter().map(|&v| v as f32).collect(); let y32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
        let jd32 = s.tl32(&x0, &d32); let jty32 = s.ad32(&x0, &y32);
        let jd32d: Vec<f64> = jd32.iter().map(|&v| v as f64).collect(); let jty32d: Vec<f64> = jty32.iter().map(|&v| v as f64).collect();
        let d32d: Vec<f64> = d32.iter().map(|&v| v as f64).collect(); let y32d: Vec<f64> = y32.iter().map(|&v| v as f64).collect();
        let (l3, r3) = (dot(&jd32d, &y32d), dot(&d32d, &jty32d));
        let rel32 = (l3 - r3).abs() / l3.abs().max(r3.abs()).max(1e-300);
        let ns32 = (l3 - r3).abs() / (nrm(&jd32d) * nrm(&y32d)).max(1e-300);
        if l != 0.0 || r != 0.0 { worst64 = worst64.max(rel64); worst32 = worst32.max(rel32); worst32ns = worst32ns.max(ns32); }
        rows.push(json!({"case": label, "f64": {"lhs": l, "rhs": r, "rel": rel64}, "f32": {"lhs": l3, "rhs": r3, "rel": rel32, "norm_scaled": ns32}}));
    };
    for sd in 1..=seeds { dot_case(format!("full seed {sd}"), dir_full(sd), ybar_full(sd), &mut rows); }
    for j in 0..NF { if active(j) { let mut d = dir_full(100 + j as u64); for k in 0..NF { if k != j { for i in 0..n { d[k * n + i] = 0.0; } } } dot_case(format!("field {}", SFC_FIELDS[j]), d, ybar_full(100 + j as u64), &mut rows); } }
    let (z, _) = s.tl64(&x0, &vec![0.0; NF * n]);
    let zt = z.iter().filter(|v| **v != 0.0).count();
    let za = s.ad64(&x0, &vec![0.0; NF * n]).iter().filter(|v| **v != 0.0).count();
    // determinism
    let y1 = ybar_full(1);
    let a1 = s.ad64(&x0, &y1); let a2 = s.ad64(&x0, &y1);
    let det = a1.iter().zip(&a2).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
    // Taylor: TL along the realized float direction vs centred FD of the captured image
    let mut taylor = vec![];
    for sd in [7u64, 8u64] {
        let d = dir_full(sd);
        let mut curve = vec![];
        for &eps in &EPS {
            let xp: Vec<f32> = x0.iter().zip(&d).map(|(&x, &v)| (x as f64 + eps * v) as f32).collect();
            let xm: Vec<f32> = x0.iter().zip(&d).map(|(&x, &v)| (x as f64 - eps * v) as f32).collect();
            let dr: Vec<f64> = xp.iter().zip(&xm).map(|(&a, &b)| (a as f64 - b as f64) / (2.0 * eps)).collect();
            let (fp, _) = s.orig(&xp, true); let (fm, _) = s.orig(&xm, true);
            let fd: Vec<f64> = fp.iter().zip(&fm).map(|(&a, &b)| (a as f64 - b as f64) / (2.0 * eps)).collect();
            let (tl, sing) = s.tl64(&x0, &dr);
            let (_, _, brp, bcp) = s.aug(&xp); let (_, _, brm, bcm) = s.aug(&xm);
            let mut fl = vec![0u64; n]; let mut trunc = 0;
            for i in 0..n {
                let (a, t1) = flips(&br0, bc0[i], &brp, bcp[i], n, i); let (b, t2) = flips(&br0, bc0[i], &brm, bcm[i], n, i);
                fl[i] = a + b; if t1 || t2 { trunc += 1; }
            }
            let stable: Vec<usize> = (0..n).filter(|&i| fl[i] == 0).collect();
            let cls = |cols: &[usize]| { let (e, _, _) = scaled_err(&tl, &fd, &sy, NF, n, cols); let f: Vec<f64> = cols.iter().map(|&i| fl[i] as f64).collect();
                json!({"columns": cols.len(), "rel": e, "flips_mean": f.iter().sum::<f64>() / f.len().max(1) as f64, "flips_max": f.iter().cloned().fold(0.0, f64::max), "columns_with_flips": f.iter().filter(|&&v| v > 0.0).count()}) };
            let mut perfield = vec![];
            for j in 0..NF { if sy[j] > 0.0 { let mut sc = vec![0.0; NF]; sc[j] = 1.0; let (e, _, den) = scaled_err(&tl, &fd, &sc, NF, n, &all); if den > 0.0 { perfield.push((SFC_FIELDS[j], e)); } } }
            perfield.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let mut pf_stable = vec![];
            for j in 0..NF { if sy[j] > 0.0 { let mut sc = vec![0.0; NF]; sc[j] = 1.0; let (e, _, den) = scaled_err(&tl, &fd, &sc, NF, n, &stable); if den > 0.0 { pf_stable.push((SFC_FIELDS[j], e)); } } }
            pf_stable.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            curve.push(json!({"eps": eps, "all": cls(&all), "land": cls(&land), "water": cls(&water), "ice": cls(&ice), "branch_stable_columns": cls(&stable),
                "worst_fields": perfield.iter().take(6).map(|(f, e)| json!([f, e])).collect::<Vec<_>>(), "worst_fields_branch_stable": pf_stable.iter().take(6).map(|(f, e)| json!([f, e])).collect::<Vec<_>>(), "singular_partials": sing, "flip_record_truncated_columns": trunc}));
        }
        let best = curve.iter().map(|r| r["all"]["rel"].as_f64().unwrap()).fold(f64::INFINITY, f64::min);
        taylor.push(json!({"seed": sd, "best_all": best, "curve": curve}));
    }
    // tangent budgets on a full direction: stress = ust^2, lh = c*qfx, qfx = flqc*(qsfc - qv)
    let d = dir_full(7);
    let (tl, _) = s.tl64(&x0, &d);
    let (yv, dv) = (&post, &tl);
    let g = |j: usize, i: usize| yv[j * n + i] as f64; let gd = |j: usize, i: usize| dv[j * n + i];
    let gx = |j: usize, i: usize| x0[j * n + i] as f64; let gdx = |j: usize, i: usize| d[j * n + i];
    let (mut b1, mut b2, mut b3) = (vec![], vec![], vec![]);
    for i in 0..n {
        let (ust, st) = (g(22, i), g(56, i));
        if st != 0.0 && ((ust * ust - st) / st).abs() < 1e-6 { let (a, b) = (gd(56, i), 2.0 * ust * gd(22, i)); b1.push((a - b).abs() / a.abs().max(b.abs()).max(1e-300)); }
        let (qfx, lh) = (g(34, i), g(35, i));
        if qfx != 0.0 && qfx > -0.0195 { let cc = lh / qfx; let (a, b) = (gd(35, i), cc * gd(34, i)); if a != 0.0 || b != 0.0 { b2.push((a - b).abs() / a.abs().max(b.abs())); } }
        let (flqc, qsfc, qv) = (g(38, i), g(39, i), gx(6, i));
        if qfx != 0.0 && qfx > -0.0195 && ((flqc * (qsfc - qv) - qfx) / qfx).abs() < 1e-6 {
            let a = gd(34, i); let b = gd(38, i) * (qsfc - qv) + flqc * (gd(39, i) - gdx(6, i));
            if a != 0.0 || b != 0.0 { b3.push((a - b).abs() / a.abs().max(b.abs())); }
        }
    }
    let summ = |v: &Vec<f64>| json!({"columns": v.len(), "max": v.iter().cloned().fold(0.0, f64::max), "median": median(v.clone())});
    // cost
    let fin = dbuf_f32(&x0);
    let t = Timer::new();
    let mut fwd = vec![]; let mut tlc = vec![]; let mut adc = vec![];
    let d32: Vec<f32> = d.iter().map(|&v| v as f32).collect(); let y32: Vec<f32> = y1.iter().map(|&v| v as f32).collect();
    let mut caps = 0;
    for _ in 0..10 {
        s.dev.reset(&c); t.start(); launch(img.func("mynnsfc_columns"), c.grid(), c.block(), 0, &s.dev.fwd_args(&c)); fwd.push(t.stop_ms());
        let db = dbuf_f32(&d32); let sg = DBuf::new(4); let mut a = s.head(fin.ptr); a.extend([p(&db), p(&sg)]);
        t.start(); launch(m.tl.func("mynnsfc_columns__tl"), c.grid(), c.block(), 0, &a); tlc.push(t.stop_ms());
        let b = dbuf_f32(&y32); let (_, cp, mc, mt) = s.ad_t(&x0, &b, false, 2 << 30); adc.push(mc + mt); caps = cp;
    }
    let (f_ms, t_ms, a_ms) = (median(fwd), median(tlc), median(adc));
    json!({"capture": dir, "kernel": "mynnsfc_columns", "columns": n, "classes": {"land": land.len(), "water": water.len(), "ice": ice.len()},
        "identity": ident,
        "dot": {"cases": rows.len(), "worst_f64_rel": worst64, "worst_f32_rel": worst32, "worst_f32_norm_scaled": worst32ns, "zero_tl_nonzero_words": zt, "zero_ad_nonzero_words": za, "rows": rows},
        "reverse_determinism_differing_words": det,
        "taylor": taylor,
        "budgets": {"stress_eq_ust2_note": "stress is not computed here (present mask has_stress off): 0 columns expected", "stress_eq_ust2": summ(&b1), "lh_eq_c_qfx": summ(&b2), "qfx_eq_flqc_dq": summ(&b3)},
        "cost_ms": {"forward_captured_image": f_ms, "tl_f32": t_ms, "ad_f32_count_pass_plus_tape_sweep_kernels": a_ms, "tl_over_forward": t_ms / f_ms, "ad_over_forward": a_ms / f_ms,
                    "tape_nodes_per_column_max": caps, "tape_bytes_per_column_max": caps * 32}})
}

// ----------------------------------------------------------------------- MYNN-SFC slot chain
/// from_mpas -> columns(packed) -> sea.pblh = packed.pblh -> columns(sea) -> to_mpas, composed.
fn sfc_chain(fdir: &Path, c0dir: &Path, c1dir: &Path, tdir: &Path, m: &Mods) -> Value {
    let (cf, cc0, cc1, ct) = (Cap::load(fdir), Cap::load(c0dir), Cap::load(c1dir), Cap::load(tdir));
    let img = Module::load(&cf.image()); tables(&img);
    let n = cf.scalar_i32("n") as usize; let nn = NF * n;
    let thr = cf.scalar_f32("threshold");
    let (f0, sea0, dens, sst0, ice0) = (cf.pre_f32("f"), cf.pre_f32("sea"), cf.pre_f32("density"), cf.pre_f32("sst"), cf.pre_f32("ice"));
    let opt = DBuf::new(56); opt.upload_at(0, &cc0.pre[cc0.alloc("options")][cc0.offset("options")..cc0.offset("options") + 56]);
    let veg = dbuf_f32(&f32s(&cc0.pre[cc0.alloc("vegtype")][cc0.offset("vegtype")..], n)); // int32 bytes, carried as raw words
    let errs = DBuf::new(4 * n);
    let colargs = |f: u64, cc: &Cap| -> Vec<KArg> { vec![KArg::Ptr(f), p(&opt), p(&veg), KArg::Bytes(cc.scalar("xice_threshold")), KArg::Bytes(cc.scalar("present")), KArg::Bytes(cc.scalar("n")), KArg::Bytes(cc.scalar("step_override")), KArg::Bytes(cc.scalar("init_override"))] };
    let pbl = 24usize;
    // forward chain with the captured image; returns all stage primals
    let fwd = |f: &[f32], sst: &[f32], ice: &[f32]| -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let (bf, bs, bd, bt, bi) = (dbuf_f32(f), dbuf_f32(&sea0), dbuf_f32(&dens), dbuf_f32(sst), dbuf_f32(ice));
        launch(img.func("mynnsfc_from_mpas"), cf.grid(), cf.block(), 0, &[p(&bf), p(&bs), p(&bd), p(&bt), p(&bi), KArg::Bytes(cf.scalar("len_disp")), KArg::Bytes(cf.scalar("threshold")), KArg::Bytes(cf.scalar("fraction")), KArg::Bytes(cf.scalar("n"))]);
        sync(); let (f1, s1) = (read_f32(&bf, nn), read_f32(&bs, nn));
        let mut a = colargs(bf.ptr, &cc0); a.insert(3, p(&errs)); launch(img.func("mynnsfc_columns"), cc0.grid(), cc0.block(), 0, &a);
        sync(); let f2 = read_f32(&bf, nn);
        let mut s2 = s1.clone(); s2[pbl * n..(pbl + 1) * n].copy_from_slice(&f2[pbl * n..(pbl + 1) * n]); bs.upload_at(0, &bytes_f32(&s2));
        let mut a = colargs(bs.ptr, &cc1); a.insert(3, p(&errs)); launch(img.func("mynnsfc_columns"), cc1.grid(), cc1.block(), 0, &a);
        sync(); let s3 = read_f32(&bs, nn);
        launch(img.func("mynnsfc_to_mpas"), ct.grid(), ct.block(), 0, &[p(&bf), p(&bs), p(&bi), KArg::Bytes(ct.scalar("threshold")), KArg::Bytes(ct.scalar("n"))]);
        sync(); (f1, s2, f2, s3, read_f32(&bf, nn), s1)
    };
    let (f1, s2, f2, s3, f4, _) = fwd(&f0, &sst0, &ice0);
    let ident = json!({"final_f_vs_captured_to_mpas_post": ndiff(&f4, &ct.post_f32("f")), "final_sea_vs_captured_columns_sea_post": ndiff(&s3, &cc1.post_f32("f")),
        "after_from_vs_capture": ndiff(&f1, &cf.post_f32("f")), "after_columns_packed_vs_capture": ndiff(&f2, &cc0.post_f32("f"))});
    // TL chain (f64)
    let tl = |df: &[f64], dsst: &[f64], dice: &[f64]| -> (Vec<f64>, Vec<f64>) {
        let (bdf, bds, bdt, bdi) = (dbuf_f64(df), DBuf::new(8 * nn), dbuf_f64(dsst), dbuf_f64(dice));
        let (bf, bs, bd, bt, bi) = (dbuf_f32(&f0), dbuf_f32(&sea0), dbuf_f32(&dens), dbuf_f32(&sst0), dbuf_f32(&ice0));
        launch(m.tl.func("mynnsfc_from_mpas__tl64"), cf.grid(), cf.block(), 0, &[p(&bf), p(&bs), p(&bd), p(&bt), p(&bi), KArg::Bytes(cf.scalar("len_disp")), KArg::Bytes(cf.scalar("threshold")), KArg::Bytes(cf.scalar("fraction")), KArg::Bytes(cf.scalar("n")), p(&bdf), p(&bds), p(&bdt)]);
        let x1 = dbuf_f32(&f1); let mut a = colargs(x1.ptr, &cc0); a.extend([p(&bdf), null()]); launch(m.tl.func("mynnsfc_columns__tl64"), cc0.grid(), cc0.block(), 0, &a);
        sync();
        let (mut vdf, mut vds) = (read_f64(&bdf, nn), read_f64(&bds, nn));
        for i in 0..n { vds[pbl * n + i] = vdf[pbl * n + i]; } bds.upload_at(0, &bytes_f64(&vds));
        let x2 = dbuf_f32(&s2); let mut a = colargs(x2.ptr, &cc1); a.extend([p(&bds), null()]); launch(m.tl.func("mynnsfc_columns__tl64"), cc1.grid(), cc1.block(), 0, &a);
        let (pf, ps) = (dbuf_f32(&f2), dbuf_f32(&s3));
        launch(m.tl.func("mynnsfc_to_mpas__tl64"), ct.grid(), ct.block(), 0, &[p(&pf), p(&ps), p(&bi), KArg::Bytes(ct.scalar("threshold")), KArg::Bytes(ct.scalar("n")), p(&bdf), p(&bds), p(&bdi)]);
        sync(); vdf = read_f64(&bdf, nn); vds = read_f64(&bds, nn); (vdf, vds)
    };
    // AD chain (f64)
    let ad = |yf: &[f64], ys: &[f64]| -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (bf_, bs_, bt_, bi_) = (dbuf_f64(yf), dbuf_f64(ys), DBuf::new(8 * n), DBuf::new(8 * n));
        let (pf, ps, pi, pt) = (dbuf_f32(&f2), dbuf_f32(&s3), dbuf_f32(&ice0), dbuf_f32(&sst0));
        launch(m.rev.func("mynnsfc_to_mpas__ad64"), ct.grid(), ct.block(), 0, &[p(&pf), p(&ps), p(&pi), KArg::Bytes(ct.scalar("threshold")), KArg::Bytes(ct.scalar("n")), p(&bf_), p(&bs_), p(&bi_)]);
        sync();
        let s_ = Sfc { c: &cc1, m, img: &img, dev: Dev::new(&cc1), n }; s_.ad(&s2, &bs_, true, 2 << 30);
        let mut vs = read_f64(&bs_, nn); let mut vf = read_f64(&bf_, nn);
        for i in 0..n { vf[pbl * n + i] += vs[pbl * n + i]; vs[pbl * n + i] = 0.0; }
        bf_.upload_at(0, &bytes_f64(&vf)); bs_.upload_at(0, &bytes_f64(&vs));
        let s0 = Sfc { c: &cc0, m, img: &img, dev: Dev::new(&cc0), n }; s0.ad(&f1, &bf_, true, 2 << 30);
        let (bfx, bsx, bd) = (dbuf_f32(&f0), dbuf_f32(&sea0), dbuf_f32(&dens));
        launch(m.rev.func("mynnsfc_from_mpas__ad64"), cf.grid(), cf.block(), 0, &[p(&bfx), p(&bsx), p(&bd), p(&pt), p(&pi), KArg::Bytes(cf.scalar("len_disp")), KArg::Bytes(cf.scalar("threshold")), KArg::Bytes(cf.scalar("fraction")), KArg::Bytes(cf.scalar("n")), p(&bf_), p(&bs_), p(&bt_)]);
        sync(); (read_f64(&bf_, nn), read_f64(&bt_, n), read_f64(&bi_, n))
    };
    let sx = plane_std(&f0, NF, n); let sy = plane_std(&f4, NF, n); let sys = plane_std(&s3, NF, n);
    let ssst = plane_std(&sst0, 1, n)[0]; let sice = plane_std(&ice0, 1, n)[0];
    let mk = |seed: u64| { let mut r = Rng::new(seed); let mut d = vec![0.0; nn]; for j in 0..NF { if !SFC_FIXED.contains(&j) && sx[j] > 0.0 { for i in 0..n { let v = r.n() * sx[j]; if sfc_interior(j, f0[j * n + i]) { d[j * n + i] = v; } } } }
        let dt: Vec<f64> = (0..n).map(|_| r.n() * ssst).collect(); let di: Vec<f64> = (0..n).map(|i| { let v = r.n() * sice; if ice0[i] > 0.0 && ice0[i] < 1.0 { v } else { 0.0 } }).collect(); (d, dt, di) };
    let mut dots = vec![];
    for sd in 1..=3u64 {
        let (d, dt, di) = mk(sd);
        let mut r = Rng::new(sd + 50);
        let yf: Vec<f64> = (0..nn).map(|k| if sy[k / n] > 0.0 { r.n() / sy[k / n] } else { 0.0 }).collect();
        let ys: Vec<f64> = (0..nn).map(|k| if sys[k / n] > 0.0 { r.n() / sys[k / n] } else { 0.0 }).collect();
        let (jf, js) = tl(&d, &dt, &di); let (bf, bt, bi) = ad(&yf, &ys);
        let l = dot(&jf, &yf) + dot(&js, &ys); let rr = dot(&d, &bf) + dot(&dt, &bt) + dot(&di, &bi);
        dots.push(json!({"seed": sd, "lhs": l, "rhs": rr, "rel": (l - rr).abs() / l.abs().max(rr.abs())}));
    }
    // variants: continuous state only (no xice row, no sst, no ice), + sst, + ice and the xice row
    let mut variants = vec![];
    for (label, use_sst, use_ice) in [("f rows except xice", false, false), ("f rows except xice, + sst", true, false), ("f rows incl. xice, + sst, + ice", true, true)] {
        let (mut d, mut dt, mut di) = mk(7);
        if !use_ice { for i in 0..n { d[66 * n + i] = 0.0; } di.iter_mut().for_each(|v| *v = 0.0); }
        if !use_sst { dt.iter_mut().for_each(|v| *v = 0.0); }
        let mut curve = vec![];
        for &eps in &EPS {
            let pert = |x: &[f32], v: &[f64], sg: f64| -> Vec<f32> { x.iter().zip(v).map(|(&a, &b)| (a as f64 + sg * eps * b) as f32).collect() };
            let (fp, tp, ip) = (pert(&f0, &d, 1.0), pert(&sst0, &dt, 1.0), pert(&ice0, &di, 1.0));
            let (fm, tm, im) = (pert(&f0, &d, -1.0), pert(&sst0, &dt, -1.0), pert(&ice0, &di, -1.0));
            let real = |a: &[f32], b: &[f32]| -> Vec<f64> { a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64) / (2.0 * eps)).collect() };
            let (op, om) = (fwd(&fp, &tp, &ip), fwd(&fm, &tm, &im));
            let fdf = real(&op.4, &om.4); let fds = real(&op.3, &om.3);
            let (jf, js) = tl(&real(&fp, &fm), &real(&tp, &tm), &real(&ip, &im));
            let all: Vec<usize> = (0..n).collect();
            let (_, nu1, de1) = scaled_err(&jf, &fdf, &sy, NF, n, &all); let (_, nu2, de2) = scaled_err(&js, &fds, &sys, NF, n, &all);
            // columns ranked by their share of the error (the switch crossings show up here)
            let mut colerr: Vec<(usize, f64)> = (0..n).map(|i| { let (_, a, _) = scaled_err(&jf, &fdf, &sy, NF, n, &[i]); let (_, b, _) = scaled_err(&js, &fds, &sys, NF, n, &[i]); (i, a * a + b * b) }).collect();
            colerr.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let tot: f64 = colerr.iter().map(|x| x.1).sum();
            let top10: f64 = colerr.iter().take(10).map(|x| x.1).sum();
            let keep: Vec<usize> = colerr.iter().skip(10).map(|x| x.0).collect();
            let (_, k1, kd1) = scaled_err(&jf, &fdf, &sy, NF, n, &keep); let (_, k2, kd2) = scaled_err(&js, &fds, &sys, NF, n, &keep);
            curve.push(json!({"eps": eps, "rel": ((nu1 * nu1 + nu2 * nu2) / (de1 * de1 + de2 * de2)).sqrt(), "share_of_error_in_worst_10_columns": top10 / tot.max(1e-300),
                              "rel_without_worst_10_columns": ((k1 * k1 + k2 * k2) / (kd1 * kd1 + kd2 * kd2)).sqrt(), "worst_columns": colerr.iter().take(5).map(|x| x.0).collect::<Vec<_>>()}));
        }
        let best = curve.iter().map(|r| r["rel"].as_f64().unwrap()).fold(f64::INFINITY, f64::min);
        variants.push(json!({"direction": label, "best": best, "curve": curve}));
    }
    json!({"chain": [fdir, c0dir, c1dir, tdir], "replay_identity_differing_words": ident, "dot_f64": dots, "taylor": variants})
}

// ------------------------------------------------------------------------------------------ GSL
const NZ: usize = 59;
const SURF_STATE: [usize; 2] = [2, 3];                                      // hpbl, br1
const SURF_PARAM: [usize; 11] = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];     // var, oc1, oa x4, ol x4, varss

struct Gsl<'a> { c: &'a Cap, m: &'a Mods, img: &'a Module, dev: Dev, nc: usize }
impl<'a> Gsl<'a> {
    fn put(&self, l: &[f32], pr: &[f32], s: &[f32]) { self.dev.reset(self.c); self.dev.put_f32(self.c, "layers", l); self.dev.put_f32(self.c, "prsi0", pr); self.dev.put_f32(self.c, "surface", s); }
    fn orig(&self, l: &[f32], pr: &[f32], s: &[f32], img: bool) -> (Vec<f32>, Vec<f32>, Vec<i32>) {
        self.put(l, pr, s);
        let m = if img { self.img } else { &self.m.tl };
        launch(m.func("ugwp_oro_run"), self.c.grid(), self.c.block(), 0, &self.dev.fwd_args(self.c)); sync();
        (self.dev.get_f32(self.c, "out"), self.dev.get_f32(self.c, "sout"), self.dev.get_i32(self.c, "status"))
    }
    fn aug(&self, l: &[f32], pr: &[f32], s: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<u32>, Vec<i32>, i32) {
        self.put(l, pr, s);
        let nc = self.nc; let br = DBuf::new(4 * BR_WORDS * nc); let bc = DBuf::new(4 * nc); let rbw = DBuf::new(4);
        let mut a = self.dev.fwd_args(self.c); a.extend([p(&br), i32b(BR_WORDS as i32), p(&bc), p(&rbw)]);
        launch(self.m.tl.func("ugwp_oro_run__aug"), self.c.grid(), self.c.block(), 0, &a); sync();
        (self.dev.get_f32(self.c, "out"), self.dev.get_f32(self.c, "sout"), read_u32(&br, BR_WORDS * nc), read_i32(&bc, nc), read_i32(&rbw, 1)[0])
    }
    fn tl64(&self, d: &(Vec<f64>, Vec<f64>, Vec<f64>)) -> (Vec<f64>, Vec<f64>, i32) {
        self.dev.reset(self.c);
        let (no, ns) = (self.c.count("out"), self.c.count("sout"));
        let (dl, dp, ds, dout, dso, st, sg, rb) = (dbuf_f64(&d.0), dbuf_f64(&d.1), dbuf_f64(&d.2), DBuf::new(8 * no), DBuf::new(8 * ns), DBuf::new(4), DBuf::new(4), DBuf::new(4));
        let mut a = self.dev.fwd_args(self.c); a.extend([p(&dl), p(&dp), p(&ds), p(&dout), p(&dso), p(&st), p(&sg), p(&rb)]);
        launch(self.m.tl.func("ugwp_oro_run__tl64"), self.c.grid(), self.c.block(), 0, &a); sync();
        (read_f64(&dout, no), read_f64(&dso, ns), read_i32(&sg, 1)[0])
    }
    fn tl32(&self, d: &(Vec<f32>, Vec<f32>, Vec<f32>)) -> (Vec<f32>, Vec<f32>) {
        self.dev.reset(self.c);
        let (no, ns) = (self.c.count("out"), self.c.count("sout"));
        let (dl, dp, ds, dout, dso, st, sg, rb) = (dbuf_f32(&d.0), dbuf_f32(&d.1), dbuf_f32(&d.2), DBuf::new(4 * no), DBuf::new(4 * ns), DBuf::new(4), DBuf::new(4), DBuf::new(4));
        let mut a = self.dev.fwd_args(self.c); a.extend([p(&dl), p(&dp), p(&ds), p(&dout), p(&dso), p(&st), p(&sg), p(&rb)]);
        launch(self.m.tl.func("ugwp_oro_run__tl"), self.c.grid(), self.c.block(), 0, &a); sync();
        (read_f32(&dout, no), read_f32(&dso, ns))
    }
    /// adjoint: b_layers/b_prsi0/b_surface accumulate (start at 0); b_out/b_sout are consumed.
    fn ad(&self, bo: &DBuf, bs: &DBuf, bl: &DBuf, bp: &DBuf, bsf: &DBuf, wide: bool) -> usize { self.ad_t(bo, bs, bl, bp, bsf, wide).0 }
    fn ad_t(&self, bo: &DBuf, bs: &DBuf, bl: &DBuf, bp: &DBuf, bsf: &DBuf, wide: bool) -> (usize, f64, f64) {
        self.dev.reset(self.c);
        let nc = self.nc;
        let counts = DBuf::new(4 * nc); let ov = DBuf::new(4); let sg = DBuf::new(4); let st = DBuf::new(4);
        let name = if wide { "ugwp_oro_run__ad64" } else { "ugwp_oro_run__ad" };
        let mk = |ab: KArg, w: KArg, adj: KArg, cap: i32, stride: i64, c0: i32, ncol: i32, cnt: KArg| {
            let mut a = self.dev.fwd_args(self.c);
            a.extend([p(bl), p(bp), p(bsf), p(bo), p(bs), ab, w, adj, i32b(cap), i64b(stride), i32b(c0), i32b(ncol), cnt, p(&ov), p(&sg), p(&st)]); a };
        let tm = Timer::new(); tm.start();
        launch(self.m.rev.func(name), grid1(nc, 64), [64, 1, 1], 0, &mk(null(), null(), null(), 0, nc as i64, 0, nc as i32, p(&counts)));
        let ms_count = tm.stop_ms(); sync();
        let cap = *read_i32(&counts, nc).iter().max().unwrap() as usize;
        let chunk = (((2usize << 30) / (32 * cap.max(1))) / 64 * 64).clamp(64, nc.div_ceil(64) * 64);
        let (ab, w, adj) = (DBuf::new(8 * cap * chunk), DBuf::new(16 * cap * chunk), DBuf::new(8 * cap * chunk));
        let mut c0 = 0;
        tm.start();
        while c0 < nc { let k = chunk.min(nc - c0); launch(self.m.rev.func(name), grid1(k, 64), [64, 1, 1], 0, &mk(p(&ab), p(&w), p(&adj), cap as i32, chunk as i64, c0 as i32, k as i32, null())); c0 += k; }
        let ms_tape = tm.stop_ms();
        sync();
        assert_eq!(read_i32(&ov, 1)[0], 0, "tape overflow");
        (cap, ms_count, ms_tape)
    }
    fn tl32_ms(&self, d: &(Vec<f32>, Vec<f32>, Vec<f32>)) -> f64 {
        self.dev.reset(self.c);
        let (no, ns) = (self.c.count("out"), self.c.count("sout"));
        let (dl, dp, ds, dout, dso, st, sg, rb) = (dbuf_f32(&d.0), dbuf_f32(&d.1), dbuf_f32(&d.2), DBuf::new(4 * no), DBuf::new(4 * ns), DBuf::new(4), DBuf::new(4), DBuf::new(4));
        let mut a = self.dev.fwd_args(self.c); a.extend([p(&dl), p(&dp), p(&ds), p(&dout), p(&dso), p(&st), p(&sg), p(&rb)]);
        let tm = Timer::new(); tm.start();
        launch(self.m.tl.func("ugwp_oro_run__tl"), self.c.grid(), self.c.block(), 0, &a);
        let ms = tm.stop_ms(); sync(); ms
    }
    fn ad64(&self, yo: &[f64], ys: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (nl, np, nsf) = (self.c.count("layers"), self.c.count("prsi0"), self.c.count("surface"));
        let (bo, bs, bl, bp, bsf) = (dbuf_f64(yo), dbuf_f64(ys), DBuf::new(8 * nl), DBuf::new(8 * np), DBuf::new(8 * nsf));
        self.ad(&bo, &bs, &bl, &bp, &bsf, true);
        (read_f64(&bl, nl), read_f64(&bp, np), read_f64(&bsf, nsf), read_f64(&bo, yo.len()), read_f64(&bs, ys.len()))
    }
    fn ad32(&self, yo: &[f32], ys: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (nl, np, nsf) = (self.c.count("layers"), self.c.count("prsi0"), self.c.count("surface"));
        let (bo, bs, bl, bp, bsf) = (dbuf_f32(yo), dbuf_f32(ys), DBuf::new(4 * nl), DBuf::new(4 * np), DBuf::new(4 * nsf));
        self.ad(&bo, &bs, &bl, &bp, &bsf, false);
        (read_f32(&bl, nl), read_f32(&bp, np), read_f32(&bsf, nsf))
    }
}

fn gsl_unit(dir: &Path, m: &Mods, seeds: u64) -> Value {
    let c = Cap::load(dir);
    let img = Module::load(&c.image());
    let nc = c.scalar_i32("nc") as usize;
    assert_eq!(c.scalar_i32("nz") as usize, NZ);
    let g = Gsl { c: &c, m, img: &img, dev: Dev::new(&c), nc };
    let (l0, p0, s0) = (c.pre_f32("layers"), c.pre_f32("prsi0"), c.pre_f32("surface"));
    let (po, ps, pst) = (c.post_f32("out"), c.post_f32("sout"), c.post_i32("status"));
    let (oi, si, sti) = g.orig(&l0, &p0, &s0, true);
    let (om, sm, stm) = g.orig(&l0, &p0, &s0, false);
    let (oa, sa, br0, bc0, rbw) = g.aug(&l0, &p0, &s0);
    let ident = json!({"words": po.len() + ps.len(), "captured_image_vs_post": ndiff(&oi, &po) + ndiff(&si, &ps) + idiff(&sti, &pst),
        "derivative_module_original_kernel_vs_post": ndiff(&om, &po) + ndiff(&sm, &ps) + idiff(&stm, &pst),
        "aug_vs_post": ndiff(&oa, &po) + ndiff(&sa, &ps), "aug_read_before_write": rbw,
        "branch_records_per_column_max": bc0.iter().max(), "branch_records_truncated_columns": bc0.iter().filter(|&&k| k as usize > BR_WORDS * 32).count()});
    // high-terrain land columns: zs = zl(1) - dz(1)/2 >= 1000 m, var > 0, xland < 1.5
    let n3 = NZ * nc;
    let zs = |i: usize| l0[6 * n3 + i] - 0.5 * l0[7 * n3 + i];
    let high: Vec<usize> = (0..nc).filter(|&i| s0[4 * nc + i] < 1.5 && zs(i) >= 1000.0 && s0[6 * nc + i] > 0.0).collect();
    let all: Vec<usize> = (0..nc).collect();
    // level spread: std over columns at each (field, level); zl (6) and dz (7) are fixed geometry
    let lstd: Vec<f64> = plane_std(&l0, 10 * NZ, nc).into_iter().enumerate().map(|(q, v)| if q / NZ == 6 || q / NZ == 7 { 0.0 } else { v }).collect();
    let pstd: Vec<f64> = plane_std(&p0, NZ + 1, nc);
    let sstd = plane_std(&s0, 26, nc);
    let ostd = plane_std(&po, 12, n3); let sostd = plane_std(&ps, 10, nc);
    let mkdir = |seed: u64, params: bool| -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut r = Rng::new(seed);
        let dl: Vec<f64> = (0..l0.len()).map(|k| r.n() * lstd[k / nc]).collect();
        let dp: Vec<f64> = (0..p0.len()).map(|k| r.n() * pstd[k / nc]).collect();
        let mut ds = vec![0.0; s0.len()];
        for &row in SURF_STATE.iter().chain(if params { SURF_PARAM.iter() } else { [].iter() }) { for i in 0..nc { ds[row * nc + i] = r.n() * sstd[row]; } }
        (dl, dp, ds) };
    let mkybar = |seed: u64| -> (Vec<f64>, Vec<f64>) { let mut r = Rng::new(seed ^ 0x5151);
        ((0..po.len()).map(|k| if ostd[k / n3] > 0.0 { r.n() / ostd[k / n3] } else { 0.0 }).collect(), (0..ps.len()).map(|k| if sostd[k / nc] > 0.0 { r.n() / sostd[k / nc] } else { 0.0 }).collect()) };
    // dot
    let mut rows = vec![]; let (mut w64, mut w32, mut w32ns) = (0f64, 0f64, 0f64);
    let mut case = |label: String, d: (Vec<f64>, Vec<f64>, Vec<f64>), y: (Vec<f64>, Vec<f64>), rows: &mut Vec<Value>| {
        let (jo, js, _) = g.tl64(&d); let (bl, bp, bs, bo_after, bso_after) = g.ad64(&y.0, &y.1);
        let l = dot(&jo, &y.0) + dot(&js, &y.1); let r = dot(&d.0, &bl) + dot(&d.1, &bp) + dot(&d.2, &bs);
        let rel = (l - r).abs() / l.abs().max(r.abs()).max(1e-300);
        let consumed = bo_after.iter().chain(bso_after.iter()).filter(|v| **v != 0.0).count();
        let f = |v: &Vec<f64>| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
        let d32 = (f(&d.0), f(&d.1), f(&d.2)); let y32 = (f(&y.0), f(&y.1));
        let (jo3, js3) = g.tl32(&d32); let (bl3, bp3, bs3) = g.ad32(&y32.0, &y32.1);
        let up = |v: &[f32]| -> Vec<f64> { v.iter().map(|&x| x as f64).collect() };
        let l3 = dot(&up(&jo3), &up(&y32.0)) + dot(&up(&js3), &up(&y32.1));
        let r3 = dot(&up(&d32.0), &up(&bl3)) + dot(&up(&d32.1), &up(&bp3)) + dot(&up(&d32.2), &up(&bs3));
        let rel3 = (l3 - r3).abs() / l3.abs().max(r3.abs()).max(1e-300);
        let ns3 = (l3 - r3).abs() / ((dot(&up(&jo3), &up(&jo3)) + dot(&up(&js3), &up(&js3))).sqrt() * (dot(&up(&y32.0), &up(&y32.0)) + dot(&up(&y32.1), &up(&y32.1))).sqrt()).max(1e-300);
        if l != 0.0 || r != 0.0 { w64 = w64.max(rel); w32 = w32.max(rel3); w32ns = w32ns.max(ns3); }
        rows.push(json!({"case": label, "f64": {"lhs": l, "rhs": r, "rel": rel}, "f32": {"lhs": l3, "rhs": r3, "rel": rel3, "norm_scaled": ns3}, "b_out_nonzero_after": consumed}));
    };
    for sd in 1..=seeds { case(format!("state seed {sd}"), mkdir(sd, false), mkybar(sd), &mut rows); }
    for sd in 1..=2 { case(format!("state+terrain-params seed {sd}"), mkdir(10 + sd, true), mkybar(10 + sd), &mut rows); }
    for f in 0..10 { let mut rr = Rng::new(200 + f as u64); let sf = plane_std(&l0[f * n3..(f + 1) * n3], NZ, nc); let dl: Vec<f64> = (0..l0.len()).map(|k| if k / n3 == f { rr.n() * sf[(k % n3) / nc] } else { 0.0 }).collect();
        case(format!("layer field {f}"), (dl, vec![0.0; p0.len()], vec![0.0; s0.len()]), mkybar(200 + f as u64), &mut rows); }
    { let (_, dp, _) = mkdir(300, false); case("prsi0".into(), (vec![0.0; l0.len()], dp, vec![0.0; s0.len()]), mkybar(300), &mut rows); }
    let (zo, zs_, _) = g.tl64(&(vec![0.0; l0.len()], vec![0.0; p0.len()], vec![0.0; s0.len()]));
    let zt = zo.iter().chain(zs_.iter()).filter(|v| **v != 0.0).count();
    let (zb1, zb2, zb3, _, _) = g.ad64(&vec![0.0; po.len()], &vec![0.0; ps.len()]);
    let za = zb1.iter().chain(zb2.iter()).chain(zb3.iter()).filter(|v| **v != 0.0).count();
    let y1 = mkybar(1);
    let a1 = g.ad64(&y1.0, &y1.1); let a2 = g.ad64(&y1.0, &y1.1);
    let det = [(&a1.0, &a2.0), (&a1.1, &a2.1), (&a1.2, &a2.2)].iter().map(|(x, y)| x.iter().zip(y.iter()).filter(|(p, q)| p.to_bits() != q.to_bits()).count()).sum::<usize>();
    // Taylor
    let mut taylor = vec![];
    for (sd, params) in [(7u64, false), (8u64, false), (9u64, true)] {
        let d = mkdir(sd, params);
        let mut curve = vec![];
        for &eps in &EPS {
            let pe = |x: &[f32], v: &[f64], sg: f64| -> Vec<f32> { x.iter().zip(v).map(|(&a, &b)| (a as f64 + sg * eps * b) as f32).collect() };
            let real = |a: &[f32], b: &[f32]| -> Vec<f64> { a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64) / (2.0 * eps)).collect() };
            let (lp, pp, sp) = (pe(&l0, &d.0, 1.0), pe(&p0, &d.1, 1.0), pe(&s0, &d.2, 1.0));
            let (lm, pm, sm2) = (pe(&l0, &d.0, -1.0), pe(&p0, &d.1, -1.0), pe(&s0, &d.2, -1.0));
            let (op, sop, _) = g.orig(&lp, &pp, &sp, true); let (om2, som, _) = g.orig(&lm, &pm, &sm2, true);
            let fdo = real(&op, &om2); let fds = real(&sop, &som);
            let (tlo, tls, sing) = g.tl64(&(real(&lp, &lm), real(&pp, &pm), real(&sp, &sm2)));
            let (_, _, brp, bcp, _) = g.aug(&lp, &pp, &sp); let (_, _, brm, bcm, _) = g.aug(&lm, &pm, &sm2);
            let fl: Vec<u64> = (0..nc).map(|i| flips(&br0, bc0[i], &brp, bcp[i], nc, i).0 + flips(&br0, bc0[i], &brm, bcm[i], nc, i).0).collect();
            let stable: Vec<usize> = (0..nc).filter(|&i| fl[i] == 0).collect();
            let cls = |cols: &[usize]| {
                // out is [12][nz][nc]: treat (m,k) as the field, i as the column
                let (mut num, mut den) = (0.0, 0.0);
                for mm in 0..12 { if !(ostd[mm] > 0.0) { continue; } for k in 0..NZ { for &i in cols { let q = mm * n3 + k * nc + i; let (t, f) = (tlo[q], fdo[q]); if t.is_finite() && f.is_finite() { num += ((t - f) / ostd[mm]).powi(2); den += (f / ostd[mm]).powi(2); } } } }
                for mm in 0..10 { if !(sostd[mm] > 0.0) { continue; } for &i in cols { let q = mm * nc + i; let (t, f) = (tls[q], fds[q]); if t.is_finite() && f.is_finite() { num += ((t - f) / sostd[mm]).powi(2); den += (f / sostd[mm]).powi(2); } } }
                let fv: Vec<f64> = cols.iter().map(|&i| fl[i] as f64).collect();
                json!({"columns": cols.len(), "rel": if den > 0.0 { (num / den).sqrt() } else { f64::NAN }, "flips_mean": fv.iter().sum::<f64>() / fv.len().max(1) as f64, "flips_max": fv.iter().cloned().fold(0.0, f64::max), "columns_with_flips": fv.iter().filter(|&&v| v > 0.0).count()}) };
            curve.push(json!({"eps": eps, "all": cls(&all), "high_terrain": cls(&high), "branch_stable_columns": cls(&stable), "singular_partials": sing}));
        }
        let best = curve.iter().map(|r| r["all"]["rel"].as_f64().unwrap()).fold(f64::INFINITY, f64::min);
        taylor.push(json!({"seed": sd, "terrain_params_perturbed": params, "best_all": best, "curve": curve}));
    }
    // budget: d(sout_m) = -1/g sum_k (d out(2+m,k) del_k + out(2+m,k) d del_k), m = 0 (u), 1 (v)
    let d = mkdir(7, false);
    let (tlo, tls, _) = g.tl64(&d);
    let gg = c.scalar_f32("g") as f64;
    let mut res = vec![]; let mut checked = 0;
    for i in 0..nc { for mm in 0..2 {
        let mut s = 0.0; let mut mag = 0.0;
        for k in 0..NZ {
            let del = p0[k * nc + i] as f64 - p0[(k + 1) * nc + i] as f64; let ddel = d.1[k * nc + i] - d.1[(k + 1) * nc + i];
            let q = (2 + mm) * n3 + k * nc + i; let t = tlo[q] * del + po[q] as f64 * ddel; s += t; mag += t.abs();
        }
        let lhs = tls[mm * nc + i]; let rhs = -s / gg;
        if lhs != 0.0 || rhs != 0.0 { checked += 1; res.push((lhs - rhs).abs() / (mag / gg).max(lhs.abs()).max(1e-300)); }
    } }
    // cost
    let t = Timer::new(); let (mut fw, mut tc, mut ac) = (vec![], vec![], vec![]); let mut capn = 0;
    let d32 = (d.0.iter().map(|&v| v as f32).collect::<Vec<_>>(), d.1.iter().map(|&v| v as f32).collect::<Vec<_>>(), d.2.iter().map(|&v| v as f32).collect::<Vec<_>>());
    let y32 = (y1.0.iter().map(|&v| v as f32).collect::<Vec<_>>(), y1.1.iter().map(|&v| v as f32).collect::<Vec<_>>());
    for _ in 0..10 {
        g.dev.reset(&c); t.start(); launch(img.func("ugwp_oro_run"), c.grid(), c.block(), 0, &g.dev.fwd_args(&c)); fw.push(t.stop_ms());
        tc.push(g.tl32_ms(&d32));
        let (no, ns) = (dbuf_f32(&y32.0), dbuf_f32(&y32.1)); let (bl, bp, bs) = (DBuf::new(4 * l0.len()), DBuf::new(4 * p0.len()), DBuf::new(4 * s0.len()));
        let (cp, mc, mt) = g.ad_t(&no, &ns, &bl, &bp, &bs, false); capn = cp; ac.push(mc + mt);
    }
    let (f_ms, t_ms, a_ms) = (median(fw), median(tc), median(ac));
    json!({"capture": dir, "kernel": "ugwp_oro_run", "columns": nc, "high_terrain_columns": high.len(), "identity": ident,
        "dot": {"cases": rows.len(), "worst_f64_rel": w64, "worst_f32_rel": w32, "worst_f32_norm_scaled": w32ns, "zero_tl_nonzero_words": zt, "zero_ad_nonzero_words": za, "rows": rows},
        "reverse_determinism_differing_words": det, "taylor": taylor,
        "budget_surface_stress_vs_column_tendency": {"column_components_checked": checked, "max": res.iter().cloned().fold(0.0, f64::max), "median": median(res.clone())},
        "cost_ms": {"forward_captured_image": f_ms, "tl_f32": t_ms, "ad_f32_count_pass_plus_tape_sweep_kernels": a_ms, "tl_over_forward": t_ms / f_ms, "ad_over_forward": a_ms / f_ms, "tape_nodes_per_column_max": capn, "tape_bytes_per_column_max": capn * 32}})
}

/// ugwp_oro_dx / ugwp_oro_diff: identity of the original kernel in the derivative module, TL and AD
/// (dot) and FD of the captured image; both are one-line linear/inactive maps.
fn gsl_small(dir: &Path, m: &Mods) -> Value {
    let c = Cap::load(dir);
    let img = Module::load(&c.image());
    let k = c.m["kernel"].as_str().unwrap().to_string();
    let dev = Dev::new(&c);
    launch(m.tl.func(&k), c.grid(), c.block(), 0, &dev.fwd_args(&c)); sync();
    let outname = if k == "ugwp_oro_dx" { "dx" } else { "diff" };
    let ident = ndiff(&dev.get_f32(&c, outname), &c.post_f32(outname));
    dev.reset(&c); launch(img.func(&k), c.grid(), c.block(), 0, &dev.fwd_args(&c)); sync();
    let ident_img = ndiff(&dev.get_f32(&c, outname), &c.post_f32(outname));
    let n = c.scalar_i32("n") as usize;
    let mut r = Rng::new(3);
    if k == "ugwp_oro_dx" {
        let d = dbuf_f64(&(0..n).map(|_| r.n()).collect::<Vec<_>>());
        let mut a = dev.fwd_args(&c); a.push(p(&d)); launch(m.tl.func("ugwp_oro_dx__tl64"), c.grid(), c.block(), 0, &a); sync();
        let tl_nonzero = read_f64(&d, n).iter().filter(|v| **v != 0.0).count();
        let b = dbuf_f64(&(0..n).map(|_| r.n()).collect::<Vec<_>>());
        let mut a = dev.fwd_args(&c); a.push(p(&b)); launch(m.rev.func("ugwp_oro_dx__ad64"), c.grid(), c.block(), 0, &a); sync();
        let ad_nonzero = read_f64(&b, n).iter().filter(|v| **v != 0.0).count();
        return json!({"capture": dir, "kernel": k, "identity_derivative_module_vs_post": ident, "identity_captured_image_vs_post": ident_img,
                      "rule": "dx from meshDensity (geometry): tangent 0, cotangent of dx zeroed", "tl_nonzero_words": tl_nonzero, "ad_nonzero_words_after": ad_nonzero});
    }
    let (bv, av) = (c.pre_f32("before"), c.pre_f32("after"));
    let dbv: Vec<f64> = (0..n).map(|_| r.n()).collect(); let dav: Vec<f64> = (0..n).map(|_| r.n()).collect(); let yv: Vec<f64> = (0..n).map(|_| r.n()).collect();
    let (db, da, dd) = (dbuf_f64(&dbv), dbuf_f64(&dav), DBuf::new(8 * n));
    let mut a = dev.fwd_args(&c); a.extend([p(&db), p(&da), p(&dd)]); launch(m.tl.func("ugwp_oro_diff__tl64"), c.grid(), c.block(), 0, &a); sync();
    let jd = read_f64(&dd, n);
    let (bb, ba, bd) = (DBuf::new(8 * n), DBuf::new(8 * n), dbuf_f64(&yv));
    let mut a = dev.fwd_args(&c); a.extend([p(&bb), p(&ba), p(&bd)]); launch(m.rev.func("ugwp_oro_diff__ad64"), c.grid(), c.block(), 0, &a); sync();
    let (l, rr) = (dot(&jd, &yv), dot(&dbv, &read_f64(&bb, n)) + dot(&dav, &read_f64(&ba, n)));
    // FD of the captured image along (dbefore, dafter) at eps 1e-3 of field std
    let sb = plane_std(&bv, 1, n)[0]; let sa = plane_std(&av, 1, n)[0];
    let eps = 1e-3;
    let run = |sg: f64| { dev.reset(&c);
        dev.put_f32(&c, "before", &bv.iter().zip(&dbv).map(|(&x, &v)| (x as f64 + sg * eps * sb * v) as f32).collect::<Vec<_>>());
        dev.put_f32(&c, "after", &av.iter().zip(&dav).map(|(&x, &v)| (x as f64 + sg * eps * sa * v) as f32).collect::<Vec<_>>());
        launch(img.func(&k), c.grid(), c.block(), 0, &dev.fwd_args(&c)); sync(); dev.get_f32(&c, "diff") };
    let (pp, mm) = (run(1.0), run(-1.0));
    let fd: Vec<f64> = pp.iter().zip(&mm).map(|(&x, &y)| (x as f64 - y as f64) / (2.0 * eps)).collect();
    let tlv: Vec<f64> = dav.iter().zip(&dbv).map(|(&a2, &b2)| a2 * sa - b2 * sb).collect();
    let e = nrm(&tlv.iter().zip(&fd).map(|(x, y)| x - y).collect::<Vec<_>>()) / nrm(&fd).max(1e-300);
    json!({"capture": dir, "kernel": k, "identity_derivative_module_vs_post": ident, "identity_captured_image_vs_post": ident_img,
           "dot_f64": {"lhs": l, "rhs": rr, "rel": (l - rr).abs() / l.abs().max(rr.abs())}, "fd_rel_eps_1e-3": e, "b_diff_nonzero_after": read_f64(&bd, n).iter().filter(|v| **v != 0.0).count()})
}

fn sfc_small(dir: &Path, m: &Mods) -> Value {
    // original from_mpas / to_mpas kernels of the derivative module against the capture
    let c = Cap::load(dir);
    let k = c.m["kernel"].as_str().unwrap().to_string();
    let dev = Dev::new(&c);
    launch(m.tl.func(&k), c.grid(), c.block(), 0, &dev.fwd_args(&c)); sync();
    let mut d = 0;
    for a in c.args() { if a["kind"] == "device" && a["dtype"] == "<f4" { let nm = a["name"].as_str().unwrap(); d += ndiff(&dev.get_f32(&c, nm), &c.post_f32(nm)); } }
    json!({"capture": dir, "kernel": k, "identity_derivative_module_original_kernel_vs_post": d})
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 { eprintln!("usage: p4-proof build <gen> <out> <libnvrtc> <arch> | run <cubins> <arch> <captures> <receipt.json> [seeds]"); std::process::exit(2); }
    if a[1] == "build" { build(&a[2..]); return; }
    if a[1] == "check" { init(); check::run(&a[2..]); return; }
    if a[1] == "spdiag" { init(); check::spdiag(&a[2..]); return; }
    if a[1] == "pdiag" { init(); check::pdiag(&a[2..]); return; }
    if a[1] == "path" { init(); check::path(&a[2..]); return; }
    if a[1] == "gdiag" { init(); check::gdiag(&a[2..]); return; }
    if a[1] == "diag" { init(); check::diag(&a[2..]); return; }
    init();
    let (cub, arch, root, rec) = (PathBuf::from(&a[2]), a[3].clone(), PathBuf::from(&a[4]), PathBuf::from(&a[5]));
    let seeds: u64 = a.get(6).map(|s| s.parse().unwrap()).unwrap_or(6);
    let only = a.get(7).cloned().unwrap_or_default();
    let ms = load_mods(&cub, "mynnsfc", &arch);
    let mg = load_mods(&cub, "ugwp", &arch);
    let t0 = std::time::Instant::now();
    let mut out = serde_json::Map::new();
    let one = std::env::var("P4_ONE").is_ok();
    let launches = |k: &str| -> Vec<PathBuf> { let mut v = vec![]; for st in ["1", "2"] { for l in 0..4 { let p = root.join(k).join(st).join(format!("launch-{l:06}")); if p.join("manifest.json").exists() { v.push(p); } } } if one { v.truncate(1); } v };
    let want = |k: &str| only.is_empty() || only.split(',').any(|x| x == k);
    if want("sfc") { out.insert("mynnsfc_columns".into(), Value::Array(launches("mynnsfc_columns").iter().map(|d| { let r = sfc_unit(d, &ms, seeds); eprintln!("sfc {} {:.0}s", d.display(), t0.elapsed().as_secs_f64()); r }).collect())); }
    if want("sfcsmall") { out.insert("mynnsfc_from_to".into(), Value::Array(launches("mynnsfc_from_mpas").iter().chain(launches("mynnsfc_to_mpas").iter()).map(|d| sfc_small(d, &ms)).collect())); }
    if want("chain") {
        let mut v = vec![];
        for st in ["1", "2"] { for l in 0..2 {
            let f = root.join("mynnsfc_from_mpas").join(st).join(format!("launch-{l:06}"));
            let c0 = root.join("mynnsfc_columns").join(st).join(format!("launch-{:06}", 2 * l));
            let c1 = root.join("mynnsfc_columns").join(st).join(format!("launch-{:06}", 2 * l + 1));
            let t = root.join("mynnsfc_to_mpas").join(st).join(format!("launch-{l:06}"));
            v.push(sfc_chain(&f, &c0, &c1, &t, &ms)); eprintln!("chain {st}/{l} {:.0}s", t0.elapsed().as_secs_f64());
        } }
        out.insert("mynnsfc_slot_chain".into(), Value::Array(v));
    }
    if want("gsl") { out.insert("ugwp_oro_run".into(), Value::Array(launches("ugwp_oro_run").iter().map(|d| { let r = gsl_unit(d, &mg, seeds); eprintln!("gsl {} {:.0}s", d.display(), t0.elapsed().as_secs_f64()); r }).collect())); }
    if want("gslsmall") { out.insert("ugwp_oro_dx_diff".into(), Value::Array(launches("ugwp_oro_dx").iter().chain(launches("ugwp_oro_diff").iter()).map(|d| gsl_small(d, &mg)).collect())); }
    out.insert("wall_seconds".into(), json!(t0.elapsed().as_secs_f64()));
    std::fs::write(&rec, serde_json::to_string_pretty(&Value::Object(out)).unwrap()).unwrap();
    eprintln!("receipt {}", rec.display());
}
