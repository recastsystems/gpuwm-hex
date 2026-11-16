// SPDX-License-Identifier: Apache-2.0
//! p4-proof check: independent P4 check (ad/phys-p4-check). Per-column, per-regime proof of the
//! whole-launch tangents against the captured forward image:
//!  - one direction per input field (plus a full-state direction), all columns at once (columns are
//!    independent), centred FD of the unchanged captured image at 5 eps;
//!  - per column: best error over the eps at which the column takes exactly the same branches on both
//!    sides (two-sided), and best one-sided error where only one side flips (the TL must then match
//!    the side that keeps the executed branch);
//!  - grouped by weather regime (moist/dry, near freezing, saturated/unsaturated, stable (night) /
//!    unstable (day), land/water/ice/snow, terrain);
//!  - dot tests restricted to each regime;
//!  - GSL drag again at the 3 km grid spacing the target hrrrv5 runs at (dx row overridden to 3000 m:
//!    the large-scale branch is tapered off there, a different code path than the 480 km capture).
use super::*;

const CEPS: [f64; 8] = [3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4];
const BAD: f64 = 1e-2;
const RESP_FLOOR: f64 = 1e-3;

struct Probe<'a> {
    n: usize,
    run: Box<dyn Fn(&[f32]) -> Vec<f32> + 'a>,
    aug: Box<dyn Fn(&[f32]) -> (Vec<u32>, Vec<i32>) + 'a>,
    tl: Box<dyn Fn(&[f64]) -> Vec<f64> + 'a>,
}

/// per column: (sum ((tl-fd)/s)^2, sum (fd/s)^2, sum (tl/s)^2, non-finite count); packed planes of n.
fn colerr(tl: &[f64], fd: &[f64], sc: &[f64], n: usize) -> Vec<(f64, f64, f64, usize)> {
    let mut r = vec![(0.0, 0.0, 0.0, 0usize); n];
    for q in 0..sc.len() {
        let s = sc[q];
        if !(s > 0.0) { continue; }
        for i in 0..n {
            let (t, f) = (tl[q * n + i], fd[q * n + i]);
            if !t.is_finite() || !f.is_finite() { r[i].3 += 1; continue; }
            r[i].0 += ((t - f) / s).powi(2); r[i].1 += (f / s).powi(2); r[i].2 += (t / s).powi(2);
        }
    }
    r
}
fn rel(e: (f64, f64, f64, usize)) -> f64 {
    if e.3 > 0 { return f64::INFINITY; }
    let d = e.1.max(e.2);
    if d == 0.0 { return if e.0 == 0.0 { 0.0 } else { f64::INFINITY }; }
    (e.0 / d).sqrt()
}

#[derive(Clone)]
struct ColBest { two: Vec<f64>, one: Vec<f64>, resp: Vec<f64>, nfree: Vec<u32>, eps_two: Vec<f64>, nonfinite: usize, any: Vec<f64>, eps_any: Vec<f64>, curve: Vec<Vec<(f32, u32)>> }

fn probe(p: &Probe, x0: &[f32], y0: &[f32], br0: &[u32], bc0: &[i32], d: &[f64], sc: &[f64]) -> ColBest {
    let n = p.n;
    let mut cb = ColBest { two: vec![f64::INFINITY; n], one: vec![f64::INFINITY; n], resp: vec![0.0; n], nfree: vec![0; n], eps_two: vec![0.0; n], nonfinite: 0, any: vec![f64::INFINITY; n], eps_any: vec![0.0; n], curve: vec![vec![]; n] };
    for &eps in &CEPS {
        let xp: Vec<f32> = x0.iter().zip(d).map(|(&x, &v)| (x as f64 + eps * v) as f32).collect();
        let xm: Vec<f32> = x0.iter().zip(d).map(|(&x, &v)| (x as f64 - eps * v) as f32).collect();
        let (yp, ym) = ((p.run)(&xp), (p.run)(&xm));
        let (brp, bcp) = (p.aug)(&xp); let (brm, bcm) = (p.aug)(&xm);
        let fp: Vec<u64> = (0..n).map(|i| flips(br0, bc0[i], &brp, bcp[i], n, i).0).collect();
        let fm: Vec<u64> = (0..n).map(|i| flips(br0, bc0[i], &brm, bcm[i], n, i).0).collect();
        let drc: Vec<f64> = xp.iter().zip(&xm).map(|(&a, &b)| (a as f64 - b as f64) / (2.0 * eps)).collect();
        let fdc: Vec<f64> = yp.iter().zip(&ym).map(|(&a, &b)| (a as f64 - b as f64) / (2.0 * eps)).collect();
        let ec = colerr(&(p.tl)(&drc), &fdc, sc, n);
        let one_sided = (0..n).any(|i| (fp[i] == 0) != (fm[i] == 0));
        let (ep, em) = if one_sided {
            let drp: Vec<f64> = xp.iter().zip(x0).map(|(&a, &b)| (a as f64 - b as f64) / eps).collect();
            let drm: Vec<f64> = x0.iter().zip(&xm).map(|(&a, &b)| (a as f64 - b as f64) / eps).collect();
            let fdp: Vec<f64> = yp.iter().zip(y0).map(|(&a, &b)| (a as f64 - b as f64) / eps).collect();
            let fdm: Vec<f64> = y0.iter().zip(&ym).map(|(&a, &b)| (a as f64 - b as f64) / eps).collect();
            (colerr(&(p.tl)(&drp), &fdp, sc, n), colerr(&(p.tl)(&drm), &fdm, sc, n))
        } else { (vec![], vec![]) };
        for i in 0..n {
            if eps == 1e-3 { cb.resp[i] = ec[i].2.sqrt(); }
            cb.nonfinite += ec[i].3;
            let ra = rel(ec[i]); if ra < cb.any[i] { cb.any[i] = ra; cb.eps_any[i] = eps; }
            cb.curve[i].push((ra as f32, (fp[i] + fm[i]) as u32));
            if fp[i] == 0 && fm[i] == 0 {
                cb.nfree[i] += 1;
                let r = rel(ec[i]);
                if r < cb.two[i] { cb.two[i] = r; cb.eps_two[i] = eps; }
            } else if one_sided && fp[i] == 0 { cb.one[i] = cb.one[i].min(rel(ep[i])); }
            else if one_sided && fm[i] == 0 { cb.one[i] = cb.one[i].min(rel(em[i])); }
        }
    }
    cb
}

fn pct(mut v: Vec<f64>, q: f64) -> f64 {
    v.retain(|x| !x.is_nan());
    if v.is_empty() { return f64::NAN; }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * q).round() as usize]
}
fn jf(x: f64) -> Value { if x.is_finite() { json!(x) } else { json!(format!("{x}")) } }

/// Summary of the per-column bests of a set of directions over a set of columns.
fn summarize(bests: &[(String, ColBest)], cols: &[usize]) -> Value {
    let (mut two, mut one, mut weak, mut never, mut weak_abs) = (vec![], vec![], 0usize, 0usize, 0f64);
    let mut bad_two = 0usize; let mut bad_any = 0usize; let mut anyv = vec![]; let mut bad_agree = 0usize;
    for (_, b) in bests {
        for &i in cols {
            if b.resp[i] <= RESP_FLOOR { weak += 1; if b.two[i].is_finite() { weak_abs = weak_abs.max(b.two[i] * b.resp[i]); } continue; }
            anyv.push(b.any[i]); if b.any[i] > BAD { bad_agree += 1; }
            let best = b.two[i].min(b.one[i]);
            if b.two[i].is_finite() { two.push(b.two[i]); if b.two[i] > BAD { bad_two += 1; } }
            else if b.one[i].is_finite() { one.push(b.one[i]); }
            else { never += 1; }
            if best.is_finite() && best > BAD { bad_any += 1; }
        }
    }
    json!({"columns": cols.len(), "pairs": anyv.len(), "agree_median": jf(pct(anyv.clone(), 0.5)), "agree_p99": jf(pct(anyv.clone(), 0.99)), "agree_p999": jf(pct(anyv.clone(), 0.999)), "agree_max": jf(pct(anyv, 1.0)), "pairs_agree_over_1e-2": bad_agree, "pairs_two_sided": two.len(), "two_sided_median": jf(pct(two.clone(), 0.5)), "two_sided_p99": jf(pct(two.clone(), 0.99)), "two_sided_max": jf(pct(two, 1.0)),
           "pairs_one_sided_only": one.len(), "one_sided_median": jf(pct(one.clone(), 0.5)), "one_sided_p99": jf(pct(one.clone(), 0.99)), "one_sided_max": jf(pct(one, 1.0)),
           "pairs_two_sided_over_1e-2": bad_two, "pairs_best_over_1e-2": bad_any, "pairs_flip_at_every_eps_both_sides": never,
           "pairs_weak_response": weak, "weak_response_max_abs_scaled_error": weak_abs})
}

fn worst(bests: &[(String, ColBest)], tags: &dyn Fn(usize) -> Vec<&'static str>, k: usize) -> Value {
    let mut v = vec![];
    for (name, b) in bests { for i in 0..b.two.len() { if b.resp[i] > RESP_FLOOR { v.push((b.any[i], name.clone(), i)); } } }
    v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    Value::Array(v.iter().take(k).map(|(e, nm, i)| { let b = &bests.iter().find(|x| &x.0 == nm).unwrap().1;
        json!({"direction": nm, "column": i, "agree_best": jf(*e), "agree_eps": b.eps_any[*i], "curve_rel_flips": b.curve[*i].iter().map(|(r, f)| json!([jf(*r as f64), f])).collect::<Vec<_>>(), "two_sided_best": jf(b.two[*i]), "at_eps": b.eps_two[*i], "flip_free_eps_count": b.nfree[*i], "one_sided_best": jf(b.one[*i]), "response": b.resp[*i], "regimes": tags(*i)}) }).collect())
}

// ------------------------------------------------------------------------------------- MYNN-SFC
/// The FD referee: the captured image, or (P4_IMAGE_DIR/P4_IMAGE_ARCH set, for cards the captured sm_120 image
/// cannot load on) the verbatim original kernel inside the derivative module built for that card; its identity
/// against the captured post state is reported either way.
fn ref_image(c: &Cap, fam: &str) -> Module {
    match (std::env::var("P4_IMAGE_DIR"), std::env::var("P4_IMAGE_ARCH")) {
        (Ok(d), Ok(a)) => Module::load(&std::fs::read(PathBuf::from(d).join(format!("{fam}.tl.{a}.cubin"))).unwrap()),
        _ => Module::load(&c.image()),
    }
}
fn qsat(t: f64, p: f64) -> f64 { let es = 611.2 * (17.67 * (t - 273.15) / (t - 29.65)).exp(); 0.622 * es / (p - 0.378 * es).max(1.0) }

fn sfc_check(dir: &Path, m: &Mods) -> Value {
    let c = Cap::load(dir);
    let img = ref_image(&c, "mynnsfc"); tables(&img);
    let n = c.scalar_i32("n") as usize;
    let s = Sfc { c: &c, m, img: &img, dev: Dev::new(&c), n };
    let x0 = c.pre_f32("f"); let post = c.post_f32("f");
    let (y0, _) = s.orig(&x0, true);
    let (ya, _, br0, bc0) = s.aug(&x0);
    let ident = json!({"image_vs_post": ndiff(&y0, &post), "aug_vs_post": ndiff(&ya, &post)});
    let g = |j: usize, i: usize| x0[j * n + i] as f64; let gy = |j: usize, i: usize| post[j * n + i] as f64;
    let thr = c.scalar_f32("xice_threshold") as f64;
    let rh = |i: usize| g(6, i) / qsat(g(5, i), g(7, i));
    let mut reg: Vec<(&'static str, Vec<usize>)> = vec![];
    let f = |pred: &dyn Fn(usize) -> bool| -> Vec<usize> { (0..n).filter(|&i| pred(i)).collect() };
    let land = |i: usize| g(66, i) < thr && g(31, i) < 1.5;
    reg.push(("land", f(&land)));
    reg.push(("water", f(&|i| g(66, i) < thr && g(31, i) >= 1.5)));
    reg.push(("ice", f(&|i| g(66, i) >= thr)));
    reg.push(("snow-covered land", f(&|i| land(i) && g(46, i) > 0.0)));
    reg.push(("moist (qv > 12 g/kg)", f(&|i| g(6, i) > 0.012)));
    reg.push(("dry (qv < 3 g/kg)", f(&|i| g(6, i) < 0.003)));
    reg.push(("near freezing (|T or Tsk - 0C| < 3 K)", f(&|i| (g(5, i) - 273.15).abs() < 3.0 || (g(36, i) - 273.15).abs() < 3.0)));
    reg.push(("saturated (RH >= 95%)", f(&|i| rh(i) >= 0.95)));
    reg.push(("unsaturated (RH < 60%)", f(&|i| rh(i) < 0.6)));
    reg.push(("stable (z/L > 0, night-like)", f(&|i| gy(26, i) > 0.0)));
    reg.push(("unstable (z/L < 0, day-like)", f(&|i| gy(26, i) < 0.0)));
    reg.push(("land day (H > 20 W/m2)", f(&|i| land(i) && gy(33, i) > 20.0)));
    reg.push(("land night (H < 0)", f(&|i| land(i) && gy(33, i) < 0.0)));
    reg.push(("very stable (z/L > 1)", f(&|i| gy(26, i) > 1.0)));
    let regc = reg.clone();
    let tags = move |i: usize| -> Vec<&'static str> { regc.iter().filter(|(_, v)| v.binary_search(&i).is_ok()).map(|(k, _)| *k).collect() };
    let sx = plane_std(&x0, NF, n); let sy = plane_std(&post, NF, n);
    // error planes: fields the kernel writes (post differs from pre somewhere)
    let written: Vec<usize> = (0..NF).filter(|&j| (0..n).any(|i| x0[j * n + i].to_bits() != post[j * n + i].to_bits())).collect();
    let sc: Vec<f64> = (0..NF).map(|j| if written.contains(&j) { sy[j] } else { 0.0 }).collect();
    let pr = Probe { n, run: Box::new(|x| s.orig(x, true).0), aug: Box::new(|x| { let r = s.aug(x); (r.2, r.3) }), tl: Box::new(|d| s.tl64(&x0, d).0) };
    let active = |j: usize| !SFC_FIXED.contains(&j) && sx[j] > 0.0;
    let mkd = |seed: u64, only: Option<usize>| -> Vec<f64> { let mut r = Rng::new(seed); let mut d = vec![0.0; NF * n];
        for j in 0..NF { if active(j) && only.map_or(true, |o| o == j) { for i in 0..n { let v = r.n() * sx[j]; if sfc_interior(j, x0[j * n + i]) { d[j * n + i] = v; } } } } d };
    let mut bests = vec![];
    for j in 0..NF { if active(j) { bests.push((SFC_FIELDS[j].to_string(), probe(&pr, &x0, &y0, &br0, &bc0, &mkd(1000 + j as u64, Some(j)), &sc))); } }
    bests.push(("full state".into(), probe(&pr, &x0, &y0, &br0, &bc0, &mkd(999, None), &sc)));
    let all: Vec<usize> = (0..n).collect();
    let per_dir: Vec<Value> = bests.iter().map(|(nm, b)| { let mut v = summarize(std::slice::from_ref(&(nm.clone(), b.clone())), &all); v["direction"] = json!(nm); v["nonfinite_words"] = json!(b.nonfinite); v }).collect();
    let per_reg: Vec<Value> = reg.iter().map(|(k, cols)| { let mut v = summarize(&bests, cols); v["regime"] = json!(k);
        // regime-restricted dot (f64), full direction
        let mut d = mkd(4242, None); let mut y = vec![0.0; NF * n]; let mut r = Rng::new(77);
        let inr: Vec<bool> = (0..n).map(|i| cols.binary_search(&i).is_ok()).collect();
        for j in 0..NF { for i in 0..n { if !inr[i] { d[j * n + i] = 0.0; } else if sy[j] > 0.0 { y[j * n + i] = r.n() / sy[j]; } } }
        let (jd, _) = s.tl64(&x0, &d); let jty = s.ad64(&x0, &y);
        let (l, rr) = (dot(&jd, &y), dot(&d, &jty));
        v["dot_f64_rel"] = json!(if cols.is_empty() { 0.0 } else { (l - rr).abs() / l.abs().max(rr.abs()).max(1e-300) }); v }).collect();
    json!({"capture": dir, "columns": n, "identity": ident, "written_fields": written.iter().map(|&j| SFC_FIELDS[j]).collect::<Vec<_>>(),
           "all_columns_all_directions": summarize(&bests, &all), "per_regime": per_reg, "per_direction": per_dir, "worst_agree": worst(&bests, &tags, 40)})
}

// ------------------------------------------------------------------------------------------ GSL
fn gtl64(g: &Gsl, l: &[f32], pr: &[f32], s: &[f32], d: &[f64]) -> Vec<f64> {
    let c = g.c; let (nl, np) = (l.len(), pr.len());
    g.put(l, pr, s);
    let (no, ns) = (c.count("out"), c.count("sout"));
    let (dl, dp, ds, dout, dso, st, sg, rb) = (dbuf_f64(&d[..nl]), dbuf_f64(&d[nl..nl + np]), dbuf_f64(&d[nl + np..]), DBuf::new(8 * no), DBuf::new(8 * ns), DBuf::new(4), DBuf::new(4), DBuf::new(4));
    let mut a = g.dev.fwd_args(c); a.extend([p(&dl), p(&dp), p(&ds), p(&dout), p(&dso), p(&st), p(&sg), p(&rb)]);
    launch(g.m.tl.func("ugwp_oro_run__tl64"), c.grid(), c.block(), 0, &a); sync();
    let mut o = read_f64(&dout, no); o.extend(read_f64(&dso, ns)); o
}
fn gad64(g: &Gsl, l: &[f32], pr: &[f32], s: &[f32], y: &[f64]) -> Vec<f64> {
    let c = g.c; let nc = g.nc;
    g.put(l, pr, s);
    let (no, nl, np, nsf) = (c.count("out"), c.count("layers"), c.count("prsi0"), c.count("surface"));
    let (bo, bs, bl, bp, bsf) = (dbuf_f64(&y[..no]), dbuf_f64(&y[no..]), DBuf::new(8 * nl), DBuf::new(8 * np), DBuf::new(8 * nsf));
    let counts = DBuf::new(4 * nc); let ov = DBuf::new(4); let sg = DBuf::new(4); let st = DBuf::new(4);
    let mk = |ab: KArg, w: KArg, adj: KArg, cap: i32, stride: i64, c0: i32, ncol: i32, cnt: KArg| {
        let mut a = g.dev.fwd_args(c);
        a.extend([p(&bl), p(&bp), p(&bsf), p(&bo), p(&bs), ab, w, adj, i32b(cap), i64b(stride), i32b(c0), i32b(ncol), cnt, p(&ov), p(&sg), p(&st)]); a };
    launch(g.m.rev.func("ugwp_oro_run__ad64"), grid1(nc, 64), [64, 1, 1], 0, &mk(null(), null(), null(), 0, nc as i64, 0, nc as i32, p(&counts))); sync();
    let cap = *read_i32(&counts, nc).iter().max().unwrap() as usize;
    let chunk = (((2usize << 30) / (32 * cap.max(1))) / 64 * 64).clamp(64, nc.div_ceil(64) * 64);
    let (ab, w, adj) = (DBuf::new(8 * cap * chunk), DBuf::new(16 * cap * chunk), DBuf::new(8 * cap * chunk));
    let mut c0 = 0;
    while c0 < nc { let k = chunk.min(nc - c0); launch(g.m.rev.func("ugwp_oro_run__ad64"), grid1(k, 64), [64, 1, 1], 0, &mk(p(&ab), p(&w), p(&adj), cap as i32, chunk as i64, c0 as i32, k as i32, null())); c0 += k; }
    sync();
    assert_eq!(read_i32(&ov, 1)[0], 0, "tape overflow");
    let mut o = read_f64(&bl, nl); o.extend(read_f64(&bp, np)); o.extend(read_f64(&bsf, nsf)); o
}

fn gsl_check(dir: &Path, m: &Mods, dx_override: Option<f32>) -> Value {
    let c = Cap::load(dir);
    let img = ref_image(&c, "ugwp");
    let nc = c.scalar_i32("nc") as usize; let n3 = NZ * nc;
    let g = Gsl { c: &c, m, img: &img, dev: Dev::new(&c), nc };
    let (l0, p0, mut s0) = (c.pre_f32("layers"), c.pre_f32("prsi0"), c.pre_f32("surface"));
    if let Some(dx) = dx_override { for i in 0..nc { s0[5 * nc + i] = dx; } }
    let (nl, np) = (l0.len(), p0.len());
    let mut x0 = l0.clone(); x0.extend(&p0); x0.extend(&s0);
    let split = |x: &[f32]| -> (Vec<f32>, Vec<f32>, Vec<f32>) { (x[..nl].to_vec(), x[nl..nl + np].to_vec(), x[nl + np..].to_vec()) };
    let run = |x: &[f32], img_: bool| -> Vec<f32> { let (l, pp, s) = split(x); let (mut o, so, _) = g.orig(&l, &pp, &s, img_); o.extend(so); o };
    let y0 = run(&x0, true);
    let ym = run(&x0, false);
    let (oa, sa, br0, bc0, rbw) = g.aug(&l0, &p0, &s0);
    let mut ya = oa; ya.extend(sa);
    let mut ident = json!({"image_vs_derivative_module_original": ndiff(&y0, &ym), "image_vs_aug": ndiff(&y0, &ya), "aug_read_before_write": rbw,
        "branch_records_per_column_max": bc0.iter().max()});
    if dx_override.is_none() { let mut post = c.post_f32("out"); post.extend(c.post_f32("sout")); ident["image_vs_post"] = json!(ndiff(&y0, &post)); }
    let gl = |f: usize, k: usize, i: usize| l0[f * n3 + (k - 1) * nc + i] as f64; let gs = |r: usize, i: usize| s0[r * nc + i] as f64;
    let zs = |i: usize| gl(6, 1, i) - 0.5 * gl(7, 1, i);
    let f = |pred: &dyn Fn(usize) -> bool| -> Vec<usize> { (0..nc).filter(|&i| pred(i)).collect() };
    let mut reg: Vec<(&'static str, Vec<usize>)> = vec![];
    reg.push(("high terrain (>= 1000 m)", f(&|i| gs(4, i) < 1.5 && zs(i) >= 1000.0 && gs(6, i) > 0.0)));
    reg.push(("low terrain land", f(&|i| gs(4, i) < 1.5 && zs(i) < 1000.0 && gs(6, i) > 0.0)));
    reg.push(("no subgrid terrain (var = 0)", f(&|i| gs(6, i) <= 0.0)));
    reg.push(("stable surface (Rib > 0, night-like)", f(&|i| gs(3, i) > 0.0)));
    reg.push(("unstable surface (Rib < 0, day-like)", f(&|i| gs(3, i) < 0.0)));
    reg.push(("shallow PBL (< 300 m)", f(&|i| gs(2, i) < 300.0)));
    reg.push(("deep PBL (> 1500 m)", f(&|i| gs(2, i) > 1500.0)));
    reg.push(("moist (qv1 > 12 g/kg)", f(&|i| gl(3, 1, i) > 0.012)));
    reg.push(("dry (qv1 < 3 g/kg)", f(&|i| gl(3, 1, i) < 0.003)));
    reg.push(("near freezing (|T1 - 0C| < 3 K)", f(&|i| (gl(2, 1, i) - 273.15).abs() < 3.0)));
    let regc = reg.clone();
    let tags = move |i: usize| -> Vec<&'static str> { regc.iter().filter(|(_, v)| v.binary_search(&i).is_ok()).map(|(k, _)| *k).collect() };
    let (po, ps) = (y0[..12 * n3].to_vec(), y0[12 * n3..].to_vec());
    let ostd = plane_std(&po, 12, n3); let sostd = plane_std(&ps, 10, nc);
    let mut sc = vec![0.0; 12 * NZ + 10];
    for mm in 0..12 { for k in 0..NZ { sc[mm * NZ + k] = ostd[mm]; } }
    for mm in 0..10 { sc[12 * NZ + mm] = sostd[mm]; }
    let pr = Probe { n: nc, run: Box::new(|x| run(x, true)), aug: Box::new(|x| { let (l, pp, s) = split(x); let r = g.aug(&l, &pp, &s); (r.2, r.3) }), tl: Box::new(|d| gtl64(&g, &l0, &p0, &s0, d)) };
    let lstd = plane_std(&l0, 10 * NZ, nc); let pstd = plane_std(&p0, NZ + 1, nc); let sstd = plane_std(&s0, 26, nc);
    // direction over packed planes; field selectors: layer field f (all levels), prsi0, surface row r
    let mkd = |seed: u64, lf: &[usize], pr_: bool, sr: &[usize]| -> Vec<f64> {
        let mut r = Rng::new(seed); let mut d = vec![0.0; x0.len()];
        for &fl in lf { for q in fl * NZ..(fl + 1) * NZ { for i in 0..nc { let v = r.n() * lstd[q]; if fl != 3 || l0[q * nc + i] > 0.0 { d[q * nc + i] = v; } } } }
        if pr_ { for q in 0..NZ + 1 { for i in 0..nc { d[nl + q * nc + i] = r.n() * pstd[q]; } } }
        for &row in sr { for i in 0..nc { let x = s0[row * nc + i]; let v = r.n() * sstd[row];
            let ok = match row { 6 | 7 | 16 => x > 0.0, 12..=15 => x > 0.0 && x < 1.0, 2 => x > 0.0, _ => true };
            if ok { d[nl + np + row * nc + i] = v; } } }
        d };
    let lname = ["u", "v", "T", "qv", "p", "exner"];
    let sname = [(2, "hpbl"), (3, "bulk Rib"), (6, "var"), (7, "oc1"), (8, "oa1"), (9, "oa2"), (10, "oa3"), (11, "oa4"), (12, "ol1"), (13, "ol2"), (14, "ol3"), (15, "ol4"), (16, "varss")];
    let mut bests = vec![];
    for (fl, nm) in lname.iter().enumerate() { bests.push((format!("layer {nm}"), probe(&pr, &x0, &y0, &br0, &bc0, &mkd(500 + fl as u64, &[fl], false, &[]), &sc))); }
    bests.push(("prsi0".into(), probe(&pr, &x0, &y0, &br0, &bc0, &mkd(520, &[], true, &[]), &sc)));
    for (row, nm) in sname { bests.push((format!("surface {nm}"), probe(&pr, &x0, &y0, &br0, &bc0, &mkd(530 + row as u64, &[], false, &[row]), &sc))); }
    bests.push(("full state".into(), probe(&pr, &x0, &y0, &br0, &bc0, &mkd(599, &[0, 1, 2, 3, 4, 5], true, &[2, 3]), &sc)));
    let all: Vec<usize> = (0..nc).collect();
    let per_dir: Vec<Value> = bests.iter().map(|(nm, b)| { let mut v = summarize(std::slice::from_ref(&(nm.clone(), b.clone())), &all); v["direction"] = json!(nm); v["nonfinite_words"] = json!(b.nonfinite); v }).collect();
    let ysc: Vec<f64> = (0..y0.len()).map(|k| if k < 12 * n3 { ostd[k / n3] } else { sostd[(k - 12 * n3) / nc] }).collect();
    let per_reg: Vec<Value> = reg.iter().map(|(k, cols)| { let mut v = summarize(&bests, cols); v["regime"] = json!(k);
        let inr: Vec<bool> = (0..nc).map(|i| cols.binary_search(&i).is_ok()).collect();
        let mut d = mkd(4243, &[0, 1, 2, 3, 4, 5], true, &[2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        for (k2, v2) in d.iter_mut().enumerate() { if !inr[k2 % nc] { *v2 = 0.0; } }
        let mut r = Rng::new(78);
        let y: Vec<f64> = (0..y0.len()).map(|k2| if inr[k2 % nc] && ysc[k2] > 0.0 { r.n() / ysc[k2] } else { 0.0 }).collect();
        let jd = gtl64(&g, &l0, &p0, &s0, &d); let jty = gad64(&g, &l0, &p0, &s0, &y);
        let (l, rr) = (dot(&jd, &y), dot(&d, &jty));
        v["dot_f64_rel"] = json!(if cols.is_empty() || (l == 0.0 && rr == 0.0) { 0.0 } else { (l - rr).abs() / l.abs().max(rr.abs()) }); v["dot_lhs"] = json!(l); v }).collect();
    json!({"capture": dir, "dx_override_m": dx_override, "columns": nc, "identity": ident,
           "all_columns_all_directions": summarize(&bests, &all), "per_regime": per_reg, "per_direction": per_dir, "worst_agree": worst(&bests, &tags, 40)})
}

pub fn run(a: &[String]) {
    let (cub, arch, root, rec) = (PathBuf::from(&a[0]), a[1].clone(), PathBuf::from(&a[2]), PathBuf::from(&a[3]));
    let only = a.get(4).cloned().unwrap_or_default();
    let want = |k: &str| only.is_empty() || only.split(',').any(|x| x == k);
    let ms = load_mods(&cub, "mynnsfc", &arch);
    let mg = load_mods(&cub, "ugwp", &arch);
    let t0 = std::time::Instant::now();
    let launches = |k: &str| -> Vec<PathBuf> { let mut v = vec![]; for st in ["1", "2"] { for l in 0..4 { let p = root.join(k).join(st).join(format!("launch-{l:06}")); if p.join("manifest.json").exists() { v.push(p); } } } v };
    let mut out = serde_json::Map::new();
    if want("sfc") { out.insert("mynnsfc_columns".into(), Value::Array(launches("mynnsfc_columns").iter().map(|d| { let r = sfc_check(d, &ms); eprintln!("sfc {} {:.0}s", d.display(), t0.elapsed().as_secs_f64()); r }).collect())); }
    if want("gsl") { out.insert("ugwp_oro_run".into(), Value::Array(launches("ugwp_oro_run").iter().map(|d| { let r = gsl_check(d, &mg, None); eprintln!("gsl {} {:.0}s", d.display(), t0.elapsed().as_secs_f64()); r }).collect())); }
    if want("gsl3km") { out.insert("ugwp_oro_run_dx3000".into(), Value::Array(launches("ugwp_oro_run").iter().map(|d| { let r = gsl_check(d, &mg, Some(3000.0)); eprintln!("gsl3km {} {:.0}s", d.display(), t0.elapsed().as_secs_f64()); r }).collect())); }
    out.insert("wall_seconds".into(), json!(t0.elapsed().as_secs_f64()));
    std::fs::write(&rec, serde_json::to_string_pretty(&Value::Object(out)).unwrap()).unwrap();
    eprintln!("receipt {}", rec.display());
}

/// diag <cubins> <arch> <mynnsfc launch dir> <field> <column>: one column, one single-field direction
/// (the same seed as the sfc check), per eps: flips each side and every written output's TL and FDs.
pub fn diag(a: &[String]) {
    let (cub, arch, dir) = (PathBuf::from(&a[0]), a[1].clone(), PathBuf::from(&a[2]));
    let (j, col): (usize, usize) = (a[3].parse().unwrap(), a[4].parse().unwrap());
    let m = load_mods(&cub, "mynnsfc", &arch);
    let c = Cap::load(&dir);
    let img = Module::load(&c.image()); tables(&img);
    let n = c.scalar_i32("n") as usize;
    let s = Sfc { c: &c, m: &m, img: &img, dev: Dev::new(&c), n };
    let x0 = c.pre_f32("f"); let post = c.post_f32("f");
    let (y0, _) = s.orig(&x0, true);
    let (_, _, br0, bc0) = s.aug(&x0);
    let sx = plane_std(&x0, NF, n);
    let mut r = Rng::new(1000 + j as u64); let mut d = vec![0.0; NF * n];
    for jj in 0..NF { if !SFC_FIXED.contains(&jj) && sx[jj] > 0.0 && jj == j { for i in 0..n { let v = r.n() * sx[jj]; if sfc_interior(jj, x0[jj * n + i]) { d[jj * n + i] = v; } } } }
    let written: Vec<usize> = (0..NF).filter(|&k| (0..n).any(|i| x0[k * n + i].to_bits() != post[k * n + i].to_bits())).collect();
    println!("field {} column {} x0 {:e} d {:e} branch records {}", SFC_FIELDS[j], col, x0[j * n + col], d[j * n + col], bc0[col]);
    println!("column state: {}", (0..NF).map(|k| format!("{}={:e}", SFC_FIELDS[k], x0[k * n + col])).collect::<Vec<_>>().join(" "));
    for &eps in &[1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5] {
        let xp: Vec<f32> = x0.iter().zip(&d).map(|(&x, &v)| (x as f64 + eps * v) as f32).collect();
        let xm: Vec<f32> = x0.iter().zip(&d).map(|(&x, &v)| (x as f64 - eps * v) as f32).collect();
        let (yp, ym) = (s.orig(&xp, true).0, s.orig(&xm, true).0);
        let (_, _, brp, bcp) = s.aug(&xp); let (_, _, brm, bcm) = s.aug(&xm);
        let fp = flips(&br0, bc0[col], &brp, bcp[col], n, col).0; let fm = flips(&br0, bc0[col], &brm, bcm[col], n, col).0;
        let drc: Vec<f64> = xp.iter().zip(&xm).map(|(&a, &b)| (a as f64 - b as f64) / (2.0 * eps)).collect();
        let (tl, _) = s.tl64(&x0, &drc);
        println!("eps {eps:e} flips +{fp} -{fm} realized d {:e}", drc[j * n + col]);
        for &k in &written {
            let q = k * n + col;
            let (t, f, f1, f2) = (tl[q], (yp[q] as f64 - ym[q] as f64) / (2.0 * eps), (yp[q] as f64 - y0[q] as f64) / eps, (y0[q] as f64 - ym[q] as f64) / eps);
            if t != 0.0 || f != 0.0 { println!("   {:8} y0 {:>13.6e} tl {:>13.6e} fd {:>13.6e} fd+ {:>13.6e} fd- {:>13.6e} ulp/eps {:.1e}", SFC_FIELDS[k], y0[q], t, f, f1, f2, (y0[q].abs() * f32::EPSILON) as f64 / eps); }
        }
    }
}

/// gdiag <cubins> <arch> <ugwp launch dir> <layer:F | surface:R> <column> [dx]: one GSL column, one direction
/// (same seeds as gsl_check), per eps: flips each side, per output field the level norms of TL, FD and TL-FD.
pub fn gdiag(a: &[String]) {
    let (cub, arch, dir) = (PathBuf::from(&a[0]), a[1].clone(), PathBuf::from(&a[2]));
    let sel = a[3].clone(); let col: usize = a[4].parse().unwrap();
    let dxo: Option<f32> = a.get(5).map(|s| s.parse().unwrap());
    let m = load_mods(&cub, "ugwp", &arch);
    let c = Cap::load(&dir);
    let img = Module::load(&c.image());
    let nc = c.scalar_i32("nc") as usize; let n3 = NZ * nc;
    let g = Gsl { c: &c, m: &m, img: &img, dev: Dev::new(&c), nc };
    let (l0, p0, mut s0) = (c.pre_f32("layers"), c.pre_f32("prsi0"), c.pre_f32("surface"));
    if let Some(dx) = dxo { for i in 0..nc { s0[5 * nc + i] = dx; } }
    let (nl, np) = (l0.len(), p0.len());
    let mut x0 = l0.clone(); x0.extend(&p0); x0.extend(&s0);
    let split = |x: &[f32]| -> (Vec<f32>, Vec<f32>, Vec<f32>) { (x[..nl].to_vec(), x[nl..nl + np].to_vec(), x[nl + np..].to_vec()) };
    let run = |x: &[f32]| -> Vec<f32> { let (l, pp, s) = split(x); let (mut o, so, _) = g.orig(&l, &pp, &s, true); o.extend(so); o };
    let y0 = run(&x0);
    let (_, _, br0, bc0, _) = g.aug(&l0, &p0, &s0);
    let lstd = plane_std(&l0, 10 * NZ, nc); let sstd = plane_std(&s0, 26, nc);
    let mut d = vec![0.0; x0.len()];
    let (kind, idx) = sel.split_once(':').unwrap(); let idx: usize = idx.parse().unwrap();
    if kind == "layer" { let mut r = Rng::new(500 + idx as u64); for q in idx * NZ..(idx + 1) * NZ { for i in 0..nc { let v = r.n() * lstd[q]; if idx != 3 || l0[q * nc + i] > 0.0 { d[q * nc + i] = v; } } } }
    else { let mut r = Rng::new(530 + idx as u64); for i in 0..nc { let x = s0[idx * nc + i]; let v = r.n() * sstd[idx];
        let ok = match idx { 6 | 7 | 16 => x > 0.0, 12..=15 => x > 0.0 && x < 1.0, 2 => x > 0.0, _ => true }; if ok { d[nl + np + idx * nc + i] = v; } } }
    let gs = |r: usize| s0[r * nc + col];
    println!("{sel} column {col} dx {} var {} oc1 {} hpbl {} br {} xland {} branch records {}", gs(5), gs(6), gs(7), gs(2), gs(3), gs(4), bc0[col]);
    for &eps in &[1.0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4] {
        let xp: Vec<f32> = x0.iter().zip(&d).map(|(&x, &v)| (x as f64 + eps * v) as f32).collect();
        let xm: Vec<f32> = x0.iter().zip(&d).map(|(&x, &v)| (x as f64 - eps * v) as f32).collect();
        let (yp, ym) = (run(&xp), run(&xm));
        let (lp, pp, sp) = split(&xp); let (lm, pm, sm) = split(&xm);
        let (_, _, brp, bcp, _) = g.aug(&lp, &pp, &sp); let (_, _, brm, bcm, _) = g.aug(&lm, &pm, &sm);
        let fp = flips(&br0, bc0[col], &brp, bcp[col], nc, col).0; let fm = flips(&br0, bc0[col], &brm, bcm[col], nc, col).0;
        let drc: Vec<f64> = xp.iter().zip(&xm).map(|(&a, &b)| (a as f64 - b as f64) / (2.0 * eps)).collect();
        let tl = gtl64(&g, &l0, &p0, &s0, &drc);
        let mut line = format!("eps {eps:e} flips +{fp} -{fm} |");
        for mm in 0..12 { let (mut t2, mut f2, mut e2, mut fp2, mut fm2) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for k in 0..NZ { let q = mm * n3 + k * nc + col; let t = tl[q]; let f = (yp[q] as f64 - ym[q] as f64) / (2.0 * eps);
                let f1 = (yp[q] as f64 - y0[q] as f64) / eps; let f3 = (y0[q] as f64 - ym[q] as f64) / eps;
                t2 += t * t; f2 += f * f; e2 += (t - f) * (t - f); fp2 += (t - f1) * (t - f1); fm2 += (t - f3) * (t - f3); }
            if t2 > 0.0 || f2 > 0.0 { line += &format!(" o{mm}: tl {:.3e} fd {:.3e} err {:.1e} err+ {:.1e} err- {:.1e} |", t2.sqrt(), f2.sqrt(), e2.sqrt(), fp2.sqrt(), fm2.sqrt()); } }
        for mm in 0..10 { let q = 12 * n3 + mm * nc + col; let t = tl[q]; let f = (yp[q] as f64 - ym[q] as f64) / (2.0 * eps);
            if t != 0.0 || f != 0.0 { line += &format!(" s{mm}: tl {:.3e} fd {:.3e} |", t, f); } }
        println!("{line}");
    }
}

// ------------------------------------------------------------------- path-integral test (through switches)
// y(x0 + eps d) - y(x0 - eps d) = integral of J(x(s)) dx along the path, for any continuous function, even across
// kinks (min/max, table cells, iteration counts). The path is cut into M steps between consecutive float points;
// each step is checked with the trapezoid of the tangent at both ends on the realized step. A wrong tangent shows
// as a mismatch spread over the steps; a true jump (a discontinuous switch) shows as the mismatch in one step.
// Per column: total = |sum of step mismatches| / |dy|; smooth = the same with the two worst steps removed;
// conc = share of the step-mismatch mass in the worst step.
const PEPS: f64 = 3e-2;
const PM: usize = 48;
struct PathRes { total: Vec<f64>, smooth: Vec<f64>, conc: Vec<f64>, dyn_: Vec<f64>, free: Vec<f64>, nfree: Vec<u32> }
fn path_probe(p: &Probe, tlx: &dyn Fn(&[f32], &[f64]) -> Vec<f64>, x0: &[f32], d: &[f64], sc: &[f64]) -> PathRes {
    let n = p.n; let np = sc.len();
    let pt = |k: usize| -> Vec<f32> { let s = -PEPS + 2.0 * PEPS * (k as f64) / (PM as f64); x0.iter().zip(d).map(|(&x, &v)| (x as f64 + s * v) as f32).collect() };
    let mut xa = pt(0); let mut ya = (p.run)(&xa); let y_start = ya.clone();
    let (mut bra, mut bca) = (p.aug)(&xa);
    let (mut fe, mut fd_, mut nfree) = (vec![0.0f64; n], vec![0.0f64; n], vec![0u32; n]);
    let mut acc = vec![0.0f64; np * n];
    let mut tope = vec![[0.0f64; 2]; n]; let mut topv = vec![vec![0.0f64; np * n]; 2]; let mut esum = vec![0.0; n];
    for k in 0..PM {
        let xb = pt(k + 1); let yb = (p.run)(&xb);
        let dx: Vec<f64> = xb.iter().zip(&xa).map(|(&b, &a)| b as f64 - a as f64).collect();
        let (ta, tb) = (tlx(&xa, &dx), tlx(&xb, &dx));
        let (brb, bcb) = (p.aug)(&xb);
        let mut e = vec![0.0; n]; let mut dyk = vec![0.0; n];
        for q in 0..np { let s = sc[q]; if !(s > 0.0) { continue; } for i in 0..n { let ix = q * n + i;
            let m = (yb[ix] as f64 - ya[ix] as f64) - 0.5 * (ta[ix] + tb[ix]); if m.is_finite() { acc[ix] += m; e[i] += (m / s).powi(2); dyk[i] += ((yb[ix] as f64 - ya[ix] as f64) / s).powi(2); } } }
        for i in 0..n { if flips(&bra, bca[i], &brb, bcb[i], n, i).0 == 0 { nfree[i] += 1; fe[i] += e[i].sqrt(); fd_[i] += dyk[i].sqrt(); } }
        for i in 0..n { let ei = e[i].sqrt(); esum[i] += ei;
            if ei > tope[i][1] {
                let slot = if ei > tope[i][0] { for q in 0..np { topv[1][q * n + i] = topv[0][q * n + i]; } tope[i][1] = tope[i][0]; 0 } else { 1 };
                tope[i][slot] = ei;
                for q in 0..np { let ix = q * n + i; let m = (yb[ix] as f64 - ya[ix] as f64) - 0.5 * (ta[ix] + tb[ix]); topv[slot][ix] = if m.is_finite() { m } else { 0.0 }; }
            } }
        xa = xb; ya = yb; bra = brb; bca = bcb;
    }
    let mut r = PathRes { total: vec![0.0; n], smooth: vec![0.0; n], conc: vec![0.0; n], dyn_: vec![0.0; n], free: (0..n).map(|i| if fd_[i] > 0.0 { fe[i] / fd_[i] } else if fe[i] > 0.0 { f64::INFINITY } else { 0.0 }).collect(), nfree };
    for i in 0..n {
        let (mut dy2, mut t2, mut s2) = (0.0, 0.0, 0.0);
        for q in 0..np { let s = sc[q]; if !(s > 0.0) { continue; } let ix = q * n + i;
            let dy = ya[ix] as f64 - y_start[ix] as f64; if !dy.is_finite() { continue; }
            dy2 += (dy / s).powi(2); t2 += (acc[ix] / s).powi(2); s2 += ((acc[ix] - topv[0][ix] - topv[1][ix]) / s).powi(2); }
        r.dyn_[i] = dy2.sqrt() / (2.0 * PEPS);
        let den = dy2.sqrt().max(1e-300);
        r.total[i] = t2.sqrt() / den; r.smooth[i] = s2.sqrt() / den; r.conc[i] = if esum[i] > 0.0 { tope[i][0] / esum[i] } else { 0.0 };
    }
    r
}
fn path_summary(res: &[(String, PathRes)], cols: &[usize]) -> Value {
    let (mut tot, mut sm, mut fr) = (vec![], vec![], vec![]); let (mut weak, mut bad_s, mut bad_t, mut bad_f, mut steps, mut fsteps) = (0, 0, 0, 0, 0u64, 0u64);
    for (_, r) in res { for &i in cols { if r.dyn_[i] <= RESP_FLOOR { weak += 1; continue; }
        steps += PM as u64; fsteps += r.nfree[i] as u64; if r.nfree[i] > 0 { fr.push(r.free[i]); if r.free[i] > BAD { bad_f += 1; } }
        tot.push(r.total[i]); sm.push(r.smooth[i]); if r.smooth[i] > BAD { bad_s += 1; } if r.total[i] > BAD { bad_t += 1; } } }
    let nan = tot.iter().chain(sm.iter()).filter(|x| x.is_nan()).count();
    json!({"columns": cols.len(), "pairs": tot.len(), "weak": weak, "nan_values": nan, "total_median": jf(pct(tot.clone(), 0.5)), "total_p99": jf(pct(tot.clone(), 0.99)), "total_max": jf(pct(tot, 1.0)), "pairs_total_over_1e-2": bad_t,
           "smooth_median": jf(pct(sm.clone(), 0.5)), "smooth_p99": jf(pct(sm.clone(), 0.99)), "smooth_p999": jf(pct(sm.clone(), 0.999)), "smooth_max": jf(pct(sm, 1.0)), "pairs_smooth_over_1e-2": bad_s, "flip_free_step_share": fsteps as f64 / steps.max(1) as f64, "pairs_with_flip_free_steps": fr.len(), "free_median": jf(pct(fr.clone(), 0.5)), "free_p99": jf(pct(fr.clone(), 0.99)), "free_p999": jf(pct(fr.clone(), 0.999)), "free_max": jf(pct(fr, 1.0)), "pairs_free_over_1e-2": bad_f})
}
fn path_worst(res: &[(String, PathRes)], tags: &dyn Fn(usize) -> Vec<&'static str>, k: usize) -> Value {
    let mut v = vec![];
    for (nm, r) in res { for i in 0..r.total.len() { if r.dyn_[i] > RESP_FLOOR { if r.nfree[i] > 0 { v.push((r.free[i], nm.clone(), i, r.total[i], r.conc[i], r.dyn_[i])); } } } }
    v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    Value::Array(v.iter().take(k).map(|(s, nm, i, t, c, dn)| json!({"direction": nm, "column": i, "flip_free_step_error": jf(*s), "flip_free_steps": res.iter().find(|x| &x.0 == nm).unwrap().1.nfree[*i], "total": jf(*t), "worst_step_share": c, "response": dn, "regimes": tags(*i)})).collect())
}
fn clone_path(r: &PathRes) -> PathRes { PathRes { total: r.total.clone(), smooth: r.smooth.clone(), conc: r.conc.clone(), dyn_: r.dyn_.clone(), free: r.free.clone(), nfree: r.nfree.clone() } }

pub fn path(a: &[String]) {
    let (cub, arch, root, rec) = (PathBuf::from(&a[0]), a[1].clone(), PathBuf::from(&a[2]), PathBuf::from(&a[3]));
    let only = a.get(4).cloned().unwrap_or_default();
    let want = |k: &str| only.is_empty() || only.split(',').any(|x| x == k);
    let ms = load_mods(&cub, "mynnsfc", &arch); let mg = load_mods(&cub, "ugwp", &arch);
    let t0 = std::time::Instant::now();
    let mut out = serde_json::Map::new();
    let distinct = |k: &str, ls: &[(&str, usize)]| -> Vec<PathBuf> { ls.iter().map(|(s, l)| root.join(k).join(s).join(format!("launch-{l:06}"))).collect() };
    if want("sfc") {
        let mut v = vec![];
        for dir in distinct("mynnsfc_columns", &[("1", 0), ("1", 1), ("2", 0), ("2", 1)]) {
            let c = Cap::load(&dir); let img = ref_image(&c, "mynnsfc"); tables(&img);
            let n = c.scalar_i32("n") as usize;
            let s = Sfc { c: &c, m: &ms, img: &img, dev: Dev::new(&c), n };
            let x0 = c.pre_f32("f"); let post = c.post_f32("f");
            let sx = plane_std(&x0, NF, n); let sy = plane_std(&post, NF, n);
            let written: Vec<usize> = (0..NF).filter(|&j| (0..n).any(|i| x0[j * n + i].to_bits() != post[j * n + i].to_bits())).collect();
            let sc: Vec<f64> = (0..NF).map(|j| if written.contains(&j) { sy[j] } else { 0.0 }).collect();
            let pr = Probe { n, run: Box::new(|x| s.orig(x, true).0), aug: Box::new(|x| { let r = s.aug(x); (r.2, r.3) }), tl: Box::new(|d| s.tl64(&x0, d).0) };
            let tlx = |x: &[f32], d: &[f64]| s.tl64(x, d).0;
            let active = |j: usize| !SFC_FIXED.contains(&j) && sx[j] > 0.0;
            // znt (21) is a positive length: perturbed only where it is positive (the path stays inside its range)
            let mkd = |seed: u64, only: Option<usize>| -> Vec<f64> { let mut r = Rng::new(seed); let mut d = vec![0.0; NF * n];
                for j in 0..NF { if active(j) && only.map_or(true, |o| o == j) { for i in 0..n { let v = r.n() * sx[j]; if sfc_interior(j, x0[j * n + i]) && !(j == 21 && x0[j * n + i] <= 0.0) { d[j * n + i] = v; } } } } d };
            let mut res = vec![];
            for j in 0..NF { if active(j) { res.push((SFC_FIELDS[j].to_string(), path_probe(&pr, &tlx, &x0, &mkd(1000 + j as u64, Some(j)), &sc))); } }
            res.push(("full state".into(), path_probe(&pr, &tlx, &x0, &mkd(999, None), &sc)));
            let all: Vec<usize> = (0..n).collect();
            let g = |j: usize, i: usize| x0[j * n + i] as f64;
            let thr = c.scalar_f32("xice_threshold") as f64;
            let tags = |i: usize| -> Vec<&'static str> { let mut t = vec![];
                if g(66, i) >= thr { t.push("ice") } else if g(31, i) < 1.5 { t.push("land") } else { t.push("water") }
                if post[26 * n + i] > 0.0 { t.push("stable") } else { t.push("unstable") }
                if g(46, i) > 0.0 { t.push("snow") } if g(6, i) > 0.012 { t.push("moist") } if g(6, i) < 0.003 { t.push("dry") } t };
            let per_dir: Vec<Value> = res.iter().map(|(nm, r)| { let mut v = path_summary(std::slice::from_ref(&(nm.clone(), clone_path(r))), &all); v["direction"] = json!(nm); v }).collect();
            v.push(json!({"capture": dir, "all": path_summary(&res, &all), "per_direction": per_dir, "worst": path_worst(&res, &tags, 30)}));
            eprintln!("path sfc {} {:.0}s", dir.display(), t0.elapsed().as_secs_f64());
        }
        out.insert("mynnsfc_columns".into(), Value::Array(v));
    }
    for (key, dxo) in [("gsl", None), ("gsl3km", Some(3000.0f32))] {
        if !want(key) { continue; }
        let mut v = vec![];
        for dir in distinct("ugwp_oro_run", &[("1", 0), ("2", 0)]) {
            let c = Cap::load(&dir); let img = ref_image(&c, "ugwp");
            let nc = c.scalar_i32("nc") as usize; let n3 = NZ * nc;
            let g = Gsl { c: &c, m: &mg, img: &img, dev: Dev::new(&c), nc };
            let (l0, p0, mut s0) = (c.pre_f32("layers"), c.pre_f32("prsi0"), c.pre_f32("surface"));
            if let Some(dx) = dxo { for i in 0..nc { s0[5 * nc + i] = dx; } }
            let (nl, np) = (l0.len(), p0.len());
            let mut x0 = l0.clone(); x0.extend(&p0); x0.extend(&s0);
            let split = |x: &[f32]| -> (Vec<f32>, Vec<f32>, Vec<f32>) { (x[..nl].to_vec(), x[nl..nl + np].to_vec(), x[nl + np..].to_vec()) };
            let run = |x: &[f32]| -> Vec<f32> { let (l, pp, s) = split(x); let (mut o, so, _) = g.orig(&l, &pp, &s, true); o.extend(so); o };
            let y0 = run(&x0);
            let ostd = plane_std(&y0[..12 * n3], 12, n3); let sostd = plane_std(&y0[12 * n3..], 10, nc);
            let mut sc = vec![0.0; 12 * NZ + 10];
            for mm in 0..12 { for k in 0..NZ { sc[mm * NZ + k] = ostd[mm]; } }
            for mm in 0..10 { sc[12 * NZ + mm] = sostd[mm]; }
            let pr = Probe { n: nc, run: Box::new(|x| run(x)), aug: Box::new(|x| { let (l, pp, s) = split(x); let r = g.aug(&l, &pp, &s); (r.2, r.3) }), tl: Box::new(|d| gtl64(&g, &l0, &p0, &s0, d)) };
            let tlx = |x: &[f32], d: &[f64]| { let (l, pp, s) = split(x); gtl64(&g, &l, &pp, &s, d) };
            let lstd = plane_std(&l0, 10 * NZ, nc); let pstd = plane_std(&p0, NZ + 1, nc); let sstd = plane_std(&s0, 26, nc);
            let mkd = |seed: u64, lf: &[usize], pr_: bool, sr: &[usize]| -> Vec<f64> {
                let mut r = Rng::new(seed); let mut d = vec![0.0; x0.len()];
                for &fl in lf { for q in fl * NZ..(fl + 1) * NZ { for i in 0..nc { let v = r.n() * lstd[q]; if fl != 3 || l0[q * nc + i] > 0.0 { d[q * nc + i] = v; } } } }
                if pr_ { for q in 0..NZ + 1 { for i in 0..nc { d[nl + q * nc + i] = r.n() * pstd[q]; } } }
                for &row in sr { for i in 0..nc { let x = s0[row * nc + i]; let v = r.n() * sstd[row];
                    let ok = match row { 6 | 7 | 16 => x > 0.0, 12..=15 => x > 0.0 && x < 1.0, 2 => x > 0.0, _ => true };
                    if ok { d[nl + np + row * nc + i] = v; } } }
                d };
            let mut res = vec![];
            for (fl, nm) in ["u", "v", "T", "qv", "p", "exner"].iter().enumerate() { res.push((format!("layer {nm}"), path_probe(&pr, &tlx, &x0, &mkd(500 + fl as u64, &[fl], false, &[]), &sc))); }
            res.push(("prsi0".into(), path_probe(&pr, &tlx, &x0, &mkd(520, &[], true, &[]), &sc)));
            for (row, nm) in [(2usize, "hpbl"), (3, "bulk Rib"), (6, "var"), (7, "oc1"), (8, "oa1"), (12, "ol1")] { res.push((format!("surface {nm}"), path_probe(&pr, &tlx, &x0, &mkd(530 + row as u64, &[], false, &[row]), &sc))); }
            res.push(("full state".into(), path_probe(&pr, &tlx, &x0, &mkd(599, &[0, 1, 2, 3, 4, 5], true, &[2, 3]), &sc)));
            let all: Vec<usize> = (0..nc).collect();
            let gl = |f: usize, k: usize, i: usize| l0[f * n3 + (k - 1) * nc + i] as f64; let gs = |r: usize, i: usize| s0[r * nc + i] as f64;
            let tags = |i: usize| -> Vec<&'static str> { let mut t = vec![];
                if gs(4, i) < 1.5 { t.push("land") } else { t.push("water") }
                if gs(6, i) > 0.0 && gl(6, 1, i) - 0.5 * gl(7, 1, i) >= 1000.0 { t.push("high terrain") }
                if gs(3, i) > 0.0 { t.push("stable") } else { t.push("unstable") } t };
            let per_dir: Vec<Value> = res.iter().map(|(nm, r)| { let mut v = path_summary(std::slice::from_ref(&(nm.clone(), clone_path(r))), &all); v["direction"] = json!(nm); v }).collect();
            v.push(json!({"capture": dir, "dx_override_m": dxo, "all": path_summary(&res, &all), "per_direction": per_dir, "worst": path_worst(&res, &tags, 30)}));
            eprintln!("path {key} {} {:.0}s", dir.display(), t0.elapsed().as_secs_f64());
        }
        out.insert(format!("ugwp_oro_run{}", if dxo.is_some() { "_dx3000" } else { "" }), Value::Array(v));
    }
    out.insert("path_eps".into(), json!(PEPS)); out.insert("path_steps".into(), json!(PM));
    out.insert("wall_seconds".into(), json!(t0.elapsed().as_secs_f64()));
    std::fs::write(&rec, serde_json::to_string_pretty(&Value::Object(out)).unwrap()).unwrap();
    eprintln!("receipt {}", rec.display());
}

/// pdiag <cubins> <arch> <ugwp launch dir> <layer:F | surface:R> <column> <eps> <steps> [dx]: one GSL column along the
/// path x0 + s d, s in [-eps, eps]: per step the branch flips, |dy|, |TL trapezoid|, |mismatch| (output-std scaled).
pub fn pdiag(a: &[String]) {
    let (cub, arch, dir) = (PathBuf::from(&a[0]), a[1].clone(), PathBuf::from(&a[2]));
    let sel = a[3].clone(); let col: usize = a[4].parse().unwrap();
    let eps: f64 = a[5].parse().unwrap(); let mm: usize = a[6].parse().unwrap();
    let dxo: Option<f32> = a.get(7).map(|s| s.parse().unwrap());
    let m = load_mods(&cub, "ugwp", &arch);
    let c = Cap::load(&dir);
    let img = ref_image(&c, "ugwp");
    let nc = c.scalar_i32("nc") as usize; let n3 = NZ * nc;
    let g = Gsl { c: &c, m: &m, img: &img, dev: Dev::new(&c), nc };
    let (l0, p0, mut s0) = (c.pre_f32("layers"), c.pre_f32("prsi0"), c.pre_f32("surface"));
    if let Some(dx) = dxo { for i in 0..nc { s0[5 * nc + i] = dx; } }
    let (nl, np) = (l0.len(), p0.len());
    let mut x0 = l0.clone(); x0.extend(&p0); x0.extend(&s0);
    let split = |x: &[f32]| -> (Vec<f32>, Vec<f32>, Vec<f32>) { (x[..nl].to_vec(), x[nl..nl + np].to_vec(), x[nl + np..].to_vec()) };
    let run = |x: &[f32]| -> Vec<f32> { let (l, pp, s) = split(x); let (mut o, so, _) = g.orig(&l, &pp, &s, true); o.extend(so); o };
    let y0 = run(&x0);
    let ostd = plane_std(&y0[..12 * n3], 12, n3); let sostd = plane_std(&y0[12 * n3..], 10, nc);
    let scale = |q: usize| -> f64 { if q < 12 * n3 { ostd[q / n3] } else { sostd[(q - 12 * n3) / nc] } };
    let lstd = plane_std(&l0, 10 * NZ, nc); let sstd = plane_std(&s0, 26, nc); let pstd = plane_std(&p0, NZ + 1, nc);
    let mut d = vec![0.0; x0.len()];
    let (kind, idx) = sel.split_once(':').unwrap(); let idx: usize = idx.parse().unwrap();
    if kind == "layer" { let mut r = Rng::new(500 + idx as u64); for q in idx * NZ..(idx + 1) * NZ { for i in 0..nc { let v = r.n() * lstd[q]; if idx != 3 || l0[q * nc + i] > 0.0 { d[q * nc + i] = v; } } } }
    else if kind == "prsi0" { let mut r = Rng::new(520); for q in 0..NZ + 1 { for i in 0..nc { d[nl + q * nc + i] = r.n() * pstd[q]; } } }
    else { let mut r = Rng::new(530 + idx as u64); for i in 0..nc { let x = s0[idx * nc + i]; let v = r.n() * sstd[idx];
        let ok = match idx { 6 | 7 | 16 => x > 0.0, 12..=15 => x > 0.0 && x < 1.0, 2 => x > 0.0, _ => true }; if ok { d[nl + np + idx * nc + i] = v; } } }
    // only this column moves
    for (k, v) in d.iter_mut().enumerate() { if k % nc != col { *v = 0.0; } }
    let colq: Vec<usize> = (0..12 * NZ).map(|p| p * nc + col).chain((0..10).map(|p| 12 * n3 + p * nc + col)).collect();
    let pt = |k: usize| -> Vec<f32> { let s = -eps + 2.0 * eps * (k as f64) / (mm as f64); x0.iter().zip(&d).map(|(&x, &v)| (x as f64 + s * v) as f32).collect() };
    let tlx = |x: &[f32], dd: &[f64]| { let (l, pp, s) = split(x); gtl64(&g, &l, &pp, &s, dd) };
    let mut xa = pt(0); let mut ya = run(&xa); let ys = ya.clone();
    let (mut bra, mut bca) = { let (l, pp, s) = split(&xa); let r = g.aug(&l, &pp, &s); (r.2, r.3) };
    let (mut sdy, mut sm, mut sfree_m, mut sfree_dy) = (vec![0.0f64; colq.len()], vec![0.0f64; colq.len()], 0.0, 0.0);
    println!("{sel} column {col} eps {eps} steps {mm} dx {}", s0[5 * nc + col]);
    for k in 0..mm {
        let xb = pt(k + 1); let yb = run(&xb);
        let (brb, bcb) = { let (l, pp, s) = split(&xb); let r = g.aug(&l, &pp, &s); (r.2, r.3) };
        let fl = flips(&bra, bca[col], &brb, bcb[col], nc, col).0;
        let dx: Vec<f64> = xb.iter().zip(&xa).map(|(&b, &a)| b as f64 - a as f64).collect();
        let (ta, tb) = (tlx(&xa, &dx), tlx(&xb, &dx));
        let (mut n_dy, mut n_t, mut n_m) = (0.0, 0.0, 0.0);
        for (j, &q) in colq.iter().enumerate() { let s = scale(q); if !(s > 0.0) { continue; }
            let dy = yb[q] as f64 - ya[q] as f64; let t = 0.5 * (ta[q] + tb[q]);
            n_dy += (dy / s).powi(2); n_t += (t / s).powi(2); n_m += ((dy - t) / s).powi(2); sdy[j] += dy / s; sm[j] += (dy - t) / s; }
        if fl == 0 { sfree_m += n_m.sqrt(); sfree_dy += n_dy.sqrt(); }
        println!("step {k:3} flips {fl:3} |dy| {:.3e} |tl| {:.3e} |dy-tl| {:.3e}", n_dy.sqrt(), n_t.sqrt(), n_m.sqrt());
        xa = xb; ya = yb; bra = brb; bca = bcb;
    }
    let _ = ys;
    let ndy = sdy.iter().map(|v| v * v).sum::<f64>().sqrt(); let nm = sm.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!("total |dy| {ndy:.3e} |sum mismatch| {nm:.3e} rel {:.3e}; flip-free steps: sum|dy-tl| / sum|dy| = {:.3e}", nm / ndy.max(1e-300), sfree_m / sfree_dy.max(1e-300));
}

/// spdiag <cubins> <arch> <mynnsfc launch dir> <field> <column> <eps> <steps>: one surface-layer column along the
/// path x0 + s d (field-only direction, same seed as the sfc check): per step flips, |dy|, |TL trapezoid|, |mismatch|.
pub fn spdiag(a: &[String]) {
    let (cub, arch, dir) = (PathBuf::from(&a[0]), a[1].clone(), PathBuf::from(&a[2]));
    let (j, col): (usize, usize) = (a[3].parse().unwrap(), a[4].parse().unwrap());
    let eps: f64 = a[5].parse().unwrap(); let mm: usize = a[6].parse().unwrap();
    let m = load_mods(&cub, "mynnsfc", &arch);
    let c = Cap::load(&dir);
    let img = ref_image(&c, "mynnsfc"); tables(&img);
    let n = c.scalar_i32("n") as usize;
    let s = Sfc { c: &c, m: &m, img: &img, dev: Dev::new(&c), n };
    let x0 = c.pre_f32("f"); let post = c.post_f32("f");
    let sx = plane_std(&x0, NF, n); let sy = plane_std(&post, NF, n);
    let written: Vec<usize> = (0..NF).filter(|&k| (0..n).any(|i| x0[k * n + i].to_bits() != post[k * n + i].to_bits())).collect();
    let mut r = Rng::new(1000 + j as u64); let mut d = vec![0.0; NF * n];
    for jj in 0..NF { if !SFC_FIXED.contains(&jj) && sx[jj] > 0.0 && jj == j { for i in 0..n { let v = r.n() * sx[jj]; if sfc_interior(jj, x0[jj * n + i]) { d[jj * n + i] = v; } } } }
    for (k, v) in d.iter_mut().enumerate() { if k % n != col { *v = 0.0; } }
    let pt = |k: usize| -> Vec<f32> { let t = -eps + 2.0 * eps * (k as f64) / (mm as f64); x0.iter().zip(&d).map(|(&x, &v)| (x as f64 + t * v) as f32).collect() };
    let mut xa = pt(0); let mut ya = s.orig(&xa, true).0;
    let (_, _, mut bra, mut bca) = s.aug(&xa);
    let mut sm = vec![0.0f64; NF]; let mut sdy = vec![0.0f64; NF]; let (mut fm, mut fdy) = (0.0, 0.0);
    println!("field {} column {col} eps {eps} steps {mm}", SFC_FIELDS[j]);
    for k in 0..mm {
        let xb = pt(k + 1); let yb = s.orig(&xb, true).0;
        let (_, _, brb, bcb) = s.aug(&xb);
        let fl = flips(&bra, bca[col], &brb, bcb[col], n, col).0;
        let dx: Vec<f64> = xb.iter().zip(&xa).map(|(&b, &a)| b as f64 - a as f64).collect();
        let (ta, tb) = (s.tl64(&xa, &dx).0, s.tl64(&xb, &dx).0);
        let (mut n_dy, mut n_t, mut n_m) = (0.0, 0.0, 0.0);
        for &jj in &written { let q = jj * n + col; let sc = sy[jj]; if !(sc > 0.0) { continue; }
            let dy = yb[q] as f64 - ya[q] as f64; let t = 0.5 * (ta[q] + tb[q]);
            n_dy += (dy / sc).powi(2); n_t += (t / sc).powi(2); n_m += ((dy - t) / sc).powi(2); sdy[jj] += dy / sc; sm[jj] += (dy - t) / sc; }
        if fl == 0 { fm += n_m.sqrt(); fdy += n_dy.sqrt(); }
        println!("step {k:3} flips {fl:3} |dy| {:.3e} |tl| {:.3e} |dy-tl| {:.3e}", n_dy.sqrt(), n_t.sqrt(), n_m.sqrt());
        xa = xb; ya = yb; bra = brb; bca = bcb;
    }
    let ndy = sdy.iter().map(|v| v * v).sum::<f64>().sqrt(); let nm = sm.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!("total |dy| {ndy:.3e} |sum mismatch| {nm:.3e} rel {:.3e}; flip-free steps: sum|dy-tl| / sum|dy| = {:.3e}", nm / ndy.max(1e-300), fm / fdy.max(1e-300));
}
