// SPDX-License-Identifier: Apache-2.0
//! Proofs for the per-column TEMPO launches tempo_precip and tempo_radar:
//! binding (captured image replays its post-state), identity of the generated
//! record entry on every column, dot (all columns at once), tangent vs centred
//! differences of the unchanged captured image with path-change counts.
use crate::gpu::{self, Args, Buf, Module};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::Path;

fn read_object(root: &Path, item: &Value) -> Vec<u8> {
    let want = item["bytes"].as_u64().unwrap() as usize;
    let data = if let Some(ch) = item["chunks"].as_array() {
        let mut o = Vec::with_capacity(want);
        for c in ch { o.extend(read_object(root, c)); }
        o
    } else { std::fs::read(root.join(item["path"].as_str().unwrap())).expect("object") };
    assert_eq!(data.len(), want);
    assert_eq!(format!("{:x}", Sha256::digest(&data)), item["sha256"].as_str().unwrap());
    data
}
struct Rng(u64);
impl Rng {
    fn normal(&mut self) -> f64 {
        let mut u = || { self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15); let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9); z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB); z ^= z >> 31;
            ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64 };
        let (a, b) = (u(), u()); (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
    }
}

/// One slot group: (table argument or direct pointer, index in the table, per level?, direction scale kind)
#[derive(Clone)]
struct Slot { arg: &'static str, j: Option<usize>, lev: bool, dir: f64, rel: bool }

pub fn run(a: &[String]) {
    let arg = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    let dir = arg("--capture").expect("--capture");
    let out = arg("--out").expect("--out");
    let eps_list: Vec<f64> = arg("--eps").unwrap_or("1,0.3,0.1,0.03,0.01,0.003,0.001,0.0003,0.0001".into()).split(',').map(|s| s.parse().unwrap()).collect();
    gpu::init();
    let cubin = std::fs::read(arg("--cubin").expect("--cubin")).unwrap();
    let gm = Module::load(&cubin);
    let (g_ad, gn) = gm.global("g_ad"); assert_eq!(gn, 104);
    let sz = Buf::new(24); Args::new().ptr(sz.p).launch(gm.func("tempo_ad_sizes"), 1, 1);
    let prim = sz.get::<u64>(3)[2] as usize; // 8: float64-promoted grade build
    let m: Value = serde_json::from_slice(&std::fs::read(Path::new(&dir).join("manifest.json")).unwrap()).unwrap();
    let kernel = m["kernel"].as_str().unwrap().to_string();
    let root = Path::new(&dir);
    let allocs = m["allocations"].as_array().unwrap();
    let mut pre: Vec<Vec<u8>> = allocs.iter().map(|x| read_object(root, &x["pre"])).collect();
    let mut post: Vec<Vec<u8>> = allocs.iter().map(|x| read_object(root, &x["post"])).collect();
    let dev: Vec<Buf> = pre.iter().map(|v| Buf::new(v.len())).collect();
    for ip in m["indirect_pointers"].as_array().unwrap() {
        let (ta, to, tg, tgo) = (ip["table_allocation"].as_u64().unwrap() as usize, ip["table_offset"].as_u64().unwrap() as usize,
                                 ip["target_allocation"].as_u64().unwrap() as usize, ip["target_offset"].as_u64().unwrap());
        let addr = (dev[tg].p + tgo).to_le_bytes();
        pre[ta][to..to + 8].copy_from_slice(&addr); post[ta][to..to + 8].copy_from_slice(&addr);
    }
    for (i, v) in pre.iter().enumerate() { dev[i].put(v); }
    let argv = |n: &str| m["arguments"].as_array().unwrap().iter().find(|x| x["name"] == n).unwrap().clone();
    let sc = |n: &str| { let h = argv(n)["value_hex"].as_str().unwrap().to_string(); let mut b = [0u8; 4]; for i in 0..4 { b[i] = u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap(); } u32::from_le_bytes(b) };
    let nz = sc("nz") as usize; let nc = sc("nc") as usize;
    let alloc_of = |n: &str| argv(n)["allocation"].as_u64().unwrap() as usize;
    // (allocation, byte offset) of table entry j of argument n, or of the direct pointer argument.
    let target = |n: &str, j: Option<usize>| -> Option<(usize, usize)> {
        let ta = alloc_of(n);
        match j {
            None => Some((ta, argv(n)["offset"].as_u64().unwrap() as usize)),
            Some(j) => m["indirect_pointers"].as_array().unwrap().iter()
                .find(|ip| ip["table_allocation"].as_u64().unwrap() as usize == ta && ip["table_offset"].as_u64().unwrap() as usize == 8 * j)
                .map(|ip| (ip["target_allocation"].as_u64().unwrap() as usize, ip["target_offset"].as_u64().unwrap() as usize)),
        }
    };
    let s = |arg: &'static str, j: Option<usize>, lev: bool, dir: f64, rel: bool| Slot { arg, j, lev, dir, rel };
    let (ins, outs, entry): (Vec<Slot>, Vec<Slot>, &str) = if kernel == "tempo_precip" {
        let mut i = vec![s("f", Some(3), true, 0.0, false), s("f", Some(4), true, 0.05, true), s("f", Some(18), true, 0.01, true)];
        for j in 0..10 { i.push(s("s", Some(j), false, 0.1, true)); }
        for j in 0..13 { i.push(s("d", Some(j), false, 0.1, true)); }
        let mut o = vec![]; for j in 0..10 { o.push(s("s", Some(j), false, 0.0, false)); } for j in 0..13 { o.push(s("d", Some(j), false, 0.0, false)); }
        (i, o, "tempo_ad_precip_record")
    } else if kernel == "tempo_radar" {
        let mut i = vec![s("f", Some(0), true, 0.5, false), s("f", Some(1), true, 0.0, false), s("f", Some(3), true, 0.0, false),
                         s("f", Some(4), true, 0.05, true), s("f", Some(17), true, 0.0, false), s("f", Some(19), true, 0.0, false),
                         s("profile", Some(0), true, 1.0, false)];
        for j in 0..3 { i.push(s("radar", Some(j), false, 1.0, false)); }
        let mut o = vec![]; for j in 0..3 { o.push(s("radar", Some(j), false, 0.0, false)); } o.push(s("rh", None, true, 0.0, false));
        (i, o, "tempo_ad_radar_record")
    } else { panic!("unsupported kernel {kernel}") };
    let width = |v: &Vec<Slot>| v.iter().map(|x| if x.lev { nz } else { 1 }).sum::<usize>();
    let (nin, nout) = (width(&ins), width(&outs));
    let word = |h: &[Vec<u8>], t: (usize, usize), i: usize| f32::from_le_bytes(h[t.0][t.1 + 4 * i..t.1 + 4 * i + 4].try_into().unwrap());
    let gather = |h: &[Vec<u8>], v: &Vec<Slot>, c: usize| -> Vec<f32> {
        let mut x = vec![];
        for sl in v { let tg = target(sl.arg, sl.j); for k in 0..(if sl.lev { nz } else { 1 }) {
            x.push(tg.map(|t| word(h, t, if sl.lev { k * nc + c } else { c })).unwrap_or(0.0)); } }
        x
    };
    let scatter = |h: &mut [Vec<u8>], c: usize, x: &[f32]| {
        let mut q = 0;
        for sl in &ins { let tg = target(sl.arg, sl.j); for k in 0..(if sl.lev { nz } else { 1 }) {
            if let Some(t) = tg { let i = t.1 + 4 * (if sl.lev { k * nc + c } else { c }); h[t.0][i..i + 4].copy_from_slice(&x[q].to_le_bytes()); } q += 1; } }
    };
    // --------------------------------------------------------- A. binding
    let img = std::fs::read(root.join(m["image"]["path"].as_str().unwrap())).unwrap();
    let fwd = Module::load(&img); let ff = fwd.func(&kernel);
    let run_fwd = || {
        let mut ar = Args::new();
        for x in m["arguments"].as_array().unwrap() {
            if x["kind"] == "scalar" { let h = x["value_hex"].as_str().unwrap(); ar = ar.raw((0..h.len() / 2).map(|i| u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap()).collect()); }
            else { ar = ar.ptr(dev[x["allocation"].as_u64().unwrap() as usize].p + x["offset"].as_u64().unwrap()); }
        }
        let t = std::time::Instant::now();
        ar.launch(ff, m["geometry"]["grid"][0].as_u64().unwrap() as u32, m["geometry"]["block"][0].as_u64().unwrap() as u32);
        t.elapsed().as_secs_f64()
    };
    let fwd_s = run_fwd();
    let mut bdiff = 0usize;
    let got: Vec<Vec<u8>> = dev.iter().map(|b| b.get::<u8>(b.n)).collect();
    for i in 0..dev.len() { bdiff += got[i].chunks(4).zip(post[i].chunks(4)).filter(|(x, y)| x != y).count(); }
    eprintln!("[A] {kernel}: captured image replay {bdiff} differing words");
    // --------------------------------------------------------- record on every column
    let cols: Vec<i32> = (0..nc as i32).collect();
    let n = nc;
    let colsb = Buf::from(&cols); let buf = Buf::new(16 * (nin + nz) * n);
    let yout = Buf::new(prim * nout * n); let oidx = Buf::new(4 * nout * n); let cout = Buf::new(4 * n); let cnt = Buf::new(16 * n);
    let rec = gm.func(entry);
    let record = |cap: usize, ta: &Buf, tb: &Buf, tpa: &Buf, tpb: &Buf, br: &Buf, bcap: usize, xpert: u64, rbr: u64| -> f64 {
        let mut b = vec![];
        for p in [ta.p, tb.p, tpa.p, tpb.p, br.p, br.p, cnt.p, rbr, 0] { b.extend(p.to_le_bytes()); }
        for v in [n as i64, cap as i64, bcap as i64, 0] { b.extend(v.to_le_bytes()); }
        let gb = std::mem::ManuallyDrop::new(Buf { p: g_ad, n: b.len() }); gb.put(&b);
        br.zero();
        let mut ar = Args::new();
        for x in m["arguments"].as_array().unwrap() {
            let nm = x["name"].as_str().unwrap();
            if nm == "rh" { continue; }
            if x["kind"] == "scalar" { let h = x["value_hex"].as_str().unwrap(); ar = ar.raw((0..h.len() / 2).map(|i| u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap()).collect()); }
            else { ar = ar.ptr(dev[x["allocation"].as_u64().unwrap() as usize].p + x["offset"].as_u64().unwrap()); }
        }
        let t = std::time::Instant::now();
        ar.ptr(colsb.p).i32(n as i32).ptr(buf.p).ptr(yout.p).ptr(oidx.p).ptr(cout.p).ptr(xpert).launch(rec, ((n + 63) / 64) as u32, 64);
        t.elapsed().as_secs_f64()
    };
    let z = Buf::new(8);
    for (i, v) in pre.iter().enumerate() { dev[i].put(v); }
    record(0, &z, &z, &z, &z, &z, 0, 0, 0);
    let c4: Vec<i32> = cnt.get(4 * n);
    let capn = (0..n).map(|t| c4[4 * t] as usize).max().unwrap() + 1;
    let bmax = (0..n).map(|t| c4[4 * t + 1] as usize).max().unwrap() + 64;
    let (ta, tb, tpa, tpb) = (Buf::new(4 * capn * n), Buf::new(4 * capn * n), Buf::new(8 * capn * n), Buf::new(8 * capn * n));
    let br = Buf::new(4 * ((bmax + 31) / 32) * n);
    let rec_s = record(capn, &ta, &tb, &tpa, &tpb, &br, bmax, 0, 0);
    let base_bits: Vec<u32> = br.get((bmax + 31) / 32 * n);
    let c4: Vec<i32> = cnt.get(4 * n);
    // identity
    let y: Vec<f32> = if prim == 8 { yout.get::<f64>(nout * n).iter().map(|&v| v as f32).collect() } else { yout.get(nout * n) };
    let mut idiff = 0usize; let mut icmp = 0usize;
    let full = kernel != "tempo_radar" || sc("full") == 1;
    for c in 0..n {
        let want = gather(&post, &outs, c);
        for o in 0..nout { if kernel == "tempo_radar" && !full && o >= 3 { continue; } icmp += 1; if y[o * n + c].to_bits() != want[o].to_bits() { idiff += 1; } }
    }
    let mut cdiff = 0usize;
    if kernel == "tempo_precip" {
        let cc: Vec<i32> = cout.get(n); let ct = target("counter", None).unwrap();
        for c in 0..n { if cc[c] != i32::from_le_bytes(post[ct.0][ct.1 + 4 * c..ct.1 + 4 * c + 4].try_into().unwrap()) { cdiff += 1; } }
    }
    eprintln!("[B] identity: {idiff} of {icmp} output words differ (counter {cdiff} of {n}); nodes per column max {}", capn - 1);
    // dot
    let tl = gm.func("tempo_ad_tl"); let adj = gm.func("tempo_ad_adj");
    let sweep = |f, nin_: usize, inp: &[f64], outn: usize| -> Vec<f64> {
        let ib = Buf::from(inp); let ob = Buf::new(8 * outn * n); let scr = Buf::new(8 * capn * n);
        Args::new().i32(n as i32).i32(nin_ as i32).i32(nout as i32).ptr(ib.p).ptr(scr.p).ptr(oidx.p).ptr(ob.p).launch(f, ((n + 63) / 64) as u32, 64);
        ob.get(outn * n)
    };
    let mut rows = vec![]; let mut worst = 0f64;
    for seed in 1..=6u64 {
        let mut g = Rng(seed * 7919);
        let xd: Vec<f64> = (0..nin * n).map(|_| g.normal()).collect(); let yb: Vec<f64> = (0..nout * n).map(|_| g.normal()).collect();
        let yd = sweep(tl, nin, &xd, nout); let xb = sweep(adj, nin, &yb, nin);
        let l: f64 = (0..nout * n).map(|i| yd[i] * yb[i]).sum(); let r: f64 = (0..nin * n).map(|i| xd[i] * xb[i]).sum();
        let rel = (l - r).abs() / l.abs().max(r.abs()); worst = worst.max(rel); rows.push(json!({"seed": seed, "lhs": l, "rhs": r, "rel": rel}));
        if seed == 1 { let xb2 = sweep(adj, nin, &yb, nin); assert!(xb2.iter().zip(&xb).all(|(a, b)| a.to_bits() == b.to_bits()), "reverse not repeatable"); }
    }
    let yz = sweep(tl, nin, &vec![0f64; nin * n], nout);
    let zero = yz.iter().all(|v| *v == 0.0);
    eprintln!("[C] dot worst {worst:.2e} (6 seeds, all {n} columns), zero direction exact {zero}, reverse repeat bitwise true");
    // direction and finite differences of the unchanged captured image
    let mut g = Rng(7);
    let mut dvec = vec![0f64; nin * n];
    let xin: Vec<Vec<f32>> = (0..n).map(|c| gather(&pre, &ins, c)).collect();
    for c in 0..n { let mut q = 0; for sl in &ins { for _ in 0..(if sl.lev { nz } else { 1 }) {
        let r = g.normal();
        // Bounded reflectivity fields: no additive perturbation at the -35 dBZ floor (a tie of every max there).
        let at_floor = (sl.arg == "profile" || sl.arg == "radar") && xin[c][q] <= -34.99;
        dvec[q * n + c] = if at_floor { 0.0 } else if sl.rel { sl.dir * r * xin[c][q] as f64 } else { sl.dir * r }; q += 1; } } }
    let ytl = sweep(tl, nin, &dvec, nout);
    let mut curve = vec![]; let mut changed = vec![];
    if prim == 8 {
        // float64 grade: branch-held centred differences of the promoted record entry, perturbation in binary64
        let base = Buf::from(&base_bits); let y0: Vec<f64> = yout.get(nout * n);
        let mut held = vec![];
        for &eps in &eps_list {
            let mut ys = vec![];
            for sgn in [1.0, -1.0] {
                let xp: Vec<f64> = dvec.iter().map(|d| sgn * eps * d).collect(); let xb = Buf::from(&xp);
                let brh = Buf::new(4 * ((bmax + 31) / 32) * n);
                record(0, &z, &z, &z, &z, &brh, bmax, xb.p, base.p);
                ys.push(yout.get::<f64>(nout * n));
            }
            let (mut num, mut den) = (0f64, 0f64);
            for i in 0..nout * n { if !y0[i].is_finite() || !ys[0][i].is_finite() || !ys[1][i].is_finite() { continue; }
                let fd = (ys[0][i] - ys[1][i]) / (2.0 * eps); num += (fd - ytl[i]).powi(2); den += ytl[i].powi(2).max(fd.powi(2)); }
            held.push(if den > 0.0 { (num / den).sqrt() } else { 0.0 });
        }
        let best = held.iter().cloned().fold(f64::INFINITY, f64::min);
        eprintln!("[P] float64 grade, branch-held, all outputs, eps {:?}: {:?} (best {:.2e})", eps_list, held.iter().map(|v| format!("{:.1e}", v)).collect::<Vec<_>>(), best);
        let r = json!({"schema": "ad.phys.tempo-small-proof-f64/v1", "kernel": kernel, "capture": dir, "cubin_sha256": format!("{:x}", Sha256::digest(&cubin)),
            "identity_against_float_forward_words_differing": idiff, "dot_worst_rel": worst, "branch_held_curve": held, "best": best, "eps": eps_list});
        std::fs::write(&out, serde_json::to_string_pretty(&r).unwrap()).unwrap();
        return;
    }
    for &eps in &eps_list {
        let mut ys = vec![]; let mut pc = 0usize;
        for sgn in [1.0, -1.0] {
            let mut h = pre.clone();
            for c in 0..n { let x: Vec<f32> = (0..nin).map(|q| (xin[c][q] as f64 + sgn * eps * dvec[q * n + c]) as f32).collect(); scatter(&mut h, c, &x); }
            for (i, v) in h.iter().enumerate() { dev[i].put(v); }
            record(0, &z, &z, &z, &z, &br, bmax, 0, 0);
            let pb: Vec<u32> = br.get((bmax + 31) / 32 * n); let pc4: Vec<i32> = cnt.get(4 * n);
            for c in 0..n { let nb = c4[4 * c + 1] as usize; let diff = pc4[4 * c + 1] != c4[4 * c + 1] || (0..(nb + 31) / 32).any(|w| {
                let mask = if (w + 1) * 32 <= nb { u32::MAX } else { (1u32 << (nb - 32 * w)) - 1 }; (pb[w * n + c] ^ base_bits[w * n + c]) & mask != 0 }); if diff { pc += 1; } }
            for (i, v) in h.iter().enumerate() { dev[i].put(v); }
            run_fwd();
            let hp: Vec<Vec<u8>> = dev.iter().map(|b| b.get::<u8>(b.n)).collect();
            ys.push((0..n).map(|c| gather(&hp, &outs, c)).collect::<Vec<_>>());
        }
        let (mut num, mut den) = (0f64, 0f64);
        for c in 0..n { for o in 0..nout { if kernel == "tempo_radar" && !full && o >= 3 { continue; }
            let fd = (ys[0][c][o] as f64 - ys[1][c][o] as f64) / (2.0 * eps); let t = ytl[o * n + c];
            num += (fd - t).powi(2); den += t.powi(2).max(fd.powi(2)); } }
        curve.push(if den > 0.0 { (num / den).sqrt() } else { 0.0 }); changed.push(pc);
    }
    for (i, v) in pre.iter().enumerate() { dev[i].put(v); }
    let best = curve.iter().cloned().fold(f64::INFINITY, f64::min);
    eprintln!("[D] tangent vs forward differences, all outputs, eps {:?}: {:?} (best {:.2e}); columns whose path changed: {:?} of {n}",
              eps_list, curve.iter().map(|v| format!("{:.1e}", v)).collect::<Vec<_>>(), best, changed);
    let r = json!({"schema": "ad.phys.tempo-small-proof/v1", "kernel": kernel, "capture": dir, "cubin_sha256": format!("{:x}", Sha256::digest(&cubin)),
        "binding_differing_words": bdiff, "identity": {"differing_words": idiff, "compared_words": icmp, "counter_differing_columns": cdiff},
        "tape_nodes_per_column_max": capn - 1, "branch_bits_per_column_max": bmax - 64,
        "dot": {"rows": rows, "worst_rel": worst, "zero_direction_exact": zero, "reverse_repeat_bitwise": true},
        "taylor": {"eps": eps_list, "curve": curve, "best": best, "path_changed_columns": changed,
                   "direction": "per-kernel realistic sizes (see small.rs slots); reflectivity values at the -35 dBZ floor held"},
        "cost": {"forward_launch_ms": fwd_s * 1e3, "record_with_tape_ms": rec_s * 1e3, "columns": n}});
    std::fs::write(&out, serde_json::to_string_pretty(&r).unwrap()).unwrap();
    eprintln!("receipt {out}");
}
