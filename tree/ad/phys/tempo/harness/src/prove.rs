// SPDX-License-Identifier: Apache-2.0
//! Proofs on one or more captured tempo_warp_steps launches:
//!  A. binding: the captured forward image replays its captured post-state;
//!  B. identity: the generated record entry (augmented primal) gives the
//!     captured outputs bit for bit on every column;
//!  C. dot: <J dx, w> = <dx, J^T w> per column, all outputs and per output
//!     field, zero direction, two reverse runs bitwise;
//!  D. tangent vs centred finite differences of the UNCHANGED captured image,
//!     per output field, realistic all-field directions, with branch and
//!     integer-log (table cell / count) flips per column;
//!  E. water budget of the tangent (vapour + condensate + precipitation);
//!  F. cost against the forward.
use crate::gpu::{self, Args, Buf, Module};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Rng { Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xD1B54A32D192ED03) }
    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn unif(&mut self) -> f64 { ((self.u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64 }
    fn normal(&mut self) -> f64 { let (a, b) = (self.unif(), self.unif()); (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos() }
}

const NAMES_F: [&str; 18] = ["theta", "p", "w", "dz", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc", "nwfa", "nifa", "ng", "qb", "exner"];
const NAMES_S: [&str; 10] = ["precip_step", "precip_total", "snow_ice_step", "snow_ice_total", "graupel_step", "graupel_total",
                             "frozen_fraction", "freezing_rain", "hail_diam_sfc", "hail_diam_max"];
const NAMES_P: [&str; 4] = ["refl10cm", "re_cloud", "re_ice", "re_snow"];

fn read_object(root: &Path, item: &Value) -> Vec<u8> {
    let want = item["bytes"].as_u64().unwrap() as usize;
    let data = if let Some(ch) = item["chunks"].as_array() {
        assert_eq!(item["encoding"], "raw-concat-v1");
        let mut o = Vec::with_capacity(want);
        for c in ch { o.extend(read_object(root, c)); }
        o
    } else {
        std::fs::read(root.join(item["path"].as_str().unwrap())).expect("object")
    };
    assert_eq!(data.len(), want, "object length");
    assert_eq!(format!("{:x}", Sha256::digest(&data)), item["sha256"].as_str().unwrap(), "object digest");
    data
}

struct Cap {
    dir: PathBuf,
    m: Value,
    pre: Vec<Vec<u8>>,   // host pre images, pointer tables patched to our device addresses
    post: Vec<Vec<u8>>,  // captured post images (tables patched the same way)
    dev: Vec<Buf>,
    blob_alloc: usize,
    cost_alloc: usize,
    table_allocs: Vec<usize>,
    scratch: Vec<usize>,
    nz: usize, columns: usize, steps: i32, dt: f32,
    flags: [i32; 7], // aero hail ml_bl melt first presence radii
    f_t: Vec<Option<(usize, usize)>>, s_t: Vec<Option<(usize, usize)>>, p_t: Vec<Option<(usize, usize)>>,
    f_tab: usize, s_tab: usize, p_tab: usize,
}

fn scalar_u32(m: &Value, name: &str) -> u32 {
    let a = m["arguments"].as_array().unwrap().iter().find(|x| x["name"] == name).unwrap();
    let h = a["value_hex"].as_str().unwrap();
    let mut b = [0u8; 4];
    for i in 0..4 { b[i] = u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap(); }
    u32::from_le_bytes(b)
}
fn arg_alloc(m: &Value, name: &str) -> usize {
    let a = m["arguments"].as_array().unwrap().iter().find(|x| x["name"] == name).unwrap();
    assert_eq!(a["offset"], 0); a["allocation"].as_u64().unwrap() as usize
}

impl Cap {
    fn load(dir: &Path) -> Cap {
        let m: Value = serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(m["kernel"], "tempo_warp_steps"); assert_eq!(m["complete"], true);
        let root = dir.to_path_buf(); // objects/ in the launch folder links to the capture store
        let allocs = m["allocations"].as_array().unwrap();
        let t = Instant::now();
        // Scratch allocations (warp state and work areas) are written before they are read by every column;
        // a capture whose scratch object was swept loads them as zeros and leaves them out of the binding check.
        let scratch: Vec<usize> = ["states", "work", "blwork"].iter().map(|n| arg_alloc(&m, n)).collect();
        let mut missing_scratch = vec![];
        let mut load = |i: usize, a: &Value, k: &str| -> Vec<u8> {
            let it = &a[k];
            let p = it["path"].as_str().map(|p| root.join(p));
            if scratch.contains(&i) && p.as_ref().map(|p| !p.exists()).unwrap_or(false) {
                missing_scratch.push(json!([i, k])); return vec![0u8; it["bytes"].as_u64().unwrap() as usize];
            }
            read_object(&root, it)
        };
        let mut pre: Vec<Vec<u8>> = allocs.iter().enumerate().map(|(i, a)| load(i, a, "pre")).collect();
        let mut post: Vec<Vec<u8>> = allocs.iter().enumerate().map(|(i, a)| load(i, a, "post")).collect();
        eprintln!("[load] scratch objects absent from the store (zero-filled, not compared): {:?}", missing_scratch);
        eprintln!("[load] {} allocations, {:.1} GB, {:.1} s", pre.len(), pre.iter().map(|v| v.len()).sum::<usize>() as f64 / 1e9, t.elapsed().as_secs_f64());
        let dev: Vec<Buf> = pre.iter().map(|v| Buf::new(v.len())).collect();
        let mut table_allocs = vec![];
        for ip in m["indirect_pointers"].as_array().unwrap() {
            let ta = ip["table_allocation"].as_u64().unwrap() as usize;
            let to = ip["table_offset"].as_u64().unwrap() as usize;
            let tg = ip["target_allocation"].as_u64().unwrap() as usize;
            let tgo = ip["target_offset"].as_u64().unwrap();
            let addr = (dev[tg].p + tgo).to_le_bytes();
            pre[ta][to..to + 8].copy_from_slice(&addr);
            post[ta][to..to + 8].copy_from_slice(&addr);
            if !table_allocs.contains(&ta) { table_allocs.push(ta); }
        }
        let f_tab = arg_alloc(&m, "f"); let s_tab = arg_alloc(&m, "s"); let p_tab = arg_alloc(&m, "profiles");
        let targets = |tab: usize, n: usize| -> Vec<Option<(usize, usize)>> {
            (0..n).map(|j| m["indirect_pointers"].as_array().unwrap().iter()
                .find(|ip| ip["table_allocation"].as_u64().unwrap() as usize == tab && ip["table_offset"].as_u64().unwrap() as usize == 8 * j)
                .map(|ip| (ip["target_allocation"].as_u64().unwrap() as usize, ip["target_offset"].as_u64().unwrap() as usize))).collect()
        };
        let f_t = targets(f_tab, 20); let s_t = targets(s_tab, 12); let p_t = targets(p_tab, 4);
        // A null slot must really be zero in the captured table.
        for (tab, tt) in [(f_tab, &f_t), (s_tab, &s_t), (p_tab, &p_t)] {
            for (j, x) in tt.iter().enumerate() {
                if x.is_none() { assert!(pre[tab][8 * j..8 * j + 8].iter().all(|&b| b == 0), "unlisted nonzero pointer slot {j} in table {tab}"); }
            }
        }
        let nz = scalar_u32(&m, "nz") as usize; let columns = scalar_u32(&m, "columns") as usize;
        let steps = scalar_u32(&m, "steps") as i32; let dt = f32::from_bits(scalar_u32(&m, "dt"));
        let fl = |n| scalar_u32(&m, n) as i32;
        let flags = [fl("aero"), fl("hail"), fl("ml_bl"), fl("melt"), fl("first"), fl("presence"), fl("radii")];
        let bl = m["arguments"].as_array().unwrap().iter().find(|x| x["name"] == "bl").unwrap();
        assert!(bl["kind"] == "scalar" && bl["value_hex"] == "0000000000000000", "bl present: the subgrid-radii path is not in this entry");
        assert_eq!(flags[2], 0, "ml_bl");
        let c = Cap { dir: dir.to_path_buf(), blob_alloc: arg_alloc(&m, "blob"), cost_alloc: arg_alloc(&m, "cost"), scratch, m, pre, post, dev, table_allocs,
                      nz, columns, steps, dt, flags, f_t, s_t, p_t, f_tab, s_tab, p_tab };
        for (i, v) in c.pre.iter().enumerate() { c.dev[i].put(v); }
        c
    }
    /// Restore every allocation the kernel may write (the 1 GB table blob is read-only and stays).
    fn restore(&self, host: &[Vec<u8>]) { for (i, v) in host.iter().enumerate() { if i != self.blob_alloc { self.dev[i].put(v); } } }
    fn run_captured(&self, f: *mut std::ffi::c_void) {
        let mut a = Args::new();
        for x in self.m["arguments"].as_array().unwrap() {
            if x["kind"] == "scalar" {
                let h = x["value_hex"].as_str().unwrap();
                a = a.raw((0..h.len() / 2).map(|i| u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap()).collect());
            } else {
                a = a.ptr(self.dev[x["allocation"].as_u64().unwrap() as usize].p + x["offset"].as_u64().unwrap());
            }
        }
        let g = &self.m["geometry"];
        a.launch(f, g["grid"][0].as_u64().unwrap() as u32, g["block"][0].as_u64().unwrap() as u32);
    }
    fn word(v: &[u8], off: usize, i: usize) -> f32 { f32::from_le_bytes(v[off + 4 * i..off + 4 * i + 4].try_into().unwrap()) }
    /// Inputs of column c, slot order (18*nz fields, then 10 s).
    fn inputs(&self, host: &[Vec<u8>], c: usize) -> Vec<f32> {
        let (nz, nc) = (self.nz, self.columns);
        let mut x = vec![0f32; 18 * nz + 10];
        for j in 0..18 { if let Some((a, o)) = self.f_t[j] { for k in 0..nz { x[j * nz + k] = Self::word(&host[a], o, k * nc + c); } } }
        for j in 0..10 { if let Some((a, o)) = self.s_t[j] { x[18 * nz + j] = Self::word(&host[a], o, c); } }
        x
    }
    fn set_inputs(&self, host: &mut [Vec<u8>], c: usize, x: &[f32]) {
        let (nz, nc) = (self.nz, self.columns);
        for j in 0..18 { if let Some((a, o)) = self.f_t[j] { for k in 0..nz { let i = o + 4 * (k * nc + c); host[a][i..i + 4].copy_from_slice(&x[j * nz + k].to_le_bytes()); } } }
        for j in 0..10 { if let Some((a, o)) = self.s_t[j] { let i = o + 4 * c; host[a][i..i + 4].copy_from_slice(&x[18 * nz + j].to_le_bytes()); } }
    }
    /// Outputs of column c from a post image, slot order (f0, f4..16, s0..9, profiles 0..3); NaN for unwritten.
    fn outputs(&self, host: &[Vec<u8>], c: usize) -> Vec<f32> {
        let (nz, nc) = (self.nz, self.columns);
        let mut y = vec![f32::NAN; 18 * nz + 10];
        let fo = |q: usize| if q == 0 { 0 } else { q + 3 };
        for q in 0..14 { if let Some((a, o)) = self.f_t[fo(q)] { for k in 0..nz { y[q * nz + k] = Self::word(&host[a], o, k * nc + c); } } }
        for j in 0..10 { if let Some((a, o)) = self.s_t[j] { y[14 * nz + j] = Self::word(&host[a], o, c); } }
        for p in 0..4 {
            if p > 0 && self.flags[6] == 0 { continue; }
            if let Some((a, o)) = self.p_t[p] { for k in 0..nz { y[14 * nz + 10 + p * nz + k] = Self::word(&host[a], o, k * nc + c); } }
        }
        y
    }
    fn download_outputs(&self) -> Vec<Vec<u8>> {
        // Only the output targets are needed; other allocations stay empty.
        let mut h: Vec<Vec<u8>> = self.pre.iter().map(|_| vec![]).collect();
        for t in self.f_t.iter().chain(self.s_t.iter()).chain(self.p_t.iter()).flatten() {
            if h[t.0].is_empty() { h[t.0] = self.dev[t.0].get::<u8>(self.dev[t.0].n); }
        }
        h
    }
}

/// Output groups: (name, first slot, count).
fn groups(nz: usize) -> Vec<(String, usize, usize)> {
    let mut g = vec![];
    let fo = |q: usize| if q == 0 { 0 } else { q + 3 };
    for q in 0..14 { g.push((NAMES_F[fo(q)].to_string(), q * nz, nz)); }
    for j in 0..10 { g.push((NAMES_S[j].to_string(), 14 * nz + j, 1)); }
    for p in 0..4 { g.push((NAMES_P[p].to_string(), 14 * nz + 10 + p * nz, nz)); }
    g
}

struct Gen { m: Module, rec: *mut std::ffi::c_void, tl: *mut std::ffi::c_void, adj: *mut std::ffi::c_void, g: u64, work_bytes: usize, state_bytes: usize, prim: usize }
impl Gen {
    fn load(cubin: &[u8]) -> Gen {
        let m = Module::load(cubin);
        let (g, gn) = m.global("g_ad"); assert_eq!(gn, 104, "AdTape layout");
        let sizes = Buf::new(24);
        Args::new().ptr(sizes.p).launch(m.func("tempo_ad_sizes"), 1, 1);
        let s: Vec<u64> = sizes.get(3);
        let rec = m.func("tempo_ad_record");
        eprintln!("[gen] record entry: local bytes/thread {}, registers {}, max threads/block {}", gpu::func_attr(rec, 3), gpu::func_attr(rec, 4), gpu::func_attr(rec, 0));
        Gen { rec: m.func("tempo_ad_record"), tl: m.func("tempo_ad_tl"), adj: m.func("tempo_ad_adj"), m, g, work_bytes: s[0] as usize, state_bytes: s[1] as usize, prim: s[2] as usize }
    }
}

struct Tile { cols: Vec<usize>, n: usize, cap: usize, bcap: usize, icap: usize,
              ta: Buf, tb: Buf, tpa: Buf, tpb: Buf, br: Buf, il: Buf, cnt: Buf, colsb: Buf, st: Buf, wk: Buf, yout: Buf, oidx: Buf, sc: Buf,
              rbr: Option<Buf>, ril: Option<Buf> }
impl Tile {
    fn new(gen: &Gen, cols: &[usize], nin: usize, cap: usize, bcap: usize, icap: usize) -> Tile {
        let n = cols.len();
        let c32: Vec<i32> = cols.iter().map(|&c| c as i32).collect();
        Tile { cols: cols.to_vec(), n, cap, bcap, icap,
               ta: Buf::new(4 * cap * n), tb: Buf::new(4 * cap * n), tpa: Buf::new(8 * cap * n), tpb: Buf::new(8 * cap * n),
               br: Buf::new(4 * ((bcap + 31) / 32) * n), il: Buf::new(4 * icap * n), cnt: Buf::new(16 * n), colsb: Buf::from(&c32),
               st: Buf::new(gen.state_bytes * n), wk: Buf::new(gen.work_bytes * n), yout: Buf::new(gen.prim * nin * n), oidx: Buf::new(4 * nin * n),
               sc: Buf::new(8 * cap.max(1) * n), rbr: None, ril: None }
    }
    /// Impose the base run's branch bits and integer log (branch-held finite difference).
    fn with_replay(mut self, logs: &[(Vec<u32>, usize, Vec<i32>)]) -> Tile {
        let words = (self.bcap + 31) / 32;
        let mut b = vec![0u32; words * self.n]; let mut i = vec![0i32; self.icap * self.n];
        for (t, l) in logs.iter().enumerate() {
            for (w, &v) in l.0.iter().enumerate() { b[w * self.n + t] = v; }
            for (k, &v) in l.2.iter().enumerate() { i[k * self.n + t] = v; }
        }
        self.rbr = Some(Buf::from(&b)); self.ril = Some(Buf::from(&i)); self
    }
    fn set_tape(&self, gen: &Gen) {
        let mut b = vec![];
        let rb = self.rbr.as_ref().map(|x| x.p).unwrap_or(0); let ri = self.ril.as_ref().map(|x| x.p).unwrap_or(0);
        for p in [self.ta.p, self.tb.p, self.tpa.p, self.tpb.p, self.br.p, self.il.p, self.cnt.p, rb, ri] { b.extend(p.to_le_bytes()); }
        for v in [self.n as i64, self.cap as i64, self.bcap as i64, self.icap as i64] { b.extend(v.to_le_bytes()); }
        let tmp = Buf::from(&b); // device-to-device would need another binding; copy through host
        let host: Vec<u8> = tmp.get(b.len());
        unsafe_put(gen.g, &host);
    }
    fn record(&self, gen: &Gen, cap: &Cap) -> f64 { self.record_x(gen, cap, 0) }
    /// Outputs as f64 whatever the module's primal precision.
    fn yout64(&self, gen: &Gen, nin: usize) -> Vec<f64> {
        if gen.prim == 8 { self.yout.get::<f64>(nin * self.n) } else { self.yout.get::<f32>(nin * self.n).iter().map(|&v| v as f64).collect() }
    }
    fn record_x(&self, gen: &Gen, cap: &Cap, xpert: u64) -> f64 {
        self.set_tape(gen);
        self.br.zero();
        let t = Instant::now();
        Args::new().i32(cap.nz as i32).i32(cap.columns as i32).i32(cap.steps).f32(cap.dt)
            .ptr(cap.dev[cap.f_tab].p).ptr(cap.dev[cap.s_tab].p).ptr(cap.dev[cap.blob_alloc].p)
            .i32(cap.flags[0]).i32(cap.flags[1]).i32(cap.flags[3]).i32(cap.flags[4]).i32(cap.flags[5]).i32(cap.flags[6])
            .ptr(self.colsb.p).i32(self.n as i32).ptr(self.st.p).ptr(self.wk.p).ptr(self.yout.p).ptr(self.oidx.p).ptr(xpert)
            .launch(gen.rec, ((self.n + 63) / 64) as u32, 64);
        t.elapsed().as_secs_f64()
    }
    fn counts(&self) -> Vec<[i32; 4]> { let v: Vec<i32> = self.cnt.get(4 * self.n); v.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect() }
    fn branch_log(&self, t: usize, nbits: usize) -> Vec<u32> {
        let words = (self.bcap + 31) / 32; let all: Vec<u32> = self.br.get(words * self.n);
        (0..(nbits + 31) / 32).map(|w| all[w * self.n + t]).collect()
    }
    fn int_log(&self, t: usize, n: usize) -> Vec<i32> { let all: Vec<i32> = self.il.get(self.icap * self.n); (0..n).map(|i| all[i * self.n + t]).collect() }
    fn tl(&self, gen: &Gen, nin: usize, xdot: &[f64]) -> (Vec<f64>, f64) {
        self.set_tape(gen); // the module global may point at another (since freed) tile
        let xb = Buf::from(xdot); let yb = Buf::new(8 * nin * self.n);
        let t = Instant::now();
        Args::new().i32(self.n as i32).i32(nin as i32).i32(nin as i32).ptr(xb.p).ptr(self.sc.p).ptr(self.oidx.p).ptr(yb.p)
            .launch(gen.tl, ((self.n + 63) / 64) as u32, 64);
        let dt = t.elapsed().as_secs_f64();
        (yb.get(nin * self.n), dt)
    }
    fn adj(&self, gen: &Gen, nin: usize, ybar: &[f64]) -> (Vec<f64>, f64) {
        self.set_tape(gen); // the module global may point at another (since freed) tile
        let yb = Buf::from(ybar); let xb = Buf::new(8 * nin * self.n);
        let t = Instant::now();
        Args::new().i32(self.n as i32).i32(nin as i32).i32(nin as i32).ptr(yb.p).ptr(self.sc.p).ptr(self.oidx.p).ptr(xb.p)
            .launch(gen.adj, ((self.n + 63) / 64) as u32, 64);
        let dt = t.elapsed().as_secs_f64();
        (xb.get(nin * self.n), dt)
    }
}
fn unsafe_put(dst: u64, v: &[u8]) {
    // A Buf view of a module global (not owned): copy without taking ownership.
    let b = std::mem::ManuallyDrop::new(Buf { p: dst, n: v.len() });
    b.put(v);
}

/// Index of the first differing branch bit (n = identical logs).
fn first_diff(a: &[u32], na: usize, b: &[u32], nb: usize) -> usize {
    let n = na.min(nb);
    for w in 0..(n + 31) / 32 {
        let x = a[w] ^ b[w];
        if x != 0 { let i = w * 32 + x.trailing_zeros() as usize; if i < n { return i; } }
    }
    if na == nb { na } else { n }
}
fn hamming(a: &[u32], na: usize, b: &[u32], nb: usize) -> usize {
    let n = na.min(nb);
    let mut d = 0usize;
    for w in 0..(n + 31) / 32 {
        let mask = if (w + 1) * 32 <= n { u32::MAX } else { (1u32 << (n - w * 32)) - 1 };
        d += ((a[w] ^ b[w]) & mask).count_ones() as usize;
    }
    d + na.max(nb) - n
}

/// Input scale per slot for random dot directions.
fn in_scale(nz: usize, x: &[f32], q: usize) -> f64 {
    let floor = [1.0, 1.0, 0.1, 1.0, 1e-6, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1e-9, 1e-3];
    let f = if q < 18 * nz { floor[q / nz] } else { 1e-3 };
    (x[q] as f64).abs() + f
}

/// Realistic all-field direction: 0.5 K theta, 0.5 m/s w, 5 % qv, 10 % of every hydrometeor, number and aerosol;
/// pressure, layer depth, exner and the accumulators held (geometry and the column's pressure are not perturbed).
fn direction(nz: usize, x: &[f32], rng: &mut Rng) -> Vec<f64> {
    let mut d = vec![0f64; x.len()];
    for j in 0..18 { for k in 0..nz {
        let q = j * nz + k; let r = rng.normal();
        d[q] = match j { 0 => 0.5 * r, 2 => 0.5 * r, 4 => 0.05 * r * x[q] as f64, 5..=16 => 0.10 * r * x[q] as f64, _ => 0.0 };
    } }
    d
}

pub fn run(a: &[String]) {
    let arg = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    let caps: Vec<String> = a.iter().enumerate().filter(|(i, x)| *x == "--capture" && *i + 1 < a.len()).map(|(i, _)| a[i + 1].clone()).collect();
    let named: Vec<usize> = arg("--columns").unwrap_or("135,973,1".into()).split(',').map(|s| s.parse().unwrap()).collect();
    let extra: usize = arg("--extra").unwrap_or("61".into()).parse().unwrap();
    let tile_max: usize = arg("--tile").unwrap_or("64".into()).parse().unwrap();
    let eps_list: Vec<f64> = arg("--eps").unwrap_or("1,0.3,0.1,0.03,0.01,0.003,0.001,0.0003,0.0001".into()).split(',').map(|s| s.parse().unwrap()).collect();
    let dirs: Vec<u64> = arg("--dirs").unwrap_or("7,11".into()).split(',').map(|s| s.parse().unwrap()).collect();
    let out = arg("--out").expect("--out");
    gpu::init();
    let cubin = std::fs::read(arg("--cubin").expect("--cubin")).unwrap();
    let gen = Gen::load(&cubin);
    let mut receipt = json!({"schema": "ad.phys.tempo-proof/v1", "cubin_sha256": format!("{:x}", Sha256::digest(&cubin)),
        "work_bytes_per_column": gen.work_bytes, "state_bytes_per_column": gen.state_bytes, "captures": []});
    for cd in &caps {
        let cap = Cap::load(Path::new(cd));
        let (nz, nc) = (cap.nz, cap.columns);
        let nin = 18 * nz + 10;
        let mut r = json!({"capture": cd, "step": cap.m["step"], "config": cap.m["config"], "nz": nz, "columns": nc, "substeps": cap.steps,
                           "dt": cap.dt, "flags": {"aero": cap.flags[0], "hail": cap.flags[1], "melt": cap.flags[3], "first": cap.flags[4], "presence": cap.flags[5], "radii": cap.flags[6]}});
        // ---------------------------------------------------------------- A. binding
        let img = std::fs::read(cap.dir.join(cap.m["image"]["path"].as_str().unwrap())).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&img)), cap.m["image"]["sha256"].as_str().unwrap());
        let fwd = Module::load(&img);
        let ff = fwd.func("tempo_warp_steps");
        let mut fwd_ms = vec![];
        for rep in 0..3 {
            cap.restore(&cap.pre);
            let t = Instant::now(); cap.run_captured(ff); fwd_ms.push(t.elapsed().as_secs_f64() * 1e3);
            if rep == 0 {
                let mut diff_words = 0usize; let mut per = vec![];
                for (i, b) in cap.dev.iter().enumerate() {
                    if i == cap.cost_alloc || i == cap.blob_alloc || cap.scratch.contains(&i) { continue; }
                    let got: Vec<u8> = b.get(b.n);
                    let d = got.chunks(4).zip(cap.post[i].chunks(4)).filter(|(x, y)| x != y).count();
                    if d > 0 { per.push(json!([i, d])); } diff_words += d;
                }
                eprintln!("[A] captured image replay: {diff_words} differing words (cost and blob excluded)");
                r["binding"] = json!({"differing_words": diff_words, "per_allocation": per, "excluded": ["cost (clock)", "blob (read-only tables)", "states/work/blwork (per-column scratch, written before read)"]});
            }
        }
        fwd_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // ---------------------------------------------------------------- B. identity, all columns, count mode
        let all: Vec<usize> = (0..nc).collect();
        cap.restore(&cap.pre);
        let tall = Tile::new(&gen, &all, nin, 0, 0, 0);
        let rec_all_s = tall.record(&gen, &cap);
        let yall: Vec<f32> = tall.yout.get(nin * nc);
        let cnt = tall.counts();
        let mut idiff = vec![0usize; 3]; let mut icmp = vec![0usize; 3]; let mut ex = vec![];
        for c in 0..nc {
            let y = cap.outputs(&cap.post, c);
            for o in 0..nin {
                if y[o].is_nan() && y[o].to_bits() == f32::NAN.to_bits() { continue; }
                let g = if o < 14 * nz { 0 } else if o < 14 * nz + 10 { 1 } else { 2 };
                icmp[g] += 1;
                let got = yall[o * nc + c];
                if got.to_bits() != y[o].to_bits() { idiff[g] += 1; if ex.len() < 10 { ex.push(json!({"column": c, "slot": o, "got": got, "want": y[o]})); } }
            }
        }
        let mut nodes: Vec<i32> = cnt.iter().map(|c| c[0]).collect(); nodes.sort();
        let bits: Vec<i32> = cnt.iter().map(|c| c[1]).collect(); let ints: Vec<i32> = cnt.iter().map(|c| c[2]).collect();
        eprintln!("[B] identity: fields {}/{} differ, accumulators {}/{}, profiles {}/{}; nodes per column median {} max {}; record (all columns, no tape) {:.1} ms vs forward {:.1} ms",
                  idiff[0], icmp[0], idiff[1], icmp[1], idiff[2], icmp[2], nodes[nc / 2], nodes[nc - 1], rec_all_s * 1e3, fwd_ms[1]);
        r["identity"] = json!({"columns": nc, "compared_words": icmp.iter().sum::<usize>(), "differing_words": idiff.iter().sum::<usize>(),
            "fields": [idiff[0], icmp[0]], "accumulators": [idiff[1], icmp[1]], "profiles": [idiff[2], icmp[2]], "examples": ex,
            "tape_nodes_per_column": {"min": nodes[0], "median": nodes[nc / 2], "p90": nodes[nc * 9 / 10], "max": nodes[nc - 1]},
            "branch_bits_max": bits.iter().max(), "int_log_max": ints.iter().max()});
        drop(tall);
        if a.iter().any(|x| x == "--identity-only") { receipt["captures"].as_array_mut().unwrap().push(r); std::fs::write(&out, serde_json::to_string_pretty(&receipt).unwrap()).unwrap(); continue; }
        // ---------------------------------------------------------------- derivative columns
        let mut dcols: Vec<usize> = named.clone();
        let mut c = 0usize; let step = (nc / extra.max(1)).max(1);
        while dcols.len() < named.len() + extra && c < nc { if !dcols.contains(&c) { dcols.push(c); } c += step; }
        let nd = dcols.len();
        // Tile width from free card memory (shared card): tape 24 B + sweep scratch 8 B per node per column.
        let maxcap = dcols.iter().map(|&c| cnt[c][0] as usize).max().unwrap() + 1;
        let (free, _) = gpu::mem_info();
        let budget = (free.saturating_sub(3usize << 30)).min(10usize << 30);
        let per_col = 32 * maxcap + gen.work_bytes + gen.state_bytes + 16 * nin;
        let tile_max = (budget / per_col).clamp(1, tile_max);
        eprintln!("[C] {} derivative columns, max tape {} nodes ({:.0} MB/column), tile {} columns (free {:.1} GB)", nd, maxcap, per_col as f64 / 1e6, tile_max, free as f64 / 1e9);
        // Per-column inputs and directions.
        let xin: Vec<Vec<f32>> = dcols.iter().map(|&c| cap.inputs(&cap.pre, c)).collect();
        let mut dot_rows = vec![]; let mut field_dot_worst = 0f64; let mut zero_ok = true; let mut det_ok = true;
        let mut tl_dirs: Vec<Vec<Vec<f64>>> = vec![]; // [dir][col] -> ydot
        let mut dvecs: Vec<Vec<Vec<f64>>> = vec![]; // [dir][col] -> d
        let mut base_logs: Vec<(Vec<u32>, usize, Vec<i32>)> = vec![];
        let mut cost = json!({}); let mut mem_peak = 0usize;
        let mut drng: Vec<Rng> = dirs.iter().map(|&s| Rng::new(s)).collect();
        for d in 0..dirs.len() { tl_dirs.push(vec![vec![]; nd]); dvecs.push(vec![vec![]; nd]); }
        for chunk0 in (0..nd).step_by(tile_max) {
            let ch: Vec<usize> = dcols[chunk0..(chunk0 + tile_max).min(nd)].to_vec();
            let n = ch.len();
            let capn = ch.iter().map(|&c| cnt[c][0] as usize).max().unwrap() + 1;
            let bcap = ch.iter().map(|&c| cnt[c][1] as usize).max().unwrap() + 64;
            let icap = ch.iter().map(|&c| cnt[c][2] as usize).max().unwrap() + 16;
            let (free0, _) = gpu::mem_info();
            let tile = Tile::new(&gen, &ch, nin, capn, bcap, icap);
            cap.restore(&cap.pre);
            let rec_s = tile.record(&gen, &cap);
            let (free1, _) = gpu::mem_info(); mem_peak = mem_peak.max(free0.saturating_sub(free1));
            let tc = tile.counts();
            assert!(tc.iter().all(|c| c[3] == 0), "tape overflow");
            for (t, &c) in ch.iter().enumerate() { assert_eq!(tc[t][0], cnt[c][0], "node count differs between runs"); }
            for t in 0..n { base_logs.push((tile.branch_log(t, tc[t][1] as usize), tc[t][1] as usize, tile.int_log(t, tc[t][2] as usize))); }
            // C. dot tests
            let xs = |seed: u64| -> Vec<f64> { let mut g = Rng::new(seed); let mut v = vec![0f64; nin * n];
                for t in 0..n { let xi = &xin[chunk0 + t]; for q in 0..nin { v[q * n + t] = g.normal() * in_scale(nz, xi, q); } } v };
            let yv = |seed: u64, lo: usize, cnt_: usize| -> Vec<f64> { let mut g = Rng::new(seed); let mut v = vec![0f64; nin * n];
                for t in 0..n { for q in lo..lo + cnt_ { v[q * n + t] = g.normal(); } } v };
            let mut tl_s = 0f64; let mut ad_s = 0f64;
            for seed in 1..=6u64 {
                let xd = xs(seed); let yb = yv(100 + seed, 0, nin);
                let (yd, s1) = tile.tl(&gen, nin, &xd); let (xb, s2) = tile.adj(&gen, nin, &yb);
                tl_s = s1; ad_s = s2;
                let (mut l, mut rr) = (0f64, 0f64); let mut worst_col = 0f64;
                for t in 0..n {
                    let (mut lc, mut rc) = (0f64, 0f64);
                    for q in 0..nin { lc += yd[q * n + t] * yb[q * n + t]; rc += xd[q * n + t] * xb[q * n + t]; }
                    worst_col = worst_col.max((lc - rc).abs() / lc.abs().max(rc.abs()).max(1e-300)); l += lc; rr += rc;
                }
                let rel = (l - rr).abs() / l.abs().max(rr.abs());
                dot_rows.push(json!({"chunk": chunk0 / tile_max, "seed": seed, "lhs": l, "rhs": rr, "rel": rel, "worst_column_rel": worst_col}));
                if seed == 1 {
                    let (xb2, _) = tile.adj(&gen, nin, &yb);
                    if xb2.iter().zip(&xb).any(|(a, b)| a.to_bits() != b.to_bits()) { det_ok = false; }
                }
            }
            for g in groups(nz) {
                let xd = xs(31); let yb = yv(77, g.1, g.2);
                let (yd, _) = tile.tl(&gen, nin, &xd); let (xb, _) = tile.adj(&gen, nin, &yb);
                let l: f64 = (0..nin * n).map(|i| yd[i] * yb[i]).sum(); let rr: f64 = (0..nin * n).map(|i| xd[i] * xb[i]).sum();
                if l != 0.0 || rr != 0.0 { field_dot_worst = field_dot_worst.max((l - rr).abs() / l.abs().max(rr.abs())); }
            }
            let (yz, _) = tile.tl(&gen, nin, &vec![0f64; nin * n]);
            if yz.iter().any(|v| *v != 0.0) { zero_ok = false; }
            // Tangents for the finite-difference directions.
            for (di, g) in drng.iter_mut().enumerate() {
                let mut xd = vec![0f64; nin * n];
                for t in 0..n { let dv = direction(nz, &xin[chunk0 + t], g); for q in 0..nin { xd[q * n + t] = dv[q]; } dvecs[di][chunk0 + t] = dv; }
                let (yd, _) = tile.tl(&gen, nin, &xd);
                for t in 0..n { tl_dirs[di][chunk0 + t] = (0..nin).map(|q| yd[q * n + t]).collect(); }
            }
            if chunk0 == 0 {
                cost = json!({"forward_launch_ms_median": fwd_ms[1], "forward_launch_ms_all": fwd_ms, "forward_columns": nc,
                    "record_tile_ms": rec_s * 1e3, "tl_sweep_tile_ms": tl_s * 1e3, "adjoint_sweep_tile_ms": ad_s * 1e3, "tile_columns": n,
                    "record_all_columns_no_tape_ms": rec_all_s * 1e3,
                    "tape_bytes_per_column": 24 * capn, "sweep_scratch_bytes_per_column": 8 * capn});
            }
            eprintln!("[C] chunk {}: {} columns, tape cap {} nodes, record {:.0} ms, TL sweep {:.0} ms, AD sweep {:.0} ms", chunk0 / tile_max, n, capn, rec_s * 1e3, tl_s * 1e3, ad_s * 1e3);
        }
        let dot_worst = dot_rows.iter().map(|r| r["rel"].as_f64().unwrap()).fold(0f64, f64::max);
        let dot_col_worst = dot_rows.iter().map(|r| r["worst_column_rel"].as_f64().unwrap()).fold(0f64, f64::max);
        eprintln!("[C] dot: worst total rel {dot_worst:.2e}, worst single column {dot_col_worst:.2e}, per-field worst {field_dot_worst:.2e}, zero direction exact {zero_ok}, reverse repeat bitwise {det_ok}");
        r["derivative_columns"] = json!(dcols);
        r["dot"] = json!({"rows": dot_rows, "worst_rel": dot_worst, "worst_column_rel": dot_col_worst, "per_field_worst_rel": field_dot_worst,
                          "zero_direction_exact": zero_ok, "reverse_repeat_bitwise": det_ok, "precision": "tape partials and sweeps in float64"});
        r["cost"] = cost; r["derivative_device_bytes_peak"] = json!(mem_peak);
        // ---------------------------------------------------------------- D. finite differences of the unchanged image
        let gr = groups(nz);
        let mut taylor = vec![];
        for (di, &seed) in dirs.iter().enumerate() {
            let mut curves: Vec<Vec<f64>> = vec![vec![]; gr.len()];
            let mut hcurves: Vec<Vec<f64>> = vec![vec![]; gr.len()];
            let mut held_nonfinite = vec![0usize; gr.len()];
            let mut named_curves: Vec<Vec<Vec<f64>>> = vec![vec![vec![]; gr.len()]; named.len()];
            let mut flips = vec![]; let mut bud = vec![];
            for &eps in &eps_list {
                let mut ys: Vec<Vec<Vec<f32>>> = vec![]; let mut logs_pm: Vec<Vec<(usize, usize)>> = vec![];
                let mut fdiv_pm: Vec<Vec<f64>> = vec![];
                let mut yh: Vec<Vec<Vec<f32>>> = vec![]; // branch-held outputs (record entry, base decisions imposed)
                for sgn in [1.0f64, -1.0] {
                    let mut host = cap.pre.clone();
                    for (t, &c) in dcols.iter().enumerate() {
                        let x: Vec<f32> = xin[t].iter().zip(&dvecs[di][t]).map(|(&x, &d)| (x as f64 + sgn * eps * d) as f32).collect();
                        cap.set_inputs(&mut host, c, &x);
                    }
                    cap.restore(&host);
                    // branch logs of the perturbed state (record, count mode with logs)
                    let mut lp = vec![]; let mut hp: Vec<Vec<f32>> = vec![]; let mut fdiv: Vec<f64> = vec![];
                    for chunk0 in (0..nd).step_by(tile_max) {
                        let ch: Vec<usize> = dcols[chunk0..(chunk0 + tile_max).min(nd)].to_vec();
                        let bcap = ch.iter().map(|&c| cnt[c][1] as usize).max().unwrap() * 2 + 64;
                        let icap = ch.iter().map(|&c| cnt[c][2] as usize).max().unwrap() * 2 + 16;
                        let tile = Tile::new(&gen, &ch, nin, 0, bcap, icap);
                        tile.record(&gen, &cap);
                        let tc = tile.counts();
                        for t in 0..ch.len() {
                            let (bl, nb, il) = &base_logs[chunk0 + t];
                            let pb = tile.branch_log(t, (tc[t][1] as usize).min(bcap)); let pi = tile.int_log(t, (tc[t][2] as usize).min(icap));
                            let ih = il.iter().zip(&pi).filter(|(a, b)| a != b).count() + il.len().max(pi.len()) - il.len().min(pi.len());
                            let pn = (tc[t][1] as usize).min(bcap);
                            let _ = hamming(bl, *nb, &pb, pn);
                            let fdv = first_diff(bl, *nb, &pb, pn);
                            // flips counted up to the first change of path, plus 1 if the path changed at all
                            lp.push((if fdv == *nb && pn == *nb { 0 } else { 1 + (*nb - fdv.min(*nb)) * 0 }, ih));
                            fdiv.push(fdv as f64 / (*nb).max(1) as f64);
                        }
                        drop(tile);
                        let base = &base_logs[chunk0..chunk0 + ch.len()];
                        let hb = ch.iter().map(|&c| cnt[c][1] as usize).max().unwrap() + 64;
                        let hi = ch.iter().map(|&c| cnt[c][2] as usize).max().unwrap() + 16;
                        let held = Tile::new(&gen, &ch, nin, 0, hb, hi).with_replay(base);
                        held.record(&gen, &cap);
                        let hc = held.counts();
                        for t in 0..ch.len() { assert!(hc[t][1] as usize == base[t].1 && hc[t][2] as usize == base[t].2.len() && hc[t][3] == 0, "replay lost alignment"); }
                        let yo: Vec<f32> = held.yout.get(nin * ch.len());
                        for t in 0..ch.len() { hp.push((0..nin).map(|q| yo[q * ch.len() + t]).collect()); }
                    }
                    yh.push(hp);
                    logs_pm.push(lp); fdiv_pm.push(fdiv);
                    cap.run_captured(ff);
                    let h = cap.download_outputs();
                    ys.push(dcols.iter().map(|&c| cap.outputs(&h, c)).collect());
                }
                for (gi, g) in gr.iter().enumerate() {
                    let (mut num, mut den) = (0f64, 0f64);
                    let mut per_named = vec![(0f64, 0f64); named.len()];
                    for t in 0..nd { for q in g.1..g.1 + g.2 {
                        let (yp, ym) = (ys[0][t][q], ys[1][t][q]);
                        if yp.is_nan() || ym.is_nan() { continue; }
                        let fd = (yp as f64 - ym as f64) / (2.0 * eps);
                        let tl = tl_dirs[di][t][q];
                        num += (fd - tl).powi(2); den += tl.powi(2).max(fd.powi(2));
                        if let Some(ni) = named.iter().position(|&c| c == dcols[t]) { per_named[ni].0 += (fd - tl).powi(2); per_named[ni].1 += tl.powi(2).max(fd.powi(2)); }
                    } }
                    curves[gi].push(if den > 0.0 { (num / den).sqrt() } else { 0.0 });
                    let (mut hn, mut hd) = (0f64, 0f64);
                    for t in 0..nd { for q in g.1..g.1 + g.2 {
                        if ys[0][t][q].is_nan() { continue; }
                        if !yh[0][t][q].is_finite() || !yh[1][t][q].is_finite() { held_nonfinite[gi] += 1; continue; }
                        let fd = (yh[0][t][q] as f64 - yh[1][t][q] as f64) / (2.0 * eps); let tl = tl_dirs[di][t][q];
                        hn += (fd - tl).powi(2); hd += tl.powi(2).max(fd.powi(2));
                    } }
                    hcurves[gi].push(if hd > 0.0 { (hn / hd).sqrt() } else { 0.0 });
                    for ni in 0..named.len() { named_curves[ni][gi].push(if per_named[ni].1 > 0.0 { (per_named[ni].0 / per_named[ni].1).sqrt() } else { 0.0 }); }
                }
                // flips: per column, max of the + and - runs
                let fl: Vec<(usize, usize)> = (0..nd).map(|t| (logs_pm[0][t].0.max(logs_pm[1][t].0), logs_pm[0][t].1.max(logs_pm[1][t].1))).collect();
                let mut bsort: Vec<usize> = fl.iter().map(|x| x.0).collect(); bsort.sort();
                let stable = fl.iter().filter(|x| x.0 == 0 && x.1 == 0).count();
                let fpos: Vec<f64> = (0..nd).map(|t| fdiv_pm[0][t].min(fdiv_pm[1][t])).collect();
                flips.push(json!({"eps": eps, "path_changed_columns": bsort.iter().filter(|&&b| b > 0).count(),
                    "first_path_change_fraction": fpos, "branch_flips_median": bsort[nd / 2], "branch_flips_max": bsort[nd - 1],
                    "int_log_changes_max": fl.iter().map(|x| x.1).max(), "stable_columns": stable,
                    "named": named.iter().map(|&c| { let t = dcols.iter().position(|&x| x == c).unwrap(); json!({"column": c, "branch_flips": fl[t].0, "int_log_changes": fl[t].1}) }).collect::<Vec<_>>()}));
                // E. water budget along the direction: B = sum_k w_k (Qout - Qin) + precip_out - precip_in,
                // w_k = rho_k dz_k from the base input state (TEMPO's density formula, R_d 287.04, R_v 461.5).
                let (rd, rv) = (287.04f64, 461.5f64); let rovr = rd / rv;
                let mut br = vec![];
                for t in 0..nd {
                    let x = &xin[t]; let dv = &dvecs[di][t]; let yt = &tl_dirs[di][t];
                    let (mut dwin, mut dwout, mut fdb) = (0f64, 0f64, 0f64);
                    let mut scale = 0f64;
                    for k in 0..nz {
                        let temp = x[k] as f64 * x[17 * nz + k] as f64; let p = x[nz + k] as f64; let qv = x[4 * nz + k] as f64;
                        let w = rovr * p / (rd * temp * (qv + rovr)) * x[3 * nz + k] as f64;
                        for j in 4..=9 {
                            dwin += w * dv[j * nz + k];
                            dwout += w * yt[(j - 3) * nz + k];
                            let fd = (ys[0][t][(j - 3) * nz + k] as f64 - ys[1][t][(j - 3) * nz + k] as f64) / (2.0 * eps);
                            fdb += w * fd;
                            scale = scale.max((w * yt[(j - 3) * nz + k]).abs());
                        }
                    }
                    let dp = yt[14 * nz + 1]; // precip_total: input accumulator held, so its tangent is the step precipitation
                    let fdp = (ys[0][t][14 * nz + 1] as f64 - ys[1][t][14 * nz + 1] as f64) / (2.0 * eps);
                    let tl_res = dwout + dp - dwin; let fd_res = fdb + fdp - dwin;
                    let s = scale.max(dp.abs()).max(dwin.abs()).max(1e-300);
                    br.push((dcols[t], tl_res / s, fd_res / s));
                }
                let worst_tl = br.iter().map(|b| b.1.abs()).fold(0f64, f64::max);
                let worst_fd = br.iter().map(|b| b.2.abs()).fold(0f64, f64::max);
                bud.push(json!({"eps": eps, "tangent_residual_rel_worst_column": worst_tl, "fd_residual_rel_worst_column": worst_fd,
                    "named": named.iter().map(|&c| { let b = br.iter().find(|b| b.0 == c).unwrap(); json!({"column": c, "tangent_rel": b.1, "fd_rel": b.2}) }).collect::<Vec<_>>()}));
                eprintln!("[D] dir {seed} eps {eps:e}: path changed in {}/{} columns, first change at {:?} of the branch log, int-log changes max {:?}; budget tangent {:.1e} fd {:.1e}",
                          bsort.iter().filter(|&&b| b > 0).count(), nd, fpos.iter().map(|v| format!("{:.3}", v)).collect::<Vec<_>>(), fl.iter().map(|x| x.1).max(), worst_tl, worst_fd);
            }
            cap.restore(&cap.pre);
            let per_group: Vec<Value> = gr.iter().enumerate().map(|(gi, g)| {
                let c = &curves[gi]; let (bi, bv) = c.iter().enumerate().fold((0, f64::INFINITY), |a, (i, &v)| if v < a.1 { (i, v) } else { a });
                let h = &hcurves[gi]; let hb = h.iter().cloned().fold(f64::INFINITY, f64::min);
                json!({"field": g.0, "curve": c, "best": bv, "best_eps": eps_list[bi], "branch_held_curve": h, "branch_held_best": hb, "branch_held_nonfinite_words": held_nonfinite[gi]})
            }).collect();
            let fmt = |v: &Value| v.as_array().unwrap().iter().map(|v| v.as_f64().map(|x| format!("{:.1e}", x)).unwrap_or("nan".into())).collect::<Vec<_>>().join(" ");
            for g in &per_group { eprintln!("[D] dir {seed} {:>16}: forward FD best {:.2e} [{}]   branch-held [{}]", g["field"].as_str().unwrap(), g["best"].as_f64().unwrap_or(f64::NAN),
                                            fmt(&g["curve"]), fmt(&g["branch_held_curve"])); }
            eprintln!("[D] dir {seed} branch-held non-finite words per field: {:?}", gr.iter().zip(&held_nonfinite).filter(|x| *x.1 > 0).map(|(g, n)| (g.0.clone(), *n)).collect::<Vec<_>>());
            taylor.push(json!({"direction_seed": seed, "direction": "theta 0.5 K, w 0.5 m/s, qv 5 %, every hydrometeor/number/aerosol 10 % (times N(0,1) per element); p, dz, exner, accumulators held",
                "eps": eps_list, "per_field": per_group,
                "named_columns": named.iter().enumerate().map(|(ni, &c)| json!({"column": c, "per_field": gr.iter().enumerate().map(|(gi, g)| json!({"field": g.0, "curve": named_curves[ni][gi]})).collect::<Vec<_>>()})).collect::<Vec<_>>(),
                "flips": flips, "water_budget": bud}));
        }
        r["taylor"] = json!(taylor);
        receipt["captures"].as_array_mut().unwrap().push(r);
        std::fs::write(&out, serde_json::to_string_pretty(&receipt).unwrap()).unwrap();
    }
    eprintln!("receipt {out}");
}

/// Float64-promoted grade: the same generated column code with every float as
/// binary64. Tangent against the branch-held centred finite difference of the
/// promoted record entry itself, perturbation applied in binary64 (so the
/// curve shows the truncation slope instead of float32 rounding).
pub fn promoted(a: &[String]) {
    let arg = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    let cd = arg("--capture").expect("--capture");
    let cols: Vec<usize> = arg("--columns").unwrap_or("135,973,1".into()).split(',').map(|s| s.parse().unwrap()).collect();
    let eps_list: Vec<f64> = arg("--eps").unwrap_or("0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001".into()).split(',').map(|s| s.parse().unwrap()).collect();
    let dirs: Vec<u64> = arg("--dirs").unwrap_or("7,11".into()).split(',').map(|s| s.parse().unwrap()).collect();
    let out = arg("--out").expect("--out");
    gpu::init();
    let cubin = std::fs::read(arg("--cubin64").expect("--cubin64")).unwrap();
    let gen = Gen::load(&cubin);
    assert_eq!(gen.prim, 8, "--cubin64 must be the promoted build");
    let cap = Cap::load(Path::new(&cd));
    let nz = cap.nz; let nin = 18 * nz + 10; let n = cols.len();
    let cnt0 = Tile::new(&gen, &cols, nin, 0, 0, 0); cnt0.record(&gen, &cap); let cc = cnt0.counts(); drop(cnt0);
    let capn = cc.iter().map(|c| c[0] as usize).max().unwrap() + 1;
    let bcap = cc.iter().map(|c| c[1] as usize).max().unwrap() + 64;
    let icap = cc.iter().map(|c| c[2] as usize).max().unwrap() + 16;
    let tile = Tile::new(&gen, &cols, nin, capn, bcap, icap);
    tile.record(&gen, &cap);
    let tc = tile.counts(); assert!(tc.iter().all(|c| c[3] == 0));
    let base: Vec<(Vec<u32>, usize, Vec<i32>)> = (0..n).map(|t| (tile.branch_log(t, tc[t][1] as usize), tc[t][1] as usize, tile.int_log(t, tc[t][2] as usize))).collect();
    let y0 = tile.yout64(&gen, nin);
    let mut g = Rng::new(5); let xd: Vec<f64> = (0..nin * n).map(|_| g.normal()).collect(); let yb: Vec<f64> = (0..nin * n).map(|_| g.normal()).collect();
    let (yd, _) = tile.tl(&gen, nin, &xd); let (xb, _) = tile.adj(&gen, nin, &yb);
    let l: f64 = (0..nin * n).map(|i| yd[i] * yb[i]).sum(); let r: f64 = (0..nin * n).map(|i| xd[i] * xb[i]).sum();
    let dot64 = (l - r).abs() / l.abs().max(r.abs());
    eprintln!("[P] promoted build: nodes per column {:?}, dot rel {:.2e}", tc.iter().map(|c| c[0]).collect::<Vec<_>>(), dot64);
    let xin: Vec<Vec<f32>> = cols.iter().map(|&c| cap.inputs(&cap.pre, c)).collect();
    let gr = groups(nz);
    let mut res = vec![];
    for &seed in &dirs {
        let mut rng = Rng::new(seed);
        let mut dmat = vec![0f64; nin * n];
        for t in 0..n { let dv = direction(nz, &xin[t], &mut rng); for q in 0..nin { dmat[q * n + t] = dv[q]; } }
        let (ytl, _) = tile.tl(&gen, nin, &dmat);
        let mut curves: Vec<Vec<f64>> = vec![vec![]; gr.len()]; let mut fcurves: Vec<Vec<f64>> = vec![vec![]; gr.len()];
        let mut per_col: Vec<Vec<Vec<f64>>> = vec![vec![vec![]; gr.len()]; n];
        let mut nonfinite = vec![0usize; gr.len()]; let mut changed = vec![];
        for &eps in &eps_list {
            let mut yh = vec![]; let mut yf = vec![]; let mut pc = 0usize;
            for sgn in [1.0f64, -1.0] {
                let xp: Vec<f64> = dmat.iter().map(|d| sgn * eps * d).collect(); let xbuf = Buf::from(&xp);
                let held = Tile::new(&gen, &cols, nin, 0, bcap, icap).with_replay(&base);
                held.record_x(&gen, &cap, xbuf.p); yh.push(held.yout64(&gen, nin)); drop(held);
                let free = Tile::new(&gen, &cols, nin, 0, 2 * bcap, 2 * icap);
                free.record_x(&gen, &cap, xbuf.p); yf.push(free.yout64(&gen, nin));
                let fcnt = free.counts();
                for t in 0..n {
                    let nb = (fcnt[t][1] as usize).min(2 * bcap);
                    let fb = free.branch_log(t, nb);
                    if fcnt[t][1] as usize != base[t].1 || first_diff(&base[t].0, base[t].1, &fb, nb) != base[t].1 { pc += 1; }
                }
            }
            changed.push(pc);
            for (gi, gg) in gr.iter().enumerate() {
                let (mut hn, mut hd, mut fnn, mut fdd) = (0f64, 0f64, 0f64, 0f64);
                let mut pcn = vec![(0f64, 0f64); n];
                for t in 0..n { for q in gg.1..gg.1 + gg.2 {
                    let i = q * n + t; let tl = ytl[i];
                    if !y0[i].is_finite() { continue; }
                    if yh[0][i].is_finite() && yh[1][i].is_finite() {
                        let fd = (yh[0][i] - yh[1][i]) / (2.0 * eps); hn += (fd - tl).powi(2); hd += tl.powi(2).max(fd.powi(2));
                        pcn[t].0 += (fd - tl).powi(2); pcn[t].1 += tl.powi(2).max(fd.powi(2));
                    } else { nonfinite[gi] += 1; }
                    if yf[0][i].is_finite() && yf[1][i].is_finite() { let fd = (yf[0][i] - yf[1][i]) / (2.0 * eps); fnn += (fd - tl).powi(2); fdd += tl.powi(2).max(fd.powi(2)); }
                } }
                curves[gi].push(if hd > 0.0 { (hn / hd).sqrt() } else { 0.0 });
                fcurves[gi].push(if fdd > 0.0 { (fnn / fdd).sqrt() } else { 0.0 });
                for t in 0..n { per_col[t][gi].push(if pcn[t].1 > 0.0 { (pcn[t].0 / pcn[t].1).sqrt() } else { 0.0 }); }
            }
        }
        let fmt = |c: &Vec<f64>| c.iter().map(|v| format!("{:.1e}", v)).collect::<Vec<_>>().join(" ");
        eprintln!("[P] dir {seed}: eps {:?}; free-run path changed in {:?} of {} +/- runs", eps_list, changed, 2 * n);
        let mut fields = vec![];
        for (gi, gg) in gr.iter().enumerate() {
            let best = curves[gi].iter().cloned().fold(f64::INFINITY, f64::min);
            if curves[gi].iter().all(|v| *v == 0.0) && fcurves[gi].iter().all(|v| *v == 0.0) { continue; }
            eprintln!("[P] dir {seed} {:>16}: branch-held best {:.2e} [{}]  free [{}]  nonfinite {}", gg.0, best, fmt(&curves[gi]), fmt(&fcurves[gi]), nonfinite[gi]);
            fields.push(json!({"field": gg.0, "branch_held_curve": curves[gi], "branch_held_best": best, "free_curve": fcurves[gi], "held_nonfinite_words": nonfinite[gi],
                "per_column": cols.iter().enumerate().map(|(t, &c)| json!({"column": c, "branch_held_curve": per_col[t][gi]})).collect::<Vec<_>>()}));
        }
        res.push(json!({"direction_seed": seed, "eps": eps_list, "free_path_changed_runs": changed, "fields": fields}));
    }
    let rec = json!({"schema": "ad.phys.tempo-proof-f64/v1", "capture": cd, "columns": cols, "cubin_sha256": format!("{:x}", Sha256::digest(&cubin)),
                     "nodes_per_column": tc.iter().map(|c| c[0]).collect::<Vec<_>>(), "dot_rel": dot64, "directions": res});
    std::fs::write(&out, serde_json::to_string_pretty(&rec).unwrap()).unwrap();
    eprintln!("receipt {out}");
}

/// Warp-shaped (forecast shape) tangent/adjoint proof and cost, lane L5.
pub mod warp;
