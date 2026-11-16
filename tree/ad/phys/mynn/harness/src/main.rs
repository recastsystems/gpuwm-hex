// SPDX-License-Identifier: Apache-2.0
//! P3: whole-launch tangent and adjoint of hex's MYNN-EDMF c6d08709 launch `rrfs_mynnedmf_run_l16`.
//!
//! `p3-mynn build`  NVRTC-compiles the generated unit with the forward image's own options (build time).
//! `p3-mynn prove`  runs the proofs on captured launches of the real hrrrv5 forecast:
//!   replay   the captured forward image on the captured pre-state reproduces the captured post-state,
//!   identity the derivative entries' primal equals the forward bitwise (base state and every perturbed state),
//!   zero     zero seed gives exactly zero tangent and zero adjoint,
//!   dot      <J dx, ybar> = <dx, J^T ybar> (whole launch, all 91 float arrays, and per input field),
//!   fd       tangent vs centred differences of the UNCHANGED forward image, per output field, eps 0.1..1e-4,
//!            with switch flips per column counted from the recorded comparison outcomes,
//!   cost     forward, tangent, tape sizing and adjoint times, tape bytes per column, device memory.
//! All field arithmetic runs on the GPU; this host program only schedules, reduces and scores (Rust).
mod cuda;
mod nvrtc;
use cuda::*;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// A capture object at `rel` under the launch folder or any parent folder (stores keep objects per config).
fn find_up(dir: &Path, rel: &str) -> Option<PathBuf> {
    let mut d = Some(dir);
    while let Some(x) = d { let p = x.join(rel); if p.exists() { return Some(p); } d = x.parent(); }
    None
}
fn read(p: &Path) -> Vec<u8> { std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {}", p.display(), e)) }
fn f32s(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() }
fn f64s(b: &[u8]) -> Vec<f64> { b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect() }
fn bytes_f32(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn bytes_f64(v: &[f64]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn sha(b: &[u8]) -> String { use sha2::Digest; format!("{:x}", sha2::Sha256::digest(b)) }

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn uni(&mut self) -> f64 { ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64 }
    fn normal(&mut self) -> f64 { (-2.0 * self.uni().ln()).sqrt() * (2.0 * std::f64::consts::PI * self.uni()).cos() }
}

// ------------------------------------------------------------------ capture
#[allow(dead_code)]
struct Arg { name: String, device: bool, dtype: String, scalar: Vec<u8>, alloc: i64, shape: Vec<usize>, pre: Vec<u8>, post: Vec<u8> }
struct Capture { dir: PathBuf, kernel: String, grid: [u32; 3], block: [u32; 3], image: Vec<u8>, image_sha: String, args: Vec<Arg> }
impl Capture {
    fn load(dir: &Path) -> Capture {
        let m: Value = serde_json::from_slice(&read(&dir.join("manifest.json"))).unwrap();
        let g = |v: &Value| -> [u32; 3] { let a = v.as_array().unwrap(); [0, 1, 2].map(|i| a[i].as_u64().unwrap() as u32) };
        let mut args = Vec::new();
        for a in m["arguments"].as_array().unwrap() {
            let name = a["name"].as_str().unwrap().to_string();
            let device = a["kind"].as_str().unwrap() == "device";
            let dtype = a["dtype"].as_str().unwrap_or("").to_string();
            if !device {
                let hx = a["value_hex"].as_str().unwrap_or("");
                let b: Vec<u8> = (0..hx.len() / 2).map(|i| u8::from_str_radix(&hx[2 * i..2 * i + 2], 16).unwrap()).collect();
                args.push(Arg { name, device, dtype, scalar: b, alloc: -1, shape: vec![], pre: vec![], post: vec![] });
                continue;
            }
            assert_eq!(a["offset"].as_u64().unwrap_or(0), 0, "{} offset", name);
            let obj = |key: &str| find_up(dir, &format!("objects/{}.bin", a[key].as_str().unwrap()));
            let shape: Vec<usize> = a["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
            let esz = if dtype.ends_with('8') { 8 } else if dtype.ends_with('1') { 1 } else { 4 };
            let nbytes = shape.iter().product::<usize>() * esz;
            // Scratch whose content the capture did not keep (the MYNN workspace) starts as zeros; replay then shows
            // whether any kept output depends on it.
            let (pre, post) = match (obj("sha256"), obj("post_sha256")) {
                (Some(p), Some(q)) => (read(&p), read(&q)),
                _ if name == "workspace" => (vec![0u8; nbytes], vec![]),
                _ => panic!("capture object missing for {}", name),
            };
            args.push(Arg { name, device, dtype, scalar: vec![], alloc: a["allocation"].as_i64().unwrap(), shape, pre, post });
        }
        Capture {
            dir: dir.to_path_buf(), kernel: m["kernel"].as_str().unwrap().to_string(),
            grid: g(&m["geometry"]["grid"]), block: g(&m["geometry"]["block"]),
            image: read(&find_up(dir, m["image"]["path"].as_str().unwrap()).expect("forward image")), image_sha: m["image"]["sha256"].as_str().unwrap().to_string(), args,
        }
    }
    fn arg(&self, n: &str) -> &Arg { self.args.iter().find(|a| a.name == n).unwrap_or_else(|| panic!("no arg {}", n)) }
    fn int(&self, n: &str) -> i64 { let s = &self.arg(n).scalar; i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as i64 }
}

/// The unchanged captured forward image, run on (possibly perturbed) pre-state.
struct Original { m: Module, f: CUfunction, bufs: BTreeMap<String, DBuf> }
impl Original {
    fn new(c: &Capture) -> Original {
        let m = Module::load(&c.image);
        let f = m.func(&c.kernel);
        let bufs = c.args.iter().filter(|a| a.device).map(|a| (a.name.clone(), DBuf::new(a.pre.len().max(a.post.len())))).collect();
        Original { m, f, bufs }
    }
    fn run(&self, c: &Capture, over: &BTreeMap<String, Vec<f32>>) -> f64 {
        for a in c.args.iter().filter(|a| a.device) {
            match over.get(&a.name) { Some(v) => self.bufs[&a.name].upload(&bytes_f32(v)), None => self.bufs[&a.name].upload(&a.pre) }
        }
        let args: Vec<KArg> = c.args.iter().map(|a| if a.device { KArg::Ptr(self.bufs[&a.name].ptr) } else { KArg::Bytes(a.scalar.clone()) }).collect();
        sync();
        let t = Timer::new(); t.start();
        launch(self.f, c.grid, c.block, 0, &args);
        let ms = t.stop_ms();
        sync();
        let _ = &self.m;
        ms
    }
    fn get(&self, n: &str) -> Vec<u8> { self.bufs.get(n).map(|b| b.download()).unwrap_or_default() }
}

// ------------------------------------------------------------------ generated entries
struct Gen {
    ms: Vec<Module>, tl: CUfunction, ad: CUfunction, count: CUfunction,
    names: Vec<String>, scalars: Vec<String>, present: Vec<bool>,
    x: Vec<DBuf>, y: Vec<DBuf>, tin: Vec<DBuf>, tout: Vec<DBuf>, ybar: Vec<DBuf>, xbar: Vec<DBuf>,
    kin: DBuf, kout: DBuf, eflags: DBuf, launch_buf: DBuf,
    sz_tl: usize, sz_tape: usize, cols: usize, col_elems: usize, arena_col: usize,
    fail: DBuf, used: DBuf, used_hw: DBuf, dom: DBuf, br_bits: Option<DBuf>, br_n: Option<DBuf>, br_hash: Option<DBuf>, br_words: usize,
}
const NODE_BYTES: usize = 16 + 8; // MynnAdNode (16 B since mynn-tape; 24 B before) + bar
impl Gen {
    fn new(cubin: &str, info: &Value, c: &Capture, br_words: usize) -> Gen {
        let ms: Vec<Module> = (0..3).map(|k| Module::load(&read(Path::new(&format!("{}.{}.cubin", cubin, k))))).collect();
        let names: Vec<String> = info["float_arrays"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
        let scalars: Vec<String> = info["scalars"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
        let (tl, ad, count) = (ms[0].func("mynn_whole_tl"), ms[1].func("mynn_whole_ad"), ms[2].func("mynn_whole_count"));
        let szk = ms[0].func("mw_sizes");
        eprintln!("entries (registers, local bytes per thread): tl {:?} ad {:?} count {:?}", func_info(tl), func_info(ad), func_info(count));
        let o = DBuf::new(64);
        launch(szk, [1, 1, 1], [1, 1, 1], 0, &[KArg::Ptr(o.ptr)]);
        sync();
        let s: Vec<u64> = o.download_at(0, 32).chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        assert_eq!(s[2] as usize, 16, "MynnAdNode size (mynn-tape: 16 B compact node)");
        assert_eq!(s[3] as usize, names.len() * 48 + 8 * 14 + 16, "MwLaunch layout");
        let mk = |bytes_of: &dyn Fn(&Arg) -> usize| -> Vec<DBuf> { names.iter().map(|n| DBuf::new(bytes_of(c.arg(n)))).collect() };
        let x = mk(&|a| a.pre.len()); let y = mk(&|a| a.pre.len());
        let tin = mk(&|a| a.pre.len() * 2); let tout = mk(&|a| a.pre.len() * 2);
        let ybar = mk(&|a| a.pre.len() * 2); let xbar = mk(&|a| a.pre.len() * 2);
        let ni = (c.int("ime") - c.int("ims") + 1) as usize;
        let nk = (c.int("kme") - c.int("kms") + 1) as usize;
        let col_elems: usize = names.len() * nk + 8;
        let present: Vec<bool> = names.iter().map(|n| c.arg(n).device).collect();
        let _ = ni;
        let cols = (c.int("batch_count") as usize).min(get_env_usize("P3_COLS", usize::MAX));
        let ws = u64::from_le_bytes(c.arg("workspace_stride").scalar[..8].try_into().unwrap()) as usize;
        let arena_col = ws * get_env_usize("P3_ARENA_X", 12);
        let k = c.arg("kpbl");
        Gen {
            tl, ad, count, names, scalars, present, x, y, tin, tout, ybar, xbar,
            kin: DBuf::new(k.pre.len()), kout: DBuf::new(k.pre.len()), eflags: DBuf::new(c.arg("error_flags").pre.len()),
            launch_buf: DBuf::new(s[3] as usize), sz_tl: s[0] as usize, sz_tape: s[1] as usize, cols, col_elems, arena_col,
            fail: DBuf::new(cols * 16), used: DBuf::new(cols * 8), used_hw: DBuf::new(cols * 8), dom: DBuf::new(cols * 4),
            br_bits: if br_words > 0 { Some(DBuf::new(cols * br_words * 4)) } else { None },
            br_n: if br_words > 0 { Some(DBuf::new(cols * 8)) } else { None },
            br_hash: if br_words > 0 { Some(DBuf::new(cols * 8)) } else { None },
            br_words, ms,
        }
    }
    fn set_inputs(&self, c: &Capture, over: &BTreeMap<String, Vec<f32>>) {
        for (i, n) in self.names.iter().enumerate() {
            let b = match over.get(n) { Some(v) => bytes_f32(v), None => c.arg(n).pre.clone() };
            self.x[i].upload(&b);
            self.y[i].upload(&b);
        }
        self.kin.upload(&c.arg("kpbl").pre);
        self.kout.upload(&c.arg("kpbl").pre);
        self.eflags.upload(&c.arg("error_flags").pre);
    }
    fn set_globals(&self, record: bool) {
        let p = |o: &Option<DBuf>| if record { o.as_ref().map(|b| b.ptr).unwrap_or(0) } else { 0 };
        for m in &self.ms {
            m.set_global("mw_arena_limit", &(self.arena_col as u64).to_le_bytes());
            m.set_global("mw_fail", &self.fail.ptr.to_le_bytes());
            m.set_global("mw_br_bits", &p(&self.br_bits).to_le_bytes());
            m.set_global("mw_br_n", &p(&self.br_n).to_le_bytes());
            m.set_global("mw_br_hash", &p(&self.br_hash).to_le_bytes());
            m.set_global("mw_br_words", &(self.br_words as u64).to_le_bytes());
            m.set_global("mw_arena_hw", &self.used_hw.ptr.to_le_bytes());
            m.set_global("mw_dom", &self.dom.ptr.to_le_bytes());
        }
        if record { for b in [&self.br_bits, &self.br_n, &self.br_hash] { if let Some(b) = b { b.zero() } } }
        self.used_hw.zero();
        self.dom.zero();
    }
    /// mode 0 TL (all columns), 2 count (all columns), 1 AD (one tile)
    #[allow(clippy::too_many_arguments)]
    fn launch(&self, c: &Capture, mode: i32, col_first: usize, col_count: usize, arena: &DBuf, store: &DBuf,
              leaf: Option<&DBuf>, kstore: &DBuf, nodes: Option<&DBuf>, bars: Option<&DBuf>, capacity: usize) -> f64 {
        let mut w: Vec<u8> = Vec::new();
        let p = |w: &mut Vec<u8>, v: u64| w.extend_from_slice(&v.to_le_bytes());
        for i in 0..self.names.len() {
            if !self.present[i] { for _ in 0..6 { p(&mut w, 0) } continue; } // null in the forward: absent here too
            p(&mut w, self.x[i].ptr); p(&mut w, self.y[i].ptr);
            p(&mut w, if mode == 0 { self.tin[i].ptr } else { 0 }); p(&mut w, if mode == 0 { self.tout[i].ptr } else { 0 });
            p(&mut w, if mode == 1 { self.ybar[i].ptr } else { 0 }); p(&mut w, if mode == 1 { self.xbar[i].ptr } else { 0 });
        }
        p(&mut w, self.kin.ptr); p(&mut w, self.kout.ptr); p(&mut w, self.eflags.ptr);
        p(&mut w, arena.ptr); p(&mut w, self.arena_col as u64);
        p(&mut w, store.ptr); p(&mut w, self.col_elems as u64);
        p(&mut w, leaf.map(|b| b.ptr).unwrap_or(0)); p(&mut w, kstore.ptr);
        p(&mut w, nodes.map(|b| b.ptr).unwrap_or(0)); p(&mut w, bars.map(|b| b.ptr).unwrap_or(0));
        p(&mut w, capacity as u64); p(&mut w, self.used.ptr); p(&mut w, self.fail.ptr);
        for v in [col_first as i32, col_count as i32, mode, 0] { w.extend_from_slice(&v.to_le_bytes()); }
        assert_eq!(w.len(), self.launch_buf.bytes);
        self.launch_buf.upload(&w);
        arena.zero(); store.zero(); self.fail.zero();
        let mut args = vec![KArg::Ptr(self.launch_buf.ptr)];
        for s in &self.scalars {
            let n = s.split_whitespace().last().unwrap();
            let a = c.arg(n);
            args.push(KArg::Bytes(a.scalar.clone()));
        }
        let f = match mode { 0 => self.tl, 1 => self.ad, _ => self.count };
        if std::env::var("P3_TRACE").is_ok() { eprintln!("launch mode {} first {} count {} capacity {}", mode, col_first, col_count, capacity); }
        let blocks = col_count.div_ceil(32) as u32;
        sync();
        let t = Timer::new(); t.start();
        launch(f, [blocks, 1, 1], [32, 1, 1], 0, &args);
        let ms = t.stop_ms();
        sync();
        let fails: Vec<i32> = self.fail.download_at(0, col_count * 16).chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
        let nf = [0, 1, 2, 3].map(|s| fails.iter().skip(s).step_by(4).filter(|&&v| v != 0).count());
        if nf.iter().any(|&v| v > 0) {
            let cols: Vec<usize> = (0..col_count).filter(|&k| fails[k * 4 + 1] != 0).take(8).map(|k| col_first + k).collect();
            eprintln!("mode {} failure slots (overflow, domain, context, arena): {:?}; domain columns {:?}", mode, nf, cols);
        }
        ms
    }
    fn tl_run(&self, c: &Capture, dirs: Option<&[Vec<f64>]>) -> f64 {
        for i in 0..self.names.len() {
            match dirs { Some(d) => self.tin[i].upload(&bytes_f64(&d[i])), None => self.tin[i].zero() }
            self.tout[i].zero();
        }
        let arena = DBuf::new(self.cols.div_ceil(32) * 32 * self.arena_col);
        let store = DBuf::new(self.cols * self.col_elems * self.sz_tl);
        let ks = DBuf::new(self.cols * 32);
        self.launch(c, 0, get_env_usize("P3_COL0", 0), self.cols, &arena, &store, None, &ks, None, None, 0)
    }
    /// Returns (count ms, ad ms, capacity nodes, used per column, tile)
    fn ad_run(&self, c: &Capture, ybar: &[Vec<f64>], budget: usize) -> (f64, f64, usize, Vec<i64>, usize) {
        let store = DBuf::new(self.cols * self.col_elems * self.sz_tape);
        let ks = DBuf::new(self.cols * 32);
        let arena = DBuf::new(self.cols.div_ceil(32) * 32 * self.arena_col);
        self.used.zero();
        let tc = self.launch(c, 2, 0, self.cols, &arena, &store, None, &ks, None, None, 0);
        let used: Vec<i64> = self.used.download_at(0, self.cols * 8).chunks_exact(8).map(|x| i64::from_le_bytes(x.try_into().unwrap())).collect();
        let cap = (*used.iter().max().unwrap() as usize) + 16;
        let tile = ((budget / (cap * NODE_BYTES)) / 32 * 32).clamp(32, self.cols.div_ceil(32) * 32);
        for i in 0..self.names.len() { self.ybar[i].upload(&bytes_f64(&ybar[i])); self.xbar[i].zero(); }
        let nodes = DBuf::new(tile * cap * 24);
        let bars = DBuf::new(tile * cap * 8);
        let leaf = DBuf::new(tile * self.col_elems * 4);
        let mut ta = 0.0;
        let mut first = 0;
        while first < self.cols {
            let n = tile.min(self.cols - first);
            ta += self.launch(c, 1, first, n, &arena, &store, Some(&leaf), &ks, Some(&nodes), Some(&bars), cap);
            first += n;
        }
        (tc, ta, cap, used, tile)
    }
    fn branch_record(&self) -> (Vec<u32>, Vec<u64>, Vec<u64>) {
        let b = self.br_bits.as_ref().unwrap().download();
        (b.chunks_exact(4).map(|x| u32::from_le_bytes(x.try_into().unwrap())).collect(),
         self.br_n.as_ref().unwrap().download().chunks_exact(8).map(|x| u64::from_le_bytes(x.try_into().unwrap())).collect(),
         self.br_hash.as_ref().unwrap().download().chunks_exact(8).map(|x| u64::from_le_bytes(x.try_into().unwrap())).collect())
    }
}

/// per-column flips between two branch records: differing outcomes in the common recorded prefix + length change
fn flips(a: &(Vec<u32>, Vec<u64>, Vec<u64>), b: &(Vec<u32>, Vec<u64>, Vec<u64>), words: usize) -> Vec<u64> {
    (0..a.1.len()).map(|col| {
        if a.2[col] == b.2[col] && a.1[col] == b.1[col] { return 0; }
        let n = a.1[col].min(b.1[col]).min(words as u64 * 32);
        let mut f = 0u64;
        for wi in 0..(n as usize).div_ceil(32) {
            let mut x = a.0[col * words + wi] ^ b.0[col * words + wi];
            let rem = n as usize - wi * 32;
            if rem < 32 { x &= (1u32 << rem) - 1; }
            f += x.count_ones() as u64;
        }
        f + a.1[col].abs_diff(b.1[col])
    }).collect()
}

// ------------------------------------------------------------------ scoring helpers
const AEROSOL: &[&str] = &["nc", "ni", "nifa", "nwfa", "nbca", "rncblten", "rniblten", "rnifablten", "rnwfablten", "rnbcablten"];
const GEOMETRY: &[&str] = &["dx", "xland", "dz", "pattern_spp", "chem3d", "settle3d", "vd3d", "frp_mean", "emis_ant_no"];

/// level-wise standard deviation of a field (i fastest, then k): one value per element
fn level_std(v: &[f32], ni: usize) -> Vec<f64> {
    let nl = v.len() / ni;
    let mut out = vec![0.0; v.len()];
    for l in 0..nl {
        let s = &v[l * ni..(l + 1) * ni];
        let m = s.iter().map(|&x| x as f64).sum::<f64>() / ni as f64;
        let sd = (s.iter().map(|&x| (x as f64 - m).powi(2)).sum::<f64>() / ni as f64).sqrt();
        for i in 0..ni { out[l * ni + i] = sd; }
    }
    out
}
fn rms(v: &[f32]) -> f64 { let r = (v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / v.len().max(1) as f64).sqrt(); if r > 0.0 && r.is_finite() { r } else { 1.0 } }
fn dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() }
fn norm(a: &[f64]) -> f64 { dot(a, a).sqrt() }

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    let all = |k: &str| a.iter().enumerate().filter(|(_, x)| *x == k).map(|(i, _)| a[i + 1].clone()).collect::<Vec<_>>();
    match a.get(1).map(|s| s.as_str()) {
        Some("build") => {
            let src = read(Path::new(&get("--src").unwrap()));
            let mut opts: Vec<String> = ["--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true",
                "--split-compile=0", "--ptxas-options=--split-compile=0"].iter().map(|s| s.to_string()).collect();
            opts.push(format!("--gpu-architecture={}", get("--arch").unwrap_or("sm_120".into())));
            opts.push(format!("--include-path={}", get("--include").unwrap()));
            if a.iter().any(|x| x == "--verbose") { opts.push("--ptxas-options=-v".into()); }
            for d in all("--define") { opts.push(format!("-D{}", d)); }
            for o in all("--opt") { opts.push(o); } // extra NVRTC options (debug builds, e.g. --generate-line-info)
            let t0 = std::time::Instant::now();
            let out = get("--out").unwrap();
            let lib = get("--nvrtc").unwrap();
            // three entries, three NVRTC programs, three threads
            let res: Vec<Value> = std::thread::scope(|s| {
                let hs: Vec<_> = (0..3).map(|k| { let (src, opts, lib, out) = (&src, &opts, &lib, &out); s.spawn(move || {
                    let mut o = opts.clone(); o.push(format!("-DMW_ENTRY={}", k));
                    let t = std::time::Instant::now();
                    let (cubin, log) = nvrtc::compile(lib, src, "mynn_whole.gen.cu", &o);
                    let path = format!("{}.{}.cubin", out, k);
                    std::fs::write(&path, &cubin).unwrap();
                    json!({"entry": k, "path": path, "cubin_sha256": sha(&cubin), "bytes": cubin.len(), "options": o, "seconds": t.elapsed().as_secs_f64(),
                           "log_tail": log.chars().rev().take(3000).collect::<String>().chars().rev().collect::<String>()})
                })}).collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let r = json!({"source_sha256": sha(&src), "wall_seconds": t0.elapsed().as_secs_f64(), "entries": res});
            println!("{}", serde_json::to_string_pretty(&r).unwrap());
        }
        Some("prove") => prove(&get("--cubin").unwrap(), &get("--gen-info").unwrap(), &all("--capture"), &get("--out").unwrap(),
            get("--seeds").map(|s| s.parse().unwrap()).unwrap_or(3), a.iter().any(|x| x == "--field-dots"),
            get("--budget-gb").map(|s| s.parse::<f64>().unwrap()).unwrap_or(6.0),
            get("--tests").unwrap_or("replay,identity,zero,dot,fd,cost".into())),
        _ => { eprintln!("usage: p3-mynn build|prove ..."); std::process::exit(2) }
    }
}

#[allow(clippy::too_many_arguments)]
fn prove(cubin: &str, info: &str, caps: &[String], out: &str, seeds: u64, field_dots: bool, budget_gb: f64, tests: String) {
    let has = |t: &str| tests.split(',').any(|x| x == t);
    init();
    set_stack(get_env_usize("P3_STACK", 32 * 1024));
    let cubin_b: Vec<u8> = (0..3).flat_map(|k| read(Path::new(&format!("{}.{}.cubin", cubin, k)))).collect();
    let info_v: Value = serde_json::from_slice(&read(Path::new(info))).unwrap();
    let budget = (budget_gb * 1e9) as usize;
    let mut results = Vec::new();
    for cdir in caps {
        let t0 = std::time::Instant::now();
        let c = Capture::load(Path::new(cdir));
        let ni = (c.int("ime") - c.int("ims") + 1) as usize;
        eprintln!("== {} ({} columns, step file {}) loaded in {:.1}s", cdir, c.int("batch_count"), c.dir.display(), t0.elapsed().as_secs_f64());
        let mut r = serde_json::Map::new();
        r.insert("capture".into(), json!(cdir));
        r.insert("image_sha256".into(), json!(c.image_sha));
        r.insert("columns".into(), json!(c.int("batch_count")));
        r.insert("initflag".into(), json!(c.int("initflag")));
        let orig = Original::new(&c);
        let base = BTreeMap::new();
        // 1. replay of the unchanged forward image
        let fwd_ms = orig.run(&c, &base);
        let mut diff_words = 0u64; let mut total = 0u64;
        for x in c.args.iter().filter(|x| x.device && !x.post.is_empty()) {
            let got = orig.get(&x.name);
            total += (x.post.len() / 4) as u64;
            diff_words += got.chunks_exact(4).zip(x.post.chunks_exact(4)).filter(|(p, q)| p != q).count() as u64;
        }
        r.insert("replay".into(), json!({"forward_ms": fwd_ms, "words": total, "differing_words": diff_words}));
        if has("lanes") {
            // the same captured image's one-lane entry (32 columns per warp tile, the layout the derivative entries use)
            for (entry, lanes) in [("rrfs_mynnedmf_run_l1", 1u32), ("rrfs_mynnedmf_run_l4", 4u32)] {
                let f = orig.m.func(entry);
                let cols = c.int("batch_count") as usize;
                let tile = 32 / lanes as usize;
                let ws = u64::from_le_bytes(c.arg("workspace_stride").scalar[..8].try_into().unwrap()) as usize;
                let wsbuf = DBuf::new(cols.div_ceil(tile) * tile * ws + (1 << 20));
                for a in c.args.iter().filter(|a| a.device) { orig.bufs[&a.name].upload(&a.pre); }
                let args: Vec<KArg> = c.args.iter().map(|a| if a.name == "workspace" { KArg::Ptr(wsbuf.ptr) } else if a.device { KArg::Ptr(orig.bufs[&a.name].ptr) } else { KArg::Bytes(a.scalar.clone()) }).collect();
                let threads = cols as u32 * lanes;
                launch(f, [threads.div_ceil(32), 1, 1], [32, 1, 1], 0, &args);
                sync();
                let mut d = 0u64;
                for x in c.args.iter().filter(|x| x.device && !x.post.is_empty() && x.name != "workspace") {
                    d += orig.get(&x.name).chunks_exact(4).zip(x.post.chunks_exact(4)).filter(|(p, q)| p != q).count() as u64;
                }
                eprintln!("lanes: captured image entry {} ({} lanes per column) vs captured post-state: {} words differ", entry, lanes, d);
                r.insert(format!("replay_{}", entry), json!({"differing_words": d}));
            }
        }
        eprintln!("replay: {} of {} words differ, forward {:.2} ms", diff_words, total, fwd_ms);

        let br_words: usize = get_env_usize("P3_BR_WORDS", 8192);
        let g = Gen::new(cubin, &info_v, &c, br_words);
        let nf = g.names.len();
        let (rt, lt) = func_info(g.tl); let (ra, la) = func_info(g.ad);
        r.insert("entries".into(), json!({"tl_regs": rt, "tl_local_bytes": lt, "ad_regs": ra, "ad_local_bytes": la,
            "sizeof_tangent_elem": g.sz_tl, "sizeof_tape_elem": g.sz_tape, "arena_bytes_per_column": g.arena_col, "packed_elems_per_column": g.col_elems}));
        // identity at the base state + branch record
        g.set_inputs(&c, &base);
        g.set_globals(true);
        let tl0_ms = g.tl_run(&c, None);
        let base_rec = g.branch_record();
        let hw: Vec<u64> = g.used_hw.download().chunks_exact(8).map(|x| u64::from_le_bytes(x.try_into().unwrap())).collect();
        r.insert("arena_high_water_bytes_per_column".into(), json!({"max": hw.iter().max(), "limit": g.arena_col}));
        let dom: Vec<i32> = g.dom.download().chunks_exact(4).map(|x| i32::from_le_bytes(x.try_into().unwrap())).collect();
        let dcols: Vec<(usize, i32)> = dom.iter().enumerate().filter(|(_, v)| **v != 0).map(|(k, v)| (k, *v)).collect();
        eprintln!("domain: {} columns with a derivative-undefined operation on a dependent value: {:?}", dcols.len(), &dcols[..dcols.len().min(10)]);
        r.insert("domain_columns".into(), json!(dcols));
        let bf = c.int("batch_first") as usize;
        let in_cols = |idx: usize| { let col = idx % ni; col >= bf && col < bf + g.cols };
        // P3_COLSET (independent check): file of absolute column indices (0-based i); dot and fd/path perturb, seed
        // and grade only those columns (columns are independent, so this is a per-regime test of the same launch).
        let colset: Option<Vec<bool>> = std::env::var("P3_COLSET").ok().map(|p| { let mut m = vec![false; ni];
            for l in std::fs::read_to_string(&p).unwrap().lines() { if let Ok(v) = l.trim().parse::<usize>() { if v < ni { m[v] = true; } } } m });
        let in_set = |idx: usize| colset.as_ref().map(|m| m[idx % ni]).unwrap_or(true);
        if let Some(m) = &colset { r.insert("colset".into(), json!({"file": std::env::var("P3_COLSET").unwrap(), "columns": m.iter().filter(|v| **v).count()})); }
        let mut id_diff = 0u64; let mut id_total = 0u64; let mut nonzero_tangent = 0u64; let mut worst = Vec::new();
        for (i, n) in g.names.iter().enumerate() {
            let y = g.y[i].download(); let post = &c.arg(n).post;
            let d = y.chunks_exact(4).zip(post.chunks_exact(4)).enumerate().filter(|(k, (p, q))| in_cols(*k) && p != q).count() as u64;
            if d > 0 { worst.push(json!([n, d])); }
            id_diff += d; id_total += (post.len() / 4) as u64;
            nonzero_tangent += f64s(&g.tout[i].download()).iter().filter(|v| **v != 0.0).count() as u64;
        }
        let kd = g.kout.download().chunks_exact(4).zip(c.arg("kpbl").post.chunks_exact(4)).filter(|(p, q)| p != q).count();
        let ed = g.eflags.download().chunks_exact(4).zip(c.arg("error_flags").post.chunks_exact(4)).filter(|(p, q)| p != q).count();
        let comps: Vec<u64> = base_rec.1.clone();
        r.insert("identity".into(), json!({"float_words": id_total, "differing_float_words": id_diff, "differing_fields": worst,
            "kpbl_differing": kd, "error_flags_differing": ed, "tl_ms_zero_seed": tl0_ms,
            "zero_seed_nonzero_tangent_words": nonzero_tangent,
            "comparisons_per_column": {"min": comps.iter().min(), "max": comps.iter().max(), "mean": comps.iter().sum::<u64>() as f64 / comps.len() as f64,
                                        "record_capacity_bits": br_words * 32}}));
        eprintln!("identity: {} of {} float words differ (kpbl {}, error_flags {}); zero seed nonzero tangent words {}; TL {:.1} ms",
            id_diff, id_total, kd, ed, nonzero_tangent, tl0_ms);
        g.set_globals(false);
        if !has("dot") && !has("fd") && !has("path") && !has("seg") { results.push(Value::Object(r)); continue; }

        // fields
        let pre: Vec<Vec<f32>> = g.names.iter().map(|n| f32s(&c.arg(n).pre)).collect();
        let post: Vec<Vec<f32>> = g.names.iter().map(|n| f32s(&c.arg(n).post)).collect();
        let logical: Vec<usize> = g.names.iter().map(|n| { let a = c.arg(n); if a.device { a.shape.iter().product::<usize>() } else { 0 } }).collect();
        let lstd: Vec<Vec<f64>> = pre.iter().zip(&logical).map(|(v, &l)| { let mut s = level_std(&v[..l.min(v.len())], ni); s.resize(v.len(), 0.0); s }).collect();
        let orms: Vec<f64> = post.iter().map(|v| rms(v)).collect();
        // weight of each output field in the dot test: rms of what the launch changed (falls back to the field rms)
        let drms: Vec<f64> = (0..nf).map(|i| {
            let d: Vec<f32> = post[i].iter().zip(&pre[i]).map(|(a, b)| a - b).collect();
            let r = (d.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / d.len().max(1) as f64).sqrt();
            if r > 0.0 && r.is_finite() { r } else { orms[i] }
        }).collect();
        // P3_DOT_FLOOR (independent check): a field the launch changes only by rounding-size amounts gets a weight of
        // 1/drms ~ 1e40 and then one product dominates the whole dot sum, so nothing else is tested. Floor the
        // change scale at floor * field rms so every output field is weighted on a comparable footing.
        let dfloor: f64 = std::env::var("P3_DOT_FLOOR").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
        let drms: Vec<f64> = drms.iter().zip(&orms).map(|(d, o)| d.max(dfloor * o)).collect();
        if let Ok(fp) = std::env::var("P3_FEATURES") {
            let nk = (c.int("kme") - c.int("kms") + 1) as usize;
            let fi = |n: &str| g.names.iter().position(|x| x == n).unwrap();
            let (tt, qv, qc, qi, hfx, xl, pb, ps, p3) = (&pre[fi("tt")], &pre[fi("qv")], &pre[fi("qc")], &pre[fi("qi")], &pre[fi("hfx")], &pre[fi("xland")], &pre[fi("pblh")], &pre[fi("ps")], &pre[fi("p")]);
            let feats: Vec<Value> = (bf..bf + g.cols).map(|i| {
                let lv = |v: &Vec<f32>, k: usize| v[k * ni + i] as f64;
                // lowest 3 km is roughly the lowest 20 levels of this 59-level grid; use pressure > 700 hPa instead
                let low: Vec<usize> = (0..nk).filter(|&k| k * ni + i < p3.len() && lv(p3, k) > 70000.0).collect();
                let mqc = low.iter().map(|&k| lv(qc, k)).fold(0.0, f64::max);
                let mqi = low.iter().map(|&k| lv(qi, k)).fold(0.0, f64::max);
                let near0 = low.iter().map(|&k| (lv(tt, k) - 273.15).abs()).fold(1e9, f64::min);
                json!({"i": i, "t1": lv(tt, 0), "qv1": lv(qv, 0), "qc_low_max": mqc, "qi_low_max": mqi, "t_low_min_dist_0c": near0,
                       "hfx": hfx[i], "xland": xl[i], "pblh": pb[i], "ps": ps[i]})
            }).collect();
            std::fs::write(&fp, serde_json::to_string(&feats).unwrap()).unwrap();
        }
        // 2. dot tests
        if has("dot") {
            let mut dots = Vec::new();
            let zero_y: Vec<Vec<f64>> = pre.iter().map(|v| vec![0.0; v.len()]).collect();
            let (_, ad0, _, _, _) = g.ad_run(&c, &zero_y, budget);
            let xbar0_nz: usize = (0..nf).map(|i| f64s(&g.xbar[i].download()).iter().filter(|v| **v != 0.0).count()).sum();
            for seed in 1..=seeds {
                let mut rng = Rng(0x9e3779b97f4a7c15 ^ seed.wrapping_mul(0x2545F4914F6CDD1D));
                let dx: Vec<Vec<f64>> = (0..nf).map(|i| lstd[i].iter().enumerate().map(|(k, s)| { let z = rng.normal(); if k < logical[i] && in_set(k) { z * if *s > 0.0 { *s } else { 1e-3 } } else { 0.0 } }).collect()).collect();
                let mut yb: Vec<Vec<f64>> = (0..nf).map(|i| (0..pre[i].len()).map(|k| { let z = rng.normal(); if in_set(k) { z / drms[i] } else { 0.0 } }).collect()).collect();
                let tl_ms = g.tl_run(&c, Some(&dx));
                let tout: Vec<Vec<f64>> = (0..nf).map(|i| f64s(&g.tout[i].download())).collect();
                if std::env::var("P3_DOT_NORM").is_ok() {
                    // independent check: weight each output field by 1/|K_i dx| (else 1/|J_i dx|) so every field's term
                    // is O(1) and no single field (rqsblten, qdiss) hides the rest of the launch
                    for i in 0..nf {
                        let kn = tout[i].iter().zip(&dx[i]).map(|(t, d)| (t - d) * (t - d)).sum::<f64>().sqrt();
                        let jn = norm(&tout[i]);
                        let sc = if kn > 0.0 { kn } else if jn > 0.0 { jn } else { 1.0 };
                        let w = drms[i] / sc;
                        for v in yb[i].iter_mut() { *v *= w; }
                    }
                }
                let (cnt_ms, ad_ms, cap, used, tile) = g.ad_run(&c, &yb, budget);
                let xbar: Vec<Vec<f64>> = (0..nf).map(|i| f64s(&g.xbar[i].download())).collect();
                let lhs: f64 = (0..nf).map(|i| dot(&tout[i], &yb[i])).sum();
                { // largest output-field terms of <J dx, ybar> and the largest weights, so a dominating field is visible
                  let mut t: Vec<(f64, usize)> = (0..nf).map(|i| (dot(&tout[i], &yb[i]).abs(), i)).collect();
                  t.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                  let tot: f64 = t.iter().map(|x| x.0).sum();
                  eprintln!("dot terms (output field, |term|, share): {}", t.iter().take(5).map(|(v, i)| format!("{} {:.3e} {:.2e}", g.names[*i], v, v / tot.max(1e-300))).collect::<Vec<_>>().join("; "));
                  let mut w: Vec<(f64, usize)> = (0..nf).map(|i| (1.0 / drms[i], i)).collect();
                  w.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                  eprintln!("weights 1/drms (top 3): {}", w.iter().take(3).map(|(v, i)| format!("{} {:.2e}", g.names[*i], v)).collect::<Vec<_>>().join("; "));
                }
                let rhs: f64 = (0..nf).map(|i| dot(&dx[i], &xbar[i])).sum();
                let rel = (lhs - rhs).abs() / lhs.abs().max(rhs.abs()).max(1e-300);
                // every array is an in/out buffer, so J = I + K; grade K (what the launch does) on its own as well
                let kl: f64 = (0..nf).map(|i| tout[i].iter().zip(&dx[i]).zip(&yb[i]).map(|((t, d), y)| (t - d) * y).sum::<f64>()).sum();
                let kr: f64 = (0..nf).map(|i| xbar[i].iter().zip(&yb[i]).zip(&dx[i]).map(|((x, y), d)| (x - y) * d).sum::<f64>()).sum();
                let krel = (kl - kr).abs() / kl.abs().max(kr.abs()).max(1e-300);
                let kdiff: Vec<Vec<f64>> = (0..nf).map(|i| tout[i].iter().zip(&dx[i]).map(|(t, d)| t - d).collect()).collect();
                let kns = (kl - kr).abs() / ((0..nf).map(|i| norm(&kdiff[i]).powi(2)).sum::<f64>().sqrt() * (0..nf).map(|i| norm(&yb[i]).powi(2)).sum::<f64>().sqrt()).max(1e-300);
                let ns = (lhs - rhs).abs() / ((0..nf).map(|i| norm(&tout[i]).powi(2)).sum::<f64>().sqrt() * (0..nf).map(|i| norm(&yb[i]).powi(2)).sum::<f64>().sqrt()).max(1e-300);
                // second reverse, bitwise
                let (_, _, _, _, _) = g.ad_run(&c, &yb, budget);
                let xbar2: Vec<Vec<u8>> = (0..nf).map(|i| g.xbar[i].download()).collect();
                let rev_bitwise = (0..nf).all(|i| xbar2[i] == bytes_f64(&xbar[i]));
                let mut per_field = Vec::new();
                if field_dots && seed == 1 {
                    for f in 0..nf {
                        if pre[f].is_empty() { continue; }
                        let d1: Vec<Vec<f64>> = (0..nf).map(|i| if i == f { dx[i].clone() } else { vec![0.0; dx[i].len()] }).collect();
                        g.tl_run(&c, Some(&d1));
                        let l: f64 = (0..nf).map(|i| { let t = f64s(&g.tout[i].download()); if i == f { t.iter().zip(&dx[i]).zip(&yb[i]).map(|((t, d), y)| (t - d) * y).sum::<f64>() } else { dot(&t, &yb[i]) } }).sum();
                        let rr: f64 = xbar[f].iter().zip(&yb[f]).zip(&dx[f]).map(|((x, y), d)| (x - y) * d).sum();
                        let id = dot(&dx[f], &yb[f]);
                        let e = (l - rr).abs() / l.abs().max(rr.abs()).max(1e-300);
                        per_field.push(json!({"input": g.names[f], "tl_side": l, "ad_side": rr, "identity_part": id, "rel": e}));
                    }
                }
                let tape_mb: Vec<f64> = used.iter().map(|u| *u as f64 * NODE_BYTES as f64 / 1e6).collect();
                eprintln!("dot seed {}: whole J: lhs {:.12e} rhs {:.12e} rel {:.3e}; launch part K=J-I: lhs {:.12e} rhs {:.12e} rel {:.3e} norm-scaled {:.3e}; reverse repeat bitwise {}; TL {:.0} ms, count {:.0} ms, AD {:.0} ms, tape max {:.1} MB/column, tile {}",
                    seed, lhs, rhs, rel, kl, kr, krel, kns, rev_bitwise, tl_ms, cnt_ms, ad_ms, tape_mb.iter().cloned().fold(0.0, f64::max), tile);
                let worst_field = per_field.iter().map(|v| v["rel"].as_f64().unwrap()).fold(0.0, f64::max);
                dots.push(json!({"seed": seed, "lhs": lhs, "rhs": rhs, "rel": rel, "norm_scaled": ns, "k_lhs": kl, "k_rhs": kr, "k_rel": krel, "k_norm_scaled": kns, "reverse_repeat_bitwise": rev_bitwise,
                    "tl_ms": tl_ms, "count_ms": cnt_ms, "ad_ms": ad_ms, "tape_nodes_capacity": cap, "ad_tile_columns": tile,
                    "tape_mb_per_column": {"max": tape_mb.iter().cloned().fold(0.0, f64::max), "mean": tape_mb.iter().sum::<f64>() / tape_mb.len() as f64},
                    "per_input_field_worst_rel": if per_field.is_empty() { Value::Null } else { json!(worst_field) }, "per_input_field": per_field}));
            }
            r.insert("dot".into(), json!({"zero_ybar_nonzero_xbar_words": xbar0_nz, "zero_ybar_ad_ms": ad0, "seeds": dots}));
        }
        // 3. TL vs centred FD of the unchanged forward, with flips
        if has("fd") || has("path") || has("seg") {
            // direction: P3_DIR = level (level std), relative (|x|), realistic (0.5 K theta/T, 5% qv, 0.5 m/s u v w,
            // 10% condensate, numbers and turbulence moments); P3_FIELDS restricts the perturbed inputs.
            let modes: Vec<String> = std::env::var("P3_DIR").unwrap_or("realistic,level".into()).split(',').map(|s| s.to_string()).collect();
            let only: Vec<String> = std::env::var("P3_FIELDS").map(|s| s.split(',').map(|x| x.to_string()).collect()).unwrap_or_default();
            let realistic = |n: &str, x: f32| -> f64 { match n {
                "th" | "tt" => 0.5, "u" | "v" | "w" => 0.5, "qv" => 0.05 * x.abs() as f64,
                "qc" | "qi" | "qs" | "qc_bl" | "qi_bl" | "cldfra_bl" | "nc" | "ni" | "nifa" | "nwfa" | "qke" | "qke_adv" | "tsq" | "qsq" | "cov" | "el_pbl" => 0.1 * x.abs() as f64,
                "hfx" | "qfx" | "ust" | "ch" | "wspd" | "znt" | "ts" | "qsfc" | "ps" => 0.02 * x.abs() as f64,
                _ => 0.0 } };
            let active: Vec<usize> = (0..nf).filter(|&i| !GEOMETRY.contains(&g.names[i].as_str()) && (only.is_empty() || only.contains(&g.names[i]))).collect();
            let eps_list: Vec<f64> = std::env::var("P3_EPS").map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
                .unwrap_or(vec![0.1, 0.03, 0.01, 0.003, 0.001, 3e-4, 1e-4]).into_iter().filter(|_| has("fd")).collect::<Vec<f64>>();
            let bf = c.int("batch_first") as usize;
            let mut fdres = Vec::new();
            for mode in &modes { for seed in 1..=get_env_usize("P3_FD_SEEDS", 1) as u64 {
                let mut rng = Rng(0xD1B54A32D192ED03 ^ seed.wrapping_mul(0x9E3779B97F4A7C15));
                let dx: Vec<Vec<f64>> = (0..nf).map(|i| if active.contains(&i) {
                    pre[i].iter().zip(&lstd[i]).enumerate().map(|(k, (x, s))| if k >= logical[i] || !in_set(k) { 0.0 } else { rng.normal() * match mode.as_str() { "relative" => x.abs() as f64, "realistic" => realistic(&g.names[i], *x), _ => *s } }).collect()
                } else { vec![0.0; pre[i].len()] }).collect();
                g.tl_run(&c, Some(&dx));
                let tl: Vec<Vec<f64>> = (0..nf).map(|i| f64s(&g.tout[i].download())).collect();
                if let Ok(dn) = std::env::var("P3_DUMP") { for i in 0..nf { if g.names[i] == dn {
                    let ti = f64s(&g.tin[i].download());
                    for k in 0..8 { eprintln!("dump {} [{}]: pre {:e} tin {:e} dx {:e} tout {:e}", dn, k, pre[i][k], ti[k], dx[i][k], tl[i][k]); } } } }
                let mut curve = Vec::new();
                for &eps in &eps_list {
                    let mut outs = Vec::new(); let mut recs = Vec::new(); let mut gen_vs_orig = 0u64;
                    for sgn in [1.0f64, -1.0] {
                        let over: BTreeMap<String, Vec<f32>> = active.iter().map(|&i| (g.names[i].clone(),
                            pre[i].iter().zip(&dx[i]).map(|(x, d)| (*x as f64 + sgn * eps * d) as f32).collect())).collect();
                        orig.run(&c, &over);
                        let o: Vec<Vec<f32>> = g.names.iter().map(|n| f32s(&orig.get(n))).collect();
                        // generated primal at the same perturbed state, with the branch record
                        g.set_inputs(&c, &over); g.set_globals(true);
                        g.tl_run(&c, None);
                        recs.push(g.branch_record());
                        for i in 0..nf { gen_vs_orig += f32s(&g.y[i].download()).iter().zip(&o[i]).filter(|(p, q)| p.to_bits() != q.to_bits()).count() as u64; }
                        g.set_globals(false);
                        outs.push(o);
                    }
                    g.set_inputs(&c, &base);
                    // P3_FD_EFF (independent check): the perturbed inputs are float32, so x +- eps*dx is rounded and the
                    // step actually taken differs from eps*dx by up to half an ulp per element (6% of a 0.5 K * 1e-3
                    // theta step). Grade against the tangent along the step actually taken, (x+ - x-) / (2 eps).
                    let tle: Vec<Vec<f64>> = if std::env::var("P3_FD_EFF").is_ok() {
                        let de: Vec<Vec<f64>> = (0..nf).map(|i| if active.contains(&i) { pre[i].iter().zip(&dx[i]).map(|(x, d)| {
                            let p = (*x as f64 + eps * d) as f32; let m = (*x as f64 - eps * d) as f32; (p as f64 - m as f64) / (2.0 * eps) }).collect()
                        } else { vec![0.0; pre[i].len()] }).collect();
                        g.tl_run(&c, Some(&de));
                        (0..nf).map(|i| f64s(&g.tout[i].download())).collect()
                    } else { tl.clone() };
                    let tl = &tle;
                    // a column is branch-stable when both perturbed runs took exactly the base run's comparison outcomes
                    let stable_col: Vec<bool> = (0..g.cols).map(|k| (0..2).all(|s| recs[s].1[k] == base_rec.1[k] && recs[s].2[k] == base_rec.2[k])).collect();
                    let n_stable = stable_col.iter().filter(|v| **v).count();
                    let is_stable = |idx: usize| { let col = idx % ni; col >= bf && col < bf + g.cols && stable_col[col - bf] };
                    let mut fields = Vec::new();
                    // sums: [all columns, stable columns] x [state fields, aerosol numbers] x [err, fd]
                    let mut acc = [[[0.0f64; 2]; 2]; 2];
                    let mut cerr = vec![0.0f64; g.cols]; let mut cfd = vec![0.0f64; g.cols];
                    for i in 0..nf {
                        let fd: Vec<f64> = outs[0][i].iter().zip(&outs[1][i]).map(|(p, m)| (*p as f64 - *m as f64) / (2.0 * eps)).collect();
                        let (mut ne, mut nfd, mut ntl, mut se, mut sfd, mut stl) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                        let aer = AEROSOL.contains(&g.names[i].as_str());
                        for (k, (f, t)) in fd.iter().zip(&tl[i]).enumerate().take(logical[i]) {
                            if !in_set(k) { continue; }
                            let e2 = (t - f) * (t - f);
                            { let col = k % ni; if !aer && col >= bf && col < bf + g.cols { cerr[col - bf] += e2 / (orms[i] * orms[i]); cfd[col - bf] += f * f / (orms[i] * orms[i]); } }
                            ne += e2; nfd += f * f; ntl += t * t;
                            if is_stable(k) { se += e2; sfd += f * f; stl += t * t; }
                        }
                        if nfd == 0.0 && ntl == 0.0 { continue; }
                        let s2 = orms[i] * orms[i];
                        let a = AEROSOL.contains(&g.names[i].as_str()) as usize;
                        acc[0][a][0] += ne / s2; acc[0][a][1] += nfd / s2; acc[1][a][0] += se / s2; acc[1][a][1] += sfd / s2;
                        fields.push(json!({"field": g.names[i], "rel": ne.sqrt() / nfd.max(ntl).sqrt().max(1e-300), "fd_norm": nfd.sqrt(), "tl_norm": ntl.sqrt(), "dx_norm": norm(&dx[i]),
                            "rel_stable_columns": se.sqrt() / sfd.max(stl).sqrt().max(1e-300)}));
                    }
                    if std::env::var("P3_SPIKES").is_ok() {
                        // largest tangent-vs-difference gaps, output-rms weighted: where the exact tangent disagrees
                        let mut top: Vec<(f64, usize, usize)> = Vec::new();
                        for i in 0..nf { if AEROSOL.contains(&g.names[i].as_str()) { continue; }
                            for k in 0..logical[i] {
                                if !in_set(k) { continue; }
                                let f = (outs[0][i][k] as f64 - outs[1][i][k] as f64) / (2.0 * eps);
                                let e = (tl[i][k] - f).abs() / orms[i];
                                if e.is_finite() && e > 0.0 { top.push((e, i, k)); } } }
                        top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                        for &(e, i, k) in top.iter().take(12) {
                            let f = (outs[0][i][k] as f64 - outs[1][i][k] as f64) / (2.0 * eps);
                            eprintln!("spike eps {:.0e}: {} column {} level {} weighted gap {:.3e}: tangent {:.4e} difference {:.4e} pre {:.4e} post {:.4e}",
                                eps, g.names[i], k % ni, k / ni + 1, e, tl[i][k], f, pre[i][k], post[i][k]);
                        }
                    }
                    let q = |x: [f64; 2]| (x[0] / x[1].max(1e-300)).sqrt();
                    let (rel, rela, rels, relsa) = (q(acc[0][0]), q(acc[0][1]), q(acc[1][0]), q(acc[1][1]));
                    let fp = flips(&base_rec, &recs[0], g.br_words); let fm = flips(&base_rec, &recs[1], g.br_words);
                    let mut fl: Vec<u64> = fp.iter().zip(&fm).map(|(a, b)| *a.max(b)).collect();
                    fl.sort();
                    eprintln!("fd seed {} eps {:.0e} dir {}: all columns rel {:.3e} (aerosol numbers {:.3e}); branch-stable columns {}/{}: rel {:.3e} (aerosol {:.3e}); outcome-sequence differences per flipped column median {}; generated primal vs forward at perturbed states: {} words differ",
                        seed, eps, mode, rel, rela, n_stable, g.cols, rels, relsa, fl[fl.len() / 2], gen_vs_orig);
                    let percol: Vec<Value> = (0..g.cols).filter(|&k| in_set(bf + k) && colset.is_some())
                        .map(|k| json!([bf + k, stable_col[k], (cerr[k] / cfd[k].max(1e-300)).sqrt(), cfd[k].sqrt()])).collect();
                    let n_stable = (0..g.cols).filter(|&k| in_set(bf + k) && stable_col[k]).count();
                    curve.push(json!({"per_column": percol, "eps": eps, "rel_state_scaled": rel, "rel_aerosol_numbers": rela,
                        "branch_stable_columns": n_stable, "rel_state_scaled_stable_columns": rels, "rel_aerosol_numbers_stable_columns": relsa,
                        "outcome_sequence_differences_per_column": {"median": fl[fl.len() / 2], "p90": fl[fl.len() * 9 / 10], "max": fl[fl.len() - 1],
                            "note": "Hamming distance of the recorded outcome sequences plus length change; after the first flip the sequences misalign, so this is an upper bound on switches flipped"},
                        "generated_vs_forward_differing_words": gen_vs_orig, "fields": fields}));
                }
                // Path test: the average of the exact executed-branch tangent along the segment x + t dx, |t| <= eps
                // (midpoint rule, P3_PATH_N points), against the forward's centred difference over the same segment.
                // Where the forward is continuous along the segment (kinks and min/max switches included) the two agree
                // up to quadrature and float32 rounding, whatever switches flip; what remains is jumps.
                let mut path = Vec::new();
                if has("path") {
                    let np = get_env_usize("P3_PATH_N", 32);
                    let path_eps: Vec<f64> = std::env::var("P3_PATH_EPS").map(|s| s.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![0.1, 0.01]);
                    for &eps in &path_eps {
                        let mut avg: Vec<Vec<f64>> = (0..nf).map(|i| vec![0.0; pre[i].len()]).collect();
                        for j in 0..np {
                            let t = -eps + (j as f64 + 0.5) * 2.0 * eps / np as f64;
                            let over: BTreeMap<String, Vec<f32>> = active.iter().map(|&i| (g.names[i].clone(),
                                pre[i].iter().zip(&dx[i]).map(|(x, d)| (*x as f64 + t * d) as f32).collect())).collect();
                            g.set_inputs(&c, &over);
                            g.tl_run(&c, Some(&dx));
                            for i in 0..nf { for (a, v) in avg[i].iter_mut().zip(f64s(&g.tout[i].download())) { *a += v / np as f64; } }
                        }
                        g.set_inputs(&c, &base);
                        let mut outs = Vec::new();
                        for sgn in [1.0f64, -1.0] {
                            let over: BTreeMap<String, Vec<f32>> = active.iter().map(|&i| (g.names[i].clone(),
                                pre[i].iter().zip(&dx[i]).map(|(x, d)| (*x as f64 + sgn * eps * d) as f32).collect())).collect();
                            orig.run(&c, &over);
                            outs.push(g.names.iter().map(|n| f32s(&orig.get(n))).collect::<Vec<Vec<f32>>>());
                        }
                        let mut acc = [[0.0f64; 2]; 2]; // [state, aerosol] x [err, fd]
                        let mut colerr = vec![0.0f64; g.cols]; let mut colfd = vec![0.0f64; g.cols];
                        let mut fields = Vec::new();
                        for i in 0..nf {
                            let (mut ne, mut nfd) = (0.0, 0.0);
                            for k in 0..logical[i] {
                                if !in_set(k) { continue; }
                                let f = (outs[0][i][k] as f64 - outs[1][i][k] as f64) / (2.0 * eps);
                                let e = avg[i][k] - f;
                                ne += e * e; nfd += f * f;
                                let col = k % ni;
                                if col >= bf && col < bf + g.cols { colerr[col - bf] += e * e / (orms[i] * orms[i]); colfd[col - bf] += f * f / (orms[i] * orms[i]); }
                            }
                            if nfd == 0.0 { continue; }
                            let a = AEROSOL.contains(&g.names[i].as_str()) as usize;
                            acc[a][0] += ne / (orms[i] * orms[i]); acc[a][1] += nfd / (orms[i] * orms[i]);
                            fields.push(json!({"field": g.names[i], "rel": (ne / nfd).sqrt()}));
                        }
                        let pcol: Vec<Value> = (0..g.cols).filter(|&k| colset.is_some() && in_set(bf + k)).map(|k| json!([bf + k, (colerr[k] / colfd[k].max(1e-300)).sqrt()])).collect();
                        let mut cr: Vec<f64> = (0..g.cols).filter(|&k| in_set(bf + k)).map(|k| (colerr[k] / colfd[k].max(1e-300)).sqrt()).collect();
                        cr.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let frac = |th: f64| cr.iter().filter(|v| **v <= th).count();
                        let (rs, ra) = ((acc[0][0] / acc[0][1]).sqrt(), (acc[1][0] / acc[1][1]).sqrt());
                        eprintln!("path dir {} eps {:.0e} n {}: mean tangent along the segment vs forward difference: state-scaled rel {:.3e} (aerosol {:.3e}); columns within 1e-3: {}/{}, within 1e-2: {}, median column rel {:.2e}",
                            mode, eps, np, rs, ra, frac(1e-3), cr.len(), frac(1e-2), cr[cr.len() / 2]);
                        path.push(json!({"eps": eps, "points": np, "rel_state_scaled": rs, "rel_aerosol_numbers": ra,
                            "columns_within_1e-3": frac(1e-3), "columns_within_1e-2": frac(1e-2), "column_rel_median": cr[cr.len() / 2],
                            "column_rel_p90": cr[cr.len() * 9 / 10], "per_column": pcol, "fields": fields}));
                    }
                }
                // Segment test (independent check, P3_SEG_EPS, P3_SEG_N): split x + t dx, |t| <= eps, into N pieces. On
                // each piece compare the forward's own difference (y(t_j+1) - y(t_j)) / dt with the exact tangent at the
                // piece's midpoint along the step actually taken between the two float32 endpoints. A jump or a kink
                // spoils only the piece it falls in; a wrong derivative spoils every piece of the column.
                let mut segres = Vec::new();
                if has("seg") {
                    let sn = get_env_usize("P3_SEG_N", 16);
                    let seg_eps: Vec<f64> = std::env::var("P3_SEG_EPS").map(|s| s.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![0.1]);
                    for &eps in &seg_eps {
                        let dt = 2.0 * eps / sn as f64;
                        let pt = |t: f64| -> BTreeMap<String, Vec<f32>> { active.iter().map(|&i| (g.names[i].clone(),
                            pre[i].iter().zip(&dx[i]).map(|(x, d)| (*x as f64 + t * d) as f32).collect())).collect() };
                        let mut ys: Vec<Vec<Vec<f32>>> = Vec::new();
                        for j in 0..=sn { orig.run(&c, &pt(-eps + j as f64 * dt)); ys.push(g.names.iter().map(|n| f32s(&orig.get(n))).collect()); }
                        // per column, per piece: state-scaled error and scale
                        let mut perr = vec![vec![0.0f64; sn]; g.cols]; let mut pfd = vec![vec![0.0f64; sn]; g.cols];
                        let mut ferr = vec![0.0f64; nf]; let mut ffd = vec![0.0f64; nf];
                        // smooth-gated grading (independent check): words graded, and the tangent per piece on them
                        let gidx: Vec<Vec<usize>> = (0..nf).map(|i| if AEROSOL.contains(&g.names[i].as_str()) { vec![] } else {
                            (0..logical[i]).filter(|&k| { let col = k % ni; in_set(k) && col >= bf && col < bf + g.cols }).collect() }).collect();
                        let mut tstore: Vec<Vec<Vec<f64>>> = vec![vec![]; sn];
                        for j in 0..sn {
                            let (a, b) = (pt(-eps + j as f64 * dt), pt(-eps + (j + 1) as f64 * dt));
                            let mid: BTreeMap<String, Vec<f32>> = a.iter().map(|(n, va)| (n.clone(), va.iter().zip(&b[n]).map(|(p, q)| ((*p as f64 + *q as f64) / 2.0) as f32).collect())).collect();
                            let de: Vec<Vec<f64>> = (0..nf).map(|i| match (a.get(&g.names[i]), b.get(&g.names[i])) {
                                (Some(va), Some(vb)) => va.iter().zip(vb).map(|(p, q)| (*q as f64 - *p as f64) / dt).collect(),
                                _ => vec![0.0; pre[i].len()] }).collect();
                            g.set_inputs(&c, &mid);
                            g.tl_run(&c, Some(&de));
                            for i in 0..nf {
                                if AEROSOL.contains(&g.names[i].as_str()) { tstore[j].push(vec![]); continue; }
                                let t = f64s(&g.tout[i].download());
                                tstore[j].push(gidx[i].iter().map(|&k| t[k]).collect());
                                let s2 = orms[i] * orms[i];
                                for k in 0..logical[i] {
                                    if !in_set(k) { continue; }
                                    let col = k % ni; if col < bf || col >= bf + g.cols { continue; }
                                    let f = (ys[j + 1][i][k] as f64 - ys[j][i][k] as f64) / dt;
                                    perr[col - bf][j] += (t[k] - f) * (t[k] - f) / s2; pfd[col - bf][j] += f * f / s2;
                                    ferr[i] += (t[k] - f) * (t[k] - f) / s2; ffd[i] += f * f / s2;
                                }
                            }
                        }
                        g.set_inputs(&c, &base);
                        // Smooth gate: piece j of a word is graded only where the forward's own differences on the
                        // neighbouring pieces agree with it (|fd_j - fd_j-1| + |fd_j+1 - fd_j| <= 2% |fd_j|), i.e. the
                        // forward is smooth and above its float32 rounding there. A jump or rounding noise fails the gate;
                        // a missing or wrong derivative passes it and shows as error.
                        let mut gated = Vec::new(); let (mut gse, mut gsf, mut gn, mut gbad) = (0.0f64, 0.0f64, 0usize, 0usize);
                        let mut gworst: Vec<(f64, usize, usize, usize, f64, f64)> = Vec::new();
                        let gate: f64 = std::env::var("P3_GATE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.02);
                        let mut gcol = vec![[0.0f64; 3]; g.cols];
                        for i in 0..nf { if gidx[i].is_empty() { continue; }
                            let (mut se, mut sf, mut n, mut tot) = (0.0f64, 0.0f64, 0usize, 0usize);
                            let mut hist = [0usize; 4]; // gated pieces with relative error <=1e-3, <=1e-2, <=1e-1, >1e-1
                            for (p, &k) in gidx[i].iter().enumerate() {
                                let fdv: Vec<f64> = (0..sn).map(|j| (ys[j + 1][i][k] as f64 - ys[j][i][k] as f64) / dt).collect();
                                for j in 1..sn - 1 { tot += 1;
                                    let f = fdv[j]; if f == 0.0 { continue; }
                                    if (f - fdv[j - 1]).abs() + (fdv[j + 1] - f).abs() > gate * f.abs() { continue; }
                                    // and the forward's change over each of the three pieces is at least 256 float32 ulps of the
                                    // value (else the difference is a rounding staircase: one ulp over dt looks smooth but is noise)
                                    let ulp = |v: f32| { let a = v.abs(); if a == 0.0 { f32::MIN_POSITIVE as f64 } else { (f32::from_bits(a.to_bits() + 1) - a) as f64 } };
                                    if (j - 1..=j + 1).any(|q| ((ys[q + 1][i][k] as f64 - ys[q][i][k] as f64).abs()) < 256.0 * ulp(ys[q][i][k]).max(ulp(ys[q + 1][i][k]))) { continue; }
                                    // a staircase (the same change on all three pieces, bit for bit) is the rounding unit of an
                                    // intermediate (e.g. a tendency (u_new - u) / delt quantised by ulp(u)), not a derivative
                                    if fdv[j - 1] == fdv[j] && fdv[j] == fdv[j + 1] { continue; }
                                    let t = tstore[j][i][p]; let e = (t - f) * (t - f);
                                    se += e; sf += f * f; n += 1;
                                    { let gc = &mut gcol[k % ni - bf]; gc[0] += e / (orms[i] * orms[i]); gc[1] += f * f / (orms[i] * orms[i]); gc[2] += 1.0; }
                                    let r = (t - f).abs() / f.abs(); if r > 0.1 { gbad += 1; gworst.push((r, i, k, j, t, f)); }
                                    hist[if r <= 1e-3 { 0 } else if r <= 1e-2 { 1 } else if r <= 1e-1 { 2 } else { 3 }] += 1;
                                }
                            }
                            if n > 0 { let s2 = orms[i] * orms[i]; gse += se / s2; gsf += sf / s2; gn += n;
                                gated.push(json!([g.names[i], n, tot, (se / sf).sqrt(), hist])); }
                        }
                        gworst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                        let mut gl: Vec<&Value> = gated.iter().collect(); gl.sort_by(|a, b| b[3].as_f64().unwrap().partial_cmp(&a[3].as_f64().unwrap()).unwrap());
                        eprintln!("seg eps {:.0e} smooth-gated: {} word-pieces, state-scaled rel {:.3e}, pieces off by >10%: {}; worst fields (field, n, of, rel): {}", eps, gn, (gse / gsf.max(1e-300)).sqrt(), gbad,
                            gl.iter().take(6).map(|v| format!("{} {} {} {:.2e}", v[0].as_str().unwrap(), v[1], v[2], v[3].as_f64().unwrap())).collect::<Vec<_>>().join("; "));
                        { let mut h = [0usize; 4]; for v in &gated { for q in 0..4 { h[q] += v[4][q].as_u64().unwrap() as usize; } }
                          eprintln!("seg eps {:.0e} smooth-gated relative error per word-piece: <=1e-3 {}, <=1e-2 {}, <=1e-1 {}, >1e-1 {}", eps, h[0], h[1], h[2], h[3]); }
                        for w in gworst.iter().take(6) { let p = gidx[w.1].iter().position(|&q| q == w.2).unwrap();
                            let fdq = |j: usize| (ys[j + 1][w.1][w.2] as f64 - ys[j][w.1][w.2] as f64) / dt;
                            eprintln!("  gated worst: {} col {} lev {} piece {}: tangent {:.4e} fd {:.4e} (off {:.2e}); pieces j-1,j,j+1: tangent {:.4e} {:.4e} {:.4e}, fd {:.4e} {:.4e} {:.4e}; value {:.6e}",
                                g.names[w.1], w.2 % ni, w.2 / ni, w.3, w.4, w.5, w.0, tstore[w.3 - 1][w.1][p], tstore[w.3][w.1][p], tstore[w.3 + 1][w.1][p], fdq(w.3 - 1), fdq(w.3), fdq(w.3 + 1), ys[w.3][w.1][w.2]); }
                        if let Ok(sd) = std::env::var("P3_SEGDUMP") { for spec in sd.split(';') { let v: Vec<&str> = spec.split(':').collect();
                            if let (Some(i), Ok(col), Ok(lev)) = (g.names.iter().position(|n| n == v[0]), v[1].parse::<usize>(), v[2].parse::<usize>()) {
                                let k = lev * ni + col; if let Some(p) = gidx[i].iter().position(|&q| q == k) {
                                for j in 0..=sn { eprintln!("segdump {} col {} lev {} point {}: y {:.9e} (bits {:08x}){}", v[0], col, lev, j, ys[j][i][k], ys[j][i][k].to_bits(),
                                    if j < sn { format!("; piece fd {:.6e} tangent {:.6e}", (ys[j + 1][i][k] as f64 - ys[j][i][k] as f64) / dt, tstore[j][i][p]) } else { String::new() }); } } } } }
                        let mut cols_out = Vec::new(); let mut meds = Vec::new(); let mut all_bad = 0usize; let mut good_frac_sum = 0.0; let mut ncol = 0usize;
                        for k in 0..g.cols {
                            if !in_set(bf + k) { continue; }
                            let r: Vec<f64> = (0..sn).map(|j| (perr[k][j] / pfd[k][j].max(1e-300)).sqrt()).collect();
                            if pfd[k].iter().all(|v| *v == 0.0) { continue; }
                            let mut srt = r.clone(); srt.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            let med = srt[sn / 2]; let good = r.iter().filter(|v| **v <= 1e-2).count();
                            if good == 0 { all_bad += 1; }
                            good_frac_sum += good as f64 / sn as f64; ncol += 1; meds.push(med);
                            cols_out.push(json!([bf + k, med, good, r]));
                        }
                        meds.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let q = |p: f64| if meds.is_empty() { f64::NAN } else { meds[((meds.len() - 1) as f64 * p) as usize] };
                        eprintln!("seg dir {} eps {:.0e} n {}: {} columns; per-column median piece error: median {:.2e} p90 {:.2e} max {:.2e}; mean share of pieces within 1e-2 {:.3}; columns with no piece within 1e-2: {}",
                            mode, eps, sn, ncol, q(0.5), q(0.9), q(1.0), good_frac_sum / ncol.max(1) as f64, all_bad);
                        let mut fl: Vec<(f64, usize)> = (0..nf).filter(|&i| ffd[i] > 0.0 || ferr[i] > 0.0).map(|i| (ferr[i], i)).collect();
                        fl.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                        let tot_fd: f64 = ffd.iter().sum();
                        eprintln!("seg eps {:.0e} largest field errors (field, err^2 share of total fd^2, own rel): {}", eps,
                            fl.iter().take(6).map(|(e, i)| format!("{} {:.2e} {:.2e}", g.names[*i], e / tot_fd.max(1e-300), (e / ffd[*i].max(1e-300)).sqrt())).collect::<Vec<_>>().join("; "));
                        let gpc: Vec<Value> = (0..g.cols).filter(|&k| gcol[k][2] > 0.0).map(|k| json!([bf + k, gcol[k][2], (gcol[k][0] / gcol[k][1].max(1e-300)).sqrt()])).collect();
                        segres.push(json!({"smooth_gated": {"gate": gate, "per_column": gpc, "word_pieces": gn, "rel_state_scaled": (gse / gsf.max(1e-300)).sqrt(), "off_by_10pct": gbad, "fields": gated}, "fields": fl.iter().map(|(e, i)| json!([g.names[*i], e / tot_fd.max(1e-300), (e / ffd[*i].max(1e-300)).sqrt()])).collect::<Vec<_>>(), "eps": eps, "pieces": sn, "columns": ncol, "column_median_piece_err": {"median": q(0.5), "p90": q(0.9), "max": q(1.0)},
                            "mean_share_pieces_within_1e-2": good_frac_sum / ncol.max(1) as f64, "columns_no_piece_within_1e-2": all_bad, "per_column": cols_out}));
                    }
                }
                fdres.push(json!({"seg": segres, "seed": seed, "direction": mode, "perturbed_fields": active.iter().map(|&i| g.names[i].clone()).collect::<Vec<_>>(), "curve": curve, "path": path}));
            } }
            r.insert("fd".into(), json!(fdres));
        }
        let (free, total) = mem_info();
        r.insert("device_memory_in_use_gb".into(), json!((total - free) as f64 / 1e9));
        r.insert("wall_seconds".into(), json!(t0.elapsed().as_secs_f64()));
        results.push(Value::Object(r));
        std::fs::write(out, serde_json::to_string_pretty(&json!({"unit": "P3 MYNN-EDMF c6d08709 whole launch", "cubin_sha256": sha(&cubin_b), "results": results})).unwrap()).unwrap();
    }
    std::fs::write(out, serde_json::to_string_pretty(&json!({"unit": "P3 MYNN-EDMF c6d08709 whole launch", "cubin_sha256": sha(&cubin_b), "results": results})).unwrap()).unwrap();
}
fn get_env_usize(k: &str, d: usize) -> usize { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }
