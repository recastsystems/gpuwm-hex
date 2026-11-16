// SPDX-License-Identifier: Apache-2.0
//! Derivative entries over the sealed plan. Each plan kernel launch is bound to
//! its spec's launch_abi (ad.entry/v1): primal arguments to the forward's own
//! device storage (pre-launch snapshot, or live post-state where `post`), tangent
//! and adjoint arguments to one shadow per forward allocation at the same byte
//! offset (scaled for float64 shadows), so aliasing in the forward is aliasing in
//! the derivative. Scalar parameters get one-element device cells.
use crate::cuda::{checked, Cuda};
use crate::dev::{self, Arena};
use crate::expr::{self, Env};
use crate::submit;
use serde_json::Value;
use std::{collections::{BTreeMap, BTreeSet, HashMap}, ffi::{c_void, CString}, path::Path, ptr};

pub struct KernelDeriv {
    pub function: String,
    pub spec: Value,
    pub names: Vec<String>,
    pub meta: Vec<Value>,
    pub module: *mut c_void,
    pub functions: HashMap<String, *mut c_void>,
    pub tl_order: Vec<String>,
    pub ad_order: Vec<String>,
    pub aug: bool,
    pub unsupported: Option<String>,
    /// Derived topology inputs (content-addressed by the spec SHA256), on the device.
    pub derived: std::cell::RefCell<HashMap<String, (u64, Vec<u8>)>>,
}

fn order(abi: &Value, mode: &str) -> Vec<String> {
    let pick = |v: &Value| v.as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_owned)).collect::<Vec<_>>());
    if let Some(v) = pick(&abi["launch_order"][mode]) { return v; }
    let key = if mode == "tl" { "tangent_launch_order" } else { "adjoint_launch_order" };
    if let Some(v) = pick(&abi[key]) { return v; }
    if mode == "ad" { if let Some(v) = pick(&abi["reverse_order"]) { return v; } }
    vec![mode.to_owned()]
}

/// Reasons a spec cannot be dispatched by this unit yet (named, never skipped silently).
fn unsupported(spec: &Value, entries: &[String], derived: &HashMap<String, Vec<u8>>) -> Option<String> {
    let abi = &spec["launch_abi"];
    if abi["schema"] != "ad.entry/v1" { return Some("no ad.entry/v1 launch_abi".into()); }
    for name in entries {
        let e = &abi["entries"][name];
        if !e.is_object() { return Some(format!("missing entry {name}")); }
        for b in e["args"].as_array().into_iter().flatten() {
            let kind = b["bind"].as_str().unwrap_or("");
            if kind == "derived" && !b["sha256"].as_str().is_some_and(|h| derived.contains_key(h)) && !constructible(b) { return Some(format!("derived binding {} ({}) not in the derived store and no recipe", b["name"].as_str().or(b["output"].as_str()).unwrap_or("?"), b["recipe"].as_str().unwrap_or("?"))); }
            if b.get("indirect").is_some() && b["indirect"]["mirror"] != true { return Some("indirect pointer-table binding without mirror (U3)".into()); }
            if kind == "scalar" && b.get("arg").is_none() && b.get("scalar_arg").is_none() && b.get("value").is_none() && b.get("derived_from").is_none()
                && b.get("value_from").is_none() {
                return Some(format!("scalar binding without arg/value: {b}"));
            }
            if kind == "workspace" && b.get("routing").is_some() { return Some("workspace alias routing (U3)".into()); }
        }
    }
    None
}

/// Where a kernel-set row's derivative spec is read. A set records each spec at the absolute path it was generated at.
/// THE BREAKAGE (hex 0.5.0 integration): a set copied to another machine (the doors' operator row) read the generating
/// machine's files, and stopped where they do not exist. A copy beside the set, `<kernels.json folder>/specs/<recorded
/// path without its leading />` (the layout of tree/ad/kset/int7f), is read first; otherwise the recorded path, for a set
/// that lives where it was made. Admission (window::check_spec_hashes) hashes the same file this loader reads.
pub fn spec_path(kernels: &Path, recorded: &str) -> std::path::PathBuf {
    let local = kernels.parent().unwrap_or(Path::new(".")).join("specs").join(recorded.trim_start_matches('/'));
    if local.is_file() { local } else { std::path::PathBuf::from(recorded) }
}

pub unsafe fn load(cuda: &Cuda, kernels: &Path, cubins: &Path, only: Option<&BTreeSet<String>>) -> Result<HashMap<String, KernelDeriv>, String> {
    // Content-addressed derived topology arrays (file name = SHA256 of the bytes, checked here).
    let mut derived: HashMap<String, Vec<u8>> = HashMap::new();
    if let Ok(dir) = std::fs::read_dir(cubins.parent().unwrap_or(Path::new(".")).join("derived")) {
        for e in dir.flatten() {
            let bytes = std::fs::read(e.path()).map_err(|e| e.to_string())?;
            let h = crate::exec::digest(&bytes);
            if e.file_name().to_string_lossy().starts_with(&h) { derived.insert(h, bytes); }
        }
    }
    let set: Value = serde_json::from_slice(&std::fs::read(kernels).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let mut out = HashMap::new();
    for k in set["kernels"].as_array().ok_or("kernels missing")? {
        let function = k["function"].as_str().ok_or("function")?.to_owned();
        if only.is_some_and(|o| !o.contains(&function)) { continue; }
        let d = &k["derivative"];
        if !d.is_object() { continue; }
        let spec: Value = serde_json::from_slice(&std::fs::read(spec_path(kernels, d["spec"]["path"].as_str().ok_or("spec path")?)).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        let abi = &spec["launch_abi"];
        let names: Vec<String> = k["arguments"].as_array().ok_or("arguments")?.iter()
            .map(|a| a["name"].as_str().unwrap_or("").to_owned()).collect();
        let meta = k["arguments"].as_array().unwrap().clone();
        let aug = abi["entries"]["aug"].is_object();
        let tl_order = order(abi, "tl");
        let ad_order = order(abi, "ad");
        let mut all: Vec<String> = tl_order.iter().chain(ad_order.iter()).cloned().collect();
        if aug { all.push("aug".into()); }
        let mut why = if abi.is_object() { unsupported(&spec, &all, &derived) } else { Some("no launch_abi".into()) };
        let mut module = ptr::null_mut();
        let mut functions = HashMap::new();
        if why.is_none() {
            let path = cubins.join(format!("{function}.fatbin"));
            match std::fs::read(&path) {
                Err(e) => why = Some(format!("no derivative image {}: {e}", path.display())),
                Ok(image) => {
                    checked(&format!("cuModuleLoadData {function}"), (cuda.module_load)(&mut module, image.as_ptr().cast()))?;
                    for name in &all {
                        let f = abi["entries"][name]["function"].as_str().ok_or("entry function")?;
                        let c = CString::new(f).map_err(|e| e.to_string())?;
                        let mut h = ptr::null_mut();
                        checked(&format!("cuModuleGetFunction {f}"), (cuda.module_function)(&mut h, module, c.as_ptr()))?;
                        functions.insert(name.clone(), h);
                    }
                },
            }
        }
        let mut on_device = HashMap::new();
        // S01b (e72756f5): AD_DERIVED_REPORT=1 (diagnostic, never a proof leg) builds every constructible derived table from
        // this run's own mesh instead of taking the stored (x1.2562) table of the same spec SHA256; see `construct`.
        let report = std::env::var("AD_DERIVED_REPORT").as_deref() == Ok("1");
        if why.is_none() && !report {
            for name in &all {
                for b in abi["entries"][name]["args"].as_array().into_iter().flatten() {
                    let Some(h) = b["sha256"].as_str().filter(|_| b["bind"] == "derived") else { continue };
                    if on_device.contains_key(h) { continue; }
                    let Some(bytes) = derived.get(h).cloned() else { continue };
                    let mut p = 0;
                    checked("cuMemAlloc derived", (cuda.allocate)(&mut p, bytes.len().max(4)))?;
                    dev::upload(cuda, p, &bytes)?;
                    on_device.insert(h.to_owned(), (p, bytes));
                }
            }
        }
        out.insert(function.clone(), KernelDeriv { function, spec, names, meta, module, functions, tl_order, ad_order, aug, unsupported: why, derived: std::cell::RefCell::new(on_device) });
    }
    Ok(out)
}

/// One forward launch as resolved from the plan.
pub struct LaunchInfo<'a> {
    pub args: &'a [submit::Argument],
    pub grid: [u32; 3], pub block: [u32; 3], pub shared: u32,
}

#[derive(Clone, Copy)]
pub struct Shadow { pub address: u64, pub bytes: usize, pub width: usize, pub lo: usize }

/// Per-launch storage for derivative entries.
pub struct Bound {
    pub pre: HashMap<u64, u64>,
    pub shadows: BTreeMap<u64, Shadow>,
    pub tl_cells: BTreeMap<u64, (u64, usize)>,
    pub ad_cells: BTreeMap<u64, (u64, usize)>,
    pub saved: HashMap<String, (u64, usize)>,
    pub seed_copies: Vec<(String, u64, usize)>,
    pub workspace: HashMap<String, (u64, usize)>,
    pub failures: Vec<(String, u64, usize, Vec<String>)>,
    pub env: Env,
    /// Separate-buffer forms (the convention the unit proofs used): the
    /// incoming tangent / output seed is read from a snapshot of the shadow
    /// view, while the shadow itself receives the result.
    pub snap_tl: BTreeMap<String, (u64, usize)>,
    pub snap_ad: BTreeMap<String, (u64, usize)>,
    /// Views whose pre-launch cotangent is rebuilt by the entry (or is zero
    /// because the forward overwrote them): cleared after the seed snapshot.
    pub clear_ad: Vec<String>,
    /// Seed-copied arguments whose entry writes the whole new cotangent to a separate
    /// `out` buffer while consuming the live seed: written to a temporary, then copied over the view.
    pub replace_ad: BTreeMap<String, (u64, usize)>,
    /// Seed-copied in-place arguments whose `accumulate` (pre-launch) cotangent is a separate
    /// buffer: accumulated into a zeroed temporary, added to the view after the entries.
    pub post_ad: BTreeMap<String, (u64, usize)>,
    pub phase_ad: bool,
    /// U9: mirrored pointer tables (`indirect.mirror`): forward argument name -> device
    /// table whose entries point at the sweep shadows of the forward table's targets.
    pub mirror: HashMap<String, u64>,
    /// U9: arguments whose derivative storage the spec declares float64 (COMMON-AD 49.2)
    /// while the sweep keeps float32 shadows: name -> (float64 temporary, elements). The
    /// tangent is widened before the TL entries; the adjoint accumulates into a zeroed
    /// temporary that is added (rounded once) into the float32 view after the AD entries.
    pub widen: BTreeMap<String, (u64, usize)>,
    /// Physics-on window L3: the I/O each widened argument is bound with, (tangent io, adjoint io). Tangent
    /// `out`/`inout` results are rounded once into the float32 view after the TL entries (finish_tl); an adjoint
    /// `inout` seed is widened before and rounded back after the AD entries; `accumulate` is added as before.
    pub widen_io: BTreeMap<String, (String, String)>,
    /// Physics-on window L3: mirrored-table targets whose derivative the entry reads and writes as float64 (the
    /// hrrrv5 rk_cells rate table: slot 0 and every slot aliasing it, `indirect.f64_alias_of_slot0`):
    /// (float32 sweep-shadow address, float64 temporary, elements). Widened before the TL entries, zeroed before
    /// the AD entries and added back (rounded once) after them.
    pub widen_mirror: Vec<(u64, u64, usize)>,
    /// Held-kernel lane tempo_to (ad/held-tempo_to): mirrored PRIMAL pointer tables (`primal` + `indirect.mirror`):
    /// forward argument name -> device table whose entries point at pre-launch copies of the forward table's targets.
    /// U9's mirror covers tangents and adjoints only; a `primal` table argument bound plain hands the entry the live
    /// targets, which a launch like tempo_to (it reads and overwrites m, d and t through its tables) has already
    /// rewritten by the time the entries run. Each target allocation is copied once over the span its entries reach
    /// (`indirect.target_bytes` past the highest pointer, clipped to the allocation), into the launch's arena.
    pub primal_mirror: HashMap<String, u64>,
    /// packed-tape: arena bytes this launch took, by part: [pre-launch primal copies taken, pre-launch bytes bound to an
    /// earlier launch's copy instead (not taken), shadows + scalar cells + mirrored tables, transient buffers (widen,
    /// saved, snapshots, replace/post temporaries)].
    pub parts: [usize; 4],
    /// packed-tape: the transient buffers are not bound yet (`prepare_tape` with defer); the widened argument names.
    /// `bind_transient` binds them from the reverse scratch right before the launch's adjoint.
    pub deferred: Option<Vec<String>>,
    /// packed-tape: pre-launch copies bound to an earlier launch's copy: (allocation, lo, hi, that copy's address of lo).
    pub shared: Vec<(u64, usize, usize, u64)>,
    /// tape-dedupe-and-pack: the pre-launch copies this launch took: (allocation, bound lo, bound hi, copied lo, copied hi,
    /// copy address of lo). For the per-row reverse-tape receipt and the AD_TAPE_CENSUS diagnostic only.
    pub copies: Vec<(u64, usize, usize, usize, usize, u64)>,
    /// tape-dedupe-and-pack: packed reverse tapes bind each primal argument to the copy of its own span cluster:
    /// argument name -> base (copy address - cluster lo). Overrides `pre` for that argument.
    pub pre_arg: HashMap<String, u64>,
    /// The arguments of each shared cluster, by (allocation, cluster lo): rebound when the forward changed the span.
    pub cluster_args: HashMap<(u64, usize), Vec<String>>,
    /// tape-dedupe-and-pack (blocks): pre-launch spans rebuilt in the reverse scratch at walk-back: (allocation, lo, hi,
    /// runs (span offset, bytes, source address holding those exact pre-launch words), argument names bound to it, set the
    /// allocation's `pre` too). Sources are copies on this segment's tape or in the window cache, never the scratch. The
    /// runs are a device table (dev::run_rows layout: span offset, source, words) on the tape: (address, rows).
    pub recon: Vec<(u64, usize, usize, (u64, usize), Vec<String>, bool)>,
}

/// packed-tape: the pre-launch primal copies already on a step's tape, by allocation: (lo, hi, copy address), newest
/// last. THE BREAKAGE THIS FIXES: every launch copied its primal inputs again although most were unchanged since an
/// earlier launch copied them (pk120: a 187.5 GB one-pass tape for one step, 4.6 MB per cell-step, so the reverse ran in
/// 26-48 segments replaying 5.9-10x the step). A later launch whose span lies inside a cached copy binds that copy only
/// when the live bytes equal it word for word (device compare), so the bound words are the same bytes.
/// packed-tape (dead stores): `AD_TAPE_UNREAD` names a JSON file (tools/unread.py over the run's kernel set:
/// {"unread": {function: [argument, ..]}}) listing the primal arguments whose constant-bank slot no reverse entry (aug +
/// adjoint entries) references in its sm_120 SASS. Their pre-launch copies are never read by the walk-back, so a
/// deferred (reverse) tape binds the live allocation instead of copying it. THE BREAKAGE: overwritten outputs (result,
/// tendency, out_*, recover_cells' rho/theta/...) were copied onto every reverse tape although no adjoint reads their
/// pre-launch value. Unset: nothing changes. A file that cannot be read refuses the run (panic), never silently.
pub fn tape_unread(function: &str, arg: &str) -> bool { unread_lists().0.get(function).is_some_and(|v| v.iter().any(|x| x == arg)) }
/// tape-dedupe-and-pack: the same file's `unread_post` (tools/packed-tape/unread2.py, schema v2): POST-launch primal
/// arguments whose constant-bank slot no aug/adjoint entry references. A v1 file has none, so nothing changes with it.
pub fn tape_unread_post(function: &str, arg: &str) -> bool { unread_lists().1.get(function).is_some_and(|v| v.iter().any(|x| x == arg)) }
type Lists = (HashMap<String, Vec<String>>, HashMap<String, Vec<String>>);
fn unread_lists() -> &'static Lists {
    static U: std::sync::OnceLock<Lists> = std::sync::OnceLock::new();
    U.get_or_init(|| {
        let Some(p) = std::env::var_os("AD_TAPE_UNREAD") else { return (HashMap::new(), HashMap::new()) };
        let t = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("AD_TAPE_UNREAD {}: {e}", p.to_string_lossy()));
        let v: Value = serde_json::from_str(&t).unwrap_or_else(|e| panic!("AD_TAPE_UNREAD {}: {e}", p.to_string_lossy()));
        let read = |k: &str| {
            let mut m = HashMap::new();
            for (f, a) in v[k].as_object().into_iter().flatten() {
                m.insert(f.clone(), a.as_array().into_iter().flatten().filter_map(|x| x.as_str().map(str::to_owned)).collect());
            }
            m
        };
        let (m, mp) = (read("unread"), read("unread_post"));
        eprintln!("{}", serde_json::json!({"event":"tape_unread_loaded","path":p.to_string_lossy(),"functions":m.len(),"post_functions":mp.len()}));
        (m, mp)
    })
}

#[derive(Default)]
pub struct PreCache {
    pub spans: HashMap<u64, Vec<(usize, usize, u64)>>,
    pub checks: usize, pub check_bytes: usize, pub hits: usize, pub hit_bytes: usize, pub unshared: usize, pub unshared_bytes: usize,
    /// tape-dedupe-and-pack (window cache): copies that outlive the segment's tape, by allocation: (lo, hi, address of lo),
    /// in the window arena. Never cleared by `clear`; a copy here is immutable once written.
    pub win: HashMap<u64, Vec<(usize, usize, u64)>>,
    /// This segment's new pre-launch copies (allocation, lo, hi, copy address of lo): the window cache's candidates.
    pub fresh: Vec<(u64, usize, usize, u64)>,
    pub win_hits: usize, pub win_hit_bytes: usize, pub win_misses: usize,
    /// tape-dedupe-and-pack (blocks): spans rebuilt from blocks, blocks compared, equal (referenced) bytes, fresh bytes.
    pub blk_spans: usize, pub blk_checks: usize, pub blk_hit_bytes: usize, pub blk_fresh_bytes: usize,
}
/// tape-dedupe-and-pack (blocks): AD_TAPE_BLOCK_KIB (default 64; 0 = off, every unshared span is copied whole).
pub fn tape_block_bytes() -> usize {
    static B: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *B.get_or_init(|| std::env::var("AD_TAPE_BLOCK_KIB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(64) << 10)
}
impl PreCache {
    /// x20 replay-compiled: the copies lookup() would compare, without comparing them: (segment copy, window copy),
    /// each as (its lo, its address of lo). The segment copy is the newest covering one and only when word aligned,
    /// the window copy the newest covering aligned one, exactly as lookup() picks them.
    pub fn candidates(&self, alloc: u64, lo: usize, hi: usize) -> (Option<(usize, u64)>, Option<(usize, u64)>) {
        if hi <= lo || (hi - lo) % 4 != 0 { return (None, None); }
        let s = self.spans.get(&alloc).and_then(|v| v.iter().rev().find(|(clo, chi, _)| *clo <= lo && hi <= *chi))
            .filter(|(clo, _, _)| (lo - clo) % 4 == 0).map(|&(clo, _, c)| (clo, c));
        let w = self.win.get(&alloc).and_then(|v| v.iter().rev().find(|(clo, chi, _)| *clo <= lo && hi <= *chi && (lo - clo) % 4 == 0))
            .map(|&(clo, _, c)| (clo, c));
        (s, w)
    }
    /// lookup()'s decision and counters, from compares of candidates() already read back (segment / window equal).
    pub fn decide(&mut self, c: (Option<(usize, u64)>, Option<(usize, u64)>), lo: usize, hi: usize, s_eq: bool, w_eq: bool) -> Option<u64> {
        if let Some((clo, copy)) = c.0 {
            self.checks += 1; self.check_bytes += hi - lo;
            if s_eq { self.hits += 1; self.hit_bytes += hi - lo; return Some(copy.wrapping_sub(clo as u64)); }
        }
        if let Some((clo, copy)) = c.1 {
            self.checks += 1; self.check_bytes += hi - lo;
            if w_eq { self.hits += 1; self.hit_bytes += hi - lo; self.win_hits += 1; self.win_hit_bytes += hi - lo; return Some(copy.wrapping_sub(clo as u64)); }
            self.win_misses += 1;
        }
        None
    }
    pub fn clear(&mut self) { self.spans.clear(); self.fresh.clear(); }
    pub fn insert(&mut self, alloc: u64, lo: usize, hi: usize, copy: u64) { self.spans.entry(alloc).or_default().push((lo, hi, copy)); }
    /// A window-cache copy of `alloc` covering [lo, hi) already exists.
    pub fn win_covers(&self, alloc: u64, lo: usize, hi: usize) -> bool {
        self.win.get(&alloc).is_some_and(|v| v.iter().any(|(clo, chi, _)| *clo <= lo && hi <= *chi && (lo - clo) % 4 == 0))
    }
    /// tape-dedupe-and-pack (blocks): the newest copy (segment first, then window cache) of `alloc` covering [lo, hi), as
    /// the address of lo in it.
    fn block_source(&self, alloc: u64, lo: usize, hi: usize) -> Option<u64> {
        let f = |v: &Vec<(usize, usize, u64)>| v.iter().rev().find(|(clo, chi, _)| *clo <= lo && hi <= *chi && (lo - clo) % 4 == 0)
            .map(|&(clo, _, copy)| copy + (lo - clo) as u64);
        self.spans.get(&alloc).and_then(f).or_else(|| self.win.get(&alloc).and_then(f))
    }
    /// tape-dedupe-and-pack (blocks): split [lo, hi) of `alloc` into `bs`-byte blocks aligned to the allocation's offsets and
    /// compare each block's live words with the same block of the newest earlier copy that holds it (one batched device
    /// count, one readback per dev::SLOTS blocks). Runs (offset, bytes, Some(source) | None = copy it), merged where the
    /// sources are contiguous; None when fewer than a sixteenth of the span (and less than one block) is equal, so the
    /// caller copies it whole. A referenced block holds the live pre-launch words exactly (checked word for word).
    pub unsafe fn block_runs(&mut self, cuda: &Cuda, sup: &dev::Support, alloc: u64, live: u64, lo: usize, hi: usize, bs: usize)
        -> Result<Option<Vec<(usize, usize, Option<u64>)>>, String> {
        if bs == 0 || hi <= lo || (hi - lo) % 4 != 0 || lo % 4 != 0 { return Ok(None); }
        let mut pieces: Vec<(usize, usize, Option<u64>)> = Vec::new();
        let mut o = lo;
        while o < hi {
            let e = ((o / bs + 1) * bs).min(hi);
            pieces.push((o, e - o, self.block_source(alloc, o, e)));
            o = e;
        }
        if pieces.iter().all(|p| p.2.is_none()) { return Ok(None); }
        let cand: Vec<usize> = (0..pieces.len()).filter(|&k| pieces[k].2.is_some()).collect();
        let pairs: Vec<(u64, u64, usize)> = cand.iter().map(|&k| (live + pieces[k].0 as u64, pieces[k].2.unwrap(), pieces[k].1)).collect();
        let differ = sup.pairs_differ(cuda, &pairs)?;
        self.blk_checks += cand.len();
        for (&k, dif) in cand.iter().zip(differ) { if dif { pieces[k].2 = None; } }
        let equal: usize = pieces.iter().filter(|p| p.2.is_some()).map(|p| p.1).sum();
        if equal < bs.min((hi - lo) / 16).max(4096) { return Ok(None); }
        let mut runs: Vec<(usize, usize, Option<u64>)> = Vec::new();
        for (po, pl, src) in pieces {
            match (runs.last_mut(), src) {
                (Some((ro, rl, Some(rs))), Some(s)) if *ro + *rl == po && *rs + *rl as u64 == s => *rl += pl,
                (Some((ro, rl, None)), None) if *ro + *rl == po => *rl += pl,
                _ => runs.push((po, pl, src)),
            }
        }
        self.blk_spans += 1; self.blk_hit_bytes += equal; self.blk_fresh_bytes += (hi - lo) - equal;
        Ok(Some(runs))
    }
    /// The newest cached copy of `alloc` covering [lo, hi), if its bytes equal the live ones: its base (copy - its lo).
    /// The segment's copies are tried first, then the window cache's (tape-dedupe-and-pack), each word for word.
    pub unsafe fn lookup(&mut self, cuda: &Cuda, sup: &dev::Support, alloc: u64, live: u64, lo: usize, hi: usize) -> Result<Option<u64>, String> {
        if hi <= lo || (hi - lo) % 4 != 0 { return Ok(None); }
        if let Some(&(clo, _, copy)) = self.spans.get(&alloc).and_then(|v| v.iter().rev().find(|(clo, chi, _)| *clo <= lo && hi <= *chi)) {
            if (lo - clo) % 4 == 0 {
                self.checks += 1; self.check_bytes += hi - lo;
                if same_words(cuda, sup, live + lo as u64, copy + (lo - clo) as u64, hi - lo)? {
                    self.hits += 1; self.hit_bytes += hi - lo;
                    return Ok(Some(copy.wrapping_sub(clo as u64)));
                }
            }
        }
        if let Some(&(clo, _, copy)) = self.win.get(&alloc).and_then(|v| v.iter().rev().find(|(clo, chi, _)| *clo <= lo && hi <= *chi && (lo - clo) % 4 == 0)) {
            self.checks += 1; self.check_bytes += hi - lo;
            if same_words(cuda, sup, live + lo as u64, copy + (lo - clo) as u64, hi - lo)? {
                self.hits += 1; self.hit_bytes += hi - lo; self.win_hits += 1; self.win_hit_bytes += hi - lo;
                return Ok(Some(copy.wrapping_sub(clo as u64)));
            }
            self.win_misses += 1;
        }
        Ok(None)
    }
}
/// packed-tape: true when [a, a + bytes) and [b, b + bytes) hold the same 32-bit words (one device count, one readback).
pub unsafe fn same_words(cuda: &Cuda, sup: &dev::Support, a: u64, b: u64, bytes: usize) -> Result<bool, String> {
    if bytes == 0 { return Ok(true); }
    sup.reset_words(cuda)?;
    sup.words_into(cuda, 0, a, b, bytes)?;
    let (_, w) = sup.collect(cuda, 0, 1)?;
    Ok(w[0][0] == 0)
}

/// Integer value of a captured by-value scalar (`value_hex`, little-endian).
fn scalar_value(m: &Value) -> Option<f64> {
    let b = crate::exec::decode_hex(m["value_hex"].as_str()?).ok()?;
    match (m["dtype"].as_str()?, b.len()) {
        ("<i4", 4) => Some(i32::from_le_bytes(b[..4].try_into().ok()?) as f64),
        ("<u4", 4) => Some(u32::from_le_bytes(b[..4].try_into().ok()?) as f64),
        ("<i8", 8) => Some(i64::from_le_bytes(b[..8].try_into().ok()?) as f64),
        _ => None,
    }
}

/// How a launch's entries treat each argument's derivative storage.
pub struct Roles { pub tl_snapshot: Vec<String>, pub ad_snapshot: Vec<String>, pub ad_clear: Vec<String>, pub ad_replace: Vec<String>, pub ad_post: Vec<String> }

impl KernelDeriv {
    fn position(&self, name: &str) -> Result<usize, String> {
        self.names.iter().position(|n| n == name).ok_or_else(|| format!("{}: no launch argument {name}", self.function))
    }
    pub fn env(&self, l: &LaunchInfo) -> Result<Env, String> {
        let mut env = Env::new();
        for (i, m) in self.meta.iter().enumerate() {
            if m["kind"] != "scalar" { continue; }
            let Some(submit::Argument::Bytes(b)) = l.args.get(i) else { continue };
            let v = match (m["dtype"].as_str().unwrap_or(""), b.len()) {
                ("<i4", 4) => i32::from_le_bytes(b[..4].try_into().unwrap()) as f64,
                ("<u4", 4) => u32::from_le_bytes(b[..4].try_into().unwrap()) as f64,
                ("<i8", 8) => i64::from_le_bytes(b[..8].try_into().unwrap()) as f64,
                ("<u8", 8) => u64::from_le_bytes(b[..8].try_into().unwrap()) as f64,
                ("<f4", 4) => f32::from_le_bytes(b[..4].try_into().unwrap()) as f64,
                ("<f8", 8) => f64::from_le_bytes(b[..8].try_into().unwrap()),
                _ => continue,
            };
            env.insert(self.names[i].clone(), v);
        }
        for axis in 0..3 {
            env.insert(format!("grid[{axis}]"), l.grid[axis] as f64);
            // The 11-02 hrrrv5 glue specs (RRFS copy plan) name the launch geometry grid_x / grid_y / block_x ...
            let c = ["x", "y", "z"][axis];
            env.insert(format!("grid_{c}"), l.grid[axis] as f64);
            env.insert(format!("block_{c}"), l.block[axis] as f64);
            env.insert(format!("block[{axis}]"), l.block[axis] as f64);
        }
        // Shape symbols that are not launch scalars (e.g. an allocation extent such as
        // ninterfaces) take the captured extent at their position.
        for decl in self.spec["args"].as_array().into_iter().flatten() {
            let (Some(shape), Some(cap)) = (decl["shape"].as_str(), decl["capture_reference_shape"].as_array()) else { continue };
            let dims: Vec<&str> = shape.trim().trim_start_matches('[').trim_end_matches(']').split(',').map(str::trim).collect();
            for (d, c) in dims.iter().zip(cap) {
                if d.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '_') && d.chars().next().is_some_and(|ch| ch.is_ascii_alphabetic()) && !env.contains_key(*d) {
                    if let Some(v) = c.as_f64() { env.insert((*d).to_owned(), v); }
                }
            }
        }
        if let Some(derived) = self.spec["launch_abi"]["derived_scalars"].as_object() {
            for (name, e) in derived {
                let text = e.as_str().map(|s| s.strip_prefix("forward.geometry.").unwrap_or(s).to_owned()).unwrap_or(e.to_string());
                let v = expr::eval(&text, &env)?;
                env.insert(name.clone(), v);
            }
        }
        Ok(env)
    }
    fn device_target(&self, l: &LaunchInfo, name: &str) -> Result<Option<(u64, usize)>, String> {
        let i = self.position(name)?;
        Ok(match &l.args[i] {
            submit::Argument::Device(s) => Some((s.allocation, s.offset)),
            submit::Argument::Patched { pointers, .. } if pointers.len() == 1 && pointers[0].byte_offset == 0 =>
                Some((pointers[0].target.allocation, pointers[0].target.offset)),
            _ => None,
        })
    }
    /// Element count of a forward argument at this launch.
    /// A declared shape written symbolically is evaluated with this launch's
    /// scalars. A concrete (captured) shape is re-expressed through the capture's
    /// own scalar values: a dimension equal to a captured scalar (or that scalar
    /// plus one) follows that scalar at this launch; any other stays constant.
    /// Element count of argument `name` at this launch. A fixed numeric extent in the spec (`"shape": [104303]`) is
    /// the spec's declared extent and is never undercut by the kernel-set row's capture shape. step6-radiation
    /// (2026-11-07), THE BREAKAGE THIS PREVENTS: the held rsw_setcoef_b / rsw_sfluxzen_b rows carry `tab` (the packed
    /// RRTMG-SW table, 104,303 words) with an empty row shape, so its pre-launch primal copy held 1 word plus the
    /// 256 KiB margin (65,536 words) and the entries read preflog/tref (words 94,184..94,301) from whatever the scratch
    /// arena held: the fac00..fac11 tangents differed between two processes by 34x in norm (s6r-ta1/ta2 at step 1
    /// command 639) and the per-launch dots of both kernels changed run to run (s6r-rd11/rd12).
    fn elements(&self, name: &str, env: &Env) -> Result<usize, String> {
        let n = self.elements_from_shapes(name, env)?;
        let decl = self.spec["args"].as_array().and_then(|a| a.iter().find(|d| d["name"] == name));
        let fixed = decl.and_then(|d| d["shape"].as_array()).filter(|a| !a.is_empty() && a.iter().all(|x| x.is_u64()))
            .map(|a| a.iter().map(|x| x.as_u64().unwrap() as usize).product::<usize>());
        Ok(match fixed { Some(m) => n.max(m), None => n })
    }
    fn elements_from_shapes(&self, name: &str, env: &Env) -> Result<usize, String> {
        let decl = self.spec["args"].as_array().and_then(|a| a.iter().find(|d| d["name"] == name));
        let symbolic = |v: &Value| match v {
            Value::String(s) => s.chars().any(|c| c.is_ascii_alphabetic()),
            Value::Array(a) => a.iter().any(|d| d.is_string()),
            _ => false,
        };
        if let Some(d) = decl {
            if symbolic(&d["shape"]) { return expr::shape_elements(&d["shape"], env); }
            // Physics-on window L3: a fixed extent written as a string ("[17]", the hrrrv5 rk_cells rate pointer table)
            // is that literal extent. It fell through to the kernel-set row's shape, which add_glue.py leaves empty
            // for string shapes, so the table counted 1 element: the mirror held 1 of 17 pointers and the entries read
            // past it (CUDA 700 at step 1 with persistent shadows, rkc1 and fix1 on 2026-11-04/05).
            if let Value::String(t) = &d["shape"] {
                let inner = t.trim().trim_start_matches('[').trim_end_matches(']');
                if !inner.trim().is_empty() && inner.chars().all(|c| c.is_ascii_digit() || c == ',' || c == ' ' || c == '*') {
                    return expr::shape_elements(&d["shape"], env);
                }
            }
        }
        let i = self.position(name)?;
        let dims: Vec<u64> = self.meta[i]["shape"].as_array().ok_or("capture shape")?.iter().map(|d| d.as_u64().unwrap_or(0)).collect();
        // pk120-recording: an argument the spec sizes `{"shape_like_forward_arg": ..}` whose kernel-set row carries no
        // capture shape (an empty one) counted 1 element, so its pre-launch primal copy held 1 word plus the 256 KiB
        // margin. THE BREAKAGE THIS PREVENTS: rrfs_pbl_radiation_sum (out = lw + sw over n = nc * nz words) on x1.40962
        // (n = 2,416,758) read lw/sw past the copy from whatever the tape arena held: "step 2 rrfs_pbl_radiation_sum
        // tangent refused: aug: driver_rrfs.py:1315:nonfinite" in pk-T3, pk-T3a, pk-T4 and c-T2, and at reverse step 1
        // in dg-W1, while pk-T3t, pk-T3c and rep-e passed the same step (arena contents). When the launch has exactly
        // one integer scalar, that scalar's launch value is the element count (the flat elementwise glue kernels).
        if dims.is_empty() && d_is_like_forward(decl) {
            let ints: Vec<usize> = self.meta.iter().enumerate().filter(|(_, m)| m["kind"] == "scalar"
                && matches!(m["dtype"].as_str(), Some("<i4") | Some("<i8") | Some("<u4") | Some("<u8"))).map(|(j, _)| j).collect();
            if ints.len() == 1 { if let Some(now) = env.get(&self.names[ints[0]]) { return Ok(*now as usize); } }
        }
        // Physics-on window (ad/phys-window): a flat element count passed as one scalar (transport_interpolate_target's
        // nvalues = nlev * ncells, captured 140,910 on the 55-level dry route) is the whole extent; no shape dimension
        // equals a scalar there, so on the 59-level hrrrv5 plan the span stayed 140,910 words while the kernel ran
        // 151,158 (ZERO-FAIL: 20,497 nonzero adjoint words from a zero seed at step 1). When the captured product
        // equals exactly one captured integer scalar, that scalar's launch value is the element count.
        let product: u64 = dims.iter().product();
        if dims.len() >= 2 && product > 2 {
            let hits: Vec<usize> = self.meta.iter().enumerate().filter(|(_, m)| m["kind"] == "scalar"
                && scalar_value(m) == Some(product as f64)).map(|(j, _)| j).collect();
            if hits.len() == 1 { if let Some(now) = env.get(&self.names[hits[0]]) { return Ok(*now as usize); } }
        }
        let mut n = 1usize;
        let rank = dims.len();
        for (pos, d) in dims.into_iter().enumerate() {
            let mut value = d as f64;
            // Physics-on window (ad/phys-window): the leading species axis of a rank >= 3 field (ntracers, nmass,
            // nspecies) was captured as 1 on the dry route and is 1-2 there, so the small-dimension guard below kept
            // it constant. On hrrrv5 (TEMPO tracers) the launch's own count is larger, the shadow span covered one
            // species only and the entries read past it: NONFINITE / ZERO-FAIL on transport_edge_values,
            // transport_vertical_flux, fct_*, moist_cell/edge_coefficients at step 1 of the 18-step plan.
            // Rule: a leading dimension of 1 or 2 of a rank >= 3 shape follows the launch scalar whose captured
            // value equals it, only when exactly one scalar matches (unchanged on the dry route: same value).
            if d <= 2 && pos == 0 && rank >= 3 && d >= 1 {
                let hits: Vec<usize> = self.meta.iter().enumerate().filter(|(_, m)| m["kind"] == "scalar"
                    && scalar_value(m) == Some(d as f64)).map(|(j, _)| j).collect();
                if hits.len() == 1 { if let Some(now) = env.get(&self.names[hits[0]]) { value = *now; } }
            }
            'found: for plus in [0.0, 1.0] {
                if d <= 2 { break; }
                for (j, m) in self.meta.iter().enumerate() {
                    if m["kind"] != "scalar" { continue; }
                    let Some(captured) = scalar_value(m) else { continue };
                    if captured + plus == d as f64 {
                        if let Some(now) = env.get(&self.names[j]) { value = now + plus; break 'found; }
                    }
                }
            }
            n *= value as usize;
        }
        Ok(n)
    }
    fn entries(&self) -> Vec<String> {
        let mut v: Vec<String> = Vec::new();
        if self.aug { v.push("aug".into()); }
        for e in self.tl_order.iter().chain(self.ad_order.iter()) { if !v.contains(e) { v.push(e.clone()); } }
        v
    }
    fn bindings_all(&self) -> impl Iterator<Item = &Value> {
        self.spec["launch_abi"]["entries"].as_object().into_iter().flat_map(|m| m.values())
            .flat_map(|e| e["args"].as_array().into_iter().flatten())
    }
    fn bindings(&self, entry: &str) -> impl Iterator<Item = &Value> {
        self.spec["launch_abi"]["entries"][entry]["args"].as_array().into_iter().flatten()
    }

    /// Classify derivative arguments by the I/O roles their entries declare.
    /// TL: an argument bound both `in` and `out` reads its incoming tangent from
    /// a snapshot (the unit proofs give the two roles separate buffers).
    /// AD: an argument whose seed is a read-only `in`, or whose seed (`in`/`inout`)
    /// comes with a separate `accumulate`/`out` cotangent, and that has no saved
    /// seed copy of its own, reads the seed from a snapshot; its shadow view is
    /// cleared first, because the entry rebuilds the pre-launch cotangent (an
    /// overwritten output has none). `inout`-only and saved-seed forms consume in place.
    pub fn roles(&self) -> Roles {
        let mut tl: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut ad: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut seed_copied = BTreeSet::new();
        for (phase, entries) in [("tl", &self.tl_order), ("ad", &self.ad_order)] {
            for e in entries {
                for b in self.bindings(e) {
                    let kind = b["bind"].as_str().unwrap_or("");
                    if kind == "saved" {
                        let producer = b["producer"].as_str().unwrap_or("");
                        if producer.contains("seed") || producer.starts_with("scheduler-copy") {
                            if let Some(a) = b["arg"].as_str().or(b["from"]["arg"].as_str())
                                .or(producer.strip_prefix("scheduler-copy:adjoint:")).or(b["elems"]["shape_like_forward_arg"].as_str()) {
                                seed_copied.insert(a.to_owned());
                            }
                        }
                    }
                    if !matches!(kind, "tangent" | "adjoint") || b["scalar_arg"].is_string() { continue; }
                    // A mirrored pointer table (prep_mass qout on hrrrv5) has no shadow of its own: the entries reach
                    // the targets' shadows through the mirror, so it takes no snapshot/clear role ("shadow missing").
                    if b["indirect"].is_object() { continue; }
                    let (Some(arg), Some(io)) = (b["arg"].as_str(), b["io"].as_str()) else { continue };
                    let map = if phase == "tl" { &mut tl } else { &mut ad };
                    map.entry(arg.to_owned()).or_default().insert(io.to_owned());
                }
            }
        }
        let tl_snapshot = tl.iter().filter(|(_, r)| r.contains("in") && r.contains("out")).map(|(a, _)| a.clone()).collect();
        let mut ad_snapshot = Vec::new();
        let mut ad_replace = Vec::new();
        let mut ad_post = Vec::new();
        for (arg, r) in &ad {
            if seed_copied.contains(arg) {
                if r.contains("out") && (r.contains("inout") || r.contains("in")) { ad_replace.push(arg.clone()); }
                else if r.contains("accumulate") && (r.contains("inout") || r.contains("in")) { ad_post.push(arg.clone()); }
                continue;
            }
            // Integration (ad/phys-window): an adjoint `in` binding of a pure forward input (spec role "in") is a
            // read-only pointer, not the seed of an overwritten output. prep_mass and prep_interfaces route every
            // signature pointer of their reduction phase as `in`; taking that for a seed snapshotted and cleared each
            // input's incoming cotangent (integ1pairs: every input diagonal pair rhs = -x.y with lhs 0; per-launch
            // NS-ONLY rel 4.2e-4; composed 1-step window rel 4.5e-3). Only these two kernels have the form.
            if self.spec["args"].as_array().is_some_and(|a| a.iter().any(|x| x["name"] == arg.as_str() && x["role"] == "in")) { continue; }
            let seed_read_only = r.contains("in");
            let separate = (r.contains("in") || r.contains("inout")) && (r.contains("accumulate") || r.contains("out"));
            if seed_read_only || separate { ad_snapshot.push(arg.clone()); }
        }
        Roles { tl_snapshot, ad_clear: ad_snapshot.clone(), ad_snapshot, ad_replace, ad_post }
    }
    /// step6-radiation diagnostic (AD_SUBNORMAL): per device argument, the f32 shadow view's (words, nonzero, subnormal,
    /// min nonzero |v|, max |v|). Reads only; never changes a word.
    pub unsafe fn shadow_census(&self, cuda: &Cuda, b: &Bound, l: &LaunchInfo) -> Vec<Value> {
        let mut out = Vec::new();
        for name in &self.names {
            let Ok((a, n)) = self.view(b, l, name) else { continue };
            let Some(s) = b.shadows.values().find(|s| a >= s.address && a < s.address + s.bytes as u64) else { continue };
            if s.width != 4 || n == 0 { continue; }
            let Ok(v) = dev::download(cuda, a, n) else { continue };
            let (mut nz, mut sub, mut lo, mut hi, mut at) = (0u64, 0u64, f64::MAX, 0f64, 0usize);
            for (i, c) in v.chunks_exact(4).enumerate() {
                let x = f32::from_le_bytes(c.try_into().unwrap());
                if x != 0.0 { nz += 1; let y = (x as f64).abs(); lo = lo.min(y); if y > hi { hi = y; at = i; } if !x.is_normal() && x.is_finite() { sub += 1; } }
            }
            // integ7 (diagnostic only): the word index of the largest |t| is appended as a 7th field.
            out.push(serde_json::json!([name, n / 4, nz, sub, if nz > 0 { lo } else { 0.0 }, hi, at]));
        }
        out
    }
    /// integ7 diagnostic (AD_CENSUS_DUMP=dir, default off): write every device argument's float32 primal view
    /// (<tag>.<name>.primal.f32, from the live allocation) and tangent view (<tag>.<name>.tl.f32) at this launch.
    pub unsafe fn census_dump(&self, cuda: &Cuda, b: &Bound, l: &LaunchInfo, addr: &dyn Fn(u64) -> Option<u64>, dir: &str, tag: &str) {
        for name in &self.names {
            let Ok(Some((alloc, off))) = self.device_target(l, name) else { continue };
            if let (Some(a), Ok(n)) = (addr(alloc), self.elements(name, &b.env)) {
                if let Ok(v) = dev::download(cuda, a + off as u64, n * 4) { let _ = std::fs::write(format!("{dir}/{tag}.{name}.primal.f32"), v); }
            }
            if let Ok((a, n)) = self.view(b, l, name) {
                let w = b.shadows.values().find(|s| a >= s.address && a < s.address + s.bytes as u64).map(|s| s.width).unwrap_or(4);
                if w == 4 { if let Ok(v) = dev::download(cuda, a, n) { let _ = std::fs::write(format!("{dir}/{tag}.{name}.tl.f32"), v); } }
            }
        }
    }
    /// step6-radiation bisection diagnostic (AD_ZERO_DERIV, default off): zero this launch's written derivative views
    /// (tangent out/inout in the TL, adjoint inout seeds in the AD), so its outputs act as constants both ways.
    pub unsafe fn zero_written(&self, cuda: &Cuda, b: &Bound, l: &LaunchInfo, adjoint: bool) -> Result<usize, String> {
        let (en, bind) = if adjoint { ("ad", "adjoint") } else { ("tl", "tangent") };
        let mut n = 0;
        for x in self.spec["launch_abi"]["entries"][en]["args"].as_array().into_iter().flatten() {
            if x["bind"] != bind { continue; }
            let io = x["io"].as_str().unwrap_or("");
            if !(io == "out" || io == "inout") { continue; }
            let Some(name) = x["arg"].as_str() else { continue };
            if let Ok((a, bytes)) = self.view(b, l, name) { dev::zero(cuda, a, bytes)?; n += 1; }
        }
        Ok(n)
    }
    fn view(&self, b: &Bound, l: &LaunchInfo, name: &str) -> Result<(u64, usize), String> {
        let (alloc, offset) = self.device_target(l, name)?.ok_or_else(|| format!("{name} is not a device argument"))?;
        let s = b.shadows.get(&alloc).ok_or("shadow missing")?;
        let start = offset.checked_sub(s.lo).ok_or("view below shadow span")? / 4 * s.width;
        let bytes = self.elements(name, &b.env)? * s.width;
        if start + bytes > s.bytes { return Err(format!("{}: view of {name} exceeds its allocation", self.function)); }
        Ok((s.address + start as u64, bytes))
    }
    /// Before the tangent entries: snapshot incoming tangents of separate in/out forms.
    /// readbacks-device: the float32 -> float64 widen and the float64 -> float32 rounding of widened arguments run on
    /// the device (Support::widen_f32 / narrow_f64 / narrow_add_f64), stream-ordered like the entries, with the same
    /// conversions the host loops made (exact widen, round-to-nearest-even narrowing, add in float64 then round once).
    /// Before, each widened argument was a synchronous pageable download, a host loop and an upload, per launch.
    pub unsafe fn begin_tl(&self, cuda: &Cuda, sup: &dev::Support, b: &mut Bound, l: &LaunchInfo) -> Result<(), String> {
        b.phase_ad = false;
        for (sh, tmp, n) in b.widen_mirror.clone() { sup.widen_f32(cuda, sh, tmp, n)?; }
        for (name, (tmp, n)) in b.widen.clone() {
            let (src, _) = self.view(b, l, &name)?;
            sup.widen_f32(cuda, src, tmp, n)?;
        }
        for (name, (dst, bytes)) in b.snap_tl.clone() {
            let (src, n) = self.view(b, l, &name)?;
            dev::copy(cuda, dst, src, n.min(bytes))?;
        }
        Ok(())
    }
    /// After the tangent entries: a widened tangent `out`/`inout` is rounded once into its float32 view.
    pub unsafe fn finish_tl(&self, cuda: &Cuda, sup: &dev::Support, b: &mut Bound, l: &LaunchInfo) -> Result<(), String> {
        for (name, (tmp, n)) in b.widen.clone() {
            if !b.widen_io.get(&name).is_some_and(|io| io.0 == "out" || io.0 == "inout") { continue; }
            let (dst, _) = self.view(b, l, &name)?;
            sup.narrow_f64(cuda, tmp, dst, n)?;
        }
        Ok(())
    }
    /// Before the reverse entries: snapshot seeds, then clear the views the entry rebuilds.
    pub unsafe fn begin_ad(&self, cuda: &Cuda, sup: &dev::Support, b: &mut Bound, l: &LaunchInfo) -> Result<(), String> {
        b.phase_ad = true;
        for (_, tmp, n) in b.widen_mirror.clone() { dev::zero(cuda, tmp, n * 8)?; }
        for (name, (tmp, n)) in b.widen.clone() {
            if b.widen_io.get(&name).is_some_and(|io| io.1 == "inout") {
                let (src, _) = self.view(b, l, &name)?;
                sup.widen_f32(cuda, src, tmp, n)?;
            } else { dev::zero(cuda, tmp, n * 8)?; }
        }
        for (name, (dst, bytes)) in b.snap_ad.clone() {
            let (src, n) = self.view(b, l, &name)?;
            dev::copy(cuda, dst, src, n.min(bytes))?;
        }
        for name in b.clear_ad.clone() {
            let (a, n) = self.view(b, l, &name)?;
            dev::zero(cuda, a, n)?;
        }
        for (a, n) in b.replace_ad.values().chain(b.post_ad.values()) { dev::zero(cuda, *a, *n)?; }
        Ok(())
    }
    /// After the reverse entries: a separately written whole cotangent replaces the view.
    pub unsafe fn finish_ad(&self, cuda: &Cuda, b: &mut Bound, l: &LaunchInfo, support: Option<&crate::dev::Support>) -> Result<(), String> {
        if !b.widen.is_empty() || !b.widen_mirror.is_empty() {
            let sup = support.ok_or_else(|| format!("{}: widened arguments need the support kernels", self.function))?;
            for (name, (tmp, n)) in b.widen.clone() {
                let (dst, _) = self.view(b, l, &name)?;
                let replace = b.widen_io.get(&name).is_some_and(|io| io.1 == "inout");
                sup.narrow_add_f64(cuda, dst, tmp, n, replace)?;
            }
            for (sh, tmp, n) in b.widen_mirror.clone() { sup.narrow_add_f64(cuda, sh, tmp, n, false)?; }
        }
        for (name, (src, bytes)) in b.replace_ad.clone() {
            let (dst, n) = self.view(b, l, &name)?;
            dev::copy(cuda, dst, src, n.min(bytes))?;
        }
        if !b.post_ad.is_empty() {
            let support = support.ok_or("post-launch accumulation needs the support kernels")?;
            for (name, (src, bytes)) in b.post_ad.clone() {
                let (dst, n) = self.view(b, l, &name)?;
                let w = b.shadows.values().next().map(|s| s.width).unwrap_or(4);
                support.move_add(cuda, dst, src, n.min(bytes), w)?;
            }
        }
        Ok(())
    }

    /// Allocate everything a launch's derivative entries bind: shadows, scalar
    /// cells, saved buffers. `width` is the derivative storage width (4 or 8).
    pub unsafe fn prepare(&self, cuda: &Cuda, arena: &mut Arena, l: &LaunchInfo, exec_alloc: &HashMap<u64, submit::Allocation>,
                          width: usize) -> Result<Bound, String> { self.prepare_with(cuda, arena, l, exec_alloc, width, None) }
    /// As `prepare`, but bind derivative arguments to persistent whole-allocation
    /// shadows (the composed tangent and reverse sweeps) instead of per-launch spans.
    pub unsafe fn prepare_with(&self, cuda: &Cuda, arena: &mut Arena, l: &LaunchInfo, exec_alloc: &HashMap<u64, submit::Allocation>,
                          width: usize, persistent: Option<&BTreeMap<u64, Shadow>>) -> Result<Bound, String> {
        self.prepare_tape(cuda, arena, l, exec_alloc, width, persistent, false, None)
    }
    /// packed-tape: `prepare_with`, plus: with `defer` the transient buffers (widen, saved, snapshots, replace/post
    /// temporaries: nothing the forward state decides, all rebuilt or zeroed before the launch's derivative entries run)
    /// are not taken from `arena`; `bind_transient` takes them later. With `cache` a pre-launch primal copy is bound to
    /// an earlier launch's copy when the live bytes equal it word for word (PreCache::lookup), and every new copy is
    /// cached. Without either, the arena takes exactly what `prepare_with` took.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn prepare_tape(&self, cuda: &Cuda, arena: &mut Arena, l: &LaunchInfo, exec_alloc: &HashMap<u64, submit::Allocation>,
                          width: usize, persistent: Option<&BTreeMap<u64, Shadow>>, defer: bool,
                          mut cache: Option<(&mut PreCache, &dev::Support)>) -> Result<Bound, String> {

        let mut env = self.env(l)?;
        // pk120 (x1.40962): a mesh extent the launch does not pass as a scalar (theta_filter_lap4_v841_f32 has no nedges
        // argument) is read from a named argument that is one whole mesh array in its own allocation:
        // `launch_abi.mesh_extents = {"nedges": "dc_edge"}` gives nedges = that allocation's 4-byte words. THE BREAKAGE
        // THIS PREVENTS: the row's edge arrays were declared "[7680]" (x1.2562's nEdges, from its capture), so on
        // x1.40962 (122,880 edges) the pre-launch primal copies of cells_on_edge/dc_edge/dv_edge/del4 held 7,680 words
        // plus the 256 KiB margin and the tangent read arena bytes past them: jr-120-s1 (2026-11-07) refused "step 1
        // theta_filter_lap4_v841_f32 tangent refused: cuda_horizontal_v841.py:335, 336, 340" (nonfinite dc_edge,
        // delsq_theta at garbage cells_on_edge indices, and the output). An argument at a nonzero offset is refused:
        // then its allocation is not that one array.
        // A mesh extent overrides the captured-extent fallback of env() (a shape symbol that is not a launch scalar took
        // x1.2562's value from capture_reference_shape: w_filter_lap4_v841_f32 nedges = 7,680, per-launch dot REFUSED on
        // x1.40962 at w_filter_lap4_v841_f32.ad.cu:33 field 5), never a launch scalar of the same name.
        if let Some(m) = self.spec["launch_abi"]["mesh_extents"].as_object() {
            for (sym, arg) in m {
                if self.meta.iter().zip(&self.names).any(|(x, n)| x["kind"] == "scalar" && n == sym) { continue; }
                let name = arg.as_str().ok_or_else(|| format!("{}: mesh_extents {sym} names no argument", self.function))?;
                let (alloc, offset) = self.device_target(l, name)?
                    .ok_or_else(|| format!("{}: mesh extent {sym}: {name} is not a device argument", self.function))?;
                if offset != 0 {
                    return Err(format!("{}: mesh extent {sym}: {name} starts at byte {offset} of allocation {alloc}, not a whole mesh array", self.function));
                }
                let a = exec_alloc.get(&alloc).ok_or_else(|| format!("{}: mesh extent {sym}: unknown allocation {alloc}", self.function))?;
                env.insert(sym.clone(), (a.bytes / 4) as f64);
            }
        }
        let mut b = Bound { pre: HashMap::new(), shadows: BTreeMap::new(), tl_cells: BTreeMap::new(), ad_cells: BTreeMap::new(),
                            saved: HashMap::new(), seed_copies: Vec::new(), workspace: HashMap::new(), failures: Vec::new(), env,
                            snap_tl: BTreeMap::new(), snap_ad: BTreeMap::new(), clear_ad: Vec::new(), replace_ad: BTreeMap::new(), post_ad: BTreeMap::new(), phase_ad: false, mirror: HashMap::new(), widen: BTreeMap::new(), widen_io: BTreeMap::new(), widen_mirror: Vec::new(), primal_mirror: HashMap::new(),
                            parts: [0; 4], deferred: None, shared: Vec::new(), copies: Vec::new(), pre_arg: HashMap::new(), cluster_args: HashMap::new(), recon: Vec::new() };
        let mut widths: BTreeMap<u64, usize> = BTreeMap::new();
        let mut extents: BTreeMap<u64, (usize, usize)> = BTreeMap::new();
        let mut pre_spans: BTreeMap<u64, (usize, usize)> = BTreeMap::new();
        let mut arg_spans: BTreeMap<u64, BTreeMap<String, (usize, usize)>> = BTreeMap::new();
        let mut indirect: Vec<String> = Vec::new();
        let mut primal_indirect: Vec<String> = Vec::new();
        let mut widen: Vec<String> = Vec::new();
        let mut widen_io: BTreeMap<String, (String, String)> = BTreeMap::new();
        let mut live_only: Vec<u64> = Vec::new();
        for entry in self.entries() {
            // packed-tape (dead stores): a deferred (reverse) tape serves only the walk-back, which runs `aug` and the
            // adjoint entries. A pre-launch primal span that only a tangent entry reads is never read from the tape.
            // THE BREAKAGE: fct_edge_residual / fct_finish / fct_horizontal_low_order / transport_edge_values_mono /
            // transport_standard_finish / transport_target_density copied tangent-only primal args (horizontal_residual,
            // upwind_flux, output, mass, scale_in/out, edge_values, target_density) onto every reverse tape. A span the
            // walk-back does need and that is missing fails loudly ("pre-copy missing"), never silently.
            let rev = entry == "aug" || self.ad_order.contains(&entry);
            for bind in self.bindings(&entry) {
                let kind = bind["bind"].as_str().unwrap_or("");
                match kind {
                    "primal" if defer && !rev => (),
                    "primal" if defer && bind["post"] != true && !bind["indirect"].is_object()
                        && bind["arg"].as_str().is_some_and(|n| tape_unread(&self.function, n)) => {
                        if let Some((alloc, _)) = self.device_target(l, bind["arg"].as_str().unwrap_or(""))? { live_only.push(alloc); }
                    },
                    "primal" if bind["indirect"].is_object() && bind["post"] != true => {
                        if let Some(name) = bind["arg"].as_str() { if !primal_indirect.contains(&name.to_owned()) { primal_indirect.push(name.to_owned()); } }
                    },
                    "primal" if bind["post"] != true => {
                        if let Some(name) = bind["arg"].as_str() {
                            if let Some((alloc, offset)) = self.device_target(l, name)? {
                                b.pre.insert(alloc, 0);
                                let span = match self.elements(name, &b.env) { Ok(n) => (offset, offset + n * 4), Err(_) => (0, usize::MAX) };
                                let e = pre_spans.entry(alloc).or_insert(span);
                                e.0 = e.0.min(span.0); e.1 = e.1.max(span.1);
                                let e = arg_spans.entry(alloc).or_default().entry(name.to_owned()).or_insert(span);
                                e.0 = e.0.min(span.0); e.1 = e.1.max(span.1);
                            }
                        }
                    },
                    "tangent" | "adjoint" if bind["indirect"].is_object() => {
                        if let Some(name) = bind["arg"].as_str() { if !indirect.contains(&name.to_owned()) { indirect.push(name.to_owned()); } }
                    },
                    "tangent" | "adjoint" => {
                        let field = bind["field_id"].as_u64().unwrap_or(u64::MAX);
                        let mut w = if bind["shadow_dtype"] == "f64" && std::env::var_os("AD_E2E_IGNORE_SHADOW_DTYPE").is_none() { 8 } else { width };
                        let target = match (bind["scalar_arg"].as_str(), bind["arg"].as_str()) {
                            (Some(_), _) => None,
                            (None, Some(name)) => self.device_target(l, name)?,
                            _ => return Err(format!("{} {kind} binding without arg", self.function)),
                        };
                        if let (Some(p), Some((alloc, _)), Some(name)) = (persistent, target, bind["arg"].as_str()) {
                            if w == 8 && p.get(&alloc).is_some_and(|s| s.width == 4) {
                                let io = bind["io"].as_str().unwrap_or("");
                                // Physics-on window L3: mpas_transport_source_baseline binds its float64 tangent `out` and
                                // adjoint `inout` (baseline) over the float32 sweep shadow; refused before (9 ERROR rows in
                                // stage A). Those forms now go through the float64 temporary as well (see widen_io).
                                if !((kind == "tangent" && matches!(io, "in" | "out" | "inout")) || (kind == "adjoint" && matches!(io, "accumulate" | "inout"))) {
                                    return Err(format!("{}: float64 {kind} {io} binding of {name} over a float32 sweep shadow", self.function));
                                }
                                if !widen.contains(&name.to_owned()) { widen.push(name.to_owned()); }
                                let e = widen_io.entry(name.to_owned()).or_insert((String::new(), String::new()));
                                if kind == "tangent" { if e.0.is_empty() || io != "in" { e.0 = io.to_owned(); } }
                                else if e.1.is_empty() || io == "inout" { e.1 = io.to_owned(); }
                                w = 4;
                            }
                        }
                        match target {
                            None => { b.tl_cells.insert(field, (0, w)); b.ad_cells.insert(field, (0, w)); },
                            Some((alloc, offset)) => {
                                if let Some(prev) = widths.insert(alloc, w) {
                                    if prev != w { return Err(format!("{}: allocation {alloc} bound with float32 and float64 shadows", self.function)); }
                                }
                                // Shadow only the bound span of the allocation (with a margin).
                                let name = bind["arg"].as_str().unwrap_or("");
                                let span = match self.elements(name, &b.env) {
                                    Ok(n) => (offset, offset + n * 4),
                                    Err(_) => (0, usize::MAX),
                                };
                                let e = extents.entry(alloc).or_insert(span);
                                e.0 = e.0.min(span.0); e.1 = e.1.max(span.1);
                            },
                        }
                    },
                    _ => (),
                }
            }
        }
        // Pre-launch primal: copy only the bound span (with a margin); the
        // binding address is (copy - span start) + offset, so offsets are unchanged.
        let u_pre = arena.used;
        let mut pre_ids: Vec<u64> = b.pre.keys().copied().collect();
        pre_ids.sort_unstable();
        // x20 replay-compiled: the first-stage cache check of every pre-launch cluster of this launch (the copies
        // PreCache::lookup would compare) is queued into its own word slot and read back with ONE collect; the loop below
        // then decides each cluster in the old order with the old counters (PreCache::decide). A miss's new copy cannot
        // become another cluster's candidate: clusters of one allocation are disjoint and lookup needs a covering copy.
        // THE BREAKAGE THIS FIXES: one count launch and one blocking readback (a full drain) per cluster before every
        // taped launch. AD_CHECK_BATCH=0 keeps one readback per check.
        let mut pre_eq: HashMap<(u64, usize), ((Option<(usize, u64)>, Option<(usize, u64)>), bool, bool)> = HashMap::new();
        if cache.is_some() && std::env::var("AD_TAPE_CLUSTER").as_deref() != Ok("0") && std::env::var("AD_CHECK_BATCH").as_deref() != Ok("0") {
            const MARGIN: usize = 256 << 10;
            let (c, sup) = cache.as_mut().unwrap();
            let mut q: Vec<(u64, usize, usize, u64, (Option<(usize, u64)>, Option<(usize, u64)>))> = Vec::new();
            for &alloc in &pre_ids {
                let a = exec_alloc.get(&alloc).ok_or("unknown allocation")?;
                let mut cl: Vec<(usize, usize)> = Vec::new();
                let mut spans: Vec<(usize, usize, String)> = arg_spans.get(&alloc).into_iter().flatten()
                    .map(|(n, &(l0, h0))| (l0, h0.min(a.bytes), n.clone())).collect();
                spans.sort();
                for (l0, h0, _) in spans {
                    let (lo, hi) = ((l0.saturating_sub(MARGIN) / 256) * 256, h0.saturating_add(MARGIN).min(a.bytes));
                    match cl.last_mut() { Some(x) if lo <= x.1 => { x.1 = x.1.max(hi); }, _ => cl.push((lo, hi)) }
                }
                for (lo, hi) in cl { q.push((alloc, lo, hi, a.address, c.candidates(alloc, lo, hi))); }
            }
            let nslot = 2 * q.len();
            if nslot > 2 && nslot <= dev::SLOTS {
                sup.reset_words(cuda)?;
                for (j, (_, lo, hi, live, (s, w))) in q.iter().enumerate() {
                    if let Some((clo, copy)) = s { sup.words_into(cuda, 2 * j, live + *lo as u64, copy + (lo - clo) as u64, hi - lo)?; }
                    if let Some((clo, copy)) = w { sup.words_into(cuda, 2 * j + 1, live + *lo as u64, copy + (lo - clo) as u64, hi - lo)?; }
                }
                let (_, wv) = sup.collect(cuda, 0, nslot)?;
                for (j, (alloc, lo, _, _, cands)) in q.into_iter().enumerate() { pre_eq.insert((alloc, lo), (cands, wv[2 * j][0] == 0, wv[2 * j + 1][0] == 0)); }
            }
        }
        for alloc in pre_ids {
            let a = exec_alloc.get(&alloc).ok_or("unknown allocation")?;
            const MARGIN: usize = 256 << 10;
            // tape-dedupe-and-pack: the packed path copies per argument span cluster (see Bound::pre_arg). The words each
            // argument reads are the same bytes: its cluster copy holds its whole span with the same margins, checked or
            // copied from the live pre-launch words. AD_TAPE_CLUSTER=0 keeps the one-span-per-allocation layout.
            if cache.is_some() && std::env::var("AD_TAPE_CLUSTER").as_deref() != Ok("0") {
                let (c, sup) = cache.as_mut().unwrap();
                let mut cl: Vec<(usize, usize, usize, usize, Vec<String>)> = Vec::new();
                let mut spans: Vec<(usize, usize, String)> = arg_spans.get(&alloc).into_iter().flatten()
                    .map(|(n, &(l0, h0))| (l0, h0.min(a.bytes), n.clone())).collect();
                spans.sort();
                for (l0, h0, n) in spans {
                    let (lo, hi) = ((l0.saturating_sub(MARGIN) / 256) * 256, h0.saturating_add(MARGIN).min(a.bytes));
                    match cl.last_mut() {
                        Some(x) if lo <= x.1 => { x.1 = x.1.max(hi); x.2 = x.2.min(l0); x.3 = x.3.max(h0); x.4.push(n); },
                        _ => cl.push((lo, hi, l0, h0, vec![n])),
                    }
                }
                let mut first = true;
                for (lo, hi, l0, h0, names) in cl {
                    let found = match pre_eq.get(&(alloc, lo)) {
                        Some(&(cands, se, we)) => c.decide(cands, lo, hi, se, we),
                        None => c.lookup(cuda, sup, alloc, a.address, lo, hi)?,
                    };
                    let base = if let Some(base) = found {
                        b.shared.push((alloc, lo, hi, base.wrapping_add(lo as u64)));
                        b.cluster_args.insert((alloc, lo), names.clone());
                        b.parts[1] += hi - lo;
                        base
                    } else if let Some(runs) = c.block_runs(cuda, sup, alloc, a.address, lo, hi, tape_block_bytes())? {
                        let sup = &**sup;
                        // tape-dedupe-and-pack (blocks): only the blocks that differ from every earlier copy are taped; the
                        // span is rebuilt in the reverse scratch at walk-back (bind_recon). THE BREAKAGE: a span that
                        // changed in a few blocks since an earlier launch copied it was copied whole again (pkg18m2:
                        // 0.46 GB of the 1.39 GB of pre-launch copies per step were blocks equal to an earlier copy).
                        let mut rr: Vec<(u64, u64, usize)> = Vec::with_capacity(runs.len());
                        let mut fresh: Vec<(u64, u64, usize)> = Vec::new();
                        for (ro, rl, src) in runs {
                            match src {
                                Some(s) => { rr.push(((ro - lo) as u64, s, rl)); b.parts[1] += rl; },
                                None => {
                                    let cp = arena.take(rl)?;
                                    fresh.push((cp, a.address + ro as u64, rl));
                                    b.copies.push((alloc, ro, ro + rl, ro, ro + rl, cp));
                                    c.insert(alloc, ro, ro + rl, cp); c.fresh.push((alloc, ro, ro + rl, cp));
                                    rr.push(((ro - lo) as u64, cp, rl));
                                },
                            }
                        }
                        // The fresh runs: one launch from the live words (stream-ordered before the forward launch).
                        let ft = dev::run_rows(&fresh);
                        if !ft.is_empty() {
                            let tab = arena.take(ft.len() * 8)?;
                            dev::upload(cuda, tab, std::slice::from_raw_parts(ft.as_ptr().cast::<u8>(), ft.len() * 8))?;
                            sup.runs_copy(cuda, 0, tab, ft.len() / 3)?;
                        }
                        // The rebuild table stays on the tape for the walk-back.
                        let rt = dev::run_rows(&rr);
                        let tab = arena.take(rt.len().max(1) * 8)?;
                        dev::upload(cuda, tab, std::slice::from_raw_parts(rt.as_ptr().cast::<u8>(), rt.len() * 8))?;
                        b.recon.push((alloc, lo, hi, (tab, rt.len() / 3), names.clone(), first));
                        0
                    } else {
                        let copy = arena.take(hi - lo)?;
                        dev::copy(cuda, copy, a.address + lo as u64, hi - lo)?;
                        b.copies.push((alloc, l0, h0, lo, hi, copy));
                        c.insert(alloc, lo, hi, copy); c.fresh.push((alloc, lo, hi, copy));
                        copy.wrapping_sub(lo as u64)
                    };
                    if first { b.pre.insert(alloc, base); first = false; }
                    if base != 0 { for n in names { b.pre_arg.insert(n, base); } } // 0: rebuilt at walk-back (bind_recon)
                }
                continue;
            }
            let (lo0, hi0) = pre_spans.get(&alloc).copied().unwrap_or((0, usize::MAX));
            let lo = (lo0.saturating_sub(MARGIN) / 256) * 256;
            let hi = hi0.saturating_add(MARGIN).min(a.bytes);
            if let Some((c, sup)) = cache.as_mut() {
                if let Some(base) = c.lookup(cuda, sup, alloc, a.address, lo, hi)? {
                    b.pre.insert(alloc, base);
                    b.shared.push((alloc, lo, hi, base.wrapping_add(lo as u64)));
                    b.parts[1] += hi - lo;
                    continue;
                }
            }
            let copy = arena.take(hi - lo)?;
            dev::copy(cuda, copy, a.address + lo as u64, hi - lo)?;
            b.pre.insert(alloc, copy.wrapping_sub(lo as u64));
            b.copies.push((alloc, lo0.min(a.bytes), hi0.min(a.bytes), lo, hi, copy));
            if let Some((c, _)) = cache.as_mut() { c.insert(alloc, lo, hi, copy); c.fresh.push((alloc, lo, hi, copy)); }
        }
        // packed-tape (dead stores): an allocation that only unread arguments bind gets the live address (never
        // dereferenced by any reverse entry, see tape_unread); one that a read argument also binds keeps its copy.
        for alloc in live_only {
            if b.pre.contains_key(&alloc) { continue; }
            let a = exec_alloc.get(&alloc).ok_or("unknown allocation")?;
            b.pre.insert(alloc, a.address);
        }
        b.parts[0] = arena.used - u_pre;
        let u_meta = arena.used;
        // (indirect.primal_pre, seaice_finish: served by bind_mirror_primal, the one primal pointer-table rule.)
        for (alloc, w) in widths {
            let a = exec_alloc.get(&alloc).ok_or("unknown allocation")?;
            if let Some(p) = persistent {
                let s = *p.get(&alloc).ok_or("persistent shadow missing")?;
                if s.width != w { return Err(format!("{}: allocation {alloc} needs a {w}-byte shadow, sweep uses {}", self.function, s.width)); }
                b.shadows.insert(alloc, s);
                continue;
            }
            const MARGIN: usize = 1 << 20;
            let (lo, hi) = extents.get(&alloc).copied().unwrap_or((0, usize::MAX));
            let lo = (lo.saturating_sub(MARGIN) / 256) * 256;
            let hi = hi.saturating_add(MARGIN).min(a.bytes);
            let bytes = (hi - lo) / 4 * w;
            b.shadows.insert(alloc, Shadow { address: arena.take(bytes)?, bytes, width: w, lo });
        }
        for cells in [&mut b.tl_cells, &mut b.ad_cells] {
            for (_, cell) in cells.iter_mut() { cell.0 = arena.take(8)?; dev::zero(cuda, cell.0, 8)?; }
        }
        for name in &indirect {
            let p = persistent.ok_or_else(|| format!("{}: mirrored pointer table {name} needs the sweep shadows", self.function))?;
            let (alloc, offset) = self.device_target(l, name)?.ok_or("pointer table is not a device argument")?;
            // Table length from the captured forward argument (the spec shape describes the targets).
            // Physics-on window: the table length follows the spec shape when it is symbolic ([nspecies] on hrrrv5),
            // else the captured forward argument (unchanged on the dry route).
            // Physics-on window L3: a descriptor table in P8's row form (`indirect.rows`: the RRFS copy plan, 128-byte
            // rows whose only pointers are dst and src) mirrors those words only. The word-per-pointer mirror read every
            // descriptor word as an address (strides 0x1, extents 0xa02) and read 8x the table's bytes past its end
            // (CUDA status 1 on the download: 13 ERROR rows at step 1).
            if let Some(rows) = self.bindings_all().find_map(|x| (x["arg"] == name.as_str() && x["indirect"].is_object()).then(|| x["indirect"]["rows"].clone())).filter(|v| v.is_object()) {
                let stride = rows["stride_bytes"].as_u64().ok_or("rows stride_bytes")? as usize;
                let ptrs: Vec<usize> = rows["pointer_bytes"].as_array().ok_or("rows pointer_bytes")?.iter().filter_map(|v| v.as_u64().map(|x| x as usize)).collect();
                let count_at = rows["count_bytes"].as_u64().map(|x| x as usize);
                let nbytes = self.elements(name, &b.env)?;
                if stride == 0 || nbytes % stride != 0 { return Err(format!("{}: {name} table of {nbytes} bytes is not whole {stride}-byte rows", self.function)); }
                let a = &exec_alloc[&alloc];
                if offset + nbytes > a.bytes { return Err(format!("{}: {name} table exceeds its allocation", self.function)); }
                let mut table = dev::download(cuda, a.address + offset as u64, nbytes)?;
                let mut sum_count = 0f64;
                for row in table.chunks_exact_mut(stride) {
                    for &at in &ptrs {
                        let ptr = u64::from_le_bytes(row[at..at + 8].try_into().unwrap());
                        if ptr == 0 { continue; }
                        let t = exec_alloc.values().find(|a| ptr >= a.address && ptr < a.address + a.bytes as u64)
                            .ok_or_else(|| format!("{}: {name} row pointer {ptr:#x} is not in a plan allocation", self.function))?;
                        let s = p.get(&t.id).ok_or("row target has no shadow")?;
                        b.shadows.entry(t.id).or_insert(*s);
                        let m = s.address + (((ptr - t.address) as usize - s.lo) / 4 * s.width) as u64;
                        row[at..at + 8].copy_from_slice(&m.to_le_bytes());
                    }
                    if let Some(c) = count_at { sum_count += i64::from_le_bytes(row[c..c + 8].try_into().unwrap()) as f64; }
                }
                b.env.insert("sum_op_count".into(), sum_count);
                let t = arena.take(table.len())?;
                dev::upload(cuda, t, &table)?;
                b.mirror.insert(name.clone(), t);
                continue;
            }
            let n: usize = self.elements(name, &b.env)?;
            let table = dev::download(cuda, exec_alloc[&alloc].address + offset as u64, n * 8)?;
            let mut mirrored = Vec::with_capacity(n * 8);
            // Physics-on window L3: the rk_cells derivative reads/writes slot 0 of its rate table (and every slot
            // holding the same pointer) as float64; over the float32 sweep shadow those 8-byte stores ran past each
            // element (NONFINITE and 348,129 differing reverse words at step 1, once the table length was right).
            let f64_elems = self.bindings_all().find_map(|x| (x["arg"] == name.as_str() && x["indirect"].is_object()).then(|| x["indirect"]["f64_alias_of_slot0"].clone()))
                .filter(|v| !v.is_null()).map(|v| expr::eval_usize(&v, &b.env)).transpose()?;
            let slot0 = table.get(..8).map(|w| u64::from_le_bytes(w.try_into().unwrap())).unwrap_or(0);
            let mut slot0_tmp = 0u64;
            for w in table.chunks_exact(8) {
                let ptr = u64::from_le_bytes(w.try_into().unwrap());
                let m = if ptr == 0 { 0 } else if let (Some(ne), true) = (f64_elems, ptr == slot0) {
                    if slot0_tmp == 0 {
                        let a = exec_alloc.values().find(|a| ptr >= a.address && ptr < a.address + a.bytes as u64)
                            .ok_or_else(|| format!("{}: {name} slot 0 {ptr:#x} is not in a plan allocation", self.function))?;
                        let s = *p.get(&a.id).ok_or("pointer table target has no shadow")?;
                        if s.width != 4 { return Err(format!("{}: {name} slot 0 shadow is not float32", self.function)); }
                        b.shadows.entry(a.id).or_insert(s);
                        let sh = s.address + (((ptr - a.address) as usize - s.lo) / 4 * s.width) as u64;
                        if (ptr - a.address) as usize + ne * 4 > a.bytes { return Err(format!("{}: {name} slot 0 extent exceeds its allocation", self.function)); }
                        slot0_tmp = arena.take(ne * 8)?;
                        dev::zero(cuda, slot0_tmp, ne * 8)?;
                        b.widen_mirror.push((sh, slot0_tmp, ne));
                    }
                    slot0_tmp
                } else {
                    let a = exec_alloc.values().find(|a| ptr >= a.address && ptr < a.address + a.bytes as u64)
                        .ok_or_else(|| format!("{}: {name} entry {ptr:#x} is not in a plan allocation", self.function))?;
                    let s = p.get(&a.id).ok_or("pointer table target has no shadow")?;
                    // Per-launch dot over persistent shadows: a table target is part of this launch's derivative memory.
                    b.shadows.entry(a.id).or_insert(*s);
                    s.address + (((ptr - a.address) as usize - s.lo) / 4 * s.width) as u64
                };
                mirrored.extend_from_slice(&m.to_le_bytes());
            }
            let t = arena.take(mirrored.len())?;
            dev::upload(cuda, t, &mirrored)?;
            b.mirror.insert(name.clone(), t);
        }
        for name in &primal_indirect {
            if bind_mirror_primal(self, cuda, arena, l, exec_alloc, &mut b, name)? { continue; }
        }
        b.widen_io = widen_io;
        b.parts[2] = arena.used - u_meta;
        if defer { b.deferred = Some(widen); }
        else { let u = arena.used; self.transient(cuda, arena, &mut b, l, &widen)?; b.parts[3] = arena.used - u; }
        Ok(b)
    }
    /// packed-tape: the post-launch primal spans of a launch: allocation -> [lo, hi), the union over every `post`
    /// binding of its AD entries and aug of the bound span with the pre-copy margin (the whole allocation when an
    /// extent is unknown). THE BREAKAGE THIS FIXES: the post copy took the whole allocation (pkg18m2: 2.9 GB of a
    /// 9.3 GB step, recover_edges_f32 57 MB per launch out of a 36.5 MB pooled allocation).
    pub fn post_spans(&self, l: &LaunchInfo, env: &Env, live: &HashMap<u64, submit::Allocation>) -> Result<BTreeMap<u64, (usize, usize)>, String> {
        let mut m: BTreeMap<u64, (usize, usize)> = BTreeMap::new();
        let aug = "aug".to_owned();
        // tape-dedupe-and-pack: unread post arguments (tape_unread_post) leave the post span union, so an allocation whose
        // post bindings are all unread gets no post copy and one with read ones copies only their spans. An allocation
        // that the walk-back also reads through the live map (derived inputs, patched by-value pointers, binding kinds
        // this rule does not know) keeps the full union, as before. THE BREAKAGE: 37 of kset-twh's 53 post arguments
        // (pv_edge, ke_edge/rho_edge, prep_mass/prep_interfaces outputs, reconstruct_winds' u_p/v_p/x/y/z) were copied
        // after every launch although no entry reads them.
        let mut keep: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let resolve = |n: &str, keep: &mut std::collections::HashSet<u64>, patched_only: bool| {
            if let Some(k) = self.names.iter().position(|x| x == n) {
                match l.args.get(k) {
                    Some(submit::Argument::Device(s)) if !patched_only => { keep.insert(s.allocation); },
                    Some(submit::Argument::Patched { pointers, .. }) => { for p in pointers { keep.insert(p.target.allocation); } },
                    _ => (),
                }
            }
        };
        for e in self.ad_order.iter().chain(self.aug.then_some(&aug)) {
            for bind in self.spec["launch_abi"]["entries"][e]["args"].as_array().into_iter().flatten() {
                let kind = bind["bind"].as_str().unwrap_or("");
                for n in bind["from"].as_array().into_iter().flatten().filter_map(|x| x.as_str()) { resolve(n, &mut keep, false); }
                if let Some(n) = bind["from"]["arg"].as_str() { resolve(n, &mut keep, false); }
                match kind {
                    "primal" | "tangent" | "adjoint" | "saved" | "failure" | "branch_record" | "workspace" | "derived" => (),
                    "scalar" => { for k in ["arg", "scalar_arg"] { if let Some(n) = bind[k].as_str() { resolve(n, &mut keep, true); } } },
                    _ => { for k in ["arg", "scalar_arg"] { if let Some(n) = bind[k].as_str() { resolve(n, &mut keep, false); } } },
                }
            }
        }
        for e in self.ad_order.iter().chain(self.aug.then_some(&aug)) {
            for bind in self.spec["launch_abi"]["entries"][e]["args"].as_array().into_iter().flatten() {
                if bind["bind"] != "primal" || bind["post"] != true { continue; }
                let Some(name) = bind["arg"].as_str() else { continue };
                if tape_unread_post(&self.function, name) && !bind["indirect"].is_object() {
                    if let Some(k) = self.names.iter().position(|n| n == name) {
                        if let Some(submit::Argument::Device(s)) = l.args.get(k) { if !keep.contains(&s.allocation) { continue; } }
                    }
                }
                let Some(k) = self.names.iter().position(|n| n == name) else { continue };
                let Some(submit::Argument::Device(s)) = l.args.get(k) else { continue };
                let Some(a) = live.get(&s.allocation) else { continue };
                const MARGIN: usize = 256 << 10;
                let (lo, hi) = match self.elements(name, env) { Ok(n) => (s.offset, s.offset + n * 4), Err(_) => (0, a.bytes) };
                let lo = (lo.saturating_sub(MARGIN) / 256) * 256;
                let hi = hi.saturating_add(MARGIN).min(a.bytes);
                let x = m.entry(s.allocation).or_insert((lo, hi));
                x.0 = x.0.min(lo); x.1 = x.1.max(hi);
            }
        }
        Ok(m)
    }
    /// packed-tape: bind the transient buffers a deferred `prepare_tape` left out, from `arena` (the reverse scratch,
    /// reset before each launch's adjoint). Same buffers, same zeroing, other addresses.
    /// tape-dedupe-and-pack (blocks): rebuild each `Bound::recon` span in `arena` (the reverse scratch, after
    /// bind_transient) from its runs and bind its arguments there. The rebuilt words are the live pre-launch words.
    pub unsafe fn bind_recon(&self, cuda: &Cuda, sup: &dev::Support, arena: &mut Arena, b: &mut Bound) -> Result<usize, String> {
        let mut bytes = 0;
        for (al, lo, hi, (tab, rows), names, set_pre) in std::mem::take(&mut b.recon) {
            let s = arena.take(hi - lo)?;
            sup.runs_copy(cuda, s, tab, rows)?;
            let base = s.wrapping_sub(lo as u64);
            if set_pre { b.pre.insert(al, base); }
            for n in names { b.pre_arg.insert(n, base); }
            bytes += hi - lo;
        }
        Ok(bytes)
    }
    pub unsafe fn bind_transient(&self, cuda: &Cuda, arena: &mut Arena, b: &mut Bound, l: &LaunchInfo) -> Result<(), String> {
        if let Some(widen) = b.deferred.take() { let u = arena.used; self.transient(cuda, arena, b, l, &widen)?; b.parts[3] = arena.used - u; }
        Ok(())
    }
    /// The transient part of `prepare_with`, in its order: widen temporaries, saved buffers (aug tapes, seed snapshots),
    /// TL/AD snapshots, the AD clear list, replace and post temporaries.
    unsafe fn transient(&self, cuda: &Cuda, arena: &mut Arena, b: &mut Bound, l: &LaunchInfo, widen: &[String]) -> Result<(), String> {
        for name in widen {
            let (_, bytes) = self.view(b, l, name)?;
            let n = bytes / 4;
            let t = arena.take(n * 8)?;
            dev::zero(cuda, t, n * 8)?;
            b.widen.insert(name.clone(), (t, n));
        }
        // Saved buffers: aug tapes and seed snapshots.
        for entry in self.entries() {
            for bind in self.bindings(&entry) {
                if bind["bind"] != "saved" { continue; }
                let name = bind["name"].as_str().ok_or("saved name")?.to_owned();
                if b.saved.contains_key(&name) { continue; }
                let elems = if let Some(arg) = bind["elems"]["shape_like_forward_arg"].as_str().or(bind["shape_like_forward_arg"].as_str()) {
                    self.elements(arg, &b.env)?
                } else if !bind["elems"].is_null() {
                    expr::eval_usize(&bind["elems"], &b.env)?
                } else if !bind["bytes"].is_null() {
                    expr::eval_usize(&bind["bytes"], &b.env)? / 4
                } else { return Err(format!("{} saved {name} without size", self.function)); };
                let producer = bind["producer"].as_str().unwrap_or("");
                let seed_copy = producer.contains("seed") || producer.starts_with("scheduler-copy");
                let source = bind["arg"].as_str().or(bind["from"]["arg"].as_str())
                    .or(producer.strip_prefix("scheduler-copy:adjoint:"))
                    .or(bind["elems"]["shape_like_forward_arg"].as_str()).map(str::to_owned);
                let mut w = match bind["dtype"].as_str() { Some("<f8" | "f64") => 8, _ => 4 };
                if seed_copy {
                    let src = source.clone().ok_or_else(|| format!("{} seed copy {name} without source", self.function))?;
                    let (alloc, _) = self.device_target(l, &src)?.ok_or("seed copy source is not device")?;
                    w = b.shadows.get(&alloc).map(|s| s.width).ok_or_else(|| format!("seed copy source {src} has no shadow"))?;
                    b.seed_copies.push((src, 0, elems * w));
                }
                let address = arena.take(elems * w)?;
                dev::zero(cuda, address, elems * w)?;
                if seed_copy { b.seed_copies.last_mut().unwrap().1 = address; }
                b.saved.insert(name, (address, elems * w));
            }
        }
        let roles = self.roles();
        for (names, target) in [(&roles.tl_snapshot, 0), (&roles.ad_snapshot, 1)] {
            for name in names {
                if self.device_target(l, name)?.is_none() { continue; }
                let (_, bytes) = self.view(b, l, name)?;
                let a = arena.take(bytes)?;
                if target == 0 { b.snap_tl.insert(name.clone(), (a, bytes)); } else { b.snap_ad.insert(name.clone(), (a, bytes)); }
            }
        }
        b.clear_ad = roles.ad_clear.into_iter().filter(|n| b.snap_ad.contains_key(n)).collect();
        for name in &roles.ad_replace {
            if self.device_target(l, name)?.is_none() { continue; }
            let (_, bytes) = self.view(b, l, name)?;
            b.replace_ad.insert(name.clone(), (arena.take(bytes)?, bytes));
        }
        for name in &roles.ad_post {
            if self.device_target(l, name)?.is_none() { continue; }
            let (_, bytes) = self.view(b, l, name)?;
            b.post_ad.insert(name.clone(), (arena.take(bytes)?, bytes));
        }
        Ok(())
    }

    fn shadow_address(&self, b: &Bound, l: &LaunchInfo, name: &str) -> Result<u64, String> {
        let (alloc, offset) = self.device_target(l, name)?.ok_or_else(|| format!("{name} is not a device argument"))?;
        let s = b.shadows.get(&alloc).ok_or("shadow missing")?;
        Ok(s.address + (offset.checked_sub(s.lo).ok_or("argument below shadow span")? / 4 * s.width) as u64)
    }

    /// Copy incoming output-cotangent seeds into their snapshots (before the reverse entries).
    pub unsafe fn snapshot_seeds(&self, cuda: &Cuda, b: &Bound, l: &LaunchInfo) -> Result<(), String> {
        for (src, dst, bytes) in &b.seed_copies {
            dev::copy(cuda, *dst, self.shadow_address(b, l, src)?, *bytes)?;
        }
        Ok(())
    }
    pub unsafe fn zero_workspaces(&self, cuda: &Cuda, b: &mut Bound) -> Result<(), String> {
        for (_, (a, n)) in &b.workspace { dev::zero(cuda, *a, *n)?; }
        Ok(())
    }

    /// Launch one entry with every argument bound; returns failure-slot sources that fired.
    pub unsafe fn launch(&self, cuda: &Cuda, arena: &mut Arena, b: &mut Bound, l: &LaunchInfo,
                         live: &HashMap<u64, submit::Allocation>, entry: &str) -> Result<Vec<String>, String> {
        let e = &self.spec["launch_abi"]["entries"][entry];
        let mut store: Vec<Vec<u8>> = Vec::new();
        let mut failure_slots = Vec::new();
        for bind in e["args"].as_array().ok_or("entry args")? {
            let kind = bind["bind"].as_str().unwrap_or("");
            let value: Vec<u8> = match kind {
                "scalar" => {
                    if let Some(name) = bind["arg"].as_str().or(bind["scalar_arg"].as_str()) {
                        match &l.args[self.position(name)?] {
                            submit::Argument::Bytes(v) => v.clone(),
                            // g3-radland: a CuPy CArray by value (data pointer patched to the live forward address), so
                            // the entry reads the view's size, shape and strides.
                            submit::Argument::Patched { bytes, pointers } => {
                                let mut v = bytes.clone();
                                for p in pointers {
                                    let a = live.get(&p.target.allocation).ok_or_else(|| format!("{} scalar {name}: allocation {} not live", self.function, p.target.allocation))?;
                                    if p.byte_offset + 8 > v.len() { return Err(format!("{} scalar {name}: pointer patch past the struct", self.function)); }
                                    v[p.byte_offset..p.byte_offset + 8].copy_from_slice(&(a.address + p.target.offset as u64).to_le_bytes());
                                }
                                // "carray": true re-lays a CArray<T, nd> {data, size, shape[nd], strides[nd]} (nd = 0..4) as the
                                // fixed 88-byte {data, size, nd, shape[4], strides[4]}, so one entry handles every rank.
                                if bind["carray"] == true {
                                    if v.len() < 16 || (v.len() - 16) % 16 != 0 || (v.len() - 16) / 16 > 4 {
                                        return Err(format!("{} scalar {name}: {} bytes is not a CArray of rank <= 4", self.function, v.len()));
                                    }
                                    let nd = (v.len() - 16) / 16;
                                    let w = |k: usize| i64::from_le_bytes(v[8 * k..8 * k + 8].try_into().unwrap());
                                    let mut f: Vec<u8> = Vec::with_capacity(88);
                                    f.extend_from_slice(&v[0..16]);
                                    f.extend_from_slice(&(nd as i64).to_le_bytes());
                                    for d in 0..4 { f.extend_from_slice(&(if d < nd { w(2 + d) } else { 1 }).to_le_bytes()); }
                                    for d in 0..4 { f.extend_from_slice(&(if d < nd { w(2 + nd + d) } else { 0 }).to_le_bytes()); }
                                    v = f;
                                }
                                v
                            },
                            other => return Err(format!("{} scalar {name} is {other:?}", self.function)),
                        }
                    } else {
                        // value_from (the 11-02 hrrrv5 glue specs, e.g. nlev*ncells) is the same expression form as derived_from.
                        let v = match bind["derived_from"].as_str().or(bind["value_from"].as_str()) { Some(e) => expr::eval(e, &b.env)?, None => bind["value"].as_f64().ok_or("scalar value")? };
                        match bind["dtype"].as_str().unwrap_or("<i4") {
                            "<i4" | "i32" => (v as i32).to_le_bytes().to_vec(),
                            "<i8" | "i64" => (v as i64).to_le_bytes().to_vec(),
                            "<f4" | "f32" => (v as f32).to_le_bytes().to_vec(),
                            "<f8" | "f64" => v.to_le_bytes().to_vec(),
                            d => return Err(format!("scalar literal dtype {d}")),
                        }
                    }
                },
                "primal" if bind["indirect"].is_object() && bind["post"] != true => {
                    let name = bind["arg"].as_str().ok_or("primal arg")?;
                    b.primal_mirror.get(name).ok_or_else(|| format!("primal mirror of {name} missing"))?.to_le_bytes().to_vec()
                },
                "primal" => {
                    let name = bind["arg"].as_str().ok_or("primal arg")?;
                    match &l.args[self.position(name)?] {
                        submit::Argument::Bytes(v) => v.clone(),
                        _ => {
                            let (alloc, offset) = self.device_target(l, name)?.ok_or("primal target")?;
                            let base = if bind["post"] == true { live[&alloc].address }
                                       else if let Some(x) = b.pre_arg.get(name) { *x }
                                       else { *b.pre.get(&alloc).ok_or("pre-copy missing")? };
                            (base + offset as u64).to_le_bytes().to_vec()
                        },
                    }
                },
                "tangent" | "adjoint" if bind["arg"].as_str().is_some_and(|n| b.widen.contains_key(n)) => {
                    b.widen[bind["arg"].as_str().unwrap()].0.to_le_bytes().to_vec()
                },
                "tangent" | "adjoint" if bind["indirect"].is_object() => {
                    let name = bind["arg"].as_str().ok_or("indirect arg")?;
                    b.mirror.get(name).ok_or_else(|| format!("mirror of {name} missing"))?.to_le_bytes().to_vec()
                },
                "tangent" | "adjoint" => {
                    let field = bind["field_id"].as_u64().unwrap_or(u64::MAX);
                    let cells = if kind == "tangent" { &b.tl_cells } else { &b.ad_cells };
                    if bind["scalar_arg"].is_string() {
                        cells.get(&field).ok_or("scalar cell missing")?.0.to_le_bytes().to_vec()
                    } else {
                        let name = bind["arg"].as_str().ok_or("derivative arg")?;
                        let io = bind["io"].as_str().unwrap_or("");
                        let snapshot = match (kind, io) {
                            ("tangent", "in") if !b.phase_ad => b.snap_tl.get(name),
                            ("adjoint", "in" | "inout") if b.phase_ad => b.snap_ad.get(name),
                            ("adjoint", "out") if b.phase_ad => b.replace_ad.get(name),
                            ("adjoint", "accumulate") if b.phase_ad => b.post_ad.get(name),
                            _ => None,
                        };
                        if self.device_target(l, name)?.is_none() { cells.get(&field).ok_or("scalar cell missing")?.0.to_le_bytes().to_vec() }
                        else if let Some((a, _)) = snapshot { a.to_le_bytes().to_vec() }
                        else { self.shadow_address(b, l, name)?.to_le_bytes().to_vec() }
                    }
                },
                "workspace" => {
                    let name = bind["name"].as_str().ok_or("workspace name")?.to_owned();
                    let bytes = expr::eval_usize(&bind["bytes"], &b.env)?;
                    let a = match b.workspace.get(&name) {
                        Some((a, n)) if *n >= bytes => *a,
                        _ => { let a = arena.take(bytes)?; dev::zero(cuda, a, bytes)?; b.workspace.insert(name, (a, bytes)); a },
                    };
                    a.to_le_bytes().to_vec()
                },
                "failure" => {
                    let slots = if bind["slots"].is_string() { expr::eval_usize(&bind["slots"], &b.env)? } else { bind["slots"].as_u64().unwrap_or(1) as usize }
                        .max(bind["minimum_slots"].as_u64().unwrap_or(0) as usize);
                    let per = bind["bytes_per_slot"].as_u64().unwrap_or(4) as usize;
                    let bytes = slots * per.max(4) * 2;
                    let a = arena.take(bytes)?;
                    dev::zero(cuda, a, bytes)?;
                    let sources: Vec<String> = bind["source_ids"].as_array().map(|s| s.iter().map(|x| x.as_str().unwrap_or("").to_owned()).collect()).unwrap_or_default();
                    failure_slots.push((a, bytes, sources));
                    a.to_le_bytes().to_vec()
                },
                "branch_record" => {
                    let slots = bind["slots"].as_u64().unwrap_or(1) as usize;
                    let shaped = bind["shape"].as_array().map(|s| s.iter().map(|x| x.as_u64().unwrap_or(1) as usize).product()).unwrap_or(0);
                    let bytes = (slots * 2).max(shaped) * 4 * 2;
                    let a = arena.take(bytes)?;
                    dev::zero(cuda, a, bytes)?;
                    a.to_le_bytes().to_vec()
                },
                "derived" => {
                    let key = match bind["sha256"].as_str() {
                        Some(h) => h.to_owned(),
                        None if bind["recipe"].as_str().is_some_and(|r| r.contains("glue.interface_shapes.rs")) =>
                            format!("interface_shapes:{}:{}x{}", bind["name"].as_str().unwrap_or(""), b.env.get("nlev").copied().unwrap_or(0.0), b.env.get("ncells").copied().unwrap_or(0.0)),
                        None => return Err("derived sha256".into()),
                    };
                    let h = key.as_str();
                    if !self.derived.borrow().contains_key(h) {
                        let bytes = self.construct(cuda, bind, l, &b.env, live)?;
                        let mut p = 0;
                        checked("cuMemAlloc derived", (cuda.allocate)(&mut p, bytes.len().max(4)))?;
                        dev::upload(cuda, p, &bytes)?;
                        self.derived.borrow_mut().insert(h.to_owned(), (p, bytes));
                    }
                    let store = self.derived.borrow();
                    let (p, bytes) = store.get(h).ok_or_else(|| format!("derived {h} not loaded"))?;
                    // Scalars travel by value: declared so, or (undeclared) a single 4- or 8-byte word.
                    let by_value = bind["by_value"] == true || bind["as"] == "scalar" || bind["kind"] == "scalar"
                        || (bind["by_value"].is_null() && bytes.len() <= 8);
                    if by_value { bytes.clone() } else { p.to_le_bytes().to_vec() }
                },
                // Physics-on window L3: the RRFS copy plan's per-operation fill-scalar derivative pointers. Fills are
                // treated as fixed (null), which is P8's copy-rows rule (fill rows: tangent 0, nothing propagates);
                // active fill provenance is not traced (named in PHYS-STATUS).
                "fill_tangent_pointers" | "fill_adjoint_pointers" => 0u64.to_le_bytes().to_vec(),
                "saved" => {
                    let name = bind["name"].as_str().ok_or("saved name")?;
                    b.saved.get(name).ok_or_else(|| format!("saved {name} missing"))?.0.to_le_bytes().to_vec()
                },
                other => return Err(format!("{}: unsupported binding {other}", self.function)),
            };
            store.push(value);
        }
        let geometry = &e["geometry"];
        let (grid, block, shared) = if geometry.is_object() {
            let axis = |key: &str, i: usize| -> Result<u32, String> {
                Ok(expr::eval_usize(&geometry[key][i], &b.env)? as u32)
            };
            ([axis("grid", 0)?, axis("grid", 1)?, axis("grid", 2)?], [axis("block", 0)?, axis("block", 1)?, axis("block", 2)?],
             if geometry["shared_bytes"].is_null() { 0 } else { expr::eval_usize(&geometry["shared_bytes"], &b.env)? as u32 })
        } else { (l.grid, l.block, l.shared) };
        let mut params: Vec<*mut c_void> = store.iter_mut().map(|v| v.as_mut_ptr().cast()).collect();
        let f = *self.functions.get(entry).ok_or("entry function not loaded")?;
        checked(&format!("{} {entry} launch", self.function), (cuda.launch)(f, grid[0], grid[1], grid[2], block[0], block[1], block[2],
            shared, ptr::null_mut(), params.as_mut_ptr(), ptr::null_mut()))?;
        // No host wait here: failure slots are read once, after the launch's entries.
        for (a, bytes, sources) in failure_slots { b.failures.push((entry.to_owned(), a, bytes, sources)); }
        Ok(Vec::new())
    }
    /// readbacks-device: queue every failure slot of this launch into the step ledger (a stream-ordered device copy,
    /// no host wait); the ledger is read once when the step ends and a fired slot refuses the step with `label`.
    pub unsafe fn queue_failures(&self, cuda: &Cuda, b: &mut Bound, led: &mut dev::Ledger, label: &str) -> Result<(), String> {
        for (entry, a, bytes, sources) in b.failures.drain(..) { led.queue_words(cuda, label, &entry, a, bytes, sources)?; }
        Ok(())
    }
    /// Read every queued failure slot (one synchronization) and name the sources that fired.
    pub unsafe fn check_failures(&self, cuda: &Cuda, b: &mut Bound) -> Result<Vec<String>, String> {
        dev::sync(cuda).map_err(|e| format!("{}: {e}", self.function))?;
        let mut fired = Vec::new();
        for (entry, a, bytes, sources) in b.failures.drain(..) {
            let host = dev::download(cuda, a, bytes)?;
            for (i, w) in host.chunks_exact(4).enumerate() {
                if w != [0, 0, 0, 0] {
                    fired.push(format!("{entry}: {}", sources.get(i).cloned().unwrap_or_else(|| format!("slot word {i}"))));
                }
            }
        }
        Ok(fired)
    }
}

/// Recipes this dispatcher can restate (crate::recipes); every result is checked
/// against the SHA256 its spec pins before it is used.
/// ad/held-tempo_to: build the primal mirror of pointer table `name` (pre-launch copies of its targets). Called from
/// prepare_with, before the forward launch runs. Returns true (always handled here; errors name the table).
unsafe fn bind_mirror_primal(kd: &KernelDeriv, cuda: &Cuda, arena: &mut Arena, l: &LaunchInfo,
                             exec_alloc: &HashMap<u64, submit::Allocation>, b: &mut Bound, name: &str) -> Result<bool, String> {
    let f = &kd.function;
    let (alloc, offset) = kd.device_target(l, name)?.ok_or_else(|| format!("{f}: primal pointer table {name} is not a device argument"))?;
    let decl = kd.bindings_all().find_map(|x| (x["bind"] == "primal" && x["arg"] == name && x["indirect"].is_object()).then(|| x["indirect"].clone()))
        .ok_or_else(|| format!("{f}: primal mirror {name} without indirect"))?;
    let n = kd.elements(name, &b.env)?;
    let a = exec_alloc.get(&alloc).ok_or_else(|| format!("{f}: {name} table allocation {alloc} not live"))?;
    if offset + n * 8 > a.bytes { return Err(format!("{f}: primal table {name} ({n} entries) exceeds its allocation")); }
    // One rule for every primal pointer table (ad/g1-merged): tempo_to spells the reach past each pointer
    // indirect.target_bytes (with indirect.mirror), the RUC sea-ice drivers indirect.target_bytes + primal_pre
    // (seaice_finish) or indirect.slot_bytes + primal_snapshot (seaice_prepare). All three mean the same thing: the
    // entries read pre-launch copies of every target.
    let key = if decl["target_bytes"].is_null() { "slot_bytes" } else { "target_bytes" };
    if decl[key].is_null() { return Err(format!("{f}: primal pointer table {name}: indirect has neither target_bytes nor slot_bytes")); }
    let reach = expr::eval_usize(&decl[key], &b.env).map_err(|e| format!("{f}: {name} {key}: {e}"))?;
    if reach == 0 { return Err(format!("{f}: primal pointer table {name}: {key} is 0")); }
    let table = dev::download(cuda, a.address + offset as u64, n * 8)?;
    let ptrs: Vec<u64> = table.chunks_exact(8).map(|w| u64::from_le_bytes(w.try_into().unwrap())).collect();
    let owner = |p: u64| exec_alloc.values().find(|t| p >= t.address && p < t.address + t.bytes as u64)
        .ok_or_else(|| format!("{f}: {name} entry {p:#x} is not in a plan allocation"));
    let mut spans: BTreeMap<u64, (usize, usize)> = BTreeMap::new();
    for &p in &ptrs {
        if p == 0 { continue; }
        let t = owner(p)?;
        let off = (p - t.address) as usize;
        let e = spans.entry(t.id).or_insert((off, off));
        e.0 = e.0.min(off); e.1 = e.1.max((off + reach).min(t.bytes));
    }
    let mut copies: BTreeMap<u64, (u64, usize)> = BTreeMap::new();
    for (id, (lo, hi)) in &spans {
        let t = &exec_alloc[id];
        let c = arena.take((hi - lo).max(4))?;
        dev::copy(cuda, c, t.address + *lo as u64, hi - lo)?;
        copies.insert(*id, (c, *lo));
    }
    let mut mirrored = Vec::with_capacity(n * 8);
    for &p in &ptrs {
        let m = if p == 0 { 0 } else { let t = owner(p)?; let (c, lo) = copies[&t.id]; c + (p - t.address) - lo as u64 };
        mirrored.extend_from_slice(&m.to_le_bytes());
    }
    let tab = arena.take(mirrored.len())?;
    dev::upload(cuda, tab, &mirrored)?;
    b.primal_mirror.insert(name.to_owned(), tab);
    Ok(true)
}

fn constructible(b: &Value) -> bool {
    let r = b["recipe"].as_str().unwrap_or("");
    // Physics-on window L3: the prep_interfaces capacities have no pinned SHA256 (they follow the launch's views);
    // the restated recipe's own checks are the admission (refused, never guessed).
    if r.contains("glue.interface_shapes.rs") { return true; }
    // G6: derive_edge_incidence.py (pv_apvm_v841_f32) and u2:inverse_adv_cells (transport_edge_values) are restated
    // too. Before, both kernels were refused in every window (202 launches per 18 steps) and their tangent outputs kept
    // stale bytes (allocation 448, norm 1e53, injected into every tracer by transport_standard_finish).
    b["sha256"].is_string() && ["transpose_fixed", "transpose.rs#build", "tools/dycore_csr.rs", "edge_to_slots",
                                "inverse_topology.rs::theta", "inverse_topology.rs::momentum",
                                "derive_edge_incidence.py", "inverse_adv_cells"].iter().any(|k| r.contains(k))
}

impl KernelDeriv {
    fn from_names(bind: &Value) -> Vec<String> {
        bind["from"].as_array().into_iter().flatten().filter_map(|v| v.as_str()).map(|s| {
            let s = s.trim_start_matches("mesh:").trim_start_matches("same-mesh ").trim_start_matches("captured mesh ");
            s.rsplit('.').next().unwrap_or(s).to_owned()
        }).collect()
    }
    unsafe fn array(&self, cuda: &Cuda, l: &LaunchInfo, env: &Env, live: &HashMap<u64, submit::Allocation>, name: &str) -> Result<Vec<i32>, String> {
        let (alloc, offset) = self.device_target(l, name)?.ok_or_else(|| format!("derived input {name} is not a device argument"))?;
        let n = self.elements(name, env)?;
        let a = live.get(&alloc).ok_or("derived input allocation")?;
        if offset + n * 4 > a.bytes { return Err(format!("derived input {name} exceeds its allocation")); }
        let mut raw = vec![0u8; n * 4];
        crate::cuda::checked("cuMemcpyDtoH derived", (cuda.download)(raw.as_mut_ptr().cast(), a.address + offset as u64, n * 4))?;
        Ok(raw.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect())
    }
    /// G6: an int32 device argument of a known element count (the recipe's own shape, not the capture row's).
    unsafe fn array_n(&self, cuda: &Cuda, l: &LaunchInfo, live: &HashMap<u64, submit::Allocation>, name: &str, n: usize) -> Result<Vec<i32>, String> {
        let (alloc, offset) = self.device_target(l, name)?.ok_or_else(|| format!("derived input {name} is not a device argument"))?;
        let a = live.get(&alloc).ok_or("derived input allocation")?;
        if offset + n * 4 > a.bytes { return Err(format!("derived input {name} exceeds its allocation")); }
        let mut raw = vec![0u8; n * 4];
        crate::cuda::checked("cuMemcpyDtoH derived", (cuda.download)(raw.as_mut_ptr().cast(), a.address + offset as u64, n * 4))?;
        Ok(raw.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect())
    }
    fn scalar_named(env: &Env, name: &str) -> Result<usize, String> {
        env.get(name).map(|v| *v as usize).ok_or_else(|| format!("derived input scalar {name} missing"))
    }
    /// Physics-on window L3: prep_interfaces' derived capacities and level lists, built in-process from the launch's
    /// bound views (glue.interface_shapes.rs restated in recipes.rs). Before, the dispatcher could not run that recipe
    /// and the 18 prep_interfaces launches of the 18-step plan stayed unbound. Each view is the declared spec shape at
    /// this launch, contiguous float32, and must lie inside its plan allocation.
    unsafe fn interface_shapes(&self, bind: &Value, l: &LaunchInfo, env: &Env, live: &HashMap<u64, submit::Allocation>) -> Result<Vec<u8>, String> {
        use crate::recipes as r;
        let from = Self::from_names(bind);
        if from.len() != 8 { return Err(format!("{}: interface_shapes takes 8 inputs, got {}", self.function, from.len())); }
        let nlev = Self::scalar_named(env, &from[0])? as i32;
        let ncells = Self::scalar_named(env, &from[1])? as i32;
        let nc = ncells as usize;
        let view = |name: &str| -> Result<([usize; 2], [usize; 2]), String> {
            let n = self.elements(name, env)?;
            let (alloc, offset) = self.device_target(l, name)?.ok_or_else(|| format!("{name} is not a device argument"))?;
            let a = live.get(&alloc).ok_or("interface view allocation")?;
            if nc == 0 || n % nc != 0 || offset + n * 4 > a.bytes { return Err(format!("{}: view of {name} ({n} words) does not fit its allocation", self.function)); }
            Ok(([n / nc, nc], [4 * nc, 4]))
        };
        let mass = [view(&from[2])?, view(&from[3])?, view(&from[4])?, view(&from[5])?];
        let weights = [view(&from[6])?, view(&from[7])?];
        let threads = (l.grid[0] as u64) * (l.grid[1] as u64) * (l.grid[2] as u64) * (l.block[0] as u64) * (l.block[1] as u64) * (l.block[2] as u64);
        let (caps, mass_levels, interior_levels) = r::interface_shapes(nlev, ncells, threads, mass, weights)?;
        let name = bind["name"].as_str().unwrap_or("");
        let values = match name {
            "pres_hyd_p_capacity" => vec![caps[0]], "pres_hydd_p_capacity" => vec![caps[1]],
            "znu_p_capacity" => vec![caps[2]], "znu_hyd_p_capacity" => vec![caps[3]],
            "mass_levels" => mass_levels, "interior_levels" => interior_levels,
            other => return Err(format!("{}: interface_shapes has no output {other}", self.function)),
        };
        Ok(r::pack(&values))
    }
    /// Build one derived binding from the fixed topology this launch binds, and
    /// refuse it unless its SHA256 equals the spec's.
    pub unsafe fn construct(&self, cuda: &Cuda, bind: &Value, l: &LaunchInfo, env: &Env, live: &HashMap<u64, submit::Allocation>) -> Result<Vec<u8>, String> {
        use crate::recipes as r;
        let recipe = bind["recipe"].as_str().unwrap_or("");
        if recipe.contains("glue.interface_shapes.rs") { return self.interface_shapes(bind, l, env, live); }
        let want = bind["sha256"].as_str().ok_or("derived without sha256")?;
        let name = bind["name"].as_str().or(bind["output"].as_str()).unwrap_or("");
        let component = bind["component"].as_str().unwrap_or("");
        let from = Self::from_names(bind);
        let f = |i: usize| from.get(i).cloned().ok_or_else(|| format!("{recipe}: missing input {i}"));
        let pick2 = |(o, e): (Vec<i32>, Vec<i32>)| if name.contains("offsets") || component == "offsets" { o } else { e };
        let values: Vec<i32> = if recipe.contains("transpose_fixed") {
            if from.len() >= 5 {
                let idx = self.array(cuda, l, env, live, &f(0)?)?; let cnt = self.array(cuda, l, env, live, &f(1)?)?;
                let (w, cols, rows) = (Self::scalar_named(env, &f(2)?)?, Self::scalar_named(env, &f(3)?)?, Self::scalar_named(env, &f(4)?)?);
                pick2(r::transpose_fixed(rows, cols, w, &cnt, &idx)?)
            } else {
                let idx = self.array(cuda, l, env, live, &f(0)?)?;
                let (cols, rows) = (Self::scalar_named(env, &f(1)?)?, Self::scalar_named(env, &f(2)?)?);
                let w = idx.len() / rows.max(1);
                pick2(r::transpose_fixed(rows, cols, w, &vec![w as i32; rows], &idx)?)
            }
        } else if recipe.contains("transpose.rs#build") {
            let cnt = self.array(cuda, l, env, live, &f(0)?)?; let idx = self.array(cuda, l, env, live, &f(1)?)?;
            let (nfrom, nto, w) = (Self::scalar_named(env, &f(2)?)?, Self::scalar_named(env, &f(3)?)?, Self::scalar_named(env, &f(4)?)?);
            let (o, owners, slots) = r::build(nfrom, nto, w, &cnt, &idx)?;
            match component { "offsets" => o, "slots" => slots, _ => owners }
        } else if recipe.contains("tools/dycore_csr.rs") {
            let p = &bind["parameters"];
            let values = self.array(cuda, l, env, live, &f(0)?)?;
            let counts = if from.len() > 1 { Some(self.array(cuda, l, env, live, &f(1)?)?) } else { None };
            let width = match &p["width"] { Value::Number(n) => n.as_u64().unwrap_or(1) as usize, Value::String(s) => Self::scalar_named(env, s)?, _ => return Err("csr width".into()) };
            let nrows = match p["nrows"].as_str() {
                Some(s) => env.get(s).map(|v| *v as usize).unwrap_or_else(|| values.iter().copied().max().unwrap_or(-1).max(-1) as usize + 1),
                None => return Err("csr nrows".into()),
            };
            let (o, e) = r::csr_pack(&values, counts.as_deref(), nrows, width, p["pack"] == "owner", p["skip_negative"] == true)?;
            if component == "offsets" { o } else { e }
        } else if recipe.contains("edge_to_slots") {
            let cnt = self.array(cuda, l, env, live, &f(0)?)?; let edges = self.array(cuda, l, env, live, &f(1)?)?;
            let ncells = cnt.len();
            let nedges = Self::scalar_named(env, "nedges").or_else(|_| Self::scalar_named(env, "n_edges"))?;
            pick2(r::edge_to_slots(ncells, nedges, edges.len() / ncells.max(1), &cnt, &edges)?)
        } else if recipe.contains("inverse_topology.rs::theta") || recipe.contains("inverse_topology.rs::momentum") {
            let (nc, ne, w) = (Self::scalar_named(env, &f(0)?)?, Self::scalar_named(env, &f(1)?)?, Self::scalar_named(env, &f(2)?)?);
            let cnt = self.array(cuda, l, env, live, &f(3)?)?; let inc = self.array(cuda, l, env, live, &f(4)?)?; let cells = self.array(cuda, l, env, live, &f(5)?)?;
            let (first, second) = if recipe.contains("theta") { r::theta(nc, ne, w, &cnt, &inc, &cells)? } else { r::momentum(nc, ne, w, &cnt, &inc, &cells)? };
            let first_names = if recipe.contains("theta") { "edge_" } else { "edge_" };
            let csr = if name.starts_with(first_names) { first } else { second };
            pick2(csr)
        } else if recipe.contains("derive_edge_incidence.py") {
            // from = [endpoints [nedges,2], entity count scalar]; outputs by name: the padded rows, the counts (n_*), the width.
            let n = Self::scalar_named(env, &f(1)?)?;
            let ne = Self::scalar_named(env, "nedges").or_else(|_| Self::scalar_named(env, "n_edges"))?;
            let endpoints = self.array_n(cuda, l, live, &f(0)?, 2 * ne)?;
            let (edges, counts, width) = r::edge_incidence(&endpoints, n)?;
            if name.starts_with("n_") || component == "counts" { counts }
            else if name.starts_with("max_") || name.ends_with("_degree") || component == "width" { vec![width] }
            else { edges }
        } else if recipe.contains("inverse_adv_cells") {
            let (nc, ne, w) = (Self::scalar_named(env, "ncells")?, Self::scalar_named(env, "nedges")?, Self::scalar_named(env, "width")?);
            let n_adv = self.array_n(cuda, l, live, "n_adv", ne)?; let cells = self.array_n(cuda, l, live, "adv_cells", ne * w)?;
            let (off, ent) = r::inverse_adv_cells(nc, ne, w, &n_adv, &cells)?;
            if name.ends_with("off") || name.contains("offsets") || component == "offsets" { off } else { ent }
        } else { return Err(format!("no restatement for recipe {recipe}")); };
        let bytes = r::pack(&values);
        let got = crate::exec::digest(&bytes);
        if got != want {
            // S01b diagnostic (AD_DERIVED_REPORT=1): report the table this mesh gives and use it, so a kernel set can be
            // re-pinned to a new mesh (kset/s01b: the 3 km LAM; kset/pk120: x1.40962). THE BREAKAGE THIS PREVENTS: every
            // derived topology table in the kernel set is the x1.2562 mesh's; on another mesh a stored table was used
            // silently (LAM: vertical_u_flux_f32 per-launch dot FAIL rel 1.2, theta_finish_v841_f32 CUDA 700) and the
            // restated ones refused.
            if std::env::var("AD_DERIVED_REPORT").as_deref() == Ok("1") {
                eprintln!("{}", serde_json::json!({"event":"derived_restated","function":self.function,"name":name,"recipe":recipe,"sha256":got,"spec_sha256":want,"bytes":bytes.len()}));
                return Ok(bytes);
            }
            return Err(format!("{}: derived {name} ({recipe}) SHA256 {got} differs from the spec's {want}", self.function));
        }
        Ok(bytes)
    }
}

#[cfg(test)]
mod held_mpas_cmbgb_tests {
    use super::*;
    /// Held lane mpas_cmbgb (2026-11-06). Its out extent is points*ngc[band]. The spec writes ngc[band] in the
    /// expr.rs grammar through launch_abi.derived_scalars, the first spec to use that path.
    /// THE BREAKAGE THIS PREVENTS: a wrong extent makes view() refuse "view of out exceeds its allocation", or binds a
    /// short shadow span the entry writes past. Checked for every band and the captured point counts, with roles().
    #[test]
    fn mpas_cmbgb_spec_extents_and_roles() {
        let spec: Value = serde_json::from_str(include_str!("../../phys/held/mpas_cmbgb/mpas_cmbgb.json")).unwrap();
        let row: Value = serde_json::from_str(include_str!("../../phys/held/mpas_cmbgb/kset-row.json")).unwrap();
        let meta: Vec<Value> = row["arguments"].as_array().unwrap().clone();
        let names: Vec<String> = meta.iter().map(|m| m["name"].as_str().unwrap().to_owned()).collect();
        let k = KernelDeriv { function: "mpas_cmbgb".into(), spec, names, meta, module: ptr::null_mut(), functions: HashMap::new(),
            tl_order: vec!["tl".into()], ad_order: vec!["ad".into()], aug: true, unsupported: None, derived: Default::default() };
        let ngc: [u32; 16] = [10, 12, 16, 14, 16, 8, 12, 8, 12, 6, 8, 8, 4, 2, 2, 2];
        for (band, &g) in ngc.iter().enumerate() {
            for points in [1u32, 4, 10, 19, 65, 235] {
                let i = |v: u32| submit::Argument::Bytes((v as i32).to_le_bytes().to_vec());
                let d = |a: u64| submit::Argument::Device(submit::Slice { allocation: a, offset: 0 });
                let args = vec![i(band as u32), i(points), i(0), i(1), d(0), d(1), d(2)];
                let l = LaunchInfo { args: &args, grid: [(points * g + 127) / 128, 1, 1], block: [128, 1, 1], shared: 0 };
                let env = k.env(&l).unwrap();
                assert_eq!(k.elements("out", &env).unwrap(), (points * g) as usize, "band {band} points {points}");
                assert_eq!(k.elements("raw", &env).unwrap(), (points * 16) as usize);
                assert_eq!(k.elements("rwgt", &env).unwrap(), 256);
            }
        }
        // raw/rwgt: TL in, AD accumulate with spec role in (no snapshot); out: TL out, AD inout (consumed in place).
        let r = k.roles();
        assert!(r.tl_snapshot.is_empty() && r.ad_snapshot.is_empty() && r.ad_clear.is_empty() && r.ad_replace.is_empty() && r.ad_post.is_empty());
        assert!(unsupported(&k.spec, &["aug".into(), "tl".into(), "ad".into()], &HashMap::new()).is_none());
    }
}

/// integ8: the held fixture rows live in tree/ad/phys/held. This file is compiled by two crates (tree/ad/e2e and the
/// runner at tree/ad/phys/binder/tools/binder/phys/runner), so the fixture tests resolve that folder from whichever
/// crate is building them (S09 fixed the e2e crate's path and broke the runner's: 4 runner tests NotFound).
#[cfg(test)]
pub(crate) fn held_fixture_dir() -> std::path::PathBuf {
    let m = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let e2e = m.join("../phys/held");
    if e2e.is_dir() { e2e } else { m.join("../../../../../held") }
}
#[cfg(test)]
mod held_mpas_init_rwgt {
    //! ad/held-mpas_init_rwgt: the RRTMG LW setup kernel mpas_init_rwgt (tree/ad/phys/held/mpas_init_rwgt/spec) writes its
    //! one argument, rwgt[256], from compiled-in constant tables only, so its derivative is the zero map on rwgt. The
    //! entry is dispatchable by this unit, binds in place (TL `out` writes +0.0, AD `inout` drops the cotangent of the
    //! overwritten words: no snapshot, clear, replace or post copy) and its shadow span is the 256 words of the launch.
    use super::*;

    #[test]
    fn mpas_init_rwgt_constant_table_entry_binds_in_place() {
        let root = held_fixture_dir().join("mpas_init_rwgt/spec");
        let spec: Value = serde_json::from_slice(&std::fs::read(root.join("mpas_init_rwgt.spec.json")).unwrap()).unwrap();
        let row: Value = serde_json::from_slice(&std::fs::read(root.join("mpas_init_rwgt.kset-row.json")).unwrap()).unwrap();
        let abi = spec["launch_abi"].clone();
        let names = row["arguments"].as_array().unwrap().iter().map(|a| a["name"].as_str().unwrap().to_owned()).collect();
        let meta = row["arguments"].as_array().unwrap().clone();
        let (tl_order, ad_order, aug) = (order(&abi, "tl"), order(&abi, "ad"), abi["entries"]["aug"].is_object());
        let all: Vec<String> = tl_order.iter().chain(ad_order.iter()).cloned().collect();
        let why = unsupported(&spec, &all, &HashMap::new());
        let k = KernelDeriv { function: row["function"].as_str().unwrap().to_owned(), spec, names, meta, module: ptr::null_mut(),
                              functions: HashMap::new(), tl_order, ad_order, aug, unsupported: why, derived: std::cell::RefCell::new(HashMap::new()) };
        assert_eq!(k.unsupported, None);
        assert_eq!((k.tl_order.clone(), k.ad_order.clone(), k.aug), (vec!["tl".to_owned()], vec!["ad".to_owned()], false));
        let r = k.roles();
        assert!(r.tl_snapshot.is_empty() && r.ad_snapshot.is_empty() && r.ad_clear.is_empty() && r.ad_replace.is_empty() && r.ad_post.is_empty(),
                "mpas_init_rwgt roles: tl_snapshot {:?} ad_snapshot {:?} ad_clear {:?} ad_replace {:?} ad_post {:?}",
                r.tl_snapshot, r.ad_snapshot, r.ad_clear, r.ad_replace, r.ad_post);
        assert_eq!(k.elements("rwgt", &Env::new()).unwrap(), 256);
        // No failure slot: a fired slot refuses the window step (window.rs), and this map discards every incoming word.
        assert!(k.bindings_all().all(|b| b["bind"] != "failure"));
    }
}

#[cfg(test)]
mod held_mpas_layer_histogram {
    //! ad/held-mpas_layer_histogram: the RRTMG LW wrapper layer-count histogram entry
    //! (tree/ad/phys/held/mpas_layer_histogram) is dispatchable by this unit, binds in place (no tangent snapshot, no
    //! adjoint snapshot/clear/replace/post copy), and its shadow spans follow each launch's nc (counts) and nbins+1 (hist).
    use super::*;

    fn entry(spec: &Path, row: &Path) -> KernelDeriv {
        let spec: Value = serde_json::from_slice(&std::fs::read(spec).unwrap()).unwrap();
        let row: Value = serde_json::from_slice(&std::fs::read(row).unwrap()).unwrap();
        let abi = spec["launch_abi"].clone();
        let names = row["arguments"].as_array().unwrap().iter().map(|a| a["name"].as_str().unwrap().to_owned()).collect();
        let meta = row["arguments"].as_array().unwrap().clone();
        let (tl_order, ad_order, aug) = (order(&abi, "tl"), order(&abi, "ad"), abi["entries"]["aug"].is_object());
        let mut all: Vec<String> = tl_order.iter().chain(ad_order.iter()).cloned().collect();
        if aug { all.push("aug".into()); }
        let why = unsupported(&spec, &all, &HashMap::new());
        KernelDeriv { function: row["function"].as_str().unwrap().to_owned(), spec, names, meta, module: ptr::null_mut(),
                      functions: HashMap::new(), tl_order, ad_order, aug, unsupported: why, derived: std::cell::RefCell::new(HashMap::new()) }
    }

    #[test]
    fn mpas_layer_histogram_inactive_entry_binds_in_place() {
        let root = held_fixture_dir().join("mpas_layer_histogram");
        let k = entry(&root.join("mpas_layer_histogram.spec.json"), &root.join("mpas_layer_histogram.kset-row.json"));
        assert_eq!(k.unsupported, None);
        assert_eq!((k.tl_order.clone(), k.ad_order.clone(), k.aug), (vec!["tl".to_owned()], vec!["ad".to_owned()], false));
        let r = k.roles();
        assert!(r.tl_snapshot.is_empty() && r.ad_snapshot.is_empty() && r.ad_clear.is_empty() && r.ad_replace.is_empty() && r.ad_post.is_empty(),
                "mpas_layer_histogram roles: tl_snapshot {:?} ad_snapshot {:?} ad_clear {:?} ad_replace {:?} ad_post {:?}",
                r.tl_snapshot, r.ad_snapshot, r.ad_clear, r.ad_replace, r.ad_post);
        // Shadow spans per launch: the captures (nc 2562, nbins 1024), a small batch, a wider histogram.
        for (nc, nbins) in [(2562usize, 1024usize), (37, 1024), (216187, 2048)] {
            let mut env = Env::new();
            env.insert("nc".into(), nc as f64);
            env.insert("base".into(), 60.0);
            env.insert("nbins".into(), nbins as f64);
            assert_eq!(k.elements("counts", &env).unwrap(), nc);
            assert_eq!(k.elements("hist", &env).unwrap(), nbins + 1);
        }
    }
}

#[cfg(test)]
mod held_mpas_pressure_mb {
    //! ad/held-mpas_pressure_mb: the RRTMG LW mpas_pressure_mb entry (tree/ad/phys/held/mpas_pressure_mb/spec) is
    //! dispatchable by this unit and takes the same in-place roles as the live affine glue add rrfs_pbl_radiation_sum
    //! (kset-integ): no tangent snapshot, no adjoint snapshot/clear/replace/post copy; its shadow spans follow each
    //! launch's n.
    use super::*;

    fn entry(spec: &Path, row: &Path) -> KernelDeriv {
        let spec: Value = serde_json::from_slice(&std::fs::read(spec).unwrap()).unwrap();
        let row: Value = serde_json::from_slice(&std::fs::read(row).unwrap()).unwrap();
        let abi = spec["launch_abi"].clone();
        let names = row["arguments"].as_array().unwrap().iter().map(|a| a["name"].as_str().unwrap().to_owned()).collect();
        let meta = row["arguments"].as_array().unwrap().clone();
        let (tl_order, ad_order, aug) = (order(&abi, "tl"), order(&abi, "ad"), abi["entries"]["aug"].is_object());
        let mut all: Vec<String> = tl_order.iter().chain(ad_order.iter()).cloned().collect();
        if aug { all.push("aug".into()); }
        let why = unsupported(&spec, &all, &HashMap::new());
        KernelDeriv { function: row["function"].as_str().unwrap().to_owned(), spec, names, meta, module: ptr::null_mut(),
                      functions: HashMap::new(), tl_order, ad_order, aug, unsupported: why, derived: std::cell::RefCell::new(HashMap::new()) }
    }

    #[test]
    fn mpas_pressure_mb_entry_binds_in_place_like_the_live_affine_glue_add() {
        let root = held_fixture_dir().join("mpas_pressure_mb/spec");
        let k = entry(&root.join("mpas_pressure_mb.spec.json"), &root.join("mpas_pressure_mb.kset-row.json"));
        assert_eq!(k.unsupported, None);
        assert_eq!((k.tl_order.clone(), k.ad_order.clone(), k.aug), (vec!["tl".to_owned()], vec!["ad".to_owned()], true));
        let r = k.roles();
        assert!(r.tl_snapshot.is_empty() && r.ad_snapshot.is_empty() && r.ad_clear.is_empty() && r.ad_replace.is_empty() && r.ad_post.is_empty(),
                "mpas_pressure_mb roles: tl_snapshot {:?} ad_snapshot {:?} ad_clear {:?} ad_replace {:?} ad_post {:?}",
                r.tl_snapshot, r.ad_snapshot, r.ad_clear, r.ad_replace, r.ad_post);
        // The live template, where the kernel set is present (the integration box path of kset-integ).
        let t = Path::new("/work/ad/physon/kset-integ/specs/rrfs_pbl_radiation_sum.json");
        if t.exists() {
            let tmp = std::env::temp_dir().join("mpas_pressure_mb_template_row.json");
            std::fs::write(&tmp, r#"{"function":"rrfs_pbl_radiation_sum","arguments":[{"name":"n","kind":"scalar","dtype":"<i4","shape":[]},{"name":"lw","kind":"device","dtype":"<f4","shape":[]},{"name":"sw","kind":"device","dtype":"<f4","shape":[]},{"name":"out","kind":"device","dtype":"<f4","shape":[]}]}"#).unwrap();
            let g = entry(t, &tmp).roles();
            assert!(g.tl_snapshot.is_empty() && g.ad_snapshot.is_empty() && g.ad_clear.is_empty() && g.ad_replace.is_empty() && g.ad_post.is_empty());
        }
        // Shadow spans per launch (captures: n = 151,158 = 2,562 cells x 59 levels; any n follows the scalar).
        for n in [151158usize, 2562 * 60, 1] {
            let mut env = Env::new();
            env.insert("n".into(), n as f64);
            assert_eq!(k.elements("p", &env).unwrap(), n);
            assert_eq!(k.elements("out", &env).unwrap(), n);
        }
    }
}

#[cfg(test)]
mod held_rlw_taut {
    //! ad/held-rlw_taut: the RRTMG LW rlw_taut entry (tree/ad/phys/held/rlw_taut/spec) is dispatchable by this unit and
    //! takes the same in-place roles as the live affine glue add rrfs_pbl_radiation_sum (kset-integ): no tangent
    //! snapshot, no adjoint snapshot/clear/replace/post copy; its shadow spans follow each launch's ncol and nl.
    use super::*;

    fn entry(spec: &Path, row: &Path) -> KernelDeriv {
        let spec: Value = serde_json::from_slice(&std::fs::read(spec).unwrap()).unwrap();
        let row: Value = serde_json::from_slice(&std::fs::read(row).unwrap()).unwrap();
        let abi = spec["launch_abi"].clone();
        let names = row["arguments"].as_array().unwrap().iter().map(|a| a["name"].as_str().unwrap().to_owned()).collect();
        let meta = row["arguments"].as_array().unwrap().clone();
        let (tl_order, ad_order, aug) = (order(&abi, "tl"), order(&abi, "ad"), abi["entries"]["aug"].is_object());
        let mut all: Vec<String> = tl_order.iter().chain(ad_order.iter()).cloned().collect();
        if aug { all.push("aug".into()); }
        let why = unsupported(&spec, &all, &HashMap::new());
        KernelDeriv { function: row["function"].as_str().unwrap().to_owned(), spec, names, meta, module: ptr::null_mut(),
                      functions: HashMap::new(), tl_order, ad_order, aug, unsupported: why, derived: std::cell::RefCell::new(HashMap::new()) }
    }

    #[test]
    fn rlw_taut_entry_binds_in_place_like_the_live_affine_glue_add() {
        let root = held_fixture_dir().join("rlw_taut/spec");
        let k = entry(&root.join("rlw_taut.spec.json"), &root.join("rlw_taut.kset-row.json"));
        assert_eq!(k.unsupported, None);
        assert_eq!((k.tl_order.clone(), k.ad_order.clone(), k.aug), (vec!["tl".to_owned()], vec!["ad".to_owned()], true));
        let r = k.roles();
        assert!(r.tl_snapshot.is_empty() && r.ad_snapshot.is_empty() && r.ad_clear.is_empty() && r.ad_replace.is_empty() && r.ad_post.is_empty(),
                "rlw_taut roles: tl_snapshot {:?} ad_snapshot {:?} ad_clear {:?} ad_replace {:?} ad_post {:?}",
                r.tl_snapshot, r.ad_snapshot, r.ad_clear, r.ad_replace, r.ad_post);
        // The live template, where the kernel set is present (the integration box path of kset-integ).
        let t = Path::new("/work/ad/physon/kset-integ/specs/rrfs_pbl_radiation_sum.json");
        if t.exists() {
            let tmp = std::env::temp_dir().join("rlw_taut_template_row.json");
            std::fs::write(&tmp, r#"{"function":"rrfs_pbl_radiation_sum","arguments":[{"name":"n","kind":"scalar","dtype":"<i4","shape":[]},{"name":"lw","kind":"device","dtype":"<f4","shape":[]},{"name":"sw","kind":"device","dtype":"<f4","shape":[]},{"name":"out","kind":"device","dtype":"<f4","shape":[]}]}"#).unwrap();
            let g = entry(t, &tmp).roles();
            assert!(g.tl_snapshot.is_empty() && g.ad_snapshot.is_empty() && g.ad_clear.is_empty() && g.ad_replace.is_empty() && g.ad_post.is_empty());
        }
        // Shadow spans per launch (captures: 2,496-2,525 x 64 and 37-66 x 65 columns x layers).
        for (ncol, nl) in [(2525usize, 64usize), (37, 65)] {
            let mut env = Env::new();
            env.insert("ncol".into(), ncol as f64);
            env.insert("nl".into(), nl as f64);
            assert_eq!(k.elements("taug", &env).unwrap(), ncol * nl * 140);
            assert_eq!(k.elements("taut", &env).unwrap(), ncol * nl * 140);
            assert_eq!(k.elements("taua", &env).unwrap(), ncol * nl * 16);
            assert_eq!(k.elements("ngb", &env).unwrap(), 140);
        }
    }
}

fn d_is_like_forward(decl: Option<&Value>) -> bool { decl.is_some_and(|d| d["shape"]["shape_like_forward_arg"].is_string()) }
