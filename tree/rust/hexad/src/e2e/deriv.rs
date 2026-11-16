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
    /// Only the audited image/spec pair may leave its private column scratch uninitialized.
    elide_column_workspace_zero: bool,
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

/// dycore: a spec path written relative in kernels.json is relative to the folder kernels.json is in, so a kernel
/// set is one folder that can be copied to another machine (the kernel-set builder writes them that way).
/// THE BREAKAGE THIS ENDS: every set named its specs by absolute lane paths in up to 13 folders; a test machine
/// <local path> (2026-11-09) named two specs under kset-c7/, which was deleted an hour later, and the set
/// could no longer load.
/// A set that records absolute paths (tree/ad/kset/int7f, the set of the pinned x1.2562 row) carries each spec beside
/// itself at `specs/<the recorded path without its leading />`. That copy is read first, and the recorded path only
/// for a set that lives where it was made. THE BREAKAGE THIS PREVENTS (hex 0.6.0 candidate, 2026-11-11): this
/// function used an absolute path as written, so the pinned row's set, 153 absolute paths of the machine that made
/// it, could load on that machine and nowhere else. Admission (window::check_spec_hashes) hashes the file this returns.
pub fn spec_path(kernels: &Path, p: &str) -> std::path::PathBuf {
    let q = Path::new(p);
    let dir = kernels.parent().unwrap_or(Path::new("."));
    if !q.is_absolute() { return dir.join(q); }
    let local = dir.join("specs").join(p.trim_start_matches('/'));
    if local.is_file() { local } else { q.to_path_buf() }
}

/// dycore: the launch geometry every derivative entry really ran with, beside the forward launch's own geometry.
/// THE BREAKAGE THIS SHOWS: an entry whose spec carries a numeric grid captured on x1.2562 keeps that grid on every mesh
/// (w_vertical_flux_f32 `ad` [21] x 128 ran 40,962 columns on 2,688 threads at 120 km; add_inplace_f32 `ad` [1121] x 128
/// ran 2.29 M words on 143,488 threads). A grid-stride kernel stays correct and slow, any other one is wrong, and nothing
/// said so. `geom_report` lists every entry (grid, block, launches, forward grid and block of its last launch) and flags
/// `literal_undersized`: a numeric multi-block grid, not marked `"mesh_free"` in its geometry, running fewer than a
/// quarter of the forward launch's threads. The window receipt carries the flagged rows; AD_GEOM_CENSUS=FILE writes all.
pub struct GeomRow { pub function: String, pub entry: String, pub grid: [u32; 3], pub block: [u32; 3], pub fgrid: [u32; 3], pub fblock: [u32; 3],
                     pub launches: u64, pub literal: bool, pub mesh_free: bool }
static GEOM: std::sync::Mutex<BTreeMap<u64, GeomRow>> = std::sync::Mutex::new(BTreeMap::new());
fn geom_note(function: &str, entry: &str, geometry: &Value, grid: [u32; 3], block: [u32; 3], fgrid: [u32; 3], fblock: [u32; 3]) {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    function.hash(&mut h); entry.hash(&mut h);
    let Ok(mut g) = GEOM.lock() else { return };
    let r = g.entry(h.finish()).or_insert_with(|| GeomRow { function: function.to_owned(), entry: entry.to_owned(), grid, block, fgrid, fblock, launches: 0,
        literal: geometry["grid"].as_array().is_some_and(|a| a.iter().all(|x| x.is_number())),
        mesh_free: geometry["mesh_free"].as_str().is_some_and(|s| !s.is_empty()) });
    r.launches += 1; r.grid = grid; r.block = block; r.fgrid = fgrid; r.fblock = fblock;
}
pub fn geom_report() -> Value {
    let Ok(g) = GEOM.lock() else { return Value::Null };
    let threads = |a: [u32; 3], b: [u32; 3]| a.iter().chain(b.iter()).map(|&x| x as u64).product::<u64>();
    let rows: Vec<Value> = g.values().map(|r| {
        let (t, ft) = (threads(r.grid, r.block), threads(r.fgrid, r.fblock));
        let blocks: u64 = r.grid.iter().map(|&x| x as u64).product();
        serde_json::json!({"function":r.function,"entry":r.entry,"grid":r.grid,"block":r.block,"threads":t,"forward_grid":r.fgrid,"forward_block":r.fblock,
            "forward_threads":ft,"launches":r.launches,"literal":r.literal,"mesh_free":r.mesh_free,
            "literal_undersized": r.literal && !r.mesh_free && blocks > 1 && t * 4 < ft})
    }).collect();
    Value::Array(rows)
}

/// Reasons a spec cannot be dispatched by this unit yet (named, never skipped silently).
fn unsupported(spec: &Value, entries: &[String], derived: &HashMap<String, Vec<u8>>) -> Option<String> {
    unsupported_with(spec, entries, derived, LOCAL_TOPOLOGY_REQUESTED.load(std::sync::atomic::Ordering::SeqCst))
}
/// g2 r2 (limiter live): set by the window BEFORE deriv::load when --local-topology 1 is in force. Under it, a derived
/// binding whose recipe is a topology recipe that topo::needs can build (crate::topo::topology_recipe) is admitted at
/// load: dispatch builds it from the launch's own mesh (local_build) and records equal/differs against the pin in the
/// local_topology receipt. THE BREAKAGE THIS PREVENTS: fct_minmax_source's inv_off / inv_nbr / inv_slot
/// (u2:inverse_cells_on_cell, pinned to the x1.2562 capture mesh) were judged at load, before the mesh registry
/// existed, so with --fct-frozen 0 every window on x1.40962 stopped at step 1 ("derived binding inv_nbr
/// (u2:inverse_cells_on_cell) not in the derived store and no recipe") although local_build has that recipe.
static LOCAL_TOPOLOGY_REQUESTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn request_local_topology(on: bool) { LOCAL_TOPOLOGY_REQUESTED.store(on, std::sync::atomic::Ordering::SeqCst); }
fn local_buildable(b: &Value) -> bool {
    b["recipe"].as_str().is_some_and(|r| crate::topo::topology_recipe(r)
        && crate::topo::needs(r, b["name"].as_str().or(b["output"].as_str()).unwrap_or("")).is_some())
}
fn unsupported_with(spec: &Value, entries: &[String], derived: &HashMap<String, Vec<u8>>, local: bool) -> Option<String> {
    let abi = &spec["launch_abi"];
    if abi["schema"] != "ad.entry/v1" { return Some("no ad.entry/v1 launch_abi".into()); }
    for name in entries {
        let e = &abi["entries"][name];
        if !e.is_object() { return Some(format!("missing entry {name}")); }
        for b in e["args"].as_array().into_iter().flatten() {
            let kind = b["bind"].as_str().unwrap_or("");
            if kind == "derived" && !b["sha256"].as_str().is_some_and(|h| derived.contains_key(h)) && !constructible(b) && !(local && local_buildable(b)) { return Some(format!("derived binding {} ({}) not in the derived store and no recipe", b["name"].as_str().or(b["output"].as_str()).unwrap_or("?"), b["recipe"].as_str().unwrap_or("?"))); }
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

/// The file a row's derivative entries are loaded from: `cubins/<function>.fatbin`, or the file the row names
/// (`derivative.shared_image`, a plain file name under `cubins`), so that rows built into one image share one file.
/// The key is not `image`: rows of the shipped kernel sets carry `derivative.image` as a record of where their image
/// came from, and reading that record as a name refused the whole kernel set (the first card run of this rule).
/// THE BREAKAGE THIS PREVENTS: the 26 New Tiedtke rows are one image of 6 MB that does not pack smaller; a kernel
/// set that had to hold it once a row was a 155 MB download for one scheme.
pub fn image_path(cubins: &Path, function: &str, derivative: &Value) -> Result<std::path::PathBuf, String> {
    match derivative.get("shared_image") {
        None | Some(Value::Null) => Ok(cubins.join(format!("{function}.fatbin"))),
        Some(Value::String(name)) if !name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\\']) => Ok(cubins.join(name)),
        Some(other) => Err(format!("{function}: derivative.shared_image {other} is not a file name inside the kernel set's cubins folder")),
    }
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
    // tape-set: the kernel set's zero-word table (tape_finite); rows applied, rows whose pins differ
    let finite = finite_table(kernels)?;
    let (mut fin_on, mut fin_off): (Vec<String>, Vec<String>) = (Vec::new(), Vec::new());
    for k in set["kernels"].as_array().ok_or("kernels missing")? {
        let function = k["function"].as_str().ok_or("function")?.to_owned();
        if only.is_some_and(|o| !o.contains(&function)) { continue; }
        let d = &k["derivative"];
        if !d.is_object() { continue; }
        let spec_file = spec_path(kernels, d["spec"]["path"].as_str().ok_or("spec path")?);
        let spec_bytes = std::fs::read(&spec_file).map_err(|e| format!("{}: {e}", spec_file.display()))?;
        let spec: Value = serde_json::from_slice(&spec_bytes).map_err(|e| format!("{}: {e}", spec_file.display()))?;
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
        let mut elide_column_workspace_zero = false;
        let mut module = ptr::null_mut();
        let mut functions = HashMap::new();
        if why.is_none() {
            let path = image_path(cubins, &function, d)?;
            match std::fs::read(&path) {
                Err(e) => why = Some(format!("no derivative image {}: {e}", path.display())),
                Ok(image) => {
                    if let Some((_, t)) = finite.as_ref() {
                        match finite_admit(t, &function, &image, &spec_bytes) {
                            Ok(Some(row)) => { FINITE.lock().map_err(|_| "tape-finite table poisoned".to_string())?.insert(function.clone(), row); fin_on.push(function.clone()); },
                            Ok(None) => {},
                            Err(what) => { FINITE.lock().map_err(|_| "tape-finite table poisoned".to_string())?.remove(&function); fin_off.push(format!("{function}: {what} differs")); },
                        }
                    }
                    // AD_MIN_WORKSPACE_CLEAR is opt-in. Pin BOTH bytes actually loaded and the
                    // binding protocol: older graph rows share this name and can need initialization.
                    // This hand-written row writes every scratch word before reading it; maps is unused.
                    elide_column_workspace_zero = std::env::var("AD_MIN_WORKSPACE_CLEAR").as_deref() == Ok("1")
                        && function == "acoustic_column_solve_regional_v841"
                        && crate::exec::digest(&image) == "07b4fc722a1a0bc99a6f7b5536e69c054941778e01d5423bef0305389e1690b1"
                        && crate::exec::digest(&spec_bytes)
                            == "4b0bbe4e498a05b3b3804b144a4de889c5d530ce40fd5842ded938f81adf6ade";
                    if std::env::var("AD_MIN_WORKSPACE_CLEAR").as_deref() == Ok("1")
                        && function == "acoustic_column_solve_regional_v841" {
                        eprintln!("{}", serde_json::json!({"event":"workspace_clear_policy", "function":function,
                            "elide_column_workspace_zero":elide_column_workspace_zero}));
                    }
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
        out.insert(function.clone(), KernelDeriv { function, spec, names, meta, module, functions, tl_order, ad_order, aug, elide_column_workspace_zero, unsupported: why, derived: std::cell::RefCell::new(on_device) });
    }
    if let Some((p, t)) = finite.as_ref() {
        eprintln!("{}", serde_json::json!({"event":"tape_finite_loaded","path":p.to_string_lossy(),"rows_in_table":t["rows"].as_object().map(|r| r.len()).unwrap_or(0),
                                           "rows_applied":fin_on.len(),"applied":fin_on,"not_applied":fin_off}));
    }
    Ok(out)
}

/// One forward launch as resolved from the plan.
/// span memo (AD_TAPE_SPAN_MEMO; on with AD_TAPE_PREDICT=1, =0 off): what prepare_tape's binding walk (prep_spans:
/// 562 calls and 0.040 s a 3 km re-run step in rr2-Bp) makes from the kernel's spec and the launch's arguments, kept for the
/// same kernel and the same arguments (every argument, grid, block and shared bytes hashed) and handed back on the next equal
/// launch, which in a kept-loaded program is the same launch of the gradient before. The only other inputs the walk reads
/// (the bytes of a mesh-extent allocation; whether a persistent shadow is float32) are recorded and checked on every reuse;
/// a difference walks again. Same bindings, same spans, same words.
#[derive(Clone)]
pub struct SpanMemo {
    env: Env, pre: Vec<u64>, tl_cells: BTreeMap<u64, (u64, usize)>, ad_cells: BTreeMap<u64, (u64, usize)>,
    widths: BTreeMap<u64, usize>, extents: BTreeMap<u64, (usize, usize)>, pre_spans: BTreeMap<u64, (usize, usize)>,
    arg_spans: BTreeMap<u64, BTreeMap<String, (usize, usize)>>, indirect: Vec<String>, primal_indirect: Vec<String>,
    widen: Vec<String>, widen_io: BTreeMap<String, (String, String)>, live_only: Vec<u64>, zero_args: Vec<(String, bool)>,
    /// (allocation, 0 = its bytes | 1 = persistent shadow is float32, value) as read by the walk
    consult: Vec<(u64, u8, usize)>,
}
thread_local! { static SPAN_MEMO: std::cell::RefCell<HashMap<(usize, u64), std::rc::Rc<SpanMemo>>> = std::cell::RefCell::new(HashMap::new()); }
pub fn span_memo_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| predict::on() && std::env::var("AD_TAPE_SPAN_MEMO").as_deref() != Ok("0"))
}
fn span_key(l: &LaunchInfo, width: usize, defer: bool, has_persistent: bool) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (l.grid, l.block, l.shared, width, defer, has_persistent, l.args.len()).hash(&mut h);
    for a in l.args {
        match a {
            submit::Argument::Bytes(b) => { 0u8.hash(&mut h); b.hash(&mut h); },
            submit::Argument::Device(s) => { 1u8.hash(&mut h); s.allocation.hash(&mut h); s.offset.hash(&mut h); },
            other => { 2u8.hash(&mut h); format!("{other:?}").hash(&mut h); },
        }
    }
    h.finish()
}
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
    /// S01b: Roles::ad_live_inout of this launch.
    pub live_inout: Vec<String>,
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
    /// tape-set: post-launch primal arguments bound to zero words (tape_finite): argument name -> base (zero words -
    /// the argument's offset). Overrides the post-launch copy for that argument; the pre-launch ones go in `pre_arg`.
    pub post_zero: HashMap<String, u64>,
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

/// tape-set (a12, 2026-11-13): primal arguments a row's reverse entries read only to refuse, or not at all, bound to
/// resident zero words on the reverse tape instead of a copy of their pre- or post-launch values. THE WASTE THIS REMOVES:
/// add_inplace_f32__ad reads `target` and `increment` only in `if (!isfinite(..) ..) ad_fail(..)`; recover_edges_f32 and
/// recover_interfaces_f32 read all five of their primal arguments only that way; yet every launch copied them onto the
/// tape (60 km, s8-m5hitB step 8: add_inplace 4.83 GB, recover_edges 4.30 GB, recover_interfaces 3.14 GB of a 94.9 GB
/// step tape, all of it out to the host and back). Zero words pass every finiteness test, and in a run that is not
/// refused (a fired failure slot refuses the step, queue_failures) the cotangent words do not depend on these values,
/// so the reverse words are the same. What the test no longer sees is a non-finite forward value of such an argument:
/// the forward values are the recorded run's, checked word for word at each re-run's end (the redo compare), and the
/// next kernel that reads them in arithmetic still tests them.
/// The table is the kernel set's own `tape-finite.json` beside kernels.json (tools/tape-finite: finite.py from each
/// row's pinned source, test_cpu.py the CPU proof that zero words leave every output word equal), schema
/// ad.tape-finite/v1: rows {function: {image_sha256, spec_sha256, pre: [arg], post: [arg]}}. A row is applied only when
/// the loaded image and spec hash to its pins (load), so a set whose images changed applies nothing stale.
/// AD_TAPE_FINITE=0 turns it off; AD_TAPE_FINITE=PATH names another table. A table that exists but cannot be read
/// refuses the run.
type FiniteRows = HashMap<String, (Vec<String>, Vec<String>)>;
static FINITE: std::sync::LazyLock<std::sync::Mutex<FiniteRows>> = std::sync::LazyLock::new(Default::default);
pub fn tape_finite(function: &str, arg: &str, post: bool) -> bool {
    FINITE.lock().map(|m| m.get(function).is_some_and(|(pre, po)| if post { po } else { pre }.iter().any(|x| x == arg))).unwrap_or(false)
}
fn finite_table(kernels: &Path) -> Result<Option<(std::path::PathBuf, Value)>, String> {
    let p = match std::env::var_os("AD_TAPE_FINITE") {
        Some(v) if v == "0" => return Ok(None),
        Some(v) => std::path::PathBuf::from(v),
        None => kernels.parent().unwrap_or(Path::new(".")).join("tape-finite.json"),
    };
    if !p.is_file() {
        if std::env::var_os("AD_TAPE_FINITE").is_some() { return Err(format!("AD_TAPE_FINITE {}: no such file", p.display())); }
        return Ok(None);
    }
    let t = std::fs::read(&p).map_err(|e| format!("tape-finite {}: {e}", p.display()))?;
    let v: Value = serde_json::from_slice(&t).map_err(|e| format!("tape-finite {}: {e}", p.display()))?;
    if v["schema"] != "ad.tape-finite/v1" || !v["rows"].is_object() { return Err(format!("tape-finite {}: not an ad.tape-finite/v1 table", p.display())); }
    Ok(Some((p, v)))
}
/// tape-set: the row of `function` when the loaded image and spec are the ones it was proved on.
fn finite_admit(table: &Value, function: &str, image: &[u8], spec: &[u8]) -> Result<Option<(Vec<String>, Vec<String>)>, String> {
    let r = &table["rows"][function];
    if !r.is_object() { return Ok(None); }
    if r["image_sha256"].as_str() != Some(crate::exec::digest(image).as_str()) { return Err("image".into()); }
    if r["spec_sha256"].as_str() != Some(crate::exec::digest(spec).as_str()) { return Err("spec".into()); }
    let list = |k: &str| r[k].as_array().into_iter().flatten().filter_map(|x| x.as_str().map(str::to_owned)).collect::<Vec<_>>();
    Ok(Some((list("pre"), list("post"))))
}
/// tape-set: the address of at least `bytes` zero words on the card, resident for the rest of the process (one
/// allocation of at least 64 MiB serves every binding that fits; a larger need makes one more). Nothing writes it:
/// the arguments bound there are inputs their reverse entries only test.
static ZERO_WORDS: std::sync::Mutex<Vec<(u64, usize)>> = std::sync::Mutex::new(Vec::new());
pub unsafe fn zero_words(cuda: &Cuda, bytes: usize) -> Result<u64, String> {
    let mut z = ZERO_WORDS.lock().map_err(|_| "zero words pool poisoned".to_string())?;
    if let Some(&(a, _)) = z.iter().find(|(_, n)| *n >= bytes) { return Ok(a); }
    let n = bytes.max(64 << 20).div_ceil(1 << 20) << 20;
    let mut p = 0u64;
    checked("cuMemAlloc tape-finite zero words", (cuda.allocate)(&mut p, n))?;
    dev::zero(cuda, p, n)?;
    dev::sync(cuda)?;
    z.push((p, n));
    eprintln!("{}", serde_json::json!({"event":"tape_finite_zero_words","bytes":n,"buffers":z.len()}));
    Ok(p)
}

/// rerun-tape: the pure decision book (tape_predict.rs) and its device glue (`predict`).
#[path = "tape_predict.rs"] pub mod tape_predict;
/// rerun-tape (AD_TAPE_PREDICT=1, off unless asked): the taped re-run's share decisions taken from the same
/// step of the gradient before (AD_TAPE_PREDICT_BOOK=file, read at the first use and written when step 1 is taped), with
/// the device compares still queued and their answers copied, on the device and in stream order, into a log that is
/// read back once at the chunk end (see tape_predict.rs). Unset: `on()` is false and every caller takes the unchanged
/// path (the same calls, in the same order).
pub mod predict {
    use super::tape_predict::{self, Book, Key};
    use crate::cuda::{checked, Cuda};
    use crate::dev;
    use serde_json::{json, Value};
    /// The error a taping attempt returns when a predicted decision did not hold; the caller tapes the step again.
    pub const MISS: &str = "speed16 tape predict: a predicted share decision did not hold";
    /// (AD_TAPE_PREDICT_TABLES, on with AD_TAPE_PREDICT=1 unless set to 0): the descriptor tables a rule reads
    /// from the device while a step is taped (wrules.rs: spec arrays, scrub tables, concatenate pointer arrays, copy-plan
    /// and halo tables with their index and dims arrays) get the book treatment. THE WASTE THIS REMOVES: each read was a
    /// blocking cuMemcpyDtoH that drained the card before the host could classify the next command (rule_classify 0.068
    /// -> 0.245 s a 3 km step once the share readbacks were gone). HOW: the bytes the same read gave in the same step of
    /// the gradient before (key: step, the book's current command, the read's order since that command, address, size),
    /// or failing that the bytes the same address and size gave in at least two real reads in a row with no miss, are
    /// handed to the rule at once; the real bytes are copied on the device, in stream order, into the predict log and
    /// compared at the chunk end by verify(). A difference is a miss: the step is taped again with every read real, and
    /// that address is never predicted from its address alone again. So the rules see the real bytes or the step is
    /// redone: the answer bytes cannot change. Only the taped re-run's own rule call predicts (table_scope, set by the
    /// taping loop around self.rule), never the walk-back.
    type TabKey = (u32, u64, u32, u8, u64, usize);
    #[derive(Default)]
    struct Tab { scope: bool, n: u8, by_step: std::collections::HashMap<TabKey, Vec<u8>>,
                 by_addr: std::collections::HashMap<(u64, usize), (Vec<u8>, u32, bool)>,
                 pend: Vec<(usize, Vec<u8>, TabKey)>, predicted: u64, real: u64, misses: u64, miss_reads: u64, bytes: u64 }
    thread_local! { static TAB: std::cell::RefCell<Tab> = std::cell::RefCell::new(Tab::default()); }
    pub fn tables_on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| on() && std::env::var("AD_TAPE_PREDICT_TABLES").as_deref() != Ok("0"))
    }
    /// The taping loop's rule call begins (true) or ends (false).
    pub fn table_scope(on: bool) { if tables_on() { TAB.with(|t| t.borrow_mut().scope = on) } }
    fn tab_note(t: &mut Tab, k: TabKey, v: &[u8]) {
        t.by_step.insert(k, v.to_vec());
        let e = t.by_addr.entry((k.4, k.5)).or_insert_with(|| (Vec::new(), 0, false));
        if e.0 == v { e.1 = e.1.saturating_add(1); } else { e.0 = v.to_vec(); e.1 = 1; }
    }
    /// dev::download(cuda, src, bytes) for a rule's descriptor table: predicted (see above) or read for real.
    pub unsafe fn table(cuda: &Cuda, src: u64, bytes: usize) -> Result<Vec<u8>, String> {
        if bytes == 0 || !tables_on() { return dev::download(cuda, src, bytes); }
        let (step, h, occ, _) = with(|s| s.book.key(0));
        let nth = TAB.with(|t| { let mut t = t.borrow_mut(); let n = t.n; t.n = t.n.wrapping_add(1); n });
        let k: TabKey = (step, h, occ, nth, src, bytes);
        if TAB.with(|t| t.borrow().scope) && active() {
            // measured (a test machine, 12 steps): the address-only guess missed at every step of a first gradient (11
            // of 11 steps taped twice), so it is used only with AD_TAPE_PREDICT_TABLES=2; the default takes the same
            // step of the gradient before only (0 misses in the later gradient).
            static ADDR: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let addr = *ADDR.get_or_init(|| std::env::var("AD_TAPE_PREDICT_TABLES").as_deref() == Ok("2"));
            let p = TAB.with(|t| { let t = t.borrow(); t.by_step.get(&k).cloned().or_else(|| if !addr { None } else {
                t.by_addr.get(&(src, bytes)).filter(|(_, agree, bad)| *agree >= 2 && !*bad).map(|(v, _, _)| v.clone()) }) });
            if let Some(p) = p {
                if let Some((at, off)) = room(cuda, bytes.div_ceil(8) * 8)? {
                    dev::copy(cuda, at, src, bytes)?;
                    TAB.with(|t| { let mut t = t.borrow_mut(); t.predicted += 1; t.bytes += bytes as u64; t.pend.push((off, p.clone(), k)); });
                    return Ok(p);
                }
            }
        }
        let v = dev::download(cuda, src, bytes)?;
        TAB.with(|t| { let mut t = t.borrow_mut(); t.real += 1; tab_note(&mut t, k, &v); });
        Ok(v)
    }
    struct State { book: Book, log: u64, cap: usize, used: usize, verify_s: f64, no_room: u64, force_real: bool, alloc_failed: bool, file: Option<String> }
    thread_local! { static ST: std::cell::RefCell<Option<State>> = const { std::cell::RefCell::new(None) }; }
    pub fn on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("AD_TAPE_PREDICT").as_deref() == Ok("1"))
    }
    fn with<R>(f: impl FnOnce(&mut State) -> R) -> R {
        ST.with(|c| {
            let mut g = c.borrow_mut();
            let s = g.get_or_insert_with(|| {
                let need = std::env::var("AD_TAPE_PREDICT_AGREE").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(1);
                let sites = tape_predict::sites_mask(std::env::var("AD_TAPE_PREDICT_SITES").ok().as_deref());
                let mut book = Book::new(need, sites);
                let file = std::env::var("AD_TAPE_PREDICT_BOOK").ok().filter(|f| !f.is_empty());
                let loaded = match file.as_deref().map(std::fs::read) {
                    Some(Ok(b)) => book.load_bytes(&b).map(|n| n as i64).unwrap_or_else(|e| { eprintln!("{}", json!({"event":"tape_predict_book_refused","why":e})); -1 }),
                    _ => 0,
                };
                eprintln!("{}", json!({"event":"tape_predict_on","book":file,"keys_loaded":loaded,"agree":need,"sites":sites}));
                State { book, log: 0, cap: 0, used: 0, verify_s: 0.0, no_room: 0, force_real: false, alloc_failed: false, file }
            });
            f(s)
        })
    }
    /// A step's taping attempt begins; `active` false reads every decision for real (the first step, a redo).
    pub fn begin(step: usize, active: bool) { with(|s| { let a = active && !s.force_real; s.book.begin(step as u32, a); s.used = 0; });
        TAB.with(|t| { let mut t = t.borrow_mut(); t.n = 0; t.pend.clear(); }) }
    /// The redo of a missed step: its attempt reads every decision for real (set by reverse_step around the redo).
    pub fn force_real(on: bool) { with(|s| s.force_real = on) }
    pub fn active() -> bool { on() && with(|s| s.book.active) }
    /// Read the rest of this step's decisions for real (after a chunk that went to host memory).
    pub fn deactivate() { with(|s| s.book.active = false) }
    pub fn set_cmd(name: &str) { with(|s| s.book.set_cmd(name)); TAB.with(|t| t.borrow_mut().n = 0) }
    /// Device log room for `bytes` (allocated at the first predicted read; AD_TAPE_PREDICT_LOG_MIB, default 64). None:
    /// the log is full, and the read is made for real.
    unsafe fn room(cuda: &Cuda, bytes: usize) -> Result<Option<(u64, usize)>, String> {
        let (log, cap) = with(|s| (s.log, s.cap));
        if log == 0 && !with(|s| s.alloc_failed) {
            let cap = std::env::var("AD_TAPE_PREDICT_LOG_MIB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(64) << 20;
            let mut a = 0u64;
            // rerun-tape (checker pass): the log is allocated after the tape was sized to the card's free memory.
            // THE BREAKAGE THIS PREVENTS: a card left with less than the log's room stopped the whole gradient at its first
            // predicted read (checked() returned the out-of-memory error). Now no log means every decision is read for
            // real, as with the option off; the receipt says so (log_full counts the reads, log_alloc_failed is true).
            if checked("cuMemAlloc tape predict log", (cuda.allocate)(&mut a, cap)).is_err() {
                eprintln!("{}", json!({"event":"tape_predict_log_refused","bytes":cap}));
                with(|s| s.alloc_failed = true);
            } else {
                with(|s| { s.log = a; s.cap = cap; });
            }
        }
        if with(|s| s.alloc_failed) { with(|s| s.no_room += 1); return Ok(None); }
        Ok(with(|s| if s.used + bytes <= s.cap { let off = s.used; s.used += bytes; Some((s.log + off as u64, off)) } else { s.no_room += 1; None }))
    }
    /// Support::collect(cuda, 0, n) for a share decision at `site`: predicted (the slots' words are copied into the log
    /// and the decisions come from the book; only word 0 of each slot, zero = equal, is set) or read for real.
    pub unsafe fn words(cuda: &Cuda, sup: &dev::Support, site: u8, n: usize) -> Result<Vec<[u64; 3]>, String> {
        if let Some(p) = with(|s| s.book.predict(site, n)) {
            if let Some((at, off)) = room(cuda, tape_predict::log_bytes(n, false))? {
                dev::copy(cuda, at, sup.counts_addr(), 24 * n)?;
                let w = p.iter().map(|&eq| [if eq { 0 } else { 1 }, 0, 0]).collect();
                with(|s| s.book.queue(site, off, n, false, p));
                return Ok(w);
            }
        }
        let (_, w) = sup.collect(cuda, 0, n)?;
        with(|s| s.book.record_real(site, w.iter().map(|x| x[0] == 0).collect()));
        Ok(w)
    }
    /// (on with AD_TAPE_PREDICT=1; AD_TAPE_PREDICT_ROLL=0 turns it off, the exact route does not set AD_TAPE_PREDICT): in the taped
    /// re-run, the rolling-copy block compares of each MYNN-EDMF launch (schemes.rs prepare_impl, 58 a 3 km step, each one
    /// upload, launch and blocking readback: 0.060 s a step in rr2-Bp) are queued as ONE list a launch and answered from the
    /// book (the same launch of the same step of the gradient before); the real answer is copied into the predict log and
    /// checked at the chunk end, and a miss re-tapes the step with every read real, so the tape and the answer cannot change.
    /// With no prediction the one list is read for real (one readback a launch instead of one a segment). The launch's two
    /// card drains (after its prepare, before its forward) go too in an ordered plan (AD_TAPE_ROLL_DRAIN=1 keeps them).
    pub fn roll_on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| on() && std::env::var("AD_TAPE_PREDICT_ROLL").as_deref() != Ok("0"))
    }
    pub fn roll_drain() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("AD_TAPE_ROLL_DRAIN").as_deref() == Ok("1"))
    }
    /// pairs() at the rolling-copy site.
    pub unsafe fn roll_pairs(cuda: &Cuda, sup: &dev::Support, n: usize) -> Result<Vec<bool>, String> {
        pairs(cuda, sup, tape_predict::SITE_ROLL_PAIRS, n)
    }
    /// Support::pairs_differ_read(cuda, n) for the block compares at `site`: predicted or read for real.
    pub unsafe fn pairs(cuda: &Cuda, sup: &dev::Support, site: u8, n: usize) -> Result<Vec<bool>, String> {
        if let Some(p) = with(|s| s.book.predict(site, n)) {
            if let Some((at, off)) = room(cuda, tape_predict::log_bytes(n, true))? {
                dev::copy(cuda, at, sup.pairs_result_addr(), 4 * n)?;
                with(|s| s.book.queue(site, off, n, true, p.clone()));
                return Ok(p);
            }
        }
        let d = sup.pairs_differ_read(cuda, n)?;
        with(|s| s.book.record_real(site, d.clone()));
        Ok(d)
    }
    /// One readback of the log (it waits for everything queued): the keys whose predicted decisions did not hold.
    pub unsafe fn verify(cuda: &Cuda) -> Result<Vec<Key>, String> {
        let (log, used, any) = with(|s| (s.log, s.used, !s.book.pending.is_empty()));
        let any_tab = TAB.with(|t| !t.borrow().pend.is_empty());
        if !any && !any_tab { return Ok(Vec::new()); }
        let t = std::time::Instant::now();
        let raw = dev::download(cuda, log, used)?;
        let mut bad = with(|s| { s.used = 0; s.verify_s += t.elapsed().as_secs_f64(); s.book.verify(&raw) });
        TAB.with(|tb| {
            let mut tb = tb.borrow_mut();
            let mut missed = 0u64;
            for (off, p, k) in std::mem::take(&mut tb.pend) {
                let real = raw[off..off + p.len()].to_vec();
                if real != p {
                    missed += 1;
                    bad.push((k.0, k.1, k.2, 200));
                    tb.by_addr.entry((k.4, k.5)).or_insert_with(|| (Vec::new(), 0, true)).2 = true;
                }
                tab_note(&mut tb, k, &real);
            }
            if missed > 0 { tb.misses += 1; tb.miss_reads += missed; }
        });
        Ok(bad)
    }
    /// The attempt is over; `held` keeps its decisions in the book for this step. After step 1 (the last step a sweep
    /// tapes) the book is written to AD_TAPE_PREDICT_BOOK (a temporary file, then renamed), for the next gradient.
    pub fn end_step(step: usize, held: bool) -> Result<(), String> {
        TAB.with(|t| t.borrow_mut().pend.clear());
        let out = with(|s| { s.book.end_step(held); (held && step <= 1).then(|| s.file.clone()).flatten().map(|f| (f, s.book.to_bytes())) });
        if let Some((f, bytes)) = out {
            let tmp = format!("{f}.tmp{}", std::process::id());
            std::fs::write(&tmp, &bytes).and_then(|_| std::fs::rename(&tmp, &f)).map_err(|e| format!("tape predict book {f}: {e}"))?;
            eprintln!("{}", json!({"event":"tape_predict_book_written","book":f,"bytes":bytes.len()}));
        }
        Ok(())
    }
    /// Counters for the receipts (cumulative over the run).
    pub fn receipt() -> Value {
        with(|s| { let t = &s.book.stats; json!({"predicted_reads":t.predicted_reads,"predicted_decisions":t.predicted_decisions,"real_reads":t.real_reads,
            "misses":t.misses,"miss_keys":t.miss_keys,"steps_held":t.steps_held,"log_bytes":t.log_bytes,"verify_s":s.verify_s,"log_full":s.no_room,"log_alloc_failed":s.alloc_failed,
            "book_keys":s.book.keys(),"agree":s.book.need,
            "tables":TAB.with(|t| { let t = t.borrow(); json!({"on":tables_on(),"predicted":t.predicted,"real":t.real,"misses":t.misses,"miss_reads":t.miss_reads,"bytes":t.bytes,"keys":t.by_step.len()}) })}) })
    }
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
        let Some(pieces) = self.block_plan(alloc, lo, hi, bs) else { return Ok(None) };
        let pairs: Vec<(u64, u64, usize)> = pieces.iter().filter_map(|p| p.2.map(|s| (live + p.0 as u64, s, p.1))).collect();
        let differ = sup.pairs_differ(cuda, &pairs)?;
        Ok(self.block_finish(pieces, &differ, lo, hi, bs))
    }
    /// bs-tape (a12 backstep): block_runs' pieces of [lo, hi) of `alloc` with the source each would be compared with (None:
    /// no earlier copy holds the block), or None when block_runs would return None before comparing anything.
    pub fn block_plan(&self, alloc: u64, lo: usize, hi: usize, bs: usize) -> Option<Vec<(usize, usize, Option<u64>)>> {
        if bs == 0 || hi <= lo || (hi - lo) % 4 != 0 || lo % 4 != 0 { return None; }
        let mut pieces: Vec<(usize, usize, Option<u64>)> = Vec::new();
        let mut o = lo;
        while o < hi {
            let e = ((o / bs + 1) * bs).min(hi);
            pieces.push((o, e - o, self.block_source(alloc, o, e)));
            o = e;
        }
        if pieces.iter().all(|p| p.2.is_none()) { return None; }
        Some(pieces)
    }
    /// bs-tape: block_runs from its pieces and the compares of their sourced pieces (in order), with the same counters.
    pub fn block_finish(&mut self, mut pieces: Vec<(usize, usize, Option<u64>)>, differ: &[bool], lo: usize, hi: usize, bs: usize)
        -> Option<Vec<(usize, usize, Option<u64>)>> {
        let cand: Vec<usize> = (0..pieces.len()).filter(|&k| pieces[k].2.is_some()).collect();
        let differ = differ.to_vec();
        self.blk_checks += cand.len();
        for (&k, dif) in cand.iter().zip(differ) { if dif { pieces[k].2 = None; } }
        let equal: usize = pieces.iter().filter(|p| p.2.is_some()).map(|p| p.1).sum();
        if equal < bs.min((hi - lo) / 16).max(4096) { return None; }
        let mut runs: Vec<(usize, usize, Option<u64>)> = Vec::new();
        for (po, pl, src) in pieces {
            match (runs.last_mut(), src) {
                (Some((ro, rl, Some(rs))), Some(s)) if *ro + *rl == po && *rs + *rl as u64 == s => *rl += pl,
                (Some((ro, rl, None)), None) if *ro + *rl == po => *rl += pl,
                _ => runs.push((po, pl, src)),
            }
        }
        self.blk_spans += 1; self.blk_hit_bytes += equal; self.blk_fresh_bytes += (hi - lo) - equal;
        Some(runs)
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
pub struct Roles { pub tl_snapshot: Vec<String>, pub ad_snapshot: Vec<String>, pub ad_clear: Vec<String>, pub ad_replace: Vec<String>, pub ad_post: Vec<String>,
    /// S01b: in-plus-inout arguments whose `inout` binding is the live cotangent (not the seed snapshot).
    pub ad_live_inout: Vec<String> }

impl KernelDeriv {
    fn position(&self, name: &str) -> Result<usize, String> {
        self.names.iter().position(|n| n == name).ok_or_else(|| format!("{}: no launch argument {name}", self.function))
    }
    /// The launch's own scalar arguments only (no geometry, no captured or derived extents).
    pub fn scalars(&self, l: &LaunchInfo) -> Env {
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
        // S01b (restored; a regional merge had dropped it): a by-value CArray argument (a
        // binding with "carray": true) puts its element count (the struct's second word) in the environment as
        // size_<name>, so a workspace or a bound can follow the launch's real array (g3 cupy_scatter_update's claims: an
        // `a` with a leading dimension is L*adim*rdim words, not adim*rdim). THE BREAKAGE THIS PREVENTS: the 3 km LAM
        // window on the ad/final-merged binder stopped in tangent step 1 with "missing symbol size_a" (a test machine lam4-b,
        // 2026-11-09 20:06Z): the LAM kernel set's cupy_scatter_update spec sizes its claims workspace as 4*size_a.
        for b in self.spec["launch_abi"]["entries"].as_object().into_iter().flat_map(|e| e.values()).flat_map(|e| e["args"].as_array().into_iter().flatten()) {
            if b["carray"] != true { continue; }
            let Some(name) = b["arg"].as_str() else { continue };
            let Some(i) = self.names.iter().position(|n| n == name) else { continue };
            let bytes = match l.args.get(i) { Some(submit::Argument::Bytes(v)) => v.as_slice(), Some(submit::Argument::Patched { bytes, .. }) => bytes.as_slice(), _ => continue };
            if bytes.len() >= 16 { env.insert(format!("size_{name}"), i64::from_le_bytes(bytes[8..16].try_into().unwrap()) as f64); }
        }
        env
    }
    pub fn env(&self, l: &LaunchInfo) -> Result<Env, String> {
        let mut env = self.scalars(l);
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
    pub(crate) fn device_target(&self, l: &LaunchInfo, name: &str) -> Result<Option<(u64, usize)>, String> {
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
    ///
    /// F099 (swarm 2026-11-07): the fixed extent stands in only where the kernel-set row has no capture shape (empty or
    /// absent), which is the case above. Where the row carries the shape, the spec's numeric shape is that same capture
    /// extent (the dycore specs write `"w": [56, 2562]`, the x1.2562 capture), and it is re-expressed through the launch
    /// scalars by elements_from_shapes like any captured shape. THE BREAKAGE THIS PREVENTS: flooring those at the
    /// capture product (c519d229) made every launch with fewer cells than x1.2562 (per-rank meshes, small LAMs) bind a
    /// 2562-cell span, so view() refused "view of w exceeds its allocation" (w_vertical_flux_f32 at ncells 400: 143,472
    /// words asked of a 22,400-word field). Unchanged on the capture mesh (the product equals the row extent there).
    fn elements(&self, name: &str, env: &Env) -> Result<usize, String> {
        let decl = self.spec["args"].as_array().and_then(|a| a.iter().find(|d| d["name"] == name));
        let fixed = decl.and_then(|d| d["shape"].as_array()).filter(|a| !a.is_empty() && a.iter().all(|x| x.is_u64()))
            .map(|a| a.iter().map(|x| x.as_u64().unwrap() as usize).product::<usize>());
        let row_shapeless = self.position(name).ok()
            .map_or(true, |i| self.meta[i]["shape"].as_array().map_or(true, |a| a.is_empty()));
        match fixed {
            Some(m) if row_shapeless => Ok(m),
            _ => self.elements_from_shapes(name, env),
        }
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
        // S01b: an adjoint `inout` binding whose spec says it starts from the incoming cotangent ("initialization":
        // "Copy incoming ... cotangent before the adjoint", the g43 regional rows) keeps that cotangent: the seed is
        // still snapshotted for the `in` binding, but the view is not cleared, because the entry zeroes only the
        // words the forward overwrote and passes the others through. THE BREAKAGE THIS PREVENTS:
        // regional_bdy_adjust_scalars_compute_v841 (partial overwrite of `updates`, relaxation and specified cells
        // only) lost every pass-through cotangent: per-launch dots FAIL rel 0.26-0.95 on the 3 km LAM plan.
        let keep: BTreeSet<String> = self.ad_order.iter().flat_map(|e| self.bindings(e)).filter(|b| b["bind"] == "adjoint" && b["io"] == "inout"
            && b["initialization"].as_str().is_some_and(|s| s.starts_with("Copy incoming"))).filter_map(|b| b["arg"].as_str().map(str::to_owned)).collect();
        let ad_clear = ad_snapshot.iter().filter(|a| !keep.contains(*a)).cloned().collect();
        // ...and its `inout` binding is that live cotangent (the `in` binding alone reads the seed snapshot), so the
        // entry's zeroing of the overwritten words never races its own reads of the seed (NONDETERMINISTIC reverse,
        // 31,870 differing words per launch, when both pointers were the snapshot).
        let ad_live_inout = ad_snapshot.iter().filter(|a| keep.contains(*a)).cloned().collect();
        Roles { tl_snapshot, ad_clear, ad_snapshot, ad_replace, ad_post, ad_live_inout }
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
            if let Ok((a, bytes)) = self.owned_view(b, l, name) { dev::zero(cuda, a, bytes)?; n += 1; }
        }
        Ok(n)
    }
    /// D1 (--refuse holds its kind, wrules::hold_refused): the forward arguments whose tangent this row's tangent
    /// entries write at this launch, as (name, allocation, byte offset, elements), from the spec and the launch alone
    /// (nothing is bound). The same views zero_written clears; an argument that is not a device pointer at this
    /// launch (a null optional output) writes nothing and is left out. The count is the shadow words the argument
    /// owns (owned_words), not its elements: a u8 output owns one word per 4 elements.
    pub fn written_targets(&self, l: &LaunchInfo) -> Result<Vec<(String, u64, usize, usize)>, String> {
        let env = self.env(l)?;
        let mut out = Vec::new();
        for name in tangent_written_names(&self.spec, &self.tl_order) {
            let Some((alloc, off)) = self.device_target(l, &name)? else { continue };
            let n = owned_words(self.elements(&name, &env)?, self.dtype_of(&name));
            out.push((name, alloc, off, n));
        }
        Ok(out)
    }
    pub(crate) fn view(&self, b: &Bound, l: &LaunchInfo, name: &str) -> Result<(u64, usize), String> {
        self.view_words(b, l, name, self.elements(name, &b.env)?)
    }
    /// The declared element type of a forward argument (spec `args[].dtype`: "f32", "u8", ...; "" when not declared).
    pub(crate) fn dtype_of(&self, name: &str) -> &str {
        self.spec["args"].as_array().and_then(|a| a.iter().find(|d| d["name"] == name)).and_then(|d| d["dtype"].as_str()).unwrap_or("")
    }
    /// switchoff (g2, 2026-11-11): the part of an argument's shadow view that a ZERO rule may clear (AD_ZERO_DERIV,
    /// the --refuse hold): the shadow words the argument owns (owned_words), not one word per element.
    pub(crate) fn owned_view(&self, b: &Bound, l: &LaunchInfo, name: &str) -> Result<(u64, usize), String> {
        self.view_words(b, l, name, owned_words(self.elements(name, &b.env)?, self.dtype_of(name)))
    }
    fn view_words(&self, b: &Bound, l: &LaunchInfo, name: &str, words: usize) -> Result<(u64, usize), String> {
        let (alloc, offset) = self.device_target(l, name)?.ok_or_else(|| format!("{name} is not a device argument"))?;
        let s = b.shadows.get(&alloc).ok_or("shadow missing")?;
        let start = offset.checked_sub(s.lo).ok_or("view below shadow span")? / 4 * s.width;
        let bytes = words * s.width;
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

        let tp_spans = dev::prof::start();
        // span memo (see SpanMemo): a launch equal to one already walked takes the walk's results
        let memo_key = if span_memo_on() { Some((self as *const Self as usize, span_key(l, width, defer, persistent.is_some()))) } else { None };
        let memo_hit: Option<std::rc::Rc<SpanMemo>> = memo_key.and_then(|k| SPAN_MEMO.with(|m| m.borrow().get(&k).cloned())).filter(|m| m.consult.iter().all(|&(al, kind, v)| match kind {
            0 => exec_alloc.get(&al).map(|a| a.bytes) == Some(v),
            _ => persistent.and_then(|p| p.get(&al)).is_some_and(|s| s.width == 4) as usize == v,
        }));
        let mut consult: Vec<(u64, u8, usize)> = Vec::new();
        let mut env = match &memo_hit { Some(m) => m.env.clone(), None => self.env(l)? };
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
        if let Some(m) = self.spec["launch_abi"]["mesh_extents"].as_object().filter(|_| memo_hit.is_none()) {
            for (sym, arg) in m {
                if self.meta.iter().zip(&self.names).any(|(x, n)| x["kind"] == "scalar" && n == sym) { continue; }
                let name = arg.as_str().ok_or_else(|| format!("{}: mesh_extents {sym} names no argument", self.function))?;
                let (alloc, offset) = self.device_target(l, name)?
                    .ok_or_else(|| format!("{}: mesh extent {sym}: {name} is not a device argument", self.function))?;
                if offset != 0 {
                    return Err(format!("{}: mesh extent {sym}: {name} starts at byte {offset} of allocation {alloc}, not a whole mesh array", self.function));
                }
                let a = exec_alloc.get(&alloc).ok_or_else(|| format!("{}: mesh extent {sym}: unknown allocation {alloc}", self.function))?;
                consult.push((alloc, 0, a.bytes));
                env.insert(sym.clone(), (a.bytes / 4) as f64);
            }
        }
        let mut b = Bound { pre: HashMap::new(), shadows: BTreeMap::new(), tl_cells: BTreeMap::new(), ad_cells: BTreeMap::new(),
                            saved: HashMap::new(), seed_copies: Vec::new(), workspace: HashMap::new(), failures: Vec::new(), env,
                            snap_tl: BTreeMap::new(), snap_ad: BTreeMap::new(), clear_ad: Vec::new(), replace_ad: BTreeMap::new(), post_ad: BTreeMap::new(), phase_ad: false, live_inout: Vec::new(), mirror: HashMap::new(), widen: BTreeMap::new(), widen_io: BTreeMap::new(), widen_mirror: Vec::new(), primal_mirror: HashMap::new(),
                            parts: [0; 4], deferred: None, shared: Vec::new(), copies: Vec::new(), pre_arg: HashMap::new(), cluster_args: HashMap::new(), recon: Vec::new(), post_zero: HashMap::new() };
        let mut widths: BTreeMap<u64, usize> = BTreeMap::new();
        let mut extents: BTreeMap<u64, (usize, usize)> = BTreeMap::new();
        let mut pre_spans: BTreeMap<u64, (usize, usize)> = BTreeMap::new();
        let mut arg_spans: BTreeMap<u64, BTreeMap<String, (usize, usize)>> = BTreeMap::new();
        let mut indirect: Vec<String> = Vec::new();
        let mut primal_indirect: Vec<String> = Vec::new();
        let mut widen: Vec<String> = Vec::new();
        let mut widen_io: BTreeMap<String, (String, String)> = BTreeMap::new();
        let mut live_only: Vec<u64> = Vec::new();
        // tape-set: (argument, post?) bound to zero words, not copied (tape_finite)
        let mut zero_args: Vec<(String, bool)> = Vec::new();
        if let Some(m) = &memo_hit {
            for &al in &m.pre { b.pre.insert(al, 0); }
            b.tl_cells = m.tl_cells.clone(); b.ad_cells = m.ad_cells.clone();
            widths = m.widths.clone(); extents = m.extents.clone(); pre_spans = m.pre_spans.clone(); arg_spans = m.arg_spans.clone();
            indirect = m.indirect.clone(); primal_indirect = m.primal_indirect.clone(); widen = m.widen.clone(); widen_io = m.widen_io.clone();
            live_only = m.live_only.clone(); zero_args = m.zero_args.clone();
        }
        for entry in self.entries().into_iter().filter(|_| memo_hit.is_none()) {
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
                    // tape-set: an argument every reverse entry of this row reads only to refuse (or not at all) is bound
                    // to zero words below, not copied; one whose size is unknown keeps its copy
                    "primal" if defer && !bind["indirect"].is_object() && bind["arg"].as_str().is_some_and(|n|
                        tape_finite(&self.function, n, bind["post"] == true) && self.elements(n, &b.env).is_ok()) => {
                        let z = (bind["arg"].as_str().unwrap_or("").to_owned(), bind["post"] == true);
                        if !zero_args.contains(&z) { zero_args.push(z); }
                    },
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
                            if w == 8 { consult.push((alloc, 1, p.get(&alloc).is_some_and(|s| s.width == 4) as usize)); }
                            if w == 8 && p.get(&alloc).is_some_and(|s| s.width == 4) {
                                let io = bind["io"].as_str().unwrap_or("");
                                // Physics-on window L3: mpas_transport_source_baseline binds its float64 tangent `out` and
                                // adjoint `inout` (baseline) over the float32 sweep shadow; refused before (9 ERROR rows in
                                // stage A). Those forms now go through the float64 temporary as well (see widen_io).
                                // S01b: an adjoint `in` seed read is allowed when another entry of the launch binds the same argument `inout`
                                // (transport_standard_finish_regional_v841: f64 `output` seed read in ad/phase1-3, written in phase 4).
                                let inout_elsewhere = kind == "adjoint" && io == "in" && self.ad_order.iter().flat_map(|e| self.bindings(e))
                                    .any(|x| x["bind"] == "adjoint" && x["io"] == "inout" && x["arg"].as_str() == Some(name));
                                if !((kind == "tangent" && matches!(io, "in" | "out" | "inout")) || (kind == "adjoint" && matches!(io, "accumulate" | "inout")) || inout_elsewhere) {
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
        if let (Some(k), None) = (memo_key, &memo_hit) {
            let m = SpanMemo { env: b.env.clone(), pre: b.pre.keys().copied().collect(), tl_cells: b.tl_cells.clone(), ad_cells: b.ad_cells.clone(),
                widths: widths.clone(), extents: extents.clone(), pre_spans: pre_spans.clone(), arg_spans: arg_spans.clone(),
                indirect: indirect.clone(), primal_indirect: primal_indirect.clone(), widen: widen.clone(), widen_io: widen_io.clone(),
                live_only: live_only.clone(), zero_args: zero_args.clone(), consult };
            SPAN_MEMO.with(|c| { c.borrow_mut().insert(k, std::rc::Rc::new(m)); });
        }
        // Pre-launch primal: copy only the bound span (with a margin); the
        // binding address is (copy - span start) + offset, so offsets are unchanged.
        dev::prof::add(18, tp_spans);
        let tp_batch = dev::prof::start();
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
        // bs-tape (a12 backstep): each cluster's block compares (what block_runs would compare if the cluster misses) are
        // queued beside the cluster checks and read back after the same collect, instead of one more upload, launch and
        // blocking readback (a full drain of the card) per missed cluster: 1,482 a 3 km step. The pieces and their sources
        // are the ones block_runs picks in the loop below: a copy taken there for one cluster lies inside that cluster,
        // and clusters of one allocation are disjoint, so no piece of another cluster can take it as a source. Same
        // decisions, same copies, same words. AD_TAPE_BLOCK_EARLY=0 compares in the loop as before.
        let mut pre_blk: HashMap<(u64, usize), (Vec<(usize, usize, Option<u64>)>, Vec<bool>)> = HashMap::new();
        let blk_early = tape_block_bytes() > 0 && std::env::var("AD_TAPE_BLOCK_EARLY").as_deref() != Ok("0");
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
            // bs-tape: the block plans of every cluster that has a block source, and their compares in one list
            let mut plans: Vec<(u64, usize, Vec<(usize, usize, Option<u64>)>, usize)> = Vec::new();
            let mut pairs: Vec<(u64, u64, usize)> = Vec::new();
            if blk_early && nslot >= 2 && nslot <= dev::SLOTS {
                for (alloc, lo, hi, live, _) in q.iter() {
                    if let Some(p) = c.block_plan(*alloc, *lo, *hi, tape_block_bytes()) {
                        let k0 = pairs.len();
                        pairs.extend(p.iter().filter_map(|x| x.2.map(|s| (live + x.0 as u64, s, x.1))));
                        plans.push((*alloc, *lo, p, k0));
                    }
                }
                if pairs.len() > sup.pairs_per() { plans.clear(); pairs.clear(); }
            }
            // bs-tape: no candidate copy and no block source anywhere in this launch: nothing to compare, so no readback
            // (every cluster is decided "copy" exactly as decide() would decide it from two empty candidates)
            let nothing = pairs.is_empty() && q.iter().all(|(_, _, _, _, (s, w))| s.is_none() && w.is_none());
            if nothing && nslot <= dev::SLOTS && std::env::var("AD_TAPE_SKIP_EMPTY").as_deref() != Ok("0") {
                for (alloc, lo, _, _, cands) in q.into_iter() { pre_eq.insert((alloc, lo), (cands, false, false)); }
            // rerun-tape (AD_TAPE_PREDICT=1): a launch with one cluster and no block compares is batched too (its
            // two candidate compares queued at once and decided by decide(), exactly as lookup() decides them), so its
            // decision can be predicted instead of read in lookup(). The extra compare writes only a word slot.
            } else if (nslot > 2 || !pairs.is_empty() || (nslot >= 2 && predict::on())) && nslot <= dev::SLOTS {
                sup.reset_words(cuda)?;
                for (j, (_, lo, hi, live, (s, w))) in q.iter().enumerate() {
                    if let Some((clo, copy)) = s { sup.words_into(cuda, 2 * j, live + *lo as u64, copy + (lo - clo) as u64, hi - lo)?; }
                    if let Some((clo, copy)) = w { sup.words_into(cuda, 2 * j + 1, live + *lo as u64, copy + (lo - clo) as u64, hi - lo)?; }
                }
                if !pairs.is_empty() { sup.pairs_differ_queue(cuda, &pairs)?; }
                let wv = if predict::on() { predict::words(cuda, sup, tape_predict::SITE_PRE_WORDS, nslot)? } else { sup.collect(cuda, 0, nslot)?.1 };
                if !pairs.is_empty() {
                    let differ = if predict::on() { predict::pairs(cuda, sup, tape_predict::SITE_PRE_PAIRS, pairs.len())? } else { sup.pairs_differ_read(cuda, pairs.len())? };
                    for (alloc, lo, p, k0) in plans {
                        let n = p.iter().filter(|x| x.2.is_some()).count();
                        pre_blk.insert((alloc, lo), (p, differ[k0..k0 + n].to_vec()));
                    }
                }
                for (j, (alloc, lo, _, _, cands)) in q.into_iter().enumerate() { pre_eq.insert((alloc, lo), (cands, wv[2 * j][0] == 0, wv[2 * j + 1][0] == 0)); }
            }
        }
        dev::prof::add(19, tp_batch);
        let tp_loop = dev::prof::start();
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
                    } else if let Some(runs) = match pre_blk.remove(&(alloc, lo)) {
                        Some((p, d)) => c.block_finish(p, &d, lo, hi, tape_block_bytes()),
                        None => c.block_runs(cuda, sup, alloc, a.address, lo, hi, tape_block_bytes())?,
                    } {
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
        // tape-set: the zero-bound arguments read [offset, offset + view) of their allocation through base + offset, so
        // base = zero words - offset puts that view on zero words (tape_finite)
        for (name, post) in zero_args {
            let Some((_, offset)) = self.device_target(l, &name)? else { continue };
            let w = if matches!(self.dtype_of(&name), "f64" | "<f8" | "i64" | "<i8" | "u64" | "<u8") { 8 } else { 4 };
            let z = zero_words(cuda, self.elements(&name, &b.env)? * w + 256)?;
            if post { b.post_zero.insert(name, z.wrapping_sub(offset as u64)); } else { b.pre_arg.insert(name, z.wrapping_sub(offset as u64)); }
        }
        b.parts[0] = arena.used - u_pre;
        dev::prof::add(20, tp_loop);
        let tp_meta = dev::prof::start();
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
        dev::prof::add(21, tp_meta);
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
                // tape-set: bound to zero words by prepare_tape (the same test), so it needs no post copy unless the
                // walk-back also reads its allocation through the live map
                if tape_finite(&self.function, name, true) && !bind["indirect"].is_object() && self.elements(name, env).is_ok() {
                    if let Some(k) = self.names.iter().position(|n| n == name) {
                        if let Some(submit::Argument::Device(s)) = l.args.get(k) { if !keep.contains(&s.allocation) { continue; } }
                    }
                }
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
        b.live_inout = roles.ad_live_inout;
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
    fn workspace_needs_zero(&self, name: &str) -> bool {
        !(self.elide_column_workspace_zero && matches!(name, "nodes" | "maps" | "controls" | "partial"))
    }

    pub unsafe fn zero_workspaces(&self, cuda: &Cuda, b: &mut Bound) -> Result<(), String> {
        for (name, (a, n)) in &b.workspace {
            if self.workspace_needs_zero(name) { dev::zero(cuda, *a, *n)?; }
        }
        Ok(())
    }

    fn derived_key(bind: &Value, b: &Bound) -> Result<String, String> {
        match bind["sha256"].as_str() {
            Some(h) => Ok(h.to_owned()),
            None if bind["recipe"].as_str().is_some_and(|r| r.contains("glue.interface_shapes.rs")) =>
                Ok(format!("interface_shapes:{}:{}x{}", bind["name"].as_str().unwrap_or(""), b.env.get("nlev").copied().unwrap_or(0.0), b.env.get("ncells").copied().unwrap_or(0.0))),
            None => Err("derived sha256".into()),
        }
    }

    /// Materialize the original binder's derived topology tables before lowering.
    /// This is an explicit host boundary, counted separately from graph replay.
    /// Its recipe/hash checks and cache keys are identical to eager dispatch.
    pub unsafe fn prepare_graph_tables(&self, cuda: &Cuda, b: &Bound, l: &LaunchInfo,
                                       live: &HashMap<u64, submit::Allocation>) -> Result<(usize, usize), String> {
        let (mut tables, mut total_bytes) = (0, 0);
        for entry in &self.ad_order {
            for bind in self.spec["launch_abi"]["entries"][entry]["args"].as_array().ok_or("entry args")? {
                if bind["bind"] != "derived" { continue; }
                // partition: under --local-topology 1 a topology table is the launch's own rank-local mesh's,
                // keyed by its sources (local_resolve), exactly as eager dispatch keys it. THE BREAKAGE THIS PREVENTS:
                // keyed by the pinned SHA (derived_key), graph preparation of an N-rank plan built ONE table (the first
                // rank's, or the global mesh's) and every other rank's launches read it.
                if local_topology_on() && bind["recipe"].as_str().is_some_and(crate::topo::topology_recipe) {
                    let (key, plan) = self.local_resolve(bind, l, &b.env)?;
                    if self.derived.borrow().contains_key(&key) { continue; }
                    let (p, bytes) = self.local_table(cuda, bind, l, &b.env, live, &key, plan)?;
                    tables += 1; total_bytes += bytes.len().max(4);
                    self.derived.borrow_mut().insert(key, (p, bytes));
                    continue;
                }
                let key = Self::derived_key(bind, b)?;
                if self.derived.borrow().contains_key(&key) { self.pinned_mesh_check(cuda, bind, l, &b.env, live, &key)?; continue; }
                let bytes = self.construct(cuda, bind, l, &b.env, live)?;
                pinned_mesh_checked(&self.function, &key);
                let mut p = 0;
                checked("cuMemAlloc graph derived preparation", (cuda.allocate)(&mut p, bytes.len().max(4)))?;
                dev::upload(cuda, p, &bytes)?;
                tables += 1; total_bytes += bytes.len().max(4);
                self.derived.borrow_mut().insert(key, (p, bytes));
            }
        }
        Ok((tables, total_bytes))
    }

    /// Lower the selected AD sequence before CUDA capture. The existing binder
    /// still resolves every scalar, alias, branch buffer and launch geometry.
    /// No command executes here. Scratch is suballocated from the already-bound
    /// arena; no device allocation or host readback is permitted by this path.
    /// A missing derived table is an explicit dependency, never an eager fallback.
    pub unsafe fn lower_graph_ad_entries(&self, cuda: &Cuda, arena: &mut Arena, b: &mut Bound,
                                 l: &LaunchInfo, live: &HashMap<u64, submit::Allocation>, entries: &[&str])
        -> Result<Vec<submit::Command>, String> {
        let mut commands = Vec::new();
        for entry in entries {
            self.launch_impl(cuda, arena, b, l, live, entry, Some(&mut commands))?;
            if commands.len() > 64 {
                return Err(format!("DEFECT graph {}: more than 64 nodes in the selected segment", self.function));
            }
        }
        Ok(commands)
    }

    /// Launch one entry with every argument bound; returns failure-slot sources that fired.
    pub unsafe fn launch(&self, cuda: &Cuda, arena: &mut Arena, b: &mut Bound, l: &LaunchInfo,
                         live: &HashMap<u64, submit::Allocation>, entry: &str) -> Result<Vec<String>, String> {
        self.launch_impl(cuda, arena, b, l, live, entry, None)
    }

    unsafe fn launch_impl(&self, cuda: &Cuda, arena: &mut Arena, b: &mut Bound, l: &LaunchInfo,
                         live: &HashMap<u64, submit::Allocation>, entry: &str,
                         mut lowered: Option<&mut Vec<submit::Command>>) -> Result<Vec<String>, String> {
        let graph_lowering = lowered.is_some();
        let (scratch_base, scratch_bytes) = (arena.base, arena.bytes);
        // The graph binds scratch as allocation 0; kernel pointers are already
        // resolved by the same binder as the eager path, so aliases stay exact.
        let mut zero = |address: u64, bytes: usize| -> Result<(), String> {
            if bytes == 0 { return Ok(()); }
            if let Some(commands) = lowered.as_deref_mut() {
                let offset = address.checked_sub(scratch_base).ok_or("graph clear below scratch")? as usize;
                if offset > scratch_bytes || bytes > scratch_bytes - offset { return Err("graph clear outside scratch".into()); }
                commands.push(submit::Command::Clear { destination: submit::Slice { allocation: 0, offset }, value: 0, bytes });
                Ok(())
            } else { dev::zero(cuda, address, bytes) }
        };
        let lp = lprof::on();
        let mut lt = std::time::Instant::now();
        let e = &self.spec["launch_abi"]["entries"][entry];
        let mut store: Vec<Vec<u8>> = Vec::new();
        let mut failure_slots = Vec::new();
        if lp { lprof::add(0, &mut lt); }
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
                        // S01b: value_from as an object {"operation":"element_count","shape_like_forward_arg":A} is the launch's
                        // element count of forward argument A (the regional route rows, g43: the reverse's mesh-array bound).
                        // THE BREAKAGE THIS PREVENTS: regional_relaxzone_filter_cell/edge_v841 and
                        // regional_bdy_adjust_scalars_compute_v841 refused every launch with "scalar value" (3 km LAM per-launch dots).
                        let count = bind["value_from"].as_object().filter(|o| o.get("operation").and_then(Value::as_str) == Some("element_count"))
                            .and_then(|o| o.get("shape_like_forward_arg").and_then(Value::as_str));
                        let v = if let Some(arg) = count { self.elements(arg, &b.env)? as f64 } else {
                            match bind["derived_from"].as_str().or(bind["value_from"].as_str()) { Some(e) => expr::eval(e, &b.env)?, None => bind["value"].as_f64().ok_or("scalar value")? } };
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
                            let base = if bind["post"] == true { b.post_zero.get(name).copied().unwrap_or_else(|| live[&alloc].address) }
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
                            ("adjoint", "inout") if b.phase_ad && b.live_inout.iter().any(|x| x == name) => None,
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
                        _ => { let a = arena.take(bytes)?;
                            if self.workspace_needs_zero(&name) { zero(a, bytes)?; }
                            b.workspace.insert(name, (a, bytes)); a },
                    };
                    a.to_le_bytes().to_vec()
                },
                "failure" => {
                    let slots = if bind["slots"].is_string() { expr::eval_usize(&bind["slots"], &b.env)? } else { bind["slots"].as_u64().unwrap_or(1) as usize }
                        .max(bind["minimum_slots"].as_u64().unwrap_or(0) as usize);
                    let per = bind["bytes_per_slot"].as_u64().unwrap_or(4) as usize;
                    let bytes = slots * per.max(4) * 2;
                    let a = arena.take(bytes)?;
                    zero(a, bytes)?;
                    let sources: Vec<String> = bind["source_ids"].as_array().map(|s| s.iter().map(|x| x.as_str().unwrap_or("").to_owned()).collect()).unwrap_or_default();
                    failure_slots.push((a, bytes, sources));
                    a.to_le_bytes().to_vec()
                },
                // bs-launch (AD_BRANCH_NULL=1): a branch record is a set of counters the entries add to with atomics on a
                // few words (fct_copy_back_v841__ad: three adds a thread over 29 M threads, 51 ms against 0.16 ms for its
                // forward; vector_momentum_v841_f32__ad re-runs its forward to count). Nothing reads them back here, so
                // they are bound to null: every entry tests the pointer before counting. A kernel that does not would
                // fault, never pass silently.
                "branch_record" if branch_null(&self.function) => 0u64.to_le_bytes().to_vec(),
                "branch_record" => {
                    let slots = bind["slots"].as_u64().unwrap_or(1) as usize;
                    let shaped = bind["shape"].as_array().map(|s| s.iter().map(|x| x.as_u64().unwrap_or(1) as usize).product()).unwrap_or(0);
                    let bytes = (slots * 2).max(shaped) * 4 * 2;
                    let a = arena.take(bytes)?;
                    zero(a, bytes)?;
                    a.to_le_bytes().to_vec()
                },
                // multirank-2 (--local-topology 1): every topology derived binding is built from this launch's own
                // (rank-local) mesh, keyed by its sources, never taken from the store pinned to the global mesh.
                "derived" if local_topology_on() && bind["recipe"].as_str().is_some_and(crate::topo::topology_recipe) => {
                    let (key, plan) = self.local_resolve(bind, l, &b.env)?;
                    if !self.derived.borrow().contains_key(&key) {
                        let (p, bytes) = self.local_table(cuda, bind, l, &b.env, live, &key, plan)?;
                        self.derived.borrow_mut().insert(key.clone(), (p, bytes));
                    }
                    let store = self.derived.borrow();
                    let (p, bytes) = store.get(&key).ok_or_else(|| format!("derived {key} not loaded"))?;
                    let by_value = bind["by_value"] == true || bind["as"] == "scalar" || bind["kind"] == "scalar"
                        || (bind["by_value"].is_null() && bytes.len() <= 8);
                    if by_value { bytes.clone() } else { p.to_le_bytes().to_vec() }
                },
                "derived" => {
                    let key = if bind["sha256"].is_null() && bind["recipe"] == "local_edge_incidence" {
                            let from = Self::from_names(bind);
                            let src = from.first().ok_or("local_edge_incidence: no endpoints input")?;
                            let (alloc, off) = self.device_target(l, src)?.ok_or_else(|| format!("local_edge_incidence: {src} is not a device argument"))?;
                            format!("local_edge_incidence:{}:{alloc}:{off}:{}x{}", bind["name"].as_str().unwrap_or(""),
                                    b.env.get(from.get(1).map(String::as_str).unwrap_or("ncells")).copied().unwrap_or(0.0), b.env.get("nedges").copied().unwrap_or(0.0))
                        } else { Self::derived_key(bind, b)? };
                    let h = key.as_str();
                    if self.derived.borrow().contains_key(h) {
                        self.pinned_mesh_check(cuda, bind, l, &b.env, live, h)?;
                    } else {
                        if graph_lowering {
                            return Err(format!("DEFECT graph {}: derived table {h} is not resident before lowering", self.function));
                        }
                        let t = std::time::Instant::now();
                        let bytes = self.construct(cuda, bind, l, &b.env, live)?;
                        pinned_mesh_checked(&self.function, h);
                        if derived_trace() { eprintln!("{}", serde_json::json!({"event":"derived_bind","function":self.function,"path":"pinned","recipe":bind["recipe"],"hit":false,"bytes":bytes.len(),"ms":t.elapsed().as_secs_f64() * 1e3,"key":h})); }
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
            if lp { lprof::add(lprof::kind(kind), &mut lt); lprof::count(12); }
        }
        let geometry = &e["geometry"];
        let (grid, block, shared) = if geometry.is_object() {
            let axis = |key: &str, i: usize| -> Result<u32, String> {
                Ok(expr::eval_usize(&geometry[key][i], &b.env)? as u32)
            };
            ([axis("grid", 0)?, axis("grid", 1)?, axis("grid", 2)?], [axis("block", 0)?, axis("block", 1)?, axis("block", 2)?],
             if geometry["shared_bytes"].is_null() { 0 } else { expr::eval_usize(&geometry["shared_bytes"], &b.env)? as u32 })
        } else { (l.grid, l.block, l.shared) };
        geom_note(&self.function, entry, geometry, grid, block, l.grid, l.block);
        drop(zero);
        if lp { lprof::add(10, &mut lt); }
        let f = *self.functions.get(entry).ok_or("entry function not loaded")?;
        if let Some(commands) = lowered {
            commands.push(submit::Command::Kernel { function: f, grid, block, shared_bytes: shared,
                arguments: store.into_iter().map(submit::Argument::Bytes).collect() });
        } else {
            let mut params: Vec<*mut c_void> = store.iter_mut().map(|v| v.as_mut_ptr().cast()).collect();
            checked(&format!("{} {entry} launch", self.function), (cuda.launch)(f, grid[0], grid[1], grid[2], block[0], block[1], block[2],
                shared, ptr::null_mut(), params.as_mut_ptr(), ptr::null_mut()))?;
        }
        if lp { lprof::add(11, &mut lt); }
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

/// D14: (function|table SHA256) whose stored topology table was found to be this run's mesh's (pinned_mesh_check).
static PINNED_MESH_CHECKED: std::sync::Mutex<BTreeSet<String>> = std::sync::Mutex::new(BTreeSet::new());
fn pinned_mesh_checked(function: &str, key: &str) {
    if let Ok(mut s) = PINNED_MESH_CHECKED.lock() { s.insert(format!("{function}|{key}")); }
}
/// D14: the mesh sizes where this launch (`env`) and the row's capture (`meta` value_hex) differ, as text.
fn mesh_differs(meta: &[Value], env: &Env) -> Vec<String> {
    let mut differ = Vec::new();
    for d in ["ncells", "nedges", "nvertices", "n_cells", "n_edges", "n_vertices"] {
        let Some(here) = env.get(d).copied() else { continue };
        let Some(captured) = meta.iter().find(|m| m["name"] == d).and_then(scalar_value) else { continue };
        if here != captured { differ.push(format!("{d} {captured} captured, {here} here")); }
    }
    differ
}
#[cfg(test)]
mod d14_pinned_mesh {
    use super::*;
    /// D14: the x1.2562 capture (ncells 2,562, nedges 7,680) against the 96 km globe (64,002 / 192,000) and itself.
    #[test]
    fn stored_table_mesh_compare() {
        let meta = vec![serde_json::json!({"name": "nlev", "kind": "scalar", "dtype": "<i4", "value_hex": "37000000"}),
                        serde_json::json!({"name": "ncells", "kind": "scalar", "dtype": "<i4", "value_hex": "020a0000"}),
                        serde_json::json!({"name": "nedges", "kind": "scalar", "dtype": "<i4", "value_hex": "001e0000"})];
        let mut env: Env = HashMap::new();
        env.insert("nlev".into(), 55.0); env.insert("ncells".into(), 2562.0); env.insert("nedges".into(), 7680.0);
        assert!(mesh_differs(&meta, &env).is_empty());
        env.insert("ncells".into(), 64002.0); env.insert("nedges".into(), 192000.0);
        assert_eq!(mesh_differs(&meta, &env), vec!["ncells 2562 captured, 64002 here".to_owned(), "nedges 7680 captured, 192000 here".to_owned()]);
        // nlev is not a mesh size: a different level count alone is not a different mesh.
        env.insert("ncells".into(), 2562.0); env.insert("nedges".into(), 7680.0); env.insert("nlev".into(), 59.0);
        assert!(mesh_differs(&meta, &env).is_empty());
    }
}

fn constructible(b: &Value) -> bool {
    let r = b["recipe"].as_str().unwrap_or("");
    // Physics-on window L3: the prep_interfaces capacities have no pinned SHA256 (they follow the launch's views);
    // the restated recipe's own checks are the admission (refused, never guessed).
    if r.contains("glue.interface_shapes.rs") { return true; }
    // F057 (multirank): the inverse incidence of the launch's OWN cells_on_edge (divergence_damping_selected_f32 adjoint).
    // No pinned SHA256: each rank of a decomposed forward has its own local mesh, and the pinned arrays of the
    // single-rank rows are the captured global mesh. Admission is edge_incidence's own checks (every endpoint in
    // [0, count), every row ascending, unique and consistent with cells_on_edge): refused, never guessed.
    if r == "local_edge_incidence" { return true; }
    // G6: derive_edge_incidence.py (pv_apvm_v841_f32) and u2:inverse_adv_cells (transport_edge_values) are restated
    // too. Before, both kernels were refused in every window (202 launches per 18 steps) and their tangent outputs kept
    // stale bytes (allocation 448, norm 1e53, injected into every tracer by transport_standard_finish).
    b["sha256"].is_string() && ["transpose_fixed", "transpose.rs#build", "tools/dycore_csr.rs", "edge_to_slots",
                                "inverse_topology.rs::theta", "inverse_topology.rs::momentum",
                                "derive_edge_incidence.py", "inverse_adv_cells"].iter().any(|k| r.contains(k))
}

/// F057: recipe local_edge_incidence, one output of the inverse incidence of `endpoints` ([nedges, 2], int32) over
/// `count` entities (recipes::edge_incidence: rows ascending and padded with -1): `n_*` the per-entity counts, `max_*`
/// the row width (one int32, passed by value), anything else the padded rows. Refused when an endpoint lies outside
/// [0, count): an entry would gather that cotangent from a cell it does not own.
pub fn local_edge_incidence(name: &str, endpoints: &[i32], count: usize) -> Result<Vec<u8>, String> {
    let (edges, counts, width) = crate::recipes::edge_incidence(endpoints, count)?;
    Ok(crate::recipes::pack(&if name.starts_with("n_") { counts } else if name.starts_with("max_") { vec![width] } else { edges }))
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
        mesh_read_memo(cuda, alloc, a.address, offset, n)
    }
    /// G6: an int32 device argument of a known element count (the recipe's own shape, not the capture row's).
    unsafe fn array_n(&self, cuda: &Cuda, l: &LaunchInfo, live: &HashMap<u64, submit::Allocation>, name: &str, n: usize) -> Result<Vec<i32>, String> {
        let (alloc, offset) = self.device_target(l, name)?.ok_or_else(|| format!("derived input {name} is not a device argument"))?;
        let a = live.get(&alloc).ok_or("derived input allocation")?;
        if offset + n * 4 > a.bytes { return Err(format!("derived input {name} exceeds its allocation")); }
        mesh_read_memo(cuda, alloc, a.address, offset, n)
    }
    /// hunt dyn-entry-gap: the identity of a restated recipe's array input (allocation, offset, length): with the scalar
    /// inputs, the memo key of the recipe's whole output set.
    fn input_id(&self, l: &LaunchInfo, env: &Env, live: &HashMap<u64, submit::Allocation>, name: &str) -> Result<String, String> {
        let (alloc, offset) = self.device_target(l, name)?.ok_or_else(|| format!("derived input {name} is not a device argument"))?;
        let a = live.get(&alloc).ok_or("derived input allocation")?;
        Ok(format!("{alloc}@{:x}+{offset}x{}", a.address, self.elements(name, env)?))
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
    /// D14 (MODEL-AUDIT 15 Nov): a STORED topology table (derived/<sha>.bin, loaded at deriv::load and keyed by its spec
    /// SHA256) is used only on the mesh it was pinned to. The launch's ncells / nedges / nvertices are compared with the
    /// row's captured values (kernels.json `value_hex`); when they differ, a recipe `restate` can build is rebuilt from
    /// this launch's own mesh arguments and must give the same SHA256, and any other is refused. Checked once per
    /// (function, table) a process; a table construct() just built from this launch is already checked.
    /// THE BREAKAGE THIS PREVENTS: a silent wrong-mesh adjoint. listB U8 u12-g96dot (96 km globe, 64,002 cells) took the
    /// x1.2562 (2,562-cell) tables: reconstruct_winds_v841_f32 FAIL rel 1.379, vertical_u_flux_f32 FAIL rel 1.425,
    /// pressure_gradient_v841_f32 NS-ONLY 2.6e-03, divergence_damping_f32 CUDA 700 (2,562-cell edges_on_cell on a
    /// 64,002-cell launch). Under --local-topology 1 / AD_DOT_LOCAL_TOPOLOGY=1 topology tables never come from the store.
    unsafe fn pinned_mesh_check(&self, cuda: &Cuda, bind: &Value, l: &LaunchInfo, env: &Env, live: &HashMap<u64, submit::Allocation>, key: &str) -> Result<(), String> {
        let recipe = bind["recipe"].as_str().unwrap_or("");
        if bind["sha256"].as_str() != Some(key) || !crate::topo::topology_recipe(recipe) { return Ok(()); }
        // AD_DERIVED_REPORT=1 loads no stored table: every table there is restated from this mesh (construct).
        if std::env::var("AD_DERIVED_REPORT").as_deref() == Ok("1") { return Ok(()); }
        let tag = format!("{}|{key}", self.function);
        if PINNED_MESH_CHECKED.lock().map_err(|_| "pinned-mesh table poisoned".to_string())?.contains(&tag) { return Ok(()); }
        let differ = mesh_differs(&self.meta, env);
        let name = bind["name"].as_str().or(bind["output"].as_str()).unwrap_or("?");
        if !differ.is_empty() {
            let restated = if constructible(bind) { Some(crate::exec::digest(&self.restate(cuda, bind, l, env, live)?)) } else { None };
            if restated.as_deref() != Some(key) {
                return Err(format!("{}: stored derived table {name} ({recipe}) is pinned to the capture mesh and this launch's mesh \
                    differs ({}){}. Breakage this prevents: a silent wrong-mesh adjoint (D14). Build the tables from the run's own \
                    mesh: window --local-topology 1, per-launch dot AD_DOT_LOCAL_TOPOLOGY=1 (its default), or re-pin the set with \
                    AD_DERIVED_REPORT=1", self.function, differ.join(", "),
                    match restated { Some(h) => format!("; restated from this launch it is {h}, not the stored {key}"), None => String::new() }));
            }
        }
        pinned_mesh_checked(&self.function, key);
        Ok(())
    }

    pub unsafe fn construct(&self, cuda: &Cuda, bind: &Value, l: &LaunchInfo, env: &Env, live: &HashMap<u64, submit::Allocation>) -> Result<Vec<u8>, String> {
        use crate::recipes as r;
        let recipe = bind["recipe"].as_str().unwrap_or("");
        if recipe.contains("glue.interface_shapes.rs") { return self.interface_shapes(bind, l, env, live); }
        if recipe == "local_edge_incidence" {
            let from = Self::from_names(bind);
            if from.len() != 2 { return Err(format!("{}: local_edge_incidence takes [endpoints, count], got {from:?}", self.function)); }
            let n = Self::scalar_named(env, &from[1])?;
            let ne = Self::scalar_named(env, "nedges").or_else(|_| Self::scalar_named(env, "n_edges"))?;
            let endpoints = self.array_n(cuda, l, live, &from[0], 2 * ne)?;
            let name = bind["name"].as_str().or(bind["output"].as_str()).unwrap_or("");
            return local_edge_incidence(name, &endpoints, n).map_err(|e| format!("{}: local_edge_incidence {name}: {e}", self.function));
        }
        let want = bind["sha256"].as_str().ok_or("derived without sha256")?;
        let name = bind["name"].as_str().or(bind["output"].as_str()).unwrap_or("");
        let bytes = self.restate(cuda, bind, l, env, live)?;
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
    /// The restated recipes that bind this launch's own arguments (construct's body before its SHA256 check).
    unsafe fn restate(&self, cuda: &Cuda, bind: &Value, l: &LaunchInfo, env: &Env, live: &HashMap<u64, submit::Allocation>) -> Result<Vec<u8>, String> {
        use crate::recipes as r;
        let recipe = bind["recipe"].as_str().unwrap_or("");
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
            let nedges = Self::scalar_named(env, "nedges").or_else(|_| Self::scalar_named(env, "n_edges"))?;
            let mk = format!("edge_to_slots|{}|{}|{nedges}", self.input_id(l, env, live, &f(0)?)?, self.input_id(l, env, live, &f(1)?)?);
            let out = recipe_memo(&mk, || {
                let cnt = self.array(cuda, l, env, live, &f(0)?)?; let edges = self.array(cuda, l, env, live, &f(1)?)?;
                let ncells = cnt.len();
                let (o, e) = r::edge_to_slots(ncells, nedges, edges.len() / ncells.max(1), &cnt, &edges)?;
                Ok(vec![o, e])
            })?;
            pick2((out[0].clone(), out[1].clone()))
        } else if recipe.contains("inverse_topology.rs::theta") && from.len() == 5 {
            // multirank-2: theta_finish / w_finish declare [ncells, nedges, max_edges, n_edges_on_cell, edges_on_cell]
            // for edge_offsets / edge_owners; theta's first output needs no cells_on_edge (the 6th input), which these
            // launches do not take. Their pinned arrays were only ever taken from the store ("missing input 5" when built).
            if !name.starts_with("edge_") { return Err(format!("{recipe}: {name} needs cells_on_edge (6 inputs), the row declares 5")); }
            let (nc, ne, w) = (Self::scalar_named(env, &f(0)?)?, Self::scalar_named(env, &f(1)?)?, Self::scalar_named(env, &f(2)?)?);
            let mk = format!("theta_edges|{nc}|{ne}|{w}|{}|{}", self.input_id(l, env, live, &f(3)?)?, self.input_id(l, env, live, &f(4)?)?);
            let out = recipe_memo(&mk, || {
                let cnt = self.array(cuda, l, env, live, &f(3)?)?; let inc = self.array(cuda, l, env, live, &f(4)?)?;
                let (o, e) = r::theta_edges(nc, ne, w, &cnt, &inc)?;
                Ok(vec![o, e])
            })?;
            pick2((out[0].clone(), out[1].clone()))
        } else if recipe.contains("inverse_topology.rs::theta") || recipe.contains("inverse_topology.rs::momentum") {
            let (nc, ne, w) = (Self::scalar_named(env, &f(0)?)?, Self::scalar_named(env, &f(1)?)?, Self::scalar_named(env, &f(2)?)?);
            let kind = if recipe.contains("theta") { "theta" } else { "momentum" };
            let mk = format!("{kind}|{nc}|{ne}|{w}|{}|{}|{}", self.input_id(l, env, live, &f(3)?)?, self.input_id(l, env, live, &f(4)?)?, self.input_id(l, env, live, &f(5)?)?);
            let out = recipe_memo(&mk, || {
                let cnt = self.array(cuda, l, env, live, &f(3)?)?; let inc = self.array(cuda, l, env, live, &f(4)?)?; let cells = self.array(cuda, l, env, live, &f(5)?)?;
                let ((a, b), (c, d)) = if kind == "theta" { r::theta(nc, ne, w, &cnt, &inc, &cells)? } else { r::momentum(nc, ne, w, &cnt, &inc, &cells)? };
                Ok(vec![a, b, c, d])
            })?;
            let (first, second) = ((out[0].clone(), out[1].clone()), (out[2].clone(), out[3].clone()));
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
        } else if recipe.contains("acreg:build_kset_acreg.py") || recipe.contains("s01b:make_lam_derived.py") {
            // storm3km: the gather tables of the four limited-area rows (acoustic_ru_regional, acoustic_rs_ts_regional,
            // transport_edge_values_regional, transport_standard_finish_regional), restated from THIS launch's own mesh
            // arguments, the way tree/ad/kset/acreg/build_kset_acreg.py and kset/s01b/make_lam_derived.py built them
            // from the 2,787-cell plan. THE BREAKAGE THIS PREVENTS: these 13 tables were pinned by the SHA256 of the
            // small test box's arrays and were not topology recipes, so on any other limited-area mesh (the 38,278-cell
            // storm box) --local-topology 1 left the small box's tables bound to the new mesh.
            let (nc, ne) = (Self::scalar_named(env, "ncells")?, Self::scalar_named(env, "nedges")?);
            let off = name.ends_with("_off");
            match name {
                "ar_ce_off" | "ar_ce_ent" | "s01b_ce_off" | "s01b_ce_ent" => {
                    // per cell, the edges that name it on either side, ascending, an edge twice when both sides name it;
                    // a side outside [0, ncells) names no cell (acoustic_ru_regional's cells_on_edge carries -1), exactly
                    // as build_kset_acreg.py and make_lam_derived.py cell_edge_csr() skip it.
                    let coe = self.array_n(cuda, l, live, "cells_on_edge", 2 * ne)?;
                    let mut rows = vec![Vec::<i32>::new(); nc];
                    for (e, ab) in coe.chunks_exact(2).enumerate() { for &t in ab { if t >= 0 && (t as usize) < nc { rows[t as usize].push(e as i32); } } }
                    let (mut o, mut x) = (vec![0i32], Vec::new());
                    for r in rows { x.extend(r); o.push(x.len() as i32); }
                    if off { o } else { x }
                }
                "ar_eo_off" | "ar_eo_cell" | "ar_th_off" | "ar_th_cell" => {
                    let w = Self::scalar_named(env, "max_edges")?;
                    let cnt = self.array_n(cuda, l, live, "n_edges_on_cell", nc)?; let inc = self.array_n(cuda, l, live, "edges_on_cell", nc * w)?;
                    let coe = self.array_n(cuda, l, live, "cells_on_edge", 2 * ne)?;
                    let (eo, th) = r::theta(nc, ne, w, &cnt, &inc, &coe)?;
                    let (o, x) = if name.starts_with("ar_eo") { eo } else { th };
                    if off { o } else { x }
                }
                "s01b_own_off" | "s01b_own_cell" | "s01b_own_slot" => {
                    let w = Self::scalar_named(env, "max_edges")?;
                    let cnt = self.array_n(cuda, l, live, "n_edges_on_cell", nc)?; let inc = self.array_n(cuda, l, live, "edges_on_cell", nc * w)?;
                    let (o, cs, ss) = crate::topo::wind_inverse_csr(nc, ne, w, &inc, &cnt)?;
                    if off { o } else if name.ends_with("_slot") { ss } else { cs }
                }
                "s01b_inv_off" | "s01b_inv_ent" => {
                    let w = Self::scalar_named(env, "width")?;
                    let n_adv = self.array_n(cuda, l, live, "n_adv", ne)?; let cells = self.array_n(cuda, l, live, "adv_cells", ne * w)?;
                    let (o, x) = r::inverse_adv_cells(nc, ne, w, &n_adv, &cells)?;
                    if off { o } else { x }
                }
                _ => return Err(format!("no restatement for {recipe} table {name}")),
            }
        } else { return Err(format!("no restatement for recipe {recipe}")); };
        Ok(r::pack(&values))
    }
    /// --local-topology 1: (allocation, offset) of every mesh-topology argument this launch passes.
    fn mesh_keys(&self, l: &LaunchInfo) -> Vec<(String, crate::topo::Key)> {
        self.names.iter().filter(|n| crate::topo::mesh_array(n).is_some())
            .filter_map(|n| self.device_target(l, n).ok().flatten().map(|k| (n.clone(), k))).collect()
    }
    /// --local-topology 1: what this launch tells the mesh registry: its mesh arguments with the row width the launch
    /// gives each (fixed, its width scalar, or its element count over its row scalar), and its ncells/nedges/nvertices.
    pub fn mesh_observation(&self, l: &LaunchInfo) -> crate::topo::Observation {
        use crate::topo::{mesh_array, Width, DIMS};
        let sc = self.scalars(l);
        let env = self.env(l).ok();
        let arrays = self.mesh_keys(l).into_iter().map(|(n, k)| {
            let (rows, w) = mesh_array(&n).unwrap();
            let width = match w {
                Width::Fixed(w) => Some(w),
                Width::Scalar(s) => s.iter().find_map(|x| sc.get(*x)).map(|v| *v as usize).or_else(|| {
                    let r = *sc.get(rows)? as usize;
                    let e = self.elements(&n, env.as_ref()?).ok()?;
                    (r > 0 && e % r == 0).then(|| e / r)
                }),
            };
            (n, k, width)
        }).collect();
        let dims = DIMS.iter().filter_map(|d| sc.get(*d).map(|v| (d.to_string(), *v as usize))).collect();
        crate::topo::Observation { function: self.function.clone(), arrays, dims }
    }
    /// --local-topology 1: the cache key of one topology derived binding at this launch (recipe, output, and the
    /// resolved sources and extents, so each rank's mesh gets its own array) and what building it reads.
    fn local_resolve(&self, bind: &Value, l: &LaunchInfo, env: &Env) -> Result<(String, LocalPlan), String> {
        let guard = LOCAL_TOPOLOGY.lock().unwrap();
        self.local_resolve_with(&guard.as_ref().ok_or("--local-topology: no mesh registry")?.registry, bind, l, env)
    }
    fn local_resolve_with(&self, reg: &crate::topo::Registry, bind: &Value, l: &LaunchInfo, env: &Env) -> Result<(String, LocalPlan), String> {
        use crate::topo::{mesh_array, DIMS};
        let recipe = bind["recipe"].as_str().unwrap_or("");
        let name = bind["name"].as_str().or(bind["output"].as_str()).unwrap_or("");
        let mut key = format!("local:{recipe}:{name}:{}:{}", bind["component"], bind["parameters"]);
        let Some(needs) = crate::topo::needs(recipe, name) else {
            // A recipe over this launch's own arguments: keyed by them and the launch's extents.
            for f in Self::from_names(bind) {
                match self.device_target(l, &f) { Ok(Some((a, o))) => key += &format!(":{f}@{a}+{o}"), _ => key += &format!(":{f}={:?}", env.get(&f)) }
            }
            // storm3km: a restated binding that names no inputs (the limited-area gather tables) reads this launch's
            // mesh arguments; they are part of its identity, so another mesh (another rank) gets its own table.
            if Self::from_names(bind).is_empty() { for (n, (a, o)) in self.mesh_keys(l) { key += &format!(":{n}@{a}+{o}"); } }
            for d in ["ncells", "nedges", "n_edges", "nvertices", "max_edges", "maxedges", "max_edges2", "width"] {
                if let Some(v) = env.get(d) { key += &format!(":{d}={v}"); }
            }
            return Ok((key, LocalPlan::Restate));
        };
        let own = self.mesh_keys(l);
        let keys: Vec<crate::topo::Key> = own.iter().map(|x| x.1).collect();
        let g = reg.group(&keys).map_err(|e| format!("{}: {recipe} {name}: {e}", self.function))?;
        let mut dims = BTreeMap::new();
        let mut dim_errors = Vec::new();
        for d in DIMS { match reg.dim(g, d) { Ok(v) => { dims.insert(d.to_owned(), v); }, Err(e) => dim_errors.push(e) } }
        let mut arrays = Vec::new();
        for (n, pick, required) in needs {
            let cand = match own.iter().find(|x| x.0 == n) {
                Some((_, k)) => vec![(*k, reg.width(g, n, *k)?)],
                None => reg.pick(g, n, pick).map_err(|e| format!("{}: {recipe} {name}: {e}", self.function))?,
            };
            if cand.is_empty() {
                if required { return Err(format!("{}: {recipe} {name} needs the rank's {n}, and no launch of its mesh group {g} passes one", self.function)); }
                continue;
            }
            let rd = mesh_array(n).unwrap().0;
            let rows = *dims.get(rd).ok_or_else(|| format!("{}: {recipe} {name}: {n} needs {rd}: {}", self.function, dim_errors.join("; ")))?;
            arrays.push((n.to_owned(), cand, rows));
        }
        // The f32-count / del4 sources: the rank-local extent must lie inside the argument the row names.
        let extent = (recipe.contains("transpose.rs#f32-count") || recipe.contains("del4_edge_count") || (recipe.contains("derive_launch_scalar.py") && name == "nedges"))
            .then(|| Self::from_names(bind).first().cloned()).flatten();
        key += &format!(":group{g}:{dims:?}:{arrays:?}");
        Ok((key, LocalPlan::Mesh { group: g, dims, dim_errors, arrays, extent }))
    }
    /// --local-topology 1: build one topology derived binding from the rank's mesh (no SHA256 refusal: the pinned
    /// value is the global mesh's); record whether it equals the pinned SHA256 (on one rank it must).
    /// hunt dyn-entry-gap: one device table per mesh table. A LocalPlan::Mesh table is a function of its key alone
    /// (recipe, output, component, parameters, the rank's mesh group, its dims and the mesh arrays it reads by
    /// allocation), never of the kernel that binds it, so every kernel and entry that binds the same key binds the same
    /// device copy (read only: a derived binding is an input). Before, each of the 75 dynamics rows built its own copy on
    /// its first launch, on the host, with the card idle: 1.4 s of host time in the first backward step and most of the
    /// first tangent step's extra at 120 km (nsys, hunt/dyn-entry-gap t2). The receipt still counts every binding row
    /// (a shared table is recorded against each function with its digest). AD_DERIVED_SHARED=0 builds per kernel (the old
    /// path). Restated tables (LocalPlan::Restate) read the launch's own arguments and stay per kernel.
    unsafe fn local_table(&self, cuda: &Cuda, bind: &Value, l: &LaunchInfo, env: &Env, live: &HashMap<u64, submit::Allocation>,
                          key: &str, plan: LocalPlan) -> Result<(u64, Vec<u8>), String> {
        let shared = matches!(plan, LocalPlan::Mesh { .. }) && derived_shared_on();
        let mesh = matches!(plan, LocalPlan::Mesh { .. });
        if shared {
            let hit = SHARED_DERIVED.lock().unwrap().get(key).cloned();
            if let Some((p, bytes, got)) = hit {
                self.local_receipt(bind, &got);
                DERIVED_STATS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if derived_trace() { eprintln!("{}", serde_json::json!({"event":"derived_bind","function":self.function,"path":"local","plan":"mesh","hit":true,"bytes":bytes.len(),"ms":0.0,"key":key})); }
                return Ok((p, bytes));
            }
        }
        let t = std::time::Instant::now();
        let (bytes, got) = self.local_build(cuda, bind, l, env, live, plan)?;
        let build_ms = t.elapsed().as_secs_f64() * 1e3;
        let mut p = 0;
        checked("cuMemAlloc derived", (cuda.allocate)(&mut p, bytes.len().max(4)))?;
        dev::upload(cuda, p, &bytes)?;
        if derived_trace() { eprintln!("{}", serde_json::json!({"event":"derived_bind","function":self.function,"path":"local","plan":if mesh {"mesh"} else {"restate"},"hit":false,"bytes":bytes.len(),"ms":t.elapsed().as_secs_f64() * 1e3,"build_ms":build_ms,"key":key})); }
        DERIVED_STATS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        DERIVED_STATS[2].fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        if shared { SHARED_DERIVED.lock().unwrap().insert(key.to_owned(), (p, bytes.clone(), got)); }
        Ok((p, bytes))
    }
    fn local_receipt(&self, bind: &Value, got: &str) {
        let recipe = bind["recipe"].as_str().unwrap_or("");
        let name = bind["name"].as_str().or(bind["output"].as_str()).unwrap_or("");
        let pinned = bind["sha256"].as_str();
        if let Some(t) = LOCAL_TOPOLOGY.lock().unwrap().as_mut() {
            let r = t.built.entry(self.function.clone()).or_default();
            r.built += 1;
            match pinned {
                Some(h) if h == got => { r.pinned += 1; r.equal += 1; },
                Some(_) => { r.pinned += 1; r.differs.push(format!("{name} ({recipe})")); },
                None => r.unpinned.push(format!("{name} ({recipe})")),
            }
        }
    }
    unsafe fn local_build(&self, cuda: &Cuda, bind: &Value, l: &LaunchInfo, env: &Env, live: &HashMap<u64, submit::Allocation>, plan: LocalPlan) -> Result<(Vec<u8>, String), String> {
        let recipe = bind["recipe"].as_str().unwrap_or("");
        let name = bind["name"].as_str().or(bind["output"].as_str()).unwrap_or("");
        let bytes = match plan {
            LocalPlan::Restate => self.restate(cuda, bind, l, env, live)?,
            LocalPlan::Mesh { group, dims, dim_errors, arrays, extent } => {
                let read = |(a, o): crate::topo::Key, n: usize| -> Result<Vec<i32>, String> {
                    let al = live.get(&a).ok_or_else(|| format!("mesh allocation {a} is not live"))?;
                    if o + n * 4 > al.bytes { return Err(format!("{n} words at byte {o} pass the end of mesh allocation {a} ({} bytes)", al.bytes)); }
                    // hunt dyn-entry-gap: a mesh array is read from the card once per run (the plan's mesh arrays are
                    // constants: no command writes them), keyed by allocation, address, offset and length.
                    let ck = (a, al.address, o, n);
                    if derived_shared_on() { if let Some(v) = MESH_READS.lock().unwrap().get(&ck) { return Ok(v.clone()); } }
                    let mut raw = vec![0u8; n * 4];
                    crate::cuda::checked("cuMemcpyDtoH local topology", (cuda.download)(raw.as_mut_ptr().cast(), al.address + o as u64, n * 4))?;
                    let v: Vec<i32> = raw.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
                    if derived_shared_on() { MESH_READS.lock().unwrap().insert(ck, v.clone()); }
                    Ok(v)
                };
                if let Some(src) = extent {
                    let n = if recipe.contains("f32-count") { dims.get("ncells") } else { dims.get("nedges") }
                        .copied().ok_or_else(|| format!("{}: {recipe} {name}: {}", self.function, dim_errors.join("; ")))?;
                    let (a, o) = self.device_target(l, &src)?.ok_or_else(|| format!("{}: {recipe}: {src} is not a device argument", self.function))?;
                    let al = live.get(&a).ok_or_else(|| format!("{}: {src} allocation {a} is not live", self.function))?;
                    if o + 4 * n > al.bytes { return Err(format!("{}: {recipe} {name}: the rank's {n} words of {src} pass the end of its allocation {a} ({} bytes at {o})", self.function, al.bytes)); }
                }
                let mut m = crate::topo::Mesh { dims, ..Default::default() };
                for (n, cand, rows) in arrays {
                    let w = cand[0].1;
                    let first = read(cand[0].0, rows * w).map_err(|e| format!("{}: {recipe} {name}: {n}: {e}", self.function))?;
                    for (k, _) in &cand[1..] {
                        if read(*k, rows * w).map_err(|e| format!("{}: {n}: {e}", self.function))? != first {
                            return Err(format!("{}: {recipe} {name}: mesh group {group} has different {n} arrays of width {w} (allocations {:?}): the rank's mesh is ambiguous",
                                               self.function, cand.iter().map(|x| x.0).collect::<Vec<_>>()));
                        }
                    }
                    m.arrays.insert(n, (first, w));
                }
                crate::recipes::pack(&crate::topo::build(recipe, name, bind["component"].as_str().unwrap_or(""), &m)
                    .map_err(|e| format!("{}: local {recipe} {name}: {e}{}", self.function, if e.contains("mesh dimension") { format!(" ({})", dim_errors.join("; ")) } else { String::new() }))?)
            },
        };
        let got = crate::exec::digest(&bytes);
        let _ = (recipe, name);
        self.local_receipt(bind, &got);
        Ok((bytes, got))
    }
}

/// --local-topology 1: how a topology derived binding is built at this launch.
enum LocalPlan {
    /// From this launch's own arguments (restate, the recipes F057 and G6 restated).
    Restate,
    /// From the rank's mesh: each needed array's candidates (all equal word for word) and rows, the group's dims.
    Mesh { group: usize, dims: BTreeMap<String, usize>, dim_errors: Vec<String>, arrays: Vec<(String, Vec<(crate::topo::Key, usize)>, usize)>, extent: Option<String> },
}

/// --local-topology 1 receipt, per function: arrays built, how many carry a pinned SHA256 and how many equal it.
#[derive(Default, Debug)]
struct LocalReceipt { built: u64, pinned: u64, equal: u64, differs: Vec<String>, unpinned: Vec<String> }
/// multirank-2 window state: the plan's mesh registry and the build receipts. One process runs one window; the state
/// is installed by `ad-binder window --local-topology 1` before the first launch (None: today's behaviour exactly).
pub struct LocalTopology { pub registry: crate::topo::Registry, built: BTreeMap<String, LocalReceipt> }
static LOCAL_TOPOLOGY: std::sync::Mutex<Option<LocalTopology>> = std::sync::Mutex::new(None);
/// hunt dyn-entry-gap: the mesh tables shared by every kernel (key -> device copy, bytes, SHA256) and the mesh arrays
/// read from the card; both belong to the installed registry and are dropped with it.
type SharedTables = std::sync::Mutex<std::collections::HashMap<String, (u64, Vec<u8>, String)>>;
static SHARED_DERIVED: std::sync::LazyLock<SharedTables> = std::sync::LazyLock::new(Default::default);
static MESH_READS: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<(u64, u64, usize, usize), Vec<i32>>>> = std::sync::LazyLock::new(Default::default);
/// [shared hits, builds, build ns]
static DERIVED_STATS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
fn derived_shared_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AD_DERIVED_SHARED").as_deref() != Ok("0"))
}
/// bs-launch: AD_BRANCH_NULL=1 binds branch records to null (see the "branch_record" binding in launch_impl), except for
/// the kernels whose source adds to the record without testing the pointer (read in their .cu files on 2026-11-13; a
/// per-block or per-column count, not the per-thread adds that cost) and any named in AD_BRANCH_KEEP (comma list).
const BRANCH_KEEP: &[&str] = &["edge_fields_from_saved_u_f32", "laplacian_divergence_v841_f32", "laplacian_vorticity_v841_f32",
    "pv_apvm_v841_f32", "pv_edge_base_f32", "smagorinsky_v841_f32", "tangential_velocity_f32", "theta_filter_lap2_v841_f32",
    "theta_filter_lap4_v841_f32", "vertex_diagnostics_v841_f32", "w_vertical_flux_f32", "fork_minmax", "prep_mass_v841_f32",
    "mpas_lw_prepare", "oracle_drivers_seaice_prepare", "regional_relaxzone_filter_edge_v841", "regional_clamp_negative_scalars_v841",
    "acoustic_ru_regional_v841", "acoustic_rs_ts_regional_v841", "acoustic_column_solve_regional_v841", "transport_edge_values_regional_v841"];
fn branch_null(function: &str) -> bool {
    static ON: std::sync::OnceLock<Option<Vec<String>>> = std::sync::OnceLock::new();
    let keep = ON.get_or_init(|| (std::env::var("AD_BRANCH_NULL").as_deref() == Ok("1")).then(|| {
        let mut k: Vec<String> = BRANCH_KEEP.iter().map(|s| s.to_string()).collect();
        if let Ok(v) = std::env::var("AD_BRANCH_KEEP") { k.extend(v.split(',').filter(|x| !x.is_empty()).map(str::to_owned)); }
        k }));
    keep.as_ref().is_some_and(|k| !k.iter().any(|x| x == function))
}
/// bs-launch (AD_BS_PROF >= 1, diagnostic, default off): host wall inside launch_impl by binding kind.
pub mod lprof {
    use std::sync::atomic::{AtomicU64, AtomicU8, Ordering::Relaxed};
    pub const NAMES: [&str; 13] = ["prologue", "scalar", "primal", "tangent_adjoint", "workspace", "failure", "branch_record",
                                   "derived", "saved", "other", "geometry", "launch_call", "bindings_counted"];
    static NS: [AtomicU64; 13] = [const { AtomicU64::new(0) }; 13];
    static ON: AtomicU8 = AtomicU8::new(255);
    pub fn on() -> bool {
        let v = ON.load(Relaxed);
        if v != 255 { return v == 1; }
        let o = std::env::var("AD_BS_PROF").ok().and_then(|s| s.parse::<u8>().ok()).unwrap_or(0) > 0;
        ON.store(o as u8, Relaxed); o
    }
    pub fn add(k: usize, t: &mut std::time::Instant) { let n = std::time::Instant::now(); NS[k].fetch_add(n.duration_since(*t).as_nanos() as u64, Relaxed); *t = n; }
    pub fn count(k: usize) { NS[k].fetch_add(1, Relaxed); }
    pub fn take() -> serde_json::Value {
        serde_json::Value::Object(NAMES.iter().enumerate().map(|(k, n)| {
            let v = NS[k].swap(0, Relaxed);
            ((*n).to_owned(), if k == 12 { serde_json::json!(v) } else { serde_json::json!(v as f64 / 1e9) })
        }).collect())
    }
    pub fn kind(k: &str) -> usize {
        match k { "scalar" => 1, "primal" => 2, "tangent" | "adjoint" => 3, "workspace" => 4, "failure" => 5, "branch_record" => 6,
                  "derived" => 7, "saved" => 8, _ => 9 }
    }
}
fn derived_trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AD_DERIVED_TRACE").as_deref() == Ok("1"))
}
fn drop_shared_tables() { SHARED_DERIVED.lock().unwrap().clear(); MESH_READS.lock().unwrap().clear(); RECIPE_MEMO.lock().unwrap().clear(); }
/// hunt dyn-entry-gap: a restated recipe's whole output set, by its inputs' identities (see input_id).
static RECIPE_MEMO: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<String, std::sync::Arc<Vec<Vec<i32>>>>>> = std::sync::LazyLock::new(Default::default);
fn recipe_memo(key: &str, f: impl FnOnce() -> Result<Vec<Vec<i32>>, String>) -> Result<std::sync::Arc<Vec<Vec<i32>>>, String> {
    if derived_shared_on() { if let Some(v) = RECIPE_MEMO.lock().unwrap().get(key) { return Ok(v.clone()); } }
    let v = std::sync::Arc::new(f()?);
    if derived_shared_on() { RECIPE_MEMO.lock().unwrap().insert(key.to_owned(), v.clone()); }
    Ok(v)
}
/// hunt dyn-entry-gap: one int32 topology array read from the card (blocking copy, which also drains the queue), once per
/// run per (allocation, address, offset, length).
unsafe fn mesh_read_memo(cuda: &Cuda, alloc: u64, address: u64, offset: usize, n: usize) -> Result<Vec<i32>, String> {
    let ck = (alloc, address, offset, n);
    if derived_shared_on() { if let Some(v) = MESH_READS.lock().unwrap().get(&ck) { return Ok(v.clone()); } }
    let mut raw = vec![0u8; n * 4];
    crate::cuda::checked("cuMemcpyDtoH derived", (cuda.download)(raw.as_mut_ptr().cast(), address + offset as u64, n * 4))?;
    let v: Vec<i32> = raw.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
    if derived_shared_on() { MESH_READS.lock().unwrap().insert(ck, v.clone()); }
    Ok(v)
}
pub fn local_topology_on() -> bool { LOCAL_TOPOLOGY.lock().unwrap().is_some() }
/// seam-dot: drop the registry so a later global-mesh launch binds its sha-pinned topology again.
pub fn clear_local_topology() { *LOCAL_TOPOLOGY.lock().unwrap() = None; drop_shared_tables(); }
/// Install the registry built from the plan's launches (every launch of a kernel-set row, before the forward sweep).
pub fn install_local_topology<'a>(kernels: &HashMap<String, KernelDeriv>, launches: impl Iterator<Item = (&'a str, Vec<submit::Argument>, [u32; 3], [u32; 3], u32)>) -> Value {
    install_local_topology_with(kernels, launches, Vec::new())
}
/// seam-dot: as install_local_topology, plus observations that stand in for the plan's other launches of a rank mesh.
pub fn install_local_topology_with<'a>(kernels: &HashMap<String, KernelDeriv>, launches: impl Iterator<Item = (&'a str, Vec<submit::Argument>, [u32; 3], [u32; 3], u32)>, extra: Vec<crate::topo::Observation>) -> Value {
    let mut registry = crate::topo::Registry::default();
    for o in extra { registry.observe(o); }
    let mut seen = std::collections::HashSet::new();
    for (f, args, grid, block, shared) in launches {
        let Some(k) = kernels.get(f) else { continue };
        let l = LaunchInfo { args: &args, grid, block, shared };
        let o = k.mesh_observation(&l);
        // The same launch signature repeats every step: observe each distinct one once.
        let sig = format!("{f}:{:?}:{:?}", o.arrays, o.dims);
        if seen.insert(sig) { registry.observe(o); }
    }
    registry.finish();
    let summary = local_topology_registry_summary(&registry);
    drop_shared_tables();
    *LOCAL_TOPOLOGY.lock().unwrap() = Some(LocalTopology { registry, built: BTreeMap::new() });
    summary
}
fn local_topology_registry_summary(r: &crate::topo::Registry) -> Value {
    serde_json::json!({"observations": r.observations(), "groups": r.groups.iter().enumerate().map(|(i, g)| serde_json::json!({
        "group": i, "dims": g.dims.iter().map(|(d, v)| (d.clone(), v.iter().map(|x| x.0).collect::<BTreeSet<_>>().into_iter().collect::<Vec<_>>())).collect::<BTreeMap<_, _>>(),
        "arrays": g.arrays.iter().map(|(n, vs)| (n.clone(), vs.iter().map(|v| format!("{}+{} w{:?}", v.key.0, v.key.1, v.widths)).collect::<Vec<_>>())).collect::<BTreeMap<_, _>>()})).collect::<Vec<_>>()})
}
/// The receipt line: built N, equal to pinned M, per function. None when --local-topology is off.
pub fn local_topology_receipt() -> Option<Value> {
    let g = LOCAL_TOPOLOGY.lock().unwrap();
    let t = g.as_ref()?;
    let (mut n, mut p, mut m) = (0, 0, 0);
    let mut per = serde_json::Map::new();
    for (f, r) in &t.built {
        n += r.built; p += r.pinned; m += r.equal;
        per.insert(f.clone(), serde_json::json!({"built": r.built, "pinned": r.pinned, "equal_to_pinned": r.equal, "differs": r.differs, "unpinned": r.unpinned}));
    }
    let st = |i: usize| DERIVED_STATS[i].load(std::sync::atomic::Ordering::Relaxed);
    Some(serde_json::json!({"event": "local_topology", "built": n, "pinned": p, "equal_to_pinned": m, "functions": per,
                            "shared_tables": {"on": derived_shared_on(), "hits": st(0), "builds": st(1), "build_s": st(2) as f64 / 1e9,
                                              "mesh_arrays_cached": MESH_READS.lock().unwrap().len()},
                            "registry": local_topology_registry_summary(&t.registry)}))
}

#[cfg(test)]
mod local_topology_admission_tests {
    use super::*;
    /// g2 r2: fct_minmax_source's ad entry binds inv_off / inv_nbr / inv_slot (u2:inverse_cells_on_cell) pinned to a
    /// mesh whose tables are not in the store. Without --local-topology it is refused at load (named); with it the
    /// binding is admitted (local_build makes it from the launch's own n_edges_on_cell / cells_on_cell).
    #[test]
    fn inverse_cells_on_cell_admitted_only_under_local_topology() {
        let spec: Value = serde_json::json!({"launch_abi": {"schema": "ad.entry/v1", "entries": {"ad_1": {"args": [
            {"bind": "derived", "name": "inv_nbr", "sha256": "5ae160d12e4d1b3d93c85477389e43231ec5ee1cacbe762bad6f58a98b445a24", "recipe": "u2:inverse_cells_on_cell", "by_value": false}]}}}});
        let all = vec!["ad_1".to_string()];
        let why = unsupported_with(&spec, &all, &HashMap::new(), false);
        assert!(why.as_deref().is_some_and(|w| w.contains("inv_nbr (u2:inverse_cells_on_cell) not in the derived store")), "{why:?}");
        assert!(unsupported_with(&spec, &all, &HashMap::new(), true).is_none());
        // A non-topology recipe stays refused under local topology.
        let other: Value = serde_json::json!({"launch_abi": {"schema": "ad.entry/v1", "entries": {"ad_1": {"args": [
            {"bind": "derived", "name": "x", "sha256": "00", "recipe": "some_physics_table.py"}]}}}});
        assert!(unsupported_with(&other, &all, &HashMap::new(), true).is_some());
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
        let spec: Value = serde_json::from_str(include_str!("../../tests/data/held/mpas_cmbgb/mpas_cmbgb.json")).unwrap();
        let row: Value = serde_json::from_str(include_str!("../../tests/data/held/mpas_cmbgb/kset-row.json")).unwrap();
        let meta: Vec<Value> = row["arguments"].as_array().unwrap().clone();
        let names: Vec<String> = meta.iter().map(|m| m["name"].as_str().unwrap().to_owned()).collect();
        let k = KernelDeriv { function: "mpas_cmbgb".into(), spec, names, meta, module: ptr::null_mut(), functions: HashMap::new(),
            tl_order: vec!["tl".into()], ad_order: vec!["ad".into()], aug: true, elide_column_workspace_zero: false, unsupported: None, derived: Default::default() };
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
/// runner at tree/rust/hexad), so the fixture tests resolve that folder from whichever
/// crate is building them (S09 fixed the e2e crate's path and broke the runner's: 4 runner tests NotFound).
#[cfg(test)]
pub(crate) fn held_fixture_dir() -> std::path::PathBuf {
    let m = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    m.join("tests/data/held")
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
                              functions: HashMap::new(), tl_order, ad_order, aug, elide_column_workspace_zero: false, unsupported: why, derived: std::cell::RefCell::new(HashMap::new()) };
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
                      functions: HashMap::new(), tl_order, ad_order, aug, elide_column_workspace_zero: false, unsupported: why, derived: std::cell::RefCell::new(HashMap::new()) }
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
                      functions: HashMap::new(), tl_order, ad_order, aug, elide_column_workspace_zero: false, unsupported: why, derived: std::cell::RefCell::new(HashMap::new()) }
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
        // The live template, where the kernel set is present (a local copy of the kernel set).
        let t = Path::new("<kernel set>");
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
                      functions: HashMap::new(), tl_order, ad_order, aug, elide_column_workspace_zero: false, unsupported: why, derived: std::cell::RefCell::new(HashMap::new()) }
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
        // The live template, where the kernel set is present (a local copy of the kernel set).
        let t = Path::new("<kernel set>");
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

#[cfg(test)]
mod f057_divergence_damping_selected {
    //! F057 (SWARM-20261107): the 2-rank forward launches divergence_damping_selected_f32 in place of every
    //! divergence_damping_f32 (interior pass during the halo DMA, boundary pass after the receive fence). With no
    //! kernel-set row the window rule fell to wrules.rs "no derivative row and no rule": Refused, identity both ways,
    //! so every 2-rank window composed the dycore with d(u_new)/d(u) = I for damping. The row's entry must be
    //! dispatchable by this unit with derived topology built from the launch's own (local) cells_on_edge.
    use super::*;

    fn kd() -> KernelDeriv {
        let spec: Value = serde_json::from_str(include_str!("../../tests/data/multirank/dd_selected/divergence_damping_selected_f32.json")).unwrap();
        let row: Value = serde_json::from_str(include_str!("../../tests/data/multirank/dd_selected/kset-row.json")).unwrap();
        let meta: Vec<Value> = row["arguments"].as_array().unwrap().clone();
        let names: Vec<String> = meta.iter().map(|m| m["name"].as_str().unwrap().to_owned()).collect();
        let abi = spec["launch_abi"].clone();
        let (tl_order, ad_order) = (order(&abi, "tl"), order(&abi, "ad"));
        let all: Vec<String> = tl_order.iter().chain(ad_order.iter()).cloned().collect();
        let why = unsupported(&spec, &all, &HashMap::new());
        KernelDeriv { function: row["function"].as_str().unwrap().to_owned(), spec, names, meta, module: ptr::null_mut(),
                      functions: HashMap::new(), tl_order, ad_order, aug: false, elide_column_workspace_zero: false, unsupported: why, derived: Default::default() }
    }

    #[test]
    fn selected_damping_row_is_dispatchable_on_a_local_mesh() {
        let k = kd();
        // Before the fix: "derived binding edges_on_cell (local_edge_incidence) not in the derived store and no recipe".
        assert_eq!(k.unsupported, None);
        assert_eq!((k.tl_order.clone(), k.ad_order.clone()), (vec!["tl".to_owned()], vec!["ad".to_owned()]));
        // Forward argument order of hexcore/partition_overlap_v841.py selected_source.
        assert_eq!(k.names, ["interior_edge", "select_interior", "theta_m", "rtheta_pp", "rtheta_pp_old", "cells_on_edge",
                             "spec_zone_mask_edge", "coefficient", "n_cells_solve", "nlev", "ncells", "nedges", "rho_u_perturbation"]);
        // Same roles as divergence_damping_f32: TL rho_u in/out from a snapshot; AD seed snapshot + live seed + out.
        let r = k.roles();
        assert_eq!(r.tl_snapshot, ["rho_u_perturbation"]);
        assert_eq!(r.ad_replace, ["rho_u_perturbation"]);
        assert!(r.ad_snapshot.is_empty() && r.ad_post.is_empty());
        // Extents follow the launch (a rank's local extents, not the capture's 2562 / 7680).
        let i = |v: i32| submit::Argument::Bytes(v.to_le_bytes().to_vec());
        let f = |v: f32| submit::Argument::Bytes(v.to_le_bytes().to_vec());
        let d = |a: u64| submit::Argument::Device(submit::Slice { allocation: a, offset: 0 });
        let (nlev, ncells, nedges) = (59, 1349, 4041);
        let args = vec![d(0), i(1), d(1), d(2), d(3), d(4), d(5), f(960.0), i(1281), i(nlev), i(ncells), i(nedges), d(6)];
        let l = LaunchInfo { args: &args, grid: [(nedges as u32 + 255) / 256, 1, 1], block: [256, 1, 1], shared: 0 };
        let env = k.env(&l).unwrap();
        assert_eq!(env.get("select_interior"), Some(&1.0));
        let (nl, nc, ne) = (nlev as usize, ncells as usize, nedges as usize);
        for (name, n) in [("interior_edge", ne), ("theta_m", nl * nc), ("rtheta_pp_old", nl * nc), ("cells_on_edge", 2 * ne),
                          ("spec_zone_mask_edge", ne), ("rho_u_perturbation", nl * ne)] {
            assert_eq!(k.elements(name, &env).unwrap(), n, "{name}");
        }
        // Both entries take the selection arguments first, as the entry source declares them.
        for e in ["tl", "ad"] {
            let a: Vec<&str> = k.bindings(e).filter_map(|b| b["cuda_name"].as_str()).collect();
            assert_eq!(&a[..3], ["interior_edge", "select_interior", "theta"], "{e}");
        }
        let derived: Vec<&Value> = k.bindings("ad").filter(|b| b["bind"] == "derived").collect();
        assert_eq!(derived.len(), 3);
        assert!(derived.iter().all(|b| b["recipe"] == "local_edge_incidence" && b["sha256"].is_null()));
    }

    #[test]
    fn local_edge_incidence_rows_counts_width_and_refusal() {
        // 4 local cells, 5 edges; edge 4 is a halo edge whose endpoints are both cell 3 (MPAS one-past garbage form).
        let ends = [0, 1, 1, 2, 2, 0, 3, 0, 3, 3];
        let w = |b: Vec<u8>| -> Vec<i32> { b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect() };
        assert_eq!(w(local_edge_incidence("max_edges", &ends, 4).unwrap()), [3]);
        assert_eq!(w(local_edge_incidence("n_edges_on_cell", &ends, 4).unwrap()), [3, 2, 2, 2]);
        assert_eq!(w(local_edge_incidence("edges_on_cell", &ends, 4).unwrap()), [0, 2, 3, 0, 1, -1, 1, 2, -1, 3, 4, -1]);
        // An endpoint outside the local cells is refused by name, never wrapped.
        assert!(local_edge_incidence("edges_on_cell", &[0, 1, 1, -1], 2).is_err());
        assert!(local_edge_incidence("edges_on_cell", &[0, 1, 1, 2], 2).is_err());
    }
}

#[cfg(test)]
mod multirank2_local_topology {
    //! multirank-2 (--local-topology 1) on the real kernel-set rows (tree/ad/kset/int7f: the same dycore specs kset-int8
    //! carries): the mesh registry built from two ranks' launches, and the derived bindings of momentum_filter_lap4
    //! (same-mesh arrays it does not take), recover_edge_velocity (the compact edges_on_cell) and theta_finish (a
    //! launch-argument recipe) resolved to the launch's own rank. Host only: no store blob, no device.
    use super::*;

    fn kset_dir() -> std::path::PathBuf {
        let m = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        [m.join("../kset/int7f"), m.join("../../../../../../kset/int7f")].into_iter().find(|d| d.join("kernels.json").is_file()).expect("tree/ad/kset/int7f")
    }
    fn rows(fs: &[&str]) -> HashMap<String, KernelDeriv> {
        let dir = kset_dir();
        let set: Value = serde_json::from_slice(&std::fs::read(dir.join("kernels.json")).unwrap()).unwrap();
        let mut out = HashMap::new();
        for k in set["kernels"].as_array().unwrap() {
            let f = k["function"].as_str().unwrap();
            if !fs.contains(&f) { continue; }
            let sp = dir.join("specs").join(k["derivative"]["spec"]["path"].as_str().unwrap().trim_start_matches('/'));
            let spec: Value = serde_json::from_slice(&std::fs::read(&sp).unwrap()).unwrap();
            let meta: Vec<Value> = k["arguments"].as_array().unwrap().clone();
            let names: Vec<String> = meta.iter().map(|m| m["name"].as_str().unwrap().to_owned()).collect();
            let abi = spec["launch_abi"].clone();
            let (tl_order, ad_order) = (order(&abi, "tl"), order(&abi, "ad"));
            out.insert(f.to_owned(), KernelDeriv { function: f.to_owned(), spec, names, meta, module: ptr::null_mut(), functions: HashMap::new(),
                                                   tl_order, ad_order, aug: false, elide_column_workspace_zero: false, unsupported: None, derived: Default::default() });
        }
        assert_eq!(out.len(), fs.len(), "rows {fs:?}");
        out
    }
    /// One launch of `k` on rank `r`: mesh arrays at the rank's allocations, the rank's extents.
    fn launch(k: &KernelDeriv, r: u64, compact: bool) -> Vec<submit::Argument> {
        let (nc, ne, nv) = if r == 0 { (1283, 3900, 2600) } else { (1281, 3890, 2590) };
        let base = 1000 * (r + 1);
        k.names.iter().zip(&k.meta).enumerate().map(|(i, (n, m))| {
            if m["kind"] == "scalar" {
                let v: i32 = match n.as_str() { "ncells" => nc, "nedges" => ne, "nvertices" => nv, "nlev" => 55,
                                                "max_edges" | "maxedges" => if compact { 6 } else { 10 }, _ => 1 };
                return submit::Argument::Bytes(if m["dtype"] == "<f4" { (v as f32).to_le_bytes().to_vec() } else { v.to_le_bytes().to_vec() });
            }
            let a = match n.as_str() {
                "cells_on_edge" => 1, "edges_on_cell" if compact => 6, "edges_on_cell" => 2, "n_edges_on_cell" => 3,
                "vertices_on_edge" => 4, "edges_on_vertex" => 5, _ => 100 + i as u64,
            };
            submit::Argument::Device(submit::Slice { allocation: base + a, offset: 0 })
        }).collect()
    }

    #[test]
    #[ignore = "reads the kernel set tree/ad/kset/int7f, which this crate does not carry"]
    fn derived_bindings_resolve_to_the_launchs_own_rank() {
        // theta_finish_corrected passes edges_on_cell and cells_on_edge together: it joins each rank's cell and edge
        // arrays into one group (on pkg18m2 many dycore launches do).
        let fs = ["momentum_filter_lap4_v841_f32", "recover_edge_velocity_f32", "recover_terrain_w_f32", "theta_finish_v841_f32",
                  "theta_finish_corrected_v841_f32"];
        let k = rows(&fs);
        let compact = |f: &str| f == "recover_terrain_w_f32";
        let mut reg = crate::topo::Registry::default();
        for r in [1u64, 0] { for f in fs { let a = launch(&k[f], r, compact(f));
            reg.observe(k[f].mesh_observation(&LaunchInfo { args: &a, grid: [1, 1, 1], block: [1, 1, 1], shared: 0 })); } }
        reg.finish();
        assert_eq!(reg.groups.len(), 2, "{:?}", reg.groups);
        // Default mode: no test installs the window state, so the derived arm takes today's path.
        assert!(!local_topology_on());
        let mut keys: BTreeMap<(String, String), BTreeSet<String>> = BTreeMap::new();
        for r in [0u64, 1] {
            let base = 1000 * (r + 1);
            let (nc, ne) = if r == 0 { (1283, 3900) } else { (1281, 3890) };
            for f in fs {
                let kd = &k[f];
                let a = launch(kd, r, compact(f));
                let l = LaunchInfo { args: &a, grid: [1, 1, 1], block: [1, 1, 1], shared: 0 };
                let env = kd.env(&l).unwrap();
                let derived: Vec<&Value> = kd.bindings_all().filter(|b| b["bind"] == "derived").collect();
                assert!(!derived.is_empty() && derived.iter().all(|b| b["recipe"].as_str().is_some_and(crate::topo::topology_recipe)), "{f}");
                for b in derived {
                    let name = b["name"].as_str().or(b["output"].as_str()).unwrap().to_owned();
                    let (key, plan) = kd.local_resolve_with(&reg, b, &l, &env).unwrap_or_else(|e| panic!("{f} {name}: {e}"));
                    assert!(key.starts_with("local:") && !key.contains(&b["sha256"].as_str().unwrap_or("-").to_owned()), "{key}");
                    keys.entry((f.to_owned(), name.clone())).or_default().insert(key);
                    match (f, plan) {
                        ("theta_finish_v841_f32" | "theta_finish_corrected_v841_f32", LocalPlan::Restate) => {},
                        ("theta_finish_v841_f32" | "theta_finish_corrected_v841_f32", _) => panic!("{f} {name}: a launch-argument recipe"),
                        (_, LocalPlan::Restate) => panic!("{f} {name}: a mesh-input recipe"),
                        (_, LocalPlan::Mesh { dims, arrays, .. }) => {
                            assert_eq!((dims["ncells"], dims["nedges"]), (nc, ne), "{f} {name}");
                            for (n, cand, rows) in arrays {
                                let want = match (f, n.as_str()) {
                                    (_, "cells_on_edge") => (1, 2, ne),
                                    ("recover_terrain_w_f32", "edges_on_cell") => (6, 6, nc),
                                    (_, "edges_on_cell") => (2, 10, nc),
                                    (_, "n_edges_on_cell") => (3, 1, nc),
                                    (_, "vertices_on_edge") => (4, 2, ne),
                                    other => panic!("{f} {name}: unexpected input {other:?}"),
                                };
                                assert_eq!((cand, rows), (vec![((base + want.0, 0), want.1)], want.2), "{f} {name} {n}");
                            }
                        },
                    }
                }
            }
        }
        // Each binding gets one key per rank: rank 0's array is never bound on rank 1.
        assert!(keys.values().all(|s| s.len() == 2), "{keys:?}");
        // recover_edge_velocity takes only cells_on_edge; a launch whose mesh is not in the registry is refused by name.
        let kd = &k["recover_edge_velocity_f32"];
        let mut a = launch(kd, 0, false);
        let i = kd.names.iter().position(|n| n == "cells_on_edge").unwrap();
        a[i] = submit::Argument::Device(submit::Slice { allocation: 77, offset: 0 });
        let l = LaunchInfo { args: &a, grid: [1, 1, 1], block: [1, 1, 1], shared: 0 };
        let b = kd.bindings_all().find(|b| b["bind"] == "derived").unwrap();
        let e = kd.local_resolve_with(&reg, b, &l, &kd.env(&l).unwrap()).err().unwrap();
        assert!(e.contains("recover_edge_velocity_f32") && e.contains("mesh registry"), "{e}");
    }
}


#[cfg(test)]
mod f099_fixed_extent_follows_launch {
    //! F099 (swarm 2026-11-07): a numeric spec shape that the kernel-set row also carries (the dycore specs, captured on
    //! x1.2562) follows the launch scalars; only a row with no capture shape takes the spec's literal extent (c519d229's
    //! RRTMG-SW tab/ngb). Spec and row copied from kset-int7f w_vertical_flux_f32 and rsw_sfluxzen_b.
    use super::*;

    fn kd(function: &str, spec: Value, row: Value) -> KernelDeriv {
        let meta: Vec<Value> = row.as_array().unwrap().clone();
        let names = meta.iter().map(|m| m["name"].as_str().unwrap().to_owned()).collect();
        KernelDeriv { function: function.into(), spec, names, meta, module: ptr::null_mut(), functions: HashMap::new(),
            tl_order: vec!["tl".into()], ad_order: vec!["ad".into()], aug: false, elide_column_workspace_zero: false, unsupported: None, derived: Default::default() }
    }
    fn launch_env(k: &KernelDeriv, scalars: &[i32], devices: usize) -> Env {
        let mut args: Vec<submit::Argument> = scalars.iter().map(|v| submit::Argument::Bytes(v.to_le_bytes().to_vec())).collect();
        for a in 0..devices { args.push(submit::Argument::Device(submit::Slice { allocation: a as u64, offset: 0 })); }
        let l = LaunchInfo { args: &args, grid: [1, 1, 1], block: [128, 1, 1], shared: 0 };
        k.env(&l).unwrap()
    }

    #[test]
    fn dycore_numeric_shape_follows_ncells_below_capture() {
        let spec = serde_json::json!({"args": [
            {"name": "nlev", "shape": [], "role": "scalar"}, {"name": "ncells", "shape": [], "role": "scalar"},
            {"name": "w", "shape": [56, 2562], "role": "in"}, {"name": "rw", "shape": [56, 2562], "role": "in"},
            {"name": "flux", "shape": [56, 2562], "role": "out"}]});
        let dev = |n: &str| serde_json::json!({"name": n, "kind": "device", "dtype": "<f4", "shape": [56, 2562],
            "strides_bytes": [10248, 4], "pointer_depth": 1, "value_hex": null});
        let row = serde_json::json!([
            {"name": "nlev", "kind": "scalar", "dtype": "<i4", "shape": [], "value_hex": "37000000"},
            {"name": "ncells", "kind": "scalar", "dtype": "<i4", "shape": [], "value_hex": "020a0000"},
            dev("w"), dev("rw"), dev("flux")]);
        let k = kd("w_vertical_flux_f32", spec, row);
        // Capture mesh: unchanged, 56 x 2562.
        let env = launch_env(&k, &[55, 2562], 3);
        for a in ["w", "rw", "flux"] { assert_eq!(k.elements(a, &env).unwrap(), 56 * 2562, "{a} at the capture"); }
        // A per-rank / LAM mesh smaller than the capture: the span follows ncells (was floored at 143,472).
        for ncells in [400, 1, 2561] {
            let env = launch_env(&k, &[55, ncells], 3);
            for a in ["w", "rw", "flux"] { assert_eq!(k.elements(a, &env).unwrap(), 56 * ncells as usize, "{a} at ncells {ncells}"); }
        }
        // Larger mesh and more levels still follow.
        let env = launch_env(&k, &[59, 40962], 3);
        assert_eq!(k.elements("w", &env).unwrap(), 60 * 40962);
    }

    #[test]
    fn shapeless_row_keeps_the_literal_extent() {
        let spec = serde_json::json!({"args": [
            {"name": "ncol", "shape": [], "role": "scalar"}, {"name": "nlayers", "shape": [], "role": "scalar"},
            {"name": "tab", "shape": [104303], "role": "in"}, {"name": "ngb", "shape": [112], "role": "in"},
            {"name": "jp", "shape": "[ncol, nlayers]", "role": "in"}]});
        let row = serde_json::json!([
            {"name": "ncol", "kind": "scalar", "dtype": "<i4", "shape": []},
            {"name": "nlayers", "kind": "scalar", "dtype": "<i4", "shape": []},
            {"name": "tab", "kind": "device", "dtype": "<f4", "shape": []},
            {"name": "ngb", "kind": "device", "dtype": "<i4", "shape": []},
            {"name": "jp", "kind": "device", "dtype": "<i4", "shape": []}]);
        let k = kd("rsw_sfluxzen_b", spec, row);
        for ncol in [1, 64, 2562] {
            let env = launch_env(&k, &[ncol, 59], 3);
            assert_eq!(k.elements("tab", &env).unwrap(), 104303);
            assert_eq!(k.elements("ngb", &env).unwrap(), 112);
            assert_eq!(k.elements("jp", &env).unwrap(), ncol as usize * 59);
        }
    }
}

/// D1: the forward argument names a row's tangent entries write (bind "tangent", io out or inout), over every entry
/// of `tl_order` (the single entry "tl" when the order is empty), each name once, in entry order.
pub fn tangent_written_names(spec: &Value, tl_order: &[String]) -> Vec<String> {
    let one = ["tl".to_string()];
    let order = if tl_order.is_empty() { &one[..] } else { tl_order };
    let mut names: Vec<String> = Vec::new();
    for e in order {
        for x in spec["launch_abi"]["entries"][e.as_str()]["args"].as_array().into_iter().flatten() {
            if x["bind"] != "tangent" { continue; }
            let io = x["io"].as_str().unwrap_or("");
            if !(io == "out" || io == "inout") { continue; }
            if let Some(n) = x["arg"].as_str() { if !names.iter().any(|m| m == n) { names.push(n.to_owned()); } }
        }
    }
    names
}

/// switchoff (g2, 2026-11-11): the shadow words an argument of `elements` elements of type `dtype` owns. A shadow
/// holds one word per 4 bytes of the forward array, so a 1-byte or 2-byte argument (u8, bool, i16) owns
/// elements * size / 4 words, rounded up. Every other type keeps one word per element, as before.
/// THE BREAKAGE THIS PREVENTS: a zero rule (AD_ZERO_DERIV, the --refuse hold of fe7f4916) cleared one word per ELEMENT
/// of every tangent view a row writes. rlw_rtrn_march_coalesced writes the u8 flag iclddn_p (ncol, nl, 140): its
/// entries write one shadow word per 4 flags (march_ad.cuh `t_iclddn_p[RM_IDX_PRO(lev) >> 2] = 0.0f`), the zero rule
/// cleared 4 times that, so the 3 * ncol * nl * 140 / 4 words after the flag's own were wiped at every infrared
/// launch: tangent words of other arrays that the model still reads. With radiation refused the 120 km 6 h derivative
/// read 0.690 / 0.472 of the real temperature change at 5 / 6 h where the identity form read 0.785 / 0.698
/// (<local path>), and AD_ZERO_DERIV of this one kind read 0.689 / 0.471
/// (<local path> test machine/lwzB/rows-0323Z.txt, run g2-c2-lwzB3).
pub fn owned_words(elements: usize, dtype: &str) -> usize {
    let size = match dtype { "u8" | "i8" | "bool" | "|u1" | "|i1" | "|b1" => 1, "u16" | "i16" | "<u2" | "<i2" => 2, _ => 4 };
    (elements * size + 3) / 4
}

#[cfg(test)]
mod refuse_hold_written_tests {
    use super::*;

    /// switchoff: a zero rule clears the words an argument owns. One word per element for 4-byte types (unchanged),
    /// one word per 4 elements, rounded up, for a u8.
    #[test]
    fn a_zero_rule_clears_only_the_words_the_argument_owns() {
        for n in [0usize, 1, 3, 4, 5, 140, 1874 * 64 * 140] {
            for t in ["f32", "i32", "u32", "<f4", "", "f64", "u64"] { assert_eq!(owned_words(n, t), n, "{t} {n}"); }
            for t in ["u8", "i8", "bool", "|u1"] { assert_eq!(owned_words(n, t), (n + 3) / 4, "{t} {n}"); }
            for t in ["u16", "i16"] { assert_eq!(owned_words(n, t), (n + 1) / 2, "{t} {n}"); }
        }
    }
    /// switchoff: the infrared march row. Refused or zeroed, it is held on the six radiance views (one word per
    /// element) and on the u8 flag iclddn_p's own ncol * nl * 140 / 4 words. Before the fix iclddn_p counted
    /// ncol * nl * 140 words: 3 in 4 of them belong to whatever lies after the flag array.
    #[test]
    fn the_refused_infrared_march_row_does_not_zero_past_its_u8_flag_output() {
        let dir = held_fixture_dir().join("rlw_rtrn_march_coalesced");
        let spec: Value = serde_json::from_slice(&std::fs::read(dir.join("rlw_rtrn_march_coalesced.spec.json")).unwrap()).unwrap();
        let row: Value = serde_json::from_slice(&std::fs::read(dir.join("kernel-row.json")).unwrap()).unwrap();
        let abi = spec["launch_abi"].clone();
        let names: Vec<String> = row["arguments"].as_array().unwrap().iter().map(|a| a["name"].as_str().unwrap().to_owned()).collect();
        let meta = row["arguments"].as_array().unwrap().clone();
        let (tl_order, ad_order) = (order(&abi, "tl"), order(&abi, "ad"));
        let k = KernelDeriv { function: "rlw_rtrn_march_coalesced".into(), spec, names: names.clone(), meta: meta.clone(), module: ptr::null_mut(),
                              functions: HashMap::new(), tl_order, ad_order, aug: false, elide_column_workspace_zero: false, unsupported: None, derived: std::cell::RefCell::new(HashMap::new()) };
        assert_eq!(k.dtype_of("iclddn_p"), "u8");
        assert_eq!(k.dtype_of("radld_p"), "f32");
        for (ncol, nl) in [(1874usize, 64usize), (40962, 59), (37, 65)] {
            let args: Vec<submit::Argument> = names.iter().zip(meta.iter()).enumerate().map(|(i, (n, m))| {
                if m["kind"] == "scalar" {
                    match n.as_str() {
                        "ncol" => submit::Argument::Bytes((ncol as i32).to_le_bytes().to_vec()),
                        "nl" => submit::Argument::Bytes((nl as i32).to_le_bytes().to_vec()),
                        _ => submit::Argument::Bytes(3.5971222f32.to_le_bytes().to_vec()),
                    }
                } else { submit::Argument::Device(submit::Slice { allocation: 100 + i as u64, offset: 0 }) }
            }).collect();
            let l = LaunchInfo { args: &args, grid: [1, 1, 1], block: [1, 1, 1], shared: 0 };
            let w = k.written_targets(&l).unwrap();
            let got: Vec<(&str, usize)> = w.iter().map(|(n, _, _, c)| (n.as_str(), *c)).collect();
            let g3 = ncol * nl * 140;
            assert_eq!(got, vec![("radld_p", g3), ("radclrd_p", g3), ("radlu_p", g3), ("radclru_p", g3),
                                 ("radlu_sfc", ncol * 140), ("radclru_sfc", ncol * 140), ("iclddn_p", (g3 + 3) / 4)], "ncol {ncol} nl {nl}");
            // No written view is longer than the forward array it shadows (4 bytes a word).
            let env = k.env(&l).unwrap();
            for (n, _, _, c) in &w {
                let bytes = k.elements(n, &env).unwrap() * if n == "iclddn_p" { 1 } else { 4 };
                assert!(c * 4 <= bytes + 3, "{n}: {c} words past a {bytes}-byte array");
            }
        }
    }
    /// D1: what a refused row is held on = the out / inout tangent views of every tangent entry, nothing else.
    #[test]
    fn written_names_are_the_out_and_inout_tangent_views_of_the_tangent_entries() {
        let spec: Value = serde_json::json!({"launch_abi": {"entries": {
            "tl_1": {"args": [{"bind": "primal", "arg": "t", "io": "in"}, {"bind": "tangent", "arg": "t", "io": "in"}, {"bind": "tangent", "arg": "flux", "io": "out"}]},
            "tl_2": {"args": [{"bind": "tangent", "arg": "flux", "io": "in"}, {"bind": "tangent", "arg": "heat", "io": "inout"}, {"bind": "tangent", "arg": "flux", "io": "inout"}]},
            "ad_1": {"args": [{"bind": "adjoint", "arg": "t", "io": "inout"}, {"bind": "tangent", "arg": "never", "io": "out"}]}}}});
        let order = vec!["tl_1".to_string(), "tl_2".to_string()];
        assert_eq!(tangent_written_names(&spec, &order), vec!["flux".to_string(), "heat".to_string()]);
        let single: Value = serde_json::json!({"launch_abi": {"entries": {"tl": {"args": [{"bind": "tangent", "arg": "out", "io": "out"}]}}}});
        assert_eq!(tangent_written_names(&single, &[]), vec!["out".to_string()]);
        assert!(tangent_written_names(&serde_json::json!({}), &[]).is_empty());
    }
}

#[cfg(test)]
mod spec_path_tests {
    use super::*;
    /// A kernel set is one folder that loads on any machine: a spec is found beside the set whether its row records
    /// it relative (kset-d12) or by the absolute path of the machine that made it (kset-int7f).
    #[test]
    fn a_spec_is_read_beside_the_kernel_set_however_its_path_is_recorded() {
        let root = std::env::temp_dir().join(format!("hexad-spec-path-{}", std::process::id()));
        std::fs::create_dir_all(root.join("specs/srv/origin/x")).unwrap();
        std::fs::write(root.join("specs/srv/origin/x/k.json"), b"{}").unwrap();
        let kernels = root.join("kernels.json");
        assert_eq!(spec_path(&kernels, "/srv/origin/x/k.json"), root.join("specs/srv/origin/x/k.json"));
        assert_eq!(spec_path(&kernels, "/srv/origin/y/k.json"), std::path::PathBuf::from("/srv/origin/y/k.json"));
        assert_eq!(spec_path(&kernels, "specs/k.json"), root.join("specs/k.json"));
        std::fs::remove_dir_all(&root).unwrap();
    }
}

#[cfg(test)]
mod image_path_tests {
    use super::*;
    /// A row without `image` keeps its own file; a named image is shared; a name that leaves cubins/ is refused.
    #[test]
    fn a_row_names_a_shared_image_or_keeps_its_own() {
        let c = Path::new("/k/cubins");
        let own = serde_json::json!({"row": "r"});
        assert_eq!(image_path(c, "ntiedtke_prep", &own).unwrap(), c.join("ntiedtke_prep.fatbin"));
        let shared = serde_json::json!({"row": "r", "shared_image": "nt_ad.fatbin"});
        // a shipped row's record of where its image came from is not a name: the row keeps its own file
        let record = serde_json::json!({"row": "r", "image": {"path": "<box>/kset-ovl/cubins/mpas_layer_histogram.fatbin", "sha256": "8c60"}});
        assert_eq!(image_path(c, "mpas_layer_histogram", &record).unwrap(), c.join("mpas_layer_histogram.fatbin"));
        assert_eq!(image_path(c, "ntiedtke_prep", &shared).unwrap(), c.join("nt_ad.fatbin"));
        assert_eq!(image_path(c, "ntiedtke_cuascn", &shared).unwrap(), c.join("nt_ad.fatbin"));
        for bad in [serde_json::json!({"shared_image": "../x.fatbin"}), serde_json::json!({"shared_image": "/abs/x.fatbin"}),
                    serde_json::json!({"shared_image": ""}), serde_json::json!({"shared_image": ".."}), serde_json::json!({"shared_image": 3})] {
            assert!(image_path(c, "f", &bad).is_err(), "{bad}");
        }
    }
}
