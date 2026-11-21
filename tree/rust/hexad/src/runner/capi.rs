// SPDX-License-Identifier: Apache-2.0
//! libhexad-phys: `libhexad-phys` C ABI v2 over the physics window
//! (window.rs). the host owns the State and Increment; the runner owns the trajectory (plan P_k and its 19 boundaries), the
//! tangent and cotangent shadows (one float32 arena), the power-of-two scale table and the operator manifest. The runner
//! never sees a the host type: only this ABI, device pointers and an allocation table (hexad_phys.h).
//!
//! * open(plan dir, kernel set, bundle, operator manifest, options): the same load, preflight, spec-hash admission and
//!   Win::new as `ad-binder window`, with the manifest translated into the window's flags (and its sha256s checked).
//! * tl_init / tl_step / tl_get: the tangent sweep one plan step per call (forward_step_c with the tangent). The first
//!   tangent pass of a trajectory checks every boundary hash against reference.jsonl and stores the boundaries.
//! * ad_init / ad_put / ad_step / ad_get: the reverse one step per call (reverse_step over the stored boundaries), with
//!   the observation forcing added between calls (rung 6's contract, nothing counted twice).
//! * Increment <-> arena: T0 / T0^T at boundary 0 and Ts / Ts^T at boundary s (hexad_t.cu, alloc-map.json).
//! * Boundaries in pinned host RAM (a budget; beyond it, local-NVMe files with async prefetch, counted).
//! * Counters: H2D / D2H bytes and calls and host barriers by category (glue / checkpoint / forward replay), through
//!   trampolines on the driver entry points: no added synchronisation.
//! A stage with no GPU path refuses with a DEFECT line; there is no CPU fallback.
use super::*;
use std::ffi::{c_char, c_int, CStr, CString};
use std::sync::Arc;

#[path = "live.rs"] pub mod live;
#[path = "native_rewind.rs"] pub mod native_rewind;

pub const ABI_VERSION: u32 = 2;
/// The pinned CuPy libraries' loader symbols, exported from the cdylib only (see capi_exports.rs).
#[cfg(feature = "capi-exports")] #[path = "capi_exports.rs"] pub mod capi_exports;
const T_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/hexad_t.fatbin"));

// ------------------------------------------------------------------------------------------------ counters
pub mod xfer {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    /// 0: sweep (the tangent / reverse rule machinery: tapes, rule dispatch, failure flags), 1: checkpoint (boundary
    /// store / restore / anchors / redo compare), 2: forward replay (the recorded forward's own transfers inside the
    /// sweeps), 3: glue (what the ABI adds: increment maps, seeds, dots, result hashes).
    pub const SWEEP: usize = 0;
    pub const CHECKPOINT: usize = 1;
    pub const FORWARD: usize = 2;
    pub const GLUE: usize = 3;
    pub const K: usize = 4;
    thread_local! { static CAT: Cell<usize> = const { Cell::new(0) }; }
    pub struct Cat(usize);
    impl Cat {
        pub fn checkpoint() -> Cat { Cat(CAT.with(|x| x.replace(CHECKPOINT))) }
        pub fn forward() -> Cat { Cat(CAT.with(|x| x.replace(FORWARD))) }
        pub fn glue() -> Cat { Cat(CAT.with(|x| x.replace(GLUE))) }
    }
    impl Drop for Cat { fn drop(&mut self) { CAT.with(|x| x.set(self.0)); } }
    pub fn cat() -> usize { CAT.with(|x| x.get()) }
    pub struct C { pub h2d: [AtomicU64; K], pub d2h: [AtomicU64; K], pub h2d_n: [AtomicU64; K], pub d2h_n: [AtomicU64; K],
                   pub ctx: [AtomicU64; K], pub stream: [AtomicU64; K], pub event: [AtomicU64; K] }
    #[allow(clippy::declare_interior_mutable_const)]
    const Z: AtomicU64 = AtomicU64::new(0);
    pub static N: C = C { h2d: [Z; K], d2h: [Z; K], h2d_n: [Z; K], d2h_n: [Z; K], ctx: [Z; K], stream: [Z; K], event: [Z; K] };
    pub fn add(a: &[AtomicU64; K], v: u64) { a[cat()].fetch_add(v, Relaxed); }
    pub fn get(a: &[AtomicU64; K]) -> [u64; K] { [a[0].load(Relaxed), a[1].load(Relaxed), a[2].load(Relaxed), a[3].load(Relaxed)] }
}

struct Real {
    upload: unsafe extern "C" fn(u64, *const c_void, usize) -> i32,
    upload_async: unsafe extern "C" fn(u64, *const c_void, usize, *mut c_void) -> i32,
    download: unsafe extern "C" fn(*mut c_void, u64, usize) -> i32,
    download_async: unsafe extern "C" fn(*mut c_void, u64, usize, *mut c_void) -> i32,
    ctx_sync: unsafe extern "C" fn() -> i32,
    stream_sync: crate::cuda::PtrCall,
    event_sync: crate::cuda::PtrCall,
}
static REAL: std::sync::OnceLock<Real> = std::sync::OnceLock::new();
unsafe extern "C" fn c_upload(d: u64, s: *const c_void, n: usize) -> i32 { xfer::add(&xfer::N.h2d, n as u64); xfer::add(&xfer::N.h2d_n, 1); (REAL.get().unwrap().upload)(d, s, n) }
unsafe extern "C" fn c_upload_async(d: u64, s: *const c_void, n: usize, st: *mut c_void) -> i32 { xfer::add(&xfer::N.h2d, n as u64); xfer::add(&xfer::N.h2d_n, 1); (REAL.get().unwrap().upload_async)(d, s, n, st) }
unsafe extern "C" fn c_download(d: *mut c_void, s: u64, n: usize) -> i32 { xfer::add(&xfer::N.d2h, n as u64); xfer::add(&xfer::N.d2h_n, 1); (REAL.get().unwrap().download)(d, s, n) }
unsafe extern "C" fn c_download_async(d: *mut c_void, s: u64, n: usize, st: *mut c_void) -> i32 { xfer::add(&xfer::N.d2h, n as u64); xfer::add(&xfer::N.d2h_n, 1); (REAL.get().unwrap().download_async)(d, s, n, st) }
unsafe extern "C" fn c_ctx_sync() -> i32 { xfer::add(&xfer::N.ctx, 1); (REAL.get().unwrap().ctx_sync)() }
unsafe extern "C" fn c_stream_sync(s: *mut c_void) -> i32 { xfer::add(&xfer::N.stream, 1); (REAL.get().unwrap().stream_sync)(s) }
unsafe extern "C" fn c_event_sync(e: *mut c_void) -> i32 { xfer::add(&xfer::N.event, 1); (REAL.get().unwrap().event_sync)(e) }
/// Route the driver's transfer and wait entry points through the counters (the same calls, one atomic add each).
pub(crate) unsafe fn install_counters(c: &mut Cuda) {
    let _ = REAL.set(Real { upload: c.upload, upload_async: c.upload_async, download: c.download, download_async: c.download_async,
                            ctx_sync: c.context_synchronize, stream_sync: c.stream_synchronize, event_sync: c.event_synchronize });
    c.upload = c_upload; c.upload_async = c_upload_async; c.download = c_download; c.download_async = c_download_async;
    c.context_synchronize = c_ctx_sync; c.stream_synchronize = c_stream_sync; c.event_synchronize = c_event_sync;
}

// ------------------------------------------------------------------------------------------------ host bytes
/// A boundary's bytes in page-locked host RAM (cuMemHostAlloc), freed with cuMemFreeHost.
/// Page-locked host bytes (a boundary's RAM copy). fwdcheck-digest (host-audit row 13: a fresh 3.4 GB cuMemHostAlloc per
/// boundary, 19 per outer loop, freed again at hexad_close): `cap` is the allocation (>= len); bytes drawn for a session
/// in a retained primary context (`pool` = that context) go back to the process pin pool on drop instead of
/// cuMemFreeHost, and the next session (the next outer loop's HexAD) takes them. HEXAD_PIN_POOL=0: allocated and freed
/// per boundary, as before. Only where the bytes live changes; every byte written and read is the same.
pub struct HostBytes { mem: Option<crate::pinned::PinMem>, len: usize, pool: usize }
unsafe impl Send for HostBytes {}
unsafe impl Sync for HostBytes {}
impl HostBytes {
    /// host-leftovers-2: a buffer made ahead (the CLI window's next boundary), not from the pool.
    pub(crate) fn from_mem(m: crate::pinned::PinMem, len: usize) -> Self { Self { mem: Some(m), len, pool: 0 } }
    /// step-hold: a buffer made ahead on a helper thread that goes back to the pool of `ctx` when dropped, as a
    /// pooled one does (0: freed on drop).
    pub(crate) fn from_mem_pooled(m: crate::pinned::PinMem, len: usize, ctx: usize) -> Self { Self { mem: Some(m), len, pool: ctx } }
    /// fwdcheck-digest: `len` bytes, from the pin pool of `ctx` (0: no pool, freed on drop) when it holds a buffer that
    /// large, else newly allocated (pooled buffers are rounded up to PIN_GRANULE so the next outer loop's sizes fit).
    /// host-leftovers-2: a new buffer is a crate::pinned::PinMem (huge pages first-touched on every core, then
    /// cuMemHostRegister) instead of one cuMemHostAlloc on one thread; AD_PIN_ALLOC=driver keeps cuMemHostAlloc.
    pub(crate) unsafe fn pooled(cuda: &Cuda, ctx: usize, len: usize) -> Result<Self, String> {
        if ctx != 0 { if let Some(m) = pin_take(ctx, len) { return Ok(Self { mem: Some(m), len, pool: ctx }); } }
        let cap = if ctx != 0 { len.max(1).div_ceil(PIN_GRANULE) * PIN_GRANULE } else { len.max(1) };
        let t = Instant::now();
        let m = crate::pinned::PinMem::new(crate::pinned::PinFns::of(cuda), cap)?;
        if let Ok(mut g) = PIN_POOL.lock() { g.allocated += 1; g.alloc_s += t.elapsed().as_secs_f64(); if m.registered_kind() { g.registered += 1; } }
        Ok(Self { mem: Some(m), len, pool: ctx })
    }
    pub fn as_slice(&self) -> &[u8] { unsafe { std::slice::from_raw_parts(self.mem.as_ref().unwrap().ptr(), self.len) } }
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] { unsafe { std::slice::from_raw_parts_mut(self.mem.as_ref().unwrap().ptr(), self.len) } }
}
impl Drop for HostBytes {
    fn drop(&mut self) {
        let Some(m) = self.mem.take() else { return };
        if self.pool != 0 { if let Err(m) = pin_keep(self.pool, m) { drop(m); } } else { drop(m); }
    }
}
/// fwdcheck-digest: the process pin pool. Buffers belong to one primary context, which the pool retains for the life of
/// the process (so they stay valid between sessions); idle bytes are capped at the largest session RAM budget seen.
struct PinPool { ctx: usize, bufs: Vec<crate::pinned::PinMem>, bytes: usize, max: usize, allocated: u64, reused: u64, alloc_s: f64, prefilled: u64, prefill_s: f64, registered: u64 }
static PIN_POOL: std::sync::Mutex<PinPool> = std::sync::Mutex::new(PinPool { ctx: 0, bufs: Vec::new(), bytes: 0, max: 0, allocated: 0, reused: 0, alloc_s: 0.0, prefilled: 0, prefill_s: 0.0, registered: 0 });
/// fwdcheck-digest: the first session of a process (the pool is empty) gets `want` page-locked buffers of `cap` bytes
/// allocated ahead on a helper thread (context `ctx` made current there) into the pool, while open and the first steps
/// run; a store that finds none ready allocates inline as before. Later sessions reuse the pool and prefill nothing.
unsafe fn pin_prefill(cuda: &Cuda, ctx: usize, cap: usize, want: usize) {
    // opt-in (HEXAD_PIN_PREFILL=1): on a shared card (D6) the helper allocations did not shorten outer loop 1 and the one
    // inline allocation beside them took 4.7 s (driver serialisation); not default until measured on an empty card.
    // host-leftovers-2: with PinMem each inline allocation is ~0.19 s (a test machine), so the default stays off.
    if want == 0 || std::env::var("HEXAD_PIN_PREFILL").as_deref() != Ok("1") { return; }
    { let Ok(g) = PIN_POOL.lock() else { return }; if g.ctx != ctx || !g.bufs.is_empty() || g.allocated > 0 || g.prefilled > 0 { return; } }
    #[link(name = "dl")] extern "C" { fn dlopen(f: *const c_char, fl: c_int) -> *mut c_void; fn dlsym(h: *mut c_void, s: *const c_char) -> *mut c_void; }
    let lib = dlopen(c"libcuda.so.1".as_ptr(), 2);
    if lib.is_null() { return; }
    let set = dlsym(lib, c"cuCtxSetCurrent".as_ptr());
    if set.is_null() { return; }
    let set: unsafe extern "C" fn(*mut c_void) -> i32 = std::mem::transmute(set);
    let fns = crate::pinned::PinFns::of(cuda);
    let _ = std::thread::Builder::new().name("hexad-pin-prefill".into()).spawn(move || {
        if set(ctx as *mut c_void) != 0 { return; }
        for _ in 0..want {
            let t = Instant::now();
            let Ok(m) = crate::pinned::PinMem::new(fns, cap) else { return };
            let s = t.elapsed().as_secs_f64();
            if let Ok(mut g) = PIN_POOL.lock() { g.prefilled += 1; g.prefill_s += s; }
            if pin_keep(ctx, m).is_err() { return; }
        }
    });
}
/// speed storm3: the pin pool of a checkpointed CLI window (window.rs BS_POOL): the key to draw with, or 0.
/// Through a sweep the pool keeps every dropped checkpoint's buffer (up to the window's host-memory budget for
/// boundaries), so a store finds one; when the sweep ends the idle ones are freed (pin_pool_release_idle), because
/// what follows the sweep (the dots, which bring whole states back) is where a pool that kept them raised the peak:
/// 59-step 3 km storm window, speed/fast/storm3 runs/fast-4: 225.88 GiB with every buffer kept to the end (o59-i-p5)
/// against 207.33 with no pool (o59-i-p5-nopool). A pool of one idle buffer kept the peak but left 31.8 s of stores
/// in the sweep (fast-3 o59-q-p7-rev5) where keeping all left 9.2 s. Only where the bytes live changes.
/// Tried and taken out: a helper thread that made the buffers of the forward-sensitivity pass's stores ahead. Those
/// steps are not slow in the plain recipe (fast-4 o59-i-p5: one step of 59 over 3.2 s, the first), and the helper made
/// the first step 3 s longer (o59-k-p6: 10.1 s against 6.7).
pub(crate) fn pin_pool_for_checkpoints(ctx: usize, budget: usize) -> usize { pin_open(ctx, budget) }
/// speed storm3: free the idle buffers of the pool of `ctx`; (buffers, bytes) freed. The pool stays open.
/// drains (AD_FREE_BESIDE): the idle buffers of the pool of `ctx` taken out, for the caller to release
/// (pin_pool_release_idle's take, without the drop).
pub(crate) fn pin_pool_take_idle(ctx: usize) -> Vec<crate::pinned::PinMem> {
    let Ok(mut g) = PIN_POOL.lock() else { return Vec::new() };
    if ctx == 0 || g.ctx != ctx { return Vec::new(); }
    g.bytes = 0;
    std::mem::take(&mut g.bufs)
}
pub(crate) fn pin_pool_release_idle(ctx: usize) -> (usize, usize) {
    let bufs = {
        let Ok(mut g) = PIN_POOL.lock() else { return (0, 0) };
        if ctx == 0 || g.ctx != ctx { return (0, 0); }
        g.bytes = 0;
        std::mem::take(&mut g.bufs)
    };
    let (n, bytes) = (bufs.len(), bufs.iter().map(|m| m.cap()).sum());
    drop(bufs);
    (n, bytes)
}
/// bs-tape (a12 backstep): the largest idle buffer of the pool of `ctx` holding at least `min` bytes, taken out of the
/// pool (the streamed tape borrows it for a chunk instead of page-locking a new buffer with the card idle).
pub(crate) fn pin_pool_take_largest(ctx: usize, min: usize) -> Option<crate::pinned::PinMem> {
    let mut g = PIN_POOL.lock().ok()?;
    if ctx == 0 || g.ctx != ctx { return None; }
    let i = g.bufs.iter().enumerate().filter(|(_, b)| b.cap() >= min).max_by_key(|(_, b)| b.cap()).map(|(i, _)| i)?;
    let b = g.bufs.swap_remove(i);
    g.bytes -= b.cap(); g.reused += 1;
    Some(b)
}
/// (idle buffers, idle bytes) in the pool of `ctx` now.
pub(crate) fn pin_pool_idle(ctx: usize) -> (usize, usize) {
    PIN_POOL.lock().ok().filter(|g| ctx != 0 && g.ctx == ctx).map(|g| (g.bufs.len(), g.bytes)).unwrap_or((0, 0))
}
/// bs-tape: a borrowed buffer back into the pool of `ctx` (freed when the pool is full or closed).
pub(crate) fn pin_pool_give(ctx: usize, m: crate::pinned::PinMem) { if let Err(m) = pin_keep(ctx, m) { drop(m); } }
const PIN_GRANULE: usize = 64 << 20;
/// step-hold: the capacity a pooled buffer of `len` bytes is made with (rounded up to the pool's granule).
pub(crate) fn pin_cap(len: usize) -> usize { len.max(1).div_ceil(PIN_GRANULE) * PIN_GRANULE }
fn pin_pool_on() -> bool { std::env::var("HEXAD_PIN_POOL").as_deref() != Ok("0") }
/// The pool key for a session in primary context `ctx` with RAM budget `budget` (0: this session draws no pool bytes).
fn pin_open(ctx: usize, budget: usize) -> usize {
    let Ok(mut g) = PIN_POOL.lock() else { return 0 };
    if g.ctx == 0 { g.ctx = ctx; }
    if g.ctx != ctx { return 0; }
    g.max = g.max.max(budget);
    ctx
}
fn pin_take(ctx: usize, len: usize) -> Option<crate::pinned::PinMem> {
    let mut g = PIN_POOL.lock().ok()?;
    if g.ctx != ctx { return None; }
    let i = g.bufs.iter().enumerate().filter(|(_, b)| b.cap() >= len).min_by_key(|(_, b)| b.cap()).map(|(i, _)| i)?;
    let b = g.bufs.swap_remove(i);
    g.bytes -= b.cap(); g.reused += 1;
    Some(b)
}
/// Into the pool (Ok), or handed back (Err) for the caller to free.
fn pin_keep(ctx: usize, m: crate::pinned::PinMem) -> Result<(), crate::pinned::PinMem> {
    let Ok(mut g) = PIN_POOL.lock() else { return Err(m) };
    if g.ctx != ctx || g.bytes + m.cap() > g.max { return Err(m); }
    g.bytes += m.cap(); g.bufs.push(m);
    Ok(())
}
/// fwdcheck-digest proof counters (HEXAD_FC_VERIFY=1): boundaries compared, bytes compared, bytes differing.
static FC_VERIFY: (std::sync::atomic::AtomicU64, std::sync::atomic::AtomicU64, std::sync::atomic::AtomicU64) =
    (std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0));
/// (allocated, reused, cuMemHostAlloc seconds, idle pool bytes) so far in this process (receipts).
fn pin_stats() -> (u64, u64, f64, usize) { PIN_POOL.lock().map(|g| (g.allocated, g.reused, g.alloc_s, g.bytes)).unwrap_or((0, 0, 0.0, 0)) }
/// host-leftovers-2: buffers made by the fast path (huge pages + cuMemHostRegister) so far in this process (receipts).
fn pin_registered() -> u64 { PIN_POOL.lock().map(|g| g.registered).unwrap_or(0) }
/// (buffers allocated ahead, their cuMemHostAlloc seconds on the helper thread) so far in this process (receipts).
fn pin_prefill_stats() -> (u64, f64) { PIN_POOL.lock().map(|g| (g.prefilled, g.prefill_s)).unwrap_or((0, 0.0)) }

// ------------------------------------------------------------------------------------------------ C types
#[repr(C)] pub struct HexadAlloc { pub id: u64, pub bytes: u64, pub primal: u64, pub shadow: u64, pub flags: u32, pub scale_exp: i32 }
#[repr(C)] pub struct HexadField { pub name: *const c_char, pub dptr: u64, pub dtype: i32, pub nlev: i32, pub nhoriz: i64, pub lev_stride: i64, pub horiz_stride: i64 }
#[repr(C)] pub struct HexadVec { pub kind: i32, pub nfields: i32, pub fields: *const HexadField, pub seed: u64, pub arena: u64, pub words: u64 }
#[repr(C)] #[derive(Default)]
pub struct HexadCounters {
    pub h2d_bytes: [u64; 4], pub d2h_bytes: [u64; 4], pub h2d_calls: [u64; 4], pub d2h_calls: [u64; 4],
    pub ctx_syncs: [u64; 4], pub stream_syncs: [u64; 4], pub event_syncs: [u64; 4],
    pub boundaries_ram: u64, pub boundaries_disk: u64, pub boundary_ram_bytes: u64, pub disk_reads_on_path: u64,
    pub disk_wait_s: f64, pub tl_steps: u64, pub ad_steps: u64,
}
pub const VEC_ZERO: i32 = 0;
pub const VEC_INCREMENT: i32 = 1;
pub const VEC_SEED: i32 = 2;
pub const VEC_ARENA: i32 = 3;
pub const F64: i32 = 0;
pub const F32: i32 = 1;
pub const ALLOC_INITIAL: u32 = 1;
pub const ALLOC_FIXED: u32 = 2;
pub const ALLOC_ADDRESS_BEARING: u32 = 4;
pub const ALLOC_LIVE: u32 = 8;

thread_local! { static LAST: std::cell::RefCell<CString> = std::cell::RefCell::new(CString::new("").unwrap()); }
fn set_last(e: &str) { LAST.with(|l| *l.borrow_mut() = CString::new(e.replace('\0', " ")).unwrap()); }
fn rc(r: Result<(), String>) -> c_int {
    match r { Ok(()) => 0, Err(e) => { let code = if e.starts_with("DEFECT") { -2 } else { -1 }; eprintln!("{}", json!({"event":"hexad_error","error":e})); set_last(&e); code } }
}
fn guard<F: FnOnce() -> Result<(), String>>(f: F) -> c_int {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) { Ok(r) => rc(r), Err(_) => rc(Err("hexad: panic inside the library".into())) }
}
unsafe fn cstr<'a>(p: *const c_char, what: &str) -> Result<&'a str, String> {
    if p.is_null() { return Err(format!("{what}: null")); }
    CStr::from_ptr(p).to_str().map_err(|_| format!("{what}: not UTF-8"))
}
fn sha_file(p: &Path) -> Result<String, String> { fs::read(p).map(|b| crate::exec::digest(&b)).map_err(|e| format!("{}: {e}", p.display())) }
fn jfile(p: &Path) -> Result<Value, String> { serde_json::from_slice(&fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?).map_err(|e| format!("{}: {e}", p.display())) }

// ------------------------------------------------------------------------------------------------ manifest
/// The operator manifest (hexad.operator-manifest/v1), translated into the window's flags. Every file it names carries
/// a sha256 that must match: a manifest names exactly one operator.
pub(crate) struct Manifest { pub sha256: String, pub doc: Value, pub args: Vec<(String, String)>, pub env: Vec<(String, String)>, pub anchors: usize }
pub(crate) fn parse_manifest(path: &Path, kset: &Path, bundle: &Path) -> Result<Manifest, String> {
    let bytes = fs::read(path).map_err(|e| format!("manifest {}: {e}", path.display()))?;
    let sha256 = crate::exec::digest(&bytes);
    let doc: Value = serde_json::from_slice(&bytes).map_err(|e| format!("manifest {}: {e}", path.display()))?;
    if doc["schema"] != "hexad.operator-manifest/v1" { return Err(format!("DEFECT manifest {}: schema is not hexad.operator-manifest/v1", path.display())); }
    let want = |v: &Value, file: &Path, what: &str| -> Result<(), String> {
        let Some(w) = v.as_str() else { return Err(format!("DEFECT manifest: {what} sha256 missing (a manifest must name its operator exactly)")); };
        let got = sha_file(file)?;
        if !got.starts_with(w) || w.len() < 8 { return Err(format!("DEFECT manifest: {what} {} has sha256 {got}, the manifest names {w}", file.display())); }
        Ok(())
    };
    want(&doc["kernel_set"]["kernels_sha256"], &kset.join("kernels.json"), "kernel set")?;
    want(&doc["bundle"]["schemes_sha256"], &bundle.join("schemes.json"), "bundle")?;
    let mut args: Vec<(String, String)> = Vec::new();
    let mut env: Vec<(String, String)> = vec![("AD_SCHEMES".into(), bundle.display().to_string())];
    let fct = doc["fct_frozen"].as_bool().ok_or("DEFECT manifest: fct_frozen must be true or false")?;
    args.push(("fct-frozen".into(), if fct { "1" } else { "0" }.into()));
    let sc = &doc["schemes"];
    if let Some(rows) = sc["rows"].as_str() {
        if sc["rows_sha256"].is_string() { want(&sc["rows_sha256"], Path::new(rows), "scheme rows")?; }
        args.push(("schemes".into(), rows.into()));
    }
    if sc["native"].as_bool() == Some(true) { args.push(("native-schemes".into(), "1".into())); }
    let t = &doc["tempo"];
    match t["mode"].as_str().unwrap_or("hold") {
        "all" => {
            let c = t["cubin"].as_str().ok_or("DEFECT manifest: tempo.mode all needs tempo.cubin")?;
            want(&t["cubin_sha256"], Path::new(c), "TEMPO tangent image")?;
            args.push(("tempo".into(), "all".into())); args.push(("tempo-cubin".into(), c.into()));
            let s = t["serial_cubin"].as_str().ok_or("DEFECT manifest: tempo.mode all needs tempo.serial_cubin")?;
            want(&t["serial_cubin_sha256"], Path::new(s), "TEMPO serial image")?;
            env.push(("AD_DOT_TEMPO_SERIAL_CUBIN".into(), s.into()));
            if let Some(g) = t["arena_gib"].as_f64() { args.push(("tempo-arena-gib".into(), format!("{g}"))); }
        }
        "hold" => {}
        m => return Err(format!("DEFECT manifest: tempo.mode {m}: all or hold")),
    }
    let mut refuse: Vec<String> = Vec::new();
    let r = &doc["radiation"];
    match r["state"].as_str().unwrap_or("") {
        "refuse-rad" => {
            let f = r["refuse_file"].as_str().ok_or("DEFECT manifest: radiation refuse-rad needs refuse_file")?;
            want(&r["refuse_sha256"], Path::new(f), "radiation refuse list")?;
            let text = fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?;
            refuse.extend(text.split(|c: char| c == ',' || c.is_whitespace()).filter(|s| !s.is_empty()).map(str::to_owned));
        }
        "live" | "held" => {}
        s => return Err(format!("DEFECT manifest: radiation.state {s:?}: refuse-rad, live or held")),
    }
    for x in doc["refuse"].as_array().into_iter().flatten().filter_map(|x| x.as_str()) { refuse.push(x.to_owned()); }
    if !refuse.is_empty() { args.push(("refuse".into(), refuse.join(","))); }
    let hold: Vec<&str> = doc["hold"].as_array().into_iter().flatten().filter_map(|x| x.as_str()).collect();
    if !hold.is_empty() { args.push(("hold".into(), hold.join(","))); }
    // torch bridge: "window_args" {word: value} passes the window's own words a made row runs with (local-topology,
    // mesh-constants, ...), so a library session can be opened with the same words as `ad-binder window`. Absent: as
    // before. (It does not fix the open defect: the 30 km made row's library session still stops with a segmentation
    // fault in step 1 of its forward check, after the radiation unpack, with these words and without them: three runs,
    // 2026-11-19, where `ad-binder window` with the same plan, kernel set and bundle runs.)
    for (k, v) in doc["window_args"].as_object().into_iter().flatten() {
        let v = match v { Value::String(x) => x.clone(), x => x.to_string() };
        args.push((k.clone(), v));
    }
    let rn = &doc["runner"];
    for (k, d) in [("tape-gib", 2.0), ("scratch-gib", 1.0)] {
        let key = k.replace('-', "_");
        args.push((k.into(), format!("{}", rn[key.as_str()].as_f64().unwrap_or(d))));
    }
    let anchors = rn["anchors"].as_u64().unwrap_or(2) as usize;
    Ok(Manifest { sha256, doc, args, env, anchors })
}

// ------------------------------------------------------------------------------------------------ scale table
/// Power-of-two scales: runner tangent = 2^tangent_exp * physical, runner cotangent = 2^cotangent_exp * physical dual.
/// Applied only inside the T kernels, so the host sees physical units. One exponent per side for the whole arena: a
/// per-allocation input scale D would make the map Ts D^-1 M D T0, not Ts M T0, because M mixes allocations (the
/// table carries per-allocation rows for the record, and refuses rows that differ).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Scale { pub t: i32, pub a: i32 }
pub(crate) fn parse_scale(doc: &Value) -> Result<Scale, String> {
    if doc["schema"] != "hexad.scale-table/v1" { return Err("DEFECT scale table: schema is not hexad.scale-table/v1".into()); }
    let t = doc["tangent_exp"].as_i64().ok_or("DEFECT scale table: tangent_exp")? as i32;
    let a = doc["cotangent_exp"].as_i64().ok_or("DEFECT scale table: cotangent_exp")? as i32;
    for r in doc["allocations"].as_array().into_iter().flatten() {
        if r["tangent_exp"].as_i64() != Some(t as i64) || r["cotangent_exp"].as_i64() != Some(a as i64) {
            return Err(format!("DEFECT scale table: allocation {} has its own exponent; per-allocation scales change the operator (Ts D^-1 M D T0 != Ts M T0) and are refused", r["id"]));
        }
    }
    if !(-120..=120).contains(&t) || !(-120..=120).contains(&a) { return Err("DEFECT scale table: exponent outside [-120, 120]".into()); }
    Ok(Scale { t, a })
}

// ------------------------------------------------------------------------------------------------ T maps
struct TFns { f: HashMap<&'static str, *mut c_void> }
const TNAMES: &[&str] = &["t_inc_get", "t_inc_put", "t_put", "t_sum", "t_widen", "t_zero", "t_axpy", "t_randn", "t_dot", "t0_cell",
    "t0_ru", "t0_rw", "t0_rw_T_cell", "t0_rw_T_edge", "t0_ru_T_edge", "t0_ru_T_cell", "t0_cell_T", "ts_cell", "ts_cell_T", "t_flux_sign"];
struct KA(Vec<Box<[u8]>>);
impl KA {
    fn new() -> Self { Self(Vec::new()) }
    fn p(mut self, v: u64) -> Self { self.0.push(Box::new(v.to_le_bytes())); self }
    fn l(mut self, v: i64) -> Self { self.0.push(Box::new(v.to_le_bytes())); self }
    fn i(mut self, v: i32) -> Self { self.0.push(Box::new(v.to_le_bytes())); self }
    fn d(mut self, v: f64) -> Self { self.0.push(Box::new(v.to_le_bytes())); self }
}
impl TFns {
    unsafe fn load(cu: &Cuda) -> Result<Self, String> {
        let m = module(cu, T_IMAGE, "hexad T maps")?;
        let mut f = HashMap::new();
        for &n in TNAMES { f.insert(n, function(cu, m, n)?); }
        Ok(Self { f })
    }
    unsafe fn go(&self, cu: &Cuda, name: &str, n: i64, a: KA) -> Result<(), String> {
        let f = *self.f.get(name).ok_or_else(|| format!("no T kernel {name}"))?;
        let blocks = ((n.max(1) + 255) / 256).clamp(1, 4096) as u32;
        let mut a = a;
        let mut p: Vec<*mut c_void> = a.0.iter_mut().map(|b| b.as_mut_ptr().cast()).collect();
        checked(name, (cu.launch)(f, blocks, 1, 1, 256, 1, 1, 0, ptr::null_mut(), p.as_mut_ptr(), ptr::null_mut()))
    }
}
/// One field of the map: its copies' plan locations (allocation, byte offset) and its word count.
#[derive(Clone, Debug)]
struct FMap { canon: (u64, usize), copies: Vec<(u64, usize)>, words: usize }
/// Where a copy lives for the T kernels: the float32 arena (full sessions) or a float64 mirror (the tmap test).
#[derive(Clone, Copy)]
struct Dst { addr: u64, f64: bool }

pub(crate) struct TMaps {
    k: TFns, nc: i64, ne: i64, nl: i64, maxe: i32, rvord: f64, rgas: f64, rcv: f64,
    fields: BTreeMap<String, FMap>, ns: usize, slot_words: usize, species: BTreeMap<String, usize>,
    // coefficients (device)
    zz: u64, fzm: u64, fzp: u64, rbase: u64, rtbase: u64, zbc: u64, zb3c: u64, signs: u64, coe: u64, eoc: u64, noe: u64,
    fzm32: u64, fzp32: u64,
    // primal at boundary 0 (float64) and the frozen terrain-flux sign
    p_thm: u64, p_qv: u64, p_rp: u64, p_rtp: u64, p_ex: u64, p_rz: u64, p_u: u64, p_w: u64, sgnf: u64, primal_bound: bool,
    // work (float64)
    w: BTreeMap<&'static str, u64>, wsc: u64, part: u64, srcs: u64,
    pub doc_sha256: String,
}
const WORK_CELL: &[&str] = &["dth", "drho", "dthm", "drz", "drp", "drtp", "drth", "dex", "dpp", "t_thm", "t_rz", "g_th", "g_rho"];
const WORK_EDGE: &[&str] = &["du", "dru"];
const WORK_W: &[&str] = &["dw", "drw"];
unsafe fn dalloc(cu: &Cuda, bytes: usize) -> Result<u64, String> { let mut p = 0; checked("cuMemAlloc hexad", (cu.allocate)(&mut p, bytes.max(8)))?; dev::zero(cu, p, bytes.max(8))?; Ok(p) }
unsafe fn dup(cu: &Cuda, b: &[u8]) -> Result<u64, String> { let p = dalloc(cu, b.len())?; dev::upload(cu, p, b)?; Ok(p) }
fn f64b(v: &[f64]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

impl TMaps {
    /// Load alloc-map.json and coef.bin (the plan package's adjoint/ directory).
    unsafe fn load(cu: &Cuda, dir: &Path) -> Result<Self, String> {
        let mp = dir.join("alloc-map.json");
        let doc_sha256 = sha_file(&mp)?;
        let doc = jfile(&mp)?;
        if doc["schema"] != "hexad.alloc-map/v1" { return Err("DEFECT alloc map: schema is not hexad.alloc-map/v1".into()); }
        if let Some(p) = doc["problems"].as_array() { for x in p { eprintln!("{}", json!({"event":"hexad_alloc_map_note","note":x})); } }
        let g = |k: &str| doc[k].as_i64().ok_or_else(|| format!("alloc map: {k}"));
        let (nc, ne, nl, maxe) = (g("ncells")?, g("nedges")?, g("nlev")?, g("maxedges")? as i32);
        let c = &doc["constants"];
        let (rgas, rv, cp) = (c["rgas"].as_f64().ok_or("rgas")?, c["rv"].as_f64().ok_or("rv")?, c["cp"].as_f64().ok_or("cp")?);
        let rvord = (rv as f32 / rgas as f32) as f64;   // the forward's float32 constant, as the write-back forms it
        let rcv = (rgas / (cp - rgas)) as f32 as f64;
        let mut fields = BTreeMap::new();
        for (name, f) in doc["fields"].as_object().ok_or("alloc map fields")? {
            let pos = |v: &Value| -> Result<(u64, usize), String> { Ok((v["alloc"].as_u64().ok_or("alloc")?, v["offset"].as_u64().ok_or("offset")? as usize)) };
            let canon = pos(&f["canonical"])?;
            let copies = f["copies"].as_array().ok_or("copies")?.iter().map(pos).collect::<Result<Vec<_>, _>>()?;
            let shape: Vec<usize> = f["canonical"]["shape"].as_array().ok_or("shape")?.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect();
            fields.insert(name.clone(), FMap { canon, copies, words: shape.iter().product() });
        }
        let s = &doc["fields"]["scalars"];
        let ns = s["canonical"]["shape"][0].as_u64().ok_or("scalars shape")? as usize;
        let slot_words = s["slot_words"].as_u64().ok_or("slot_words")? as usize;
        let species: BTreeMap<String, usize> = s["species_slot"].as_object().ok_or("species_slot")?.iter()
            .map(|(k, v)| (k.clone(), v.as_u64().unwrap_or(0) as usize)).collect();
        if species.get("qv") != Some(&0) && !species.contains_key("qv") { return Err("DEFECT alloc map: no qv slot".into()); }
        let ce = &doc["coefficients"];
        let blob = fs::read(dir.join(ce["file"].as_str().ok_or("coef file")?)).map_err(|e| format!("coef.bin: {e}"))?;
        if crate::exec::digest(&blob) != ce["sha256"].as_str().unwrap_or("") { return Err("DEFECT alloc map: coef.bin sha256 differs from the map".into()); }
        let ent = |k: &str| -> Result<u64, String> {
            let e = &ce["entries"][k]; let off = e["offset"].as_u64().ok_or_else(|| format!("coef {k}"))? as usize;
            let n = e["count"].as_u64().ok_or("count")? as usize * if e["dtype"] == "i32" { 4 } else { 8 };
            dup(cu, &blob[off..off + n])
        };
        let f32dev = |k: &str| -> Result<u64, String> {
            let e = &ce["entries"][k]; let off = e["offset"].as_u64().ok_or("off")? as usize; let n = e["count"].as_u64().ok_or("n")? as usize;
            let v: Vec<u8> = blob[off..off + 8 * n].chunks_exact(8).flat_map(|c| (f64::from_le_bytes(c.try_into().unwrap()) as f32).to_le_bytes()).collect();
            dup(cu, &v)
        };
        let (cw, ew, ww) = ((nl * nc) as usize, (nl * ne) as usize, ((nl + 1) * nc) as usize);
        let mut w = BTreeMap::new();
        for &n in WORK_CELL { w.insert(n, dalloc(cu, cw * 8)?); }
        for &n in WORK_EDGE { w.insert(n, dalloc(cu, ew * 8)?); }
        for &n in WORK_W { w.insert(n, dalloc(cu, ww * 8)?); }
        let sgnf = dalloc(cu, ww * maxe as usize * 8)?;
        let t = Self { k: TFns::load(cu)?, nc, ne, nl, maxe, rvord, rgas, rcv, fields, ns, slot_words, species,
            zz: ent("zz")?, fzm: ent("fzm")?, fzp: ent("fzp")?, rbase: ent("rho_base")?, rtbase: ent("rtheta_base")?,
            zbc: ent("zb_cell")?, zb3c: ent("zb3_cell")?, signs: ent("signs")?, coe: ent("cellsOnEdge")?, eoc: ent("edgesOnCell")?, noe: ent("nEdgesOnCell")?,
            fzm32: f32dev("fzm")?, fzp32: f32dev("fzp")?,
            p_thm: dalloc(cu, cw * 8)?, p_qv: dalloc(cu, cw * 8)?, p_rp: dalloc(cu, cw * 8)?, p_rtp: dalloc(cu, cw * 8)?, p_ex: dalloc(cu, cw * 8)?,
            p_rz: dalloc(cu, cw * 8)?, p_u: dalloc(cu, ew * 8)?, p_w: dalloc(cu, ww * 8)?, sgnf, primal_bound: false,
            w, wsc: dalloc(cu, ns * slot_words * 8)?, part: dalloc(cu, 1024 * 8)?, srcs: dalloc(cu, 64 * 8)?, doc_sha256 };
        for n in ["theta_m", "rho_zz", "rho_p", "rtheta_p", "exner", "u", "w", "ru", "scalars"] {
            if !t.fields.contains_key(n) { return Err(format!("DEFECT alloc map: no field {n}")); }
        }
        Ok(t)
    }
    fn fm(&self, n: &str) -> &FMap { &self.fields[n] }
    /// Bind the boundary-0 primal from float32 words at the given addresses (live allocations, or test buffers).
    unsafe fn bind_primal(&mut self, cu: &Cuda, at: &dyn Fn(u64, usize) -> Result<u64, String>) -> Result<(), String> {
        let cw = self.nl * self.nc;
        let get = |n: &str| -> Result<u64, String> { let f = &self.fields[n]; at(f.canon.0, f.canon.1) };
        let (thm, rp, rtp, ex, rz, u, w, ru, sc) = (get("theta_m")?, get("rho_p")?, get("rtheta_p")?, get("exner")?, get("rho_zz")?, get("u")?, get("w")?, get("ru")?, get("scalars")?);
        let qv = sc + 4 * (self.species["qv"] * self.slot_words) as u64;
        for (dst, src, n) in [(self.p_thm, thm, cw), (self.p_qv, qv, cw), (self.p_rp, rp, cw), (self.p_rtp, rtp, cw), (self.p_ex, ex, cw),
                              (self.p_rz, rz, cw), (self.p_u, u, self.nl * self.ne), (self.p_w, w, (self.nl + 1) * self.nc)] {
            self.k.go(cu, "t_widen", n, KA::new().p(dst).p(src).l(n))?;
        }
        self.k.go(cu, "t_flux_sign", (self.nl + 1) * self.nc, KA::new().l(self.nl).l(self.nc).l(self.ne).i(self.maxe).p(self.eoc).p(self.noe)
            .p(self.fzm32).p(self.fzp32).p(ru).p(self.sgnf))?;
        dev::sync(cu)?;
        self.primal_bound = true;
        Ok(())
    }
    fn field_words(&self, name: &str) -> Result<(i64, i64), String> {
        Ok(match name {
            "theta" | "rho" => (self.nl, self.nc),
            "u" => (self.nl, self.ne),
            "w" => (self.nl + 1, self.nc),
            "ps" | "surface_pressure" => return Err("DEFECT increment field ps: the hex write-back consumes rho (MPAS atm_init_coupled_diagnostics), not surface pressure; the ps -> rho hydrostatic map is a linear variable change and belongs on the calling host's side of HexAD".into()),
            s if self.species.contains_key(s) => (self.nl, self.nc),
            s => return Err(format!("DEFECT increment field {s}: not a write-back input of this alloc map (theta, rho, u, w, {})", self.species.keys().cloned().collect::<Vec<_>>().join(", "))),
        })
    }
    unsafe fn check_field(&self, f: &HexadField) -> Result<String, String> {
        let name = cstr(f.name, "field name")?.to_owned();
        let (nl, nh) = self.field_words(&name)?;
        if f.nlev as i64 != nl || f.nhoriz != nh { return Err(format!("DEFECT increment field {name}: {} x {} given, the map has {nl} x {nh}", f.nlev, f.nhoriz)); }
        if f.dptr == 0 { return Err(format!("increment field {name}: null device pointer")); }
        if f.dtype != F64 && f.dtype != F32 { return Err(format!("increment field {name}: dtype {}", f.dtype)); }
        Ok(name)
    }
    fn work_of(&self, name: &str) -> u64 {
        match name { "theta" => self.w["dth"], "rho" => self.w["drho"], "u" => self.w["du"], "w" => self.w["dw"],
                     s => self.wsc + 8 * (self.species[s] * self.slot_words) as u64 }
    }
    unsafe fn zero_inputs(&self, cu: &Cuda) -> Result<(), String> {
        for (b, n) in [(self.w["dth"], self.nl * self.nc), (self.w["drho"], self.nl * self.nc), (self.w["du"], self.nl * self.ne),
                       (self.w["dw"], (self.nl + 1) * self.nc), (self.wsc, (self.ns * self.slot_words) as i64)] {
            self.k.go(cu, "t_zero", n, KA::new().p(b).l(n))?;
        }
        Ok(())
    }
    /// increment fields -> work (scale 1)
    unsafe fn gather(&self, cu: &Cuda, v: &HexadVec) -> Result<(), String> {
        self.zero_inputs(cu)?;
        for f in std::slice::from_raw_parts(v.fields, v.nfields.max(0) as usize) {
            let name = self.check_field(f)?;
            self.k.go(cu, "t_inc_get", f.nlev as i64 * f.nhoriz, KA::new().p(self.work_of(&name)).p(f.dptr).i((f.dtype == F64) as i32)
                .l(f.nlev as i64).l(f.nhoriz).l(f.lev_stride).l(f.horiz_stride).d(1.0))?;
        }
        Ok(())
    }
    /// work -> increment fields (assign or add)
    unsafe fn scatter(&self, cu: &Cuda, v: &HexadVec, scale: f64, add: bool) -> Result<(), String> {
        for f in std::slice::from_raw_parts(v.fields, v.nfields.max(0) as usize) {
            let name = self.check_field(f)?;
            self.k.go(cu, "t_inc_put", f.nlev as i64 * f.nhoriz, KA::new().p(f.dptr).i((f.dtype == F64) as i32).p(self.work_of(&name))
                .l(f.nlev as i64).l(f.nhoriz).l(f.lev_stride).l(f.horiz_stride).d(scale).i(add as i32))?;
        }
        Ok(())
    }
    unsafe fn put(&self, cu: &Cuda, n: &str, work: u64, dst: &dyn Fn(u64, usize) -> Result<Dst, String>, scale: f64, canon_only: bool, add: bool) -> Result<(), String> {
        let f = self.fm(n);
        let list: Vec<(u64, usize)> = if canon_only { vec![f.canon] } else { f.copies.clone() };
        for (a, o) in list {
            let d = dst(a, o)?;
            self.k.go(cu, "t_put", f.words as i64, KA::new().p(d.addr).i(d.f64 as i32).p(work).l(f.words as i64).d(scale).i(add as i32))?;
        }
        Ok(())
    }
    unsafe fn sum(&self, cu: &Cuda, n: &str, work: u64, dst: &dyn Fn(u64, usize) -> Result<Dst, String>, scale: f64, canon_only: bool) -> Result<(), String> {
        let f = self.fm(n);
        let list: Vec<(u64, usize)> = if canon_only { vec![f.canon] } else { f.copies.clone() };
        if list.len() > 64 { return Err(format!("field {n}: {} copies (at most 64)", list.len())); }
        let mut ptrs = Vec::new(); let mut is64 = None;
        for (a, o) in &list { let d = dst(*a, *o)?; if is64.is_some_and(|x| x != d.f64) { return Err("mixed copy types".into()); } is64 = Some(d.f64); ptrs.push(d.addr); }
        // one small descriptor upload per field (8 bytes a copy, counted as glue)
        dev::upload(cu, self.srcs, &ptrs.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
        self.k.go(cu, "t_sum", f.words as i64, KA::new().p(work).p(self.srcs).i(ptrs.len() as i32).i(is64.unwrap_or(false) as i32).l(f.words as i64).d(scale))?;
        dev::sync(cu)   // srcs is reused by the next field
    }
    /// T0: increment -> every copy (assign), tangent scale 2^e.
    unsafe fn t0(&self, cu: &Cuda, v: &HexadVec, dst: &dyn Fn(u64, usize) -> Result<Dst, String>, e: i32) -> Result<(), String> {
        if !self.primal_bound { return Err("T0: boundary-0 primal not bound".into()); }
        self.gather(cu, v)?;
        let w = &self.w; let (nl, nc, ne) = (self.nl, self.nc, self.ne);
        let dqv = self.work_of("qv");
        self.k.go(cu, "t0_cell", nl * nc, KA::new().l(nl).l(nc).d(self.rvord).d(self.rgas).d(self.rcv)
            .p(self.p_thm).p(self.p_qv).p(self.p_rp).p(self.p_rtp).p(self.p_ex).p(self.zz).p(self.rbase).p(self.rtbase)
            .p(w["dth"]).p(dqv).p(w["drho"]).p(w["dthm"]).p(w["drz"]).p(w["drp"]).p(w["drtp"]).p(w["drth"]).p(w["dex"]).p(w["dpp"]))?;
        self.k.go(cu, "t0_ru", nl * ne, KA::new().l(nl).l(ne).l(nc).p(self.coe).p(self.p_rz).p(self.p_u).p(w["du"]).p(w["drz"]).p(w["dru"]))?;
        self.k.go(cu, "t0_rw", (nl + 1) * nc, KA::new().l(nl).l(nc).l(ne).i(self.maxe).p(self.eoc).p(self.noe).p(self.fzm).p(self.fzp).p(self.zz)
            .p(self.p_rz).p(self.p_w).p(self.zbc).p(self.zb3c).p(self.signs).p(self.sgnf).p(w["dw"]).p(w["drz"]).p(w["dru"]).p(w["drw"]))?;
        let s = 2f64.powi(e);
        for (n, wk) in [("theta_m", "dthm"), ("rho_zz", "drz"), ("rho_p", "drp"), ("rtheta_p", "drtp"), ("rho_theta", "drth"), ("exner", "dex"),
                        ("pressure_p", "dpp"), ("ru", "dru"), ("rw", "drw"), ("u", "du"), ("w", "dw"), ("theta", "dth"), ("rho", "drho")] {
            if self.fields.contains_key(n) { self.put(cu, n, w[wk], dst, s, false, false)?; }
        }
        self.put(cu, "scalars", self.wsc, dst, s, false, false)?;
        dev::sync(cu)
    }
    /// T0^T: every copy's cotangent (scale 2^-e) -> increment (ADD).
    unsafe fn t0t(&self, cu: &Cuda, v: &HexadVec, dst: &dyn Fn(u64, usize) -> Result<Dst, String>, e: i32) -> Result<(), String> {
        if !self.primal_bound { return Err("T0^T: boundary-0 primal not bound".into()); }
        let w = &self.w; let (nl, nc, ne) = (self.nl, self.nc, self.ne);
        let s = 2f64.powi(-e);
        let zero = |b: u64, n: i64| self.k.go(cu, "t_zero", n, KA::new().p(b).l(n));
        for (n, wk) in [("theta_m", "dthm"), ("rho_zz", "drz"), ("rho_p", "drp"), ("rtheta_p", "drtp"), ("rho_theta", "drth"), ("exner", "dex"),
                        ("pressure_p", "dpp"), ("ru", "dru"), ("rw", "drw"), ("u", "du"), ("w", "dw"), ("theta", "dth"), ("rho", "drho")] {
            if self.fields.contains_key(n) { self.sum(cu, n, w[wk], dst, s, false)?; }
            else { let words = if wk == "du" || wk == "dru" { nl * ne } else if wk == "dw" || wk == "drw" { (nl + 1) * nc } else { nl * nc }; zero(w[wk], words)?; }
        }
        self.sum(cu, "scalars", self.wsc, dst, s, false)?;
        self.k.go(cu, "t0_rw_T_cell", (nl + 1) * nc, KA::new().l(nl).l(nc).p(self.fzm).p(self.fzp).p(self.zz).p(self.p_rz).p(self.p_w)
            .p(w["drw"]).p(w["dw"]).p(w["drz"]))?;
        self.k.go(cu, "t0_rw_T_edge", nl * ne, KA::new().l(nl).l(nc).l(ne).i(self.maxe).p(self.coe).p(self.eoc).p(self.noe).p(self.fzm).p(self.fzp)
            .p(self.zz).p(self.zbc).p(self.zb3c).p(self.signs).p(self.sgnf).p(w["drw"]).p(w["dru"]))?;
        self.k.go(cu, "t0_ru_T_edge", nl * ne, KA::new().l(nl).l(ne).l(nc).p(self.coe).p(self.p_rz).p(w["dru"]).p(w["du"]))?;
        self.k.go(cu, "t0_ru_T_cell", nl * nc, KA::new().l(nl).l(nc).l(ne).i(self.maxe).p(self.eoc).p(self.noe).p(self.p_u).p(w["dru"]).p(w["drz"]))?;
        let dqv = self.work_of("qv");
        self.k.go(cu, "t0_cell_T", nl * nc, KA::new().l(nl).l(nc).d(self.rvord).d(self.rgas).d(self.rcv)
            .p(self.p_thm).p(self.p_qv).p(self.p_rp).p(self.p_rtp).p(self.p_ex).p(self.zz).p(self.rbase).p(self.rtbase)
            .p(w["dthm"]).p(w["drz"]).p(w["drp"]).p(w["drtp"]).p(w["drth"]).p(w["dex"]).p(w["dpp"]).p(w["dth"]).p(dqv).p(w["drho"]))?;
        self.scatter(cu, v, 1.0, true)?;
        dev::sync(cu)
    }
    /// Ts: canonical tangents at boundary s (scale 2^-e) -> increment (assign). thm32/qv32: boundary-s float32 primal.
    unsafe fn ts(&self, cu: &Cuda, v: &HexadVec, dst: &dyn Fn(u64, usize) -> Result<Dst, String>, e: i32, thm32: u64, qv32: u64) -> Result<(), String> {
        let w = &self.w; let n = self.nl * self.nc; let s = 2f64.powi(-e);
        self.sum(cu, "theta_m", w["t_thm"], dst, s, true)?;
        self.sum(cu, "rho_zz", w["t_rz"], dst, s, true)?;
        self.sum(cu, "u", w["du"], dst, s, true)?;
        self.sum(cu, "w", w["dw"], dst, s, true)?;
        self.sum(cu, "scalars", self.wsc, dst, s, true)?;
        self.k.go(cu, "ts_cell", n, KA::new().l(n).d(self.rvord).p(thm32).p(qv32).p(self.zz).p(w["t_thm"]).p(self.work_of("qv")).p(w["t_rz"]).p(w["dth"]).p(w["drho"]))?;
        self.scatter(cu, v, 1.0, false)?;
        dev::sync(cu)
    }
    /// The nonlinear state at a boundary into increment-kind fields (assign): the canonical copies read from the live
    /// primal allocations (dst), theta = theta_m / (1 + rvord qv) and rho = rho_zz zz by ts_cell with the primal as its
    /// tangent and a zero qv tangent (so Ts is exactly the linearisation of this map), u, w and the species as stored.
    unsafe fn state(&self, cu: &Cuda, v: &HexadVec, dst: &dyn Fn(u64, usize) -> Result<Dst, String>, thm32: u64, qv32: u64) -> Result<(), String> {
        let w = &self.w; let n = self.nl * self.nc;
        self.sum(cu, "theta_m", w["t_thm"], dst, 1.0, true)?;
        self.sum(cu, "rho_zz", w["t_rz"], dst, 1.0, true)?;
        self.sum(cu, "u", w["du"], dst, 1.0, true)?;
        self.sum(cu, "w", w["dw"], dst, 1.0, true)?;
        self.sum(cu, "scalars", self.wsc, dst, 1.0, true)?;
        self.k.go(cu, "t_zero", n, KA::new().p(w["dpp"]).l(n))?;
        self.k.go(cu, "ts_cell", n, KA::new().l(n).d(self.rvord).p(thm32).p(qv32).p(self.zz).p(w["t_thm"]).p(w["dpp"]).p(w["t_rz"]).p(w["dth"]).p(w["drho"]))?;
        self.scatter(cu, v, 1.0, false)?;
        dev::sync(cu)
    }
    /// Ts^T: increment cotangent -> canonical copies (ADD, scale 2^e).
    unsafe fn tst(&self, cu: &Cuda, v: &HexadVec, dst: &dyn Fn(u64, usize) -> Result<Dst, String>, e: i32, thm32: u64, qv32: u64) -> Result<(), String> {
        let w = &self.w; let n = self.nl * self.nc; let s = 2f64.powi(e);
        self.gather(cu, v)?;
        self.k.go(cu, "ts_cell_T", n, KA::new().l(n).d(self.rvord).p(thm32).p(qv32).p(self.zz).p(w["dth"]).p(w["drho"]).p(w["t_thm"]).p(self.work_of("qv")).p(w["t_rz"]))?;
        self.put(cu, "theta_m", w["t_thm"], dst, s, true, true)?;
        self.put(cu, "rho_zz", w["t_rz"], dst, s, true, true)?;
        self.put(cu, "u", w["du"], dst, s, true, true)?;
        self.put(cu, "w", w["dw"], dst, s, true, true)?;
        self.put(cu, "scalars", self.wsc, dst, s, true, true)?;
        dev::sync(cu)
    }
    unsafe fn dot(&self, cu: &Cuda, a: u64, b: u64, n: i64) -> Result<f64, String> {
        let f = self.k.f["t_dot"];
        let mut ka = KA::new().p(a).p(b).l(n).p(self.part);
        let mut p: Vec<*mut c_void> = ka.0.iter_mut().map(|b| b.as_mut_ptr().cast()).collect();
        checked("t_dot", (cu.launch)(f, 1024, 1, 1, 256, 1, 1, 0, ptr::null_mut(), p.as_mut_ptr(), ptr::null_mut()))?;
        let h = dev::download(cu, self.part, 1024 * 8)?;
        Ok(h.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).sum())
    }
    /// Copies of every field, by plan location (the f64 mirror of the tmap test).
    fn all_copies(&self) -> Vec<(u64, usize, usize)> {
        let mut v: Vec<(u64, usize, usize)> = self.fields.values().flat_map(|f| f.copies.iter().map(move |&(a, o)| (a, o, f.words))).collect();
        v.sort(); v.dedup(); v
    }
}

// ------------------------------------------------------------------------------------------------ session
enum Mode { Idle, Tl, Ad }

pub struct Session {
    // order matters for Drop: boundaries (pinned host) and maps before the Execution (context)
    bounds: Vec<Boundary>, flats: Vec<Vec<Flat>>, prim: Vec<(u64, u64)>,
    tm: Option<TMaps>, w: Option<Win>, frame: Option<ReplayFrame>, refs: References, ex: Execution, plan: Plan, base: PathBuf,
    steps: usize, anchors: usize, bdir: PathBuf, manifest: Manifest, scale: Scale,
    ram_budget: usize, ram_used: usize, disk_bounds: u64, disk_reads: u64, disk_wait: f64,
    tl_s: usize, ad_s: usize, mode: Mode, tl_steps: u64, ad_steps: u64,
    /// fold-measure: the window tape was sized from the card before the first reverse (once per session)
    tape_sized: bool,
    check_every: bool, receipt: Value, primary: Option<(i32, unsafe extern "C" fn(i32) -> i32)>,
    /// fwdcheck-digest: this session's pin-pool key (its retained primary context; 0 = no pool)
    pin_ctx: usize,
    // tmap-only test sessions: the f64 mirror of every copy, the boundary-0 float32 primal words
    mirror: BTreeMap<(u64, usize), u64>, tbufs: Vec<u64>, tmap_only: bool,
    /// disk-tier boundaries read back (least recently used first, at most 4)
    dcache: Vec<(usize, crate::window::Bytes)>,
}
impl Drop for Session {
    fn drop(&mut self) {
        unsafe {
            let cu = &self.ex.cuda as *const Cuda;
            let _ = ((*cu).context_synchronize)();
            self.bounds.clear(); self.flats.clear();
            for (a, b) in self.prim.drain(..) { let _ = ((*cu).free)(a); let _ = ((*cu).free)(b); }
            for (_, p) in std::mem::take(&mut self.mirror) { let _ = ((*cu).free)(p); }
            for p in self.tbufs.drain(..) { let _ = ((*cu).free)(p); }
            self.dcache.clear();
            self.tm = None;
            if let Some(w) = self.w.take() {
                // the window's own device buffers (the arena, tape, scratch, dot temporaries, TEMPO arena): freed here so a
                // primary-context session (whose context outlives it) does not keep them
                for p in [w.sh_base, w.tape.base, w.scratch.base, w.tmp[0], w.tmp[1], w.tmp[2], w.part, w.fail] { if p != 0 { let _ = ((*cu).free)(p); } }
                if let Some((_, ar)) = w.tempo.as_ref() { let _ = ((*cu).free)(ar.base); }
                for (p, _) in &w.dev_anchors { let _ = ((*cu).free)(*p); }
                drop(w);
            }
            if let Some((dev, release)) = self.primary.take() { self.ex.context = ptr::null_mut(); let _ = release(dev); }
        }
    }
}
/// Bytes of every allocation a step makes (mid-step Allocate ids persist and are part of every boundary).
fn plan_mid_bytes(plan: &Plan) -> usize {
    let mut m: BTreeMap<u64, usize> = BTreeMap::new();
    let init: std::collections::HashSet<u64> = plan.allocations.iter().map(|a| a.id).collect();
    for ops in plan.blocks.values() { for op in ops { if let Operation::Allocate { id, bytes, .. } = op { if !init.contains(id) { let e = m.entry(*id).or_insert(0); *e = (*e).max(*bytes); } } } }
    m.values().sum()
}
/// cuMemGetInfo free bytes of the current context.
unsafe fn mem_free() -> Option<usize> {
    #[link(name = "dl")] extern "C" { fn dlopen(f: *const c_char, fl: c_int) -> *mut c_void; fn dlsym(h: *mut c_void, s: *const c_char) -> *mut c_void; }
    let lib = dlopen(c"libcuda.so.1".as_ptr(), 2);
    if lib.is_null() { return None; }
    let f = dlsym(lib, c"cuMemGetInfo_v2".as_ptr());
    if f.is_null() { return None; }
    let g: unsafe extern "C" fn(*mut usize, *mut usize) -> i32 = std::mem::transmute(f);
    let (mut free, mut total) = (0usize, 0usize);
    if g(&mut free, &mut total) == 0 { Some(free) } else { None }
}
fn sx(s: u64) -> u64 { 0x5eed_0000_0000 + 2 * s + 1 }
fn sy(s: u64) -> u64 { 0x5eed_0000_0000 + 2 * s + 2 }

/// Retain the device's primary context and make it current, so device pointers from a CUDA runtime client (the host's
/// resident mpas_fields) are valid in the runner's kernels without a copy.
unsafe fn use_primary(ex: &mut Execution) -> Result<(i32, unsafe extern "C" fn(i32) -> i32), String> {
    #[link(name = "dl")] extern "C" { fn dlopen(f: *const c_char, fl: c_int) -> *mut c_void; fn dlsym(h: *mut c_void, s: *const c_char) -> *mut c_void; }
    let lib = dlopen(c"libcuda.so.1".as_ptr(), 2);
    if lib.is_null() { return Err("libcuda.so.1".into()); }
    let sym = |n: &CStr| { let p = dlsym(lib, n.as_ptr()); if p.is_null() { Err(format!("missing {n:?}")) } else { Ok(p) } };
    let retain: unsafe extern "C" fn(*mut *mut c_void, i32) -> i32 = std::mem::transmute(sym(c"cuDevicePrimaryCtxRetain")?);
    let release: unsafe extern "C" fn(i32) -> i32 = std::mem::transmute(sym(c"cuDevicePrimaryCtxRelease_v2")?);
    let set: unsafe extern "C" fn(*mut c_void) -> i32 = std::mem::transmute(sym(c"cuCtxSetCurrent")?);
    let flags: unsafe extern "C" fn(i32, u32) -> i32 = std::mem::transmute(sym(c"cuDevicePrimaryCtxSetFlags_v2")?);
    checked("cuCtxDestroy own", (ex.cuda.context_destroy)(ex.context))?;
    let _ = flags(0, crate::CTX_SCHED_BLOCKING_SYNC);   // fails harmlessly when the client activated it first
    let mut c = ptr::null_mut();
    checked("cuDevicePrimaryCtxRetain", retain(&mut c, 0))?;
    // fwdcheck-digest: the pin pool's buffers live in this context; one retain the pool never releases keeps them valid
    // between sessions (the runtime client usually holds it too).
    static POOL_RETAIN: std::sync::Once = std::sync::Once::new();
    if pin_pool_on() { POOL_RETAIN.call_once(|| { let mut c2 = ptr::null_mut(); let _ = retain(&mut c2, 0); }); }
    checked("cuCtxSetCurrent", set(c))?;
    ex.context = c;
    Ok((0, release))
}

impl Session {
    unsafe fn open(plan_dir: &Path, kset: &Path, bundle: &Path, manifest_path: &Path, opts: &Value) -> Result<Self, String> {
        let t0 = Instant::now();
        let manifest = parse_manifest(manifest_path, kset, bundle)?;
        if let Some(want) = opts["expect_manifest_sha256"].as_str() {
            if !manifest.sha256.starts_with(want) { return Err(format!("DEFECT manifest {} has sha256 {}, the caller names {want} (a manifest change between outer loops is refused)", manifest_path.display(), manifest.sha256)); }
        }
        for (k, v) in &manifest.env { std::env::set_var(k, v); }
        if std::env::var_os("CUDA_MODULE_LOADING").is_none() { std::env::set_var("CUDA_MODULE_LOADING", "EAGER"); }
        let config = plan_dir.join("plan.json");
        let bytes = fs::read(&config).map_err(|e| format!("cannot read {}: {e}", config.display()))?;
        let envelope: Value = serde_json::from_slice(&bytes).map_err(|e| format!("invalid plan JSON: {e}"))?;
        drop(bytes);
        if envelope.get("schema").and_then(|x| x.as_str()) != Some("ad.binder.plan/v1") || envelope.get("complete").and_then(|x| x.as_bool()) != Some(true) {
            return Err("DEFECT refusing incomplete or unknown plan".into());
        }
        // The CLI run_window (window.rs 5871, 5942-5946 at 876b9a06) types every cupy_concatenate_same_size module before the first step;
        // without it the link refuses each one at step 1 ("element type None not established from the module's recorded source").
        let concat_checked = crate::wrules::concat_host_checked(&envelope);
        let plan: Plan = serde_json::from_value(envelope).map_err(|e| format!("invalid sealed plan: {e}"))?;
        let steps = opts["steps"].as_u64().map(|s| s as usize).unwrap_or(plan.cadence.len());
        crate::preflight(&plan, steps)?;
        let base = plan_dir.to_owned();
        crate::verify_fixed_argument_sources(&plan, &base)?;
        crate::verify_copy_indexer_sources(&plan, &base)?;
        let reference = plan.reference.as_deref().map(|p| crate::path(&base, p)).ok_or("DEFECT the plan names no ordinary-forward reference")?;
        let refs = crate::read_references(&reference, &plan, steps)?;
        crate::set_quiet_hashes(opts["hash_lines"].as_bool() != Some(true));
        let mut ex = Execution::new(None)?;
        install_counters(&mut ex.cuda);
        let primary = if opts["context"].as_str() == Some("primary") { Some(use_primary(&mut ex)?) } else { None };
        ex.address_bearing = plan.address_bearing.iter().copied().collect();
        ex.prepare(&plan, &base)?;
        let frame = plan.physics.as_ref().map(ReplayFrame::new).transpose()?;
        if let (Some(f), Some(spec)) = (frame.as_ref(), plan.physics.as_ref()) { ex.host_state.insert(spec.frame_state_id, f.snapshot()?)?; }
        start_staging(&mut ex, &plan, &base, &frame);
        let mut a: HashMap<String, String> = HashMap::new();
        a.insert("config".into(), config.display().to_string());
        a.insert("kernels".into(), kset.join("kernels.json").display().to_string());
        a.insert("cubins".into(), kset.join("cubins").display().to_string());
        for (k, v) in &manifest.args { a.insert(k.clone(), v.clone()); }
        let td = &opts["test_direction"];
        if let Some(d) = td["direction"].as_str() { a.insert("direction".into(), d.into()); }
        if td["stable_dir"].as_bool() == Some(true) { a.insert("stable-dir".into(), "1".into()); }
        if let Some(x) = td["x_scale"].as_f64() { a.insert("x-scale".into(), format!("{x:e}")); }
        if let Some(y) = td["y_scale"].as_f64() { a.insert("y-scale".into(), format!("{y:e}")); }
        let spec_check = check_spec_hashes(&kset.join("kernels.json"))?;
        let mut w = Win::new(&ex, &plan, &a)?;
        w.concat_types = crate::wrules::concat_module_types(&plan, &base)?;
        crate::wrules::mark_host_checked(&mut w.concat_types, &concat_checked);
        eprintln!("{}", json!({"event":"concat_module_types","types":w.concat_types}));
        w.prepare_pass_start(&mut ex, &plan, &base, &frame, steps)?;
        ex.finish_staging_ahead();
        // reverse-step anchors on the device when the card has room for them plus a margin ("device_anchors": false keeps
        // them in host RAM; the default 24 GiB margin covers a physics-live leg growing ~22 GB from open to its peak): the same bytes, D2D instead of a pageable D2H snapshot and an H2D restore per segment
        let state: usize = ex.allocations.values().map(|x| x.bytes).sum::<usize>() + plan_mid_bytes(&plan);
        let mut anchor_note = json!({"device":false});
        if opts["device_anchors"].as_bool() != Some(false) {
            let free = mem_free().unwrap_or(0);
            let need = manifest.anchors * state;
            let margin = (opts["device_anchor_margin_gib"].as_f64().unwrap_or(24.0) * (1u64 << 30) as f64) as usize;
            if need > 0 && free > need + margin {
                for _ in 0..manifest.anchors { let mut p = 0u64; checked("cuMemAlloc device anchor", (ex.cuda.allocate)(&mut p, state))?; w.dev_anchors.push((p, state)); }
                anchor_note = json!({"device":true,"buffers":manifest.anchors,"bytes_each":state,"free_before":free});
            } else { anchor_note = json!({"device":false,"reason":"card has no room","free":free,"need":need,"margin":margin}); }
        }
        let adir = opts["alloc_map_dir"].as_str().map(PathBuf::from).unwrap_or_else(|| plan_dir.join("adjoint"));
        let tm = if adir.join("alloc-map.json").exists() { Some(TMaps::load(&ex.cuda, &adir)?) } else { None };
        let scale = match opts["scale_table"].as_str().map(PathBuf::from).or_else(|| Some(adir.join("scale-table.json")).filter(|p| p.exists())) {
            Some(p) => parse_scale(&jfile(&p)?)?, None => Scale::default() };
        let bdir = PathBuf::from(opts["bstore"].as_str().ok_or("options: bstore (a local NVMe directory for boundaries beyond the RAM budget)")?);
        fs::create_dir_all(&bdir).map_err(|e| format!("{}: {e}", bdir.display()))?;
        let avail = fs::read_to_string("/proc/meminfo").ok().and_then(|m| m.lines().find(|l| l.starts_with("MemAvailable")).and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())).unwrap_or(0) * 1024;
        let ram_budget = match opts["boundary_ram_gib"].as_f64() { Some(g) => (g * (1u64 << 30) as f64) as usize, None => (avail as f64 * 0.5) as usize };
        let receipt = json!({"event":"hexad_open","abi":ABI_VERSION,"plan":plan_dir.display().to_string(),"kset":kset.display().to_string(),
            "bundle":bundle.display().to_string(),"manifest":manifest_path.display().to_string(),"manifest_sha256":manifest.sha256,
            "manifest_doc":manifest.doc,"window_args":a,"spec_sha256":spec_check,"steps":steps,"context":if primary.is_some() {"primary"} else {"own"},
            "alloc_map":tm.as_ref().map(|t| json!({"dir":adir.display().to_string(),"sha256":t.doc_sha256})),
            "scale":{"tangent_exp":scale.t,"cotangent_exp":scale.a},"boundary_ram_budget_bytes":ram_budget,"mem_available_bytes":avail,
            "open_s":t0.elapsed().as_secs_f64(),"anchors":anchor_note});
        let pin_ctx = if primary.is_some() && pin_pool_on() { pin_open(ex.context as usize, ram_budget) } else { 0 };
        if pin_ctx != 0 {
            // every boundary after boundary 0 at most `state` bytes (all live plus every mid-window allocation)
            let cap = state.max(1).div_ceil(PIN_GRANULE) * PIN_GRANULE;
            pin_prefill(&ex.cuda, pin_ctx, cap, steps.min(ram_budget / cap));
        }
        eprintln!("{receipt}");
        Ok(Self { pin_ctx, bounds: Vec::new(), flats: Vec::new(), prim: Vec::new(), tm, w: Some(w), frame, refs, ex, plan, base, steps,
            anchors: manifest.anchors, bdir, manifest, scale, ram_budget, ram_used: 0, disk_bounds: 0, disk_reads: 0, disk_wait: 0.0,
            tl_s: 0, ad_s: 0, mode: Mode::Idle, tl_steps: 0, ad_steps: 0, tape_sized: false, check_every: opts["check_every_pass"].as_bool() == Some(true),
            receipt, primary, mirror: BTreeMap::new(), tbufs: Vec::new(), tmap_only: false, dcache: Vec::new() })
    }
    fn win(&mut self) -> &mut Win { self.w.as_mut().unwrap() }
    fn traj(&self) -> bool { self.bounds.len() == self.steps + 1 }
    /// Store boundary s: pinned host RAM while the budget lasts, else a local-NVMe file (counted).
    unsafe fn store(&mut self, s: usize) -> Result<Boundary, String> {
        let _c = xfer::Cat::checkpoint();
        // The whole-state capture is valid only for the boundary it was taken at: it is used only when it holds exactly
        // the live allocations at their addresses (store follows the step's finish directly, nothing ran since), and a
        // capture left behind never reaches a later store.
        let cap = self.ex.captured.take().filter(|(_, index)| index.len() == self.ex.allocations.len()
            && index.iter().all(|(id, _, n, addr)| self.ex.allocations.get(id).is_some_and(|a| a.bytes == *n && a.address == *addr)));
        let total: usize = self.ex.allocations.values().map(|a| a.bytes).sum();
        if self.ram_used + total > self.ram_budget {
            self.disk_bounds += 1;
            self.ex.captured = cap;   // Win::store checks it again and writes it (or downloads afresh)
            let w = self.w.as_mut().unwrap();
            return w.store(&mut self.ex, &self.frame, &self.bdir, s);
        }
        // fwdcheck-digest: the step's capture went straight into page-locked bytes: they are this boundary's RAM copy.
        if let Some((d, ix)) = cap {
            if let crate::CapData::Host(hb) = &*d {
                if std::env::var("HEXAD_FC_VERIFY").as_deref() == Ok("1") {
                    // proof only: the bytes the old store downloaded here, compared with the capture it now keeps
                    dev::sync(&self.ex.cuda)?;
                    let mut v = vec![0u8; hb.as_slice().len()];
                    for (id, off, n, _) in &ix {
                        if *n != 0 { checked("cuMemcpyDtoH verify", (self.ex.cuda.download)(v[*off..*off + *n].as_mut_ptr().cast(), self.ex.allocations[id].address, *n))?; }
                    }
                    let diff = v.iter().zip(hb.as_slice()).filter(|(a, b)| a != b).count() as u64;
                    FC_VERIFY.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    FC_VERIFY.1.fetch_add(v.len() as u64, std::sync::atomic::Ordering::Relaxed);
                    FC_VERIFY.2.fetch_add(diff, std::sync::atomic::Ordering::Relaxed);
                }
                self.ram_used += hb.as_slice().len();
                let index = ix.iter().map(|(id, off, n, _)| (*id, *off, *n)).collect();
                return Ok(Boundary { sub: None, path: self.bdir.join(format!("b{s}.bin")), index, host: host_snap(&self.ex, &self.frame), step: s, cs: None, delta: false, file_bytes: 0, ram: Some(hb.clone()), dev: 0 });
            }
        }
        dev::sync(&self.ex.cuda)?;
        let mut ids: Vec<u64> = self.ex.allocations.keys().copied().collect();
        ids.sort_unstable();
        let mut index = Vec::with_capacity(ids.len());
        let mut at = 0usize;
        for id in &ids { let b = self.ex.allocations[id].bytes; index.push((*id, at, b)); at += b; }
        let mut hb = HostBytes::pooled(&self.ex.cuda, self.pin_ctx, at)?;
        { let buf = hb.as_mut_slice();
          for (id, off, b) in &index { checked("cuMemcpyDtoH boundary", (self.ex.cuda.download)(buf[*off..*off + *b].as_mut_ptr().cast(), self.ex.allocations[id].address, *b))?; } }
        self.ram_used += at;
        Ok(Boundary { sub: None, path: self.bdir.join(format!("b{s}.bin")), index, host: host_snap(&self.ex, &self.frame), step: s, cs: None, delta: false, file_bytes: 0, ram: Some(Arc::new(hb)), dev: 0 })
    }
    /// Boundary i's bytes: the pinned RAM copy, or the disk-tier file read back (the wait is counted). The pointer stays
    /// valid while the boundary is stored (RAM) or among the 4 most recently read (disk).
    unsafe fn data(&mut self, i: usize) -> Result<*const [u8], String> {
        if let Some(r) = &self.bounds[i].ram { return Ok(r.as_slice() as *const [u8]); }
        if let Some(k) = self.dcache.iter().position(|(j, _)| *j == i) { let e = self.dcache.remove(k); self.dcache.push(e); }
        else {
            let t = Instant::now();
            let w = self.w.as_mut().unwrap();
            let d = { let _c = xfer::Cat::checkpoint(); w.load_at(&self.bounds, i)? };
            self.disk_reads += 1; self.disk_wait += t.elapsed().as_secs_f64();
            if self.dcache.len() >= 4 { self.dcache.remove(0); }
            self.dcache.push((i, d));
        }
        Ok(&self.dcache.last().unwrap().1[..] as *const [u8])
    }
    unsafe fn restore0(&mut self) -> Result<(), String> {
        let d = self.data(0)?;
        let b = &self.bounds[0] as *const Boundary;
        let w = self.w.as_mut().unwrap();
        w.restore(&mut self.ex, &mut self.frame, &*b, &*d)
    }
    /// Device copies of the canonical theta_m and qv at the current live state (Ts / Ts^T primal of a boundary).
    unsafe fn capture_prim(&mut self) -> Result<(), String> {
        let _g = xfer::Cat::glue();
        let Some(t) = self.tm.as_ref() else { self.prim.push((0, 0)); return Ok(()) };
        let n = t.slot_words * 4;
        let (fa, fo) = t.fields["theta_m"].canon; let (sa, so) = t.fields["scalars"].canon;
        let qo = so + 4 * t.species["qv"] * t.slot_words;
        let cu = &self.ex.cuda;
        let (a, b) = (dalloc(cu, n)?, dalloc(cu, n)?);
        dev::copy(cu, a, self.ex.allocations.get(&fa).ok_or("theta_m not live")?.address + fo as u64, n)?;
        dev::copy(cu, b, self.ex.allocations.get(&sa).ok_or("scalars not live")?.address + qo as u64, n)?;
        dev::sync(cu)?;
        self.prim.push((a, b));
        Ok(())
    }
    fn arena_dst(&self) -> impl Fn(u64, usize) -> Result<Dst, String> + '_ {
        move |a: u64, o: usize| {
            let w = self.w.as_ref().unwrap();
            let el = *w.base_el.get(&a).ok_or_else(|| format!("allocation {a} has no shadow"))?;
            Ok(Dst { addr: w.sh(el) + o as u64, f64: false })
        }
    }
    unsafe fn bind_primal0(&mut self) -> Result<(), String> {
        let _g = xfer::Cat::glue();
        let ex = &self.ex as *const Execution;
        let Some(t) = self.tm.as_mut() else { return Ok(()) };
        if t.primal_bound { return Ok(()); }
        let cu = &(*ex).cuda;
        t.bind_primal(cu, &|a, o| (*ex).allocations.get(&a).map(|x| x.address + o as u64).ok_or_else(|| format!("allocation {a} not live")))
    }
    unsafe fn seed(&mut self, v: &HexadVec, at_end: bool) -> Result<(), String> {
        let _g = xfer::Cat::glue();
        let cu = &self.ex.cuda as *const Cuda;
        match v.kind {
            VEC_ZERO => Ok(()),
            VEC_SEED => {
                let i = if at_end { self.steps } else { 0 };
                let d = self.data(i)?;
                let b = &self.bounds[i] as *const Boundary;
                let w = self.w.as_ref().unwrap();
                w.fill(&*cu, if at_end { sy(v.seed) } else { sx(v.seed) }, Some((&*b, &*d)))
            }
            VEC_ARENA => {
                let w = self.w.as_ref().unwrap();
                if v.words != w.words as u64 { return Err(format!("arena vector has {} words, the arena {}", v.words, w.words)); }
                dev::copy(&*cu, w.sh_base, v.arena, w.words as usize * 4)
            }
            VEC_INCREMENT => {
                let t = self.tm.as_ref().ok_or("DEFECT increment vectors need the plan's alloc map (adjoint/alloc-map.json)")?;
                let dst = self.arena_dst();
                if at_end {
                    let (a, b) = self.prim[self.steps];
                    t.tst(&*cu, v, &dst, self.scale.a, a, b)
                } else { t.t0(&*cu, v, &dst, self.scale.t) }
            }
            k => Err(format!("vector kind {k}")),
        }
    }
    unsafe fn tl_init(&mut self, v: &HexadVec) -> Result<(), String> {
        if self.tmap_only { return self.tmap_tl(v); }
        if self.bounds.is_empty() {
            let b0 = self.store(0)?; self.bounds.push(b0); self.capture_prim()?;
        } else { self.restore0()?; }
        self.bind_primal0()?;
        let cu = &self.ex.cuda as *const Cuda;
        self.win().zero(&*cu)?;
        if v.kind == VEC_SEED && !self.traj() { // the window's seed directions need the stored boundary they weight
            let d = self.data(0)?; let b = &self.bounds[0] as *const Boundary;
            self.w.as_ref().unwrap().fill(&*cu, sx(v.seed), Some((&*b, &*d)))?;
        } else { self.seed(v, false)?; }
        let w = self.w.as_mut().unwrap(); w.trace_last = f64::MAX; w.entry_ms = 0.0;
        w.prepare_pass_start(&mut self.ex, &self.plan, &self.base, &self.frame, self.steps)?;
        self.tl_s = 0; self.mode = Mode::Tl;
        Ok(())
    }
    unsafe fn tl_step(&mut self) -> Result<(), String> {
        if !matches!(self.mode, Mode::Tl) { return Err("tl_step outside a tangent sweep (call tl_init)".into()); }
        if self.tl_s >= self.steps { return Err(format!("tl_step past the window end ({} steps)", self.steps)); }
        let s = self.tl_s + 1;
        let first = self.bounds.len() == s;
        let refs = if first || self.check_every { Some(&self.refs as *const References) } else { None };
        // per-step-host-hash: an unchecked step of a bound trajectory skips the boundary digest nothing reads.
        self.ex.unchecked_digest_skip = refs.is_none() && self.traj()
            && std::env::var("HEXAD_TL_UNCHECKED_DIGEST").as_deref() != Ok("1");
        let room = self.ram_budget.saturating_sub(self.ram_used);
        let w = self.w.as_mut().unwrap();
        w.counting = first;
        // fwdcheck-digest: a step that stores its boundary captures the whole state once, into the boundary's page-locked
        // RAM copy; the host digest reads that capture (deferred beside the next step inside forward_check) and store()
        // keeps it, so there is no second download and no per-boundary cuMemHostAlloc on the step path.
        // (A storing step is `first`, so it always has refs and never takes the per-step-host-hash skip.)
        w.capture = first; w.capture_host = if first { Some((self.pin_ctx, room)) } else { None };
        let r = w.forward_step_c(&mut self.ex, &self.plan, &self.base, &mut self.frame, refs.map(|r| &*r), s, true);
        w.capture = false; w.capture_host = None;
        self.ex.unchecked_digest_skip = false;
        let flat = r?;
        if first {
            let b = self.store(s)?; self.bounds.push(b); self.flats.push(flat); self.capture_prim()?;
            if self.traj() { let p = self.bounds[self.steps].path.clone(); self.win().y_path = Some(p);
                             if !self.ex.digest_defer { self.receipt["trajectory_boundaries_equal"] = json!(self.steps); } }
        }
        self.ex.captured = None;
        self.win().counting = false;
        self.tl_s = s; self.tl_steps += 1;
        Ok(())
    }
    /// HexRecord::step: the recorded nonlinear state at boundary `step` into State fields (device copy). Binds the
    /// trajectory first if needed (forward_check: a zero-direction sweep, every boundary hash checked against
    /// reference.jsonl), restores boundary `step` into the live allocations and leaves the session idle (the next
    /// tangent sweep restores boundary 0).
    unsafe fn state_get(&mut self, step: i32, v: &HexadVec) -> Result<(), String> {
        if self.tmap_only { return Err("DEFECT state_get needs a plan session".into()); }
        if v.kind != VEC_INCREMENT { return Err(format!("state_get: vector kind {} (INCREMENT fields only)", v.kind)); }
        if step < 0 || step as usize > self.steps { return Err(format!("state_get: boundary {step} outside 0..={}", self.steps)); }
        if !self.traj() { self.forward_check()?; }
        let i = step as usize;
        let _g = xfer::Cat::glue();
        let d = self.data(i)?;
        let b = &self.bounds[i] as *const Boundary;
        { let w = self.w.as_mut().unwrap(); w.restore(&mut self.ex, &mut self.frame, &*b, &*d)?; }
        self.mode = Mode::Idle;
        let t = self.tm.as_ref().ok_or("DEFECT state_get needs the plan's alloc map")?;
        let ex = &self.ex as *const Execution;
        let dst = move |a: u64, o: usize| -> Result<Dst, String> {
            let al = (*ex).allocations.get(&a).ok_or_else(|| format!("allocation {a} not live"))?;
            Ok(Dst { addr: al.address + o as u64, f64: false })
        };
        let (thm32, qv32) = self.prim[i];
        t.state(&(*ex).cuda, v, &dst, thm32, qv32)
    }
    unsafe fn tl_get(&mut self, v: &HexadVec) -> Result<(), String> {
        let _g = xfer::Cat::glue();
        if self.tmap_only { return self.tmap_ts(v); }
        let cu = &self.ex.cuda as *const Cuda;
        match v.kind {
            VEC_ARENA => { let w = self.w.as_ref().unwrap(); dev::copy(&*cu, v.arena, w.sh_base, w.words as usize * 4) }
            VEC_INCREMENT => {
                let t = self.tm.as_ref().ok_or("DEFECT increment vectors need the plan's alloc map")?;
                let (a, b) = self.prim[self.tl_s];
                let dst = self.arena_dst();
                t.ts(&*cu, v, &dst, self.scale.t, a, b)
            }
            k => Err(format!("tl_get: vector kind {k}")),
        }
    }
    unsafe fn ad_init(&mut self, v: &HexadVec) -> Result<(), String> {
        if self.tmap_only { return self.tmap_zero_mirror(); }
        if !self.traj() { return Err("DEFECT ad_init before the trajectory is bound: run one tangent sweep (or forward_check) over every step first".into()); }
        let cu = &self.ex.cuda as *const Cuda;
        if !self.tape_sized {
            // fold-measure (tape-cap e2ec708f in the library): size the tape from the card once, before the first reverse,
            // as the CLI window does after its first tangent pass. Segmentation never changes the walk-back order or a word.
            self.tape_sized = true;
            if std::env::var("AD_TAPE_SIZE").as_deref() == Ok("0") {
                eprintln!("{}", json!({"event":"hexad_tape_not_sized","reason":"AD_TAPE_SIZE=0"}));
            } else if self.win().one_pass {
                let (ex, fl) = (&self.ex as *const Execution, &self.flats as *const Vec<Vec<Flat>>);
                let (mx, rs) = (std::env::var("AD_TAPE_MAX_GIB").ok(), std::env::var("AD_TAPE_RESERVE_GIB").ok());
                self.win().size_tape(&*ex, &*fl, mx.as_deref(), rs.as_deref())?;
            }
        }
        self.win().zero(&*cu)?;
        if v.kind == VEC_INCREMENT {
            let _g = xfer::Cat::glue();
            let t = self.tm.as_ref().ok_or("DEFECT increment vectors need the plan's alloc map")?;
            let (a, b) = self.prim[self.steps];
            let dst = self.arena_dst();
            t.tst(&*cu, v, &dst, self.scale.a, a, b)?;
        } else { self.seed(v, true)?; }
        let w = self.win(); w.trace_last = f64::MAX; w.trace_hits = 0; w.entry_ms = 0.0;
        self.ad_s = self.steps; self.mode = Mode::Ad;
        Ok(())
    }
    /// torch bridge reverse (form v1 "step"): a reverse that starts at boundary `step` instead of the window end,
    /// from `v`: ZERO, ARENA (the whole cotangent arena an earlier piece handed out with ad_get(ARENA)) or INCREMENT
    /// (T_step^T(v): the caller's cotangent on the named state fields at that boundary). Then ad_put / ad_step down to
    /// k0 and ad_get(ARENA) hand out the state cotangent at k0. The arena is the whole cotangent state carried between
    /// steps, so pieces [k1, k0) chained through ARENA give the whole window's reverse.
    unsafe fn ad_init_at(&mut self, step: i32, v: &HexadVec) -> Result<(), String> {
        if step < 0 || step as usize > self.steps { return Err(format!("ad_init_at: boundary {step} outside 0..={}", self.steps)); }
        let zero = HexadVec { kind: VEC_ZERO, nfields: 0, fields: std::ptr::null(), seed: 0, arena: 0, words: 0 };
        self.ad_init(&zero)?;
        self.ad_s = step as usize;
        match v.kind {
            VEC_ZERO => Ok(()),
            VEC_ARENA => self.seed(v, true),
            VEC_INCREMENT => {
                let _g = xfer::Cat::glue();
                let t = self.tm.as_ref().ok_or("DEFECT increment vectors need the plan's alloc map")?;
                let cu = &self.ex.cuda as *const Cuda;
                let (a, b) = self.prim[self.ad_s];
                let dst = self.arena_dst();
                t.tst(&*cu, v, &dst, self.scale.a, a, b)
            }
            k => Err(format!("ad_init_at: vector kind {k} (ZERO, ARENA or INCREMENT)")),
        }
    }
    /// torch bridge: the device address and bytes of the float32 cotangent (or tangent) shadow of plan allocation `alloc`
    /// from byte `offset`: zero-copy reads and adds by the caller (DLPack / CUDA array interface), e.g. a recorded
    /// external-tendency table's slice k once the reverse has passed step k, or a state field at the current boundary.
    fn shadow_ptr(&self, alloc: u64, offset: u64) -> Result<(u64, u64), String> {
        let w = self.w.as_ref().ok_or("DEFECT shadow_ptr needs a plan session")?;
        let (Some(&el), Some(sh)) = (w.base_el.get(&alloc), w.shadows.get(&alloc)) else { return Err(format!("allocation {alloc} has no shadow")) };
        if offset % 4 != 0 || offset as usize >= sh.bytes as usize { return Err(format!("offset {offset} outside allocation {alloc} or not on a word")); }
        Ok((w.sh(el + (offset / 4) as i64), sh.bytes as u64 - offset))
    }
    unsafe fn ad_put(&mut self, v: &HexadVec) -> Result<(), String> {
        let _g = xfer::Cat::glue();
        if self.tmap_only { return self.tmap_tst(v); }
        if !matches!(self.mode, Mode::Ad) { return Err("ad_put outside a reverse sweep (call ad_init)".into()); }
        let t = self.tm.as_ref().ok_or("DEFECT increment vectors need the plan's alloc map")?;
        if v.kind != VEC_INCREMENT { return Err("ad_put takes an increment vector".into()); }
        let cu = &self.ex.cuda as *const Cuda;
        let (a, b) = self.prim[self.ad_s];
        let dst = self.arena_dst();
        t.tst(&*cu, v, &dst, self.scale.a, a, b)
    }
    unsafe fn ad_step(&mut self) -> Result<Value, String> {
        if !matches!(self.mode, Mode::Ad) { return Err("ad_step outside a reverse sweep (call ad_init)".into()); }
        if self.ad_s == 0 { return Err("ad_step past boundary 0".into()); }
        let s = self.ad_s;
        // prefetch the disk-tier boundary the next call needs (RAM boundaries need none)
        if s >= 2 && self.bounds[s - 2].ram.is_none() { let b = &self.bounds as *const Vec<Boundary>; self.win().prefetch_load(&*b, s - 2)?; }
        let prev = self.data(s - 1)?;
        let cur = self.data(s)?;
        let (pb, cb) = (&self.bounds[s - 1] as *const Boundary, &self.bounds[s] as *const Boundary);
        let w = self.w.as_mut().unwrap();
        w.stage_blobs(&mut self.ex, &self.base, flat_blobs(&self.flats[s - 1]), (s >= 2).then(|| flat_blobs(&self.flats[s - 2])))?;
        let fl = &self.flats[s - 1] as *const Vec<Flat>;
        let r = w.reverse_step(&mut self.ex, &self.base, &mut self.frame, s, &*fl, &*pb, &*prev, &*cb, &*cur, self.anchors)?;
        self.ad_s = s - 1; self.ad_steps += 1;
        Ok(r)
    }
    unsafe fn ad_get(&mut self, v: &HexadVec) -> Result<(), String> {
        let _g = xfer::Cat::glue();
        if self.tmap_only { return self.tmap_t0t(v); }
        let cu = &self.ex.cuda as *const Cuda;
        match v.kind {
            VEC_ARENA => { let w = self.w.as_ref().unwrap(); dev::copy(&*cu, v.arena, w.sh_base, w.words as usize * 4) }
            VEC_INCREMENT => {
                if self.ad_s != 0 { return Err(format!("ad_get: the reverse is at boundary {}, T0^T needs boundary 0", self.ad_s)); }
                let t = self.tm.as_ref().ok_or("DEFECT increment vectors need the plan's alloc map")?;
                let dst = self.arena_dst();
                t.t0t(&*cu, v, &dst, self.scale.a)
            }
            k => Err(format!("ad_get: vector kind {k}")),
        }
    }
    /// Zero-direction trajectory pass: binds (or re-checks) the trajectory; every boundary hash against reference.jsonl.
    /// fwdcheck-digest (host-audit row 13): each boundary's host SHA-256 digest (the same readings, comparisons, prints and
    /// refusals; terrain-w-hash's digest_post) runs on a background thread beside the next step's kernels, joined at the
    /// next boundary and before this returns; a mismatch still refuses here. HEXAD_FC_DEFER=0: at each boundary.
    unsafe fn forward_check(&mut self) -> Result<(u64, u64), String> {
        let t0 = Instant::now();
        let (a0, r0, s0, _) = pin_stats();
        let (p0, ps0) = pin_prefill_stats();
        let g0 = pin_registered();
        let (dg0, bg0) = (self.ex.digest_s, self.ex.digest_bg_s);
        let save = self.check_every; self.check_every = true;
        let defer = std::env::var("HEXAD_FC_DEFER").as_deref() != Ok("0");
        self.ex.digest_defer = defer;
        let r = (|| {
            let z = HexadVec { kind: VEC_ZERO, nfields: 0, fields: ptr::null(), seed: 0, arena: 0, words: 0 };
            self.tl_init(&z)?;
            for _ in 0..self.steps { self.tl_step()?; }
            let cu = &self.ex.cuda as *const Cuda;
            let nz = self.w.as_ref().unwrap().nonzero(&*cu)?;
            Ok((self.steps as u64, nz))
        })();
        self.ex.digest_defer = false;
        let joined = self.ex.digest_flush();
        self.check_every = save;
        let r = r.and_then(|x| joined.map(|_| x));
        if r.is_ok() && self.traj() { self.receipt["trajectory_boundaries_equal"] = json!(self.steps); }
        let (a1, r1, s1, idle) = pin_stats();
        let (p1, ps1) = pin_prefill_stats();
        eprintln!("{}", json!({"event":"hexad_forward_check","ok":r.is_ok(),"steps":self.steps,"wall_s":t0.elapsed().as_secs_f64(),
            "digest":if defer {"deferred beside the next step"} else {"at each boundary"},"digest_main_s":self.ex.digest_s - dg0,
            "digest_background_s":self.ex.digest_bg_s - bg0,"pin_pool":{"key":self.pin_ctx != 0,"allocated":a1 - a0,"reused":r1 - r0,
            "cuMemHostAlloc_s":s1 - s0,"alloc_s":s1 - s0,"allocated_registered":pin_registered() - g0,
            "alloc_path":if std::env::var("AD_PIN_ALLOC").as_deref() == Ok("driver") {"cuMemHostAlloc (AD_PIN_ALLOC=driver)"} else {"parallel first touch + cuMemHostRegister"},"page_policy":crate::pinned::page_policy(),"idle_bytes_after":idle,
            "prefilled_by_helper":p1 - p0,"prefill_helper_s":ps1 - ps0},
            "verify":{"boundaries":FC_VERIFY.0.load(std::sync::atomic::Ordering::Relaxed),"bytes":FC_VERIFY.1.load(std::sync::atomic::Ordering::Relaxed),
                      "differing_bytes":FC_VERIFY.2.load(std::sync::atomic::Ordering::Relaxed)},"boundaries_ram":self.bounds.iter().filter(|b| b.ram.is_some()).count()}));
        r
    }
    unsafe fn test_dot(&mut self, which: i32, seed: u64) -> Result<(f64, f64, u64), String> {
        let _g = xfer::Cat::glue();
        if !self.traj() { return Err("test_dot needs the trajectory".into()); }
        let cu = &self.ex.cuda as *const Cuda;
        let d0 = self.data(0)?; let dn = self.data(self.steps)?;
        let (b0, bn) = (&self.bounds[0] as *const Boundary, &self.bounds[self.steps] as *const Boundary);
        let w = self.w.as_ref().unwrap();
        let (xs, ys) = (Some((&*b0, &*d0)), Some((&*bn, &*dn)));
        if which == 0 { w.dots(&*cu, sy(seed), ys, sx(seed), xs) } else { w.dots(&*cu, sx(seed), xs, sy(seed), ys) }
    }
    fn counters(&self) -> HexadCounters {
        HexadCounters { h2d_bytes: xfer::get(&xfer::N.h2d), d2h_bytes: xfer::get(&xfer::N.d2h), h2d_calls: xfer::get(&xfer::N.h2d_n),
            d2h_calls: xfer::get(&xfer::N.d2h_n), ctx_syncs: xfer::get(&xfer::N.ctx), stream_syncs: xfer::get(&xfer::N.stream),
            event_syncs: xfer::get(&xfer::N.event), boundaries_ram: self.bounds.iter().filter(|b| b.ram.is_some()).count() as u64,
            boundaries_disk: self.disk_bounds, boundary_ram_bytes: self.ram_used as u64, disk_reads_on_path: self.disk_reads,
            disk_wait_s: self.disk_wait, tl_steps: self.tl_steps, ad_steps: self.ad_steps }
    }
    fn alloc_table(&self) -> Vec<HexadAlloc> {
        let w = self.w.as_ref().unwrap();
        let init: std::collections::HashSet<u64> = self.plan.allocations.iter().map(|a| a.id).collect();
        w.shadows.iter().map(|(id, sh)| {
            let live = self.ex.allocations.get(id);
            let mut flags = 0u32;
            if init.contains(id) { flags |= ALLOC_INITIAL; }
            if w.fixed.contains(id) { flags |= ALLOC_FIXED; }
            if w.address_bearing.contains(id) { flags |= ALLOC_ADDRESS_BEARING; }
            if live.is_some() { flags |= ALLOC_LIVE; }
            HexadAlloc { id: *id, bytes: live.map(|a| a.bytes as u64).unwrap_or(sh.bytes as u64), primal: live.map(|a| a.address).unwrap_or(0),
                         shadow: sh.address, flags, scale_exp: self.scale.t }
        }).collect()
    }

    // ---- tmap-only test session: T maps on the plan's boundary-0 words, every copy a float64 mirror
    unsafe fn open_tmap(plan_dir: &Path, adir: &Path, opts: &Value) -> Result<Self, String> {
        let mut ex = Execution::new(None)?;
        install_counters(&mut ex.cuda);
        let primary = if opts["context"].as_str() == Some("primary") { Some(use_primary(&mut ex)?) } else { None };
        let config = plan_dir.join("plan.json");
        let envelope: Value = jfile(&config)?;
        let plan: Plan = serde_json::from_value(envelope).map_err(|e| format!("invalid sealed plan: {e}"))?;
        let mut tm = TMaps::load(&ex.cuda, adir)?;
        let mut tbufs = Vec::new(); let mut at: HashMap<u64, u64> = HashMap::new();
        let need: std::collections::BTreeSet<u64> = tm.fields.values().map(|f| f.canon.0).collect();
        for a in &plan.allocations {
            if !need.contains(&a.id) { continue; }
            let b = crate::blob(plan_dir, a.initial.as_ref().ok_or("allocation without initial bytes")?)?;
            let p = dup(&ex.cuda, &b)?; tbufs.push(p); at.insert(a.id, p);
        }
        tm.bind_primal(&ex.cuda, &|a, o| at.get(&a).map(|p| p + o as u64).ok_or_else(|| format!("allocation {a} not in the plan")))?;
        let (fa, fo) = tm.fields["theta_m"].canon; let (sa, so) = tm.fields["scalars"].canon;
        let prim0 = (at[&fa] + fo as u64, at[&sa] + (so + 4 * tm.species["qv"] * tm.slot_words) as u64);
        let mut mirror = BTreeMap::new();
        for (a, o, n) in tm.all_copies() { mirror.insert((a, o), dalloc(&ex.cuda, n * 8)?); }
        let refs = References::new();
        let manifest = Manifest { sha256: String::new(), doc: Value::Null, args: vec![], env: vec![], anchors: 0 };
        let mut s = Self { pin_ctx: 0, bounds: Vec::new(), flats: Vec::new(), prim: vec![], tm: Some(tm), w: None, frame: None, refs, ex, plan, base: plan_dir.to_owned(),
            steps: 0, anchors: 0, bdir: PathBuf::new(), manifest, scale: match opts["scale_table"].as_str() { Some(p) => parse_scale(&jfile(Path::new(p))?)?, None => Scale::default() },
            ram_budget: 0, ram_used: 0, disk_bounds: 0, disk_reads: 0, disk_wait: 0.0, tl_s: 0, ad_s: 0, mode: Mode::Idle, tl_steps: 0, ad_steps: 0, tape_sized: false,
            check_every: false, receipt: json!({"event":"hexad_tmap_open"}), primary, mirror, tbufs, tmap_only: true, dcache: Vec::new() };
        s.tbufs.push(0); // marker (prim0 lives inside the plan buffers)
        s.receipt["prim0"] = json!([prim0.0, prim0.1]);
        Ok(s)
    }
    fn mirror_dst(&self) -> impl Fn(u64, usize) -> Result<Dst, String> + '_ {
        move |a, o| self.mirror.get(&(a, o)).map(|&p| Dst { addr: p, f64: true }).ok_or_else(|| format!("no mirror for {a}@{o}"))
    }
    fn prim0(&self) -> (u64, u64) { (self.receipt["prim0"][0].as_u64().unwrap_or(0), self.receipt["prim0"][1].as_u64().unwrap_or(0)) }
    unsafe fn tmap_tl(&mut self, v: &HexadVec) -> Result<(), String> {
        let cu = &self.ex.cuda as *const Cuda; let t = self.tm.as_ref().unwrap(); let d = self.mirror_dst();
        t.t0(&*cu, v, &d, self.scale.t)
    }
    unsafe fn tmap_t0t(&mut self, v: &HexadVec) -> Result<(), String> {
        let cu = &self.ex.cuda as *const Cuda; let t = self.tm.as_ref().unwrap(); let d = self.mirror_dst();
        t.t0t(&*cu, v, &d, self.scale.a)
    }
    unsafe fn tmap_ts(&mut self, v: &HexadVec) -> Result<(), String> {
        let cu = &self.ex.cuda as *const Cuda; let (a, b) = self.prim0(); let t = self.tm.as_ref().unwrap(); let d = self.mirror_dst();
        t.ts(&*cu, v, &d, self.scale.t, a, b)
    }
    unsafe fn tmap_tst(&mut self, v: &HexadVec) -> Result<(), String> {
        let cu = &self.ex.cuda as *const Cuda; let (a, b) = self.prim0(); let t = self.tm.as_ref().unwrap(); let d = self.mirror_dst();
        t.tst(&*cu, v, &d, self.scale.a, a, b)
    }
    unsafe fn tmap_zero_mirror(&mut self) -> Result<(), String> {
        for (&(a, o), &p) in &self.mirror {
            let n = self.tm.as_ref().unwrap().fields.values().find(|f| f.copies.contains(&(a, o))).map(|f| f.words).unwrap_or(0);
            dev::zero(&self.ex.cuda, p, n * 8)?;
        }
        dev::sync(&self.ex.cuda)
    }
}

// ------------------------------------------------------------------------------------------------ extern "C"
#[no_mangle] pub extern "C" fn hexad_abi_version() -> u32 { ABI_VERSION }
#[no_mangle] pub extern "C" fn hexad_last_error() -> *const c_char { LAST.with(|l| l.borrow().as_ptr()) }

/// Open a session on a sealed plan package. options_json may be NULL (defaults) or a JSON object (see hexad_phys.h).
#[no_mangle] pub unsafe extern "C" fn hexad_open(plan_dir: *const c_char, kset_dir: *const c_char, bundle_dir: *const c_char,
                                                manifest: *const c_char, options_json: *const c_char) -> *mut Session {
    let mut out: *mut Session = ptr::null_mut();
    let code = guard(|| {
        let opts: Value = if options_json.is_null() { json!({}) } else { serde_json::from_str(cstr(options_json, "options")?).map_err(|e| format!("options: {e}"))? };
        let s = Session::open(Path::new(cstr(plan_dir, "plan dir")?), Path::new(cstr(kset_dir, "kset dir")?), Path::new(cstr(bundle_dir, "bundle dir")?),
                              Path::new(cstr(manifest, "manifest")?), &opts)?;
        out = Box::into_raw(Box::new(s));
        Ok(())
    });
    if code != 0 { ptr::null_mut() } else { out }
}
/// A T-map test session: the alloc map's T0 / T0^T / Ts / Ts^T on the plan's boundary-0 words, every copy a float64
/// mirror (no window, no kernel set: a ~100 MB card is enough). Used by the pair-dot and finite-difference tests.
#[no_mangle] pub unsafe extern "C" fn hexad_tmap_open(plan_dir: *const c_char, alloc_map_dir: *const c_char, options_json: *const c_char) -> *mut Session {
    let mut out: *mut Session = ptr::null_mut();
    let code = guard(|| {
        let opts: Value = if options_json.is_null() { json!({}) } else { serde_json::from_str(cstr(options_json, "options")?).map_err(|e| format!("options: {e}"))? };
        let s = Session::open_tmap(Path::new(cstr(plan_dir, "plan dir")?), Path::new(cstr(alloc_map_dir, "alloc map dir")?), &opts)?;
        out = Box::into_raw(Box::new(s));
        Ok(())
    });
    if code != 0 { ptr::null_mut() } else { out }
}
#[no_mangle] pub unsafe extern "C" fn hexad_close(s: *mut Session) { if !s.is_null() { drop(Box::from_raw(s)); } }
#[no_mangle] pub unsafe extern "C" fn hexad_steps(s: *const Session) -> c_int { if s.is_null() { -1 } else { (*s).steps as c_int } }
#[no_mangle] pub unsafe extern "C" fn hexad_arena_words(s: *const Session) -> u64 { if s.is_null() { 0 } else { (*s).w.as_ref().map(|w| w.words as u64).unwrap_or(0) } }
#[no_mangle] pub unsafe extern "C" fn hexad_alloc_table(s: *const Session, rows: *mut HexadAlloc, cap: usize, n: *mut usize) -> c_int {
    guard(|| {
        let s = s.as_ref().ok_or("null session")?;
        if s.tmap_only { return Err("tmap session has no arena".into()); }
        let t = s.alloc_table();
        if !n.is_null() { *n = t.len(); }
        if !rows.is_null() { for (i, r) in t.into_iter().take(cap).enumerate() { rows.add(i).write(r); } }
        Ok(())
    })
}
#[no_mangle] pub unsafe extern "C" fn hexad_forward_check(s: *mut Session, steps_equal: *mut u64, nonzero_words: *mut u64) -> c_int {
    guard(|| { let (e, nz) = s.as_mut().ok_or("null session")?.forward_check()?; if !steps_equal.is_null() { *steps_equal = e; } if !nonzero_words.is_null() { *nonzero_words = nz; } Ok(()) })
}
#[no_mangle] pub unsafe extern "C" fn hexad_tl_init(s: *mut Session, v: *const HexadVec) -> c_int { guard(|| s.as_mut().ok_or("null session")?.tl_init(v.as_ref().ok_or("null vector")?)) }
#[no_mangle] pub unsafe extern "C" fn hexad_tl_step(s: *mut Session) -> c_int { guard(|| s.as_mut().ok_or("null session")?.tl_step()) }
#[no_mangle] pub unsafe extern "C" fn hexad_tl_get(s: *mut Session, v: *const HexadVec) -> c_int { guard(|| s.as_mut().ok_or("null session")?.tl_get(v.as_ref().ok_or("null vector")?)) }
#[no_mangle] pub unsafe extern "C" fn hexad_state_get(s: *mut Session, step: c_int, v: *const HexadVec) -> c_int { guard(|| s.as_mut().ok_or("null session")?.state_get(step, v.as_ref().ok_or("null vector")?)) }
#[no_mangle] pub unsafe extern "C" fn hexad_ad_init(s: *mut Session, v: *const HexadVec) -> c_int { guard(|| s.as_mut().ok_or("null session")?.ad_init(v.as_ref().ok_or("null vector")?)) }
#[no_mangle] pub unsafe extern "C" fn hexad_ad_put(s: *mut Session, v: *const HexadVec) -> c_int { guard(|| s.as_mut().ok_or("null session")?.ad_put(v.as_ref().ok_or("null vector")?)) }
#[no_mangle] pub unsafe extern "C" fn hexad_ad_step(s: *mut Session) -> c_int {
    guard(|| { let s = s.as_mut().ok_or("null session")?; let r = s.ad_step()?; eprintln!("{}", json!({"event":"hexad_reverse_step","row":r})); Ok(()) })
}
/// torch bridge: start a reverse at boundary `step` (form v1: a window cut at step boundaries).
#[no_mangle] pub unsafe extern "C" fn hexad_ad_init_at(s: *mut Session, step: c_int, v: *const HexadVec) -> c_int { guard(|| s.as_mut().ok_or("null session")?.ad_init_at(step, v.as_ref().ok_or("null vector")?)) }
/// torch bridge: the shadow of a plan allocation on the card (zero copy).
#[no_mangle] pub unsafe extern "C" fn hexad_shadow_ptr(s: *const Session, alloc: u64, offset: u64, dptr: *mut u64, bytes: *mut u64) -> c_int {
    guard(|| { let s = s.as_ref().ok_or("null session")?; let (p, n) = s.shadow_ptr(alloc, offset)?; if dptr.is_null() || bytes.is_null() { return Err("null out".into()); } *dptr = p; *bytes = n; Ok(()) })
}
/// torch bridge: the caller's function at every reversed launch named in `list` (AD_TAP's form; NULL or "" = the
/// physics tendency sum), called with the device address of the launch's cotangent view before its adjoint entries.
/// `f` NULL clears it. Process-wide (one reverse at a time).
#[no_mangle] pub unsafe extern "C" fn hexad_set_tap(f: Option<super::torch_bridge::TapFn>, user: *mut c_void, list: *const c_char) -> c_int {
    guard(|| { let l = if list.is_null() { String::new() } else { CStr::from_ptr(list).to_string_lossy().into_owned() }; super::torch_bridge::set_callback(f, user, &l) })
}
#[no_mangle] pub unsafe extern "C" fn hexad_ad_get(s: *mut Session, v: *const HexadVec) -> c_int { guard(|| s.as_mut().ok_or("null session")?.ad_get(v.as_ref().ok_or("null vector")?)) }
#[no_mangle] pub unsafe extern "C" fn hexad_get_counters(s: *const Session, c: *mut HexadCounters) -> c_int {
    guard(|| { let s = s.as_ref().ok_or("null session")?; if c.is_null() { return Err("null counters".into()); } c.write(s.counters()); Ok(()) })
}
/// The current boundary (tangent sweep) or reverse position.
#[no_mangle] pub unsafe extern "C" fn hexad_position(s: *const Session) -> c_int {
    match s.as_ref() { None => -1, Some(s) => match s.mode { Mode::Tl => s.tl_s as c_int, Mode::Ad => s.ad_s as c_int, Mode::Idle => 0 } }
}
/// print(): the operator manifest and the session receipt as JSON (copied into buf; *needed is the full length + 1).
#[no_mangle] pub unsafe extern "C" fn hexad_print(s: *const Session, buf: *mut c_char, cap: usize, needed: *mut usize) -> c_int {
    guard(|| {
        let s = s.as_ref().ok_or("null session")?;
        let mut r = s.receipt.clone();
        r["counters"] = { let c = s.counters(); json!({"h2d_bytes":c.h2d_bytes,"d2h_bytes":c.d2h_bytes,"h2d_calls":c.h2d_calls,"d2h_calls":c.d2h_calls,
            "ctx_syncs":c.ctx_syncs,"stream_syncs":c.stream_syncs,"event_syncs":c.event_syncs,"categories":["sweep","checkpoint","forward replay","glue"],
            "boundaries_ram":c.boundaries_ram,"boundaries_disk":c.boundaries_disk,"boundary_ram_bytes":c.boundary_ram_bytes,
            "disk_reads_on_path":c.disk_reads_on_path,"disk_wait_s":c.disk_wait_s,"tl_steps":c.tl_steps,"ad_steps":c.ad_steps}) };
        if let Some(w) = s.w.as_ref() { r["held_upload_dropped_cotangent_sq"] = json!(w.held_dropped); r["commands_by_rule"] = json!(w.labels); }
        // host-leftovers-2: page-locked allocation receipt for the process so far (boundary buffers, staging, direct restores).
        { let (a, u, secs, idle) = pin_stats();
          r["host_leftovers2"] = json!({"boundary_buffers_allocated":a,"boundary_buffers_reused":u,"boundary_alloc_s":secs,"boundary_buffers_fast_path":pin_registered(),
              "pool_idle_bytes":idle,"staging_allocs":crate::staging_allocs(),"restores_direct_from_page_locked":s.w.as_ref().map(|w| w.restore_direct).unwrap_or(0),
              "alloc_path":if std::env::var("AD_PIN_ALLOC").as_deref() == Ok("driver") {"cuMemHostAlloc (AD_PIN_ALLOC=driver)"} else {"parallel first touch + cuMemHostRegister"},
              "page_policy":crate::pinned::page_policy()}); }
        let text = CString::new(r.to_string()).unwrap();
        let b = text.as_bytes_with_nul();
        if !needed.is_null() { *needed = b.len(); }
        if !buf.is_null() && cap > 0 { let n = b.len().min(cap); ptr::copy_nonoverlapping(b.as_ptr().cast(), buf, n); *buf.add(n - 1) = 0; }
        Ok(())
    })
}
/// Test entries (the window's seed-direction proofs through the ABI): which 0 = <y, M x> after a tangent sweep, which 1
/// = <x, M^T y> after a reverse; p, q (aligned form), non-finite words of the arena.
#[no_mangle] pub unsafe extern "C" fn hexad_test_dot(s: *mut Session, which: c_int, seed: u64, p: *mut f64, q: *mut f64, nonfinite: *mut u64) -> c_int {
    guard(|| { let (a, b, c) = s.as_mut().ok_or("null session")?.test_dot(which, seed)?; *p = a; *q = b; *nonfinite = c; Ok(()) })
}
/// result_sha256 of the whole arena (the window's bitwise fingerprint: 16 chunks hashed on all cores).
#[no_mangle] pub unsafe extern "C" fn hexad_result_sha256(s: *const Session, out65: *mut c_char) -> c_int {
    guard(|| {
        let s = s.as_ref().ok_or("null session")?; let w = s.w.as_ref().ok_or("no arena")?;
        let _g = xfer::Cat::glue();
        let a = w.download_all(&s.ex.cuda)?; let h = result_sha256(&a);
        ptr::copy_nonoverlapping(h.as_ptr().cast(), out65, 64); *out65.add(64) = 0; Ok(())
    })
}
#[no_mangle] pub unsafe extern "C" fn hexad_test_nonzero(s: *const Session, out: *mut u64) -> c_int {
    guard(|| { let s = s.as_ref().ok_or("null session")?; *out = s.w.as_ref().ok_or("no arena")?.nonzero(&s.ex.cuda)?; Ok(()) })
}
/// Raw bytes of one allocation range at a stored boundary (host copy; for receipts and the species-slot check).
#[no_mangle] pub unsafe extern "C" fn hexad_test_boundary_bytes(s: *mut Session, step: c_int, alloc: u64, offset: u64, bytes: u64, out: *mut u8) -> c_int {
    guard(|| {
        let s = s.as_mut().ok_or("null session")?;
        let i = step as usize; if i >= s.bounds.len() { return Err(format!("boundary {i} not stored")); }
        let (_, off, n) = *s.bounds[i].index.iter().find(|x| x.0 == alloc).ok_or("allocation not in the boundary")?;
        if offset + bytes > n as u64 { return Err("range outside the allocation".into()); }
        let d = s.data(i)?; ptr::copy_nonoverlapping((&*d)[off + offset as usize..].as_ptr(), out, bytes as usize); Ok(())
    })
}
// device helpers for tests (the session's context)
#[no_mangle] pub unsafe extern "C" fn hexad_test_alloc(s: *mut Session, bytes: u64, out: *mut u64) -> c_int {
    guard(|| { let s = s.as_mut().ok_or("null session")?; let p = dalloc(&s.ex.cuda, bytes as usize)?; s.tbufs.push(p); *out = p; Ok(()) })
}
#[no_mangle] pub unsafe extern "C" fn hexad_test_upload(s: *const Session, dptr: u64, host: *const u8, bytes: u64) -> c_int {
    guard(|| { let s = s.as_ref().ok_or("null session")?; dev::upload(&s.ex.cuda, dptr, std::slice::from_raw_parts(host, bytes as usize))?; dev::sync(&s.ex.cuda) })
}
#[no_mangle] pub unsafe extern "C" fn hexad_test_download(s: *const Session, dptr: u64, host: *mut u8, bytes: u64) -> c_int {
    guard(|| { let s = s.as_ref().ok_or("null session")?; let v = dev::download(&s.ex.cuda, dptr, bytes as usize)?; ptr::copy_nonoverlapping(v.as_ptr(), host, v.len()); Ok(()) })
}
/// tmap sessions: the float64 mirror of copy (alloc, byte offset) as a device pointer, and its word count.
#[no_mangle] pub unsafe extern "C" fn hexad_test_mirror(s: *const Session, alloc: u64, offset: u64, dptr: *mut u64, words: *mut u64) -> c_int {
    guard(|| {
        let s = s.as_ref().ok_or("null session")?;
        let p = *s.mirror.get(&(alloc, offset as usize)).ok_or("no such copy")?;
        let n = s.tm.as_ref().unwrap().fields.values().find(|f| f.copies.contains(&(alloc, offset as usize))).map(|f| f.words).unwrap_or(0);
        *dptr = p; *words = n as u64; Ok(())
    })
}
#[no_mangle] pub unsafe extern "C" fn hexad_test_randn(s: *const Session, dptr: u64, n: u64, seed: u64) -> c_int {
    guard(|| { let s = s.as_ref().ok_or("null session")?; let t = s.tm.as_ref().ok_or("no T maps")?; t.k.go(&s.ex.cuda, "t_randn", n as i64, KA::new().p(dptr).l(n as i64).l(seed as i64))?; dev::sync(&s.ex.cuda) })
}
#[no_mangle] pub unsafe extern "C" fn hexad_test_dot64(s: *const Session, a: u64, b: u64, n: u64, out: *mut f64) -> c_int {
    guard(|| { let s = s.as_ref().ok_or("null session")?; let t = s.tm.as_ref().ok_or("no T maps")?; *out = t.dot(&s.ex.cuda, a, b, n as i64)?; Ok(()) })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn abi_struct_sizes_match_the_header() {
        assert_eq!(std::mem::size_of::<HexadAlloc>(), 40);
        assert_eq!(std::mem::size_of::<HexadField>(), 48);
        assert_eq!(std::mem::size_of::<HexadVec>(), 40);
        assert_eq!(std::mem::size_of::<HexadCounters>(), 7 * 32 + 4 * 8 + 8 + 16);
    }
    #[test]
    fn scale_table_refuses_per_allocation_rows() {
        let ok = json!({"schema":"hexad.scale-table/v1","tangent_exp":-40,"cotangent_exp":-66,"allocations":[{"id":1,"tangent_exp":-40,"cotangent_exp":-66}]});
        let s = parse_scale(&ok).unwrap(); assert_eq!((s.t, s.a), (-40, -66));
        let bad = json!({"schema":"hexad.scale-table/v1","tangent_exp":-40,"cotangent_exp":-66,"allocations":[{"id":1,"tangent_exp":-39,"cotangent_exp":-66}]});
        assert!(parse_scale(&bad).unwrap_err().starts_with("DEFECT"));
        assert!(parse_scale(&json!({"schema":"x"})).is_err());
    }
    #[test]
    fn manifest_names_its_operator_exactly() {
        let d = std::env::temp_dir().join(format!("hexad-manifest-{}", std::process::id()));
        let (k, b) = (d.join("k"), d.join("b"));
        fs::create_dir_all(&k).unwrap(); fs::create_dir_all(&b).unwrap();
        fs::write(k.join("kernels.json"), b"{}").unwrap(); fs::write(b.join("schemes.json"), b"{\"rows\":{}}").unwrap();
        fs::write(d.join("refuse.txt"), "solar,rlw_inatm\nrsw_post_b").unwrap();
        let ks = sha_file(&k.join("kernels.json")).unwrap(); let bs = sha_file(&b.join("schemes.json")).unwrap();
        let rs = sha_file(&d.join("refuse.txt")).unwrap();
        let m = json!({"schema":"hexad.operator-manifest/v1","fct_frozen":true,"kernel_set":{"kernels_sha256":ks},"bundle":{"schemes_sha256":bs},
            "schemes":{"native":true},"tempo":{"mode":"hold"},"radiation":{"state":"refuse-rad","refuse_file":d.join("refuse.txt"),"refuse_sha256":rs},
            "runner":{"tape_gib":2,"scratch_gib":1,"anchors":2}});
        fs::write(d.join("m.json"), serde_json::to_vec(&m).unwrap()).unwrap();
        let p = parse_manifest(&d.join("m.json"), &k, &b).unwrap();
        let args: HashMap<_, _> = p.args.into_iter().collect();
        assert_eq!(args["fct-frozen"], "1"); assert_eq!(args["native-schemes"], "1"); assert_eq!(args["refuse"], "solar,rlw_inatm,rsw_post_b");
        let mut m2 = m.clone(); m2["kernel_set"]["kernels_sha256"] = json!("0000000000000000");
        fs::write(d.join("m2.json"), serde_json::to_vec(&m2).unwrap()).unwrap();
        assert!(parse_manifest(&d.join("m2.json"), &k, &b).err().unwrap().starts_with("DEFECT"));
        let _ = fs::remove_dir_all(&d);
    }
}

/// Explicitly enable one callback in a recorded window (first add_inplace in step).
/// Callback and opaque pointer must live until finish/close; same host thread only.
#[no_mangle]
pub unsafe extern "C" fn hexad_join_one_install(s: *mut Session, step: usize,
    callback: Option<join_one::Callback>, user: *mut c_void) -> c_int {
    guard(|| {
        let s = s.as_mut().ok_or("null session")?;
        if step == 0 || step > s.steps { return Err("join step outside window".into()); }
        if s.primary.is_none() { return Err("join requires context=primary".into()); }
        if std::env::var("AD_LIVE_SELF").as_deref() == Ok("1") { return Err("join-one and AD_LIVE_SELF are exclusive".into()); }
        let w = s.w.as_mut().ok_or("join requires a window")?;
        if w.join_one.is_some() || w.graph_segment { return Err("join already installed or graph segment enabled".into()); }
        w.join_one = Some(join_one::JoinOne { step, callback: callback.ok_or("null join callback")?, user, hits: 0, failed: false });
        Ok(())
    })
}
/// Refuse a silent no-op. Detach after exactly one completed producer launch.
#[no_mangle]
pub unsafe extern "C" fn hexad_join_one_finish(s: *mut Session) -> c_int {
    guard(|| {
        let w = s.as_mut().ok_or("null session")?.w.as_mut().ok_or("no window")?;
        let j = w.join_one.as_ref().ok_or("join not installed")?;
        if j.failed || j.hits != 1 { return Err("join did not complete exactly one launch".into()); }
        w.join_one = None;
        Ok(())
    })
}
