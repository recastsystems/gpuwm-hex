// SPDX-License-Identifier: Apache-2.0
//! any-mesh mem-oocore: out-of-core memory budget for the window (--vram-budget-gb, --host-budget-gb, --spill-dir).
//!
//! THE BREAKAGE THIS PREVENTS: the TL+AD window refused, or was never tried, whenever a mesh needed more card memory
//! than one card has (120 km filled a 96 GB card; 60 km and 30 km never ran). Every arena (primal state, shadows, tape,
//! reverse scratch, TEMPO arena, native-scheme scratch) was a cuMemAlloc that either fit or failed.
//!
//! With a budget installed, every device allocation of this process goes through these trampolines:
//! - while this process's device bytes stay inside --vram-budget-gb (and the card has them), it is an ordinary
//!   cuMemAlloc / cuMemAllocAsync, exactly as before;
//! - past the budget, or when the card itself is out of memory (CUDA status 2), the allocation is made in host RAM
//!   as managed memory with the preferred location on the host and the card mapped to it (cuMemAllocManaged +
//!   cuMemAdvise SET_PREFERRED_LOCATION host + SET_ACCESSED_BY device): the same kernels read and write it over the
//!   bus, so the run gets slower and never refuses, and the card holds no more than the budget;
//! - cuMemGetInfo reports the budget (free = budget less this process's device bytes, total = the budget), so every
//!   self-sizing arena (tape, TEMPO arena, MYNN tiles, device anchors) sizes itself to the budget.
//! Host bytes (managed spill + page-locked buffers made by pinned::PinMem) are held inside --host-budget-gb through
//! `host_room`, which window.rs's MemAvailable reads; what does not fit there goes to files under --spill-dir (the
//! boundary store tier, written behind by its writer thread and read ahead by its prefetch).
//! Where bytes live is the only change: every kernel, copy and word is the same, so results are byte-identical.
//! No budget flag: nothing is installed and every call is the driver's own.
use crate::cuda::Cuda;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct MemLocation { kind: i32, id: i32 }
const LOC_DEVICE: i32 = 1;
const LOC_HOST: i32 = 2;
const ADVISE_PREFERRED: i32 = 3;
const ADVISE_ACCESSED_BY: i32 = 5;
const ATTACH_GLOBAL: u32 = 1;
const OOM: i32 = 2;

#[derive(Clone, Copy)]
struct Real {
    alloc: unsafe extern "C" fn(*mut u64, usize) -> i32,
    free: unsafe extern "C" fn(u64) -> i32,
    alloc_async: unsafe extern "C" fn(*mut u64, usize, *mut c_void) -> i32,
    free_async: unsafe extern "C" fn(u64, *mut c_void) -> i32,
    mem_info: unsafe extern "C" fn(*mut usize, *mut usize) -> i32,
    stream_sync: unsafe extern "C" fn(*mut c_void) -> i32,
    managed: unsafe extern "C" fn(*mut u64, usize, u32) -> i32,
    advise: unsafe extern "C" fn(u64, usize, i32, MemLocation) -> i32,
    /// memory (cold tier): cuMemPrefetchAsync_v2, when the driver has it.
    prefetch: Option<unsafe extern "C" fn(u64, usize, MemLocation, u32, *mut c_void) -> i32>,
    /// memory (AD_ALLOC_FILL): cuMemsetD32_v2 / cuMemsetD32Async.
    memset32: Option<unsafe extern "C" fn(u64, u32, usize) -> i32>,
    memset32_async: Option<unsafe extern "C" fn(u64, u32, usize, *mut c_void) -> i32>,
    /// memory (cold discard): cuMemDiscardAndPrefetchBatchAsync and cuMemDiscardBatchAsync, when the driver has
    /// them, and the blocking stream they are queued on, as an address (both refuse the legacy stream; 0 = none).
    discard_prefetch: Option<unsafe extern "C" fn(*mut u64, *mut usize, usize, *mut MemLocation, *mut usize, usize, u64, *mut c_void) -> i32>,
    discard: Option<unsafe extern "C" fn(*mut u64, *mut usize, usize, u64, *mut c_void) -> i32>,
    stream: usize,
    device: i32,
}
static REAL: OnceLock<Real> = OnceLock::new();
static VRAM_BUDGET: AtomicUsize = AtomicUsize::new(0);
static HOST_BUDGET: AtomicUsize = AtomicUsize::new(0);
/// --spill-mode migrate: managed spill without placement advice, so the driver pages it onto the card while a kernel
/// uses it and back out under pressure (fast when each kernel's working set fits the card; the card's free memory is
/// the cache, so the budget then bounds cuMemAlloc bytes only). Default `mapped`: host-resident, card-mapped, strict.
static MIGRATE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// address -> (bytes, managed)
static LIVE: Mutex<Option<HashMap<u64, (usize, bool)>>> = Mutex::new(None);
static DEV_NOW: AtomicUsize = AtomicUsize::new(0);
static DEV_PEAK: AtomicUsize = AtomicUsize::new(0);
static MAN_NOW: AtomicUsize = AtomicUsize::new(0);
static MAN_PEAK: AtomicUsize = AtomicUsize::new(0);
static MAN_COUNT: AtomicU64 = AtomicU64::new(0);
static MAN_OOM: AtomicU64 = AtomicU64::new(0);
static DEV_COUNT: AtomicU64 = AtomicU64::new(0);
static MAN_LARGEST: AtomicUsize = AtomicUsize::new(0);
/// memory (cold tier; --spill-cold 1 or AD_SPILL_COLD=1, with a VRAM budget). THE BREAKAGE THIS PREVENTS: with a
/// budget the allocations that came LAST were the ones spilled, whatever they were: at x1.163842 (60 km, 1 rank) that
/// was the reverse scratch every adjoint launch works in, step-2 primal blocks and the copy-row gather tables, as
/// host-resident mapped memory, while native-scheme scratch used twice a step sat on the card (a test machine
/// memory-w60-2c-st-a: 48.9 GB spilled, tangent steps 101-114 s). With the cold tier, an allocation of 64 MiB or
/// more made inside a `cold(tag)` scope (native-scheme tangent/adjoint, gather install, TEMPO io) is managed memory
/// with NO placement advice: it lives where it was last used, the driver evicts it under pressure, and every later
/// scope with the same tag pulls its blocks onto the card in bulk first (cuMemPrefetchAsync on the legacy stream).
/// The card budget then holds only what is not cold. Where bytes live is the only change.
static COLD_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static COLD_TAG: Mutex<Option<String>> = Mutex::new(None);
/// address -> (bytes, tag)
static COLD: Mutex<Option<HashMap<u64, (usize, String)>>> = Mutex::new(None);
static COLD_NOW: AtomicUsize = AtomicUsize::new(0);
static COLD_PEAK: AtomicUsize = AtomicUsize::new(0);
static COLD_COUNT: AtomicU64 = AtomicU64::new(0);
static PF_CALLS: AtomicU64 = AtomicU64::new(0);
static PF_FAIL: AtomicU64 = AtomicU64::new(0);
static PF_BYTES: AtomicU64 = AtomicU64::new(0);
/// machine: wall time inside the prefetch calls (PF_NS), inside the diagnostic wait after them (PF_SYNC_NS,
/// AD_COLD_PF_SYNC=1 only), and per tag family [calls, bytes asked, call ns, wait ns] (gather:<hash> counts as gather).
static PF_NS: AtomicU64 = AtomicU64::new(0);
static PF_SYNC_NS: AtomicU64 = AtomicU64::new(0);
static PF_TAGS: Mutex<Option<std::collections::BTreeMap<String, [u64; 4]>>> = Mutex::new(None);
fn pf_sync() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("AD_COLD_PF_SYNC").as_deref() == Ok("1"))
}
fn pf_tags() -> serde_json::Value {
    let m = PF_TAGS.lock().ok().and_then(|g| g.clone()).unwrap_or_default();
    serde_json::json!({"fields":"[calls, bytes asked, seconds in the calls, seconds in the diagnostic wait]",
        "rows":m.iter().map(|(k, v)| serde_json::json!([k, v[0], v[1], v[2] as f64 / 1e9, v[3] as f64 / 1e9])).collect::<Vec<_>>()})
}
const COLD_MIN: usize = 64 << 20;
/// memory (AD_ALLOC_FILL=big|nan|zero|<8 hex digits>; diagnostic, never a timing). THE BREAKAGE THIS FINDS: a 480 km
/// gate leg on a shared, nearly full card ended rc 0 with a wrong adjoint (a test machine <local path>: dot
/// 1.25e-5, other words; the same settings give the reference words on a card with room). A buffer read before it is
/// written gives the reference words only while fresh card memory happens to be zero. With this set, every device
/// allocation this process makes through the driver table is filled with the 32-bit pattern before it is returned
/// (big = 1e30 as float32, 0x7149f2ca), on any card and with or without a VRAM budget, so such a read shows as wrong
/// words, a failed dot, a non-finite cotangent or an illegal address, in every run and not once in six.
/// `zero` fills zeros (does the defect go away when fresh memory is zero by construction).
static FILL_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static FILL_PATTERN: AtomicU64 = AtomicU64::new(0);
static FILL_COUNT: AtomicU64 = AtomicU64::new(0);
static FILL_BYTES: AtomicU64 = AtomicU64::new(0);
static FILL_FAIL: AtomicU64 = AtomicU64::new(0);
fn fill_pattern() -> Result<Option<u32>, String> {
    match std::env::var("AD_ALLOC_FILL").ok().as_deref() {
        None | Some("") | Some("0") => Ok(None),
        Some("big") => Ok(Some(0x7149_f2ca)),
        Some("nan") => Ok(Some(0x7fc0_0000)),
        Some("zero") => Ok(Some(0)),
        Some(h) => u32::from_str_radix(h.trim_start_matches("0x"), 16).map(Some).map_err(|_| format!("AD_ALLOC_FILL={h}: big, nan, zero or 8 hex digits")),
    }
}
/// Fill a new allocation (stream-ordered when it was made on a stream).
unsafe fn fill(r: &Real, a: u64, bytes: usize, stream: Option<*mut c_void>) {
    if !FILL_ON.load(Relaxed) { return; }
    let n = bytes / 4;
    if n == 0 { return; }
    let pat = FILL_PATTERN.load(Relaxed) as u32;
    let st = match (stream, r.memset32_async, r.memset32) {
        (Some(s), Some(f), _) => f(a, pat, n, s),
        (_, _, Some(f)) => f(a, pat, n),
        _ => -1,
    };
    FILL_COUNT.fetch_add(1, Relaxed);
    if st != 0 { FILL_FAIL.fetch_add(1, Relaxed); } else { FILL_BYTES.fetch_add(4 * n as u64, Relaxed); }
}
/// memory (cold discard; AD_COLD_DISCARD=<tag families>, e.g. p3,p4,p6,ruc; with the cold tier). THE BREAKAGE THIS
/// PREVENTS: a cold block is scheme scratch (the MYNN-EDMF arena and store, the drag and surface-layer frames and
/// tapes, the RUC land tape, the sea-ice block): every launch writes it before it reads it and nothing reads it after
/// the launch. Yet when a cold scope opened, the prefetch moved the block's old bytes from the host to the card, and
/// when the card ran short the driver copied them back out: at x1.163842 on 4 cards the dycore card is asked for
/// 83.2 GB that cannot move plus 25.5 to 35.8 GB of cold blocks on a 102.6 GB card, and a 2-step window asks for
/// 211.8 to 461.5 GB of prefetch (a test machine <local path> and <local path>, window.json
/// out_of_core.cold; mynn 01:47Z). For a family named in AD_COLD_DISCARD the scope entry discards and prefetches
/// its blocks in one call (the driver gives the card fresh pages and moves no bytes) and the scope exit discards
/// them (the driver may drop them from the card without copying them out). Only for families whose blocks are dead
/// between launches: AD_COLD_SCRUB=1 is the proof, it fills every such block with 1e30f at scope entry and exit
/// instead of discarding it, so a block that IS read across launches gives other result words. Families not named
/// (gather tables, tempo_io) keep their bytes and the plain prefetch. Where bytes live is the only change.
static DEAD_FAM: OnceLock<Vec<String>> = OnceLock::new();
static DP_CALLS: AtomicU64 = AtomicU64::new(0);
static DP_BYTES: AtomicU64 = AtomicU64::new(0);
static DP_FAIL: AtomicU64 = AtomicU64::new(0);
static DD_CALLS: AtomicU64 = AtomicU64::new(0);
static DD_BYTES: AtomicU64 = AtomicU64::new(0);
static DD_FAIL: AtomicU64 = AtomicU64::new(0);
static DD_NS: AtomicU64 = AtomicU64::new(0);
static SCRUB_BYTES: AtomicU64 = AtomicU64::new(0);
const SCRUB_PATTERN: u32 = 0x7149_f2ca;
fn dead_families() -> &'static [String] {
    DEAD_FAM.get_or_init(|| std::env::var("AD_COLD_DISCARD").ok().map(|v| v.split(',').map(|t| t.trim().to_owned()).filter(|t| !t.is_empty()).collect()).unwrap_or_default())
}
fn scrub() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("AD_COLD_SCRUB").as_deref() == Ok("1"))
}
fn dead(tag: &str) -> bool { let fam = tag.split(':').next().unwrap_or(tag); dead_families().iter().any(|f| f == fam) }
fn tagged(tag: &str) -> Vec<(u64, usize)> {
    COLD.lock().ok().and_then(|g| g.as_ref().map(|m| m.iter().filter(|(_, v)| v.1 == tag).map(|(a, v)| (*a, v.0)).collect())).unwrap_or_default()
}
/// Scope entry or exit of a dead family. Entry: discard + prefetch to the card. Exit: discard. Whole 2 MiB pieces
/// only (a block's tail keeps its bytes). In proof mode: the plain prefetch at entry, then the fill, at both ends.
fn dead_edge(tag: &str, entry: bool) {
    let Some(r) = REAL.get() else { return };
    let blocks = tagged(tag);
    if blocks.is_empty() { return; }
    let t0 = std::time::Instant::now();
    if scrub() {
        if entry { prefetch(tag); }
        for (a, b) in &blocks {
            let n = b / 4;
            // SAFETY: a live managed block of this process; stream-ordered on the legacy stream after the launch's work.
            let st = unsafe { match (r.memset32_async, r.memset32) { (Some(f), _) => f(*a, SCRUB_PATTERN, n, std::ptr::null_mut()), (_, Some(f)) => f(*a, SCRUB_PATTERN, n), _ => -1 } };
            if st == 0 { SCRUB_BYTES.fetch_add(4 * n as u64, Relaxed); } else { DD_FAIL.fetch_add(1, Relaxed); }
        }
        DD_NS.fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
        return;
    }
    let (mut ptrs, mut sizes): (Vec<u64>, Vec<usize>) = blocks.iter().map(|(a, b)| (*a, b & !((2usize << 20) - 1))).filter(|x| x.1 > 0).unzip();
    if ptrs.is_empty() { return; }
    let bytes: u64 = sizes.iter().map(|b| *b as u64).sum();
    let stream = r.stream as *mut c_void;
    // SAFETY: live managed blocks of this process, the driver entry points resolved at install, a blocking stream of
    // this context: the legacy stream's work queued before this call finishes first and its later work waits.
    let st = unsafe {
        if entry {
            match r.discard_prefetch {
                Some(f) => { let (mut loc, mut idx) = ([MemLocation { kind: LOC_DEVICE, id: r.device }], [0usize]);
                             f(ptrs.as_mut_ptr(), sizes.as_mut_ptr(), ptrs.len(), loc.as_mut_ptr(), idx.as_mut_ptr(), 1, 0, stream) },
                None => -1,
            }
        } else {
            match r.discard { Some(f) => f(ptrs.as_mut_ptr(), sizes.as_mut_ptr(), ptrs.len(), 0, stream), None => -1 }
        }
    };
    if entry {
        DP_CALLS.fetch_add(1, Relaxed);
        if st == 0 { DP_BYTES.fetch_add(bytes, Relaxed); } else { DP_FAIL.fetch_add(1, Relaxed); prefetch(tag); }
        // diagnostic only (AD_COLD_PF_SYNC=1): the time the driver takes for the batch
        if st == 0 && pf_sync() { let t1 = std::time::Instant::now(); let _ = unsafe { (r.stream_sync)(stream) }; PF_SYNC_NS.fetch_add(t1.elapsed().as_nanos() as u64, Relaxed); }
    } else {
        DD_CALLS.fetch_add(1, Relaxed);
        if st == 0 { DD_BYTES.fetch_add(bytes, Relaxed); } else { DD_FAIL.fetch_add(1, Relaxed); }
    }
    DD_NS.fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
}
fn discard_stats() -> serde_json::Value {
    serde_json::json!({"families":dead_families(),"proof_fill":scrub(),"available":REAL.get().map(|r| r.discard_prefetch.is_some() && r.discard.is_some() && r.stream != 0),
        "entry_calls":DP_CALLS.load(Relaxed),"entry_bytes":DP_BYTES.load(Relaxed),"entry_failures":DP_FAIL.load(Relaxed),
        "exit_calls":DD_CALLS.load(Relaxed),"exit_bytes":DD_BYTES.load(Relaxed),"exit_failures":DD_FAIL.load(Relaxed),
        "proof_fill_bytes":SCRUB_BYTES.load(Relaxed),"call_s":DD_NS.load(Relaxed) as f64 / 1e9})
}
pub fn set_cold(on: bool) { COLD_ON.store(on, Relaxed); }
/// Cumulative cold-tier counters (for a reverse-step row).
pub fn cold_stats() -> serde_json::Value {
    serde_json::json!({"on":COLD_ON.load(Relaxed),"bytes_now":COLD_NOW.load(Relaxed),"prefetch_calls":PF_CALLS.load(Relaxed),
                       "prefetch_failures":PF_FAIL.load(Relaxed),"prefetch_bytes_requested":PF_BYTES.load(Relaxed),
                       "prefetch_call_s":PF_NS.load(Relaxed) as f64 / 1e9,"prefetch_diagnostic_wait_s":PF_SYNC_NS.load(Relaxed) as f64 / 1e9,
                       "prefetch_by_tag":pf_tags(),"discard":discard_stats()})
}
fn cold_on() -> bool { COLD_ON.load(Relaxed) && REAL.get().is_some() }
/// A cold scope: allocations of 64 MiB or more made while it is open are cold, tagged `tag`; opening it prefetches
/// the cold blocks that already carry the tag. Scopes nest (the inner tag wins, the outer one comes back).
pub struct ColdScope { prev: Option<String>, on: bool, dead: Option<String> }
pub fn cold(tag: &str) -> ColdScope {
    if !cold_on() { return ColdScope { prev: None, on: false, dead: None }; }
    let prev = COLD_TAG.lock().ok().and_then(|mut g| g.replace(tag.to_owned()));
    let is_dead = dead(tag);
    if is_dead { dead_edge(tag, true); } else { prefetch(tag); }
    ColdScope { prev, on: true, dead: is_dead.then(|| tag.to_owned()) }
}
impl Drop for ColdScope {
    fn drop(&mut self) {
        if !self.on { return; }
        if let Some(t) = self.dead.take() { dead_edge(&t, false); }
        if let Ok(mut g) = COLD_TAG.lock() { *g = self.prev.take(); }
    }
}
/// s60-cardmem: live cold bytes by family (the tag before ':'), largest first. One family's scheme runs at a time, so
/// the card must keep the largest family's bytes free beside what is resident (card_room::reserve).
pub fn cold_families() -> Vec<(String, usize)> {
    let mut by: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    if let Ok(g) = COLD.lock() { if let Some(m) = g.as_ref() { for (b, t) in m.values() { *by.entry(t.split(':').next().unwrap_or(t).to_owned()).or_default() += b; } } }
    let mut v: Vec<(String, usize)> = by.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v
}
/// s60-cardmem: this process's device bytes now (not managed, not cold).
pub fn device_now() -> usize { DEV_NOW.load(Relaxed) }
/// 60 km tape: the device bytes (as `device_now` counts them: not managed) of these live allocations.
pub fn device_bytes_of(addrs: &[u64]) -> usize {
    let Ok(g) = LIVE.lock() else { return 0 };
    let Some(m) = g.as_ref() else { return 0 };
    addrs.iter().filter_map(|a| m.get(a)).filter(|(_, managed)| !managed).map(|(b, _)| *b).sum()
}
#[cfg(test)]
mod s60_tape_tests {
    use super::*;
    #[test]
    fn device_bytes_of_counts_only_live_device_blocks() {
        // addresses no allocator returns (odd, high); untracked again at the end, so the counters come back
        let (d, m, gone) = (0xfff0_0000_0000_0001u64, 0xfff0_0000_0000_0003u64, 0xfff0_0000_0000_0005u64);
        track(d, 4_096, false);
        track(m, 3_000, true);
        assert_eq!(device_bytes_of(&[d, m, gone]), 4_096);
        assert_eq!(device_bytes_of(&[]), 0);
        untrack(d); untrack(m);
        assert_eq!(device_bytes_of(&[d, m]), 0);
    }
}
/// s60-cardmem: while a warm scope is open, allocations are never cold, whatever cold scope is open around it (its tag
/// comes back on drop). THE BREAKAGE: a ring grown inside the boundary layer's cold scope became managed memory the
/// driver copies out to the host when the next family's scheme runs, and back for the next launch, though nothing
/// in it lives past the launch.
pub struct WarmScope { prev: Option<String>, on: bool }
pub fn warm() -> WarmScope {
    if !cold_on() { return WarmScope { prev: None, on: false }; }
    let prev = COLD_TAG.lock().ok().and_then(|mut g| g.take());
    WarmScope { prev, on: true }
}
impl Drop for WarmScope {
    fn drop(&mut self) { if self.on { if let Ok(mut g) = COLD_TAG.lock() { *g = self.prev.take(); } } }
}
/// Pull every cold block tagged `tag` onto the card (stream-ordered on the legacy stream; a block already there costs
/// nothing). Without the driver entry the blocks migrate on first touch instead.
pub fn prefetch(tag: &str) {
    if !cold_on() { return; }
    let Some(r) = REAL.get() else { return };
    let Some(pf) = r.prefetch else { return };
    let blocks: Vec<(u64, usize)> = COLD.lock().ok().and_then(|g| g.as_ref().map(|m| m.iter().filter(|(_, v)| v.1 == tag).map(|(a, v)| (*a, v.0)).collect())).unwrap_or_default();
    if blocks.is_empty() { return; }
    let (t0, mut calls, mut asked) = (std::time::Instant::now(), 0u64, 0u64);
    for (a, b) in blocks {
        // SAFETY: a live managed block of this process, the driver entry point resolved at install.
        let st = unsafe { pf(a, b, MemLocation { kind: LOC_DEVICE, id: r.device }, 0, std::ptr::null_mut()) };
        PF_CALLS.fetch_add(1, Relaxed);
        calls += 1;
        if st != 0 { PF_FAIL.fetch_add(1, Relaxed); } else { PF_BYTES.fetch_add(b as u64, Relaxed); asked += b as u64; }
    }
    let call_ns = t0.elapsed().as_nanos() as u64;
    // diagnostic only: wait for the legacy stream, so the time the driver takes to move the blocks is read here
    let wait_ns = if pf_sync() { let t1 = std::time::Instant::now(); let _ = unsafe { (r.stream_sync)(std::ptr::null_mut()) }; t1.elapsed().as_nanos() as u64 } else { 0 };
    PF_NS.fetch_add(call_ns, Relaxed);
    PF_SYNC_NS.fetch_add(wait_ns, Relaxed);
    let fam = tag.split(':').next().unwrap_or(tag).to_owned();
    if let Ok(mut g) = PF_TAGS.lock() {
        let e = g.get_or_insert_with(Default::default).entry(fam).or_insert([0; 4]);
        e[0] += calls; e[1] += asked; e[2] += call_ns; e[3] += wait_ns;
    }
}
unsafe fn managed_cold(r: &Real, p: *mut u64, bytes: usize, tag: String) -> i32 {
    let s = (r.managed)(p, bytes.max(1), ATTACH_GLOBAL);
    if s != 0 { return s; }
    if let Ok(mut g) = COLD.lock() { g.get_or_insert_with(HashMap::new).insert(*p, (bytes, tag.clone())); }
    let n = COLD_NOW.fetch_add(bytes, Relaxed) + bytes;
    COLD_PEAK.fetch_max(n, Relaxed);
    COLD_COUNT.fetch_add(1, Relaxed);
    track_as(*p, bytes, true, Some(&tag));
    0
}
/// The open cold scope's tag for an allocation of `bytes`, if it is to be cold.
fn cold_tag(bytes: usize) -> Option<String> {
    if bytes < COLD_MIN || !COLD_ON.load(Relaxed) { return None; }
    COLD_TAG.lock().ok().and_then(|g| g.clone())
}
/// memory (allocation census): the phase the window is in (set by window.rs), and for every live allocation of
/// 64 MiB or more the phase it was made in. Read by `census`; nothing here changes where bytes live.
static PHASE: Mutex<String> = Mutex::new(String::new());
static BIG: Mutex<Option<HashMap<u64, (usize, bool, String, u64)>>> = Mutex::new(None);
static SEQ: AtomicU64 = AtomicU64::new(0);
const BIG_BYTES: usize = 64 << 20;
pub fn on() -> bool { REAL.get().is_some() }
pub fn phase(p: &str) { if on() { if let Ok(mut g) = PHASE.lock() { g.clear(); g.push_str(p); } } }
/// Live allocations of 64 MiB or more, largest first (at most `top`), and their sums by phase and by kind.
pub fn census(top: usize) -> serde_json::Value {
    let mut rows: Vec<(usize, bool, String, u64)> = BIG.lock().ok().and_then(|g| g.as_ref().map(|m| m.values().cloned().collect())).unwrap_or_default();
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(a.3.cmp(&b.3)));
    let mut by: std::collections::BTreeMap<String, (usize, usize)> = std::collections::BTreeMap::new();
    for (b, m, p, _) in &rows { let e = by.entry(p.clone()).or_default(); if *m { e.1 += b; } else { e.0 += b; } }
    let (dev_big, man_big): (usize, usize) = rows.iter().fold((0, 0), |a, r| if r.1 { (a.0, a.1 + r.0) } else { (a.0 + r.0, a.1) });
    serde_json::json!({"device_bytes_now":DEV_NOW.load(Relaxed),"spilled_bytes_now":MAN_NOW.load(Relaxed),"device_bytes_peak":DEV_PEAK.load(Relaxed),
        "cold_bytes_now":COLD_NOW.load(Relaxed),"mapped_spill_bytes_now":MAN_NOW.load(Relaxed).saturating_sub(COLD_NOW.load(Relaxed)),
        "big_allocations":rows.len(),"big_device_bytes":dev_big,"big_spilled_bytes":man_big,
        "fields":"[bytes, spilled to managed host memory, phase it was made in, allocation sequence number]",
        "largest":rows.iter().take(top).map(|(b, m, p, q)| serde_json::json!([b, m, p, q])).collect::<Vec<_>>(),
        "by_phase_device_and_spilled_bytes":by.iter().map(|(p, (d, m))| serde_json::json!([p, d, m])).collect::<Vec<_>>()})
}

fn track(a: u64, bytes: usize, managed: bool) { track_as(a, bytes, managed, None) }
fn track_as(a: u64, bytes: usize, managed: bool, cold: Option<&str>) {
    if let Ok(mut g) = LIVE.lock() { g.get_or_insert_with(HashMap::new).insert(a, (bytes, managed)); }
    let q = SEQ.fetch_add(1, Relaxed);
    if bytes >= BIG_BYTES {
        let p = PHASE.lock().map(|g| g.clone()).unwrap_or_default();
        let p = match cold { Some(t) => format!("COLD[{t}] {p}"), None => p };
        if let Ok(mut g) = BIG.lock() { g.get_or_insert_with(HashMap::new).insert(a, (bytes, managed, p, q)); }
    }
    if managed {
        let n = MAN_NOW.fetch_add(bytes, Relaxed) + bytes;
        MAN_PEAK.fetch_max(n, Relaxed);
        MAN_COUNT.fetch_add(1, Relaxed);
        MAN_LARGEST.fetch_max(bytes, Relaxed);
    } else {
        let n = DEV_NOW.fetch_add(bytes, Relaxed) + bytes;
        DEV_PEAK.fetch_max(n, Relaxed);
        DEV_COUNT.fetch_add(1, Relaxed);
    }
}
fn untrack(a: u64) -> Option<(usize, bool)> {
    let r = LIVE.lock().ok().and_then(|mut g| g.as_mut().and_then(|m| m.remove(&a)));
    if let Ok(mut g) = BIG.lock() { if let Some(m) = g.as_mut() { m.remove(&a); } }
    if let Some((b, _)) = COLD.lock().ok().and_then(|mut g| g.as_mut().and_then(|m| m.remove(&a))) { COLD_NOW.fetch_sub(b, Relaxed); }
    if let Some((b, m)) = r { if m { MAN_NOW.fetch_sub(b, Relaxed); } else { DEV_NOW.fetch_sub(b, Relaxed); } }
    r
}
/// Whether a device allocation of `bytes` stays inside the VRAM budget.
fn fits(bytes: usize) -> bool {
    let b = VRAM_BUDGET.load(Relaxed);
    b == 0 || DEV_NOW.load(Relaxed).saturating_add(bytes) <= b
}
unsafe fn managed(r: &Real, p: *mut u64, bytes: usize, on_oom: bool) -> i32 {
    let s = (r.managed)(p, bytes.max(1), ATTACH_GLOBAL);
    if s != 0 { return s; }
    // Host-resident, mapped for the card: no migration, no fault thrash, the card holds none of it.
    if !MIGRATE.load(Relaxed) {
    let _ = (r.advise)(*p, bytes.max(1), ADVISE_PREFERRED, MemLocation { kind: LOC_HOST, id: 0 });
    let _ = (r.advise)(*p, bytes.max(1), ADVISE_ACCESSED_BY, MemLocation { kind: LOC_DEVICE, id: r.device });
    }
    if on_oom { MAN_OOM.fetch_add(1, Relaxed); }
    track(*p, bytes, true);
    0
}
unsafe fn alloc_inner(r: &Real, p: *mut u64, bytes: usize) -> i32 {
    if let Some(tag) = cold_tag(bytes) { return managed_cold(r, p, bytes, tag); }
    if fits(bytes) {
        let s = (r.alloc)(p, bytes);
        if s == 0 { track(*p, bytes, false); return 0; }
        // no VRAM budget (AD_ALLOC_FILL alone): an out-of-memory card is the error it always was
        if s != OOM || VRAM_BUDGET.load(Relaxed) == 0 { return s; }
        return managed(r, p, bytes, true);
    }
    managed(r, p, bytes, false)
}
unsafe extern "C" fn b_alloc(p: *mut u64, bytes: usize) -> i32 {
    let r = REAL.get().unwrap();
    let s = alloc_inner(r, p, bytes);
    if s == 0 { fill(r, *p, bytes, None); }
    s
}
unsafe extern "C" fn b_free(a: u64) -> i32 {
    let r = REAL.get().unwrap();
    untrack(a);
    (r.free)(a)
}
unsafe fn alloc_async_inner(r: &Real, p: *mut u64, bytes: usize, s: *mut c_void) -> i32 {
    if let Some(tag) = cold_tag(bytes) { return managed_cold(r, p, bytes, tag); }
    if fits(bytes) {
        let st = (r.alloc_async)(p, bytes, s);
        if st == 0 { track(*p, bytes, false); return 0; }
        if st != OOM || VRAM_BUDGET.load(Relaxed) == 0 { return st; }
        return managed(r, p, bytes, true);
    }
    managed(r, p, bytes, false)
}
unsafe extern "C" fn b_alloc_async(p: *mut u64, bytes: usize, s: *mut c_void) -> i32 {
    let r = REAL.get().unwrap();
    let st = alloc_async_inner(r, p, bytes, s);
    if st == 0 { fill(r, *p, bytes, Some(s)); }
    st
}
unsafe extern "C" fn b_free_async(a: u64, s: *mut c_void) -> i32 {
    let r = REAL.get().unwrap();
    match untrack(a) {
        // A managed block is freed in stream order: the stream's work that reads it finishes first.
        Some((_, true)) => { let st = (r.stream_sync)(s); if st != 0 { return st; } (r.free)(a) },
        _ => (r.free_async)(a, s),
    }
}
unsafe extern "C" fn b_mem_info(free: *mut usize, total: *mut usize) -> i32 {
    let r = REAL.get().unwrap();
    let st = (r.mem_info)(free, total);
    if st != 0 { return st; }
    let b = VRAM_BUDGET.load(Relaxed);
    if b > 0 {
        // memory (cold tier): cold blocks cached on the card are evictable, so they do not lower the room.
        *free = if COLD_ON.load(Relaxed) { b.saturating_sub(DEV_NOW.load(Relaxed)) } else { (*free).min(b.saturating_sub(DEV_NOW.load(Relaxed))) };
        *total = (*total).min(b);
    }
    st
}

fn gib(v: Option<&str>) -> Result<Option<usize>, String> {
    match v {
        None => Ok(None),
        Some(s) => match s.parse::<f64>() {
            Ok(g) if g.is_finite() && g > 0.0 => Ok(Some((g * (1u64 << 30) as f64) as usize)),
            _ => Err(format!("budget {s}: a size in GiB > 0")),
        },
    }
}

/// Install the budget on this process's driver table (before the first device allocation of the run).
/// `vram`/`host`: --vram-budget-gb / --host-budget-gb values. Returns whether a VRAM budget was installed.
pub unsafe fn install(c: &mut Cuda, vram: Option<&str>, host: Option<&str>, mode: Option<&str>) -> Result<bool, String> {
    let (v, h) = (gib(vram)?, gib(host)?);
    match mode { None | Some("mapped") => {}, Some("migrate") => MIGRATE.store(true, Relaxed),
                 Some(m) => return Err(format!("--spill-mode {m}: mapped or migrate")) }
    if let Some(h) = h { HOST_BUDGET.store(h, Relaxed); }
    let fillp = fill_pattern()?;
    if v.is_none() && fillp.is_none() { return Ok(false); }
    let mut device = 0i32;
    crate::cuda::checked("cuCtxGetDevice", (c.ctx_device)(&mut device))?;
    // memory (cold discard): the two entry points and their stream, only when a family is named and not in proof mode
    let (mut discard_prefetch, mut discard, mut stream) = (None, None, 0usize);
    if v.is_some() && !dead_families().is_empty() && !scrub() {
        discard_prefetch = crate::cuda::symbol(c.library, "cuMemDiscardAndPrefetchBatchAsync").ok();
        discard = crate::cuda::symbol(c.library, "cuMemDiscardBatchAsync").ok();
        if discard_prefetch.is_none() || discard.is_none() { return Err("AD_COLD_DISCARD: the driver has no cuMemDiscardAndPrefetchBatchAsync / cuMemDiscardBatchAsync".into()); }
        let mut st: *mut c_void = std::ptr::null_mut();
        crate::cuda::checked("cuStreamCreate cold discard", (c.stream_create)(&mut st, 0))?;
        stream = st as usize;
    }
    let real = Real { alloc: c.allocate, free: c.free, alloc_async: c.allocate_async, free_async: c.free_async,
                      mem_info: c.mem_info, stream_sync: c.stream_synchronize,
                      managed: crate::cuda::symbol(c.library, "cuMemAllocManaged")?,
                      advise: crate::cuda::symbol(c.library, "cuMemAdvise_v2")?,
                      prefetch: crate::cuda::symbol(c.library, "cuMemPrefetchAsync_v2").ok(),
                      memset32: crate::cuda::symbol(c.library, "cuMemsetD32_v2").ok(),
                      memset32_async: crate::cuda::symbol(c.library, "cuMemsetD32Async").ok(), discard_prefetch, discard, stream, device };
    if let Some(pat) = fillp {
        if real.memset32.is_none() { return Err("AD_ALLOC_FILL: the driver has no cuMemsetD32_v2".into()); }
        FILL_PATTERN.store(pat as u64, Relaxed);
        FILL_ON.store(true, Relaxed);
        eprintln!("{}", serde_json::json!({"event":"alloc_fill","pattern":format!("{pat:08x}"),"vram_budget":v.is_some(),
            "note":"diagnostic: every new device allocation is filled with this 32-bit pattern; not a timing run"}));
    }
    if REAL.set(real).is_err() { return Err("spill budget installed twice".into()); }
    // no budget (AD_ALLOC_FILL alone): 0 = unlimited (fits), cuMemGetInfo passes through, an out-of-memory card errors
    VRAM_BUDGET.store(v.unwrap_or(0), Relaxed);
    c.allocate = b_alloc; c.free = b_free; c.allocate_async = b_alloc_async; c.free_async = b_free_async; c.mem_info = b_mem_info;
    Ok(v.is_some())
}
/// Host bytes this process may still take for RAM tiers: --host-budget-gb less the managed spill and the page-locked
/// buffers made by pinned::PinMem (None: no host budget).
pub fn host_room() -> Option<usize> {
    let h = HOST_BUDGET.load(Relaxed);
    if h == 0 { return None; }
    Some(h.saturating_sub(MAN_NOW.load(Relaxed) + crate::pinned::live_bytes()))
}
pub fn receipt() -> serde_json::Value {
    serde_json::json!({"vram_budget_bytes":VRAM_BUDGET.load(Relaxed),"host_budget_bytes":HOST_BUDGET.load(Relaxed),
        "device_bytes_peak":DEV_PEAK.load(Relaxed),"device_bytes_now":DEV_NOW.load(Relaxed),"device_allocations":DEV_COUNT.load(Relaxed),
        "spilled_bytes_peak":MAN_PEAK.load(Relaxed),"spilled_bytes_now":MAN_NOW.load(Relaxed),"spilled_allocations":MAN_COUNT.load(Relaxed),
        "spilled_on_card_oom":MAN_OOM.load(Relaxed),"largest_spilled_bytes":MAN_LARGEST.load(Relaxed),
        "pinned_host_bytes_now":crate::pinned::live_bytes(),"census":census(60),
        "alloc_fill":{"on":FILL_ON.load(Relaxed),"pattern":format!("{:08x}", FILL_PATTERN.load(Relaxed) as u32),"allocations":FILL_COUNT.load(Relaxed),
                      "bytes":FILL_BYTES.load(Relaxed),"failures":FILL_FAIL.load(Relaxed)},
        "cold":{"on":COLD_ON.load(Relaxed),"bytes_now":COLD_NOW.load(Relaxed),"bytes_peak":COLD_PEAK.load(Relaxed),"allocations":COLD_COUNT.load(Relaxed),
                "prefetch_available":REAL.get().map(|r| r.prefetch.is_some()),"prefetch_calls":PF_CALLS.load(Relaxed),"prefetch_failures":PF_FAIL.load(Relaxed),
                "prefetch_bytes_requested":PF_BYTES.load(Relaxed),
                "prefetch_call_s":PF_NS.load(Relaxed) as f64 / 1e9,"prefetch_diagnostic_wait_s":PF_SYNC_NS.load(Relaxed) as f64 / 1e9,"prefetch_by_tag":pf_tags(),
                "discard":discard_stats(),
                "note":"cold bytes are counted in spilled_bytes_* too; mapped (strict) spill = spilled - cold"},
        "spill_kind":if MIGRATE.load(Relaxed) { "managed, driver-migrated on use (cuMemAllocManaged, no placement advice)" } else { "managed host-resident, card-mapped (cuMemAllocManaged + preferred location host + accessed-by device)" }})
}
