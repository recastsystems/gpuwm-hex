// SPDX-License-Identifier: Apache-2.0
//! S04 (taylor-row-host): a reusable page-locked host buffer.
//!
//! Whole-state device-to-host and host-to-device copies (the per-step host digest's carried bytes, the reverse
//! step's mid-step anchors) into fresh pageable vectors cost one host core twice over: the driver copies through
//! its own staging buffer with a host memcpy, and every fresh page faults and is zeroed on first touch. Into a
//! page-locked buffer that is allocated once and reused, the copy is DMA and the host core is free. The bytes are
//! the same bytes; only where they sit changes.
//!
//! host-leftovers-2 (host-audit FIXES2 open item 7: 19.1 s of cuMemHostAlloc for the 19 boundary buffers of outer
//! loop 1 on a test machine, card idle; 36.6 s for the same 19 x 3.44 GiB on a test machine): `PinMem` makes a large page-locked
//! buffer as anonymous memory on 2 MiB transparent huge pages, first-touched on every core, then page-locked with
//! cuMemHostRegister. cuMemHostAlloc faults, zeroes and locks 4 KiB pages on one thread inside the driver; here the
//! zeroing runs on all cores and the driver locks 512x fewer pages. Measured (pinbench, a test machine, 19 x 3.44 GiB):
//! cuMemHostAlloc 36.6 s, this 3.6 s (touch 1.0 s + register 2.6 s), free 13.1 -> 1.1 s; device-to-host 52.6 GiB/s
//! into both. The memory is page-locked for the current context exactly as cuMemHostAlloc(flags 0) memory is, and a
//! fresh anonymous page reads zero as a fresh cuMemHostAlloc page does. Buffers under PIN_FAST_MIN, a failed
//! register, or AD_PIN_ALLOC=driver take cuMemHostAlloc as before.
use crate::cuda::{checked, Cuda, PtrCall};
use std::ffi::c_void;

/// The driver entry points a page-locked buffer needs (Copy, so a helper thread can carry them).
#[derive(Clone, Copy)]
pub struct PinFns { pub host_alloc: unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> i32, pub host_free: PtrCall,
                    pub register: unsafe extern "C" fn(*mut c_void, usize, u32) -> i32, pub unregister: PtrCall }
impl PinFns { pub fn of(c: &Cuda) -> Self { Self { host_alloc: c.host_alloc, host_free: c.host_free, register: c.host_register, unregister: c.host_unregister } } }

#[link(name = "c")]
extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
    fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
}
const HUGE: usize = 2 << 20;
/// host-leftovers-2: every live page-locked range made here [start, end), so a copy can tell that its source is already
/// page-locked (Win::restore DMAs straight from it instead of staging it through its own buffer first).
/// host-syncs: the third field is the registration piece size (0: one registration for the whole range).
static LIVE: std::sync::Mutex<Vec<(usize, usize, usize)>> = std::sync::Mutex::new(Vec::new());
/// host-syncs: number of live piece-registered buffers; the copy wrappers skip the LIVE lookup while it is 0.
static CHUNKED_LIVE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
fn live_add(p: *mut u8, cap: usize, chunk: usize) {
    if let Ok(mut g) = LIVE.lock() { g.push((p as usize, p as usize + cap, chunk)); }
    if chunk != 0 { CHUNKED_LIVE.fetch_add(1, std::sync::atomic::Ordering::SeqCst); }
}
fn live_drop(p: *mut u8) {
    if let Ok(mut g) = LIVE.lock() { if let Some(i) = g.iter().position(|r| r.0 == p as usize) {
        if g[i].2 != 0 { CHUNKED_LIVE.fetch_sub(1, std::sync::atomic::Ordering::SeqCst); }
        g.swap_remove(i);
    } }
}
/// mem-oocore: page-locked bytes held now by every live PinMem (the host budget counts them).
pub fn live_bytes() -> usize { LIVE.lock().map(|g| g.iter().map(|r| r.1 - r.0).sum()).unwrap_or(0) }
/// Whether all of `s` lies inside one live page-locked buffer made by PinMem.
pub fn is_page_locked(s: &[u8]) -> bool {
    let (a, z) = (s.as_ptr() as usize, s.as_ptr() as usize + s.len());
    !s.is_empty() && LIVE.lock().map(|g| g.iter().any(|&(lo, hi, _)| lo <= a && z <= hi)).unwrap_or(false)
}

/// host-syncs: a large buffer is page-locked in REG_CHUNK pieces, never in one cuMemHostRegister call.
/// THE BREAKAGE THIS FIXES (a test machine, pk120 merge-5, nsys bl-hostsync-nsys120, 2026-11-09): one cuMemHostRegister of a
/// multi-GB boundary buffer on the pin-ahead helper thread holds the driver lock (and the mm lock) for 0.58-0.89 s, and
/// the main thread's next cuLaunchKernel / cuMemcpyDtoDAsync / pthread_create waits the whole time with the card idle:
/// 0.60 + 0.72 s of TL step 2 and 0.65 + 0.39 s of TL step 3 were these waits (a test machine regtest: one 4 GiB register blocks
/// a concurrent cuMemsetD8Async for 0.273 s; 64 MiB pieces with a short pause between them, 8 ms at most). The same
/// bytes are page-locked, in pieces. The driver refuses an async copy whose host span crosses two pieces (regtest:
/// cuMemcpyDtoHAsync "invalid argument"), so every host copy entry point (Cuda::load installs install_split) issues
/// one driver call per piece, in order, on the same stream: the same bytes reach the same addresses.
/// AD_PIN_REG_CHUNK_MIB=0 keeps one registration per buffer (as before).
fn reg_chunk() -> usize {
    static C: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("AD_PIN_REG_CHUNK_MIB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(64) << 20)
}
/// The pieces (offset, bytes) of the host range [a, a + n): None when it lies in no piece-registered buffer or within one piece.
fn pieces(a: usize, n: usize) -> Option<Vec<(usize, usize)>> {
    if n == 0 || CHUNKED_LIVE.load(std::sync::atomic::Ordering::SeqCst) == 0 { return None; }
    let (lo, chunk) = LIVE.lock().ok()?.iter().find(|&&(lo, hi, c)| c != 0 && lo <= a && a < hi).map(|&(lo, _, c)| (lo, c))?;
    let z = a + n;
    if (a - lo) / chunk == (z - 1 - lo) / chunk { return None; }
    let mut out = Vec::new();
    let mut o = a;
    while o < z {
        let end = (lo + ((o - lo) / chunk + 1) * chunk).min(z);
        out.push((o - a, end - o));
        o = end;
    }
    Some(out)
}
type Up = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
type UpA = unsafe extern "C" fn(u64, *const c_void, usize, *mut c_void) -> i32;
type Down = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
type DownA = unsafe extern "C" fn(*mut c_void, u64, usize, *mut c_void) -> i32;
struct Real { up: Up, up_a: UpA, down: Down, down_a: DownA }
static REAL: std::sync::OnceLock<Real> = std::sync::OnceLock::new();
unsafe extern "C" fn s_up(d: u64, s: *const c_void, n: usize) -> i32 {
    let r = REAL.get().unwrap();
    match pieces(s as usize, n) { None => (r.up)(d, s, n), Some(v) => {
        for (o, k) in v { let e = (r.up)(d + o as u64, s.cast::<u8>().add(o).cast(), k); if e != 0 { return e; } } 0 } }
}
unsafe extern "C" fn s_up_a(d: u64, s: *const c_void, n: usize, st: *mut c_void) -> i32 {
    let r = REAL.get().unwrap();
    match pieces(s as usize, n) { None => (r.up_a)(d, s, n, st), Some(v) => {
        for (o, k) in v { let e = (r.up_a)(d + o as u64, s.cast::<u8>().add(o).cast(), k, st); if e != 0 { return e; } } 0 } }
}
unsafe extern "C" fn s_down(d: *mut c_void, s: u64, n: usize) -> i32 {
    let r = REAL.get().unwrap();
    match pieces(d as usize, n) { None => (r.down)(d, s, n), Some(v) => {
        for (o, k) in v { let e = (r.down)(d.cast::<u8>().add(o).cast(), s + o as u64, k); if e != 0 { return e; } } 0 } }
}
unsafe extern "C" fn s_down_a(d: *mut c_void, s: u64, n: usize, st: *mut c_void) -> i32 {
    let r = REAL.get().unwrap();
    match pieces(d as usize, n) { None => (r.down_a)(d, s, n, st), Some(v) => {
        for (o, k) in v { let e = (r.down_a)(d.cast::<u8>().add(o).cast(), s + o as u64, k, st); if e != 0 { return e; } } 0 } }
}
/// host-syncs: route this Cuda's host<->device copies through the piece split (see reg_chunk). Every loaded Cuda resolves
/// the same driver symbols, so one set of real entry points serves all of them.
pub(crate) unsafe fn install_split(c: &mut Cuda) {
    let _ = REAL.set(Real { up: c.upload, up_a: c.upload_async, down: c.download, down_a: c.download_async });
    c.upload = s_up; c.upload_async = s_up_a; c.download = s_down; c.download_async = s_down_a;
}
/// Smaller buffers keep cuMemHostAlloc (one call, no threads).
pub const PIN_FAST_MIN: usize = 64 << 20;
fn fast_on() -> bool { std::env::var("AD_PIN_ALLOC").as_deref() != Ok("driver") }
/// Whether PinMem asks for 2 MiB pages (see PinMem::registered): AD_PIN_HUGE=1/0, else only when the THP defrag
/// policy for madvised memory is "defer" or "never" (no synchronous compaction on the first touch).
fn huge_pages() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("AD_PIN_HUGE").as_deref() {
        Ok("1") => true, Ok("0") => false,
        _ => std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/defrag").ok()
                 .and_then(|t| { let a = t.find('[')?; let b = t[a..].find(']')?; Some(t[a + 1..a + b].to_owned()) })
                 .map_or(false, |v| v == "defer" || v == "never"),
    })
}
/// The page policy PinMem uses in this process (receipts).
pub fn page_policy() -> &'static str { if huge_pages() { "2 MiB pages (MADV_HUGEPAGE)" } else { "4 KiB pages (THP defrag would compact on the first touch)" } }

/// One page-locked host allocation: `cap` bytes at `ptr`, freed (unregistered and unmapped, or cuMemFreeHost) on drop.
pub struct PinMem { ptr: *mut u8, cap: usize, map: *mut c_void, map_len: usize, fns: PinFns, chunk: usize }
// Plain host memory owned by this value (see PinnedBuf).
unsafe impl Send for PinMem {}
unsafe impl Sync for PinMem {}
impl PinMem {
    /// `len` page-locked bytes (at least 1) in the current context.
    #[track_caller]
    pub unsafe fn new(fns: PinFns, len: usize) -> Result<Self, String> {
        let site = std::panic::Location::caller();
        let t = std::time::Instant::now();
        let r = Self::new_inner(fns, len);
        pin_site_add(site, len, t.elapsed().as_secs_f64());
        r
    }
    unsafe fn new_inner(fns: PinFns, len: usize) -> Result<Self, String> {
        let len = len.max(1);
        if len >= PIN_FAST_MIN && fast_on() { if let Some(m) = Self::registered(fns, len) { return Ok(m); } }
        let mut p: *mut c_void = std::ptr::null_mut();
        checked("cuMemHostAlloc page-locked buffer", (fns.host_alloc)(&mut p, len, 0))?;
        live_add(p.cast(), len, 0);
        Ok(Self { ptr: p.cast(), cap: len, map: std::ptr::null_mut(), map_len: 0, fns, chunk: 0 })
    }
    unsafe fn registered(fns: PinFns, len: usize) -> Option<Self> {
        let cap = len.div_ceil(HUGE) * HUGE;
        let map_len = cap + HUGE;
        // PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS
        let map = mmap(std::ptr::null_mut(), map_len, 3, 0x02 | 0x20, -1, 0);
        if map as isize == -1 { return None; }
        let p = ((map as usize).div_ceil(HUGE) * HUGE) as *mut u8;
        // MADV_HUGEPAGE only when the kernel will not stall the first touch in direct compaction: with THP defrag
        // "madvise", "defer+madvise" or "always", a madvised fault reclaims and compacts synchronously. Measured on a test machine
        // (1 TB, 737 GB of page cache, defrag madvise): one 2.8 GB buffer took 7.9 s that way inside a reverse step
        // (redo_download), with 32 threads at 100 % and the card idle. AD_PIN_HUGE=1 / 0 forces it on / off.
        if huge_pages() { let _ = madvise(p.cast(), cap, 14); } // MADV_HUGEPAGE
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).clamp(1, 32);
        let per = (cap / HUGE).div_ceil(threads).max(1) * HUGE;
        let base = p as usize;
        std::thread::scope(|sc| {
            for t in 0..threads {
                let (a, b) = ((t * per).min(cap), ((t + 1) * per).min(cap));
                if a < b { sc.spawn(move || { let mut o = a; while o < b { std::ptr::write_volatile((base + o) as *mut u8, 0); o += 4096; } }); }
            }
        });
        // CU_MEMHOSTREGISTER_PORTABLE: page-locked for every context, as cuMemHostAlloc memory is under unified addressing.
        let chunk = reg_chunk();
        if chunk == 0 || cap <= chunk {
            if (fns.register)(p.cast(), cap, 1) != 0 { let _ = munmap(map, map_len); return None; }
            live_add(p, cap, 0);
            return Some(Self { ptr: p, cap, map, map_len, fns, chunk: 0 });
        }
        // host-syncs: one piece at a time, with a short pause so other threads' driver calls get the lock between pieces.
        let mut o = 0usize;
        while o < cap {
            let k = chunk.min(cap - o);
            if (fns.register)(p.add(o).cast(), k, 1) != 0 {
                let mut u = 0usize;
                while u < o { let _ = (fns.unregister)(p.add(u).cast()); u += chunk; }
                let _ = munmap(map, map_len);
                return None;
            }
            o += k;
            if o < cap { std::thread::sleep(std::time::Duration::from_micros(200)); }
        }
        live_add(p, cap, chunk);
        Some(Self { ptr: p, cap, map, map_len, fns, chunk })
    }
    pub fn ptr(&self) -> *mut u8 { self.ptr }
    pub fn cap(&self) -> usize { self.cap }
    /// Whether the bytes came from the fast path (receipts).
    pub fn registered_kind(&self) -> bool { !self.map.is_null() }
}
impl Drop for PinMem {
    fn drop(&mut self) {
        live_drop(self.ptr);
        unsafe {
            if self.map.is_null() { if !self.ptr.is_null() { let _ = (self.fns.host_free)(self.ptr.cast()); } }
            else if self.chunk == 0 { let _ = (self.fns.unregister)(self.ptr.cast()); let _ = munmap(self.map, self.map_len); }
            else {
                let mut o = 0usize;
                while o < self.cap { let _ = (self.fns.unregister)(self.ptr.add(o).cast()); o += self.chunk; }
                let _ = munmap(self.map, self.map_len);
            }
        }
    }
}

/// a12 bs-tape (diagnostic, always on, a few host words a call): page-locked buffers made, by the source line that asked
/// (through PinnedBuf::new), as (calls, bytes, seconds); pin_sites_take() hands them over and clears them.
static PIN_SITES: std::sync::Mutex<std::collections::BTreeMap<String, (u64, u64, f64)>> = std::sync::Mutex::new(std::collections::BTreeMap::new());
fn pin_site_add(site: &std::panic::Location<'static>, len: usize, s: f64) {
    if let Ok(mut m) = PIN_SITES.lock() {
        let th = std::thread::current().name().unwrap_or("?").to_owned();
        let e = m.entry(format!("{}:{} [{th}]", site.file().rsplit('/').next().unwrap_or(""), site.line())).or_insert((0, 0, 0.0));
        e.0 += 1; e.1 += len as u64; e.2 += s;
    }
}
pub fn pin_sites_take() -> Vec<(String, u64, u64, f64)> {
    PIN_SITES.lock().map(|mut m| std::mem::take(&mut *m).into_iter().map(|(k, v)| (k, v.0, v.1, v.2)).collect()).unwrap_or_default()
}

pub struct PinnedBuf { mem: PinMem, len: usize }

impl PinnedBuf {
    #[track_caller]
    pub unsafe fn new(cuda: &Cuda, len: usize) -> Result<Self, String> {
        Ok(Self { mem: PinMem::new(PinFns::of(cuda), len)?, len })
    }
    pub fn len(&self) -> usize { self.len }
    /// anchors-ahead: wrap a buffer a helper thread made (`len` at most its capacity).
    pub fn from_mem(mem: PinMem, len: usize) -> Self { let len = len.min(mem.cap()); Self { mem, len } }
    /// bs-tape (a12 backstep): the page-locked memory back, e.g. to return a borrowed buffer to its pool.
    pub fn into_mem(self) -> PinMem { self.mem }
}
impl std::ops::Deref for PinnedBuf {
    type Target = [u8];
    fn deref(&self) -> &[u8] { unsafe { std::slice::from_raw_parts(self.mem.ptr, self.len) } }
}
impl std::ops::DerefMut for PinnedBuf {
    fn deref_mut(&mut self) -> &mut [u8] { unsafe { std::slice::from_raw_parts_mut(self.mem.ptr, self.len) } }
}

/// F1: one owner for a boundary's pinned D2H copies. The caller has completed all
/// producers, including non-default streams. Keep the mutable borrow until the
/// null-stream batch completes, also on a failed enqueue or unwinding path.
pub unsafe fn download_spans(cuda: &Cuda, dst: &mut [u8], spans: &[(u64, usize, usize, u64)]) -> Result<(), String> {
    for &(_, off, n, _) in spans {
        let end = off.checked_add(n).ok_or("boundary download range overflow")?;
        if end > dst.len() { return Err("boundary download outside pinned buffer".into()); }
    }
    struct Pending<'a> { cuda: &'a Cuda, dst: &'a mut [u8], armed: bool }
    impl Drop for Pending<'_> {
        fn drop(&mut self) {
            if self.armed { unsafe { let _ = (self.cuda.stream_synchronize)(std::ptr::null_mut()); } }
        }
    }
    let mut pending = Pending { cuda, dst, armed: false };
    let mut calls = 0usize;
    let mut bytes = 0usize;
    for &(_, off, n, src) in spans {
        if n == 0 { continue; }
        pending.armed = true;
        checked("cuMemcpyDtoHAsync boundary batch", (cuda.download_async)(
            pending.dst.as_mut_ptr().add(off).cast(), src, n, std::ptr::null_mut()))?;
        calls += 1;
        bytes += n;
    }
    if pending.armed {
        checked("cuStreamSynchronize boundary batch", (cuda.stream_synchronize)(std::ptr::null_mut()))?;
        pending.armed = false;
    }
    if std::env::var("AD_DIGEST_PROF").as_deref() == Ok("1") {
        eprintln!("{}", serde_json::json!({"event":"boundary_download_batch", "copies":calls,
            "bytes":bytes,"completion_waits":usize::from(calls != 0),"published_after_completion":true}));
    }
    Ok(())
}
