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
//! loop 1 on a test box, card idle; 36.6 s for the same 19 x 3.44 GiB on a test box): `PinMem` makes a large page-locked
//! buffer as anonymous memory on 2 MiB transparent huge pages, first-touched on every core, then page-locked with
//! cuMemHostRegister. cuMemHostAlloc faults, zeroes and locks 4 KiB pages on one thread inside the driver; here the
//! zeroing runs on all cores and the driver locks 512x fewer pages. Measured (pinbench, a test box, 19 x 3.44 GiB):
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
static LIVE: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());
fn live_add(p: *mut u8, cap: usize) { if let Ok(mut g) = LIVE.lock() { g.push((p as usize, p as usize + cap)); } }
fn live_drop(p: *mut u8) { if let Ok(mut g) = LIVE.lock() { if let Some(i) = g.iter().position(|r| r.0 == p as usize) { g.swap_remove(i); } } }
/// Whether all of `s` lies inside one live page-locked buffer made by PinMem.
pub fn is_page_locked(s: &[u8]) -> bool {
    let (a, z) = (s.as_ptr() as usize, s.as_ptr() as usize + s.len());
    !s.is_empty() && LIVE.lock().map(|g| g.iter().any(|&(lo, hi)| lo <= a && z <= hi)).unwrap_or(false)
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
pub struct PinMem { ptr: *mut u8, cap: usize, map: *mut c_void, map_len: usize, fns: PinFns }
// Plain host memory owned by this value (see PinnedBuf).
unsafe impl Send for PinMem {}
unsafe impl Sync for PinMem {}
impl PinMem {
    /// `len` page-locked bytes (at least 1) in the current context.
    pub unsafe fn new(fns: PinFns, len: usize) -> Result<Self, String> {
        let len = len.max(1);
        if len >= PIN_FAST_MIN && fast_on() { if let Some(m) = Self::registered(fns, len) { return Ok(m); } }
        let mut p: *mut c_void = std::ptr::null_mut();
        checked("cuMemHostAlloc page-locked buffer", (fns.host_alloc)(&mut p, len, 0))?;
        live_add(p.cast(), len);
        Ok(Self { ptr: p.cast(), cap: len, map: std::ptr::null_mut(), map_len: 0, fns })
    }
    unsafe fn registered(fns: PinFns, len: usize) -> Option<Self> {
        let cap = len.div_ceil(HUGE) * HUGE;
        let map_len = cap + HUGE;
        // PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS
        let map = mmap(std::ptr::null_mut(), map_len, 3, 0x02 | 0x20, -1, 0);
        if map as isize == -1 { return None; }
        let p = ((map as usize).div_ceil(HUGE) * HUGE) as *mut u8;
        // MADV_HUGEPAGE only when the kernel will not stall the first touch in direct compaction: with THP defrag
        // "madvise", "defer+madvise" or "always", a madvised fault reclaims and compacts synchronously. Measured on a test box
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
        if (fns.register)(p.cast(), cap, 1) != 0 { let _ = munmap(map, map_len); return None; }
        live_add(p, cap);
        Some(Self { ptr: p, cap, map, map_len, fns })
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
            else { let _ = (self.fns.unregister)(self.ptr.cast()); let _ = munmap(self.map, self.map_len); }
        }
    }
}

pub struct PinnedBuf { mem: PinMem, len: usize }

impl PinnedBuf {
    pub unsafe fn new(cuda: &Cuda, len: usize) -> Result<Self, String> {
        Ok(Self { mem: PinMem::new(PinFns::of(cuda), len)?, len })
    }
    pub fn len(&self) -> usize { self.len }
    /// anchors-ahead: wrap a buffer a helper thread made (`len` at most its capacity).
    pub fn from_mem(mem: PinMem, len: usize) -> Self { let len = len.min(mem.cap()); Self { mem, len } }
}
impl std::ops::Deref for PinnedBuf {
    type Target = [u8];
    fn deref(&self) -> &[u8] { unsafe { std::slice::from_raw_parts(self.mem.ptr, self.len) } }
}
impl std::ops::DerefMut for PinnedBuf {
    fn deref_mut(&mut self) -> &mut [u8] { unsafe { std::slice::from_raw_parts_mut(self.mem.ptr, self.len) } }
}
