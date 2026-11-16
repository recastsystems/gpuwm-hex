// SPDX-License-Identifier: Apache-2.0
//! Device helpers for derivative dispatch: a bump arena for per-launch
//! buffers, synchronous copies, and the support kernels (fills, fixed-order
//! reductions, word counts).
use crate::cuda::{checked, Cuda};
use std::{ffi::c_void, ptr};

/// G5 (ad/phys-window-g5): host-side time per dispatch category, only with AD_G5_PROF=1 (one relaxed load otherwise).
pub mod prof {
    use std::sync::atomic::{AtomicU64, AtomicU8, Ordering::Relaxed};
    pub const NAMES: [&str; 14] = ["sync_wait", "widen_host", "prepare", "command", "rule", "entry_launch", "restore_snap",
                                   "check_failures", "native_rule", "xfer", "tape_copy", "walk_rule", "rule_classify", "gather_build"];
    static NS: [AtomicU64; 14] = [const { AtomicU64::new(0) }; 14];
    static N: [AtomicU64; 14] = [const { AtomicU64::new(0) }; 14];
    static ON: AtomicU8 = AtomicU8::new(2);
    pub fn on() -> bool {
        let v = ON.load(Relaxed);
        if v != 2 { return v == 1; }
        let o = std::env::var("AD_G5_PROF").as_deref() == Ok("1");
        ON.store(o as u8, Relaxed);
        o
    }
    pub fn start() -> Option<std::time::Instant> { on().then(std::time::Instant::now) }
    pub fn add(k: usize, t: Option<std::time::Instant>) {
        if let Some(t) = t { NS[k].fetch_add(t.elapsed().as_nanos() as u64, Relaxed); N[k].fetch_add(1, Relaxed); }
    }
    pub fn take() -> Vec<(&'static str, f64, u64)> {
        NAMES.iter().enumerate().map(|(k, n)| (*n, NS[k].swap(0, Relaxed) as f64 / 1e9, N[k].swap(0, Relaxed))).collect()
    }
}

pub const BLOCKS: u32 = 512;
pub const SLOTS: usize = 512;
const SUPPORT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/e2e_support.fatbin"));

pub struct Arena { pub base: u64, pub bytes: usize, pub used: usize, pub peak: usize }
impl Arena {
    pub unsafe fn new(cuda: &Cuda, bytes: usize) -> Result<Self, String> {
        let mut base = 0;
        checked("cuMemAlloc arena", (cuda.allocate)(&mut base, bytes))?;
        Ok(Self { base, bytes, used: 0, peak: 0 })
    }
    pub fn take(&mut self, bytes: usize) -> Result<u64, String> {
        let start = self.used.div_ceil(256) * 256;
        let end = start.checked_add(bytes.max(1)).ok_or("arena overflow")?;
        if end > self.bytes { return Err(format!("derivative arena exhausted: need {end} of {} bytes", self.bytes)); }
        self.used = end;
        self.peak = self.peak.max(end);
        Ok(self.base + start as u64)
    }
    pub fn reset(&mut self) { self.used = 0; }
}

pub struct Support {
    module: *mut c_void,
    fill32: *mut c_void, fill64: *mut c_void,
    dotdiff32: *mut c_void, dotdiff64: *mut c_void,
    sqdiff32: *mut c_void, sqdiff64: *mut c_void,
    words: *mut c_void,
    strided32: *mut c_void, strided64: *mut c_void, bcast32: *mut c_void, bcast64: *mut c_void,
    move32: *mut c_void, move64: *mut c_void, scale32: *mut c_void, scale64: *mut c_void, axpy32: *mut c_void, axpy64: *mut c_void,
    /// readbacks-device: the float32/float64 conversions of widened derivative arguments, on the device.
    widen: *mut c_void, narrow: *mut c_void, narrow_add: *mut c_void,
    partial: u64, counts: u64,
    /// tape-dedupe-and-pack (blocks): e2e_pairs_differ / e2e_runs_copy and their staging buffer (STAGE bytes: pair tables
    /// in the first 3/4, results in the last 1/4).
    pairs_differ: *mut c_void, runs_copy: *mut c_void, stage: u64,
}
/// tape-dedupe-and-pack (blocks): staging buffer bytes (pair tables + results).
pub const STAGE: usize = 8 << 20;
/// Runs longer than this many words are split, so one thread block never copies a long run alone.
pub const RUN_WORDS: usize = 1 << 15;
/// Split (dst offset, src, bytes) runs into table rows (dst offset, src, words) of at most RUN_WORDS words.
pub fn run_rows(runs: &[(u64, u64, usize)]) -> Vec<u64> {
    let mut t = Vec::with_capacity(runs.len() * 3);
    for &(o, s, b) in runs {
        let mut k = 0usize;
        while k < b / 4 { let n = (b / 4 - k).min(RUN_WORDS); t.extend_from_slice(&[o + 4 * k as u64, s + 4 * k as u64, n as u64]); k += n; }
    }
    t
}
impl Support {
    pub unsafe fn load(cuda: &Cuda) -> Result<Self, String> {
        let mut module = ptr::null_mut();
        checked("cuModuleLoadData support", (cuda.module_load)(&mut module, SUPPORT.as_ptr().cast()))?;
        let get = |name: &std::ffi::CStr| -> Result<*mut c_void, String> {
            let mut f = ptr::null_mut();
            checked("cuModuleGetFunction support", (cuda.module_function)(&mut f, module, name.as_ptr()))?;
            Ok(f)
        };
        let mut partial = 0; let mut counts = 0;
        checked("cuMemAlloc partial", (cuda.allocate)(&mut partial, SLOTS * BLOCKS as usize * 8))?;
        checked("cuMemAlloc counts", (cuda.allocate)(&mut counts, SLOTS * 24))?;
        Ok(Self { module, fill32: get(c"e2e_fill_f32")?, fill64: get(c"e2e_fill_f64")?,
                  dotdiff32: get(c"e2e_dotdiff_f32")?, dotdiff64: get(c"e2e_dotdiff_f64")?,
                  sqdiff32: get(c"e2e_sqdiff_f32")?, sqdiff64: get(c"e2e_sqdiff_f64")?,
                  words: get(c"e2e_words_u32")?, move32: get(c"e2e_move_add_f32")?, move64: get(c"e2e_move_add_f64")?,
                  scale32: get(c"e2e_scale_f32")?, scale64: get(c"e2e_scale_f64")?,
                  strided32: get(c"e2e_strided_f32")?, strided64: get(c"e2e_strided_f64")?,
                  bcast32: get(c"e2e_bcast_ad_f32")?, bcast64: get(c"e2e_bcast_ad_f64")?,
                  axpy32: get(c"e2e_axpy_clear_f32")?, axpy64: get(c"e2e_axpy_clear_f64")?,
                  widen: get(c"e2e_widen_f32_f64")?, narrow: get(c"e2e_narrow_f64_f32")?, narrow_add: get(c"e2e_narrow_add_f64_f32")?,
                  partial, counts, pairs_differ: get(c"e2e_pairs_differ")?, runs_copy: get(c"e2e_runs_copy")?,
                  stage: { let mut s = 0; checked("cuMemAlloc stage", (cuda.allocate)(&mut s, STAGE))?; s } })
    }
    unsafe fn launch(&self, cuda: &Cuda, f: *mut c_void, args: &mut [*mut c_void]) -> Result<(), String> {
        checked("support launch", (cuda.launch)(f, BLOCKS, 1, 1, 256, 1, 1, 0, ptr::null_mut(), args.as_mut_ptr(), ptr::null_mut()))
    }
    pub unsafe fn fill(&self, cuda: &Cuda, address: u64, bytes: usize, width: usize, seed: u64) -> Result<(), String> {
        let (mut a, mut n, mut s) = (address, (bytes / width) as u64, seed);
        let f = if width == 8 { self.fill64 } else { self.fill32 };
        self.launch(cuda, f, &mut [(&mut a as *mut u64).cast(), (&mut n as *mut u64).cast(), (&mut s as *mut u64).cast()])
    }
    /// Queue sum a*(b-c) (c = 0 means none) into a result slot. No host wait.
    pub unsafe fn dotdiff_into(&self, cuda: &Cuda, slot: usize, a: u64, b: u64, c: u64, bytes: usize, width: usize) -> Result<(), String> {
        if slot >= SLOTS { return Err("reduction slots exhausted".into()); }
        let (mut a, mut b, mut c, mut n, mut p) = (a, b, c, (bytes / width) as u64, self.partial + (slot * BLOCKS as usize * 8) as u64);
        let f = if width == 8 { self.dotdiff64 } else { self.dotdiff32 };
        self.launch(cuda, f, &mut [(&mut a as *mut u64).cast(), (&mut b as *mut u64).cast(), (&mut c as *mut u64).cast(),
                                   (&mut n as *mut u64).cast(), (&mut p as *mut u64).cast()])
    }
    /// Queue sum (b-c)^2 (c = 0 gives a squared norm) into a result slot.
    pub unsafe fn sqdiff_into(&self, cuda: &Cuda, slot: usize, b: u64, c: u64, bytes: usize, width: usize) -> Result<(), String> {
        if slot >= SLOTS { return Err("reduction slots exhausted".into()); }
        let (mut b, mut c, mut n, mut p) = (b, c, (bytes / width) as u64, self.partial + (slot * BLOCKS as usize * 8) as u64);
        let f = if width == 8 { self.sqdiff64 } else { self.sqdiff32 };
        self.launch(cuda, f, &mut [(&mut b as *mut u64).cast(), (&mut c as *mut u64).cast(),
                                   (&mut n as *mut u64).cast(), (&mut p as *mut u64).cast()])
    }
    /// Zero the word counters before queueing word counts.
    pub unsafe fn reset_words(&self, cuda: &Cuda) -> Result<(), String> {
        checked("cuMemsetD8 counts", (cuda.clear)(self.counts, 0, SLOTS * 24, ptr::null_mut()))
    }
    /// Queue [differing words a vs b (b may be 0), nonzero-magnitude words of a, negative zeros of a].
    pub unsafe fn words_into(&self, cuda: &Cuda, slot: usize, a: u64, b: u64, bytes: usize) -> Result<(), String> {
        if slot >= SLOTS { return Err("word slots exhausted".into()); }
        let (mut a, mut b, mut n, mut c) = (a, b, (bytes / 4) as u64, self.counts + (slot * 24) as u64);
        self.launch(cuda, self.words, &mut [(&mut a as *mut u64).cast(), (&mut b as *mut u64).cast(),
                                            (&mut n as *mut u64).cast(), (&mut c as *mut u64).cast()])
    }
    /// One readback for everything queued: per-slot sums (block partials added in
    /// index order) and per-slot word counts.
    pub unsafe fn collect(&self, cuda: &Cuda, sums: usize, words: usize) -> Result<(Vec<f64>, Vec<[u64; 3]>), String> {
        let mut host = vec![0f64; sums * BLOCKS as usize];
        if sums > 0 { checked("cuMemcpyDtoH partial", (cuda.download)(host.as_mut_ptr().cast(), self.partial, host.len() * 8))?; }
        let s = host.chunks(BLOCKS as usize).map(|c| c.iter().sum()).collect();
        let mut raw = vec![0u64; words * 3];
        if words > 0 { checked("cuMemcpyDtoH counts", (cuda.download)(raw.as_mut_ptr().cast(), self.counts, raw.len() * 8))?; }
        Ok((s, raw.chunks(3).map(|c| [c[0], c[1], c[2]]).collect()))
    }
    unsafe fn launch_rows(&self, cuda: &Cuda, f: *mut c_void, n: usize, args: &mut [*mut c_void]) -> Result<(), String> {
        let g = n.clamp(1, 65535) as u32;
        checked("support launch", (cuda.launch)(f, g, 1, 1, 256, 1, 1, 0, ptr::null_mut(), args.as_mut_ptr(), ptr::null_mut()))
    }
    /// tape-dedupe-and-pack (blocks): for each (a, b, bytes) pair, whether the 32-bit words differ. One upload, one launch
    /// and one readback per STAGE-sized chunk of pairs.
    pub unsafe fn pairs_differ(&self, cuda: &Cuda, pairs: &[(u64, u64, usize)]) -> Result<Vec<bool>, String> {
        let per = (STAGE * 3 / 4) / 24;
        let mut out = Vec::with_capacity(pairs.len());
        for c in pairs.chunks(per) {
            let t: Vec<u64> = c.iter().flat_map(|&(a, b, n)| [a, b, (n / 4) as u64]).collect();
            upload(cuda, self.stage, std::slice::from_raw_parts(t.as_ptr().cast::<u8>(), t.len() * 8))?;
            let (mut tab, mut n, mut o) = (self.stage, c.len() as u64, self.stage + (STAGE * 3 / 4) as u64);
            self.launch_rows(cuda, self.pairs_differ, c.len(), &mut [(&mut tab as *mut u64).cast(), (&mut n as *mut u64).cast(), (&mut o as *mut u64).cast()])?;
            let r = download(cuda, self.stage + (STAGE * 3 / 4) as u64, c.len() * 4)?;
            out.extend(r.chunks_exact(4).map(|x| u32::from_le_bytes(x.try_into().unwrap()) != 0));
        }
        Ok(out)
    }
    /// tape-dedupe-and-pack (blocks): queue one copy of every row of a device run table (run_rows layout) to dst + offset.
    /// No host wait.
    pub unsafe fn runs_copy(&self, cuda: &Cuda, dst: u64, table: u64, rows: usize) -> Result<(), String> {
        if rows == 0 { return Ok(()); }
        let (mut d, mut t, mut n) = (dst, table, rows as u64);
        self.launch_rows(cuda, self.runs_copy, rows, &mut [(&mut d as *mut u64).cast(), (&mut t as *mut u64).cast(), (&mut n as *mut u64).cast()])
    }
    /// readbacks-device: d[i] = s[i] as f64 (exact), `n` words, on the device (was a host round trip per argument).
    pub unsafe fn widen_f32(&self, cuda: &Cuda, s: u64, d: u64, n: usize) -> Result<(), String> {
        if n == 0 { return Ok(()); }
        let (mut s, mut d, mut n) = (s, d, n as u64);
        self.launch(cuda, self.widen, &mut [(&mut s as *mut u64).cast(), (&mut d as *mut u64).cast(), (&mut n as *mut u64).cast()])
    }
    /// readbacks-device: d[i] = s[i] as f32 (round to nearest even, as Rust `as f32`).
    pub unsafe fn narrow_f64(&self, cuda: &Cuda, s: u64, d: u64, n: usize) -> Result<(), String> {
        if n == 0 { return Ok(()); }
        let (mut s, mut d, mut n) = (s, d, n as u64);
        self.launch(cuda, self.narrow, &mut [(&mut s as *mut u64).cast(), (&mut d as *mut u64).cast(), (&mut n as *mut u64).cast()])
    }
    /// readbacks-device: cur = replace ? acc as f32 : (cur as f64 + acc) as f32.
    pub unsafe fn narrow_add_f64(&self, cuda: &Cuda, cur: u64, acc: u64, n: usize, replace: bool) -> Result<(), String> {
        if n == 0 { return Ok(()); }
        let (mut c, mut a, mut n, mut r) = (cur, acc, n as u64, replace as i32);
        self.launch(cuda, self.narrow_add, &mut [(&mut c as *mut u64).cast(), (&mut a as *mut u64).cast(), (&mut n as *mut u64).cast(), (&mut r as *mut i32).cast()])
    }
    pub unsafe fn unload(&mut self, cuda: &Cuda) {
        let _ = (cuda.free)(self.partial); let _ = (cuda.free)(self.counts); let _ = (cuda.free)(self.stage);
        let _ = (cuda.module_unload)(self.module);
    }
}

pub unsafe fn copy(cuda: &Cuda, dst: u64, src: u64, bytes: usize) -> Result<(), String> {
    if bytes == 0 { return Ok(()); }
    checked("cuMemcpyDtoDAsync", (cuda.copy)(dst, src, bytes, ptr::null_mut()))
}
pub unsafe fn zero(cuda: &Cuda, dst: u64, bytes: usize) -> Result<(), String> {
    if bytes == 0 { return Ok(()); }
    checked("cuMemsetD8Async", (cuda.clear)(dst, 0, bytes, ptr::null_mut()))
}
pub unsafe fn sync(cuda: &Cuda) -> Result<(), String> {
    let t = prof::start();
    let r = checked("cuCtxSynchronize", (cuda.context_synchronize)());
    prof::add(0, t);
    r
}
pub unsafe fn download(cuda: &Cuda, src: u64, bytes: usize) -> Result<Vec<u8>, String> {
    let mut v = vec![0u8; bytes];
    if bytes != 0 { checked("cuMemcpyDtoH", (cuda.download)(v.as_mut_ptr().cast(), src, bytes))?; }
    Ok(v)
}
/// x20 replay-compiled: a small host->device upload (<= 4 MiB) is a stream-ordered cuMemcpyHtoDAsync on the legacy
/// stream from a per-thread page-locked ring, not a pageable cuMemcpyHtoD. THE BREAKAGE THIS FIXES: a pageable
/// cuMemcpyHtoD first waits for every queued command, so each per-launch table, key and segment upload in the TL and
/// AD step drained the card (cuda-graph-capture: 744 + 84 such drains after e2e_first_i32 alone per 4 pkg18m2 AD steps)
/// and the host could never run ahead of the queue. The copy lands at the same position in the same in-order stream
/// with the same bytes, so every word is unchanged. The ring is reused only after a context synchronize. The ring is
/// plain host memory registered with cuMemHostRegister (PORTABLE), so a context torn down under it (the C ABI reopens
/// HexAD per outer loop) leaves valid host memory and the copy degrades to CUDA's own synchronous pageable path.
/// AD_UPLOAD_SYNC=1 keeps the old blocking copy.
struct UpRing { base: *mut u8, cap: usize, head: usize, registered: bool }
impl Drop for UpRing { fn drop(&mut self) { unsafe { std::alloc::dealloc(self.base, std::alloc::Layout::from_size_align_unchecked(self.cap, 4096)); } } }
thread_local! { static UP_RING: std::cell::RefCell<Option<UpRing>> = const { std::cell::RefCell::new(None) }; }
const UP_RING_BYTES: usize = 64 << 20;
const UP_RING_MAX: usize = 4 << 20;
pub fn upload_sync_forced() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AD_UPLOAD_SYNC").as_deref() == Ok("1"))
}
/// Uploads issued through the ring and ring wraps (each wrap is one context synchronize), for receipts.
pub static UP_STAGED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static UP_WRAPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub unsafe fn upload(cuda: &Cuda, dst: u64, data: &[u8]) -> Result<(), String> {
    if data.is_empty() { return Ok(()); }
    if data.len() <= UP_RING_MAX && !upload_sync_forced() {
        let staged = UP_RING.with(|cell| -> Result<bool, String> {
            let mut g = cell.borrow_mut();
            if g.is_none() {
                let layout = std::alloc::Layout::from_size_align(UP_RING_BYTES, 4096).map_err(|e| e.to_string())?;
                let base = std::alloc::alloc_zeroed(layout);
                if base.is_null() { return Ok(false); }
                let registered = (cuda.host_register)(base.cast(), UP_RING_BYTES, 1) == 0;
                *g = Some(UpRing { base, cap: UP_RING_BYTES, head: 0, registered });
            }
            let ring = g.as_mut().unwrap();
            if !ring.registered { return Ok(false); }
            let mut at = ring.head.div_ceil(256) * 256;
            if at + data.len() > ring.cap {
                sync(cuda)?;
                UP_WRAPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                at = 0;
            }
            ptr::copy_nonoverlapping(data.as_ptr(), ring.base.add(at), data.len());
            checked("cuMemcpyHtoDAsync staged", (cuda.upload_async)(dst, ring.base.add(at).cast(), data.len(), ptr::null_mut()))?;
            ring.head = at + data.len();
            UP_STAGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(true)
        })?;
        if staged { return Ok(()); }
    }
    checked("cuMemcpyHtoD", (cuda.upload)(dst, data.as_ptr().cast(), data.len()))
}

impl Support {
    /// Copy adjoint: src += dst; dst = 0 (width-byte floats, `bytes` long).
    pub unsafe fn move_add(&self, cuda: &Cuda, src: u64, dst: u64, bytes: usize, width: usize) -> Result<(), String> {
        let (mut s, mut d, mut n) = (src, dst, (bytes / width) as u64);
        self.launch(cuda, if width == 8 { self.move64 } else { self.move32 },
                    &mut [(&mut s as *mut u64).cast(), (&mut d as *mut u64).cast(), (&mut n as *mut u64).cast()])
    }
    pub unsafe fn scale(&self, cuda: &Cuda, x: u64, c: f64, bytes: usize, width: usize) -> Result<(), String> {
        let (mut x, mut c, mut n) = (x, c, (bytes / width) as u64);
        self.launch(cuda, if width == 8 { self.scale64 } else { self.scale32 },
                    &mut [(&mut x as *mut u64).cast(), (&mut c as *mut f64).cast(), (&mut n as *mut u64).cast()])
    }
    /// in += c * out; out = 0.
    pub unsafe fn axpy_clear(&self, cuda: &Cuda, input: u64, output: u64, c: f64, bytes: usize, width: usize) -> Result<(), String> {
        let (mut i, mut o, mut c, mut n) = (input, output, c, (bytes / width) as u64);
        self.launch(cuda, if width == 8 { self.axpy64 } else { self.axpy32 },
                    &mut [(&mut i as *mut u64).cast(), (&mut o as *mut u64).cast(), (&mut c as *mut f64).cast(), (&mut n as *mut u64).cast()])
    }
}

/// Layout of a strided view for e2e_strided_*: shape, source and destination
/// element strides (row-major index order), rank.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct View { pub shape: [i64; 4], pub ss: [i64; 4], pub ds: [i64; 4], pub ndim: i32 }
impl Support {
    pub unsafe fn strided(&self, cuda: &Cuda, src: u64, dst: u64, n: usize, v: View, width: usize, adjoint: bool) -> Result<(), String> {
        let broadcast = (0..v.ndim as usize).any(|k| v.shape[k] > 1 && v.ss[k] == 0);
        if adjoint && broadcast {
            // U9: fixed-order gather per source element (see e2e_bcast_ad).
            let (mut s, mut d, mut v) = (src, dst, v);
            return self.launch(cuda, if width == 8 { self.bcast64 } else { self.bcast32 },
                               &mut [(&mut s as *mut u64).cast(), (&mut d as *mut u64).cast(), (&mut v as *mut View).cast()]);
        }
        let (mut s, mut d, mut n, mut v, mut a) = (src, dst, n as u64, v, adjoint as i32);
        self.launch(cuda, if width == 8 { self.strided64 } else { self.strided32 },
                    &mut [(&mut s as *mut u64).cast(), (&mut d as *mut u64).cast(), (&mut n as *mut u64).cast(),
                          (&mut v as *mut View).cast(), (&mut a as *mut i32).cast()])
    }
}

/// readbacks-device (round 3): failure words, flags, counters and fixed-order partial sums that
/// derivative entries and native rows used to read back one at a time (436 check_failures readbacks per pkg18m2 step,
/// each a context synchronize and a pageable download, plus the P3 and TEMPO failure and counter reads) are queued
/// into one device ledger and read with ONE download when the step ends (`flush`). A fired item refuses the step with
/// the message the per-entry check gave, at the step end instead of at the entry. If the ledger fills inside a step it
/// is flushed early (same refusals). Values that only feed receipts (TEMPO norms and identity counts, P3 domain flags)
/// are handed back by token after the flush.
pub enum LedgerItem {
    /// 32-bit failure slot words: a nonzero word i fires `sources[i]` (KernelDeriv failure slots).
    Words { label: String, entry: String, sources: Vec<String> },
    /// Four u64 counters; a nonzero code whose `fatal` bit is set refuses the step (all four counts named).
    Codes4 { label: String, fatal: [bool; 4], names: [&'static str; 4] },
    /// One u32 flag: nonzero refuses with `message`.
    Flag { message: String },
    /// One u64 first-nonzero key (e2e_first_i32): nonzero refuses with `message`, the bits and the column.
    First { message: String },
    /// Values for receipts only: `BLOCKS` float64 partials (added in index order), or one u64 counter.
    Sum, Count,
}
pub struct Ledger {
    pub base: u64, pub bytes: usize, used: usize,
    items: Vec<(usize, usize, LedgerItem, u64)>,
    next_token: u64,
    /// Receipt values read at the flushes, by token: Sum -> [sum], Count -> [count], Codes4 -> [c0, c1, c2, c3].
    pub values: std::collections::HashMap<u64, Vec<f64>>,
    /// Non-fatal nonzero Codes4 readouts (label, counts) since the last take_report.
    pub report: Vec<(String, [u64; 4])>,
    /// Flushes that found something queued (one download each), items queued, early (ledger-full) flushes.
    pub reads: u64, pub queued: u64, pub early: u64,
    module: *mut c_void,
    pub f_widen: *mut c_void, pub f_add_round_sq: *mut c_void, pub f_tile: *mut c_void,
    pub f_codes4: *mut c_void, pub f_any_lane: *mut c_void, pub f_first: *mut c_void, pub f_diff: *mut c_void, pub f_sq64: *mut c_void,
    pub f_marks: *mut c_void,
}
impl Ledger {
    pub unsafe fn new(cuda: &Cuda, bytes: usize) -> Result<Self, String> {
        let mut module = ptr::null_mut();
        checked("cuModuleLoadData ledger", (cuda.module_load)(&mut module, SUPPORT.as_ptr().cast()))?;
        let get = |name: &std::ffi::CStr| -> Result<*mut c_void, String> {
            let mut f = ptr::null_mut();
            checked("cuModuleGetFunction ledger", (cuda.module_function)(&mut f, module, name.as_ptr()))?;
            Ok(f)
        };
        let mut base = 0;
        checked("cuMemAlloc ledger", (cuda.allocate)(&mut base, bytes))?;
        zero(cuda, base, bytes)?;
        Ok(Self { base, bytes, used: 0, items: Vec::new(), next_token: 1, values: Default::default(), report: Vec::new(),
                  reads: 0, queued: 0, early: 0, module,
                  f_widen: get(c"e2e_widen_f32_f64")?, f_add_round_sq: get(c"e2e_add_round_sq_f64_f32")?, f_tile: get(c"e2e_tile_f64")?,
                  f_codes4: get(c"e2e_codes4_i32")?, f_any_lane: get(c"e2e_any_lane_i32")?, f_first: get(c"e2e_first_i32")?,
                  f_diff: get(c"e2e_diff_words")?, f_sq64: get(c"e2e_sqdiff_f64")?, f_marks: get(c"e2e_tempo_marks")? })
    }
    /// Zeroed device words for one item (16-byte aligned); flushes first when the ledger is full.
    pub unsafe fn reserve(&mut self, cuda: &Cuda, bytes: usize, item: LedgerItem) -> Result<(u64, u64), String> {
        let bytes = bytes.max(4);
        if bytes > self.bytes { return Err(format!("ledger item of {bytes} bytes exceeds the {}-byte ledger", self.bytes)); }
        let mut start = self.used.div_ceil(16) * 16;
        if start + bytes > self.bytes { self.early += 1; self.flush(cuda)?; start = 0; }
        self.used = start + bytes;
        let token = self.next_token; self.next_token += 1;
        self.items.push((start, bytes, item, token));
        self.queued += 1;
        Ok((self.base + start as u64, token))
    }
    /// Queue a copy of failure slot words (stream-ordered after the entry that writes them; no host wait).
    pub unsafe fn queue_words(&mut self, cuda: &Cuda, label: &str, entry: &str, src: u64, bytes: usize, sources: Vec<String>) -> Result<(), String> {
        let (a, _) = self.reserve(cuda, bytes, LedgerItem::Words { label: label.to_owned(), entry: entry.to_owned(), sources })?;
        copy(cuda, a, src, bytes)
    }
    pub unsafe fn launch(&self, cuda: &Cuda, f: *mut c_void, args: &mut [*mut c_void]) -> Result<(), String> {
        checked("ledger launch", (cuda.launch)(f, BLOCKS, 1, 1, 256, 1, 1, 0, ptr::null_mut(), args.as_mut_ptr(), ptr::null_mut()))
    }
    /// float32 -> float64 widen on the device (exact).
    pub unsafe fn widen(&self, cuda: &Cuda, s: u64, d: u64, n: usize) -> Result<(), String> {
        if n == 0 { return Ok(()); }
        let (mut s, mut d, mut n) = (s, d, n as u64);
        self.launch(cuda, self.f_widen, &mut [(&mut s as *mut u64).cast(), (&mut d as *mut u64).cast(), (&mut n as *mut u64).cast()])
    }
    /// Queue a fixed-order sum of squares of `n` float64 words; returns its token.
    pub unsafe fn sq_sum_f64(&mut self, cuda: &Cuda, a: u64, n: usize) -> Result<u64, String> {
        let (p, tok) = self.reserve(cuda, BLOCKS as usize * 8, LedgerItem::Sum)?;
        let (mut a, mut z, mut n, mut p) = (a, 0u64, n as u64, p);
        self.launch(cuda, self.f_sq64, &mut [(&mut a as *mut u64).cast(), (&mut z as *mut u64).cast(), (&mut n as *mut u64).cast(), (&mut p as *mut u64).cast()])?;
        Ok(tok)
    }
    /// a[i * stride + lane] != 0 for any i < n refuses the step with `message`.
    pub unsafe fn any_lane(&mut self, cuda: &Cuda, a: u64, n: usize, stride: usize, lane: usize, message: &str) -> Result<(), String> {
        let (flag, _) = self.reserve(cuda, 4, LedgerItem::Flag { message: message.to_owned() })?;
        let (mut a, mut n, mut st, mut ln, mut f) = (a, n as u64, stride as u64, lane as u64, flag);
        self.launch(cuda, self.f_any_lane, &mut [(&mut a as *mut u64).cast(), (&mut n as *mut u64).cast(), (&mut st as *mut u64).cast(),
                                                  (&mut ln as *mut u64).cast(), (&mut f as *mut u64).cast()])
    }
    /// The first nonzero int32 a[i] (i < n) refuses the step: "`message` <bits> at column <base + i>".
    pub unsafe fn first_nonzero(&mut self, cuda: &Cuda, a: u64, n: usize, base: usize, message: &str) -> Result<(), String> {
        let (key, _) = self.reserve(cuda, 8, LedgerItem::First { message: message.to_owned() })?;
        let (mut a, mut n, mut b, mut k) = (a, n as u64, base as u64, key);
        self.launch(cuda, self.f_first, &mut [(&mut a as *mut u64).cast(), (&mut n as *mut u64).cast(), (&mut b as *mut u64).cast(), (&mut k as *mut u64).cast()])
    }
    /// count += words whose float32 bits differ: a (float32 when w == 4, float64 rounded to float32 when w == 8) vs b.
    pub unsafe fn diff_words(&self, cuda: &Cuda, a: u64, b: u64, n: usize, w: usize, count: u64) -> Result<(), String> {
        if n == 0 { return Ok(()); }
        let (mut a, mut b, mut n, mut w, mut c) = (a, b, n as u64, w as i32, count);
        self.launch(cuda, self.f_diff, &mut [(&mut a as *mut u64).cast(), (&mut b as *mut u64).cast(), (&mut n as *mut u64).cast(),
                                              (&mut w as *mut i32).cast(), (&mut c as *mut u64).cast()])
    }
    /// TEMPO count-pass marks reduced on the device (e2e_tempo_marks): cap[steps][nc] and pr[2] as u64, read with one
    /// download each; returns (cap, pmax, rmax) exactly as the host scan of the whole mark array computed them.
    pub unsafe fn tempo_marks(&self, cuda: &Cuda, mk: u64, steps: usize, nc: usize, cap: u64, pr: u64) -> Result<(Vec<u64>, u64, u64), String> {
        zero(cuda, pr, 16)?;
        let (mut m, mut s, mut n, mut c, mut p) = (mk, steps as u64, nc as u64, cap, pr);
        self.launch(cuda, self.f_marks, &mut [(&mut m as *mut u64).cast(), (&mut s as *mut u64).cast(), (&mut n as *mut u64).cast(),
                                               (&mut c as *mut u64).cast(), (&mut p as *mut u64).cast()])?;
        let b = download(cuda, cap, 8 * steps * nc)?;
        let cv: Vec<u64> = b.chunks_exact(8).map(|w| u64::from_le_bytes(w.try_into().unwrap())).collect();
        let q = download(cuda, pr, 16)?;
        Ok((cv, u64::from_le_bytes(q[0..8].try_into().unwrap()), u64::from_le_bytes(q[8..16].try_into().unwrap())))
    }
    /// Rows of a [rows][nc] float64 vector to a [rows][n] column tile from column c0 (back: the tile into the vector).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tile(&self, cuda: &Cuda, src: u64, dst: u64, rows: usize, nc: usize, c0: usize, n: usize, back: bool) -> Result<(), String> {
        if rows * n == 0 { return Ok(()); }
        let (mut s, mut d, mut r, mut c, mut o, mut k, mut dir) = (src, dst, rows as u64, nc as u64, c0 as u64, n as u64, back as i32);
        self.launch(cuda, self.f_tile, &mut [(&mut s as *mut u64).cast(), (&mut d as *mut u64).cast(), (&mut r as *mut u64).cast(),
                                              (&mut c as *mut u64).cast(), (&mut o as *mut u64).cast(), (&mut k as *mut u64).cast(), (&mut dir as *mut i32).cast()])
    }
    /// sh = ((accumulate ? sh as f64 : 0.0) + r) as f32 over n words; a non-finite result refuses the step with
    /// `message`; returns the token of the fixed-order sum of r^2 (receipt only).
    pub unsafe fn add_round_sq(&mut self, cuda: &Cuda, sh: u64, r: u64, n: usize, accumulate: bool, message: &str) -> Result<u64, String> {
        // Both items in one span: an early flush between the two reservations would read the partials unwritten.
        if self.used.div_ceil(16) * 16 + BLOCKS as usize * 8 + 32 > self.bytes { self.early += 1; self.flush(cuda)?; }
        let (part, tok) = self.reserve(cuda, BLOCKS as usize * 8, LedgerItem::Sum)?;
        let (bad, _) = self.reserve(cuda, 4, LedgerItem::Flag { message: message.to_owned() })?;
        let (mut s, mut r, mut n, mut acc, mut p, mut b) = (sh, r, n as u64, accumulate as i32, part, bad);
        self.launch(cuda, self.f_add_round_sq, &mut [(&mut s as *mut u64).cast(), (&mut r as *mut u64).cast(), (&mut n as *mut u64).cast(),
                                                      (&mut acc as *mut i32).cast(), (&mut p as *mut u64).cast(), (&mut b as *mut u64).cast()])?;
        Ok(tok)
    }
    /// ONE synchronize and ONE download of everything queued, then the refusals (all of them, in queue order).
    pub unsafe fn flush(&mut self, cuda: &Cuda) -> Result<(), String> {
        if self.items.is_empty() { return Ok(()); }
        sync(cuda)?;
        let host = download(cuda, self.base, self.used)?;
        zero(cuda, self.base, self.used)?;
        self.reads += 1;
        let items = std::mem::take(&mut self.items);
        self.used = 0;
        let mut fired: Vec<String> = Vec::new();
        let mut words_fired: Vec<(String, Vec<String>)> = Vec::new();
        let u64at = |o: usize| u64::from_le_bytes(host[o..o + 8].try_into().unwrap());
        for (off, bytes, item, token) in items {
            match item {
                LedgerItem::Words { label, entry, sources } => {
                    let mut f = Vec::new();
                    for (i, w) in host[off..off + bytes].chunks_exact(4).enumerate() {
                        if w != [0, 0, 0, 0] { f.push(format!("{entry}: {}", sources.get(i).cloned().unwrap_or_else(|| format!("slot word {i}")))); }
                    }
                    if !f.is_empty() {
                        match words_fired.iter_mut().find(|x| x.0 == label) { Some(x) => x.1.extend(f), None => words_fired.push((label, f)) }
                    }
                },
                LedgerItem::Codes4 { label, fatal, names } => {
                    let c = [u64at(off), u64at(off + 8), u64at(off + 16), u64at(off + 24)];
                    self.values.insert(token, c.iter().map(|&x| x as f64).collect());
                    if (0..4).any(|s| fatal[s] && c[s] != 0) {
                        fired.push(format!("{label} failure slots: {}", (0..4).map(|s| format!("{} {}", names[s], c[s])).collect::<Vec<_>>().join(" ")));
                    } else if c.iter().any(|&x| x != 0) { self.report.push((label, c)); }
                },
                LedgerItem::Flag { message } => {
                    if host[off..off + 4] != [0, 0, 0, 0] { fired.push(message); }
                },
                LedgerItem::First { message } => {
                    let k = u64at(off);
                    if k != 0 { fired.push(format!("{message} {} at column {}", (k & 0xffff_ffff) as u32 as i32, 0xffff_ffffu64 - (k >> 32))); }
                },
                LedgerItem::Sum => {
                    let s: f64 = host[off..off + bytes].chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).sum();
                    self.values.insert(token, vec![s]);
                },
                LedgerItem::Count => { self.values.insert(token, vec![u64at(off) as f64]); },
            }
        }
        for (label, f) in words_fired.into_iter().rev() { fired.insert(0, format!("{label} refused: {f:?}")); }
        if fired.is_empty() { Ok(()) } else { Err(fired.join("; ")) }
    }
    pub fn value(&mut self, token: u64) -> Option<Vec<f64>> { self.values.remove(&token) }
    pub fn take_report(&mut self) -> Vec<(String, [u64; 4])> { std::mem::take(&mut self.report) }
    pub unsafe fn unload(&mut self, cuda: &Cuda) { let _ = (cuda.free)(self.base); let _ = (cuda.module_unload)(self.module); }
}
