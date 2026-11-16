// SPDX-License-Identifier: Apache-2.0
//! Device helpers for derivative dispatch: a bump arena for per-launch
//! buffers, synchronous copies, and the support kernels (fills, fixed-order
//! reductions, word counts).
use crate::cuda::{checked, Cuda};
use std::{ffi::c_void, ptr};

// Reverse capture routes only audited kernel/copy/clear regions to its private
// stream. All ordinary callers retain the legacy stream. The guard restores it
// on errors and unwinding; uploads and host reads are forbidden in these regions.
thread_local! { static DISPATCH_STREAM: std::cell::Cell<*mut c_void> = const { std::cell::Cell::new(ptr::null_mut()) }; }
pub fn dispatch_stream() -> *mut c_void { DISPATCH_STREAM.with(|s| s.get()) }
pub struct DispatchStream(*mut c_void);
impl DispatchStream {
    pub fn enter(stream: *mut c_void) -> Result<Self, String> {
        if stream.is_null() { return Err("DEFECT derivative capture requires a private stream".into()); }
        if !dispatch_stream().is_null() { return Err("DEFECT nested derivative stream scope".into()); }
        Ok(Self(DISPATCH_STREAM.with(|s| s.replace(stream))))
    }
}
impl Drop for DispatchStream {
    fn drop(&mut self) { DISPATCH_STREAM.with(|s| s.set(self.0)); }
}

/// G5 (ad/phys-window-g5): host-side time per dispatch category, only with AD_G5_PROF=1 (one relaxed load otherwise).
pub mod prof {
    use std::sync::atomic::{AtomicU64, AtomicU8, Ordering::Relaxed};
    // bs-tape (a12 backstep): four more host timers for the taped re-run (AD_G5_PROF=1 only): word-check readbacks
    // (collect), block compares (pairs_differ), the forward commands of a taped launch, and its rule lookup.
    pub const NAMES: [&str; 27] = ["sync_wait", "widen_host", "prepare", "command", "rule", "entry_launch", "restore_snap",
                                   "check_failures", "native_rule", "xfer", "tape_copy", "walk_rule", "rule_classify", "gather_build",
                                   "check_collect", "block_compare", "tape_exec", "tape_rule",
                                   "prep_spans", "prep_batch", "prep_loop", "prep_meta", "tape_post",
                                   "cmd_kernel", "cmd_upload", "cmd_copy", "cmd_other"];
    static NS: [AtomicU64; 27] = [const { AtomicU64::new(0) }; 27];
    static N: [AtomicU64; 27] = [const { AtomicU64::new(0) }; 27];
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

pub struct Arena { pub base: u64, pub bytes: usize, pub used: usize, pub peak: usize, pub spill: Option<Spill>, pub fence: Option<Fence>,
                   /// memory: how many times the arena was reset (a launch that resets it owns what it then takes).
                   pub resets: u64 }
/// revmach: write fences on an arena whose earlier bytes are still being copied out by an asynchronous device->host copy
/// (the fused tape's chunk flush). `pieces` are (start offset, event recorded after that piece's copy), in offset order;
/// a take whose range reaches a piece first makes the dispatch stream wait for that event. So a later write is ordered
/// after the copy of the bytes it overwrites, by stream order (no host wait, no timing assumption).
pub struct Fence { pub pieces: std::collections::VecDeque<(usize, *mut c_void)>, pub wait: unsafe extern "C" fn(*mut c_void, *mut c_void, u32) -> i32, pub waited: u64 }
impl Fence {
    /// make the dispatch stream wait for every piece starting before `end`.
    pub unsafe fn upto(&mut self, end: usize) -> Result<(), String> {
        while let Some(&(s, ev)) = self.pieces.front() {
            if s >= end { break; }
            checked("cuStreamWaitEvent arena fence", (self.wait)(dispatch_stream(), ev, 0))?;
            self.pieces.pop_front();
            self.waited += 1;
        }
        Ok(())
    }
}
/// x20 scratch-arena: the overflow chain of an arena that may grow mid-launch (the window's reverse scratch only).
/// A take past the arena continues in kept slabs (allocated once, never freed while the plan replays, as the loader's
/// grow-only scheme scratch: a freed block would be reused by the plan's own later allocations), walked in order after
/// every reset. THE BREAKAGE IT PREVENTS: the reverse scratch is sized from the tangent pass's measured workspace plus
/// one launch's transient; a reverse launch whose adjoint entries need more workspace than the tangent's (pkg18m2:
/// 557,656,576 of 545,259,520 bytes) refused the whole window with "derivative arena exhausted". The words never
/// depend on where a buffer lives.
pub struct Spill { pub alloc: unsafe extern "C" fn(*mut u64, usize) -> i32, pub slabs: Vec<(u64, usize)>, pub cur: usize, pub off: usize,
                   pub held: usize, pub peak: usize }
impl Arena {
    pub unsafe fn new(cuda: &Cuda, bytes: usize) -> Result<Self, String> {
        let mut base = 0;
        checked("cuMemAlloc arena", (cuda.allocate)(&mut base, bytes))?;
        Ok(Self { base, bytes, used: 0, peak: 0, spill: None, fence: None, resets: 0 })
    }
    pub fn take(&mut self, bytes: usize) -> Result<u64, String> {
        let start = self.used.div_ceil(256) * 256;
        let end = start.checked_add(bytes.max(1)).ok_or("arena overflow")?;
        if let Some(f) = self.fence.as_mut() { if !f.pieces.is_empty() { unsafe { f.upto(end.min(self.bytes))?; } } }
        if end > self.bytes {
            let Some(s) = self.spill.as_mut() else { return Err(format!("derivative arena exhausted: need {end} of {} bytes", self.bytes)) };
            let need = bytes.max(1);
            loop {
                if s.cur < s.slabs.len() {
                    let (b, n) = s.slabs[s.cur];
                    let st = s.off.div_ceil(256) * 256;
                    if st + need <= n {
                        s.off = st + need;
                        let used: usize = s.slabs[..s.cur].iter().map(|x| x.1).sum::<usize>() + s.off;
                        s.peak = s.peak.max(used);
                        self.peak = self.peak.max(self.bytes + used);
                        return Ok(b + st as u64);
                    }
                    s.cur += 1; s.off = 0;
                    continue;
                }
                let n = (need + need / 16).max(64 << 20).div_ceil(2 << 20) * (2 << 20);
                let mut a = 0u64;
                // SAFETY: the driver entry point the arena was made with; a refusal is returned, never retried.
                checked(&format!("cuMemAlloc reverse scratch spill {n} bytes"), unsafe { (s.alloc)(&mut a, n) })?;
                s.slabs.push((a, n));
                s.held += n;
            }
        }
        self.used = end;
        self.peak = self.peak.max(end);
        Ok(self.base + start as u64)
    }
    pub fn reset(&mut self) { self.used = 0; self.resets += 1; if let Some(s) = self.spill.as_mut() { s.cur = 0; s.off = 0; } }
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
    /// speed tape60 (host-shared tape blocks): e2e_spans_digest.
    spans_digest: *mut c_void,
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
                  partial, counts, pairs_differ: get(c"e2e_pairs_differ")?, runs_copy: get(c"e2e_runs_copy")?, spans_digest: get(c"e2e_spans_digest")?,
                  stage: { let mut s = 0; checked("cuMemAlloc stage", (cuda.allocate)(&mut s, STAGE))?; s } })
    }
    unsafe fn launch(&self, cuda: &Cuda, f: *mut c_void, args: &mut [*mut c_void]) -> Result<(), String> {
        checked("support launch", (cuda.launch)(f, BLOCKS, 1, 1, 256, 1, 1, 0, dispatch_stream(), args.as_mut_ptr(), ptr::null_mut()))
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
    /// revmach: the device address of the word-count slots (for an asynchronous readback of what words_into queued).
    pub fn counts_addr(&self) -> u64 { self.counts }
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
        let tp = prof::start();
        let r = self.collect_inner(cuda, sums, words);
        prof::add(14, tp);
        r
    }
    unsafe fn collect_inner(&self, cuda: &Cuda, sums: usize, words: usize) -> Result<(Vec<f64>, Vec<[u64; 3]>), String> {
        let mut host = vec![0f64; sums * BLOCKS as usize];
        if sums > 0 { checked("cuMemcpyDtoH partial", (cuda.download)(host.as_mut_ptr().cast(), self.partial, host.len() * 8))?; }
        let s = host.chunks(BLOCKS as usize).map(|c| c.iter().sum()).collect();
        let mut raw = vec![0u64; words * 3];
        if words > 0 { checked("cuMemcpyDtoH counts", (cuda.download)(raw.as_mut_ptr().cast(), self.counts, raw.len() * 8))?; }
        Ok((s, raw.chunks(3).map(|c| [c[0], c[1], c[2]]).collect()))
    }
    unsafe fn launch_rows(&self, cuda: &Cuda, f: *mut c_void, n: usize, args: &mut [*mut c_void]) -> Result<(), String> {
        let g = n.clamp(1, 65535) as u32;
        checked("support launch", (cuda.launch)(f, g, 1, 1, 256, 1, 1, 0, dispatch_stream(), args.as_mut_ptr(), ptr::null_mut()))
    }
    /// tape-dedupe-and-pack (blocks): for each (a, b, bytes) pair, whether the 32-bit words differ. One upload, one launch
    /// and one readback per STAGE-sized chunk of pairs.
    pub unsafe fn pairs_differ(&self, cuda: &Cuda, pairs: &[(u64, u64, usize)]) -> Result<Vec<bool>, String> {
        let tp = prof::start();
        let r = self.pairs_differ_inner(cuda, pairs);
        prof::add(15, tp);
        r
    }
    /// bs-tape (a12 backstep): the most pairs one pairs_differ_queue takes (one chunk of pairs_differ).
    pub fn pairs_per(&self) -> usize { (STAGE * 3 / 4) / 24 }
    /// bs-tape: pairs_differ's upload and launch for at most pairs_per() pairs, with no readback (pairs_differ_read reads
    /// the answer later; nothing else may use the staging buffer in between).
    pub unsafe fn pairs_differ_queue(&self, cuda: &Cuda, pairs: &[(u64, u64, usize)]) -> Result<(), String> {
        if pairs.len() > self.pairs_per() { return Err(format!("DEFECT pairs_differ_queue: {} pairs, at most {}", pairs.len(), self.pairs_per())); }
        if pairs.is_empty() { return Ok(()); }
        let t: Vec<u64> = pairs.iter().flat_map(|&(a, b, n)| [a, b, (n / 4) as u64]).collect();
        upload(cuda, self.stage, std::slice::from_raw_parts(t.as_ptr().cast::<u8>(), t.len() * 8))?;
        let (mut tab, mut n, mut o) = (self.stage, pairs.len() as u64, self.stage + (STAGE * 3 / 4) as u64);
        self.launch_rows(cuda, self.pairs_differ, pairs.len(), &mut [(&mut tab as *mut u64).cast(), (&mut n as *mut u64).cast(), (&mut o as *mut u64).cast()])
    }
    /// rerun-tape: the device address of pairs_differ_queue's answer (u32 a pair, nonzero = differ), for a copy
    /// of it in stream order instead of a readback.
    pub fn pairs_result_addr(&self) -> u64 { self.stage + (STAGE * 3 / 4) as u64 }
    /// bs-tape: the answer of the last pairs_differ_queue of `n` pairs (whether each pair's words differ).
    pub unsafe fn pairs_differ_read(&self, cuda: &Cuda, n: usize) -> Result<Vec<bool>, String> {
        let tp = prof::start();
        let r = download(cuda, self.stage + (STAGE * 3 / 4) as u64, n * 4)?;
        prof::add(15, tp);
        Ok(r.chunks_exact(4).map(|x| u32::from_le_bytes(x.try_into().unwrap()) != 0).collect())
    }
    unsafe fn pairs_differ_inner(&self, cuda: &Cuda, pairs: &[(u64, u64, usize)]) -> Result<Vec<bool>, String> {
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
        // bs-launch: about 2,048 blocks a launch (each run split across gridDim.y), not one block a run.
        let g = rows.clamp(1, 65535) as u32;
        let y = (2048 / g).clamp(1, 64);
        let mut args: [*mut c_void; 3] = [(&mut d as *mut u64).cast(), (&mut t as *mut u64).cast(), (&mut n as *mut u64).cast()];
        checked("support launch runs_copy", (cuda.launch)(self.runs_copy, g, y, 1, 256, 1, 1, 0, dispatch_stream(), args.as_mut_ptr(), ptr::null_mut()))
    }
    /// speed tape60 (host-shared tape blocks): two 64-bit digests of the 32-bit words of each (address, bytes) span.
    /// One upload, one launch and one readback per chunk of spans (tables in the first half of the staging buffer,
    /// results in the second).
    pub unsafe fn spans_digest(&self, cuda: &Cuda, spans: &[(u64, usize)]) -> Result<Vec<[u64; 2]>, String> {
        let per = (STAGE / 2) / 16;
        let mut out = Vec::with_capacity(spans.len());
        for c in spans.chunks(per) {
            let t: Vec<u64> = c.iter().flat_map(|&(a, n)| [a, (n / 4) as u64]).collect();
            upload(cuda, self.stage, std::slice::from_raw_parts(t.as_ptr().cast::<u8>(), t.len() * 8))?;
            let (mut tab, mut n, mut o) = (self.stage, c.len() as u64, self.stage + (STAGE / 2) as u64);
            self.launch_rows(cuda, self.spans_digest, c.len(), &mut [(&mut tab as *mut u64).cast(), (&mut n as *mut u64).cast(), (&mut o as *mut u64).cast()])?;
            let r = download(cuda, self.stage + (STAGE / 2) as u64, c.len() * 16)?;
            out.extend(r.chunks_exact(16).map(|x| [u64::from_le_bytes(x[..8].try_into().unwrap()), u64::from_le_bytes(x[8..].try_into().unwrap())]));
        }
        Ok(out)
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
    checked("cuMemcpyDtoDAsync", (cuda.copy)(dst, src, bytes, dispatch_stream()))
}
pub unsafe fn zero(cuda: &Cuda, dst: u64, bytes: usize) -> Result<(), String> {
    if bytes == 0 { return Ok(()); }
    checked("cuMemsetD8Async", (cuda.clear)(dst, 0, bytes, dispatch_stream()))
}
/// lane tl-host-syncs (AD_SYNC_SITES=1, diagnostic, default off): every host drain the runner issues (cuCtxSynchronize
/// through dev::sync, blocking cuMemcpyDtoH through dev::download, the replayed stream/context/event waits of the plan) is
/// counted by its call site with its wait time. It adds no device call and no synchronisation: it times calls that
/// already block. The per-step tallies go into window_tangent_step as "sync_sites".
pub mod sites {
    use std::sync::atomic::{AtomicU8, Ordering::Relaxed};
    static ON: AtomicU8 = AtomicU8::new(2);
    static MAP: std::sync::Mutex<std::collections::BTreeMap<String, (u64, u64)>> = std::sync::Mutex::new(std::collections::BTreeMap::new());
    pub fn on() -> bool {
        let v = ON.load(Relaxed);
        if v != 2 { return v == 1; }
        let o = std::env::var("AD_SYNC_SITES").as_deref() == Ok("1");
        ON.store(o as u8, Relaxed);
        o
    }
    pub fn start() -> Option<std::time::Instant> { on().then(std::time::Instant::now) }
    pub fn add(site: &str, t: Option<std::time::Instant>) {
        if let Some(t) = t { if let Ok(mut m) = MAP.lock() { let e = m.entry(site.to_owned()).or_insert((0, 0)); e.0 += 1; e.1 += t.elapsed().as_nanos() as u64; } }
    }
    #[track_caller]
    pub fn add_here(kind: &str, t: Option<std::time::Instant>) {
        if t.is_some() { let l = std::panic::Location::caller(); add(&format!("{kind} {}:{}", l.file().rsplit('/').next().unwrap_or(""), l.line()), t); }
    }
    /// [site, calls, wait s], most calls first.
    pub fn take() -> Vec<(String, u64, f64)> {
        let Ok(mut m) = MAP.lock() else { return Vec::new() };
        let mut v: Vec<(String, u64, f64)> = std::mem::take(&mut *m).into_iter().map(|(k, (n, ns))| (k, n, ns as f64 / 1e9)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v
    }
}
#[track_caller]
pub unsafe fn sync(cuda: &Cuda) -> Result<(), String> {
    if !dispatch_stream().is_null() { return Err("DEFECT host drain inside derivative graph capture".into()); }
    let t = prof::start();
    let st = sites::start();
    let r = checked("cuCtxSynchronize", (cuda.context_synchronize)());
    sites::add_here("ctx", st);
    prof::add(0, t);
    r
}
#[track_caller]
pub unsafe fn download(cuda: &Cuda, src: u64, bytes: usize) -> Result<Vec<u8>, String> {
    if !dispatch_stream().is_null() { return Err("DEFECT host read inside derivative graph capture".into()); }
    let mut v = vec![0u8; bytes];
    let st = sites::start();
    if bytes != 0 { checked("cuMemcpyDtoH", (cuda.download)(v.as_mut_ptr().cast(), src, bytes))?; }
    sites::add_here("d2h", st);
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
#[track_caller]
pub unsafe fn upload(cuda: &Cuda, dst: u64, data: &[u8]) -> Result<(), String> {
    if !dispatch_stream().is_null() { return Err("DEFECT host upload inside derivative graph capture".into()); }
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
    let st = sites::start();
    let r = checked("cuMemcpyHtoD", (cuda.upload)(dst, data.as_ptr().cast(), data.len()));
    sites::add_here("h2d-pageable", st);
    r
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

/// readbacks-device : failure words, flags, counters and fixed-order partial sums that
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
    /// ranks: set in a rank process run with AD_PEER_TOLERATE_GUARDS=1. Fired failure-slot words then do not refuse
    /// the step; each fired label is kept here (label, fired words, distinct sources) for the window receipt.
    pub tolerate_words: bool, pub tolerated: Vec<(String, usize, Vec<String>)>,
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
                  reads: 0, queued: 0, early: 0, tolerate_words: false, tolerated: Vec::new(), module,
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
        checked("ledger launch", (cuda.launch)(f, BLOCKS, 1, 1, 256, 1, 1, 0, dispatch_stream(), args.as_mut_ptr(), ptr::null_mut()))
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
        if self.tolerate_words {
            for (label, f) in words_fired { let n = f.len(); let mut d = f; d.sort(); d.dedup(); d.truncate(6); self.tolerated.push((label, n, d)); }
        } else {
            for (label, f) in words_fired.into_iter().rev() { fired.insert(0, format!("{label} refused: {f:?}")); }
        }
        if fired.is_empty() { Ok(()) } else { Err(fired.join("; ")) }
    }
    pub fn value(&mut self, token: u64) -> Option<Vec<f64>> { self.values.remove(&token) }
    pub fn take_report(&mut self) -> Vec<(String, [u64; 4])> { std::mem::take(&mut self.report) }
    pub unsafe fn unload(&mut self, cuda: &Cuda) { let _ = (cuda.free)(self.base); let _ = (cuda.module_unload)(self.module); }
}
