// SPDX-License-Identifier: Apache-2.0
//! Live before/after adapter. Never dispatches the original forward command.
//! Item 2 owns the pinned Win/Execution and borrowed context; item 4 consumes Tp.
use super::*;
use crate::plan;
use std::ffi::{c_char, c_int};

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    operation: Operation,
    // Logical owner+generation IDs, INCLUDING indirect writes. No recycled IDs.
    writes: Vec<u64>,
}

/// Not a legacy capi::Session. Its owners must remain pinned at stable addresses;
/// calls are serialized on the owning host thread with its CUDA context current.
pub(crate) struct LiveSave {
    w: *mut Win,
    ex: *mut Execution,
    pending: Option<(u64, FzPre, Vec<u64>)>,
    records: Vec<Tp>,
    next: u64,
    failed: bool,
    usage: Vec<Value>,
    capacity: usize,
    walked: bool,
    expected: Option<(usize, Value)>,
    // Same content-checked span policy as tape_range; fixed for this capture.
    share: bool,
}
impl LiveSave {
    /// Opt-in only. Content equality, not a declared writer list, certifies reuse.
    pub(crate) fn sharing_enabled() -> bool {
        std::env::var("AD_LIVE_SHARE").as_deref() == Ok("1")
    }

    pub(crate) unsafe fn new(w: *mut Win, ex: *mut Execution) -> Result<Self, String> {
        if w.is_null() || ex.is_null() { return Err("null live window/execution".into()); }
        if (*w).tape.spill.is_some() { return Err("live tape must be bounded, without spill".into()); }
        if Self::sharing_enabled() { return Self::attach(w, ex); }
        // A recording's content cache is not a live-version certificate.
        (*w).precache = deriv::PreCache::default();
        Ok(Self { w, ex, pending: None, records: Vec::new(), next: 1, failed: false, usage: Vec::new(), capacity: (*w).tape.bytes, walked: false, expected: None, share: Self::sharing_enabled() })
    }

    /// No-recording (AD_LIVE_SELF_KEEP=1): as `new`, but the window's pre-launch cache and window cache are kept. Every use
    /// of a cached copy is still checked word for word against the live bytes (PreCache::lookup, prepare_tape), and
    /// `after` evicts every written owner unless AD_LIVE_SHARE=1 selects content-checked sharing.
    /// What it is for: `new` empties both caches at every tape chunk, and the all-live run then moved 51 GB of tape
    /// a step each way where the recorded route moves 18 and 23 (no-recording run 2, k12-fb5-all against k12-m5plain-c1).
    pub(crate) unsafe fn attach(w: *mut Win, ex: *mut Execution) -> Result<Self, String> {
        if w.is_null() || ex.is_null() { return Err("null live window/execution".into()); }
        if (*w).tape.spill.is_some() { return Err("live tape must be bounded, without spill".into()); }
        Ok(Self { w, ex, pending: None, records: Vec::new(), next: 1, failed: false, usage: Vec::new(), capacity: (*w).tape.bytes, walked: false, expected: None, share: Self::sharing_enabled() })
    }

    /// No-recording self-driven test (window.rs, AD_LIVE_SELF): the same Win and Execution, re-derived from the caller's
    /// current borrows before each call, so no stale pointer is used across the caller's own use of them.
    pub(crate) fn expect(&mut self, step: usize, request: Value) { self.expected = Some((step, request)); }

    pub(crate) fn rebind(&mut self, w: *mut Win, ex: *mut Execution) { self.w = w; self.ex = ex; }

    pub(crate) fn take_records(&mut self) -> Result<Vec<Tp>, String> {
        if self.failed || self.walked || self.pending.is_some() { return Err("unfinished, consumed or poisoned live tape".into()); }
        Ok(std::mem::take(&mut self.records))
    }

    /// A new bounded segment reuses the same arena and persistent cotangents.
    unsafe fn begin_tape(&mut self, capacity: usize) -> Result<(), String> {
        if self.pending.is_some() || !self.records.is_empty() || self.failed {
            return Err("release the previous live tape before beginning another".into());
        }
        if capacity == 0 || capacity > self.capacity { return Err("live tape capacity exceeds its allocation".into()); }
        let (w, ex) = (&mut *self.w, &*self.ex);
        dev::sync(&ex.cuda)?;
        w.tape.reset();
        w.tape.bytes = capacity;
        self.clear_segment();
        self.walked = false;
        self.usage.clear();
        Ok(())
    }

    unsafe fn walk_back(&mut self, step: usize) -> Result<(), String> {
        let records = self.take_records()?;
        self.walked = true; // A failed reverse must never be retried on partial bars.
        let (w, ex) = (&mut *self.w, &*self.ex);
        // Promote while forward bytes still describe the end of this segment.
        if self.share && w.pack { w.promote_window(ex)?; }
        w.live_walk_back(&ex.cuda, ex, step, records)
    }

    pub(super) unsafe fn release_tape(&mut self) -> Result<(), String> {
        let (w, ex) = (&mut *self.w, &*self.ex);
        dev::sync(&ex.cuda)?; // Including failed capture/reverse consumers.
        self.records.clear();
        self.pending = None;
        w.tape.reset();
        w.tape.bytes = self.capacity;
        self.clear_segment();
        // Remain poisoned after a failed command/reverse; create a fresh capture.
        self.walked = true;
        Ok(())
    }

    /// Tape addresses expire at a segment reset; window copies do not.
    unsafe fn clear_segment(&mut self) {
        let w = &mut *self.w;
        if self.share {
            w.precache.clear();
            if let Some(n) = w.native.as_mut() { n.roll_clear(); }
        } else {
            w.precache = deriv::PreCache::default();
        }
    }

    /// Allocation accounting, explicitly distinct from requested/copy traffic.
    /// Other item adapters may attach this object to their tape.json receipt.
    pub(crate) unsafe fn receipt(&self) -> Value {
        json!({"schema":"hexad.live-save-usage/v1", "commands":self.usage.len(),
            "poisoned":self.failed, "pending":self.pending.is_some(), "operations":self.usage,
            "peak_tape_bytes":(*self.w).tape.peak, "tape_capacity_bytes":(*self.w).tape.bytes,
            "requested_bytes":null, "unique_pre_copy_bytes":null, "unique_post_copy_bytes":null,
            "constants_bytes":null, "workspace_bytes":null, "complete_copy_census":false,
            "note":"arena allocation counters include alignment; native/Tempo parts are aggregated"})
    }

    unsafe fn admit(&mut self, data: &[u8], step: usize) -> Result<(R, Option<LiveLaunch>, Vec<u64>), String> {
        if self.failed || self.walked || self.pending.is_some() { return Err("unfinished, consumed or poisoned live tape".into()); }
        if let Some((expected_step, expected)) = &self.expected {
            let got: Value = serde_json::from_slice(data).map_err(|e| e.to_string())?;
            if &got != expected || step != *expected_step || !self.records.is_empty() { return Err("join live descriptor mismatch or duplicate command".into()); }
        }
        let req: Request = serde_json::from_slice(data).map_err(|e| e.to_string())?;
        let (w, ex) = (&mut *self.w, &*self.ex);
        for id in &req.writes {
            if !ex.allocations.contains_key(id) { return Err(format!("unregistered live write owner {id}")); }
        }
        // Admit only live descriptors; no Blob or recorded frame can be read.
        match &req.operation {
            Operation::Kernel { arguments, .. } => for a in arguments {
                match a {
                    plan::Argument::Device { target } => { ex.address(*target, 0)?; },
                    plan::Argument::Bytes { .. } => (),
                    plan::Argument::Patched { pointers, .. } => for p in pointers { ex.address(p.target, 0)?; },
                    _ => return Err("live kernel requires resolved device/bytes/patched arguments".into()),
                }
            },
            Operation::DeviceCopy { destination, source, bytes, .. } => {
                ex.address(*destination, *bytes)?; ex.address(*source, *bytes)?;
                if !req.writes.contains(&destination.allocation) { return Err("copy destination missing from writes".into()); }
            },
            Operation::Clear { destination, bytes, .. } => {
                ex.address(*destination, *bytes)?;
                if !req.writes.contains(&destination.allocation) { return Err("fill destination missing from writes".into()); }
            },
            Operation::Fill16 { destination, count, .. } | Operation::Fill32 { destination, count, .. } => {
                let width = if matches!(&req.operation, Operation::Fill16 { .. }) { 2 } else { 4 };
                ex.address(*destination, count.checked_mul(width).ok_or("fill size overflow")?)?;
                if !req.writes.contains(&destination.allocation) { return Err("fill destination missing from writes".into()); }
            },
            _ => return Err("live producer needs a typed kernel/copy/fill descriptor; payload operations forbidden".into()),
        }
        if w.fz_pre.is_some() || w.fz_words.is_some() { return Err("legacy fused capture is active".into()); }
        // Conservative cross-stream fencing. Existing native helpers have legacy-
        // stream work and host reads; DispatchStream is a graph-only scope.
        // Item 2 sets the caller's current context; no device-0 assumption here.
        dev::sync(&ex.cuda)?;
        let (r, l) = w.rule_inner(ex, &req.operation, step, Path::new(""), None)?;
        if matches!(r, R::Refused(_)) { return Err("unresolved live derivative rule".into()); }
        if let (R::Entry, Some((f, args, _, _, _))) = (&r, &l) {
            let kd = &w.kernels[f];
            for a in kd.spec["args"].as_array().into_iter().flatten() {
                if !matches!(a["role"].as_str(), Some("out") | Some("inout")) { continue; }
                let Some(k) = kd.names.iter().position(|n| Some(n.as_str()) == a["name"].as_str()) else { continue };
                if let Some(submit::Argument::Device(s)) = args.get(k) {
                    if !req.writes.contains(&s.allocation) { return Err(format!("{f}: output owner missing from live writes")); }
                }
            }
        }
        Ok((r, l, req.writes))
    }

    pub(crate) unsafe fn before(&mut self, data: &[u8], step: usize) -> Result<u64, String> {
        let (r, l, writes) = self.admit(data, step)?;
        let (w, ex) = (&mut *self.w, &*self.ex);
        let start = w.tape.used;
        let prepared = (|| -> Result<FzPre, String> {
            // Match tape_range's native path, including its reverse-order undo.
            // rec_pre is also used by fused/TL recording, where rolling is unsafe.
            let pre = if self.share && w.pack {
                if let (R::Native(_), Some((f, args, grid, block, _))) = (&r, &l) {
                    let n = w.native.as_mut().ok_or("native rows not loaded")?;
                    let p = match n.prepare_deferred_roll(&ex.cuda, &mut w.tape, &ex.allocations,
                                                         f, args, *grid, *block, &w.support) {
                        Ok(p) => p,
                        Err(e) if exhausted(&e) => return Err("live pre-save budget exhausted".into()),
                        Err(e) => return Err(e),
                    };
                    FzPre::Done(Tp::Native { f: f.clone(), p })
                } else {
                    w.rec_pre(ex, &r, &l, step)?.ok_or("live pre-save budget exhausted")?
                }
            } else {
                w.rec_pre(ex, &r, &l, step)?.ok_or("live pre-save budget exhausted")?
            };
            if let FzPre::Entry { f, b, args, grid, block, shared } = &pre {
                let kd = &w.kernels[f];
                let li = deriv::LaunchInfo { args, grid: *grid, block: *block, shared: *shared };
                // tape_range's conservative allowance, before Python executes.
                let mut need = 4096usize;
                for &(_, lo, hi, _) in &b.shared {
                    let n = hi.saturating_sub(lo);
                    let extra = if deriv::tape_block_bytes() > 0 { (n / 4).div_ceil(dev::RUN_WORDS).max(1) * 24 } else { n };
                    need = need.checked_add(extra + 256).ok_or("live post budget overflow")?;
                }
                if w.pack {
                    for (_, (lo, hi)) in kd.post_spans(&li, &b.env, &ex.allocations)? {
                        need = need.checked_add(hi.saturating_sub(lo) + 256).ok_or("live post budget overflow")?;
                    }
                } else {
                    need = need.checked_add(Win::post_bytes(kd, args, &ex.allocations) + 256 * args.len()).ok_or("live post budget overflow")?;
                }
                if need > w.tape.bytes.saturating_sub(w.tape.used) { return Err("live post-save budget exhausted before launch".into()); }
            }
            Ok(pre)
        })();
        let pre = match prepared {
            Ok(p) => p,
            Err(e) => {
                dev::sync(&ex.cuda)?;
                w.tape.used = start;
                // Keep completed copies for st_flush/promotion; discard only the
                // failed command's allocations. Native rolling preparation makes
                // every tape take before mutating its copy, so exhaustion is atomic.
                if self.share {
                    let end = w.tape.base + start as u64;
                    w.precache.spans.retain(|_, v| {
                        v.retain(|&(lo, hi, at)| at < end && (hi - lo) as u64 <= end - at);
                        !v.is_empty()
                    });
                    w.precache.fresh.retain(|&(_, lo, hi, at)| at < end && (hi - lo) as u64 <= end - at);
                } else {
                    w.precache = deriv::PreCache::default();
                }
                return Err(e);
            },
        };
        dev::sync(&ex.cuda)?; // old inputs fully saved before any caller stream writes
        let ticket = self.next;
        self.next = self.next.checked_add(1).ok_or("live ticket overflow")?;
        let parts = match &pre { FzPre::Entry { b, .. } => Some(b.parts), _ => None };
        self.usage.push(json!({"ticket":ticket, "pre_arena_bytes":w.tape.used - start,
            "entry_parts":parts, "post_arena_bytes":null}));
        self.pending = Some((ticket, pre, writes));
        Ok(ticket)
    }

    /// Separate TL pass: shared pre/forward/post body, callback exactly once.
    /// Never call this while a reverse tape references the resettable TL arena.
    unsafe fn tangent(&mut self, data: &[u8], step: usize,
        forward: unsafe extern "C" fn(*mut c_void) -> c_int, opaque: *mut c_void) -> Result<(), String> {
        if !self.records.is_empty() { return Err("live tangent cannot share a retained reverse tape".into()); }
        let (r, l, _) = self.admit(data, step)?;
        let (w, ex) = (&mut *self.w, &mut *self.ex);
        w.live_tangent(ex, r, l, step, |ex| {
            dev::sync(&ex.cuda)?; // Pre saves complete before any caller stream writes.
            if forward(opaque) != 0 { return Err("live forward callback failed".into()); }
            dev::sync(&ex.cuda) // Caller-stream result ready for legacy derivative dispatch.
        })?;
        dev::sync(&ex.cuda)
    }

    pub(crate) unsafe fn after(&mut self, ticket: u64, success: bool) -> Result<(), String> {
        if self.failed { return Err("poisoned live tape".into()); }
        if self.pending.as_ref().map(|p| p.0) != Some(ticket) { return Err("wrong live before/after ticket".into()); }
        if !success { self.failed = true; self.pending = None; return Err("forward command failed; tape poisoned".into()); }
        let (_, pre, writes) = self.pending.take().unwrap();
        let (w, ex) = (&mut *self.w, &*self.ex);
        dev::sync(&ex.cuda)?; // includes Python's nondefault stream
        // Legacy live mode uses writer eviction. Sharing mode retains immutable
        // candidates just like tape_range: every lookup compares current bytes.
        // Keeping old candidates also exposes them to st_flush host-block sharing.
        if !self.share {
            // A possible write begins a NEW version, even if the bytes compare equal.
            // Keep the pre record itself; evict candidates before post-spans lookup.
            for id in &writes {
                w.precache.spans.remove(id);
                w.precache.win.remove(id);
            }
            w.precache.fresh.retain(|(id, _, _, _)| !writes.contains(id));
        }
        let start = w.tape.used;
        let record = w.rec_post(ex, pre)?.ok_or("live post-save exceeded preflight")?;
        dev::sync(&ex.cuda)?; // post copies finish before scrub, release, or reuse
        if let Some(row) = self.usage.last_mut() { row["post_arena_bytes"] = json!(w.tape.used - start); }
        self.records.push(record);
        Ok(())
    }
}

unsafe fn ffi_call(ctx: *mut LiveSave, error: *mut c_char, capacity: usize,
                   run: impl FnOnce(&mut LiveSave) -> Result<(), String>) -> c_int {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let s = ctx.as_mut().ok_or("null live save context")?;
        run(s)
    })).unwrap_or_else(|_| Err("panic in live save callback".into()));
    match result {
        Ok(()) => 0,
        Err(message) => {
            if let Some(s) = ctx.as_mut() { s.failed = true; }
            if !error.is_null() && capacity > 0 {
                let n = message.len().min(capacity - 1);
                std::ptr::copy_nonoverlapping(message.as_ptr(), error.cast(), n);
                *error.add(n) = 0;
            }
            -1
        },
    }
}

/// C ABI: metadata is UTF-8 Request JSON. A successful before authorizes ONE
/// caller execution. A failed before means DO NOT EXECUTE. Never calls exec().
#[no_mangle]
pub(crate) unsafe extern "C" fn live_before(ctx: *mut LiveSave, data: *const u8, bytes: usize,
    step: usize, ticket: *mut u64, error: *mut c_char, capacity: usize) -> c_int {
    ffi_call(ctx, error, capacity, |s| {
        if data.is_null() || ticket.is_null() { return Err("null live request/ticket".into()); }
        *ticket = s.before(std::slice::from_raw_parts(data, bytes), step)?;
        Ok(())
    })
}

/// C ABI: success=0 aborts and poisons, success=1 completes post saves.
#[no_mangle]
pub(crate) unsafe extern "C" fn live_after(ctx: *mut LiveSave, ticket: u64, success: c_int,
    error: *mut c_char, capacity: usize) -> c_int {
    ffi_call(ctx, error, capacity, |s| {
        if success != 0 && success != 1 { return Err("invalid live completion status".into()); }
        s.after(ticket, success == 1)
    })
}

/// C ABI lifecycle for the SAME LiveSave pointer used by live_before/live_after.
/// Construction still requires a pinned, initialized Win and borrowed Execution.
#[no_mangle]
pub(crate) unsafe extern "C" fn live_tape_begin(ctx: *mut LiveSave, capacity: usize,
    error: *mut c_char, error_capacity: usize) -> c_int {
    ffi_call(ctx, error, error_capacity, |s| s.begin_tape(capacity))
}
#[no_mangle]
pub(crate) unsafe extern "C" fn live_walk_back(ctx: *mut LiveSave, step: usize,
    error: *mut c_char, error_capacity: usize) -> c_int {
    ffi_call(ctx, error, error_capacity, |s| s.walk_back(step))
}
#[no_mangle]
pub(crate) unsafe extern "C" fn live_tape_release(ctx: *mut LiveSave,
    error: *mut c_char, error_capacity: usize) -> c_int {
    ffi_call(ctx, error, error_capacity, |s| s.release_tape())
}

/// Synchronous TL callback. The caller must suppress nested save hooks and must
/// not re-enter this Win/session from the callback. Errors poison this context.
#[no_mangle]
pub(crate) unsafe extern "C" fn live_tangent(ctx: *mut LiveSave, data: *const u8, bytes: usize,
    step: usize, forward: Option<unsafe extern "C" fn(*mut c_void) -> c_int>, opaque: *mut c_void,
    error: *mut c_char, capacity: usize) -> c_int {
    ffi_call(ctx, error, capacity, |s| {
        if data.is_null() { return Err("null live tangent request".into()); }
        let forward = forward.ok_or("null live tangent callback")?;
        s.tangent(std::slice::from_raw_parts(data, bytes), step, forward, opaque)
    })
}

/// The model callback failed outside live_before/after. Its partial bars/tape
/// cannot be resumed by another sweep. Called with no outstanding Rust borrows.
pub(super) unsafe fn poison_for_sweep(ctx: *mut LiveSave) {
    if let Some(s) = ctx.as_mut() { s.failed = true; }
}
