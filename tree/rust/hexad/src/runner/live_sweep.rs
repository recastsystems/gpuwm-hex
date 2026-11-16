// SPDX-License-Identifier: Apache-2.0
//! Opt-in model-driven Revolve sweep. No Plan, Blob, or recorded forward call.
//! The constructor supplies a pinned LiveSave; callbacks supply complete rewind.
use super::{live_save, Revolve};
use std::collections::BTreeSet;
use std::ffi::{c_char, c_int, c_void};

pub type BoundaryCallback = unsafe extern "C" fn(*mut c_void, usize) -> c_int;
pub type StepCallback = unsafe extern "C" fn(*mut c_void, usize, u32) -> c_int;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Options {
    pub abi_version: u32,
    /// Zero is a no-op, before any other pointer is accessed. Default off.
    pub enabled: u32,
    pub steps: usize,
    /// Additional slots, excluding boundary zero, end, and one transient start.
    pub checkpoints: usize,
    /// Whole-step live tape capacity. Overflow refuses; no silent partial reverse.
    pub tape_bytes: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Callbacks {
    pub opaque: *mut c_void,
    pub store: Option<BoundaryCallback>,
    pub discard: Option<BoundaryCallback>,
    pub restore_device: Option<BoundaryCallback>,
    pub restore_host: Option<BoundaryCallback>,
    /// mode 0: first forward; 1: redo without saves; 2: redo with live saves.
    /// Only mode 0 may publish output. All modes execute numerical finalization.
    pub run_step: Option<StepCallback>,
    /// mode 0: retain boundary/command truth; modes 1/2: compare with that truth.
    pub check_step: Option<StepCallback>,
    /// Clear shadows and seed the objective, exactly once at the window end.
    pub seed_end: Option<BoundaryCallback>,
}

trait Backend {
    fn store(&mut self, k: usize) -> Result<(), String>;
    fn discard(&mut self, k: usize) -> Result<(), String>;
    fn restore(&mut self, k: usize) -> Result<(), String>;
    fn step(&mut self, k: usize, mode: u32) -> Result<(), String>;
    fn seed(&mut self, k: usize) -> Result<(), String>;
    fn reverse(&mut self, k: usize) -> Result<(), String>;
}

/// Same split, descent, right-before-left recursion and zero-slot loop as
/// Win::sweep_revolve_body/rev_range. `held` includes only additional slots.
struct Sweep<'a, B> { backend: &'a mut B, rv: Revolve, held: BTreeSet<usize> }
impl<B: Backend> Sweep<'_, B> {
    fn range(&mut self, lo: usize, hi: usize, slots: usize) -> Result<(), String> {
        if hi - lo == 1 {
            self.backend.restore(lo)?;
            return self.backend.reverse(hi);
        }
        if slots == 0 {
            for s in (lo + 1..=hi).rev() {
                self.backend.restore(lo)?;
                for t in lo + 1..s { self.backend.step(t, 1)?; }
                if s - 1 != lo {
                    // rev_range's snap_ram / rev_one start. This is a transient,
                    // not an additional Revolve slot. Rewind never touches bars.
                    self.backend.store(s - 1)?;
                    self.backend.restore(s - 1)?;
                }
                self.backend.reverse(s)?;
                if s - 1 != lo { self.backend.discard(s - 1)?; }
            }
            return Ok(());
        }
        let mid = lo + self.rv.split(hi - lo, slots);
        if !self.held.contains(&mid) {
            self.backend.restore(lo)?;
            for t in lo + 1..=mid { self.backend.step(t, 1)?; }
            self.backend.store(mid)?;
            self.held.insert(mid);
        }
        self.range(mid, hi, slots - 1)?;
        self.backend.discard(mid)?;
        self.held.remove(&mid);
        self.range(lo, mid, slots)
    }
}

fn sweep<B: Backend>(backend: &mut B, steps: usize, slots: usize) -> Result<(), String> {
    let rv = Revolve::new(steps, slots);
    let descent = rv.descent(steps, slots);
    backend.store(0)?;
    for s in 1..=steps {
        backend.step(s, 0)?;
        if s == steps || descent.contains(&s) { backend.store(s)?; }
    }
    backend.seed(steps)?;
    let mut driver = Sweep { backend, rv, held: descent.into_iter().collect() };
    driver.range(0, steps, slots)?;
    driver.backend.discard(steps)?;
    driver.backend.discard(0)
}

struct Foreign {
    // Raw throughout: run_step re-enters live_before/live_after on this handle.
    // No Rust reference to LiveSave, Win or Execution may span a callback.
    save: *mut live_save::LiveSave,
    cb: Callbacks,
    tape_bytes: usize,
    owned: BTreeSet<usize>,
}
impl Foreign {
    fn call(&self, f: Option<BoundaryCallback>, k: usize, name: &str) -> Result<(), String> {
        let rc = unsafe { f.ok_or_else(|| format!("missing {name} callback"))?(self.cb.opaque, k) };
        if rc == 0 { Ok(()) } else { Err(format!("live sweep {name}({k}) returned {rc}")) }
    }
    fn tape(&self, name: &str, f: impl FnOnce(*mut c_char, usize) -> c_int) -> Result<(), String> {
        let mut error = [0u8; 1024];
        let rc = f(error.as_mut_ptr().cast(), error.len());
        if rc == 0 { return Ok(()); }
        let n = error.iter().position(|&b| b == 0).unwrap_or(error.len());
        Err(format!("live sweep {name}: {}", String::from_utf8_lossy(&error[..n])))
    }
    fn release(&self) -> Result<(), String> {
        self.tape("release", |e, n| unsafe { live_save::live_tape_release(self.save, e, n) })
    }
}
impl Backend for Foreign {
    fn store(&mut self, k: usize) -> Result<(), String> {
        // Include failed partial stores in cleanup; discard must be idempotent.
        self.owned.insert(k);
        self.call(self.cb.store, k, "store")
    }
    fn discard(&mut self, k: usize) -> Result<(), String> {
        self.call(self.cb.discard, k, "discard")?;
        self.owned.remove(&k);
        Ok(())
    }
    fn restore(&mut self, k: usize) -> Result<(), String> {
        self.call(self.cb.restore_device, k, "restore_device")?;
        self.call(self.cb.restore_host, k, "restore_host")
    }
    fn step(&mut self, k: usize, mode: u32) -> Result<(), String> {
        let run = self.cb.run_step.ok_or("missing run_step callback")?;
        let rc = unsafe { run(self.cb.opaque, k, mode) };
        if rc != 0 { return Err(format!("live sweep run_step({k}, {mode}) returned {rc}")); }
        let check = self.cb.check_step.ok_or("missing check_step callback")?;
        let rc = unsafe { check(self.cb.opaque, k, mode) };
        if rc != 0 { return Err(format!("live sweep check_step({k}, {mode}) returned {rc}")); }
        Ok(())
    }
    fn seed(&mut self, k: usize) -> Result<(), String> { self.call(self.cb.seed_end, k, "seed_end") }
    fn reverse(&mut self, k: usize) -> Result<(), String> {
        self.tape("begin", |e, n| unsafe { live_save::live_tape_begin(self.save, self.tape_bytes, e, n) })?;
        let result = self.step(k, 2).and_then(|_| self.tape("walk_back", |e, n| unsafe {
            live_save::live_walk_back(self.save, k, e, n)
        }));
        let released = self.release();
        result.and(released)
    }
}

thread_local! { static RUNNING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
struct Running;
impl Drop for Running { fn drop(&mut self) { RUNNING.with(|v| v.set(false)); } }

/// Synchronous, owning-thread-only entry. Callbacks may call live_before/after,
/// but must not close/move this live window or recursively start another sweep.
/// All pointers and callbacks remain valid until return; no exception may cross C.
/// On failure the model/bars are partial: dispose the window, never reuse them.
#[no_mangle]
pub(crate) unsafe extern "C" fn hexad_live_sweep(
    save: *mut live_save::LiveSave, options: *const Options, callbacks: *const Callbacks,
    error: *mut c_char, error_capacity: usize,
) -> c_int {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), String> {
        let options = *options.as_ref().ok_or("null live sweep options")?;
        if options.enabled == 0 { return Ok(()); }
        if options.enabled != 1 || options.abi_version != 1 || options.steps == 0
            || options.checkpoints > options.steps || options.tape_bytes == 0 {
            return Err("invalid live sweep options".into());
        }
        if RUNNING.with(|v| v.replace(true)) { return Err("recursive live sweep".into()); }
        let _running = Running;
        if save.is_null() { return Err("null live save from live constructor".into()); }
        let cb = *callbacks.as_ref().ok_or("null live sweep callbacks")?;
        if cb.store.is_none() || cb.discard.is_none() || cb.restore_device.is_none()
            || cb.restore_host.is_none() || cb.run_step.is_none() || cb.check_step.is_none() || cb.seed_end.is_none() {
            return Err("incomplete live sweep callbacks".into());
        }
        let mut backend = Foreign { save, cb, tape_bytes: options.tape_bytes, owned: BTreeSet::new() };
        // Panic cleanup lives inside the backend's lifetime too.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sweep(&mut backend, options.steps, options.checkpoints)
        })).unwrap_or_else(|_| Err("panic in live sweep".into()));
        if result.is_err() {
            let _ = backend.release();
            for k in backend.owned.clone() { let _ = backend.discard(k); }
            live_save::poison_for_sweep(save);
        }
        result
    })).unwrap_or_else(|_| Err("panic in live sweep entry".into()));
    match result {
        Ok(()) => 0,
        Err(message) => {
            if !error.is_null() && error_capacity > 0 {
                let n = message.len().min(error_capacity - 1);
                std::ptr::copy_nonoverlapping(message.as_ptr(), error.cast(), n);
                *error.add(n) = 0;
            }
            -1
        }
    }
}
