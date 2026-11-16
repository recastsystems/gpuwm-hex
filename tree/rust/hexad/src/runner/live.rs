// SPDX-License-Identifier: Apache-2.0
//! Borrowed context/owners, separate from the recording Session. Explicit opt-in only.
use super::*;
use crate::live_owner::{self as om, Owner, View};

pub struct LiveSession {
    pub(crate) captures: usize,
    pub(crate) window_attached: bool,
    pub(crate) kset_path: PathBuf,
    pub(crate) owners: BTreeMap<u64, Owner>,
    pub(crate) kernels: HashMap<String, deriv::KernelDeriv>,
    pub(crate) ex: Execution,
}
/// JOIN (AD_LIVE_BARS_IN_WINDOW=1, off unless set): no bar is allocated at registration; the capture's window arena
/// holds the only copy, zeroed at capture open, and nothing is copied back at capture close (read the bars before
/// closing). THE BREAKAGE THIS PREVENTS: with one bar per owner here AND one in the window arena, the bars of one
/// 3 km step (18.66 GB) sat on the card twice and left 8 GiB for the tape (join-1, 2026-11-13, out-t400g: the tape
/// was full after 302 commands of one step).
pub(crate) fn bars_in_window() -> bool { std::env::var("AD_LIVE_BARS_IN_WINDOW").as_deref() == Ok("1") }
fn status(code: i32) -> Result<(), String> {
    if code < 0 { Err(format!("live owner error {code} (invalid=-1, stale/busy=-2, overlap=-3)")) } else { Ok(()) }
}
impl LiveSession {
    /// Every API call is serialized by the caller on its context's host thread.
    pub(crate) unsafe fn current(&self) -> Result<(), String> {
        let mut c = ptr::null_mut();
        checked("cuCtxGetCurrent live", (self.ex.cuda.ctx_get_current)(&mut c))?;
        if c != self.ex.context { return Err("live context is not current".into()); }
        Ok(())
    }
    pub(crate) fn owner(&self, id: u64, generation: u64) -> Result<&Owner, String> {
        self.owners.get(&id).filter(|o| o.generation == generation && o.refs != 0).ok_or("stale live owner".into())
    }
    /// Same loader and spec-hash admission as the old route; no Plan, References or Blob.
    unsafe fn open(device: i32, context: *mut c_void, stream: *mut c_void, kset: &Path) -> Result<Self, String> {
        let ex = Execution::from_live(device, context, stream)?;
        check_spec_hashes(&kset.join("kernels.json"))?;
        let kernels = deriv::load(&ex.cuda, &kset.join("kernels.json"), &kset.join("cubins"), None)?;
        Ok(Self { captures: 0, window_attached: false, kset_path: kset.to_path_buf(), owners: BTreeMap::new(), kernels, ex })
    }
    /// Unsupported entries retain the loader's reason; consumers must use this lookup.
    pub(crate) fn kernel(&self, name: &str) -> Result<&deriv::KernelDeriv, String> {
        let k = self.kernels.get(name).ok_or_else(|| format!("unknown live derivative {name}"))?;
        if let Some(why) = &k.unsupported { return Err(format!("{name}: {why}")); }
        Ok(k)
    }
    unsafe fn register(&mut self, mut n: Owner) -> Result<(), String> {
        if self.window_attached { return Err("owner inventory is fixed while live window is attached".into()); }
        status(om::valid(&n))?;
        if let Some(o) = self.owners.get(&n.id) {
            let code = om::admit(o, &n); status(code)?;
            if code == 1 { return Ok(()); }
        }
        if self.owners.values().any(|o| om::overlaps(o, &n)) { return status(-3); }
        n.refs = 1; n.shadow = 0;
        // One float32 byte-layout shadow per non-CONST owner. No primal copy.
        if n.constant == 0 && !bars_in_window() {
            checked("cuMemAlloc live shadow", (self.ex.cuda.allocate)(&mut n.shadow, n.bytes as usize))?;
            if let Err(e) = checked("cuMemsetD8Async live shadow", (self.ex.cuda.clear)(n.shadow, 0, n.bytes as usize, self.ex.streams[&0])) {
                let _ = (self.ex.cuda.free)(n.shadow); return Err(e);
            }
        }
        self.ex.allocations.insert(n.id, submit::Allocation { id: n.id, address: n.address, bytes: n.bytes as usize });
        self.ex.borrowed_allocations.insert(n.id);
        self.owners.insert(n.id, n);
        Ok(())
    }
    pub(crate) unsafe fn consumed(&mut self, id: u64, generation: u64) -> Result<bool, String> {
        let mut next = *self.owner(id, generation)?;
        if self.window_attached && next.refs <= 2 { return Err("live window retains its owner and registration pins".into()); }
        // Synchronize before decrement: a failed fence leaves the owner pinned and retryable.
        checked("cuStreamSynchronize live consumers", (self.ex.cuda.stream_synchronize)(self.ex.streams[&0]))?;
        let code = om::release(&mut next); status(code)?;
        if code == 0 {
            if next.shadow != 0 { checked("cuMemFree live shadow", (self.ex.cuda.free)(next.shadow))?; }
            next.shadow = 0;
            self.ex.allocations.remove(&id); self.ex.borrowed_allocations.remove(&id);
        }
        self.owners.insert(id, next);
        Ok(code == 0)
    }
}
impl Drop for LiveSession {
    fn drop(&mut self) { unsafe {
        let cu = &self.ex.cuda;
        let mut previous = ptr::null_mut();
        // If the device is lost, leak private buffers rather than free anything in a wrong context.
        if (cu.ctx_get_current)(&mut previous) != 0 || (cu.ctx_set_current)(self.ex.context) != 0 { return; }
        if (cu.stream_synchronize)(self.ex.streams[&0]) == 0 {
            for o in self.owners.values() { if o.shadow != 0 { let _ = (cu.free)(o.shadow); } }
            for k in self.kernels.values() {
                for (p, _) in k.derived.borrow().values() { let _ = (cu.free)(*p); }
                if !k.module.is_null() { let _ = (cu.module_unload)(k.module); }
            }
        }
        let _ = (cu.ctx_set_current)(previous);
    } }
}
#[no_mangle] pub unsafe extern "C" fn hexad_live_open(device: i32, context: *mut c_void, stream: *mut c_void, kset: *const c_char) -> *mut LiveSession {
    let mut out = ptr::null_mut();
    let code = guard(|| { out = Box::into_raw(Box::new(LiveSession::open(device, context, stream, Path::new(cstr(kset, "live kset")?))?)); Ok(()) });
    if code == 0 { out } else { ptr::null_mut() }
}
#[no_mangle] pub unsafe extern "C" fn hexad_live_register_owner(s: *mut LiveSession, id: u64, generation: u64, address: u64, bytes: u64, constant: u64) -> i32 {
    guard(|| { let s = s.as_mut().ok_or("null live session")?; s.current()?;
        s.register(Owner { id, generation, address, bytes, constant, ..Owner::default() }) })
}
#[no_mangle] pub unsafe extern "C" fn hexad_live_view(s: *mut LiveSession, v: *const View, primal: *mut u64, shadow: *mut u64) -> i32 {
    guard(|| { let s = s.as_ref().ok_or("null live session")?; s.current()?;
        let v = v.as_ref().ok_or("null live view")?; let o = s.owner(v.id, v.generation)?;
        status(om::resolve(o, v))?;
        let p = primal.as_mut().ok_or("null primal result")?; let b = shadow.as_mut().ok_or("null shadow result")?;
        *p = o.address + v.offset; *b = if o.shadow == 0 { 0 } else { o.shadow + v.offset }; Ok(()) })
}
#[no_mangle] pub unsafe extern "C" fn hexad_live_pin_owner(s: *mut LiveSession, id: u64, generation: u64) -> i32 {
    guard(|| { let s = s.as_mut().ok_or("null live session")?; s.current()?; s.owner(id, generation)?;
        status(om::pin(s.owners.get_mut(&id).unwrap())) })
}
#[no_mangle] pub unsafe extern "C" fn hexad_live_release_owner(s: *mut LiveSession, id: u64, generation: u64, released: *mut i32) -> i32 {
    guard(|| { let s = s.as_mut().ok_or("null live session")?; s.current()?;
        let out = released.as_mut().ok_or("null released result")?; *out = s.consumed(id, generation)? as i32; Ok(()) })
}
/// On failure the handle and all Python pins remain live; caller may retry in its context.
#[no_mangle] pub unsafe extern "C" fn hexad_live_close(s: *mut LiveSession) -> i32 {
    guard(|| { if let Some(v) = s.as_ref() { v.current()?;
        if v.captures != 0 { return Err("live session has a join capture".into()); }
        checked("cuStreamSynchronize live close", (v.ex.cuda.stream_synchronize)(v.ex.streams[&0]))?;
        drop(Box::from_raw(s)); } Ok(()) })
}
