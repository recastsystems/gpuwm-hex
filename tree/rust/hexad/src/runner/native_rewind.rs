// SPDX-License-Identifier: Apache-2.0
//! Opt-in native storage for a host adapter's complete, explicitly enumerated state.
//! This module owns checkpoint bytes, NOT a NativePhysicsSession. See hexad_native_rewind.h.
use std::{cell::RefCell, collections::BTreeSet, ffi::{c_char, c_void, CString}, ptr,
          sync::atomic::{AtomicU64, Ordering}};

#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Layout { pub generation: u64, pub bytes: u64, pub layout: u64 }
type Fence = unsafe extern "C" fn(*mut c_void) -> i32;
type Inspect = unsafe extern "C" fn(*mut c_void, u64, *mut Layout) -> i32;
type Read = unsafe extern "C" fn(*mut c_void, u64, *mut u8, usize) -> i32;
type Validate = unsafe extern "C" fn(*mut c_void, u64, *const u8, usize) -> i32;
type Write = unsafe extern "C" fn(*mut c_void, u64, *const u8, usize) -> i32;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Callbacks {
    pub fence: Option<Fence>, pub inspect: Option<Inspect>, pub read: Option<Read>,
    pub validate: Option<Validate>, pub write: Option<Write>,
}
/// Calls are serialized by the owner, including its step loop and shared module caches.
pub struct Rewind {
    token: u64, user: *mut c_void, cb: Callbacks, required: BTreeSet<u64>,
    registered: BTreeSet<u64>, limit: usize, sealed: bool, poisoned: bool,
}
struct Region { id: u64, layout: Layout, data: Vec<u8> }
/// Immutable; does not borrow the provider. Valid only with the originating Rewind.
pub struct Snapshot { token: u64, regions: Vec<Region>, bytes: usize }
static NEXT: AtomicU64 = AtomicU64::new(1);
thread_local! { static ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap()); }
fn guard(f: impl FnOnce() -> Result<(), String>) -> i32 {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    let error = match result { Ok(Ok(())) => None, Ok(Err(e)) => Some(e), Err(_) => Some("native rewind panic".into()) };
    ERROR.with(|v| *v.borrow_mut() = CString::new(error.as_deref().unwrap_or("").replace('\0', "?")).unwrap());
    if error.is_some() { -1 } else { 0 }
}
fn check(code: i32, what: &str, id: u64) -> Result<(), String> {
    if code == 0 { Ok(()) } else { Err(format!("native rewind {what} region {id}: callback status {code}")) }
}
impl Rewind {
    fn ready(&self) -> Result<(), String> {
        if self.poisoned { return Err("native rewind provider is poisoned after a partial restore".into()); }
        if !self.sealed { return Err("native rewind inventory is not sealed".into()); }
        Ok(())
    }
    unsafe fn fence(&self) -> Result<(), String> { check((self.cb.fence.unwrap())(self.user), "fence", 0) }
    unsafe fn inspect(&self, id: u64) -> Result<Layout, String> {
        let mut v = Layout::default();
        check((self.cb.inspect.unwrap())(self.user, id, &mut v), "inspect", id)?;
        if v.generation == 0 || v.layout == 0 { return Err(format!("native rewind absent identity for region {id}")); }
        usize::try_from(v.bytes).map_err(|_| "native rewind extent exceeds size_t")?;
        Ok(v)
    }
    unsafe fn capture(&self) -> Result<Snapshot, String> {
        self.ready()?;
        self.fence()?;
        // Inspect every extent and budget before allocating or reading any region.
        let mut layouts = Vec::new();
        let mut bytes = 0usize;
        for &id in &self.registered {
            let layout = self.inspect(id)?;
            bytes = bytes.checked_add(layout.bytes as usize).ok_or("native rewind byte sum overflow")?;
            if bytes > self.limit { return Err("native rewind snapshot exceeds byte limit".into()); }
            layouts.push((id, layout));
        }
        let mut regions = Vec::new();
        for (id, layout) in layouts {
            let mut data = Vec::new();
            data.try_reserve_exact(layout.bytes as usize).map_err(|e| format!("native rewind host allocation: {e}"))?;
            data.resize(layout.bytes as usize, 0);
            check((self.cb.read.unwrap())(self.user, id, data.as_mut_ptr(), data.len()), "read", id)?;
            regions.push(Region { id, layout, data });
        }
        self.fence()?;
        // Detect a provider that replaced a region during capture. No snapshot is published on failure.
        for r in &regions {
            if self.inspect(r.id)? != r.layout { return Err(format!("native rewind layout changed during capture: {}", r.id)); }
        }
        Ok(Snapshot { token: self.token, regions, bytes })
    }
    unsafe fn restore(&mut self, snap: &Snapshot) -> Result<(), String> {
        self.ready()?;
        if snap.token != self.token || snap.regions.len() != self.registered.len() {
            return Err("native rewind snapshot belongs to another provider".into());
        }
        self.fence()?;
        // All metadata and semantic validation precede the first write. No cached address is replayed.
        for r in &snap.regions {
            if !self.registered.contains(&r.id) || self.inspect(r.id)? != r.layout {
                return Err(format!("native rewind stale owner or layout: {}", r.id));
            }
            check((self.cb.validate.unwrap())(self.user, r.id, r.data.as_ptr(), r.data.len()), "validate", r.id)?;
        }
        // An external write can fail after earlier regions changed. Do not claim transactional rollback.
        self.poisoned = true;
        for r in &snap.regions {
            check((self.cb.write.unwrap())(self.user, r.id, r.data.as_ptr(), r.data.len()), "write", r.id)?;
        }
        self.fence()?;
        for r in &snap.regions {
            if self.inspect(r.id)? != r.layout { return Err(format!("native rewind write changed owner or layout: {}", r.id)); }
        }
        self.poisoned = false;
        Ok(())
    }
}

#[no_mangle]
pub extern "C" fn hexad_native_rewind_last_error() -> *const c_char { ERROR.with(|e| e.borrow().as_ptr()) }

#[no_mangle]
pub unsafe extern "C" fn hexad_native_rewind_open(enabled: i32, required: *const u64, count: usize,
    byte_limit: usize, callbacks: *const Callbacks, user: *mut c_void, out: *mut *mut Rewind) -> i32 {
    guard(|| {
        let out = out.as_mut().ok_or("native rewind null result")?; *out = ptr::null_mut();
        if enabled == 0 { return Ok(()); } // Default-off does not touch callbacks, inputs or the model.
        if enabled != 1 || count == 0 || count > isize::MAX as usize / 8 || required.is_null() || byte_limit == 0 {
            return Err("native rewind invalid option, manifest or limit".into());
        }
        let cb = *callbacks.as_ref().ok_or("native rewind null callbacks")?;
        if cb.fence.is_none() || cb.inspect.is_none() || cb.read.is_none() || cb.validate.is_none() || cb.write.is_none() {
            return Err("native rewind incomplete callbacks".into());
        }
        let ids: BTreeSet<u64> = std::slice::from_raw_parts(required, count).iter().copied().collect();
        if ids.len() != count || ids.contains(&0) { return Err("native rewind duplicate or zero region id".into()); }
        let token = NEXT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
            .map_err(|_| "native rewind identity exhausted")?;
        *out = Box::into_raw(Box::new(Rewind { token, user, cb, required: ids, registered: BTreeSet::new(),
            limit: byte_limit, sealed: false, poisoned: false }));
        Ok(())
    })
}
#[no_mangle]
pub unsafe extern "C" fn hexad_native_rewind_register(s: *mut Rewind, id: u64) -> i32 {
    guard(|| {
        let s = s.as_mut().ok_or("native rewind null provider")?;
        if s.sealed || !s.required.contains(&id) || !s.registered.insert(id) {
            return Err("native rewind sealed, unknown or duplicate region".into());
        }
        Ok(())
    })
}
#[no_mangle]
pub unsafe extern "C" fn hexad_native_rewind_seal(s: *mut Rewind) -> i32 {
    guard(|| {
        let s = s.as_mut().ok_or("native rewind null provider")?;
        if s.registered != s.required { return Err("native rewind missing required region".into()); }
        s.sealed = true; Ok(())
    })
}
#[no_mangle]
pub unsafe extern "C" fn hexad_native_rewind_capture(s: *const Rewind, out: *mut *mut Snapshot) -> i32 {
    guard(|| {
        let out = out.as_mut().ok_or("native rewind null snapshot result")?; *out = ptr::null_mut();
        let s = s.as_ref().ok_or("native rewind null provider")?;
        *out = Box::into_raw(Box::new(s.capture()?)); Ok(())
    })
}
#[no_mangle]
pub unsafe extern "C" fn hexad_native_rewind_restore(s: *mut Rewind, snap: *const Snapshot) -> i32 {
    guard(|| s.as_mut().ok_or("native rewind null provider")?.restore(snap.as_ref().ok_or("native rewind null snapshot")?))
}
#[no_mangle]
pub unsafe extern "C" fn hexad_native_rewind_bytes(snap: *const Snapshot, out: *mut usize) -> i32 {
    guard(|| {
        let out = out.as_mut().ok_or("native rewind null byte result")?;
        *out = snap.as_ref().ok_or("native rewind null snapshot")?.bytes; Ok(())
    })
}
#[no_mangle]
pub unsafe extern "C" fn hexad_native_rewind_snapshot_close(snap: *mut Snapshot) {
    if !snap.is_null() { drop(Box::from_raw(snap)); }
}
#[no_mangle]
pub unsafe extern "C" fn hexad_native_rewind_close(s: *mut Rewind) {
    if !s.is_null() { drop(Box::from_raw(s)); }
}
