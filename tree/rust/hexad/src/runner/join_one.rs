// SPDX-License-Identifier: Apache-2.0
//! One real producer command inside a recorded reverse window. Explicit C API only.
use super::*;
use crate::plan;
use std::ffi::{c_char, c_int};
use capi::live::LiveSession;

pub(crate) const KIND: &str = "add_inplace_f32";
pub(crate) type Callback = unsafe extern "C" fn(*mut c_void, *const u8, usize, *mut c_void) -> c_int;
pub(crate) struct JoinOne {
    pub step: usize,
    pub callback: Callback,
    pub user: *mut c_void,
    pub hits: usize,
    pub failed: bool,
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Binding {
    argument: usize, owner: u64, generation: u64, offset: u64, address: u64,
    bytes: usize, allocation: u64, plan_offset: usize,
}
struct Copy { live: u64, plan: u64, bytes: usize }
// Only valid during Callback; never a Session, HexadLive, or an arbitrary Win pointer.
struct Invocation {
    w: *mut Win, ex: *mut Execution, request: Vec<u8>, step: usize,
    save: Option<Box<live_save::LiveSave>>, owners: *mut LiveSession,
    pins: Vec<(u64, u64)>, copies: Vec<Copy>, copied_back: bool,
}
impl Drop for Invocation {
    fn drop(&mut self) { unsafe {
        self.save = None;
        if !self.owners.is_null() {
            for &(id, generation) in &self.pins { let _ = (*self.owners).consumed(id, generation); }
            (*self.owners).captures -= 1;
        }
    } }
}
unsafe fn call(error: *mut c_char, cap: usize, f: impl FnOnce() -> Result<(), String>) -> c_int {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|_| Err("panic in join-one".into())) {
        Ok(()) => 0,
        Err(e) => { if !error.is_null() && cap > 0 { let n = e.len().min(cap - 1); ptr::copy_nonoverlapping(e.as_ptr(), error.cast(), n); *error.add(n) = 0; } -1 }
    }
}

/// The four resolved arguments, with exact scalar bytes, come from this replay's
/// current command. This bridge does not assume a recorded address is an identity.
pub(crate) unsafe fn run(w: *mut Win, ex: *mut Execution, step: usize, command: usize,
    request: Vec<u8>, hook: &mut JoinOne) -> Result<Tp, String> {
    if hook.failed || hook.hits != 0 { return Err("join-one already used or poisoned (one launch only)".into()); }
    hook.failed = true; // Any early return is terminal; never silently use the old launch.
    dev::sync(&(*ex).cuda)?;
    let mut inv = Invocation { w, ex, request, step, save: None, owners: ptr::null_mut(), pins: Vec::new(), copies: Vec::new(), copied_back: false };
    let data = inv.request.clone();
    let rc = (hook.callback)(&mut inv as *mut Invocation as *mut c_void, data.as_ptr(), data.len(), hook.user);
    if rc != 0 { return Err("join-one producer callback failed; binder launch was not executed".into()); }
    if !inv.copied_back { return Err("join-one producer did not return its target".into()); }
    let save = inv.save.as_mut().ok_or("join-one producer did not construct LiveSave")?;
    let mut records = save.take_records()?;
    if records.len() != 1 { return Err("join-one needs exactly one completed live record".into()); }
    dev::sync(&(*ex).cuda)?;
    hook.hits = 1;
    hook.failed = false;
    eprintln!("{}", json!({"event":"join_one","step":step,"command":command,"function":KIND,
        "producer_launches":1,"binder_launches":0,"live_records":1,"argument_bytes":inv.copies.iter().map(|c| c.bytes).sum::<usize>()}));
    Ok(records.pop().unwrap())
}

/// Factory for a callback-scoped capture. The invocation owns/destroys the result;
/// do not close/re-enter the recorded Session or the owner session from Callback.
/// A map is exactly two Binding rows, argument 2 (increment), 3 (target).
#[no_mangle]
pub unsafe extern "C" fn hexad_live_save_open(invocation: *mut c_void, owners: *mut LiveSession,
    map: *const u8, bytes: usize, out: *mut *mut c_void, error: *mut c_char, cap: usize) -> c_int {
    if !out.is_null() { *out = ptr::null_mut(); }
    call(error, cap, || {
        let i = (invocation as *mut Invocation).as_mut().ok_or("null join invocation")?;
        if i.save.is_some() || !i.owners.is_null() { return Err("join capture already constructed".into()); }
        let live = owners.as_mut().ok_or("null owner session")?;
        live.current()?;
        if live.window_attached { return Err("join-one cannot attach to a full live window".into()); }
        if live.ex.context != (*i.ex).context { return Err("join requires the same current CUDA context".into()); }
        if map.is_null() || out.is_null() { return Err("null map/result".into()); }
        let rows: Vec<Binding> = serde_json::from_slice(std::slice::from_raw_parts(map, bytes)).map_err(|e| e.to_string())?;
        let req: Value = serde_json::from_slice(&i.request).map_err(|e| e.to_string())?;
        let op: Operation = serde_json::from_value(req["operation"].clone()).map_err(|e| e.to_string())?;
        let Operation::Kernel { function, arguments, .. } = op else { return Err("join requires a kernel".into()) };
        if function != KIND || arguments.len() != 4 || rows.len() != 2 { return Err("join requires add_inplace's four arguments and two owners".into()); }
        let mut n = 1usize;
        for a in &arguments[..2] {
            let submit::Argument::Bytes(b) = (*i.ex).argument(a, i.step, None)? else { return Err("join dimension is not bytes".into()) };
            let dim = i32::from_le_bytes(b.try_into().map_err(|_| "join dimension must be int32")?);
            if dim <= 0 { return Err("join dimension must be positive".into()); }
            n = n.checked_mul(dim as usize).ok_or("join extent overflow")?;
        }
        let n = n.checked_mul(4).ok_or("join byte extent overflow")?;
        let mut copies = Vec::new();
        for (arg, b) in (2..4).zip(&rows) {
            if b.argument != arg || b.bytes != n { return Err("join owner argument/extent mismatch".into()); }
            let plan::Argument::Device { target } = &arguments[arg] else { return Err("join requires direct device arguments".into()) };
            if target.allocation != b.allocation || target.offset != b.plan_offset { return Err("join wrong plan owner or offset".into()); }
            let owner = live.owner(b.owner, b.generation)?;
            if b.offset > owner.bytes || n as u64 > owner.bytes - b.offset || owner.address.checked_add(b.offset) != Some(b.address) {
                return Err("join wrong live owner, address or extent".into());
            }
            if owner.constant != 0 { return Err("join mutable operands cannot be CONST owners".into()); }
            copies.push(Copy { live: b.address, plan: (*i.ex).address(*target, n)?, bytes: n });
        }
        let overlap = |a: u64, b: u64| a < b.saturating_add(n as u64) && b < a.saturating_add(n as u64);
        if overlap(copies[0].live, copies[1].live) || overlap(copies[0].plan, copies[1].plan) {
            return Err("join requires nonoverlapping increment/target views".into());
        }
        // Pin validated generations before any copy. The session cannot close while captured.
        live.captures = live.captures.checked_add(1).ok_or("join capture count overflow")?;
        i.owners = owners;
        for b in &rows {
            let owner = live.owners.get_mut(&b.owner).unwrap();
            if crate::live_owner::pin(owner) < 0 { return Err("cannot pin join owner".into()); }
            i.pins.push((b.owner, b.generation));
        }
        dev::sync(&(*i.ex).cuda)?;
        for c in &copies { dev::copy(&(*i.ex).cuda, c.live, c.plan, c.bytes)?; }
        dev::sync(&(*i.ex).cuda)?;
        i.copies = copies;
        let mut save = Box::new(live_save::LiveSave::attach(i.w, i.ex)?);
        save.expect(i.step, serde_json::from_slice(&i.request).map_err(|e| e.to_string())?);
        *out = &mut *save as *mut live_save::LiveSave as *mut c_void;
        i.save = Some(save);
        Ok(())
    })
}

/// Called after the one forward launch, before live_after, so rec_post sees its
/// result at the plan address and the ordinary remainder of the step sees it too.
#[no_mangle]
pub unsafe extern "C" fn hexad_join_one_return(invocation: *mut c_void, error: *mut c_char, cap: usize) -> c_int {
    call(error, cap, || {
        let i = (invocation as *mut Invocation).as_mut().ok_or("null join invocation")?;
        if i.save.is_none() || i.copied_back { return Err("join target returned twice or before open".into()); }
        dev::sync(&(*i.ex).cuda)?;
        let c = &i.copies[1];
        dev::copy(&(*i.ex).cuda, c.plan, c.live, c.bytes)?;
        dev::sync(&(*i.ex).cuda)?;
        i.copied_back = true;
        Ok(())
    })
}
