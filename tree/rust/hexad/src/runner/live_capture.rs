// SPDX-License-Identifier: Apache-2.0
//! Explicit borrowed-session factory. No legacy caller enters this module.
use super::*;
use super::capi::live::LiveSession;
use super::live_construct::{Input, LiveProgram};
use super::live_save::LiveSave;
use std::cell::RefCell;
use std::collections::HashSet;
use std::ffi::{c_char, c_int};

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Configuration {
    schema: String,
    program: Input,
    options: HashMap<String, String>,
}
struct Capture {
    // Drop capture before its window; the session remains caller-owned.
    save: Box<LiveSave>,
    win: Box<Win>,
    session: *mut LiveSession,
    program: LiveProgram,
    old_shadows: BTreeMap<u64, u64>,
    old_address_bearing: HashSet<u64>,
}
thread_local! {
    // Only factory-owned captures appear here, on the owning host thread.
    static CAPTURES: RefCell<BTreeMap<usize, Capture>> = RefCell::new(BTreeMap::new());
}
unsafe fn call(error: *mut c_char, cap: usize, f: impl FnOnce() -> Result<(), String>) -> c_int {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .unwrap_or_else(|_| Err("panic in live capture factory".into())) {
        Ok(()) => 0,
        Err(e) => { if !error.is_null() && cap != 0 {
            let n = e.len().min(cap - 1); ptr::copy_nonoverlapping(e.as_ptr(), error.cast(), n); *error.add(n) = 0;
        } -1 }
    }
}
// Mirrors LiveWindow's current major-resource cleanup. Lazy-resource closure
// remains subject to the same card audit as that constructor.
unsafe fn free_window(w: &Win, ex: &Execution) {
    let cu = &ex.cuda;
    for p in [w.sh_base, w.tape.base, w.scratch.base, w.tmp[0], w.tmp[1], w.tmp[2], w.part, w.fail] {
        if p != 0 { let _ = (cu.free)(p); }
    }
    if let Some((_, ar)) = &w.tempo { let _ = (cu.free)(ar.base); }
    for (p, _) in &w.dev_anchors { let _ = (cu.free)(*p); }
    for k in w.kernels.values() {
        for (p, _) in k.derived.borrow().values() { let _ = (cu.free)(*p); }
        if !k.module.is_null() { let _ = (cu.module_unload)(k.module); }
    }
}
#[no_mangle]
pub unsafe extern "C" fn hexad_live_capture_open(session: *mut LiveSession,
    data: *const u8, bytes: usize, out: *mut *mut c_void, error: *mut c_char, cap: usize) -> c_int {
    if !out.is_null() { *out = ptr::null_mut(); }
    call(error, cap, || {
        if data.is_null() || out.is_null() { return Err("null live capture configuration/result".into()); }
        let s = session.as_mut().ok_or("null live session")?;
        s.current()?;
        if s.captures != 0 { return Err("live session already captured".into()); }
        let cfg: Configuration = serde_json::from_slice(std::slice::from_raw_parts(data, bytes)).map_err(|e| e.to_string())?;
        if cfg.schema != "hexad.live-capture/v1" { return Err("live capture schema".into()); }
        // Both loaders must refer to the same immutable kernel set and images.
        for (key, file) in [("kernels", "kernels.json"), ("cubins", "cubins")] {
            let supplied = std::fs::canonicalize(req(&cfg.options, key)?).map_err(|e| e.to_string())?;
            let opened = std::fs::canonicalize(s.kset_path.join(file)).map_err(|e| e.to_string())?;
            if supplied != opened { return Err(format!("capture {key} differs from live session")); }
        }
        let program = LiveProgram::new(cfg.program, &s.owners)?;
        // Refuse pin overflow before allocating or changing any owner.
        if s.owners.values().any(|o| o.refs == 0 || o.refs == u64::MAX) { return Err("cannot pin live owner".into()); }
        // jn-provider card finding (2026-11-13, join-1, jp-live3): the session's own shadows stay allocated
        // while the window works in its arena, so every bar is held twice (19.4 GB of a 3 km step on a 102 GB
        // card), and the whole-step tape did not fit ("live pre-save budget exhausted" at command 797 of 1,733
        // with 12 GiB). With free-session-shadows=1 (the caller's word that no bar has been written yet: the
        // joined driver seeds after the capture opens) they are freed BEFORE the window allocates, the window's
        // shadows start at zero, and close allocates the session's again before copying the bars back.
        // Off unless asked: the default keeps both.
        let free_first = cfg.options.get("free-session-shadows").map(String::as_str) == Some("1");
        if free_first {
            dev::sync(&s.ex.cuda)?;
            for o in s.owners.values_mut() {
                if o.constant == 0 && o.shadow != 0 { checked("cuMemFree session shadow", (s.ex.cuda.free)(o.shadow))?; o.shadow = 0; }
            }
        }
        let mut win = Box::new(Win::from_live(&s.ex, &program, &cfg.options)?);
        let prepared = (|| -> Result<Box<LiveSave>, String> {
            for m in &program.input.owners {
                let o = &s.owners[&m.owner];
                if !m.constant {
                    if o.shadow == 0 {
                        if !free_first { return Err("missing live shadow".into()); }
                        checked("cuMemsetD8Async window shadow",
                            (s.ex.cuda.clear)(win.shadows[&m.owner].address, 0, m.bytes, s.ex.streams[&0]))?;
                    } else {
                        dev::copy(&s.ex.cuda, win.shadows[&m.owner].address, o.shadow, m.bytes)?;
                    }
                }
            }
            dev::sync(&s.ex.cuda)?;
            Ok(Box::new(LiveSave::new(&mut *win, &mut s.ex)?))
        })();
        let save = match prepared { Ok(v) => v, Err(e) => { free_window(&win, &s.ex); return Err(e); } };
        let old_shadows = s.owners.iter().map(|(&id,o)| (id,o.shadow)).collect();
        let old_address_bearing = s.ex.address_bearing.clone();
        for o in s.owners.values_mut() {
            o.refs += 1;
            o.shadow = if o.constant == 0 { win.shadows[&o.id].address } else { 0 };
        }
        s.ex.address_bearing = program.input.owners.iter().filter(|m| m.address_bearing).map(|m| m.owner).collect();
        win.relocs = program.input.owners.iter().filter(|m| !m.pointer_offsets.is_empty())
            .map(|m| (m.owner, m.pointer_offsets.iter().map(|p| (*p,8)).collect())).collect();
        win.address_bearing = s.ex.address_bearing.clone();
        s.captures = 1;
        s.window_attached = true;
        let mut c = Capture { save, win, session, program, old_shadows, old_address_bearing };
        let handle = &mut *c.save as *mut LiveSave as *mut c_void;
        CAPTURES.with(|all| all.borrow_mut().insert(handle as usize, c));
        *out = handle;
        Ok(())
    })
}
/// Size-query then copy; bytes are Request JSON, with no trailing NUL.
#[no_mangle]
pub unsafe extern "C" fn hexad_live_capture_request(handle: *mut c_void, step_index: usize,
    command_index: usize, out: *mut u8, capacity: usize, needed: *mut usize,
    error: *mut c_char, cap: usize) -> c_int {
    call(error, cap, || CAPTURES.with(|all| {
        let all = all.borrow();
        let c = all.get(&(handle as usize)).ok_or("unknown live capture on this thread")?;
        let s = &*c.session; s.current()?;
        let bytes = c.program.request(step_index, command_index, &s.owners)?;
        *needed.as_mut().ok_or("null request size")? = bytes.len();
        if out.is_null() && capacity == 0 { return Ok(()); }
        if out.is_null() || capacity < bytes.len() { return Err("request buffer too small".into()); }
        ptr::copy_nonoverlapping(bytes.as_ptr(), out, bytes.len());
        Ok(())
    }))
}
#[no_mangle]
pub unsafe extern "C" fn hexad_live_capture_close(handle: *mut c_void, error: *mut c_char, cap: usize) -> c_int {
    call(error, cap, || CAPTURES.with(|all| {
        let mut all = all.borrow_mut();
        let c = all.get_mut(&(handle as usize)).ok_or("unknown live capture on this thread")?;
        let s = &mut *c.session; s.current()?;
        // On fence/copy error retain every allocation and pin for caller cleanup.
        c.save.release_tape()?;
        // free-session-shadows=1 (jp-live8): allocating the session's shadows again beside the window's tape and
        // scratch failed ("cuMemAlloc live shadow failed with CUDA status 2" on a 102 GB card after a whole 3 km
        // step was taped and walked back). The tape is released above; its memory and the scratch are given back
        // here first (free_window skips a zero pointer), then the shadows are allocated and the bars copied.
        if c.program.input.owners.iter().any(|m| !m.constant && c.old_shadows[&m.owner] == 0) {
            dev::sync(&s.ex.cuda)?;
            for p in [&mut c.win.tape.base, &mut c.win.scratch.base] {
                if *p != 0 { checked("cuMemFree window tape or scratch", (s.ex.cuda.free)(*p))?; *p = 0; }
            }
        }
        for m in &c.program.input.owners {
            if !m.constant {
                let mut dst = c.old_shadows[&m.owner];
                if dst == 0 {
                    // freed at open (free-session-shadows=1): allocate the session shadow again for the bars
                    checked("cuMemAlloc live shadow", (s.ex.cuda.allocate)(&mut dst, m.bytes))?;
                    c.old_shadows.insert(m.owner, dst);
                }
                dev::copy(&s.ex.cuda, dst, c.win.shadows[&m.owner].address, m.bytes)?;
            }
        }
        dev::sync(&s.ex.cuda)?;
        for o in s.owners.values_mut() { o.shadow = c.old_shadows[&o.id]; o.refs -= 1; }
        s.ex.address_bearing = std::mem::take(&mut c.old_address_bearing);
        s.window_attached = false; s.captures = 0;
        let c = all.remove(&(handle as usize)).unwrap();
        drop(c.save);
        free_window(&c.win, &s.ex);
        Ok(())
    }))
}
