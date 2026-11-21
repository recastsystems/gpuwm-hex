// SPDX-License-Identifier: Apache-2.0
//! Torch bridge, the reverse side (2026-11-19). Default off: with none of the words below set, nothing
//! here runs and every window, plan and test is the same bytes.
//!
//! A learned correction adds a small tendency to the model's physics tendency sum once a step. Training it needs, at
//! every step k of the reverse, the cotangent of that slot: dJ/d(tendency added at step k). In the recorded plan the
//! physics tendency sum is what `mpas_rk_cells_v842_f32` (physics_get_tend: out_theta, out_q, out_rho) and
//! `mpas_rk_edges_v842_f32` (out_u) write once a step; the dynamics then reads it through every RK stage. An external
//! tendency added right after those launches has the identity as its derivative, so its cotangent is exactly the
//! cotangent view of those outputs at the moment the reverse reaches their launch, before their adjoint entries run.
//!
//! * `AD_TAP=function:arg[,function:arg..]` with `AD_TAP_DIR=dir`: at every launch of `function` in every reverse, the
//!   cotangent view of argument `arg` (a name, or an index into the kernel row's argument list) is read before the
//!   launch's adjoint entries and written raw (float32, the forward argument's own layout) to
//!   `tap-s<step>-<function>-<arg>-r<n>.f32` (`n` counts launches of `function` within the step in reverse order;
//!   one launch a step gives r0). A later reverse of the same window overwrites the file; the stderr line
//!   `tb_tap` carries the reverse's count, the sha256, the nonzero words and the sum of squares.
//! * In-process (C ABI, `hexad_set_tap`): the same point calls the caller's function with the DEVICE address of the
//!   view and its bytes (no copy; valid only during the call; the caller may read it, copy it into its own per-step
//!   card buffer, or ADD into it, which adds a cotangent at the slot).
//! * `AD_KICK=step:function:arg:file.f32[:occurrence]` (proof only): in the Taylor rows' perturbed forwards, right
//!   after launch number `occurrence` (forward order, default 0) of `function` at step `step`, h times the float32
//!   words of `file` are added to argument `arg` (index into the recorded launch arguments) from its start, so the
//!   row's difference quotient (J(h) - J(0)) / h can be compared with <file, tap at that step>. The base forward, the
//!   tangent and every reverse are untouched (the kick's size is zero outside the rows).
use super::*;
use std::ffi::{c_char, c_void, CString};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

pub(crate) struct TapSpec { pub f: String, pub arg: String }

fn parse_taps(v: &str) -> Result<Vec<TapSpec>, String> {
    v.split(',').map(str::trim).filter(|x| !x.is_empty()).map(|x| match x.split_once(':') {
        Some((f, a)) if !f.is_empty() && !a.is_empty() => Ok(TapSpec { f: f.to_owned(), arg: a.to_owned() }),
        _ => Err(format!("AD_TAP item {x}: want function:arg")),
    }).collect()
}

fn taps() -> Result<&'static Vec<TapSpec>, String> {
    static T: OnceLock<Result<Vec<TapSpec>, String>> = OnceLock::new();
    T.get_or_init(|| parse_taps(&std::env::var("AD_TAP").unwrap_or_default())).as_ref().map_err(Clone::clone)
}

/// The caller's per-launch function (C ABI): (user, step, function, argument, device address, bytes) -> 0 to go on.
pub type TapFn = unsafe extern "C" fn(*mut c_void, i32, *const c_char, *const c_char, u64, u64) -> i32;
struct Cb { f: TapFn, user: usize, list: Vec<TapSpec> }
static CB: Mutex<Option<Cb>> = Mutex::new(None);
/// Set (or with `f` None, clear) the in-process tap. `list` is AD_TAP's form; empty means the physics tendency sum.
pub(crate) fn set_callback(f: Option<TapFn>, user: *mut c_void, list: &str) -> Result<(), String> {
    let list = if list.trim().is_empty() { parse_taps(SLOT_DEFAULT)? } else { parse_taps(list)? };
    *CB.lock().map_err(|_| "tap lock")? = f.map(|f| Cb { f, user: user as usize, list });
    Ok(())
}
/// The physics tendency sum of the hrrrv5 plans: theta, every scalar (qv is plane 0 of out_q), density and the
/// edge-normal momentum, as physics_get_tend leaves them.
pub const SLOT_DEFAULT: &str = "mpas_rk_cells_v842_f32:out_theta,mpas_rk_cells_v842_f32:out_q,mpas_rk_cells_v842_f32:out_rho,mpas_rk_edges_v842_f32:out_u";

pub(crate) fn tap_on() -> bool {
    std::env::var_os("AD_TAP").is_some() || CB.lock().map(|g| g.is_some()).unwrap_or(false)
}

static TAP_OCC: Mutex<(usize, Vec<(String, usize)>)> = Mutex::new((usize::MAX, Vec::new()));
static TAP_PASS: AtomicUsize = AtomicUsize::new(0);

/// At a reversed launch of `f`, before its adjoint entries: hand out the listed cotangent views.
pub(crate) unsafe fn tap_entry(cuda: &Cuda, step: usize, f: &str, kd: &deriv::KernelDeriv, b: &deriv::Bound,
                               li: &deriv::LaunchInfo) -> Result<(), String> {
    let file_list = taps()?;
    let cb = CB.lock().map_err(|_| "tap lock")?;
    let in_file = file_list.iter().any(|s| s.f == f);
    let in_cb = cb.as_ref().is_some_and(|c| c.list.iter().any(|s| s.f == f));
    if !in_file && !in_cb { return Ok(()); }
    let occ = {
        let mut g = TAP_OCC.lock().map_err(|_| "tap lock")?;
        if g.0 != step {
            if g.0 < step || g.0 == usize::MAX { TAP_PASS.fetch_add(1, Relaxed); }
            g.0 = step; g.1.clear();
        }
        match g.1.iter_mut().find(|x| x.0 == f) { Some(x) => { x.1 += 1; x.1 - 1 }, None => { g.1.push((f.to_owned(), 1)); 0 } }
    };
    let resolve = |arg: &str| -> Result<String, String> {
        match arg.parse::<usize>() { Ok(i) => kd.names.get(i).cloned().ok_or_else(|| format!("AD_TAP: {f} has no argument {i}")), Err(_) => Ok(arg.to_owned()) }
    };
    dev::sync(cuda)?;
    if in_file {
        let dir = std::env::var("AD_TAP_DIR").map_err(|_| "AD_TAP needs AD_TAP_DIR (where the per-step cotangents are written)".to_string())?;
        for s in file_list.iter().filter(|s| s.f == f) {
            let name = resolve(&s.arg)?;
            let (a, n) = kd.view(b, li, &name).map_err(|e| format!("AD_TAP: {f} argument {name}: {e}"))?;
            let wd = b.shadows.values().find(|x| a >= x.address && a < x.address + x.bytes as u64).map(|x| x.width).unwrap_or(0);
            if wd != 4 { return Err(format!("AD_TAP: {f} argument {name} has a {wd}-byte cotangent view; the slot is float32")); }
            let bytes = dev::download(cuda, a, n)?;
            let (mut ss, mut nz) = (0f64, 0u64);
            for c in bytes.chunks_exact(4) { let x = f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64; if x != 0.0 { nz += 1; ss += x * x; } }
            let p = Path::new(&dir).join(format!("tap-s{step}-{f}-{name}-r{occ}.f32"));
            fs::write(&p, &bytes).map_err(|e| format!("{}: {e}", p.display()))?;
            let sha: String = { use sha2::{Digest, Sha256}; Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect() };
            let target = kd.device_target(li, &name).ok().flatten();
            eprintln!("{}", json!({"event":"tb_tap","reverse":TAP_PASS.load(Relaxed),"step":step,"function":f,"argument":name,"launch_in_step_reversed":occ,
                "plan_allocation":target.map(|t| t.0),"plan_byte_offset":target.map(|t| t.1),"bytes":n,"nonzero_words":nz,"sum_sq":ss,"sha256":sha,"file":p.display().to_string()}));
        }
    }
    if let Some(c) = cb.as_ref() {
        let fc = CString::new(f).map_err(|e| e.to_string())?;
        for s in c.list.iter().filter(|s| s.f == f) {
            let name = resolve(&s.arg)?;
            let (a, n) = kd.view(b, li, &name).map_err(|e| format!("tap: {f} argument {name}: {e}"))?;
            let ac = CString::new(name.as_str()).map_err(|e| e.to_string())?;
            let rc = (c.f)(c.user as *mut c_void, step as i32, fc.as_ptr(), ac.as_ptr(), a, n as u64);
            if rc != 0 { return Err(format!("tap: the caller's function returned {rc} at step {step} {f} {name}")); }
        }
        dev::sync(cuda)?;
    }
    Ok(())
}

pub(crate) struct Kick { pub step: usize, pub f: String, pub arg: usize, pub occ: usize, pub v: Vec<u8>, pub file: String }
fn parse_kick(s: &str) -> Result<Kick, String> {
    let q: Vec<&str> = s.split(':').collect();
    let bad = || format!("AD_KICK={s}: want step:function:arg:file.f32[:occurrence]");
    if q.len() < 4 || q.len() > 5 { return Err(bad()); }
    let step = q[0].parse::<usize>().map_err(|_| bad())?;
    let arg = q[2].parse::<usize>().map_err(|_| bad())?;
    let occ = match q.get(4) { Some(x) => x.parse::<usize>().map_err(|_| bad())?, None => 0 };
    let v = fs::read(q[3]).map_err(|e| format!("AD_KICK {}: {e}", q[3]))?;
    if v.is_empty() || v.len() % 4 != 0 { return Err(format!("AD_KICK {}: {} bytes is not float32 words", q[3], v.len())); }
    Ok(Kick { step, f: q[1].to_owned(), arg, occ, v, file: q[3].to_owned() })
}
fn kick() -> Result<Option<&'static Kick>, String> {
    static K: OnceLock<Result<Option<Kick>, String>> = OnceLock::new();
    K.get_or_init(|| match std::env::var("AD_KICK") { Ok(s) => parse_kick(&s).map(Some), Err(_) => Ok(None) })
        .as_ref().map(Option::as_ref).map_err(Clone::clone)
}
static KICK_H: AtomicU64 = AtomicU64::new(0);
static KICK_SEEN: AtomicUsize = AtomicUsize::new(0);
pub(crate) static KICK_HITS: AtomicU64 = AtomicU64::new(0);
/// The size of the kick in the perturbed forward now running (0: none; the base forward and the tangent run at 0).
pub(crate) fn kick_set(h: f64) { KICK_H.store(h.to_bits(), Relaxed); }
pub(crate) fn kick_reset() { KICK_SEEN.store(0, Relaxed); }
pub(crate) fn kick_receipt() -> Value {
    match kick() { Ok(Some(k)) => json!({"step":k.step,"function":k.f,"argument":k.arg,"occurrence":k.occ,"file":k.file,"words":k.v.len() / 4,
                                         "hits":KICK_HITS.load(Relaxed)}), _ => Value::Null }
}
/// After a forward launch (no tangent): add h * v to the named argument when this is the kicked launch.
pub(crate) unsafe fn kick_hook(ex: &Execution, command: &Operation, step: usize) -> Result<(), String> {
    let h = f64::from_bits(KICK_H.load(Relaxed));
    if h == 0.0 { return Ok(()); }
    let Some(k) = kick()? else { return Ok(()) };
    if step != k.step { return Ok(()); }
    let Operation::Kernel { function, arguments, .. } = command else { return Ok(()) };
    if *function != k.f { return Ok(()); }
    let o = KICK_SEEN.fetch_add(1, Relaxed);
    if o != k.occ { return Ok(()); }
    let Some(crate::plan::Argument::Device { target }) = arguments.get(k.arg) else {
        return Err(format!("AD_KICK: argument {} of {function} is not a device array", k.arg)) };
    let a = ex.address(*target, k.v.len())?;
    dev::sync(&ex.cuda)?;
    let mut w = dev::download(&ex.cuda, a, k.v.len())?;
    let mut moved = 0u64;
    for (q, p) in w.chunks_exact_mut(4).zip(k.v.chunks_exact(4)) {
        let d = f32::from_le_bytes([p[0], p[1], p[2], p[3]]);
        if d == 0.0 { continue; }
        let x = f32::from_le_bytes([q[0], q[1], q[2], q[3]]);
        let n = (x as f64 + h * d as f64) as f32;
        if n.to_bits() != x.to_bits() { moved += 1; }
        q.copy_from_slice(&n.to_le_bytes());
    }
    dev::upload(&ex.cuda, a, &w)?;
    dev::sync(&ex.cuda)?;
    KICK_HITS.fetch_add(1, Relaxed);
    eprintln!("{}", json!({"event":"tb_kick","step":step,"function":function,"argument":k.arg,"occurrence":o,"h":h,"words":k.v.len() / 4,"words_moved":moved}));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tap_and_kick_words_parse_and_bad_ones_are_refused() {
        let t = parse_taps("mpas_rk_cells_v842_f32:out_theta, mpas_rk_edges_v842_f32:9").unwrap();
        assert_eq!((t[0].f.as_str(), t[0].arg.as_str(), t[1].arg.as_str()), ("mpas_rk_cells_v842_f32", "out_theta", "9"));
        assert!(parse_taps("nocolon").is_err());
        assert_eq!(parse_taps(SLOT_DEFAULT).unwrap().len(), 4);
        assert!(parse_kick("x:f:1:/nonexistent").is_err());
        let p = std::env::temp_dir().join(format!("tb-kick-{}.f32", std::process::id()));
        fs::write(&p, 1.5f32.to_le_bytes()).unwrap();
        let k = parse_kick(&format!("6:mpas_rk_cells_v842_f32:15:{}", p.display())).unwrap();
        assert_eq!((k.step, k.arg, k.occ, k.v.len()), (6, 15, 0, 4));
        let k = parse_kick(&format!("6:mpas_rk_cells_v842_f32:15:{}:2", p.display())).unwrap();
        assert_eq!(k.occ, 2);
        let _ = fs::remove_file(p);
    }
}
