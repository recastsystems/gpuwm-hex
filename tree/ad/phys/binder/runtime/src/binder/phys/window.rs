// SPDX-License-Identifier: Apache-2.0
//! Physics-on window, lane L1 (ad/window-L1-dispatcher, private autodiff program): the composed tangent and the
//! segment-checkpointed reverse over the sealed physics plan's command stream, inside the P1 replay runner
//! (`ad-binder window`). The runner's own `Execution::command` replays every command exactly as the strict replay
//! does (held uploads, pointer-word translation, native calls, TEMPO binds); this file only adds derivatives:
//!
//! * one float shadow arena over every plan allocation (initial and mid-step `Allocate`, keyed by plan allocation
//!   id, P8/U7 word convention), bound to the ad.entry/v1 entries as persistent shadows (deriv.rs, as U6/U9 do);
//! * the rule of every other command from wrules.rs (P8 rows / view / where / max / zero, copies, clears, held
//!   uploads, tables, native calls); commands without an exact rule are refused by name and listed;
//! * augmented forward with the tangent sweep: every step's boundary hashes are still checked against the
//!   ordinary-forward reference by the runner's finish_step;
//! * boundary store: the whole device state (every live allocation) to NVMe files plus the host state (host
//!   allocations incl. the physics frame, TEMPO cache and bindings, pointer-translation history, held pointer
//!   words, CUDA event set, the live frame) in RAM, at every step boundary;
//! * reverse: steps last to first; inside a step, segments last to first, each replayed from the step's boundary
//!   (host branches replay their recorded outcome from the flat command list), taped (pre-launch primal spans,
//!   `post` copies, masks and max inputs) and walked back. A segment whose tape does not fit is split where it
//!   overflowed. The first segment of every step replays to the step end and compares every live word with the
//!   stored boundary (restore + redo);
//! * whole-window proofs: dot <y, M x> vs <M^T y, x> (plain and aligned (M - I) forms) per seed, zero direction,
//!   two reverse runs bitwise, all on the real trajectory.
//! * `--fct-frozen 1` (default off): the FCT flux limiter's coefficients are frozen from the forward trajectory and
//!   are constants in the tangent and the adjoint (a deliberate choice): an APPROXIMATION of the
//!   forward's derivative, named in the receipt (wrules.rs `fct_frozen`, window.cu `win_fct_minmax_frozen_*`);
//! * `--taylor h1,h2,..` (relative directions only): the gradient Taylor check of J(x0) = <y, x_N> along the first
//!   seed's direction x (address-bearing allocations masked): (J(x0 + h x) - J(x0)) / (h <x, M^T y>) on the
//!   unchecked nonlinear forward, declared TEMPO residue words excluded from J.
//! * `--native-schemes 1` (with AD_SCHEMES): the L2 physics rows (P3 MYNN-EDMF, P4 MYNN surface layer and GSL drag)
//!   are live in the composed operator through the float32-shadow adapter: each launch's bound segments are widened
//!   from the window's float32 shadows into the row's float64 shadows, the row's tangent / adjoint runs in place
//!   (schemes.rs), and the segments are rounded back. Without the flag these launches stay held.
//! * `--objective FILE` (JSON {"name", "terms": [[allocation, word, weight], ..]}): one more reverse from y = the
//!   objective's weights, J(x0) = sum weight * x_N[allocation, word]; `--dump-grad id,id,..` writes that reverse's
//!   gradient over each listed allocation (raw float32, grad-<id>.f32); with `--taylor` the Taylor check uses this J.
//! * `--tempo-small tempo_radar` (with AD_DOT_TEMPO_SERIAL_CUBIN, ad/held-tempo_radar): tempo_radar is live in the
//!   composed operator through its P5 per-thread entry (tempo_adw::TempoSmall, float64 tape): inputs snapshotted
//!   into the tape arena before the forward, the tape re-recorded from that snapshot in the tangent and in the
//!   reverse, input tangents widened from the float32 shadows and outputs rounded back (adjoint: outputs cleared,
//!   inputs accumulated in float64, one rounding). The max / floor selections (-35 dBZ floor, composite max,
//!   hourly max) and the 1 km level index come from the trajectory (the tape records the executed branch):
//!   frozen coefficients. Without the flag tempo_radar keeps its inactive rule (output diagnostic).
//! * `--taylor-dir FILE` (JSON {"name", "terms": [[allocation, word, value], ..]}, default off): the Taylor check runs
//!   along this fixed direction instead of seed 1's relative one (lane taylor-repose: a smooth single-field bump);
//!   primal[allocation, word] += h * value, and <x, dJ/dx0> = sum value * gradient[allocation, word].
//! * `--objective a.json,b.json,..`: the first file is the gradient objective; every file's dJ and forward difference
//!   quotient dJ/h is reported on each Taylor row.
//! * `--fd-only 1` (with --taylor-dir, --objective, --taylor): forward only. The base forward runs without the tangent,
//!   no reverse runs, and each Taylor row reports the forward's own difference quotient dJ/h per objective.
//! * `--regime FILE` (JSON {"fields": [{"name", "allocation", "offset", "count", "dtype": "i4"|"f4", "test":
//!   "eq"|"sign"|"pos"}]}): regime words captured at every step boundary of the base forward and compared with the
//!   perturbed forward of each Taylor row (eq: bits, sign: -1/0/+1 class, pos: > 0); changed words per field and step.
//! * `--dump-state id[:byte_offset:bytes],..` (lane runner-dump, default off): the window-end state of the listed
//!   allocations, raw float32: the base forward's (state-<id>.f32, from the stored window-end boundary), the
//!   tangent's (tl-<id>.f32, M x at the window end, not with --fd-only) and every Taylor row's perturbed forward
//!   (state-h<row>-<id>.f32) with its difference field fd-<id>-h<h>.f32 = (x_N(h) - x_N(0)) / h (declared residue
//!   words 0), scored against the tangent field per allocation. A sub-range id:off:bytes adds -<off> to each name.
//!   Reads only: the forward is unchanged. File names match j4-repose's interim flag.
//! * `--tangent-taylor-dir 1` (alias --tangent-dir; with --taylor-dir and --tangent-only 1, default off): the base forward's tangent starts from
//!   the --taylor-dir terms instead of seed 1's direction, so tl-<id>.f32 is the prediction the Taylor rows' difference
//!   fields are compared with, and <w, M x> (w = the first objective) is reported beside <x, M^T w>.
//! * Kernel-set admission: every row's recorded derivative spec sha256 is checked against the spec file (all cores)
//!   before any image loads, and the window refuses on a mismatch, as the e2e bind tool does.
//! CUDA does the field arithmetic; Rust schedules, checkpoints and scores. Without `window` nothing changes.
use crate::cuda::{checked, Cuda};
use crate::plan::{Operation, Plan};
use crate::wrules::{self, Ctx, DevRow, R};
use crate::{deriv, dev, replay_frame::ReplayFrame, rk_schedule, submit, tempo, host_state, Execution, PointerHistory, References};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::{ffi::c_void, fs, ptr, time::Instant};

/// anchors-ahead: page-locked anchor buffers made on a helper thread (the buffers made so far, whether the helper is done).
type AheadPool = std::sync::Arc<(std::sync::Mutex<(Vec<crate::pinned::PinnedBuf>, bool)>, std::sync::Condvar)>;
fn mem_available() -> usize {
    fs::read_to_string("/proc/meminfo").ok().and_then(|m| m.lines().find(|l| l.starts_with("MemAvailable"))
        .and_then(|l| l.split_whitespace().nth(1)?.parse::<usize>().ok())).unwrap_or(0) * 1024
}

/// An optional extension driver over this physics window (`--ext-driver`, default off), compiled only
/// where its sources are present (build.rs sets cfg hexad_ext_driver).
#[cfg(hexad_ext_driver)]
#[path = "ext_driver.rs"] mod ext_driver;
/// The libhexad-phys C ABI v2 over this window (cdylib hexad-phys; the
/// ad-binder binary carries it but never calls it). Its counters' category guards below are thread-local writes only.
#[path = "capi.rs"] pub mod capi;

const P8_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/p8_rules.fatbin"));
const WIN_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/window_support.fatbin"));
pub const DOT_F32: f64 = 2e-5;
/// Largest |primal| a relative direction weights (base pressure is ~1e5 Pa). The 1-step trace (trace1) showed the
/// tangent norm grow from 1.8e26 to 2.6e35 in edge_fields_from_saved_u with a 1e12 cap: stale scratch words of up to
/// 1e12 at boundary 0 got weights that large and their kinetic-energy tangents (2 u du) squared them.
const WEIGHT_MAX: f32 = 2e5;

macro_rules! kargs { ($($x:ident),*) => { &mut [$((&mut $x) as *mut _ as *mut c_void),*] } }

struct Fns {
    rows_tl: *mut c_void, gather_ad: *mut c_void, rows_ad_direct: *mut c_void, clear_dst: *mut c_void,
    view_tl: *mut c_void, view_ad: *mut c_void, where_tl: *mut c_void, where_ad: *mut c_void,
    max_tl: *mut c_void, max_ad: *mut c_void, view_zero: *mut c_void, fill: *mut c_void, dot2: *mut c_void, nonfinite: *mut c_void,
    fct_minmax_tl: *mut c_void, fct_minmax_ad: *mut c_void, axpy: *mut c_void, widen: *mut c_void, narrow: *mut c_void,
}
unsafe fn module(cuda: &Cuda, image: &[u8], what: &str) -> Result<*mut c_void, String> {
    if image.is_empty() { return Err(format!("{what} image is empty: rebuild the runner with NVCC set")); }
    let mut m = ptr::null_mut();
    checked(&format!("cuModuleLoadData {what}"), (cuda.module_load)(&mut m, image.as_ptr().cast()))?;
    Ok(m)
}
unsafe fn function(cuda: &Cuda, m: *mut c_void, name: &str) -> Result<*mut c_void, String> {
    let c = std::ffi::CString::new(name).unwrap();
    let mut f = ptr::null_mut();
    checked(&format!("cuModuleGetFunction {name}"), (cuda.module_function)(&mut f, m, c.as_ptr()))?;
    Ok(f)
}
unsafe fn launch(cuda: &Cuda, f: *mut c_void, grid: [u32; 3], args: &mut [*mut c_void]) -> Result<(), String> {
    checked("window launch", (cuda.launch)(f, grid[0], grid[1], grid[2], 256, 1, 1, 0, ptr::null_mut(), args.as_mut_ptr(), ptr::null_mut()))
}
fn grid1(n: i64) -> [u32; 3] { [((n.max(1) + 255) / 256).min(i32::MAX as i64) as u32, 1, 1] }
fn grid_rows(rows: &[DevRow]) -> [u32; 3] {
    let maxc = rows.iter().map(|r| r.count).max().unwrap_or(0);
    [((maxc + 255) / 256).clamp(1, 1024) as u32, rows.len() as u32, 1]
}

/// One executed leaf command of a step (host branches already resolved), with its RK scope and a tape estimate.
pub struct Flat { op: *const Operation, rk: Option<rk_schedule::Scope>, est: usize }

/// Host side of a checkpoint.
#[derive(Clone)]
struct HostSnap {
    host_state: host_state::HostState, tempo_cache: tempo::Cache, tempo_bindings: HashMap<String, (crate::plan::Slice, usize)>,
    pointer_history: crate::HistSnap, held_pointer_writes: HashMap<u64, PointerHistory>, events: Vec<u64>,
    frame: Option<ReplayFrame>, rk: Option<rk_schedule::Scope>,
}
fn host_snap(ex: &Execution, frame: &Option<ReplayFrame>) -> HostSnap {
    // per-step-host-hash: a boundary stored after a skipped digest would keep a stale pointer history (capi catches the panic).
    assert!(!ex.history_stale, "DEFECT boundary stored after a skipped unchecked digest with no restore between (per-step-host-hash)");
    HostSnap { host_state: ex.host_state.clone(), tempo_cache: ex.tempo_cache.clone(), tempo_bindings: ex.tempo_bindings.clone(),
               pointer_history: ex.history_snap(), held_pointer_writes: ex.held_pointer_writes.clone(),
               events: ex.events.keys().copied().collect(), frame: frame.clone(), rk: ex.rk.clone() }
}
unsafe fn restore_host(ex: &mut Execution, frame: &mut Option<ReplayFrame>, h: &HostSnap) -> Result<(), String> {
    // terrain-w-hash: a deferred boundary digest is joined (compared, its history filled) before any restore.
    ex.digest_flush()?;
    ex.host_state = h.host_state.clone(); ex.tempo_cache = h.tempo_cache.clone(); ex.tempo_bindings = h.tempo_bindings.clone();
    ex.pointer_history = h.pointer_history.get().clone(); ex.held_pointer_writes = h.held_pointer_writes.clone();
    ex.history_stale = false;
    *frame = h.frame.clone(); ex.rk = h.rk.clone();
    let live: Vec<u64> = ex.events.keys().copied().collect();
    for id in live { if !h.events.contains(&id) {
        if let Some(e) = ex.events.remove(&id) { checked("cuEventDestroy restore", (ex.cuda.event_destroy)(e))?; }
    } }
    for id in &h.events { if !ex.events.contains_key(id) {
        let mut e = ptr::null_mut();
        checked("cuEventCreate restore", (ex.cuda.event_create)(&mut e, crate::event_flags(0)))?;
        ex.events.insert(*id, e);
    } }
    Ok(())
}
unsafe fn drain_pinned(ex: &mut Execution) -> Result<(), String> {
    dev::sync(&ex.cuda)?;
    ex.release_pinned()?;
    Ok(())
}
/// G5: blobs (Upload / HeldUploadRanges data) of a block tree, both arms of every host branch.
pub(crate) fn block_blobs<'a>(plan: &'a Plan, name: &str, seen: &mut std::collections::HashSet<String>, out: &mut Vec<&'a crate::plan::Blob>) {
    if !seen.insert(name.to_owned()) { return; }
    let Some(commands) = plan.blocks.get(name) else { return };
    for c in commands {
        match c {
            Operation::Upload { data, .. } | Operation::HeldUploadRanges { data, .. } => out.push(data),
            Operation::HostBranch { if_zero, if_nonzero, .. } => { block_blobs(plan, if_zero, seen, out); block_blobs(plan, if_nonzero, seen, out); },
            Operation::RkSchedule { body, .. } | Operation::RkSplits { body } | Operation::RkStages { body } | Operation::RkSubsteps { body } =>
                block_blobs(plan, body, seen, out),
            _ => {},
        }
    }
}
fn flat_blobs(flat: &[Flat]) -> Vec<crate::plan::Blob> {
    flat.iter().filter_map(|f| match unsafe { &*f.op } {
        Operation::Upload { data, .. } | Operation::HeldUploadRanges { data, .. } => Some(data.clone()), _ => None }).collect()
}
/// G5: decode blobs on all cores (one thread per blob, at most the core count); each is `crate::blob` (read, gunzip,
/// SHA-256 check), so the bytes are exactly what Execution::command would decode.
pub(crate) fn decode_blobs(base: &Path, want: &[crate::plan::Blob]) -> Result<Vec<(String, Vec<u8>)>, String> {
    let mut todo: Vec<&crate::plan::Blob> = Vec::new();
    for b in want { if !todo.iter().any(|t| crate::blob_sha(t) == crate::blob_sha(b)) { todo.push(b); } }
    if todo.is_empty() { return Ok(Vec::new()); }
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).clamp(1, 32).min(todo.len());
    let out: Vec<Result<Vec<(String, Vec<u8>)>, String>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads).map(|t| { let todo = &todo; sc.spawn(move || {
            let mut v = Vec::new();
            for b in todo.iter().skip(t).step_by(threads) {
                let t0 = std::time::Instant::now();
                let d = crate::blob(base, b)?;
                if std::env::var("AD_DIGEST_PROF").as_deref() == Ok("1") {
                    eprintln!("{}", serde_json::json!({"event":"blob_decode_prof","kind":match b { crate::plan::Blob::Hex { .. } => "hex", crate::plan::Blob::File { .. } => "file", crate::plan::Blob::Gzip { .. } => "gzip" },
                        "bytes":d.len(),"s":t0.elapsed().as_secs_f64(),"thread":t,"threads":threads}));
                }
                v.push((crate::blob_sha(b).to_owned(), d));
            }
            Ok(v) }) }).collect();
        hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("blob decoder panicked".into()))).collect()
    });
    let mut all = Vec::new();
    for r in out { all.extend(r?); }
    Ok(all)
}
/// S13: one LW group section replayed once per pass. The wrapper (rrtmg_lw_v842.py _layer_groups and the batch
/// loop) gathers each batch's columns through the stable-sorted column order at that batch's start, runs them at
/// the batch's layer count, and scatters them back. Per-column words do not depend on the batch or its other
/// columns (independent KISS state per column, column-independent solver), so a live grouping the plan did not
/// record is reproduced by rewriting the order array and replaying the recorded batches: every live column of
/// layer count L is placed in a batch recorded at L, in as many passes as the live group needs (repeats pad a
/// batch). Proved on a test box with AD_LW_REGROUP_TEST: mode 2 (a rotated pass, then the recorded layout) keeps every
/// reference step hash (S13-T2), mode 3 (recorded layout, then a rotated pass) gives 0 of 632,814 LW output words
/// different (S13-T3).
/// `events`: the live events at the section start (id, recorded create flags). Before each later pass the event set
/// is put back: events the section created are destroyed, events it destroyed are created again (THE BREAKAGES:
/// pass 2 of the i7-FtA h 0.03 row refused "event 1470151824 created while live" at step 7, and the test leg's
/// pass 2 refused "unknown event 1462729056" at step 1, an event made before the section and destroyed in it).
/// A section Allocate runs in pass 1 only: later passes keep the block, so earlier passes' scatters stay (step 1's
/// section allocates its pool blocks; the guard refused it in S13-T).
/// `outputs` (test mode 3 only): the scatter destinations (address, bytes) and their words after pass 1.
struct LwRegroup { start: usize, end: usize, address: u64, passes: Vec<Vec<i32>>, pass: usize, original: Vec<u8>, events: Vec<(u64, u32)>,
                   test: u8, outputs: Vec<(u64, usize, Vec<u8>)> }

fn lw_order_slices(op: &Operation) -> Vec<crate::plan::Slice> {
    use crate::plan::Argument as A;
    match op {
        Operation::Kernel { arguments, .. } => arguments.iter().flat_map(|a| match a {
            A::Device { target } => vec![*target],
            A::CupyArray { target: Some(t), .. } => vec![*t],
            A::Patched { pointers, .. } => pointers.iter().map(|q| q.target).collect(),
            _ => Vec::new(),
        }).collect(),
        Operation::DeviceToHost { source, .. } => vec![*source],
        _ => Vec::new(),
    }
}

unsafe fn lw_regroup_begin(ex: &mut Execution, commands: &[Operation], at: usize, step: usize, hist: &[i32]) -> Result<LwRegroup, String> {
    use crate::plan::Argument as A;
    let e = |m: String| format!("step {step}: LW regroup: {m}");
    let lg = match &commands[at] { Operation::DeviceToHost { layer_groups: Some(lg), .. } => lg.clone(), _ => return Err(e("not a LW histogram read".into())) };
    // _layer_groups: order = cp.argsort(counts, kind='stable').astype(cp.int32), then sizes = hist.get().
    let (Some(Operation::NativeThrustArgsort { indices, .. }), Some(Operation::Kernel { function, arguments, .. })) =
        (at.checked_sub(2).and_then(|k| commands.get(k)), at.checked_sub(1).and_then(|k| commands.get(k))) else {
        return Err(e("the read does not follow the stable argsort and its int32 cast".into()));
    };
    let ptr0 = |a: &A| match a { A::Patched { pointers, .. } if pointers.len() == 1 && pointers[0].byte_offset == 0 => Some(pointers[0].target), _ => None };
    let (Some(src), Some(order)) = (arguments.first().and_then(ptr0), arguments.get(1).and_then(ptr0)) else { return Err(e("int32 cast arguments".into())) };
    if function != "cupy_copy__int64_int32" || src != *indices { return Err(e(format!("the cast {function} does not read the argsort indices"))); }
    let (nc, bytes) = (lg.nc, 4 * lg.nc);
    let address = ex.address(order, bytes)?;
    let in_order = |t: &crate::plan::Slice| t.allocation == order.allocation && t.offset >= order.offset && t.offset < order.offset + bytes;
    let mut starts = Vec::with_capacity(lg.batches.len());
    let mut first = 0usize;
    for b in &lg.batches { starts.push(order.offset + 4 * first); first += usize::try_from(b[1]).map_err(|_| e("negative batch".into()))?; }
    if first != nc { return Err(e(format!("recorded batches cover {first} of {nc} columns"))); }
    // The section: through the last batch's mpas_lw_outputs and the scatters after it that read the order at that
    // batch's start. The pool reuses the order block after the section, so the end is found by structure.
    let (mut prepares, mut outputs, mut last_out) = (0usize, 0usize, None);
    for (k, op) in commands.iter().enumerate().skip(at + 1) {
        match op {
            Operation::Kernel { function, .. } if function == "mpas_lw_prepare" => prepares += 1,
            Operation::Kernel { function, .. } if function == "mpas_lw_outputs" => { outputs += 1; if outputs == lg.batches.len() { last_out = Some(k); break; } },
            Operation::DeviceToHost { layer_groups: Some(_), .. } => break,
            _ => {},
        }
    }
    let last_out = last_out.ok_or_else(|| e(format!("{outputs} mpas_lw_outputs launches for {} batches", lg.batches.len())))?;
    if prepares != lg.batches.len() { return Err(e(format!("{prepares} mpas_lw_prepare launches for {} batches", lg.batches.len()))); }
    let last_start = *starts.last().unwrap();
    let mut end = last_out;
    while let Some(op) = commands.get(end + 1) {
        if lw_order_slices(op).iter().any(|t| in_order(t) && t.offset == last_start) { end += 1; } else { break; }
    }
    if end == last_out { return Err(e("no scatter after the last batch".into())); }
    let mut seen = vec![false; starts.len()];
    for op in &commands[at + 1..=end] {
        match op {
            Operation::HostBranch { .. } | Operation::RkSchedule { .. } | Operation::RkSplits { .. } | Operation::RkStages { .. }
            | Operation::RkSubsteps { .. } => return Err(e("control flow inside the LW group section".into())),
            Operation::Clear { destination: d, bytes: n, .. } | Operation::DeviceCopy { destination: d, bytes: n, .. }
                if d.allocation == order.allocation && d.offset < order.offset + bytes && order.offset < d.offset + n =>
                return Err(e("the LW group section writes the column order".into())),
            _ => {},
        }
        for t in lw_order_slices(op).iter().filter(|t| in_order(t)) {
            match starts.iter().position(|&s0| s0 == t.offset) { Some(b) => seen[b] = true,
                None => return Err(e(format!("order view at column {} is not a batch start", (t.offset - order.offset) / 4))) }
        }
    }
    if seen.iter().any(|v| !v) { return Err(e("a recorded batch does not read the order".into())); }
    let created_flags = |id: u64| commands[..=end].iter().rev().find_map(|op| match op {
        Operation::Synchronization { operation, event_handle_recorded, flags, .. } if operation == "EventCreate" && *event_handle_recorded == id => Some(*flags),
        _ => None }).unwrap_or(0);
    let mut events: Vec<(u64, u32)> = ex.events.keys().map(|&id| (id, created_flags(id))).collect();
    events.sort_unstable();
    dev::sync(&ex.cuda)?;
    let original = dev::download(&ex.cuda, address, bytes)?;
    let live: Vec<i32> = original.chunks_exact(4).map(|w| i32::from_le_bytes(w.try_into().unwrap())).collect();
    let passes = crate::layer_groups::regroup_passes(nc, lg.nm, hist, &live, &lg.batches, ex.lw_regroup_test).map_err(e)?;
    let groups: Vec<[i64; 2]> = hist[..crate::layer_groups::HISTOGRAM_BINS].iter().enumerate().filter(|(_, &n)| n > 0)
        .map(|(b, &n)| [i64::from(lg.nm) + 1 + b as i64, i64::from(n)]).collect();
    eprintln!("{}", json!({"event":"lw_regroup","step":step,"live_groups":groups,"recorded_batches":lg.batches,
                           "passes":passes.len(),"section_commands":end - at,"test":ex.lw_regroup_test}));
    let bytes0: Vec<u8> = passes[0].iter().flat_map(|v| v.to_le_bytes()).collect();
    dev::upload(&ex.cuda, address, &bytes0)?;
    dev::sync(&ex.cuda)?;
    // test mode 3: the wrapper's scatter destinations (cupy_scatter_update argument 6, a 1-D CArray: ptr, size, shape, stride)
    let mut outputs: Vec<(u64, usize, Vec<u8>)> = Vec::new();
    if ex.lw_regroup_test == 3 {
        for op in &commands[at + 1..=end] {
            if let Operation::Kernel { function, arguments, .. } = op {
                if function != "cupy_scatter_update" { continue; }
                if let Some(A::Patched { bytes_hex, pointers, .. }) = arguments.get(5) {
                    let raw: Vec<u8> = (0..bytes_hex.len() / 2).map(|i| u8::from_str_radix(&bytes_hex[2 * i..2 * i + 2], 16).unwrap_or(0)).collect();
                    if raw.len() != 32 || pointers.len() != 1 { continue; }
                    let size = u64::from_le_bytes(raw[8..16].try_into().unwrap()) as usize;
                    let stride = u64::from_le_bytes(raw[24..32].try_into().unwrap()) as usize;
                    let a = ex.address(pointers[0].target, size * stride)?;
                    if !outputs.iter().any(|o| o.0 == a) { outputs.push((a, size * stride, Vec::new())); }
                }
            }
        }
    }
    Ok(LwRegroup { start: at, end, address, passes, pass: 0, original, events, test: ex.lw_regroup_test, outputs })
}

/// S13: after the section's last command: start the next pass (true), or restore the live order (false).
unsafe fn lw_regroup_next(ex: &mut Execution, r: &mut LwRegroup, step: usize) -> Result<bool, String> {
    dev::sync(&ex.cuda)?;
    if r.test == 3 && r.pass == 0 {
        for o in r.outputs.iter_mut() { o.2 = dev::download(&ex.cuda, o.0, o.1)?; }
    } else if r.test == 3 && r.pass + 1 == r.passes.len() {
        let (mut words, mut differ) = (0usize, 0usize);
        let mut per = Vec::new();
        for o in &r.outputs {
            let now = dev::download(&ex.cuda, o.0, o.1)?;
            let n = now.chunks_exact(4).zip(o.2.chunks_exact(4)).filter(|(a, b)| a != b).count();
            words += o.1 / 4; differ += n; per.push(json!([o.1 / 4, n]));
            dev::upload(&ex.cuda, o.0, &o.2)?;
        }
        eprintln!("{}", json!({"event":"lw_regroup_outputs","step":step,"arrays":r.outputs.len(),"words":words,"differ":differ,"per_array":per}));
    }
    r.pass += 1;
    if r.pass < r.passes.len() {
        let extra: Vec<u64> = ex.events.keys().copied().filter(|id| !r.events.iter().any(|(e, _)| e == id)).collect();
        for id in extra {
            if let Some(h) = ex.events.remove(&id) { checked("cuEventDestroy LW regroup", (ex.cuda.event_destroy)(h))?; }
        }
        for &(id, flags) in &r.events {
            if !ex.events.contains_key(&id) {
                let mut h = ptr::null_mut();
                checked("cuEventCreate LW regroup", (ex.cuda.event_create)(&mut h, crate::event_flags(flags)))?;
                ex.events.insert(id, h);
            }
        }
        let b: Vec<u8> = r.passes[r.pass].iter().flat_map(|v| v.to_le_bytes()).collect();
        dev::upload(&ex.cuda, r.address, &b)?;
    } else {
        dev::upload(&ex.cuda, r.address, &r.original)?;
    }
    dev::sync(&ex.cuda)?;
    Ok(r.pass < r.passes.len())
}

/// Forward of one command. A mid-step Allocate whose id is already live (a replay from a restored checkpoint;
/// allocations are never freed in this plan) keeps the same storage: its contents are overwritten before read.
/// host-leftovers: take a device sync at the end of every derivative entry (the old behaviour) under AD_G5_PROF=1 or
/// AD_ENTRY_SYNC=1; otherwise the sync is owed (Win::sync_owed).
/// x20 replay-compiled: true when every command of the plan runs on stream 0, the legacy null stream (the plan
/// declares no streams of its own), so the runner's support kernels, tape copies, the forward command and the
/// derivative entries already execute in issue order on one in-order queue. Then a cuCtxSynchronize whose only job is
/// to order device work (pre-launch copies before the forward that overwrites their source, an allocation's zero fill
/// before its first use, an entry before the next rule's kernels) is a host drain with no effect on any word: it only
/// stops the host from issuing the next command while the card works. THE BREAKAGE THIS FIXES: those drains (one or
/// two per derivative entry in the TL step and in the AD taping forward, one per replayed allocation, one before every
/// non-entry rule) kept the host from running ahead of the queue, so every launch paid the host's per-command work
/// with the card idle. Syncs stay where the host reads device results, before native, TEMPO and TEMPO-serial rows
/// (their libraries may run their own streams), and at every step and pass end. AD_ORDER_SYNC=1 keeps every drain.
pub(crate) fn ordered(ex: &Execution) -> bool {
    static FORCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *FORCE.get_or_init(|| std::env::var("AD_ORDER_SYNC").as_deref() == Ok("1") || entry_sync()) { return false; }
    ex.streams.len() == 1 && ex.streams.get(&0).is_some_and(|s| s.is_null())
}
fn entry_sync() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| dev::prof::on() || std::env::var("AD_ENTRY_SYNC").as_deref() == Ok("1"))
}

unsafe fn exec(ex: &mut Execution, op: &Operation, step: usize, base: &Path, frame: &mut Option<ReplayFrame>) -> Result<(), String> {
    let _c = capi::xfer::Cat::forward();
    // G2 (P1 milestone 2): an allocation made inside a step starts from zeros in every pass of the window (first pass,
    // anchors, redo, segments), so a pass never reads what an earlier pass left in it. THE BREAKAGE THIS PREVENTS: with
    // radiation live the LW solver (rlw_rtrn_march_coalesced / rlw_rtrn_accum_rows) reads words of its pool block 441
    // before writing them; the redo of step 1 found pass 1's words there, not the fresh block's, and the window refused
    // (restore + redo differs in 25 words). The forward-hash check of the first pass shows the zeros are the forward's.
    if let Operation::Allocate { id, bytes, stream, .. } = op {
        if let Some(a) = ex.allocations.get(id) {
            if a.bytes != *bytes { return Err(format!("allocation {id} replayed with a different size")); }
            let clear = Operation::Clear { stream: *stream, destination: crate::plan::Slice { allocation: *id, offset: 0 }, bytes: *bytes, value: 0 };
            ex.command(&clear, step, base, frame)?;
        } else {
            // F11: a new allocation is born zero in Execution::command itself (main.rs), so it is not cleared twice.
            ex.command(op, step, base, frame)?;
        }
        return if ordered(ex) { Ok(()) } else { dev::sync(&ex.cuda) };
    }
    let t = dev::prof::start();
    let r = ex.command(op, step, base, frame);
    dev::prof::add(3, t);
    r
}

/// adjoint-speed-2: dst <- src on all cores (64 MiB pieces).
fn par_copy(dst: &mut [u8], src: &[u8]) {
    let piece = 64usize << 20;
    let threads = s02_threads();
    let parts: Vec<(&mut [u8], &[u8])> = dst.chunks_mut(piece).zip(src.chunks(piece)).collect();
    let per = parts.len().div_ceil(threads.max(1)).max(1);
    std::thread::scope(|sc| {
        let mut parts = parts;
        while !parts.is_empty() {
            let rest = parts.split_off(parts.len().min(per));
            let mine = std::mem::replace(&mut parts, rest);
            sc.spawn(move || for (d, s) in mine { d.copy_from_slice(s); });
        }
    });
}
/// adjoint-speed-2: the redo check of `differing` (same counts, same window_redo_differs lines) over a downloaded state
/// `live` laid out as the stored boundary `want` (index entries: allocation, byte offset, bytes).
fn compare_boundary(index: &[(u64, usize, usize)], live: &[u8], want_all: &[u8], ranges: &[Vec<(usize, usize)>], step: usize) -> (u64, u64) {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16);
    let (mut d, mut r) = (0u64, 0u64);
    for ((id, off, n), ranges) in index.iter().zip(ranges) {
        let buf = &live[*off..*off + *n];
        let want = &want_all[*off..*off + *n];
        let count = |lo: usize, x: &[u8], y: &[u8]| -> (u64, u64) {
            let (mut d, mut r) = (0u64, 0u64);
            for (w, (p, q)) in x.chunks(4).zip(y.chunks(4)).enumerate() {
                if p == q { continue; }
                let at = lo + 4 * w;
                if ranges.iter().any(|&(o, l)| at >= o && at < o + l) { r += 1 } else { d += 1 }
            }
            (d, r)
        };
        let c = if *n < (16 << 20) { count(0, buf, want) } else {
            let chunk = (*n / 4).div_ceil(threads) * 4;
            std::thread::scope(|sc| {
                let hs: Vec<_> = buf.chunks(chunk).zip(want.chunks(chunk)).enumerate()
                    .map(|(k, (x, y))| { let count = &count; sc.spawn(move || count(k * chunk, x, y)) }).collect();
                hs.into_iter().map(|h| h.join().unwrap()).fold((0, 0), |s, c| (s.0 + c.0, s.1 + c.1))
            })
        };
        if c.0 > 0 {
            let first = buf.chunks(4).zip(want.chunks(4)).enumerate().find(|(w, (p, q))| p != q
                && !ranges.iter().any(|&(o, l)| 4 * w >= o && 4 * w < o + l)).map(|(w, _)| 4 * w);
            eprintln!("{}", json!({"event":"window_redo_differs","step":step,"allocation":id,"words":c.0,"first_byte":first,"bytes":n}));
        }
        d += c.0; r += c.1;
    }
    (d, r)
}

/// A whole-device checkpoint at a step boundary: bytes on NVMe, host side in RAM. `delta` (S02, --checkpoints): the
/// file holds only the chunks that differ from boundary 0 (delta_encode); `file_bytes` is its size on disk.
#[derive(Clone)]
struct Boundary { path: PathBuf, index: Vec<(u64, usize, usize)>, host: HostSnap, step: usize, delta: bool, file_bytes: usize,
                 /// runner-lib: the bytes in pinned host RAM instead of the file at `path` (None: the file)
                 ram: Option<std::sync::Arc<capi::HostBytes>>,
                 /// runner-lib: a reverse-step anchor held on the device at this address (0: host bytes)
                 dev: u64 }

/// S02 (ad/s02-checkpoints): changed-chunk boundary files. A boundary is stored as the 512-byte chunks that differ from
/// its reference: boundary 0's bytes for an allocation that was live there with the same size, zeros for an allocation
/// made later (every mid-step Allocate starts from zeros, see exec). Decoding rebuilds exactly the bytes a whole-arena
/// file holds, so restore writes the same words: the store is lossless, only smaller.
const DCHUNK: usize = 512;
const DMAGIC: &[u8; 8] = b"ADBDLT01";
/// Zero reference of one chunk (slice equality is a memcmp; a byte loop here cost about 2.7 s per 3.4 GB store, s02-c4b).
static ZERO_CHUNK: [u8; DCHUNK] = [0; DCHUNK];
fn s02_threads() -> usize { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).clamp(1, 32) }
/// Work items (data offset, reference offset or None for zeros, bytes); large allocations are cut into 64 MiB pieces.
fn delta_items(index: &[(u64, usize, usize)], base_index: &[(u64, usize, usize)]) -> Vec<(usize, Option<usize>, usize)> {
    let m: HashMap<u64, (usize, usize)> = base_index.iter().map(|&(id, o, n)| (id, (o, n))).collect();
    let piece = 64usize << 20;
    let mut v = Vec::new();
    for &(id, off, n) in index {
        let r = m.get(&id).filter(|x| x.1 == n).map(|x| x.0);
        let mut at = 0;
        while at < n { let len = (n - at).min(piece); v.push((off + at, r.map(|o| o + at), len)); at += len; }
    }
    v
}
/// Changed-chunk encoding of `data` (laid out by `index`) against boundary 0 (`base_index`, `base`), on all cores:
/// magic, total bytes, run count, (offset, bytes) per run, then the runs' bytes.
pub(crate) fn delta_encode(index: &[(u64, usize, usize)], data: &[u8], base_index: &[(u64, usize, usize)], base: &[u8]) -> Vec<u8> {
    let items = delta_items(index, base_index);
    let t = s02_threads().min(items.len().max(1));
    let parts: Vec<Vec<(usize, usize)>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..t).map(|k| { let items = &items; sc.spawn(move || {
            let mut runs: Vec<(usize, usize)> = Vec::new();
            for &(off, r, len) in items.iter().skip(k).step_by(t) {
                let x = &data[off..off + len];
                let mut c = 0;
                while c < len {
                    let e = (c + DCHUNK).min(len);
                    let same = match r { Some(ro) => x[c..e] == base[ro + c..ro + e], None => x[c..e] == ZERO_CHUNK[..e - c] };
                    if !same {
                        match runs.last_mut() { Some(l) if l.0 + l.1 == off + c => l.1 += e - c, _ => runs.push((off + c, e - c)) }
                    }
                    c = e;
                }
            }
            runs }) }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut runs: Vec<(usize, usize)> = parts.into_iter().flatten().collect();
    runs.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
    for r in runs { match merged.last_mut() { Some(l) if l.0 + l.1 == r.0 => l.1 += r.1, _ => merged.push(r) } }
    let payload: usize = merged.iter().map(|r| r.1).sum();
    let mut out = Vec::with_capacity(24 + 16 * merged.len() + payload);
    out.extend_from_slice(DMAGIC);
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    out.extend_from_slice(&(merged.len() as u64).to_le_bytes());
    for &(o, l) in &merged { out.extend_from_slice(&(o as u64).to_le_bytes()); out.extend_from_slice(&(l as u64).to_le_bytes()); }
    for &(o, l) in &merged { out.extend_from_slice(&data[o..o + l]); }
    out
}
/// Inverse of delta_encode: boundary 0's bytes (zeros where none), then the stored runs. Reference copies on all cores.
pub(crate) fn delta_decode(file: &[u8], index: &[(u64, usize, usize)], base_index: &[(u64, usize, usize)], base: &[u8]) -> Result<Vec<u8>, String> {
    let u = |at: usize| -> Result<usize, String> {
        file.get(at..at + 8).map(|b| u64::from_le_bytes(b.try_into().unwrap()) as usize).ok_or_else(|| "changed-chunk boundary file is truncated".to_owned())
    };
    if file.get(..8) != Some(&DMAGIC[..]) { return Err("not a changed-chunk boundary file".into()); }
    let (total, nruns) = (u(8)?, u(16)?);
    let want: usize = index.iter().map(|x| x.2).sum();
    if total != want { return Err(format!("changed-chunk boundary holds {total} bytes, its index {want}")); }
    let mut full = vec![0u8; total];
    {
        let items = delta_items(index, base_index);
        let mut pieces: Vec<(&mut [u8], usize)> = Vec::new();
        let (mut rest, mut pos): (&mut [u8], usize) = (&mut full[..], 0);
        for &(off, r, len) in &items {
            let tail = std::mem::take(&mut rest);
            let (_, t1) = tail.split_at_mut(off - pos);
            let (piece, t2) = t1.split_at_mut(len);
            rest = t2; pos = off + len;
            if let Some(ro) = r { pieces.push((piece, ro)); }
        }
        let t = s02_threads().min(pieces.len().max(1));
        let mut groups: Vec<Vec<(&mut [u8], usize)>> = (0..t).map(|_| Vec::new()).collect();
        for (i, pc) in pieces.into_iter().enumerate() { groups[i % t].push(pc); }
        std::thread::scope(|sc| { for g in groups { sc.spawn(move || { for (d, ro) in g { let n = d.len(); d.copy_from_slice(&base[ro..ro + n]); } }); } });
    }
    let mut at = 24 + 16 * nruns;
    for i in 0..nruns {
        let (o, l) = (u(24 + 16 * i)?, u(32 + 16 * i)?);
        if o + l > total || at + l > file.len() { return Err("changed-chunk boundary run outside the state".into()); }
        full[o..o + l].copy_from_slice(&file[at..at + l]);
        at += l;
    }
    if at != file.len() { return Err("changed-chunk boundary file has trailing bytes".into()); }
    Ok(full)
}

/// S02: binomial (revolve) step checkpointing. cost[n][k] is the fewest forward steps to recompute while reversing n
/// steps from a stored start with k free checkpoint slots: one step needs none; with no slot every step s is reached
/// again from the start (n (n - 1) / 2); otherwise advance m steps, store there, reverse the right part with k - 1 slots,
/// free the slot, reverse the left part with k (best[n][k] = that m). Griewank's binomial bound is the same optimum.
pub(crate) struct Revolve { best: Vec<Vec<usize>>, pub cost: Vec<Vec<u64>> }
impl Revolve {
    pub fn new(n: usize, c: usize) -> Self {
        let mut cost = vec![vec![0u64; c + 1]; n + 1];
        let mut best = vec![vec![0usize; c + 1]; n + 1];
        for len in 2..=n {
            cost[len][0] = (len * (len - 1) / 2) as u64;
            for k in 1..=c {
                let (mut bc, mut bm) = (u64::MAX, 1);
                for m in (1..len).rev() {
                    let v = m as u64 + cost[len - m][k - 1] + cost[m][k];
                    if v < bc { bc = v; bm = m; }
                }
                cost[len][k] = bc; best[len][k] = bm;
            }
        }
        Self { best, cost }
    }
    pub fn split(&self, n: usize, k: usize) -> usize { self.best[n][k] }
    /// The checkpoints the first descent stores (from 0 towards n): the forward stores exactly these.
    pub fn descent(&self, n: usize, c: usize) -> Vec<usize> {
        let (mut lo, mut k, mut v) = (0usize, c, Vec::new());
        while n - lo > 1 && k > 0 { lo += self.best[n - lo][k]; v.push(lo); k -= 1; }
        v
    }
}
/// S02: the reverse sweep's carried state: the window state at the step being reversed (`cur`), the rows, counters.
struct RevSt { cur: Option<(Boundary, Bytes)>, rows: Vec<Value>, tag: Value }
/// host-leftovers-2: a loaded boundary's bytes: read from a file (owned), or the page-locked RAM copy itself (shared,
/// no copy). Either way the same bytes.
pub(crate) enum Bytes { Own(Vec<u8>), Ram(std::sync::Arc<capi::HostBytes>) }
impl std::ops::Deref for Bytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] { match self { Bytes::Own(v) => v, Bytes::Ram(r) => r.as_slice() } }
}
impl From<Vec<u8>> for Bytes { fn from(v: Vec<u8>) -> Self { Bytes::Own(v) } }

/// x20 fused tape: a tape entry before its forward command ran (an Entry still owes its post-launch part).
enum FzPre {
    Entry { f: String, b: deriv::Bound, args: Vec<submit::Argument>, grid: [u32; 3], block: [u32; 3], shared: u32 },
    Done(Tp),
}
enum Tp {
    Entry { f: String, b: deriv::Bound, args: Vec<submit::Argument>, grid: [u32; 3], block: [u32; 3], shared: u32,
            post: HashMap<u64, submit::Allocation> },
    Rule { r: R, prim: u64 },
    Native { f: String, p: crate::schemes::Prepared },
    /// tempo_warp_steps (--tempo-live 1): the launch inputs, copied into the tape before the launch ran.
    Tempo { f: String, snap: crate::tempo_adw::Snap },
    /// TEMPO serial entry: the launch's pre-launch copy, and its input / output shadow segments
    /// (allocation, lo byte, hi byte, byte offset in the float64 vector), resolved when the launch ran.
    TempoSerial { f: String, snap: crate::tempo_adw::SmallSnap, ins: Vec<(u64, usize, usize, u64)>, outs: Vec<(u64, usize, usize, u64)> },
    /// --tempo-small: a P5 per-thread TEMPO launch (tempo_radar): its input snapshot (in the tape arena) and the shadow
    /// element of every input / output slot block (None: null input, or an output the forward does not write).
    TempoSmall { f: String, snap: crate::tempo_adw::SmallSnap, ins: Vec<(Option<i64>, usize)>, outs: Vec<(Option<i64>, usize)> },
}
enum Taping { Done(Vec<Tp>), Overflow(usize) }
fn exhausted(e: &str) -> bool { e.contains("derivative arena exhausted") }

struct GatherDev { injective: bool, bufs: [u64; 5], nsrc: i64, ndst: i64 }

pub struct Win {
    /// multidevice-runner: reused device buffer (address, bytes) for received cotangents (peer_tmp).
    peer_buf: (u64, usize),
    /// adjoint-speed: the TEMPO arena was sized (once, at the first TEMPO launch).
    tempo_sized: Option<Value>,
    /// adjoint-speed: gathers being built off the dispatch thread during a tangent step (hash, label, worker).
    pending: Vec<(u64, String, std::thread::JoinHandle<Result<wrules::Gather, String>>, usize)>, defer: bool, defer_step: usize,
    /// adjoint-speed: capture the whole state at the next step boundaries (set while the first tangent stores them).
    capture: bool,
    /// fwdcheck-digest (C-ABI session only): capture straight into a page-locked buffer (the pin pool of this context,
    /// 0 = no pool) when the boundary fits the RAM room left (bytes); else the pageable capture, for the disk tier.
    pub(crate) capture_host: Option<(usize, usize)>,
    /// host-leftovers-2 (CLI): the next boundary's page-locked buffer, made on a helper thread while the step runs
    /// (cap bytes, the thread); `pin_ahead` = another boundary will be captured into RAM after this one.
    pin_next: Option<(usize, std::thread::JoinHandle<Result<crate::pinned::PinMem, String>>)>, pin_ahead: bool, pin_ahead_hits: u64,
    /// x20 whole-step owner: every boundary buffer the CLI tangent pass will capture into, started together at the pass start
    /// (one thread each), so no step waits on a 10.5 GB page-lock made one step ahead (pk120: TL step 2 waited 47 s, nsys x20-P1b).
    pin_q: std::collections::VecDeque<(usize, std::thread::JoinHandle<Result<crate::pinned::PinMem, String>>)>, pin_q_made: u64,
    /// x20: the CLI tangent pass joins the copy-row gathers at its end instead of at every step.
    gather_pass_join: bool,
    /// adjoint-speed: (pass, function) -> (entry wall ms, calls), from the entry timers that already exist.
    fn_ms: BTreeMap<(String, String), (f64, u64)>,
    /// adjoint-speed: one-pass reverse steps (default on; --one-pass 0 keeps the segmented path), fallbacks counted.
    one_pass: bool, one_pass_fallbacks: usize, tape_grown: Option<Value>,
    /// pk120: tangent binds that outgrew the tape arena and grew it (bound: --tape-max-gib, card free less 6 GiB).
    tape_max: usize, tl_tape_grown: Vec<Value>,
    /// x20 reverse-structure (fused TL/AD tape; AD_FUSE_TAPE=1 or --fuse-tape 1): the tangent pass, which already runs
    /// every forward command of the step, also records the reverse tape (the same pre-launch / post-launch copies,
    /// native and TEMPO snapshots and rule primals tape_range takes, in a second arena `rtape`), so the reverse walks
    /// it back without restoring the step start and replaying the step. Step tapes pack into the arena back to back; a
    /// full arena is downloaded once into page-locked RAM (a "chunk") and uploaded to the same addresses when the
    /// reverse reaches it, so every address in the tape stays valid. A step whose recording does not fit is not fused
    /// and reverses on the ordinary path (the same words).
    fuse: bool, fuse_rec: bool, fuse_ok: bool, rtape: Option<dev::Arena>,
    rule_cache: Option<(R, Option<(String, Vec<submit::Argument>, [u32; 3], [u32; 3], u32)>)>,
    fz_cur: Option<(usize, Vec<Tp>)>, fz_tapes: BTreeMap<usize, (usize, Vec<Tp>)>,
    fz_chunks: Vec<(Vec<usize>, Option<crate::pinned::PinnedBuf>, usize)>, fz_resident: Option<usize>,
    fz_last_bytes: usize, fz_step_start: usize, fz_info: Vec<Value>, fz_d2h_s: f64, fz_d2h_bytes: usize, fz_failed: bool, fz_force_flush: bool, fz_steps: usize,
    /// packed-tape (default on; AD_TAPE_PACK=0 keeps the old tape): transient buffers bound from the scratch at
    /// walk-back, pre-launch copies shared when word-equal (deriv::PreCache). `rows`: per-row tape bytes of the current
    /// tangent step, function -> [launches, pre bytes, meta bytes, transient bytes, post bytes, other tape bytes].
    /// `rev_scratch`: the largest transient bytes of one launch; the reverse scratch holds them on top of its old size
    /// (`scratch_base`, which already held every launch's workspace).
    /// `pack_ratio`: the largest taped / estimated bytes of a packed one-pass step so far (0 before the first).
    pack: bool, precache: deriv::PreCache, rows: BTreeMap<String, [usize; 6]>, rev_rows: BTreeMap<String, [usize; 8]>, seg_rows: BTreeMap<String, [usize; 8]>, census: Option<Census>, win: Option<dev::Arena>, win_cap: usize, win_promoted: (usize, usize), rev_scratch: usize, scratch_base: usize, pack_ratio: f64, seg_ratio: f64,
    /// step6-sw: element type of each cupy_concatenate_same_size module (wrules::concat_module_types).
    pub concat_types: HashMap<String, String>,
    kernels: HashMap<String, deriv::KernelDeriv>,
    classes: HashMap<String, String>,
    schemes: HashMap<String, String>,
    refuse: HashMap<String, String>,
    base_el: HashMap<u64, i64>,
    sh_base: u64,
    words: i64,
    shadows: BTreeMap<u64, deriv::Shadow>,
    support: dev::Support,
    /// readbacks-device (round 3): the step ledger. Every derivative entry's failure slots, the
    /// P3 failure codes and the TEMPO overflow/fault/non-finite flags and receipt sums are queued here on the device
    /// and read with one download at the end of each tangent and reverse step (step_flush). None only while a native
    /// row holds it.
    ledger: Option<dev::Ledger>,
    /// readbacks-device: the TEMPO derivative's float64 input and result vectors on the device (address, bytes).
    tempo_io: (u64, usize),
    /// readbacks-device: window_tempo receipt rows waiting for their ledger values:
    /// (pass, step, ms, tiles, input sum-of-squares token, output tokens, identity count token, compared words).
    tempo_rows: Vec<(&'static str, usize, f64, usize, u64, Vec<u64>, Option<u64>, usize, [f64; 4])>,
    /// host-leftovers: a derivative entry's closing device sync is owed, not taken. Every entry launches on the null
    /// stream, so entry k+1's host binding (prepare_with, the reverse's seed snapshot and launches) runs while entry k's
    /// kernels do, in stream order. The owed sync is paid before anything that is not an entry (a forward command on a
    /// recorded stream, native/TEMPO rows, rules) and at the end of each walk. Off (sync per entry) under AD_G5_PROF=1,
    /// whose per-category timers need it, and with AD_ENTRY_SYNC=1.
    sync_owed: bool,
    f: Fns,
    tape: dev::Arena,
    scratch: dev::Arena,
    tmp: [u64; 3],
    tmp_words: i64,
    part: u64,
    fail: u64,
    gathers: HashMap<u64, Result<GatherDev, String>>,
    counting: bool,
    labels: BTreeMap<String, u64>,
    held_dropped: BTreeMap<String, f64>,
    entry_ms: f64,
    writer: Option<std::thread::JoinHandle<Result<(), String>>>,
    trace: bool,
    trace_last: f64,
    trace_hits: u32,
    /// Trace growth factor that logs a command (AD_TRACE_GROWTH, default 100), log cap per pass (AD_TRACE_HITS, default
    /// 40 forward / 60 reverse), and the metric: AD_TRACE_STATE=1 sums only the plan's initial allocations (the carried
    /// model state; mid-step Allocate ids are per-step scratch, zeroed at their Allocate).
    trace_growth: f64,
    trace_cap: u32,
    trace_state: bool,
    state_words: i64,
    /// Per-step tangent norms per allocation (AD_STEP_NORMS=1, or --trace-norm 1).
    step_norms: bool,
    /// --direction field: y_i = u_i / rms_f (rms of the valid words of allocation f at the window end), x relative.
    field_y: bool,
    /// --direction fieldxy (candidate fix B): also x_i = u_i * rms_f, rms_f of allocation f's valid words at boundary 0
    /// (per-field typical magnitude), with y_i = u_i / rms_f at the window end as in field. Only the test's directions
    /// change; the operator M is the same. The same x seeds the tangent, <x, M^T y> and the Taylor perturbation.
    /// --x-scale (see the constructor): uniform factor on the x-side direction weights.
    x_scale: f32,
    y_scale: f32,
    field_x: bool,
    y_path: Option<PathBuf>,
    /// --stable-dir 1: process-dependent words get zero direction weight (see scale_chunk).
    stable_dir: bool,
    relocs: HashMap<u64, Vec<(usize, usize)>>,
    residue: HashMap<(usize, u64), Vec<(usize, usize)>>,
    /// Allocations the sealed catalog shows unchanged at every boundary (initial bytes = every step boundary):
    /// fixed geometry and tables. Directions put no weight there; their cotangent is a sink no command reads.
    fixed: std::collections::HashSet<u64>,
    /// --fct-frozen 1 (default off): frozen FCT limiter coefficients (see wrules.rs `fct_frozen`).
    fct_frozen: bool,
    /// Taylor check: directions skip address-bearing allocations (their pointer words must not be perturbed).
    taylor_mask: bool,
    address_bearing: std::collections::HashSet<u64>,
    /// --native-schemes 1: the L2 physics rows (AD_SCHEMES), live through the float32-shadow adapter.
    native: Option<crate::schemes::Native>,
    /// --tempo-live 1 (ad/held-tempo_warp_steps): TEMPO's tempo_warp_steps derivative (the L5 tempo_adw image, the
    /// one the per-launch dot proves) live in the composed operator, with its own work arena (--tempo-arena-gib, 6).
    tempo: Option<(crate::tempo_adw::TempoAdw, dev::Arena)>,
    tempo_stats: BTreeMap<String, f64>,
    /// --tempo-precip 1 (default off): tempo_precip live through P5's per-thread entry (AD_DOT_TEMPO_SERIAL_CUBIN);
    /// without the flag, or with --hold tempo / physics, it stays held.
    tempo_serial: Option<crate::tempo_adw::TempoSmall>,
    /// --tempo-small tempo_radar (ad/held-tempo_radar): launches bound to the P5 per-thread entry (AD_DOT_TEMPO_SERIAL_CUBIN)
    /// through the float32-shadow adapter (float64 tape and buffers, shadows widened before and rounded after).
    tempo_small: Option<crate::tempo_adw::TempoSmall>,
    tempo_small_fns: std::collections::HashSet<String>,
    tempo_small_identity_words: u64,
    /// S04 (taylor-row-host): reused page-locked buffers for the reverse step's mid-step anchors (snapshot and
    /// restore by DMA instead of a host memcpy through fresh pageable memory), and restore wall seconds (receipt only).
    anchor_pool: Vec<crate::pinned::PinnedBuf>,
    /// anchors-ahead: the page-locked anchor buffers a helper thread makes during the tangent pass (made so far, helper
    /// done), the main-thread seconds spent waiting for one, how many were taken, and the plan (receipt).
    anchor_ahead: Option<AheadPool>, anchor_ahead_wait_s: f64, anchor_ahead_taken: u64, anchor_ahead_info: Vec<Value>,
    restore_s: f64,
    /// G5: background decodes in flight (the blob hashes each holds, the decoder). integ8: one batch per prefetched
    /// step, so a step waits only for the batch that holds a blob it needs.
    blob_next: Vec<(std::collections::HashSet<String>, std::thread::JoinHandle<Result<Vec<(String, Vec<u8>)>, String>>)>,
    /// integ8: blobs the next forward steps need (decoded ahead): stage_blobs keeps them in the cache.
    blob_ahead: std::collections::HashSet<String>,
    /// integ8: reused page-locked staging buffer of capture_state.
    capture_pin: Option<crate::pinned::PinnedBuf>,
    /// adjoint-speed-2: page-locked buffers of the one-pass reverse (the redo-check download, the restore staging).
    cmp_pin: Option<crate::pinned::PinnedBuf>, restore_pin: Option<crate::pinned::PinnedBuf>,
    /// adjoint-speed-2: per-step reverse stage walls (receipt only; Instant reads, no syncs added).
    rv_restore_s: f64, rv_tape_s: f64, rv_redo_dl_s: f64, rv_redo_wait_s: f64, rv_walk_s: f64,
    /// G5: background read of the next reverse step's start boundary (index of `bounds`, bytes).
    load_next: Option<(usize, std::thread::JoinHandle<Result<Vec<u8>, String>>)>,
    /// runner-lib: preallocated device buffers for the reverse step's anchors (empty: anchors in host RAM, as before).
    /// Only libhexad-phys fills it, when the card has room; the bytes are the same, only where they wait changes.
    pub(crate) dev_anchors: Vec<(u64, usize)>,
    /// S02 (--checkpoints): boundary 0 (index, bytes), the reference of changed-chunk boundary files; whether stores are
    /// changed-chunk; boundary-store bytes on disk now / at peak / written, files now / at peak; recomputed forward steps.
    ckpt_base: Option<(Vec<(u64, usize, usize)>, std::sync::Arc<Bytes>)>,
    delta_store: bool,
    bs_now: u64, bs_peak: u64, bs_written: u64, bs_files: u64, bs_files_peak: u64,
    recompute_steps: u64, recompute_s: f64,
    /// S02: background read + decode of the checkpoint the reverse sweep loads next (path, bytes); seconds spent in load
    /// (incl. waiting for a prefetch) and store; prefetch hits.
    ck_pf: Option<(PathBuf, std::thread::JoinHandle<Result<Vec<u8>, String>>)>,
    load_s: f64, store_s: f64, pf_hits: u64,
    /// host-leftovers-2: CLI boundaries kept in page-locked RAM instead of b{s}.bin (budget, bytes held now, peak, count).
    bs_ram_budget: usize, bs_ram_now: usize, bs_ram_peak: usize, bs_ram_count: u64,
    /// host-leftovers-2: restores DMA'd straight from page-locked bytes (no staging copy).
    pub(crate) restore_direct: u64,
    /// host-leftovers-2: seed-weight prep per (boundary path, boundary step, allocation, stable, inv, fwd): the merged
    /// unstable byte ranges and the field rms (see scale_chunk). The boundary bytes at a path and step do not change.
    wcache: std::sync::Mutex<HashMap<(PathBuf, usize, u64, bool, bool, bool), std::sync::Arc<(Vec<(usize, usize)>, f32)>>>,
    /// lane rung3-params: --params FILE, the parameter carrier (P4 module globals); empty without the flag.
    params: Vec<Par>,
}
/// lane rung3-params: one carried parameter: P4 family, carrier slot, value at the trajectory (the recorded literal).
#[derive(Clone)]
struct Par { name: String, fam: String, slot: usize, p0: f32 }
/// J(perturbed end) - J(stored end) over sparse objective terms (allocation, word, weight), and the changed words.
unsafe fn obj_dj_terms(ex: &Execution, b: &Boundary, end: &[u8], terms: &[(u64, usize, f32)]) -> Result<(f64, u64), String> {
    dev::sync(&ex.cuda)?;
    let (mut dj, mut ch) = (0f64, 0u64);
    for &(al, wd, wt) in terms {
        let a = ex.allocations.get(&al).ok_or("objective allocation not live")?;
        let v = dev::download(&ex.cuda, a.address + 4 * wd as u64, 4)?;
        let q = f32::from_le_bytes(v[..4].try_into().unwrap());
        let (_, off, _) = b.index.iter().find(|x| x.0 == al).ok_or("objective allocation not in the boundary")?;
        let p = f32::from_le_bytes(end[off + 4 * wd..off + 4 * wd + 4].try_into().unwrap());
        if !q.is_finite() { return Err(format!("objective word {al}:{wd} not finite")); }
        if q.to_bits() != p.to_bits() { ch += 1; }
        dj += wt as f64 * (q as f64 - p as f64);
    }
    Ok((dj, ch))
}

fn opt<'a>(a: &'a HashMap<String, String>, k: &str) -> Option<&'a str> { a.get(k).map(String::as_str) }
fn req<'a>(a: &'a HashMap<String, String>, k: &str) -> Result<&'a str, String> { opt(a, k).ok_or_else(|| format!("missing --{k}")) }

/// Lane j4-repose: a smooth nonlinear objective of window-end words (objective JSON "form", default off: linear).
/// Per group (a model column) a list of levels; each level gives a reflectivity s_k in dBZ:
///  * kind "dbz" (default): one word per level, already in dBZ, with a smooth floor s = F + w softplus((v - F) / w)
///    (w = 0: s = v);
///  * kind "mass": s_k = 10 log10(z0 + sum_t c_t max(q_t, 0)^e_t) over the level's terms (rain / snow / graupel mass
///    mixing ratio words, c_t carrying the frozen density factor), so z0 is a smooth floor in linear Ze.
/// Then the log-sum-exp column aggregate A = T ln sum_k exp(s_k / T) (T in dB: T = 10 / ln 10 makes 10^(A/10) the
/// column sum of linear Ze, smaller T approaches the column maximum); out "dbz": J_g = A, out "ze": J_g = 10^(A/10)
/// (linear Ze, mm^6 m^-3). J = sum_g weight_g J_g, evaluated and differentiated in float64 on the host; the gradient
/// at the window-end state seeds the objective reverse.
#[derive(Clone, Debug)]
pub(crate) struct Form {
    pub t_db: f64, pub floor_db: f64, pub floor_w_db: f64, pub ze: bool, pub z0: Option<f64>,
    /// per level: term range; per term: (coefficient, exponent) (mass kind only); per group: (weight, level range)
    pub levels: Vec<(usize, usize)>, pub coef: Vec<(f64, f64)>, pub groups: Vec<(f64, usize, usize)>,
    /// lane s16: kind "misfit": one observation per group, y_g (the group weight is r_g = 1 / sigma_g^2).
    pub misfit_y: Option<Vec<f64>>,
}
impl Form {
    /// Parse {"form": {"kind", "t_db", "floor_db", "floor_width_db", "z0", "out"}, "groups": [[weight, [level, ..]], ..]}
    /// (a level is [allocation, word] for kind dbz, [[allocation, word, coefficient, exponent], ..] for kind mass) into
    /// the form and its flat word list (weight 0: the reverse seed comes from eval()).
    pub(crate) fn parse(doc: &Value, p: &str) -> Result<(Form, Vec<(u64, usize, f32)>), String> {
        let f = &doc["form"];
        // lane s16: {"form": {"kind": "misfit"}, "groups": [[r, [[allocation, word, coefficient], ..], y], ..]}
        if f["kind"].as_str() == Some("misfit") {
            let (mut groups, mut levels, mut coef, mut terms, mut ys) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for g in doc["groups"].as_array().ok_or_else(|| format!("{p}: groups"))? {
                let r = g[0].as_f64().ok_or_else(|| format!("{p}: misfit weight"))?;
                let y = g[2].as_f64().ok_or_else(|| format!("{p}: misfit observation"))?;
                if !(r >= 0.0) || !y.is_finite() { return Err(format!("{p}: misfit needs r >= 0 and a finite observation")); }
                let t0 = terms.len();
                for t in g[1].as_array().ok_or_else(|| format!("{p}: misfit terms"))? {
                    let (Some(al), Some(wd), Some(c)) = (t[0].as_u64(), t[1].as_u64(), t[2].as_f64()) else { return Err(format!("{p}: misfit term")) };
                    terms.push((al, wd as usize, 0f32)); coef.push((c, 1.0));
                }
                if terms.len() == t0 { return Err(format!("{p}: empty misfit group")); }
                levels.push((t0, terms.len())); groups.push((r, levels.len() - 1, levels.len())); ys.push(y);
            }
            if groups.is_empty() { return Err(format!("{p}: misfit has no observations")); }
            return Ok((Form { t_db: 1.0, floor_db: 0.0, floor_w_db: 0.0, ze: false, z0: None, levels, coef, groups, misfit_y: Some(ys) }, terms));
        }
        let num = |k: &str, d: Option<f64>| f[k].as_f64().or(d).ok_or_else(|| format!("{p}: form.{k}"));
        let (t_db, floor_db, floor_w_db) = (num("t_db", None)?, num("floor_db", Some(-35.0))?, num("floor_width_db", Some(0.0))?);
        if !(t_db > 0.0) || floor_w_db < 0.0 { return Err(format!("{p}: form needs t_db > 0 and floor_width_db >= 0")); }
        let ze = match f["out"].as_str().unwrap_or("dbz") { "ze" => true, "dbz" => false, o => return Err(format!("{p}: form.out {o}")) };
        let mass = match f["kind"].as_str().unwrap_or("dbz") { "mass" => true, "dbz" | "lse" => false, o => return Err(format!("{p}: form.kind {o}")) };
        let z0 = if mass { let z = num("z0", None)?; if !(z > 0.0) { return Err(format!("{p}: form.z0 must be > 0")); } Some(z) } else { None };
        let (mut groups, mut levels, mut coef, mut terms) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for g in doc["groups"].as_array().ok_or_else(|| format!("{p}: groups"))? {
            let wt = g[0].as_f64().ok_or_else(|| format!("{p}: group weight"))?;
            let l0 = levels.len();
            for lv in g[1].as_array().ok_or_else(|| format!("{p}: group levels"))? {
                let t0 = terms.len();
                let list: Vec<&Value> = if mass { lv.as_array().ok_or_else(|| format!("{p}: mass level"))?.iter().collect() } else { vec![lv] };
                for t in list {
                    let (Some(al), Some(wd)) = (t[0].as_u64(), t[1].as_u64()) else { return Err(format!("{p}: level word")) };
                    terms.push((al, wd as usize, 0f32));
                    coef.push(if mass { (t[2].as_f64().ok_or_else(|| format!("{p}: coefficient"))?, t[3].as_f64().ok_or_else(|| format!("{p}: exponent"))?) } else { (1.0, 1.0) });
                }
                if terms.len() == t0 { return Err(format!("{p}: empty level")); }
                levels.push((t0, terms.len()));
            }
            if levels.len() == l0 { return Err(format!("{p}: empty group")); }
            groups.push((wt, l0, levels.len()));
        }
        Ok((Form { t_db, floor_db, floor_w_db, ze, z0, levels, coef, groups, misfit_y: None }, terms))
    }
    fn soft(&self, v: f64) -> (f64, f64) {
        if self.floor_w_db == 0.0 { return (v, 1.0); }
        let z = (v - self.floor_db) / self.floor_w_db;
        let sp = if z > 30.0 { z } else { z.exp().ln_1p() };
        (self.floor_db + self.floor_w_db * sp, 1.0 / (1.0 + (-z).exp()))
    }
    /// One level's dBZ and d(dBZ)/d(word) over its term range.
    fn level(&self, v: &[f64], t0: usize, t1: usize, d: &mut [f64]) -> f64 {
        match self.z0 {
            None => { let (s, ds) = self.soft(v[t0]); d[t0] = ds; s }
            Some(z0) => {
                let mut z = z0;
                for t in t0..t1 { let (c, e) = self.coef[t]; if v[t] > 0.0 { z += c * v[t].powf(e); } }
                for t in t0..t1 { let (c, e) = self.coef[t]; d[t] = if v[t] > 0.0 { 10.0 / std::f64::consts::LN_10 * c * e * v[t].powf(e - 1.0) / z } else { 0.0 }; }
                10.0 * z.log10()
            }
        }
    }
    /// J and dJ/dv at the word values v (aligned with the flat word list).
    pub(crate) fn eval(&self, v: &[f64]) -> (f64, Vec<f64>) {
        let (mut j, mut g, mut d) = (0f64, vec![0f64; v.len()], vec![0f64; v.len()]);
        if let Some(ys) = &self.misfit_y {
            for (gi, &(r, l0, _)) in self.groups.iter().enumerate() {
                let (t0, t1) = self.levels[l0];
                let hx: f64 = (t0..t1).map(|t| self.coef[t].0 * v[t]).sum();
                let dd = hx - ys[gi];
                j += 0.5 * r * dd * dd;
                for t in t0..t1 { g[t] = r * dd * self.coef[t].0; }
            }
            return (j, g);
        }
        for &(wt, l0, l1) in &self.groups {
            let s: Vec<f64> = (l0..l1).map(|l| { let (t0, t1) = self.levels[l]; self.level(v, t0, t1, &mut d) }).collect();
            let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = s.iter().map(|x| ((x - m) / self.t_db).exp()).collect();
            let sum: f64 = e.iter().sum();
            let a = m + self.t_db * sum.ln();
            let (jg, dja) = if self.ze { let z = 10f64.powf(a / 10.0); (z, z * std::f64::consts::LN_10 / 10.0) } else { (a, 1.0) };
            j += wt * jg;
            for (k, l) in (l0..l1).enumerate() { let (t0, t1) = self.levels[l]; for t in t0..t1 { g[t] = wt * dja * (e[k] / sum) * d[t]; } }
        }
        (j, g)
    }
}

/// j4-repose: an objective's words at the window end, base (from boundary data) and live (downloaded), float64.
fn base_words(b: &Boundary, data: &[u8], terms: &[(u64, usize, f32)]) -> Result<Vec<f64>, String> {
    // S04: the boundary index looked up once per allocation, not scanned once per term (first entry wins, as find did).
    let at: HashMap<u64, usize> = b.index.iter().rev().map(|x| (x.0, x.1)).collect();
    terms.iter().map(|&(al, wd, _)| {
        let off = *at.get(&al).ok_or("objective allocation not in the boundary")?;
        Ok(f32::from_le_bytes(data[off + 4 * wd..off + 4 * wd + 4].try_into().unwrap()) as f64)
    }).collect()
}
unsafe fn live_words(ex: &Execution, terms: &[(u64, usize, f32)]) -> Result<(Vec<f64>, Vec<u32>), String> {
    // S04: one download of the span of each allocation's terms, not one 4-byte download per term (same words).
    let mut span: BTreeMap<u64, (usize, usize)> = BTreeMap::new();
    for &(al, wd, _) in terms { let e = span.entry(al).or_insert((wd, wd)); e.0 = e.0.min(wd); e.1 = e.1.max(wd); }
    let mut got: HashMap<u64, Vec<u8>> = HashMap::new();
    for (al, &(lo, hi)) in &span {
        let a = ex.allocations.get(al).ok_or("objective allocation not live")?;
        got.insert(*al, dev::download(&ex.cuda, a.address + 4 * lo as u64, 4 * (hi - lo + 1))?);
    }
    let (mut v, mut bits) = (Vec::with_capacity(terms.len()), Vec::with_capacity(terms.len()));
    for &(al, wd, _) in terms {
        let k = 4 * (wd - span[&al].0);
        let q = f32::from_le_bytes(got[&al][k..k + 4].try_into().unwrap());
        v.push(q as f64); bits.push(q.to_bits());
    }
    Ok((v, bits))
}
fn now() -> String {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("{t}")
}

/// Kernel-set admission (lane runner-dump): each row's recorded derivative spec sha256 against the spec file it points
/// at, hashed on all cores, refused on any mismatch before an image loads. Breakage it prevents: a row whose spec file
/// changed after the set recorded it is bound with a launch ABI the derivative image was not built from, and nothing
/// says so (INTEGRATE-20261106 item 5: kset-integ rows had stale records the window never checked). The e2e bind
/// tool refuses the same way. Rows without a recorded sha256 are counted, not refused (the e2e rule).
pub fn check_spec_hashes(kernels: &Path) -> Result<Value, String> {
    let set: Value = serde_json::from_slice(&fs::read(kernels).map_err(|e| format!("{}: {e}", kernels.display()))?)
        .map_err(|e| format!("{}: {e}", kernels.display()))?;
    let rows: Vec<(String, String, Option<String>)> = set["kernels"].as_array().ok_or("kernels missing")?.iter().filter_map(|k| {
        let sp = &k["derivative"]["spec"];
        Some((k["function"].as_str().unwrap_or("?").to_owned(), sp["path"].as_str()?.to_owned(), sp["sha256"].as_str().map(str::to_owned)))
    }).collect();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, 32);
    let chunk = rows.len().div_ceil(threads).max(1);
    let got: Vec<Result<Option<String>, String>> = std::thread::scope(|sc| {
        let hs: Vec<_> = rows.chunks(chunk).map(|c| sc.spawn(move || c.iter().map(|(_, p, want)| {
            if want.is_none() { return Ok(None); }
            fs::read(crate::deriv::spec_path(kernels, p)).map(|b| Some(crate::exec::digest(&b))).map_err(|e| format!("{p}: {e}"))
        }).collect::<Vec<_>>())).collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let (mut bad, mut checked, mut unrecorded) = (Vec::new(), 0u64, 0u64);
    for ((f, p, want), g) in rows.iter().zip(got) {
        match (want, g?) {
            (Some(w), Some(h)) => { checked += 1; if *w != h { bad.push(format!("{f}: spec sha256 {h} != recorded {w} ({p})")); } }
            _ => unrecorded += 1,
        }
    }
    if !bad.is_empty() {
        return Err(format!("kernel set {}: {} spec file(s) differ from their recorded sha256; the derivative images were not made from these files (rebuild the set, or re-hash it if the spec content is unchanged): {}",
                           kernels.display(), bad.len(), bad.join("; ")));
    }
    Ok(json!({"kernels":kernels.display().to_string(),"specs_checked":checked,"specs_unrecorded":unrecorded}))
}

/// --dump-state: one listed window-end field written raw (float32 words) with its receipt line.
fn write_state(out: &Path, name: String, bytes: &[u8]) -> Result<Value, String> {
    let p = out.join(name);
    fs::write(&p, bytes).map_err(|e| format!("{}: {e}", p.display()))?;
    let v: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    Ok(json!({"file":p.display().to_string(),"sha256":crate::exec::digest(bytes),"words":v.len(),
              "nonfinite":v.iter().filter(|x| !x.is_finite()).count(),"max_abs":v.iter().filter(|x| x.is_finite()).fold(0f32, |m, x| m.max(x.abs()))}))
}

impl Win {
    unsafe fn new(ex: &Execution, plan: &Plan, a: &HashMap<String, String>) -> Result<Self, String> {
        let cuda = &ex.cuda;
        let mut kernels = deriv::load(cuda, Path::new(req(a, "kernels")?), Path::new(req(a, "cubins")?), None)?;
        let set: Value = serde_json::from_slice(&fs::read(req(a, "kernels")?).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let mut classes: HashMap<String, String> = set["kernels"].as_array().ok_or("kernels")?.iter()
            .map(|k| (k["function"].as_str().unwrap_or("").to_owned(), k["class"].as_str().unwrap_or("").to_owned())).collect();
        let held = crate::held::apply(a, &mut kernels, &mut classes)?;
        let mut schemes = HashMap::new();
        if let Some(p) = opt(a, "schemes") {
            let doc: Value = serde_json::from_slice(&fs::read(p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
            for (k, r) in doc["rows"].as_object().ok_or("schemes rows")? { schemes.insert(k.clone(), r["scheme"].as_str().unwrap_or("?").to_owned()); }
        }
        let mut refuse = HashMap::new();
        for f in opt(a, "refuse").unwrap_or("").split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let (f, why) = f.split_once('=').unwrap_or((f, "refused for this run (--refuse)"));
            refuse.insert(f.to_owned(), why.to_owned());
        }
        // Shadow arena: every plan allocation, including the mid-step Allocate ids of every block.
        let mut sizes: BTreeMap<u64, usize> = plan.allocations.iter().map(|x| (x.id, x.bytes)).collect();
        for ops in plan.blocks.values() { for op in ops { if let Operation::Allocate { id, bytes, .. } = op {
            let e = sizes.entry(*id).or_insert(0); *e = (*e).max(*bytes);
        } } }
        // multidevice-runner: a --global-arena rank plan takes the unsplit plan's layout (every id this rank holds must be
        // in it with the same size; THE BREAKAGE THIS PREVENTS: a layout that differs would make the comparison with
        // the one-process run compare different words).
        if let Some(full) = plan.peer_arena_sizes.as_ref() {
            let full: BTreeMap<u64, usize> = full.iter().map(|x| (x[0], x[1] as usize)).collect();
            for (id, b) in &sizes { if full.get(id) != Some(b) { return Err(format!("peer_arena_sizes: allocation {id} ({b} bytes) differs from the unsplit layout")); } }
            sizes = full;
        }
        let mut base_el = HashMap::new();
        let mut at = 0i64;
        for (id, bytes) in &sizes { base_el.insert(*id, at); at += ((*bytes as i64 + 3) / 4 + 63) / 64 * 64; }
        let words = at;
        let init: std::collections::HashSet<u64> = plan.allocations.iter().map(|x| x.id).collect();
        let max_init = init.iter().copied().max().unwrap_or(0);
        let state_words = if sizes.keys().all(|id| init.contains(id) || *id > max_init) {
            sizes.keys().find(|id| **id > max_init).map(|id| base_el[id]).unwrap_or(words)
        } else { words };
        let mut sh_base = 0u64;
        checked("cuMemAlloc shadow arena", (cuda.allocate)(&mut sh_base, words as usize * 4))?;
        dev::zero(cuda, sh_base, words as usize * 4)?;
        let shadows = sizes.iter().map(|(id, bytes)| (*id, deriv::Shadow { address: sh_base + 4 * base_el[id] as u64,
                                                                          bytes: bytes.div_ceil(4) * 4, width: 4, lo: 0 })).collect();
        let (mp, mw) = (module(cuda, P8_IMAGE, "p8 rules")?, module(cuda, WIN_IMAGE, "window support")?);
        let f = Fns { rows_tl: function(cuda, mp, "p8_rows_tl_float")?, gather_ad: function(cuda, mp, "p8_gather_ad_float")?,
            rows_ad_direct: function(cuda, mp, "p8_rows_ad_direct_float")?, clear_dst: function(cuda, mp, "p8_clear_dst_float")?,
            view_tl: function(cuda, mp, "p8_view_tl_float")?, view_ad: function(cuda, mp, "p8_view_ad_float")?,
            where_tl: function(cuda, mp, "p8_where_tl_float")?, where_ad: function(cuda, mp, "p8_where_ad_float")?,
            max_tl: function(cuda, mp, "p8_max_tl_float")?, max_ad: function(cuda, mp, "p8_max_ad_float")?,
            view_zero: function(cuda, mw, "win_view_zero_f32")?, fill: function(cuda, mw, "win_fill_f32")?, dot2: function(cuda, mw, "win_dot2_f32")?, nonfinite: function(cuda, mw, "win_nonfinite_f32")?,
            fct_minmax_tl: function(cuda, mw, "win_fct_minmax_frozen_tl")?, fct_minmax_ad: function(cuda, mw, "win_fct_minmax_frozen_ad")?,
            axpy: function(cuda, mw, "win_axpy_primal_f32")?, widen: function(cuda, mw, "win_widen_f32_f64")?,
            narrow: function(cuda, mw, "win_narrow_f64_f32")? };
        let native = if opt(a, "native-schemes") == Some("1") {
            let names: std::collections::HashSet<String> = kernels.keys().cloned().collect();
            Some(crate::schemes::Native::from_env(cuda, &names)?.ok_or("--native-schemes 1 needs AD_SCHEMES (the L2 bundle)")?)
        } else { None };
        let fct_frozen = opt(a, "fct-frozen") == Some("1");
        let hold_tempo = opt(a, "hold").unwrap_or("").split(',').map(str::trim).any(|s| s == "tempo" || s == "physics");
        // --tempo all (ad/g1-merged): every TEMPO derivative this binary carries is live in one window: tempo_warp_steps
        // (L5 tempo_adw, as --tempo-live 1), tempo_precip (P5 serial, as --tempo-precip 1) and tempo_radar (P5 serial, as
        // --tempo-small tempo_radar). The hand-offs tempo_from (scheme row kind tempo_from, --native-schemes 1 with a
        // bundle carrying it) and tempo_to (kernel-set ad.entry row) bind through their rows, so warp_steps receives and
        // returns a state tangent. The single-mechanism flags stay as they were; --hold tempo holds all of them.
        let tempo_all = match opt(a, "tempo") { None | Some("0") => false, Some("all") | Some("1") => true,
            Some(v) => return Err(format!("--tempo {v}: expected all (or 0)")) };
        if tempo_all && hold_tempo { return Err("--tempo all with --hold tempo/physics: pick one".into()); }
        let tempo_serial = if (opt(a, "tempo-precip") == Some("1") || tempo_all) && !hold_tempo {
            let p = std::env::var("AD_DOT_TEMPO_SERIAL_CUBIN").map_err(|_| "--tempo-precip 1 needs AD_DOT_TEMPO_SERIAL_CUBIN (P5's serial tempo_ad image)")?;
            Some(crate::tempo_adw::TempoSmall::load(cuda, &p)?)
        } else { None };
        let mut tempo_small_fns: std::collections::HashSet<String> = opt(a, "tempo-small").map(|v| v.split(',').filter(|x| !x.is_empty()).map(str::to_owned).collect()).unwrap_or_default();
        if tempo_all { tempo_small_fns.insert("tempo_radar".to_owned()); }
        if hold_tempo { tempo_small_fns.clear(); }
        if let Some(f) = tempo_small_fns.iter().find(|f| f.as_str() != "tempo_radar") {
            return Err(format!("--tempo-small {f}: the window binds the P5 per-thread entry for tempo_radar only"));
        }
        let tempo_small = if tempo_small_fns.is_empty() { None } else {
            let path = std::env::var("AD_DOT_TEMPO_SERIAL_CUBIN").ok().filter(|v| !v.is_empty())
                .ok_or("--tempo-small needs AD_DOT_TEMPO_SERIAL_CUBIN (the P5 tempo_ad_serial sm_120 image)")?;
            Some(crate::tempo_adw::TempoSmall::load(cuda, &path)?)
        };
        // lane rung3-params: {"name", "params": [{"name", "family": "mynnsfc"|"ugwp", "slot", "p0_bits": "0x........"}]}
        let mut params = Vec::new();
        if let Some(p) = opt(a, "params") {
            if native.is_none() { return Err("--params needs --native-schemes 1 and AD_SCHEMES (the carrier images)".into()); }
            let doc: Value = serde_json::from_slice(&fs::read(p).map_err(|e| format!("{p}: {e}"))?).map_err(|e| format!("{p}: {e}"))?;
            for q in doc["params"].as_array().ok_or("--params: params")? {
                let bits = u32::from_str_radix(q["p0_bits"].as_str().ok_or("--params: p0_bits")?.trim_start_matches("0x"), 16).map_err(|e| e.to_string())?;
                let slot = q["slot"].as_u64().ok_or("--params: slot")? as usize;
                if slot >= 8 { return Err("--params: slot must be 0..7".into()); }
                params.push(Par { name: q["name"].as_str().ok_or("--params: name")?.into(), fam: q["family"].as_str().ok_or("--params: family")?.into(),
                                  slot, p0: f32::from_bits(bits) });
            }
        }
        let gib = |k: &str, d: f64| opt(a, k).and_then(|v| v.parse::<f64>().ok()).unwrap_or(d);
        let tape = dev::Arena::new(cuda, (gib("tape-gib", 4.0) * (1u64 << 30) as f64) as usize)?;
        let scratch = dev::Arena::new(cuda, (gib("scratch-gib", 1.0) * (1u64 << 30) as f64) as usize)?;
        let tempo = if (opt(a, "tempo-live") == Some("1") || tempo_all) && !hold_tempo {
            let p = opt(a, "tempo-cubin").map(str::to_owned).or_else(|| std::env::var("AD_WIN_TEMPO_CUBIN").ok().filter(|v| !v.is_empty()))
                .or_else(|| std::env::var("AD_DOT_TEMPO_CUBIN").ok().filter(|v| !v.is_empty()))
                .ok_or("--tempo-live 1 needs --tempo-cubin (or AD_WIN_TEMPO_CUBIN): the sm_120 tempo_adw image")?;
            let mut t = crate::tempo_adw::TempoAdw::load(cuda, &p)?;
            // tempo-once: the fused tangent module beside the image (<image>.tlf.cubin, mk_tlf.py), or --tempo-tlf-cubin /
            // AD_WIN_TEMPO_TLF_CUBIN; --tempo-tlf 0 keeps the segmented tangent.
            if opt(a, "tempo-tlf") != Some("0") {
                let sib = p.strip_suffix(".cubin").map(|b| format!("{b}.tlf.cubin")).filter(|q| std::path::Path::new(q).exists());
                let q = opt(a, "tempo-tlf-cubin").map(str::to_owned).or_else(|| std::env::var("AD_WIN_TEMPO_TLF_CUBIN").ok().filter(|v| !v.is_empty())).or(sib);
                if let Some(q) = q { t.load_tlf(cuda, &q)?; }
            }
            let ar = dev::Arena::new(cuda, (gib("tempo-arena-gib", 6.0) * (1u64 << 30) as f64) as usize)?;
            Some((t, ar))
        } else { None };
        let tmp_words: i64 = 1 << 25;
        let mut tmp = [0u64; 3];
        for t in tmp.iter_mut() { checked("cuMemAlloc dot temp", (cuda.allocate)(t, tmp_words as usize * 4))?; }
        let (mut part, mut fail) = (0u64, 0u64);
        checked("cuMemAlloc partials", (cuda.allocate)(&mut part, 1024 * 3 * 8))?;
        checked("cuMemAlloc failure flag", (cuda.allocate)(&mut fail, 4))?;
        dev::zero(cuda, fail, 4)?;
        let mut fixed = std::collections::HashSet::new();
        let catalog = Path::new(req(a, "config")?).parent().unwrap_or(Path::new(".")).join("catalog-a.json");
        if let Ok(bytes) = fs::read(&catalog) {
            let c: Value = serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", catalog.display()))?;
            for x in c["allocations"].as_array().into_iter().flatten() {
                let (Some(id), Some(init), Some(b)) = (x["id"].as_u64(), x["initial_sha256"].as_str(), x["boundary_sha256"].as_object()) else { continue };
                if !b.is_empty() && b.values().all(|v| v.as_str() == Some(init)) { fixed.insert(id); }
            }
        }
        let loaded = kernels.values().filter(|k| k.unsupported.is_none()).count();
        eprintln!("{}", json!({"event":"window_loaded","kernels":kernels.len(),"dispatchable":loaded,"shadow_words":words,
            "shadow_bytes":words * 4,"allocations":sizes.len(),"tape_bytes":tape.bytes,"scratch_bytes":scratch.bytes,
            "refuse":refuse,"held":held,"fixed_allocations":fixed.len(),"fct_frozen":fct_frozen,
            "native_schemes":native.as_ref().map(|n| n.loaded.clone()),
            "tempo_live":tempo.as_ref().map(|t| json!({"tempo_adw":t.0.image_sha256,"tempo_adw_tlf":t.0.tlf.as_ref().map(|x| x.image_sha256.clone()),"arena_bytes":t.1.bytes})),
            "tempo_precip":tempo_serial.as_ref().map(|t| t.image_sha256.clone()),
            "tempo_small":tempo_small.as_ref().map(|t| json!({"functions":tempo_small_fns.iter().collect::<Vec<_>>(),"image_sha256":t.image_sha256}))}));
        Ok(Self { peer_buf: (0, 0), concat_types: HashMap::new(), kernels, classes, schemes, refuse, base_el, sh_base, words, shadows, support: dev::Support::load(cuda)?,
                  ledger: Some(dev::Ledger::new(cuda, 8 << 20)?), tempo_io: (0, 0), tempo_rows: Vec::new(), sync_owed: false, f,
                  tape, scratch, tmp, tmp_words, part, fail, gathers: HashMap::new(), counting: false, labels: BTreeMap::new(),
                  held_dropped: BTreeMap::new(), entry_ms: 0.0, writer: None,
                  trace: opt(a, "trace-norm") == Some("1"), trace_last: 0.0, trace_hits: 0, fixed,
                  trace_growth: std::env::var("AD_TRACE_GROWTH").ok().and_then(|v| v.parse().ok()).unwrap_or(100.0),
                  trace_cap: std::env::var("AD_TRACE_HITS").ok().and_then(|v| v.parse().ok()).unwrap_or(40),
                  trace_state: std::env::var("AD_TRACE_STATE").ok().as_deref() == Some("1"), state_words,
                  step_norms: std::env::var("AD_STEP_NORMS").ok().as_deref() == Some("1") || opt(a, "trace-norm") == Some("1"),
                  field_y: matches!(opt(a, "direction"), Some("field") | Some("fieldxy")), y_path: None,
                  field_x: opt(a, "direction") == Some("fieldxy"),
                  // --x-scale c (default 1, ad/g1-merged): every x-side direction word times c > 0. The operator is
                  // unchanged and rel = |<y,Mx> - <M^T y,x>| / max is invariant to c; it keeps a tangent that grows
                  // 1e18-fold over the window (all entries live, TEMPO warp_steps 5.2e4x at one launch) inside float32.
                  // --y-scale c (default 1): the same on the y side (the window-end weights that seed <y, Mx> and the reverse).
                  y_scale: match opt(a, "y-scale") { None => 1.0, Some(v) => match v.parse::<f32>() {
                      Ok(c) if c.is_finite() && c > 0.0 => c, _ => return Err(format!("--y-scale {v}: a finite number > 0")) } },
                  x_scale: match opt(a, "x-scale") { None => 1.0, Some(v) => match v.parse::<f32>() {
                      Ok(c) if c.is_finite() && c > 0.0 => c, _ => return Err(format!("--x-scale {v}: a finite number > 0")) } },
                  stable_dir: opt(a, "stable-dir") == Some("1"),
                  relocs: plan.allocations.iter().filter(|x| !x.relocations.is_empty())
                      .map(|x| (x.id, x.relocations.iter().map(|r| (r.byte_offset, 8)).collect())).collect(),
                  residue: ex.residue.iter().map(|(k, v)| (*k, v.0.clone())).collect(),
                  fct_frozen, taylor_mask: false, address_bearing: ex.address_bearing.clone(), native, tempo, tempo_stats: BTreeMap::new(), tempo_serial,
                  tempo_small, tempo_small_fns, tempo_small_identity_words: 0, blob_next: Vec::new(), blob_ahead: std::collections::HashSet::new(), capture_pin: None, cmp_pin: None, restore_pin: None, rv_restore_s: 0.0, rv_tape_s: 0.0, rv_redo_dl_s: 0.0, rv_redo_wait_s: 0.0, rv_walk_s: 0.0, load_next: None,
                  ckpt_base: None, delta_store: false, bs_now: 0, bs_peak: 0, bs_written: 0, bs_files: 0, bs_files_peak: 0, bs_ram_budget: 0, bs_ram_now: 0, bs_ram_peak: 0, bs_ram_count: 0, restore_direct: 0, wcache: std::sync::Mutex::new(HashMap::new()), pin_next: None, pin_ahead: false, pin_ahead_hits: 0, pin_q: std::collections::VecDeque::new(), pin_q_made: 0, gather_pass_join: false,
                  recompute_steps: 0, recompute_s: 0.0, ck_pf: None, load_s: 0.0, store_s: 0.0, pf_hits: 0, anchor_pool: Vec::new(), anchor_ahead: None, anchor_ahead_wait_s: 0.0, anchor_ahead_taken: 0, anchor_ahead_info: Vec::new(), dev_anchors: Vec::new(), restore_s: 0.0, params,
                  one_pass: opt(a, "one-pass") != Some("0"), one_pass_fallbacks: 0, tape_grown: None, tape_max: (gib("tape-max-gib", 16.0) * (1u64 << 30) as f64) as usize, tl_tape_grown: Vec::new(), pack: std::env::var("AD_TAPE_PACK").as_deref() != Ok("0"), precache: deriv::PreCache::default(), rows: BTreeMap::new(), rev_rows: BTreeMap::new(), seg_rows: BTreeMap::new(), census: Census::from_env(), win: None, win_cap: std::env::var("AD_TAPE_WINCACHE_MIB").ok().and_then(|v| v.parse::<usize>().ok()).map(|m| m << 20).unwrap_or(usize::MAX), win_promoted: (0, 0), rev_scratch: 0, scratch_base: 0, pack_ratio: 0.0, seg_ratio: 0.0, fn_ms: BTreeMap::new(), capture: false, capture_host: None, pending: Vec::new(), defer: false, defer_step: 0, tempo_sized: None,
                  fuse: std::env::var("AD_FUSE_TAPE").as_deref() == Ok("1") || opt(a, "fuse-tape") == Some("1"), fuse_rec: false, fuse_ok: true, rtape: None, rule_cache: None,
                  fz_cur: None, fz_tapes: BTreeMap::new(), fz_chunks: Vec::new(), fz_resident: None, fz_last_bytes: 0, fz_step_start: 0, fz_info: Vec::new(),
                  fz_d2h_s: 0.0, fz_d2h_bytes: 0, fz_failed: false, fz_force_flush: false, fz_steps: 0 })
    }
    /// G5: make `want` the cache: keep held entries, take the background decode, decode the rest on all cores, drop
    /// what this step does not use (bounded host memory); then start decoding `next` in the background.
    unsafe fn stage_blobs(&mut self, ex: &mut Execution, base: &Path, want: Vec<crate::plan::Blob>,
                          next: Option<Vec<crate::plan::Blob>>) -> Result<(), String> {
        let mut old = ex.blob_cache.take().unwrap_or_default();
        // integ8: join the decodes that hold a blob this step needs (and any already done); the rest stay in flight.
        let wanted: std::collections::HashSet<String> = want.iter().map(|b| crate::blob_sha(b).to_owned()).collect();
        for (set, h) in std::mem::take(&mut self.blob_next) {
            if h.is_finished() || set.iter().any(|x| wanted.contains(x)) {
                for (k, v) in h.join().map_err(|_| "blob prefetch panicked")?? { old.insert(k, std::sync::Arc::new(v)); }
            } else { self.blob_next.push((set, h)); }
        }
        let mut keep: HashMap<String, std::sync::Arc<Vec<u8>>> = HashMap::new();
        let mut missing = Vec::new();
        for b in want {
            let s = crate::blob_sha(&b).to_owned();
            if keep.contains_key(&s) { continue; }
            match old.remove(&s) { Some(v) => { keep.insert(s, v); }, None => missing.push(b) }
        }
        // integ8: blobs already decoded for the next forward steps stay (bounded: at most two steps ahead).
        for (k, v) in old { if self.blob_ahead.contains(&k) { keep.insert(k, v); } }
        for (k, v) in decode_blobs(base, &missing)? { keep.insert(k, std::sync::Arc::new(v)); }
        ex.blob_cache = Some(keep);
        if let Some(n) = next.filter(|n| !n.is_empty()) {
            let base = base.to_owned();
            let set = n.iter().map(|b| crate::blob_sha(b).to_owned()).collect();
            self.blob_next.push((set, std::thread::spawn(move || decode_blobs(&base, &n))));
        }
        Ok(())
    }
    /// G5: decode `want` in the background (merged by the next stage_blobs). An earlier unfinished prefetch is kept:
    /// its results are merged into the new one when it ends.
    unsafe fn prefetch_blobs(&mut self, base: &Path, want: Vec<crate::plan::Blob>) -> Result<(), String> {
        // integ8: skip blobs already being decoded; one batch per call (stage_blobs joins only what it needs).
        let in_flight: std::collections::HashSet<String> = self.blob_next.iter().flat_map(|(s, _)| s.iter().cloned()).collect();
        let want: Vec<crate::plan::Blob> = want.into_iter().filter(|b| !in_flight.contains(crate::blob_sha(b))).collect();
        if want.is_empty() { return Ok(()); }
        let base = base.to_owned();
        let set = want.iter().map(|b| crate::blob_sha(b).to_owned()).collect();
        self.blob_next.push((set, std::thread::spawn(move || decode_blobs(&base, &want))));
        Ok(())
    }
    /// G5: boundary `i`'s bytes, from the background read when one was started for it.
    unsafe fn load_at(&mut self, bounds: &[Boundary], i: usize) -> Result<Bytes, String> {
        if let Some((k, h)) = self.load_next.take() {
            let d = h.join().map_err(|_| "boundary prefetch panicked")?;
            if k == i {
                if !bounds[i].delta { return d.map(Bytes::Own); }
                let (bi, bd) = self.ckpt_base.as_ref().ok_or("changed-chunk boundary without boundary 0")?;
                return delta_decode(&d?, &bounds[i].index, bi, bd).map(Bytes::Own);
            }
        }
        self.load(&bounds[i])
    }
    unsafe fn prefetch_load(&mut self, bounds: &[Boundary], i: usize) -> Result<(), String> {
        if bounds[i].ram.is_some() || bounds[i].dev != 0 { return Ok(()); } // host-leftovers-2: RAM boundary, no file to read
        self.join_writer()?;
        let p = bounds[i].path.clone();
        self.load_next = Some((i, std::thread::spawn(move || fs::read(&p).map_err(|e| format!("{}: {e}", p.display())))));
        Ok(())
    }
    /// lane rung3-params: upload every family's carrier. Values are p0 except `val` (index, value); tangents are
    /// tl[k] * p0_k (a direction in d ln p) or 0.
    unsafe fn par_set(&self, cuda: &Cuda, val: Option<(usize, f32)>, tl: Option<&[f64]>) -> Result<(), String> {
        let Some(n) = self.native.as_ref() else { return Ok(()) };
        let mut fams: Vec<&str> = self.params.iter().map(|p| p.fam.as_str()).collect();
        fams.sort(); fams.dedup();
        for f in fams {
            let (mut on, mut v, mut t) = ([0i32; 8], [0f32; 8], [0f64; 8]);
            for (k, p) in self.params.iter().enumerate().filter(|(_, p)| p.fam == f) {
                on[p.slot] = 1;
                v[p.slot] = match val { Some((i, x)) if i == k => x, _ => p.p0 };
                t[p.slot] = tl.map(|c| c.get(k).copied().unwrap_or(0.0) * p.p0 as f64).unwrap_or(0.0);
            }
            n.par_upload(cuda, f, &on, &v, &t)?;
        }
        Ok(())
    }
    fn ctx(&self) -> Ctx<'_> {
        Ctx { base_el: &self.base_el, kernels: &self.kernels, classes: &self.classes, schemes: &self.schemes, refuse: &self.refuse,
              fct_frozen: self.fct_frozen }
    }
    fn sh(&self, el: i64) -> u64 { self.sh_base + 4 * el as u64 }
    /// packed-tape: add one launch to the current tangent step's per-row tape bytes.
    fn row(&mut self, f: &str, v: [usize; 5]) {
        let e = self.rows.entry(f.to_owned()).or_insert([0; 6]);
        e[0] += 1; for k in 0..5 { e[k + 1] += v[k]; }
    }
    /// packed-tape: the per-row tape bytes of the tangent step just done (the reverse tapes the same launches), largest
    /// first, with totals; then cleared.
    fn take_rows(&mut self, cells: usize) -> Value {
        let rows = std::mem::take(&mut self.rows);
        let mut v: Vec<(String, [usize; 6])> = rows.into_iter().collect();
        let tot = |r: &[usize; 6]| r[1] + r[2] + r[3] + r[4] + r[5];
        v.sort_by(|a, b| tot(&b.1).cmp(&tot(&a.1)));
        let mut sum = [0usize; 6];
        for (_, r) in &v { for k in 0..6 { sum[k] += r[k]; } }
        let all = tot(&sum);
        json!({"fields":"[launches, pre-launch primal bytes, shadows/cells/tables bytes, transient bytes, post-launch primal bytes, other (rules, native, TEMPO snapshots) bytes]",
               "totals":sum,"total_bytes":all,"tape_bytes_unpacked_old_layout":all,"tape_bytes_without_transient":all - sum[3],
               "bytes_per_cell":if cells > 0 { all as f64 / cells as f64 } else { 0.0 },
               "rows":v.iter().take(60).map(|(f, r)| json!([f, r, tot(r)])).collect::<Vec<_>>()})
    }
    fn pack_json(&mut self) -> Value {
        let c = &self.precache;
        let v = json!({"on":self.pack,"checks":c.checks,"check_bytes":c.check_bytes,"shared":c.hits,"shared_bytes":c.hit_bytes,
                       "unshared_after_forward":c.unshared,"unshared_bytes":c.unshared_bytes,"tape_used_bytes":self.tape.used,
                       "window_cache":{"hits":c.win_hits,"hit_bytes":c.win_hit_bytes,"misses":c.win_misses,"promoted":self.win_promoted.0,"promoted_bytes":self.win_promoted.1,"arena_used_bytes":self.win.as_ref().map(|a| a.used).unwrap_or(0)},
                       "blocks":{"block_bytes":deriv::tape_block_bytes(),"spans":c.blk_spans,"checks":c.blk_checks,"equal_bytes":c.blk_hit_bytes,"fresh_bytes":c.blk_fresh_bytes},
                       "scratch_bytes":self.scratch.bytes,"scratch_peak_bytes":self.scratch.peak});
        // tape-dedupe-and-pack: the window cache outlives the step's counters.
        let win = std::mem::take(&mut self.precache.win);
        self.precache = deriv::PreCache::default();
        self.precache.win = win;
        v
    }
    /// packed-tape: the reverse scratch holds one launch's transient buffers + workspace (measured in the tangent pass).
    /// Grown, never shrunk, only between launches (nothing lives in the scratch across a launch's walk-back).
    unsafe fn fit_scratch(&mut self, cuda: &Cuda) -> Result<(), String> {
        if self.scratch_base == 0 { self.scratch_base = self.scratch.bytes; }
        let want = self.scratch_base + self.rev_scratch + (64 << 20);
        if !self.pack || want <= self.scratch.bytes { return Ok(()); }
        dev::sync(cuda)?;
        let before = self.scratch.bytes;
        checked("cuMemFree scratch", (cuda.free)(self.scratch.base))?;
        self.scratch = match dev::Arena::new(cuda, want) { Ok(s) => s, Err(e) => { self.scratch = dev::Arena::new(cuda, before)?;
            return Err(format!("packed-tape: the reverse scratch needs {want} bytes (one launch's transient + workspace), card refused: {e}")) } };
        eprintln!("{}", json!({"event":"window_scratch_grown","before":before,"bytes":want}));
        Ok(())
    }
    fn note(&mut self, pass: &str, f: &str, t0: Instant) {
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        self.entry_ms += ms;
        let e = self.fn_ms.entry((pass.to_owned(), f.to_owned())).or_insert((0.0, 0));
        e.0 += ms; e.1 += 1;
    }

    /// Rule of a command (and, for a kernel, its resolved launch). Row rules whose transpose cannot be built
    /// deterministically are refused here, so tangent and reverse agree.
    unsafe fn rule(&mut self, ex: &Execution, op: &Operation, step: usize, base: &Path, frame: Option<&ReplayFrame>)
                   -> Result<(R, Option<(String, Vec<submit::Argument>, [u32; 3], [u32; 3], u32)>), String> {
        // x20 fused tape: the fused tangent resolved this command's rule once; the tangent takes it unchanged.
        if let Some(c) = self.rule_cache.take() { return Ok(c); }
        let t = dev::prof::start();
        let r = self.rule_inner(ex, op, step, base, frame);
        dev::prof::add(4, t);
        r
    }
    unsafe fn rule_inner(&mut self, ex: &Execution, op: &Operation, step: usize, base: &Path, frame: Option<&ReplayFrame>)
                   -> Result<(R, Option<(String, Vec<submit::Argument>, [u32; 3], [u32; 3], u32)>), String> {
        let tcl = dev::prof::start();
        let (r, l) = match op {
            Operation::Kernel { module, function, grid, block, shared_bytes, arguments, .. } => {
                let args = arguments.iter().map(|x| ex.argument(x, step, frame)).collect::<Result<Vec<_>, _>>()?;
                let live_row = self.native.as_ref().is_some_and(|n| n.physics_row(function)) && !self.refuse.contains_key(function.as_str());
                let tempo_row = function == "tempo_warp_steps" && self.tempo.is_some() && !self.refuse.contains_key(function.as_str());
                let serial_row = self.tempo_serial.is_some() && function == "tempo_precip" && !self.refuse.contains_key(function.as_str());
                let small = self.tempo_small.is_some() && self.tempo_small_fns.contains(function.as_str()) && !self.refuse.contains_key(function.as_str());
                let r = if tempo_row { R::Tempo(function.clone()) }
                        else if serial_row { R::TempoSerial(function.clone()) }
                        else if live_row { R::Native(function.clone()) } else if small { R::TempoSmall(function.clone()) }
                        // step6-sw: concatenate is bound by its module's element type, never by name (wrules::concatenate).
                        else if function == "cupy_concatenate_same_size" && !self.refuse.contains_key(function.as_str()) {
                            self.ctx().concatenate(ex, function, &args, self.concat_types.get(module).map(String::as_str))
                        }
                        else { self.ctx().kernel(ex, function, &args, *grid, *block)? };
                (r, Some((function.clone(), args, *grid, *block, *shared_bytes)))
            },
            Operation::PeerSend { channel, source, bytes, .. } | Operation::PeerRecv { channel, destination: source, bytes, .. } => {
                // multidevice-runner: the halves of a cross-rank copy. Unaligned words cannot move as float shadow words.
                if bytes % 4 != 0 || source.offset % 4 != 0 { return Err(format!("step {step}: peer copy channel {channel} is not word aligned")); }
                let (e, n, key) = (self.ctx().el(source.allocation, source.offset)?, bytes / 4, crate::peer::key(step, *channel, &ex.peer_scope()));
                (if matches!(op, Operation::PeerSend { .. }) { R::PeerSend { src: e, n, key } } else { R::PeerRecv { dst: e, n, key } }, None)
            },
            _ => (self.ctx().command(ex, op, base, frame)?, None),
        };
        dev::prof::add(12, tcl);
        if let R::Rows { rows, label } = &r {
            let tg = dev::prof::start();
            let g = if self.defer && std::env::var("AD_GATHER_SYNC").as_deref() != Ok("1") { self.gather_defer(rows, label); Ok(false) }
                    else { self.gather_for(&ex.cuda, rows) };
            dev::prof::add(13, tg);
            if let Err(e) = g { return Ok((R::Refused(format!("{label}: {e}")), l)); }
        }
        if self.counting {
            let name = match (&r, &l) {
                (R::Entry, Some(l)) => format!("entry:{}", l.0),
                (R::Rows { label, .. }, _) => format!("rule:rows:{label}"),
                (R::View { .. }, Some(l)) => format!("rule:view:{}", l.0),
                (R::ZeroView { label, .. }, _) => format!("rule:zero-view:{label}"),
                (R::Max { .. }, _) => "rule:max".into(), (R::Where { .. }, _) => "rule:where".into(),
                (R::Scale { .. }, _) => "rule:scale".into(), (R::Copy { .. }, _) => "rule:copy".into(),
                (R::PeerSend { .. }, _) => "rule:peer-send".into(), (R::PeerRecv { .. }, _) => "rule:peer-recv".into(),
                (R::Zero { label, .. }, _) => format!("rule:zero:{label}"),
                (R::Held { origin, .. }, _) => format!("held-upload:{origin}"),
                (R::Identity(s), _) => s.clone(), (R::Refused(s), _) => format!("REFUSED:{s}"),
                (R::FctMinmax { .. }, _) => "rule:fct-frozen:fct_minmax_source (source_old exact, limiter bounds constant)".into(),
                (R::Native(f), _) => format!("native:{}:{f}", self.native.as_ref().and_then(|n| n.row(f)).map(|r| r.unit.clone()).unwrap_or_default()),
                (R::Tempo(f), _) => format!("tempo-adw:{f}"),
                (R::TempoSerial(f), _) => format!("tempo-serial:{f} (P5 per-thread entry, float64 tape)"),
                (R::TempoSmall(f), _) => format!("tempo-small:P5:{f}"),
                _ => "?".into(),
            };
            *self.labels.entry(name).or_default() += 1;
        }
        Ok((r, l))
    }
    fn rows_hash(rows: &[DevRow]) -> u64 {
        let bytes = unsafe { std::slice::from_raw_parts(rows.as_ptr().cast::<u8>(), std::mem::size_of_val(rows)) };
        let mut h = 0xcbf29ce484222325u64;
        for b in bytes { h ^= *b as u64; h = h.wrapping_mul(0x100000001b3); }
        h
    }
    /// adjoint-speed: start building a missing gather on a worker thread (tangent pass only).
    fn gather_defer(&mut self, rows: &[DevRow], label: &str) {
        let h = Self::rows_hash(rows);
        if self.gathers.contains_key(&h) || self.pending.iter().any(|p| p.0 == h) { return; }
        let r = rows.to_vec();
        let st = self.defer_step;
        self.pending.push((h, label.to_owned(), std::thread::spawn(move || wrules::gather(&r)), st));
    }
    /// adjoint-speed: install every gather built during this tangent step (uploads on the dispatch thread, in the order
    /// the misses occurred). A gather that failed stops the run here: the tangent already used that row.
    unsafe fn join_pending(&mut self, cuda: &Cuda, step: usize) -> Result<(), String> { self.join_pending_if(cuda, step, |_| true) }
    /// tempo-once: install the pending gathers `keep` selects (in the order the misses occurred); the others keep building.
    unsafe fn join_pending_if(&mut self, cuda: &Cuda, _step: usize, keep: impl Fn(&(u64, String, std::thread::JoinHandle<Result<wrules::Gather, String>>, usize)) -> bool)
                              -> Result<(), String> {
        if self.pending.is_empty() { return Ok(()); }
        let t = dev::prof::start();
        let (now, later): (Vec<_>, Vec<_>) = std::mem::take(&mut self.pending).into_iter().partition(|p| keep(p));
        self.pending = later;
        for (h, label, w, step) in now {
            let g = w.join().map_err(|_| "gather worker panicked")?;
            if let Err(e) = &g {
                return Err(format!("step {step}: copy-row gather {label} refused ({e}) after the tangent used the row; AD_GATHER_SYNC=1 refuses it in the tangent instead"));
            }
            self.install_gather(cuda, h, g)?;
        }
        dev::prof::add(13, t);
        Ok(())
    }
    unsafe fn gather_for(&mut self, cuda: &Cuda, rows: &[DevRow]) -> Result<bool, String> {
        let h = Self::rows_hash(rows);
        if self.pending.iter().any(|p| p.0 == h) { let s0 = self.defer_step; self.join_pending_if(cuda, s0, |p| p.0 == h)?; }
        if !self.gathers.contains_key(&h) {
            let tm = Instant::now();
            let g0 = wrules::gather(rows);
            if dev::prof::on() {
                let words: i64 = rows.iter().map(|r| r.count).sum();
                eprintln!("{}", json!({"event":"window_gather_miss","rows":rows.len(),"words":words,
                    "injective":g0.as_ref().map(|x| x.injective).ok(),"ms":tm.elapsed().as_secs_f64() * 1e3,"cached":self.gathers.len()}));
            }
            self.install_gather(cuda, h, g0)?;
        }
        match &self.gathers[&h] { Ok(g) => Ok(g.injective), Err(e) => Err(e.clone()) }
    }
    unsafe fn install_gather(&mut self, cuda: &Cuda, h: u64, g0: Result<wrules::Gather, String>) -> Result<(), String> {
        {
            let g = match g0 {
                Err(e) => Err(e),
                Ok(g) if g.injective => Ok(GatherDev { injective: true, bufs: [0; 5], nsrc: 0, ndst: 0 }),
                Ok(g) => {
                    let mut bufs = [0u64; 5];
                    let parts: [&[u8]; 5] = [
                        std::slice::from_raw_parts(g.src_word.as_ptr().cast(), g.src_word.len() * 8),
                        std::slice::from_raw_parts(g.start.as_ptr().cast(), g.start.len() * 8),
                        std::slice::from_raw_parts(g.item.as_ptr().cast(), g.item.len() * 8),
                        std::slice::from_raw_parts(g.dst_word.as_ptr().cast(), g.dst_word.len() * 8), &g.sourceless];
                    for (b, p) in bufs.iter_mut().zip(parts) {
                        checked("cuMemAlloc gather", (cuda.allocate)(b, p.len().max(8)))?;
                        dev::upload(cuda, *b, p)?;
                    }
                    Ok(GatherDev { injective: false, bufs, nsrc: g.src_word.len() as i64, ndst: g.dst_word.len() as i64 })
                },
            };
            self.gathers.insert(h, g);
        }
        Ok(())
    }
    unsafe fn rows_dev(&mut self, cuda: &Cuda, rows: &[DevRow]) -> Result<u64, String> {
        self.scratch.reset();
        let bytes = std::slice::from_raw_parts(rows.as_ptr().cast::<u8>(), std::mem::size_of_val(rows));
        let p = self.scratch.take(bytes.len())?;
        dev::upload(cuda, p, bytes)?;
        Ok(p)
    }
    /// Copy a primal span (allocation, first unit, units, unit bytes) into the tape.
    unsafe fn tape_primal(&mut self, ex: &Execution, alloc: u64, first: usize, units: usize, unit: usize) -> Result<u64, String> {
        let a = ex.allocations.get(&alloc).ok_or("primal allocation not live")?;
        let (off, bytes) = (first * unit, units * unit);
        if off + bytes > a.bytes { return Err("primal span outside its allocation".into()); }
        let p = self.tape.take(bytes)?;
        dev::copy(&ex.cuda, p, a.address + off as u64, bytes)?;
        Ok(p)
    }

    /// Frozen fct_minmax_source: tape its primal inputs source and rho_old (contiguous: source, then rho_old at the next
    /// 256-byte boundary) before the launch.
    unsafe fn tape_fct(&mut self, ex: &Execution, prim: &[(u64, usize, usize); 2]) -> Result<u64, String> {
        let b0 = (prim[0].2 * 4).div_ceil(256) * 256;
        let p = self.tape.take(b0 + prim[1].2 * 4)?;
        for (k, (al, first, n)) in prim.iter().enumerate() {
            let a = ex.allocations.get(al).ok_or("fct-frozen primal allocation not live")?;
            if (first + n) * 4 > a.bytes { return Err("fct-frozen primal span outside its allocation".into()); }
            dev::copy(&ex.cuda, p + if k == 0 { 0 } else { b0 as u64 }, a.address + 4 * *first as u64, n * 4)?;
        }
        Ok(p)
    }
    unsafe fn fct_minmax(&self, cuda: &Cuda, r: &R, prim: u64, adjoint: bool) -> Result<(), String> {
        let R::FctMinmax { nt, nl, nc, dt, e, prim: spans } = r else { return Err("not a frozen minmax rule".into()) };
        let (mut a, mut nt, mut nl, mut nc, mut dt) = (self.sh_base, *nt, *nl, *nc, *dt);
        let (mut src, mut rho) = (prim, prim + ((spans[0].2 * 4).div_ceil(256) * 256) as u64);
        let [mut e0, mut e1, mut e2, mut e3, mut e4, mut e5] = *e;
        let f = if adjoint { self.f.fct_minmax_ad } else { self.f.fct_minmax_tl };
        launch(cuda, f, grid1(nl as i64 * nc as i64), kargs!(a, nt, nl, nc, dt, src, rho, e0, e1, e2, e3, e4, e5))
    }

    /// tempo_warp_steps live in the window (--tempo-live 1). The launch reads f[0..17] (nz x columns each) and s[0..9]
    /// (columns each) through its pointer tables and writes in place f[0] (theta), f[4..16], s[0..9] and the profile
    /// outputs p[0] (p[1..3] with radii); f[1..3] and f[17] are read only.
    /// Tangent: x-dot gathered from the input shadows, y-dot = J x-dot scattered into the output shadows (read-only
    /// inputs keep their tangent). Adjoint: y-bar gathered from the output shadows, the output shadows cleared, then
    /// x-bar = J^T y-bar added into the input shadows. Float64 in the derivative, rounded to the float32 shadows.
    unsafe fn tempo_apply(&mut self, ex: &Execution, s: &crate::tempo_adw::Snap, step: usize, adjoint: bool) -> Result<(), String> {
        let cuda = &ex.cuda;
        let (nz, nc) = (s.nz, s.columns);
        let nin = 18 * nz + 10;
        let shb = self.sh_base;
        let base_el = &self.base_el;
        let el = |p: u64, n: usize| -> Result<i64, String> {
            let a = ex.allocations.values().find(|a| p >= a.address && p + 4 * n as u64 <= a.address + a.bytes as u64)
                .ok_or_else(|| format!("TEMPO pointer {p:#x} ({n} words) is in no live allocation"))?;
            if (p - a.address) % 4 != 0 { return Err(format!("TEMPO pointer {p:#x} is not word aligned in allocation {}", a.id)); }
            Ok(base_el.get(&a.id).ok_or_else(|| format!("TEMPO allocation {} has no window shadow", a.id))? + ((p - a.address) / 4) as i64)
        };
        // (first slot, shadow element, levels per slot group): words = levels * columns, word (k, c) at k * nc + c.
        let mut ins: Vec<(usize, i64, usize)> = vec![];
        for j in 0..18 { if s.fpost[j] != 0 { ins.push((j * nz, el(s.fpost[j], nz * nc)?, nz)); } }
        for j in 0..10 { if s.spost[j] != 0 { ins.push((18 * nz + j, el(s.spost[j], nc)?, 1)); } }
        let mut outs: Vec<(usize, i64, usize)> = vec![];
        if s.fpost[0] != 0 { outs.push((0, el(s.fpost[0], nz * nc)?, nz)); }
        for j in 4..17 { if s.fpost[j] != 0 { outs.push(((j - 3) * nz, el(s.fpost[j], nz * nc)?, nz)); } }
        for j in 0..10 { if s.spost[j] != 0 { outs.push((14 * nz + j, el(s.spost[j], nc)?, 1)); } }
        for p in 0..4 { if (p == 0 || s.flags[6] != 0) && s.ppost[p] != 0 { outs.push((14 * nz + 10 + p * nz, el(s.ppost[p], nz * nc)?, nz)); } }
        if !adjoint && std::env::var_os("AD_TEMPO_PROFILE_LOG").is_some() {
            // j4-repose: where the profile outputs (p[0] = refl10cm, dBZ) live, by plan allocation and byte offset.
            let at = |p: u64| ex.allocations.values().find(|a| p >= a.address && p < a.address + a.bytes as u64).map(|a| json!([a.id, p - a.address]));
            eprintln!("{}", json!({"event":"window_tempo_profile","step":step,"nz":nz,"columns":nc,"flags6":s.flags[6],
                                   "profile":(0..4).map(|p| at(s.ppost[p])).collect::<Vec<_>>(),"f":(0..18).map(|j| at(s.fpost[j])).collect::<Vec<_>>()}));
        }
        // Output spans must not overlap one another (each output word is written once by the launch).
        let mut spans: Vec<(i64, i64)> = outs.iter().map(|(_, e, l)| (*e, *e + (*l * nc) as i64)).collect();
        spans.sort_unstable();
        if spans.windows(2).any(|w| w[1].0 < w[0].1) { return Err("TEMPO output arrays overlap: the in-place scatter would be ambiguous".into()); }
        // readbacks-device: the input vector is gathered and widened on the device (it was downloaded slot by slot into a
        // host float64 vector), the derivative reads and writes device vectors, and the result is added into the
        // shadows and rounded once on the device (it was downloaded and accumulated on the host), the same conversions.
        // The non-finite check and the receipt norms (fixed-order device sums now, not host sequential sums: the norms
        // in window_tempo may differ in the last digits, the shadow words do not) go to the step ledger.
        let t0 = Instant::now();
        let (g, side) = if !adjoint { (&ins, &outs) } else { (&outs, &ins) };
        let nv = nin * nc;
        if self.tempo_io.1 < 16 * nv {
            if self.tempo_io.0 != 0 { dev::sync(cuda)?; checked("cuMemFree tempo io", (cuda.free)(self.tempo_io.0))?; }
            let mut a = 0u64;
            checked("cuMemAlloc tempo io", (cuda.allocate)(&mut a, 16 * nv))?;
            self.tempo_io = (a, 16 * nv);
        }
        let (vd, rd) = (self.tempo_io.0, self.tempo_io.0 + 8 * nv as u64);
        dev::zero(cuda, vd, 8 * nv)?;
        let led = self.ledger.as_mut().ok_or("step ledger missing")?;
        for (q0, e, lv) in g.iter() { led.widen(cuda, shb + 4 * *e as u64, vd + 8 * (q0 * nc) as u64, lv * nc)?; }
        let tin = led.sq_sum_f64(cuda, vd, nv)?;
        if adjoint { for (_, e, lv) in outs.iter() { dev::zero(cuda, shb + 4 * *e as u64, 4 * lv * nc)?; } }
        if self.tempo_sized.is_none() {
            let ar = &mut self.tempo.as_mut().ok_or("TEMPO derivative not loaded")?.1;
            let (mut free, mut total) = (0usize, 0usize);
            checked("cuMemGetInfo", (cuda.mem_info)(&mut free, &mut total))?;
            let want = (std::env::var("AD_TEMPO_ARENA_GIB").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(16.0) * (1u64 << 30) as f64) as usize;
            let before = ar.bytes;
            let target = want.min(before + free.saturating_sub(16usize << 30));
            let grown = target > before;
            if grown {
                dev::sync(cuda)?;
                checked("cuMemFree tempo arena", (cuda.free)(ar.base))?;
                *ar = dev::Arena::new(cuda, target)?;
            }
            self.tempo_sized = Some(json!({"arena_bytes_before":before,"arena_bytes":ar.bytes,"grown":grown,"card_free_bytes":free}));
            eprintln!("{}", json!({"event":"window_tempo_arena","info":self.tempo_sized}));
        }
        let (t, ar) = self.tempo.as_mut().ok_or("TEMPO derivative not loaded")?;
        let led = self.ledger.as_mut().ok_or("step ledger missing")?;
        let (itok, icmp, tiles) = t.apply(cuda, ar, s, vd, rd, adjoint, !adjoint, led)?;
        let parts = t.parts.get();
        let peak = ar.peak as f64;
        let what = if adjoint { "cotangent" } else { "tangent" };
        let mut touts = Vec::with_capacity(side.len());
        for (q0, e, lv) in side.iter() {
            touts.push(led.add_round_sq(cuda, shb + 4 * *e as u64, rd + 8 * (q0 * nc) as u64, lv * nc, adjoint,
                                        &format!("step {step}: non-finite TEMPO {what} at slot {q0}"))?);
        }
        dev::sync(cuda)?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let k = if adjoint { "ad" } else { "tl" };
        let e = self.tempo_stats.entry("arena_peak_bytes".into()).or_default(); *e = e.max(peak);
        self.tempo_rows.push((k, step, ms, tiles, tin, touts, itok, icmp, parts));
        Ok(())
    }

    /// readbacks-device: the step's one ledger read, then the receipt lines that waited for it (window_tempo rows,
    /// tempo_stats, and window_device_flags for nonzero non-fatal counters such as P3 domain flags). A fired item
    /// refuses the step here, with the text the per-entry check gave.
    unsafe fn step_flush(&mut self, cuda: &Cuda, step: usize, pass: &str) -> Result<(), String> {
        let led = self.ledger.as_mut().ok_or("step ledger missing")?;
        led.flush(cuda)?;
        let flags: Vec<Value> = led.take_report().into_iter().map(|(row, c)| json!({"row":row,"counts":c})).collect();
        if !flags.is_empty() {
            eprintln!("{}", json!({"event":"window_device_flags","pass":pass,"step":step,"fields":"[overflow, domain, context, arena]","rows":flags}));
        }
        for (k, st, ms, tiles, tin, touts, itok, icmp, parts) in std::mem::take(&mut self.tempo_rows) {
            let led = self.ledger.as_mut().unwrap();
            let vin_ss = led.value(tin).map(|v| v[0]).ok_or("TEMPO input norm not read")?;
            let mut vout_ss = 0f64;
            for t in &touts { vout_ss += led.value(*t).map(|v| v[0]).ok_or("TEMPO output norm not read")?; }
            let idiff = match itok { Some(t) => led.value(t).map(|v| v[0] as usize).ok_or("TEMPO identity count not read")?, None => 0 };
            for (name, x) in [("launches", 1.0), ("ms", ms), ("in_sq", vin_ss), ("out_sq", vout_ss), ("tiles", tiles as f64),
                              ("count_pass_ms", parts[0]), ("apply_rest_ms", parts[1]),
                              ("identity_differing_words", idiff as f64), ("identity_compared_words", icmp as f64)] {
                *self.tempo_stats.entry(format!("{k}_{name}")).or_default() += x;
            }
            eprintln!("{}", json!({"event":"window_tempo","pass":k,"step":st,"ms":ms,"tiles":tiles,"parts_ms":{"count_pass":parts[0],"rest_enqueue":parts[1]},"in_norm":vin_ss.sqrt(),"out_norm":vout_ss.sqrt(),
                                   "identity_differing_words":idiff,"identity_compared_words":icmp}));
        }
        Ok(())
    }

    /// Float32-shadow adapter: copy every active segment of a prepared L2 row between the window's float32 shadows and
    /// the row's float64 shadows (widen before the row's entries, round back after).
    unsafe fn xfer(&self, cuda: &Cuda, segs: &[(u64, usize, usize, u64)], widen: bool) -> Result<(), String> {
        let t = dev::prof::start();
        let r = self.xfer_inner(cuda, segs, widen);
        dev::prof::add(9, t);
        r
    }
    unsafe fn xfer_inner(&self, cuda: &Cuda, segs: &[(u64, usize, usize, u64)], widen: bool) -> Result<(), String> {
        for (al, lo, hi, d) in segs {
            if lo % 4 != 0 { return Err(format!("native segment of allocation {al} starts mid-word")); }
            let e0 = self.base_el.get(al).ok_or_else(|| format!("native segment: allocation {al} has no window shadow"))? + (*lo / 4) as i64;
            let (mut a, mut e, mut d, mut n) = (self.sh_base, e0, *d, (hi - lo).div_ceil(4) as i64);
            launch(cuda, if widen { self.f.widen } else { self.f.narrow }, [grid1(n)[0].min(65535), 1, 1], kargs!(a, e, d, n))?;
        }
        Ok(())
    }

    /// Shadow element of a device span (pointer, words): None for a null pointer; the span must lie in one live allocation.
    fn span_el(&self, live: &HashMap<u64, submit::Allocation>, p: u64, words: usize) -> Result<Option<i64>, String> {
        if p == 0 { return Ok(None); }
        for a in live.values() {
            if p >= a.address && p + 4 * words as u64 <= a.address + a.bytes as u64 {
                let off = p - a.address;
                if off % 4 != 0 { return Err(format!("span {p:#x} starts mid-word in allocation {}", a.id)); }
                let b = self.base_el.get(&a.id).ok_or_else(|| format!("allocation {} has no window shadow", a.id))?;
                return Ok(Some(b + (off / 4) as i64));
            }
        }
        Err(format!("span {p:#x} ({words} words) lies in no single live allocation"))
    }
    /// Slot blocks of a --tempo-small launch as shadow elements (read before the forward: the pointer tables are fixed).
    unsafe fn small_slots(&self, ex: &Execution, f: &str, args: &[submit::Argument])
                          -> Result<(Vec<(Option<i64>, usize)>, Vec<(Option<i64>, usize)>), String> {
        let ts = self.tempo_small.as_ref().ok_or("--tempo-small entry not loaded")?;
        let (ins, outs) = ts.window_slots(&ex.cuda, &ex.allocations, f, args)?;
        let map = |v: Vec<(u64, usize)>| -> Result<Vec<(Option<i64>, usize)>, String> {
            v.into_iter().map(|(p, w)| Ok((self.span_el(&ex.allocations, p, w)?, w))).collect() };
        Ok((map(ins)?, map(outs)?))
    }
    /// Tangent or adjoint of one --tempo-small launch from its snapshot: the tape is re-recorded from the snapshot
    /// (count pass, record pass; float64), the input tangents are widened from the shadows into the entry's
    /// [slot][column] buffer, swept, and rounded back into the output shadows. Adjoint: the output cotangents are
    /// widened and cleared (the forward overwrote those words), swept back, and added into the input cotangents in
    /// float64 (widen, add, one rounding). `check` (tangent, right after the forward) compares the recorded primal
    /// with the forward's post-state word for word and refuses on any difference.
    unsafe fn small_sweep(&mut self, cuda: &Cuda, f: &str, snap: &crate::tempo_adw::SmallSnap, ins: &[(Option<i64>, usize)],
                          outs: &[(Option<i64>, usize)], adjoint: bool, check: bool) -> Result<(), String> {
        let ts = self.tempo_small.as_ref().ok_or("--tempo-small entry not loaded")?;
        self.scratch.reset();
        let (rec, idiff) = ts.window_record(cuda, &mut self.scratch, snap, check)?;
        if idiff != 0 { return Err(format!("{f}: {idiff} of {} recorded primal words differ from the forward", rec.identity_compared)); }
        self.tempo_small_identity_words += rec.identity_compared as u64;
        let n = rec.n;
        let (win, wout): (usize, usize) = (ins.iter().map(|x| x.1).sum(), outs.iter().map(|x| x.1).sum());
        if win != rec.nin * n || wout != rec.nout * n { return Err(format!("{f}: slot words {win}/{wout} against the entry's {}x{n} / {}x{n}", rec.nin, rec.nout)); }
        let xb = self.scratch.take(8 * win)?; let yb = self.scratch.take(8 * wout)?;
        let a = self.sh_base;
        let span = |e: i64, d: u64, w: usize, widen: bool| -> Result<(), String> {
            let (mut a, mut e, mut d, mut n) = (a, e, d, w as i64);
            launch(cuda, if widen { self.f.widen } else { self.f.narrow }, [grid1(n)[0].min(65535), 1, 1], kargs!(a, e, d, n))
        };
        if !adjoint {
            let mut off = 0u64;
            for (el, w) in ins { match el { Some(e) => span(*e, xb + 8 * off, *w, true)?, None => dev::zero(cuda, xb + 8 * off, 8 * w)? } off += *w as u64; }
            ts.window_sweep(cuda, &rec, false, xb, yb)?;
            let mut off = 0u64;
            for (el, w) in outs { if let Some(e) = el { span(*e, yb + 8 * off, *w, false)?; } off += *w as u64; }
        } else {
            let mut off = 0u64;
            for (el, w) in outs {
                match el { Some(e) => { span(*e, yb + 8 * off, *w, true)?; dev::zero(cuda, a + 4 * *e as u64, 4 * w)?; }, None => dev::zero(cuda, yb + 8 * off, 8 * w)? }
                off += *w as u64;
            }
            ts.window_sweep(cuda, &rec, true, yb, xb)?;
            let wb = self.scratch.take(8 * win)?;
            let mut off = 0u64;
            for (el, w) in ins {
                if let Some(e) = el {
                    span(*e, wb + 8 * off, *w, true)?;
                    self.support.move_add(cuda, wb + 8 * off, xb + 8 * off, 8 * w, 8)?;
                    span(*e, wb + 8 * off, *w, false)?;
                }
                off += *w as u64;
            }
        }
        dev::sync(cuda)
    }

    /// Tangent of a non-entry rule (after the forward command ran). `prim` is the taped primal for max / where.
    unsafe fn tl(&mut self, cuda: &Cuda, r: &R, prim: u64) -> Result<(), String> {
        let mut a = self.sh_base;
        match r {
            R::Rows { rows, .. } => { let mut rd = self.rows_dev(cuda, rows)?; launch(cuda, self.f.rows_tl, grid_rows(rows), kargs!(a, rd))?; },
            R::View { dst, src } => { let (mut d, mut s) = (*dst, *src); launch(cuda, self.f.view_tl, grid1(d.size), kargs!(a, d, s))?; },
            R::ZeroView { dst, .. } => { let mut d = *dst; launch(cuda, self.f.view_zero, [grid1(d.size)[0].min(65535), 1, 1], kargs!(a, d))?; },
            R::Max { out, x, xv, c, .. } => { let (mut o, mut i, mut p, mut v, mut c) = (*out, *x, prim, *xv, *c);
                launch(cuda, self.f.max_tl, grid1(o.size), kargs!(a, o, i, p, v, c))?; },
            R::Where { dst, src, mv, .. } => { let (mut d, mut s, mut p, mut m) = (*dst, *src, prim, *mv);
                launch(cuda, self.f.where_tl, grid1(d.size), kargs!(a, d, s, p, m))?; },
            R::Scale { input, output, n, c } => {
                if input != output { dev::copy(cuda, self.sh(*output), self.sh(*input), n * 4)?; }
                self.support.scale(cuda, self.sh(*output), *c, n * 4, 4)?;
            },
            R::Copy { dst, src, n } => dev::copy(cuda, self.sh(*dst), self.sh(*src), n * 4)?,
            R::PeerSend { src, n, key } => {
                let p = crate::peer::require("peer_send tangent")?;
                dev::sync(cuda)?;
                p.put_tl(key, &dev::download(cuda, self.sh(*src), n * 4)?)?;
            },
            R::PeerRecv { dst, n, key } => {
                let p = crate::peer::require("peer_recv tangent")?;
                let d = p.get_tl(key, n * 4)?;
                dev::sync(cuda)?;
                dev::upload(cuda, self.sh(*dst), &d)?;
                dev::sync(cuda)?;
            },
            R::Zero { ranges, .. } | R::Held { ranges, .. } => for (e, n) in ranges { dev::zero(cuda, self.sh(*e), n * 4)?; },
            R::FctMinmax { .. } => self.fct_minmax(cuda, r, prim, false)?,
            R::Native(f) => return Err(format!("{f}: native row reached the generic tangent")),
            R::Tempo(f) => return Err(format!("{f}: TEMPO launch reached the generic tangent")),
            R::TempoSerial(f) => return Err(format!("{f}: TEMPO serial entry reached the generic tangent")),
            R::TempoSmall(f) => return Err(format!("{f}: P5 per-thread entry reached the generic tangent")),
            R::Entry | R::Identity(_) | R::Refused(_) => {},
        }
        Ok(())
    }
    /// multidevice-runner: a reused device buffer of at least `bytes` for a received cotangent (grown, never shrunk).
    unsafe fn peer_tmp(&mut self, cuda: &Cuda, bytes: usize) -> Result<u64, String> {
        if self.peer_buf.1 < bytes {
            if self.peer_buf.0 != 0 { checked("cuMemFree peer", (cuda.free)(self.peer_buf.0))?; }
            let mut p = 0u64;
            checked("cuMemAlloc peer", (cuda.allocate)(&mut p, bytes.max(1 << 20)))?;
            self.peer_buf = (p, bytes.max(1 << 20));
        }
        Ok(self.peer_buf.0)
    }
    /// Transpose of `tl` (walking back).
    unsafe fn ad(&mut self, cuda: &Cuda, r: &R, prim: u64) -> Result<(), String> {
        let mut a = self.sh_base;
        let mut fail = self.fail;
        match r {
            R::Rows { rows, .. } => {
                let injective = self.gather_for(cuda, rows)?;
                if injective {
                    let mut rd = self.rows_dev(cuda, rows)?;
                    launch(cuda, self.f.rows_ad_direct, grid_rows(rows), kargs!(a, rd, fail))?;
                } else {
                    let h = Self::rows_hash(rows);
                    let g = self.gathers[&h].as_ref().map_err(|e| e.clone())?;
                    let [mut sw, mut st, mut it, mut dw, mut sl] = g.bufs;
                    let (mut ns, mut nd) = (g.nsrc, g.ndst);
                    launch(cuda, self.f.gather_ad, grid1(ns), kargs!(a, sw, st, it, dw, ns, fail))?;
                    launch(cuda, self.f.clear_dst, grid1(nd), kargs!(a, dw, sl, nd, fail))?;
                }
            },
            R::View { dst, src } => { let (mut d, mut s) = (*dst, *src); launch(cuda, self.f.view_ad, grid1(d.size), kargs!(a, d, s))?; },
            R::ZeroView { dst, .. } => { let mut d = *dst; launch(cuda, self.f.view_zero, [grid1(d.size)[0].min(65535), 1, 1], kargs!(a, d))?; },
            R::Max { out, x, xv, c, .. } => { let (mut o, mut i, mut p, mut v, mut c) = (*out, *x, prim, *xv, *c);
                launch(cuda, self.f.max_ad, grid1(o.size), kargs!(a, o, i, p, v, c))?; },
            R::Where { dst, src, mv, .. } => { let (mut d, mut s, mut p, mut m) = (*dst, *src, prim, *mv);
                launch(cuda, self.f.where_ad, grid1(d.size), kargs!(a, d, s, p, m))?; },
            R::Scale { input, output, n, c } => {
                if input == output { self.support.scale(cuda, self.sh(*output), *c, n * 4, 4)?; }
                else { self.support.axpy_clear(cuda, self.sh(*input), self.sh(*output), *c, n * 4, 4)?; }
            },
            R::Copy { dst, src, n } => self.support.move_add(cuda, self.sh(*src), self.sh(*dst), n * 4, 4)?,
            // multidevice-runner: the transpose of the cross-rank copy dst <- src is src += dst_bar, dst_bar = 0, split
            // across the two ranks: the receiving rank sends its cotangent and zeroes it, the sending rank adds it.
            R::PeerRecv { dst, n, key } => {
                let p = crate::peer::require("peer_recv adjoint")?;
                dev::sync(cuda)?;
                p.put_ad(key, &dev::download(cuda, self.sh(*dst), n * 4)?)?;
                dev::zero(cuda, self.sh(*dst), n * 4)?;
            },
            R::PeerSend { src, n, key } => {
                let p = crate::peer::require("peer_send adjoint")?;
                let d = p.get_ad(key, n * 4)?;
                let tmp = self.peer_tmp(cuda, n * 4)?;
                dev::sync(cuda)?;
                dev::upload(cuda, tmp, &d)?;
                dev::sync(cuda)?;
                self.support.move_add(cuda, self.sh(*src), tmp, n * 4, 4)?;
            },
            R::Zero { ranges, .. } => for (e, n) in ranges { dev::zero(cuda, self.sh(*e), n * 4)?; },
            R::Held { ranges, origin } => {
                // The dropped cotangent's size, per origin (what holding this upload cuts), then the transpose of the zero tangent.
                self.support.reset_words(cuda)?;
                for (k, (e, n)) in ranges.iter().enumerate().take(dev::SLOTS) { self.support.sqdiff_into(cuda, k, self.sh(*e), 0, n * 4, 4)?; }
                let (s, _) = self.support.collect(cuda, ranges.len().min(dev::SLOTS), 0)?;
                *self.held_dropped.entry(origin.clone()).or_default() += s.iter().sum::<f64>();
                for (e, n) in ranges { dev::zero(cuda, self.sh(*e), n * 4)?; }
            },
            R::FctMinmax { .. } => self.fct_minmax(cuda, r, prim, true)?,
            R::Native(f) => return Err(format!("{f}: native row reached the generic adjoint")),
            R::Tempo(f) => return Err(format!("{f}: TEMPO launch reached the generic adjoint")),
            R::TempoSerial(f) => return Err(format!("{f}: TEMPO serial entry reached the generic adjoint")),
            R::TempoSmall(f) => return Err(format!("{f}: P5 per-thread entry reached the generic adjoint")),
            R::Entry | R::Identity(_) | R::Refused(_) => {},
        }
        Ok(())
    }

    /// Bytes of the `post` primal copies a kernel's reverse tapes (whole allocations, as U6).
    fn post_bytes(kd: &deriv::KernelDeriv, args: &[submit::Argument], live: &HashMap<u64, submit::Allocation>) -> usize {
        let mut seen = std::collections::HashSet::new();
        let mut total = 0;
        for e in kd.ad_order.iter().chain(kd.aug.then_some(&"aug".to_owned())) {
            for bind in kd.spec["launch_abi"]["entries"][e]["args"].as_array().into_iter().flatten() {
                if bind["bind"] != "primal" || bind["post"] != true { continue; }
                let Some(name) = bind["arg"].as_str() else { continue };
                let Some(i) = kd.names.iter().position(|n| n == name) else { continue };
                if let Some(submit::Argument::Device(s)) = args.get(i) {
                    if seen.insert(s.allocation) { total += live.get(&s.allocation).map(|a| a.bytes).unwrap_or(0); }
                }
            }
        }
        total
    }

    /// TEMPO serial entry: shadow segments (allocation, lo byte, hi byte, byte offset into the float64 vector) of a
    /// snapshot's (live address, words, first float64 word) list. Refuses a span outside every live allocation.
    fn tempo_segs(ex: &Execution, list: &[(u64, usize, usize)]) -> Result<Vec<(u64, usize, usize, u64)>, String> {
        let mut v = Vec::with_capacity(list.len());
        for &(addr, words, off) in list {
            let a = ex.allocations.values().find(|a| addr >= a.address && addr < a.address + a.bytes as u64)
                .ok_or_else(|| format!("tempo_precip: pointer {addr:#x} is in no live allocation"))?;
            let lo = (addr - a.address) as usize;
            let hi = lo + 4 * words;
            if hi > a.bytes { return Err(format!("tempo_precip: {words} words at byte {lo} pass the end of allocation {}", a.id)); }
            v.push((a.id, lo, hi, 8 * off as u64));
        }
        Ok(v)
    }
    /// Outputs must not overlap each other or a read-only input (the in-place adjoint assumes it).
    fn tempo_disjoint(ins: &[(u64, usize, usize, u64)], outs: &[(u64, usize, usize, u64)], ro_bytes: u64) -> Result<(), String> {
        let mut s: Vec<(u64, usize, usize)> = outs.iter().map(|x| (x.0, x.1, x.2)).collect();
        s.extend(ins.iter().filter(|x| x.3 < ro_bytes).map(|x| (x.0, x.1, x.2)));
        s.sort_unstable();
        for w in s.windows(2) { if w[0].0 == w[1].0 && w[1].1 < w[0].2 { return Err(format!("tempo_precip: overlapping slots in allocation {}", w[0].0)); } }
        Ok(())
    }
    fn based(segs: &[(u64, usize, usize, u64)], buf: u64) -> Vec<(u64, usize, usize, u64)> { segs.iter().map(|&(a, l, h, o)| (a, l, h, buf + o)).collect() }

    /// Augmented forward of one command: forward, then its tangent. Returns the reverse tape estimate.
    unsafe fn tangent(&mut self, ex: &mut Execution, op: &Operation, step: usize, base: &Path, frame: &mut Option<ReplayFrame>) -> Result<usize, String> {
        let (r, l) = self.rule(ex, op, step, base, frame.as_ref())?;
        let light = ordered(ex) && !matches!(r, R::Native(_) | R::Tempo(_) | R::TempoSerial(_) | R::TempoSmall(_));
        if self.sync_owed && !light && !matches!((&r, &l), (R::Entry, Some(_))) { dev::sync(&ex.cuda)?; self.sync_owed = false; }
        match (&r, l) {
            (R::Entry, Some((f, args, grid, block, shared))) => {
                let t0 = Instant::now();
                let kd = &self.kernels[&f];
                let li = deriv::LaunchInfo { args: &args, grid, block, shared };
                self.tape.reset(); self.scratch.reset();
                let tp = dev::prof::start();
                let mut b = loop {
                    match self.kernels[&f].prepare_with(&ex.cuda, &mut self.tape, &li, &ex.allocations, 4, Some(&self.shadows)) {
                        Ok(b) => break b,
                        Err(e) if self.grow_tape_for(ex, &e, step, &f)? => { self.tape.reset(); self.scratch.reset(); }
                        Err(e) => return Err(format!("step {step} {f} tangent bind: {e}")),
                    }
                };
                let kd = &self.kernels[&f];
                dev::prof::add(2, tp);
                let post = if self.pack { kd.post_spans(&li, &b.env, &ex.allocations)?.values().map(|(lo, hi)| hi - lo).sum() }
                           else { Self::post_bytes(kd, &args, &ex.allocations) };
                let est = (self.tape.used - if self.pack { b.parts[3] } else { 0 }) + post;
                let parts = b.parts;
                if !ordered(ex) { dev::sync(&ex.cuda)?; self.sync_owed = false; }
                exec(ex, op, step, base, frame)?;
                let cuda = &ex.cuda;
                for c in b.tl_cells.values() { dev::zero(cuda, c.0, 8)?; }
                kd.zero_workspaces(cuda, &mut b)?;
                if kd.aug { kd.launch(cuda, &mut self.scratch, &mut b, &li, &ex.allocations, "aug")?; }
                // AD_SUBNORMAL=f,.. (diagnostic only): the argument shadows' census before and after this tangent too.
                let census = std::env::var("AD_SUBNORMAL").ok().is_some_and(|v| v.split(',').any(|x| x == f));
                let before = if census { dev::sync(cuda)?; kd.shadow_census(cuda, &b, &li) } else { Vec::new() };
                { let t = dev::prof::start(); kd.begin_tl(cuda, &self.support, &mut b, &li)?; dev::prof::add(1, t); }
                { let t = dev::prof::start(); for e in &kd.tl_order { kd.launch(cuda, &mut self.scratch, &mut b, &li, &ex.allocations, e)?; } dev::prof::add(5, t); }
                if census { dev::sync(cuda)?; eprintln!("{}", json!({"event":"window_tl_census","step":step,"function":f,
                    "fields":"[arg, words, nonzero, subnormal, min nonzero |t|, max |t|]","before":before,"after":kd.shadow_census(cuda, &b, &li)})); }
                if census { if let Ok(dir) = std::env::var("AD_CENSUS_DUMP") {
                    let tag = format!("s{step}.{f}.{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
                    let addr = |id: u64| ex.allocations.get(&id).map(|a| a.address);
                    kd.census_dump(cuda, &b, &li, &addr, &dir, &tag); } }
                // Integration (ad/phys-window): round widened float64 tangent outs/inouts back into their float32 views,
                // as dot.rs and reverse.rs do since L3 widen_io (8e51e4ba). Without it the TL output of
                // source_baseline (baseline) stayed in the temporary while its adjoint consumed the seed:
                // integ1win-e window dot rel 3.9e-4 at 1 step (4.6e-11 with it refused, L1 smoke3e).
                { let t = dev::prof::start(); kd.finish_tl(cuda, &self.support, &mut b, &li)?; dev::prof::add(1, t); }
                if std::env::var("AD_ZERO_DERIV").ok().is_some_and(|v| v.split(',').any(|x| x == f)) { dev::sync(cuda)?; kd.zero_written(cuda, &b, &li, false)?; }
                // readbacks-device: the failure slots go to the step ledger (read once at the step end, same refusal text).
                let t = dev::prof::start();
                kd.queue_failures(cuda, &mut b, self.ledger.as_mut().ok_or("step ledger missing")?, &format!("step {step} {f} tangent"))?;
                dev::prof::add(7, t);
                if entry_sync() { dev::sync(cuda)?; } else { self.sync_owed = true; }
                self.rev_scratch = self.rev_scratch.max(parts[3] + if deriv::tape_block_bytes() > 0 { parts[0] } else { 0 });
                self.row(&f, [parts[0], parts[2], parts[3], post, 0]);
                self.note("tl", &f, t0);
                Ok(est)
            },
            (R::Max { prim: (al, first, n), .. }, _) => {
                self.tape.reset();
                let p = self.tape_primal(ex, *al, *first, *n, 4)?;
                exec(ex, op, step, base, frame)?;
                self.tl(&ex.cuda, &r, p)?;
                self.row("rule:max", [0, 0, 0, 0, n * 4]);
                Ok(n * 4)
            },
            (R::Where { prim: (al, first, n), .. }, _) => {
                self.tape.reset();
                let p = self.tape_primal(ex, *al, *first, *n, 1)?;
                exec(ex, op, step, base, frame)?;
                self.tl(&ex.cuda, &r, p)?;
                self.row("rule:where", [0, 0, 0, 0, *n]);
                Ok(*n)
            },
            (R::Native(_), Some((f, args, grid, block, _))) => {
                let t0 = Instant::now();
                self.tape.reset(); self.scratch.reset();
                let tp = dev::prof::start();
                let mut p = loop {
                    match self.native.as_mut().ok_or("native rows not loaded")?.prepare(&ex.cuda, &mut self.tape, &ex.allocations, &f, &args, grid, block) {
                        Ok(p) => break p,
                        Err(e) if self.grow_tape_for(ex, &e, step, &f)? => { self.tape.reset(); self.scratch.reset(); }
                        Err(e) => return Err(format!("step {step} {f} native tangent bind: {e}")),
                    }
                };
                dev::prof::add(2, tp);
                let est = self.tape.used;
                // tape-dedupe-and-pack: the reverse binds this row's transient buffers from the reverse scratch.
                if self.pack { self.rev_scratch = self.rev_scratch.max(p.transient_bytes); }
                dev::sync(&ex.cuda)?;
                let mut ids: Vec<u64> = p.active_segments().iter().map(|s| s.0).collect();
                ids.sort_unstable(); ids.dedup();
                let before = if self.trace { Some((self.alloc_norms(&ex.cuda, &ids)?, self.state_norm(&ex.cuda)?)) } else { None };
                let seg_before = if self.trace { Some(self.seg_norms(ex, &p.active_segments())?) } else { None };
                exec(ex, op, step, base, frame)?;
                let cuda = &ex.cuda;
                let segs = p.active_segments();
                self.xfer(cuda, &segs, true)?;
                let tn = dev::prof::start();
                let nn = self.native.as_mut().unwrap();
                nn.ledger = self.ledger.take();
                let info = nn.tangent(cuda, &mut self.scratch, &mut p, false);
                self.ledger = nn.ledger.take();
                let info = info.map_err(|e| format!("step {step} {f} native tangent: {e}"))?;
                dev::prof::add(8, tn);
                self.xfer(cuda, &segs, false)?;
                dev::sync(cuda)?;
                eprintln!("{}", json!({"event":"window_native","pass":"tl","step":step,"function":f,"ms":t0.elapsed().as_secs_f64() * 1e3,"info":info}));
                self.row(&format!("native:{f}"), [0, 0, 0, 0, est]);
                if let Some((b, sb)) = before {
                    let a = self.alloc_norms(cuda, &ids)?;
                    let sa = self.state_norm(cuda)?;
                    let mut rows: Vec<(u64, f64, f64)> = ids.iter().zip(b.iter().zip(a.iter())).map(|(i, (x, y))| (*i, *x, *y)).collect();
                    rows.sort_by(|p, q| (q.2 / q.1.max(1e-300)).partial_cmp(&(p.2 / p.1.max(1e-300))).unwrap_or(std::cmp::Ordering::Equal));
                    eprintln!("{}", json!({"event":"window_trace_native","step":step,"function":f,"state_before":sb,"state_after":sa,
                        "state_ratio":sa / sb.max(1e-300),"allocs":rows.iter().take(12).collect::<Vec<_>>()}));
                    if let Some(sb) = seg_before {
                        let sa = self.seg_norms(ex, &p.active_segments())?;
                        // [alloc, lo, hi, words, tangent ss before, tangent ss after, primal ss before, primal ss after]
                        let segs: Vec<Value> = p.active_segments().iter().zip(sb.iter().zip(sa.iter()))
                            .map(|(s, (b, a))| json!([s.0, s.1, s.2, (s.2 - s.1) / 4, b.0, a.0, b.1, a.1])).collect();
                        eprintln!("{}", json!({"event":"window_trace_native_segments","step":step,"function":f,"segments":segs}));
                    }
                }
                self.note("tl", &f, t0);
                Ok(est)
            },
            (R::TempoSerial(_), Some((f, args, _, _, _))) => {
                // tempo_precip on P5's per-thread entry: snapshot its inputs, run the forward, record the tape from the
                // snapshot (identity: the record must equal the forward word for word), then the tangent sweep on the
                // widened window shadows of its inputs; the outputs are rounded back (the --native-schemes adapter rule).
                let t0 = Instant::now();
                self.tape.reset(); self.scratch.reset();
                let t = self.tempo_serial.as_ref().ok_or("TEMPO serial entry not loaded")?;
                let snap = t.snapshot(&ex.cuda, &mut self.tape, &ex.allocations, &f, &args).map_err(|e| format!("step {step} {f} tempo tangent bind: {e}"))?;
                let est = self.tape.used;
                let ins = Self::tempo_segs(ex, &snap.input_list())?;
                let outs = Self::tempo_segs(ex, &snap.outputs())?;
                Self::tempo_disjoint(&ins, &outs, 8 * snap.read_only_words() as u64)?;
                exec(ex, op, step, base, frame)?;
                let cuda = &ex.cuda;
                let (nin, nout) = snap.io_words();
                let (xd, yd) = (self.scratch.take(8 * nin)?, self.scratch.take(8 * nout)?);
                dev::zero(cuda, xd, 8 * nin)?; dev::zero(cuda, yd, 8 * nout)?;
                self.xfer(cuda, &Self::based(&ins, xd), true)?;
                let tp = t.record_tape(cuda, &mut self.scratch, &snap).map_err(|e| format!("step {step} {f} tempo record: {e}"))?;
                let (idiff, icmp) = t.identity(cuda, &snap, &tp)?;
                if idiff != 0 { return Err(format!("step {step} {f}: TEMPO serial record differs from the forward in {idiff} of {icmp} words")); }
                t.sweep_dev(cuda, &snap, &tp, false, xd, yd)?;
                self.xfer(cuda, &Self::based(&outs, yd), false)?;
                dev::sync(cuda)?;
                self.row(&format!("tempo_serial:{f}"), [0, 0, 0, 0, est]);
                self.note("tl", &f, t0);
                Ok(est)
            },
            (R::TempoSmall(_), Some((f, args, _, _, _))) => {
                let t0 = Instant::now();
                self.tape.reset(); self.scratch.reset();
                let (ins, outs) = self.small_slots(ex, &f, &args).map_err(|e| format!("step {step} {f} tempo-small slots: {e}"))?;
                let ts = self.tempo_small.as_ref().ok_or("--tempo-small entry not loaded")?;
                let snap = ts.window_snapshot(&ex.cuda, &mut self.tape, &ex.allocations, &f, &args)
                    .map_err(|e| format!("step {step} {f} tempo-small snapshot: {e}"))?;
                let est = self.tape.used;
                exec(ex, op, step, base, frame)?;
                self.small_sweep(&ex.cuda, &f, &snap, &ins, &outs, false, true).map_err(|e| format!("step {step} {f} tempo-small tangent: {e}"))?;
                self.row(&format!("tempo_small:{f}"), [0, 0, 0, 0, est]);
                self.note("tl", &f, t0);
                Ok(est)
            },
            (R::FctMinmax { prim, .. }, _) => {
                self.tape.reset();
                let p = self.tape_fct(ex, prim)?;
                exec(ex, op, step, base, frame)?;
                self.tl(&ex.cuda, &r, p)?;
                self.row("rule:fct_minmax", [0, 0, 0, 0, (prim[0].2 + prim[1].2) * 4 + 256]);
                Ok((prim[0].2 + prim[1].2) * 4 + 256)
            },
            (R::Tempo(_), Some((f, args, _, _, _))) => {
                let t0 = Instant::now();
                self.tape.reset();
                let snap = { let (t, _) = self.tempo.as_ref().ok_or("TEMPO derivative not loaded")?;
                    t.snapshot(&ex.cuda, &mut self.tape, &ex.allocations, &args).map_err(|e| format!("step {step} {f} TEMPO snapshot: {e}"))? };
                let est = self.tape.used;
                exec(ex, op, step, base, frame)?;
                self.tempo_apply(ex, &snap, step, false).map_err(|e| format!("step {step} {f} TEMPO tangent: {e}"))?;
                self.row(&format!("tempo:{f}"), [0, 0, 0, 0, est]);
                self.note("tl", &f, t0);
                Ok(est)
            },
            _ => { exec(ex, op, step, base, frame)?; self.tl(&ex.cuda, &r, 0)?; Ok(0) },
        }
    }

    /// x20 fused tape: one tangent command that also records its reverse tape entry into `rtape` (see the field).
    unsafe fn fused_tangent(&mut self, ex: &mut Execution, op: &Operation, step: usize, base: &Path, frame: &mut Option<ReplayFrame>) -> Result<usize, String> {
        let (r, l) = self.rule(ex, op, step, base, frame.as_ref())?;
        let mut pre = None;
        if self.fuse_ok {
            let mut rt = self.rtape.take().ok_or("fused tape arena missing")?;
            std::mem::swap(&mut self.tape, &mut rt);
            let p = self.rec_pre(ex, &r, &l, step);
            std::mem::swap(&mut self.tape, &mut rt);
            self.rtape = Some(rt);
            match p? { Some(p) => pre = Some(p), None => self.fz_overflow(step, "pre-launch") }
        }
        self.rule_cache = Some((r, l));
        let est = self.tangent(ex, op, step, base, frame);
        self.rule_cache = None;
        let est = est?;
        if let Some(p) = pre {
            let mut rt = self.rtape.take().ok_or("fused tape arena missing")?;
            std::mem::swap(&mut self.tape, &mut rt);
            let t = self.rec_post(ex, p);
            std::mem::swap(&mut self.tape, &mut rt);
            self.rtape = Some(rt);
            match t? { Some(tp) => { if let Some((_, v)) = self.fz_cur.as_mut() { v.push(tp); } }, None => self.fz_overflow(step, "post-launch") }
        }
        Ok(est)
    }
    fn fz_overflow(&mut self, step: usize, at: &str) {
        self.fuse_ok = false;
        let (used, bytes) = self.rtape.as_ref().map(|t| (t.used, t.bytes)).unwrap_or((0, 0));
        eprintln!("{}", json!({"event":"window_fused_tape_overflow","step":step,"at":at,"arena_used":used,"arena_bytes":bytes,"step_start":self.fz_step_start}));
    }
    /// x20 fused tape: tape_range's part before the forward command runs (self.tape is the fused arena here).
    unsafe fn rec_pre(&mut self, ex: &Execution, r: &R, l: &Option<(String, Vec<submit::Argument>, [u32; 3], [u32; 3], u32)>, step: usize) -> Result<Option<FzPre>, String> {
        Ok(Some(match (r, l) {
            (R::Entry, Some((f, args, grid, block, shared))) => {
                let kd = &self.kernels[f];
                let li = deriv::LaunchInfo { args, grid: *grid, block: *block, shared: *shared };
                let tp = dev::prof::start();
                let cache = if self.pack { Some((&mut self.precache, &self.support)) } else { None };
                let b = match kd.prepare_tape(&ex.cuda, &mut self.tape, &li, &ex.allocations, 4, Some(&self.shadows), self.pack, cache) {
                    Ok(b) => b, Err(e) if exhausted(&e) => return Ok(None),
                    Err(e) => return Err(format!("step {step} {f} fused reverse bind: {e}")),
                };
                dev::prof::add(2, tp);
                dev::sync(&ex.cuda)?;
                FzPre::Entry { f: f.clone(), args: args.clone(), grid: *grid, block: *block, shared: *shared, b }
            },
            (R::Native(_), Some((f, args, grid, block, _))) => {
                let n = self.native.as_mut().ok_or("native rows not loaded")?;
                let p = match if self.pack { n.prepare_deferred(&ex.cuda, &mut self.tape, &ex.allocations, f, args, *grid, *block) } else { n.prepare(&ex.cuda, &mut self.tape, &ex.allocations, f, args, *grid, *block) } {
                    Ok(p) => p, Err(e) if exhausted(&e) => return Ok(None),
                    Err(e) => return Err(format!("step {step} {f} fused native reverse bind: {e}")),
                };
                dev::sync(&ex.cuda)?;
                FzPre::Done(Tp::Native { f: f.clone(), p })
            },
            (R::Tempo(_), Some((f, args, _, _, _))) => {
                let (t, _) = self.tempo.as_ref().ok_or("TEMPO derivative not loaded")?;
                let snap = match t.snapshot(&ex.cuda, &mut self.tape, &ex.allocations, args) {
                    Ok(s) => s, Err(e) if exhausted(&e) => return Ok(None),
                    Err(e) => return Err(format!("step {step} {f} fused TEMPO reverse snapshot: {e}")),
                };
                FzPre::Done(Tp::Tempo { f: f.clone(), snap })
            },
            (R::TempoSerial(_), Some((f, args, _, _, _))) => {
                let t = self.tempo_serial.as_ref().ok_or("TEMPO serial entry not loaded")?;
                let snap = match t.snapshot(&ex.cuda, &mut self.tape, &ex.allocations, f, args) {
                    Ok(s) => s, Err(e) if exhausted(&e) => return Ok(None),
                    Err(e) => return Err(format!("step {step} {f} fused tempo reverse bind: {e}")),
                };
                let ins = Self::tempo_segs(ex, &snap.input_list())?;
                let outs = Self::tempo_segs(ex, &snap.outputs())?;
                FzPre::Done(Tp::TempoSerial { f: f.clone(), snap, ins, outs })
            },
            (R::TempoSmall(_), Some((f, args, _, _, _))) => {
                let (ins, outs) = self.small_slots(ex, f, args).map_err(|e| format!("step {step} {f} tempo-small slots: {e}"))?;
                let ts = self.tempo_small.as_ref().ok_or("--tempo-small entry not loaded")?;
                let snap = match ts.window_snapshot(&ex.cuda, &mut self.tape, &ex.allocations, f, args) {
                    Ok(s) => s, Err(e) if exhausted(&e) => return Ok(None),
                    Err(e) => return Err(format!("step {step} {f} fused tempo-small reverse snapshot: {e}")),
                };
                FzPre::Done(Tp::TempoSmall { f: f.clone(), snap, ins, outs })
            },
            (R::FctMinmax { prim, .. }, _) => {
                let p = match self.tape_fct(ex, prim) { Ok(p) => p, Err(e) if exhausted(&e) => return Ok(None), Err(e) => return Err(e) };
                FzPre::Done(Tp::Rule { r: r.clone(), prim: p })
            },
            (R::Max { prim: (al, first, n), .. }, _) => {
                let p = match self.tape_primal(ex, *al, *first, *n, 4) { Ok(p) => p, Err(e) if exhausted(&e) => return Ok(None), Err(e) => return Err(e) };
                FzPre::Done(Tp::Rule { r: r.clone(), prim: p })
            },
            (R::Where { prim: (al, first, n), .. }, _) => {
                let p = match self.tape_primal(ex, *al, *first, *n, 1) { Ok(p) => p, Err(e) if exhausted(&e) => return Ok(None), Err(e) => return Err(e) };
                FzPre::Done(Tp::Rule { r: r.clone(), prim: p })
            },
            (r, _) => FzPre::Done(Tp::Rule { r: r.clone(), prim: 0 }),
        }))
    }
    /// x20 fused tape: tape_range's part after the forward command ran (shared-span check, post-launch copies).
    unsafe fn rec_post(&mut self, ex: &Execution, pre: FzPre) -> Result<Option<Tp>, String> {
        let (f, mut b, args, grid, block, shared) = match pre { FzPre::Done(t) => return Ok(Some(t)), FzPre::Entry { f, b, args, grid, block, shared } => (f, b, args, grid, block, shared) };
        let li = deriv::LaunchInfo { args: &args, grid, block, shared };
        if !b.shared.is_empty() {
            dev::sync(&ex.cuda)?;
            for (al, lo, hi, at) in std::mem::take(&mut b.shared) {
                let live = ex.allocations.get(&al).ok_or("shared primal allocation not live")?.address + lo as u64;
                if deriv::same_words(&ex.cuda, &self.support, live, at, hi - lo)? { b.shared.push((al, lo, hi, at)); continue; }
                if deriv::tape_block_bytes() > 0 {
                    let names = b.cluster_args.get(&(al, lo)).cloned().unwrap_or_default();
                    let rt = dev::run_rows(&[(0, at, hi - lo)]);
                    let tab = match self.tape.take(rt.len() * 8) { Ok(c) => c, Err(e) if exhausted(&e) => return Ok(None), Err(e) => return Err(e) };
                    dev::upload(&ex.cuda, tab, std::slice::from_raw_parts(rt.as_ptr().cast::<u8>(), rt.len() * 8))?;
                    b.recon.push((al, lo, hi, (tab, rt.len() / 3), names, true));
                    self.precache.blk_spans += 1; self.precache.blk_hit_bytes += hi - lo;
                    continue;
                }
                let c = match self.tape.take(hi - lo) { Ok(c) => c, Err(e) if exhausted(&e) => return Ok(None), Err(e) => return Err(e) };
                dev::copy(&ex.cuda, c, at, hi - lo)?;
                b.pre.insert(al, c.wrapping_sub(lo as u64));
                if let Some(ns) = b.cluster_args.get(&(al, lo)).cloned() { for n in ns { b.pre_arg.insert(n, c.wrapping_sub(lo as u64)); } }
                self.precache.unshared += 1; self.precache.unshared_bytes += hi - lo;
            }
        }
        let kd = &self.kernels[&f];
        let mut post = HashMap::new();
        if self.pack {
            for (al, (lo, hi)) in kd.post_spans(&li, &b.env, &ex.allocations)? {
                let a = ex.allocations[&al];
                if let Some(base) = self.precache.lookup(&ex.cuda, &self.support, al, a.address, lo, hi)? {
                    post.insert(al, submit::Allocation { id: a.id, address: base, bytes: a.bytes });
                    continue;
                }
                let copy = match self.tape.take(hi - lo) { Ok(c) => c, Err(e) if exhausted(&e) => return Ok(None), Err(e) => return Err(e) };
                dev::copy(&ex.cuda, copy, a.address + lo as u64, hi - lo)?;
                self.precache.insert(al, lo, hi, copy);
                post.insert(al, submit::Allocation { id: a.id, address: copy.wrapping_sub(lo as u64), bytes: a.bytes });
            }
        }
        for e in kd.ad_order.iter().chain(kd.aug.then_some(&"aug".to_owned())).filter(|_| !self.pack) {
            for bind in kd.spec["launch_abi"]["entries"][e]["args"].as_array().into_iter().flatten() {
                if bind["bind"] != "primal" || bind["post"] != true { continue; }
                let Some(name) = bind["arg"].as_str() else { continue };
                let Some(k) = kd.names.iter().position(|n| n == name) else { continue };
                if let submit::Argument::Device(s) = &args[k] {
                    if post.contains_key(&s.allocation) { continue; }
                    let a = ex.allocations[&s.allocation];
                    let copy = match self.tape.take(a.bytes) { Ok(c) => c, Err(e) if exhausted(&e) => return Ok(None), Err(e) => return Err(e) };
                    dev::copy(&ex.cuda, copy, a.address, a.bytes)?;
                    post.insert(s.allocation, submit::Allocation { id: a.id, address: copy, bytes: a.bytes });
                }
            }
        }
        Ok(Some(Tp::Entry { f, b, args, grid, block, shared, post }))
    }
    /// x20 fused tape: a tangent step starts recording (the arena is made at the first step; a chunk that cannot hold
    /// another step of the largest size seen so far goes to page-locked RAM first).
    unsafe fn fz_begin(&mut self, ex: &Execution, step: usize) -> Result<(), String> {
        let cuda = &ex.cuda;
        if self.census.is_some() || self.fz_failed { return Ok(()); }
        if self.rtape.is_none() {
            let (mut free, mut total) = (0usize, 0usize);
            checked("cuMemGetInfo", (cuda.mem_info)(&mut free, &mut total))?;
            let want = match std::env::var("AD_FUSE_TAPE_GIB").ok().and_then(|v| v.parse::<f64>().ok()) {
                Some(g) => (g * (1u64 << 30) as f64) as usize,
                None => free.saturating_sub(28usize << 30).min(free / 2),
            } & !((1usize << 20) - 1);
            if want < (64 << 20) { self.fz_failed = true; eprintln!("{}", json!({"event":"window_fused_tape_off","reason":"no room","card_free_bytes":free})); return Ok(()); }
            match dev::Arena::new(cuda, want) {
                Ok(a) => {
                    eprintln!("{}", json!({"event":"window_fused_tape_arena","step":step,"bytes":want,"card_free_bytes":free}));
                    self.rtape = Some(a);
                    // The page-locked buffers that full chunks go to are made on a helper thread during the tangent pass (at
                    // most one per later step, each the size of the arena, at most half the available RAM), not on the main
                    // thread at the step boundary: v1 paid 21.8 s of page-locking per pk120 window there (x20-C-c3revstruct).
                    let room = mem_available() / 2;
                    let count = self.fz_steps.saturating_sub(1).min(room / want.max(1));
                    self.ahead_start(ex, count, want)?;
                },
                Err(e) => { self.fz_failed = true; eprintln!("{}", json!({"event":"window_fused_tape_off","reason":e,"card_free_bytes":free})); return Ok(()); },
            }
        }
        let (used, bytes) = { let t = self.rtape.as_ref().unwrap(); (t.used, t.bytes) };
        let need = self.fz_last_bytes + self.fz_last_bytes / 8 + (64 << 20);
        if self.fz_chunks.is_empty() { self.fz_chunks.push((Vec::new(), None, 0)); }
        else if self.fz_force_flush || (used > 0 && bytes - used < need) { self.fz_flush(cuda)?; self.fz_chunks.push((Vec::new(), None, 0)); }
        self.fz_force_flush = false;
        self.fz_step_start = self.rtape.as_ref().unwrap().used;
        self.precache.clear();
        self.fuse_ok = true;
        self.fz_cur = Some((self.fz_chunks.len() - 1, Vec::new()));
        Ok(())
    }
    /// x20 fused tape: the current chunk goes to page-locked RAM (once) and the arena starts empty.
    unsafe fn fz_flush(&mut self, cuda: &Cuda) -> Result<(), String> {
        let ci = self.fz_chunks.len() - 1;
        let used = self.fz_chunks[ci].2;
        if used > 0 && self.fz_chunks[ci].1.is_none() {
            let t = Instant::now();
            dev::sync(cuda)?;
            let mut buf = match self.take_ahead(used) { Some(b) => b, None => crate::pinned::PinnedBuf::new(cuda, used)? };
            checked("cuMemcpyDtoH fused tape", (cuda.download)(buf.as_mut_ptr().cast(), self.rtape.as_ref().unwrap().base, used))?;
            self.fz_chunks[ci].1 = Some(buf);
            self.fz_d2h_s += t.elapsed().as_secs_f64(); self.fz_d2h_bytes += used;
        }
        self.rtape.as_mut().unwrap().used = 0;
        self.fz_resident = None;
        Ok(())
    }
    /// x20 fused tape: a tangent step ends; its tape is kept for the reverse, or (overflow) dropped and the arena rolled back.
    unsafe fn fz_end(&mut self, step: usize) -> Result<(), String> {
        let Some((ci, tape)) = self.fz_cur.take() else { return Ok(()) };
        let used = self.rtape.as_ref().map(|t| t.used).unwrap_or(0);
        if self.fuse_ok {
            let bytes = used - self.fz_step_start;
            self.fz_last_bytes = self.fz_last_bytes.max(bytes);
            self.fz_chunks[ci].0.push(step); self.fz_chunks[ci].2 = used;
            self.fz_info.push(json!({"step":step,"chunk":ci,"bytes":bytes,"commands":tape.len()}));
            self.fz_tapes.insert(step, (ci, tape));
        } else {
            if let Some(t) = self.rtape.as_mut() { t.used = self.fz_step_start; }
            // a step that does not fit an empty arena will not fit later either: stop recording.
            if self.fz_step_start == 0 { self.fz_failed = true; }
            self.fz_force_flush = true;
            self.fz_info.push(json!({"step":step,"chunk":ci,"fused":false}));
            drop(tape);
        }
        self.fz_resident = Some(ci);
        self.fuse_ok = true;
        Ok(())
    }
    /// x20 fused tape: the reverse of a fused step: the state of the step's end boundary (what the one-pass replay
    /// reaches), the step's tape in the arena (uploaded to the addresses it was recorded at when its chunk is not the
    /// one resident), and the walk-back. No replay, no anchors, no segments.
    unsafe fn fused_reverse_step(&mut self, ex: &mut Execution, frame: &mut Option<ReplayFrame>, step: usize, last: bool, cur: &Boundary, cur_data: &[u8]) -> Result<Value, String> {
        let t0 = Instant::now();
        let ms0 = self.entry_ms;
        let (ci, tape) = self.fz_tapes.remove(&step).ok_or("no fused tape for this step")?;
        let n = tape.len();
        self.fit_scratch(&ex.cuda)?;
        let _ = self.pack_json();
        self.rev_rows.clear();
        let t_r = Instant::now();
        // the tangent pass left the live state at the last step's end boundary; earlier steps restore theirs.
        let restore_mode = std::env::var("AD_FUSE_RESTORE").unwrap_or_else(|_| "notlast".into());
        let restored = match restore_mode.as_str() { "all" => true, "none" => false, _ => !last };
        if restored { self.restore(ex, frame, cur, cur_data)?; }
        let restore_s = t_r.elapsed().as_secs_f64();
        let t_h = Instant::now();
        let mut h2d = 0usize;
        if self.fz_resident != Some(ci) {
            let used = self.fz_chunks[ci].2;
            let base = self.rtape.as_ref().ok_or("fused tape arena missing")?.base;
            let host = self.fz_chunks[ci].1.as_ref().ok_or("fused tape chunk is neither resident nor in RAM")?;
            dev::upload(&ex.cuda, base, &host[..used])?;
            h2d = used;
            self.fz_resident = Some(ci);
        }
        let h2d_s = t_h.elapsed().as_secs_f64();
        let cuda = &ex.cuda as *const Cuda;
        let t_w = Instant::now();
        self.walk_back(&*cuda, ex, step, tape)?;
        let walk_s = t_w.elapsed().as_secs_f64();
        self.step_flush(&ex.cuda, step, "ad")?;
        drain_pinned(ex)?;
        let mut fl = [0i32];
        checked("cuMemcpyDtoH failure", (ex.cuda.download)(fl.as_mut_ptr().cast(), self.fail, 4))?;
        if fl[0] != 0 { return Err(format!("step {step}: P8 row transpose saw a non-finite cotangent")); }
        if !self.fz_tapes.values().any(|(c, _)| *c == ci) { self.fz_chunks[ci].1 = None; }
        let host_prof: Vec<Value> = dev::prof::take().into_iter().filter(|x| x.2 > 0).map(|(k, s, c)| json!([k, s, c])).collect();
        let pack = self.pack_json();
        Ok(json!({"step":step,"commands":n,"segments":0,"fused":true,"overflow_splits":0,"commands_replayed":0,"host_prof":host_prof,"pack":pack,
                  "stages_s":{"restore":restore_s,"tape_upload":h2d_s,"walk_back":walk_s},"restored":restored,"tape_upload_bytes":h2d,"chunk":ci,
                  "adjoint_entry_s":(self.entry_ms - ms0) / 1e3,"pass1_s":0.0,"wall_s":t0.elapsed().as_secs_f64()}))
    }

    /// Execute a block like the runner's execute_block, recording each executed leaf (host branches resolved live).
    #[allow(clippy::too_many_arguments)]
    unsafe fn walk(&mut self, ex: &mut Execution, plan: &Plan, name: &str, step: usize, base: &Path,
                   frame: &mut Option<ReplayFrame>, depth: usize, flat: &mut Vec<Flat>, tangent: bool) -> Result<(), String> {
        if depth > plan.blocks.len() { return Err("host branch recursion exceeds block count".into()); }
        let commands = plan.blocks.get(name).ok_or_else(|| format!("unknown live block {name}"))?;
        // S13: the LW group section being replayed per pass (perturbed forwards only; see lw_regroup_begin).
        let mut regroup: Option<LwRegroup> = None;
        let mut index = 0usize;
        while index < commands.len() {
            let command = &commands[index];
            match command {
                Operation::HostBranch { condition, if_zero, if_nonzero } => {
                    let value = *ex.host_state.get(condition.allocation)?.get(condition.offset).ok_or("host branch condition outside allocation")?;
                    let next = if value == 0 { if_zero } else { if_nonzero };
                    self.walk(ex, plan, next, step, base, frame, depth + 1, flat, tangent)?;
                },
                Operation::RkSchedule { dt_f64, order_i64, acoustic_substeps_i64, dynamics_splits_i64, body } => {
                    if ex.rk.is_some() { return Err("nested RK schedule".into()); }
                    let v = ex.host_i64_values(&[*dt_f64, *order_i64, *acoustic_substeps_i64, *dynamics_splits_i64])?;
                    let schedule = rk_schedule::Schedule::new(f64::from_ne_bytes(v[0].to_ne_bytes()), v[1], v[2], v[3])?;
                    ex.rk = Some(rk_schedule::Scope { schedule, split: None, stage: None, small_step: None });
                    let r = self.walk(ex, plan, body, step, base, frame, depth + 1, flat, tangent);
                    ex.rk = None; r?;
                },
                Operation::RkSplits { body } => {
                    let count = ex.rk.as_ref().ok_or("split loop without RK schedule")?.schedule.dynamics_splits;
                    for index in 0..count {
                        ex.rk.as_mut().unwrap().split = Some(index);
                        let r = self.walk(ex, plan, body, step, base, frame, depth + 1, flat, tangent);
                        ex.rk.as_mut().unwrap().split = None; r?;
                    }
                },
                Operation::RkStages { body } => {
                    let stages = ex.rk.as_ref().ok_or("stage loop without RK schedule")?.schedule.stages;
                    for stage in stages {
                        ex.rk.as_mut().unwrap().stage = Some(stage);
                        let r = self.walk(ex, plan, body, step, base, frame, depth + 1, flat, tangent);
                        ex.rk.as_mut().unwrap().stage = None; r?;
                    }
                },
                Operation::RkSubsteps { body } => {
                    let count = ex.rk.as_ref().and_then(|c| c.stage).ok_or("acoustic loop without RK stage")?.acoustic_steps;
                    for index in 1..=count {
                        ex.rk.as_mut().unwrap().small_step = Some(index);
                        let r = self.walk(ex, plan, body, step, base, frame, depth + 1, flat, tangent);
                        ex.rk.as_mut().unwrap().small_step = None; r?;
                    }
                },
                Operation::Allocate { .. } if regroup.as_ref().is_some_and(|r| r.pass > 0 && index > r.start && index <= r.end) => {},
                _ => {
                    let rk = ex.rk.clone();
                    let watch = if tangent { self.g6_watch_ids(ex, command, step, frame)? } else { None };
                    let before = match &watch { Some((_, ids)) => Some(self.alloc_norms(&ex.cuda, ids)?), None => None };
                    let est = if tangent { if self.fz_cur.is_some() { self.fused_tangent(ex, command, step, base, frame)? } else { self.tangent(ex, command, step, base, frame)? } } else { exec(ex, command, step, base, frame)?; 0 };
                    if let (Some((f, ids)), Some(b)) = (&watch, &before) {
                        let a = self.alloc_norms(&ex.cuda, ids)?;
                        let rows: Vec<Value> = ids.iter().zip(b.iter().zip(a.iter())).map(|(id, (x, y))| json!([id, x.sqrt(), y.sqrt()])).collect();
                        eprintln!("{}", json!({"event":"g6_watch","step":step,"index":flat.len(),"function":f,"alloc_before_after":rows}));
                    }
                    if tangent && self.trace { self.trace_norm(&ex.cuda, step, flat.len(), command)?; }
                    flat.push(Flat { op: command as *const Operation, rk, est });
                    if let Some(hist) = ex.lw_live_hist.take() {
                        if tangent || regroup.is_some() { return Err(format!("step {step}: live LW histogram outside a perturbed forward")); }
                        regroup = Some(lw_regroup_begin(ex, commands, index, step, &hist)?);
                    }
                },
            }
            if let Some(r) = regroup.as_mut() {
                if index == r.end {
                    if lw_regroup_next(ex, r, step)? { index = r.start + 1; continue; }
                    regroup = None;
                }
            }
            index += 1;
        }
        if regroup.is_some() { return Err(format!("step {step}: LW regroup section did not close")); }
        Ok(())
    }

    /// One whole step of the forward (augmented with the tangent when `tangent`); boundary hashes are checked
    /// against the ordinary-forward reference exactly as the strict replay does.
    #[allow(clippy::too_many_arguments)]
    unsafe fn forward_step(&mut self, ex: &mut Execution, plan: &Plan, base: &Path, frame: &mut Option<ReplayFrame>,
                           refs: &References, step: usize, tangent: bool) -> Result<Vec<Flat>, String> {
        self.forward_step_c(ex, plan, base, frame, Some(refs), step, tangent)
    }
    /// forward_step with the reference check optional (None: the Taylor check's perturbed forward, which by design
    /// differs from the reference; nothing else calls it without references).
    #[allow(clippy::too_many_arguments)]
    unsafe fn forward_step_c(&mut self, ex: &mut Execution, plan: &Plan, base: &Path, frame: &mut Option<ReplayFrame>,
                             refs: Option<&References>, step: usize, tangent: bool) -> Result<Vec<Flat>, String> {
        // S13: a perturbed forward (no references, no tangent) follows a live LW histogram that differs from the
        // recorded one; AD_LW_REGROUP_TEST=1/2/3 (diagnostic, default off) also forces a multi-pass order on
        // reference-checked forwards without the tangent (see layer_groups::regroup_passes).
        let test: u8 = match std::env::var("AD_LW_REGROUP_TEST").as_deref() { Ok("1") => 1, Ok("2") => 2, Ok("3") => 3, _ => 0 };
        ex.lw_regroup = !tangent && (refs.is_none() || test != 0);
        ex.lw_regroup_test = if ex.lw_regroup { test } else { 0 };
        ex.lw_live_hist = None;
        let r = self.forward_step_body(ex, plan, base, frame, refs, step, tangent);
        ex.lw_regroup = false; ex.lw_regroup_test = 0; ex.lw_live_hist = None;
        r
    }
    #[allow(clippy::too_many_arguments)]
    unsafe fn forward_step_body(&mut self, ex: &mut Execution, plan: &Plan, base: &Path, frame: &mut Option<ReplayFrame>,
                                refs: Option<&References>, step: usize, tangent: bool) -> Result<Vec<Flat>, String> {
        // integ8 (AD_DIGEST_PROF=1, diagnostic, default off): wall of each stage of the step, no syncs added.
        let sprof = std::env::var("AD_DIGEST_PROF").as_deref() == Ok("1");
        let t_s = Instant::now();
        let blocks = match frame.as_mut() {
            Some(f) => { f.begin(step)?; f.blocks(plan.physics.as_ref().ok_or("missing physics plan")?)? },
            None => plan.cadence[step - 1].clone(),
        };
        {
            let (mut seen, mut want) = (std::collections::HashSet::new(), Vec::new());
            for name in &blocks { block_blobs(plan, name, &mut seen, &mut want); }
            let want: Vec<crate::plan::Blob> = want.into_iter().cloned().collect();
            self.stage_blobs(ex, base, want, None)?;
        }
        let t_stage = t_s.elapsed().as_secs_f64();
        let mut flat = Vec::new();
        if tangent && self.fuse_rec { self.fz_begin(ex, step)?; }
        self.defer = tangent; self.defer_step = step;
        let walked = (|| -> Result<(), String> { for name in &blocks { self.walk(ex, plan, name, step, base, frame, 0, &mut flat, tangent)?; } Ok(()) })();
        if self.sync_owed { dev::sync(&ex.cuda)?; self.sync_owed = false; }
        self.defer = false;
        // tempo-once: a tangent step installs the gathers of the steps before it (built during this step); its own
        // keep building through the next step. Any other forward installs them all.
        // x20 whole-step owner: the copy-row gathers are read only by the reverse (ad(), gather_for joins a pending one on
        // demand) and the CLI tangent pass joins them all at its end (before the tape is sized, the same memory picture), so a
        // tangent step never waits for them. Measured pk120 fold: TL step 2 waited 16.2 s here (gather_build, 17 joins) for
        // gathers begun in step 1. AD_GATHER_JOIN_STEP=1 restores the per-step join. Same tables, same refusals.
        let per_step_join = std::env::var("AD_GATHER_JOIN_STEP").as_deref() == Ok("1") || !self.gather_pass_join;
        let joined = if tangent { if per_step_join { self.join_pending_if(&ex.cuda, step, |p| p.3 != step) } else { Ok(()) } } else { self.join_pending(&ex.cuda, step) };
        walked?; joined?;
        if tangent { self.fz_end(step)?; }
        if tangent { self.step_flush(&ex.cuda, step, "tl")?; }
        let t_walk = t_s.elapsed().as_secs_f64();
        if let Some(f) = frame.as_mut() {
            f.finish()?;
            let id = plan.physics.as_ref().ok_or("missing physics plan")?.frame_state_id;
            ex.host_state.replace(id, &f.snapshot()?)?;
        }
        // integ8 (U8/L1, measured on the pkg18 window forward: three 1.1-1.25 GB held split blobs per physics step took
        // 3.6-4.8 s to inflate and the step waited for them, because their decode started only one step ahead): the
        // next two steps' blobs are decoded ahead, one batch per step, and kept in the cache until used.
        let ahead_blocks: Vec<Vec<String>> = match (frame.as_ref(), plan.physics.as_ref()) {
            (Some(f), Some(ph)) => f.blocks_ahead(ph, step + 1, 2),
            _ => (0..2).map_while(|i| plan.cadence.get(step + i).cloned()).collect(),
        };
        self.blob_ahead.clear();
        for nb in &ahead_blocks {
            let (mut seen, mut want) = (std::collections::HashSet::new(), Vec::new());
            for name in nb { block_blobs(plan, name, &mut seen, &mut want); }
            for b in &want { self.blob_ahead.insert(crate::blob_sha(b).to_owned()); }
            let cached = ex.blob_cache.as_ref();
            let want: Vec<crate::plan::Blob> = want.into_iter()
                .filter(|b| !cached.is_some_and(|c| c.contains_key(crate::blob_sha(b)))).cloned().collect();
            self.prefetch_blobs(base, want)?;
        }
        let t_pref = t_s.elapsed().as_secs_f64();
        let views = crate::carried_views(plan, step, &ex.allocations)?;
        let actual = crate::resolve_views(&views, &ex.allocations)?;
        if self.capture && std::env::var("AD_BINDER_HASH").as_deref() != Ok("device") {
            let (d, ix) = match self.capture_host.take() {
                Some((ctx, room)) => self.capture_state_host(ex, ctx, room)?,
                None => { let (d, ix) = self.capture_state(ex)?; (crate::CapData::Vec(d), ix) }
            };
            ex.captured = Some((std::sync::Arc::new(d), ix));
        }
        let t_cap = t_s.elapsed().as_secs_f64();
        ex.finish_step(step, &plan.carried, &actual, refs)?;
        let t_fin = t_s.elapsed().as_secs_f64();
        ex.finish_host_step(step, &plan.host_carried, refs)?;
        if sprof {
            eprintln!("{}", json!({"event":"forward_step_prof","step":step,"tangent":tangent,"checked":refs.is_some(),
                "stage_blobs_s":t_stage,"walk_s":t_walk - t_stage,"frame_prefetch_s":t_pref - t_walk,"views_capture_s":t_cap - t_pref,
                "finish_step_s":t_fin - t_cap,"host_finish_s":t_s.elapsed().as_secs_f64() - t_fin,"total_s":t_s.elapsed().as_secs_f64()}));
        }
        Ok(flat)
    }

    unsafe fn join_writer(&mut self) -> Result<(), String> {
        if let Some(h) = self.writer.take() { h.join().map_err(|_| "boundary writer panicked")??; }
        Ok(())
    }
    /// Store the whole device state (every live allocation) to dir/b{s}.bin (written by a background thread)
    /// and the host side in RAM.
    unsafe fn store(&mut self, ex: &mut Execution, frame: &Option<ReplayFrame>, dir: &Path, s: usize) -> Result<Boundary, String> {
        let t_store = Instant::now();
        let r = self.store_inner(ex, frame, dir, s);
        self.store_s += t_store.elapsed().as_secs_f64();
        r
    }
    /// adjoint-speed: the whole device state (every live allocation, ascending id), downloaded once.
    /// integ8 (U8/L1, measured: 2.1 s per stored step on one core, the pageable download of 3.4 GB): the state is
    /// DMA'd into one reused page-locked buffer, then copied into the owned bytes on all cores. The same bytes.
    unsafe fn capture_state(&mut self, ex: &Execution) -> Result<(Vec<u8>, Vec<(u64, usize, usize, u64)>), String> {
        let _c = capi::xfer::Cat::checkpoint();
        dev::sync(&ex.cuda)?;
        let mut ids: Vec<u64> = ex.allocations.keys().copied().collect();
        ids.sort_unstable();
        let mut index = Vec::with_capacity(ids.len());
        let mut total = 0usize;
        for id in &ids { let a = ex.allocations[id]; index.push((*id, total, a.bytes, a.address)); total += a.bytes; }
        let mut pin = match self.capture_pin.take() { Some(b) if b.len() >= total => b, _ => crate::pinned::PinnedBuf::new(&ex.cuda, total)? };
        for (_, off, b, addr) in &index {
            if *b != 0 { checked("cuMemcpyDtoH boundary", (ex.cuda.download)(pin[*off..*off + *b].as_mut_ptr().cast(), *addr, *b))?; }
        }
        let mut data: Vec<u8> = Vec::with_capacity(total);
        {
            let dst = data.spare_capacity_mut();
            let src = &pin[..total];
            let piece = (64usize << 20).max(1);
            std::thread::scope(|sc| {
                let threads = s02_threads();
                let parts: Vec<(&mut [std::mem::MaybeUninit<u8>], &[u8])> = dst.chunks_mut(piece).zip(src.chunks(piece)).collect();
                let per = parts.len().div_ceil(threads.max(1)).max(1);
                let mut parts = parts;
                while !parts.is_empty() {
                    let rest = parts.split_off(parts.len().min(per));
                    let mine = std::mem::replace(&mut parts, rest);
                    sc.spawn(move || for (d, s) in mine {
                        // SAFETY: MaybeUninit<u8> and u8 share layout; every byte of d is written here.
                        unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), d.as_mut_ptr().cast::<u8>(), s.len()); }
                    });
                }
            });
        }
        // SAFETY: the scoped copies above wrote all `total` bytes.
        unsafe { data.set_len(total); }
        self.capture_pin = Some(pin);
        Ok((data, index))
    }
    /// fwdcheck-digest (host-audit row 13): the C-ABI first tangent sweep's capture, DMA'd straight into page-locked bytes
    /// (from the process pin pool of `ctx` when there is one) that become the boundary's RAM copy: one download, read by
    /// the boundary digest and kept by Session::store, instead of a digest download plus a fresh 3.4 GB cuMemHostAlloc
    /// and a second download per boundary. The same ids, offsets and bytes as capture_state. A boundary that would not
    /// fit the `room` left in the RAM budget goes to the disk tier, so it takes the pageable capture.
    unsafe fn capture_state_host(&mut self, ex: &Execution, ctx: usize, room: usize) -> Result<(crate::CapData, Vec<(u64, usize, usize, u64)>), String> {
        let total: usize = ex.allocations.values().map(|a| a.bytes).sum();
        if total > room { let (d, ix) = self.capture_state(ex)?; return Ok((crate::CapData::Vec(d), ix)); }
        let _c = capi::xfer::Cat::checkpoint();
        // host-leftovers-2 (CLI, no pool): take the buffer the helper made during this step, then start the next one,
        // so making a 3.4 GB page-locked buffer (~0.15-0.4 s on a test box) is off the step path. Same bytes.
        let queued = if ctx == 0 { self.pin_q.pop_front() } else { None };
        let from_q = queued.is_some();
        let ahead = match (ctx, queued.or_else(|| self.pin_next.take())) {
            (0, Some((cap, h))) => match h.join() { Ok(Ok(m)) if cap >= total => Some(m), _ => None },
            _ => None,
        };
        if from_q && ahead.is_some() { self.pin_q_made += 1; }
        let mut hb = match ahead { Some(m) => { self.pin_ahead_hits += 1; capi::HostBytes::from_mem(m, total) }, None => capi::HostBytes::pooled(&ex.cuda, ctx, total)? };
        if ctx == 0 && self.pin_ahead && self.pin_q.is_empty() && !from_q && room >= 2 * total {
            let (fns, set, c) = (crate::pinned::PinFns::of(&ex.cuda), ex.cuda.ctx_set_current, ex.context as usize);
            self.pin_next = Some((total, std::thread::Builder::new().name("hl2-pin-ahead".into()).spawn(move || {
                let r = set(c as *mut std::ffi::c_void); if r != 0 { return Err(format!("cuCtxSetCurrent pin-ahead: {r}")); }
                crate::pinned::PinMem::new(fns, total)
            }).map_err(|e| e.to_string())?));
        }
        dev::sync(&ex.cuda)?;
        let mut ids: Vec<u64> = ex.allocations.keys().copied().collect();
        ids.sort_unstable();
        let mut index = Vec::with_capacity(ids.len());
        let mut at = 0usize;
        for id in &ids { let a = ex.allocations[id]; index.push((*id, at, a.bytes, a.address)); at += a.bytes; }
        { let buf = hb.as_mut_slice();
          for (_, off, b, addr) in &index {
              if *b != 0 { checked("cuMemcpyDtoH boundary", (ex.cuda.download)(buf[*off..*off + *b].as_mut_ptr().cast(), *addr, *b))?; }
          } }
        Ok((crate::CapData::Host(std::sync::Arc::new(hb)), index))
    }
    /// x20: start `n` boundary buffers of the current whole-state size together (one helper thread each, the same PinMem
    /// as pin-ahead), bounded by the RAM budget left. Same bytes: only when each buffer is made changes.
    unsafe fn pin_prefill(&mut self, ex: &Execution, n: usize) -> Result<(), String> {
        let total: usize = ex.allocations.values().map(|a| a.bytes).sum();
        if total == 0 || std::env::var("AD_PIN_PREFILL_CLI").as_deref() == Ok("0") { return Ok(()); }
        let room = self.bs_ram_budget.saturating_sub(self.bs_ram_now);
        let n = n.min(room / total);
        for i in 0..n {
            let (fns, set, c) = (crate::pinned::PinFns::of(&ex.cuda), ex.cuda.ctx_set_current, ex.context as usize);
            self.pin_q.push_back((total, std::thread::Builder::new().name(format!("x20-pin-q{i}")).spawn(move || {
                let r = set(c as *mut std::ffi::c_void); if r != 0 { return Err(format!("cuCtxSetCurrent pin prefill: {r}")); }
                crate::pinned::PinMem::new(fns, total)
            }).map_err(|e| e.to_string())?));
        }
        Ok(())
    }
    unsafe fn store_inner(&mut self, ex: &mut Execution, frame: &Option<ReplayFrame>, dir: &Path, s: usize) -> Result<Boundary, String> {
        let _c = capi::xfer::Cat::checkpoint();
        dev::sync(&ex.cuda)?;
        // The capture is the state at this boundary only if nothing ran since it was taken (store follows the step's
        // finish directly); the live addresses are checked against it, and anything else downloads afresh.
        let cap = ex.captured.take().filter(|(_, index)| index.len() == ex.allocations.len()
            && index.iter().all(|(id, _, n, addr)| ex.allocations.get(id).is_some_and(|a| a.bytes == *n && a.address == *addr)));
        let (data, index): (std::sync::Arc<crate::CapData>, Vec<(u64, usize, usize)>) = match cap {
            Some((d, ix)) => (d, ix.into_iter().map(|(id, off, n, _)| (id, off, n)).collect()),
            None => { let (d, ix) = self.capture_state(ex)?; (std::sync::Arc::new(crate::CapData::Vec(d)), ix.into_iter().map(|(id, off, n, _)| (id, off, n)).collect()) },
        };
        let path = dir.join(format!("b{s}.bin"));
        // S02: with --checkpoints a boundary after boundary 0 is stored as its changed chunks (lossless, see delta_encode).
        let delta = self.delta_store && self.ckpt_base.is_some();
        // host-leftovers-2: the CLI tangent sweep captured this boundary straight into page-locked RAM inside the RAM
        // budget (capture_host, set by the window loop): keep those bytes as the boundary, as the C-ABI session does,
        // instead of writing 3.4 GB to b{s}.bin on one writer thread (s4-C18: store 105 s of a 369 s window).
        if !delta { if let crate::CapData::Host(h) = &*data {
            let n = h.as_slice().len();
            self.bs_ram_now += n; self.bs_ram_peak = self.bs_ram_peak.max(self.bs_ram_now); self.bs_ram_count += 1;
            return Ok(Boundary { path, index, host: host_snap(ex, frame), step: s, delta: false, file_bytes: 0, ram: Some(h.clone()), dev: 0 });
        } }
        self.join_writer()?;
        let p = path.clone();
        let data = if delta { let (bi, bd) = self.ckpt_base.as_ref().unwrap(); std::sync::Arc::new(crate::CapData::Vec(delta_encode(&index, &data, bi, bd))) } else { data };
        let file_bytes = data.len();
        self.bs_now += file_bytes as u64; self.bs_peak = self.bs_peak.max(self.bs_now); self.bs_written += file_bytes as u64;
        self.bs_files += 1; self.bs_files_peak = self.bs_files_peak.max(self.bs_files);
        self.writer = Some(std::thread::spawn(move || fs::write(&p, &data[..]).map_err(|e| format!("{}: {e}", p.display()))));
        Ok(Boundary { path, index, host: host_snap(ex, frame), step: s, delta, file_bytes, ram: None, dev: 0 })
    }
    unsafe fn load(&mut self, b: &Boundary) -> Result<Bytes, String> {
        let t = Instant::now();
        let r = self.load_inner(b);
        self.load_s += t.elapsed().as_secs_f64();
        r
    }
    unsafe fn load_inner(&mut self, b: &Boundary) -> Result<Bytes, String> {
        // host-leftovers-2: a RAM boundary is its own page-locked bytes, shared, not copied (a 3.4 GB copy per load, and
        // restore then DMAs straight from it); the same bytes.
        if let Some(r) = &b.ram { return Ok(Bytes::Ram(r.clone())); }
        self.join_writer()?;
        if let Some((p, h)) = self.ck_pf.take() {
            let d = h.join().map_err(|_| "checkpoint prefetch panicked")?;
            if p == b.path { self.pf_hits += 1; return d.map(Bytes::Own); }
        }
        let f = fs::read(&b.path).map_err(|e| format!("{}: {e}", b.path.display()))?;
        if !b.delta { return Ok(Bytes::Own(f)); }
        let (bi, bd) = self.ckpt_base.as_ref().ok_or("changed-chunk boundary without boundary 0")?;
        delta_decode(&f, &b.index, bi, bd).map(Bytes::Own)
    }
    /// S02: read (and decode) checkpoint `b` in the background; the next load of the same file takes it.
    unsafe fn prefetch_ck(&mut self, b: &Boundary) -> Result<(), String> {
        if b.ram.is_some() || b.dev != 0 { return Ok(()); } // runner-lib: a RAM or device boundary has no file to read
        if self.ck_pf.as_ref().is_some_and(|(p, _)| *p == b.path) { return Ok(()); }
        if let Some((_, h)) = self.ck_pf.take() { let _ = h.join(); }
        self.join_writer()?;
        let (path, index, delta) = (b.path.clone(), b.index.clone(), b.delta);
        let base = if delta { Some(self.ckpt_base.clone().ok_or("changed-chunk boundary without boundary 0")?) } else { None };
        let p = path.clone();
        self.ck_pf = Some((path, std::thread::spawn(move || {
            let f = fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            match base { None => Ok(f), Some((bi, bd)) => delta_decode(&f, &index, &bi, &bd) }
        })));
        Ok(())
    }
    /// S02: delete a checkpoint file the schedule no longer needs.
    unsafe fn drop_boundary(&mut self, b: Boundary) -> Result<(), String> {
        if let Some(r) = &b.ram { self.bs_ram_now = self.bs_ram_now.saturating_sub(r.as_slice().len()); }
        if b.ram.is_some() || b.dev != 0 { return Ok(()); } // runner-lib: nothing on disk; the bytes go with the Boundary
        self.join_writer()?;
        if self.ck_pf.as_ref().is_some_and(|(p, _)| *p == b.path) { if let Some((_, h)) = self.ck_pf.take() { let _ = h.join(); } }
        fs::remove_file(&b.path).map_err(|e| format!("{}: {e}", b.path.display()))?;
        self.bs_now -= b.file_bytes as u64; self.bs_files -= 1;
        Ok(())
    }
    /// S02: one forward step without the tangent (the boundary hashes still checked against the reference), recomputing
    /// a stretch of the trajectory from a checkpoint; its command list must be the one the tangent forward recorded.
    #[allow(clippy::too_many_arguments)]
    unsafe fn advance(&mut self, ex: &mut Execution, plan: &Plan, base: &Path, frame: &mut Option<ReplayFrame>, refs: &References,
                      step: usize, flats: &[Vec<Flat>]) -> Result<(), String> {
        let t = Instant::now();
        let fl = self.forward_step(ex, plan, base, frame, refs, step, false)?;
        if fl.len() != flats[step - 1].len() {
            return Err(format!("step {step}: the recompute ran {} commands, the recorded forward {}", fl.len(), flats[step - 1].len()));
        }
        self.recompute_steps += 1;
        self.recompute_s += t.elapsed().as_secs_f64();
        Ok(())
    }
    /// S02: the reverse sweep over every step with `c` checkpoints (revolve schedule `rv`), from the cotangent already in
    /// the shadows at the window end. `ck` holds the checkpoints the tangent forward stored (the first descent); a later
    /// sweep recomputes them from boundary 0. Each step is reversed by reverse_step from its start state exactly as in
    /// the store-every-step sweep, the start state coming from a checkpoint or from a recompute.
    #[allow(clippy::too_many_arguments)]
    unsafe fn sweep_revolve(&mut self, ex: &mut Execution, plan: &Plan, base: &Path, frame: &mut Option<ReplayFrame>, refs: &References,
                            flats: &[Vec<Flat>], b0: &Boundary, bn: &Boundary, ck: &mut BTreeMap<usize, Boundary>, bdir: &Path,
                            rv: &Revolve, c: usize, anchors: usize, tag: Value) -> Result<Vec<Value>, String> {
        let steps = flats.len();
        let want = rv.descent(steps, c);
        let missing: Vec<usize> = want.iter().copied().filter(|p| !ck.contains_key(p)).collect();
        if let Some(&last) = missing.last() {
            let d = self.load(b0)?; self.restore(ex, frame, b0, &d)?; drop(d);
            for s in 1..=last {
                self.advance(ex, plan, base, frame, refs, s, flats)?;
                if missing.contains(&s) { let b = self.store(ex, frame, bdir, s)?; ck.insert(s, b); }
            }
        }
        let cur = self.load(bn)?;
        let mut st = RevSt { cur: Some((bn.clone(), cur)), rows: Vec::new(), tag };
        self.rev_range(ex, plan, base, frame, refs, flats, b0, ck, bdir, rv, anchors, 0, steps, c, &mut st)?;
        Ok(st.rows)
    }
    #[allow(clippy::too_many_arguments)]
    unsafe fn rev_range(&mut self, ex: &mut Execution, plan: &Plan, base: &Path, frame: &mut Option<ReplayFrame>, refs: &References,
                        flats: &[Vec<Flat>], b0: &Boundary, ck: &mut BTreeMap<usize, Boundary>, bdir: &Path, rv: &Revolve,
                        anchors: usize, lo: usize, hi: usize, k: usize, st: &mut RevSt) -> Result<(), String> {
        if hi - lo == 1 { return self.rev_one(ex, base, frame, flats, b0, ck, anchors, lo, None, hi, st); }
        if k == 0 {
            for s in (lo + 1..=hi).rev() {
                if s - 1 == lo { self.rev_one(ex, base, frame, flats, b0, ck, anchors, lo, None, s, st)?; continue; }
                {
                    let b = if lo == 0 { b0 } else { ck.get(&lo).ok_or("checkpoint missing")? };
                    let d = self.load(b)?; self.restore(ex, frame, b, &d)?;
                }
                for t in lo + 1..s { self.advance(ex, plan, base, frame, refs, t, flats)?; }
                let (mut b, d) = self.snap_ram(ex, frame)?;
                b.step = s - 1;
                // integ8 (S02 + S04): snap_ram hands out a reused page-locked buffer; this start state becomes the sweep's
                // owned current state (RevSt), so it is copied out once and the buffer goes back to the pool.
                let dv = d.to_vec();
                self.anchor_pool.push(d);
                self.rev_one(ex, base, frame, flats, b0, ck, anchors, lo, Some((b, Bytes::Own(dv))), s, st)?;
            }
            return Ok(());
        }
        let mid = lo + rv.split(hi - lo, k);
        if !ck.contains_key(&mid) {
            {
                let b = if lo == 0 { b0 } else { ck.get(&lo).ok_or("checkpoint missing")? };
                let d = self.load(b)?; self.restore(ex, frame, b, &d)?;
            }
            for t in lo + 1..=mid { self.advance(ex, plan, base, frame, refs, t, flats)?; }
            let b = self.store(ex, frame, bdir, mid)?;
            ck.insert(mid, b);
        }
        self.rev_range(ex, plan, base, frame, refs, flats, b0, ck, bdir, rv, anchors, mid, hi, k - 1, st)?;
        let b = ck.remove(&mid).ok_or("checkpoint missing")?;
        self.drop_boundary(b)?;
        self.rev_range(ex, plan, base, frame, refs, flats, b0, ck, bdir, rv, anchors, lo, mid, k, st)
    }
    /// Reverse step `s` from its start state: the checkpoint at `lo` (= s - 1) or a recomputed RAM state `prev`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn rev_one(&mut self, ex: &mut Execution, base: &Path, frame: &mut Option<ReplayFrame>, flats: &[Vec<Flat>], b0: &Boundary,
                      ck: &BTreeMap<usize, Boundary>, anchors: usize, lo: usize, prev: Option<(Boundary, Bytes)>, s: usize, st: &mut RevSt)
                      -> Result<(), String> {
        let (pb, pd) = match prev {
            Some(x) => x,
            None => {
                if lo != s - 1 { return Err("reverse step without its start state".into()); }
                let b = if lo == 0 { b0 } else { ck.get(&lo).ok_or("checkpoint missing")? };
                let d = self.load(b)?;
                (b.clone(), d)
            }
        };
        // The next step reversed is s - 1; its start state s - 2 comes from the nearest stored state at or below s - 2
        // (a checkpoint load, or a restore before a recompute), which no schedule step drops first: read it now.
        if s >= 2 {
            let nb = ck.range(..=s - 2).next_back().map(|(_, b)| b).unwrap_or(b0);
            self.prefetch_ck(nb)?;
        }
        self.stage_blobs(ex, base, flat_blobs(&flats[s - 1]), (s >= 2).then(|| flat_blobs(&flats[s - 2])))?;
        let (cb, cd) = st.cur.take().ok_or("reverse sweep lost its current state")?;
        let r = self.reverse_step(ex, base, frame, s, &flats[s - 1], &pb, &pd, &cb, &cd, anchors)?;
        drop((cb, cd));
        eprintln!("{}", json!({"event":"window_reverse_step","seed":st.tag["seed"],"repeat":st.tag["repeat"],"sweep":st.tag["sweep"],"row":r,
                               "recompute_steps":self.recompute_steps,"bstore_bytes_now":self.bs_now}));
        st.rows.push(r);
        st.cur = Some((pb, pd));
        Ok(())
    }
    unsafe fn restore(&mut self, ex: &mut Execution, frame: &mut Option<ReplayFrame>, b: &Boundary, data: &[u8]) -> Result<(), String> {
        let _c = capi::xfer::Cat::checkpoint();
        let t = dev::prof::start();
        let t1 = Instant::now();
        drain_pinned(ex)?;
        // adjoint-speed-2 (pkg18m2 reverse: 3.4 GB pageable upload per step): the boundary bytes are copied into one
        // reused page-locked buffer on all cores, then DMA'd from there. The same bytes reach the same addresses.
        let total = b.index.iter().map(|(_, off, n)| off + n).max().unwrap_or(0);
        // host-leftovers-2: bytes that are already page-locked (a RAM boundary, a RAM anchor, a C-ABI boundary) are DMA'd
        // from where they are; only other bytes are staged through the reused buffer (pk120: restore 8-16 s per reverse
        // step over 26-48 segments, most of it this copy). The same bytes reach the same addresses.
        let direct = crate::pinned::is_page_locked(&data[..total]);
        let mut pin = if direct { None } else {
            let mut p = match self.restore_pin.take() { Some(p) if p.len() >= total => p, _ => crate::pinned::PinnedBuf::new(&ex.cuda, total.max(1))? };
            par_copy(&mut p[..total], &data[..total]);
            Some(p)
        };
        let src: &[u8] = match &pin { Some(p) => &p[..total], None => &data[..total] };
        for (id, off, n) in &b.index {
            let a = ex.allocations.get(id).ok_or("boundary allocation not live")?;
            if a.bytes != *n { return Err(format!("allocation {id} changed size since the boundary")); }
            checked("cuMemcpyHtoD restore", (ex.cuda.upload)(a.address, src[*off..*off + *n].as_ptr().cast(), *n))?;
        }
        dev::sync(&ex.cuda)?;
        if direct { self.restore_direct += 1; }
        if let Some(p) = pin.take() { self.restore_pin = Some(p); }
        let r = restore_host(ex, frame, &b.host);
        dev::prof::add(6, t);
        self.restore_s += t1.elapsed().as_secs_f64();
        r
    }
    /// adjoint-speed-2: the live state in the stored boundary's layout, DMA'd into a reused page-locked buffer.
    unsafe fn redo_download(&mut self, ex: &Execution, b: &Boundary) -> Result<crate::pinned::PinnedBuf, String> {
        dev::sync(&ex.cuda)?;
        let total = b.index.iter().map(|(_, off, n)| off + n).max().unwrap_or(0);
        let mut pin = match self.cmp_pin.take() { Some(p) if p.len() >= total => p, _ => crate::pinned::PinnedBuf::new(&ex.cuda, total.max(1))? };
        for (id, off, n) in &b.index {
            if *n != 0 { checked("cuMemcpyDtoH compare", (ex.cuda.download)(pin[*off..*off + *n].as_mut_ptr().cast(), ex.allocations[id].address, *n))?; }
        }
        Ok(pin)
    }
    /// 32-bit words of the live state that differ from a stored boundary (declared TEMPO residue ranges counted apart).
    /// Streamed one allocation at a time; large allocations are compared on all cores.
    unsafe fn differing(&self, ex: &Execution, b: &Boundary, data: &[u8], step: usize) -> Result<(u64, u64), String> {
        let _c = capi::xfer::Cat::checkpoint();
        dev::sync(&ex.cuda)?;
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16);
        let (mut d, mut r) = (0u64, 0u64);
        let mut buf: Vec<u8> = Vec::new();
        for (id, off, n) in &b.index {
            buf.resize(*n, 0);
            checked("cuMemcpyDtoH compare", (ex.cuda.download)(buf.as_mut_ptr().cast(), ex.allocations[id].address, *n))?;
            let ranges = ex.residue.get(&(step, *id)).map(|x| x.0.clone()).unwrap_or_default();
            let want = &data[*off..*off + *n];
            let count = |lo: usize, x: &[u8], y: &[u8]| -> (u64, u64) {
                let (mut d, mut r) = (0u64, 0u64);
                for (w, (p, q)) in x.chunks(4).zip(y.chunks(4)).enumerate() {
                    if p == q { continue; }
                    let at = lo + 4 * w;
                    if ranges.iter().any(|&(o, l)| at >= o && at < o + l) { r += 1 } else { d += 1 }
                }
                (d, r)
            };
            let c = if *n < (16 << 20) { count(0, &buf, want) } else {
                let chunk = (*n / 4).div_ceil(threads) * 4;
                std::thread::scope(|sc| {
                    let hs: Vec<_> = buf.chunks(chunk).zip(want.chunks(chunk)).enumerate()
                        .map(|(k, (x, y))| { let count = &count; sc.spawn(move || count(k * chunk, x, y)) }).collect();
                    hs.into_iter().map(|h| h.join().unwrap()).fold((0, 0), |s, c| (s.0 + c.0, s.1 + c.1))
                })
            };
            if c.0 > 0 {
                // G2: name what a failed restore + redo differs in (allocation, words, first differing byte offset).
                let first = buf.chunks(4).zip(want.chunks(4)).enumerate().find(|(w, (p, q))| p != q
                    && !ranges.iter().any(|&(o, l)| 4 * w >= o && 4 * w < o + l)).map(|(w, _)| 4 * w);
                eprintln!("{}", json!({"event":"window_redo_differs","step":step,"allocation":id,"words":c.0,"first_byte":first,"bytes":n}));
            }
            d += c.0; r += c.1;
        }
        Ok((d, r))
    }

    /// Replay [lo, end) of a step's flat list, taping every derivative input. Stops at the first command whose
    /// tape does not fit (state is then just before it, nothing of it executed).
    #[allow(clippy::too_many_arguments)]
    unsafe fn tape_range(&mut self, ex: &mut Execution, base: &Path, frame: &mut Option<ReplayFrame>, step: usize,
                         flat: &[Flat], lo: usize, end: usize) -> Result<Taping, String> {
        self.tape.reset();
        self.precache.clear();
        self.seg_rows.clear();
        if let Some(c) = self.census.as_mut() { c.begin_segment(step); }
        let mut tape = Vec::with_capacity(end - lo);
        for (i, fl) in flat.iter().enumerate().take(end).skip(lo) {
            ex.rk = fl.rk.clone();
            let op = &*fl.op;
            // tape-dedupe-and-pack: what this command takes from the tape, by part (host counters only, no sync).
            let (u_op, ur_op) = (self.tape.used, self.precache.unshared_bytes);
            let mut post_taken = 0usize;
            let mut cpost: Vec<(u64, usize, usize, u64)> = Vec::new();
            let (r, l) = self.rule(ex, op, step, base, frame.as_ref())?;
            match (r, l) {
                (R::Entry, Some((f, args, grid, block, shared))) => {
                    let kd = &self.kernels[&f];
                    let li = deriv::LaunchInfo { args: &args, grid, block, shared };
                    let tp = dev::prof::start();
                    let cache = if self.pack { Some((&mut self.precache, &self.support)) } else { None };
                    let mut b = match kd.prepare_tape(&ex.cuda, &mut self.tape, &li, &ex.allocations, 4, Some(&self.shadows), self.pack, cache) {
                        Ok(b) => b, Err(e) if exhausted(&e) => return Ok(Taping::Overflow(i)),
                        Err(e) => return Err(format!("step {step} {f} reverse bind: {e}")),
                    };
                    dev::prof::add(2, tp);
                    if !ordered(ex) { dev::sync(&ex.cuda)?; }
                    exec(ex, op, step, base, frame)?;
                    // packed-tape: a shared pre-launch copy stays shared only if this launch left that span unchanged
                    // (then its derivative entries, which recompute the same forward values, cannot change the shared
                    // copy an earlier launch reads later in the walk-back). Otherwise it gets its own copy of the same
                    // bytes (taken from the shared copy, which still holds the pre-launch values).
                    // x20 replay-compiled: every word check this launch needs after its forward (each shared pre-launch
                    // span against the live words, each post span against the copies PreCache::lookup would try) is
                    // queued into its own word slot and read back with ONE collect, then decided in the old order with
                    // the old counters. THE BREAKAGE THIS FIXES: each check was its own count launch and a blocking
                    // 24-byte readback (cuda-graph-capture: ~3,700-4,000 per pkg18m2 AD step, 2,257-3,318 per pk120
                    // step), a full drain of the card per check. Same decisions, same copies, same words.
                    // AD_CHECK_BATCH=0 keeps one readback per check.
                    let posts: Vec<(u64, (usize, usize))> = if self.pack { self.kernels[&f].post_spans(&li, &b.env, &ex.allocations)?.into_iter().collect() } else { Vec::new() };
                    let cands: Vec<(Option<(usize, u64)>, Option<(usize, u64)>)> = posts.iter().map(|&(al, (lo, hi))| self.precache.candidates(al, lo, hi)).collect();
                    let nsh = b.shared.len();
                    let nslot = nsh + 2 * posts.len();
                    let words: Option<Vec<[u64; 3]>> = if ordered(ex) && nslot > 1 && nslot <= dev::SLOTS && std::env::var("AD_CHECK_BATCH").as_deref() != Ok("0") {
                        self.support.reset_words(&ex.cuda)?;
                        for (k, &(al, lo, hi, at)) in b.shared.iter().enumerate() {
                            let live = ex.allocations.get(&al).ok_or("shared primal allocation not live")?.address + lo as u64;
                            if hi > lo { self.support.words_into(&ex.cuda, k, live, at, hi - lo)?; }
                        }
                        for (j, (&(al, (lo, hi)), &(s, wc))) in posts.iter().zip(&cands).enumerate() {
                            let a = ex.allocations.get(&al).ok_or("post allocation not live")?.address + lo as u64;
                            if let Some((clo, copy)) = s { self.support.words_into(&ex.cuda, nsh + 2 * j, a, copy + (lo - clo) as u64, hi - lo)?; }
                            if let Some((clo, copy)) = wc { self.support.words_into(&ex.cuda, nsh + 2 * j + 1, a, copy + (lo - clo) as u64, hi - lo)?; }
                        }
                        Some(self.support.collect(&ex.cuda, 0, nslot)?.1)
                    } else { None };
                    if !b.shared.is_empty() {
                        // same_words reads back with a blocking cuMemcpyDtoH, which already waits for the forward above.
                        if !ordered(ex) { dev::sync(&ex.cuda)?; }
                        for (k, (al, lo, hi, at)) in std::mem::take(&mut b.shared).into_iter().enumerate() {
                            let live = ex.allocations.get(&al).ok_or("shared primal allocation not live")?.address + lo as u64;
                            let same = match &words { Some(wv) => hi <= lo || wv[k][0] == 0, None => deriv::same_words(&ex.cuda, &self.support, live, at, hi - lo)? };
                            if same { b.shared.push((al, lo, hi, at)); continue; }
                            if deriv::tape_block_bytes() > 0 {
                                // tape-dedupe-and-pack (blocks): the shared copy still holds the pre-launch words; the walk-back
                                // rebuilds the span from it in the reverse scratch instead of a second tape copy.
                                let names = b.cluster_args.get(&(al, lo)).cloned().unwrap_or_default();
                                let rt = dev::run_rows(&[(0, at, hi - lo)]);
                                let tab = match self.tape.take(rt.len() * 8) { Ok(c) => c, Err(e) if exhausted(&e) => return Ok(Taping::Overflow(i)), Err(e) => return Err(e) };
                                dev::upload(&ex.cuda, tab, std::slice::from_raw_parts(rt.as_ptr().cast::<u8>(), rt.len() * 8))?;
                                b.recon.push((al, lo, hi, (tab, rt.len() / 3), names, true));
                                self.precache.blk_spans += 1; self.precache.blk_hit_bytes += hi - lo;
                                continue;
                            }
                            let c = match self.tape.take(hi - lo) { Ok(c) => c, Err(e) if exhausted(&e) => return Ok(Taping::Overflow(i)), Err(e) => return Err(e) };
                            dev::copy(&ex.cuda, c, at, hi - lo)?;
                            b.pre.insert(al, c.wrapping_sub(lo as u64));
                            if let Some(ns) = b.cluster_args.get(&(al, lo)).cloned() { for n in ns { b.pre_arg.insert(n, c.wrapping_sub(lo as u64)); } }
                            if self.census.is_some() { b.copies.push((al, lo, hi, lo, hi, c | (1u64 << 63))); }
                            self.precache.unshared += 1; self.precache.unshared_bytes += hi - lo;
                        }
                    }
                    let kd = &self.kernels[&f];
                    let mut post = HashMap::new();
                    if self.pack {
                        // packed-tape: the bound span only (see deriv::post_spans), shared with an equal earlier copy.
                        for (j, &(al, (lo, hi))) in posts.iter().enumerate() {
                            let a = ex.allocations[&al];
                            let found = match &words {
                                Some(wv) => self.precache.decide(cands[j], lo, hi, wv[nsh + 2 * j][0] == 0, wv[nsh + 2 * j + 1][0] == 0),
                                None => self.precache.lookup(&ex.cuda, &self.support, al, a.address, lo, hi)?,
                            };
                            if let Some(base) = found {
                                post.insert(al, submit::Allocation { id: a.id, address: base, bytes: a.bytes });
                                continue;
                            }
                            let u_p = self.tape.used;
                            let copy = match self.tape.take(hi - lo) { Ok(c) => c, Err(e) if exhausted(&e) => return Ok(Taping::Overflow(i)), Err(e) => return Err(e) };
                            post_taken += self.tape.used - u_p;
                            let tc = dev::prof::start(); dev::copy(&ex.cuda, copy, a.address + lo as u64, hi - lo)?; dev::prof::add(10, tc);
                            if self.census.is_some() { cpost.push((al, lo, hi, copy)); }
                            self.precache.insert(al, lo, hi, copy);
                            post.insert(al, submit::Allocation { id: a.id, address: copy.wrapping_sub(lo as u64), bytes: a.bytes });
                        }
                    }
                    for e in kd.ad_order.iter().chain(kd.aug.then_some(&"aug".to_owned())).filter(|_| !self.pack) {
                        for bind in kd.spec["launch_abi"]["entries"][e]["args"].as_array().into_iter().flatten() {
                            if bind["bind"] != "primal" || bind["post"] != true { continue; }
                            let Some(name) = bind["arg"].as_str() else { continue };
                            let Some(k) = kd.names.iter().position(|n| n == name) else { continue };
                            if let submit::Argument::Device(s) = &args[k] {
                                if post.contains_key(&s.allocation) { continue; }
                                let a = ex.allocations[&s.allocation];
                                let copy = match self.tape.take(a.bytes) { Ok(c) => c, Err(e) if exhausted(&e) => {
                                    // The command already ran: the segment ends after it is undone by the caller's restore.
                                    return Ok(Taping::Overflow(i)) }, Err(e) => return Err(e) };
                                let tc = dev::prof::start(); dev::copy(&ex.cuda, copy, a.address, a.bytes)?; dev::prof::add(10, tc);
                                post.insert(s.allocation, submit::Allocation { id: a.id, address: copy, bytes: a.bytes });
                            }
                        }
                    }
                    tape.push(Tp::Entry { f, b, args, grid, block, shared, post });
                },
                (R::Native(_), Some((f, args, grid, block, _))) => {
                    let n = self.native.as_mut().ok_or("native rows not loaded")?;
                    let tp = dev::prof::start();
                    let p = match if self.pack { n.prepare_deferred(&ex.cuda, &mut self.tape, &ex.allocations, &f, &args, grid, block) } else { n.prepare(&ex.cuda, &mut self.tape, &ex.allocations, &f, &args, grid, block) } {
                        Ok(p) => p, Err(e) if exhausted(&e) => return Ok(Taping::Overflow(i)),
                        Err(e) => return Err(format!("step {step} {f} native reverse bind: {e}")),
                    };
                    dev::prof::add(2, tp);
                    dev::sync(&ex.cuda)?;
                    exec(ex, op, step, base, frame)?;
                    tape.push(Tp::Native { f, p });
                },
                (R::Tempo(_), Some((f, args, _, _, _))) => {
                    let snap = { let (t, _) = self.tempo.as_ref().ok_or("TEMPO derivative not loaded")?;
                        match t.snapshot(&ex.cuda, &mut self.tape, &ex.allocations, &args) {
                            Ok(s) => s, Err(e) if exhausted(&e) => return Ok(Taping::Overflow(i)),
                            Err(e) => return Err(format!("step {step} {f} TEMPO reverse snapshot: {e}")),
                        } };
                    exec(ex, op, step, base, frame)?;
                    tape.push(Tp::Tempo { f, snap });
                },
                (R::TempoSerial(_), Some((f, args, _, _, _))) => {
                    let t = self.tempo_serial.as_ref().ok_or("TEMPO serial entry not loaded")?;
                    let snap = match t.snapshot(&ex.cuda, &mut self.tape, &ex.allocations, &f, &args) {
                        Ok(s) => s, Err(e) if exhausted(&e) => return Ok(Taping::Overflow(i)),
                        Err(e) => return Err(format!("step {step} {f} tempo reverse bind: {e}")),
                    };
                    let ins = Self::tempo_segs(ex, &snap.input_list())?;
                    let outs = Self::tempo_segs(ex, &snap.outputs())?;
                    exec(ex, op, step, base, frame)?;
                    tape.push(Tp::TempoSerial { f, snap, ins, outs });
                },
                (R::TempoSmall(_), Some((f, args, _, _, _))) => {
                    let (ins, outs) = self.small_slots(ex, &f, &args).map_err(|e| format!("step {step} {f} tempo-small slots: {e}"))?;
                    let ts = self.tempo_small.as_ref().ok_or("--tempo-small entry not loaded")?;
                    let snap = match ts.window_snapshot(&ex.cuda, &mut self.tape, &ex.allocations, &f, &args) {
                        Ok(s) => s, Err(e) if exhausted(&e) => return Ok(Taping::Overflow(i)),
                        Err(e) => return Err(format!("step {step} {f} tempo-small reverse snapshot: {e}")),
                    };
                    exec(ex, op, step, base, frame)?;
                    tape.push(Tp::TempoSmall { f, snap, ins, outs });
                },
                (r @ R::FctMinmax { .. }, _) => {
                    let R::FctMinmax { prim, .. } = &r else { unreachable!() };
                    let p = match self.tape_fct(ex, prim) { Ok(p) => p, Err(e) if exhausted(&e) => return Ok(Taping::Overflow(i)), Err(e) => return Err(e) };
                    exec(ex, op, step, base, frame)?;
                    tape.push(Tp::Rule { r, prim: p });
                },
                (r @ R::Max { .. }, _) | (r @ R::Where { .. }, _) => {
                    let (al, first, n, unit) = match &r { R::Max { prim: (a, f, n), .. } => (*a, *f, *n, 4), R::Where { prim: (a, f, n), .. } => (*a, *f, *n, 1), _ => unreachable!() };
                    let p = match self.tape_primal(ex, al, first, n, unit) { Ok(p) => p, Err(e) if exhausted(&e) => return Ok(Taping::Overflow(i)), Err(e) => return Err(e) };
                    exec(ex, op, step, base, frame)?;
                    tape.push(Tp::Rule { r, prim: p });
                },
                (r, _) => { exec(ex, op, step, base, frame)?; tape.push(Tp::Rule { r, prim: 0 }); },
            }
            let taken = self.tape.used - u_op;
            let recopy = self.precache.unshared_bytes - ur_op;
            let (key, row) = match tape.last() {
                Some(Tp::Entry { f, b, .. }) => {
                    let other = taken.saturating_sub(b.parts[0] + b.parts[2] + post_taken + recopy);
                    (f.clone(), [1, b.parts[0], b.parts[1], b.parts[2], post_taken, recopy, other, taken])
                },
                Some(Tp::Native { f, .. }) => (format!("native:{f}"), [1, 0, 0, 0, 0, 0, taken, taken]),
                Some(Tp::Tempo { f, .. }) | Some(Tp::TempoSerial { f, .. }) | Some(Tp::TempoSmall { f, .. }) => (format!("tempo:{f}"), [1, 0, 0, 0, 0, 0, taken, taken]),
                Some(Tp::Rule { r, .. }) => (format!("rule:{}", format!("{r:?}").split(|c: char| !c.is_alphanumeric()).next().unwrap_or("?")), [1, 0, 0, 0, 0, 0, taken, taken]),
                None => (String::new(), [0; 8]),
            };
            if taken > 0 || row[2] > 0 || matches!(tape.last(), Some(Tp::Entry { .. })) {
                let e = self.seg_rows.entry(key.clone()).or_insert([0; 8]);
                for k in 0..8 { e[k] += row[k]; }
            }
            if let Some(c) = self.census.as_mut() {
                let copies: Vec<(u64, usize, usize, usize, usize, u64)> = match tape.last() { Some(Tp::Entry { b, .. }) => b.copies.clone(), _ => Vec::new() };
                c.op(&ex.cuda, i, &key, u_op, self.tape.used, self.tape.base, &copies, &cpost)?;
            }
        }
        if self.pack { self.promote_window(ex)?; }
        let seg = std::mem::take(&mut self.seg_rows);
        for (k, r) in seg { let e = self.rev_rows.entry(k).or_insert([0; 8]); for j in 0..8 { e[j] += r[j]; } }
        if let Some(c) = self.census.as_mut() { c.end_segment()?; }
        Ok(Taping::Done(tape))
    }
    /// tape-dedupe-and-pack (window cache): promote this segment's pre-launch copies whose span the segment's whole forward
    /// left unchanged (live == copy now, word for word, batched: one readback per 512 compares) into the window arena, so
    /// later segments and steps bind them instead of copying again (every use is still checked word for word against the
    /// live bytes by PreCache::lookup, so a span that changes later is simply not bound). THE BREAKAGE: mesh, metric and
    /// other window-constant inputs were copied onto every segment's tape again (pk120: 26-48 segments per step). The
    /// window arena is taken once, at the first promotion: AD_TAPE_WINCACHE_MIB, default 1/4 of the tape, and never more
    /// than the card's free memory less 2 GiB; AD_TAPE_WINCACHE_MIB=0 turns it off. Words: a bound copy holds the same
    /// bytes the launch would have copied (checked), so the reverse words are unchanged.
    unsafe fn promote_window(&mut self, ex: &Execution) -> Result<(), String> {
        let cands: Vec<(u64, usize, usize, u64)> = std::mem::take(&mut self.precache.fresh).into_iter()
            .filter(|(al, lo, hi, _)| hi > lo && (hi - lo) % 4 == 0 && !self.precache.win_covers(*al, *lo, *hi)).collect();
        if cands.is_empty() || self.win_cap == 0 { return Ok(()); }
        if self.win.is_none() {
            let (mut free, mut total) = (0usize, 0usize);
            let _ = (ex.cuda.mem_info)(&mut free, &mut total);
            let want = self.win_cap.min(self.tape.bytes / 4).min(free.saturating_sub(2 << 30)) & !((1 << 20) - 1);
            if want < (16 << 20) { self.win_cap = 0; eprintln!("{}", json!({"event":"window_cache_off","reason":"no room","card_free_bytes":free})); return Ok(()); }
            match dev::Arena::new(&ex.cuda, want) {
                Ok(a) => { eprintln!("{}", json!({"event":"window_cache_arena","bytes":want})); self.win = Some(a); },
                Err(e) => { self.win_cap = 0; eprintln!("{}", json!({"event":"window_cache_off","reason":e})); return Ok(()); },
            }
        }
        for chunk in cands.chunks(dev::SLOTS) {
            self.support.reset_words(&ex.cuda)?;
            for (k, (al, lo, hi, copy)) in chunk.iter().enumerate() {
                let a = ex.allocations.get(al).ok_or("window cache: allocation not live")?;
                self.support.words_into(&ex.cuda, k, a.address + *lo as u64, *copy, hi - lo)?;
            }
            let (_, w) = self.support.collect(&ex.cuda, 0, chunk.len())?;
            for ((al, lo, hi, copy), c) in chunk.iter().zip(w) {
                if c[0] != 0 || self.precache.win_covers(*al, *lo, *hi) { continue; }
                let Some(arena) = self.win.as_mut() else { return Ok(()) };
                let Ok(dst) = arena.take(hi - lo) else { continue };
                dev::copy(&ex.cuda, dst, *copy, hi - lo)?;
                self.precache.win.entry(*al).or_default().push((*lo, *hi, dst));
                self.win_promoted.0 += 1; self.win_promoted.1 += hi - lo;
            }
        }
        Ok(())
    }
    /// tape-dedupe-and-pack: the step's reverse tape by row (summed over the segments that were walked back), largest
    /// first, with totals; then cleared.
    fn take_rev_rows(&mut self, cells: usize) -> Value {
        let rows = std::mem::take(&mut self.rev_rows);
        let mut v: Vec<(String, [usize; 8])> = rows.into_iter().collect();
        v.sort_by(|a, b| b.1[7].cmp(&a.1[7]));
        let mut sum = [0usize; 8];
        for (_, r) in &v { for k in 0..8 { sum[k] += r[k]; } }
        json!({"fields":"[commands, pre-launch copies taken, pre-launch bytes shared (not taken), shadows/cells/tables, post-launch copies taken, re-copies after the forward changed a shared span, other (native, TEMPO, rules), all taken]",
               "totals":sum,"bytes_per_cell":if cells > 0 { sum[7] as f64 / cells as f64 } else { 0.0 },
               "rows":v.iter().take(48).map(|(f, r)| json!([f, r])).collect::<Vec<_>>()})
    }
    unsafe fn walk_back(&mut self, cuda: &Cuda, ex: &Execution, step: usize, tape: Vec<Tp>) -> Result<(), String> {
        let ntape = tape.len();
        let light = ordered(ex);
        for (j, t) in tape.into_iter().enumerate().rev() {
            if self.sync_owed && !matches!(t, Tp::Entry { .. }) && !(light && matches!(t, Tp::Rule { .. })) { dev::sync(cuda)?; self.sync_owed = false; }
            let label = if self.trace { match &t { Tp::Entry { f, .. } | Tp::Native { f, .. } | Tp::Tempo { f, .. } | Tp::TempoSerial { f, .. } | Tp::TempoSmall { f, .. } => f.clone(), Tp::Rule { r, .. } => format!("{r:?}").chars().take(160).collect() } } else { String::new() };
            match t {
                Tp::Entry { f, mut b, args, grid, block, shared, post } => {
                    let t0 = Instant::now();
                    let kd = &self.kernels[&f];
                    let li = deriv::LaunchInfo { args: &args, grid, block, shared };
                    let mut live = ex.allocations.clone();
                    for (id, a) in &post { live.insert(*id, *a); }
                    self.scratch.reset();
                    b.workspace.clear();
                    kd.bind_transient(cuda, &mut self.scratch, &mut b, &li).map_err(|e| format!("step {step} {f} transient bind: {e}"))?;
                    kd.bind_recon(cuda, &self.support, &mut self.scratch, &mut b).map_err(|e| format!("step {step} {f} block rebuild: {e}"))?;
                    if kd.aug { kd.launch(cuda, &mut self.scratch, &mut b, &li, &live, "aug")?; }
                    for c in b.ad_cells.values() { dev::zero(cuda, c.0, 8)?; }
                    kd.snapshot_seeds(cuda, &b, &li)?;
                    // AD_SUBNORMAL=f1,f2 (diagnostic only): the argument shadows' census before and after this adjoint.
                    let census = std::env::var("AD_SUBNORMAL").ok().is_some_and(|v| v.split(',').any(|x| x == f));
                    let before = if census { dev::sync(cuda)?; kd.shadow_census(cuda, &b, &li) } else { Vec::new() };
                    { let t = dev::prof::start(); kd.begin_ad(cuda, &self.support, &mut b, &li)?; dev::prof::add(1, t); }
                    // AD_ZERO_DERIV (bisection diagnostic): the transpose of "outputs constant" consumes the seeds and adds
                    // nothing to the inputs, so the adjoint entries are skipped and the inout seed views zeroed.
                    let zero_d = std::env::var("AD_ZERO_DERIV").ok().is_some_and(|v| v.split(',').any(|x| x == f));
                    if zero_d { dev::sync(cuda)?; kd.zero_written(cuda, &b, &li, true)?; }
                    else { let t = dev::prof::start(); for e in &kd.ad_order { kd.launch(cuda, &mut self.scratch, &mut b, &li, &live, e)?; } dev::prof::add(5, t); }
                    { let t = dev::prof::start(); kd.finish_ad(cuda, &mut b, &li, Some(&self.support))?; dev::prof::add(1, t); }
                    if zero_d { dev::sync(cuda)?; kd.zero_written(cuda, &b, &li, true)?; }
                    if census { dev::sync(cuda)?; eprintln!("{}", json!({"event":"window_ad_census","step":step,"command":j,"function":f,
                        "fields":"[arg, words, nonzero, subnormal, min nonzero |b|, max |b|]","before":before,"after":kd.shadow_census(cuda, &b, &li)})); }
                    let t = dev::prof::start();
                    kd.queue_failures(cuda, &mut b, self.ledger.as_mut().ok_or("step ledger missing")?, &format!("step {step} {f} adjoint"))?;
                    dev::prof::add(7, t);
                    if entry_sync() { dev::sync(cuda)?; } else { self.sync_owed = true; }
                    self.note("ad", &f, t0);
                },
                Tp::Rule { r, prim } => { let t = dev::prof::start(); self.ad(cuda, &r, prim)?; dev::prof::add(11, t); },
                Tp::Tempo { f, snap } => {
                    let t0 = Instant::now();
                    self.tempo_apply(ex, &snap, step, true).map_err(|e| format!("step {step} {f} TEMPO adjoint: {e}"))?;
                    self.note("ad", &f, t0);
                },
                Tp::TempoSerial { f, snap, ins, outs } => {
                    // Transpose of the tangent above: ybar from the output shadows (then cleared: the forward overwrote
                    // them), the adjoint sweep, then s/d (written in place) take their xbar and the read-only f[3], f[4],
                    // f[18] slots accumulate it (widen, add in float64, round back once).
                    let t0 = Instant::now();
                    self.scratch.reset();
                    let (nin, nout) = snap.io_words();
                    let ro = 8 * snap.read_only_words() as u64;
                    let (yb, xb, w) = (self.scratch.take(8 * nout)?, self.scratch.take(8 * nin)?, self.scratch.take((ro as usize).max(8))?);
                    dev::zero(cuda, yb, 8 * nout)?;
                    self.xfer(cuda, &Self::based(&outs, yb), true)?;
                    for (al, lo, hi, _) in &outs {
                        let e = self.base_el.get(al).ok_or_else(|| format!("tempo_precip: allocation {al} has no window shadow"))? + (*lo / 4) as i64;
                        dev::zero(cuda, self.sh(e), hi - lo)?;
                    }
                    let t = self.tempo_serial.as_ref().ok_or("TEMPO serial entry not loaded")?;
                    let tp = t.record_tape(cuda, &mut self.scratch, &snap).map_err(|e| format!("step {step} {f} tempo reverse record: {e}"))?;
                    t.sweep_dev(cuda, &snap, &tp, true, yb, xb)?;
                    let (ro_in, rw_in): (Vec<_>, Vec<_>) = ins.iter().copied().partition(|s| s.3 < ro);
                    self.xfer(cuda, &Self::based(&rw_in, xb), false)?;
                    let wsegs = Self::based(&ro_in, w);
                    self.xfer(cuda, &wsegs, true)?;
                    for (_, lo, hi, o) in &ro_in { self.support.move_add(cuda, w + o, xb + o, 2 * (hi - lo), 8)?; }
                    self.xfer(cuda, &wsegs, false)?;
                    dev::sync(cuda)?;
                    self.note("ad", &f, t0);
                },
                Tp::Native { f, mut p } => {
                    let t0 = Instant::now();
                    self.scratch.reset();
                    self.native.as_mut().ok_or("native rows not loaded")?.bind_transient(cuda, &mut self.scratch, &ex.allocations, &mut p)
                        .map_err(|e| format!("step {step} {f} native transient bind: {e}"))?;
                    let segs = p.active_segments();
                    self.xfer(cuda, &segs, true)?;
                    let tn = dev::prof::start();
                    let nn = self.native.as_mut().ok_or("native rows not loaded")?;
                    nn.ledger = self.ledger.take();
                    let info = nn.reverse(cuda, &mut self.scratch, &mut p);
                    self.ledger = nn.ledger.take();
                    let info = info.map_err(|e| format!("step {step} {f} native adjoint: {e}"))?;
                    dev::prof::add(8, tn);
                    self.xfer(cuda, &segs, false)?;
                    dev::sync(cuda)?;
                    eprintln!("{}", json!({"event":"window_native","pass":"ad","step":step,"function":f,"ms":t0.elapsed().as_secs_f64() * 1e3,"info":info}));
                    self.note("ad", &f, t0);
                },
                Tp::TempoSmall { f, snap, ins, outs } => {
                    let t0 = Instant::now();
                    self.small_sweep(cuda, &f, &snap, &ins, &outs, true, false).map_err(|e| format!("step {step} {f} tempo-small adjoint: {e}"))?;
                    self.note("ad", &f, t0);
                },
            }
            if self.trace {
                // Reverse diagnostic: the first command (walking back) that leaves a non-finite cotangent, by name.
                let mut z = [0u64]; dev::zero(cuda, self.part, 8)?;
                let (mut a, mut n, mut c) = (self.sh_base, self.words, self.part);
                launch(cuda, self.f.nonfinite, [4096, 1, 1], kargs!(a, n, c))?;
                checked("cuMemcpyDtoH nonfinite", (cuda.download)(z.as_mut_ptr().cast(), self.part, 8))?;
                let mut fl = [0i32];
                checked("cuMemcpyDtoH failure", (cuda.download)(fl.as_mut_ptr().cast(), self.fail, 4))?;
                let nw = if self.trace_state { self.state_words } else { self.words };
                self.support.sqdiff_into(cuda, 0, self.sh_base, 0, nw as usize * 4, 4)?;
                let (s, _) = self.support.collect(cuda, 1, 0)?;
                if s[0] > self.trace_growth * self.trace_last.max(1e-30) && self.trace_hits < self.trace_cap.max(60) {
                    self.trace_hits += 1;
                    eprintln!("{}", json!({"event":"window_trace_growth_reverse","step":step,"segment_command":j,"command":label,
                        "before":self.trace_last,"after":s[0],"top":self.norms(cuda, 3)?}));
                }
                self.trace_last = s[0];
                if z[0] != 0 || fl[0] != 0 {
                    let top = self.norms(cuda, 4)?;
                    eprintln!("{}", json!({"event":"window_trace_nonfinite","step":step,"segment_command":j,"segment_commands":ntape,
                        "command":label,"nonfinite_words":z[0],"row_flag":fl[0],"top":top}));
                    return Err(format!("step {step}: non-finite cotangent after walking back {label}"));
                }
            }
        }
        self.sync_owed = false;
        dev::sync(cuda)
    }
    /// pk120: a tangent bind takes its derivative segments from the tape arena, which was sized once from --tape-gib
    /// (2 GiB in the round-2 argv, chosen on x1.2562). On x1.40962 (40,962 cells) the MYNN-EDMF bind alone needs more
    /// (step 1 rrfs_mynnedmf_run_l16: 2,297,824,256 of 2,147,483,648 bytes) and the window refused before its first
    /// tangent step. The tangent resets the tape before every bind and keeps nothing in it between commands, so the
    /// arena is re-made at max(need + 1/8, 2x) (MiB-rounded, at most --tape-max-gib) and the bind runs again until it
    /// fits, before the forward command runs (the exhausted take names only its own end, not the bind's total).
    /// Bounded like size_tape (--tape-max-gib, card free less 6 GiB); returns false when it cannot grow, and the
    /// original error stands. Nothing compared changes: the derivative words do not depend on the arena's size.
    unsafe fn grow_tape_for(&mut self, ex: &Execution, e: &str, step: usize, f: &str) -> Result<bool, String> {
        if !exhausted(e) { return Ok(false); }
        let need = e.split("need ").nth(1).and_then(|t| t.split(' ').next()).and_then(|t| t.parse::<usize>().ok()).unwrap_or(0);
        if need <= self.tape.bytes { return Ok(false); }
        // an exhausted take names only where that take ends, not the bind's total, so grow by at least 2x
        let want = (need + need / 8).max(2 * self.tape.bytes).min(self.tape_max.max(need)).div_ceil(1 << 20) << 20;
        let (mut free, mut total) = (0usize, 0usize);
        checked("cuMemGetInfo", (ex.cuda.mem_info)(&mut free, &mut total))?;
        let before = self.tape.bytes;
        if want > self.tape_max || want - before + (6usize << 30) > free { return Ok(false); }
        dev::sync(&ex.cuda)?;
        checked("cuMemFree tape", (ex.cuda.free)(self.tape.base))?;
        self.tape = dev::Arena::new(&ex.cuda, want)?;
        let ev = json!({"step":step,"function":f,"need_bytes":need,"tape_bytes_before":before,"tape_bytes":want,"card_free_bytes":free});
        eprintln!("{}", json!({"event":"window_tl_tape_grown","info":ev}));
        self.tl_tape_grown.push(ev);
        Ok(true)
    }
    /// tape-cap: grow the tape to the largest step's measured estimate (same rounding as reverse_step), so every step
    /// reverses in one pass. The bound is the card: what it has free now (after the first tangent pass, so the shadows,
    /// the TEMPO arena and the P3 tangent arena are already held) less --tape-reserve-gib (default 16 GiB, the reserve
    /// the P3 tangent arena and the TEMPO arena keep: room for the reverse's own later allocations, the MYNN reverse
    /// tiles first, and for other processes on a shared card), and --tape-max-gib only when it is given. THE BREAKAGE
    /// THIS FIXES: the old fixed 16 GiB default cap kept pkg60lam's 20.3 GB step segmented (10-16 segments, 3.5x the
    /// commands replayed) on a card with 55-60 GB free. When the card is too crowded, or the allocation itself fails,
    /// the tape stays at its old size (re-made after a failed allocation) and the segmented path runs: the same reverse
    /// words, segmentation never changes the result. Called once, after the first tangent pass, before any reverse.
    unsafe fn size_tape(&mut self, ex: &Execution, flats: &[Vec<Flat>], max_gib: Option<&str>, reserve_gib: Option<&str>) -> Result<(), String> {
        let need = flats.iter().map(|f| f.iter().map(|x| x.est.div_ceil(256) * 256 + 256).sum::<usize>()).max().unwrap_or(0);
        let want = need + (64 << 20);
        let gib = |v: Option<&str>| v.and_then(|v| v.parse::<f64>().ok()).map(|g| (g * (1u64 << 30) as f64) as usize);
        let max = gib(max_gib);
        // x20 checkpoint-not-tape: with the packed tape (AD_TAPE_PACK, the default) the fixed 16 GiB default reserve is
        // what kept pk120 segmented. THE BREAKAGE: tdpp-G3c (x1.40962, whole 96 GB card, card free 26.7 GB after the
        // tangent pass) got a 10.7 GB tape for an 18.6 GB packed step: 4-7 segments, 2.0-3.0x replay, host RAM anchors
        // (snap 5-19 s per step), AD 78-139 s; the same argv with 12 GiB (tdpp-H3) reversed every step in one pass, AD
        // 72-79 s, the same words (e01260eb). The reverse scratch growth and the window cache already come out of the tape
        // budget (59b62afd); the reserve covers the native rows' own reverse scratch (an 8 GiB floor when packed, see below). The tape
        // itself is sized to the packed need (AD_TAPE_PACK_GUESS x the unpacked estimate, default 0.2; pk120 measured
        // 18.58 GB of a 105.0 GB estimate = 0.18 in tdpp-H3), never to all free memory and never to the whole unpacked
        // estimate (the old path took need + 64 MiB whenever it fit: wasted card memory on small meshes), so
        // a shared card keeps what this window does not need. A step that still overflows goes segmented: the same
        // words, segmentation never changes the result. --tape-reserve-gib still overrides.
        // x20ck-C (a test box, card 3): with a 4 GiB floor and the window cache's 1/5 share taken first, the tape (21.9 GB)
        // and the cache (5.5 GB) left 4.3 GB, and step 3's ugwp_oro_run adjoint refused its 3.2 GB scheme scratch (CUDA
        // status 2): native reverse scratch is allocated per scheme, outside the tape budget. So the floor is 8 GiB and a
        // packed tape that fits takes no window cache (see below).
        // x20ck-C2 (05:55Z): an 8 GiB floor with no cache still left only 10.6 GB, and step 3's rrfs_mynnedmf_run_l16 adjoint
        // refused 1.12 GB (the MYNN tiles size themselves from the card's free memory, UGWP takes 3.2 GB, retired blocks are
        // kept): the native rows' reverse scratch needs ~12.9 GB at pk120 (x20ck-A ran with 12.88 GB left). 12 GiB is the
        // floor that has run there (tdpp-H3, x20ck-A).
        let default_reserve = if self.pack { 12usize << 30 } else { 16usize << 30 };
        let reserve = gib(reserve_gib).unwrap_or(default_reserve);
        let pack_guess = std::env::var("AD_TAPE_PACK_GUESS").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.2).clamp(0.01, 1.0);
        // tape-dedupe-and-pack: the reverse scratch reaches its packed size before the card's free memory is read.
        self.fit_scratch(&ex.cuda)?;
        let (mut free, mut total) = (0usize, 0usize);
        checked("cuMemGetInfo", (ex.cuda.mem_info)(&mut free, &mut total))?;
        let before = self.tape.bytes;
        let fits_cap = max.is_none_or(|m| want <= m);
        let fits_card = want.saturating_sub(before) + reserve <= free;
        // packed-tape: shared pre-launch copies take no tape, so the unpacked estimate overstates the need; grow to what
        // the card allows (free + the old tape, less the reserve) and let a step that still overflows go segmented.
        // tape-dedupe-and-pack: the window cache takes its share of the same budget (1/5), never the reserve.
        let budget = (free + before).saturating_sub(reserve);
        // x20: packed, the tape wants the packed guess of the estimate (plus 64 MiB), bounded by the budget.
        let packed_want = ((need as f64 * pack_guess) as usize + (64 << 20)).div_ceil(1 << 20) << 20;
        // x20: the window cache only saves copies across segments; when the packed tape fits the budget the step reverses in
        // one pass and the cache's share goes to the native rows' reverse scratch instead (x20ck-C refusal above).
        // x20ck-C2: packed, the cache is off unless AD_TAPE_WINCACHE_MIB asks for it (its 1/5 share would cut a tape that
        // just fits, as at pk120: 18.8 GB budget for an 18.6 GB step).
        if self.pack && std::env::var("AD_TAPE_WINCACHE_MIB").is_err() { self.win_cap = 0; }
        let one_pass_fits = self.pack && packed_want <= budget;
        let win_share = if self.pack && !one_pass_fits && self.win_cap > 0 && self.win.is_none() { self.win_cap.min(budget / 5) & !((1usize << 20) - 1) } else { 0 };
        let want = if self.pack { budget.saturating_sub(win_share).min(packed_want.max(before)).max(before) } else { want };
        let fits_card = if self.pack { want > before } else { fits_card };
        let mut alloc_error: Option<String> = None;
        let grown = if want > before && fits_cap && fits_card {
            dev::sync(&ex.cuda)?;
            checked("cuMemFree tape", (ex.cuda.free)(self.tape.base))?;
            match dev::Arena::new(&ex.cuda, want) {
                Ok(t) => { self.tape = t; true },
                Err(e) => { alloc_error = Some(e.to_string()); self.tape = dev::Arena::new(&ex.cuda, before)?; false },
            }
        } else { false };
        let limited_by = if want <= before { "tape already large enough" } else if !fits_cap { "--tape-max-gib" }
            else if !fits_card { "card free less reserve" } else if alloc_error.is_some() { "allocation failed" } else { "none" };
        if one_pass_fits && self.win.is_none() { self.win_cap = 0; }
        if self.pack && self.win.is_none() && self.win_cap > 0 {
            // What is left above the reserve after the tape, at most the share set aside: the window cache's cap.
            let left = (free + before).saturating_sub(self.tape.bytes).saturating_sub(reserve);
            self.win_cap = self.win_cap.min(left).min(if win_share > 0 { win_share } else { left });
            if self.win_cap < (16 << 20) { self.win_cap = 0; }
        }
        self.tape_grown = Some(json!({"largest_step_estimate_bytes":need,"tape_bytes_before":before,"tape_bytes":self.tape.bytes,"window_cache_cap":self.win_cap,
                                      "grown":grown,"max_bytes":max,"reserve_bytes":reserve,"pack_guess":pack_guess,"packed_want_bytes":packed_want,"limited_by":limited_by,"alloc_error":alloc_error,
                                      "card_free_bytes":free,"card_total_bytes":total,"one_pass_possible":need <= self.tape.bytes}));
        eprintln!("{}", json!({"event":"window_tape_sized","info":self.tape_grown}));
        Ok(())
    }
    fn guess(&self, flat: &[Flat], end: usize) -> usize {
        // tape-dedupe-and-pack: the estimate overstates a packed segment by 1 / seg_ratio (the largest measured ratio).
        let r = if self.pack && self.seg_ratio > 0.0 { self.seg_ratio.clamp(0.05, 1.0) } else { 1.0 };
        let cap = (self.tape.bytes as f64 * 0.7 / r) as usize;
        let (mut lo, mut sum) = (end - 1, flat[end - 1].est);
        while lo > 0 && sum + flat[lo - 1].est <= cap { lo -= 1; sum += flat[lo].est; }
        lo
    }

    /// anchors-ahead (card-need 2a, a test box: the first segmented reverse step made every RAM anchor's page-locked buffer
    /// on the main thread, card idle: pk120 24 x 10.5 GB, 14-57 s per window alone, up to 181 s on a crowded host):
    /// start a helper thread that makes `count` buffers of `bytes` during the tangent pass. The anchors' bytes, offsets
    /// and restores are the same; only who made the buffer and when changes.
    unsafe fn ahead_start(&mut self, ex: &Execution, count: usize, bytes: usize) -> Result<(), String> {
        if count == 0 || bytes == 0 || self.anchor_ahead.is_some() { return Ok(()); }
        let pool: AheadPool = std::sync::Arc::new((std::sync::Mutex::new((Vec::new(), false)), std::sync::Condvar::new()));
        let (fns, set, c, p) = (crate::pinned::PinFns::of(&ex.cuda), ex.cuda.ctx_set_current, ex.context as usize, pool.clone());
        std::thread::Builder::new().name("anchors-ahead".into()).spawn(move || {
            let ok = set(c as *mut std::ffi::c_void) == 0;
            for _ in 0..count {
                if !ok { break; }
                let Ok(m) = crate::pinned::PinMem::new(fns, bytes) else { break };
                let (l, cv) = &*p;
                if let Ok(mut g) = l.lock() { g.0.push(crate::pinned::PinnedBuf::from_mem(m, bytes)); }
                cv.notify_all();
            }
            let (l, cv) = &*p;
            if let Ok(mut g) = l.lock() { g.1 = true; }
            cv.notify_all();
        }).map_err(|e| e.to_string())?;
        self.anchor_ahead = Some(pool);
        Ok(())
    }
    /// anchors-ahead: a helper-made buffer of at least `total` bytes, waiting for the helper while it is still making
    /// them (each wait is shorter than making one here); None once it is done with none that fits (then the caller
    /// makes one, as before, and buffers too small to fit are freed).
    fn take_ahead(&mut self, total: usize) -> Option<crate::pinned::PinnedBuf> {
        let pool = self.anchor_ahead.clone()?;
        let (l, cv) = &*pool;
        let t = Instant::now();
        let mut g = l.lock().ok()?;
        let got = loop {
            if let Some(i) = g.0.iter().position(|b| b.len() >= total) { break Some(g.0.swap_remove(i)); }
            if g.1 { g.0.clear(); break None; }
            g = cv.wait(g).ok()?;
        };
        let done_empty = g.1 && g.0.is_empty();
        drop(g);
        self.anchor_ahead_wait_s += t.elapsed().as_secs_f64();
        if got.is_some() { self.anchor_ahead_taken += 1; }
        if done_empty { self.anchor_ahead = None; }
        got
    }
    /// anchors-ahead: whether this window's reverse will want RAM anchors (a segmented step), judged from the
    /// tangent pass's tape estimates the same way reverse_step and size_tape judge them, and if so start making them.
    /// A wrong guess changes no word: unused buffers only hold RAM until the window ends, and a missing or too small
    /// one is made in the reverse step as before. AD_ANCHORS_AHEAD=0 never starts it, =1 always does.
    unsafe fn ahead_maybe(&mut self, ex: &Execution, flats: &[Vec<Flat>], anchors: usize, reserve_gib: Option<&str>, tl_done: bool, at_step: usize,
                          plan_bytes: usize) -> Result<(), String> {
        let mode = std::env::var("AD_ANCHORS_AHEAD").unwrap_or_default();
        let ram_anchors = anchors.saturating_sub(self.dev_anchors.len());
        let started = self.anchor_ahead_info.iter().any(|v| v["started"] == json!(true));
        if mode == "0" || ram_anchors == 0 || self.anchor_ahead.is_some() || started { return Ok(()); }
        let need = flats.iter().map(|f| f.iter().map(|x| x.est.div_ceil(256) * 256 + 256).sum::<usize>()).max().unwrap_or(0);
        let (mut free, mut total) = (0usize, 0usize);
        checked("cuMemGetInfo", (ex.cuda.mem_info)(&mut free, &mut total))?;
        let reserve = reserve_gib.and_then(|v| v.parse::<f64>().ok()).map(|g| (g * (1u64 << 30) as f64) as usize).unwrap_or(16usize << 30);
        // what the one-pass tape can become: the tape now, or (before size_tape) the card's free room less the reserve
        let room = self.tape.bytes.max((free + self.tape.bytes).saturating_sub(reserve));
        // When the pass ends the tape is sized and every step's estimate is known: the step is one-pass unless its estimate
        // (packed: the written tape measured 0.44-0.64 of it, round 4/5; 0.4 here) exceeds the tape. After step 1 only
        // that step is known and the card fills further (a test box, leg aa-c120A: pk120's step-1 estimate 62.8 GB, the largest
        // 108 GB; 30.8 GB of room after step 1, an 18.3 GB tape at the end), so one pass is assumed only with 2x headroom
        // on the unpacked estimate. A wrong 'segmented' costs RAM only; a wrong 'one pass' leaves the card idle as before.
        let segmented = !self.one_pass || if tl_done { (if self.pack { (need as f64 * 0.4) as usize } else { need }) > self.tape.bytes }
                                          else { need.saturating_mul(2) > room };
        // the anchors of the first reverse step (the window's last) hold the whole live state there: the plan's bound
        // (a test box, leg aa-c18S8: pkg18m2 holds 2.82 GB after step 1 and 3.38 GB from step 3, so step 1's peak was too small)
        let bytes = ex.live_peak.max(ex.live_bytes).max(plan_bytes).div_ceil(2 << 20) * (2 << 20);
        let avail = mem_available();
        let to_come = if tl_done { 0 } else { self.bs_ram_budget.saturating_sub(self.bs_ram_now) };
        let allow = (avail.saturating_sub(to_come) as f64 * 0.75) as usize;
        let start = segmented || mode == "1";
        let count = if start { ram_anchors.min(allow / bytes.max(1)) } else { 0 };
        self.anchor_ahead_info.push(json!({"started":count > 0,"buffers":count,"bytes_each":bytes,"ram_anchors":ram_anchors,"predicted_segmented":segmented,
            "largest_step_estimate_bytes":need,"tape_room_bytes":room,"mem_available_bytes":avail,"ram_allowed_bytes":allow,"after_tangent_step":at_step,"plan_live_bytes_bound":plan_bytes,"live_peak_now":ex.live_peak,
            "tangent_pass_done":tl_done,"forced":mode == "1","tape_bytes_now":self.tape.bytes}));
        eprintln!("{}", json!({"event":"window_anchors_ahead","info":self.anchor_ahead_info.last()}));
        self.ahead_start(ex, count, bytes)
    }

    /// Whole device state and host side in RAM (a mid-step anchor; the frame may be mid-step).
    /// S04: into a reused page-locked buffer from `anchor_pool` (returned there when the reverse step ends).
    unsafe fn snap_ram(&mut self, ex: &Execution, frame: &Option<ReplayFrame>) -> Result<(Boundary, crate::pinned::PinnedBuf), String> {
        let _c = capi::xfer::Cat::checkpoint();
        dev::sync(&ex.cuda)?;
        let mut ids: Vec<u64> = ex.allocations.keys().copied().collect();
        ids.sort_unstable();
        let mut index = Vec::with_capacity(ids.len());
        let mut total = 0usize;
        for id in &ids { let b = ex.allocations[id].bytes; index.push((*id, total, b)); total += b; }
        let mut data = match self.anchor_pool.iter().position(|b| b.len() >= total) {
            Some(i) => self.anchor_pool.swap_remove(i),
            None => match self.take_ahead(total) { Some(b) => b, None => crate::pinned::PinnedBuf::new(&ex.cuda, total)? } };
        for (id, off, b) in &index {
            checked("cuMemcpyDtoH anchor", (ex.cuda.download)(data[*off..*off + *b].as_mut_ptr().cast(), ex.allocations[id].address, *b))?;
        }
        Ok((Boundary { path: PathBuf::new(), index, host: host_snap(ex, frame), step: usize::MAX, delta: false, file_bytes: 0, ram: None, dev: 0 }, data))
    }

    /// runner-lib: snapshot the whole live state into device buffer `buf` (bytes; D2D, no host copy).
    unsafe fn snap_dev(&self, ex: &Execution, frame: &Option<ReplayFrame>, buf: u64, cap: usize) -> Result<Boundary, String> {
        let _c = capi::xfer::Cat::checkpoint();
        dev::sync(&ex.cuda)?;
        let mut ids: Vec<u64> = ex.allocations.keys().copied().collect();
        ids.sort_unstable();
        let mut index = Vec::with_capacity(ids.len());
        let mut total = 0usize;
        for id in &ids { let b = ex.allocations[id].bytes; index.push((*id, total, b)); total += b; }
        if total > cap { return Err(format!("device anchor buffer holds {cap} bytes, the state has {total}")); }
        for (id, off, b) in &index { dev::copy(&ex.cuda, buf + *off as u64, ex.allocations[id].address, *b)?; }
        dev::sync(&ex.cuda)?;
        Ok(Boundary { path: PathBuf::new(), index, host: host_snap(ex, frame), step: usize::MAX, delta: false, file_bytes: 0, ram: None, dev: buf })
    }
    unsafe fn restore_dev(&mut self, ex: &mut Execution, frame: &mut Option<ReplayFrame>, b: &Boundary) -> Result<(), String> {
        let _c = capi::xfer::Cat::checkpoint();
        let t = dev::prof::start();
        drain_pinned(ex)?;
        for (id, off, n) in &b.index {
            let a = ex.allocations.get(id).ok_or("anchor allocation not live")?;
            if a.bytes != *n { return Err(format!("allocation {id} changed size since the anchor")); }
            dev::copy(&ex.cuda, a.address, b.dev + *off as u64, *n)?;
        }
        dev::sync(&ex.cuda)?;
        let r = restore_host(ex, frame, &b.host);
        dev::prof::add(6, t);
        r
    }

    /// Reverse of one step from the cotangent at its end. Pass 1 replays the step from its start boundary (`prev`),
    /// keeping `anchors` whole-state snapshots in RAM, and compares the end state word by word with the stored end
    /// boundary (`cur`): restore + redo. Then segments, last first, each replayed from the nearest anchor at or before
    /// it, taped and walked back; a segment whose tape overflows is split where it overflowed.
    #[allow(clippy::too_many_arguments)]
    unsafe fn reverse_step(&mut self, ex: &mut Execution, base: &Path, frame: &mut Option<ReplayFrame>, step: usize, flat: &[Flat],
                           prev: &Boundary, prev_data: &[u8], cur: &Boundary, cur_data: &[u8], anchors_max: usize) -> Result<Value, String> {
        let t0 = Instant::now();
        let ms0 = self.entry_ms;
        let n = flat.len();
        let (mut end, mut hint, mut segments, mut replayed, mut overflows) = (n, None::<usize>, 0usize, 0usize, 0usize);
        let mut tape_peak = 0usize;
        // packed-tape: the step's tape bytes summed over its segments (what one pass would need, less the copies two
        // segments' caches each hold): the per-step tape receipt the segmented path otherwise lacks.
        let mut tape_sum = 0usize;
        // adjoint-speed: when the step's measured tape estimate fits the tape, one replay from the step's
        // start boundary tapes the whole step, the end state is compared word by word with the stored boundary there
        // (the same restore + redo check pass 1 makes, on the same replay), and the tape is walked back. Pass 1, its
        // anchors and the per-segment restores and replays are then not needed. The walk-back order and every taped
        // word are the same as the segmented path's, so the reverse is bitwise the same (segmentation never changes the
        // result). A step whose tape does not fit after all (Overflow) falls back to the segmented path below.
        let total_est: usize = flat.iter().map(|f| f.est.div_ceil(256) * 256 + 256).sum();
        self.fit_scratch(&ex.cuda)?;
        let _ = self.pack_json();
        self.rev_rows.clear();
        // packed-tape: the estimate counts every pre-launch copy; shared copies take none, so a packed step is tried in
        // one pass when the shared fraction measured so far says it fits (before the first packed step: whenever the
        // tape is at least a tenth of the estimate). A step that does not fit after all overflows into the segmented
        // path below: the same words, one replay more.
        // x20: no 5 % margin on the measured ratio (pk120: 18.55 GB steps under an 18.8 GB tape were sent segmented); a step
        // that overflows after all goes segmented below with the same words.
        let packed_fits = self.pack && if self.pack_ratio > 0.0 { (total_est as f64 * self.pack_ratio) as usize <= self.tape.bytes }
                                       else { total_est / 10 <= self.tape.bytes };
        if self.one_pass && n > 0 && (total_est <= self.tape.bytes || packed_fits) {
            let t_r = Instant::now();
            self.restore(ex, frame, prev, prev_data)?;
            self.rv_restore_s = t_r.elapsed().as_secs_f64();
            let t_one = Instant::now();
            if let Some(f) = frame.as_mut() { f.begin(step)?; }
            match self.tape_range(ex, base, frame, step, flat, 0, n)? {
                Taping::Done(tape) => {
                    replayed += n;
                    let t_tape = t_one.elapsed().as_secs_f64();
                    // adjoint-speed-2: the redo check (end state word for word against the stored boundary) runs on worker
                    // threads beside the walk-back, off the critical path: the state is downloaded first (page-locked, before
                    // any adjoint command), the walk-back writes only derivative and tape memory, and a mismatch still
                    // refuses the step with the same message (after the walk instead of before it).
                    let t_dl = Instant::now();
                    let live = self.redo_download(ex, cur)?;
                    let ranges: Vec<Vec<(usize, usize)>> = cur.index.iter()
                        .map(|(id, _, _)| ex.residue.get(&(step, *id)).map(|x| x.0.clone()).unwrap_or_default()).collect();
                    let dl_s = t_dl.elapsed().as_secs_f64();
                    let t_pass1 = t0.elapsed().as_secs_f64();
                    tape_peak = tape_peak.max(self.tape.used);
                    tape_sum += self.tape.used;
                    let cuda = &ex.cuda as *const Cuda;
                    let t_w = Instant::now();
                    let (cmp, walked, walk_s) = std::thread::scope(|sc| {
                        let (index, live_ref, ranges_ref) = (&cur.index, &live[..], &ranges);
                        let h = sc.spawn(move || compare_boundary(index, live_ref, cur_data, ranges_ref, step));
                        let w = self.walk_back(&*cuda, ex, step, tape);
                        let walk_s = t_w.elapsed().as_secs_f64();
                        (h.join(), w, walk_s)
                    });
                    let wait_s = t_w.elapsed().as_secs_f64() - walk_s;
                    self.cmp_pin = Some(live);
                    let (d, r) = cmp.map_err(|_| "redo compare thread panicked".to_string())?;
                    if d != 0 { return Err(format!("step {step}: restore + redo differs from the stored boundary in {d} words")); }
                    walked?;
                    self.rv_tape_s = t_tape; self.rv_redo_dl_s = dl_s; self.rv_walk_s = walk_s; self.rv_redo_wait_s = wait_s;
                    let redo = json!({"differing_words": d, "residue_words": r, "anchors": 0, "one_pass": true, "beside_walk": true});
                    self.step_flush(&ex.cuda, step, "ad")?;
                    drain_pinned(ex)?;
                    let mut fl = [0i32];
                    checked("cuMemcpyDtoH failure", (ex.cuda.download)(fl.as_mut_ptr().cast(), self.fail, 4))?;
                    if fl[0] != 0 { return Err(format!("step {step}: P8 row transpose saw a non-finite cotangent")); }
                    let host_prof: Vec<Value> = dev::prof::take().into_iter().filter(|x| x.2 > 0).map(|(k, s, c)| json!([k, s, c])).collect();
                    if self.pack && total_est > 0 { self.pack_ratio = self.pack_ratio.max(tape_peak as f64 / total_est as f64); }
                    let pack = self.pack_json();
                    let rrows = self.take_rev_rows(0);
                    return Ok(json!({"step":step,"commands":n,"segments":1,"overflow_splits":0,"commands_replayed":replayed,"host_prof":host_prof,"pack":pack,"tape_rows_packed":rrows,
                              "stages_s":{"restore":self.rv_restore_s,"replay_and_tape":self.rv_tape_s,"redo_download":self.rv_redo_dl_s,
                                          "walk_back":self.rv_walk_s,"redo_compare_wait_after_walk":self.rv_redo_wait_s},
                              "tape_peak_bytes":tape_peak,"tape_sum_bytes":tape_sum,"adjoint_entry_s":(self.entry_ms - ms0) / 1e3,"pass1_s":0.0,"one_pass_tape_s":t_pass1,
                              "tape_estimate_bytes":total_est,"wall_s":t0.elapsed().as_secs_f64(),"redo":redo}));
                },
                Taping::Overflow(k) => { replayed += k; overflows += 1; self.one_pass_fallbacks += 1; },
            }
        }
        // Pass 1 (anchors + redo check).
        self.restore(ex, frame, prev, prev_data)?;
        if let Some(f) = frame.as_mut() { f.begin(step)?; }
        let stride = n.div_ceil(anchors_max + 1).max(1);
        let mut anchors: Vec<(usize, Boundary, Option<crate::pinned::PinnedBuf>)> = Vec::new(); // runner-lib: None = a device anchor
        let (rs0, mut snap_s) = (self.restore_s, 0f64);
        for (i, fl) in flat.iter().enumerate() {
            if i > 0 && i % stride == 0 && anchors.len() < anchors_max {
                let ts = Instant::now();
                let (b, d) = match self.dev_anchors.get(anchors.len()).copied() {
                    Some((buf, cap)) => (self.snap_dev(ex, frame, buf, cap)?, None),
                    None => { let (b, d) = self.snap_ram(ex, frame)?; (b, Some(d)) }
                };
                anchors.push((i, b, d)); snap_s += ts.elapsed().as_secs_f64();
            }
            ex.rk = fl.rk.clone();
            exec(ex, &*fl.op, step, base, frame)?;
        }
        replayed += n;
        // host-leftovers-2: the end state is DMA'd into the reused page-locked buffer and compared on all cores
        // (compare_boundary, the one-pass path's compare: the same counts and the same window_redo_differs lines) on a
        // worker beside the segments, instead of a pageable download per allocation and a compare on the main thread
        // before them (the segments write only derivative, tape and replayed state memory; the compare reads the
        // downloaded copy and the stored boundary). A mismatch still refuses the step with the same message, after the
        // segments instead of before. AD_REDO_COMPARE=host keeps the old compare, before the segments.
        let host_cmp = std::env::var("AD_REDO_COMPARE").as_deref() == Ok("host");
        let (dh, rh) = if host_cmp { self.differing(ex, cur, cur_data, step)? } else { (0, 0) };
        if dh != 0 { return Err(format!("step {step}: restore + redo differs from the stored boundary in {dh} words")); }
        let live = if host_cmp { None } else { Some(self.redo_download(ex, cur)?) };
        let ranges: Vec<Vec<(usize, usize)>> = if host_cmp { Vec::new() } else { cur.index.iter()
            .map(|(id, _, _)| ex.residue.get(&(step, *id)).map(|x| x.0.clone()).unwrap_or_default()).collect() };
        let t_pass1 = t0.elapsed().as_secs_f64();
        let (cmp, segs) = std::thread::scope(|sc| {
            let h = live.as_ref().map(|l| { let (index, lr, rr) = (&cur.index, &l[..], &ranges); sc.spawn(move || compare_boundary(index, lr, cur_data, rr, step)) });
            let segs = (|| -> Result<(), String> {
        while end > 0 {
            let mut lo = hint.take().filter(|&h| h < end).unwrap_or_else(|| self.guess(flat, end));
            loop {
                match anchors.iter().rev().find(|(k, _, _)| *k <= lo) {
                    Some((k, b, data)) => {
                        if b.dev != 0 { self.restore_dev(ex, frame, b)?; } else { self.restore(ex, frame, b, data.as_ref().ok_or("host anchor without its bytes")?)?; }
                        for fl in flat.iter().take(lo).skip(*k) { ex.rk = fl.rk.clone(); exec(ex, &*fl.op, step, base, frame)?; }
                        replayed += lo - k;
                    },
                    None => {
                        self.restore(ex, frame, prev, prev_data)?;
                        if let Some(f) = frame.as_mut() { f.begin(step)?; }
                        for fl in flat.iter().take(lo) { ex.rk = fl.rk.clone(); exec(ex, &*fl.op, step, base, frame)?; }
                        replayed += lo;
                    },
                }
                match self.tape_range(ex, base, frame, step, flat, lo, end)? {
                    Taping::Overflow(k) => {
                        overflows += 1;
                        replayed += k - lo;
                        if k == lo { return Err(format!("step {step}: command {k} alone does not fit the tape ({} bytes); raise --tape-gib", self.tape.bytes)); }
                        hint = Some(lo); lo = k;
                    },
                    Taping::Done(tape) => {
                        replayed += end - lo;
                        tape_peak = tape_peak.max(self.tape.used);
                        tape_sum += self.tape.used;
                        if self.pack {
                            let est: usize = flat[lo..end].iter().map(|f| f.est).sum();
                            if est > 0 { self.seg_ratio = self.seg_ratio.max(self.tape.used as f64 / est as f64); }
                        }
                        let cuda = &ex.cuda as *const Cuda;
                        self.walk_back(&*cuda, ex, step, tape)?;
                        segments += 1;
                        end = lo;
                        break;
                    },
                }
            }
            drain_pinned(ex)?;
        }
        Ok(()) })();
            (h.map(|h| h.join()), segs)
        });
        let (d, r) = match cmp { Some(c) => c.map_err(|_| "redo compare thread panicked".to_string())?, None => (dh, rh) };
        if let Some(l) = live { self.cmp_pin = Some(l); }
        if d != 0 { return Err(format!("step {step}: restore + redo differs from the stored boundary in {d} words")); }
        segs?;
        let redo = json!({"differing_words": d, "residue_words": r, "anchors": anchors.len(), "beside_segments": !host_cmp});
        for (_, _, d) in anchors { if let Some(d) = d { self.anchor_pool.push(d); } }
        if self.pack && total_est > 0 { self.pack_ratio = self.pack_ratio.max(tape_sum as f64 / total_est as f64); }
        self.step_flush(&ex.cuda, step, "ad")?;
        let mut fl = [0i32];
        checked("cuMemcpyDtoH failure", (ex.cuda.download)(fl.as_mut_ptr().cast(), self.fail, 4))?;
        if fl[0] != 0 { return Err(format!("step {step}: P8 row transpose saw a non-finite cotangent")); }
        let host_prof: Vec<Value> = dev::prof::take().into_iter().filter(|x| x.2 > 0).map(|(k, s, c)| json!([k, s, c])).collect();
        let pack = self.pack_json();
                    let rrows = self.take_rev_rows(0);
        Ok(json!({"step":step,"commands":n,"segments":segments,"overflow_splits":overflows,"commands_replayed":replayed,"host_prof":host_prof,"pack":pack,"tape_rows_packed":rrows,
                  "tape_estimate_bytes":total_est,"tape_peak_bytes":tape_peak,"tape_sum_bytes":tape_sum,"adjoint_entry_s":(self.entry_ms - ms0) / 1e3,"pass1_s":t_pass1,
                  "wall_s":t0.elapsed().as_secs_f64(),"redo":redo,"anchor_snap_s":snap_s,"restore_s":self.restore_s - rs0,
                  "seg_ratio":self.seg_ratio,"pack_ratio":self.pack_ratio}))
    }

    /// Per-word weights of a relative direction over the arena words [f, f + n): |primal| at the boundary the
    /// direction lives on (finite, at most WEIGHT_MAX; zero for words no allocation held there, and for integer and
    /// address words, which read as denormals or huge values). Built on all cores.
    /// host-leftovers-2 (host-audit row 19: 37 s of host time per dot window, card idle): each allocation's unstable byte
    /// ranges (translated address words, declared residue, relocation slots) are gathered once, sorted and merged, and
    /// read by a moving cursor instead of three map/list lookups per word; the --direction field/fieldxy weight
    /// (rms of the allocation's valid words) is summed once per allocation, in the same word order, instead of once per
    /// worker whose word range touches the allocation (16x for the 979 MB block). Every weight is the same f32.
    fn scale_chunk(&self, sc: Option<(&Boundary, &[u8])>, f: i64, n: i64) -> Option<Vec<f32>> {
        let (b, data) = sc?;
        let mut out = vec![0f32; n as usize];
        let threads = std::thread::available_parallelism().map(|t| t.get()).unwrap_or(4).min(64);
        let per = (n as usize).div_ceil(threads).max(1);
        let (base_el, index, fixed) = (&self.base_el, &b.index, &self.fixed);
        let (mask, ab) = (self.taylor_mask, &self.address_bearing);
        // --direction field: the window-end boundary's weights are 1 / rms of the allocation's valid words (the dual of
        // the relative x); every other boundary keeps |primal|. Same y in <y, M x> and in the reverse seed.
        let inv = self.field_y && self.y_path.as_deref() == Some(b.path.as_path());
        // --direction fieldxy: every other boundary (the x side) gets rms_f instead of the word's own |primal|.
        let fwd = self.field_x && !inv;
        let xs = if inv { self.y_scale } else { self.x_scale };
        let valid = |v: f32| v.is_finite() && v <= WEIGHT_MAX && v > 1e-30;
        let stable = self.stable_dir;
        let (relocs, residue, history, bstep) = (&self.relocs, &self.residue, b.host.pointer_history.get(), b.step);
        let (lo_all, hi_all) = (f, f + n);
        // the allocations whose words fall in [f, f + n), in index order: (id, off, bytes, e0)
        let als: Vec<(u64, usize, usize, i64)> = index.iter().filter(|(id, _, _)| !(fixed.contains(id) || (mask && ab.contains(id)) || (stable && ab.contains(id))))
            .filter_map(|(id, off, bytes)| { let e0 = base_el[id]; let e1 = e0 + (*bytes / 4) as i64;
                (e0.max(lo_all) < e1.min(hi_all)).then_some((*id, *off, *bytes, e0)) }).collect();
        // per allocation: sorted, merged unstable byte ranges [a, z) and the field weight r
        let mut prep: Vec<std::sync::Arc<(Vec<(usize, usize)>, f32)>> = Vec::with_capacity(als.len());
        let key = |id: u64| (b.path.clone(), b.step, id, stable, inv, fwd);
        let miss: Vec<usize> = { let g = self.wcache.lock().unwrap();
            (0..als.len()).filter(|&i| !g.contains_key(&key(als[i].0))).collect() };
        {
            let rd = |k: usize| f32::from_le_bytes(data[k..k + 4].try_into().unwrap()).abs();
            let one = |&(id, off, bytes, _): &(u64, usize, usize, i64)| -> (Vec<(usize, usize)>, f32) {
                let mut rg: Vec<(usize, usize)> = Vec::new();
                if stable {
                    if let Some(h) = history.get(&id) { rg.extend(h.keys().map(|&w| (8 * w, 8 * w + 8))); }
                    if let Some(r) = residue.get(&(bstep, id)) { rg.extend(r.iter().map(|&(lo, n)| (lo, lo + n))); }
                    if let Some(r) = relocs.get(&id) { rg.extend(r.iter().map(|&(lo, n)| (lo, lo + n))); }
                    rg.retain(|x| x.0 < x.1);
                    rg.sort_unstable();
                    let mut m: Vec<(usize, usize)> = Vec::with_capacity(rg.len());
                    for x in rg { match m.last_mut() { Some(l) if x.0 <= l.1 => { l.1 = l.1.max(x.1); }, _ => m.push(x) } }
                    rg = m;
                }
                let r = if inv || fwd {
                    let (mut s2, mut c) = (0f64, 0u64);
                    let mut p = 0usize;
                    for k in (off..off + (bytes / 4) * 4).step_by(4) {
                        let o = k - off;
                        while p < rg.len() && rg[p].1 <= o { p += 1; }
                        let un = p < rg.len() && rg[p].0 <= o;
                        let v = rd(k); if valid(v) && !un { s2 += (v as f64) * (v as f64); c += 1; }
                    }
                    if c == 0 || s2 == 0.0 { 0.0 } else if inv { (1.0 / (s2 / c as f64).sqrt()) as f32 } else { (s2 / c as f64).sqrt() as f32 }
                } else { 0.0 };
                (rg, r)
            };
            let next = std::sync::atomic::AtomicUsize::new(0);
            let slots: Vec<std::sync::Mutex<Option<(Vec<(usize, usize)>, f32)>>> = (0..miss.len()).map(|_| std::sync::Mutex::new(None)).collect();
            if !miss.is_empty() { std::thread::scope(|s| {
                for _ in 0..threads.min(miss.len()).max(1) {
                    s.spawn(|| loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= miss.len() { break; }
                        let v = one(&als[miss[i]]);
                        *slots[i].lock().unwrap() = Some(v);
                    });
                }
            }); }
            let mut g = self.wcache.lock().unwrap();
            for (i, m) in slots.into_iter().enumerate() { g.insert(key(als[miss[i]].0), std::sync::Arc::new(m.into_inner().unwrap().unwrap())); }
            for a in &als { prep.push(g[&key(a.0)].clone()); }
        }
        let (als, prep) = (&als, &prep);
        std::thread::scope(|s| {
            for (c, part) in out.chunks_mut(per).enumerate() {
                let lo = f + (c * per) as i64; let hi = lo + part.len() as i64;
                s.spawn(move || {
                    for (&(_, off, bytes, e0), pr) in als.iter().zip(prep.iter()) {
                        let (rg, r) = (&pr.0, &pr.1);
                        let e1 = e0 + (bytes / 4) as i64;
                        let (a, z) = (e0.max(lo), e1.min(hi));
                        if a >= z { continue; }
                        let rd = |k: usize| f32::from_le_bytes(data[k..k + 4].try_into().unwrap()).abs();
                        let o0 = 4 * (a - e0) as usize;
                        let mut p = rg.partition_point(|x| x.1 <= o0);
                        for e in a..z {
                            let o = 4 * (e - e0) as usize;
                            let k = off + o;
                            while p < rg.len() && rg[p].1 <= o { p += 1; }
                            let un = p < rg.len() && rg[p].0 <= o;
                            let v = rd(k);
                            part[(e - lo) as usize] = if !valid(v) || un { 0.0 } else if inv || fwd { r * xs } else { v * xs };
                        }
                    }
                });
            }
        });
        Some(out)
    }
    /// Generate u(seed) (times the relative weights when given) into `dst` for arena words [f, f + n).
    unsafe fn gen(&self, cuda: &Cuda, dst: u64, f: i64, n: i64, seed: u64, sc: Option<(&Boundary, &[u8])>) -> Result<(), String> {
        let w = self.scale_chunk(sc, f, n);
        let mut sp = 0u64;
        if let Some(w) = &w {
            sp = self.tmp[2];
            dev::upload(cuda, sp, std::slice::from_raw_parts(w.as_ptr().cast::<u8>(), w.len() * 4))?;
        }
        let (mut t, mut first, mut nn, mut s) = (dst, f, n, seed);
        launch(cuda, self.f.fill, [65535, 1, 1], kargs!(t, first, nn, s, sp))?;
        dev::sync(cuda)
    }
    unsafe fn fill(&self, cuda: &Cuda, seed: u64, sc: Option<(&Boundary, &[u8])>) -> Result<(), String> {
        let mut f = 0i64;
        while f < self.words { let n = (self.words - f).min(self.tmp_words); self.gen(cuda, self.sh_base + 4 * f as u64, f, n, seed, sc)?; f += n; }
        Ok(())
    }
    unsafe fn zero(&self, cuda: &Cuda) -> Result<(), String> { dev::zero(cuda, self.sh_base, self.words as usize * 4)?; dev::sync(cuda) }
    /// Taylor check: primal += h * x over every live allocation (x regenerated chunk by chunk; words with x = 0 untouched).
    unsafe fn perturb(&self, ex: &Execution, h: f64, seed: u64, sc: Option<(&Boundary, &[u8])>) -> Result<(), String> {
        let cuda = &ex.cuda;
        let mut f = 0i64;
        while f < self.words {
            let n = (self.words - f).min(self.tmp_words);
            self.gen(cuda, self.tmp[0], f, n, seed, sc)?;
            for (id, al) in &ex.allocations {
                let Some(&e0) = self.base_el.get(id) else { continue };
                let e1 = e0 + (al.bytes / 4) as i64;
                let (lo, hi) = (e0.max(f), e1.min(f + n));
                if lo >= hi { continue; }
                let (mut y, mut x, mut hh, mut nn) = (al.address + 4 * (lo - e0) as u64, self.tmp[0] + 4 * (lo - f) as u64, h, hi - lo);
                launch(cuda, self.f.axpy, [grid1(nn)[0].min(65535), 1, 1], kargs!(y, x, hh, nn))?;
            }
            dev::sync(cuda)?;
            f += n;
        }
        Ok(())
    }
    /// --taylor-dir: primal[allocation, word] += h * value (in float64, rounded once to float32), allocation by allocation.
    unsafe fn perturb_terms(&self, ex: &Execution, h: f64, terms: &[(u64, usize, f32)]) -> Result<(), String> {
        let cuda = &ex.cuda;
        dev::sync(cuda)?;
        let mut by: BTreeMap<u64, Vec<(usize, f32)>> = BTreeMap::new();
        for &(al, wd, v) in terms { by.entry(al).or_default().push((wd, v)); }
        for (al, ws) in &by {
            let a = ex.allocations.get(al).ok_or_else(|| format!("taylor-dir allocation {al} not live at boundary 0"))?;
            let mut buf = dev::download(cuda, a.address, a.bytes)?;
            for &(wd, v) in ws {
                let k = 4 * wd;
                if k + 4 > buf.len() { return Err(format!("taylor-dir word {wd} outside allocation {al}")); }
                let p = f32::from_le_bytes(buf[k..k + 4].try_into().unwrap());
                buf[k..k + 4].copy_from_slice(&(((p as f64) + h * v as f64) as f32).to_le_bytes());
            }
            dev::upload(cuda, a.address, &buf)?;
        }
        dev::sync(cuda)
    }
    /// --taylor-dir: <x, shadow> = sum value * shadow[allocation, word] (the objective reverse's gradient at boundary 0).
    unsafe fn sparse_dot(&self, cuda: &Cuda, terms: &[(u64, usize, f32)]) -> Result<f64, String> {
        let mut cache: HashMap<u64, Vec<u8>> = HashMap::new();
        let mut g = 0f64;
        for &(al, wd, v) in terms {
            if !cache.contains_key(&al) {
                let (Some(&el), Some(sh)) = (self.base_el.get(&al), self.shadows.get(&al)) else { return Err(format!("taylor-dir allocation {al} has no shadow")) };
                cache.insert(al, dev::download(cuda, self.sh(el), sh.bytes)?);
            }
            let b = &cache[&al];
            g += v as f64 * f32::from_le_bytes(b[4 * wd..4 * wd + 4].try_into().unwrap()) as f64;
        }
        Ok(g)
    }
    /// Taylor check: J(x0 + h x) - J(x0) = <y, x_N(h) - x_N(0)>, the difference built word by word on all cores
    /// (declared TEMPO residue words and bitwise-equal words are 0), placed in the shadow arena and dotted with y.
    unsafe fn taylor_diff(&self, ex: &Execution, b: &Boundary, data: &[u8], step: usize, seed: u64, sc: Option<(&Boundary, &[u8])>)
                          -> Result<(f64, u64, u64), String> {
        let cuda = &ex.cuda;
        dev::sync(cuda)?;
        dev::zero(cuda, self.sh_base, self.words as usize * 4)?;
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16);
        let mut buf: Vec<u8> = Vec::new();
        let mut changed = 0u64;
        for (id, off, n) in &b.index {
            buf.resize(*n, 0);
            checked("cuMemcpyDtoH taylor", (cuda.download)(buf.as_mut_ptr().cast(), ex.allocations[id].address, *n))?;
            let ranges = ex.residue.get(&(step, *id)).map(|x| x.0.clone()).unwrap_or_default();
            let want = &data[*off..*off + *n];
            let mut out = vec![0u8; *n / 4 * 4];
            let chunk = (*n / 4).div_ceil(threads).max(1) * 4;
            let ch: u64 = std::thread::scope(|scope| {
                let hs: Vec<_> = out.chunks_mut(chunk).zip(buf.chunks(chunk).zip(want.chunks(chunk))).enumerate().map(|(k, (o, (x, y)))| {
                    let ranges = &ranges;
                    scope.spawn(move || {
                        let mut c = 0u64;
                        for (w, (q, p)) in x.chunks_exact(4).zip(y.chunks_exact(4)).enumerate() {
                            if q == p { continue; }
                            let at = k * chunk + 4 * w;
                            if ranges.iter().any(|&(ro, rl)| at >= ro && at < ro + rl) { continue; }
                            c += 1;
                            let (qv, pv) = (f32::from_le_bytes(q.try_into().unwrap()), f32::from_le_bytes(p.try_into().unwrap()));
                            o[4 * w..4 * w + 4].copy_from_slice(&(((qv as f64) - (pv as f64)) as f32).to_le_bytes());
                        }
                        c
                    })
                }).collect();
                hs.into_iter().map(|h| h.join().unwrap()).sum()
            });
            changed += ch;
            if !out.is_empty() { dev::upload(cuda, self.sh(self.base_el[id]), &out)?; }
        }
        let (p, _, bad) = self.dots(cuda, seed, sc, seed, sc)?;
        Ok((p, bad, changed))
    }
    /// (sum a*S, sum a*(S - b), non-finite words of S) with a, b regenerated from their seeds (and weights) chunk by chunk.
    unsafe fn dots(&self, cuda: &Cuda, seed_a: u64, sa: Option<(&Boundary, &[u8])>, seed_b: u64, sb: Option<(&Boundary, &[u8])>)
                   -> Result<(f64, f64, u64), String> {
        let (mut p, mut q, mut bad) = (0f64, 0f64, 0u64);
        let mut f = 0i64;
        let mut alloc: Option<BTreeMap<u64, (f64, f64)>> = std::env::var("AD_DOT_ALLOC").ok().filter(|v| v == "1").map(|_| BTreeMap::new());
        let ranges: Vec<(u64, (i64, i64))> = if alloc.is_some() { match sa { Some((b, _)) => b.index.iter()
            .map(|(id, _, bytes)| (*id, (self.base_el[id], self.base_el[id] + (*bytes / 4) as i64))).collect(), None => Vec::new() } } else { Vec::new() };
        while f < self.words {
            let n = (self.words - f).min(self.tmp_words);
            self.gen(cuda, self.tmp[0], f, n, seed_a, sa)?;
            self.gen(cuda, self.tmp[1], f, n, seed_b, sb)?;
            let (mut x, mut sv, mut y, mut nn, mut part) = (self.tmp[0], self.sh_base + 4 * f as u64, self.tmp[1], n, self.part);
            checked("dot launch", (cuda.launch)(self.f.dot2, 1024, 1, 1, 256, 1, 1, 0, ptr::null_mut(), kargs!(x, sv, y, nn, part).as_mut_ptr(), ptr::null_mut()))?;
            let mut host = vec![0f64; 3072];
            checked("cuMemcpyDtoH partials", (cuda.download)(host.as_mut_ptr().cast(), self.part, 3072 * 8))?;
            for b in 0..1024 { p += host[3 * b]; q += host[3 * b + 1]; bad += host[3 * b + 2] as u64; }
            if let Some(acc) = alloc.as_mut() {
                // AD_DOT_ALLOC=1 (diagnostic only, step6-radiation): the same sum a*S split by allocation, in float64 on
                // the host. It reads the chunk's a and S back; nothing it computes feeds the window.
                let mut av = vec![0f32; n as usize]; let mut sv2 = vec![0f32; n as usize];
                checked("cuMemcpyDtoH dot a", (cuda.download)(av.as_mut_ptr().cast(), self.tmp[0], n as usize * 4))?;
                checked("cuMemcpyDtoH dot S", (cuda.download)(sv2.as_mut_ptr().cast(), self.sh_base + 4 * f as u64, n as usize * 4))?;
                for (id, (e0, e1)) in &ranges {
                    let (lo, hi) = ((*e0).max(f), (*e1).min(f + n));
                    if lo >= hi { continue; }
                    let mut s1 = 0f64; let mut big = 0f64;
                    for e in lo..hi { let k = (e - f) as usize; let t = av[k] as f64 * sv2[k] as f64; s1 += t; big = big.max((sv2[k] as f64).abs()); }
                    let ent = acc.entry(*id).or_insert((0.0, 0.0)); ent.0 += s1; ent.1 = ent.1.max(big);
                }
            }
            f += n;
        }
        if let Some(acc) = alloc {
            let mut v: Vec<(u64, f64, f64)> = acc.into_iter().map(|(k, (a, b))| (k, a, b)).collect();
            v.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
            v.truncate(24);
            eprintln!("{}", json!({"event":"window_dot_alloc","seed_a":seed_a,"sum":p,"top":v}));
        }
        Ok((p, q, bad))
    }
    /// Diagnostic (--trace-norm 1, tangent passes only): the whole-arena sum of squares after every command; a
    /// command that grows it more than 100x (or makes it non-finite) is logged with the allocations that hold the mass.
    unsafe fn trace_norm(&mut self, cuda: &Cuda, step: usize, index: usize, op: &Operation) -> Result<(), String> {
        let nw = if self.trace_state { self.state_words } else { self.words };
        self.support.sqdiff_into(cuda, 0, self.sh_base, 0, nw as usize * 4, 4)?;
        let (s, _) = self.support.collect(cuda, 1, 0)?;
        let v = s[0];
        // AD_TRACE_ALL=step:lo:hi (diagnostic only, step6-radiation): every allocation's tangent sum of squares after
        // each command lo..hi of that step, to diff two processes command by command. Reads only.
        if let Ok(r) = std::env::var("AD_TRACE_ALL") {
            let p: Vec<usize> = r.split(':').filter_map(|x| x.parse().ok()).collect();
            if p.len() == 3 && step == p[0] && index >= p[1] && index <= p[2] {
                let what = match op { Operation::Kernel { function, .. } => function.clone(), _ => format!("{:?}", std::mem::discriminant(op)) };
                eprintln!("{}", json!({"event":"window_trace_cmd","step":step,"index":index,"command":what,"norms":self.norms(cuda, 100000)?}));
            }
        }
        if !v.is_finite() || (v > self.trace_growth * self.trace_last.max(1e-30) && self.trace_hits < self.trace_cap) {
            let what = match op { Operation::Kernel { function, .. } => function.clone(), _ => format!("{:?}", std::mem::discriminant(op)) };
            let top = self.norms(cuda, 4)?;
            eprintln!("{}", json!({"event":"window_trace_growth","step":step,"index":index,"command":what,"before":self.trace_last,"after":v,"top":top}));
            self.trace_hits += 1;
            if !v.is_finite() { return Err(format!("step {step}: non-finite tangent after command {index} ({what})")); }
        }
        self.trace_last = v;
        Ok(())
    }
    /// Per-allocation shadow norm (sum of squares), largest first: where the derivative mass sits.
    unsafe fn norms(&self, cuda: &Cuda, top: usize) -> Result<Vec<(u64, f64)>, String> {
        let ids: Vec<(u64, i64, usize)> = self.shadows.iter().map(|(id, s)| (*id, self.base_el[id], s.bytes)).collect();
        let mut out = Vec::new();
        for chunk in ids.chunks(dev::SLOTS) {
            for (k, (_, el, bytes)) in chunk.iter().enumerate() { self.support.sqdiff_into(cuda, k, self.sh(*el), 0, *bytes, 4)?; }
            let (s, _) = self.support.collect(cuda, chunk.len(), 0)?;
            out.extend(chunk.iter().zip(s).map(|((id, _, _), v)| (*id, v)));
        }
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(top);
        Ok(out)
    }
    /// Sum of squares of the shadow of each listed allocation (in order).
    /// G6 diagnostic: (function, shadowed device-argument allocations) when the command is a kernel named in AD_G6_WATCH.
    unsafe fn g6_watch_ids(&self, ex: &Execution, op: &Operation, step: usize, frame: &Option<ReplayFrame>) -> Result<Option<(String, Vec<u64>)>, String> {
        static WATCH: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
        let w = WATCH.get_or_init(|| std::env::var("AD_G6_WATCH").map(|v| v.split(',').map(str::to_owned).collect()).unwrap_or_default());
        if w.is_empty() { return Ok(None); }
        let Operation::Kernel { function, arguments, .. } = op else { return Ok(None) };
        if !w.iter().any(|x| x == function) { return Ok(None); }
        let mut ids = Vec::new();
        for x in arguments {
            if let submit::Argument::Device(sl) = ex.argument(x, step, frame.as_ref())? { if self.shadows.contains_key(&sl.allocation) { ids.push(sl.allocation); } }
        }
        ids.sort_unstable(); ids.dedup();
        Ok(Some((function.clone(), ids)))
    }
    unsafe fn alloc_norms(&self, cuda: &Cuda, ids: &[u64]) -> Result<Vec<f64>, String> {
        let mut out = Vec::with_capacity(ids.len());
        for chunk in ids.chunks(dev::SLOTS) {
            for (k, id) in chunk.iter().enumerate() {
                let (el, bytes) = (self.base_el[id], self.shadows[id].bytes);
                self.support.sqdiff_into(cuda, k, self.sh(el), 0, bytes, 4)?;
            }
            let (s, _) = self.support.collect(cuda, chunk.len(), 0)?;
            out.extend(s.into_iter().take(chunk.len()));
        }
        Ok(out)
    }
    /// (tangent, primal) sums of squares over each native segment (alloc, lo, hi): the shadow words and the live words.
    unsafe fn seg_norms(&self, ex: &Execution, segs: &[(u64, usize, usize, u64)]) -> Result<Vec<(f64, f64)>, String> {
        let cuda = &ex.cuda;
        let mut out = Vec::with_capacity(segs.len());
        for chunk in segs.chunks(dev::SLOTS / 2) {
            for (k, (al, lo, hi, _)) in chunk.iter().enumerate() {
                let n = (hi - lo) / 4 * 4;
                self.support.sqdiff_into(cuda, 2 * k, self.sh(self.base_el[al] + (*lo / 4) as i64), 0, n, 4)?;
                let a = ex.allocations.get(al).ok_or("segment allocation not live")?.address + *lo as u64;
                self.support.sqdiff_into(cuda, 2 * k + 1, a, 0, n, 4)?;
            }
            let (s, _) = self.support.collect(cuda, 2 * chunk.len(), 0)?;
            for k in 0..chunk.len() { out.push((s[2 * k], s[2 * k + 1])); }
        }
        Ok(out)
    }
    /// Sum of squares over the plan's initial allocations (the carried state; see trace_state).
    unsafe fn state_norm(&self, cuda: &Cuda) -> Result<f64, String> {
        self.support.sqdiff_into(cuda, 0, self.sh_base, 0, self.state_words as usize * 4, 4)?;
        Ok(self.support.collect(cuda, 1, 0)?.0[0])
    }
    /// The per-step tangent record (AD_STEP_NORMS / --trace-norm): whole arena, state, and every nonzero allocation.
    unsafe fn step_norm_line(&self, cuda: &Cuda, seed: u64, step: usize) -> Result<Value, String> {
        let all = self.norms(cuda, usize::MAX)?;
        let total: f64 = all.iter().map(|x| x.1).sum();
        let state: f64 = all.iter().filter(|x| self.base_el[&x.0] < self.state_words).map(|x| x.1).sum();
        let nonfinite = all.iter().filter(|x| !x.1.is_finite()).count();
        let m: BTreeMap<String, f64> = all.iter().filter(|x| x.1 != 0.0).map(|x| (x.0.to_string(), x.1)).collect();
        Ok(json!({"event":"window_tangent_step_norms","seed":seed,"step":step,"total":total,"state":state,"nonfinite_allocs":nonfinite,"alloc":m}))
    }
    unsafe fn nonzero(&self, cuda: &Cuda) -> Result<u64, String> {
        self.support.reset_words(cuda)?;
        self.support.words_into(cuda, 0, self.sh_base, 0, self.words as usize * 4)?;
        let (_, w) = self.support.collect(cuda, 0, 1)?;
        Ok(w[0][1])
    }
    unsafe fn download_all(&self, cuda: &Cuda) -> Result<Vec<u8>, String> { dev::sync(cuda)?; dev::download(cuda, self.sh_base, self.words as usize * 4) }
}

fn rel(a: f64, b: f64) -> f64 { if a == b { 0.0 } else { (a - b).abs() / a.abs().max(b.abs()).max(1e-300) } }
/// SHA-256 over 16 equal chunks hashed on all cores, then the SHA-256 of the chunk digests in order: a bitwise
/// fingerprint of the reverse result, so reverse runs in separate processes (leases) can be compared word for word.
fn result_sha256(a: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let chunk = a.len().div_ceil(16).max(1);
    let parts: Vec<Vec<u8>> = std::thread::scope(|sc| {
        let hs: Vec<_> = a.chunks(chunk).map(|x| sc.spawn(move || Sha256::digest(x).to_vec())).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut h = Sha256::new();
    for p in &parts { h.update(p); }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}
fn differing_words(a: &[u8], b: &[u8]) -> u64 {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16);
    let chunk = (a.len() / 4).div_ceil(threads).max(1) * 4;
    std::thread::scope(|sc| {
        let hs: Vec<_> = a.chunks(chunk).zip(b.chunks(chunk)).map(|(x, y)| sc.spawn(move ||
            x.chunks(4).zip(y.chunks(4)).filter(|(p, q)| p != q).count() as u64)).collect();
        hs.into_iter().map(|h| h.join().unwrap()).sum()
    })
}

/// `ad-binder window --config plan.json --steps N --kernels K.json --cubins DIR --out DIR --bstore DIR
///   [--schemes schemes.json] [--hold s1,s2] [--refuse f1,f2=why] [--seeds 1,2] [--tape-gib 4] [--tape-max-gib G] [--tape-reserve-gib 16] [--scratch-gib 1]
///   [--repeat-reverse 1] [--zero 1]`
pub fn run(argv: Vec<String>) -> Result<(), String> {
    let mut a = HashMap::new();
    let mut it = argv.into_iter();
    while let Some(k) = it.next() {
        let key = k.strip_prefix("--").ok_or_else(|| format!("unexpected argument {k}"))?.to_owned();
        a.insert(key, it.next().ok_or_else(|| format!("missing value for {k}"))?);
    }
    let t_start = Instant::now();
    let config = PathBuf::from(req(&a, "config")?);
    let steps: usize = req(&a, "steps")?.parse().map_err(|_| "invalid --steps")?;
    let out = PathBuf::from(req(&a, "out")?);
    let bdir = PathBuf::from(req(&a, "bstore")?);
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    fs::create_dir_all(&bdir).map_err(|e| e.to_string())?;
    let seeds: Vec<u64> = opt(&a, "seeds").unwrap_or("1,2").split(',').filter_map(|s| s.trim().parse().ok()).collect();
    let bytes = fs::read(&config).map_err(|e| format!("cannot read {}: {e}", config.display()))?;
    let envelope: Value = serde_json::from_slice(&bytes).map_err(|e| format!("invalid plan JSON: {e}"))?;
    drop(bytes);
    if envelope.get("schema").and_then(|x| x.as_str()) != Some("ad.binder.plan/v1") || envelope.get("complete").and_then(|x| x.as_bool()) != Some(true) {
        return Err("refusing incomplete or unknown plan".into());
    }
    let plan: Plan = serde_json::from_value(envelope).map_err(|e| format!("invalid sealed plan: {e}"))?;
    crate::peer::set_host_unchecked(&plan.peer_host_unchecked);
    crate::preflight(&plan, steps)?;
    let base = config.parent().unwrap_or(Path::new(".")).to_owned();
    crate::verify_fixed_argument_sources(&plan, &base)?;
    crate::verify_copy_indexer_sources(&plan, &base)?;
    let reference = plan.reference.as_deref().map(|p| crate::path(&base, p)).ok_or("window needs the ordinary-forward reference")?;
    let refs = crate::read_references(&reference, &plan, steps)?;
    let mut ex = unsafe { Execution::new(None)? };
    ex.address_bearing = plan.address_bearing.iter().copied().collect();
    unsafe { ex.prepare(&plan, &base)? };
    let mut frame = plan.physics.as_ref().map(ReplayFrame::new).transpose()?;
    if let (Some(f), Some(spec)) = (frame.as_ref(), plan.physics.as_ref()) { ex.host_state.insert(spec.frame_state_id, f.snapshot()?)?; }
    let t_loaded = t_start.elapsed().as_secs_f64();
    let spec_check = check_spec_hashes(Path::new(req(&a, "kernels")?))?;
    let mut w = unsafe { Win::new(&ex, &plan, &a)? };
    // lane rung3-params: carrier at p0 and the forward of every carried family through the carrier image (its
    // boundary hashes are checked against the ordinary-forward reference like every step).
    let param_fd: Option<String> = opt(&a, "param-fd").map(str::to_owned);
    // lane s16: --params-unchecked 1 (default off): the iterate is not the recorded literal, so the base forward is
    // not compared with the ordinary-forward reference (the comparison would refuse step 1 by design).
    let params_unchecked = opt(&a, "params-unchecked") == Some("1");
    if params_unchecked && opt(&a, "params").is_none() { return Err("--params-unchecked 1 needs --params".into()); }
    let fwd_only = opt(&a, "forward-only") == Some("1");
    // multidevice-runner: the forward peer messages are keyed to the base trajectory (peer.rs), so a pass whose forward
    // is not the base trajectory cannot run on rank plans. THE BREAKAGE THIS PREVENTS: a Taylor row's perturbed forward
    // (or any other perturbed iterate) would read the base trajectory's halo bytes from its peer and replay a wrong forward.
    if let Some(p) = crate::peer::get() {
        for k in ["taylor", "taylor-dir", "ext-driver", "params", "param-fd", "param-tl"] {
            if opt(&a, k).is_some() { return Err(format!("--{k} on a rank plan (rank {} of {}): perturbed forwards are not exchanged yet", p.rank, p.ranks)); }
        }
    }
    let no_tangent = opt(&a, "no-tangent") == Some("1");
    if no_tangent && (opt(&a, "param-tl").is_some() || opt(&a, "tangent-only") != Some("1")) { return Err("--no-tangent 1 needs --tangent-only 1 and no --param-tl (no tangent to test)".into()); }
    let param_tl: Option<Vec<f64>> = opt(&a, "param-tl").map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect());
    if !w.params.is_empty() {
        unsafe { w.par_set(&ex.cuda, None, None)? };
        let n = w.native.as_ref().unwrap();
        for p in &w.params {
            let f = match p.fam.as_str() { "mynnsfc" => "mynnsfc_columns", "ugwp" => "ugwp_oro_run", "mynnedmf" => "rrfs_mynnedmf_run_l16",
                                           x => return Err(format!("--params: family {x}")) };
            let h = unsafe { n.par_forward(&ex.cuda, &p.fam, f)? };
            ex.kernel_override.insert(f.to_owned(), h);
        }
        eprintln!("{}", json!({"event":"param_carrier","params":w.params.iter().map(|p| json!({"name":p.name,"family":p.fam,"slot":p.slot,
            "p0":p.p0,"p0_bits":format!("0x{:08x}", p.p0.to_bits())})).collect::<Vec<_>>(),"overrides":ex.kernel_override.keys().collect::<Vec<_>>()}));
    }
    w.concat_types = crate::wrules::concat_module_types(&plan, &base)?;
    eprintln!("{}", json!({"event":"concat_module_types","types":w.concat_types}));
    let cuda = &ex.cuda as *const Cuda;
    let mut receipt = json!({"event":"window_receipt","plan":config.display().to_string(),"steps":steps,"seeds":seeds,
        "args":a,"stages":{"plan_loaded_s":t_loaded},"tolerance":DOT_F32,"started_unix":now(),"spec_sha256":spec_check});
    if !w.params.is_empty() {
        receipt["params"] = json!(w.params.iter().map(|p| json!({"name":p.name,"family":p.fam,"slot":p.slot,"p0":p.p0,
            "p0_bits":format!("0x{:08x}", p.p0.to_bits())})).collect::<Vec<_>>());
        receipt["param_forward_overrides"] = json!(ex.kernel_override.keys().collect::<Vec<_>>());
    }
    let log = |v: Value| eprintln!("{v}");
    #[cfg(hexad_ext_driver)]
    if opt(&a, "ext-driver").is_some() {
        let r = unsafe { ext_driver::run(&mut w, &mut ex, &plan, &base, &mut frame, &refs, &a, &out, &bdir, steps, receipt) };
        if let Err(e) = &r { eprintln!("{}", json!({"event":"ext_driver_error","error":e})); }
        return r;
    }
    #[cfg(not(hexad_ext_driver))]
    if opt(&a, "ext-driver").is_some() {
        return Err("DEFECT: --ext-driver: this build does not carry an extension driver".into());
    }
    unsafe {
        // 1. Augmented forward, seed 0's tangent, storing every boundary.
        let mut bounds: Vec<Boundary> = Vec::with_capacity(steps + 1);
        let mut flats: Vec<Vec<Flat>> = Vec::with_capacity(steps);
        let mut per_step = Vec::new();
        let sx = |s: u64| 0x5eed_0000_0000 + 2 * s + 1;
        let sy = |s: u64| 0x5eed_0000_0000 + 2 * s + 2;
        let mut dots = Vec::new();
        let mut rev_first: Option<Vec<u8>> = None;
        let repeat = opt(&a, "repeat-reverse").map(|v| v != "0").unwrap_or(true);
        let anchors: usize = opt(&a, "anchors").and_then(|v| v.parse().ok()).unwrap_or(3);
        // S02 (default off: every boundary stored whole, as before): --checkpoints c (or "sqrt": ceil(sqrt(steps))) keeps
        // boundary 0, the window end and at most c step checkpoints, recomputing the rest in the reverse (revolve);
        // the files are changed-chunk unless --bstore-delta 0.
        let ckpts: Option<usize> = match opt(&a, "checkpoints") {
            None | Some("every") => None,
            Some("sqrt") => Some((steps as f64).sqrt().ceil() as usize),
            Some(v) => Some(v.parse().map_err(|_| format!("--checkpoints {v}: a count, sqrt or every"))?),
        };
        // integ8: the --params reverses (rung3/S16: the parameter reverse and --grad-all) walk every stored boundary;
        // with --checkpoints only boundary 0 and the window end are stored, so that pairing would read past the store.
        if ckpts.is_some() && opt(&a, "params").is_some() { return Err("--checkpoints with --params: the parameter reverses need every boundary stored (drop --checkpoints)".into()); }
        w.delta_store = ckpts.is_some() && opt(&a, "bstore-delta") != Some("0");
        w.bs_ram_budget = match opt(&a, "bstore-ram-gib").and_then(|v| v.parse::<f64>().ok()) {
            Some(g) => (g.max(0.0) * (1u64 << 30) as f64) as usize,
            None => (fs::read_to_string("/proc/meminfo").ok().and_then(|m| m.lines().find(|l| l.starts_with("MemAvailable"))
                        .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())).unwrap_or(0) as f64 * 1024.0 * 0.4) as usize,
        };
        let rv = ckpts.map(|c| Revolve::new(steps, c));
        let descent: Option<Vec<usize>> = rv.as_ref().map(|r| r.descent(steps, ckpts.unwrap()));
        let mut ck: BTreeMap<usize, Boundary> = BTreeMap::new();
        let mut sweeps: Vec<Value> = Vec::new();
        if let (Some(r), Some(c)) = (&rv, ckpts) {
            receipt["checkpointing"] = json!({"checkpoints":c,"schedule":"revolve (binomial, cost-optimal DP)","first_descent":descent,
                "planned_recompute_steps_per_sweep":r.cost[steps][c],"bstore":if w.delta_store { "changed-chunk (512 B, vs boundary 0)" } else { "whole" }});
        }
        // Directions: relative (each word weighted by its |primal| at the boundary it lives on) unless --direction plain.
        let rel_dir = opt(&a, "direction") != Some("plain");
        let taylor: Vec<f64> = opt(&a, "taylor").map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default();
        // lane taylor-repose: fixed direction, forward-only quotients, regime words (all default off)
        let read_terms = |p: &str| -> Result<(Value, Vec<(u64, usize, f32)>), String> {
            let doc: Value = serde_json::from_slice(&fs::read(p).map_err(|e| format!("{p}: {e}"))?).map_err(|e| format!("{p}: {e}"))?;
            let mut v = Vec::new();
            for t in doc["terms"].as_array().ok_or_else(|| format!("{p}: terms"))? {
                let (Some(al), Some(wd), Some(wt)) = (t[0].as_u64(), t[1].as_u64(), t[2].as_f64()) else { return Err(format!("{p}: term")) };
                v.push((al, wd as usize, wt as f32));
            }
            Ok((doc["name"].clone(), v))
        };
        let tdir: Option<(Value, Vec<(u64, usize, f32)>)> = match opt(&a, "taylor-dir") { Some(p) => Some(read_terms(p)?), None => None };
        // lane j4-repose: an objective file with "form" is a smooth nonlinear objective (Form); without it, linear as before.
        let mut forms: Vec<Option<Form>> = Vec::new();
        let mut objs: Vec<(String, Value, Vec<(u64, usize, f32)>)> = Vec::new();
        for p in opt(&a, "objective").unwrap_or("").split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let doc: Value = serde_json::from_slice(&fs::read(p).map_err(|e| format!("{p}: {e}"))?).map_err(|e| format!("{p}: {e}"))?;
            if doc.get("form").is_some() {
                let (f, t) = Form::parse(&doc, p)?;
                objs.push((p.to_owned(), doc["name"].clone(), t)); forms.push(Some(f));
            } else { let (n, t) = read_terms(p)?; objs.push((p.to_owned(), n, t)); forms.push(None); }
        }
        // J(h) - J(0) of objective i from the live window-end words q and the stored base words p.
        let obj_dj = |i: usize, q: &[f64], p: &[f64]| -> f64 {
            match &forms[i] { Some(f) => f.eval(q).0 - f.eval(p).0,
                              None => objs[i].2.iter().zip(q.iter().zip(p)).map(|(t, (a, b))| t.2 as f64 * (a - b)).sum() }
        };
        let fd_only = opt(&a, "fd-only") == Some("1") || fwd_only;
        if fd_only && !fwd_only && ((tdir.is_none() && param_fd.is_none()) || objs.is_empty() || taylor.is_empty()) { return Err("--fd-only 1 needs --taylor-dir or --param-fd, --objective and --taylor".into()); }
        if param_fd.is_some() && (!fd_only || w.params.is_empty()) { return Err("--param-fd needs --fd-only 1 and --params".into()); }
        if param_tl.is_some() && (w.params.is_empty() || objs.is_empty()) { return Err("--param-tl needs --params and --objective".into()); }
        // lane runner-dump: window-end state dumps (default off) and the tangent along the Taylor direction
        let dump_state: Vec<(u64, usize, Option<usize>)> = opt(&a, "dump-state").unwrap_or("").split(',').map(str::trim).filter(|x| !x.is_empty()).map(|x| {
            let q: Vec<&str> = x.split(':').collect();
            let id = q[0].parse::<u64>().map_err(|_| format!("--dump-state {x}: allocation id"))?;
            let r = match q.len() {
                1 => (id, 0, None),
                3 => (id, q[1].parse::<usize>().map_err(|_| format!("--dump-state {x}: byte offset"))?, Some(q[2].parse::<usize>().map_err(|_| format!("--dump-state {x}: bytes"))?)),
                _ => return Err(format!("--dump-state {x}: want id or id:byte_offset:bytes")),
            };
            if r.1 % 4 != 0 || r.2.is_some_and(|n| n % 4 != 0 || n == 0) { return Err(format!("--dump-state {x}: offset and bytes must be whole float32 words")); }
            Ok(r)
        }).collect::<Result<_, String>>()?;
        let dump_name = |tag: &str, (id, off, n): (u64, usize, Option<usize>)| if n.is_some() { format!("{tag}-{id}-{off}.f32") } else { format!("{tag}-{id}.f32") };
        let tangent_dir = opt(&a, "tangent-taylor-dir") == Some("1") || opt(&a, "tangent-dir") == Some("1");
        if tangent_dir && (tdir.is_none() || opt(&a, "tangent-only") != Some("1") || fd_only) {
            return Err("--tangent-taylor-dir 1 needs --taylor-dir and --tangent-only 1 (the seed dots would mix directions), and not --fd-only".into());
        }
        // the base window-end words and the tangent's, kept for the Taylor rows' scores
        let mut state_base: Vec<Vec<u8>> = Vec::new();
        let mut state_tl: Vec<Vec<u8>> = Vec::new();
        let mut state_dumps = json!({"base":[],"tangent":[],"rows":[]});
        let regime: Vec<(String, u64, usize, usize, String, String, Option<(usize, Vec<bool>)>)> = match opt(&a, "regime") {
            Some(p) => {
                let doc: Value = serde_json::from_slice(&fs::read(p).map_err(|e| format!("{p}: {e}"))?).map_err(|e| format!("{p}: {e}"))?;
                doc["fields"].as_array().ok_or("regime fields")?.iter().map(|f| Ok((f["name"].as_str().unwrap_or("?").to_owned(),
                    f["allocation"].as_u64().ok_or("regime allocation")?, f["offset"].as_u64().ok_or("regime offset")? as usize,
                    f["count"].as_u64().ok_or("regime count")? as usize, f["dtype"].as_str().unwrap_or("f4").to_owned(),
                    f["test"].as_str().unwrap_or("eq").to_owned(),
                    // S04: the cell set as a mask over 0..ncells (a listed id >= ncells never matched i % ncells).
                    f["cells"].as_array().map(|c| { let nc = f["ncells"].as_u64().unwrap_or(2562) as usize; let mut m = vec![false; nc];
                        for x in c.iter().filter_map(|x| x.as_u64()) { if (x as usize) < nc { m[x as usize] = true; } } (nc, m) }))))
                    .collect::<Result<Vec<_>, String>>()?
            }
            None => Vec::new(),
        };
        let read_regime = |ex: &Execution| -> Result<Vec<Vec<u8>>, String> {
            dev::sync(&ex.cuda)?;
            regime.iter().map(|(_, al, off, n, _, _, _)| {
                let a = ex.allocations.get(al).ok_or_else(|| format!("regime allocation {al} not live"))?;
                dev::download(&ex.cuda, a.address + *off as u64, 4 * n)
            }).collect()
        };
        let regime_diff = |x: &[Vec<u8>], y: &[Vec<u8>]| -> Vec<u64> {
            regime.iter().zip(x.iter().zip(y)).map(|((_, _, _, _, dt, test, cells), (p, q))| {
                p.chunks_exact(4).zip(q.chunks_exact(4)).enumerate().filter(|(i, _)| cells.as_ref().is_none_or(|(nc, m)| m[i % nc]))
                    .map(|(_, uv)| uv).filter(|(u, v)| {
                    if dt == "i4" || test == "eq" { return u != v; }
                    let (a, b) = (f32::from_le_bytes((*u).try_into().unwrap()), f32::from_le_bytes((*v).try_into().unwrap()));
                    let cls = |z: f32| if test == "pos" { (z > 0.0) as i32 } else if z > 0.0 { 1 } else if z < 0.0 { -1 } else { 0 };
                    cls(a) != cls(b)
                }).count() as u64
            }).collect()
        };
        let mut regime_base: Vec<Vec<Vec<u8>>> = Vec::new();
        if !taylor.is_empty() && !rel_dir && tdir.is_none() { return Err("--taylor needs relative directions (address words must get zero weight)".into()); }
        let mut taylor_gx: Option<f64> = None;
        receipt["fct_frozen"] = json!(w.fct_frozen);
        if w.fct_frozen {
            receipt["approximation"] = json!("FCT limiter coefficients frozen from the forward trajectory (constants in TL and AD): fct_scale's scale_in/scale_out carry zero tangent and drop their cotangent; fct_minmax_source differentiates source_old only. This operator is not the exact derivative of the forward's limiter.");
        }
        let (mut d0, mut dn): (Option<std::sync::Arc<Bytes>>, Option<Bytes>) = (None, None);
        receipt["stable_directions"] = json!(if w.stable_dir { "on: translated address words, relocation slots, address-bearing tables and declared residue ranges of each boundary get zero weight, so x and y are the same in every process" } else { "off" });
        receipt["direction"] = json!(if opt(&a, "direction") == Some("fieldxy") { "fieldxy (candidate fix B, test-only): x = u * rms_f(boundary 0), y = u / rms_f(window end), rms_f the RMS of allocation f's valid words (finite, at most 2e5, above 1e-30) at that boundary; invalid words 0 on both sides. The operator is unchanged; the same x and y seed <y, M x>, the reverse and <x, M^T y>." } else if opt(&a, "direction") == Some("field") { "field: x = u * |primal at boundary 0| (relative); y = u / rms_f, rms_f the RMS of allocation f's valid words (finite, at most 2e5, above 1e-30) at the window end; invalid words 0. The same y seeds <y, M x> and the reverse." } else if rel_dir { "relative: x = u * |primal at boundary 0|, y = u * |primal at the window end| (finite, at most 2e5; larger words are stale scratch or address words and get 0)" } else { "plain uniform [-1,1) on every shadow word" });
        macro_rules! xs { () => { if rel_dir { Some((&bounds[0], &d0.as_ref().unwrap()[..])) } else { None } } }
        macro_rules! ys { () => { if rel_dir { Some((&bounds[bounds.len() - 1], dn.as_deref().unwrap())) } else { None } } }
        let plan_live_bound: usize = {
            let names: Vec<&String> = if plan.physics.is_some() { plan.blocks.keys().collect() }
                                      else { plan.cadence.iter().take(steps).flatten().collect::<std::collections::BTreeSet<_>>().into_iter().collect() };
            plan.allocations.iter().map(|x| x.bytes).sum::<usize>() + names.iter().filter_map(|n| plan.blocks.get(*n)).flatten()
                .map(|o| if let Operation::Allocate { bytes, .. } = o { *bytes } else { 0 }).sum::<usize>()
        };
        for (k, &seed) in seeds.iter().enumerate() {
            let t0 = Instant::now();
            if k > 0 { let d = w.load(&bounds[0])?; w.restore(&mut ex, &mut frame, &bounds[0], &d)?; }
            else {
                bounds.push(w.store(&mut ex, &frame, &bdir, 0)?);
                let b0 = std::sync::Arc::new(w.load(&bounds[0])?);
                if w.delta_store { w.ckpt_base = Some((bounds[0].index.clone(), b0.clone())); }
                d0 = Some(b0);
            }
            if tangent_dir && k == 0 {
                w.zero(&*cuda)?;
                for &(al, wd, v) in &tdir.as_ref().unwrap().1 {
                    let el = *w.base_el.get(&al).ok_or_else(|| format!("taylor-dir allocation {al} has no shadow"))?;
                    dev::upload(&*cuda, w.sh(el + wd as i64), &v.to_le_bytes())?;
                }
                dev::sync(&*cuda)?;
            } else if let (Some(c), 0) = (&param_tl, k) { w.zero(&*cuda)?; w.par_set(&*cuda, None, Some(c))?; }
            else { w.fill(&*cuda, sx(seed), xs!())?; }
            if let Some(p) = crate::peer::get() { p.begin_tl(); }
            w.trace_last = f64::MAX;
            w.counting = k == 0;
            w.entry_ms = 0.0;
            // terrain-w-hash: this pass's boundary digests run beside the next step (joined at the next boundary and
            // when the pass ends, below); every comparison and refusal is the same.
            ex.digest_defer = true;
            if k == 0 && descent.is_none() && !w.delta_store && w.bs_ram_budget > w.bs_ram_now { w.pin_prefill(&ex, steps)?; }
            w.gather_pass_join = true;
            for s in 1..=steps {
                let ts = Instant::now();
                let ms0 = w.entry_ms;
                let sa0 = crate::staging_allocs();
                // integ8: capture the whole state only at a step this pass stores (with S02 --checkpoints most steps are not
                // stored); a capture never outlives its step (cleared below), so no later step can read it.
                w.fuse_rec = w.fuse && k == 0 && descent.is_none(); w.fz_steps = steps;
                w.capture = k == 0 && descent.as_ref().map_or(true, |p| s == steps || p.contains(&s));
                // host-leftovers-2: a stored boundary goes straight into page-locked RAM while the budget has room
                // (--bstore-ram-gib, default 40 % of MemAvailable at start; 0 = files as before; never with changed-chunk files).
                w.capture_host = if w.capture && !w.delta_store && w.bs_ram_budget > w.bs_ram_now { Some((0, w.bs_ram_budget - w.bs_ram_now)) } else { None };
                w.pin_ahead = k == 0 && s < steps && descent.is_none();
                let flat = if params_unchecked { w.forward_step_c(&mut ex, &plan, &base, &mut frame, None, s, !fd_only && !no_tangent)? }
                           else { w.forward_step(&mut ex, &plan, &base, &mut frame, &refs, s, !fd_only && !no_tangent)? };
                w.capture = false; w.capture_host = None;
                if k == 0 {
                    match &descent {
                        None => bounds.push(w.store(&mut ex, &frame, &bdir, s)?),
                        Some(p) => if s == steps { bounds.push(w.store(&mut ex, &frame, &bdir, s)?); }
                                   else if p.contains(&s) { let b = w.store(&mut ex, &frame, &bdir, s)?; ck.insert(s, b); },
                    }
                    ex.captured = None;
                    if !regime.is_empty() { regime_base.push(read_regime(&ex)?); }
                    per_step.push(json!({"step":s,"commands":flat.len(),"tangent_wall_s":ts.elapsed().as_secs_f64(),
                                         "tangent_entry_s":(w.entry_ms - ms0) / 1e3}));
                    flats.push(flat);
                    if s == 1 && steps >= 2 && !w.fuse { w.ahead_maybe(&ex, &flats, anchors, opt(&a, "tape-reserve-gib"), false, s, plan_live_bound)?; }
                    if w.step_norms { log(w.step_norm_line(&*cuda, seed, s)?); }
                }
                let host_prof: Vec<Value> = dev::prof::take().into_iter().filter(|x| x.2 > 0).map(|(k, s, c)| json!([k, s, c])).collect();
                let rows = w.take_rows(0);
                log(json!({"event":"window_tangent_step","seed":seed,"step":s,"tape_rows":rows,"wall_s":ts.elapsed().as_secs_f64(),"staging_allocs":crate::staging_allocs() - sa0,"boundary_hashes":"equal","boundary_check":if ex.digest_defer { "deferred: compared while the next step runs, and before this pass ends" } else { "at this boundary" },"host_prof":host_prof,"reference_check":!params_unchecked}));
            }
            w.gather_pass_join = false;
            // the gathers begun in the last step stay pending (installed on demand in the reverse, after the tape is sized), as
            // the per-step join left them: x20-E-gj joined them all here and the tape shrank 10.3 -> 7.1 GiB (segments 5/8/9 -> 8/13/14).
            { let t = dev::prof::start(); w.join_pending_if(&*cuda, steps, |p| p.3 != steps)?; dev::prof::add(13, t); }
            ex.digest_defer = false;
            ex.digest_flush()?;
            w.counting = false;
            if let (Some(c), 0) = (&param_tl, k) {
                // lane rung3-params: x = 0 in the state, d ln p = c in the parameters; <objective weights, tangent at N>.
                // lane s16: a form objective's weights are its seed at the base window end.
                let endb = w.load(&bounds[bounds.len() - 1])?;
                let mut wterms: Vec<Vec<(u64, usize, f32)>> = Vec::new();
                for (i, (_, _, terms)) in objs.iter().enumerate() {
                    wterms.push(match &forms[i] { Some(f) => { let g = f.eval(&base_words(&bounds[bounds.len() - 1], &endb, terms)?).1;
                        terms.iter().zip(&g).map(|(t, &x)| (t.0, t.1, x as f32)).collect() }, None => terms.clone() });
                }
                drop(endb);
                let tl_dj = w.sparse_dot(&*cuda, &wterms[0])?;
                let mut tl_all = Vec::new();
                for (i, (pth, name, _)) in objs.iter().enumerate() { tl_all.push(json!({"file":pth,"name":name,"tl_dJ":w.sparse_dot(&*cuda, &wterms[i])?})); }
                w.par_set(&*cuda, None, None)?;
                receipt["param_tl"] = json!({"coefficients_dlnp":c,"objective":objs[0].1,"tl_dJ":tl_dj,"all":tl_all});
                log(json!({"event":"param_tl","tl_dJ":tl_dj}));
            }
            if dn.is_none() { dn = Some(w.load(&bounds[bounds.len() - 1])?); w.y_path = Some(bounds[bounds.len() - 1].path.clone()); }
            if k == 0 && !objs.is_empty() {
                // lane s16: J of every objective at the base window end (form: float64 on the host; linear: sum w x).
                let end = dn.as_deref().unwrap();
                let mut vals = Vec::new();
                for (i, (pth, name, terms)) in objs.iter().enumerate() {
                    let v = base_words(&bounds[bounds.len() - 1], end, terms)?;
                    let j = match &forms[i] { Some(f) => f.eval(&v).0, None => terms.iter().zip(&v).map(|(t, x)| t.2 as f64 * x).sum() };
                    let bits: Vec<u8> = v.iter().flat_map(|x| (*x as f32).to_le_bytes()).collect();
                    vals.push(json!({"file":pth,"name":name,"J":j,"words":v.len(),"words_sha256":result_sha256(&bits)}));
                }
                receipt["objective_values"] = json!(vals);
                log(json!({"event":"objective_values","rows":receipt["objective_values"]}));
                fs::write(out.join("window.json"), serde_json::to_vec_pretty(&receipt).unwrap()).map_err(|e| e.to_string())?;
            }
            if k == 0 && !dump_state.is_empty() {
                let end = dn.as_deref().unwrap();
                for &(id, off, n) in &dump_state {
                    let (_, bo, bn) = *bounds[bounds.len() - 1].index.iter().find(|x| x.0 == id).ok_or_else(|| format!("--dump-state allocation {id} is not live at the window end"))?;
                    let len = n.unwrap_or(bn.saturating_sub(off));
                    if off + len > bn || len == 0 { return Err(format!("--dump-state {id}:{off}: outside allocation {id} ({bn} bytes)")); }
                    let b = end[bo + off..bo + off + len].to_vec();
                    let mut r = write_state(&out, dump_name("state", (id, off, n)), &b)?;
                    r["allocation"] = json!(id); r["byte_offset"] = json!(off);
                    state_dumps["base"].as_array_mut().unwrap().push(r);
                    state_base.push(b);
                    if !fd_only {
                        let (Some(&el), Some(_)) = (w.base_el.get(&id), w.shadows.get(&id)) else { return Err(format!("--dump-state allocation {id} has no shadow")) };
                        let t = dev::download(&*cuda, w.sh(el) + off as u64, len)?;
                        let mut r = write_state(&out, dump_name("tl", (id, off, n)), &t)?;
                        r["allocation"] = json!(id); r["byte_offset"] = json!(off);
                        r["direction"] = if tangent_dir { json!({"taylor_dir":opt(&a, "taylor-dir"),"name":tdir.as_ref().unwrap().0}) } else { json!({"seed":seed}) };
                        state_dumps["tangent"].as_array_mut().unwrap().push(r);
                        state_tl.push(t);
                    }
                }
                if tangent_dir && !objs.is_empty() {
                    // <w, M x>: the first objective's weights on the tangent at the window end (to set beside <x, M^T w>)
                    let mut ymx = 0f64;
                    // j4-repose: a form objective's weights are dJ/dx_N at the base window end (its seed)
                    let ow: Vec<f32> = match &forms[0] { Some(f) => f.eval(&base_words(&bounds[bounds.len() - 1], end, &objs[0].2)?).1.iter().map(|&x| x as f32).collect(),
                                                         None => objs[0].2.iter().map(|t| t.2).collect() };
                    for (&(al, wd, _), &wt) in objs[0].2.iter().zip(&ow) {
                        let el = *w.base_el.get(&al).ok_or_else(|| format!("objective allocation {al} has no shadow"))?;
                        let v = dev::download(&*cuda, w.sh(el + wd as i64), 4)?;
                        ymx += wt as f64 * f32::from_le_bytes(v[..4].try_into().unwrap()) as f64;
                    }
                    receipt["tangent_dir"] = json!({"taylor_dir":opt(&a, "taylor-dir"),"name":tdir.as_ref().unwrap().0,"objective":objs[0].1,"w_Mx":ymx});
                }
                receipt["state_dumps"] = state_dumps.clone();
                log(json!({"event":"window_state_dump","where":"base","fields":dump_state.len()}));
            }
            if fd_only { log(json!({"event":"window_fd_base_done","wall_s":t0.elapsed().as_secs_f64()})); break; }
            if no_tangent { log(json!({"event":"window_base_done_no_tangent","wall_s":t0.elapsed().as_secs_f64()})); break; }
            let (pt, qt, bad_t) = w.dots(&*cuda, sy(seed), ys!(), sx(seed), xs!())?;
            // multidevice-runner: <y, Mx> over all ranks (rank-ordered sum, the same in every rank); the local part is logged.
            let (pt, qt, bad_t) = match crate::peer::get() {
                Some(p) => { let g = p.reduce_sum(&format!("tl{seed}"), &[pt, qt, bad_t as f64])?;
                             log(json!({"event":"peer_tangent_dot","seed":seed,"rank":p.rank,"local":[pt, qt, bad_t],"global":g}));
                             (g[0], g[1], g[2] as u64) },
                None => (pt, qt, bad_t) };
            let t_tl = t0.elapsed().as_secs_f64();
            if k == 0 { receipt["tangent_norm_top_allocations"] = json!(w.norms(&*cuda, 12)?); }
            log(json!({"event":"window_tangent_done","seed":seed,"y_Mx":pt,"y_Mx_minus_x":qt,"nonfinite":bad_t,"wall_s":t_tl}));
            if opt(&a, "tangent-only") == Some("1") { continue; }
            w.fuse_rec = false;
            let all_fused = w.fuse && k == 0 && (1..=steps).all(|s| w.fz_tapes.contains_key(&s));
            if k == 0 && w.fuse { log(json!({"event":"window_fused_tape","steps_fused":w.fz_tapes.len(),"steps":steps,"chunks":w.fz_chunks.len(),
                "d2h_s":w.fz_d2h_s,"d2h_bytes":w.fz_d2h_bytes,"arena_bytes":w.rtape.as_ref().map(|t| t.bytes),"per_step":w.fz_info})); }
            // x20 fused tape: every step is fused, so no reverse bind takes the tangent arena: give its memory back to the
            // reverse (the MYNN-EDMF adjoint sizes its tiles to the card free memory) and keep a small one for later binds.
            if all_fused && w.tape.bytes > (64 << 20) {
                dev::sync(&*cuda)?;
                checked("cuMemFree tape", ((*cuda).free)(w.tape.base))?;
                w.tape = dev::Arena::new(&*cuda, 64 << 20)?;
                log(json!({"event":"window_fused_tape_tl_arena_released","tape_bytes":w.tape.bytes}));
            }
            // x20 fused tape: the reverse no longer replays the step, so nothing installs the gathers the last tangent step
            // left building until the walk-back's first rule needs one, after the MYNN-EDMF adjoint has taken the card's
            // free memory (x20rs-F2: "cuMemAlloc gather failed with CUDA status 2"). Install them all here, before the
            // reverse, where the replay used to install them.
            if all_fused { w.join_pending(&*cuda, steps)?; }
            if k == 0 && w.one_pass && !all_fused { w.size_tape(&ex, &flats, opt(&a, "tape-max-gib"), opt(&a, "tape-reserve-gib"))?; }
            if k == 0 && !all_fused { w.ahead_maybe(&ex, &flats, anchors, opt(&a, "tape-reserve-gib"), true, steps, plan_live_bound)?; }
            // 2. Reverse from y at the window end.
            let mut runs = Vec::new();
            for rep in 0..(if k == 0 && repeat { 2 } else { 1 }) {
                let tr = Instant::now();
                w.fill(&*cuda, sy(seed), ys!())?;
                if let Some(p) = crate::peer::get() { p.begin_ad(); }
                w.trace_last = f64::MAX; w.trace_hits = 0;
                if w.trace { log(json!({"event":"window_trace_y","norms":w.norms(&*cuda, 6)?})); }
                w.entry_ms = 0.0;
                let rows = if let (Some(r), Some(c)) = (&rv, ckpts) {
                    let (r0, t0s) = (w.recompute_steps, w.recompute_s);
                    let rows = w.sweep_revolve(&mut ex, &plan, &base, &mut frame, &refs, &flats, &bounds[0], &bounds[bounds.len() - 1], &mut ck, &bdir,
                                               r, c, anchors, json!({"seed":seed,"repeat":rep,"sweep":"dot"}))?;
                    sweeps.push(json!({"sweep":"dot","seed":seed,"repeat":rep,"recompute_steps":w.recompute_steps - r0,"recompute_s":w.recompute_s - t0s,
                                       "wall_s":tr.elapsed().as_secs_f64()}));
                    rows
                } else {
                let mut cur_data = w.load(&bounds[bounds.len() - 1])?;
                let mut rows = Vec::new();
                for s in (1..=steps).rev() {
                    let (sa0, rd0, tl0) = (crate::staging_allocs(), w.restore_direct, Instant::now());
                    let prev_data = w.load_at(&bounds, s - 1)?;
                    let load_s = tl0.elapsed().as_secs_f64();
                    if s >= 2 { w.prefetch_load(&bounds, s - 2)?; }
                    w.stage_blobs(&mut ex, &base, flat_blobs(&flats[s - 1]), (s >= 2).then(|| flat_blobs(&flats[s - 2])))?;
                    let r = if w.fz_tapes.contains_key(&s) { w.fused_reverse_step(&mut ex, &mut frame, s, s == steps, &bounds[s], &cur_data)? }
                            else { w.reverse_step(&mut ex, &base, &mut frame, s, &flats[s - 1], &bounds[s - 1], &prev_data, &bounds[s], &cur_data, anchors)? };
                    log(json!({"event":"window_reverse_step","seed":seed,"repeat":rep,"row":r,"host_leftovers2":{"staging_allocs":crate::staging_allocs() - sa0,
                               "restores_direct":w.restore_direct - rd0,"start_load_s":load_s,"step_wall_s":tl0.elapsed().as_secs_f64()}}));
                    rows.push(r);
                    cur_data = prev_data;
                }
                drop(cur_data);
                rows
                };
                let (pa, qa, bad_a) = w.dots(&*cuda, sx(seed), xs!(), sy(seed), ys!())?;
                let (pa, qa, bad_a) = match crate::peer::get() {
                    Some(p) => { let g = p.reduce_sum(&format!("ad{seed}.{rep}"), &[pa, qa, bad_a as f64])?;
                                 log(json!({"event":"peer_reverse_dot","seed":seed,"repeat":rep,"rank":p.rank,"local":[pa, qa, bad_a],"global":g}));
                                 (g[0], g[1], g[2] as u64) },
                    None => (pa, qa, bad_a) };
                let result = w.download_all(&*cuda)?;
                // multidevice-runner (AD_RESULT_DUMP=<dir>, proof runs only): the reverse result words and the arena layout,
                // so a multi-process result can be assembled per allocation and compared with the one-process run.
                if let Ok(dir) = std::env::var("AD_RESULT_DUMP") {
                    let tag = match crate::peer::get() { Some(p) => format!("r{}", p.rank), None => "r1p".into() };
                    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
                    fs::write(Path::new(&dir).join(format!("result-{tag}-s{seed}-rep{rep}.bin")), &result).map_err(|e| e.to_string())?;
                    let mut lay: Vec<(u64, i64, usize)> = w.base_el.iter().map(|(id, el)| (*id, *el, ex.allocations.get(id).map(|a| a.bytes).unwrap_or(0))).collect();
                    lay.sort();
                    let owned: Vec<u64> = plan.allocations.iter().map(|x| x.id).collect();
                    fs::write(Path::new(&dir).join(format!("layout-{tag}.json")), serde_json::to_vec(&json!({"layout":lay,"plan_allocations":owned,"words":w.words})).unwrap()).map_err(|e| e.to_string())?;
                }
                let mut row = json!({"seed":seed,"repeat":rep,"x_MTy":pa,"x_MTy_minus_y":qa,"nonfinite":bad_a,
                    "wall_s":tr.elapsed().as_secs_f64(),"adjoint_entry_s":w.entry_ms / 1e3,"steps":rows,
                    "result_sha256":result_sha256(&result),"result_bytes":result.len()});
                if rep == 0 && k == 0 && !taylor.is_empty() {
                    // <x', M^T y> with x' = the first seed's direction without address-bearing allocations (the Taylor direction).
                    let gx = if let Some((_, t)) = &tdir { w.sparse_dot(&*cuda, t)? } else {
                        w.taylor_mask = true;
                        let (gx, _, _) = w.dots(&*cuda, sx(seed), xs!(), sx(seed), xs!())?;
                        w.taylor_mask = false;
                        gx
                    };
                    taylor_gx = Some(gx);
                    row["taylor_x_MTy_masked"] = json!(gx);
                }
                if rep == 0 && k == 0 { rev_first = Some(result); }
                else if rep == 1 {
                    row["differing_words_vs_first_reverse"] = json!(differing_words(rev_first.as_ref().unwrap(), &result));
                    rev_first = None;
                }
                log(json!({"event":"window_reverse_done","seed":seed,"repeat":rep,"x_MTy":pa,"x_MTy_minus_y":qa,"wall_s":row["wall_s"]}));
                runs.push(row);
            }
            let (pa, qa) = (runs[0]["x_MTy"].as_f64().unwrap(), runs[0]["x_MTy_minus_y"].as_f64().unwrap());
            let (rp, rq) = (rel(pt, pa), rel(qt, qa));
            let pass = (rp <= DOT_F32 || rq <= DOT_F32) && bad_t == 0;
            dots.push(json!({"seed":seed,"y_Mx":pt,"x_MTy":pa,"rel_plain":rp,"y_Mx_minus_x":qt,"x_MTy_minus_y":qa,"rel_aligned":rq,
                             "pass":pass,"tangent_wall_s":t_tl,"reverse":runs}));
            log(json!({"event":"window_dot","seed":seed,"rel_plain":rp,"rel_aligned":rq,"pass":pass}));
            receipt["dots"] = json!(dots);
            fs::write(out.join("window.json"), serde_json::to_vec_pretty(&receipt).unwrap()).map_err(|e| e.to_string())?;
        }
        receipt["per_step_forward"] = json!(per_step);
        if let Some(p) = crate::peer::get() { receipt["peer"] = p.stats_json(); log(json!({"event":"peer_stats","stats":receipt["peer"]})); }
        // 3. Zero direction: tangent from x = 0 over the whole window.
        if opt(&a, "zero").map(|v| v != "0").unwrap_or(true) {
            let t0 = Instant::now();
            let d = w.load(&bounds[0])?; w.restore(&mut ex, &mut frame, &bounds[0], &d)?; drop(d);
            w.zero(&*cuda)?;
            if let Some(p) = crate::peer::get() { p.begin_tl(); }
            for s in 1..=steps { w.forward_step(&mut ex, &plan, &base, &mut frame, &refs, s, true)?; }
            let nz = w.nonzero(&*cuda)?;
            let nz = match crate::peer::get() { Some(p) => p.reduce_sum("zero", &[nz as f64])?[0] as u64, None => nz };
            receipt["zero_direction"] = json!({"nonzero_words":nz,"pass":nz == 0,"wall_s":t0.elapsed().as_secs_f64()});
            log(json!({"event":"window_zero_direction","nonzero_words":nz}));
        }
        // 3b. Objective reverse (--objective): y = the objective's sparse weights at the window end, one reverse, the
        // gradient dumped for the listed allocations; <x', dJ/dx0> for the Taylor check.
        let obj_terms: Vec<(u64, usize, f32)> = objs.first().map(|o| o.2.clone()).unwrap_or_default();
        if fd_only {
            taylor_gx = if param_fd.is_some() || fwd_only { None } else { Some(f64::NAN) };
            receipt["objective"] = json!({"files":objs.iter().map(|o| o.0.clone()).collect::<Vec<_>>(),"names":objs.iter().map(|o| o.1.clone()).collect::<Vec<_>>(),
                "name":objs[0].1,"note":"--fd-only 1: no reverse; forward difference quotients only"});
        }
        if let (Some((pth, oname, _)), false) = (objs.first(), fd_only) {
            let pth = pth.as_str();
            let doc = json!({"name": oname});
            let tr = Instant::now();
            w.zero(&*cuda)?;
            // j4-repose: a form objective's seed is dJ/dx_N at the base window-end state (float64, rounded once).
            let mut form_j0 = Value::Null;
            let seed_w: Vec<f32> = if let Some(f) = &forms[0] {
                let p = base_words(&bounds[bounds.len() - 1], dn.as_deref().ok_or("window-end state not loaded")?, &obj_terms)?;
                let (j0, g) = f.eval(&p);
                form_j0 = json!({"J0":j0,"seed_max_abs":g.iter().fold(0f64, |m, x| m.max(x.abs())),"seed_nonzero":g.iter().filter(|x| **x != 0.0).count(),
                                 "form":{"t_db":f.t_db,"floor_db":f.floor_db,"floor_width_db":f.floor_w_db,"out":if f.ze { "ze" } else { "dbz" },"groups":f.groups.len()}});
                g.iter().map(|&x| x as f32).collect()
            } else { obj_terms.iter().map(|t| t.2).collect() };
            for (&(al, wd, _), wt) in obj_terms.iter().zip(&seed_w) {
                let el = *w.base_el.get(&al).ok_or_else(|| format!("objective allocation {al} has no shadow"))?;
                dev::upload(&*cuda, w.sh(el + wd as i64), &wt.to_le_bytes())?;
            }
            w.entry_ms = 0.0;
            let mut pfams: Vec<String> = w.params.iter().map(|p| p.fam.clone()).collect();
            pfams.sort(); pfams.dedup();
            for f in &pfams { w.native.as_ref().unwrap().par_zero_bar(&*cuda, f)?; }
            let rows: Vec<Value> = if let (Some(r), Some(c)) = (&rv, ckpts) {
                let (r0, t0s) = (w.recompute_steps, w.recompute_s);
                let rows = w.sweep_revolve(&mut ex, &plan, &base, &mut frame, &refs, &flats, &bounds[0], &bounds[bounds.len() - 1], &mut ck, &bdir,
                                           r, c, anchors, json!({"sweep":"objective"}))?;
                sweeps.push(json!({"sweep":"objective","recompute_steps":w.recompute_steps - r0,"recompute_s":w.recompute_s - t0s,"wall_s":tr.elapsed().as_secs_f64()}));
                rows.iter().map(|r| json!({"step":r["step"],"wall_s":r["wall_s"]})).collect()
            } else {
            let mut cur_data = w.load(&bounds[bounds.len() - 1])?;
            let mut rows = Vec::new();
            for s in (1..=steps).rev() {
                let prev_data = w.load_at(&bounds, s - 1)?;
                if s >= 2 { w.prefetch_load(&bounds, s - 2)?; }
                w.stage_blobs(&mut ex, &base, flat_blobs(&flats[s - 1]), (s >= 2).then(|| flat_blobs(&flats[s - 2])))?;
                let r = w.reverse_step(&mut ex, &base, &mut frame, s, &flats[s - 1], &bounds[s - 1], &prev_data, &bounds[s], &cur_data, anchors)?;
                rows.push(json!({"step":s,"wall_s":r["wall_s"]}));
                cur_data = prev_data;
            }
            drop(cur_data);
            rows
            };
            let mut dumps = Vec::new();
            for id in opt(&a, "dump-grad").unwrap_or("").split(',').filter_map(|x| x.trim().parse::<u64>().ok()) {
                let (Some(&el), Some(sh)) = (w.base_el.get(&id), w.shadows.get(&id)) else { continue };
                let bytes = dev::download(&*cuda, w.sh(el), sh.bytes)?;
                let p = out.join(format!("grad-{id}.f32"));
                fs::write(&p, &bytes).map_err(|e| e.to_string())?;
                let v: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                let nz = v.iter().filter(|x| **x != 0.0).count();
                let mx = v.iter().fold(0f32, |m, x| m.max(x.abs()));
                dumps.push(json!({"allocation":id,"file":p.display().to_string(),"words":v.len(),"nonzero":nz,"max_abs":mx}));
            }
            let mut gxo = Value::Null;
            if !taylor.is_empty() {
                let gx = if let Some((_, t)) = &tdir { w.sparse_dot(&*cuda, t)? } else {
                    w.taylor_mask = true;
                    let (gx, _, _) = w.dots(&*cuda, sx(seeds[0]), xs!(), sx(seeds[0]), xs!())?;
                    w.taylor_mask = false;
                    gx
                };
                taylor_gx = Some(gx);
                gxo = json!(gx);
            }
            receipt["objective"] = json!({"file":pth,"name":doc["name"],"terms":obj_terms.len(),"reverse_wall_s":tr.elapsed().as_secs_f64(),
                "adjoint_entry_s":w.entry_ms / 1e3,"steps":rows,"grad_dumps":dumps,"x_dJdx0_masked":gxo,"form":form_j0});
            if let (Some(ymx), Some(gx)) = (receipt["tangent_dir"]["w_Mx"].as_f64(), gxo.as_f64()) {
                receipt["tangent_dir"]["x_MTw"] = json!(gx);
                receipt["tangent_dir"]["rel"] = json!(rel(ymx, gx));
            }
            if !w.params.is_empty() {
                // lane rung3-params: dJ/dp_k = sum over columns (index order, float64) of the REV sweeps' leaf adjoints.
                let mut bars: HashMap<String, Vec<f64>> = HashMap::new();
                for f in &pfams { bars.insert(f.clone(), w.native.as_ref().unwrap().par_bar(&*cuda, f)?); }
                let mut grads = Vec::new();
                let mut adj_dot = 0f64;
                for (k, p) in w.params.iter().enumerate() {
                    let col = &bars[&p.fam][p.slot * 4096..(p.slot + 1) * 4096];
                    let g: f64 = col.iter().fold(0.0, |s, x| s + x);
                    let bytes: Vec<u8> = col.iter().flat_map(|x| x.to_le_bytes()).collect();
                    // lane rung3-18: AD_PARAM_COLS_DIR (diagnostic, default off) writes each parameter's per-column cotangents (float64, index order) for the maps.
                    if let Some(dd) = std::env::var_os("AD_PARAM_COLS_DIR") { fs::write(Path::new(&dd).join(format!("pcol-o{}-{}.f64", 0, p.name)), &bytes).map_err(|e| e.to_string())?; }
                    if let Some(c) = &param_tl { adj_dot += c.get(k).copied().unwrap_or(0.0) * p.p0 as f64 * g; }
                    grads.push(json!({"name":p.name,"family":p.fam,"slot":p.slot,"p0":p.p0,"dJdp":g,"dJdlnp":g * p.p0 as f64,
                        "nonzero_columns":col.iter().filter(|x| **x != 0.0).count(),"columns_sha256":result_sha256(&bytes),
                        "max_abs_column":col.iter().fold(0f64, |m, x| m.max(x.abs()))}));
                }
                receipt["param_gradients"] = json!(grads);
                receipt["objective_result_sha256"] = json!(result_sha256(&w.download_all(&*cuda)?));
                if let Some(tl) = receipt["param_tl"]["tl_dJ"].as_f64() {
                    receipt["param_dot"] = json!({"tl_dJ":tl,"adjoint_sum_c_dJdlnp":adj_dot,"rel":rel(tl, adj_dot),"pass":rel(tl, adj_dot) <= DOT_F32});
                }
                log(json!({"event":"param_gradients","rows":receipt["param_gradients"],"dot":receipt["param_dot"]}));
            }
            // lane rung3-params --grad-all 1: one more reverse per further objective (same trajectory), parameter
            // gradients and the parameter dot against that objective's TL value.
            if !w.params.is_empty() && opt(&a, "grad-all") == Some("1") {
                let mut more = Vec::new();
                for (oi, (opth, oname2, terms)) in objs.iter().enumerate().skip(1) {
                    let tr2 = Instant::now();
                    w.zero(&*cuda)?;
                    let terms: Vec<(u64, usize, f32)> = match &forms[oi] { Some(f) => {
                        let g = f.eval(&base_words(&bounds[bounds.len() - 1], dn.as_deref().ok_or("window-end state not loaded")?, terms)?).1;
                        terms.iter().zip(&g).map(|(t, &x)| (t.0, t.1, x as f32)).collect() }, None => terms.clone() };
                    for &(al, wd, wt) in &terms {
                        let el = *w.base_el.get(&al).ok_or_else(|| format!("objective allocation {al} has no shadow"))?;
                        dev::upload(&*cuda, w.sh(el + wd as i64), &wt.to_le_bytes())?;
                    }
                    for f in &pfams { w.native.as_ref().unwrap().par_zero_bar(&*cuda, f)?; }
                    let mut cur_data = w.load(&bounds[bounds.len() - 1])?;
                    for s in (1..=steps).rev() {
                        let prev_data = w.load_at(&bounds, s - 1)?;
                        if s >= 2 { w.prefetch_load(&bounds, s - 2)?; }
                        w.stage_blobs(&mut ex, &base, flat_blobs(&flats[s - 1]), (s >= 2).then(|| flat_blobs(&flats[s - 2])))?;
                        w.reverse_step(&mut ex, &base, &mut frame, s, &flats[s - 1], &bounds[s - 1], &prev_data, &bounds[s], &cur_data, anchors)?;
                        cur_data = prev_data;
                    }
                    drop(cur_data);
                    let mut bars: HashMap<String, Vec<f64>> = HashMap::new();
                    for f in &pfams { bars.insert(f.clone(), w.native.as_ref().unwrap().par_bar(&*cuda, f)?); }
                    let mut grads = Vec::new();
                    let mut adj_dot = 0f64;
                    for (k, p) in w.params.iter().enumerate() {
                        let col = &bars[&p.fam][p.slot * 4096..(p.slot + 1) * 4096];
                        let g: f64 = col.iter().fold(0.0, |s, x| s + x);
                        let bytes: Vec<u8> = col.iter().flat_map(|x| x.to_le_bytes()).collect();
                        if let Some(dd) = std::env::var_os("AD_PARAM_COLS_DIR") { fs::write(Path::new(&dd).join(format!("pcol-o{}-{}.f64", oi, p.name)), &bytes).map_err(|e| e.to_string())?; }
                        if let Some(c) = &param_tl { adj_dot += c.get(k).copied().unwrap_or(0.0) * p.p0 as f64 * g; }
                        grads.push(json!({"name":p.name,"family":p.fam,"slot":p.slot,"p0":p.p0,"dJdp":g,"dJdlnp":g * p.p0 as f64,
                            "nonzero_columns":col.iter().filter(|x| **x != 0.0).count(),"columns_sha256":result_sha256(&bytes)}));
                    }
                    let result = w.download_all(&*cuda)?;
                    let mut row = json!({"file":opth,"name":oname2,"param_gradients":grads,"reverse_wall_s":tr2.elapsed().as_secs_f64(),
                                         "result_sha256":result_sha256(&result)});
                    if let Some(tl) = receipt["param_tl"]["all"][oi]["tl_dJ"].as_f64() {
                        row["param_dot"] = json!({"tl_dJ":tl,"adjoint_sum_c_dJdlnp":adj_dot,"rel":rel(tl, adj_dot),"pass":rel(tl, adj_dot) <= DOT_F32});
                    }
                    log(json!({"event":"param_gradients_more","row":row}));
                    more.push(row);
                    receipt["param_gradients_more"] = json!(more);
                    fs::write(out.join("window.json"), serde_json::to_vec_pretty(&receipt).unwrap()).map_err(|e| e.to_string())?;
                }
            }
            log(json!({"event":"window_objective_reverse","wall_s":tr.elapsed().as_secs_f64()}));
            fs::write(out.join("window.json"), serde_json::to_vec_pretty(&receipt).unwrap()).map_err(|e| e.to_string())?;
        }
        // 4p. lane rung3-params: centred differences of the unchecked nonlinear forward in one parameter,
        // p = p0 (1 +- h) rounded to float32; quotient (J+ - J-) / ((p+ - p-) / p0) = dJ/d ln p for every objective.
        if let (Some(pname), true) = (&param_fd, fd_only) {
          let end = w.load(&bounds[bounds.len() - 1])?;
          let mut all_fd = Vec::new();
          for pname in pname.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let pk = w.params.iter().position(|p| p.name == pname).ok_or_else(|| format!("--param-fd {pname}: not in --params"))?;
            let p0 = w.params[pk].p0;
            let mut rows = Vec::new();
            for &h in &taylor {
                let mut sides = Vec::new();
                for sgn in [1.0f64, -1.0] {
                    let t0 = Instant::now();
                    let d = w.load(&bounds[0])?; w.restore(&mut ex, &mut frame, &bounds[0], &d)?; drop(d);
                    let pv = (p0 as f64 * (1.0 + sgn * h)) as f32;
                    w.par_set(&*cuda, Some((pk, pv)), None)?;
                    let mut err = None;
                    for s in 1..=steps {
                        if let Err(e) = w.forward_step_c(&mut ex, &plan, &base, &mut frame, None, s, false) { err = Some(format!("step {s}: {e}")); break; }
                    }
                    w.par_set(&*cuda, None, None)?;
                    let mut djs = Vec::new();
                    if err.is_none() { for (_, _, terms) in &objs { djs.push(obj_dj_terms(&ex, &bounds[bounds.len() - 1], &end, terms)?); } }
                    log(json!({"event":"param_fd_side","param":pname,"h":h,"sign":sgn,"p":pv,"dJ":djs.iter().map(|x| x.0).collect::<Vec<_>>(),"error":err,"wall_s":t0.elapsed().as_secs_f64()}));
                    sides.push((pv, djs, err, t0.elapsed().as_secs_f64()));
                }
                let (pp, pm) = (sides[0].0 as f64, sides[1].0 as f64);
                let dlnp = (pp - pm) / p0 as f64;
                let objr: Vec<Value> = objs.iter().enumerate().map(|(i, (pth, name, _))| {
                    let (jp, jm) = (sides[0].1.get(i).map(|x| x.0), sides[1].1.get(i).map(|x| x.0));
                    json!({"file":pth,"name":name,"dJ_plus":jp,"dJ_minus":jm,"changed_plus":sides[0].1.get(i).map(|x| x.1),"changed_minus":sides[1].1.get(i).map(|x| x.1),
                           "centred_dJdlnp":jp.zip(jm).map(|(a, b)| (a - b) / dlnp),
                           "plus_dJdlnp":jp.map(|a| a / ((pp - p0 as f64) / p0 as f64)),"minus_dJdlnp":jm.map(|b| b / ((pm - p0 as f64) / p0 as f64))})
                }).collect();
                let row = json!({"h":h,"p_plus":pp,"p_minus":pm,"dlnp":dlnp,"objectives":objr,"errors":[sides[0].2, sides[1].2],"wall_s":[sides[0].3, sides[1].3]});
                log(json!({"event":"param_fd","row":row}));
                rows.push(row);
                let mut cur = all_fd.clone();
                cur.push(json!({"param":pname,"p0":p0,"rows":rows}));
                receipt["param_fd"] = json!(cur);
                fs::write(out.join("window.json"), serde_json::to_vec_pretty(&receipt).unwrap()).map_err(|e| e.to_string())?;
            }
            let mut conv = Vec::new();
            for (i, (_, name, _)) in objs.iter().enumerate() {
                let q: Vec<f64> = rows.iter().filter_map(|r| r["objectives"][i]["centred_dJdlnp"].as_f64()).collect();
                let worst = q.windows(2).map(|p| ((p[1] - p[0]) / p[1]).abs()).fold(0f64, f64::max);
                conv.push(json!({"name":name,"centred_dJdlnp":q,"max_rel_change":worst}));
            }
            all_fd.push(json!({"param":pname,"p0":p0,"rows":rows,"convergence":conv}));
            receipt["param_fd"] = json!(all_fd);
            fs::write(out.join("window.json"), serde_json::to_vec_pretty(&receipt).unwrap()).map_err(|e| e.to_string())?;
          }
        }
        // 4. Taylor check of J(x0) (the --objective J, else <y, x_N> with seed 1's y) along seed 1's masked direction x',
        // on the unchecked forward.
        if let Some(gx) = taylor_gx {
            let s0 = seeds[0];
            // S04 (taylor-row-host): the window-end and boundary-0 bytes are the ones the tangent pass already holds
            // (dn, d0: the same files, written once), not a 3.4 GB re-read per row.
            let end_own; let d0_own;
            let end: &[u8] = match dn.as_deref() { Some(e) => e, None => { end_own = w.load(&bounds[bounds.len() - 1])?; &end_own } };
            let d0b: &[u8] = match d0.as_deref() { Some(e) => e, None => { d0_own = w.load(&bounds[0])?; &d0_own } };
            let mut rows = Vec::new();
            for &h in &taylor {
                let t0 = Instant::now();
                let mut ph: BTreeMap<&str, f64> = BTreeMap::new();
                let (dg0, rs0) = (ex.digest_s, w.restore_s);
                w.restore(&mut ex, &mut frame, &bounds[0], d0b)?;
                let tp = Instant::now();
                if let Some((_, t)) = &tdir { w.perturb_terms(&ex, h, t)?; } else {
                    w.taylor_mask = true;
                    w.perturb(&ex, h, sx(s0), xs!())?;
                    w.taylor_mask = false;
                }
                ph.insert("perturb_s", tp.elapsed().as_secs_f64());
                let mut fwd_err = None;
                let (mut rg_step, mut rg_field) = (Vec::new(), vec![0u64; regime.len()]);
                let (tf, mut rg_s) = (Instant::now(), 0f64);
                for s in 1..=steps {
                    if let Err(e) = w.forward_step_c(&mut ex, &plan, &base, &mut frame, None, s, false) { fwd_err = Some(format!("step {s}: {e}")); break; }
                    if !regime.is_empty() {
                        let tr = Instant::now();
                        let d = regime_diff(&regime_base[s - 1], &read_regime(&ex)?);
                        for (f, n) in rg_field.iter_mut().zip(&d) { *f += n; }
                        rg_step.push(d.iter().sum::<u64>());
                        rg_s += tr.elapsed().as_secs_f64();
                    }
                }
                ph.insert("restore_s", w.restore_s - rs0);
                ph.insert("forward_s", tf.elapsed().as_secs_f64() - rg_s);
                ph.insert("forward_digest_s", ex.digest_s - dg0);
                ph.insert("regime_s", rg_s);
                let td = Instant::now();
                // lane runner-dump: the perturbed window-end fields, and (x_N(h) - x_N(0)) / h scored against the tangent
                let mut row_dumps = Vec::new();
                if fwd_err.is_none() && !dump_state.is_empty() {
                    dev::sync(&ex.cuda)?;
                    // S04: the downloads on this thread (the CUDA context's), then every field scored and written on its
                    // own core. Each field's sums keep their word order, so every number is the one the serial loop gave.
                    let mut qs = Vec::with_capacity(dump_state.len());
                    for (i, &(id, off, _)) in dump_state.iter().enumerate() {
                        let al = ex.allocations.get(&id).ok_or_else(|| format!("--dump-state allocation {id} not live at the window end"))?;
                        qs.push(dev::download(&ex.cuda, al.address + off as u64, state_base[i].len())?);
                    }
                    let nrow = rows.len();
                    let (outp, sbase, stl, resid, dname) = (&out, &state_base, &state_tl, &ex.residue, &dump_name);
                    let scored: Vec<Result<Value, String>> = std::thread::scope(|sc| {
                        let hs: Vec<_> = dump_state.iter().copied().enumerate().zip(qs.iter()).map(|((i, (id, off, n)), q)| sc.spawn(move || -> Result<Value, String> {
                        let (out, state_base, state_tl) = (outp, sbase, stl);
                        let mut r = write_state(out, dname(&format!("state-h{nrow}"), (id, off, n)), q)?;
                        r["allocation"] = json!(id); r["byte_offset"] = json!(off); r["h"] = json!(h);
                        let ranges = resid.get(&(steps, id)).map(|x| x.0.clone()).unwrap_or_default();
                        let (mut changed, mut dd, mut tt, mut ft, mut ff, mut masked) = (0u64, 0f64, 0f64, 0f64, 0f64, 0u64);
                        let tl = state_tl.get(i);
                        let mut fdb = vec![0u8; q.len()];
                        for (w_, (a_, b_)) in q.chunks_exact(4).zip(state_base[i].chunks_exact(4)).enumerate() {
                            let at = off + 4 * w_;
                            if ranges.iter().any(|&(ro, rl)| at >= ro && at < ro + rl) { masked += 1; continue; }
                            let fd = if a_ == b_ { 0.0 } else { changed += 1; (f32::from_le_bytes(a_.try_into().unwrap()) as f64 - f32::from_le_bytes(b_.try_into().unwrap()) as f64) / h };
                            fdb[4 * w_..4 * w_ + 4].copy_from_slice(&(fd as f32).to_le_bytes());
                            ff += fd * fd;
                            if let Some(t) = tl {
                                let tv = f32::from_le_bytes(t[4 * w_..4 * w_ + 4].try_into().unwrap()) as f64;
                                dd += (fd - tv) * (fd - tv); tt += tv * tv; ft += fd * tv;
                            }
                        }
                        r["changed_words"] = json!(changed); r["residue_words_excluded"] = json!(masked); r["fd_l2"] = json!(ff.sqrt());
                        let fname = if n.is_some() { format!("fd-{id}-{off}-h{h}.f32") } else { format!("fd-{id}-h{h}.f32") };
                        r["fd"] = write_state(&out, fname, &fdb)?;
                        if tl.is_some() {
                            r["tl_l2"] = json!(tt.sqrt());
                            r["fd_minus_tl_rel_l2"] = json!(if tt > 0.0 { (dd / tt).sqrt() } else { f64::NAN });
                            r["fd_tl_cos"] = json!(if tt > 0.0 && ff > 0.0 { ft / (tt * ff).sqrt() } else { f64::NAN });
                        }
                        Ok(r)
                        })).collect();
                        hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("dump scorer panicked".into()))).collect()
                    });
                    for r in scored { row_dumps.push(r?); }
                    state_dumps["rows"].as_array_mut().unwrap().push(json!({"h":h,"fields":row_dumps.clone()}));
                    receipt["state_dumps"] = state_dumps.clone();
                }
                ph.insert("dumps_s", td.elapsed().as_secs_f64());
                let to = Instant::now();
                // every objective's J(h) - J(0) on the perturbed window end
                let mut all_obj = Vec::new();
                if fwd_err.is_none() {
                    dev::sync(&ex.cuda)?;
                    for (i, (pth, name, terms)) in objs.iter().enumerate() {
                        let (q, _) = live_words(&ex, terms)?;
                        let dj = obj_dj(i, &q, &base_words(&bounds[bounds.len() - 1], &end, terms)?);
                        all_obj.push(json!({"file":pth,"name":name,"dJ":dj,"fd_quotient":dj / h}));
                    }
                }
                ph.insert("objectives_s", to.elapsed().as_secs_f64());
                let row = match fwd_err {
                    Some(e) => json!({"h":h,"error":e}),
                    None => {
                        let (dj, bad, changed) = if obj_terms.is_empty() { w.taylor_diff(&ex, &bounds[bounds.len() - 1], &end, steps, sy(s0), ys!())? } else {
                            // J(h) - J(0) on the objective words: live value minus the stored window-end value.
                            dev::sync(&ex.cuda)?;
                            let (q, qb) = live_words(&ex, &obj_terms)?;
                            let p = base_words(&bounds[bounds.len() - 1], &end, &obj_terms)?;
                            let bad = q.iter().filter(|x| !x.is_finite()).count() as u64;
                            let ch = qb.iter().zip(&p).filter(|(b, p)| **b != (**p as f32).to_bits()).count() as u64;
                            (obj_dj(0, &q, &p), bad, ch)
                        };
                        ph.insert("final_diff_s", to.elapsed().as_secs_f64() - ph["objectives_s"]);
                        let lin = h * gx;
                        let ratio = dj / lin;
                        let mut r = json!({"h":h,"dJ":dj,"fd_quotient":dj / h,"h_x_MTy":lin,"ratio":ratio,"err":(ratio - 1.0).abs(),"nonfinite":bad,"changed_words":changed,
                               "objectives":all_obj,"wall_s":t0.elapsed().as_secs_f64()});
                        r["phases_s"] = json!(ph);
                        if !row_dumps.is_empty() {
                            r["state_fields"] = json!(row_dumps.iter().map(|d| json!({"allocation":d["allocation"],"byte_offset":d["byte_offset"],"changed_words":d["changed_words"],
                                "fd_minus_tl_rel_l2":d["fd_minus_tl_rel_l2"],"fd_tl_cos":d["fd_tl_cos"]})).collect::<Vec<_>>());
                        }
                        if !regime.is_empty() {
                            r["regime_changes"] = json!(rg_field.iter().sum::<u64>());
                            r["regime_by_field"] = json!(regime.iter().zip(&rg_field).map(|(f, n)| (f.0.clone(), *n)).collect::<BTreeMap<_, _>>());
                            r["regime_by_step"] = json!(rg_step);
                        }
                        r
                    },
                };
                log(json!({"event":"window_taylor","row":row}));
                rows.push(row);
                receipt["taylor"] = json!({"objective":if obj_terms.is_empty() { json!("J(x0) = <y, x_N>, y = seed-1 relative direction at the window end; declared TEMPO residue words excluded") } else { receipt["objective"]["name"].clone() },
                    "direction":if let Some((n, t)) = &tdir { json!({"fixed":opt(&a, "taylor-dir"),"name":n,"terms":t.len()}) } else { json!("seed-1 relative direction x, address-bearing and fixed allocations masked") },
                    "x_MTy_masked":gx,"fd_only":fd_only,"rows":rows});
                fs::write(out.join("window.json"), serde_json::to_vec_pretty(&receipt).unwrap()).map_err(|e| e.to_string())?;
            }
            let best = receipt["taylor"]["rows"].as_array().into_iter().flatten().filter_map(|r| r["err"].as_f64()).fold(f64::INFINITY, f64::min);
            receipt["taylor"]["best_err"] = json!(best);
            // forward quotient convergence per objective: largest relative change of dJ/h between consecutive h
            let rows = receipt["taylor"]["rows"].as_array().cloned().unwrap_or_default();
            let mut conv = Vec::new();
            for (i, (_, name, _)) in objs.iter().enumerate() {
                let q: Vec<f64> = rows.iter().filter_map(|r| r["objectives"][i]["fd_quotient"].as_f64()).collect();
                let worst = q.windows(2).map(|p| ((p[1] - p[0]) / p[1]).abs()).fold(0f64, f64::max);
                conv.push(json!({"name":name,"fd_quotients":q,"max_rel_change":worst}));
            }
            receipt["taylor"]["fd_convergence"] = json!(conv);
        }
        let refused: BTreeMap<&String, &u64> = w.labels.iter().filter(|(k, _)| k.starts_with("REFUSED:")).collect();
        receipt["commands_by_rule"] = json!(w.labels);
        receipt["refused"] = json!(refused);
        receipt["held_upload_dropped_cotangent_sq"] = json!(w.held_dropped);
        if w.tempo.is_some() {
            receipt["tempo_live"] = json!({"stats":w.tempo_stats,"tempo_adw":w.tempo.as_ref().map(|t| t.0.image_sha256.clone()),
                "tempo_adw_tlf":w.tempo.as_ref().and_then(|t| t.0.tlf.as_ref().map(|x| x.image_sha256.clone())),
                "tlf_stats":w.tempo.as_ref().map(|t| { let z = t.0.tlf_stats.get(); json!({"tlf_launches":z[0],"columns_rerun_larger_cap":z[1],"fallbacks_segmented":z[2],"check_differing_words":z[3]}) }),
                "treatment":"tempo_warp_steps live (--tempo-live 1): the L5 tempo_adw whole-launch derivative (warp shape, per-substep checkpoints, float64 executed-branch tape) on float64 copies of the window's float32 shadows, rounded back as the L2 rows are. Branches (min/max/clamps, limiter and threshold tests) are the forward's own executed branches, frozen from the trajectory: the derivative of the piecewise path the forward took."});
        }
        receipt["boundary_store"] = json!({"dir":bdir.display().to_string(),"files":bounds.len(),
            "bytes_each":bounds.last().map(|b| b.index.iter().map(|x| x.2).sum::<usize>()).unwrap_or(0),
            "file_bytes_peak":w.bs_peak,"file_bytes_written":w.bs_written,"files_peak":w.bs_files_peak,
            "window_end_file_bytes":bounds.last().map(|b| b.file_bytes).unwrap_or(0),"load_s":w.load_s,"store_s":w.store_s,
            "ram":{"budget_bytes":w.bs_ram_budget,"boundaries":w.bs_ram_count,"peak_bytes":w.bs_ram_peak},
            "restores_direct_from_page_locked":w.restore_direct,"boundary_buffers_made_ahead":w.pin_ahead_hits,"boundary_buffers_made_at_pass_start":w.pin_q_made,"anchors_ahead":{"plan":w.anchor_ahead_info,"taken":w.anchor_ahead_taken,"wait_s":w.anchor_ahead_wait_s,"live_peak_bytes":ex.live_peak},"staging_allocs_total":crate::staging_allocs(),"pin_page_policy":crate::pinned::page_policy()});
        if ckpts.is_some() {
            receipt["checkpointing"]["sweeps"] = json!(sweeps);
            receipt["checkpointing"]["recompute_steps"] = json!(w.recompute_steps);
            receipt["checkpointing"]["recompute_s"] = json!(w.recompute_s);
            receipt["checkpointing"]["prefetch_hits"] = json!(w.pf_hits);
        }
        receipt["shadow_bytes"] = json!(w.words * 4);
        receipt["tape_bytes"] = json!(w.tape.bytes);
        {
            let mut by: BTreeMap<String, Vec<Value>> = BTreeMap::new();
            let mut rows: Vec<(&(String, String), &(f64, u64))> = w.fn_ms.iter().collect();
            rows.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap_or(std::cmp::Ordering::Equal));
            for ((pass, f), (ms, n)) in rows { by.entry(pass.clone()).or_default().push(json!([f, (ms * 10.0).round() / 10.0, n])); }
            receipt["entry_ms_by_function"] = json!(by);
        }
        receipt["one_pass"] = json!({"on":w.one_pass,"fallback_steps":w.one_pass_fallbacks,"tape":w.tape_grown});
        receipt["tl_tape_grown"] = json!(w.tl_tape_grown);
        receipt["tempo_arena"] = json!(w.tempo_sized);
        receipt["tape_peak"] = json!(w.tape.peak);
        receipt["scratch_peak"] = json!(w.scratch.peak);
        receipt["wall_s"] = json!(t_start.elapsed().as_secs_f64());
        let all_pass = receipt["dots"].as_array().is_some_and(|d| d.iter().all(|x| x["pass"] == true))
            && receipt["zero_direction"]["pass"] != false
            && receipt["dots"][0]["reverse"].as_array().is_some_and(|r| r.iter().skip(1).all(|x| x["differing_words_vs_first_reverse"] == 0));
        receipt["milestone_a_numbers_pass"] = json!(all_pass);
        fs::write(out.join("window.json"), serde_json::to_vec_pretty(&receipt).unwrap()).map_err(|e| e.to_string())?;
        log(json!({"event":"window_summary","pass":all_pass,"refused_kinds":refused.len(),"wall_s":receipt["wall_s"]}));
        w.join_writer()?;
        if let Some((_, h)) = w.ck_pf.take() { let _ = h.join(); }
        if opt(&a, "keep-bstore") != Some("1") { for b in bounds.iter().chain(ck.values()) { let _ = fs::remove_file(&b.path); } }
    }
    Ok(())
}

#[cfg(test)]
mod j4_form_tests {
    use super::Form;
    use serde_json::json;
    fn fd_check(f: &Form, v: &[f64]) {
        let (_, g) = f.eval(v);
        let gmax = g.iter().fold(0f64, |m, x| m.max(x.abs()));
        for i in 0..v.len() {
            if v[i] == 0.0 { assert_eq!(g[i], 0.0); continue; }
            let h = 1e-5 * v[i].abs();
            let (mut a, mut b) = (v.to_vec(), v.to_vec());
            a[i] += h; b[i] -= h;
            let fd = (f.eval(&a).0 - f.eval(&b).0) / (2.0 * h);
            assert!((fd - g[i]).abs() <= 1e-5 * fd.abs().max(g[i].abs()) + 1e-9 * gmax, "word {i}: fd {fd} grad {}", g[i]);
        }
    }
    #[test]
    fn lse_ze_and_dbz_gradients_match_central_differences() {
        let v = [-35.0, -12.0, 3.5, 21.0, 38.2, 44.0, 41.5, -35.0, 10.0];
        for (out, t, w) in [("ze", 2.0, 2.0), ("dbz", 1.0, 1.0), ("ze", 10.0 / std::f64::consts::LN_10, 0.0), ("dbz", 0.5, 0.0)] {
            let doc = json!({"form":{"t_db":t,"floor_db":-35.0,"floor_width_db":w,"out":out},
                             "groups":[[0.25,[[374,0],[374,1],[374,2],[374,3],[374,4]]],[0.75,[[374,5],[374,6],[374,7],[374,8]]]]});
            let (f, terms) = Form::parse(&doc, "t").unwrap();
            assert_eq!(terms.len(), 9);
            fd_check(&f, &v);
        }
    }
    #[test]
    fn column_sum_temperature_is_linear_ze_sum_and_small_t_approaches_the_max() {
        let v = [10.0, 20.0, 30.0];
        let doc = json!({"form":{"t_db":10.0 / std::f64::consts::LN_10,"out":"ze"},"groups":[[1.0,[[1,0],[1,1],[1,2]]]]});
        let (f, _) = Form::parse(&doc, "t").unwrap();
        let s: f64 = v.iter().map(|x| 10f64.powf(x / 10.0)).sum();
        assert!((f.eval(&v).0 - s).abs() < 1e-9 * s);
        let doc = json!({"form":{"t_db":0.05,"out":"dbz"},"groups":[[1.0,[[1,0],[1,1],[1,2]]]]});
        let (f, _) = Form::parse(&doc, "t").unwrap();
        assert!((f.eval(&v).0 - 30.0).abs() < 1e-3);
        assert!(Form::parse(&json!({"form":{"t_db":0.0},"groups":[]}), "t").is_err());
        // lane s16: misfit J = 0.5 r (c1 v1 + c2 v2 - y)^2, gradient r (Hx - y) c
        let mf = json!({"form":{"kind":"misfit"},"groups":[[0.25,[[1,0,1.0],[1,1,0.5]],3.0],[1.0,[[2,4,1.0]],-1.0]]});
        let (f, terms) = Form::parse(&mf, "t").unwrap();
        assert_eq!(terms.len(), 3);
        let (j, g) = f.eval(&[2.0, 4.0, 1.0]);
        assert!((j - (0.5 * 0.25 * 1.0 + 0.5 * 4.0)).abs() < 1e-12);
        assert!((g[0] - 0.25).abs() < 1e-12 && (g[1] - 0.125).abs() < 1e-12 && (g[2] - 2.0).abs() < 1e-12);
        assert!(Form::parse(&json!({"form":{"kind":"misfit"},"groups":[]}), "t").is_err());
    }
    #[test]
    fn mass_form_gradient_matches_central_differences_and_reduces_to_ze_of_one_species() {
        let doc = json!({"form":{"kind":"mass","t_db":2.0,"z0":10f64.powf(-3.5),"out":"ze"},
                         "groups":[[0.6,[[[1,0,3.63e9,1.75],[1,1,9.8e8,1.75],[1,2,4.33e10,1.75]],[[1,3,3.63e9,1.75],[1,4,9.8e8,1.75],[1,5,4.33e10,1.75]]]],
                                   [0.4,[[[1,6,3.63e9,1.75],[1,7,9.8e8,1.75],[1,8,4.33e10,1.75]]]]]});
        let (f, terms) = Form::parse(&doc, "t").unwrap();
        assert_eq!((terms.len(), f.levels.len(), f.groups.len()), (9, 3, 2));
        fd_check(&f, &[1.2e-3, 2e-4, 5e-5, 3e-4, 0.0, 1e-3, 2e-3, 1e-4, 0.0]);
        let one = json!({"form":{"kind":"mass","t_db":1.0,"z0":1e-30,"out":"ze"},"groups":[[1.0,[[[1,0,3.63e9,1.75]]]]]});
        let (f, _) = Form::parse(&one, "t").unwrap();
        let q: f64 = 1e-3;
        assert!((f.eval(&[q]).0 / (3.63e9 * q.powf(1.75)) - 1.0).abs() < 1e-12);
    }
}

#[cfg(test)]
mod spec_check_tests {
    use super::check_spec_hashes;
    use std::fs;
    #[test]
    fn refuses_stale_spec_record_and_counts_unrecorded() {
        let d = std::env::temp_dir().join(format!("ad-spec-check-{}", std::process::id()));
        fs::create_dir_all(&d).unwrap();
        let (sp, sp2) = (d.join("a.json"), d.join("b.json"));
        fs::write(&sp, b"{\"launch_abi\":{}}").unwrap();
        fs::write(&sp2, b"{}").unwrap();
        let good = crate::exec::digest(&fs::read(&sp).unwrap());
        let k = d.join("kernels.json");
        let set = |sha: &str| serde_json::json!({"kernels":[
            {"function":"a","derivative":{"spec":{"path":sp.display().to_string(),"sha256":sha}}},
            {"function":"b","derivative":{"spec":{"path":sp2.display().to_string()}}},
            {"function":"c","class":"held"}]});
        fs::write(&k, serde_json::to_vec(&set(&good)).unwrap()).unwrap();
        let r = check_spec_hashes(&k).unwrap();
        assert_eq!((r["specs_checked"].as_u64(), r["specs_unrecorded"].as_u64()), (Some(1), Some(1)));
        fs::write(&sp, b"{\"launch_abi\": {}}").unwrap();
        let e = check_spec_hashes(&k).unwrap_err();
        assert!(e.contains("a: spec sha256") && e.contains("1 spec file(s)"), "{e}");
        let _ = fs::remove_dir_all(&d);
    }
}

#[cfg(test)]
mod s02_checkpoint_tests {
    use super::*;
    /// Brute force: simulate the schedule and count forward steps; every step reversed once, in order, slots <= c.
    fn simulate(rv: &Revolve, lo: usize, hi: usize, k: usize, live: &mut Vec<usize>, peak: &mut usize, order: &mut Vec<usize>) -> u64 {
        if hi - lo == 1 { order.push(hi); return 0; }
        if k == 0 { let mut c = 0; for s in (lo + 1..=hi).rev() { c += (s - 1 - lo) as u64; order.push(s); } return c; }
        let m = rv.split(hi - lo, k);
        live.push(lo + m); *peak = (*peak).max(live.len());
        let a = m as u64 + simulate(rv, lo + m, hi, k - 1, live, peak, order);
        live.pop();
        a + simulate(rv, lo, lo + m, k, live, peak, order)
    }
    #[test]
    fn revolve_schedule_is_valid_and_matches_its_cost() {
        for n in 1..=72 { for c in 0..=9 {
            let rv = Revolve::new(n, c);
            let (mut live, mut peak, mut order) = (Vec::new(), 0, Vec::new());
            let cost = if n >= 1 { simulate(&rv, 0, n, c, &mut live, &mut peak, &mut order) } else { 0 };
            assert_eq!(cost, rv.cost[n][c], "n {n} c {c}");
            assert!(peak <= c, "n {n} c {c} peak {peak}");
            assert_eq!(order, (1..=n).rev().collect::<Vec<_>>(), "n {n} c {c}");
            let d = rv.descent(n, c);
            assert!(d.len() <= c && d.windows(2).all(|w| w[0] < w[1]) && d.iter().all(|&x| x > 0 && x < n));
        } }
        // binomial optimum (Griewank): 18 steps, 4 slots -> at most 3 repetitions
        let rv = Revolve::new(18, 4);
        assert!(rv.cost[18][4] <= 3 * 18);
    }
    #[test]
    fn changed_chunk_store_round_trips() {
        let base_index = vec![(1u64, 0usize, 5000usize), (2, 5000, 3000)];
        let mut base = vec![0u8; 8000];
        for (i, b) in base.iter_mut().enumerate() { *b = (i * 7 % 251) as u8; }
        let index = vec![(1u64, 0usize, 5000usize), (2, 5000, 3000), (7, 8000, 2100)];
        let mut data = vec![0u8; 10100];
        data[..8000].copy_from_slice(&base);
        data[17] ^= 1; data[4999] ^= 9; data[5000] = 3; data[9000] = 1; data[10099] = 2;
        let f = delta_encode(&index, &data, &base_index, &base);
        assert!(f.len() < 4 * DCHUNK + 200);
        assert_eq!(delta_decode(&f, &index, &base_index, &base).unwrap(), data);
        let same = delta_encode(&base_index, &base, &base_index, &base);
        assert_eq!(same.len(), 24);
        assert_eq!(delta_decode(&same, &base_index, &base_index, &base).unwrap(), base);
    }
}

/// tape-dedupe-and-pack: AD_TAPE_CENSUS=<dir> (diagnostic only, never in a timed leg: it syncs and downloads every taped
/// byte). For every reverse segment that is walked back it writes, per taped piece (pre-launch copy, post-launch copy,
/// re-copy, or the other bytes a command took), one JSON line with the piece's word statistics, and per 4 KiB block of
/// it (aligned to the source allocation's offsets for copies) one 24-byte record [hash u64, piece u32, block u32,
/// kind u8, 7 pad] in blocks.bin. tools/tape_census.py reads them: duplicate content across launches and allocations,
/// zero, small-integer and 0/1 words, and the copy margins.
pub(crate) struct Census { dir: PathBuf, lines: Vec<String>, blocks: Vec<u8>, piece: u32, step: usize, seg: u32, out_l: Option<fs::File>, out_b: Option<fs::File> }
impl Census {
    fn from_env() -> Option<Census> {
        let d = std::env::var_os("AD_TAPE_CENSUS")?;
        let dir = PathBuf::from(d);
        fs::create_dir_all(&dir).ok()?;
        Some(Census { out_l: fs::File::create(dir.join("pieces.jsonl")).ok(), out_b: fs::File::create(dir.join("blocks.bin")).ok(), dir, lines: Vec::new(), blocks: Vec::new(), piece: 0, step: 0, seg: 0 })
    }
    fn begin_segment(&mut self, step: usize) { self.lines.clear(); self.blocks.clear(); self.step = step; }
    fn end_segment(&mut self) -> Result<(), String> {
        use std::io::Write;
        let _ = &self.dir;
        if let Some(f) = self.out_l.as_mut() { for l in &self.lines { writeln!(f, "{l}").map_err(|e| e.to_string())?; } }
        if let Some(f) = self.out_b.as_mut() { f.write_all(&self.blocks).map_err(|e| e.to_string())?; }
        self.lines.clear(); self.blocks.clear(); self.seg += 1;
        Ok(())
    }
    fn hash(b: &[u8], tag: u64) -> u64 {
        let mut h: u64 = 0x9e37_79b9_7f4a_7c15 ^ tag;
        for c in b.chunks(8) {
            let mut w = [0u8; 8]; w[..c.len()].copy_from_slice(c);
            h = (h.rotate_left(5) ^ u64::from_le_bytes(w)).wrapping_mul(0x517c_c1b7_2722_0a95);
        }
        h ^ (b.len() as u64).wrapping_mul(0xff51_afd7_ed55_8ccd)
    }
    #[allow(clippy::too_many_arguments)]
    unsafe fn piece(&mut self, cuda: &Cuda, cmd: usize, f: &str, kind: u8, alloc: u64, lo0: usize, hi0: usize, lo: usize, hi: usize, at: u64) -> Result<(), String> {
        if hi <= lo { return Ok(()); }
        let data = dev::download(cuda, at, hi - lo)?;
        let (mut zero, mut i8w, mut b01, mut i16w, mut fint) = (0usize, 0usize, 0usize, 0usize, 0usize);
        for c in data.chunks_exact(4) {
            let u = u32::from_le_bytes(c.try_into().unwrap());
            let i = u as i32; let x = f32::from_bits(u);
            if u == 0 { zero += 1; }
            if (0..=255).contains(&i) { i8w += 1; }
            if (-32768..=32767).contains(&i) { i16w += 1; }
            if x == 0.0 || x == 1.0 { b01 += 1; }
            if x.is_finite() && x == x.trunc() && x.abs() <= 32767.0 { fint += 1; }
        }
        // blocks aligned to the source offset (copies) or to the piece start (other bytes)
        const B: usize = 4096;
        let mut o = lo;
        while o < hi {
            let e = ((o / B) + 1) * B;
            let e = e.min(hi);
            let h = Self::hash(&data[o - lo..e - lo], ((o % B) as u64) << 32);
            self.blocks.extend_from_slice(&h.to_le_bytes());
            self.blocks.extend_from_slice(&self.piece.to_le_bytes());
            self.blocks.extend_from_slice(&((o / B) as u32).to_le_bytes());
            self.blocks.push(kind); self.blocks.extend_from_slice(&[0u8; 7]);
            o = e;
        }
        self.lines.push(json!({"piece":self.piece,"step":self.step,"seg":self.seg,"cmd":cmd,"f":f,"kind":kind,"alloc":alloc,"lo0":lo0,"hi0":hi0,"lo":lo,"hi":hi,
            "bytes":hi - lo,"words":(hi - lo) / 4,"zero":zero,"int8":i8w,"int16":i16w,"f01":b01,"fint":fint}).to_string());
        self.piece += 1;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    unsafe fn op(&mut self, cuda: &Cuda, cmd: usize, f: &str, u0: usize, u1: usize, base: u64,
                 copies: &[(u64, usize, usize, usize, usize, u64)], post: &[(u64, usize, usize, u64)]) -> Result<(), String> {
        dev::sync(cuda)?;
        let mut covered = 0usize;
        for &(al, lo0, hi0, lo, hi, at) in copies {
            let kind = if at >> 63 == 1 { 3 } else { 1 };
            self.piece(cuda, cmd, f, kind, al, lo0, hi0, lo, hi, at & !(1u64 << 63))?;
            covered += hi - lo;
        }
        for &(al, lo, hi, at) in post { self.piece(cuda, cmd, f, 2, al, lo, hi, lo, hi, at)?; covered += hi - lo; }
        if copies.is_empty() && post.is_empty() && u1 > u0 {
            self.piece(cuda, cmd, f, 4, 0, 0, u1 - u0, 0, u1 - u0, base + u0 as u64)?;
        }
        let _ = covered;
        Ok(())
    }
}
