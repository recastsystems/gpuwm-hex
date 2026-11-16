// SPDX-License-Identifier: Apache-2.0
//! Physics-on window, lane L1 (ad/window-L1-dispatcher): the composed tangent and the
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
//! * `--objective-reverse 0` (truth, f52; with --tangent-taylor-dir 1, --objective and --taylor; default 1): the
//!   right check. The objective reverse does not run; each Taylor row is scored against the tangent's own
//!   prediction <w, M x> (w = the first objective), so one tangent pass and the real perturbed forwards say what
//!   share of the real change the linear model explains (state_fields: fd_l2, tl_l2, fd_minus_tl_rel_l2, fd_tl_cos).
//! * `AD_HOLD_MCICA=1|census` and `AD_HOLD_OUT=function:arg:unit_bytes:count_arg,..` (truth, f52; default off):
//!   named kernel outputs (the RRTMG McICA seeds) of every perturbed forward (Taylor rows, --param-fd sides) are held
//!   at the base forward's words for the same launch; `census` only counts the words that would differ. The base
//!   forward, the tangent and the reverse are unchanged word for word. See `Hold`.
//! * `--dump-at` in Taylor rows (truth, f52; default off): each perturbed forward writes its words at the named
//!   step boundaries (state-h<row>-s<step>-<id>-<offset>.f32), so the right check reads every lead of a window
//!   from one run. (The unread boundary digest of those forwards is skipped by lane any-mesh's rule.)
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
    let real = fs::read_to_string("/proc/meminfo").ok().and_then(|m| m.lines().find(|l| l.starts_with("MemAvailable"))
        .and_then(|l| l.split_whitespace().nth(1)?.parse::<usize>().ok())).unwrap_or(0) * 1024;
    // mem-oocore: --host-budget-gb bounds every RAM tier (boundaries, anchors, fused-tape chunks); the rest goes to files.
    match crate::spill::host_room() { Some(r) => real.min(r), None => real }
}

/// An optional extension driver over this physics window (`--ext-driver`, default off). Not part of this crate:
/// compiled only where its source file is present (build.rs declares cfg hexad_ext_driver and never sets it).
#[cfg(hexad_ext_driver)]
#[path = "ext_driver.rs"] mod ext_driver;
/// What `--ext-driver` answers in a build that does not carry the driver (every build of this crate).
#[cfg(not(hexad_ext_driver))]
pub(crate) const EXT_DRIVER_ABSENT: &str = "DEFECT: --ext-driver: this build does not carry an extension driver";
/// Lane runner-lib : the libhexad-phys C ABI v2 over this window (cdylib hexad-phys; the
/// ad-binder binary carries it but never calls it). Its counters' category guards below are thread-local writes only.
#[path = "capi.rs"] pub mod capi;
#[path = "live_save.rs"] pub(crate) mod live_save;
#[path = "join_one.rs"] pub(crate) mod join_one;
#[path = "window_vectors.rs"] mod window_vectors;
/// Opt-in live Python replay; no changes to the recorded window route.
#[path = "live_window.rs"] pub mod live_window;
#[path = "live_construct.rs"] pub(crate) mod live_construct;
#[path = "live_capture.rs"] pub(crate) mod live_capture;
#[path = "live_sweep.rs"] pub mod live_sweep;
/// central hunt-3km-slot: the self dot's arithmetic, AD_DOT_SELF_FLOOR and AD_DOT_ROUNDINGS (all diagnostics; valid unchanged).
#[path = "self_dot.rs"] mod self_dot;

const P8_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/p8_rules.fatbin"));
const WIN_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/window_support.fatbin"));
pub const DOT_F32: f64 = 2e-5;
/// S01c: the last dots() call's sum of |a_i S_i| when AD_DOT_ALLOC=1 (f64 bits; 0 otherwise). The window dot's float32
/// rounding scales with it, not with the signed sum, so a dot whose signed sum cancels (|sum| << sum |terms|) reads a
/// large rel from rounding alone; the receipt records rel_terms = |y Mx - x M^T y| / max(sum |terms|) beside rel.
static LAST_ABS_TERMS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// F14 (15 Nov): whether dots() takes the absolute terms sum |a_i S_i|. It does by default, for every form, because the
/// check's verdict is stated in float32 roundings of those terms (dot_verdict, miss_roundings; coordinator 8:10 pm PT 15 Nov:
/// the window check is judged in roundings of the dot). Before, the terms were taken only with AD_DOT_ALLOC=1, so a plain
/// run of the independent form (--direction fieldxy) could only be read on its signed sum, which a random y cancels by luck.
/// AD_DOT_TERMS=0 leaves them out (a timing run only; such a run has no roundings and falls back to rel_aligned).
fn abs_terms_wanted() -> bool {
    std::env::var("AD_DOT_ALLOC").as_deref() == Ok("1") || std::env::var("AD_DOT_TERMS").as_deref() != Ok("0")
}
/// F14 (15 Nov): the miss a dot is judged on: (pass measure, valid measure, measure name).
/// A dot's float32 rounding is bounded by the sum of its absolute terms, not by its signed sum (Higham, Accuracy and
/// Stability, 3.1: |fl(x.y) - x.y| <= gamma_n sum |x_i y_i|). The self form cannot cancel (its terms are its y_Mx), so its
/// rel IS its miss over its terms, as before. A random y cancels by luck of the draw: on the 12 km cut the three seeds of
/// the independent form missed by the same absolute amount (0.14, 0.33, 0.24) on dots that are 1.1, 14.9 and 7.4 per cent
/// of their terms, so rel read 2.8e-05, 4.0e-06, 7.0e-06 and seed 1 alone failed the 2e-5 bar, on the arithmetic the other
/// two passed with. THE BREAKAGE THIS PREVENTS: a random-y check that fails or passes by its draw's cancellation, and a
/// 1e-07 bar on its rel that float32 words cannot meet on any dot cancelling below about 60 per cent (one rounding of the
/// terms is 6e-08 of them). With the terms the random forms are judged on rel_terms = |y_Mx - x_MTy| / max(terms of the
/// two sums): DOT_F32 for pass here, and window.json dots[].bar_rel carries the same number for the program's bar
/// (valid_derivative; check-bar's DotBar reads bar_rel where present, else rel_aligned). For the self form bar_rel IS
/// rel_aligned, so its verdict does not move. Without terms (a rank or parameter dot that has none) the old reading
/// stands: pass on min(rel_plain, rel_aligned), bar on rel_aligned.
fn dot_verdict(self_form: bool, rel_plain: f64, rel_aligned: f64, rel_terms: Option<f64>) -> (f64, f64, &'static str) {
    match (self_form, rel_terms) {
        (false, Some(rt)) if rt.is_finite() => (rt, rt, "rel_terms"),
        _ => (rel_plain.min(rel_aligned), rel_aligned, if self_form { "rel (self form: its terms are its y_Mx)" } else { "rel_aligned (no absolute terms)" }),
    }
}
/// inst (AD_DOT_F64=1): the last dots() call's sum a*S taken again on the host, every product exact in binary64
/// and the sum compensated (Neumaier). The default sum (win_dot2_f32) is ALREADY binary64: each float32 word is widened
/// before the product, 262,144 strided partial sums are added in a fixed order, uncompensated. What is float32 in the
/// check is the data: the direction words, the tangent result, the y = W S of the self dot and the reverse result.
/// window.json dots[].f64 prints both readings side by side.
static LAST_F64: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
fn neumaier(acc: &mut (f64, f64), x: f64) { let t = acc.0 + x; acc.1 += if acc.0.abs() >= x.abs() { (acc.0 - t) + x } else { (x - t) + acc.0 }; acc.0 = t; }
/// inst (AD_PLANT=<kernel>:<argument name or number>:<factor>): a PLANTED one-sided error, to prove a check can
/// fail. In the reverse only, each launch of that kernel has the change its adjoint entries made to that argument's
/// cotangent view scaled by the factor (new = before + factor * (after - before), in binary64, rounded once); the
/// tangent is untouched. THE BREAKAGE THIS PREVENTS: a check or a left-out-memory rule that would read "valid" on a
/// wrong adjoint row. A run with it set is never valid (window.json planted_error; run reads valid false).
static PLANT_HITS: [std::sync::atomic::AtomicU64; 2] = [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];
fn plant_scale(after: &mut [u8], before: &[u8], pf: f64) -> u64 {
    let mut ch = 0u64;
    for (q, p) in after.chunks_exact_mut(4).zip(before.chunks_exact(4)) {
        if &q[..] == p { continue; }
        let (qv, pv) = (f32::from_le_bytes([q[0], q[1], q[2], q[3]]), f32::from_le_bytes([p[0], p[1], p[2], p[3]]));
        let nv = (pv as f64 + pf * (qv as f64 - pv as f64)) as f32;
        if nv != qv { ch += 1; }
        q.copy_from_slice(&nv.to_le_bytes());
    }
    ch
}
fn plant() -> &'static Option<(String, String, f64)> {
    static P: std::sync::OnceLock<Option<(String, String, f64)>> = std::sync::OnceLock::new();
    P.get_or_init(|| { let v = std::env::var("AD_PLANT").ok()?; let mut it = v.splitn(3, ':');
        let (k, a, f) = (it.next()?, it.next()?, it.next()?.parse::<f64>().ok()?); Some((k.to_owned(), a.to_owned(), f)) })
}
/// speed (fold7; asked by a development line checker, 2026-11-11T08:10Z): AD_PLANT_RECOMPUTE=<step>[:<allocation>[:<word>]], a
/// PLANTED wrong re-run step, TEST ONLY. In a checkpointed reverse sweep, the first time step <step> is re-run from a
/// checkpoint (advance), one float32 word of its start state on the card has one mantissa bit flipped (bit 16: the value
/// moves by under 1 %) before the step runs. With no allocation named: the largest allocation that carries a tangent,
/// and from its middle word on the first finite float32 that is not tiny. The recorded forward, the checkpoints and the
/// tangent are untouched. THE BREAKAGE THIS PREVENTS: nothing else in the program can change the model state inside
/// advance, so with checkpoints in host memory no whole run could show that the joined hash of a re-run step (main.rs
/// digest_lazy_end, restore_join) refuses a wrong re-run state. A run with it set is never valid: the receipt carries
/// planted_error from the start, and window.json is written with it at the planted step, so run reads valid false
/// also when the run is then refused. Nothing happens unless it is set.
const PLANT_RC_BIT: u32 = 1 << 16;
const PLANT_RC_SIDE: &str = "the start state of the first re-run of this step in a checkpointed reverse sweep (window.rs advance, before forward_step): one float32 word on the card has bit 16 flipped; the recorded forward, the checkpoints and the tangent are untouched";
static PLANT_RC_OUT: std::sync::Mutex<Option<(PathBuf, Value)>> = std::sync::Mutex::new(None);
static PLANT_RC_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) fn plant_recompute_parse(v: &str) -> Option<(usize, Option<u64>, Option<usize>)> {
    let mut it = v.splitn(3, ':');
    let s = it.next()?.parse::<usize>().ok().filter(|s| *s >= 1)?;
    let al = match it.next() { Some(x) => Some(x.parse::<u64>().ok()?), None => None };
    let wd = match it.next() { Some(x) => Some(x.parse::<usize>().ok()?), None => None };
    Some((s, al, wd))
}
fn plant_recompute() -> &'static Option<(usize, Option<u64>, Option<usize>)> {
    static P: std::sync::OnceLock<Option<(usize, Option<u64>, Option<usize>)>> = std::sync::OnceLock::new();
    P.get_or_init(|| plant_recompute_parse(&std::env::var("AD_PLANT_RECOMPUTE").ok()?))
}
/// The word AD_PLANT_RECOMPUTE changes among little-endian float32 `words`: the first finite one of magnitude 1e-30 or
/// more (a zero or a tiny word would take a change the step rounds away), with its bits before and after.
pub(crate) fn plant_recompute_word(words: &[u8]) -> Option<(usize, u32, u32)> {
    words.chunks_exact(4).enumerate().find_map(|(i, c)| {
        let b = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        let v = f32::from_bits(b);
        (v.is_finite() && v.abs() >= 1e-30).then_some((i, b, b ^ PLANT_RC_BIT))
    })
}
/// speed live (--bstore-live 1): whether the state stored at boundary `s` of a window of `steps` steps leaves the
/// plan's non-carried allocations out. Boundary 0 and the window end stay whole: the dots, the directions and the
/// window-end dumps read them as whole bytes.
pub(crate) fn bstore_live_cuts(s: usize, steps: usize) -> bool { s >= 1 && s < steps }
/// speed live: `index` (ascending allocation, offsets one after the other) without the allocations `out` names,
/// the offsets laid out again one after the other.
pub(crate) fn live_index(index: &[(u64, usize, usize)], out: &dyn Fn(u64) -> bool) -> Vec<(u64, usize, usize)> {
    let mut at = 0usize;
    index.iter().filter(|x| !out(x.0)).map(|&(id, _, n)| { let r = (id, at, n); at += n; r }).collect()
}
/// speed live: the allocations a window with --bstore-live 1 leaves out of an intermediate boundary: every plan
/// allocation (initial, or made by an Allocate in any block) that plan.carried does not name. The producer certifies those
/// scratch: overwritten before read on each lifetime, and nothing of them crosses a step boundary (finish_step hashes
/// plan.carried only). An allocation the plan does not know (one the window itself makes) is kept.
pub(crate) fn plan_not_carried(plan: &Plan) -> std::collections::HashSet<u64> {
    let carried: std::collections::HashSet<u64> = plan.carried.iter().copied().collect();
    let mut cut: std::collections::HashSet<u64> = plan.allocations.iter().map(|a| a.id).filter(|id| !carried.contains(id)).collect();
    for ops in plan.blocks.values() { for op in ops { if let Operation::Allocate { id, .. } = op { if !carried.contains(id) { cut.insert(*id); } } } }
    cut
}
/// speed live (AD_PLANT_LEFT_OUT=<allocation>|largest, TEST ONLY, with --bstore-live 1): one CARRIED allocation is
/// left out of every intermediate boundary's state as well (zero-filled on restore like the non-carried ones), while the
/// redo check still compares it wherever a truth holds it. THE BREAKAGE THIS PREVENTS: a cut that leaves out an allocation
/// the next step reads would restore a wrong start state; with nothing planted no run could show that the redo check (or
/// the recomputed step's reference hash) refuses it. A run with it set is never valid (window.json planted_error from the
/// start). "largest": the largest live allocation that carries a tangent (the model state), lowest id on a tie.
const PLANT_LO_SIDE: &str = "the state of every intermediate boundary with --bstore-live 1 (window.rs cs_index_at): one carried allocation is left out too and zero-filled on restore; the recorded forward, boundary 0, the window end and the tangent are untouched";
pub(crate) fn plant_left_out_parse(v: &str) -> Option<Option<u64>> {
    match v { "largest" => Some(None), x => x.parse::<u64>().ok().map(Some) }
}
fn plant_left_out() -> &'static Option<Option<u64>> {
    static P: std::sync::OnceLock<Option<Option<u64>>> = std::sync::OnceLock::new();
    P.get_or_init(|| plant_left_out_parse(&std::env::var("AD_PLANT_LEFT_OUT").ok()?))
}
/// The allocation AD_PLANT_LEFT_OUT=largest names among `index`: the largest one `eligible` admits, lowest id on a tie.
pub(crate) fn plant_left_out_pick(index: &[(u64, usize, usize)], eligible: &dyn Fn(u64) -> bool) -> Option<u64> {
    index.iter().filter(|x| eligible(x.0)).fold(None, |b: Option<(usize, u64)>, &(id, _, n)| match b { Some((bn, bi)) if bn > n || (bn == n && bi < id) => b, _ => Some((n, id)) }).map(|x| x.1)
}
/// Writes `rec` as window.json's planted_error now (the receipt set by the window's start), so the run reads planted
/// and not valid also when it is then refused and never writes its receipt.
fn plant_note(rec: Value) -> Result<(), String> {
    if let Ok(g) = PLANT_RC_OUT.lock() { if let Some((p, r0)) = g.as_ref() {
        let mut r: Value = fs::read(p).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_else(|| r0.clone());
        r["planted_error"] = rec;
        fs::write(p, serde_json::to_vec_pretty(&r).unwrap()).map_err(|e| format!("{}: {e}", p.display()))?;
    } }
    Ok(())
}
/// The binder's own verdict (window.json valid_derivative and the window_summary event). A planted one-sided error
/// (AD_PLANT) is a test of the check, so such a run is never valid, whatever its dot reads. THE BREAKAGE THIS PREVENTS:
/// a planted factor close to 1, or a kernel with a small share, keeps the dot under the bar, and the verdict read valid
/// true with a wrong adjoint row in the run (FOLD audit3, <local path> section 3).
/// `base_unchecked`: the window's forward was not compared with the recording (--params-unchecked 1 or --x0-dir; with
/// --checkpoints its recomputed stretches are not either). THE BREAKAGE THIS PREVENTS: such a window with a dot under
/// the bar wrote valid_derivative true with no step compared with anything (a development line reading of a development line
/// --checkpoints with --params patch, <local path> 2026-11-11T06:51Z, point 4).
fn valid_rule(numbers_pass: bool, dots_at_bar: bool, none_refused_beyond_list: bool, planted: bool, base_unchecked: bool) -> bool {
    numbers_pass && dots_at_bar && none_refused_beyond_list && !planted && !base_unchecked
}
/// Whether the reverse taping counts this command's label. The labels (window.json commands_by_rule, refused,
/// refuse_list_kinds, complete_operator) are counted in the first tangent pass. A window with no tangent pass
/// (--no-tangent 1) asks no rule in its forward, so its first objective reverse counts them instead: each
/// (step, command) once, however often the taping visits it (a chunk that does not fit is taped again). With `seen`
/// None (a tangent pass ran, or a later sweep) nothing is counted here. THE BREAKAGE THIS PREVENTS: such a window
/// wrote refused {}, commands_by_rule {}, refuse_list_kinds 0 and complete_operator true although its 64 listed
/// kinds ran as identity (DOORS 2026-11-11T04:01Z, <local path>).
/// No-recording first build, the self-driven live save (off unless AD_LIVE_SELF=1). The commands in scope are saved through
/// the live adapter (live_save::LiveSave::before / after): a resolved descriptor in, as the forward model would hand
/// it over, with no recording path and no replay frame; the binder's own exec stands in for the caller's forward
/// launch; the shared walk-back consumes the records. Scope: AD_LIVE_SELF_STEPS=a:b (steps a..b),
/// AD_LIVE_SELF_STAGE=k (commands inside RK stage k only), AD_LIVE_SELF_SPLIT=k, AD_LIVE_SELF_OPS=a:b (command
/// numbers a..b of a step). What it is for: the first card comparison of the live save route with the recorded one.
struct LiveSelf { steps: (usize, usize), stage: Option<i64>, split: Option<i64>, ops: (usize, usize) }
impl LiveSelf {
    fn covers(&self, step: usize, i: usize, rk: Option<&rk_schedule::Scope>) -> bool {
        step >= self.steps.0 && step < self.steps.1 && i >= self.ops.0 && i < self.ops.1
            && self.stage.map_or(true, |k| rk.and_then(|s| s.stage).is_some_and(|st| st.stage == k))
            && self.split.map_or(true, |k| rk.and_then(|s| s.split) == Some(k))
    }
}
fn live_self_cfg() -> Option<&'static LiveSelf> {
    static CFG: std::sync::OnceLock<Option<LiveSelf>> = std::sync::OnceLock::new();
    CFG.get_or_init(|| {
        if std::env::var("AD_LIVE_SELF").as_deref() != Ok("1") { return None; }
        let range = |name: &str| -> (usize, usize) {
            std::env::var(name).ok().and_then(|v| { let (a, b) = v.split_once(':')?; Some((a.parse().ok()?, b.parse().ok()?)) }).unwrap_or((0, usize::MAX))
        };
        let int = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<i64>().ok());
        let c = LiveSelf { steps: range("AD_LIVE_SELF_STEPS"), stage: int("AD_LIVE_SELF_STAGE"), split: int("AD_LIVE_SELF_SPLIT"), ops: range("AD_LIVE_SELF_OPS") };
        eprintln!("{}", json!({"event":"live_self_on","steps":[c.steps.0, c.steps.1],"stage":c.stage,"split":c.split,"ops":[c.ops.0, c.ops.1]}));
        Some(c)
    }).as_ref()
}
fn live_self_kind(op: &Operation) -> String {
    match op {
        Operation::Kernel { function, .. } => format!("kernel-with-frame-argument:{function}"),
        Operation::Upload { .. } => "upload".into(), Operation::HeldUploadRanges { .. } => "held_upload_ranges".into(),
        Operation::FrameUpload { .. } => "frame_upload".into(), Operation::Allocate { .. } => "allocate".into(),
        Operation::Free { .. } | Operation::FreeNull { .. } => "free".into(), Operation::TransportCopy { .. } => "transport_copy".into(),
        Operation::AddInplace { .. } => "add_inplace".into(), Operation::RadiationUnpack { .. } => "radiation_unpack".into(),
        Operation::VinterpOzone { .. } => "vinterp_ozone".into(), Operation::ModuleGlobalCopy { .. } => "module_global_copy".into(),
        Operation::HostUpload { .. } => "host_upload".into(),
        Operation::HostPublish { .. } => "host_publish".into(),
        Operation::PointerTableUpload { .. } => "pointer_table_upload".into(),
        Operation::DescriptorTableUpload { .. } => "descriptor_table_upload".into(),
        Operation::TempoPointerTableBind { .. } => "tempo_pointer_table_bind".into(),
        Operation::TempoNamedTableBind { .. } => "tempo_named_table_bind".into(),
        Operation::TempoTableRelease { .. } => "tempo_table_release".into(),
        Operation::TempoMainComplete { .. } => "tempo_main_complete".into(),
        Operation::SeaicePointerTableUpload { .. } => "seaice_pointer_table_upload".into(),
        Operation::RucPointerTableUpload { .. } => "ruc_pointer_table_upload".into(),
        Operation::RrfsCopyPlanUpload { .. } => "rrfs_copy_plan_upload".into(),
        Operation::RrfsCopyPlanKernel { .. } => "rrfs_copy_plan_kernel".into(),
        Operation::NativeCubReduce { .. } => "native_cub_reduce".into(),
        Operation::NativeCubSegmentedReduce { .. } => "native_cub_segmented_reduce".into(),
        Operation::NativeThrustArgsort { .. } => "native_thrust_argsort".into(),
        Operation::NativeCublasTranspose { .. } => "native_cublas_transpose".into(),
        Operation::DeviceToHost { .. } => "device_to_host".into(),
        Operation::Synchronization { .. } => "synchronization".into(),
        Operation::EventRecord { .. } => "event_record".into(),
        Operation::EventWait { .. } => "event_wait".into(),
        Operation::HostBranch { .. } => "host_branch".into(),
        Operation::HostBoundary { .. } => "host_boundary".into(),
        Operation::RkSchedule { .. } => "rk_schedule".into(),
        Operation::RkSplits { .. } => "rk_splits".into(),
        Operation::RkStages { .. } => "rk_stages".into(),
        Operation::RkSubsteps { .. } => "rk_substeps".into(),
        Operation::PeerSend { .. } => "peer_send".into(),
        Operation::PeerRecv { .. } => "peer_recv".into(),
        _ => "other".into(),
    }
}
fn rev_count_once(seen: &mut Option<std::collections::HashSet<(usize, usize)>>, step: usize, command: usize) -> bool {
    seen.as_mut().is_some_and(|s| s.insert((step, command)))
}
/// The counted labels of commands that ran as identity ("REFUSED:<kernel>: <why>"); complete_operator is their absence.
fn refused_labels(labels: &BTreeMap<String, u64>) -> BTreeMap<&String, &u64> {
    labels.iter().filter(|(k, _)| k.starts_with("REFUSED:")).collect()
}
/// One dense block of a direction added to the state words it covers: word += h * block word, in float64, rounded
/// once to float32. A product of exactly zero leaves the word's bits alone (-0.0 stays -0.0). Returns the words changed.
fn add_dense(state: &mut [u8], block: &[u8], h: f64) -> u64 {
    let mut moved = 0u64;
    for (c, d) in state.chunks_exact_mut(4).zip(block.chunks_exact(4)) {
        let step = h * f32::from_le_bytes(d.try_into().unwrap()) as f64;
        if step == 0.0 { continue; }
        let pv = f32::from_le_bytes((&*c).try_into().unwrap());
        let nv = (pv as f64 + step) as f32;
        if nv.to_bits() != pv.to_bits() { moved += 1; }
        c.copy_from_slice(&nv.to_le_bytes());
    }
    moved
}
/// <block, shadow> of one dense block, in float64, in word order.
fn dot_dense(block: &[u8], shadow: &[u8]) -> f64 {
    block.chunks_exact(4).zip(shadow.chunks_exact(4)).map(|(d, q)| f32::from_le_bytes(d.try_into().unwrap()) as f64 * f32::from_le_bytes(q.try_into().unwrap()) as f64).sum()
}
/// Largest |primal| a relative direction weights (base pressure is ~1e5 Pa). The 1-step trace (trace1) showed the
/// tangent norm grow from 1.8e26 to 2.6e35 in edge_fields_from_saved_u with a 1e12 cap: stale scratch words of up to
/// 1e12 at boundary 0 got weights that large and their kinetic-energy tangents (2 u du) squared them.
const WEIGHT_MAX: f32 = 2e5;

/// Species planes (AD_DOT_PLANES, off unless set): the window check treats one stacked field (the 13 scalar
/// species of MPAS `scalars`, plane-major) as N planes, each with its own weight and its own self-dot unit, at boundary 0
/// (the x side) and at the window end (the y side).
/// THE BREAKAGE THIS PREVENTS: the species shared one allocation, so one field weight and one self-dot unit, set by the
/// largest words. On the 12 km cut the number concentrations held 0.999999 of sum |x| (nifa 0.680, ni 0.262, nr 0.037,
/// nc 0.022), the six masses about 1.07e-06 (qr 1.0e-09, qs 4.4e-09, qg 1.0e-10), and nwfa none (every word over
/// WEIGHT_MAX): a wrong transpose on the qr, qs and qg paths (reflectivity's) passed the check unseen.
/// Form: `X_ALLOC@X_OFF,Y_ALLOC@Y_OFF/NxPLANE_BYTES[/name,name,...]`, offsets and plane size in bytes; or the one word
/// `scalars`, which writes that form from the package's own field map (planes_from_map). With it:
///  - each plane's field weight is the rms of its own valid words (--direction field: y side; fieldxy: both sides), and
///    the rest of the allocation's is the rms of the words outside the planes;
///  - inside a plane a word is valid when finite and above 1e-30: the WEIGHT_MAX cap (meant for stale scratch and address
///    words) does not apply to a declared species plane, so nwfa is weighted too (AD_DOT_PLANES_CAP=1 keeps the cap);
///  - AD_DOT_SELF=1: at the window end each plane is its own unit (id PLANE_UNIT0 + plane); the rest of the allocation
///    stays a unit of its own id;
///  - AD_DOT_SELF_SPLIT=planes: one more reverse per plane seeded with that plane's self y alone, and one for all other
///    units, so x_MTy_g - y_Mx_g is read species by species; AD_DOT_SELF_SPLIT=planes:qr,qs,qg only the planes named.
#[derive(Clone, Debug)]
struct PlaneSpec { x: (u64, usize), y: (u64, usize), n: usize, pb: usize, names: Vec<String>, cap: bool }
const PLANE_UNIT0: u64 = 1_000_000;
/// species planes: a planed allocation's table: per plane (byte lo, byte hi, field weight), whether the cap holds in the
/// planes, and the field weight of the allocation's words outside the planes.
type PlaneW = (Vec<(usize, usize, f32)>, bool, f32);
/// AD_DOT_PLANES=scalars: the full form, written by run_window from the package's field map before the first
/// plane_spec() call (planes_from_map), or why it could not be written.
static PLANES_FROM_MAP: std::sync::OnceLock<Result<String, String>> = std::sync::OnceLock::new();
/// The memory order of the 13 scalar planes of the hrrrv5 suite (the MPAS Registry order; checked byte for byte against
/// a 12 km start file). A package's alloc-map.json `species_order` list holds the same 13 names but is not the memory
/// order for planes 6 to 12, so the names are taken from here and the list only says that the set is this one.
const SCALAR_PLANES_13: [&str; 13] = ["qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "ng", "volg", "nc", "nifa", "nwfa"];
fn plane_spec() -> Result<Option<&'static PlaneSpec>, String> {
    static P: std::sync::OnceLock<Result<Option<PlaneSpec>, String>> = std::sync::OnceLock::new();
    match P.get_or_init(|| match std::env::var("AD_DOT_PLANES").ok().filter(|v| !v.trim().is_empty()) {
        None => Ok(None),
        Some(v) if v.trim() == "scalars" => match PLANES_FROM_MAP.get() {
            Some(Ok(full)) => parse_plane_spec(full).map(Some),
            Some(Err(e)) => Err(e.clone()),
            None => Err("AD_DOT_PLANES=scalars: only the window subcommand reads the package's field map".into()) },
        Some(v) => parse_plane_spec(&v).map(Some) }) {
        Ok(p) => Ok(p.as_ref()),
        Err(e) => Err(e.clone()),
    }
}
/// The package's field map (alloc-map.json beside the plan), or why there is none to read.
fn package_field_map(config: &Path) -> Result<(PathBuf, Value), String> {
    let m = fs::canonicalize(config).ok().and_then(|c| c.parent().map(|d| d.join("alloc-map.json"))).filter(|m| m.is_file())
        .or_else(|| config.parent().map(|d| d.join("alloc-map.json")).filter(|m| m.is_file()))
        .ok_or("no alloc-map.json beside the plan")?;
    let v = fs::read(&m).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()).ok_or(format!("{} is unreadable", m.display()))?;
    Ok((m, v))
}
/// The window-end boundary of a window of `steps` steps in a package's field map: boundary 1 is the recorded start, so
/// the end is boundary steps + 1. Err names the map and what it holds when that boundary is not in it.
fn map_window_end(m: &Path, v: &Value, steps: usize) -> Result<String, String> {
    let b = format!("{}", steps + 1);
    if v["boundaries"][&b].is_object() { Ok(b) } else {
        Err(format!("{} has no boundary {} (the window end of {} steps; it holds {} boundaries, the recorded start being boundary 1)",
                    m.display(), steps + 1, steps, v["boundaries"].as_object().map_or(0, |o| o.len()))) }
}
/// AD_DOT_PLANES=scalars: the full spec from the package's field map. The x side is the `scalars` field at boundary 1
/// (the recorded start), the y side the same field at the window end (boundary steps + 1); the planes are the field's
/// first axis (plane-major float32, strides[0] bytes each). Names: SCALAR_PLANES_13 when the map's species set is that
/// one, plane numbers otherwise. Refused, with the reason, when the map lacks the field, the layout or the boundary.
fn planes_from_map(config: &Path, steps: usize) -> Result<String, String> {
    let why = |m: String| format!("AD_DOT_PLANES=scalars: {m}; give the planes in full (X_ALLOC@X_OFF,Y_ALLOC@Y_OFF/NxPLANE_BYTES[/name,...])");
    let (m, v) = package_field_map(config).map_err(|e| why(e))?;
    let f = &v["fields"]["scalars"];
    let nums = |k: &str| -> Vec<u64> { f[k].as_array().map(|a| a.iter().filter_map(Value::as_u64).collect()).unwrap_or_default() };
    let (shape, strides) = (nums("shape"), nums("strides"));
    if shape.len() != 3 || strides.len() != 3 || strides[2] != 4 || strides[1] != 4 * shape[2] || strides[0] != strides[1] * shape[1] {
        return Err(why(format!("{} has no plane-major float32 `scalars` field (shape {:?}, strides {:?})", m.display(), shape, strides))); }
    let end = map_window_end(&m, &v, steps).map_err(|e| why(e))?;
    let at = |b: &str| -> Option<(u64, u64)> { let e = &v["boundaries"][b]["scalars"]; Some((e["allocation"].as_u64()?, e["offset"].as_u64()?)) };
    let (x, y) = (at("1").ok_or_else(|| why(format!("{} names no `scalars` at boundary 1", m.display())))?,
                  at(&end).ok_or_else(|| why(format!("{} names no `scalars` at boundary {end}", m.display())))?);
    let mut listed: Vec<&str> = v["species_order"].as_array().map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
    listed.sort_unstable();
    let mut known = SCALAR_PLANES_13.to_vec(); known.sort_unstable();
    let names = if shape[0] == 13 && listed == known { format!("/{}", SCALAR_PLANES_13.join(",")) } else { String::new() };
    Ok(format!("{}@{},{}@{}/{}x{}{}", x.0, x.1, y.0, y.1, shape[0], strides[0], names))
}
fn parse_plane_spec(v: &str) -> Result<PlaneSpec, String> {
    let bad = |m: &str| format!("AD_DOT_PLANES={v:?}: {m} (form X_ALLOC@X_OFF,Y_ALLOC@Y_OFF/NxPLANE_BYTES[/name,...])");
    let mut parts = v.trim().split('/');
    let ends = parts.next().ok_or_else(|| bad("no ends"))?;
    let shape = parts.next().ok_or_else(|| bad("no NxPLANE_BYTES"))?;
    let names: Vec<String> = parts.next().map(|s| s.split(',').map(|x| x.trim().to_string()).collect()).unwrap_or_default();
    if parts.next().is_some() { return Err(bad("too many parts")); }
    let end = |s: &str| -> Result<(u64, usize), String> {
        let (a, o) = s.trim().split_once('@').ok_or_else(|| bad("an end without '@'"))?;
        Ok((a.trim().parse().map_err(|_| bad("allocation"))?, o.trim().parse().map_err(|_| bad("offset"))?)) };
    let (xe, ye) = ends.split_once(',').ok_or_else(|| bad("two ends wanted"))?;
    let (x, y) = (end(xe)?, end(ye)?);
    let (n, pb) = shape.trim().split_once('x').ok_or_else(|| bad("NxPLANE_BYTES"))?;
    let (n, pb): (usize, usize) = (n.trim().parse().map_err(|_| bad("N"))?, pb.trim().parse().map_err(|_| bad("PLANE_BYTES"))?);
    if n == 0 || pb == 0 || pb % 4 != 0 || x.1 % 4 != 0 || y.1 % 4 != 0 { return Err(bad("N and PLANE_BYTES above 0, offsets and sizes in whole 32-bit words")); }
    if !names.is_empty() && names.len() != n { return Err(bad("names: one per plane")); }
    let names = if names.is_empty() { (0..n).map(|p| format!("plane{p}")).collect() } else { names };
    Ok(PlaneSpec { x, y, n, pb, names, cap: std::env::var("AD_DOT_PLANES_CAP").ok().as_deref() == Some("1") })
}
impl PlaneSpec {
    /// (allocation, byte offset of plane 0) on the side asked: the window end (y) or boundary 0 (x)
    fn end(&self, y_side: bool) -> (u64, usize) { if y_side { self.y } else { self.x } }
    /// per plane its (byte lo, byte hi) inside that allocation
    fn ranges(&self, y_side: bool) -> Vec<(usize, usize)> { let o = self.end(y_side).1; (0..self.n).map(|p| (o + p * self.pb, o + (p + 1) * self.pb)).collect() }
    fn receipt(&self) -> Value {
        json!({"x":{"allocation":self.x.0,"offset":self.x.1},"y":{"allocation":self.y.0,"offset":self.y.1},"planes":self.n,"plane_bytes":self.pb,
               "names":self.names,"unit_ids":(0..self.n).map(|p| PLANE_UNIT0 + p as u64).collect::<Vec<_>>(),"cap_in_planes":self.cap,
               "rule":"each plane its own field weight (rms of its own valid words) and its own self-dot unit at the window end; inside a plane valid = finite and above 1e-30 (no 2e5 cap unless cap_in_planes)"})
    }
}

macro_rules! kargs { ($($x:ident),*) => { &mut [$((&mut $x) as *mut _ as *mut c_void),*] } }

struct Fns {
    rows_tl: *mut c_void, gather_ad: *mut c_void, rows_ad_direct: *mut c_void, clear_dst: *mut c_void, rows_ad_fan: *mut c_void,
    view_tl: *mut c_void, view_ad: *mut c_void, where_tl: *mut c_void, where_ad: *mut c_void,
    max_tl: *mut c_void, max_ad: *mut c_void, view_zero: *mut c_void, fill: *mut c_void, dot2: *mut c_void, nonfinite: *mut c_void,
    fct_minmax_tl: *mut c_void, fct_minmax_ad: *mut c_void, axpy: *mut c_void, widen: *mut c_void, narrow: *mut c_void,
    /// 60 km tape: the pitched pair (null when the fatbin lacks them; then the one-row-a-launch loop).
    widen2: *mut c_void, narrow2: *mut c_void,
}
/// 60 km tape: AD_P3_STRIPE=1 also moves a banded MYNN-EDMF launch's row stripes in pitched launches (read once).
fn stripe_xfer_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AD_P3_STRIPE").as_deref() == Ok("1"))
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

/// bs-launch (AD_BS_PROF=1, diagnostic only, default off: one relaxed load otherwise): host wall of each part of a
/// reverse launch's handling in the walk-back, and per function. AD_BS_PROF=2 also drains the card before and after
/// each entry and rule row, so the card time of that row is read apart from the host time (timing only; no word moves).
mod bsprof {
    use std::sync::atomic::{AtomicU64, AtomicU8, Ordering::Relaxed};
    use std::time::Instant;
    static ON: AtomicU8 = AtomicU8::new(255);
    pub fn level() -> u8 {
        let v = ON.load(Relaxed);
        if v != 255 { return v; }
        let l = std::env::var("AD_BS_PROF").ok().and_then(|s| s.parse::<u8>().ok()).unwrap_or(0).min(2);
        ON.store(l, Relaxed);
        l
    }
    pub const PARTS: [&str; 14] = ["e_live_clone", "e_bind", "e_aug", "e_cells_seeds", "e_begin_ad", "e_entries", "e_finish_ad",
                                   "e_failures", "e_note", "e_card_drain", "r_host", "r_card_drain", "e_pre_drain", "r_pre_drain"];
    static NS: [AtomicU64; 14] = [const { AtomicU64::new(0) }; 14];
    static N: [AtomicU64; 14] = [const { AtomicU64::new(0) }; 14];
    static LAUNCHES: AtomicU64 = AtomicU64::new(0);
    static FN: std::sync::Mutex<Option<std::collections::HashMap<String, [f64; 3]>>> = std::sync::Mutex::new(None);
    pub fn lap(k: usize, t: &mut Instant) -> f64 {
        let now = Instant::now();
        let d = now.duration_since(*t);
        NS[k].fetch_add(d.as_nanos() as u64, Relaxed); N[k].fetch_add(1, Relaxed);
        *t = now;
        d.as_secs_f64()
    }
    pub fn launches(n: usize) { LAUNCHES.fetch_add(n as u64, Relaxed); }
    pub fn add_fn(f: &str, host: f64, card: f64) {
        if let Ok(mut g) = FN.lock() {
            let m = g.get_or_insert_with(Default::default);
            let e = m.entry(f.to_owned()).or_insert([0.0; 3]);
            e[0] += host; e[1] += card; e[2] += 1.0;
        }
    }
    pub fn take() -> serde_json::Value {
        let parts: serde_json::Map<String, serde_json::Value> = PARTS.iter().enumerate().map(|(k, n)|
            ((*n).to_owned(), serde_json::json!([NS[k].swap(0, Relaxed) as f64 / 1e9, N[k].swap(0, Relaxed)]))).collect();
        let mut fns: Vec<(String, [f64; 3])> = FN.lock().ok().and_then(|mut g| g.take()).map(|m| m.into_iter().collect()).unwrap_or_default();
        fns.sort_by(|a, b| (b.1[0] + b.1[1]).partial_cmp(&(a.1[0] + a.1[1])).unwrap_or(std::cmp::Ordering::Equal));
        serde_json::json!({"level": level(), "parts_s_n": parts, "entry_launches": LAUNCHES.swap(0, Relaxed), "launch_impl_s": crate::deriv::lprof::take(),
            "fields": "functions: [name, host s, card drain s (level 2), rows]",
            "functions": fns.iter().map(|(f, v)| serde_json::json!([f, v[0], v[1], v[2] as u64])).collect::<Vec<_>>()})
    }
    pub fn rkind(r: &crate::wrules::R) -> &'static str {
        use crate::wrules::R;
        match r {
            R::Rows { .. } => "rule:rows", R::View { .. } => "rule:view", R::ZeroView { .. } => "rule:zero_view", R::Max { .. } => "rule:max",
            R::Where { .. } => "rule:where", R::Scale { .. } => "rule:scale", R::Copy { .. } => "rule:copy", R::Axpy { .. } => "rule:axpy", R::Zero { .. } => "rule:zero",
            R::Held { .. } => "rule:held", R::FctMinmax { .. } => "rule:fct_minmax", R::Entry | R::Identity(_) | R::Refused(_) => "rule:none",
            _ => "rule:other",
        }
    }
}
fn grid1(n: i64) -> [u32; 3] { [((n.max(1) + 255) / 256).min(i32::MAX as i64) as u32, 1, 1] }
fn grid_rows(rows: &[DevRow]) -> [u32; 3] {
    let maxc = rows.iter().map(|r| r.count).max().unwrap_or(0);
    [((maxc + 255) / 256).clamp(1, 1024) as u32, rows.len() as u32, 1]
}

/// truth (2026-11-09): named kernel outputs held at the base trajectory's words in perturbed forwards.
/// AD_HOLD_MCICA=1 holds the RRTMG McICA seeds (mpas_sw_mcica_seeds argument 9, mpas_lw_mcica_seeds argument 8;
/// 16 bytes per column, column count = argument 0); AD_HOLD_MCICA=census only counts the seed words a perturbed
/// forward changes. AD_HOLD_OUT=function:arg:unit_bytes:count_arg,.. adds any other kernel output.
/// THE BREAKAGE THIS NAMES: the forward draws each column's McICA seeds from the fractional Pascals of its four
/// lowest layer pressures (seed = (u32)((p - (int)p) * 1e9)), so a one-ulp pressure change re-draws the column's
/// cloud sample (measured: 85-86 % of seed words, tree/ad/phys/held/mpas_sw_mcica_seeds/RESULT.md). A perturbed
/// forward then differs from the base by re-sampled radiation wherever the perturbation has reached at the ulp
/// level, which no tangent predicts. With the hold on, a perturbed forward uses the base forward's seeds for the
/// same launch; the base forecast itself is unchanged word for word. Default off: the exact forward (live seeds)
/// stays the default and stays selectable. Launches inside a live LW regroup section are counted and left live
/// (their batches hold other columns than the base forward's).
struct HoldSpec { function: String, arg: usize, unit: usize, count_arg: usize }
#[derive(Default)]
struct Hold {
    specs: Vec<HoldSpec>, apply: bool,
    /// 0: base forward (keep the words); 1: perturbed forward (count, and write the base words back when `apply`).
    mode: u8, ord: usize,
    /// the pass of the live LW regroup section the walk is in (None outside one); set by the walk before each leaf
    regroup: Option<usize>,
    store: HashMap<(usize, usize), (String, Vec<u8>)>,
    words: u64, differ: u64, launches: u64, unheld: u64, by_step: BTreeMap<usize, (u64, u64)>,
}
impl Hold {
    fn from_env() -> Result<Hold, String> {
        let mut h = Hold::default();
        let m = std::env::var("AD_HOLD_MCICA").unwrap_or_default();
        let mut text = std::env::var("AD_HOLD_OUT").unwrap_or_default();
        if m == "1" || m == "census" {
            if !text.is_empty() { text.push(','); }
            text.push_str("mpas_sw_mcica_seeds:9:16:0,mpas_lw_mcica_seeds:8:16:0");
        } else if !m.is_empty() && m != "0" { return Err(format!("AD_HOLD_MCICA={m}: want 1, census or 0")); }
        for item in text.split(',').map(str::trim).filter(|x| !x.is_empty()) {
            let q: Vec<&str> = item.split(':').collect();
            let n = |i: usize| q.get(i).and_then(|v| v.parse::<usize>().ok());
            match (q.first(), n(1), n(2), n(3)) {
                (Some(f), Some(arg), Some(unit), Some(count_arg)) if q.len() == 4 && unit > 0 =>
                    h.specs.push(HoldSpec { function: (*f).to_owned(), arg, unit, count_arg }),
                _ => return Err(format!("AD_HOLD_OUT item {item}: want function:arg:unit_bytes:count_arg")),
            }
        }
        h.apply = !h.specs.is_empty() && m != "census";
        if !h.specs.is_empty() {
            eprintln!("{}", json!({"event":"hold","held":h.apply,"outputs":h.specs.iter().map(|s| json!([s.function, s.arg, s.unit, s.count_arg])).collect::<Vec<_>>()}));
        }
        Ok(h)
    }
    fn on(&self) -> bool { !self.specs.is_empty() }
    /// The counts of one perturbed forward (since the last call), as a receipt object.
    fn take(&mut self) -> Value {
        let v = json!({"held":self.apply,"launches":self.launches,"launches_left_live_lw_regroup":self.unheld,"words":self.words,
            "words_live_differ_from_base":self.differ,"by_step":self.by_step.iter().map(|(s, (w, d))| json!([s, w, d])).collect::<Vec<_>>()});
        self.words = 0; self.differ = 0; self.launches = 0; self.unheld = 0; self.by_step.clear();
        v
    }
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
    // a12 speed: a moved-start run (--x0-dir) skips every step-end digest for the whole run (x0_digest_skip below), so the
    // history it stores with a step end is one nothing reads; every other run refuses as before.
    assert!(!ex.history_stale || ex.unchecked_digest_skip, "DEFECT boundary stored after a skipped unchecked digest with no restore between (per-step-host-hash)");
    HostSnap { host_state: ex.host_state.clone(), tempo_cache: ex.tempo_cache.clone(), tempo_bindings: ex.tempo_bindings.clone(),
               pointer_history: ex.history_snap(), held_pointer_writes: ex.held_pointer_writes.clone(),
               events: ex.events.keys().copied().collect(), frame: frame.clone(), rk: ex.rk.clone() }
}
/// speed (merge, round 4): the end of a checkpointed sweep (sweep_revolve): the sweep's own refusal first; else the
/// join of its boundary digests. A recomputed boundary that differs from the recording refuses the sweep here, so its
/// rows never count (main.rs digest_join_tests).
pub(crate) fn sweep_joined<T>(rows: Result<T, String>, joined: Result<(), String>) -> Result<T, String> {
    let rows = rows?;
    joined?;
    Ok(rows)
}
/// central wide-tools (2026-11-11): whether a checkpointed sweep (sweep_revolve) hashes its re-run boundaries beside the
/// card (digest_lazy). a development line condition (<local path>, 06:51:09Z, point 2): only when the re-run is checked.
/// THE BREAKAGE IT PREVENTS: on the tuning path (--checkpoints with --params and --params-unchecked 1) a re-run step is
/// compared with nothing (advance calls forward_step_c with no references), yet with digest_lazy set its boundary was
/// deferred and counted in window.json checkpointing.recompute_hash_beside.boundaries as a boundary hashed beside the
/// card and compared. Standard runs (recompute_unchecked false) are unchanged: the result is recompute_hash_beside().
pub(crate) fn sweep_digest_lazy(beside: bool, recompute_unchecked: bool) -> bool { beside && !recompute_unchecked }
/// keep-forward: AD_KEEP_FORWARD=1 or 2 (default off: nothing below runs), a kept-loaded
/// gradient program (--resident, --checkpoints, checkpoints in host memory) only. A later request whose start is the
/// start of the request before does not run the forward again: it takes that forward's checkpoints, window-end state
/// and step lists (KeptForward) and goes to the backward sweep. THE COST THIS REMOVES: the forward of the same forecast,
/// run again for every target (17.68 s of the 208.8 s later gradient at 59 steps on the 3 km storm box,/// request 2 of one loaded program: fill_x 231.92 s to tl_step_59 249.60 s). What makes it the same forward: start_digest, a
/// SHA-256 over every byte of this request's boundary 0 (after the planted start and the --x0-dir move), its layout,
/// the step count and the payload of every step-1 upload a planted start swapped; any difference and the forward runs
/// as before and its results replace the kept ones. What it holds: the sweep moves the forward's checkpoints aside as
/// it passes them instead of dropping them (kf_saved), so up to one more set of checkpoints is in page-locked host
/// memory while the sweep runs and between requests (ceil(sqrt(steps)) states: 8 x about 10.5 GB at 59 steps), outside
/// the --bstore-ram-gib budget (the sweep's own checkpoints get the room they had). 1: the device and the host state
/// are put at the kept window end before the sweep, where the forward would have left them. 2: they are not (the
/// sweep's first act is a checkpoint restore), and the random x a forward without tangent never reads is not filled.
pub(crate) fn keep_forward_word() -> u8 {
    static ON: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("AD_KEEP_FORWARD").as_deref() { Ok("1") => 1, Ok("2") => 2, _ => 0 })
}
/// keep-forward: what one request's forward leaves for the next request of the same start.
struct KeptForward { key: String, from: String, end: Boundary, ck: BTreeMap<usize, Boundary>, flats: Vec<Vec<Flat>>, per_step: Vec<Value>,
                     bytes: usize, forward_s: f64 }
/// keep-forward: the fingerprint of a request's start (see keep_forward_word). `d` is this request's boundary 0 and
/// `index` its layout; `fresh` names the allocations a fresh program holds at boundary 0 (the kept first boundary 0's):
/// only their bytes are the start. THE BREAKAGE THE CUT PREVENTS: a later request's boundary 0 also holds the
/// allocations the window before made (restore frees nothing), with whatever that window left in them, which every
/// forward overwrites before reading; hashed, they would make the first later request look like another start. The
/// bytes are hashed in pieces of 64 MiB on up to 32 cores, the piece digests in order; `ops` are the step-1 uploads a
/// planted start holds swapped now, with their payload sha256.
fn start_digest(index: &[(u64, usize, usize)], fresh: Option<&std::collections::HashSet<u64>>, d: &[u8], ops: &[(String, usize, String)], steps: usize) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    const PIECE: usize = 64 << 20;
    let ent: Vec<&(u64, usize, usize)> = index.iter().filter(|x| fresh.is_none_or(|f| f.contains(&x.0))).collect();
    if let Some(f) = fresh { if ent.len() != f.len() { return Err(format!("boundary 0 holds {} of the {} allocations of a fresh start", ent.len(), f.len())); } }
    let mut pieces: Vec<&[u8]> = Vec::new();
    for (id, off, n) in ent.iter().copied() {
        let seg = d.get(*off..*off + *n).ok_or_else(|| format!("allocation {id} lies outside boundary 0's bytes"))?;
        pieces.extend(seg.chunks(PIECE));
    }
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).clamp(1, 32).min(pieces.len().max(1));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out: Vec<std::sync::Mutex<Vec<u8>>> = (0..pieces.len()).map(|_| std::sync::Mutex::new(Vec::new())).collect();
    std::thread::scope(|sc| { for _ in 0..threads { sc.spawn(|| loop {
        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if i >= pieces.len() { break; }
        *out[i].lock().unwrap() = Sha256::digest(pieces[i]).to_vec();
    }); } });
    let mut h = Sha256::new();
    h.update(b"keep-forward/v2");
    h.update((steps as u64).to_le_bytes()); h.update((PIECE as u64).to_le_bytes()); h.update((ent.len() as u64).to_le_bytes());
    for (id, _, n) in ent.iter().copied() { h.update(id.to_le_bytes()); h.update((*n as u64).to_le_bytes()); }
    for (blk, op, sha) in ops { h.update(blk.as_bytes()); h.update([0u8]); h.update((*op as u64).to_le_bytes()); h.update(sha.as_bytes()); h.update([0u8]); }
    for p in &out { let g = p.lock().unwrap(); if g.len() != 32 { return Err("a piece of boundary 0 was not hashed".into()); } h.update(&g[..]); }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}
#[cfg(test)]
mod keep_forward_tests {
    #[test]
    fn the_start_digest_moves_with_a_fresh_byte_a_size_a_swapped_upload_and_the_step_count_and_not_with_what_a_window_left() {
        let d0: Vec<u8> = (0..6144u32).flat_map(|x| x.to_le_bytes()).collect();
        let ix = vec![(3u64, 0usize, 8192usize), (9, 8192, 8192)];
        let k = super::start_digest(&ix, None, &d0[..16384], &[], 12).unwrap();
        assert_eq!(k, super::start_digest(&ix, None, &d0[..16384], &[], 12).unwrap());
        let mut d1 = d0.clone(); d1[16383] ^= 1;
        assert_ne!(k, super::start_digest(&ix, None, &d1[..16384], &[], 12).unwrap());
        assert_ne!(k, super::start_digest(&[(3, 0, 8192), (10, 8192, 8192)], None, &d0[..16384], &[], 12).unwrap());
        assert_ne!(k, super::start_digest(&ix, None, &d0[..16384], &[("step_01".into(), 7, "ab".into())], 12).unwrap());
        assert_ne!(super::start_digest(&ix, None, &d0[..16384], &[("step_01".into(), 7, "ab".into())], 12).unwrap(),
                   super::start_digest(&ix, None, &d0[..16384], &[("step_01".into(), 7, "ac".into())], 12).unwrap());
        assert_ne!(k, super::start_digest(&ix, None, &d0[..16384], &[], 59).unwrap());
        // a later request: allocation 5, made by the window before, sits between the two fresh ones with its leftovers
        let fresh: std::collections::HashSet<u64> = [3u64, 9].into_iter().collect();
        let mut later = d0[..8192].to_vec(); later.extend(std::iter::repeat(0xa5u8).take(8192)); later.extend_from_slice(&d0[8192..16384]);
        let ix2 = vec![(3u64, 0usize, 8192usize), (5, 8192, 8192), (9, 16384, 8192)];
        assert_eq!(k, super::start_digest(&ix, Some(&fresh), &d0[..16384], &[], 12).unwrap());
        assert_eq!(k, super::start_digest(&ix2, Some(&fresh), &later, &[], 12).unwrap());
        later[20000] ^= 1;
        assert_ne!(k, super::start_digest(&ix2, Some(&fresh), &later, &[], 12).unwrap());
        assert!(super::start_digest(&ix[..1], Some(&fresh), &d0[..16384], &[], 12).is_err());
        assert!(super::start_digest(&ix, None, &d0[..100], &[], 12).is_err());
    }
}
/// a12 s60-rerun: AD_RECOMPUTE_UNCHECKED=1 (default off; see the window's --checkpoints reading).
pub(crate) fn rerun_unchecked_word() -> bool { std::env::var("AD_RECOMPUTE_UNCHECKED").as_deref() == Ok("1") }
#[cfg(test)]
mod sweep_digest_lazy_tests {
    #[test]
    fn an_unchecked_re_run_is_never_hashed_beside_the_card_and_a_checked_one_keeps_the_switch() {
        // the tuning path: re-run steps compared with nothing are not deferred, whatever AD_RECOMPUTE_HASH_BESIDE says
        assert!(!super::sweep_digest_lazy(true, true));
        assert!(!super::sweep_digest_lazy(false, true));
        // a checked re-run (every standard run): the switch decides, as before this change
        assert!(super::sweep_digest_lazy(true, false));
        assert!(!super::sweep_digest_lazy(false, false));
    }
}
unsafe fn restore_host(ex: &mut Execution, frame: &mut Option<ReplayFrame>, h: &HostSnap) -> Result<(), String> {
    // terrain-w-hash: a deferred boundary digest is joined (compared, its history filled) before any restore.
    // speed (storm, round 3): not inside a checkpointed sweep while the chain is still running: then the restored
    // boundary's history goes to the next boundary digest and the chain is joined at the sweep's end (main.rs
    // digest_arenas_max, restore_history_lazy).
    let lazy = ex.restore_join(&h.pointer_history)?;
    ex.host_state = h.host_state.clone(); ex.tempo_cache = h.tempo_cache.clone(); ex.tempo_bindings = h.tempo_bindings.clone();
    match &h.pointer_history {
        _ if lazy => {},
        p => { ex.pointer_history = p.get().clone(); ex.history_stale = false; },
    }
    ex.held_pointer_writes = h.held_pointer_writes.clone();
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
/// bs-tape (a12 backstep): AD_HELD_SKIP=0 turns off the restore that uploads nothing when the card holds the state.
/// drains (AD_NATIVE_NODRAIN=1, off unless asked): in an ordered plan (ordered(ex): every command on the legacy
/// null stream of the one context) a native derivative row neither drains the card before it nor after it. Its launches,
/// copies and zero fills go on that same stream, and each host read it makes (counts, flags, overflow words) is behind
/// its own dev::sync or a synchronous download, which waits for every command queued before it; so the card runs the
/// same commands in the same order and only the host stops waiting where it reads nothing. THE WAIT THIS REMOVES: the
/// drain owed by the entries walked before each native row (12 native rows a 3 km step: fin-later has 732 in 59 steps;
/// drain_before_row 0.55 s a step in fin-later, most of it card time that moves to the next wait) and the drain after the row, so the host binds the
/// next rows while the card still runs this one. Kept: the drain before TEMPO rows (they use streams of their own), the
/// drain at the chunk end, every drain under AD_PLANT on that row, and every drain when MYNN-EDMF runs on shard contexts
/// (AD_MYNN_SHARD_DEVICES: those copies cross contexts). The row's "ms" is then host time only (card time lands in the
/// next drain).
fn native_nodrain() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AD_NATIVE_NODRAIN").as_deref() == Ok("1")
        && std::env::var("AD_MYNN_SHARD_DEVICES").map(|v| v.trim().is_empty()).unwrap_or(true))
}
/// drains (AD_SNAP_KEEP=1, off unless asked): a recomputed start state that snap_ram downloaded into a page-locked
/// buffer stays in that buffer as the sweep's current state (Bytes::Pin), instead of being copied on all cores into a
/// new pageable buffer while the card is idle; the buffer goes back to the anchor pool once the next step is reversed,
/// and the next snap_ram takes it from there. THE WAIT THIS REMOVES: rerun_state_copy_s, 3.48 s a 59-step 3 km gradient
/// (fin-later window_sweep_parts), with the card idle. The same bytes: the copy is a byte copy, and the compare and the
/// held restore read the same words from where they already are. The cost: one more page-locked buffer of the state
/// (10.53 GB at 3 km) while the sweep runs. The same design was tried and taken out before (commit 1e176a1d: no faster on
/// fast-4, 10 GiB more host memory), so it is here only to be measured again on today's line; AD_SNAP_REUSE is the
/// candidate that adds no memory.
fn snap_keep() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AD_SNAP_KEEP").as_deref() == Ok("1"))
}
/// drains (AD_FREE_BESIDE=1, off unless asked): the idle checkpoint buffers released at a sweep's end are
/// unregistered and unmapped by a helper thread, so the program goes on (the dots, the writer, the receipts) while they
/// go. THE WAIT THIS REMOVES: idle_free_s, 4.66 s a 59-step 3 km gradient (fin-later: 9 buffers, 94.8 GB) on the main
/// thread. The bytes of the run do not change: nothing reads those buffers again (they were idle in the pool). With the
/// fast exit (AD_WINDOW_FAST_EXIT, on by default) the process may end before the helper does; the kernel releases what
/// is left, as it does for every other page-locked buffer at that exit. With AD_WINDOW_FAST_EXIT=0 the helper is joined
/// before the unwinding.
/// drains (AD_SNAP_REUSE=1, off unless asked): the copy of a recomputed start state out of its page-locked buffer
/// goes into the pageable buffer the sweep let go of at the end of the step before (the old current state), instead of
/// into a fresh one, and that old state is not freed. THE WAIT THIS REMOVES: the first touch of 10.53 GB of fresh pages
/// inside rerun_state_copy_s (3.48 s for 10 copies in fin-later, card idle) and the free of the old current state at
/// each step end (card idle). a test machine, 2.68 GB, 24 threads: copy into a fresh Vec 0.126-0.138 s, into a kept one
/// 0.094-0.097 s, and the free 0.046-0.050 s (drains receipts/copy-reuse-a test machine.jsonl). The same bytes: the copy
/// writes every byte of the state, and the buffer's length is set to the state's first. Memory: at most two pageable
/// states, as the old line held during every re-run step's reverse; the kept one is let go before a checkpoint is
/// stored and at the sweep end. Does nothing with AD_SNAP_KEEP=1 (no copy is made there).
fn snap_reuse() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AD_SNAP_REUSE").as_deref() == Ok("1"))
}
/// drains (AD_SNAP_REUSE): copies made into a kept buffer, and buffers kept (receipt).
static SNAP_REUSED: (std::sync::atomic::AtomicU64, std::sync::atomic::AtomicU64) = (std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0));
fn free_beside() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AD_FREE_BESIDE").as_deref() == Ok("1"))
}
static FREE_BESIDE: std::sync::Mutex<Vec<std::thread::JoinHandle<()>>> = std::sync::Mutex::new(Vec::new());
/// drains: join the helper threads of AD_FREE_BESIDE (seconds waited).
pub(crate) fn free_beside_join() -> f64 {
    let t = Instant::now();
    let hs: Vec<_> = FREE_BESIDE.lock().map(|mut g| std::mem::take(&mut *g)).unwrap_or_default();
    for h in hs { let _ = h.join(); }
    t.elapsed().as_secs_f64()
}
fn held_skip_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AD_HELD_SKIP").as_deref() != Ok("0"))
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

/// Immutable plan metadata, compiled once per window. Both arms and every RK body are covered, as in block_blobs.
/// Step preparation now visits blob descriptors only, never the plan's kernel/copy/host command tree.
struct BlockBlobIndex(HashMap<String, Vec<std::sync::Arc<crate::plan::Blob>>>);
impl BlockBlobIndex {
    fn new(plan: &Plan) -> Self {
        // Shared sub-blocks must not duplicate inline blob payloads for every enclosing root. Intern by descriptor
        // identity, not hash: distinct descriptors for the same content can have different paths and error behavior.
        let mut descriptors: HashMap<*const crate::plan::Blob, std::sync::Arc<crate::plan::Blob>> = HashMap::new();
        Self(plan.blocks.keys().map(|name| {
            let (mut seen, mut want) = (std::collections::HashSet::new(), Vec::new());
            block_blobs(plan, name, &mut seen, &mut want);
            let mut hashes = std::collections::HashSet::new();
            let blobs = want.into_iter().filter(|b| hashes.insert(crate::blob_sha(b).to_owned()))
                .map(|b| descriptors.entry(b as *const _).or_insert_with(|| std::sync::Arc::new(b.clone())).clone()).collect();
            (name.clone(), blobs)
        }).collect())
    }
    fn get(&self, names: &[String]) -> Vec<crate::plan::Blob> {
        let mut hashes = std::collections::HashSet::new();
        names.iter().filter_map(|n| self.0.get(n)).flatten()
            .filter(|b| hashes.insert(crate::blob_sha(b).to_owned())).map(|b| b.as_ref().clone()).collect()
    }
}

/// Start session preparation before Win::new so page-locks overlap module/tape setup in both entry points.
unsafe fn start_staging(ex: &mut Execution, plan: &Plan, base: &Path, frame: &Option<ReplayFrame>) {
    let first = match (frame.as_ref(), plan.physics.as_ref()) {
        (Some(f), Some(ph)) => f.blocks_ahead(ph, 1, 1).into_iter().next(),
        _ => plan.cadence.first().cloned(),
    };
    let (mut seen, mut want) = (std::collections::HashSet::new(), Vec::new());
    for name in first.iter().flatten() { block_blobs(plan, name, &mut seen, &mut want); }
    let sizes = want.into_iter().filter_map(|b| match b {
        crate::plan::Blob::Gzip { bytes, .. } => Some(*bytes),
        crate::plan::Blob::Hex { bytes, .. } => Some(bytes.len() / 2),
        crate::plan::Blob::File { path, .. } => fs::metadata(crate::path(base, path)).ok().map(|m| m.len() as usize),
    }).filter(|n| *n > 0).collect();
    ex.start_staging_ahead(sizes);
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
/// batch). Proved on a test machine with AD_LW_REGROUP_TEST: mode 2 (a rotated pass, then the recorded layout) keeps every
/// reference step hash (S13-T2), mode 3 (recorded layout, then a rotated pass) gives 0 of 632,814 LW output words
/// different (S13-T3).
/// `events`: the live events at the section start (id, recorded create flags). Before each later pass the event set
/// is put back: events the section created are destroyed, events it destroyed are created again (THE BREAKAGES:
/// pass 2 of the i7-FtA h 0.03 row refused "event 1470151824 created while live" at step 7, and the test leg's
/// pass 2 refused "unknown event 1462729056" at step 1, an event made before the section and destroyed in it).
/// A section Allocate runs in pass 1 only: later passes keep the block, so earlier passes' scatters stay (step 1's
/// section allocates its pool blocks; the guard refused it in S13-T).
/// `outputs` (test mode 3 only): the scatter destinations (address, bytes) and their words after pass 1.
struct LwRegroup { start: usize, end: usize, address: u64, order: crate::plan::Slice, passes: Vec<Vec<i32>>, pass: usize, original: Vec<u8>, events: Vec<(u64, u32)>,
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

/// anyparam: one column order of the LW regroup as a command of the step list (an upload of int32 column indices
/// into the order array), so every later pass over that list (the tangent, the reverse's replays, the redo check)
/// repeats exactly what this forward did. THE BREAKAGE THIS FIXES: the regroup wrote the order with dev::upload, outside
/// the step list; the reverse of a run whose constants moved replayed the recorded order and refused at the histogram
/// read (box <local path>, step 16: "checked device-to-host value differs from the proven value").
fn lw_order_op(order: crate::plan::Slice, words: &[u8], what: &str) -> Operation {
    Operation::Upload { stream: 0, destination: order, data: crate::plan::Blob::Hex { bytes: crate::hex_bytes(words), sha256: crate::digest(words) },
                        pointers: Vec::new(), origin: Some(format!("live: LW column order ({what})")) }
}
/// anyparam: what a checked read-back guards, for the refusal: the kernels that produced the value.
fn read_producers(commands: &[Operation], at: usize) -> String {
    let mut names: Vec<&str> = Vec::new();
    for op in commands[..at].iter().rev() {
        match op {
            Operation::Kernel { function, .. } => { if !names.contains(&function.as_str()) { names.push(function); } if names.len() == 3 { break; } },
            Operation::NativeCubReduce { .. } | Operation::NativeCubSegmentedReduce { .. } => { if !names.contains(&"cub reduce") { names.push("cub reduce"); } },
            Operation::NativeThrustArgsort { .. } => { if !names.contains(&"thrust argsort") { names.push("thrust argsort"); } },
            Operation::DeviceToHost { .. } if !names.is_empty() => break,
            _ => {},
        }
    }
    let what = match names.first().copied() {
        Some(f) if f.starts_with("cupy_cub_any") => "a host any() test: the forward takes one branch when no element is set and another when one is",
        Some(f) if f.starts_with("hex_transport_species") => "the set of tracer levels that are all zero, which selects the transport launches",
        Some(f) if f.contains("scan") || f.contains("bsum") || f.contains("nonzero") => "a host count of selected columns, which sizes the launches after it",
        Some(f) if f.starts_with("mpas_sw") => "short-wave driver scalars read on the host (solar time and constants)",
        Some("thrust argsort") | Some("cupy_copy__int64_int32") => "the long-wave layer-count histogram, which sets the long-wave batches",
        _ => "a value the host reads to choose or size the launches after it",
    };
    format!("produced by {}; it is {what}. This package recorded one host choice for it, so a run on which it moves (other settings, another start state or date) cannot be replayed by this package: record the package at that state",
            if names.is_empty() { "an earlier command".to_owned() } else { names.join(" <- ") })
}
unsafe fn lw_regroup_begin(ex: &mut Execution, commands: &[Operation], at: usize, step: usize, hist: &[i32]) -> Result<(LwRegroup, Operation), String> {
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
    // anyparam: pass 1's order goes into the step list (the caller runs it), not straight to the card.
    let order0 = lw_order_op(order, &bytes0, &format!("pass 1 of {}", passes.len()));
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
    Ok((LwRegroup { start: at, end, address, order, passes, pass: 0, original, events, test: ex.lw_regroup_test, outputs }, order0))
}

/// S13: after the section's last command: start the next pass (true), or restore the live order (false).
unsafe fn lw_regroup_next(ex: &mut Execution, r: &mut LwRegroup, step: usize) -> Result<(bool, Vec<Operation>), String> {
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
    // anyparam: the event set put back and the next order are returned as commands for the step list (the caller
    // runs them through the same dispatch as plan commands), so the replays of this list do the same.
    let mut ops: Vec<Operation> = Vec::new();
    if r.pass < r.passes.len() {
        let mut extra: Vec<u64> = ex.events.keys().copied().filter(|id| !r.events.iter().any(|(e, _)| e == id)).collect();
        extra.sort_unstable();
        for id in extra { ops.push(Operation::Synchronization { stream: 0, operation: "EventDestroy".into(), event_handle_recorded: id, flags: 0 }); }
        for &(id, flags) in &r.events {
            if !ex.events.contains_key(&id) { ops.push(Operation::Synchronization { stream: 0, operation: "EventCreate".into(), event_handle_recorded: id, flags }); }
        }
        let b: Vec<u8> = r.passes[r.pass].iter().flat_map(|v| v.to_le_bytes()).collect();
        ops.push(lw_order_op(r.order, &b, &format!("pass {} of {}", r.pass + 1, r.passes.len())));
    } else {
        ops.push(lw_order_op(r.order, &r.original, "the live order put back"));
    }
    Ok((r.pass < r.passes.len(), ops))
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
/// with the card idle. Native derivative rows also launch on the null stream. Syncs stay where the host reads
/// device results, at TEMPO boundaries, and at every step and pass end. AD_ORDER_SYNC=1 keeps every drain.
pub(crate) fn ordered(ex: &Execution) -> bool {
    static FORCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *FORCE.get_or_init(|| std::env::var("AD_ORDER_SYNC").as_deref() == Ok("1") || entry_sync()) { return false; }
    ex.streams.len() == 1 && ex.streams.get(&0).is_some_and(|s| s.is_null())
}
/// x20-rest tl-entries-dycore: adjoint entries whose only output is the cotangent of a BY-VALUE launch scalar (dt, h4,
/// coefficient, scale, rgas/cp/p0, cf1-3, the glue dt/rv/rd/rvord), kept apart from the field reverse by the r5/L4/r7
/// splits. The window never reads those cotangents: every scalar cell is zeroed before the entry (walk_back) and nothing
/// after it reads `ad_cells` (deriv.rs binds each to a private 8-byte cell). Their bitwise ordered sums are the cost
/// (dycore-adjoint-rows-41k open item 1: w_edge_flux coefficient ~50 ms, handoff 4 x 44 ms, theta_filter_lap4 h4,
/// recover_terrain_w cf rows, ... per launch at 40,962 cells). So the window skips them; the per-launch dot harnesses
/// (dot.rs, reverse.rs) still run every entry and grade the scalar cotangents. An entry is skipped only if it is listed
/// here for its kernel AND its spec binding accumulates into a scalar cell; anything else runs. AD_WINDOW_SCALAR_SKIP=0
/// runs every entry (the old path; same words in every field: the skipped entries write only scalar cells and their own
/// workspace).
const SCALAR_COT_ENTRIES: &[(&str, &[&str])] = &[
    ("w_edge_flux_f32", &["ad_coef_t", "ad_coef"]),
    ("theta_filter_lap4_v841_f32", &["ad_h4_t", "ad_h4"]),
    ("w_filter_lap4_v841_f32", &["r5_h4_terms", "r5_h4_sum"]),
    ("momentum_filter_lap4_v841_f32", &["r5_hf_terms", "r5_hf_sum"]),
    ("recover_cells_f32", &["r5_scal_terms", "r5_scal_sum"]),
    ("theta_edge_flux_f32", &["r5_coef_terms", "r5_coef_sum"]),
    ("divergence_damping_f32", &["r5_coef_terms", "r5_coef_sum"]),
    ("smagorinsky_v841_f32", &["r5_sc_terms", "r5_sc_sum"]),
    ("pv_apvm_v841_f32", &["r5_scale_terms", "ad_scale"]),
    ("acoustic_ru_v841", &["ad_scalar"]),
    ("fct_edge_residual_v841", &["l4_dt_partial", "l4_dt_reduce"]),
    ("fct_vertical_low_order_v841", &["l4_dt_partial", "l4_dt_reduce"]),
    ("transport_edge_values_mono_v841", &["l4_coef_partial", "l4_coef_reduce"]),
    ("transport_standard_finish_v841", &["l4_dt_partial", "l4_dt_reduce"]),
    ("transport_target_density_v841", &["ad_dt_part", "ad_dt"]),
    ("transport_interpolate_target_v841", &["ad_weight"]),
    ("mpas_transport_handoff_v842_f32", &["ad_sum_dt", "ad_sum_rv", "ad_sum_rd", "ad_sum_rvord"]),
    ("mpas_transport_source_baseline_v842_f32", &["ad_sum_dt"]),
    ("mpas_rk_cells_v842_f32", &["ad_sum_rvord"]),
    // G6-inactive sinks: the rdzw chains whose reduce writes only the spec's g6_inactive_sink_rdzw workspace
    // ("fixed mesh constant: inactive (zero tangent in, cotangent discarded), GAPS G6"); nothing reads it.
    ("transport_standard_finish_v841", &["r5_rdzw_terms", "l4_rdzw"]),
    ("fct_vertical_low_order_v841", &["l4_rdzw_terms", "l4_rdzw_sum"]),
    ("transport_target_density_v841", &["ad_rdzw_part", "ad_rdzw"]),
    ("fct_finish_v841", &["ad_rdzw_part", "ad_rdzw"]),
];
fn scalar_skip_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AD_WINDOW_SCALAR_SKIP").as_deref() != Ok("0"))
}
/// True when the window skips adjoint entry `e` of `kd` (see SCALAR_COT_ENTRIES). The first decision per pair is logged.
/// A listed partial-sum entry (l4_*_partial, ad_dt_part: it writes only the partials its listed reduce reads) carries no
/// scalar binding of its own; it is skipped when a listed entry of the same kernel accumulates into a scalar cell.
/// AD_WINDOW_SCALAR_SKIP_EXCEPT=function:entry,... runs those listed entries anyway (bisection).
fn scalar_skip(kd: &deriv::KernelDeriv, e: &str) -> bool {
    if !scalar_skip_on() { return false; }
    let Some((_, listed)) = SCALAR_COT_ENTRIES.iter().find(|(f, es)| *f == kd.function.as_str() && es.contains(&e)) else { return false };
    let key = format!("{}:{e}", kd.function);
    static EXCEPT: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    if EXCEPT.get_or_init(|| std::env::var("AD_WINDOW_SCALAR_SKIP_EXCEPT").unwrap_or_default().split(',').filter(|x| !x.is_empty()).map(str::to_owned).collect())
        .iter().any(|x| *x == key || *x == kd.function) { return false; }
    // the listed group must end in a discarded cotangent: a scalar cell, or a G6-inactive sink workspace
    let scalar_in = |n: &str| kd.spec["launch_abi"]["entries"][n]["args"].as_array().is_some_and(|a| a.iter().any(|b|
        (b["bind"] == "adjoint" && b["scalar_arg"].is_string() && matches!(b["io"].as_str(), Some("accumulate" | "inout" | "out")))
        || (b["bind"] == "workspace" && b["g6_inactive"].is_object() && b["name"].as_str().is_some_and(|x| x.starts_with("g6_inactive_sink_")))));
    let binds_scalar = listed.iter().any(|n| scalar_in(n));
    static SEEN: std::sync::OnceLock<std::sync::Mutex<std::collections::BTreeSet<String>>> = std::sync::OnceLock::new();
    if SEEN.get_or_init(Default::default).lock().map(|mut s| s.insert(key.clone())).unwrap_or(false) {
        eprintln!("{}", json!({"event":"window_scalar_skip","entry":key,"skipped":binds_scalar,
            "why": if binds_scalar { "only output is a discarded cotangent (by-value launch scalar or G6-inactive sink)" } else { "listed but no listed entry of this kernel accumulates into a scalar cell: runs" }}));
    }
    binds_scalar
}
fn entry_sync() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // bs-tape: AD_G5_PROF_NOSYNC=1 keeps the host timers without the per-entry drains (a profile of the real ordering)
    *ON.get_or_init(|| (dev::prof::on() && std::env::var("AD_G5_PROF_NOSYNC").as_deref() != Ok("1")) || std::env::var("AD_ENTRY_SYNC").as_deref() == Ok("1"))
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
    // bs-tape (AD_G5_PROF only): the same time again by kind of command
    if t.is_some() {
        let k = match op { Operation::Kernel { .. } => 23, Operation::Upload { .. } | Operation::HeldUploadRanges { .. } | Operation::FrameUpload { .. } => 24,
                           Operation::DeviceCopy { .. } | Operation::Clear { .. } | Operation::TransportCopy { .. } | Operation::AddInplace { .. } => 25, _ => 26 };
        dev::prof::add(k, t);
    }
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

/// speed ckpt60: a whole boundary file read on 16 cores (the window end the dots read back after a sweep: one core
/// took 12 s for its 26.5 GB on box fast-1, leg h8). The same bytes.
fn read_whole(path: &Path, bytes: usize) -> Result<Vec<u8>, String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        let f = fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let n = f.metadata().map_err(|e| format!("{}: {e}", path.display()))?.len() as usize;
        if n != bytes { return Err(format!("{}: the file holds {n} bytes, the boundary {bytes}", path.display())); }
        let mut v = vec![0u8; n];
        let piece = n.div_ceil(16).max(1 << 20);
        let r: Vec<Result<(), String>> = std::thread::scope(|sc| {
            let hs: Vec<_> = v.chunks_mut(piece).enumerate().map(|(k, c)| { let f = &f; sc.spawn(move || f.read_exact_at(c, (k * piece) as u64).map_err(|e| format!("{}: {e}", path.display()))) }).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for x in r { x?; }
        Ok(v)
    }
    #[cfg(not(unix))]
    { let _ = bytes; fs::read(path).map_err(|e| format!("{}: {e}", path.display())) }
}
/// speed ckpt60: the redo compare of a step. `pre`: the counts redo_download already made against the content store's
/// truth (cs_redo), when the window runs with --bstore-content; otherwise the word-by-word compare with the stored bytes.
fn compare_stored(pre: Option<(u64, u64)>, index: &[(u64, usize, usize)], live: &[u8], want_all: &[u8], ranges: &[Vec<(usize, usize)>], step: usize) -> (u64, u64) {
    match pre { Some(c) => c, None => compare_boundary(index, live, want_all, ranges, step) }
}
/// 60 km tape (a12 SIXTYFAST; AD_CS_REDO_BESIDE=0 turns it off): the content store's redo check of a streamed reverse
/// step (cs_redo), made on the compare thread beside the walk-back instead of before it. THE WASTE THIS REMOVES: with
/// --bstore-content 1 the whole end state (26.5 GB at 60 km) was scanned through one 1 GiB page-locked window with
/// blocking copies, the card idle, before every walk-back: 1.9 to 2.3 s a 60 km step (a test machine s8-pub redo_download).
/// Here the same windows go down on the redo stream (non-blocking), which waits on the device for the re-run's last
/// command, into two halves of one 2 GiB buffer (the next window downloads while this one is hashed), and the same
/// Checker takes them in the same order. The walk-back writes only derivative and tape memory (why the word compare
/// already runs beside it); a state word it did write would make this check refuse the step, never change an answer.
/// The pieces, their SHA-256, the exact allocations' word compare and the refusal are cs_redo's.
struct CsBeside {
    step: usize, truth: std::sync::Arc<crate::cstore::Truth>, want: Vec<bool>,
    /// each window: (first piece, piece after the last, bytes, its device runs (address, bytes) in order)
    wins: Vec<(usize, usize, usize, Vec<(u64, usize)>)>,
    /// declared residue ranges of each allocation of the layout at this step
    res: Vec<Vec<(usize, usize)>>,
    pin: crate::pinned::PinnedBuf, stream: usize, evs: [usize; 3], ctx: usize,
    fns: (unsafe extern "C" fn(*mut c_void) -> i32, crate::cuda::Wait, unsafe extern "C" fn(*mut c_void, u64, usize, *mut c_void) -> i32, crate::cuda::Record, crate::cuda::PtrCall),
    dl_s: f64, all_s: f64, err: Option<String>,
}
/// 60 km tape: the redo check of CsBeside on the calling (compare) thread; (differing, residue) as cs_redo counts them.
unsafe fn cs_beside_run(c: &mut CsBeside) -> Result<(u64, u64), String> {
    let t0 = Instant::now();
    let (ctx_set, wait, dl_async, record, ev_sync) = c.fns;
    if ctx_set(c.ctx as *mut c_void) != 0 { return Err("the redo check beside the walk-back could not take the context".into()); }
    let (stream, ev) = (c.stream as *mut c_void, c.evs.map(|e| e as *mut c_void));
    checked("cuStreamWaitEvent redo beside", wait(stream, ev[0], 0))?;
    let half = CS_WIN;
    if c.pin.len() < 2 * half { return Err("the redo check beside the walk-back has no 2 GiB buffer".into()); }
    let base = c.pin.as_mut_ptr();
    let issue = |w: usize, wins: &[(usize, usize, usize, Vec<(u64, usize)>)]| -> Result<(), String> {
        let mut at = (w % 2) * half;
        for &(addr, n) in &wins[w].3 { checked("cuMemcpyDtoHAsync redo beside", dl_async(base.add(at).cast(), addr, n, stream))?; at += n; }
        checked("cuEventRecord redo beside window", record(ev[1 + w % 2], stream))
    };
    let mut ck = crate::cstore::Checker::new(&c.truth, std::mem::take(&mut c.want));
    let nw = c.wins.len();
    if nw > 0 { issue(0, &c.wins)?; }
    let mut dl = 0f64;
    for w in 0..nw {
        // the other half was fed (hashed) in the iteration before: free for the next window
        if w + 1 < nw { issue(w + 1, &c.wins)?; }
        let tw = Instant::now();
        checked("cuEventSynchronize redo beside window", ev_sync(ev[1 + w % 2]))?;
        dl += tw.elapsed().as_secs_f64();
        let (a, b, bytes, _) = &c.wins[w];
        ck.feed(*a, *b, std::slice::from_raw_parts(base.add((w % 2) * half), *bytes));
    }
    let (bad, live) = (std::mem::take(&mut ck.bad), std::mem::take(&mut ck.live));
    drop(ck);
    let truth = &c.truth;
    let (mut d, mut r) = (0u64, 0u64);
    let npieces = bad.len() as u64;
    for p in bad {
        let (i, off, n) = truth.lay.piece(p);
        eprintln!("{}", json!({"event":"window_redo_differs","step":c.step,"allocation":truth.lay.index[i].0,"first_byte":off,"bytes":truth.lay.index[i].2,
                               "piece_bytes":n,"by":"SHA-256 of the piece; the differing words inside it are not counted","beside_walk":true}));
        d += 1;
    }
    let mut ex_ids: Vec<usize> = live.keys().copied().collect();
    ex_ids.sort_unstable();
    for i in ex_ids {
        let (id, _, n) = truth.lay.index[i];
        let cmp = compare_boundary(&[(id, 0, n)], &live[&i], truth.exact(i).ok_or("content store: an exact allocation without its bytes")?, std::slice::from_ref(&c.res[i]), c.step);
        d += cmp.0; r += cmp.1;
    }
    if d != 0 { eprintln!("{}", json!({"event":"window_redo_differs_total","step":c.step,"pieces_whose_sha256_differs":npieces,"differing_words_in_word_compared_allocations":d - npieces,
                                       "counted_as":"each differing piece counts one in the refusal's number; the words inside it are not counted","beside_walk":true})); }
    c.dl_s = dl; c.all_s = t0.elapsed().as_secs_f64();
    Ok((d, r))
}

/// A whole-device checkpoint at a step boundary: bytes on NVMe, host side in RAM. `delta` (S02, --checkpoints): the
/// file holds only the chunks that differ from boundary 0 (delta_encode); `file_bytes` is its size on disk.
#[derive(Clone)]
struct Boundary { path: PathBuf, index: Vec<(u64, usize, usize)>, host: HostSnap, step: usize, delta: bool, file_bytes: usize,
                 /// runner-lib: the bytes in pinned host RAM instead of the file at `path` (None: the file)
                 ram: Option<std::sync::Arc<capi::HostBytes>>,
                 /// runner-lib: a reverse-step anchor held on the device at this address (0: host bytes)
                 dev: u64,
                 /// speed ckpt60: the bytes are in the content store (cstore.rs) or on the device, not in a file or a whole RAM copy
                 cs: Option<CsB>,
                 /// step-hold (--step-hold FILE): a held step-start state. Entry i of `index` is then a span of its
                 /// allocation that starts `sub[i]` bytes into it (an allocation may have several entries); None: every
                 /// entry is a whole allocation, as before.
                 sub: Option<std::sync::Arc<Vec<usize>>> }
/// step-hold: the declared TEMPO residue ranges of each redo entry (bytes from the entry's start). A whole
/// allocation's are its own ranges, as before; a held span's are the parts inside the span, moved to its start.
fn redo_ranges(ex: &Execution, cur: &Boundary, cidx: &[(u64, usize, usize)], step: usize) -> Vec<Vec<(usize, usize)>> {
    let subs: Option<HashMap<usize, usize>> = cur.sub.as_ref().map(|s| cur.index.iter().zip(s.iter()).map(|(e, &u)| (e.1, u)).collect());
    cidx.iter().map(|(id, off, n)| {
        let r = ex.residue.get(&(step, *id)).map(|x| x.0.clone()).unwrap_or_default();
        match &subs { None => r, Some(m) => crate::step_hold::shift_ranges(&r, m.get(off).copied().unwrap_or(0), *n) }
    }).collect()
}
/// step-hold: the card side of boundary entries `es` (all of `b_index`, or the subset redo_index gives):
/// (allocation, offset in the boundary's bytes, bytes, device address). Every path that moves a boundary's bytes
/// between host and card (store_hold, restore, both redo downloads, differing) takes its addresses from here, so a held
/// span lands at its own offset in its allocation on every path or on none, and the CPU and home-card tests
/// (step_hold_tests) run these very addresses. `b_sub` is Boundary::sub (None: every entry is a whole allocation, at
/// its start, as before). `sized`: refuse a whole entry whose allocation changed size, as restore did; a held span is
/// always refused unless it lies inside its allocation.
fn entry_copies(b_index: &[(u64, usize, usize)], b_sub: Option<&[usize]>, es: &[(u64, usize, usize)],
                alloc: impl Fn(u64) -> Option<(u64, usize)>, sized: bool) -> Result<Vec<(u64, usize, usize, u64)>, String> {
    let subs: Option<HashMap<usize, usize>> = b_sub.map(|s| b_index.iter().zip(s.iter()).map(|(e, &u)| (e.1, u)).collect());
    es.iter().map(|&(id, off, n)| {
        let (addr, bytes) = alloc(id).ok_or("boundary allocation not live")?;
        let (u, fits) = match &subs {
            None => (0, !sized || bytes == n),
            Some(m) => { let u = *m.get(&off).ok_or_else(|| format!("--step-hold: entry at {off} of allocation {id} is not in the boundary"))?;
                         (u, u.checked_add(n).is_some_and(|e| e <= bytes)) },
        };
        if !fits { return Err(format!("allocation {id} changed size since the boundary")); }
        Ok((id, off, n, addr + u as u64))
    }).collect()
}
/// entry_copies over the live allocations of `ex`.
fn ex_copies(ex: &Execution, b: &Boundary, es: &[(u64, usize, usize)], sized: bool) -> Result<Vec<(u64, usize, usize, u64)>, String> {
    entry_copies(&b.index, b.sub.as_ref().map(|s| s.as_slice()), es, |id| ex.allocations.get(&id).map(|a| (a.address, a.bytes)), sized)
}
/// restore's blocking upload of `src` (a boundary's bytes) along `cps` (entry_copies). Zero-byte entries are sent as
/// before.
unsafe fn upload_copies(cuda: &Cuda, src: &[u8], cps: &[(u64, usize, usize, u64)]) -> Result<(), String> {
    for &(_, off, n, at) in cps { checked("cuMemcpyHtoD restore", (cuda.upload)(at, src[off..off + n].as_ptr().cast(), n))?; }
    Ok(())
}
/// redo_download's blocking download of the live words along `cps` (entry_copies) into `dst`, in the boundary's layout.
unsafe fn download_copies(cuda: &Cuda, dst: &mut [u8], cps: &[(u64, usize, usize, u64)]) -> Result<(), String> {
    for &(_, off, n, at) in cps { if n != 0 { checked("cuMemcpyDtoH compare", (cuda.download)(dst[off..off + n].as_mut_ptr().cast(), at, n))?; } }
    Ok(())
}
/// step-hold: whether entry `i` of `b.index` (`n` bytes) fits allocation `a`: a span must lie inside it, a whole
/// entry must be exactly its size (as before).
fn entry_fits(b: &Boundary, i: usize, a_bytes: usize, n: usize) -> bool {
    match &b.sub { None => a_bytes == n, Some(s) => s[i].checked_add(n).is_some_and(|e| e <= a_bytes) }
}
/// speed ckpt60: where a content-store boundary's bytes are. State: its pieces in host memory (a checkpoint). Dev: only
/// on the device, right now: the recomputed start state of the step about to be reversed, numbered so that restore can
/// tell it is still there (cs_dev_at); nothing else may ask for its bytes.
#[derive(Clone)]
enum CsB { State(std::sync::Arc<crate::cstore::State>), Dev(u64) }
impl CsB { fn key(&self) -> (u8, usize) { match self { CsB::State(s) => (0, std::sync::Arc::as_ptr(s) as usize), CsB::Dev(n) => (1, *n as usize) } } }
/// speed ckpt60: one window of the content store's scans (the small page-locked buffer every state moves through).
const CS_WIN: usize = 1 << 30;

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
/// speed storm3: seconds of a checkpointed sweep outside its reverse steps, by part: [checkpoint restore before a
/// re-run, the re-run steps, the download of the re-run's end state, the copy of that state, the checkpoint
/// stores, the checkpoint drops]. Printed once a sweep as window_sweep_parts; a receipt of time, nothing reads it.
static SWEEP_NS: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];
/// speed storm3: the pin-pool key of a checkpointed CLI window (0: none). THE WASTE THIS REMOVES: with
/// --checkpoints and the checkpoints in host memory, every store made a new page-locked buffer of the whole state
/// (10.53 GB at 3 km: first touch of every page, then 157 driver registrations) and every dropped checkpoint
/// unregistered and unmapped one: 50 stores and 41 drops in the 59-step 3 km storm window, most of the 85 s its sweep
/// spent outside the reverse steps, the re-run steps and the restores (speed/fast/storm3 runs/fast-4/o59-c-p2copy:
/// sweep 520.1 s, reverse steps 380.6 s, the four counted parts 54.4 s). With a key the store takes a dropped
/// checkpoint's buffer from the process pin pool (capi.rs, as the C-ABI session does) and a dropped one goes back
/// there; when a sweep ends the idle ones are freed (capi::pin_pool_for_checkpoints). Only which page-locked bytes a
/// state lands in changes. AD_BSTORE_POOL=0 keeps a new buffer per store (the pair's base leg).
static BS_POOL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// (2026-11-13) a resident program (--resident) keeps the checkpoint pool's idle page-locked
/// buffers at the sweep's end for its next request instead of freeing them (AD_PIN_KEEP=0 frees them as before).
/// THE COST THIS REMOVES: the second gradient of a kept-loaded 12-step 3 km storm-box program page-locked 5 buffers,
/// 52.7 GB, 3.17 s, inside its first reverse step again (capi.rs HostBytes::pooled; a test machine's stderr, pin_made),
/// and 0.6 s more in its seventh. The same buffers hold the same bytes; only when they are made changes. Host memory:
/// the idle buffers stay page-locked between requests, within the pool's cap (the largest session RAM budget seen).
static PIN_KEEP_RESIDENT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// AD_BS_PREFILL=N (default 0: off) makes up to N page-locked buffers of the plan's live-bytes bound for the
/// checkpoint pool on one helper thread while the forward runs, less the idle buffers the pool already holds, within
/// the pool's RAM budget. THE COST THIS MOVES: the first gradient of a 12-step 3 km storm-box program page-locked 8
/// buffers, 60.7 GB, 3.90 s, inside its first reverse step (capi.rs HostBytes::pooled; a test machine, pin_made).
/// The reverse takes them from the pool (the smallest that fits) as it took its own; the same bytes are written.
/// kfstore (the speed lane's ad/kfstore, 14 Nov): `kept` more buffers (the forward checkpoints keep-forward will move
/// aside in this request, AD_KEEP_FORWARD on, first forward of a start; 0 otherwise) are made the same way, within the
/// budget plus their room. THE COST THIS MOVES: with the forward's checkpoints kept, the sweep's own checkpoint stores no
/// longer find the dropped forward buffers in the pool and page-locked 12 new buffers inline on the main thread,
/// 102.8 GB, 26.84 s, in the first gradient of a fresh program (59 steps, the speed lane's s9b-b-r1; 1.55 s without
/// keep-forward). AD_KF_PREFILL=1 turns it on; unset, the engine is as before.
unsafe fn bs_prefill(ex: &Execution, bound: usize, budget: usize, kept: usize) {
    let kept = if std::env::var("AD_KF_PREFILL").as_deref() == Ok("1") { kept } else { 0 };
    let want: usize = std::env::var("AD_BS_PREFILL").ok().and_then(|v| v.parse().ok()).unwrap_or(0) + kept;
    let budget = budget + kept * capi::pin_cap(bound);
    let pool = bs_pool();
    if want == 0 || pool == 0 || bound == 0 { return; }
    let cap = capi::pin_cap(bound);
    let (idle_n, idle_b) = capi::pin_pool_idle(pool);
    let n = want.saturating_sub(idle_n).min(budget.saturating_sub(idle_b) / cap);
    if n == 0 { eprintln!("{}", json!({"event":"window_bs_prefill","buffers":0,"idle_buffers":idle_n})); return; }
    let (fns, set, c) = (crate::pinned::PinFns::of(&ex.cuda), ex.cuda.ctx_set_current, ex.context as usize);
    let _ = std::thread::Builder::new().name("bs-prefill".into()).spawn(move || {
        if set(c as *mut std::ffi::c_void) != 0 { return; }
        let t = Instant::now();
        let mut made = 0usize;
        for _ in 0..n {
            let Ok(m) = crate::pinned::PinMem::new(fns, cap) else { break };
            capi::pin_pool_give(pool, m); made += 1;
        }
        eprintln!("{}", json!({"event":"window_bs_prefill","buffers":made,"buffer_bytes":cap,"idle_buffers":idle_n,"seconds":t.elapsed().as_secs_f64()}));
    });
}
fn bs_pool() -> usize { BS_POOL.load(std::sync::atomic::Ordering::Relaxed) }
fn sweep_add(part: usize, since: Instant) {
    SWEEP_NS[part].fetch_add(since.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
}
/// S02: the reverse sweep's carried state: the window state at the step being reversed (`cur`), the rows, counters.
struct RevSt { cur: Option<(Boundary, Bytes)>, rows: Vec<Value>, tag: Value }
/// host-leftovers-2: a loaded boundary's bytes: read from a file (owned), or the page-locked RAM copy itself (shared,
/// no copy). Either way the same bytes.
pub(crate) enum Bytes { Own(Vec<u8>), Ram(std::sync::Arc<capi::HostBytes>), Pin(crate::pinned::PinnedBuf) }
impl std::ops::Deref for Bytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] { match self { Bytes::Own(v) => v, Bytes::Ram(r) => r.as_slice(), Bytes::Pin(p) => p } }
}
impl From<Vec<u8>> for Bytes { fn from(v: Vec<u8>) -> Self { Bytes::Own(v) } }

/// x20 fused tape: a tape entry before its forward command ran (an Entry still owes its post-launch part).
/// revmach prefetch: the upload of chunk `ci` into the arena, piece by piece from the top, each piece issued on the copy
/// stream once the walk-back has finished every row that could still read the bytes it overwrites (an event recorded on
/// the legacy stream after that row's reverse). `pending` holds (start, bytes) highest first; `done` the completion
/// events the next reverse step's stream waits on before its first command.
struct FzPrefetch { ci: usize, base: u64, host: *const u8, offs: Vec<usize>, pending: Vec<(usize, usize)>, done: Vec<*mut std::ffi::c_void>,
                    stream: *mut std::ffi::c_void, issued_during_walk: usize, bytes: usize,
                    /// speed tape60 (host-shared tape blocks): when set, the chunk's bytes by run (arena offset, bytes, host
                    /// address), in arena order and covering the whole chunk; `host` is then unused.
                    runs: Option<Vec<(usize, usize, usize)>> }
/// revmach early checks: the word checks of one recorded launch, queued right after its forward ran and copied to
/// page-locked memory by an asynchronous readback (event `fz_wev`), with the spans and candidates they were made for.
struct FzWords { posts: Vec<(u64, (usize, usize))>, cands: Vec<(Option<(usize, u64)>, Option<(usize, u64)>)>, nsh: usize, nslot: usize }
enum FzPre {
    Entry { f: String, b: deriv::Bound, args: Vec<submit::Argument>, grid: [u32; 3], block: [u32; 3], shared: u32 },
    Done(Tp),
}
/// Resolved launch from the live operation contract; no recording paths or payloads.
pub(crate) type LiveLaunch = (String, Vec<submit::Argument>, [u32; 3], [u32; 3], u32);

pub(crate) enum Tp {
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
/// memory: `Full(i, rows)` (streamed tape only): the arena cannot hold command `i`; `rows` are the taped
/// commands before it, the live state is the state right before command `i` (it has not run).
enum Taping { Done(Vec<Tp>), Overflow(usize), Full(usize, Vec<Tp>) }
/// memory (streamed tape): page-locked chunk buffers the helper thread keeps ready ahead of use.
const ST_AHEAD: usize = 2;
/// speed tape60 (host-shared tape blocks): the proof mode (AD_TAPE_SHARE_VERIFY=1); block bytes.
/// 60 km tape (a12 SIXTYFAST): ON unless AD_TAPE_SHARE=0 (sixtyfast lead, shift 2: the off default of 336f163f6 is put back to
/// on: its built program gave no step gain at 60 km (t-c2 median 38.15 against t-c1 38.36 s, mean 39.59 against 38.48 s;
/// t-c2r8 median 38.33 s) and cost 5 GiB more host and 23 GB more copied a step; AD_TAPE_SHARE=0 keeps it off). WHAT OFF REMOVES: at 60 km (a test machine card 1, 8-step
/// leg) the sharing's block digests cost 0.43 s of every re-run (chunk flush 0.50 -> 0.07 s a step) and its 140,000
/// upload calls a step held the walk-back's host thread 5.4 s a step in the upload issue while the card was busy;
/// without it the backward step was 38.36 -> 37.72 s median (t-c1 -> t-c1ns), same answer bytes, for 23 GB more
/// copied out a step and 5 GiB more host memory (190.1 -> 195.3 GiB).
fn st_share_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AD_TAPE_SHARE").as_deref() != Ok("0"))
}
fn st_trim_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AD_TAPE_STREAM_TRIM").as_deref() != Ok("0"))
}
fn st_share_verify() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AD_TAPE_SHARE_VERIFY").as_deref() == Ok("1"))
}
fn st_share_block() -> usize {
    static B: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *B.get_or_init(|| std::env::var("AD_TAPE_SHARE_KIB").ok().and_then(|v| v.parse::<usize>().ok()).filter(|k| *k >= 4).unwrap_or(64) << 10)
}
fn exhausted(e: &str) -> bool { e.contains("derivative arena exhausted") }
/// memory (AD_TL_BIND_SCRATCH=1): the MYNN-EDMF tangent bind takes its shadow and snapshot regions from the
/// reverse scratch (see the tangent's native row and `tl_scratch_room`).
fn tl_bind_scratch() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AD_TL_BIND_SCRATCH").as_deref() == Ok("1"))
}
/// memory: the reverse scratch is the larger of workspace and one launch's transient (`max`) instead of their
/// sum. AD_REV_SCRATCH_FIT=max asks for it, =sum refuses it; unset, it follows AD_TL_BIND_SCRATCH (whose scratch
/// already has that size when the reverse starts).
fn rev_fit_max() -> bool {
    // s60-cardmem: AD_CARD_ROOM=1 asks for `max` too (60 km b8-head: 13.55 -> 9.53 GB for a 9.19 GB peak)
    match std::env::var("AD_REV_SCRATCH_FIT").as_deref() { Ok("max") => true, Ok("sum") => false, _ => tl_bind_scratch() || crate::card_room::on() }
}

struct GatherDev { injective: bool, bufs: [u64; 5], nsrc: i64, ndst: i64, direct: Vec<DevRow>, fan: Vec<DevRow>, fan_groups: Vec<i64> }

pub struct Win {
    join_one: Option<join_one::JoinOne>,
    /// anyparam: commands the forward made itself (LW regroup orders, the live histogram read, event put-backs),
    /// owned here for the whole window; the step lists point at them. Empty when the run follows the recording.
    synth: Vec<Box<Operation>>,
    /// anyparam: one row per LW regroup a forward of this window followed (receipt key live_trajectory).
    lw_follow: Vec<Value>,
    /// multidevice-runner: reused device buffer (address, bytes) for received cotangents (peer_tmp).
    peer_buf: (u64, usize),
    /// ranks: derivative guard words tolerated in this rank process (AD_PEER_TOLERATE_GUARDS=1): "pass function" -> [launch
    /// items fired, words fired], and the distinct sources seen.
    peer_tolerated: BTreeMap<String, (u64, u64, Vec<String>)>,
    /// Optional package-bound test inputs for cross-layout gates; ordinary seeds are unchanged.
    vectors: Option<window_vectors::Pair>,
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
    /// lane tl-host-syncs: the smallest page-locked boundary buffer made ahead (the plan's live-bytes bound); see pin_prefill.
    pin_min: usize,
    /// lane tl-host-syncs (AD_SYNC_SITES=1 only): the rule label of the command being walked, for the per-label host tally.
    last_label: String,
    /// x20: the CLI tangent pass joins the copy-row gathers at its end instead of at every step.
    gather_pass_join: bool,
    /// adjoint-speed: (pass, function) -> (entry wall ms, calls), from the entry timers that already exist.
    fn_ms: BTreeMap<(String, String), (f64, u64)>,
    /// memory: (pass, function) -> (scratch bytes in use when its largest launch ended, arena + spill slabs;
    /// tangent tape bytes of its largest launch). THE BREAKAGE THIS PREVENTS: the receipt gave one scratch peak and one
    /// tape size for the window, so sizing another mesh or deck meant scaling a sum by cell count without knowing which
    /// launch made it (x1.163842: a 13.55 GB scratch for a 9.19 GB peak; the 3 km box: an 11.35 GiB tangent peak
    /// nobody could attribute). Host bookkeeping in `note`, which every entry already calls: no device call, no wait.
    arena_use: BTreeMap<(String, String), (usize, usize)>,
    /// (scratch resets, tape resets) at the last launch counted in arena_use.
    arena_noted: (u64, u64),
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
    /// revmach: the copy-out stream of the fused tape's chunk flush, every event it made, and per chunk the event after
    /// its last piece (a chunk's bytes are complete in RAM once it has fired).
    fz_copy: *mut std::ffi::c_void, fz_evs: Vec<*mut std::ffi::c_void>, fz_done: HashMap<usize, *mut std::ffi::c_void>, fz_h2d_s: f64, fz_fence_waits: u64,
    /// machine: host wall of the walk-back by part since the last reverse row: [entries, rules, native schemes,
    /// TEMPO, drains owed before a row, tape-upload issue, drain at the end of a chunk] and the row counts of the
    /// first four. Launches are asynchronous, so card time shows where the host next waits (the two drains, or a
    /// blocking call inside a row); nothing here adds a wait.
    wk_s: [f64; 7], wk_n: [u64; 4],
    /// revmach prefetch: per fused step, each tape row's first arena offset (rows only ever reference offsets at or above
    /// their own first offset's predecessors: after the reverse of row j, the arena from offs[j] up is dead); the offsets of
    /// the step being recorded; and the upload of the next chunk issued piecewise during the current walk-back.
    fz_offs: HashMap<usize, Vec<usize>>, fz_cur_offs: Vec<usize>, fz_pf: Option<FzPrefetch>, fz_pf_info: Vec<Value>,
    /// revmach early checks: the recording's pre-launch part while its tangent runs, the page-locked word buffer and its
    /// event, and the checks queued right after the forward launch (read after the tangent entries are queued).
    fz_pre: Option<FzPre>, fz_wbuf: Option<crate::pinned::PinnedBuf>, fz_wev: *mut std::ffi::c_void, fz_words: Option<FzWords>, fz_early: u64,
    /// memory (streamed tape; AD_TAPE_STREAM=1 or --tape-stream 1): a reverse step whose tape does not fit the
    /// arena is replayed once and taped in chunks; a full chunk goes to page-locked RAM and comes back for the
    /// walk-back (see stream_reverse_step). `seg_offs`: the arena offset each row of the current tape_range began at.
    /// `st_pool`: chunk buffers kept for the next step; `st_ahead`: the helper thread making them.
    st_on: bool, seg_offs: Vec<usize>, st_pool: Vec<crate::pinned::PinnedBuf>, st_ahead: Option<AheadPool>, st_info: Vec<Value>, st_made: usize, st_planned: bool, st_ahead_bytes: usize,
    /// speed tape60: how many page-locked buffers the helper keeps made ahead of use: st_plan's count before the first
    /// reversed step, then what the step in hand is still expected to take (at most ST_AHEAD), 0 once it needs none.
    st_want: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// speed tape60: the look-ahead of the first reversed step (st_plan).
    st_first: usize,
    /// bs-tape: each step's tape estimate (index step - 1), from st_plan, for st_trim.
    st_est: Vec<usize>,
    /// bs-tape: host addresses of the chunk buffers borrowed from the checkpoint pool for the current step.
    st_borrowed: Vec<usize>,
    /// packed-tape (default on; AD_TAPE_PACK=0 keeps the old tape): transient buffers bound from the scratch at
    /// walk-back, pre-launch copies shared when word-equal (deriv::PreCache). `rows`: per-row tape bytes of the current
    /// tangent step, function -> [launches, pre bytes, meta bytes, transient bytes, post bytes, other tape bytes].
    /// `rev_scratch`: the largest transient bytes of one launch; the reverse scratch holds them on top of its old size
    /// (`scratch_base`, which already held every launch's workspace).
    /// `pack_ratio`: the largest taped / estimated bytes of a packed one-pass step so far (0 before the first).
    pack: bool, precache: deriv::PreCache, rows: BTreeMap<String, [usize; 6]>, rev_rows: BTreeMap<String, [usize; 8]>, seg_rows: BTreeMap<String, [usize; 8]>, census: Option<Census>, win: Option<dev::Arena>, win_cap: usize, win_promoted: (usize, usize), rev_scratch: usize, scratch_base: usize, scratch_cli: usize, scratch_fit: Option<Value>, pack_ratio: f64, seg_ratio: f64,
    /// step6-sw: element type of each cupy_concatenate_same_size module (wrules::concat_module_types).
    pub concat_types: HashMap<String, String>,
    kernels: HashMap<String, deriv::KernelDeriv>,
    /// First admitted segment only; every other row keeps its existing GPU path.
    graph_segment: bool,
    segment_graphs: crate::reverse_machinery::GraphCache,
    graph_tape_epoch: u64,
    graph_scratch_epoch: u64,
    classes: HashMap<String, String>,
    schemes: HashMap<String, String>,
    refuse: HashMap<String, String>,
    /// D1: how each kind named with --refuse ran (held on what it writes, or identity kept because its writes are unknown).
    refuse_how: BTreeMap<String, String>,
    base_el: HashMap<u64, i64>,
    sh_base: u64,
    words: i64,
    shadows: BTreeMap<u64, deriv::Shadow>,
    support: dev::Support,
    /// readbacks-device : the step ledger. Every derivative entry's failure slots, the
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
    /// Some while the reverse taping counts the labels itself (a window with no tangent pass): the (step, command)
    /// pairs counted so far. See rev_count_once.
    rev_count: Option<std::collections::HashSet<(usize, usize)>>,
    labels: BTreeMap<String, u64>,
    /// The kernels that ran as identity under the kernel-set class "held" (wrules.rs label "held:<scheme>"), by label.
    /// THE BREAKAGE THIS PREVENTS: window.json named the frozen limiter and the refuse list as all the window leaves out
    /// of the derivative, while two more commands ran as identity under this class in every 120 km and 60 km leg and
    /// were counted nowhere (commands_by_rule held:dycore 1, held:mynn_sfc 1; long60 checker 2026-11-11T07:09Z).
    held_kernels: BTreeMap<String, std::collections::BTreeSet<String>>,
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
    /// bs-tape (a12 backstep): the recomputed start state the card holds right now, as (step, address of its host copy),
    /// set by rev_range after snap_ram and taken by the next restore (see restore).
    held_prev: Option<(usize, usize)>,
    /// bs-tape: restores whose upload was skipped because the card already held every byte (count, bytes not sent).
    held_skips: (usize, usize),
    /// bs-tape: the redo download's own copy stream and its two events (start: the re-run's end; done: the copy's end).
    redo_stream: *mut std::ffi::c_void, redo_ev: [*mut std::ffi::c_void; 2],
    /// a12 bs-tape: the asynchronous restore's copy stream, events (start, done), whether one is in flight, the event wait,
    /// and how many restores went that way.
    xfer_stream: *mut std::ffi::c_void, xfer_ev: [*mut std::ffi::c_void; 2], xfer_pending: bool, xfer_sync: Option<crate::cuda::PtrCall>, restore_async_n: usize,
    /// --fct-frozen 1 (default off): frozen FCT limiter coefficients (see wrules.rs `fct_frozen`).
    fct_frozen: bool,
    /// --mesh-constants 1 (S01b, default off): mixing_scaling_f32's del2/del4 are functions of the fixed mesh density only:
    /// R::Zero on its outputs (wrules.rs `mesh_scaling`).
    mesh_constants: bool,
    /// Taylor check: directions skip address-bearing allocations (their pointer words must not be perturbed).
    taylor_mask: bool,
    /// a development line: a Taylor row's forward is running with the recording as its reference (AD_TAYLOR_ROW_CHECK=1,
    /// diagnostic): the step keeps everything a rerun does (the live LW histogram) though references are given.
    row_live: bool,
    /// truth: AD_HOLD_MCICA / AD_HOLD_OUT, default off (see `Hold`).
    hold: Hold,
    address_bearing: std::collections::HashSet<u64>,
    /// --native-schemes 1: the L2 physics rows (AD_SCHEMES), live through the float32-shadow adapter.
    native: Option<crate::schemes::Native>,
    /// --tempo-live 1 (ad/held-tempo_warp_steps): TEMPO's tempo_warp_steps derivative (the L5 tempo_adw image, the
    /// one the per-launch dot proves) live in the composed operator, with its own work arena (--tempo-arena-gib, 6).
    tempo: Option<(crate::tempo_adw::TempoAdw, dev::Arena)>,
    tempo_shards: Option<crate::tempo_shard::Pool>,
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
    /// drains (AD_SNAP_REUSE, see snap_reuse): the pageable buffer of the current state the last reverse step let go.
    snap_spare: Option<Vec<u8>>,
    /// keep-forward (AD_KEEP_FORWARD): the steps of the forward's checkpoints the sweep moves aside instead of dropping
    /// (empty: none, as before), and the ones moved aside so far in this request.
    kf_keep: std::collections::BTreeSet<usize>,
    kf_saved: BTreeMap<usize, Boundary>,
    /// anchors-ahead: the page-locked anchor buffers a helper thread makes during the tangent pass (made so far, helper
    /// done), the main-thread seconds spent waiting for one, how many were taken, and the plan (receipt).
    anchor_ahead: Option<AheadPool>, anchor_ahead_wait_s: f64, anchor_ahead_taken: u64, anchor_ahead_info: Vec<Value>,
    restore_s: f64,
    /// G5: background decodes in flight (the blob hashes each holds, the decoder). integ8: one batch per prefetched
    /// step, so a step waits only for the batch that holds a blob it needs.
    blob_index: BlockBlobIndex,
    blob_next: Vec<(std::collections::HashSet<String>, std::thread::JoinHandle<Result<Vec<(String, Vec<u8>)>, String>>)>,
    /// integ8: blobs the next forward steps need (decoded ahead): stage_blobs keeps them in the cache.
    blob_ahead: std::collections::HashSet<String>,
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
    /// longwin: the step whose forward first made each allocation (0: live before step 1), filled by every
    /// forward step. The redo compare reads it: see redo_index.
    born: HashMap<u64, usize>,
    /// S02: background read + decode of the checkpoint the reverse sweep loads next (path, bytes); seconds spent in load
    /// (incl. waiting for a prefetch) and store; prefetch hits.
    ck_pf: Option<(PathBuf, std::thread::JoinHandle<Result<Vec<u8>, String>>)>,
    load_s: f64, store_s: f64, pf_hits: u64,
    /// speed ckpt60 (--bstore-content): checkpoints are states of the content store, the redo check compares with a
    /// truth (cstore.rs). `cs_refs`: the states captured so far that are still held somewhere (the newest one alive is
    /// the next capture's reference). `cs_dev_at`: the boundary whose state the device holds right now, set by the sweep
    /// straight after a capture or a recompute and taken by the next restore, which then uploads nothing when it is
    /// asked for that very boundary. `cs_pin`: the one small page-locked buffer. `cs_truth`: (step, truth) of the end
    /// state the next redo is compared with; `cs_end_truth`: the window end's; `cs_redo_res`: the counts of the redo
    /// compare made by redo_download, taken by the step's compare. Counters: captures, restores uploaded, restores not
    /// needed, truths read from the device, truths read from a state, redo compares; seconds: scan downloads, state
    /// build, restore fill, restore upload, truth, redo compare.
    pub(crate) cs_on: bool,
    cs_refs: Vec<std::sync::Weak<crate::cstore::State>>,
    cs_dev_at: Option<(u8, usize)>,
    cs_serial: u64,
    cs_pin: Option<crate::pinned::PinnedBuf>,
    cs_truth: Option<(usize, std::sync::Arc<crate::cstore::Truth>)>,
    cs_end_truth: Option<std::sync::Arc<crate::cstore::Truth>>,
    cs_redo_res: Option<(u64, u64)>,
    cs_n: [u64; 6], cs_s: [f64; 6], cs_exact_peak: usize, cs_spares: u64,
    /// speed live (--bstore-live 1, with --bstore-content 1): the allocations a state at an intermediate boundary
    /// (1 to steps - 1) leaves out: those the plan does not list as carried (plan.carried; the producer certifies them
    /// scratch, overwritten before read). They are not stored, not compared by the redo check, and zero-filled on a
    /// restore (the bytes exec gives a mid-step Allocate). `cs_cut_steps`: the window's steps (boundary 0 and the window
    /// end stay whole: the dots read them). `cs_plant_ask` / `cs_plant`: AD_PLANT_LEFT_OUT, TEST ONLY (plant_left_out):
    /// one CARRIED allocation also left out, still compared wherever a truth holds it. Counters: states cut, bytes left
    /// out of them, bytes zero-filled on restores; seconds zero-filling.
    cs_cut: Option<std::collections::HashSet<u64>>, cs_cut_steps: usize, cs_plant_ask: Option<Option<u64>>, cs_plant: Option<u64>,
    cs_cut_n: [u64; 3], cs_cut_s: f64,
    /// host-leftovers-2: CLI boundaries kept in page-locked RAM instead of b{s}.bin (budget, bytes held now, peak, count).
    bs_ram_budget: usize, bs_ram_now: usize, bs_ram_peak: usize, bs_ram_count: u64,
    /// host-leftovers-2: restores DMA'd straight from page-locked bytes (no staging copy).
    pub(crate) restore_direct: u64,
    /// host-leftovers-2: seed-weight prep per (boundary path, boundary step, allocation, stable, inv, fwd): the merged
    /// unstable byte ranges and the field rms (see scale_chunk). The boundary bytes at a path and step do not change.
    wcache: std::sync::Mutex<HashMap<(PathBuf, usize, u64, bool, bool, bool), std::sync::Arc<(Vec<(usize, usize)>, f32)>>>,
    /// machine: page-locked chunk buffers of the dot test's direction weights (weights_piped), made on first use.
    wpins: std::sync::Mutex<Vec<crate::pinned::PinnedBuf>>,
    /// speed outside: page-locked chunk buffers of the dot's per-allocation split (AD_DOT_ALLOC), made on first use.
    dpins: std::sync::Mutex<Vec<crate::pinned::PinnedBuf>>,
    /// lane rung3-params: --params FILE, the parameter carrier (P4 module globals); empty without the flag.
    params: Vec<Par>,
    /// tune mem: --checkpoints with --params-unchecked 1: a recomputed stretch (advance) is not compared with the
    /// recorded reference, as that run's base forward is not. False in every other run.
    recompute_unchecked: bool,
    /// step-hold (--step-hold FILE, default off): the hold set (step_hold.rs). Every intermediate boundary of the
    /// window is stored as its held spans, so the sweep re-runs no step. None: as before.
    shold: Option<std::sync::Arc<crate::step_hold::HoldSet>>,
    /// step-hold: the window's steps (boundary 0 and the window end stay whole); the stores (count, bytes,
    /// seconds, buffers made on the step path); page-locked buffers being made ahead on a helper thread (capacity, maker).
    hold_steps: usize, hold_n: (u64, u64, f64, u64),
    hold_ahead: std::collections::VecDeque<(usize, std::thread::JoinHandle<Result<crate::pinned::PinMem, String>>)>,
}
/// lane rung3-params: one carried parameter: P4 family, carrier slot, value at the trajectory (the recorded literal).
#[derive(Clone)]
struct Par { name: String, fam: String, slot: usize, p0: f32 }
/// a12 truephys-term (2026-11-13): <(0, c), M^T y> = sum_k c_k p0_k (sum over columns of parameter k's cotangent), from
/// the per-column parameter cotangents (`par_bar`, 8 slots per family) a reverse seeded with y has accumulated.
/// THE BREAKAGE THIS PREVENTS: with --param-tl the first seed's tangent starts from x = 0 in the state and d ln p = c in
/// the parameters (see the seed loop), so its <y, M x> is the parameter tangent's; the dot scored the reverse against
/// that seed's random state x instead, which the tangent never used, and read about 1 (tp-moist-pg-grad 0.99986,
/// tp-moist-pgp-grad 1.00073, 2026-11-13) whatever the operator. The x side of that dot is this sum.
fn param_x_mty(params: &[Par], bars: &HashMap<String, Vec<f64>>, c: &[f64]) -> f64 {
    let mut s = 0f64;
    for (k, p) in params.iter().enumerate() {
        let Some(b) = bars.get(&p.fam) else { continue };
        let pw = b.len() / 8;
        let g: f64 = b[p.slot * pw..(p.slot + 1) * pw].iter().fold(0.0, |s, x| s + x);
        s += c.get(k).copied().unwrap_or(0.0) * p.p0 as f64 * g;
    }
    s
}
/// a12 truephys-term: the x side of a seed's dot. The pass whose tangent started from (0, c) (--param-tl, first seed) is
/// scored against the parameter half; every other pass against its own random state direction.
fn seed_x_mty(param_tl: Option<&[f64]>, k: usize, state_x_mty: f64, param_side: Option<f64>) -> Result<f64, String> {
    match (param_tl, k) {
        (Some(_), 0) => param_side.ok_or_else(|| "--param-tl: the first seed's reverse left no parameter cotangents to score".to_string()),
        _ => Ok(state_x_mty),
    }
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
/// x20 taylor-crash: --taylor-dir and --objective terms name (allocation, word) pairs of ONE recording; allocation ids are not
/// stable across recordings. THE BREAKAGE THIS PREVENTS: dir-theta-low / objective-Jth* were built for another recording
/// (theta_m pool 351); in pkg18m2 allocation 351 is the mesh's cellsOnCell, so every Taylor row wrote floats into neighbour
/// indices and the first perturbed forward faulted in smooth_helicity at step 2 (CUDA 700/717, at every h down to 1e-30),
/// and Jth read integers (J = 1.5e-42). Refused when most of an allocation's term words hold integer-shaped bits at
/// boundary 0 (nonzero with a zero exponent: an int32 below 2^23 read as float32), counted over the nonzero words; no float32
/// model field is mostly subnormal.
fn refuse_index_words(what: &str, file: &str, b: &Boundary, data: &[u8], terms: &[(u64, usize, f32)]) -> Result<(), String> {
    let at: HashMap<u64, (usize, usize)> = b.index.iter().rev().map(|x| (x.0, (x.1, x.2))).collect();
    let mut per: BTreeMap<u64, (u64, u64, u32)> = BTreeMap::new();
    for &(al, wd, _) in terms {
        let &(off, n) = at.get(&al).ok_or_else(|| format!("{what} {file}: allocation {al} is not in boundary 0"))?;
        if 4 * wd + 4 > n { return Err(format!("{what} {file}: word {wd} is outside allocation {al} ({n} bytes)")); }
        let bits = u32::from_le_bytes(data[off + 4 * wd..off + 4 * wd + 4].try_into().unwrap());
        let e = per.entry(al).or_insert((0, 0, bits));
        if bits & 0x7fff_ffff != 0 { e.0 += 1; }
        if bits & 0x7f80_0000 == 0 && bits & 0x007f_ffff != 0 { e.1 += 1; }
    }
    for (al, (nz, int_like, first)) in per {
        if 2 * int_like > nz {
            return Err(format!("{what} {file}: {int_like} of its {nz} nonzero words in allocation {al} hold integers at boundary 0 (first word {first} as int32), not a float field: the file was built for another recording's allocation ids. Perturbing integer words corrupts index arrays (pkg18m2: allocation 351 is cellsOnCell; the perturbed forward faulted in smooth_helicity, CUDA 700/717). Rebuild or translate the file for this plan."));
        }
    }
    Ok(())
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
        Some((k["function"].as_str().unwrap_or("?").to_owned(), deriv::spec_path(kernels, sp["path"].as_str()?).display().to_string(), sp["sha256"].as_str().map(str::to_owned)))
    }).collect();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, 32);
    let chunk = rows.len().div_ceil(threads).max(1);
    let got: Vec<Result<Option<String>, String>> = std::thread::scope(|sc| {
        let hs: Vec<_> = rows.chunks(chunk).map(|c| sc.spawn(move || c.iter().map(|(_, p, want)| {
            if want.is_none() { return Ok(None); }
            fs::read(p).map(|b| Some(crate::exec::digest(&b))).map_err(|e| format!("{p}: {e}"))
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
        Self::new_inner(ex, plan, a, None)
    }
    /// Explicit live-window=1 only. The in-memory plan contains metadata, never
    /// payloads or paths. LiveProgram owns its resolved commands and boundaries.
    pub(crate) unsafe fn from_live(ex: &Execution, program: &live_construct::LiveProgram,
        a: &HashMap<String, String>) -> Result<Self, String> {
        if opt(a, "live-window") != Some("1") { return Err("live-window=1 required".into()); }
        let mut w = Self::new_inner(ex, &program.plan, a, Some(&program.fixed))?;
        w.concat_types = program.concat_types.clone();
        Ok(w)
    }
    unsafe fn new_inner(ex: &Execution, plan: &Plan, a: &HashMap<String, String>,
        live_fixed: Option<&std::collections::HashSet<u64>>) -> Result<Self, String> {
        let cuda = &ex.cuda;
        // g2 r2: deriv::load judges each derived binding; under --local-topology 1 (asked, or forced by selected
        // damping, as below) a buildable topology recipe is admitted there (deriv::request_local_topology).
        let forced_local = plan.blocks.values().flatten().any(|op| matches!(op,
            Operation::Kernel { function, .. } if function == "divergence_damping_selected_f32"));
        deriv::request_local_topology(match opt(a, "local-topology") { Some("1") => true, None => forced_local, _ => false });
        let mut kernels = deriv::load(cuda, Path::new(req(a, "kernels")?), Path::new(req(a, "cubins")?), None)?;
        let set: Value = serde_json::from_slice(&fs::read(req(a, "kernels")?).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let mut classes: HashMap<String, String> = set["kernels"].as_array().ok_or("kernels")?.iter()
            .map(|k| (k["function"].as_str().unwrap_or("").to_owned(), k["class"].as_str().unwrap_or("").to_owned())).collect();
        let held = crate::held::apply(a, &mut kernels, &mut classes)?;
        // Check required rows before allocating the window shadow arena or replaying any step.
        for op in plan.blocks.values().flatten() {
            if let Operation::Kernel { function, .. } = op {
                if matches!(function.as_str(), "divergence_damping_f32" | "divergence_damping_selected_f32") {
                    let row = kernels.get(function);
                    let refused = opt(a, "refuse").unwrap_or("").split(',').map(str::trim)
                        .find(|s| s.split('=').next() == Some(function.as_str()));
                    wrules::require_damping_row(function, row.is_some(), row.and_then(|k| k.unsupported.as_deref()),
                        refused, row.is_some_and(|k| !k.tl_order.is_empty() && !k.ad_order.is_empty()))?;
                }
            }
        }
        // multirank-2: --local-topology 1 builds every topology derived binding from the launch's own (rank-local)
        // mesh. The mesh registry is built here, from every launch of the plan, before the forward sweep.
        let selected_damping = plan.blocks.values().flatten().any(|op| matches!(op,
            Operation::Kernel { function, .. } if function == "divergence_damping_selected_f32"));
        let local_topology = match (selected_damping, opt(a, "local-topology")) {
            (true, Some("0")) => return Err("DEFECT: selected damping requires rank-local topology; --local-topology 0 is invalid".into()),
            (true, None) => Some("1"),
            (_, value) => value,
        };
        match local_topology {
            None | Some("0") => {},
            Some("1") => {
                let launches = plan.blocks.values().flatten().filter_map(|op| match op {
                    Operation::Kernel { function, grid, block, shared_bytes, arguments, .. } => Some((function.as_str(),
                        arguments.iter().map(|x| match x {
                            crate::plan::Argument::Device { target } => submit::Argument::Device(submit::Slice { allocation: target.allocation, offset: target.offset }),
                            crate::plan::Argument::Bytes { bytes_hex, .. } => submit::Argument::Bytes(crate::decode_hex(bytes_hex).unwrap_or_default()),
                            // Live values (host scalars, frames) are not mesh topology; a placeholder keeps positions.
                            _ => submit::Argument::Bytes(Vec::new()),
                        }).collect::<Vec<_>>(), *grid, *block, *shared_bytes)),
                    _ => None,
                });
                let summary = deriv::install_local_topology(&kernels, launches);
                eprintln!("{}", json!({"event":"local_topology_registry","registry":summary}));
            },
            Some(v) => return Err(format!("--local-topology {v}: expected 0 or 1")),
        }
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
        // dycore: a plan recorded with hex's large-mesh kernel forms cannot be differentiated by a kernel set that
        // has rows for the plain forms only, and the window must say so before it loads anything else.
        // THE BREAKAGE THIS PREVENTS: a test machine <local path> (2026-11-09 19:15Z), the first 60 km
        // window: its plan (nEdges >= 30,000, so hex chose *_chunk_v841, *_hoist*, *_elem_v841 and the split
        // acoustic_column_* kernels) ran 24 dycore kernel kinds, 428 launches in 2 steps, as "REFUSED: no derivative
        // row and no rule", which the tangent and the reverse both treat as identity. The dot test still printed
        // (3.09e-7) because both sides skipped the same kernels. The plain forms carry the same bits (60 km, 12 state
        // fields, 270,994,048 words per boundary, 2 steps, 0 different: <local path>), so the fix is the
        // recording, which record_hrrrv5.sh now pins by itself. --refuse NAME=why still freezes a kernel on purpose.
        {
            let mut unrowed: BTreeMap<&str, usize> = BTreeMap::new();
            for ops in plan.blocks.values() { for op in ops { if let Operation::Kernel { function, .. } = op {
                let f = function.as_str();
                let form = f.contains("_chunk_v841") || f.contains("_hoist") || f.contains("_elem_v841")
                    || (f.starts_with("acoustic_column_") && f != "acoustic_column_solve_v841");
                if form && !kernels.contains_key(f) && !refuse.contains_key(f) { *unrowed.entry(f).or_default() += 1; }
            } } }
            if !unrowed.is_empty() && opt(a, "no-tangent") != Some("1") {
                return Err(format!("DEFECT window: this plan carries {} large-mesh kernel form(s) the kernel set has no derivative row for ({}). They would run as identity in the tangent and the adjoint. Record the plan again: ad-record pins the plain forms (GPUWM_HEX_EDGE_HOIST=0, GPUWM_HEX_MYNNEDMF_LANES=16), which give the same forward bits.",
                    unrowed.len(), unrowed.iter().map(|(f, n)| format!("{f} x{n}")).collect::<Vec<_>>().join(", ")));
            }
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
            rows_ad_direct: function(cuda, mp, "p8_rows_ad_direct_float")?, rows_ad_fan: function(cuda, mp, "p8_rows_ad_fan_float")?, clear_dst: function(cuda, mp, "p8_clear_dst_float")?,
            view_tl: function(cuda, mp, "p8_view_tl_float")?, view_ad: function(cuda, mp, "p8_view_ad_float")?,
            where_tl: function(cuda, mp, "p8_where_tl_float")?, where_ad: function(cuda, mp, "p8_where_ad_float")?,
            max_tl: function(cuda, mp, "p8_max_tl_float")?, max_ad: function(cuda, mp, "p8_max_ad_float")?,
            view_zero: function(cuda, mw, "win_view_zero_f32")?, fill: function(cuda, mw, "win_fill_f32")?, dot2: function(cuda, mw, "win_dot2_f32")?, nonfinite: function(cuda, mw, "win_nonfinite_f32")?,
            fct_minmax_tl: function(cuda, mw, "win_fct_minmax_frozen_tl")?, fct_minmax_ad: function(cuda, mw, "win_fct_minmax_frozen_ad")?,
            axpy: function(cuda, mw, "win_axpy_primal_f32")?, widen: function(cuda, mw, "win_widen_f32_f64")?,
            narrow: function(cuda, mw, "win_narrow_f64_f32")?,
            widen2: if stripe_xfer_on() { function(cuda, mw, "win_widen2_f32_f64").unwrap_or(ptr::null_mut()) } else { ptr::null_mut() },
            narrow2: if stripe_xfer_on() { function(cuda, mw, "win_narrow2_f64_f32").unwrap_or(ptr::null_mut()) } else { ptr::null_mut() } };
        let native = if opt(a, "native-schemes") == Some("1") {
            let names: std::collections::HashSet<String> = kernels.keys().cloned().collect();
            Some(crate::schemes::Native::from_env(cuda, &names)?.ok_or("--native-schemes 1 needs AD_SCHEMES (the L2 bundle)")?)
        } else { None };
        let fct_frozen = opt(a, "fct-frozen") == Some("1");
        let mesh_constants = opt(a, "mesh-constants") == Some("1");
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
        let mut tempo_paths = None;
        let tempo = if (opt(a, "tempo-live") == Some("1") || tempo_all) && !hold_tempo {
            let p = opt(a, "tempo-cubin").map(str::to_owned).or_else(|| std::env::var("AD_WIN_TEMPO_CUBIN").ok().filter(|v| !v.is_empty()))
                .or_else(|| std::env::var("AD_DOT_TEMPO_CUBIN").ok().filter(|v| !v.is_empty()))
                .ok_or("--tempo-live 1 needs --tempo-cubin (or AD_WIN_TEMPO_CUBIN): the sm_120 tempo_adw image")?;
            let mut t = crate::tempo_adw::TempoAdw::load(cuda, &p)?;
            let mut tlf_path = None;
            // tempo-once: the fused tangent module beside the image (<image>.tlf.cubin, mk_tlf.py), or --tempo-tlf-cubin /
            // AD_WIN_TEMPO_TLF_CUBIN; --tempo-tlf 0 keeps the segmented tangent.
            if opt(a, "tempo-tlf") != Some("0") {
                let sib = p.strip_suffix(".cubin").map(|b| format!("{b}.tlf.cubin")).filter(|q| std::path::Path::new(q).exists());
                let q = opt(a, "tempo-tlf-cubin").map(str::to_owned).or_else(|| std::env::var("AD_WIN_TEMPO_TLF_CUBIN").ok().filter(|v| !v.is_empty())).or(sib);
                if let Some(q) = q { t.load_tlf(cuda, &q)?; tlf_path = Some(q); }
            }
            let ar = dev::Arena::new(cuda, (gib("tempo-arena-gib", 6.0) * (1u64 << 30) as f64) as usize)?;
            tempo_paths = Some((p, tlf_path));
            Some((t, ar))
        } else { None };
        let tempo_shards = if let Some((p, q)) = tempo_paths {
            crate::tempo_shard::Pool::from_env(cuda, ex.context, &p, q.as_deref())?
        } else {
            if std::env::var_os("AD_TEMPO_SHARD_DEVICES").is_some() {
                return Err("DEFECT AD_TEMPO_SHARD_DEVICES requires live TEMPO".into());
            }
            None
        };
        let tmp_words: i64 = 1 << 25;
        let mut tmp = [0u64; 3];
        for t in tmp.iter_mut() { checked("cuMemAlloc dot temp", (cuda.allocate)(t, tmp_words as usize * 4))?; }
        let (mut part, mut fail) = (0u64, 0u64);
        checked("cuMemAlloc partials", (cuda.allocate)(&mut part, 1024 * 3 * 8))?;
        checked("cuMemAlloc failure flag", (cuda.allocate)(&mut fail, 4))?;
        dev::zero(cuda, fail, 4)?;
        let mut fixed = std::collections::HashSet::new();
        if let Some(live) = live_fixed {
            fixed = live.clone();
        } else {
        let catalog = Path::new(req(a, "config")?).parent().unwrap_or(Path::new(".")).join("catalog-a.json");
        if let Ok(bytes) = fs::read(&catalog) {
            let c: Value = serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", catalog.display()))?;
            for x in c["allocations"].as_array().into_iter().flatten() {
                let (Some(id), Some(init), Some(b)) = (x["id"].as_u64(), x["initial_sha256"].as_str(), x["boundary_sha256"].as_object()) else { continue };
                if !b.is_empty() && b.values().all(|v| v.as_str() == Some(init)) { fixed.insert(id); }
            }
        }
        } // recorded catalog only
        let loaded = kernels.values().filter(|k| k.unsupported.is_none()).count();
        eprintln!("{}", json!({"event":"window_loaded","kernels":kernels.len(),"dispatchable":loaded,"shadow_words":words,
            "shadow_bytes":words * 4,"allocations":sizes.len(),"tape_bytes":tape.bytes,"scratch_bytes":scratch.bytes,
            "refuse":refuse,"held":held,"fixed_allocations":fixed.len(),"fct_frozen":fct_frozen,
            "native_schemes":native.as_ref().map(|n| n.loaded.clone()),
            "tempo_live":tempo.as_ref().map(|t| json!({"tempo_adw":t.0.image_sha256,"tempo_adw_tlf":t.0.tlf.as_ref().map(|x| x.image_sha256.clone()),"arena_bytes":t.1.bytes})),
            "tempo_precip":tempo_serial.as_ref().map(|t| t.image_sha256.clone()),
            "tempo_small":tempo_small.as_ref().map(|t| json!({"functions":tempo_small_fns.iter().collect::<Vec<_>>(),"image_sha256":t.image_sha256}))}));
        let graph_segment = match std::env::var("AD_WHOLESTEP_GRAPH_SEGMENT").as_deref() {
            Ok("w_edge_flux_f32") => true,
            Err(std::env::VarError::NotPresent) | Ok("off") => false,
            _ => return Err("DEFECT AD_WHOLESTEP_GRAPH_SEGMENT accepts w_edge_flux_f32 or off".into()),
        };
        Ok(Self { join_one: None, synth: Vec::new(), lw_follow: Vec::new(), peer_buf: (0, 0), peer_tolerated: BTreeMap::new(), vectors: None, graph_segment, segment_graphs: Default::default(), graph_tape_epoch: 0, graph_scratch_epoch: 0, concat_types: HashMap::new(), kernels, classes, schemes, refuse, refuse_how: BTreeMap::new(), base_el, sh_base, words, shadows, support: dev::Support::load(cuda)?,
                  ledger: Some(dev::Ledger::new(cuda, 8 << 20)?), tempo_io: (0, 0), tempo_rows: Vec::new(), sync_owed: false, f,
                  tape, scratch, tmp, tmp_words, part, fail, gathers: HashMap::new(), counting: false, rev_count: None, labels: BTreeMap::new(), held_kernels: BTreeMap::new(),
                  held_dropped: BTreeMap::new(), entry_ms: 0.0, writer: None,
                  trace: opt(a, "trace-norm") == Some("1"), trace_last: 0.0, trace_hits: 0, fixed, held_prev: None, held_skips: (0, 0), redo_stream: std::ptr::null_mut(), redo_ev: [std::ptr::null_mut(); 2], xfer_stream: std::ptr::null_mut(), xfer_ev: [std::ptr::null_mut(); 2], xfer_pending: false, xfer_sync: None, restore_async_n: 0,
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
                  fct_frozen, mesh_constants, taylor_mask: false, row_live: false, hold: Hold::from_env()?, address_bearing: ex.address_bearing.clone(), native, tempo, tempo_shards, tempo_stats: BTreeMap::new(), tempo_serial,
                  tempo_small, tempo_small_fns, tempo_small_identity_words: 0, blob_index: BlockBlobIndex::new(plan), blob_next: Vec::new(), blob_ahead: std::collections::HashSet::new(), rv_restore_s: 0.0, rv_tape_s: 0.0, rv_redo_dl_s: 0.0, rv_redo_wait_s: 0.0, rv_walk_s: 0.0, load_next: None,
                  ckpt_base: None, delta_store: false, bs_now: 0, bs_peak: 0, bs_written: 0, bs_files: 0, bs_files_peak: 0, bs_ram_budget: 0, bs_ram_now: 0, bs_ram_peak: 0, bs_ram_count: 0, restore_direct: 0, wcache: std::sync::Mutex::new(HashMap::new()), wpins: std::sync::Mutex::new(Vec::new()), dpins: std::sync::Mutex::new(Vec::new()), pin_next: None, pin_ahead: false, pin_ahead_hits: 0, pin_q: std::collections::VecDeque::new(), pin_q_made: 0, pin_min: 0, last_label: String::new(), gather_pass_join: false,
                  recompute_steps: 0, recompute_s: 0.0, born: HashMap::new(), ck_pf: None, load_s: 0.0, store_s: 0.0, pf_hits: 0, cs_on: false, cs_refs: Vec::new(), cs_dev_at: None, cs_serial: 0, cs_pin: None, cs_truth: None, cs_end_truth: None, cs_redo_res: None, cs_n: [0; 6], cs_s: [0.0; 6], cs_exact_peak: 0, cs_spares: 0, cs_cut: None, cs_cut_steps: 0, cs_plant_ask: None, cs_plant: None, cs_cut_n: [0; 3], cs_cut_s: 0.0, anchor_pool: Vec::new(), anchor_ahead: None, anchor_ahead_wait_s: 0.0, anchor_ahead_taken: 0, anchor_ahead_info: Vec::new(), dev_anchors: Vec::new(), restore_s: 0.0, params, recompute_unchecked: false, shold: None, hold_steps: 0, hold_n: (0, 0, 0.0, 0), hold_ahead: std::collections::VecDeque::new(),
                  one_pass: opt(a, "one-pass") != Some("0"), one_pass_fallbacks: 0, tape_grown: None, tape_max: (gib("tape-max-gib", 16.0) * (1u64 << 30) as f64) as usize, tl_tape_grown: Vec::new(), pack: std::env::var("AD_TAPE_PACK").as_deref() != Ok("0"), precache: deriv::PreCache::default(), rows: BTreeMap::new(), rev_rows: BTreeMap::new(), seg_rows: BTreeMap::new(), census: Census::from_env(), win: None, win_cap: std::env::var("AD_TAPE_WINCACHE_MIB").ok().and_then(|v| v.parse::<usize>().ok()).map(|m| m << 20).unwrap_or(usize::MAX), win_promoted: (0, 0), rev_scratch: 0, scratch_base: 0, scratch_cli: 0, scratch_fit: None, pack_ratio: 0.0, seg_ratio: 0.0, fn_ms: BTreeMap::new(), arena_use: BTreeMap::new(), arena_noted: (0, 0), capture: false, capture_host: None, pending: Vec::new(), defer: false, defer_step: 0, tempo_sized: None,
                  fuse: std::env::var("AD_FUSE_TAPE").as_deref() == Ok("1") || opt(a, "fuse-tape") == Some("1"), fuse_rec: false, fuse_ok: true, rtape: None, rule_cache: None,
                  fz_cur: None, fz_tapes: BTreeMap::new(), fz_chunks: Vec::new(), fz_resident: None, fz_last_bytes: 0, fz_step_start: 0, fz_info: Vec::new(),
                  fz_d2h_s: 0.0, fz_d2h_bytes: 0, fz_failed: false, fz_force_flush: false, fz_steps: 0,
                  fz_copy: std::ptr::null_mut(), fz_evs: Vec::new(), fz_done: HashMap::new(), fz_h2d_s: 0.0, fz_fence_waits: 0, wk_s: [0.0; 7], wk_n: [0; 4], fz_offs: HashMap::new(), fz_cur_offs: Vec::new(), fz_pf: None, fz_pf_info: Vec::new(),
                  fz_pre: None, fz_wbuf: None, fz_wev: std::ptr::null_mut(), fz_words: None, fz_early: 0,
                  snap_spare: None, // drains (AD_SNAP_REUSE), on a line of its own so the long line above merges cleanly
                  kf_keep: std::collections::BTreeSet::new(), kf_saved: BTreeMap::new(), // keep-forward (AD_KEEP_FORWARD)
                  st_on: std::env::var("AD_TAPE_STREAM").as_deref() == Ok("1") || opt(a, "tape-stream") == Some("1"), seg_offs: Vec::new(), st_pool: Vec::new(), st_ahead: None, st_info: Vec::new(), st_made: 0, st_planned: false, st_ahead_bytes: 0, st_want: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(usize::MAX)), st_first: ST_AHEAD, st_est: Vec::new(), st_borrowed: Vec::new() })
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
    /// Prepare the first step's authenticated blobs outside the step loop and start the next two steps' decode.
    /// The same bounded cache serves every sweep. C API sessions call this at open and at tl_init, never in tl_step.
    unsafe fn prepare_pass_start(&mut self, ex: &mut Execution, plan: &Plan, base: &Path, frame: &Option<ReplayFrame>, steps: usize) -> Result<(), String> {
        let t = Instant::now();
        let k = steps.min(3);
        if k == 0 { return Ok(()); }
        let ahead: Vec<Vec<String>> = match (frame.as_ref(), plan.physics.as_ref()) {
            (Some(f), Some(ph)) => f.blocks_ahead(ph, 1, k),
            _ => (0..k).map_while(|i| plan.cadence.get(i).cloned()).collect(),
        };
        let Some(first_blocks) = ahead.first() else { return Ok(()); };
        let first_blobs = self.blob_index.get(first_blocks);
        let first: std::collections::HashSet<String> = first_blobs.iter().map(|b| crate::blob_sha(b).to_owned()).collect();
        let cached = ex.blob_cache.as_ref();
        let mut want: Vec<crate::plan::Blob> = Vec::new();
        // A reinitialized/aborted sweep can leave another step's keep-set. In-flight work is deduplicated separately.
        self.blob_ahead.clear();
        for names in &ahead[1..] { for b in self.blob_index.get(names) {
            let sha = crate::blob_sha(&b).to_owned();
            if first.contains(&sha) || !self.blob_ahead.insert(sha.clone()) || cached.is_some_and(|c| c.contains_key(&sha)) { continue; }
            want.push(b);
        } }
        self.prefetch_blobs(base, want)?;
        // Decode and authenticate step 1 before entering the step loop. Later passes reuse retained bytes when present.
        self.stage_blobs(ex, base, first_blobs, None)?;
        eprintln!("{}", json!({"event":"window_pass_prepare","wall_s":t.elapsed().as_secs_f64(),"prefetched_steps":k}));
        Ok(())
    }
    /// G5: decode `want` in the background (merged by the next stage_blobs). An earlier unfinished prefetch is kept:
    /// its results are merged into the new one when it ends.
    /// machine: start decoding step 1's blobs as soon as the window exists (prepare_pass_start joins them). They were
    /// decoded inside prepare_pass_start with everything waiting: 0.5-1.0 s at 120 km (mark pass_prepare). Same blobs,
    /// same checks.
    unsafe fn prefetch_first(&mut self, plan: &Plan, base: &Path, frame: &Option<ReplayFrame>) -> Result<(), String> {
        let ahead: Vec<Vec<String>> = match (frame.as_ref(), plan.physics.as_ref()) {
            (Some(f), Some(ph)) => f.blocks_ahead(ph, 1, 1),
            _ => plan.cadence.first().cloned().into_iter().collect(),
        };
        let Some(first_blocks) = ahead.first() else { return Ok(()); };
        let want = self.blob_index.get(first_blocks);
        self.prefetch_blobs(base, want)
    }
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
        if bounds[i].ram.is_some() || bounds[i].dev != 0 || bounds[i].cs.is_some() { return Ok(()); } // host-leftovers-2: RAM boundary, no file to read
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
              fct_frozen: self.fct_frozen, mesh_constants: self.mesh_constants }
    }
    fn sh(&self, el: i64) -> u64 { self.sh_base + 4 * el as u64 }
    /// JOIN (AD_DUMP_BARS_AFTER=step:allocation:byte_offset:bytes,.., off unless set): when the reverse of `step` has
    /// ended (the bars then stand at boundary step-1), the listed bar bytes are read and their sha256 is logged; with
    /// AD_DUMP_BARS_DIR they are also written raw (bars-s<step>-<allocation>-<offset>.f32). Reads only.
    /// THE BREAKAGE THIS PREVENTS: the recorded route gave its bars only at the window start, so a one-step backward
    /// of the joined route (which cannot run loop step 1) had nothing on the recorded route to be read against.
    unsafe fn dump_bars_after(&self, cuda: &Cuda, step: usize, sweep: &str) -> Result<(), String> {
        let Ok(list) = std::env::var("AD_DUMP_BARS_AFTER") else { return Ok(()) };
        let mut synced = false;
        for x in list.split(',').map(str::trim).filter(|x| !x.is_empty()) {
            let q: Vec<u64> = x.split(':').map(|v| v.parse::<u64>().map_err(|_| format!("AD_DUMP_BARS_AFTER {x}: numbers")))
                .collect::<Result<_, String>>()?;
            if q.len() != 4 { return Err(format!("AD_DUMP_BARS_AFTER {x}: want step:allocation:byte_offset:bytes")); }
            if q[0] as usize != step { continue; }
            let (Some(&el), Some(sh)) = (self.base_el.get(&q[1]), self.shadows.get(&q[1])) else {
                return Err(format!("AD_DUMP_BARS_AFTER {x}: allocation {} has no shadow", q[1])) };
            if q[2] % 4 != 0 || q[3] == 0 || (q[2] + q[3]) as usize > sh.bytes as usize {
                return Err(format!("AD_DUMP_BARS_AFTER {x}: outside allocation {} or not on a word", q[1]));
            }
            if !synced { dev::sync(cuda)?; synced = true; }
            let bytes = dev::download(cuda, self.sh(el + (q[2] / 4) as i64), q[3] as usize)?;
            let sha: String = { use sha2::{Digest, Sha256}; Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect() };
            let nonzero = bytes.chunks_exact(4).filter(|c| c.iter().any(|b| *b != 0)).count();
            let mut file = Value::Null;
            if let Ok(dir) = std::env::var("AD_DUMP_BARS_DIR") {
                let p = Path::new(&dir).join(format!("bars-s{step}-{}-{}.f32", q[1], q[2]));
                fs::write(&p, &bytes).map_err(|e| e.to_string())?;
                file = json!(p.display().to_string());
            }
            eprintln!("{}", json!({"event":"bars_after_reverse_step","sweep":sweep,"step":step,"allocation":q[1],"byte_offset":q[2],
                "bytes":q[3],"sha256":sha,"nonzero_words":nonzero,"file":file}));
        }
        Ok(())
    }
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
    /// x20 scratch-arena: the base is the workspace the tangent pass MEASURED (the arena's peak when the reverse first
    /// fits it), not the size the CLI asked for, and the arena is cut down to base + one launch's transient when that is
    /// smaller. THE BREAKAGE THIS FIXES: pk120 (x1.40962, --scratch-gib 9) held a 12.03 GB reverse scratch whose peak was
    /// 5.70 GB (x20M-def3); with the 12 GiB native reserve that left the packed tape 18.08 GB for 20.2-20.3 GB steps, so
    /// steps 2 and 3 reversed in 4-5 segments (2.3-3.3x replay, restore + pass 1 + snap + replay 6.7-12.6 s per step).
    /// The freed card memory goes to the tape (one pass on every step) and to the native rows' reverse scratch (the MYNN
    /// tiles size themselves from free memory). AD_REV_SCRATCH_FIT=0 keeps the old base; AD_REV_SCRATCH_SLACK_MIB
    /// (default 256) is added to the measured peak. A reverse launch that needs more than the arena refuses with
    /// "derivative arena exhausted" (raise the slack or set AD_REV_SCRATCH_FIT=0); the words never depend on the size.
    unsafe fn fit_scratch(&mut self, cuda: &Cuda) -> Result<(), String> {
        let first = self.scratch_base == 0;
        if first {
            let fit = self.pack && std::env::var("AD_REV_SCRATCH_FIT").as_deref() != Ok("0");
            let slack = std::env::var("AD_REV_SCRATCH_SLACK_MIB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(256) << 20;
            let tl_peak = self.scratch.peak;
            // memory (AD_TL_BIND_SCRATCH=1): the tangent pass may have re-made the scratch larger; the base is still
            // bounded by the size the command line asked for.
            let cli = if self.scratch_cli > 0 { self.scratch_cli } else { self.scratch.bytes };
            self.scratch_base = if fit && tl_peak > 0 { (tl_peak + slack).min(cli) } else { cli };
            self.scratch_fit = Some(json!({"on":fit,"tangent_peak_bytes":tl_peak,"slack_bytes":slack,"cli_bytes":cli,
                                           "tangent_bind_from_scratch":tl_bind_scratch(),"bytes_after_tangent":self.scratch.bytes,
                                           "base_bytes":self.scratch_base,"transient_bytes":self.rev_scratch,
                                           "sizing":if rev_fit_max() { "max" } else { "sum" }}));
        }
        // memory (AD_REV_SCRATCH_FIT=max). THE BREAKAGE THIS PREVENTS: the arena was sized base + the largest
        // one-launch transient, as if the launch with the largest transient also needed the largest workspace. At
        // x1.163842 (60 km, kset-d7) that is 4 GiB + the 9.19 GB MYNN-EDMF bind = 13.55 GB on card 0, whose measured
        // peak is 9.19 GB (a test machine <local path> scratch_peak): the MYNN bind takes no
        // workspace from this arena and no dycore launch has a transient that size. 4.1 GB of the fullest card for
        // nothing. With `max` the arena is the larger of the two (+ the same slack); a launch that does need both at
        // once takes the overflow from a kept spill slab, as a fitted arena already does, so no launch is refused and
        // the words do not depend on the size.
        let fit_max = rev_fit_max();
        let want = if fit_max { self.scratch_base.max(self.rev_scratch + (256 << 20)) + (64 << 20) }
                   else { self.scratch_base + self.rev_scratch + (64 << 20) };
        let want = want.div_ceil(2 << 20) * (2 << 20);
        // Cut down only at the first fit, and only when that frees at least 256 MiB (every later call grows or keeps it).
        let shrink = first && self.pack && want + (256 << 20) <= self.scratch.bytes;
        let spill_on = self.scratch_fit.as_ref().is_some_and(|f| f["on"] == true);
        if !self.pack || (want <= self.scratch.bytes && !shrink) {
            if first { if let Some(f) = self.scratch_fit.as_mut() { f["reverse_bytes"] = json!(self.scratch.bytes); } }
            if spill_on && self.scratch.spill.is_none() { self.scratch.spill = Some(dev::Spill { alloc: cuda.allocate, slabs: Vec::new(), cur: 0, off: 0, held: 0, peak: 0 }); }
            return Ok(());
        }
        dev::sync(cuda)?;
        let before = self.scratch.bytes;
        if let Some(f) = self.scratch_fit.as_mut() { f["reverse_bytes"] = json!(want); }
        checked("cuMemFree scratch", (cuda.free)(self.scratch.base))?;
        let spill = self.scratch.spill.take();
        self.scratch = match dev::Arena::new(cuda, want) { Ok(s) => s, Err(e) => { self.scratch = dev::Arena::new(cuda, before)?;
            return Err(format!("packed-tape: the reverse scratch needs {want} bytes (one launch's transient + workspace), card refused: {e}")) } };
        // x20 scratch-arena: a fitted reverse scratch may spill into kept slabs (dev::Spill) instead of refusing.
        self.scratch.spill = spill.or_else(|| spill_on.then(|| dev::Spill { alloc: cuda.allocate, slabs: Vec::new(), cur: 0, off: 0, held: 0, peak: 0 }));
        eprintln!("{}", json!({"event":"window_scratch_grown","before":before,"bytes":want}));
        Ok(())
    }
    fn note(&mut self, pass: &str, f: &str, t0: Instant) {
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        self.entry_ms += ms;
        let e = self.fn_ms.entry((pass.to_owned(), f.to_owned())).or_insert((0.0, 0));
        e.0 += ms; e.1 += 1;
        let spill = self.scratch.spill.as_ref().map(|s| s.slabs.iter().take(s.cur).map(|x| x.1).sum::<usize>() + if s.cur < s.slabs.len() { s.off } else { 0 }).unwrap_or(0);
        // a launch is counted only for an arena it reset itself (a TEMPO entry works in its own arena)
        let (sr, tr) = (self.scratch.resets, self.tape.resets);
        if sr != self.arena_noted.0 || (pass == "tl" && tr != self.arena_noted.1) {
            let u = self.arena_use.entry((pass.to_owned(), f.to_owned())).or_insert((0, 0));
            if sr != self.arena_noted.0 { u.0 = u.0.max(self.scratch.used + spill); }
            // the tangent tapes one launch at a time (the arena is reset before each); the reverse tape holds a whole chunk
            if pass == "tl" && tr != self.arena_noted.1 { u.1 = u.1.max(self.tape.used); }
        }
        self.arena_noted = (sr, tr);
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
    /// dycore (round 2): true when the command is a kernel the run froze on purpose with --refuse.
    fn listed_refusal(&self, op: &Operation) -> bool {
        matches!(op, Operation::Kernel { function, .. } if self.refuse.contains_key(function.as_str()))
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
                // D1b (g2, 2026-11-11): a refused P7 gas-optics kind has no kernel-set row and no rule, so it kept
                // identity; its native row says what it writes, and it is held on those words (wrules::hold_refused_native).
                let r = if matches!(r, R::Refused(_)) && self.refuse.contains_key(function.as_str()) {
                    let written = self.native.as_ref().and_then(|n| n.refused_written(function, &args)).map(|w| w.and_then(|v| {
                        let c = self.ctx();
                        v.into_iter().map(|(al, off, n)| Ok((c.el(al, off)?, n))).collect::<Result<Vec<_>, String>>()
                    }));
                    crate::wrules::hold_refused_native(function, r, written)
                } else { r };
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
        // dycore: a launch with no derivative at all is a defect of the plan or the kernel set, never an identity.
        // THE BREAKAGE THIS PREVENTS: the same 60 km window (see Win::new) also ran rrfs_mynnedmf_run_l1 (12 launches)
        // with no row; and a row whose image or entry cannot be bound ("entry not dispatchable") was skipped the same
        // silent way. A kernel named with --refuse keeps its own label and is not touched by this.
        if let R::Refused(s) = &r {
            if (s.ends_with(": no derivative row and no rule") || s.contains(": entry not dispatchable: ")) {
                return Err(format!("DEFECT window: step {step}: {s}. It would run as identity in the tangent and the adjoint. Record the plan with the plain kernel forms (ad-record does), add the row to the kernel set, or freeze the kernel on purpose with --refuse NAME=why."));
            }
            // dycore (round 2): every OTHER refusal that --refuse did not ask for is the same defect.
            // THE BREAKAGE THIS PREVENTS: every 120 km window of 2026-11-09 ran cupy_concatenate_same_size as identity
            // ("element type Some(double) not established") beside the 64 kinds its deck froze on purpose; the dot
            // passed (tangent and adjoint skip the same command) and the receipt read like a complete derivative. The
            // guard above stopped two refusal classes only (verifier dycore:r1, COORD 22:04Z). A refusal is now an error
            // unless its kernel is named with --refuse. AD_UNLISTED_REFUSED=identity keeps the identity for a diagnostic
            // run; the receipt lists such kinds under refused_beyond_refuse_list and says valid_derivative false.
            if !self.listed_refusal(op) && std::env::var("AD_UNLISTED_REFUSED").as_deref() != Ok("identity") {
                return Err(format!("DEFECT window: step {step}: {s}. Nothing asked for this refusal (--refuse does not name it), and it would run as identity in the tangent and the adjoint while the dot test still passes. Give the command a rule or a row, or freeze it on purpose with --refuse NAME=why (AD_UNLISTED_REFUSED=identity: diagnostic run, labelled invalid)."));
            }
        }
        if let R::Rows { rows, label } = &r {
            let tg = dev::prof::start();
            let g = if self.defer && std::env::var("AD_GATHER_SYNC").as_deref() != Ok("1") { self.gather_defer(rows, label); Ok(false) }
                    else { self.gather_for(&ex.cuda, rows) };
            dev::prof::add(13, tg);
            if let Err(e) = g {
                // dycore (round 2): a copy plan that cannot be built was an unlisted identity too (see above).
                if std::env::var("AD_UNLISTED_REFUSED").as_deref() != Ok("identity") {
                    return Err(format!("DEFECT window: step {step}: {label}: {e}. The copy rows of this command cannot be built, and it would run as identity in the tangent and the adjoint (AD_UNLISTED_REFUSED=identity: diagnostic run, labelled invalid)."));
                }
                return Ok((R::Refused(format!("{label}: {e}")), l));
            }
        }
        // D1 (g2, 2026-11-11): a kind named with --refuse runs HELD (wrules::hold_refused): zero tangent on the words it
        // writes, their cotangent consumed. How each such kind ran goes in the receipt (refuse_list_rule), and a kind whose
        // writes are unknown, so it kept identity, is named there (refuse_list_identity).
        let listed = match (&l, self.listed_refusal(op)) { (Some(l), true) => Some(l.0.clone()), _ => None };
        if let Some(f) = &listed {
            if !self.refuse_how.contains_key(f) {
                // D1c: the text is wrules::refuse_how; a class-held identity is named IDENTITY like an unknown write.
                self.refuse_how.insert(f.clone(), crate::wrules::refuse_how(&r));
            }
        }
        if self.counting {
            let name = match (&r, &l) {
                // a listed refusal keeps the label it always had, whatever shape its hold takes
                (_, Some(l)) if listed.is_some() => format!("REFUSED:{}: {}", l.0, self.refuse.get(&l.0).map(String::as_str).unwrap_or("")),
                (R::Entry, Some(l)) => format!("entry:{}", l.0),
                (R::Rows { label, .. }, _) => format!("rule:rows:{label}"),
                (R::View { .. }, Some(l)) => format!("rule:view:{}", l.0),
                (R::ZeroView { label, .. }, _) => format!("rule:zero-view:{label}"),
                (R::Max { .. }, _) => "rule:max".into(), (R::Where { .. }, _) => "rule:where".into(),
                (R::Scale { .. }, _) => "rule:scale".into(), (R::Copy { .. }, _) => "rule:copy".into(),
                (R::Axpy { .. }, _) => "rule:axpy".into(),
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
            if dev::sites::on() { self.last_label = name.chars().take(70).collect(); }
            if name.starts_with("held:") {
                let k = if let Operation::Kernel { function, .. } = op { function.to_string() } else { "?".to_string() };
                self.held_kernels.entry(name.clone()).or_default().insert(k);
            }
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
        let l2 = label.to_owned();
        self.pending.push((h, label.to_owned(), std::thread::spawn(move || { let t = Instant::now(); let g = wrules::gather_quick(&r);
            // machine: one line per deferred gather build (label, size, kind, seconds on its worker thread)
            eprintln!("{}", json!({"event":"window_gather","label":l2,"step":st,"rows":r.len(),"words":r.iter().map(|x| x.count).sum::<i64>(),
                "injective":g.as_ref().map(|x| x.injective).ok(),"error":g.as_ref().err(),"table_bytes":g.as_ref().map(|x| 8 * (x.src_word.len() + x.start.len() + x.item.len() + x.dst_word.len()) + x.sourceless.len()).unwrap_or(0),
                "fan_rows":g.as_ref().map(|x| x.fan.len()).unwrap_or(0),"fan_groups":g.as_ref().map(|x| x.fan_groups.len() / 2).unwrap_or(0),"direct_rows":g.as_ref().map(|x| x.direct.len()).unwrap_or(0),"direct_words":g.as_ref().map(|x| x.direct.iter().map(|r| r.count).sum::<i64>()).unwrap_or(0),
                "build_s":t.elapsed().as_secs_f64()}));
            g }), st));
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
            let g0 = wrules::gather_quick(rows);
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
                Ok(g) if g.injective => Ok(GatherDev { injective: true, bufs: [0; 5], nsrc: 0, ndst: 0, direct: Vec::new(), fan: Vec::new(), fan_groups: Vec::new() }),
                // machine round 2: a fan-in plan has no tables: nothing is allocated or uploaded
                Ok(g) if !g.fan.is_empty() && g.item.is_empty() && g.dst_word.is_empty() =>
                    Ok(GatherDev { injective: false, bufs: [0; 5], nsrc: 0, ndst: 0, direct: g.direct, fan: g.fan, fan_groups: g.fan_groups }),
                Ok(g) => {
                    let mut bufs = [0u64; 5];
                    let parts: [&[u8]; 5] = [
                        std::slice::from_raw_parts(g.src_word.as_ptr().cast(), g.src_word.len() * 8),
                        std::slice::from_raw_parts(g.start.as_ptr().cast(), g.start.len() * 8),
                        std::slice::from_raw_parts(g.item.as_ptr().cast(), g.item.len() * 8),
                        std::slice::from_raw_parts(g.dst_word.as_ptr().cast(), g.dst_word.len() * 8), &g.sourceless];
                    let _cold = crate::spill::cold(&format!("gather:{h:x}"));
                    for (b, p) in bufs.iter_mut().zip(parts) {
                        checked("cuMemAlloc gather", (cuda.allocate)(b, p.len().max(8)))?;
                        dev::upload(cuda, *b, p)?;
                    }
                    Ok(GatherDev { injective: false, bufs, nsrc: g.src_word.len() as i64, ndst: g.dst_word.len() as i64, direct: g.direct, fan: Vec::new(), fan_groups: Vec::new() })
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
        if std::env::var("AD_TEMPO_HASH").as_deref() == Ok("1") {
            // S01c diagnostic: the linearisation point of this launch (snapshot inputs) and the shadow spans it binds.
            let (t, _) = self.tempo.as_ref().ok_or("TEMPO derivative not loaded")?;
            let h = t.snap_hashes(cuda, s)?;
            eprintln!("{}", json!({"event":"window_tempo_hash","pass":if adjoint { "ad" } else { "tl" },"step":step,"snap":h,
                                   "ins":ins.iter().map(|(q, e, l)| json!([q, e, l])).collect::<Vec<_>>(),
                                   "outs":outs.iter().map(|(q, e, l)| json!([q, e, l])).collect::<Vec<_>>()}));
        }
        let t0 = Instant::now();
        let (g, side) = if !adjoint { (&ins, &outs) } else { (&outs, &ins) };
        let nv = nin * nc;
        if self.tempo_io.1 < 16 * nv {
            if self.tempo_io.0 != 0 { dev::sync(cuda)?; checked("cuMemFree tempo io", (cuda.free)(self.tempo_io.0))?; }
            let mut a = 0u64;
            { let _cold = crate::spill::cold("tempo_io"); checked("cuMemAlloc tempo io", (cuda.allocate)(&mut a, 16 * nv))?; }
            self.tempo_io = (a, 16 * nv);
        }
        crate::spill::prefetch("tempo_io");
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
            let lend = std::env::var("AD_PHYS_SHARED_ARENA").as_deref() != Ok("0");
            if lend { *crate::tempo_adw::LEND.lock().map_err(|_| "TEMPO lend lock poisoned")? = (ar.base, ar.bytes); }
            self.tempo_sized = Some(json!({"arena_bytes_before":before,"arena_bytes":ar.bytes,"grown":grown,"card_free_bytes":free,"lent_to_mynn_ring":lend}));
            eprintln!("{}", json!({"event":"window_tempo_arena","info":self.tempo_sized}));
        }
        let (t, ar) = self.tempo.as_mut().ok_or("TEMPO derivative not loaded")?;
        let led = self.ledger.as_mut().ok_or("step ledger missing")?;
        let (itok, icmp, tiles) = if let Some(pool) = self.tempo_shards.as_mut() {
            pool.apply(cuda, t, ar, s, vd, rd, adjoint, led)?
        } else { t.apply(cuda, ar, s, vd, rd, adjoint, !adjoint, led)? };
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
        led.tolerate_words = crate::peer::tolerate_guards();
        led.flush(cuda)?;
        let tol = std::mem::take(&mut led.tolerated);
        if !tol.is_empty() {
            let mut rows: BTreeMap<String, (u64, u64, Vec<String>)> = BTreeMap::new();
            for (label, n, src) in tol {
                // label = "step N <function> tangent|adjoint": keyed without the step
                let key = label.splitn(3, ' ').nth(2).unwrap_or(&label).to_owned();
                for m in [&mut rows, &mut self.peer_tolerated] {
                    let e = m.entry(key.clone()).or_default(); e.0 += 1; e.1 += n as u64;
                    for s in &src { if e.2.len() < 6 && !e.2.contains(s) { e.2.push(s.clone()); } }
                }
            }
            eprintln!("{}", json!({"event":"window_peer_guards_tolerated","pass":pass,"step":step,"fields":"function pass -> [flagged ledger items, fired words, sources]","rows":rows}));
        }
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
        // 60 km tape (AD_P3_STRIPE=1): a banded MYNN-EDMF launch gives one segment a row (3,506 a launch at 60 km).
        // Runs of them, of one allocation and one width at one pitch on both sides (the pitch in words is the same: one
        // float64 a float32 word), go as one launch. THE BREAKAGE: one launch a row is 7,012 launches a MYNN-EDMF launch
        // each way through the window. Off (or a fatbin without the pair): the head's loop, launch for launch.
        if stripe_xfer_on() && !self.f.widen2.is_null() && !self.f.narrow2.is_null() {
            for (i, j, pitch) in crate::p3_stripe::pitched_runs(segs) {
                let (al, lo, hi, d) = segs[i];
                if lo % 4 != 0 { return Err(format!("native segment of allocation {al} starts mid-word")); }
                let e0 = self.base_el.get(&al).ok_or_else(|| format!("native segment: allocation {al} has no window shadow"))? + (lo / 4) as i64;
                let (mut a, mut e, mut dd) = (self.sh_base, e0, d);
                if j - i >= 2 {
                    let (mut n, mut rows, mut pw) = (((hi - lo) / 4) as i64, (j - i) as i64, (pitch / 4) as i64);
                    launch(cuda, if widen { self.f.widen2 } else { self.f.narrow2 }, [grid1(n * rows)[0].min(65535), 1, 1], kargs!(a, e, dd, n, rows, pw))?;
                } else {
                    let mut n = (hi - lo).div_ceil(4) as i64;
                    launch(cuda, if widen { self.f.widen } else { self.f.narrow }, [grid1(n)[0].min(65535), 1, 1], kargs!(a, e, dd, n))?;
                }
            }
            return Ok(());
        }
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
            R::Axpy { y, x, n, c } => if *n > 0 {
                let (mut yp, mut xp, mut cc, mut nn) = (self.sh(*y), self.sh(*x), *c, *n as i64);
                launch(cuda, self.f.axpy, [grid1(nn)[0].min(65535), 1, 1], kargs!(yp, xp, cc, nn))?;
            },
            R::PeerSend { src, n, key } if crate::peer_shm::on() => {
                if !dev::dispatch_stream().is_null() { return Err("DEFECT peer copy inside derivative graph capture".into()); }
                crate::peer_shm::fence(cuda)?;
                crate::peer_shm::get(cuda)?.send(b't', ptr::null_mut(), key, self.sh(*src), n * 4)?;
                crate::peer_shm::fence(cuda)?;   // fenced: the sent bytes are read before any later write, any stream
            },
            R::PeerRecv { dst, n, key } if crate::peer_shm::on() => {
                if !dev::dispatch_stream().is_null() { return Err("DEFECT peer copy inside derivative graph capture".into()); }
                crate::peer_shm::fence(cuda)?;
                crate::peer_shm::get(cuda)?.recv(b't', ptr::null_mut(), key, self.sh(*dst), n * 4)?;
                crate::peer_shm::fence(cuda)?;
            },
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
            // two-cards: with AD_PEER_SHM a queued receive or move_add may still use the old buffer
            if self.peer_buf.0 != 0 && crate::peer_shm::on() { dev::sync(cuda)?; }
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
                    // machine: the rows gather_quick proved independent and injective go straight through
                    let direct: Vec<DevRow> = self.gathers[&h].as_ref().map_err(|e| e.clone())?.direct.clone();
                    if !direct.is_empty() {
                        let mut rd = self.rows_dev(cuda, &direct)?;
                        launch(cuda, self.f.rows_ad_direct, grid_rows(&direct), kargs!(a, rd, fail))?;
                    }
                    // machine round 2: fan-in groups (same source map read by several rows) go through
                    // p8_rows_ad_fan: the gather's own sums, no tables. Then there are no tables to launch.
                    let (fan, fgr): (Vec<DevRow>, Vec<i64>) = { let g = self.gathers[&h].as_ref().map_err(|e| e.clone())?; (g.fan.clone(), g.fan_groups.clone()) };
                    if !fan.is_empty() {
                        let mut rd = self.rows_dev(cuda, &fan)?;
                        let gb = std::slice::from_raw_parts(fgr.as_ptr().cast::<u8>(), fgr.len() * 8);
                        let mut gd = self.scratch.take(gb.len())?;
                        dev::upload(cuda, gd, gb)?;
                        let maxc = fan.iter().map(|r| r.count).max().unwrap_or(0);
                        launch(cuda, self.f.rows_ad_fan, [((maxc + 255) / 256).clamp(1, 1024) as u32, (fgr.len() / 2) as u32, 1], kargs!(a, rd, gd, fail))?;
                    } else {
                        let _cold = crate::spill::cold(&format!("gather:{h:x}"));
                        let g = self.gathers[&h].as_ref().map_err(|e| e.clone())?;
                        let [mut sw, mut st, mut it, mut dw, mut sl] = g.bufs;
                        let (mut ns, mut nd) = (g.nsrc, g.ndst);
                        launch(cuda, self.f.gather_ad, grid1(ns), kargs!(a, sw, st, it, dw, ns, fail))?;
                        launch(cuda, self.f.clear_dst, grid1(nd), kargs!(a, dw, sl, nd, fail))?;
                    }
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
            R::Axpy { y, x, n, c } => if *n > 0 {
                let (mut xp, mut yp, mut cc, mut nn) = (self.sh(*x), self.sh(*y), *c, *n as i64);
                launch(cuda, self.f.axpy, [grid1(nn)[0].min(65535), 1, 1], kargs!(xp, yp, cc, nn))?;
            },
            // multidevice-runner: the transpose of the cross-rank copy dst <- src is src += dst_bar, dst_bar = 0, split
            // across the two ranks: the receiving rank sends its cotangent and zeroes it, the sending rank adds it.
            // two-cards (AD_PEER_SHM=1): the same transpose, stream-ordered through the shared slots
            R::PeerRecv { dst, n, key } if crate::peer_shm::on() => {
                if !dev::dispatch_stream().is_null() { return Err("DEFECT peer copy inside derivative graph capture".into()); }
                crate::peer_shm::fence(cuda)?;
                crate::peer_shm::get(cuda)?.send(b'a', ptr::null_mut(), key, self.sh(*dst), n * 4)?;
                crate::peer_shm::fence(cuda)?;   // fenced: the sent cotangent is read before it is zeroed (as the base)
                dev::zero(cuda, self.sh(*dst), n * 4)?;
            },
            R::PeerSend { src, n, key } if crate::peer_shm::on() => {
                if !dev::dispatch_stream().is_null() { return Err("DEFECT peer copy inside derivative graph capture".into()); }
                let tmp = self.peer_tmp(cuda, n * 4)?;
                crate::peer_shm::fence(cuda)?;
                crate::peer_shm::get(cuda)?.recv(b'a', ptr::null_mut(), key, tmp, n * 4)?;
                crate::peer_shm::fence(cuda)?;
                self.support.move_add(cuda, self.sh(*src), tmp, n * 4, 4)?;
            },
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
        self.tangent_command(ex, r, l, step, |ex| exec(ex, op, step, base, frame))
    }

    /// Run a live command once between its pre-launch saves and its tangent sweep.
    /// The caller supplies an admitted rule and resolved owner-relative arguments.
    /// This is opt-in: the recorded route keeps its existing rule/exec providers.
    ///
    /// Safety: `forward` must enqueue only this command, on the ordered dispatch
    /// stream (or establish the same dependencies), and may not re-enter this Win.
    /// Owners, shadows and entry images must remain valid through queued consumers.
    /// TL-only saves use the existing per-invocation arena; do not use that arena
    /// for a reverse tape concurrently. On failure abort this pass, do not retry.
    pub(crate) unsafe fn live_tangent<F>(&mut self, ex: &mut Execution, r: R,
        l: Option<LiveLaunch>, step: usize, forward: F) -> Result<usize, String>
    where F: FnOnce(&mut Execution) -> Result<(), String> {
        if let R::Refused(reason) = &r { return Err(format!("live tangent: unadmitted rule: {reason}")); }
        match (&r, &l) {
            (R::Entry, Some((f, ..))) => {
                if !self.kernels.contains_key(f) { return Err(format!("live tangent: no derivative entry for {f}")); }
            },
            (R::Entry | R::Native(_) | R::Tempo(_) | R::TempoSerial(_) | R::TempoSmall(_), None) => {
                return Err("live tangent: derivative rule requires a resolved launch".into());
            },
            (R::Native(f) | R::Tempo(f) | R::TempoSerial(f) | R::TempoSmall(f), Some((name, ..))) if f != name => {
                return Err(format!("live tangent: rule {f} does not match launch {name}"));
            },
            _ => {},
        }
        // Recorded rule_inner prepares the same gather before entering the walk.
        if let R::Rows { rows, .. } = &r { self.gather_for(&ex.cuda, rows)?; }
        self.tangent_command(ex, r, l, step, forward)
    }

    /// Shared pre / forward / post walk. Keep the forward call at each original
    /// exec site: moving it across a save or a shadow store changes in-place J.
    unsafe fn tangent_command<F>(&mut self, ex: &mut Execution, r: R,
        l: Option<LiveLaunch>, step: usize, forward: F) -> Result<usize, String>
    where F: FnOnce(&mut Execution) -> Result<(), String> {
        let light = ordered(ex) && !matches!(r, R::Tempo(_) | R::TempoSerial(_) | R::TempoSmall(_));
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
                forward(ex)?;
                let fzw = if self.fz_pre.is_some() { self.fz_words_enqueue(ex)? } else { None };
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
                self.fz_words = fzw;
                Ok(est)
            },
            (R::Max { prim: (al, first, n), .. }, _) => {
                self.tape.reset();
                let p = self.tape_primal(ex, *al, *first, *n, 4)?;
                forward(ex)?;
                self.tl(&ex.cuda, &r, p)?;
                self.row("rule:max", [0, 0, 0, 0, n * 4]);
                Ok(n * 4)
            },
            (R::Where { prim: (al, first, n), .. }, _) => {
                self.tape.reset();
                let p = self.tape_primal(ex, *al, *first, *n, 1)?;
                forward(ex)?;
                self.tl(&ex.cuda, &r, p)?;
                self.row("rule:where", [0, 0, 0, 0, *n]);
                Ok(*n)
            },
            (R::Native(_), Some((f, args, grid, block, _))) => {
                let t0 = Instant::now();
                crate::spill::phase(&format!("tl{step}:{f}"));
                let queued = ordered(ex) && !self.trace;
                self.tape.reset(); self.scratch.reset();
                let tp = dev::prof::start();
                // memory (AD_TL_BIND_SCRATCH=1). THE BREAKAGE THIS PREVENTS: the MYNN-EDMF tangent bind put its float64
                // shadow region, its seed snapshot region and its pre-launch copies all on the tape arena. Each covers the
                // whole mesh, whatever column batch the launch has (the extents of the row are the memory dimensions
                // ims:ime): 4.60 + 4.60 + 2.30 GB per launch at x1.163842, a quarter of that at x1.40962. At x1.163842 a
                // 7 GiB arena was therefore doubled to 15.03 GB in step 1 and stayed that size for the whole window, while
                // the reverse binds the same two regions from the reverse scratch, which is then sized for them a second
                // time (13.55 GB): 28.6 GB of the dycore card for one 9.19 GB bind (x1.163842, a test machine
                // <local path>: tl_tape_grown, tape_bytes, scratch_fit). With the switch
                // the tangent binds the row as the reverse does (prepare_deferred: pre-launch copies on the tape; the
                // two regions from the scratch, which is reset before every native launch), so the tape arena keeps its
                // --tape-gib size and the scratch is made once at the size the reverse needs. Where the two regions
                // live is the only change: the same takes in the same order, the same words.
                let tl_split = self.pack && tl_bind_scratch() && self.native.as_ref().and_then(|n| n.row(&f)).is_some_and(|r| r.kind == "p3_mynn");
                let mut p = loop {
                    let n = self.native.as_mut().ok_or("native rows not loaded")?;
                    let prepared = if tl_split && queued {
                        n.prepare_deferred_ordered(&ex.cuda, &mut self.tape, &ex.allocations, &f, &args, grid, block)
                    } else if tl_split {
                        n.prepare_deferred(&ex.cuda, &mut self.tape, &ex.allocations, &f, &args, grid, block)
                    } else if queued {
                        n.prepare_ordered(&ex.cuda, &mut self.tape, &ex.allocations, &f, &args, grid, block)
                    } else {
                        n.prepare(&ex.cuda, &mut self.tape, &ex.allocations, &f, &args, grid, block)
                    };
                    match prepared {
                        Ok(p) => break p,
                        Err(e) if self.grow_tape_for(ex, &e, step, &f)? => { self.tape.reset(); self.scratch.reset(); }
                        Err(e) => return Err(format!("step {step} {f} native tangent bind: {e}")),
                    }
                };
                dev::prof::add(2, tp);
                // (scratch peak before this launch, bytes of the two regions): the regions are not workspace, so the
                // peak the reverse scratch base is fitted from is put back without them after the tangent entries.
                let mut tl_ws: Option<(usize, usize)> = None;
                if tl_split {
                    let need = p.p3_transient_need().ok_or_else(|| format!("step {step} {f}: AD_TL_BIND_SCRATCH: not a deferred MYNN-EDMF bind"))?;
                    self.tl_scratch_room(ex, need, step, &f)?;
                    let pk0 = std::mem::take(&mut self.scratch.peak);
                    let n = self.native.as_mut().ok_or("native rows not loaded")?;
                    n.bind_transient(&ex.cuda, &mut self.scratch, &ex.allocations, &mut p).map_err(|e| format!("step {step} {f} native tangent transient bind: {e}"))?;
                    p.transient_bytes = self.scratch.used;
                    tl_ws = Some((pk0, self.scratch.used));
                }
                // the row's estimate counts the two regions wherever they live (tape sizing and chunk planning read it)
                let est = self.tape.used + tl_ws.map_or(0, |w| w.1);
                // tape-dedupe-and-pack: the reverse binds this row's transient buffers from the reverse scratch.
                // 60 km tape (AD_P3_STRIPE=1): a MYNN-EDMF row bound whole here is bound banded in the reverse, so the
                // reverse scratch is sized for the banded regions (p3_banded_need; None when off or not bandable).
                if self.pack { self.rev_scratch = self.rev_scratch.max(if tl_split { p.transient_bytes } else { p.p3_banded_need().unwrap_or(p.transient_bytes) }); }
                if !queued { dev::sync(&ex.cuda)?; self.sync_owed = false; }
                let mut ids: Vec<u64> = p.active_segments().iter().map(|s| s.0).collect();
                ids.sort_unstable(); ids.dedup();
                let before = if self.trace { Some((self.alloc_norms(&ex.cuda, &ids)?, self.state_norm(&ex.cuda)?)) } else { None };
                let seg_before = if self.trace { Some(self.seg_norms(ex, &p.active_segments())?) } else { None };
                forward(ex)?;
                let cuda = &ex.cuda;
                let segs = p.active_segments();
                self.xfer(cuda, &segs, true)?;
                let tn = dev::prof::start();
                // memory (cold tier): this scheme's scratch is cold, by family (p3, p4, p6, ruc, tempo, p7).
                let ctag = self.native.as_ref().and_then(|n| n.row(&f)).map(|r| r.kind.split('_').next().unwrap_or("native").to_owned()).unwrap_or_else(|| "native".into());
                let _cold = crate::spill::cold(&ctag);
                let nn = self.native.as_mut().unwrap();
                nn.ledger = self.ledger.take();
                nn.p3_step = step as i64;
                let info = nn.tangent(cuda, &mut self.scratch, &mut p, false);
                self.ledger = nn.ledger.take();
                if let Some((pk0, tr)) = tl_ws { self.scratch.peak = pk0.max(self.scratch.peak.saturating_sub(tr)); }
                let info = info.map_err(|e| format!("step {step} {f} native tangent: {e}"))?;
                dev::prof::add(8, tn);
                self.xfer(cuda, &segs, false)?;
                if queued { self.sync_owed = true; } else { dev::sync(cuda)?; }
                eprintln!("{}", json!({"event":"window_native","pass":"tl","step":step,"function":f,"ms":t0.elapsed().as_secs_f64() * 1e3,
                    "timing_kind":if queued { "enqueue_wall" } else { "completed_wall" },"info":info}));
                self.row(&format!("native:{f}"), [0, 0, 0, 0, est]);
                crate::spill::phase(&format!("tl{step}"));
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
                forward(ex)?;
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
                forward(ex)?;
                self.small_sweep(&ex.cuda, &f, &snap, &ins, &outs, false, true).map_err(|e| format!("step {step} {f} tempo-small tangent: {e}"))?;
                self.row(&format!("tempo_small:{f}"), [0, 0, 0, 0, est]);
                self.note("tl", &f, t0);
                Ok(est)
            },
            (R::FctMinmax { prim, .. }, _) => {
                self.tape.reset();
                let p = self.tape_fct(ex, prim)?;
                forward(ex)?;
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
                forward(ex)?;
                crate::spill::phase(&format!("tl{step}:{f}"));
                self.tempo_apply(ex, &snap, step, false).map_err(|e| format!("step {step} {f} TEMPO tangent: {e}"))?;
                crate::spill::phase(&format!("tl{step}"));
                self.row(&format!("tempo:{f}"), [0, 0, 0, 0, est]);
                self.note("tl", &f, t0);
                Ok(est)
            },
            _ => { forward(ex)?; self.tl(&ex.cuda, &r, 0)?; Ok(0) },
        }
    }

    /// x20 fused tape: one tangent command that also records its reverse tape entry into `rtape` (see the field).
    unsafe fn fused_tangent(&mut self, ex: &mut Execution, op: &Operation, step: usize, base: &Path, frame: &mut Option<ReplayFrame>) -> Result<usize, String> {
        let (r, l) = self.rule(ex, op, step, base, frame.as_ref())?;
        let mut pre = None;
        let row_off = self.rtape.as_ref().map(|t| t.used).unwrap_or(0);
        if self.fuse_ok {
            let mut rt = self.rtape.take().ok_or("fused tape arena missing")?;
            std::mem::swap(&mut self.tape, &mut rt);
            let p = self.rec_pre(ex, &r, &l, step);
            std::mem::swap(&mut self.tape, &mut rt);
            self.rtape = Some(rt);
            match p? { Some(p) => pre = Some(p), None => self.fz_overflow(step, "pre-launch") }
        }
        self.rule_cache = Some((r, l));
        // revmach early checks: the pre-launch part rides into the tangent, which queues this launch's word checks right
        // after the forward (before its tangent entries) and reads them back by event, not by a drain after the entries.
        let early = ordered(ex) && std::env::var("AD_FUSE_EARLY_CHECKS").as_deref() != Ok("0") && std::env::var("AD_CHECK_BATCH").as_deref() != Ok("0")
            && matches!(pre, Some(FzPre::Entry { .. }));
        if early {
            if self.fz_wbuf.is_none() { self.fz_wbuf = Some(crate::pinned::PinnedBuf::new(&ex.cuda, dev::SLOTS * 24)?); }
            if self.fz_wev.is_null() { checked("cuEventCreate fused checks", (ex.cuda.event_create)(&mut self.fz_wev, 2))?; }
            self.fz_pre = pre.take();
        }
        self.fz_words = None;
        let est = self.tangent(ex, op, step, base, frame);
        if early { pre = self.fz_pre.take(); }
        self.rule_cache = None;
        let est = est?;
        if let Some(p) = pre {
            let mut rt = self.rtape.take().ok_or("fused tape arena missing")?;
            std::mem::swap(&mut self.tape, &mut rt);
            let t = self.rec_post(ex, p);
            std::mem::swap(&mut self.tape, &mut rt);
            self.rtape = Some(rt);
            match t? { Some(tp) => { if let Some((_, v)) = self.fz_cur.as_mut() { v.push(tp); self.fz_cur_offs.push(row_off); } }, None => self.fz_overflow(step, "post-launch") }
        }
        Ok(est)
    }
    /// revmach early checks: queue the word checks rec_post needs for the launch whose forward just ran (the same slots in
    /// the same order as tape_range), then one asynchronous readback into page-locked memory and an event. Reads only.
    unsafe fn fz_words_enqueue(&self, ex: &Execution) -> Result<Option<FzWords>, String> {
        let Some(FzPre::Entry { f, b, args, grid, block, shared }) = self.fz_pre.as_ref() else { return Ok(None) };
        let li = deriv::LaunchInfo { args, grid: *grid, block: *block, shared: *shared };
        let posts: Vec<(u64, (usize, usize))> = if self.pack { self.kernels[f].post_spans(&li, &b.env, &ex.allocations)?.into_iter().collect() } else { Vec::new() };
        let cands: Vec<(Option<(usize, u64)>, Option<(usize, u64)>)> = posts.iter().map(|&(al, (lo, hi))| self.precache.candidates(al, lo, hi)).collect();
        let nsh = b.shared.len();
        let nslot = nsh + 2 * posts.len();
        if !(nslot > 1 && nslot <= dev::SLOTS) { return Ok(None); }
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
        let buf = self.fz_wbuf.as_ref().ok_or("fused check buffer missing")?;
        checked("cuMemcpyDtoHAsync fused checks", (ex.cuda.download_async)(buf.as_ptr() as *mut std::ffi::c_void, self.support.counts_addr(), nslot * 24, std::ptr::null_mut()))?;
        checked("cuEventRecord fused checks", (ex.cuda.event_record)(self.fz_wev, std::ptr::null_mut()))?;
        Ok(Some(FzWords { posts, cands, nsh, nslot }))
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
                // revmach: the same host-drain rule as tape_range (x20 replay-compiled): no drain when every command of
                // the plan is ordered on the legacy stream. THE BREAKAGE THIS FIXES: the fused recording drained the card
                // after every taped launch, so recording in the tangent pass cost what the reverse replay it replaces cost.
                if !ordered(ex) { dev::sync(&ex.cuda)?; }
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
        // revmach: every word check this launch needs (shared pre-launch spans, post-span candidates) queued into word
        // slots and read back with ONE collect, decided in the old order, exactly as tape_range does (x20 replay-compiled).
        // Same decisions, same copies, same words; AD_CHECK_BATCH=0 keeps one readback per check.
        let early = self.fz_words.take();
        let (posts, cands, nsh, nslot, early_words) = if let Some(w) = early {
            if w.nsh != b.shared.len() { return Err(format!("DEFECT fused early checks for {f}: {} shared spans queued, {} at rec_post", w.nsh, b.shared.len())); }
            checked("cuEventSynchronize fused checks", (ex.cuda.event_synchronize)(self.fz_wev))?;
            let buf = self.fz_wbuf.as_ref().ok_or("fused check buffer missing")?;
            let raw = std::slice::from_raw_parts(buf.as_ptr().cast::<u64>(), w.nslot * 3);
            let wv: Vec<[u64; 3]> = raw.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
            self.fz_early += 1;
            (w.posts, w.cands, w.nsh, w.nslot, Some(wv))
        } else {
            let posts: Vec<(u64, (usize, usize))> = if self.pack { self.kernels[&f].post_spans(&li, &b.env, &ex.allocations)?.into_iter().collect() } else { Vec::new() };
            let cands: Vec<(Option<(usize, u64)>, Option<(usize, u64)>)> = posts.iter().map(|&(al, (lo, hi))| self.precache.candidates(al, lo, hi)).collect();
            let nsh = b.shared.len();
            let nslot = nsh + 2 * posts.len();
            (posts, cands, nsh, nslot, None)
        };
        let words: Option<Vec<[u64; 3]>> = if early_words.is_some() { early_words } else if ordered(ex) && nslot > 1 && nslot <= dev::SLOTS && std::env::var("AD_CHECK_BATCH").as_deref() != Ok("0") {
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
            if !ordered(ex) { dev::sync(&ex.cuda)?; }
            for (k, (al, lo, hi, at)) in std::mem::take(&mut b.shared).into_iter().enumerate() {
                let live = ex.allocations.get(&al).ok_or("shared primal allocation not live")?.address + lo as u64;
                let same = match &words { Some(wv) => hi <= lo || wv[k][0] == 0, None => deriv::same_words(&ex.cuda, &self.support, live, at, hi - lo)? };
                if same { b.shared.push((al, lo, hi, at)); continue; }
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
        self.fz_cur_offs.clear();
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
            let base = self.rtape.as_ref().unwrap().base;
            if std::env::var("AD_FUSE_FLUSH_SYNC").as_deref() == Ok("1") {
                checked("cuMemcpyDtoH fused tape", (cuda.download)(buf.as_mut_ptr().cast(), base, used))?;
            } else {
                // revmach: the chunk is copied out asynchronously on its own stream, in pieces, and the arena carries a
                // write fence per piece: the next step's recording, which reuses the arena from offset 0, waits on the
                // device for exactly the piece it is about to overwrite. THE BREAKAGE THIS FIXES: a blocking copy of the
                // whole chunk (pk120: one 18.6 GB step) held the host, so the tangent pass stopped issuing work while it ran.
                // Same bytes in RAM; the reverse uploads them after the chunk's last piece has completed (fz_done).
                if let Some(f) = self.rtape.as_mut().unwrap().fence.as_mut() { f.upto(usize::MAX)?; }
                if self.fz_copy.is_null() { checked("cuStreamCreate fused copy-out", (cuda.stream_create)(&mut self.fz_copy, 1))?; }
                let mut ev0 = std::ptr::null_mut();
                checked("cuEventCreate fused copy-out", (cuda.event_create)(&mut ev0, 2))?;
                self.fz_evs.push(ev0);
                checked("cuEventRecord fused copy-out", (cuda.event_record)(ev0, std::ptr::null_mut()))?;
                checked("cuStreamWaitEvent fused copy-out", (cuda.stream_wait_event)(self.fz_copy, ev0, 0))?;
                let piece = std::env::var("AD_FUSE_PIECE_MIB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(256).max(1) << 20;
                let mut pieces = std::collections::VecDeque::new();
                let (mut off, mut last) = (0usize, std::ptr::null_mut());
                while off < used {
                    let n = piece.min(used - off);
                    checked("cuMemcpyDtoHAsync fused tape", (cuda.download_async)(buf.as_mut_ptr().add(off).cast(), base + off as u64, n, self.fz_copy))?;
                    let mut ev = std::ptr::null_mut();
                    checked("cuEventCreate fused piece", (cuda.event_create)(&mut ev, 2))?;
                    self.fz_evs.push(ev);
                    checked("cuEventRecord fused piece", (cuda.event_record)(ev, self.fz_copy))?;
                    pieces.push_back((off, ev));
                    last = ev; off += n;
                }
                self.fz_done.insert(ci, last);
                let t = self.rtape.as_mut().unwrap();
                let waited = t.fence.as_ref().map(|f| f.waited).unwrap_or(0);
                t.fence = Some(dev::Fence { pieces, wait: cuda.stream_wait_event, waited });
            }
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
            let offs = std::mem::take(&mut self.fz_cur_offs);
            if offs.len() == tape.len() { self.fz_offs.insert(step, offs); }
            self.fz_tapes.insert(step, (ci, tape));
        } else {
            if let Some(t) = self.rtape.as_mut() { t.used = self.fz_step_start; }
            // a step that does not fit an empty arena will not fit later either: stop recording.
            if self.fz_step_start == 0 { self.fz_failed = true; }
            self.fz_force_flush = true;
            self.fz_info.push(json!({"step":step,"chunk":ci,"fused":false}));
            self.fz_cur_offs.clear();
            drop(tape);
        }
        self.fz_resident = Some(ci);
        self.fuse_ok = true;
        Ok(())
    }
    /// revmach prefetch: issue every pending piece the walk-back has freed: after row `j` (rows >= j reversed, so the
    /// arena from offs[j] up is no longer read), or all of them when `all` (the walk-back is over).
    unsafe fn fz_pf_issue(&mut self, cuda: &Cuda, j: usize, all: bool) -> Result<(), String> {
        let mut evs = Vec::new();
        let Some(pf) = self.fz_pf.as_mut() else { return Ok(()) };
        let floor = if all { 0 } else { match pf.offs.get(j) { Some(&o) => o, None => return Ok(()) } };
        let mut ev_row: *mut std::ffi::c_void = std::ptr::null_mut();
        while let Some(&(a, k)) = pf.pending.last() {
            if a < floor { break; }
            if ev_row.is_null() {
                checked("cuEventCreate fused prefetch", (cuda.event_create)(&mut ev_row, 2))?;
                evs.push(ev_row);
                checked("cuEventRecord fused prefetch", (cuda.event_record)(ev_row, std::ptr::null_mut()))?;
                checked("cuStreamWaitEvent fused prefetch", (cuda.stream_wait_event)(pf.stream, ev_row, 0))?;
            }
            match pf.runs.as_ref() {
                None => checked("cuMemcpyHtoDAsync fused prefetch", (cuda.upload_async)(pf.base + a as u64, pf.host.add(a).cast(), k, pf.stream))?,
                Some(runs) => {
                    // speed tape60: this piece's bytes run by run, each from where its words sit in host memory
                    let mut i = runs.partition_point(|r| r.0 + r.1 <= a);
                    let mut covered = 0usize;
                    while i < runs.len() && runs[i].0 < a + k {
                        let (ro, rl, hp) = runs[i];
                        let (lo, hi) = (ro.max(a), (ro + rl).min(a + k));
                        checked("cuMemcpyHtoDAsync streamed tape run", (cuda.upload_async)(pf.base + lo as u64, ((hp + (lo - ro)) as *const u8).cast(), hi - lo, pf.stream))?;
                        covered += hi - lo;
                        i += 1;
                    }
                    if covered != k { return Err(format!("DEFECT streamed tape: chunk {} piece at {a}: {covered} of {k} bytes have a host copy", pf.ci)); }
                },
            }
            let mut ev = std::ptr::null_mut();
            checked("cuEventCreate fused prefetch piece", (cuda.event_create)(&mut ev, 2))?;
            evs.push(ev);
            checked("cuEventRecord fused prefetch piece", (cuda.event_record)(ev, pf.stream))?;
            pf.done.push(ev);
            pf.pending.pop();
            if !all { pf.issued_during_walk += k; }
        }
        self.fz_evs.extend(evs);
        Ok(())
    }
    /// x20 fused tape: the reverse of a fused step: the state of the step's end boundary (what the one-pass replay
    /// reaches), the step's tape in the arena (uploaded to the addresses it was recorded at when its chunk is not the
    /// one resident), and the walk-back. No replay, no anchors, no segments.
    unsafe fn fused_reverse_step(&mut self, ex: &mut Execution, frame: &mut Option<ReplayFrame>, step: usize, last: bool, cur: &Boundary, cur_data: &[u8]) -> Result<Value, String> {
        let _profile = crate::step_profile::StepRange::begin("ad", step)?;
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
        let mut prefetched = false;
        if let Some(pf) = self.fz_pf.take() {
            // revmach prefetch: this step's chunk was uploaded during the previous walk-back; the legacy stream waits
            // for its pieces on the device (no host wait) and the chunk is resident.
            if pf.ci == ci && pf.pending.is_empty() {
                for &ev in &pf.done { checked("cuStreamWaitEvent fused prefetch", (ex.cuda.stream_wait_event)(std::ptr::null_mut(), ev, 0))?; }
                self.fz_resident = Some(ci);
                prefetched = true;
                self.fz_pf_info.push(json!({"step":step,"chunk":ci,"bytes":pf.bytes,"issued_during_walk":pf.issued_during_walk,"pieces":pf.done.len()}));
            } else {
                return Err(format!("DEFECT fused prefetch for chunk {} incomplete or for the wrong chunk at step {step} (chunk {ci})", pf.ci));
            }
        }
        if self.fz_resident != Some(ci) {
            let used = self.fz_chunks[ci].2;
            let base = self.rtape.as_ref().ok_or("fused tape arena missing")?.base;
            let host = self.fz_chunks[ci].1.as_ref().ok_or("fused tape chunk is neither resident nor in RAM")?;
            if std::env::var("AD_FUSE_FLUSH_SYNC").as_deref() == Ok("1") { dev::upload(&ex.cuda, base, &host[..used])?; }
            else {
                // revmach: a stream-ordered upload from the page-locked chunk (after its copy-out completed on the device),
                // not a blocking copy: the host goes on issuing the walk-back while it runs. The chunk is dropped only after
                // the blocking failure-flag read below, which follows the upload on the same stream.
                if let Some(f) = self.rtape.as_mut().and_then(|t| t.fence.as_mut()) { f.upto(usize::MAX)?; }
                if let Some(&ev) = self.fz_done.get(&ci) { checked("cuStreamWaitEvent fused upload", (ex.cuda.stream_wait_event)(std::ptr::null_mut(), ev, 0))?; }
                checked("cuMemcpyHtoDAsync fused tape", (ex.cuda.upload_async)(base, host.as_ptr().cast(), used, std::ptr::null_mut()))?;
            }
            h2d = used;
            self.fz_resident = Some(ci);
        }
        let h2d_s = t_h.elapsed().as_secs_f64();
        // revmach prefetch: the next reverse step's chunk, if it is another chunk held in RAM, is uploaded during this
        // walk-back, each piece once every row that could still read the bytes it overwrites has been reversed.
        // AD_FUSE_PREFETCH=0 uploads it at the next step as before (same bytes, same addresses).
        if step > 1 && std::env::var("AD_FUSE_PREFETCH").as_deref() != Ok("0") && std::env::var("AD_FUSE_FLUSH_SYNC").as_deref() != Ok("1") {
            let next = self.fz_tapes.get(&(step - 1)).map(|(c, _)| *c);
            if let (Some(cj), Some(offs)) = (next, self.fz_offs.get(&step)) {
                if cj != ci && offs.len() == n {
                    if let Some(host) = self.fz_chunks[cj].1.as_ref() {
                        let used = self.fz_chunks[cj].2;
                        let hp = host.as_ptr();
                        let offs = offs.clone();
                        if self.fz_copy.is_null() { checked("cuStreamCreate fused copy", (ex.cuda.stream_create)(&mut self.fz_copy, 1))?; }
                        let piece = std::env::var("AD_FUSE_PIECE_MIB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(256).max(1) << 20;
                        let mut pending = Vec::new();
                        let mut a = 0usize;
                        while a < used { let k = piece.min(used - a); pending.push((a, k)); a += k; }
                        if let Some(&ev) = self.fz_done.get(&cj) { checked("cuStreamWaitEvent fused prefetch copy-out", (ex.cuda.stream_wait_event)(self.fz_copy, ev, 0))?; }
                        let base = self.rtape.as_ref().ok_or("fused tape arena missing")?.base;
                        self.fz_pf = Some(FzPrefetch { ci: cj, base, host: hp, offs, pending, done: Vec::new(), stream: self.fz_copy, issued_during_walk: 0, bytes: used, runs: None });
                    }
                }
            }
        }
        let cuda = &ex.cuda as *const Cuda;
        let t_w = Instant::now();
        self.walk_back(&*cuda, ex, step, tape)?;
        let walk_s = t_w.elapsed().as_secs_f64();
        if self.fz_pf.is_some() { self.fz_pf_issue(&ex.cuda, 0, true)?; }
        self.step_flush(&ex.cuda, step, "ad")?;
        drain_pinned(ex)?;
        let mut fl = [0i32];
        checked("cuMemcpyDtoH failure", (ex.cuda.download)(fl.as_mut_ptr().cast(), self.fail, 4))?;
        if fl[0] != 0 { return Err(format!("step {step}: P8 row transpose saw a non-finite cotangent")); }
        if !self.fz_tapes.values().any(|(c, _)| *c == ci) { self.fz_chunks[ci].1 = None; }
        let host_prof: Vec<Value> = dev::prof::take().into_iter().filter(|x| x.2 > 0).map(|(k, s, c)| json!([k, s, c])).collect();
        let pack = self.pack_json();
        Ok(json!({"step":step,"commands":n,"segments":0,"fused":true,"overflow_splits":0,"commands_replayed":0,"host_prof":host_prof,"pack":pack,
                  "stages_s":{"restore":restore_s,"tape_upload":h2d_s,"walk_back":walk_s},"restored":restored,"tape_upload_bytes":h2d,"chunk":ci,"prefetched":prefetched,
                  "adjoint_entry_s":(self.entry_ms - ms0) / 1e3,"pass1_s":0.0,"wall_s":t0.elapsed().as_secs_f64()}))
    }

    /// One executed leaf of the walk: forward (with the tangent when `tangent`), then its entry in the step list.
    #[allow(clippy::too_many_arguments)]
    unsafe fn leaf(&mut self, ex: &mut Execution, command: &Operation, step: usize, base: &Path, frame: &mut Option<ReplayFrame>,
                   flat: &mut Vec<Flat>, tangent: bool) -> Result<(), String> {
        let rk = ex.rk.clone();
        let watch = if tangent { self.g6_watch_ids(ex, command, step, frame)? } else { None };
        let before = match &watch { Some((_, ids)) => Some(self.alloc_norms(&ex.cuda, ids)?), None => None };
        let tw = dev::sites::start();
        let est = if tangent { if self.fz_cur.is_some() { self.fused_tangent(ex, command, step, base, frame)? } else { self.tangent(ex, command, step, base, frame)? } } else { exec(ex, command, step, base, frame)?; 0 };
        if tw.is_some() { let l = format!("walk:{}", self.last_label); dev::sites::add(&l, tw); }
        if self.hold.on() { let rp = self.hold.regroup; self.hold_hook(ex, command, step, rp)?; }
        if let (Some((f, ids)), Some(b)) = (&watch, &before) {
            let a = self.alloc_norms(&ex.cuda, ids)?;
            let rows: Vec<Value> = ids.iter().zip(b.iter().zip(a.iter())).map(|(id, (x, y))| json!([id, x.sqrt(), y.sqrt()])).collect();
            eprintln!("{}", json!({"event":"g6_watch","step":step,"index":flat.len(),"function":f,"alloc_before_after":rows}));
        }
        if tangent && self.trace { self.trace_norm(&ex.cuda, step, flat.len(), command)?; }
        // hunt-120-start (AD_WORD_PROBE with AD_WORD_PROBE_STEPS, off by default): the asked words after this command.
        if tangent && crate::word_probe::cmd_on("tangent", step, flat.len()) {
            let what = match command { Operation::Kernel { function, .. } => function.clone(), _ => format!("{:?}", std::mem::discriminant(command)) };
            self.word_probe(&ex.cuda, "tangent", "command", step, Some((flat.len(), what.as_str())), &Value::Null)?;
        }
        flat.push(Flat { op: command as *const Operation, rk, est });
        Ok(())
    }
    /// anyparam: a command the forward itself made (not a plan command) run as a leaf: it is owned by the window
    /// (`synth`, boxed, so the step lists' pointers stay valid) and enters the step list like any plan command.
    #[allow(clippy::too_many_arguments)]
    unsafe fn synth_leaf(&mut self, ex: &mut Execution, op: Operation, step: usize, base: &Path, frame: &mut Option<ReplayFrame>,
                         flat: &mut Vec<Flat>, tangent: bool) -> Result<(), String> {
        self.synth.push(Box::new(op));
        let p: *const Operation = &**self.synth.last().unwrap();
        self.leaf(ex, &*p, step, base, frame, flat, tangent)
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
                    self.hold.regroup = regroup.as_ref().map(|r| r.pass);
                    match self.leaf(ex, command, step, base, frame, flat, tangent) {
                        // anyparam: a checked read-back that moved names what it guards (the forward is where it is met first).
                        Err(e) if matches!(command, Operation::DeviceToHost { .. }) && e.contains(crate::READ_DIFFERS) =>
                            return Err(format!("{e}. The value is {}", read_producers(commands, index))),
                        r => r?,
                    }
                    if let Some(hist) = ex.lw_live_hist.take() {
                        if regroup.is_some() { return Err(format!("step {step}: live LW histogram inside a LW regroup section")); }
                        // anyparam: later passes over this step's list (reverse replays, redo) read the histogram THIS
                        // forward followed: the list gets the read with the live histogram as its checked value (a replay
                        // that reads another one refuses), then the regroup's own commands.
                        if let Operation::DeviceToHost { stream, source, bytes, asynchronous, host_consumer, .. } = command {
                            let live: Vec<u8> = hist.iter().flat_map(|v| v.to_le_bytes()).collect();
                            self.synth.push(Box::new(Operation::DeviceToHost { stream: *stream, source: *source, bytes: *bytes, asynchronous: *asynchronous,
                                host_consumer: host_consumer.clone(), expected_hex: Some(crate::hex_bytes(&live)), expected_zero_words: None, layer_groups: None, proof: None }));
                            if let Some(last) = flat.last_mut() { last.op = &**self.synth.last().unwrap() as *const Operation; }
                        }
                        let (r, order0) = lw_regroup_begin(ex, commands, index, step, &hist)?;
                        self.lw_follow.push(json!({"step":step,"tangent":tangent,"passes":r.passes.len(),"section_commands":r.end - r.start}));
                        self.synth_leaf(ex, order0, step, base, frame, flat, tangent)?;
                        dev::sync(&ex.cuda)?;
                        regroup = Some(r);
                    }
                },
            }
            if let Some(r) = regroup.as_mut() {
                if index == r.end {
                    let (more, ops) = lw_regroup_next(ex, r, step)?;
                    let start = r.start;
                    for op in ops { self.synth_leaf(ex, op, step, base, frame, flat, tangent)?; }
                    dev::sync(&ex.cuda)?;
                    if more { index = start + 1; continue; }
                    regroup = None;
                }
            }
            index += 1;
        }
        if regroup.is_some() { return Err(format!("step {step}: LW regroup section did not close")); }
        Ok(())
    }

    /// truth: see `Hold`. After a held kernel's launch: the base forward keeps its output words; a perturbed
    /// forward counts the words that differ from the base and (unless census) writes the base words back.
    /// `regroup_pass`: the pass of a live LW regroup section the launch is in (None outside one).
    unsafe fn hold_hook(&mut self, ex: &Execution, command: &Operation, step: usize, regroup_pass: Option<usize>) -> Result<(), String> {
        use crate::plan::Argument as A;
        let Operation::Kernel { function, arguments, .. } = command else { return Ok(()) };
        let (arg, unit, count_arg) = match self.hold.specs.iter().find(|s| s.function == *function) { Some(s) => (s.arg, s.unit, s.count_arg), None => return Ok(()) };
        let e = |m: String| format!("step {step}: hold {function}: {m}");
        let count = match arguments.get(count_arg) {
            Some(A::Bytes { bytes_hex, .. }) if bytes_hex.len() >= 8 => {
                let b: Vec<u8> = (0..4).map(|i| u8::from_str_radix(&bytes_hex[2 * i..2 * i + 2], 16).unwrap_or(0)).collect();
                i32::from_le_bytes([b[0], b[1], b[2], b[3]])
            },
            _ => return Err(e(format!("argument {count_arg} is not a recorded int32 count"))),
        };
        if count <= 0 { return Err(e(format!("count {count}"))); }
        let Some(A::Device { target }) = arguments.get(arg) else { return Err(e(format!("argument {arg} is not a device array"))) };
        let bytes = unit * count as usize;
        let address = ex.address(*target, bytes)?;
        if regroup_pass.is_some_and(|p| p > 0) {
            if self.hold.mode != 0 { self.hold.launches += 1; self.hold.unheld += 1; }
            return Ok(());
        }
        let key = (step, self.hold.ord);
        self.hold.ord += 1;
        dev::sync(&ex.cuda)?;
        if self.hold.mode == 0 {
            let d = dev::download(&ex.cuda, address, bytes)?;
            self.hold.store.insert(key, (function.clone(), d));
            return Ok(());
        }
        self.hold.launches += 1;
        if regroup_pass.is_some() { self.hold.unheld += 1; return Ok(()); }
        let base = match self.hold.store.get(&key) {
            Some((f0, b)) if f0 == function && b.len() == bytes => b.clone(),
            Some((f0, b)) => return Err(e(format!("launch {} of the step was {f0} with {} bytes in the base forward, now {bytes} bytes", key.1, b.len()))),
            None => return Err(e(format!("launch {} of the step has no base words", key.1))),
        };
        let live = dev::download(&ex.cuda, address, bytes)?;
        let differ = live.chunks_exact(4).zip(base.chunks_exact(4)).filter(|(a, b)| a != b).count() as u64;
        self.hold.words += (bytes / 4) as u64; self.hold.differ += differ;
        let s = self.hold.by_step.entry(step).or_insert((0, 0)); s.0 += (bytes / 4) as u64; s.1 += differ;
        if self.hold.apply && differ > 0 { dev::upload(&ex.cuda, address, &base)?; dev::sync(&ex.cuda)?; }
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
        // anyparam: every forward that is not compared with the recorded reference follows the live histogram,
        // the tangent forward too (the regroup is written into the step list, see walk). A reference-checked forward
        // still refuses a histogram that differs from the recorded one, as before.
        ex.lw_regroup = refs.is_none() || self.row_live || (!tangent && test != 0);
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
        self.hold.ord = 0; self.hold.regroup = None;
        let blocks = match frame.as_mut() {
            Some(f) => { f.begin(step)?; f.blocks(plan.physics.as_ref().ok_or("missing physics plan")?)? },
            None => plan.cadence[step - 1].clone(),
        };
        self.stage_blobs(ex, base, self.blob_index.get(&blocks), None)?;
        let t_stage = t_s.elapsed().as_secs_f64();
        let mut flat = Vec::new();
        if tangent && self.fuse_rec { self.fz_begin(ex, step)?; }
        self.defer = tangent; self.defer_step = step;
        for id in ex.allocations.keys() { self.born.entry(*id).or_insert(step - 1); }
        let walked = (|| -> Result<(), String> { for name in &blocks { self.walk(ex, plan, name, step, base, frame, 0, &mut flat, tangent)?; } Ok(()) })();
        for id in ex.allocations.keys() { self.born.entry(*id).or_insert(step); }
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
            let want = self.blob_index.get(nb);
            for b in &want { self.blob_ahead.insert(crate::blob_sha(b).to_owned()); }
            let cached = ex.blob_cache.as_ref();
            let want: Vec<crate::plan::Blob> = want.into_iter()
                .filter(|b| !cached.is_some_and(|c| c.contains_key(crate::blob_sha(b)))).collect();
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
        // step-hold: an intermediate boundary is held as its spans (store_hold); boundary 0 and the end stay whole
        let r = match self.shold.clone() {
            Some(h) if s >= 1 && s < self.hold_steps => self.store_hold(ex, frame, dir, s, &h),
            _ => self.store_inner(ex, frame, dir, s),
        };
        self.store_s += t_store.elapsed().as_secs_f64();
        r
    }
    /// step-hold: the state at intermediate boundary `s` held as its spans (HoldSet::spans): only those bytes come
    /// down from the card, into one page-locked buffer (made ahead on a helper thread when one is ready, else from the
    /// checkpoint pool), and restore puts only those back. THE WASTE THIS REMOVES: the 59-step 3 km storm gradient kept
    /// 8 checkpoints of the whole 8.4 GB state and re-ran 55 steps to reach the others: outside its rows 48.8 s of re-run
    /// steps, 10.2 s of checkpoint restores, 8.2 s of stores, 6.1 s of re-run state copies (fin-first window_sweep_parts),
    /// plus 9.8 s of whole-state restores in the rows. Held every step at 2.68 GB (set C2), 58 states take 155 GB of host
    /// memory, the same order as today's 8 x 8.4 GB plus the re-run buffers. The same words reach the same addresses for
    /// every held span; the spans left out are constants (checked against boundary 0 after the sweep) or words each step
    /// writes before it reads them, and the redo check compares every held word as before.
    unsafe fn store_hold(&mut self, ex: &mut Execution, frame: &Option<ReplayFrame>, dir: &Path, s: usize, h: &crate::step_hold::HoldSet) -> Result<Boundary, String> {
        let _c = capi::xfer::Cat::checkpoint();
        let t = Instant::now();
        dev::sync(&ex.cuda)?;
        ex.captured = None;
        let mut live: Vec<(u64, usize)> = ex.allocations.iter().map(|(id, a)| (*id, a.bytes)).collect();
        live.sort_unstable();
        let spans = h.spans(&live)?;
        let (index, sub) = crate::step_hold::layout(&spans);
        let total: usize = index.iter().map(|x| x.2).sum();
        // the budget counts the buffer as made: rounded up to the pool's granule (drop_boundary gives back the same)
        let cap = capi::pin_cap(total);
        if self.bs_ram_now + cap > self.bs_ram_budget {
            return Err(format!("--step-hold: boundary {s} needs {cap} more page-locked bytes and {} of the {} byte --bstore-ram-gib budget are held already: raise --bstore-ram-gib", self.bs_ram_now, self.bs_ram_budget));
        }
        let mut hb = match self.hold_take_ahead(total) {
            Some(m) => capi::HostBytes::from_mem_pooled(m, total, bs_pool()),
            None => { self.hold_n.3 += 1; capi::HostBytes::pooled(&ex.cuda, bs_pool(), total)? },
        };
        let dl = entry_copies(&index, Some(&sub), &index, |id| ex.allocations.get(&id).map(|a| (a.address, a.bytes)), true)?;
        crate::pinned::download_spans(&ex.cuda, hb.as_mut_slice(), &dl)?;
        // AD_PLANT_HOLD (test only): one word of this held state scaled; the redo check of step s must refuse it
        if let Some((ps, f)) = *crate::step_hold::plant() { if ps == s {
            let (w, b0, b1) = crate::step_hold::plant_word(hb.as_mut_slice(), f).ok_or("AD_PLANT_HOLD: no finite word to plant in")?;
            let al = index.iter().zip(sub.iter()).find(|(&(_, off, n), _)| 4 * w >= off && 4 * w < off + n)
                .map(|(&(id, off, _), &u)| (id, u + 4 * w - off)).unwrap_or((0, 0));
            let rec = json!({"switch":"AD_PLANT_HOLD","boundary":s,"factor":f,"allocation":al.0,"byte_in_allocation":al.1,"bits_before":format!("{b0:08x}"),"bits_after":format!("{b1:08x}"),
                "side":"the held state of one intermediate boundary with --step-hold: one float32 word scaled; the recorded forward, boundary 0, the window end and the tangent are untouched","planted":true,"valid":false});
            eprintln!("{}", json!({"event":"window_plant_hold","planted_error":rec}));
            plant_note(rec)?;
        } }
        self.bs_ram_now += cap; self.bs_ram_peak = self.bs_ram_peak.max(self.bs_ram_now); self.bs_ram_count += 1;
        self.hold_refill(ex, total, s)?;
        self.hold_n.0 += 1; self.hold_n.1 += total as u64; self.hold_n.2 += t.elapsed().as_secs_f64();
        Ok(Boundary { sub: Some(std::sync::Arc::new(sub)), path: dir.join(format!("b{s}.bin")), index, host: host_snap(ex, frame), step: s, delta: false,
                      file_bytes: 0, ram: Some(std::sync::Arc::new(hb)), dev: 0, cs: None })
    }
    /// step-hold: a page-locked buffer of at least `total` bytes made ahead (hold_refill), when one has been asked.
    /// One too small (the live state grew) is freed and the next is tried.
    fn hold_take_ahead(&mut self, total: usize) -> Option<crate::pinned::PinMem> {
        while let Some((cap, h)) = self.hold_ahead.pop_front() {
            if let Ok(Ok(m)) = h.join() { if cap >= total { return Some(m); } }
        }
        None
    }
    /// step-hold: keep AD_STEP_HOLD_AHEAD (default 2; 0 = none) buffers of the last held size (rounded up to the
    /// pool's granule) being made on helper threads, inside the host-memory budget. THE WASTE THIS REMOVES: a new 2.68 GB
    /// page-locked buffer on the step path at every store of the forward (first touch of every page and its driver
    /// registrations, about 0.15 s each at the rate host-leftovers-2 read on a test machine). Only where the bytes live changes.
    /// Called after boundary `s` is counted in bs_ram_now; no buffer is made for stores past the last held boundary.
    unsafe fn hold_refill(&mut self, ex: &Execution, total: usize, s: usize) -> Result<(), String> {
        let want = std::env::var("AD_STEP_HOLD_AHEAD").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(2)
            .min(self.hold_steps.saturating_sub(s + 1));
        let cap = capi::pin_cap(total);
        while self.hold_ahead.len() < want && self.bs_ram_now + cap * (self.hold_ahead.len() + 1) <= self.bs_ram_budget {
            let (fns, set, c) = (crate::pinned::PinFns::of(&ex.cuda), ex.cuda.ctx_set_current, ex.context as usize);
            let h = std::thread::Builder::new().name("step-hold-ahead".into()).spawn(move || {
                let r = set(c as *mut std::ffi::c_void); if r != 0 { return Err(format!("cuCtxSetCurrent step-hold ahead: {r}")); }
                crate::pinned::PinMem::new(fns, cap)
            }).map_err(|e| e.to_string())?;
            self.hold_ahead.push_back((cap, h));
        }
        Ok(())
    }
    /// step-hold: after a sweep, every allocation the hold set names constant must still hold boundary 0's
    /// bytes (no restore put them back). THE BREAKAGE THIS PREVENTS: a set made from another recording, or a step that
    /// writes an allocation the recording showed unchanged, would leave a later re-run reading a wrong word that no redo
    /// check of a held span could name. Returns (allocations checked, bytes); refuses on the first that differs.
    unsafe fn hold_const_check(&mut self, ex: &Execution, b0: &Boundary) -> Result<(usize, usize), String> {
        let Some(h) = self.shold.clone() else { return Ok((0, 0)) };
        dev::sync(&ex.cuda)?;
        let d0 = self.load(b0)?;
        let (mut n_al, mut n_b) = (0usize, 0usize);
        let mut buf: Vec<u8> = Vec::new();
        // every constant allocation that moved is named in one refusal, so one run tells what a set must hold
        let mut bad: Vec<String> = Vec::new();
        if b0.sub.is_some() { return Err("--step-hold: boundary 0 must be whole".into()); }
        for (id, off, n) in &b0.index {
            if !h.consts.contains(id) { continue; }
            let a =ex.allocations.get(id).ok_or_else(|| format!("--step-hold: constant allocation {id} is not live after the sweep"))?;
            if a.bytes != *n { return Err(format!("--step-hold: constant allocation {id} changed size")); }
            buf.resize(*n, 0);
            if *n != 0 { checked("cuMemcpyDtoH step-hold constant", (ex.cuda.download)(buf.as_mut_ptr().cast(), a.address, *n))?; }
            let d = differing_words(&buf, &d0[*off..*off + *n]);
            if d != 0 { bad.push(format!("{id} ({d} words)")); continue; }
            n_al += 1; n_b += n;
        }
        if !bad.is_empty() { return Err(format!("--step-hold: {} allocations named constant by the hold set {} differ from boundary 0 after the sweep: {}: the set does not fit this window", bad.len(), h.name, bad.join(", "))); }
        Ok((n_al, n_b))
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
        let mut pin = match self.state_pin_take(total) { Some(b) => b, None => crate::pinned::PinnedBuf::new(&ex.cuda, total)? };
        crate::pinned::download_spans(&ex.cuda, &mut pin, &index)?;
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
        self.anchor_pool.push(pin);
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
        // so making a 3.4 GB page-locked buffer (~0.15-0.4 s on a test machine) is off the step path. Same bytes.
        let queued = if ctx == 0 { self.pin_q.pop_front() } else { None };
        let from_q = queued.is_some();
        let ahead = match (ctx, queued.or_else(|| self.pin_next.take())) {
            (0, Some((cap, h))) => match h.join() {
                Ok(Ok(m)) if cap >= total => Some(m),
                Ok(Ok(_)) => { eprintln!("{}", json!({"event":"window_pin_ahead_too_small","buffer_bytes":cap,"state_bytes":total})); None },
                Ok(Err(e)) => { eprintln!("{}", json!({"event":"window_pin_ahead_failed","error":e})); None },
                Err(_) => None,
            },
            _ => None,
        };
        if from_q && ahead.is_some() { self.pin_q_made += 1; }
        let mut hb = match ahead { Some(m) => { self.pin_ahead_hits += 1; capi::HostBytes::from_mem(m, total) }, None => capi::HostBytes::pooled(&ex.cuda, ctx, total)? };
        let next = total.max(self.pin_min);
        if ctx == 0 && self.pin_ahead && self.pin_q.is_empty() && !from_q && room >= total + next {
            let (fns, set, c) = (crate::pinned::PinFns::of(&ex.cuda), ex.cuda.ctx_set_current, ex.context as usize);
            self.pin_next = Some((next, std::thread::Builder::new().name("hl2-pin-ahead".into()).spawn(move || {
                let r = set(c as *mut std::ffi::c_void); if r != 0 { return Err(format!("cuCtxSetCurrent pin-ahead: {r}")); }
                crate::pinned::PinMem::new(fns, next)
            }).map_err(|e| e.to_string())?));
        }
        dev::sync(&ex.cuda)?;
        let mut ids: Vec<u64> = ex.allocations.keys().copied().collect();
        ids.sort_unstable();
        let mut index = Vec::with_capacity(ids.len());
        let mut at = 0usize;
        for id in &ids { let a = ex.allocations[id]; index.push((*id, at, a.bytes, a.address)); at += a.bytes; }
        crate::pinned::download_spans(&ex.cuda, hb.as_mut_slice(), &index)?;
        Ok((crate::CapData::Host(std::sync::Arc::new(hb)), index))
    }
    /// x20: start `n` boundary buffers of the current whole-state size together (one helper thread each, the same PinMem
    /// as pin-ahead), bounded by the RAM budget left. Same bytes: only when each buffer is made changes.
    /// lane tl-host-syncs: each buffer is made `bound` bytes (the plan's live-bytes bound: every plan allocation plus every
    /// Allocate of the window's blocks) when that is larger than the state now. THE BREAKAGE THIS FIXES: sized at the pass
    /// start (8.3 GB at pk120) the buffers were smaller than the state at the step-1/2/3 boundaries (9.8 GB, after the steps'
    /// own allocations), so capture_state_host threw every one away (boundary_buffers_made_at_pass_start 0) and made a
    /// 9.8 GB page-locked buffer inline on the step path: views_capture 2.5 / 3.3 / 4.9 s of the 10-17 s TL steps (x20tls-I1).
    unsafe fn pin_prefill(&mut self, ex: &Execution, n: usize, bound: usize) -> Result<(), String> {
        let total: usize = ex.allocations.values().map(|a| a.bytes).sum::<usize>().max(bound);
        self.pin_min = total;
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
    /// speed ckpt60: the index of the live state: every live allocation, ascending id, offsets one after the other.
    fn cs_index(ex: &Execution) -> Vec<(u64, usize, usize)> {
        let mut ids: Vec<u64> = ex.allocations.keys().copied().collect();
        ids.sort_unstable();
        let mut total = 0usize;
        ids.iter().map(|id| { let n = ex.allocations[id].bytes; let r = (*id, total, n); total += n; r }).collect()
    }
    /// speed live: the index of the state stored at boundary `s`: every live allocation (cs_index), less, at an
    /// intermediate boundary with --bstore-live 1, the plan's non-carried allocations and a planted one (AD_PLANT_LEFT_OUT,
    /// resolved here the first time: see plant_left_out_pick).
    fn cs_index_at(&mut self, ex: &Execution, s: usize) -> Result<Vec<(u64, usize, usize)>, String> {
        let full = Self::cs_index(ex);
        if self.cs_cut.is_none() || !bstore_live_cuts(s, self.cs_cut_steps) { return Ok(full); }
        if let (Some(ask), None) = (self.cs_plant_ask, self.cs_plant) {
            let cut = self.cs_cut.as_ref().unwrap();
            let id = match ask { Some(id) => id, None => plant_left_out_pick(&full, &|id| self.base_el.contains_key(&id) && !cut.contains(&id))
                .ok_or("AD_PLANT_LEFT_OUT: no live carried allocation carries a tangent")? };
            if cut.contains(&id) { return Err(format!("AD_PLANT_LEFT_OUT: allocation {id} is not carried; the test leaves out a carried one")); }
            let bytes = ex.allocations.get(&id).ok_or_else(|| format!("AD_PLANT_LEFT_OUT: allocation {id} is not live at boundary {s}"))?.bytes;
            self.cs_plant = Some(id);
            let rec = json!({"switch":"AD_PLANT_LEFT_OUT","allocation":id,"allocation_bytes":bytes,"first_boundary":s,"side":PLANT_LO_SIDE,"planted":true,"valid":false});
            eprintln!("{}", json!({"event":"window_plant_left_out","planted_error":rec}));
            plant_note(rec)?;
        }
        let (cut, plant) = (self.cs_cut.as_ref().unwrap(), self.cs_plant);
        Ok(live_index(&full, &|id| cut.contains(&id) || plant == Some(id)))
    }
    /// speed ckpt60: every window of `lay` downloaded from the device into the small page-locked buffer and handed to
    /// `f(first piece, piece after the last, bytes)`, in order. Returns the seconds spent downloading.
    unsafe fn cs_scan(&mut self, ex: &Execution, lay: &crate::cstore::Layout, mut f: impl FnMut(usize, usize, &[u8])) -> Result<f64, String> {
        let _c = capi::xfer::Cat::checkpoint();
        dev::sync(&ex.cuda)?;
        let mut pin = match self.cs_pin.take() { Some(p) => p, None => crate::pinned::PinnedBuf::new(&ex.cuda, CS_WIN)? };
        let mut dl = 0f64;
        for (a, b, bytes) in lay.windows(CS_WIN) {
            let t = Instant::now();
            let mut at = 0usize;
            for (i, off, n) in lay.runs(a, b) {
                let (id, _, size) = lay.index[i];
                let al = ex.allocations.get(&id).ok_or_else(|| format!("content store: allocation {id} is not live"))?;
                if al.bytes != size { return Err(format!("content store: allocation {id} holds {} bytes, the boundary {size}", al.bytes)); }
                checked("cuMemcpyDtoH content store", (ex.cuda.download)(pin[at..at + n].as_mut_ptr().cast(), al.address + off as u64, n))?;
                at += n;
            }
            dl += t.elapsed().as_secs_f64();
            f(a, b, &pin[..bytes]);
        }
        self.cs_pin = Some(pin);
        self.cs_s[0] += dl;
        Ok(dl)
    }
    /// speed ckpt60: the whole live state as a state of the content store: a window at a time through the small
    /// page-locked buffer; the pieces that differ from the newest state still held are copied out on all cores.
    /// speed live: `s` is the boundary the state is stored for (cs_index_at).
    unsafe fn cs_capture(&mut self, ex: &Execution, s: usize) -> Result<std::sync::Arc<crate::cstore::State>, String> {
        let t0 = Instant::now();
        let index = self.cs_index_at(ex, s)?;
        let left_out = ex.allocations.values().map(|a| a.bytes).sum::<usize>() - index.iter().map(|x| x.2).sum::<usize>();
        let lay = crate::cstore::Layout::new(&index);
        self.cs_refs.retain(|w| w.strong_count() > 0);
        let reference = self.cs_refs.last().and_then(|w| w.upgrade());
        let mut bld = crate::cstore::Builder::new(lay.clone(), reference);
        let dl = self.cs_scan(ex, &lay, |a, b, buf| bld.feed(a, b, buf))?;
        let mut st = bld.finish();
        st.left_out_bytes = left_out;
        if left_out > 0 { self.cs_cut_n[0] += 1; self.cs_cut_n[1] += left_out as u64; }
        let st = std::sync::Arc::new(st);
        self.cs_refs.push(std::sync::Arc::downgrade(&st));
        self.cs_n[0] += 1; self.cs_s[1] += t0.elapsed().as_secs_f64() - dl;
        if std::env::var("AD_DIGEST_PROF").as_deref() == Ok("1") {
            eprintln!("{}", json!({"event":"boundary_store_tier","tier":"content","bytes":lay.total,"new_bytes":st.new_bytes,"shared_bytes":st.shared_bytes,
                "zero_bytes":st.zero_bytes,"left_out_bytes":st.left_out_bytes,"boundary":s,"download_s":dl,"total_s":t0.elapsed().as_secs_f64(),"store_live_bytes":crate::cstore::live_bytes()}));
        }
        Ok(st)
    }
    /// speed ckpt60: the allocations of `index` the redo of `step` compares word by word: those with declared residue
    /// ranges at that step (their residue words are counted apart, so the truth keeps their bytes).
    fn cs_exact(ex: &Execution, index: &[(u64, usize, usize)], step: usize) -> Vec<usize> {
        index.iter().enumerate().filter(|(_, x)| ex.residue.get(&(step, x.0)).is_some_and(|r| !r.0.is_empty())).map(|(i, _)| i).collect()
    }
    /// speed ckpt60: the truth of the state the device holds now, as the end state of `step` (the redo of `step` is
    /// compared with it later): one pass over the device, a window at a time.
    unsafe fn cs_truth_dev(&mut self, ex: &Execution, step: usize) -> Result<(Vec<(u64, usize, usize)>, std::sync::Arc<crate::cstore::Truth>), String> {
        let t0 = Instant::now();
        // speed live: the truth covers what the boundary's state would hold (the recomputed start state's index)
        let index = self.cs_index_at(ex, step)?;
        let lay = crate::cstore::Layout::new(&index);
        let mut tb = crate::cstore::TruthBuilder::new(lay.clone(), &Self::cs_exact(ex, &index, step));
        let dl = self.cs_scan(ex, &lay, |a, b, buf| tb.feed(a, b, buf))?;
        let t = tb.finish();
        self.cs_exact_peak = self.cs_exact_peak.max(t.exact_bytes());
        self.cs_n[3] += 1; self.cs_s[4] += t0.elapsed().as_secs_f64() - dl;
        Ok((index, std::sync::Arc::new(t)))
    }
    /// speed ckpt60: the truth of a stored state as the end state of `step`, read from its pieces (no download).
    fn cs_truth_state(&mut self, ex: &Execution, st: &crate::cstore::State, step: usize) -> std::sync::Arc<crate::cstore::Truth> {
        let t0 = Instant::now();
        let t = st.truth(&Self::cs_exact(ex, &st.lay.index, step));
        self.cs_exact_peak = self.cs_exact_peak.max(t.exact_bytes());
        self.cs_n[4] += 1; self.cs_s[4] += t0.elapsed().as_secs_f64();
        std::sync::Arc::new(t)
    }
    /// speed ckpt60: the redo check of the step whose truth is in cs_truth: the state the device holds now against
    /// that truth, a window at a time. Only the allocations the whole-copy compare covers (redo_index) are compared.
    /// An allocation the truth holds as bytes goes through compare_boundary (the same counts, residue words apart, the
    /// same window_redo_differs line); every other piece is compared by its SHA-256, and a piece that differs counts as
    /// one differing word (its own window_redo_differs line names the allocation and the piece).
    unsafe fn cs_redo(&mut self, ex: &Execution, b: &Boundary) -> Result<(u64, u64), String> {
        let t0 = Instant::now();
        let (step, truth) = self.cs_truth.clone().ok_or("content store: no truth for this redo")?;
        if truth.lay.index != b.index { return Err(format!("step {step}: the truth of the end state is of another boundary")); }
        let ids: std::collections::HashSet<u64> = self.redo_index(b, step).iter().map(|x| x.0).collect();
        // speed live: the plan's non-carried allocations are never compared, also where a whole truth holds them
        // (the window end): a re-made step starts them from zeros, so their end words are the step's, not the forward's.
        // A planted left-out allocation (AD_PLANT_LEFT_OUT) is carried and stays compared.
        let cut = self.cs_cut.as_ref();
        let want: Vec<bool> = b.index.iter().map(|x| ids.contains(&x.0) && !cut.is_some_and(|c| c.contains(&x.0))).collect();
        let mut ck = crate::cstore::Checker::new(&truth, want);
        let dl = self.cs_scan(ex, &truth.lay, |a, z, buf| ck.feed(a, z, buf))?;
        let (bad, live) = (std::mem::take(&mut ck.bad), std::mem::take(&mut ck.live));
        let (mut d, mut r) = (0u64, 0u64);
        let npieces = bad.len() as u64;
        for p in bad {
            let (i, off, n) = truth.lay.piece(p);
            eprintln!("{}", json!({"event":"window_redo_differs","step":step,"allocation":truth.lay.index[i].0,"first_byte":off,"bytes":truth.lay.index[i].2,
                                   "piece_bytes":n,"by":"SHA-256 of the piece; the differing words inside it are not counted"}));
            d += 1;
        }
        let mut ex_ids: Vec<usize> = live.keys().copied().collect();
        ex_ids.sort_unstable();
        for i in ex_ids {
            let (id, _, n) = truth.lay.index[i];
            let ranges = ex.residue.get(&(step, id)).map(|x| x.0.clone()).unwrap_or_default();
            let c = compare_boundary(&[(id, 0, n)], &live[&i], truth.exact(i).ok_or("content store: an exact allocation without its bytes")?, std::slice::from_ref(&ranges), step);
            d += c.0; r += c.1;
        }
        // the refusal that follows says "differs ... in N words": with the content store N is this sum, said here in full
        if d != 0 { eprintln!("{}", json!({"event":"window_redo_differs_total","step":step,"pieces_whose_sha256_differs":npieces,"differing_words_in_word_compared_allocations":d - npieces,
                                           "counted_as":"each differing piece counts one in the refusal's number; the words inside it are not counted"})); }
        self.cs_n[5] += 1; self.cs_s[5] += t0.elapsed().as_secs_f64() - dl;
        Ok((d, r))
    }
    /// speed ckpt60: the streamed tape's helper keeps two chunk buffers made ahead of use (ST_AHEAD). Once a step has
    /// been reversed its chunk buffers are back in st_pool and st_take reads the pool first, so the two waiting ones are
    /// never taken: 2 x 15 GB page-locked for the rest of the sweep (box fast-1, leg s8-after: 8 chunk buffers resident
    /// where a step uses 7 at most at a time). In the first step reversed they are 2 more than the step's own 7, and
    /// that step was the leg's memory peak (206 GiB, then 179: leg s8b-after), so they are let go before the sweep
    /// too: the first step then makes its chunk buffers itself, once (st_take), and every later step reuses them. Each waiting buffer is swapped here for a token of one page: the helper
    /// stays asleep (it still counts two waiting), and a step that needs more buffers than the pool holds makes one
    /// itself (st_take does that when none of the waiting ones fits). The tape's own lane may want a proper stop for
    /// the helper instead; this touches nothing but the buffers nobody would take.
    unsafe fn cs_tape_spares(&mut self, cuda: &Cuda) -> Result<(), String> {
        // speed two60 (ckpt60 and tape60 in one build): the tape's own trim (st_trim, st_release; AD_TAPE_STREAM_TRIM)
        // frees the buffers nobody took and puts the helper to sleep, so the tokens are only for a run that turns it off.
        if st_trim_on() { return Ok(()); }
        let Some(pool) = self.st_ahead.clone() else { return Ok(()) };
        let mut g = pool.0.lock().map_err(|_| "streamed tape buffer pool poisoned".to_string())?;
        for b in g.0.iter_mut() { if b.len() > 4096 { *b = crate::pinned::PinnedBuf::new(cuda, 4096)?; self.cs_spares += 1; } }
        Ok(())
    }
    /// speed ckpt60: the window end after the tangent pass: its truth is read from its state (the last step's redo is
    /// compared with it), the state is written to the boundary's file on the writer thread, straight from its pieces, and
    /// the boundary becomes that whole file. THE WASTE THIS REMOVES: the dots hold the window end as whole bytes anyway;
    /// its pieces in the store were a second copy through the dots, and the whole copy stayed through the sweep.
    unsafe fn cs_end(&mut self, ex: &Execution, bn: &mut Boundary, steps: usize) -> Result<(), String> {
        let Some(CsB::State(st)) = bn.cs.clone() else { return Ok(()) };
        if self.cs_end_truth.is_none() { self.cs_end_truth = Some(self.cs_truth_state(ex, &st, steps)); }
        self.join_writer()?;
        let (p, n) = (bn.path.clone(), st.lay.total);
        self.bs_now += n as u64; self.bs_peak = self.bs_peak.max(self.bs_now); self.bs_written += n as u64;
        self.bs_files += 1; self.bs_files_peak = self.bs_files_peak.max(self.bs_files);
        self.writer = Some(std::thread::spawn(move || st.write(&p).map_err(|e| format!("{}: {e}", p.display()))));
        bn.cs = None; bn.delta = false; bn.file_bytes = n;
        Ok(())
    }
    unsafe fn store_inner(&mut self, ex: &mut Execution, frame: &Option<ReplayFrame>, dir: &Path, s: usize) -> Result<Boundary, String> {
        let _c = capi::xfer::Cat::checkpoint();
        dev::sync(&ex.cuda)?;
        // speed ckpt60 (--bstore-content): the boundary is a state of the content store; no file, no whole copy kept.
        // (A capture the step made for its own check is not used: the store reads the device a window at a time.)
        if self.cs_on {
            ex.captured = None;
            let cs = self.cs_capture(ex, s)?;
            return Ok(Boundary { sub: None, path: dir.join(format!("b{s}.bin")), index: cs.lay.index.clone(), host: host_snap(ex, frame), step: s, delta: false, file_bytes: 0,
                                 ram: None, dev: 0, cs: Some(CsB::State(cs)) });
        }
        // The capture is the state at this boundary only if nothing ran since it was taken (store follows the step's
        // finish directly); the live addresses are checked against it, and anything else downloads afresh.
        let cap = ex.captured.take().filter(|(_, index)| index.len() == ex.allocations.len()
            && index.iter().all(|(id, _, n, addr)| ex.allocations.get(id).is_some_and(|a| a.bytes == *n && a.address == *addr)));
        let (data, index): (std::sync::Arc<crate::CapData>, Vec<(u64, usize, usize)>) = match cap {
            Some((d, ix)) => (d, ix.into_iter().map(|(id, off, n, _)| (id, off, n)).collect()),
            None => {
                // F1: boundary zero has no preceding forward capture. Use the
                // same budgeted RAM tier as later checkpoints, avoiding a disk
                // write followed immediately by load(). The ordinary Arc owners
                // retain these exact bytes; delta/disk tiers keep their policy.
                let (d, ix) = if !self.delta_store && self.bs_ram_budget > self.bs_ram_now {
                    self.capture_state_host(ex, bs_pool(), self.bs_ram_budget - self.bs_ram_now)?
                } else { let (d, ix) = self.capture_state(ex)?; (crate::CapData::Vec(d), ix) };
                (std::sync::Arc::new(d), ix.into_iter().map(|(id, off, n, _)| (id, off, n)).collect())
            },
        };
        let path = dir.join(format!("b{s}.bin"));
        // S02: with --checkpoints a boundary after boundary 0 is stored as its changed chunks (lossless, see delta_encode).
        let delta = self.delta_store && self.ckpt_base.is_some();
        // host-leftovers-2: the CLI tangent sweep captured this boundary straight into page-locked RAM inside the RAM
        // budget (capture_host, set by the window loop): keep those bytes as the boundary, as the C-ABI session does,
        // instead of writing 3.4 GB to b{s}.bin on one writer thread (a test machine s4-C18: store 105 s of a 369 s window).
        if !delta { if let crate::CapData::Host(h) = &*data {
            let n = h.as_slice().len();
            self.bs_ram_now += n; self.bs_ram_peak = self.bs_ram_peak.max(self.bs_ram_now); self.bs_ram_count += 1;
            if std::env::var("AD_DIGEST_PROF").as_deref() == Ok("1") {
                eprintln!("{}", json!({"event":"boundary_store_tier","step":s,"tier":"ram","bytes":n,
                    "ram_now_bytes":self.bs_ram_now,"ram_budget_bytes":self.bs_ram_budget}));
            }
            return Ok(Boundary { sub: None, path, index, host: host_snap(ex, frame), step: s, delta: false, file_bytes: 0, ram: Some(h.clone()), dev: 0, cs: None });
        } }
        self.join_writer()?;
        let p = path.clone();
        let data = if delta { let (bi, bd) = self.ckpt_base.as_ref().unwrap(); std::sync::Arc::new(crate::CapData::Vec(delta_encode(&index, &data, bi, bd))) } else { data };
        let file_bytes = data.len();
        self.bs_now += file_bytes as u64; self.bs_peak = self.bs_peak.max(self.bs_now); self.bs_written += file_bytes as u64;
        self.bs_files += 1; self.bs_files_peak = self.bs_files_peak.max(self.bs_files);
        self.writer = Some(std::thread::spawn(move || fs::write(&p, &data[..]).map_err(|e| format!("{}: {e}", p.display()))));
        Ok(Boundary { sub: None, path, index, host: host_snap(ex, frame), step: s, delta, file_bytes, ram: None, dev: 0, cs: None })
    }
    /// g1-mem: the store's capture, the restore staging, the redo download and a recomputed start state each need a
    /// page-locked buffer of the whole live state (26.5 GB at 60 km), one after the other, never two at a time. Each
    /// used to keep its own (capture_pin, restore_pin, cmp_pin, anchor_pool), so four whole states stayed resident
    /// through the reverse sweep. They now take from and return to `anchor_pool`. THE BREAKAGE THIS PREVENTS: a 7-step
    /// 60 km window with --checkpoints sqrt peaked at 345.74 GiB of host memory against a 290 GiB share (a test machine
    /// e29-w7-f0). The same bytes move in the same order; only which buffer they pass through changes. A pool whose
    /// buffers are all too small is emptied before the caller makes a fitting one (the live state grew).
    fn state_pin_take(&mut self, total: usize) -> Option<crate::pinned::PinnedBuf> {
        match self.anchor_pool.iter().position(|b| b.len() >= total) {
            Some(i) => Some(self.anchor_pool.swap_remove(i)),
            None => { self.anchor_pool.clear(); None },
        }
    }
    unsafe fn load(&mut self, b: &Boundary) -> Result<Bytes, String> {
        let t = Instant::now();
        let r = self.load_inner(b);
        self.load_s += t.elapsed().as_secs_f64();
        r
    }
    /// speed ckpt60: what the checkpointed sweep hands to restore and to the redo compare as a boundary's bytes: nothing
    /// with --bstore-content (restore reads the state's pieces, the redo compare its truth), the loaded bytes otherwise.
    unsafe fn load_sw(&mut self, b: &Boundary) -> Result<Bytes, String> {
        if self.cs_on { Ok(Bytes::Own(Vec::new())) } else { self.load(b) }
    }
    unsafe fn load_inner(&mut self, b: &Boundary) -> Result<Bytes, String> {
        // host-leftovers-2: a RAM boundary is its own page-locked bytes, shared, not copied (a 3.4 GB copy per load, and
        // restore then DMAs straight from it); the same bytes.
        if let Some(r) = &b.ram { return Ok(Bytes::Ram(r.clone())); }
        // speed ckpt60: a content-store boundary read as whole bytes (the dots and the checks outside the sweep): the same bytes
        match &b.cs {
            // speed live: a state that left allocations out has no whole bytes to give (only restore reads it)
            Some(CsB::State(st)) if st.left_out_bytes > 0 => return Err(format!("--bstore-live: the state of boundary {} holds only the carried allocations ({} bytes left out); it has no whole bytes to load", b.step, st.left_out_bytes)),
            Some(CsB::State(st)) => return Ok(Bytes::Own(st.whole())),
            Some(CsB::Dev(_)) => return Err("content store: a recomputed start state is only on the device; it has no bytes to load".into()),
            None => {},
        }
        self.join_writer()?;
        if let Some((p, h)) = self.ck_pf.take() {
            let d = h.join().map_err(|_| "checkpoint prefetch panicked")?;
            if p == b.path { self.pf_hits += 1; return d.map(Bytes::Own); }
        }
        let f = if self.cs_on && !b.delta { read_whole(&b.path, b.file_bytes)? } else { fs::read(&b.path).map_err(|e| format!("{}: {e}", b.path.display()))? };
        if !b.delta { return Ok(Bytes::Own(f)); }
        let (bi, bd) = self.ckpt_base.as_ref().ok_or("changed-chunk boundary without boundary 0")?;
        delta_decode(&f, &b.index, bi, bd).map(Bytes::Own)
    }
    /// S02: read (and decode) checkpoint `b` in the background; the next load of the same file takes it.
    unsafe fn prefetch_ck(&mut self, b: &Boundary) -> Result<(), String> {
        if b.ram.is_some() || b.dev != 0 || b.cs.is_some() { return Ok(()); } // runner-lib: a RAM or device boundary has no file to read
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
    /// a12 bs-tape: wait until the last asynchronous restore's upload has ended (its source may then change or go).
    unsafe fn xfer_wait(&mut self, cuda: &Cuda) -> Result<(), String> {
        if self.xfer_pending { checked("cuEventSynchronize restore", (cuda.event_synchronize)(self.xfer_ev[1]))?; self.xfer_pending = false; }
        Ok(())
    }
    /// resident: the end of a request, after its files were removed (bstore_removed): the boundary's share of the
    /// store's counts is given back as drop_boundary gives it, and its bytes go with it.
    /// dist47: a kept-loaded program's request that was refused in its base forward (the forecast stops at its start)
    /// leaves what a finished request leaves: nothing queued on the card, no boundary of its own in the store or on
    /// disk, no forward kept, no pending read or writer. The kept boundary 0 stays; the next request restores it.
    unsafe fn resident_refused(&mut self, ex: &mut Execution, bounds: &mut Vec<Boundary>, ck: &mut BTreeMap<usize, Boundary>) -> Result<(), String> {
        dev::sync(&ex.cuda)?;
        self.capture = false; self.capture_host = None; ex.captured = None; self.pin_ahead = false; self.gather_pass_join = false; self.counting = false;
        ex.digest_defer = false; ex.digest_flush()?;
        self.join_writer()?;
        if let Some((_, h)) = self.ck_pf.take() { let _ = h.join(); }
        for b in bounds.iter().chain(ck.values()) { let _ = fs::remove_file(&b.path); }
        self.xfer_wait(&ex.cuda)?;
        self.kf_saved.clear(); self.kf_keep.clear();
        for b in bounds.drain(..).chain(std::mem::take(ck).into_values()) { self.resident_forget(b); }
        if let Some((_, h)) = self.load_next.take() { let _ = h.join(); }
        self.ckpt_base = None; self.y_path = None; self.held_prev = None; self.cs_dev_at = None; self.lw_follow.clear();
        Ok(())
    }
    fn resident_forget(&mut self, b: Boundary) {
        if let Some(r) = &b.ram { self.bs_ram_now = self.bs_ram_now.saturating_sub(r.as_slice().len()); }
        else if b.dev == 0 && b.cs.is_none() { self.bs_now = self.bs_now.saturating_sub(b.file_bytes as u64); self.bs_files = self.bs_files.saturating_sub(1); }
    }
    unsafe fn drop_boundary(&mut self, b: Boundary) -> Result<(), String> {
        // a12 bs-tape: this boundary's bytes may be the source of the last asynchronous restore
        if self.xfer_pending {
            if let Some(f) = self.xfer_sync { checked("cuEventSynchronize restore", f(self.xfer_ev[1]))?; }
            self.xfer_pending = false;
        }
        // step-hold: a held state was counted as made (pin_cap); every other boundary by its length, as before
        if let Some(r) = &b.ram { let n = r.as_slice().len(); self.bs_ram_now = self.bs_ram_now.saturating_sub(if b.sub.is_some() { capi::pin_cap(n) } else { n }); }
        if b.ram.is_some() || b.dev != 0 || b.cs.is_some() { return Ok(()); } // runner-lib: nothing on disk; the bytes go with the Boundary
        self.join_writer()?;
        if self.ck_pf.as_ref().is_some_and(|(p, _)| *p == b.path) { if let Some((_, h)) = self.ck_pf.take() { let _ = h.join(); } }
        fs::remove_file(&b.path).map_err(|e| format!("{}: {e}", b.path.display()))?;
        self.bs_now -= b.file_bytes as u64; self.bs_files -= 1;
        Ok(())
    }
    /// speed (fold7): the planted change of AD_PLANT_RECOMPUTE (test only, see plant_recompute), made once.
    unsafe fn plant_recompute_now(&mut self, ex: &Execution, step: usize, al: Option<u64>, wd: Option<usize>) -> Result<(), String> {
        use std::sync::atomic::Ordering::Relaxed;
        if PLANT_RC_HITS.load(Relaxed) != 0 { return Ok(()); }
        dev::sync(&ex.cuda)?;
        let id = match al { Some(id) => id, None => {
            let mut best: Option<(usize, u64)> = None;
            for id in self.base_el.keys() { if let Some(a) = ex.allocations.get(id) {
                if best.is_none_or(|(b, i)| a.bytes > b || (a.bytes == b && *id < i)) { best = Some((a.bytes, *id)); }
            } }
            best.ok_or("AD_PLANT_RECOMPUTE: no live allocation carries a tangent")?.1
        } };
        let a = *ex.allocations.get(&id).ok_or_else(|| format!("AD_PLANT_RECOMPUTE: allocation {id} is not live before step {step}"))?;
        let nw = a.bytes / 4;
        let (from, span) = match wd { Some(w) => (w, 1), None => (nw / 2, (nw - nw / 2).min(65536)) };
        if from >= nw { return Err(format!("AD_PLANT_RECOMPUTE: allocation {id} has {nw} words, none at {from}")); }
        let mut buf = vec![0u8; span * 4];
        checked("cuMemcpyDtoH plant recompute", (ex.cuda.download)(buf.as_mut_ptr().cast(), a.address + (from * 4) as u64, buf.len()))?;
        let (i, before, after) = match wd {
            Some(_) => { let b = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]); (0, b, b ^ PLANT_RC_BIT) },
            None => plant_recompute_word(&buf).ok_or_else(|| format!("AD_PLANT_RECOMPUTE: allocation {id} has no finite float32 of magnitude 1e-30 or more in words {from}..{}", from + span))?,
        };
        dev::upload(&ex.cuda, a.address + ((from + i) * 4) as u64, &after.to_le_bytes())?;
        dev::sync(&ex.cuda)?;
        PLANT_RC_HITS.store(1, Relaxed);
        let rec = json!({"switch":"AD_PLANT_RECOMPUTE","step":step,"allocation":id,"allocation_bytes":a.bytes,"word":from + i,
            "bits_before":format!("{before:08x}"),"bits_after":format!("{after:08x}"),"value_before":f32::from_bits(before),"value_after":f32::from_bits(after),
            "planted":true,"recompute_steps_before":self.recompute_steps,"side":PLANT_RC_SIDE,"valid":false});
        eprintln!("{}", json!({"event":"window_plant_recompute","planted_error":rec}));
        // window.json says planted from this moment on, also when the run is then refused and never writes its receipt
        if let Ok(g) = PLANT_RC_OUT.lock() { if let Some((p, r0)) = g.as_ref() {
            let mut r: Value = fs::read(p).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_else(|| r0.clone());
            r["planted_error"] = rec;
            fs::write(p, serde_json::to_vec_pretty(&r).unwrap()).map_err(|e| format!("{}: {e}", p.display()))?;
        } }
        Ok(())
    }
    /// S02: one forward step without the tangent (the boundary hashes still checked against the reference), recomputing
    /// a stretch of the trajectory from a checkpoint; its command list must be the one the tangent forward recorded.
    #[allow(clippy::too_many_arguments)]
    unsafe fn advance(&mut self, ex: &mut Execution, plan: &Plan, base: &Path, frame: &mut Option<ReplayFrame>, refs: &References,
                      step: usize, flats: &[Vec<Flat>]) -> Result<(), String> {
        let t = Instant::now();
        self.cs_dev_at = None;
        // speed (fold7): AD_PLANT_RECOMPUTE, test only: one word of this re-run step's start state is changed first
        if let Some((ps, al, wd)) = plant_recompute() { if *ps == step { self.plant_recompute_now(ex, step, *al, *wd)?; } }
        // speed (storm, round 3): inside a checkpointed sweep the boundary hash of this step runs beside the card
        let defer0 = ex.digest_defer;
        if ex.digest_lazy { ex.digest_defer = true; }
        let fl = if self.recompute_unchecked { self.forward_step_c(ex, plan, base, frame, None, step, false) }
                 else { self.forward_step(ex, plan, base, frame, refs, step, false) };
        ex.digest_defer = defer0;
        let fl = fl?;
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
        // speed (storm, round 3): the boundary hashes of this sweep's recomputed steps run beside the card and are
        // all joined (compared, refused if one differs) before the sweep returns its rows (main.rs digest_arenas_max).
        ex.digest_flush()?;
        // central wide-tools: not on the tuning path, where a re-run step is compared with nothing (sweep_digest_lazy)
        ex.digest_lazy = sweep_digest_lazy(crate::recompute_hash_beside(), self.recompute_unchecked);
        // a12 s60-rerun: with AD_RECOMPUTE_UNCHECKED=1 a re-run step is compared with nothing, so its end-state digest (a
        // 26.55 GB download at 60 km, host_digest_prof download_s 1.5 to 4.0 s a re-run step) is read by nothing either:
        // it is skipped for the sweep, as a moved start (--x0-dir) skips it for the whole run (x0_digest_skip). The
        // boundaries the sweep stores carry a history nothing reads (host_snap allows it while the skip is on); after the
        // sweep the skip goes back to what it was, so a checked digest after a skipped one still refuses (history_stale).
        let skip0 = ex.unchecked_digest_skip;
        if self.recompute_unchecked && rerun_unchecked_word() { ex.unchecked_digest_skip = true; }
        let t_sweep = Instant::now();
        let rows = self.sweep_revolve_body(ex, plan, base, frame, refs, flats, b0, bn, ck, bdir, rv, c, anchors, tag);
        ex.unchecked_digest_skip = skip0;
        let t_join = Instant::now();
        self.xfer_wait(&ex.cuda)?;
        let joined = ex.digest_lazy_end();
        // speed storm3: the sweep is over: the idle checkpoint buffers go back to the system before the dots
        let t_idle = Instant::now();
        let (idle_n, idle_bytes) = if PIN_KEEP_RESIDENT.load(std::sync::atomic::Ordering::Relaxed) { (0, 0) } else if free_beside() {
            // drains (AD_FREE_BESIDE=1, see free_beside): counted here, released by a helper thread
            let bufs = capi::pin_pool_take_idle(bs_pool());
            let (n, by) = (bufs.len(), bufs.iter().map(|m| m.cap()).sum::<usize>());
            let h = std::thread::Builder::new().name("free-beside".into()).spawn(move || drop(bufs)).map_err(|e| format!("free-beside thread: {e}"))?;
            if let Ok(mut g) = FREE_BESIDE.lock() { g.push(h); }
            (n, by)
        } else { capi::pin_pool_release_idle(bs_pool()) };
        let idle_s = t_idle.elapsed().as_secs_f64();
        if snap_reuse() {
            // drains (AD_SNAP_REUSE): the kept state goes with the sweep (beside it under AD_FREE_BESIDE)
            let sp = self.snap_spare.take();
            let tf = Instant::now();
            if free_beside() { if let Some(v) = sp { if let Ok(h) = std::thread::Builder::new().name("free-beside".into()).spawn(move || drop(v)) { if let Ok(mut g) = FREE_BESIDE.lock() { g.push(h); } } } } else { drop(sp); }
            eprintln!("{}", json!({"event":"window_snap_reuse","copies_into_kept":SNAP_REUSED.0.swap(0, std::sync::atomic::Ordering::Relaxed),
                "kept":SNAP_REUSED.1.swap(0, std::sync::atomic::Ordering::Relaxed),"free_at_end_s":tf.elapsed().as_secs_f64()}));
        }
        let parts: Vec<f64> = SWEEP_NS.iter().map(|n| n.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e9).collect();
        eprintln!("{}", json!({"event":"window_sweep_parts","checkpoint_restore_s":parts[0],"rerun_steps_s":parts[1],
            "rerun_state_download_s":parts[2],"rerun_state_copy_s":parts[3],"checkpoint_store_s":parts[4],"checkpoint_drop_s":parts[5],
            "checkpoint_buffer_pool":bs_pool() != 0,"held_restores_skipped":self.held_skips.0,"held_bytes_not_uploaded":self.held_skips.1,"restores_async":self.restore_async_n,"idle_buffers_freed_at_end":idle_n,"idle_bytes_freed_at_end":idle_bytes,"idle_free_s":idle_s,
            "digests_joined_at_end_s":t_join.elapsed().as_secs_f64() - idle_s,
            "sweep_wall_s":t_sweep.elapsed().as_secs_f64()}));
        // step-hold: the constants the held states left out still hold boundary 0's bytes (hold_const_check)
        let rows = match rows {
            Ok(r) if self.shold.is_some() => {
                let t_c = Instant::now();
                let c = self.hold_const_check(ex, b0);
                let h = self.shold.clone().unwrap();
                eprintln!("{}", json!({"event":"window_step_hold","set":h.name,"set_sha256":h.sha256,"stores":self.hold_n.0,"held_bytes":self.hold_n.1,
                    "store_s":self.hold_n.2,"buffers_made_on_step_path":self.hold_n.3,"rerun_steps_total":self.recompute_steps,
                    "const_check":match &c { Ok((a, b)) => json!({"allocations":a,"bytes":b,"equal":true}), Err(e) => json!({"error":e}) },"const_check_s":t_c.elapsed().as_secs_f64()}));
                c.map(|_| r)
            },
            x => x,
        };
        sweep_joined(rows, joined)
    }
    #[allow(clippy::too_many_arguments)]
    unsafe fn sweep_revolve_body(&mut self, ex: &mut Execution, plan: &Plan, base: &Path, frame: &mut Option<ReplayFrame>, refs: &References,
                            flats: &[Vec<Flat>], b0: &Boundary, bn: &Boundary, ck: &mut BTreeMap<usize, Boundary>, bdir: &Path,
                            rv: &Revolve, c: usize, anchors: usize, tag: Value) -> Result<Vec<Value>, String> {
        let steps = flats.len();
        // step-hold: every intermediate boundary is held (revolve's cost-optimal descent with steps - 1 slots stores
        // only the last one, its ties going to the largest split, and re-makes the rest)
        let want: Vec<usize> = if self.shold.is_some() { (1..steps).collect() } else { rv.descent(steps, c) };
        let missing: Vec<usize> = want.iter().copied().filter(|p| !ck.contains_key(p)).collect();
        if let Some(&last) = missing.last() {
            let d = self.load_sw(b0)?; self.restore(ex, frame, b0, &d)?; drop(d);
            for s in 1..=last {
                self.advance(ex, plan, base, frame, refs, s, flats)?;
                if missing.contains(&s) { let b = self.store(ex, frame, bdir, s)?; ck.insert(s, b); }
            }
        }
        // speed ckpt60: the first redo (the window's last step) is compared with the truth of the window end
        if self.cs_on {
            self.cs_tape_spares(&ex.cuda)?;
            let have = self.cs_end_truth.clone();
            let t = match (have, &bn.cs) {
                (Some(t), _) => t,
                (None, Some(CsB::State(s))) => { let s = s.clone(); let t = self.cs_truth_state(ex, &s, steps); self.cs_end_truth = Some(t.clone()); t },
                _ => return Err("content store: the window end has neither a truth nor a state".into()),
            };
            self.cs_truth = Some((steps, t));
        }
        let cur = self.load_sw(bn)?;
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
                    let t_part = Instant::now();
                    let b = if lo == 0 { b0 } else { ck.get(&lo).ok_or("checkpoint missing")? };
                    let d = self.load_sw(b)?; self.restore(ex, frame, b, &d)?;
                    sweep_add(0, t_part);
                }
                let t_part = Instant::now();
                for t in lo + 1..s { self.advance(ex, plan, base, frame, refs, t, flats)?; }
                sweep_add(1, t_part);
                // speed ckpt60: the recomputed start state is not downloaded, copied and uploaded again: the device holds
                // it (cs_dev_at, see restore), and rev_one reads its truth for the next step's redo from the device. That
                // needs the step to be reversed with one restore (the streamed tape); otherwise the state goes into the
                // content store, from where every segment can restore it.
                if self.cs_on {
                    let cs = if self.st_on && self.pack && !flats[s - 1].is_empty() && self.census.is_none() { self.cs_serial += 1; CsB::Dev(self.cs_serial) }
                             else { CsB::State(self.cs_capture(ex, s - 1)?) };
                    let index = match &cs { CsB::State(st) => st.lay.index.clone(), CsB::Dev(_) => self.cs_index_at(ex, s - 1)? };
                    self.cs_dev_at = Some(cs.key());
                    let b = Boundary { sub: None, path: PathBuf::new(), index, host: host_snap(ex, frame), step: s - 1, delta: false, file_bytes: 0, ram: None, dev: 0, cs: Some(cs) };
                    self.rev_one(ex, base, frame, flats, b0, ck, anchors, lo, Some((b, Bytes::Own(Vec::new()))), s, st)?;
                    continue;
                }
                let t_part = Instant::now();
                let (mut b, d) = self.snap_ram(ex, frame)?;
                sweep_add(2, t_part);
                b.step = s - 1;
                // integ8 (S02 + S04): snap_ram hands out a reused page-locked buffer; this start state becomes the sweep's
                // owned current state (RevSt), so it is copied out once and the buffer goes back to the pool.
                // speed sweep3: the copy runs on all cores (par_copy) instead of one (to_vec). THE WASTE THIS REMOVES: a
                // 10.53 GB state copied by one core took 5.0 and 5.1 s, twice in a 12-step 3 km window (marks sw_copied_11
                // and sw_copied_5 of <local path> test machine/runs/m12b/M12/stderr), with the card idle. The same bytes.
                // speed storm3: keeping the page-locked buffer itself as the sweep's state instead of this copy was
                // tried and taken out: the 50 copies of the 59-step 3 km storm window cost 3.5 s in all, and the leg
                // without them was no faster and held 10 GiB more (runs/fast-4/o59-c-p2 against o59-c-p2copy).
                let t_part = Instant::now();
                // drains (AD_SNAP_KEEP=1, see snap_keep): the page-locked buffer itself is the state, no copy
                let dv = if snap_keep() { Bytes::Pin(d) } else {
                    // drains (AD_SNAP_REUSE=1, see snap_reuse): into the buffer the step before let go
                    let mut dv = match self.snap_spare.take() {
                        Some(mut v) => { v.resize(d.len(), 0); SNAP_REUSED.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed); v },
                        None => vec![0u8; d.len()],
                    };
                    par_copy(&mut dv[..], &d[..]);
                    self.anchor_pool.push(d);
                    Bytes::Own(dv)
                };
                sweep_add(3, t_part);
                // bs-tape: the card holds this start state right now (snap_ram just downloaded it, nothing ran since), so
                // the reverse step's restore of it uploads nothing (see restore).
                if held_skip_on() { self.held_prev = Some((s - 1, dv.as_ptr() as usize)); }
                let r = self.rev_one(ex, base, frame, flats, b0, ck, anchors, lo, Some((b, dv)), s, st);
                self.held_prev = None;
                r?;
            }
            return Ok(());
        }
        let mid = lo + rv.split(hi - lo, k);
        if !ck.contains_key(&mid) {
            // drains (AD_SNAP_REUSE): no kept state while a checkpoint is stored (host memory as before there)
            self.snap_spare = None;
            {
                let t_part = Instant::now();
                let b = if lo == 0 { b0 } else { ck.get(&lo).ok_or("checkpoint missing")? };
                let d = self.load_sw(b)?; self.restore(ex, frame, b, &d)?;
                sweep_add(0, t_part);
            }
            let t_part = Instant::now();
            for t in lo + 1..=mid { self.advance(ex, plan, base, frame, refs, t, flats)?; }
            sweep_add(1, t_part);
            let t_part = Instant::now();
            let b = self.store(ex, frame, bdir, mid)?;
            // speed ckpt60: the device holds this checkpoint's state until something runs (see restore)
            if let Some(cs) = &b.cs { self.cs_dev_at = Some(cs.key()); }
            ck.insert(mid, b);
            sweep_add(4, t_part);
        }
        self.rev_range(ex, plan, base, frame, refs, flats, b0, ck, bdir, rv, anchors, mid, hi, k - 1, st)?;
        let t_part = Instant::now();
        let b = ck.remove(&mid).ok_or("checkpoint missing")?;
        // keep-forward: a checkpoint the forward stored is moved aside for the next request, its share of the store's
        // RAM count given back as drop_boundary gives it (the sweep's own checkpoints then find the room they had)
        if self.kf_keep.contains(&mid) && !self.kf_saved.contains_key(&mid) && b.ram.is_some() && b.sub.is_none() {
            self.xfer_wait(&ex.cuda)?;
            if let Some(r) = &b.ram { self.bs_ram_now = self.bs_ram_now.saturating_sub(r.as_slice().len()); }
            self.kf_saved.insert(mid, b);
        } else { self.drop_boundary(b)?; }
        sweep_add(5, t_part);
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
                let d = self.load_sw(b)?;
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
        // speed ckpt60: the truth of this step's start state, which the next step reversed (s - 1) re-makes as its end
        // state: read from the checkpoint's pieces, or from the device, which holds a recomputed start state right now.
        // The start boundary then keeps no state: the sweep's current state is its index and its truth.
        let mut pb = pb;
        // 60 km tape 4 (a12 SIXTYFAST; AD_CS_TRUTH_BESIDE=0 turns it off): a stored state's truth (the SHA-256 of its
        // pieces in host memory) is made on a thread beside this step's reverse, which does not read it: it is the
        // truth of the NEXT step's redo check. THE WASTE THIS REMOVES: about 1.3 s a step of host hashing with the card
        // idle before every reverse step whose start is a stored state (60 km, 8-step leg t-c2u: truth_s 10.8 s for 8
        // truths, 6 of them from stored states). Same pieces, same hashes, same truth; a recomputed start state on the
        // device is read before the reverse as before (the reverse overwrites it).
        let mut truth_job: Option<std::thread::JoinHandle<(crate::cstore::Truth, f64)>> = None;
        if self.cs_on && s >= 2 && std::env::var("AD_CS_TRUTH_BESIDE").as_deref() != Ok("0") {
            if let Some(CsB::State(x)) = &pb.cs {
                let (x, exact) = (x.clone(), Self::cs_exact(ex, &x.lay.index, s - 1));
                truth_job = Some(std::thread::Builder::new().name("cs-truth-beside".into())
                    .spawn(move || { let t0 = Instant::now(); let t = x.truth(&exact); (t, t0.elapsed().as_secs_f64()) }).map_err(|e| e.to_string())?);
            }
        }
        let next_truth = if self.cs_on && s >= 2 && truth_job.is_none() {
            Some((s - 1, match &pb.cs {
                Some(CsB::State(x)) => { let x = x.clone(); self.cs_truth_state(ex, &x, s - 1) },
                Some(c @ CsB::Dev(_)) => {
                    if self.cs_dev_at != Some(c.key()) { return Err(format!("step {s}: the recomputed start state is no longer on the device")); }
                    let (index, t) = self.cs_truth_dev(ex, s - 1)?;
                    if index != pb.index { return Err(format!("step {s}: the live allocations changed since the start state was recomputed")); }
                    t
                },
                None => return Err(format!("step {s}: its start boundary has no state in the content store")),
            }))
        } else { None };
        let r = self.reverse_step(ex, base, frame, s, &flats[s - 1], &pb, &pd, &cb, &cd, anchors)?;
        let next_truth = match truth_job {
            Some(h) => {
                let (t, secs) = h.join().map_err(|_| format!("step {s}: the truth thread of the stored start state panicked"))?;
                self.cs_exact_peak = self.cs_exact_peak.max(t.exact_bytes());
                self.cs_n[4] += 1; self.cs_s[4] += secs;
                Some((s - 1, std::sync::Arc::new(t)))
            },
            None => next_truth,
        };
        self.dump_bars_after(&ex.cuda, s, st.tag["sweep"].as_str().unwrap_or(""))?;
        self.cs_dev_at = None;
        if self.cs_on { self.cs_truth = next_truth; pb.cs = None; self.cs_tape_spares(&ex.cuda)?; }
        drop(cb);
        match cd {
            Bytes::Pin(p) => self.anchor_pool.push(p),
            // drains (AD_SNAP_REUSE=1, see snap_reuse): kept for the next re-run state copy (one at most)
            Bytes::Own(v) if snap_reuse() && !snap_keep() && !v.is_empty() => { SNAP_REUSED.1.fetch_add(1, std::sync::atomic::Ordering::Relaxed); self.snap_spare = Some(v); },
            other => drop(other),
        }
        eprintln!("{}", json!({"event":"window_reverse_step","seed":st.tag["seed"],"repeat":st.tag["repeat"],"sweep":st.tag["sweep"],"row":r,
                               "recompute_steps":self.recompute_steps,"bstore_bytes_now":self.bs_now}));
        self.word_probe(&ex.cuda, "reverse", "step", s, None, &st.tag)?;
        self.rev_digest(&ex.cuda, "step", s, None, &st.tag)?;
        st.rows.push(r);
        st.cur = Some((pb, pd));
        Ok(())
    }
    unsafe fn restore(&mut self, ex: &mut Execution, frame: &mut Option<ReplayFrame>, b: &Boundary, data: &[u8]) -> Result<(), String> {
        let _c = capi::xfer::Cat::checkpoint();
        let t = dev::prof::start();
        let t1 = Instant::now();
        drain_pinned(ex)?;
        // speed ckpt60: a content-store boundary asked for without bytes. Its pieces go through the small page-locked
        // buffer a window at a time (put together on all cores, sent from there): the same bytes reach the same addresses.
        // When the sweep has just captured or recomputed this very state on the device (cs_dev_at) and nothing ran since,
        // the device already holds every byte the upload would write, so nothing is uploaded. THE WASTE THAT REMOVES: a
        // recomputed start state was downloaded, copied and uploaded again to the addresses it came from (26.5 GB each
        // way at 60 km, every step).
        let dev_at = self.cs_dev_at.take();
        if self.cs_on && data.is_empty() {
            match &b.cs {
                Some(c) if dev_at == Some(c.key()) => { self.cs_n[2] += 1; },
                Some(CsB::State(st)) => {
                    let st = st.clone();
                    let mut pin = match self.cs_pin.take() { Some(p) => p, None => crate::pinned::PinnedBuf::new(&ex.cuda, CS_WIN)? };
                    for (a, z, bytes) in st.lay.windows(CS_WIN) {
                        let tf = Instant::now();
                        st.fill(a, z, &mut pin[..bytes]);
                        let tu = Instant::now();
                        let mut at = 0usize;
                        for (i, off, n) in st.lay.runs(a, z) {
                            let (id, _, size) = st.lay.index[i];
                            let al = ex.allocations.get(&id).ok_or("boundary allocation not live")?;
                            if al.bytes != size { return Err(format!("allocation {id} changed size since the boundary")); }
                            checked("cuMemcpyHtoD restore", (ex.cuda.upload)(al.address + off as u64, pin[at..at + n].as_ptr().cast(), n))?;
                            at += n;
                        }
                        dev::sync(&ex.cuda)?;
                        self.cs_s[2] += (tu - tf).as_secs_f64(); self.cs_s[3] += tu.elapsed().as_secs_f64();
                    }
                    self.cs_pin = Some(pin);
                    self.cs_n[1] += 1;
                    // speed live: what the state left out starts from zeros, as a mid-step Allocate does (exec)
                    if st.left_out_bytes > 0 {
                        let tz = Instant::now();
                        let held: std::collections::HashSet<u64> = st.lay.index.iter().map(|x| x.0).collect();
                        let mut z = 0usize;
                        for (id, al) in ex.allocations.iter() {
                            if !held.contains(id) && (self.cs_cut.as_ref().is_some_and(|c| c.contains(id)) || self.cs_plant == Some(*id)) {
                                dev::zero(&ex.cuda, al.address, al.bytes)?; z += al.bytes;
                            }
                        }
                        dev::sync(&ex.cuda)?;
                        self.cs_cut_n[2] += z as u64; self.cs_cut_s += tz.elapsed().as_secs_f64();
                    }
                },
                Some(CsB::Dev(_)) => return Err("content store: the recomputed start state of this step was only on the device, and the device has moved on".into()),
                None => return Err("content store: restore of a boundary that has no state in the store".into()),
            }
            let r = restore_host(ex, frame, &b.host);
            dev::prof::add(6, t);
            self.restore_s += t1.elapsed().as_secs_f64();
            return r;
        }
        // adjoint-speed-2 (pkg18m2 reverse: 3.4 GB pageable upload per step): the boundary bytes are copied into one
        // reused page-locked buffer on all cores, then DMA'd from there. The same bytes reach the same addresses.
        let total = b.index.iter().map(|(_, off, n)| off + n).max().unwrap_or(0);
        // bs-tape (a12 backstep): a recomputed start state that snap_ram has just downloaded from the card, with nothing run
        // since (rev_range sets held_prev for exactly these bytes, and it is taken by this first restore), is still on the
        // card word for word: the upload would write every byte back where it already is. THE WASTE THIS REMOVES: 10.5 GB
        // uploaded at the start of every reverse step whose start state was recomputed (0.38 s a step at 3 km). The host
        // side is restored as before; every allocation is checked for its size as before; the step's redo compare still
        // checks the replayed end state word for word. AD_HELD_SKIP=0 uploads as before.
        let held = self.held_prev.take();
        if held.is_some_and(|(st, p)| st == b.step && p == data.as_ptr() as usize && data.len() >= total) {
            for (i, (id, _, n)) in b.index.iter().enumerate() {
                let a = ex.allocations.get(id).ok_or("boundary allocation not live")?;
                if !entry_fits(b, i, a.bytes, *n) { return Err(format!("allocation {id} changed size since the boundary")); }
            }
            self.held_skips.0 += 1; self.held_skips.1 += total;
            let r = restore_host(ex, frame, &b.host);
            dev::prof::add(6, t);
            self.restore_s += t1.elapsed().as_secs_f64();
            return r;
        }
        // host-leftovers-2: bytes that are already page-locked (a RAM boundary, a RAM anchor, a C-ABI boundary) are DMA'd
        // from where they are; only other bytes are staged through the reused buffer (pk120: restore 8-16 s per reverse
        // step over 26-48 segments, most of it this copy). The same bytes reach the same addresses.
        let direct = crate::pinned::is_page_locked(&data[..total]);
        let mut pin = if direct { None } else {
            let mut p = match self.state_pin_take(total) { Some(p) => p, None => crate::pinned::PinnedBuf::new(&ex.cuda, total.max(1))? };
            par_copy(&mut p[..total], &data[..total]);
            Some(p)
        };
        let src: &[u8] = match &pin { Some(p) => &p[..total], None => &data[..total] };
        // a12 bs-tape (backstep): bytes that are already page-locked (a RAM checkpoint) go up on a copy stream of their own,
        // and the legacy stream waits for them on the device: the next commands (a recompute, or the re-run of the step to
        // reverse) are queued by the host while the 10.5 GB upload runs, and none of them runs on the card before it ends.
        // The source must not change or go before the upload ends: a RAM checkpoint stays in the sweep's map until
        // drop_boundary, which waits for it first (xfer_wait), as does the next restore. THE WASTE THIS REMOVES: the host
        // stood in a blocking upload, then drained the card, at every restore (about 0.2 s, 100 restores in a 59-step
        // sweep). The same bytes reach the same addresses before any command reads them. AD_RESTORE_ASYNC=0 as before.
        self.xfer_wait(&ex.cuda)?;
        let async_up = direct && ordered(ex) && std::env::var("AD_RESTORE_ASYNC").as_deref() == Ok("1"); // measured no faster (bt-t11g26): off unless 1
        if async_up {
            let cuda = &ex.cuda;
            if self.xfer_stream.is_null() { checked("cuStreamCreate restore", (cuda.stream_create)(&mut self.xfer_stream, 1))?; }
            for e in self.xfer_ev.iter_mut() { if e.is_null() { checked("cuEventCreate restore", (cuda.event_create)(e, 2))?; } }
            checked("cuEventRecord restore start", (cuda.event_record)(self.xfer_ev[0], std::ptr::null_mut()))?;
            checked("cuStreamWaitEvent restore", (cuda.stream_wait_event)(self.xfer_stream, self.xfer_ev[0], 0))?;
            for (_, off, n, at) in ex_copies(ex, b, &b.index, true)? {
                if n != 0 { checked("cuMemcpyHtoDAsync restore", (cuda.upload_async)(at, src[off..off + n].as_ptr().cast(), n, self.xfer_stream))?; }
            }
            checked("cuEventRecord restore done", (cuda.event_record)(self.xfer_ev[1], self.xfer_stream))?;
            checked("cuStreamWaitEvent legacy on restore", (cuda.stream_wait_event)(std::ptr::null_mut(), self.xfer_ev[1], 0))?;
            self.xfer_pending = true;
            self.xfer_sync = Some(cuda.event_synchronize);
            self.restore_async_n += 1;
        } else {
        upload_copies(&ex.cuda, src, &ex_copies(ex, b, &b.index, true)?)?;
        dev::sync(&ex.cuda)?;
        }
        if direct { self.restore_direct += 1; }
        if let Some(p) = pin.take() { self.anchor_pool.push(p); }
        let r = restore_host(ex, frame, &b.host);
        dev::prof::add(6, t);
        self.restore_s += t1.elapsed().as_secs_f64();
        r
    }
    /// longwin: the part of a stored boundary the redo of `step` is compared with: every allocation that existed
    /// at the end of `step`. Allocations are never freed, so a boundary taken during the reverse (a recomputed start
    /// state, a checkpoint stored by a later sweep) also holds the allocations LATER steps make, with whatever the last
    /// pass over those steps left in them; the redo of `step` never touches them, and every pass zero-fills an
    /// allocation where its step makes it (exec), so no pass reads those words. THE BREAKAGE THIS FIXES: the 30-step
    /// 120 km window with --checkpoints sqrt refused its last reverse step ("step 1: restore + redo differs from the
    /// stored boundary in 184529246 words", all in allocation 437, the 2.1 GB pool block step 2 makes): boundary 1 was
    /// a recompute taken while 437 held one pass's words, and the redo of step 1 met another pass's. A boundary the
    /// first forward stored never holds such an allocation, so nothing changes without checkpoints. An allocation
    /// with no recorded birth is compared as before.
    fn redo_index(&self, cur: &Boundary, step: usize) -> Vec<(u64, usize, usize)> {
        cur.index.iter().filter(|(id, _, _)| self.born.get(id).map_or(true, |&b| b <= step)).copied().collect()
    }
    /// adjoint-speed-2: the live state in the stored boundary's layout, DMA'd into a reused page-locked buffer.
    unsafe fn redo_download(&mut self, ex: &Execution, b: &Boundary) -> Result<crate::pinned::PinnedBuf, String> {
        dev::sync(&ex.cuda)?;
        // speed ckpt60 (--bstore-content): the end state is not downloaded whole. It is compared here, a window at a
        // time, with the truth of the stored end state (cs_redo); the counts wait in cs_redo_res for the step's compare
        // (compare_stored), which refuses the step exactly where it did. The buffer handed back is a token of one page.
        if self.cs_on && self.cs_truth.is_some() {
            self.cs_redo_res = Some(self.cs_redo(ex, b)?);
            return match self.anchor_pool.pop() { Some(p) => Ok(p), None => crate::pinned::PinnedBuf::new(&ex.cuda, 4096) };
        }
        let total = b.index.iter().map(|(_, off, n)| off + n).max().unwrap_or(0);
        // x20 scratch-arena: with every step reversed in one pass the anchors-ahead buffers (made on a helper thread during
        // the tangent pass, each the whole live state) are never snapped into; the redo compare takes one of them instead
        // of page-locking its own 10.5 GB on the first one-pass step (pk120 x20sa-p2-C: redo_download 2.07 s on step 3,
        // 0.21 s on step 2). A missing or too small one is made here as before.
        let mut pin = match self.anchor_pool.iter().position(|b| b.len() >= total) {
            Some(i) => self.anchor_pool.swap_remove(i),
            None => match self.take_ahead(total) { Some(b) => b, None => crate::pinned::PinnedBuf::new(&ex.cuda, total.max(1))? },
        };
        download_copies(&ex.cuda, &mut pin, &ex_copies(ex, b, &b.index, false)?)?;
        Ok(pin)
    }
    /// 60 km tape: everything the redo check beside the walk-back needs from this thread (see CsBeside), and the event that
    /// marks the re-run's end on the legacy stream. Same checks as cs_redo makes before it scans.
    unsafe fn cs_beside_prep(&mut self, ex: &Execution, b: &Boundary) -> Result<CsBeside, String> {
        let (step, truth) = self.cs_truth.clone().ok_or("content store: no truth for this redo")?;
        if truth.lay.index != b.index { return Err(format!("step {step}: the truth of the end state is of another boundary")); }
        let ids: std::collections::HashSet<u64> = self.redo_index(b, step).iter().map(|x| x.0).collect();
        let cut = self.cs_cut.as_ref();
        let want: Vec<bool> = b.index.iter().map(|x| ids.contains(&x.0) && !cut.is_some_and(|c| c.contains(&x.0))).collect();
        let lay = &truth.lay;
        let mut wins = Vec::new();
        for (a, z, bytes) in lay.windows(CS_WIN) {
            let mut v = Vec::new();
            for (i, off, n) in lay.runs(a, z) {
                let (id, _, size) = lay.index[i];
                let al = ex.allocations.get(&id).ok_or_else(|| format!("content store: allocation {id} is not live"))?;
                if al.bytes != size { return Err(format!("content store: allocation {id} holds {} bytes, the boundary {size}", al.bytes)); }
                v.push((al.address + off as u64, n));
            }
            wins.push((a, z, bytes, v));
        }
        let res: Vec<Vec<(usize, usize)>> = lay.index.iter().map(|(id, _, _)| ex.residue.get(&(step, *id)).map(|x| x.0.clone()).unwrap_or_default()).collect();
        let cuda = &ex.cuda;
        let pin = match self.cs_pin.take() { Some(p) if p.len() >= 2 * CS_WIN => p, _ => crate::pinned::PinnedBuf::new(cuda, 2 * CS_WIN)? };
        if self.redo_stream.is_null() { checked("cuStreamCreate redo", (cuda.stream_create)(&mut self.redo_stream, 1))?; }
        let mut evs = [std::ptr::null_mut(); 3];
        for e in evs.iter_mut() { checked("cuEventCreate redo beside", (cuda.event_create)(e, 2))?; }
        checked("cuEventRecord redo beside start", (cuda.event_record)(evs[0], std::ptr::null_mut()))?;
        Ok(CsBeside { step, truth: truth.clone(), want, wins, res, pin, stream: self.redo_stream as usize, evs: evs.map(|e| e as usize), ctx: ex.context as usize,
                      fns: (cuda.ctx_set_current, cuda.stream_wait_event, cuda.download_async, cuda.event_record, cuda.event_synchronize), dl_s: 0.0, all_s: 0.0, err: None })
    }
    /// bs-tape (a12 backstep): redo_download's bytes, queued on the redo stream after everything queued so far on the
    /// legacy stream; returns the buffer and the event that marks the copy complete (nothing may read the buffer before).
    unsafe fn redo_download_async(&mut self, ex: &Execution, b: &Boundary) -> Result<(crate::pinned::PinnedBuf, *mut std::ffi::c_void), String> {
        let cuda = &ex.cuda;
        let total = b.index.iter().map(|(_, off, n)| off + n).max().unwrap_or(0);
        let mut pin = match self.anchor_pool.iter().position(|b| b.len() >= total) {
            Some(i) => self.anchor_pool.swap_remove(i),
            None => match self.take_ahead(total) { Some(b) => b, None => crate::pinned::PinnedBuf::new(cuda, total.max(1))? },
        };
        if self.redo_stream.is_null() { checked("cuStreamCreate redo", (cuda.stream_create)(&mut self.redo_stream, 1))?; }
        for e in self.redo_ev.iter_mut() { if e.is_null() { checked("cuEventCreate redo", (cuda.event_create)(e, 2))?; } }
        checked("cuEventRecord redo start", (cuda.event_record)(self.redo_ev[0], std::ptr::null_mut()))?;
        checked("cuStreamWaitEvent redo", (cuda.stream_wait_event)(self.redo_stream, self.redo_ev[0], 0))?;
        for (_, off, n, at) in ex_copies(ex, b, &b.index, false)? {
            if n != 0 { checked("cuMemcpyDtoHAsync redo", (cuda.download_async)(pin[off..off + n].as_mut_ptr().cast(), at, n, self.redo_stream))?; }
        }
        checked("cuEventRecord redo done", (cuda.event_record)(self.redo_ev[1], self.redo_stream))?;
        Ok((pin, self.redo_ev[1]))
    }
    /// 32-bit words of the live state that differ from a stored boundary (declared TEMPO residue ranges counted apart).
    /// Streamed one allocation at a time; large allocations are compared on all cores.
    unsafe fn differing(&self, ex: &Execution, b: &Boundary, data: &[u8], step: usize) -> Result<(u64, u64), String> {
        let _c = capi::xfer::Cat::checkpoint();
        dev::sync(&ex.cuda)?;
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16);
        let (mut d, mut r) = (0u64, 0u64);
        let mut buf: Vec<u8> = Vec::new();
        let ridx = self.redo_index(b, step);
        let rranges = redo_ranges(ex, b, &ridx, step);
        let rcps = ex_copies(ex, b, &ridx, false)?;
        for ((id, off, n, at), ranges) in rcps.iter().zip(rranges) {
            buf.resize(*n, 0);
            checked("cuMemcpyDtoH compare", (ex.cuda.download)(buf.as_mut_ptr().cast(), *at, *n))?;
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
    /// No-recording self-driven live save: the request the forward model would send for this command (live_save::Request
    /// JSON: every argument already resolved to a device slice, bytes or patched bytes; `writes` = every owner the
    /// command can reach through a device or patched argument, a superset of what it writes). Err(kind) = a command
    /// the live adapter does not take (it stays on the recorded route and is counted).
    /// `exact_writes` (AD_LIVE_SELF_KEEP=1): for a kernel with a derivative row, `writes` holds only the owners of its
    /// out and inout arguments (the row's own roles) and of every patched pointer, as a real producer would declare.
    unsafe fn live_self_request(&self, ex: &Execution, op: &Operation, step: usize, frame: Option<&ReplayFrame>, exact_writes: bool) -> Result<Result<Vec<u8>, String>, String> {
        let sl = |s: &crate::plan::Slice| json!({"allocation": s.allocation, "offset": s.offset});
        let hex = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
        let (operation, writes) = match op {
            Operation::Kernel { stream, module, function, grid, block, shared_bytes, arguments } => {
                let (mut args, mut writes) = (Vec::new(), Vec::<u64>::new());
                let outs: Option<Vec<bool>> = if exact_writes { self.kernels.get(function.as_str()).map(|kd| (0..arguments.len()).map(|k| {
                    let Some(name) = kd.names.get(k) else { return true };
                    kd.spec["args"].as_array().into_iter().flatten().any(|a| a["name"].as_str() == Some(name.as_str()) && matches!(a["role"].as_str(), Some("out") | Some("inout")))
                }).collect()) } else { None };
                for (k, a) in arguments.iter().enumerate() {
                    match ex.argument(a, step, frame)? {
                        submit::Argument::Device(s) => { if outs.as_ref().map_or(true, |o| o[k]) { writes.push(s.allocation); } args.push(json!({"kind":"device","target":{"allocation":s.allocation,"offset":s.offset}})); },
                        submit::Argument::Bytes(b) => args.push(json!({"kind":"bytes","bytes_hex":hex(&b)})),
                        submit::Argument::Patched { bytes, pointers } => {
                            for p in &pointers { writes.push(p.target.allocation); }
                            args.push(json!({"kind":"patched","bytes_hex":hex(&bytes),"pointers":pointers.iter().map(|p| json!({"byte_offset":p.byte_offset,"target":{"allocation":p.target.allocation,"offset":p.target.offset}})).collect::<Vec<_>>()}));
                        },
                        submit::Argument::Frame { .. } => return Ok(Err(live_self_kind(op))),
                    }
                }
                writes.sort_unstable(); writes.dedup();
                (json!({"kind":"kernel","stream":stream,"module":module,"function":function,"grid":grid,"block":block,"shared_bytes":shared_bytes,"arguments":args}), writes)
            },
            Operation::DeviceCopy { stream, destination, source, bytes } =>
                (json!({"kind":"device_copy","stream":stream,"destination":sl(destination),"source":sl(source),"bytes":bytes}), vec![destination.allocation]),
            Operation::Clear { stream, destination, bytes, value } =>
                (json!({"kind":"clear","stream":stream,"destination":sl(destination),"bytes":bytes,"value":value}), vec![destination.allocation]),
            Operation::Fill16 { stream, destination, count, value } =>
                (json!({"kind":"fill16","stream":stream,"destination":sl(destination),"count":count,"value":value}), vec![destination.allocation]),
            Operation::Fill32 { stream, destination, count, value } =>
                (json!({"kind":"fill32","stream":stream,"destination":sl(destination),"count":count,"value":value}), vec![destination.allocation]),
            _ => return Ok(Err(live_self_kind(op))),
        };
        Ok(Ok(serde_json::to_vec(&json!({"operation":operation,"writes":writes})).map_err(|e| e.to_string())?))
    }
    unsafe fn tape_range(&mut self, ex: &mut Execution, base: &Path, frame: &mut Option<ReplayFrame>, step: usize,
                         flat: &[Flat], lo: usize, end: usize, guard: bool) -> Result<Taping, String> {
        self.tape.reset();
        self.precache.clear();
        // speed tape60: the rolling pre-launch copies of the native rows live on the chunk that just ended.
        if let Some(n) = self.native.as_mut() { n.roll_clear(); }
        self.seg_rows.clear();
        self.seg_offs.clear();
        // memory (streamed tape, `guard`): a command the arena cannot hold ends the chunk BEFORE it runs
        // (Taping::Full), so the caller can copy the chunk out and go on from the live state, with no restore.
        let mut full: Option<usize> = None;
        if let Some(c) = self.census.as_mut() { c.begin_segment(step); }
        let mut tape = Vec::with_capacity(end - lo);
        // No-recording self-driven live save (AD_LIVE_SELF; see live_self_cfg). The adapter is made before the first command
        // of a call that has commands in scope (its constructor empties the pre-launch cache, as tape_range just did).
        let live_cfg = live_self_cfg().filter(|c| step >= c.steps.0 && step < c.steps.1);
        let mut live_ls: Option<live_save::LiveSave> = None;
        let live_keep = live_cfg.is_some() && (std::env::var("AD_LIVE_SELF_KEEP").as_deref() == Ok("1") || live_save::LiveSave::sharing_enabled());
        if live_cfg.is_some() {
            live_ls = Some(if live_keep { live_save::LiveSave::attach(self as *mut Win, ex as *mut Execution)? } else { live_save::LiveSave::new(self as *mut Win, ex as *mut Execution)? });
        }
        let mut live_n = (0usize, 0usize);
        let mut live_not: BTreeMap<String, usize> = BTreeMap::new();
        let mut live_stage: BTreeMap<String, usize> = BTreeMap::new();
        let mut live_kernels: BTreeMap<String, usize> = BTreeMap::new();
        let mut live_span: Option<(usize, usize)> = None;
        'ops: for (i, fl) in flat.iter().enumerate().take(end).skip(lo) {
            ex.rk = fl.rk.clone();
            let op = &*fl.op;
            // tape-dedupe-and-pack: what this command takes from the tape, by part (host counters only, no sync).
            let (u_op, ur_op) = (self.tape.used, self.precache.unshared_bytes);
            self.seg_offs.push(u_op);
            let mut post_taken = 0usize;
            let mut cpost: Vec<(u64, usize, usize, u64)> = Vec::new();
            // A window with no tangent pass counts its labels here, each command once (rev_count_once).
            if self.rev_count.is_some() { self.counting = rev_count_once(&mut self.rev_count, step, i); }
            let mut live_done = false;
            // An explicitly installed callback takes ONE first add of the selected step.
            // No binder exec occurs on this arm, and its Tp joins the ordinary tape.
            if self.join_one.as_ref().map_or(false, |j| j.step == step)
                && matches!(op, Operation::Kernel { function, .. } if function == join_one::KIND)
                && !flat[..i].iter().any(|f| matches!(unsafe { &*f.op }, Operation::Kernel { function, .. } if function == join_one::KIND)) {
                let request = self.live_self_request(ex, op, step, frame.as_ref(), true)?.map_err(|k| format!("join unresolved {k}"))?;
                let mut hook = self.join_one.take().unwrap();
                let result = join_one::run(self as *mut Win, ex as *mut Execution, step, i, request, &mut hook);
                self.join_one = Some(hook);
                tape.push(result?);
                if self.rev_count.is_some() { self.counting = false; }
                live_done = true;
            }
            if let Some(c) = live_cfg.filter(|_| !live_done) {
                *live_stage.entry(format!("split {:?} stage {:?} small {:?}", fl.rk.as_ref().and_then(|s| s.split), fl.rk.as_ref().and_then(|s| s.stage).map(|s| s.stage), fl.rk.as_ref().and_then(|s| s.small_step).is_some())).or_insert(0) += 1;
                if c.covers(step, i, fl.rk.as_ref()) {
                    match self.live_self_request(ex, op, step, frame.as_ref(), live_keep)? {
                        Ok(req) => {
                            let s = live_ls.as_mut().ok_or("AD_LIVE_SELF: no adapter")?;
                            s.rebind(self as *mut Win, ex as *mut Execution);
                            let t = s.before(&req, step);
                            if self.rev_count.is_some() { self.counting = false; }
                            let ticket = match t {
                                Ok(t) => t,
                                Err(e) if e.contains("budget exhausted") => { if guard { self.tape.used = u_op; full = Some(i); break 'ops; } return Ok(Taping::Overflow(i)) },
                                Err(e) => return Err(format!("AD_LIVE_SELF step {step} command {i}: live before: {e}")),
                            };
                            // the caller's forward launch: here the binder's own, once, between before and after
                            exec(ex, op, step, base, frame)?;
                            s.rebind(self as *mut Win, ex as *mut Execution);
                            s.after(ticket, true).map_err(|e| format!("AD_LIVE_SELF step {step} command {i}: live after: {e}"))?;
                            tape.push(s.take_records()?.pop().ok_or("AD_LIVE_SELF: live after left no record")?);
                            live_n.0 += 1;
                            live_done = true;
                            live_span = Some((live_span.map_or(i, |s| s.0), i));
                            *live_kernels.entry(match op { Operation::Kernel { function, .. } => function.clone(), Operation::DeviceCopy { .. } => "(device copy)".into(), _ => "(clear or fill)".into() }).or_insert(0) += 1;
                        },
                        Err(kind) => { live_n.1 += 1; *live_not.entry(kind).or_insert(0) += 1; },
                    }
                }
            }
            if !live_done {
            let trl = dev::prof::start();
            // the taped re-run's own rule call may take its descriptor tables from the book (deriv::predict::table)
            deriv::predict::table_scope(true);
            let rl = self.rule(ex, op, step, base, frame.as_ref());
            deriv::predict::table_scope(false);
            dev::prof::add(17, trl);
            if self.rev_count.is_some() { self.counting = false; }
            let (r, l) = rl?;
            match (r, l) {
                (R::Entry, Some((f, args, grid, block, shared))) => {
                    let kd = &self.kernels[&f];
                    if deriv::predict::on() { deriv::predict::set_cmd(&f); }
                    let li = deriv::LaunchInfo { args: &args, grid, block, shared };
                    let tp = dev::prof::start();
                    let cache = if self.pack { Some((&mut self.precache, &self.support)) } else { None };
                    let mut b = match kd.prepare_tape(&ex.cuda, &mut self.tape, &li, &ex.allocations, 4, Some(&self.shadows), self.pack, cache) {
                        Ok(b) => b, Err(e) if exhausted(&e) => { if guard { self.tape.used = u_op; full = Some(i); break 'ops; } return Ok(Taping::Overflow(i)) },
                        Err(e) => return Err(format!("step {step} {f} reverse bind: {e}")),
                    };
                    dev::prof::add(2, tp);
                    if guard {
                        // memory (streamed tape): what this launch can still take after its forward has run (the
                        // rebuild table or copy of each shared span it changes, the copy of each post span) must fit
                        // now: a launch that has run cannot move to the next chunk. If it does not fit, the chunk
                        // ends before this launch.
                        let blk = deriv::tape_block_bytes() > 0;
                        let mut need = 4096usize;
                        for &(_, lo_b, hi_b, _) in b.shared.iter() {
                            let n_b = hi_b.saturating_sub(lo_b);
                            need += 256 + if blk { (n_b / 4).div_ceil(dev::RUN_WORDS).max(1) * 24 } else { n_b };
                        }
                        need += if self.pack { kd.post_spans(&li, &b.env, &ex.allocations)?.values().map(|(lo_b, hi_b)| hi_b.saturating_sub(*lo_b) + 256).sum::<usize>() }
                                else { Self::post_bytes(kd, &args, &ex.allocations) + 256 * args.len() };
                        if self.tape.used + need > self.tape.bytes { self.tape.used = u_op; full = Some(i); break 'ops; }
                    }
                    if !ordered(ex) { dev::sync(&ex.cuda)?; }
                    let tex = dev::prof::start();
                    exec(ex, op, step, base, frame)?;
                    dev::prof::add(16, tex);
                    let tpost = dev::prof::start();
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
                    // bs-tape (a12 backstep): with tape blocks on, every shared pre-launch span is bound through the
                    // rebuild table (bind_recon: the walk-back rebuilds it from the shared copy in the reverse scratch), as
                    // a span this launch changed always was, so no check of the live words against the shared copy is
                    // made after the forward. The shared copy holds the pre-launch words either way, and the walk-back
                    // reads the same words; only an unchanged span's entries now write the scratch copy instead of the
                    // shared one (with the same values). THE WASTE THIS REMOVES: one blocking readback after most taped
                    // launches (those with no post span), each a drain that kept the next launch's host work from running
                    // beside this one on the card. Measured no faster (back-2 bt-t6g26 against bt-t5g26/b), so off unless
                    // AD_TAPE_SHARED_RECON=1.
                    let sh_recon = deriv::tape_block_bytes() > 0 && std::env::var("AD_TAPE_SHARED_RECON").as_deref() == Ok("1");
                    let nsh = if sh_recon { 0 } else { b.shared.len() };
                    let nslot = nsh + 2 * posts.len();
                    // bs-tape: nothing to compare (no shared span checked, no post candidate): no readback; lookup() below
                    // decides "copy" for a span with no candidate without reading anything, as before
                    let empty = nsh == 0 && cands.iter().all(|c| c.0.is_none() && c.1.is_none()) && std::env::var("AD_TAPE_SKIP_EMPTY").as_deref() != Ok("0");
                    // rerun-tape (AD_TAPE_PREDICT=1): one slot is batched too (the same reset, compare and readback same_words
                    // makes), so its decision can be predicted; the answer then comes from the book, checked at the chunk end.
                    let words: Option<Vec<[u64; 3]>> = if !empty && ordered(ex) && (nslot > 1 || (nslot == 1 && deriv::predict::on())) && nslot <= dev::SLOTS && std::env::var("AD_CHECK_BATCH").as_deref() != Ok("0") {
                        self.support.reset_words(&ex.cuda)?;
                        for (k, &(al, lo, hi, at)) in b.shared.iter().enumerate().filter(|_| !sh_recon) {
                            let live = ex.allocations.get(&al).ok_or("shared primal allocation not live")?.address + lo as u64;
                            if hi > lo { self.support.words_into(&ex.cuda, k, live, at, hi - lo)?; }
                        }
                        for (j, (&(al, (lo, hi)), &(s, wc))) in posts.iter().zip(&cands).enumerate() {
                            let a = ex.allocations.get(&al).ok_or("post allocation not live")?.address + lo as u64;
                            if let Some((clo, copy)) = s { self.support.words_into(&ex.cuda, nsh + 2 * j, a, copy + (lo - clo) as u64, hi - lo)?; }
                            if let Some((clo, copy)) = wc { self.support.words_into(&ex.cuda, nsh + 2 * j + 1, a, copy + (lo - clo) as u64, hi - lo)?; }
                        }
                        Some(if deriv::predict::on() { deriv::predict::words(&ex.cuda, &self.support, deriv::tape_predict::SITE_POST_WORDS, nslot)? }
                             else { self.support.collect(&ex.cuda, 0, nslot)?.1 })
                    } else { None };
                    if !b.shared.is_empty() {
                        // same_words reads back with a blocking cuMemcpyDtoH, which already waits for the forward above.
                        if !ordered(ex) && !sh_recon { dev::sync(&ex.cuda)?; }
                        for (k, (al, lo, hi, at)) in std::mem::take(&mut b.shared).into_iter().enumerate() {
                            let live = ex.allocations.get(&al).ok_or("shared primal allocation not live")?.address + lo as u64;
                            let same = if sh_recon { hi <= lo } else { match &words { Some(wv) => hi <= lo || wv[k][0] == 0, None => deriv::same_words(&ex.cuda, &self.support, live, at, hi - lo)? } };
                            if same { b.shared.push((al, lo, hi, at)); continue; }
                            if deriv::tape_block_bytes() > 0 {
                                // tape-dedupe-and-pack (blocks): the shared copy still holds the pre-launch words; the walk-back
                                // rebuilds the span from it in the reverse scratch instead of a second tape copy.
                                let names = b.cluster_args.get(&(al, lo)).cloned().unwrap_or_default();
                                let rt = dev::run_rows(&[(0, at, hi - lo)]);
                                let tab = match self.tape.take(rt.len() * 8) { Ok(c) => c, Err(e) if exhausted(&e) => { if guard { return Err(format!("DEFECT streamed tape: step {step} command {i} needed tape after its forward ran (post bound too small)")); } return Ok(Taping::Overflow(i)) }, Err(e) => return Err(e) };
                                dev::upload(&ex.cuda, tab, std::slice::from_raw_parts(rt.as_ptr().cast::<u8>(), rt.len() * 8))?;
                                b.recon.push((al, lo, hi, (tab, rt.len() / 3), names, true));
                                self.precache.blk_spans += 1; self.precache.blk_hit_bytes += hi - lo;
                                continue;
                            }
                            let c = match self.tape.take(hi - lo) { Ok(c) => c, Err(e) if exhausted(&e) => { if guard { return Err(format!("DEFECT streamed tape: step {step} command {i} needed tape after its forward ran (post bound too small)")); } return Ok(Taping::Overflow(i)) }, Err(e) => return Err(e) };
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
                            let copy = match self.tape.take(hi - lo) { Ok(c) => c, Err(e) if exhausted(&e) => { if guard { return Err(format!("DEFECT streamed tape: step {step} command {i} needed tape after its forward ran (post bound too small)")); } return Ok(Taping::Overflow(i)) }, Err(e) => return Err(e) };
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
                                    { if guard { return Err(format!("DEFECT streamed tape: step {step} command {i} needed tape after its forward ran (post bound too small)")); } return Ok(Taping::Overflow(i)) } }, Err(e) => return Err(e) };
                                let tc = dev::prof::start(); dev::copy(&ex.cuda, copy, a.address, a.bytes)?; dev::prof::add(10, tc);
                                post.insert(s.allocation, submit::Allocation { id: a.id, address: copy, bytes: a.bytes });
                            }
                        }
                    }
                    dev::prof::add(22, tpost);
                    tape.push(Tp::Entry { f, b, args, grid, block, shared, post });
                },
                (R::Native(_), Some((f, args, grid, block, _))) => {
                    let n = self.native.as_mut().ok_or("native rows not loaded")?;
                    let tp = dev::prof::start();
                    // (deriv::predict::roll_on): the launch's rolling compares batched and predicted under its own book key,
                    // and, in an ordered plan, no drain after its prepare or before its forward (AD_TAPE_ROLL_DRAIN=1 keeps them)
                    let rr3 = self.pack && deriv::predict::roll_on();
                    let rr3_nodrain = rr3 && ordered(ex) && !deriv::predict::roll_drain();
                    if rr3 { deriv::predict::set_cmd(&f); }
                    // speed tape60: on a reverse tape chunk the MYNN-EDMF launches share one rolling copy of each argument span.
                    let p = match if self.pack { n.prepare_deferred_roll_w(&ex.cuda, &mut self.tape, &ex.allocations, &f, &args, grid, block, &self.support, !rr3_nodrain, rr3) } else { n.prepare(&ex.cuda, &mut self.tape, &ex.allocations, &f, &args, grid, block) } {
                        Ok(p) => p, Err(e) if exhausted(&e) => { if guard { self.tape.used = u_op; full = Some(i); break 'ops; } return Ok(Taping::Overflow(i)) },
                        Err(e) => return Err(format!("step {step} {f} native reverse bind: {e}")),
                    };
                    dev::prof::add(2, tp);
                    if !rr3_nodrain { dev::sync(&ex.cuda)?; }
                    exec(ex, op, step, base, frame)?;
                    tape.push(Tp::Native { f, p });
                },
                (R::Tempo(_), Some((f, args, _, _, _))) => {
                    let snap = { let (t, _) = self.tempo.as_ref().ok_or("TEMPO derivative not loaded")?;
                        match t.snapshot(&ex.cuda, &mut self.tape, &ex.allocations, &args) {
                            Ok(s) => s, Err(e) if exhausted(&e) => { if guard { self.tape.used = u_op; full = Some(i); break 'ops; } return Ok(Taping::Overflow(i)) },
                            Err(e) => return Err(format!("step {step} {f} TEMPO reverse snapshot: {e}")),
                        } };
                    exec(ex, op, step, base, frame)?;
                    tape.push(Tp::Tempo { f, snap });
                },
                (R::TempoSerial(_), Some((f, args, _, _, _))) => {
                    let t = self.tempo_serial.as_ref().ok_or("TEMPO serial entry not loaded")?;
                    let snap = match t.snapshot(&ex.cuda, &mut self.tape, &ex.allocations, &f, &args) {
                        Ok(s) => s, Err(e) if exhausted(&e) => { if guard { self.tape.used = u_op; full = Some(i); break 'ops; } return Ok(Taping::Overflow(i)) },
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
                        Ok(s) => s, Err(e) if exhausted(&e) => { if guard { self.tape.used = u_op; full = Some(i); break 'ops; } return Ok(Taping::Overflow(i)) },
                        Err(e) => return Err(format!("step {step} {f} tempo-small reverse snapshot: {e}")),
                    };
                    exec(ex, op, step, base, frame)?;
                    tape.push(Tp::TempoSmall { f, snap, ins, outs });
                },
                (r @ R::FctMinmax { .. }, _) => {
                    let R::FctMinmax { prim, .. } = &r else { unreachable!() };
                    let p = match self.tape_fct(ex, prim) { Ok(p) => p, Err(e) if exhausted(&e) => { if guard { self.tape.used = u_op; full = Some(i); break 'ops; } return Ok(Taping::Overflow(i)) }, Err(e) => return Err(e) };
                    exec(ex, op, step, base, frame)?;
                    tape.push(Tp::Rule { r, prim: p });
                },
                (r @ R::Max { .. }, _) | (r @ R::Where { .. }, _) => {
                    let (al, first, n, unit) = match &r { R::Max { prim: (a, f, n), .. } => (*a, *f, *n, 4), R::Where { prim: (a, f, n), .. } => (*a, *f, *n, 1), _ => unreachable!() };
                    let p = match self.tape_primal(ex, al, first, n, unit) { Ok(p) => p, Err(e) if exhausted(&e) => { if guard { self.tape.used = u_op; full = Some(i); break 'ops; } return Ok(Taping::Overflow(i)) }, Err(e) => return Err(e) };
                    exec(ex, op, step, base, frame)?;
                    tape.push(Tp::Rule { r, prim: p });
                },
                (r, _) => { exec(ex, op, step, base, frame)?; tape.push(Tp::Rule { r, prim: 0 }); },
            }
            }
            let taken = self.tape.used - u_op;
            let recopy = self.precache.unshared_bytes.saturating_sub(ur_op);
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
        self.seg_offs.truncate(tape.len());
        if live_cfg.is_some() {
            eprintln!("{}", json!({"event":"live_self","step":step,"lo":lo,"end":end,"full_at":full,"records":tape.len(),"live_commands":live_n.0,
                "in_scope_not_live":live_n.1,"keep":live_keep,"not_live_kinds":live_not,"commands_by_rk_scope":live_stage,"tape_used":self.tape.used,"live_first_last":live_span,"live_kinds":live_kernels}));
        }
        Ok(match full { Some(i) => Taping::Full(i, tape), None => Taping::Done(tape) })
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
    /// memory (streamed tape): start the helper thread that makes `count` page-locked chunk buffers of `bytes`
    /// (the anchors-ahead pattern, with its own pool: the redo compare empties the anchors' pool when none fits it).
    unsafe fn st_ahead_start(&mut self, ex: &Execution, count: usize, bytes: usize) -> Result<(), String> {
        if count == 0 || bytes == 0 || self.st_ahead.is_some() { return Ok(()); }
        let pool: AheadPool = std::sync::Arc::new((std::sync::Mutex::new((Vec::new(), false)), std::sync::Condvar::new()));
        let (fns, set, c, p) = (crate::pinned::PinFns::of(&ex.cuda), ex.cuda.ctx_set_current, ex.context as usize, pool.clone());
        let want = self.st_want.clone();
        self.st_ahead_bytes = bytes;
        std::thread::Builder::new().name("tape-chunks-ahead".into()).spawn(move || {
            let ok = set(c as *mut std::ffi::c_void) == 0;
            for _ in 0..count {
                if !ok { break; }
                // a bounded number of buffers wait unused: the next one is made when the reverse has taken one (st_take
                // notifies).
                {
                    let (l, cv) = &*p;
                    let Ok(mut g) = l.lock() else { break };
                    // speed tape60: st_want buffers wait unused (set by st_plan, then by each chunk of a step; 0: it sleeps)
                    while g.0.len() >= want.load(std::sync::atomic::Ordering::SeqCst) { g = match cv.wait(g) { Ok(g) => g, Err(_) => return }; }
                }
                let Ok(m) = crate::pinned::PinMem::new(fns, bytes) else { break };
                let (l, cv) = &*p;
                // speed tape60: a buffer finished after the step said it needs no more is freed here, not kept
                if let Ok(mut g) = l.lock() { if want.load(std::sync::atomic::Ordering::SeqCst) > 0 { g.0.push(crate::pinned::PinnedBuf::from_mem(m, bytes)); } }
                cv.notify_all();
            }
            let (l, cv) = &*p;
            if let Ok(mut g) = l.lock() { g.1 = true; }
            cv.notify_all();
        }).map_err(|e| e.to_string())?;
        self.st_ahead = Some(pool);
        Ok(())
    }
    /// memory (streamed tape): start making the page-locked chunk buffers a step's tape needs beyond the arena (one
    /// per full chunk; the last chunk stays in the arena), from the tape estimates known so far. Called after the first
    /// tangent step (the arena has its tangent size by then: a later tangent bind that grows it only makes these buffers
    /// too small, and st_take then makes a fitting one) and again before the reverse (a no-op once started).
    unsafe fn st_plan(&mut self, ex: &Execution, flats: &[Vec<Flat>], when: &str) -> Result<(), String> {
        // bs-tape: every step's estimate, kept for st_trim (called again before the reverse, when all are known)
        if self.st_on && self.pack { self.st_est = flats.iter().map(|f| f.iter().map(|x| x.est.div_ceil(256) * 256 + 256).sum::<usize>()).collect(); }
        if !(self.st_on && self.pack) || self.st_ahead.is_some() || self.st_planned { return Ok(()); }
        self.st_planned = true;
        let need = flats.iter().map(|f| f.iter().map(|x| x.est.div_ceil(256) * 256 + 256).sum::<usize>()).max().unwrap_or(0);
        let guess = std::env::var("AD_TAPE_STREAM_GUESS").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.6).clamp(0.01, 1.0);
        let per = self.tape.bytes.max(1);
        let count = ((need as f64 * guess) as usize / per).min(mem_available() / 2 / per);
        // speed tape60 (AD_TAPE_STREAM_FIRST, default 0.18 of the unpacked estimate; AD_TAPE_STREAM_TRIM=0: ST_AHEAD
        // throughout, as before): how many buffers the helper may run ahead by once the first reversed step begins.
        // THE BREAKAGE THIS REMOVES: the helper kept 2 made ahead and the first reversed step took one every 0.3 s, so
        // the step waited for page-locking: 13.06 s on a test machine (<local path> test machine/f-60-a) and
        // 13.48 s here when the other card ran a 60 km leg too (speed/fast/tape60 runs/exp2/e4), 1 to 1.6 s alone.
        // During the tangent pass it still makes ST_AHEAD only: made there, 7 buffers sat under the self dot's own
        // downloads and the process peaked at 198.3 GiB before the reverse began (runs/pair5/v1.rss) against 167.7.
        let first_guess = std::env::var("AD_TAPE_STREAM_FIRST").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.18).clamp(0.0, 1.0);
        self.st_first = if st_trim_on() { ((need as f64 * first_guess) as usize / per).clamp(ST_AHEAD.min(count), count.max(ST_AHEAD.min(count))) } else { ST_AHEAD };
        self.st_want.store(ST_AHEAD, std::sync::atomic::Ordering::SeqCst);
        eprintln!("{}", json!({"event":"window_tape_stream","when":when,"tape_bytes":self.tape.bytes,"largest_step_estimate_bytes":need,"packed_guess":guess,
                               "chunk_buffers_most":count,"made_ahead_of_use":ST_AHEAD,"made_ahead_in_the_first_reversed_step":self.st_first,"bytes_each":per}));
        self.st_ahead_start(ex, count, per)
    }
    /// speed tape60 (AD_TAPE_STREAM_TRIM=0 keeps the old behaviour): no reverse sweep follows, so the streamed tape's
    /// page-locked buffers go back to the system. THE BREAKAGE THIS REMOVES: the run's host memory peak is the download of
    /// the whole result after the last sweep (25.5 GB at x1.163842, a new pageable vector), made while the tape's
    /// buffers (9 to 15 of 7 GiB) sat page-locked and unused: read in a half-second series of the process's memory,
    /// 167.8 GiB through the sweep, 191.5 GiB in that download (a test machine <local path>).
    /// They are freed on a thread of their own (9 buffers took 2.5 s on this thread, runs/pair/p1), beside the result's dot.
    unsafe fn st_release(&mut self, ex: &Execution) {
        if !st_trim_on() || !self.st_on { return; }
        let mut bufs: Vec<crate::pinned::PinnedBuf> = std::mem::take(&mut self.st_pool);
        if let Some(pool) = self.st_ahead.as_ref() {
            let (l, cv) = &**pool;
            if let Ok(mut g) = l.lock() { self.st_want.store(0, std::sync::atomic::Ordering::SeqCst); bufs.append(&mut g.0); }
            cv.notify_all();
        }
        if bufs.is_empty() { return; }
        eprintln!("{}", json!({"event":"window_tape_stream_release","buffers_freed":bufs.len(),"bytes_freed":bufs.iter().map(|b| b.len()).sum::<usize>()}));
        let (set, c) = (ex.cuda.ctx_set_current, ex.context as usize);
        // a thread that cannot start, or cannot take the context, leaves them to the end of the process as before
        let _ = std::thread::Builder::new().name("tape-chunks-free".into()).spawn(move || {
            if set(c as *mut std::ffi::c_void) == 0 { drop(bufs); } else { std::mem::forget(bufs); }
        });
    }
    /// speed tape60 (AD_TAPE_STREAM_TRIM=0 keeps the old behaviour): a streamed step has taped its last chunk, so the
    /// number of chunk buffers it needs is known. THE BREAKAGE THIS REMOVES: the helper thread always kept ST_AHEAD (2)
    /// page-locked buffers made ahead; a step takes its buffers from the pool of the step before, so those two (2 x 7 GiB
    /// at x1.163842) stayed page-locked and untouched for the rest of the window. The buffers nobody took are freed and
    /// the helper sleeps (st_want 0) until a later step is expected to take more than the pool holds.
    fn st_trim(&mut self, step: usize) {
        if !st_trim_on() { return; }
        // bs-tape (a12 backstep): a step still to be reversed (a lower step number) whose tape is expected to overflow the
        // arena (its estimate times the largest packed ratio measured so far) keeps that many buffers (at most ST_AHEAD)
        // made ahead, from the pool of the step before or by the helper, instead of none. THE BREAKAGE THIS REMOVES: with
        // the tape on the card (--tape-gib 24) every step but the radiation step fits the arena, so the pool was emptied
        // after each step and the radiation step page-locked its 24 GiB chunk buffer on the main thread with the card
        // idle (a 27 s wait in the last step reversed, a development line 2026-11-12). Bytes and words are unchanged: only who
        // makes a buffer and when. AD_TAPE_STREAM_KEEP=0 frees them as before.
        let keep = if std::env::var("AD_TAPE_STREAM_KEEP").as_deref() == Ok("0") || self.pack_ratio <= 0.0 { 0 } else {
            let per = self.tape.bytes.max(1) as f64;
            self.st_est.iter().take(step.saturating_sub(1))
                .map(|&e| (((e as f64 * self.pack_ratio) / per).ceil() as usize).saturating_sub(1)).max().unwrap_or(0).min(ST_AHEAD)
        };
        let keep = keep.saturating_sub(self.st_pool.len());
        let Some(pool) = self.st_ahead.as_ref() else { return };
        let (l, cv) = &**pool;
        let freed = match l.lock() { Ok(mut g) => {
            self.st_want.store(keep, std::sync::atomic::Ordering::SeqCst);
            let n = g.0.len().saturating_sub(keep);
            let v: Vec<crate::pinned::PinnedBuf> = g.0.drain(..n).collect();
            v.len() }, Err(_) => 0 };
        cv.notify_all();
        if freed > 0 || keep > 0 { eprintln!("{}", json!({"event":"window_tape_stream_trim","step":step,"buffers_freed":freed,"buffers_kept_for_a_later_step":keep,"bytes_each":self.st_ahead_bytes})); }
    }
    /// memory (streamed tape): a page-locked chunk buffer of at least `bytes`: one kept from an earlier step, one the
    /// helper thread made (waiting while it is still making them), or a new one made here.
    unsafe fn st_take(&mut self, cuda: &Cuda, bytes: usize) -> Result<crate::pinned::PinnedBuf, String> {
        if let Some(i) = self.st_pool.iter().position(|b| b.len() >= bytes) { return Ok(self.st_pool.swap_remove(i)); }
        if let Some(pool) = self.st_ahead.clone() {
            let (l, cv) = &*pool;
            let mut g = l.lock().map_err(|_| "streamed tape buffer pool poisoned".to_string())?;
            let mut done = false;
            loop {
                if let Some(i) = g.0.iter().position(|b| b.len() >= bytes) { let b = g.0.swap_remove(i); drop(g); cv.notify_all(); return Ok(b); }
                if g.1 { done = true; break; }
                // none fits and the helper cannot make one that does (its buffers are smaller, or its look-ahead is
                // full of them): this one is made here.
                if self.st_ahead_bytes < bytes || !g.0.is_empty() { break; }
                // speed tape60: the helper was told no more buffers are expected; this one is made here
                if self.st_want.load(std::sync::atomic::Ordering::SeqCst) == 0 { break; }
                g = cv.wait(g).map_err(|_| "streamed tape buffer pool poisoned".to_string())?;
            }
            drop(g);
            if done { self.st_ahead = None; }
        }
        // bs-tape (a12 backstep): an idle page-locked checkpoint buffer of the sweep's pool is borrowed for the chunk (a
        // chunk may span several buffers: st_flush's put takes the next one when one is full) and goes back to the pool
        // when the step ends. THE WASTE THIS REMOVES: with the tape on the card (--tape-gib 26) the radiation step's first
        // chunk page-locked a new 26 GiB buffer here with the card idle: 1.9 s (back-2 bt-t3g26 step 1), while dropped
        // checkpoints' buffers sat idle in the pool. Only where the chunk's bytes sit on the host changes; the bytes put
        // back are the same. AD_TAPE_BORROW=0 page-locks a new buffer as before.
        if bs_pool() != 0 && std::env::var("AD_TAPE_BORROW").as_deref() != Ok("0") {
            if let Some(m) = capi::pin_pool_take_largest(bs_pool(), 256 << 20) {
                let n = m.cap();
                let b = crate::pinned::PinnedBuf::from_mem(m, n);
                self.st_borrowed.push(b.as_ptr() as usize);
                return Ok(b);
            }
        }
        self.st_made += 1;
        crate::pinned::PinnedBuf::new(cuda, bytes.max(self.tape.bytes))
    }
    /// memory (streamed tape): the chunk taped so far ([0, used) of the tape arena) goes to page-locked host memory, on
    /// the copy stream, in pieces; the arena carries a write fence per piece, so the next chunk's first write into a
    /// piece waits on the device for that piece's copy (revmach's fused-tape flush, on the reverse tape arena).
    /// speed tape60 (host-shared tape blocks; AD_TAPE_SHARE=0 turns the sharing off). THE BREAKAGE THIS REMOVES: every
    /// chunk starts with an empty copy cache, so a field that did not change is copied onto the tape again in each
    /// chunk that reads it, and every chunk went out to its own page-locked buffer whole (x1.163842: 16 chunks, 108 GB
    /// out and back each step; a census of one step found 47 GB of it equal to bytes of an earlier chunk). Now the
    /// chunks are packed one after another into the step's host buffers (`log`), and a block of a pre- or post-launch
    /// copy (AD_TAPE_SHARE_KIB, 64 KiB, aligned to its allocation) whose bytes and digest (two 64-bit sums of mixed
    /// (word, position) terms, e2e_spans_digest) equal those of a block already in host memory (of any allocation: an
    /// unchanged field, a field copied to another array, a block of zeros) is not copied out: its run names the host
    /// address of the earlier block. The upload puts the same words at the
    /// same arena addresses (fz_pf_issue), so the tape the walk-back reads is byte for byte the tape that was written.
    /// AD_TAPE_SHARE_VERIFY=1 is the proof mode: every shared block is compared word for word with its source (an
    /// earlier chunk's block is uploaded to the reverse scratch for it) and a block that differs is copied out and counted.
    /// Returns the chunk's runs (arena offset, bytes, host address), the seconds spent waiting for buffers, the bytes
    /// copied out, the bytes shared, the blocks digested and the proof mode's mismatches.
    #[allow(clippy::type_complexity)]
    unsafe fn st_flush(&mut self, cuda: &Cuda, used: usize, log: &mut Vec<crate::pinned::PinnedBuf>, log_off: &mut usize,
                       share: &mut HashMap<([u64; 2], usize), usize>) -> Result<(Vec<(usize, usize, usize)>, f64, usize, usize, usize, usize), String> {
        // every taping copy and forward command issued so far is complete before the chunk is read (and every earlier
        // chunk's copy-out is complete before its host bytes are named as a source)
        dev::sync(cuda)?;
        if let Some(f) = self.tape.fence.as_mut() { f.upto(usize::MAX)?; }
        if self.fz_copy.is_null() { checked("cuStreamCreate streamed tape", (cuda.stream_create)(&mut self.fz_copy, 1))?; }
        let piece = std::env::var("AD_FUSE_PIECE_MIB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(256).max(1) << 20;
        let base = self.tape.base;
        // 1. the blocks of this chunk's pre- and post-launch copies: (arena offset, bytes)
        let mut blocks: Vec<(usize, usize)> = Vec::new();
        if st_share_on() {
            let bs = st_share_block();
            for v in self.precache.spans.values() {
                for &(lo, hi, copy) in v {
                    if copy < base || hi <= lo || lo % 4 != 0 || (hi - lo) % 4 != 0 { continue; }
                    let a0 = (copy - base) as usize;
                    if a0 + (hi - lo) > used { continue; }
                    let mut o = lo;
                    while o < hi {
                        let e = ((o / bs + 1) * bs).min(hi);
                        if e - o >= 4096 { blocks.push((a0 + (o - lo), e - o)); }
                        o = e;
                    }
                }
            }
            blocks.sort_unstable_by_key(|b| b.0);
            // copies do not overlap on the tape; a block that would is left out (its bytes are copied out as they are)
            let mut end = 0usize;
            blocks.retain(|b| { let ok = b.0 >= end; if ok { end = b.0 + b.1; } ok });
        }
        let dig = if blocks.is_empty() { Vec::new() } else { self.support.spans_digest(cuda, &blocks.iter().map(|b| (base + b.0 as u64, b.1)).collect::<Vec<_>>())? };
        // 2. each block's source: Ok(host address) = a block of an earlier chunk, Err(index) = an earlier block of this chunk
        let mut src: Vec<Option<Result<usize, usize>>> = Vec::with_capacity(blocks.len());
        let mut local: HashMap<([u64; 2], usize), usize> = HashMap::with_capacity(blocks.len());
        for (i, b) in blocks.iter().enumerate() {
            let key = (dig[i], b.1);
            src.push(match share.get(&key) { Some(&h) => Some(Ok(h)), None => local.get(&key).map(|&j| Err(j)) });
            local.entry(key).or_insert(i);
        }
        // 3. the proof mode: word for word
        let mut bad = 0usize;
        if st_share_verify() {
            let near: Vec<usize> = (0..blocks.len()).filter(|&i| matches!(src[i], Some(Err(_)))).collect();
            let pairs: Vec<(u64, u64, usize)> = near.iter().map(|&i| { let Some(Err(j)) = src[i] else { unreachable!() }; (base + blocks[i].0 as u64, base + blocks[j].0 as u64, blocks[i].1) }).collect();
            for (&i, d) in near.iter().zip(self.support.pairs_differ(cuda, &pairs)?) { if d { src[i] = None; bad += 1; } }
            let far: Vec<usize> = (0..blocks.len()).filter(|&i| matches!(src[i], Some(Ok(_)))).collect();
            let cap = self.scratch.bytes;
            let mut k = 0usize;
            while k < far.len() {
                let (mut at, mut batch) = (0usize, Vec::new());
                while k < far.len() && at + blocks[far[k]].1 <= cap {
                    let i = far[k];
                    let Some(Ok(h)) = src[i] else { unreachable!() };
                    checked("cuMemcpyHtoDAsync streamed tape proof", (cuda.upload_async)(self.scratch.base + at as u64, (h as *const u8).cast(), blocks[i].1, std::ptr::null_mut()))?;
                    batch.push((i, at));
                    at += blocks[i].1;
                    k += 1;
                }
                if batch.is_empty() { return Err(format!("streamed tape proof mode: a {} byte block does not fit the {cap} byte reverse scratch", blocks[far[k]].1)); }
                let pairs: Vec<(u64, u64, usize)> = batch.iter().map(|&(i, at)| (base + blocks[i].0 as u64, self.scratch.base + at as u64, blocks[i].1)).collect();
                for (&(i, _), d) in batch.iter().zip(self.support.pairs_differ(cuda, &pairs)?) { if d { src[i] = None; bad += 1; } }
            }
        }
        // 4. pack: bytes that are not shared go to the host log in arena order; a shared block names its source's address
        let mut runs: Vec<(usize, usize, usize)> = Vec::new();
        let mut host_of: Vec<usize> = vec![0; blocks.len()];
        let (mut waited, mut sent, mut shared_b) = (0f64, 0usize, 0usize);
        let mut pieces = std::collections::VecDeque::new();
        let (mut since, mut floor) = (0usize, 0usize);
        // the copy-out not yet issued: (arena offset, bytes, host address); one driver call per run of bytes that are
        // next to each other on the tape and in host memory (a call per 64 KiB block cost 0.16 s a chunk)
        let mut pend: (usize, usize, usize) = (0, 0, 0);
        // [a, a + n) of the arena: copied out (host None; returns the host address when it landed in one piece, else 0) or shared
        let mut put = |me: &mut Self, a: usize, n: usize, host: Option<usize>| -> Result<usize, String> {
            let push = |runs: &mut Vec<(usize, usize, usize)>, a: usize, n: usize, h: usize| match runs.last_mut() {
                Some(r) if r.0 + r.1 == a && r.2 + r.1 == h => r.1 += n,
                _ => runs.push((a, n, h)),
            };
            if let Some(h) = host { push(&mut runs, a, n, h); shared_b += n; return Ok(h); }
            let (mut a, mut n, mut first, mut done, mut whole) = (a, n, 0usize, 0usize, true);
            while n > 0 {
                if log.last().is_none_or(|b| *log_off >= b.len()) {
                    let tb = Instant::now();
                    let want = me.tape.bytes;
                    log.push(me.st_take(cuda, want)?);
                    *log_off = 0;
                    waited += tb.elapsed().as_secs_f64();
                }
                let buf = log.last_mut().ok_or("streamed tape: no host buffer")?;
                let k = n.min(buf.len() - *log_off).min(piece - since);
                let hp = buf.as_mut_ptr().add(*log_off);
                if pend.1 > 0 && pend.0 + pend.1 == a && pend.2 + pend.1 == hp as usize { pend.1 += k; }
                else {
                    if pend.1 > 0 { checked("cuMemcpyDtoHAsync streamed tape", (cuda.download_async)((pend.2 as *mut u8).cast(), base + pend.0 as u64, pend.1, me.fz_copy))?; }
                    pend = (a, k, hp as usize);
                }
                push(&mut runs, a, k, hp as usize);
                if done == 0 { first = hp as usize; } else if hp as usize != first + done { whole = false; }
                done += k;
                *log_off += k; a += k; n -= k; sent += k; since += k;
                if since >= piece {
                    if pend.1 > 0 { checked("cuMemcpyDtoHAsync streamed tape", (cuda.download_async)((pend.2 as *mut u8).cast(), base + pend.0 as u64, pend.1, me.fz_copy))?; pend.1 = 0; }
                    let mut ev = std::ptr::null_mut();
                    checked("cuEventCreate streamed tape piece", (cuda.event_create)(&mut ev, 2))?;
                    me.fz_evs.push(ev);
                    checked("cuEventRecord streamed tape piece", (cuda.event_record)(ev, me.fz_copy))?;
                    pieces.push_back((floor, ev));
                    floor = a; since = 0;
                }
            }
            Ok(if whole { first } else { 0 })
        };
        let mut cur = 0usize;
        for i in 0..blocks.len() {
            let (bo, bl) = blocks[i];
            if bo > cur { put(self, cur, bo - cur, None)?; }
            let from = match src[i] { Some(Ok(h)) => h, Some(Err(j)) => host_of[j], None => 0 };
            if from != 0 { host_of[i] = put(self, bo, bl, Some(from))?; }
            else {
                host_of[i] = put(self, bo, bl, None)?;
                if host_of[i] != 0 { share.entry((dig[i], bl)).or_insert(host_of[i]); }
            }
            cur = bo + bl;
        }
        if used > cur { put(self, cur, used - cur, None)?; }
        if pend.1 > 0 { checked("cuMemcpyDtoHAsync streamed tape", (cuda.download_async)((pend.2 as *mut u8).cast(), base + pend.0 as u64, pend.1, self.fz_copy))?; }
        if since > 0 {
            let mut ev = std::ptr::null_mut();
            checked("cuEventCreate streamed tape piece", (cuda.event_create)(&mut ev, 2))?;
            self.fz_evs.push(ev);
            checked("cuEventRecord streamed tape piece", (cuda.event_record)(ev, self.fz_copy))?;
            pieces.push_back((floor, ev));
        }
        let w0 = self.tape.fence.as_ref().map(|f| f.waited).unwrap_or(0);
        self.tape.fence = Some(dev::Fence { pieces, wait: cuda.stream_wait_event, waited: w0 });
        Ok((runs, waited, sent, shared_b, blocks.len(), bad))
    }
    /// memory (streamed tape; AD_TAPE_STREAM=1 or --tape-stream 1). THE BREAKAGE THIS PREVENTS: a step whose packed
    /// tape does not fit the tape arena was cut into segments, each of which restored an anchor and replayed the forward
    /// up to its start (x1.163842, 60 km, a 10 GiB arena for a 78 GB step tape: 21 segments, 82,805 commands replayed
    /// for 10,634; a test machine final-60km-8c). Here the step is replayed ONCE from its start boundary and taped into the same
    /// arena; when the arena cannot hold the next command, the chunk taped so far is copied to page-locked host RAM
    /// (st_flush) and the next chunk starts at offset 0. A chunk is self-contained exactly as a segment is (tape_range
    /// starts it with an empty shared-copy cache). The walk-back takes the chunks last first: the last one is still in
    /// the arena; each earlier one is uploaded to the addresses it was taped at, piece by piece, into arena bytes the
    /// walk-back has already finished with (the revmach prefetch, here between chunks of one step). The commands, the
    /// walk-back order and every taped word are the segmented path's, so the reverse words are the same. No anchors, no
    /// second arena, and it works for any window length and with --checkpoints.
    #[allow(clippy::too_many_arguments)]
    unsafe fn stream_reverse_step(&mut self, ex: &mut Execution, base: &Path, frame: &mut Option<ReplayFrame>, step: usize, flat: &[Flat],
                                  prev: &Boundary, prev_data: &[u8], cur: &Boundary, cur_data: &[u8], t0: Instant, ms0: f64, total_est: usize) -> Result<Value, String> {
        // speed tape60 (host-shared tape blocks): `runs` = where the chunk's bytes sit in host memory (None: the chunk is
        // still in the arena). `log` = the page-locked buffers of this step in the order they were filled; `share` = the
        // host address of a copied-out block of each (digest, bytes).
        struct Chunk { rows: Vec<Tp>, offs: Vec<usize>, used: usize, runs: Option<Vec<(usize, usize, usize)>> }
        let mut log: Vec<crate::pinned::PinnedBuf> = Vec::new();
        let mut log_off = 0usize;
        let mut share: HashMap<([u64; 2], usize), usize> = HashMap::new();
        let (mut shared_bytes, mut share_blocks, mut share_bad) = (0usize, 0usize, 0usize);
        let n = flat.len();
        // bs-tape (AD_G5_PROF only): what the host timers counted before this step began (the sweep's recompute and
        // restores), kept apart so the step's own host_prof covers this step only
        let prof_before: Vec<Value> = if dev::prof::on() { dev::prof::take().into_iter().filter(|x| x.2 > 0).map(|(k, s, c)| json!([k, s, c])).collect() } else { Vec::new() };
        // speed tape60: the first streamed step of the run begins; the helper may now run ahead by st_first buffers
        if self.st_info.is_empty() && st_trim_on() {
            if let Some(pool) = self.st_ahead.as_ref() {
                let (l, cv) = &**pool;
                if let Ok(_g) = l.lock() { self.st_want.store(self.st_first, std::sync::atomic::Ordering::SeqCst); }
                cv.notify_all();
            }
        }
        crate::spill::phase(&format!("ad{step}:restore"));
        let t_r = Instant::now();
        self.restore(ex, frame, prev, prev_data)?;
        dead_poison_rev(ex, step, "restore")?;
        let restore_s = t_r.elapsed().as_secs_f64();
        if let Some(f) = frame.as_mut() { f.begin(step)?; }
        crate::spill::phase(&format!("ad{step}:tape"));
        let t_tape = Instant::now();
        // bs-tape (a12 backstep): the copy-row gathers of this step's re-run are built on worker threads (gather_defer, as
        // the tangent pass does), not on the main thread inside rule(): nothing in the re-run reads them, and the
        // walk-back's first use of each one joins its build (gather_for). THE WASTE THIS REMOVES: every reverse step's
        // rows are new to the cache (about 490 a 3 km step, none hit), and building them one after another took 0.75 to
        // 1.09 s of every re-run with the card waiting (back-2 bt-t1p, AD_G5_PROF gather_build). The same rows give the
        // same gather, so the walk-back's words are unchanged. AD_REV_GATHER_DEFER=0 builds them in place as before.
        let defer0 = (self.defer, self.defer_step);
        if std::env::var("AD_REV_GATHER_DEFER").as_deref() != Ok("0") { self.defer = true; self.defer_step = step; }
        let mut chunks: Vec<Chunk> = Vec::new();
        let (mut lo, mut tape_sum, mut tape_peak, mut d2h_bytes, mut flush_s, mut buf_wait_s) = (0usize, 0usize, 0usize, 0usize, 0f64, 0f64);
        // rerun-tape (AD_TAPE_PREDICT=1): this step's share decisions come from the same step of the gradient before
        // (the book file) and are checked at the chunk end (deriv::predict). A miss returns predict::MISS before anything of the step is copied
        // out or walked back; reverse_step then tapes the step again from its restored start with every decision read.
        let predicting = deriv::predict::on();
        if predicting { deriv::predict::begin(step, true); }
        loop {
            match self.tape_range(ex, base, frame, step, flat, lo, n, true)? {
                Taping::Done(rows) => {
                    if predicting && deriv::predict::active() {
                        let bad = deriv::predict::verify(&ex.cuda)?;
                        if !bad.is_empty() {
                            self.defer = defer0.0; self.defer_step = defer0.1;
                            deriv::predict::end_step(step, false)?;
                            eprintln!("{}", json!({"event":"window_tape_predict_miss","step":step,"keys":bad.len(),"chunk":chunks.len(),
                                                   "tape_elapsed_s":t_tape.elapsed().as_secs_f64(),"predict":deriv::predict::receipt()}));
                            return Err(deriv::predict::MISS.to_owned());
                        }
                    }
                    let used = self.tape.used;
                    tape_sum += used; tape_peak = tape_peak.max(used);
                    // s60-cardmem: the step's whole tape (its chunks' sum), for AD_CARD_ROOM_FILE
                    crate::card_room::note_tape_step(tape_sum as u64); crate::card_room::save();
                    eprintln!("{}", json!({"event":"window_tape_chunk","step":step,"chunk":chunks.len(),"commands":[lo, n],"rows":rows.len(),"used_bytes":used,
                                           "stays_in_arena":true,"tape_elapsed_s":t_tape.elapsed().as_secs_f64()}));
                    chunks.push(Chunk { rows, offs: std::mem::take(&mut self.seg_offs), used, runs: None });
                    if total_est > 0 { self.pack_ratio = self.pack_ratio.max(tape_sum as f64 / total_est as f64); }
                    self.st_trim(step);
                    break;
                },
                Taping::Full(i, rows) => {
                    if i == lo { return Err(format!("step {step}: command {i} alone does not fit the tape ({} bytes); raise --tape-gib", self.tape.bytes)); }
                    // rerun-tape: the chunk's decisions are checked before it is copied out; the rest of the step
                    // reads its decisions for real (a later miss could not be undone once this chunk is in host memory)
                    if predicting && deriv::predict::active() {
                        let bad = deriv::predict::verify(&ex.cuda)?;
                        if !bad.is_empty() {
                            self.defer = defer0.0; self.defer_step = defer0.1;
                            deriv::predict::end_step(step, false)?;
                            eprintln!("{}", json!({"event":"window_tape_predict_miss","step":step,"keys":bad.len(),"chunk":chunks.len(),
                                                   "tape_elapsed_s":t_tape.elapsed().as_secs_f64(),"predict":deriv::predict::receipt()}));
                            return Err(deriv::predict::MISS.to_owned());
                        }
                    }
                    if predicting { deriv::predict::deactivate(); }
                    let used = self.tape.used;
                    tape_sum += used; tape_peak = tape_peak.max(used);
                    let tf = Instant::now();
                    let (runs, waited, sent, sh, nb, bad) = self.st_flush(&ex.cuda, used, &mut log, &mut log_off, &mut share)?;
                    flush_s += tf.elapsed().as_secs_f64(); buf_wait_s += waited; d2h_bytes += sent; shared_bytes += sh; share_blocks += nb; share_bad += bad;
                    eprintln!("{}", json!({"event":"window_tape_chunk","step":step,"chunk":chunks.len(),"commands":[lo, i],"rows":rows.len(),"used_bytes":used,
                                           "copied_out_bytes":sent,"shared_bytes":sh,"blocks_digested":nb,"host_runs":runs.len(),
                                           "flush_issue_s":tf.elapsed().as_secs_f64(),"buffer_wait_s":waited,"tape_elapsed_s":t_tape.elapsed().as_secs_f64()}));
                    chunks.push(Chunk { rows, offs: std::mem::take(&mut self.seg_offs), used, runs: Some(runs) });
                    // speed tape60: how many more host buffers this step will take, from the commands still to tape (their
                    // tape estimates times what the taped ones took of theirs; the last chunk stays in the arena). THE
                    // BREAKAGE THIS REMOVES: the helper kept ST_AHEAD buffers made ahead to the end of the first reversed
                    // step, where the page-locked memory is at its peak: 2 x 7 GiB that no chunk ever used.
                    if let Some(pool) = self.st_ahead.as_ref().filter(|_| st_trim_on()) {
                        let est_done: usize = flat[..i].iter().map(|x| x.est).sum();
                        // the first streamed step of the run has no pool from a step before: one buffer is always ready,
                        // and the helper may make all the step is still expected to take
                        let (floor, cap) = if self.st_info.is_empty() { (1, usize::MAX) } else { (0, ST_AHEAD) };
                        let more = if total_est <= est_done { floor } else if est_done == 0 || tape_sum == 0 { ST_AHEAD } else {
                            let to_come = (total_est - est_done) as f64 * (tape_sum as f64 / est_done as f64);
                            let leave = ((to_come / self.tape.bytes as f64).ceil() - 1.0).max(0.0) * self.tape.bytes as f64 * (d2h_bytes as f64 / tape_sum as f64);
                            let room = log.last().map_or(0, |b| b.len() - log_off) as f64;
                            // less what the pool of the step before already holds
                            (((leave - room).max(0.0) / self.tape.bytes as f64).ceil() as usize).saturating_sub(self.st_pool.len()).clamp(floor, cap)
                        };
                        let (l, cv) = &**pool;
                        if let Ok(_g) = l.lock() { self.st_want.store(more, std::sync::atomic::Ordering::SeqCst); }
                        cv.notify_all();
                    }
                    lo = i;
                },
                Taping::Overflow(k) => return Err(format!("DEFECT streamed tape: step {step} command {k} overflowed outside the guard")),
            }
        }
        self.defer = defer0.0; self.defer_step = defer0.1;
        if predicting {
            deriv::predict::end_step(step, true)?;
            eprintln!("{}", json!({"event":"window_tape_predict","step":step,"chunks":chunks.len(),"tape_elapsed_s":t_tape.elapsed().as_secs_f64(),"predict":deriv::predict::receipt()}));
        }
        let tape_s = t_tape.elapsed().as_secs_f64();
        // bs-tape (AD_G5_PROF only): the restore and re-run's host timers; host_prof below then holds the walk-back's
        let prof_tape: Vec<Value> = if dev::prof::on() { dev::prof::take().into_iter().filter(|x| x.2 > 0).map(|(k, s, c)| json!([k, s, c])).collect() } else { Vec::new() };
        let nchunks = chunks.len();
        if total_est > 0 { self.pack_ratio = self.pack_ratio.max(tape_sum as f64 / total_est as f64); }
        // The redo check (the end state word for word against the stored boundary) runs beside the walk-back, as the
        // one-pass path makes it: the state is downloaded first, the walk-back writes only derivative and tape memory.
        let t_dl = Instant::now();
        // bs-tape (a12 backstep): the end state goes down on a copy stream of its own while the walk-back runs (it waits on
        // the device for the re-run's last command), and the compare thread waits for that copy before it compares. THE
        // WASTE THIS REMOVES: 10.5 GB downloaded with the card idle before every walk-back (0.20 s a 3 km step). The walk-back
        // writes only derivative and tape memory (the reason the compare could already run beside it); a word it did
        // write would make the compare refuse the step, never change an answer. AD_REDO_ASYNC=0 downloads first, as before.
        let redo_async = std::env::var("AD_REDO_ASYNC").as_deref() != Ok("0") && ordered(ex) && !(self.cs_on && self.cs_truth.is_some()) && !dead_poison_mode().1;
        // 60 km tape: the content store's redo check beside the walk-back (CsBeside; AD_CS_REDO_BESIDE=0: before it, as before)
        let cs_beside = std::env::var("AD_CS_REDO_BESIDE").as_deref() != Ok("0") && ordered(ex) && self.cs_on && self.cs_truth.is_some() && !dead_poison_mode().1;
        let mut csb: Option<CsBeside> = if cs_beside { Some(self.cs_beside_prep(ex, cur)?) } else { None };
        let (live, redo_ev) = if cs_beside { (match self.anchor_pool.pop() { Some(p) => p, None => crate::pinned::PinnedBuf::new(&ex.cuda, 4096)? }, 0usize) }
                              else if redo_async { let (p, e) = self.redo_download_async(ex, cur)?; (p, e as usize) } else { (self.redo_download(ex, cur)?, 0usize) };
        dead_poison_rev(ex, step, "walk")?;
        let cidx = self.redo_index(cur, step);
        let ranges: Vec<Vec<(usize, usize)>> = redo_ranges(ex, cur, &cidx, step);
        let dl_s = t_dl.elapsed().as_secs_f64();
        crate::spill::phase(&format!("ad{step}:walk"));
        let cuda = &ex.cuda as *const Cuda;
        let t_w = Instant::now();
        let mut h2d_bytes = 0usize;
        let mut uploads: Vec<Value> = Vec::new();
        let mut stalls: Vec<(usize, usize, usize)> = Vec::new();
        let mut walks: Vec<(usize, usize, usize)> = Vec::new();
        let (ev_sync, ctx_set, ctx) = (ex.cuda.event_synchronize, ex.cuda.ctx_set_current, ex.context as usize);
        let (cmp, walked, walk_s) = std::thread::scope(|sc| {
            let (index, live_ref, ranges_ref, cs_ref) = (&cidx, &live[..], &ranges, self.cs_redo_res.take());
            let csb_ref = csb.as_mut();
            let h = sc.spawn(move || {
                if let Some(c) = csb_ref {
                    return match cs_beside_run(c) { Ok(x) => x, Err(e) => { c.err = Some(e); (u64::MAX, 0) } };
                }
                if redo_ev != 0 {
                    if ctx_set(ctx as *mut std::ffi::c_void) != 0 || ev_sync(redo_ev as *mut std::ffi::c_void) != 0 { return (u64::MAX, 0); }
                }
                compare_stored(cs_ref, index, live_ref, cur_data, ranges_ref, step)
            });
            let w = (|| -> Result<(), String> {
                while let Some(mut c) = chunks.pop() {
                    let ci = chunks.len();
                    // this chunk is in the arena already (the last one taped), or was uploaded during the walk-back of
                    // the chunk after it: the dispatch stream waits on the device for its pieces (no host wait).
                    match (self.fz_pf.take(), c.runs.is_some()) {
                        (Some(pf), true) => {
                            if !pf.pending.is_empty() || pf.ci != ci { return Err(format!("DEFECT streamed tape: step {step} chunk {ci}: upload incomplete or of chunk {}", pf.ci)); }
                            // 60 km tape: how long the card's walk-back waits for this chunk's upload (two timing events on the
                            // legacy stream around the waits; read after the step). Measurement only.
                            let (mut s0, mut s1) = (std::ptr::null_mut(), std::ptr::null_mut());
                            let timed = ((*cuda).event_create)(&mut s0, 0) == 0 && ((*cuda).event_create)(&mut s1, 0) == 0;
                            if timed { checked("cuEventRecord upload stall start", ((*cuda).event_record)(s0, std::ptr::null_mut()))?; }
                            for &ev in &pf.done { checked("cuStreamWaitEvent streamed tape upload", ((*cuda).stream_wait_event)(std::ptr::null_mut(), ev, 0))?; }
                            if timed { checked("cuEventRecord upload stall end", ((*cuda).event_record)(s1, std::ptr::null_mut()))?; stalls.push((ci, s0 as usize, s1 as usize)); }
                            h2d_bytes += pf.bytes;
                            uploads.push(json!({"chunk":ci,"bytes":pf.bytes,"issued_during_walk":pf.issued_during_walk,"pieces":pf.done.len()}));
                        },
                        (None, false) => {},
                        _ => return Err(format!("DEFECT streamed tape: step {step} chunk {ci} is neither in the arena nor uploaded")),
                    }
                    // the chunk before this one goes up while this one is walked back
                    if let Some(p) = chunks.last() {
                        let runs = p.runs.clone().ok_or("DEFECT streamed tape: an earlier chunk has no host copy")?;
                        let piece = std::env::var("AD_FUSE_PIECE_MIB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(256).max(1) << 20;
                        let mut pending = Vec::new();
                        let mut a = 0usize;
                        while a < p.used { let k = piece.min(p.used - a); pending.push((a, k)); a += k; }
                        self.fz_pf = Some(FzPrefetch { ci: ci - 1, base: self.tape.base, host: std::ptr::null(), offs: std::mem::take(&mut c.offs), pending,
                                                       done: Vec::new(), stream: self.fz_copy, issued_during_walk: 0, bytes: p.used, runs: Some(runs) });
                    }
                    let (tc, nrows) = (Instant::now(), c.rows.len());
                    // 60 km tape: the card's time for this chunk's walk-back (timing events around it; measurement only)
                    let (mut w0, mut w1) = (std::ptr::null_mut(), std::ptr::null_mut());
                    let wtimed = ((*cuda).event_create)(&mut w0, 0) == 0 && ((*cuda).event_create)(&mut w1, 0) == 0;
                    if wtimed { checked("cuEventRecord chunk walk start", ((*cuda).event_record)(w0, std::ptr::null_mut()))?; }
                    self.walk_back(&*cuda, ex, step, std::mem::take(&mut c.rows))?;
                    if wtimed { checked("cuEventRecord chunk walk end", ((*cuda).event_record)(w1, std::ptr::null_mut()))?; walks.push((ci, w0 as usize, w1 as usize)); }
                    if self.fz_pf.is_some() { self.fz_pf_issue(&*cuda, 0, true)?; }
                    eprintln!("{}", json!({"event":"window_tape_chunk_walked","step":step,"chunk":ci,"rows":nrows,"walk_s":tc.elapsed().as_secs_f64(),
                                           "walk_elapsed_s":t_w.elapsed().as_secs_f64()}));
                }
                Ok(())
            })();
            let walk_s = t_w.elapsed().as_secs_f64();
            (h.join(), w, walk_s)
        });
        let wait_s = t_w.elapsed().as_secs_f64() - walk_s;
        if redo_ev != 0 { checked("cuEventSynchronize redo", (ex.cuda.event_synchronize)(redo_ev as *mut std::ffi::c_void))?; }
        self.anchor_pool.push(live);
        let (d, r) = cmp.map_err(|_| "redo compare thread panicked".to_string())?;
        // 60 km tape: the beside check's buffer, events and times (cs_redo's counters: windows downloading, the check)
        let mut beside_s = Value::Null;
        if let Some(c) = csb.take() {
            for e in c.evs { let _ = (ex.cuda.event_destroy)(e as *mut std::ffi::c_void); }
            if let Some(e) = c.err { return Err(format!("step {step}: the redo check beside the walk-back failed: {e}")); }
            self.cs_n[5] += 1; self.cs_s[5] += c.all_s - c.dl_s; self.cs_s[0] += c.dl_s;
            beside_s = json!({"download_wait_s":c.dl_s,"check_s":c.all_s,"windows":c.wins.len()});
            self.cs_pin = Some(c.pin);
        }
        if d == u64::MAX && redo_ev != 0 { return Err(format!("step {step}: the redo compare thread could not wait for the end state's download")); }
        if d != 0 { return Err(format!("step {step}: restore + redo differs from the stored boundary in {d} words")); }
        walked?;
        // bs-tape: a gather the walk-back did not ask for (none expected) is joined here, so no build outlives its step
        if !self.pending.is_empty() { self.join_pending(&ex.cuda, step)?; }
        self.rv_restore_s = restore_s; self.rv_tape_s = tape_s; self.rv_redo_dl_s = dl_s; self.rv_walk_s = walk_s; self.rv_redo_wait_s = wait_s;
        let redo = json!({"differing_words": d, "residue_words": r, "anchors": 0, "one_pass": true, "beside_walk": true, "content_store_beside": beside_s});
        self.step_flush(&ex.cuda, step, "ad")?;
        drain_pinned(ex)?;
        // 60 km tape: the card's waits for the chunk uploads, by chunk (ms from the two timing events), then the events go
        let mut stall_ms: Vec<Value> = Vec::new();
        let mut stall_s = 0f64;
        for (ci, s0, s1) in stalls.drain(..) {
            let mut ms = 0f32;
            if (ex.cuda.event_elapsed)(&mut ms, s0 as *mut std::ffi::c_void, s1 as *mut std::ffi::c_void) == 0 { stall_s += ms as f64 / 1e3; stall_ms.push(json!([ci, ms])); }
            let _ = (ex.cuda.event_destroy)(s0 as *mut std::ffi::c_void); let _ = (ex.cuda.event_destroy)(s1 as *mut std::ffi::c_void);
        }
        let mut walk_card_ms: Vec<Value> = Vec::new();
        for (ci, w0, w1) in walks.drain(..) {
            let mut ms = 0f32;
            if (ex.cuda.event_elapsed)(&mut ms, w0 as *mut std::ffi::c_void, w1 as *mut std::ffi::c_void) == 0 { walk_card_ms.push(json!([ci, ms])); }
            let _ = (ex.cuda.event_destroy)(w0 as *mut std::ffi::c_void); let _ = (ex.cuda.event_destroy)(w1 as *mut std::ffi::c_void);
        }
        // every copy-out and upload of this step is complete: the fences and their events are spent
        let fence_waits = self.tape.fence.as_ref().map(|f| f.waited).unwrap_or(0);
        self.tape.fence = None;
        if !self.fuse { for ev in self.fz_evs.drain(..) { let _ = (ex.cuda.event_destroy)(ev); } }
        // speed tape60: this step's host buffers go back to the pool for the next step (nothing reads them any more)
        let host_buffers = log.len();
        // bs-tape: borrowed checkpoint buffers go back to the sweep's pool; the step's own ones stay for the next step
        for b in log {
            if let Some(i) = self.st_borrowed.iter().position(|&p| p == b.as_ptr() as usize) {
                self.st_borrowed.swap_remove(i);
                capi::pin_pool_give(bs_pool(), b.into_mem());
            } else { self.st_pool.push(b); }
        }
        let mut fl = [0i32];
        checked("cuMemcpyDtoH failure", (ex.cuda.download)(fl.as_mut_ptr().cast(), self.fail, 4))?;
        if fl[0] != 0 { return Err(format!("step {step}: P8 row transpose saw a non-finite cotangent")); }
        let host_prof: Vec<Value> = dev::prof::take().into_iter().filter(|x| x.2 > 0).map(|(k, s, c)| json!([k, s, c])).collect();
        let pack = self.pack_json();
        let rrows = self.take_rev_rows(0);
        let wk = self.take_wk();
        let roll = self.native.as_mut().map(|n| n.roll_take()).unwrap_or([0; 4]);
        self.st_info.push(json!({"step":step,"chunks":nchunks,"tape_sum_bytes":tape_sum,"d2h_bytes":d2h_bytes,"h2d_bytes":h2d_bytes,"buffer_wait_s":buf_wait_s,
                                 "host_shared":{"on":st_share_on(),"shared_bytes":shared_bytes,"blocks_digested":share_blocks,"host_buffers":host_buffers,
                                                "proof_mode":st_share_verify(),"proof_mismatches":share_bad},
                                 "native_roll":{"spans_on_a_rolling_copy":roll[0],"their_bytes":roll[1],"changed_bytes_taped":roll[2],"first_copy_bytes":roll[3]}}));
        if crate::spill::on() { eprintln!("{}", json!({"event":"window_alloc_census","at":format!("after reverse step {step}"),"census":crate::spill::census(40)})); }
        Ok(json!({"step":step,"commands":n,"segments":nchunks,"streamed":true,"overflow_splits":0,"commands_replayed":n,"host_prof":host_prof,"host_prof_before_step":prof_before,"host_prof_restore_and_tape":prof_tape,
                  "pin_made":crate::pinned::pin_sites_take().into_iter().map(|(k, c, b, s)| json!([k, c, b, s])).collect::<Vec<_>>(),"pack":pack,"tape_rows_packed":rrows,
                  "stages_s":{"restore":restore_s,"replay_and_tape":tape_s,"chunk_flush_in_tape":flush_s,"chunk_buffer_wait_in_tape":buf_wait_s,"redo_download":dl_s,
                              "walk_back":walk_s,"redo_compare_wait_after_walk":wait_s},"walk_parts_s":wk,
                  "tape_d2h_bytes":d2h_bytes,"tape_h2d_bytes":h2d_bytes,"chunk_uploads":uploads,"fence_waits":fence_waits,"cold":crate::spill::cold_stats(),
                  "upload_stall_s":stall_s,"upload_stall_ms_by_chunk":stall_ms,"walk_card_ms_by_chunk":walk_card_ms,
                  "tape_peak_bytes":tape_peak,"tape_sum_bytes":tape_sum,"adjoint_entry_s":(self.entry_ms - ms0) / 1e3,"pass1_s":0.0,
                  "tape_estimate_bytes":total_est,"wall_s":t0.elapsed().as_secs_f64(),"redo":redo}))
    }
    /// machine: the walk-back parts since the last call (see wk_s), and the counters zeroed.
    fn take_wk(&mut self) -> Value {
        let (s, n) = (std::mem::take(&mut self.wk_s), std::mem::take(&mut self.wk_n));
        json!({"entries":s[0],"rules":s[1],"native":s[2],"tempo":s[3],"drain_before_row":s[4],"tape_upload_issue":s[5],"drain_chunk_end":s[6],
               "rows":{"entries":n[0],"rules":n[1],"native":n[2],"tempo":n[3]},
               "bs":if bsprof::level() > 0 { bsprof::take() } else { Value::Null },
               "note":"host wall; launches are asynchronous, card time lands in the drains or in a row that blocks"})
    }
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
        self.walk_back_records(cuda, ex, step, tape)
    }

    /// Admission boundary for a live tape, in forward command order. Validation
    /// finishes before the first cotangent store; unknown rules never act as identity.
    /// Safety: all saved spans and borrowed owners must outlive this synchronous
    /// walk, and records must have been completed by live_after on the same stream.
    pub(crate) unsafe fn live_walk_back(&mut self, cuda: &Cuda, ex: &Execution,
        step: usize, tape: Vec<Tp>) -> Result<(), String> {
        for t in &tape {
            match t {
                Tp::Entry { f, .. } if !self.kernels.contains_key(f) => {
                    return Err(format!("live reverse: no derivative entry for {f}"));
                },
                Tp::Rule { r: R::Refused(reason), .. } => {
                    return Err(format!("live reverse: unadmitted rule: {reason}"));
                },
                Tp::Rule { r: R::Entry | R::Native(_) | R::Tempo(_) | R::TempoSerial(_) | R::TempoSmall(_), .. } => {
                    return Err("live reverse: derivative entry stored as a generic rule".into());
                },
                _ => {},
            }
        }
        self.walk_back_records(cuda, ex, step, tape)
    }

    /// The recorded and live routes share every reverse operation below. No
    /// forward command is executed here; records already contain pre/post saves.
    /// Safety: tape admission, span lifetimes and stream dependencies are supplied
    /// by the producer. Live callers use live_walk_back to reject unknown rules.
    pub(crate) unsafe fn walk_back_records(&mut self, cuda: &Cuda, ex: &Execution,
        step: usize, tape: Vec<Tp>) -> Result<(), String> {
        self.graph_tape_epoch = self.graph_tape_epoch.checked_add(1).ok_or("DEFECT graph tape epoch overflow")?;
        let ntape = tape.len();
        let light = ordered(ex);
        let bsl = bsprof::level();
        let mut tape_it = tape.into_iter().enumerate().rev().peekable();
        while let Some((j, t)) = tape_it.next() {
            let nodrain = light && native_nodrain() && matches!(&t, Tp::Native { f, .. } if plant().as_ref().is_none_or(|(pk, _, _)| pk.as_str() != f.as_str()));
            if self.sync_owed && !matches!(t, Tp::Entry { .. }) && !(light && matches!(t, Tp::Rule { .. })) && !nodrain { let tw = Instant::now(); dev::sync(cuda)?; self.sync_owed = false; self.wk_s[4] += tw.elapsed().as_secs_f64(); }
            let (wk_k, wk_t) = (match &t { Tp::Entry { .. } => 0usize, Tp::Rule { .. } => 1, Tp::Native { .. } => 2, _ => 3 }, Instant::now());
            let probe_cmd = crate::word_probe::cmd_on("reverse", step, j);
            let digest_cmd = crate::rev_digest::cmd_step(step);
            let label = if self.trace || probe_cmd || digest_cmd { match &t { Tp::Entry { f, .. } | Tp::Native { f, .. } | Tp::Tempo { f, .. } | Tp::TempoSerial { f, .. } | Tp::TempoSmall { f, .. } => f.clone(), Tp::Rule { r, .. } => format!("{r:?}").chars().take(160).collect() } } else { String::new() };
            match t {
                Tp::Entry { f, mut b, args, grid, block, shared, post } => {
                    if bsl > 1 { let mut tp = Instant::now(); dev::sync(cuda)?; bsprof::lap(12, &mut tp); }
                    let t0 = Instant::now();
                    let mut bt = t0;
                    let kd = &self.kernels[&f];
                    let li = deriv::LaunchInfo { args: &args, grid, block, shared };
                    let mut live = ex.allocations.clone();
                    for (id, a) in &post { live.insert(*id, *a); }
                    if bsl > 0 { bsprof::lap(0, &mut bt); }
                    self.graph_scratch_epoch = self.graph_scratch_epoch.checked_add(1).ok_or("DEFECT graph scratch epoch overflow")?;
                    self.scratch.reset();
                    b.workspace.clear();
                    kd.bind_transient(cuda, &mut self.scratch, &mut b, &li).map_err(|e| format!("step {step} {f} transient bind: {e}"))?;
                    kd.bind_recon(cuda, &self.support, &mut self.scratch, &mut b).map_err(|e| format!("step {step} {f} block rebuild: {e}"))?;
                    if bsl > 0 { bsprof::lap(1, &mut bt); }
                    if kd.aug { kd.launch(cuda, &mut self.scratch, &mut b, &li, &live, "aug")?; }
                    if bsl > 0 { bsprof::lap(2, &mut bt); if kd.aug { bsprof::launches(1); } }
                    for c in b.ad_cells.values() { dev::zero(cuda, c.0, 8)?; }
                    kd.snapshot_seeds(cuda, &b, &li)?;
                    if bsl > 0 { bsprof::lap(3, &mut bt); }
                    // inst (AD_PLANT): this launch's cotangent view of the named argument before its adjoint entries.
                    let planted: Option<(u64, Vec<u8>, f64)> = match plant() {
                        Some((pk, pa, pf)) if *pk == f => {
                            let name = match pa.parse::<usize>() { Ok(i) => kd.names.get(i).cloned().ok_or_else(|| format!("AD_PLANT: {f} has no argument {i}"))?, Err(_) => pa.clone() };
                            let (a, n) = kd.view(&b, &li, &name).map_err(|e| format!("AD_PLANT: {f} argument {name}: {e}"))?;
                            let wd = b.shadows.values().find(|s| a >= s.address && a < s.address + s.bytes as u64).map(|s| s.width).unwrap_or(0);
                            if wd != 4 { return Err(format!("AD_PLANT: {f} argument {name} has a {wd}-byte cotangent view; the planted error wants a float32 one")); }
                            dev::sync(cuda)?;
                            Some((a, dev::download(cuda, a, n)?, *pf)) },
                        _ => None };
                    // AD_SUBNORMAL=f1,f2 (diagnostic only): the argument shadows' census before and after this adjoint.
                    let census = std::env::var("AD_SUBNORMAL").ok().is_some_and(|v| v.split(',').any(|x| x == f));
                    let before = if census { dev::sync(cuda)?; kd.shadow_census(cuda, &b, &li) } else { Vec::new() };
                    { let t = dev::prof::start(); kd.begin_ad(cuda, &self.support, &mut b, &li)?; dev::prof::add(1, t); }
                    if bsl > 0 { bsprof::lap(4, &mut bt); }
                    // AD_ZERO_DERIV (bisection diagnostic): the transpose of "outputs constant" consumes the seeds and adds
                    // nothing to the inputs, so the adjoint entries are skipped and the inout seed views zeroed.
                    let zero_d = std::env::var("AD_ZERO_DERIV").ok().is_some_and(|v| v.split(',').any(|x| x == f));
                    if zero_d { dev::sync(cuda)?; kd.zero_written(cuda, &b, &li, true)?; }
                    else if self.graph_segment && f == "w_edge_flux_f32" {
                        // Existing host decisions bind this invocation before any graph update.
                        // The cached executable retains topology only. No old pointer frame is
                        // usable after a scratch reset or tape/replay generation boundary.
                        let t = dev::prof::start();
                        let scratch_before = self.scratch.used;
                        let entries: Vec<&str> = kd.ad_order.iter().filter(|e| !scalar_skip(kd, e)).map(String::as_str).collect();
                        let (derived_tables, derived_bytes) = kd.prepare_graph_tables(cuda, &b, &li, &live)?;
                        let commands = kd.lower_graph_ad_entries(cuda, &mut self.scratch, &mut b, &li, &live, &entries)?;
                        let kernels = commands.iter().filter(|c| matches!(c, submit::Command::Kernel { .. })).count();
                        let clears = commands.iter().filter(|c| matches!(c, submit::Command::Clear { .. })).count();
                        let allocations = [submit::Allocation { id: 0, address: self.scratch.base, bytes: self.scratch.bytes }];
                        let batch = submit::BoundBatch::new(&allocations, &commands, &[], crate::hex_step_validate)
                            .map_err(|e| format!("DEFECT graph {f} bind: {e:?}"))?;
                        let epoch = crate::reverse_machinery::BindingEpoch { tape: self.graph_tape_epoch,
                            scratch: self.graph_scratch_epoch, scratch_base: self.scratch.base, scratch_bytes: self.scratch.bytes };
                        let run = self.segment_graphs.run_bound(cuda, ex.context, &f, &batch, epoch)?;
                        eprintln!("{}", json!({"event":"window_graph_segment", "pass":"ad", "step":step,
                            "command":j, "function":f, "entry_order":entries, "graph_nodes":run.nodes,
                            "kernel_nodes":kernels, "clear_nodes":clears, "graph_launches":1,
                            "capture_updates":0, "captures":0, "graph_builds":run.builds,
                            "graph_parameter_updates":usize::from(run.node_updates != 0), "node_parameter_updates":run.node_updates,
                            "prepare_host_waits":0, "device_event_waits":2,
                            "derived_preparation_tables":derived_tables, "derived_preparation_bytes":derived_bytes,
                            "derived_preparation_host_boundary":derived_tables != 0, "replay_host_waits":0,
                            "replay_entry_launches":0, "scratch_added_bytes":self.scratch.used-scratch_before,
                            "scratch_capacity_bytes":self.scratch.bytes, "scratch_base":self.scratch.base,
                            "tape_epoch":epoch.tape, "scratch_epoch":epoch.scratch,
                            "lifetime":"fresh complete binding per invocation; revmach-c producer/consumer fences",
                            "outside_segment":"binding, begin_ad, finish_ad, failure ledger and all other physics remain enabled"}));
                        dev::prof::add(5, t);
                    }
                    else { let t = dev::prof::start(); for e in kd.ad_order.iter().filter(|e| !scalar_skip(kd, e)) { kd.launch(cuda, &mut self.scratch, &mut b, &li, &live, e)?; if bsl > 0 { bsprof::launches(1); } } dev::prof::add(5, t); }
                    if bsl > 0 { bsprof::lap(5, &mut bt); }
                    { let t = dev::prof::start(); kd.finish_ad(cuda, &mut b, &li, Some(&self.support))?; dev::prof::add(1, t); }
                    if bsl > 0 { bsprof::lap(6, &mut bt); }
                    if zero_d { dev::sync(cuda)?; kd.zero_written(cuda, &b, &li, true)?; }
                    if let Some((a, before, pf)) = planted {
                        dev::sync(cuda)?;
                        let mut after = dev::download(cuda, a, before.len())?;
                        let ch = plant_scale(&mut after, &before, pf);
                        dev::upload(cuda, a, &after)?;
                        if PLANT_HITS[0].load(std::sync::atomic::Ordering::Relaxed) == 0 { eprintln!("{}", json!({"event":"window_planted","step":step,"function":f,"words":before.len() / 4,"words_changed":ch})); }
                        PLANT_HITS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed); PLANT_HITS[1].fetch_add(ch, std::sync::atomic::Ordering::Relaxed);
                    }
                    if census { dev::sync(cuda)?; eprintln!("{}", json!({"event":"window_ad_census","step":step,"command":j,"function":f,
                        "fields":"[arg, words, nonzero, subnormal, min nonzero |b|, max |b|]","before":before,"after":kd.shadow_census(cuda, &b, &li)})); }
                    let t = dev::prof::start();
                    kd.queue_failures(cuda, &mut b, self.ledger.as_mut().ok_or("step ledger missing")?, &format!("step {step} {f} adjoint"))?;
                    dev::prof::add(7, t);
                    let (bs_host, mut bs_card) = (if bsl > 0 { bsprof::lap(7, &mut bt); t0.elapsed().as_secs_f64() } else { 0.0 }, 0.0);
                    if bsl > 1 { dev::sync(cuda)?; bs_card = bsprof::lap(9, &mut bt); }
                    if entry_sync() { dev::sync(cuda)?; } else { self.sync_owed = true; }
                    self.note("ad", &f, t0);
                    if bsl > 0 { bsprof::lap(8, &mut bt); bsprof::add_fn(&f, bs_host, bs_card); }
                },
                Tp::Rule { r, prim } => {
                    if bsl > 1 { let mut tp = Instant::now(); dev::sync(cuda)?; bsprof::lap(13, &mut tp); }
                    let mut bt = Instant::now();
                    let t = dev::prof::start(); self.ad(cuda, &r, prim)?; dev::prof::add(11, t);
                    if bsl > 0 {
                        let h = bsprof::lap(10, &mut bt);
                        let c = if bsl > 1 { dev::sync(cuda)?; bsprof::lap(11, &mut bt) } else { 0.0 };
                        bsprof::add_fn(bsprof::rkind(&r), h, c);
                    }
                },
                Tp::Tempo { f, snap } => {
                    let t0 = Instant::now();
                    crate::spill::phase(&format!("ad{step}:{f}"));
                    self.tempo_apply(ex, &snap, step, true).map_err(|e| format!("step {step} {f} TEMPO adjoint: {e}"))?;
                    crate::spill::phase(&format!("ad{step}:walk"));
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
                    // a12 bs-mynn (P3 fuse, opt-in AD_P3_FUSE=1; see Native::p3_fuse_check): this MYNN-EDMF launch and the one
                    // taped just before it run their adjoint as one launch. This launch's undo goes first, so the rolling
                    // copy holds the earlier launch's pre-launch words, then the earlier launch's row runs over both ranges.
                    if std::env::var("AD_P3_FUSE").as_deref() == Ok("1") && f == "rrfs_mynnedmf_run_l16" {
                        let fuse_n = match (self.native.as_ref(), tape_it.peek()) {
                            (Some(nn), Some((_, Tp::Native { f: f2, p: lower }))) if *f2 == f => nn.p3_fuse_check(lower, &p),
                            _ => Err("no MYNN-EDMF launch taped just before".into()),
                        };
                        match fuse_n {
                            Ok(n) => {
                                let Some((_, Tp::Native { p: mut lower, .. })) = tape_it.next() else { return Err("DEFECT P3 fuse: peeked entry vanished".into()) };
                                self.native.as_ref().ok_or("native rows not loaded")?.p3_fuse_apply(&mut lower, n)?;
                                for &(dst, tab, rows) in &p.undo { self.support.runs_copy(cuda, dst, tab, rows)?; }
                                eprintln!("{}", json!({"event":"window_p3_fused","step":step,"columns":n}));
                                p = lower;
                            },
                            Err(why) => { eprintln!("{}", json!({"event":"window_p3_not_fused","step":step,"why":why})); },
                        }
                    }
                    crate::spill::phase(&format!("ad{step}:{f}"));
                    let ctag = self.native.as_ref().and_then(|n| n.row(&f)).map(|r| r.kind.split('_').next().unwrap_or("native").to_owned()).unwrap_or_else(|| "native".into());
                    self.scratch.reset();
                    self.native.as_mut().ok_or("native rows not loaded")?.bind_transient(cuda, &mut self.scratch, &ex.allocations, &mut p)
                        .map_err(|e| format!("step {step} {f} native transient bind: {e}"))?;
                    let segs = p.active_segments();
                    // inst (AD_PLANT on a native scheme): the argument is a cotangent segment number, or "auto" = the first
                    // segment the reverse changed and did not consume (zero).
                    let planted: Option<(Vec<(usize, u64, Vec<u8>)>, f64)> = match plant() {
                        Some((pk, pa, pf)) if *pk == f => {
                            dev::sync(cuda)?;
                            let mut v = Vec::new();
                            for (i, (al, lo, hi, _)) in segs.iter().enumerate() {
                                if pa != "auto" && pa.parse::<usize>().ok() != Some(i) { continue; }
                                let Some(e) = self.base_el.get(al) else { continue };
                                let a = self.sh(*e + (*lo / 4) as i64);
                                v.push((i, a, dev::download(cuda, a, hi - lo)?));
                            }
                            if v.is_empty() { return Err(format!("AD_PLANT: native {f} has no cotangent segment {pa} ({} segments)", segs.len())); }
                            Some((v, *pf)) },
                        _ => None };
                    self.xfer(cuda, &segs, true)?;
                    let tn = dev::prof::start();
                    let _cold = crate::spill::cold(&ctag);
                    let nn = self.native.as_mut().ok_or("native rows not loaded")?;
                    nn.ledger = self.ledger.take();
                    nn.p3_step = step as i64;
                    let info = nn.reverse(cuda, &mut self.scratch, &mut p);
                    self.ledger = nn.ledger.take();
                    let info = info.map_err(|e| format!("step {step} {f} native adjoint: {e}"))?;
                    dev::prof::add(8, tn);
                    self.xfer(cuda, &segs, false)?;
                    // drains (AD_NATIVE_NODRAIN=1, see native_nodrain): the drain is owed to the next row that needs one
                    if nodrain && planted.is_none() { self.sync_owed = true; } else { dev::sync(cuda)?; }
                    // speed tape60 (rolling pre-launch copy): this launch's adjoint is complete; the words its bind changed
                    // in the rolling copy go back, so the launch taped before it reads its own pre-launch words. Queued on
                    // the dispatch stream before the next chunk's upload may reuse this row's tape bytes (fz_pf_issue below
                    // orders the upload after an event recorded on this stream).
                    for &(dst, tab, rows) in &p.undo { self.support.runs_copy(cuda, dst, tab, rows)?; }
                    if let Some((v, pf)) = planted {
                        for (i, a, before) in v {
                            let mut after = dev::download(cuda, a, before.len())?;
                            if after == before || after.iter().all(|b| *b == 0) { continue; }
                            let ch = plant_scale(&mut after, &before, pf);
                            dev::upload(cuda, a, &after)?;
                            if PLANT_HITS[0].load(std::sync::atomic::Ordering::Relaxed) == 0 { eprintln!("{}", json!({"event":"window_planted","step":step,"function":f,"segment":i,"segments":segs.len(),"words":before.len() / 4,"words_changed":ch})); }
                            PLANT_HITS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed); PLANT_HITS[1].fetch_add(ch, std::sync::atomic::Ordering::Relaxed);
                            break;
                        }
                    }
                    eprintln!("{}", json!({"event":"window_native","pass":"ad","step":step,"function":f,"ms":t0.elapsed().as_secs_f64() * 1e3,"info":info}));
                    crate::spill::phase(&format!("ad{step}:walk"));
                    self.note("ad", &f, t0);
                },
                Tp::TempoSmall { f, snap, ins, outs } => {
                    let t0 = Instant::now();
                    self.small_sweep(cuda, &f, &snap, &ins, &outs, true, false).map_err(|e| format!("step {step} {f} tempo-small adjoint: {e}"))?;
                    self.note("ad", &f, t0);
                },
            }
            self.wk_s[wk_k] += wk_t.elapsed().as_secs_f64(); self.wk_n[wk_k] += 1;
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
            // hunt-120-start (AD_WORD_PROBE with AD_WORD_PROBE_STEPS): the asked words after this command walked back.
            if probe_cmd { self.word_probe(cuda, "reverse", "command", step, Some((j, label.as_str())), &Value::Null)?; }
            // hunt-60-repeat (AD_REV_DIGEST_CMD=<steps>, with AD_REV_DIGEST=only:<allocations>): the asked allocations after this command.
            if digest_cmd { self.rev_digest(cuda, "command", step, Some((j, &label)), &Value::Null)?; }
            if self.fz_pf.is_some() { let tw = Instant::now(); self.fz_pf_issue(cuda, j, false)?; self.wk_s[5] += tw.elapsed().as_secs_f64(); }
        }
        self.sync_owed = false;
        let tw = Instant::now();
        let r = dev::sync(cuda);
        self.wk_s[6] += tw.elapsed().as_secs_f64();
        r
    }
    /// memory (AD_TL_BIND_SCRATCH=1): room in the scratch for one MYNN-EDMF tangent bind's two regions. The scratch
    /// is re-made at need + 256 MiB + 64 MiB (the size `fit_scratch` gives a `max`-sized reverse scratch, so the reverse
    /// keeps it) when it is smaller; nothing lives in it here (the native row reset it, and the context is drained
    /// first). From then on it carries the overflow chain a fitted reverse scratch has, so a tangent launch that wants
    /// more workspace than is left takes a kept slab instead of refusing.
    unsafe fn tl_scratch_room(&mut self, ex: &Execution, need: usize, step: usize, f: &str) -> Result<(), String> {
        let want = (need + (256 << 20) + (64 << 20)).div_ceil(2 << 20) * (2 << 20);
        if want > self.scratch.bytes {
            dev::sync(&ex.cuda)?;
            let before = self.scratch.bytes;
            if self.scratch_cli == 0 { self.scratch_cli = before; }
            let (peak, resets, spill, fence) = (self.scratch.peak, self.scratch.resets, self.scratch.spill.take(), self.scratch.fence.take());
            checked("cuMemFree scratch", (ex.cuda.free)(self.scratch.base))?;
            self.scratch = match dev::Arena::new(&ex.cuda, want) { Ok(s) => s, Err(e) => { self.scratch = dev::Arena::new(&ex.cuda, before)?;
                return Err(format!("step {step} {f}: AD_TL_BIND_SCRATCH: the scratch needs {want} bytes for the tangent bind, card refused: {e}")) } };
            self.scratch.peak = peak; self.scratch.resets = resets; self.scratch.spill = spill; self.scratch.fence = fence;
            eprintln!("{}", json!({"event":"window_tl_scratch_grown","step":step,"function":f,"need_bytes":need,"before":before,"bytes":want}));
        }
        if self.scratch.spill.is_none() { self.scratch.spill = Some(dev::Spill { alloc: ex.cuda.allocate, slabs: Vec::new(), cur: 0, off: 0, held: 0, peak: 0 }); }
        Ok(())
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
        // an exhausted take names only where that take ends, not the bind's total: ring-60km grows to the need + 1/8 (at
        // least 1/8 over the arena) and lets the bind run again, not 2x (60 km: 7 -> 14 GiB for a 9.19 GB bind, the room the
        // boundary layer's ring could not get; see ring_room::tape_grow_want). AD_TAPE_GROW_2X=1 keeps the 2x rule.
        let want = crate::ring_room::tape_grow_want(need, self.tape.bytes, self.tape_max, std::env::var("AD_TAPE_GROW_2X").as_deref() == Ok("1"));
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
        // x20ck-C (a test machine card 3, 05:48Z): with a 4 GiB floor and the window cache's 1/5 share taken first, the tape (21.9 GB)
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
        // s60-cardmem (AD_CARD_RELEASE=1): what no later step reads goes before the room is read. At 60 km the census
        // before the reverse (six-3 b8-head) holds no such block of 64 MiB or more on the card: the 15.58 GB the
        // tangent pass made are the plan's own allocations 389-441 (carried state) and 462-464 (pool scratch made in
        // steps 2, 3 and 6), which every reversed step's re-run reads. So this frees the retired scheme blocks only.
        if crate::card_room::release_on() {
            dev::sync(&ex.cuda)?;
            let (n, b) = match self.native.as_mut() { Some(nn) => nn.release_retired(&ex.cuda), None => (0, 0) };
            eprintln!("{}", json!({"event":"window_card_release","retired_blocks_freed":n,"retired_bytes_freed":b,
                                   "cold_families":crate::spill::cold_families(),"census":crate::spill::census(40)}));
        }
        // tape-dedupe-and-pack: the reverse scratch reaches its packed size before the card's free memory is read.
        self.fit_scratch(&ex.cuda)?;
        let (mut free, mut total) = (0usize, 0usize);
        checked("cuMemGetInfo", (ex.cuda.mem_info)(&mut free, &mut total))?;
        let before = self.tape.bytes;
        // s60-cardmem (AD_CARD_ROOM=1): the room (card_room::split) is free less the largest cold family, what the reverse
        // allocates after it starts and 1/64 of the card, as measured (an earlier run's, through AD_CARD_ROOM_FILE, or
        // this one's); the boundary layer's ring (the lent TEMPO arena) takes it first, up to the ring that keeps every
        // slot busy, then the tape up to the largest step's tape. Not measured yet: nothing is shared. In place of the
        // fixed 12 GiB reserve below, and of the streamed tape's cap at its own size.
        let lend_on = std::env::var("AD_PHYS_SHARED_ARENA").as_deref() != Ok("0");
        let room_split = if self.pack && crate::card_room::on() {
            let fam = crate::spill::cold_families();
            let k = crate::card_room::known();
            let lent = if lend_on { self.tempo.as_ref().map(|t| t.1.bytes).unwrap_or(0) as u64 } else { 0 };
            let n = crate::card_room::Need { cold_peak: (fam.first().map(|f| f.1).unwrap_or(0) as u64).max(k.cold_peak), rev_growth: k.rev_growth,
                bl_held: lent, bl_full: k.bl_full, bl_nowrap: k.bl_nowrap, tape_held: before as u64, tape_step: k.tape_step };
            Some((crate::card_room::split(free as u64, total as u64, &n), fam, n))
        } else { None };
        // the boundary layer first: the lent TEMPO arena grows by its share (TEMPO is idle here; the ring and TEMPO's tiles
        // are made in it per launch, and the MYNN rows read its address at each launch from LEND)
        let mut bl_grown: Option<(usize, usize)> = None;
        if let (Some((s, _, _)), true) = (&room_split, lend_on) {
            let add = s.bl_add as usize;
            if add > 0 {
                if let Some((_, ar)) = self.tempo.as_mut() {
                    let old = ar.bytes;
                    dev::sync(&ex.cuda)?;
                    checked("cuMemFree tempo arena", (ex.cuda.free)(ar.base))?;
                    *ar = match dev::Arena::new(&ex.cuda, old + add) { Ok(a) => a, Err(_) => dev::Arena::new(&ex.cuda, old)? };
                    *crate::tempo_adw::LEND.lock().map_err(|_| "TEMPO lend lock poisoned")? = (ar.base, ar.bytes);
                    bl_grown = Some((old, ar.bytes));
                }
            }
        }
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
        let want = match &room_split { Some((s, _, _)) => before + s.tape_add as usize, None => want };
        let fits_card = if self.pack { want > before } else { fits_card };
        let fits_cap = fits_cap || room_split.is_some();
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
        if let (Some((s, fam, n)), Some(t)) = (&room_split, self.tape_grown.as_mut()) {
            t["card_room"] = json!({"reserve_bytes":if s.reserve == u64::MAX { Value::Null } else { json!(s.reserve) },"room_bytes":s.room,"bl_target_bytes":s.bl_target,
                "bl_add_bytes":s.bl_add,"lent_ring_bytes":bl_grown.map(|g| json!({"before":g.0,"after":g.1})),"tape_add_bytes":s.tape_add,"tape_after_bytes":s.tape_after,
                "tape_share_of_step":s.tape_share,"left_bytes":s.left,"cold_families":fam,
                "need":{"cold_peak_bytes":n.cold_peak,"rev_growth_bytes":n.rev_growth,"bl_held_bytes":n.bl_held,"bl_full_bytes":n.bl_full,"bl_nowrap_bytes":n.bl_nowrap,
                        "tape_held_bytes":n.tape_held,"tape_step_bytes":n.tape_step}});
        }
        // s60-cardmem: what the reverse allocates from here on is its own growth (AD_CARD_ROOM_FILE)
        crate::card_room::note_rev_start(crate::spill::device_now() as u64);
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

    /// anchors-ahead (a test machine card-need 2a: the first segmented reverse step made every RAM anchor's page-locked buffer
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
        // that step is known and the card fills further (a test machine aa-c120A: pk120's step-1 estimate 62.8 GB, the largest
        // 108 GB; 30.8 GB of room after step 1, an 18.3 GB tape at the end), so one pass is assumed only with 2x headroom
        // on the unpacked estimate. A wrong 'segmented' costs RAM only; a wrong 'one pass' leaves the card idle as before.
        // host-syncs: with the packed tape the reverse goes one pass whenever the tape holds the step's packed estimate
        // (size_tape sizes it to need x AD_TAPE_PACK_GUESS, 0.2), so the guess decides here too, and only once the tape is
        // sized (after the tangent pass). THE BREAKAGE THIS FIXES: the old 0.4 / 2x-headroom guesses predicted
        // 'segmented' on every measured window although every reverse step then ran one pass (pk120 merge-5: 4 anchors
        // x 10.49 GB, pkg18m2: 2 x 3.38 GB, all unused), and page-locking those 42 GB on the anchors-ahead thread during
        // tangent steps 2-3 held the driver lock 0.58-0.89 s at a time, the main thread waiting with the card idle (a test machine
        // nsys bl-hostsync-nsys120). A wrong 'one pass' still runs: the step overflows into the segmented path, which makes
        // its anchors on demand (snap_ram). Segmentation never changes a word.
        let pack_guess = std::env::var("AD_TAPE_PACK_GUESS").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.2).clamp(0.01, 1.0);
        if self.one_pass && self.pack && !tl_done && mode != "1" && std::env::var("AD_ANCHORS_AHEAD_PACKED").as_deref() != Ok("old") { return Ok(()); }
        let segmented = !self.one_pass || if tl_done { (if self.pack { if std::env::var("AD_ANCHORS_AHEAD_PACKED").as_deref() == Ok("old") { (need as f64 * 0.4) as usize } else { (need as f64 * pack_guess) as usize } } else { need }) > self.tape.bytes }
                                          else { need.saturating_mul(2) > room };
        // the anchors of the first reverse step (the window's last) hold the whole live state there: the plan's bound
        // (a test machine aa-c18S8: pkg18m2 holds 2.82 GB after step 1 and 3.38 GB from step 3, so step 1's peak was too small)
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
            None => match self.take_ahead(total) { Some(b) => b, None => match capi::pin_pool_take_largest(bs_pool(), total) {
                // a buffer AD_BS_PREFILL made ahead (12-step runs on test machines: this 10.5 GB buffer took 0.6-0.8 s here)
                Some(m) => crate::pinned::PinnedBuf::from_mem(m, total), None => crate::pinned::PinnedBuf::new(&ex.cuda, total)? } } };
        for (id, off, b) in &index {
            checked("cuMemcpyDtoH anchor", (ex.cuda.download)(data[*off..*off + *b].as_mut_ptr().cast(), ex.allocations[id].address, *b))?;
        }
        Ok((Boundary { sub: None, path: PathBuf::new(), index, host: host_snap(ex, frame), step: usize::MAX, delta: false, file_bytes: 0, ram: None, dev: 0, cs: None }, data))
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
        Ok(Boundary { sub: None, path: PathBuf::new(), index, host: host_snap(ex, frame), step: usize::MAX, delta: false, file_bytes: 0, ram: None, cs: None, dev: buf })
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
        let _profile = crate::step_profile::StepRange::begin("ad", step)?;
        // s60-cardmem: what the reversed steps so far measured (the boundary layer's rings, the reverse's own growth, the
        // largest cold family), for AD_CARD_ROOM_FILE
        crate::card_room::note_rev_now(crate::spill::device_now() as u64, crate::spill::cold_families().first().map(|f| f.1).unwrap_or(0) as u64);
        crate::card_room::save();
        let t0 = Instant::now();
        let ms0 = self.entry_ms;
        let n = flat.len();
        let (mut end, mut hint, mut segments, mut replayed, mut overflows) = (n, None::<usize>, 0usize, 0usize, 0usize);
        let mut tape_peak = 0usize;
        // packed-tape: the step's tape bytes summed over its segments (what one pass would need, less the copies two
        // segments' caches each hold): the per-step tape receipt the segmented path otherwise lacks.
        let mut tape_sum = 0usize;
        // adjoint-speed (ad/host-speed): when the step's measured tape estimate fits the tape, one replay from the step's
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
        // fast rules (tape guess): the largest ratio measured on ANOTHER step is not this step's ratio. THE BREAKAGE:
        // fast recording pk120e4-fast-warp-e6, cloud held (a test machine w3/fast-held-s1): step 3 was reversed first and packed at
        // 0.238 of its 74.5 GB estimate (17.73 GB); that ratio times step 2's 96.3 GB estimate is 22.9 GB, over the 19.32 GB
        // tape, so step 2 was sent to 4 segments (7.844 s) without a try, although it needs 17.56 GB (0.182) and reverses
        // in one pass in 3.926 s when tried (tape24/fast-held-s1-tape24). size_tape made the tape AD_TAPE_PACK_GUESS x the
        // largest estimate, so a step whose own estimate x that guess fits the tape is tried in one pass too; one that does
        // not fit after all overflows into the segmented path below as before (the same words, one replay more).
        let pack_guess = std::env::var("AD_TAPE_PACK_GUESS").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.2).clamp(0.01, 1.0);
        let packed_fits = self.pack && if self.pack_ratio > 0.0 { (total_est as f64 * self.pack_ratio) as usize <= self.tape.bytes
                                                                  || (total_est as f64 * pack_guess) as usize <= self.tape.bytes }
                                       else { total_est / 10 <= self.tape.bytes };
        // memory (streamed tape): one replay, the tape in chunks through page-locked RAM (stream_reverse_step).
        if self.st_on && self.pack && n > 0 && self.census.is_none() {
            // rerun-tape (AD_TAPE_PREDICT=1): a step whose predicted share decisions did not hold is taped again
            // from its restored start (restore uploads it: the skip flags were taken by the first restore), every decision
            // read for real. Nothing of the first attempt was copied out or walked back.
            // rerun-tape (checker pass): a start state held only on the device (--bstore-content 1, CsB::Dev) can be
            // restored once. THE BREAKAGE THIS PREVENTS: after a miss, the redo's restore stopped the run with "the device has
            // moved on". Such a step reads every decision for real (its decisions still go into the book).
            if deriv::predict::on() && matches!(prev.cs, Some(CsB::Dev(_))) {
                eprintln!("{}", json!({"event":"window_tape_predict_skip","step":step,"why":"start state only on the device: no redo possible"}));
                deriv::predict::force_real(true);
                let r = self.stream_reverse_step(ex, base, frame, step, flat, prev, prev_data, cur, cur_data, t0, ms0, total_est);
                deriv::predict::force_real(false);
                return r;
            }
            if deriv::predict::on() {
                match self.stream_reverse_step(ex, base, frame, step, flat, prev, prev_data, cur, cur_data, t0, ms0, total_est) {
                    Err(e) if e == deriv::predict::MISS => {
                        self.rev_rows.clear();
                        deriv::predict::force_real(true);
                        let r = self.stream_reverse_step(ex, base, frame, step, flat, prev, prev_data, cur, cur_data, t0, ms0, total_est);
                        deriv::predict::force_real(false);
                        return r;
                    },
                    r => return r,
                }
            }
            return self.stream_reverse_step(ex, base, frame, step, flat, prev, prev_data, cur, cur_data, t0, ms0, total_est);
        }
        if self.one_pass && n > 0 && (total_est <= self.tape.bytes || packed_fits) {
            let t_r = Instant::now();
            self.restore(ex, frame, prev, prev_data)?;
            dead_poison_rev(ex, step, "restore")?;
            self.rv_restore_s = t_r.elapsed().as_secs_f64();
            let t_one = Instant::now();
            if let Some(f) = frame.as_mut() { f.begin(step)?; }
            match self.tape_range(ex, base, frame, step, flat, 0, n, false)? {
                Taping::Done(tape) => {
                    replayed += n;
                    let t_tape = t_one.elapsed().as_secs_f64();
                    // adjoint-speed-2: the redo check (end state word for word against the stored boundary) runs on worker
                    // threads beside the walk-back, off the critical path: the state is downloaded first (page-locked, before
                    // any adjoint command), the walk-back writes only derivative and tape memory, and a mismatch still
                    // refuses the step with the same message (after the walk instead of before it).
                    let t_dl = Instant::now();
                    let live = self.redo_download(ex, cur)?;
                    dead_poison_rev(ex, step, "walk")?;
                    let cidx = self.redo_index(cur, step);
                    let ranges: Vec<Vec<(usize, usize)>> = redo_ranges(ex, cur, &cidx, step);
                    let dl_s = t_dl.elapsed().as_secs_f64();
                    let t_pass1 = t0.elapsed().as_secs_f64();
                    tape_peak = tape_peak.max(self.tape.used);
                    tape_sum += self.tape.used;
                    let cuda = &ex.cuda as *const Cuda;
                    let t_w = Instant::now();
                    let (cmp, walked, walk_s) = std::thread::scope(|sc| {
                        let (index, live_ref, ranges_ref, cs_ref) = (&cidx, &live[..], &ranges, self.cs_redo_res.take());
                        let h = sc.spawn(move || compare_stored(cs_ref, index, live_ref, cur_data, ranges_ref, step));
                        let w = self.walk_back(&*cuda, ex, step, tape);
                        let walk_s = t_w.elapsed().as_secs_f64();
                        (h.join(), w, walk_s)
                    });
                    let wait_s = t_w.elapsed().as_secs_f64() - walk_s;
                    self.anchor_pool.push(live);
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
                    let wk = self.take_wk();
                    return Ok(json!({"step":step,"commands":n,"segments":1,"overflow_splits":0,"commands_replayed":replayed,"host_prof":host_prof,"pack":pack,"tape_rows_packed":rrows,"walk_parts_s":wk,
                              "stages_s":{"restore":self.rv_restore_s,"replay_and_tape":self.rv_tape_s,"redo_download":self.rv_redo_dl_s,
                                          "walk_back":self.rv_walk_s,"redo_compare_wait_after_walk":self.rv_redo_wait_s},
                              "tape_peak_bytes":tape_peak,"tape_sum_bytes":tape_sum,"adjoint_entry_s":(self.entry_ms - ms0) / 1e3,"pass1_s":0.0,"one_pass_tape_s":t_pass1,
                              "tape_estimate_bytes":total_est,"wall_s":t0.elapsed().as_secs_f64(),"redo":redo}));
                },
                Taping::Overflow(k) => { replayed += k; overflows += 1; self.one_pass_fallbacks += 1; },
                Taping::Full(..) => return Err("DEFECT tape_range ended a chunk outside the streamed path".into()),
            }
        }
        // Pass 1 (anchors + redo check).
        self.restore(ex, frame, prev, prev_data)?;
        dead_poison_rev(ex, step, "restore")?;
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
        let host_cmp = std::env::var("AD_REDO_COMPARE").as_deref() == Ok("host") && !self.cs_on;
        let (dh, rh) = if host_cmp { self.differing(ex, cur, cur_data, step)? } else { (0, 0) };
        if dh != 0 { return Err(format!("step {step}: restore + redo differs from the stored boundary in {dh} words")); }
        let live = if host_cmp { None } else { Some(self.redo_download(ex, cur)?) };
        let cidx = self.redo_index(cur, step);
        let ranges: Vec<Vec<(usize, usize)>> = if host_cmp { Vec::new() } else { redo_ranges(ex, cur, &cidx, step) };
        let t_pass1 = t0.elapsed().as_secs_f64();
        let (cmp, segs) = std::thread::scope(|sc| {
            let h = live.as_ref().map(|l| { let (index, lr, rr, cs_ref) = (&cidx, &l[..], &ranges, self.cs_redo_res.take()); sc.spawn(move || compare_stored(cs_ref, index, lr, cur_data, rr, step)) });
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
                match self.tape_range(ex, base, frame, step, flat, lo, end, false)? {
                    Taping::Full(..) => return Err("DEFECT tape_range ended a chunk outside the streamed path".into()),
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
                        if segments == 0 { dead_poison_rev(ex, step, "walk")?; }
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
        if let Some(l) = live { self.anchor_pool.push(l); }
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
        let wk = self.take_wk();
        Ok(json!({"step":step,"commands":n,"segments":segments,"overflow_splits":overflows,"commands_replayed":replayed,"host_prof":host_prof,"pack":pack,"tape_rows_packed":rrows,"walk_parts_s":wk,
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
    /// machine (scale_chunk split in two, the same arithmetic): the allocations whose words fall in arena words
    /// [f, f + n) in index order (id, off, bytes, e0), per allocation its sorted merged unstable byte ranges and field
    /// weight (cached per boundary), whether the field weight is used, and the direction's scale.
    #[allow(clippy::type_complexity)]
    fn weights_prep(&self, b: &Boundary, data: &[u8], f: i64, n: i64)
                    -> (Vec<(u64, usize, usize, i64)>, Vec<std::sync::Arc<(Vec<(usize, usize)>, f32)>>, bool, f32, Vec<Option<std::sync::Arc<PlaneW>>>) {
        let threads = std::thread::available_parallelism().map(|t| t.get()).unwrap_or(4).min(64);
        let (base_el, index, fixed) = (&self.base_el, &b.index, &self.fixed);
        let (mask, ab) = (self.taylor_mask, &self.address_bearing);
        // --direction field: the window-end boundary's weights are 1 / rms of the allocation's valid words (the dual of
        // the relative x); every other boundary keeps |primal|. Same y in <y, M x> and in the reverse seed.
        let inv = self.field_y && self.y_path.as_deref() == Some(b.path.as_path());
        // --direction fieldxy: every other boundary (the x side) gets rms_f instead of the word's own |primal|.
        let fwd = self.field_x && !inv;
        let xs = if inv { self.y_scale } else { self.x_scale };
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
            // hunt-429: the per-allocation work is weights_alloc_prep, which holds this closure's body word for word
            // (its lookups by allocation are made here and passed in), so a unit test can run it without a card.
            let one = |&(id, off, bytes, _): &(u64, usize, usize, i64)| -> (Vec<(usize, usize)>, f32) {
                weights_alloc_prep(data, off, bytes, stable, inv, fwd, history.get(&id), residue.get(&(bstep, id)).map(|r| &r[..]),
                                   relocs.get(&id).map(|r| &r[..]), if stable && inv { dead_ranges().get(&id).map(|r| &r[..]) } else { None })
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
        // species planes (AD_DOT_PLANES): the planed allocation of this side (the window end is the y side, every other
        // boundary the x side) gets its plane table; every other allocation None (its weights as before, word for word).
        // An unreadable spec was refused at the start of the run (run_window), so here it reads as set or not set.
        let y_side = self.y_path.as_deref() == Some(b.path.as_path());
        let pw: Vec<Option<std::sync::Arc<PlaneW>>> = match plane_spec().ok().flatten() {
            Some(ps) => {
                let (pid, _) = ps.end(y_side);
                let planes = ps.ranges(y_side);
                als.iter().zip(prep.iter()).map(|(&(id, off, bytes, _), pr)| (id == pid).then(||
                    std::sync::Arc::new(plane_table(data, off, bytes, stable, inv, fwd, &pr.0, &planes, ps.cap)))).collect() },
            None => Vec::new() };
        (als, prep, inv || fwd, xs, pw)
    }
    /// machine: the weights of arena words [lo, lo + part.len()) into `part` (a word no allocation covers is left
    /// as the caller set it: zero).
    fn weights_span(data: &[u8], als: &[(u64, usize, usize, i64)], prep: &[std::sync::Arc<(Vec<(usize, usize)>, f32)>],
                    field: bool, xs: f32, stable: bool, lo: i64, part: &mut [f32]) {
        Self::weights_span_pw(data, als, prep, &[], field, xs, stable, lo, part)
    }
    /// species planes: weights_span with each allocation's plane table (`pw[i]` for `als[i]`; missing or None: no planes,
    /// the same arithmetic as before). In a plane the word's field weight is the plane's and the cap is the table's;
    /// outside the planes of a planed allocation the field weight is the table's rest weight.
    #[allow(clippy::too_many_arguments)]
    fn weights_span_pw(data: &[u8], als: &[(u64, usize, usize, i64)], prep: &[std::sync::Arc<(Vec<(usize, usize)>, f32)>],
                       pw: &[Option<std::sync::Arc<PlaneW>>], field: bool, xs: f32, stable: bool, lo: i64, part: &mut [f32]) {
        let hi = lo + part.len() as i64;
        let valid = |v: f32| v.is_finite() && v <= WEIGHT_MAX && v > 1e-30;
        for (i, (&(_, off, bytes, e0), pr)) in als.iter().zip(prep.iter()).enumerate() {
            let (rg, r) = (&pr.0, &pr.1);
            let e1 = e0 + (bytes / 4) as i64;
            let (a, z) = (e0.max(lo), e1.min(hi));
            if a >= z { continue; }
            let rd = |k: usize| f32::from_le_bytes(data[k..k + 4].try_into().unwrap()).abs();
            let o0 = 4 * (a - e0) as usize;
            let mut p = rg.partition_point(|x| x.1 <= o0);
            match pw.get(i).and_then(|t| t.as_ref()) {
                None => for e in a..z {
                    let o = 4 * (e - e0) as usize;
                    let k = off + o;
                    while p < rg.len() && rg[p].1 <= o { p += 1; }
                    let un = (p < rg.len() && rg[p].0 <= o) || (stable && address_shaped(data, off, bytes, o));
                    let v = rd(k);
                    part[(e - lo) as usize] = if !valid(v) || un { 0.0 } else if field { r * xs } else { v * xs };
                },
                Some(t) => {
                    let (tab, cap, rest) = (&t.0, t.1, t.2);
                    let mut q = tab.partition_point(|x| x.1 <= o0);
                    for e in a..z {
                        let o = 4 * (e - e0) as usize;
                        let k = off + o;
                        while p < rg.len() && rg[p].1 <= o { p += 1; }
                        while q < tab.len() && tab[q].1 <= o { q += 1; }
                        let un = (p < rg.len() && rg[p].0 <= o) || (stable && address_shaped(data, off, bytes, o));
                        let v = rd(k);
                        let inp = q < tab.len() && tab[q].0 <= o;
                        let ok = if inp { v.is_finite() && v > 1e-30 && (!cap || v <= WEIGHT_MAX) } else { valid(v) };
                        let rr = if inp { tab[q].2 } else { rest };
                        part[(e - lo) as usize] = if !ok || un { 0.0 } else if field { rr * xs } else { v * xs };
                    }
                }
            }
        }
    }
    fn scale_chunk(&self, sc: Option<(&Boundary, &[u8])>, f: i64, n: i64) -> Option<Vec<f32>> {
        let (b, data) = sc?;
        let (als, prep, field, xs, pw) = self.weights_prep(b, data, f, n);
        let mut out = vec![0f32; n as usize];
        let threads = std::thread::available_parallelism().map(|t| t.get()).unwrap_or(4).min(64);
        let per = (n as usize).div_ceil(threads).max(1);
        let (als, prep, pw) = (&als, &prep, &pw);
        let stable = self.stable_dir;
        std::thread::scope(|s| {
            for (c, part) in out.chunks_mut(per).enumerate() {
                let lo = f + (c * per) as i64;
                s.spawn(move || Self::weights_span_pw(data, als, prep, pw, field, xs, stable, lo, part));
            }
        });
        Some(out)
    }
    /// machine: the dot test's direction weights, computed ahead. For each direction in `dirs` and each arena
    /// chunk in order, `use_chunk(first word, words, one page-locked weight slice per direction)` runs on this thread
    /// while worker threads fill the next two chunks' slices. THE BREAKAGE THIS FIXES: every 128 MiB chunk made a new
    /// pageable vector, started 64 threads and went up through CUDA's pageable copy, with the card waiting: at 120 km
    /// fill x 1.2 s, <y, M x> 6.5 s, fill y 1.7 s, <x, M^T y> 6.6 s per window (a test machine machine pair1-b marks).
    /// The weights are scale_chunk's (weights_prep / weights_span), word for word. AD_DIR_PIPE=0 keeps the old path.
    unsafe fn weights_piped(&self, cuda: &Cuda, dirs: &[(&Boundary, &[u8])],
                            mut use_chunk: impl FnMut(i64, i64, &[&[u8]]) -> Result<(), String>) -> Result<(), String> {
        const WAVE: usize = 2;
        const SPLIT: usize = 4;
        let nd = dirs.len();
        let cb = self.tmp_words as usize * 4;
        let preps: Vec<_> = dirs.iter().map(|d| self.weights_prep(d.0, d.1, 0, self.words)).collect();
        let datas: Vec<&[u8]> = dirs.iter().map(|d| d.1).collect();
        let mut pins = self.wpins.lock().map_err(|_| "direction weight buffers poisoned")?;
        while pins.len() < 2 * WAVE * nd { pins.push(crate::pinned::PinnedBuf::new(cuda, cb)?); }
        // one writer per buffer per wave (the worker that owns that chunk and direction); this thread reads a buffer
        // only after its wave's workers are joined
        let base: Vec<usize> = pins.iter().map(|b| b.as_ptr() as usize).collect();
        let mut chunks: Vec<(i64, i64)> = Vec::new();
        let mut f = 0i64;
        while f < self.words { let n = (self.words - f).min(self.tmp_words); chunks.push((f, n)); f += n; }
        let nw = chunks.len().div_ceil(WAVE);
        let (chunks, preps, datas, base) = (&chunks, &preps, &datas, &base);
        let stable = self.stable_dir;
        let slot = move |w: usize, j: usize, d: usize| base[((w % 2) * WAVE + j) * nd + d];
        let compute = move |w: usize| {
            std::thread::scope(|s| {
                for (j, &(f, n)) in chunks[w * WAVE..((w + 1) * WAVE).min(chunks.len())].iter().enumerate() {
                    for d in 0..nd {
                        let (als, prep, field, xs, pw) = (&preps[d].0, &preps[d].1, preps[d].2, preps[d].3, &preps[d].4);
                        let data = datas[d];
                        let out = unsafe { std::slice::from_raw_parts_mut(slot(w, j, d) as *mut f32, n as usize) };
                        let per = (n as usize).div_ceil(SPLIT).max(1);
                        for (c, part) in out.chunks_mut(per).enumerate() {
                            let lo = f + (c * per) as i64;
                            s.spawn(move || { part.fill(0.0); Self::weights_span_pw(data, als, prep, pw, field, xs, stable, lo, part); });
                        }
                    }
                }
            });
        };
        if nw > 0 { compute(0); }
        for w in 0..nw {
            std::thread::scope(|s| -> Result<(), String> {
                let next = (w + 1 < nw).then(|| s.spawn(move || compute(w + 1)));
                let mut r = Ok(());
                for (j, &(f, n)) in chunks[w * WAVE..((w + 1) * WAVE).min(chunks.len())].iter().enumerate() {
                    let bufs: Vec<&[u8]> = (0..nd).map(|d| unsafe { std::slice::from_raw_parts(slot(w, j, d) as *const u8, n as usize * 4) }).collect();
                    r = use_chunk(f, n, &bufs);
                    if r.is_err() { break; }
                }
                if let Some(h) = next { h.join().map_err(|_| "direction weight worker panicked".to_string())?; }
                r
            })?;
        }
        Ok(())
    }
    /// machine: `gen` with the chunk's weights already in page-locked memory (weights_piped).
    unsafe fn gen_from(&self, cuda: &Cuda, dst: u64, f: i64, n: i64, seed: u64, w: &[u8]) -> Result<(), String> {
        let mut sp = self.tmp[2];
        checked("cuMemcpyHtoD direction weights", (cuda.upload)(sp, w.as_ptr().cast(), w.len()))?;
        let (mut t, mut first, mut nn, mut s) = (dst, f, n, seed);
        launch(cuda, self.f.fill, [65535, 1, 1], kargs!(t, first, nn, s, sp))?;
        dev::sync(cuda)
    }
    /// Generate u(seed) (times the relative weights when given) into `dst` for arena words [f, f + n).
    unsafe fn gen(&self, cuda: &Cuda, dst: u64, f: i64, n: i64, seed: u64, sc: Option<(&Boundary, &[u8])>) -> Result<(), String> {
        if let Some(pair) = &self.vectors {
            dev::zero(cuda, dst, n as usize * 4)?;
            for segment in pair.side(seed)? {
                let lo=self.base_el[&segment.allocation]+(segment.offset/4) as i64;
                let hi=lo+(segment.data.len()/4) as i64;
                let (a,b)=(lo.max(f),hi.min(f+n));
                if a<b {
                    dev::upload(cuda,dst+4*(a-f) as u64,
                        &segment.data[4*(a-lo) as usize..4*(b-lo) as usize])?;
                }
            }
            return dev::sync(cuda);
        }
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
        if let (Some(d), true) = (sc, dir_pipe() && self.vectors.is_none()) {
            return self.weights_piped(cuda, &[d], |f, n, w| self.gen_from(cuda, self.sh_base + 4 * f as u64, f, n, seed, w[0]));
        }
        let mut f = 0i64;
        while f < self.words { let n = (self.words - f).min(self.tmp_words); self.gen(cuda, self.sh_base + 4 * f as u64, f, n, seed, sc)?; f += n; }
        Ok(())
    }
    unsafe fn zero(&self, cuda: &Cuda) -> Result<(), String> { dev::zero(cuda, self.sh_base, self.words as usize * 4)?; dev::sync(cuda) }
    /// Full shadow-arena witness, including non-live allocations and alignment
    /// padding. The result fingerprint and dot use this arena, not just the
    /// allocation set live at the physical boundary.
    unsafe fn export_vectors(&self, cuda: &Cuda, b: &Boundary, root: &Path, tag: &str) -> Result<Value,String> {
        let dir=root.join(tag);
        if dir.exists() {return Err(format!("DEFECT: vector export already exists {}",dir.display()));}
        fs::create_dir_all(&dir).map_err(|e|e.to_string())?;
        dev::sync(cuda)?;
        let mut rows=Vec::new();
        let live: std::collections::HashSet<u64> = b.index.iter().map(|x| x.0).collect();
        for (id, start, n) in vector_export_ranges(&self.base_el, self.words)? {
            let data=dev::download(cuda,self.sh(start),n)?;
            let file=format!("allocation-{id}.f32");
            fs::write(dir.join(&file),&data).map_err(|e|e.to_string())?;
            rows.push(json!({"allocation":id,"offset":0,"file":file,"bytes":n,
                "allocation_shadow_bytes":self.shadows[&id].bytes,
                "live_at_boundary":live.contains(&id),
                "arena_word":start,"sha256":crate::exec::digest(&data)}));
        }
        let doc=json!({"schema":"ad.window-vector-export/v2","tag":tag,"boundary":b.step,
            "scope":"complete_shadow_arena_including_padding",
            "segments":rows,"arena_words":self.words,"whole_window_gate":false});
        fs::write(dir.join("manifest.json"),serde_json::to_vec_pretty(&doc).unwrap()).map_err(|e|e.to_string())?;
        Ok(json!({"manifest":dir.join("manifest.json"),"allocations":rows.len(),"boundary":b.step,
            "scope":"complete_shadow_arena_including_padding","bytes":self.words as u64 * 4}))
    }
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
                // a product of exactly zero moves no bit (a nudge of size zero keeps a -0.0 word)
                let d = h * v as f64;
                if d != 0.0 { buf[k..k + 4].copy_from_slice(&(((p as f64) + d) as f32).to_le_bytes()); }
            }
            dev::upload(cuda, a.address, &buf)?;
        }
        dev::sync(cuda)
    }
    /// --taylor-dir "dense" blocks in a Taylor row: primal[allocation, byte offset ..] += h * block (float64, rounded once
    /// to float32), as perturb_terms moves the sparse terms. Returns the words it changed. THE BREAKAGE THIS FIXES: the
    /// rows moved only the sparse terms, so a dense-only direction reran the real model unchanged while the tangent was
    /// seeded from the dense blocks (TABLE, <local path> Case B).
    unsafe fn perturb_dense(&self, ex: &Execution, h: f64, blocks: &[(u64, usize, Vec<u8>)]) -> Result<u64, String> {
        let cuda = &ex.cuda;
        dev::sync(cuda)?;
        let mut moved = 0u64;
        for (al, off, bytes) in blocks {
            let a0 = ex.allocations.get(al).ok_or_else(|| format!("taylor-dir dense allocation {al} not live at boundary 0"))?;
            if off + bytes.len() > a0.bytes { return Err(format!("taylor-dir dense block {al}:{off} (+{} bytes) is outside allocation {al} ({} bytes)", bytes.len(), a0.bytes)); }
            let mut buf = dev::download(cuda, a0.address + *off as u64, bytes.len())?;
            moved += add_dense(&mut buf, bytes, h);
            dev::upload(cuda, a0.address + *off as u64, &buf)?;
        }
        dev::sync(cuda)?;
        Ok(moved)
    }
    /// --taylor-dir "dense" blocks: <x, shadow> over the blocks (added to sparse_dot's sum for the rows' prediction).
    unsafe fn dense_dot(&self, cuda: &Cuda, blocks: &[(u64, usize, Vec<u8>)]) -> Result<f64, String> {
        let mut g = 0f64;
        for (al, off, bytes) in blocks {
            let (Some(&el), Some(sh)) = (self.base_el.get(al), self.shadows.get(al)) else { return Err(format!("taylor-dir dense allocation {al} has no shadow")) };
            if off + bytes.len() > sh.bytes { return Err(format!("taylor-dir dense block {al}:{off} is outside the shadow of allocation {al}")); }
            g += dot_dense(bytes, &dev::download(cuda, self.sh(el) + *off as u64, bytes.len())?);
        }
        Ok(g)
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
        let f64on = std::env::var("AD_DOT_F64").as_deref() == Ok("1"); let mut comp = (0f64, 0f64);
        let alloc_asked = std::env::var("AD_DOT_ALLOC").as_deref() == Ok("1");
        let mut alloc: Option<BTreeMap<u64, (f64, f64, f64)>> = abs_terms_wanted().then(BTreeMap::new);
        let ranges: Vec<(u64, (i64, i64))> = if alloc.is_some() { match sa { Some((b, _)) => b.index.iter()
            .map(|(id, _, bytes)| (*id, (self.base_el[id], self.base_el[id] + (*bytes / 4) as i64))).collect(), None => Vec::new() } } else { Vec::new() };
        // speed outside: the AD_DOT_ALLOC split off this thread. THE BREAKAGE THIS FIXES: every 128 MiB chunk made two new
        // pageable vectors, read a and S back through CUDA's pageable copy and summed them on this one thread with the card
        // waiting: at 3 km 12 steps 22.4 s of <y, M x> and about 12 s of <x, M^T y> per window (a test machine win12f BUDGET.md).
        // Now a and S of a chunk come down into one of DSLOTS pairs of page-locked buffers and a worker thread sums that chunk
        // (per allocation, word order, as before); the chunks' parts are added in chunk order after the loop, so every
        // per-allocation sum and abs_terms have the bits they had.
        const DSLOTS: usize = 6;
        let alloc_on = alloc.is_some();
        let mut dpins = self.dpins.lock().map_err(|_| "dot split buffers poisoned")?;
        if alloc_on { while dpins.len() < 2 * DSLOTS { dpins.push(crate::pinned::PinnedBuf::new(cuda, self.tmp_words as usize * 4)?); } }
        let dbase: Vec<usize> = dpins.iter_mut().map(|b| b.as_mut_ptr() as usize).collect();
        let mut done: Vec<Vec<(usize, f64, f64, f64)>> = Vec::new();
        std::thread::scope(|ts| -> Result<(), String> {
        let mut busy: Vec<Option<std::thread::ScopedJoinHandle<Vec<(usize, f64, f64, f64)>>>> = (0..DSLOTS).map(|_| None).collect();
        let mut ci = 0usize;
        let (ranges, dbase, done) = (&ranges, &dbase, &mut done);
        {
        let mut chunk = |f: i64, n: i64, w: Option<&[&[u8]]>| -> Result<(), String> {
            match w {
                Some(w) => { self.gen_from(cuda, self.tmp[0], f, n, seed_a, w[0])?; self.gen_from(cuda, self.tmp[1], f, n, seed_b, w[1])?; },
                None => { self.gen(cuda, self.tmp[0], f, n, seed_a, sa)?; self.gen(cuda, self.tmp[1], f, n, seed_b, sb)?; },
            }
            let (mut x, mut sv, mut y, mut nn, mut part) = (self.tmp[0], self.sh_base + 4 * f as u64, self.tmp[1], n, self.part);
            checked("dot launch", (cuda.launch)(self.f.dot2, 1024, 1, 1, 256, 1, 1, 0, ptr::null_mut(), kargs!(x, sv, y, nn, part).as_mut_ptr(), ptr::null_mut()))?;
            let mut host = vec![0f64; 3072];
            checked("cuMemcpyDtoH partials", (cuda.download)(host.as_mut_ptr().cast(), self.part, 3072 * 8))?;
            for b in 0..1024 { p += host[3 * b]; q += host[3 * b + 1]; bad += host[3 * b + 2] as u64; }
            if f64on {
                // inst (AD_DOT_F64=1): the same words read back, exact binary64 products, compensated sum (a = 0 skipped as the kernel does).
                let mut av = vec![0f32; n as usize]; let mut sv2 = vec![0f32; n as usize];
                checked("cuMemcpyDtoH dot a f64", (cuda.download)(av.as_mut_ptr().cast(), self.tmp[0], n as usize * 4))?;
                checked("cuMemcpyDtoH dot S f64", (cuda.download)(sv2.as_mut_ptr().cast(), self.sh_base + 4 * f as u64, n as usize * 4))?;
                for k in 0..n as usize { let u = av[k] as f64; if u == 0.0 { continue; } neumaier(&mut comp, u * sv2[k] as f64); }
            }
            if alloc_on {
                // AD_DOT_ALLOC=1 (diagnostic only, step6-radiation): the same sum a*S split by allocation, in float64 on
                // the host. It reads the chunk's a and S back; nothing it computes feeds the window.
                let slot = ci % DSLOTS; ci += 1;
                // a buffer pair is written again only after the worker summing out of it has ended (it holds chunk ci - DSLOTS)
                if let Some(h) = busy[slot].take() { done.push(h.join().map_err(|_| "dot split worker panicked".to_string())?); }
                let (pa, ps) = (dbase[2 * slot], dbase[2 * slot + 1]);
                checked("cuMemcpyDtoH dot a", (cuda.download)(pa as *mut c_void, self.tmp[0], n as usize * 4))?;
                checked("cuMemcpyDtoH dot S", (cuda.download)(ps as *mut c_void, self.sh_base + 4 * f as u64, n as usize * 4))?;
                busy[slot] = Some(ts.spawn(move || {
                    let av = unsafe { std::slice::from_raw_parts(pa as *const f32, n as usize) };
                    let sv2 = unsafe { std::slice::from_raw_parts(ps as *const f32, n as usize) };
                    let mut out: Vec<(usize, f64, f64, f64)> = Vec::new();
                    for (ri, (_, (e0, e1))) in ranges.iter().enumerate() {
                        let (lo, hi) = ((*e0).max(f), (*e1).min(f + n));
                        if lo >= hi { continue; }
                        let mut s1 = 0f64; let mut big = 0f64; let mut sa = 0f64;
                        for e in lo..hi { let k = (e - f) as usize; let t = av[k] as f64 * sv2[k] as f64; s1 += t; sa += t.abs(); big = big.max((sv2[k] as f64).abs()); }
                        out.push((ri, s1, big, sa));
                    }
                    out
                }));
            }
            Ok(())
        };
        match (sa, sb, dir_pipe() && self.vectors.is_none()) {
            (Some(da), Some(db), true) => self.weights_piped(cuda, &[da, db], |f, n, w| chunk(f, n, Some(w)))?,
            _ => while f < self.words { let n = (self.words - f).min(self.tmp_words); chunk(f, n, None)?; f += n; },
        }
        }
        // the workers still out hold the last chunks, oldest in slot ci % DSLOTS
        for j in 0..DSLOTS { if let Some(h) = busy[(ci + j) % DSLOTS].take() { done.push(h.join().map_err(|_| "dot split worker panicked".to_string())?); } }
        Ok(())
        })?;
        drop(dpins);
        if let Some(acc) = alloc.as_mut() {
            for part in &done { for &(ri, s1, big, sa) in part {
                let ent = acc.entry(ranges[ri].0).or_insert((0.0, 0.0, 0.0)); ent.0 += s1; ent.1 = ent.1.max(big); ent.2 += sa;
            } }
        }
        if let Some(acc) = alloc {
            // S01c: also the sum of |a*S| per allocation and in total (the dot's conditioning: the float32 rounding of
            // the terms scales with it, not with the signed sum).
            let abs_total: f64 = acc.values().map(|e| e.2).sum();
            LAST_ABS_TERMS.store(abs_total.to_bits(), std::sync::atomic::Ordering::Relaxed);
            let mut v: Vec<(u64, f64, f64, f64)> = acc.into_iter().map(|(k, (a, b, c))| (k, a, b, c)).collect();
            v.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
            v.truncate(40);
            if alloc_asked { eprintln!("{}", json!({"event":"window_dot_alloc","seed_a":seed_a,"sum":p,"abs_sum":abs_total,"top":v})); }
        }
        LAST_F64.store(if f64on { (comp.0 + comp.1).to_bits() } else { f64::NAN.to_bits() }, std::sync::atomic::Ordering::Relaxed);
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
    /// hunt-120-start (AD_WORD_PROBE, off by default; see word_probe.rs): one `window_word_probe` line with the asked
    /// words of the derivative memory as they stand now, bit for bit. Read-only: each word is one 4-byte copy to the
    /// host after a drain; nothing is written to the card. With the switch unset this returns at its first line.
    /// THE BREAKAGE IT LOCATES: the 120 km 3-hour check moves with the size of the test direction, and the stored
    /// dumps (the two ends of the window) cannot say which kernel first breaks exact proportion in the heavy columns.
    unsafe fn word_probe(&self, cuda: &Cuda, sweep: &str, at: &str, step: usize, cmd: Option<(usize, &str)>, tag: &Value) -> Result<(), String> {
        let Some(cfg) = crate::word_probe::config()? else { return Ok(()) };
        // a host read inside an open derivative graph capture is refused by dev::sync and dev::download: a diagnostic
        // must not end the run it watches, so that point is skipped and says so
        if !dev::dispatch_stream().is_null() {
            eprintln!("{}", json!({"event":"window_word_probe_skipped","sweep":sweep,"at":at,"step":step,"index":cmd.map(|c| c.0),
                                   "why":"a derivative graph capture is open here: no host read"}));
            return Ok(());
        }
        dev::sync(cuda)?;
        let mut rows: Vec<Vec<u32>> = Vec::with_capacity(cfg.stripes.len());
        for s in &cfg.stripes {
            let sh = self.shadows.get(&s.alloc).ok_or_else(|| format!("AD_WORD_PROBE: stripe {} names allocation {}, which has no derivative memory in this window", s.name, s.alloc))?;
            s.check(sh.bytes)?;
            let mut v = Vec::with_capacity(s.count as usize);
            for k in s.words() { let b = dev::download(cuda, sh.address + 4 * k, 4)?; v.push(u32::from_le_bytes([b[0], b[1], b[2], b[3]])); }
            rows.push(v);
        }
        if let Some(e) = crate::word_probe::point(cfg, sweep, at, step, cmd, rows, tag) { eprintln!("{e}"); }
        Ok(())
    }
    /// hunt-60-repeat (AD_REV_DIGEST, off by default; see rev_digest.rs): one `window_rev_digest` line with the scale-free
    /// digest of the cotangent in every asked shadow allocation as it stands now. Read-only: the words come down in
    /// groups of at most 4 GiB (pageable copies, after a drain) and are digested on worker threads; nothing is written
    /// to the card. With the switch unset this returns at its first line.
    unsafe fn rev_digest(&self, cuda: &Cuda, at: &str, step: usize, cmd: Option<(usize, &str)>, tag: &Value) -> Result<(), String> {
        let Some(cfg) = crate::rev_digest::config()? else { return Ok(()) };
        // a host read inside an open derivative graph capture is refused by dev::sync and dev::download: a diagnostic must
        // not end the run it watches, so that point is skipped and says so
        if !dev::dispatch_stream().is_null() {
            eprintln!("{}", json!({"event":"window_rev_digest_skipped","at":at,"step":step,"segment_command":cmd.map(|c| c.0),
                                   "why":"a derivative graph capture is open here: no host read"}));
            return Ok(());
        }
        const GROUP: usize = 4 << 30;
        let t0 = Instant::now();
        dev::sync(cuda)?;
        let ids: Vec<(u64, u64, usize)> = self.shadows.iter().filter(|(id, _)| cfg.wants(**id)).map(|(id, s)| (*id, s.address, s.bytes)).collect();
        let mut rows: Vec<(u64, crate::rev_digest::BlockDigest)> = Vec::with_capacity(ids.len());
        let mut i = 0usize;
        while i < ids.len() {
            let (mut j, mut tot) = (i, 0usize);
            while j < ids.len() && (j == i || tot + ids[j].2 <= GROUP) { tot += ids[j].2; j += 1; }
            let mut bufs: Vec<Vec<u8>> = Vec::with_capacity(j - i);
            for x in &ids[i..j] { bufs.push(dev::download(cuda, x.1, x.2)?); }
            let d = crate::rev_digest::digest_many(&bufs);
            rows.extend(ids[i..j].iter().zip(d).map(|(x, d)| (x.0, d)));
            i = j;
        }
        let mut e = crate::rev_digest::event(at, step, cmd, tag, &rows);
        e["wall_s"] = json!(t0.elapsed().as_secs_f64());
        eprintln!("{e}");
        Ok(())
    }
    unsafe fn nonzero(&self, cuda: &Cuda) -> Result<u64, String> {
        self.support.reset_words(cuda)?;
        self.support.words_into(cuda, 0, self.sh_base, 0, self.words as usize * 4)?;
        let (_, w) = self.support.collect(cuda, 0, 1)?;
        Ok(w[0][1])
    }
    /// AD_DOT_SELF=1 (any-mesh; a second form of the window dot, diagnostic): y := W S, with S = M x the tangent result
    /// in the arena and W = 1 / sum(S^2) per allocation over the words the standard y direction weights (its nonzero
    /// words; fixed, address-bearing and unweighted words stay zero). Then <y, M x> is the number of allocations that
    /// carry tangent: a sum of squares with no cancellation, in which every allocation weighs the same, where the random
    /// y dot is one allocation's cancelling sum. Any y is a valid transpose test; this one is well conditioned.
    /// AD_DOT_SELF_ONLY=id,id keeps only those allocations (which output field is inexact).
    /// Returns (y bytes, <y, M x> in float64 with the float32 y actually used, a receipt).
    /// central hunt-3km-slot: the arithmetic is self_dot::self_weights (unchanged; unit-tested on the processor). The fifth
    /// value is AD_DOT_SELF_FLOOR's (floored units as (first byte, words, v), floored <y, M x>), None without it.
    #[allow(clippy::type_complexity)]
    unsafe fn self_y(&self, cuda: &Cuda, seed_y: u64, sc: Option<(&Boundary, &[u8])>, fields_default: Result<String, String>)
                     -> Result<(Vec<u8>, f64, Value, Vec<(u64, usize, usize, f64)>, Option<(Vec<(usize, usize, f64)>, f64)>), String> {
        let floor = self_dot::FloorSpec::from_env()?;
        if floor.is_some() && crate::peer::get().is_some() { return Err("AD_DOT_SELF_FLOOR is one-process only: its floored <y, M x> is not summed over ranks".into()); }
        let s = self.download_all(cuda)?; mark("self_s_down");
        self.fill(cuda, seed_y, sc)?; mark("self_fill");
        let g = self.download_all(cuda)?; mark("self_g_down");
        let only: Option<Vec<u64>> = std::env::var("AD_DOT_SELF_ONLY").ok().map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect());
        // hunt-60-repeat (AD_DOT_SELF_YPOW2=k, 0 unless set): y times 2^k, exactly; both sums of the check then carry 2^k.
        let ypow2 = crate::rev_digest::ypow2()?;
        // sixty (AD_DOT_SELF_FIELDS=<alloc-map.json>:<boundary>): the units of the self dot are the model's own prognostic
        // fields at that recorded boundary (the window end), not memory-pool allocations: one pool allocation holds several
        // fields and scratch words, so its single weight 1 / sum(S^2) is set by whichever words carry the largest tangent.
        // A unit is (number, first byte in the arena, words); fields are numbered in name order (legend in the receipt).
        let mut legend: Vec<Value> = Vec::new();
        // The default units are those fields: with AD_DOT_SELF_FIELDS unset the map and boundary are `fields_default`
        // (the package's alloc-map.json beside the plan at the window end, see the caller). AD_DOT_SELF_UNITS=pool asks for
        // memory-pool allocations (the diagnostic). When the default cannot be the fields (no map, no window-end boundary
        // in it, a rank plan, a map that cannot place every field in a tangent-carrying allocation) the units are the
        // pool's and the receipt says why (`units_why`): never a silent change, and never a run stopped by a default.
        // A map the user names (AD_DOT_SELF_FIELDS, AD_DOT_SELF_UNITS=fields) that cannot give the units stops the run.
        // THE BREAKAGE THIS PREVENTS: pool units give every allocation weight 1, so a few scratch or non-state float32
        // words weigh as much as a field: a 45-step 12 km window read 3.2e-07 on pool units, over the 1e-07 bar, while
        // its 12 fields read 1.9e-09 on the same tangent.
        let units_asked = std::env::var("AD_DOT_SELF_UNITS").ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
        let fields_env = std::env::var("AD_DOT_SELF_FIELDS").ok().filter(|v| !v.trim().is_empty());
        let fields_named = fields_env.is_some() || units_asked.as_deref() == Some("fields");
        let (fields_spec, units_why): (Option<String>, Option<String>) = match (units_asked.as_deref(), fields_env) {
            (Some("pool"), Some(_)) => return Err("AD_DOT_SELF_UNITS=pool with AD_DOT_SELF_FIELDS: both set the self-dot units".into()),
            (Some("pool"), None) => (None, Some("AD_DOT_SELF_UNITS=pool".into())),
            (None | Some("fields"), Some(v)) => (Some(v), None),
            (None, None) => match fields_default { Ok(d) => (Some(d), None), Err(why) => (None, Some(why)) },
            (Some("fields"), None) => (Some(fields_default.map_err(|why| format!("AD_DOT_SELF_UNITS=fields: {why}"))?), None),
            (Some(o), _) => return Err(format!("AD_DOT_SELF_UNITS={o}: fields or pool")) };
        // The field units of a map and boundary: (units, each unit's (allocation, byte offset in it), the legend), or why
        // this map cannot give them.
        #[allow(clippy::type_complexity)]
        let field_units = |spec: &str| -> Result<(Vec<(u64, usize, usize)>, Vec<(u64, usize)>, Vec<Value>), String> {
            let (path, bnd) = spec.rsplit_once(':').ok_or("AD_DOT_SELF_FIELDS wants <alloc-map.json>:<boundary>")?;
            let m: Value = serde_json::from_slice(&fs::read(path).map_err(|e| format!("AD_DOT_SELF_FIELDS {path}: {e}"))?).map_err(|e| e.to_string())?;
            let b = m["boundaries"][bnd].as_object().ok_or(format!("AD_DOT_SELF_FIELDS: no boundary {bnd} in {path}"))?;
            let mut names: Vec<&String> = b.keys().collect(); names.sort();
            let (mut u, mut src, mut lg) = (Vec::new(), Vec::new(), Vec::new());
            for (i, name) in names.iter().enumerate() {
                let (id, off) = (b[*name]["allocation"].as_u64().ok_or("AD_DOT_SELF_FIELDS: allocation")?, b[*name]["offset"].as_u64().ok_or("AD_DOT_SELF_FIELDS: offset")? as usize);
                let bytes = m["fields"][*name]["bytes"].as_u64().ok_or("AD_DOT_SELF_FIELDS: field bytes")? as usize;
                let sh = self.shadows.get(&id).ok_or(format!("AD_DOT_SELF_FIELDS: {name} is in allocation {id}, which has no shadow"))?;
                if off + bytes > sh.bytes { return Err(format!("AD_DOT_SELF_FIELDS: {name} [{off}, {off}+{bytes}) is outside allocation {id} ({} bytes)", sh.bytes)); }
                lg.push(json!([i, name, id, off, bytes]));
                u.push((i as u64, self.base_el[&id] as usize * 4 + off, bytes / 4));
                src.push((id, off));
            }
            Ok((u, src, lg))
        };
        let attempt = fields_spec.as_deref().map(field_units);
        let (made, fields_spec, units_why) = match attempt {
            Some(Ok(x)) => (Some(x), fields_spec, units_why),
            Some(Err(e)) if fields_named => return Err(e),
            // the default map could not give the units: memory allocations, with the map's own reason
            Some(Err(e)) => {
                let why = format!("the package's field map could not give the units ({})", e.trim_start_matches("AD_DOT_SELF_FIELDS").trim_start_matches(':').trim());
                eprintln!("{}", json!({"event":"window_self_dot_units_fallback","to":"pool","why":why}));
                (None, None, Some(why)) },
            None => (None, fields_spec, units_why) };
        let units_name = if made.is_some() { "fields" } else { "pool" };
        // each unit's (allocation, byte offset in it), beside `units`: the species planes are found by allocation bytes
        let (mut units, unit_src): (Vec<(u64, usize, usize)>, Vec<(u64, usize)>) = match made {
            Some((u, src, lg)) => { legend = lg; (u, src) },
            None => { let mut src = Vec::new();
                let u = self.shadows.iter().map(|(id, sh)| { src.push((*id, 0)); (*id, self.base_el[id] as usize * 4, sh.bytes / 4) }).collect();
                (u, src) } };
        // species planes (AD_DOT_PLANES): at the window end each plane of the planed allocation is its own unit (PLANE_UNIT0 +
        // plane); the allocation's bytes before and after the planes stay units of the allocation's own id.
        let mut plane_legend: Vec<Value> = Vec::new();
        // The unit that holds the planes is a whole allocation (pool units) or the stacked field itself (field units): it
        // is found by the allocation bytes it covers, and a unit the planes only partly lie in is refused.
        if let Some(ps) = plane_spec()? {
            let (pid, po) = ps.y;
            let end = po + ps.n * ps.pb;
            let mut out = Vec::with_capacity(units.len() + ps.n + 1);
            for (j, &(id, lo, n)) in units.iter().enumerate() {
                let (aid, a0) = unit_src[j];
                let a1 = a0 + 4 * n;
                if aid != pid || end <= a0 || po >= a1 { out.push((id, lo, n)); continue; }
                if po < a0 || end > a1 { return Err(format!("AD_DOT_PLANES: planes [{po}, {end}) of allocation {pid} are not inside one self-dot unit (the {units_name} unit {id} covers [{a0}, {a1}))")); }
                let r = po - a0;
                if r > 0 { out.push((id, lo, r / 4)); }
                for p in 0..ps.n {
                    let u = PLANE_UNIT0 + p as u64;
                    out.push((u, lo + r + p * ps.pb, ps.pb / 4));
                    plane_legend.push(json!([u, ps.names[p], pid, po + p * ps.pb, ps.pb]));
                }
                if end < a1 { out.push((id, lo + r + ps.n * ps.pb, (a1 - end) / 4)); }
            }
            if plane_legend.is_empty() { return Err(format!("AD_DOT_PLANES: no self-dot unit ({units_name} units) holds the window-end planes [{po}, {end}) of allocation {pid}")); }
            units = out;
        }
        // sixty (AD_DOT_SELF_DUMP=N, diagnostic): a unit with at most N weighted words prints them: [word in the unit,
        // tangent, y direction weight, primal at the window end when the end boundary has the arena layout]. Names the
        // cells and levels of a small set of words that carries a unit s whole tangent.
        let dump = std::env::var("AD_DOT_SELF_DUMP").ok().and_then(|v| v.parse::<u64>().ok());
        let prim = sc.map(|d| d.1).filter(|b| b.len() == s.len());
        let o = self_dot::self_weights_pow2(&s, &g, &units, only.as_deref(), dump, prim, floor.as_ref(), ypow2);
        mark("self_units");
        for d in &o.dumps { eprintln!("{d}"); }
        let (per, pcomp) = (&o.per, o.pcomp);   // pcomp (inst): <y, M x> again, compensated (printed as y_Mx_compensated)
        let mut info = json!({"event":"window_self_dot","units":units_name,"fields_spec":fields_spec,"units_why":units_why,"y_Mx":o.p,"y_Mx_compensated":pcomp.0 + pcomp.1,"allocations":o.used,"words":o.words,"skipped":o.skipped,"only":only,"fields":legend,
                          "largest_sum_sq":per.iter().take(8).collect::<Vec<_>>(),"smallest_sum_sq":per.iter().rev().take(8).collect::<Vec<_>>()});
        if let Some(f) = &o.floor { info["floor"] = self_dot::floor_receipt(f); }
        // species planes: the plane units ([unit id, name, allocation, byte offset, bytes]) and each one's weighted words and
        // sum of squares, so a plane with no tangent at the window end is named, not silently absent
        if !plane_legend.is_empty() {
            // the plane's largest |primal| at the window end (when the end boundary has the arena layout): a content check
            // that the planes are where the spec says (qv about 1e-2, the numbers 1e5 to 1e10)
            let lo_of = |u: u64| units.iter().find(|x| x.0 == u).map(|x| (x.1, x.2));
            let rows: Vec<Value> = plane_legend.iter().map(|l| { let u = l[0].as_u64().unwrap_or(0);
                let hit = per.iter().find(|x| x.0 == u);
                let pmax = match (prim, lo_of(u)) { (Some(b), Some((lo, n))) => Some(b[lo..lo + 4 * n].chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()).abs()).filter(|v| v.is_finite()).fold(0f32, f32::max)), _ => None };
                json!({"unit":u,"name":l[1].clone(),"allocation":l[2].clone(),"offset":l[3].clone(),"bytes":l[4].clone(),"words":hit.map(|x| x.1),"sum_sq":hit.map(|x| x.2),
                       "primal_max_abs_at_window_end":pmax}) }).collect();
            // A plane that carries no tangent at the window end is said so, by name, in the receipt and on stderr: the
            // check cannot test a path the linear model in use holds (in the shipped hrrrv5 set the cloud scheme's number
            // concentrations ni, nr, ng and nc are held: they have values at the window end and no tangent).
            let held: Vec<Value> = rows.iter().filter(|r| r["words"].as_u64().unwrap_or(0) == 0)
                .map(|r| json!({"name":r["name"].clone(),"primal_max_abs_at_window_end":r["primal_max_abs_at_window_end"].clone()})).collect();
            info["planes_without_tangent"] = json!({"planes":held,
                "what":"planes with no tangent word at the window end: this check does not test them (a species the linear model in use holds: in the shipped hrrrv5 set the number concentrations ni, nr, ng and nc)"});
            if !held.is_empty() { eprintln!("{}", json!({"event":"window_planes_without_tangent","planes":held,
                "what":"no tangent at the window end: the window check does not test these planes"})); }
            info["planes"] = json!(rows);
        }
        // hunt-60-repeat: the receipt names the factor only when one was asked, so a default run's receipt has the keys it always had
        if ypow2 != 0 { info["y_pow2"] = json!(ypow2); }
        eprintln!("{info}");
        let fl = o.floor.map(|f| (f.scale, f.y_mx));
        Ok((o.y, o.p, info, o.parts, fl))
    }
    /// machine: the whole shadow arena, read through two page-locked chunk buffers and copied into the result on
    /// worker threads while the next chunk comes down (it was one pageable cuMemcpyDtoH of the whole arena, 10.5 GB at
    /// 120 km). The same bytes. AD_DIR_PIPE=0 keeps the single copy.
    /// speed outside: the whole shadow arena written from host bytes through two page-locked chunk buffers, the next
    /// chunk copied in on worker threads while this one goes up (it was one pageable cuMemcpyHtoD of the whole arena,
    /// 10.5 GB at 3 km). The same bytes. AD_DIR_PIPE=0 keeps the single copy.
    unsafe fn upload_all(&self, cuda: &Cuda, src: &[u8]) -> Result<(), String> {
        let cb = self.tmp_words as usize * 4;
        if !dir_pipe() || src.len() < 2 * cb { return dev::upload(cuda, self.sh_base, src); }
        let mut pins = self.wpins.lock().map_err(|_| "direction weight buffers poisoned")?;
        while pins.len() < 2 { pins.push(crate::pinned::PinnedBuf::new(cuda, cb)?); }
        let pp = [pins[0].as_mut_ptr() as usize, pins[1].as_mut_ptr() as usize];
        let chunks: Vec<&[u8]> = src.chunks(cb).collect();
        let put = |dst: usize, part: &[u8]| {
            std::thread::scope(|t| {
                let per = part.len().div_ceil(8).max(1);
                for (c, x) in part.chunks(per).enumerate() {
                    t.spawn(move || unsafe { std::ptr::copy_nonoverlapping(x.as_ptr(), (dst + c * per) as *mut u8, x.len()) });
                }
            });
        };
        let (put, chunks) = (&put, &chunks);
        put(pp[0], chunks[0]);
        let sh = self.sh_base;
        std::thread::scope(|s| -> Result<(), String> {
            for (i, c) in chunks.iter().enumerate() {
                // buffer (i + 1) % 2 went up in the round before; it is filled while buffer i % 2 goes up
                let next = (i + 1 < chunks.len()).then(|| { let d = pp[(i + 1) % 2]; s.spawn(move || put(d, chunks[i + 1])) });
                let r = checked("cuMemcpyHtoD self y", (cuda.upload)(sh + (i * cb) as u64, pp[i % 2] as *const c_void, c.len()));
                if let Some(h) = next { h.join().map_err(|_| "self y copy worker panicked".to_string())?; }
                r?;
            }
            Ok(())
        })
    }
    unsafe fn download_all(&self, cuda: &Cuda) -> Result<Vec<u8>, String> {
        dev::sync(cuda)?;
        let total = self.words as usize * 4;
        let cb = self.tmp_words as usize * 4;
        if !dir_pipe() || total < 2 * cb { return dev::download(cuda, self.sh_base, total); }
        let mut pins = self.wpins.lock().map_err(|_| "direction weight buffers poisoned")?;
        while pins.len() < 2 { pins.push(crate::pinned::PinnedBuf::new(cuda, cb)?); }
        let pp = [pins[0].as_mut_ptr() as usize, pins[1].as_mut_ptr() as usize];
        let mut v = vec![0u8; total];
        let base = v.as_mut_ptr() as usize;
        let sh = self.sh_base;
        std::thread::scope(|s| -> Result<(), String> {
            let mut busy: [Option<std::thread::ScopedJoinHandle<()>>; 2] = [None, None];
            let (mut off, mut i) = (0usize, 0usize);
            while off < total {
                let n = cb.min(total - off);
                // a chunk buffer is read again only after the worker copying out of it has ended
                if let Some(h) = busy[i % 2].take() { h.join().map_err(|_| "result copy worker panicked".to_string())?; }
                let src = pp[i % 2];
                checked("cuMemcpyDtoH result", (cuda.download)(src as *mut std::ffi::c_void, sh + off as u64, n))?;
                let dst = base + off;
                busy[i % 2] = Some(s.spawn(move || {
                    std::thread::scope(|t| {
                        let per = n.div_ceil(8).max(1);
                        let mut o = 0usize;
                        while o < n {
                            let k = per.min(n - o);
                            t.spawn(move || unsafe { std::ptr::copy_nonoverlapping((src + o) as *const u8, (dst + o) as *mut u8, k) });
                            o += k;
                        }
                    });
                }));
                off += n; i += 1;
            }
            for b in busy.iter_mut() { if let Some(h) = b.take() { h.join().map_err(|_| "result copy worker panicked".to_string())?; } }
            Ok(())
        })?;
        Ok(v)
    }
}

/// hunt-429 (the central line): one allocation's direction-weight preparation, moved out of weights_prep's closure so it
/// runs without a card. Returns its sorted, merged unstable byte ranges [a, z) and its field weight r (rms of its valid
/// stable words; the inverse for the window-end y of --direction field). `history`, `residue`, `relocs` and `dead` are
/// this allocation's entries in the boundary's pointer history, the step's declared residue, the relocation slots and
/// AD_DOT_DEAD (the caller passes `dead` only when stable and inv, where the closure read dead_ranges()). The body is the
/// closure's, word for word; a429_mask_tests::prep_and_span_equal_fold9_code compares it with a copy of fold 9's closure.
#[allow(clippy::too_many_arguments)]
fn weights_alloc_prep(data: &[u8], off: usize, bytes: usize, stable: bool, inv: bool, fwd: bool, history: Option<&PointerHistory>,
                      residue: Option<&[(usize, usize)]>, relocs: Option<&[(usize, usize)]>, dead: Option<&[(usize, usize)]>)
                      -> (Vec<(usize, usize)>, f32) {
    let valid = |v: f32| v.is_finite() && v <= WEIGHT_MAX && v > 1e-30;
    let rd = |k: usize| f32::from_le_bytes(data[k..k + 4].try_into().unwrap()).abs();
    let mut rg: Vec<(usize, usize)> = Vec::new();
    if stable {
        if let Some(h) = history { rg.extend(h.keys().map(|&w| (8 * w, 8 * w + 8))); }
        if let Some(r) = residue { rg.extend(r.iter().map(|&(lo, n)| (lo, lo + n))); }
        if let Some(r) = relocs { rg.extend(r.iter().map(|&(lo, n)| (lo, lo + n))); }
        if inv { if let Some(r) = dead { rg.extend(r.iter().map(|&(lo, hi)| (lo, hi.min(bytes)))); } }
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
            let un = (p < rg.len() && rg[p].0 <= o) || (stable && address_shaped(data, off, bytes, o));
            let v = rd(k); if valid(v) && !un { s2 += (v as f64) * (v as f64); c += 1; }
        }
        if c == 0 || s2 == 0.0 { 0.0 } else if inv { (1.0 / (s2 / c as f64).sqrt()) as f32 } else { (s2 / c as f64).sqrt() as f32 }
    } else { 0.0 };
    (rg, r)
}
/// a429 (FOLD round 7, for a development line): true when the 8-byte word of the allocation that holds byte `o` has the shape
/// of a user-space address in ANY process (2^32 <= word < 2^47). With --stable-dir 1 both of its 4-byte words get zero
/// weight. THE BREAKAGE THIS PREVENTS: the pointer history lists every word the boundary reading translated, and that
/// reading translates any 8-byte word that lies inside a live device allocation of THIS process, model numbers included
/// (a float beside a small positive denormal reads as 0x00007f..: allocation 429 at 120 km holds such pairs). Where the
/// process put its device memory then decided which of those words y weighs, so the check's direction, its self-dot
/// count (102034904, ..05 or ..06 for allocation 429 in about one run in ten) and at times its result words changed
/// from run to run with a byte-identical derivative (a test machine <local path>). The bytes are the model's
/// own and equal in every process, so this test gives the same answer in every process; a true address always has
/// this shape, so every word the history can name by its own value is covered as before.
#[inline]
fn address_shaped(data: &[u8], off: usize, bytes: usize, o: usize) -> bool {
    let b = o & !7;
    if b + 8 > bytes { return false; }
    let w = u64::from_le_bytes(data[off + b..off + b + 8].try_into().unwrap());
    w >> 32 != 0 && w >> 47 == 0
}

/// species planes (AD_DOT_PLANES): one planed allocation's plane table. Per plane (byte lo, byte hi inside the allocation,
/// field weight), the field weight being weights_alloc_prep's rule on the plane's own words alone: the rms of its valid
/// stable words (the inverse on the window-end side of --direction field; 0 when neither field side is on), valid =
/// finite and above 1e-30, at most WEIGHT_MAX only when `cap`. The rest weight is the same rule, with the cap, over the
/// allocation's words outside every plane. `rg` is the allocation's merged unstable byte ranges (weights_alloc_prep's);
/// address shapes are read in the allocation's own 8-byte frame, as weights_span reads them.
#[allow(clippy::too_many_arguments)]
fn plane_table(data: &[u8], off: usize, bytes: usize, stable: bool, inv: bool, fwd: bool, rg: &[(usize, usize)],
               planes: &[(usize, usize)], cap: bool) -> PlaneW {
    let rd = |k: usize| f32::from_le_bytes(data[k..k + 4].try_into().unwrap()).abs();
    let words = (bytes / 4) * 4;
    let weight = |s2: f64, c: u64| if !(inv || fwd) || c == 0 || s2 == 0.0 { 0.0 } else if inv { (1.0 / (s2 / c as f64).sqrt()) as f32 } else { (s2 / c as f64).sqrt() as f32 };
    let mut tab = Vec::with_capacity(planes.len());
    let mut rest = (0f64, 0u64);
    let (mut p, mut q, mut o) = (0usize, 0usize, 0usize);
    let mut acc = (0f64, 0u64);
    while o < words {
        while p < rg.len() && rg[p].1 <= o { p += 1; }
        while q < planes.len() && planes[q].1 <= o {
            tab.push((planes[q].0, planes[q].1, weight(acc.0, acc.1))); acc = (0.0, 0); q += 1;
        }
        let inp = q < planes.len() && planes[q].0 <= o;
        let un = (p < rg.len() && rg[p].0 <= o) || (stable && address_shaped(data, off, bytes, o));
        let v = rd(off + o);
        let capped = !inp || cap;
        let ok = v.is_finite() && v > 1e-30 && (!capped || v <= WEIGHT_MAX);
        if ok && !un { if inp { acc.0 += v as f64 * v as f64; acc.1 += 1; } else { rest.0 += v as f64 * v as f64; rest.1 += 1; } }
        o += 4;
    }
    while q < planes.len() { tab.push((planes[q].0, planes[q].1.min(words), weight(acc.0, acc.1))); acc = (0.0, 0); q += 1; }
    (tab, cap, weight(rest.0, rest.1))
}

/// machine: AD_DIR_PIPE=0 keeps the dot test's direction weights on the old per-chunk path (see weights_piped).
/// sixty (AD_DOT_DEAD=<file.json>, off unless set): byte ranges of work memory that are DEAD at the end of a step
/// (the next step overwrites them before anything reads them). The y direction at the window end gives them zero weight,
/// as it does for declared residue and address words: a cotangent on a word nothing reads is outside the operator.
/// File: {"ranges":[{"allocation":id,"offset":bytes,"bytes":n,"name":"..."}]}. THE BREAKAGE THIS PREVENTS: the dot weighed
/// a few words of the transport limiter's flux arrays (those under the 2e5 value mask), where float32 cancellation between
/// the high-order and the upwind flux reads 1e-4 and hides the model state (60 km: 1,102 words carried the self-dot miss).
/// The claim is proved in the same run by AD_DOT_DEAD_POISON=1: the ranges are filled with NaN before the last step and
/// every step boundary must still equal the reference.
fn dead_ranges() -> &'static HashMap<u64, Vec<(usize, usize)>> {
    static D: std::sync::OnceLock<HashMap<u64, Vec<(usize, usize)>>> = std::sync::OnceLock::new();
    D.get_or_init(|| {
        let mut m: HashMap<u64, Vec<(usize, usize)>> = HashMap::new();
        if let Ok(path) = std::env::var("AD_DOT_DEAD") {
            let v: Value = serde_json::from_slice(&fs::read(&path).unwrap_or_else(|e| panic!("AD_DOT_DEAD {path}: {e}"))).unwrap_or_else(|e| panic!("AD_DOT_DEAD {path}: {e}"));
            for r in v["ranges"].as_array().unwrap_or_else(|| panic!("AD_DOT_DEAD {path}: no ranges")) {
                let (id, lo, n) = (r["allocation"].as_u64().expect("AD_DOT_DEAD allocation"), r["offset"].as_u64().expect("AD_DOT_DEAD offset") as usize, r["bytes"].as_u64().expect("AD_DOT_DEAD bytes") as usize);
                assert!(lo % 4 == 0 && n % 4 == 0, "AD_DOT_DEAD: ranges are whole 32-bit words");
                m.entry(id).or_default().push((lo, lo + n));
            }
        }
        m })
}
/// g1: AD_DOT_DEAD_POISON is a comma list. "1" (or "fwd"): NaN in the listed primal bytes before the last forward
/// step (sixty's forward proof). "rev": NaN in the listed primal bytes in the reverse of the LAST step, after the redo has
/// rebuilt that step's end state and its copy for the redo check is taken, and before the walk-back runs: the walk-back
/// is the only reader left of the end-of-window bytes the dot gives zero weight, so equal gradient words against a leg
/// without NaN (AD_DOT_DEAD_CONTROL in run) prove the reverse never reads them. "rev-all": the same before the
/// walk-back of EVERY reverse step, and also right after the restore of each reverse step s >= 2 (the bytes a model
/// step left), so the redo of every later step is shown not to read them either (the redo check then compares).
/// THE BREAKAGE IT PREVENTS: sixty's forward proof put the NaN in before the last forward step, and the backward pass
/// restores from a boundary stored before that, so no reverse kernel ever saw the NaN (verifier, COORD.md 03:22:37Z).
fn dead_poison_mode() -> (bool, bool, bool) {
    static M: std::sync::OnceLock<(bool, bool, bool)> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        let v = std::env::var("AD_DOT_DEAD_POISON").unwrap_or_default();
        let has = |k: &str| v.split(',').any(|x| x.trim() == k);
        (has("1") || has("fwd"), has("rev") || has("rev-all"), has("rev-all"))
    })
}
static WIN_STEPS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static DEAD_REV: std::sync::Mutex<Vec<Value>> = std::sync::Mutex::new(Vec::new());
/// g1: write NaN (0x7fc00000) over every AD_DOT_DEAD range in the primal, in the reverse of `step`, at `at`
/// ("restore" = right after the step's start boundary was restored, "walk" = after the redo, before the walk-back).
unsafe fn dead_poison_rev(ex: &Execution, step: usize, at: &str) -> Result<(), String> {
    let (_, rev, all) = dead_poison_mode();
    if !rev || dead_ranges().is_empty() { return Ok(()); }
    let last = WIN_STEPS.load(std::sync::atomic::Ordering::Relaxed);
    let on = match at { "walk" => all || step == last, "restore" => all && step >= 2, _ => false };
    if !on { return Ok(()); }
    let (mut nb, mut nr) = (0usize, 0usize);
    for (id, rs) in dead_ranges() {
        let a = ex.allocations.get(id).ok_or_else(|| format!("AD_DOT_DEAD: allocation {id} is not live in the reverse of step {step}"))?;
        for &(lo, hi) in rs {
            if hi > a.bytes { return Err(format!("AD_DOT_DEAD: range [{lo}, {hi}) is outside allocation {id} ({} bytes)", a.bytes)); }
            let nan: Vec<u8> = std::iter::repeat([0x00u8, 0x00, 0xc0, 0x7f]).take((hi - lo) / 4).flatten().collect();
            dev::upload(&ex.cuda, a.address + lo as u64, &nan)?; nb += hi - lo; nr += 1;
        }
    }
    dev::sync(&ex.cuda)?;
    let e = json!({"event":"window_dead_poison_rev","step":step,"at":at,"ranges":nr,"bytes":nb,"word":"NaN 0x7fc00000"});
    eprintln!("{e}");
    if let Ok(mut v) = DEAD_REV.lock() { v.push(e); }
    Ok(())
}
fn dir_pipe() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AD_DIR_PIPE").as_deref() != Ok("0"))
}
/// machine: window timeline marks. One Instant read and one stderr line each, no device call, so a timed run keeps them.
static MARK_T0: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
pub(crate) fn mark(at: &str) {
    let t0 = *MARK_T0.get_or_init(Instant::now);
    eprintln!("{}", json!({"event":"window_mark","at":at,"t_s":t0.elapsed().as_secs_f64()}));
}
/// Partition exactly the domain used by `download_all` and `result_sha256`.
/// Allocation sizes alone omit padding; live-boundary indices also omit all
/// intermediate allocations once reverse reaches boundary zero.
fn vector_export_ranges(base_el: &HashMap<u64, i64>, words: i64) -> Result<Vec<(u64, i64, usize)>, String> {
    let mut starts: Vec<_> = base_el.iter().map(|(&id, &start)| (id, start)).collect();
    starts.sort_unstable_by_key(|x| x.1);
    if words < 0 || starts.first().map_or(words != 0, |x| x.1 != 0) {
        return Err("DEFECT: vector export arena does not begin at zero".into());
    }
    let mut ranges = Vec::new();
    for (i, &(id, start)) in starts.iter().enumerate() {
        let end = starts.get(i + 1).map_or(words, |x| x.1);
        if start < 0 || end <= start || end > words {
            return Err("DEFECT: overlapping or out-of-bounds vector export arena".into());
        }
        let bytes = usize::try_from(end - start).ok().and_then(|n| n.checked_mul(4))
            .ok_or("DEFECT: vector export byte count overflow")?;
        ranges.push((id, start, bytes));
    }
    Ok(ranges)
}

#[cfg(test)]
mod vector_export_tests {
    use super::vector_export_ranges;
    use std::collections::HashMap;

    #[test]
    fn full_arena_retains_non_live_allocations_and_padding() {
        // Only allocation 9 might be live at boundary zero. The physical
        // lengths can be smaller than these 64-word-aligned shadow spans.
        let map = HashMap::from([(9, 0), (2, 64), (80, 192)]);
        let ranges = vector_export_ranges(&map, 256).unwrap();
        assert_eq!(ranges, vec![(9, 0, 256), (2, 64, 512), (80, 192, 256)]);
        assert_eq!(ranges.iter().map(|x| x.2).sum::<usize>(), 256 * 4);
        assert!(vector_export_ranges(&HashMap::from([(9, 1)]), 64).is_err());
        assert!(vector_export_ranges(&HashMap::from([(9, 0), (2, 0)]), 64).is_err());
        assert!(vector_export_ranges(&HashMap::from([(9, 0), (2, 65)]), 64).is_err());
    }
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
    let out = argv.iter().position(|x| x == "--out").and_then(|i| argv.get(i + 1)).map(PathBuf::from);
    let r = run_window(argv);
    // resident: a resident program that stops on an error answers the request it was serving (rc 1).
    if let Err(e) = &r { crate::resident::fail(e); }
    // multirank-2: the --local-topology 1 receipt (built N, equal to pinned M, per function), success or not.
    if let Some(rec) = deriv::local_topology_receipt() {
        eprintln!("{rec}");
        if let Some(o) = out { let _ = fs::write(o.join("local-topology.json"), serde_json::to_vec_pretty(&rec).unwrap()); }
    }
    r
}
fn run_window(argv: Vec<String>) -> Result<(), String> {
    let mut a = HashMap::new();
    let mut it = argv.into_iter();
    while let Some(k) = it.next() {
        let key = k.strip_prefix("--").ok_or_else(|| format!("unexpected argument {k}"))?.to_owned();
        a.insert(key, it.next().ok_or_else(|| format!("missing value for {k}"))?);
    }
    let t_start = Instant::now();
    let config = PathBuf::from(req(&a, "config")?);
    mark("start");
    // hunt-120-start: an unreadable AD_WORD_PROBE file or word is refused here, not after the tangent pass.
    crate::word_probe::config()?;
    // hunt-60-repeat: an unreadable AD_REV_DIGEST / AD_REV_DIGEST_CMD / AD_DOT_SELF_YPOW2 word is refused here, not after the tangent pass.
    crate::rev_digest::config()?; crate::rev_digest::ypow2()?;
    let steps: usize = req(&a, "steps")?.parse().map_err(|_| "invalid --steps")?;
    let out = PathBuf::from(req(&a, "out")?);
    // mem-oocore: --spill-dir is the file tier of the boundary store (written behind, read ahead); --bstore as before.
    let bdir = PathBuf::from(match opt(&a, "spill-dir") { Some(d) => d, None => req(&a, "bstore")? });
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    fs::create_dir_all(&bdir).map_err(|e| e.to_string())?;
    let seeds: Vec<u64> = opt(&a, "seeds").unwrap_or("1,2").split(',').filter_map(|s| s.trim().parse().ok()).collect();
    // AD_DOT_PLANES=scalars is written from the package's own field map here; an unreadable AD_DOT_PLANES is
    // refused here, before any card work
    if std::env::var("AD_DOT_PLANES").ok().as_deref().map(str::trim) == Some("scalars") { let _ = PLANES_FROM_MAP.set(planes_from_map(&config, steps)); }
    plane_spec()?;
    let bytes = fs::read(&config).map_err(|e| format!("cannot read {}: {e}", config.display()))?; mark("plan_read");
    let envelope: Value = serde_json::from_slice(&bytes).map_err(|e| format!("invalid plan JSON: {e}"))?; mark("plan_json");
    drop(bytes);
    if envelope.get("schema").and_then(|x| x.as_str()) != Some("ad.binder.plan/v1") || envelope.get("complete").and_then(|x| x.as_bool()) != Some(true) {
        return Err("refusing incomplete or unknown plan".into());
    }
    let concat_checked = crate::wrules::concat_host_checked(&envelope);
    let plan: Plan = serde_json::from_value(envelope).map_err(|e| format!("invalid sealed plan: {e}"))?; mark("plan_typed");
    // resident: --x0-planted DIR (default off) runs this package as x0_plant.py's moved package of the start DIR
    // was planted from (ad-binder x0-plant): the named allocations' initial bytes and step_01 uploads are swapped for the
    // planted payloads before anything reads the plan (resident::install). It needs --x0-dir (the empty move DIR holds):
    // THE BREAKAGE THIS PREVENTS is a planted start compared with the first start's recording, refused at step 1.
    let mut plan = plan;
    // resident: --resident DIR (default off: None, and nothing below changes) keeps this program up after its
    // gradient and serves later ones from DIR (crate::resident::Resident; the loop around the window body below).
    let mut rz = crate::resident::Resident::from_words(&a)?;
    let x0_planted = match opt(&a, "x0-planted") {
        Some(d) => {
            if opt(&a, "x0-dir").is_none() { return Err("--x0-planted needs --x0-dir (DIR/x0-empty.json): a planted start is not the recording's".into()); }
            let r = match rz.as_mut() {
                Some(z) => z.swap(&mut plan, config.parent().unwrap_or(Path::new(".")), Some(d))?.ok_or("--x0-planted: nothing swapped")?,
                None => crate::resident::install(&mut plan, config.parent().unwrap_or(Path::new(".")), Path::new(d))?,
            };
            eprintln!("{}", json!({"event":"window_x0_planted","info":r})); mark("x0_planted"); Some(r)
        }
        None => None,
    };
    crate::peer::set_host_unchecked(&plan.peer_host_unchecked);
    crate::preflight(&plan, steps)?;
    let base = config.parent().unwrap_or(Path::new(".")).to_owned();
    crate::verify_fixed_argument_sources(&plan, &base)?;
    crate::verify_copy_indexer_sources(&plan, &base)?; mark("plan_verified");
    let reference = plan.reference.as_deref().map(|p| crate::path(&base, p)).ok_or("window needs the ordinary-forward reference")?;
    let refs = crate::read_references(&reference, &plan, steps)?; mark("references");
    let mut ex = unsafe { Execution::new(None)? }; mark("cuda_ctx");
    // mem-oocore: the VRAM / host budget goes on before the first device allocation of the run (ex.prepare).
    let oocore = unsafe { crate::spill::install(&mut ex.cuda, opt(&a, "vram-budget-gb"), opt(&a, "host-budget-gb"), opt(&a, "spill-mode"))? };
    if oocore || opt(&a, "host-budget-gb").is_some() { eprintln!("{}", json!({"event":"window_out_of_core","info":crate::spill::receipt()})); }
    crate::spill::set_cold(oocore && (opt(&a, "spill-cold") == Some("1") || std::env::var("AD_SPILL_COLD").as_deref() == Ok("1")));
    crate::spill::phase("load");
    // lane tl-host-syncs: AD_SYNC_SITES=1 also counts every driver wait (the C ABI's counting trampolines, one atomic add per call).
    if dev::sites::on() { unsafe { capi::install_counters(&mut ex.cuda); } }
    ex.address_bearing = plan.address_bearing.iter().copied().collect();
    unsafe { ex.prepare(&plan, &base)? };
    let mut frame = plan.physics.as_ref().map(ReplayFrame::new).transpose()?;
    mark("prepared");
    if let (Some(f), Some(spec)) = (frame.as_ref(), plan.physics.as_ref()) { ex.host_state.insert(spec.frame_state_id, f.snapshot()?)?; }
    unsafe { start_staging(&mut ex, &plan, &base, &frame); }
    let t_loaded = t_start.elapsed().as_secs_f64(); mark("staging_started");
    let spec_check = check_spec_hashes(Path::new(req(&a, "kernels")?))?; mark("spec_hashes");
    let mut w = unsafe { Win::new(&ex, &plan, &a)? }; mark("win_new");
    unsafe { w.prefetch_first(&plan, &base, &frame)?; }
    if let Some(p)=opt(&a,"vector-gate") {
        let sizes=w.shadows.iter().map(|(id,s)|(*id,s.bytes)).collect();
        let pair=window_vectors::Pair::load(Path::new(p),&crate::exec::digest(&fs::read(&config).map_err(|e|e.to_string())?),&sizes)?;
        if seeds!=vec![pair.seed] {return Err("DEFECT: --vector-gate requires its single declared seed".into());}
        if opt(&a,"direction") != Some("explicit") || ["tangent-dir","tangent-taylor-dir","param-tl","ext-driver"].iter().any(|k| opt(&a,k).is_some()) {
            return Err("DEFECT: --vector-gate requires --direction explicit and no other input-vector override".into());
        }
        w.vectors=Some(pair);
    }
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
    // (or any other perturbed forward) would read the base trajectory's halo bytes from its peer and replay a wrong forward.
    if let Some(p) = crate::peer::get() {
        for k in ["taylor", "taylor-dir", "ext-driver", "params", "param-fd", "param-tl", "x0-dir"] {
            // ranks (R2): --taylor-dir with --tangent-taylor-dir 1 (alias --tangent-dir) only seeds the base tangent and
            // weights the objective gradient. That mode requires --tangent-only 1 and refuses --fd-only, and --taylor (the
            // perturbed rows) stays refused here, so no perturbed forward runs and the named breakage cannot happen.
            if k == "taylor-dir" && (opt(&a, "tangent-taylor-dir") == Some("1") || opt(&a, "tangent-dir") == Some("1")) { continue; }
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
    // integrator: a concatenate module that is not float and not integer (its type may not be established at all)
    // and whose every launch only feeds a host check is inactive, not refused (wrules::mark_host_checked names the breakage).
    crate::wrules::mark_host_checked(&mut w.concat_types, &concat_checked);
    eprintln!("{}", json!({"event":"concat_module_types","types":w.concat_types}));
    let cuda = &ex.cuda as *const Cuda;
    if std::env::var("AD_PLANT").is_ok() && plant().is_none() { return Err("AD_PLANT wants <kernel>:<argument name or number>:<factor>".into()); }
    let mut receipt = json!({"event":"window_receipt","plan":config.display().to_string(),"steps":steps,"seeds":seeds,
        "args":a,"stages":{"plan_loaded_s":t_loaded},"tolerance":DOT_F32,"started_unix":now(),"spec_sha256":spec_check});
    if let Some(v)=&w.vectors {receipt["explicit_vectors"]=json!({"sha256":v.sha256,"seed":v.seed,"x_segments":v.x.len(),"y_segments":v.y.len()});}
    // speed (fold7): AD_PLANT_RECOMPUTE (test only, see plant_recompute): the receipt says planted from the start
    if std::env::var("AD_PLANT_RECOMPUTE").is_ok() {
        let (ps, pal, pwd) = (*plant_recompute()).ok_or("AD_PLANT_RECOMPUTE wants <step>[:<allocation>[:<word>]], step 1 or more")?;
        if ps > steps { return Err(format!("AD_PLANT_RECOMPUTE: step {ps} is outside this window of {steps} steps")); }
        if plant().is_some() { return Err("AD_PLANT and AD_PLANT_RECOMPUTE are two tests: set one".into()); }
        receipt["planted_error"] = json!({"switch":"AD_PLANT_RECOMPUTE","step":ps,"allocation":pal,"word":pwd,"side":PLANT_RC_SIDE,
            "detail":"the allocation, the word and its bits before and after are in the stderr event window_plant_recompute, and in window.json as written at the planted step","valid":false});
        if let Ok(mut g) = PLANT_RC_OUT.lock() { *g = Some((out.join("window.json"), receipt.clone())); }
    }
    // speed live: AD_PLANT_LEFT_OUT (test only, see plant_left_out): the receipt says planted from the start
    if std::env::var("AD_PLANT_LEFT_OUT").is_ok() {
        let al = (*plant_left_out()).ok_or("AD_PLANT_LEFT_OUT wants <allocation> or largest")?;
        if plant().is_some() || plant_recompute().is_some() { return Err("AD_PLANT_LEFT_OUT is a test of its own: set no other AD_PLANT*".into()); }
        if opt(&a, "bstore-live").is_none_or(|v| v == "0") { return Err("AD_PLANT_LEFT_OUT tests --bstore-live 1: give it".into()); }
        w.cs_plant_ask = Some(al);
        receipt["planted_error"] = json!({"switch":"AD_PLANT_LEFT_OUT","allocation":al,"side":PLANT_LO_SIDE,
            "detail":"the allocation and its size are in the stderr event window_plant_left_out, and in window.json as written at the first intermediate boundary stored","valid":false});
        if let Ok(mut g) = PLANT_RC_OUT.lock() { *g = Some((out.join("window.json"), receipt.clone())); }
    }
    if !w.params.is_empty() {
        receipt["params"] = json!(w.params.iter().map(|p| json!({"name":p.name,"family":p.fam,"slot":p.slot,"p0":p.p0,
            "p0_bits":format!("0x{:08x}", p.p0.to_bits())})).collect::<Vec<_>>());
        receipt["param_forward_overrides"] = json!(ex.kernel_override.keys().collect::<Vec<_>>());
    }
    let log = |v: Value| eprintln!("{v}");
    #[cfg(not(hexad_ext_driver))]
    if opt(&a, "ext-driver").is_some() {
        return Err(EXT_DRIVER_ABSENT.into());
    }
    #[cfg(hexad_ext_driver)]
    if opt(&a, "ext-driver").is_some() {
        let r = unsafe { ext_driver::run(&mut w, &mut ex, &plan, &base, &mut frame, &refs, &a, &out, &bdir, steps, receipt) };
        if let Err(e) = &r { eprintln!("{}", json!({"event":"ext_driver_error","error":e})); }
        return r;
    }
    // resident: the window body below runs once per request (once, as before, without --resident). A later
    // request's words are the command's with the request's put over them (Resident::next, merge_words); out, the store
    // folder, the objective and the planted start may change, nothing that the loaded state was made from.
    if rz.is_some() && crate::peer::get().is_some() { return Err("--resident on a rank plan: not served".into()); }
    if rz.is_some() && std::env::var("AD_PIN_KEEP").as_deref() != Ok("0") { PIN_KEEP_RESIDENT.store(true, std::sync::atomic::Ordering::Relaxed); }
    let words0 = if rz.is_some() { a.clone() } else { HashMap::new() };
    let receipt0 = if rz.is_some() { receipt.clone() } else { Value::Null };
    let mut b0_keep: Option<Boundary> = None;
    // keep-forward (AD_KEEP_FORWARD, default off): the forward of the request before, for a request of the same start
    let mut kf: Option<KeptForward> = None;
    let (mut a, mut out, mut bdir, mut x0_planted, mut t_start) = (a, out, bdir, x0_planted, t_start);
    loop {
    if let Some(z) = rz.as_ref() { receipt["resident"] = json!({"dir":z.dir.display().to_string(),"request":z.name,"index":z.index}); }
    // dist47: Some(sentence) when this request's start is one the forecast itself stops at (the base forward of a
    // kept-loaded program): the request is answered rc 1 and the program goes on to the next one (below).
    let refused: Option<String> = 'request: {
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
        // tune mem: --params runs with --checkpoints. A parameter's reverse is part of reverse_step (the carrier
        // image adds its leaf adjoints into its own module global while the step is reversed), so it needs no boundary
        // beyond the step's own start and end states, which sweep_revolve hands to reverse_step from a checkpoint or a
        // recompute. The objective reverse already took that path; the --grad-all reverses take it below. With
        // --params-unchecked 1 the recomputed stretches are not compared with the recorded reference (advance), as the
        // base forward is not: each is still compared, one step on, with the state the reverse holds (reverse_step's redo).
        // a12 speed: so does --x0-dir (a gradient at a moved start). THE DEFECT: a moved start was refused with
        // --checkpoints, so its gradient had to keep every step end (10.5 GB each at the 3 km storm box: 230 GiB of host
        // at 59 steps, most of them on disk) and every new start of a run of gradients was recorded again instead (1,749 s
        // of recording for a 694 s gradient, a test machine, 2026-11-12). The re-run stretches of a moved start are compared
        // with nothing recorded, as its base forward is not (base_unchecked); the redo check of every reversed step
        // still compares the re-made step end with the state the reverse holds. Runs without --x0-dir are unchanged.
        w.recompute_unchecked = ckpts.is_some() && ((params_unchecked && opt(&a, "params").is_some()) || opt(&a, "x0-dir").is_some());
        // a12 s60-rerun (a development line): AD_RECOMPUTE_UNCHECKED=1 (with --checkpoints; default off) runs every re-run step of
        // a checkpointed sweep as a moved start's re-run steps are run: not compared with the recorded reference. THE WASTE
        // THIS REMOVES: at 60 km a checked re-run step downloads its 26.55 GB end state for the boundary hash
        // (host_digest_prof download_s 1.5 to 4.0 s of a 4.24 s re-run step; 56 re-run steps, 237.65 s, in the 29-step
        // sweep of pk60e29p-20230331, rel-sixty run l29-m5fill). What still checks a re-run state: the redo check of every
        // reversed step compares the re-made end state with the state the sweep holds, back to the window end, which the
        // tangent forward checked against the recording. The same arithmetic runs; only the compare with the recording goes.
        if ckpts.is_some() && rerun_unchecked_word() { w.recompute_unchecked = true; }
        // speed (storm): a command that names a host-memory budget (--bstore-ram-gib G, G > 0) keeps its checkpoints
        // in that memory, as the store-every-step window does, and writes whole files only past the budget. THE DEFECT:
        // with --checkpoints the changed-chunk files were on by default and the host-memory tier is never used with
        // them, so the 3 km storm window s27-S (--checkpoints sqrt --bstore-ram-gib 150) wrote 273.9 GB to files and
        // read them back: store_s 178.2 s and load_s 237.4 s of a 1,726 s window, ram.boundaries 0. --bstore-delta 1
        // asks for the changed-chunk files by name (a host too small for the checkpoints); --bstore-delta 0 is as before;
        // a command without --bstore-ram-gib is unchanged. The same bytes are restored either way.
        let ram_asked = opt(&a, "bstore-ram-gib").and_then(|v| v.parse::<f64>().ok()).is_some_and(|g| g > 0.0);
        w.delta_store = ckpts.is_some() && match opt(&a, "bstore-delta") { Some("0") => false, Some(_) => true, None => !ram_asked };
        w.bs_ram_budget = match opt(&a, "bstore-ram-gib").and_then(|v| v.parse::<f64>().ok()) {
            Some(g) => (g.max(0.0) * (1u64 << 30) as f64) as usize,
            None => (mem_available() as f64 * 0.4) as usize,
        };
        // speed ckpt60: --bstore-content 1 (with --checkpoints) keeps every boundary of the window as a state of the
        // content store (cstore.rs) in host memory: no changed-chunk files, no whole copies. --bstore-content 0, or no
        // such word, is as before. The same bytes are restored and compared either way.
        w.cs_on = ckpts.is_some() && opt(&a, "bstore-content").is_some_and(|v| v != "0");
        if w.cs_on { w.delta_store = false; w.bs_ram_budget = 0; }
        // speed live: --bstore-live 1 (with --bstore-content 1, default off) stores an intermediate boundary without
        // the allocations the plan does not carry (plan_not_carried). 60 km pk60e29p-20230331 plan.json: 26.55 GB a
        // boundary, of which 6.31 GB in 23 non-carried allocations (three pool blocks of 2.15, 2.15 and 1.23 GB made in
        // steps 2, 3 and 6, and twenty of 39 MB made in step 2); 20.24 GB is carried. The same words are restored to every
        // carried allocation; the non-carried ones start the re-made step from zeros and are never compared.
        w.cs_cut = match opt(&a, "bstore-live") {
            None | Some("0") => None,
            Some(_) if !w.cs_on => return Err("--bstore-live 1 needs --bstore-content 1 (with --checkpoints N)".into()),
            Some(_) => Some(plan_not_carried(&plan)),
        };
        w.cs_cut_steps = steps;
        // step-hold: --step-hold FILE (default off; with --checkpoints and --bstore-ram-gib G) holds every
        // intermediate boundary as the spans the hold set FILE keeps (step_hold.rs), in page-locked host memory, so the
        // sweep re-runs no step. The checkpoint count is then steps - 1 (revolve's every-boundary schedule). Without the
        // word nothing changes.
        let mut hold_receipt = Value::Null;
        let ckpts = match opt(&a, "step-hold") {
            None | Some("0") => ckpts,
            Some(f) => {
                if ckpts.is_none() { return Err("--step-hold needs --checkpoints (any count: every intermediate boundary is then held)".into()); }
                if w.cs_on { return Err("--step-hold keeps its own spans: not with --bstore-content 1".into()); }
                if w.delta_store || w.bs_ram_budget == 0 { return Err("--step-hold holds in page-locked host memory: give --bstore-ram-gib G (and not --bstore-delta 1)".into()); }
                let bytes = fs::read(f).map_err(|e| format!("--step-hold {f}: {e}"))?;
                let set = crate::step_hold::HoldSet::parse(&bytes)?;
                hold_receipt = json!({"on":true,"file":f,"set":set.name,"set_sha256":set.sha256,"const_allocations":set.consts.len(),"drop_ranges":set.drop.values().map(|v| v.len()).sum::<usize>(),"drop_bytes":set.drop_bytes,
                    "rule":"every intermediate boundary is held as the live state less the set's constant allocations (boundary 0 holds them; checked equal after each sweep) and its dropped ranges (written before read by each step); boundary 0 and the window end stay whole; the redo check compares every held word"});
                w.shold = Some(std::sync::Arc::new(set));
                w.hold_steps = steps;
                Some(steps.saturating_sub(1))
            },
        };
        let mut live_receipt = Value::Null;
        if let Some(c) = &w.cs_cut {
            let sizes: HashMap<u64, usize> = plan.allocations.iter().map(|x| (x.id, x.bytes)).chain(plan.blocks.values().flatten()
                .filter_map(|o| if let Operation::Allocate { id, bytes, .. } = o { Some((*id, *bytes)) } else { None })).collect();
            let mut ids: Vec<u64> = c.iter().copied().collect(); ids.sort_unstable();
            live_receipt = json!({"on":true,"rule":"an intermediate boundary's state holds the allocations plan.carried names; the others (producer-certified scratch) are not stored, not compared by the redo check, and start a re-made step from zeros; boundary 0 and the window end stay whole",
                "left_out_allocations":ids.len(),"left_out_bytes_most":ids.iter().map(|i| sizes.get(i).copied().unwrap_or(0)).sum::<usize>(),"left_out_ids":ids});
        }
        let rv = ckpts.map(|c| Revolve::new(steps, c));
        // step-hold: the forward stores every intermediate boundary (see sweep_revolve_body)
        let descent: Option<Vec<usize>> = rv.as_ref().map(|r| if w.shold.is_some() { (1..steps).collect() } else { r.descent(steps, ckpts.unwrap()) });
        let mut ck: BTreeMap<usize, Boundary> = BTreeMap::new();
        let mut sweeps: Vec<Value> = Vec::new();
        if let (Some(r), Some(c)) = (&rv, ckpts) {
            receipt["checkpointing"] = json!({"checkpoints":c,"schedule":"revolve (binomial, cost-optimal DP)","first_descent":descent,
                "planned_recompute_steps_per_sweep":r.cost[steps][c],"bstore":if w.cs_on { "content store (checkpoints as 1 MiB pieces in host memory, a piece equal to the state before it is shared; the redo check by SHA-256 of the pieces)" } else if w.delta_store { "changed-chunk (512 B, vs boundary 0)" } else { "whole" }});
        }
        if !live_receipt.is_null() { receipt["checkpointing"]["bstore_live"] = live_receipt; }
        if !hold_receipt.is_null() { receipt["checkpointing"]["step_hold"] = hold_receipt; }
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
        // longwin: optional "dense" blocks of a --taylor-dir / --x0-dir file: (allocation, byte offset, f32 LE bytes).
        let read_dense = |p: &str| -> Result<Vec<(u64, usize, Vec<u8>)>, String> {
            let doc: Value = serde_json::from_slice(&fs::read(p).map_err(|e| format!("{p}: {e}"))?).map_err(|e| format!("{p}: {e}"))?;
            let dir = Path::new(p).parent().unwrap_or(Path::new(".")).to_owned();
            let mut v = Vec::new();
            for b in doc["dense"].as_array().map(|x| x.as_slice()).unwrap_or(&[]) {
                let (Some(al), Some(off), Some(f)) = (b["alloc"].as_u64(), b["offset"].as_u64(), b["file"].as_str()) else { return Err(format!("{p}: dense block wants alloc, offset, file")) };
                let bytes = fs::read(dir.join(f)).map_err(|e| format!("{p}: dense file {f}: {e}"))?;
                if off % 4 != 0 || bytes.is_empty() || bytes.len() % 4 != 0 { return Err(format!("{p}: dense block {al}:{off} ({f}): offset and length must be whole float32 words")); }
                if bytes.chunks_exact(4).any(|c| !f32::from_le_bytes(c.try_into().unwrap()).is_finite()) { return Err(format!("{p}: dense file {f} holds a non-finite word")); }
                v.push((al, off as usize, bytes));
            }
            Ok(v)
        };
        // a dense block must lie inside a boundary-0 allocation whose words there are floats (the refuse_index_words rule)
        let check_dense = |what: &str, b: &Boundary, data: &[u8], blocks: &[(u64, usize, Vec<u8>)]| -> Result<(), String> {
            for (al, off, bytes) in blocks {
                let &(_, bo, bn) = b.index.iter().rev().find(|x| x.0 == *al).ok_or_else(|| format!("{what}: dense allocation {al} is not in boundary 0"))?;
                if off + bytes.len() > bn { return Err(format!("{what}: dense block {al}:{off} (+{} bytes) is outside allocation {al} ({bn} bytes)", bytes.len())); }
                let (mut nz, mut il) = (0u64, 0u64);
                for (c, d) in data[bo + off..bo + off + bytes.len()].chunks_exact(4).zip(bytes.chunks_exact(4)) {
                    if d == [0u8; 4] { continue; }
                    let bits = u32::from_le_bytes(c.try_into().unwrap());
                    if bits & 0x7fff_ffff != 0 { nz += 1; }
                    if bits & 0x7f80_0000 == 0 && bits & 0x007f_ffff != 0 { il += 1; }
                }
                if 2 * il > nz { return Err(format!("{what}: dense block {al}:{off}: {il} of the {nz} nonzero words it moves hold integers at boundary 0, not a float field (built for another recording?)")); }
            }
            Ok(())
        };
        let tdense: Vec<(u64, usize, Vec<u8>)> = match opt(&a, "taylor-dir") { Some(p) => read_dense(p)?, None => Vec::new() };
        let x0dense: Vec<(u64, usize, Vec<u8>)> = match opt(&a, "x0-dir") { Some(p) => read_dense(p)?, None => Vec::new() };
        let dump_steps: Vec<usize> = opt(&a, "dump-steps").unwrap_or("").split(',').map(str::trim).filter(|x| !x.is_empty())
            .map(|x| x.parse::<usize>().map_err(|_| format!("--dump-steps {x}: a step number"))).collect::<Result<_, String>>()?;
        if dump_steps.iter().any(|&s| s == 0 || s > steps) { return Err(format!("--dump-steps: steps must lie in 1..={steps}")); }
        // longwin: --dump-at step:allocation:byte_offset:bytes,... (default off): words read at ONE step boundary each,
        // right after that step's tangent. Between step boundaries the model state does not live in the boundary-0
        // allocations: the recorder's alloc-map.json "boundaries" names a different allocation and offset for every field
        // at every boundary, and a boundary-0 allocation holds stale words until the package end. Files are named as
        // --dump-steps names them: state-s{step}-{allocation}-{offset}.f32 and tl-s{step}-{allocation}-{offset}.f32.
        let dump_at: Vec<(usize, u64, usize, usize)> = opt(&a, "dump-at").unwrap_or("").split(',').map(str::trim).filter(|x| !x.is_empty()).map(|x| {
            let q: Vec<&str> = x.split(':').collect();
            if q.len() != 4 { return Err(format!("--dump-at {x}: want step:allocation:byte_offset:bytes")); }
            let n = |i: usize, what: &str| q[i].parse::<u64>().map_err(|_| format!("--dump-at {x}: {what}"));
            let r = (n(0, "step")? as usize, n(1, "allocation id")?, n(2, "byte offset")? as usize, n(3, "bytes")? as usize);
            if r.0 == 0 || r.0 > steps { return Err(format!("--dump-at {x}: the step must lie in 1..={steps}")); }
            if r.2 % 4 != 0 || r.3 % 4 != 0 || r.3 == 0 { return Err(format!("--dump-at {x}: offset and bytes must be whole float32 words")); }
            Ok(r)
        }).collect::<Result<_, String>>()?;
        // anyparam: --x0-dir F [--x0-h h] (default off): the window's base trajectory starts from the recorded
        // start state + h * F's terms (the --taylor-dir file format), i.e. a gradient at a moved initial state. The base
        // forward is then not compared with the recorded reference (it differs by design), like --params-unchecked 1.
        let x0: Option<(f64, Value, Vec<(u64, usize, f32)>)> = match opt(&a, "x0-dir") {
            Some(p) => { let h = match opt(&a, "x0-h") { Some(v) => v.parse::<f64>().map_err(|_| format!("--x0-h {v}: a number"))?, None => 1.0 };
                         let (n, t) = read_terms(p)?; Some((h, n, t)) },
            None => None,
        };
        let unchecked = params_unchecked || x0.is_some();
        // a12 speed: a moved start's step-end digests are read by nothing. THE DEFECT: with --x0-dir no step end is compared
        // with the recording (there is none for a moved start), yet every forward step still downloaded its end state and
        // read it for the recording's address history: 5.8 to 9.3 s of a 6.4 to 9.9 s forward step whose launches take 0.4 s
        // (3 km storm box, a test machine, run k12-x0p, forward_step_prof finish_step_s and host_digest_prof, 2026-11-13).
        // Now such a run skips them as a Taylor row does (unchecked_digest_skip): no card word is read or written by the
        // digest, so the gradient's bytes are the same. AD_X0_DIGEST=1 keeps the digests (the pair's base leg).
        let x0_digest_skip = x0.is_some() && std::env::var("AD_X0_DIGEST").as_deref() != Ok("1");
        if x0_digest_skip { ex.unchecked_digest_skip = true; receipt["x0_digest_skip"] = json!(true); }
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
        // ranks: --dump-state @FILE and --dump-grad @FILE read the list from FILE. A rank launcher replaces {rank} in
        // the file name; the lists differ per rank (allocation ids), so they cannot be given inline to every rank.
        let list_arg = |k: &str| -> Result<String, String> { match opt(&a, k) {
            Some(v) if v.starts_with('@') => fs::read_to_string(&v[1..]).map(|s| s.trim().to_owned()).map_err(|e| format!("--{k} {v}: {e}")),
            Some(v) => Ok(v.to_owned()), None => Ok(String::new()) } };
        let dump_state_arg = list_arg("dump-state")?;
        let dump_grad_arg = list_arg("dump-grad")?;
        let dump_state: Vec<(u64, usize, Option<usize>)> = dump_state_arg.as_str().split(',').map(str::trim).filter(|x| !x.is_empty()).map(|x| {
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
        receipt["mesh_constants"] = json!(w.mesh_constants);
        if w.fct_frozen {
            receipt["approximation"] = json!("FCT limiter coefficients frozen from the forward trajectory (constants in TL and AD): fct_scale's scale_in/scale_out carry zero tangent and drop their cotangent; fct_minmax_source differentiates source_old only. Coordinator ruling 2026-11-05 08:35Z. This operator is not the exact derivative of the forward's limiter.");
        }
        let (mut d0, mut dn): (Option<std::sync::Arc<Bytes>>, Option<Bytes>) = (None, None);
        if !dead_ranges().is_empty() { receipt["dot_dead_ranges"] = json!({"file":std::env::var("AD_DOT_DEAD").ok(),"allocations":dead_ranges().len(),"bytes":dead_ranges().values().flatten().map(|r| r.1 - r.0).sum::<usize>(),
            "rule":"zero y weight at the window end on work memory the next step overwrites before reading. NOT proved by this run: the proof is a separate run named by AD_DOT_DEAD_PROOF (NaN in these bytes in the forward and/or before the reverse walk-back, AD_DOT_DEAD_POISON=1|rev|rev-all, every boundary equal and gradient words equal to a control leg); f52-run checks that run's own files and reads valid false without it","needs":"--stable-dir 1 and --direction field",
            "proof_run":std::env::var("AD_DOT_DEAD_PROOF").ok(),"poison":std::env::var("AD_DOT_DEAD_POISON").ok(),
            "ranges":dead_ranges().iter().flat_map(|(id, rs)| rs.iter().map(move |&(lo, hi)| json!({"allocation":id,"offset":lo,"bytes":hi - lo}))).collect::<Vec<_>>()}); }
        receipt["stable_directions"] = json!(if w.stable_dir { "on: translated address words, relocation slots, address-bearing tables and declared residue ranges of each boundary get zero weight, so x and y are the same in every process" } else { "off" });
        receipt["direction"] = json!(if w.vectors.is_some() { "explicit: package-bound ownership-aligned x/y vectors; unlisted words zero" } else if opt(&a, "direction") == Some("fieldxy") { "fieldxy: x = u * rms_f(boundary 0), y = u / rms_f(window end), rms_f the RMS of allocation f's valid words (finite, at most 2e5, above 1e-30) at that boundary; invalid words 0 on both sides. The operator is unchanged; the same x and y seed <y, M x>, the reverse and <x, M^T y>." } else if opt(&a, "direction") == Some("field") { "field: x = u * |primal at boundary 0| (relative); y = u / rms_f, rms_f the RMS of allocation f's valid words (finite, at most 2e5, above 1e-30) at the window end; invalid words 0. The same y seeds <y, M x> and the reverse." } else if rel_dir { "relative: x = u * |primal at boundary 0|, y = u * |primal at the window end| (finite, at most 2e5; larger words are stale scratch or address words and get 0)" } else { "plain uniform [-1,1) on every shadow word" });
        macro_rules! xs { () => { if rel_dir { Some((&bounds[0], &d0.as_ref().unwrap()[..])) } else { None } } }
        macro_rules! ys { () => { if rel_dir { Some((&bounds[bounds.len() - 1], dn.as_deref().unwrap())) } else { None } } }
        let plan_live_bound: usize = {
            let names: Vec<&String> = if plan.physics.is_some() { plan.blocks.keys().collect() }
                                      else { plan.cadence.iter().take(steps).flatten().collect::<std::collections::BTreeSet<_>>().into_iter().collect() };
            plan.allocations.iter().map(|x| x.bytes).sum::<usize>() + names.iter().filter_map(|n| plan.blocks.get(*n)).flatten()
                .map(|o| if let Operation::Allocate { bytes, .. } = o { *bytes } else { 0 }).sum::<usize>()
        };
        // hunt grad-only: whether the tape was prepared for a reverse (size_tape, ahead_maybe, st_plan) inside the seed
        // loop. --no-tangent 1 breaks and --tangent-only 1 continues before that point; the objective sweep then prepares it.
        let mut rev_prepared = false;
        // keep-forward: this request's start fingerprint (None: the word is off or this is not the line it serves),
        // whether the kept forward is this start's, and the seconds of the forward step loop that made what is kept
        let (mut kf_key, mut kf_hit, mut kf_forward_s): (Option<String>, bool, f64) = (None, false, 0.0);
        // species planes (AD_DOT_PLANES): both ends' planes must lie inside a shadow, or the run stops here, before the forward
        if let Some(ps) = plane_spec()? {
            if !rel_dir { return Err("AD_DOT_PLANES needs a weighted direction (--direction field, fieldxy or relative)".into()); }
            for (side, (id, off)) in [("x (boundary 0)", ps.x), ("y (window end)", ps.y)] {
                let sh = w.shadows.get(&id).ok_or(format!("AD_DOT_PLANES: the {side} allocation {id} has no shadow"))?;
                if off + ps.n * ps.pb > sh.bytes { return Err(format!("AD_DOT_PLANES: the {side} planes [{off}, {}) are outside allocation {id} ({} bytes)", off + ps.n * ps.pb, sh.bytes)); }
            }
            receipt["planes"] = ps.receipt();
        }
        for (k, &seed) in seeds.iter().enumerate() {
            let t0 = Instant::now();
            if k > 0 { let d = w.load(&bounds[0])?; w.restore(&mut ex, &mut frame, &bounds[0], &d)?; }
            else {
                // resident (request 2 on): the first request's boundary 0, kept, restored as a second seed restores
                // boundary 0; then each allocation planted in it or in this request gets the plan's current payload (this
                // request's planted words, or the recording's): the start ex.prepare uploads for a fresh program of this request.
                if let (Some(z), Some(kb)) = (rz.as_ref(), b0_keep.as_ref()) {
                    w.held_prev = None; w.cs_dev_at = None;
                    let d = w.load(kb)?; w.restore(&mut ex, &mut frame, kb, &d)?; drop(d);
                    let info = resident_replant(&mut ex, &plan, &base, &z.replant_ids())?;
                    log(json!({"event":"window_resident_start","request":z.name,"index":z.index,"replant":info}));
                    receipt["resident"]["replant"] = info;
                    mark("resident_start");
                }
                bounds.push(w.store(&mut ex, &frame, &bdir, 0)?);
                // species planes: the x-side planes' allocation must be live at boundary 0, or its planes would get no weight unseen
                if let Some(ps) = plane_spec()? { if !bounds[0].index.iter().any(|e| e.0 == ps.x.0) {
                    return Err(format!("AD_DOT_PLANES: allocation {} is not live at boundary 0", ps.x.0)); } }
                // resident (first request): boundary 0 as ex.prepare left it, before any move of the start, kept for
                // the later requests in its own folder (the store's RAM tier when the budget has room).
                if let (Some(z), true) = (rz.as_mut(), b0_keep.is_none()) {
                    let kd = z.dir.join("keep"); fs::create_dir_all(&kd).map_err(|e| e.to_string())?;
                    b0_keep = Some(w.store(&mut ex, &frame, &kd, 0)?); z.kept(); mark("resident_kept");
                }
                if let Some((h, name, t)) = &x0 {
                    { let b = w.load(&bounds[0])?; refuse_index_words("--x0-dir", opt(&a, "x0-dir").unwrap_or(""), &bounds[0], &b, t)?; }
                    w.perturb_terms(&ex, *h, t)?;
                    if !x0dense.is_empty() {
                        { let b = w.load(&bounds[0])?; check_dense("--x0-dir", &bounds[0], &b, &x0dense)?; }
                        dev::sync(&*cuda)?;
                        for (al, off, bytes) in &x0dense {
                            let a0 = ex.allocations.get(al).ok_or_else(|| format!("--x0-dir dense allocation {al} not live at boundary 0"))?;
                            let mut buf = dev::download(&*cuda, a0.address + *off as u64, bytes.len())?;
                            for (c, d) in buf.chunks_exact_mut(4).zip(bytes.chunks_exact(4)) {
                                let pv = f32::from_le_bytes((&*c).try_into().unwrap()) as f64; let dv = f32::from_le_bytes(d.try_into().unwrap()) as f64;
                                c.copy_from_slice(&((pv + *h * dv) as f32).to_le_bytes());
                            }
                            dev::upload(&*cuda, a0.address + *off as u64, &buf)?;
                        }
                        dev::sync(&*cuda)?;
                        log(json!({"event":"window_x0_dense","h":h,"blocks":x0dense.len(),"words":x0dense.iter().map(|x| x.2.len() / 4).sum::<usize>()}));
                    }
                    let old = bounds.pop().unwrap();
                    w.drop_boundary(old)?;
                    bounds.push(w.store(&mut ex, &frame, &bdir, 0)?);
                    receipt["x0"] = json!({"file":opt(&a, "x0-dir"),"name":name,"h":h,"terms":t.len()});
                    if let Some(p) = &x0_planted { receipt["x0_planted"] = p.clone(); }
                    log(json!({"event":"window_x0_moved","h":h,"terms":t.len()}));
                }
                let b0 = std::sync::Arc::new(w.load(&bounds[0])?); mark("store_load_b0");
                if w.delta_store { w.ckpt_base = Some((bounds[0].index.clone(), b0.clone())); }
                if let Some((_, t)) = &tdir { refuse_index_words("--taylor-dir", opt(&a, "taylor-dir").unwrap_or(""), &bounds[0], &b0, t)?; }
                if !tdense.is_empty() { check_dense("--taylor-dir", &bounds[0], &b0, &tdense)?; }
                for (pth, _, t) in &objs {
                    // An objective weighs the WINDOW-END state, which may live in an allocation the window itself
                    // allocates (pk120e4: rho_theta at boundary 1 is allocation 429, absent at boundary 0), so
                    // the boundary-0 integer-word check covers the terms whose allocation exists there; the rest
                    // must still be a plan allocation of this window (stale files for another recording refuse).
                    let live0: std::collections::HashSet<u64> = bounds[0].index.iter().map(|x| x.0).collect();
                    let (at0, later): (Vec<_>, Vec<_>) = t.iter().copied().partition(|x| live0.contains(&x.0));
                    if let Some(x) = later.iter().find(|x| !w.base_el.contains_key(&x.0)) {
                        return Err(format!("--objective {pth}: allocation {} is not an allocation of this plan", x.0));
                    }
                    refuse_index_words("--objective", pth, &bounds[0], &b0, &at0)?;
                }
                d0 = Some(b0);
                if keep_forward_word() != 0 && rz.is_some() && no_tangent && !fd_only && descent.is_some() && seeds.len() == 1 && w.shold.is_none() && !w.cs_on && !w.delta_store
                   && !w.fuse && regime.is_empty() && dump_steps.is_empty() && dump_at.is_empty() && !w.step_norms && !dead_poison_mode().0 && param_tl.is_none() {
                    let t_k = Instant::now();
                    let ops: Vec<(String, usize, String)> = rz.as_ref().map(|z| z.cur_ops()).unwrap_or_default().into_iter().map(|(blk, op)| {
                        let sha = match plan.blocks.get(&blk).and_then(|o| o.get(op)) { Some(Operation::Upload { data, .. }) => crate::blob_sha(data).to_owned(), _ => "not an upload".to_owned() };
                        (blk, op, sha) }).collect();
                    let fresh: Option<std::collections::HashSet<u64>> = b0_keep.as_ref().map(|b| b.index.iter().map(|x| x.0).collect());
                    match start_digest(&bounds[0].index, fresh.as_ref(), &d0.as_ref().unwrap()[..], &ops, steps) {
                        Ok(key) => {
                            kf_hit = kf.as_ref().is_some_and(|x| x.key == key);
                            log(json!({"event":"window_keep_forward","at":"start","start_sha256":key,"kept_start_sha256":kf.as_ref().map(|x| x.key.clone()),"kept_from":kf.as_ref().map(|x| x.from.clone()),
                                       "same_start":kf_hit,"swapped_uploads":ops.len(),"allocations_now":bounds[0].index.len(),"allocations_of_a_fresh_start":fresh.as_ref().map(|f| f.len()),"digest_s":t_k.elapsed().as_secs_f64()}));
                            kf_key = Some(key);
                        },
                        Err(e) => log(json!({"event":"window_keep_forward","at":"no_key","why":e})),
                    }
                    mark("keep_forward_key");
                }
                // another start (or the word off for this request): what was kept goes before this forward stores its own
                if !kf_hit { if let Some(old) = kf.take() {
                    log(json!({"event":"window_keep_forward","at":"dropped","kept_from":old.from,"bytes":old.bytes,"why":"this request's start is another start: its forward runs"}));
                    drop(old); w.kf_saved.clear(); w.kf_keep.clear();
                } }
            }
            if tangent_dir && k == 0 {
                w.zero(&*cuda)?;
                for &(al, wd, v) in &tdir.as_ref().unwrap().1 {
                    let el = *w.base_el.get(&al).ok_or_else(|| format!("taylor-dir allocation {al} has no shadow"))?;
                    dev::upload(&*cuda, w.sh(el + wd as i64), &v.to_le_bytes())?;
                }
                for (al, off, bytes) in &tdense {
                    let el = *w.base_el.get(al).ok_or_else(|| format!("taylor-dir dense allocation {al} has no shadow"))?;
                    dev::upload(&*cuda, w.sh(el) + *off as u64, bytes)?;
                }
                if !tdense.is_empty() { log(json!({"event":"window_tangent_dense_seed","blocks":tdense.len(),"words":tdense.iter().map(|x| x.2.len() / 4).sum::<usize>()})); }
                dev::sync(&*cuda)?;
            } else if let (Some(c), 0) = (&param_tl, k) { w.zero(&*cuda)?; w.par_set(&*cuda, None, Some(c))?; }
            else if !(kf_hit && keep_forward_word() == 2) { w.fill(&*cuda, sx(seed), xs!())?; }
            if let Some(p) = crate::peer::get() { p.begin_tl(); }
            if k==0 {if let Some(p)=opt(&a,"export-vectors") {
                receipt["vector_exports"]["x"]=w.export_vectors(&*cuda,&bounds[0],Path::new(p),"x")?;
            }}
            w.trace_last = f64::MAX;
            w.counting = k == 0; mark("fill_x");
            w.entry_ms = 0.0;
            // terrain-w-hash: this pass's boundary digests run beside the next step (joined at the next boundary and
            // when the pass ends, below); every comparison and refusal is the same.
            ex.digest_defer = true;
            if k == 0 && descent.is_none() && !w.delta_store && w.bs_ram_budget > w.bs_ram_now { w.pin_prefill(&ex, steps, plan_live_bound)?; } mark("pin_prefill");
            // speed storm3: a checkpointed window with its checkpoints in host memory reuses their buffers (BS_POOL)
            if k == 0 && descent.is_some() && !w.delta_store && w.bs_ram_budget > 0 && std::env::var("AD_BSTORE_POOL").as_deref() != Ok("0") {
                // kfstore: the forward checkpoints keep-forward will keep are outside the store's budget: room for them in the pool too
                let kf_kept = if kf_key.is_some() && !kf_hit && std::env::var("AD_KF_PREFILL").as_deref() == Ok("1") { ckpts.unwrap_or(0) } else { 0 };
                BS_POOL.store(capi::pin_pool_for_checkpoints(ex.context as usize, w.bs_ram_budget + kf_kept * capi::pin_cap(plan_live_bound)), std::sync::atomic::Ordering::Relaxed);
                bs_prefill(&ex, plan_live_bound, w.bs_ram_budget, kf_kept);
            }
            w.gather_pass_join = true;
            if !kf_hit { w.prepare_pass_start(&mut ex, &plan, &base, &frame, steps)?; } mark("pass_prepare");
            ex.finish_staging_ahead(); mark("staging_ahead_done");
            if dev::sites::on() { log(json!({"event":"window_pass_setup_sync_sites","seed":seed,"sync_sites":dev::sites::take()})); }
            let pass_setup_s = t0.elapsed().as_secs_f64();
            let t_step_loop = Instant::now();
            // keep-forward: no step runs when the kept forward is this start's (the range is empty)
            for s in (if kf_hit { steps + 1 } else { 1 })..=steps {
                let ts = Instant::now();
                // Match window_tangent_step.wall_s, including boundary storage,
                // gather preparation and the host work following the inner step.
                let step_profile = crate::step_profile::StepRange::begin(
                    if !fd_only && !no_tangent { "tl" } else { "forward" }, s)?;
                let ms0 = w.entry_ms;
                let sa0 = crate::staging_allocs();
                // integ8: capture the whole state only at a step this pass stores (with S02 --checkpoints most steps are not
                // stored); a capture never outlives its step (cleared below), so no later step can read it.
                w.fuse_rec = w.fuse && k == 0 && descent.is_none(); w.fz_steps = steps;
                crate::spill::phase(&format!("tl{s}"));
                w.capture = !w.cs_on && k == 0 && descent.as_ref().map_or(true, |p| s == steps || (p.contains(&s) && w.shold.is_none()));
                // host-leftovers-2: a stored boundary goes straight into page-locked RAM while the budget has room
                // (--bstore-ram-gib, default 40 % of MemAvailable at start; 0 = files as before; never with changed-chunk files).
                w.capture_host = if w.capture && !w.delta_store && w.bs_ram_budget > w.bs_ram_now { Some((bs_pool(), w.bs_ram_budget - w.bs_ram_now)) } else { None };
                w.pin_ahead = k == 0 && s < steps && descent.is_none();
                // sixty (AD_DOT_DEAD_POISON=1 with AD_DOT_DEAD): the proof that the listed ranges are dead. Before the last
                // step (a steady step, s >= 3) every range is filled with NaN in the primal; the step boundary check below
                // then shows whether anything read them.
                WIN_STEPS.store(steps, std::sync::atomic::Ordering::Relaxed);
                if k == 0 && s == steps && s >= 3 && dead_poison_mode().0 && !dead_ranges().is_empty() {
                    let (mut nb, mut nr) = (0usize, 0usize);
                    for (id, rs) in dead_ranges() {
                        let a = *ex.allocations.get(id).ok_or_else(|| format!("AD_DOT_DEAD: allocation {id} is not live before step {s}"))?;
                        for &(lo, hi) in rs {
                            if hi > a.bytes { return Err(format!("AD_DOT_DEAD: range [{lo}, {hi}) is outside allocation {id} ({} bytes)", a.bytes)); }
                            let nan: Vec<u8> = std::iter::repeat([0x00u8, 0x00, 0xc0, 0x7f]).take((hi - lo) / 4).flatten().collect();
                            dev::upload(&*cuda, a.address + lo as u64, &nan)?; nb += hi - lo; nr += 1;
                        }
                    }
                    dev::sync(&*cuda)?;
                    log(json!({"event":"window_dead_poison","before_step":s,"ranges":nr,"bytes":nb,"word":"NaN 0x7fc00000"}));
                    receipt["dead_poison"] = json!({"before_step":s,"ranges":nr,"bytes":nb});
                }
                let flat = if unchecked { w.forward_step_c(&mut ex, &plan, &base, &mut frame, None, s, !fd_only && !no_tangent) }
                           else { w.forward_step(&mut ex, &plan, &base, &mut frame, &refs, s, !fd_only && !no_tangent) };
                let flat = match flat {
                    // dist47: a kept-loaded program whose request starts where the forecast itself stops (its fatal-condition
                    // words read nonzero in the base forward) refuses that request and stays loaded. THE BREAKAGE THIS FIXES:
                    // the program ended on the refusal, so a minimisation lost its loaded program at the first trial start
                    // the model could not run (the 9th gradient call of a run on the 12 km storm cut, 2026-11-14) and every call after it
                    // paid a program start. What the request left (its boundaries, the kept forward's bookkeeping, the
                    // pass's pending work) goes here as a finished request's goes; the next request restores the kept start.
                    Err(e) if k == 0 && rz.is_some() && b0_keep.is_some() && crate::refusal_leaves_program_loaded(&e) => {
                        w.resident_refused(&mut ex, &mut bounds, &mut ck)?;
                        log(json!({"event":"window_resident_refused","step":s,"error":e}));
                        break 'request Some(e);
                    },
                    r => r?,
                };
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
                    // longwin: --dump-steps: the same fields as --dump-state at this boundary (tangent and primal)
                    let mut dump_now: Vec<(u64, usize, Option<usize>)> = if s < steps && dump_steps.contains(&s) { dump_state.clone() } else { Vec::new() };
                    dump_now.extend(dump_at.iter().filter(|x| x.0 == s).map(|x| (x.1, x.2, Some(x.3))));
                    if !dump_now.is_empty() {
                        dev::sync(&*cuda)?;
                        let mut rows = Vec::new();
                        for &(id, off, n) in &dump_now {
                            let a0 = ex.allocations.get(&id).ok_or_else(|| format!("--dump-steps: allocation {id} is not live at boundary {s}"))?;
                            let len = n.unwrap_or(a0.bytes.saturating_sub(off));
                            if off + len > a0.bytes || len == 0 { return Err(format!("--dump-steps {id}:{off}: outside allocation {id} ({} bytes)", a0.bytes)); }
                            let b = dev::download(&*cuda, a0.address + off as u64, len)?;
                            let mut r = write_state(&out, dump_name(&format!("state-s{s}"), (id, off, n)), &b)?;
                            r["allocation"] = json!(id); r["byte_offset"] = json!(off); r["step"] = json!(s); r["kind"] = json!("base");
                            rows.push(r);
                            if !fd_only && !no_tangent {
                                let (Some(&el), Some(_)) = (w.base_el.get(&id), w.shadows.get(&id)) else { return Err(format!("--dump-steps allocation {id} has no shadow")) };
                                let t = dev::download(&*cuda, w.sh(el) + off as u64, len)?;
                                let mut r = write_state(&out, dump_name(&format!("tl-s{s}"), (id, off, n)), &t)?;
                                r["allocation"] = json!(id); r["byte_offset"] = json!(off); r["step"] = json!(s); r["kind"] = json!("tangent");
                                rows.push(r);
                            }
                        }
                        if state_dumps.get("steps").is_none() { state_dumps["steps"] = json!([]); }
                        state_dumps["steps"].as_array_mut().unwrap().extend(rows);
                        log(json!({"event":"window_state_dump","where":"step","step":s,"fields":dump_now.len()}));
                    }
                    if s == 1 && steps >= 2 && !w.fuse && !(w.st_on && w.pack) { w.ahead_maybe(&ex, &flats, anchors, opt(&a, "tape-reserve-gib"), false, s, plan_live_bound)?; }
                    if s == 1 && !w.fuse { w.st_plan(&ex, &flats, "after tangent step 1")?; }
                    if w.step_norms { log(w.step_norm_line(&*cuda, seed, s)?); }
                }
                let host_prof: Vec<Value> = dev::prof::take().into_iter().filter(|x| x.2 > 0).map(|(k, s, c)| json!([k, s, c])).collect();
                let rows = w.take_rows(0);
                let step_wall_s = ts.elapsed().as_secs_f64();
                drop(step_profile); mark(&format!("tl_step_{s}"));
                w.word_probe(&*cuda, "tangent", "step", s, None, &json!({"seed":seed}))?;
                log(json!({"event":"window_tangent_step","seed":seed,"step":s,"tape_rows":rows,"wall_s":step_wall_s,"staging_allocs":crate::staging_allocs() - sa0,"boundary_hashes":if params_unchecked || opt(&a, "x0-dir").is_some() { "not compared" } else { "equal" },"boundary_check":if ex.digest_defer { "deferred: compared while the next step runs, and before this pass ends" } else { "at this boundary" },"host_prof":host_prof,"sync_sites":if dev::sites::on() { json!(dev::sites::take()) } else { Value::Null },"driver_waits_cum":if dev::sites::on() { let n = &capi::xfer::N; json!({"ctx":capi::xfer::get(&n.ctx),"stream":capi::xfer::get(&n.stream),"event":capi::xfer::get(&n.event),"d2h_calls":capi::xfer::get(&n.d2h_n),"h2d_calls":capi::xfer::get(&n.h2d_n),"h2d_bytes":capi::xfer::get(&n.h2d),"d2h_bytes":capi::xfer::get(&n.d2h)}) } else { Value::Null },"reference_check":!params_unchecked}));
            }
            if kf_hit {
                // keep-forward: the kept forward's checkpoints, window end and step lists become this request's, their
                // bytes counted in the store's RAM again as when the forward stored them
                let t_k = Instant::now();
                let KeptForward { key: _, from, end, ck: kck, flats: kfl, per_step: kps, bytes, forward_s } = kf.take().ok_or("keep-forward: nothing kept")?;
                for b in kck.values().chain(std::iter::once(&end)) { if let Some(r) = &b.ram { w.bs_ram_now += r.as_slice().len(); } }
                let n_ck = kck.len();
                w.kf_saved.clear(); w.kf_keep = kck.keys().copied().collect();
                ck = kck; bounds.push(end); flats = kfl; per_step = kps; kf_forward_s = forward_s;
                if keep_forward_word() == 1 {
                    let last = bounds.len() - 1;
                    let d = w.load(&bounds[last])?; w.restore(&mut ex, &mut frame, &bounds[last], &d)?; drop(d);
                }
                let info = json!({"used":true,"mode":keep_forward_word(),"forward_from_request":from,"start_sha256":kf_key,"checkpoints":n_ck,"kept_bytes":bytes,
                                  "forward_step_loop_s_not_run":forward_s,"window_end_restored":keep_forward_word() == 1,"take_s":t_k.elapsed().as_secs_f64()});
                log(json!({"event":"window_keep_forward","at":"reused","info":info}));
                receipt["keep_forward"] = info;
                mark("keep_forward_reused");
            } else if kf_key.is_some() && k == 0 {
                // keep-forward: this forward's checkpoints are kept by the sweep (all of them in host memory, whole)
                let whole = bounds.len() == 2 && ck.values().chain(bounds.last()).all(|b| b.ram.is_some() && b.sub.is_none() && b.cs.is_none() && b.dev == 0);
                w.kf_saved.clear();
                w.kf_keep = if whole { ck.keys().copied().collect() } else { std::collections::BTreeSet::new() };
                if !whole { kf_key = None; }
                kf_forward_s = t_step_loop.elapsed().as_secs_f64();
                let info = json!({"used":false,"mode":keep_forward_word(),"start_sha256":kf_key,"keeping":whole,"checkpoints":ck.len(),
                                  "why_not":if whole { Value::Null } else { json!("a checkpoint or the window end is not whole in host memory (files, changed-chunk, content store or held states)") }});
                log(json!({"event":"window_keep_forward","at":"forward_ran","info":info}));
                receipt["keep_forward"] = info;
            }
            let step_loop_s = t_step_loop.elapsed().as_secs_f64();
            let t_completion = Instant::now();
            // machine: the first one-pass reverse step page-locks a whole-state buffer for its redo compare
            // (redo_download; the anchors-ahead buffers it used to borrow are no longer made when the reverse is one
            // pass). Measured at 120 km: 1.1-1.2 s inside reverse step 3 with the card idle (a test machine pair1-b / pair2-a,
            // redo_download). It is made here instead, on a helper, while this thread waits for the pass's checks
            // (gathers, boundary digests), and handed to the pool redo_download and snap_ram take from. The same
            // buffer, the same bytes. AD_CMP_PIN_AHEAD=0 makes it in the reverse step as before.
            let cmp_ahead = if k == 0 && !fd_only && !no_tangent && opt(&a, "tangent-only") != Some("1")
                               && w.anchor_pool.is_empty() && !w.cs_on && std::env::var("AD_CMP_PIN_AHEAD").as_deref() != Ok("0") {
                let total = bounds.last().map(|b| b.index.iter().map(|(_, off, n)| off + n).max().unwrap_or(0)).unwrap_or(0);
                if total == 0 { None } else {
                    let (fns, set, c) = (crate::pinned::PinFns::of(&ex.cuda), ex.cuda.ctx_set_current, ex.context as usize);
                    Some((total, std::thread::Builder::new().name("f52-cmp-pin".into()).spawn(move || {
                        let r = set(c as *mut std::ffi::c_void); if r != 0 { return Err(format!("cuCtxSetCurrent compare buffer: {r}")); }
                        crate::pinned::PinMem::new(fns, total)
                    }).map_err(|e| e.to_string())?))
                }
            } else { None };
            w.gather_pass_join = false;
            // the gathers begun in the last step stay pending (installed on demand in the reverse, after the tape is sized), as
            // the per-step join left them: x20-E-gj joined them all here and the tape shrank 10.3 -> 7.1 GiB (segments 5/8/9 -> 8/13/14).
            { let t = dev::prof::start(); w.join_pending_if(&*cuda, steps, |p| p.3 != steps)?; dev::prof::add(13, t); } mark("gathers_joined");
            ex.digest_defer = false;
            ex.digest_flush()?; mark("digests_joined");
            if let Some((total, h)) = cmp_ahead {
                if let Ok(Ok(m)) = h.join() { w.anchor_pool.push(crate::pinned::PinnedBuf::from_mem(m, total)); }
                mark("cmp_pin_made");
            }
            let completion = json!({"event":"window_pass_completion","seed":seed,"tangent":!fd_only && !no_tangent,
                "setup_s":pass_setup_s,"step_loop_s":step_loop_s,"completion_s":t_completion.elapsed().as_secs_f64(),
                "pass_outer_s":t0.elapsed().as_secs_f64(),"window_elapsed_s":t_start.elapsed().as_secs_f64(),
                "required_checks_complete":true});
            log(completion.clone());
            receipt["pass_completions"].as_array_mut().map(|v| v.push(completion.clone())).unwrap_or_else(|| {
                receipt["pass_completions"] = json!([completion]);
            });
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
            if dn.is_none() { dn = Some(w.load(&bounds[bounds.len() - 1])?); w.y_path = Some(bounds[bounds.len() - 1].path.clone()); } mark("load_end_boundary");
            // speed ckpt60: the window end is held as whole bytes by the dots (dn) from here on; its state in the content
            // store is read once more for the truth the last step's redo is compared with, written to the boundary's file
            // behind (the writer thread, from the pieces) and let go. Whoever asks for the window end later reads the file.
            if w.cs_on { let last = bounds.len() - 1; w.cs_end(&ex, &mut bounds[last], steps)?; }
            if k==0 {if let Some(p)=opt(&a,"export-vectors") {
                receipt["vector_exports"]["tl"]=w.export_vectors(&*cuda,&bounds[bounds.len()-1],Path::new(p),"tl")?;
            }}
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
                { receipt["out_of_core"] = crate::spill::receipt(); fs::write(out.join("window.json"), serde_json::to_vec_pretty(&receipt).unwrap()) }.map_err(|e| e.to_string())?;
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
                    // ranks (R2): the objective weighs owned words only, so <w, M x> over the whole mesh is the rank-ordered
                    // sum of the local parts (every rank must dump state in this mode: the reduce waits for all ranks).
                    let ymx = match crate::peer::get() {
                        Some(p) => { let g = p.reduce_sum("wmx", &[ymx])?; log(json!({"event":"peer_w_Mx","rank":p.rank,"local":ymx,"global":g[0]})); g[0] },
                        None => ymx };
                    receipt["tangent_dir"] = json!({"taylor_dir":opt(&a, "taylor-dir"),"name":tdir.as_ref().unwrap().0,"objective":objs[0].1,"w_Mx":ymx});
                }
                receipt["state_dumps"] = state_dumps.clone();
                log(json!({"event":"window_state_dump","where":"base","fields":dump_state.len()}));
            }
            if fd_only { log(json!({"event":"window_fd_base_done","wall_s":t0.elapsed().as_secs_f64()})); break; }
            if no_tangent { log(json!({"event":"window_base_done_no_tangent","wall_s":t0.elapsed().as_secs_f64()})); break; }
            let (pt, qt, bad_t) = w.dots(&*cuda, sy(seed), ys!(), sx(seed), xs!())?; mark("dots_tl");
            let pt64_std = f64::from_bits(LAST_F64.load(std::sync::atomic::Ordering::Relaxed));
            // multidevice-runner: <y, Mx> over all ranks (rank-ordered sum, the same in every rank); the local part is logged.
            let (pt, qt, bad_t) = match crate::peer::get() {
                Some(p) => { let g = p.reduce_sum(&format!("tl{seed}"), &[pt, qt, bad_t as f64])?;
                             log(json!({"event":"peer_tangent_dot","seed":seed,"rank":p.rank,"local":[pt, qt, bad_t],"global":g}));
                             (g[0], g[1], g[2] as u64) },
                None => (pt, qt, bad_t) };
            let abs_terms_tl = f64::from_bits(LAST_ABS_TERMS.load(std::sync::atomic::Ordering::Relaxed));
            // F14: on a rank plan the terms are summed over the ranks like the dot (every rank asks, so the reduce matches).
            let abs_terms_tl = match crate::peer::get() { Some(p) if abs_terms_wanted() => p.reduce_sum(&format!("tlabs{seed}"), &[abs_terms_tl])?[0], _ => abs_terms_tl };
            let t_tl = t0.elapsed().as_secs_f64();
            if k == 0 { receipt["tangent_norm_top_allocations"] = json!(w.norms(&*cuda, 12)?); } mark("norms");
            log(json!({"event":"window_tangent_done","seed":seed,"y_Mx":pt,"y_Mx_minus_x":qt,"nonfinite":bad_t,"wall_s":t_tl}));
            if opt(&a, "tangent-only") == Some("1") { continue; }
            w.fuse_rec = false;
            let all_fused = w.fuse && k == 0 && (1..=steps).all(|s| w.fz_tapes.contains_key(&s));
            if k == 0 && w.fuse { log(json!({"event":"window_fused_tape","steps_fused":w.fz_tapes.len(),"steps":steps,"chunks":w.fz_chunks.len(),
                "d2h_s":w.fz_d2h_s,"d2h_bytes":w.fz_d2h_bytes,"fence_waits":w.rtape.as_ref().and_then(|t| t.fence.as_ref()).map(|f| f.waited),"offsets_kept":w.fz_offs.len(),"early_checks":w.fz_early,"copy_out_async":!w.fz_copy.is_null(),"arena_bytes":w.rtape.as_ref().map(|t| t.bytes),"per_step":w.fz_info})); }
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
            crate::spill::phase("before-reverse");
            let streamed = w.st_on && w.pack && !all_fused;
            // memory (streamed tape): the arena keeps its size (size_tape still runs for its receipt and the window-cache
            // rule, capped at the arena's own bytes); a streamed reverse makes no anchors.
            let cap_now = format!("{}", w.tape.bytes as f64 / (1u64 << 30) as f64);
            if k == 0 && w.one_pass && !all_fused { w.size_tape(&ex, &flats, if streamed { Some(cap_now.as_str()) } else { opt(&a, "tape-max-gib") }, opt(&a, "tape-reserve-gib"))?; } mark("tape_sized");
            if k == 0 && !all_fused && !streamed { w.ahead_maybe(&ex, &flats, anchors, opt(&a, "tape-reserve-gib"), true, steps, plan_live_bound)?; } mark("anchors_decided");
            if k == 0 && streamed { w.st_plan(&ex, &flats, "before the reverse")?; }
            if k == 0 { rev_prepared = true; }
            if crate::spill::on() { log(json!({"event":"window_alloc_census","at":"before the reverse","census":crate::spill::census(40)})); }
            // any-mesh (AD_DOT_SELF=1): the cancellation-free dot. y := W M x (see self_y); its <y, M x> replaces the random y's.
            // Its units default to the model's prognostic fields at the window end (the package's field map, boundary
            // steps + 1); a rank plan has no whole-mesh map to read, so its ranks keep pool units (see self_y).
            let selfdot = if std::env::var("AD_DOT_SELF").ok().as_deref() == Some("1") {
                let fields_default: Result<String, String> = if crate::peer::get().is_some() { Err("a rank plan: the package's field map is whole-mesh".into()) }
                    else { package_field_map(&config).and_then(|(m, v)| map_window_end(&m, &v, steps).map(|b| format!("{}:{b}", m.display()))) };
                Some(w.self_y(&*cuda, sy(seed), ys!(), fields_default)?) } else { None };
            // central hunt-3km-slot: a floor asked for on a run without the self dot is refused, not dropped without a word.
            if selfdot.is_none() && std::env::var("AD_DOT_SELF_FLOOR").is_ok_and(|v| !v.trim().is_empty()) { return Err("AD_DOT_SELF_FLOOR needs AD_DOT_SELF=1".into()); }
            // ranks: on a rank plan each rank builds y from its own tangent words (halo copies included: the dot runs
            // over the product space, every local word once) and <y, M x> is the rank-ordered sum, the same in every rank.
            let mut selfdot = match (selfdot, crate::peer::get()) {
                (Some(mut sd), Some(p)) => {
                    let g = p.reduce_sum(&format!("self{seed}"), &[sd.1])?;
                    sd.2["local_y_Mx"] = json!(sd.1); sd.2["y_Mx"] = json!(g[0]); sd.2["rank"] = json!(p.rank);
                    log(json!({"event":"peer_self_dot","seed":seed,"rank":p.rank,"local":sd.1,"global":g[0]}));
                    sd.1 = g[0]; Some(sd) },
                (s, _) => s };
            let (pt, abs_terms_tl) = match &selfdot { Some(sd) => (sd.1, sd.1), None => (pt, abs_terms_tl) };
            if let Some(sd) = &selfdot { receipt["self_dot"] = sd.2.clone(); }
            // sixty (AD_DOT_SELF_SPLIT=G, diagnostic, with AD_DOT_SELF=1): after the dot's own reverse, G more reverses on the
            // same tangent, each seeded with the self y of ONE group of allocations (contiguous in id order). <y_g, M x> is the
            // sum of the group's own terms, so x_MTy_g - y_Mx_g names the group whose transpose is inexact, and the groups'
            // x_MTy must sum to the whole dot's (the check that a repeated reverse is clean). Rows: receipt self_split.
            // AD_DOT_SELF_SPLIT_PART=i/n keeps the groups whose number is i modulo n (several cards share one split; then the
            // groups of all parts together sum to the whole dot). Each kept group carries its number.
            // species planes (AD_DOT_SELF_SPLIT=planes, with AD_DOT_PLANES): group p is plane p's unit alone, group N every other
            // unit; a plane that carried no tangent at the window end has no group (its row in self_dot.planes says so).
            let split_part: Option<(usize, usize)> = std::env::var("AD_DOT_SELF_SPLIT_PART").ok().and_then(|v| { let mut it = v.split('/');
                Some((it.next()?.trim().parse().ok()?, it.next()?.trim().parse().ok()?)) }).filter(|x: &(usize, usize)| x.1 >= 1);
            let split_env = std::env::var("AD_DOT_SELF_SPLIT").ok();
            // AD_DOT_SELF_SPLIT=planes:qr,qs,7 keeps only the planes named (by name or number) and no rest group: a long
            // window pays one reverse a group, so a leg can read the planes it is about.
            let split: Vec<(usize, Vec<(u64, usize, usize, f64)>)> = match (&selfdot, split_env.as_deref().map(str::trim)) {
                (Some(sd), Some(v)) if (v == "planes" || v.starts_with("planes:")) && k == 0 && !sd.3.is_empty() => {
                    let ps = plane_spec()?.ok_or("AD_DOT_SELF_SPLIT=planes needs AD_DOT_PLANES")?;
                    let keep: Option<Vec<usize>> = match v.strip_prefix("planes:") {
                        None => None,
                        Some(l) => Some(l.split(',').map(|t| { let t = t.trim();
                            t.parse::<usize>().ok().filter(|&p| p < ps.n).or_else(|| ps.names.iter().position(|n| n == t))
                                .ok_or(format!("AD_DOT_SELF_SPLIT={v}: no plane {t:?} (names {:?})", ps.names)) }).collect::<Result<Vec<_>, String>>()?) };
                    let ids = PLANE_UNIT0..PLANE_UNIT0 + ps.n as u64;
                    let mut groups: Vec<Vec<(u64, usize, usize, f64)>> = (0..ps.n).map(|p| if keep.as_ref().is_none_or(|kp| kp.contains(&p)) {
                        sd.3.iter().filter(|x| x.0 == PLANE_UNIT0 + p as u64).cloned().collect() } else { Vec::new() }).collect();
                    if keep.is_none() {
                        let mut rest: Vec<(u64, usize, usize, f64)> = sd.3.iter().filter(|x| !ids.contains(&x.0)).cloned().collect(); rest.sort_by_key(|x| x.0);
                        groups.push(rest);
                    }
                    groups.into_iter().enumerate().filter(|(g, v)| !v.is_empty() && split_part.is_none_or(|(i, n)| g % n == i)).collect() },
                (Some(sd), Some(v)) => match v.parse::<usize>() {
                    Ok(g) if g >= 1 && k == 0 && !sd.3.is_empty() => { let mut p = sd.3.clone(); p.sort_by_key(|x| x.0);
                        let per = p.len().div_ceil(g).max(1);
                        p.chunks(per).map(|c| c.to_vec()).enumerate().filter(|(g, _)| split_part.is_none_or(|(i, n)| g % n == i)).collect() },
                    _ => Vec::new() },
                _ => Vec::new() };
            let mut split_rows: Vec<Value> = Vec::new();
            // central hunt-3km-slot (AD_DOT_SELF_FLOOR, diagnostic): one more reverse after the dot's own (and its repeat), seeded
            // with the floored y, first seed only. It writes receipt self_floor and nothing else: runs[0], the unfloored dot,
            // stays the valid reading.
            let base_reps = if !split.is_empty() { 1 + split.len() } else if k == 0 && repeat { 2 } else { 1 };
            let floor_rep: Option<usize> = match &selfdot {
                Some(sd) if sd.4.is_some() && k == 0 => {
                    if !split.is_empty() { return Err("AD_DOT_SELF_FLOOR with AD_DOT_SELF_SPLIT is not supported: each adds its own reverses".into()); }
                    Some(base_reps) },
                _ => None };
            // 2. Reverse from y at the window end.
            // a12 truephys-term: a --param-tl first seed is scored on the parameter side (param_x_mty), so the parameter
            // cotangents are zeroed before each of its reverses and read after it.
            let ptl_fams: Vec<String> = if param_tl.is_some() && k == 0 {
                let mut f: Vec<String> = w.params.iter().map(|p| p.fam.clone()).collect(); f.sort(); f.dedup(); f } else { Vec::new() };
            let mut runs = Vec::new();
            for rep in 0..(base_reps + floor_rep.is_some() as usize) {
                let tr = Instant::now();
                for f in &ptl_fams { w.native.as_ref().ok_or("--param-tl needs the native schemes")?.par_zero_bar(&*cuda, f)?; }
                // speed outside: with AD_DOT_SELF the whole arena is overwritten by the self y just below, so the random
                // y is not filled first (it was filled and thrown away: one pass of the card and the host over the arena).
                if selfdot.is_none() { w.fill(&*cuda, sy(seed), ys!())?; } mark("fill_y");
                if let Some(sd) = &selfdot {
                    if rep >= 1 && !split.is_empty() {
                        let mut yg = vec![0u8; sd.0.len()];
                        for (_, lo, nb, _) in &split[rep - 1].1 { yg[*lo..*lo + *nb].copy_from_slice(&sd.0[*lo..*lo + *nb]); }
                        dev::upload(&*cuda, w.sh_base, &yg)?;
                    } else if Some(rep) == floor_rep {
                        let mut yf = sd.0.clone();
                        if let Some((scale, _)) = &sd.4 { self_dot::apply_floor(&mut yf, scale); }
                        w.upload_all(&*cuda, &yf)?;
                    } else { w.upload_all(&*cuda, &sd.0)?; }
                    dev::sync(&*cuda)?; mark("self_y_up");
                }
                // speed ckpt60: with --bstore-content the self dot's y (one more whole copy of the arena in host memory,
                // 26.5 GB at 60 km) is let go once the last reverse that uploads it has it on the card. Nothing reads it after.
                if w.cs_on && rep + 1 == base_reps + floor_rep.is_some() as usize { if let Some(sd) = selfdot.as_mut() { sd.0 = Vec::new(); } }
                if let Some(p) = crate::peer::get() { p.begin_ad(); }
                w.word_probe(&*cuda, "reverse", "seed", steps + 1, None, &json!({"seed":seed,"repeat":rep}))?;
                w.rev_digest(&*cuda, "seed", steps + 1, None, &json!({"seed":seed,"repeat":rep}))?;
                if k==0 && rep==0 {if let Some(p)=opt(&a,"export-vectors") {
                    receipt["vector_exports"]["y"]=w.export_vectors(&*cuda,&bounds[bounds.len()-1],Path::new(p),"y")?;
                }}
                w.trace_last = f64::MAX; w.trace_hits = 0;
                if w.trace { log(json!({"event":"window_trace_y","norms":w.norms(&*cuda, 6)?})); }
                w.entry_ms = 0.0;
                let rows = if let (Some(r), Some(c)) = (&rv, ckpts) {
                    let (r0, t0s) = (w.recompute_steps, w.recompute_s);
                    // speed ckpt60: the whole copies of boundary 0 and of the window end (the dots' weights) are not held
                    // through the sweep; they are read again from the content store after it (the same bytes).
                    let held = (d0.is_some(), dn.is_some());
                    if w.cs_on { d0 = None; dn = None; }
                    let rows = w.sweep_revolve(&mut ex, &plan, &base, &mut frame, &refs, &flats, &bounds[0], &bounds[bounds.len() - 1], &mut ck, &bdir,
                                               r, c, anchors, json!({"seed":seed,"repeat":rep,"sweep":"dot"}))?;
                    if w.cs_on {
                        // The streamed tape keeps its page-locked chunk buffers for the next step (st_pool: about 100 GiB at
                        // 60 km). After the sweep nothing reads them, and the dots that follow need the whole copies of
                        // boundary 0 and of the window end again: the buffers are let go first, so that the end of the leg
                        // is not its memory peak (box fast-1, leg h8: 228 GiB in the sweep, 266 GiB in the dots after it).
                        // A later sweep makes its buffers again (st_take).
                        w.st_pool.clear();
                        if held.0 { d0 = Some(std::sync::Arc::new(w.load(&bounds[0])?)); }
                        if held.1 { dn = Some(w.load(&bounds[bounds.len() - 1])?); }
                    }
                    sweeps.push(json!({"sweep":"dot","seed":seed,"repeat":rep,"recompute_steps":w.recompute_steps - r0,"recompute_s":w.recompute_s - t0s,
                                       "wall_s":tr.elapsed().as_secs_f64()}));
                    rows
                } else {
                let mut cur_data = w.load(&bounds[bounds.len() - 1])?;
                let mut rows = Vec::new();
                for s in (1..=steps).rev() {
                    let (sa0, rd0, tl0) = (crate::staging_allocs(), w.restore_direct, Instant::now()); mark(&format!("ad_step_{s}_begin"));
                    let prev_data = w.load_at(&bounds, s - 1)?;
                    let load_s = tl0.elapsed().as_secs_f64();
                    if s >= 2 { w.prefetch_load(&bounds, s - 2)?; }
                    w.stage_blobs(&mut ex, &base, flat_blobs(&flats[s - 1]), (s >= 2).then(|| flat_blobs(&flats[s - 2])))?;
                    crate::spill::phase(&format!("ad{s}"));
                    let r = if w.fz_tapes.contains_key(&s) { w.fused_reverse_step(&mut ex, &mut frame, s, s == steps, &bounds[s], &cur_data)? }
                            else { w.reverse_step(&mut ex, &base, &mut frame, s, &flats[s - 1], &bounds[s - 1], &prev_data, &bounds[s], &cur_data, anchors)? };
                    log(json!({"event":"window_reverse_step","seed":seed,"repeat":rep,"row":r,"host_leftovers2":{"staging_allocs":crate::staging_allocs() - sa0,
                               "restores_direct":w.restore_direct - rd0,"start_load_s":load_s,"step_wall_s":tl0.elapsed().as_secs_f64()},"host_prof":dev::prof::take().into_iter().filter(|x| x.2 > 0).map(|(k, s, c)| json!([k, s, c])).collect::<Vec<Value>>(),"sync_sites":if dev::sites::on() { json!(dev::sites::take()) } else { Value::Null },"driver_waits_cum":if dev::sites::on() { let n = &capi::xfer::N; json!({"ctx":capi::xfer::get(&n.ctx),"stream":capi::xfer::get(&n.stream),"event":capi::xfer::get(&n.event),"d2h_calls":capi::xfer::get(&n.d2h_n),"h2d_calls":capi::xfer::get(&n.h2d_n),"h2d_bytes":capi::xfer::get(&n.h2d),"d2h_bytes":capi::xfer::get(&n.d2h)}) } else { Value::Null }}));
                    rows.push(r);
                    w.word_probe(&*cuda, "reverse", "step", s, None, &json!({"seed":seed,"repeat":rep}))?;
                    w.rev_digest(&*cuda, "step", s, None, &json!({"seed":seed,"repeat":rep}))?;
                    cur_data = prev_data; mark(&format!("ad_step_{s}_end"));
                }
                drop(cur_data);
                rows
                };
                LAST_ABS_TERMS.store(0, std::sync::atomic::Ordering::Relaxed);
                mark("sweep_done");
                // speed tape60: the last reverse sweep of the run is over (no further seed, repeat or objective sweep)
                if k + 1 == seeds.len() && rep + 1 == base_reps + floor_rep.is_some() as usize && objs.is_empty() { w.st_release(&ex); }
                let (pa, qa, bad_a) = w.dots(&*cuda, sx(seed), xs!(), sy(seed), ys!())?; mark("dots_ad");
                let pa64 = f64::from_bits(LAST_F64.load(std::sync::atomic::Ordering::Relaxed));
                let (pa, qa, bad_a) = match crate::peer::get() {
                    Some(p) => { let g = p.reduce_sum(&format!("ad{seed}.{rep}"), &[pa, qa, bad_a as f64])?;
                                 log(json!({"event":"peer_reverse_dot","seed":seed,"repeat":rep,"rank":p.rank,"local":[pa, qa, bad_a],"global":g}));
                                 (g[0], g[1], g[2] as u64) },
                    None => (pa, qa, bad_a) };
                // a12 truephys-term: the parameter side of a --param-tl first seed (x = 0 in the state, d ln p = c).
                let (pa, pa_state) = if ptl_fams.is_empty() { (pa, None) } else {
                    let mut bars: HashMap<String, Vec<f64>> = HashMap::new();
                    for f in &ptl_fams { bars.insert(f.clone(), w.native.as_ref().unwrap().par_bar(&*cuda, f)?); }
                    let v = param_x_mty(&w.params, &bars, param_tl.as_deref().unwrap_or(&[]));
                    let v = match crate::peer::get() { Some(p) => p.reduce_sum(&format!("ptl{seed}.{rep}"), &[v])?[0], None => v };
                    (seed_x_mty(param_tl.as_deref(), k, pa, Some(v))?, Some(pa)) };
                let qa = if pa_state.is_some() { pa - pt } else { qa };
                let pa64 = if pa_state.is_some() { pa } else { pa64 };
                if rep >= 1 && !split.is_empty() {
                    let (gno, grp) = (split[rep - 1].0, &split[rep - 1].1); let pg: f64 = grp.iter().map(|x| x.3).sum();
                    // with AD_DOT_ALLOC=1: abs_terms = sum |x_i (M^T y)_i| of this reverse, the size of the terms that cancel to
                    // x_MTy; diff / abs_terms is the error per unit of terms (single-precision rounding is about 6e-8 of a term).
                    let at = f64::from_bits(LAST_ABS_TERMS.load(std::sync::atomic::Ordering::Relaxed));
                    let mut row = json!({"event":"window_self_split","group":gno,"groups":split.len(),"allocations":grp.iter().map(|x| x.0).collect::<Vec<_>>(),
                        "y_Mx":pg,"x_MTy":pa,"diff":pa - pg,"rel":rel(pg, pa),"nonfinite":bad_a,"wall_s":tr.elapsed().as_secs_f64(),
                        "abs_terms":if at > 0.0 { json!(at) } else { Value::Null },"rel_terms":if at > 0.0 { json!((pa - pg).abs() / at) } else { Value::Null }});
                    if self_dot::roundings_on() { row["miss_roundings"] = json!(self_dot::roundings(pg, pa, at)); }
                    log(row.clone()); split_rows.push(row); receipt["self_split"] = json!(split_rows);
                    continue;
                }
                if Some(rep) == floor_rep {
                    // central hunt-3km-slot: the floored reading, beside the unfloored dot (needs AD_DOT_ALLOC=1 for its abs terms).
                    let at = f64::from_bits(LAST_ABS_TERMS.load(std::sync::atomic::Ordering::Relaxed));
                    let yfl = selfdot.as_ref().and_then(|sd| sd.4.as_ref()).map(|f| f.1).unwrap_or(f64::NAN);
                    let row = json!({"event":"window_self_floor","form":"self dot with the floored unit weight (AD_DOT_SELF_FLOOR); a diagnostic, never the valid reading",
                        "y_Mx":yfl,"x_MTy":pa,"diff":pa - yfl,"rel":rel(yfl, pa),"nonfinite":bad_a,
                        "abs_terms":if at > 0.0 { json!(at) } else { Value::Null },"rel_terms":if at > 0.0 { json!((pa - yfl).abs() / at) } else { Value::Null },
                        "miss_roundings":self_dot::roundings(yfl, pa, at),"floored_units":selfdot.as_ref().and_then(|sd| sd.4.as_ref()).map(|f| f.0.len()),
                        "unfloored":"dots[0]: y_Mx, x_MTy, rel_aligned (the valid reading, unchanged)","wall_s":tr.elapsed().as_secs_f64()});
                    log(row.clone()); receipt["self_floor"] = row;
                    continue;
                }
                if k==0 && rep==0 {if let Some(p)=opt(&a,"export-vectors") {
                    receipt["vector_exports"]["ad"]=w.export_vectors(&*cuda,&bounds[0],Path::new(p),"ad")?;
                }}
                let abs_terms = f64::from_bits(LAST_ABS_TERMS.load(std::sync::atomic::Ordering::Relaxed));
                let abs_terms = match crate::peer::get() { Some(p) if abs_terms_wanted() => p.reduce_sum(&format!("adabs{seed}.{rep}"), &[abs_terms])?[0], _ => abs_terms };
                let result = w.download_all(&*cuda)?; mark("dots_ad_download");
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
                let mut row = json!({"seed":seed,"repeat":rep,"x_MTy":pa,"x_MTy_f64":pa64,"x_MTy_minus_y":qa,"nonfinite":bad_a,"abs_terms":abs_terms,
                    "wall_s":tr.elapsed().as_secs_f64(),"adjoint_entry_s":w.entry_ms / 1e3,"steps":rows,
                    "result_sha256":result_sha256(&result),"result_bytes":result.len()}); mark("result_sha");
                if let Some(s) = pa_state {
                    row["x_side"] = json!("--param-tl first seed: x = 0 in the state, d ln p = c; x_MTy = sum_k c_k p0_k (M^T y)_k from the parameter cotangents (param_x_mty)");
                    row["x_MTy_against_unused_random_x"] = json!(s);
                    row["abs_terms"] = Value::Null;   // the state terms do not enter this dot
                }
                if rep == 0 && k == 0 && !taylor.is_empty() {
                    // <x', M^T y> with x' = the first seed's direction without address-bearing allocations (the Taylor direction).
                    let gx = if let Some((_, t)) = &tdir { w.sparse_dot(&*cuda, t)? + w.dense_dot(&*cuda, &tdense)? } else {
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
            let rq = if selfdot.is_some() { rp } else { rq };   // AD_DOT_SELF: the (M - I) form has no meaning for y = W M x
            let abs_terms_ad = runs[0]["abs_terms"].as_f64().unwrap_or(0.0);
            let conditioning = if abs_terms_tl > 0.0 && abs_terms_ad > 0.0 {
                json!({"abs_terms_tl":abs_terms_tl,"abs_terms_ad":abs_terms_ad,"cancellation":pt.abs() / abs_terms_tl,
                       "rel_terms":(pt - pa).abs() / abs_terms_tl.max(abs_terms_ad)}) } else { Value::Null };
            let rel_terms = (abs_terms_tl > 0.0 && abs_terms_ad > 0.0).then(|| (pt - pa).abs() / abs_terms_tl.max(abs_terms_ad));
            let (pass_rel, valid_rel, measure) = dot_verdict(selfdot.is_some(), rp, rq, rel_terms);
            let pass = pass_rel <= DOT_F32 && bad_t == 0;
            // inst (AD_DOT_F64=1): both sides again, compensated binary64 on the host, beside the default reading.
            let mut f64r = if std::env::var("AD_DOT_F64").as_deref() == Ok("1") {
                let t64 = match &selfdot { Some(sd) => sd.2["y_Mx_compensated"].as_f64().unwrap_or(f64::NAN), None => pt64_std };
                let a64 = runs[0]["x_MTy_f64"].as_f64().unwrap_or(f64::NAN);
                json!({"sum":"both sides again on the host: float32 words widened, products exact in binary64, Neumaier-compensated sum",
                       "y_Mx":t64,"x_MTy":a64,"rel":rel(t64, a64),"default_y_Mx":pt,"default_x_MTy":pa,"default_rel":rp,
                       "default_sum":"binary64 too (win_dot2_f32 widens each float32 word; fixed strided order, not compensated); the float32 is in the words, not in the sum"})
            } else { Value::Null };
            // central hunt-3km-slot (AD_DOT_ROUNDINGS=1): the miss in float32 roundings of the absolute terms, every reading of this dot.
            let roundings_at = if abs_terms_ad > 0.0 { abs_terms_tl.max(abs_terms_ad) } else { 0.0 };
            if self_dot::roundings_on() { if let Some(o) = f64r.as_object_mut() {
                let (t64, a64) = (o["y_Mx"].as_f64().unwrap_or(f64::NAN), o["x_MTy"].as_f64().unwrap_or(f64::NAN));
                o.insert("miss_roundings".into(), json!(self_dot::roundings(t64, a64, roundings_at)));
            } }
            let mut dentry = json!({"seed":seed,"f64":f64r,"y_Mx":pt,"x_MTy":pa,"rel_plain":rp,"y_Mx_minus_x":qt,"x_MTy_minus_y":qa,"rel_aligned":rq,"conditioning":conditioning,
                             "pass":pass,"tangent_wall_s":t_tl,"reverse":runs,"bar_rel":valid_rel,
                             "verdict":{"measure":measure,"rel":valid_rel,"pass_rel":pass_rel,"rel_terms":rel_terms,
                                        "rule":"pass: pass_rel <= 2e-5 and no non-finite tangent word; the program's bar (valid_derivative) reads bar_rel. A random-y dot (no AD_DOT_SELF) is read on rel_terms = |y_Mx - x_MTy| / max(abs_terms_tl, abs_terms_ad), the scale of its float32 rounding; the self form on rel_aligned, whose terms are its y_Mx"}});
            // F14: every dot states its miss in roundings of its terms (no longer only with AD_DOT_ROUNDINGS=1).
            dentry["miss_roundings"] = json!(self_dot::roundings(pt, pa, roundings_at));
            dentry["miss_roundings_rule"] = json!(if roundings_at > 0.0 { "(x_MTy - y_Mx) / (2^-24 * max(abs_terms_tl, abs_terms_ad)); the self form's abs_terms_tl is its y_Mx" }
                                                  else { "null: no absolute terms (AD_DOT_TERMS=0, or a dot that has none)" });
            dots.push(dentry);
            log(json!({"event":"window_dot","seed":seed,"rel_plain":rp,"rel_aligned":rq,"pass":pass}));
            { let (n, h, b) = crate::cuda::poison_totals(); if h != 0 { eprintln!("{}", json!({"event":"alloc_poison_totals","allocations":n,"filled":h,"filled_bytes":b})); } }
            receipt["dots"] = json!(dots);
            // The form of this check, in one word and one sentence: `self` seeds the reverse with the tangent's own result
            // (cancellation-free, units as self_dot.units says); `independent` is a seeded random y not made from the tangent
            // with an additive x (--direction fieldxy, no AD_DOT_SELF); any other direction is the `standard` random y.
            receipt["check_form"] = match (&selfdot, opt(&a, "direction")) {
                (Some(sd), _) => json!({"form":"self","units":sd.2["units"].clone(),
                    "what":"y is the tangent result at unit weight per unit (AD_DOT_SELF=1): the dot that cannot cancel"}),
                (None, Some("fieldxy")) => json!({"form":"independent",
                    "what":"y is a seeded random vector not made from the tangent, x is additive (--direction fieldxy, no AD_DOT_SELF)"}),
                (None, d) => json!({"form":"standard","what":format!("the seeded random y of --direction {}", d.unwrap_or("relative"))}) };
            // central hunt-3km-slot: a run with AD_DOT_SELF_FLOOR or AD_DOT_ROUNDINGS names the form it read and the unchanged valid rule.
            let floor_spec = self_dot::FloorSpec::from_env()?;
            if self_dot::roundings_on() || floor_spec.is_some() {
                receipt["dot_form"] = self_dot::form_receipt(selfdot.is_some(), selfdot.as_ref().and_then(|sd| sd.2["fields_spec"].as_str()),
                                                             std::env::var("AD_DOT_SELF_ONLY").ok().as_deref(), floor_spec.as_ref(), self_dot::roundings_on());
            }
            if let Some((pk, pa, pf)) = plant() { receipt["planted_error"] = json!({"switch":"AD_PLANT","kernel":pk,"argument":pa,"factor":pf,
                "side":"reverse only: the change this kernel's adjoint entries make to that argument's cotangent view is scaled by the factor at every launch; the tangent is untouched",
                "launches":PLANT_HITS[0].load(std::sync::atomic::Ordering::Relaxed),"words_changed":PLANT_HITS[1].load(std::sync::atomic::Ordering::Relaxed),"valid":false}); }
            if let Ok(v) = DEAD_REV.lock() { if !v.is_empty() { receipt["dead_poison_rev"] = json!(*v); } }
            { receipt["out_of_core"] = crate::spill::receipt(); fs::write(out.join("window.json"), serde_json::to_vec_pretty(&receipt).unwrap()) }.map_err(|e| e.to_string())?;
        }
        receipt["per_step_forward"] = json!(per_step);
        // anyparam: what this window's forwards followed that the recording did not (empty on the recorded trajectory).
        receipt["live_trajectory"] = json!({"base_unchecked":unchecked,"lw_live_regroups":w.lw_follow,"own_commands":w.synth.len()});
        if let Some(p) = crate::peer::get() { receipt["peer"] = p.stats_json(); log(json!({"event":"peer_stats","stats":receipt["peer"]})); }
        if crate::peer::get().is_some() { receipt["peer_guards_tolerated"] = json!({"on":crate::peer::tolerate_guards(),"rows":w.peer_tolerated}); }
        // 3. Zero direction: tangent from x = 0 over the whole window.
        if opt(&a, "zero").map(|v| v != "0").unwrap_or(true) {
            let t0 = Instant::now();
            let d = w.load(&bounds[0])?; w.restore(&mut ex, &mut frame, &bounds[0], &d)?; drop(d);
            w.zero(&*cuda)?;
            if let Some(p) = crate::peer::get() { p.begin_tl(); }
            // anyparam: on a moved base trajectory the zero-direction pass is unchecked like the base pass.
            for s in 1..=steps { if unchecked { w.forward_step_c(&mut ex, &plan, &base, &mut frame, None, s, true)?; } else { w.forward_step(&mut ex, &plan, &base, &mut frame, &refs, s, true)?; } }
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
        // truth: --objective-reverse 0: no reverse; the tangent's <w, M x> is the Taylor rows' prediction.
        let no_obj_rev = opt(&a, "objective-reverse") == Some("0");
        if no_obj_rev && !fd_only {
            if !tangent_dir || objs.is_empty() || taylor.is_empty() || dump_state.is_empty() {
                return Err("--objective-reverse 0 needs --tangent-taylor-dir 1, --objective, --taylor and --dump-state".into());
            }
            taylor_gx = Some(receipt["tangent_dir"]["w_Mx"].as_f64().ok_or("--objective-reverse 0: the tangent pass left no <w, M x>")?);
            receipt["objective"] = json!({"files":objs.iter().map(|o| o.0.clone()).collect::<Vec<_>>(),"names":objs.iter().map(|o| o.1.clone()).collect::<Vec<_>>(),
                "name":objs[0].1,"note":"--objective-reverse 0: no reverse; Taylor rows are scored against the tangent's <w, M x>"});
        }
        if let (Some((pth, oname, _)), false) = (objs.first(), fd_only || no_obj_rev) {
            let pth = pth.as_str();
            let doc = json!({"name": oname});
            // hunt grad-only: the seed loop left before the reverse was prepared (--no-tangent 1 or --tangent-only 1), so
            // the objective sweep ran on the --tape-gib tape with no anchors decided and fell back to split segments on
            // every step (door d3k-c sensitivity: one_pass fallback_steps 4, tape null). The same preparation the dot
            // reverse makes, for the plain, streamed and checkpointed forms (the objective sweep never takes a fused tape).
            // A window that ran no tangent (--no-tangent 1) has no tape estimates (every est is 0), so neither a sized
            // one-pass tape nor a segment guess can be made: there the sweep takes the streamed tape (one replay per step,
            // the chunks through page-locked RAM; the same taped words and walk-back order, so the same gradient).
            // Measured (hunt grad-only r1, a test machine, 120 km pk120e4, 3 steps, one card): the --tape-gib tape went segmented
            // (base: 16.7, 9.6, 12.1 s); the tape sized to the card budget (45.6 GB, one pass) was slower still (25.9,
            // 19.0, 21.6 s); the streamed tape 9.6, 6.5, 7.7 s against the normal run's objective sweep 11.1, 7.7, 6.9 s.
            if !rev_prepared {
                let t_prep = Instant::now();
                w.fuse_rec = false;
                crate::spill::phase("before-objective-reverse");
                let no_est = w.pack && !flats.is_empty() && flats.iter().flatten().all(|x| x.est == 0);
                let st_auto = no_est && !w.st_on;
                if st_auto { w.st_on = true; }
                let streamed = w.st_on && w.pack;
                let cap_now = format!("{}", w.tape.bytes as f64 / (1u64 << 30) as f64);
                if w.one_pass { w.size_tape(&ex, &flats, if streamed { Some(cap_now.as_str()) } else { opt(&a, "tape-max-gib") }, opt(&a, "tape-reserve-gib"))?; }
                if !streamed { w.ahead_maybe(&ex, &flats, anchors, opt(&a, "tape-reserve-gib"), true, steps, plan_live_bound)?; }
                if streamed { w.st_plan(&ex, &flats, "before the objective reverse")?; }
                let prep = json!({"streamed":streamed,"streamed_because_no_tape_estimates":st_auto,"tape_bytes":w.tape.bytes,"wall_s":t_prep.elapsed().as_secs_f64()});
                log(json!({"event":"window_objective_reverse_prepared","info":prep}));
                receipt["objective_reverse_prepared"] = prep;
            }
            let tr = Instant::now();
            w.zero(&*cuda)?;
            // No tangent pass ran, so nothing has counted the labels: this sweep's taping does (rev_count_once).
            if no_tangent { w.rev_count = Some(Default::default()); }
            // ranks (R1): every reverse sweep gets its own cotangent message names. THE BREAKAGE THIS PREVENTS: a rank
            // that reaches its add before the peer has rewritten the file would add the previous sweep's cotangent.
            if let Some(p) = crate::peer::get() { p.begin_ad(); }
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
                w.dump_bars_after(&ex.cuda, s, "objective")?;
                rows.push(json!({"step":s,"wall_s":r["wall_s"]}));
                cur_data = prev_data;
            }
            drop(cur_data);
            rows
            };
            if let Some(seen) = w.rev_count.take() { log(json!({"event":"window_labels_counted_in_reverse","commands":seen.len(),"labels":w.labels.len(),
                "refused_kinds":refused_labels(&w.labels).len()})); }
            let mut dumps = Vec::new();
            for id in dump_grad_arg.split(',').filter_map(|x| x.trim().parse::<u64>().ok()) {
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
            // ranks (R2): in the tangent-direction mode <x, M^T w> needs no --taylor row (no perturbed forward), and on
            // a rank plan it is the rank-ordered sum: x has the same value on the owner and on every halo copy, and the
            // gradient of a mesh word is the sum of the cotangents of all its local copies.
            if !taylor.is_empty() || (tangent_dir && tdir.is_some()) {
                let gx = if let Some((_, t)) = &tdir { w.sparse_dot(&*cuda, t)? + w.dense_dot(&*cuda, &tdense)? } else {
                    w.taylor_mask = true;
                    let (gx, _, _) = w.dots(&*cuda, sx(seeds[0]), xs!(), sx(seeds[0]), xs!())?;
                    w.taylor_mask = false;
                    gx
                };
                let gx = match crate::peer::get() {
                    Some(p) => { let g = p.reduce_sum("xmtw", &[gx])?; log(json!({"event":"peer_x_MTw","rank":p.rank,"local":gx,"global":g[0]})); g[0] },
                    None => gx };
                if !taylor.is_empty() { taylor_gx = Some(gx); }
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
                    // anyparam: the per-column width comes from the carrier image (its par_bar global is 8 slots wide: 4,096
                    // columns in the first carrier images, more in images built for larger meshes), not from a number here.
                    let pw = bars[&p.fam].len() / 8;
                    let col = &bars[&p.fam][p.slot * pw..(p.slot + 1) * pw];
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
                    if let Some(p) = crate::peer::get() { p.begin_ad(); }   // ranks (R1): see the objective reverse
                    let terms: Vec<(u64, usize, f32)> = match &forms[oi] { Some(f) => {
                        let g = f.eval(&base_words(&bounds[bounds.len() - 1], dn.as_deref().ok_or("window-end state not loaded")?, terms)?).1;
                        terms.iter().zip(&g).map(|(t, &x)| (t.0, t.1, x as f32)).collect() }, None => terms.clone() };
                    for &(al, wd, wt) in &terms {
                        let el = *w.base_el.get(&al).ok_or_else(|| format!("objective allocation {al} has no shadow"))?;
                        dev::upload(&*cuda, w.sh(el + wd as i64), &wt.to_le_bytes())?;
                    }
                    for f in &pfams { w.native.as_ref().unwrap().par_zero_bar(&*cuda, f)?; }
                    // tune mem: with --checkpoints this reverse is the revolve sweep, like the objective reverse above.
                    if let (Some(r), Some(c)) = (&rv, ckpts) {
                        let (r0, t0s) = (w.recompute_steps, w.recompute_s);
                        w.sweep_revolve(&mut ex, &plan, &base, &mut frame, &refs, &flats, &bounds[0], &bounds[bounds.len() - 1], &mut ck, &bdir,
                                        r, c, anchors, json!({"sweep":format!("objective-{oi}")}))?;
                        sweeps.push(json!({"sweep":format!("objective-{oi}"),"recompute_steps":w.recompute_steps - r0,"recompute_s":w.recompute_s - t0s,"wall_s":tr2.elapsed().as_secs_f64()}));
                    } else {
                    let mut cur_data = w.load(&bounds[bounds.len() - 1])?;
                    for s in (1..=steps).rev() {
                        let prev_data = w.load_at(&bounds, s - 1)?;
                        if s >= 2 { w.prefetch_load(&bounds, s - 2)?; }
                        w.stage_blobs(&mut ex, &base, flat_blobs(&flats[s - 1]), (s >= 2).then(|| flat_blobs(&flats[s - 2])))?;
                        w.reverse_step(&mut ex, &base, &mut frame, s, &flats[s - 1], &bounds[s - 1], &prev_data, &bounds[s], &cur_data, anchors)?;
                        cur_data = prev_data;
                    }
                    drop(cur_data);
                    }
                    let mut bars: HashMap<String, Vec<f64>> = HashMap::new();
                    for f in &pfams { bars.insert(f.clone(), w.native.as_ref().unwrap().par_bar(&*cuda, f)?); }
                    let mut grads = Vec::new();
                    let mut adj_dot = 0f64;
                    for (k, p) in w.params.iter().enumerate() {
                        // anyparam: the per-column width comes from the carrier image (its par_bar global is 8 slots wide: 4,096
                    // columns in the first carrier images, more in images built for larger meshes), not from a number here.
                    let pw = bars[&p.fam].len() / 8;
                    let col = &bars[&p.fam][p.slot * pw..(p.slot + 1) * pw];
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
                    { receipt["out_of_core"] = crate::spill::receipt(); fs::write(out.join("window.json"), serde_json::to_vec_pretty(&receipt).unwrap()) }.map_err(|e| e.to_string())?;
                }
            }
            log(json!({"event":"window_objective_reverse","wall_s":tr.elapsed().as_secs_f64()}));
            { receipt["out_of_core"] = crate::spill::receipt(); fs::write(out.join("window.json"), serde_json::to_vec_pretty(&receipt).unwrap()) }.map_err(|e| e.to_string())?;
        }
        // bhunt real-dir-plant: the planted-error guard runs here, after the objective reverse (3b), not after the
        // dot loop. THE BREAKAGE THIS PREVENTS: with --tangent-only 1 --tangent-taylor-dir 1 the objective reverse is the
        // window's only reverse, so a guard placed before it refused every planted run of the realistic-direction check
        // (never ran as a reverse entry) although the kernel runs 180 times in that reverse. The receipt count is
        // rewritten here so it covers every reverse of the window.
        if let Some((pk, pa, pf)) = plant() { receipt["planted_error"] = json!({"switch":"AD_PLANT","kernel":pk,"argument":pa,"factor":pf,
            "side":"reverse only: the change this kernel's adjoint entries make to that argument's cotangent view is scaled by the factor at every launch; the tangent is untouched",
            "launches":PLANT_HITS[0].load(std::sync::atomic::Ordering::Relaxed),"words_changed":PLANT_HITS[1].load(std::sync::atomic::Ordering::Relaxed),"valid":false}); }
        if plant().is_some() && PLANT_HITS[0].load(std::sync::atomic::Ordering::Relaxed) == 0 { return Err("AD_PLANT: the named kernel never ran as a reverse entry in this window: nothing was planted".into()); }
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
                    w.hold.mode = 1;
                    let mut err = None;
                    for s in 1..=steps {
                        if let Err(e) = w.forward_step_c(&mut ex, &plan, &base, &mut frame, None, s, false) { err = Some(format!("step {s}: {e}")); break; }
                    }
                    w.par_set(&*cuda, None, None)?;
                    w.hold.mode = 0;
                    let hold_side = if w.hold.on() { Some(w.hold.take()) } else { None };
                    let mut djs = Vec::new();
                    // anyparam: an objective with a "form" (misfit, dBZ) carries weight 0 in its word list (its value comes
                    // from Form::eval), so the sparse sum read J(+h) - J(0) = 0 for every constant while half a million words
                    // had changed (a test machine <local path>). obj_dj handles both kinds, and the
                    // words come in one download per allocation, not one per word.
                    if err.is_none() {
                        dev::sync(&ex.cuda)?;
                        for (i, (_, _, terms)) in objs.iter().enumerate() {
                            let (q, qb) = live_words(&ex, terms)?;
                            if let Some(k) = q.iter().position(|x| !x.is_finite()) { return Err(format!("objective word {}:{} not finite", terms[k].0, terms[k].1)); }
                            let p = base_words(&bounds[bounds.len() - 1], &end, terms)?;
                            let ch = qb.iter().zip(&p).filter(|(b, v)| **b != (**v as f32).to_bits()).count() as u64;
                            djs.push((obj_dj(i, &q, &p), ch));
                        }
                    }
                    log(json!({"event":"param_fd_side","hold":hold_side,"param":pname,"h":h,"sign":sgn,"p":pv,"dJ":djs.iter().map(|x| x.0).collect::<Vec<_>>(),"error":err,"wall_s":t0.elapsed().as_secs_f64()}));
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
                { receipt["out_of_core"] = crate::spill::receipt(); fs::write(out.join("window.json"), serde_json::to_vec_pretty(&receipt).unwrap()) }.map_err(|e| e.to_string())?;
            }
            let mut conv = Vec::new();
            for (i, (_, name, _)) in objs.iter().enumerate() {
                let q: Vec<f64> = rows.iter().filter_map(|r| r["objectives"][i]["centred_dJdlnp"].as_f64()).collect();
                let worst = q.windows(2).map(|p| ((p[1] - p[0]) / p[1]).abs()).fold(0f64, f64::max);
                conv.push(json!({"name":name,"centred_dJdlnp":q,"max_rel_change":worst}));
            }
            all_fd.push(json!({"param":pname,"p0":p0,"rows":rows,"convergence":conv}));
            receipt["param_fd"] = json!(all_fd);
            { receipt["out_of_core"] = crate::spill::receipt(); fs::write(out.join("window.json"), serde_json::to_vec_pretty(&receipt).unwrap()) }.map_err(|e| e.to_string())?;
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
                if let Some((_, t)) = &tdir {
                    w.perturb_terms(&ex, h, t)?;
                    if !tdense.is_empty() { let moved = w.perturb_dense(&ex, h, &tdense)?; ph.insert("dense_words_moved", moved as f64); }
                } else {
                    w.taylor_mask = true;
                    w.perturb(&ex, h, sx(s0), xs!())?;
                    w.taylor_mask = false;
                }
                ph.insert("perturb_s", tp.elapsed().as_secs_f64());
                let mut fwd_err = None;
                let (mut rg_step, mut rg_field) = (Vec::new(), vec![0u64; regime.len()]);
                let (tf, mut rg_s) = (Instant::now(), 0f64);
                w.hold.mode = 1;
                // any-mesh (taylor-digest): a Taylor row's perturbed forward has no reference and stores no boundary,
                // so nothing reads its boundary digests: J(h) comes from the live device words below, and the next row
                // (or anything after the rows) starts from a restore, which puts back a stored pointer history. THE
                // BREAKAGE THIS PREVENTS: every such step hashed the whole state on one host thread with the card idle
                // (120 km, 3 steps: 48 to 53 s of each 50 to 56 s row; five rows = 250 s of a 354 s adjoint-test window).
                // The skip is the library's (unchecked_digest_skip: the history is marked stale, and a digest or a stored
                // boundary that would read it refuses by name). HEXAD_TL_UNCHECKED_DIGEST=1 keeps the digests.
                // truth: AD_TAYLOR_DIGEST_SKIP=1 (default off): a perturbed Taylor forward has no reference
                // and stores no boundary, so its boundary digest (a download and scan of every carried block: 45 of the
                // 50 s of a 4-step 120 km row, a test machine <local path> phases_s) is read by
                // nothing. The step skips it exactly as the C-ABI tangent step does (unchecked_digest_skip); the pointer
                // history is stale until the next row's restore. The row's words must equal the unskipped row's.
                // fold (merge of any-mesh 5287674c and truth a4c01591, which both skip these unread digests): the skip is ON by
                // default, as published since LATEST 20261109T2350Z-7904c5c3 (any-mesh); HEXAD_TL_UNCHECKED_DIGEST=1 keeps the
                // digests; truth's AD_TAYLOR_DIGEST_SKIP=1 still asks for the skip by name and wins over the keep switch.
                let taylor_digest_skip = std::env::var("AD_TAYLOR_DIGEST_SKIP").as_deref() == Ok("1")
                    || std::env::var("HEXAD_TL_UNCHECKED_DIGEST").as_deref() != Ok("1");
                ex.unchecked_digest_skip = taylor_digest_skip;
                // a development line (round 3): AD_TAYLOR_ROW_CHECK=1 (diagnostic, default off): the rerun's step boundaries
                // are compared with the recording, as the base forward's are. A rerun with a nudge of size zero must equal
                // the plain run bit for bit, so with --taylor 0 the first refusal names the step and the memory block where
                // the rerun rows themselves leave the plain run (TABLE 2026-11-11T03:10Z: every nudged rerun of w120x30
                // differed from the plain run where the nudge cannot have reached). A nudged row refuses by design.
                let row_check = std::env::var("AD_TAYLOR_ROW_CHECK").as_deref() == Ok("1");
                let row_refs = if row_check { Some(&refs) } else { None };
                let mut row_equal = 0usize;
                w.row_live = row_check;
                // truth: --dump-at in a Taylor row: the perturbed words at each named step boundary, read where
                // the recorder map puts the field at that boundary (state-h{row}-s{step}-{allocation}-{offset}.f32), so one
                // window gives the real response at every lead beside the tangent's (tl-s{step}-..).
                let mut row_step_dumps: Vec<Value> = Vec::new();
                let mut dump_s = 0f64;
                for s in 1..=steps {
                    if let Err(e) = w.forward_step_c(&mut ex, &plan, &base, &mut frame, row_refs, s, false) { fwd_err = Some(format!("step {s}: {e}")); break; }
                    row_equal += 1;
                    if dump_at.iter().any(|x| x.0 == s) {
                        let tdmp = Instant::now();
                        dev::sync(&ex.cuda)?;
                        for &(_, id, off, len) in dump_at.iter().filter(|x| x.0 == s) {
                            let a0 = ex.allocations.get(&id).ok_or_else(|| format!("--dump-at: allocation {id} is not live at boundary {s} of the Taylor row h={h}"))?;
                            if off + len > a0.bytes { return Err(format!("--dump-at {s}:{id}:{off}: outside allocation {id} ({} bytes)", a0.bytes)); }
                            let b = dev::download(&ex.cuda, a0.address + off as u64, len)?;
                            let mut r = write_state(&out, format!("state-h{}-s{s}-{id}-{off}.f32", rows.len()), &b)?;
                            r["allocation"] = json!(id); r["byte_offset"] = json!(off); r["step"] = json!(s); r["h"] = json!(h);
                            row_step_dumps.push(r);
                        }
                        dump_s += tdmp.elapsed().as_secs_f64();
                    }
                    if !regime.is_empty() {
                        let tr = Instant::now();
                        let d = regime_diff(&regime_base[s - 1], &read_regime(&ex)?);
                        for (f, n) in rg_field.iter_mut().zip(&d) { *f += n; }
                        rg_step.push(d.iter().sum::<u64>());
                        rg_s += tr.elapsed().as_secs_f64();
                    }
                }
                ex.unchecked_digest_skip = x0_digest_skip;
                if !row_step_dumps.is_empty() {
                    if state_dumps.get("row_steps").is_none() { state_dumps["row_steps"] = json!([]); }
                    state_dumps["row_steps"].as_array_mut().unwrap().push(json!({"h":h,"fields":row_step_dumps}));
                    receipt["state_dumps"] = state_dumps.clone();
                }
                ph.insert("step_dumps_s", dump_s);
                ph.insert("digest_skipped", if taylor_digest_skip && !row_check { 1.0 } else { 0.0 });
                w.row_live = false;
                w.hold.mode = 0;
                let hold_row = if w.hold.on() { Some(w.hold.take()) } else { None };
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
                        if let Some(hr) = &hold_row { r["hold"] = hr.clone(); }
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
                let mut row = row;
                if row_check { row["row_check"] = json!({"switch":"AD_TAYLOR_ROW_CHECK","boundaries":steps,"boundaries_equal_to_the_recording":row_equal}); }
                log(json!({"event":"window_taylor","row":row}));
                rows.push(row);
                receipt["taylor"] = json!({"objective":if obj_terms.is_empty() { json!("J(x0) = <y, x_N>, y = seed-1 relative direction at the window end; declared TEMPO residue words excluded") } else { receipt["objective"]["name"].clone() },
                    "direction":if let Some((n, t)) = &tdir { json!({"fixed":opt(&a, "taylor-dir"),"name":n,"terms":t.len(),"dense_words":tdense.iter().map(|x| x.2.len() / 4).sum::<usize>()}) } else { json!("seed-1 relative direction x, address-bearing and fixed allocations masked") },
                    "x_MTy_masked":gx,"fd_only":fd_only,"rows":rows});
                { receipt["out_of_core"] = crate::spill::receipt(); fs::write(out.join("window.json"), serde_json::to_vec_pretty(&receipt).unwrap()) }.map_err(|e| e.to_string())?;
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
        let refused = refused_labels(&w.labels);
        receipt["commands_by_rule"] = json!(w.labels);
        receipt["refused"] = json!(refused);
        // dycore (round 2): the validity rule of the program, in the receipt of the run itself: a refused kind
        // that --refuse did not name ran as identity without anyone asking (label "REFUSED:<kernel>: <why>").
        let beyond: Vec<&String> = refused.keys().copied().filter(|k| {
            let name = k.trim_start_matches("REFUSED:").split(": ").next().unwrap_or("");
            !w.refuse.contains_key(name)
        }).collect();
        receipt["refused_beyond_refuse_list"] = json!(beyond);
        // D1: how each --refuse kind ran, and the ones that kept identity because what they write is unknown.
        receipt["refuse_list_rule"] = json!(w.refuse_how);
        receipt["refuse_list_identity"] = json!(w.refuse_how.iter().filter(|(_, h)| crate::wrules::refuse_ran_identity(h)).map(|(k, _)| k).collect::<Vec<_>>());
        // dycore: entries that ran a captured numeric grid far below their forward launch (deriv::geom_report).
        let geom = deriv::geom_report();
        let flagged: Vec<&Value> = geom.as_array().into_iter().flatten().filter(|r| r["literal_undersized"] == true).collect();
        receipt["literal_undersized_grids"] = json!(flagged);
        if let Ok(p) = std::env::var("AD_GEOM_CENSUS") { if !p.is_empty() {
            fs::write(&p, serde_json::to_vec_pretty(&geom).map_err(|e| e.to_string())?).map_err(|e| format!("AD_GEOM_CENSUS {p}: {e}"))?;
        } }
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
            "content":if w.cs_on { json!({"piece_bytes":crate::cstore::PIECE,"window_bytes":CS_WIN,"live_bytes_peak":crate::cstore::peak_bytes(),"live_bytes_end":crate::cstore::live_bytes(),
                "bytes_copied_in":crate::cstore::made_bytes(),"captures":w.cs_n[0],"restores_uploaded":w.cs_n[1],"restores_not_needed":w.cs_n[2],
                "tape_chunk_buffers_never_taken_let_go":w.cs_spares,"truths_from_device":w.cs_n[3],"truths_from_state":w.cs_n[4],"redo_compares":w.cs_n[5],"truth_exact_bytes_peak":w.cs_exact_peak,
                "scan_download_s":w.cs_s[0],"capture_build_s":w.cs_s[1],"restore_fill_s":w.cs_s[2],"restore_upload_s":w.cs_s[3],"truth_s":w.cs_s[4],"redo_compare_s":w.cs_s[5],
                "live_cut":if w.cs_cut.is_some() { json!({"states_cut":w.cs_cut_n[0],"bytes_left_out":w.cs_cut_n[1],"bytes_zero_filled_on_restore":w.cs_cut_n[2],"zero_fill_s":w.cs_cut_s,"planted_left_out":w.cs_plant}) } else { Value::Null },
                "redo_check":"SHA-256 of every 1 MiB piece of the re-made end state against the stored end state's; allocations with declared residue ranges word by word"}) } else { Value::Null },
            "restores_direct_from_page_locked":w.restore_direct,"boundary_buffers_made_ahead":w.pin_ahead_hits,"boundary_buffers_made_at_pass_start":w.pin_q_made,"anchors_ahead":{"plan":w.anchor_ahead_info,"taken":w.anchor_ahead_taken,"wait_s":w.anchor_ahead_wait_s,"live_peak_bytes":ex.live_peak},"staging_allocs_total":crate::staging_allocs(),"pin_page_policy":crate::pinned::page_policy()});
        if ckpts.is_some() {
            receipt["checkpointing"]["sweeps"] = json!(sweeps);
            receipt["checkpointing"]["recompute_steps"] = json!(w.recompute_steps);
            receipt["checkpointing"]["recompute_s"] = json!(w.recompute_s);
            // speed: the boundary hash of every recomputed step is made (no switch leaves it out)
            receipt["checkpointing"]["recompute_hash"] = json!("checked");
            // speed (storm, round 3): how many recomputed boundaries were hashed beside the card (all joined and compared)
            receipt["checkpointing"]["recompute_hash_beside"] = json!({"on":sweep_digest_lazy(crate::recompute_hash_beside(), w.recompute_unchecked),
                "boundaries":ex.recompute_digests_beside,"arenas_max":crate::digest_arenas_max()});
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
            let mut use_by: BTreeMap<String, Vec<Value>> = BTreeMap::new();
            let mut urows: Vec<(&(String, String), &(usize, usize))> = w.arena_use.iter().collect();
            urows.sort_by(|a, b| (b.1 .0.max(b.1 .1)).cmp(&a.1 .0.max(a.1 .1)));
            for ((pass, f), (sc, tp)) in urows { let v = use_by.entry(pass.clone()).or_default(); if v.len() < 16 { v.push(json!([f, sc, tp])); } }
            receipt["arena_use_by_function"] = json!({"fields":"[function, scratch bytes in use when its largest launch ended (arena + spill slabs), tangent tape bytes of its largest launch (tl only)]",
                                                      "by_pass":use_by});
        }
        receipt["one_pass"] = json!({"on":w.one_pass,"fallback_steps":w.one_pass_fallbacks,"tape":w.tape_grown});
        receipt["tape_stream"] = json!({"on":w.st_on,"steps":w.st_info,"buffers_made_on_the_main_thread":w.st_made});
        receipt["tl_tape_grown"] = json!(w.tl_tape_grown);
        receipt["tempo_arena"] = json!(w.tempo_sized);
        receipt["tape_peak"] = json!(w.tape.peak);
        receipt["scratch_peak"] = json!(w.scratch.peak);
        receipt["scheme_scratch_by_key"] = json!(w.native.as_ref().map(|n| n.scratch_by_key()));
        let mut sf = json!(w.scratch_fit);
        if let (Some(o), Some(sp)) = (sf.as_object_mut(), w.scratch.spill.as_ref()) {
            o.insert("spill_slabs".into(), json!(sp.slabs.len())); o.insert("spill_held_bytes".into(), json!(sp.held)); o.insert("spill_peak_bytes".into(), json!(sp.peak));
        }
        receipt["scratch_fit"] = sf;
        receipt["wall_s"] = json!(t_start.elapsed().as_secs_f64());
        let all_pass = receipt["dots"].as_array().is_some_and(|d| d.iter().all(|x| x["pass"] == true))
            && receipt["zero_direction"]["pass"] != false
            && receipt["dots"][0]["reverse"].as_array().is_some_and(|r| r.iter().skip(1).all(|x| x["differing_words_vs_first_reverse"] == 0));
        // A passing dot for identity substitutes is a diagnostic, not a complete-operator gate.
        receipt["milestone_a_numbers_pass"] = json!(all_pass);
        receipt["complete_operator"] = json!(refused.is_empty());
        let all_pass = all_pass && refused.is_empty();
        receipt["complete_operator_pass"] = json!(all_pass);
        // dot, zero direction and repeats pass, and no command ran as identity beyond the --refuse list. (Step
        // boundaries are checked against the reference while the tangent runs and are reported per step.)
        // The dot bar here is the program's (self_dot::DotBar: each dot within 4 float32 roundings of the dot, or the
        // strict 1e-7 with AD_DOT_BAR=relative:1e-7; the door sets AD_DOT_BAR from its table), not this binder's own
        // tolerance (DOT_F32, 2e-5), under which the first 60 km window (3.09e-7, 2.6 roundings: within the bar) and the
        // 240 km windows (1.6e-6, 13 roundings: outside it) read "pass".
        // F14: each dot is judged on its bar_rel (rel_terms for a random y with terms, else rel_aligned as before).
        let dot_bar = self_dot::DotBar::from_env()?;
        if let Some(ds) = receipt["dots"].as_array_mut() { for x in ds.iter_mut() {
            let r = match (x["y_Mx"].as_f64(), x["x_MTy"].as_f64()) { (Some(a), Some(b)) => self_dot::dot_roundings(a, b), _ => None };
            // F14: a random-y dot (the independent and standard forms) cancels by the luck of its draw; its reading is the
            // signed miss over 2^-23 times its absolute terms (bar_rel, verdict.measure rel_terms), the size its float32
            // rounding has. The self form keeps the reading above (its bar_rel is its rel_aligned).
            let r = match (x["verdict"]["measure"].as_str(), x["bar_rel"].as_f64(), r) {
                (Some("rel_terms"), Some(b), Some(r0)) => Some(if r0 < 0.0 { -b } else { b } / self_dot::EPS32), (_, _, r0) => r0 };
            x["roundings"] = json!(r);
        } }
        let dots_at_bar = receipt["dots"].as_array().is_some_and(|d| !d.is_empty() && d.iter().all(|x| x["bar_rel"].as_f64().or_else(|| x["rel_aligned"].as_f64()).is_some_and(|r| dot_bar.passes(r))));
        let valid = valid_rule(receipt["milestone_a_numbers_pass"] == true, dots_at_bar, receipt["refused_beyond_refuse_list"].as_array().is_some_and(|b| b.is_empty()), plant().is_some() || plant_recompute().is_some() || plant_left_out().is_some(),
                               receipt["live_trajectory"]["base_unchecked"] == true);
        receipt["valid_derivative"] = json!(valid);
        // Kernels run as identity under the class "held": named and counted beside the limiter and the refuse list. They
        // do not change valid_derivative (the published recipes carry them); the count says what "valid" leaves out.
        let held_kinds: usize = w.held_kernels.values().map(|s| s.len()).sum();
        receipt["held_identity"] = json!({"what":"kernels that ran as identity in the tangent and the adjoint under the kernel-set class held (wrules.rs held:<scheme>): no derivative of theirs is in this window",
            "kinds":held_kinds,"kernels_by_label":w.held_kernels});
        // prec-floor: output arrays of the boundary-layer row held by the image set's own table (or AD_P3_HOLD_OUT).
        let bl_held = crate::schemes::p3_hold_outputs();
        receipt["boundary_layer_held_outputs"] = json!({"what":"output arrays of the boundary-layer row whose derivative is held in the tangent and the adjoint (tangent zero after the row, cotangent dropped before its reverse): this window carries no derivative through them",
            "outputs":bl_held});
        receipt["valid_derivative_rule"] = json!({"dot_rel_aligned_at_most":dot_bar.rel(),"dot_bar":dot_bar.receipt(),"dot_measure":"each dot's bar_rel: rel_terms for a random-y dot with absolute terms (the independent and standard forms), rel_aligned otherwise (the self form)","dots_at_bar":dots_at_bar,"planted_error":plant().is_some() || plant_recompute().is_some() || plant_left_out().is_some(),"base_unchecked":receipt["live_trajectory"]["base_unchecked"] == true,"recompute_unchecked":w.recompute_unchecked,
            "numbers_pass":receipt["milestone_a_numbers_pass"],"refused_beyond_refuse_list":receipt["refused_beyond_refuse_list"].as_array().map(|b| b.len()),
            "named_approximations":{"fct_frozen":receipt["fct_frozen"],"refuse_list_kinds":refused.len() - receipt["refused_beyond_refuse_list"].as_array().map(|b| b.len()).unwrap_or(0),
                "held_identity_kinds":held_kinds,"boundary_layer_held_outputs":bl_held.len()}});
        { receipt["out_of_core"] = crate::spill::receipt(); fs::write(out.join("window.json"), serde_json::to_vec_pretty(&receipt).unwrap()) }.map_err(|e| e.to_string())?;
        // s60-cardmem: the last reversed step's measurements too
        crate::card_room::note_rev_now(crate::spill::device_now() as u64, crate::spill::cold_families().first().map(|f| f.1).unwrap_or(0) as u64);
        crate::card_room::save();
        log(json!({"event":"window_summary","pass":all_pass,"refused_kinds":refused.len(),"held_identity_kinds":held_kinds,"refused_beyond_refuse_list":receipt["refused_beyond_refuse_list"].as_array().map(|b| b.len()),
                   "valid_derivative":valid,"wall_s":receipt["wall_s"]}));
        w.join_writer()?; mark("writer_joined");
        if let Some((_, h)) = w.ck_pf.take() { let _ = h.join(); }
        if opt(&a, "keep-bstore") != Some("1") { for b in bounds.iter().chain(ck.values()) { let _ = fs::remove_file(&b.path); } } mark("bstore_removed");
        // resident: this request's boundaries go, with their RAM given back to the store's budget as
        // drop_boundary gives it (else each later request would find less room and write its checkpoints to files); the
        // kept boundary 0 stays. A pending boundary read is joined.
        if rz.is_some() {
            w.xfer_wait(&*cuda)?;
            // keep-forward: the window end and the checkpoints the sweep moved aside stay for the next request, out of
            // the store's RAM count (given back here for the window end as resident_forget gives it; for the
            // checkpoints when the sweep moved them). Kept only whole: every forward checkpoint, and the window end.
            if let Some(key) = kf_key.take() {
                let saved = std::mem::take(&mut w.kf_saved);
                if bounds.len() == 2 && saved.len() == w.kf_keep.len() && w.kf_keep.iter().all(|p| saved.contains_key(p)) {
                    let end = bounds.pop().unwrap();
                    if let Some(r) = &end.ram { w.bs_ram_now = w.bs_ram_now.saturating_sub(r.as_slice().len()); }
                    let bytes: usize = saved.values().chain(std::iter::once(&end)).map(|b| b.ram.as_ref().map_or(0, |r| r.as_slice().len())).sum();
                    let from = if kf_hit { receipt["keep_forward"]["forward_from_request"].as_str().unwrap_or("").to_owned() } else { rz.as_ref().map(|z| z.name.clone()).unwrap_or_default() };
                    log(json!({"event":"window_keep_forward","at":"kept","start_sha256":key,"forward_from_request":from,"checkpoints":saved.len(),"kept_bytes":bytes,"store_ram_now":w.bs_ram_now}));
                    kf = Some(KeptForward { key, from, end, ck: saved, flats: std::mem::take(&mut flats), per_step: per_step.clone(), bytes, forward_s: kf_forward_s });
                } else {
                    log(json!({"event":"window_keep_forward","at":"not_kept","why":"the sweep did not pass every forward checkpoint","moved_aside":saved.len(),"forward_checkpoints":w.kf_keep.len()}));
                    w.kf_keep.clear();
                }
            }
            for b in bounds.drain(..).chain(std::mem::take(&mut ck).into_values()) { w.resident_forget(b); }
            if let Some((_, h)) = w.load_next.take() { let _ = h.join(); }
            w.ckpt_base = None; w.y_path = None; w.held_prev = None; w.cs_dev_at = None; w.lw_follow.clear();
            mark("resident_request_end");
        }
        // machine: every output of the window is written and joined above. What is left is unwinding the process: the
        // page-locked buffers are unregistered and freed one by one and every device allocation freed (measured 3.2-4.9 s
        // at 120 km, 3.3 s at 480 km: mark run_returned). The operating system and the driver release all of it when the
        // process ends, so the CLI ends here. AD_WINDOW_FAST_EXIT=0 keeps the unwinding.
        if std::env::var("AD_WINDOW_FAST_EXIT").as_deref() == Ok("0") { let jw = free_beside_join(); if free_beside() { eprintln!("{}", json!({"event":"window_free_beside_joined","wait_s":jw})); } }
        if rz.is_none() && std::env::var("AD_WINDOW_FAST_EXIT").as_deref() != Ok("0") {
            use std::io::Write;
            let _ = std::io::stdout().flush(); let _ = std::io::stderr().flush();
            mark("exit");
            // _exit, not exit: exit() runs the CUDA library's own at-exit teardown of the context (measured 3.2 s at 480 km
            // and 3.8-4.9 s at 120 km after the last output, a test machine gp4-g480 / pair1-b); the kernel releases the context,
            // the device memory and the page-locked host memory when the process is gone.
            extern "C" { fn _exit(code: i32) -> !; }
            _exit(0);
        }
    }
    None };
    // resident: answer this request, then wait for the next (or stop, or end after --resident-idle-s).
    let Some(z) = rz.as_mut() else { break };
    match &refused {
        Some(e) => {
            z.refused(&out, e)?;
            eprintln!("{}", json!({"event":"window_resident_refused_answered","request":z.name,"index":z.index,"wall_s":t_start.elapsed().as_secs_f64()}));
        },
        None => {
    z.done(&out, json!({"window_wall_s":t_start.elapsed().as_secs_f64(),"x0_planted":x0_planted}))?;
    eprintln!("{}", json!({"event":"window_resident_done","request":z.name,"index":z.index,"out":out.display().to_string(),"wall_s":t_start.elapsed().as_secs_f64()}));
        },
    }
    mark("resident_answered");
    let Some(na) = z.next(&words0)? else {
        z.state("ended")?;
        eprintln!("{}", json!({"event":"window_resident_end","served":z.served.len(),"requests":z.served}));
        if std::env::var("AD_WINDOW_FAST_EXIT").as_deref() != Ok("0") {
            use std::io::Write;
            let _ = std::io::stdout().flush(); let _ = std::io::stderr().flush();
            mark("exit");
            extern "C" { fn _exit(code: i32) -> !; }
            unsafe { _exit(0); }
        }
        break;
    };
    t_start = Instant::now();
    mark("resident_request");
    a = na;
    out = PathBuf::from(req(&a, "out")?);
    bdir = PathBuf::from(match opt(&a, "spill-dir") { Some(d) => d, None => req(&a, "bstore")? });
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    fs::create_dir_all(&bdir).map_err(|e| e.to_string())?;
    x0_planted = z.swap(&mut plan, &base, opt(&a, "x0-planted"))?;
    if x0_planted.is_some() { eprintln!("{}", json!({"event":"window_x0_planted","info":x0_planted})); }
    // the step blobs a pass stages are read through this index (Arc copies of the plan's descriptors): made again
    // from the swapped plan, or step 1 would upload the last request's planted bytes.
    w.blob_index = BlockBlobIndex::new(&plan);
    receipt = receipt0.clone();
    receipt["args"] = json!(a); receipt["started_unix"] = json!(now()); receipt["stages"] = json!({"plan_loaded_s":0.0});
    }
    Ok(())
}

/// resident: a later request's start. The plan's current initial payload of each named allocation (the
/// request's planted words, or the recording's for one the kept start had planted) is read and checked as ex.prepare
/// reads it (crate::blob, by sha256, on all cores) and uploaded over the restored boundary 0. An allocation with
/// relocated words is refused at the swap (Resident::swap), so a raw upload is what ex.prepare would give it.
unsafe fn resident_replant(ex: &mut Execution, plan: &Plan, base: &Path, ids: &[u64]) -> Result<Value, String> {
    let t = Instant::now();
    let mut want = Vec::with_capacity(ids.len());
    for id in ids {
        let x = plan.allocations.iter().find(|x| x.id == *id).ok_or_else(|| format!("resident: allocation {id} is not in the plan"))?;
        if !x.relocations.is_empty() { return Err(format!("resident: allocation {id} carries relocated words")); }
        want.push((x.bytes, x.initial.clone().ok_or_else(|| format!("resident: allocation {id} has no initial payload"))?));
    }
    let blobs: Vec<crate::plan::Blob> = want.iter().map(|w| w.1.clone()).collect();
    let dec: HashMap<String, Vec<u8>> = decode_blobs(base, &blobs)?.into_iter().collect();
    let t_read = t.elapsed().as_secs_f64();
    dev::sync(&ex.cuda)?;
    let mut bytes = 0usize;
    for (id, (n, b)) in ids.iter().zip(&want) {
        let data = dec.get(crate::blob_sha(b)).ok_or_else(|| format!("resident: allocation {id}: its payload was not read"))?;
        if data.len() != *n { return Err(format!("resident: allocation {id}: payload of {} bytes for {n}", data.len())); }
        let a = ex.allocations.get(id).ok_or_else(|| format!("resident: allocation {id} is not live at boundary 0"))?;
        if a.bytes != *n { return Err(format!("resident: allocation {id} is {} bytes live, {n} in the plan", a.bytes)); }
        checked("cuMemcpyHtoD resident replant", (ex.cuda.upload)(a.address, data.as_ptr().cast(), data.len()))?;
        bytes += data.len();
    }
    dev::sync(&ex.cuda)?;
    Ok(json!({"allocations":ids.len(),"ids":ids,"bytes":bytes,"read_s":t_read,"wall_s":t.elapsed().as_secs_f64()}))
}

#[cfg(test)]
mod integrated_taylor_guard_tests {
    use super::*;

    fn boundary(index: Vec<(u64, usize, usize)>) -> Boundary {
        Boundary {
            sub: None, path: PathBuf::new(), index, step: 0, delta: false, file_bytes: 0, ram: None, dev: 0, cs: None,
            host: HostSnap {
                host_state: Default::default(), tempo_cache: Default::default(),
                tempo_bindings: HashMap::new(), pointer_history: crate::HistSnap::Ready(HashMap::new()),
                held_pointer_writes: HashMap::new(), events: Vec::new(), frame: None, rk: None,
            },
        }
    }

    #[test]
    fn another_recordings_mesh_indices_are_refused_for_direction_and_objective() {
        // Allocation 351 was theta in the stale inputs but cellsOnCell in pkg18m2.
        let data: Vec<u8> = [0u32, 1, 42, 40962].iter().flat_map(|w| w.to_le_bytes()).collect();
        let b = boundary(vec![(351, 0, data.len())]);
        let terms: Vec<_> = (0..4).map(|w| (351, w, 1.0)).collect();
        for what in ["--taylor-dir", "--objective"] {
            let error = refuse_index_words(what, "stale.json", &b, &data, &terms).unwrap_err();
            assert!(error.contains(&format!("{what} stale.json: 3 of its 3 nonzero words in allocation 351")));
            assert!(error.contains("cellsOnCell"));
            assert!(error.contains("Rebuild or translate the file for this plan."));
        }
    }

    #[test]
    fn float_fields_and_signed_zeros_pass() {
        let data: Vec<u8> = [0.0f32, -0.0, 1.0, 300.0].iter().flat_map(|w| w.to_le_bytes()).collect();
        let b = boundary(vec![(67, 0, data.len())]);
        let terms: Vec<_> = (0..4).map(|w| (67, w, 1.0)).collect();
        for what in ["--taylor-dir", "--objective"] {
            refuse_index_words(what, "tin-m2.json", &b, &data, &terms).unwrap();
        }
    }

    #[test]
    fn valid_float_allocation_cannot_hide_an_integer_allocation() {
        let words = [1u32, 2, 280f32.to_bits(), 290f32.to_bits(), 300f32.to_bits()];
        let data: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let b = boundary(vec![(351, 0, 8), (67, 8, 12)]);
        let terms = [(351, 0, 1.0), (351, 1, 1.0), (67, 0, 1.0), (67, 1, 1.0), (67, 2, 1.0)];
        let error = refuse_index_words("--objective", "mixed.json", &b, &data, &terms).unwrap_err();
        assert!(error.contains("2 of its 2 nonzero words in allocation 351"));
    }

    #[test]
    fn missing_allocation_and_word_past_end_are_refused() {
        let b = boundary(vec![(67, 0, 4)]);
        let data = 300f32.to_le_bytes();
        let error = refuse_index_words("--taylor-dir", "dir.json", &b, &data, &[(351, 0, 1.0)]).unwrap_err();
        assert!(error.contains("allocation 351 is not in boundary 0"));
        let error = refuse_index_words("--objective", "obj.json", &b, &data, &[(67, 1, 1.0)]).unwrap_err();
        assert!(error.contains("word 1 is outside allocation 67 (4 bytes)"));
    }
}

#[cfg(test)]
mod tlsync_blob_tests {
    use super::*;
    fn descriptor(b: &crate::plan::Blob) -> (u8, &str, &str, usize) {
        match b {
            crate::plan::Blob::Hex { bytes, sha256 } => (0, bytes, sha256, 0),
            crate::plan::Blob::File { path, sha256 } => (1, path, sha256, 0),
            crate::plan::Blob::Gzip { path, sha256, bytes } => (2, path, sha256, *bytes),
        }
    }
    fn plan() -> Plan {
        let mut p: Plan = serde_json::from_value(json!({"schema":"test", "complete":true, "source_pin":"test",
            "end_step_barrier_proven":true,"modules":[],"allocations":[],"carried":[],"blocks":{}})).unwrap();
        let upload = |sha: &str| Operation::Upload { stream: 0, destination: crate::plan::Slice { allocation: 1, offset: 0 },
            data: crate::plan::Blob::Hex { bytes: "00".into(), sha256: sha.into() }, pointers: Vec::new(), origin: None };
        p.blocks.insert("root".into(), vec![upload("before"), Operation::HostBranch {
            condition: crate::plan::Slice { allocation: 1, offset: 0 }, if_zero: "zero".into(), if_nonzero: "nonzero".into()
        }, upload("after")]);
        p.blocks.insert("zero".into(), vec![upload("zero"), Operation::RkSplits { body: "shared".into() }]);
        p.blocks.insert("nonzero".into(), vec![upload("nonzero"), Operation::RkStages { body: "shared".into() }]);
        let mut alternate = upload("zero");
        if let Operation::Upload { data, .. } = &mut alternate {
            *data = crate::plan::Blob::File { path: "alternate-path".into(), sha256: "zero".into() };
        }
        p.blocks.insert("shared".into(), vec![upload("shared"), alternate, Operation::RkSubsteps { body: "cycle".into() }]);
        p.blocks.insert("cycle".into(), vec![Operation::RkSplits { body: "root".into() }]);
        p
    }
    #[test]
    fn cached_blobs_match_recursive_discovery_for_live_branch_and_rk_roots() {
        let p = plan();
        let index = BlockBlobIndex::new(&p);
        for roots in [vec!["root"], vec!["zero", "nonzero"], vec!["shared", "root", "shared"], vec!["missing"], vec![]] {
            let names: Vec<String> = roots.iter().map(|n| n.to_string()).collect();
            let (mut seen, mut expected) = (std::collections::HashSet::new(), Vec::new());
            for n in &names { block_blobs(&p, n, &mut seen, &mut expected); }
            let mut hashes = std::collections::HashSet::new();
            let expected: Vec<_> = expected.into_iter().filter(|b| hashes.insert(crate::blob_sha(b)))
                .map(descriptor).collect();
            let got = index.get(&names);
            assert_eq!(got.iter().map(descriptor).collect::<Vec<_>>(), expected, "{names:?}");
        }
        let got = index.get(&["root".into()]);
        assert_eq!(got.iter().map(crate::blob_sha).collect::<Vec<_>>(), vec!["before", "zero", "shared", "nonzero", "after"]);
    }
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

#[cfg(test)]
mod plant_recompute_tests {
    use super::{plant_recompute_parse, plant_recompute_word, PLANT_RC_BIT};
    #[test]
    fn spec_is_a_step_then_an_optional_allocation_and_word() {
        assert_eq!(plant_recompute_parse("7"), Some((7, None, None)));
        assert_eq!(plant_recompute_parse("11:421"), Some((11, Some(421), None)));
        assert_eq!(plant_recompute_parse("11:421:9"), Some((11, Some(421), Some(9))));
        for bad in ["", "0", "x", "7:", "7:a", "7:1:b", "-1"] { assert_eq!(plant_recompute_parse(bad), None, "{bad}"); }
    }
    #[test]
    fn the_changed_word_is_the_first_live_float_and_moves_by_one_bit() {
        let w: Vec<u8> = [0.0f32, f32::NAN, 1e-38, f32::INFINITY, 287.5, 3.0].iter().flat_map(|x| x.to_le_bytes()).collect();
        let (i, b, a) = plant_recompute_word(&w).unwrap();
        assert_eq!((i, b), (4, 287.5f32.to_bits()));
        assert_eq!(b ^ a, PLANT_RC_BIT);
        let (x, y) = (f32::from_bits(b), f32::from_bits(a));
        assert!(y.is_finite() && y != x && ((y - x) / x).abs() < 0.01);
        assert_eq!(plant_recompute_word(&[0u8; 64]), None);
    }
}

#[cfg(test)]
mod valid_rule_tests {
    use super::*;

    #[test]
    fn a_planted_error_is_never_valid_whatever_the_dot_reads() {
        assert!(valid_rule(true, true, true, false, false));
        assert!(!valid_rule(true, true, true, true, false));
        assert!(!valid_rule(false, true, true, false, false));
        assert!(!valid_rule(true, false, true, false, false));
        assert!(!valid_rule(true, true, false, false, false));
        assert!(!valid_rule(true, true, true, false, true));
    }
}

#[cfg(test)]
mod no_tangent_label_tests {
    use super::*;

    #[test]
    fn a_window_with_no_tangent_pass_counts_each_label_once_in_its_reverse_taping() {
        // A tangent pass ran: the reverse taping counts nothing (the first pass did).
        let mut off = None;
        assert!(!rev_count_once(&mut off, 1, 0));
        assert!(off.is_none());
        // Nothing counted is what the defect wrote: refused empty, so complete_operator read true.
        assert!(refused_labels(&BTreeMap::new()).is_empty());
        // No tangent pass: every (step, command) counts once, also when a chunk is taped a second time.
        let mut seen = Some(Default::default());
        let mut labels: BTreeMap<String, u64> = BTreeMap::new();
        let commands = ["entry:acoustic_ru_v841", "REFUSED:rlw_taut: refused for this run (--refuse)", "rule:copy",
                        "REFUSED:cloud_from: refused for this run (--refuse)"];
        for step in [3usize, 2, 1] {
            for _visit in 0..2 {
                for (i, name) in commands.iter().enumerate() {
                    if rev_count_once(&mut seen, step, i) { *labels.entry((*name).to_owned()).or_default() += 1; }
                }
            }
        }
        assert_eq!(seen.as_ref().map(|s| s.len()), Some(12));
        assert_eq!(labels["entry:acoustic_ru_v841"], 3);
        assert_eq!(labels["rule:copy"], 3);
        let refused = refused_labels(&labels);
        assert_eq!(refused.len(), 2);
        assert!(refused.values().all(|n| **n == 3));
        assert!(!refused.is_empty(), "complete_operator must read false when a listed kind ran as identity");
    }
}

#[cfg(test)]
mod taylor_dense_tests {
    use super::*;

    fn words(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    #[test]
    fn a_dense_block_moves_the_rerun_state_and_a_zero_nudge_moves_no_bit() {
        let start = words(&[300.0, -0.0, 1.5, 0.0]);
        let block = words(&[2.0, 3.0, 0.0, -4.0]);
        let mut st = start.clone();
        assert_eq!(add_dense(&mut st, &block, 0.5), 3);
        assert_eq!(st, words(&[301.0, 1.5, 1.5, -2.0]));
        let mut st = start.clone();
        assert_eq!(add_dense(&mut st, &block, 0.0), 0);
        assert_eq!(st, start);
    }

    #[test]
    fn the_dense_part_of_the_prediction_is_the_block_dotted_with_the_gradient() {
        assert_eq!(dot_dense(&words(&[2.0, 3.0, 0.0]), &words(&[0.5, -1.0, 9.0])), -2.0);
    }
}

#[cfg(test)]
mod a429_mask_tests {
    //! hunt-429 (the central line): with --stable-dir 1 the check's direction weights must not depend on where the process
    //! put its device memory. THE BREAKAGE THIS PREVENTS: allocation 429 at 120 km holds model numbers whose 8 bytes read
    //! as 0x00007..., so in some processes one of them lay inside a device range, entered the pointer history and lost its
    //! weight: the self-dot count of block 429 moved (102034906, ..05, ..04) and once the check number and proof words did
    //! too, with a byte-identical derivative (a test machine <local path>).
    use super::*;
    use std::sync::Arc;

    /// a pointer word the model holds, inside a device range of both processes
    const POINTER: u64 = 0x0000_7a00_0000_1000;
    /// the word of allocation 429 in the odd run a429r7-c1-2: 8.58e-29 beside a 4.1e-41 denormal (FOUND2.md section 2)
    const MODEL_PAIR: u64 = 0x0000_7322_10d9_9d32;
    /// (start, end inclusive, recorded) device ranges: the odd process's range that holds MODEL_PAIR, and a normal
    /// process's range (both from FOUND2.md section 2), each with one range around POINTER
    const ODD: [(u64, u64, u64); 2] = [(0x7322_0a00_0000, 0x7322_2e18_0000, 0x7b10_0000_0000), (0x7a00_0000_0000, 0x7a00_1000_0000, 0x7b00_0000_0000)];
    const NORMAL: [(u64, u64, u64); 2] = [(0x7060_f800_0000, 0x707b_f3cf_0200, 0x7c00_0000_0000), (0x7a00_0000_0000, 0x7a00_1000_0000, 0x7b00_0000_0000)];

    /// the boundary reading's own rule: device ranges -> pointer history (main.rs forward_address_view_tracked)
    fn history_for(data: &[u8], ranges: &[(u64, u64, u64)]) -> PointerHistory {
        let low = ranges.iter().map(|r| r.0).min().unwrap();
        let high = ranges.iter().map(|r| r.1).max().unwrap();
        crate::forward_address_view_tracked(data, ranges, low, high, None).1
    }
    #[allow(clippy::too_many_arguments)]
    fn new_span(data: &[u8], als: &[(u64, usize, usize, i64)], prep: &[Arc<(Vec<(usize, usize)>, f32)>], field: bool, xs: f32,
                stable: bool, lo: i64, part: &mut [f32]) {
        Win::weights_span(data, als, prep, field, xs, stable, lo, part);
    }
    /// 1024 bytes of model numbers: 256 floats 1.0, 1.25, ... (no 8-byte word has an address shape), then POINTER at
    /// 8-byte word 5, MODEL_PAIR at word 77 and 1.0 beside a zero (below 2^32: no address shape) at word 100
    fn model_bytes() -> Vec<u8> {
        let mut d = Vec::with_capacity(1024);
        for i in 0..256u32 { d.extend_from_slice(&(1.0f32 + i as f32 * 0.25).to_le_bytes()); }
        for (w, v) in [(5usize, POINTER), (77, MODEL_PAIR), (100, 0x0000_0000_3f80_0000u64)] { d[8 * w..8 * w + 8].copy_from_slice(&v.to_le_bytes()); }
        d
    }

    #[test]
    fn same_bytes_two_device_layouts_same_mask_and_count() {
        let d = model_bytes();
        let (ha, hb) = (history_for(&d, &ODD), history_for(&d, &NORMAL));
        // the setup is the defect: the odd process's history holds the model word, the normal one's does not
        assert!(ha.contains_key(&77) && !hb.contains_key(&77) && ha.contains_key(&5) && hb.contains_key(&5));
        for (inv, fwd) in [(false, false), (false, true), (true, false)] {
            let field = inv || fwd;
            let (pa, pb) = (weights_alloc_prep(&d, 0, d.len(), true, inv, fwd, Some(&ha), None, None, None),
                            weights_alloc_prep(&d, 0, d.len(), true, inv, fwd, Some(&hb), None, None, None));
            let (ra, rb) = (pa.1, pb.1);
            let (mut wa, mut wb) = (vec![0f32; 256], vec![0f32; 256]);
            new_span(&d, &[(429, 0, 1024, 0)], &[Arc::new(pa)], field, 1.0, true, 0, &mut wa);
            new_span(&d, &[(429, 0, 1024, 0)], &[Arc::new(pb)], field, 1.0, true, 0, &mut wb);
            let count = |w: &[f32]| w.iter().filter(|&&v| v != 0.0).count();
            let differ: Vec<usize> = (0..256).filter(|&i| wa[i].to_bits() != wb[i].to_bits()).collect();
            eprintln!("a429 inv {inv} fwd {fwd}: count odd {} normal {}, field weight odd {ra:e} normal {rb:e}, words that differ {differ:?}", count(&wa), count(&wb));
            assert_eq!(count(&wa), count(&wb), "self-dot count, inv {inv} fwd {fwd}");
            assert!(differ.is_empty(), "mask, inv {inv} fwd {fwd}");
            assert_eq!(ra.to_bits(), rb.to_bits(), "field weight, inv {inv} fwd {fwd}");
            // the pointer word has no weight in either; 1.0 beside a zero keeps its weight; an ordinary float keeps its weight
            assert!(wa[10] == 0.0 && wa[11] == 0.0 && wa[200] != 0.0 && wa[201] == 0.0 && wa[30] != 0.0);
        }
        // without --stable-dir nothing is masked by history or shape, as before
        let p = weights_alloc_prep(&d, 0, d.len(), false, false, false, Some(&ha), None, None, None);
        assert!(p.0.is_empty());
    }

    // ---- fold 9 b0da827a window.rs, copied word for word: weights_prep's closure `one` (its captures made parameters,
    // ---- dead_ranges() -> dead) and weights_span (renamed old_span)
    #[allow(clippy::too_many_arguments)]
    fn old_one(data: &[u8], (id, off, bytes): (u64, usize, usize), stable: bool, inv: bool, fwd: bool, bstep: usize,
               history: &HashMap<u64, PointerHistory>, residue: &HashMap<(usize, u64), Vec<(usize, usize)>>,
               relocs: &HashMap<u64, Vec<(usize, usize)>>, dead: &HashMap<u64, Vec<(usize, usize)>>) -> (Vec<(usize, usize)>, f32) {
        let valid = |v: f32| v.is_finite() && v <= WEIGHT_MAX && v > 1e-30;
        let rd = |k: usize| f32::from_le_bytes(data[k..k + 4].try_into().unwrap()).abs();
        let mut rg: Vec<(usize, usize)> = Vec::new();
        if stable {
            if let Some(h) = history.get(&id) { rg.extend(h.keys().map(|&w| (8 * w, 8 * w + 8))); }
            if let Some(r) = residue.get(&(bstep, id)) { rg.extend(r.iter().map(|&(lo, n)| (lo, lo + n))); }
            if let Some(r) = relocs.get(&id) { rg.extend(r.iter().map(|&(lo, n)| (lo, lo + n))); }
            if inv { if let Some(r) = dead.get(&id) { rg.extend(r.iter().map(|&(lo, hi)| (lo, hi.min(bytes)))); } }
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
    }
    fn old_span(data: &[u8], als: &[(u64, usize, usize, i64)], prep: &[std::sync::Arc<(Vec<(usize, usize)>, f32)>],
                    field: bool, xs: f32, lo: i64, part: &mut [f32]) {
        let hi = lo + part.len() as i64;
        let valid = |v: f32| v.is_finite() && v <= WEIGHT_MAX && v > 1e-30;
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
                part[(e - lo) as usize] = if !valid(v) || un { 0.0 } else if field { r * xs } else { v * xs };
            }
        }
    }

    fn shaped(w: u64) -> bool { w >> 32 != 0 && w >> 47 == 0 }

    #[test]
    fn prep_and_span_equal_fold9_code() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let (mut same, mut shaped_cases) = (0, 0);
        for case in 0..400 {
            let addr = case % 2 == 1;   // odd cases hold address-shaped words, even cases none
            let words8 = 8 + (rnd() % 120) as usize; let bytes = 8 * words8 - 4 * (rnd() % 2) as usize;
            let off = 4 * (rnd() % 16) as usize;
            let mut d = vec![0u8; off + bytes + 16];
            for k in (0..d.len()).step_by(4) {
                let v: f32 = match rnd() % 10 { 0 => 0.0, 1 => f32::NAN, 2 => 3e5, 3 => 1e-35, 4 => -((rnd() % 1000) as f32) * 1e-3, _ => (rnd() % 100000) as f32 * 1e-2 + 1e-3 };
                d[k..k + 4].copy_from_slice(&v.to_le_bytes());
            }
            for w in 0..words8 {
                let at = off + 8 * w; if at + 8 > d.len() { break; }
                let cur = u64::from_le_bytes(d[at..at + 8].try_into().unwrap());
                let v = if addr && rnd() % 5 == 0 { 0x0000_7000_0000_0000 | (rnd() & 0x0fff_ffff_ffff) } else if shaped(cur) { cur | (1u64 << 62) } else { cur };
                d[at..at + 8].copy_from_slice(&v.to_le_bytes());
            }
            let any_shaped = (0..bytes / 8).any(|w| shaped(u64::from_le_bytes(d[off + 8 * w..off + 8 * w + 8].try_into().unwrap())));
            let (id, bstep) = (429u64, 3usize);
            let mut hist: PointerHistory = HashMap::new();
            for _ in 0..rnd() % 6 { hist.insert((rnd() as usize) % words8, (0, 0)); }
            let r = |n: u64, rnd: &mut dyn FnMut() -> u64| -> Vec<(usize, usize)> { (0..n).map(|_| (4 * (rnd() as usize % (bytes / 4)), 4 * (1 + rnd() as usize % 9))).collect() };
            let residue = HashMap::from([((bstep, id), r(rnd() % 3, &mut rnd))]);
            let relocs = HashMap::from([(id, r(rnd() % 3, &mut rnd))]);
            let dead: HashMap<u64, Vec<(usize, usize)>> = HashMap::from([(id, r(rnd() % 3, &mut rnd).into_iter().map(|(a, n)| (a, a + n + 64)).collect())]);
            let history = HashMap::from([(id, hist)]);
            for stable in [false, true] { for (inv, fwd) in [(false, false), (false, true), (true, false)] {
                let old = old_one(&d, (id, off, bytes), stable, inv, fwd, bstep, &history, &residue, &relocs, &dead);
                let new = weights_alloc_prep(&d, off, bytes, stable, inv, fwd, history.get(&id), residue.get(&(bstep, id)).map(|r| &r[..]),
                                             relocs.get(&id).map(|r| &r[..]), if stable && inv { dead.get(&id).map(|r| &r[..]) } else { None });
                assert_eq!(old.0, new.0, "unstable ranges, case {case} stable {stable} inv {inv} fwd {fwd}");
                let field = inv || fwd;
                let e0 = 7i64; let n = (bytes / 4) as i64;
                let (mut wo, mut wn) = (vec![0f32; n as usize + 3], vec![0f32; n as usize + 3]);
                // three parts, as the callers split the arena among threads
                let cut = [e0 - 2, e0 + n / 3, e0 + 2 * n / 3, e0 + n + 1];
                let (po, pn) = (Arc::new(old.clone()), Arc::new(new.clone()));
                for c in 0..3 {
                    let (lo, hi) = (cut[c], cut[c + 1]);
                    let (a, z) = ((lo - (e0 - 2)) as usize, (hi - (e0 - 2)) as usize);
                    old_span(&d, &[(id, off, bytes, e0)], &[po.clone()], field, 0.5, lo, &mut wo[a..z]);
                    new_span(&d, &[(id, off, bytes, e0)], &[pn.clone()], field, 0.5, stable, lo, &mut wn[a..z]);
                }
                if !stable || !any_shaped {
                    assert_eq!(old.1.to_bits(), new.1.to_bits(), "field weight, case {case} stable {stable} inv {inv} fwd {fwd}");
                    assert_eq!(wo.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), wn.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                               "weights, case {case} stable {stable} inv {inv} fwd {fwd}");
                    same += 1;
                } else {
                    // the fix may only take weight away (address-shaped words); a word it keeps has fold 9's weight unless
                    // the field weight itself moved
                    shaped_cases += 1;
                    for i in 0..wn.len() { assert!(wn[i] == 0.0 || !field || wo[i] != 0.0, "case {case} word {i}");
                                           if !field { assert!(wn[i] == 0.0 || wn[i].to_bits() == wo[i].to_bits(), "case {case} word {i}"); } }
                }
            } }
        }
        eprintln!("a429 equal-to-fold-9 cases: {same} compared bit for bit, {shaped_cases} with address-shaped words and --stable-dir");
        assert!(same >= 1200 && shaped_cases >= 300);
    }
}
#[cfg(test)]
mod bstore_live_tests {
    //! speed live (--bstore-live 1): which boundaries are cut, what a cut state holds, that a re-made step from it
    //! gives the forward's carried words, and that a carried allocation wrongly left out is refused by the redo check.
    use super::*;
    use crate::cstore::{Builder, Checker, Layout, State, PIECE};
    #[test]
    fn bstore_live_cuts_only_the_intermediate_boundaries() {
        assert!(!bstore_live_cuts(0, 29));
        assert!(bstore_live_cuts(1, 29) && bstore_live_cuts(19, 29) && bstore_live_cuts(28, 29));
        assert!(!bstore_live_cuts(29, 29) && !bstore_live_cuts(30, 29));
        assert!(!bstore_live_cuts(1, 1));
    }
    #[test]
    fn bstore_live_index_drops_the_named_allocations_and_lays_the_rest_out_again() {
        let index = vec![(3u64, 0usize, 10usize), (5, 10, 7), (8, 17, 4), (9, 21, 0), (12, 21, 5)];
        assert_eq!(live_index(&index, &|id| id == 5 || id == 9), vec![(3, 0, 10), (8, 10, 4), (12, 14, 5)]);
        assert_eq!(live_index(&index, &|_| false), index);
        assert!(live_index(&index, &|_| true).is_empty());
    }
    #[test]
    fn bstore_live_plant_left_out_parses_and_picks_the_largest_eligible() {
        assert_eq!(plant_left_out_parse("largest"), Some(None));
        assert_eq!(plant_left_out_parse("441"), Some(Some(441)));
        assert_eq!(plant_left_out_parse("x"), None);
        let index = vec![(3u64, 0usize, 10usize), (5, 10, 40), (8, 50, 40), (9, 90, 99)];
        assert_eq!(plant_left_out_pick(&index, &|id| id != 9), Some(5));
        assert_eq!(plant_left_out_pick(&index, &|id| id == 3), Some(3));
        assert_eq!(plant_left_out_pick(&index, &|_| false), None);
    }
    #[test]
    fn bstore_live_leaves_out_what_the_plan_does_not_carry() {
        let plan: Plan = serde_json::from_value(json!({"schema":"t","complete":true,"source_pin":"t","end_step_barrier_proven":true,"modules":[],
            "allocations":[{"id":1,"bytes":8,"initial":null,"overwritten_before_read":true},{"id":2,"bytes":8,"initial":null,"overwritten_before_read":true}],
            "carried":[1,3],
            "blocks":{"step_01":[{"kind":"allocate","stream":0,"id":3,"bytes":4,"asynchronous":false,"overwritten_before_read":true}],
                      "step_02":[{"kind":"allocate","stream":0,"id":4,"bytes":4,"asynchronous":false,"overwritten_before_read":true}]}})).unwrap();
        let mut cut: Vec<u64> = plan_not_carried(&plan).into_iter().collect();
        cut.sort_unstable();
        assert_eq!(cut, vec![2, 4]);
    }
    // A toy window on the host: allocations as byte vectors, one "step" reading A and B and writing the scratch S first.
    const A: u64 = 1;
    const B: u64 = 2;
    const S: u64 = 3;
    fn f32s(v: &[u8]) -> Vec<f32> { v.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() }
    fn bytes(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
    fn start(n: usize) -> BTreeMap<u64, Vec<u8>> {
        let mut dev: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
        dev.insert(A, bytes(&(0..n).map(|i| 1.0 + (i % 97) as f32 * 0.01).collect::<Vec<_>>()));
        dev.insert(B, bytes(&(0..n).map(|i| 2.0 - (i % 89) as f32 * 0.02).collect::<Vec<_>>()));
        dev.insert(S, vec![0u8; n * 4]);
        dev
    }
    fn step(dev: &mut BTreeMap<u64, Vec<u8>>) {
        let (a, b) = (f32s(&dev[&A]), f32s(&dev[&B]));
        // S is overwritten before it is read (the plan's certificate for a non-carried allocation)
        let s: Vec<f32> = a.iter().zip(&b).map(|(x, y)| 0.5 * x + 0.25 * y).collect();
        dev.insert(S, bytes(&s));
        let s = f32s(&dev[&S]);
        dev.insert(A, bytes(&a.iter().zip(&s).map(|(x, z)| x + 0.125 * z).collect::<Vec<_>>()));
        dev.insert(B, bytes(&b.iter().map(|y| y * 0.875 + 1.0).collect::<Vec<_>>()));
    }
    fn index(dev: &BTreeMap<u64, Vec<u8>>) -> Vec<(u64, usize, usize)> {
        let mut at = 0;
        dev.iter().map(|(id, v)| { let r = (*id, at, v.len()); at += v.len(); r }).collect()
    }
    fn image(dev: &BTreeMap<u64, Vec<u8>>, ix: &[(u64, usize, usize)]) -> Vec<u8> { ix.iter().flat_map(|x| dev[&x.0].clone()).collect() }
    /// a state of boundary `dev` as cs_capture makes it, without the allocations `out` names
    fn store(dev: &BTreeMap<u64, Vec<u8>>, out: &dyn Fn(u64) -> bool) -> State {
        let ix = live_index(&index(dev), out);
        let lay = Layout::new(&ix);
        let data = image(dev, &ix);
        let mut b = Builder::new(lay.clone(), None);
        let mut at = 0;
        for (a, z, n) in lay.windows(PIECE) { b.feed(a, z, &data[at..at + n]); at += n; }
        let mut st = b.finish();
        st.left_out_bytes = dev.values().map(|v| v.len()).sum::<usize>() - lay.total;
        st
    }
    /// restore as window.rs restore does: the held allocations' bytes, zeros for the left-out ones
    fn restore(dev: &mut BTreeMap<u64, Vec<u8>>, st: &State) {
        let mut whole = vec![0u8; st.lay.total];
        st.fill(0, st.lay.pieces(), &mut whole);
        let held: BTreeMap<u64, (usize, usize)> = st.lay.index.iter().map(|x| (x.0, (x.1, x.2))).collect();
        for (id, v) in dev.iter_mut() { match held.get(id) { Some(&(o, n)) => v.copy_from_slice(&whole[o..o + n]), None => v.fill(0) } }
    }
    /// the redo check of window.rs cs_redo: the device against the truth of the stored end state, the cut not compared
    fn redo(dev: &BTreeMap<u64, Vec<u8>>, truth_of: &State, cut: &dyn Fn(u64) -> bool) -> Vec<usize> {
        let t = truth_of.truth(&[]);
        let data = image(dev, &truth_of.lay.index);
        let mut ck = Checker::new(&t, t.lay.index.iter().map(|x| !cut(x.0)).collect());
        let mut at = 0;
        for (a, z, n) in t.lay.windows(PIECE) { ck.feed(a, z, &data[at..at + n]); at += n; }
        ck.bad
    }
    #[test]
    fn bstore_live_a_re_made_step_gives_the_forward_words_and_a_carried_allocation_left_out_is_refused() {
        let n = PIECE / 4 + 300; // a little over one piece a field
        let not_carried = |id: u64| id == S;
        // the forward: boundary 1 is stored without S; boundary 2 (the window end here) whole
        let mut dev = start(n);
        step(&mut dev);
        let b1 = store(&dev, &not_carried);
        assert_eq!((b1.left_out_bytes, b1.lay.index.len()), (n * 4, 2));
        step(&mut dev);
        let forward_end = dev.clone();
        let b2 = store(&dev, &|_| false);
        assert_eq!(b2.left_out_bytes, 0);
        // the reverse: the device holds other words (a later pass), boundary 1 is restored, step 2 is re-made
        for v in dev.values_mut() { for (i, x) in v.iter_mut().enumerate() { *x = (i % 251) as u8; } }
        restore(&mut dev, &b1);
        assert!(dev[&S].iter().all(|&x| x == 0));
        step(&mut dev);
        assert_eq!(dev[&A], forward_end[&A]);
        assert_eq!(dev[&B], forward_end[&B]);
        assert!(redo(&dev, &b2, &not_carried).is_empty(), "the carried words of the re-made step are the forward's");
        // the planted test (AD_PLANT_LEFT_OUT): B is carried and read by the step, and is wrongly left out of boundary 1 too
        let mut fwd = start(n);
        step(&mut fwd);
        let b1_wrong = store(&fwd, &|id| id == S || id == B);
        let mut dev = forward_end.clone();
        restore(&mut dev, &b1_wrong);
        step(&mut dev);
        let bad = redo(&dev, &b2, &not_carried);
        // A's two pieces and B's two pieces differ: refused (cs_redo counts each piece, and the window then stops)
        assert_eq!(bad.len(), 4, "a carried allocation wrongly left out is refused: {bad:?}");
    }
}

#[cfg(test)]
mod step_hold_tests {
    //! step-hold: what a hold set keeps at the 3 km storm box, and that a window reversed from held step-start
    //! states (no re-run between steps) meets the redo check where whole states do, and refuses a planted x1.001 word
    //! and a set that leaves out a word a step reads.
    use super::*;
    use crate::step_hold::{layout, plant_word, shift_ranges, HoldSet};

    const C2: &str = include_str!("../../tests/data/step_hold/hold-C2-0527.json");
    const C1: &str = include_str!("../../tests/data/step_hold/hold-C1-0537.json");

    fn live_of(doc: &str) -> Vec<(u64, usize)> {
        let d: Value = serde_json::from_str(doc).unwrap();
        let mut v: Vec<(u64, usize)> = d["sizes"].as_object().unwrap().iter().map(|(k, x)| (k.parse().unwrap(), x.as_u64().unwrap() as usize)).collect();
        v.sort_unstable();
        v
    }

    #[test]
    fn the_3km_sets_hold_what_nr_probe_measured() {
        for (doc, want) in [(C2, 2_679_377_352usize), (C1, 4_498_619_560usize)] {
            let set = HoldSet::parse(doc.as_bytes()).unwrap();
            let live = live_of(doc);
            assert_eq!(live.iter().map(|x| x.1).sum::<usize>(), 8_385_001_472);
            let spans = set.spans(&live).unwrap();
            let held: usize = spans.iter().map(|x| x.2).sum();
            assert_eq!(held, want);
            // no span touches a constant or a dropped byte, and spans are inside their allocations
            let size: HashMap<u64, usize> = live.iter().copied().collect();
            for &(id, o, n) in &spans {
                assert!(!set.consts.contains(&id) && n > 0 && o + n <= size[&id]);
                for &(lo, hi) in set.drop.get(&id).map(|v| v.as_slice()).unwrap_or(&[]) { assert!(o + n <= lo || o >= hi); }
            }
            // an allocation the set does not know is held whole; one of another size refuses
            let mut more = live.clone(); more.push((900_200, 4096));
            assert_eq!(set.spans(&more).unwrap().iter().map(|x| x.2).sum::<usize>(), want + 4096);
            let mut wrong = live.clone(); wrong[100].1 += 4;
            assert!(set.spans(&wrong).unwrap_err().contains("another plan"));
        }
    }

    #[test]
    fn residue_ranges_follow_the_span() {
        assert_eq!(shift_ranges(&[(0, 8), (100, 50), (300, 4)], 96, 64), vec![(4, 50)]);
        assert_eq!(shift_ranges(&[(90, 20)], 96, 64), vec![(0, 14)]);
        assert!(shift_ranges(&[(0, 8)], 96, 64).is_empty());
        assert_eq!(crate::step_hold::plant_parse("30"), Some((30, 1.001)));
        assert_eq!(crate::step_hold::plant_parse("7:1.5"), Some((7, 1.5)));
        assert_eq!(crate::step_hold::plant_parse("0"), None);
    }

    /// The 3 km layout through the program's own copy lists (entry_copies, as store_hold, restore and the redo
    /// downloads take them), allocations at made-up addresses: every copy lies inside its allocation, at its span's
    /// offset, touches no dropped byte and no other copy, and the host side is one packed run. A copy list that lost a
    /// span's offset (it would put a span over the dropped words before it) fails here.
    #[test]
    fn the_3km_copy_lists_put_every_span_at_its_offset() {
        for (doc, want) in [(C2, 2_679_377_352usize), (C1, 4_498_619_560usize)] {
            let set = HoldSet::parse(doc.as_bytes()).unwrap();
            let live = live_of(doc);
            let mut at: HashMap<u64, (u64, usize)> = HashMap::new();
            let mut a = 1u64 << 40;
            for &(id, n) in &live { at.insert(id, (a, n)); a += (n as u64).div_ceil(2 << 20) * (2 << 20); }
            let spans = set.spans(&live).unwrap();
            let (index, sub) = layout(&spans);
            let cps = entry_copies(&index, Some(sub.as_slice()), &index, |id| at.get(&id).copied(), true).unwrap();
            assert_eq!(cps.len(), spans.len());
            let mut host = 0usize;
            let mut dev: Vec<(u64, u64)> = Vec::new();
            for (&(id, o, n), &(cid, off, cn, addr)) in spans.iter().zip(&cps) {
                let (base, bytes) = at[&id];
                assert_eq!((cid, off, cn), (id, host, n));
                assert_eq!(addr, base + o as u64, "allocation {id}: span at {o} copied at {}", addr - base);
                assert!(addr + n as u64 <= base + bytes as u64);
                for &(lo, hi) in set.drop.get(&id).map(|v| v.as_slice()).unwrap_or(&[]) {
                    assert!(addr + n as u64 <= base + lo as u64 || addr >= base + hi as u64, "allocation {id}: a copy over dropped bytes");
                }
                dev.push((addr, addr + n as u64));
                host += n;
            }
            assert_eq!(host, want);
            dev.sort_unstable();
            assert!(dev.windows(2).all(|w| w[0].1 <= w[1].0), "two copies overlap");
            // the redo compare's subset (redo_index) finds each entry's own address
            let some: Vec<(u64, usize, usize)> = index.iter().step_by(3).copied().collect();
            let sc = entry_copies(&index, Some(sub.as_slice()), &some, |id| at.get(&id).copied(), false).unwrap();
            assert!(sc.iter().all(|c| cps.contains(c)));
            // a whole boundary (word off) is the old arithmetic: every entry at its allocation's start, size checked
            let mut wi = Vec::new(); let mut o = 0usize;
            for &(id, n) in &live { wi.push((id, o, n)); o += n; }
            let wc = entry_copies(&wi, None, &wi, |id| at.get(&id).copied(), true).unwrap();
            assert!(wc.iter().all(|&(id, _, n, addr)| (addr, n) == at[&id]));
            let shrunk = |id: u64| at.get(&id).map(|&(a, n)| (a, if id == live[7].0 { n - 4 } else { n }));
            assert!(entry_copies(&wi, None, &wi, shrunk, true).unwrap_err().contains("changed size"));
            assert!(entry_copies(&wi, None, &wi, shrunk, false).is_ok());
        }
    }

    // An emulated card: one flat byte memory with the allocations at their own addresses and marked gaps between them.
    // Every move between host and card goes along entry_copies' lists, as store_hold, restore and the redo downloads
    // move them; the truth is read apart from them (Card::f). Allocation 1 is constant, 2 and 3 carry the state, 4 is
    // scratch every step writes before it reads; allocation 2's words 32..64 are scratch too.
    const GAP: u8 = 0xA5;
    const BASE: u64 = 0x7f00_0000_0000;
    struct Card { mem: Vec<u8>, at: BTreeMap<u64, (u64, usize)> }
    type Copies = Vec<(u64, usize, usize, u64)>;
    impl Card {
        fn new(sizes: &[(u64, usize)]) -> Self {
            let (mut at, mut a) = (BTreeMap::new(), 4096usize);
            for &(id, n) in sizes { at.insert(id, (BASE + a as u64, n)); a += n.div_ceil(4096) * 4096 + 4096; }
            Card { mem: vec![GAP; a], at }
        }
        fn i(&self, addr: u64) -> usize { (addr - BASE) as usize }
        fn f(&self, id: u64) -> Vec<f32> {
            let (a, n) = self.at[&id]; let i = self.i(a);
            self.mem[i..i + n].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        }
        fn put(&mut self, id: u64, v: &[f32]) {
            let (a, n) = self.at[&id]; assert_eq!(n, 4 * v.len()); let i = self.i(a);
            for (k, x) in v.iter().enumerate() { self.mem[i + 4 * k..i + 4 * k + 4].copy_from_slice(&x.to_le_bytes()); }
        }
        fn live(&self) -> Vec<(u64, usize)> { self.at.iter().map(|(k, v)| (*k, v.1)).collect() }
        fn copies(&self, index: &[(u64, usize, usize)], sub: Option<&[usize]>, sized: bool) -> Copies {
            entry_copies(index, sub, index, |id| self.at.get(&id).copied(), sized).unwrap()
        }
        fn download(&self, cps: &Copies, dst: &mut [u8]) { for &(_, off, n, a) in cps { let i = self.i(a); dst[off..off + n].copy_from_slice(&self.mem[i..i + n]); } }
        fn upload(&mut self, cps: &Copies, src: &[u8]) { for &(_, off, n, a) in cps { let i = self.i(a); self.mem[i..i + n].copy_from_slice(&src[off..off + n]); } }
        fn gaps_clean(&self) -> bool {
            let mut inside = vec![false; self.mem.len()];
            for &(a, n) in self.at.values() { let i = self.i(a); inside[i..i + n].iter_mut().for_each(|x| *x = true); }
            self.mem.iter().zip(&inside).all(|(b, &k)| k || *b == GAP)
        }
    }
    fn step(c: &mut Card) {
        let (a, s1, s2) = (c.f(1), c.f(2), c.f(3));
        let w: Vec<f32> = (0..32).map(|i| s1[i] * 0.5 + a[i] + s1[64 + 2 * i] * 0.125).collect();
        c.put(4, &w);
        let mut n1 = s1.clone();
        for i in 0..32 { n1[32 + i] = w[i] * 1.5; }
        for i in (0..32).chain(64..128) { n1[i] = s1[i] + 0.25 * n1[32 + i % 32] + s2[i % 64]; }
        let n2: Vec<f32> = (0..64).map(|i| s2[i] * 0.9 + w[i % 32] - s1[127 - i] * 0.0625).collect();
        c.put(2, &n1); c.put(3, &n2);
    }
    fn set(drop: &str) -> HoldSet {
        HoldSet::parse(format!(r#"{{"name":"toy","sizes":{{"1":256,"2":512,"3":256,"4":128}},"const":[1],"drop":[{drop}]}}"#).as_bytes()).unwrap()
    }
    /// What a sweep saw, by step from the window end down: (step, words the redo check counted, words of the re-run
    /// end that differ from the unchanged forward's end state outside the set's constants and dropped ranges).
    type Seen = Vec<(usize, u64, u64)>;
    /// The forward stores every intermediate boundary as store_hold does (boundary 0 whole); the reverse sweep, from
    /// the window end down, restores the start state of step s over whatever the card holds, re-runs step s and
    /// compares the held words of the re-run end with boundary s (the redo check), and, apart from the program's paths,
    /// with the unchanged forward's whole end state of step s read straight off the card.
    fn sweep(h: &HoldSet, steps: usize, plant: Option<usize>) -> Seen {
        let mut d = Card::new(&[(1, 256), (2, 512), (3, 256), (4, 128)]);
        d.put(1, &(0..64).map(|i| 0.01 * i as f32 + 1.0).collect::<Vec<_>>());
        d.put(2, &(0..128).map(|i| (i as f32).sin()).collect::<Vec<_>>());
        d.put(3, &(0..64).map(|i| (i as f32 * 0.3).cos()).collect::<Vec<_>>());
        d.put(4, &[0.0; 32]);
        let mut wi = Vec::new(); let mut o = 0usize;
        for (id, n) in d.live() { wi.push((id, o, n)); o += n; }
        let mut b0 = vec![0u8; o];
        d.download(&d.copies(&wi, None, true), &mut b0);
        let mut truth: Vec<BTreeMap<u64, Vec<f32>>> = vec![d.at.keys().map(|&k| (k, d.f(k))).collect()];
        let mut held: Vec<(Vec<(u64, usize, usize)>, Vec<usize>, Vec<u8>)> = vec![(Vec::new(), Vec::new(), Vec::new())];
        for s in 1..=steps {
            step(&mut d);
            truth.push(d.at.keys().map(|&k| (k, d.f(k))).collect());
            let (index, sub) = layout(&h.spans(&d.live()).unwrap());
            let mut data = vec![0u8; index.iter().map(|x| x.2).sum()];
            d.download(&d.copies(&index, Some(sub.as_slice()), true), &mut data);
            if plant == Some(s) { plant_word(&mut data, 1.001).unwrap(); }
            held.push((index, sub, data));
        }
        let mut out = Vec::new();
        for s in (1..=steps).rev() {
            if s == 1 { d.upload(&d.copies(&wi, None, true), &b0); }
            else { let (i, u, x) = &held[s - 1]; let c = d.copies(i, Some(u.as_slice()), true); d.upload(&c, x); }
            step(&mut d);
            let (index, sub, want) = &held[s];
            let mut live_b = vec![0u8; want.len()];
            d.download(&d.copies(index, Some(sub.as_slice()), false), &mut live_b);
            let ranges: Vec<Vec<(usize, usize)>> = index.iter().map(|_| Vec::new()).collect();
            let (diff, _) = compare_boundary(index, &live_b, want, &ranges, s);
            let mut tdiff = 0u64;
            for (id, v) in &truth[s] {
                if h.consts.contains(id) { continue; }
                let dropped = |k: usize| h.drop.get(id).is_some_and(|r| r.iter().any(|&(lo, hi)| 4 * k >= lo && 4 * k < hi));
                tdiff += d.f(*id).iter().zip(v).enumerate().filter(|(k, (x, y))| !dropped(*k) && x.to_bits() != y.to_bits()).count() as u64;
            }
            out.push((s, diff, tdiff));
        }
        assert!(d.gaps_clean(), "a copy wrote outside every allocation");
        out
    }

    #[test]
    fn held_states_reverse_with_no_re_run_and_the_redo_check_sees_a_planted_word() {
        // the right set: words 32..64 of allocation 2 and all of allocation 4 are written before read. Every span of
        // allocation 2 but the first starts inside it (sub > 0), so an address that lost its offset is seen.
        let right = set("[2, 128, 128], [4, 0, 128]");
        let r = sweep(&right, 6, None);
        assert!(r.iter().all(|&(_, d, t)| d == 0 && t == 0), "{r:?}");
        assert_eq!(right.spans(&[(1, 256), (2, 512), (3, 256), (4, 128)]).unwrap().iter().map(|x| x.2).sum::<usize>(), 384 + 256);
        // a planted x1.001 in boundary 3's held state: the sweep meets it first at step 4, which starts from it (its re-run
        // end differs from boundary 4's held words and from the unchanged forward), and again at step 3, which ends
        // there; steps 5 and 6 are clean. The program refuses at the first (reverse_step's redo check).
        let p = sweep(&right, 6, Some(3));
        assert_eq!(p.iter().map(|x| x.0).collect::<Vec<_>>(), vec![6, 5, 4, 3, 2, 1]);
        assert!(p[0].1 == 0 && p[1].1 == 0 && p[2].1 > 0 && p[2].2 > 0 && p[3].1 > 0, "{p:?}");
        // a set that leaves out words a step reads (allocation 3's first 16 words): the redo check refuses
        let bad = set("[2, 128, 128], [4, 0, 128], [3, 0, 64]");
        let b = sweep(&bad, 6, None);
        assert!(b.iter().any(|&(_, d, _)| d > 0), "{b:?}");
    }

    /// words a home-card fill or check moves at a time (64 MB of host memory)
    const CHUNK: usize = 16 << 20;
    /// A finite word in [1, 2) for word k of allocation id (the home-card test's state, made again to check it).
    fn pattern(id: u64, k: usize) -> u32 {
        let mut x = id.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (k as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        x ^= x >> 29; x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9); x ^= x >> 32;
        0x3f80_0000 | ((x as u32) & 0x007f_ffff)
    }

    /// HOME CARD: the 3 km storm box's real allocation sizes on a card (8.4 GB), moved by the program's own functions
    /// (entry_copies, pinned::download_spans as store_hold, upload_copies as restore, download_copies as redo_download).
    /// Store held spans, overwrite the whole card with NaN, restore: every held word is back, every constant and dropped
    /// byte still NaN (nothing written outside a span); the redo download equals the held state; a planted x1.001 word is
    /// counted. Then the store and restore seconds, held (C2) against whole, median of 5.
    /// STEP_HOLD_CARD_OUT=<file> writes the seconds as JSON.
    #[test]
    #[ignore = "card only: about 8.4 GB of card memory and 5.4 GB of page-locked host memory (14 GB with STEP_HOLD_CARD_WHOLE=1)"]
    fn step_hold_on_a_card_at_the_3km_layout() {
        unsafe { card_run().unwrap() }
    }
    unsafe fn card_run() -> Result<(), String> {
        let cuda = Cuda::load()?;
        checked("cuInit", (cuda.init)(0))?;
        let mut device = 0;
        checked("cuDeviceGet", (cuda.device_get)(&mut device, 0))?;
        let mut context = std::ptr::null_mut();
        checked("cuCtxCreate", (cuda.context_create)(&mut context, 0, device))?;
        let set = HoldSet::parse(C2.as_bytes())?;
        let live = live_of(C2);
        let mut at: HashMap<u64, (u64, usize)> = HashMap::new();
        let t = Instant::now();
        for &(id, n) in &live {
            let mut p = 0u64;
            checked("cuMemAlloc", (cuda.allocate)(&mut p, n.max(4)))?;
            for k0 in (0..n / 4).step_by(CHUNK) {
                let v: Vec<u32> = (k0..(k0 + CHUNK).min(n / 4)).map(|k| pattern(id, k)).collect();
                dev::upload(&cuda, p + 4 * k0 as u64, std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), 4 * v.len()))?;
            }
            at.insert(id, (p, n));
        }
        let fill_s = t.elapsed().as_secs_f64();
        let alloc = |id: u64| at.get(&id).copied();
        let spans = set.spans(&live)?;
        let (index, sub) = layout(&spans);
        let total: usize = index.iter().map(|x| x.2).sum();
        let store = entry_copies(&index, Some(sub.as_slice()), &index, alloc, true)?;
        let redo = entry_copies(&index, Some(sub.as_slice()), &index, alloc, false)?;
        let mut held = crate::pinned::PinnedBuf::new(&cuda, total)?;
        let mut back = crate::pinned::PinnedBuf::new(&cuda, total)?;
        crate::pinned::download_spans(&cuda, &mut held, &store)?;
        // the held bytes are the pattern at each span's place
        for (&(id, o, n), &(_, off, _)) in spans.iter().zip(&index) {
            for k in 0..n / 4 {
                let b = &held[off + 4 * k..off + 4 * k + 4];
                if u32::from_le_bytes([b[0], b[1], b[2], b[3]]) != pattern(id, o / 4 + k) { return Err(format!("store: allocation {id} word {} is not the card's", o / 4 + k)); }
            }
        }
        // the card moves on: every word NaN
        let nan = 0x7fc0_dead_u32;
        for &(p, n) in at.values() { checked("cuMemsetD32", (cuda.fill32)(p, nan, n / 4, std::ptr::null_mut()))?; }
        dev::sync(&cuda)?;
        upload_copies(&cuda, &held, &entry_copies(&index, Some(sub.as_slice()), &index, alloc, true)?)?;
        dev::sync(&cuda)?;
        let (mut held_ok, mut left_nan) = (0u64, 0u64);
        for &(id, n) in &live {
            let kept = |k: usize| !set.consts.contains(&id) && !set.drop.get(&id).is_some_and(|r| r.iter().any(|&(lo, hi)| 4 * k >= lo && 4 * k < hi));
            for k0 in (0..n / 4).step_by(CHUNK) {
            let v = dev::download(&cuda, at[&id].0 + 4 * k0 as u64, 4 * ((k0 + CHUNK).min(n / 4) - k0))?;
            for (k, c) in v.chunks_exact(4).enumerate() {
                let k = k0 + k;
                let w = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                if kept(k) { if w != pattern(id, k) { return Err(format!("restore: allocation {id} word {k} is {w:08x}, not the held word")); } held_ok += 1; }
                else { if w != nan { return Err(format!("restore wrote allocation {id} word {k}, which the set leaves out")); } left_nan += 1; }
            }
            }
        }
        download_copies(&cuda, &mut back, &redo)?;
        let ranges: Vec<Vec<(usize, usize)>> = index.iter().map(|_| Vec::new()).collect();
        let same = compare_boundary(&index, &back, &held, &ranges, 0);
        if same != (0, 0) { return Err(format!("redo download of the restored state differs from the held state: {same:?}")); }
        // plant in a span that starts inside its allocation (sub > 0), so the word goes to an offset address
        let pi = sub.iter().position(|&u| u > 0).ok_or("no span starts inside its allocation")?;
        let (poff, pn) = (index[pi].1, index[pi].2);
        let (w, b0, b1) = plant_word(&mut held[poff..poff + pn], 1.001).ok_or("no word to plant")?;
        let w = poff / 4 + w;
        upload_copies(&cuda, &held, &store)?;
        dev::sync(&cuda)?;
        download_copies(&cuda, &mut back, &redo)?;
        let planted = compare_boundary(&index, &back, &held, &ranges, 0);
        // the card now holds the planted word: compare it with the unplanted state instead
        let mut clean = held.to_vec(); clean[4 * w..4 * w + 4].copy_from_slice(&b0.to_le_bytes());
        let seen = compare_boundary(&index, &back, &clean, &ranges, 0);
        if planted != (0, 0) || seen.0 != 1 { return Err(format!("planted word {w} ({b0:08x} -> {b1:08x}): {planted:?} {seen:?}")); }
        drop(clean);
        // and the planted word sits on the card at its allocation's own byte (span offset + place in the span)
        let pbyte = sub[pi] + 4 * w - poff;
        let pw = dev::download(&cuda, at[&index[pi].0].0 + pbyte as u64, 4)?;
        if u32::from_le_bytes([pw[0], pw[1], pw[2], pw[3]]) != b1 { return Err(format!("the planted word is not at byte {pbyte} of allocation {}", index[pi].0)); }
        // seconds: held (C2), and with STEP_HOLD_CARD_WHOLE=1 whole (8.4 GB more page-locked host memory), median of 5
        let mut wi = Vec::new(); let mut o = 0usize;
        for &(id, n) in &live { wi.push((id, o, n)); o += n; }
        let whole = entry_copies(&wi, None, &wi, alloc, true)?;
        let mut wbuf = if std::env::var("STEP_HOLD_CARD_WHOLE").as_deref() == Ok("1") { Some(crate::pinned::PinnedBuf::new(&cuda, o)?) } else { None };
        let med = |mut v: Vec<f64>| -> Value { if v.is_empty() { return Value::Null; } v.sort_by(|a, b| a.partial_cmp(b).unwrap()); json!(v[v.len() / 2]) };
        let (mut sh, mut rh, mut sw, mut rw) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..5 {
            let t = Instant::now(); crate::pinned::download_spans(&cuda, &mut held, &store)?; sh.push(t.elapsed().as_secs_f64());
            let t = Instant::now(); upload_copies(&cuda, &held, &store)?; dev::sync(&cuda)?; rh.push(t.elapsed().as_secs_f64());
            if let Some(wb) = wbuf.as_mut() {
                let t = Instant::now(); crate::pinned::download_spans(&cuda, wb, &whole)?; sw.push(t.elapsed().as_secs_f64());
                let t = Instant::now(); upload_copies(&cuda, wb, &whole)?; dev::sync(&cuda)?; rw.push(t.elapsed().as_secs_f64());
            }
        }
        let r = json!({"test":"step_hold_on_a_card_at_the_3km_layout","card_ordinal":device,"planted_allocation":index[pi].0,"planted_span_sub":sub[pi],"set":set.name,"set_sha256":set.sha256,
            "allocations":live.len(),"state_bytes":o,"held_bytes":total,"spans":spans.len(),"held_words_restored":held_ok,"left_out_words_untouched":left_nan,
            "redo_equal":true,"planted_word":w,"planted_words_seen":seen.0,"fill_s":fill_s,
            "store_held_s":med(sh.clone()),"restore_held_s":med(rh.clone()),"store_whole_s":med(sw.clone()),"restore_whole_s":med(rw.clone()),
            "all":{"store_held":sh,"restore_held":rh,"store_whole":sw,"restore_whole":rw},"note":"program functions entry_copies, pinned::download_spans, upload_copies, download_copies; median of 5"});
        println!("{r}");
        if let Ok(f) = std::env::var("STEP_HOLD_CARD_OUT") { fs::write(&f, serde_json::to_string_pretty(&r).unwrap()).map_err(|e| e.to_string())?; }
        drop(wbuf); drop(held); drop(back);
        for &(p, _) in at.values() { let _ = (cuda.free)(p); }
        let _ = (cuda.context_destroy)(context);
        Ok(())
    }
}

#[cfg(test)]
mod param_tl_seed_dot_tests {
    use super::*;

    /// a12 truephys-term. A two-word state and one carried parameter, M = [A | B] on (x, d ln p), y the weights at the
    /// window end. A --param-tl first seed starts from (0, c): <y, M (0, c)> = y.B c p0. The reverse leaves M^T y; its
    /// state half dotted with the seed's random x is NOT that number (the old pairing, about 1 against it in the
    /// 2026-11-13 B1 runs); its parameter half summed by param_x_mty is.
    #[test]
    fn param_tl_first_seed_is_scored_on_the_parameter_side() {
        let a = [[2.0, -1.0], [0.5, 3.0]];             // d out / d x
        let bcol = [0.7, -0.2];                          // d out / d p (per unit p)
        let p0 = 24.0f32; let c = [0.01f64];             // d ln p = c, so dp = c p0
        let y = [1.5, -0.25];
        let x_rand = [0.3, -0.9];                        // the seed's random state direction, unused by this tangent
        let y_mx: f64 = (0..2).map(|i| y[i] * bcol[i] * c[0] * p0 as f64).sum();
        // M^T y: state half and parameter half; the carrier keeps the parameter half per column (two columns here)
        let mty_state = [a[0][0] * y[0] + a[1][0] * y[1], a[0][1] * y[0] + a[1][1] * y[1]];
        let mty_p = bcol[0] * y[0] + bcol[1] * y[1];
        let mut bar = vec![0f64; 8 * 2]; bar[0] = 0.25 * mty_p; bar[1] = 0.75 * mty_p;   // slot 0, columns 0 and 1
        let bars: HashMap<String, Vec<f64>> = [("mynnedmf".to_string(), bar)].into_iter().collect();
        let params = vec![Par { name: "mynn_qke_dissipation_b1".into(), fam: "mynnedmf".into(), slot: 0, p0 }];
        let old = mty_state[0] * x_rand[0] + mty_state[1] * x_rand[1];
        let side = param_x_mty(&params, &bars, &c);
        let new = seed_x_mty(Some(&c), 0, old, Some(side)).unwrap();
        let rel = |a: f64, b: f64| (a - b).abs() / a.abs().max(b.abs());
        assert!(rel(y_mx, new) < 1e-12, "parameter side {new} against <y, M x> {y_mx}");
        assert!(rel(y_mx, old) > 0.5, "the old pairing should miss grossly: {old} against {y_mx}");
        // every other pass keeps its own random direction, and a --param-tl pass with no cotangents is refused
        assert_eq!(seed_x_mty(Some(&c), 1, old, Some(side)).unwrap(), old);
        assert_eq!(seed_x_mty(None, 0, old, None).unwrap(), old);
        assert!(seed_x_mty(Some(&c), 0, old, None).is_err());
    }
}

#[cfg(test)]
mod species_plane_tests {
    //! AD_DOT_PLANES gives each plane of a stacked field its own weight. THE BREAKAGE THIS
    //! PREVENTS: one rms and one 2e5 cap for 13 species, so the masses had about 1e-6 of the weight and nwfa none.
    use super::*;
    use std::sync::Arc;

    fn words(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    #[test]
    fn the_spec_reads_and_bad_specs_are_refused() {
        let p = parse_plane_spec("53@0,412@32004096/13x682276/qv,qc,qr,qi,qs,qg,ni,nr,ng,volg,nc,nifa,nwfa").unwrap();
        assert_eq!((p.x, p.y, p.n, p.pb), ((53, 0), (412, 32004096), 13, 682276));
        assert_eq!(p.names[2], "qr");
        assert_eq!(p.ranges(true)[1], (32004096 + 682276, 32004096 + 2 * 682276));
        assert_eq!(parse_plane_spec("1@0,2@8/2x8").unwrap().names, vec!["plane0", "plane1"]);
        for bad in ["53@0/13x682276", "53@0,412@3/13x682276", "53@0,412@0/0x8", "53@0,412@0/2x6", "53@0,412@0/2x8/a", "53,412/2x8", "53@0,412@0/2x8/a,b/c"] {
            assert!(parse_plane_spec(bad).is_err(), "{bad}");
        }
    }

    /// a 3-plane allocation of 4 words a plane after 2 words of something else: plane 0 small masses, plane 1 numbers,
    /// plane 2 all over the cap; then 2 words of rest
    fn alloc() -> Vec<u8> { words(&[300.0, 290.0, 1e-5, 2e-5, 0.0, 1e-5, 1e3, 3e3, 1e3, 3e3, 1e9, 2e9, 1e9, 2e9, 280.0, 0.0]) }

    #[test]
    fn without_planes_the_weights_are_as_before() {
        let d = alloc();
        for (inv, fwd) in [(false, false), (false, true), (true, false)] {
            let pr = weights_alloc_prep(&d, 0, d.len(), true, inv, fwd, None, None, None, None);
            let (mut a, mut b) = (vec![0f32; 16], vec![0f32; 16]);
            Win::weights_span(&d, &[(53, 0, d.len(), 0)], &[Arc::new(pr.clone())], inv || fwd, 1.0, true, 0, &mut a);
            Win::weights_span_pw(&d, &[(53, 0, d.len(), 0)], &[Arc::new(pr)], &[None], inv || fwd, 1.0, true, 0, &mut b);
            assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
        }
    }

    #[test]
    fn each_plane_gets_its_own_rms_and_no_cap() {
        let d = alloc();
        let planes = [(8, 24), (24, 40), (40, 56)];
        let rms = |v: &[f64]| (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt();
        // fieldxy x side: rms per plane; the zero word of plane 0 is not valid
        let pr = weights_alloc_prep(&d, 0, d.len(), true, false, true, None, None, None, None);
        let t = plane_table(&d, 0, d.len(), true, false, true, &pr.0, &planes, false);
        let want = [rms(&[1e-5f32 as f64, 2e-5f32 as f64, 1e-5f32 as f64]), rms(&[1e3, 3e3, 1e3, 3e3]), rms(&[1e9, 2e9, 1e9, 2e9])];
        for (q, w) in want.iter().enumerate() { assert_eq!(t.0[q].2, *w as f32, "plane {q}"); assert_eq!((t.0[q].0, t.0[q].1), planes[q]); }
        assert_eq!(t.2, rms(&[300.0, 290.0, 280.0]) as f32);
        let mut x = vec![0f32; 16];
        Win::weights_span_pw(&d, &[(53, 0, d.len(), 0)], &[Arc::new(pr.clone())], &[Some(Arc::new(t))], true, 2.0, true, 0, &mut x);
        assert_eq!(x[2], 2.0 * want[0] as f32); assert_eq!(x[4], 0.0); assert_eq!(x[6], 2.0 * want[1] as f32);
        assert_eq!(x[10], 2.0 * want[2] as f32, "a plane word over 2e5 is weighted");
        assert_eq!(x[0], 2.0 * rms(&[300.0, 290.0, 280.0]) as f32); assert_eq!(x[15], 0.0);
        // relative x (field direction, x side): |primal| per word, the cap lifted in the planes only
        let pr = weights_alloc_prep(&d, 0, d.len(), true, false, false, None, None, None, None);
        let t = plane_table(&d, 0, d.len(), true, false, false, &pr.0, &planes, false);
        let mut x = vec![0f32; 16];
        Win::weights_span_pw(&d, &[(53, 0, d.len(), 0)], &[Arc::new(pr.clone())], &[Some(Arc::new(t))], false, 1.0, true, 0, &mut x);
        assert_eq!(x[3], 2e-5); assert_eq!(x[11], 2e9);
        // AD_DOT_PLANES_CAP=1 (cap kept): plane 2 has no weight, as without planes
        let t = plane_table(&d, 0, d.len(), true, false, false, &pr.0, &planes, true);
        let mut x = vec![0f32; 16];
        Win::weights_span_pw(&d, &[(53, 0, d.len(), 0)], &[Arc::new(pr)], &[Some(Arc::new(t))], false, 1.0, true, 0, &mut x);
        assert_eq!(x[11], 0.0); assert_eq!(x[3], 2e-5);
        // field y side: the inverse rms per plane
        let pr = weights_alloc_prep(&d, 0, d.len(), true, true, false, None, None, None, None);
        let t = plane_table(&d, 0, d.len(), true, true, false, &pr.0, &planes, false);
        assert_eq!(t.0[1].2, (1.0 / want[1]) as f32);
    }
}

#[cfg(test)]
mod field_units_tests {
    //! The self dot's default units and AD_DOT_PLANES=scalars read the package's field map. THE BREAKAGE THIS PREVENTS:
    //! a window end the map does not hold scored on another boundary's layout, and the species planes named in the
    //! map's list order (not the memory order for planes 6 to 12).
    use super::*;

    fn package(dir: &Path, last: usize, order: &[&str]) -> PathBuf {
        let mut b = serde_json::Map::new();
        for s in 1..=last { b.insert(format!("{s}"), json!({"scalars":{"allocation":if s == 1 { 53 } else { 400 + s as u64 },"offset":4 * s as u64}})); }
        let m = json!({"species_order":order,
                       "fields":{"scalars":{"shape":[13,59,2891],"strides":[682276,11564,4],"allocation":53,"offset":0,"bytes":8869888}},
                       "boundaries":b});
        fs::create_dir_all(dir).unwrap();
        fs::write(dir.join("alloc-map.json"), serde_json::to_vec(&m).unwrap()).unwrap();
        fs::write(dir.join("plan.json"), b"{}").unwrap();
        dir.join("plan.json")
    }

    #[test]
    fn the_window_end_is_boundary_steps_plus_one_or_a_named_miss() {
        let d = std::env::temp_dir().join(format!("hexad-field-units-{}", std::process::id()));
        // the list order a recorded package carries: the same 13 names, not the memory order
        let plan = package(&d, 15, &["qv", "qc", "qr", "qi", "qs", "qg", "nc", "ni", "nr", "ng", "nifa", "nwfa", "volg"]);
        let (m, v) = package_field_map(&plan).unwrap();
        assert_eq!(map_window_end(&m, &v, 14).unwrap(), "15");
        let e = map_window_end(&m, &v, 15).unwrap_err();
        assert!(e.contains("no boundary 16") && e.contains("15 boundaries"), "{e}");
        // the planes: x at boundary 1, y at the window end, names in the memory order
        let full = planes_from_map(&plan, 14).unwrap();
        assert_eq!(full, "53@4,415@60/13x682276/qv,qc,qr,qi,qs,qg,ni,nr,ng,volg,nc,nifa,nwfa");
        let ps = parse_plane_spec(&full).unwrap();
        assert_eq!((ps.x, ps.y, ps.n, ps.pb), ((53, 4), (415, 60), 13, 682276));
        assert_eq!((ps.names[6].as_str(), ps.names[9].as_str(), ps.names[12].as_str()), ("ni", "volg", "nwfa"));
        let e = planes_from_map(&plan, 15).unwrap_err();
        assert!(e.starts_with("AD_DOT_PLANES=scalars: ") && e.contains("no boundary 16") && e.contains("in full"), "{e}");
        // another species set: the planes are numbered, never named by guess
        let other = package(&d.join("other"), 3, &["qv", "qc", "qr"]);
        assert_eq!(planes_from_map(&other, 2).unwrap(), "53@4,403@12/13x682276");
        // no map beside the plan
        fs::create_dir_all(d.join("bare")).unwrap();
        fs::write(d.join("bare/plan.json"), b"{}").unwrap();
        assert!(package_field_map(&d.join("bare/plan.json")).unwrap_err().contains("no alloc-map.json"));
        let _ = fs::remove_dir_all(&d);
    }
}
#[cfg(test)]
mod dot_verdict_tests {
    //! F14: a random-y dot is judged on its miss over its absolute terms. THE BREAKAGE THIS PREVENTS: the 12 km
    //! independent form failing on the one seed whose draw cancels (seed 1, 1.1 per cent of its terms) on the
    //! arithmetic its other two seeds pass with; and a real error hiding because its dot happens not to cancel.
    use super::*;

    #[test]
    fn a_random_y_is_read_on_its_terms_and_the_self_form_is_unchanged() {
        // the three 12 km seeds of box E relck12/B ind15-new: (y_Mx, x_MTy, abs_tl, abs_ad)
        let seeds: [(f64, f64, f64, f64); 3] = [(-5072.190836715281, -5072.047616286671, 471636.72266557586, 238389.20771406792),
                     (81833.04785996996, 81833.37667254054, 548380.0685021635, 309319.45640768437),
                     (-33641.6149028656, -33641.378708717966, 454943.3755835526, 260749.60983082687)];
        let want: [f64; 3] = [3.036668302688001e-07, 5.996070781451427e-07, 5.191726274304611e-07];
        for (&(t, a, at, aa), w) in seeds.iter().zip(want) {
            let rp = (t - a).abs() / t.abs().max(a.abs());
            let rt = (t - a).abs() / at.max(aa);
            assert!((rt - w).abs() < 1e-18, "{rt} {w}");
            let (p, v, m) = dot_verdict(false, rp, rp, Some(rt));
            assert_eq!((p, v, m), (rt, rt, "rel_terms"));
            assert!(p <= DOT_F32, "every seed passes 2e-5 on its terms");
            assert!(v > 1e-7, "and none is at the program's 1e-7: the 12 km float32 scatter still shows");
        }
        // seed 1 before: rel 2.8e-05 over DOT_F32
        assert!((-5072.190836715281f64 - -5072.047616286671f64).abs() / 5072.190836715281 > DOT_F32);
        // the self form is read as before: pass on min, valid on rel_aligned (= rel)
        assert_eq!(dot_verdict(true, 7.0e-8, 7.0e-8, Some(1e-9)), (7.0e-8, 7.0e-8, "rel (self form: its terms are its y_Mx)"));
        // no terms: the old reading
        assert_eq!(dot_verdict(false, 3e-6, 4e-6, None), (3e-6, 4e-6, "rel_aligned (no absolute terms)"));
        assert_eq!(dot_verdict(false, 3e-6, 4e-6, Some(f64::NAN)).2, "rel_aligned (no absolute terms)");
        // a real error is still caught: a miss of 1e-5 of the terms fails valid whatever the cancellation
        let (_, v, _) = dot_verdict(false, 1e-5 / 0.9, 1e-5 / 0.9, Some(1e-5));
        assert!(v > 1e-7);
    }
}
