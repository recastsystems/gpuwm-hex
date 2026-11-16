// SPDX-License-Identifier: Apache-2.0
//! Physics-on window, lane L1: derivative rule of every command of the physics plan, for the composed
//! tangent and the segment-checkpointed reverse (window.rs). Kernel launches with an ad.entry/v1 row run
//! their entries (deriv.rs); everything else gets one of the rules below, applied on one float shadow arena
//! (P8 / U7 convention: one shadow element per forward 4-byte word, an 8-byte element's shadow at its first
//! word). The copy-row, view, where, max and zero rules are P8's (tree/ad/phys/p8: rules.rs assignment,
//! rows.rs row form and parallel gather transpose, cuda/p8.cu kernels). Nothing is passed silently: every
//! command that has no exact rule is `Refused` (identity in both directions, so the composed operator stays
//! the exact transpose of the tangent) and counted by name.
use crate::{plan::{self, Operation}, submit, Execution};
use std::collections::HashMap;

/// Element type `T` of every module that launches cupy_concatenate_same_size, from the module's recorded CuPy source
/// (`typedef <T> T;` in recording/sources/<sha>.cu, named by any argument of any launch of that module; the runner has
/// already checked each named source against its sha256, crate::verify_copy_indexer_sources /
/// verify_fixed_argument_sources). The kernel name carries no dtype, so this is the only place the type comes from.
/// THE BREAKAGE THIS PREVENTS (step6-sw, review finding ad/phys-p8-check rules.rs:25): the window bound the launch by
/// NAME as inactive (schemes18.json, built on pkg18 whose only instantiation is `typedef long long T`). pkg18m2 also has
/// module 5396101c, `typedef float T`: the SW radiation input stack `f = cp.stack(p3d, t3d, dz8w, pi3d, qv3d, qc3d, qi3d,
/// qs3d, cldfra3d, re_cloud, re_ice, re_snow)` (rrtmg_sw_v842.py:581) that mpas_sw_prep and mpas_sw_outputs read. As an
/// identity the SW tangent never saw the state, and mpas_sw_prep read whatever tangent the reused pool words held.
pub fn concat_module_types(plan: &plan::Plan, base: &std::path::Path) -> Result<HashMap<String, String>, String> {
    let mut out = HashMap::new();
    let mut want: Vec<&str> = Vec::new();
    for ops in plan.blocks.values() { for op in ops { if let Operation::Kernel { module, function, .. } = op {
        if function == "cupy_concatenate_same_size" && !want.contains(&module.as_str()) { want.push(module); }
    } } }
    for m in want {
        let mut src: Option<String> = None;
        'find: for ops in plan.blocks.values() { for op in ops { if let Operation::Kernel { module, arguments, .. } = op {
            if module != m { continue; }
            for a in arguments {
                let s = match a {
                    plan::Argument::CupyCopyIndexer { source, .. } => Some(&source.source),
                    plan::Argument::Bytes { fixed_source: Some(f), .. } | plan::Argument::Patched { fixed_source: Some(f), .. } => Some(&f.source),
                    _ => None,
                };
                if let Some(s) = s.filter(|s| s.starts_with("recording/sources/")) { src = Some(s.clone()); break 'find; }
            }
        } } }
        let Some(src) = src else { continue };
        let text = std::fs::read_to_string(base.join(&src)).map_err(|e| format!("concatenate module {m}: {src}: {e}"))?;
        if let Some(t) = text.lines().map(str::trim).find_map(|l| l.strip_prefix("typedef ").and_then(|r| r.strip_suffix(" T;"))) {
            out.insert(m.to_string(), t.to_string());
        }
    }
    Ok(out)
}

/// P8 DevRow (cuda/p8.cu struct P8Row).
#[repr(C)] #[derive(Clone, Copy, Debug)]
pub struct DevRow { pub dst: i64, pub src: i64, pub ds: [i64; 4], pub ss: [i64; 4], pub n: [i64; 3], pub count: i64 }
/// P8 DevView (cuda/p8.cu struct P8View).
#[repr(C)] #[derive(Clone, Copy, Debug, PartialEq)]
pub struct DevView { pub base: i64, pub size: i64, pub ndim: i32, pub pad: i32, pub shape: [i64; 4], pub stride: [i64; 4] }

#[derive(Clone, Debug)]
pub enum R {
    /// Kernel with an ad.entry/v1 derivative.
    Entry,
    /// Copy rows (post-transport copy, glue copy plan, cuBLAS transpose, broadcasting view copy): P8 rows.
    Rows { rows: Vec<DevRow>, label: &'static str },
    /// Elementwise float copy (casts exact on the shadow).
    View { dst: DevView, src: DevView },
    /// Overwrite with state-independent data on a strided view.
    ZeroView { dst: DevView, label: String },
    /// out = max(c, x): x's primal span (allocation, first word, words) is taped before the launch.
    Max { out: DevView, x: DevView, prim: (u64, usize, usize), xv: DevView, c: f32 },
    /// dst = mask ? src : dst; the mask's byte span (allocation, first byte, bytes) is taped before the launch.
    Where { dst: DevView, src: DevView, prim: (u64, usize, usize), mv: DevView },
    /// out = c * in (contiguous): element indices.
    Scale { input: i64, output: i64, n: usize, c: f64 },
    /// dst <- src over n words.
    Copy { dst: i64, src: i64, n: usize },
    /// multidevice-runner: the source half of a cross-rank copy (n words at shadow element src). Tangent: send the words;
    /// reverse: src += the receiving rank's cotangent.
    PeerSend { src: i64, n: usize, key: String },
    /// multidevice-runner: the destination half. Tangent: dst <- the sending rank's words; reverse: send dst's cotangent
    /// and zero it.
    PeerRecv { dst: i64, n: usize, key: String },
    /// dst overwritten by a constant (clear, fill, immutable upload, fresh allocation, host-frame constant).
    Zero { ranges: Vec<(i64, usize)>, label: String },
    /// Held upload (a held scheme's step result, a host value): zero tangent, cotangent dropped; per origin.
    Held { ranges: Vec<(i64, usize)>, origin: String },
    /// No floating state written (or a held physics launch): identity both ways; counted by label.
    Identity(String),
    /// No exact rule: identity both ways, listed by name.
    Refused(String),
    /// Frozen FCT limiter (--fct-frozen 1): fct_minmax_source with the limiter bounds held constant. `e` are the shadow
    /// element indices of old, source, rho_old, source_old, minimum, maximum; `prim` the primal spans of source and
    /// rho_old (allocation, first word, words), taped before the launch.
    FctMinmax { nt: i32, nl: i32, nc: i32, dt: f32, e: [i64; 6], prim: [(u64, usize, usize); 2] },
    /// L2 physics row (P3 MYNN-EDMF, P4 surface layer / GSL drag) bound through the float32-shadow adapter
    /// (--native-schemes 1, window.rs): its float64 entries run on widened copies of the window shadows.
    Native(String),
    /// TEMPO's whole-launch derivative (tempo_warp_steps, lane L5 tempo_adw: warp shape, per-substep checkpoints,
    /// executed-branch tape) bound in the window (--tempo-live 1, window.rs `tempo_apply`): the launch's inputs are
    /// snapshot before it runs, the tangent / adjoint run on float64 copies of the window's float32 shadows.
    Tempo(String),
    /// TEMPO per-column launch (tempo_precip) bound to P5's per-thread entry (--tempo-precip 1, window.rs):
    /// its float64 tape runs on widened copies of the window shadows, outputs rounded back.
    TempoSerial(String),
    /// --tempo-small (window.rs, ad/held-tempo_radar): a TEMPO per-column launch bound to the P5 per-thread entry
    /// (tempo_adw::TempoSmall, float64 tape) through the float32-shadow adapter. Without the flag the launch keeps its
    /// rule below (tempo_radar: inactive, an output diagnostic with no prognostic feedback).
    TempoSmall(String),
}

pub struct Ctx<'a> {
    pub base_el: &'a HashMap<u64, i64>,
    pub kernels: &'a HashMap<String, crate::deriv::KernelDeriv>,
    pub classes: &'a HashMap<String, String>,
    pub schemes: &'a HashMap<String, String>,
    pub refuse: &'a HashMap<String, String>,
    /// --fct-frozen 1: the FCT limiter coefficients are frozen from the trajectory (see `fct_frozen`).
    pub fct_frozen: bool,
}

/// Kernels whose output is a constant of the module image: (function, bytes written through argument 0). Each entry is
/// checked against its source and SASS before it is listed.
/// * tempo_warp_sizes (hexcore/mpasphys/rrfs/tempo_v303.py:4024): `out[0]=sizeof(TempoWork); out[1]=sizeof(TempoState);
///   out[2]=sizeof(TempoBlob)`, three uint64. SASS of pkg18 module 477158923b00908c: three MOV.64 immediates
///   (0xfe30, 0x1078, 0x3a647620) and three STG.E.64 through the one pointer parameter, no load. The host reads the
///   words back (checked_value, tempo_v303.py:4312) to size TEMPO's work areas; the pool slot is then rebound as a
///   TEMPO pointer table. Grid 1x1x1, block 1x1x1, step 1 only.
pub fn image_constant(f: &str) -> Option<usize> {
    match f { "tempo_warp_sizes" => Some(24), _ => None }
}

/// A CuPy CArray by value: target and the view (byte strides).
pub struct CArr { pub alloc: u64, pub off: usize, pub size: i64, pub shape: Vec<i64>, pub strides: Vec<i64> }
pub fn carr(a: &submit::Argument) -> Option<CArr> {
    let submit::Argument::Patched { bytes, pointers } = a else { return None };
    if pointers.len() != 1 || pointers[0].byte_offset != 0 || bytes.len() < 16 || (bytes.len() - 16) % 16 != 0 { return None; }
    let w = |k: usize| i64::from_le_bytes(bytes[8 * k..8 * k + 8].try_into().unwrap());
    let ndim = (bytes.len() - 16) / 16;
    Some(CArr { alloc: pointers[0].target.allocation, off: pointers[0].target.offset, size: w(1),
                shape: (0..ndim).map(|d| w(2 + d)).collect(), strides: (0..ndim).map(|d| w(2 + ndim + d)).collect() })
}
fn scalar_f32(a: &submit::Argument) -> Option<f32> {
    match a { submit::Argument::Bytes(b) if b.len() >= 4 => Some(f32::from_le_bytes(b[..4].try_into().unwrap())), _ => None }
}

impl Ctx<'_> {
    pub fn el(&self, alloc: u64, off: usize) -> Result<i64, String> {
        Ok(self.base_el.get(&alloc).ok_or_else(|| format!("allocation {alloc} has no shadow"))? + (off / 4) as i64)
    }
    /// Shadow view of a CArray of `item`-byte elements. The view starts at the shadow word of its first byte; a
    /// float64 view steps one shadow word per element (byte stride / 8), the convention the entries' float64
    /// bindings over the float32 sweep shadows use (deriv.rs `widen`: element i of a view at byte offset o is shadow
    /// word o/4 + i), so a whole-array float64 copy and an entry binding the same array agree word for word.
    fn view(&self, c: &CArr, item: i64) -> Result<DevView, String> {
        if c.shape.len() > 4 { return Err(format!("view rank {} above 4", c.shape.len())); }
        if c.off % 4 != 0 { return Err("view origin not word aligned".into()); }
        if item != 4 && item != 8 { return Err(format!("{item}-byte elements have no float shadow rule")); }
        let mut v = DevView { base: self.el(c.alloc, c.off)?, size: c.shape.iter().product::<i64>().max(0), ndim: c.shape.len() as i32,
                              pad: 0, shape: [1; 4], stride: [0; 4] };
        for (k, (&n, &s)) in c.shape.iter().zip(&c.strides).enumerate() {
            if s % item != 0 { return Err(format!("stride {s} not a multiple of the {item}-byte element")); }
            v.shape[k] = n; v.stride[k] = s / item;
        }
        if v.ndim == 0 { v.ndim = 1; }
        Ok(v)
    }
    /// Byte span [lo, hi) of a view relative to its allocation (item bytes per element).
    fn span(c: &CArr, item: i64) -> (i64, i64) {
        let (mut lo, mut hi) = (c.off as i64, c.off as i64 + item);
        for (n, s) in c.shape.iter().zip(&c.strides) { let r = (n - 1).max(0) * s; if r < 0 { lo += r } else { hi += r } }
        (lo, hi)
    }
    fn overlap(a: &CArr, b: &CArr) -> bool {
        if a.alloc != b.alloc { return false; }
        let (x, y) = (Self::span(a, 4), Self::span(b, 4));
        x.0 < y.1 && y.0 < x.1
    }
    pub fn scheme(&self, f: &str) -> String {
        if let Some(s) = self.schemes.get(f) { return s.clone(); }
        if f.starts_with("tempo_") { "tempo".into() } else if f.starts_with("mynnsfc_") { "mynn_sfc".into() }
        else if f.starts_with("ugwp_oro_") { "gsl_drag".into() } else if f.starts_with("oracle_") || f.starts_with("ruc_") { "ruc".into() }
        else if f == "rrfs_mynnedmf_run_l16" { "mynn_edmf".into() } else { "?".into() }
    }

    /// The rule of one kernel launch, from its live arguments.
    pub fn kernel(&self, ex: &Execution, f: &str, args: &[submit::Argument], grid: [u32; 3], block: [u32; 3]) -> Result<R, String> {
        if let Some(why) = self.refuse.get(f) { return Ok(R::Refused(format!("{f}: {why}"))); }
        if self.fct_frozen && (f == "fct_scale" || f == "fct_minmax_source") {
            return Ok(self.fct_frozen(ex, f, args).unwrap_or_else(|e| R::Refused(format!("{f}: fct-frozen: {e}"))));
        }
        match f {
            // P8 row form for both directions (overrides the 11-02 glue entries, whose transpose is serial and
            // whose copy-plan descriptor table is mirrored as if every word were a pointer).
            "rrfs_post_transport_copy" => return Ok(self.post_transport(args).unwrap_or_else(|e| R::Refused(format!("{f}: {e}")))),
            "rrfs_glue_copy_plan" => return Ok(self.copy_plan_rows(ex, args, grid[1]).unwrap_or_else(|e| R::Refused(format!("{f}: {e}")))),
            // multirank (L20): the same-host halo pack/unpack (hexcore/partition_samehost_transport_v841.py _BYTE_COPY /
            // _WORD_COPY_2D), blockIdx.y = field: pure word moves, so P8 rows (tangent copies, reverse sums the halo
            // cotangent back into the owner through the packet and its DeviceCopy rule).
            "halo_copy_bytes" | "halo_copy_words_2d" =>
                return Ok(self.halo_copy_rows(ex, f, args, grid[1]).unwrap_or_else(|e| R::Refused(format!("{f}: {e}")))),
            // RUC land tile preflight: two int32 results, an exact zero rule (Ctx::integer_flags).
            "ruc_land_tile_preflight" => return Ok(Self::integer_flags(f, args, |a, o| self.el(a, o), |a| ex.allocations.get(&a).map(|x| x.bytes))
                .unwrap_or_else(|e| R::Refused(format!("{f}: {e}")))),
            "rrfs_radiation_unpack" => return Ok(self.radiation_unpack_rows(args).unwrap_or_else(|e| R::Refused(format!("{f}: {e}")))),
            _ => {}
        }
        if let Some(r) = Self::integer_result(f, args, |a, o| self.el(a, o)) { return Ok(r.unwrap_or_else(|e| R::Refused(format!("{f}: {e}")))); }
        if let Some(r) = Self::module_constant(f, args) { return Ok(r.unwrap_or_else(|e| R::Refused(format!("{f}: {e}")))); }
        if let Some(r) = Self::constant_tables(f, args, grid, block, |a, o| self.el(a, o)) { return Ok(r.unwrap_or_else(|e| R::Refused(format!("{f}: {e}")))); }
        if let Some(kd) = self.kernels.get(f) {
            let validation_only = kd.spec["launch_abi"]["entries"].as_object().is_some_and(|e| e.is_empty())
                || (f.starts_with("validate_") && kd.functions.is_empty());
            if validation_only { return Ok(R::Identity(format!("inactive:{f} (validation-only row)"))); }
            if let Some(why) = &kd.unsupported { return Ok(R::Refused(format!("{f}: entry not dispatchable: {why}"))); }
            return Ok(R::Entry);
        }
        // Input-free image constants (image_constant): the only argument is the output, and every word written is an
        // immediate of the module image, so the launch overwrites its output with a state-independent constant: zero
        // tangent, cotangent stops (R::Zero), the exact derivative. Not a held scheme launch (held = identity both ways,
        // which would keep a stale tangent on the overwritten words).
        if let Some(bytes) = image_constant(f) {
            let Some(submit::Argument::Device(out)) = args.first() else {
                return Ok(R::Refused(format!("{f}: image constant whose output is not a device pointer argument")));
            };
            if args.len() != 1 || out.offset % 4 != 0 {
                return Ok(R::Refused(format!("{f}: image constant with {} arguments / unaligned output", args.len())));
            }
            return Ok(R::Zero { ranges: vec![(self.el(out.allocation, out.offset)?, bytes.div_ceil(4))],
                                label: format!("constant:{f} (input-free image constant, {bytes} bytes)") });
        }
        let arrays: Vec<CArr> = args.iter().filter_map(carr).collect();
        if let Some(d) = f.strip_prefix("cupy_copy__") {
            let t: Vec<&str> = d.split('_').collect();
            let float = |s: &str| s.starts_with("float");
            // No float destination: integer or bool result, nothing to differentiate (whatever the argument layout).
            if t.len() == 2 && !float(t[1]) { return Ok(R::Identity(format!("inactive:{f} (integer or bool result)"))); }
            if t.len() == 2 && arrays.len() >= 2 {
                let (src, dst) = (&arrays[0], &arrays[1]);
                let item = |s: &str| if s == "float64" { 8 } else if s == "float32" { 4 } else { 0 };
                if float(t[0]) && float(t[1]) { return Ok(self.copy_view(f, src, dst, item(t[0]), item(t[1]))); }
                if float(t[1]) {
                    return Ok(match self.view(dst, item(t[1])) { Ok(v) => R::ZeroView { dst: v, label: format!("inactive-source copy {f}") },
                                                     Err(e) => R::Refused(format!("{f}: {e}")) });
                }
                return Ok(R::Identity(format!("inactive:{f} (integer or bool result)")));
            }
            let item = |s: &str| if s == "float64" { 8 } else if s == "float32" { 4 } else { 0 };
            // g3-radland: `a[...] = x` with x a host scalar (CuPy names its type `float` or `int`): one float destination
            // view, state-independent words.
            if t.len() == 2 && float(t[1]) && arrays.len() == 1 {
                return Ok(match self.view(&arrays[0], item(t[1])) { Ok(v) => R::ZeroView { dst: v, label: format!("host-scalar fill {f}") },
                                                                    Err(e) => R::Refused(format!("{f}: {e}")) });
            }
            return Ok(R::Refused(format!("{f}: unexpected copy arguments")));
        }
        match f {
            "cupy_fill" => if let Some(dst) = arrays.last() {
                return Ok(match self.view(dst, 4) { Ok(v) => R::ZeroView { dst: v, label: "constant fill".into() }, Err(e) => R::Refused(format!("{f}: {e}")) });
            },
            "cupy_maximum__float_float32_float32" => if arrays.len() == 2 {
                let (x, out) = (&arrays[0], &arrays[1]);
                let c = args.iter().find_map(|a| if carr(a).is_none() { scalar_f32(a) } else { None }).ok_or("maximum constant")?;
                let (lo, hi) = Self::span(x, 4);
                let (ov, iv) = match (self.view(out, 4), self.view(x, 4)) { (Ok(o), Ok(i)) => (o, i), (Err(e), _) | (_, Err(e)) => return Ok(R::Refused(format!("{f}: {e}"))) };
                if lo < 0 || lo % 4 != 0 { return Ok(R::Refused(format!("{f}: unaligned primal span"))); }
                let mut xv = iv; xv.base = (x.off as i64 - lo) / 4;
                if ov.base != iv.base && Self::overlap(x, out) { return Ok(R::Refused(format!("{f}: partially overlapping in/out"))); }
                return Ok(R::Max { out: ov, x: iv, prim: (x.alloc, lo as usize / 4, ((hi - lo + 3) / 4) as usize), xv, c });
            },
            "cupy_copy_where__float32_bool_float32" => if arrays.len() == 3 {
                let (src, mask, dst) = (&arrays[0], &arrays[1], &arrays[2]);
                let (dv, sv) = match (self.view(dst, 4), self.view(src, 4)) { (Ok(d), Ok(s)) => (d, s), (Err(e), _) | (_, Err(e)) => return Ok(R::Refused(format!("{f}: {e}"))) };
                if Self::overlap(src, dst) && dv.base != sv.base { return Ok(R::Refused(format!("{f}: overlapping src/dst"))); }
                if mask.shape.len() > 4 { return Ok(R::Refused(format!("{f}: mask rank"))); }
                let (lo, hi) = Self::span(mask, 1);
                if lo < 0 { return Ok(R::Refused(format!("{f}: mask span"))); }
                let mut mv = DevView { base: mask.off as i64 - lo, size: mask.shape.iter().product::<i64>().max(0), ndim: mask.shape.len().max(1) as i32,
                                       pad: 0, shape: [1; 4], stride: [0; 4] };
                for (k, (&n, &s)) in mask.shape.iter().zip(&mask.strides).enumerate() { mv.shape[k] = n; mv.stride[k] = s; }
                return Ok(R::Where { dst: dv, src: sv, prim: (mask.alloc, lo as usize, (hi - lo) as usize), mv });
            },
            // g3-radland: x / c with c a host constant: d_out = d_x / c (the multiply rule's Scale, factor 1/c).
            "cupy_true_divide__float32_float_float32" => if arrays.len() == 2 {
                let (i, o) = (&arrays[0], &arrays[1]);
                let contiguous = |c: &CArr| { let mut e = 4; for d in (0..c.shape.len()).rev() { if c.shape[d] > 1 && c.strides[d] != e { return false; } e *= c.shape[d].max(1); } true };
                // the divisor is the one 4-byte by-value argument (CuPy passes a float32 scalar; a float64 one is refused)
                let fours: Vec<f32> = args.iter().filter_map(|a| match a { submit::Argument::Bytes(b) if b.len() == 4 => Some(f32::from_le_bytes(b[..4].try_into().unwrap())), _ => None }).collect();
                if fours.len() != 1 { return Ok(R::Refused(format!("{f}: divisor is not one float32 scalar"))); }
                let c = fours[0];
                if i.size != o.size || !contiguous(i) || !contiguous(o) { return Ok(R::Refused(format!("{f}: strided true_divide"))); }
                if i.alloc == o.alloc && i.off != o.off && Self::overlap(i, o) { return Ok(R::Refused(format!("{f}: partially overlapping true_divide"))); }
                if c == 0.0 || !c.is_finite() { return Ok(R::Refused(format!("{f}: divisor {c} (the forward is not finite)"))); }
                return Ok(R::Scale { input: self.el(i.alloc, i.off)?, output: self.el(o.alloc, o.off)?, n: i.size as usize, c: 1.0 / c as f64 });
            },
            "cupy_multiply__float32_float_float32" => if arrays.len() == 2 {
                let (i, o) = (&arrays[0], &arrays[1]);
                let contiguous = |c: &CArr| { let mut e = 4; for d in (0..c.shape.len()).rev() { if c.shape[d] > 1 && c.strides[d] != e { return false; } e *= c.shape[d].max(1); } true };
                let c = args.iter().find_map(|a| if carr(a).is_none() { scalar_f32(a) } else { None }).ok_or("multiply scalar")?;
                if i.size != o.size || !contiguous(i) || !contiguous(o) { return Ok(R::Refused(format!("{f}: strided multiply"))); }
                if i.alloc == o.alloc && i.off != o.off && Self::overlap(i, o) { return Ok(R::Refused(format!("{f}: partially overlapping multiply"))); }
                return Ok(R::Scale { input: self.el(i.alloc, i.off)?, output: self.el(o.alloc, o.off)?, n: i.size as usize, c: c as f64 });
            },
            _ => {}
        }
        let class = self.classes.get(f).map(String::as_str).unwrap_or("");
        match class {
            // COMMON-AD 87.1: the TEMPO order kernels come in as "held" from the P9 manifest (e2e held.rs: permutation),
            // but they write integer schedule words only, so their identity is their exact derivative, not a held
            // scheme. Labelled inactive so held:tempo counts only real TEMPO physics (rule unchanged: identity).
            "held" if matches!(f, "tempo_order_hist" | "tempo_order_scan" | "tempo_order_scatter") =>
                return Ok(R::Identity(format!("inactive:{f} (COMMON-AD 87.1 column-order permutation)"))),
            "held" => return Ok(R::Identity(format!("held:{}", self.scheme(f)))),
            "output-diagnostic" => return Ok(R::Identity(format!("inactive:{f} (output diagnostic)"))),
            "needs-rule" => return Ok(R::Refused(format!("{f}: P8 rule or glue entry not loaded"))),
            // g3-radland: a kernel-set data row for a launch whose every written word is bool or integer (the radiation
            // slot's CuPy masks, scans and integer arithmetic): no float state written, identity both ways.
            "inactive" => return Ok(R::Identity(format!("inactive:{f} (bool or integer result, kernel-set class)"))),
            _ => {}
        }
        // P8 rules.rs kernel_class: inactive (bool / integer results, validators, fixed geometry, diagnostics,
        // TEMPO column-order permutations) and pending (fused physics owned by P3-P6: held while unbound).
        const INACTIVE: &[&str] = &["cupy_less__float32_float32_bool", "cupy_not_equal__int32_int_bool", "cupy_cub_any_pass1",
            "cupy_cub_any_pass2", "hex_transport_species_words", "hex_transport_species_levels", "validate_density_v841",
            "validate_finite_array_v841_f32", "validate_recovered_v841_f32", "w_filter_geometry_v841_f32", "smooth_helicity",
            "split_helicity_pool", "vertex_helicity_gather_pool", "update_diag_pool", "mpas_output_diagnostics_v842",
            "oracle_diags_sfcdiags_ruclsm", "tempo_radar", "tempo_order_hist", "tempo_order_scan", "tempo_order_scatter",
            // diagnostic pool kernels (kernel-set class output-diagnostic for compute_diag / reset_diag)
            "compute_diag_cols", "reset_diag_pool"];
        if INACTIVE.contains(&f) || f.starts_with("validate_") { return Ok(R::Identity(format!("inactive:{f}"))); }
        if f.starts_with("tempo_") || f.starts_with("mynnsfc_") || f.starts_with("ugwp_oro_") || f.starts_with("oracle_")
            || f == "rrfs_mynnedmf_run_l16" { return Ok(R::Identity(format!("held:{}", self.scheme(f)))); }
        Ok(R::Refused(format!("{f}: no derivative row and no rule")))
    }

    /// Setup launches whose only output is an integer (held lane rrfs_init_microphysics_top, 2026-11-06).
    /// rrfs_init_microphysics_top (TEMPO setup, RRFS 44c0849 mpas_atmphys_init.F:284-292, forward in
    /// hexcore/mpasphys/rrfs/driver_rrfs.py) writes ONE int32: the last level k whose running height
    /// 0.5/rdzw[0] + dzu[1] + ... + dzu[k-1] is <= top. It is piecewise constant in (top, rdzw, dzu), so its derivative
    /// is exactly zero wherever it exists: the tangent of the output word is zero and the reverse drops its cotangent
    /// (the exact transpose; Identity would pass the overwritten word's cotangent through and is not). No float word is
    /// written, no input is read for a derivative. At a switch point (running height == top) the level jumps by one: a
    /// step with no derivative, never approximated here; the margin on the captured inputs is in the lane's RESULT.md.
    /// Before this rule the launch fell through to "no derivative row and no rule" (Refused) if a window ever composed
    /// step-0 setup (pkg18 does not: 0 launches in plan.json d06ab544).
    fn integer_result(f: &str, args: &[submit::Argument], el: impl Fn(u64, usize) -> Result<i64, String>) -> Option<Result<R, String>> {
        if f != "rrfs_init_microphysics_top" { return None; }
        Some((|| {
            if args.len() != 5 { return Err(format!("{} arguments, not 5 (levels, top, rdzw, dzu, level)", args.len())); }
            let submit::Argument::Device(s) = &args[4] else { return Err("level (argument 4) is not a device pointer".into()) };
            if s.offset % 4 != 0 { return Err("level word not aligned".into()); }
            Ok(R::Zero { ranges: vec![(el(s.allocation, s.offset)?, 1)],
                         label: "integer result: microphysics top level (rrfs_init_microphysics_top)".into() })
        })())
    }
    /// Module-constant initializers (held lane ruc_tbq_init, 2026-11-06).
    /// ruc_tbq_init (RUC setup, RRFS-MPAS 1d6e370e module_ruc_land.F90:479-496 and module_ruc_ice.F90:364-380, forward in
    /// hexcore/mpasphys/rrfs/ruc_lsm_8c9f1a8.py:86) takes NO arguments and writes only the module global
    /// `ruc_tbq_table[5001]` (the saturation-pressure table) from compiled-in constants: cq = 173.15 + 0.05 (k-1), exp of a
    /// constant expression, times 6.1153 * 0.62198. Nothing it reads is model state, so its derivative is exactly zero:
    /// the tangent of every table word is 0 and the reverse has no input to send a cotangent to. The table is not a
    /// driver allocation, so it has no window shadow; land and sea ice (oracle_ice_ruc_ice, oracle_land_ruc_land) read it
    /// as a primal constant, which is that same zero tangent. On the shadow arena the launch therefore writes no word:
    /// Identity both ways is the exact operator here, not a refusal. Its words do not depend on the launch either
    /// (explicitly rounded arithmetic; the lane's RESULT.md has the table sha256 from three images and two loads).
    /// THE BREAKAGE THIS PREVENTS: a window that composes the step-0 setup (pkg18 runs it as plan.module_initializers
    /// at load, sequence 14003, and never as a command) reached "no derivative row and no rule" and listed the launch as
    /// Refused. Refuses by name if the launch ever carries an argument: then it could write a shadowed word.
    fn module_constant(f: &str, args: &[submit::Argument]) -> Option<Result<R, String>> {
        if f != "ruc_tbq_init" { return None; }
        Some(if args.is_empty() {
            Ok(R::Identity("inactive:ruc_tbq_init (constant module table ruc_tbq_table, no shadowed word)".into()))
        } else {
            Err(format!("{} arguments, not 0: a module-constant initializer takes none", args.len()))
        })
    }
    /// RRTMG LW lookup-table setup (held lane mpas_init_tables, 2026-11-06).
    /// mpas_init_tables (MPAS module_ra_rrtmg_lw.F:8054-8079 rrtmg_lw_ini, forward in hexcore/mpasphys/rrtmg_lw_v842.py)
    /// reads NO input: thread i = blockIdx.x*blockDim.x+threadIdx.x (i <= 10000) overwrites tau[i], exp_table[i] and
    /// tfn[i] from i and compiled-in constants (t = i/1e4, tau = t/0.278/(1-t), exp = max(exp(-tau), 1e-20), tfn the Pade
    /// transition). Its derivative with respect to everything is exactly zero, so the rule is R::Zero on the 3 x 10001
    /// written words: tangent zeroed, cotangent dropped (the exact transpose; Identity would pass the overwritten words'
    /// cotangent through and is not). The fmaxf(.., 1e-20) clamp and the tau < 0.06 branch select between constants of the
    /// index only, so no frozen-coefficient treatment is needed. Words past 10000 (allocation tails) are not written and
    /// keep identity. Refused by name unless the launch has the 3 device-pointer arguments, word-aligned, every table has a
    /// window shadow, and grid.x*block.x covers the 10001 words (a narrower launch would leave words unwritten that
    /// R::Zero would clear). Before this rule a window composing step-0 setup reached "no derivative row and no rule"
    /// (Refused); pkg18 has no radiation launch (0 of plan.json d06ab544), so the 18-step window is unchanged.
    fn constant_tables(f: &str, args: &[submit::Argument], grid: [u32; 3], block: [u32; 3],
                       el: impl Fn(u64, usize) -> Result<i64, String>) -> Option<Result<R, String>> {
        if f != "mpas_init_tables" { return None; }
        const WORDS: usize = 10001;
        Some((|| {
            if args.len() != 3 { return Err(format!("{} arguments, not 3 (tau, exp_table, tfn)", args.len())); }
            let span = grid[0] as u64 * block[0] as u64;
            if span < WORDS as u64 { return Err(format!("grid.x*block.x = {span} writes fewer than the {WORDS} table words")); }
            let mut ranges = Vec::with_capacity(3);
            for (k, name) in ["tau", "exp_table", "tfn"].iter().enumerate() {
                let submit::Argument::Device(s) = &args[k] else { return Err(format!("{name} (argument {k}) is not a device pointer")) };
                if s.offset % 4 != 0 { return Err(format!("{name} not word aligned")); }
                ranges.push((el(s.allocation, s.offset)?, WORDS));
            }
            Ok(R::Zero { ranges, label: "constant table: RRTMG LW tau/exp/tfn lookup tables (mpas_init_tables)".into() })
        })())
    }

    fn copy_view(&self, f: &str, src: &CArr, dst: &CArr, si: i64, di: i64) -> R {
        let (dv, sv) = match (self.view(dst, di), self.view(src, si)) { (Ok(d), Ok(s)) => (d, s), (Err(e), _) | (_, Err(e)) => return R::Refused(format!("{f}: {e}")) };
        if src.shape != dst.shape { return R::Refused(format!("{f}: source and destination shapes differ")); }
        if dst.shape.iter().zip(&dst.strides).any(|(&n, &s)| n > 1 && s == 0) { return R::Refused(format!("{f}: repeated destination")); }
        if si == di && dv.base == sv.base && dv.stride == sv.stride { return R::Identity(format!("inactive:{f} (in place)")); }
        if Self::overlap(src, dst) { return R::Refused(format!("{f}: overlapping copy")); }
        if src.shape.iter().zip(&src.strides).any(|(&n, &s)| n > 1 && s == 0) {
            // Broadcasting source: fan-in, so the transpose goes through P8's deterministic gather (rows.rs from_view).
            let pad = 4 - dst.shape.len();
            let (mut n, mut ds, mut ss) = ([1i64; 4], [0i64; 4], [0i64; 4]);
            for k in 0..dst.shape.len() { n[pad + k] = dst.shape[k]; ds[pad + k] = dst.strides[k] / di; ss[pad + k] = src.strides[k] / si; }
            return R::Rows { rows: vec![DevRow { dst: dv.base, src: sv.base, ds, ss, n: [n[1], n[2], n[3]], count: n.iter().product() }], label: "broadcast copy" };
        }
        R::View { dst: dv, src: sv }
    }

    /// rrfs_post_transport_copy: twelve rows of (dst, src, count, n1, n2, n3, d0..d3, s0..s3) (P8 rows::post_transport).
    fn post_transport(&self, a: &[submit::Argument]) -> Result<R, String> {
        if a.len() != 168 { return Err(format!("{} arguments, not 168", a.len())); }
        let int = |x: &submit::Argument| match x { submit::Argument::Bytes(b) if b.len() == 8 => Ok(i64::from_le_bytes(b[..8].try_into().unwrap())), _ => Err("bad scalar".to_string()) };
        let mut rows = Vec::new();
        for r in 0..12 {
            let g = &a[14 * r..14 * r + 14];
            let w: Vec<i64> = g[2..].iter().map(int).collect::<Result<_, _>>()?;
            if w[0] == 0 { continue; }
            let (submit::Argument::Device(d), submit::Argument::Device(s)) = (&g[0], &g[1]) else { return Err("row without pointers".into()) };
            rows.push(DevRow { dst: self.el(d.allocation, d.offset)?, src: self.el(s.allocation, s.offset)?, n: [w[1], w[2], w[3]],
                               ds: [w[4], w[5], w[6], w[7]], ss: [w[8], w[9], w[10], w[11]], count: w[0] });
        }
        Ok(R::Rows { rows, label: "post-transport copy" })
    }

    /// rrfs_glue_copy_plan: the table the live RrfsCopyPlanUpload built (copy_plan::table: 128-byte rows, dims already in
    /// the kernel's order), read from the device and mapped back to plan storage. Fill rows have no source.
    /// G2 (P1 milestone 2): the held radiation tendency rule. rrfs_radiation_unpack (resident_radiation_rrfs.py:19-30,
    /// 23 rows of ARGS(N) = unsigned d, const unsigned s, a, b, c, f, n, w (i64), v (u32)) moves each radiation call's
    /// results into the MPAS pool arrays (rthratenlw/sw, fluxes, radii) that every later step reads until the next
    /// radiation call: d[row*a+col*b] = s ? s[row*c+col*f] : v, row = e/w, col = e%w, word strides. As P8 copy rows
    /// (n1 = n2 = 1, n3 = w, ds = [a, 0, 0, b]) the tangent copies and the reverse moves the cotangent the carried steps
    /// accumulated on the pool arrays back onto the radiation outputs; fill rows (constant v) are zero. THE BREAKAGE
    /// THIS PREVENTS: Refused (identity) claimed d(pool)/d(pool) = 1 for the overwritten tendency and dropped every
    /// later step's cotangent of the radiation step.
    fn radiation_unpack_rows(&self, a: &[submit::Argument]) -> Result<R, String> {
        if a.len() % 9 != 0 { return Err(format!("{} arguments, not rows of 9", a.len())); }
        let int = |x: &submit::Argument| -> Result<i64, String> { match x {
            submit::Argument::Bytes(b) if b.len() == 8 => Ok(i64::from_le_bytes(b[..8].try_into().unwrap())),
            _ => Err("stride/count word is not a by-value i64".into()) } };
        let mut rows = Vec::new();
        for g in a.chunks(9) {
            let n = int(&g[6])?;
            if n == 0 { continue; }
            let w = int(&g[7])?;
            if n < 0 || w <= 0 { return Err(format!("row n {n} w {w}")); }
            let submit::Argument::Device(d) = &g[0] else { return Err("row destination is not a device pointer".into()) };
            let src = match &g[1] {
                submit::Argument::Device(s) => self.el(s.allocation, s.offset)?,
                submit::Argument::Bytes(b) if b.iter().all(|&x| x == 0) => -1,
                _ => return Err("row source is neither a device pointer nor null".into()),
            };
            rows.push(DevRow { dst: self.el(d.allocation, d.offset)?, src, ds: [int(&g[2])?, 0, 0, int(&g[3])?],
                               ss: [int(&g[4])?, 0, 0, int(&g[5])?], n: [1, 1, w], count: n });
        }
        // AD_UNPACK_ZERO=n1,n2 (bisection diagnostic, default off): a launch with that many nonempty rows has every row
        // turned into a fill row, so the radiation outputs it hands to the model carry zero derivative (consistent both
        // ways). Used to split the LW and SW radiation paths in a Taylor check.
        if let Ok(z) = std::env::var("AD_UNPACK_ZERO") {
            if z.split(',').any(|x| x.trim().parse::<usize>().ok() == Some(rows.len())) { for r in rows.iter_mut() { r.src = -1; } }
        }
        // AD_UNPACK_ZERO_OPS=N:i,j,.. (bisection diagnostic): in a launch with N nonempty rows, only rows i, j, .. (their
        // order among the nonempty rows) become fill rows.
        if let Ok(z) = std::env::var("AD_UNPACK_ZERO_OPS") {
            if let Some((n, list)) = z.split_once(':') {
                if n.trim().parse::<usize>().ok() == Some(rows.len()) {
                    for i in list.split(',').filter_map(|x| x.trim().parse::<usize>().ok()) { if let Some(r) = rows.get_mut(i) { r.src = -1; } }
                }
            }
        }
        Ok(R::Rows { rows, label: "radiation unpack (held tendency)" })
    }

    /// cupy_concatenate_same_size (CuPy `_concatenate_kernel_same_size`): y[i] = x[a][top*base + r] with
    /// i = (top*nx + a)*base + r, x a by-value CArray of nx device pointers, base the by-value i64 block length (P8
    /// rows::concatenate, the same rows). `ty` is the module's element type (concat_module_types):
    /// * float: copy rows, one per source array (tangent copies, the reverse moves y's cotangent onto the sources and
    ///   clears y's words: the exact transpose of an overwrite). The pointer table is read from the device (the forward
    ///   uploaded it) and every address mapped back to a live allocation, as copy_plan_rows does;
    /// * an integer or bool type: no float word moves, identity both ways (inactive);
    /// * anything else, or no recorded source: Refused, never guessed.
    /// Refused also when y is not contiguous 4-byte words, a source span leaves its allocation, a source overlaps y, or
    /// y's size is not a multiple of nx*base.
    pub fn concatenate(&self, ex: &Execution, f: &str, a: &[submit::Argument], ty: Option<&str>) -> R {
        match ty {
            Some("float") => {}
            Some(t @ ("long long" | "long" | "int" | "unsigned long long" | "unsigned int" | "bool")) =>
                return R::Identity(format!("inactive:{f} (integer concatenation, typedef {t} T)")),
            t => return R::Refused(format!("{f}: element type {t:?} not established from the module's recorded source")),
        }
        (|| -> Result<R, String> {
            let x = a.first().and_then(carr).ok_or("argument 0 is not the pointer-array CArray")?;
            let base = match a.get(1) { Some(submit::Argument::Bytes(b)) if b.len() == 8 => i64::from_le_bytes(b[..8].try_into().unwrap()),
                                        _ => return Err("argument 1 (base) is not a by-value i64".into()) };
            let y = a.get(2).and_then(carr).ok_or("argument 2 is not the output CArray")?;
            if x.shape.len() != 1 || x.strides.first() != Some(&8) || x.size <= 0 { return Err(format!("pointer array shape {:?} strides {:?}", x.shape, x.strides)); }
            if y.shape.len() != 1 || y.strides.first() != Some(&4) || y.shape[0] != y.size { return Err(format!("output not contiguous 4-byte words: shape {:?} strides {:?}", y.shape, y.strides)); }
            let nx = x.size;
            if base <= 0 || y.size % (base * nx) != 0 { return Err(format!("output size {} not a multiple of nx {nx} x base {base}", y.size)); }
            let ntop = y.size / (base * nx);
            let table = ex.allocations.get(&x.alloc).ok_or("pointer table allocation not live")?;
            if x.off + 8 * nx as usize > table.bytes { return Err("pointer table exceeds its allocation".into()); }
            let data = unsafe { crate::dev::download(&ex.cuda, table.address + x.off as u64, 8 * nx as usize)? };
            let ylo = y.off as i64; let yhi = ylo + 4 * y.size;
            let mut rows = Vec::with_capacity(nx as usize);
            for (k, w) in data.chunks_exact(8).enumerate() {
                let addr = u64::from_le_bytes(w.try_into().unwrap());
                let s = ex.allocations.values().find(|q| addr >= q.address && addr < q.address + q.bytes as u64)
                    .ok_or_else(|| format!("source {k} address {addr:#x} not in a live allocation"))?;
                let off = (addr - s.address) as usize;
                if off % 4 != 0 { return Err(format!("source {k} not word aligned")); }
                if off + 4 * (ntop * base) as usize > s.bytes { return Err(format!("source {k} span leaves allocation {}", s.id)); }
                if s.id == y.alloc && (off as i64) < yhi && ylo < off as i64 + 4 * ntop * base { return Err(format!("source {k} overlaps the output")); }
                rows.push(DevRow { dst: self.el(y.alloc, y.off)? + k as i64 * base, src: self.el(s.id, off)?,
                                   ds: [0, 0, nx * base, 1], ss: [0, 0, base, 1], n: [1, ntop, base], count: ntop * base });
            }
            Ok(R::Rows { rows, label: "concatenate (float32 stack)" })
        })().unwrap_or_else(|e| R::Refused(format!("{f}: {e}")))
    }

    fn copy_plan_rows(&self, ex: &Execution, a: &[submit::Argument], nrows: u32) -> Result<R, String> {
        let Some(submit::Argument::Device(t)) = a.first() else { return Err("copy plan without table".into()) };
        let table = ex.allocations.get(&t.allocation).ok_or("table allocation not live")?;
        let bytes = nrows as usize * 128;
        if t.offset + bytes > table.bytes { return Err("copy plan table exceeds its allocation".into()); }
        let data = unsafe { crate::dev::download(&ex.cuda, table.address + t.offset as u64, bytes)? };
        let find = |addr: u64| -> Result<(u64, usize), String> {
            ex.allocations.values().find(|x| addr >= x.address && addr < x.address + x.bytes as u64)
                .map(|x| (x.id, (addr - x.address) as usize)).ok_or_else(|| format!("descriptor address {addr:#x} not in a live allocation"))
        };
        let mut rows = Vec::with_capacity(nrows as usize);
        for r in data.chunks_exact(128) {
            let w = |k: usize| i64::from_le_bytes(r[8 * k..8 * k + 8].try_into().unwrap());
            let mode = i32::from_le_bytes(r[124..128].try_into().unwrap());
            let (da, doff) = find(w(0) as u64)?;
            if doff % 4 != 0 { return Err("unaligned descriptor".into()); }
            let src = if mode == 1 { -1 } else { let (sa, so) = find(w(1) as u64)?; self.el(sa, so)? };
            rows.push(DevRow { dst: self.el(da, doff)?, src, ds: [w(2), w(3), w(4), w(5)], ss: [w(6), w(7), w(8), w(9)],
                               n: [w(11), w(12), w(13)], count: w(14) });
        }
        Ok(R::Rows { rows, label: "glue copy plan" })
    }

    /// multirank (L20): halo_copy_bytes / halo_copy_words_2d as P8 rows. The launch's descriptor table (9 i64 per field:
    /// live, packet, indices, rows, count, itemsize, ndim, dims, unit; partition_samehost_transport_v841.py) is read from
    /// the device with its index and dims arrays and mapped back to plan storage. Packet words keep the word convention
    /// (packet word p = shadow word base + p, as the DeviceCopy rule moves them); live float32/int32 words too; a float64
    /// field's element k (byte offset 8k from the field origin) is shadow word origin/4 + k (deriv.rs `widen`), carried in
    /// the first of its two packet words (the second gets no shadow move). Byte fields (bool/int8) carry no float word.
    /// THE BREAKAGE THIS PREVENTS: with no rule the launches are Refused (identity both ways), so the tangent never
    /// crosses a rank boundary and the reverse drops every halo cotangent instead of summing it into the owner.
    fn halo_copy_rows(&self, ex: &Execution, f: &str, a: &[submit::Argument], fields: u32) -> Result<R, String> {
        let (Some(submit::Argument::Device(t)), Some(submit::Argument::Bytes(sc))) = (a.first(), a.get(1)) else {
            return Err("arguments are not (table, scatter)".into()) };
        if a.len() != 2 || sc.len() < 4 { return Err(format!("{} arguments", a.len())); }
        let scatter = i32::from_le_bytes(sc[..4].try_into().unwrap()) != 0;
        let table = ex.allocations.get(&t.allocation).ok_or("table allocation not live")?;
        let bytes = fields as usize * 72;
        if t.offset + bytes > table.bytes { return Err("halo table exceeds its allocation".into()); }
        let data = unsafe { crate::dev::download(&ex.cuda, table.address + t.offset as u64, bytes)? };
        let find = |addr: u64| -> Result<(u64, usize), String> {
            ex.allocations.values().find(|x| addr >= x.address && addr < x.address + x.bytes as u64)
                .map(|x| (x.id, (addr - x.address) as usize)).ok_or_else(|| format!("halo address {addr:#x} not in a live allocation"))
        };
        let words = |addr: u64, n: usize| -> Result<Vec<i64>, String> {
            let (id, off) = find(addr)?;
            let x = &ex.allocations[&id];
            if off + n * 8 > x.bytes { return Err("halo index/dims array exceeds its allocation".into()); }
            let b = unsafe { crate::dev::download(&ex.cuda, x.address + off as u64, n * 8)? };
            Ok(b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect())
        };
        let mut rows = Vec::new();
        let (mut f64_fields, mut byte_fields) = (0usize, 0usize);
        for r in data.chunks_exact(72) {
            let w = |k: usize| i64::from_le_bytes(r[8 * k..8 * k + 8].try_into().unwrap());
            let (nrows, count) = (w(3), w(4));
            if nrows <= 0 || count <= 0 { continue; }
            let (la, lo) = find(w(0) as u64)?;
            let (pa, po) = find(w(1) as u64)?;
            if po % 4 != 0 || lo % 4 != 0 { return Err("unaligned live/packet origin".into()); }
            let idx = words(w(2) as u64, count as usize)?;
            let (live0, pack0) = (self.el(la, lo)?, self.el(pa, po)?);
            let mut push = |pk: i64, lv: i64, ds: [i64; 4], ss: [i64; 4], n: [i64; 3]| {
                let (dst, src, d, s) = if scatter { (lv, pk, ss, ds) } else { (pk, lv, ds, ss) };
                rows.push(DevRow { dst, src, ds: d, ss: s, n, count: nrows });
            };
            if f == "halo_copy_words_2d" {
                // offset = row * row_stride + indices[column] (words); packet q = row * columns + column.
                let dims = words(w(7) as u64, 2)?;
                if dims[1] % 4 != 0 { return Err("word row stride not a multiple of 4 bytes".into()); }
                let rs = dims[1] / 4;
                for (c, &i) in idx.iter().enumerate() {
                    push(pack0 + c as i64, live0 + i, [count, 0, 0, 0], [rs, 0, 0, 0], [1, 1, 1]);
                }
                continue;
            }
            let (item, ndim, unit) = (w(5), w(6), w(8));
            if unit == 1 {
                if item < 4 { byte_fields += 1; continue; }
                return Err(format!("{item}-byte field moved bytewise"));
            }
            if unit != 4 || (item != 4 && item != 8) { return Err(format!("unit {unit} item {item}")); }
            if !(1..=5).contains(&ndim) { return Err(format!("field rank {ndim}")); }
            let dims = words(w(7) as u64, 2 * ndim as usize)?;
            let upi = item / 4;
            let width = count * upi;
            // leading dims d_0..d_{m-1} (m = ndim - 1), fastest last; P8 decomposition i3 fastest, then i2, i1, q.
            let m = (ndim - 1) as usize;
            let (mut n, mut ls, mut ps) = ([1i64; 4], [0i64; 4], [0i64; 4]);
            let mut span = width;
            for k in 0..m {
                let slot = 3 - k;
                let dim = m - 1 - k;
                let (shape, stride) = (dims[2 * dim], dims[2 * dim + 1]);
                if stride % item != 0 { return Err(format!("stride {stride} not a multiple of the {item}-byte item")); }
                n[slot] = shape; ls[slot] = stride / item; ps[slot] = span; span *= shape;
            }
            if n.iter().product::<i64>() != nrows { return Err(format!("rows {nrows} != leading shape {n:?}")); }
            let cell = dims[2 * m + 1];
            if cell % item != 0 { return Err(format!("cell stride {cell} not a multiple of the {item}-byte item")); }
            if item == 8 { f64_fields += 1; }
            for (c, &i) in idx.iter().enumerate() {
                // float32/int32: live word = origin + i*cell/4 + sum coord*stride/4; float64: element units (deriv.rs widen).
                push(pack0 + c as i64 * upi, live0 + i * (cell / item), ps, ls, [n[1], n[2], n[3]]);
            }
        }
        let label: &'static str = match (f64_fields > 0, byte_fields > 0) {
            (false, false) => "same-host halo copy",
            (true, false) => "same-host halo copy (float64 fields)",
            (false, true) => "same-host halo copy (byte fields inactive)",
            (true, true) => "same-host halo copy (float64 + byte fields)",
        };
        Ok(R::Rows { rows, label })
    }

    /// Frozen FCT limiter (`--fct-frozen 1`, off by default; a deliberate choice over iterating on a limiter
    /// that cannot be differentiated). The limiter coefficients are taken from the forward trajectory and treated as constants in
    /// the tangent and the adjoint: an APPROXIMATION of the forward's derivative (the standard
    /// treatment of a non-differentiable limiter in variational systems), named in the receipt. fct_scale's outputs scale_in / scale_out (the
    /// coefficients, written in place over the low-order flux sums) get zero tangent and their cotangent is dropped;
    /// fct_minmax_source keeps the exact derivative of source_old (transported state) and gives the limiter bounds
    /// none. The other FCT kernels run their entries, so fct_limit_* are linear in the antidiffusive flux with the
    /// trajectory's weights. Why: the exact limiter derivative divides by (flux sum + 1e-20) and overflowed float32
    /// at degenerate tracer bounds (L1 trace4, 2026-11-05 07:15Z).
    fn fct_frozen(&self, ex: &Execution, f: &str, args: &[submit::Argument]) -> Result<R, String> {
        let kd = self.kernels.get(f).ok_or("no kernel row (argument names)")?;
        let pos = |n: &str| kd.names.iter().position(|x| x == n).ok_or_else(|| format!("no argument {n}"));
        let int = |n: &str| -> Result<i32, String> { match args.get(pos(n)?) {
            Some(submit::Argument::Bytes(b)) if b.len() >= 4 => Ok(i32::from_le_bytes(b[..4].try_into().unwrap())),
            _ => Err(format!("{n} is not a by-value int")) } };
        let (nt, nl, nc) = (int("ntracers")?, int("nlev")?, int("ncells")?);
        if nt <= 0 || nl <= 0 || nc <= 0 { return Err(format!("shape {nt} x {nl} x {nc}")); }
        let q = (nt as usize) * (nl as usize) * (nc as usize);
        // (allocation, byte offset) of a device argument, checked to hold `words` float words.
        let dev = |n: &str, words: usize| -> Result<(u64, usize), String> {
            match args.get(pos(n)?) {
                Some(submit::Argument::Device(s)) => {
                    let a = ex.allocations.get(&s.allocation).ok_or_else(|| format!("{n}: allocation not live"))?;
                    if s.offset % 4 != 0 || s.offset + 4 * words > a.bytes {
                        return Err(format!("{n}: {words} words at byte {} exceed allocation {} ({} bytes)", s.offset, s.allocation, a.bytes));
                    }
                    Ok((s.allocation, s.offset))
                },
                _ => Err(format!("{n} is not a device pointer")),
            }
        };
        if f == "fct_scale" {
            let (si, so) = (dev("scale_in", q)?, dev("scale_out", q)?);
            return Ok(R::Zero { ranges: vec![(self.el(si.0, si.1)?, q), (self.el(so.0, so.1)?, q)],
                                label: "fct-frozen: limiter coefficients scale_in/scale_out (fct_scale)".into() });
        }
        let dt = match args.get(pos("dt")?) { Some(submit::Argument::Bytes(b)) if b.len() >= 4 => f32::from_le_bytes(b[..4].try_into().unwrap()),
                                               _ => return Err("dt is not a by-value float".into()) };
        let r = (nl as usize) * (nc as usize);
        let spans = [("old", q), ("source", q), ("rho_old", r), ("source_old", q), ("minimum", q), ("maximum", q)];
        let mut at = Vec::new();
        for (n, w) in spans { let d = dev(n, w)?; at.push((d.0, d.1, w)); }
        // The kernels assume every output is a separate array from every input (true in the forward source).
        for i in 0..6 { for j in i + 1..6 {
            let (a, b) = (at[i], at[j]);
            if a.0 == b.0 && a.1 < b.1 + 4 * b.2 && b.1 < a.1 + 4 * a.2 { return Err(format!("{} and {} overlap", spans[i].0, spans[j].0)); }
        } }
        let mut e = [0i64; 6];
        for (k, x) in at.iter().enumerate() { e[k] = self.el(x.0, x.1)?; }
        Ok(R::FctMinmax { nt, nl, nc, dt, e, prim: [(at[1].0, at[1].1 / 4, q), (at[2].0, at[2].1 / 4, r)] })
    }

    /// ruc_land_tile_preflight (held lane ruc_land_tile_preflight, 2026-11-06): the RUC land driver's tile stop words,
    /// RRFS-MPAS 1d6e370e module_ruc_land.F90:498-582, forward in hexcore/mpasphys/rrfs/ruc_lsm_8c9f1a8.py:5096 (captured
    /// source e6fd85f0 lines 6395-6400). One thread writes TWO int32 words and nothing else: stop = {nc, 0}; then, if
    /// lakemodel (int, ptrs[8][0]) == 1, the first cell c with lakemask (float, ptrs[9][c]) == 1.0f gives stop = {c, 1}.
    /// The result is integer and piecewise constant in every input (an integer test and a float equality), so its
    /// derivative is exactly zero wherever it exists: zero tangent on the two stop words and their cotangent dropped (the
    /// exact transpose; Identity would claim d(stop_out)/d(stop_in) = 1 for an overwritten word, which is wrong). No float
    /// word is written and no input gets a cotangent. The test lakemask == 1.0f is a step: a mask word exactly at 1.0 has
    /// no derivative (one ulp flips stop). It is never approximated: the branch is the trajectory's, like every integer
    /// word downstream (oracle_land_ruc_land reads stop as its cell bound and preparation-only flag). Before this rule
    /// the launch fell through to "no derivative row and no rule" (Refused, identity both ways) whenever a window composed
    /// RUC land live (pkg18 does not: land is a held upload there, 0 launches in plan.json d06ab544).
    fn integer_flags(f: &str, args: &[submit::Argument], el: impl Fn(u64, usize) -> Result<i64, String>,
                     bytes: impl Fn(u64) -> Option<usize>) -> Result<R, String> {
        if args.len() != 3 { return Err(format!("{} arguments, not 3 (ptrs, stop, nc)", args.len())); }
        if !matches!(&args[0], submit::Argument::Device(_)) { return Err("ptrs (argument 0) is not a device pointer".into()); }
        let submit::Argument::Device(s) = &args[1] else { return Err("stop (argument 1) is not a device pointer".into()) };
        if !matches!(&args[2], submit::Argument::Bytes(b) if b.len() == 4) { return Err("nc (argument 2) is not a by-value int32".into()); }
        if s.offset % 4 != 0 { return Err("stop words not aligned".into()); }
        let n = bytes(s.allocation).ok_or_else(|| format!("stop: allocation {} not live", s.allocation))?;
        if s.offset + 8 > n { return Err(format!("stop: 2 words at byte {} exceed allocation {} ({n} bytes)", s.offset, s.allocation)); }
        Ok(R::Zero { ranges: vec![(el(s.allocation, s.offset)?, 2)], label: format!("integer result: RUC land tile stop words ({f})") })
    }

    /// Rule of a non-kernel command.
    pub fn command(&self, ex: &Execution, op: &Operation, base: &std::path::Path,
                   frame: Option<&crate::replay_frame::ReplayFrame>) -> Result<R, String> {
        let range = |s: &plan::Slice, bytes: usize| -> Result<(i64, usize), String> { Ok((self.el(s.allocation, s.offset)?, bytes.div_ceil(4))) };
        Ok(match op {
            Operation::DeviceCopy { destination, source, bytes, .. } => {
                if bytes % 4 != 0 || destination.offset % 4 != 0 || source.offset % 4 != 0 { return Ok(R::Refused("device_copy: unaligned".into())); }
                let (d, s) = (self.el(destination.allocation, destination.offset)?, self.el(source.allocation, source.offset)?);
                let n = bytes / 4;
                if d == s { R::Identity("inactive:device_copy (in place)".into()) }
                else if d < s + n as i64 && s < d + n as i64 { R::Refused("device_copy: overlapping".into()) }
                else { R::Copy { dst: d, src: s, n } }
            },
            Operation::PeerSend { .. } | Operation::PeerRecv { .. } => R::Refused("peer copy: rule is built in the window (rule_inner)".into()),
            Operation::ModuleGlobalCopy { symbol, .. } => R::Identity(format!("inactive:module_global_copy {symbol} (module constant outside every allocation; no shadow)")),
            Operation::Clear { destination, bytes, .. } => R::Zero { ranges: vec![range(destination, *bytes)?], label: "clear".into() },
            Operation::Fill16 { destination, count, .. } => R::Zero { ranges: vec![range(destination, count * 2)?], label: "fill16".into() },
            Operation::Fill32 { destination, count, .. } => R::Zero { ranges: vec![range(destination, count * 4)?], label: "fill32".into() },
            Operation::Upload { destination, data, pointers, origin, .. } => {
                if !pointers.is_empty() { R::Identity("table:pointer upload".into()) } else {
                    let bytes = match data {
                        plan::Blob::Hex { bytes, .. } => bytes.len() / 2,
                        plan::Blob::File { path, .. } => std::fs::metadata(crate::path(base, path)).map_err(|e| e.to_string())?.len() as usize,
                        plan::Blob::Gzip { bytes, .. } => *bytes,
                    };
                    R::Zero { ranges: vec![range(destination, bytes)?], label: format!("upload:{}", origin.as_deref().unwrap_or("?")) }
                }
            },
            Operation::HeldUploadRanges { ranges, origin, .. } => R::Held {
                ranges: ranges.iter().map(|r| range(&r.destination, r.bytes)).collect::<Result<_, _>>()?, origin: origin.clone() },
            Operation::FrameUpload { destination, name, .. } => {
                let n = frame.ok_or("frame upload without frame")?.scalar(name)?.len();
                R::Zero { ranges: vec![range(destination, n)?], label: format!("host-frame constant {name}") }
            },
            Operation::HostUpload { destination, view, order, .. } => {
                let n = ex.host_state.pack(view.allocation, &view.view(), order.order())?.len();
                // A host value uploaded into device state: held by name (the host copy carries no tangent).
                R::Held { ranges: vec![range(destination, n)?], origin: format!("host-upload:{}", view.allocation) }
            },
            Operation::HostPublish { .. } => R::Identity("noop:host_publish (device to host)".into()),
            Operation::PointerTableUpload { rule, .. } => R::Identity(format!("table:{rule}")),
            Operation::TempoPointerTableBind { .. } | Operation::TempoNamedTableBind { .. } => R::Identity("table:tempo bind".into()),
            Operation::TempoTableRelease { .. } => R::Identity("noop:tempo release".into()),
            Operation::SeaicePointerTableUpload { .. } => R::Identity("table:seaice".into()),
            Operation::RucPointerTableUpload { .. } => R::Identity("table:ruc".into()),
            Operation::RrfsCopyPlanUpload { .. } => R::Identity("table:rrfs copy plan".into()),
            Operation::NativeCubReduce { dtype, .. } | Operation::NativeCubSegmentedReduce { dtype, .. } =>
                // CuPy cub dtype codes 0-7 are the integer types (P8 rules.rs command_class).
                if (0..=7).contains(dtype) { R::Identity("inactive:cub integer reduction".into()) }
                else { R::Refused(format!("native cub reduce dtype {dtype}: float reduction has no rule")) },
            // G2 (P1 milestone 2): the LW wrapper's stable argsort of the integer layer counts (rrtmg_lw_v842.py:1018).
            // CuPy Thrust dtype codes 4 and 6 are int32 and int64 (native_thrust::word_bytes); the sorted copy and the
            // size_t indices are integers, so no float word is written. THE BREAKAGE THIS PREVENTS: with radiation live
            // the argsort was Refused (identity both ways) and listed as an unbound radiation command at every LW call.
            Operation::NativeThrustArgsort { dtype, .. } =>
                if matches!(dtype, 4 | 6) { R::Identity("inactive:thrust argsort (integer keys and indices)".into()) }
                else { R::Refused(format!("native thrust argsort dtype {dtype}: float keys have no rule")) },
            Operation::NativeCublasTranspose { input, output, m, n, .. } => {
                let g = ex.library_dimensions(&[*m, *n], i32::MAX as usize)?;
                let (m, n) = (g[0] as i64, g[1] as i64);
                // cublasSgeam(OP_T, OP_T, m, n, 1, A lda=n, 0, A, C ldc=m): C[i + j*m] = A[j + i*n] (P8 rows::cublas_transpose).
                R::Rows { rows: vec![DevRow { dst: self.el(output.allocation, output.offset)?, src: self.el(input.allocation, input.offset)?,
                                              ds: [0, 0, 1, m], ss: [0, 0, n, 1], n: [1, m, n], count: m * n }], label: "cublas transpose" }
            },
            Operation::DeviceToHost { host_consumer, .. } => R::Identity(format!("d2h:{host_consumer}")),
            Operation::Synchronization { .. } | Operation::EventRecord { .. } | Operation::EventWait { .. } => R::Identity("noop:sync".into()),
            Operation::HostBoundary { .. } => R::Identity("noop:host boundary".into()),
            Operation::TempoMainComplete { .. } => R::Identity("noop:tempo main complete".into()),
            Operation::FreeNull { .. } | Operation::Free { .. } => R::Identity("noop:free".into()),
            Operation::Allocate { id, bytes, .. } => R::Zero { ranges: vec![(self.el(*id, 0)?, bytes.div_ceil(4))], label: "fresh allocation".into() },
            Operation::RrfsCopyPlanKernel { table, cache, .. } => {
                let data = ex.host_state.get(cache.allocation)?;
                let (launch, _, _) = crate::copy_plan::decode_cache(data.get(cache.offset..cache.offset + crate::copy_plan::CACHE_BYTES).ok_or("cache")?)?;
                let args = [submit::Argument::Device(submit::Slice { allocation: table.allocation, offset: table.offset })];
                self.copy_plan_rows(ex, &args, launch.grid[1]).unwrap_or_else(|e| R::Refused(format!("rrfs copy plan kernel: {e}")))
            },
            Operation::TransportCopy { .. } => R::Refused("transport_copy op: no rule in this dispatcher".into()),
            // G2 (P1 milestone 2): the held radiation tendency rule. rrfs_radiation_unpack (resident_radiation_rrfs.py:28)
            // moves each radiation call's results into the MPAS pool arrays (rthratenlw/sw, fluxes, radii) that every
            // later step reads until the next radiation call: d[row*a+col*b] = s ? s[row*c+col*f] : v, row = e/w,
            // col = e%w. As P8 copy rows (n1 = n2 = 1, n3 = w, ds = [a, 0, 0, b]) the tangent copies and the reverse
            // moves the pool cotangent accumulated over the carried steps back onto the radiation outputs; fill rows
            // (constant v) are zero. Views are the live host shape/stride words, as the forward's RadiationUnpack.
            // THE BREAKAGE THIS PREVENTS: Refused (identity) claimed d(pool)/d(pool) = 1 for the overwritten tendency
            // and dropped every steps-2..N cotangent of the radiation step.
            Operation::RadiationUnpack { rows, .. } => {
                let words = |view: &plan::RadiationView| -> Result<(i64, i64, [i64; 2], Option<i64>), String> {
                    let shape = ex.host_i64_values(&view.shape)?;
                    let strides = ex.host_i64_values(&view.strides)?;
                    if !(1..=2).contains(&shape.len()) || shape.len() != strides.len() || shape.iter().any(|&n| n < 0)
                        || strides.iter().any(|b| b % 4 != 0) { return Err("radiation unpack: invalid live view".into()); }
                    let n = shape.iter().product::<i64>();
                    let w = *shape.last().unwrap();
                    let st = if shape.len() == 1 { [0, strides[0] / 4] } else { [strides[0] / 4, strides[1] / 4] };
                    let el = match view.target { Some(t) if n > 0 => Some(self.el(t.allocation, t.offset)?), _ => None };
                    Ok((n, w, st, el))
                };
                let mut out = Vec::with_capacity(rows.len());
                for row in rows {
                    let (n, w, ds, dst) = words(&row.dst)?;
                    if n == 0 { continue; }
                    let dst = dst.ok_or("radiation unpack: nonempty view without target")?;
                    let (src, ss) = match &row.src {
                        Some(v) => { let (m, _, ss, el) = words(v)?; if m != n { return Err("radiation unpack: copy shape mismatch".into()); }
                                     (el.ok_or("radiation unpack: source without target")?, ss) },
                        None => (-1, [0, 0]),
                    };
                    out.push(DevRow { dst, src, ds: [ds[0], 0, 0, ds[1]], ss: [ss[0], 0, 0, ss[1]], n: [1, 1, w], count: n });
                }
                R::Rows { rows: out, label: "radiation unpack (held tendency)" }
            },
            Operation::AddInplace { .. } => R::Refused("add_inplace op: no rule in this dispatcher".into()),
            Operation::VinterpOzone { .. } => R::Refused("vinterp_ozone op: no rule in this dispatcher".into()),
            Operation::Kernel { .. } | Operation::HostBranch { .. } | Operation::RkSchedule { .. } | Operation::RkSplits { .. }
            | Operation::RkStages { .. } | Operation::RkSubsteps { .. } => return Err("not a leaf command".into()),
        })
    }
}

/// P8's deterministic transpose of a copy-row launch as a gather (algorithm of tree/ad/phys/p8/src/rows.rs `gather`,
/// copied because that file binds P8's own plan module): CSR over distinct source elements of (dst id << 1 | repeat)
/// items in the serial reference's order (rows last to first, elements ascending), plus the distinct destinations.
pub struct Gather { pub src_word: Vec<i64>, pub start: Vec<i64>, pub item: Vec<i64>, pub dst_word: Vec<i64>,
                    pub sourceless: Vec<u8>, pub injective: bool }
pub fn gather(rows: &[DevRow]) -> Result<Gather, String> {
    let order: Vec<usize> = (0..rows.len()).rev().collect();
    let mut first = vec![0i64; rows.len() + 1];
    for (i, &r) in order.iter().enumerate() { first[i + 1] = first[i] + rows[r].count; }
    let total = first[rows.len()] as usize;
    if total == 0 { return Ok(Gather { src_word: vec![], start: vec![0], item: vec![], dst_word: vec![], sourceless: vec![], injective: true }); }
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16);
    let mut visits = vec![(0i64, 0i64, 0i64); total];
    let chunk = total.div_ceil(threads).max(1);
    std::thread::scope(|sc| {
        for (c, part) in visits.chunks_mut(chunk).enumerate() {
            let (order, first) = (&order, &first);
            sc.spawn(move || {
                let k0 = (c * chunk) as i64;
                let mut i = first.partition_point(|&f| f <= k0) - 1;
                for (j, v) in part.iter_mut().enumerate() {
                    let k = k0 + j as i64;
                    while first[i + 1] <= k { i += 1; }
                    let r = &rows[order[i]];
                    let mut q = k - first[i];
                    let i3 = q % r.n[2]; q /= r.n[2]; let i2 = q % r.n[1]; q /= r.n[1]; let i1 = q % r.n[0]; q /= r.n[0];
                    let d = r.dst + q * r.ds[0] + i1 * r.ds[1] + i2 * r.ds[2] + i3 * r.ds[3];
                    let s = if r.src >= 0 { r.src + q * r.ss[0] + i1 * r.ss[1] + i2 * r.ss[2] + i3 * r.ss[3] } else { -1 };
                    *v = (d, k, s);
                }
            });
        }
    });
    let mut by_dst: Vec<(i64, i64)> = visits.iter().map(|&(d, k, _)| (d, k)).collect();
    par_sort(&mut by_dst, threads);
    let mut dst_word: Vec<i64> = Vec::new(); let mut first_k: Vec<i64> = Vec::new();
    for &(d, k) in &by_dst { if dst_word.last() != Some(&d) { dst_word.push(d); first_k.push(k); } }
    let mut parts: Vec<Vec<(i64, i64, i64)>> = Vec::new();
    std::thread::scope(|sc| {
        let hs: Vec<_> = visits.chunks(chunk).map(|part| { let (dw, fk) = (&dst_word, &first_k); sc.spawn(move || {
            part.iter().filter(|v| v.2 >= 0).map(|&(d, k, s)| {
                let id = dw.binary_search(&d).unwrap() as i64;
                (s, k, (id << 1) | (fk[id as usize] != k) as i64) }).collect::<Vec<_>>() }) }).collect();
        for h in hs { parts.push(h.join().unwrap()); }
    });
    let mut by_src: Vec<(i64, i64, i64)> = parts.concat();
    let mut sourceless = vec![1u8; dst_word.len()];
    for &(_, _, it) in &by_src { sourceless[(it >> 1) as usize] = 0; }
    par_sort(&mut by_src, threads);
    let (mut src_word, mut start, mut item) = (Vec::new(), Vec::new(), Vec::with_capacity(by_src.len()));
    for &(s, _, it) in &by_src {
        if src_word.last() != Some(&s) { src_word.push(s); start.push(item.len() as i64); }
        item.push(it);
    }
    start.push(item.len() as i64);
    if let Some(&s) = src_word.iter().find(|s| dst_word.binary_search(s).is_ok()) {
        return Err(format!("source element {s} is also a destination: transpose must stay serial for this launch"));
    }
    let injective = dst_word.len() == total && src_word.len() == item.len();
    Ok(Gather { src_word, start, item, dst_word, sourceless, injective })
}
/// Sort in `threads` chunks in parallel, then merge pairwise in parallel.
fn par_sort<T: Ord + Copy + Send + Sync>(v: &mut Vec<T>, threads: usize) {
    let n = v.len();
    if n < 1 << 16 || threads < 2 { v.sort_unstable(); return; }
    let chunk = n.div_ceil(threads);
    std::thread::scope(|sc| { for p in v.chunks_mut(chunk) { sc.spawn(move || p.sort_unstable()); } });
    let mut width = chunk;
    while width < n {
        let runs: Vec<&[T]> = v.chunks(width).collect();
        let merged: Vec<Vec<T>> = std::thread::scope(|sc| {
            let hs: Vec<_> = runs.chunks(2).map(|pair| { let pair = pair.to_vec(); sc.spawn(move || {
                if pair.len() == 1 { return pair[0].to_vec(); }
                let (a, b) = (pair[0], pair[1]);
                let (mut i, mut j, mut m) = (0, 0, Vec::with_capacity(a.len() + b.len()));
                while i < a.len() && j < b.len() { if a[i] <= b[j] { m.push(a[i]); i += 1 } else { m.push(b[j]); j += 1 } }
                m.extend_from_slice(&a[i..]); m.extend_from_slice(&b[j..]); m }) }).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        *v = merged.concat(); width *= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gather_of_a_transpose_is_injective_and_a_broadcast_is_not() {
        let t = DevRow { dst: 100, src: 0, ds: [0, 0, 1, 3], ss: [0, 0, 4, 1], n: [1, 3, 4], count: 12 };
        let g = gather(&[t]).unwrap();
        assert!(g.injective);
        assert_eq!(g.dst_word.len(), 12);
        let b = DevRow { dst: 100, src: 0, ds: [0, 0, 4, 1], ss: [0, 0, 0, 1], n: [1, 3, 4], count: 12 };
        let g = gather(&[b]).unwrap();
        assert!(!g.injective);
        assert_eq!(g.src_word, vec![0, 1, 2, 3]);
        assert_eq!(g.start, vec![0, 3, 6, 9, 12]);
    }
    #[test]
    fn microphysics_top_level_is_an_exact_zero_rule() {
        use submit::{Argument, Slice};
        let el = |a: u64, o: usize| if a == 7 { Ok(100 + (o / 4) as i64) } else { Err(format!("allocation {a} has no shadow")) };
        let dev = |a, o| Argument::Device(Slice { allocation: a, offset: o });
        let args = vec![Argument::Bytes(59i32.to_le_bytes().to_vec()), Argument::Bytes(45000f32.to_le_bytes().to_vec()),
                        dev(5, 0), dev(6, 0), dev(7, 8)];
        match Ctx::integer_result("rrfs_init_microphysics_top", &args, el) {
            Some(Ok(R::Zero { ranges, .. })) => assert_eq!(ranges, vec![(102, 1)]),
            other => panic!("rule {other:?}"),
        }
        assert!(Ctx::integer_result("rrfs_init_ocean_temperature", &args, el).is_none());
        assert!(matches!(Ctx::integer_result("rrfs_init_microphysics_top", &args[..4], el), Some(Err(_))));
        let bad = vec![args[0].clone(), args[1].clone(), dev(5, 0), dev(6, 0), dev(9, 0)];
        assert!(matches!(Ctx::integer_result("rrfs_init_microphysics_top", &bad, el), Some(Err(_))));
    }
    #[test]
    fn tbq_init_is_an_exact_constant_table_rule() {
        use submit::{Argument, Slice};
        match Ctx::module_constant("ruc_tbq_init", &[]) {
            Some(Ok(R::Identity(l))) => assert!(l.starts_with("inactive:ruc_tbq_init")),
            other => panic!("rule {other:?}"),
        }
        assert!(Ctx::module_constant("oracle_init_rucinit", &[]).is_none());
        let one = vec![Argument::Device(Slice { allocation: 3, offset: 0 })];
        assert!(matches!(Ctx::module_constant("ruc_tbq_init", &one), Some(Err(_))));
    }
    #[test]
    fn rrtmg_lw_table_setup_is_an_exact_zero_rule() {
        use submit::{Argument, Slice};
        let el = |a: u64, o: usize| match a { 0 => Ok((o / 4) as i64), 1 => Ok(10112 + (o / 4) as i64), 2 => Ok(20224 + (o / 4) as i64),
                                              _ => Err(format!("allocation {a} has no shadow")) };
        let dev = |a, o| Argument::Device(Slice { allocation: a, offset: o });
        let (g, b) = ([79u32, 1, 1], [128u32, 1, 1]);
        let args = vec![dev(0, 0), dev(1, 0), dev(2, 0)];
        match Ctx::constant_tables("mpas_init_tables", &args, g, b, el) {
            Some(Ok(R::Zero { ranges, .. })) => assert_eq!(ranges, vec![(0, 10001), (10112, 10001), (20224, 10001)]),
            other => panic!("rule {other:?}"),
        }
        assert!(Ctx::constant_tables("mpas_init_rwgt", &args, g, b, el).is_none());
        assert!(matches!(Ctx::constant_tables("mpas_init_tables", &args[..2], g, b, el), Some(Err(_))));
        assert!(matches!(Ctx::constant_tables("mpas_init_tables", &args, [78, 1, 1], b, el), Some(Err(_))));
        assert!(matches!(Ctx::constant_tables("mpas_init_tables", &args, g, b, el), Some(Ok(_))));
        let bad = vec![dev(0, 0), dev(1, 0), dev(9, 0)];
        assert!(matches!(Ctx::constant_tables("mpas_init_tables", &bad, g, b, el), Some(Err(_))));
        let unaligned = vec![dev(0, 0), dev(1, 2), dev(2, 0)];
        assert!(matches!(Ctx::constant_tables("mpas_init_tables", &unaligned, g, b, el), Some(Err(_))));
        let scalar = vec![dev(0, 0), Argument::Bytes(vec![0; 8]), dev(2, 0)];
        assert!(matches!(Ctx::constant_tables("mpas_init_tables", &scalar, g, b, el), Some(Err(_))));
    }
    #[test]
    fn par_sort_matches_sort() {
        let mut v: Vec<(i64, i64)> = (0..200_000).map(|i| ((i * 7919) % 100_003, i)).collect();
        let mut w = v.clone(); w.sort_unstable();
        par_sort(&mut v, 8);
        assert_eq!(v, w);
    }
    #[test]
    fn ruc_land_tile_preflight_stop_words_are_an_exact_zero_rule() {
        use submit::{Argument, Slice};
        let el = |a: u64, o: usize| if a == 14 { Ok(500 + (o / 4) as i64) } else { Err(format!("allocation {a} has no shadow")) };
        let bytes = |a: u64| if a == 14 || a == 1 { Some(512) } else { None };
        let dev = |a, o| Argument::Device(Slice { allocation: a, offset: o });
        let args = vec![dev(1, 0), dev(14, 8), Argument::Bytes(2562i32.to_le_bytes().to_vec())];
        match Ctx::integer_flags("ruc_land_tile_preflight", &args, el, bytes) {
            Ok(R::Zero { ranges, .. }) => assert_eq!(ranges, vec![(502, 2)]),
            other => panic!("rule {other:?}"),
        }
        assert!(Ctx::integer_flags("ruc_land_tile_preflight", &args[..2], el, bytes).is_err());
        let past_end = vec![dev(1, 0), dev(14, 508), args[2].clone()];
        assert!(Ctx::integer_flags("ruc_land_tile_preflight", &past_end, el, bytes).is_err());
        let unaligned = vec![dev(1, 0), dev(14, 6), args[2].clone()];
        assert!(Ctx::integer_flags("ruc_land_tile_preflight", &unaligned, el, bytes).is_err());
        let no_shadow = vec![dev(1, 0), dev(1, 0), args[2].clone()];
        assert!(Ctx::integer_flags("ruc_land_tile_preflight", &no_shadow, el, bytes).is_err());
        let by_value_stop = vec![dev(1, 0), Argument::Bytes(vec![0; 8]), args[2].clone()];
        assert!(Ctx::integer_flags("ruc_land_tile_preflight", &by_value_stop, el, bytes).is_err());
    }
}
