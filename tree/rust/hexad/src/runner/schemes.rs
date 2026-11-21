// SPDX-License-Identifier: Apache-2.0
//! L2 bindings (ad/window-L2-bindings): the checked physics derivatives P3 (MYNN-EDMF whole launch), P4 (MYNN
//! surface layer, GSL orographic drag), P6 (RUC sea ice whole launch, oracle_ice_ruc_ice; ad/held-oracle_ice_ruc_ice)
//! surface layer, GSL orographic drag), P7's RRTMG-LW gas optics (rlw_taumol_batched; ad/held-rlw_taumol_batched),
//! P7's RRTMG-SW gas optics (rsw_taumol_b; ad/held-rsw_taumol_b), the RUC land whole launch (oracle_land_ruc_land, kind
//! ruc_land: the ad/held-oracle_land_ruc_land entry bound by ad/step6-rucland) and P8's command rules as plan-bindable rows.
//!
//! The rows are data (`schemes.json`, schema ad.scheme.rows/v1, written at build time by
//! tree/ad/phys/bind/gen_rows.py from the captures, P3's gen-info, P4-ENTRIES.json and P8's rule table): per
//! function the unit, every argument's position, role, dtype and extent expression (evaluated with the live
//! launch's scalars), the tile schedule and the pre-state hand-off. This loader binds a row to a live plan
//! launch: every derivative argument gets a float64 shadow over its exact byte extent (overlapping or touching
//! extents of one allocation share one shadow, so aliasing in the forward is aliasing here; two different
//! arguments over overlapping words are refused by name), every primal argument reads a snapshot taken before
//! the forward launch, and the entries run in place on the shadows (tangent: shadow in = pre tangent, out =
//! post tangent; adjoint: shadow in = post cotangent, out = pre cotangent), the convention of the ad.entry/v1
//! sweeps. P8's rule table (`rules::kernel_class`) is loaded natively and every row is checked against it at
//! load. The per-launch dot (U6's algorithm and fields) runs here for these rows.
//!
//! Width: every native entry keeps float64 derivative storage (the images' derivative arithmetic is double),
//! so the launch dot is graded at the float64 bar (1e-10) unless AD_NATIVE_TOL says otherwise. A composed
//! float32 sweep binds them through a widened temporary, as deriv.rs does for U9's float64 spec arguments.
use crate::cuda::{checked, Cuda};
use crate::dev::{self, Arena, Support};
use crate::expr::{self, Env};
use crate::p8bridge::{rows, rules};
use crate::submit::{Allocation, Argument};
use serde_json::{json, Value};
use std::{collections::{BTreeMap, HashMap, HashSet}, ffi::{c_void, CString}, path::{Path, PathBuf}, ptr};

const W: usize = 8;

// ------------------------------------------------------------------------------------------- device helpers
pub(crate) struct Module { handle: *mut c_void }
impl Module {
    pub(crate) unsafe fn load(cuda: &Cuda, path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut handle = ptr::null_mut();
        // central home-card: a refused bundle image names this card's class beside its path (cuda::load_refusal).
        let status = (cuda.module_load)(&mut handle, bytes.as_ptr().cast());
        if status != 0 { return Err(crate::cuda::load_refusal(&format!("cuModuleLoadData {}", path.display()), status, "", &cuda.card_class())); }
        Ok(Self { handle })
    }
    pub(crate) unsafe fn func(&self, cuda: &Cuda, name: &str) -> Result<*mut c_void, String> {
        let c = CString::new(name).map_err(|e| e.to_string())?;
        let mut f = ptr::null_mut();
        checked(&format!("cuModuleGetFunction {name}"), (cuda.module_function)(&mut f, self.handle, c.as_ptr()))?;
        Ok(f)
    }
    /// lane rung3-params: address and size of a module global.
    unsafe fn global(&self, cuda: &Cuda, name: &str) -> Result<(u64, usize), String> {
        let c = CString::new(name).map_err(|e| e.to_string())?;
        let (mut at, mut size) = (0u64, 0usize);
        checked(&format!("cuModuleGetGlobal {name}"), (cuda.module_global)(&mut at, &mut size, self.handle, c.as_ptr()))?;
        Ok((at, size))
    }
    unsafe fn set_global(&self, cuda: &Cuda, name: &str, bytes: &[u8]) -> Result<(), String> {
        let c = CString::new(name).map_err(|e| e.to_string())?;
        let (mut at, mut size) = (0u64, 0usize);
        checked(&format!("cuModuleGetGlobal {name}"), (cuda.module_global)(&mut at, &mut size, self.handle, c.as_ptr()))?;
        if size != bytes.len() { return Err(format!("module global {name}: {size} bytes, not {}", bytes.len())); }
        dev::upload(cuda, at, bytes)
    }
}

/// A kernel parameter: a device pointer or by-value bytes (kept 8-byte aligned).
#[derive(Clone)]
enum K { P(u64), B(Vec<u8>) }
fn i32b(v: i32) -> K { K::B(v.to_le_bytes().to_vec()) }

/// speed mynn (count cache): one counted MYNN-EDMF reverse launch as a file: 8 u64 (the six words of `head`: magic,
/// columns, lanes, elements per column, tape node size, node bytes; then the largest group tape; then 0), then one
/// i64 count per column. A file of another shape or image layout is not read.
fn p3_cc_read(f: &Path, head: &[u64; 6]) -> Option<(Vec<i64>, usize)> {
    let b = std::fs::read(f).ok()?;
    if b.len() != (8 + head[1] as usize) * 8 { return None; }
    let w = |i: usize| u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
    if (0..6).any(|i| w(i) != head[i]) || w(6) == 0 { return None; }
    Some((b[64..].chunks_exact(8).map(|x| i64::from_le_bytes(x.try_into().unwrap())).collect(), w(6) as usize))
}
/// Written beside and renamed into place, so two runs of one package on one box never read half a file.
fn p3_cc_write(f: &Path, head: &[u64; 6], region: usize, counts: &[i64]) -> Result<(), String> {
    let mut b: Vec<u8> = Vec::with_capacity((8 + counts.len()) * 8);
    for v in head.iter().copied().chain([region as u64, 0u64]) { b.extend_from_slice(&v.to_le_bytes()); }
    for c in counts { b.extend_from_slice(&c.to_le_bytes()); }
    if let Some(d) = f.parent() { std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?; }
    let tmp = f.with_extension(format!("tmp{}", std::process::id()));
    std::fs::write(&tmp, &b).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, f).map_err(|e| format!("{}: {e}", f.display()))
}
fn i64b(v: i64) -> K { K::B(v.to_le_bytes().to_vec()) }

unsafe fn launch(cuda: &Cuda, f: *mut c_void, grid: [u32; 3], block: [u32; 3], args: &[K]) -> Result<(), String> {
    let store: Vec<Vec<u64>> = args.iter().map(|a| match a {
        K::P(p) => vec![*p],
        K::B(b) => {
            let mut v = vec![0u64; b.len().div_ceil(8).max(1)];
            std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), b.len()).copy_from_slice(b);
            v
        },
    }).collect();
    let mut params: Vec<*mut c_void> = store.iter().map(|v| v.as_ptr() as *mut c_void).collect();
    checked("cuLaunchKernel (native entry)", (cuda.launch)(f, grid[0], grid[1], grid[2], block[0], block[1], block[2], 0,
                                                          ptr::null_mut(), params.as_mut_ptr(), ptr::null_mut()))
}
fn grid1(n: usize, b: usize) -> [u32; 3] { [n.div_ceil(b).max(1) as u32, 1, 1] }
/// P4 v2: nodes per column the tangent-time reservation of the P4 tape assumes (ugwp_oro_run measured 9,613-9,624
/// on hrrrv5 x1.2562, mynnsfc_columns 1,150-1,955: tree/ad/phys/p4/evidence/p4-full-sm120.json). Only the reservation:
/// the adjoint sizes every chunk from the counts it measures and grows the block when a launch needs more.
const P4_RESERVE_NODES: usize = 10_240;
/// P4 v2: the column chunks of a taped P4 adjoint (mynnsfc_columns, ugwp_oro_run). counts[c] = tape nodes of
/// column c (count pass); a chunk of k columns with node capacity cap needs 32 x cap x k bytes of tape.
/// Returns (order, chunks), a chunk being (first position, columns, capacity).
/// - sorted (a reverse image with p4_order): positions run over the columns in order of falling count (stable), so a
///   chunk's capacity is its own first column's count: the tape bytes a launch touches follow the columns' own tape
///   lengths, and short columns travel many to a launch (GSL drag: ocean columns record a fraction of a mountain
///   column's 9,6xx nodes).
/// - not sorted (an older image): positions are the columns, every chunk has the launch's largest count (as before).
/// Every chunk has between blk and lim columns (the last one what is left), whatever the budget: a count above
/// budget / (32 x blk) makes the block grow, never an error. Columns are independent, so the grouping changes no word.
fn p4_tape_chunks(counts: &[i32], budget: usize, blk: usize, lim: usize, sorted: bool) -> (Vec<i32>, Vec<(usize, usize, usize)>) {
    let n = counts.len();
    let lim = lim.max(blk);
    let mut chunks = Vec::new();
    if !sorted {
        let cap = counts.iter().copied().max().unwrap_or(0).max(1) as usize;
        let chunk = ((budget / (32 * cap)) / blk * blk).clamp(blk, lim);
        let mut c0 = 0;
        while c0 < n { let k = chunk.min(n - c0); chunks.push((c0, k, cap)); c0 += k; }
        return (Vec::new(), chunks);
    }
    let mut order: Vec<i32> = (0..n as i32).collect();
    order.sort_by_key(|&c| std::cmp::Reverse(counts[c as usize]));
    let mut c0 = 0;
    while c0 < n {
        let cap = counts[order[c0] as usize].max(1) as usize;
        let k = ((budget / (32 * cap)) / blk * blk).clamp(blk, lim).min(n - c0);
        chunks.push((c0, k, cap));
        c0 += k;
    }
    (order, chunks)
}

/// A grow-only device buffer owned by the loader (scheme scratch: tapes, MYNN arena), outside the dot arena.
/// Nothing is ever freed while the plan replays: a freed block is reused by the plan's own later allocations,
/// and an allocation the forward does not fully write then carries this scratch's bytes into the boundary hash
/// (seen: bind1, allocation 395, a 1,024-byte step-1 allocation, differed after a mid-step cuMemFree). A grown
/// buffer keeps its old block (retired) and grows by half again, so growth is rare and bounded.
#[derive(Default)]
struct Grow { at: u64, bytes: usize, retired: Vec<u64>, /** s60-cardmem: bytes of `retired` */ retired_b: usize }
impl Grow {
    unsafe fn need(&mut self, cuda: &Cuda, bytes: usize) -> Result<u64, String> { self.grow(cuda, bytes, bytes / 2) }
    /// F070: as need, without the half-again headroom, for a buffer whose size is fixed per row (the ugwp frame arena).
    unsafe fn need_exact(&mut self, cuda: &Cuda, bytes: usize) -> Result<u64, String> { self.grow(cuda, bytes, 0) }
    fn retire(&mut self) { if self.at != 0 { self.retired.push(self.at); self.retired_b += self.bytes; self.at = 0; } }
    unsafe fn grow(&mut self, cuda: &Cuda, bytes: usize, headroom: usize) -> Result<u64, String> {
        if bytes > self.bytes {
            self.retire();
            let b = (bytes + headroom).div_ceil(1 << 20) << 20;
            checked(&format!("cuMemAlloc scheme scratch {b} bytes"), (cuda.allocate)(&mut self.at, b))?;
            self.bytes = b;
        }
        Ok(self.at)
    }
    /// mynn-tape: as `need`, with the slack chosen by the caller: `half` grows to `bytes` plus 1/2, else plus 1/8. The
    /// packed MYNN tape takes 1/2 at its first allocation when that fits, because a retired block is never freed (see
    /// above): pkg18m2's reverse walks from step 18 (mean 115k nodes per column) to step 1 (170k), and 1/8 steps would
    /// keep three retired slabs.
    unsafe fn need_slack(&mut self, cuda: &Cuda, bytes: usize, half: bool) -> Result<u64, String> {
        if bytes > self.bytes {
            self.retire();
            let b = (bytes + if half { bytes / 2 } else { bytes / 8 }).div_ceil(1 << 20) << 20;
            checked(&format!("cuMemAlloc scheme scratch {b} bytes"), (cuda.allocate)(&mut self.at, b))?;
            self.bytes = b;
        }
        Ok(self.at)
    }
    /// adjoint-speed-2: as `need`, but a grown block is `bytes` plus 1/8 (not 1/2): the reverse's packed MYNN tape
    /// tiles are sized from the card's free memory, and the old half-again slack would allocate past what was measured.
    unsafe fn need_slack8(&mut self, cuda: &Cuda, bytes: usize) -> Result<u64, String> {
        if bytes > self.bytes {
            self.retire();
            let b = (bytes + bytes / 8).div_ceil(1 << 20) << 20;
            checked(&format!("cuMemAlloc scheme scratch {b} bytes"), (cuda.allocate)(&mut self.at, b))?;
            self.bytes = b;
        }
        Ok(self.at)
    }
}

// ------------------------------------------------------------------------------------------- launch arguments
/// Device target of a pointer argument: (allocation, byte offset). Null pointers are None.
fn target(a: &Argument) -> Option<(u64, usize)> {
    match a {
        Argument::Device(s) => Some((s.allocation, s.offset)),
        Argument::Patched { bytes, pointers } if bytes.len() == 8 && pointers.len() == 1 && pointers[0].byte_offset == 0 =>
            Some((pointers[0].target.allocation, pointers[0].target.offset)),
        _ => None,
    }
}
fn scalar_bytes(a: &Argument) -> Option<&[u8]> { if let Argument::Bytes(b) = a { Some(b) } else { None } }
fn decode(dtype: &str, b: &[u8]) -> Option<f64> {
    Some(match (dtype, b.len()) {
        ("<i4", 4) => i32::from_le_bytes(b.try_into().ok()?) as f64,
        ("<u4", 4) => u32::from_le_bytes(b.try_into().ok()?) as f64,
        ("<i8", 8) => i64::from_le_bytes(b.try_into().ok()?) as f64,
        ("<u8", 8) => u64::from_le_bytes(b.try_into().ok()?) as f64,
        ("<f4", 4) => f32::from_le_bytes(b.try_into().ok()?) as f64,
        ("<f8", 8) => f64::from_le_bytes(b.try_into().ok()?),
        _ => return None,
    })
}

/// CuPy CArray by value (data pointer, size, shape[n], strides[n] in bytes).
#[derive(Clone, Debug)]
struct CArr { alloc: u64, off: usize, shape: Vec<i64>, strides: Vec<i64> }
fn carray(a: &Argument) -> Option<CArr> {
    let Argument::Patched { bytes: b, pointers } = a else { return None };
    if pointers.len() != 1 || pointers[0].byte_offset != 0 || b.len() < 16 || (b.len() - 16) % 16 != 0 { return None; }
    let n = (b.len() - 16) / 16;
    let le = |at: usize| i64::from_le_bytes(b[at..at + 8].try_into().unwrap());
    Some(CArr { alloc: pointers[0].target.allocation, off: pointers[0].target.offset,
                shape: (0..n).map(|k| le(16 + 8 * k)).collect(), strides: (0..n).map(|k| le(16 + 8 * (n + k))).collect() })
}
fn broadcasts(c: &CArr) -> bool { c.shape.iter().zip(&c.strides).any(|(&n, &s)| n > 1 && s == 0) }

// ------------------------------------------------------------------------------------------- segments
/// One bound byte range of one allocation. Active segments carry a float64 shadow (one element per forward
/// 4-byte word, the U7/P8 convention) inside one contiguous region; every segment carries a pre-launch snapshot.
#[derive(Clone, Debug)]
pub struct Seg { pub alloc: u64, pub lo: usize, pub hi: usize, pub active: bool, pub shadow: u64, pub pre: u64, pub scratch: u64, pub snap: u64, pub live: u64 }
impl Seg { fn words(&self) -> usize { (self.hi - self.lo).div_ceil(4) } fn shadow_bytes(&self) -> usize { self.words() * W } }

struct Segs { v: Vec<Seg> }
impl Segs {
    /// Merge the requested extents (alloc, lo, hi, active) per allocation: overlapping or touching ranges share one.
    fn build(want: &[(u64, usize, usize, bool)]) -> Self {
        let mut by: BTreeMap<u64, Vec<(usize, usize, bool)>> = BTreeMap::new();
        for &(a, lo, hi, act) in want { if hi > lo { by.entry(a).or_default().push((lo & !3, hi.div_ceil(4) * 4, act)); } }
        let mut v = Vec::new();
        let seg = |a: u64, lo: usize, hi: usize, active: bool| Seg { alloc: a, lo, hi, active, shadow: 0, pre: 0, scratch: 0, snap: 0, live: 0 };
        for (a, mut r) in by {
            r.sort();
            let mut cur: Option<(usize, usize, bool)> = None;
            for (lo, hi, act) in r {
                cur = match cur {
                    Some((l, h, x)) if lo <= h => Some((l, h.max(hi), x || act)),
                    Some((l, h, x)) => { v.push(seg(a, l, h, x)); Some((lo, hi, act)) },
                    None => Some((lo, hi, act)),
                };
            }
            if let Some((l, h, x)) = cur { v.push(seg(a, l, h, x)); }
        }
        Segs { v }
    }
    fn find(&self, a: u64, off: usize) -> Result<&Seg, String> {
        self.v.iter().find(|s| s.alloc == a && off >= s.lo && off < s.hi)
            .ok_or_else(|| format!("allocation {a} offset {off} is outside the bound segments"))
    }
    fn shadow(&self, a: u64, off: usize) -> Result<u64, String> {
        let s = self.find(a, off)?;
        if !s.active { return Err(format!("allocation {a} offset {off} has no derivative shadow")); }
        Ok(s.shadow + ((off - s.lo) / 4 * W) as u64)
    }
    fn snap(&self, a: u64, off: usize) -> Result<u64, String> {
        let s = self.find(a, off)?;
        if s.snap == 0 { return Err(format!("allocation {a} offset {off} has no shadow snapshot")); }
        Ok(s.snap + ((off - s.lo) / 4 * W) as u64)
    }
    fn pre(&self, a: u64, off: usize) -> Result<u64, String> { let s = self.find(a, off)?; Ok(s.pre + (off - s.lo) as u64) }
    /// Live device address of a bound word (the plan's own allocation).
    fn live(&self, a: u64, off: usize) -> Result<u64, String> { let s = self.find(a, off)?; Ok(s.live + (off - s.lo) as u64) }
    fn scratch(&self, a: u64, off: usize) -> Result<u64, String> {
        let s = self.find(a, off)?;
        if s.scratch == 0 { return Err(format!("allocation {a}: no primal scratch")); }
        Ok(s.scratch + (off - s.lo) as u64)
    }
}

// ------------------------------------------------------------------------------------------- rows
#[derive(Clone, Debug)]
pub struct ArgRow { pub name: String, pub pos: usize, pub device: bool, pub dtype: String, pub extent: Option<String>,
                    pub role: String, pub active: bool }
#[derive(Clone, Debug)]
pub struct Row { pub function: String, pub unit: String, pub kind: String, pub args: Vec<ArgRow>, pub raw: Value }
impl Row {
    fn arg(&self, name: &str) -> Result<&ArgRow, String> {
        self.args.iter().find(|a| a.name == name).ok_or_else(|| format!("{}: row has no argument {name}", self.function))
    }
    /// Launch scalars by name, for the extent expressions.
    fn env(&self, args: &[Argument]) -> Env {
        let mut env = Env::new();
        for a in self.args.iter().filter(|a| !a.device) {
            let Some(b) = args.get(a.pos).and_then(scalar_bytes) else { continue };
            if let Some(v) = decode(&a.dtype, b) { env.insert(a.name.clone(), v); }
        }
        env
    }
    /// Refuse a launch whose dimension scalars differ from the values the row's constant extents assume.
    fn check_fixed(&self, env: &Env) -> Result<(), String> {
        for (k, v) in self.raw["fixed"].as_object().into_iter().flatten() {
            let want = v.as_f64().ok_or("fixed value")?;
            match env.get(k) {
                Some(&got) if got == want => {},
                got => return Err(format!("{}: dimension scalar {k} is {got:?} at this launch, the row's extents assume {want}", self.function)),
            }
        }
        Ok(())
    }
    fn extent_bytes(&self, a: &ArgRow, env: &Env) -> Result<usize, String> {
        let e = a.extent.as_deref().ok_or_else(|| format!("{}: argument {} has no extent", self.function, a.name))?;
        let n = expr::eval(e, env).map_err(|x| format!("{}: extent of {} ({e}): {x}", self.function, a.name))?;
        let item = match a.dtype.as_str() { "<f8" | "<i8" | "<u8" => 8, "|b1" | "|u1" => 1, _ => 4 };
        Ok(n as usize * item)
    }
    /// D1b (g2, 2026-11-11, --refuse holds its kind): the float32 words this row's entries write at a launch, as
    /// (allocation, byte offset, words): its active device arguments of role out / inout, from the row's argument
    /// table and the launch alone (nothing is bound). An output that is absent (null) at this launch writes nothing.
    pub(crate) fn written_f32(&self, args: &[Argument]) -> Result<Vec<(u64, usize, usize)>, String> {
        let env = self.env(args);
        self.check_fixed(&env)?;
        let mut out = Vec::new();
        for a in self.args.iter().filter(|a| a.device && a.active && (a.role == "out" || a.role == "inout")) {
            if a.dtype != "<f4" { return Err(format!("{}: written argument {} is {}, not float32", self.function, a.name, a.dtype)); }
            let arg = args.get(a.pos).ok_or_else(|| format!("{}: launch has no argument {} ({})", self.function, a.pos, a.name))?;
            let Some((al, off)) = target(arg) else {
                if matches!(arg, Argument::Bytes(b) if b.iter().all(|&x| x == 0)) { continue; }
                return Err(format!("{}: argument {} is not a device pointer", self.function, a.name));
            };
            out.push((al, off, self.extent_bytes(a, &env)? / 4));
        }
        Ok(out)
    }
}

/// Primal and derivative buffers of one bound launch.
impl Prepared {
    /// memory (AD_TL_BIND_SCRATCH=1): the bytes `bind_transient` takes for a deferred MYNN-EDMF row (the float64
    /// shadow region and the seed snapshot region, each rounded up to the arena's 256-byte take), or None for any
    /// other row and for a row that was bound whole.
    pub fn p3_transient_need(&self) -> Option<usize> {
        match (&self.kind, self.deferred) {
            (Kind::P3, Some((_, true, _))) if self.stripe_ok => match self.stripe_plan() {
                Some((pl, ..)) => Some(2 * (pl.total.max(256).div_ceil(256) * 256)),
                None => self.deferred.map(|(total, ..)| 2 * (total.max(256).div_ceil(256) * 256)),
            },
            (Kind::P3, Some((total, true, _))) => Some(2 * (total.max(256).div_ceil(256) * 256)),
            _ => None,
        }
    }
    /// 60 km tape (AD_P3_STRIPE=1): the banded layout of this deferred MYNN-EDMF launch's two transient regions
    /// (p3_stripe.rs), with (ni, i0, cols) from the launch's own scalars as they stand now (a fused launch has its
    /// fused batch_count by the time it is bound). None, and the whole-mesh layout, unless the launch is one row of
    /// columns (jms = jme, jts = jte = jms), its columns lie in [its, ite], every active segment is whole rows of ni
    /// words with each argument starting on a row (checked at prepare: stripe_ok), and the bands are smaller.
    fn stripe_plan(&self) -> Option<(crate::p3_stripe::Plan, usize, usize, usize)> {
        if !self.stripe_geom || !matches!(self.kind, Kind::P3) { return None; }
        let gi = |k: &str| self.env.get(k).map(|v| *v as i64);
        let (ims, ime, jms, jme, its, ite, jts, jte) = (gi("ims")?, gi("ime")?, gi("jms")?, gi("jme")?, gi("its")?, gi("ite")?, gi("jts")?, gi("jte")?);
        let (bf, cols) = (gi("batch_first")?, gi("batch_count")?);
        if jms != jme || jts != jte || jts != jms || ime < ims || its < ims || bf < 0 || cols <= 0 || bf + cols > ite - its + 1 || ite > ime { return None; }
        let ni = (ime - ims + 1) as usize;
        let i0 = (its - ims + bf) as usize;
        let mut rows = Vec::new();
        for s in self.segs.v.iter().filter(|s| s.active) {
            if s.lo % 4 != 0 || (s.hi - s.lo) % (ni * 4) != 0 { return None; }
            rows.push((s.hi - s.lo) / (ni * 4));
        }
        crate::p3_stripe::plan(ni, i0, cols as usize, &rows).map(|p| (p, ni, i0, cols as usize))
    }
    /// 60 km tape (AD_P3_STRIPE=1): the bytes the reverse's banded bind of this MYNN-EDMF launch takes (both
    /// regions), for a launch the tangent bound whole; None when off or not bandable (the reverse then binds whole too).
    pub fn p3_banded_need(&self) -> Option<usize> {
        self.stripe_plan().map(|(pl, ..)| 2 * (pl.total.max(256).div_ceil(256) * 256))
    }
    /// Active segments (allocation, first byte, end byte, float64 shadow address): one float64 shadow element per
    /// forward 4-byte word from `lo`, the window's float32 shadow convention widened. A banded MYNN-EDMF launch
    /// (AD_P3_STRIPE=1) gives each row's stripe of the launch's columns as a segment: the window moves only those
    /// words between its shadow and the region (the entries read and write no other; the others' round trip through
    /// float64 is exact, so the head's words are the same).
    pub fn active_segments(&self) -> Vec<(u64, usize, usize, u64)> {
        if let Some((ni, i0, cols, _)) = self.stripe {
            return self.segs.v.iter().filter(|s| s.active)
                .flat_map(|s| crate::p3_stripe::stripes(s.lo, s.hi, s.shadow, ni, i0, cols).into_iter().map(move |(lo, hi, d)| (s.alloc, lo, hi, d)))
                .collect();
        }
        self.segs.v.iter().filter(|s| s.active).map(|s| (s.alloc, s.lo, s.hi, s.shadow)).collect()
    }
}
/// 60 km tape: AD_P3_STRIPE=1 bands the MYNN-EDMF transient regions (off by default; read once). Not with
/// AD_P3_FUSE=1: a fused reverse launch has more columns than the tangent sized its scratch for (wider stripes), nor
/// with AD_P3_BENCH_REPLICATE (columns past the batch).
fn p3_stripe_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AD_P3_STRIPE").as_deref() == Ok("1") && std::env::var("AD_P3_FUSE").as_deref() != Ok("1")
        && std::env::var("AD_P3_BENCH_REPLICATE").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1) <= 1)
}
pub struct Prepared {
    pub function: String,
    pub unit: String,
    kind: Kind,
    segs: Segs,
    /// All active shadows, contiguous: [start, start + bytes). The dot reduces over this region in one slot.
    region: (u64, usize),
    args: Vec<Argument>,
    grid: [u32; 3], block: [u32; 3],
    env: Env,
    /// P3: contiguous float32 primal outputs of the entry and a mirror of the forward's post state, compared word for word.
    identity: Option<(u64, u64, usize)>,
    pub notes: Vec<String>,
    /// tape-dedupe-and-pack: the transient buffers are not bound yet (`prepare_deferred`): (shadow region bytes, P3 snapshot
    /// region, P4/P7/RUC primal scratch). `Native::bind_transient` binds them from the reverse scratch before the adjoint.
    deferred: Option<(usize, bool, bool)>,
    /// tape-dedupe-and-pack: bytes of the transient buffers (region, snapshot region, scratch, device tables) this row took,
    /// for the reverse scratch sizing (measured in the tangent pass, where they are still on the tangent tape).
    pub transient_bytes: usize,
    /// speed tape60 (rolling pre-launch copy): (rolling copy base, device run table, rows) to apply, in this order, after
    /// this row's adjoint: they put back the words this launch's bind changed in the rolling copy, so the launch taped
    /// before this one reads its own pre-launch words. Empty for a row that holds its own copies.
    pub undo: Vec<(u64, u64, usize)>,
    /// 60 km tape: a MYNN-EDMF launch under AD_P3_STRIPE=1 whose arguments each start on a row of ni words of their
    /// segment (the banded layout may be planned; `stripe_plan` checks the rest).
    stripe_geom: bool,
    /// 60 km tape: stripe_geom on a deferred bind (the banded layout is used at bind_transient).
    stripe_ok: bool,
    /// 60 km tape: bound banded (ni, i0, cols, snapshot region base); None for the whole-mesh layout.
    stripe: Option<(usize, usize, usize, u64)>,
}

#[derive(Clone)]
enum Kind {
    Inactive(String),
    View { dst: CArr, src: CArr, in_place: bool, rows: Option<RowsGather> },
    ZeroOut { out: CArr },
    Max { x: CArr, out: CArr, c: f32 },
    Where { src: CArr, mask: CArr, dst: CArr },
    Scale { x: CArr, out: CArr, c: f64 },
    P3,
    P4,
    /// ad/held-tempo_from: the TEMPO MPAS -> TEMPO hand-off through its five pointer tables.
    TempoFrom(Box<Tf>),
    P6(Box<P6Bind>),
    /// step6-rucland: oracle_land_ruc_land through its 122-slot pointer table (ad/held-oracle_land_ruc_land entry).
    RucLand(Box<RlBind>),
    P7Taumol,
    P7RswTaumol,
}

/// P6 sea-ice helper scalars per column (P6's RucRow x[72]) and tso / zs levels (nsl).
const P6_NS: usize = 72;
const P6_NZ: usize = 9;

/// A bound oracle_ice_ruc_ice launch. `base[k]`: (allocation, byte offset) of column 0 of helper scalar k (None:
/// the helper has no word in this launch); `first[k]`: the first helper spelling the same word (aliases, e.g.
/// p8w_kms and p8w_1); tso and zs are level-major arrays of nc words per level. `lists` are device address lists
/// (p6_bind.cu): shadow words in compact input order, output order, adjoint-add order, the pre snapshot words in
/// RucRow order and the live post words in the augmented primal's output order.
#[derive(Clone)]
struct P6Bind { nc: usize, cols: Vec<usize>, flags: [i32; 5], base: Vec<Option<(u64, usize)>>, first: Vec<usize>,
                tso: (u64, usize), zs: (u64, usize), lists: [u64; 5] }

/// A bound oracle_land_ruc_land launch (step6-rucland). The entry (tree/ad/phys/held/oracle_land_ruc_land,
/// oracle_land_ruc_land__{tl,ad,sizes}) reads its trajectory through a 122-slot primal pointer table and works in place on
/// float32 tangent / cotangent arrays named by a second table of the same layout.
/// - `fields`: (slot, allocation, byte offset, words per cell) of the 70 float fields, land_R_fields.json order. Every one
///   is a derivative input and output (the entry's in-place convention: untouched words pass through).
/// - `others`: (slot, allocation, byte offset, bytes) of every other non-null slot the entries read (the per-cell ints and
///   scalars, the tile stop pair, ilnb).
/// - The primal table (`ptab`, built after the segments) points every field and every `others` slot at the private primal
///   copy of the pre-launch state (restore before each call), so the entries never read a live word after the forward and
///   never write one; `live_slots` (ptrs[121], the snowtemp_calls diagnostic the entries do not touch) keep their live
///   address; `tables` is the forward's constant table argument (live); `work` is the work arena argument, whose pre copy
///   is the entries' work_pre.
#[derive(Clone)]
struct RlBind { nc: usize, nwork: usize, nslot: usize, fields: Vec<(usize, u64, usize, usize)>, others: Vec<(usize, u64, usize, usize)>,
                live_slots: Vec<(usize, u64)>, tables: u64, work: (u64, usize), ptab: u64,
                /// ruc-land-adjoint-v2: bytes of the tables allocation from `tables` on (AD_RUC_LAND_DUMP only).
                tables_bytes: usize }

#[derive(Clone)]
struct RowsGather { dev_rows: Vec<rows::DevRow>, src_word: Vec<i64>, start: Vec<i64>, item: Vec<i64>, dst_word: Vec<i64>, sourceless: Vec<u8>, injective: bool }

/// P8View (cuda/p8.cu): element base, size, rank, shape, element strides.
#[repr(C)]
#[derive(Clone, Copy)]
struct V { base: i64, size: i64, ndim: i32, pad: i32, shape: [i64; 4], stride: [i64; 4] }
fn vbytes(v: &V) -> K { K::B(unsafe { std::slice::from_raw_parts((v as *const V).cast::<u8>(), std::mem::size_of::<V>()) }.to_vec()) }

include!("p3_direct.rs");
// inner-model (0.6.2): the simplified linear model's boundary layer, off unless AD_IM_BL=fixedk.
include!("inner_model.rs");

struct P3Image { direct: Option<P3DirectImage>, tl: Module, ad: Module, count: Module, f_tl: *mut c_void, f_ad: *mut c_void, f_count: *mut c_void,
                 sz_tl: usize, sz_tape: usize, struct_bytes: usize,
                 /// mynn-tape: tape node bytes (16: compact node, the image reads per-column tape offsets when MwLaunch.layout
                 /// is 1; 24: the old node, capacity layout only).
                 node_bytes: usize,
                 /// mynn-tape: the image's warp tile (MW_TILE: columns per warp and arena interleave width); every P3 launch
                 /// runs this many threads per block. 32 for every image before mynn-tape.
                 tile_w: usize,
                 /// lane rung3-params: the carrier forward (images.p3_parfwd: the captured forward with mw_parf), if any.
                 parfwd: Option<Module>,
                 /// mynn-adjoint-v2: the persistent entries (mynn_whole_tl_p, mynn_whole_ad_p, mynn_whole_count_p), when the
                 /// image has all three.
                 f_p: Option<[*mut c_void; 3]>,
                 /// mynn-adjoint-v2: a scale-probe image (built with -DMW_BENCH_REPLICATE; mw_sizes o[5] = 1) runs launch column c as
                 /// captured column c % batch_count, so AD_P3_BENCH_REPLICATE=R times R x the columns with the same words.
                 bench: bool }
/// frame_tl / frame_rev: bytes per launch thread of the image's frame arena (p4_frame_bytes; ugwp only, G4), 0 if none.
struct P4Image { tl: Module, rev: Module, frame_tl: usize, frame_rev: usize,
                 /// P4 v2: the reverse image has the module globals p4_order / p4_rev_order (launch thread t runs column
                 /// p4_order[col0 + t]), so the binder may hand it the columns sorted by tape length.
                 order: bool }
/// other-schemes: the tape-free adjoint of ugwp_oro_run (p4/cuda/ugwp_direct.cuh, entry ugwp_oro_run__dad64), loaded
/// only when AD_P4_UGWP_DIRECT=<cubin> is set; per_level = its ugwp_dad_frame_per_level (frame bytes per thread and level).
/// It gives the taped reverse's own answer bytes (tools/ugwp_direct_check.cu). Unset: the taped reverse runs as before.
struct P4Dad { _m: Module, f: *mut c_void, per_level: usize }
/// G4: bytes per thread of a module's frame arena (its p4_frame_bytes global), or 0 when the module has none.
unsafe fn p4_frame_bytes(cuda: &Cuda, m: &Module) -> Result<usize, String> {
    let c = CString::new("p4_frame_bytes").map_err(|e| e.to_string())?;
    let (mut at, mut size) = (0u64, 0usize);
    if (cuda.module_global)(&mut at, &mut size, m.handle, c.as_ptr()) != 0 || size != 8 { return Ok(0); }
    Ok(u64::from_le_bytes(dev::download(cuda, at, 8)?[..8].try_into().unwrap()) as usize)
}
/// P6: the unchanged sea-ice derivative image (tree/ad/phys/p6), the bind kernels (bind/cuda/p6_bind.cu) and the
/// Vilka saturation table the forward's module carries (5,001 float32, uploaded once).
struct P6Image { _m: Module, _b: Module, construct: *mut c_void, aug: *mut c_void, tl: *mut c_void, ad: *mut c_void,
                 rows: *mut c_void, gather: *mut c_void, scatter: *mut c_void, zero: *mut c_void, mirror: *mut c_void, table: u64,
                 /// Bytes of one column's augmented-primal frame in the scratch arena (the image's si_frame_bytes, G4).
                 frame: usize }
/// P7: the generated rlw_taumol_batched tangent/adjoint image (tree/ad/phys/p7/gen/gen_taumol.py over the pinned
/// captured forward source 683d7a07; NVRTC --ftz=false --fmad=false --prec-div=true --prec-sqrt=true).
struct P7Image { _m: Module, tl: *mut c_void, ad: *mut c_void, /** adjoint frame bytes per launch thread (p7_frame_bytes, G4) */ frame: usize }
/// G4: threads of one P7 adjoint launch (a grid-stride loop over column x layer): 768 blocks of 64. At 255 registers a
/// 188-SM RTX PRO 6000 keeps 256 threads per SM resident (48,128), so the cap costs no parallelism there, and the frame
/// arena is 49,152 x 15,360 B = 755 MB instead of a 4.29 GiB stack reservation.
const P7_AD_THREADS: usize = 768 * 64;
/// F070: the most threads one ugwp_oro_run derivative launch runs (tangent, adjoint count pass and adjoint sweep): 768
/// blocks of 64, the P7 cap. The G4 frame arena holds one frame per launch thread (p4_frame_bytes: 24,960 B in the
/// tangent), so one launch over every column grew it with the mesh: x1.655362 needed 16.4 GB (24.5 GB with the 1.5x
/// headroom) against the 6.8 GiB stack reservation it replaced. The launches now run in column chunks of at most this
/// many threads, so the arena is at most 49,152 x 24,960 B = 1.23 GB on any mesh. AD_P4_UGWP_THREADS (a positive
/// multiple of 64) sets a smaller cap, so a proof on a small mesh can force several chunks.
const P4_UGWP_THREADS: usize = 768 * 64;
fn p4_ugwp_threads() -> Result<usize, String> {
    match std::env::var("AD_P4_UGWP_THREADS") {
        Err(_) => Ok(P4_UGWP_THREADS),
        Ok(v) => match v.parse::<usize>() {
            Ok(n) if n > 0 && n % 64 == 0 => Ok(n),
            _ => Err(format!("AD_P4_UGWP_THREADS={v}: not a positive multiple of 64")),
        },
    }
}
/// F070: (first column, columns) of each ugwp launch over nc columns, at most lim columns each.
fn p4_ugwp_chunks(nc: usize, lim: usize) -> Vec<(usize, usize)> { (0..nc).step_by(lim.max(1)).map(|c0| (c0, lim.min(nc - c0))).collect() }
/// F070: frame-arena threads for the ugwp launches over nc columns in chunks of at most lim: the tangent's blocks of
/// tl_block threads and the adjoint's blocks of 64, whichever launch is larger.
fn p4_ugwp_frame_threads(nc: usize, lim: usize, tl_block: usize) -> usize {
    let k = nc.min(lim).max(1);
    (k.div_ceil(tl_block.max(1)) * tl_block.max(1)).max(k.div_ceil(64) * 64)
}
/// P7 SW (ad/held-rsw_taumol_b): the generated rsw_taumol_b tangent/adjoint image (tree/ad/phys/p7/gen/gen_rsw_taumol.py
/// over the pinned captured forward source 3c499c7b; NVRTC --ftz=false --fmad=false --prec-div=true --prec-sqrt=true).
struct P7SwImage { _m: Module, tl: *mut c_void, ad: *mut c_void }
/// step6-rucland: the unchanged RUC land derivative entry (ad/held-oracle_land_ruc_land, built by its build.sh from the
/// captured forward source e6fd85f0) and the data-movement kernels of bind/cuda/ruc_land_bind.cu.
struct RlImage { _m: Module, _b: Module, tl: *mut c_void, ad: *mut c_void, sizes: *mut c_void,
                 narrow: *mut c_void, widen: *mut c_void, cmp: *mut c_void, round: *mut c_void,
                 /// ruc-land-adjoint-v2: (oracle_land_ruc_land__ad2, __scan, __order) when the image carries them (exact per-cell tape).
                 ad2: Option<(*mut c_void, *mut c_void, *mut c_void)>,
                 /// ruc-land-adjoint-v2 round 2: oracle_land_ruc_land__caps (null on an image without it: no capped path).
                 caps: *mut c_void,
                 /// ruc-land-adjoint-v2 round 2: the image's tape is lazy (oracle_land_ruc_land__lazy marker): 16-byte entries
                 /// (op, operand words, ids) at pa and 8-byte adjoints at adj, 24 bytes per entry; else pa, pb, (a,b), adj (32).
                 lazy: bool }
/// RUC land AD tape entries per cell to start from: the entry lane's sizing, 1.25 x the measured maximum (5,726 entries,
/// mean 782, five real launches) + 1,024. An overflow is flagged by the entry (ad_scalar.cuh ad_push stops recording), and
/// the binding then doubles the tape and re-runs the launch's adjoint from its unchanged cotangents (AD_RUC_LAND_TAPE overrides).
const RL_TAPE0: usize = 5726 * 5 / 4 + 1024;
const RL_SLOTS: usize = 122;
/// ruc-land-adjoint-v2: threads per block of the v2 adjoint (one cell per thread; the entry's RL2_MAXB is 256).
/// Measured on the dumped pkg18m2 launches (round5/ruc-land-adjoint-v2/RESULT.md); AD_RUC_LAND_BS overrides.
const RL2_BS: usize = 64;
/// ruc-land-adjoint-v2 round 2: the capped path's per-cell room, cap = 2 x previous count + RL2_CAP_ADD (see ruc_land_ad2).
const RL2_CAP_ADD: usize = 64;
/// ruc-land-adjoint-v2 round 2: bytes of the flags block before the redo list (err 16 | ovf 16 | stat 32).
const RL2_FLAGS: usize = 64;
/// ruc-land-adjoint-v2 round 2: tape entries per cell the tangent apply reserves for the adjoint (ruc_land_reserve):
/// pkg18m2's mean is 782 (18 launches) and the per-warp padding 1.01-1.04, so 1,024 covers a launch with room to spare.
const RL2_RESERVE_ENTRIES: usize = 1024;
/// ruc-land-adjoint-v2 round 2: bytes of one ADWork2 block (ad_scalar2.cuh, layout L2).
const RL2_WK: usize = 136;

pub struct Native {
    rows: HashMap<String, Row>,
    p8: Module,
    l2: Module,
    p3: Option<P3Image>,
    p4: HashMap<String, P4Image>,
    /// other-schemes: the tape-free ugwp_oro_run adjoint (AD_P4_UGWP_DIRECT), None unless asked for.
    p4_dad: Option<P4Dad>,
    /// ad/held-tempo_from: images.tempo_from (tempo_from_ad.cu), when the bundle carries the row.
    tf: Option<TfImage>,
    p6: Option<P6Image>,
    p7: Option<P7Image>,
    p7sw: Option<P7SwImage>,
    /// step6-rucland: images.ruc_land + images.ruc_land_bind, and the AD tape entries per cell now in use (doubles on overflow).
    rl: Option<RlImage>,
    rl_cap: usize,
    /// ruc-land-adjoint-v2 round 2: (cells, total entries, largest count, count buffer) of the last RUC land adjoint; its
    /// per-cell counts stay on the card in scratch `ruc_land_cnt` and size the next apply's tape (capped path).
    rl_prev: Option<(usize, usize, usize, u64)>,
    scratch: HashMap<&'static str, Grow>,
    pub tol: f64,
    /// P3 tile (columns per entry launch, a multiple of the 32-column warp tile) and arena factor; P4 tape budget per chunk.
    p3_tile: usize, p3_arena_x: usize, p4_budget: usize,
    /// adjoint-speed: P3 memory budget in bytes (None until the first launch reads the card), and whether AD_P3_TILE pins the old tiling.
    p3_budget: Option<usize>, p3_pinned: bool, p3_ad_budget: Option<usize>,
    /// set while the tape-free route's refused columns are reversed one by one on the taped route (see p3). Such a
    /// launch takes the adjoint budget measured before instead of asking the driver for the card's free memory again.
    p3_fallback: bool,
    /// mynn (arena fit): what the MYNN-EDMF adjoint has measured in this run. ratio = the largest (adjoint pass
    /// arena high-water) / (count pass arena high-water) seen, 0.0 until one adjoint launch was read back; pending =
    /// (thread slots, count high-water) of the last adjoint launch, whose own high-water buffer is read at the next
    /// launch's count-pass read (no extra wait).
    p3_fit_ratio: f64, p3_fit_pending: Option<(usize, usize)>,
    /// mynn (receipt only): the last counted launch's per-column tape counts by launch shape (columns, initflag).
    p3_prev_counts: HashMap<(usize, bool), Vec<i64>>,
    /// mynn (count-free adjoint launches): the largest per-column tape count a counted launch of this shape
    /// (columns, initflag) has measured in this run, and the largest count-pass arena high-water.
    p3_cap_seen: HashMap<(usize, bool), i64>, p3_hw_seen: usize,
    /// mynn (count at tangent time): the window's step for the launch in hand (set by the window before each
    /// tangent and reverse call; -1 = unknown, the adjoint then counts as before), the per-launch tape counts the
    /// tangent pass left on the card, by (step, kpbl allocation, kpbl offset, first column, columns, initflag):
    /// address of (columns + 1) u64 = the counts, then the largest group tape (0 = a key seen twice in one tangent
    /// step, never used), and the keys of the tangent step in hand.
    pub p3_step: i64,
    p3_pre: HashMap<(i64, u64, usize, i64, usize, bool), u64>, p3_pre_seen: (i64, Vec<(i64, u64, usize, i64, usize, bool)>),
    /// speed mynn (count cache, opt-in AD_P3_COUNT_CACHE): the exact tape counts and the largest group tape of each
    /// counted reverse launch, by (step, first column, columns, initflag), and the keys whose kept counts were too small
    /// in this run (counted again, not used again).
    p3_cc: HashMap<(i64, i64, usize, bool), (Vec<i64>, usize)>, p3_cc_bad: HashSet<(i64, i64, usize, bool)>,
    /// a12 bs-mynn (previous-step counts, opt-in AD_P3_COUNT_PREV): the last exact per-column counts of each launch shape
    /// (columns, initflag) in this run and how many launches have used them since.
    p3_pc: HashMap<(usize, bool), (Vec<i64>, usize)>,
    pub loaded: Value,
    /// readbacks-device: when set (the window sets it), the P3 failure codes go to this device ledger, read once per
    /// step by the window, instead of a synchronize and a download after every P3 launch. None (the per-launch dot
    /// tools) keeps the immediate read.
    pub ledger: Option<dev::Ledger>,
    /// AD_NATIVE_ONLY=prefix,prefix: only physics rows whose function starts with one of these are live (the others
    /// stay held as without --native-schemes). Diagnostic for one-scheme-live attribution runs.
    pub only: Option<Vec<String>>,
    /// multicard-merge: MYNN-EDMF column shards on other contexts (AD_MYNN_SHARD_DEVICES), and whether --params set
    /// carrier globals on the primary's P3 modules (the helpers do not carry them, so sharding then refuses).
    p3_shards: Option<P3Shards>,
    p3_par: std::cell::Cell<bool>,
    /// speed tape60 (rolling pre-launch copy): the rolling copy of each (function, allocation, first byte, end byte) span
    /// bound in the tape chunk in hand (cleared when a chunk starts), the support module of the call in hand (null: off),
    /// and [spans bound to a rolling copy, their bytes, bytes of them that had changed and were taped, bytes of first copies].
    roll: HashMap<(String, u64, usize, usize), u64>,
    roll_sup: *const Support,
    /// batch and predict the rolling compares of this prepare (prepare_deferred_roll_w).
    roll_predict: bool,
    roll_stats: [usize; 4],
}

fn env_var(k: &str) -> Option<String> { std::env::var(k).ok().filter(|v| !v.is_empty()) }
/// prec-floor: the boundary-layer row's held OUTPUT arrays (see Native::p3_hold_out), set once when the rows load.
static P3_HOLD: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
/// The held boundary-layer outputs, for the window receipt.
pub fn p3_hold_outputs() -> Vec<String> { P3_HOLD.get().cloned().unwrap_or_default() }
fn p3_hold_load(dir: &Path, table: &Value) -> Result<Vec<String>, String> {
    if let Some(v) = env_var("AD_P3_HOLD_OUT") {
        if v.trim() == "none" { return Ok(Vec::new()); }
        return Ok(v.split(',').map(str::trim).filter(|x| !x.is_empty()).map(str::to_owned).collect());
    }
    let Some(tl) = table["images"]["p3_tl"].as_str() else { return Ok(Vec::new()) };
    let f = dir.join(tl).with_file_name("gen-info.json");
    let Ok(bytes) = std::fs::read(&f) else { return Ok(Vec::new()) };
    let g: Value = serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", f.display()))?;
    match &g["hold_outputs"] {
        Value::Null => Ok(Vec::new()),
        Value::Array(a) => a.iter().map(|x| x.as_str().map(str::to_owned)
            .ok_or_else(|| format!("{}: hold_outputs holds a value that is not a name", f.display()))).collect(),
        _ => Err(format!("{}: hold_outputs is not a list", f.display())),
    }
}

/// (2026-11-13) GiB words for the boundary layer's reserves, `default` when unset (the old fixed values).
/// AD_P3_RESERVE_GIB: what the tangent's budget keeps free (16 GiB). AD_P3_AD_RESERVE_GIB: what the reverse's budget
/// keeps free (12 GiB on a card of 64 GiB or more, an eighth below). AD_P3_ROOM_KEEP_GIB: under AD_P3_ROOM_RULE, the
/// least the rule leaves for the step's other schemes when no cold family names them yet (0). THE BREAKAGE THE LAST
/// PREVENTS: the no-recording 3 km route (a 96 GB test card, tape 44 GiB, 11.1 GB free) gave the boundary layer 6.37 GB
/// at the first reverse launch, since no cold family was on the card yet, and the RUC land adjoint after it stopped
/// with "one warp's adjoint tape (78912 entries) does not fit the card (0 bytes of room)".
fn gib_word(k: &str, default: usize) -> usize {
    env_var(k).and_then(|v| v.parse::<f64>().ok()).filter(|v| v.is_finite() && *v >= 0.0)
        .map(|v| (v * (1u64 << 30) as f64) as usize).unwrap_or(default)
}
/// speed tape60 (rolling pre-launch copy): on unless AD_TAPE_NATIVE_ROLL=0; block bytes (AD_TAPE_NATIVE_ROLL_KIB, 16).
fn roll_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AD_TAPE_NATIVE_ROLL").as_deref() != Ok("0"))
}
fn roll_block() -> usize {
    static B: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *B.get_or_init(|| std::env::var("AD_TAPE_NATIVE_ROLL_KIB").ok().and_then(|v| v.parse::<usize>().ok()).filter(|k| *k > 0).unwrap_or(16) << 10)
}

impl Native {
    /// a12 bs-mynn (P3 fuse, opt-in AD_P3_FUSE=1): whether the MYNN-EDMF launch `lower` (taped just before `upper`) can run
    /// its adjoint together with `upper` as ONE launch over both column ranges. Every argument and scalar but batch_first and
    /// batch_count equal, the same segments, and lower's columns end where upper's start. THE BREAKAGE: the 3 km step runs the
    /// scheme as two launches of 29,216 and 9,062 columns; the small one is one wave of 1,133 of the card's 1,504 slots and
    /// the big one 2.4 waves, so the card idles at both tails (back-2 k12-mb-hit: 200 + 670 ms a step). Columns are
    /// independent (G5) and each launch reads and writes only its own columns, so after upper's undo (lower's own pre-launch
    /// words) one launch over both ranges computes every column as the two did: the bytes are compared on a card, not assumed.
    pub fn p3_fuse_check(&self, lower: &Prepared, upper: &Prepared) -> Result<usize, String> {
        if !matches!(lower.kind, Kind::P3) || !matches!(upper.kind, Kind::P3) || lower.function != upper.function { return Err("not two P3 launches".into()); }
        let row = self.rows.get(&lower.function).ok_or("P3 row missing")?;
        let (pf, pc) = (row.arg("batch_first")?.pos, row.arg("batch_count")?.pos);
        if lower.args.len() != upper.args.len() { return Err("argument counts differ".into()); }
        for (i, (a, b)) in lower.args.iter().zip(&upper.args).enumerate() {
            if i == pf || i == pc { continue; }
            if format!("{a:?}") != format!("{b:?}") { return Err(format!("argument {i} differs")); }
        }
        for (k, v) in &lower.env { if k != "batch_first" && k != "batch_count" && upper.env.get(k) != Some(v) { return Err(format!("scalar {k} differs")); } }
        if lower.env.len() != upper.env.len() { return Err("scalar sets differ".into()); }
        let g = |p: &Prepared, k: &str| p.env.get(k).copied().ok_or_else(|| format!("P3 scalar {k}"));
        let (f0, c0, f1, c1) = (g(lower, "batch_first")?, g(lower, "batch_count")?, g(upper, "batch_first")?, g(upper, "batch_count")?);
        if f0 + c0 != f1 { return Err(format!("columns not contiguous: {f0} + {c0} != {f1}")); }
        if lower.segs.v.len() != upper.segs.v.len() || lower.segs.v.iter().zip(&upper.segs.v).any(|(a, b)| a.alloc != b.alloc || a.lo != b.lo || a.hi != b.hi || a.active != b.active) {
            return Err("segments differ".into());
        }
        match &lower.args[pc] { Argument::Bytes(b) if b.len() == 4 => {}, _ => return Err("batch_count is not a 4-byte scalar".into()) }
        Ok((c0 + c1) as usize)
    }
    /// a12 bs-mynn (P3 fuse): make `lower` cover `n` columns from its first (after p3_fuse_check said yes).
    pub fn p3_fuse_apply(&self, lower: &mut Prepared, n: usize) -> Result<(), String> {
        let row = self.rows.get(&lower.function).ok_or("P3 row missing")?;
        let pc = row.arg("batch_count")?.pos;
        lower.args[pc] = Argument::Bytes((n as i32).to_le_bytes().to_vec());
        lower.env.insert("batch_count".into(), n as f64);
        Ok(())
    }
    /// s60-cardmem (AD_CARD_RELEASE=1; the window calls it after a sync when the reverse starts): free every retired
    /// scheme-scratch block. A grown block keeps its old one (see Grow: a block freed while the plan replays can be
    /// reused by a later plan allocation the forward does not fully write). Once the forward has run, every plan
    /// allocation is live and a replayed Allocate only clears the live block (window exec), so that reuse cannot happen,
    /// and nothing reads a retired block: it is card (or host) memory held for nothing. Returns (blocks, bytes).
    pub unsafe fn release_retired(&mut self, cuda: &Cuda) -> (usize, usize) {
        let (mut n, mut b) = (0usize, 0usize);
        for g in self.scratch.values_mut() {
            for a in g.retired.drain(..) { if (cuda.free)(a) == 0 { n += 1; } }
            b += std::mem::take(&mut g.retired_b);
        }
        self.note_own_ring();
        (n, b)
    }
    /// 60 km tape: tell card_room what the boundary layer's ring of its own holds now, for the reverse growth the room
    /// file keeps (card_room::growth). Only with AD_CARD_ROOM_FILE, the one place that growth is read: without it nothing
    /// is looked up or stored.
    fn note_own_ring(&self) {
        if crate::card_room::file().is_some() { crate::card_room::note_bl_own(self.bl_own_device() as u64); }
    }
    /// 60 km tape: device bytes the boundary layer's ring of its own holds (its nodes and bars blocks, retired ones
    /// included; a block made as managed memory is not counted, as spill::device_now does not count it).
    fn bl_own_device(&self) -> usize {
        let mut a: Vec<u64> = Vec::new();
        for k in ["p3_nodes", "p3_bars"] {
            if let Some(g) = self.scratch.get(k) { if g.at != 0 { a.push(g.at); } a.extend(g.retired.iter().copied()); }
        }
        crate::spill::device_bytes_of(&a)
    }
}

impl Native {
    /// AD_SCHEMES=<dir holding schemes.json and the images it names>.
    pub unsafe fn from_env(cuda: &Cuda, spec_kernels: &HashSet<String>) -> Result<Option<Self>, String> {
        let Some(dir) = env_var("AD_SCHEMES") else { return Ok(None) };
        let dir = PathBuf::from(dir);
        let table: Value = serde_json::from_slice(&std::fs::read(dir.join("schemes.json")).map_err(|e| format!("schemes.json: {e}"))?)
            .map_err(|e| format!("schemes.json: {e}"))?;
        if table["schema"] != "ad.scheme.rows/v1" { return Err("schemes.json is not ad.scheme.rows/v1".into()); }
        let _ = P3_HOLD.set(p3_hold_load(&dir, &table)?);
        let mut rows_map = HashMap::new();
        let mut disagree = Vec::new();
        for r in table["rows"].as_array().ok_or("rows")? {
            let args = r["args"].as_array().into_iter().flatten().map(|a| ArgRow {
                name: a["name"].as_str().unwrap_or("").into(), pos: a["pos"].as_u64().unwrap_or(0) as usize,
                device: a["device"].as_bool().unwrap_or(false), dtype: a["dtype"].as_str().unwrap_or("").into(),
                extent: a["extent"].as_str().map(str::to_owned), role: a["role"].as_str().unwrap_or("").into(),
                active: a["active"].as_bool().unwrap_or(false) }).collect();
            let row = Row { function: r["function"].as_str().ok_or("row function")?.into(), unit: r["unit"].as_str().unwrap_or("").into(),
                            kind: r["kind"].as_str().ok_or("row kind")?.into(), args, raw: r.clone() };
            // P8's rule table is the authority on what each command is: a row must agree with it.
            let (class, binding) = rules::kernel_class(&row.function, spec_kernels);
            let want = r["p8_class"].as_str().unwrap_or("");
            // step6-sw: a by-name row cannot bind cupy_concatenate_same_size (float32 and integer instantiations share the
            // name; the window binds it per module from the recorded source). THE BREAKAGE THIS PREVENTS: the float32 SW
            // radiation input stack bound inactive by a name row, and the agreement check passing "rule" against "rule".
            if row.function == "cupy_concatenate_same_size" {
                disagree.push(format!("{}: a scheme row cannot bind this launch by name ({binding})", row.function));
            }
            if class != want { disagree.push(format!("{}: row says {want}, P8 rules say {class} ({binding})", row.function)); }
            rows_map.insert(row.function.clone(), row);
        }
        if !disagree.is_empty() { return Err(format!("schemes.json disagrees with P8's rule table: {disagree:?}")); }
        let img = |k: &str| dir.join(table["images"][k].as_str().unwrap_or(""));
        let p8 = Module::load(cuda, &img("p8"))?;
        let l2 = Module::load(cuda, &img("l2_rules"))?;
        let p3 = if table["images"]["p3_tl"].is_string() {
            let (tl, ad, count) = (Module::load(cuda, &img("p3_tl"))?, Module::load(cuda, &img("p3_ad"))?, Module::load(cuda, &img("p3_count"))?);
            let sizes = tl.func(cuda, "mw_sizes")?;
            let mut o = 0u64;
            checked("cuMemAlloc", (cuda.allocate)(&mut o, 64))?;
            dev::zero(cuda, o, 64)?; // mynn-tape: images before it write only o[0..4]; o[4] = 0 then means a 32-column tile
            launch(cuda, sizes, [1, 1, 1], [1, 1, 1], &[K::P(o)])?;
            dev::sync(cuda)?;
            let s: Vec<u64> = dev::download(cuda, o, 48)?.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
            let p3_tile_w = if s[4] == 0 { 32 } else { s[4] as usize };
            if ![1usize, 2, 4, 8, 16, 32].contains(&p3_tile_w) { return Err(format!("P3 image warp tile {p3_tile_w} (MW_TILE must divide 32)")); }
            // kept allocated (see Grow: no cuMemFree while the plan replays)
            let p3row = rows_map.get("rrfs_mynnedmf_run_l16").ok_or("P3 image without its row")?;
            let direct = p3_direct_load(cuda, p3row)?;
            let nf = p3row.raw["float_arrays"].as_array().map(|a| a.len()).unwrap_or(0);
            if (s[2] != 24 && s[2] != 16) || s[3] as usize != nf * 48 + 8 * 14 + 16 {
                return Err(format!("P3 image layout: node {} bytes, launch struct {} bytes (row expects {})", s[2], s[3], nf * 48 + 8 * 14 + 16));
            }
            let (f_tl, f_ad, f_count) = (tl.func(cuda, "mynn_whole_tl")?, ad.func(cuda, "mynn_whole_ad")?, count.func(cuda, "mynn_whole_count")?);
            let f_p = match (tl.func(cuda, "mynn_whole_tl_p"), ad.func(cuda, "mynn_whole_ad_p"), count.func(cuda, "mynn_whole_count_p")) {
                (Ok(a), Ok(b), Ok(c)) => Some([a, b, c]), _ => None };
            // G4: the three P3 entries keep a 26.1-26.5 KB per-thread local frame (measured 2026-11-06, about 7.1 GiB for
            // every resident thread of a 188-SM RTX PRO 6000), which the driver reserves at the first launch. Reserved
            // here once with empty launches (a zeroed MwLaunch: col_count 0, every thread returns before reading any
            // other argument), so a card too full for it refuses at load, by name, instead of mid-plan at the first
            // MYNN-EDMF launch (seen: g4u-all18-s1 try 3, step 1 rrfs_mynnedmf_run_l16 native tangent, CUDA status 2).
            // Only when the row can be live: a one-scheme-live leg (AD_NATIVE_ONLY) that leaves MYNN-EDMF held never
            // launches P3 and does not pay it.
            // S10 (2026-11-07): the frame is 3,544 / 3,944 / 3,944 B (tl / ad / count) since the float carrier is trivially
            // copyable (no stack temporaries for by-value operands), carriers hold a one-byte context reference and
            // gen_whole.py --frame inline1 inlines the one-call-site helpers and makes the library tables static: about
            // 1.1 GiB card-wide instead of 7.2 GiB.
            // The reservation at load stays: a card too full for the smaller frame still has to refuse here, not mid-plan.
            let p3_live = env_var("AD_NATIVE_ONLY").map_or(true, |v| v.split(',').map(str::trim).filter(|q| !q.is_empty())
                .any(|q| "rrfs_mynnedmf_run_l16".starts_with(q)));
            if p3_live {
                let mut lz = 0u64;
                checked("cuMemAlloc P3 empty launch", (cuda.allocate)(&mut lz, (s[3] as usize).max(64)))?; // kept (see Grow)
                dev::zero(cuda, lz, (s[3] as usize).max(64))?;
                let mut a = vec![K::P(lz)];
                a.extend((0..96).map(|_| i64b(0))); // every scalar is 4 or 8 bytes; the driver reads only the kernel's own
                // mynn-adjoint-v2: the persistent entries reserve their frame too (mw_sched -> a zeroed counter, so the
                // empty launch takes group 0 of none and returns).
                let mut ents = vec![(f_tl, "mynn_whole_tl"), (f_ad, "mynn_whole_ad"), (f_count, "mynn_whole_count")];
                if let Some([a, b, c]) = f_p {
                    let mut z = 0u64;
                    checked("cuMemAlloc P3 sched", (cuda.allocate)(&mut z, 64))?; // kept (see Grow)
                    dev::zero(cuda, z, 64)?;
                    for m in [&tl, &ad, &count] { m.set_global(cuda, "mw_sched", &z.to_le_bytes())?; }
                    ents.extend([(a, "mynn_whole_tl_p"), (b, "mynn_whole_ad_p"), (c, "mynn_whole_count_p")]);
                }
                for (f, n) in ents {
                    launch(cuda, f, [1, 1, 1], [32, 1, 1], &a).and_then(|_| dev::sync(cuda)).map_err(|e| format!(
                        "P3 {n}: local-memory reservation at load failed ({e}); it needs its local frame (about 3.9 KB since S10) for every resident thread of the card"))?;
                }
            }
            let parfwd = if table["images"]["p3_parfwd"].is_string() { Some(Module::load(cuda, &img("p3_parfwd"))?) } else { None };
            Some(P3Image { direct, tl, ad, count, f_tl, f_ad, f_count, sz_tl: s[0] as usize, sz_tape: s[1] as usize, struct_bytes: s[3] as usize, node_bytes: s[2] as usize, tile_w: p3_tile_w, parfwd, f_p, bench: s[5] == 1 })
        } else { None };
        let mut p4 = HashMap::new();
        for fam in ["mynnsfc", "ugwp"] {
            let (t, r) = (format!("p4_{fam}_tl"), format!("p4_{fam}_rev"));
            if !table["images"][&t].is_string() { continue; }
            let (tl, rev) = (Module::load(cuda, &img(&t))?, Module::load(cuda, &img(&r))?);
            let (frame_tl, frame_rev) = (p4_frame_bytes(cuda, &tl)?, p4_frame_bytes(cuda, &rev)?);
            if fam == "ugwp" && (frame_tl == 0 || frame_rev == 0) {
                return Err("P4 ugwp images carry no p4_frame_bytes: rebuild them with p4/gen/p4_frame.py (G4 frame arena)".into()); }
            // F070: the binder launches the ugwp tangent in column chunks (col0, ncols); an image built before that reads
            // every chunk as columns 0..ncols (wrong tangents once a mesh has more than P4_UGWP_THREADS columns).
            if fam == "ugwp" {
                let c = CString::new("p4_tl_chunks").map_err(|e| e.to_string())?;
                let (mut at, mut size) = (0u64, 0usize);
                if (cuda.module_global)(&mut at, &mut size, tl.handle, c.as_ptr()) != 0 || size != 8 {
                    return Err("P4 ugwp tangent image has no p4_tl_chunks: it predates the (col0, ncols) tangent entries (F070); rebuild it with p4/build.sh".into()); }
            }
            let order = {
                let c = CString::new("p4_rev_order").map_err(|e| e.to_string())?;
                let (mut at, mut size) = (0u64, 0usize);
                (cuda.module_global)(&mut at, &mut size, rev.handle, c.as_ptr()) == 0 && size == 8
            };
            let m = P4Image { tl, rev, frame_tl, frame_rev, order };
            if fam == "mynnsfc" {
                // The derivative modules build their own constant psi tables once (P4-ENTRIES: mynnsfc_tables).
                for md in [&m.tl, &m.rev] {
                    let f = md.func(cuda, "mynnsfc_tables")?;
                    let mut e = 0u64;
                    checked("cuMemAlloc", (cuda.allocate)(&mut e, 16))?;
                    dev::zero(cuda, e, 16)?;
                    launch(cuda, f, [1, 1, 1], [1, 1, 1], &[K::P(e)])?;
                    dev::sync(cuda)?;
                    let v = dev::download(cuda, e, 4)?;
                    if v != [0, 0, 0, 0] { return Err("mynnsfc_tables in a derivative module reported an error".into()); }
                }
            }
            p4.insert(fam.to_string(), m);
        }
        let p4_dad = match env_var("AD_P4_UGWP_DIRECT") {
            Some(path) => {
                let m = Module::load(cuda, Path::new(&path))?;
                let f = m.func(cuda, "ugwp_oro_run__dad64")?;
                let c = CString::new("ugwp_dad_frame_per_level").map_err(|e| e.to_string())?;
                let (mut at, mut size) = (0u64, 0usize);
                checked("cuModuleGetGlobal ugwp_dad_frame_per_level", (cuda.module_global)(&mut at, &mut size, m.handle, c.as_ptr()))?;
                if size != 8 { return Err("AD_P4_UGWP_DIRECT: ugwp_dad_frame_per_level is not 8 bytes".into()); }
                let per_level = u64::from_le_bytes(dev::download(cuda, at, 8)?[..8].try_into().unwrap()) as usize;
                if !p4.contains_key("ugwp") || !p4["ugwp"].order {
                    return Err("AD_P4_UGWP_DIRECT needs the bundle's ugwp images with a column order (p4_rev_order): its fallback columns run on the taped reverse".into()); }
                eprintln!("{}", json!({"event":"ad_schemes_ugwp_direct","image":path,"frame_per_level":per_level}));
                Some(P4Dad { _m: m, f, per_level })
            }
            None => None,
        };
        let tf = if table["images"]["tempo_from"].is_string() {
            let m = Module::load(cuda, &img("tempo_from"))?;
            let (primal, tl, ad, ad_zgrid) = (m.func(cuda, "tempo_from_primal")?, m.func(cuda, "tempo_from_tl")?, m.func(cuda, "tempo_from_ad")?, m.func(cuda, "tempo_from_ad_zgrid")?);
            eprintln!("{}", json!({"event":"ad_schemes_tempo_from","image":img("tempo_from")}));
            Some(TfImage { _m: m, primal, tl, ad, ad_zgrid })
        } else { None };
        let only: Option<Vec<String>> = env_var("AD_NATIVE_ONLY").map(|v| v.split(',').map(|s| s.trim().to_owned()).filter(|s| !s.is_empty()).collect());
        // P6 is loaded (and its local-memory reservation paid) only when its row can be live: not on a one-scheme-live
        // leg (AD_NATIVE_ONLY) that leaves the sea ice held.
        let p6_live = only.as_ref().map_or(true, |o| o.iter().any(|p| "oracle_ice_ruc_ice".starts_with(p.as_str())));
        let p6 = if p6_live && table["images"]["p6"].is_string() {
            let (m, b) = (Module::load(cuda, &img("p6"))?, Module::load(cuda, &img("p6_bind"))?);
            let t = std::fs::read(img("p6_table")).map_err(|e| format!("P6 table: {e}"))?;
            if t.len() != 5001 * 4 { return Err(format!("P6 Vilka table: {} bytes, not 20,004", t.len())); }
            let mut at = 0u64;
            checked("cuMemAlloc P6 table", (cuda.allocate)(&mut at, t.len()))?; // kept (see Grow)
            dev::upload(cuda, at, &t)?;
            // The augmented primal's fragment tapes live in a per-column frame of the P6 scratch arena (G4, 2026-11-06),
            // sized from the image's si_frame_bytes. Before, they sat on the thread stack: a 49,744-byte local frame that
            // the driver reserved for every resident thread of the card (13.1 GiB on a 188-SM RTX PRO 6000, measured).
            // What is left on the stack (about 2 KB per thread, about 0.6 GiB per card) is still reserved here once with
            // an empty launch (n = 0: every thread returns at once), so a full card refuses at load, by name.
            let frame = {
                let c = CString::new("si_frame_bytes").map_err(|e| e.to_string())?;
                let (mut at, mut size) = (0u64, 0usize);
                checked("cuModuleGetGlobal si_frame_bytes", (cuda.module_global)(&mut at, &mut size, m.handle, c.as_ptr()))?;
                if size != 8 { return Err(format!("P6 image si_frame_bytes: {size} bytes, not 8")); }
                let f = u64::from_le_bytes(dev::download(cuda, at, 8)?[..8].try_into().unwrap()) as usize;
                if f == 0 || f % 16 != 0 { return Err(format!("P6 image si_frame_bytes {f}: not a positive multiple of 16")); }
                f
            };
            let aug = m.func(cuda, "seaice_ice_ruc_ice__aug")?;
            launch(cuda, aug, [1, 1, 1], [32, 1, 1], &[K::P(0), K::P(0), i32b(0), i32b(P6_NZ as i32), K::P(0), K::P(0), K::P(0), K::P(0), K::P(0)])
                .and_then(|_| dev::sync(cuda))
                .map_err(|e| format!("P6 augmented primal: local-memory reservation at load failed ({e}); it needs its local frame (about 2 KB) for every resident thread of the card"))?;
            Some(P6Image { construct: m.func(cuda, "p6_construct")?, aug,
                           tl: m.func(cuda, "seaice_ice_ruc_ice__tl")?, ad: m.func(cuda, "seaice_ice_ruc_ice__ad")?,
                           rows: b.func(cuda, "p6b_rows")?, gather: b.func(cuda, "p6b_gather")?, scatter: b.func(cuda, "p6b_scatter")?,
                           zero: b.func(cuda, "p6b_zero")?, mirror: b.func(cuda, "p6b_mirror")?, table: at, frame, _m: m, _b: b })
        } else { None };
        // P7 (ad/held-rlw_taumol_batched): RRTMG-LW gas optics, one launch per layer group (hexcore/mpasphys/rrtmg_lw_v842.py:795).
        // Its adjoint keeps a per-thread tape (P7_TMAX 512 entries) and adjoint vector in local memory, and its tangent carries
        // (value, tangent) pairs; the driver reserves each kernel's local frame for every resident thread at the first launch.
        // Both are reserved here once with an empty launch (ncol = 0: every thread returns before any read), so a card too full
        // for the frames refuses at load, by name, not mid-plan (seen: tempo-ruc-rad-entries tre-caps2, taumol failed at
        // context create with 4 GB free on the card).
        let p7 = if table["images"]["p7_taumol"].is_string() {
            let m = Module::load(cuda, &img("p7_taumol"))?;
            let (tl, ad) = (m.func(cuda, "rlw_taumol_batched__tl")?, m.func(cuda, "rlw_taumol_batched__ad")?);
            let mut a = vec![i32b(0), i32b(0)];
            a.extend((0..5).map(|_| K::P(0)));
            a.push(K::B(0f32.to_le_bytes().to_vec()));
            a.push(K::P(0));
            let mut t = a.clone();
            t.extend((0..6).map(|_| K::P(0)));
            let mut r = a;
            r.extend((0..5).map(|_| K::P(0)));
            launch(cuda, tl, [1, 16, 1], [128, 1, 1], &t).and_then(|_| launch(cuda, ad, [1, 1, 1], [64, 1, 1], &r)).and_then(|_| dev::sync(cuda))
                .map_err(|e| format!("P7 rlw_taumol_batched: local-memory reservation at load failed ({e}); the tangent and adjoint need their local frames for every resident thread of the card"))?;
            let frame = {
                let c = CString::new("p7_frame_bytes").map_err(|e| e.to_string())?;
                let (mut at, mut size) = (0u64, 0usize);
                checked("cuModuleGetGlobal p7_frame_bytes", (cuda.module_global)(&mut at, &mut size, m.handle, c.as_ptr()))?;
                if size != 8 { return Err(format!("P7 image p7_frame_bytes: {size} bytes, not 8")); }
                u64::from_le_bytes(dev::download(cuda, at, 8)?[..8].try_into().unwrap()) as usize
            };
            Some(P7Image { _m: m, tl, ad, frame })
        } else { None };
        // P7 SW (ad/held-rsw_taumol_b): RRTMG-SW gas optics (hexcore/mpasphys/rrtmg_sw_v842.py:1712). Its adjoint keeps a
        // per-thread tape (P7SW_TMAX 96 entries, reset per g-point) and adjoint vector in local memory; both frames are
        // reserved here once with an empty launch (ncol = 0: every thread returns before any read), as for the LW image.
        let p7sw = if table["images"]["p7_rsw_taumol"].is_string() {
            let m = Module::load(cuda, &img("p7_rsw_taumol"))?;
            let (tl, ad) = (m.func(cuda, "rsw_taumol_b__tl")?, m.func(cuda, "rsw_taumol_b__ad")?);
            let mut t = vec![i32b(0), i32b(0)];
            t.extend((0..40).map(|_| K::P(0)));
            let mut r = vec![i32b(0), i32b(0)];
            r.extend((0..39).map(|_| K::P(0)));
            launch(cuda, tl, [1, 1, 1], [128, 1, 1], &t).and_then(|_| launch(cuda, ad, [1, 1, 1], [64, 1, 1], &r)).and_then(|_| dev::sync(cuda))
                .map_err(|e| format!("P7 rsw_taumol_b: local-memory reservation at load failed ({e}); the tangent and adjoint need their local frames for every resident thread of the card"))?;
            Some(P7SwImage { _m: m, tl, ad })
        } else { None };
        // step6-rucland: RUC land (oracle_land_ruc_land). Loaded only when its row can be live (not on an AD_NATIVE_ONLY leg
        // that leaves it held). The entry module carries its own copy of the forward's module table ruc_tbq_table, built once
        // here by the module's own ruc_tbq_init (the plan's ruc_tbq_init writes the forward module's copy only). The tangent's
        // and adjoint's local frames are reserved once with empty launches (nc = 0; an adjoint chunk of W = 0 cells), so a
        // card too full for them refuses at load, by name.
        let rl_live = only.as_ref().map_or(true, |o| o.iter().any(|p| "oracle_land_ruc_land".starts_with(p.as_str())));
        let rl = if rl_live && table["images"]["ruc_land"].is_string() {
            let (m, b) = (Module::load(cuda, &img("ruc_land"))?, Module::load(cuda, &img("ruc_land_bind"))?);
            launch(cuda, m.func(cuda, "ruc_tbq_init")?, [1, 1, 1], [1, 1, 1], &[]).and_then(|_| dev::sync(cuda))
                .map_err(|e| format!("RUC land entry ruc_tbq_init: {e}"))?;
            // A v2 image (oracle_land_ruc_land.ad2.cu) carries __ad2 in place of v1's __ad (the two tape layouts share the AR
            // operators, so one image holds one of them); `ad` is then null and only the v2 adjoint runs.
            let (tl, sizes) = (m.func(cuda, "oracle_land_ruc_land__tl")?, m.func(cuda, "oracle_land_ruc_land__sizes")?);
            let ad = m.func(cuda, "oracle_land_ruc_land__ad").unwrap_or(ptr::null_mut());
            // ruc-land-adjoint-v2: an image built from oracle_land_ruc_land.ad2.cu carries __ad2 and __scan; an older image
            // (5ce95b45) does not, and keeps the v1 adjoint. AD_RUC_LAND_V1=1 keeps v1 on a v2 image (A/B only).
            // round 2 (layout L2): __wscan gives per-warp offsets; an image without it (10d34907) is not bound by this binder.
            let ad2 = match (m.func(cuda, "oracle_land_ruc_land__ad2"), m.func(cuda, "oracle_land_ruc_land__wscan"), m.func(cuda, "oracle_land_ruc_land__order")) {
                (Ok(a), Ok(sc), Ok(o)) if env_var("AD_RUC_LAND_V1").as_deref() != Some("1") => Some((a, sc, o)),
                _ => None,
            };
            if ad.is_null() && ad2.is_none() { return Err(format!("RUC land entry {}: no v1 adjoint (__ad) and the v2 adjoint is off or missing (AD_RUC_LAND_V1=1 needs a v1 image)", img("ruc_land").display())); }
            let mut wz = 0u64;
            checked("cuMemAlloc RUC land empty launch", (cuda.allocate)(&mut wz, 256))?; // kept (see Grow)
            dev::zero(cuda, wz, 256)?;
            let z = || K::P(0);
            launch(cuda, tl, [1, 1, 1], [32, 1, 1], &[z(), z(), z(), z(), i32b(0), z(), z(), z(), i32b(0), z()])
                .and_then(|_| if ad.is_null() { Ok(()) } else { launch(cuda, ad, [1, 1, 1], [32, 1, 1], &[z(), z(), z(), z(), i32b(0), z(), z(), z(), i32b(0), K::P(wz)]) })
                .and_then(|_| match ad2 { Some((a2, _, _)) => launch(cuda, a2, [1, 1, 1], [RL2_BS as u32, 1, 1],
                    &[z(), z(), z(), z(), i32b(0), z(), z(), z(), i32b(0), K::P(wz)]), None => Ok(()) })
                .and_then(|_| dev::sync(cuda))
                .map_err(|e| format!("RUC land entry: local-memory reservation at load failed ({e}); the tangent and adjoint need their local frames for every resident thread of the card"))?;
            eprintln!("{}", json!({"event":"ad_schemes_ruc_land","image":img("ruc_land"),"bind":img("ruc_land_bind"),"adjoint":if ad2.is_some() { "v2" } else { "v1" }}));
            Some(RlImage { tl, ad, sizes, narrow: b.func(cuda, "rlb_narrow")?, widen: b.func(cuda, "rlb_widen")?, cmp: b.func(cuda, "rlb_cmp")?, round: b.func(cuda, "rlb_round")?,
                           caps: if ad2.is_some() { m.func(cuda, "oracle_land_ruc_land__caps").unwrap_or(ptr::null_mut()) } else { ptr::null_mut() },
                           lazy: ad2.is_some() && m.func(cuda, "oracle_land_ruc_land__lazy").is_ok(),
                           _m: m, _b: b, ad2 })
        } else { None };
        let rl_cap = env_var("AD_RUC_LAND_TAPE").and_then(|t| t.parse().ok()).unwrap_or(RL_TAPE0);
        let tol = env_var("AD_NATIVE_TOL").and_then(|t| t.parse().ok()).unwrap_or(1e-10);
        // G5 (ad/phys-window-g5): the bundle table asks for 128-column tiles; a 128-column MYNN-EDMF tile is 4 blocks of 32 threads,
        // so the 21 serial tiles of a 2562-column launch kept the card idle while the host waited (reverse step 2: 17.7 of
        // 30.8 s in 9 launches). Columns are independent: tiles of 128, 640 and 2592 give the same reverse bit for bit
        // (2-step window, result_sha256 5b9317c0 all three). 384 columns cost about 16 MB per column more scratch than 128
        // (2592: 65 GB peak), which keeps an all-live 18-step leg near 30 GB, three per card. AD_P3_TILE still overrides.
        let p3_tile = env_var("AD_P3_TILE").and_then(|t| t.parse().ok()).unwrap_or((table["tiles"]["p3_columns"].as_u64().unwrap_or(128) as usize).max(384));
        if p3_tile == 0 || p3_tile % 32 != 0 { return Err("AD_P3_TILE must be a positive multiple of 32 (MYNN warp tiles)".into()); }
        let loaded = json!({"event":"ad_schemes_loaded","rows":rows_map.len(),"dir":dir,"p3":p3.is_some(),"p4":p4.keys().collect::<Vec<_>>(),"p6":p6.is_some(),"p7":p7.is_some(),"p7sw":p7sw.is_some(),"ruc_land":rl.is_some(),
                            "tol":tol,"p3_tile":p3_tile,"only":only});
        eprintln!("{loaded}");
        let p3_pinned = env_var("AD_P3_TILE").is_some();

        // AD_SCHEME_PRESIZE=name=bytes,... asks for the grow-only scratch buffers up front (the same request the
        // first launch makes, so the same 1.5x block), before the window's trajectory: a long leg on a shared card otherwise
        // meets "cuMemAlloc scheme scratch" at its first adjoint after other lanes filled the card (fC: p3_bars 6.54 GB
        // at reverse step 72, card 1). Unset: nothing is allocated early (old behaviour).
        let mut me = Self { rows: rows_map, p8, l2, p3, p4, p4_dad, tf, p6, p7, p7sw, rl, rl_cap, rl_prev: None, scratch: HashMap::new(), tol, p3_tile, p3_budget: None, p3_pinned, p3_ad_budget: None, p3_fallback: false, p3_fit_ratio: 0.0, p3_fit_pending: None, p3_prev_counts: HashMap::new(), p3_cap_seen: HashMap::new(), p3_hw_seen: 0, p3_step: -1, p3_pre: HashMap::new(), p3_pre_seen: (-1, Vec::new()), p3_cc: HashMap::new(), p3_cc_bad: HashSet::new(), p3_pc: HashMap::new(), ledger: None,
                       p3_arena_x: table["tiles"]["p3_arena_x"].as_u64().unwrap_or(12) as usize,
                       p4_budget: table["tiles"]["p4_tape_budget_bytes"].as_u64().unwrap_or(2 << 30) as usize, loaded, only,
                       p3_shards: None, p3_par: std::cell::Cell::new(false), roll: HashMap::new(), roll_sup: ptr::null(), roll_predict: false, roll_stats: [0; 4] };
        if let Some(i) = me.p3.as_ref() {
            me.p3_shards = P3Shards::from_env(cuda, [img("p3_tl"), img("p3_ad"), img("p3_count")],
                                              [i.sz_tl, i.sz_tape, i.node_bytes, i.tile_w, i.struct_bytes])?;
        } else if std::env::var_os("AD_MYNN_SHARD_DEVICES").is_some() {
            return Err("DEFECT AD_MYNN_SHARD_DEVICES requires the P3 (MYNN-EDMF) image".into());
        }
        if let Ok(spec) = std::env::var("AD_SCHEME_PRESIZE") {
            for item in spec.split(',').filter(|x| !x.trim().is_empty()) {
                let (k, v) = item.split_once('=').ok_or_else(|| format!("AD_SCHEME_PRESIZE item {item}: name=bytes"))?;
                let b: usize = v.trim().parse().map_err(|_| format!("AD_SCHEME_PRESIZE {item}: bytes"))?;
                let key: &'static str = Box::leak(k.trim().to_owned().into_boxed_str());
                me.scratch.entry(key).or_default().need(cuda, b)?;
            }
        }
        Ok(Some(me))
    }

    pub fn row(&self, function: &str) -> Option<&Row> { self.rows.get(function) }
    /// D1b (g2, 2026-11-11): the float32 words a P7 gas-optics row (rlw_taumol_batched, rsw_taumol_b) writes at
    /// this launch (Row::written_f32), for the hold of a kind named with --refuse (wrules::hold_refused_native). None:
    /// the kind is not such a row. These two kinds have no kernel-set row and no rule in wrules.rs; live they run
    /// here as native:P7, and this row is the one place that says what they write.
    pub fn refused_written(&self, function: &str, args: &[Argument]) -> Option<Result<Vec<(u64, usize, usize)>, String>> {
        let row = self.rows.get(function)?;
        if row.kind != "p7_taumol" && row.kind != "p7_rsw_taumol" { return None; }
        Some(row.written_f32(args))
    }

    // ------------------------------------------------------------------- lane rung3-params: parameter carrier
    /// Upload the carrier of P4 family `fam` (gen_params.py): p4_par_on / p4_par_val in both modules (the verbatim
    /// forward and the derivative copy read p4_par_on[k] ? p4_par_val[k] : literal), p4_par_t (TL tangent) in TL.
    pub unsafe fn par_upload(&self, cuda: &Cuda, fam: &str, on: &[i32; 8], val: &[f32; 8], t: &[f64; 8]) -> Result<(), String> {
        if fam == "mynnedmf" {
            self.p3_par.set(true);
            let img = self.p3.as_ref().ok_or("--params: P3 image not loaded")?;
            let ob: Vec<u8> = on.iter().flat_map(|x| x.to_le_bytes()).collect();
            let vb: Vec<u8> = val.iter().flat_map(|x| x.to_le_bytes()).collect();
            let tb: Vec<u8> = t.iter().flat_map(|x| x.to_le_bytes()).collect();
            for md in [&img.tl, &img.ad, &img.count] {
                md.set_global(cuda, "mw_par_on", &ob)?; md.set_global(cuda, "mw_par_val", &vb)?; md.set_global(cuda, "mw_par_t", &tb)?;
            }
            let fw = img.parfwd.as_ref().ok_or("--params mynnedmf: bundle has no images.p3_parfwd")?;
            fw.set_global(cuda, "mw_par_on", &ob)?; fw.set_global(cuda, "mw_par_val", &vb)?;
            return dev::sync(cuda);
        }
        let m = self.p4.get(fam).ok_or_else(|| format!("--params: P4 {fam} image not loaded"))?;
        let ob: Vec<u8> = on.iter().flat_map(|x| x.to_le_bytes()).collect();
        let vb: Vec<u8> = val.iter().flat_map(|x| x.to_le_bytes()).collect();
        let tb: Vec<u8> = t.iter().flat_map(|x| x.to_le_bytes()).collect();
        for md in [&m.tl, &m.rev] { md.set_global(cuda, "p4_par_on", &ob)?; md.set_global(cuda, "p4_par_val", &vb)?; }
        m.tl.set_global(cuda, "p4_par_t", &tb)?;
        dev::sync(cuda)
    }
    pub unsafe fn par_zero_bar(&self, cuda: &Cuda, fam: &str) -> Result<(), String> {
        if fam == "mynnedmf" {
            let img = self.p3.as_ref().ok_or("--params: P3 image not loaded")?;
            let (at, size) = img.ad.global(cuda, "mw_par_bar")?;
            dev::zero(cuda, at, size)?;
            img.ad.set_global(cuda, "mw_par_cols_overflow", &[0u8; 4])?;
            return dev::sync(cuda);
        }
        let m = self.p4.get(fam).ok_or_else(|| format!("--params: P4 {fam} image not loaded"))?;
        let (at, size) = m.rev.global(cuda, "p4_par_bar")?;
        dev::zero(cuda, at, size)?;
        m.rev.set_global(cuda, "p4_par_cols_overflow", &[0u8; 4])?;
        dev::sync(cuda)
    }
    /// The per-column parameter cotangents p4_par_bar[slot * 4096 + column] (float64), as accumulated by the REV
    /// sweeps since the last par_zero_bar.
    pub unsafe fn par_bar(&self, cuda: &Cuda, fam: &str) -> Result<Vec<f64>, String> {
        if fam == "mynnedmf" {
            let img = self.p3.as_ref().ok_or("--params: P3 image not loaded")?;
            dev::sync(cuda)?;
            let (o, _) = img.ad.global(cuda, "mw_par_cols_overflow")?;
            if dev::download(cuda, o, 4)? != [0, 0, 0, 0] { return Err("mynnedmf: parameter carrier has fewer columns than the launch".into()); }
            let (at, size) = img.ad.global(cuda, "mw_par_bar")?;
            return Ok(dev::download(cuda, at, size)?.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect());
        }
        let m = self.p4.get(fam).ok_or_else(|| format!("--params: P4 {fam} image not loaded"))?;
        dev::sync(cuda)?;
        let (o, _) = m.rev.global(cuda, "p4_par_cols_overflow")?;
        if dev::download(cuda, o, 4)? != [0, 0, 0, 0] { return Err(format!("{fam}: parameter carrier has fewer columns than the launch")); }
        let (at, size) = m.rev.global(cuda, "p4_par_bar")?;
        Ok(dev::download(cuda, at, size)?.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect())
    }
    /// The carrier image's verbatim forward function (same name and signature as the plan's).
    pub unsafe fn par_forward(&self, cuda: &Cuda, fam: &str, name: &str) -> Result<*mut c_void, String> {
        if fam == "mynnedmf" {
            return self.p3.as_ref().and_then(|i| i.parfwd.as_ref()).ok_or("--params mynnedmf: bundle has no images.p3_parfwd")?.func(cuda, name);
        }
        self.p4.get(fam).ok_or_else(|| format!("--params: P4 {fam} image not loaded"))?.tl.func(cuda, name)
    }
    /// A P3 / P4 physics row (the window binds these through the float32-shadow adapter; P8 command rows it rules itself).
    pub fn physics_row(&self, function: &str) -> bool {
        self.rows.get(function).is_some_and(|r| r.kind == "p3_mynn" || r.kind.starts_with("p4_") || r.kind == "tempo_from" || r.kind == "p6_seaice" || r.kind == "ruc_land" || r.kind == "p7_taumol" || r.kind == "p7_rsw_taumol")
            && self.only.as_ref().map_or(true, |o| o.iter().any(|p| function.starts_with(p.as_str())))
    }
    pub fn rows(&self) -> impl Iterator<Item = &Row> { self.rows.values() }
    pub fn scratch_bytes(&self) -> usize { self.scratch.values().map(|g| g.bytes).sum() }
    /// any-mesh mem-ckpt receipt: each scheme's grow-only device buffer by key (bytes held at the window end), largest first.
    pub fn scratch_by_key(&self) -> Vec<(String, usize)> {
        let mut v: Vec<(String, usize)> = self.scratch.iter().map(|(k, g)| (k.to_string(), g.bytes)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1)); v
    }

    /// Bind a row to the live launch (before the forward runs): segments, shadows, pre snapshots.
    pub unsafe fn prepare(&mut self, cuda: &Cuda, arena: &mut Arena, live: &HashMap<u64, Allocation>, function: &str,
                          args: &[Argument], grid: [u32; 3], block: [u32; 3]) -> Result<Prepared, String> {
        self.prepare_impl(cuda, arena, live, function, args, grid, block, false, true)
    }
    /// Bind without draining the context after the queued snapshot copies and table uploads.
    /// The caller must establish that the forward and derivative commands use the same legacy null stream as
    /// this module's launch, dev::copy, dev::zero and dev::upload helpers, and keep the arena allocation alive
    /// until those commands finish. Reusing a bump-arena span is safe only through commands queued after its
    /// previous consumers. Host reads, cross-stream consumers and arena frees still require their own waits.
    /// The window's ordered(ex) check establishes this precondition; other prepare callers retain the wait.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn prepare_ordered(&mut self, cuda: &Cuda, arena: &mut Arena, live: &HashMap<u64, Allocation>, function: &str,
                                  args: &[Argument], grid: [u32; 3], block: [u32; 3]) -> Result<Prepared, String> {
        self.prepare_impl(cuda, arena, live, function, args, grid, block, false, false)
    }
    /// tape-dedupe-and-pack: `prepare` for a reverse tape. Only the pre-launch primal snapshots are taken from `arena`; the
    /// float64 shadow region (zeroed), the P3 seed snapshot region, the P4/P7/RUC private primal scratch and the device
    /// tables that hold their addresses are bound later from the reverse scratch (`bind_transient`), right before the
    /// row's adjoint. THE BREAKAGE: those buffers are written only at walk-back (xfer, restore, the adjoint itself) but sat
    /// on the reverse tape for the whole segment: 3-5x a row's primal span (pkg18m2: 0.45 GB of every step's 2.1 GB tape,
    /// MYNN-EDMF 180 MB, 67-88 % zero words). Words: the same buffers with the same contents (the region is zeroed at
    /// bind as at prepare; the rest is written before it is read), only their addresses differ.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn prepare_deferred(&mut self, cuda: &Cuda, arena: &mut Arena, live: &HashMap<u64, Allocation>, function: &str,
                          args: &[Argument], grid: [u32; 3], block: [u32; 3]) -> Result<Prepared, String> {
        self.prepare_impl(cuda, arena, live, function, args, grid, block, true, true)
    }
    /// memory (AD_TL_BIND_SCRATCH=1): `prepare_deferred` without the context drain, under the same precondition as
    /// `prepare_ordered` (the window's ordered(ex) check).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn prepare_deferred_ordered(&mut self, cuda: &Cuda, arena: &mut Arena, live: &HashMap<u64, Allocation>, function: &str,
                          args: &[Argument], grid: [u32; 3], block: [u32; 3]) -> Result<Prepared, String> {
        self.prepare_impl(cuda, arena, live, function, args, grid, block, true, false)
    }
    /// speed tape60 (rolling pre-launch copy; AD_TAPE_NATIVE_ROLL=0 turns it off): `prepare_deferred` for a reverse tape
    /// chunk, where a MYNN-EDMF launch shares one copy of each argument span with the other launches of its chunk.
    /// THE BREAKAGE THIS REMOVES: the scheme runs in column batches (6 launches a step at x1.163842), every launch binds
    /// the same 58 whole-mesh level arrays and 26 surface arrays, and each launch copied all of them onto the reverse
    /// tape again (2.30 GB a launch, 13.79 GB of a 105 GB step tape that goes out to host memory and back), although
    /// between two launches only the words the batch before wrote differ. Here the first launch of a chunk takes the
    /// whole copy. A later launch compares the live words with that copy block by block (AD_TAPE_NATIVE_ROLL_KIB, 16 KiB;
    /// one device compare, word for word), tapes the OLD words of the blocks that differ, and writes the live words
    /// into the copy: the copy then holds this launch's pre-launch words exactly, over the whole span. The walk-back runs
    /// the launches last first; after a launch's adjoint its old words are put back (`Prepared::undo`), so the launch
    /// before it reads its own pre-launch words exactly. The entries only read the copy (P3 slots: x is read, y is null
    /// or a private buffer; kpbl and error_flags are copied to private scratch), so nothing else changes it. Every
    /// tape take is made before the copy is touched: a launch that does not fit the chunk leaves the copy as it was.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn prepare_deferred_roll(&mut self, cuda: &Cuda, arena: &mut Arena, live: &HashMap<u64, Allocation>, function: &str,
                          args: &[Argument], grid: [u32; 3], block: [u32; 3], sup: &Support) -> Result<Prepared, String> {
        self.prepare_deferred_roll_w(cuda, arena, live, function, args, grid, block, sup, true, false)
    }
    /// prepare_deferred_roll with the closing drain optional (`wait` false only in an ordered plan) and the rolling
    /// compares of the whole launch batched and predicted (`predict`, see deriv::predict::roll_on).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn prepare_deferred_roll_w(&mut self, cuda: &Cuda, arena: &mut Arena, live: &HashMap<u64, Allocation>, function: &str,
                          args: &[Argument], grid: [u32; 3], block: [u32; 3], sup: &Support, wait: bool, predict: bool) -> Result<Prepared, String> {
        self.roll_sup = if roll_on() { sup as *const Support } else { ptr::null() };
        self.roll_predict = predict;
        let r = self.prepare_impl(cuda, arena, live, function, args, grid, block, true, wait);
        self.roll_sup = ptr::null();
        self.roll_predict = false;
        r
    }
    /// speed tape60: a new tape chunk starts; the rolling copies of the last one are not on this chunk's tape.
    pub fn roll_clear(&mut self) { self.roll.clear(); }
    /// speed tape60: the counters since the last call (see `roll_stats`), then zeroed.
    pub fn roll_take(&mut self) -> [usize; 4] { std::mem::take(&mut self.roll_stats) }
    #[allow(clippy::too_many_arguments)]
    unsafe fn prepare_impl(&mut self, cuda: &Cuda, arena: &mut Arena, live: &HashMap<u64, Allocation>, function: &str,
                          args: &[Argument], grid: [u32; 3], block: [u32; 3], defer: bool, wait: bool) -> Result<Prepared, String> {
        let u0 = arena.used;
        let row = self.rows.get(function).ok_or_else(|| format!("no row for {function}"))?.clone();
        let env = row.env(args);
        let mut want: Vec<(u64, usize, usize, bool)> = Vec::new();
        // 60 km tape: where each MYNN-EDMF argument starts (the banded layout needs each on a row of its segment)
        let mut p3_offs: Vec<(u64, usize)> = Vec::new();
        let mut notes = Vec::new();
        let items = crate::p8bridge::plan::cupy_items(function);
        let arrays: Vec<CArr> = args.iter().filter_map(carray).collect();
        let span = |c: &CArr, item: i64| -> (usize, usize) {
            let (lo, hi) = crate::p8bridge::plan::carray_span(&c.shape, &c.strides, item);
            ((c.off as i64 + lo).max(0) as usize, (c.off as i64 + hi).max(0) as usize)
        };
        let mut kind = match row.kind.as_str() {
            "inactive" => Kind::Inactive(row.raw["why"].as_str().unwrap_or("inactive").into()),
            "p8_view" | "p8_zero_out" | "p8_max" | "p8_where" | "p8_scale" => {
                let need = if row.kind == "p8_where" { 3 } else { 2 };
                if arrays.len() != need { return Err(format!("{function}: {} CArray arguments, row expects {need}", arrays.len())); }
                if items.len() != arrays.len() { return Err(format!("{function}: element sizes {items:?} do not match {} arrays", arrays.len())); }
                for (c, &it) in arrays.iter().zip(&items) {
                    if c.shape.len() > 4 { return Err(format!("{function}: CArray rank {} above 4", c.shape.len())); }
                    if it >= 4 && c.strides.iter().any(|s| s % 4 != 0) { return Err(format!("{function}: stride not a multiple of 4 bytes")); }
                }
                match row.kind.as_str() {
                    "p8_view" => {
                        let (src, dst) = (arrays[0].clone(), arrays[1].clone());
                        let (s, d) = (span(&src, items[0]), span(&dst, items[1]));
                        want.push((src.alloc, s.0, s.1, true));
                        want.push((dst.alloc, d.0, d.1, true));
                        let in_place = src.alloc == dst.alloc && src.off == dst.off && src.strides == dst.strides && src.shape == dst.shape;
                        if in_place { notes.push("in place: identity".into()); }
                        Kind::View { dst, src, in_place, rows: None }
                    },
                    "p8_zero_out" => {
                        let out = arrays[1].clone();
                        let d = span(&out, items[1]);
                        want.push((out.alloc, d.0, d.1, true));
                        Kind::ZeroOut { out }
                    },
                    "p8_max" => {
                        let c = match args.first().and_then(scalar_bytes) {
                            Some(b) if b.len() == 4 => f32::from_le_bytes(b.try_into().unwrap()),
                            Some(b) if b.len() == 8 => f64::from_le_bytes(b.try_into().unwrap()) as f32,
                            _ => return Err(format!("{function}: constant operand missing")) };
                        let (x, out) = (arrays[0].clone(), arrays[1].clone());
                        if broadcasts(&x) || broadcasts(&out) { return Err(format!("{function}: broadcast operand (refused: the elementwise adjoint would race)")); }
                        let (s, d) = (span(&x, items[0]), span(&out, items[1]));
                        want.push((x.alloc, s.0, s.1, true));
                        want.push((out.alloc, d.0, d.1, true));
                        Kind::Max { x, out, c }
                    },
                    "p8_where" => {
                        let (src, mask, dst) = (arrays[0].clone(), arrays[1].clone(), arrays[2].clone());
                        if broadcasts(&src) || broadcasts(&dst) { return Err(format!("{function}: broadcast operand (refused: the elementwise adjoint would race)")); }
                        let (s, d, m) = (span(&src, items[0]), span(&dst, items[2]), span(&mask, items[1]));
                        want.push((src.alloc, s.0, s.1, true));
                        want.push((dst.alloc, d.0, d.1, true));
                        want.push((mask.alloc, m.0, m.1, false));
                        Kind::Where { src, mask, dst }
                    },
                    _ => {
                        let c = match args.get(1).and_then(scalar_bytes) {
                            Some(b) if b.len() == 4 => f32::from_le_bytes(b.try_into().unwrap()) as f64,
                            Some(b) if b.len() == 8 => f64::from_le_bytes(b.try_into().unwrap()),
                            _ => return Err(format!("{function}: scalar operand missing")) };
                        let (x, out) = (arrays[0].clone(), arrays[1].clone());
                        if broadcasts(&x) || broadcasts(&out) { return Err(format!("{function}: broadcast operand (refused: the elementwise adjoint would race)")); }
                        let (s, d) = (span(&x, items[0]), span(&out, items[1]));
                        want.push((x.alloc, s.0, s.1, true));
                        want.push((out.alloc, d.0, d.1, true));
                        Kind::Scale { x, out, c }
                    },
                }
            },
            k if k == "p3_mynn" || k.starts_with("p4_") || k == "p7_taumol" || k == "p7_rsw_taumol" => {
                row.check_fixed(&env)?;
                let mut views: Vec<(String, u64, usize, usize)> = Vec::new();
                for a in row.args.iter().filter(|a| a.device) {
                    let arg = args.get(a.pos).ok_or_else(|| format!("{function}: launch has no argument {} ({})", a.pos, a.name))?;
                    let Some((al, off)) = target(arg) else {
                        if matches!(arg, Argument::Bytes(b) if b.iter().all(|&x| x == 0)) { continue; } // absent (null) in the forward
                        // step6-radiation: a constant table the plan binds as a module global of the forward's own image
                        // (pkg18m2 rlw_taumol_batched chi_mls = mpas_static_chi; the captures had it in an allocation). The
                        // plan passes its live replay address by value; the entries only read it, so it needs no view,
                        // copy or shadow. Breakage without it: every radiation-live window refused at step 1.
                        if a.role == "const" && !a.active && matches!(arg, Argument::Bytes(b) if b.len() == 8) { continue; }
                        return Err(format!("{function}: argument {} is not a device pointer", a.name));
                    };
                    if a.role == "scratch" { continue; } // forward workspace: no derivative, never read by the entries
                    let n = row.extent_bytes(a, &env)?;
                    let la = live.get(&al).ok_or_else(|| format!("{function}: {} in unknown allocation {al}", a.name))?;
                    if off + n > la.bytes { return Err(format!("{function}: {} extent {n} bytes at {off} exceeds allocation {al} ({} bytes)", a.name, la.bytes)); }
                    want.push((al, off, off + n, a.active));
                    views.push((a.name.clone(), al, off, off + n));
                    p3_offs.push((al, off));
                }
                // Two different arguments over overlapping words: the separate-buffer entries would be wrong. Refused by name.
                for i in 0..views.len() {
                    for j in i + 1..views.len() {
                        let (a, b) = (&views[i], &views[j]);
                        if a.1 == b.1 && a.2 < b.3 && b.2 < a.3 { return Err(format!("{function}: arguments {} and {} overlap in allocation {}", a.0, b.0, a.1)); }
                    }
                }
                if k == "p3_mynn" { Kind::P3 } else if k == "p7_taumol" { Kind::P7Taumol } else if k == "p7_rsw_taumol" { Kind::P7RswTaumol } else { Kind::P4 }
            },
            "tempo_from" => {
                if self.tf.is_none() { return Err(format!("{function}: tempo_from row without its image (schemes.json images.tempo_from)")); }
                let tf = tf_bind(cuda, live, function, args)?;
                for v in &tf.views { want.push((v.alloc, v.off, v.off + v.bytes, true)); }
                notes.push(format!("tempo_from: {} input and {} output views through its 5 pointer tables (aero {}, hail {})",
                    tf.views.iter().filter(|v| !v.output).count(), tf.views.iter().filter(|v| v.output).count(), tf.aero, tf.hail));
                Kind::TempoFrom(Box::new(tf))
            },
            "p6_seaice" => Kind::P6(Box::new(Self::p6_bind(cuda, &row, live, args, &env, &mut want)?)),
            "ruc_land" => Kind::RucLand(Box::new(self.rl_bind(cuda, &row, live, args, &env, &mut want)?)),
            k => return Err(format!("{function}: unknown row kind {k}")),
        };
        let mut segs = Segs::build(&want);
        // P4, P7 and RUC land run on private primal copies of the pre state (restore), never on the live plan words.
        // P4 v2: the P4 derivative entries never write a primal word (P4-ENTRIES: the tangent entries never write
        // forward outputs, the adjoint entries recompute each column from the pre state), so they read the pre snapshot
        // itself: no private copy and no restore. THE BREAKAGE: every P4 call copied its whole primal span (ugwp_oro_run:
        // 22 float planes per level, 2.1 GB at x1.163842) into the reverse scratch, and ugwp_oro_run's adjoint copied it
        // again for every column chunk: 24 chunks x 2.1 GB per reverse step at 60 km, through host-resident scratch when
        // the card is full (a test machine final-60km-8c: 10.6-11.0 s per reverse step for a 151 ms row at 120 km).
        let p4 = row.kind == "p7_taumol" || row.kind == "p7_rsw_taumol" || row.kind == "ruc_land";
        let p3 = row.kind == "p3_mynn";
        // Active shadows contiguous (one reduction region); then the snapshots.
        let total: usize = segs.v.iter().filter(|s| s.active).map(|s| s.shadow_bytes().div_ceil(256) * 256).sum();
        let (region, snap_region) = if defer { (0, 0) } else {
            let region = arena.take(total.max(256))?;
            dev::zero(cuda, region, total.max(256))?;
            (region, if p3 { arena.take(total.max(256))? } else { 0 })
        };
        let mut at = 0u64;
        let mut pre_bytes = 0usize;
        // speed tape60 (rolling pre-launch copy, see prepare_deferred_roll): P3 on a reverse tape chunk only.
        let roll_sup: Option<&Support> = if defer && p3 && !self.roll_sup.is_null() { Some(&*self.roll_sup) } else { None };
        let mut roll_new: Vec<((String, u64, usize, usize), u64)> = Vec::new();
        // (rolling copy, saved old words, save table, update table, undo table, span bytes, changed bytes)
        let mut roll_jobs: Vec<(u64, u64, Vec<u64>, Vec<u64>, Vec<u64>, usize, usize)> = Vec::new();
        // (roll_predict): every segment's block compares below, queued as one list (same pairs, same order) before
        // the loop and answered once (predicted from the book, or one real readback). Exact: the loop writes neither the
        // live words nor a rolling copy before its last compare (new copies go to fresh tape; the runs copies come after).
        let mut roll_pre: Option<(Vec<bool>, Vec<usize>)> = None;
        if let Some(sup) = roll_sup.filter(|_| self.roll_predict) {
            let bs = roll_block();
            let mut all: Vec<(u64, u64, usize)> = Vec::new();
            let mut starts = vec![usize::MAX; segs.v.len()];
            for (si, s) in segs.v.iter().enumerate() {
                let la = live.get(&s.alloc).ok_or_else(|| format!("unknown allocation {}", s.alloc))?;
                let hi = s.hi.min(la.bytes);
                if hi < s.lo { continue; }
                let n = hi - s.lo;
                if !(n % 4 == 0 && s.lo % 4 == 0 && n >= (1 << 20)) { continue; }
                let Some(&r) = self.roll.get(&(function.to_owned(), s.alloc, s.lo, hi)) else { continue };
                let lv = la.address + s.lo as u64;
                starts[si] = all.len();
                let mut o = s.lo;
                while o < hi { let e = ((o / bs + 1) * bs).min(hi); all.push((lv + (o - s.lo) as u64, r + (o - s.lo) as u64, e - o)); o = e; }
            }
            if !all.is_empty() && all.len() <= sup.pairs_per() {
                let tp = dev::prof::start();
                sup.pairs_differ_queue(cuda, &all)?;
                let d = crate::deriv::predict::roll_pairs(cuda, sup, all.len())?;
                dev::prof::add(15, tp);
                roll_pre = Some((d, starts));
            }
        }
        for (si, s) in segs.v.iter_mut().enumerate() {
            let la = live.get(&s.alloc).ok_or_else(|| format!("unknown allocation {}", s.alloc))?;
            if s.hi > la.bytes { s.hi = la.bytes; }
            s.live = la.address + s.lo as u64;
            if s.active && !defer {
                s.shadow = region + at;
                if p3 { s.snap = snap_region + at; }
                at += (s.shadow_bytes().div_ceil(256) * 256) as u64;
            }
            let u = arena.used;
            let n = s.hi - s.lo;
            if let Some(sup) = roll_sup.filter(|_| n % 4 == 0 && s.lo % 4 == 0 && n >= (1 << 20)) {
                let key = (function.to_owned(), s.alloc, s.lo, s.hi);
                if let Some(&r) = self.roll.get(&key) {
                    // Blocks aligned to the allocation's offsets; the live words against the rolling copy, word for word.
                    let bs = roll_block();
                    let mut pairs: Vec<(u64, u64, usize)> = Vec::with_capacity(n / bs + 2);
                    let mut o = s.lo;
                    while o < s.hi { let e = ((o / bs + 1) * bs).min(s.hi); pairs.push((s.live + (o - s.lo) as u64, r + (o - s.lo) as u64, e - o)); o = e; }
                    let differ = match &roll_pre {
                        Some((d, st)) if st[si] != usize::MAX && st[si] + pairs.len() <= d.len() => d[st[si]..st[si] + pairs.len()].to_vec(),
                        _ => sup.pairs_differ(cuda, &pairs)?,
                    };
                    let mut runs: Vec<(usize, usize)> = Vec::new();
                    let mut off = 0usize;
                    for (p, d) in pairs.iter().zip(differ) {
                        if d { match runs.last_mut() { Some(x) if x.0 + x.1 == off => x.1 += p.2, _ => runs.push((off, p.2)) } }
                        off += p.2;
                    }
                    let changed: usize = runs.iter().map(|x| x.1).sum();
                    // Only tape takes here (a take that does not fit ends the chunk before this launch, with the copy untouched).
                    let saved = if changed > 0 { arena.take(changed)? } else { 0 };
                    let (mut sv, mut up, mut un) = (Vec::new(), Vec::new(), Vec::new());
                    let mut so = 0u64;
                    for &(ro, rl) in &runs {
                        sv.push((so, r + ro as u64, rl));
                        up.push((ro as u64, s.live + ro as u64, rl));
                        un.push((ro as u64, saved + so, rl));
                        so += rl as u64;
                    }
                    roll_jobs.push((r, saved, dev::run_rows(&sv), dev::run_rows(&up), dev::run_rows(&un), n, changed));
                    s.pre = r;
                    pre_bytes += arena.used - u;
                    continue;
                }
                s.pre = arena.take(n)?;
                pre_bytes += arena.used - u;
                dev::copy(cuda, s.pre, s.live, n)?;
                roll_new.push((key, s.pre));
                continue;
            }
            s.pre = arena.take(n)?;
            pre_bytes += arena.used - u;
            dev::copy(cuda, s.pre, la.address + s.lo as u64, s.hi - s.lo)?;
            if p4 && !defer { s.scratch = arena.take(s.hi - s.lo)?; }
        }
        // speed tape60: the three tables of every changed span, taken before any rolling copy is written.
        let mut undo: Vec<(u64, u64, usize)> = Vec::new();
        if !roll_jobs.is_empty() || !roll_new.is_empty() {
            let u = arena.used;
            let mut tabs: Vec<[u64; 3]> = Vec::with_capacity(roll_jobs.len());
            for j in &roll_jobs {
                tabs.push(if j.6 > 0 { [arena.take(j.2.len() * 8)?, arena.take(j.3.len() * 8)?, arena.take(j.4.len() * 8)?] } else { [0; 3] });
            }
            pre_bytes += arena.used - u;
            let sup = roll_sup.ok_or("DEFECT rolling pre-launch copy without its support module")?;
            let bytes_of = |t: &Vec<u64>| std::slice::from_raw_parts(t.as_ptr().cast::<u8>(), t.len() * 8);
            for (j, t) in roll_jobs.iter().zip(&tabs) {
                self.roll_stats[0] += 1; self.roll_stats[1] += j.5; self.roll_stats[2] += j.6;
                if j.6 == 0 { continue; }
                dev::upload(cuda, t[0], bytes_of(&j.2))?;
                dev::upload(cuda, t[1], bytes_of(&j.3))?;
                dev::upload(cuda, t[2], bytes_of(&j.4))?;
                // the old words of the changed blocks go to the tape, then the live words go into the copy (stream order)
                sup.runs_copy(cuda, j.1, t[0], j.2.len() / 3)?;
                sup.runs_copy(cuda, j.0, t[1], j.3.len() / 3)?;
                undo.push((j.0, t[2], j.4.len() / 3));
            }
            for (k, r) in roll_new { self.roll_stats[3] += k.3 - k.2; self.roll.insert(k, r); }
        }
        // 60 km tape (AD_P3_STRIPE=1): a deferred MYNN-EDMF bind may band its regions when every argument starts on a
        // row of ni words of its segment (so the kernel's (k * ni + i) element offsets fall on the segment's rows).
        let stripe_geom = p3 && p3_stripe_on() && {
            let ni = match (env.get("ims"), env.get("ime")) { (Some(&a), Some(&b)) if b >= a => (b - a + 1.0) as usize, _ => 0 };
            ni > 0 && !p3_offs.is_empty() && p3_offs.iter().all(|&(al, off)| match segs.find(al, off) {
                Ok(s) => !s.active || (off - s.lo) % (ni * 4) == 0,
                Err(_) => false,
            })
        };
        let mut p = Prepared { function: function.into(), unit: row.unit.clone(), kind, segs, region: (region, total.max(256)), args: args.to_vec(),
                      grid, block, env, identity: None, notes, deferred: Some((total, p3, p4)), transient_bytes: 0, undo,
                      stripe_geom, stripe_ok: defer && stripe_geom, stripe: None };
        if !defer { self.bind_tables(cuda, arena, live, &mut p)?; p.deferred = None; }
        if wait { dev::sync(cuda)?; }
        p.transient_bytes = (arena.used - u0).saturating_sub(pre_bytes);
        Ok(p)
    }

    /// tape-dedupe-and-pack: bind a `prepare_deferred` row's transient buffers from `arena` (the reverse scratch, reset
    /// before this row's adjoint): the same layout `prepare` builds, at new addresses. A row prepared whole is unchanged.
    pub unsafe fn bind_transient(&mut self, cuda: &Cuda, arena: &mut Arena, live: &HashMap<u64, Allocation>, p: &mut Prepared) -> Result<(), String> {
        let Some((total, p3, p4)) = p.deferred.take() else { return Ok(()) };
        // 60 km tape (AD_P3_STRIPE=1): the banded layout (it reads the segments and the launch's scalars only).
        // Off: None, and the head's layout below, byte for byte.
        let banded = if p.stripe_ok { p.stripe_plan() } else { None };
        p.stripe = None;
        if let Some((pl, ni, i0, cols)) = banded.filter(|_| p3 && !p4) {
            let region = arena.take(pl.total.max(256))?;
            dev::zero(cuda, region, pl.total.max(256))?;
            let snap_region = arena.take(pl.total.max(256))?;
            for (s, &place) in p.segs.v.iter_mut().filter(|s| s.active).zip(&pl.place) {
                s.shadow = crate::p3_stripe::vbase(region, place, i0);
                s.snap = crate::p3_stripe::vbase(snap_region, place, i0);
            }
            p.region = (region, pl.total.max(256));
            p.stripe = Some((ni, i0, cols, snap_region));
            p.notes.push(format!("p3 stripe: {cols} columns from {i0} of {ni}, {} arrays a band, region {} bytes against {}", pl.per_band, pl.total, pl.full));
            return self.bind_tables(cuda, arena, live, p);
        }
        let region = arena.take(total.max(256))?;
        dev::zero(cuda, region, total.max(256))?;
        let snap_region = if p3 { arena.take(total.max(256))? } else { 0 };
        let mut at = 0u64;
        for s in p.segs.v.iter_mut() {
            if s.active {
                s.shadow = region + at;
                if p3 { s.snap = snap_region + at; }
                at += (s.shadow_bytes().div_ceil(256) * 256) as u64;
            }
            if p4 { s.scratch = arena.take(s.hi - s.lo)?; }
        }
        p.region = (region, total.max(256));
        self.bind_tables(cuda, arena, live, p)
    }

    /// The device tables that hold shadow / snapshot / scratch addresses (TEMPO-from pointer tables, the broadcast view
    /// copy-row gather, the P6 lists, the RUC land pointer table), built once the buffers are bound.
    unsafe fn bind_tables(&mut self, cuda: &Cuda, arena: &mut Arena, live: &HashMap<u64, Allocation>, p: &mut Prepared) -> Result<(), String> {
        let segs = &p.segs;
        let notes = &mut p.notes;
        if let Kind::TempoFrom(tf) = &mut p.kind { tf_tables(cuda, arena, segs, tf)?; }
        // Copy-row form for a broadcast view copy (P8: fan-in transpose through the deterministic gather).
        if let Kind::View { dst, src, in_place: false, rows: r } = &mut p.kind {
            if broadcasts(src) {
                let abase = arena.base;
                let el = |a: u64, off: usize| -> Result<i64, String> { Ok(((segs.shadow(a, off)? - abase) / W as u64) as i64) };
                let pad = 4 - dst.shape.len();
                let (mut n, mut ds, mut ss) = ([1i64; 4], [0i64; 4], [0i64; 4]);
                for k in 0..dst.shape.len() { n[pad + k] = dst.shape[k]; ds[pad + k] = dst.strides[k] / 4; ss[pad + k] = src.strides[k] / 4; }
                let dr = rows::DevRow { dst: el(dst.alloc, dst.off)?, src: el(src.alloc, src.off)?, ds, ss, n: [n[1], n[2], n[3]], count: n.iter().product() };
                let g = rows::gather(&[dr])?;
                *r = Some(RowsGather { dev_rows: vec![dr], src_word: g.src_word, start: g.start, item: g.item, dst_word: g.dst_word,
                                       sourceless: g.sourceless, injective: g.injective });
                notes.push("broadcast source: copy-row gather transpose".into());
            }
        }
        if let Kind::P6(b) = &mut p.kind { b.lists = Self::p6_lists(cuda, arena, live, segs, b)?; }
        if let Kind::RucLand(b) = &mut p.kind { b.ptab = Self::rl_table(cuda, arena, segs, b)?; }
        Ok(())
    }

    /// Restore the P4 primal scratch from the pre snapshots (the entries get private primal copies on every call).
    unsafe fn restore(cuda: &Cuda, p: &Prepared) -> Result<(), String> {
        for s in &p.segs.v { if s.scratch != 0 { dev::copy(cuda, s.scratch, s.pre, s.hi - s.lo)?; } }
        Ok(())
    }

    // --------------------------------------------------------------------------------------- P8 rules
    fn view(p: &Prepared, abase: u64, c: &CArr) -> Result<V, String> {
        let mut v = V { base: ((p.segs.shadow(c.alloc, c.off)? - abase) / W as u64) as i64, size: c.shape.iter().product(),
                        ndim: c.shape.len() as i32, pad: 0, shape: [1; 4], stride: [0; 4] };
        for k in 0..c.shape.len() { v.shape[k] = c.shape[k]; v.stride[k] = c.strides[k] / 4; }
        Ok(v)
    }
    unsafe fn rule(&self, cuda: &Cuda, arena: &mut Arena, p: &Prepared, adjoint: bool) -> Result<(), String> {
        let abase = arena.base;
        let n = |v: &V| v.size.max(0) as usize;
        let m = &self.p8;
        match &p.kind {
            Kind::View { in_place: true, .. } | Kind::Inactive(_) => Ok(()),
            Kind::View { rows: Some(g), .. } => {
                let mut up = |v: &[u8]| -> Result<u64, String> { let a = arena.take(v.len().max(8))?; dev::upload(cuda, a, v)?; Ok(a) };
                let as_bytes = |v: &[i64]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
                let drows: Vec<u8> = g.dev_rows.iter().flat_map(|r| std::slice::from_raw_parts((r as *const rows::DevRow).cast::<u8>(), std::mem::size_of::<rows::DevRow>()).to_vec()).collect();
                let dr = up(&drows)?;
                let maxc = g.dev_rows.iter().map(|r| r.count).max().unwrap_or(0) as usize;
                let rg = [maxc.div_ceil(256).clamp(1, 1024) as u32, g.dev_rows.len() as u32, 1];
                if !adjoint { return launch(cuda, m.func(cuda, "p8_rows_tl_double")?, rg, [256, 1, 1], &[K::P(abase), K::P(dr)]); }
                let fail = up(&[0u8; 8])?;
                if g.injective { return launch(cuda, m.func(cuda, "p8_rows_ad_direct_double")?, rg, [256, 1, 1], &[K::P(abase), K::P(dr), K::P(fail)]); }
                let (sw, st, it, dw) = (up(&as_bytes(&g.src_word))?, up(&as_bytes(&g.start))?, up(&as_bytes(&g.item))?, up(&as_bytes(&g.dst_word))?);
                let sl = up(&g.sourceless)?;
                let (ns, nd) = (g.src_word.len(), g.dst_word.len());
                launch(cuda, m.func(cuda, "p8_gather_ad_double")?, grid1(ns, 256), [256, 1, 1], &[K::P(abase), K::P(sw), K::P(st), K::P(it), K::P(dw), i64b(ns as i64), K::P(fail)])?;
                launch(cuda, m.func(cuda, "p8_clear_dst_double")?, grid1(nd, 256), [256, 1, 1], &[K::P(abase), K::P(dw), K::P(sl), i64b(nd as i64), K::P(fail)])
            },
            Kind::View { dst, src, .. } => {
                let (dv, sv) = (Self::view(p, abase, dst)?, Self::view(p, abase, src)?);
                let f = m.func(cuda, if adjoint { "p8_view_ad_double" } else { "p8_view_tl_double" })?;
                launch(cuda, f, grid1(n(&dv), 256), [256, 1, 1], &[K::P(abase), vbytes(&dv), vbytes(&sv)])
            },
            Kind::ZeroOut { out } => {
                let ov = Self::view(p, abase, out)?;
                launch(cuda, self.l2.func(cuda, "l2_zero_view_double")?, grid1(n(&ov), 256), [256, 1, 1], &[K::P(abase), vbytes(&ov)])
            },
            Kind::Max { x, out, c } => {
                let (ov, iv) = (Self::view(p, abase, out)?, Self::view(p, abase, x)?);
                // Primal x before the launch: the snapshot of x's segment, indexed in floats from its origin.
                let xs = p.segs.find(x.alloc, x.off)?;
                let mut xv = iv;
                xv.base = ((x.off - xs.lo) / 4) as i64;
                let f = m.func(cuda, if adjoint { "p8_max_ad_double" } else { "p8_max_tl_double" })?;
                launch(cuda, f, grid1(n(&ov), 256), [256, 1, 1], &[K::P(abase), vbytes(&ov), vbytes(&iv), K::P(xs.pre), vbytes(&xv), K::B(c.to_le_bytes().to_vec())])
            },
            Kind::Where { src, mask, dst } => {
                let (dv, sv) = (Self::view(p, abase, dst)?, Self::view(p, abase, src)?);
                let ms = p.segs.find(mask.alloc, mask.off)?;
                let mut mv = V { base: (mask.off - ms.lo) as i64, size: mask.shape.iter().product(), ndim: mask.shape.len() as i32, pad: 0, shape: [1; 4], stride: [0; 4] };
                for k in 0..mask.shape.len() { mv.shape[k] = mask.shape[k]; mv.stride[k] = mask.strides[k]; }
                let f = m.func(cuda, if adjoint { "p8_where_ad_double" } else { "p8_where_tl_double" })?;
                launch(cuda, f, grid1(n(&dv), 256), [256, 1, 1], &[K::P(abase), vbytes(&dv), vbytes(&sv), K::P(ms.pre), vbytes(&mv)])
            },
            Kind::Scale { x, out, c } => {
                let (ov, iv) = (Self::view(p, abase, out)?, Self::view(p, abase, x)?);
                let f = self.l2.func(cuda, if adjoint { "l2_scale_ad_double" } else { "l2_scale_tl_double" })?;
                launch(cuda, f, grid1(n(&ov), 256), [256, 1, 1], &[K::P(abase), vbytes(&ov), vbytes(&iv), K::B(c.to_le_bytes().to_vec())])
            },
            _ => Err("not a rule".into()),
        }
    }

    // --------------------------------------------------------------------------------------- P4
    fn dev_arg(row: &Row, p: &Prepared, name: &str) -> Result<(u64, usize), String> {
        let a = row.arg(name)?;
        target(p.args.get(a.pos).ok_or("argument position")?).ok_or_else(|| format!("{}: {name} is not a device pointer", row.function))
    }
    fn sc(row: &Row, p: &Prepared, name: &str) -> Result<K, String> {
        let a = row.arg(name)?;
        Ok(K::B(scalar_bytes(p.args.get(a.pos).ok_or("argument position")?).ok_or_else(|| format!("{}: {name} is not by value", row.function))?.to_vec()))
    }
    /// The forward's own argument list with every device pointer moved to its pre-launch primal snapshot (P4 v2:
    /// the P4 entries read it and never write it, so there is no private copy).
    fn fwd(row: &Row, p: &Prepared) -> Result<Vec<K>, String> {
        let mut pos: Vec<&ArgRow> = row.args.iter().collect();
        pos.sort_by_key(|a| a.pos);
        let mut v = Vec::new();
        for a in pos {
            let arg = p.args.get(a.pos).ok_or("argument position")?;
            v.push(if a.device { match target(arg) { Some((al, off)) => K::P(p.segs.pre(al, off)?), None => K::P(0) } }
                   else { K::B(scalar_bytes(arg).ok_or("scalar")?.to_vec()) });
        }
        Ok(v)
    }
    /// P4 v2: at a tangent apply, reserve the taped P4 adjoints' workspace (tape, counts, order) while the card still
    /// has room, as ruc_land_reserve does. THE BREAKAGE: the tape was first allocated in the reverse, when a full card has
    /// no device memory left; under a VRAM budget (mem-oocore) it then lands in host-resident managed memory and every tape
    /// node crosses the bus (a test machine final-60km-8c: mynnsfc_columns 1.4-1.6 s per call at x1.163842 against 8 ms at
    /// x1.40962). The size is the budget or 32 B x P4_RESERVE_NODES per column, whichever is smaller; the adjoint still
    /// grows the block when a launch needs more. A reservation that cannot be made is logged, not an error.
    /// AD_P4_RESERVE=0 turns it off (A/B only).
    unsafe fn p4_reserve(&mut self, cuda: &Cuda, n: usize, budget: usize) {
        if env_var("AD_P4_RESERVE").as_deref() == Some("0") { return; }
        let want = budget.min(32 * P4_RESERVE_NODES * n.max(1));
        if self.scratch.get("p4_tape").map(|g| g.bytes).unwrap_or(0) >= want { return; }
        let (mut free, mut total) = (0usize, 0usize);
        let _ = (cuda.mem_info)(&mut free, &mut total);
        let r = self.scratch.entry("p4_tape").or_default().need_exact(cuda, want)
            .and_then(|_| self.scratch.entry("p4_counts").or_default().need_exact(cuda, 4 * n.max(1)))
            .and_then(|_| self.scratch.entry("p4_order").or_default().need_exact(cuda, 4 * n.max(1)));
        eprintln!("{}", json!({"event": "p4_scratch_reserved", "columns": n, "bytes": want, "card_free_before": free, "ok": r.is_ok(), "error": r.err()}));
    }
    unsafe fn p4(&mut self, cuda: &Cuda, p: &Prepared, adjoint: bool) -> Result<(), String> {
        let row = self.rows[&p.function].clone();
        let fam = if matches!(row.kind.as_str(), "p4_run" | "p4_diff" | "p4_dx") { "ugwp" } else { "mynnsfc" };
        let sh = |name: &str| -> Result<K, String> { let (a, o) = Self::dev_arg(&row, p, name)?; Ok(K::P(p.segs.shadow(a, o)?)) };
        // P4 v2: the primal pointers are the pre snapshots themselves (read only in every P4 entry); no restore.
        let scr = |name: &str| -> Result<K, String> { let (a, o) = Self::dev_arg(&row, p, name)?; Ok(K::P(p.segs.pre(a, o)?)) };
        let flags = self.scratch.entry("p4_flags").or_default().need(cuda, 64)?;
        dev::zero(cuda, flags, 64)?;
        let (ov, sg, st, rb) = (flags, flags + 4, flags + 8, flags + 12);
        // AD_P4_TAPE_MIB: the taped adjoints' workspace budget for this run (default: the bundle's p4_tape_budget_bytes).
        let budget = env_var("AD_P4_TAPE_MIB").and_then(|v| v.parse::<usize>().ok()).filter(|m| *m > 0).map(|m| m << 20).unwrap_or(self.p4_budget);
        let sort_on = env_var("AD_P4_SORT").as_deref() != Some("0");
        // G4: the ugwp entries keep their column arrays in a frame arena, one frame per launch thread; the module global
        // p4_frame_base points each module at it. F070: every ugwp launch below runs in column chunks of at most lim
        // threads, so the arena is sized for one chunk (at most lim frames), not for every column of the mesh.
        let lim = p4_ugwp_threads()?;
        if row.kind == "p4_run" {
            let nc = p.env.get("nc").copied().ok_or("ugwp_oro_run: nc")? as usize;
            if p.grid[1] != 1 || p.grid[2] != 1 || p.block[1] != 1 || p.block[2] != 1 {
                return Err(format!("ugwp_oro_run: frame arena needs a 1-D launch, got grid {:?} block {:?}", p.grid, p.block)); }
            let nt = p4_ugwp_frame_threads(nc, lim, p.block[0] as usize);
            let fb = { let img = self.p4.get("ugwp").ok_or("P4 ugwp image not loaded")?; img.frame_tl.max(img.frame_rev) };
            let frame = self.scratch.entry("p4_frame").or_default().need_exact(cuda, nt * fb)?;
            let img = self.p4.get("ugwp").ok_or("P4 ugwp image not loaded")?;
            img.tl.set_global(cuda, "p4_frame_base", &frame.to_le_bytes())?;
            img.rev.set_global(cuda, "p4_frame_base", &frame.to_le_bytes())?;
        }
        let (f_tl, f_ad) = {
            let img = self.p4.get(fam).ok_or_else(|| format!("P4 {fam} image not loaded"))?;
            let names = match row.kind.as_str() {
                "p4_from" => ("mynnsfc_from_mpas__tl64", "mynnsfc_from_mpas__ad64"), "p4_to" => ("mynnsfc_to_mpas__tl64", "mynnsfc_to_mpas__ad64"),
                "p4_columns" => ("mynnsfc_columns__tl64", "mynnsfc_columns__ad64"), "p4_run" => ("ugwp_oro_run__tl64", "ugwp_oro_run__ad64"),
                "p4_diff" => ("ugwp_oro_diff__tl64", "ugwp_oro_diff__ad64"), "p4_dx" => ("ugwp_oro_dx__tl64", "ugwp_oro_dx__ad64"),
                k => return Err(format!("P4 kind {k}")) };
            (img.tl.func(cuda, names.0)?, img.rev.func(cuda, names.1)?)
        };
        match row.kind.as_str() {
            "p4_from" => {
                let mut a = vec![scr("f")?, scr("sea")?, scr("density")?, scr("sst")?, scr("ice")?, Self::sc(&row, p, "len_disp")?,
                                 Self::sc(&row, p, "threshold")?, Self::sc(&row, p, "fraction")?, Self::sc(&row, p, "n")?];
                a.extend([sh("f")?, sh("sea")?, sh("sst")?]);
                launch(cuda, if adjoint { f_ad } else { f_tl }, p.grid, p.block, &a)
            },
            "p4_to" => {
                let mut a = vec![scr("f")?, scr("sea")?, scr("ice")?, Self::sc(&row, p, "threshold")?, Self::sc(&row, p, "n")?];
                a.extend([sh("f")?, sh("sea")?, sh("ice")?]);
                launch(cuda, if adjoint { f_ad } else { f_tl }, p.grid, p.block, &a)
            },
            "p4_diff" => {
                let mut a = Self::fwd(&row, p)?;
                a.extend([sh("before")?, sh("after")?, sh("diff")?]);
                launch(cuda, if adjoint { f_ad } else { f_tl }, p.grid, p.block, &a)
            },
            "p4_dx" => {
                let mut a = Self::fwd(&row, p)?;
                a.push(sh("dx")?);
                launch(cuda, if adjoint { f_ad } else { f_tl }, p.grid, p.block, &a)
            },
            "p4_columns" => {
                let n = p.env.get("n").copied().ok_or("mynnsfc_columns: n")? as usize;
                let head = || -> Result<Vec<K>, String> { Ok(vec![scr("f")?, scr("options")?, scr("vegtype")?, Self::sc(&row, p, "xice_threshold")?,
                    Self::sc(&row, p, "present")?, Self::sc(&row, p, "n")?, Self::sc(&row, p, "step_override")?, Self::sc(&row, p, "init_override")?]) };
                if !adjoint {
                    self.p4_reserve(cuda, n, budget);
                    let mut a = head()?;
                    a.extend([sh("f")?, K::P(sg)]);
                    return launch(cuda, f_tl, p.grid, p.block, &a);
                }
                // Count pass, then tape + sweep in column chunks under the budget. f is the PRE state (the hand-off).
                let t0 = std::time::Instant::now();
                let ordered = sort_on && self.p4.get("mynnsfc").map_or(false, |i| i.order);
                let counts = self.scratch.entry("p4_counts").or_default().need_exact(cuda, 4 * n)?;
                if ordered { self.p4["mynnsfc"].rev.set_global(cuda, "p4_order", &0u64.to_le_bytes())?; }
                let mut a = head()?;
                a.extend([sh("f")?, K::P(0), K::P(0), K::P(0), i32b(0), i64b(n as i64), i32b(0), i32b(n as i32), K::P(counts), K::P(ov), K::P(sg)]);
                launch(cuda, f_ad, grid1(n, 128), [128, 1, 1], &a)?;
                dev::sync(cuda)?;
                let cnt: Vec<i32> = dev::download(cuda, counts, 4 * n)?.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
                let (order, chunks) = p4_tape_chunks(&cnt, budget, 128, n.div_ceil(128) * 128, ordered);
                let tape_b = chunks.iter().map(|&(_, k, cap)| 32 * cap * (k.div_ceil(128) * 128)).max().unwrap_or(0);
                let tape = self.scratch.entry("p4_tape").or_default().need_exact(cuda, tape_b)?;
                if ordered {
                    let ob = self.scratch.entry("p4_order").or_default().need_exact(cuda, 4 * n)?;
                    dev::upload(cuda, ob, &order.iter().flat_map(|c| c.to_le_bytes()).collect::<Vec<u8>>())?;
                    self.p4["mynnsfc"].rev.set_global(cuda, "p4_order", &ob.to_le_bytes())?;
                }
                for &(c0, k, cap) in &chunks {
                    let stride = k.div_ceil(128) * 128;
                    let (ab, w, adj) = (tape, tape + (8 * cap * stride) as u64, tape + (24 * cap * stride) as u64);
                    let mut a = head()?;
                    a.extend([sh("f")?, K::P(ab), K::P(w), K::P(adj), i32b(cap as i32), i64b(stride as i64), i32b(c0 as i32), i32b(k as i32), K::P(0), K::P(ov), K::P(0)]);
                    launch(cuda, f_ad, grid1(k, 128), [128, 1, 1], &a)?;
                }
                dev::sync(cuda)?;
                if ordered { self.p4["mynnsfc"].rev.set_global(cuda, "p4_order", &0u64.to_le_bytes())?; }
                if dev::download(cuda, ov, 4)? != [0, 0, 0, 0] { return Err("mynnsfc_columns adjoint tape overflow".into()); }
                eprintln!("{}", json!({"event": "native_p4", "function": row.function, "pass": "ad", "columns": n, "chunks": chunks.len(), "sorted": ordered,
                    "nodes": cnt.iter().map(|&c| c.max(0) as u64).sum::<u64>(), "cap_max": cnt.iter().copied().max().unwrap_or(0), "tape_bytes": tape_b,
                    "budget_bytes": budget, "ms": t0.elapsed().as_secs_f64() * 1e3}));
                Ok(())
            },
            "p4_run" => {
                let nc = p.env.get("nc").copied().ok_or("ugwp_oro_run: nc")? as usize;
                // out and sout are fully overwritten by the forward (P4-ENTRIES): their tangent is rebuilt from the
                // inputs alone (zeroed first) and their whole cotangent is consumed (zeroed after).
                let pure = |name: &str| -> Result<(u64, usize), String> {
                    let a = row.arg(name)?;
                    let (al, o) = Self::dev_arg(&row, p, name)?;
                    Ok((p.segs.shadow(al, o)?, row.extent_bytes(a, &p.env)? / 4 * W)) };
                let (o_out, s_out) = (pure("out")?, pure("sout")?);
                if !adjoint {
                    self.p4_reserve(cuda, nc, budget);
                    dev::zero(cuda, o_out.0, o_out.1)?;
                    dev::zero(cuda, s_out.0, s_out.1)?;
                    // F070: column chunks of at most lim (col0, ncols), in the forward's block size.
                    let b = p.block[0] as usize;
                    for (c0, k) in p4_ugwp_chunks(nc, lim) {
                        let mut a = Self::fwd(&row, p)?;
                        a.extend([sh("layers")?, sh("prsi0")?, sh("surface")?, sh("out")?, sh("sout")?, K::P(st), K::P(sg), K::P(rb),
                                  i32b(c0 as i32), i32b(k as i32)]);
                        launch(cuda, f_tl, grid1(k, b), p.block, &a)?;
                    }
                    return Ok(());
                }
                let t0 = std::time::Instant::now();
                // other-schemes (AD_P4_UGWP_DIRECT): the tape-free adjoint takes every column it covers in one pass
                // per F070 chunk, with no count pass, no tape and one 4-byte read back; the columns it does not cover (diags,
                // an invalid kpbl) come back as a list, and only they run the taped reverse below (sub). Same answer bytes.
                let mut sub: Option<Vec<i32>> = None;
                if let Some((fd, per_level)) = self.p4_dad.as_ref().map(|d| (d.f, d.per_level)) {
                    let nz = p.env.get("nz").copied().ok_or("ugwp_oro_run: nz")? as usize;
                    let nt = nc.min(lim).div_ceil(64) * 64;
                    let frame = self.scratch.entry("p4_dframe").or_default().need_exact(cuda, nt * (nz + 2) * per_level)?;
                    let fb = self.scratch.entry("p4_dfb").or_default().need_exact(cuda, 4 * nc + 16)?;
                    let fbn = fb + (4 * nc) as u64;
                    dev::zero(cuda, fbn, 4)?;
                    for (c0, k) in p4_ugwp_chunks(nc, lim) {
                        let mut a = Self::fwd(&row, p)?;
                        a.extend([sh("layers")?, sh("prsi0")?, sh("surface")?, sh("out")?, sh("sout")?, K::P(frame), i32b(c0 as i32), i32b(k as i32), K::P(fb), K::P(fbn)]);
                        launch(cuda, fd, grid1(k, 64), [64, 1, 1], &a)?;
                    }
                    let nfb = i32::from_le_bytes(dev::download(cuda, fbn, 4)?[..4].try_into().unwrap()).max(0) as usize;
                    if nfb == 0 {
                        dev::zero(cuda, o_out.0, o_out.1)?;
                        dev::zero(cuda, s_out.0, s_out.1)?;
                        eprintln!("{}", json!({"event": "native_p4", "function": row.function, "pass": "ad", "route": "direct", "columns": nc, "fallback": 0,
                            "frame_bytes": nt * (nz + 2) * per_level, "ms": t0.elapsed().as_secs_f64() * 1e3}));
                        return Ok(());
                    }
                    let mut l: Vec<i32> = dev::download(cuda, fb, 4 * nfb)?.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
                    l.sort_unstable();
                    sub = Some(l);
                }
                let ordered = sort_on && self.p4.get("ugwp").map_or(false, |i| i.order);
                if sub.is_some() && !ordered { return Err("AD_P4_UGWP_DIRECT: the fallback columns need a column order (p4_rev_order and AD_P4_SORT on)".into()); }
                let counts = self.scratch.entry("p4_counts").or_default().need_exact(cuda, 4 * nc)?;
                let nsub = sub.as_ref().map_or(nc, |s| s.len());
                if let Some(s) = sub.as_ref() {
                    let ob = self.scratch.entry("p4_order").or_default().need_exact(cuda, 4 * nc)?;
                    dev::upload(cuda, ob, &s.iter().flat_map(|c| c.to_le_bytes()).collect::<Vec<u8>>())?;
                    self.p4["ugwp"].rev.set_global(cuda, "p4_order", &ob.to_le_bytes())?;
                } else if ordered { self.p4["ugwp"].rev.set_global(cuda, "p4_order", &0u64.to_le_bytes())?; }
                let mk = |ab: u64, w: u64, adj: u64, cap: i32, stride: i64, c0: i32, ncol: i32, cnt: u64| -> Result<Vec<K>, String> {
                    let mut a = Self::fwd(&row, p)?;
                    a.extend([sh("layers")?, sh("prsi0")?, sh("surface")?, sh("out")?, sh("sout")?, K::P(ab), K::P(w), K::P(adj), i32b(cap), i64b(stride),
                              i32b(c0), i32b(ncol), K::P(cnt), K::P(ov), K::P(sg), K::P(st)]);
                    Ok(a) };
                // F070: the count pass runs in the same column chunks of at most lim (counts[col0 + t]).
                for (c0, k) in p4_ugwp_chunks(nsub, lim) {
                    launch(cuda, f_ad, grid1(k, 64), [64, 1, 1], &mk(0, 0, 0, 0, nc as i64, c0 as i32, k as i32, counts)?)?;
                }
                dev::sync(cuda)?;
                let cnt_all: Vec<i32> = dev::download(cuda, counts, 4 * nc)?.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
                let cnt: Vec<i32> = match sub.as_ref() { Some(s) => s.iter().map(|&c| cnt_all[c as usize]).collect(), None => cnt_all };
                // P4 v2: chunks by the columns' own counts (sorted when the image takes an order); the primal inputs are the
                // pre snapshot, read in place, so nothing is copied per chunk.
                let (order0, chunks) = p4_tape_chunks(&cnt, budget, 64, lim.min(nsub.div_ceil(64) * 64), ordered);
                let order: Vec<i32> = match sub.as_ref() { Some(s) => order0.iter().map(|&j| s[j as usize]).collect(), None => order0 };
                let tape_b = chunks.iter().map(|&(_, k, cap)| 32 * cap * (k.div_ceil(64) * 64)).max().unwrap_or(0);
                let tape = self.scratch.entry("p4_tape").or_default().need_exact(cuda, tape_b)?;
                if ordered {
                    let ob = self.scratch.entry("p4_order").or_default().need_exact(cuda, 4 * nc)?;
                    dev::upload(cuda, ob, &order.iter().flat_map(|c| c.to_le_bytes()).collect::<Vec<u8>>())?;
                    self.p4["ugwp"].rev.set_global(cuda, "p4_order", &ob.to_le_bytes())?;
                }
                for &(c0, k, cap) in &chunks {
                    let stride = k.div_ceil(64) * 64;
                    let (ab, w, adj) = (tape, tape + (8 * cap * stride) as u64, tape + (24 * cap * stride) as u64);
                    launch(cuda, f_ad, grid1(k, 64), [64, 1, 1], &mk(ab, w, adj, cap as i32, stride as i64, c0 as i32, k as i32, 0)?)?;
                }
                dev::sync(cuda)?;
                if ordered { self.p4["ugwp"].rev.set_global(cuda, "p4_order", &0u64.to_le_bytes())?; }
                if dev::download(cuda, ov, 4)? != [0, 0, 0, 0] { return Err("ugwp_oro_run adjoint tape overflow".into()); }
                dev::zero(cuda, o_out.0, o_out.1)?;
                dev::zero(cuda, s_out.0, s_out.1)?;
                eprintln!("{}", json!({"event": "native_p4", "function": row.function, "pass": "ad", "route": if sub.is_some() { "direct+tape" } else { "tape" },
                    "taped_columns": nsub, "columns": nc, "chunks": chunks.len(), "sorted": ordered,
                    "nodes": cnt.iter().map(|&c| c.max(0) as u64).sum::<u64>(), "cap_max": cnt.iter().copied().max().unwrap_or(0), "tape_bytes": tape_b,
                    "budget_bytes": budget, "ms": t0.elapsed().as_secs_f64() * 1e3}));
                Ok(())
            },
            k => Err(format!("P4 kind {k}")),
        }
    }

    // --------------------------------------------------------------------------------------- P6
    /// oracle_ice_ruc_ice: read the launch's pointer table (live, before the forward runs), resolve every helper word
    /// to (allocation, offset), select the sea-ice columns and the forward's own flags from the pre state as P6's
    /// driver does (tree/ad/phys/p6/src/main.rs), and request the bound extents (all active: every helper is a
    /// derivative input or output). Two helpers over partly overlapping words are refused by name.
    unsafe fn p6_bind(cuda: &Cuda, row: &Row, live: &HashMap<u64, Allocation>, args: &[Argument], env: &Env,
                      want: &mut Vec<(u64, usize, usize, bool)>) -> Result<P6Bind, String> {
        let f = row.function.as_str();
        let d = &row.raw["p6"];
        let u = |k: &str| -> Result<usize, String> { d[k].as_u64().map(|v| v as usize).ok_or_else(|| format!("{f}: row p6.{k}")) };
        let list = |k: &str| -> Result<Vec<i64>, String> {
            d[k].as_array().ok_or_else(|| format!("{f}: row p6.{k}"))?.iter().map(|v| v.as_i64().ok_or_else(|| format!("{f}: row p6.{k} entry"))).collect() };
        let (slots, lwork) = (list("slots")?, list("local_work")?);
        let names: Vec<String> = d["names"].as_array().ok_or_else(|| format!("{f}: row p6.names"))?.iter().map(|v| v.as_str().unwrap_or("").to_owned()).collect();
        if slots.len() + lwork.len() != P6_NS || names.len() != P6_NS || u("nz")? != P6_NZ {
            return Err(format!("{f}: row has {} slots + {} arena helpers, {} names, nz {}; the image expects {P6_NS} helpers, nz {P6_NZ}",
                               slots.len(), lwork.len(), names.len(), u("nz")?));
        }
        let nc = env.get("nc").copied().ok_or_else(|| format!("{f}: launch has no nc scalar"))? as usize;
        let ptr_arg = |pos: usize, name: &str| -> Result<(u64, usize), String> {
            args.get(pos).and_then(target).ok_or_else(|| format!("{f}: {name} is not a device pointer")) };
        let ((tal, toff), (wal, woff)) = (ptr_arg(0, "ptrs")?, ptr_arg(2, "work")?);
        let nptr = u("pointer_slots")?;
        let tl = live.get(&tal).ok_or_else(|| format!("{f}: pointer table in unknown allocation {tal}"))?;
        if toff + 8 * nptr > tl.bytes { return Err(format!("{f}: {nptr}-slot pointer table at {toff} exceeds allocation {tal}")); }
        let words: Vec<u64> = dev::download(cuda, tl.address + toff as u64, 8 * nptr)?.chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        let mut spans: Vec<(u64, usize, u64)> = live.values().map(|a| (a.address, a.bytes, a.id)).collect();
        spans.sort_unstable();
        let slot = |s: usize| -> Result<(u64, usize), String> {
            let w = *words.get(s).ok_or_else(|| format!("{f}: pointer slot {s} beyond the table"))?;
            if w == 0 { return Err(format!("{f}: pointer slot {s} is null")); }
            let i = spans.partition_point(|x| x.0 <= w);
            match i.checked_sub(1).map(|i| spans[i]) {
                Some((a, n, id)) if w < a + n as u64 => Ok((id, (w - a) as usize)),
                _ => Err(format!("{f}: pointer slot {s} ({w:#x}) is in no live allocation")),
            }
        };
        let read = |(al, off): (u64, usize), bytes: usize| -> Result<Vec<u8>, String> {
            let a = live.get(&al).ok_or_else(|| format!("{f}: unknown allocation {al}"))?;
            if off + bytes > a.bytes { return Err(format!("{f}: {bytes} bytes at {off} exceed allocation {al}")); }
            dev::download(cuda, a.address + off as u64, bytes) };
        let int0 = |k: &str| -> Result<i32, String> {
            let s = d["flags"][k].as_u64().ok_or_else(|| format!("{f}: row p6.flags.{k}"))? as usize;
            Ok(i32::from_le_bytes(read(slot(s)?, 4)?[..4].try_into().unwrap())) };
        let (ktau, restart, cycling) = (int0("ktau")?, int0("restart")?, int0("cycling")?);
        let flags = [int0("isice")?, (ktau == 1 && restart == 0 && cycling == 0) as i32, int0("frpcpn")?,
                     (int0("graupel")? != 0) as i32, (int0("myj")? != 0) as i32];
        let xice: Vec<f32> = read(slot(u("xice_slot")?)?, 4 * nc)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let thr = f32::from_le_bytes(read(slot(u("threshold_slot")?)?, 4)?[..4].try_into().unwrap());
        let cols: Vec<usize> = (0..nc).filter(|&c| xice[c] >= thr).collect();
        let mut base: Vec<Option<(u64, usize)>> = slots.iter().map(|&s| slot(s as usize).map(Some)).collect::<Result<_, _>>()?;
        base.extend(lwork.iter().map(|&w| (w >= 0).then(|| (wal, woff + 4 * w as usize * nc))));
        let (tso, zs) = (slot(u("tso_slot")?)?, slot(u("zs_slot")?)?);
        let first: Vec<usize> = (0..P6_NS).map(|k| (0..k).find(|&j| base[j].is_some() && base[j] == base[k]).unwrap_or(k)).collect();
        let mut ext: Vec<(&str, u64, usize, usize)> = Vec::new();
        for k in 0..P6_NS { if let (true, Some((al, off))) = (first[k] == k, base[k]) { ext.push((names[k].as_str(), al, off, off + 4 * nc)); } }
        ext.push(("tso", tso.0, tso.1, tso.1 + 4 * nc * P6_NZ));
        ext.push(("zs", zs.0, zs.1, zs.1 + 4 * nc * P6_NZ));
        for i in 0..ext.len() {
            for j in i + 1..ext.len() {
                let (a, b) = (&ext[i], &ext[j]);
                if a.1 == b.1 && a.2 < b.3 && b.2 < a.3 { return Err(format!("{f}: helpers {} and {} overlap in allocation {}", a.0, b.0, a.1)); }
            }
        }
        for &(name, al, lo, hi) in &ext {
            let la = live.get(&al).ok_or_else(|| format!("{f}: {name} in unknown allocation {al}"))?;
            if hi > la.bytes { return Err(format!("{f}: {name} extent {} bytes at {lo} exceeds allocation {al}", hi - lo)); }
            want.push((al, lo, hi, true));
        }
        Ok(P6Bind { nc, cols, flags, base, first, tso, zs, lists: [0; 5] })
    }

    /// The five device address lists of a bound P6 launch (p6_bind.cu), in the arena the bind lives in.
    unsafe fn p6_lists(cuda: &Cuda, arena: &mut Arena, live: &HashMap<u64, Allocation>, segs: &Segs, b: &P6Bind) -> Result<[u64; 5], String> {
        let n = b.cols.len();
        if n == 0 { return Ok([0; 5]); }
        let sh = |w: Option<(u64, usize)>| -> Result<u64, String> { w.map_or(Ok(0), |(a, o)| segs.shadow(a, o)) };
        let pr = |w: Option<(u64, usize)>| -> Result<u64, String> { w.map_or(Ok(0), |(a, o)| segs.pre(a, o)) };
        let lv = |w: Option<(u64, usize)>| -> Result<u64, String> {
            w.map_or(Ok(0), |(a, o)| live.get(&a).map(|x| x.address + o as u64).ok_or_else(|| format!("unknown allocation {a}"))) };
        let lev = |(a, o): (u64, usize), c: usize, l: usize| Some((a, o + 4 * (l * b.nc + c)));
        let (mut a_in, mut a_out, mut a_add) = (vec![0u64; 90 * n], vec![0u64; 81 * n], vec![0u64; 90 * n]);
        let (mut a_pre, mut a_post) = (vec![0u64; 90 * n], vec![0u64; 81 * n]);
        for (j, &c) in b.cols.iter().enumerate() {
            for k in 0..P6_NS {
                let w = b.base[k].map(|(a, o)| (a, o + 4 * c));
                let s = sh(w)?;
                a_in[k * n + j] = s;
                if b.first[k] == k { a_out[k * n + j] = s; a_add[k * n + j] = s; }
                a_pre[j * 90 + k] = pr(w)?;
                a_post[j * 72 + k] = lv(w)?;
            }
            for l in 0..P6_NZ {
                let (t, z) = (lev(b.tso, c, l), lev(b.zs, c, l));
                let (st, sz) = (sh(t)?, sh(z)?);
                let (it, iz) = (72 * n + j * 9 + l, 81 * n + j * 9 + l);
                a_in[it] = st; a_out[it] = st; a_add[it] = st;
                a_in[iz] = sz; a_add[iz] = sz;
                a_pre[j * 90 + 72 + l] = pr(t)?;
                a_pre[j * 90 + 81 + l] = pr(z)?;
                a_post[72 * n + j * 9 + l] = lv(t)?;
            }
        }
        let mut up = |v: &[u64]| -> Result<u64, String> {
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            let a = arena.take(bytes.len())?;
            dev::upload(cuda, a, &bytes)?;
            Ok(a) };
        Ok([up(&a_in)?, up(&a_out)?, up(&a_add)?, up(&a_pre)?, up(&a_post)?])
    }

    /// The P6 whole launch on a bound launch: rows from the pre snapshot, p6_construct with the forward's flags, the
    /// augmented primal (column graphs at the launch's own state: branch-frozen), then the TL or AD sweep on compact
    /// float64 copies of the shadow words, written back in place (tangent: outputs set; adjoint: outputs zeroed,
    /// inputs accumulated). Buffer sizes are the P6 driver's (tree/ad/phys/p6/src/main.rs Deriv::new).
    unsafe fn p6(&mut self, cuda: &Cuda, p: &mut Prepared, adjoint: bool, identity: bool) -> Result<Value, String> {
        let Kind::P6(b) = &p.kind else { return Err(format!("{}: not a P6 row", p.function)) };
        let (n, lists, flags, first) = (b.cols.len(), b.lists, b.flags, b.first.clone());
        if n == 0 { return Ok(json!({"sea_ice_columns": 0})); }
        let img = self.p6.as_ref().ok_or("P6 image not loaded")?;
        let (fc, fa, ft, fd, kr, kg, ks, kz, km, table) =
            (img.construct, img.aug, img.tl, img.ad, img.rows, img.gather, img.scatter, img.zero, img.mirror, img.table);
        let sizes = [n * 1756, n * 4096 * 32, n * 2048 * 12, n * 4096 * 32, n * 1024 * 12, n * 2048 * 32, n * 512 * 12, n * 4096 * 4,
                     n * 2048 * 4, n * 145 * 4, n * 59 * 4, n * 64 * 4, n * 2 * 4, n * 1024, n * 2048, n * 81 * 4, n * 81 * 4, n * 4,
                     256 * 2 * 4, 256, n * 4096 * 8, n * 5091 * 8, n * 81 * 8, (90 * n + 5001) * 8, 81 * n * 8, 79 * 32, 73 * 32,
                     n * img.frame];
        let mut offs = vec![0usize; sizes.len()];
        let mut total = 0usize;
        for (i, s) in sizes.iter().enumerate() { offs[i] = total; total += s.div_ceil(256) * 256; }
        let base = self.scratch.entry("p6").or_default().grow(cuda, total, total / 16)?; // mynn: 1/16 headroom, not half again (5.44 -> 3.86 GB at x1.163842); the sea-ice column count can move a little between steps
        let v: Vec<u64> = offs.iter().map(|o| base + *o as u64).collect();
        let (rows, pn, pl, inn, il, cn, cl, map, cmap, layer, flayer, rec, count, inp) =
            (v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13]);
        let (tape, y, mirror, active, branch, fail, work, fw, ow, xin, yout, vin, vout) =
            (v[14], v[15], v[16], v[17], v[18], v[19], v[20], v[21], v[22], v[23], v[24], v[25], v[26]);
        let frame = v[27];
        let failed = |what: &str| -> Result<(), String> {
            let c = i32::from_le_bytes(dev::download(cuda, fail, 4)?[..4].try_into().unwrap());
            if c == 0 { Ok(()) } else { Err(format!("oracle_ice_ruc_ice {what} refused (failure code {c})")) } };
        let (g, b32, g256) = (grid1(n, 32), [32u32, 1, 1], [256u32, 1, 1]);
        let ni = n as i32;
        dev::zero(cuda, rows, sizes[0])?;
        dev::zero(cuda, rec, sizes[11])?;
        dev::zero(cuda, branch, sizes[18])?;
        dev::zero(cuda, fail, 4)?;
        launch(cuda, kr, grid1(90 * n, 256), g256, &[K::P(lists[3]), K::P(rows), i32b(ni)])?;
        launch(cuda, fc, g, b32, &[K::P(rows), K::P(table), i32b(ni), K::P(pn), K::P(pl), K::P(inn), K::P(il), K::P(cn), K::P(cl),
            K::P(map), K::P(cmap), K::P(layer), K::P(flayer), K::P(rec), K::P(count), K::P(inp), K::P(tape),
            i32b(flags[0]), i32b(flags[1]), i32b(flags[2]), i32b(flags[3]), i32b(flags[4])])?;
        // G4: the augmented primal's per-column fragment frames (column c at frame + c * img.frame).
        img._m.set_global(cuda, "si_frame_base", &frame.to_le_bytes())?;
        img._m.set_global(cuda, "si_frame_stride", &(img.frame as u64).to_le_bytes())?;
        launch(cuda, fa, g, b32, &[K::P(inp), K::P(tape), i32b(ni), i32b(P6_NZ as i32), K::P(y), K::P(y + (n * 72 * 4) as u64),
            K::P(active), K::P(branch), K::P(fail)])?;
        dev::sync(cuda)?;
        failed("augmented primal")?;
        let act = dev::download(cuda, active, 4 * n)?.chunks_exact(4).filter(|c| *c != [0, 0, 0, 0]).count();
        if identity && !adjoint {
            launch(cuda, km, grid1(81 * n, 256), g256, &[K::P(lists[4]), K::P(y), K::P(mirror), i64b((81 * n) as i64)])?;
            p.identity = Some((y, mirror, 81 * n * 4));
        }
        // SiLaunchView {address, column stride, element stride (bytes), width, shadow_f64}: inputs 79 (72 scalars, an
        // alias reads its first spelling's array; tso; zs; the table, whose tangent is zero: a module constant), outputs 73.
        let view = |a: u64, col: i64, el: i64, w: i32| -> Vec<u8> {
            let mut x = a.to_le_bytes().to_vec(); x.extend(col.to_le_bytes()); x.extend(el.to_le_bytes()); x.extend(w.to_le_bytes()); x.extend(1i32.to_le_bytes()); x };
        let none = [0u8; 32];
        let mut vi: Vec<u8> = Vec::with_capacity(79 * 32);
        for k in 0..P6_NS { vi.extend(view(xin + (first[k] * n * 8) as u64, 8, 8, 1)); }
        vi.extend(view(xin + (72 * n * 8) as u64, 72, 8, 9)); vi.extend(none); vi.extend(none);
        vi.extend(view(xin + (81 * n * 8) as u64, 72, 8, 9)); vi.extend(none); vi.extend(none);
        vi.extend(view(xin + (90 * n * 8) as u64, 0, 8, 5001));
        let mut vo: Vec<u8> = Vec::with_capacity(73 * 32);
        for k in 0..P6_NS { vo.extend(view(yout + (k * n * 8) as u64, 8, 8, 1)); }
        vo.extend(view(yout + (72 * n * 8) as u64, 72, 8, 9));
        dev::upload(cuda, vin, &vi)?;
        dev::upload(cuda, vout, &vo)?;
        let (nin, nout) = (90 * n, 81 * n);
        dev::zero(cuda, fail, 4)?;
        let sweep = [K::P(tape), i32b(ni), i32b(P6_NZ as i32), i32b(14), K::P(vin), K::P(vout), K::P(work), i32b(4096), K::P(fw), i32b(5091),
                     K::P(ow), i32b(81), K::P(fail)];
        if !adjoint {
            launch(cuda, kg, grid1(nin, 256), g256, &[K::P(lists[0]), K::P(xin), i64b(nin as i64)])?;
            dev::zero(cuda, xin + (nin * 8) as u64, 5001 * 8)?;
            dev::zero(cuda, yout, nout * 8)?;
            launch(cuda, ft, g, b32, &sweep)?;
            dev::sync(cuda)?;
            failed("tangent")?;
            launch(cuda, ks, grid1(nout, 256), g256, &[K::P(lists[1]), K::P(yout), i64b(nout as i64), i32b(0)])?;
        } else {
            launch(cuda, kg, grid1(nout, 256), g256, &[K::P(lists[1]), K::P(yout), i64b(nout as i64)])?;
            dev::zero(cuda, xin, (nin + 5001) * 8)?;
            launch(cuda, fd, g, b32, &sweep)?;
            dev::sync(cuda)?;
            failed("adjoint")?;
            launch(cuda, kz, grid1(nout, 256), g256, &[K::P(lists[1]), i64b(nout as i64)])?;
            launch(cuda, ks, grid1(nin, 256), g256, &[K::P(lists[2]), K::P(xin), i64b(nin as i64), i32b(1)])?;
        }
        Ok(json!({"sea_ice_columns": n, "active_columns": act, "flags_isice_cold_frpcpn_graupel_myj": flags}))
    }

    // --------------------------------------------------------------------------------------- RUC land (step6-rucland)
    /// oracle_land_ruc_land: read the launch's 122-slot pointer table live, before the forward runs; resolve every slot to
    /// (allocation, offset); size the 70 float fields with the entry's own __sizes on that live table (cell 0's dimension
    /// words, as the entry itself reads them); request the extents (fields active, every other read slot and the work arena
    /// as primal-only copies). Refused by name: a null field slot, a non-null slot the row does not name, a field over a
    /// word of another field or of an integer slot (the in-place tangent would alias), an extent past its allocation.
    unsafe fn rl_bind(&mut self, cuda: &Cuda, row: &Row, live: &HashMap<u64, Allocation>, args: &[Argument], env: &Env,
                      want: &mut Vec<(u64, usize, usize, bool)>) -> Result<RlBind, String> {
        let f = row.function.as_str();
        let sizes_fn = self.rl.as_ref().ok_or_else(|| format!("{f}: ruc_land row without its image (schemes.json images.ruc_land)"))?.sizes;
        let d = &row.raw["ruc_land"];
        let u = |k: &str| -> Result<usize, String> { d[k].as_u64().map(|v| v as usize).ok_or_else(|| format!("{f}: row ruc_land.{k}")) };
        let slots_of = |k: &str| -> Result<Vec<usize>, String> {
            d[k].as_array().ok_or_else(|| format!("{f}: row ruc_land.{k}"))?.iter()
                .map(|v| v.as_u64().map(|x| x as usize).ok_or_else(|| format!("{f}: row ruc_land.{k} entry"))).collect() };
        let fields_row: Vec<(String, usize)> = d["fields"].as_array().ok_or_else(|| format!("{f}: row ruc_land.fields"))?.iter()
            .map(|v| Ok((v[0].as_str().ok_or("field name")?.to_owned(), v[1].as_u64().ok_or("field slot")? as usize)))
            .collect::<Result<_, String>>()?;
        let (cell_slots, live_only) = (slots_of("cell_slots")?, slots_of("untouched_slots")?);
        let (tile_slot, ilnb_slot, nwork) = (u("tile_slot")?, u("ilnb_slot")?, u("nwork")?);
        if u("pointer_slots")? != RL_SLOTS { return Err(format!("{f}: row has {} pointer slots, the entry reads {RL_SLOTS}", u("pointer_slots")?)); }
        let nc = env.get("nc").copied().ok_or_else(|| format!("{f}: launch has no nc scalar"))? as usize;
        let ptr_arg = |pos: usize, name: &str| -> Result<(u64, usize), String> {
            args.get(pos).and_then(target).ok_or_else(|| format!("{f}: {name} is not a device pointer")) };
        let ((tal, toff), (bal, boff), (wal, woff)) = (ptr_arg(0, "ptrs")?, ptr_arg(1, "tables")?, ptr_arg(2, "work")?);
        let tl = live.get(&tal).ok_or_else(|| format!("{f}: pointer table in unknown allocation {tal}"))?;
        if toff + 8 * RL_SLOTS > tl.bytes { return Err(format!("{f}: {RL_SLOTS}-slot pointer table at {toff} exceeds allocation {tal}")); }
        let table_live = tl.address + toff as u64;
        let words: Vec<u64> = dev::download(cuda, table_live, 8 * RL_SLOTS)?.chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        let mut spans: Vec<(u64, usize, u64)> = live.values().map(|a| (a.address, a.bytes, a.id)).collect();
        spans.sort_unstable();
        let slot = |s: usize| -> Result<Option<(u64, usize)>, String> {
            let w = words[s];
            if w == 0 { return Ok(None); }
            let i = spans.partition_point(|x| x.0 <= w);
            match i.checked_sub(1).map(|i| spans[i]) {
                Some((a, n, id)) if w < a + n as u64 => Ok(Some((id, (w - a) as usize))),
                _ => Err(format!("{f}: pointer slot {s} ({w:#x}) is in no live allocation")),
            }
        };
        let tables = live.get(&bal).map(|a| a.address + boff as u64).ok_or_else(|| format!("{f}: tables in unknown allocation {bal}"))?;
        let tables_bytes = live.get(&bal).map(|a| a.bytes.saturating_sub(boff)).unwrap_or(0);
        // Field sizes: the entry's __sizes on the live table (words per cell; -1 for a non-field slot).
        let out = self.scratch.entry("ruc_land_sizes").or_default().need(cuda, 4 * RL_SLOTS)?;
        dev::zero(cuda, out, 4 * RL_SLOTS)?;
        launch(cuda, sizes_fn, [1, 1, 1], [1, 1, 1], &[K::P(table_live), K::P(tables), K::P(out), i32b(nc as i32)])?;
        dev::sync(cuda)?;
        let nw: Vec<i32> = dev::download(cuda, out, 4 * RL_SLOTS)?.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
        let mut named = vec![false; RL_SLOTS];
        let mut fields = Vec::new();
        let mut ext: Vec<(String, u64, usize, usize, bool)> = Vec::new(); // (name, allocation, lo, hi, field)
        for (name, k) in &fields_row {
            let (al, off) = slot(*k)?.ok_or_else(|| format!("{f}: field {name} (slot {k}) is null"))?;
            let n = nw[*k];
            if n <= 0 { return Err(format!("{f}: field {name} (slot {k}) has {n} words per cell from __sizes")); }
            fields.push((*k, al, off, n as usize));
            ext.push((name.clone(), al, off, off + 4 * nc * n as usize, true));
            named[*k] = true;
        }
        if let Some(k) = (0..RL_SLOTS).find(|&k| !named[k] && nw[k] != -1) {
            return Err(format!("{f}: __sizes names slot {k} as a float field ({} words per cell) but the row does not", nw[k])); }
        let mut others = Vec::new();
        let mut live_slots = Vec::new();
        for &k in &cell_slots {
            let (al, off) = slot(k)?.ok_or_else(|| format!("{f}: per-cell slot {k} is null"))?;
            others.push((k, al, off, 4 * nc)); ext.push((format!("slot {k}"), al, off, off + 4 * nc, false)); named[k] = true;
        }
        let (al, off) = slot(tile_slot)?.ok_or_else(|| format!("{f}: tile stop slot {tile_slot} is null"))?;
        others.push((tile_slot, al, off, 8)); ext.push(("tile_stop".into(), al, off, off + 8, false)); named[tile_slot] = true;
        if let Some((al, off)) = slot(ilnb_slot)? {
            others.push((ilnb_slot, al, off, 4 * nc)); ext.push(("ilnb".into(), al, off, off + 4 * nc, false));
        }
        named[ilnb_slot] = true;
        for &k in &live_only {
            if let Some((al, off)) = slot(k)? { live_slots.push((k, live.get(&al).ok_or("allocation")?.address + off as u64)); }
            named[k] = true;
        }
        if let Some(k) = (0..RL_SLOTS).find(|&k| !named[k] && words[k] != 0) {
            return Err(format!("{f}: pointer slot {k} is set at this launch but the row does not name it")); }
        for i in 0..ext.len() {
            for j in i + 1..ext.len() {
                let (a, b) = (&ext[i], &ext[j]);
                if (a.4 || b.4) && a.1 == b.1 && a.2 < b.3 && b.2 < a.3 {
                    return Err(format!("{f}: {} and {} overlap in allocation {}", a.0, b.0, a.1)); }
            }
        }
        for (name, al, lo, hi, active) in &ext {
            let la = live.get(al).ok_or_else(|| format!("{f}: {name} in unknown allocation {al}"))?;
            if *hi > la.bytes { return Err(format!("{f}: {name} extent {} bytes at {lo} exceeds allocation {al}", hi - lo)); }
            want.push((*al, *lo, *hi, *active));
        }
        let wl = live.get(&wal).ok_or_else(|| format!("{f}: work arena in unknown allocation {wal}"))?;
        if woff + 4 * nwork * nc > wl.bytes { return Err(format!("{f}: work arena of {nwork} x {nc} words at {woff} exceeds allocation {wal}")); }
        want.push((wal, woff, woff + 4 * nwork * nc, false));
        let nslot = fields.iter().map(|x| x.3).sum();
        Ok(RlBind { nc, nwork, nslot, fields, others, live_slots, tables, work: (wal, woff), ptab: 0, tables_bytes })
    }

    /// The primal pointer table of a bound RUC land launch, in the arena the bind lives in: fields and every other read slot
    /// at their private primal copy (Segs scratch, restored from the pre snapshot before each call), the untouched slots live.
    unsafe fn rl_table(cuda: &Cuda, arena: &mut Arena, segs: &Segs, b: &RlBind) -> Result<u64, String> {
        let mut t = vec![0u64; RL_SLOTS];
        for &(k, al, off, _) in &b.fields { t[k] = segs.scratch(al, off)?; }
        for &(k, al, off, _) in &b.others { t[k] = segs.scratch(al, off)?; }
        for &(k, a) in &b.live_slots { t[k] = a; }
        let bytes: Vec<u8> = t.iter().flat_map(|x| x.to_le_bytes()).collect();
        let a = arena.take(bytes.len())?;
        dev::upload(cuda, a, &bytes)?;
        Ok(a)
    }

    /// RUC land TL (adjoint = false) or AD on a bound launch, in place on the float64 shadows of its 70 fields:
    /// - every primal word is the pre-launch state (private copies, restored here); the entry recomputes the trajectory with
    ///   the forward's own operations, and its float32 tangents and cotangents are the shadows narrowed (exact: the shadow
    ///   words are float32 words widened) and widened back after the call;
    /// - tangent: the entry's primal words of all 70 fields must equal the forward's live post state bit for bit at this
    ///   launch (rlb_cmp), else the call is refused by name;
    /// - adjoint: one chunk of all cells (AD_RUC_LAND_CHUNK cells per chunk otherwise); a tape overflow doubles the tape and
    ///   re-runs from the unchanged cotangents; a slot overflow (nslot) refuses.
    unsafe fn ruc_land(&mut self, cuda: &Cuda, p: &mut Prepared, adjoint: bool) -> Result<Value, String> {
        let Kind::RucLand(b) = &p.kind else { return Err(format!("{}: not a RUC land row", p.function)) };
        let b = (**b).clone();
        if adjoint && self.rl.as_ref().is_some_and(|i| i.ad2.is_some()) { return self.ruc_land_ad2(cuda, p, b); }
        if !adjoint { self.ruc_land_reserve(cuda, &b); }
        let t0 = std::time::Instant::now();
        Self::restore(cuda, p)?;
        let img = self.rl.as_ref().ok_or("RUC land image not loaded")?;
        let (f_tl, f_ad, f_nar, f_wid, f_cmp) = (img.tl, img.ad, img.narrow, img.widen, img.cmp);
        let (nc, nf) = (b.nc, b.fields.len());
        let work_pre = p.segs.pre(b.work.0, b.work.1)?;
        let chunk_env = env_var("AD_RUC_LAND_CHUNK").and_then(|t| t.parse::<usize>().ok()).filter(|&w| w > 0).map(|w| w.min(nc));
        let mut cap = self.rl_cap;
        let mut retries = 0usize;
        loop {
            // Scratch layout: f32 tangents, entry field arrays (AF 16 B / AR 8 B per word), entry work arena, tables, lists,
            // the error word; the adjoint's tape, adjoints, input adjoints, counts and overflow flag.
            let el = if adjoint { 8 } else { 16 };
            let mut sizes: Vec<usize> = b.fields.iter().map(|x| 4 * nc * x.3).collect();            // 0..nf: f32 tangents
            sizes.extend(b.fields.iter().map(|x| el * nc * x.3));                                   // nf..2nf: AF / AR
            sizes.extend([el * b.nwork * nc, 8 * RL_SLOTS, 8 * RL_SLOTS, 24 * nf, 24 * nf, 16, 16]); // work, tables, seg lists, err, diff
            // pk120-recording: unset AD_RUC_LAND_CHUNK, the adjoint chunk is the largest one (all cells when it fits) whose
            // scratch block, with the grow-only buffer's half-again slack, fits the card's free memory less 2 GiB plus the
            // block the buffer already holds. Breakage: x1.40962 (40,962 cells, tape cap 8,181) reverse step 3 asked one
            // 17,156,800,512-byte block and refused "cuMemAlloc scheme scratch ... CUDA status 2" with 25.8 GB free before
            // MYNN's arenas. Cells are independent (one thread per cell, the chunk loop below), so the words are those of one chunk.
            let w = match (adjoint, chunk_env) {
                (_, Some(w)) => w,
                (false, None) => nc,
                (true, None) => {
                    let fixed: usize = sizes.iter().map(|s| (*s).max(1).div_ceil(256) * 256).sum::<usize>() + 16 * 256;
                    let per = 24 * cap + 8 * cap + 8 * b.nslot + 4;
                    let held = self.scratch.get("ruc_land").map(|g| g.bytes).unwrap_or(0);
                    if fixed + per * nc <= held { nc } else {
                        let (mut free, mut total) = (0usize, 0usize);
                        checked("cuMemGetInfo", (cuda.mem_info)(&mut free, &mut total))?;
                        let room = free.saturating_sub(2usize << 30) * 2 / 3;
                        let fit = room.saturating_sub(fixed) / per;
                        if fit >= nc { nc } else { (fit / 32 * 32).max(32).min(nc) }
                    }
                }
            };
            if adjoint { sizes.extend([24 * cap * w, 8 * cap * w, 8 * b.nslot * w, 4 * w, 16, 64]); }
            let mut offs = vec![0usize; sizes.len()];
            let mut total = 0usize;
            for (i, s) in sizes.iter().enumerate() { offs[i] = total; total += (*s).max(1).div_ceil(256) * 256; }
            let base = self.scratch.entry("ruc_land").or_default().need_exact(cuda, total)?; // mynn: fixed per mesh, no half-again headroom (5.50 -> 3.67 GB at x1.163842)
            let v: Vec<u64> = offs.iter().map(|o| base + *o as u64).collect();
            let (wr, tn_t, sh_t, nar_l, cmp_l, err, diff) = (v[2 * nf], v[2 * nf + 1], v[2 * nf + 2], v[2 * nf + 3], v[2 * nf + 4], v[2 * nf + 5], v[2 * nf + 6]);
            let (mut tn, mut sh) = (vec![0u64; RL_SLOTS], vec![0u64; RL_SLOTS]);
            let (mut nl, mut cl): (Vec<u8>, Vec<u8>) = (Vec::new(), Vec::new());
            let mut maxn = 0usize;
            for (j, &(k, al, off, n)) in b.fields.iter().enumerate() {
                tn[k] = v[j]; sh[k] = v[nf + j];
                let words = (nc * n) as i64;
                maxn = maxn.max(nc * n);
                nl.extend(p.segs.shadow(al, off)?.to_le_bytes()); nl.extend(v[j].to_le_bytes()); nl.extend(words.to_le_bytes());
                cl.extend(v[nf + j].to_le_bytes()); cl.extend(p.segs.live(al, off)?.to_le_bytes()); cl.extend(words.to_le_bytes());
            }
            let le = |t: &[u64]| -> Vec<u8> { t.iter().flat_map(|x| x.to_le_bytes()).collect() };
            dev::upload(cuda, tn_t, &le(&tn))?;
            dev::upload(cuda, sh_t, &le(&sh))?;
            dev::upload(cuda, nar_l, &nl)?;
            dev::upload(cuda, cmp_l, &cl)?;
            dev::zero(cuda, err, 16)?;
            dev::zero(cuda, diff, 16)?;
            let lg = [maxn.div_ceil(256).clamp(1, 1024) as u32, nf as u32, 1];
            launch(cuda, f_nar, lg, [256, 1, 1], &[K::P(nar_l)])?;
            let dump = if adjoint { Self::rl_dump_dir() } else { None };
            if let Some(d) = &dump { Self::rl_dump(cuda, p, &b, &v, work_pre, d, "in")?; }
            let common = |last: K| vec![K::P(b.ptab), K::P(b.tables), K::P(wr), K::P(work_pre), i32b(b.nwork as i32), K::P(sh_t), K::P(tn_t),
                                        K::P(err), i32b(nc as i32), last];
            let mut info = json!({"cells": nc, "fields": nf, "nslot": b.nslot});
            if !adjoint {
                launch(cuda, f_tl, grid1(nc, 32), [32, 1, 1], &common(K::P(0)))?;
                launch(cuda, f_cmp, lg, [256, 1, 1], &[K::P(cmp_l), K::P(diff), i32b(4)])?;
                dev::sync(cuda)?;
                let d = u64::from_le_bytes(dev::download(cuda, diff, 8)?[..8].try_into().unwrap());
                if d != 0 { return Err(format!("{}: the entry's primal words differ from the forward's post state in {d} words (identity refused)", p.function)); }
                info["identity_differing_words"] = json!(0);
            } else {
                let (tape, adj, adjin, cnt, ovf, wk) = (v[2 * nf + 7], v[2 * nf + 8], v[2 * nf + 9], v[2 * nf + 10], v[2 * nf + 11], v[2 * nf + 12]);
                dev::zero(cuda, ovf, 16)?;
                let mut tmax = 0i32;
                let mut c0 = 0usize;
                while c0 < nc {
                    let mut s: Vec<u8> = Vec::with_capacity(56);
                    for a in [tape, adj, adjin, cnt, ovf] { s.extend(a.to_le_bytes()); }
                    for x in [cap as i32, w as i32, c0 as i32, b.nslot as i32] { s.extend(x.to_le_bytes()); }
                    dev::upload(cuda, wk, &s)?;
                    dev::zero(cuda, cnt, 4 * w)?;
                    launch(cuda, f_ad, grid1(w, 32), [32, 1, 1], &common(K::P(wk)))?;
                    let c: Vec<i32> = dev::download(cuda, cnt, 4 * w)?.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
                    tmax = tmax.max(c.into_iter().take(nc - c0).max().unwrap_or(0));
                    c0 += w;
                }
                let o = i32::from_le_bytes(dev::download(cuda, ovf, 4)?[..4].try_into().unwrap());
                if o & 2 != 0 { return Err(format!("{}: adjoint slot overflow ({} input words per cell)", p.function, b.nslot)); }
                if o & 1 != 0 {
                    if retries >= 4 { return Err(format!("{}: adjoint tape overflow at {cap} entries per cell after {retries} doublings", p.function)); }
                    cap *= 2; retries += 1; self.rl_cap = cap;
                    continue; // the shadows are unchanged until the widen below: re-narrow and re-run
                }
                info["tape_max_per_cell"] = json!(tmax); info["tape_cap"] = json!(cap); info["tape_retries"] = json!(retries);
                // AD_RUC_LAND_ADCHECK=1 (per-launch dot / replay only, where the live state is the forward's post state):
                // the adjoint's own primal words (AR arrays, float32 at byte 0 of 8) against the forward, as the tangent's are.
                if env_var("AD_RUC_LAND_ADCHECK").as_deref() == Some("1") {
                    launch(cuda, f_cmp, lg, [256, 1, 1], &[K::P(cmp_l), K::P(diff), i32b(2)])?;
                    dev::sync(cuda)?;
                    info["ad_identity_differing_words"] = json!(u64::from_le_bytes(dev::download(cuda, diff, 8)?[..8].try_into().unwrap()));
                }
                info["chunk_cells"] = json!(w);
                if let Some(d) = &dump {
                    Self::rl_dump(cuda, p, &b, &v, work_pre, d, "out")?;
                    std::fs::write(d.join("info.json"), serde_json::to_vec(&info).unwrap()).map_err(|e| e.to_string())?;
                }
            }
            launch(cuda, f_wid, lg, [256, 1, 1], &[K::P(nar_l)])?;
            dev::sync(cuda)?;
            let e = i32::from_le_bytes(dev::download(cuda, err, 4)?[..4].try_into().unwrap());
            info["error_flag"] = json!(e);
            info["ms"] = json!(t0.elapsed().as_secs_f64() * 1e3);
            info["pass"] = json!(if adjoint { "ad" } else { "tl" });
            eprintln!("{}", json!({"event":"native_ruc_land","info":info}));
            return Ok(info);
        }
    }


    /// ruc-land-adjoint-v2: the RUC land adjoint on the exact per-cell tape (oracle_land_ruc_land__ad2, ad_scalar2.cuh).
    /// v1 gave every cell `rl_cap` (8,181) entries of 32 bytes ([cap][W]: 262 KB per cell; the measured mean is ~782), so
    /// x1.40962 asked 17.2 GB in one block and was chunked to fit. Here:
    /// - mode 0 (count) runs the taped forward with every store off and writes each cell's entry count;
    /// - __order puts cells in descending count order and __scan turns the counts into per-thread offsets in that order
    ///   (one block each, on the card) with the total and largest count;
    /// - mode 1 runs v1's adjoint with cell c's entries at off[c] (structure of arrays: pa, pb, (a,b), adj), in as many cell
    ///   chunks as the card's free memory needs (normally one). The executed operations, partials and per-cell reverse order
    ///   are v1's, so the cotangent words are v1's (same reverse words on pkg18m2; RESULT.md).
    /// A count that changes between the two passes is refused by name (the trajectory is the restored pre state both times).
    unsafe fn ruc_land_ad2(&mut self, cuda: &Cuda, p: &mut Prepared, b: RlBind) -> Result<Value, String> {
        let t0 = std::time::Instant::now();
        Self::restore(cuda, p)?;
        let img = self.rl.as_ref().ok_or("RUC land image not loaded")?;
        let (f_nar, f_wid) = (img.narrow, img.widen);
        let (f_ad2, f_scan, f_order) = img.ad2.ok_or("RUC land image has no v2 adjoint")?;
        let (f_caps, lazy) = (img.caps, img.lazy);
        // The store pass runs cells in descending entry-count order (__order), so a warp holds cells with the same work
        // (water, ice, bare land, snow): x1.2562 adjoint 30.5 -> 27.0 ms, 40,992 cells 49.5 -> 40.6 ms, same words.
        // AD_RUC_LAND_SORT=0 keeps the cell order (A/B only).
        let sort = env_var("AD_RUC_LAND_SORT").as_deref() != Some("0");
        let (nc, nf) = (b.nc, b.fields.len());
        let bs = env_var("AD_RUC_LAND_BS").and_then(|t| t.parse::<usize>().ok()).filter(|&w| w > 0 && w <= 256 && w % 32 == 0).unwrap_or(RL2_BS);
        let work_pre = p.segs.pre(b.work.0, b.work.1)?;
        // Fixed scratch (as v1's adjoint): f32 cotangents, AR field arrays (8 B per word), AR work arena, tables, lists,
        // (unused), per-warp offsets, the scan's (total, max), (unused), the ADWork2 block, the cell order, then round 2's
        // capacities, the flags block (err | ovf | stat[4] | redo list, one readback), the two capped-path ADWork2 blocks
        // and the per-warp lane capacities. The AR arrays and the input adjoints are indexed by thread slot (layout L2).
        let nw = nc.div_ceil(32);
        let mut sizes: Vec<usize> = b.fields.iter().map(|x| 4 * nc * x.3).collect();
        sizes.extend(b.fields.iter().map(|x| 8 * nc * x.3));
        sizes.extend([8 * b.nwork * nc, 8 * RL_SLOTS, 8 * RL_SLOTS, 24 * nf, 16, 4 * nc, 8 * nw, 16, 16, RL2_WK, 4 * nc,
                      4 * nc, RL2_FLAGS + 4 * (1 + nc), 2 * RL2_WK, 4 * nw]);
        let mut offs = vec![0usize; sizes.len()];
        let mut total = 0usize;
        for (i, s) in sizes.iter().enumerate() { offs[i] = total; total += (*s).max(1).div_ceil(256) * 256; }
        let base = self.scratch.entry("ruc_land").or_default().need_exact(cuda, total)?; // mynn: fixed per mesh, no half-again headroom (5.50 -> 3.67 GB at x1.163842)
        let v: Vec<u64> = offs.iter().map(|o| base + *o as u64).collect();
        let (wr, tn_t, sh_t, nar_l, off, sout, wk) = (v[2 * nf], v[2 * nf + 1], v[2 * nf + 2], v[2 * nf + 3], v[2 * nf + 6], v[2 * nf + 7], v[2 * nf + 9]);
        let perm = if sort { v[2 * nf + 10] } else { 0 };
        let (capv, flags, wk23, wcap) = (v[2 * nf + 11], v[2 * nf + 12], v[2 * nf + 13], v[2 * nf + 14]);
        let (err, ovf, stat, redo) = (flags, flags + 16, flags + 32, flags + RL2_FLAGS as u64);
        // The per-cell counts live in their own buffer: the next apply sizes its tape from them (capped path).
        let cnt = self.scratch.entry("ruc_land_cnt").or_default().need(cuda, 4 * nc)?;
        let (mut tn, mut sh) = (vec![0u64; RL_SLOTS], vec![0u64; RL_SLOTS]);
        let mut nl: Vec<u8> = Vec::new();
        let mut maxn = 0usize;
        for (j, &(k, al, off, n)) in b.fields.iter().enumerate() {
            tn[k] = v[j]; sh[k] = v[nf + j];
            let words = (nc * n) as i64;
            maxn = maxn.max(nc * n);
            nl.extend(p.segs.shadow(al, off)?.to_le_bytes()); nl.extend(v[j].to_le_bytes()); nl.extend(words.to_le_bytes());
        }
        let le = |t: &[u64]| -> Vec<u8> { t.iter().flat_map(|x| x.to_le_bytes()).collect() };
        dev::upload(cuda, tn_t, &le(&tn))?;
        dev::upload(cuda, sh_t, &le(&sh))?;
        dev::upload(cuda, nar_l, &nl)?;
        let lg = [maxn.div_ceil(256).clamp(1, 1024) as u32, nf as u32, 1];
        // ADWork2 (ad_scalar2.cuh): pa, pb, ab, adj, adjin, off (per warp), cnt, ovf, perm, base, W, c0, nslot, mode,
        // then the capped path's cap, redo, stat, pool (zero in modes 0 and 1, which never read them) and wcap.
        let wk2 = |ptrs: [u64; 5], base_e: i64, w: usize, c0: usize, mode: i32, perm: u64, pool: i64| -> Vec<u8> {
            let mut s: Vec<u8> = Vec::with_capacity(RL2_WK);
            for a in ptrs { s.extend(a.to_le_bytes()); }
            for a in [off, cnt, ovf, perm] { s.extend(a.to_le_bytes()); }
            s.extend(base_e.to_le_bytes());
            for x in [w as i32, c0 as i32, b.nslot as i32, mode] { s.extend(x.to_le_bytes()); }
            let tail = if mode >= 2 { [capv, redo, stat] } else { [0; 3] };
            for a in tail { s.extend(a.to_le_bytes()); }
            s.extend((if mode >= 2 { pool } else { 0 }).to_le_bytes());
            s.extend(wcap.to_le_bytes());
            debug_assert_eq!(s.len(), RL2_WK);
            s
        };
        let common = |last: K| vec![K::P(b.ptab), K::P(b.tables), K::P(wr), K::P(work_pre), i32b(b.nwork as i32), K::P(sh_t), K::P(tn_t),
                                    K::P(err), i32b(nc as i32), last];
        // Tape bytes: 24 B per entry (lazy: the 16-byte entry and its adjoint) or 32 B (pa, pb, (a,b), adj), plus the
        // input adjoints (8 B per slot per cell).
        let eb = if lazy { 24 } else { 32 };
        let tape_bytes = |e: usize, cells: usize| eb * e + 8 * b.nslot * cells + 4 * 256;
        let rnd = |x: usize| x.div_ceil(256) * 256;
        // pa, pb, ab, adj, adjin of a tape of e entries at tb
        let tptr = |tb: u64, e: usize| -> [u64; 5] {
            if lazy { let adj = tb + rnd(16 * e) as u64; [tb, 0, 0, adj, adj + rnd(8 * e) as u64] }
            else { [tb, tb + rnd(8 * e) as u64, tb + 2 * rnd(8 * e) as u64, tb + 3 * rnd(8 * e) as u64, tb + 4 * rnd(8 * e) as u64] }
        };
        let room_for = |held: usize, need: usize| -> Result<usize, String> {
            if need <= held { return Ok(need); }
            let (mut free, mut tot) = (0usize, 0usize);
            checked("cuMemGetInfo", (cuda.mem_info)(&mut free, &mut tot))?;
            Ok(held + free.saturating_sub(2usize << 30) * 2 / 3)
        };
        let dump = Self::rl_dump_dir();
        let mut info = json!({"cells": nc, "fields": nf, "nslot": b.nslot, "adjoint": "v2", "block": bs, "sorted": sort, "lazy": lazy, "entry_bytes": eb});
        dev::zero(cuda, flags, RL2_FLAGS + 4)?;
        launch(cuda, f_nar, lg, [256, 1, 1], &[K::P(nar_l)])?;
        if let Some(d) = &dump { Self::rl_dump(cuda, p, &b, &v, work_pre, d, "in")?; }

        // Capped path (round 2, opt-in): no count pass. The previous apply's true counts (on the card) give each cell
        // cap = 2 x count + 64 entries (__caps; pkg18m2's heavy land cells flip between ~2,400-3,000 and ~4,300-5,700
        // entries from one step to the next, up to +93 %, and no cell outgrew 2 x + 64 in any of the six dumped launch
        // pairs). __order and __scan run on the caps, mode 2 stores and reverses every cell that fits its cap and records
        // every cell's true count; a cell that outgrew its cap writes no cotangent word and joins the redo list, which
        // mode 3 runs on an exact tape past the capped region. Words: the same executed operations, partials and per-cell
        // reverse order as the counted path. One readback per apply (flags: error, overflow, totals, redo count). If the
        // redo room runs out (flag 4) the apply re-narrows the untouched cotangents and runs the counted path.
        // Opt-in (AD_RUC_LAND_CAPPED=1); the counted path is the default. Measured on a test machine RTX PRO 6000 (the dumped
        // pkg18m2 launch, min of 45; round5/ruc-land-adjoint-v2/RESULT.md): the capped store pass writes the same entries
        // into a 2.2x larger pool, and that costs more than the count pass it saves below ~100k cells per card
        // (2,562 cells 38.3 against 30.4 ms, 40,992 cells 35.0 against 31.4 ms); at 163,968 cells it wins (61.4 against 75.0).
        let prev = self.rl_prev.filter(|&(n, _, _, at)| n == nc && at == cnt);
        let capped_on = env_var("AD_RUC_LAND_CAPPED").as_deref() == Some("1") && sort && !f_caps.is_null();
        let mut done = false;
        if let (true, Some((_, ptot, pmax, _))) = (capped_on, prev) {
            let cap_sum = 2 * ptot + RL2_CAP_ADD * nc;
            // warp padding: a warp holds 32 x its largest cap; the sorted caps keep that within ~1/4 of the caps' sum
            let pool = cap_sum * 5 / 4 + (8 * (2 * pmax + RL2_CAP_ADD)).max(cap_sum / 16);
            let need = tape_bytes(pool, nc) + 4 * 256;
            let held = self.scratch.get("ruc_land_tape").map(|g| g.bytes).unwrap_or(0);
            if need <= room_for(held, need)? {
                let tb = self.scratch.entry("ruc_land_tape").or_default().need_slack(cuda, need, true)?;
                let tp = tptr(tb, pool);
                let mut w23 = wk2(tp, 0, nc, 0, 2, perm, pool as i64);
                w23.extend(wk2(tp, 0, nc, 0, 3, perm, pool as i64));
                dev::upload(cuda, wk23, &w23)?;
                launch(cuda, f_caps, grid1(nc, 256), [256, 1, 1], &[K::P(cnt), K::P(capv), i32b(nc as i32), i32b(RL2_CAP_ADD as i32), i32b(0)])?;
                launch(cuda, f_order, [1, 1, 1], [1024, 1, 1], &[K::P(capv), K::P(perm), i32b(nc as i32), i32b(7)])?;
                // the scan's (total, max) land in stat[2..4]: stat[2] is the redo cursor, starting at the capped region's
                // end (mode 2 refuses with flag 4 when that end is past the pool)
                launch(cuda, f_scan, [1, 1, 1], [1024, 1, 1], &[K::P(capv), K::P(perm), K::P(off), K::P(wcap), K::P(stat + 16), i32b(nc as i32)])?;
                launch(cuda, f_ad2, grid1(nc, bs), [bs as u32, 1, 1], &common(K::P(wk23)))?;
                launch(cuda, f_ad2, grid1(nc, bs), [bs as u32, 1, 1], &common(K::P(wk23 + RL2_WK as u64)))?;
                dev::sync(cuda)?;
                let fl = dev::download(cuda, flags, RL2_FLAGS + 4)?;
                let i32at = |o: usize| i32::from_le_bytes(fl[o..o + 4].try_into().unwrap());
                let u64at = |o: usize| u64::from_le_bytes(fl[o..o + 8].try_into().unwrap()) as usize;
                // stat: [0] true entries, [1] largest true count, [2] cursor (capped region end + redo entries), [3] largest cap
                let (e, o, entries, tmax, cur, nredo) = (i32at(0), i32at(16), u64at(32), u64at(40), u64at(48), i32at(RL2_FLAGS));
                if o & 2 != 0 { return Err(format!("{}: adjoint slot overflow ({} input words per cell)", p.function, b.nslot)); }
                if o & 1 != 0 { return Err(format!("{}: a cell's adjoint tape count changed inside the capped store pass (refused)", p.function)); }
                info["path"] = json!(if o & 4 != 0 { "capped_redo_room_out_then_counted" } else { "capped" });
                info["redo_cells"] = json!(nredo);
                info["pool_entries"] = json!(pool);
                info["cap_entries"] = json!(cap_sum);
                info["tape_bytes_allocated"] = json!(need);
                if o & 4 == 0 {
                    done = true;
                    info["tape_held_entries"] = json!(cur);
                    info["tape_entries"] = json!(entries); info["tape_max_per_cell"] = json!(tmax);
                    info["tape_mean_per_cell"] = json!(entries as f64 / nc as f64);
                    info["tape_bytes"] = json!(tape_bytes(entries, nc));
                    info["chunks"] = json!(1); info["chunk_cells"] = json!(nc);
                    info["error_flag"] = json!(e);
                    self.rl_prev = Some((nc, entries, tmax, cnt));
                } else {
                    // the redo cells kept their cotangents untouched, the others did not: start over from the shadows
                    dev::zero(cuda, flags, RL2_FLAGS + 4)?;
                    launch(cuda, f_nar, lg, [256, 1, 1], &[K::P(nar_l)])?;
                }
            } else {
                info["path"] = json!("counted_capped_no_room");
            }
        }
        if !done {
            // Counted path: mode 0 counts, then the scan (total entries, largest count); one 16-byte read.
            dev::zero(cuda, cnt, 4 * nc)?;
            dev::upload(cuda, wk, &wk2([0; 5], 0, nc, 0, 0, 0, 0))?;
            launch(cuda, f_ad2, grid1(nc, bs), [bs as u32, 1, 1], &common(K::P(wk)))?;
            if sort { launch(cuda, f_order, [1, 1, 1], [1024, 1, 1], &[K::P(cnt), K::P(perm), i32b(nc as i32), i32b(6)])?; }
            launch(cuda, f_scan, [1, 1, 1], [1024, 1, 1], &[K::P(cnt), K::P(perm), K::P(off), K::P(wcap), K::P(sout), i32b(nc as i32)])?;
            let so: Vec<i64> = dev::download(cuda, sout, 16)?.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
            // entries: the tape held (each warp 32 x its largest count); tmax the largest count
            let (entries, tmax) = (so[0].max(0) as usize, so[1]);
            let o = i32::from_le_bytes(dev::download(cuda, ovf, 4)?[..4].try_into().unwrap());
            if o & 2 != 0 { return Err(format!("{}: adjoint slot overflow ({} input words per cell)", p.function, b.nslot)); }
            // Cell chunks: one when the tape and the input adjoints fit the room the grow-only buffer leaves (half-again
            // slack, 2 GiB kept free); otherwise the largest prefix chunks that fit.
            let held = self.scratch.get("ruc_land_tape").map(|g| g.bytes).unwrap_or(0);
            let need_all = tape_bytes(entries, nc);
            let room = room_for(held, need_all)?;
            // Chunks are whole warps (layout L2: a warp's lanes share its interleaved region).
            let chunks: Vec<(usize, usize, i64, i64)> = if need_all <= room {
                vec![(0, nc, 0, entries as i64)]
            } else {
                let offh: Vec<i64> = dev::download(cuda, off, 8 * nw)?.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
                let at = |w: usize| if w == nw { entries as i64 } else { offh[w] };
                let cells = |w0: usize, w1: usize| (32 * w1).min(nc) - 32 * w0;
                let mut out = Vec::new();
                let mut w0 = 0usize;
                while w0 < nw {
                    let mut w1 = w0 + 1;
                    while w1 < nw && tape_bytes((at(w1 + 1) - at(w0)) as usize, cells(w0, w1 + 1)) <= room { w1 += 1; }
                    if tape_bytes((at(w1) - at(w0)) as usize, cells(w0, w1)) > room {
                        return Err(format!("{}: one warp's adjoint tape ({} entries) does not fit the card ({room} bytes of room)", p.function, at(w1) - at(w0))); }
                    out.push((32 * w0, (32 * w1).min(nc), at(w0), at(w1)));
                    w0 = w1;
                }
                out
            };
            let biggest = chunks.iter().map(|&(c0, c1, e0, e1)| tape_bytes((e1 - e0) as usize, c1 - c0)).max().unwrap_or(0);
            // The next apply takes the capped path: size the tape for it now (when it fits), so the grow-only buffer does
            // not retire this block one apply later.
            let want = if capped_on && chunks.len() == 1 {
                let cs = 2 * entries + RL2_CAP_ADD * nc;
                let w2 = tape_bytes(cs * 5 / 4 + (8 * (2 * tmax.max(0) as usize + RL2_CAP_ADD)).max(cs / 16), nc) + 4 * 256;
                if w2 <= room_for(held, w2)? { biggest.max(w2) } else { biggest }
            } else { biggest };
            let tb = self.scratch.entry("ruc_land_tape").or_default().need_slack(cuda, want, true)?;
            for &(c0, c1, e0, e1) in &chunks {
                let (w, e) = (c1 - c0, (e1 - e0) as usize);
                dev::upload(cuda, wk, &wk2(tptr(tb, e), e0, w, c0, 1, perm, 0))?;
                launch(cuda, f_ad2, grid1(w, bs), [bs as u32, 1, 1], &common(K::P(wk)))?;
            }
            let o = i32::from_le_bytes(dev::download(cuda, ovf, 4)?[..4].try_into().unwrap());
            if o & 1 != 0 { return Err(format!("{}: a cell's adjoint tape count changed between the count and store passes (refused)", p.function)); }
            if info.get("path").is_none() { info["path"] = json!("counted"); }
            info["tape_held_entries"] = json!(entries); info["tape_max_per_cell"] = json!(tmax);
            info["tape_bytes"] = json!(tape_bytes(entries, nc));
            info["chunks"] = json!(chunks.len()); info["chunk_cells"] = json!(chunks.first().map(|c| c.1 - c.0).unwrap_or(0));
            self.rl_prev = Some((nc, entries, tmax.max(0) as usize, cnt));
        }
        if let Some(d) = &dump {
            Self::rl_dump(cuda, p, &b, &v, work_pre, d, "out")?;
            std::fs::write(d.join("info.json"), serde_json::to_vec(&info).unwrap()).map_err(|e| e.to_string())?;
        }
        launch(cuda, f_wid, lg, [256, 1, 1], &[K::P(nar_l)])?;
        if info.get("error_flag").is_none() {
            dev::sync(cuda)?;
            let e = i32::from_le_bytes(dev::download(cuda, err, 4)?[..4].try_into().unwrap());
            info["error_flag"] = json!(e);
        }
        info["ms"] = json!(t0.elapsed().as_secs_f64() * 1e3);
        info["pass"] = json!("ad");
        eprintln!("{}", json!({"event":"native_ruc_land","info":info}));
        Ok(info)
    }

    /// ruc-land-adjoint-v2 round 2: at a tangent apply, reserve the v2 adjoint's tape (RL2_RESERVE_ENTRIES entries per cell
    /// and the input adjoints, plus 1/8) and its count buffer, while the card still has room. The pk120 1-step window on a
    /// 96 GB card (rucl2-P120v3c) had no room left by the first reverse step: the adjoint refused with 0 bytes of room.
    /// The adjoint still grows the block when a launch needs more. A reservation that cannot be made is logged, not an error
    /// (the adjoint then sizes from what is free, as before). AD_RUC_LAND_RESERVE=0 turns it off (A/B only).
    unsafe fn ruc_land_reserve(&mut self, cuda: &Cuda, b: &RlBind) {
        let Some(img) = self.rl.as_ref() else { return };
        if img.ad2.is_none() || env_var("AD_RUC_LAND_RESERVE").as_deref() == Some("0") { return; }
        let eb = if img.lazy { 24 } else { 32 };
        let want = eb * RL2_RESERVE_ENTRIES * b.nc + 8 * b.nslot * b.nc + 8 * 256;
        if self.scratch.get("ruc_land_tape").map(|g| g.bytes).unwrap_or(0) >= want { return; }
        let r = self.scratch.entry("ruc_land_tape").or_default().need_slack(cuda, want, false)
            .and_then(|_| self.scratch.entry("ruc_land_cnt").or_default().need(cuda, 4 * b.nc));
        eprintln!("{}", json!({"event": "ruc_land_tape_reserved", "cells": b.nc, "bytes": want, "ok": r.is_ok(), "error": r.err()}));
    }

    /// ruc-land-adjoint-v2 (diagnostic, default off): AD_RUC_LAND_DUMP=<dir> writes the first AD_RUC_LAND_DUMP_N (1) RUC land
    /// adjoint launches' inputs (pre-launch primal words of every read slot, the tables allocation, work_pre, the float32
    /// cotangents in) and outputs (the float32 cotangents out) to <dir>/<n>/, for the standalone per-launch harness
    /// (tree/ad/phys/held/oracle_land_ruc_land/harness/rl_ad_bench.py). Nothing is written to a device word.
    fn rl_dump_dir() -> Option<PathBuf> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let d = env_var("AD_RUC_LAND_DUMP")?;
        let max = env_var("AD_RUC_LAND_DUMP_N").and_then(|t| t.parse().ok()).unwrap_or(1usize);
        let i = N.fetch_add(1, Ordering::SeqCst);
        if i >= max { return None; }
        let p = PathBuf::from(d).join(format!("{i}"));
        std::fs::create_dir_all(&p).ok()?;
        Some(p)
    }
    unsafe fn rl_dump(cuda: &Cuda, p: &Prepared, b: &RlBind, v: &[u64], work_pre: u64, d: &Path, stage: &str) -> Result<(), String> {
        dev::sync(cuda)?;
        let w = |name: String, bytes: Vec<u8>| std::fs::write(d.join(name), bytes).map_err(|e| e.to_string());
        let nc = b.nc;
        for (j, &(k, _, _, n)) in b.fields.iter().enumerate() { w(format!("bar_{stage}_{k}.bin"), dev::download(cuda, v[j], 4 * nc * n)?)?; }
        if stage == "in" {
            for &(k, al, off, n) in &b.fields { w(format!("p_{k}.bin"), dev::download(cuda, p.segs.scratch(al, off)?, 4 * nc * n)?)?; }
            for &(k, al, off, bytes) in &b.others { w(format!("o_{k}.bin"), dev::download(cuda, p.segs.scratch(al, off)?, bytes)?)?; }
            w("tables.bin".into(), dev::download(cuda, b.tables, b.tables_bytes)?)?;
            w("work_pre.bin".into(), dev::download(cuda, work_pre, 4 * b.nwork * nc)?)?;
            let meta = json!({"nc": nc, "nwork": b.nwork, "nslot": b.nslot, "fields": b.fields.iter().map(|x| json!([x.0, x.3])).collect::<Vec<_>>(),
                              "others": b.others.iter().map(|x| json!([x.0, x.3])).collect::<Vec<_>>(),
                              "live_slots": b.live_slots.iter().map(|x| x.0).collect::<Vec<_>>(), "tables_bytes": b.tables_bytes});
            w("meta.json".into(), serde_json::to_vec_pretty(&meta).unwrap())?;
        }
        Ok(())
    }

    // --------------------------------------------------------------------------------------- P7
    /// rlw_taumol_batched (RRTMG-LW gas optics) on a bound launch. Every primal argument is a private copy of the pre
    /// state (restore), so the derivative entries never write live plan words; the tangent's augmented primal writes
    /// taug/fracs into those copies, and with `identity` (the launch dot) they must equal the forward's live post state
    /// word for word, or the call is refused by name.
    /// - Tangent (in place, float64 shadows): dtaug and dfracs are set (the forward overwrites every word of both:
    ///   140 g-points per column and layer); dfs and dwx (inputs, not written by the forward) are left as they are.
    /// - Adjoint: one thread per (column, layer) runs the 16 bands' g-points in a fixed order (deterministic, no atomics),
    ///   accumulates into bfs/bwx and consumes (zeroes) btaug/bfracs. A tape overflow (P7_TMAX) refuses by name.
    /// - Frozen coefficients: the integer inputs (laytrop_v, and isv = jp/jt/jt1/indself/indfor/indminor from
    ///   rlw_setcoef) and every switch on a primal value inside the band bodies (the eta binning `js = 1 + int(specmult)`,
    ///   the `specparm < 0.125` / `> 0.875` stencil choice, the lower/upper atmosphere split at laytrop) take the
    ///   trajectory's branch; the tables (bandptrs, chi_mls) are constants. The derivative is the exact executed-branch
    ///   derivative; it does not exist on a switch point.
    unsafe fn p7(&mut self, cuda: &Cuda, p: &mut Prepared, adjoint: bool, identity: bool) -> Result<Value, String> {
        let row = self.rows[&p.function].clone();
        let (f_tl, f_ad, fbytes) = { let i = self.p7.as_ref().ok_or("P7 rlw_taumol_batched image not loaded")?; (i.tl, i.ad, i.frame) };
        let _sh = |name: &str| -> Result<K, String> { let (a, o) = Self::dev_arg(&row, p, name)?; Ok(K::P(p.segs.shadow(a, o)?)) };
        let scr = |name: &str| -> Result<K, String> { let (a, o) = Self::dev_arg(&row, p, name)?; Ok(K::P(p.segs.scratch(a, o)?)) };
        // A constant table bound as a module global arrives as its live address by value (see the bind check).
        let cst = |name: &str| -> Result<K, String> {
            let a = row.arg(name)?;
            match p.args.get(a.pos) { Some(Argument::Bytes(b)) if b.len() == 8 && a.role == "const" => Ok(K::B(b.clone())), _ => scr(name) } };
        let dim = |k: &str| -> Result<usize, String> { p.env.get(k).copied().map(|v| v as usize).ok_or_else(|| format!("{}: launch has no {k}", row.function)) };
        let (ncol, nl) = (dim("ncol")?, dim("nl")?);
        Self::restore(cuda, p)?;
        let head = vec![Self::sc(&row, p, "ncol")?, Self::sc(&row, p, "nl")?, scr("laytrop_v")?, scr("fs")?, scr("isv")?, scr("wx")?,
                        cst("chi_mls")?, Self::sc(&row, p, "oneminus")?, scr("bandptrs")?];
        // step6-radiation: the fs[coldry] slot (GPU_FSLOTS index 1) is in molecules/cm2, so its window shadow is in the
        // molecule unit of rlw_inatm / rlw_setcoef (tangent stored t * 2^-72, cotangent b * 2^72). These float64
        // shadows are brought to true units around both entries and back after (exact: powers of two), so the
        // entries are unchanged. THE BREAKAGE THIS PREVENTS: the true cotangent of that slot is 1e-45..1e-38 at the
        // window's y-scale 1e-20, f32 subnormal in 35,437 of 35,479 nonzero words (s6r-c1).
        let slot = {
            let (a, o) = Self::dev_arg(&row, p, "fs")?;
            (p.segs.shadow(a, o + 4 * ncol * nl)?, ncol * nl)
        };
        let unit = |c: f64| -> Result<(), String> {
            let v: Vec<u8> = dev::download(cuda, slot.0, 8 * slot.1)?.chunks_exact(8)
                .flat_map(|w| (f64::from_le_bytes(w.try_into().unwrap()) * c).to_le_bytes()).collect();
            dev::upload(cuda, slot.0, &v)
        };
        // Stored -> true: a tangent is stored t * 2^-72 (times 2^72), a cotangent b * 2^72 (times 2^-72); back after.
        let to_true = if adjoint { 2f64.powi(-72) } else { 2f64.powi(72) };
        dev::sync(cuda)?;
        unit(to_true)?;
        let r = self.p7_body(cuda, p, adjoint, identity, &row, head, f_tl, f_ad, fbytes, ncol, nl);
        dev::sync(cuda)?;
        unit(1.0 / to_true)?;
        r
    }
    #[allow(clippy::too_many_arguments)]
    unsafe fn p7_body(&mut self, cuda: &Cuda, p: &mut Prepared, adjoint: bool, identity: bool, row: &Row, head: Vec<K>,
                      f_tl: *mut c_void, f_ad: *mut c_void, fbytes: usize, ncol: usize, nl: usize) -> Result<Value, String> {
        let row = row.clone();
        let sh = |name: &str| -> Result<K, String> { let (a, o) = Self::dev_arg(&row, p, name)?; Ok(K::P(p.segs.shadow(a, o)?)) };
        let scr = |name: &str| -> Result<K, String> { let (a, o) = Self::dev_arg(&row, p, name)?; Ok(K::P(p.segs.scratch(a, o)?)) };
        if !adjoint {
            let mut a = head;
            a.extend([scr("taug")?, scr("fracs")?, sh("fs")?, sh("wx")?, sh("taug")?, sh("fracs")?]);
            launch(cuda, f_tl, p.grid, p.block, &a)?;
            if !identity { return Ok(Value::Null); }
            dev::sync(cuda)?;
            let mut out = json!({});
            let mut bad = 0usize;
            for name in ["taug", "fracs"] {
                let (al, o) = Self::dev_arg(&row, p, name)?;
                let n = row.extent_bytes(row.arg(name)?, &p.env)?;
                let s = p.segs.find(al, o)?;
                let (x, y) = (dev::download(cuda, p.segs.scratch(al, o)?, n)?, dev::download(cuda, s.live + (o - s.lo) as u64, n)?);
                let d = x.chunks_exact(4).zip(y.chunks_exact(4)).filter(|(u, v)| u != v).count();
                bad += d;
                out[format!("augmented_primal_differing_words_{name}")] = json!(d);
            }
            out["augmented_primal_words"] = json!(2 * 140 * ncol * nl);
            if bad != 0 { return Err(format!("{}: augmented primal differs from the forward's post state in {bad} words", row.function)); }
            return Ok(out);
        }
        let fail = self.scratch.entry("p7_fail").or_default().need(cuda, 256)?;
        dev::zero(cuda, fail, 4)?;
        let mut a = head;
        a.extend([sh("fs")?, sh("wx")?, sh("taug")?, sh("fracs")?, K::P(fail)]);
        // G4: the adjoint's tapes live in a frame arena sized for the launch's threads (capped grid, grid-stride loop).
        let threads = (ncol * nl).div_ceil(64).max(1).min(P7_AD_THREADS / 64) * 64;
        let frame = self.scratch.entry("p7_frame").or_default().need(cuda, threads * fbytes)?;
        self.p7.as_ref().unwrap()._m.set_global(cuda, "p7_frame_base", &frame.to_le_bytes())?;
        launch(cuda, f_ad, [(threads / 64) as u32, 1, 1], [64, 1, 1], &a)?;
        dev::sync(cuda)?;
        let c = i32::from_le_bytes(dev::download(cuda, fail, 4)?[..4].try_into().unwrap());
        if c != 0 { return Err(format!("{}: adjoint tape overflow (code {c}; P7_TMAX entries per column, layer and g-point)", row.function)); }
        Ok(Value::Null)
    }

    // --------------------------------------------------------------------------------------- P7 SW
    /// rsw_taumol_b (RRTMG-SW gas optics, all 14 bands 16-29, 112 g-points) on a bound launch. Every primal argument is a
    /// private copy of the pre state (restore), so the entries never write live plan words; the tangent's augmented
    /// primal writes taug/taur into those copies, and with `identity` (the launch dot) they must equal the forward's live
    /// post state word for word, or the call is refused by name.
    /// - Tangent (in place, float64 shadows): dtaug and dtaur are set (the forward overwrites every word of both: 112
    ///   g-points per column and layer); the 14 active inputs' shadows (not written by the forward) are left as they are.
    /// - Adjoint: one thread per (column, layer) runs the 112 g-points in ascending order (deterministic, no atomics on
    ///   derivative words), adds into the 14 input cotangents and consumes (zeroes) btaug/btaur. A tape overflow
    ///   (P7SW_TMAX) or a body pattern outside the generator's assumptions refuses by name.
    /// - Frozen coefficients: the integer inputs (laytrop, ngb, and jp/jt/jt1/indself/indfor from rsw_setcoef_b) and every
    ///   switch on a primal value in the band body take the trajectory's branch: the eta bin js = 1 + (int)specmult (its
    ///   fraction fs = specmult - truncf(specmult) is differentiated), the limiter-like clamp `if (specparm >=
    ///   RSW_C_ONEMINUS) specparm = RSW_C_ONEMINUS` (clamped: zero derivative, the limit is a constant), and the lower /
    ///   upper split lay <= laytrop. The k-coefficient table `tab` is a constant. The derivative is the exact
    ///   executed-branch derivative; it does not exist on a switch point.
    unsafe fn p7sw(&mut self, cuda: &Cuda, p: &mut Prepared, adjoint: bool, identity: bool) -> Result<Value, String> {
        const ACTIVE: [&str; 14] = ["colh2o", "colco2", "colch4", "colo2", "colo3", "colmol", "fac00", "fac01", "fac10", "fac11",
                                    "selffac", "selffrac", "forfac", "forfrac"];
        let row = self.rows[&p.function].clone();
        let (f_tl, f_ad) = { let i = self.p7sw.as_ref().ok_or("P7 rsw_taumol_b image not loaded")?; (i.tl, i.ad) };
        let sh = |name: &str| -> Result<K, String> { let (a, o) = Self::dev_arg(&row, p, name)?; Ok(K::P(p.segs.shadow(a, o)?)) };
        let dim = |k: &str| -> Result<usize, String> { p.env.get(k).copied().map(|v| v as usize).ok_or_else(|| format!("{}: launch has no {k}", row.function)) };
        let (ncol, nl) = (dim("ncol")?, dim("nlayers")?);
        Self::restore(cuda, p)?;
        // The forward's own argument list on the private copies: ncol, nlayers, the 22 inputs, taug, taur.
        let fwd = Self::fwd(&row, p)?;
        if fwd.len() != 26 { return Err(format!("{}: row has {} arguments, the rsw_taumol_b entries take 26", row.function, fwd.len())); }
        if !adjoint {
            let mut a = fwd;
            for n in ACTIVE { a.push(sh(n)?); }
            a.extend([sh("taug")?, sh("taur")?]);
            launch(cuda, f_tl, p.grid, p.block, &a)?;
            if !identity { return Ok(Value::Null); }
            dev::sync(cuda)?;
            let mut out = json!({});
            let mut bad = 0usize;
            for name in ["taug", "taur"] {
                let (al, o) = Self::dev_arg(&row, p, name)?;
                let n = row.extent_bytes(row.arg(name)?, &p.env)?;
                let s = p.segs.find(al, o)?;
                let (x, y) = (dev::download(cuda, p.segs.scratch(al, o)?, n)?, dev::download(cuda, s.live + (o - s.lo) as u64, n)?);
                let d = x.chunks_exact(4).zip(y.chunks_exact(4)).filter(|(u, v)| u != v).count();
                bad += d;
                out[format!("augmented_primal_differing_words_{name}")] = json!(d);
            }
            out["augmented_primal_words"] = json!(2 * 112 * ncol * nl);
            if bad != 0 { return Err(format!("{}: augmented primal differs from the forward's post state in {bad} words", row.function)); }
            return Ok(out);
        }
        let fail = self.scratch.entry("p7sw_fail").or_default().need(cuda, 256)?;
        dev::zero(cuda, fail, 8)?;
        let mut a: Vec<K> = fwd.into_iter().take(24).collect();
        for n in ACTIVE { a.push(sh(n)?); }
        a.extend([sh("taug")?, sh("taur")?, K::P(fail)]);
        launch(cuda, f_ad, grid1(ncol * nl, 64), [64, 1, 1], &a)?;
        dev::sync(cuda)?;
        let f = dev::download(cuda, fail, 8)?;
        let (c, high) = (i32::from_le_bytes(f[..4].try_into().unwrap()), i32::from_le_bytes(f[4..8].try_into().unwrap()));
        if c != 0 {
            let why = match c { 1 => "tape overflow (P7SW_TMAX entries per column, layer and g-point)", 4 => "an input read off the thread's layer",
                                5 => "an output written twice in one g-point", 6 => "an output word that is not (layer, g-point) or not written", _ => "unknown" };
            return Err(format!("{}: adjoint refused, code {c}: {why} (tape high-water {high})", row.function));
        }
        Ok(json!({"tape_high_water": high}))
    }

    // --------------------------------------------------------------------------------------- P3
    /// MYNN-EDMF whole launch in 32-column-aligned tiles. mode 0 tangent, 1 adjoint. tin / ybar = the shadow
    /// snapshot taken just before (separate-buffer entries), tout / xbar = the shadow (in-place convention).
    unsafe fn p3(&mut self, cuda: &Cuda, p: &mut Prepared, mode: i32, identity: bool) -> Result<Value, String> {
        // inner-model: AD_IM_BL=fixedk runs the held-coefficient linear diffusion in place of the scheme's derivative.
        if im_bl_on() { return self.im_p3_fixedk(cuda, p, mode); }
        let mut init_taped = false;
        if mode == 1 && self.p3.as_ref().and_then(|i| i.direct.as_ref()).is_some() {
            // The joined backward has no f_mym_initialize: an initflag launch (the first step of a cold start) reverses on
            // the taped route, which has it, and says so. Every other launch takes the tape-free route.
            let init = {
                let row = &self.rows["rrfs_mynnedmf_run_l16"];
                let a = row.arg("initflag")?;
                let b = scalar_bytes(&p.args[a.pos]).ok_or("MYNN direct: initflag is not a scalar")?;
                i32::from_le_bytes(b.get(..4).ok_or("MYNN direct: initflag width")?.try_into().unwrap()) != 0
            };
            // tape-free fall-back (3): the taped route's card budget is measured before the direct route takes its scratch.
            // THE BREAKAGE: measured first at a fallback, after the direct buffers, it was 0 (384-column tiles, 9-17 s a launch).
            if self.p3_budget.is_none() { self.p3_budget_init(cuda)?; }
            if !init {
                if let Some(mut v) = self.p3_direct(cuda, p, identity)? {
                    // tape-free fall-back (2): a column the join refused is reversed alone on the taped route (one-column
                    // launch at its own batch_first); every other column of the launch stays tape-free. Columns are
                    // independent (G5) and the direct route left the refused column's shadow words untouched (= ybar).
                    let redo: Vec<i64> = v["taped_columns"].as_array().map(|a| a.iter().filter_map(|x| x.as_i64()).collect()).unwrap_or_default();
                    if redo.is_empty() { return Ok(v); }
                    let row = &self.rows["rrfs_mynnedmf_run_l16"];
                    let (pf, pc) = (row.arg("batch_first")?.pos, row.arg("batch_count")?.pos);
                    let env0 = p.env.clone();
                    let first = *env0.get("batch_first").ok_or("MYNN direct batch_first")? as i64;
                    let t0 = std::time::Instant::now();
                    let mut res: Result<(), String> = Ok(());
                    // a one-column launch reuses the adjoint budget (p3_fallback, read in p3_taped).
                    let budget_kept = self.p3_ad_budget.is_some();
                    self.p3_fallback = true;
                    for &c in &redo {
                        let a0 = std::mem::replace(&mut p.args[pf], Argument::Bytes(((first + c) as i32).to_le_bytes().to_vec()));
                        let a1 = std::mem::replace(&mut p.args[pc], Argument::Bytes(1i32.to_le_bytes().to_vec()));
                        p.env.insert("batch_first".into(), (first + c) as f64);
                        p.env.insert("batch_count".into(), 1.0);
                        let r = self.p3_taped(cuda, p, mode, identity);
                        p.args[pf] = a0; p.args[pc] = a1; p.env = env0.clone();
                        if let Err(e) = r { res = Err(e); break; }
                    }
                    self.p3_fallback = false;
                    res?;
                    let ms = t0.elapsed().as_secs_f64() * 1e3;
                    eprintln!("{}", json!({"event":"mynn_direct_column_taped","columns":redo,"ms":ms,"budget":if budget_kept { "kept" } else { "read" }}));
                    v["taped_ms"] = json!(ms);
                    return Ok(v);
                }
            }
            else { eprintln!("{}", json!({"event":"mynn_direct_taped","why":"initflag launch: f_mym_initialize is not in the joined backward, so this launch reverses on the taped route"})); }
            // tape-free fall-back (3): a whole launch on the taped route gets the card memory the direct route held.
            // THE BREAKAGE: a first step (initflag, whole launch taped) "cuMemAlloc scheme scratch 1398800384 bytes failed
            // with CUDA status 2" with the direct buffers held. They are per-launch transients (zeroed or copied each
            // launch) and are made again by the next direct launch.
            self.p3_direct_release(cuda)?;
            init_taped = init;
        }
        let v = self.p3_taped(cuda, p, mode, identity)?;
        // the initflag launch is the last backward step of a gradient. Its whole-launch tape (p3_arena, p3_store,
        // p3_nodes, ...) is freed after it, so the next gradient of a kept-loaded program starts its first backward step
        // with the card room the first gradient had. THE BREAKAGE: with one direct batch of 40,960 columns the second
        // gradient's step 12 refused "cuMemAlloc scheme scratch 1370488832 bytes ... CUDA status 2": the taped scratch
        // of gradient 1's initflag launch was still held. AD_P3_TAPED_RELEASE=0: as before.
        if init_taped && env_var("AD_P3_TAPED_RELEASE").as_deref() != Some("0") { self.p3_taped_release(cuda)?; }
        Ok(v)
    }

    /// free the taped MYNN route's scratch (every p3_* block but p3_direct_*), live and retired, after a sync.
    unsafe fn p3_taped_release(&mut self, cuda: &Cuda) -> Result<(), String> {
        dev::sync(cuda)?;
        let (mut n, mut b) = (0usize, 0usize);
        for (k, g) in self.scratch.iter_mut() {
            if !k.starts_with("p3_") || k.starts_with("p3_direct_") { continue; }
            b += g.bytes + g.retired_b;
            g.retire();
            for a in g.retired.drain(..) { if (cuda.free)(a) == 0 { n += 1; } }
            g.retired_b = 0; g.bytes = 0;
        }
        self.note_own_ring();
        if n > 0 { eprintln!("{}", json!({"event":"mynn_taped_released","blocks":n,"bytes":b})); }
        Ok(())
    }

    /// tape-free fall-back (3): free the direct route's scratch (p3_direct_*), live and retired, after a sync.
    unsafe fn p3_direct_release(&mut self, cuda: &Cuda) -> Result<(), String> {
        dev::sync(cuda)?;
        let (mut n, mut b) = (0usize, 0usize);
        for (k, g) in self.scratch.iter_mut() {
            if !k.starts_with("p3_direct_") { continue; }
            b += g.bytes + g.retired_b;
            g.retire();
            for a in g.retired.drain(..) { if (cuda.free)(a) == 0 { n += 1; } }
            g.retired_b = 0; g.bytes = 0;
        }
        if n > 0 { eprintln!("{}", json!({"event":"mynn_direct_released","blocks":n,"bytes":b})); }
        Ok(())
    }

    /// tape-free fall-back (3): the taped MYNN route's card budget, as p3_taped measures it at its first launch.
    unsafe fn p3_budget_init(&mut self, cuda: &Cuda) -> Result<(), String> {
        let (mut free, mut total) = (0usize, 0usize);
        checked("cuMemGetInfo", (cuda.mem_info)(&mut free, &mut total))?;
        let want = (env_var("AD_P3_BUDGET_GIB").and_then(|v| v.parse::<f64>().ok()).unwrap_or(10.0) * (1u64 << 30) as f64) as usize;
        let _ = total; let reserve = 16usize << 30;
        self.p3_budget = Some(want.min(free.saturating_sub(reserve)));
        eprintln!("{}", json!({"event":"mynn_p3_budget","budget_bytes":self.p3_budget,"free":free,"at":"before the direct route"}));
        Ok(())
    }

    unsafe fn p3_taped(&mut self, cuda: &Cuda, p: &mut Prepared, mode: i32, identity: bool) -> Result<Value, String> {
        let row = self.rows["rrfs_mynnedmf_run_l16"].clone();
        let (f_tl, f_ad, f_count, sz_tl, sz_tape, sbytes, node_b, lanes) = {
            let img = self.p3.as_ref().ok_or("P3 image not loaded")?;
            (img.f_tl, img.f_ad, img.f_count, img.sz_tl, img.sz_tape, img.struct_bytes, img.node_bytes, img.tile_w)
        };
        // mynn-tape: one block = one warp = one arena tile of `lanes` columns (the image's MW_TILE). 32 before; an
        // MW_TILE=8 image spreads pkg18m2's 2,562 columns over 321 blocks instead of 81 on a 188-SM card, and each warp
        // runs the union of 8 columns' branch paths instead of 32. Columns are independent (G5): the words do not change.
        let pgrid = |n: usize| -> [u32; 3] { [n.div_ceil(lanes) as u32, 1, 1] };
        let pblock = [lanes as u32, 1, 1];
        // 60 km tape: a banded bind's snapshot is the whole region at once (the same layout; a segment's virtual
        // extent is not memory). The head copies segment by segment.
        if let Some((.., snap)) = p.stripe { dev::copy(cuda, snap, p.region.0, p.region.1)?; }
        else { for s in p.segs.v.iter().filter(|s| s.active) { dev::copy(cuda, s.snap, s.shadow, s.shadow_bytes())?; } }
        let names: Vec<String> = row.raw["float_arrays"].as_array().ok_or("float_arrays")?.iter().map(|v| v.as_str().unwrap_or("").to_owned()).collect();
        let scalars: Vec<String> = row.raw["scalars"].as_array().ok_or("scalars")?.iter()
            .map(|v| v.as_str().unwrap_or("").split_whitespace().last().unwrap_or("").to_owned()).collect();
        let env = p.env.clone();
        let g = |k: &str| -> Result<usize, String> { env.get(k).map(|v| *v as usize).ok_or_else(|| format!("P3 scalar {k}")) };
        let cols = g("batch_count")?;
        // mynn-adjoint-v2: scale probe (bench images only): R copies of every column in each P3 launch.
        let rep = env_var("AD_P3_BENCH_REPLICATE").and_then(|v| v.parse::<usize>().ok()).unwrap_or(1).max(1);
        if rep > 1 && !self.p3.as_ref().map_or(false, |i| i.bench) { return Err("AD_P3_BENCH_REPLICATE needs a scale-probe P3 image (-DMW_BENCH_REPLICATE)".into()); }
        let cols = cols * rep;
        let nk = g("kme")? + 1 - g("kms")?;
        let col_elems = names.len() * nk + 8;
        let ws = u64::from_le_bytes(scalar_bytes(&p.args[row.arg("workspace_stride")?.pos]).ok_or("workspace_stride")?[..8].try_into().map_err(|_| "workspace_stride width")?) as usize;
        // mynn-fused: AD_P3_ARENA_X overrides the table's arena multiple; AD_P3_ARENA_X_INIT applies to the adjoint of
        // an initflag launch only. Breakage it prevents: the tape-free MYNN column reverse keeps its checkpoints in the
        // arena, and f_mym_initialize (step 1 only) needs a 1.08 MB checkpoint stack, over the 12 x 73 KB arena: every
        // column failed slot 3 (code 33) at pkg18m2 AD step 1 while steps 2-4 ran.
        let ax_env = |k: &str| env_var(k).and_then(|v| v.parse::<usize>().ok()).filter(|v| *v > 0);
        let ax = if mode == 1 && env.get("initflag").map_or(false, |v| *v as i64 != 0) {
            ax_env("AD_P3_ARENA_X_INIT").or_else(|| ax_env("AD_P3_ARENA_X")).unwrap_or(self.p3_arena_x)
        } else { ax_env("AD_P3_ARENA_X").unwrap_or(self.p3_arena_x) };
        let arena_col = ws * ax;
        let cols32 = cols.div_ceil(32) * 32;
        if self.p3_budget.is_none() {
            let (mut free, mut total) = (0usize, 0usize);
            checked("cuMemGetInfo", (cuda.mem_info)(&mut free, &mut total))?;
            let want = (env_var("AD_P3_BUDGET_GIB").and_then(|v| v.parse::<f64>().ok()).unwrap_or(10.0) * (1u64 << 30) as f64) as usize;
            // tempo-mynn-kernels: the 16 GiB reserve is a quarter of the card at most. On a 32 GB card it left a 0.72 GB
            // budget for pkg18m2 (2,562 columns), so MYNN ran 384-column tiles: adjoint 2.3-2.9 s per step against 0.45 s
            // whole on a 96 GB card (a test machine tm-A4). Cards of 64 GiB or more keep 16 GiB, as before.
            // adjoint-speed-4: the quarter-card reserve below 64 GiB is withdrawn (16 GiB on every card, as speed-3). It won the
            // 4-step window (tm-A4 -> tm-R4) but its whole-launch arena (2.28 GB at pkg18m2) is held through the window, so
            // the 18-step reverse is memory-starved: a test machine 5090, s4-B18 (speed3) MYNN adjoint 43.2 s over 18 steps (384-column
            // tiles) vs s4-A18b 214.3 s (68-80 reverse tiles), same words. Cards of 64 GiB or more are unchanged.
            let _ = total; let reserve = gib_word("AD_P3_RESERVE_GIB", 16usize << 30);
            self.p3_budget = Some(want.min(free.saturating_sub(reserve)));
        }
        let budget = self.p3_budget.unwrap_or(0);
        let col_fixed = arena_col + col_elems * sz_tl.max(sz_tape) + col_elems * 4 + 64;
        // Whole-launch tile for the tangent and the count pass when every column's arena + store fits the budget.
        let whole = !self.p3_pinned && cols32 * col_fixed <= budget;
        // tempo-mynn-kernels: a launch that does not fit one tile takes the most columns whose arena + store fit the budget
        // (2/3 of it, for the scratch's half-again growth), never fewer than the table tile; AD_P3_TILE still pins it.
        let fit = (budget / 3 * 2 / col_fixed.max(1)) / 32 * 32;
        // At most AD_P3_TILE_MAX columns (default 4096: 512 one-warp blocks at MW_TILE 8, about 2.7 per SM on a 188-SM card),
        // so the arena held for the rest of the window grows by at most ~3.6 GB over the old 384-column tile at pk120.
        // mynn-adjoint-v2: the 4,096 default stays on every card. Lifted on a 96 GB card (a416a2c7), the tangent's arena/store
        // grew to the budget fit (864 slots, 10.7 GB) and is held through the window: the pkg18m2 bench probes gained (MYNN
        // adjoint 11.1-14.2 -> 7.7-9.5 s per launch at 163,968 columns, mav2-R64P/R64PT), but at pk120 (x1.40962 on one 96 GB
        // card) it left the reverse 9 MB of room: one adjoint slot, 275 s for one launch (mav2-K120E4). AD_P3_TILE_MAX lifts
        // it where the window leaves the room.
        let tile_max = env_var("AD_P3_TILE_MAX").and_then(|t| t.parse::<usize>().ok()).unwrap_or(4096).max(32) / 32 * 32;
        let tile = if whole { cols32 } else if self.p3_pinned { self.p3_tile.min(cols32) } else { fit.min(tile_max).max(self.p3_tile).min(cols32) };
        // mynn (dycore-card relief, opt-in AD_MYNN_SHARD_PRIMARY=<0.0 to below 1.0>, with AD_MYNN_SHARD_DEVICES). THE BREAKAGE:
        // at x1.163842 on 4 cards (a test machine <local path> r3-w60-kd7) MYNN-EDMF was 11.3 s/step of 26.0 s of adjoint entries and
        // 3.1 of 9.5 s of tangent entries: the dycore card took an equal quarter of every launch on a full card, where its
        // arena, store and y blocks (13.25 GB, COLD[p3]) are cold-tier memory pulled back for each of the 12 launches a
        // step, with a 4 GiB lent tape ring (2-3 wraps), while each helper card had 50 GB free. With the switch the
        // dycore card keeps that fraction of an equal share (never under one group) and its arena/store/leaf blocks are
        // sized for its own columns, not the 4,096-column tile; the helpers keep the full tile's slot cap. Column
        // ownership only: columns are independent (G5), so no word changes. Tangent launches follow when
        // AD_MYNN_SHARD_TL=1 (else the tangent stays whole on the dycore card and keeps the full tile).
        let prim_share = env_var("AD_MYNN_SHARD_PRIMARY").and_then(|v| v.parse::<f64>().ok()).filter(|v| (0.0..1.0).contains(v));
        let tile_full = tile;
        let tile = match (prim_share, self.p3_shards.as_ref()) {
            (Some(share), Some(sh)) if self.p3.as_ref().and_then(|i| i.f_p).is_some() && env_var("AD_P3_PERSIST").as_deref() != Some("0") && !self.p3_pinned
                && node_b == 16 && rep == 1 && cols >= sh.ordinals.len() && (mode == 1 || env_var("AD_MYNN_SHARD_TL").as_deref() == Some("1")) => {
                let c0 = crate::column_shard::partition_primary(cols, sh.ordinals.len(), share, lanes)?[0].count;
                tile.min(c0.div_ceil(32) * 32).max(32)
            },
            _ => tile,
        };
        // Primal y buffers (the entries write the forward's own outputs there): contiguous, one per active segment,
        // initialised to the pre state; after the first tangent they are compared word for word with the forward.
        let total: usize = p.segs.v.iter().filter(|s| s.active).map(|s| s.hi - s.lo).sum();
        // mynn: exact size (need_exact). The half-again headroom of `need` made this 3 x the launch's primal span
        // (6.89 GB at x1.163842, where the block is cold-tier memory the dycore card pulls in for every launch).
        // mynn (y buffer only for its reader). THE BREAKAGE: the y buffer is read by the identity compare alone (the
        // launch-dot tool; the window's tangent and every adjoint pass call with identity = false), yet every launch
        // allocated it, copied the launch's whole primal span into it and had every pass of every card write its columns
        // there. At x1.163842 on 4 cards (a test machine <local path> m60d-A, window_alloc_census) it is a 4.6 GB COLD[p3] block:
        // managed memory with no placement, so the dycore card pulled it in and rewrote 2.3 GB of it for each of the 12
        // launches a step, and each helper card's count, adjoint and tangent pass wrote into it from another card (every
        // helper pass cost 0.6-1.7 s whatever it computed; MYNN-EDMF adjoint 11.5 of 19.9 s/step of adjoint entries).
        // Without a reader the entries get a null y (the generated store is `if (B.y)`), and nothing is allocated or
        // copied. AD_P3_Y_ALWAYS=1 keeps the old binding (the before leg of a timing pair). No word depends on y.
        let keep_y = identity || env_var("AD_P3_Y_ALWAYS").as_deref() == Some("1");
        let ybase = if keep_y { self.scratch.entry("p3_y").or_default().need_exact(cuda, 2 * total.max(8))? } else { 0 };
        let mut at = 0u64;
        let mut seg_y: HashMap<(u64, usize), u64> = HashMap::new();
        for s in p.segs.v.iter().filter(|s| s.active && keep_y) {
            seg_y.insert((s.alloc, s.lo), ybase + at);
            dev::copy(cuda, ybase + at, s.pre, s.hi - s.lo)?;
            at += (s.hi - s.lo) as u64;
        }
        let mut slots: Vec<[u64; 6]> = Vec::new();
        for n in &names {
            let a = row.arg(n)?;
            match p.args.get(a.pos).and_then(target) {
                None => slots.push([0; 6]),
                Some((al, off)) => {
                    let s = p.segs.find(al, off)?;
                    let x = s.pre + (off - s.lo) as u64;
                    let yy = if keep_y { seg_y[&(s.alloc, s.lo)] + (off - s.lo) as u64 } else { 0 };
                    let (sh, sn) = (p.segs.shadow(al, off)?, p.segs.snap(al, off)?);
                    slots.push(if mode == 0 { [x, yy, sn, sh, 0, 0] } else { [x, yy, 0, 0, sn, sh] });
                },
            }
        }
        let karg = |n: &str| -> Result<(u64, usize, usize), String> {
            let a = row.arg(n)?;
            let (al, off) = p.args.get(a.pos).and_then(target).ok_or_else(|| format!("P3 {n}"))?;
            Ok((al, off, row.extent_bytes(a, &env)?)) };
        let (kal, koff, kbytes) = karg("kpbl")?;
        let (eal, eoff, ebytes) = karg("error_flags")?;
        let kin = p.segs.pre(kal, koff)?;
        let epre = p.segs.pre(eal, eoff)?;
        let sc = &mut self.scratch;
        let mut need = |k: &'static str, b: usize| -> Result<u64, String> { sc.entry(k).or_default().need(cuda, b) };
        let kout = need("p3_kout", kbytes)?;
        dev::copy(cuda, kout, kin, kbytes)?;
        let eflags = need("p3_eflags", ebytes)?;
        dev::copy(cuda, eflags, epre, ebytes)?;
        let store_b = tile * col_elems * sz_tl.max(sz_tape);
        let (arena, store, kstore, fail) = (need("p3_arena", tile * arena_col)?, need("p3_store", store_b)?, need("p3_kstore", tile * 32)?, need("p3_fail", tile.max(cols32) * 16)?);
        let (used, used_hw, dom, lbuf, leaf) = (need("p3_used", tile.max(cols32 + 8) * 8)?, need("p3_used_hw", tile.max(cols32) * 8)?, need("p3_dom", tile.max(cols32) * 4)?,
                                                need("p3_launch", sbytes)?, need("p3_leaf", tile * col_elems * 4)?);
        {
            let img = self.p3.as_ref().unwrap();
            for m in [&img.tl, &img.ad, &img.count] {
                m.set_global(cuda, "mw_arena_limit", &(arena_col as u64).to_le_bytes())?;
                m.set_global(cuda, "mw_fail", &fail.to_le_bytes())?;
                for z in ["mw_br_bits", "mw_br_n", "mw_br_hash", "mw_br_words"] { m.set_global(cuda, z, &0u64.to_le_bytes())?; }
                m.set_global(cuda, "mw_arena_hw", &used_hw.to_le_bytes())?;
                m.set_global(cuda, "mw_dom", &dom.to_le_bytes())?;
            }
        }
        let sargs: Vec<K> = scalars.iter().map(|s| Ok(K::B(scalar_bytes(&p.args[row.arg(s)?.pos]).ok_or_else(|| format!("P3 scalar {s}"))?.to_vec())))
            .collect::<Result<_, String>>()?;
        let mut fails = [0u64; 4];
        // readbacks-device: with the window's ledger, every launch's failure codes are counted on the device into one
        // Codes4 item (overflow, context and arena refuse the step at its end with the message below; domain is
        // reported in window_device_flags). Integer counts: the totals are the host loop's.
        let codes: Option<(u64, u64, *mut c_void)> = match self.ledger.as_mut() {
            Some(l) => { let (a, t) = l.reserve(cuda, 32, dev::LedgerItem::Codes4 { label: format!("P3 mode {mode}"), fatal: [true, false, true, true],
                                                                                     names: ["overflow", "domain", "context", "arena"] })?;
                         Some((a, t, l.f_codes4)) },
            None => None,
        };
        let count_fails = |n: usize, fails: &mut [u64; 4]| -> Result<(), String> {
            match codes {
                Some((cnt, _, f4)) => {
                    let (mut a, mut nc, mut c) = (fail, n as u64, cnt);
                    let mut args: [*mut c_void; 3] = [(&mut a as *mut u64).cast(), (&mut nc as *mut u64).cast(), (&mut c as *mut u64).cast()];
                    checked("P3 codes4 launch", (cuda.launch)(f4, dev::BLOCKS, 1, 1, 256, 1, 1, 0, std::ptr::null_mut(), args.as_mut_ptr(), std::ptr::null_mut()))
                },
                None => {
                    dev::sync(cuda)?;
                    let fl: Vec<i32> = dev::download(cuda, fail, n * 16)?.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
                    for s in 0..4 { fails[s] += fl.iter().skip(s).step_by(4).filter(|&&v| v != 0).count() as u64; }
                    Ok(())
                },
            }
        };
        let mut caps: Vec<usize> = Vec::new();
        let mut first = 0usize;
        let mut persisted: Option<Value> = None;
        let mut pers_counted: Vec<i64> = Vec::new();
        let mut pers_tiles: Option<(Vec<(usize, usize, usize)>, usize)> = None;
        // ---- mynn-adjoint-v2: persistent P3 launches ----------------------------------------------------------------
        // An image with the persistent entries runs each pass as ONE launch of `slots` blocks. A block owns MW_TILE column
        // slots (arena tile, store, leaf, kstore and, in the adjoint, a tape region of the largest group tape the count
        // pass measured) and loops over column groups taken from a device counter, zeroing its arena tile and store per
        // group as the host memset did per launch. The tangent and the count pass are sized by the tangent budget, the
        // adjoint by the reverse-time room, so memory follows the columns in flight, not the launch: no host tiles, one
        // host read per reverse (the region size), no tile tails. Columns are independent (G5) and run the one-shot
        // entries' column routine, so the words do not change. AD_P3_PERSIST=0 keeps the tiles; AD_P3_SLOTS caps the
        // blocks (default: what the card keeps resident, from the occupancy API).
        let pers = self.p3.as_ref().and_then(|i| i.f_p).filter(|_| env_var("AD_P3_PERSIST").as_deref() != Some("0") && !self.p3_pinned && node_b == 16);
        // multicard-merge: MYNN column shards. Helpers take ranges 1.., this context keeps range 0 (cols_here columns).
        let shard_plan: Option<Vec<crate::column_shard::Range>> = match (pers.is_some(), self.p3_shards.as_ref()) {
            // The tangent is 0.18 s/step at 120 km; split it cost 0.84 s (a test machine mcm-pk120-n12): it stays here unless AD_MYNN_SHARD_TL=1.
            (true, Some(_)) if mode == 0 && env_var("AD_MYNN_SHARD_TL").as_deref() != Some("1") => None,
            (true, Some(sh)) => {
                if rep > 1 { return Err("DEFECT MYNN column shards with AD_P3_BENCH_REPLICATE".into()); }
                if self.p3_par.get() { return Err("DEFECT MYNN column shards with --params mynnedmf: the carrier globals are on this context only".into()); }
                Some(match prim_share { Some(share) => crate::column_shard::partition_primary(cols, sh.ordinals.len(), share, lanes)?, None => crate::column_shard::partition(cols, sh.ordinals.len())? })
            },
            (false, Some(_)) => return Err("DEFECT AD_MYNN_SHARD_DEVICES needs the persistent MYNN entries (image with the *_p entries, AD_P3_PERSIST not 0, no AD_P3_TILE)".into()),
            _ => None,
        };
        let cols_here = shard_plan.as_ref().map_or(cols, |r| r[0].count);
        let mut shard_sent = 0usize;
        let shard_wall = std::time::Instant::now();
        if let (Some(sh), Some(r)) = (self.p3_shards.as_ref(), shard_plan.as_ref()) {
            // The snapshots, y buffers, kpbl/error-flag copies above are queued on this context: done before helpers read.
            dev::sync(cuda)?;
            let sbytes_args: Vec<Vec<u8>> = sargs.iter().map(|k| match k { K::B(b) => b.clone(), K::P(p) => p.to_le_bytes().to_vec() }).collect();
            let cap_env = env_var("AD_P3_SLOTS").and_then(|v| v.parse::<usize>().ok()).unwrap_or(usize::MAX).max(1);
            // mynn (helper-local stripes; AD_MYNN_SHARD_LOCAL=0 keeps the in-place peer access). THE BREAKAGE: a helper's
            // kernel read the pre state and the derivative inputs, and wrote the derivative outputs, in the dycore card's
            // memory through the peer mapping, element by element: a test machine <local path> m60e-Ncft, a helper's tangent pass of
            // 9,617 columns 0.5-0.64 s at 512 blocks where the dycore card's own tangent of 29,216 columns is issued and
            // done inside 0.6 s. Now the helper copies its columns' stripes of every array to its own card (one pitched
            // peer copy per array), runs on them and copies the derivative outputs' stripes back. Same kernels, same
            // columns, same element values in and out: no word changes. Used when the launch is one row of columns
            // (jms = jme), does not wrap, every array extent is whole rows of ni elements and no y buffer is bound.
            let local: Option<P3Local> = (|| {
                if keep_y || env_var("AD_MYNN_SHARD_LOCAL").as_deref() == Some("0") { return None; }
                let gi = |k: &str| env.get(k).map(|v| *v as i64);
                let (ims, ime, jms, jme, its, ite, jts, jte) = (gi("ims")?, gi("ime")?, gi("jms")?, gi("jme")?, gi("its")?, gi("ite")?, gi("jts")?, gi("jte")?);
                let bf = gi("batch_first")?;
                if jms != jme || jts != jte || jts != jms || ime < ims || its < ims || bf < 0 || bf + cols as i64 > ite - its + 1 { return None; }
                let ni = (ime - ims + 1) as usize;
                let mut rows = Vec::with_capacity(names.len());
                for (n, sl) in names.iter().zip(&slots) {
                    if sl[0] == 0 { rows.push(0); continue; }
                    let b = row.extent_bytes(row.arg(n).ok()?, &env).ok()?;
                    if b == 0 || b % (ni * 4) != 0 { return None; }
                    rows.push(b / (ni * 4));
                }
                let mut main = ptr::null_mut();
                if (cuda.ctx_get_current)(&mut main) != 0 || main.is_null() { return None; }
                Some(P3Local { ni, i_first: (its - ims + bf) as usize, rows, main: main as usize })
            })();
            for (w, rg) in sh.workers.iter().zip(&r[1..]) {
                let job = P3Job { slots: slots.clone(), ptrs: [kin, kout, eflags], sargs: sbytes_args.clone(), first: rg.first, count: rg.count, mode,
                                  arena_col, col_elems, sz_store: sz_tl.max(sz_tape), cap_env,
                                  // mynn (helper blocks follow the helper's card). THE BREAKAGE: a helper ran at most the dycore card's
                                  // tile in blocks (4,096 columns = 512 blocks) although it has its own card: a test machine <local path> m60e-N, every
                                  // helper pass at 512 blocks, count pass 0.30-0.35 s and adjoint pass 0.6 s for 7,304 columns. Now a helper
                                  // takes what its card keeps resident and its free memory holds (P3Helper::run);
                                  // AD_MYNN_SHARD_HELPER_SLOTS=<n> caps it (512 = the old figure at the default tile).
                                  s_cap: env_var("AD_MYNN_SHARD_HELPER_SLOTS").and_then(|v| v.parse::<usize>().ok()).filter(|v| *v > 0).unwrap_or(usize::MAX),
                                  init: env.get("initflag").map_or(false, |v| *v as i64 != 0), count_free: env_var("AD_MYNN_SHARD_COUNT_FREE").as_deref() == Some("1"),
                                  local: local.clone() };
                if w.tx.send(job).is_err() { break; }
                shard_sent += 1;
            }
            if shard_sent + 1 != r.len() {
                for w in &sh.workers[..shard_sent] { let _ = w.rx.lock().map(|rx| rx.recv()); }
                return Err("DEFECT MYNN shard worker stopped".into());
            }
        }
        let t_prim = std::time::Instant::now();
        let primary_run = (|| -> Result<(), String> {
        if let Some([f_tl_p, f_ad_p, f_count_p]) = pers {
            let groups = cols_here.div_ceil(lanes);
            let resident = |f: *mut c_void| -> usize {
                let (mut nb, mut dv, mut sms) = (0i32, 0i32, 0i32);
                if (cuda.occupancy)(&mut nb, f, lanes as i32, 0) != 0 || (cuda.ctx_device)(&mut dv) != 0 || (cuda.device_attribute)(&mut sms, 16, dv) != 0 { return 1024; }
                (nb.max(1) as usize) * (sms.max(1) as usize)
            };
            let cap_env = env_var("AD_P3_SLOTS").and_then(|v| v.parse::<usize>().ok()).unwrap_or(usize::MAX).max(1);
            let slot_b = lanes * col_fixed;
            let sched = self.scratch.entry("p3_sched").or_default().need(cuda, 64)?;
            let (m_tl, m_ad, m_count) = { let i = self.p3.as_ref().unwrap(); (&i.tl as *const Module, &i.ad as *const Module, &i.count as *const Module) };
            let mk = |acol: usize, mode_k: i32, layout: i32, ar: u64, st: u64, lf: u64, ks: u64, nodes: u64, bars: u64, capacity: u64| -> Vec<u8> {
                let mut w: Vec<u8> = Vec::with_capacity(sbytes);
                let put = |w: &mut Vec<u8>, v: u64| w.extend_from_slice(&v.to_le_bytes());
                for sl in &slots { let sl = if mode_k == 2 { [sl[0], sl[1], 0, 0, 0, 0] } else { *sl }; for v in sl { put(&mut w, v); } }
                for v in [kin, kout, eflags, ar, acol as u64, st, col_elems as u64, lf, ks, nodes, bars, capacity, used, fail] { put(&mut w, v); }
                for v in [0i32, cols_here as i32, mode_k, layout] { w.extend_from_slice(&v.to_le_bytes()); }
                w
            };
            // mynn (count-free adjoint launches). THE BREAKAGE: every MYNN-EDMF adjoint launch ran the taped forward
            // twice, once only to size each column's tape (the count pass) and once to record it: at x1.40962 the count
            // pass is 0.59-0.73 s of the 1.90 s/step the adjoint takes (a test machine <local path> w1-B9x), and at 1,500 slots the card
            // is issue-bound, so no memory or slot change removes it. Now a launch whose shape (columns, initflag) was
            // counted before in this run gives every column one capacity: 1.25 x the largest count that shape has shown
            // (2 x the other initflag shape's when this one was never counted), and skips the count pass. The kernel is
            // unchanged: a column that needs more refuses its pushes and sets its overflow slot, never a wrong word. The
            // overflow slots are read after the launch (the one wait the count pass's read was); any overflow reruns the
            // launch the old way (count pass, exact capacities), which also refreshes the largest count. The entry assigns
            // its outputs, so the rerun leaves the words of a counted launch. Not used when the uniform capacity would
            // leave fewer groups in flight than half the resident blocks (set-A images: columns of 150-300 thousand nodes
            // beside a mean of 60 thousand), when the ring it needs is larger than the ring already held or lent (it never
            // grows memory), with column shards, or with the layout-2 tape.
            // OPT-IN (AD_P3_COUNT_FREE=1): measured at x1.40962 (a test machine <local path> w4-B9 against w4-B9fit) the uniform capacity
            // makes the record pass slower (277 -> 488 ms and 801 -> 961 ms per launch: 1.64 x the ring bytes per group, one
            // more ring wrap), so the two launches that used it gained 0.35 s where the count passes they skipped cost 0.73.
            // The same run's count_drift record says per-column capacities would do: no column of the plume-frozen image
            // grew past 1.13 x its previous count between steps (set A: up to 7 x, 385-918 columns past 1.25 x).
            let cf_key = (cols_here, env.get("initflag").map_or(false, |v| *v as i64 != 0));
            let mut cf_cap: Option<i64> = if mode == 1 && shard_plan.is_none() && env_var("AD_P3_COUNT_FREE").as_deref() == Some("1") {
                match (self.p3_cap_seen.get(&cf_key), self.p3_cap_seen.get(&(cf_key.0, !cf_key.1))) {
                    (Some(&m), _) => Some(m + m / 4 + 1024),
                    (None, Some(&m)) => Some(2 * m + 1024),
                    _ => None,
                }
            } else { None };
            let mut cf_redo: Option<(usize, &'static str)> = None;
            let caps_len0 = caps.len();
            // speed mynn (count cache, opt-in AD_P3_COUNT_CACHE=<directory>, or =mem for this run only). THE BREAKAGE:
            // every MYNN-EDMF adjoint launch runs the taped forward once only to size each column's tape: at x1.163842 on
            // one card 2.49 s of the 8.1 s/step the adjoint takes with every speed setting on (a test machine <local path> pD-sync,
            // six 16-byte reads that wait for the count launch). The counts depend on the forward trajectory of that step
            // and on nothing else, and a package is swept backward many times (every check direction, every gate, every
            // minimisation step is another sweep of one trajectory). So the first counted launch of a (step, first column,
            // columns, initflag) keeps its exact counts and its largest group tape, in memory and as a 1.3 MB file per
            // step at that mesh; a later launch of the same key in this run, or in a later run given the same directory,
            // uploads them and runs no count launch. Exact counts: the ring plan, the slots and so the words are those of
            // a counted launch. Kept counts that are too small (another package or image behind the same directory) cannot
            // give a wrong word: the columns refuse their pushes and set their overflow slots, which are read after the
            // launch (the wait the count pass's read was); then the launch is counted and run again the old way, the key
            // is not used again in this run, and its file is rewritten. One directory per package and scheme image.
            // Not used with column shards, the arena fit, the scale probe, or when the window gave no step.
            let cc_key = (self.p3_step, env.get("batch_first").map_or(-1, |v| *v as i64), cols_here, cf_key.1);
            let cc_dir: Option<PathBuf> = env_var("AD_P3_COUNT_CACHE").filter(|_| mode == 1 && self.p3_step >= 0 && shard_plan.is_none() && self.p3_shards.is_none()
                && rep == 1 && env_var("AD_P3_ARENA_FIT").as_deref() != Some("1")).map(PathBuf::from);
            let cc_file = cc_dir.as_ref().filter(|d| d.as_os_str() != "mem")
                .map(|d| d.join(format!("p3count-s{}-b{}-n{}-i{}.bin", cc_key.0, cc_key.1, cc_key.2, cc_key.3 as u8)));
            let cc_head = [0x3532_4633_4343_3350u64, cols_here as u64, lanes as u64, col_elems as u64, sz_tape as u64, node_b as u64];
            let mut cc: Option<(Vec<i64>, usize)> = if cc_dir.is_some() && !self.p3_cc_bad.contains(&cc_key) {
                self.p3_cc.get(&cc_key).cloned().or_else(|| cc_file.as_ref().and_then(|f| p3_cc_read(f, &cc_head)))
            } else { None };
            let mut cc_redo: Option<String> = None;
            // a12 bs-mynn (previous-step counts, opt-in AD_P3_COUNT_PREV=<k>, with AD_P3_COUNT_PREV_EVERY=<n> (default 4) and
            // AD_P3_COUNT_PREV_ADD=<nodes> (default 4096)). THE BREAKAGE: the first gradient of a start has no kept counts, so
            // every MYNN-EDMF reverse launch runs its count pass (0.51 s of a 3 km backward step, back-2 k12-mb-ffill). A step's
            // counts are those of the step after it give or take a little (3 km d4-15: the largest growth of one column over
            // 1 to 4 steps is 3,922 nodes on 27,000; cc/k12 files of k12-mb-fill), so a launch of a shape counted in this sweep
            // takes those counts, each c as c + (c >> k) + ADD, and skips its count pass; every n-th launch of a shape counts
            // again, and so does a launch whose columns overflow (caught by the overflow slots, run again counted, as for the
            // count cache). A capacity is only room: each column's tape is its own pushes in its own order, so the words are
            // those of a counted launch.
            let pc_k = env_var("AD_P3_COUNT_PREV").and_then(|v| v.parse::<u32>().ok()).filter(|k| (1..=16).contains(k))
                .filter(|_| mode == 1 && self.p3_step >= 0 && shard_plan.is_none() && self.p3_shards.is_none() && rep == 1 && env_var("AD_P3_ARENA_FIT").as_deref() != Some("1"));
            let mut cc_prev = false;
            if cc.is_none() {
                if let Some(k) = pc_k {
                    let every = env_var("AD_P3_COUNT_PREV_EVERY").and_then(|v| v.parse::<usize>().ok()).unwrap_or(4).max(1);
                    let add = env_var("AD_P3_COUNT_PREV_ADD").and_then(|v| v.parse::<i64>().ok()).unwrap_or(4096).max(0);
                    if let Some((base, uses)) = self.p3_pc.get_mut(&cf_key) {
                        if base.len() == cols_here && *uses + 1 < every {
                            *uses += 1;
                            let cnt2: Vec<i64> = base.iter().map(|&c| if c < 0 { c } else { c + (c >> k) + add }).collect();
                            let rg = cnt2.chunks(lanes.max(1)).map(|g| g.iter().map(|&u| u.max(0) as usize + 16).sum::<usize>()).max().unwrap_or(0);
                            cc = Some((cnt2, rg));
                            cc_prev = true;
                        }
                    }
                }
            }
            let rec_cf: Value = 'cf: loop {
            // (1) the tangent, or the reverse's count pass, over every column
            let f1 = if mode == 0 { f_tl_p } else { f_count_p };
            // the slots never outgrow the tile buffers allocated above (a grown block retires the old one: see Grow)
            // `tile` is already the budget fit with the table-tile floor (384 columns), as the tiled path uses it.
            // mynn (arena fit). THE BREAKAGE: every column in flight in the adjoint held the table's arena multiple
            // (12 x the forward's workspace: 881 KB at 59 levels) whatever it used; the high-water measured in whole windows
            // is 6.0-6.5 x (a test machine <local path> g1, w1), so on a full 96 GB card the arena capped the adjoint at 761 of the 1,504
            // resident blocks (x1.40962: MYNN-EDMF adjoint 2.37 s/step, 1.90 s/step at the measured multiple, same words).
            // The count pass runs the same taped forward as the adjoint pass on the same columns, so its arena high-water
            // (one small read beside the counts) is the adjoint pass's need. The adjoint pass takes it, times the largest
            // adjoint/count ratio this run has measured, plus 1/16 and 4 KiB. The first adjoint launch of a run keeps the
            // table multiple and reports its own high-water (read at the next launch, no extra wait): that sets the ratio.
            // An arena overflow in the generated column is a trap (the run stops with a CUDA error, never a wrong word),
            // which is why the count pass keeps the table multiple and nothing is fitted before a ratio was measured.
            // The count pass only takes more slots: as many as the arena block already held covers.
            // Columns are independent and the arena holds each column's own automatic arrays, so no word changes.
            // OPT-IN (AD_P3_ARENA_FIT=1) until one unexplained arm is understood: a test machine <local path> g4-B9count (pkg18m2 on a shared
            // card, TEMPO arena 15.05 GB instead of 16) gave dot 1.25e-5 and other words with the fit on and every launch
            // counted, while the same settings give the reference words on a whole card at x1.2562 and x1.40962 (w4).
            // Any AD_P3_ARENA_X keeps the fixed multiple everywhere.
            let fit_on = mode == 1 && env_var("AD_P3_ARENA_FIT").as_deref() == Some("1") && ax_env("AD_P3_ARENA_X").is_none() && ax_env("AD_P3_ARENA_X_INIT").is_none();
            let held_slots = if fit_on { self.scratch.get("p3_arena").map(|g| g.bytes).unwrap_or(0) / (lanes * arena_col).max(1) } else { 0 };
            // speed mynn (the count pass on the blocks already held). THE BREAKAGE: at x1.163842 on one card the reverse's
            // count pass ran 512 groups of columns at a time (the tangent's 4,096-column tile) while the adjoint launch that
            // follows it runs 761 on arena, store and kstore blocks that stay held between launches: the six count passes
            // are 4.1 s of the 19.4 s/step the MYNN-EDMF adjoint takes (a test machine <local path> dd60-one-self), and the count pass is
            // slot-bound (a test machine <local path> w1-B9 against w1-B9x: 23.6 -> 15.9 us per column from 512 to 1,024 slots). Now
            // the count pass takes as many slots as the three blocks it uses already cover, so it allocates nothing and
            // the arena multiple is untouched. Slots only group columns (G5): the words do not change.
            // AD_P3_COUNT_HELD=0 keeps the tile.
            let held_count = if mode == 1 && !fit_on && env_var("AD_P3_COUNT_HELD").as_deref() != Some("0") {
                let hb = |k: &str| self.scratch.get(k).map(|g| g.bytes).unwrap_or(0);
                (hb("p3_arena") / (lanes * arena_col).max(1)).min(hb("p3_store") / (lanes * col_elems * sz_tl.max(sz_tape)).max(1)).min(hb("p3_kstore") / (lanes * 32).max(1))
            } else { 0 };
            let s1 = groups.min(resident(f1)).min(cap_env).min((tile / lanes).max(held_slots).max(held_count).max(1)).max(1);
            let hw_ad_buf = if fit_on { self.scratch.entry("p3_used_hw_ad").or_default().need(cuda, cols32.max(8) * 8)? } else { 0 };
            let ar1 = self.scratch.entry("p3_arena").or_default().need(cuda, s1 * lanes * arena_col)?;
            let st1 = self.scratch.entry("p3_store").or_default().need(cuda, s1 * lanes * col_elems * sz_tl.max(sz_tape))?;
            let ks1 = self.scratch.entry("p3_kstore").or_default().need(cuda, s1 * lanes * 32)?;
            let w = mk(arena_col, if mode == 0 { 0 } else { 2 }, 2, ar1, st1, 0, ks1, 0, 0, 0);
            if w.len() != sbytes { return Err(format!("P3 launch struct {} bytes, image expects {sbytes}", w.len())); }
            dev::upload(cuda, lbuf, &w)?;
            dev::zero(cuda, sched, 64)?;
            dev::zero(cuda, fail, cols32 * 16)?;
            (*(if mode == 0 { m_tl } else { m_count })).set_global(cuda, "mw_sched", &sched.to_le_bytes())?;
            if fit_on { dev::zero(cuda, used_hw, cols32 * 8)?; }
            let mut a = vec![K::P(lbuf)];
            a.extend(sargs.iter().cloned());
            // mynn: wall of the reverse's count pass in the record (the blocking read below ends it). Work queued before
            // this launch drains inside the same wait; AD_P3_PASS_MS=1 syncs first for a clean split (diagnostic runs only).
            if mode == 1 && env_var("AD_P3_PASS_MS").as_deref() == Some("1") { dev::sync(cuda)?; }
            let t_count = std::time::Instant::now();
            // mynn (count at tangent time, opt-in AD_P3_COUNT_AT_TL=1). THE BREAKAGE: every MYNN-EDMF adjoint launch
            // ran the taped forward twice, once only to size each column's tape: 0.60-0.73 s of the 1.90 s/step the adjoint
            // takes at x1.40962 (a test machine <local path> w1-B9x), on a card that is busy issuing blocks, so no slot or arena change
            // removes it. The counts depend on the forward trajectory only, and the tangent pass visits the same launch
            // on the same trajectory first, where the card waits on the host most of the step (a test machine <local path>
            // PROFILE-fold-c-120-arena.md: 2.7 of 4.2 s/step of the tangent pass are host time). So the tangent launch is
            // followed by the count launch, whose counts and largest group tape stay on the card in a small block of their
            // own; the adjoint of that launch copies them into place and skips its count pass. Exact counts, as before: a
            // column that needed more than its count would refuse its pushes and fail the step (overflow slot), never a
            // wrong word. Not used with column shards, the arena fit, or when the window gave no step; a launch whose key
            // came up twice in one tangent step counts in the adjoint as before.
            let pre_key = (self.p3_step, kal, koff, env.get("batch_first").map_or(-1, |v| *v as i64), cols_here, cf_key.1);
            let pre_on = self.p3_step >= 0 && shard_plan.is_none() && self.p3_shards.is_none() && rep == 1 && env_var("AD_P3_COUNT_AT_TL").as_deref() == Some("1");
            let pre_buf = if mode == 1 && pre_on && !fit_on { self.p3_pre.get(&pre_key).copied().filter(|b| *b != 0) } else { None };
            if pre_buf.is_some() { cf_cap = None; cc = None; }
            if cc.is_some() { cf_cap = None; }
            if cf_cap.is_none() && pre_buf.is_none() && cc.is_none() {
                launch(cuda, f1, [s1 as u32, 1, 1], pblock, &a)?;
                count_fails(cols_here, &mut fails)?;
            }
            let mut rec = json!({"slots_first": s1, "resident_first": resident(f1)});
            if mode == 0 && pre_on {
                if self.p3_pre_seen.0 != self.p3_step { self.p3_pre_seen = (self.p3_step, Vec::new()); }
                if self.p3_pre_seen.1.contains(&pre_key) {
                    self.p3_pre.insert(pre_key, 0);
                    rec["count_at_tl"] = json!("key seen twice in this step: the adjoint counts");
                } else {
                    self.p3_pre_seen.1.push(pre_key);
                    let buf = match self.p3_pre.get(&pre_key).copied().filter(|b| *b != 0) {
                        Some(b) => b,
                        None => { let mut b = 0u64; checked("cuMemAlloc P3 tangent-time counts", (cuda.allocate)(&mut b, (cols_here + 8) * 8))?; b }, // kept (see Grow)
                    };
                    self.p3_pre.insert(pre_key, buf);
                    let mut w2 = mk(arena_col, 2, 2, ar1, st1, 0, ks1, 0, 0, 0);
                    let at = slots.len() * 48 + 12 * 8;   // the launch struct's `used` pointer
                    w2[at..at + 8].copy_from_slice(&buf.to_le_bytes());
                    dev::upload(cuda, lbuf, &w2)?;
                    dev::zero(cuda, sched, 64)?;
                    dev::zero(cuda, fail, cols32 * 16)?;
                    (*m_count).set_global(cuda, "mw_sched", &sched.to_le_bytes())?;
                    let mut a2 = vec![K::P(lbuf)];
                    a2.extend(sargs.iter().cloned());
                    let sc1 = groups.min(resident(f_count_p)).min(s1).max(1);
                    launch(cuda, f_count_p, [sc1 as u32, 1, 1], pblock, &a2)?;
                    count_fails(cols_here, &mut fails)?;
                    dev::copy(cuda, buf + (cols_here * 8) as u64, sched + 8, 8)?;
                    rec["count_at_tl"] = json!({"slots": sc1, "columns": cols_here});
                }
            }
            if let Some(b) = pre_buf { dev::copy(cuda, used, b, cols_here * 8)?; rec["count_at_tl"] = json!(true); }
            if let Some((n, why)) = cf_redo { rec["count_free_redo"] = json!({"columns": n, "why": why}); }
            if let Some(why) = cc_redo.as_ref() { rec["count_cache_redo"] = json!(why); }
            if mode == 1 {
                let region;
                if let Some(cap) = cf_cap {
                    // count-free: every column gets `cap` nodes; the kernel reads its capacity from `used`, as after a count pass
                    let bytes: Vec<u8> = std::iter::repeat(cap.to_le_bytes()).take(cols_here).flatten().collect();
                    dev::upload(cuda, used, &bytes)?;
                    region = lanes * (cap as usize + 16);
                    pers_counted = vec![cap; cols_here];
                    rec["count_free"] = json!({"cap_nodes": cap});
                } else if let Some((cnt, rg)) = cc.as_ref() {
                    // count cache: the kept counts go where the count pass leaves them; nothing is read back
                    // a12 bs-mynn (opt-in AD_P3_COUNT_CACHE_SLACK=<k>, k in 1..=16): each kept count c becomes c + (c >> k) + 64 and
                    // the largest group tape is taken again from those counts (the count pass's sum of count + 16 over each group
                    // of `lanes` columns). THE BREAKAGE: a gradient at a moved start (--x0-dir)
                    // follows a trajectory a little off the one whose counts were kept, so a column that takes a branch with a few
                    // more tape nodes overflows its exact kept count and the whole launch is counted and run again (one wasted
                    // record-and-walk-back pass). A capacity is only room: a column's tape is its own pushes in its own order,
                    // wherever it sits, so the words are those of a counted launch; a column that still needs more refuses as before.
                    if !cc_prev && pc_k.is_some() { self.p3_pc.insert(cf_key, (cnt.clone(), 0)); }
                    let slack = if cc_prev { None } else { env_var("AD_P3_COUNT_CACHE_SLACK").and_then(|v| v.parse::<u32>().ok()).filter(|k| (1..=16).contains(k)) };
                    let cnt2: Vec<i64> = match slack { Some(k) => cnt.iter().map(|&c| if c < 0 { c } else { c + (c >> k) + 64 }).collect(), None => cnt.clone() };
                    let bytes: Vec<u8> = cnt2.iter().flat_map(|c| c.to_le_bytes()).collect();
                    dev::upload(cuda, used, &bytes)?;
                    region = match slack {
                        Some(_) => cnt2.chunks(lanes.max(1)).map(|g| g.iter().map(|&u| u.max(0) as usize + 16).sum::<usize>()).max().unwrap_or(0).max(*rg),
                        None => *rg,
                    };
                    pers_counted = cnt2;
                    rec["count_cache"] = json!(if cc_prev { "previous counts of this shape, with slack".to_owned() } else { match slack { Some(k) => format!("hit, slack 1/2^{k} + 64"), None => "hit".to_owned() } });
                } else {
                    // (2) one host read: the counts (receipts) and the largest group tape (the adjoint's region per block)
                    let sc: Vec<u64> = match pre_buf {
                        Some(b) => vec![0, u64::from_le_bytes(dev::download(cuda, b + (cols_here * 8) as u64, 8)?[..8].try_into().unwrap())],
                        None => dev::download(cuda, sched, 16)?.chunks_exact(8).map(|x| u64::from_le_bytes(x.try_into().unwrap())).collect(),
                    };
                    region = sc[1] as usize;
                    pers_counted = dev::download(cuda, used, cols_here * 8)?.chunks_exact(8).map(|x| i64::from_le_bytes(x.try_into().unwrap())).collect();
                    rec["count_pass_ms"] = json!(t_count.elapsed().as_secs_f64() * 1e3);
                    let mx = pers_counted.iter().copied().max().unwrap_or(0).max(0);
                    let e = self.p3_cap_seen.entry(cf_key).or_insert(0);
                    *e = (*e).max(mx);
                    if pc_k.is_some() { self.p3_pc.insert(cf_key, (pers_counted.clone(), 0)); }
                    if cc_dir.is_some() {
                        if !self.p3_cc_bad.contains(&cc_key) { self.p3_cc.insert(cc_key, (pers_counted.clone(), region)); }
                        rec["count_cache"] = json!(match cc_file.as_ref() {
                            Some(f) => match p3_cc_write(f, &cc_head, region, &pers_counted) { Ok(()) => "filled".to_owned(), Err(e) => format!("kept in memory only, file not written: {e}") },
                            None => "filled (memory)".to_owned(),
                        });
                    }
                }
                // mynn (arena fit): this launch's count high-water, the previous adjoint launch's own (the ratio), and
                // from them the adjoint pass's arena bytes per column. The names below shadow the table's on purpose: the
                // slot and ring sizing that follows is the adjoint pass's.
                let max_u64 = |b: Vec<u8>| -> usize { b.chunks_exact(8).map(|x| u64::from_le_bytes(x.try_into().unwrap()) as usize).max().unwrap_or(0) };
                let hw_c = if !fit_on { 0 } else if cf_cap.is_some() { self.p3_hw_seen } else {
                    let h = max_u64(dev::download(cuda, used_hw, (s1 * lanes).min(cols32) * 8)?);
                    self.p3_hw_seen = self.p3_hw_seen.max(h);
                    self.p3_hw_seen
                };
                if let (true, Some((n_prev, hw_prev))) = (fit_on, self.p3_fit_pending.take()) {
                    let h = max_u64(dev::download(cuda, hw_ad_buf, n_prev.min(cols32) * 8)?);
                    if hw_prev > 0 && h > 0 { self.p3_fit_ratio = self.p3_fit_ratio.max(h as f64 / hw_prev as f64); }
                }
                let arena_col_table = arena_col;
                let arena_col = if fit_on && hw_c > 0 && self.p3_fit_ratio > 0.0 {
                    let need = (hw_c as f64 * self.p3_fit_ratio.max(1.0)).ceil() as usize;
                    ((need + need / 16 + 4096).div_ceil(4096) * 4096).min(arena_col_table)
                } else { arena_col_table };
                let col_fixed = arena_col + col_elems * sz_tl.max(sz_tape) + col_elems * 4 + 64;
                let slot_b = lanes * col_fixed;
                rec["arena_fit"] = json!({"on": fit_on, "table_col": arena_col_table, "count_hw": hw_c, "ad_col": arena_col, "ratio": self.p3_fit_ratio,
                                          "count_slots_held": held_slots});
                if cf_cap.is_none() {
                rec["nodes_total"] = json!(pers_counted.iter().map(|&u| u.max(0) as u64 + 16).sum::<u64>());
                rec["nodes_col_max"] = json!(pers_counted.iter().copied().max().unwrap_or(0));
                // mynn (receipt only): drift of the per-column counts against the last counted launch of this shape
                // (the next reverse step's launch): what a count-free adjoint launch planned from the previous counts
                // would have to absorb. No effect on the run.
                {
                    let key = (cols_here, env.get("initflag").map_or(false, |v| *v as i64 != 0));
                    if let Some(prev) = self.p3_prev_counts.get(&key).filter(|v| v.len() == pers_counted.len()) {
                        let (mut o125, mut o150, mut o200, mut worst, mut grew) = (0usize, 0usize, 0usize, 0.0f64, 0u64);
                        for (&c, &q) in pers_counted.iter().zip(prev.iter()) {
                            let (c, q) = (c.max(0), q.max(0));
                            if c > q + q / 4 + 64 { o125 += 1; }
                            if c > q + q / 2 + 64 { o150 += 1; }
                            if c > 2 * q + 64 { o200 += 1; }
                            if c > q { grew += (c - q) as u64; }
                            worst = worst.max(c as f64 / (q.max(1)) as f64);
                        }
                        rec["count_drift"] = json!({"columns": pers_counted.len(), "over_1.25": o125, "over_1.5": o150, "over_2": o200,
                                                    "worst_ratio": worst, "grown_nodes": grew, "prev_nodes": prev.iter().map(|&u| u.max(0) as u64).sum::<u64>()});
                    }
                    self.p3_prev_counts.insert(key, pers_counted.clone());
                }
                }
                // a refused column's one-column launch (p3_fallback) takes the budget measured before: one group runs on
                // one slot whatever the budget, and the ring is the one held or lent, so the budget changes no slot, no block and
                // no word there. THE BREAKAGE: cuMemGetInfo waits on the driver lock while another process on the box frees or
                // pins tens of GB of page-locked host memory (a program on the other card ending or starting): a test machine, four
                // boundary-layer backward steps 4.3 to 4.7 s longer (a 59-step later gradient step 36: the fall-back 4,288 ms
                // against 72 ms), each at a job end or start on card 1; a probe on another machine: cuMemGetInfo 0.82 s, cuMemAlloc 0.65 s
                // while a process holding 18 GB page-locked exits, a host-to-card copy never over 50 ms.
                let budget_ad = if let (true, Some(b)) = (self.p3_fallback, self.p3_ad_budget) { b } else {
                    let (mut free, mut total) = (0usize, 0usize);
                    checked("cuMemGetInfo", (cuda.mem_info)(&mut free, &mut total))?;
                    let held = ["p3_nodes", "p3_bars", "p3_arena", "p3_store", "p3_leaf", "p3_kstore"]
                        .iter().map(|k| self.scratch.get(k).map(|g| g.bytes).unwrap_or(0)).sum::<usize>();
                    let cap = (env_var("AD_P3_AD_BUDGET_GIB").and_then(|v| v.parse::<f64>().ok()).unwrap_or(32.0) * (1u64 << 30) as f64) as usize;
                    let reserve_ad = gib_word("AD_P3_AD_RESERVE_GIB", if total >= (64usize << 30) { 12usize << 30 } else { total / 8 });
                    let head = if total >= (64usize << 30) { 2usize << 30 } else { reserve_ad };
                    (free + held).saturating_sub(reserve_ad).min(cap).max(budget.min((free + held).saturating_sub(head)))
                };
                self.p3_ad_budget = Some(budget_ad);
                let per_slot = slot_b + region * (node_b + 8);
                // mynn-adjoint-v2 (pk120 fix): the slots are sized by the memory they ADD. budget_ad counts every P3 block
                // already held as room, but those blocks are kept (a grown block retires the old one, never freed: see Grow),
                // so the old `budget_ad / per_slot` let the tape take the room of the tangent's arena/store slots that the
                // adjoint does not use. a test machine mav2-K120G (pk120, 40,962 cells, 96 GB card at ~89 GB in use): the first
                // persistent adjoint (step 3) failed `cuMemAlloc scheme scratch 3386900480 bytes, CUDA status 2`. Now each
                // block's growth (tape 1/8 or 1/2 slack, arena/store/kstore/leaf 1/2, a block that already covers the slots
                // adds nothing) must fit budget_ad less everything held, the room the tiled path also leaves (room_raw).
                let hb = |k: &str| self.scratch.get(k).map(|g| g.bytes).unwrap_or(0);
                let (h_nodes, h_bars, h_ar, h_st, h_ks, h_lf) = (hb("p3_nodes"), hb("p3_bars"), hb("p3_arena"), hb("p3_store"), hb("p3_kstore"), hb("p3_leaf"));
                let room_new = budget_ad.saturating_sub(h_nodes + h_bars + h_ar + h_st + h_ks + h_lf);
                let grow = |want: usize, held: usize, den: usize| if want > held { want + want / den } else { 0 };
                let new_bytes = |s: usize, half: bool| -> usize {
                    let td = if half { 2 } else { 8 };
                    grow(s * region * node_b, h_nodes, td) + grow(s * region * 8, h_bars, td)
                        + grow(s * lanes * arena_col, h_ar, 2) + grow(s * lanes * col_elems * sz_tl.max(sz_tape), h_st, 2)
                        + grow(s * lanes * 32, h_ks, 2) + grow(s * lanes * col_elems * 4, h_lf, 2)
                };
                let s_top = groups.min(resident(f_ad_p)).min(budget_ad / per_slot.max(1)).min(cap_env).min((tile / lanes).max(1)).max(1);
                let (mut lo, mut hi) = (1usize, s_top);
                while lo < hi { let mid = (lo + hi + 1) / 2; if new_bytes(mid, false) <= room_new { lo = mid; } else { hi = mid - 1; } }
                let mut s2 = lo;
                // A tape block already held is reused while it gives at least half the slots (a grown block retires the
                // old one, which is never freed: see Grow).
                let have = self.scratch.get("p3_nodes").map(|g| g.bytes / node_b.max(1)).unwrap_or(0);
                if have >= region && have / region.max(1) >= s2.div_ceil(2) { s2 = s2.min(have / region.max(1)); }
                // ---- mynn-adjoint-v2: ring tape (layout 3) ------------------------------------------------------------
                // Layout 2 gives every block a region of the LARGEST group's tape, so a short card holds about half the
                // columns the packed tiled path holds in the same bytes (a test machine pk120: 88-115 slots, 704-920 columns in
                // flight in 4.4 GB, MYNN adjoint 31.0 s over the window against 28.3 s tiled, mav2-K120EC4 / K120T4).
                // The ring packs the groups in order by their own counts into ONE tape all blocks share; a block waits
                // until the groups whose ring nodes it would reuse are done (see mw_ring in gen_whole.py). The slots are
                // the ones the held arena/store already cover (growth at most a quarter of the room), the rest of the
                // room is ring. AD_P3_RING=0 keeps layout 2; an image without mw_ring keeps layout 2.
                // launches lane: the TEMPO arena, lent while TEMPO is idle (tempo_adw::LEND)
                let lend = if env_var("AD_PHYS_SHARED_ARENA").as_deref() != Some("0") { *crate::tempo_adw::LEND.lock().map_err(|_| "TEMPO lend lock poisoned")? } else { (0, 0) };
                let lend_nodes = if lend.0 != 0 { lend.1.saturating_sub(4096) / (node_b + 8) } else { 0 };
                let ring_img = env_var("AD_P3_RING").as_deref() != Some("0") && (*m_ad).set_global(cuda, "mw_ring", &0u64.to_le_bytes()).is_ok();
                let mut ring: Option<(usize, usize)> = None;
                if ring_img {
                    // Ring sizing (v2). Groups in flight = min(slots, ring nodes / mean group). Each slot past what the held
                    // arena/store cover reallocates them (1/2 slack, the old blocks stay held), each ring node costs 24 B
                    // (+1/8 slack). The slots are the most that keep the groups in flight at their maximum for the room (a plateau:
                    // slots past it only cost ring), and the ring is at most what they can use (slots x largest group, never more than
                    // the whole tape), so no ring bytes sit idle (a test machine mav2-G18R4 v1: a 30 GB ring for an 8.4 GB tape).
                    let nontape = |s: usize| grow(s * lanes * arena_col, h_ar, 2) + grow(s * lanes * col_elems * sz_tl.max(sz_tape), h_st, 2)
                        + grow(s * lanes * 32, h_ks, 2) + grow(s * lanes * col_elems * 4, h_lf, 2);
                    let total_nodes: usize = pers_counted.iter().map(|&u| u.max(0) as usize + 16).sum();
                    let mean_g = (total_nodes / groups.max(1)).max(1);
                    // launches lane: the lent TEMPO arena (lend, above) counts as a held ring
                    let have_ring = (h_nodes / node_b.max(1)).min(h_bars / 8).max(lend_nodes);
                    let fresh = |s: usize| room_new.saturating_sub(nontape(s)) / ((node_b + 8) * 9 / 8);
                    let ringcap = |s: usize| have_ring.max(fresh(s));
                    let s_cap = groups.min(resident(f_ad_p)).min(cap_env).max(1);
                    let (mut best_s, mut best_e) = (1usize, 0usize);
                    let mut sc = 1usize;
                    while sc <= s_cap {
                        // a slot count whose arena/store/kstore/leaf growth does not fit the room is out (a test machine mav2-K120F4,
                        // ring v2: the second pk120 launch reused the ring, so every larger slot count tied, the plateau rule
                        // took 2,256 slots and asked for 23.9 GB of arena: cuMemAlloc CUDA status 2)
                        let e = if nontape(sc) > room_new { 0 } else { sc.min(ringcap(sc) / mean_g) };
                        if e >= best_e { best_e = e; best_s = sc; }
                        sc = if sc < s_cap && sc + 8 > s_cap { s_cap } else { sc + 8 };
                    }
                    let want = (best_s * region).min(total_nodes + region);
                    let pn = if have_ring >= want.min(ringcap(best_s)) && have_ring >= region { have_ring } else { want.min(fresh(best_s)) };
                    // mynn (fold 22:43Z, a test machine gate-20261109T2228Z-439f758f b-120 against c-120): when other work freed
                    // 3 GB of the card, this rule allocated a ring of its own (p3_nodes 6.31 + p3_bars 3.16 GiB, 423.6 M nodes)
                    // instead of borrowing the idle TEMPO arena (357.9 M nodes, nothing allocated): 9.5 GB of card for
                    // 0.07 s/step (1.940 -> 1.869). Now, while no ring of its own is held, the lent ring is taken whenever
                    // it keeps at least 4/5 of the groups in flight the fresh ring would. Where the tape lives only:
                    // the words do not change. AD_P3_RING_OWN=1 keeps the old rule.
                    let own_held = (h_nodes / node_b.max(1)).min(h_bars / 8);
                    let (best_s, pn) = if lend_nodes >= region && lend_nodes > 0 && pn > lend_nodes && own_held < region && env_var("AD_P3_RING_OWN").as_deref() != Some("1") {
                        let (mut bs, mut be) = (1usize, 0usize);
                        let mut sc = 1usize;
                        while sc <= s_cap {
                            let e = if nontape(sc) > room_new { 0 } else { sc.min(lend_nodes / mean_g) };
                            if e >= be { be = e; bs = sc; }
                            sc = if sc < s_cap && sc + 8 > s_cap { s_cap } else { sc + 8 };
                        }
                        if be > 0 && be * 5 >= best_e * 4 { (bs, lend_nodes) } else { (best_s, pn) }
                    } else { (best_s, pn) };
                    // s60-cardmem (AD_P3_ROOM_RULE=1). THE BREAKAGE: the room above is budget_ad, which keeps 12 GiB free (and
                    // its floor, the first launch's tangent budget, 16 GiB) whatever the card holds, and the ring is the lent
                    // TEMPO arena unless a fresh one keeps 5/4 as many groups in flight. At 60 km on a 96 GB card that gave a
                    // 4 GiB ring for 761 slots (b8-head: 11.3 wraps, the ring that keeps 761 slots busy is 15.2 GB, L8-r5), and in
                    // a gradient-only run 305 to 457 slots (G29-f6: budget 4.41 to 5.93 GB). With the rule the slots are the
                    // most whose ring that keeps them all busy (ring_room::ring_for_slots, at most the ring that never wraps)
                    // and slot growth fit the card's free bytes now less the larger of the other cold families and this
                    // scheme's own cold blocks, and 1/64 of the card (card_room::bl_fits); that ring is the lent arena when it
                    // is large enough, else a ring of its own made as device memory (spill::warm), so it is not copied out
                    // to the host when the land scheme's blocks come in. Taken when it keeps more groups in flight than the
                    // rule above. Where the tape lives and how many slots run only: the words do not change.
                    let mut room_rule = Value::Null;
                    // s60-cardmem (AD_CARD_ROOM_FILE without the rule): the rings this launch would want, for the next run's
                    // share-out at the reverse start (every resident slot busy; never wrapping)
                    if !crate::card_room::bl_on() && crate::card_room::file().is_some() {
                        let gsz_n = crate::ring_room::group_sizes(&pers_counted[..cols_here.min(pers_counted.len())], lanes);
                        let (nb, nw) = ((node_b + 8) as u64, (total_nodes + region) as u64);
                        crate::card_room::note_bl(crate::ring_room::ring_for_slots(&gsz_n, s_cap).min(nw).max(region as u64) * nb, nw * nb);
                    }
                    let (best_s, pn) = if crate::card_room::bl_on() {
                        let (mut free_r, mut total_r) = (0usize, 0usize);
                        checked("cuMemGetInfo", (cuda.mem_info)(&mut free_r, &mut total_r))?;
                        let fam = crate::spill::cold_families();
                        let others = (fam.iter().filter(|f| f.0 != "p3").map(|f| f.1).max().unwrap_or(0) as u64)
                            .max(gib_word("AD_P3_ROOM_KEEP_GIB", 0) as u64);
                        let p3_cold = fam.iter().find(|f| f.0 == "p3").map(|f| f.1).unwrap_or(0) as u64;
                        let gsz_r = crate::ring_room::group_sizes(&pers_counted[..cols_here.min(pers_counted.len())], lanes);
                        let nb = (node_b + 8) as u64;
                        let nowrap_nodes = (total_nodes + region) as u64;
                        let ring_of = |s: usize| crate::ring_room::ring_for_slots(&gsz_r, s).min(nowrap_nodes).max(region as u64) * nb;
                        let lend_b = lend_nodes as u64 * nb;
                        let own_b = ((h_nodes / node_b.max(1)).min(h_bars / 8)) as u64 * nb;
                        // a ring the lent arena holds costs nothing; a larger one is made whole (the lent arena is not part of it)
                        // checker: a ring the lent arena or the held own ring holds costs nothing; a larger one is made whole (Grow retires
                        // the held block, it stays allocated), so nothing of the held ring is subtracted (ring_have 0 below)
                        let cost_of = |s: usize| { let r = ring_of(s); if r <= lend_b || r <= own_b { 0 } else { r } };
                        let nontape_of = |s: usize| nontape(s) as u64;
                        crate::card_room::note_bl(ring_of(s_cap), nowrap_nodes * nb);
                        let before_inflight = best_s.min(pn / mean_g);
                        let pick = crate::card_room::bl_choose(s_cap, free_r as u64, total_r as u64, others, p3_cold, 0, &cost_of, &nontape_of);
                        room_rule = json!({"free_bytes": free_r, "total_bytes": total_r, "cold_other_peak_bytes": others, "own_cold_bytes": p3_cold,
                                           "lent_ring_bytes": lend_b, "own_ring_bytes": own_b, "ring_full_bytes": ring_of(s_cap), "ring_nowrap_bytes": nowrap_nodes * nb,
                                           "pick": pick.map(|p| json!([p, ring_of(p)])), "inflight_before": before_inflight});
                        match pick {
                            Some(s) if s > before_inflight => {
                                let nodes = (ring_of(s) / nb) as usize;
                                room_rule["taken"] = json!(true);
                                (s, if nodes <= lend_nodes { lend_nodes } else { nodes })
                            },
                            _ => { room_rule["taken"] = json!(false); (best_s, pn) },
                        }
                    } else { (best_s, pn) };
                    if !room_rule.is_null() { rec["room_rule"] = room_rule; }
                    if pn >= region { ring = Some((best_s, pn)); }
                }
                if let Some((sr, pn0)) = ring {
                    if let Some(cap) = cf_cap {
                        let inflight = sr.min(pn0.max(lend_nodes) / (lanes * (cap as usize + 16)).max(1));
                        let held_ring = (h_nodes / node_b.max(1)).min(h_bars / 8).max(lend_nodes);
                        if inflight * 2 < groups.min(resident(f_ad_p)) || pn0 > held_ring {
                            cf_cap = None; cf_redo = Some((0, "uniform capacity leaves too few groups in flight")); caps.truncate(caps_len0);
                            continue 'cf;
                        }
                    }
                    let lent = lend_nodes >= pn0 && lend_nodes > 0;
                    let (nodes, bars, pn) = if lent {
                        // nodes then bars in the lent TEMPO arena (bars 256-byte aligned); nothing is allocated
                        (lend.0, lend.0 + (((pn0 * node_b) as u64 + 255) & !255), pn0)
                    } else {
                        // s60-cardmem: under AD_P3_ROOM_RULE the ring of its own is device memory (see room_rule above)
                        let _warm = crate::card_room::bl_on().then(crate::spill::warm);
                        let nodes = self.scratch.entry("p3_nodes").or_default().need_slack(cuda, pn0 * node_b, false)?;
                        let bars = self.scratch.entry("p3_bars").or_default().need_slack(cuda, pn0 * 8, false)?;
                        // 60 km tape: the room file's reverse growth does not count this ring (card_room::growth)
                        self.note_own_ring();
                        (nodes, bars, (self.scratch.get("p3_nodes").map(|g| g.bytes).unwrap_or(0) / node_b.max(1)).min(self.scratch.get("p3_bars").map(|g| g.bytes).unwrap_or(0) / 8))
                    };
                    rec["ring_lent"] = json!(lent);
                    let ar2 = self.scratch.entry("p3_arena").or_default().need(cuda, sr * lanes * arena_col)?;
                    let st2 = self.scratch.entry("p3_store").or_default().need(cuda, sr * lanes * col_elems * sz_tl.max(sz_tape))?;
                    let ks2 = self.scratch.entry("p3_kstore").or_default().need(cuda, sr * lanes * 32)?;
                    let lf2 = self.scratch.entry("p3_leaf").or_default().need(cuda, sr * lanes * col_elems * 4)?;
                    // the ring plan: vstart[groups + 1] (virtual, monotone; a group never straddles the ring end), phys[groups], done[groups]
                    // ring-60km: the same plan and bytes from crate::ring_room (CPU-tested against this loop as it was).
                    let pnu = pn as u64;
                    if pers_counted.len() < cols_here { return Err(format!("DEFECT P3 ring plan: {} counts for {cols_here} columns", pers_counted.len())); }
                    let gsz = crate::ring_room::group_sizes(&pers_counted[..cols_here], lanes);
                    let plan = crate::ring_room::ring_plan(&gsz, pnu)?;
                    let rb = crate::ring_room::ring_bytes(&plan);
                    let rbuf = self.scratch.entry("p3_ring").or_default().need(cuda, rb.len())?;
                    dev::upload(cuda, rbuf, &rb)?;
                    (*m_ad).set_global(cuda, "mw_ring", &rbuf.to_le_bytes())?;
                    let w = mk(arena_col, 1, 3, ar2, st2, lf2, ks2, nodes, bars, pnu);
                    dev::upload(cuda, lbuf, &w)?;
                    dev::zero(cuda, sched, 64)?;
                    dev::zero(cuda, fail, cols32 * 16)?;
                    (*m_ad).set_global(cuda, "mw_sched", &sched.to_le_bytes())?;
                    if fit_on {
                        (*m_ad).set_global(cuda, "mw_arena_limit", &(arena_col as u64).to_le_bytes())?;
                        (*m_ad).set_global(cuda, "mw_arena_hw", &hw_ad_buf.to_le_bytes())?;
                        dev::zero(cuda, hw_ad_buf, cols32 * 8)?;
                        self.p3_fit_pending = Some((sr * lanes, hw_c));
                    }
                    let mut a = vec![K::P(lbuf)];
                    a.extend(sargs.iter().cloned());
                    launch(cuda, f_ad_p, [sr as u32, 1, 1], pblock, &a)?;
                    if cf_cap.is_some() || cc.is_some() {
                        // count-free and count cache: the overflow slots of this launch decide (one wait, where the count pass's read was)
                        let fl = dev::download(cuda, fail, cols_here * 16)?;
                        let over = fl.chunks_exact(16).filter(|c| c[..4] != [0u8, 0, 0, 0]).count();
                        if over != 0 {
                            if cc.take().is_some() {
                                if cc_prev { cc_prev = false; cc_redo = Some(format!("{over} columns needed more than the previous counts: counted and run again")); }
                                else {
                                self.p3_cc.remove(&cc_key); self.p3_cc_bad.insert(cc_key);
                                cc_redo = Some(format!("{over} columns needed more than their kept counts: counted and run again"));
                                }
                            } else { cf_redo = Some((over, "tape overflow")); }
                            cf_cap = None; caps.truncate(caps_len0); self.p3_fit_pending = None;
                            continue 'cf;
                        }
                    }
                    count_fails(cols_here, &mut fails)?;
                    let tot: usize = pers_counted.iter().map(|&u| u.max(0) as usize + 16).sum();
                    caps.push(region);
                    pers_tiles = Some((vec![(0, cols, tot)], budget_ad));
                    rec["layout"] = json!(3); rec["slots_ad"] = json!(sr); rec["resident_ad"] = json!(resident(f_ad_p)); rec["region_nodes"] = json!(region);
                    rec["ring_nodes"] = json!(pn); rec["ring_wraps"] = json!(plan.wraps); rec["tape_bytes_held"] = json!(pn * (node_b + 8));
                    // ring-60km (receipt only, no effect on the run): the groups this ring lets be in flight (in flight is the
                    // lesser of this and slots_ad), and the ring that would keep all slots_ad slots busy, from this launch's
                    // counts; ring_nodes_for_slots x (node_b + 8) bytes is the ring to ask for (AD_TEMPO_ARENA_GIB when lent).
                    let (inf_min, inf_mean) = crate::ring_room::ring_inflight(&plan, &gsz, pnu);
                    let rfs = crate::ring_room::ring_for_slots(&gsz, sr);
                    rec["ring_inflight_groups"] = json!({"min": inf_min, "mean": inf_mean});
                    rec["ring_nodes_for_slots"] = json!(rfs); rec["ring_bytes_for_slots"] = json!(rfs as usize * (node_b + 8));
                    rec["ring_bound"] = json!(if inf_mean < sr as f64 { "ring" } else { "slots" });
                    rec["budget_ad"] = json!(budget_ad); rec["room_new"] = json!(room_new); rec["slots_top"] = json!(s_top);
                } else {
                if cf_cap.is_some() { cf_cap = None; cf_redo = Some((0, "no ring tape")); caps.truncate(caps_len0); continue 'cf; }
                if cc.take().is_some() { if !cc_prev { self.p3_cc_bad.insert(cc_key); } cc_prev = false; cc_redo = Some("no ring tape: counted".to_owned()); caps.truncate(caps_len0); continue 'cf; }
                let half = have == 0 && new_bytes(s2, true) <= room_new;
                let nodes = self.scratch.entry("p3_nodes").or_default().need_slack(cuda, s2 * region * node_b, half)?;
                let bars = self.scratch.entry("p3_bars").or_default().need_slack(cuda, s2 * region * 8, half)?;
                let ar2 = self.scratch.entry("p3_arena").or_default().need(cuda, s2 * lanes * arena_col)?;
                let st2 = self.scratch.entry("p3_store").or_default().need(cuda, s2 * lanes * col_elems * sz_tl.max(sz_tape))?;
                let ks2 = self.scratch.entry("p3_kstore").or_default().need(cuda, s2 * lanes * 32)?;
                let lf2 = self.scratch.entry("p3_leaf").or_default().need(cuda, s2 * lanes * col_elems * 4)?;
                // (3) the adjoint over every column, tape regions of `region` nodes per block
                let w = mk(arena_col, 1, 2, ar2, st2, lf2, ks2, nodes, bars, region as u64);
                dev::upload(cuda, lbuf, &w)?;
                dev::zero(cuda, sched, 64)?;
                dev::zero(cuda, fail, cols32 * 16)?;
                (*m_ad).set_global(cuda, "mw_sched", &sched.to_le_bytes())?;
                if fit_on {
                    (*m_ad).set_global(cuda, "mw_arena_limit", &(arena_col as u64).to_le_bytes())?;
                    (*m_ad).set_global(cuda, "mw_arena_hw", &hw_ad_buf.to_le_bytes())?;
                    dev::zero(cuda, hw_ad_buf, cols32 * 8)?;
                    self.p3_fit_pending = Some((s2 * lanes, hw_c));
                }
                let mut a = vec![K::P(lbuf)];
                a.extend(sargs.iter().cloned());
                launch(cuda, f_ad_p, [s2 as u32, 1, 1], pblock, &a)?;
                count_fails(cols_here, &mut fails)?;
                let tot: usize = pers_counted.iter().map(|&u| u.max(0) as usize + 16).sum();
                caps.push(region);
                pers_tiles = Some((vec![(0, cols, tot)], budget_ad));
                rec["slots_ad"] = json!(s2); rec["resident_ad"] = json!(resident(f_ad_p)); rec["region_nodes"] = json!(region);
                rec["tape_bytes_held"] = json!(s2 * region * (node_b + 8));
                rec["budget_ad"] = json!(budget_ad); rec["room_new"] = json!(room_new); rec["slots_top"] = json!(s_top);
                }
            }
            break 'cf rec;
            };
            persisted = Some(rec_cf);
            first = cols;
        }
        Ok(())
        })();
        // mynn (receipt): what the dycore card's own part of this launch cost the host to issue (its adjoint launch
        // is queued, not waited for); AD_P3_PASS_MS=1 (diagnostic runs only) waits for it here, so the second figure is
        // the card time of this card's share while the helpers run on theirs.
        let prim_issue_ms = t_prim.elapsed().as_secs_f64() * 1e3;
        let prim_done_ms = if shard_sent > 0 && env_var("AD_P3_PASS_MS").as_deref() == Some("1") { let _ = dev::sync(cuda); Some(t_prim.elapsed().as_secs_f64() * 1e3) } else { None };
        // Drain every submitted helper, even after a failure here: no helper may still write this launch's buffers.
        if shard_sent > 0 {
            let sh = self.p3_shards.as_ref().ok_or("DEFECT MYNN shard pool vanished")?;
            let mut dones: Vec<P3Done> = Vec::new();
            let mut err: Option<String> = None;
            for w in &sh.workers[..shard_sent] {
                match w.rx.lock().map_err(|_| "DEFECT MYNN shard reply lock".to_owned()).and_then(|rx| rx.recv().map_err(|_| "DEFECT MYNN shard reply missing".to_owned())) {
                    Ok(Ok(d)) => dones.push(d),
                    Ok(Err(e)) | Err(e) => { if err.is_none() { err = Some(e); } },
                }
            }
            primary_run?;
            if let Some(e) = err { return Err(e); }
            let mut hf = [0u64; 4];
            for d in &dones { for i in 0..4 { hf[i] += d.fails[i]; } }
            if hf[0] + hf[2] + hf[3] != 0 {
                return Err(format!("P3 mode {mode} shard failure slots: overflow {} domain {} context/unencodable node {} arena {}", hf[0], hf[1], hf[2], hf[3]));
            }
            fails[1] += hf[1];
            if let Some(r) = persisted.as_mut() {
                r["shards"] = json!({"ranges": shard_plan.as_ref().map(|v| v.iter().map(|x| json!([x.first, x.count])).collect::<Vec<_>>()),
                                     "ordinals": sh.ordinals, "helpers": dones.iter().map(|d| &d.receipt).collect::<Vec<_>>(),
                                     "helper_domain_flags": hf[1], "wall_ms": shard_wall.elapsed().as_secs_f64() * 1000.0,
                                     "primary_columns": cols_here, "primary_issue_ms": prim_issue_ms, "primary_done_ms": prim_done_ms, "primary_share": prim_share});
            }
        } else { primary_run?; }
        // adjoint-speed: the reverse's count pass once over every column, then tiles packed by the measured counts.
        // tempo-mynn-kernels: when the launch does not fit one tile, the same packed reverse runs per count tile (the
        // budget-sized tile above) instead of the 384-column count + capacity-layout tiles below. pk120 (40,962 columns,
        // arena 0.88 MB per column) ran 107 serial 384-column tiles per launch, 12-48 one-warp blocks each on a 188-SM
        // card: MYNN was 71-85 s of its 208.7 s TL+AD step. Columns are independent (G5): tiles only group columns, so
        // the reverse words do not change. With one count tile (the whole launch) this is the code it replaces.
        let mut ad_tiles: Option<(Vec<(usize, usize, usize)>, usize)> = None;
        let mut first_counted: Vec<i64> = Vec::new();
        let multi = mode == 1 && !whole && !self.p3_pinned && node_b == 16 && tile > 0;
        let ctiles: Vec<(usize, usize)> = if persisted.is_some() { vec![] } else if mode == 1 && whole { vec![(0, cols)] }
            else if multi { (0..cols).step_by(tile).map(|c| (c, tile.min(cols - c))).collect() } else { vec![] };
        let mut all_t: Vec<(usize, usize, usize)> = Vec::new();
        let mut room_last = 0usize;
        for &(ct0, ctn) in &ctiles {
            let n = ctn;
            let mut w: Vec<u8> = Vec::with_capacity(sbytes);
            let put = |w: &mut Vec<u8>, v: u64| w.extend_from_slice(&v.to_le_bytes());
            for sl in &slots { for v in [sl[0], sl[1], 0, 0, 0, 0] { put(&mut w, v); } }
            for v in [kin, kout, eflags, arena, arena_col as u64, store, col_elems as u64, 0, kstore, 0, 0, 0, used, fail] { put(&mut w, v); }
            for v in [ct0 as i32, n as i32, 2, 0] { w.extend_from_slice(&v.to_le_bytes()); }
            if w.len() != sbytes { return Err(format!("P3 launch struct {} bytes, image expects {sbytes}", w.len())); }
            dev::upload(cuda, lbuf, &w)?;
            dev::zero(cuda, arena, tile * arena_col)?;
            dev::zero(cuda, store, store_b)?;
            dev::zero(cuda, fail, tile * 16)?;
            let mut a = vec![K::P(lbuf)];
            a.extend(sargs.iter().cloned());
            launch(cuda, f_count, pgrid(n), pblock, &a)?;
            count_fails(n, &mut fails)?;
            // The per-column tape counts size the reverse tiles: a host decision, so this one read stays (blocking download).
            let counted: Vec<i64> = dev::download(cuda, used, n * 8)?.chunks_exact(8).map(|x| i64::from_le_bytes(x.try_into().unwrap())).collect();
            // Pack 32-column groups while columns x (largest count + 16) x 32 bytes of tape fits what the budget leaves
            // after the arena/store of the widest tile (bounded below by one 32-column group, as the old tiling is).
            // adjoint-speed-2: the budget for these tiles is read now, when the reverse runs: the tangent's budget was
            // read at the first tangent step, before the window tape and TEMPO arena existed, and its 10 GiB default
            // cut pkg18m2's reverse into 3 tiles (1.60 s per step; 1 tile 0.69 s, the same words: tiles only group
            // columns). Card free now plus the tape scratch this row already holds, less 12 GiB, at most
            // AD_P3_AD_BUDGET_GIB (default 32), never below the tangent's budget; the slab grows by 1/8 (need_slack8).
            // as in the persistent path, a refused column's one-column launch takes the budget measured before.
            let budget_ad = if let (true, Some(b)) = (self.p3_fallback, self.p3_ad_budget) { b } else {
                let (mut free, mut total) = (0usize, 0usize);
                checked("cuMemGetInfo", (cuda.mem_info)(&mut free, &mut total))?;
                // tempo-mynn-kernels: the tile's arena/store/leaf blocks were allocated above (before the count pass), so
                // free memory already excludes them, while room_raw below subtracts tile * col_fixed for them again. On a
                // 32 GB card with a whole-launch tile that double count took ~3.2 GB of real room: a test machine tm-N18b packed 81
                // reverse tiles (one per 32-column group) and the MYNN-EDMF adjoint took 12-17 s per step. Count them as held.
                let held = ["p3_nodes", "p3_bars", "p3_arena", "p3_store", "p3_leaf", "p3_kstore", "p3_fail", "p3_used", "p3_used_hw", "p3_dom"]
                    .iter().map(|k| self.scratch.get(k).map(|g| g.bytes).unwrap_or(0)).sum::<usize>();
                let cap = (env_var("AD_P3_AD_BUDGET_GIB").and_then(|v| v.parse::<f64>().ok()).unwrap_or(32.0) * (1u64 << 30) as f64) as usize;
                // tempo-mynn-kernels: reserve 12 GiB, or an eighth of a card under 64 GiB; the floor at the tangent budget
                // never exceeds what is free (less 2 GiB), so a tight card packs smaller reverse tiles instead of failing.
                let reserve_ad = gib_word("AD_P3_AD_RESERVE_GIB", if total >= (64usize << 30) { 12usize << 30 } else { total / 8 });
                // tempo-mynn-adjoint-kernels: below 64 GiB the floor keeps the same eighth of the card free as the main term:
                // with the arena counted as held (above), a 2 GiB headroom let the 18-step window's MYNN tape take the memory
                // the RUC land adjoint needs later in the step (a test machine tm-R18: cuMemAlloc 1 GiB failed, CUDA 2, step 18).
                let head = if total >= (64usize << 30) { 2usize << 30 } else { reserve_ad };
                (free + held).saturating_sub(reserve_ad).min(cap).max(budget.min((free + held).saturating_sub(head)))
            };
            self.p3_ad_budget = Some(budget_ad);
            // tempo-mynn-kernels: the arena/store held are the tile's (whole: tile == cols32, as before).
            let room_raw = budget_ad.saturating_sub(tile * col_fixed);
            // Split count tiles never pack smaller reverse tiles than the 384-column capacity tile they replace allocated
            // (that path sized its tape from the counts alone, not from free memory).
            let room_raw = if multi { room_raw.max(self.p3_tile.min(n) * (counted.iter().copied().max().unwrap_or(0).max(0) as usize + 16) * (node_b + 8)) } else { room_raw };
            let room = (room_raw / 9 * 8).max(1);
            let mut t: Vec<(usize, usize, usize)> = Vec::new();
            if node_b == 16 {
                // mynn-tape: packed tape. A column takes its own measured count + 16 nodes (16 B node + 8 B bar), not the
                // tile's largest count; 32-column groups are packed while the tile's sum fits the room. The third tuple
                // entry is the tile's total nodes.
                let (mut c0, mut tot) = (0usize, 0usize);
                let mut c = 0usize;
                while c < n {
                    let g1 = (c + 32).min(n);
                    let gn: usize = counted[c..g1].iter().map(|&u| u.max(0) as usize + 16).sum();
                    if c > c0 && (tot + gn) * (node_b + 8) > room { t.push((ct0 + c0, c - c0, tot)); c0 = c; tot = 0; }
                    tot += gn; c = g1;
                }
                t.push((ct0 + c0, n - c0, tot));
            } else {
                let (mut c0, mut mx) = (0usize, 0i64);
                let mut c = 0usize;
                while c < n {
                    let g1 = (c + 32).min(n);
                    let gm = counted[c..g1].iter().copied().max().unwrap_or(0).max(0);
                    let nm = mx.max(gm);
                    let ncols = g1 - c0;
                    if c > c0 && ncols.div_ceil(32) * 32 * (nm as usize + 16) * 32 > room {
                        t.push((ct0 + c0, c - c0, mx as usize + 16)); c0 = c; mx = gm;
                    } else { mx = nm; }
                    c = g1;
                }
                t.push((ct0 + c0, n - c0, mx as usize + 16));
            }
            let packed = node_b == 16;
            let biggest = t.iter().map(|x| x.2).max().unwrap_or(0);
            for &(c0, n, cap) in &t {
                caps.push(cap);
                let (nodes, bars, offs) = if packed {
                    // half slack at the first allocation when it fits the room (a retired block is never freed), else 1/8
                    let half = self.scratch.get("p3_nodes").map_or(true, |g| g.bytes == 0) && biggest * (node_b + 8) * 3 / 2 <= room_raw;
                    let nodes = self.scratch.entry("p3_nodes").or_default().need_slack(cuda, cap * node_b, half)?;
                    let bars = self.scratch.entry("p3_bars").or_default().need_slack(cuda, cap * 8, half)?;
                    let mut o: Vec<u8> = Vec::with_capacity((n + 1) * 8);
                    let mut acc = 0i64;
                    o.extend_from_slice(&acc.to_le_bytes());
                    for k in 0..n { acc += counted[c0 - ct0 + k].max(0) + 16; o.extend_from_slice(&acc.to_le_bytes()); }
                    if acc as usize != cap { return Err(format!("P3 packed tape offsets {acc} != tile nodes {cap}")); }
                    let ob = self.scratch.entry("p3_off").or_default().need(cuda, o.len())?;
                    dev::upload(cuda, ob, &o)?;
                    (nodes, bars, ob)
                } else {
                    (self.scratch.entry("p3_nodes").or_default().need_slack8(cuda, n.div_ceil(32) * 32 * cap * node_b)?,
                     self.scratch.entry("p3_bars").or_default().need_slack8(cuda, n.div_ceil(32) * 32 * cap * 8)?, used)
                };
                let mut w: Vec<u8> = Vec::with_capacity(sbytes);
                let put = |w: &mut Vec<u8>, v: u64| w.extend_from_slice(&v.to_le_bytes());
                for sl in &slots { for v in *sl { put(&mut w, v); } }
                for v in [kin, kout, eflags, arena, arena_col as u64, store, col_elems as u64, leaf, kstore, nodes, bars, if packed { 0 } else { cap as u64 }, offs, fail] { put(&mut w, v); }
                for v in [c0 as i32, n as i32, 1, packed as i32] { w.extend_from_slice(&v.to_le_bytes()); }
                if w.len() != sbytes { return Err(format!("P3 launch struct {} bytes, image expects {sbytes}", w.len())); }
                dev::upload(cuda, lbuf, &w)?;
                dev::zero(cuda, arena, tile * arena_col)?;
                dev::zero(cuda, store, store_b)?;
                dev::zero(cuda, fail, tile * 16)?;
                let mut a = vec![K::P(lbuf)];
                a.extend(sargs.iter().cloned());
                launch(cuda, f_ad, pgrid(n), pblock, &a)?;
                count_fails(n, &mut fails)?;
            }
            first_counted.extend_from_slice(&counted);
            all_t.extend(t);
            room_last = room_raw;
        }
        if !ctiles.is_empty() {
            ad_tiles = Some((all_t, room_last));
            first = cols;
        }
        if persisted.is_some() && mode == 1 { ad_tiles = pers_tiles.take(); first_counted = std::mem::take(&mut pers_counted); }
        while first < cols {
            let n = tile.min(cols - first);
            let run = |mode_k: i32, f: *mut c_void, nodes: u64, bars: u64, capacity: usize, leafp: u64, fails: &mut [u64; 4]| -> Result<Vec<u8>, String> {
                let mut w: Vec<u8> = Vec::with_capacity(sbytes);
                let put = |w: &mut Vec<u8>, v: u64| w.extend_from_slice(&v.to_le_bytes());
                for sl in &slots {
                    let sl = if mode_k == 2 { [sl[0], sl[1], 0, 0, 0, 0] } else { *sl };
                    for v in sl { put(&mut w, v); }
                }
                for v in [kin, kout, eflags, arena, arena_col as u64, store, col_elems as u64, leafp, kstore, nodes, bars, capacity as u64, used, fail] { put(&mut w, v); }
                for v in [first as i32, n as i32, mode_k, 0] { w.extend_from_slice(&v.to_le_bytes()); }
                if w.len() != sbytes { return Err(format!("P3 launch struct {} bytes, image expects {sbytes}", w.len())); }
                dev::upload(cuda, lbuf, &w)?;
                dev::zero(cuda, arena, tile * arena_col)?;
                dev::zero(cuda, store, store_b)?;
                dev::zero(cuda, fail, tile * 16)?;
                let mut a = vec![K::P(lbuf)];
                a.extend(sargs.iter().cloned());
                launch(cuda, f, pgrid(n), pblock, &a)?;
                count_fails(n, fails)?;
                // mode 2 (count pass): the counts size this tile's tape, a host decision (blocking download).
                if mode_k == 2 { dev::download(cuda, used, n * 8) } else { Ok(vec![]) }
            };
            if mode == 0 {
                run(0, f_tl, 0, 0, 0, 0, &mut fails)?;
            } else {
                let u = run(2, f_count, 0, 0, 0, 0, &mut fails)?;
                let cap = u.chunks_exact(8).map(|x| i64::from_le_bytes(x.try_into().unwrap())).max().unwrap_or(0).max(0) as usize + 16;
                caps.push(cap);
                let nodes = self.scratch.entry("p3_nodes").or_default().need(cuda, tile * cap * node_b)?;
                let bars = self.scratch.entry("p3_bars").or_default().need(cuda, tile * cap * 8)?;
                run(1, f_ad, nodes, bars, cap, leaf, &mut fails)?;
            }
            first += n;
        }
        if codes.is_none() && fails[0] + fails[2] + fails[3] != 0 {
            return Err(format!("P3 mode {mode} failure slots: overflow {} domain {} context/unencodable node {} arena {}", fails[0], fails[1], fails[2], fails[3]));
        }
        if identity {
            // Mirror the forward's post state of every active segment beside the y buffers, contiguously.
            let mirror = ybase + total as u64;
            let mut at = 0u64;
            for s in p.segs.v.iter().filter(|s| s.active) {
                dev::copy(cuda, mirror + at, s.live, s.hi - s.lo)?;
                at += (s.hi - s.lo) as u64;
            }
            p.identity = Some((ybase, mirror, total));
        }
        let cmax = caps.iter().max().copied();
        // mynn-adjoint-v2: AD_P3_ARENA_RECEIPT=1 reads the arena high-water per column of the last tile (bytes of the
        // logical per-column offset) so the arena multiplier can be sized from use. Off by default: a timed run carries
        // no extra readback.
        let arena_hw = if env_var("AD_P3_ARENA_RECEIPT").as_deref() == Some("1") {
            dev::sync(cuda)?;
            let v: Vec<u64> = dev::download(cuda, used_hw, tile * 8)?.chunks_exact(8).map(|x| u64::from_le_bytes(x.try_into().unwrap())).collect();
            Some(json!({"max": v.iter().max(), "arena_col": arena_col, "ws": ws}))
        } else { None };
        let domain = if codes.is_some() { json!("in window_device_flags at the step end") } else { json!(fails[1]) };
        let mut out = json!({"domain_flags": domain, "tile_columns": tile, "tape_nodes_per_column_max": if node_b == 16 && ad_tiles.is_some() { first_counted.iter().max().map(|&m| m as usize + 16) } else { cmax },
                  "tape_nodes_tile_max": cmax,
                  "tape_bytes_per_tile_max": if node_b == 16 && ad_tiles.is_some() { cmax.map(|c| c * (node_b + 8)) } else { cmax.map(|c| c * (node_b + 8) * tile) },
                  "tape_layout": if node_b == 16 && ad_tiles.is_some() { "packed" } else { "capacity" }, "lanes": lanes, "arena_bytes_per_tile": tile * arena_col,
                  "whole_launch": whole, "budget_bytes": budget, "ad_budget_bytes": self.p3_ad_budget, "ad_tiles": ad_tiles.as_ref().map(|t| t.0.iter().map(|x| json!([x.0, x.1, x.2])).collect::<Vec<_>>()),
                  "arena_hw": arena_hw, "persistent": persisted, "count_max": first_counted.iter().max(), "count_mean": if first_counted.is_empty() { None } else { Some(first_counted.iter().sum::<i64>() / first_counted.len() as i64) }});
        // 60 km tape: a banded bind's receipt (absent when off)
        if let Some((ni, i0, cols, _)) = p.stripe { out["stripe"] = json!({"ni": ni, "i0": i0, "cols": cols, "region_bytes": p.region.1}); }
        Ok(out)
    }

    // --------------------------------------------------------------------------------------- dispatch
    /// prec-floor: the boundary-layer row with a list of its OUTPUT arrays held in both sweeps (their tangent zero after the
    /// row; their cotangent dropped before the row's reverse), a transpose pair: M' = P M and M'^T = M^T P, P the projection
    /// that zeroes those outputs. The list is DATA of the image set (`gen-info.json` beside images.p3_tl, key
    /// "hold_outputs"); AD_P3_HOLD_OUT=a,b,.. replaces it and AD_P3_HOLD_OUT=none empties it (the everything-live proof mode).
    /// THE BREAKAGE IT PREVENTS: with the closure-live images, on the 12 km cut 96 to 98.5 per cent of the row's theta
    /// sensitivity (sum of squares, one real launch in double) enters through the cotangent of the number, aerosol and
    /// cloud-species tendencies, as opposite-signed neighbours up to 5.8e9 that single precision cannot carry through the
    /// dynamics; the window check then scatters at 6 to 15 roundings (2 with the seven outputs held), while the gradient
    /// against the real model reads the same to three digits with and without them (box D pf-fd-*, pf-fdh-*, pf-h2-*).
    unsafe fn p3_hold_out(&self, cuda: &Cuda, p: &Prepared) -> Result<(), String> {
        let Some(names) = P3_HOLD.get().filter(|v| !v.is_empty()) else { return Ok(()) };
        let row = &self.rows["rrfs_mynnedmf_run_l16"];
        let mut held = Vec::new();
        for name in names {
            let a = row.arg(name)?;
            let (al, o) = Self::dev_arg(row, p, name)?;
            let n = row.extent_bytes(a, &p.env)? / 4 * W;
            dev::zero(cuda, p.segs.shadow(al, o)?, n)?;
            held.push(json!([name, n / W]));
        }
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| eprintln!("{}", json!({"event":"p3_hold_out","held":held,"fields":"[output array, words]",
            "note":"said once; applied at every boundary-layer launch of both sweeps"})));
        Ok(())
    }
    /// Tangent in place on the shadows.
    pub unsafe fn tangent(&mut self, cuda: &Cuda, arena: &mut Arena, p: &mut Prepared, identity: bool) -> Result<Value, String> {
        match p.kind {
            Kind::P3 => { let r = self.p3(cuda, p, 0, identity)?; self.p3_hold_out(cuda, p)?; Ok(r) },
            Kind::P4 => { self.p4(cuda, p, false)?; Ok(Value::Null) },
            Kind::TempoFrom(_) => self.tf_run(cuda, arena, p, false, identity),
            Kind::P6(_) => self.p6(cuda, p, false, identity),
            Kind::RucLand(_) => self.ruc_land(cuda, p, false),
            Kind::P7Taumol => self.p7(cuda, p, false, identity),
            Kind::P7RswTaumol => self.p7sw(cuda, p, false, identity),
            _ => { self.rule(cuda, arena, p, false)?; Ok(Value::Null) },
        }
    }
    /// Adjoint in place on the shadows.
    pub unsafe fn reverse(&mut self, cuda: &Cuda, arena: &mut Arena, p: &mut Prepared) -> Result<Value, String> {
        match p.kind {
            Kind::P3 => { self.p3_hold_out(cuda, p)?; self.p3(cuda, p, 1, false) },
            Kind::P4 => { self.p4(cuda, p, true)?; Ok(Value::Null) },
            Kind::TempoFrom(_) => self.tf_run(cuda, arena, p, true, false),
            Kind::P6(_) => self.p6(cuda, p, true, false),
            Kind::RucLand(_) => self.ruc_land(cuda, p, true),
            Kind::P7Taumol => self.p7(cuda, p, true, false),
            Kind::P7RswTaumol => self.p7sw(cuda, p, true, false),
            _ => { self.rule(cuda, arena, p, true)?; Ok(Value::Null) },
        }
    }

    /// U6's per-launch dot on a native row (dot.rs's algorithm and fields): x, y random over the shadow region;
    /// <y, Mx - x> against <M^T y - y, x>; on checked launches the zero direction and a second reverse run.
    /// Called after the forward launch; reads the forward's post state only for the P3 identity check.
    pub unsafe fn launch_dot(&mut self, cuda: &Cuda, sup: &Support, arena: &mut Arena, p: &mut Prepared, live: &HashMap<u64, Allocation>,
                             seed: u64, checks: bool) -> Result<Value, String> {
        let tol = self.tol;
        if let Kind::Inactive(why) = &p.kind { return Ok(json!({"status":"INACTIVE","why":why,"unit":p.unit})); }
        let _ = live;
        let (rg, rb) = p.region;
        let (x0, y) = (arena.take(rb)?, arena.take(rb)?);
        let r = if checks { arena.take(rb)? } else { 0 };
        let t0 = std::time::Instant::now();
        sup.reset_words(cuda)?;
        let mut info = json!({});
        if checks {
            dev::zero(cuda, rg, rb)?;
            self.tangent(cuda, arena, p, false)?;
            sup.words_into(cuda, 0, rg, 0, rb)?;
            dev::zero(cuda, rg, rb)?;
            self.reverse(cuda, arena, p)?;
            sup.words_into(cuda, 1, rg, 0, rb)?;
        }
        sup.fill(cuda, x0, rb, W, crate::dot::mix(seed ^ 1))?;
        // The RUC land entry takes float32 tangents: its random directions are float32 words (widened), so the narrowing
        // is exact and the dot grades the entry, not the rounding of the direction.
        let rl_round = if matches!(p.kind, Kind::RucLand(_)) { self.rl.as_ref().map(|i| i.round) } else { None };
        if let Some(f) = rl_round { launch(cuda, f, grid1(rb / W, 256), [256, 1, 1], &[K::P(x0), i64b((rb / W) as i64)])?; }
        dev::copy(cuda, rg, x0, rb)?;
        let t1 = std::time::Instant::now();
        let tli = self.tangent(cuda, arena, p, true)?;
        dev::sync(cuda)?;
        let tl_ms = t1.elapsed().as_secs_f64() * 1e3;
        if !tli.is_null() { info["tangent"] = tli; }
        // Primal identity of the derivative entry against the forward's own post state (P3 writes its outputs), read
        // here, before any reverse. THE BREAKAGE (F42): read after the reverse, the initflag launch's reverse had already
        // run p3_taped_release, which frees every p3_* block, p3_y (the readout's y and mirror) included: the launch dot
        // at step 1 with the tape-free route loaded stopped with "cuMemcpyDtoH partial failed with CUDA status 700"
        // (bl-fast-live sdot/sd3), and a freed block handed out again would have been counted as the readout.
        if let Some((yb, mirror, bytes)) = p.identity { sup.words_into(cuda, 3, yb, mirror, bytes)?; }
        sup.fill(cuda, y, rb, W, crate::dot::mix(seed ^ 2))?;
        if let Some(f) = rl_round { launch(cuda, f, grid1(rb / W, 256), [256, 1, 1], &[K::P(y), i64b((rb / W) as i64)])?; }
        sup.dotdiff_into(cuda, 0, y, rg, x0, rb, W)?;
        sup.sqdiff_into(cuda, 1, rg, x0, rb, W)?;
        sup.sqdiff_into(cuda, 2, y, 0, rb, W)?;
        sup.sqdiff_into(cuda, 6, rg, 0, rb, W)?; // ||M x||^2 (the in-place tangent of the shadow region)
        dev::copy(cuda, rg, y, rb)?;
        let t2 = std::time::Instant::now();
        let adi = self.reverse(cuda, arena, p)?;
        dev::sync(cuda)?;
        let ad_ms = t2.elapsed().as_secs_f64() * 1e3;
        if !adi.is_null() { info["adjoint"] = adi; }
        sup.dotdiff_into(cuda, 3, x0, rg, y, rb, W)?;
        sup.sqdiff_into(cuda, 4, rg, y, rb, W)?;
        sup.sqdiff_into(cuda, 5, x0, 0, rb, W)?;
        sup.sqdiff_into(cuda, 7, rg, 0, rb, W)?; // ||M^T y||^2
        if checks {
            dev::copy(cuda, r, rg, rb)?;
            dev::copy(cuda, rg, y, rb)?;
            self.reverse(cuda, arena, p)?;
            sup.words_into(cuda, 2, rg, r, rb)?;
        }
        let (sums, wv) = sup.collect(cuda, 8, 4)?;
        let (lhs, dtl2, ny2, rhs, dad2, nx2) = (sums[0], sums[1], sums[2], sums[3], sums[4], sums[5]);
        let (mx2, mty2) = (sums[6], sums[7]);
        let (tl_zero, ad_zero, differing) = if checks { (wv[0][1], wv[1][1], wv[2][0]) } else { (0, 0, 0) };
        let ident = p.identity.map(|_| wv[3][0]);
        let abs = (lhs - rhs).abs();
        let scale = f64::max(lhs.abs(), rhs.abs());
        let rel = if scale == 0.0 { if abs == 0.0 { 0.0 } else { f64::INFINITY } } else { abs / scale };
        let den = (ny2 * dtl2).sqrt() + (nx2 * dad2).sqrt();
        let ns = if den == 0.0 { if abs == 0.0 { 0.0 } else { f64::INFINITY } } else { abs / den };
        let status = if !(lhs.is_finite() && rhs.is_finite()) { "NONFINITE" }
            else if tl_zero != 0 || ad_zero != 0 { "ZERO-FAIL" }
            else if differing != 0 { "NONDETERMINISTIC" }
            else if ident.is_some_and(|d| d != 0) { "IDENTITY-FAIL" }
            else if rel <= tol { "PASS" } else if ns <= tol { "NS-ONLY" } else { "FAIL" };
        let mut out = json!({"status":status,"lhs":lhs,"rhs":rhs,"abs":abs,"rel":rel,"norm_scaled":ns,"tol":tol,"width":W,
            "tl_change_norm":dtl2.sqrt(),"ad_change_norm":dad2.sqrt(),"checks":checks,
            "x_norm":nx2.sqrt(),"y_norm":ny2.sqrt(),"tl_out_norm":mx2.sqrt(),"ad_out_norm":mty2.sqrt(),
            "tl_gain":if nx2 > 0.0 { (mx2 / nx2).sqrt() } else { 0.0 },"ad_gain":if ny2 > 0.0 { (mty2 / ny2).sqrt() } else { 0.0 },
            "zero_tl_nonzero_words":tl_zero,"zero_ad_nonzero_words":ad_zero,"reverse_differing_words":differing,
            "shadow_bytes":rb,"segments":p.segs.v.iter().filter(|s| s.active).count(),
            "tl_ms":tl_ms,"ad_ms":ad_ms,"dot_ms":t0.elapsed().as_secs_f64()*1e3,"unit":p.unit,"row":"ad.scheme.rows/v1"});
        if let Some(d) = ident { out["identity_differing_words"] = json!(d); }
        if !p.notes.is_empty() { out["notes"] = json!(p.notes); }
        if info != json!({}) { out["entry"] = info; }
        Ok(out)
    }
}

// ------------------------------------------------------------------------------------------- tempo_from (held lane)
// Held-kernel lane ad/held-tempo_from: TEMPO's MPAS -> TEMPO hand-off (tempo_v303.py:4429) as the row kind
// "tempo_from". All five arguments are pointer tables (m: MODEL_ARGUMENTS, mesh: zz zgrid, d: DIAG_ARGUMENTS,
// t: TEND_ARGUMENTS, f: FIELD_ARGUMENTS); every slot the forward reads or writes is bound as its own view, the
// tables are mirrored onto the views' float64 shadows (and, for the inputs, onto their pre-launch snapshots), and
// the entries of tree/ad/phys/held/tempo_from/tempo_from_ad.cu run in place on the shadows.
const TF_TABLES: [&str; 5] = ["m", "mesh", "d", "t", "f"];
const TF_LEN: [usize; 5] = [16, 2, 10, 12, 20];
const TF_NAMES: [&[&str]; 5] = [
    &["qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc", "nwfa", "nifa", "ng", "volg", "theta_m", "rho_zz", "w"],
    &["zz", "zgrid"],
    &["exner", "pressure_base", "pressure_p", "exner_base", "rtheta_base", "rtheta_p", "dtheta_dt_mp", "rt_diabatic_tend", "surface_pressure", "tend_sfc_pressure"],
    &["rthmpten", "rqvmpten", "rqcmpten", "rqrmpten", "rqimpten", "rqsmpten", "rqgmpten", "rnimpten", "rnrmpten", "rncmpten", "rnifampten", "rnwfampten"],
    &["th", "p", "w", "dz", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc", "nwfa", "nifa", "ng", "qb", "pii", "rho", "z"],
];

#[derive(Clone, Debug)]
struct TfView { name: String, table: usize, slot: usize, alloc: u64, off: usize, bytes: usize, output: bool }
#[derive(Clone, Debug)]
pub struct Tf { nz: i32, nc: i32, aero: i32, hail: i32, rv: f32, rd: f32, views: Vec<TfView>,
                /// device tables: shadows of m, mesh, d, t, f; pre-launch primal of m, mesh, d
                tabs: [u64; 8] }
struct TfImage { _m: Module, primal: *mut c_void, tl: *mut c_void, ad: *mut c_void, ad_zgrid: *mut c_void }

/// Resolve tempo_from's five pointer tables to plan views (before the forward launch).
unsafe fn tf_bind(cuda: &Cuda, live: &HashMap<u64, Allocation>, function: &str, args: &[Argument]) -> Result<Tf, String> {
    if args.len() != 11 { return Err(format!("{function}: {} arguments, the forward has 11", args.len())); }
    let w4 = |i: usize| -> Result<[u8; 4], String> {
        match args.get(i).and_then(scalar_bytes) { Some(b) if b.len() >= 4 => Ok([b[0], b[1], b[2], b[3]]), _ => Err(format!("{function}: argument {i} is not a 4-byte scalar")) }
    };
    let (nz, nc, aero, hail) = (i32::from_le_bytes(w4(5)?), i32::from_le_bytes(w4(6)?), i32::from_le_bytes(w4(7)?), i32::from_le_bytes(w4(8)?));
    let (rv, rd) = (f32::from_le_bytes(w4(9)?), f32::from_le_bytes(w4(10)?));
    if nz <= 0 || nc <= 0 { return Err(format!("{function}: nz {nz}, nc {nc}")); }
    let n = nz as usize * nc as usize;
    let mut sorted: Vec<&Allocation> = live.values().collect();
    sorted.sort_by_key(|a| a.address);
    let find = |p: u64| -> Option<(u64, usize, usize)> {
        let i = sorted.partition_point(|a| a.address <= p);
        if i == 0 { return None; }
        let a = sorted[i - 1];
        (p < a.address + a.bytes as u64).then(|| (a.id, (p - a.address) as usize, a.bytes))
    };
    let mut tables: Vec<Vec<u64>> = Vec::new();
    for ti in 0..5 {
        let (al, off) = target(&args[ti]).ok_or_else(|| format!("{function}: pointer table {} is not a device argument", TF_TABLES[ti]))?;
        let a = live.get(&al).ok_or_else(|| format!("{function}: table {} in unknown allocation {al}", TF_TABLES[ti]))?;
        if off + 8 * TF_LEN[ti] > a.bytes { return Err(format!("{function}: table {} ({} entries) exceeds allocation {al}", TF_TABLES[ti], TF_LEN[ti])); }
        let w = dev::download(cuda, a.address + off as u64, 8 * TF_LEN[ti])?;
        tables.push(w.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect());
    }
    let (ae, he) = (aero != 0, hail != 0);
    let used_m = |j: usize| j < 8 || j >= 13 || if j < 11 { ae } else { he };
    let used_f = |j: usize| match j { 12..=14 => ae, 15..=16 => he, _ => true };
    let used_t = |j: usize| j < 9 || ae;
    let mut views: Vec<TfView> = Vec::new();
    let mut add = |ti: usize, slot: usize, bytes: usize, output: bool| -> Result<(), String> {
        let p = tables[ti][slot];
        let name = format!("{}[{slot}] ({})", TF_TABLES[ti], TF_NAMES[ti][slot]);
        if p == 0 { return Err(format!("{function}: {name} is null where the forward uses it")); }
        let (id, off, ab) = find(p).ok_or_else(|| format!("{function}: {name} pointer {p:#x} is not in a live plan allocation"))?;
        if off % 4 != 0 { return Err(format!("{function}: {name} at byte {off} of allocation {id} is not word aligned")); }
        if off + bytes > ab { return Err(format!("{function}: {name} needs {bytes} bytes at offset {off}; allocation {id} has {ab}")); }
        views.push(TfView { name, table: ti, slot, alloc: id, off, bytes, output });
        Ok(())
    };
    for j in 0..16 { if used_m(j) { add(0, j, 4 * n, false)?; } }
    add(1, 0, 4 * n, false)?;
    add(1, 1, 4 * (n + nc as usize), false)?;
    for j in 0..3 { add(2, j, 4 * n, false)?; }
    for j in 0..12 { if used_t(j) { add(3, j, 4 * n, true)?; } }
    for j in 0..20 { if used_f(j) { add(4, j, 4 * n, true)?; } }
    // Refused by name: an output over any other bound word (the in-place entries would read a written word), and
    // a shifted overlap of two inputs the element adjoint both writes (a race between threads). Exact duplicate
    // inputs (one thread adds both) and zgrid against an element input (separate launches) are safe.
    for i in 0..views.len() {
        for k in i + 1..views.len() {
            let (a, b) = (&views[i], &views[k]);
            if a.alloc != b.alloc || a.off + a.bytes <= b.off || b.off + b.bytes <= a.off { continue; }
            let zg = |v: &TfView| v.table == 1 && v.slot == 1;
            let fine = !a.output && !b.output && ((a.off == b.off && a.bytes == b.bytes) || zg(a) != zg(b));
            if !fine { return Err(format!("{function}: {} and {} overlap in allocation {} (bytes {}..{} and {}..{})",
                                          a.name, b.name, a.alloc, a.off, a.off + a.bytes, b.off, b.off + b.bytes)); }
        }
    }
    Ok(Tf { nz, nc, aero, hail, rv, rd, views, tabs: [0; 8] })
}

/// After the segments are placed: the mirrored tables (shadow and pre-launch primal addresses per slot).
unsafe fn tf_tables(cuda: &Cuda, arena: &mut Arena, segs: &Segs, tf: &mut Tf) -> Result<(), String> {
    let mut sh: Vec<Vec<u64>> = TF_LEN.iter().map(|&l| vec![0u64; l]).collect();
    let mut pr: Vec<Vec<u64>> = TF_LEN[..3].iter().map(|&l| vec![0u64; l]).collect();
    for v in &tf.views {
        sh[v.table][v.slot] = segs.shadow(v.alloc, v.off)?;
        if v.table < 3 { pr[v.table][v.slot] = segs.pre(v.alloc, v.off)?; }
    }
    for (i, t) in sh.iter().chain(pr.iter()).enumerate() {
        let a = arena.take(8 * t.len())?;
        dev::upload(cuda, a, std::slice::from_raw_parts(t.as_ptr().cast::<u8>(), 8 * t.len()))?;
        tf.tabs[i] = a;
    }
    Ok(())
}

fn tf_f64(b: &[u8]) -> Vec<f64> { b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect() }
fn tf_f32(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }
struct TfRng(u64);
impl TfRng {
    fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0; z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9); z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        (((z ^ (z >> 31)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 { let (a, b) = (self.u(), self.u()); (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos() }
}

impl Native {
    fn tf_scalars(tf: &Tf, v: &mut Vec<K>) {
        v.extend([i32b(tf.nz), i32b(tf.nc), i32b(tf.aero), i32b(tf.hail), K::B(tf.rv.to_le_bytes().to_vec()), K::B(tf.rd.to_le_bytes().to_vec())]);
    }
    unsafe fn tf_tl_launch(img: &TfImage, cuda: &Cuda, tf: &Tf) -> Result<(), String> {
        let n = tf.nz as usize * tf.nc as usize;
        let t = tf.tabs;
        let mut a = vec![K::P(t[0]), K::P(t[1]), K::P(t[2]), K::P(t[3]), K::P(t[4]), K::P(t[5]), K::P(t[6])];
        Self::tf_scalars(tf, &mut a);
        launch(cuda, img.tl, grid1(n, 256), [256, 1, 1], &a)
    }
    /// Sum of squares over the distinct views of one side (inputs or outputs), read from the shadows.
    unsafe fn tf_norm2(cuda: &Cuda, p: &Prepared, tf: &Tf, output: bool) -> Result<f64, String> {
        let mut seen = HashSet::new();
        let mut s = 0f64;
        for v in tf.views.iter().filter(|v| v.output == output) {
            if !seen.insert((v.alloc, v.off, v.bytes)) { continue; }
            let x = tf_f64(&dev::download(cuda, p.segs.shadow(v.alloc, v.off)?, v.bytes / 4 * W)?);
            s += x.iter().map(|q| q * q).sum::<f64>();
        }
        Ok(s)
    }
    /// The launch dot's tangent call (identity = true): primal identity against the forward's post state, the raw
    /// gain ||TL(x)|| / ||x|| on the dot's own direction, and the state-relative gain on x_d = r |x| (r ~ N(0,1)),
    /// reported as rms(y_d / |y|) per output field. The region is left exactly as the dot set it.
    unsafe fn tf_diag(img: &TfImage, cuda: &Cuda, arena: &mut Arena, p: &mut Prepared, tf: &Tf) -> Result<Value, String> {
        let n = tf.nz as usize * tf.nc as usize;
        let (rg, rb) = p.region;
        let save = arena.take(rb)?;
        dev::copy(cuda, save, rg, rb)?;
        let x_in2 = Self::tf_norm2(cuda, p, tf, false)?;
        // primal identity: 32 slots (f 0..19, t 0..11), unwritten slots zero on both sides
        let ib = 32 * n * 4;
        let (yb, mirror) = (arena.take(ib)?, arena.take(ib)?);
        dev::zero(cuda, yb, ib)?; dev::zero(cuda, mirror, ib)?;
        let mut a = vec![K::P(tf.tabs[5]), K::P(tf.tabs[6]), K::P(tf.tabs[7]), K::P(yb)];
        Self::tf_scalars(tf, &mut a);
        launch(cuda, img.primal, grid1(n, 256), [256, 1, 1], &a)?;
        let slot_of = |v: &TfView| if v.table == 4 { v.slot } else { 20 + v.slot };
        for v in tf.views.iter().filter(|v| v.output) {
            let s = p.segs.find(v.alloc, v.off)?;
            dev::copy(cuda, mirror + (slot_of(v) * n * 4) as u64, s.live + (v.off - s.lo) as u64, v.bytes)?;
        }
        dev::sync(cuda)?;
        p.identity = Some((yb, mirror, ib));
        // clamp census (pre-launch qv)
        let qv = tf.views.iter().find(|v| v.table == 0 && v.slot == 0).ok_or("qv view")?;
        let qvp = tf_f32(&dev::download(cuda, p.segs.pre(qv.alloc, qv.off)?, qv.bytes)?);
        let (neg, zero) = (qvp.iter().filter(|&&q| 0.0f32 > q).count(), qvp.iter().filter(|&&q| q == 0.0).count());
        // relative gain: x_d = r |x| on every input view, outputs zero, one tangent
        dev::zero(cuda, rg, rb)?;
        let mut rng = TfRng(0x7e3f_0f20_d1b5_4a32);
        let mut r2 = 0f64; let mut nr = 0usize;
        for v in tf.views.iter().filter(|v| !v.output) {
            let x = tf_f32(&dev::download(cuda, p.segs.pre(v.alloc, v.off)?, v.bytes)?);
            let d: Vec<f64> = x.iter().map(|&q| { let r = rng.normal(); r2 += r * r; nr += 1; r * (q as f64).abs() }).collect();
            dev::upload(cuda, p.segs.shadow(v.alloc, v.off)?, std::slice::from_raw_parts(d.as_ptr().cast::<u8>(), 8 * d.len()))?;
        }
        Self::tf_tl_launch(img, cuda, tf)?;
        dev::sync(cuda)?;
        let post = tf_f32(&dev::download(cuda, mirror, ib)?);
        let mut per = serde_json::Map::new();
        let (mut e2, mut ne) = (0f64, 0usize);
        for v in tf.views.iter().filter(|v| v.output) {
            let yd = tf_f64(&dev::download(cuda, p.segs.shadow(v.alloc, v.off)?, v.bytes / 4 * W)?);
            let y = &post[slot_of(v) * n..slot_of(v) * n + n];
            let (mut s, mut c, mut mx, mut at, mut zeros) = (0f64, 0usize, 0f64, 0usize, 0usize);
            for i in 0..n {
                if y[i] == 0.0 { zeros += 1; continue; }
                let e = yd[i] / (y[i] as f64).abs();
                s += e * e; c += 1;
                if e.abs() > mx { mx = e.abs(); at = i; }
            }
            e2 += s; ne += c;
            per.insert(v.name.clone(), json!({"rms_rel": if c > 0 { (s / c as f64).sqrt() } else { 0.0 }, "max_rel": mx,
                "max_at_level_cell": [at / tf.nc as usize, at % tf.nc as usize], "zero_primal_words": zeros}));
        }
        dev::copy(cuda, rg, save, rb)?;
        dev::sync(cuda)?;
        let rel_overall = if ne > 0 && nr > 0 { (e2 / ne as f64).sqrt() / (r2 / nr as f64).sqrt() } else { 0.0 };
        Ok(json!({"x_in_norm": x_in2.sqrt(), "identity_compared_words": ib / 4,
                  "clamp_qv": {"negative_words": neg, "zero_words": zero, "words": qvp.len(), "rule": "frozen executed arm of tmax(0,qv): H = !(0 > qv)"},
                  "relative_gain": {"direction": "x_d = r |x|, r ~ N(0,1), every input view", "rms_ratio": rel_overall, "per_output": per}}))
    }
    /// tempo_from tangent (in place) or adjoint (in place, outputs consumed).
    unsafe fn tf_run(&mut self, cuda: &Cuda, arena: &mut Arena, p: &mut Prepared, adjoint: bool, identity: bool) -> Result<Value, String> {
        let img = self.tf.as_ref().ok_or("tempo_from row without its image (schemes.json images.tempo_from)")?;
        let tf = match &p.kind { Kind::TempoFrom(t) => (**t).clone(), _ => return Err("not a tempo_from binding".into()) };
        let n = tf.nz as usize * tf.nc as usize;
        let t = tf.tabs;
        if adjoint {
            launch(cuda, img.ad_zgrid, grid1(n + tf.nc as usize, 256), [256, 1, 1], &[K::P(t[1]), K::P(t[4]), i32b(tf.nz), i32b(tf.nc)])?;
            let mut a = vec![K::P(t[0]), K::P(t[1]), K::P(t[2]), K::P(t[3]), K::P(t[4]), K::P(t[5]), K::P(t[6])];
            Self::tf_scalars(&tf, &mut a);
            launch(cuda, img.ad, grid1(n, 256), [256, 1, 1], &a)?;
            return Ok(Value::Null);
        }
        if !identity { Self::tf_tl_launch(img, cuda, &tf)?; return Ok(Value::Null); }
        let mut info = Self::tf_diag(img, cuda, arena, p, &tf)?;
        Self::tf_tl_launch(img, cuda, &tf)?;
        dev::sync(cuda)?;
        let y2 = Self::tf_norm2(cuda, p, &tf, true)?;
        let x = info["x_in_norm"].as_f64().unwrap_or(0.0);
        info["y_out_norm"] = json!(y2.sqrt());
        info["gain_raw"] = json!(if x > 0.0 { y2.sqrt() / x } else { 0.0 });
        Ok(info)
    }
}

// ------------------------------------------------------------------------------------------- P3 column shards
// multicard-merge (2026-11-09): MYNN-EDMF TL/AD split by columns across CUDA contexts, one per card
// (AD_MYNN_SHARD_DEVICES=0,1,...; ordinal 0 is the dycore card and keeps the first range). With
// AD_SHARD_SAME_CARD_CONTEXTS=1 an ordinal may repeat (separate contexts on one card: correctness proofs only).
// Columns are independent (G5): every context runs the same persistent entries over its own range
// [first, first + count) of the one launch (MwLaunch.col_first/col_count), reading the primary's primal snapshots and
// derivative shadows through the unified address space (a peer mapping between cards) and writing only its own
// columns' outputs (y, tangent / cotangent, kpbl, error flags) in place. Arena, store, tape and failure buffers stay
// on the helper. A helper's adjoint uses the per-block region tape (layout 2); layout only places the tape, so the
// words do not change. Breakage it removes: MYNN AD (6.2 s of 26.3 s/step at 120 km) ran on one card while the
// others waited at the dependency boundary.
struct P3Job { slots: Vec<[u64; 6]>, ptrs: [u64; 3], sargs: Vec<Vec<u8>>, first: usize, count: usize, mode: i32,
               arena_col: usize, col_elems: usize, sz_store: usize, cap_env: usize, s_cap: usize,
               /// mynn: the launch's initflag (its tapes are larger), and whether the helper may skip the count pass.
               init: bool, count_free: bool,
               /// mynn: the launch's array geometry, when the helper may work on local stripes (see P3Local).
               local: Option<P3Local> }
/// mynn (helper-local stripes). Every float array of the launch is rows x ni elements with the column index
/// fastest (rows = levels, or 1 for a surface array); a helper's columns are the element range
/// [i_first + first, i_first + first + count) of every row. `main` is the dycore context.
#[derive(Clone)]
struct P3Local { ni: usize, i_first: usize, rows: Vec<usize>, main: usize }
struct P3Done { fails: [u64; 4], receipt: Value }
struct P3Worker { tx: std::sync::mpsc::Sender<P3Job>, rx: std::sync::Mutex<std::sync::mpsc::Receiver<Result<P3Done, String>>>,
                  join: Option<std::thread::JoinHandle<()>> }
pub(crate) struct P3Shards { ordinals: Vec<i32>, workers: Vec<P3Worker> }

struct P3Helper { tl: Module, ad: Module, count: Module, f: [*mut c_void; 3], node_b: usize, lanes: usize, sbytes: usize,
                  scratch: HashMap<&'static str, Grow>,
                  /// mynn: by initflag, the largest per-column tape count and the largest group tape this helper has counted.
                  cap_seen: HashMap<bool, (i64, usize)> }

impl P3Helper {
    unsafe fn load(cuda: &Cuda, paths: &[PathBuf; 3], want: [usize; 5]) -> Result<Self, String> {
        let (tl, ad, count) = (Module::load(cuda, &paths[0])?, Module::load(cuda, &paths[1])?, Module::load(cuda, &paths[2])?);
        let sizes = tl.func(cuda, "mw_sizes")?;
        let mut o = 0u64;
        checked("cuMemAlloc", (cuda.allocate)(&mut o, 64))?;
        dev::zero(cuda, o, 64)?;
        launch(cuda, sizes, [1, 1, 1], [1, 1, 1], &[K::P(o)])?;
        dev::sync(cuda)?;
        let s: Vec<u64> = dev::download(cuda, o, 48)?.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        let lanes = if s[4] == 0 { 32 } else { s[4] as usize };
        let got = [s[0] as usize, s[1] as usize, s[2] as usize, lanes, s[3] as usize];
        if got != want { return Err(format!("DEFECT MYNN shard image sizes {got:?} differ from the primary's {want:?}")); }
        let f = [tl.func(cuda, "mynn_whole_tl_p")?, ad.func(cuda, "mynn_whole_ad_p")?, count.func(cuda, "mynn_whole_count_p")?];
        Ok(Self { tl, ad, count, f, node_b: got[2], lanes, sbytes: got[4], scratch: HashMap::new(), cap_seen: HashMap::new() })
    }

    unsafe fn globals(&self, cuda: &Cuda, arena_col: usize, fail: u64, used_hw: u64, dom: u64) -> Result<(), String> {
        for m in [&self.tl, &self.ad, &self.count] {
            m.set_global(cuda, "mw_arena_limit", &(arena_col as u64).to_le_bytes())?;
            m.set_global(cuda, "mw_fail", &fail.to_le_bytes())?;
            for z in ["mw_br_bits", "mw_br_n", "mw_br_hash", "mw_br_words"] { m.set_global(cuda, z, &0u64.to_le_bytes())?; }
            m.set_global(cuda, "mw_arena_hw", &used_hw.to_le_bytes())?;
            m.set_global(cuda, "mw_dom", &dom.to_le_bytes())?;
        }
        let _ = self.ad.set_global(cuda, "mw_ring", &0u64.to_le_bytes());
        Ok(())
    }

    unsafe fn need(&mut self, cuda: &Cuda, k: &'static str, b: usize) -> Result<u64, String> {
        self.scratch.entry(k).or_default().need(cuda, b.max(64))
    }

    /// mynn: one pitched copy of `rows` runs of `width` bytes between two contexts (this helper's default stream,
    /// so it is ordered with the helper's launches).
    unsafe fn stripe(cuda: &Cuda, src: u64, sc: *mut c_void, dst: u64, dc: *mut c_void, pitch: usize, width: usize, rows: usize) -> Result<(), String> {
        if width == 0 || rows == 0 { return Ok(()); }
        let copy = crate::cuda::PeerCopy3D {
            src: crate::cuda::PeerCopyEndpoint { device: src, context: sc, pitch, memory_type: 2, ..Default::default() },
            dst: crate::cuda::PeerCopyEndpoint { device: dst, context: dc, pitch, memory_type: 2, ..Default::default() },
            width_bytes: width, height: rows, depth: 1,
        };
        checked("MYNN shard stripe copy", (cuda.copy_peer_3d_async)(&copy, ptr::null_mut()))
    }

    unsafe fn pass(&self, cuda: &Cuda, f: *mut c_void, m: &Module, slots: usize, w: &[u8], lbuf: u64, sched: u64, fail: u64,
                   sargs: &[K], n: usize, fails: &mut [u64; 4]) -> Result<(), String> {
        if w.len() != self.sbytes { return Err(format!("P3 shard launch struct {} bytes, image expects {}", w.len(), self.sbytes)); }
        dev::upload(cuda, lbuf, w)?;
        dev::zero(cuda, sched, 64)?;
        dev::zero(cuda, fail, n.div_ceil(32).max(1) * 32 * 16)?;
        m.set_global(cuda, "mw_sched", &sched.to_le_bytes())?;
        let mut a = vec![K::P(lbuf)];
        a.extend(sargs.iter().cloned());
        launch(cuda, f, [slots as u32, 1, 1], [self.lanes as u32, 1, 1], &a)?;
        dev::sync(cuda)?;
        let fl: Vec<i32> = dev::download(cuda, fail, n * 16)?.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
        for s in 0..4 { fails[s] += fl.iter().skip(s).step_by(4).filter(|&&v| v != 0).count() as u64; }
        Ok(())
    }

    unsafe fn run(&mut self, cuda: &Cuda, j: &P3Job) -> Result<P3Done, String> {
        let t_run = std::time::Instant::now();
        // mynn (helper scratch is freed when outgrown). THE BREAKAGE: a grown scratch block retires the old one and
        // never frees it (on the dycore card a queued launch may still read it). A helper's tape blocks change size with
        // every launch kind (a test machine <local path> m60d-Acf: cards 1 and 2 stood at 90-92 GB with 22 MB of tape in use, and
        // the next launch was left one adjoint slot, 93.8 s for one launch). A helper waits for its own launch before it
        // answers (the worker loop's sync), so nothing reads a retired block when the next job starts.
        let mut freed = 0usize;
        for g in self.scratch.values_mut() { for a in g.retired.drain(..) { if (cuda.free)(a) == 0 { freed += 1; } } }
        let (lanes, node_b, sbytes) = (self.lanes, self.node_b, self.sbytes);
        let n = j.count;
        let groups = n.div_ceil(lanes);
        // mynn (helper-local stripes): mirrors of the launch's arrays with the dycore card's own layout (so the
        // kernel's element offsets hold), of which only this helper's stripes are copied in, used and copied out.
        let mut slots_run: Vec<[u64; 6]> = j.slots.clone();
        let mut outs: Vec<(u64, u64, usize)> = Vec::new();
        let mut local_rec = Value::Null;
        let mut here = ptr::null_mut();
        if let Some(g) = j.local.as_ref() {
            checked("MYNN shard context", (cuda.ctx_get_current)(&mut here))?;
            let main = g.main as *mut c_void;
            let tot: usize = g.rows.iter().sum::<usize>() * g.ni;
            let lx = self.scratch.entry("lx").or_default().need_exact(cuda, (tot * 4).max(64))?;
            let lin = self.scratch.entry("lin").or_default().need_exact(cuda, (tot * 8).max(64))?;
            let lout = self.scratch.entry("lout").or_default().need_exact(cuda, (tot * 8).max(64))?;
            let i0 = g.i_first + j.first;
            let (mut e, mut bytes_in) = (0usize, 0usize);
            for (k, sl) in j.slots.iter().enumerate() {
                let r = g.rows.get(k).copied().unwrap_or(0);
                if sl[0] == 0 || r == 0 { continue; }
                if sl[1] != 0 { return Err("DEFECT MYNN shard local stripes with a y buffer bound".into()); }
                let (din, dout) = if j.mode == 0 { (sl[2], sl[3]) } else { (sl[4], sl[5]) };
                Self::stripe(cuda, sl[0] + (i0 * 4) as u64, main, lx + ((e + i0) * 4) as u64, here, g.ni * 4, n * 4, r)?;
                bytes_in += n * 4 * r;
                if din != 0 { Self::stripe(cuda, din + (i0 * 8) as u64, main, lin + ((e + i0) * 8) as u64, here, g.ni * 8, n * 8, r)?; bytes_in += n * 8 * r; }
                // the outputs' stripes start from the dycore card's values, so an element the kernel leaves keeps them
                if dout != 0 { Self::stripe(cuda, dout + (i0 * 8) as u64, main, lout + ((e + i0) * 8) as u64, here, g.ni * 8, n * 8, r)?; bytes_in += n * 8 * r;
                               outs.push((lout + ((e + i0) * 8) as u64, dout + (i0 * 8) as u64, r)); }
                let (a, b, c) = (lx + (e * 4) as u64, if din != 0 { lin + (e * 8) as u64 } else { 0 }, if dout != 0 { lout + (e * 8) as u64 } else { 0 });
                slots_run[k] = if j.mode == 0 { [a, 0, b, c, 0, 0] } else { [a, 0, 0, 0, b, c] };
                e += r * g.ni;
            }
            local_rec = json!({"mirror_bytes": tot * 20, "bytes_in": bytes_in, "bytes_out": outs.iter().map(|o| o.2 * n * 8).sum::<usize>(), "arrays": outs.len()});
        }
        let resident = |f: *mut c_void| -> usize {
            let (mut nb, mut dv, mut sms) = (0i32, 0i32, 0i32);
            if (cuda.occupancy)(&mut nb, f, lanes as i32, 0) != 0 || (cuda.ctx_device)(&mut dv) != 0 || (cuda.device_attribute)(&mut sms, 16, dv) != 0 { return 1024; }
            (nb.max(1) as usize) * (sms.max(1) as usize)
        };
        let col_fixed = j.arena_col + j.col_elems * j.sz_store + j.col_elems * 4 + 64;
        let mk = |mode_k: i32, ar: u64, st: u64, lf: u64, ks: u64, nodes: u64, bars: u64, capacity: u64, used: u64, fail: u64| -> Vec<u8> {
            let mut w: Vec<u8> = Vec::with_capacity(sbytes);
            let put = |w: &mut Vec<u8>, v: u64| w.extend_from_slice(&v.to_le_bytes());
            for sl in &slots_run { let sl = if mode_k == 2 { [sl[0], sl[1], 0, 0, 0, 0] } else { *sl }; for v in sl { put(&mut w, v); } }
            for v in [j.ptrs[0], j.ptrs[1], j.ptrs[2], ar, j.arena_col as u64, st, j.col_elems as u64, lf, ks, nodes, bars, capacity, used, fail] { put(&mut w, v); }
            for v in [j.first as i32, n as i32, mode_k, 2] { w.extend_from_slice(&v.to_le_bytes()); }
            w
        };
        let sargs: Vec<K> = j.sargs.iter().map(|b| K::B(b.clone())).collect();
        let mut fails = [0u64; 4];
        let f1 = if j.mode == 0 { self.f[0] } else { self.f[2] };
        // mynn: the first pass's blocks are bounded by this card's free memory (arena + store + kstore per block, with
        // the half-again growth of the scratch), now that the dycore card's tile no longer bounds them.
        let s1_room = {
            let (mut free, mut total) = (0usize, 0usize);
            checked("cuMemGetInfo", (cuda.mem_info)(&mut free, &mut total))?;
            let held: usize = ["arena", "store", "kstore"].iter().map(|k| self.scratch.get(k).map(|g| g.bytes).unwrap_or(0)).sum();
            let reserve = if total >= (64usize << 30) { 12usize << 30 } else { total / 8 };
            ((free + held).saturating_sub(reserve) / (lanes * col_fixed * 3 / 2).max(1)).max(1)
        };
        let s1 = groups.min(resident(f1)).min(j.cap_env).min(j.s_cap).min(s1_room).max(1);
        let wide = n.div_ceil(32) * 32;
        let lbuf = self.need(cuda, "lbuf", sbytes)?;
        let sched = self.need(cuda, "sched", 64)?;
        let used = self.need(cuda, "used", (n + 8) * 8)?;
        let fail = self.need(cuda, "fail", wide.max(s1 * lanes) * 16)?;
        let used_hw = self.need(cuda, "used_hw", wide.max(s1 * lanes) * 8)?;
        let dom = self.need(cuda, "dom", wide.max(s1 * lanes) * 4)?;
        let ar1 = self.need(cuda, "arena", s1 * lanes * j.arena_col)?;
        let st1 = self.need(cuda, "store", s1 * lanes * j.col_elems * j.sz_store)?;
        let ks1 = self.need(cuda, "kstore", s1 * lanes * 32)?;
        // mynn (helper count-free, opt-in AD_MYNN_SHARD_COUNT_FREE=1). THE BREAKAGE: at x1.163842 on 4 cards (a test machine
        // <local path> m60b-A) every pass a helper runs costs 0.6-1.4 s whatever it computes (its kernel reads and writes the
        // dycore card's arrays in place through the peer mapping: the helper's TANGENT pass of 9,617 columns takes 0.6-1.4 s
        // where the dycore card runs 40,962 columns in 0.15 s), and the adjoint ran two passes per launch, the first only
        // to size each column's tape: MYNN-EDMF 11.4 s/step of 19.9 s of adjoint entries. A helper has tens of GB free and
        // gives each block its own tape region, so once it has counted a launch of this kind it gives every column one
        // capacity (1.25 x the largest count it has seen, + 1,024) and runs the adjoint pass alone. The kernel is
        // unchanged: a column that needs more refuses its pushes and sets its overflow slot, which the pass reads back;
        // any overflow reruns the launch the counted way (the entry assigns its outputs, so the rerun leaves the counted
        // launch's words). Not used when the uniform regions would leave under 3/4 of the blocks the counted regions give.
        let mut done_free = false;
        let mut rec = json!({"first": j.first, "columns": n, "slots_first": s1, "retired_freed": freed});
        if let (1, true, Some(&(cmax, region_seen))) = (j.mode, j.count_free, self.cap_seen.get(&j.init)) {
            let cap = cmax + cmax / 4 + 1024;
            let region = lanes * (cap as usize + 16);
            let (mut free, mut total) = (0usize, 0usize);
            checked("cuMemGetInfo", (cuda.mem_info)(&mut free, &mut total))?;
            let held: usize = ["nodes", "bars"].iter().map(|k| self.scratch.get(k).map(|g| g.bytes).unwrap_or(0)).sum();
            let reserve = if total >= (64usize << 30) { 12usize << 30 } else { total / 8 };
            let budget = (free + held).saturating_sub(reserve);
            let top = groups.min(resident(self.f[1])).min(j.cap_env).min(j.s_cap).max(1);
            let s_free = top.min(budget / (lanes * col_fixed + region * (node_b + 8)).max(1));
            let s_counted = top.min(budget / (lanes * col_fixed + region_seen.max(1) * (node_b + 8)).max(1)).max(1);
            // a card with no room gives both figures as 1: count-free is then skipped (under 1/8 of the blocks it could run)
            if s_free >= 1 && s_free * 4 >= s_counted * 3 && s_free * 8 >= top {
                let caps: Vec<u8> = std::iter::repeat(cap.to_le_bytes()).take(n).flatten().collect();
                dev::upload(cuda, used, &caps)?;
                let nodes = self.need(cuda, "nodes", s_free * region * node_b)?;
                let bars = self.need(cuda, "bars", s_free * region * 8)?;
                let ar2 = self.need(cuda, "arena", s_free * lanes * j.arena_col)?;
                let st2 = self.need(cuda, "store", s_free * lanes * j.col_elems * j.sz_store)?;
                let ks2 = self.need(cuda, "kstore", s_free * lanes * 32)?;
                let lf2 = self.need(cuda, "leaf", s_free * lanes * j.col_elems * 4)?;
                let fail = self.need(cuda, "fail", wide.max(s_free * lanes) * 16)?;
                let used_hw = self.need(cuda, "used_hw", wide.max(s_free * lanes) * 8)?;
                let dom = self.need(cuda, "dom", wide.max(s_free * lanes) * 4)?;
                self.globals(cuda, j.arena_col, fail, used_hw, dom)?;
                let w = mk(1, ar2, st2, lf2, ks2, nodes, bars, region as u64, used, fail);
                let mut ftry = [0u64; 4];
                self.pass(cuda, self.f[1], &self.ad, s_free, &w, lbuf, sched, fail, &sargs, n, &mut ftry)?;
                if ftry[0] == 0 {
                    fails = ftry; done_free = true;
                    rec["count_free"] = json!({"cap_nodes": cap}); rec["slots_ad"] = json!(s_free); rec["region_nodes"] = json!(region);
                    rec["tape_bytes_held"] = json!(s_free * region * (node_b + 8)); rec["first_pass_ms"] = json!(0.0);
                } else {
                    rec["count_free_redo"] = json!({"columns_over": ftry[0], "cap_nodes": cap});
                }
            } else {
                rec["count_free_skipped"] = json!({"slots_uniform": s_free, "slots_counted": s_counted});
            }
        }
        if !done_free {
        let fail = self.need(cuda, "fail", wide.max(s1 * lanes) * 16)?;
        let used_hw = self.need(cuda, "used_hw", wide.max(s1 * lanes) * 8)?;
        let dom = self.need(cuda, "dom", wide.max(s1 * lanes) * 4)?;
        self.globals(cuda, j.arena_col, fail, used_hw, dom)?;
        let w = mk(if j.mode == 0 { 0 } else { 2 }, ar1, st1, 0, ks1, 0, 0, 0, used, fail);
        let m1 = if j.mode == 0 { &self.tl } else { &self.count };
        let t_first = std::time::Instant::now();
        self.pass(cuda, f1, m1, s1, &w, lbuf, sched, fail, &sargs, n, &mut fails)?;
        rec["first_pass_ms"] = json!(t_first.elapsed().as_secs_f64() * 1e3);
        if j.mode == 1 {
            let sc: Vec<u64> = dev::download(cuda, sched, 16)?.chunks_exact(8).map(|x| u64::from_le_bytes(x.try_into().unwrap())).collect();
            let region = sc[1] as usize;
            if j.count_free {
                let cmax = dev::download(cuda, used, n * 8)?.chunks_exact(8).map(|x| i64::from_le_bytes(x.try_into().unwrap())).max().unwrap_or(0).max(0);
                let e = self.cap_seen.entry(j.init).or_insert((0, 0));
                *e = (e.0.max(cmax), e.1.max(region));
            }
            let (mut free, mut total) = (0usize, 0usize);
            checked("cuMemGetInfo", (cuda.mem_info)(&mut free, &mut total))?;
            let reserve = if total >= (64usize << 30) { 12usize << 30 } else { total / 8 };
            let budget = free.saturating_sub(reserve);
            let per_slot = lanes * col_fixed + region * (node_b + 8);
            let s2 = groups.min(resident(self.f[1])).min(j.cap_env).min(j.s_cap).min(budget / per_slot.max(1)).max(1);
            let nodes = self.need(cuda, "nodes", s2 * region * node_b)?;
            let bars = self.need(cuda, "bars", s2 * region * 8)?;
            let ar2 = self.need(cuda, "arena", s2 * lanes * j.arena_col)?;
            let st2 = self.need(cuda, "store", s2 * lanes * j.col_elems * j.sz_store)?;
            let ks2 = self.need(cuda, "kstore", s2 * lanes * 32)?;
            let lf2 = self.need(cuda, "leaf", s2 * lanes * j.col_elems * 4)?;
            let fail = self.need(cuda, "fail", wide.max(s2 * lanes) * 16)?;
            let used_hw = self.need(cuda, "used_hw", wide.max(s2 * lanes) * 8)?;
            let dom = self.need(cuda, "dom", wide.max(s2 * lanes) * 4)?;
            self.globals(cuda, j.arena_col, fail, used_hw, dom)?;
            let w = mk(1, ar2, st2, lf2, ks2, nodes, bars, region as u64, used, fail);
            self.pass(cuda, self.f[1], &self.ad, s2, &w, lbuf, sched, fail, &sargs, n, &mut fails)?;
            rec["slots_ad"] = json!(s2); rec["region_nodes"] = json!(region); rec["tape_bytes_held"] = json!(s2 * region * (node_b + 8));
        }
        }
        if let Some(g) = j.local.as_ref() {
            for &(src, dst, r) in &outs { Self::stripe(cuda, src, here, dst, g.main as *mut c_void, g.ni * 8, n * 8, r)?; }
            dev::sync(cuda)?;
            rec["local"] = local_rec;
        }
        rec["fails"] = json!(fails);
        rec["ms"] = json!(t_run.elapsed().as_secs_f64() * 1e3);
        Ok(P3Done { fails, receipt: rec })
    }
}

impl P3Shards {
    unsafe fn from_env(cuda: &Cuda, paths: [PathBuf; 3], want: [usize; 5]) -> Result<Option<Self>, String> {
        let text = match std::env::var("AD_MYNN_SHARD_DEVICES") { Ok(s) => s, Err(std::env::VarError::NotPresent) => return Ok(None),
            Err(e) => return Err(format!("DEFECT AD_MYNN_SHARD_DEVICES: {e}")) };
        let ordinals = crate::column_shard::devices(&text)?;
        let mut primary = ptr::null_mut();
        checked("MYNN shard primary context", (cuda.ctx_get_current)(&mut primary))?;
        let mut main_device = 0;
        checked("MYNN shard primary device", (cuda.ctx_device)(&mut main_device))?;
        let mut first = 0;
        checked("MYNN shard first device", (cuda.device_get)(&mut first, ordinals[0]))?;
        if first != main_device { return Err("DEFECT first MYNN shard ordinal must be the dycore device".into()); }
        let mut me = Self { ordinals: ordinals.clone(), workers: Vec::new() };
        for &ordinal in &ordinals[1..] {
            let mut device = 0;
            checked("MYNN shard device", (cuda.device_get)(&mut device, ordinal))?;
            if device != main_device {
                let mut can = 0;
                checked("MYNN shard peer capability", (cuda.device_can_access_peer)(&mut can, device, main_device))?;
                if can == 0 { return Err(format!("DEFECT MYNN sharding reads the dycore card's fields directly: devices {device} and {main_device} have no peer access")); }
            }
            let (tx, rx) = std::sync::mpsc::channel::<P3Job>();
            let (answer, result) = std::sync::mpsc::channel();
            let (ready, startup) = std::sync::mpsc::sync_channel(1);
            let (paths, main) = (paths.clone(), primary as usize);
            let join = std::thread::spawn(move || {
                let mut context = ptr::null_mut();
                let cuda = match Cuda::load() { Ok(c) => c, Err(e) => { let _ = ready.send(Err(e)); return; } };
                let init = (|| {
                    checked("MYNN shard cuInit", (cuda.init)(0))?;
                    let mut dv = 0;
                    checked("MYNN shard cuDeviceGet", (cuda.device_get)(&mut dv, ordinal))?;
                    checked("MYNN shard cuCtxCreate", (cuda.context_create)(&mut context, 4, dv))?;
                    if dv != main_device {
                        let status = (cuda.context_enable_peer)(main as *mut c_void, 0);
                        if status != 704 { checked("MYNN shard peer enable", status)?; }
                    }
                    P3Helper::load(&cuda, &paths, want)
                })();
                match init {
                    Err(e) => { let _ = ready.send(Err(format!("DEFECT MYNN shard device {ordinal}: {e}"))); },
                    Ok(mut h) => {
                        if ready.send(Ok(())).is_ok() {
                            while let Ok(job) = rx.recv() {
                                let done = h.run(&cuda, &job);
                                // Synchronize even on error: no helper write may land after the primary moves on.
                                let sync = dev::sync(&cuda);
                                let done = done.and_then(|d| sync.map(|_| d));
                                let failed = done.is_err();
                                if answer.send(done).is_err() || failed { break; }
                            }
                        }
                    },
                }
                if !context.is_null() { let _ = (cuda.context_synchronize)(); let _ = (cuda.context_destroy)(context); }
            });
            me.workers.push(P3Worker { tx, rx: std::sync::Mutex::new(result), join: Some(join) });
            startup.recv().map_err(|_| "DEFECT MYNN shard startup thread exited".to_owned())??;
        }
        eprintln!("{}", json!({"event":"mynn_shards_loaded","visible_ordinals":ordinals,"dycore_device":main_device,"images":paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>()}));
        Ok(Some(me))
    }
}
impl Drop for P3Shards {
    fn drop(&mut self) {
        let mut joins = Vec::new();
        for w in self.workers.iter_mut() { if let Some(j) = w.join.take() { joins.push(j); } }
        self.workers.clear();
        for j in joins { let _ = j.join(); }
    }
}

#[cfg(test)]
mod p4v2_tests {
    use super::*;
    /// speed mynn (count cache): a kept launch reads back exactly; a file of another column count, another image
    /// layout, a cut file, or one with no largest group tape is not read (the launch then counts as before).
    #[test]
    fn count_cache_file_round_trip_and_refusals() {
        let d = std::env::temp_dir().join(format!("p3cc-test-{}", std::process::id()));
        let f = d.join("p3count-s2-b0-n5-i0.bin");
        let head = [0x3532_4633_4343_3350u64, 5, 8, 479, 24, 16];
        let counts = vec![66_296i64, 0, -1, 233_228, 77_760];
        p3_cc_write(&f, &head, 1_377_776, &counts).unwrap();
        assert_eq!(p3_cc_read(&f, &head), Some((counts.clone(), 1_377_776)));
        let mut other = head; other[1] = 6;
        assert_eq!(p3_cc_read(&f, &other), None);
        let mut image = head; image[4] = 16;
        assert_eq!(p3_cc_read(&f, &image), None);
        let b = std::fs::read(&f).unwrap();
        std::fs::write(&f, &b[..b.len() - 8]).unwrap();
        assert_eq!(p3_cc_read(&f, &head), None);
        p3_cc_write(&f, &head, 0, &counts).unwrap();
        assert_eq!(p3_cc_read(&f, &head), None);
        assert_eq!(p3_cc_read(&d.join("absent.bin"), &head), None);
        let _ = std::fs::remove_dir_all(&d);
    }
    /// P4 v2: sorted chunks cover every column once, in order of falling count; each chunk's capacity holds its
    /// longest column and its tape fits the budget unless one block of the longest column alone is larger.
    #[test]
    fn sorted_chunks_cover_and_fit() {
        let n = 163_842usize;
        let counts: Vec<i32> = (0..n).map(|c| if c % 10 < 7 { 1_200 + (c % 97) as i32 } else { 4_000 + ((c * 7919) % 5_624) as i32 }).collect();
        for (budget, blk, lim) in [(2usize << 30, 64usize, 49_152usize), (256 << 20, 64, 49_152), (2 << 30, 128, 163_968), (1 << 20, 64, 49_152)] {
            let (order, chunks) = p4_tape_chunks(&counts, budget, blk, lim, true);
            assert_eq!(order.len(), n);
            let mut seen = vec![false; n];
            for &c in &order { assert!(!seen[c as usize]); seen[c as usize] = true; }
            for w in order.windows(2) { assert!(counts[w[0] as usize] >= counts[w[1] as usize]); }
            let mut next = 0usize;
            for &(c0, k, cap) in &chunks {
                assert_eq!(c0, next); assert!(k >= 1 && k <= lim.max(blk));
                for q in c0..c0 + k { assert!(counts[order[q] as usize] as usize <= cap); }
                assert!(32 * cap * k <= budget || k <= blk, "chunk {c0}+{k} cap {cap} over budget {budget}");
                next = c0 + k;
            }
            assert_eq!(next, n);
            // the sorted tape traffic is the columns' own tapes, not columns x the largest tape
            let own: usize = chunks.iter().map(|&(_, k, cap)| k * cap).sum();
            let old: usize = n * *counts.iter().max().unwrap() as usize;
            assert!(own < old);
        }
        // an older image (no order): the old uniform chunks
        let (order, chunks) = p4_tape_chunks(&counts, 2 << 30, 64, 49_152, false);
        assert!(order.is_empty());
        let cap = *counts.iter().max().unwrap() as usize;
        assert!(chunks.iter().all(|&(_, _, c)| c == cap));
        assert_eq!(chunks.iter().map(|&(_, k, _)| k).sum::<usize>(), n);
    }
}

#[cfg(test)]
mod f070_tests {
    use super::*;
    /// The tangent frame of the G4 ugwp images (p4_frame_bytes, P4F_BYTES at UGWP_LEVELS 59, rounded to 16).
    const FB_TL: usize = 24_960;
    /// F070: the ugwp frame arena is bounded on any mesh: no more than the 6.8 GiB stack reservation it replaced
    /// (25,280 B x 188 SMs x 1,536 threads), and in fact at most P4_UGWP_THREADS frames.
    #[test]
    fn ugwp_frame_arena_bounded_on_large_meshes() {
        let old_reservation = 25_280usize * 188 * 1536;
        for nc in [2_562usize, 40_962, 163_842, 655_362, 2_621_442, 65_536_002] {
            for blk in [32usize, 64, 128] {
                let nt = p4_ugwp_frame_threads(nc, P4_UGWP_THREADS, blk);
                assert!(nt * FB_TL <= old_reservation, "nc {nc} block {blk}: arena {} B > {old_reservation} B", nt * FB_TL);
                assert!(nt <= P4_UGWP_THREADS, "nc {nc} block {blk}: {nt} frames > {P4_UGWP_THREADS}");
                assert!(nt >= nc.min(P4_UGWP_THREADS), "nc {nc}: arena smaller than one chunk");
            }
        }
        assert_eq!(p4_ugwp_frame_threads(2_562, P4_UGWP_THREADS, 64), 2_624);
    }
    /// Every launch the arena serves fits it, and the chunks cover each column exactly once.
    #[test]
    fn ugwp_chunks_cover_columns_and_fit_the_arena() {
        for (nc, lim) in [(1usize, 64usize), (63, 64), (64, 64), (2_562, 1_024), (2_562, P4_UGWP_THREADS), (655_362, P4_UGWP_THREADS)] {
            let ch = p4_ugwp_chunks(nc, lim);
            let nt = p4_ugwp_frame_threads(nc, lim, 64);
            let mut next = 0;
            for &(c0, k) in &ch {
                assert_eq!(c0, next); assert!(k >= 1 && k <= lim);
                assert!(k.div_ceil(64) * 64 <= nt, "nc {nc} lim {lim}: a {k}-column launch needs more frames than {nt}");
                next = c0 + k;
            }
            assert_eq!(next, nc);
        }
    }
}

#[cfg(test)]
mod refuse_hold_native_tests {
    use super::*;
    use crate::submit::Slice;
    fn arg(name: &str, pos: usize, device: bool, dtype: &str, extent: Option<&str>, role: &str, active: bool) -> ArgRow {
        ArgRow { name: name.into(), pos, device, dtype: dtype.into(), extent: extent.map(str::to_owned), role: role.into(), active }
    }
    /// D1b: a refused P7 row is held on the float32 outputs its row names (taug, fracs), and on nothing it only reads.
    #[test]
    fn a_p7_row_names_the_float32_words_it_writes() {
        let row = Row { function: "rlw_taumol_batched".into(), unit: "P7".into(), kind: "p7_taumol".into(), raw: json!({}), args: vec![
            arg("ncol", 0, false, "<i4", None, "scalar", false), arg("nl", 1, false, "<i4", None, "scalar", false),
            arg("laytrop_v", 2, true, "<i4", Some("ncol"), "in-branch", false),
            arg("fs", 3, true, "<f4", Some("33*ncol*nl"), "in", true),
            arg("chi_mls", 4, true, "<f4", Some("413"), "const", false),
            arg("taug", 5, true, "<f4", Some("140*ncol*nl"), "out", true),
            arg("fracs", 6, true, "<f4", Some("140*ncol*nl"), "out", true)] };
        let dv = |allocation: u64, offset: usize| Argument::Device(Slice { allocation, offset });
        let args = vec![Argument::Bytes(3i32.to_le_bytes().to_vec()), Argument::Bytes(2i32.to_le_bytes().to_vec()),
            dv(10, 0), dv(11, 64), Argument::Bytes(vec![0x10, 0, 0, 0, 0, 0, 0, 0]), dv(12, 400), dv(13, 0)];
        assert_eq!(row.written_f32(&args).unwrap(), vec![(12, 400, 840), (13, 0, 840)]);
        // an output that is absent (null) at this launch writes nothing
        let mut a2 = args.clone();
        a2[6] = Argument::Bytes(vec![0; 8]);
        assert_eq!(row.written_f32(&a2).unwrap(), vec![(12, 400, 840)]);
        // an output that is neither a pointer nor null is an error by name, never a silent identity
        a2[5] = Argument::Bytes(vec![1; 8]);
        assert!(row.written_f32(&a2).unwrap_err().contains("taug is not a device pointer"));
    }
}
