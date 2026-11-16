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
        checked(&format!("cuModuleLoadData {}", path.display()), (cuda.module_load)(&mut handle, bytes.as_ptr().cast()))?;
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

/// A grow-only device buffer owned by the loader (scheme scratch: tapes, MYNN arena), outside the dot arena.
/// Nothing is ever freed while the plan replays: a freed block is reused by the plan's own later allocations,
/// and an allocation the forward does not fully write then carries this scratch's bytes into the boundary hash
/// (seen: bind1, allocation 395, a 1,024-byte step-1 allocation, differed after a mid-step cuMemFree). A grown
/// buffer keeps its old block (retired) and grows by half again, so growth is rare and bounded.
#[derive(Default)]
struct Grow { at: u64, bytes: usize, retired: Vec<u64> }
impl Grow {
    unsafe fn need(&mut self, cuda: &Cuda, bytes: usize) -> Result<u64, String> {
        if bytes > self.bytes {
            if self.at != 0 { self.retired.push(self.at); self.at = 0; }
            let b = (bytes + bytes / 2).div_ceil(1 << 20) << 20;
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
            if self.at != 0 { self.retired.push(self.at); self.at = 0; }
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
            if self.at != 0 { self.retired.push(self.at); self.at = 0; }
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
}

/// Primal and derivative buffers of one bound launch.
impl Prepared {
    /// Active segments (allocation, first byte, end byte, float64 shadow address): one float64 shadow element per
    /// forward 4-byte word from `lo`, the window's float32 shadow convention widened.
    pub fn active_segments(&self) -> Vec<(u64, usize, usize, u64)> {
        self.segs.v.iter().filter(|s| s.active).map(|s| (s.alloc, s.lo, s.hi, s.shadow)).collect()
    }
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

struct P3Image { tl: Module, ad: Module, count: Module, f_tl: *mut c_void, f_ad: *mut c_void, f_count: *mut c_void,
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
struct P4Image { tl: Module, rev: Module, frame_tl: usize, frame_rev: usize }
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
    pub loaded: Value,
    /// readbacks-device: when set (the window sets it), the P3 failure codes go to this device ledger, read once per
    /// step by the window, instead of a synchronize and a download after every P3 launch. None (the per-launch dot
    /// tools) keeps the immediate read.
    pub ledger: Option<dev::Ledger>,
    /// AD_NATIVE_ONLY=prefix,prefix: only physics rows whose function starts with one of these are live (the others
    /// stay held as without --native-schemes). Diagnostic for one-scheme-live attribution runs.
    pub only: Option<Vec<String>>,
}

fn env_var(k: &str) -> Option<String> { std::env::var(k).ok().filter(|v| !v.is_empty()) }

impl Native {
    /// AD_SCHEMES=<dir holding schemes.json and the images it names>.
    pub unsafe fn from_env(cuda: &Cuda, spec_kernels: &HashSet<String>) -> Result<Option<Self>, String> {
        let Some(dir) = env_var("AD_SCHEMES") else { return Ok(None) };
        let dir = PathBuf::from(dir);
        let table: Value = serde_json::from_slice(&std::fs::read(dir.join("schemes.json")).map_err(|e| format!("schemes.json: {e}"))?)
            .map_err(|e| format!("schemes.json: {e}"))?;
        if table["schema"] != "ad.scheme.rows/v1" { return Err("schemes.json is not ad.scheme.rows/v1".into()); }
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
            Some(P3Image { tl, ad, count, f_tl, f_ad, f_count, sz_tl: s[0] as usize, sz_tape: s[1] as usize, struct_bytes: s[3] as usize, node_bytes: s[2] as usize, tile_w: p3_tile_w, parfwd, f_p, bench: s[5] == 1 })
        } else { None };
        let mut p4 = HashMap::new();
        for fam in ["mynnsfc", "ugwp"] {
            let (t, r) = (format!("p4_{fam}_tl"), format!("p4_{fam}_rev"));
            if !table["images"][&t].is_string() { continue; }
            let (tl, rev) = (Module::load(cuda, &img(&t))?, Module::load(cuda, &img(&r))?);
            let (frame_tl, frame_rev) = (p4_frame_bytes(cuda, &tl)?, p4_frame_bytes(cuda, &rev)?);
            if fam == "ugwp" && (frame_tl == 0 || frame_rev == 0) {
                return Err("P4 ugwp images carry no p4_frame_bytes: rebuild them with p4/gen/p4_frame.py (G4 frame arena)".into()); }
            let m = P4Image { tl, rev, frame_tl, frame_rev };
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

        // 24 h recording: AD_SCHEME_PRESIZE=name=bytes,... asks for the grow-only scratch buffers up front (the same request the
        // first launch makes, so the same 1.5x block), before the window's trajectory: a long leg on a shared card otherwise
        // meets "cuMemAlloc scheme scratch" at its first adjoint after other lanes filled the card (fC: p3_bars 6.54 GB
        // at reverse step 72, card 1). Unset: nothing is allocated early (old behaviour).
        let mut me = Self { rows: rows_map, p8, l2, p3, p4, tf, p6, p7, p7sw, rl, rl_cap, rl_prev: None, scratch: HashMap::new(), tol, p3_tile, p3_budget: None, p3_pinned, p3_ad_budget: None, ledger: None,
                       p3_arena_x: table["tiles"]["p3_arena_x"].as_u64().unwrap_or(12) as usize,
                       p4_budget: table["tiles"]["p4_tape_budget_bytes"].as_u64().unwrap_or(2 << 30) as usize, loaded, only };
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

    // ------------------------------------------------------------------- lane rung3-params: parameter carrier
    /// Upload the carrier of P4 family `fam` (gen_params.py): p4_par_on / p4_par_val in both modules (the verbatim
    /// forward and the derivative copy read p4_par_on[k] ? p4_par_val[k] : literal), p4_par_t (TL tangent) in TL.
    pub unsafe fn par_upload(&self, cuda: &Cuda, fam: &str, on: &[i32; 8], val: &[f32; 8], t: &[f64; 8]) -> Result<(), String> {
        if fam == "mynnedmf" {
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

    /// Bind a row to the live launch (before the forward runs): segments, shadows, pre snapshots.
    pub unsafe fn prepare(&mut self, cuda: &Cuda, arena: &mut Arena, live: &HashMap<u64, Allocation>, function: &str,
                          args: &[Argument], grid: [u32; 3], block: [u32; 3]) -> Result<Prepared, String> {
        self.prepare_impl(cuda, arena, live, function, args, grid, block, false)
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
        self.prepare_impl(cuda, arena, live, function, args, grid, block, true)
    }
    #[allow(clippy::too_many_arguments)]
    unsafe fn prepare_impl(&mut self, cuda: &Cuda, arena: &mut Arena, live: &HashMap<u64, Allocation>, function: &str,
                          args: &[Argument], grid: [u32; 3], block: [u32; 3], defer: bool) -> Result<Prepared, String> {
        let u0 = arena.used;
        let row = self.rows.get(function).ok_or_else(|| format!("no row for {function}"))?.clone();
        let env = row.env(args);
        let mut want: Vec<(u64, usize, usize, bool)> = Vec::new();
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
        let p4 = row.kind.starts_with("p4_") || row.kind == "p7_taumol" || row.kind == "p7_rsw_taumol" || row.kind == "ruc_land";
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
        for s in segs.v.iter_mut() {
            let la = live.get(&s.alloc).ok_or_else(|| format!("unknown allocation {}", s.alloc))?;
            if s.hi > la.bytes { s.hi = la.bytes; }
            s.live = la.address + s.lo as u64;
            if s.active && !defer {
                s.shadow = region + at;
                if p3 { s.snap = snap_region + at; }
                at += (s.shadow_bytes().div_ceil(256) * 256) as u64;
            }
            let u = arena.used;
            s.pre = arena.take(s.hi - s.lo)?;
            pre_bytes += arena.used - u;
            dev::copy(cuda, s.pre, la.address + s.lo as u64, s.hi - s.lo)?;
            if p4 && !defer { s.scratch = arena.take(s.hi - s.lo)?; }
        }
        let mut p = Prepared { function: function.into(), unit: row.unit.clone(), kind, segs, region: (region, total.max(256)), args: args.to_vec(),
                      grid, block, env, identity: None, notes, deferred: Some((total, p3, p4)), transient_bytes: 0 };
        if !defer { self.bind_tables(cuda, arena, live, &mut p)?; p.deferred = None; }
        dev::sync(cuda)?;
        p.transient_bytes = (arena.used - u0).saturating_sub(pre_bytes);
        Ok(p)
    }

    /// tape-dedupe-and-pack: bind a `prepare_deferred` row's transient buffers from `arena` (the reverse scratch, reset
    /// before this row's adjoint): the same layout `prepare` builds, at new addresses. A row prepared whole is unchanged.
    pub unsafe fn bind_transient(&mut self, cuda: &Cuda, arena: &mut Arena, live: &HashMap<u64, Allocation>, p: &mut Prepared) -> Result<(), String> {
        let Some((total, p3, p4)) = p.deferred.take() else { return Ok(()) };
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
    /// The forward's own argument list with every device pointer moved to its primal scratch copy.
    fn fwd(row: &Row, p: &Prepared) -> Result<Vec<K>, String> {
        let mut pos: Vec<&ArgRow> = row.args.iter().collect();
        pos.sort_by_key(|a| a.pos);
        let mut v = Vec::new();
        for a in pos {
            let arg = p.args.get(a.pos).ok_or("argument position")?;
            v.push(if a.device { match target(arg) { Some((al, off)) => K::P(p.segs.scratch(al, off)?), None => K::P(0) } }
                   else { K::B(scalar_bytes(arg).ok_or("scalar")?.to_vec()) });
        }
        Ok(v)
    }
    unsafe fn p4(&mut self, cuda: &Cuda, p: &Prepared, adjoint: bool) -> Result<(), String> {
        let row = self.rows[&p.function].clone();
        let fam = if matches!(row.kind.as_str(), "p4_run" | "p4_diff" | "p4_dx") { "ugwp" } else { "mynnsfc" };
        let sh = |name: &str| -> Result<K, String> { let (a, o) = Self::dev_arg(&row, p, name)?; Ok(K::P(p.segs.shadow(a, o)?)) };
        let scr = |name: &str| -> Result<K, String> { let (a, o) = Self::dev_arg(&row, p, name)?; Ok(K::P(p.segs.scratch(a, o)?)) };
        Self::restore(cuda, p)?;
        let flags = self.scratch.entry("p4_flags").or_default().need(cuda, 64)?;
        dev::zero(cuda, flags, 64)?;
        let (ov, sg, st, rb) = (flags, flags + 4, flags + 8, flags + 12);
        let budget = self.p4_budget;
        // G4: the ugwp entries keep their column arrays in a frame arena sized for the largest launch below (the
        // forward's grid for the tangent; ceil(nc/64)*64 threads for the adjoint's count pass, which bounds every
        // chunk); the module global p4_frame_base points each module at it.
        if row.kind == "p4_run" {
            let nc = p.env.get("nc").copied().ok_or("ugwp_oro_run: nc")? as usize;
            if p.grid[1] != 1 || p.grid[2] != 1 || p.block[1] != 1 || p.block[2] != 1 {
                return Err(format!("ugwp_oro_run: frame arena needs a 1-D launch, got grid {:?} block {:?}", p.grid, p.block)); }
            let nt = (p.grid[0] as usize * p.block[0] as usize).max(nc.div_ceil(64) * 64);
            let fb = { let img = self.p4.get("ugwp").ok_or("P4 ugwp image not loaded")?; img.frame_tl.max(img.frame_rev) };
            let frame = self.scratch.entry("p4_frame").or_default().need(cuda, nt * fb)?;
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
                    let mut a = head()?;
                    a.extend([sh("f")?, K::P(sg)]);
                    return launch(cuda, f_tl, p.grid, p.block, &a);
                }
                // Count pass, then tape + sweep in column chunks under the budget. f is the PRE state (the hand-off).
                let counts = self.scratch.entry("p4_counts").or_default().need(cuda, 4 * n)?;
                let mut a = head()?;
                a.extend([sh("f")?, K::P(0), K::P(0), K::P(0), i32b(0), i64b(n as i64), i32b(0), i32b(n as i32), K::P(counts), K::P(ov), K::P(sg)]);
                launch(cuda, f_ad, grid1(n, 128), [128, 1, 1], &a)?;
                dev::sync(cuda)?;
                let cap = dev::download(cuda, counts, 4 * n)?.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).max().unwrap_or(0).max(1) as usize;
                let chunk = ((budget / (32 * cap)) / 128 * 128).clamp(128, n.div_ceil(128) * 128);
                let tape = self.scratch.entry("p4_tape").or_default().need(cuda, 32 * cap * chunk)?;
                let (ab, w, adj) = (tape, tape + (8 * cap * chunk) as u64, tape + (24 * cap * chunk) as u64);
                let mut c0 = 0;
                while c0 < n {
                    let nc = chunk.min(n - c0);
                    let mut a = head()?;
                    a.extend([sh("f")?, K::P(ab), K::P(w), K::P(adj), i32b(cap as i32), i64b(chunk as i64), i32b(c0 as i32), i32b(nc as i32), K::P(0), K::P(ov), K::P(0)]);
                    launch(cuda, f_ad, grid1(nc, 128), [128, 1, 1], &a)?;
                    c0 += nc;
                }
                dev::sync(cuda)?;
                if dev::download(cuda, ov, 4)? != [0, 0, 0, 0] { return Err("mynnsfc_columns adjoint tape overflow".into()); }
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
                    dev::zero(cuda, o_out.0, o_out.1)?;
                    dev::zero(cuda, s_out.0, s_out.1)?;
                    let mut a = Self::fwd(&row, p)?;
                    a.extend([sh("layers")?, sh("prsi0")?, sh("surface")?, sh("out")?, sh("sout")?, K::P(st), K::P(sg), K::P(rb)]);
                    return launch(cuda, f_tl, p.grid, p.block, &a);
                }
                let counts = self.scratch.entry("p4_counts").or_default().need(cuda, 4 * nc)?;
                let mk = |ab: u64, w: u64, adj: u64, cap: i32, stride: i64, c0: i32, ncol: i32, cnt: u64| -> Result<Vec<K>, String> {
                    let mut a = Self::fwd(&row, p)?;
                    a.extend([sh("layers")?, sh("prsi0")?, sh("surface")?, sh("out")?, sh("sout")?, K::P(ab), K::P(w), K::P(adj), i32b(cap), i64b(stride),
                              i32b(c0), i32b(ncol), K::P(cnt), K::P(ov), K::P(sg), K::P(st)]);
                    Ok(a) };
                launch(cuda, f_ad, grid1(nc, 64), [64, 1, 1], &mk(0, 0, 0, 0, nc as i64, 0, nc as i32, counts)?)?;
                dev::sync(cuda)?;
                let cap = dev::download(cuda, counts, 4 * nc)?.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).max().unwrap_or(0).max(1) as usize;
                let chunk = ((budget / (32 * cap)) / 64 * 64).clamp(64, nc.div_ceil(64) * 64);
                let tape = self.scratch.entry("p4_tape").or_default().need(cuda, 32 * cap * chunk)?;
                let (ab, w, adj) = (tape, tape + (8 * cap * chunk) as u64, tape + (24 * cap * chunk) as u64);
                let mut c0 = 0;
                while c0 < nc {
                    let k = chunk.min(nc - c0);
                    Self::restore(cuda, p)?;
                    launch(cuda, f_ad, grid1(k, 64), [64, 1, 1], &mk(ab, w, adj, cap as i32, chunk as i64, c0 as i32, k as i32, 0)?)?;
                    c0 += k;
                }
                dev::sync(cuda)?;
                if dev::download(cuda, ov, 4)? != [0, 0, 0, 0] { return Err("ugwp_oro_run adjoint tape overflow".into()); }
                dev::zero(cuda, o_out.0, o_out.1)?;
                dev::zero(cuda, s_out.0, s_out.1)?;
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
        let base = self.scratch.entry("p6").or_default().need(cuda, total)?;
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
            let base = self.scratch.entry("ruc_land").or_default().need(cuda, total)?;
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
        let base = self.scratch.entry("ruc_land").or_default().need(cuda, total)?;
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
        // Opt-in (AD_RUC_LAND_CAPPED=1); the counted path is the default. Measured on a test box's RTX PRO 6000 (the dumped
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
        for s in p.segs.v.iter().filter(|s| s.active) { dev::copy(cuda, s.snap, s.shadow, s.shadow_bytes())?; }
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
        let arena_col = ws * self.p3_arena_x;
        let cols32 = cols.div_ceil(32) * 32;
        if self.p3_budget.is_none() {
            let (mut free, mut total) = (0usize, 0usize);
            checked("cuMemGetInfo", (cuda.mem_info)(&mut free, &mut total))?;
            let want = (env_var("AD_P3_BUDGET_GIB").and_then(|v| v.parse::<f64>().ok()).unwrap_or(10.0) * (1u64 << 30) as f64) as usize;
            // tempo-mynn-kernels: the 16 GiB reserve is a quarter of the card at most. On a 32 GB card it left a 0.72 GB
            // budget for pkg18m2 (2,562 columns), so MYNN ran 384-column tiles: adjoint 2.3-2.9 s per step against 0.45 s
            // whole on a 96 GB card (tm-A4). Cards of 64 GiB or more keep 16 GiB, as before.
            // adjoint-speed-4: the quarter-card reserve below 64 GiB is withdrawn (16 GiB on every card, as speed-3). It won the
            // 4-step window (tm-A4 -> tm-R4) but its whole-launch arena (2.28 GB at pkg18m2) is held through the window, so
            // the 18-step reverse is memory-starved: RTX 5090, s4-B18 (speed3) MYNN adjoint 43.2 s over 18 steps (384-column
            // tiles) vs s4-A18b 214.3 s (68-80 reverse tiles), same words. Cards of 64 GiB or more are unchanged.
            let _ = total; let reserve = 16usize << 30;
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
        // Primal y buffers (the entries write the forward's own outputs there): contiguous, one per active segment,
        // initialised to the pre state; after the first tangent they are compared word for word with the forward.
        let total: usize = p.segs.v.iter().filter(|s| s.active).map(|s| s.hi - s.lo).sum();
        let ybase = self.scratch.entry("p3_y").or_default().need(cuda, 2 * total.max(8))?;
        let mut at = 0u64;
        let mut seg_y: HashMap<(u64, usize), u64> = HashMap::new();
        for s in p.segs.v.iter().filter(|s| s.active) {
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
                    let yy = seg_y[&(s.alloc, s.lo)] + (off - s.lo) as u64;
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
        if let Some([f_tl_p, f_ad_p, f_count_p]) = pers {
            let groups = cols.div_ceil(lanes);
            let resident = |f: *mut c_void| -> usize {
                let (mut nb, mut dv, mut sms) = (0i32, 0i32, 0i32);
                if (cuda.occupancy)(&mut nb, f, lanes as i32, 0) != 0 || (cuda.ctx_device)(&mut dv) != 0 || (cuda.device_attribute)(&mut sms, 16, dv) != 0 { return 1024; }
                (nb.max(1) as usize) * (sms.max(1) as usize)
            };
            let cap_env = env_var("AD_P3_SLOTS").and_then(|v| v.parse::<usize>().ok()).unwrap_or(usize::MAX).max(1);
            let slot_b = lanes * col_fixed;
            let sched = self.scratch.entry("p3_sched").or_default().need(cuda, 64)?;
            let (m_tl, m_ad, m_count) = { let i = self.p3.as_ref().unwrap(); (&i.tl as *const Module, &i.ad as *const Module, &i.count as *const Module) };
            let mk = |mode_k: i32, layout: i32, ar: u64, st: u64, lf: u64, ks: u64, nodes: u64, bars: u64, capacity: u64| -> Vec<u8> {
                let mut w: Vec<u8> = Vec::with_capacity(sbytes);
                let put = |w: &mut Vec<u8>, v: u64| w.extend_from_slice(&v.to_le_bytes());
                for sl in &slots { let sl = if mode_k == 2 { [sl[0], sl[1], 0, 0, 0, 0] } else { *sl }; for v in sl { put(&mut w, v); } }
                for v in [kin, kout, eflags, ar, arena_col as u64, st, col_elems as u64, lf, ks, nodes, bars, capacity, used, fail] { put(&mut w, v); }
                for v in [0i32, cols as i32, mode_k, layout] { w.extend_from_slice(&v.to_le_bytes()); }
                w
            };
            // (1) the tangent, or the reverse's count pass, over every column
            let f1 = if mode == 0 { f_tl_p } else { f_count_p };
            // the slots never outgrow the tile buffers allocated above (a grown block retires the old one: see Grow)
            // `tile` is already the budget fit with the table-tile floor (384 columns), as the tiled path uses it.
            let s1 = groups.min(resident(f1)).min(cap_env).min((tile / lanes).max(1)).max(1);
            let ar1 = self.scratch.entry("p3_arena").or_default().need(cuda, s1 * lanes * arena_col)?;
            let st1 = self.scratch.entry("p3_store").or_default().need(cuda, s1 * lanes * col_elems * sz_tl.max(sz_tape))?;
            let ks1 = self.scratch.entry("p3_kstore").or_default().need(cuda, s1 * lanes * 32)?;
            let w = mk(if mode == 0 { 0 } else { 2 }, 2, ar1, st1, 0, ks1, 0, 0, 0);
            if w.len() != sbytes { return Err(format!("P3 launch struct {} bytes, image expects {sbytes}", w.len())); }
            dev::upload(cuda, lbuf, &w)?;
            dev::zero(cuda, sched, 64)?;
            dev::zero(cuda, fail, cols32 * 16)?;
            (*(if mode == 0 { m_tl } else { m_count })).set_global(cuda, "mw_sched", &sched.to_le_bytes())?;
            let mut a = vec![K::P(lbuf)];
            a.extend(sargs.iter().cloned());
            launch(cuda, f1, [s1 as u32, 1, 1], pblock, &a)?;
            count_fails(cols, &mut fails)?;
            let mut rec = json!({"slots_first": s1, "resident_first": resident(f1)});
            if mode == 1 {
                // (2) one host read: the counts (receipts) and the largest group tape (the adjoint's region per block)
                let sc: Vec<u64> = dev::download(cuda, sched, 16)?.chunks_exact(8).map(|x| u64::from_le_bytes(x.try_into().unwrap())).collect();
                let region = sc[1] as usize;
                pers_counted = dev::download(cuda, used, cols * 8)?.chunks_exact(8).map(|x| i64::from_le_bytes(x.try_into().unwrap())).collect();
                let budget_ad = {
                    let (mut free, mut total) = (0usize, 0usize);
                    checked("cuMemGetInfo", (cuda.mem_info)(&mut free, &mut total))?;
                    let held = ["p3_nodes", "p3_bars", "p3_arena", "p3_store", "p3_leaf", "p3_kstore"]
                        .iter().map(|k| self.scratch.get(k).map(|g| g.bytes).unwrap_or(0)).sum::<usize>();
                    let cap = (env_var("AD_P3_AD_BUDGET_GIB").and_then(|v| v.parse::<f64>().ok()).unwrap_or(32.0) * (1u64 << 30) as f64) as usize;
                    let reserve_ad = if total >= (64usize << 30) { 12usize << 30 } else { total / 8 };
                    let head = if total >= (64usize << 30) { 2usize << 30 } else { reserve_ad };
                    (free + held).saturating_sub(reserve_ad).min(cap).max(budget.min((free + held).saturating_sub(head)))
                };
                self.p3_ad_budget = Some(budget_ad);
                let per_slot = slot_b + region * (node_b + 8);
                // mynn-adjoint-v2 (pk120 fix): the slots are sized by the memory they ADD. budget_ad counts every P3 block
                // already held as room, but those blocks are kept (a grown block retires the old one, never freed: see Grow),
                // so the old `budget_ad / per_slot` let the tape take the room of the tangent's arena/store slots that the
                // adjoint does not use. A test box mav2-K120G (pk120, 40,962 cells, 96 GB card at ~89 GB in use): the first
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
                // columns the packed tiled path holds in the same bytes (a test box pk120: 88-115 slots, 704-920 columns in
                // flight in 4.4 GB, MYNN adjoint 31.0 s over the window against 28.3 s tiled, mav2-K120EC4 / K120T4).
                // The ring packs the groups in order by their own counts into ONE tape all blocks share; a block waits
                // until the groups whose ring nodes it would reuse are done (see mw_ring in gen_whole.py). The slots are
                // the ones the held arena/store already cover (growth at most a quarter of the room), the rest of the
                // room is ring. AD_P3_RING=0 keeps layout 2; an image without mw_ring keeps layout 2.
                let ring_img = env_var("AD_P3_RING").as_deref() != Some("0") && (*m_ad).set_global(cuda, "mw_ring", &0u64.to_le_bytes()).is_ok();
                let mut ring: Option<(usize, usize)> = None;
                if ring_img {
                    // Ring sizing (v2). Groups in flight = min(slots, ring nodes / mean group). Each slot past what the held
                    // arena/store cover reallocates them (1/2 slack, the old blocks stay held), each ring node costs 24 B
                    // (+1/8 slack). The slots are the most that keep the groups in flight at their maximum for the room (a plateau:
                    // slots past it only cost ring), and the ring is at most what they can use (slots x largest group, never more than
                    // the whole tape), so no ring bytes sit idle (a test box mav2-G18R4 v1: a 30 GB ring for an 8.4 GB tape).
                    let nontape = |s: usize| grow(s * lanes * arena_col, h_ar, 2) + grow(s * lanes * col_elems * sz_tl.max(sz_tape), h_st, 2)
                        + grow(s * lanes * 32, h_ks, 2) + grow(s * lanes * col_elems * 4, h_lf, 2);
                    let total_nodes: usize = pers_counted.iter().map(|&u| u.max(0) as usize + 16).sum();
                    let mean_g = (total_nodes / groups.max(1)).max(1);
                    let have_ring = (h_nodes / node_b.max(1)).min(h_bars / 8);
                    let fresh = |s: usize| room_new.saturating_sub(nontape(s)) / ((node_b + 8) * 9 / 8);
                    let ringcap = |s: usize| have_ring.max(fresh(s));
                    let s_cap = groups.min(resident(f_ad_p)).min(cap_env).max(1);
                    let (mut best_s, mut best_e) = (1usize, 0usize);
                    let mut sc = 1usize;
                    while sc <= s_cap {
                        // a slot count whose arena/store/kstore/leaf growth does not fit the room is out (a test box mav2-K120F4,
                        // ring v2: the second pk120 launch reused the ring, so every larger slot count tied, the plateau rule
                        // took 2,256 slots and asked for 23.9 GB of arena: cuMemAlloc CUDA status 2)
                        let e = if nontape(sc) > room_new { 0 } else { sc.min(ringcap(sc) / mean_g) };
                        if e >= best_e { best_e = e; best_s = sc; }
                        sc = if sc < s_cap && sc + 8 > s_cap { s_cap } else { sc + 8 };
                    }
                    let want = (best_s * region).min(total_nodes + region);
                    let pn = if have_ring >= want.min(ringcap(best_s)) && have_ring >= region { have_ring } else { want.min(fresh(best_s)) };
                    if pn >= region { ring = Some((best_s, pn)); }
                }
                if let Some((sr, pn0)) = ring {
                    let nodes = self.scratch.entry("p3_nodes").or_default().need_slack(cuda, pn0 * node_b, false)?;
                    let bars = self.scratch.entry("p3_bars").or_default().need_slack(cuda, pn0 * 8, false)?;
                    let pn = (self.scratch.get("p3_nodes").map(|g| g.bytes).unwrap_or(0) / node_b.max(1)).min(self.scratch.get("p3_bars").map(|g| g.bytes).unwrap_or(0) / 8);
                    let ar2 = self.scratch.entry("p3_arena").or_default().need(cuda, sr * lanes * arena_col)?;
                    let st2 = self.scratch.entry("p3_store").or_default().need(cuda, sr * lanes * col_elems * sz_tl.max(sz_tape))?;
                    let ks2 = self.scratch.entry("p3_kstore").or_default().need(cuda, sr * lanes * 32)?;
                    let lf2 = self.scratch.entry("p3_leaf").or_default().need(cuda, sr * lanes * col_elems * 4)?;
                    // the ring plan: vstart[groups + 1] (virtual, monotone; a group never straddles the ring end), phys[groups], done[groups]
                    let pnu = pn as u64;
                    let mut vs: Vec<u64> = Vec::with_capacity(groups + 1);
                    let mut ph: Vec<u64> = Vec::with_capacity(groups);
                    let mut v = 0u64;
                    for g in 0..groups {
                        let sz: u64 = pers_counted[g * lanes..((g + 1) * lanes).min(cols)].iter().map(|&u| u.max(0) as u64 + 16).sum();
                        let off = v % pnu;
                        if off + sz > pnu { v += pnu - off; }
                        vs.push(v); ph.push(v % pnu); v += sz;
                    }
                    vs.push(v);
                    let mut rb: Vec<u8> = Vec::with_capacity((2 * groups + 1) * 8 + groups * 4 + 8);
                    for x in vs.iter().chain(ph.iter()) { rb.extend_from_slice(&x.to_le_bytes()); }
                    rb.resize(rb.len() + groups * 4, 0u8);
                    let rbuf = self.scratch.entry("p3_ring").or_default().need(cuda, rb.len())?;
                    dev::upload(cuda, rbuf, &rb)?;
                    (*m_ad).set_global(cuda, "mw_ring", &rbuf.to_le_bytes())?;
                    let w = mk(1, 3, ar2, st2, lf2, ks2, nodes, bars, pnu);
                    dev::upload(cuda, lbuf, &w)?;
                    dev::zero(cuda, sched, 64)?;
                    dev::zero(cuda, fail, cols32 * 16)?;
                    (*m_ad).set_global(cuda, "mw_sched", &sched.to_le_bytes())?;
                    let mut a = vec![K::P(lbuf)];
                    a.extend(sargs.iter().cloned());
                    launch(cuda, f_ad_p, [sr as u32, 1, 1], pblock, &a)?;
                    count_fails(cols, &mut fails)?;
                    let tot: usize = pers_counted.iter().map(|&u| u.max(0) as usize + 16).sum();
                    caps.push(region);
                    pers_tiles = Some((vec![(0, cols, tot)], budget_ad));
                    rec["layout"] = json!(3); rec["slots_ad"] = json!(sr); rec["resident_ad"] = json!(resident(f_ad_p)); rec["region_nodes"] = json!(region);
                    rec["ring_nodes"] = json!(pn); rec["ring_wraps"] = json!(v / pnu.max(1)); rec["tape_bytes_held"] = json!(pn * (node_b + 8));
                    rec["budget_ad"] = json!(budget_ad); rec["room_new"] = json!(room_new); rec["slots_top"] = json!(s_top);
                } else {
                let half = have == 0 && new_bytes(s2, true) <= room_new;
                let nodes = self.scratch.entry("p3_nodes").or_default().need_slack(cuda, s2 * region * node_b, half)?;
                let bars = self.scratch.entry("p3_bars").or_default().need_slack(cuda, s2 * region * 8, half)?;
                let ar2 = self.scratch.entry("p3_arena").or_default().need(cuda, s2 * lanes * arena_col)?;
                let st2 = self.scratch.entry("p3_store").or_default().need(cuda, s2 * lanes * col_elems * sz_tl.max(sz_tape))?;
                let ks2 = self.scratch.entry("p3_kstore").or_default().need(cuda, s2 * lanes * 32)?;
                let lf2 = self.scratch.entry("p3_leaf").or_default().need(cuda, s2 * lanes * col_elems * 4)?;
                // (3) the adjoint over every column, tape regions of `region` nodes per block
                let w = mk(1, 2, ar2, st2, lf2, ks2, nodes, bars, region as u64);
                dev::upload(cuda, lbuf, &w)?;
                dev::zero(cuda, sched, 64)?;
                dev::zero(cuda, fail, cols32 * 16)?;
                (*m_ad).set_global(cuda, "mw_sched", &sched.to_le_bytes())?;
                let mut a = vec![K::P(lbuf)];
                a.extend(sargs.iter().cloned());
                launch(cuda, f_ad_p, [s2 as u32, 1, 1], pblock, &a)?;
                count_fails(cols, &mut fails)?;
                let tot: usize = pers_counted.iter().map(|&u| u.max(0) as usize + 16).sum();
                caps.push(region);
                pers_tiles = Some((vec![(0, cols, tot)], budget_ad));
                rec["slots_ad"] = json!(s2); rec["resident_ad"] = json!(resident(f_ad_p)); rec["region_nodes"] = json!(region);
                rec["tape_bytes_held"] = json!(s2 * region * (node_b + 8));
                rec["budget_ad"] = json!(budget_ad); rec["room_new"] = json!(room_new); rec["slots_top"] = json!(s_top);
                }
            }
            persisted = Some(rec);
            first = cols;
        }
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
            let budget_ad = {
                let (mut free, mut total) = (0usize, 0usize);
                checked("cuMemGetInfo", (cuda.mem_info)(&mut free, &mut total))?;
                // tempo-mynn-kernels: the tile's arena/store/leaf blocks were allocated above (before the count pass), so
                // free memory already excludes them, while room_raw below subtracts tile * col_fixed for them again. On a
                // 32 GB card with a whole-launch tile that double count took ~3.2 GB of real room: tm-N18b packed 81
                // reverse tiles (one per 32-column group) and the MYNN-EDMF adjoint took 12-17 s per step. Count them as held.
                let held = ["p3_nodes", "p3_bars", "p3_arena", "p3_store", "p3_leaf", "p3_kstore", "p3_fail", "p3_used", "p3_used_hw", "p3_dom"]
                    .iter().map(|k| self.scratch.get(k).map(|g| g.bytes).unwrap_or(0)).sum::<usize>();
                let cap = (env_var("AD_P3_AD_BUDGET_GIB").and_then(|v| v.parse::<f64>().ok()).unwrap_or(32.0) * (1u64 << 30) as f64) as usize;
                // tempo-mynn-kernels: reserve 12 GiB, or an eighth of a card under 64 GiB; the floor at the tangent budget
                // never exceeds what is free (less 2 GiB), so a tight card packs smaller reverse tiles instead of failing.
                let reserve_ad = if total >= (64usize << 30) { 12usize << 30 } else { total / 8 };
                // tempo-mynn-adjoint-kernels: below 64 GiB the floor keeps the same eighth of the card free as the main term:
                // with the arena counted as held (above), a 2 GiB headroom let the 18-step window's MYNN tape take the memory
                // the RUC land adjoint needs later in the step (tm-R18: cuMemAlloc 1 GiB failed, CUDA 2, step 18).
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
        Ok(json!({"domain_flags": domain, "tile_columns": tile, "tape_nodes_per_column_max": if node_b == 16 && ad_tiles.is_some() { first_counted.iter().max().map(|&m| m as usize + 16) } else { cmax },
                  "tape_nodes_tile_max": cmax,
                  "tape_bytes_per_tile_max": if node_b == 16 && ad_tiles.is_some() { cmax.map(|c| c * (node_b + 8)) } else { cmax.map(|c| c * (node_b + 8) * tile) },
                  "tape_layout": if node_b == 16 && ad_tiles.is_some() { "packed" } else { "capacity" }, "lanes": lanes, "arena_bytes_per_tile": tile * arena_col,
                  "whole_launch": whole, "budget_bytes": budget, "ad_budget_bytes": self.p3_ad_budget, "ad_tiles": ad_tiles.as_ref().map(|t| t.0.iter().map(|x| json!([x.0, x.1, x.2])).collect::<Vec<_>>()),
                  "arena_hw": arena_hw, "persistent": persisted, "count_max": first_counted.iter().max(), "count_mean": if first_counted.is_empty() { None } else { Some(first_counted.iter().sum::<i64>() / first_counted.len() as i64) }}))
    }

    // --------------------------------------------------------------------------------------- dispatch
    /// Tangent in place on the shadows.
    pub unsafe fn tangent(&mut self, cuda: &Cuda, arena: &mut Arena, p: &mut Prepared, identity: bool) -> Result<Value, String> {
        match p.kind {
            Kind::P3 => self.p3(cuda, p, 0, identity),
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
            Kind::P3 => self.p3(cuda, p, 1, false),
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
