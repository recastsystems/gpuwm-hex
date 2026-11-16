// SPDX-License-Identifier: Apache-2.0
//! Physics-on window, lane L5 (TEMPO live): the per-launch derivative dot of every tempo_warp_steps launch with
//! TEMPO's warp-shaped whole-launch tangent and checkpointed adjoint (tree/ad/phys/tempo, tempo-ad-gen --warp).
//!
//! The derivative is the generated tempo_adw module: one 32-lane warp per column exactly as the forecast runs
//! TEMPO, a per-lane executed-branch tape, one tape per substep (checkpoints at every substep boundary). It
//! needs host scheduling (a checkpoint/count pass, then every substep's record and sweep, tape blocks sized
//! from the counts), so it is driven here instead of from an ad.entry/v1 row. Before the forward launch the
//! inputs it reads (f[0..17], s[0..9] through the launch's pointer tables) are copied into the arena with
//! mirror tables; after the forward launch the dot runs on that copy only:
//!   - checkpoint/count pass over every column, and its outputs against the forward's own post-state
//!     (identity: the augmented primal must equal the launch's result word for word);
//!   - tangent over the launch (segments forward) and adjoint (segments in reverse) with a random
//!     direction xdot (scaled per input row) and a random weight ybar, <J xdot, ybar> = <xdot, J^T ybar>;
//!   - on checked launches: zero direction both ways and a second reverse run, bitwise.
//! Enabled by AD_DOT_TEMPO_CUBIN (the sm_120 image of tempo_adw); without it nothing changes. tempo_precip and
//! tempo_radar (one thread per column in the forecast too) take the P5 per-thread entries (AD_DOT_TEMPO_SERIAL_CUBIN).
use crate::cuda::{checked, Cuda};
use crate::dev::{self, Arena};
use crate::submit::{Allocation, Argument};
use serde_json::{json, Value};
use std::{collections::HashMap, ffi::{c_void, CString}, ptr, time::Instant};

const BLOCK: u32 = 128;
const NCNT: usize = 8;
/// Output fields of the launch (f[0] theta and f[4..16]); the state layout has 18 (driver_warp.cuh AD_NF_OUT).
const AD_NF_OUT: usize = 14;

pub struct TempoAdw { rec: *mut c_void, tl: *mut c_void, adj: *mut c_void, g: u64, work: usize, state: usize, prim: usize, pub image_sha256: String,
                      /// tempo-once: wall ms of the last apply's parts [count pass, substep records, sweeps, rest] (Instant reads only).
                      pub parts: std::cell::Cell<[f64; 4]>,
                      /// tempo-once: the whole-launch fused tangent module (tempo_adw_tlf), built by mk_tlf.py from the same generated
                      /// source; when loaded the tangent pass runs it instead of count pass + per-substep records + tl sweeps.
                      pub tlf: Option<TempoTlf>,
                      /// tempo-once: [tlf launches, columns rerun at a larger cap, fallbacks to the segmented tangent, check mismatches]
                      pub tlf_stats: std::cell::Cell<[u64; 4]>,
                      /// tempo-once: per column, the node rows tempo_adw_tlf needed at the previous call (sizes the next call's caps)
                      tlf_need: std::cell::RefCell<Vec<u32>>,
                      /// tempo-adjoint-v2: the image's column gather/scatter (tempo_adw_ptile, mk_sort.py). Present: the reverse and
                      /// the segmented tangent tile the columns in order of their tape rows (largest first), so a tile's columns
                      /// carry similar chains instead of every tile waiting on its one heaviest column. Absent (older images) or
                      /// AD_TEMPO_SORT=0: tiles in column order, as before. Columns are independent, so the words are the same.
                      ptile: Option<*mut c_void>,
                      /// tempo-adjoint-v2: the image's lane-capacity kernels (tempo_adw_lcap, tempo_adw_lsum; mk_lanecap.py). Present: a
                      /// column block holds each lane's own rows (lanes' count-pass rows + 64 each), g_ad.ccap is [column][32].
                      lanecap: Option<(*mut c_void, *mut c_void)> }

/// dual: the module is mk_tlfd.py's (tempo_adw_tlfd: every value carries its tangent; no node buffer, one launch).
/// state / work: its own AdState and TempoWork sizes (the dual build's are larger).
pub struct TempoTlf { f: *mut c_void, g: u64, gtan: u64, pub image_sha256: String, pub dual: bool, state: usize, work: usize }

pub struct Snap {
    pub nz: usize, pub columns: usize, pub steps: usize, dt: f32, pub flags: [i32; 7], blob: u64,
    ftab: u64, stab: u64, fsnap: [u64; 18], ssnap: [u64; 10],
    /// Live pointers of the launch's tables (inputs f[0..17], s[0..9]; profile outputs p[0..3]): where the launch reads
    /// and writes in place. The window binding maps them to its shadows.
    pub fpost: [u64; 18], pub spost: [u64; 10], pub ppost: [u64; 4],
}

unsafe fn launch(cuda: &Cuda, f: *mut c_void, grid: u32, args: &[Vec<u8>]) -> Result<(), String> {
    let mut store: Vec<Vec<u64>> = args.iter().map(|b| { let mut w = vec![0u64; b.len().div_ceil(8)]; ptr::copy_nonoverlapping(b.as_ptr(), w.as_mut_ptr().cast::<u8>(), b.len()); w }).collect();
    let mut ps: Vec<*mut c_void> = store.iter_mut().map(|w| w.as_mut_ptr().cast()).collect();
    checked("tempo_adw launch", (cuda.launch)(f, grid, 1, 1, BLOCK, 1, 1, 0, ptr::null_mut(), ps.as_mut_ptr(), ptr::null_mut()))?;
    dev::sync(cuda)
}
/// readbacks-device: the same launch without the host wait (the window's TEMPO path reads nothing back between launches).
unsafe fn launch_ns(cuda: &Cuda, f: *mut c_void, grid: u32, args: &[Vec<u8>]) -> Result<(), String> {
    let mut store: Vec<Vec<u64>> = args.iter().map(|b| { let mut w = vec![0u64; b.len().div_ceil(8)]; ptr::copy_nonoverlapping(b.as_ptr(), w.as_mut_ptr().cast::<u8>(), b.len()); w }).collect();
    let mut ps: Vec<*mut c_void> = store.iter_mut().map(|w| w.as_mut_ptr().cast()).collect();
    checked("tempo_adw launch", (cuda.launch)(f, grid, 1, 1, BLOCK, 1, 1, 0, ptr::null_mut(), ps.as_mut_ptr(), ptr::null_mut()))
}
fn i32b(v: i32) -> Vec<u8> { v.to_le_bytes().to_vec() }
fn u64b(v: u64) -> Vec<u8> { v.to_le_bytes().to_vec() }
fn f32b(v: f32) -> Vec<u8> { v.to_le_bytes().to_vec() }
fn grid(n: usize) -> u32 { ((n * 32) as u32).div_ceil(BLOCK) }
unsafe fn get<T: Copy + Default>(cuda: &Cuda, src: u64, n: usize) -> Result<Vec<T>, String> {
    let b = dev::download(cuda, src, n * std::mem::size_of::<T>())?;
    let mut v = vec![T::default(); n];
    ptr::copy_nonoverlapping(b.as_ptr(), v.as_mut_ptr().cast::<u8>(), b.len());
    Ok(v)
}
unsafe fn put<T: Copy>(cuda: &Cuda, dst: u64, v: &[T]) -> Result<(), String> {
    dev::upload(cuda, dst, std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)))
}

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Rng { Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xD1B54A32D192ED03) }
    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn unif(&mut self) -> f64 { ((self.u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64 }
    fn normal(&mut self) -> f64 { let (a, b) = (self.unif(), self.unif()); (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos() }
}

impl TempoAdw {
    pub unsafe fn load(cuda: &Cuda, path: &str) -> Result<Self, String> {
        let mut image = std::fs::read(path).map_err(|e| format!("AD_DOT_TEMPO_CUBIN {path}: {e}"))?;
        let image_sha256 = { use sha2::Digest; format!("{:x}", sha2::Sha256::digest(&image)) };
        image.push(0);
        let mut m = ptr::null_mut();
        checked("cuModuleLoadData tempo_adw", (cuda.module_load)(&mut m, image.as_ptr().cast()))?;
        let func = |n: &str| -> Result<*mut c_void, String> {
            let c = CString::new(n).unwrap(); let mut f = ptr::null_mut();
            checked(&format!("cuModuleGetFunction {n}"), (cuda.module_function)(&mut f, m, c.as_ptr()))?; Ok(f)
        };
        let (mut g, mut gn) = (0u64, 0usize);
        let c = CString::new("g_ad").unwrap();
        checked("cuModuleGetGlobal g_ad", (cuda.module_global)(&mut g, &mut gn, m, c.as_ptr()))?;
        if gn != 144 { return Err(format!("tempo_adw g_ad is {gn} bytes, expected 144 (warp tape layout)")); }
        let mut sz = 0u64;
        checked("cuMemAlloc sizes", (cuda.allocate)(&mut sz, 32))?;
        launch(cuda, func("tempo_adw_sizes")?, 1, &[u64b(sz)])?;
        let s: Vec<u64> = get(cuda, sz, 4)?;
        // Zero before freeing: the replay's later small allocations may reuse this block, and the sealed plan
        // relies on fresh allocations reading zero (allocation 395 at step 1 read our sizes otherwise).
        dev::zero(cuda, sz, 32)?; dev::sync(cuda)?;
        checked("cuMemFree sizes", (cuda.free)(sz))?;
        let ptile = { let c = CString::new("tempo_adw_ptile").unwrap(); let mut f = ptr::null_mut();
                      if (cuda.module_function)(&mut f, m, c.as_ptr()) == 0 && !f.is_null() { Some(f) } else { None } };
        let opt = |n: &str| -> Option<*mut c_void> { let c = CString::new(n).unwrap(); let mut f = ptr::null_mut();
                      if (cuda.module_function)(&mut f, m, c.as_ptr()) == 0 && !f.is_null() { Some(f) } else { None } };
        let lanecap = match (opt("tempo_adw_lcap"), opt("tempo_adw_lsum")) { (Some(a), Some(b)) => Some((a, b)), _ => None };
        Ok(Self { rec: func("tempo_adw_record")?, tl: func("tempo_adw_tl")?, adj: func("tempo_adw_adj")?, g, work: s[0] as usize, state: s[1] as usize, prim: s[2] as usize, image_sha256,
                   parts: std::cell::Cell::new([0.0; 4]), tlf: None, tlf_stats: std::cell::Cell::new([0; 4]), tlf_need: std::cell::RefCell::new(Vec::new()),
                   ptile, lanecap })
    }

    /// tempo-once: load the fused tangent module (mk_tlf.py output, same generated source). Its own g_ad and g_tan
    /// globals; its tempo_adw_sizes must equal this image's (state, work, primal width).
    pub unsafe fn load_tlf(&mut self, cuda: &Cuda, path: &str) -> Result<(), String> {
        let mut image = std::fs::read(path).map_err(|e| format!("TEMPO tlf image {path}: {e}"))?;
        let image_sha256 = { use sha2::Digest; format!("{:x}", sha2::Sha256::digest(&image)) };
        image.push(0);
        let mut m = ptr::null_mut();
        checked("cuModuleLoadData tempo_adw_tlf", (cuda.module_load)(&mut m, image.as_ptr().cast()))?;
        let func = |n: &str| -> Result<*mut c_void, String> {
            let c = CString::new(n).unwrap(); let mut f = ptr::null_mut();
            checked(&format!("cuModuleGetFunction {n}"), (cuda.module_function)(&mut f, m, c.as_ptr()))?; Ok(f)
        };
        let global = |n: &str, want: usize| -> Result<u64, String> {
            let (mut g, mut gn) = (0u64, 0usize); let c = CString::new(n).unwrap();
            checked(&format!("cuModuleGetGlobal {n}"), (cuda.module_global)(&mut g, &mut gn, m, c.as_ptr()))?;
            if gn != want { return Err(format!("tempo_adw_tlf {n} is {gn} bytes, expected {want}")); }
            Ok(g)
        };
        let dual = { let c = CString::new("tempo_adw_tlfd").unwrap(); let mut f = ptr::null_mut();
                     (cuda.module_function)(&mut f, m, c.as_ptr()) == 0 };
        let (g, gtan) = (global("g_ad", 144)?, if dual { 0 } else { global("g_tan", 8)? });
        let mut sz = 0u64;
        checked("cuMemAlloc sizes", (cuda.allocate)(&mut sz, 32))?;
        launch(cuda, func("tempo_adw_sizes")?, 1, &[u64b(sz)])?;
        let v: Vec<u64> = get(cuda, sz, 4)?;
        dev::zero(cuda, sz, 32)?; dev::sync(cuda)?;
        checked("cuMemFree sizes", (cuda.free)(sz))?;
        if v[2] as usize != self.prim || (!dual && (v[0] as usize, v[1] as usize) != (self.work, self.state)) {
            return Err(format!("TEMPO tlf image {path}: sizes {:?} differ from the tempo_adw image's {:?}: not built from the same source",
                               &v[..3], (self.work, self.state, self.prim)));
        }
        let f = func(if dual { "tempo_adw_tlfd" } else { "tempo_adw_tlf" })?;
        self.tlf = Some(TempoTlf { f, g, gtan, image_sha256, dual, state: v[1] as usize, work: v[0] as usize });
        Ok(())
    }

    /// tempo-once: the tangent of one launch by tempo_adw_tlfd (dual numbers: one launch, nothing but the state, work
    /// and in/out vectors in the arena). Returns as apply_tlf.
    unsafe fn apply_tlfd(&self, cuda: &Cuda, arena: &mut Arena, s: &Snap, vd: u64, rd: u64, identity: bool, led: &mut dev::Ledger)
                         -> Result<Option<(Option<u64>, usize, usize)>, String> {
        let tf = self.tlf.as_ref().ok_or("tempo_adw_tlfd not loaded")?;
        let (nz, nc, steps) = (s.nz, s.columns, s.steps);
        let nin = 18 * nz + 10;
        let prim = self.prim;
        // host-leftovers: x-dot and y-dot are the window's device vectors vd / rd ([slot][column] float64, the layout
        // the kernel reads and writes), so no host copy of either and no arena copy
        let need = tf.state * nc + tf.work * nc + prim * nin * nc + 8 * nin * nc + 4 * NCNT * 32 * nc + 4 * nc + (64 << 20);
        let cap_env = std::env::var("AD_TEMPO_TLF_TILE").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&v| v > 0);
        if need > arena.bytes.saturating_sub(arena.used) || cap_env.is_some_and(|c| c < nc) {
            return self.apply_tlfd_tiles(cuda, arena, s, vd, rd, identity, led, cap_env);
        }
        let cols: Vec<i32> = (0..nc as i32).collect();
        let colsb = arena.take(4 * nc)?; put(cuda, colsb, &cols)?;
        let (st, wk, yo, oi, ii, cn) = (arena.take(tf.state * nc)?, arena.take(tf.work * nc)?, arena.take(prim * nin * nc)?,
                                        arena.take(4 * nin * nc)?, arena.take(4 * nin * nc)?, arena.take(4 * NCNT * 32 * nc)?);
        let (xd, yd) = (vd, rd);
        put(cuda, tf.g, &[0u64, 0, 0, 0, 0, 0, cn, 0, 0, (nc * 32) as u64, 0, 0, 0, 0, 0, 0, 0, 0])?;
        let args = vec![i32b(nz as i32), i32b(nc as i32), i32b(steps as i32), f32b(s.dt), u64b(s.ftab), u64b(s.stab), u64b(s.blob),
             i32b(s.flags[0]), i32b(s.flags[1]), i32b(s.flags[3]), i32b(s.flags[4]), i32b(s.flags[5]), i32b(s.flags[6]),
             u64b(colsb), i32b(nc as i32), i32b(0), i32b(steps as i32), u64b(0), u64b(0), u64b(0), (nc as i64).to_le_bytes().to_vec(), i32b(0),
             u64b(st), u64b(wk), u64b(yo), u64b(oi), u64b(ii), u64b(0), u64b(xd), u64b(yd)];
        launch_ns(cuda, tf.f, grid(nc), &args)?;
        let (icmp, itok) = self.identity_ledger(cuda, s, yo, identity, led)?;
        let mut z = self.tlf_stats.get(); z[0] += 1; self.tlf_stats.set(z);
        Ok(Some((itok, icmp, 1)))
    }

    /// tempo-adjoint-v2: apply_tlfd in contiguous column tiles (the whole launch does not fit the arena, or
    /// AD_TEMPO_TLF_TILE caps the tile). Each tile: x-dot cut from vd, one tempo_adw_tlfd launch over the tile's
    /// columns, the identity rows compared against the forward's post-state at the tile's columns, y-dot put back
    /// into rd. None: not even one column fits (the caller takes the segmented tangent). Returns as apply_tlfd,
    /// with the tile count.
    #[allow(clippy::too_many_arguments)]
    unsafe fn apply_tlfd_tiles(&self, cuda: &Cuda, arena: &mut Arena, s: &Snap, vd: u64, rd: u64, identity: bool, led: &mut dev::Ledger,
                               cap: Option<usize>) -> Result<Option<(Option<u64>, usize, usize)>, String> {
        let tf = self.tlf.as_ref().ok_or("tempo_adw_tlfd not loaded")?;
        let (nz, nc, steps) = (s.nz, s.columns, s.steps);
        let nin = 18 * nz + 10;
        let prim = self.prim;
        let cols: Vec<i32> = (0..nc as i32).collect();
        let colsb = arena.take(4 * nc)?; put(cuda, colsb, &cols)?;
        // per tile column: state, work, augmented primal, x-dot and y-dot tiles, oidx/iidx, count rows
        let per = tf.state + tf.work + prim * nin + 2 * 8 * nin + 2 * 4 * nin + 4 * NCNT * 32;
        let avail = arena.bytes.saturating_sub(arena.used + 9 * 256 + (64 << 20));
        let mut nt = (avail / per).min(nc);
        if let Some(c) = cap { nt = nt.min(c); }
        if nt == 0 { return Ok(None); }
        let (st, wk, yo, oi, ii, cn) = (arena.take(tf.state * nt)?, arena.take(tf.work * nt)?, arena.take(prim * nin * nt)?,
                                        arena.take(4 * nin * nt)?, arena.take(4 * nin * nt)?, arena.take(4 * NCNT * 32 * nt)?);
        let (xt, yt) = (arena.take(8 * nin * nt)?, arena.take(8 * nin * nt)?);
        let cnt = if identity { Some(led.reserve(cuda, 8, dev::LedgerItem::Count)?) } else { None };
        let mut icmp = 0usize;
        let mut tiles = 0usize;
        let mut c0 = 0usize;
        while c0 < nc {
            let n = nt.min(nc - c0);
            put(cuda, tf.g, &[0u64, 0, 0, 0, 0, 0, cn, 0, 0, (n * 32) as u64, 0, 0, 0, 0, 0, 0, 0, 0])?;
            led.tile(cuda, vd, xt, nin, nc, c0, n, false)?;
            let args = vec![i32b(nz as i32), i32b(nc as i32), i32b(steps as i32), f32b(s.dt), u64b(s.ftab), u64b(s.stab), u64b(s.blob),
                 i32b(s.flags[0]), i32b(s.flags[1]), i32b(s.flags[3]), i32b(s.flags[4]), i32b(s.flags[5]), i32b(s.flags[6]),
                 u64b(colsb + 4 * c0 as u64), i32b(n as i32), i32b(0), i32b(steps as i32), u64b(0), u64b(0), u64b(0), (nc as i64).to_le_bytes().to_vec(), i32b(0),
                 u64b(st), u64b(wk), u64b(yo), u64b(oi), u64b(ii), u64b(0), u64b(xt), u64b(yt)];
            launch_ns(cuda, tf.f, grid(n), &args)?;
            if let Some((cw, _)) = cnt {
                // identity_ledger's rows, at the tile's columns: tile row q (n words, stride n) against the
                // post-state row at column c0 (stride nc)
                let mut cmp = |slot0: usize, src: u64, n_per: usize| -> Result<(), String> {
                    if src == 0 { return Ok(()); }
                    icmp += n_per * n;
                    for r in 0..n_per {
                        led.diff_words(cuda, yo + (prim * (slot0 + r) * n) as u64, src + (4 * (r * nc + c0)) as u64, n, prim, cw)?;
                    }
                    Ok(())
                };
                cmp(0, s.fpost[0], nz)?;
                for j in 4..17 { cmp((j - 3) * nz, s.fpost[j], nz)?; }
                for j in 0..10 { cmp(14 * nz + j, s.spost[j], 1)?; }
                for p in 0..4 { if p == 0 || s.flags[6] != 0 { cmp(14 * nz + 10 + p * nz, s.ppost[p], nz)?; } }
            }
            led.tile(cuda, yt, rd, nin, nc, c0, n, true)?;
            tiles += 1;
            c0 += n;
        }
        eprintln!("{}", json!({"event":"tempo_tlfd_tiles","columns":nc,"tile_columns":nt,"tiles":tiles,"bytes_per_column":per,"arena_avail":avail}));
        let mut z = self.tlf_stats.get(); z[0] += tiles as u64; self.tlf_stats.set(z);
        Ok(Some((cnt.map(|c| c.1), icmp, tiles)))
    }

    /// host-leftovers: the augmented primal `yo` ([slot][column], prim bytes a word) against the forward's post-state,
    /// word for word, counted on the device into the step ledger (as apply_in). Returns (compared words, token).
    unsafe fn identity_ledger(&self, cuda: &Cuda, s: &Snap, yo: u64, identity: bool, led: &mut dev::Ledger) -> Result<(usize, Option<u64>), String> {
        if !identity { return Ok((0, None)); }
        let (nz, nc, prim) = (s.nz, s.columns, self.prim);
        let (cnt, tok) = led.reserve(cuda, 8, dev::LedgerItem::Count)?;
        let mut icmp = 0usize;
        let mut cmp = |slot0: usize, src: u64, n_per: usize| -> Result<(), String> {
            if src == 0 { return Ok(()); }
            icmp += n_per * nc;
            led.diff_words(cuda, yo + (prim * slot0 * nc) as u64, src, n_per * nc, prim, cnt)
        };
        cmp(0, s.fpost[0], nz)?;
        for j in 4..17 { cmp((j - 3) * nz, s.fpost[j], nz)?; }
        for j in 0..10 { cmp(14 * nz + j, s.spost[j], 1)?; }
        for p in 0..4 { if p == 0 || s.flags[6] != 0 { cmp(14 * nz + 10 + p * nz, s.ppost[p], nz)?; } }
        Ok((icmp, Some(tok)))
    }

    /// tempo-once: the tangent of one launch by tempo_adw_tlf (one launch over every substep; per column one
    /// in-place node buffer of `cap` rows per lane, sized from the arena). Columns past the cap are rerun alone at
    /// the largest cap the arena gives them. None: a column does not fit even alone (the caller takes the
    /// segmented tangent). Returns (y-dot, identity differing words, identity compared words, launches).
    unsafe fn apply_tlf(&self, cuda: &Cuda, arena: &mut Arena, s: &Snap, vd: u64, rd: u64, identity: bool, led: &mut dev::Ledger)
                        -> Result<Option<(Option<u64>, usize, usize)>, String> {
        let tf = self.tlf.as_ref().ok_or("tempo_adw_tlf not loaded")?;
        if tf.dual { return self.apply_tlfd(cuda, arena, s, vd, rd, identity, led); }
        let (nz, nc, steps) = (s.nz, s.columns, s.steps);
        let nin = 18 * nz + 10;
        // the node-buffer variant (TLF=buffer, not the default) keeps its host column gather
        let v: Vec<f64> = get(cuda, vd, nin * nc)?;
        let prim = self.prim;
        let cols: Vec<i32> = (0..nc as i32).collect();
        let colsb = arena.take(4 * nc)?; put(cuda, colsb, &cols)?;
        let (st, wk, yo, oi, ii, cn) = (arena.take(tf.state * nc)?, arena.take(tf.work * nc)?, arena.take(prim * nin * nc)?,
                                        arena.take(4 * nin * nc)?, arena.take(4 * nin * nc)?, arena.take(4 * NCNT * 32 * nc)?);
        let (xd, yd, based, ccapd) = (arena.take(8 * nin * nc)?, arena.take(8 * nin * nc)?, arena.take(8 * nc)?, arena.take(4 * nc)?);
        let keep = arena.used;
        let mut out = vec![0f64; nin * nc];
        let (mut icmp, mut itok, mut launches) = (0usize, None, 0usize);
        // pass 0: every column, its cap (rows per lane of its one in-place node buffer) from the previous call.s need (+1/8 + 64 rows), or one uniform cap from the
        // arena when there is no history or it does not fit; pass 1: the columns that overflowed, at their exact need
        // (count slot 6 holds the largest node counter of any substep, counted on past the cap).
        let mut todo: Vec<usize> = (0..nc).collect();
        let mut need = self.tlf_need.borrow().clone();
        if need.len() != nc { need = vec![0; nc]; }
        let have_hist = need.iter().any(|&v| v > 0);
        const CAPMAX: usize = (1 << 26) - 1;
        for pass in 0..2 {
            let n = todo.len();
            if n == 0 { break; }
            arena.used = keep;
            let avail = arena.bytes.saturating_sub(arena.used + (64 << 20));
            let uni = (avail / (n * 32 * 8)).min(CAPMAX);
            let caps: Vec<usize> = if pass == 0 {
                let h: Vec<usize> = todo.iter().map(|&c| { let v = need[c] as usize; (v + v / 8 + 64).clamp(256, CAPMAX) }).collect();
                if have_hist && h.iter().sum::<usize>() * 32 * 8 <= avail { h } else { vec![uni; n] }
            } else {
                let h: Vec<usize> = todo.iter().map(|&c| (need[c] as usize + 64).min(CAPMAX)).collect();
                if h.iter().sum::<usize>() * 32 * 8 > avail { return Ok(None); }
                h
            };
            if caps.iter().any(|&c| c < 256) { return Ok(None); }
            let tan = arena.take(caps.iter().sum::<usize>() * 32 * 8)?;
            let mut bases = Vec::with_capacity(n); let mut acc = 0i64;
            for &c in &caps { bases.push(acc); acc += (32 * c) as i64; }
            put(cuda, based, &bases)?;
            put(cuda, ccapd, &caps.iter().map(|&c| c as i32).collect::<Vec<i32>>())?;
            if pass > 0 { let c: Vec<i32> = todo.iter().map(|&c| c as i32).collect(); put(cuda, colsb, &c)?; }
            let mut x = vec![0f64; nin * n];
            for q in 0..nin { for (t, &c) in todo.iter().enumerate() { x[q * n + t] = v[q * nc + c]; } }
            put(cuda, xd, &x)?;
            put(cuda, tf.g, &[0u64, 0, 0, 0, 0, 0, cn, 0, 0, (n * 32) as u64, 0, 0, based, ccapd, 0, 0, 0, 0])?;
            put(cuda, tf.gtan, &[tan])?;
            let args = vec![i32b(nz as i32), i32b(nc as i32), i32b(steps as i32), f32b(s.dt), u64b(s.ftab), u64b(s.stab), u64b(s.blob),
                 i32b(s.flags[0]), i32b(s.flags[1]), i32b(s.flags[3]), i32b(s.flags[4]), i32b(s.flags[5]), i32b(s.flags[6]),
                 u64b(colsb), i32b(n as i32), i32b(0), i32b(steps as i32), u64b(0), u64b(0), u64b(0), (nc as i64).to_le_bytes().to_vec(), i32b(0),
                 u64b(st), u64b(wk), u64b(yo), u64b(oi), u64b(ii), u64b(0), u64b(xd), u64b(yd)];
            launch(cuda, tf.f, grid(n), &args)?;
            launches += 1;
            let c: Vec<i32> = get(cuda, cn, NCNT * 32 * n)?;
            for (t, &col) in todo.iter().enumerate() { need[col] = (0..32).map(|l| c[NCNT * (t * 32 + l) + 6].max(0) as u32).max().unwrap_or(0); }
            let over: Vec<usize> = (0..n).filter(|&t| (0..32).any(|l| c[NCNT * (t * 32 + l) + 3] != 0)).collect();
            if pass == 0 && identity {
                // the augmented primal of every column against the forward's post-state, word for word (the primal
                // does not depend on the node cap; pass 0 holds every column in order)
                let (c, t) = self.identity_ledger(cuda, s, yo, true, led)?; icmp = c; itok = t;
            }
            let r: Vec<f64> = get(cuda, yd, nin * n)?;
            for (t, &col) in todo.iter().enumerate() { if !over.contains(&t) { for q in 0..nin { out[q * nc + col] = r[q * n + t]; } } }
            todo = over.iter().map(|&t| todo[t]).collect();
            if pass == 1 && !todo.is_empty() { *self.tlf_need.borrow_mut() = need; return Ok(None); }
            if pass == 0 && !todo.is_empty() { let mut z = self.tlf_stats.get(); z[1] += todo.len() as u64; self.tlf_stats.set(z); }
        }
        *self.tlf_need.borrow_mut() = need;
        put(cuda, rd, &out)?;
        let mut z = self.tlf_stats.get(); z[0] += launches as u64; self.tlf_stats.set(z);
        Ok(Some((itok, icmp, launches)))
    }

    /// Before the forward launch: resolve the launch, copy the inputs it reads into the arena, mirror tables.
    pub unsafe fn snapshot(&self, cuda: &Cuda, arena: &mut Arena, live: &HashMap<u64, Allocation>, args: &[Argument]) -> Result<Snap, String> {
        if args.len() != 21 { return Err(format!("tempo_warp_steps with {} arguments (21 expected)", args.len())); }
        let addr = |i: usize| -> Result<u64, String> {
            match &args[i] {
                Argument::Device(s) => Ok(live.get(&s.allocation).ok_or(format!("argument {i}: allocation {} not live", s.allocation))?.address + s.offset as u64),
                Argument::Bytes(b) if b.len() == 8 && b.iter().all(|&x| x == 0) => Ok(0),
                a => Err(format!("argument {i}: unexpected binding {a:?}")),
            }
        };
        let u32a = |i: usize| -> Result<u32, String> {
            match &args[i] { Argument::Bytes(b) if b.len() >= 4 => Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]])), a => Err(format!("argument {i}: not a scalar {a:?}")) }
        };
        let (nz, columns, steps) = (u32a(0)? as usize, u32a(1)? as usize, u32a(2)? as usize);
        let dt = f32::from_bits(u32a(3)?);
        let flags = [u32a(9)? as i32, u32a(10)? as i32, u32a(11)? as i32, u32a(12)? as i32, u32a(13)? as i32, u32a(14)? as i32, u32a(20)? as i32];
        if addr(5)? != 0 { return Err("bl present: the PBL subgrid-radii path is not in the TEMPO derivative".into()); }
        if flags[2] != 0 { return Err("ml_bl set: the PBL subgrid-radii path is not in the TEMPO derivative".into()); }
        let (ft, st, pt, blob) = (addr(4)?, addr(6)?, addr(7)?, addr(8)?);
        let fp: Vec<u64> = get(cuda, ft, 20)?; let sp: Vec<u64> = get(cuda, st, 12)?; let pp: Vec<u64> = get(cuda, pt, 4)?;
        let mut fsnap = [0u64; 18]; let mut ssnap = [0u64; 10];
        for j in 0..18 { if fp[j] != 0 { let a = arena.take(4 * nz * columns)?; dev::copy(cuda, a, fp[j], 4 * nz * columns)?; fsnap[j] = a; } }
        for j in 0..10 { if sp[j] != 0 { let a = arena.take(4 * columns)?; dev::copy(cuda, a, sp[j], 4 * columns)?; ssnap[j] = a; } }
        let ftab = arena.take(8 * 20)?; let stab = arena.take(8 * 12)?;
        let mut fv = vec![0u64; 20]; fv[..18].copy_from_slice(&fsnap);
        let mut sv = vec![0u64; 12]; sv[..10].copy_from_slice(&ssnap);
        put(cuda, ftab, &fv)?; put(cuda, stab, &sv)?;
        let mut fpost = [0u64; 18]; fpost.copy_from_slice(&fp[..18]);
        let mut spost = [0u64; 10]; spost.copy_from_slice(&sp[..10]);
        let mut ppost = [0u64; 4]; ppost.copy_from_slice(&pp[..4]);
        dev::sync(cuda)?;
        Ok(Snap { nz, columns, steps, dt, flags, blob, ftab, stab, fsnap, ssnap, fpost, spost, ppost })
    }

    unsafe fn put_tape(&self, cuda: &Cuda, v: [u64; 18]) -> Result<(), String> { put(cuda, self.g, &v) }
    fn lane_layout(&self) -> bool { self.lanecap.is_some() && std::env::var("AD_TEMPO_LANECAP").as_deref() != Ok("0") }

    /// After the forward launch: the launch dot on the snapshot (arena only).
    pub unsafe fn launch_dot(&self, cuda: &Cuda, arena: &mut Arena, s: &Snap, seed: u64, tol: f64, checks: bool) -> Result<Value, String> {
        let t_all = Instant::now();
        let (nz, nc, steps) = (s.nz, s.columns, s.steps);
        let nin = 18 * nz + 10;
        let prim = self.prim;
        // persistent buffers: checkpoints, masks, counts
        let ck = arena.take(prim * steps * nin * nc)?; let ckm = arena.take(steps * nin * nc)?;
        let mark = arena.take(4 * 3 * (steps + 1) * nc * 32)?;
        let keep = arena.used;
        let cols: Vec<i32> = (0..nc as i32).collect();
        let colsb = arena.take(4 * nc)?; put(cuda, colsb, &cols)?;
        let keep_cols = arena.used;
        let rec_args = |tile_cols: u64, n: usize, s0: usize, s1: usize, ckin: u64, ckout: u64, ckoff: usize, st: u64, wk: u64, yo: u64, oi: u64, ii: u64| -> Vec<Vec<u8>> {
            vec![i32b(nz as i32), i32b(nc as i32), i32b(steps as i32), f32b(s.dt), u64b(s.ftab), u64b(s.stab), u64b(s.blob),
                 i32b(s.flags[0]), i32b(s.flags[1]), i32b(s.flags[3]), i32b(s.flags[4]), i32b(s.flags[5]), i32b(s.flags[6]),
                 u64b(tile_cols), i32b(n as i32), i32b(s0 as i32), i32b(s1 as i32), u64b(ckin), u64b(ckout), u64b(ckm), (nc as i64).to_le_bytes().to_vec(), i32b(ckoff as i32),
                 u64b(st), u64b(wk), u64b(yo), u64b(oi), u64b(ii), u64b(0)]
        };
        // ---- checkpoint + count pass, every column
        let (st, wk, yo, oi, ii, cn) = (arena.take(self.state * nc)?, arena.take(self.work * nc)?, arena.take(prim * nin * nc)?,
                                        arena.take(4 * nin * nc)?, arena.take(4 * nin * nc)?, arena.take(4 * NCNT * 32 * nc)?);
        self.put_tape(cuda, [0, 0, 0, 0, 0, 0, cn, 0, 0, (nc * 32) as u64, 0, 0, 0, 0, 0, 0, mark, (steps + 1) as u64])?;
        let t0 = Instant::now();
        launch(cuda, self.rec, grid(nc), &rec_args(colsb, nc, 0, steps, 0, ck, 0, st, wk, yo, oi, ii))?;
        let ckpt_ms = t0.elapsed().as_secs_f64() * 1e3;
        let c0: Vec<i32> = get(cuda, cn, NCNT * 32 * nc)?;
        if (0..32 * nc).any(|t| c0[NCNT * t + 3] != 0) { return Ok(json!({"status":"ERROR","error":"count pass overflow"})); }
        // identity against the forward's post-state (f[0], f[4..16], s, profiles)
        let y: Vec<f32> = if prim == 4 { get(cuda, yo, nin * nc)? } else { get::<f64>(cuda, yo, nin * nc)?.iter().map(|&v| v as f32).collect() };
        let mut idiff = 0usize; let mut icmp = 0usize;
        let mut cmp = |slot0: usize, src: u64, n_per: usize| -> Result<(), String> {
            if src == 0 { return Ok(()); }
            let post: Vec<f32> = get(cuda, src, n_per * nc)?;
            for q in 0..n_per { for c in 0..nc { icmp += 1; if post[q * nc + c].to_bits() != y[(slot0 + q) * nc + c].to_bits() { idiff += 1; } } }
            Ok(())
        };
        cmp(0, s.fpost[0], nz)?;
        for j in 4..17 { cmp((j - 3) * nz, s.fpost[j], nz)?; }
        for j in 0..10 { cmp(14 * nz + j, s.spost[j], 1)?; }
        for p in 0..4 { if p == 0 || s.flags[6] != 0 { cmp(14 * nz + 10 + p * nz, s.ppost[p], nz)?; } }
        let mk: Vec<i32> = get(cuda, mark, 3 * (steps + 1) * nc * 32)?;
        let t_all32 = nc * 32;
        let mkv = |i: usize, t: usize, f: usize| mk[(i * t_all32 + t) * 3 + f];
        let mut seg_cap = vec![vec![0usize; nc]; steps]; let mut pmax = 0usize; let mut rmax = 0usize; let mut nodes = 0u64;
        let lanes = self.lane_layout();
        // tempo-adjoint-v2 lane layout: per segment and column the block's rows (sum of lane rows + 64) and the lane caps
        let mut lrows = vec![vec![0usize; nc]; steps]; let mut lcaps = vec![vec![0i32; nc * 32]; if lanes { steps } else { 0 }];
        for sg in 0..steps { for c in 0..nc { for l in 0..32 {
            let t = c * 32 + l; let dn = (mkv(sg + 1, t, 0) - mkv(sg, t, 0)) as usize; nodes += dn as u64;
            seg_cap[sg][c] = seg_cap[sg][c].max(dn + 64);
            lrows[sg][c] += dn + 64; if lanes { lcaps[sg][t] = (dn + 64) as i32; }
            pmax = pmax.max((mkv(sg + 1, t, 1) - mkv(sg, t, 1)) as usize); rmax = rmax.max(mkv(sg + 1, t, 2) as usize);
        } } }
        let (pcap, obcap) = (pmax + 4, rmax + 8);
        // inputs (for direction scales) from the snapshot
        let mut xin = vec![0f32; nin * nc];
        for j in 0..18 { if s.fsnap[j] != 0 { let v: Vec<f32> = get(cuda, s.fsnap[j], nz * nc)?; for k in 0..nz { for c in 0..nc { xin[(j * nz + k) * nc + c] = v[k * nc + c]; } } } }
        for j in 0..10 { if s.ssnap[j] != 0 { let v: Vec<f32> = get(cuda, s.ssnap[j], nc)?; for c in 0..nc { xin[(18 * nz + j) * nc + c] = v[c]; } } }
        arena.used = keep_cols; // the count pass's work buffers are spent
        // ---- tiles by the arena that is left
        let per_col_fixed = self.state + self.work + prim * nin + 3 * 4 * nin + 4 * NCNT * 32 + 12 + 4 * pcap * 32 + 12 * obcap * 32 + 4 + 2 * 8 * nin + 2 * 8 * nin + 4096;
        let avail = arena.bytes.saturating_sub(arena.used + (64 << 20));
        let mut tiles: Vec<(usize, usize)> = vec![];
        { let mut a0 = 0usize; let mut acc = vec![0usize; steps]; let mut fixed = 0usize;
          for c in 0..nc {
              let sl = |sg: usize| if lanes { lrows[sg][c] } else { seg_cap[sg][c] * 32 };
              let need = acc.iter().zip(0..steps).map(|(a, sg)| a + sl(sg) * 32).max().unwrap() + fixed + per_col_fixed;
              if need > avail { if c == a0 { return Ok(json!({"status":"ERROR","error":format!("TEMPO column {c} alone needs {need} bytes, arena has {avail}")})); }
                                tiles.push((a0, c - a0)); a0 = c; acc = vec![0; steps]; fixed = 0; }
              for sg in 0..steps { acc[sg] += sl(sg) * 32; } fixed += per_col_fixed; }
          tiles.push((a0, nc - a0)); }
        let mut g1 = Rng::new(seed); let mut g2 = Rng::new(seed ^ 0x5eed);
        let floor = [1.0, 1.0, 0.1, 1.0, 1e-6, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1e-9, 1e-3];
        let (mut lhs, mut rhs, mut worst_col) = (0f64, 0f64, 0f64);
        // ||TL x|| / ||x||: plain (raw units, x scaled per input row as below) and relative (each slot divided by
        // |primal| + floor at its own end: inputs at the launch start, outputs at the launch end).
        let (mut xx, mut yy, mut xr, mut yr) = (0f64, 0f64, 0f64, 0f64);
        // per output group (f0 theta, f4..f16, s0..s9, p0..p3): plain and relative sums of squares, and the slot whose
        // relative tangent is largest (|y-dot| / (|y| + floor), with y and the floor).
        let group = |q: usize| -> String { if q < nz { "f0".into() } else if q < AD_NF_OUT * nz { format!("f{}", q / nz + 3) }
            else if q < AD_NF_OUT * nz + 10 { format!("s{}", q - AD_NF_OUT * nz) } else { format!("p{}", (q - AD_NF_OUT * nz - 10) / nz) } };
        let mut by: std::collections::BTreeMap<String, (f64, f64, f64, usize, usize, f64)> = Default::default();
        let (mut zero_nonzero, mut det_ok) = (0usize, true);
        let (mut tl_ms, mut ad_ms) = (0f64, 0f64);
        for &(c0, n) in &tiles {
            let tile_mark = arena.used;
            let mut seg_base = vec![]; let mut seg_ccap = vec![]; let mut slots = 0usize;
            for sg in 0..steps {
                let mut b = vec![0i64; n]; let mut acc = 0i64;
                for t in 0..n { b[t] = acc; acc += (if lanes { lrows[sg][c0 + t] } else { seg_cap[sg][c0 + t] * 32 }) as i64; }
                slots = slots.max(acc as usize); seg_base.push(b);
                seg_ccap.push(if lanes { lcaps[sg][c0 * 32..(c0 + n) * 32].to_vec() } else { (0..n).map(|t| seg_cap[sg][c0 + t] as i32).collect::<Vec<i32>>() });
            }
            let tc = colsb + 4 * c0 as u64;
            let (tst, twk, tyo, toi, tii, tcn) = (arena.take(self.state * n)?, arena.take(self.work * n)?, arena.take(prim * nin * n)?, arena.take(4 * nin * n)?, arena.take(4 * nin * n)?, arena.take(4 * NCNT * 32 * n)?);
            let (ta, tb, tpa, tpb, tsc) = (arena.take(4 * slots)?, arena.take(4 * slots)?, arena.take(8 * slots)?, arena.take(8 * slots)?, arena.take(8 * slots)?);
            let (tbase, tccap, tph) = (arena.take(8 * n)?, arena.take(4 * n * if lanes { 32 } else { 1 })?, arena.take(4 * pcap * 32 * n)?);
            let (obid, obv, flt) = (arena.take(4 * obcap * 32 * n)?, arena.take(8 * obcap * 32 * n)?, arena.take(4 * n)?);
            let (va, vb) = (arena.take(8 * nin * n)?, arena.take(8 * nin * n)?);
            let record = |sg: usize| -> Result<(), String> {
                put(cuda, tbase, &seg_base[sg])?; put(cuda, tccap, &seg_ccap[sg])?;
                self.put_tape(cuda, [ta, tb, tpa, tpb, 0, 0, tcn, 0, 0, (n * 32) as u64, 0, 0, tbase, tccap, tph, pcap as u64, 0, 0])?;
                let ckin = if sg > 0 { ck + (prim * sg * nin * nc) as u64 } else { 0 };
                launch(cuda, self.rec, grid(n), &rec_args(tc, n, sg, sg + 1, ckin, 0, c0, tst, twk, tyo, toi, tii))?;
                let c: Vec<i32> = get(cuda, tcn, NCNT * 32 * n)?;
                if (0..32 * n).any(|t| c[NCNT * t + 3] != 0) { return Err(format!("TEMPO segment {sg}: tape/phase capacity overflow")); }
                Ok(())
            };
            let run_tl = |x: &[f64]| -> Result<Vec<f64>, String> {
                put(cuda, va, x)?; let (mut a, mut b) = (va, vb);
                for sg in 0..steps {
                    record(sg)?;
                    launch(cuda, self.tl, grid(n), &[i32b(n as i32), i32b(nin as i32), i32b(nin as i32), u64b(a), u64b(tsc), u64b(tii), u64b(toi), u64b(b)])?;
                    std::mem::swap(&mut a, &mut b);
                }
                get(cuda, a, nin * n)
            };
            let run_adj = |yb: &[f64]| -> Result<Vec<f64>, String> {
                put(cuda, va, yb)?; let (mut a, mut b) = (va, vb);
                for sg in (0..steps).rev() {
                    record(sg)?;
                    launch(cuda, self.adj, grid(n), &[i32b(n as i32), i32b(nin as i32), i32b(nin as i32), u64b(a), u64b(tsc), u64b(tii), u64b(toi), u64b(b),
                                                      u64b(obid), u64b(obv), i32b(obcap as i32), u64b(flt)])?;
                    let f: Vec<i32> = get(cuda, flt, n)?;
                    if let Some(c) = f.iter().position(|&v| v != 0) { return Err(format!("TEMPO adjoint fault bits {} at column {}", f[c], c0 + c)); }
                    std::mem::swap(&mut a, &mut b);
                }
                get(cuda, a, nin * n)
            };
            let mut xd = vec![0f64; nin * n]; let mut yb = vec![0f64; nin * n];
            for t in 0..n { for q in 0..nin {
                let fl = if q < 18 * nz { floor[q / nz] } else { 1e-3 };
                xd[q * n + t] = g1.normal() * ((xin[q * nc + c0 + t] as f64).abs() + fl);
                yb[q * n + t] = g2.normal();
            } }
            let t1 = Instant::now(); let yd = run_tl(&xd).map_err(|e| e.to_string());
            let yd = match yd { Ok(v) => v, Err(e) => return Ok(json!({"status":"ERROR","error":e})) };
            tl_ms += t1.elapsed().as_secs_f64() * 1e3;
            for t in 0..n { for q in 0..nin {
                let (xv, yv) = (xd[q * n + t], yd[q * n + t]);
                let fi = if q < 18 * nz { floor[q / nz] } else { 1e-3 };
                let fo = if q < AD_NF_OUT * nz { floor[if q < nz { 0 } else { q / nz + 3 }] } else if q < AD_NF_OUT * nz + 10 { 1e-3 } else { 1.0 };
                xx += xv * xv; yy += yv * yv;
                let sx = (xin[q * nc + c0 + t] as f64).abs() + fi; let sy = (y[q * nc + c0 + t] as f64).abs() + fo;
                xr += (xv / sx) * (xv / sx); yr += (yv / sy) * (yv / sy);
                let g = by.entry(group(q)).or_insert((0.0, 0.0, 0.0, 0, 0, 0.0));
                g.0 += yv * yv; g.1 += (yv / sy) * (yv / sy);
                if (yv / sy).abs() > g.2 { g.2 = (yv / sy).abs(); g.3 = if q < AD_NF_OUT * nz { q % nz + 1 } else { 0 }; g.4 = c0 + t; g.5 = y[q * nc + c0 + t] as f64; }
            } }
            let t2 = Instant::now(); let xb = match run_adj(&yb) { Ok(v) => v, Err(e) => return Ok(json!({"status":"ERROR","error":e})) };
            ad_ms += t2.elapsed().as_secs_f64() * 1e3;
            for t in 0..n {
                let (mut lc, mut rc) = (0f64, 0f64);
                for q in 0..nin { lc += yd[q * n + t] * yb[q * n + t]; rc += xd[q * n + t] * xb[q * n + t]; }
                worst_col = worst_col.max((lc - rc).abs() / lc.abs().max(rc.abs()).max(1e-300)); lhs += lc; rhs += rc;
            }
            if checks {
                let xb2 = run_adj(&yb)?;
                if xb2.iter().zip(&xb).any(|(p, q)| p.to_bits() != q.to_bits()) { det_ok = false; }
                let yz = run_tl(&vec![0f64; nin * n])?; let xz = run_adj(&vec![0f64; nin * n])?;
                zero_nonzero += yz.iter().chain(xz.iter()).filter(|v| **v != 0.0 || v.is_nan()).count();
            }
            arena.used = tile_mark;
        }
        arena.used = keep;
        let rel = (lhs - rhs).abs() / lhs.abs().max(rhs.abs()).max(1e-300);
        let ok = idiff == 0 && rel <= tol && rel.is_finite() && (!checks || (zero_nonzero == 0 && det_ok));
        Ok(json!({"status": if ok { "PASS" } else { "FAIL" }, "rel": rel, "lhs": lhs, "rhs": rhs, "worst_column_rel": worst_col, "tol": tol,
                  "identity_differing_words": idiff, "identity_compared_words": icmp, "checked": checks,
                  "zero_direction_nonzero_words": if checks { json!(zero_nonzero) } else { Value::Null }, "reverse_repeat_bitwise": if checks { json!(det_ok) } else { Value::Null },
                  "tl_ms": tl_ms, "ad_ms": ad_ms + ckpt_ms, "checkpoint_count_pass_ms": ckpt_ms, "wall_ms": t_all.elapsed().as_secs_f64() * 1e3,
                  "columns": nc, "substeps": steps, "tiles": tiles.len(), "tape_nodes": nodes,
                  "tl_gain": (yy / xx.max(1e-300)).sqrt(), "tl_gain_relative": (yr / xr.max(1e-300)).sqrt(),
                  "tl_gain_by_output": by.iter().map(|(k, g)| (k.clone(), json!({"plain": (g.0 / xx.max(1e-300)).sqrt(), "relative": (g.1 / xr.max(1e-300)).sqrt(),
                      "worst_relative_word": {"ratio": g.2, "level": g.3, "column": g.4, "primal": g.5}}))).collect::<serde_json::Map<String, Value>>(),
                  "derivative": "tempo_adw (warp shape, per-substep checkpoints, float64 tape)"}))
    }

    /// Window binding (ad/held-tempo_warp_steps): the whole-launch tangent or adjoint of one tempo_warp_steps launch at
    /// the snapshot's primal, the same derivative launch_dot checks. `v` and the result are [slot][column] (nin x nc).
    /// adjoint = false: v = x-dot in the input layout (f[0..17] at j*nz, s[0..9] at 18*nz + j), returns y-dot in the
    /// output layout (f[0] theta at 0, f[4..16] at (j-3)*nz, s[0..9] at 14*nz + j, profiles 0..3 at 14*nz + 10 + p*nz);
    /// adjoint = true: v = y-bar in the output layout, returns x-bar in the input layout. `identity`: also compare the
    /// augmented primal with the live post-state word for word (only right after the forward launch). Returns
    /// (result, identity differing words, identity compared words, tiles). The arena is left as it was found.
    /// readbacks-device: the input vector `vd` and the result `rd` ([18 nz + 10][columns] float64 each) live on the
    /// device; the column tiles are cut from and put back into them by a device kernel, and the count-pass overflow,
    /// the per-segment tape overflow and the adjoint fault bits go to the window's step ledger (read once per step),
    /// as does the identity word count (its token is returned). The only reads left are the count pass's segment
    /// marks, which size the tiles on the host.
    pub unsafe fn apply(&self, cuda: &Cuda, arena: &mut Arena, s: &Snap, vd: u64, rd: u64, adjoint: bool, identity: bool,
                        led: &mut dev::Ledger) -> Result<(Option<u64>, usize, usize), String> {
        let start = arena.used;
        if !adjoint && self.tlf.is_some() {
            let t0 = Instant::now();
            let r = self.apply_tlf(cuda, arena, s, vd, rd, identity, led);
            arena.used = start;
            match r {
                Ok(Some(res)) => {
                    let k_ms = t0.elapsed().as_secs_f64() * 1e3;
                    if std::env::var("AD_TEMPO_TLF_CHECK").ok().as_deref() == Some("1") {
                        // check mode (receipts only): the segmented tangent on the same snapshot, word for word (host
                        // compare of the two device results; the one-launch result is put back in rd afterwards)
                        let nv = (18 * s.nz + 10) * s.columns;
                        let new: Vec<f64> = get(cuda, rd, nv)?;
                        let old = self.apply_in(cuda, arena, s, vd, rd, false, false, led);
                        arena.used = start;
                        old?;
                        let o: Vec<f64> = get(cuda, rd, nv)?;
                        put(cuda, rd, &new)?;
                        let bad = o.iter().zip(&new).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                        let mut z = self.tlf_stats.get(); z[3] += bad as u64; self.tlf_stats.set(z);
                        eprintln!("{}", json!({"event":"window_tempo_tlf_check","words":o.len(),"differing_words":bad}));
                    }
                    self.parts.set([0.0, k_ms, 0.0, 0.0]);
                    return Ok(res);
                }
                Ok(None) => { let mut z = self.tlf_stats.get(); z[2] += 1; self.tlf_stats.set(z);
                              eprintln!("{}", json!({"event":"window_tempo_tlf_fallback","reason":"a column does not fit the arena alone"})); }
                Err(e) => return Err(e),
            }
        }
        let r = self.apply_in(cuda, arena, s, vd, rd, adjoint, identity, led);
        arena.used = start;
        r
    }
    #[allow(clippy::too_many_arguments)]
    unsafe fn apply_in(&self, cuda: &Cuda, arena: &mut Arena, s: &Snap, vd: u64, rd: u64, adjoint: bool, identity: bool,
                       led: &mut dev::Ledger) -> Result<(Option<u64>, usize, usize), String> {
        let (nz, nc, steps) = (s.nz, s.columns, s.steps);
        let t_in = Instant::now();
        let (rec_ms, sw_ms) = (std::cell::Cell::new(0f64), std::cell::Cell::new(0f64));
        let nin = 18 * nz + 10;
        let prim = self.prim;
        let ck = arena.take(prim * steps * nin * nc)?; let ckm = arena.take(steps * nin * nc)?;
        let mark = arena.take(4 * 3 * (steps + 1) * nc * 32)?;
        let cols: Vec<i32> = (0..nc as i32).collect();
        let colsb = arena.take(4 * nc)?; put(cuda, colsb, &cols)?;
        let keep_cols = arena.used;
        // ckoff < 0 (tempo-adjoint-v2 sorted tiles, images with tempo_adw_ptile): each column reads its own checkpoint column
        let rec_args = |tile_cols: u64, n: usize, s0: usize, s1: usize, ckin: u64, ckout: u64, ckoff: i64, st: u64, wk: u64, yo: u64, oi: u64, ii: u64| -> Vec<Vec<u8>> {
            vec![i32b(nz as i32), i32b(nc as i32), i32b(steps as i32), f32b(s.dt), u64b(s.ftab), u64b(s.stab), u64b(s.blob),
                 i32b(s.flags[0]), i32b(s.flags[1]), i32b(s.flags[3]), i32b(s.flags[4]), i32b(s.flags[5]), i32b(s.flags[6]),
                 u64b(tile_cols), i32b(n as i32), i32b(s0 as i32), i32b(s1 as i32), u64b(ckin), u64b(ckout), u64b(ckm), (nc as i64).to_le_bytes().to_vec(), i32b(ckoff as i32),
                 u64b(st), u64b(wk), u64b(yo), u64b(oi), u64b(ii), u64b(0)]
        };
        // checkpoint + count pass, every column (as launch_dot)
        let (st, wk, yo, oi, ii, cn) = (arena.take(self.state * nc)?, arena.take(self.work * nc)?, arena.take(prim * nin * nc)?,
                                        arena.take(4 * nin * nc)?, arena.take(4 * nin * nc)?, arena.take(4 * NCNT * 32 * nc)?);
        self.put_tape(cuda, [0, 0, 0, 0, 0, 0, cn, 0, 0, (nc * 32) as u64, 0, 0, 0, 0, 0, 0, mark, (steps + 1) as u64])?;
        let t_cp = Instant::now();
        launch_ns(cuda, self.rec, grid(nc), &rec_args(colsb, nc, 0, steps, 0, ck, 0, st, wk, yo, oi, ii))?;
        led.any_lane(cuda, cn, 32 * nc, NCNT, 3, "tempo_adw: count pass overflow")?;
        let mut icmp = 0usize;
        let mut itok = None;
        if identity {
            let (cnt, tok) = led.reserve(cuda, 8, dev::LedgerItem::Count)?;
            itok = Some(tok);
            let mut cmp = |slot0: usize, src: u64, n_per: usize| -> Result<(), String> {
                if src == 0 { return Ok(()); }
                icmp += n_per * nc;
                led.diff_words(cuda, yo + (prim * slot0 * nc) as u64, src, n_per * nc, prim, cnt)
            };
            cmp(0, s.fpost[0], nz)?;
            for j in 4..17 { cmp((j - 3) * nz, s.fpost[j], nz)?; }
            for j in 0..10 { cmp(14 * nz + j, s.spost[j], 1)?; }
            for p in 0..4 { if p == 0 || s.flags[6] != 0 { cmp(14 * nz + 10 + p * nz, s.ppost[p], nz)?; } }
        }
        // round 4: the marks are reduced on the device (per segment and column the largest lane node delta + 64; the
        // largest phase delta and record mark), so the reverse reads steps x columns words instead of 3 x (steps + 1)
        // x 32 x columns ints and scans nothing on the host. AD_TEMPO_MARKS=host keeps the old host scan; =check runs both and
        // refuses on any difference (receipts only).
        let marks_env = std::env::var("AD_TEMPO_MARKS").ok();
        let host_marks = || -> Result<(Vec<Vec<usize>>, usize, usize), String> {
            let mk: Vec<i32> = get(cuda, mark, 3 * (steps + 1) * nc * 32)?;
            let t_all32 = nc * 32;
            let mkv = |i: usize, t: usize, f: usize| mk[(i * t_all32 + t) * 3 + f];
            let mut seg_cap = vec![vec![0usize; nc]; steps]; let mut pmax = 0usize; let mut rmax = 0usize;
            for sg in 0..steps { for c in 0..nc { for l in 0..32 {
                let t = c * 32 + l; let dn = (mkv(sg + 1, t, 0) - mkv(sg, t, 0)) as usize;
                seg_cap[sg][c] = seg_cap[sg][c].max(dn.wrapping_add(64));
                pmax = pmax.max((mkv(sg + 1, t, 1) - mkv(sg, t, 1)) as usize); rmax = rmax.max(mkv(sg + 1, t, 2) as usize);
            } } }
            Ok((seg_cap, pmax, rmax))
        };
        let (seg_cap, pmax, rmax) = if marks_env.as_deref() == Some("host") { host_marks()? } else {
            let (capd, prd) = (arena.take(8 * steps * nc)?, arena.take(16)?);
            let (cv, pm, rm) = led.tempo_marks(cuda, mark, steps, nc, capd, prd)?;
            let dv = ((0..steps).map(|sg| cv[sg * nc..(sg + 1) * nc].iter().map(|&v| v as usize).collect()).collect::<Vec<Vec<usize>>>(), pm as usize, rm as usize);
            if marks_env.as_deref() == Some("check") {
                // receipts only: the host scan of the same marks, value for value
                let hv = host_marks()?;
                let bad = (0..steps).map(|sg| (0..nc).filter(|&c| hv.0[sg][c] != dv.0[sg][c]).count()).sum::<usize>();
                eprintln!("{}", json!({"event":"window_tempo_marks_check","caps":steps * nc,"differing_caps":bad,
                                       "pmax":[hv.1, dv.1],"rmax":[hv.2, dv.2],"equal": bad == 0 && hv.1 == dv.1 && hv.2 == dv.2}));
                if bad != 0 || hv.1 != dv.1 || hv.2 != dv.2 { return Err(format!("TEMPO marks: device reduction differs from the host scan ({bad} caps)")); }
            }
            dv
        };
        // tempo-adjoint-v2 lane layout: per segment and column the block's rows (tempo_adw_lsum, one download)
        let lanes = self.lane_layout();
        let lrows: Vec<usize> = if let (true, Some((_, fsum))) = (lanes, self.lanecap) {
            let d = arena.take(8 * steps * nc)?;
            launch_ns(cuda, fsum, ((steps * nc) as u32).div_ceil(BLOCK), &[u64b(mark), (steps as i64).to_le_bytes().to_vec(), (nc as i64).to_le_bytes().to_vec(), u64b(d)])?;
            let v: Vec<u64> = get(cuda, d, steps * nc)?; v.iter().map(|&x| x as usize).collect()
        } else { vec![] };
        // slots of segment sg's block for column c: the lanes' rows (lane layout) or 32 x the largest lane's
        let slots_of = |sg: usize, c: usize| -> usize { if lanes { lrows[sg * nc + c] } else { seg_cap[sg][c] * 32 } };
        let count_ms = t_cp.elapsed().as_secs_f64() * 1e3;
        let (pcap, obcap) = (pmax + 4, rmax + 8);
        // tempo-mynn-kernels: tape receipt. Compact images (tools/once/mk_compact.py) write c[6] = non-leaf pushes and
        // c[7] = binary pushes in the count pass (older images write 0); slots = what the lane-major tiles reserve
        // (each lane takes its column's largest counter + 64). Bytes: 24 per node stored before, 12 unary / 24 binary
        // compact; footprint 32 per slot (a, b, pa, pb, sweep vector). Leaves (<= one per active input slot) are not counted.
        // adjoint-speed-4: readbacks-device moved the count-pass counters onto the device ledger (no host copy), so this
        // receipt reads them back only when AD_TEMPO_TAPE_RECEIPT=1 (one blocking NCNT x 32 x columns int32 download per
        // TEMPO AD launch; off by default so a timed run carries no extra readback).
        if adjoint && std::env::var("AD_TEMPO_TAPE_RECEIPT").as_deref() == Ok("1") {
            let c0: Vec<i32> = get(cuda, cn, NCNT * 32 * nc)?;
            let (mut u, mut v) = (0u64, 0u64);
            for t in 0..32 * nc { u += c0[NCNT * t + 6].max(0) as u64; v += c0[NCNT * t + 7].max(0) as u64; }
            let slots: u64 = (0..steps).map(|sg| (0..nc).map(|c| slots_of(sg, c) as u64).sum::<u64>()).sum();
            let ncf = nc.max(1) as f64;
            eprintln!("{}", json!({"event":"tempo_tape","columns":nc,"substeps":steps,"nonleaf_nodes":u,"binary_nodes":v,"slots":slots,
                "bytes_written_per_cell_step_24":(u * 24) as f64 / ncf, "bytes_written_per_cell_step_compact":((u - v) * 12 + v * 24) as f64 / ncf,
                "slot_bytes_per_cell_step":(slots * 32) as f64 / ncf, "nodes_per_cell_step":u as f64 / ncf, "binary_share":if u > 0 { v as f64 / u as f64 } else { 0.0 }}));
        }
        arena.used = keep_cols; // the count pass's work buffers are spent
        // tempo-adjoint-v2: column order of the tiles. A tile's launch lasts as long as its longest column chain, so with
        // the image's gather (tempo_adw_ptile) the columns go in order of their rows over the launch, largest first (ties
        // by column), and each tile holds columns of similar length. Position i of the order is column perm[i].
        let sorted = self.ptile.is_some() && std::env::var("AD_TEMPO_SORT").as_deref() != Ok("0");
        let rows_of = |c: usize| -> usize { (0..steps).map(|sg| slots_of(sg, c)).sum() };
        let mut perm: Vec<usize> = (0..nc).collect();
        if sorted { perm.sort_by(|&x, &y| rows_of(y).cmp(&rows_of(x)).then(x.cmp(&y))); }
        let permb = if sorted { let b = arena.take(4 * nc)?; put(cuda, b, &perm.iter().map(|&c| c as i32).collect::<Vec<i32>>())?; b } else { colsb };
        let per_col_fixed = self.state + self.work + prim * nin + 3 * 4 * nin + 4 * NCNT * 32 + 12 + 4 * pcap * 32 + 12 * obcap * 32 + 4 + 2 * 8 * nin + 2 * 8 * nin + 4096;
        let avail = arena.bytes.saturating_sub(arena.used + (64 << 20));
        let mut tiles: Vec<(usize, usize)> = vec![];
        { let mut a0 = 0usize; let mut acc = vec![0usize; steps]; let mut fixed = 0usize;
          for i in 0..nc {
              let c = perm[i];
              let need = acc.iter().zip(0..steps).map(|(a, sg)| a + slots_of(sg, c) * 32).max().unwrap() + fixed + per_col_fixed;
              if need > avail { if i == a0 { return Err(format!("TEMPO column {c} alone needs {need} bytes, arena has {avail}")); }
                                tiles.push((a0, i - a0)); a0 = i; acc = vec![0; steps]; fixed = 0; }
              for sg in 0..steps { acc[sg] += slots_of(sg, c) * 32; } fixed += per_col_fixed; }
          tiles.push((a0, nc - a0)); }
        // receipt (host data only, no device wait): per tile [columns, largest column rows, mean column rows] over the launch
        { let tr: Vec<Value> = tiles.iter().map(|&(c0, n)| { let r: Vec<usize> = (c0..c0 + n).map(|i| rows_of(perm[i])).collect();
              json!([n, r.iter().copied().max().unwrap_or(0), (r.iter().sum::<usize>() as f64 / n.max(1) as f64).round()]) }).collect();
          eprintln!("{}", json!({"event":"tempo_tiles","pass":if adjoint { "ad" } else { "tl" },"sorted":sorted,"tiles":tr})); }
        for &(c0, n) in &tiles {
            let tile_mark = arena.used;
            let mut seg_base = vec![]; let mut seg_ccap = vec![]; let mut slots = 0usize;
            for sg in 0..steps {
                let mut b = vec![0i64; n]; let mut acc = 0i64;
                for t in 0..n { b[t] = acc; acc += slots_of(sg, perm[c0 + t]) as i64; }
                slots = slots.max(acc as usize); seg_base.push(b);
                if !lanes { seg_ccap.push((0..n).map(|t| seg_cap[sg][perm[c0 + t]] as i32).collect::<Vec<i32>>()); }
            }
            let tc = permb + 4 * c0 as u64;
            let ckoff: i64 = if sorted { -1 } else { c0 as i64 };
            // the tile's [nin][n] vectors from / into the [nin][nc] vectors: by column (sorted) or a contiguous run (as before)
            let tile_io = |led: &dev::Ledger, src: u64, dst: u64, back: bool| -> Result<(), String> {
                if let (true, Some(f)) = (sorted, self.ptile) {
                    let total = nin * n;
                    launch_ns(cuda, f, (total as u32).div_ceil(BLOCK), &[u64b(src), u64b(dst), (nin as i64).to_le_bytes().to_vec(), (nc as i64).to_le_bytes().to_vec(),
                                                                          u64b(tc), (n as i64).to_le_bytes().to_vec(), i32b(back as i32)])
                } else { led.tile(cuda, src, dst, nin, nc, c0, n, back) }
            };
            let (tst, twk, tyo, toi, tii, tcn) = (arena.take(self.state * n)?, arena.take(self.work * n)?, arena.take(prim * nin * n)?, arena.take(4 * nin * n)?, arena.take(4 * nin * n)?, arena.take(4 * NCNT * 32 * n)?);
            let (ta, tb, tpa, tpb, tsc) = (arena.take(4 * slots)?, arena.take(4 * slots)?, arena.take(8 * slots)?, arena.take(8 * slots)?, arena.take(8 * slots)?);
            let (tbase, tccap, tph) = (arena.take(8 * n)?, arena.take(4 * n * if lanes { 32 } else { 1 })?, arena.take(4 * pcap * 32 * n)?);
            let (obid, obv, flt) = (arena.take(4 * obcap * 32 * n)?, arena.take(8 * obcap * 32 * n)?, arena.take(4 * n)?);
            let (va, vb) = (arena.take(8 * nin * n)?, arena.take(8 * nin * n)?);
            // The per-segment tape bases and capacities are host data (from the count marks), uploaded per segment as before;
            // the overflow counters are checked on the device into the step ledger (was a 32 x NCNT x columns download each).
            let record = |sg: usize, led: &mut dev::Ledger| -> Result<(), String> {
                put(cuda, tbase, &seg_base[sg])?;
                if let (true, Some((fcap, _))) = (lanes, self.lanecap) {
                    // the tile's lane capacities from the count-pass marks, on the device (tile column t is column tc[t])
                    launch_ns(cuda, fcap, ((n * 32) as u32).div_ceil(BLOCK), &[u64b(mark), ((nc * 32) as i64).to_le_bytes().to_vec(), i32b(sg as i32), u64b(tc), i32b(n as i32), u64b(tccap)])?;
                } else { put(cuda, tccap, &seg_ccap[sg])?; }
                self.put_tape(cuda, [ta, tb, tpa, tpb, 0, 0, tcn, 0, 0, (n * 32) as u64, 0, 0, tbase, tccap, tph, pcap as u64, 0, 0])?;
                let ckin = if sg > 0 { ck + (prim * sg * nin * nc) as u64 } else { 0 };
                launch_ns(cuda, self.rec, grid(n), &rec_args(tc, n, sg, sg + 1, ckin, 0, ckoff, tst, twk, tyo, toi, tii))?;
                led.any_lane(cuda, tcn, 32 * n, NCNT, 3, &format!("TEMPO segment {sg}: tape/phase capacity overflow"))
            };
            tile_io(led, vd, va, false)?;
            let (mut a, mut b) = (va, vb);
            if !adjoint {
                for sg in 0..steps {
                    record(sg, led)?;
                    launch_ns(cuda, self.tl, grid(n), &[i32b(n as i32), i32b(nin as i32), i32b(nin as i32), u64b(a), u64b(tsc), u64b(tii), u64b(toi), u64b(b)])?;
                    std::mem::swap(&mut a, &mut b);
                }
            } else {
                for sg in (0..steps).rev() {
                    record(sg, led)?;
                    launch_ns(cuda, self.adj, grid(n), &[i32b(n as i32), i32b(nin as i32), i32b(nin as i32), u64b(a), u64b(tsc), u64b(tii), u64b(toi), u64b(b),
                                                      u64b(obid), u64b(obv), i32b(obcap as i32), u64b(flt)])?;
                    led.first_nonzero(cuda, flt, n, c0, "TEMPO adjoint fault bits")?;
                    std::mem::swap(&mut a, &mut b);
                }
            }
            tile_io(led, a, rd, true)?;
            arena.used = tile_mark;
        }
        let all = t_in.elapsed().as_secs_f64() * 1e3;
        // host-measured parts: the count pass waits at its mark download; the records and sweeps no longer wait per
        // launch (readbacks-device), so parts[1] is their enqueue time and the GPU time lands in the window's step sync
        self.parts.set([count_ms, all - count_ms, 0.0, 0.0]);
        let _ = (&rec_ms, &sw_ms);
        Ok((itok, icmp, tiles.len()))
    }
}

// ------------------------------------------------------------------------------------------------------------
// tempo_precip and tempo_radar run one thread per column in the forecast as well, so their derivative is the P5
// per-thread record entries (tempo_ad_precip_record / tempo_ad_radar_record of the P5 module, serial tape: leaf
// of input slot q is node q + 1) with its tangent and adjoint sweeps. Enabled by AD_DOT_TEMPO_SERIAL_CUBIN.
pub struct TempoSmall { precip: *mut c_void, radar: *mut c_void, tl: *mut c_void, adj: *mut c_void, g: u64, prim: usize, pub image_sha256: String }
/// A launch's pre-launch copy. `inputs` / `post`: (live address, words per column slot, first slot) of the arrays the
/// entry reads / writes (tempo_precip: every s[0..9] and d[0..12] slot is read and written in place, f[3], f[4],
/// f[18] are read only). `copies`: (arena copy, words, first float64 word of the input vector) of the inputs read.
pub struct SmallSnap { kind: u8, args: Vec<Vec<u8>>, nz: usize, nc: usize, nin: usize, nout: usize, post: Vec<(u64, usize, usize)>, counter_post: u64, full: bool,
                       inputs: Vec<(u64, usize, usize)>, copies: Vec<(u64, usize, usize)>,
                       /// Columns where sr (d[7], output slot 17) is frozen (see SR_SLOT).
                       sr_frozen: Vec<usize> }

/// sr, the step's frozen-precipitation fraction, is d[7] = (s[3] + s[5]) / (s[1] + 1e-12) (tempo_v303.py:4468,
/// RRFS mpas_atmphys_driver_microphysics.F). Where s[1] + 1e-12f rounds to the 1e-12f floor (no precipitation
/// this step) the ratio is 0/0 held at 0 by the floor, and its exact derivative is 1e12 (s[3], s[5]): the
/// per-launch dot measured ||TL x||/||x|| = 8.0e10, all of it in sr, at step 6. Frozen-value treatment, default
/// on: at those columns sr's tangent is 0 and its cotangent is dropped (an output projection P: TL' = P J,
/// AD' = J^T P, so the pair stays exact transposes). AD_TEMPO_SR_EXACT=1 keeps the code's exact derivative.
const SR_SLOT: usize = 17;

/// The P5 serial tape of one snapshot, every column (record pass, then the sweeps read it).
pub struct SmallTape { ta: u64, tb: u64, tpa: u64, tpb: u64, scr: u64, oidx: u64, yout: u64, cout: u64, cnt: u64, z: u64, capn: usize, pub record_ms: f64 }

impl SmallSnap {
    /// Float64 words of the entry's input and output vectors ([nin][nc] and [nout][nc]).
    pub fn io_words(&self) -> (usize, usize) { (self.nin * self.nc, self.nout * self.nc) }
    /// Input words that belong to read-only slots (tempo_precip: f[3], f[4], f[18], the first 3 * nz slots).
    pub fn read_only_words(&self) -> usize { if self.kind == 0 { 3 * self.nz * self.nc } else { 0 } }
    /// Outputs as (live address, words, first float64 word of the output vector).
    pub fn outputs(&self) -> Vec<(u64, usize, usize)> {
        self.post.iter().filter(|p| p.0 != 0).map(|&(p, per, slot)| (p, per * self.nc, slot * self.nc)).collect()
    }
    /// Inputs as (live address, words, first float64 word of the input vector).
    pub fn input_list(&self) -> Vec<(u64, usize, usize)> { self.inputs.clone() }
}

unsafe fn launch_with(cuda: &Cuda, f: *mut c_void, grid: u32, block: u32, args: &[Vec<u8>]) -> Result<(), String> {
    let mut store: Vec<Vec<u64>> = args.iter().map(|b| { let mut w = vec![0u64; b.len().div_ceil(8)]; ptr::copy_nonoverlapping(b.as_ptr(), w.as_mut_ptr().cast::<u8>(), b.len()); w }).collect();
    let mut ps: Vec<*mut c_void> = store.iter_mut().map(|w| w.as_mut_ptr().cast()).collect();
    checked("tempo_ad launch", (cuda.launch)(f, grid, 1, 1, block, 1, 1, 0, ptr::null_mut(), ps.as_mut_ptr(), ptr::null_mut()))?;
    dev::sync(cuda)
}

impl TempoSmall {
    pub unsafe fn load(cuda: &Cuda, path: &str) -> Result<Self, String> {
        let mut image = std::fs::read(path).map_err(|e| format!("AD_DOT_TEMPO_SERIAL_CUBIN {path}: {e}"))?;
        let image_sha256 = { use sha2::Digest; format!("{:x}", sha2::Sha256::digest(&image)) };
        image.push(0);
        let mut m = ptr::null_mut();
        checked("cuModuleLoadData tempo_ad", (cuda.module_load)(&mut m, image.as_ptr().cast()))?;
        let func = |n: &str| -> Result<*mut c_void, String> {
            let c = CString::new(n).unwrap(); let mut f = ptr::null_mut();
            checked(&format!("cuModuleGetFunction {n}"), (cuda.module_function)(&mut f, m, c.as_ptr()))?; Ok(f)
        };
        let (mut g, mut gn) = (0u64, 0usize);
        let c = CString::new("g_ad").unwrap();
        checked("cuModuleGetGlobal g_ad", (cuda.module_global)(&mut g, &mut gn, m, c.as_ptr()))?;
        if gn != 104 { return Err(format!("tempo_ad g_ad is {gn} bytes, expected 104 (serial tape layout)")); }
        let mut sz = 0u64;
        checked("cuMemAlloc sizes", (cuda.allocate)(&mut sz, 24))?;
        launch_with(cuda, func("tempo_ad_sizes")?, 1, 1, &[u64b(sz)])?;
        let s: Vec<u64> = get(cuda, sz, 3)?;
        dev::zero(cuda, sz, 24)?; dev::sync(cuda)?; // fresh allocations of the replay must read zero
        checked("cuMemFree sizes", (cuda.free)(sz))?;
        Ok(Self { precip: func("tempo_ad_precip_record")?, radar: func("tempo_ad_radar_record")?, tl: func("tempo_ad_tl")?, adj: func("tempo_ad_adj")?, g, prim: s[2] as usize, image_sha256 })
    }

    /// Before the forward launch: copies of every array the launch reads (tables mirrored), and where its outputs land.
    pub unsafe fn snapshot(&self, cuda: &Cuda, arena: &mut Arena, live: &HashMap<u64, Allocation>, function: &str, args: &[Argument]) -> Result<SmallSnap, String> {
        if args.len() != 9 { return Err(format!("{function} with {} arguments (9 expected)", args.len())); }
        let addr = |i: usize| -> Result<u64, String> {
            match &args[i] {
                Argument::Device(s) => Ok(live.get(&s.allocation).ok_or(format!("argument {i}: allocation {} not live", s.allocation))?.address + s.offset as u64),
                Argument::Bytes(b) if b.len() == 8 && b.iter().all(|&x| x == 0) => Ok(0),
                a => Err(format!("argument {i}: unexpected binding {a:?}")),
            }
        };
        let raw = |i: usize| -> Result<Vec<u8>, String> { match &args[i] { Argument::Bytes(b) => Ok(b.clone()), a => Err(format!("argument {i}: not a scalar {a:?}")) } };
        let u32a = |i: usize| -> Result<u32, String> { let b = raw(i)?; Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]])) };
        let (nz, nc) = (u32a(4)? as usize, u32a(5)? as usize);
        // copy of table t (n entries): entries in want, count words each; returns the mirror table, the live pointers
        // and the copy of each entry
        let copy_table = |arena: &mut Arena, t: u64, n: usize, want: &[usize], count: usize| -> Result<(u64, Vec<u64>, Vec<u64>), String> {
            let p: Vec<u64> = get(cuda, t, n)?;
            let mut v = vec![0u64; n];
            for &j in want { if j < n && p[j] != 0 { let a = arena.take(4 * count)?; dev::copy(cuda, a, p[j], 4 * count)?; v[j] = a; } }
            let tab = arena.take(8 * n)?; put(cuda, tab, &v)?;
            Ok((tab, p, v))
        };
        let mut post = vec![];
        let snap = if function == "tempo_precip" {
            let (ft, fp, fv) = copy_table(arena, addr(0)?, 20, &[3, 4, 18], nz * nc)?;
            let (st, sp, sv) = copy_table(arena, addr(1)?, 12, &(0..10).collect::<Vec<_>>(), nc)?;
            let (dt_, dp, dv) = copy_table(arena, addr(2)?, 13, &(0..13).collect::<Vec<_>>(), nc)?;
            let cp = addr(3)?; let cs = arena.take(4 * nc)?; dev::copy(cuda, cs, cp, 4 * nc)?;
            let finish = u32a(6)? != 0;
            let exact = std::env::var("AD_TEMPO_SR_EXACT").ok().as_deref() == Some("1");
            let sr_frozen: Vec<usize> = if finish && !exact && dp[7] != 0 && sv[1] != 0 {
                let s1: Vec<f32> = get(cuda, sv[1], nc)?;
                (0..nc).filter(|&c| s1[c] + 1.0e-12f32 == 1.0e-12f32).collect()
            } else { vec![] };
            for j in 0..10 { post.push((sp[j], 1, j)); }
            for j in 0..13 { post.push((dp[j], 1, 10 + j)); }
            // Input vector of tempo_ad_precip_record (slot q, column c at q * nc + c): f[3], f[4], f[18] by level,
            // then s[0..9], then d[0..12] (driver.cuh).
            let (mut inputs, mut copies) = (vec![], vec![]);
            for (i, &j) in [3usize, 4, 18].iter().enumerate() { if fp[j] != 0 { inputs.push((fp[j], nz * nc, i * nz * nc)); copies.push((fv[j], nz * nc, i * nz * nc)); } }
            for j in 0..10 { if sp[j] != 0 { inputs.push((sp[j], nc, (3 * nz + j) * nc)); copies.push((sv[j], nc, (3 * nz + j) * nc)); } }
            for j in 0..13 { if dp[j] != 0 { inputs.push((dp[j], nc, (3 * nz + 10 + j) * nc)); copies.push((dv[j], nc, (3 * nz + 10 + j) * nc)); } }
            let a = vec![u64b(ft), u64b(st), u64b(dt_), u64b(cs), raw(4)?, raw(5)?, raw(6)?, raw(7)?, raw(8)?];
            SmallSnap { kind: 0, args: a, nz, nc, nin: 3 * nz + 23, nout: 23, post, counter_post: cp, full: true, inputs, copies, sr_frozen }
        } else {
            let (ft, _, _) = copy_table(arena, addr(0)?, 20, &[0, 1, 3, 4, 17, 19], nz * nc)?;
            let (pt, _, _) = copy_table(arena, addr(1)?, 4, &[0], nz * nc)?;
            let (rt, rp, _) = copy_table(arena, addr(2)?, 3, &[0, 1, 2], nc)?;
            let hw = addr(8)?;
            let full = u32a(7)? == 1;
            for j in 0..3 { post.push((rp[j], 1, j)); }
            if full { post.push((addr(3)?, nz, 3)); }
            let a = vec![u64b(ft), u64b(pt), u64b(rt), raw(4)?, raw(5)?, raw(6)?, raw(7)?, u64b(hw)];
            SmallSnap { kind: 1, args: a, nz, nc, nin: 7 * nz + 3, nout: nz + 3, post, counter_post: 0, full, inputs: vec![], copies: vec![], sr_frozen: vec![] }
        };
        dev::sync(cuda)?;
        Ok(snap)
    }

    /// Record pass over every column of a snapshot (a count-only pass sizes the tape, then the taped pass).
    /// Err "tape overflow" if a column outgrew its tape.
    pub unsafe fn record_tape(&self, cuda: &Cuda, arena: &mut Arena, s: &SmallSnap) -> Result<SmallTape, String> {
        let (n, nz, nin, nout) = (s.nc, s.nz, s.nin, s.nout);
        let cols: Vec<i32> = (0..n as i32).collect();
        let colsb = arena.take(4 * n)?; put(cuda, colsb, &cols)?;
        let buf = arena.take(16 * (nin + nz) * n)?; let yout = arena.take(self.prim * nout * n)?; let oidx = arena.take(4 * nout * n)?;
        let cout = arena.take(4 * n)?; let cnt = arena.take(16 * n)?; let z = arena.take(8)?;
        let f = if s.kind == 0 { self.precip } else { self.radar };
        let record = |cap: usize, ta: u64, tb: u64, tpa: u64, tpb: u64| -> Result<f64, String> {
            put(cuda, self.g, &[ta, tb, tpa, tpb, z, z, cnt, 0, 0, n as u64, cap as u64, 0, 0])?;
            let mut a = s.args.clone();
            a.extend([u64b(colsb), i32b(n as i32), u64b(buf), u64b(yout), u64b(oidx), u64b(cout), u64b(0)]);
            let t = Instant::now();
            launch_with(cuda, f, n.div_ceil(64) as u32, 64, &a)?;
            Ok(t.elapsed().as_secs_f64() * 1e3)
        };
        record(0, z, z, z, z)?;
        let c4: Vec<i32> = get(cuda, cnt, 4 * n)?;
        let capn = (0..n).map(|t| c4[4 * t] as usize).max().unwrap_or(0) + 1;
        let (ta, tb, tpa, tpb, scr) = (arena.take(4 * capn * n)?, arena.take(4 * capn * n)?, arena.take(8 * capn * n)?, arena.take(8 * capn * n)?, arena.take(8 * capn * n)?);
        let record_ms = record(capn, ta, tb, tpa, tpb)?;
        let c4: Vec<i32> = get(cuda, cnt, 4 * n)?;
        if (0..n).any(|t| c4[4 * t + 3] != 0) { return Err("tape overflow".into()); }
        Ok(SmallTape { ta, tb, tpa, tpb, scr, oidx, yout, cout, cnt, z, capn, record_ms })
    }

    /// Words of the recorded outputs that differ from the live post-state (and the bucket counter for precip).
    /// Meaningful only right after the forward launch.
    pub unsafe fn identity(&self, cuda: &Cuda, s: &SmallSnap, tp: &SmallTape) -> Result<(usize, usize), String> {
        let n = s.nc;
        let y: Vec<f32> = if self.prim == 4 { get(cuda, tp.yout, s.nout * n)? } else { get::<f64>(cuda, tp.yout, s.nout * n)?.iter().map(|&v| v as f32).collect() };
        let (mut idiff, mut icmp) = (0usize, 0usize);
        for &(p, per, slot) in &s.post {
            if p == 0 { continue; }
            let v: Vec<f32> = get(cuda, p, per * n)?;
            for k in 0..per { for c in 0..n { icmp += 1; if v[k * n + c].to_bits() != y[(slot + k) * n + c].to_bits() { idiff += 1; } } }
        }
        if s.kind == 0 {
            let a: Vec<i32> = get(cuda, s.counter_post, n)?; let b: Vec<i32> = get(cuda, tp.cout, n)?;
            icmp += n; idiff += a.iter().zip(&b).filter(|(x, y)| x != y).count();
        }
        Ok((idiff, icmp))
    }

    /// The sr projection (SR_SLOT) on an output-shaped float64 buffer: frozen columns of slot 17 set to 0.
    unsafe fn freeze_sr(&self, cuda: &Cuda, s: &SmallSnap, buf: u64) -> Result<(), String> {
        if s.sr_frozen.is_empty() { return Ok(()); }
        let n = s.nc;
        let row = buf + (8 * SR_SLOT * n) as u64;
        let mut v: Vec<f64> = get(cuda, row, n)?;
        for &c in &s.sr_frozen { v[c] = 0.0; }
        put(cuda, row, &v)
    }

    /// One sweep of a recorded tape on device buffers: tangent (input xdot [nin][nc] -> output ydot [nout][nc]) or
    /// adjoint (input ybar [nout][nc] -> output xbar [nin][nc]). Float64 both sides. The sr projection is applied
    /// to ydot after the tangent and to ybar (in place) before the adjoint.
    pub unsafe fn sweep_dev(&self, cuda: &Cuda, s: &SmallSnap, tp: &SmallTape, adjoint: bool, input: u64, output: u64) -> Result<(), String> {
        let n = s.nc;
        if adjoint { self.freeze_sr(cuda, s, input)?; }
        put(cuda, self.g, &[tp.ta, tp.tb, tp.tpa, tp.tpb, tp.z, tp.z, tp.cnt, 0, 0, n as u64, tp.capn as u64, 0, 0])?;
        launch_with(cuda, if adjoint { self.adj } else { self.tl }, n.div_ceil(64) as u32, 64,
                    &[i32b(n as i32), i32b(s.nin as i32), i32b(s.nout as i32), u64b(input), u64b(tp.scr), u64b(tp.oidx), u64b(output)])?;
        if !adjoint { self.freeze_sr(cuda, s, output)?; }
        Ok(())
    }

    pub unsafe fn launch_dot(&self, cuda: &Cuda, arena: &mut Arena, s: &SmallSnap, seed: u64, tol: f64, checks: bool) -> Result<Value, String> {
        let t_all = Instant::now();
        let (n, nz, nin, nout) = (s.nc, s.nz, s.nin, s.nout);
        let tp = match self.record_tape(cuda, arena, s) { Ok(t) => t, Err(e) if e == "tape overflow" => return Ok(json!({"status":"ERROR","error":"tape overflow"})), Err(e) => return Err(e) };
        let rec_ms = tp.record_ms;
        // identity against the forward's post-state
        let (idiff, icmp) = self.identity(cuda, s, &tp)?;
        let (xb_, yb_) = (arena.take(8 * nin.max(nout) * n)?, arena.take(8 * nin.max(nout) * n)?);
        let sweep = |adjoint: bool, input: &[f64], outn: usize| -> Result<Vec<f64>, String> {
            put(cuda, xb_, input)?;
            self.sweep_dev(cuda, s, &tp, adjoint, xb_, yb_)?;
            get(cuda, yb_, outn * n)
        };
        let mut g = Rng::new(seed);
        let xd: Vec<f64> = (0..nin * n).map(|_| g.normal()).collect(); let yb: Vec<f64> = (0..nout * n).map(|_| g.normal()).collect();
        let t1 = Instant::now(); let yd = sweep(false, &xd, nout)?; let tl_ms = t1.elapsed().as_secs_f64() * 1e3;
        let t2 = Instant::now(); let xb = sweep(true, &yb, nin)?; let ad_ms = t2.elapsed().as_secs_f64() * 1e3;
        let l: f64 = (0..nout * n).map(|i| yd[i] * yb[i]).sum(); let r: f64 = (0..nin * n).map(|i| xd[i] * xb[i]).sum();
        let rel = (l - r).abs() / l.abs().max(r.abs()).max(1e-300);
        // ||TL(x)|| / ||x|| on the random direction, and per input block (one TL sweep with x on that block only,
        // checked launches): the Jacobian's size by input field, for the range-hazard table.
        let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let gain = norm(&yd) / norm(&xd).max(1e-300);
        const PNAMES: [&str; 23] = ["s0","s1","s2","s3","s4","s5","s6","s7","s8","s9","d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","d11","d12"];
        let blocks: Vec<(&str, usize)> = if s.kind == 0 {
            let mut b = vec![("f3", nz), ("f4", nz), ("f18", nz)];
            for nm in PNAMES { b.push((nm, 1)); }
            b
        } else { vec![("f0_theta", nz), ("f1_pressure", nz), ("f3_dz", nz), ("f4_qv", nz), ("f17_exner", nz), ("f19_zgrid", nz),
                      ("profile0_refl", nz), ("radar0_composite", 1), ("radar1_1km", 1), ("radar2_hourly_max", 1)] };
        let ogroups: Vec<(&str, usize, usize)> = if s.kind == 0 { vec![("all", 0, nout)] } else { vec![("radar", 0, 3), ("rh", 3, nout)] };
        let mut block_gain = serde_json::Map::new();
        if checks {
            let mut q0 = 0;
            for (name, w) in &blocks {
                let mut xb1 = vec![0f64; nin * n];
                xb1[q0 * n..(q0 + w) * n].copy_from_slice(&xd[q0 * n..(q0 + w) * n]);
                let y1 = sweep(false, &xb1, nout)?;
                let nx = norm(&xb1).max(1e-300);
                let mut row = serde_json::Map::new();
                for (g, a, b) in &ogroups { row.insert((*g).into(), json!(norm(&y1[a * n..b * n]) / nx)); }
                block_gain.insert((*name).into(), Value::Object(row));
                q0 += w;
            }
        }
        let (mut zero_nonzero, mut det) = (0usize, true);
        if checks {
            let xb2 = sweep(true, &yb, nin)?; det = xb2.iter().zip(&xb).all(|(a, b)| a.to_bits() == b.to_bits());
            let yz = sweep(false, &vec![0f64; nin * n], nout)?; let xz = sweep(true, &vec![0f64; nout * n], nin)?;
            zero_nonzero = yz.iter().chain(xz.iter()).filter(|v| **v != 0.0 || v.is_nan()).count();
        }
        // Gain ||TL x|| / ||x|| (lane ad/held-tempo_precip): on the dot's own normal direction, and on a relative
        // direction (x_q = u_q |x0_q|, the window's default x) measured relative to the state,
        // (||TL x|| / ||y0||) / (||x|| / ||x0||); each with the output slot that carries most of ||TL x||^2.
        let nrm = |v: &[f64]| v.iter().map(|a| a * a).sum::<f64>().sqrt();
        let slot_ss = |v: &[f64]| -> Vec<f64> { (0..nout).map(|q| v[q * n..(q + 1) * n].iter().map(|a| a * a).sum::<f64>()).collect() };
        let top = |v: &[f64]| { let (q, m) = v.iter().enumerate().fold((0usize, -1.0f64), |b, (q, &x)| if x > b.1 { (q, x) } else { b }); json!({"slot": q, "share": m / v.iter().sum::<f64>().max(1e-300)}) };
        let gain = nrm(&yd) / nrm(&xd).max(1e-300);
        let colmax = yd.iter().enumerate().fold((0usize, 0f64), |b, (i, &v)| if v.abs() > b.1 { (i, v.abs()) } else { b });
        let mut gainj = json!({"tl_gain": gain, "tl_gain_top_slot": top(&slot_ss(&yd)), "tl_max_abs": colmax.1, "tl_max_abs_slot": colmax.0 / n, "tl_max_abs_column": colmax.0 % n});
        if !s.copies.is_empty() {
            let mut x0 = vec![0f64; nin * n];
            for &(a, w, off) in &s.copies { let v: Vec<f32> = get(cuda, a, w)?; for (k, x) in v.iter().enumerate() { x0[off + k] = *x as f64; } }
            let y0: Vec<f64> = if self.prim == 4 { get::<f32>(cuda, tp.yout, nout * n)?.iter().map(|&v| v as f64).collect() } else { get::<f64>(cuda, tp.yout, nout * n)? };
            let mut gr = Rng::new(seed ^ 0x6a17);
            let xr: Vec<f64> = x0.iter().map(|v| gr.normal() * v.abs()).collect();
            let yr = sweep(false, &xr, nout)?;
            let rgain = (nrm(&yr) / nrm(&y0).max(1e-300)) / (nrm(&xr) / nrm(&x0).max(1e-300)).max(1e-300);
            let rmax = yr.iter().enumerate().fold((0usize, 0f64), |b, (i, &v)| if v.abs() > b.1 { (i, v.abs()) } else { b });
            gainj["tl_gain_relative"] = json!(rgain);
            gainj["tl_gain_relative_top_slot"] = top(&slot_ss(&yr));
            gainj["tl_relative_max_abs"] = json!(rmax.1); gainj["tl_relative_max_abs_slot"] = json!(rmax.0 / n); gainj["tl_relative_max_abs_column"] = json!(rmax.0 % n);
            gainj["x0_norm"] = json!(nrm(&x0)); gainj["y0_norm"] = json!(nrm(&y0));
        }
        let ok = idiff == 0 && rel <= tol && rel.is_finite() && (!checks || (zero_nonzero == 0 && det));
        let mut row = json!({"status": if ok { "PASS" } else { "FAIL" }, "rel": rel, "lhs": l, "rhs": r, "tol": tol, "identity_differing_words": idiff, "identity_compared_words": icmp,
                  "checked": checks, "zero_direction_nonzero_words": if checks { json!(zero_nonzero) } else { Value::Null },
                  "reverse_repeat_bitwise": if checks { json!(det) } else { Value::Null }, "full": s.full,
                  "tl_gain": gain, "tl_gain_by_input_block": if checks { Value::Object(block_gain) } else { Value::Null },
                  "tl_ms": tl_ms, "ad_ms": ad_ms + rec_ms, "record_ms": rec_ms, "wall_ms": t_all.elapsed().as_secs_f64() * 1e3, "columns": n,
                  "tape_nodes_per_column_max": tp.capn - 1, "sr_frozen_columns": s.sr_frozen.len(), "derivative": "P5 per-thread entry (forecast shape: one thread per column), float64 tape"});
        for (k, v) in gainj.as_object().unwrap() { row[k] = v.clone(); }
        Ok(row)
    }
}

// ------------------------------------------------------------------------------------------------------------
// Composed window binding (ad/held-tempo_radar): the same P5 per-thread entry, driven from window.rs on the window's
// float32 shadows (--tempo-small tempo_radar). tempo_radar only: the precip entry's window binding is its own lane.
//   - window_snapshot: the inputs copied into the window's tape arena before the forward (as `snapshot`), plus the
//     do_diag_dbz word, which the host may flip before the reverse re-records the launch;
//   - window_slots: the device address and word count of every input slot block and output slot block, in the
//     entry's slot order (inputs f[0], f[1], f[3], f[4], f[17], f[19], profile[0] by level, radar[0..2]; outputs
//     radar[0..2], rh by level). Both sides are [level][column] with columns 0..nc, so a level block is one
//     contiguous run of nz*nc words in the shadow and in the entry's [slot][column] buffers. An output the forward
//     does not write (rh when full == 0) has address 0 and is neither scattered nor gathered; radar[0..2] are always
//     listed: where the forward leaves them (full == 0, hourly off) the entry's output is the input leaf itself, so
//     tangent and adjoint pass them through exactly;
//   - window_record: count pass and record pass of the tape from the snapshot (float64 tape, exactly as launch_dot);
//     with `check`, the recorded primal against the forward's post-state word for word (right after the forward);
//   - window_sweep: tangent (x -> y) or adjoint (ybar -> xbar) on device buffers, [slot][column] float64.
pub struct SmallRec { pub n: usize, pub nin: usize, pub nout: usize, capn: usize, ta: u64, tb: u64, tpa: u64, tpb: u64, scr: u64,
                      oidx: u64, cnt: u64, z: u64, pub identity_compared: usize }

impl TempoSmall {
    pub unsafe fn window_snapshot(&self, cuda: &Cuda, arena: &mut Arena, live: &HashMap<u64, Allocation>, function: &str, args: &[Argument]) -> Result<SmallSnap, String> {
        if function != "tempo_radar" { return Err(format!("{function}: the window binds the P5 per-thread entry for tempo_radar only")); }
        let mut s = self.snapshot(cuda, arena, live, function, args)?;
        let hw = u64::from_le_bytes(s.args[7][..8].try_into().map_err(|_| "hourly word argument".to_string())?);
        if hw != 0 { let c = arena.take(8)?; dev::copy(cuda, c, hw, 4)?; s.args[7] = u64b(c); }
        dev::sync(cuda)?;
        Ok(s)
    }

    pub unsafe fn window_slots(&self, cuda: &Cuda, live: &HashMap<u64, Allocation>, function: &str, args: &[Argument])
                               -> Result<(Vec<(u64, usize)>, Vec<(u64, usize)>), String> {
        if function != "tempo_radar" { return Err(format!("{function}: no window slots")); }
        if args.len() != 9 { return Err(format!("{function} with {} arguments (9 expected)", args.len())); }
        let addr = |i: usize| -> Result<u64, String> {
            match &args[i] {
                Argument::Device(s) => Ok(live.get(&s.allocation).ok_or(format!("argument {i}: allocation {} not live", s.allocation))?.address + s.offset as u64),
                Argument::Bytes(b) if b.len() == 8 && b.iter().all(|&x| x == 0) => Ok(0),
                a => Err(format!("argument {i}: unexpected binding {a:?}")),
            }
        };
        let u32a = |i: usize| -> Result<u32, String> { match &args[i] { Argument::Bytes(b) if b.len() >= 4 => Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]])), a => Err(format!("argument {i}: not a scalar {a:?}")) } };
        let (nz, nc) = (u32a(4)? as usize, u32a(5)? as usize);
        let full = u32a(7)? == 1;
        let ft: Vec<u64> = get(cuda, addr(0)?, 20)?; let pt: Vec<u64> = get(cuda, addr(1)?, 4)?; let rt: Vec<u64> = get(cuda, addr(2)?, 3)?;
        let mut ins: Vec<(u64, usize)> = [0usize, 1, 3, 4, 17, 19].iter().map(|&j| (ft[j], nz * nc)).collect();
        ins.push((pt[0], nz * nc));
        for j in 0..3 { ins.push((rt[j], nc)); }
        let mut outs: Vec<(u64, usize)> = (0..3).map(|j| (rt[j], nc)).collect();
        outs.push((if full { addr(3)? } else { 0 }, nz * nc));
        if (0..3).any(|j| rt[j] == 0) { return Err("tempo_radar: a radar output pointer is null".into()); }
        Ok((ins, outs))
    }

    pub unsafe fn window_record(&self, cuda: &Cuda, arena: &mut Arena, s: &SmallSnap, check: bool) -> Result<(SmallRec, usize), String> {
        let (n, nz, nin, nout) = (s.nc, s.nz, s.nin, s.nout);
        let cols: Vec<i32> = (0..n as i32).collect();
        let colsb = arena.take(4 * n)?; put(cuda, colsb, &cols)?;
        let buf = arena.take(16 * (nin + nz) * n)?; let yout = arena.take(self.prim * nout * n)?; let oidx = arena.take(4 * nout * n)?;
        let cout = arena.take(4 * n)?; let cnt = arena.take(16 * n)?; let z = arena.take(8)?;
        dev::zero(cuda, z, 8)?;
        let f = if s.kind == 0 { self.precip } else { self.radar };
        let record = |cap: usize, ta: u64, tb: u64, tpa: u64, tpb: u64| -> Result<(), String> {
            put(cuda, self.g, &[ta, tb, tpa, tpb, z, z, cnt, 0, 0, n as u64, cap as u64, 0, 0])?;
            let mut a = s.args.clone();
            a.extend([u64b(colsb), i32b(n as i32), u64b(buf), u64b(yout), u64b(oidx), u64b(cout), u64b(0)]);
            launch_with(cuda, f, n.div_ceil(64) as u32, 64, &a)
        };
        record(0, z, z, z, z)?;
        let c4: Vec<i32> = get(cuda, cnt, 4 * n)?;
        let capn = (0..n).map(|t| c4[4 * t] as usize).max().unwrap_or(0) + 1;
        let (ta, tb, tpa, tpb, scr) = (arena.take(4 * capn * n)?, arena.take(4 * capn * n)?, arena.take(8 * capn * n)?, arena.take(8 * capn * n)?, arena.take(8 * capn * n)?);
        record(capn, ta, tb, tpa, tpb)?;
        let c4: Vec<i32> = get(cuda, cnt, 4 * n)?;
        if (0..n).any(|t| c4[4 * t + 3] != 0) { return Err("tempo_ad_serial tape overflow".into()); }
        let (mut idiff, mut icmp) = (0usize, 0usize);
        if check {
            let y: Vec<f32> = if self.prim == 4 { get(cuda, yout, nout * n)? } else { get::<f64>(cuda, yout, nout * n)?.iter().map(|&v| v as f32).collect() };
            for &(p, per, slot) in &s.post {
                if p == 0 { continue; }
                let v: Vec<f32> = get(cuda, p, per * n)?;
                for k in 0..per { for c in 0..n { icmp += 1; if v[k * n + c].to_bits() != y[(slot + k) * n + c].to_bits() { idiff += 1; } } }
            }
        }
        Ok((SmallRec { n, nin, nout, capn, ta, tb, tpa, tpb, scr, oidx, cnt, z, identity_compared: icmp }, idiff))
    }

    pub unsafe fn window_sweep(&self, cuda: &Cuda, r: &SmallRec, adjoint: bool, input: u64, output: u64) -> Result<(), String> {
        put(cuda, self.g, &[r.ta, r.tb, r.tpa, r.tpb, r.z, r.z, r.cnt, 0, 0, r.n as u64, r.capn as u64, 0, 0])?;
        let f = if adjoint { self.adj } else { self.tl };
        launch_with(cuda, f, r.n.div_ceil(64) as u32, 64, &[i32b(r.n as i32), i32b(r.nin as i32), i32b(r.nout as i32), u64b(input), u64b(r.scr), u64b(r.oidx), u64b(output)])
    }
}
