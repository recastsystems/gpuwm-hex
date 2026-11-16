// SPDX-License-Identifier: Apache-2.0
//! Physics-on window, stage A (ad/phys-window): per-launch derivative dot over the sealed hrrrv5 plan,
//! inside the P1 replay runner. The runner executes every command exactly as the strict replay does
//! (so the boundary hashes are still checked by finish_step); at each kernel launch that owns an
//! ad.entry/v1 derivative (the ad/e2e kernel set: dycore, transport, filters, recovery) the entry is
//! bound to the launch's live arguments, a pre-launch snapshot is taken into a private arena, the
//! forward launch runs unchanged, then U6's launch dot (TL vs AD on random seeds, zero direction and
//! a second reverse run on the first launches of each kernel per step) runs in the arena only.
//! L2 bindings (ad/window-L2-bindings): a launch with no dispatchable ad.entry/v1 entry that owns an
//! ad.scheme.rows/v1 row (AD_SCHEMES: P3 MYNN-EDMF, P4 surface layer and GSL drag, P8's command rules)
//! is bound and dot-tested by schemes.rs the same way. Every other launch is counted per function with
//! its class from P8's rule table (rules::kernel_class, loaded natively): the held / unbound set.
//! Enabled by AD_DOT_KERNELS + AD_DOT_CUBINS + AD_DOT_OUT; without them nothing changes.
use crate::cuda::Cuda;
use crate::{deriv, dev, dot, schemes, submit};
use crate::p8bridge::rules;
use serde_json::{json, Value};
use std::{collections::{BTreeMap, HashSet}, fs::File, io::{BufWriter, Write}, path::{Path, PathBuf}};

pub struct AdDot {
    kernels: std::collections::HashMap<String, deriv::KernelDeriv>,
    native: Option<schemes::Native>,
    spec_names: HashSet<String>,
    support: dev::Support,
    arena: dev::Arena,
    out: BufWriter<File>,
    out_path: PathBuf,
    width: usize,
    tol: f64,
    checks_per_kernel: u64,
    first: usize,
    summary: BTreeMap<String, Value>,
    pub commands: BTreeMap<String, u64>,
    checked: BTreeMap<(usize, String), u64>,
    index: u64,
    start: std::time::Instant,
    step_rows: Vec<Value>,
    /// AD_DOT_PERSIST=1: every launch binds whole-allocation shadows (one per live allocation, float32 width, made
    /// once), as the composed sweeps do; needed for mirrored pointer tables (mpas_rk_cells, prep_mass, the RRFS copy
    /// plans) and for allocations bound with both float32 and float64 shadows (fct_scale, mpas_rk_edges).
    persistent: Option<BTreeMap<u64, deriv::Shadow>>,
    pub persistent_bytes: usize,
    /// AD_DOT_TEMPO_CUBIN (lane L5): tempo_warp_steps launches take TEMPO's warp-shaped whole-launch derivative
    /// (crate::tempo_adw), driven from here; the snapshot of the launch inputs waits between before and after.
    tempo: Option<crate::tempo_adw::TempoAdw>,
    tempo_snap: Option<crate::tempo_adw::Snap>,
    /// AD_DOT_TEMPO_SERIAL_CUBIN: tempo_precip / tempo_radar launches take P5's per-thread entries.
    tempo_small: Option<crate::tempo_adw::TempoSmall>,
    tempo_small_snap: Option<crate::tempo_adw::SmallSnap>,
    /// Held-kernel lane tempo_order_scatter: the column-order scatter's launch entry (crate::tempo_order: integer
    /// outputs, zero map, schedule witness); the pre-launch words wait between before and after. Opt-in with
    /// AD_DOT_ORDER_WITNESS=1: by default the launch takes its ad.entry/v1 row (phys/held/tempo_order_scatter), which the
    /// composed window binds too.
    order_snap: Option<crate::tempo_order::Snap>,
}

/// A launch's bound derivative: an ad.entry/v1 spec binding or an ad.scheme.rows/v1 row binding.
pub enum Prep { Spec(deriv::Bound), Native(schemes::Prepared), Constant(ConstSnap) }

/// An input-free image constant launch (wrules::image_constant): its output span, the pre-launch bytes, and the
/// poison written there before the forward ran.
pub struct ConstSnap { address: u64, bytes: usize, pre: Vec<u8>, poison: Vec<u8> }

fn hex(b: &[u8]) -> String { b.iter().map(|x| format!("{x:02x}")).collect() }
fn f32s(b: &[u8]) -> Vec<f64> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64).collect() }
pub type Pending = Option<Result<Prep, String>>;

fn env(k: &str) -> Option<String> { std::env::var(k).ok().filter(|v| !v.is_empty()) }

impl AdDot {
    pub unsafe fn from_env(cuda: &Cuda) -> Result<Option<Self>, String> {
        let (Some(k), Some(c), Some(o)) = (env("AD_DOT_KERNELS"), env("AD_DOT_CUBINS"), env("AD_DOT_OUT")) else { return Ok(None) };
        let width = env("AD_DOT_WIDTH").map(|w| w.parse().unwrap_or(4)).unwrap_or(4);
        let tol = env("AD_DOT_TOL").map(|t| t.parse().unwrap_or(2e-5)).unwrap_or(if width == 8 { 1e-10 } else { 2e-5 });
        let gib: usize = env("AD_DOT_ARENA_GIB").map(|g| g.parse().unwrap_or(12)).unwrap_or(12);
        crate::window::check_spec_hashes(Path::new(&k))?;
        let kernels = deriv::load(cuda, Path::new(&k), Path::new(&c), None)?;
        let spec_names: HashSet<String> = kernels.keys().cloned().collect();
        let native = schemes::Native::from_env(cuda, &spec_names)?;
        let support = dev::Support::load(cuda)?;
        let arena = dev::Arena::new(cuda, gib << 30)?;
        let out_path = PathBuf::from(o);
        let mut out = BufWriter::new(File::create(&out_path).map_err(|e| e.to_string())?);
        let loaded = kernels.values().filter(|k| k.unsupported.is_none()).count();
        let tempo = match env("AD_DOT_TEMPO_CUBIN") { Some(p) => Some(crate::tempo_adw::TempoAdw::load(cuda, &p)?), None => None };
        let tempo_small = match env("AD_DOT_TEMPO_SERIAL_CUBIN") { Some(p) => Some(crate::tempo_adw::TempoSmall::load(cuda, &p)?), None => None };
        let ev = json!({"event":"ad_dot_loaded","kernels":kernels.len(),"dispatchable":loaded,"arena_gib":gib,
                        "scheme_rows":native.as_ref().map(|n| n.rows().count()).unwrap_or(0),
                        "tempo_adw":tempo.as_ref().map(|t| t.image_sha256.clone()),
                        "tempo_ad_serial":tempo_small.as_ref().map(|t| t.image_sha256.clone())});
        eprintln!("{ev}");
        if let Some(n) = native.as_ref() { writeln!(out, "{}", n.loaded).map_err(|e| e.to_string())?; }
        Ok(Some(Self { kernels, native, spec_names, support, arena, out, out_path, width, tol,
            checks_per_kernel: env("AD_DOT_CHECKS").map(|c| c.parse().unwrap_or(2)).unwrap_or(2),
            first: env("AD_DOT_FIRST").map(|c| c.parse().unwrap_or(1)).unwrap_or(1),
            summary: BTreeMap::new(), commands: BTreeMap::new(), checked: BTreeMap::new(), index: 0,
            start: std::time::Instant::now(), step_rows: Vec::new(),
            persistent: (env("AD_DOT_PERSIST").as_deref() == Some("1")).then(BTreeMap::new), persistent_bytes: 0, tempo, tempo_snap: None, tempo_small, tempo_small_snap: None, order_snap: None }))
    }

    pub fn count(&mut self, kind: String) { *self.commands.entry(kind).or_default() += 1; }

    fn untested(&mut self, function: &str, why: String) {
        let e = self.summary.entry(function.to_owned()).or_insert_with(|| json!({"launches":0,"tested":0,"why":why}));
        e["launches"] = json!(e["launches"].as_u64().unwrap() + 1);
    }

    /// Before the forward launch: bind and snapshot (or record why the launch has no derivative).
    pub unsafe fn before(&mut self, cuda: &Cuda, live: &std::collections::HashMap<u64, submit::Allocation>, step: usize,
                         function: &str, args: &[submit::Argument], grid: [u32; 3], block: [u32; 3], shared: u32) -> Result<Pending, String> {
        self.index += 1;
        let skipped = env("AD_DOT_SKIP").is_some_and(|v| v.split(',').any(|f| f == function))
            || env("AD_DOT_ONLY").is_some_and(|v| !v.split(',').any(|f| f == function));
        if function == "tempo_warp_steps" && self.tempo.is_some() && !skipped && step >= self.first {
            dev::sync(cuda)?;
            self.arena.reset();
            let t = self.tempo.as_ref().unwrap();
            return Ok(Some(match t.snapshot(cuda, &mut self.arena, live, args) {
                Ok(s) => { self.tempo_snap = Some(s); Err("tempo_adw".into()) }
                Err(e) => Err(e),
            }));
        }
        if (function == "tempo_precip" || function == "tempo_radar") && self.tempo_small.is_some() && !skipped && step >= self.first {
            dev::sync(cuda)?;
            self.arena.reset();
            let t = self.tempo_small.as_ref().unwrap();
            return Ok(Some(match t.snapshot(cuda, &mut self.arena, live, function, args) {
                Ok(s) => { self.tempo_small_snap = Some(s); Err("tempo_ad_serial".into()) }
                Err(e) => Err(e),
            }));
        }
        if function == crate::tempo_order::FUNCTION && !skipped && step >= self.first && env("AD_DOT_ORDER_WITNESS").as_deref() == Some("1") {
            return Ok(Some(match crate::tempo_order::snapshot(cuda, live, args) {
                Ok(s) => { self.order_snap = Some(s); Err("tempo_order".into()) }
                Err(e) => Err(e),
            }));
        }
        // Input-free image constants: the window's rule is R::Zero on the written words (wrules::image_constant).
        if let Some(bytes) = crate::wrules::image_constant(function) {
            if !skipped && step >= self.first { return Ok(Some(self.constant_before(cuda, live, step, args, bytes))); }
        }
        let spec_ok = self.kernels.get(function).is_some_and(|k| k.unsupported.is_none());
        // L2: a native row binds a launch that has no dispatchable spec entry.
        if !skipped && !spec_ok && step >= self.first {
            if let Some(n) = self.native.as_mut() {
                if n.row(function).is_some() {
                    dev::sync(cuda)?;
                    self.arena.reset();
                    let p = n.prepare(cuda, &mut self.arena, live, function, args, grid, block);
                    return Ok(Some(p.map(Prep::Native)));
                }
            }
        }
        let why = match self.kernels.get(function) {
            _ if skipped => Some("skipped by AD_DOT_SKIP / AD_DOT_ONLY".to_owned()),
            None if crate::wrules::image_constant(function).is_some() =>
                Some("before window (input-free image constant: the window rule is zero, wrules::image_constant)".to_owned()),
            None => {
                let (class, binding) = rules::kernel_class(function, &self.spec_names);
                Some(format!("no derivative row; P8 table: {class}: {binding}"))
            },
            Some(k) => k.unsupported.clone().or_else(|| (step < self.first).then(|| "before window".into())),
        };
        if let Some(why) = why { self.untested(function, why); return Ok(None); }
        dev::sync(cuda)?;
        if let Some(p) = self.persistent.as_mut() {
            for a in live.values() {
                if p.contains_key(&a.id) { continue; }
                let mut address = 0u64;
                crate::cuda::checked("cuMemAlloc persistent shadow", (cuda.allocate)(&mut address, a.bytes))?;
                dev::zero(cuda, address, a.bytes)?;
                p.insert(a.id, deriv::Shadow { address, bytes: a.bytes, width: 4, lo: 0 });
                self.persistent_bytes += a.bytes;
            }
        }
        let kd = &self.kernels[function];
        let l = deriv::LaunchInfo { args, grid, block, shared };
        self.arena.reset();
        Ok(Some(kd.prepare_with(cuda, &mut self.arena, &l, live, self.width, self.persistent.as_ref()).map(Prep::Spec)))
    }

    /// After the forward launch was submitted: the launch dot, in the arena only.
    pub unsafe fn after(&mut self, cuda: &Cuda, live: &std::collections::HashMap<u64, submit::Allocation>, step: usize,
                        function: &str, args: &[submit::Argument], grid: [u32; 3], block: [u32; 3], shared: u32,
                        prepared: Result<Prep, String>) -> Result<(), String> {
        dev::sync(cuda)?;
        let seed = (step as u64) << 40 ^ self.index << 8 ^ 0xad6;
        let c = self.checked.entry((step, function.to_owned())).or_default(); *c += 1;
        let checks = *c <= self.checks_per_kernel;
        let row = if function == "tempo_warp_steps" && self.tempo.is_some() {
            match (self.tempo_snap.take(), prepared) {
                (Some(snap), _) => match self.tempo.as_ref().unwrap().launch_dot(cuda, &mut self.arena, &snap, seed, 1e-10, checks) {
                    Ok(r) => r, Err(e) => json!({"status":"ERROR","error":e}) },
                (None, Err(e)) => json!({"status":"ERROR","error":e}),
                (None, Ok(_)) => json!({"status":"ERROR","error":"tempo_adw: no snapshot"}),
            }
        } else if (function == "tempo_precip" || function == "tempo_radar") && self.tempo_small.is_some() {
            match (self.tempo_small_snap.take(), prepared) {
                (Some(snap), _) => match self.tempo_small.as_ref().unwrap().launch_dot(cuda, &mut self.arena, &snap, seed, 1e-10, checks) {
                    Ok(r) => r, Err(e) => json!({"status":"ERROR","error":e}) },
                (None, Err(e)) => json!({"status":"ERROR","error":e}),
                (None, Ok(_)) => json!({"status":"ERROR","error":"tempo_ad_serial: no snapshot"}),
            }
        } else if function == crate::tempo_order::FUNCTION && self.order_snap.is_some() {
            match (self.order_snap.take(), prepared) {
                (Some(snap), _) => match crate::tempo_order::launch_dot(cuda, &snap, seed, 1e-10, checks) {
                    Ok(r) => r, Err(e) => json!({"status":"ERROR","error":e}) },
                (None, Err(e)) => json!({"status":"ERROR","error":e}),
                (None, Ok(_)) => json!({"status":"ERROR","error":"tempo_order: no snapshot"}),
            }
        } else {
        match prepared {
            Ok(Prep::Spec(mut b)) => {
                let kd = &self.kernels[function];
                let l = deriv::LaunchInfo { args, grid, block, shared };
                let run = dot::Run { cuda, support: &self.support, live };
                match run.launch_dot(kd, &mut self.arena, &mut b, &l, seed, self.tol, checks) {
                    Ok(r) => r, Err(e) => json!({"status":"ERROR","error":e}) }
            },
            Ok(Prep::Constant(c)) => match self.constant_dot(cuda, &c, seed) {
                Ok(r) => r, Err(e) => json!({"status":"ERROR","error":e,"unit":"image-constant"}) },
            Ok(Prep::Native(mut p)) => {
                let n = self.native.as_mut().ok_or("native rows lost")?;
                match n.launch_dot(cuda, &self.support, &mut self.arena, &mut p, live, seed, checks) {
                    Ok(r) => r, Err(e) => json!({"status":"ERROR","error":e,"unit":p.unit,"row":"ad.scheme.rows/v1"}) }
            },
            Err(e) => {
                let unit = self.native.as_ref().and_then(|n| n.row(function)).map(|r| r.unit.clone());
                json!({"status":"ERROR","error":e,"unit":unit})
            },
        }
        };
        let status = row["status"].as_str().unwrap_or("?").to_owned();
        let mut line = row.clone();
        line["step"] = json!(step); line["function"] = json!(function); line["index"] = json!(self.index);
        writeln!(self.out, "{line}").map_err(|e| e.to_string())?;
        let e = self.summary.entry(function.to_owned()).or_insert_with(|| json!({"launches":0,"tested":0,"status":{},"max_rel":0.0,"max_norm_scaled":0.0,"tl_ms":0.0,"ad_ms":0.0}));
        e["launches"] = json!(e["launches"].as_u64().unwrap_or(0) + 1);
        if status != "INACTIVE" { e["tested"] = json!(e["tested"].as_u64().unwrap_or(0) + 1); }
        if let Some(u) = row["unit"].as_str() { e["unit"] = json!(u); }
        if row["row"].is_string() { e["row"] = row["row"].clone(); }
        if let Some(w) = row["why"].as_str() { e["why"] = json!(w); }
        let n = e["status"][&status].as_u64().unwrap_or(0) + 1; e["status"][&status] = json!(n);
        for (k, src) in [("max_rel", "rel"), ("max_norm_scaled", "norm_scaled")] {
            if let Some(v) = row[src].as_f64() { if v > e[k].as_f64().unwrap_or(0.0) || !v.is_finite() { e[k] = json!(if v.is_finite() { v } else { 1e300 }); } }
        }
        for k in ["tl_ms", "ad_ms"] { if let Some(v) = row[k].as_f64() { e[k] = json!(e[k].as_f64().unwrap_or(0.0) + v); } }
        for k in ["zero_tl_nonzero_words", "zero_ad_nonzero_words", "reverse_differing_words", "identity_differing_words"] {
            if let Some(v) = row[k].as_u64() { e[k] = json!(e[k].as_u64().unwrap_or(0) + v); }
        }
        if let Some(b) = row["shadow_bytes"].as_u64() { if b > e["max_shadow_bytes"].as_u64().unwrap_or(0) { e["max_shadow_bytes"] = json!(b); } }
        if status != "PASS" && status != "INACTIVE" && e["first_bad"].is_null() { e["first_bad"] = line.clone(); }
        // Stop only on a sticky device fault (CUDA 7xx: the context is lost); a refused call (e.g. status 1) is a row.
        if status == "ERROR" && row["error"].as_str().is_some_and(|x| x.contains("CUDA status 7")) {
            self.out.flush().ok();
            return Err(format!("step {step} {function}: {}", row["error"]));
        }
        Ok(())
    }

    /// Before an image-constant launch: the only memory the launch can address is its output, so a pattern that differs
    /// from the pre-state in every byte is written there (AD_DOT_POISON_SALT varies it per run). An input-free constant
    /// overwrites it: the post words must equal the recorded checked_value the runner compares right after the launch,
    /// and two runs with different poisons must give the same post bytes. That is the finite-difference truth
    /// F(x + d) - F(x) = 0 for any d on the launch's inputs.
    unsafe fn constant_before(&mut self, cuda: &Cuda, live: &std::collections::HashMap<u64, submit::Allocation>, step: usize,
                              args: &[submit::Argument], bytes: usize) -> Result<Prep, String> {
        let Some(submit::Argument::Device(s)) = args.first() else { return Err("image constant: output is not a device pointer".into()) };
        if args.len() != 1 { return Err(format!("image constant: {} arguments, expected the output only", args.len())); }
        let a = live.get(&s.allocation).ok_or_else(|| format!("image constant: unknown allocation {}", s.allocation))?;
        if s.offset + bytes > a.bytes {
            return Err(format!("image constant: {bytes} bytes at {} exceed allocation {} ({} bytes)", s.offset, a.id, a.bytes));
        }
        let address = a.address + s.offset as u64;
        dev::sync(cuda)?;
        let pre = dev::download(cuda, address, bytes)?;
        let salt = env("AD_DOT_POISON_SALT").and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
        let mut x = (((step as u64) << 40) ^ (self.index << 8) ^ 0xc0 ^ salt.wrapping_mul(0x9e37_79b9_7f4a_7c15)) | 1;
        let poison: Vec<u8> = pre.iter().map(|&b| {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            let p = (x >> 24) as u8;
            if p == b { !b } else { p }
        }).collect();
        dev::upload(cuda, address, &poison)?;
        dev::sync(cuda)?;
        Ok(Prep::Constant(ConstSnap { address, bytes, pre, poison }))
    }

    /// After an image-constant launch: the launch dot of its derivative on the float32 shadow words of the output
    /// (the window's R::Zero: TL overwrites the tangent with zero, the transpose stops the cotangent), on random seeds,
    /// the zero direction and a second reverse, plus the poison check's bytes.
    unsafe fn constant_dot(&mut self, cuda: &Cuda, c: &ConstSnap, seed: u64) -> Result<Value, String> {
        let post = dev::download(cuda, c.address, c.bytes)?;
        let n = c.bytes.div_ceil(4);
        let nb = n * 4;
        self.arena.reset();
        let t0 = std::time::Instant::now();
        let mut bufs = [0u64; 8];
        for b in bufs.iter_mut() { *b = self.arena.take(nb)?; }
        let [x, t, y, b, b2, z, tz, bz] = bufs;
        self.support.fill(cuda, x, nb, 4, seed)?;
        self.support.fill(cuda, y, nb, 4, seed ^ 0x5eed_0ad6)?;
        dev::zero(cuda, z, nb)?;
        // TL: the pre-launch tangent of the written words, then the launch's rule (overwrite by a constant: zero).
        dev::copy(cuda, t, x, nb)?;
        dev::zero(cuda, t, nb)?;
        dev::copy(cuda, tz, z, nb)?;
        dev::zero(cuda, tz, nb)?;
        dev::sync(cuda)?;
        let tl_ms = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = std::time::Instant::now();
        // AD: the cotangent of the written words, then the transpose (the cotangent stops; no input receives any).
        dev::copy(cuda, b, y, nb)?;
        dev::zero(cuda, b, nb)?;
        dev::copy(cuda, b2, y, nb)?;
        dev::zero(cuda, b2, nb)?;
        dev::copy(cuda, bz, z, nb)?;
        dev::zero(cuda, bz, nb)?;
        dev::sync(cuda)?;
        let ad_ms = t1.elapsed().as_secs_f64() * 1e3;
        let (xb, tb, yb, bb, b2b, tzb, bzb) = (dev::download(cuda, x, nb)?, dev::download(cuda, t, nb)?, dev::download(cuda, y, nb)?,
            dev::download(cuda, b, nb)?, dev::download(cuda, b2, nb)?, dev::download(cuda, tz, nb)?, dev::download(cuda, bz, nb)?);
        let (xv, tv, yv, bv) = (f32s(&xb), f32s(&tb), f32s(&yb), f32s(&bb));
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(p, q)| p * q).sum::<f64>();
        let norm = |a: &[f64]| dot(a, a).sqrt();
        let (lhs, rhs) = (dot(&yv, &tv), dot(&bv, &xv));
        let abs = (lhs - rhs).abs();
        let rel = if lhs == 0.0 && rhs == 0.0 { 0.0 } else { abs / lhs.abs().max(rhs.abs()) };
        let norm_scaled = abs / (norm(&yv) * norm(&tv) + norm(&bv) * norm(&xv)).max(f64::MIN_POSITIVE);
        let reverse_differing_words = bb.chunks_exact(4).zip(b2b.chunks_exact(4)).filter(|(p, q)| p != q).count();
        let nonzero = |v: &[u8]| v.chunks_exact(4).filter(|w| w.iter().any(|&q| q != 0)).count();
        let (zt, za) = (nonzero(&tzb), nonzero(&bzb));
        let finite = xv.iter().chain(&tv).chain(&yv).chain(&bv).all(|v| v.is_finite());
        let pass = abs == 0.0 && reverse_differing_words == 0 && zt == 0 && za == 0 && finite && norm(&xv) > 0.0 && norm(&yv) > 0.0;
        let words64: Vec<u64> = post.chunks_exact(8).map(|w| u64::from_le_bytes(w.try_into().unwrap())).collect();
        Ok(json!({"status": if pass { "PASS" } else { "FAIL" }, "unit":"image-constant", "row":"wrules.image_constant",
            "rule":"zero (TL: the written words' tangent := 0; AD: their cotangent stops, no input receives any)",
            "lhs":lhs, "rhs":rhs, "abs":abs, "rel":rel, "norm_scaled":norm_scaled, "width":4, "tol":0.0,
            "words":n, "x_norm":norm(&xv), "y_norm":norm(&yv), "tl_change_norm":norm(&tv), "ad_change_norm":norm(&bv),
            "gain_tl":norm(&tv) / norm(&xv), "zero_tl_nonzero_words":zt, "zero_ad_nonzero_words":za,
            "reverse_differing_words":reverse_differing_words, "tl_ms":tl_ms, "ad_ms":ad_ms, "shadow_bytes":8 * nb,
            "pre_hex":hex(&c.pre), "poison_hex":hex(&c.poison), "post_hex":hex(&post), "post_u64":words64,
            "post_equals_pre":post == c.pre,
            "post_bytes_equal_poison":post.iter().zip(&c.poison).filter(|(p, q)| p == q).count()}))
    }

    pub fn step_done(&mut self, step: usize) -> Result<(), String> {
        let row = json!({"event":"ad_dot_step","step":step,"elapsed_s":self.start.elapsed().as_secs_f64(),"arena_peak_bytes":self.arena.peak,
                         "persistent_shadow_bytes":self.persistent_bytes,
                         "scheme_scratch_bytes":self.native.as_ref().map(|n| n.scratch_bytes()).unwrap_or(0)});
        eprintln!("{row}");
        writeln!(self.out, "{row}").map_err(|e| e.to_string())?;
        self.out.flush().map_err(|e| e.to_string())?;
        self.step_rows.push(row);
        Ok(())
    }

    pub fn finish(&mut self) -> Result<(), String> {
        self.out.flush().map_err(|e| e.to_string())?;
        let mut totals: BTreeMap<String, u64> = BTreeMap::new();
        let mut untested: BTreeMap<String, u64> = BTreeMap::new();
        for (f, v) in &self.summary {
            if let Some(s) = v["status"].as_object() { for (k, n) in s { *totals.entry(k.clone()).or_default() += n.as_u64().unwrap(); } }
            else { untested.insert(f.clone(), v["launches"].as_u64().unwrap_or(0)); }
        }
        let report = json!({"event":"ad_dot_summary","width":self.width,"tol":self.tol,
            "native_tol":self.native.as_ref().map(|n| n.tol),"totals":totals,
            "untested_launches_by_function":untested,"commands":self.commands,"kernels":self.summary,
            "wall_s":self.start.elapsed().as_secs_f64(),"arena_peak_bytes":self.arena.peak,"steps":self.step_rows,
            "persistent_shadow_bytes":self.persistent_bytes,
            "scheme_scratch_bytes":self.native.as_ref().map(|n| n.scratch_bytes()).unwrap_or(0)});
        std::fs::write(self.out_path.with_extension("summary.json"), serde_json::to_vec_pretty(&report).unwrap()).map_err(|e| e.to_string())?;
        eprintln!("{}", json!({"event":"ad_dot_summary","totals":totals,"wall_s":report["wall_s"]}));
        Ok(())
    }
}
