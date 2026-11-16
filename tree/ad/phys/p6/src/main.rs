// SPDX-License-Identifier: Apache-2.0
//! ad-phys P6: whole-launch proof of the RUC sea-ice parent kernel
//! `oracle_ice_ruc_ice` on real hrrrv5 captures.
//!
//! * replay: the ORIGINAL captured image is launched on the captured pre-state;
//!   every word of every allocation must equal the captured post-state.
//! * aug identity: the derivative's augmented primal on the same columns must
//!   equal the original image's outputs bitwise.
//! * dot: <TL d, w> against <d, AD w> over every input and output slot.
//! * Taylor: TL against centred finite differences of the ORIGINAL image over
//!   eps 0.1 .. 1e-4, full-state random directions, with branch flips counted
//!   by forward-only probes of the augmented primal's visit records.
//! * cost: original launch vs augmented primal, TL and AD, median of 25.
//! Rust schedules and scores; CUDA does all field arithmetic. No Python.
use serde_json::Value;
use std::{ffi::{c_char, c_int, c_void, CString}, fs, path::{Path, PathBuf}, ptr, time::Instant};

#[link(name = "cuda")]
unsafe extern "C" {
    fn cuInit(f: u32) -> c_int;
    fn cuDeviceGet(d: *mut c_int, n: c_int) -> c_int;
    fn cuCtxCreate_v2(c: *mut *mut c_void, f: u32, d: c_int) -> c_int;
    fn cuCtxSynchronize() -> c_int;
    fn cuMemAlloc_v2(p: *mut u64, n: usize) -> c_int;
    fn cuMemsetD8_v2(p: u64, v: u8, n: usize) -> c_int;
    fn cuMemcpyHtoD_v2(d: u64, s: *const c_void, n: usize) -> c_int;
    fn cuMemcpyDtoH_v2(d: *mut c_void, s: u64, n: usize) -> c_int;
    fn cuModuleLoad(m: *mut *mut c_void, p: *const c_char) -> c_int;
    fn cuModuleGetFunction(f: *mut *mut c_void, m: *mut c_void, n: *const c_char) -> c_int;
    fn cuLaunchKernel(f: *mut c_void, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32, sh: u32,
                      s: *mut c_void, a: *mut *mut c_void, e: *mut *mut c_void) -> c_int;
    fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> c_int;
    fn cuModuleGetGlobal_v2(d: *mut u64, n: *mut usize, m: *mut c_void, name: *const c_char) -> c_int;
}
fn ck(rc: c_int, what: &str) -> Result<(), String> { if rc == 0 { Ok(()) } else { Err(format!("{what}: CUDA error {rc}")) } }
fn p<T>(x: &mut T) -> *mut c_void { (x as *mut T).cast() }
fn module(path: &Path) -> Result<*mut c_void, String> {
    let s = CString::new(path.to_str().ok_or("path")?).unwrap();
    let mut m = ptr::null_mut();
    ck(unsafe { cuModuleLoad(&mut m, s.as_ptr()) }, &format!("load {}", path.display()))?;
    Ok(m)
}
fn func(m: *mut c_void, name: &str) -> Result<*mut c_void, String> {
    let s = CString::new(name).unwrap();
    let mut f = ptr::null_mut();
    ck(unsafe { cuModuleGetFunction(&mut f, m, s.as_ptr()) }, &format!("symbol {name}"))?;
    Ok(f)
}
fn launch(f: *mut c_void, g: u32, b: u32, args: &mut [*mut c_void]) -> Result<(), String> {
    ck(unsafe { cuLaunchKernel(f, g, 1, 1, b, 1, 1, 0, ptr::null_mut(), args.as_mut_ptr(), ptr::null_mut()) }, "launch")
}
fn sync() -> Result<(), String> { ck(unsafe { cuCtxSynchronize() }, "sync") }
fn alloc(n: usize) -> Result<u64, String> { let mut a = 0; ck(unsafe { cuMemAlloc_v2(&mut a, n.max(4)) }, "alloc")?; Ok(a) }
fn put(a: u64, b: &[u8]) -> Result<(), String> { ck(unsafe { cuMemcpyHtoD_v2(a, b.as_ptr().cast(), b.len()) }, "upload") }
fn get(a: u64, n: usize) -> Result<Vec<u8>, String> { let mut v = vec![0u8; n]; ck(unsafe { cuMemcpyDtoH_v2(v.as_mut_ptr().cast(), a, n) }, "read")?; Ok(v) }
fn zero(a: u64, n: usize) -> Result<(), String> { ck(unsafe { cuMemsetD8_v2(a, 0, n) }, "zero") }
fn f32s(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }
fn bytes(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn i32s(b: &[u8]) -> Vec<i32> { b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect() }
fn sha(b: &[u8]) -> String { use sha2::Digest; format!("{:x}", sha2::Sha256::digest(b)) }

/// One captured launch: pre/post allocations, argument bindings, pointer tables.
struct Capture { dir: PathBuf, image: PathBuf, grid: u32, block: u32, args: Vec<Arg>, pre: Vec<Vec<u8>>,
                 post: Vec<Vec<u8>>, indirect: Vec<(usize, usize, usize, usize)>, nc: i32 }
enum Arg { Dev(usize, usize), Scalar(Vec<u8>) }
fn load_obj(dir: &Path, e: &Value) -> Result<Vec<u8>, String> {
    if let Some(ch) = e["chunks"].as_array() {
        let mut v = vec![];
        for c in ch { v.extend(fs::read(dir.join(c["path"].as_str().ok_or("chunk")?)).map_err(|x| x.to_string())?); }
        Ok(v)
    } else { fs::read(dir.join(e["path"].as_str().ok_or("path")?)).map_err(|x| x.to_string()) }
}
impl Capture {
    fn load(manifest: &Path) -> Result<Self, String> {
        let dir = manifest.parent().unwrap().to_path_buf();
        let m: Value = serde_json::from_slice(&fs::read(manifest).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        if m["kernel"] != "oracle_ice_ruc_ice" || m["complete"] != true { return Err("not a complete oracle_ice_ruc_ice capture".into()); }
        let mut args = vec![];
        for a in m["arguments"].as_array().ok_or("arguments")? {
            if a["kind"] == "scalar" {
                let h = a["value_hex"].as_str().ok_or("hex")?;
                args.push(Arg::Scalar((0..h.len() / 2).map(|i| u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap()).collect()));
            } else { args.push(Arg::Dev(a["allocation"].as_u64().ok_or("alloc")? as usize, a["offset"].as_u64().ok_or("off")? as usize)); }
        }
        let mut pre = vec![]; let mut post = vec![];
        for a in m["allocations"].as_array().ok_or("allocations")? {
            let x = load_obj(&dir, &a["pre"])?; let y = load_obj(&dir, &a["post"])?;
            if sha(&x) != a["pre"]["sha256"].as_str().unwrap_or("") || sha(&y) != a["post"]["sha256"].as_str().unwrap_or("") {
                return Err("capture object hash mismatch".into());
            }
            pre.push(x); post.push(y);
        }
        let indirect = m["indirect_pointers"].as_array().ok_or("indirect")?.iter().map(|i| (
            i["table_allocation"].as_u64().unwrap() as usize, i["table_offset"].as_u64().unwrap() as usize,
            i["target_allocation"].as_u64().unwrap() as usize, i["target_offset"].as_u64().unwrap() as usize)).collect();
        let nc = match &args[4] { Arg::Scalar(b) => i32::from_le_bytes(b[..4].try_into().unwrap()), _ => return Err("nc".into()) };
        let image = dir.join(m["image"]["path"].as_str().ok_or("image")?);
        if sha(&fs::read(&image).map_err(|e| e.to_string())?) != m["image"]["sha256"].as_str().unwrap_or("") { return Err("image hash".into()); }
        Ok(Self { dir, image, grid: m["geometry"]["grid"][0].as_u64().unwrap() as u32,
                  block: m["geometry"]["block"][0].as_u64().unwrap() as u32, args, pre, post, indirect, nc })
    }
    /// (allocation, byte offset) of pointer-table slot `s` of the kernel's ptrs argument.
    fn slot(&self, s: usize) -> (usize, usize) {
        let Arg::Dev(ta, toff) = self.args[0] else { unreachable!() };
        let e = self.indirect.iter().find(|x| x.0 == ta && x.1 == toff + 8 * s).expect("slot");
        (e.2, e.3)
    }
    fn f(&self, mem: &[Vec<u8>], s: usize, col: usize, lev: usize) -> f32 {
        let (a, o) = self.slot(s); let at = o + 4 * (lev * self.nc as usize + col);
        f32::from_le_bytes(mem[a][at..at + 4].try_into().unwrap())
    }
    fn set(&self, mem: &mut [Vec<u8>], s: usize, col: usize, lev: usize, v: f32) {
        let (a, o) = self.slot(s); let at = o + 4 * (lev * self.nc as usize + col);
        mem[a][at..at + 4].copy_from_slice(&v.to_le_bytes());
    }
    fn i(&self, mem: &[Vec<u8>], s: usize, col: usize) -> i32 {
        let (a, o) = self.slot(s); let at = o + 4 * col;
        i32::from_le_bytes(mem[a][at..at + 4].try_into().unwrap())
    }
    fn work(&self, mem: &[Vec<u8>], k: usize, col: usize) -> f32 {
        let Arg::Dev(a, o) = self.args[2] else { unreachable!() };
        let at = o + 4 * (k * self.nc as usize + col);
        f32::from_le_bytes(mem[a][at..at + 4].try_into().unwrap())
    }
}

/// The ORIGINAL captured image on fresh device copies of the allocations.
struct Original { f: *mut c_void, bases: Vec<u64>, sizes: Vec<usize> }
impl Original {
    fn new(c: &Capture) -> Result<Self, String> {
        let m = module(&c.image)?;
        let f = func(m, "oracle_ice_ruc_ice")?;
        // The forward module's saturation lookup table is a module global filled once by
        // ruc_tbq_init (as hex does at step 1); a freshly loaded image needs it too.
        launch(func(m, "ruc_tbq_init")?, 1, 1, &mut [])?; sync()?;
        let sizes: Vec<usize> = c.pre.iter().map(|x| x.len()).collect();
        let bases = sizes.iter().map(|&n| alloc(n)).collect::<Result<Vec<_>, _>>()?;
        Ok(Self { f, bases, sizes })
    }
    fn patched(&self, c: &Capture, mem: &[Vec<u8>]) -> Vec<Vec<u8>> {
        let mut m = mem.to_vec();
        for &(ta, to, a, o) in &c.indirect { m[ta][to..to + 8].copy_from_slice(&(self.bases[a] + o as u64).to_le_bytes()); }
        m
    }
    fn upload(&self, c: &Capture, mem: &[Vec<u8>]) -> Result<(), String> {
        for (i, b) in self.patched(c, mem).iter().enumerate() { put(self.bases[i], b)?; }
        Ok(())
    }
    fn launch(&self, c: &Capture) -> Result<(), String> {
        let mut vals: Vec<u64> = c.args.iter().map(|a| match a { Arg::Dev(al, o) => self.bases[*al] + *o as u64, Arg::Scalar(b) => {
            let mut x = [0u8; 8]; x[..b.len()].copy_from_slice(b); u64::from_le_bytes(x) } }).collect();
        let mut ap: Vec<*mut c_void> = vals.iter_mut().map(|v| p(v)).collect();
        launch(self.f, c.grid, c.block, &mut ap)
    }
    fn run(&self, c: &Capture, mem: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, String> {
        self.upload(c, mem)?; self.launch(c)?; sync()?;
        (0..self.bases.len()).map(|i| get(self.bases[i], self.sizes[i])).collect()
    }
}

// Helper scalar order (SiRucIceInput, RI_* enum) and where each lives in the forward.
const NAMES: [&str; 72] = ["dt","tabs","qvatm","qcatm","p8w_kms","p8w_1","z3d","rho","rainbl","rainncv","frzfrac","snowncv",
 "graupelncv","flqc","flhc","cp","rovcp","g0","lv","stbolt","glw","gsw","alb","albbck","soilt","soilt1","tsnav","snow","snowh",
 "snowc","qsfc","qsg","qvg","qcg","hfx","qfx","lh","dew","sfcrunoff","acrunoff","sfcevp","grdflx","snowfallac","acsnow","snom",
 "precipfr","rhosnf","chklowq","znt","z0","snoalb","emiss","xice","xice_threshold","local_smelt","local_snoh","local_snflx",
 "local_edir","local_ec","local_ett","local_sublim","local_sflx","local_smf","local_evapl","local_prcpl","local_budget",
 "local_runoff1","local_runoff2","local_infiltr","local_rsm","local_snweprint","local_snheiprint"];
const SLOTS: [usize; 54] = [3,19,20,21,18,18,17,22,12,10,13,9,8,28,29,45,46,47,48,49,23,24,30,34,51,52,53,54,55,56,57,58,59,60,
 61,62,63,64,65,66,67,68,69,70,71,16,15,26,31,32,33,25,43,44];
const LOCAL_WORK: [i32; 18] = [14,15,16,17,18,19,20,21,22,23,24,12,9,10,-1,-1,-1,-1];
/// Inputs held fixed in the Taylor direction: time step, physical constants, the
/// ice threshold (configuration), the p8w alias (same word as p8w_kms) and the
/// arena/stack scratch locals (written before read by the forward; checked below).
fn perturbed(k: usize) -> bool { k < 54 && !matches!(k, 0 | 5 | 15..=19 | 53) }

const ROW: usize = 1756;
struct Deriv { m: *mut c_void, n: usize, rows: u64, table: u64, pn: u64, pl: u64, inn: u64, il: u64, cn: u64, cl: u64,
    map: u64, cmap: u64, layer: u64, flayer: u64, rec: u64, count: u64, inp: u64, tape: u64, y: u64, yt: u64,
    active: u64, branch: u64, fail: u64, vt: u64, ot: u64, vin: u64, vout: u64, work: u64, fw: u64, outwork: u64,
    tcount: u64, flags: [i32; 5],
    din: Vec<u64>, dtso: u64, dzs: u64, dtab: u64, dout: Vec<u64>, dtso_out: u64 }
impl Deriv {
    fn new(path: &Path, n: usize, table: &[u8], flags: [i32; 5]) -> Result<Self, String> {
        let m = module(path)?;
        // G4: the augmented primal's fragment tapes live in a per-column frame (si_frame_bytes each), not on the stack.
        let global = |name: &str| -> Result<(u64, usize), String> { let c = CString::new(name).unwrap(); let (mut a, mut z) = (0u64, 0usize);
            ck(unsafe { cuModuleGetGlobal_v2(&mut a, &mut z, m, c.as_ptr()) }, &format!("global {name}"))?; Ok((a, z)) };
        let fb = u64::from_le_bytes(get(global("si_frame_bytes")?.0, 8)?[..8].try_into().unwrap());
        let frame = alloc(n * fb as usize)?;
        put(global("si_frame_base")?.0, &frame.to_le_bytes())?;
        put(global("si_frame_stride")?.0, &fb.to_le_bytes())?;
        let mut din = vec![]; for _ in 0..72 { din.push(alloc(n * 4)?); }
        let mut dout = vec![]; for _ in 0..72 { dout.push(alloc(n * 4)?); }
        let t = alloc(5001 * 4)?; put(t, table)?;
        Ok(Self { m, n, rows: alloc(n * ROW)?, table: t, pn: alloc(n * 4096 * 32)?, pl: alloc(n * 2048 * 12)?,
            inn: alloc(n * 4096 * 32)?, il: alloc(n * 1024 * 12)?, cn: alloc(n * 2048 * 32)?, cl: alloc(n * 512 * 12)?,
            map: alloc(n * 4096 * 4)?, cmap: alloc(n * 2048 * 4)?, layer: alloc(n * 145 * 4)?, flayer: alloc(n * 59 * 4)?,
            rec: alloc(n * 64 * 4)?, count: alloc(n * 2 * 4)?, inp: alloc(n * 1024)?, tape: alloc(n * 2048)?,
            y: alloc(n * 72 * 4)?, yt: alloc(n * 9 * 4)?, active: alloc(n * 4)?, branch: alloc(256 * 2 * 4)?, fail: alloc(4)?,
            vt: alloc(74 * 8)?, ot: alloc(74 * 8)?, vin: alloc(79 * 32)?, vout: alloc(73 * 32)?, work: alloc(n * 4096 * 8)?,
            fw: alloc(n * 5091 * 8)?, outwork: alloc(n * 81 * 8)?, tcount: alloc(n * 6 * 4)?, flags,
            din, dtso: alloc(n * 36)?, dzs: alloc(n * 36)?, dtab: alloc(5001 * 4)?, dout, dtso_out: alloc(n * 36)? })
    }
    fn failure(&self) -> Result<(), String> {
        let v = i32::from_le_bytes(get(self.fail, 4)?[..4].try_into().unwrap());
        if v == 0 { Ok(()) } else { Err(format!("derivative refusal code {v}")) }
    }
    /// Augmented primal on the given rows; returns (scalars, tso, active, records, counts, branch histogram).
    fn aug(&self, rows: &[u8]) -> Result<(Vec<f32>, Vec<f32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>), String> {
        put(self.rows, rows)?; zero(self.fail, 4)?; zero(self.branch, 256 * 8)?; zero(self.rec, self.n * 256)?;
        self.construct_aug()?; sync()?; self.failure()?;
        let cnt = i32s(&get(self.count, self.n * 8)?);
        let mut rec = i32s(&get(self.rec, self.n * 256)?);
        // Only the visited prefix of each visit record is meaningful.
        for j in 0..self.n { for s in 0..2 { let used = cnt[j * 2 + s].max(0) as usize;
            for q in used..32 { rec[j * 64 + s * 32 + q] = i32::MIN; } } }
        Ok((f32s(&get(self.y, self.n * 288)?), f32s(&get(self.yt, self.n * 36)?), i32s(&get(self.active, self.n * 4)?),
            rec, cnt, i32s(&get(self.branch, 256 * 8)?)))
    }
    fn construct_aug(&self) -> Result<(), String> {
        let g = ((self.n + 31) / 32) as u32;
        let (mut rows, mut table, mut n, mut nz) = (self.rows, self.table, self.n as i32, 9i32);
        let (mut pn, mut pl, mut inn, mut il, mut cn, mut cl, mut map, mut cmap, mut layer, mut flayer, mut rec, mut count, mut inp, mut tape) =
            (self.pn, self.pl, self.inn, self.il, self.cn, self.cl, self.map, self.cmap, self.layer, self.flayer, self.rec, self.count, self.inp, self.tape);
        let [mut f0, mut f1, mut f2, mut f3, mut f4] = self.flags;
        launch(func(self.m, "p6_construct")?, g, 32, &mut [p(&mut rows), p(&mut table), p(&mut n), p(&mut pn), p(&mut pl), p(&mut inn), p(&mut il),
            p(&mut cn), p(&mut cl), p(&mut map), p(&mut cmap), p(&mut layer), p(&mut flayer), p(&mut rec), p(&mut count), p(&mut inp), p(&mut tape),
            p(&mut f0), p(&mut f1), p(&mut f2), p(&mut f3), p(&mut f4)])?;
        let (mut y, mut yt, mut active, mut branch, mut fail) = (self.y, self.yt, self.active, self.branch, self.fail);
        launch(func(self.m, "seaice_ice_ruc_ice__aug")?, g, 32, &mut [p(&mut inp), p(&mut tape), p(&mut n), p(&mut nz), p(&mut y), p(&mut yt),
            p(&mut active), p(&mut branch), p(&mut fail)])
    }
    fn tables(&self, input: &[u64], output: &[u64]) -> Result<(), String> {
        put(self.vt, &input.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        put(self.ot, &output.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        let views = func(self.m, "ruc_launch_views")?;
        let (mut s, mut d, mut o) = (self.vt, self.vin, 0i32);
        launch(views, 3, 32, &mut [p(&mut s), p(&mut d), p(&mut o)])?;
        let (mut s, mut d, mut o) = (self.ot, self.vout, 1i32);
        launch(views, 3, 32, &mut [p(&mut s), p(&mut d), p(&mut o)])
    }
    fn in_table(&self) -> Vec<u64> {
        let mut v: Vec<u64> = (0..72).filter(|&k| k != 5).map(|k| self.din[k]).collect();
        v.push(self.dtso); v.push(self.dzs); v.push(self.dtab); v
    }
    fn out_table(&self) -> Vec<u64> { let mut v = self.dout.clone(); v.push(self.dtso_out); v }
    fn sweep(&self, reverse: bool) -> Result<(), String> {
        let g = ((self.n + 31) / 32) as u32;
        let (mut tape, mut n, mut nz, mut nd, mut vin, mut vout, mut work, mut gs, mut fw, mut fs, mut ow, mut os, mut fail) =
            (self.tape, self.n as i32, 9i32, 14i32, self.vin, self.vout, self.work, 4096i32, self.fw, 5091i32, self.outwork, 81i32, self.fail);
        let entry = if reverse { "seaice_ice_ruc_ice__ad" } else { "seaice_ice_ruc_ice__tl" };
        launch(func(self.m, entry)?, g, 32, &mut [p(&mut tape), p(&mut n), p(&mut nz), p(&mut nd), p(&mut vin), p(&mut vout), p(&mut work),
            p(&mut gs), p(&mut fw), p(&mut fs), p(&mut ow), p(&mut os), p(&mut fail)])?;
        if reverse {
            let (mut width, mut off, mut tab) = (5001i32, 90i32, self.dtab);
            launch(func(self.m, "seaice_table_reduce__ad")?, (5001 + 127) / 128, 128, &mut [p(&mut fw), p(&mut n), p(&mut width), p(&mut fs),
                p(&mut off), p(&mut tab), p(&mut fail)])?;
        }
        Ok(())
    }
    /// TL of direction (per helper input k, column-major n) -> per output k (72) and tso.
    fn tl(&self, d: &[Vec<f32>], dtso: &[f32]) -> Result<(Vec<Vec<f32>>, Vec<f32>), String> {
        zero(self.fail, 4)?;
        for k in 0..72 { put(self.din[k], &bytes(&d[k]))?; zero(self.dout[k], self.n * 4)?; }
        put(self.dtso, &bytes(dtso))?; zero(self.dzs, self.n * 36)?; zero(self.dtab, 5001 * 4)?; zero(self.dtso_out, self.n * 36)?;
        self.tables(&self.in_table(), &self.out_table())?; self.sweep(false)?; sync()?; self.failure()?;
        let out = (0..72).map(|k| get(self.dout[k], self.n * 4).map(|b| f32s(&b))).collect::<Result<Vec<_>, _>>()?;
        Ok((out, f32s(&get(self.dtso_out, self.n * 36)?)))
    }
    /// AD of output seeds -> input cotangents (scalars, tso, zs, table).
    fn ad(&self, w: &[Vec<f32>], wtso: &[f32]) -> Result<(Vec<Vec<f32>>, Vec<f32>, Vec<f32>, Vec<f32>), String> {
        zero(self.fail, 4)?;
        for k in 0..72 { put(self.dout[k], &bytes(&w[k]))?; zero(self.din[k], self.n * 4)?; }
        put(self.dtso_out, &bytes(wtso))?; zero(self.dtso, self.n * 36)?; zero(self.dzs, self.n * 36)?; zero(self.dtab, 5001 * 4)?;
        self.tables(&self.in_table(), &self.out_table())?; self.sweep(true)?; sync()?; self.failure()?;
        let inp = (0..72).map(|k| get(self.din[k], self.n * 4).map(|b| f32s(&b))).collect::<Result<Vec<_>, _>>()?;
        Ok((inp, f32s(&get(self.dtso, self.n * 36)?), f32s(&get(self.dzs, self.n * 36)?), f32s(&get(self.dtab, 5001 * 4)?)))
    }
}

struct Rng(u64);
impl Rng { fn next(&mut self) -> f64 { self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15); let mut z = self.0;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9); z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0 } }

fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }

fn main() {
    if let Err(e) = run() { eprintln!("P6_ERROR {e}"); std::process::exit(1); }
}
fn run() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let arg = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    let manifest = PathBuf::from(arg("--capture").ok_or("--capture")?);
    let deriv = PathBuf::from(arg("--deriv").ok_or("--deriv")?);
    let table = fs::read(arg("--table").ok_or("--table")?).map_err(|e| e.to_string())?;
    let seeds: Vec<u64> = arg("--seeds").unwrap_or("1,2,3".into()).split(',').map(|s| s.parse().unwrap()).collect();
    let t0 = Instant::now();
    ck(unsafe { cuInit(0) }, "init")?;
    let mut dev = 0; ck(unsafe { cuDeviceGet(&mut dev, 0) }, "device")?;
    let mut ctx = ptr::null_mut(); ck(unsafe { cuCtxCreate_v2(&mut ctx, 0, dev) }, "context")?;
    let c = Capture::load(&manifest)?;
    println!("P6_CAPTURE dir={} nc={} grid={}x{} allocations={} image={}", c.dir.display(), c.nc, c.grid, c.block, c.pre.len(), c.image.display());
    let nc = c.nc as usize;

    // 1. Original image replay identity (all words of every allocation), twice.
    let orig = Original::new(&c)?;
    for rep in 0..2 {
        let out = orig.run(&c, &c.pre)?;
        let expect = orig.patched(&c, &c.post);
        let (mut words, mut diff) = (0usize, 0usize);
        for (i, (x, y)) in out.iter().zip(&expect).enumerate() {
            let mut di = 0; let mut first = None;
            for (w, (u, v)) in x.chunks(4).zip(y.chunks(4)).enumerate() { words += 1; if u != v { di += 1; first.get_or_insert(w); } }
            if di > 0 { println!("P6_REPLAY_DIFF allocation={i} words={} differing={di} first_word={:?}", x.len() / 4, first); }
            diff += di;
        }
        println!("P6_REPLAY run={rep} words={words} differing={diff}");
        if diff != 0 { return Err("original replay differs from captured post-state".into()); }
    }

    // 2. Active sea-ice columns and their helper rows.
    let thr = c.f(&c.pre, 44, 0, 0);
    let cols: Vec<usize> = (0..nc).filter(|&i| c.f(&c.pre, 43, i, 0) >= thr).collect();
    let n = cols.len();
    let ktau = c.i(&c.pre, 4, 0); let restart = c.i(&c.pre, 6, 0); let cycling = c.i(&c.pre, 7, 0);
    let cold = (ktau == 1 && restart == 0 && cycling == 0) as i32;
    let flags = [c.i(&c.pre, 42, 0), cold, c.i(&c.pre, 14, 0), (c.i(&c.pre, 93, 0) != 0) as i32, (c.i(&c.pre, 40, 0) != 0) as i32];
    println!("P6_COLUMNS active={n} of {nc} xice_threshold={thr} ktau={ktau} restart={restart} cycling={cycling} flags(isice,cold,frpcpn,graupel,myj)={flags:?}");
    let build_rows = |mem: &[Vec<u8>]| -> Vec<u8> {
        let mut out = Vec::with_capacity(n * ROW);
        for &col in &cols {
            let mut x = [0f32; 72];
            for (k, &s) in SLOTS.iter().enumerate() { x[k] = c.f(mem, s, col, 0); }
            for (k, &w) in LOCAL_WORK.iter().enumerate() { if w >= 0 { x[54 + k] = c.work(mem, w as usize, col); } }
            let mut r = bytes(&x);
            r.extend(bytes(&(0..9).map(|j| c.f(mem, 50, col, j)).collect::<Vec<_>>()));
            r.extend(bytes(&(0..9).map(|j| c.f(mem, 11, col, j)).collect::<Vec<_>>()));
            r.resize(ROW - 8, 0); r.extend((col as i32).to_le_bytes()); r.extend(1i32.to_le_bytes());
            out.extend(r);
        }
        out
    };
    let outputs_of = |mem: &[Vec<u8>]| -> (Vec<Vec<f32>>, Vec<f32>) {
        let s = (0..54).map(|k| cols.iter().map(|&col| c.f(mem, SLOTS[k], col, 0)).collect()).collect();
        let t = cols.iter().flat_map(|&col| (0..9).map(move |j| (col, j))).map(|(col, j)| c.f(mem, 50, col, j)).collect();
        (s, t)
    };
    let d = Deriv::new(&deriv, n, &table, flags)?;
    let base_rows = build_rows(&c.pre);

    // 3. Augmented primal identity against the original image's outputs.
    let post = orig.run(&c, &c.pre)?;
    let (ys, yt, act, rec0, cnt0, hist0) = d.aug(&base_rows)?;
    let (os, ot) = outputs_of(&post);
    let mut diff = 0; let mut words = 0;
    for k in 0..54 { for j in 0..n { words += 1; if ys[j * 72 + k].to_bits() != os[k][j].to_bits() { diff += 1;
        if diff <= 10 { println!("P6_AUG_DIFF field={} col={} aug={:e} orig={:e}", NAMES[k], cols[j], ys[j * 72 + k], os[k][j]); } } } }
    for j in 0..n * 9 { words += 1; if yt[j].to_bits() != ot[j].to_bits() { diff += 1; } }
    for (k, &w) in LOCAL_WORK.iter().enumerate() { if w >= 0 { for (j, &col) in cols.iter().enumerate() { words += 1;
        if ys[j * 72 + 54 + k].to_bits() != c.work(&post, w as usize, col).to_bits() { diff += 1; } } } }
    println!("P6_AUG_IDENTITY columns={n} words={words} differing={diff} active={}", act.iter().filter(|&&x| x != 0).count());
    if diff != 0 { return Err("augmented primal differs from the original image".into()); }
    // Scratch locals: the forward must not read them. Fill their pre-values with a finite junk value (the record refuses
    // non-finite inputs) and require bitwise-equal outputs.
    {
        let mut poisoned = base_rows.clone();
        for j in 0..n { for k in 54..72 { poisoned[j * ROW + 4 * k..j * ROW + 4 * k + 4].copy_from_slice(&(-7777.25f32).to_le_bytes()); } }
        let (ps, pt, ..) = d.aug(&poisoned)?;
        let bad = (0..n).flat_map(|j| (0..54).map(move |k| (j, k))).filter(|&(j, k)| ps[j * 72 + k].to_bits() != ys[j * 72 + k].to_bits()).count()
            + pt.iter().zip(&yt).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        println!("P6_LOCALS_POISON outputs_differing={bad} (scratch locals are written before read: {})", bad == 0);
    }
    let tc = { let mut tape = d.tape; let mut nn = n as i32; let mut out = d.tcount;
        launch(func(d.m, "p6_tape_counts")?, ((n + 31) / 32) as u32, 32, &mut [p(&mut tape), p(&mut nn), p(&mut out)])?; sync()?;
        i32s(&get(d.tcount, n * 24)?) };
    let used = |k: usize| (0..n).map(|j| tc[j * 6 + k]).max().unwrap_or(0);
    println!("P6_TAPE max_nodes parent={} icetmp={} child={} max_leaves parent={} icetmp={} child={} bytes_per_column_used<={} allocated_per_column={}",
        used(0), used(2), used(4), used(1), used(3), used(5), (used(0) + used(2) + used(4)) * 32 + (used(1) + used(3) + used(5)) * 12 + 145 * 4 + 59 * 4 + 64 * 4,
        4096 * 32 + 2048 * 12 + 4096 * 32 + 1024 * 12 + 2048 * 32 + 512 * 12 + 4096 * 4 + 2048 * 4 + 145 * 4 + 59 * 4 + 64 * 4 + 8 + 1024 + 2048);

    // Base inputs as helper vectors.
    let xb: Vec<Vec<f32>> = (0..72).map(|k| (0..n).map(|j| f32::from_le_bytes(base_rows[j * ROW + 4 * k..j * ROW + 4 * k + 4].try_into().unwrap())).collect()).collect();
    let tsob: Vec<f32> = (0..n).flat_map(|j| (0..9).map(move |l| (j, l))).map(|(j, l)| f32::from_le_bytes(base_rows[j * ROW + 288 + 4 * l..j * ROW + 292 + 4 * l].try_into().unwrap())).collect();

    // 4. Dot test (all inputs active incl. constants, scratch locals, tso, zs, table).
    let mut dot_pass = true; let mut worst = 0f64;
    for &seed in &seeds {
        let mut r = Rng(seed * 7919 + 17);
        let dv: Vec<Vec<f32>> = (0..72).map(|k| if k == 5 { vec![0.0; n] } else { (0..n).map(|_| r.next() as f32).collect() }).collect();
        let dt: Vec<f32> = (0..n * 9).map(|_| r.next() as f32).collect();
        let w: Vec<Vec<f32>> = (0..72).map(|_| (0..n).map(|_| r.next() as f32).collect()).collect();
        let wt: Vec<f32> = (0..n * 9).map(|_| r.next() as f32).collect();
        let (tv, tt) = d.tl(&dv, &dt)?;
        let (av, at, _az, _atab) = d.ad(&w, &wt)?;
        let mut lhs = 0f64; let mut lmag = 0f64;
        for k in 0..72 { for j in 0..n { lhs += tv[k][j] as f64 * w[k][j] as f64; lmag += (tv[k][j] as f64 * w[k][j] as f64).abs(); } }
        for j in 0..n * 9 { lhs += tt[j] as f64 * wt[j] as f64; lmag += (tt[j] as f64 * wt[j] as f64).abs(); }
        let mut rhs = 0f64; let mut rmag = 0f64;
        for k in 0..72 { if k == 5 { continue; } for j in 0..n { rhs += dv[k][j] as f64 * av[k][j] as f64; rmag += (dv[k][j] as f64 * av[k][j] as f64).abs(); } }
        for j in 0..n * 9 { rhs += dt[j] as f64 * at[j] as f64; rmag += (dt[j] as f64 * at[j] as f64).abs(); }
        let rel = (lhs - rhs).abs() / lhs.abs().max(rhs.abs()).max(1e-300);
        let ns = (lhs - rhs).abs() / lmag.max(rmag).max(1e-300);
        worst = worst.max(ns); let pass = ns <= 2e-5; dot_pass &= pass;
        println!("P6_DOT seed={seed} lhs={lhs:.10e} rhs={rhs:.10e} relative={rel:.3e} norm_scaled={ns:.3e} pass={pass}");
    }
    { let (tv, tt) = d.tl(&vec![vec![0.0; n]; 72], &vec![0.0; n * 9])?;
      let nz = tv.iter().flatten().chain(tt.iter()).filter(|x| **x != 0.0).count();
      println!("P6_ZERO_SEED nonzero_outputs={nz}"); if nz != 0 { dot_pass = false; } }
    { let mut r = Rng(99); let dv: Vec<Vec<f32>> = (0..72).map(|k| if k == 5 { vec![0.0; n] } else { (0..n).map(|_| r.next() as f32).collect() }).collect();
      let dt: Vec<f32> = (0..n * 9).map(|_| r.next() as f32).collect();
      let (a1, b1) = d.tl(&dv, &dt)?; let (a2, b2) = d.tl(&dv, &dt)?;
      let w = a1.clone(); let (c1, e1, ..) = d.ad(&w, &b1)?; let (c2, e2, ..) = d.ad(&w, &b1)?;
      let same = a1 == a2 && b1 == b2 && c1.iter().flatten().zip(c2.iter().flatten()).all(|(x, y)| x.to_bits() == y.to_bits())
          && e1.iter().zip(&e2).all(|(x, y)| x.to_bits() == y.to_bits());
      println!("P6_DETERMINISM two_tl_two_ad_bitwise={same}"); }
    println!("P6_DOT_SUMMARY seeds={} worst_norm_scaled={worst:.3e} pass={dot_pass}", seeds.len());

    // 5. Taylor against the ORIGINAL image, full-state random directions.
    let eps_ladder = [0.1f64, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001];
    let refs: Vec<f64> = (0..54).map(|k| { let m = xb[k].iter().map(|v| (*v as f64).abs()).sum::<f64>() / n as f64; m }).collect();
    let out_scale: Vec<f64> = (0..54).map(|k| { let v = &os[k]; (v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / n as f64).sqrt() }).collect();
    let tso_scale = (ot.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / ot.len() as f64).sqrt();
    let mut best_all = f64::INFINITY;
    let taylor_seeds: Vec<u64> = if a.iter().any(|x| x == "--skip-taylor") { vec![] } else { seeds.clone() };
    for &seed in &taylor_seeds {
        let mut r = Rng(seed * 104729 + 3);
        // d_i = u_i * |x_i| (state-scaled, zero stays zero: no sign change of bounded fields).
        let dv: Vec<Vec<f32>> = (0..72).map(|k| (0..n).map(|j| if perturbed(k) { (r.next() * (xb[k][j] as f64).abs()) as f32 } else { 0.0 }).collect()).collect();
        let mut dt: Vec<f32> = (0..n * 9).map(|j| (r.next() * (tsob[j] as f64).abs()) as f32).collect();
        // --mask-ties: elements sitting exactly on the ice cap tso = min(271.4, tso) (the cap applied on the
        // previous step) are a kink AT the base point; a centred difference straddles it by construction.
        // Masking them grades the tangent where it is differentiable; the count is reported.
        if a.iter().any(|x| x == "--mask-ties") {
            let ties = (0..n * 9).filter(|&j| tsob[j] == 271.4f32).count();
            for j in 0..n * 9 { if tsob[j] == 271.4f32 { dt[j] = 0.0; } }
            println!("P6_MASK seed={seed} tso_elements_at_ice_cap={ties}/{} (direction zeroed there)", n * 9);
        }
        let (tv, tt) = d.tl(&dv, &dt)?;
        let mut curve = vec![];
        // Score FD (centred, plus-sided, minus-sided) against TL, optionally only on branch-stable columns.
        let score = |fd_s: &Vec<Vec<f64>>, fd_t: &Vec<f64>, keep: &dyn Fn(usize) -> bool| -> (f64, f64, Vec<(f64, &'static str)>) {
            let (mut e2, mut t2) = (0f64, 0f64); let mut per = vec![];
            for k in 0..54 { if k == 5 || out_scale[k] == 0.0 { continue; }
                let (mut ek, mut tk) = (0f64, 0f64);
                for j in 0..n { if keep(j) { ek += (fd_s[k][j] - tv[k][j] as f64).powi(2); tk += (tv[k][j] as f64).powi(2); } }
                e2 += ek / out_scale[k].powi(2); t2 += tk / out_scale[k].powi(2);
                if tk > 0.0 { per.push(((ek / tk).sqrt(), NAMES[k])); } }
            let (mut ek, mut tk) = (0f64, 0f64);
            for j in 0..n * 9 { if keep(j / 9) { ek += (fd_t[j] - tt[j] as f64).powi(2); tk += (tt[j] as f64).powi(2); } }
            e2 += ek / tso_scale.powi(2); t2 += tk / tso_scale.powi(2);
            if tk > 0.0 { per.push(((ek / tk).sqrt(), "tso")); }
            per.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            ((e2 / t2.max(1e-300)).sqrt(), t2, per)
        };
        for &eps in &eps_ladder {
            let mut fd_s = vec![vec![0f64; n]; 54]; let mut fd_t = vec![0f64; n * 9];
            let mut flips_cols = 0; let mut flips = 0; let mut hist_change = 0; let mut act_change = 0;
            let mut stable = vec![true; n];
            let mut samples = vec![];
            for sign in [1.0f64, -1.0] {
                let mut mem = c.pre.clone();
                for (j, &col) in cols.iter().enumerate() {
                    for k in 0..54 { if perturbed(k) { c.set(&mut mem, SLOTS[k], col, 0, (xb[k][j] as f64 + sign * eps * dv[k][j] as f64) as f32); } }
                    for l in 0..9 { c.set(&mut mem, 50, col, l, (tsob[j * 9 + l] as f64 + sign * eps * dt[j * 9 + l] as f64) as f32); }
                }
                let out = orig.run(&c, &mem)?;
                // Forward-only branch probe at the same perturbed state (values unused).
                let (_, _, pact, prec, pcnt, phist) = d.aug(&build_rows(&mem))?;
                for j in 0..n {
                    let ch = (0..64).filter(|&q| prec[j * 64 + q] != rec0[j * 64 + q]).count() + (0..2).filter(|&q| pcnt[j * 2 + q] != cnt0[j * 2 + q]).count();
                    if ch > 0 { flips_cols += 1; flips += ch; stable[j] = false; }
                    if pact[j] != act[j] { act_change += 1; }
                }
                hist_change += phist.iter().zip(&hist0).filter(|(a, b)| a != b).count();
                if eps == 1e-4 || eps == 1e-2 {
                    let bins: Vec<String> = (0..512).filter(|&q| phist[q] != hist0[q]).map(|q| format!("site{}arm{}:{}->{}", q / 2, q % 2, hist0[q], phist[q])).collect();
                    let bare = (0..n).map(|j| (0..32).filter(|&q| prec[j * 64 + q] != rec0[j * 64 + q]).count()).sum::<usize>();
                    let snow = (0..n).map(|j| (32..64).filter(|&q| prec[j * 64 + q] != rec0[j * 64 + q]).count()).sum::<usize>();
                    let nb = (0..n).filter(|&j| pcnt[j * 2] != cnt0[j * 2]).count(); let ns = (0..n).filter(|&j| pcnt[j * 2 + 1] != cnt0[j * 2 + 1]).count();
                    println!("P6_BRANCH_DETAIL seed={seed} eps={eps:e} sign={sign} histogram={bins:?} bare_visit_entries={bare} snow_visit_entries={snow} bare_visit_count_changes={nb} snow_visit_count_changes={ns}");
                }
                samples.push(outputs_of(&out));
            }
            let ((ps, pt), (ms, mt)) = (&samples[0], &samples[1]);
            for k in 0..54 { for j in 0..n { fd_s[k][j] = (ps[k][j] as f64 - ms[k][j] as f64) / (2.0 * eps); } }
            for j in 0..n * 9 { fd_t[j] = (pt[j] as f64 - mt[j] as f64) / (2.0 * eps); }
            // Norm-scaled vector error: each output field divided by its own state scale.
            let all = |_: usize| true; let st = |j: usize| stable[j];
            let (rel, _, per) = score(&fd_s, &fd_t, &all);
            let (rel_st, _, _) = score(&fd_s, &fd_t, &st);
            // One-sided differences against the unperturbed original outputs (os/ot).
            let one = |s: &Vec<Vec<f32>>, t: &Vec<f32>, sg: f64| -> (Vec<Vec<f64>>, Vec<f64>) {
                ((0..54).map(|k| (0..n).map(|j| sg * (s[k][j] as f64 - os[k][j] as f64) / eps).collect()).collect(),
                 (0..n * 9).map(|j| sg * (t[j] as f64 - ot[j] as f64) / eps).collect()) };
            let (pls, plt) = one(ps, pt, 1.0); let (mns, mnt) = one(ms, mt, -1.0);
            let (rel_p, _, _) = score(&pls, &plt, &all); let (rel_m, _, _) = score(&mns, &mnt, &all);
            // Per column: the better one-sided difference (the side the executed branch belongs to).
            let mut best_s = pls.clone(); let mut best_t = plt.clone();
            for j in 0..n {
                let err = |s: &Vec<Vec<f64>>, t: &Vec<f64>| -> f64 {
                    (0..54).filter(|&k| k != 5 && out_scale[k] > 0.0).map(|k| ((s[k][j] - tv[k][j] as f64) / out_scale[k]).powi(2)).sum::<f64>()
                    + (0..9).map(|l| ((t[j * 9 + l] - tt[j * 9 + l] as f64) / tso_scale).powi(2)).sum::<f64>() };
                if err(&mns, &mnt) < err(&pls, &plt) { for k in 0..54 { best_s[k][j] = mns[k][j]; } for l in 0..9 { best_t[j * 9 + l] = mnt[j * 9 + l]; } }
            }
            let (rel_b, _, per_b) = score(&best_s, &best_t, &all);
            println!("P6_TAYLOR seed={seed} eps={eps:e} centred={rel:.3e} centred_stable_cols={rel_st:.3e} ({} cols) plus_sided={rel_p:.3e} minus_sided={rel_m:.3e} best_side_per_column={rel_b:.3e} worst_centred={} ({:.2e}) worst_best_side={} ({:.2e}) branch_changed_samples={flips_cols}/{} record_entries_changed={flips} histogram_bins_changed={hist_change} active_changes={act_change}",
                stable.iter().filter(|&&x| x).count(), per[0].1, per[0].0, per_b[0].1, per_b[0].0, 2 * n);
            if eps == 1e-4 || eps == 1e-2 {
                println!("P6_TAYLOR_FIELDS seed={seed} eps={eps:e} centred_top={:?}", per.iter().take(12).map(|(e, f)| format!("{f}:{e:.2e}")).collect::<Vec<_>>());
                println!("P6_TAYLOR_FIELDS seed={seed} eps={eps:e} best_side_top={:?}", per_b.iter().take(12).map(|(e, f)| format!("{f}:{e:.2e}")).collect::<Vec<_>>());
            }
            curve.push(rel);
        }
        let best = curve.iter().cloned().fold(f64::INFINITY, f64::min);
        best_all = best_all.min(best);
        println!("P6_TAYLOR_CURVE seed={seed} best={best:.3e} curve={:?}", curve.iter().map(|x| format!("{x:.2e}")).collect::<Vec<_>>());
    }

    // 5c. Independent check: per-column branch fingerprint. The visit records cover only part of the
    // switches (the per-layer ice cap, for one, is only in the launch-wide branch histogram), so a column
    // whose records did not change can still have flipped a switch. Here each column is re-run ALONE through
    // the augmented primal (n = 1, so the branch histogram is that column's own) at the base point and at
    // +eps and -eps; a column is branch-stable only if records, counts and its own histogram are unchanged
    // on both sides. Taylor is then graded on exactly those columns, overall and by regime.
    if a.iter().any(|x| x == "--fingerprint") {
        let d1 = Deriv::new(&deriv, 1, &table, flags)?;
        let fp = |rows: &[u8], j: usize| -> Result<Vec<i32>, String> {
            let (_, _, _, rec, cnt, hist) = d1.aug(&rows[j * ROW..(j + 1) * ROW])?;
            let mut v = rec; v.extend(cnt); v.extend(hist); Ok(v) };
        let fp0: Vec<Vec<i32>> = (0..n).map(|j| fp(&base_rows, j)).collect::<Result<_, _>>()?;
        let regimes: Vec<(&str, Vec<bool>)> = vec![
            ("snow_on_ice", (0..n).map(|j| xb[27][j] > 0.0).collect()),
            ("bare_ice", (0..n).map(|j| xb[27][j] <= 0.0).collect()),
            ("day_gsw>0", (0..n).map(|j| xb[21][j] > 0.0).collect()),
            ("night_gsw=0", (0..n).map(|j| xb[21][j] <= 0.0).collect()),
            ("near_freezing_tso0>268", (0..n).map(|j| tsob[j * 9] > 268.0).collect()),
            ("cold_tso0<=268", (0..n).map(|j| tsob[j * 9] <= 268.0).collect()),
            ("saturated_qv>=qsg", (0..n).map(|j| xb[2][j] >= xb[31][j]).collect()),
            ("unsaturated_qv<qsg", (0..n).map(|j| xb[2][j] < xb[31][j]).collect()),
            ("moist_qc>0", (0..n).map(|j| xb[3][j] > 0.0).collect()),
            ("cap_tie_in_column", (0..n).map(|j| (0..9).any(|l| tsob[j * 9 + l] == 271.4f32)).collect()),
            ("no_cap_tie", (0..n).map(|j| (0..9).all(|l| tsob[j * 9 + l] != 271.4f32)).collect()),
        ];
        println!("P6_FP_REGIMES {}", regimes.iter().map(|(s, m)| format!("{s}={}", m.iter().filter(|&&x| x).count())).collect::<Vec<_>>().join(" "));
        for &seed in &seeds {
            let mut r = Rng(seed * 104729 + 3);
            let dv: Vec<Vec<f32>> = (0..72).map(|k| (0..n).map(|j| if perturbed(k) { (r.next() * (xb[k][j] as f64).abs()) as f32 } else { 0.0 }).collect()).collect();
            let mut dt: Vec<f32> = (0..n * 9).map(|j| (r.next() * (tsob[j] as f64).abs()) as f32).collect();
            // --dir-set continuous: hold the precipitation and snow-state inputs fixed (rain/snow/graupel rates,
            // frozen fraction, snow mass/depth/cover, accumulations, snom, precipfr, rhosnf, chklowq) and the
            // tso values tied at the 271.4 K cap; everything else moves. --dir-set snow: only those held inputs.
            let held = |k: usize| matches!(k, 8..=12 | 27..=29 | 42..=47);
            let set = arg("--dir-set").unwrap_or("all".into());
            let mut dv = dv;
            if set == "continuous" {
                for k in 0..72 { if held(k) { dv[k].iter_mut().for_each(|x| *x = 0.0); } }
                for j in 0..n * 9 { if tsob[j] == 271.4f32 { dt[j] = 0.0; } }
            } else if set == "snow" {
                for k in 0..72 { if !held(k) { dv[k].iter_mut().for_each(|x| *x = 0.0); } }
                dt.iter_mut().for_each(|x| *x = 0.0);
            }
            println!("P6_FP_DIRSET seed={seed} set={set}");
            let (tv, tt) = d.tl(&dv, &dt)?;
            for &eps in &[1e-2f64, 3e-3, 1e-3, 3e-4, 1e-4] {
                let mut stable = vec![true; n]; let mut samples = vec![];
                for sign in [1.0f64, -1.0] {
                    let mut mem = c.pre.clone();
                    for (j, &col) in cols.iter().enumerate() {
                        for k in 0..54 { if perturbed(k) { c.set(&mut mem, SLOTS[k], col, 0, (xb[k][j] as f64 + sign * eps * dv[k][j] as f64) as f32); } }
                        for l in 0..9 { c.set(&mut mem, 50, col, l, (tsob[j * 9 + l] as f64 + sign * eps * dt[j * 9 + l] as f64) as f32); }
                    }
                    let out = orig.run(&c, &mem)?;
                    let rows = build_rows(&mem);
                    for j in 0..n { if fp(&rows, j)? != fp0[j] { stable[j] = false; } }
                    samples.push(outputs_of(&out));
                }
                let ((ps, pt), (ms, mt)) = (&samples[0], &samples[1]);
                // Per-column norm-scaled relative error (all 54 scalar outputs + 9 tso levels).
                let colerr: Vec<f64> = (0..n).map(|j| {
                    let (mut e, mut t) = (0f64, 0f64);
                    for k in 0..54 { if k == 5 || out_scale[k] == 0.0 { continue; }
                        let fd = (ps[k][j] as f64 - ms[k][j] as f64) / (2.0 * eps);
                        e += ((fd - tv[k][j] as f64) / out_scale[k]).powi(2); t += (tv[k][j] as f64 / out_scale[k]).powi(2); }
                    for l in 0..9 { let q = j * 9 + l; let fd = (pt[q] as f64 - mt[q] as f64) / (2.0 * eps);
                        e += ((fd - tt[q] as f64) / tso_scale).powi(2); t += (tt[q] as f64 / tso_scale).powi(2); }
                    (e / t.max(1e-300)).sqrt() }).collect();
                let grade = |keep: &dyn Fn(usize) -> bool| -> (f64, usize, f64, f64, usize, String) {
                    let (mut e2, mut t2) = (0f64, 0f64); let mut worstf = (0f64, String::new());
                    for k in 0..54 { if k == 5 || out_scale[k] == 0.0 { continue; }
                        let (mut ek, mut tk) = (0f64, 0f64);
                        for j in 0..n { if keep(j) { let fd = (ps[k][j] as f64 - ms[k][j] as f64) / (2.0 * eps);
                            ek += (fd - tv[k][j] as f64).powi(2); tk += (tv[k][j] as f64).powi(2); } }
                        e2 += ek / out_scale[k].powi(2); t2 += tk / out_scale[k].powi(2);
                        if tk > 0.0 && (ek / tk).sqrt() > worstf.0 { worstf = ((ek / tk).sqrt(), NAMES[k].to_string()); } }
                    let (mut ek, mut tk) = (0f64, 0f64);
                    for q in 0..n * 9 { if keep(q / 9) { let fd = (pt[q] as f64 - mt[q] as f64) / (2.0 * eps);
                        ek += (fd - tt[q] as f64).powi(2); tk += (tt[q] as f64).powi(2); } }
                    e2 += ek / tso_scale.powi(2); t2 += tk / tso_scale.powi(2);
                    if tk > 0.0 && (ek / tk).sqrt() > worstf.0 { worstf = ((ek / tk).sqrt(), "tso".into()); }
                    let ce: Vec<f64> = (0..n).filter(|&j| keep(j)).map(|j| colerr[j]).collect();
                    let cnt = ce.len();
                    let (med, mx) = if cnt > 0 { (median(ce.clone()), ce.iter().cloned().fold(0f64, f64::max)) } else { (f64::NAN, f64::NAN) };
                    let lt = ce.iter().filter(|&&x| x < 1e-3).count();
                    ((e2 / t2.max(1e-300)).sqrt(), cnt, med, mx, lt, format!("{}:{:.1e}", worstf.1, worstf.0)) };
                let (ra, ..) = grade(&|_| true);
                let (rs, cs, med, mx, lt, wf) = grade(&|j| stable[j]);
                println!("P6_FP_TAYLOR seed={seed} eps={eps:e} all_cols={ra:.3e} stable_cols={cs}/{n} stable_rel={rs:.3e} stable_col_median={med:.2e} stable_col_max={mx:.2e} stable_cols_lt_1e-3={lt} worst_field={wf}");
                for (name, m) in &regimes {
                    let (rr, cr, mr, xr, lr, wr) = grade(&|j| stable[j] && m[j]);
                    if cr > 0 { println!("P6_FP_REGIME seed={seed} eps={eps:e} regime={name} stable_cols={cr}/{} rel={rr:.3e} col_median={mr:.2e} col_max={xr:.2e} lt_1e-3={lr} worst_field={wr}", m.iter().filter(|&&x| x).count()); }
                }
            }
        }
    }

    // 5b. Attribution: one input field at a time (seed 1 values), centred error at three eps,
    // with the number of columns whose visit records change.
    if a.iter().any(|x| x == "--attribute") {
        let mut r = Rng(104732);
        let full: Vec<Vec<f32>> = (0..72).map(|k| (0..n).map(|j| if perturbed(k) { (r.next() * (xb[k][j] as f64).abs()) as f32 } else { 0.0 }).collect()).collect();
        let fullt: Vec<f32> = (0..n * 9).map(|j| (r.next() * (tsob[j] as f64).abs()) as f32).collect();
        let only: Option<String> = arg("--attribute-only");
        for k in (0..54).filter(|&k| perturbed(k)).chain([54]).chain(100..109) {
            let name = if k == 54 { "tso".to_string() } else if k >= 100 { format!("tso_l{}", k - 100) } else { NAMES[k].to_string() };
            if let Some(o) = &only { if !o.split(',').any(|x| x == name || (x == "tso*" && name.starts_with("tso"))) { continue; } }
            let dv: Vec<Vec<f32>> = (0..72).map(|q| if q == k { full[q].clone() } else { vec![0.0; n] }).collect();
            let dt: Vec<f32> = if k == 54 { fullt.clone() } else if k >= 100 { (0..n * 9).map(|j| if j % 9 == k - 100 { fullt[j] } else { 0.0 }).collect() } else { vec![0.0; n * 9] };
            if dv.iter().flatten().chain(dt.iter()).all(|x| *x == 0.0) { println!("P6_ATTRIB input={name} zero_direction (field is 0 on every ice column)"); continue; }
            let (tv, tt) = d.tl(&dv, &dt)?;
            let mut line = format!("P6_ATTRIB input={name}");
            for eps in [1e-2f64, 1e-3, 1e-4] {
                let mut s = vec![]; let mut chg = 0;
                for sign in [1.0f64, -1.0] {
                    let mut mem = c.pre.clone();
                    for (j, &col) in cols.iter().enumerate() {
                        if k < 54 { c.set(&mut mem, SLOTS[k], col, 0, (xb[k][j] as f64 + sign * eps * dv[k][j] as f64) as f32); }
                        else { for l in 0..9 { c.set(&mut mem, 50, col, l, (tsob[j * 9 + l] as f64 + sign * eps * dt[j * 9 + l] as f64) as f32); } }
                    }
                    let out = orig.run(&c, &mem)?;
                    let (_, _, _, prec, pcnt, _) = d.aug(&build_rows(&mem))?;
                    chg += (0..n).filter(|&j| (0..64).any(|q| prec[j * 64 + q] != rec0[j * 64 + q]) || pcnt[j * 2] != cnt0[j * 2] || pcnt[j * 2 + 1] != cnt0[j * 2 + 1]).count();
                    s.push(outputs_of(&out));
                }
                let (mut e2, mut t2) = (0f64, 0f64); let mut worst = (0f64, "");
                for q in 0..54 { if q == 5 || out_scale[q] == 0.0 { continue; }
                    let (mut eq, mut tq) = (0f64, 0f64);
                    for j in 0..n { let fd = (s[0].0[q][j] as f64 - s[1].0[q][j] as f64) / (2.0 * eps); eq += (fd - tv[q][j] as f64).powi(2); tq += (tv[q][j] as f64).powi(2); }
                    e2 += eq / out_scale[q].powi(2); t2 += tq / out_scale[q].powi(2);
                    if tq > 0.0 && (eq / tq).sqrt() > worst.0 { worst = ((eq / tq).sqrt(), NAMES[q]); } }
                if a.iter().any(|x| x == "--column-detail") && eps == 1e-3 {
                    // Per column: relative TL-vs-FD error on the tso outputs, the column's regime, and
                    // whether the augmented primal reproduces the original image at the +eps point.
                    let mut rows_err: Vec<(f64, usize)> = (0..n).map(|j| {
                        let (mut e, mut t) = (0f64, 0f64);
                        for l in 0..9 { let fd = (s[0].1[j * 9 + l] as f64 - s[1].1[j * 9 + l] as f64) / (2.0 * eps); e += (fd - tt[j * 9 + l] as f64).powi(2); t += (tt[j * 9 + l] as f64).powi(2); }
                        ((e / t.max(1e-300)).sqrt(), j) }).collect();
                    rows_err.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
                    let good = rows_err.iter().filter(|x| x.0 < 1e-3).count();
                    println!("P6_COLUMNS_DETAIL input={name} eps=1e-3 columns_err_lt_1e-3={good}/{n} median={:.2e}", rows_err[n / 2].0);
                    for &(e, j) in rows_err.iter().take(8).chain(rows_err.iter().rev().take(3)) {
                        println!("P6_COLUMN input={name} col={} err={e:.2e} snow={:.4e} snowh={:.4e} snowc={:.3} xice={:.3} soilt={:.2} tso0={:.2} bare_visits={} snow_visits={}",
                            cols[j], xb[27][j], xb[28][j], xb[29][j], xb[52][j], xb[24][j], tsob[j * 9], cnt0[j * 2], cnt0[j * 2 + 1]);
                    }
                }
                let mut lev = vec![(0f64, 0f64); 9];
                for j in 0..n * 9 { let fd = (s[0].1[j] as f64 - s[1].1[j] as f64) / (2.0 * eps); e2 += (fd - tt[j] as f64).powi(2) / tso_scale.powi(2); t2 += (tt[j] as f64).powi(2) / tso_scale.powi(2);
                    lev[j % 9].0 += (fd - tt[j] as f64).powi(2); lev[j % 9].1 += (tt[j] as f64).powi(2); }
                let tso_lev: Vec<String> = lev.iter().map(|(e, t)| if *t > 0.0 { format!("{:.1e}", (e / t).sqrt()) } else if *e > 0.0 { format!("tl0/fd{:.1e}", e.sqrt()) } else { "0".into() }).collect();
                line += &format!(" eps{eps:e}: rel={:.2e} worst={}:{:.1e} tso_out_by_level={tso_lev:?} changed_cols={chg}", (e2 / t2.max(1e-300)).sqrt(), worst.1, worst.0);
            }
            println!("{line}");
        }
    }

    // 6. Cost: original launch vs aug, TL, AD (median of 25, synchronized, device-resident inputs).
    let reps = 25;
    let mut tf = vec![];
    for _ in 0..reps { orig.upload(&c, &c.pre)?; sync()?; let s = Instant::now(); orig.launch(&c)?; sync()?; tf.push(s.elapsed().as_secs_f64() * 1e6); }
    let mut ta = vec![]; put(d.rows, &base_rows)?;
    for _ in 0..reps { let s = Instant::now(); d.construct_aug()?; sync()?; ta.push(s.elapsed().as_secs_f64() * 1e6); }
    d.tables(&d.in_table(), &d.out_table())?; sync()?;
    let mut tt = vec![]; for _ in 0..reps { let s = Instant::now(); d.sweep(false)?; sync()?; tt.push(s.elapsed().as_secs_f64() * 1e6); }
    let mut tr = vec![]; for _ in 0..reps { let s = Instant::now(); d.sweep(true)?; sync()?; tr.push(s.elapsed().as_secs_f64() * 1e6); }
    let (f, au, tl, ad) = (median(tf), median(ta), median(tt), median(tr));
    let (mut free, mut total) = (0usize, 0usize); unsafe { cuMemGetInfo_v2(&mut free, &mut total); }
    println!("P6_COST forward_us={f:.1} aug_us={au:.1} tl_us={tl:.1} ad_us={ad:.1} aug/fwd={:.2} tl/fwd={:.2} ad/fwd={:.2} (aug+tl)/fwd={:.2} (aug+ad)/fwd={:.2} device_used_mib={}",
        au / f, tl / f, ad / f, (au + tl) / f, (au + ad) / f, (total - free) >> 20);
    println!("P6_SUMMARY replay=0diff aug_identity=0diff dot_pass={dot_pass} taylor_best={best_all:.3e} wall_s={:.1}", t0.elapsed().as_secs_f64());
    Ok(())
}
