// SPDX-License-Identifier: Apache-2.0
//! ad/held-rlw_taumol_batched: the per-launch derivative dot of an ad.scheme.rows/v1 row on one REAL captured launch,
//! through the loader code the window and addot bind (schemes.rs prepare / tangent / reverse / launch_dot).
//!
//! Some launches never appear in the sealed plan's command stream: radiation and RUC land are held uploads in pkg18,
//! so `replay` cannot reach them and addot lists them untested. This mode takes one launch captured from the native
//! forecast (ad.hex.launch/v1: complete backing allocations, pre and post, every recorded pointer slot) instead:
//! 1. every allocation is uploaded from its pre object (sha256 checked) and every recorded pointer slot relocated;
//! 2. the row is bound to the launch (prepare, before the forward), the captured image runs the forward, and every
//!    allocation is compared word for word with the captured post state (pointer slots compared as relocated);
//! 3. launch_dot per seed (the window's in-place operator; checks: zero direction both ways and a second reverse);
//! 4. gains ||TL x|| / ||x|| on input-only directions (raw and relative to the primal), and a sha256 of everything
//!    that must not change between two processes (result_sha256);
//! 5. write coverage: the output extents are poisoned, the forward reruns, and words it left unwritten are counted.
//!
//! usage: AD_SCHEMES=<bundle> ad-binder capdot --capture <launch dir> [--seeds 1,2] [--checks 1] [--arena-gib 4] --out <file>
use crate::cuda::{checked, Cuda};
use crate::dev::{self, Arena, Support};
use crate::schemes::{Module, Native};
use crate::submit::{Allocation, Argument, Slice};
use serde_json::{json, Value};
use std::{collections::{HashMap, HashSet}, ffi::c_void, path::{Path, PathBuf}, ptr};

fn read_obj(dir: &Path, o: &Value) -> Result<Vec<u8>, String> {
    let bytes = if let Some(p) = o["path"].as_str() {
        std::fs::read(dir.join(p)).map_err(|e| format!("{p}: {e}"))?
    } else if o["encoding"] == "raw-concat-v1" {
        let mut v = Vec::new();
        for c in o["chunks"].as_array().ok_or("chunks")? {
            let p = c["path"].as_str().ok_or("chunk path")?;
            v.extend(std::fs::read(dir.join(p)).map_err(|e| format!("{p}: {e}"))?);
        }
        v
    } else {
        return Err(format!("capture object encoding {} is not read here (raw or raw-concat-v1 only)", o["encoding"]));
    };
    let want = o["sha256"].as_str().ok_or("object sha256")?;
    let got = crate::exec::digest(&bytes);
    if got != want { return Err(format!("capture object sha256 {got}, manifest says {want}")); }
    Ok(bytes)
}

fn splitmix(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

unsafe fn launch_raw(cuda: &Cuda, f: *mut c_void, grid: [u32; 3], block: [u32; 3], shared: u32, args: &[Vec<u8>]) -> Result<(), String> {
    let store: Vec<Vec<u64>> = args.iter().map(|b| {
        let mut v = vec![0u64; b.len().div_ceil(8).max(1)];
        std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), b.len()).copy_from_slice(b);
        v
    }).collect();
    let mut params: Vec<*mut c_void> = store.iter().map(|v| v.as_ptr() as *mut c_void).collect();
    checked("cuLaunchKernel (captured forward)", (cuda.launch)(f, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared,
                                                               ptr::null_mut(), params.as_mut_ptr(), ptr::null_mut()))
}

fn f64s(b: &[u8]) -> Vec<f64> { b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect() }
fn f32s(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }

pub fn run(argv: Vec<String>) -> Result<(), String> {
    let (mut cap, mut out, mut seeds, mut checks, mut gib) = (None, None, vec![1u64, 2], true, 4usize);
    let mut it = argv.into_iter();
    while let Some(a) = it.next() {
        let v = it.next().ok_or_else(|| format!("missing value for {a}"))?;
        match a.as_str() {
            "--capture" => cap = Some(PathBuf::from(v)),
            "--out" => out = Some(PathBuf::from(v)),
            "--seeds" => seeds = v.split(',').map(|s| s.parse::<u64>().map_err(|_| format!("bad seed {s}"))).collect::<Result<_, _>>()?,
            "--checks" => checks = v != "0",
            "--arena-gib" => gib = v.parse().map_err(|_| "bad --arena-gib")?,
            _ => return Err(format!("capdot: unknown argument {a}")),
        }
    }
    let (cap, out) = (cap.ok_or("capdot: --capture DIR")?, out.ok_or("capdot: --out FILE")?);
    let r = unsafe { capdot(&cap, &seeds, checks, gib) }?;
    std::fs::write(&out, serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", out.display()))?;
    println!("{}", json!({"event":"capdot_done","out":out,"status":r["status"],"result_sha256":r["result_sha256"]}));
    if r["status"] == "PASS" { Ok(()) } else { Err(format!("capdot: status {}", r["status"])) }
}

unsafe fn capdot(dir: &Path, seeds: &[u64], checks: bool, gib: usize) -> Result<Value, String> {
    let t0 = std::time::Instant::now();
    let m: Value = serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).map_err(|e| format!("manifest: {e}"))?).map_err(|e| e.to_string())?;
    if m["schema"] != "ad.hex.launch/v1" || m["complete"] != true { return Err("capdot: manifest is not a complete ad.hex.launch/v1 capture".into()); }
    let function = m["kernel"].as_str().ok_or("kernel")?.to_owned();
    let cuda = Cuda::load()?;
    checked("cuInit", (cuda.init)(0))?;
    let mut device = 0;
    checked("cuDeviceGet", (cuda.device_get)(&mut device, 0))?;
    let mut context = ptr::null_mut();
    checked("cuCtxCreate", (cuda.context_create)(&mut context, 0, device))?;

    // 1. allocations from their pre objects, then the recorded pointer slots relocated
    let allocs = m["allocations"].as_array().ok_or("allocations")?;
    let mut live: HashMap<u64, Allocation> = HashMap::new();
    for (i, a) in allocs.iter().enumerate() {
        let pre = read_obj(dir, &a["pre"])?;
        let mut at = 0u64;
        checked("cuMemAlloc capture allocation", (cuda.allocate)(&mut at, pre.len().max(4)))?;
        dev::upload(&cuda, at, &pre)?;
        live.insert(i as u64, Allocation { id: i as u64, address: at, bytes: pre.len() });
    }
    let mut reloc: HashMap<u64, Vec<(usize, u64)>> = HashMap::new();
    for ip in m["indirect_pointers"].as_array().into_iter().flatten() {
        let (ta, to) = (ip["table_allocation"].as_u64().ok_or("table_allocation")?, ip["table_offset"].as_u64().ok_or("table_offset")? as usize);
        let (ga, go) = (ip["target_allocation"].as_u64().ok_or("target_allocation")?, ip["target_offset"].as_u64().ok_or("target_offset")?);
        let v = live.get(&ga).ok_or("pointer target")?.address + go;
        let t = live.get(&ta).ok_or("pointer table")?;
        if to + 8 > t.bytes { return Err(format!("pointer slot {ta}:{to} outside its allocation")); }
        dev::upload(&cuda, t.address + to as u64, &v.to_le_bytes())?;
        reloc.entry(ta).or_default().push((to, v));
    }
    // launch arguments, as the plan would hand them to the loader
    let mut args = Vec::new();
    let mut raw = Vec::new();
    for a in m["arguments"].as_array().ok_or("arguments")? {
        match a["kind"].as_str().unwrap_or("") {
            "scalar" => {
                let b = crate::exec::decode_hex(a["value_hex"].as_str().ok_or("value_hex")?)?;
                raw.push(b.clone());
                args.push(Argument::Bytes(b));
            },
            "device" => {
                let (al, off) = (a["allocation"].as_u64().ok_or("argument allocation")?, a["offset"].as_u64().unwrap_or(0) as usize);
                raw.push((live.get(&al).ok_or("argument allocation")?.address + off as u64).to_le_bytes().to_vec());
                args.push(Argument::Device(Slice { allocation: al, offset: off }));
            },
            k => return Err(format!("capdot: argument kind {k} ({}) is not handled", a["name"])),
        }
    }
    let g = &m["geometry"];
    let dim3 = |v: &Value| -> Result<[u32; 3], String> {
        let a = v.as_array().ok_or("geometry")?;
        Ok([a[0].as_u64().unwrap_or(1) as u32, a[1].as_u64().unwrap_or(1) as u32, a[2].as_u64().unwrap_or(1) as u32]) };
    let (grid, block, shared) = (dim3(&g["grid"])?, dim3(&g["block"])?, g["shared_bytes"].as_u64().unwrap_or(0) as u32);
    let image = read_obj(dir, &m["image"])?;
    let img_path = std::env::temp_dir().join(format!("capdot-{}-{}.cubin", std::process::id(), &m["image"]["sha256"].as_str().unwrap_or("x")[..8]));
    std::fs::write(&img_path, &image).map_err(|e| e.to_string())?;
    let module = Module::load(&cuda, &img_path)?;
    let _ = std::fs::remove_file(&img_path);
    let fwd = module.func(&cuda, &function)?;

    // 2. bind, forward, post compare
    let spec: HashSet<String> = HashSet::new();
    let mut native = Native::from_env(&cuda, &spec)?.ok_or("capdot: AD_SCHEMES is not set")?;
    let row = native.row(&function).ok_or_else(|| format!("capdot: the bundle has no row for {function}"))?.clone();
    let sup = Support::load(&cuda)?;
    let mut arena = Arena::new(&cuda, gib << 30)?;
    let mut p = native.prepare(&cuda, &mut arena, &live, &function, &args, grid, block)?;
    let tf = std::time::Instant::now();
    launch_raw(&cuda, fwd, grid, block, shared, &raw)?;
    dev::sync(&cuda)?;
    let fwd_ms = tf.elapsed().as_secs_f64() * 1e3;
    let mut post_diff = serde_json::Map::new();
    let (mut words, mut differ) = (0usize, 0usize);
    let mut post_now: HashMap<u64, Vec<u8>> = HashMap::new();
    for (i, a) in allocs.iter().enumerate() {
        let mut want = read_obj(dir, &a["post"])?;
        for &(o, v) in reloc.get(&(i as u64)).into_iter().flatten() { want[o..o + 8].copy_from_slice(&v.to_le_bytes()); }
        let l = &live[&(i as u64)];
        let got = dev::download(&cuda, l.address, l.bytes)?;
        let d = got.chunks(4).zip(want.chunks(4)).filter(|(x, y)| x != y).count();
        words += want.len().div_ceil(4);
        differ += d;
        if d != 0 { post_diff.insert(i.to_string(), json!(d)); }
        post_now.insert(i as u64, got);
    }

    // 3. the launch dot (schemes.rs, the window's operator) per seed, with a hash of the shadows it leaves
    let segs = p.active_segments();
    let hash_shadows = |cuda: &Cuda| -> Result<String, String> {
        let mut all = Vec::new();
        for &(_, lo, hi, sh) in &segs { all.extend(dev::download(cuda, sh, (hi - lo).div_ceil(4) * 8)?); }
        Ok(crate::exec::digest(&all)) };
    let mut dots = Vec::new();
    let mut det = Vec::new();
    let mut status = if differ == 0 { "PASS" } else { "FORWARD-MISMATCH" };
    for &s in seeds {
        let d = native.launch_dot(&cuda, &sup, &mut arena, &mut p, &live, s, checks)?;
        let h = hash_shadows(&cuda)?;
        if d["status"] != "PASS" && status == "PASS" { status = "DOT-FAIL"; }
        det.push(json!({"seed": s, "lhs": d["lhs"], "rhs": d["rhs"], "rel": d["rel"], "zero_tl": d["zero_tl_nonzero_words"],
                        "zero_ad": d["zero_ad_nonzero_words"], "reverse_differing": d["reverse_differing_words"], "shadows_sha256": h}));
        dots.push(json!({"seed": s, "dot": d, "shadows_sha256": h}));
    }

    // 4. gains on input-only directions: ||TL x|| / ||x|| (raw) and the relative gain (||dy/y|| / ||dx/x||)
    let shadow_of = |al: u64, off: usize| -> Result<u64, String> {
        segs.iter().find(|s| s.0 == al && off >= s.1 && off < s.2).map(|s| s.3 + ((off - s.1) / 4 * 8) as u64)
            .ok_or_else(|| format!("no shadow at {al}:{off}")) };
    let extent = |a: &crate::schemes::ArgRow| -> Result<usize, String> {
        let e = a.extent.as_deref().ok_or("extent")?;
        let mut env = crate::expr::Env::new();
        // every by-value scalar of the row, decoded by its dtype (as Row::env does for the bound launch)
        for a in row.args.iter().filter(|a| !a.device) {
            let Some(Argument::Bytes(b)) = args.get(a.pos) else { continue };
            let v = match (a.dtype.as_str(), b.len()) {
                ("<i4", 4) => i32::from_le_bytes(b[..4].try_into().unwrap()) as f64,
                ("<u4", 4) => u32::from_le_bytes(b[..4].try_into().unwrap()) as f64,
                ("<i8", 8) => i64::from_le_bytes(b[..8].try_into().unwrap()) as f64,
                ("<f4", 4) => f32::from_le_bytes(b[..4].try_into().unwrap()) as f64,
                ("<f8", 8) => f64::from_le_bytes(b[..8].try_into().unwrap()),
                _ => continue,
            };
            env.insert(a.name.clone(), v);
        }
        Ok(crate::expr::eval(e, &env).map_err(|x| format!("extent {e}: {x}"))? as usize) };
    let ins: Vec<_> = row.args.iter().filter(|a| a.device && a.active && (a.role == "in" || a.role == "inout")).cloned().collect();
    let outs: Vec<_> = row.args.iter().filter(|a| a.device && a.active && (a.role == "out" || a.role == "inout")).cloned().collect();
    let slot = |a: &crate::schemes::ArgRow| -> Result<(u64, usize), String> {
        match args.get(a.pos) { Some(Argument::Device(s)) => Ok((s.allocation, s.offset)), _ => Err(format!("{} is not a device argument", a.name)) } };
    let mut gains = Vec::new();
    for (mode, gseed) in [("raw", 77u64), ("relative", 78u64)] {
        for &(_, lo, hi, sh) in &segs { dev::zero(&cuda, sh, (hi - lo).div_ceil(4) * 8)?; }
        let (mut nx2, mut nxr2) = (0f64, 0f64);
        let mut st = gseed;
        for a in &ins {
            let (al, off) = slot(a)?;
            let n = extent(a)?;
            let prim = f32s(&post_now[&al][off..off + 4 * n]);
            let x: Vec<f64> = (0..n).map(|i| { let u = splitmix(&mut st); if mode == "raw" { u } else { u * prim[i].abs() as f64 } }).collect();
            for i in 0..n { nx2 += x[i] * x[i]; if prim[i] != 0.0 { nxr2 += (x[i] / prim[i] as f64).powi(2); } }
            dev::upload(&cuda, shadow_of(al, off)?, &x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        }
        native.tangent(&cuda, &mut arena, &mut p, false)?;
        dev::sync(&cuda)?;
        let (mut ny2, mut nyr2, mut nonfinite) = (0f64, 0f64, 0usize);
        let mut per = serde_json::Map::new();
        for a in &outs {
            let (al, off) = slot(a)?;
            let n = extent(a)?;
            let prim = f32s(&post_now[&al][off..off + 4 * n]);
            let y = f64s(&dev::download(&cuda, shadow_of(al, off)?, 8 * n)?);
            let (mut s2, mut big) = (0f64, 0f64);
            for i in 0..n {
                if !y[i].is_finite() { nonfinite += 1; continue; }
                s2 += y[i] * y[i]; big = big.max(y[i].abs());
                if prim[i] != 0.0 { nyr2 += (y[i] / prim[i] as f64).powi(2); }
            }
            ny2 += s2;
            per.insert(a.name.clone(), json!({"norm": s2.sqrt(), "max_abs": big}));
        }
        let g = if mode == "raw" { ny2.sqrt() / nx2.sqrt() } else { nyr2.sqrt() / nxr2.sqrt() };
        gains.push(json!({"direction": mode, "gain": g, "x_norm": nx2.sqrt(), "tl_norm": ny2.sqrt(), "nonfinite_tangent_words": nonfinite, "outputs": per}));
        det.push(json!({"gain": mode, "value": g}));
    }

    // 5. write coverage of the outputs: poison, rerun the forward, count words left unwritten; then the state again
    let mut coverage = serde_json::Map::new();
    let mut full = true;
    for a in &outs {
        let (al, off) = slot(a)?;
        let n = extent(a)?;
        dev::upload(&cuda, live[&al].address + off as u64, &vec![0xffu8; 4 * n])?;
    }
    launch_raw(&cuda, fwd, grid, block, shared, &raw)?;
    dev::sync(&cuda)?;
    for a in &outs {
        let (al, off) = slot(a)?;
        let n = extent(a)?;
        let got = dev::download(&cuda, live[&al].address + off as u64, 4 * n)?;
        let left = got.chunks_exact(4).filter(|w| *w == [0xff, 0xff, 0xff, 0xff]).count();
        let same = got == post_now[&al][off..off + 4 * n];
        full &= left == 0 && same;
        coverage.insert(a.name.clone(), json!({"words": n, "left_unwritten": left, "rerun_equals_first_forward": same}));
    }
    if !full && status == "PASS" { status = "COVERAGE-FAIL"; }
    let det_v = json!(det);
    let rsha = crate::exec::digest(det_v.to_string().as_bytes());
    let peak = arena.peak;
    Ok(json!({"schema": "ad.held.capdot/v1", "status": status, "function": function, "capture": dir, "unit": row.unit, "kind": row.kind,
        "step": m["step"], "launch_index": m["launch_index"], "image_sha256": m["image"]["sha256"], "source_sha256": m["source"]["sha256"],
        "grid": grid, "block": block, "scheme_rows": native.loaded,
        "forward": {"post_words": words, "post_differing_words": differ, "differing_by_allocation": post_diff, "ms": fwd_ms,
                    "pointer_slots_relocated": reloc.values().map(|v| v.len()).sum::<usize>()},
        "dots": dots, "gains": gains, "write_coverage": coverage, "deterministic": det_v, "result_sha256": rsha,
        "arena_peak_bytes": peak, "scheme_scratch_bytes": native.scratch_bytes(), "wall_s": t0.elapsed().as_secs_f64()}))
}
