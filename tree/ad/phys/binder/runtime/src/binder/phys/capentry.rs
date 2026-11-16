// SPDX-License-Identifier: Apache-2.0
//! Held-kernel lanes (ad/held-rlw_cldprmc_coalesced, private autodiff): the per-launch derivative dot of one
//! ad.entry/v1 kernel-set row on ONE real captured launch (an ad.hex.launch/v1 manifest), for launches the sealed
//! pkg18 plan does not issue (hrrrv5 radiation is a held upload there, so `replay` + AD_DOT_ONLY cannot reach it).
//! It drives the window's own code unchanged:
//!   deriv::load (the kernel-set row and its fatbin, the same admission as the window: an unsupported row is
//!   reported, never run), KernelDeriv::prepare_with (pre-launch primal copies and float32 shadows), the CAPTURED
//!   forward image on the captured pre-launch state (checked bitwise against the capture's post state), then
//!   dot::Run::launch_dot (the U6 dot: random seeds over all bound derivative memory, zero direction, second
//!   reverse bitwise), once per seed, each from a fresh restore of the pre-launch state.
//! Extra, kernel-local (`--gain-out ARG --gain-in A,B,..`): the gain ||TL x||/||x|| of the named output over a
//! direction on the named inputs only (iid uniform, and relative: u * primal), and a central finite difference of
//! the real forward image along the relative direction at each `--fd-h` (output words compared with the TL).
//! Usage: ad-binder capture-dot --kernels K.json --cubins DIR --capture LAUNCH_DIR --out OUT.json
//!        [--seeds 1,2] [--arena-gib 2] [--gain-out taucmc --gain-in ciwpmc,... --fd-h 1e-2,1e-3]
use crate::cuda::{checked, Cuda};
use crate::{deriv, dev, dot, submit};
use serde_json::{json, Value};
use std::{collections::{BTreeSet, HashMap}, ffi::{c_void, CString}, path::{Path, PathBuf}, ptr};

struct Cap { function: String, image: Vec<u8>, pre: Vec<Vec<u8>>, post_sha: Vec<String>, args: Vec<submit::Argument>,
             names: Vec<String>, grid: [u32; 3], block: [u32; 3], shared: u32, man: Value }

fn read_capture(dir: &Path) -> Result<Cap, String> {
    let rd = |p: &str| std::fs::read(dir.join(p)).map_err(|e| format!("{}: {e}", dir.join(p).display()));
    let man: Value = serde_json::from_slice(&rd("manifest.json")?).map_err(|e| e.to_string())?;
    let image = rd(man["image"]["path"].as_str().ok_or("image path")?)?;
    if crate::exec::digest(&image) != man["image"]["sha256"].as_str().unwrap_or("") { return Err("image sha256 differs from the manifest".into()); }
    let (mut pre, mut post_sha) = (Vec::new(), Vec::new());
    for a in man["allocations"].as_array().ok_or("allocations")? {
        // A large allocation is stored as `raw-concat-v1` chunks (content-addressed, repeated chunks shared).
        let b = match a["pre"]["chunks"].as_array() {
            Some(chunks) => {
                let mut v = Vec::with_capacity(a["pre"]["bytes"].as_u64().unwrap_or(0) as usize);
                for c in chunks {
                    let part = rd(c["path"].as_str().ok_or("chunk path")?)?;
                    if crate::exec::digest(&part) != c["sha256"].as_str().unwrap_or("") { return Err("pre chunk sha256 differs".into()); }
                    v.extend_from_slice(&part);
                }
                v
            },
            None => rd(a["pre"]["path"].as_str().ok_or("pre path")?)?,
        };
        if crate::exec::digest(&b) != a["pre"]["sha256"].as_str().unwrap_or("") { return Err("pre object sha256 differs".into()); }
        pre.push(b);
        post_sha.push(a["post"]["sha256"].as_str().ok_or("post sha")?.to_owned());
    }
    let mut args = Vec::new(); let mut names = Vec::new();
    for a in man["arguments"].as_array().ok_or("arguments")? {
        names.push(a["name"].as_str().unwrap_or("").to_owned());
        args.push(match a["kind"].as_str() {
            Some("scalar") => submit::Argument::Bytes(crate::exec::decode_hex(a["value_hex"].as_str().ok_or("value_hex")?)?),
            Some("device") => submit::Argument::Device(submit::Slice { allocation: a["allocation"].as_u64().ok_or("allocation")?,
                                                                       offset: a["offset"].as_u64().unwrap_or(0) as usize }),
            // g3-radland: a CuPy CArray (or other struct) by value with patched device pointers.
            Some("packed") => {
                let bytes = crate::exec::decode_hex(a["value_hex"].as_str().ok_or("value_hex")?)?;
                let mut pointers = Vec::new();
                for p in a["packed_pointers"].as_array().ok_or("packed_pointers")? {
                    pointers.push(submit::PointerPatch { byte_offset: p["byte_offset"].as_u64().ok_or("byte_offset")? as usize,
                        target: submit::Slice { allocation: p["allocation"].as_u64().ok_or("allocation")?, offset: p["offset"].as_u64().unwrap_or(0) as usize } });
                }
                submit::Argument::Patched { bytes, pointers }
            },
            k => return Err(format!("argument kind {k:?} not supported here")),
        });
    }
    let g = &man["geometry"];
    let ax = |v: &Value, i: usize| v[i].as_u64().unwrap_or(1) as u32;
    Ok(Cap { function: man["kernel"].as_str().ok_or("kernel")?.to_owned(), image, pre, post_sha, args, names,
             grid: [ax(&g["grid"], 0), ax(&g["grid"], 1), ax(&g["grid"], 2)], block: [ax(&g["block"], 0), ax(&g["block"], 1), ax(&g["block"], 2)],
             shared: g["shared_bytes"].as_u64().unwrap_or(0) as u32, man })
}

fn f32s(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }
fn bytes(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

pub fn run(argv: Vec<String>) -> Result<(), String> {
    let mut o: HashMap<String, String> = HashMap::new();
    let mut it = argv.into_iter();
    while let Some(k) = it.next() { let v = it.next().ok_or_else(|| format!("missing value for {k}"))?; o.insert(k.trim_start_matches("--").to_owned(), v); }
    let req = |k: &str| o.get(k).cloned().ok_or_else(|| format!("capture-dot: --{k} is required"));
    let (kernels, cubins, capdir, out) = (PathBuf::from(req("kernels")?), PathBuf::from(req("cubins")?), PathBuf::from(req("capture")?), PathBuf::from(req("out")?));
    let seeds: Vec<u64> = o.get("seeds").map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or(vec![1, 2]);
    let gib: usize = o.get("arena-gib").and_then(|g| g.parse().ok()).unwrap_or(2);
    let cap = read_capture(&capdir)?;
    unsafe {
        let cuda = Cuda::load()?;
        checked("cuInit", (cuda.init)(0))?;
        let mut device = 0; checked("cuDeviceGet", (cuda.device_get)(&mut device, 0))?;
        let mut context = ptr::null_mut(); checked("cuCtxCreate", (cuda.context_create)(&mut context, 0, device))?;
        let mut live: HashMap<u64, submit::Allocation> = HashMap::new();
        for (i, b) in cap.pre.iter().enumerate() {
            let mut a = 0u64; checked("cuMemAlloc capture", (cuda.allocate)(&mut a, b.len().max(4)))?;
            live.insert(i as u64, submit::Allocation { id: i as u64, address: a, bytes: b.len() });
        }
        let restore = |live: &HashMap<u64, submit::Allocation>, over: &HashMap<u64, Vec<u8>>| -> Result<(), String> {
            for (i, b) in cap.pre.iter().enumerate() { dev::upload(&cuda, live[&(i as u64)].address, over.get(&(i as u64)).unwrap_or(b))?; }
            dev::sync(&cuda)
        };
        // The captured forward image.
        let mut module = ptr::null_mut();
        checked("cuModuleLoadData forward", (cuda.module_load)(&mut module, cap.image.as_ptr().cast()))?;
        let cname = CString::new(cap.function.as_str()).map_err(|e| e.to_string())?;
        let mut fwd = ptr::null_mut();
        checked("cuModuleGetFunction forward", (cuda.module_function)(&mut fwd, module, cname.as_ptr()))?;
        let forward = |live: &HashMap<u64, submit::Allocation>| -> Result<(), String> {
            let mut store: Vec<Vec<u8>> = cap.args.iter().map(|a| match a {
                submit::Argument::Bytes(v) => v.clone(),
                submit::Argument::Device(s) => (live[&s.allocation].address + s.offset as u64).to_le_bytes().to_vec(),
                submit::Argument::Patched { bytes, pointers } => {
                    let mut v = bytes.clone();
                    for p in pointers { v[p.byte_offset..p.byte_offset + 8].copy_from_slice(&(live[&p.target.allocation].address + p.target.offset as u64).to_le_bytes()); }
                    v
                },
                _ => unreachable!(),
            }).collect();
            let mut p: Vec<*mut c_void> = store.iter_mut().map(|v| v.as_mut_ptr().cast()).collect();
            checked("forward launch", (cuda.launch)(fwd, cap.grid[0], cap.grid[1], cap.grid[2], cap.block[0], cap.block[1], cap.block[2],
                                                    cap.shared, ptr::null_mut(), p.as_mut_ptr(), ptr::null_mut()))?;
            dev::sync(&cuda)
        };
        let only: BTreeSet<String> = [cap.function.clone()].into();
        crate::window::check_spec_hashes(&kernels)?;
        let set = deriv::load(&cuda, &kernels, &cubins, Some(&only))?;
        let kd = set.get(&cap.function).ok_or_else(|| format!("{}: no kernel-set row", cap.function))?;
        if let Some(why) = &kd.unsupported { return Err(format!("{}: kernel-set row refused by deriv::load: {why}", cap.function)); }
        let support = dev::Support::load(&cuda)?;
        let mut arena = dev::Arena::new(&cuda, gib << 30)?;
        let l = deriv::LaunchInfo { args: &cap.args, grid: cap.grid, block: cap.block, shared: cap.shared };
        let none = HashMap::new();
        let mut rows = Vec::new();
        let mut forward_rows = Vec::new();
        for (k, &seed) in seeds.iter().enumerate() {
            restore(&live, &none)?;
            arena.reset();
            let mut b = kd.prepare_with(&cuda, &mut arena, &l, &live, 4, None)?;
            forward(&live)?;
            let mut differ = Vec::new();
            for (i, sha) in cap.post_sha.iter().enumerate() {
                let a = live[&(i as u64)];
                if &crate::exec::digest(&dev::download(&cuda, a.address, a.bytes)?) != sha { differ.push(i); }
            }
            forward_rows.push(json!({"seed": seed, "post_allocations_differing": differ}));
            let run = dot::Run { cuda: &cuda, support: &support, live: &live };
            let mut row = run.launch_dot(kd, &mut arena, &mut b, &l, seed, 2e-5, true)?;
            row["seed"] = json!(seed);
            row["lhs_bits"] = json!(format!("{:016x}", row["lhs"].as_f64().unwrap_or(f64::NAN).to_bits()));
            row["rhs_bits"] = json!(format!("{:016x}", row["rhs"].as_f64().unwrap_or(f64::NAN).to_bits()));
            row["shadows"] = json!(b.shadows.iter().map(|(a, s)| json!({"allocation": a, "bytes": s.bytes, "lo": s.lo})).collect::<Vec<_>>());
            row["arena_peak_bytes"] = json!(arena.peak);
            eprintln!("{}", json!({"event": "capture_dot", "k": k, "seed": seed, "status": row["status"], "rel": row["rel"], "norm_scaled": row["norm_scaled"]}));
            rows.push(row);
        }
        // Kernel-local gain and finite differences (optional).
        let mut gain = Value::Null;
        if let (Some(gout), Some(gin)) = (o.get("gain-out"), o.get("gain-in")) {
            let gin: Vec<String> = gin.split(',').map(str::to_owned).collect();
            let pos = |n: &str| cap.names.iter().position(|x| x == n).ok_or_else(|| format!("no argument {n}"));
            let slice = |n: &str| -> Result<(u64, usize, usize), String> {     // (allocation, byte offset, words)
                let a = &cap.man["arguments"][pos(n)?];
                let words: u64 = a["shape"].as_array().ok_or("shape")?.iter().map(|d| d.as_u64().unwrap_or(1)).product();
                let p = if a["kind"] == "packed" { &a["packed_pointers"][0] } else { a };
                Ok((p["allocation"].as_u64().ok_or("allocation")?, p["offset"].as_u64().unwrap_or(0) as usize, words as usize))
            };
            let view = |b: &deriv::Bound, n: &str| -> Result<(u64, usize), String> {
                let (alloc, off, words) = slice(n)?;
                let s = b.shadows.get(&alloc).ok_or_else(|| format!("{n}: no shadow"))?;
                Ok((s.address + ((off - s.lo) / 4 * s.width) as u64, words))
            };
            let primal = |n: &str| -> Result<Vec<f32>, String> { let (a, off, w) = slice(n)?; Ok(f32s(&cap.pre[a as usize][off..off + 4 * w])) };
            // --fd-one-sided 1: the relative direction is |u| * primal (every word moves up) and the difference is
            // forward, (f(x + h d) - f(x)) / h: it stays on the executed side of every table node and step.
            let one = o.get("fd-one-sided").is_some_and(|v| v == "1");
            let mut seen_out = Vec::new();
            let mut tl_rel_out = Vec::new();
            let mut summary = serde_json::Map::new();
            for (mode, s0) in [("iid", 0x9a1u64), ("relative", 0x9a2u64)] {
                restore(&live, &none)?;
                arena.reset();
                let mut b = kd.prepare_with(&cuda, &mut arena, &l, &live, 4, None)?;
                forward(&live)?;
                for s in b.shadows.values() { dev::zero(&cuda, s.address, s.bytes)?; }
                let mut x2 = 0.0f64; let mut xin = serde_json::Map::new();
                for (j, n) in gin.iter().enumerate() {
                    let (addr, words) = view(&b, n)?;
                    let p = primal(n)?;
                    let x: Vec<f32> = (0..words).map(|i| {
                        let u = dot::unit(s0 ^ (j as u64) << 32, i as u64);
                        (if mode == "iid" { u } else if one { u.abs() * p[i] as f64 } else { u * p[i] as f64 }) as f32 }).collect();
                    let n2: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
                    x2 += n2; xin.insert(n.clone(), json!(n2.sqrt()));
                    dev::upload(&cuda, addr, &bytes(&x))?;
                }
                let run = dot::Run { cuda: &cuda, support: &support, live: &live };
                run.tangent(kd, &mut arena, &mut b, &l)?;
                let fired = kd.check_failures(&cuda, &mut b)?;
                let (addr, words) = view(&b, gout)?;
                let t = f32s(&dev::download(&cuda, addr, 4 * words)?);
                let t2: f64 = t.iter().map(|&v| (v as f64) * (v as f64)).sum();
                let after = { let (a, off, w) = slice(gout)?; f32s(&dev::download(&cuda, live[&a].address + off as u64, 4 * w)?) };
                let pre_out = primal(gout)?;
                let written = after.iter().zip(&pre_out).filter(|(a, p)| a.to_bits() != p.to_bits()).count();
                let nonzero = t.iter().filter(|v| **v != 0.0).count();
                let a2: f64 = after.iter().map(|&v| (v as f64) * (v as f64)).sum();
                let in2: f64 = gin.iter().map(|n| primal(n).map(|p| p.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>()).unwrap_or(0.0)).sum();
                summary.insert(mode.into(), json!({"x_norm": x2.sqrt(), "x_norm_by_input": xin, "tl_out_norm": t2.sqrt(),
                    "gain": t2.sqrt() / x2.sqrt(), "out_words": words, "out_words_tl_nonzero": nonzero,
                    "out_words_written_by_forward": written, "refusals": fired,
                    "relative_gain": if mode == "relative" { json!((t2.sqrt() / a2.sqrt()) / (x2.sqrt() / in2.sqrt())) } else { Value::Null }}));
                if mode == "relative" { tl_rel_out = t; seen_out = after; }
            }
            // Central finite differences of the real forward along the relative direction (inputs * (1 +- h u)).
            let mut fd = Vec::new();
            let mut fd_err = Vec::new();
            for h in o.get("fd-h").map(|s| s.split(',').filter_map(|x| x.parse::<f64>().ok()).collect()).unwrap_or(vec![1e-2, 1e-3]) {
                let mut res = [Vec::new(), Vec::new()];
                let signs: &[f64] = if one { &[1.0] } else { &[1.0, -1.0] };
                for (si, sign) in signs.iter().enumerate() {
                    let mut over: HashMap<u64, Vec<u8>> = HashMap::new();
                    for (j, n) in gin.iter().enumerate() {
                        let (alloc, off, words) = slice(n)?;
                        let buf = over.entry(alloc).or_insert_with(|| cap.pre[alloc as usize].clone());
                        let p = primal(n)?;
                        let v: Vec<f32> = (0..words).map(|i| {
                            let u = dot::unit(0x9a2u64 ^ (j as u64) << 32, i as u64);
                            let u = if one { u.abs() } else { u };
                            (p[i] as f64 * (1.0 + sign * h * u)) as f32 }).collect();
                        buf[off..off + 4 * words].copy_from_slice(&bytes(&v));
                    }
                    restore(&live, &over)?;
                    forward(&live)?;
                    let (a, off, w) = slice(gout)?;
                    res[si] = f32s(&dev::download(&cuda, live[&a].address + off as u64, 4 * w)?);
                    if let Ok((ea, eo, _)) = slice("err_flag") {
                        let e = dev::download(&cuda, live[&ea].address + eo as u64, 4)?;
                        if e != [0, 0, 0, 0] { fd_err.push(json!({"h": h, "sign": sign, "err_flag": i32::from_le_bytes(e[..4].try_into().unwrap())})); }
                    }
                }
                if let Some(d) = o.get("dump") {
                    std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
                    std::fs::write(Path::new(d).join(format!("tl-{h:e}.f32")), bytes(&tl_rel_out)).map_err(|e| e.to_string())?;
                    std::fs::write(Path::new(d).join(format!("base.f32")), bytes(&seen_out)).map_err(|e| e.to_string())?;
                    for (si, r) in res.iter().enumerate() { if !r.is_empty() {
                        std::fs::write(Path::new(d).join(format!("fd{si}-{h:e}.f32")), bytes(r)).map_err(|e| e.to_string())?; } }
                }
                let (mut e2, mut t2, mut emax, mut big) = (0.0f64, 0.0f64, 0.0f64, 0usize);
                for i in 0..tl_rel_out.len() {
                    let q = if one { (res[0][i] as f64 - seen_out[i] as f64) / h } else { (res[0][i] as f64 - res[1][i] as f64) / (2.0 * h) };
                    let t = tl_rel_out[i] as f64;
                    let e = (q - t).abs();
                    e2 += e * e; t2 += t * t; emax = emax.max(e);
                    if e > 1e-3 * t.abs().max(1e-30) + 1e-6 * seen_out[i].abs() as f64 { big += 1; }
                }
                fd.push(json!({"h": h, "rel_l2": (e2 / t2).sqrt(), "max_abs": emax, "words_over_1e-3_rel": big}));
            }
            gain = json!({"out": gout, "inputs": gin, "fd_one_sided": one, "modes": summary, "fd_relative_direction": fd, "fd_forward_err_flags": fd_err});
        }
        let report = json!({"schema": "ad.capture-dot/v1", "function": cap.function, "capture": capdir,
            "step": cap.man["step"], "launch_index": cap.man["launch_index"], "geometry": cap.man["geometry"],
            "image": cap.man["image"]["sha256"], "source": cap.man["source"]["sha256"],
            "kernel_set": kernels, "entries": {"tl": kd.tl_order, "ad": kd.ad_order},
            "forward": forward_rows, "dots": rows, "gain": gain});
        std::fs::write(&out, serde_json::to_vec_pretty(&report).unwrap()).map_err(|e| e.to_string())?;
        println!("{}", json!({"event": "capture_dot_done", "out": out, "dots": rows.iter().map(|r| json!({"seed": r["seed"], "status": r["status"], "rel": r["rel"]})).collect::<Vec<_>>()}));
    }
    Ok(())
}
