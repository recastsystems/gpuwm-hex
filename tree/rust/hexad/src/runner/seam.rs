// SPDX-License-Identifier: Apache-2.0
//! seam-damping: the per-row N-rank vs 1-rank check of one ad.entry/v1 dycore row whose
//! stencil reads across a partition seam. Inputs: the row's global (1-rank) captured launch and N rank-local captures
//! of the same launch (gen_local.py: arrays gathered owned-first + halo rings, connectivity remapped with the hexcore
//! clamp law, entity-count scalars set to the rank's counts), each local allocation with map-<a>.i64 (global word of
//! every local word) and own-<a>.u8 (1 where the rank owns the word).
//! It drives the window's own derivative code unchanged (deriv::load, prepare_with, the rank-local derived topology
//! registry for the local captures, dot::Run tangent and reverse) and forms the composed operator
//!   M_N = sum_r R_r^T O_r M_r P_r     (gather x to every rank, run the row, keep owned outputs)
//!   M_N^T y = sum_r P_r^T M_r^T O_r R_r y   (seed owned outputs only, reverse, add every local cotangent onto its
//!                                           global word: the owner reduction, in float64)
//! and reports, per row: forward owned words vs the 1-rank post state, TL owned words vs 1-rank TL, the composed dot
//! (aligned M - I and plain), the cross dot <y, TL_1 x> vs <x, M_N^T y>, and the reduced adjoint vs the 1-rank adjoint.
//! THE BREAKAGE IT FINDS: a row whose reverse on a rank-local mesh is not the transpose of its rank-local tangent
//! (clamped out-of-rank referents, transposed incidence built from the global mesh, a halo output seeded twice), or
//! whose forward/tangent at owned entities reads past the halo (owned words differ from 1 rank).
//! Usage: ad-binder seam-dot --kernels K.json --cubins DIR --global LAUNCH_DIR --ranks DIR --out OUT.json
//!        [--seed 1] [--arena-gib 2]
use crate::cuda::{checked, Cuda};
use crate::{deriv, dev, dot, submit};
use serde_json::{json, Value};
use std::{collections::{BTreeMap, BTreeSet, HashMap}, ffi::{c_void, CString}, path::{Path, PathBuf}, ptr};

struct Out {
    post: Vec<Vec<u8>>,
    tl: BTreeMap<u64, (usize, usize, Vec<f64>)>,   // alloc -> (lo bytes, width, words)
    ad: BTreeMap<u64, (usize, usize, Vec<f64>)>,
    ad_cells: BTreeMap<String, f64>,
    x: BTreeMap<u64, Vec<f64>>,
    y: BTreeMap<u64, Vec<f64>>,
    fired: Vec<String>,
    forward_ran: bool,
}

fn rd_vec<T: Copy>(p: &Path, f: impl Fn(&[u8]) -> T, w: usize) -> Result<Vec<T>, String> {
    let b = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    Ok(b.chunks_exact(w).map(f).collect())
}

fn key(seed: u64, galloc: u64, gword: i64) -> u64 { seed ^ (galloc << 40) ^ (gword as u64) }

unsafe fn shadow_put(cuda: &Cuda, s: &deriv::Shadow, v: &[f64]) -> Result<(), String> {
    let b: Vec<u8> = if s.width == 8 { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
                     else { v.iter().flat_map(|x| (*x as f32).to_le_bytes()).collect() };
    dev::upload(cuda, s.address, &b)
}
unsafe fn shadow_get(cuda: &Cuda, s: &deriv::Shadow) -> Result<Vec<f64>, String> {
    let b = dev::download(cuda, s.address, s.bytes)?;
    Ok(if s.width == 8 { b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect() }
       else { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64).collect() })
}

/// One launch: forward, TL along x, reverse from y (owned words only when `own` is given).
/// `gid(alloc, word)` -> (global allocation, global word) or None.
unsafe fn run_one(cuda: &Cuda, sup: &dev::Support, arena: &mut dev::Arena, kernels: &Path, cubins: &Path,
                  cap: &crate::capentry::Cap, local: bool, seed: u64,
                  gid: &dyn Fn(u64, usize) -> Option<(u64, i64)>, own: &dyn Fn(u64, usize) -> bool) -> Result<Out, String> {
    let mut live: HashMap<u64, submit::Allocation> = HashMap::new();
    for (i, b) in cap.pre.iter().enumerate() {
        let mut a = 0u64; checked("cuMemAlloc capture", (cuda.allocate)(&mut a, b.len().max(4)))?;
        dev::upload(cuda, a, b)?;
        live.insert(i as u64, submit::Allocation { id: i as u64, address: a, bytes: b.len() });
    }
    dev::sync(cuda)?;
    let only: BTreeSet<String> = [cap.function.clone()].into();
    let set = deriv::load(cuda, kernels, cubins, Some(&only))?;
    let kd = set.get(&cap.function).ok_or_else(|| format!("{}: no kernel-set row", cap.function))?;
    if let Some(why) = &kd.unsupported { return Err(format!("{}: kernel-set row refused by deriv::load: {why}", cap.function)); }
    if local {
        // the rank's mesh pack (gen_local.py) stands in for the plan's other launches of the same rank mesh
        let mut extra = Vec::new();
        if let Some(pk) = cap.man.get("mesh_pack") {
            let dims: Vec<(String, usize)> = ["ncells", "nedges", "nvertices"].iter()
                .filter_map(|d| pk["dims"][*d].as_u64().map(|v| (d.to_string(), v as usize))).collect();
            let arrays = pk["arrays"].as_array().ok_or("mesh_pack arrays")?.iter().map(|a| (a["name"].as_str().unwrap_or("").to_owned(),
                (a["allocation"].as_u64().unwrap_or(0), 0usize), a["width"].as_u64().map(|w| w as usize))).collect();
            extra.push(crate::topo::Observation { function: "seam-mesh-pack".into(), arrays, dims });
        }
        deriv::install_local_topology_with(&set, std::iter::once((cap.function.as_str(), cap.args.clone(), cap.grid, cap.block, cap.shared)), extra);
    } else {
        deriv::clear_local_topology();
    }
    // The captured forward image is the recording card's cubin. On a card of another architecture (status 209, no
    // binary for the GPU) the forward is not run and the receipt says so; TL/AD come from the multi-arch kernel set.
    let mut module = ptr::null_mut();
    let loaded = (cuda.module_load)(&mut module, cap.image.as_ptr().cast());
    let forward_ran = loaded == 0;
    if !forward_ran && loaded != 209 { checked("cuModuleLoadData forward", loaded)?; }
    let cname = CString::new(cap.function.as_str()).map_err(|e| e.to_string())?;
    let mut fwd = ptr::null_mut();
    if forward_ran { checked("cuModuleGetFunction forward", (cuda.module_function)(&mut fwd, module, cname.as_ptr()))?; }
    // A rank may be a SEQUENCE of launches of the row on the same memory (selected damping: interior pass then
    // boundary pass, manifest "selected_passes"): forward and TL in order, reverse in reverse order, all passes on one
    // persistent shadow per allocation (the window's sweep shadows), so a pass's reverse sees the seeds the later
    // pass's reverse left, exactly as in the window.
    let passes: Vec<Vec<submit::Argument>> = match cap.man.get("selected_passes").and_then(|v| v.as_array()) {
        Some(sel) => {
            let i = cap.names.iter().position(|n| n == "select_interior").ok_or("selected_passes without select_interior")?;
            sel.iter().map(|v| { let mut a = cap.args.clone(); a[i] = submit::Argument::Bytes((v.as_i64().unwrap_or(0) as i32).to_le_bytes().to_vec()); a }).collect()
        },
        None => vec![cap.args.clone()],
    };
    arena.reset();
    let persistent: Option<BTreeMap<u64, deriv::Shadow>> = if passes.len() > 1 {
        let mut m = BTreeMap::new();
        for (i, b) in cap.pre.iter().enumerate() {
            let bytes = (b.len() / 4) * 4;
            if bytes == 0 { continue; }
            m.insert(i as u64, deriv::Shadow { address: arena.take(bytes)?, bytes, width: 4, lo: 0 });
        }
        Some(m)
    } else { None };
    let mut bounds = Vec::new();
    for args in &passes {
        let l = deriv::LaunchInfo { args, grid: cap.grid, block: cap.block, shared: cap.shared };
        let mut bd = kd.prepare_with(cuda, arena, &l, &live, 4, persistent.as_ref())?;
        let mut store: Vec<Vec<u8>> = args.iter().map(|a| match a {
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
        if forward_ran { checked("forward launch", (cuda.launch)(fwd, cap.grid[0], cap.grid[1], cap.grid[2], cap.block[0], cap.block[1], cap.block[2],
                                                cap.shared, ptr::null_mut(), p.as_mut_ptr(), ptr::null_mut()))?; }
        dev::sync(cuda)?;
        if kd.aug { kd.launch(cuda, arena, &mut bd, &l, &live, "aug")?; }
        bounds.push(bd);
    }
    let mut post = Vec::new();
    for i in 0..cap.pre.len() { let a = live[&(i as u64)]; post.push(dev::download(cuda, a.address, a.bytes)?); }
    let run = dot::Run { cuda, support: sup, live: &live };
    // x over every bound derivative word, from the word's GLOBAL identity (so every rank sees the same x)
    let shadows: Vec<(u64, deriv::Shadow)> = match &persistent { Some(m) => m.iter().map(|(a, s)| (*a, *s)).collect(),
                                                                None => bounds[0].shadows.iter().map(|(a, s)| (*a, *s)).collect() };
    let mut xs = BTreeMap::new();
    for (a, s) in &shadows {
        let n = s.bytes / s.width;
        let x: Vec<f64> = (0..n).map(|i| match gid(*a, s.lo / 4 + i) { Some((g, w)) => dot::unit(key(seed, g, w), 1), None => 0.0 }).collect();
        xs.insert(*a, x);
    }
    for (a, s) in &shadows { shadow_put(cuda, s, &xs[a])?; }
    for bd in &bounds {
        let scal: Vec<(String, (u64, usize))> = bd.tl_cells.iter().map(|(f, c)| (format!("{f:?}"), *c)).collect();
        for (k, (_, cell)) in scal.iter().enumerate() { dot::put_cell(cuda, *cell, dot::unit(seed ^ 0x5ca1_ab1e, k as u64))?; }
    }
    let mut fired = Vec::new();
    for (k, bd) in bounds.iter_mut().enumerate() {
        let l = deriv::LaunchInfo { args: &passes[k], grid: cap.grid, block: cap.block, shared: cap.shared };
        run.tangent(kd, arena, bd, &l)?;
        fired.extend(kd.check_failures(cuda, bd)?);
    }
    let mut tl = BTreeMap::new();
    for (a, s) in &shadows { tl.insert(*a, (s.lo, s.width, shadow_get(cuda, s)?)); }
    // y on owned words only
    let mut ys = BTreeMap::new();
    for (a, s) in &shadows {
        let n = s.bytes / s.width;
        let y: Vec<f64> = (0..n).map(|i| {
            let w = s.lo / 4 + i;
            match gid(*a, w) { Some((g, gw)) if own(*a, w) => dot::unit(key(seed ^ 0x7777_0000_0000_0000, g, gw), 2), _ => 0.0 }
        }).collect();
        ys.insert(*a, y);
    }
    for (a, s) in &shadows { shadow_put(cuda, s, &ys[a])?; }
    for k in (0..bounds.len()).rev() {
        let l = deriv::LaunchInfo { args: &passes[k], grid: cap.grid, block: cap.block, shared: cap.shared };
        run.reverse(kd, arena, &mut bounds[k], &l)?;
        fired.extend(kd.check_failures(cuda, &mut bounds[k])?);
    }
    let mut ad = BTreeMap::new();
    for (a, s) in &shadows { ad.insert(*a, (s.lo, s.width, shadow_get(cuda, s)?)); }
    let mut ad_cells: BTreeMap<String, f64> = BTreeMap::new();
    for bd in &bounds { for (f, c) in &bd.ad_cells { *ad_cells.entry(format!("{f:?}")).or_insert(0.0) += dot::get_cell(cuda, *c)?.0; } }
    dev::sync(cuda)?;
    for a in live.values() { (cuda.free)(a.address); }
    deriv::clear_local_topology();
    Ok(Out { post, tl, ad, ad_cells, x: xs, y: ys, fired, forward_ran })
}

pub fn run(argv: Vec<String>) -> Result<(), String> {
    let mut o: HashMap<String, String> = HashMap::new();
    let mut it = argv.into_iter();
    while let Some(k) = it.next() { let v = it.next().ok_or_else(|| format!("missing value for {k}"))?; o.insert(k.trim_start_matches("--").to_owned(), v); }
    let req = |k: &str| o.get(k).cloned().ok_or_else(|| format!("seam-dot: --{k} is required"));
    let (kernels, cubins, gdir, rdir, out) = (PathBuf::from(req("kernels")?), PathBuf::from(req("cubins")?), PathBuf::from(req("global")?),
                                              PathBuf::from(req("ranks")?), PathBuf::from(req("out")?));
    let seed: u64 = o.get("seed").and_then(|s| s.parse().ok()).unwrap_or(1);
    let gib: usize = o.get("arena-gib").and_then(|g| g.parse().ok()).unwrap_or(2);
    let gcap = crate::capentry::read_capture(&gdir)?;
    let mut ranks = Vec::new();
    for r in 0.. {
        let d = rdir.join(format!("r{r}"));
        if !d.exists() { break; }
        let cap = crate::capentry::read_capture(&d)?;
        let mut maps = Vec::new();
        for (i, a) in cap.man["allocations"].as_array().ok_or("allocations")?.iter().enumerate() {
            let g = a["global_allocation"].as_u64().unwrap_or(u64::MAX);
            let m = rd_vec(&d.join(format!("map-{i}.i64")), |c| i64::from_le_bytes(c.try_into().unwrap()), 8)?;
            let ow = rd_vec(&d.join(format!("own-{i}.u8")), |c| c[0], 1)?;
            maps.push((g, m, ow));
        }
        ranks.push((cap, maps));
    }
    if ranks.is_empty() { return Err("seam-dot: no rank captures r0..".into()); }
    let mut report = serde_json::Map::new();
    unsafe {
        let cuda = Cuda::load()?;
        checked("cuInit", (cuda.init)(0))?;
        let mut device = 0; checked("cuDeviceGet", (cuda.device_get)(&mut device, 0))?;
        let mut context = ptr::null_mut(); checked("cuCtxCreate", (cuda.context_create)(&mut context, 0, device))?;
        crate::window::check_spec_hashes(&kernels)?;
        let sup = dev::Support::load(&cuda)?;
        let mut arena = dev::Arena::new(&cuda, gib << 30)?;
        // Directions only on the words the launch's arguments view: an allocation's alignment tail is in the shadow
        // span but in no rank's mesh, so it would count in the 1-rank dot and in no composed one.
        let mut views: HashMap<u64, Vec<(usize, usize)>> = HashMap::new();
        for a in gcap.man["arguments"].as_array().ok_or("arguments")? {
            if a["kind"] != "device" { continue; }
            let isz = match a["dtype"].as_str().unwrap_or("") { d if d.ends_with('8') => 8, d if d.ends_with('2') => 2, d if d.ends_with('1') => 1, _ => 4 };
            let n: u64 = a["shape"].as_array().map(|s| s.iter().map(|d| d.as_u64().unwrap_or(1)).product()).unwrap_or(1);
            let lo = a["offset"].as_u64().unwrap_or(0) as usize / 4;
            views.entry(a["allocation"].as_u64().unwrap_or(0)).or_default().push((lo, lo + ((n as usize * isz) + 3) / 4));
        }
        let in_view = |a: u64, w: usize| views.get(&a).is_some_and(|v| v.iter().any(|(l, h)| w >= *l && w < *h));
        let g = run_one(&cuda, &sup, &mut arena, &kernels, &cubins, &gcap, false, seed,
                        &|a, w| if in_view(a, w) { Some((a, w as i64)) } else { None }, &|a, w| in_view(a, w))?;
        // forward post of the global capture vs its recorded post
        let gpost_ok: Vec<usize> = if !g.forward_ran { Vec::new() } else { (0..gcap.post_sha.len()).filter(|&i| crate::exec::digest(&g.post[i]) != gcap.post_sha[i]).collect() };
        let mut rows = Vec::new();
        let mut acc: BTreeMap<u64, Vec<f64>> = g.ad.iter().map(|(a, (_, _, v))| (*a, vec![0.0; v.len()])).collect();
        let mut acc_cells: BTreeMap<String, f64> = BTreeMap::new();
        let mut comp_tl: BTreeMap<u64, Vec<f64>> = g.tl.iter().map(|(a, (_, _, v))| (*a, vec![f64::NAN; v.len()])).collect();
        let (mut fwd_diff, mut tl_diff, mut owned_words, mut uncovered) = (0usize, 0usize, 0usize, 0usize);
        let mut first_tl_diff = Value::Null;
        for (r, (cap, maps)) in ranks.iter().enumerate() {
            let gid = |a: u64, w: usize| -> Option<(u64, i64)> {
                let (gal, m, _) = &maps[a as usize];
                if *gal == u64::MAX { return None; }
                m.get(w).copied().filter(|v| *v >= 0).map(|v| (*gal, v))
            };
            let own = |a: u64, w: usize| -> bool { maps[a as usize].2.get(w).copied().unwrap_or(0) == 1 };
            let lr = run_one(&cuda, &sup, &mut arena, &kernels, &cubins, cap, true, seed, &gid, &own)?;
            let (mut rf, mut rt) = (0usize, 0usize);
            // forward owned words vs the 1-rank post (4-byte words)
            for (a, (gal, m, ow)) in maps.iter().enumerate() {
                if *gal == u64::MAX { continue; }
                let lp = &lr.post[a]; let gp = &g.post[*gal as usize];
                for (w, (&gw, &o)) in m.iter().zip(ow.iter()).enumerate() {
                    if o != 1 || gw < 0 { continue; }
                    let gw = gw as usize;
                    if lp[4 * w..4 * w + 4] != gp[4 * gw..4 * gw + 4] { rf += 1; }
                }
            }
            // tangent owned words and the composed operator
            for (a, (lo, _, v)) in &lr.tl {
                let (gal, m, ow) = &maps[*a as usize];
                if *gal == u64::MAX { continue; }
                let Some((glo, gwid, gv)) = g.tl.get(gal) else { uncovered += v.len(); continue };
                for (i, t) in v.iter().enumerate() {
                    let w = lo / 4 + i;
                    if ow.get(w).copied().unwrap_or(0) != 1 { continue; }
                    let gw = m[w] as usize;
                    if gw < glo / 4 || gw - glo / 4 >= gv.len() { uncovered += 1; continue; }
                    let gi = gw - glo / 4;
                    owned_words += 1;
                    let same = if *gwid == 8 { t.to_bits() == gv[gi].to_bits() } else { (*t as f32).to_bits() == (gv[gi] as f32).to_bits() };
                    if !same {
                        rt += 1;
                        if first_tl_diff.is_null() { first_tl_diff = json!({"rank": r, "local_alloc": a, "global_alloc": gal, "global_word": gw, "rank_tl": t, "one_rank_tl": gv[gi]}); }
                    }
                    comp_tl.get_mut(gal).unwrap()[gi] = *t;
                }
            }
            // owner reduction of every local cotangent onto its global word
            for (a, (lo, _, v)) in &lr.ad {
                let (gal, m, _) = &maps[*a as usize];
                if *gal == u64::MAX { continue; }
                let Some((glo, _, gv)) = g.ad.get(gal) else { uncovered += v.len(); continue };
                for (i, c) in v.iter().enumerate() {
                    let w = lo / 4 + i;
                    let gw = m[w];
                    if gw < 0 { continue; }
                    let gw = gw as usize;
                    if gw < glo / 4 || gw - glo / 4 >= gv.len() { if *c != 0.0 { uncovered += 1; } continue; }
                    acc.get_mut(gal).unwrap()[gw - glo / 4] += c;
                }
            }
            for (f, c) in &lr.ad_cells { *acc_cells.entry(f.clone()).or_insert(0.0) += c; }
            fwd_diff += rf; tl_diff += rt;
            rows.push(json!({"rank": r, "forward_owned_words_differing": rf, "tl_owned_words_differing": rt,
                             "refusals": lr.fired, "owned": cap.man["rank"]["owned"], "local": cap.man["rank"]["local"],
                             "grid": cap.grid}));
        }
        // dots. x and y over the global words (each global word owned exactly once)
        let (mut xy, mut lhs_c, mut lhs_g, mut rhs_c, mut rhs_g, mut nan_words) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize);
        for (a, (_, _, tg)) in &g.tl {
            let (x, y) = (&g.x[a], &g.y[a]);
            let tc = &comp_tl[a];
            for i in 0..tg.len() {
                xy += x[i] * y[i];
                lhs_g += y[i] * tg[i];
                if tc[i].is_nan() { if y[i] != 0.0 { nan_words += 1; } } else { lhs_c += y[i] * tc[i]; }
            }
        }
        for (a, (_, _, ag)) in &g.ad {
            let x = &g.x[a]; let ac = &acc[a];
            for i in 0..ag.len() { rhs_g += x[i] * ag[i]; rhs_c += x[i] * ac[i]; }
        }
        let mut scal = Vec::new();
        for (k, (f, v)) in g.ad_cells.iter().enumerate() {
            let xs = dot::unit(seed ^ 0x5ca1_ab1e, k as u64);
            rhs_g += xs * v;
            let c = acc_cells.get(f).copied().unwrap_or(0.0);
            rhs_c += xs * c;
            scal.push(json!({"field": f, "one_rank": v, "reduced": c}));
        }
        let rel = |l: f64, r: f64| (l - r).abs() / l.abs().max(r.abs()).max(1e-300);
        // reduced adjoint vs 1-rank adjoint
        let (mut ad_words, mut ad_diff_f32, mut ad_maxrel) = (0usize, 0usize, 0.0f64);
        let mut per_alloc = Vec::new();
        for (a, (_, _, ag)) in &g.ad {
            let ac = &acc[a];
            let scale = ag.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
            let (mut d, mut mr) = (0usize, 0.0f64);
            for i in 0..ag.len() {
                if (ac[i] as f32).to_bits() != (ag[i] as f32).to_bits() { d += 1; }
                mr = mr.max((ac[i] - ag[i]).abs() / scale);
            }
            ad_words += ag.len(); ad_diff_f32 += d; ad_maxrel = ad_maxrel.max(mr);
            per_alloc.push(json!({"global_alloc": a, "words": ag.len(), "f32_words_differing": d, "max_abs_diff_over_max": mr}));
        }
        report.insert("schema".into(), json!("ad.seam-dot/v1"));
        report.insert("function".into(), json!(gcap.function));
        report.insert("global_capture".into(), json!(gdir));
        report.insert("ranks_dir".into(), json!(rdir));
        report.insert("n_ranks".into(), json!(ranks.len()));
        report.insert("forward_ran".into(), json!(g.forward_ran));
        report.insert("global_forward_post_allocs_differing".into(), json!(gpost_ok));
        report.insert("global_refusals".into(), json!(g.fired));
        report.insert("forward_owned_words_differing".into(), if g.forward_ran { json!(fwd_diff) } else { Value::Null });
        report.insert("tl_owned_words".into(), json!(owned_words));
        report.insert("tl_owned_words_differing".into(), json!(tl_diff));
        report.insert("first_tl_difference".into(), first_tl_diff);
        report.insert("uncovered_words".into(), json!(uncovered));
        report.insert("global_words_without_owner_with_y".into(), json!(nan_words));
        report.insert("dot_one_rank".into(), json!({"lhs": lhs_g, "rhs": rhs_g, "rel_plain": rel(lhs_g, rhs_g), "rel_aligned": rel(lhs_g - xy, rhs_g - xy)}));
        report.insert("dot_composed".into(), json!({"lhs": lhs_c, "rhs": rhs_c, "rel_plain": rel(lhs_c, rhs_c), "rel_aligned": rel(lhs_c - xy, rhs_c - xy)}));
        report.insert("dot_cross".into(), json!({"lhs_one_rank_tl": lhs_g, "rhs_reduced_ad": rhs_c, "rel_plain": rel(lhs_g, rhs_c), "rel_aligned": rel(lhs_g - xy, rhs_c - xy)}));
        report.insert("ad_reduced_vs_one_rank".into(), json!({"words": ad_words, "f32_words_differing": ad_diff_f32, "max_abs_diff_over_max": ad_maxrel, "per_alloc": per_alloc, "scalars": scal}));
        report.insert("ranks".into(), json!(rows));
    }
    let v = Value::Object(report);
    std::fs::write(&out, serde_json::to_vec_pretty(&v).unwrap()).map_err(|e| e.to_string())?;
    println!("{}", json!({"event": "seam_dot_done", "function": v["function"], "n_ranks": v["n_ranks"],
        "fwd_diff": v["forward_owned_words_differing"], "tl_diff": v["tl_owned_words_differing"],
        "dot_one_rank": v["dot_one_rank"]["rel_aligned"], "dot_composed": v["dot_composed"]["rel_aligned"], "dot_cross": v["dot_cross"]["rel_aligned"],
        "ad_maxrel": v["ad_reduced_vs_one_rank"]["max_abs_diff_over_max"], "out": out}));
    Ok(())
}
