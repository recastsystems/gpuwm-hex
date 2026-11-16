// Tape-free MYNN route. Included in schemes.rs to share its checked row binding.
// No effect unless AD_P3_DIRECT=1 at Native construction: the default reverse is the taped P3 adjoint, word for word.
// With it, each MYNN-EDMF adjoint launch is, batch by batch: (1) the hex forward with captures (AD_P3_DIRECT_FWD, entry
// mynn_capture_l1, one lane a column) on a copy of the launch's pre-state, writing one ColumnFrame a column; (2) the joined
// tape-free backward (AD_P3_DIRECT_IMAGE, entry mynn_whole_ad_direct), which gathers the output bars from the MwBind rows,
// runs the column's transposes and scatters the input bars into the shadow. No tape, no count pass, no arena of nodes.
struct P3DirectImage { _module: Module, function: *mut c_void, threads: usize, frame_bytes: usize, bars_bytes: usize,
                       _fwd: Module, capture: *mut c_void }

/// F42: the tape-free images and the bundle's taped MYNN images (p3_tl, p3_ad, p3_count) must have been built together,
/// one form. THE BREAKAGE: the three words (AD_SCHEMES, AD_P3_DIRECT_IMAGE, AD_P3_DIRECT_FWD) are set by hand, and a
/// kit's tape-free images with another form's bundle reverse every tape-free launch in one form and every taped launch
/// (the initflag launch, refused columns) in the other: a gradient that is the transpose of no tangent, with nothing
/// said. A form's taped images travel with its tape-free images: the door's aux has rel.N.cubin beside direct120.cubin;
/// build-kit.sh's kit and the lane kits have p3/rel.N.cubin or name them in SHA256SUMS (N = 0 tangent, 1 reverse,
/// 2 count). Each one found there must be the bundle's bytes; an entry with none is reported unchecked.
fn p3_direct_same_form(direct: &Path) -> Result<Value, String> {
    use sha2::{Digest, Sha256};
    let hash = |p: &Path| -> Result<String, String> {
        let b = std::fs::read(p).map_err(|e| format!("MYNN direct form check: {}: {e}", p.display()))?;
        Ok(Sha256::digest(&b).iter().map(|x| format!("{x:02x}")).collect())
    };
    let Some(bundle) = env_var("AD_SCHEMES").map(PathBuf::from) else { return Ok(json!({"checked": 0, "why": "no AD_SCHEMES"})) };
    let table: Value = serde_json::from_str(&std::fs::read_to_string(bundle.join("schemes.json"))
        .map_err(|e| format!("MYNN direct form check: {}/schemes.json: {e}", bundle.display()))?)
        .map_err(|e| format!("MYNN direct form check: schemes.json: {e}"))?;
    let dir = direct.parent().unwrap_or(Path::new("."));
    let sums: HashMap<String, String> = std::fs::read_to_string(dir.join("SHA256SUMS")).map(|t| t.lines().filter_map(|l| {
        let mut it = l.split_whitespace();
        let (h, n) = (it.next()?, it.next()?);
        Some((n.trim_start_matches('*').trim_start_matches("./").to_owned(), h.to_ascii_lowercase()))
    }).collect()).unwrap_or_default();
    let (mut checked, mut unchecked) = (Vec::new(), Vec::new());
    for (n, key) in ["p3_tl", "p3_ad", "p3_count"].iter().enumerate() {
        let Some(rel) = table["images"][key].as_str() else { unchecked.push(json!(key)); continue };
        let name = format!("rel.{n}.cubin");
        let kit = [dir.join(&name), dir.join("p3").join(&name)].into_iter().find(|p| p.is_file());
        let (want, from) = match kit {
            Some(p) => (hash(&p)?, p.display().to_string()),
            None => match sums.get(&name).or_else(|| sums.get(&format!("p3/{name}"))) {
                Some(h) => (h.clone(), format!("{}/SHA256SUMS", dir.display())),
                None => { unchecked.push(json!(key)); continue; }
            },
        };
        let got = hash(&bundle.join(rel))?;
        if got != want {
            return Err(format!("MYNN direct: the tape-free images in {} were built with the taped image {from} (sha256 {}), but the bundle {} loads {rel} (sha256 {}) as {key}. The two routes would reverse different forms: tape-free launches in one, taped launches (the initflag launch, refused columns) in the other. Use the bundle made with these images (the kit's mkbundle.sh, or the door), or the tape-free images built with this bundle; no gradient was launched",
                dir.display(), &want[..16], bundle.display(), &got[..16]));
        }
        checked.push(json!({"image": key, "sha256": &got[..16], "from": from}));
    }
    Ok(json!({"checked": checked.len(), "same": checked, "unchecked": unchecked}))
}

unsafe fn p3_direct_load(cuda: &Cuda, row: &Row) -> Result<Option<P3DirectImage>, String> {
    if env_var("AD_P3_DIRECT").as_deref() != Some("1") { return Ok(None); }
    let path = env_var("AD_P3_DIRECT_IMAGE").ok_or("AD_P3_DIRECT=1 needs AD_P3_DIRECT_IMAGE (the joined backward)")?;
    let fpath = env_var("AD_P3_DIRECT_FWD").ok_or("AD_P3_DIRECT=1 needs AD_P3_DIRECT_FWD (the forward with captures)")?;
    let abi: Value = serde_json::from_str(include_str!("../../tests/data/mynn/direct/abi.json"))
        .map_err(|e| format!("MYNN direct built-in ABI: {e}"))?;
    for key in ["float_arrays", "scalars"] {
        if row.raw[key] != abi[key] { return Err(format!("MYNN direct {key} order differs from its pinned ABI")); }
    }
    let form = p3_direct_same_form(Path::new(&path))?;
    eprintln!("{}", json!({"event":"mynn_direct_form", "check": form}));
    let module = Module::load(cuda, Path::new(&path))?;
    let (at, bytes) = module.global(cuda, "mynn_direct_info")?;
    if bytes != 64 { return Err(format!("MYNN direct metadata: {bytes} bytes, expected 64 (version 2)")); }
    let info: Vec<u64> = dev::download(cuda, at, bytes)?.chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect();
    if info[0] != 2 || info[1] != 4496 || info[2] != 91 || info[3] != 8191 || info[5] != 32 {
        return Err(format!("MYNN direct unsupported ABI: {info:?}"));
    }
    if info[4] != info[3] {
        return Err(format!("MYNN direct missing column blocks: required mask {}, implemented {}; no gradient was launched", info[3], info[4]));
    }
    let function = module.func(cuda, "mynn_whole_ad_direct")?;
    let fwd = Module::load(cuda, Path::new(&fpath))?;
    let (fat, fbytes) = fwd.global(cuda, "mynn_capture_info")?;
    let finfo: Vec<u64> = dev::download(cuda, fat, fbytes)?.chunks_exact(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect();
    let nargs = abi["args"].as_array().map_or(0, |a| a.len()) as u64;
    if fbytes != 32 || finfo[0] != 1 || finfo[1] != info[6] || finfo[2] != nargs || finfo[3] != 32 {
        return Err(format!("MYNN direct capture image {fpath}: info {finfo:?}; the backward reads {}-byte frames and the row has {nargs} arguments", info[6]));
    }
    let capture = fwd.func(cuda, "mynn_capture_l1")?;
    // The joined backward runs its arrays from a per-thread workspace (m150o0e); the most L1 suits it (1.3x measured).
    let set_attr: unsafe extern "C" fn(*mut c_void, i32, i32) -> i32 = crate::cuda::symbol(cuda.library, "cuFuncSetAttribute")?;
    checked("cuFuncSetAttribute carveout (MYNN direct)", set_attr(function, 9, 0))?;
    Ok(Some(P3DirectImage { _module: module, function, threads: info[5] as usize, frame_bytes: info[6] as usize,
                            bars_bytes: info[7] as usize, _fwd: fwd, capture }))
}

impl Native {
    /// Ok(None): a column of this launch is outside the joined backward (its status names why). Every shadow is then put
    /// back from the snapshot taken at entry, so the launch is reversed on the taped route as if this one had not run.
    unsafe fn p3_direct(&mut self, cuda: &Cuda, p: &mut Prepared, identity: bool) -> Result<Option<Value>, String> {
        if identity { return Err("MYNN direct primal identity output is not implemented".into()); }
        if self.p3_par.get() || self.p3_shards.is_some() ||
            env_var("AD_P3_BENCH_REPLICATE").map_or(false, |s| s != "1") {
            return Err("MYNN direct does not implement parameter bars, helper-card shards or replicated columns".into());
        }
        let row = self.rows["rrfs_mynnedmf_run_l16"].clone();
        let img = self.p3.as_ref().and_then(|i| i.direct.as_ref()).ok_or("MYNN direct image missing")?;
        let (function, threads, capture, fb, bb) = (img.function, img.threads, img.capture, img.frame_bytes, img.bars_bytes);
        let env = p.env.clone();
        let cols = *env.get("batch_count").ok_or("MYNN direct batch_count")?;
        if !cols.is_finite() || cols.fract() != 0.0 || cols < 0.0 || cols > i32::MAX as f64 { return Err("MYNN direct batch_count out of range".into()); }
        let cols = cols as usize;
        if cols == 0 { return Ok(Some(json!({"direct":true,"columns":0,"launches":0}))); }
        let first = *env.get("batch_first").ok_or("MYNN direct batch_first")? as i64;
        // Snapshot every live shadow once, exactly as p3(). Seed and output never alias; xbar starts equal to ybar, so a
        // word the column does not touch keeps an identity bar.
        for s in p.segs.v.iter().filter(|s| s.active) { dev::copy(cuda, s.snap, s.shadow, s.shadow_bytes())?; }
        // The capture forward writes its outputs in place: it runs on a copy of the launch's pre-state, never on it.
        let arow: Vec<ArgRow> = row.args.clone();
        let mut copy_bytes = 0usize; let mut plan: Vec<(usize, u64, usize, usize)> = Vec::new();   // (pos, pre, bytes, copy offset)
        for a in arow.iter().filter(|a| a.device && a.name != "workspace") {
            match p.args.get(a.pos).and_then(target) {
                None => {},
                Some((al, off)) => {
                    if a.role == "absent" { return Err(format!("MYNN direct: argument {} is bound but the row marks it absent", a.name)); }
                    let b = row.extent_bytes(a, &env)?;
                    plan.push((a.pos, p.segs.pre(al, off)?, b, copy_bytes)); copy_bytes += b.div_ceil(256) * 256;
                },
            }
        }
        let wstride = u64::from_le_bytes(scalar_bytes(&p.args[row.arg("workspace_stride")?.pos]).ok_or("workspace_stride")?[..8]
            .try_into().map_err(|_| "workspace_stride width")?) as usize;
        let batch = env_var("AD_P3_DIRECT_BATCH").and_then(|v| v.parse::<usize>().ok()).unwrap_or(16384).max(32) / 32 * 32;   // 16384: 4090 capture 136 against 167 ms at 8192 (38,278 columns)
        let nb_max = batch.min(cols.div_ceil(32) * 32);
        let wbytes = env_var("AD_P3_DIRECT_WS").and_then(|v| v.parse::<usize>().ok()).unwrap_or(49152);
        let sc = &mut self.scratch;
        let mut need = |k: &'static str, b: usize| -> Result<u64, String> { sc.entry(k).or_default().need_exact(cuda, b) };
        let copies = need("p3_direct_copy", copy_bytes.max(256))?;
        let cws = need("p3_direct_fws", nb_max * wstride)?;
        let frames = need("p3_direct_frames", nb_max * fb)?;
        let bars = need("p3_direct_bars", nb_max * bb)?;
        let bws = need("p3_direct_bws", nb_max * wbytes)?;
        let failures = need("p3_direct_fail", nb_max * 16)?;
        let descriptor = need("p3_direct_launch", 4496)?;
        for &(_, pre, b, at) in &plan { dev::copy(cuda, copies + at as u64, pre, b)?; }
        let ptr_of = |pos: usize| plan.iter().find(|x| x.0 == pos).map(|x| copies + x.3 as u64);
        // MwLaunch rows: x = pre (unused by the backward), ybar = the snapshot, xbar = the shadow
        let mut head = Vec::<u8>::with_capacity(4496);
        let put = |w: &mut Vec<u8>, v: u64| w.extend_from_slice(&v.to_le_bytes());
        for name in row.raw["float_arrays"].as_array().ok_or("float_arrays")? {
            let a = row.arg(name.as_str().ok_or("float array name")?)?;
            let slot = match p.args.get(a.pos).and_then(target) {
                None => [0u64; 6],
                Some((allocation, offset)) => [p.segs.pre(allocation, offset)?, 0, 0, 0,
                    p.segs.snap(allocation, offset)?, p.segs.shadow(allocation, offset)?],
            };
            for value in slot { put(&mut head, value); }
        }
        let a = row.arg("kpbl")?;
        let (allocation, offset) = p.args.get(a.pos).and_then(target).ok_or("MYNN direct kpbl")?;
        let kpbl = p.segs.pre(allocation, offset)?;
        for value in [kpbl, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, failures] { put(&mut head, value); }
        let scalars: Vec<K> = row.raw["scalars"].as_array().ok_or("scalars")?.iter().map(|s| {
            let name = s.as_str().and_then(|s| s.split_whitespace().last()).ok_or("MYNN direct scalar name")?;
            Ok(K::B(scalar_bytes(&p.args[row.arg(name)?.pos]).ok_or_else(|| format!("MYNN direct scalar {name}"))?.to_vec()))
        }).collect::<Result<_, String>>()?;
        let (mut launches, mut b0) = (0usize, 0usize);
        let redo_max = env_var("AD_P3_DIRECT_REDO_MAX").and_then(|v| v.parse::<usize>().ok()).unwrap_or(64);
        let mut taped: Vec<i64> = Vec::new();
        while b0 < cols {
            let nb = batch.min(cols - b0);
            // (1) the forward with captures: zeroed frames and workspace, as the plan clears the forward's workspace
            dev::zero(cuda, frames, nb * fb)?;
            dev::zero(cuda, cws, nb.div_ceil(32) * 32 * wstride)?;
            let mut fa: Vec<K> = Vec::with_capacity(arow.len() + 2);
            for a in &arow {
                fa.push(match a.name.as_str() {
                    "batch_first" => i32b((first + b0 as i64) as i32),
                    "batch_count" => i32b(nb as i32),
                    "workspace" => K::P(cws),
                    _ if a.device => K::P(ptr_of(a.pos).unwrap_or(0)),
                    _ => K::B(scalar_bytes(&p.args[a.pos]).ok_or_else(|| format!("MYNN direct capture scalar {}", a.name))?.to_vec()),
                });
            }
            fa.push(K::P(frames)); fa.push(K::B((fb as u64).to_le_bytes().to_vec()));
            launch(cuda, capture, grid1(nb, 32), [32, 1, 1], &fa)?;
            // (2) the joined backward over the same columns
            let mut words = head.clone();
            for value in [b0 as i32, nb as i32, 1, 0] { words.extend_from_slice(&value.to_le_bytes()); }
            if words.len() != 4496 { return Err("MYNN direct launch descriptor size mismatch".into()); }
            dev::upload(cuda, descriptor, &words)?;
            let mut args = vec![K::P(descriptor), K::P(frames), K::B((fb as u64).to_le_bytes().to_vec()), K::P(bars), K::P(bws), i32b(wbytes as i32)];
            args.extend(scalars.iter().cloned());
            launch(cuda, function, grid1(nb, threads), [threads as u32, 1, 1], &args)?;
            dev::sync(cuda)?;
            let flags = dev::download(cuda, failures, nb * 16)?;
            for (i, b) in flags.chunks_exact(16).enumerate() {
                let code = i32::from_le_bytes(b[..4].try_into().unwrap());
                // tape-free fall-back (2): a refused column (a block, a live plume, a frame not written, workspace) goes alone
                // to the taped route after this launch; the kernel wrote none of its words. Launch-wide codes (switches,
                // initflag) and more than AD_P3_DIRECT_REDO_MAX (64) refused columns put the whole launch on the taped route.
                if code != 0 && !matches!(code & 63, 5 | 6) && taped.len() < redo_max {
                    eprintln!("{}", json!({"event":"mynn_direct_column_refused","column":b0 + i,"code":code}));
                    taped.push((b0 + i) as i64); continue;
                }
                if code != 0 {
                    let i = i * 4;
                    let why = match code & 63 { 1 => "geometry", 2 => "live plume (not in the join)", 3 => "a block refused", 5 => "launch switches differ from the proven recipe",
                                                6 => "initflag launch (f_mym_initialize is not in the join)", 7 => "frame not written by the capture", _ => "unknown" };
                    for s in p.segs.v.iter().filter(|s| s.active) { dev::copy(cuda, s.shadow, s.snap, s.shadow_bytes())?; }
                    eprintln!("{}", json!({"event":"mynn_direct_taped","column":b0 + i / 4,"slot":i % 4,"code":code,
                        "why":format!("{why}{}", if code & 64 != 0 { ", workspace too small: raise AD_P3_DIRECT_WS" } else { "" }),
                        "batches_done":launches,"then":"shadows put back from the entry snapshot; this launch reverses on the taped route"}));
                    return Ok(None);
                }
            }
            launches += 1; b0 += nb;
        }
        Ok(Some(json!({"direct":true,"columns":cols,"launches":launches,"batch":batch,"count_passes":0,"tape_bytes":0,"threads":threads,
                  "copy_bytes":copy_bytes,"frame_bytes":fb,"taped_columns":taped})))
    }
}
