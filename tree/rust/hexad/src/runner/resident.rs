// SPDX-License-Identifier: Apache-2.0
//! resident (2026-11-13): a new start of a storm planted IN MEMORY,
//! with no moved package written, and a gradient program that stays up between starts.
//!
//! 1. `ad-binder x0-plant MAP REF_DUMP NEW_DUMP PKG OUT` does what the recorder's x0_plant.py `plant`
//!    does, with the same refusals and the same words taken, but it writes no package: only the payloads that change,
//!    raw, into OUT, and OUT/planted.json naming each one by allocation (initial bytes) or by step_01 op index (an
//!    upload), with its old and its new sha256. THE COST THIS REMOVES: x0_plant.py took 19.8 to 25 s of every later
//!    gradient (no-recording run 1 x0-plant-nrl-s96b5.log: dumps read 3.5 s, 34 payloads gzip-written 11 s, plan.json rewritten
//!    5.3 s), and the program then inflated those payloads again.
//! 2. `ad-binder window ... --x0-planted OUT` (default off) runs PKG as if it were x0_plant.py's moved package: before
//!    anything reads the plan, the blob of every named allocation and op is swapped for the planted bytes (a File blob
//!    with the new sha256, which `crate::blob` reads and checks as it checks every blob). Every other byte the card
//!    sees is PKG's, which is what the moved package holds (x0_plant.py copies PKG with `cp -al` and changes only
//!    these payloads and their names in plan.json).
//! 3. `--resident DIR` (default off, window mode, see window.rs `Resident`): after its gradient the program does not
//!    exit. It answers DIR/NAME.done, waits for DIR/NAME.req and runs the next gradient in the same process, with the
//!    plan, the card context, the kernel set, the scheme images, the tape and scratch arenas, the page-locked pools and
//!    the step-1 staging kept. A request may change only the words of `REQUEST_WORDS` (a new start's --x0-planted folder,
//!    the objective, out, the store folder). Each request starts from the first request's boundary 0, kept in the store,
//!    with the allocations planted then or now uploaded from the plan's current payloads (`Resident::swap`,
//!    `Resident::replant_ids`): the words a fresh program given that request's command would upload in ex.prepare.
//!    THE COST THIS REMOVES from every later gradient of a storm: the program start (9.31 s in back-1 fin-later: plan
//!    read and typed 2.68, context 0.31, allocations and start bytes 2.73, kernel set 2.04, boundary 0 0.85, x 0.46), the
//!    first step's warm-up (about 2.1 s), the run wrapper (about 2.6 s) and the process exit.
//!    `ad-binder resident-selftest PKG PLANTED...` checks the swap and the re-planted start on the processors.
//!
//! The off state: no --x0-planted and no --resident word, and nothing here runs.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

pub const SCHEMA: &str = "speed16.x0-planted/v1";

fn sha_hex(b: &[u8]) -> String { format!("{:x}", Sha256::digest(b)) }
fn rd_json(p: &Path) -> Result<Value, String> {
    serde_json::from_slice(&fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?).map_err(|e| format!("{}: {e}", p.display()))
}
fn u(v: &Value, k: &str) -> Result<usize, String> {
    v.get(k).and_then(Value::as_u64).map(|x| x as usize).ok_or_else(|| format!("missing integer {k}"))
}
fn s<'a>(v: &'a Value, k: &str) -> Option<&'a str> { v.get(k).and_then(Value::as_str).filter(|x| !x.is_empty()) }
fn truthy(v: Option<&Value>) -> bool {
    match v { None | Some(Value::Null) => false, Some(Value::Bool(b)) => *b, Some(Value::Number(n)) => n.as_f64() != Some(0.0),
              Some(Value::String(x)) => !x.is_empty(), Some(Value::Array(a)) => !a.is_empty(), Some(Value::Object(o)) => !o.is_empty() }
}

/// One dump of x0_dump.sh: blocks.json (+ blocks.bin, host.bin), uploads.json (+ uploads.bin), DONE.json, case.env.
pub struct Dump { pub dir: PathBuf, pub blocks: Vec<Value>, pub host: Vec<Value>, pub uploads: Vec<Value>,
                  bb: Option<fs::File>, hb: Option<fs::File>, ub: Option<fs::File> }
impl Dump {
    pub fn open(dir: &Path) -> Result<Self, String> {
        let j = rd_json(&dir.join("blocks.json"))?;
        let up = dir.join("uploads.json");
        let uploads = if up.exists() { rd_json(&up)?["uploads"].as_array().cloned().unwrap_or_default() } else { Vec::new() };
        let open = |n: &str| -> Option<fs::File> { let p = dir.join(n); fs::metadata(&p).ok().filter(|m| m.len() > 0).and_then(|_| fs::File::open(p).ok()) };
        Ok(Self { dir: dir.to_owned(), blocks: j["blocks"].as_array().cloned().unwrap_or_default(),
                  host: j["host"].as_array().cloned().unwrap_or_default(),
                  bb: open("blocks.bin"), hb: open("host.bin"), ub: if uploads.is_empty() { None } else { open("uploads.bin") }, uploads })
    }
    fn read(f: &Option<fs::File>, off: usize, n: usize, what: &str) -> Result<Vec<u8>, String> {
        let f = f.as_ref().ok_or_else(|| format!("{what}: file empty or missing"))?;
        let mut v = vec![0u8; n];
        f.read_exact_at(&mut v, off as u64).map_err(|e| format!("{what} at {off} (+{n}): {e}"))?;
        Ok(v)
    }
}

/// `f(0..n)` on all cores (largest first is not needed: items are handed out one at a time), results in order.
fn par<T: Send>(n: usize, f: impl Fn(usize) -> Result<T, String> + Sync) -> Result<Vec<T>, String> {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).clamp(1, 32).min(n.max(1));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let slots: Vec<std::sync::Mutex<Option<Result<T, String>>>> = (0..n).map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|sc| { for _ in 0..threads { sc.spawn(|| loop {
        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if i >= n { break; }
        let r = f(i);
        *slots[i].lock().unwrap() = Some(r);
    }); } });
    slots.into_iter().map(|m| m.into_inner().unwrap().unwrap_or_else(|| Err("worker panicked".into()))).collect()
}

fn case_env(d: &Path) -> Option<HashMap<String, String>> {
    let t = fs::read_to_string(d.join("case.env")).ok()?;
    Some(t.lines().filter(|l| l.contains('=') && !l.trim_start().starts_with('#'))
        .map(|l| { let (k, v) = l.split_once('=').unwrap(); (k.to_owned(), v.to_owned()) }).collect())
}
fn realpath(p: &str) -> String { fs::canonicalize(p).map(|x| x.display().to_string()).unwrap_or_else(|_| {
    // os.path.realpath of a path that does not exist is its absolute form
    let q = Path::new(p); if q.is_absolute() { p.to_owned() } else { std::env::current_dir().map(|c| c.join(q).display().to_string()).unwrap_or_else(|_| p.to_owned()) } }) }

/// The first `"layer_groups": {...}` (no brace inside) of plan.json's text, as x0_plant.py's regex finds it.
pub fn layer_groups_text(text: &[u8]) -> Option<&[u8]> {
    let key = b"\"layer_groups\":";
    let mut from = 0;
    while let Some(i) = text[from..].windows(key.len()).position(|w| w == key) {
        let mut j = from + i + key.len();
        while j < text.len() && matches!(text[j], b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c) { j += 1; }
        if j < text.len() && text[j] == b'{' {
            let mut k = j + 1;
            while k < text.len() && text[k] != b'{' && text[k] != b'}' { k += 1; }
            if k < text.len() && text[k] == b'}' { return Some(&text[j..=k]); }
        }
        from = from + i + 1;
    }
    None
}

/// x0_plant.py case_and_lw_check: the refusals before any planting (another mesh or step; another start time or
/// boundary files unless X0_ALLOW_OTHER_TIME=1; step-1 long-wave layer counts the recording has no batch for).
pub fn case_and_lw_check(pkg: &Path, refd: &Path, newd: &Path, log: &mut Vec<String>) -> Result<(), String> {
    match (case_env(refd), case_env(newd)) {
        (Some(c0), Some(c1)) => {
            let bad: Vec<&str> = ["LAM_MESH_ROW", "LAM_DT"].into_iter().filter(|k| c0.get(*k) != c1.get(*k)).collect();
            if !bad.is_empty() { return Err(format!("DEFECT: the new start is another mesh or time step than the recording: {bad:?}")); }
            let other: Vec<&str> = ["LAM_START", "LAM_LBC_DIR"].into_iter().filter(|k| {
                let (a, b) = (c0.get(*k).map(String::as_str).unwrap_or(""), c1.get(*k).map(String::as_str).unwrap_or(""));
                if *k == "LAM_LBC_DIR" { realpath(a) != realpath(b) } else { c0.get(*k) != c1.get(*k) } }).collect();
            if !other.is_empty() {
                if std::env::var("X0_ALLOW_OTHER_TIME").as_deref() != Ok("1") {
                    return Err(format!("DEFECT: the new start has another start time or other boundary files than the recording: {other:?}. Use a recording of this start time."));
                }
                log.push(format!("PROBE ONLY (X0_ALLOW_OTHER_TIME=1): another start time or boundary files {other:?}"));
            }
        }
        _ => log.push("note: no case.env in a dump; the case is not compared".into()),
    }
    let live = rd_json(&newd.join("DONE.json")).ok().and_then(|d| d.get("lw_layer_counts").cloned()).filter(|v| truthy(Some(v)));
    let text = fs::read(pkg.join("plan.json")).map_err(|e| format!("{}/plan.json: {e}", pkg.display()))?;
    let m = layer_groups_text(&text);
    let (Some(live), Some(m)) = (live, m) else { log.push("note: step-1 long-wave layer counts not compared".into()); return Ok(()); };
    let g: Value = serde_json::from_slice(m).map_err(|e| format!("layer_groups: {e}"))?;
    let mut rec: BTreeMap<i64, i64> = BTreeMap::new();
    for b in g["batches"].as_array().ok_or("layer_groups batches")? {
        let nl = b[0].as_i64().ok_or("batch layer count")?; let n = b[1].as_i64().ok_or("batch columns")?;
        *rec.entry(nl).or_default() += n;
    }
    let l0 = &live[0];
    if l0.get("error").is_some() { log.push("note: step-1 long-wave layer counts not read by the dump".into()); return Ok(()); }
    let lv: BTreeMap<i64, i64> = l0.as_object().ok_or("lw_layer_counts[0]")?.iter()
        .map(|(k, v)| Ok((k.parse::<i64>().map_err(|_| "lw count key")?, v.as_i64().or_else(|| v.as_str().and_then(|x| x.parse().ok())).ok_or("lw count")?)))
        .collect::<Result<_, String>>()?;
    log.push(format!("step-1 long-wave layer counts: live {lv:?}, recorded {rec:?}"));
    if lv.keys().ne(rec.keys()) {
        return Err(format!("DEFECT: the gradient program would refuse this start at step 1: it needs a recording whose step-1 long-wave layer counts are {:?}", lv.keys().collect::<Vec<_>>()));
    }
    Ok(())
}

fn gunzip(path: &Path, expected: usize) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let f = fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut r = flate2::read::MultiGzDecoder::new(std::io::BufReader::with_capacity(4 << 20, f));
    let mut v = Vec::with_capacity(expected);
    r.read_to_end(&mut v).map_err(|e| format!("{}: invalid gzip: {e}", path.display()))?;
    Ok(v)
}

/// What a plant gives: per allocation the new initial bytes, per step_01 op index the new upload bytes, and the record.
/// allocs: allocation -> (old sha256, new sha256, new bytes); uploads: step_01 op -> (old sha256, new bytes).
pub struct Planted { pub allocs: BTreeMap<u64, (String, String, Vec<u8>)>, pub uploads: BTreeMap<usize, (String, Vec<u8>)>, pub record: Value }

/// x0_plant.py cmd_plant, in memory. `base_initial(a)` gives PKG's initial bytes of allocation a and its sha256.
pub fn plant(map: &Value, refd: &Dump, newd: &Dump, base_initial: &(dyn Fn(u64) -> Result<(String, Vec<u8>), String> + Sync)) -> Result<Planted, String> {
    let t0 = std::time::Instant::now();
    let by_path = |d: &Dump| -> HashMap<String, Value> { d.blocks.iter().filter_map(|b| s(b, "path").map(|p| (p.to_owned(), b.clone()))).collect() };
    let (r0, r1) = (by_path(refd), by_path(newd));
    // changed blocks: (ref words, new words) by index; plant rows per allocation: (byte offset, block index, path)
    // the mapped blocks of both dumps, read on all cores; a block whose words are equal in both plants nothing
    let mb = map["blocks"].as_array().ok_or("map blocks")?;
    let mut missing = Vec::new();
    let mut jobs: Vec<(usize, &Value, &Value)> = Vec::new();
    for (i, b) in mb.iter().enumerate() {
        let p = s(b, "path").ok_or("map block path")?;
        let (Some(x0), Some(x1)) = (r0.get(p), r1.get(p)) else { missing.push(p.to_owned()); continue };
        if (x0.get("lo"), x0.get("hi")) != (x1.get("lo"), x1.get("hi")) { missing.push(p.to_owned()); continue; }
        jobs.push((i, x0, x1));
    }
    if !missing.is_empty() {
        return Err(format!("DEFECT: {} mapped blocks are not in the new dump or have another extent: {:?}", missing.len(), &missing[..missing.len().min(5)]));
    }
    let read_pair = |x0: &Value, x1: &Value| -> Result<Option<(Vec<u8>, Vec<u8>)>, String> {
        let ca = Dump::read(&refd.bb, u(x0, "file_offset")?, u(x0, "bytes")?, "ref blocks.bin")?;
        let cb = Dump::read(&newd.bb, u(x1, "file_offset")?, u(x1, "bytes")?, "new blocks.bin")?;
        let (na, nb) = (ca.len() - ca.len() % 4, cb.len() - cb.len() % 4);
        if na != nb { return Err(format!("DEFECT: block {}: {} words in the first start's dump, {} in the new one", s(x0, "path").unwrap_or("?"), na / 4, nb / 4)); }
        Ok(if ca[..na] == cb[..nb] { None } else { Some((ca, cb)) })
    };
    let pairs = par(jobs.len(), |j| read_pair(jobs[j].1, jobs[j].2))?;
    // plant rows per allocation, in map order: (byte offset, index into pairs, block path)
    let mut rows: BTreeMap<u64, Vec<(usize, usize, String)>> = BTreeMap::new();
    let (mut same, mut nchanged) = (0usize, 0usize);
    for (j, (i, _, _)) in jobs.iter().enumerate() {
        if pairs[j].is_none() { same += 1; continue; }
        nchanged += 1;
        let b = &mb[*i];
        for pl in b["places"].as_array().ok_or("map places")? {
            let a = pl[0].as_u64().ok_or("place allocation")?; let o = pl[1].as_u64().ok_or("place offset")? as usize;
            rows.entry(a).or_default().push((o, j, s(b, "path").unwrap_or("").to_owned()));
        }
    }
    // per allocation (on all cores): PKG's payload, and the new start's words where the first start's dump agrees with it
    let ids: Vec<(u64, &Vec<(usize, usize, String)>)> = rows.iter().map(|(a, r)| (*a, r)).collect();
    let w = |b: &[u8], i: usize| u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap());
    let one = |a: u64, rs: &Vec<(usize, usize, String)>| -> Result<(Value, Option<(String, String, Vec<u8>)>, u64), String> {
        let (old_sha, orig) = base_initial(a)?;
        let mut out = orig.clone();
        let nw = orig.len() / 4;
        let mut written = vec![false; nw];
        let (mut took, mut left, mut clash) = (0u64, 0u64, 0u64);
        for (o, j, _) in rs {
            let (ca, cb) = pairs[*j].as_ref().unwrap();
            let n = ca.len() / 4;
            let k0 = o / 4;
            if k0 + n > nw { return Err(format!("DEFECT: allocation {a}: a place at byte {o} (+{n} words) runs past its {nw} words")); }
            for i in 0..n {
                let (wa, wb) = (w(ca, i), w(cb, i));
                if wa == wb { continue; }
                if w(&orig, k0 + i) != wa { left += 1; continue; }
                if written[k0 + i] && w(&out, k0 + i) != wb { clash += 1; }
                out[4 * (k0 + i)..4 * (k0 + i) + 4].copy_from_slice(&wb.to_le_bytes());
                written[k0 + i] = true;
                took += 1;
            }
        }
        let mut paths: Vec<&str> = rs.iter().map(|r| r.2.as_str()).collect(); paths.sort(); paths.dedup();
        let rec = json!({"a":a,"bytes":orig.len(),"blocks":rs.len(),"words_taken":took,"words_new_differs_but_noise_left":left,"clashes":clash,"paths":&paths[..paths.len().min(6)]});
        Ok((rec, if took > 0 { let h = sha_hex(&out); Some((old_sha, h, out)) } else { None }, clash))
    };
    let t_read = t0.elapsed().as_secs_f64();
    let done = par(ids.len(), |k| one(ids[k].0, ids[k].1))?;
    let t_plant = t0.elapsed().as_secs_f64();
    let mut allocs = BTreeMap::new();
    let mut arec = Vec::new();
    let mut clashes_any = false;
    for ((a, _), (rec, new, clash)) in ids.iter().zip(done) {
        clashes_any |= clash > 0;
        arec.push(rec);
        if let Some(x) = new { allocs.insert(*a, x); }
    }
    if clashes_any && std::env::var("X0_ALLOW_CLASH").as_deref() != Ok("1") {
        return Err(format!("DEFECT: two blocks put different words at one place: {:?}", arec.iter().filter(|r| r["clashes"].as_u64() != Some(0)).take(3).collect::<Vec<_>>()));
    }
    // step-1 uploads
    let hp1: HashMap<&str, &Value> = newd.host.iter().filter_map(|h| s(h, "path").map(|p| (p, h))).collect();
    let mut uploads = BTreeMap::new();
    let mut urec = Vec::new();
    for up in map["uploads"].as_array().ok_or("map uploads")? {
        let ub = u(up, "bytes")?;
        let mut data: Option<Vec<u8>> = None;
        if let Some(h) = s(up, "host_path").and_then(|p| hp1.get(p)) {
            data = Some(Dump::read(&newd.hb, u(h, "file_offset")?, u(h, "bytes")?, "new host.bin")?);
        } else if let Some(want) = s(up, "end_path").or_else(|| s(up, "end_host_path")) {
            if let Some(v) = newd.uploads.iter().find(|v| s(v, "path") == Some(want) && v.get("bytes").and_then(Value::as_u64) == Some(ub as u64)) {
                data = Some(Dump::read(&newd.ub, u(v, "file_offset")?, u(v, "bytes")?, "new uploads.bin")?);
            }
        } else if truthy(up.get("nth_of_size")) {
            let nth = u(up, "nth_of_size")?;
            let same: Vec<&Value> = newd.uploads.iter().filter(|v| v.get("bytes").and_then(Value::as_u64) == Some(ub as u64)).collect();
            if same.len() >= nth { let v = same[nth - 1]; data = Some(Dump::read(&newd.ub, u(v, "file_offset")?, u(v, "bytes")?, "new uploads.bin")?); }
        }
        let op = u(up, "op_index")?;
        let Some(d) = data else { urec.push(json!({"op_index":op,"bytes":ub,"new":"NOT IN THE NEW DUMP: left as recorded"})); continue };
        let sha = sha_hex(&d);
        let old = s(up, "sha256").unwrap_or("").to_owned();
        urec.push(json!({"op_index":op,"bytes":ub,"new_sha256":sha,"changed":sha != old}));
        if sha != old { uploads.insert(op, (old, d)); }
    }
    let record = json!({"blocks_changed":nchanged,"blocks_same":same,"allocations":arec,"uploads":urec,
        "phase_s":{"dump_blocks_read":t_read,"payloads_inflated_and_planted":t_plant - t_read,"uploads":t0.elapsed().as_secs_f64() - t_plant}});
    Ok(Planted { allocs, uploads, record })
}

/// `ad-binder x0-plant MAP REF_DUMP NEW_DUMP PKG OUT`: x0_plant.py's plant with no package written (module docs).
pub fn plant_cmd(args: Vec<String>) -> Result<(), String> {
    let [mapp, refp, newp, pkgp, outp]: [String; 5] = args.try_into().map_err(|_| "usage: ad-binder x0-plant MAP REF_DUMP NEW_DUMP PKG OUT".to_string())?;
    let t0 = std::time::Instant::now();
    let say = |m: String| eprintln!("[{:6.1} s] {m}", t0.elapsed().as_secs_f64());
    let (pkg, out) = (fs::canonicalize(&pkgp).map_err(|e| format!("{pkgp}: {e}"))?, PathBuf::from(&outp));
    let map = rd_json(Path::new(&mapp))?;
    let (refd, newd) = (Dump::open(Path::new(&refp))?, Dump::open(Path::new(&newp))?);
    let mut log = Vec::new();
    let r = case_and_lw_check(&pkg, Path::new(&refp), Path::new(&newp), &mut log);
    for l in log { say(l); }
    r?;
    // PKG's initial payload of an allocation: from the map's "initial" rows (as x0_plant.py), else from plan.json.
    let initial: HashMap<u64, Value> = match map.get("initial").and_then(Value::as_object) {
        Some(o) => o.iter().filter_map(|(k, v)| k.parse::<u64>().ok().map(|a| (a, v.clone()))).collect(),
        None => rd_json(&pkg.join("plan.json"))?["allocations"].as_array().ok_or("plan allocations")?.iter()
            .filter_map(|x| Some((x["id"].as_u64()?, x.get("initial")?.clone()))).filter(|(_, v)| !v.is_null()).collect(),
    };
    let base_initial = |a: u64| -> Result<(String, Vec<u8>), String> {
        let d = initial.get(&a).ok_or_else(|| format!("allocation {a} has no initial payload in the map or plan"))?;
        let (p, sha, n) = (s(d, "path").ok_or("initial path")?, s(d, "sha256").ok_or("initial sha256")?, u(d, "bytes")?);
        // inflated straight into its buffer with the SHA-256 beside it (crate::gzip_bytes_verified, the program's own reader)
        let refuse = |e: String| format!("allocation {a}: PKG payload {p} is not the {n} bytes of sha256 {sha} the map names (another package?): {e}");
        let b = if s(d, "encoding") == Some("gzip") { crate::gzip_bytes_verified(&pkg.join(p), n, sha).map_err(refuse)? } else {
            let b = fs::read(pkg.join(p)).map_err(|e| e.to_string())?;
            if b.len() != n || sha_hex(&b) != sha { return Err(refuse("size or sha256".into())); }
            b };
        Ok((sha.to_owned(), b))
    };
    let pl = plant(&map, &refd, &newd, &base_initial)?;
    say(format!("dumps read; {} blocks changed, {} same; {} allocations to rewrite (phases: {})", pl.record["blocks_changed"], pl.record["blocks_same"], pl.allocs.len(), pl.record["phase_s"]));
    // checked before anything is removed: OUT inside the recording, or the recording inside OUT, would delete recorded
    // payloads. THE BREAKAGE THIS PREVENTS: an OUT of PKG/recording (or of a folder holding PKG) emptied before the refusal.
    if out.exists() {
        let oc = fs::canonicalize(&out).map_err(|e| format!("{}: {e}", out.display()))?;
        if oc.starts_with(&pkg) || pkg.starts_with(&oc) { return Err(format!("DEFECT: OUT {} and the recording {} hold one another: nothing removed", oc.display(), pkg.display())); }
        fs::remove_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    }
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let outc = fs::canonicalize(&out).map_err(|e| e.to_string())?;
    if outc.starts_with(&pkg) { return Err(format!("DEFECT: OUT {} lies inside the recording {}", outc.display(), pkg.display())); }
    let mut arows = Vec::new();
    let mut urows = Vec::new();
    let mut jobs: Vec<(PathBuf, &[u8])> = Vec::new();
    for (a, (old, new, b)) in &pl.allocs {
        let f = format!("a-{a}.raw");
        arows.push(json!({"id":a,"bytes":b.len(),"old_sha256":old,"sha256":new,"file":f}));
        jobs.push((outc.join(&f), b));
    }
    for (op, (old, b)) in &pl.uploads {
        let f = format!("u-step_01-{op}.raw");
        urows.push(json!({"block":"step_01","op_index":op,"bytes":b.len(),"old_sha256":old,"sha256":sha_hex(b),"file":f}));
        jobs.push((outc.join(&f), b));
    }
    std::thread::scope(|sc| -> Result<(), String> {
        let hs: Vec<_> = jobs.iter().map(|(p, b)| sc.spawn(move || fs::write(p, b).map_err(|e| format!("{}: {e}", p.display())))).collect();
        for h in hs { h.join().map_err(|_| "writer panicked")??; }
        Ok(())
    })?;
    // the empty move a moved package carries, so the gradient line runs with --x0-dir as on a moved package
    fs::write(outc.join("x0-empty.json"), serde_json::to_vec(&json!({"name":"empty move: the start state is planted in memory (speed16 resident x0-plant)","terms":[]})).unwrap()).map_err(|e| e.to_string())?;
    let doc = json!({"schema":SCHEMA,"pkg":pkg.display().to_string(),"map":realpath(&mapp),"ref_dump":realpath(&refp),"new_dump":realpath(&newp),
        "allocations":arows,"uploads":urows,"record":pl.record,"wall_s":t0.elapsed().as_secs_f64()});
    fs::write(outc.join("planted.json"), serde_json::to_vec_pretty(&doc).unwrap()).map_err(|e| e.to_string())?;
    say(format!("planted {}: {} allocations carry the new start; {} step-1 uploads carry its bytes", outc.display(), pl.allocs.len(), pl.uploads.len()));
    Ok(())
}

/// `--x0-planted DIR` in window mode: swap the plan's blobs for the planted bytes before anything reads them.
/// Refused: a DIR planted against another recording (a swapped allocation or upload whose sha256 in this plan is not
/// the one it was planted over; every swapped payload is named by the sha256 of its recorded bytes, so this is the
/// recording's own check, wherever the runner put the plan file): THE BREAKAGE THIS PREVENTS is a new start's words laid
/// into another recording's allocations. The folder the plan was read from is only written down beside DIR's.
pub fn install(plan: &mut crate::plan::Plan, base: &Path, dir: &Path) -> Result<Value, String> {
    use crate::plan::{Blob, Operation};
    let doc = rd_json(&dir.join("planted.json"))?;
    if doc["schema"] != SCHEMA { return Err(format!("--x0-planted {}: not a {SCHEMA} folder", dir.display())); }
    let basec = fs::canonicalize(base).map(|p| p.display().to_string()).unwrap_or_else(|_| base.display().to_string());
    let dirc = fs::canonicalize(dir).map_err(|e| e.to_string())?;
    let sha_of = |b: &Blob| -> String { match b { Blob::Hex { sha256, .. } | Blob::File { sha256, .. } | Blob::Gzip { sha256, .. } => sha256.clone() } };
    let mut n = (0usize, 0usize);
    for r in doc["allocations"].as_array().ok_or("planted allocations")? {
        let id = r["id"].as_u64().ok_or("planted id")?;
        let x = plan.allocations.iter_mut().find(|x| x.id == id).ok_or_else(|| format!("--x0-planted: allocation {id} is not in the plan"))?;
        let old = x.initial.as_ref().map(sha_of).unwrap_or_default();
        if Some(old.as_str()) != r["old_sha256"].as_str() { return Err(format!("--x0-planted: allocation {id} holds {old}, planted over {}", r["old_sha256"])); }
        if r["bytes"].as_u64() != Some(x.bytes as u64) { return Err(format!("--x0-planted: allocation {id} is {} bytes, planted {}", x.bytes, r["bytes"])); }
        x.initial = Some(Blob::File { path: dirc.join(r["file"].as_str().ok_or("planted file")?).display().to_string(), sha256: r["sha256"].as_str().ok_or("planted sha256")?.to_owned() });
        n.0 += 1;
    }
    for r in doc["uploads"].as_array().ok_or("planted uploads")? {
        let blk = r["block"].as_str().ok_or("planted block")?; let op = r["op_index"].as_u64().ok_or("planted op")? as usize;
        let ops = plan.blocks.get_mut(blk).ok_or_else(|| format!("--x0-planted: no block {blk}"))?;
        let Some(Operation::Upload { data, .. }) = ops.get_mut(op) else { return Err(format!("--x0-planted: {blk}[{op}] is not an upload")) };
        let old = sha_of(data);
        if Some(old.as_str()) != r["old_sha256"].as_str() { return Err(format!("--x0-planted: {blk}[{op}] holds {old}, planted over {}", r["old_sha256"])); }
        *data = Blob::File { path: dirc.join(r["file"].as_str().ok_or("planted file")?).display().to_string(), sha256: r["sha256"].as_str().ok_or("planted sha256")?.to_owned() };
        n.1 += 1;
    }
    Ok(json!({"dir":dirc.display().to_string(),"allocations":n.0,"uploads":n.1,"new_dump":doc["new_dump"],"planted_against":doc["pkg"],"plan_folder":basec}))
}

/// One request to a resident gradient program: DIR/NAME.req, a JSON object of window words that replace the
/// command's for this gradient (at least "out"; usually "x0-planted" and "objective"). The program answers with
/// DIR/NAME.done ({"rc":0} or {"rc":1,"error":...}) once every file of the gradient is written.
pub struct Request { pub name: String, pub words: HashMap<String, String> }

/// Words a request may change. The others hold the program's loaded state (plan, kernel set, steps, store and tape
/// sizes) and are refused if a request names them with another value.
pub const REQUEST_WORDS: &[&str] = &["out", "x0-planted", "objective", "dump-grad", "dump-state", "bstore", "spill-dir", "x0-dir", "x0-h"];

/// The oldest DIR/NAME.req (renamed NAME.taken as it is read); None at DIR/stop or after `idle_s` with none.
pub fn next_request(dir: &Path, idle_s: f64) -> Result<Option<Request>, String> {
    let t0 = std::time::Instant::now();
    loop {
        if dir.join("stop").exists() { return Ok(None); }
        let mut reqs: Vec<PathBuf> = fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "req")).collect();
        reqs.sort();
        if let Some(p) = reqs.into_iter().next() {
            let name = p.file_stem().unwrap().to_string_lossy().into_owned();
            let taken = p.with_extension("taken");
            fs::rename(&p, &taken).map_err(|e| e.to_string())?;
            let v = rd_json(&taken)?;
            let words = v.as_object().ok_or("request: a JSON object of words")?.iter()
                .map(|(k, x)| (k.clone(), x.as_str().map(str::to_owned).unwrap_or_else(|| x.to_string()))).collect();
            return Ok(Some(Request { name, words }));
        }
        if t0.elapsed().as_secs_f64() > idle_s { return Ok(None); }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

fn write_atomic(p: &Path, v: &Value) -> Result<(), String> {
    let tmp = p.with_extension("tmp");
    fs::write(&tmp, serde_json::to_vec_pretty(v).unwrap()).map_err(|e| format!("{}: {e}", tmp.display()))?;
    fs::rename(&tmp, p).map_err(|e| format!("{}: {e}", p.display()))
}

/// The answer file of the request being served, for `fail` (window::run writes rc 1 there when the program stops on
/// an error, so a client never waits on a program that is gone).
static SERVING: std::sync::Mutex<Option<PathBuf>> = std::sync::Mutex::new(None);
pub fn fail(error: &str) {
    if let Some(p) = SERVING.lock().ok().and_then(|g| g.clone()) {
        let _ = write_atomic(&p, &ended_answer(error));
    }
}

/// dist47: the answer of a program that stops on an error: rc 1, the error, and `program: ended`, so a client knows
/// the next request needs a fresh program.
fn ended_answer(error: &str) -> Value { json!({"rc":1,"error":error,"program":"ended","pid":std::process::id()}) }
/// dist47: the end of the answer to a refused request the program survives.
pub const STILL_LOADED: &str = "This request is refused; the program is still loaded and answers the next request";

/// Words a resident program refuses, with what each would break. THE BREAKAGE THESE PREVENT: the program serves later
/// gradients from state one gradient left behind; only the gradient line (an objective reverse at a moved start with
/// checkpoints, no tangent) was checked for what it leaves, so the other forms would start a later request from state
/// nobody checked.
const REFUSED_WORDS: &[(&str, &str)] = &[("taylor", "Taylor rows"), ("taylor-dir", "Taylor rows"), ("fd-only", "difference quotients"),
    ("forward-only", "a forward-only window"), ("params", "parameter carriers"), ("param-fd", "parameter differences"), ("param-tl", "parameter tangents"),
    ("ext-driver", "an extension driver (it keeps its own trajectories)"), ("keep-bstore", "kept stores (their files are named per step, not per request)"),
    ("export-vectors", "vector export"), ("vector-gate", "vector gates"), ("dump-steps", "step dumps"), ("dump-at", "step dumps"), ("regime", "regime checks"),
    ("tangent-dir", "tangent directions"), ("tangent-taylor-dir", "tangent directions"), ("objective-reverse", "no-reverse runs"), ("grad-all", "more objectives")];
const REFUSED_ENV: &[&str] = &["AD_PLANT", "AD_PLANT_RECOMPUTE", "AD_PLANT_LEFT_OUT", "AD_DOT_DEAD_POISON", "AD_FUSE_TAPE"];

/// resident: the program's state between requests (window.rs drives it).
pub struct Resident {
    pub dir: PathBuf,
    pub idle_s: f64,
    /// the request being served and its number (1: the command's own gradient)
    pub name: String,
    pub index: usize,
    pub served: Vec<Value>,
    /// the recording's payload of every allocation and step_01 upload any request planted (taken before its first swap)
    orig_alloc: BTreeMap<u64, crate::plan::Blob>,
    orig_ops: BTreeMap<(String, usize), crate::plan::Blob>,
    /// what the plan holds swapped now (the current request's folder)
    cur_alloc: BTreeSet<u64>,
    cur_ops: BTreeSet<(String, usize)>,
    /// the allocations whose kept boundary-0 words are a planted start's, not the recording's (the first request's)
    pub keep_alloc: Option<BTreeSet<u64>>,
    pub t_req: std::time::Instant,
}

impl Resident {
    /// `--resident DIR [--resident-name NAME] [--resident-idle-s S]` (default off: None).
    pub fn from_words(a: &HashMap<String, String>) -> Result<Option<Self>, String> {
        let Some(d) = a.get("resident") else {
            if a.contains_key("resident-name") || a.contains_key("resident-idle-s") { return Err("--resident-name and --resident-idle-s need --resident DIR".into()); }
            return Ok(None);
        };
        for (k, what) in REFUSED_WORDS {
            if a.get(*k).is_some_and(|v| v != "0") { return Err(format!("--resident: --{k} ({what}) is not served by a resident program")); }
        }
        for k in REFUSED_ENV { if std::env::var(k).is_ok_and(|v| !v.is_empty() && v != "0") { return Err(format!("--resident: {k} is not served by a resident program")); } }
        for (k, v) in [("tangent-only", "1"), ("no-tangent", "1")] {
            if a.get(k).map(String::as_str) != Some(v) { return Err(format!("--resident serves the gradient line only: it needs --{k} {v}")); }
        }
        for k in ["objective", "x0-dir", "checkpoints"] {
            if !a.contains_key(k) { return Err(format!("--resident serves the gradient line only: it needs --{k}")); }
        }
        if a.get("seeds").is_some_and(|s| s.split(',').filter(|x| !x.trim().is_empty()).count() > 1) { return Err("--resident: one seed".into()); }
        let dir = PathBuf::from(d);
        fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let _ = fs::remove_file(dir.join("stop"));
        let idle_s = match a.get("resident-idle-s") { Some(v) => v.parse::<f64>().map_err(|_| format!("--resident-idle-s {v}: seconds"))?, None => 900.0 };
        let name = a.get("resident-name").cloned().unwrap_or_else(|| "r001".into());
        let env: BTreeMap<String, String> = std::env::vars().filter(|(k, _)| k.starts_with("AD_") || k.starts_with("CUDA_")).collect();
        let r = Self { dir, idle_s, name, index: 1, served: Vec::new(), orig_alloc: BTreeMap::new(), orig_ops: BTreeMap::new(),
                       cur_alloc: BTreeSet::new(), cur_ops: BTreeSet::new(), keep_alloc: None, t_req: std::time::Instant::now() };
        write_atomic(&r.dir.join("program.json"), &json!({"schema":"speed16.resident/v1","pid":std::process::id(),"words":a,"env":env,
            "request_words":REQUEST_WORDS,"idle_s":idle_s,"first":r.name}))?;
        r.state("loading")?;
        *SERVING.lock().map_err(|_| "resident lock")? = Some(r.dir.join(format!("{}.done", r.name)));
        Ok(Some(r))
    }
    pub fn state(&self, what: &str) -> Result<(), String> {
        write_atomic(&self.dir.join("state.json"), &json!({"state":what,"pid":std::process::id(),"request":self.name,"index":self.index,"served":self.served.len()}))
    }
    /// The plan's payloads for this request: every payload the last request swapped goes back to the recording's, then
    /// `planted`'s are swapped in by `install` (whose sha256 checks then hold against the recording).
    pub fn swap(&mut self, plan: &mut crate::plan::Plan, base: &Path, planted: Option<&str>) -> Result<Option<Value>, String> {
        use crate::plan::Operation;
        for id in std::mem::take(&mut self.cur_alloc) {
            let x = plan.allocations.iter_mut().find(|x| x.id == id).ok_or_else(|| format!("resident: allocation {id} left the plan"))?;
            x.initial = Some(self.orig_alloc[&id].clone());
        }
        for (blk, op) in std::mem::take(&mut self.cur_ops) {
            let Some(Operation::Upload { data, .. }) = plan.blocks.get_mut(&blk).and_then(|o| o.get_mut(op)) else { return Err(format!("resident: {blk}[{op}] is not an upload")) };
            *data = self.orig_ops[&(blk.clone(), op)].clone();
        }
        let Some(d) = planted else { return Ok(None) };
        let doc = rd_json(&Path::new(d).join("planted.json"))?;
        let (mut ids, mut ops) = (BTreeSet::new(), BTreeSet::new());
        for r in doc["allocations"].as_array().ok_or("planted allocations")? {
            let id = r["id"].as_u64().ok_or("planted id")?;
            let x = plan.allocations.iter().find(|x| x.id == id).ok_or_else(|| format!("--x0-planted: allocation {id} is not in the plan"))?;
            // re-planted by a raw upload (window.rs resident_replant): an allocation whose initial words the program
            // relocates (recorded addresses) would lose them. None is planted at the storm box (x0_plant.py plants fields).
            if !x.relocations.is_empty() { return Err(format!("resident: allocation {id} carries relocated address words and cannot be re-planted by upload")); }
            if let Some(b) = &x.initial { self.orig_alloc.entry(id).or_insert_with(|| b.clone()); }
            ids.insert(id);
        }
        for r in doc["uploads"].as_array().ok_or("planted uploads")? {
            let blk = r["block"].as_str().ok_or("planted block")?.to_owned(); let op = r["op_index"].as_u64().ok_or("planted op")? as usize;
            if let Some(Operation::Upload { data, .. }) = plan.blocks.get(&blk).and_then(|o| o.get(op)) { self.orig_ops.entry((blk.clone(), op)).or_insert_with(|| data.clone()); }
            ops.insert((blk, op));
        }
        let info = install(plan, base, Path::new(d))?;
        self.cur_alloc = ids; self.cur_ops = ops;
        Ok(Some(info))
    }
    /// The first request's boundary 0 is kept: from then on its planted allocations are the ones whose kept words are
    /// not the recording's.
    pub fn kept(&mut self) { self.keep_alloc = Some(self.cur_alloc.clone()); }
    /// keep-forward (window.rs AD_KEEP_FORWARD): the step-1 uploads this request's planted start holds swapped now; their
    /// payloads are part of the start a kept forward is compared by (they are not in boundary 0's bytes).
    pub fn cur_ops(&self) -> Vec<(String, usize)> { self.cur_ops.iter().cloned().collect() }
    /// The allocations a later request uploads over the kept boundary 0 from the plan's current payloads: planted in
    /// the kept start (back to the recording's words, or this request's) or planted now.
    pub fn replant_ids(&self) -> Vec<u64> {
        self.keep_alloc.iter().flatten().chain(self.cur_alloc.iter()).copied().collect::<BTreeSet<u64>>().into_iter().collect()
    }
    /// Answer the request just served.
    pub fn done(&mut self, out: &Path, extra: Value) -> Result<(), String> {
        let wall = self.t_req.elapsed().as_secs_f64();
        let mut v = json!({"rc":0,"name":self.name,"index":self.index,"out":out.display().to_string(),"wall_s":wall,"pid":std::process::id()});
        if let (Some(o), Some(e)) = (v.as_object_mut(), extra.as_object()) { for (k, x) in e { o.insert(k.clone(), x.clone()); } }
        write_atomic(&self.dir.join(format!("{}.done", self.name)), &v)?;
        self.served.push(v);
        Ok(())
    }
    /// dist47: answer a request the program refuses and survives (window.rs: the forecast stops at its start): rc 1,
    /// the refusal, and `program: loaded`, so a client knows the next request is served by this program.
    pub fn refused(&mut self, out: &Path, error: &str) -> Result<(), String> {
        let v = json!({"rc":1,"error":format!("{error}. {STILL_LOADED}"),"program":"loaded","name":self.name,"index":self.index,
                       "out":out.display().to_string(),"wall_s":self.t_req.elapsed().as_secs_f64(),"pid":std::process::id()});
        write_atomic(&self.dir.join(format!("{}.done", self.name)), &v)?;
        self.served.push(v);
        Ok(())
    }
    /// The next request: its words over the command's (merge_words). None: stop asked, or idle past --resident-idle-s.
    pub fn next(&mut self, words: &HashMap<String, String>) -> Result<Option<HashMap<String, String>>, String> {
        *SERVING.lock().map_err(|_| "resident lock")? = None;
        self.state("idle")?;
        let Some(req) = next_request(&self.dir, self.idle_s)? else { return Ok(None) };
        self.t_req = std::time::Instant::now();
        self.name = req.name.clone(); self.index += 1;
        *SERVING.lock().map_err(|_| "resident lock")? = Some(self.dir.join(format!("{}.done", self.name)));
        self.state("busy")?;
        Ok(Some(merge_words(words, &req)?))
    }
}

/// `ad-binder resident-selftest PKG PLANTED... ` (processors only): for each planted folder in turn ("-": none, the
/// recording's start), the start a resident program would hold at boundary 0 (the first request's start as ex.prepare
/// uploaded it, with `replant_ids` uploaded from the plan after `swap`) against the start a fresh program given that
/// folder alone would upload; and every upload payload of the plan against the fresh one's. rc 1 on any differing byte.
/// AD_RESIDENT_SELFTEST_ERROR=1 multiplies the first word of the second request's first re-planted allocation by 1.001
/// (the test must then fail).
pub fn selftest_cmd(args: Vec<String>) -> Result<(), String> {
    use crate::plan::{Operation, Plan};
    if args.len() < 2 { return Err("usage: ad-binder resident-selftest PKG PLANTED|- [PLANTED|- ...]".into()); }
    let base = PathBuf::from(&args[0]);
    let read_plan = || -> Result<Plan, String> {
        let v: Value = serde_json::from_slice(&fs::read(base.join("plan.json")).map_err(|e| format!("plan.json: {e}"))?).map_err(|e| e.to_string())?;
        serde_json::from_value(v).map_err(|e| format!("plan.json: {e}"))
    };
    let initial = |p: &Plan| -> Result<BTreeMap<u64, Vec<u8>>, String> {
        p.allocations.iter().filter_map(|x| x.initial.as_ref().map(|b| (x.id, b))).map(|(id, b)| Ok((id, crate::blob(&base, b)?))).collect()
    };
    let uploads = |p: &Plan| -> BTreeMap<(String, usize), String> {
        p.blocks.iter().flat_map(|(n, ops)| ops.iter().enumerate().filter_map(move |(i, o)| match o {
            Operation::Upload { data, .. } => Some(((n.clone(), i), crate::blob_sha(data).to_owned())), _ => None })).collect()
    };
    let err_on = std::env::var("AD_RESIDENT_SELFTEST_ERROR").as_deref() == Ok("1");
    let mut rs = Resident { dir: PathBuf::new(), idle_s: 0.0, name: String::new(), index: 0, served: Vec::new(), orig_alloc: BTreeMap::new(), orig_ops: BTreeMap::new(),
                            cur_alloc: BTreeSet::new(), cur_ops: BTreeSet::new(), keep_alloc: None, t_req: std::time::Instant::now() };
    let mut plan = read_plan()?;
    let mut kept: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
    let mut all_pass = true;
    for (r, p) in args[1..].iter().enumerate() {
        let pl = (p != "-").then_some(p.as_str());
        rs.swap(&mut plan, &base, pl)?;
        let held = if r == 0 { kept = initial(&plan)?; rs.kept(); kept.clone() } else {
            let mut h = kept.clone();
            for (k, id) in rs.replant_ids().into_iter().enumerate() {
                let x = plan.allocations.iter().find(|x| x.id == id).ok_or("allocation")?;
                let mut b = crate::blob(&base, x.initial.as_ref().ok_or("initial")?)?;
                if err_on && r == 1 && k == 0 { let w = f32::from_le_bytes(b[..4].try_into().unwrap()) * 1.001; b[..4].copy_from_slice(&w.to_le_bytes()); }
                h.insert(id, b);
            }
            h
        };
        let mut fresh = read_plan()?;
        if let Some(d) = pl { install(&mut fresh, &base, Path::new(d))?; }
        let want = initial(&fresh)?;
        let bad_a: Vec<u64> = want.iter().filter(|(id, b)| held.get(id) != Some(*b)).map(|(id, _)| *id).collect();
        let (ua, ub) = (uploads(&plan), uploads(&fresh));
        let bad_u: Vec<String> = ub.iter().filter(|(k, s)| ua.get(*k) != Some(*s)).map(|(k, _)| format!("{}[{}]", k.0, k.1)).collect();
        // the swapped upload payloads must also read back (their sha256 is checked by crate::blob)
        for (k, _) in ua.iter().filter(|(k, _)| rs.cur_ops.contains(*k)) {
            if let Some(Operation::Upload { data, .. }) = plan.blocks.get(&k.0).and_then(|o| o.get(k.1)) { crate::blob(&base, data)?; }
        }
        let pass = bad_a.is_empty() && bad_u.is_empty() && ua.len() == ub.len();
        all_pass &= pass;
        println!("{}", json!({"request":r + 1,"planted":p,"replanted":if r == 0 { json!(null) } else { json!(rs.replant_ids()) },
            "allocations_compared":want.len(),"bytes_compared":want.values().map(Vec::len).sum::<usize>(),"allocations_differ":bad_a,
            "uploads_compared":ub.len(),"uploads_differ":bad_u,"pass":pass}));
    }
    println!("{}", json!({"selftest":"resident","requests":args.len() - 1,"planted_error":err_on,"pass":all_pass}));
    if all_pass { Ok(()) } else { Err("resident selftest: a request's start differs from a fresh program's".into()) }
}

/// The command's words with a request's put over them; a word outside REQUEST_WORDS with another value is refused. A
/// request word of REQUEST_WORDS given empty ("") is taken out (a command without it), so no word of the first request
/// (its planted start above all) carries over to a request that does not name it.
pub fn merge_words(base: &HashMap<String, String>, req: &Request) -> Result<HashMap<String, String>, String> {
    let mut a = base.clone();
    for (k, v) in &req.words {
        if !REQUEST_WORDS.contains(&k.as_str()) && base.get(k) != Some(v) {
            return Err(format!("request {}: --{k} holds the program's loaded state and cannot change in a resident program", req.name));
        }
        if v.is_empty() { a.remove(k); } else { a.insert(k.clone(), v.clone()); }
    }
    if !req.words.contains_key("out") { return Err(format!("request {}: no out", req.name)); }
    for k in ["objective", "x0-dir"] { if !a.contains_key(k) { return Err(format!("request {}: a resident program serves the gradient line, which needs --{k}", req.name)); } }
    if a.contains_key("x0-planted") != req.words.get("x0-planted").is_some_and(|v| !v.is_empty()) {
        return Err(format!("request {}: name x0-planted (a folder, or \"\" for the recording's start): the first request's would carry over", req.name));
    }
    Ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layer_groups_first_braceless_match() {
        let t = br#"{"x":{"layer_groups": {"a":{"b":1}}}, "layer_groups":  {"batches":[[65,38278]],"n":1}}"#;
        assert_eq!(layer_groups_text(t), Some(&br#"{"batches":[[65,38278]],"n":1}"#[..]));
        assert_eq!(layer_groups_text(b"{}"), None);
    }
    #[test]
    fn request_words_refuse_loaded_state() {
        let m = |v: &[(&str, &str)]| -> HashMap<String, String> { v.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect() };
        let base = m(&[("steps", "59"), ("out", "/a"), ("kernels", "/k"), ("objective", "/o"), ("x0-dir", "/p1/x"), ("x0-planted", "/p1")]);
        let ok = Request { name: "r1".into(), words: m(&[("out", "/b"), ("steps", "59"), ("x0-planted", "/p2")]) };
        let got = merge_words(&base, &ok).unwrap();
        assert_eq!((got["out"].as_str(), got["x0-planted"].as_str(), got["kernels"].as_str()), ("/b", "/p2", "/k"));
        let bad = Request { name: "r2".into(), words: m(&[("out", "/b"), ("steps", "12"), ("x0-planted", "/p2")]) };
        assert!(merge_words(&base, &bad).is_err());
        let no_out = Request { name: "r3".into(), words: HashMap::new() };
        assert!(merge_words(&base, &no_out).is_err());
        // the first request's planted start never carries over: a request names its own, or "" for none
        let silent = Request { name: "r4".into(), words: m(&[("out", "/c")]) };
        assert!(merge_words(&base, &silent).is_err());
        let none = Request { name: "r5".into(), words: m(&[("out", "/c"), ("x0-planted", "")]) };
        assert!(!merge_words(&base, &none).unwrap().contains_key("x0-planted"));
        // the gradient line's words cannot be taken out, nor can loaded-state words
        assert!(merge_words(&base, &Request { name: "r6".into(), words: m(&[("out", "/c"), ("x0-planted", ""), ("objective", "")]) }).is_err());
        assert!(merge_words(&base, &Request { name: "r7".into(), words: m(&[("out", "/c"), ("x0-planted", ""), ("kernels", "")]) }).is_err());
    }
    #[test]
    fn resident_words_off_and_refusals() {
        let w = |v: &[(&str, &str)]| -> HashMap<String, String> { v.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect() };
        // off: no --resident word, nothing made
        assert!(Resident::from_words(&w(&[("steps", "59")])).unwrap().is_none());
        assert!(Resident::from_words(&w(&[("resident-name", "x")])).is_err());
        let d = std::env::temp_dir().join(format!("resident-test-{}", std::process::id()));
        let ds = d.display().to_string();
        let line = [("resident", ds.as_str()), ("tangent-only", "1"), ("no-tangent", "1"), ("objective", "/o"), ("x0-dir", "/x"), ("checkpoints", "sqrt")];
        assert!(Resident::from_words(&w(&line)).unwrap().is_some());
        let mut t = line.to_vec(); t.push(("taylor", "1e-3"));
        assert!(Resident::from_words(&w(&t)).is_err());
        let t2: Vec<_> = line.iter().copied().filter(|x| x.0 != "no-tangent").collect();
        assert!(Resident::from_words(&w(&t2)).is_err());
        let _ = fs::remove_dir_all(&d);
    }
    #[test]
    fn requests_in_name_order_and_idle_end() {
        let d = std::env::temp_dir().join(format!("resident-req-{}", std::process::id()));
        fs::create_dir_all(&d).unwrap();
        fs::write(d.join("b.req"), br#"{"out":"/b"}"#).unwrap();
        fs::write(d.join("a.req"), br#"{"out":"/a","x0-planted":"/p"}"#).unwrap();
        let r = next_request(&d, 1.0).unwrap().unwrap();
        assert_eq!((r.name.as_str(), r.words["out"].as_str()), ("a", "/a"));
        assert!(d.join("a.taken").exists() && !d.join("a.req").exists());
        assert_eq!(next_request(&d, 1.0).unwrap().unwrap().name, "b");
        let t = std::time::Instant::now();
        assert!(next_request(&d, 0.2).unwrap().is_none());
        assert!(t.elapsed().as_secs_f64() >= 0.2);
        fs::write(d.join("c.req"), br#"{"out":"/c"}"#).unwrap();
        fs::write(d.join("stop"), b"").unwrap();
        assert!(next_request(&d, 5.0).unwrap().is_none());
        let _ = fs::remove_dir_all(&d);
    }
    #[test]
    fn replant_is_the_union_of_kept_and_current() {
        let mut rs = Resident { dir: PathBuf::new(), idle_s: 0.0, name: String::new(), index: 0, served: Vec::new(), orig_alloc: BTreeMap::new(), orig_ops: BTreeMap::new(),
                                cur_alloc: [3u64, 5].into_iter().collect(), cur_ops: BTreeSet::new(), keep_alloc: None, t_req: std::time::Instant::now() };
        rs.kept();
        rs.cur_alloc = [5u64, 9].into_iter().collect();
        assert_eq!(rs.replant_ids(), vec![3, 5, 9]);
        rs.cur_alloc.clear();
        assert_eq!(rs.replant_ids(), vec![3, 5]);
    }
    /// dist47: a refused request is answered rc 1 with the refusal and `program: loaded`, and the session answers the
    /// next request (the program's side of the mailbox, no card).
    #[test]
    fn refused_request_is_answered_and_the_next_one_is_served() {
        let w = |v: &[(&str, &str)]| -> HashMap<String, String> { v.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect() };
        let d = std::env::temp_dir().join(format!("resident-refused-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        let ds = d.display().to_string();
        let words = w(&[("resident", ds.as_str()), ("resident-name", "q1"), ("resident-idle-s", "5"), ("tangent-only", "1"), ("no-tangent", "1"),
                        ("objective", "/o"), ("x0-dir", "/x1"), ("checkpoints", "sqrt"), ("out", "/out1")]);
        let mut z = Resident::from_words(&words).unwrap().unwrap();
        // the first request is refused: the forecast stops at its start
        let e = crate::checked_read_refusal(2, "", 132, 13312, "checked_value", false, Some(crate::DEFERRED_CHECK_PROOF), &[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]);
        assert!(crate::refusal_leaves_program_loaded(&e));
        z.refused(Path::new("/out1"), &e).unwrap();
        let a1 = rd_json(&d.join("q1.done")).unwrap();
        assert_eq!((a1["rc"].as_i64(), a1["program"].as_str(), a1["name"].as_str()), (Some(1), Some("loaded"), Some("q1")));
        let text = a1["error"].as_str().unwrap();
        assert!(text.starts_with("step 2: the forecast itself stops at this start: 1 of the 2 check words") && text.ends_with(STILL_LOADED), "{text}");
        // the next request is taken and answered by the same program
        fs::write(d.join("q2.req"), br#"{"out":"/out2","x0-dir":"/x2","x0-planted":""}"#).unwrap();
        let a = z.next(&words).unwrap().expect("the session took no request after a refusal");
        assert_eq!((a["out"].as_str(), a["x0-dir"].as_str(), z.name.as_str(), z.index), ("/out2", "/x2", "q2", 2));
        assert_eq!(rd_json(&d.join("state.json")).unwrap()["state"], "busy");
        z.done(Path::new("/out2"), json!({"window_wall_s":1.0})).unwrap();
        let a2 = rd_json(&d.join("q2.done")).unwrap();
        assert_eq!((a2["rc"].as_i64(), a2["index"].as_u64()), (Some(0), Some(2)));
        assert_eq!(z.served.len(), 2);
        // a refusal the program does not survive is answered as an ended program
        let a3 = ended_answer("checked device-to-host value differs from the proven value at step 3");
        assert_eq!((a3["rc"].as_i64(), a3["program"].as_str()), (Some(1), Some("ended")));
        let _ = fs::remove_dir_all(&d);
    }
    /// The private pattern file that HEX_PRIVATE_PATTERNS names, when this machine has one (it is never packaged).
    fn private_patterns() -> Option<Value> {
        let p = std::env::var_os("HEX_PRIVATE_PATTERNS")?;
        Some(rd_json(Path::new(&p)).unwrap_or_else(|e| panic!("HEX_PRIVATE_PATTERNS: {e}")))
    }
    fn private_list(v: &Value, key: &str) -> Vec<String> {
        v[key].as_array().map(|a| a.iter().filter_map(|w| w.as_str().map(str::to_lowercase)).collect()).unwrap_or_default()
    }
    /// dist48: no source line of the crate carries a word of the private pattern file's crate lists (`crate_words`,
    /// anywhere in a line; `crate_whole_words`, as a whole word). THE BREAKAGE THIS PREVENTS: the crate named tooling
    /// that is not in it (dist45, dist46), and this test then spelled those words in halves (dist47), so the source
    /// package still carried them. The words are read from the file; without it the test has nothing to read and
    /// passes, and the release seal reads the built packages for the same words.
    #[test]
    fn source_names_no_tooling_outside_the_crate() {
        let Some(private) = private_patterns() else { return };
        let (pats, whole) = (private_list(&private, "crate_words"), private_list(&private, "crate_whole_words"));
        assert!(!pats.is_empty(), "HEX_PRIVATE_PATTERNS: the file has no crate_words");
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut files = Vec::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(p) = stack.pop() {
            for e in fs::read_dir(&p).unwrap().filter_map(|e| e.ok()) {
                let q = e.path();
                let name = q.file_name().unwrap().to_string_lossy().into_owned();
                if q.is_dir() { if name != "target" && !name.starts_with('.') { stack.push(q); } continue; }
                if ["rs", "cu", "h", "cc", "c", "toml", "md", "json", "ptx", "txt"].iter().any(|x| q.extension().is_some_and(|e| e == *x)) || name == "NOTICE" { files.push(q); }
            }
        }
        assert!(files.len() > 100, "the crate's source was not found under {}", root.display());
        let mut hits = Vec::new();
        for f in &files {
            let Ok(text) = fs::read_to_string(f) else { continue };
            for (i, line) in text.lines().enumerate() {
                let low = line.to_lowercase();
                if let Some(p) = pats.iter().find(|p| low.contains(p.as_str())) { hits.push(format!("{}:{}: {p}", f.strip_prefix(root).unwrap().display(), i + 1)); }
                if let Some(t) = low.split(|c: char| !c.is_alphanumeric()).find(|t| whole.iter().any(|w| w == t)) { hits.push(format!("{}:{}: {t}", f.strip_prefix(root).unwrap().display(), i + 1)); }
            }
        }
        assert!(hits.is_empty(), "source lines that name tooling outside this crate: {hits:#?}");
    }
    /// dist47: a retired driver word is not a word of this crate: a resident program treats it as it treats any word
    /// it does not know (not refused by name at the start; refused in a request as a word outside the request words),
    /// while the neutral word of the absent extension driver is refused by name. dist48: the retired words are read
    /// from the private pattern file (`crate_retired_words`); without it the test has none to read.
    #[test]
    fn retired_driver_word_is_an_unknown_word() {
        let Some(private) = private_patterns() else { return };
        for old in private_list(&private, "crate_retired_words") { retired_word_is_unknown(&old); }
    }
    fn retired_word_is_unknown(old: &str) {
        let w = |v: &[(&str, &str)]| -> HashMap<String, String> { v.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect() };
        let old = old.to_string();
        assert!(REFUSED_WORDS.iter().all(|(k, what)| *k != old && !what.to_lowercase().contains("var")));
        assert!(!REQUEST_WORDS.contains(&old.as_str()));
        let d = std::env::temp_dir().join(format!("resident-oldword-{}", std::process::id()));
        let ds = d.display().to_string();
        let line = [("resident", ds.as_str()), ("tangent-only", "1"), ("no-tangent", "1"), ("objective", "/o"), ("x0-dir", "/x"), ("checkpoints", "sqrt")];
        let with = |k: &str| { let mut t = line.to_vec(); t.push((k, "run")); Resident::from_words(&w(&t)) };
        // at the start: exactly as a word nobody knows (the window reads no such word, so it changes nothing)
        assert_eq!(with(&old).map(|z| z.is_some()), with("no-such-word").map(|z| z.is_some()));
        // the neutral word is refused by name
        let e = with("ext-driver").err().expect("the extension driver word must be refused");
        assert!(e.contains("--ext-driver") && e.contains("an extension driver"), "{e}");
        // in a request: refused as any word outside the request words is, in the same sentence
        let base = w(&[("out", "/a"), ("objective", "/o"), ("x0-dir", "/x")]);
        let ask = |k: &str| merge_words(&base, &Request { name: "r1".into(), words: w(&[("out", "/b"), ("x0-planted", ""), (k, "run")]) }).unwrap_err();
        assert_eq!(ask(&old).replace(&old, "W"), ask("no-such-word").replace("no-such-word", "W"));
        let _ = fs::remove_dir_all(&d);
    }
    /// dist47: SYNC.json names every synced file by the sha256 of the file as it is in the crate. THE BREAKAGE THIS
    /// PREVENTS: seven files changed on the release line after the sync kept the sync's sha256 in the record (dist46),
    /// so a reader who checked the record against the files found them different with nothing saying why.
    #[test]
    fn sync_record_names_each_file_as_it_is() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let doc = rd_json(&root.join("SYNC.json")).unwrap();
        let files = doc["files"].as_array().unwrap();
        assert!(files.len() > 300, "SYNC.json lists {} files", files.len());
        let mut differ = Vec::new();
        for f in files {
            let name = f["public"].as_str().unwrap();
            let now = sha_hex(&fs::read(root.join(name)).unwrap_or_else(|e| panic!("{name}: {e}")));
            if Some(now.as_str()) != f["public_sha256"].as_str() { differ.push(format!("{name}: the file is {now}")); }
        }
        assert!(differ.is_empty(), "SYNC.json: set public_sha256 to the file's and \"edited_after_sync\": true for {differ:#?}");
    }
}
