// SPDX-License-Identifier: Apache-2.0
//! Experimental native runner. It refuses incomplete observations and does not
//! establish that a producer's command list, frame rules, or census are complete.
mod cuda;
mod copy_plan;
mod cadence;
mod add_inplace;
mod boundary_hook;
mod carray;
mod census;
mod checkpoint;
mod frames;
mod host_array;
mod host_scalar;
mod host_state;
mod image;
mod layer_groups;
mod native_cub;
mod native_cublas;
mod native_thrust;
mod numerics;
mod ordering;
mod plan;
mod pointer_table;
mod poison;
mod pinned;
mod replay_frame;
mod rk_schedule;
mod radiation_unpack;
mod ruc;
mod seaice;
mod sfclay;
mod submit;
mod tempo;
mod transport_copy;
mod vinterp;
mod word_reference;
mod xorshift;
mod peer;
// Physics-on window stage A (ad/phys-window): U6's derivative entries and launch dot, reused as is.
#[allow(dead_code, unused)] #[path = "../../../../../../e2e/src/deriv.rs"] mod deriv;
#[allow(dead_code, unused)] #[path = "../../../../../../e2e/src/dev.rs"] mod dev;
#[allow(dead_code, unused)] #[path = "../../../../../../e2e/src/dot.rs"] mod dot;
#[allow(dead_code, unused)] #[path = "../../../../../../e2e/src/expr.rs"] mod expr;
#[allow(dead_code, unused)] #[path = "../../../../../../e2e/src/recipes.rs"] mod recipes;
mod addot;
// ad/held-rlw_cldprmc_coalesced: the per-launch dot of an ad.entry/v1 row on one real captured launch (ad-binder capture-dot).
mod capentry;
// Physics-on window lane L1: composed tangent + segment-checkpointed reverse (ad-binder window).
mod window;
mod wrules;
#[allow(dead_code, unused)] #[path = "../../../../../../e2e/src/held.rs"] mod held;
mod p8bridge;
mod schemes;
mod tempo_adw;
mod tempo_order;
// ad/held-rlw_taumol_batched: the per-launch dot of a scheme row on one real captured launch (ad-binder capdot).
mod capdot;
/// fwdcheck-digest: a whole-state capture's bytes: pageable (Win::capture_state) or a C-ABI boundary's own page-locked
/// RAM copy (shared with the Boundary that keeps it). Either reads as the same &[u8].
pub(crate) enum CapData { Vec(Vec<u8>), Host(std::sync::Arc<window::capi::HostBytes>) }
impl std::ops::Deref for CapData {
    type Target = [u8];
    fn deref(&self) -> &[u8] { match self { CapData::Vec(v) => v, CapData::Host(h) => h.as_slice() } }
}
mod exec {
    pub fn decode_hex(text: &str) -> Result<Vec<u8>, String> { super::decode_hex(text) }
    pub fn digest(bytes: &[u8]) -> String { super::digest(bytes) }
}

use cuda::{checked, Cuda};
use plan::{Argument, Blob, Operation, Plan};
use sha2::{Digest, Sha256};
use std::{collections::{HashMap, HashSet}, ffi::{c_void, CString}, fs::{self, File},
          io::{BufRead, BufReader, BufWriter, Read, Write}, path::{Path, PathBuf}, ptr};

const PIN: &str = "16251589f19aec0012fd164d1ea060126208aed1";
extern "C" {
    fn hex_step_validate(commands: *const submit::NativeCommand, count: usize, frame_bytes: usize) -> i32;
    fn hex_step_submit(driver: *const submit::NativeDriver, commands: *const submit::NativeCommand,
                       count: usize, frame: *const c_void, stream: *mut c_void,
                       completed: *mut usize) -> i32;
}

fn decode_hex(text: &str) -> Result<Vec<u8>, String> {
    if text.len() % 2 != 0 || !text.is_ascii() { return Err("invalid hex bytes".into()); }
    text.as_bytes().chunks_exact(2).map(|pair| {
        let nibble = |b| match b { b'0'..=b'9' => Ok(b - b'0'), b'a'..=b'f' => Ok(b - b'a' + 10),
                                  b'A'..=b'F' => Ok(b - b'A' + 10), _ => Err("invalid hex bytes") };
        Ok((nibble(pair[0])? << 4) | nibble(pair[1])?)
    }).collect()
}

/// CU_CTX_SCHED_BLOCKING_SYNC (P1, as ad/e2e U8).
const CTX_SCHED_BLOCKING_SYNC: u32 = 0x4;
/// CU_EVENT_BLOCKING_SYNC.
const EVENT_BLOCKING_SYNC: u32 = 0x1;
/// P1: quick (sleeping host waits) unless AD_BINDER_QUICK=0 selects the spin-wait referee.
fn quick_requested() -> bool { std::env::var("AD_BINDER_QUICK").as_deref() != Ok("0") }
/// P1: host SHA-256 across all cores unless AD_BINDER_HASH=device selects the device digest.
fn host_hash_requested() -> bool { std::env::var("AD_BINDER_HASH").as_deref() != Ok("device") }
/// Recorded event flags plus CU_EVENT_BLOCKING_SYNC in quick mode: only how the host waits changes.
fn event_flags(recorded: u32) -> u32 { if quick_requested() { recorded | EVENT_BLOCKING_SYNC } else { recorded } }
/// Lane runner-lib: libhexad-phys sets this so a library client's stdout does not carry one state_sha256 line per
/// carried allocation and step (the hashes are still computed and checked; the ad-binder binary never sets it).
static QUIET_HASHES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// outer2 (2026-11-07): replay placement congruence modulus (see Execution::allocate_congruent); 0 = off.
fn replay_congruence() -> u64 {
    if std::env::var("AD_BINDER_CONGRUENCE").as_deref() == Ok("off") { 0 } else { 2 << 20 }
}
/// host-leftovers-2: pooled staging buffers allocated (cuMemHostAlloc) so far in this process (step receipts).
pub(crate) static STAGING_ALLOCS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) fn staging_allocs() -> u64 { STAGING_ALLOCS.load(std::sync::atomic::Ordering::Relaxed) }
fn quiet_hashes() -> bool { QUIET_HASHES.load(std::sync::atomic::Ordering::Relaxed) }
pub(crate) fn set_quiet_hashes(q: bool) { QUIET_HASHES.store(q, std::sync::atomic::Ordering::Relaxed); }
fn digest(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
fn hex_bytes(bytes: &[u8]) -> String { bytes.iter().map(|b| format!("{b:02x}")).collect() }
type References = HashMap<(usize, u64), (usize, String)>;

fn read_references(path: &Path, plan: &Plan, steps: usize) -> Result<References, String> {
    let file = File::open(path).map_err(|e| format!("cannot read reference: {e}"))?;
    let mut result = References::new();
    let mut seen = HashSet::new();
    let expected_ids: HashSet<_> = plan.carried.iter().chain(&plan.host_carried).copied().collect();
    let max_steps = plan.physics.as_ref().map_or(plan.cadence.len(), |physics| physics.max_steps);
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        let row: serde_json::Value = serde_json::from_str(&line)
            .map_err(|e| format!("invalid reference line {}: {e}", index + 1))?;
        if row.get("event").and_then(|v| v.as_str()) != Some("state_sha256") {
            return Err(format!("invalid reference event at line {}", index + 1));
        }
        let step = usize::try_from(row.get("step").and_then(|v| v.as_u64()).ok_or("invalid reference step")?)
            .map_err(|_| "reference step overflow")?;
        let id = row.get("allocation").and_then(|v| v.as_u64()).ok_or("invalid reference allocation")?;
        let bytes = usize::try_from(row.get("bytes").and_then(|v| v.as_u64()).ok_or("invalid reference size")?)
            .map_err(|_| "reference size overflow")?;
        let hash = row.get("sha256").and_then(|v| v.as_str()).ok_or("invalid reference SHA-256")?;
        if decode_hex(hash)?.len() != 32 { return Err("invalid reference SHA-256 length".into()); }
        if step == 0 || step > max_steps || bytes == 0 || !expected_ids.contains(&id) {
            return Err(format!("unknown reference step or allocation {step}:{id}"));
        }
        if !seen.insert((step, id)) {
            return Err(format!("duplicate reference for step {step} allocation {id}"));
        }
        if step <= steps { result.insert((step, id), (bytes, hash.to_ascii_lowercase())); }
    }
    let expected_count = steps.checked_mul(expected_ids.len()).ok_or("reference count overflow")?;
    if result.len() != expected_count { return Err("reference does not cover exactly every carried allocation and step".into()); }
    for step in 1..=steps {
        for id in &expected_ids {
            if !result.contains_key(&(step, *id)) {
                return Err(format!("missing reference for step {step} allocation {id}"));
            }
        }
    }
    Ok(result)
}

fn validate_view_schedule(plan: &Plan, steps: usize) -> Result<(), String> {
    let Some(frames) = &plan.carried_views else { return Ok(()); };
    let required = plan.physics.as_ref().map_or(steps, |physics| physics.max_steps);
    if frames.len() < required { return Err("carried view schedule is incomplete".into()); }
    let ids: HashSet<_> = plan.carried.iter().copied().collect();
    let mut sizes = HashMap::new();
    for (index, frame) in frames.iter().enumerate() {
        let mut observed = HashSet::new();
        let mut ranges = Vec::with_capacity(frame.len());
        for view in frame {
            let end = view.target.offset.checked_add(view.bytes).ok_or("carried view extent overflow")?;
            if view.bytes == 0 || !ids.contains(&view.id) || !observed.insert(view.id) {
                return Err(format!("invalid carried view at step {}", index + 1));
            }
            if sizes.insert(view.id, view.bytes).is_some_and(|old| old != view.bytes) {
                return Err(format!("carried view size changes for {}", view.id));
            }
            ranges.push((view.target.allocation, view.target.offset, end));
        }
        if observed != ids { return Err(format!("carried views missing at step {}", index + 1)); }
        ranges.sort_unstable();
        if ranges.windows(2).any(|pair| pair[0].0 == pair[1].0 && pair[0].2 > pair[1].1) {
            return Err(format!("overlapping carried views at step {}", index + 1));
        }
    }
    Ok(())
}

fn carried_views(plan: &Plan, step: usize,
                 live: &HashMap<u64, submit::Allocation>) -> Result<Vec<plan::CarriedView>, String> {
    if let Some(frames) = &plan.carried_views {
        return frames.get(step.checked_sub(1).ok_or("no initial carried view")?)
            .cloned().ok_or_else(|| format!("no carried view at step {step}"));
    }
    plan.carried.iter().map(|id| {
        let allocation = live.get(id).ok_or_else(|| format!("unknown carried allocation {id}"))?;
        Ok(plan::CarriedView { id: *id, target: plan::Slice { allocation: *id, offset: 0 },
                               bytes: allocation.bytes })
    }).collect()
}

fn resolve_views(views: &[plan::CarriedView],
                 live: &HashMap<u64, submit::Allocation>) -> Result<HashMap<u64, submit::Allocation>, String> {
    let mut result = HashMap::new();
    for view in views {
        let root = live.get(&view.target.allocation).ok_or_else(|| format!("missing physical allocation {}", view.target.allocation))?;
        let end = view.target.offset.checked_add(view.bytes).ok_or("carried view extent overflow")?;
        if view.bytes == 0 || end > root.bytes { return Err(format!("carried view {} exceeds physical allocation", view.id)); }
        let offset = u64::try_from(view.target.offset).map_err(|_| "carried view offset overflow")?;
        let address = root.address.checked_add(offset).ok_or("carried view address overflow")?;
        if result.insert(view.id, submit::Allocation { id: view.id, address, bytes: view.bytes }).is_some() {
            return Err(format!("duplicate carried view {}", view.id));
        }
    }
    Ok(result)
}

fn checkpoint_steps(value: &str, steps: usize) -> Result<HashSet<usize>, String> {
    let mut result = HashSet::new();
    for item in value.split(',') {
        let step = item.parse::<usize>().map_err(|_| "invalid checkpoint binding step")?;
        if step == 0 || step > steps || !result.insert(step) {
            return Err("checkpoint binding step is repeated or outside the replay".into());
        }
    }
    Ok(result)
}

/// The list remains an independently certified census requirement: these
/// checks can reject missing carried owners but cannot find omitted scratch.
fn validate_checkpoint_request(plan: &Plan, selected: &HashSet<usize>) -> Result<(), String> {
    if selected.is_empty() { return Ok(()); }
    let frame = plan.physics.as_ref().ok_or("checkpoint bindings require a physics frame")?;
    let host_ids: HashSet<_> = plan.host_allocations.iter().map(|owner| owner.id)
        .chain(std::iter::once(frame.frame_state_id)).collect();
    for &step in selected {
        let boundary = plan.boundaries.get(step - 1).ok_or("checkpoint requires live boundaries")?;
        let mut ids = HashSet::new();
        for view in &boundary.live {
            if view.id == 0 || view.bytes == 0 || view.target.offset.checked_add(view.bytes).is_none()
                || !ids.insert(view.id) {
                return Err(format!("invalid checkpoint device owner at step {step}"));
            }
        }
        if plan.carried.iter().any(|id| !ids.contains(id)) {
            return Err(format!("checkpoint misses carried device owner at step {step}"));
        }
        for id in &boundary.host_live {
            if *id == 0 || !host_ids.contains(id) || !ids.insert(*id) {
                return Err(format!("invalid checkpoint host owner at step {step}"));
            }
        }
        if plan.host_carried.iter().any(|id| !boundary.host_live.contains(id)) {
            return Err(format!("checkpoint misses carried host owner at step {step}"));
        }
    }
    Ok(())
}

fn checkpoint_bindings(boundary: &plan::CheckpointBoundary,
                       carried: &HashMap<u64, submit::Allocation>,
                       allocations: &HashMap<u64, submit::Allocation>,
                       hosts: &mut host_state::HostState) -> Result<Vec<checkpoint::Binding>, String> {
    let live = resolve_views(&boundary.live, allocations)?;
    for (id, expected) in carried {
        let found = live.get(id).ok_or("missing carried checkpoint binding")?;
        if found.address != expected.address || found.bytes != expected.bytes {
            return Err(format!("checkpoint carried extent differs for owner {id}"));
        }
    }
    let mut bindings = Vec::with_capacity(live.len() + boundary.host_live.len());
    for allocation in live.values() {
        bindings.push(checkpoint::Binding { id: allocation.id,
            bytes: u64::try_from(allocation.bytes).map_err(|_| "checkpoint size overflow")?,
            memory: checkpoint::DEVICE, address: allocation.address });
    }
    for (id, bytes, address) in hosts.binding_addresses(&boundary.host_live)? {
        bindings.push(checkpoint::Binding { id,
            bytes: u64::try_from(bytes).map_err(|_| "checkpoint host size overflow")?,
            memory: checkpoint::HOST, address });
    }
    checkpoint::validate(&mut bindings).map_err(str::to_owned)?;
    Ok(bindings)
}

fn read_host_i64_values(hosts: &host_state::HostState, values: &[plan::Slice]) -> Result<Vec<i64>, String> {
    values.iter().map(|slice| {
        let data = hosts.get(slice.allocation)?;
        let end = slice.offset.checked_add(8).ok_or("host metadata overflow")?;
        let raw = data.get(slice.offset..end).ok_or("host metadata outside allocation")?;
        Ok(i64::from_ne_bytes(raw.try_into().map_err(|_| "host metadata size")?))
    }).collect()
}

fn copy_plan_rows(hosts: &host_state::HostState, rows: &[copy_plan::RowSpec]) -> Result<Vec<copy_plan::Row>, String> {
    rows.iter().map(|row| {
        let dst_stride = read_host_i64_values(hosts, &row.dst_stride)?.try_into().map_err(|_| "copy-plan stride rank")?;
        let shape = read_host_i64_values(hosts, &row.shape)?.try_into().map_err(|_| "copy-plan shape rank")?;
        let (src_stride, fill, mode) = match (row.src, row.src_stride.as_ref(), row.fill) {
            (Some(_), Some(strides), None) =>
                (read_host_i64_values(hosts, strides)?.try_into().map_err(|_| "copy-plan stride rank")?, 0, 0),
            (None, None, Some(word)) => {
                let data = hosts.get(word.allocation)?;
                let end = word.offset.checked_add(4).ok_or("copy-plan fill overflow")?;
                let raw = data.get(word.offset..end).ok_or("copy-plan fill outside allocation")?;
                ([0; 4], u32::from_ne_bytes(raw.try_into().map_err(|_| "copy-plan fill width")?), 1)
            },
            _ => return Err("invalid copy-plan copy/fill inputs".into()),
        };
        Ok(copy_plan::Row { dst: row.dst, src: row.src, dst_stride, src_stride, shape, fill, mode })
    }).collect()
}
fn verify_sha(bytes: &[u8], expected: &str) -> Result<(), String> {
    if expected.len() != 64 || digest(bytes) != expected { Err("SHA-256 mismatch".into()) } else { Ok(()) }
}
fn path(base: &Path, name: &str) -> PathBuf {
    let candidate = Path::new(name);
    if candidate.is_absolute() { candidate.to_owned() } else { base.join(candidate) }
}

/// The library a plan pins (its recorded path and sha256), found on this machine.
///
/// THE BREAKAGE (hex 0.5.0 integration, fresh install): a plan records the absolute path of the CuPy CUB/Thrust
/// modules and cuBLAS of the machine that recorded it, so on any other machine the replay refused with "cannot
/// resolve cuBLAS library". Order: the recorded path when it holds the pinned bytes; else `HEXAD_LIBRARY_PATH`
/// (folders, `:`-separated; the gpuwm-hex doors set it to the installed CuPy and NVIDIA wheel folders) for a file of the
/// same name and the same sha256; else the same library from another build (a CuPy extension module of another
/// CPython ABI tag, a cuBLAS of another release), named in a `library_substituted` event. A substitute is admitted
/// because every replayed step's boundary hashes are compared with the recording: a substitute that changed one bit
/// stops the run at that boundary, so it can never produce a gradient of another trajectory.
fn resolve_pinned_library(recorded: &Path, sha256: &str) -> Result<(PathBuf, String), String> {
    let dirs: Vec<PathBuf> = std::env::var("HEXAD_LIBRARY_PATH").unwrap_or_default().split(':')
        .filter(|d| !d.is_empty()).map(PathBuf::from).collect();
    resolve_pinned_library_in(recorded, sha256, &dirs)
}
fn resolve_pinned_library_in(recorded: &Path, sha256: &str, dirs: &[PathBuf]) -> Result<(PathBuf, String), String> {
    fn digest(p: &Path) -> Option<String> { fs::read(p).ok().map(|b| format!("{:x}", Sha256::digest(&b))) }
    if recorded.is_file() && digest(recorded).as_deref() == Some(sha256) { return Ok((recorded.to_owned(), sha256.to_owned())); }
    let name = recorded.file_name().and_then(|n| n.to_str()).ok_or("pinned library path has no file name")?.to_owned();
    // Same library, any build: "cub.cpython-314-x86_64-linux-gnu.so" -> "cub.cpython-"; "libcublas.so.13" stays whole.
    let family = match name.find(".cpython-") { Some(at) => name[..at + ".cpython-".len()].to_owned(), None => name.clone() };
    let mut substitute = None;
    for dir in dirs {
        let exact = dir.join(&name);
        if exact.is_file() && digest(&exact).as_deref() == Some(sha256) { return Ok((exact, sha256.to_owned())); }
        if let Ok(entries) = fs::read_dir(dir) {
            let mut names: Vec<PathBuf> = entries.flatten().map(|e| e.path())
                .filter(|p| p.file_name().and_then(|n| n.to_str()).map_or(false, |n| n == name || (n.starts_with(&family) && n.ends_with(".so") && family != name)))
                .collect();
            names.sort();
            if substitute.is_none() { substitute = names.into_iter().next(); }
        }
    }
    if let Some(found) = substitute {
        let used = digest(&found).ok_or_else(|| format!("cannot read {}", found.display()))?;
        eprintln!("{}", serde_json::json!({"event":"library_substituted","recorded":recorded,"recorded_sha256":sha256,"used":found,
            "used_sha256":used,"why":"another build of the pinned library; every replayed boundary hash is compared with the recording"}));
        return Ok((found, used));
    }
    Err(format!("cannot resolve the pinned library {name} (recorded at {}, sha256 {}): not there, and no folder in                  HEXAD_LIBRARY_PATH holds it", recorded.display(), &sha256[..12.min(sha256.len())]))
}
fn gzip_bytes(path: &Path, expected: usize) -> Result<Vec<u8>, String> {
    if expected > (1_usize << 34) { return Err("compressed payload exceeds decoder limit".into()); }
    let limit = u64::try_from(expected.checked_add(1).ok_or("compressed size overflow")?)
        .map_err(|_| "compressed size overflow")?;
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut reader = flate2::read::GzDecoder::new(file).take(limit);
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes).map_err(|e| format!("invalid gzip: {e}"))?;
    if bytes.len() != expected { return Err("gzip uncompressed size mismatch".into()); }
    Ok(bytes)
}
fn blob_sha(value: &Blob) -> &str {
    match value { Blob::Hex { sha256, .. } | Blob::File { sha256, .. } | Blob::Gzip { sha256, .. } => sha256 }
}
fn blob(base: &Path, value: &Blob) -> Result<Vec<u8>, String> {
    let (bytes, sha) = match value {
        Blob::Hex { bytes, sha256 } => (decode_hex(bytes)?, sha256),
        Blob::File { path: name, sha256 } => (fs::read(path(base, name)).map_err(|e| e.to_string())?, sha256),
        Blob::Gzip { path: name, sha256, bytes } if *bytes >= GZIP_PIPE_BYTES => return gzip_bytes_verified(&path(base, name), *bytes, sha256),
        Blob::Gzip { path: name, sha256, bytes } => (gzip_bytes(&path(base, name), *bytes)?, sha256),
    };
    verify_sha(&bytes, sha)?;
    Ok(bytes)
}
/// integ8 (U8/L1, measured on the pkg18 window forward: the held split blobs are 1.06-1.25 GB each after inflate, three
/// per physics step, 3.3-4.8 s each on one core, and the step waited 3.6-4.8 s for them): a large gzip blob is inflated
/// straight into its final buffer (no growth copies) while a second thread hashes each finished 64 MiB chunk, so the
/// SHA-256 check runs beside the inflate instead of after it. The same bytes and the same checks: the size must be exact
/// and the hash must equal the plan's.
const GZIP_PIPE_BYTES: usize = 256 << 20;
fn gzip_bytes_verified(path: &Path, expected: usize, sha: &str) -> Result<Vec<u8>, String> {
    if expected > (1_usize << 34) { return Err("compressed payload exceeds decoder limit".into()); }
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut reader = flate2::read::GzDecoder::new(std::io::BufReader::with_capacity(4 << 20, file));
    let mut bytes = vec![0_u8; expected];
    let chunks: Vec<&mut [u8]> = bytes.chunks_mut(64_usize << 20).collect();
    let hashed = std::thread::scope(|scope| -> Result<String, String> {
        let (tx, rx) = std::sync::mpsc::channel::<&[u8]>();
        let hasher = scope.spawn(move || {
            let mut h = Sha256::new();
            for c in rx { h.update(c); }
            hex_bytes(&h.finalize())
        });
        let mut filled: Result<(), String> = Ok(());
        for c in chunks {
            if let Err(e) = reader.read_exact(c) {
                filled = Err(if e.kind() == std::io::ErrorKind::UnexpectedEof { "gzip uncompressed size mismatch".into() }
                             else { format!("invalid gzip: {e}") });
                break;
            }
            let done: &[u8] = c;
            if tx.send(done).is_err() { break; }
        }
        drop(tx);
        let h = hasher.join().map_err(|_| "blob hasher panicked")?;
        filled?;
        Ok(h)
    })?;
    let mut extra = [0_u8; 1];
    match reader.read(&mut extra) {
        Ok(0) => {},
        Ok(_) => return Err("gzip uncompressed size mismatch".into()),
        Err(e) => return Err(format!("invalid gzip: {e}")),
    }
    if sha.len() != 64 || hashed != sha { return Err("SHA-256 mismatch".into()); }
    Ok(bytes)
}

/// Relocate only positions certified as device addresses by the plan producer.
/// Validate the entire patch list before changing the captured bytes.
fn patch_device_addresses(data: &mut [u8], patches: &[plan::Patch],
                          allocations: &HashMap<u64, submit::Allocation>) -> Result<(), String> {
    let mut writes = Vec::with_capacity(patches.len());
    for patch in patches {
        let end = patch.byte_offset.checked_add(8).ok_or("pointer patch offset overflow")?;
        if end > data.len() { return Err("pointer patch outside upload".into()); }
        let allocation = allocations.get(&patch.target.allocation)
            .ok_or_else(|| format!("unknown pointer target {}", patch.target.allocation))?;
        if patch.target.offset >= allocation.bytes { return Err("pointer target outside allocation".into()); }
        let offset = u64::try_from(patch.target.offset).map_err(|_| "pointer target offset overflow")?;
        let address = allocation.address.checked_add(offset).ok_or("device address overflow")?;
        writes.push((patch.byte_offset, end, address));
    }
    writes.sort_unstable_by_key(|write| write.0);
    if writes.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        return Err("overlapping pointer patches".into());
    }
    for (start, end, address) in writes { data[start..end].copy_from_slice(&address.to_le_bytes()); }
    Ok(())
}

fn live_pointer_table(entries: &[plan::Slice],
                      allocations: &HashMap<u64, submit::Allocation>) -> Result<Vec<u8>, String> {
    pointer_table::build(entries, allocations)
}

/// Validate every arm, even if this execution will not visit it. Recursive
/// branches would otherwise make a sealed replay unbounded.
fn validate_branch_graph(plan: &Plan) -> Result<(), String> {
    fn visit<'a>(name: &'a str, plan: &'a Plan, marks: &mut HashMap<&'a str, u8>) -> Result<(), String> {
        match marks.get(name) {
            Some(1) => return Err(format!("cyclic host branch at {name}")),
            Some(2) => return Ok(()),
            _ => {},
        }
        let operations = plan.blocks.get(name).ok_or_else(|| format!("unknown host branch block {name}"))?;
        if operations.is_empty() { return Err(format!("empty host branch block {name}")); }
        marks.insert(name, 1);
        for operation in operations {
            if let Operation::HostBranch { if_zero, if_nonzero, .. } = operation {
                visit(if_zero, plan, marks)?;
                visit(if_nonzero, plan, marks)?;
            }
            match operation {
                Operation::RkSchedule { body, .. } | Operation::RkSplits { body }
                | Operation::RkStages { body } | Operation::RkSubsteps { body } => visit(body, plan, marks)?,
                _ => {},
            }
        }
        marks.insert(name, 2);
        Ok(())
    }
    let mut marks = HashMap::new();
    for name in plan.blocks.keys() { visit(name, plan, &mut marks)?; }
    Ok(())
}

fn validate_library_host_dimensions(host_sizes: &HashMap<u64, usize>, values: &[plan::Slice]) -> Result<(), String> {
    for value in values {
        let bytes = host_sizes.get(&value.allocation).ok_or("unknown live library dimension")?;
        if value.offset.checked_add(8).is_none_or(|end| end > *bytes) {
            return Err("live library dimension outside host allocation".into());
        }
    }
    Ok(())
}

/// P1 (ad/phys): a cupy_concatenate_same_size argument classed cupy_concat_count_from_table must carry, as its
/// CArray (ptr, size, shape, stride) or CIndexer (size, shape, index) words, the entry count of the CuPy
/// concatenate pointer table this block uploaded last (rule cupy_stack_same_size), and nothing else may vary.
fn concat_count_matches(command: &Operation, argument: &Argument, table: Option<usize>) -> bool {
    let Operation::Kernel { function, .. } = command else { return false };
    let Some(count) = table else { return false };
    if function != "cupy_concatenate_same_size" { return false; }
    let hex = match argument {
        Argument::Bytes { bytes_hex, .. } | Argument::Patched { bytes_hex, .. } => bytes_hex,
        _ => return false,
    };
    let Ok(raw) = decode_hex(hex) else { return false };
    let word = |k: usize| u64::from_le_bytes(raw[8 * k..8 * k + 8].try_into().unwrap());
    let count = count as u64;
    match raw.len() {
        32 => word(1) == count && word(2) == count,
        24 => word(0) == count && word(1) == count,
        _ => false,
    }
}

fn preflight(plan: &Plan, steps: usize) -> Result<(), String> {
    if plan.schema != "ad.binder.plan/v1" || !plan.complete {
        return Err("refusing incomplete or unknown plan".into());
    }
    if plan.source_pin != PIN { return Err("plan is not bound to the required hex pin".into()); }
    if !plan.end_step_barrier_proven { return Err("step synchronization has not been certified".into()); }
    if steps == 0 || plan.carried.is_empty() {
        return Err("requested steps or carried-state list are invalid".into());
    }
    validate_branch_graph(plan)?;
    let mut host_sizes = HashMap::new();
    for host in &plan.host_allocations {
        if host.bytes == 0 || host.origin != "initial_state" || host_sizes.insert(host.id, host.bytes).is_some() {
            return Err(format!("invalid initial host allocation {}", host.id));
        }
    }
    if plan.allocations.iter().any(|allocation| host_sizes.contains_key(&allocation.id))
        || plan.blocks.values().flatten().any(|command| matches!(command,
            Operation::Allocate { id, .. } if host_sizes.contains_key(id))) {
        return Err("host and device allocation identities overlap".into());
    }
    let mut host_carried = HashSet::new();
    let frame_id = plan.physics.as_ref().map(|physics| physics.frame_state_id);
    if frame_id.is_some_and(|id| host_sizes.contains_key(&id) || plan.carried.contains(&id)
        || plan.allocations.iter().any(|allocation| allocation.id == id)) {
        return Err("physics frame allocation identity overlaps another allocation".into());
    }
    for id in &plan.host_carried {
        if (!host_sizes.contains_key(id) && Some(*id) != frame_id)
            || plan.carried.contains(id) || !host_carried.insert(*id) {
            return Err(format!("invalid carried host allocation {id}"));
        }
    }
    if let Some(physics) = &plan.physics {
        if !host_carried.contains(&physics.frame_state_id) || plan.host_carried.len() < 2 {
            return Err("physics host carried state or frame is missing".into());
        }
        if steps > physics.max_steps || !plan.streams.is_empty() {
            return Err("physics step count or single-stream plan is invalid".into());
        }
        // Validate the entire sealed block set, not merely blocks reached by
        // this sample, so an unsampled branch cannot conceal an unsafe call.
        for (name, commands) in &plan.blocks {
            if commands.is_empty() { return Err(format!("empty block {name}")); }
            // P1 (ad/phys): entry count of the last CuPy concatenate pointer table in this block.
            let mut last_concat_table: Option<usize> = None;
            for command in commands {
                if let Operation::PointerTableUpload { rule, entries, .. } = command {
                    if rule == "cupy_stack_same_size" { last_concat_table = Some(entries.len()); }
                }
                let stream = match command {
                    Operation::Kernel { stream, .. } | Operation::VinterpOzone { stream, .. }
                    | Operation::RadiationUnpack { stream, .. } | Operation::TransportCopy { stream, .. }
                    | Operation::AddInplace { stream, .. }
                    | Operation::DeviceCopy { stream, .. } | Operation::ModuleGlobalCopy { stream, .. }
                    | Operation::PeerSend { stream, .. } | Operation::PeerRecv { stream, .. }
                    | Operation::Clear { stream, .. } | Operation::Upload { stream, .. }
                    | Operation::HeldUploadRanges { stream, .. }
                    | Operation::Fill16 { stream, .. } | Operation::Fill32 { stream, .. }
                    | Operation::Allocate { stream, .. } | Operation::Free { stream, .. }
                    | Operation::FreeNull { stream, .. }
                    | Operation::FrameUpload { stream, .. } | Operation::PointerTableUpload { stream, .. }
                    | Operation::TempoPointerTableBind { stream, .. } | Operation::TempoNamedTableBind { stream, .. }
                    | Operation::HostUpload { stream, .. } | Operation::HostPublish { stream, .. }
                    | Operation::SeaicePointerTableUpload { stream, .. }
                    | Operation::RucPointerTableUpload { stream, .. }
                    | Operation::RrfsCopyPlanUpload { stream, .. }
                    | Operation::RrfsCopyPlanKernel { stream, .. }
                    | Operation::NativeCubReduce { stream, .. }
                    | Operation::NativeCubSegmentedReduce { stream, .. }
                    | Operation::NativeThrustArgsort { stream, .. }
                    | Operation::NativeCublasTranspose { stream, .. }
                    | Operation::DeviceToHost { stream, .. }
                    | Operation::Synchronization { stream, .. } | Operation::EventRecord { stream, .. }
                    | Operation::EventWait { stream, .. } => Some(*stream),
                    Operation::HostBranch { condition, .. } => {
                        let bytes = host_sizes.get(&condition.allocation).ok_or("unknown host branch condition")?;
                        if condition.offset >= *bytes { return Err("host branch condition outside allocation".into()); }
                        None
                    },
                    Operation::RkSchedule { dt_f64, order_i64, acoustic_substeps_i64, dynamics_splits_i64, .. } => {
                        for value in [dt_f64, order_i64, acoustic_substeps_i64, dynamics_splits_i64] {
                            let bytes = host_sizes.get(&value.allocation).ok_or("unknown live RK configuration")?;
                            if value.offset.checked_add(8).is_none_or(|end| end > *bytes) {
                                return Err("live RK configuration outside allocation".into());
                            }
                        }
                        None
                    },
                    Operation::RkSplits { .. } | Operation::RkStages { .. } | Operation::RkSubsteps { .. } => None,
                    Operation::TempoTableRelease { binding } => {
                        if binding.is_empty() { return Err("empty TEMPO binding".into()); }
                        None
                    },
                    Operation::TempoMainComplete { first_call_main } => {
                        let bytes = host_sizes.get(&first_call_main.allocation).ok_or("unknown TEMPO first-call owner")?;
                        if first_call_main.offset >= *bytes { return Err("TEMPO first-call flag outside allocation".into()); }
                        None
                    },
                    Operation::HostBoundary { name } => {
                        if name != "physics_timetracker_state" && name != "update_state_time" {
                            return Err(format!("host boundary {name} requires a native ABI implementation"));
                        }
                        None
                    },
                };
                if stream.is_some_and(|s| s != 0) { return Err("physics command uses a nonzero stream".into()); }
                if let Operation::Upload { origin, .. } = command {
                    // COMMON-AD 103.3 (coordinator R3, 16:21Z): milestone 1 holds radiation and land at the slot
                    // boundary by per-step uploads recorded from the ordinary forward, labelled as such.
                    // multirank (L20): the same-host rank tables (halo pack/unpack descriptors, (shape, stride) pairs,
                    // solve-region rows) are sealed as uploads whose address words are relocation patches
                    // (seal_phys_stage2.py MIXED_TABLE_RULES); every other word is an integer of the decomposition.
                    let samehost = origin.as_deref().is_some_and(|o| o.starts_with("samehost table: "));
                    if !samehost && !matches!(origin.as_deref(), Some("immutable_input")
                                 | Some("held: radiation (milestone 1)") | Some("held: land (milestone 1)")) {
                        return Err("physics upload is not a certified immutable input".into());
                    }
                }
                if let Operation::HeldUploadRanges { origin, ranges, .. } = command {
                    if !matches!(origin.as_str(), "held: radiation (milestone 1)" | "held: land (milestone 1)")
                        || ranges.is_empty() || ranges.iter().any(|r| r.bytes == 0) {
                        return Err("held upload is not a labelled milestone-1 hold".into());
                    }
                }
                if let Operation::HostUpload { view, .. } | Operation::HostPublish { view, .. } = command {
                    let size = *host_sizes.get(&view.allocation).ok_or("unknown host view allocation")?;
                    if host_array::packed_size(size, &view.view()).map_err(str::to_owned)? == 0 {
                        return Err("empty live host view".into());
                    }
                }
                if let Operation::PointerTableUpload { rule, entries, .. } = command {
                    if !matches!(rule.as_str(), "resident_v842_species" | "cupy_stack_same_size"
                        // Continuation 17:40Z: [int(a.data.ptr) for a in arrays] as uint64, in order:
                        // cuda_physics_prep_v841.py:575-587 _device_pointer_table; rrfs/driver_rrfs.py:2657.
                        | "cuda_physics_prep_device_pointer_table" | "rrfs_stage_sources_rate_table") || entries.is_empty() {
                        return Err("unproved physics pointer table rule".into());
                    }
                }
                if let Operation::TempoPointerTableBind { binding, entries, device, .. } = command {
                    if binding.is_empty() || entries.is_empty() || *device != 0 {
                        return Err("invalid TEMPO table binding".into());
                    }
                }
                if let Operation::TempoNamedTableBind { binding, flags, group, .. } = command {
                    if binding.is_empty() || !matches!(group.as_str(), "fields" | "model" | "mesh" |
                        "diagnostics" | "tendencies" | "surface" | "precip" | "profiles") {
                        return Err("invalid named TEMPO table binding".into());
                    }
                    for flag in flags {
                        let bytes = host_sizes.get(&flag.allocation).ok_or("unknown TEMPO host flag")?;
                        if flag.offset.checked_add(4).is_none_or(|end| end > *bytes) {
                            return Err("TEMPO host flag outside allocation".into());
                        }
                    }
                }
                if let Operation::Kernel { arguments, .. } = command {
                    for argument in arguments {
                        if let Argument::HostScalar { source, cast } = argument {
                            let bytes = host_sizes.get(&source.allocation).ok_or("unknown live host scalar")?;
                            if source.offset.checked_add(cast.source_bytes()).is_none_or(|end| end > *bytes) {
                                return Err("live host scalar outside allocation".into());
                            }
                        }
                        if let Argument::CupyArray { shape, strides, item_bytes, .. } = argument {
                            if shape.len() != strides.len() || shape.len() > 64 || *item_bytes == 0 {
                                return Err("invalid CArray host metadata".into());
                            }
                            for value in shape.iter().chain(strides) {
                                let bytes = host_sizes.get(&value.allocation).ok_or("unknown CArray host metadata")?;
                                if value.offset.checked_add(8).is_none_or(|end| end > *bytes) {
                                    return Err("CArray host metadata outside allocation".into());
                                }
                            }
                        }
                        if let Argument::CupyCopyIndexer { shape, .. } = argument {
                            for value in shape {
                                let bytes = host_sizes.get(&value.allocation).ok_or("unknown CIndexer host metadata")?;
                                if value.offset.checked_add(8).is_none_or(|end| end > *bytes) {
                                    return Err("CIndexer host metadata outside allocation".into());
                                }
                            }
                        }
                        if let Argument::RkFrame { name } = argument {
                            if !matches!(name.as_str(), "full_dt_f32" | "split_i32" | "stage_i32" |
                                "acoustic_steps_i32" | "small_step_i32" | "large_dt_f32" | "acoustic_dt_f32") {
                                return Err("unknown RK frame scalar".into());
                            }
                        }
                        let fixed = match argument {
                            Argument::Bytes { fixed_source, .. }
                            | Argument::Patched { fixed_source, .. } => Some(fixed_source),
                            _ => None,
                        };
                        if let Some(fixed) = fixed {
                            let fixed = fixed.as_ref().ok_or("physics fixed argument lacks a source assertion")?;
                            // COMMON-AD 103.2 (coordinator R2, 16:21Z): a word identical at every aligned launch of every
                            // recorded step in two independent recordings; the receipt names both word tables.
                            // P1 (ad/phys): a labelled diagnostic package may carry single-recording words; only
                            // under AD_BINDER_DIAGNOSTIC=1, and such a run is never a proof.
                            let diagnostic = fixed.class == "diagnostic_single_recording"
                                && std::env::var("AD_BINDER_DIAGNOSTIC").as_deref() == Ok("1");
                            // P1 (ad/phys): CuPy concatenate array count, derived from the live table it reads.
                            let concat = fixed.class == "cupy_concat_count_from_table"
                                && concat_count_matches(command, argument, last_concat_table);
                            if !(diagnostic || concat || matches!(fixed.class.as_str(), "source_literal" | "frozen_mesh_geometry"
                                         | "recorded_constant_two_recordings"
                                         // G2: identical in both recordings at every step (word table line)
                                         | "recorded_per_step_two_recordings"))
                                || fixed.source.is_empty() || fixed.line == 0
                                || fixed.source_sha256.len() != 64
                                || !fixed.source_sha256.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)) {
                                return Err("invalid physics fixed argument source assertion".into());
                            }
                        }
                    }
                }
                if let Operation::VinterpOzone { pmid_shape, pin_size, ozone_shape, .. } = command {
                    for value in pmid_shape.iter().chain(std::iter::once(pin_size)).chain(ozone_shape) {
                        let bytes = host_sizes.get(&value.allocation).ok_or("unknown vinterp host shape")?;
                        if value.offset.checked_add(8).is_none_or(|end| end > *bytes) {
                            return Err("vinterp host shape outside allocation".into());
                        }
                    }
                }
                if let Operation::RadiationUnpack { rows, .. } = command {
                    if rows.len() > 23 { return Err("radiation unpack exceeds argument roster".into()); }
                    for row in rows {
                        if row.src.is_some() == row.fill_f32.is_some() {
                            return Err("radiation unpack copy/fill selection is invalid".into());
                        }
                        for view in std::iter::once(&row.dst).chain(row.src.iter()) {
                            if !(1..=2).contains(&view.shape.len()) || view.shape.len() != view.strides.len() {
                                return Err("invalid radiation unpack host metadata".into());
                            }
                            for word in view.shape.iter().chain(&view.strides) {
                                let bytes = host_sizes.get(&word.allocation).ok_or("unknown radiation unpack host metadata")?;
                                if word.offset.checked_add(8).is_none_or(|end| end > *bytes) {
                                    return Err("radiation unpack host metadata outside allocation".into());
                                }
                            }
                        }
                        if let Some(fill) = &row.fill_f32 {
                            let bytes = host_sizes.get(&fill.allocation).ok_or("unknown radiation unpack fill")?;
                            if fill.offset.checked_add(4).is_none_or(|end| end > *bytes) {
                                return Err("radiation unpack fill outside allocation".into());
                            }
                        }
                    }
                }
                if let Operation::TransportCopy { rows, .. } = command {
                    if rows.len() != 12 { return Err("transport copy needs twelve live rows".into()); }
                    for row in rows {
                        for view in [&row.dst, &row.src] {
                            if view.shape.len() > 4 || view.shape.len() != view.strides.len() {
                                return Err("invalid transport copy host metadata".into());
                            }
                            for word in view.shape.iter().chain(&view.strides) {
                                let bytes = host_sizes.get(&word.allocation).ok_or("unknown transport copy host metadata")?;
                                if word.offset.checked_add(8).is_none_or(|end| end > *bytes) {
                                    return Err("transport copy host metadata outside allocation".into());
                                }
                            }
                        }
                    }
                }
                if let Operation::AddInplace { target_shape, increment_shape, .. } = command {
                    for word in target_shape.iter().chain(increment_shape) {
                        let bytes = host_sizes.get(&word.allocation).ok_or("unknown add-inplace host shape")?;
                        if word.offset.checked_add(8).is_none_or(|end| end > *bytes) {
                            return Err("add-inplace host shape outside allocation".into());
                        }
                    }
                }
                if let Operation::RrfsCopyPlanUpload { cache, .. }
                    | Operation::RrfsCopyPlanKernel { cache, .. } = command {
                    let bytes = host_sizes.get(&cache.allocation).ok_or("unknown copy-plan cache owner")?;
                    if !host_carried.contains(&cache.allocation)
                        || cache.offset.checked_add(copy_plan::CACHE_BYTES).is_none_or(|end| end > *bytes) {
                        return Err("copy-plan cache must be a bounded carried host owner".into());
                    }
                }
                if let Operation::RrfsCopyPlanUpload { rows, cache, .. } = command {
                    if rows.is_empty() || rows.len() > 65535 { return Err("invalid copy-plan row count".into()); }
                    for row in rows {
                        if !matches!((row.src, row.src_stride.as_ref(), row.fill),
                                     (Some(_), Some(_), None) | (None, None, Some(_))) {
                            return Err("invalid copy-plan copy/fill inputs".into());
                        }
                        let words = row.dst_stride.iter().chain(row.shape.iter())
                            .chain(row.src_stride.iter().flatten()).map(|word| (word, 8))
                            .chain(row.fill.iter().map(|word| (word, 4)));
                        for (word, width) in words {
                            let bytes = host_sizes.get(&word.allocation).ok_or("unknown copy-plan input owner")?;
                            let end = word.offset.checked_add(width).ok_or("copy-plan input overflow")?;
                            if end > *bytes { return Err("copy-plan input outside allocation".into()); }
                            if word.allocation == cache.allocation && word.offset < cache.offset + copy_plan::CACHE_BYTES
                                && cache.offset < end {
                                return Err("copy-plan cache overlaps live input".into());
                            }
                        }
                    }
                }
                if let Operation::Allocate { bytes, overwritten_before_read, .. } = command {
                    if *bytes == 0 || !overwritten_before_read {
                        return Err("dynamic scratch is not certified overwritten before read".into());
                    }
                }
                match command {
                    Operation::NativeCubReduce { library_path, library_sha256, count, op, dtype, .. } => {
                        if library_path.is_empty() || library_sha256.len() != 64 {
                            return Err("invalid CUB reduction descriptor".into());
                        }
                        validate_library_host_dimensions(&host_sizes, &[*count])?;
                        native_cub::word_bytes(*op, *dtype)?;
                    }
                    Operation::NativeCubSegmentedReduce { library_path, library_sha256, segments, segment_size, op, dtype, .. } => {
                        if library_path.is_empty() || library_sha256.len() != 64 {
                            return Err("invalid CUB segmented reduction descriptor".into());
                        }
                        validate_library_host_dimensions(&host_sizes, &[*segments, *segment_size])?;
                        native_cub::word_bytes(*op, *dtype)?;
                    }
                    Operation::NativeThrustArgsort { library_path, library_sha256, count, dtype, .. } => {
                        if library_path.is_empty() || library_sha256.len() != 64 {
                            return Err("invalid Thrust argsort descriptor".into());
                        }
                        validate_library_host_dimensions(&host_sizes, &[*count])?;
                        native_thrust::word_bytes(*dtype)?;
                    }
                    Operation::NativeCublasTranspose { library_path, library_sha256, m, n, .. } => {
                        if library_path.is_empty() || library_sha256.len() != 64 {
                            return Err("invalid cuBLAS transpose descriptor".into());
                        }
                        validate_library_host_dimensions(&host_sizes, &[*m, *n])?;
                    }
                    _ => {}
                }
            }
        }
    } else {
        if steps > plan.cadence.len() { return Err("requested steps exceed cadence".into()); }
        for blocks in &plan.cadence[..steps] {
            if blocks.is_empty() { return Err("empty cadence frame".into()); }
            for name in blocks {
                let commands = plan.blocks.get(name).ok_or_else(|| format!("unknown block {name}"))?;
                if commands.is_empty() { return Err(format!("empty block {name}")); }
                for command in commands {
                    if let Operation::HostBoundary { name } = command {
                        return Err(format!("host boundary {name} requires a native ABI implementation"));
                    }
                }
            }
        }
    }
    let mut carried = HashSet::new();
    for id in &plan.carried {
        if !carried.insert(*id) { return Err(format!("duplicate carried allocation {id}")); }
    }
    validate_view_schedule(plan, steps)?;
    Ok(())
}

/// Bind every asserted source to its exact bytes and an existing line. This
/// checks provenance only: an independent proof must still establish that the
/// value is immutable and equals the ordinary host computation.
fn verify_fixed_argument_sources(plan: &Plan, base: &Path) -> Result<(), String> {
    if plan.physics.is_none() { return Ok(()); }
    let mut files: HashMap<PathBuf, (String, usize)> = HashMap::new();
    for commands in plan.blocks.values() {
        for command in commands {
            let Operation::Kernel { arguments, .. } = command else { continue };
            for argument in arguments {
                let fixed = match argument {
                    Argument::Bytes { fixed_source, .. }
                    | Argument::Patched { fixed_source, .. } => fixed_source.as_ref()
                        .ok_or("physics fixed argument lacks a source assertion")?,
                    _ => continue,
                };
                let source = path(base, &fixed.source);
                if !files.contains_key(&source) {
                    let bytes = fs::read(&source).map_err(|e| format!("cannot read fixed argument source {}: {e}", source.display()))?;
                    verify_sha(&bytes, &fixed.source_sha256)
                        .map_err(|_| format!("fixed argument source SHA-256 mismatch: {}", source.display()))?;
                    let text = std::str::from_utf8(&bytes).map_err(|_| "fixed argument source is not UTF-8")?;
                    let lines = text.lines().count();
                    files.insert(source.clone(), (fixed.source_sha256.clone(), lines));
                }
                let (sha, lines) = &files[&source];
                if sha != &fixed.source_sha256 || fixed.line > *lines {
                    return Err(format!("fixed argument source line or SHA-256 is invalid: {}", source.display()));
                }
            }
        }
    }
    Ok(())
}

/// Refuse CIndexer scratch initialization unless the loaded image and its
/// archived source are one of the independently checked set-before-get copy
/// pairs. The source line itself is also checked before any CUDA work.
/// Package-local CIndexer proofs (ad-record): the plan directory's `cindexer-pairs.json`, written by
/// prove_cindexer_pairs.py for THIS recording (the same tool whose results the compiled allow-lists in carray.rs
/// mirror). An on-demand recording from a new analysis state can execute CuPy images no earlier recording did (the
/// 2026-11-07 from-state pkg-id18 added four rank-1 copies), and a runner rebuilt per recording is not a recorder.
/// Only entries the tool marked set_before_get are read; the runner still re-checks the source line itself below,
/// and for a package-admitted copy also that no `_ind.get` precedes the set (the reduction rule's no-earlier-read).
fn package_cindexer_pairs(base: &Path) -> Result<Vec<(String, usize, String, String, String)>, String> {
    let file = base.join("cindexer-pairs.json");
    if !file.exists() { return Ok(Vec::new()); }
    let text = fs::read_to_string(&file).map_err(|e| format!("package CIndexer proofs: {e}"))?;
    let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("package CIndexer proofs: {e}"))?;
    let mut out = Vec::new();
    for pair in value.get("pairs").and_then(|p| p.as_array()).ok_or("package CIndexer proofs: no pairs array")? {
        if pair.get("set_before_get").and_then(|v| v.as_bool()) != Some(true) { continue; }
        let field = |k: &str| pair.get(k).and_then(|v| v.as_str()).map(str::to_string);
        if let (Some(function), Some(rank), Some(source), Some(image), Some(kind)) = (field("function"),
            pair.get("rank").and_then(|v| v.as_u64()), field("source_sha256"), field("image_sha256"), field("kind")) {
            out.push((function, rank as usize, source, image, kind));
        }
    }
    Ok(out)
}

fn verify_copy_indexer_sources(plan: &Plan, base: &Path) -> Result<(), String> {
    let package_pairs = package_cindexer_pairs(base)?;
    for command in plan.blocks.values().flatten() {
        let Operation::Kernel { module, function, arguments, .. } = command else { continue };
        for argument in arguments {
            let Argument::CupyCopyIndexer { shape, source } = argument else { continue };
            let image = plan.modules.iter().find(|entry| entry.id == *module)
                .ok_or("CIndexer module is missing")?;
            let reduction = source.class == "cupy_reduction_in_indexer_set_before_get";
            let proved = match source.class.as_str() {
                "cupy_copy_indexer_set_before_get" =>
                    carray::copy_indexer_proved(function, shape.len(), &source.source_sha256, &image.sha256),
                "cupy_reduction_in_indexer_set_before_get" =>
                    carray::reduction_in_indexer_proved(function, shape.len(), &source.source_sha256, &image.sha256),
                _ => false,
            };
            let package_kind = if reduction { "reduction_in" } else { "copy" };
            let package_proved = !proved && package_pairs.iter().any(|(f, r, s, i, k)| f == function && *r == shape.len()
                && *s == source.source_sha256 && *i == image.sha256 && k == package_kind);
            if source.line == 0 || !(proved || package_proved) {
                return Err(format!("CIndexer source/image pair is not proved: {function} rank {} source {} image {}",
                    shape.len(), &source.source_sha256[..12.min(source.source_sha256.len())], &image.sha256[..12.min(image.sha256.len())]));
            }
            let bytes = fs::read(path(base, &source.source)).map_err(|e| format!("CIndexer source: {e}"))?;
            verify_sha(&bytes, &source.source_sha256).map_err(|_| "CIndexer source SHA-256 mismatch")?;
            let text = std::str::from_utf8(&bytes).map_err(|_| "CIndexer source is not UTF-8")?;
            let mut lines = text.lines().skip(source.line - 1);
            if reduction {
                // The input indexer is set on this line and first read on the next (CuPy reduction template).
                if lines.next().map(str::trim) != Some("_in_ind.set(_j);") ||
                   !lines.find(|line| !line.trim().is_empty()).is_some_and(|line| line.contains("_in_ind.get()")) ||
                   text.lines().take(source.line - 1).any(|line| line.contains("_in_ind.get")) {
                    return Err("CIndexer set-before-get source line differs".into());
                }
            } else if (package_proved && text.lines().take(source.line - 1).any(|line| line.contains("_ind.get"))) ||
               lines.next().map(str::trim) != Some("CUPY_FOR(i, _ind.size()) {") ||
               lines.find(|line| !line.trim().is_empty()).map(str::trim) != Some("_ind.set(i);") {
                return Err("CIndexer set-before-get source line differs".into());
            }
        }
    }
    Ok(())
}

struct Execution {
    cuda: Cuda,
    pub(crate) context: *mut c_void,
    allocations: HashMap<u64, submit::Allocation>,
    /// anchors-ahead: bytes of the live allocations now, and their peak (the size a whole-state reverse anchor can reach).
    live_bytes: usize, live_peak: usize,
    host_state: host_state::HostState,
    rk: Option<rk_schedule::Scope>,
    streams: HashMap<u64, *mut c_void>,
    events: HashMap<u64, *mut c_void>,
    modules: HashMap<String, *mut c_void>,
    functions: HashMap<(String, String), *mut c_void>,
    cub_libraries: HashMap<(String, String), native_cub::NativeCub>,
    thrust_libraries: HashMap<(String, String), native_thrust::NativeThrust>,
    cublas_libraries: HashMap<(String, String), native_cublas::NativeCublas>,
    tempo_cache: tempo::Cache,
    tempo_bindings: HashMap<String, (plan::Slice, usize)>,
    digest_module: *mut c_void,
    digest_function: *mut c_void,
    digest_output: u64,
    /// S04 (taylor-row-host): the host digest's carried bytes land in this reused page-locked arena (DMA; no fresh
    /// pageable vectors to fault in and memcpy through the driver's staging buffer every step).
    digest_arena: Option<pinned::PinnedBuf>,
    /// S04: wall seconds spent in the step-boundary digest (receipt timing only; nothing reads it back).
    digest_s: f64,
    word_module: *mut c_void,
    word_function: *mut c_void,
    word_output: u64,
    pinned: Vec<*mut c_void>,
    submitter: submit::Submitter,
    trace: Option<BufWriter<File>>,
    /// P1 (ad/phys): carried allocations holding device-address tables the live rules rebuild from current
    /// allocations (their bytes differ from the recording by construction); hashed and printed, not compared.
    address_bearing: HashSet<u64>,
    /// P1 (ad/phys): recorded (reference) base address per allocation id; see plan.rs recorded_bases.
    recorded_bases: HashMap<u64, u64>,
    /// outer2 (2026-11-07): the driver pointer of every allocation placed congruent to its recorded base (see
    /// `allocate_congruent`); the allocation's address is that pointer plus the shift, and frees use this one.
    raw_bases: HashMap<u64, u64>,
    /// P1 (ad/phys): declared residue per (step, allocation): byte ranges zeroed before the compare, and the
    /// forward's boundary hash with the same ranges zeroed (see plan.rs excluded_residue).
    residue: HashMap<(usize, u64), (Vec<(usize, usize)>, String)>,
    /// P1 (ad/phys): per carried allocation, the address words translated at the previous step boundary
    /// (word index -> (replay word, forward word)); lets a word whose low or high 4 bytes were since cleared by
    /// a 4-byte clear/fill be translated as the same half of the same old pointer (forward_address_view_tracked).
    pointer_history: HashMap<u64, PointerHistory>,
    /// P1 sealer-output-steps (ad/phys): the held-blob address words relocated during the current step, by
    /// destination allocation (word index -> (replay word, forward word)). Merged into `pointer_history` at the
    /// step boundary, so a held pointer whose low 4 bytes a later 4-byte clear zeroed in the same step is read as
    /// the same high half of the same pointer. Breakage: the sealed 18-step hrrrv5 plan failed its strict replay
    /// at step 9 on allocation 133 only: the step-9 land hold writes an 11-entry table there (word 0 the forward's
    /// 0x7e579dd27000, allocation 425 + 275456, relocated), a 4-byte clear then zeroes its low half, and the
    /// boundary held 0x7e5700000000 in the forward and the replay's lone high half 0x77b500000000.
    held_pointer_writes: HashMap<u64, PointerHistory>,
    /// adjoint-speed (window only): the whole device state downloaded once at a step boundary (bytes, and per
    /// allocation id: (offset, bytes, address)). The host digest reads carried views from it; None on the replay path.
    /// terrain-w-hash: shared (Arc) so a deferred boundary digest can read the same bytes while store() writes them.
    /// fwdcheck-digest: the bytes are pageable (Win::capture_state) or, in a C-ABI session's first tangent sweep, the
    /// boundary's own page-locked RAM copy (CapData::Host), so the digest and the store read one download.
    pub(crate) captured: Option<(std::sync::Arc<CapData>, Vec<(u64, usize, usize, u64)>)>,
    /// terrain-w-hash (round 3, window tangent pass only): when set, a boundary whose carried views all lie in the
    /// step's capture is digested on background threads while the card runs the next step; the next finish_step
    /// (or digest_flush) joins it and prints, compares and refuses exactly as the synchronous path. False on the
    /// replay path and every other window pass, which stay synchronous.
    pub(crate) digest_defer: bool,
    /// per-step-host-hash (2026-11-08): set by the C-ABI tangent step (capi.rs tl_step) for a step that has no
    /// reference to compare against and stores no boundary (the trajectory is already bound). The boundary digest of
    /// such a step only downloaded every carried block and scanned it for the pointer history the next digest reads;
    /// no hash was computed or compared (integ8 U8/L1), and the library prints none (runner-lib quiet hashes). That cost
    /// 0.63 s of every 1.81 s tangent step with the card idle (inner-iteration/tl-pass-idle T3). The step skips it, and
    /// the pointer history is marked stale until the next restore (restore_host) puts back a stored one.
    /// HEXAD_TL_UNCHECKED_DIGEST=1 keeps the old per-step digest (proof runs); check_every_pass, the first sweep and
    /// forward_check pass references and are never skipped.
    pub(crate) unchecked_digest_skip: bool,
    /// per-step-host-hash: the pointer history is stale (a skipped digest did not advance it). A digest or a stored
    /// boundary reading it refuses (it would read a later boundary's pointers with an older history: a false
    /// state-hash mismatch, or a boundary stored with the wrong history); restore_host clears it.
    pub(crate) history_stale: bool,
    /// per-step-host-hash: boundary digests skipped (receipt only).
    pub(crate) digests_skipped: u64,
    digest_job: Option<DigestJob>,
    /// terrain-w-hash: seconds the deferred digests ran in the background (receipt timing only).
    pub(crate) digest_bg_s: f64,
    /// G5 (ad/phys-window-g5): the window's pre-decoded blobs (sha256 -> bytes, each decoded by `blob` and SHA-256
    /// checked). None on the replay path, which decodes every blob where it is used, as before.
    blob_cache: Option<HashMap<String, std::sync::Arc<Vec<u8>>>>,
    /// G5: window-only pinned staging pool for cached-blob uploads: buffers return here instead of cuMemFreeHost
    /// when the step or segment drains (size per buffer in `pinned_size`). Empty on the replay path.
    pinned_pool: Vec<(*mut c_void, usize)>,
    pinned_size: HashMap<usize, usize>,
    /// Physics-on window stage A: per-launch derivative dot (None unless AD_DOT_* is set).
    ad: Option<addot::AdDot>,
    /// S13 (window only): a perturbed forward (a Taylor row) may follow a live LW layer histogram that differs
    /// from the recorded one: the read stores the live histogram here instead of refusing, and the window
    /// replays the LW group section once per pass with a rebuilt column order (window.rs lw_regroup). False on
    /// the replay path and on every reference-checked or tangent forward, which refuse as before.
    lw_regroup: bool,
    /// S13: diagnostic only (AD_LW_REGROUP_TEST=1/2/3, default 0 = off): store the histogram even when it matches,
    /// so a reference-checked forward replays the LW section multi-pass (layer_groups::regroup_passes test modes).
    lw_regroup_test: u8,
    lw_live_hist: Option<Vec<i32>>,
    /// lane rung3-params (window --params, default empty): plan kernels replaced by function name with the P4
    /// parameter-carrier image's verbatim forward (same source, same signature; reads p4_parf(k, literal)). Empty
    /// on every other path, so the gradients-off replay launches exactly the plan's functions.
    kernel_override: HashMap<String, *mut c_void>,
}

/// P1 (ad/phys): the carried bytes as the forward held them: every aligned 8-byte word pointing into a live
/// replay allocation [address, address + bytes] becomes the recorded base of that allocation plus the offset.
/// `ranges` is sorted by start: (start, end inclusive, recorded base). Returns None when no word moved.
/// P1 (ad/phys): SHA-256 of a forward-address view with the declared residue ranges zeroed.
fn masked_digest(view: &mut [u8], zeroed: &[(usize, usize)]) -> Result<String, String> {
    for &(offset, bytes) in zeroed {
        let end = offset.checked_add(bytes).ok_or("residue range overflow")?;
        view.get_mut(offset..end).ok_or("residue range outside its allocation")?.fill(0);
    }
    Ok(hex_bytes(&Sha256::digest(&*view)))
}

fn forward_address_view(data: &[u8], ranges: &[(u64, u64, u64)], low: u64, high: u64) -> Option<(Vec<u8>, usize)> {
    forward_address_view_tracked(data, ranges, low, high, None).0
}

/// P1 (ad/phys): address words translated at one step boundary, by word index: (replay word, forward word).
type PointerHistory = HashMap<usize, (u64, u64)>;

/// P1 sealer-output-steps: words whose high half is still the history pointer's and whose low half was rewritten,
/// but which also lie inside a live replay allocation, so the ordinary rule translated them as pointers. Returns
/// (word index, the half-rule value). The boundary hash tries these only when the ordinary view misses the
/// forward's hash, and the low 4 bytes are kept as written either way.
fn ambiguous_half_words(data: &[u8], ranges: &[(u64, u64, u64)], low: u64, high: u64,
                        previous: Option<&PointerHistory>) -> Vec<(usize, u64)> {
    let Some(previous) = previous else { return Vec::new() };
    let mut out = Vec::new();
    for (&index, &(replay, forward)) in previous {
        let Some(chunk) = data.get(8 * index..8 * index + 8) else { continue };
        let word = u64::from_le_bytes(chunk.try_into().unwrap());
        // G2: a lone half need not land inside a live allocation (then the plain view leaves it raw, with the replay's
        // high half, which is just as wrong); the caller tries these readings in subsets, never blindly.
        let _ = (ranges, low, high);
        if word != replay && word & HIGH_HALF == replay & HIGH_HALF && replay & HIGH_HALF != 0
            && word & !HIGH_HALF != replay & !HIGH_HALF {
            out.push((index, (forward & HIGH_HALF) | (word & !HIGH_HALF)));
        }
    }
    out.sort_unstable();
    out
}

/// P1 sealer-output-steps: true when `word` lies inside a live replay allocation (the ordinary translation rule).
fn inside_live(word: u64, ranges: &[(u64, u64, u64)], low: u64, high: u64) -> bool {
    if word < low || word > high { return false; }
    let k = ranges.partition_point(|r| r.0 <= word);
    k > 0 && word <= ranges[k - 1].1
}
const HIGH_HALF: u64 = 0xffff_ffff_0000_0000;

/// P1 (ad/phys): forward_address_view plus half-cleared pointer words. A 4-byte clear or fill (cuMemsetD32, a
/// copy-plan int32 fill, a 4-byte flag store) over one half of an old pointer leaves the other half in the word
/// (hrrrv5 pkg6s step 5: allocation 100 word 2, forward 0x7e5700000000, replay 0x78ac00000000; allocation 202
/// word 0). That lone half is not a live pointer, but it can land inside an unrelated replay allocation
/// (0x78ac00000000 is allocation 453's replay base) and would be "translated" to that allocation's recorded
/// base, which the forward never held. So when `previous` (this allocation's words translated at the last
/// boundary) shows the word held the full pointer r -> f, and the word now equals r with one half cleared to
/// zero, the view gets f with the same half cleared: exactly what the forward's same clear left. It is never
/// read as a live pointer. Any other value takes the ordinary rule. Returns (view if any word changed, words
/// translated), the history for the next boundary, and the number of half words.
fn forward_address_view_tracked(data: &[u8], ranges: &[(u64, u64, u64)], low: u64, high: u64,
                                previous: Option<&PointerHistory>)
                                -> (Option<(Vec<u8>, usize)>, PointerHistory, usize) {
    let mut out: Option<Vec<u8>> = None;
    let mut moved = 0_usize;
    let mut halves = 0_usize;
    let mut history = PointerHistory::new();
    for (index, chunk) in data.chunks_exact(8).enumerate() {
        let word = u64::from_le_bytes(chunk.try_into().unwrap());
        if let Some(&(replay, forward)) = previous.and_then(|p| p.get(&index)) {
            let kept = if word == replay { None }
                else if word == replay & HIGH_HALF && replay & !HIGH_HALF != 0 { Some(forward & HIGH_HALF) }
                else if word == replay & !HIGH_HALF && replay & HIGH_HALF != 0 { Some(forward & !HIGH_HALF) }
                // P1 sealer-output-steps: low half rewritten by a 4-byte data write, high half still the old
                // pointer's (and the word is no live replay address, which takes the ordinary rule below). Breakage:
                // the sealed 18-step hrrrv5 plan, strict replay, step 10, allocation 142 word 93: forward
                // 0x7e573a48cf2f, replay 0x77b53a48cf2f (both the old pointer's high half under the same 4 data bytes).
                else if word & HIGH_HALF == replay & HIGH_HALF && replay & HIGH_HALF != 0
                    && !inside_live(word, ranges, low, high) { Some((forward & HIGH_HALF) | (word & !HIGH_HALF)) }
                else { None };
            if let Some(value) = kept {
                if value != word {
                    out.get_or_insert_with(|| data.to_vec())[8 * index..8 * index + 8].copy_from_slice(&value.to_le_bytes());
                }
                history.insert(index, (replay, forward));
                halves += 1;
                continue;
            }
        }
        if word < low || word > high { continue; }
        let k = ranges.partition_point(|r| r.0 <= word);
        if k == 0 { continue; }
        let (start, end, recorded) = ranges[k - 1];
        if word > end { continue; }
        let forward = recorded + (word - start);
        out.get_or_insert_with(|| data.to_vec())[8 * index..8 * index + 8].copy_from_slice(&forward.to_le_bytes());
        history.insert(index, (word, forward));
        moved += 1;
    }
    (out.map(|bytes| (bytes, moved)), history, halves)
}

/// P1 (ad/phys): the boundary digest of one carried block. The forward-address view translates every word that
/// lies inside a live replay allocation, but a block can also hold forward data that only looks like an address:
/// hrrrv5 pkg6s allocation 449 carries 221 stale 0x7xxx_xxxx_xxxx words (left from other processes, byte-identical
/// in forward and replay), and whichever of them a run's randomly placed replay allocation happens to cover was
/// "translated" (449 step 2, 421 step 4, 378/402/450 step 1, 447/452/453 in other runs: the run-to-run variation).
/// So when the view differs from the raw bytes and does not give the forward hash, the raw bytes are hashed
/// too: if they give the forward hash the block is byte-identical to the forward, nothing in it was a pointer, and
/// it passes on its raw bytes (`raw` true, its translated words dropped from the history). A block that matches
/// neither keeps the view hash and fails as before. Returns (hash, raw bytes matched).
fn boundary_digest(data: &[u8], view: Option<&[u8]>, expected: Option<&str>) -> (String, bool) {
    boundary_digest_pre(data, view, expected, None)
}
/// integ8: boundary_digest with the raw bytes' sha256 already computed (`raw`, from the overlapped digest of a large
/// block); the same readings in the same order, so the same (hash, raw) result.
fn boundary_digest_pre(data: &[u8], view: Option<&[u8]>, expected: Option<&str>, raw: Option<&str>) -> (String, bool) {
    let raw_hash = || raw.map(str::to_owned).unwrap_or_else(|| hex_bytes(&Sha256::digest(data)));
    let Some(view) = view else { return (raw_hash(), false) };
    let translated = hex_bytes(&Sha256::digest(view));
    match expected {
        Some(expected) if expected != translated => {
            let raw = raw_hash();
            if raw == expected { (raw, true) } else { (translated, false) }
        }
        _ => (translated, false),
    }
}
/// integ8 (U8/L1 host digest; measured on a pkg18m2 replay, steps 1-4: 1.7-1.9 s of host digest per step while the
/// card waits, of which a 980 MB block's address scan + sha256 took 1.01 s on one worker and the excluded-residue
/// blocks' masked digests 0.6 s on the main thread): blocks at least this large get their raw sha256 on a second
/// thread beside the address scan. The same bytes are hashed and the same readings chosen.
const DIGEST_OVERLAP_BYTES: usize = 64 << 20;
/// integ8: the masked digest of one excluded-residue block with its raw-bytes fallback and half-word readings, moved
/// off the main thread into the digest workers (it was a sequential loop; step6-rucland's reading order is kept).
/// Returns the masked hash and, when a half-word reading matched, its word count; a matched reading's old pointers go
/// into `history` exactly as the sequential loop inserted them.
#[allow(clippy::too_many_arguments)]
fn residue_masked_reading(data: &[u8], ranges: &[(u64, u64, u64)], low: u64, high: u64, previous: Option<&PointerHistory>,
                          zeroed: &[(usize, usize)], masked_expected: &str, history: &mut PointerHistory)
                          -> Result<(String, Option<usize>), String> {
    let translated_view = forward_address_view_tracked(data, ranges, low, high, previous).0;
    let mut view = translated_view.as_ref().map(|v| v.0.clone()).unwrap_or_else(|| data.to_vec());
    let mut masked = masked_digest(&mut view, zeroed)?;
    // same raw-bytes fallback as boundary_digest: a byte-identical block passes on its raw bytes
    if masked != *masked_expected && translated_view.is_some() {
        let raw = masked_digest(&mut data.to_vec(), zeroed)?;
        if raw == *masked_expected { masked = raw; }
    }
    // step6-rucland: the same half-word readings (subsets) as every other block (allocation 394, step 18).
    let mut found = None;
    if masked != *masked_expected {
        let alternates = ambiguous_half_words(data, ranges, low, high, previous);
        let n = alternates.len().min(12);
        let base = translated_view.as_ref().map(|v| v.0.clone()).unwrap_or_else(|| data.to_vec());
        let mut masks: Vec<u32> = (1..(1u32 << n)).collect();
        masks.sort_by_key(|m| (std::cmp::Reverse(*m == (1u32 << n) - 1), m.count_ones(), *m));
        for mask in masks {
            let mut alternate = base.clone();
            let chosen: Vec<&(usize, u64)> = alternates.iter().take(n).enumerate()
                .filter(|(k, _)| mask >> k & 1 == 1).map(|(_, a)| a).collect();
            for (index, word) in &chosen { alternate[8 * index..8 * index + 8].copy_from_slice(&word.to_le_bytes()); }
            let hashed = masked_digest(&mut alternate, zeroed)?;
            if hashed == *masked_expected {
                masked = hashed;
                // keep the old pointer for the next boundary, as the unmasked path does
                for (index, _) in &chosen {
                    if let Some(entry) = previous.and_then(|p| p.get(index)) { history.insert(*index, *entry); }
                }
                found = Some(chosen.len());
                break;
            }
        }
    }
    Ok((masked, found))
}

/// P1 a test box (8x5090, 2026-11-04): most words in doubt (changed, not proven by the last boundary) a mixed block
/// may hold for the per-word reading.
const MIXED_MAX_WORDS: usize = 6;
/// Most doubtful words for the complement search (keep 0, 1 or 2 translated, revert the rest): 64 words, 2,081 hashes.
const MIXED_MAX_COMPLEMENT: usize = 64;

/// P1 a test box: the forward hash of a block whose translated words are a mix of real stale pointers and stale data
/// that only looks like a replay address. Breakage: sealed 18-step hrrrv5 plan, a test box: allocation 454 (89.6 MB,
/// not address-bearing) failed step 15 in the strict replay with 2 translated words and passed with 1 in the
/// continue replay, and the reverse at step 3; the step-3 dump against recording A's boundary payload differs in
/// exactly one word (replay 0x7c8524162514, forward 0x798a34162514, inside allocation 447), and the replay bytes
/// with that word set to the forward's give the forward hash, while 1,142 other 0x7xxx words are byte-identical
/// data, one of which a run's replay allocation can cover. Tries every subset of the changed words (fewest
/// first, at most MIXED_MAX_WORDS words) reverted to the raw bytes; returns (hash, reverted word indices) on a
/// match with `expected`, else None (the block keeps its view hash and fails as before).
fn mixed_reading(data: &[u8], viewed: &[u8], expected: &str, previous: Option<&PointerHistory>)
                 -> Option<(String, Vec<usize>)> {
    let changed: Vec<usize> = (0..data.len() / 8)
        .filter(|&i| viewed[8 * i..8 * i + 8] != data[8 * i..8 * i + 8]).collect();
    // A word the last boundary translated from the same replay value passed that boundary's hash as a pointer and
    // is unchanged since, so it stays translated; only the others are in doubt. Breakage: a test box continue replay
    // proof18c2, allocation 446 (34.4 MB) step 18: 61 translated words (the strict run: 59) after 84 at step 17
    // that all passed, too many for a search over every changed word.
    let doubt: Vec<usize> = changed.iter().copied().filter(|i| {
        let word = u64::from_le_bytes(data[8 * i..8 * i + 8].try_into().unwrap());
        !previous.and_then(|p| p.get(i)).is_some_and(|&(replay, _)| replay == word)
    }).collect();
    let n = doubt.len();
    if n > MIXED_MAX_WORDS && n <= MIXED_MAX_COMPLEMENT {
        // Many doubtful words: the usual case is a few real stale pointers among byte-identical lookalikes, so
        // keep 0, 1 or 2 of them translated and revert every other one to raw (1 + n + n(n-1)/2 hashes).
        let mut alternate = data.to_vec();
        for &i in &changed { if !doubt.contains(&i) { alternate[8 * i..8 * i + 8].copy_from_slice(&viewed[8 * i..8 * i + 8]); } }
        // keep none: every doubtful word is data and the proven ones stay translated. Breakage: allocation 454 step 3,
        // 1 real pointer proven by the held pointer writes of the step, 7 doubtful lookalikes, all of them data.
        if hex_bytes(&Sha256::digest(&alternate)) == expected { return Some((expected.to_string(), doubt.clone())); }
        for a in 0..n { for b in a..n {
            let keep = if a == b { vec![doubt[a]] } else { vec![doubt[a], doubt[b]] };
            for &i in &keep { alternate[8 * i..8 * i + 8].copy_from_slice(&viewed[8 * i..8 * i + 8]); }
            if hex_bytes(&Sha256::digest(&alternate)) == expected {
                return Some((expected.to_string(), doubt.iter().copied().filter(|i| !keep.contains(i)).collect()));
            }
            for &i in &keep { alternate[8 * i..8 * i + 8].copy_from_slice(&data[8 * i..8 * i + 8]); }
        } }
        return None;
    }
    if n == 0 || n > MIXED_MAX_WORDS { return None; }
    // with every changed word in doubt, reverting all of them is the raw fallback boundary_digest already tried
    let top: u32 = if n == changed.len() { (1_u32 << n) - 1 } else { 1_u32 << n };
    let mut masks: Vec<u32> = (1..top).collect();
    masks.sort_by_key(|m| m.count_ones());
    let mut alternate = viewed.to_vec();
    for mask in masks {
        let back: Vec<usize> = (0..n).filter(|b| mask & (1 << b) != 0).map(|b| doubt[b]).collect();
        for &i in &back { alternate[8 * i..8 * i + 8].copy_from_slice(&data[8 * i..8 * i + 8]); }
        if hex_bytes(&Sha256::digest(&alternate)) == expected { return Some((expected.to_string(), back)); }
        for &i in &back { alternate[8 * i..8 * i + 8].copy_from_slice(&viewed[8 * i..8 * i + 8]); }
    }
    None
}

/// terrain-w-hash: background digest workers (results do not depend on the count; a cap leaves the cores the next
/// step's dispatch and gather threads use).
const DIGEST_BG_WORKERS: usize = 24;
/// terrain-w-hash: a boundary digest running beside the next step (window tangent pass).
struct DigestJob {
    step: usize,
    handle: std::thread::JoinHandle<DigestJobOut>,
    cell: std::sync::Arc<std::sync::OnceLock<HashMap<u64, PointerHistory>>>,
    workers: usize,
    dprof: bool,
    t_dl: f64,
}
struct DigestJobOut {
    out: DigestOut,
    blocks: Vec<(u64, usize)>,
    residue: HashMap<(usize, u64), (Vec<(usize, usize)>, String)>,
    refs: Option<References>,
    ab: HashSet<u64>,
    bg_s: f64,
}
/// terrain-w-hash: a boundary's pointer history as a HostSnap holds it: known, or filled when the deferred digest of
/// that boundary is joined (digest_flush; every reader runs after a flush).
#[derive(Clone)]
pub(crate) enum HistSnap { Ready(HashMap<u64, PointerHistory>), Pending(std::sync::Arc<std::sync::OnceLock<HashMap<u64, PointerHistory>>>) }
impl HistSnap {
    pub(crate) fn get(&self) -> &HashMap<u64, PointerHistory> {
        match self {
            HistSnap::Ready(m) => m,
            HistSnap::Pending(c) => c.get().expect("boundary pointer history read before its deferred digest was joined (digest_flush missing)"),
        }
    }
}
/// terrain-w-hash: what the digest workers produce for one boundary (per carried block, in carried order).
struct DigestOut {
    values: Vec<String>,
    moved: Vec<usize>,
    histories: Vec<PointerHistory>,
    halves: Vec<usize>,
    raw_equal: Vec<bool>,
    residue_out: Vec<Option<Result<(String, Option<usize>), String>>>,
    item_t: Vec<(u64, usize, f64, f64)>,
    workers_s: f64,
}
/// terrain-w-hash: the digest workers of one boundary (moved out of finish_step_host_hash unchanged): per block the
/// forward-address reading, its hash, and the raw / half-word / mixed / residue readings, largest blocks first.
#[allow(clippy::too_many_arguments)]
fn digest_blocks(step: usize, buffers: &[(u64, &[u8])], ranges: &[(u64, u64, u64)], low: u64, high: u64,
                 previous: &HashMap<u64, PointerHistory>, residue: &HashMap<(usize, u64), (Vec<(usize, usize)>, String)>,
                 address_bearing_ids: &HashSet<u64>, references: Option<&References>, workers: usize, dprof: bool) -> DigestOut {
    let t0 = std::time::Instant::now();
    let item_t: std::sync::Mutex<Vec<(u64, usize, f64, f64)>> = std::sync::Mutex::new(Vec::new());
    // S04: workers take the largest allocations first, so the longest single digest starts at once instead of
    // after the small ones (results land in their own slots: order of output unchanged).
    let mut order: Vec<usize> = (0..buffers.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(buffers[i].1.len()));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut values = vec![String::new(); buffers.len()];
    let mut moved = vec![0_usize; buffers.len()];
    let mut histories = vec![PointerHistory::new(); buffers.len()];
    let mut halves = vec![0_usize; buffers.len()];
    let mut raw_equal = vec![false; buffers.len()];
    let mut residue_out: Vec<Option<Result<(String, Option<usize>), String>>> = vec![None; buffers.len()];
    {
        type Slot<'a> = (&'a mut String, &'a mut usize, &'a mut PointerHistory, &'a mut usize, &'a mut bool,
                         &'a mut Option<Result<(String, Option<usize>), String>>);
        let slots: Vec<std::sync::Mutex<Slot>> =
            values.iter_mut().zip(moved.iter_mut()).zip(histories.iter_mut()).zip(halves.iter_mut())
                .zip(raw_equal.iter_mut()).zip(residue_out.iter_mut())
                .map(|(((((v, m), h), x), r), q)| std::sync::Mutex::new((v, m, h, x, r, q))).collect();
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| loop {
                    let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if k >= order.len() { break; }
                    let index = order[k];
                    let (id, data) = (&buffers[index].0, &buffers[index].1);
                    let t_item = std::time::Instant::now();
                    // integ8 (U8/L1): an unchecked forward (no references: a Taylor row or an analysis pass)
                    // compares no state hash, so its blocks get the address scan only (the pointer history the next
                    // boundary reads with, exactly as before) and no SHA-256: 1.0 s of the 1.3 s of every Taylor-row step.
                    let checked_step = references.is_some();
                    let ((view, mut history, half), raw_pre) = if checked_step && data.len() >= DIGEST_OVERLAP_BYTES {
                        std::thread::scope(|inner| {
                            let raw = inner.spawn(|| hex_bytes(&Sha256::digest(data)));
                            let v = forward_address_view_tracked(data, &ranges, low, high, previous.get(id));
                            (v, Some(raw.join().unwrap()))
                        })
                    } else { (forward_address_view_tracked(data, &ranges, low, high, previous.get(id)), None) };
                    let t_view = t_item.elapsed().as_secs_f64();
                    // residue blocks compare a masked digest below; the raw fallback is applied there
                    let expected = if residue.contains_key(&(step, *id)) { None }
                        else { references.and_then(|r| r.get(&(step, *id))).map(|e| e.1.as_str()) };
                    let (mut value, raw) = if checked_step { boundary_digest_pre(data, view.as_ref().map(|v| v.0.as_slice()), expected, raw_pre.as_deref()) }
                        else { (String::from("not computed (unchecked forward)"), false) };
                    let mut half = half;
                    if !raw && expected.is_some_and(|e| e != value) {
                        let alternates = ambiguous_half_words(data, &ranges, low, high, previous.get(id));
                        // G2 (P1 milestone 2): each candidate is read as a half word or as it stands, in subsets (all of
                        // them first, then fewest first, at most 2^12 readings). THE BREAKAGE THIS PREVENTS: the
                        // radiation-live plan's TEMPO table (allocation 107, step 16) got 20 new full pointers in the
                        // same 4 GB region as old history entries, so they matched the half-word test and were all
                        // rewritten, while the one real half word (word 93) was outside every live allocation and left
                        // raw: no reading gave the forward hash.
                        let base = view.as_ref().map(|v| v.0.clone()).unwrap_or_else(|| data.to_vec());
                        let n = alternates.len().min(12);
                        let mut masks: Vec<u32> = (1..(1u32 << n)).collect();
                        masks.sort_by_key(|m| (std::cmp::Reverse(*m == (1u32 << n) - 1), m.count_ones(), *m));
                        for mask in masks {
                            let mut alternate = base.clone();
                            let chosen: Vec<&(usize, u64)> = alternates.iter().take(n).enumerate()
                                .filter(|(k, _)| mask >> k & 1 == 1).map(|(_, a)| a).collect();
                            for (index, word) in &chosen {
                                alternate[8 * index..8 * index + 8].copy_from_slice(&word.to_le_bytes());
                            }
                            let hashed = hex_bytes(&Sha256::digest(&alternate));
                            if expected == Some(hashed.as_str()) {
                                value = hashed; half += chosen.len();
                                // keep the old pointer for the next boundary, so the same word gets the same reading
                                // (sealed 18-step hrrrv5 plan: allocation 163 word 93 passed step 12, failed step 13)
                                for (index, _) in &chosen {
                                    if let Some(entry) = previous.get(id).and_then(|p| p.get(index)) {
                                        history.insert(*index, *entry);
                                    }
                                }
                                break;
                            }
                        }
                    }
                    // P1 a test box: a block can hold both kinds of 0x7xxx word at once, a stale pointer that needs the
                    // forward's address and a stale data word that only looks like one and is byte-identical, so
                    // neither the all-translated view nor the raw bytes give the forward hash. Each changed word is
                    // then read either way (subsets of up to MIXED_MAX_WORDS words reverted to raw, fewest first).
                    let mut reverted = 0_usize;
                    if !raw && expected.is_some_and(|e| e != value) {
                        if let Some((viewed, _)) = view.as_ref() {
                            if let Some((hashed, back)) = mixed_reading(data, viewed, expected.unwrap(), previous.get(id)) {
                                value = hashed; reverted = back.len();
                                for index in &back { history.remove(index); }
                            }
                        }
                    }
                    let count = if raw { 0 } else { view.as_ref().map(|v| v.1).unwrap_or(0).saturating_sub(reverted) };
                    if raw { history.clear(); }
                    // integ8: the excluded-residue block's masked digest (was the main thread's sequential loop below)
                    let res = match residue.get(&(step, *id)) {
                        Some((zeroed, masked_expected)) if !address_bearing_ids.contains(id) =>
                            Some(residue_masked_reading(data, &ranges, low, high, previous.get(id), zeroed, masked_expected, &mut history)),
                        _ => None,
                    };
                    let mut slot = slots[index].lock().unwrap();
                    *slot.0 = value;
                    *slot.1 = count;
                    *slot.2 = history;
                    *slot.3 = if raw { 0 } else { half };
                    *slot.4 = raw;
                    *slot.5 = res;
                    drop(slot);
                    if dprof { item_t.lock().unwrap().push((*id, data.len(), t_view, t_item.elapsed().as_secs_f64())); }
                });
            }
        });
        drop(slots);
    }
    DigestOut { values, moved, histories, halves, raw_equal, residue_out, item_t: item_t.into_inner().unwrap(),
                workers_s: t0.elapsed().as_secs_f64() }
}
/// terrain-w-hash: the boundary's printed lines and comparisons (moved out of finish_step_host_hash unchanged);
/// returns the mismatched allocations in carried order.
fn digest_post(step: usize, blocks: &[(u64, usize)], out: &mut DigestOut,
               residue: &HashMap<(usize, u64), (Vec<(usize, usize)>, String)>, address_bearing_ids: &HashSet<u64>,
               references: Option<&References>) -> Result<Vec<u64>, String> {
        let mut mismatched = Vec::new();
        for ((((((id, len), value), translated), half), raw), res) in
                blocks.iter().zip(out.values.iter()).zip(out.moved.iter()).zip(out.halves.iter()).zip(out.raw_equal.iter())
                    .zip(out.residue_out.iter_mut()) {
            let address_bearing = address_bearing_ids.contains(id);
            if !quiet_hashes() { println!("{}", serde_json::json!({"event":"state_sha256","step":step,"allocation":id,
                                             "bytes":len,"sha256":value,"address_bearing":address_bearing,
                                             "translated_address_words":translated,"half_address_words":half,
                                             "raw_bytes_equal":raw})); }
            if address_bearing { continue; }
            if let Some((zeroed, masked_expected)) = residue.get(&(step, *id)) {
                // integ8: computed by the digest workers (residue_masked_reading), in the same reading order.
                let (masked, half_words) = res.take().ok_or("residue digest missing")??;
                if let Some(words) = half_words {
                    println!("{}", serde_json::json!({"event":"residue_half_words","step":step,"allocation":id,"words":words}));
                }
                let unmasked_equal = references.map(|r| r[&(step, *id)].1 == *value);
                println!("{}", serde_json::json!({"event":"excluded_residue","step":step,"allocation":id,
                    "ranges":zeroed.len(),"bytes":zeroed.iter().map(|r| r.1).sum::<usize>(),
                    "masked_sha256":masked,"masked_equal":masked == *masked_expected,"unmasked_equal":unmasked_equal}));
                // a residue block is compared like every other block: only against references (the Taylor check's
                // perturbed forward has none). Breakage: J2u Taylor rows refused at step 1 on allocation 394 (tH18).
                if references.is_some() && masked != *masked_expected { mismatched.push(*id); }
                continue;
            }
            if let Some(references) = references {
                let (expected_bytes, expected) = &references[&(step, *id)];
                if *expected_bytes != *len || expected != value { mismatched.push(*id); }
            }
        }
    Ok(mismatched)
}
/// AD_DIGEST_PROF=1 (diagnostic only): the boundary digest's stage walls and slowest blocks.
fn digest_prof_json(step: usize, item_t: &mut [(u64, usize, f64, f64)], t_dl: f64, t_workers: f64, total: f64,
                    extra: Option<serde_json::Value>) -> serde_json::Value {
    item_t.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
    let top: Vec<serde_json::Value> = item_t.iter().take(6).map(|x| serde_json::json!([x.0, x.1, (x.2 * 1e3).round() / 1e3, (x.3 * 1e3).round() / 1e3])).collect();
    let mut v = serde_json::json!({"event":"host_digest_prof","step":step,"blocks":item_t.len(),"download_s":t_dl,
        "workers_s":t_workers,"total_s":total,"residue_and_compare_s":total - t_workers,"slowest_id_bytes_view_s_item_s":top});
    if let (Some(serde_json::Value::Object(e)), serde_json::Value::Object(m)) = (extra, &mut v) { m.extend(e); }
    v
}

impl Execution {
    /// P1 sealer-output-steps: the live-address words a runner-built table (RUC, sea-ice, pointer table, RRFS copy
    /// plan) writes this step, kept like held pointers (held_pointer_writes) so the boundary's half-word rule knows
    /// them when a later 4-byte write in the same step leaves only one half. Breakage: the sealed 18-step hrrrv5
    /// plan, strict replay, step 10, allocation 142 word 93: the step-10 RUC table entry's high half under 4 bytes
    /// of acoustic scratch data (forward 0x7e573a48cf2f, replay 0x77b53a48cf2f).
    fn note_table_pointers(&mut self, destination: plan::Slice, data: &[u8]) {
        if destination.offset % 8 != 0 || self.recorded_bases.is_empty() { return; }
        for (index, chunk) in data.chunks_exact(8).enumerate() {
            let word = u64::from_le_bytes(chunk.try_into().unwrap());
            if word == 0 { continue; }
            let hit = self.allocations.values()
                .find(|a| a.address <= word && word < a.address + a.bytes as u64)
                .and_then(|a| self.recorded_bases.get(&a.id).map(|recorded| recorded + (word - a.address)));
            if let Some(forward) = hit {
                self.held_pointer_writes.entry(destination.allocation).or_default()
                    .insert(destination.offset / 8 + index, (word, forward));
            }
        }
    }

    /// G5: the cached bytes of a blob (window only).
    fn cached_blob(&self, value: &Blob) -> Option<std::sync::Arc<Vec<u8>>> {
        self.blob_cache.as_ref().and_then(|c| c.get(blob_sha(value))).cloned()
    }
    /// G5: a pinned staging buffer of at least `bytes` from the pool (best fit) or a new one; it is live (in
    /// `pinned`) until the next drain, which returns it to the pool.
    unsafe fn pinned_take(&mut self, bytes: usize) -> Result<*mut c_void, String> {
        let bytes = bytes.max(1);
        let pick = self.pinned_pool.iter().enumerate().filter(|(_, (_, n))| *n >= bytes).min_by_key(|(_, (_, n))| *n).map(|(i, _)| i);
        let host = match pick {
            Some(i) => self.pinned_pool.swap_remove(i).0,
            None => {
                let mut h = ptr::null_mut();
                checked("cuMemHostAlloc staging", (self.cuda.host_alloc)(&mut h, bytes, 0))?;
                self.pinned_size.insert(h as usize, bytes);
                STAGING_ALLOCS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                h
            },
        };
        self.pinned.push(host);
        Ok(host)
    }
    /// Free (or, for pool buffers, return) every live pinned buffer; call only after a synchronization.
    pub unsafe fn release_pinned(&mut self) -> Result<(), String> {
        for host in std::mem::take(&mut self.pinned) {
            match self.pinned_size.get(&(host as usize)) {
                Some(&n) => self.pinned_pool.push((host, n)),
                None => checked("cuMemFreeHost", (self.cuda.host_free)(host))?,
            }
        }
        Ok(())
    }
    /// G5: a held/upload blob from the window's cache when it holds it (the same bytes), else decoded here.
    fn command_blob(&self, base: &Path, value: &Blob) -> Result<Vec<u8>, String> {
        if let Some(b) = self.blob_cache.as_ref().and_then(|c| c.get(blob_sha(value))) { return Ok(b.as_ref().clone()); }
        blob(base, value)
    }
    /// outer2 (2026-11-07): a replay allocation placed at an address congruent to its recorded base modulo
    /// `replay_congruence()` (2 MiB, the driver's large-page granule; AD_BINDER_CONGRUENCE=off keeps the driver's
    /// pointer as before). THE BREAKAGE THIS PREVENTS: forward code that rounds a pointer up to an absolute
    /// boundary (a 64 KiB-aligned sub-buffer) inside a pooled scratch root. The recording's root 427 (604,672 B,
    /// recorded at 0x1000ba00000) held at step 2 a word rounded to recorded base + 0x30000. The CLI and the C-ABI dot
    /// test (fresh contexts) got the root at 0x1000ce00000, congruent by luck, and wrote base + 0x30000; inside a
    /// client process (the client's own small allocations first) the driver sub-allocated it at 0x1000d093a00, the same
    /// rounding gave base + 0x2c600, and HexAD's forward_check refused 'state hash mismatch at step 2, allocation
    /// 427' (outer2 runs 7 and 8 and diagnostic X1: the in-client and CLI dumps of 427 differ in exactly that word).
    /// With the shift the replay's low 21 address bits equal the recording's, so every absolute rounding up to
    /// 2 MiB lands where the recording's did. A driver pointer that is already congruent is kept; the rest are
    /// re-allocated with one granule of slack.
    unsafe fn allocate_congruent(&mut self, id: u64, bytes: usize, recorded: Option<u64>, on: Option<*mut c_void>)
                                 -> Result<u64, String> {
        let raw = |cuda: &Cuda, n: usize| -> Result<u64, String> {
            let mut a = 0;
            match on {
                Some(s) => checked("cuMemAllocAsync", (cuda.allocate_async)(&mut a, n, s))?,
                None => checked("cuMemAlloc", (cuda.allocate)(&mut a, n))?,
            }
            Ok(a)
        };
        let first = raw(&self.cuda, bytes)?;
        let m = replay_congruence();
        let Some(r) = recorded else { return Ok(first) };
        if m == 0 || first % m == r % m { return Ok(first); }
        match on {
            Some(s) => checked("cuMemFreeAsync congruence", (self.cuda.free_async)(first, s))?,
            None => checked("cuMemFree congruence", (self.cuda.free)(first))?,
        }
        let slack = bytes.checked_add(m as usize).ok_or("congruent allocation size overflow")?;
        let driver = raw(&self.cuda, slack)?;
        let shift = (r % m + m - driver % m) % m;
        self.raw_bases.insert(id, driver);
        Ok(driver + shift)
    }

    unsafe fn new(trace: Option<BufWriter<File>>) -> Result<Self, String> {
        let cuda = Cuda::load()?;
        checked("cuInit", (cuda.init)(0))?;
        let mut device = 0;
        checked("cuDeviceGet", (cuda.device_get)(&mut device, 0))?;
        let mut context = ptr::null_mut();
        // P1 (ad/phys): host waits sleep instead of spinning one core (CU_CTX_SCHED_BLOCKING_SYNC),
        // the same scheduling-only change as ad/e2e U8. AD_BINDER_QUICK=0 keeps the old spin wait.
        let flags = if quick_requested() { CTX_SCHED_BLOCKING_SYNC } else { 0 };
        checked("cuCtxCreate", (cuda.context_create)(&mut context, flags, device))?;
        let mut streams = HashMap::new();
        streams.insert(0, ptr::null_mut());
        Ok(Self { kernel_override: HashMap::new(), cuda, context, allocations: HashMap::new(), live_bytes: 0, live_peak: 0, host_state: host_state::HostState::default(), rk: None, streams, events: HashMap::new(),
                  modules: HashMap::new(), functions: HashMap::new(), cub_libraries: HashMap::new(),
                  thrust_libraries: HashMap::new(), cublas_libraries: HashMap::new(),
                  tempo_cache: tempo::Cache::default(), tempo_bindings: HashMap::new(),
                  digest_module: ptr::null_mut(), digest_function: ptr::null_mut(), digest_output: 0, digest_arena: None, digest_s: 0.0,
                  word_module: ptr::null_mut(), word_function: ptr::null_mut(), word_output: 0,
                  pinned: Vec::new(),
                  submitter: submit::Submitter::default(), trace, address_bearing: HashSet::new(),
                  recorded_bases: HashMap::new(), raw_bases: HashMap::new(), residue: HashMap::new(), pointer_history: HashMap::new(),
                  held_pointer_writes: HashMap::new(), captured: None, digest_defer: false, unchecked_digest_skip: false, history_stale: false, digests_skipped: 0, digest_job: None, digest_bg_s: 0.0, ad: None, blob_cache: None,
                  pinned_pool: Vec::new(), pinned_size: HashMap::new(),
                  lw_regroup: false, lw_regroup_test: 0, lw_live_hist: None })
    }

    fn log(&mut self, value: serde_json::Value) -> Result<(), String> {
        if let Some(trace) = &mut self.trace {
            serde_json::to_writer(&mut *trace, &value).map_err(|e| e.to_string())?;
            trace.write_all(b"\n").map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    fn stream(&self, id: u64) -> Result<*mut c_void, String> {
        self.streams.get(&id).copied().ok_or_else(|| format!("unknown stream {id}"))
    }
    fn event(&self, id: u64) -> Result<*mut c_void, String> {
        self.events.get(&id).copied().ok_or_else(|| format!("unknown event {id}"))
    }
    /// multidevice-runner: the live RK loop scope (split, stage, small step) that names one execution of a peer copy.
    pub(crate) fn peer_scope(&self) -> String {
        match &self.rk { Some(s) => format!("{:?}|{:?}|{:?}", s.split, s.stage, s.small_step), None => "-".into() }
    }
    fn address(&self, slice: plan::Slice, bytes: usize) -> Result<u64, String> {
        let a = self.allocations.get(&slice.allocation).ok_or_else(|| format!("unknown allocation {}", slice.allocation))?;
        if slice.offset >= a.bytes || bytes > a.bytes - slice.offset {
            return Err(format!("allocation {} offset out of range", slice.allocation));
        }
        a.address.checked_add(slice.offset as u64).ok_or_else(|| "device address overflow".into())
    }

    fn host_i64_values(&self, values: &[plan::Slice]) -> Result<Vec<i64>, String> {
        read_host_i64_values(&self.host_state, values)
    }

    /// Re-read each geometry word at the time of the library call. A shape
    /// observed while recording is never a substitute for current host state.
    fn library_dimensions(&self, values: &[plan::Slice], maximum: usize) -> Result<Vec<usize>, String> {
        self.host_i64_values(values)?.into_iter().map(|word| {
            let value = usize::try_from(word).map_err(|_| "negative live library dimension")?;
            if value == 0 || value > maximum { return Err("invalid live library dimension".into()); }
            Ok(value)
        }).collect()
    }

    unsafe fn bind_tempo(&mut self, stream: u64, binding: &str, destination: plan::Slice,
                         device: u64, data: Vec<u8>, step: usize) -> Result<(), String> {
        if binding.is_empty() || device != 0 || data.is_empty() {
            return Err("invalid TEMPO table binding".into());
        }
        let hit = self.tempo_cache.get(device, &data);
        let target = if let Some(target) = hit {
            self.address(target, data.len())?;
            target
        } else {
            if self.tempo_cache.clear_on_full() {
                // 24 h recording: the source's 64-entry clear frees every cached pointer table, so every binding made before it
                // points at memory the forecast may hand to the next table (pkg72m2 step 21: tempo_s21_q203043 lands on
                // tempo_s16_q163076's bytes in allocation 200). Bindings are step-scoped (named per step and recorded
                // sequence); dropping them here makes any later use of a dead one refuse ("unknown live TEMPO table
                // binding") instead of the overlap check refusing a table the forecast itself placed there.
                // Breakage: every 72-step replay stopped at step 21, "TEMPO target overlaps a live binding".
                let dropped = self.tempo_bindings.len();
                self.tempo_bindings.clear();
                self.log(serde_json::json!({"call":"tempo_cache_clear","step":step,"bindings_dropped":dropped}))?;
            }
            self.address(destination, data.len())?;
            let end = destination.offset.checked_add(data.len()).ok_or("TEMPO target overflow")?;
            for (name, (other, bytes)) in &self.tempo_bindings {
                let other_end = other.offset.checked_add(*bytes).ok_or("TEMPO target overflow")?;
                if name != binding && other.allocation == destination.allocation
                    && destination.offset < other_end && other.offset < end {
                    return Err("TEMPO target overlaps a live binding".into());
                }
            }
            self.tempo_cache.insert(device, data.clone(), destination)?;
            let address = self.address(destination, data.len())?;
            let stream = self.stream(stream)?;
            // adjoint-speed-2: from the pinned pool (returned at the next drain), not a fresh cuMemHostAlloc per command
            // (pkg18m2: ~170 cuMemHostAlloc/cuMemFreeHost pairs per step pass, both synchronizing). The same bytes are copied.
            let host = self.pinned_take(data.len())?;
            ptr::copy_nonoverlapping(data.as_ptr(), host.cast::<u8>(), data.len());
            checked("cuMemcpyHtoDAsync TEMPO pointer table", (self.cuda.upload_async)(address, host, data.len(), stream))?;
            // pk120: a TEMPO table the runner writes is a runner-built table like the RUC, sea-ice, pointer-table and RRFS
            // copy-plan ones: its live-address words go into held_pointer_writes, replacing whatever an earlier table in
            // the same words noted. THE BREAKAGE THIS PREVENTS: pkg120 (x1.40962) step 1, allocation 386's last 1,024
            // bytes hold the 42-row RRFS copy plan (op 3567) and then TEMPO's 20-entry table tempo_s1_q63063 (op 6356);
            // the boundary still translated those words by the copy plan's noted pairs, so the TEMPO pointers stayed
            // replay addresses and every strict replay refused "state hash mismatch at step 1, first 386".
            self.note_table_pointers(destination, &data);
            destination
        };
        self.tempo_bindings.insert(binding.to_owned(), (target, data.len()));
        self.log(serde_json::json!({"call":"live_tempo_table", "step":step,
            "binding":binding, "cache_hit":hit.is_some(), "entries":data.len() / 8}))
    }
    fn remaining(&self, slice: plan::Slice) -> Result<usize, String> {
        let allocation = self.allocations.get(&slice.allocation)
            .ok_or_else(|| format!("unknown allocation {}", slice.allocation))?;
        allocation.bytes.checked_sub(slice.offset).filter(|bytes| *bytes > 0)
            .ok_or_else(|| "workspace outside allocation".into())
    }
    unsafe fn cub_library(&mut self, base: &Path, library_path: &str, sha256: &str)
        -> Result<&native_cub::NativeCub, String> {
        // Keyed by the recorded pin, so the lookup (and its file hashing) runs once per library, not per launch.
        let recorded = path(base, library_path);
        let key = (recorded.to_string_lossy().into_owned(), sha256.to_owned());
        if !self.cub_libraries.contains_key(&key) {
            let (full_path, sha256) = resolve_pinned_library(&recorded, sha256)?;
            let library = native_cub::NativeCub::open(&full_path, &sha256)?;
            self.cub_libraries.insert(key.clone(), library);
        }
        Ok(&self.cub_libraries[&key])
    }
    unsafe fn thrust_library(&mut self, base: &Path, library_path: &str, sha256: &str)
        -> Result<&native_thrust::NativeThrust, String> {
        // Keyed by the recorded pin, so the lookup (and its file hashing) runs once per library, not per launch.
        let recorded = path(base, library_path);
        let key = (recorded.to_string_lossy().into_owned(), sha256.to_owned());
        if !self.thrust_libraries.contains_key(&key) {
            let (full_path, sha256) = resolve_pinned_library(&recorded, sha256)?;
            let library = native_thrust::NativeThrust::open(&full_path, &sha256)?;
            self.thrust_libraries.insert(key.clone(), library);
        }
        Ok(&self.thrust_libraries[&key])
    }
    unsafe fn cublas_library(&mut self, base: &Path, library_path: &str, sha256: &str)
        -> Result<&native_cublas::NativeCublas, String> {
        // Keyed by the recorded pin, so the lookup (and its file hashing) runs once per library, not per launch.
        let recorded = path(base, library_path);
        let key = (recorded.to_string_lossy().into_owned(), sha256.to_owned());
        if !self.cublas_libraries.contains_key(&key) {
            let (full_path, sha256) = resolve_pinned_library(&recorded, sha256)?;
            let library = native_cublas::NativeCublas::open(&full_path, &sha256)?;
            self.cublas_libraries.insert(key.clone(), library);
        }
        Ok(&self.cublas_libraries[&key])
    }
    unsafe fn prepare(&mut self, plan: &Plan, base: &Path) -> Result<(), String> {
        for host in &plan.host_allocations {
            if host.origin != "initial_state" { return Err("host bytes are not certified initial state".into()); }
            let data = blob(base, &host.initial)?;
            if data.len() != host.bytes { return Err(format!("wrong initial host length for {}", host.id)); }
            self.host_state.insert(host.id, data)?;
            self.log(serde_json::json!({"event":"host_initial_state", "id":host.id,"bytes":host.bytes}))?;
        }
        // SHA-256 runs on the device. Only its 32-byte result is read back;
        // the replay does not download a complete carried field each step.
        // The host driver's PTX JIT can lag the installed NVRTC version. Load
        // source-owned cubins for the home cards instead; fail closed elsewhere.
        let images: &[&[u8]] = &[
            include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/sha256-sm_120.cubin")),
            include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/sha256-sm_89.cubin")),
        ];
        let mut statuses = Vec::new();
        for image in images {
            let status = (self.cuda.module_load)(&mut self.digest_module, image.as_ptr().cast());
            if status == 0 { break; }
            statuses.push(status);
            self.digest_module = ptr::null_mut();
        }
        if self.digest_module.is_null() {
            return Err(format!("no compatible digest cubin, CUDA statuses {statuses:?}"));
        }
        checked("cuModuleGetFunction digest", (self.cuda.module_function)(
            &mut self.digest_function, self.digest_module, c"binder_sha256".as_ptr()))?;
        checked("cuMemAlloc digest", (self.cuda.allocate)(&mut self.digest_output, 32))?;
        for stream in &plan.streams {
            if self.streams.contains_key(&stream.id) { return Err(format!("duplicate stream {}", stream.id)); }
            let mut handle = ptr::null_mut();
            checked("cuStreamCreate", (self.cuda.stream_create)(&mut handle, stream.flags))?;
            self.streams.insert(stream.id, handle);
        }
        for event in &plan.events {
            if self.events.contains_key(&event.id) { return Err(format!("duplicate event {}", event.id)); }
            let mut handle = ptr::null_mut();
            checked("cuEventCreate", (self.cuda.event_create)(&mut handle, event_flags(event.flags)))?;
            self.events.insert(event.id, handle);
        }
        for module in &plan.modules {
            if self.modules.contains_key(&module.id) { return Err(format!("duplicate module {}", module.id)); }
            let image_path = path(base, &module.path);
            let image = match module.compression.as_deref() {
                None => fs::read(image_path).map_err(|e| e.to_string())?,
                Some("gzip") => gzip_bytes(&image_path, module.bytes.ok_or("missing uncompressed module size")?)?,
                Some(_) => return Err(format!("unsupported module compression for {}", module.id)),
            };
            verify_sha(&image, &module.sha256)?;
            let (image, format) = image::prepare(image)?;
            let image_kind = match format {
                image::Format::Elf { pad: 0 } => "captured".to_owned(),
                image::Format::Elf { pad } => format!("captured+pad{pad}"),
                image::Format::Ptx => "ptx+terminator".to_owned(),
            };
            let mut handle = ptr::null_mut();
            checked("cuModuleLoadData", (self.cuda.module_load)(&mut handle, image.as_ptr().cast()))?;
            self.modules.insert(module.id.clone(), handle);
            self.log(serde_json::json!({"call":"cuModuleLoadData","module":module.id,
                "image":image_kind,"captured_sha256":module.sha256,"loaded_sha256":digest(&image)}))?;
        }
        // P1 (ad/phys): module-scope device state the forward built before step 1 (see plan.rs). Zero-argument
        // launches in recorded order, synchronized before any allocation is uploaded or any step runs.
        let mut previous = 0_u64;
        for init in &plan.module_initializers {
            if init.recorded_sequence <= previous { return Err("module initializers out of recorded order".into()); }
            previous = init.recorded_sequence;
            let handle = *self.modules.get(&init.module)
                .ok_or_else(|| format!("unknown initializer module {}", init.module))?;
            let name = CString::new(init.function.as_str()).map_err(|e| e.to_string())?;
            let mut function = ptr::null_mut();
            checked("cuModuleGetFunction initializer", (self.cuda.module_function)(&mut function, handle, name.as_ptr()))?;
            checked("cuLaunchKernel initializer", (self.cuda.launch)(function, init.grid[0], init.grid[1], init.grid[2],
                init.block[0], init.block[1], init.block[2], init.shared_bytes, ptr::null_mut(),
                ptr::null_mut(), ptr::null_mut()))?;
            checked("cuCtxSynchronize initializer", (self.cuda.context_synchronize)())?;
            self.log(serde_json::json!({"call":"module_initializer","module":init.module,"function":init.function,
                                        "recorded_sequence":init.recorded_sequence}))?;
        }
        let recorded_early: HashMap<u64, u64> = plan.recorded_bases.iter().map(|[id, base]| (*id, *base)).collect();
        for allocation in &plan.allocations {
            if allocation.bytes == 0 || self.allocations.contains_key(&allocation.id) {
                return Err(format!("invalid or duplicate allocation {}", allocation.id));
            }
            let initial = match &allocation.initial {
                Some(data) => {
                    let data = blob(base, data)?;
                    if data.len() != allocation.bytes { return Err(format!("wrong initial length for {}", allocation.id)); }
                    Some(data)
                },
                None if allocation.overwritten_before_read &&
                    (plan.carried_views.is_some() || !plan.carried.contains(&allocation.id)) => None,
                None => return Err(format!("allocation {} has no certified initial state", allocation.id)),
            };
            let address = self.allocate_congruent(allocation.id, allocation.bytes, recorded_early.get(&allocation.id).copied(), None)?;
            self.allocations.insert(allocation.id, submit::Allocation { id: allocation.id, address, bytes: allocation.bytes });
            self.live_bytes += allocation.bytes; self.live_peak = self.live_peak.max(self.live_bytes);
            if let Some(data) = initial {
                checked("cuMemcpyHtoD", (self.cuda.upload)(address, data.as_ptr().cast(), data.len()))?;
            }
            self.log(serde_json::json!({"call":"cuMemAlloc","id":allocation.id,"bytes":allocation.bytes,"address":address}))?;
        }
        let recorded: HashMap<u64, u64> = plan.recorded_bases.iter().map(|[id, base]| (*id, *base)).collect();
        let mut relocated = 0_usize;
        for allocation in &plan.allocations {
            if allocation.relocations.is_empty() { continue; }
            let initial = blob(base, allocation.initial.as_ref().ok_or("relocation without initial bytes")?)?;
            let here = self.allocations.get(&allocation.id).ok_or("relocated allocation missing")?.address;
            for r in &allocation.relocations {
                let word = initial.get(r.byte_offset..r.byte_offset + 8).ok_or("relocation outside initial bytes")?;
                let value = u64::from_le_bytes(word.try_into().unwrap());
                let target_base = *recorded.get(&r.target.allocation).ok_or("relocation target has no recorded base")?;
                if r.byte_offset % 8 != 0 || value != r.recorded_value
                    || value != target_base + r.target.offset as u64 {
                    return Err(format!("relocation of allocation {} at {} does not match the recorded word",
                                       allocation.id, r.byte_offset));
                }
                let target = self.allocations.get(&r.target.allocation).ok_or("relocation target not allocated")?;
                if r.target.offset > target.bytes { return Err("relocation target offset outside allocation".into()); }
                let replay_word = target.address + r.target.offset as u64;
                let replay = replay_word.to_le_bytes();
                checked("cuMemcpyHtoD relocation", (self.cuda.upload)(here + r.byte_offset as u64, replay.as_ptr().cast(), 8))?;
                // F11: the relocated word is noted (replay word -> recorded word) like a word of a runner-built table
                // (note_table_pointers), so the first boundary reads it as the recorded address even after its target
                // is freed or another allocation takes the target's replay address (the S01b history rule). THE
                // BREAKAGE THIS PREVENTS: the 4090 plan 6696466e, once every step allocation was born zero, still
                // stopped at step 1 on allocations 131 and 152 (solve_region_scrub descriptor tables): words 4426
                // and 4687 still held the initial pointer to allocation 379, freed during step 1, so no live range
                // translated them; word 4682 held the initial pointer to an array whose replay address allocation
                // 413 took, so the live range translated it to 413's recorded base, not the forward's word.
                self.held_pointer_writes.entry(allocation.id).or_default().insert(r.byte_offset / 8, (replay_word, value));
                relocated += 1;
            }
        }
        self.recorded_bases = recorded;
        for r in &plan.excluded_residue {
            if r.ranges.is_empty() || r.reason.is_empty() || r.witness.is_empty() || decode_hex(&r.masked_sha256)?.len() != 32 {
                return Err(format!("excluded residue of allocation {} at step {} lacks ranges, reason, witness or hash",
                                   r.allocation, r.step));
            }
            let ranges = r.ranges.iter().map(|x| (x.offset, x.bytes)).collect();
            if self.residue.insert((r.step, r.allocation), (ranges, r.masked_sha256.clone())).is_some() {
                return Err(format!("duplicate excluded residue for step {} allocation {}", r.step, r.allocation));
            }
        }
        self.log(serde_json::json!({"event":"initial_relocations","words":relocated}))?;
        if plan.carried_views.is_none() {
            for id in &plan.carried {
                if !self.allocations.contains_key(id) { return Err(format!("unknown carried allocation {id}")); }
            }
        }
        Ok(())
    }

    unsafe fn function(&mut self, module: &str, name: &str) -> Result<*mut c_void, String> {
        let key = (module.to_owned(), name.to_owned());
        if let Some(function) = self.functions.get(&key) { return Ok(*function); }
        let handle = *self.modules.get(module).ok_or_else(|| format!("unknown module {module}"))?;
        let name_c = CString::new(name).map_err(|e| e.to_string())?;
        let mut function = ptr::null_mut();
        checked("cuModuleGetFunction", (self.cuda.module_function)(&mut function, handle, name_c.as_ptr()))?;
        self.functions.insert(key, function);
        Ok(function)
    }

    fn argument(&self, value: &Argument, step: usize, frame: Option<&replay_frame::ReplayFrame>) -> Result<submit::Argument, String> {
        let ptr = |s: plan::Slice| submit::Slice { allocation: s.allocation, offset: s.offset };
        match value {
            Argument::Device { target } => Ok(submit::Argument::Device(ptr(*target))),
            Argument::HostScalar { source, cast } => {
                let allocation = self.host_state.get(source.allocation)?;
                let end = source.offset.checked_add(cast.source_bytes()).ok_or("host scalar extent overflow")?;
                let bytes = allocation.get(source.offset..end).ok_or("live host scalar outside allocation")?;
                Ok(submit::Argument::Bytes(host_scalar::evaluate(*cast, bytes)?))
            },
            Argument::CupyArray { target, shape, strides, item_bytes } => {
                let bytes = carray::encode(*target, &self.host_i64_values(shape)?, &self.host_i64_values(strides)?, *item_bytes,
                                           &self.allocations)?;
                match target {
                    Some(target) => Ok(submit::Argument::Patched { bytes,
                        pointers: vec![submit::PointerPatch { byte_offset: 0, target: ptr(*target) }] }),
                    None => Ok(submit::Argument::Bytes(bytes)),
                }
            },
            Argument::CupyCopyIndexer { shape, .. } => Ok(submit::Argument::Bytes(
                carray::encode_copy_indexer(&self.host_i64_values(shape)?)?)),
            Argument::TempoTable { binding } => {
                let (target, bytes) = self.tempo_bindings.get(binding).ok_or("unknown live TEMPO table binding")?;
                self.address(*target, *bytes)?;
                Ok(submit::Argument::Device(ptr(*target)))
            },
            Argument::ModuleGlobal { module, symbol, offset, expected_bytes } => {
                let handle = *self.modules.get(module).ok_or_else(|| format!("unknown module {module}"))?;
                let name = CString::new(symbol.as_str()).map_err(|e| e.to_string())?;
                let (mut address, mut bytes) = (0_u64, 0_usize);
                // The symbol lookup is for the loaded replay image, never the
                // recorded device address. Its contents remain census inputs.
                unsafe { checked("cuModuleGetGlobal_v2", (self.cuda.module_global)(
                    &mut address, &mut bytes, handle, name.as_ptr()))?; }
                if address == 0 || bytes != *expected_bytes || *offset >= bytes {
                    return Err(format!("invalid module global {module}:{symbol} size or offset"));
                }
                let pointer = address.checked_add(u64::try_from(*offset).map_err(|_| "global offset overflow")?)
                    .ok_or("global address overflow")?;
                Ok(submit::Argument::Bytes(pointer.to_le_bytes().to_vec()))
            },
            Argument::Bytes { bytes_hex, .. } => Ok(submit::Argument::Bytes(decode_hex(bytes_hex)?)),
            Argument::Patched { bytes_hex, pointers, .. } => Ok(submit::Argument::Patched {
                bytes: decode_hex(bytes_hex)?,
                pointers: pointers.iter().map(|p| submit::PointerPatch { byte_offset: p.byte_offset, target: ptr(p.target) }).collect() }),
            Argument::StepInteger { scale, bias, width } => {
                let value = scale.checked_mul(i64::try_from(step).map_err(|_| "step overflow")?)
                    .and_then(|x| x.checked_add(*bias)).ok_or("integer frame overflow")?;
                let width = usize::from(*width);
                if ![1, 2, 4, 8].contains(&width) { return Err("invalid integer frame width".into()); }
                if width < 8 {
                    let bits = width * 8;
                    if value < -(1_i64 << (bits - 1)) || value >= (1_i64 << (bits - 1)) {
                        return Err("integer frame does not fit".into());
                    }
                }
                Ok(submit::Argument::Bytes(value.to_le_bytes()[..width].to_vec()))
            },
            Argument::StepSeconds { start, interval } => {
                if !start.is_finite() || !interval.is_finite() || step as u64 > (1_u64 << 53) {
                    return Err("invalid time frame".into());
                }
                let time = *start + (step as f64 * *interval);
                if !time.is_finite() { return Err("time frame overflow".into()); }
                Ok(submit::Argument::Bytes(time.to_le_bytes().to_vec()))
            },
            Argument::PhysicsFrame { name } => Ok(submit::Argument::Bytes(
                frame.ok_or("physics scalar without live frame")?.scalar(name)?)),
            Argument::RkFrame { name } => Ok(submit::Argument::Bytes(
                self.rk.as_ref().ok_or("RK scalar without live schedule")?.scalar(name)?)),
        }
    }

    unsafe fn command(&mut self, op: &Operation, step: usize, base: &Path,
                      frame: &mut Option<replay_frame::ReplayFrame>) -> Result<(), String> {
        let mut ad_launch: Option<(String, Vec<submit::Argument>, [u32; 3], [u32; 3], u32)> = None;
        if let Some(ad) = self.ad.as_mut() {
            if let Operation::HeldUploadRanges { origin, .. } = op { ad.count(format!("held_upload_ranges:{origin}")); }
        }
        let ptr = |s: plan::Slice| submit::Slice { allocation: s.allocation, offset: s.offset };
        let native = match op {
            Operation::Kernel { stream, module, function, grid, block, shared_bytes, arguments } => {
                let handle = match self.kernel_override.get(function) { Some(f) => *f, None => self.function(module, function)? };
                let arguments = arguments.iter().map(|a| self.argument(a, step, frame.as_ref())).collect::<Result<Vec<_>,_>>()?;
                if self.ad.is_some() { ad_launch = Some((function.clone(), arguments.clone(), *grid, *block, *shared_bytes)); }
                // pk120 diagnostic (AD_BINDER_TRACE_KERNELS=1 with --trace): name each kernel before it is submitted, so a
                // native submit that fails (pkg120 step 1, CUDA 700/717 after the 42-row copy plan) names its kernel.
                if std::env::var("AD_BINDER_TRACE_KERNELS").as_deref() == Ok("1") {
                    self.log(serde_json::json!({"call":"kernel","function":function,"step":step,"grid":grid,"block":block}))?;
                }
                Some((*stream, submit::Command::Kernel { function: handle, grid: *grid, block: *block,
                                                        shared_bytes: *shared_bytes, arguments }))
            },
            Operation::VinterpOzone { stream, module, pmid, pin, ozone, starts, upper, out,
                                      pmid_shape, pin_size, ozone_shape } => {
                let meta = self.host_i64_values(&[pmid_shape[0], pmid_shape[1], *pin_size,
                                                   ozone_shape[0], ozone_shape[1]])?;
                if let Some(live) = vinterp::prepare([meta[0], meta[1]], meta[2], [meta[3], meta[4]])? {
                    let buffers = [*pmid, *pin, *ozone, *starts, *upper, *out];
                    // Check the complete set before submitting the first level.
                    for (buffer, bytes) in buffers.iter().zip(live.bytes) { self.address(*buffer, bytes)?; }
                    let function = self.function(module, "mpas_vinterp_level_chain")?;
                    let stream_handle = self.stream(*stream)?;
                    let driver = submit::NativeDriver { launch: self.cuda.launch, copy: self.cuda.copy, clear: self.cuda.clear };
                    let allocations: Vec<_> = self.allocations.values().copied().collect();
                    for k in (0..live.nl).rev() {
                        let mut arguments: Vec<_> = live.scalar_bytes(k)?.into_iter()
                            .map(|bytes| submit::Argument::Bytes(bytes.to_vec())).collect();
                        arguments.extend(buffers.iter().map(|buffer| submit::Argument::Device(ptr(*buffer))));
                        let command = submit::Command::Kernel { function, grid: live.grid, block: [128, 1, 1],
                                                               shared_bytes: 0, arguments };
                        self.submitter.submit(&driver, hex_step_validate, hex_step_submit, &allocations,
                                              &[command], &[], stream_handle)
                            .map_err(|e| format!("vinterp native submit: {e:?}"))?;
                    }
                    self.log(serde_json::json!({"call":"live_vinterp_levels","step":step,"stream":stream,
                        "nc":live.nc,"nl":live.nl,"nz":live.nz,"grid":live.grid}))?;
                }
                None
            },
            Operation::RadiationUnpack { stream, module, rows } => {
                let live_view = |view: &plan::RadiationView| -> Result<radiation_unpack::View, String> {
                    Ok(radiation_unpack::View { target:view.target, shape:self.host_i64_values(&view.shape)?,
                                                strides:self.host_i64_values(&view.strides)? })
                };
                let mut live_rows = Vec::with_capacity(rows.len());
                for row in rows {
                    let dst = live_view(&row.dst)?;
                    let src = row.src.as_ref().map(&live_view).transpose()?;
                    let fill = match row.fill_f32 {
                        None if src.is_some() => 0,
                        Some(fill) if src.is_none() => {
                            let storage = self.host_state.get(fill.allocation)?;
                            let end = fill.offset.checked_add(4).ok_or("radiation fill offset overflow")?;
                            let data = storage.get(fill.offset..end).ok_or("radiation fill outside allocation")?;
                            u32::from_ne_bytes(data.try_into().map_err(|_| "radiation fill size")?)
                        },
                        _ => return Err("radiation unpack copy/fill selection is invalid".into()),
                    };
                    live_rows.push(radiation_unpack::Row { dst, src, fill });
                }
                if let Some(live) = radiation_unpack::prepare(&live_rows, &self.allocations)? {
                    let function = self.function(module, "rrfs_radiation_unpack")?;
                    Some((*stream, submit::Command::Kernel { function, grid:live.grid, block:live.block,
                                                            shared_bytes:0, arguments:live.arguments }))
                } else { None }
            },
            Operation::TransportCopy { stream, module, rows } => {
                let live_view = |view: &plan::RadiationView| -> Result<transport_copy::View, String> {
                    Ok(transport_copy::View { target:view.target, shape:self.host_i64_values(&view.shape)?,
                                              strides:self.host_i64_values(&view.strides)? })
                };
                let live_rows: Vec<_> = rows.iter().map(|r| Ok(transport_copy::Row {
                    dst:live_view(&r.dst)?, src:live_view(&r.src)? })).collect::<Result<_,String>>()?;
                let live = transport_copy::prepare(&live_rows, &self.allocations)?;
                let function = self.function(module, "rrfs_post_transport_copy")?;
                Some((*stream, submit::Command::Kernel { function, grid:live.grid, block:live.block,
                                                        shared_bytes:0, arguments:live.arguments }))
            },
            Operation::AddInplace { stream, module, target, increment, target_shape, increment_shape } => {
                let t = self.host_i64_values(target_shape)?;
                let i = self.host_i64_values(increment_shape)?;
                let live = add_inplace::prepare([t[0],t[1]], [i[0],i[1]])?;
                self.address(*target, live.bytes)?;
                self.address(*increment, live.bytes)?;
                let function = self.function(module, "add_inplace_f32")?;
                let arguments = vec![submit::Argument::Bytes(live.nlevels.to_le_bytes().to_vec()),
                    submit::Argument::Bytes(live.nowners.to_le_bytes().to_vec()),
                    submit::Argument::Device(ptr(*increment)), submit::Argument::Device(ptr(*target))];
                Some((*stream, submit::Command::Kernel { function, grid:live.grid, block:live.block,
                                                        shared_bytes:0, arguments }))
            },
            Operation::DeviceCopy { stream, destination, source, bytes } => Some((*stream,
                submit::Command::DeviceCopy { destination: ptr(*destination), source: ptr(*source), bytes: *bytes })),
            // multidevice-runner (peer.rs): the cross-rank halo copy, split into its two halves. The send waits for the
            // producers of its bytes (context sync), the receive writes them before any later command of this rank runs.
            Operation::PeerSend { channel, source, bytes, .. } => {
                let p = peer::require("peer_send")?;
                let key = peer::key(step, *channel, &self.peer_scope());
                let address = self.address(*source, *bytes)?;
                // a recompute of the base trajectory re-sends bytes its peer already holds: no sync, no download
                if p.has_forward(&key) { p.note_resend(); } else {
                    dev::sync(&self.cuda)?;
                    let data = dev::download(&self.cuda, address, *bytes)?;
                    p.put_forward(&key, &data)?;
                }
                None
            },
            Operation::PeerRecv { channel, destination, bytes, .. } => {
                let p = peer::require("peer_recv")?;
                let address = self.address(*destination, *bytes)?;
                let data = p.get_forward(&peer::key(step, *channel, &self.peer_scope()), *bytes)?;
                dev::sync(&self.cuda)?;
                dev::upload(&self.cuda, address, &data)?;
                dev::sync(&self.cuda)?;
                None
            },
            Operation::ModuleGlobalCopy { stream, module, symbol, offset, source, bytes } => {
                // S14: stream-ordered like Fill16 (a direct driver call on the command's stream, in plan order).
                let handle = *self.modules.get(module).ok_or_else(|| format!("module_global_copy: unknown module {module}"))?;
                let name = CString::new(symbol.as_str()).map_err(|e| e.to_string())?;
                let (mut address, mut size) = (0_u64, 0_usize);
                unsafe { checked("cuModuleGetGlobal_v2", (self.cuda.module_global)(&mut address, &mut size, handle, name.as_ptr()))?; }
                if address == 0 || *bytes == 0 || offset.checked_add(*bytes).map_or(true, |end| end > size) {
                    return Err(format!("module_global_copy {module}:{symbol}: {bytes} bytes at offset {offset} exceed the symbol ({size} bytes)"));
                }
                let src = self.address(*source, *bytes)?;
                checked("cuMemcpyDtoDAsync_v2", (self.cuda.copy)(address + *offset as u64, src, *bytes, self.stream(*stream)?))?;
                self.log(serde_json::json!({"call":"cuMemcpyDtoDAsync_v2","step":step,"module_global":symbol,"bytes":bytes}))?;
                None
            },
            Operation::Clear { stream, destination, bytes, value } => Some((*stream,
                submit::Command::Clear { destination: ptr(*destination), bytes: *bytes, value: *value })),
            Operation::Fill16 { stream, destination, count, value } => {
                let bytes = count.checked_mul(2).ok_or("fill16 size overflow")?;
                if bytes == 0 { return Err("empty fill16".into()); }
                let target = self.address(*destination, bytes)?;
                if target % 2 != 0 { return Err("unaligned fill16".into()); }
                checked("cuMemsetD16Async", (self.cuda.fill16)(target, *value, *count, self.stream(*stream)?))?;
                self.log(serde_json::json!({"call":"cuMemsetD16Async","step":step,"count":count}))?;
                None
            },
            Operation::Fill32 { stream, destination, count, value } => {
                let bytes = count.checked_mul(4).ok_or("fill32 size overflow")?;
                if bytes == 0 { return Err("empty fill32".into()); }
                let target = self.address(*destination, bytes)?;
                if target % 4 != 0 { return Err("unaligned fill32".into()); }
                checked("cuMemsetD32Async", (self.cuda.fill32)(target, *value, *count, self.stream(*stream)?))?;
                self.log(serde_json::json!({"call":"cuMemsetD32Async","step":step,"count":count}))?;
                None
            },
            Operation::Allocate { stream, id, bytes, asynchronous, overwritten_before_read } => {
                if *bytes == 0 || !overwritten_before_read || self.allocations.contains_key(id) {
                    return Err(format!("invalid scratch allocation {id}"));
                }
                let on = if *asynchronous { Some(self.stream(*stream)?) } else { None };
                let recorded = self.recorded_bases.get(id).copied();
                let address = self.allocate_congruent(*id, *bytes, recorded, on)?;
                self.allocations.insert(*id, submit::Allocation { id: *id, address, bytes: *bytes });
                self.live_bytes += *bytes; self.live_peak = self.live_peak.max(self.live_bytes);
                // F11: an allocation made inside a step is born zero in the replay, as the recorder makes it born zero
                // in the forward (record/native/driver_trace_cupti_held.c, trace_born_zero) and as the window already
                // does in every pass (window.rs, G2). A plain allocation's fill is waited for before the next command,
                // as the recorder waits; a stream-ordered one is filled on its own stream. THE BREAKAGE THIS PREVENTS:
                // a 14-step 12 km plan recorded by the release's own recorder (RTX 4090, plan 6696466e) sealed and
                // then failed its strict replay at once, "state hash mismatch at step 1, 8 allocations, first 131":
                // allocations 450, 564, 1115 and 1118 are [59, 2891] arrays whose first use writes only [59, 2890]
                // (682,040 of 682,496 bytes), so their last 456 bytes are never written; the forward's were zero
                // (born zero), the replay's held whatever the driver handed back (the same stop on Blackwell, lane
                // blv611, plan 4cd097dc, 4 allocations).
                let fill_on = match on { Some(s) => s, None => self.stream(*stream)? };
                checked("cuMemsetD8Async born zero", (self.cuda.clear)(address, 0, *bytes, fill_on))?;
                if on.is_none() { checked("cuStreamSynchronize born zero", (self.cuda.stream_synchronize)(fill_on))?; }
                if std::env::var("AD_BINDER_DIAGNOSTIC").as_deref() == Ok("1") && std::env::var("AD_BINDER_TRACE_ALLOCS").is_ok() {
                    println!("{}", serde_json::json!({"event":"trace_allocate","step":step,"id":id,"bytes":bytes,"address":format!("{address:#x}"),"recorded":self.recorded_bases.get(id).map(|r| format!("{r:#x}"))}));
                }
                self.log(serde_json::json!({"call":if *asynchronous {"cuMemAllocAsync"} else {"cuMemAlloc"},"step":step,"id":id,"bytes":bytes,"address":address}))?;
                None
            },
            Operation::Free { stream, id, asynchronous } => {
                if self.tempo_cache.owns_allocation(*id)
                    || self.tempo_bindings.values().any(|(target, _)| target.allocation == *id) {
                    return Err("cannot free a live TEMPO table".into());
                }
                let allocation = self.allocations.get(id).ok_or_else(|| format!("unknown free allocation {id}"))?;
                if std::env::var("AD_BINDER_DIAGNOSTIC").as_deref() == Ok("1") && std::env::var("AD_BINDER_TRACE_ALLOCS").is_ok() {
                    println!("{}", serde_json::json!({"event":"trace_free","step":step,"id":id,"address":format!("{:#x}", allocation.address),"recorded":self.recorded_bases.get(id).is_some()}));
                }
                let driver = self.raw_bases.get(id).copied().unwrap_or(allocation.address);
                if *asynchronous {
                    checked("cuMemFreeAsync", (self.cuda.free_async)(driver, self.stream(*stream)?))?;
                } else { checked("cuMemFree", (self.cuda.free)(driver))?; }
                self.raw_bases.remove(id);
                if let Some(a) = self.allocations.remove(id) { self.live_bytes = self.live_bytes.saturating_sub(a.bytes); }
                self.log(serde_json::json!({"call":if *asynchronous {"cuMemFreeAsync"} else {"cuMemFree"},"step":step,"id":id}))?;
                None
            },
            Operation::FreeNull { stream, asynchronous } => {
                if *asynchronous {
                    checked("cuMemFreeAsync null", (self.cuda.free_async)(0, self.stream(*stream)?))?;
                } else { checked("cuMemFree null", (self.cuda.free)(0))?; }
                self.log(serde_json::json!({"call":if *asynchronous {"cuMemFreeAsync"} else {"cuMemFree"},
                                            "step":step,"null":true}))?;
                None
            },
            Operation::Upload { stream, destination, data, pointers, .. } if self.cached_blob(data).is_some() => {
                // G5 (window only): straight from the decoded cache into a pooled pinned buffer, patched there.
                let src = self.cached_blob(data).unwrap();
                if src.is_empty() { return Err("empty upload".into()); }
                let n = src.len();
                let target = self.address(*destination, n)?;
                let stream = self.stream(*stream)?;
                let host = self.pinned_take(n)?;
                ptr::copy_nonoverlapping(src.as_ptr(), host.cast::<u8>(), n);
                patch_device_addresses(std::slice::from_raw_parts_mut(host.cast::<u8>(), n), pointers, &self.allocations)?;
                checked("cuMemcpyHtoDAsync", (self.cuda.upload_async)(target, host, n, stream))?;
                self.log(serde_json::json!({"call":"cuMemcpyHtoDAsync","step":step,"bytes":n}))?;
                None
            },
            Operation::Upload { stream, destination, data, pointers, .. } => {
                let mut data = self.command_blob(base, data)?;
                if data.is_empty() { return Err("empty upload".into()); }
                patch_device_addresses(&mut data, pointers, &self.allocations)?;
                let target = self.address(*destination, data.len())?;
                let stream = self.stream(*stream)?;
                // adjoint-speed-2: from the pinned pool (returned at the next drain), not a fresh cuMemHostAlloc per command
                // (pkg18m2: ~170 cuMemHostAlloc/cuMemFreeHost pairs per step pass, both synchronizing). The same bytes are copied.
                let host = self.pinned_take(data.len())?;
                ptr::copy_nonoverlapping(data.as_ptr(), host.cast::<u8>(), data.len());
                checked("cuMemcpyHtoDAsync", (self.cuda.upload_async)(target, host, data.len(), stream))?;
                self.log(serde_json::json!({"call":"cuMemcpyHtoDAsync","step":step,"bytes":data.len()}))?;
                None
            },
            Operation::HeldUploadRanges { stream, data, ranges, pointers, .. } if self.cached_blob(data).is_some() => {
                // G5 (window only): the same checks and patches as below, on a pooled pinned copy of the cached bytes.
                let src = self.cached_blob(data).unwrap();
                for patch in pointers {
                    let word = src.get(patch.byte_offset..patch.byte_offset + 8).ok_or("held pointer outside its blob")?;
                    let value = u64::from_le_bytes(word.try_into().unwrap());
                    let recorded = *self.recorded_bases.get(&patch.target.allocation)
                        .ok_or("held pointer target has no recorded base")?;
                    if value != recorded + patch.target.offset as u64 {
                        return Err(format!("held pointer at blob offset {} does not match the recorded word", patch.byte_offset));
                    }
                }
                let forward_words: Vec<(usize, u64)> = pointers.iter().map(|patch| (patch.byte_offset,
                    u64::from_le_bytes(src[patch.byte_offset..patch.byte_offset + 8].try_into().unwrap()))).collect();
                let n = src.len();
                let host = self.pinned_take(n.max(1))?;
                ptr::copy_nonoverlapping(src.as_ptr(), host.cast::<u8>(), n);
                let view = std::slice::from_raw_parts_mut(host.cast::<u8>(), n);
                patch_device_addresses(view, pointers, &self.allocations)?;
                for (offset, forward) in forward_words {
                    let replay = u64::from_le_bytes(view[offset..offset + 8].try_into().unwrap());
                    for range in ranges.iter().filter(|r| r.source_offset <= offset && offset + 8 <= r.source_offset + r.bytes) {
                        let at = range.destination.offset + (offset - range.source_offset);
                        if at % 8 == 0 {
                            self.held_pointer_writes.entry(range.destination.allocation).or_default()
                                .insert(at / 8, (replay, forward));
                        }
                    }
                }
                let stream = self.stream(*stream)?;
                for range in ranges {
                    let end = range.source_offset.checked_add(range.bytes).ok_or("held range overflow")?;
                    if end > n { return Err("held range outside its blob".into()); }
                    let target = self.address(range.destination, range.bytes)?;
                    checked("cuMemcpyHtoDAsync held", (self.cuda.upload_async)(target,
                        host.cast::<u8>().add(range.source_offset).cast(), range.bytes, stream))?;
                }
                self.log(serde_json::json!({"call":"held_upload_ranges","step":step,"ranges":ranges.len(),"bytes":n}))?;
                None
            },
            Operation::HeldUploadRanges { stream, data, ranges, pointers, .. } => {
                let mut data = self.command_blob(base, data)?;
                for patch in pointers {
                    let word = data.get(patch.byte_offset..patch.byte_offset + 8).ok_or("held pointer outside its blob")?;
                    let value = u64::from_le_bytes(word.try_into().unwrap());
                    let recorded = *self.recorded_bases.get(&patch.target.allocation)
                        .ok_or("held pointer target has no recorded base")?;
                    if value != recorded + patch.target.offset as u64 {
                        return Err(format!("held pointer at blob offset {} does not match the recorded word", patch.byte_offset));
                    }
                }
                let forward_words: Vec<(usize, u64)> = pointers.iter().map(|patch| (patch.byte_offset,
                    u64::from_le_bytes(data[patch.byte_offset..patch.byte_offset + 8].try_into().unwrap()))).collect();
                patch_device_addresses(&mut data, pointers, &self.allocations)?;
                for (offset, forward) in forward_words {
                    let replay = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
                    for range in ranges.iter().filter(|r| r.source_offset <= offset && offset + 8 <= r.source_offset + r.bytes) {
                        let at = range.destination.offset + (offset - range.source_offset);
                        if at % 8 == 0 {
                            self.held_pointer_writes.entry(range.destination.allocation).or_default()
                                .insert(at / 8, (replay, forward));
                        }
                    }
                }
                let stream = self.stream(*stream)?;
                // adjoint-speed-2: from the pinned pool (returned at the next drain), not a fresh cuMemHostAlloc per command
                // (pkg18m2: ~170 cuMemHostAlloc/cuMemFreeHost pairs per step pass, both synchronizing). The same bytes are copied.
                let host = self.pinned_take(data.len().max(1))?;
                ptr::copy_nonoverlapping(data.as_ptr(), host.cast::<u8>(), data.len());
                for range in ranges {
                    let end = range.source_offset.checked_add(range.bytes).ok_or("held range overflow")?;
                    if end > data.len() { return Err("held range outside its blob".into()); }
                    let target = self.address(range.destination, range.bytes)?;
                    checked("cuMemcpyHtoDAsync held", (self.cuda.upload_async)(target,
                        host.cast::<u8>().add(range.source_offset).cast(), range.bytes, stream))?;
                }
                self.log(serde_json::json!({"call":"held_upload_ranges","step":step,"ranges":ranges.len(),"bytes":data.len()}))?;
                None
            },
            Operation::FrameUpload { stream, destination, name } => {
                let data = frame.as_ref().ok_or("frame upload without live frame")?.scalar(name)?;
                if data.is_empty() { return Err("empty frame upload".into()); }
                let target = self.address(*destination, data.len())?;
                let stream = self.stream(*stream)?;
                // adjoint-speed-2: from the pinned pool (returned at the next drain), not a fresh cuMemHostAlloc per command
                // (pkg18m2: ~170 cuMemHostAlloc/cuMemFreeHost pairs per step pass, both synchronizing). The same bytes are copied.
                let host = self.pinned_take(data.len())?;
                ptr::copy_nonoverlapping(data.as_ptr(), host.cast::<u8>(), data.len());
                checked("cuMemcpyHtoDAsync frame", (self.cuda.upload_async)(target, host, data.len(), stream))?;
                self.log(serde_json::json!({"call":"cuMemcpyHtoDAsync", "source":"live_frame", "name":name, "step":step, "bytes":data.len()}))?;
                None
            },
            Operation::HostUpload { stream, destination, view, order } => {
                let data = self.host_state.pack(view.allocation, &view.view(), order.order())?;
                if data.is_empty() { return Err("empty live host upload".into()); }
                let target = self.address(*destination, data.len())?;
                let stream = self.stream(*stream)?;
                // adjoint-speed-2: from the pinned pool (returned at the next drain), not a fresh cuMemHostAlloc per command
                // (pkg18m2: ~170 cuMemHostAlloc/cuMemFreeHost pairs per step pass, both synchronizing). The same bytes are copied.
                let host = self.pinned_take(data.len())?;
                ptr::copy_nonoverlapping(data.as_ptr(), host.cast::<u8>(), data.len());
                checked("cuMemcpyHtoDAsync live host", (self.cuda.upload_async)(target, host, data.len(), stream))?;
                self.log(serde_json::json!({"call":"cuMemcpyHtoDAsync", "source":"live_host_state",
                                             "allocation":view.allocation,"step":step,"bytes":data.len()}))?;
                None
            },
            Operation::HostPublish { stream, source, view, order } => {
                let size = self.host_state.get(view.allocation)?.len();
                let bytes = host_array::packed_size(size, &view.view()).map_err(str::to_owned)?;
                if bytes == 0 { return Err("empty live host publication".into()); }
                let source = self.address(*source, bytes)?;
                let stream = self.stream(*stream)?;
                // adjoint-speed-2: from the pinned pool (returned at the next drain), not a fresh cuMemHostAlloc per command
                // (pkg18m2: ~170 cuMemHostAlloc/cuMemFreeHost pairs per step pass, both synchronizing). The same bytes are copied.
                let host = self.pinned_take(bytes)?;
                checked("cuMemcpyDtoHAsync host publication", (self.cuda.download_async)(host, source, bytes, stream))?;
                checked("cuStreamSynchronize host publication", (self.cuda.stream_synchronize)(stream))?;
                self.host_state.publish(view.allocation, &view.view(), order.order(),
                    std::slice::from_raw_parts(host.cast::<u8>(), bytes))?;
                // host-leftovers-2: the pooled buffer stays live (in `pinned`) and goes back to the pool at the next
                // drain. Freeing it here (cuMemFreeHost, which synchronizes) shrank the pool, so later commands paid a
                // fresh cuMemHostAlloc: cuda-graph-capture item E, 206 cuMemHostAlloc per TL step (0.39 s). Same bytes.
                self.log(serde_json::json!({"call":"cuMemcpyDtoHAsync", "destination":"live_host_state",
                                             "allocation":view.allocation,"step":step,"bytes":bytes}))?;
                None
            },
            Operation::PointerTableUpload { stream, destination, rule, entries } => {
                if !matches!(rule.as_str(), "resident_v842_species" | "cupy_stack_same_size"
                        // Continuation 17:40Z: [int(a.data.ptr) for a in arrays] as uint64, in order:
                        // cuda_physics_prep_v841.py:575-587 _device_pointer_table; rrfs/driver_rrfs.py:2657.
                        | "cuda_physics_prep_device_pointer_table" | "rrfs_stage_sources_rate_table") {
                    return Err("unproved physics pointer table rule".into());
                }
                let data = live_pointer_table(entries, &self.allocations)?;
                let target = self.address(*destination, data.len())?;
                let stream = self.stream(*stream)?;
                // adjoint-speed-2: from the pinned pool (returned at the next drain), not a fresh cuMemHostAlloc per command
                // (pkg18m2: ~170 cuMemHostAlloc/cuMemFreeHost pairs per step pass, both synchronizing). The same bytes are copied.
                let host = self.pinned_take(data.len())?;
                ptr::copy_nonoverlapping(data.as_ptr(), host.cast::<u8>(), data.len());
                checked("cuMemcpyHtoDAsync pointer table", (self.cuda.upload_async)(target, host, data.len(), stream))?;
                self.note_table_pointers(*destination, &data);
                self.log(serde_json::json!({"call":"cuMemcpyHtoDAsync", "source":"live_pointer_table",
                                            "rule":rule, "step":step, "entries":entries.len()}))?;
                None
            },
            Operation::TempoPointerTableBind { stream, binding, destination, entries,
                                               expected_shape, device, optional } => {
                let data = tempo::pointer_table(entries, expected_shape.as_deref(), *device, *optional,
                                                &self.allocations)?;
                self.bind_tempo(*stream, binding, *destination, *device, data, step)?;
                None
            },
            Operation::TempoNamedTableBind { stream, binding, destination, carriers, flags, group } => {
                let mut current_flags = [0; 2];
                for (value, flag) in current_flags.iter_mut().zip(flags) {
                    let bytes = self.host_state.get(flag.allocation)?;
                    let end = flag.offset.checked_add(4).ok_or("TEMPO flag overflow")?;
                    *value = i32::from_ne_bytes(bytes.get(flag.offset..end).ok_or("TEMPO flag outside allocation")?
                                               .try_into().map_err(|_| "TEMPO flag size")?);
                }
                let selected = tempo::selection::select(carriers, current_flags)?;
                let table = selected.iter().find(|table| table.name == group)
                    .ok_or("unknown named TEMPO group")?;
                let data = tempo::pointer_table(&table.entries, table.shape.as_deref(), table.device,
                                                table.optional, &self.allocations)?;
                self.bind_tempo(*stream, binding, *destination, table.device, data, step)?;
                None
            },
            Operation::TempoTableRelease { binding } => {
                if self.tempo_bindings.remove(binding).is_none() {
                    return Err("unknown live TEMPO table binding".into());
                }
                None
            },
            Operation::SeaicePointerTableUpload { stream, destination, arrays, dummy } => {
                let data = seaice::pointer_table(arrays, *dummy, &self.allocations)?;
                let target = self.address(*destination, data.len())?;
                let stream = self.stream(*stream)?;
                // adjoint-speed-2: from the pinned pool (returned at the next drain), not a fresh cuMemHostAlloc per command
                // (pkg18m2: ~170 cuMemHostAlloc/cuMemFreeHost pairs per step pass, both synchronizing). The same bytes are copied.
                let host = self.pinned_take(data.len())?;
                ptr::copy_nonoverlapping(data.as_ptr(), host.cast::<u8>(), data.len());
                checked("cuMemcpyHtoDAsync seaice pointer table", (self.cuda.upload_async)(target, host, data.len(), stream))?;
                self.note_table_pointers(*destination, &data);
                self.log(serde_json::json!({"call":"cuMemcpyHtoDAsync", "source":"live_pointer_table",
                                            "rule":"seaice_v842_pointers", "step":step, "entries":seaice::FIELDS.len()}))?;
                None
            },
            Operation::RucPointerTableUpload { stream, destination, entry, arguments, flags, tile_stop } => {
                let data = ruc::pointer_table(entry, arguments, flags, *tile_stop, &self.allocations)?;
                let target = self.address(*destination, data.len())?;
                let stream = self.stream(*stream)?;
                // adjoint-speed-2: from the pinned pool (returned at the next drain), not a fresh cuMemHostAlloc per command
                // (pkg18m2: ~170 cuMemHostAlloc/cuMemFreeHost pairs per step pass, both synchronizing). The same bytes are copied.
                let host = self.pinned_take(data.len())?;
                ptr::copy_nonoverlapping(data.as_ptr(), host.cast::<u8>(), data.len());
                checked("cuMemcpyHtoDAsync RUC pointer table", (self.cuda.upload_async)(target, host, data.len(), stream))?;
                self.note_table_pointers(*destination, &data);
                self.log(serde_json::json!({"call":"cuMemcpyHtoDAsync", "source":"live_pointer_table",
                                            "rule":"resident_ruc_boundary", "entry":entry,
                                            "step":step, "entries":data.len()/8}))?;
                None
            },
            Operation::RrfsCopyPlanUpload { stream, destination, rows, cache } => {
                let live_rows = copy_plan_rows(&self.host_state, rows)?;
                let data = copy_plan::table(&live_rows, &self.allocations)?;
                let launch_cache = copy_plan::encode_cache(&live_rows, *destination)?;
                let target = self.address(*destination, data.len())?;
                let stream = self.stream(*stream)?;
                // adjoint-speed-2: from the pinned pool (returned at the next drain), not a fresh cuMemHostAlloc per command
                // (pkg18m2: ~170 cuMemHostAlloc/cuMemFreeHost pairs per step pass, both synchronizing). The same bytes are copied.
                let host = self.pinned_take(data.len())?;
                ptr::copy_nonoverlapping(data.as_ptr(), host.cast::<u8>(), data.len());
                checked("cuMemcpyHtoDAsync RRFS copy plan", (self.cuda.upload_async)(target, host, data.len(), stream))?;
                self.note_table_pointers(*destination, &data);
                self.host_state.write_range(cache.allocation, cache.offset, &launch_cache)?;
                self.log(serde_json::json!({"call":"cuMemcpyHtoDAsync", "source":"live_copy_plan",
                                            "step":step, "rows":rows.len()}))?;
                None
            },
            Operation::RrfsCopyPlanKernel { stream, table, cache, normal_module, tiled_module } => {
                // The forward keeps the built table and its launch decision;
                // it does not rebuild either when source views later change.
                let data = self.host_state.get(cache.allocation)?;
                let end = cache.offset.checked_add(copy_plan::CACHE_BYTES).ok_or("copy-plan cache overflow")?;
                let (launch, table_bytes, cached_table) = copy_plan::decode_cache(data.get(cache.offset..end)
                    .ok_or("copy-plan cache outside owner")?)?;
                if cached_table != *table {
                    return Err("copy-plan cache belongs to a different table".into());
                }
                self.address(*table, table_bytes)?;
                let module = if launch.tiled { tiled_module } else { normal_module };
                let function = self.function(module, "rrfs_glue_copy_plan")?;
                self.log(serde_json::json!({"call":"rrfs_copy_plan_kernel","step":step,
                    "grid":launch.grid,"block":launch.block,"tiled":launch.tiled,
                    "table_allocation":table.allocation,"table_offset":table.offset}))?;
                Some((*stream, submit::Command::Kernel { function, grid: launch.grid,
                    block: launch.block, shared_bytes: 0,
                    arguments: vec![submit::Argument::Device(ptr(*table))] }))
            },
            Operation::NativeCubReduce { stream, library_path, library_sha256, workspace, input, output, count, op, dtype } => {
                let count = self.library_dimensions(&[*count], i32::MAX as usize)?[0];
                let word = native_cub::word_bytes(*op, *dtype)?;
                let input_bytes = count.checked_mul(word).ok_or("CUB input size overflow")?;
                if input_bytes == 0 { return Err("empty CUB reduction".into()); }
                let input = self.address(*input, input_bytes)?;
                let output = self.address(*output, word)?;
                let workspace_address = self.address(*workspace, 1)?;
                let capacity = self.remaining(*workspace)?;
                let native_stream = self.stream(*stream)?;
                let library = self.cub_library(base, library_path, library_sha256)?;
                let required = library.reduce(workspace_address, capacity, input, output, count, native_stream, *op, *dtype)?;
                self.log(serde_json::json!({"call":"original_cub_reduce","step":step,"stream":stream,
                                            "count":count,"op":op,"dtype":dtype,"workspace_bytes":required}))?;
                None
            },
            Operation::NativeCubSegmentedReduce { stream, library_path, library_sha256, workspace, input, output,
                                                   segments, segment_size, op, dtype } => {
                let geometry = self.library_dimensions(&[*segments, *segment_size], i32::MAX as usize)?;
                let (segments, segment_size) = (geometry[0], geometry[1]);
                if segments.checked_mul(segment_size).is_none_or(|n| n > i32::MAX as usize) {
                    return Err("invalid live CUB segmented extent".into());
                }
                let word = native_cub::word_bytes(*op, *dtype)?;
                let input_bytes = segments.checked_mul(segment_size).and_then(|n| n.checked_mul(word))
                    .ok_or("CUB segmented input size overflow")?;
                let output_bytes = segments.checked_mul(word).ok_or("CUB segmented output size overflow")?;
                if input_bytes == 0 || output_bytes == 0 { return Err("empty CUB segmented reduction".into()); }
                let input = self.address(*input, input_bytes)?;
                let output = self.address(*output, output_bytes)?;
                let workspace_address = self.address(*workspace, 1)?;
                let capacity = self.remaining(*workspace)?;
                let native_stream = self.stream(*stream)?;
                let library = self.cub_library(base, library_path, library_sha256)?;
                let required = library.segmented(workspace_address, capacity, input, output, segments, segment_size,
                                                  native_stream, *op, *dtype)?;
                self.log(serde_json::json!({"call":"original_cub_segmented_reduce","step":step,"stream":stream,
                                            "segments":segments,"segment_size":segment_size,"op":op,
                                            "dtype":dtype,"workspace_bytes":required}))?;
                None
            },
            Operation::NativeThrustArgsort { stream, library_path, library_sha256, workspace, input, indices, count, dtype } => {
                let count = self.library_dimensions(&[*count], isize::MAX as usize)?[0];
                let input_bytes = count.checked_mul(native_thrust::word_bytes(*dtype)?).ok_or("Thrust input size overflow")?;
                let index_bytes = count.checked_mul(8).ok_or("Thrust index size overflow")?;
                let input = self.address(*input, input_bytes)?;
                let indices = self.address(*indices, index_bytes)?;
                let workspace_address = self.address(*workspace, 1)?;
                let mut capacity = self.remaining(*workspace)?;
                // S14: the workspace is the forward's own lowest temporary address in a pool block; it extends at
                // most to the next argument buffer above it in that block, not to the block's end.
                // THE BREAKAGE THIS PREVENTS: the fast-numerics pkg18m2 places the step-1 argsort input and indices
                // in the workspace's block above it (block 389: workspace +795,648, input +1,023,488), and every
                // replay refused "overlapping Thrust buffers" although the recorded temporaries never touched them.
                for (a, _) in [(input, input_bytes), (indices, index_bytes)] {
                    if a > workspace_address && a - workspace_address < capacity as u64 {
                        capacity = (a - workspace_address) as usize;
                    }
                }
                let ends = [(input, input_bytes), (indices, index_bytes), (workspace_address, capacity)];
                for (i, (a, a_size)) in ends.iter().enumerate() {
                    let a_end = a.checked_add(*a_size as u64).ok_or("Thrust range overflow")?;
                    for (b, b_size) in ends.iter().skip(i + 1) {
                        let b_end = b.checked_add(*b_size as u64).ok_or("Thrust range overflow")?;
                        if *a < b_end && *b < a_end { return Err("overlapping Thrust buffers".into()); }
                    }
                }
                let native_stream = self.stream(*stream)?;
                let library = self.thrust_library(base, library_path, library_sha256)?;
                let (used, allocations) = library.argsort(*dtype, indices, input, count, native_stream, workspace_address, capacity)?;
                self.log(serde_json::json!({"call":"original_thrust_argsort","step":step,"stream":stream,
                                            "count":count,"dtype":dtype,"workspace_bytes":used,"allocations":allocations}))?;
                None
            },
            Operation::NativeCublasTranspose { stream, library_path, library_sha256, input, output, m, n } => {
                let geometry = self.library_dimensions(&[*m, *n], i32::MAX as usize)?;
                let (m, n) = (geometry[0], geometry[1]);
                let bytes = native_cublas::matrix_bytes(m, n)?;
                let input = self.address(*input, bytes)?;
                let output = self.address(*output, bytes)?;
                let native_stream = self.stream(*stream)?;
                let library = self.cublas_library(base, library_path, library_sha256)?;
                library.transpose(input, output, m, n, native_stream)?;
                self.log(serde_json::json!({"call":"original_cublas_sgeam_transpose", "step":step,
                                            "stream":stream,"m":m,"n":n}))?;
                None
            },
            Operation::HostBranch { .. } | Operation::RkSchedule { .. } | Operation::RkSplits { .. }
            | Operation::RkStages { .. } | Operation::RkSubsteps { .. } =>
                return Err("host control flow must be dispatched as a block".into()),
            Operation::DeviceToHost { stream, source, bytes, asynchronous, host_consumer, expected_hex, expected_zero_words, layer_groups } => {
                // A transfer whose host result feeds the forecast needs an
                // explicit provider. Never silently throw away unknown inputs.
                // checked_value: the host choice it feeds is the recorded one only
                // if the bytes equal the producer's expected value; otherwise refuse.
                let zero_words = match (host_consumer.as_str(), expected_zero_words) {
                    ("checked_zero_words", Some(words)) if *bytes % 4 == 0 && expected_hex.is_none()
                        && words.len() != 1 && words.windows(2).all(|w| w[0] < w[1])
                        && words.iter().all(|&w| w < *bytes / 4) => Some(words.clone()),
                    ("checked_zero_words", _) => return Err("checked_zero_words needs a sorted word set (not one word)".into()),
                    (_, None) => None,
                    _ => return Err("expected_zero_words only goes with checked_zero_words".into()),
                };
                let expected = match (host_consumer.as_str(), expected_hex) {
                    ("observational_output", None) => None,
                    ("checked_zero_words", None) => None,
                    ("checked_value", Some(hex)) => {
                        if hex.len() != 2 * *bytes || !hex.bytes().all(|c| c.is_ascii_hexdigit()) {
                            return Err("checked_value expected_hex does not match the copy size".into());
                        }
                        Some((0..*bytes).map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap()).collect::<Vec<u8>>())
                    },
                    _ => return Err(format!("device-to-host consumer {host_consumer} is not proven output-only")),
                };
                if *bytes == 0 { return Err("empty device-to-host transfer".into()); }
                let source_slice = *source;
                let source = self.address(*source, *bytes)?;
                let stream = self.stream(*stream)?;
                // adjoint-speed-2: from the pinned pool (returned at the next drain), not a fresh cuMemHostAlloc per command
                // (pkg18m2: ~170 cuMemHostAlloc/cuMemFreeHost pairs per step pass, both synchronizing). The same bytes are copied.
                let host = self.pinned_take(*bytes)?; // retained by Drop if a driver call fails
                if *asynchronous {
                    checked("cuMemcpyDtoHAsync", (self.cuda.download_async)(host, source, *bytes, stream))?;
                    // Output-only and one stream: completing early cannot affect
                    // subsequent commands, and bounds pinned host memory.
                    checked("cuStreamSynchronize output", (self.cuda.stream_synchronize)(stream))?;
                } else {
                    checked("cuMemcpyDtoH", (self.cuda.download)(host, source, *bytes))?;
                }
                // Synchronised above (both paths): the bytes have landed.
                let got = unsafe { std::slice::from_raw_parts(host as *const u8, *bytes) };
                let mut mismatch = expected.as_ref().is_some_and(|want| got != want.as_slice());
                if let Some(want) = &zero_words {
                    let zero: Vec<usize> = got.chunks_exact(4).enumerate()
                        .filter(|(_, w)| *w == [0u8; 4]).map(|(i, _)| i).collect();
                    mismatch |= if want.is_empty() { zero.len() > 1 } else { &zero != want };
                }
                // G2 (P1 milestone 2): the LW wrapper's launch shapes, rebuilt from the live histogram by the Rust port.
                let groups_check = match layer_groups {
                    Some(lg) if host_consumer == "checked_value" && *bytes == 4 * (layer_groups::HISTOGRAM_BINS + 1) => {
                        let hist: Vec<i32> = got.chunks_exact(4).map(|w| i32::from_le_bytes(w.try_into().unwrap())).collect();
                        layer_groups::check_recorded(lg.nc, lg.nm, &hist, &lg.batches).map(|_| lg.batches.len())
                    },
                    Some(_) => Err("layer_groups only goes with a checked LW histogram read-back".into()),
                    None => Ok(0),
                };
                // S13: a window perturbed forward follows a live LW histogram that differs from the recorded one
                // (the window replays the group section per pass, window.rs lw_regroup). THE BREAKAGE THIS
                // PREVENTS: the i7-FtA / rl-utA Taylor rows at h 0.03 and 0.01 were refused at step 7 by this
                // read. Every other checked read still refuses, and so does this one outside a perturbed forward.
                let regroup_hist = (layer_groups.is_some() && self.lw_regroup && (mismatch || self.lw_regroup_test != 0))
                    .then(|| got.chunks_exact(4).map(|w| i32::from_le_bytes(w.try_into().unwrap())).collect::<Vec<i32>>());
                let detail = if mismatch && regroup_hist.is_none() { mismatch_detail(got, expected.as_deref(), zero_words.as_deref()) } else { String::new() };
                // host-leftovers-2: the pooled buffer stays live until the next drain returns it to the pool (see the host
                // publication above); nothing reads it after this point, and the bytes copied are the same.
                if let Some(hist) = regroup_hist {
                    self.lw_live_hist = Some(hist);
                    self.log(serde_json::json!({"call":"lw_histogram_live","step":step,"bytes":bytes,"recorded_equal":!mismatch}))?;
                    None
                } else {
                if mismatch {
                    return Err(format!("checked device-to-host value differs from the proven value at step {step} ({detail}; source allocation {} offset {}, {bytes} bytes, {host_consumer}; LW layer-group histogram: {})",
                                       source_slice.allocation, source_slice.offset, layer_groups.is_some()));
                }
                let groups_checked = groups_check.map_err(|e| format!("step {step}: {e}"));
                let groups_checked = groups_checked?;
                if groups_checked > 0 {
                    self.log(serde_json::json!({"lw_layer_groups":"checked","step":step,"batches":groups_checked}))?;
                }
                self.log(serde_json::json!({"call":if *asynchronous {"cuMemcpyDtoHAsync"} else {"cuMemcpyDtoH"},
                                            "step":step,"bytes":bytes}))?;
                None
                }
            },
            Operation::Synchronization { stream, operation, event_handle_recorded: id, flags } => {
                let stream_handle = self.stream(*stream)?;
                match operation.as_str() {
                    "EventCreate" => {
                        if self.events.contains_key(id) { return Err(format!("event {id} created while live")); }
                        let mut handle = ptr::null_mut();
                        checked("cuEventCreate", (self.cuda.event_create)(&mut handle, event_flags(*flags)))?;
                        self.events.insert(*id, handle);
                    },
                    "EventDestroy" if *flags == 0 => {
                        let handle = self.event(*id)?;
                        checked("cuEventDestroy", (self.cuda.event_destroy)(handle))?;
                        self.events.remove(id);
                    },
                    "EventRecord" if *flags == 0 =>
                        checked("cuEventRecord", (self.cuda.event_record)(self.event(*id)?, stream_handle))?,
                    "StreamWaitEvent" =>
                        checked("cuStreamWaitEvent", (self.cuda.stream_wait_event)(stream_handle, self.event(*id)?, *flags))?,
                    "EventSynchronize" if *flags == 0 =>
                        checked("cuEventSynchronize", (self.cuda.event_synchronize)(self.event(*id)?))?,
                    "StreamSynchronize" if *flags == 0 && *id == 0 =>
                        checked("cuStreamSynchronize", (self.cuda.stream_synchronize)(stream_handle))?,
                    "CtxSynchronize" if *flags == 0 && *id == 0 =>
                        checked("cuCtxSynchronize", (self.cuda.context_synchronize)())?,
                    _ => return Err(format!("unknown or invalid synchronization {operation}")),
                }
                self.log(serde_json::json!({"call":operation,"step":step,"event":id,"stream":stream}))?;
                None
            },
            Operation::EventRecord { stream, event } => {
                checked("cuEventRecord", (self.cuda.event_record)(self.event(*event)?, self.stream(*stream)?))?;
                self.log(serde_json::json!({"call":"cuEventRecord","step":step,"event":event,"stream":stream}))?;
                None
            },
            Operation::EventWait { stream, event, flags } => {
                checked("cuStreamWaitEvent", (self.cuda.stream_wait_event)(self.stream(*stream)?, self.event(*event)?, *flags))?;
                self.log(serde_json::json!({"call":"cuStreamWaitEvent","step":step,"event":event,"stream":stream}))?;
                None
            },
            Operation::HostBoundary { name } => {
                frame.as_mut().ok_or("host boundary without live physics frame")?.host_call(name)?;
                self.log(serde_json::json!({"call":"host_boundary","name":name,"step":step}))?;
                None
            },
            Operation::TempoMainComplete { first_call_main } => {
                self.host_state.finish_tempo_main(first_call_main.allocation, first_call_main.offset)?;
                self.log(serde_json::json!({"call":"tempo_main_complete","step":step,
                    "allocation":first_call_main.allocation,"offset":first_call_main.offset}))?;
                None
            },
        };
        let mut ad_pending: addot::Pending = None;
        if let Some((f, a, g, b, sh)) = ad_launch.as_ref() {
            let mut ad = self.ad.take().ok_or("ad hook lost")?;
            let r = ad.before(&self.cuda, &self.allocations, step, f, a, *g, *b, *sh);
            self.ad = Some(ad);
            ad_pending = r?;
        }
        if let Some((stream, command)) = native {
            let stream = self.stream(stream)?;
            let driver = submit::NativeDriver { launch: self.cuda.launch, copy: self.cuda.copy, clear: self.cuda.clear };
            let allocations: Vec<_> = self.allocations.values().copied().collect();
            self.submitter.submit(&driver, hex_step_validate, hex_step_submit, &allocations,
                                  &[command], &[], stream).map_err(|e| format!("native submit: {e:?}"))?;
            self.log(serde_json::json!({"call":"hex_step_submit","step":step}))?;
        }
        if let (Some((f, a, g, b, sh)), Some(p)) = (ad_launch.as_ref(), ad_pending) {
            let mut ad = self.ad.take().ok_or("ad hook lost")?;
            let r = ad.after(&self.cuda, &self.allocations, step, f, a, *g, *b, *sh, p);
            self.ad = Some(ad);
            r?;
        }
        Ok(())
    }

    unsafe fn inject_poison(&mut self, writes: &[census::PoisonWrite], host_writes: &[poison::HostWrite],
                            step: usize) -> Result<(), String> {
        // Validate every target before issuing any writes. These patterns are
        // only a mechanism: the producer must separately establish the full
        // set of live touched extents and include host-held state.
        let mut checked_writes = Vec::with_capacity(writes.len());
        for write in writes {
            let offset = usize::try_from(write.extent.offset).map_err(|_| "poison offset overflow")?;
            let bytes = usize::try_from(write.extent.bytes).map_err(|_| "poison size overflow")?;
            let address = self.address(plan::Slice { allocation: write.extent.allocation, offset }, bytes)?;
            let alignment = write.extent.kind.alignment();
            if address % alignment != 0 || bytes == 0 || bytes % alignment as usize != 0 {
                return Err("unaligned poison extent".into());
            }
            checked_writes.push((address, bytes, write.word, write.extent.kind));
        }
        self.host_state.validate_poison(host_writes)?;
        let stream = self.stream(0)?;
        for (address, bytes, word, kind) in checked_writes {
            match kind {
                census::WordKind::Float | census::WordKind::Integer => {
                    checked("cuMemsetD32Async poison", (self.cuda.fill32)(address, word, bytes / 4, stream))?;
                },
                census::WordKind::Float64 => {
                    let high = address.checked_add(4).ok_or("float64 poison address overflow")?;
                    checked("cuMemsetD2D32Async poison low", (self.cuda.fill2d32)(address, 8, 0, 1, bytes / 8, stream))?;
                    checked("cuMemsetD2D32Async poison high", (self.cuda.fill2d32)(high, 8, 0x7ff8_0000, 1, bytes / 8, stream))?;
                },
                census::WordKind::IntegerBytes => {
                    for (column, byte) in word.to_le_bytes().into_iter().enumerate().take(bytes.min(4)) {
                        let target = address.checked_add(column as u64).ok_or("byte poison address overflow")?;
                        let rows = 1 + (bytes - 1 - column) / 4;
                        checked("cuMemsetD2D8Async poison", (self.cuda.fill2d8)(target, 4, byte, 1, rows, stream))?;
                    }
                },
            }
        }
        self.host_state.inject_poison(host_writes)?;
        self.log(serde_json::json!({"event":"poison_injected","next_step":step,
                                    "device_writes":writes.len(),"host_writes":host_writes.len(),
                                    "scope":"provisional supplied extents; census completeness unproved"}))?;
        Ok(())
    }

    unsafe fn prepare_word_comparison(&mut self, path: &Path, sha256: &str) -> Result<(), String> {
        let mut image = fs::read(path).map_err(|e| format!("cannot read word comparison image: {e}"))?;
        verify_sha(&image, sha256)?;
        image.push(0);
        checked("cuModuleLoadData word comparison", (self.cuda.module_load)(&mut self.word_module, image.as_ptr().cast()))?;
        checked("cuModuleGetFunction word comparison", (self.cuda.module_function)(
            &mut self.word_function, self.word_module, c"binder_count_differing_words".as_ptr()))?;
        checked("cuMemAlloc word comparison", (self.cuda.allocate)(&mut self.word_output, 8))?;
        Ok(())
    }

    unsafe fn compare_words(&mut self, step: usize, carried: &[u64], host_carried: &[u64],
                            actual: &HashMap<u64, submit::Allocation>,
                            reference: &mut word_reference::WordReference) -> Result<(), String> {
        if self.word_function.is_null() || self.word_output == 0 { return Err("word comparator not prepared".into()); }
        let external = reference.next(step, carried, actual)?;
        let mut total = 0_u64;
        let stream = self.stream(0)?;
        for item in external {
            let actual = actual.get(&item.id).ok_or("missing carried allocation")?.address;
            let mut mapped = 0_u64;
            checked("cuIpcOpenMemHandle", (self.cuda.ipc_open)(&mut mapped, item.handle, 1))?;
            let result = (|| -> Result<u64, String> {
                let mut base = 0_u64;
                let mut available = 0_usize;
                checked("cuMemGetAddressRange", (self.cuda.address_range)(&mut base, &mut available, mapped))?;
                let expected = mapped.checked_add(item.offset).ok_or("reference offset overflow")?;
                let limit = base.checked_add(u64::try_from(available).map_err(|_| "reference size overflow")?)
                    .ok_or("reference address range overflow")?;
                let end = expected.checked_add(u64::try_from(item.bytes).map_err(|_| "reference size overflow")?)
                    .ok_or("reference end overflow")?;
                if expected < base || end > limit { return Err("reference extent exceeds exported allocation".into()); }
                checked("cuMemsetD8Async word comparison", (self.cuda.clear)(self.word_output, 0, 8, stream))?;
                let mut expected_arg = expected;
                let mut actual_arg = actual;
                let mut bytes_arg = u64::try_from(item.bytes).map_err(|_| "reference size overflow")?;
                let mut output_arg = self.word_output;
                let mut args = [(&mut expected_arg as *mut u64).cast::<c_void>(),
                                (&mut actual_arg as *mut u64).cast::<c_void>(),
                                (&mut bytes_arg as *mut u64).cast::<c_void>(),
                                (&mut output_arg as *mut u64).cast::<c_void>()];
                let words = bytes_arg / 4 + u64::from(bytes_arg % 4 != 0);
                let grid = u32::try_from(((words + 255) / 256).min(65535)).map_err(|_| "comparison grid overflow")?;
                checked("cuLaunchKernel word comparison", (self.cuda.launch)(self.word_function,
                    grid, 1, 1, 256, 1, 1, 0, stream, args.as_mut_ptr(), ptr::null_mut()))?;
                checked("cuStreamSynchronize word comparison", (self.cuda.stream_synchronize)(stream))?;
                let mut count = 0_u64;
                checked("cuMemcpyDtoH word count", (self.cuda.download)((&mut count as *mut u64).cast(), self.word_output, 8))?;
                Ok(count)
            })();
            let close = (self.cuda.ipc_close)(mapped);
            let count = result?;
            checked("cuIpcCloseMemHandle", close)?;
            total = total.checked_add(count).ok_or("word difference total overflow")?;
            println!("{}", serde_json::json!({"event":"state_differing_words","step":step,
                                                "allocation":item.id,"bytes":item.bytes,"count":count}));
        }
        total = total.checked_add(reference.compare_host(step, host_carried, &self.host_state)?)
            .ok_or("word difference total overflow")?;
        reference.acknowledge(step, total)?;
        self.log(serde_json::json!({"event":"state_differing_words","step":step,
                                    "allocations":carried.len() + host_carried.len(),"total":total}))?;
        if total != 0 { return Err(format!("state word mismatch at step {step}: {total} differing words")); }
        Ok(())
    }

    unsafe fn finish_step(&mut self, step: usize, carried: &[u64],
                          actual: &HashMap<u64, submit::Allocation>,
                          references: Option<&References>) -> Result<(), String> {
        checked("cuCtxSynchronize", (self.cuda.context_synchronize)())?;
        self.release_pinned()?;
        // terrain-w-hash: the last boundary's deferred digest is joined (printed, compared) before this one starts
        // (its pointer history is this boundary's `previous`).
        self.digest_flush()?;
        if self.unchecked_digest_skip && references.is_none() {
            // per-step-host-hash: nothing reads this boundary's digest (see unchecked_digest_skip).
            self.history_stale = true;
            self.held_pointer_writes.clear();
            self.digests_skipped += 1;
            if self.digests_skipped == 1 { eprintln!("{}", serde_json::json!({"event":"hexad_unchecked_digest_skipped","first_step":step,"why":"no reference and no stored boundary: nothing reads this digest (HEXAD_TL_UNCHECKED_DIGEST=1 keeps it)"})); }
            return Ok(());
        }
        if self.history_stale {
            return Err(format!("DEFECT step {step}: boundary digest after a skipped unchecked digest with no restore between (stale pointer history; per-step-host-hash)"));
        }
        if host_hash_requested() {
            let t = std::time::Instant::now();
            let r = self.finish_step_host_hash(step, carried, actual, references);
            self.digest_s += t.elapsed().as_secs_f64();
            return r;
        }
        if self.residue.keys().any(|(s, _)| *s == step) {
            return Err("excluded residue is compared on the host hash only (unset AD_BINDER_HASH=device)".into());
        }
        for id in carried {
            let allocation = actual.get(id).ok_or("missing carried allocation")?;
            let mut address = allocation.address;
            let mut bytes = u64::try_from(allocation.bytes).map_err(|_| "allocation too large for digest")?;
            let mut output = self.digest_output;
            let mut args = [(&mut address as *mut u64).cast::<c_void>(),
                            (&mut bytes as *mut u64).cast::<c_void>(),
                            (&mut output as *mut u64).cast::<c_void>()];
            checked("cuLaunchKernel digest", (self.cuda.launch)(self.digest_function,
                1, 1, 1, 1, 1, 1, 0, ptr::null_mut(), args.as_mut_ptr(), ptr::null_mut()))?;
            let mut result = [0_u8; 32];
            checked("cuMemcpyDtoH digest", (self.cuda.download)(result.as_mut_ptr().cast(),
                                                                  self.digest_output, result.len()))?;
            let value = hex_bytes(&result);
            if !quiet_hashes() { println!("{}", serde_json::json!({"event":"state_sha256","step":step,"allocation":id,
                                             "bytes":bytes,"sha256":value})); }
            if let Some(references) = references {
                let (expected_bytes, expected) = &references[&(step, *id)];
                if *expected_bytes as u64 != bytes || *expected != value {
                    return Err(format!("state hash mismatch at step {step}, allocation {id}"));
                }
            }
        }
        self.log(serde_json::json!({"call":"cuCtxSynchronize","step":step,
                                    "digest_readback_bytes_per_allocation":32}))?;
        Ok(())
    }

    /// P1 (ad/phys): the same SHA-256 of the same carried bytes, computed on the host across all
    /// cores. The device digest runs one GPU thread per allocation (ad/e2e U8 measured 775 s of a
    /// 775 s replay in it). Downloads go through one blocking-sync context, so the host sleeps.
    /// terrain-w-hash (round 3; measured on pkg18m2, as2-p0c: 0.66-0.70 s of every 3.1 s tangent step, the card idle,
    /// spent on single SHA-256 streams of blocks 458 (979 MB), 416 (275 MB, residue) and 447 (560 MB), which no
    /// thread count shortens): with `digest_defer` (window tangent pass) and every carried view in the capture, the
    /// readings and hashes below run on background threads beside the next step and the next boundary joins them.
    /// The same readings in the same order, so the same hashes, prints, history and refusals (digest_post).
    unsafe fn finish_step_host_hash(&mut self, step: usize, carried: &[u64],
                                    actual: &HashMap<u64, submit::Allocation>,
                                    references: Option<&References>) -> Result<(), String> {
        // integ8 (S04 + adjoint-speed): carried views come from the window tangent's whole-state capture where it holds
        // them (no second download); every other carried allocation is DMA'd into one reused page-locked arena (S04).
        // The same bytes either way.
        // integ8 (AD_DIGEST_PROF=1, diagnostic only, default off): wall of each stage of this host digest, no syncs added.
        let dprof = std::env::var("AD_DIGEST_PROF").as_deref() == Ok("1");
        let t_dig = std::time::Instant::now();
        let captured = self.captured.take();
        let mut place: Vec<(u64, Result<(usize, usize), (usize, usize)>)> = Vec::with_capacity(carried.len());
        let mut miss_total = 0usize;
        for id in carried {
            let allocation = actual.get(id).ok_or("missing carried allocation")?;
            let span = captured.as_ref().and_then(|(_, index)| index.iter().find(|(_, _, n, base)|
                allocation.address >= *base && allocation.address + allocation.bytes as u64 <= *base + *n as u64)
                .map(|(_, off, _, base)| (off + (allocation.address - base) as usize, allocation.bytes)));
            match span {
                Some(s) => place.push((*id, Ok(s))),
                None => { place.push((*id, Err((miss_total, allocation.bytes)))); miss_total += allocation.bytes; },
            }
        }
        let mut arena = match self.digest_arena.take() { Some(a) if a.len() >= miss_total => a, _ => pinned::PinnedBuf::new(&self.cuda, miss_total)? };
        for (id, p) in &place {
            if let Err((off, n)) = p {
                if *n != 0 {
                    checked("cuMemcpyDtoH host digest", (self.cuda.download)(arena[*off..*off + *n].as_mut_ptr().cast(), actual[id].address, *n))?;
                }
            }
        }
        let t_dl = t_dig.elapsed().as_secs_f64();
        let mut ranges: Vec<(u64, u64, u64)> = Vec::new();
        if !self.recorded_bases.is_empty() {
            for allocation in self.allocations.values() {
                if let Some(recorded) = self.recorded_bases.get(&allocation.id) {
                    ranges.push((allocation.address, allocation.address + allocation.bytes as u64, *recorded));
                }
            }
            ranges.sort_unstable();
        }
        let (low, high) = (ranges.first().map(|r| r.0).unwrap_or(u64::MAX),
                           ranges.iter().map(|r| r.1).max().unwrap_or(0));
        // the previous boundary's translated words plus this step's relocated held words (held_pointer_writes)
        let mut merged = self.pointer_history.clone();
        for (id, words) in self.held_pointer_writes.drain() {
            merged.entry(id).or_default().extend(words);
        }
        let diagnostic = std::env::var("AD_BINDER_DIAGNOSTIC").as_deref() == Ok("1");
        if self.digest_defer && miss_total == 0 && !diagnostic && captured.is_some() {
            let data = captured.as_ref().unwrap().0.clone();
            // every view lies in the capture (a 0-byte view outside it reads as empty, as from the empty arena)
            let spans: Vec<(u64, usize, usize)> = place.iter().map(|(id, p)| match p {
                Ok((off, n)) => (*id, *off, *n), Err((_, n)) => { debug_assert_eq!(*n, 0); (*id, 0, 0) } }).collect();
            let residue: HashMap<(usize, u64), (Vec<(usize, usize)>, String)> =
                self.residue.iter().filter(|(k, _)| k.0 == step).map(|(k, v)| (*k, v.clone())).collect();
            let refs: Option<References> = references.map(|r| carried.iter()
                .filter_map(|id| r.get(&(step, *id)).map(|v| ((step, *id), v.clone()))).collect());
            let ab = self.address_bearing.clone();
            let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(DIGEST_BG_WORKERS).min(spans.len().max(1));
            let handle = std::thread::Builder::new().name("ad-boundary-digest".into()).spawn(move || {
                let t = std::time::Instant::now();
                let buffers: Vec<(u64, &[u8])> = spans.iter().map(|&(id, off, n)| (id, &data[off..off + n])).collect();
                let out = digest_blocks(step, &buffers, &ranges, low, high, &merged, &residue, &ab, refs.as_ref(), workers, dprof);
                let blocks: Vec<(u64, usize)> = buffers.iter().map(|(id, d)| (*id, d.len())).collect();
                drop(buffers);
                DigestJobOut { out, blocks, residue, refs, ab, bg_s: t.elapsed().as_secs_f64() }
            }).map_err(|e| format!("step {step}: boundary digest thread: {e}"))?;
            self.digest_job = Some(DigestJob { step, handle, cell: std::sync::Arc::new(std::sync::OnceLock::new()), workers, dprof, t_dl });
            self.digest_arena = Some(arena);
            self.captured = captured;
            return Ok(());
        }
        let buffers: Vec<(u64, &[u8])> = place.iter().map(|(id, p)| (*id, match p {
            Ok((off, n)) => &captured.as_ref().unwrap().0[*off..*off + *n],
            Err((off, n)) => &arena[*off..*off + *n],
        })).collect();
        let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(buffers.len().max(1));
        let previous = &merged;
        if std::env::var("AD_BINDER_DIAGNOSTIC").as_deref() == Ok("1") {
            if let Some((ta, tw)) = std::env::var("AD_BINDER_TRACE_WORD").ok().and_then(|v| { let (a, w) = v.split_once(':')?; Some((a.parse::<u64>().ok()?, w.parse::<usize>().ok()?)) }) {
                if let Some((_, data)) = buffers.iter().find(|(id, _)| *id == ta) {
                    if let Some(c) = data.get(8 * tw..8 * tw + 8) {
                        let word = u64::from_le_bytes(c.try_into().unwrap());
                        let live: Vec<(u64, u64, u64)> = ranges.iter().filter(|r| r.0 <= word && word <= r.1).copied().collect();
                        let ids: Vec<u64> = self.allocations.values().filter(|a| a.address <= word && word < a.address + a.bytes as u64).map(|a| a.id).collect();
                        println!("{}", serde_json::json!({"event":"trace_word","step":step,"allocation":ta,"word":tw,"value":format!("{word:#x}"),
                            "live_ranges":live.iter().map(|r| format!("{:#x}-{:#x} rec {:#x}", r.0, r.1, r.2)).collect::<Vec<_>>(),
                            "live_allocations_holding":ids,"low":format!("{low:#x}"),"high":format!("{high:#x}")}));
                    }
                }
            }
        }
        let mut out = digest_blocks(step, &buffers, &ranges, low, high, previous, &self.residue, &self.address_bearing, references, workers, dprof);
        let t_workers = t_dig.elapsed().as_secs_f64();
        // P1 diagnostic (AD_BINDER_DIAGNOSTIC=1 only, never a proof): AD_BINDER_DUMP_DIR writes the carried bytes
        // of the allocations in AD_BINDER_DUMP_IDS (comma list, or "all") to DIR/step-N/ID.bin for a word compare.
        if std::env::var("AD_BINDER_DIAGNOSTIC").as_deref() == Ok("1") {
            if let Ok(dir) = std::env::var("AD_BINDER_DUMP_DIR") {
                let ids = std::env::var("AD_BINDER_DUMP_IDS").unwrap_or_default();
                let wanted: HashSet<u64> = ids.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                let out = Path::new(&dir).join(format!("step-{step}"));
                fs::create_dir_all(&out).map_err(|e| e.to_string())?;
                // G2 diagnostic: the replay-to-forward address ranges the boundary reading used (start, end, recorded).
                let listed: Vec<String> = ranges.iter().map(|r| format!("{:#x} {:#x} {:#x}", r.0, r.1, r.2)).collect();
                fs::write(out.join("ranges.txt"), listed.join("
")).map_err(|e| e.to_string())?;
                for (id, data) in &buffers {
                    if ids == "all" || wanted.contains(id) {
                        // G2 diagnostic: the pointer history this boundary read the block with (index replay forward).
                        if let Some(h) = previous.get(id) {
                            let mut rows: Vec<String> = h.iter().map(|(i, (r, f))| format!("{i} {r:#x} {f:#x}")).collect();
                            rows.sort();
                            fs::write(out.join(format!("{id}.history.txt")), rows.join(",")).map_err(|e| e.to_string())?;
                        }
                        fs::write(out.join(format!("{id}.bin")), data).map_err(|e| e.to_string())?;
                    }
                }
            }
        }
        let blocks: Vec<(u64, usize)> = buffers.iter().map(|(id, d)| (*id, d.len())).collect();
        let mismatched = digest_post(step, &blocks, &mut out, &self.residue, &self.address_bearing, references)?;
        // The next boundary's half-word rule sees the words translated at this one.
        self.pointer_history = blocks.iter().map(|(id, _)| *id).zip(std::mem::take(&mut out.histories))
            .filter(|(_, history)| !history.is_empty()).collect();
        drop(buffers);
        self.digest_arena = Some(arena);
        self.captured = captured;
        let prof = dprof.then(|| digest_prof_json(step, &mut out.item_t, t_dl, t_workers, t_dig.elapsed().as_secs_f64(), None));
        self.digest_tail(step, mismatched, workers, prof)
    }

    /// terrain-w-hash: join the deferred digest of the last boundary, then print, compare and refuse exactly as the
    /// synchronous path (the refusal names that boundary's step); its pointer history becomes the current one and
    /// fills the HostSnap cell store() took for that boundary. No-op when nothing is pending.
    pub(crate) fn digest_flush(&mut self) -> Result<(), String> {
        let Some(job) = self.digest_job.take() else { return Ok(()) };
        let t = std::time::Instant::now();
        let mut r = job.handle.join().map_err(|_| format!("state hash at step {}: boundary digest thread panicked", job.step))?;
        let wait = t.elapsed().as_secs_f64();
        self.digest_bg_s += r.bg_s;
        let mismatched = digest_post(job.step, &r.blocks, &mut r.out, &r.residue, &r.ab, r.refs.as_ref())?;
        let history: HashMap<u64, PointerHistory> = r.blocks.iter().map(|(id, _)| *id).zip(std::mem::take(&mut r.out.histories))
            .filter(|(_, history)| !history.is_empty()).collect();
        let _ = job.cell.set(history.clone());
        self.pointer_history = history;
        let prof = job.dprof.then(|| digest_prof_json(job.step, &mut r.out.item_t, job.t_dl, r.out.workers_s, r.bg_s,
            Some(serde_json::json!({"deferred":true,"background_s":r.bg_s,"join_wait_s":wait}))));
        let res = self.digest_tail(job.step, mismatched, job.workers, prof);
        self.digest_s += t.elapsed().as_secs_f64();
        res
    }

    /// terrain-w-hash: the boundary history a HostSnap keeps: the current one, or (a deferred digest pending) the
    /// cell its join fills.
    pub(crate) fn history_snap(&self) -> HistSnap {
        match &self.digest_job {
            Some(job) => HistSnap::Pending(job.cell.clone()),
            None => HistSnap::Ready(self.pointer_history.clone()),
        }
    }

    /// The end of a boundary digest (unchanged): refuse on a mismatch, else log and print the profile line.
    fn digest_tail(&mut self, step: usize, mismatched: Vec<u64>, workers: usize, prof: Option<serde_json::Value>) -> Result<(), String> {
        if !mismatched.is_empty() {
            println!("{}", serde_json::json!({"event":"state_mismatch","step":step,"allocations":mismatched}));
            // P1 diagnostic: keep going to see every step (AD_BINDER_DIAGNOSTIC=1 and AD_BINDER_CONTINUE=1 only).
            if std::env::var("AD_BINDER_DIAGNOSTIC").as_deref() == Ok("1")
                && std::env::var("AD_BINDER_CONTINUE").as_deref() == Ok("1") {
                return Ok(());
            }
            return Err(format!("state hash mismatch at step {step}, {} allocations, first {}", mismatched.len(), mismatched[0]));
        }
        self.log(serde_json::json!({"call":"cuCtxSynchronize","step":step,"digest":"host","workers":workers}))?;
        if let Some(p) = prof { eprintln!("{p}"); }
        Ok(())
    }

    fn finish_host_step(&self, step: usize, carried: &[u64], references: Option<&References>) -> Result<(), String> {
        // per-step-host-hash: the same unchecked step prints (quiet library) and compares nothing here either.
        if self.unchecked_digest_skip && references.is_none() { return Ok(()); }
        for id in carried {
            let bytes = self.host_state.get(*id)?;
            let value = digest(bytes);
            if !quiet_hashes() { println!("{}", serde_json::json!({"event":"state_sha256","storage":"host","step":step,
                                             "allocation":id,"bytes":bytes.len(),"sha256":value})); }
            if let Some(references) = references {
                let (expected_bytes, expected) = &references[&(step, *id)];
                if (*expected_bytes != bytes.len() || *expected != value) && peer::host_unchecked(*id) {
                    println!("{}", serde_json::json!({"event":"peer_host_unchecked","step":step,"allocation":id}));
                } else if *expected_bytes != bytes.len() || *expected != value {
                    return Err(format!("host state hash mismatch at step {step}, allocation {id}"));
                }
            }
        }
        Ok(())
    }
}

impl Drop for Execution {
    fn drop(&mut self) {
        // terrain-w-hash: a deferred digest still running owns only host bytes; join it before the context goes.
        if let Some(job) = self.digest_job.take() { let _ = job.handle.join(); }
        unsafe {
            let _ = (self.cuda.context_synchronize)();
            self.cub_libraries.clear();
            self.thrust_libraries.clear();
            self.cublas_libraries.clear();
            for host in self.pinned.drain(..) { let _ = (self.cuda.host_free)(host); }
            for (host, _) in self.pinned_pool.drain(..) { let _ = (self.cuda.host_free)(host); }
            for (_, event) in self.events.drain() { let _ = (self.cuda.event_destroy)(event); }
            for (id, stream) in self.streams.drain() { if id != 0 { let _ = (self.cuda.stream_destroy)(stream); } }
            for (id, allocation) in self.allocations.drain() {
                let _ = (self.cuda.free)(self.raw_bases.get(&id).copied().unwrap_or(allocation.address));
            }
            self.raw_bases.clear();
            for (_, module) in self.modules.drain() { let _ = (self.cuda.module_unload)(module); }
            if self.digest_output != 0 { let _ = (self.cuda.free)(self.digest_output); }
            if !self.digest_module.is_null() { let _ = (self.cuda.module_unload)(self.digest_module); }
            if self.word_output != 0 { let _ = (self.cuda.free)(self.word_output); }
            if !self.word_module.is_null() { let _ = (self.cuda.module_unload)(self.word_module); }
            if !self.context.is_null() { let _ = (self.cuda.context_destroy)(self.context); }
        }
    }
}

/// A host branch reads only the currently published value. The recorded arm
/// is never used to decide which commands to submit on this replay.
/// S13: what a refused checked device-to-host read saw (how many words differ, the first few as got/want), so
/// the refusal names its read. Diagnostic text only; the refusal itself is unchanged.
fn mismatch_detail(got: &[u8], want: Option<&[u8]>, zero_words: Option<&[usize]>) -> String {
    let words = |b: &[u8]| b.chunks(4).map(|w| { let mut a = [0u8; 4]; a[..w.len()].copy_from_slice(w); u32::from_le_bytes(a) }).collect::<Vec<u32>>();
    let g = words(got);
    if let Some(w) = want {
        let w = words(w);
        let diff: Vec<usize> = (0..g.len().min(w.len())).filter(|&i| g[i] != w[i]).collect();
        let first: Vec<String> = diff.iter().take(6).map(|&i| format!("word {i}: {:#010x} want {:#010x}", g[i], w[i])).collect();
        format!("{} of {} words differ: {}", diff.len(), g.len(), first.join(", "))
    } else {
        let zero: Vec<usize> = (0..g.len()).filter(|&i| g[i] == 0).collect();
        format!("zero words {:?} want {:?}", &zero[..zero.len().min(12)], zero_words.unwrap_or(&[]))
    }
}

unsafe fn execute_block(execution: &mut Execution, plan: &Plan, name: &str, step: usize,
                        base: &Path, frame: &mut Option<replay_frame::ReplayFrame>,
                        depth: usize) -> Result<(), String> {
    if depth > plan.blocks.len() { return Err("host branch recursion exceeds block count".into()); }
    let commands = plan.blocks.get(name).ok_or_else(|| format!("unknown live block {name}"))?;
    for command in commands {
        match command {
            Operation::HostBranch { condition, if_zero, if_nonzero } => {
                let value = *execution.host_state.get(condition.allocation)?.get(condition.offset)
                    .ok_or("host branch condition outside allocation")?;
                let next = if value == 0 { if_zero } else { if_nonzero };
                execution.log(serde_json::json!({"call":"live_host_branch","step":step,
                                                 "condition_nonzero":value != 0,"block":next}))?;
                execute_block(execution, plan, next, step, base, frame, depth + 1)?;
            },
            Operation::RkSchedule { dt_f64, order_i64, acoustic_substeps_i64, dynamics_splits_i64, body } => {
                if execution.rk.is_some() { return Err("nested RK schedule".into()); }
                let values = execution.host_i64_values(&[*dt_f64, *order_i64, *acoustic_substeps_i64, *dynamics_splits_i64])?;
                let schedule = rk_schedule::Schedule::new(f64::from_ne_bytes(values[0].to_ne_bytes()),
                                                           values[1], values[2], values[3])?;
                execution.rk = Some(rk_schedule::Scope { schedule, split: None, stage: None, small_step: None });
                let result = execute_block(execution, plan, body, step, base, frame, depth + 1);
                execution.rk = None;
                result?;
            },
            Operation::RkSplits { body } => {
                let context = execution.rk.as_ref().ok_or("split loop without RK schedule")?;
                if context.split.is_some() || context.stage.is_some() || context.small_step.is_some() {
                    return Err("nested or out-of-order RK split loop".into());
                }
                let count = context.schedule.dynamics_splits;
                for index in 0..count {
                    execution.rk.as_mut().unwrap().split = Some(index);
                    let result = execute_block(execution, plan, body, step, base, frame, depth + 1);
                    execution.rk.as_mut().unwrap().split = None;
                    result?;
                }
            },
            Operation::RkStages { body } => {
                let context = execution.rk.as_ref().ok_or("stage loop without RK schedule")?;
                if context.stage.is_some() || context.small_step.is_some() {
                    return Err("nested or out-of-order RK stage loop".into());
                }
                let stages = context.schedule.stages;
                for stage in stages {
                    execution.rk.as_mut().unwrap().stage = Some(stage);
                    let result = execute_block(execution, plan, body, step, base, frame, depth + 1);
                    execution.rk.as_mut().unwrap().stage = None;
                    result?;
                }
            },
            Operation::RkSubsteps { body } => {
                let context = execution.rk.as_ref().ok_or("acoustic loop without RK schedule")?;
                let count = context.stage.ok_or("acoustic loop without RK stage")?.acoustic_steps;
                if context.small_step.is_some() { return Err("nested acoustic loop".into()); }
                for index in 1..=count {
                    execution.rk.as_mut().unwrap().small_step = Some(index);
                    let result = execute_block(execution, plan, body, step, base, frame, depth + 1);
                    execution.rk.as_mut().unwrap().small_step = None;
                    result?;
                }
            },
            _ => execution.command(command, step, base, frame)?,
        }
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("linearize") => {
            if args.next().is_some() { return Err("usage: ad-binder linearize < driver.jsonl".into()); }
            return linearize_input();
        },
        Some("window") => return window::run(args.collect()),
        Some("capture-dot") => return capentry::run(args.collect()),
        Some("capdot") => return capdot::run(args.collect()),
        Some("replay") => {},
        _ => return Err("usage: ad-binder replay --config PATH --steps N [--trace PATH] [--reference PATH] [--poison PATH] [--checkpoint-bindings-at STEPS --checkpoint-hook PATH] [--word-reference-socket PATH --word-image PATH --word-image-sha256 HEX]; or ad-binder linearize < driver.jsonl".into()),
    }
    let (mut config, mut steps, mut trace, mut reference, mut poison) = (None, None, None, None, None);
    let (mut reference_socket, mut word_image, mut word_sha) = (None, None, None);
    let (mut binding_steps, mut hook_path) = (None, None);
    while let Some(arg) = args.next() {
        let value = args.next().ok_or_else(|| format!("missing value for {arg}"))?;
        match arg.as_str() {
            "--config" if config.is_none() => config = Some(PathBuf::from(value)),
            "--steps" if steps.is_none() => steps = Some(value.parse::<usize>().map_err(|_| "invalid --steps")?),
            "--trace" if trace.is_none() => trace = Some(PathBuf::from(value)),
            "--reference" if reference.is_none() => reference = Some(PathBuf::from(value)),
            "--poison" if poison.is_none() => poison = Some(PathBuf::from(value)),
            "--checkpoint-bindings-at" if binding_steps.is_none() => binding_steps = Some(value),
            "--checkpoint-hook" if hook_path.is_none() => hook_path = Some(PathBuf::from(value)),
            "--word-reference-socket" if reference_socket.is_none() => reference_socket = Some(PathBuf::from(value)),
            "--word-image" if word_image.is_none() => word_image = Some(PathBuf::from(value)),
            "--word-image-sha256" if word_sha.is_none() => word_sha = Some(value),
            _ => return Err(format!("unknown or repeated argument {arg}")),
        }
    }
    let config = config.ok_or("missing --config")?;
    let config = match config.to_str() {
        Some("hrrrv5") => PathBuf::from("configs/binder/hrrrv5.json"),
        Some("convection_permitting") => PathBuf::from("configs/binder/convection_permitting.json"),
        _ => config,
    };
    let steps = steps.ok_or("missing --steps")?;
    let binding_steps = binding_steps.as_deref().map(|value| checkpoint_steps(value, steps))
        .transpose()?.unwrap_or_default();
    if hook_path.is_some() && binding_steps.is_empty() {
        return Err("checkpoint hook requires at least one checkpoint binding step".into());
    }
    let bytes = fs::read(&config).map_err(|e| format!("cannot read {}: {e}", config.display()))?;
    // Inspect the envelope before deserializing the full plan. A census is
    // intentionally incomplete and lacks many mandatory plan fields; make the
    // refusal explicit rather than reporting an incidental missing field.
    let envelope: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| format!("invalid plan JSON: {e}"))?;
    if envelope.get("schema").and_then(|x| x.as_str()) != Some("ad.binder.plan/v1")
        || envelope.get("complete").and_then(|x| x.as_bool()) != Some(true) {
        return Err("refusing incomplete or unknown plan".into());
    }
    let plan: Plan = serde_json::from_value(envelope).map_err(|e| format!("invalid sealed plan: {e}"))?;
    peer::set_host_unchecked(&plan.peer_host_unchecked);
    preflight(&plan, steps)?;
    validate_checkpoint_request(&plan, &binding_steps)?;
    let base = config.parent().unwrap_or(Path::new("."));
    verify_fixed_argument_sources(&plan, base)?;
    verify_copy_indexer_sources(&plan, base)?;
    let reference = reference.or_else(|| plan.reference.as_deref().map(|p| path(base, p)));
    let references = reference.as_deref().map(|p| read_references(p, &plan, steps)).transpose()?;
    if plan.physics.is_some() && references.is_none() {
        return Err("physics replay requires a complete ordinary-forward reference".into());
    }
    let poison: Option<poison::PoisonPlan> = poison.map(|path| {
        let bytes = fs::read(path).map_err(|e| format!("cannot read poison plan: {e}"))?;
        serde_json::from_slice(&bytes).map_err(|e| format!("invalid poison plan: {e}"))
    }).transpose()?;
    let poison_writes = poison.as_ref().map(|p| p.writes(steps)).transpose()?;
    if poison.is_some() && plan.physics.is_none() { return Err("poison requires physics plan".into()); }
    let poison_host_writes = poison.as_ref().map(|p| p.host_writes(&plan.host_carried,
        plan.physics.as_ref().ok_or("poison requires physics plan")?.frame_state_id)).transpose()?;
    if reference_socket.is_some() != word_image.is_some() || word_image.is_some() != word_sha.is_some() {
        return Err("word reference socket, image, and image SHA-256 must be supplied together".into());
    }
    let trace = trace.map(|p| File::create(p).map(BufWriter::new).map_err(|e| e.to_string())).transpose()?;
    let mut execution = unsafe { Execution::new(trace)? };
    let hook = hook_path.as_deref().map(|path| unsafe { boundary_hook::Hook::load(path) }).transpose()?;
    execution.address_bearing = plan.address_bearing.iter().copied().collect();
    unsafe { execution.prepare(&plan, base)? };
    execution.ad = unsafe { addot::AdDot::from_env(&execution.cuda)? };
    let mut word_reference = if let (Some(socket), Some(image), Some(sha)) =
        (reference_socket.as_deref(), word_image.as_deref(), word_sha.as_deref()) {
        unsafe { execution.prepare_word_comparison(image, sha)? };
        let mut reference = word_reference::WordReference::connect(socket)?;
        reference.ready()?;
        Some(reference)
    } else { None };
    let mut physics = plan.physics.as_ref().map(replay_frame::ReplayFrame::new).transpose()?;
    if let (Some(frame), Some(spec)) = (physics.as_ref(), plan.physics.as_ref()) {
        execution.host_state.insert(spec.frame_state_id, frame.snapshot()?)?;
    }
    for step in 1..=steps {
        if let Some(poison) = poison.as_ref().filter(|p| p.next_step == step) {
            let previous = carried_views(&plan, step - 1, &execution.allocations)?;
            let frame_id = plan.physics.as_ref().ok_or("poison requires physics plan")?.frame_state_id;
            poison.validate_bindings(&execution.allocations, &previous, &execution.host_state,
                                     &plan.host_carried, frame_id)?;
            unsafe { execution.inject_poison(poison_writes.as_deref().ok_or("missing poison writes")?,
                poison_host_writes.as_deref().ok_or("missing host poison writes")?, step)? };
        }
        let blocks = if let Some(frame) = physics.as_mut() {
            frame.begin(step)?;
            frame.blocks(plan.physics.as_ref().ok_or("missing physics plan")?)?
        } else { plan.cadence[step - 1].clone() };
        // integrate 2026-11-06 (single-core law, the G5 follow-up): decode this step's held / upload blobs on all
        // cores before the step (window::decode_blobs = the same crate::blob read, gunzip and SHA-256 check, so the
        // bytes are identical) and serve the uploads from that cache (G5's cache upload path, bitwise in the window).
        // THE BREAKAGE THIS PREVENTS: the plain replay gunzipped and hashed every held blob on one thread inside the
        // command (~2-3.4 GB per heavy step, ~10 s/step at 100 % of one core while the card waited).
        // AD_REPLAY_SERIAL_BLOBS=1 restores the in-command decode (the referee for the ABBA).
        if std::env::var("AD_REPLAY_SERIAL_BLOBS").as_deref() != Ok("1") {
            let mut seen = std::collections::HashSet::new();
            let mut want = Vec::new();
            for name in &blocks { window::block_blobs(&plan, name, &mut seen, &mut want); }
            let want: Vec<plan::Blob> = want.into_iter().cloned().collect();
            let mut cache = HashMap::new();
            for (k, v) in window::decode_blobs(base, &want)? { cache.insert(k, std::sync::Arc::new(v)); }
            execution.blob_cache = Some(cache);
        }
        for name in &blocks {
            unsafe { execute_block(&mut execution, &plan, name, step, base, &mut physics, 0)? };
        }
        if let Some(frame) = physics.as_mut() {
            frame.finish()?;
            let id = plan.physics.as_ref().ok_or("missing physics plan")?.frame_state_id;
            execution.host_state.replace(id, &frame.snapshot()?)?;
        }
        let views = carried_views(&plan, step, &execution.allocations)?;
        let actual = resolve_views(&views, &execution.allocations)?;
        if let Some(reference) = word_reference.as_mut() {
            unsafe { execution.compare_words(step, &plan.carried, &plan.host_carried, &actual, reference)? };
        }
        unsafe { execution.finish_step(step, &plan.carried, &actual, references.as_ref())? };
        execution.finish_host_step(step, &plan.host_carried, references.as_ref())?;
        if let Some(ad) = execution.ad.as_mut() { ad.step_done(step)?; }
        if binding_steps.contains(&step) {
            let boundary = plan.boundaries.get(step - 1).ok_or("checkpoint requires live boundaries")?;
            let bindings = checkpoint_bindings(boundary, &actual, &execution.allocations,
                                                &mut execution.host_state)?;
            let rows: Vec<_> = bindings.iter().map(|b|
                serde_json::json!({"id":b.id,"bytes":b.bytes,"memory":b.memory})).collect();
            execution.log(serde_json::json!({"event":"checkpoint_bindings","step":step,"bindings":rows}))?;
            if let Some(hook) = &hook {
                unsafe {
                    hook.call(step as u64, execution.context, execution.stream(0)?, &bindings)?;
                    checked("cuCtxSynchronize after checkpoint hook", (execution.cuda.context_synchronize)())?;
                }
                let frame = physics.as_mut().ok_or("checkpoint requires a physics frame")?;
                let id = plan.physics.as_ref().ok_or("checkpoint requires a physics plan")?.frame_state_id;
                frame.restore_snapshot(execution.host_state.get(id)?)?;
                execution.log(serde_json::json!({"event":"checkpoint_hook","step":step,"status":"ok"}))?;
            }
        }
    }
    if let Some(ad) = execution.ad.as_mut() { ad.finish()?; }
    let maps = fs::read_to_string("/proc/self/maps").map_err(|e| e.to_string())?;
    let python = maps.lines().any(|line| line.to_ascii_lowercase().contains("libpython"));
    execution.log(serde_json::json!({"event":"process_maps","libpython_mapped":python,
                                    "scope":"internal calls only; not a complete system call trace"}))?;
    if python { return Err("libpython is mapped in replay process".into()); }
    Ok(())
}

fn linearize_input() -> Result<(), String> {
    use ordering::{Kind, Node};
    let mut nodes = Vec::new();
    let mut steps = HashMap::<u64, u64>::new();
    let mut step = 0_u64;
    let mut skipped = 0_u64;
    for (line_number, line) in std::io::stdin().lock().lines().enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        let row: serde_json::Value = serde_json::from_str(&line)
            .map_err(|e| format!("invalid input line {}: {e}", line_number + 1))?;
        let name = row.get("type").and_then(|v| v.as_str()).ok_or("missing driver operation type")?;
        if row.get("status").and_then(|v| v.as_i64()).unwrap_or(0) != 0 {
            return Err(format!("failed driver operation {name} at line {}", line_number + 1));
        }
        if name == "Boundary" { step = row.get("a").and_then(|v| v.as_u64()).ok_or("bad boundary")?; continue; }
        let event = || row.get("a").and_then(|v| v.as_u64()).ok_or("bad event handle");
        let stream = || -> Result<u64, String> {
            let raw = row.get("stream").ok_or("missing stream")?;
            if let Some(value) = raw.as_u64() { return Ok(value); }
            let raw = raw.as_str().ok_or("bad stream")?;
            if raw == "(nil)" { return Ok(0); }
            u64::from_str_radix(raw.strip_prefix("0x").ok_or("bad stream pointer")?, 16)
                .map_err(|_| "bad stream pointer".into())
        };
        let (kind, stream) = match name {
            "EventCreate" => (Kind::EventCreate { event: event()? }, None),
            "EventDestroy" => (Kind::EventDestroy { event: event()? }, None),
            "EventRecord" => (Kind::EventRecord { event: event()? }, Some(stream()?)),
            "StreamWaitEvent" => {
                if row.get("extra").and_then(|v| v.as_u64()).unwrap_or(0) != 0 {
                    return Err("unsupported event wait flags".into());
                }
                (Kind::EventWait { event: event()? }, Some(stream()?))
            },
            "CtxSynchronize" | "StreamSynchronize" | "EventSynchronize" |
            "MemAlloc" | "MemFree" => (Kind::ContextBarrier, None),
            "LaunchKernelEx" | "LaunchCooperativeKernel" => return Err(format!("uncaptured arguments for {name}")),
            "LaunchKernel" | "Memcpy" | "MemcpyAsync" | "MemcpyHtoD" | "MemcpyHtoDAsync" |
            "MemcpyDtoH" | "MemcpyDtoHAsync" | "MemcpyDtoD" | "MemcpyDtoDAsync" |
            "MemsetD8" | "MemsetD8Async" | "MemsetD16" | "MemsetD16Async" |
            "MemsetD32" | "MemsetD32Async" | "MemAllocAsync" | "MemFreeAsync" => (Kind::Work, Some(stream()?)),
            _ => { skipped += 1; continue; },
        };
        let issue = row.get("sequence").and_then(|v| v.as_u64()).ok_or("missing issue sequence")?;
        steps.insert(issue, step);
        nodes.push(Node { issue, stream, kind });
    }
    let ordered = ordering::linearize(&nodes)?;
    let entries: Vec<_> = ordered.nodes.iter().map(|&i| serde_json::json!({
        "issue": nodes[i].issue, "step_hint": steps[&nodes[i].issue]
    })).collect();
    serde_json::to_writer(std::io::stdout().lock(), &serde_json::json!({
        "schema":"ad.binder.provisional-linearization/v1", "complete":false,
        "nodes":entries, "stream_edges":ordered.stream_edges, "event_edges":ordered.event_edges,
        "barrier_edges":ordered.barrier_edges, "skipped_noncommand_rows":skipped,
        "limitations":["requires independent recorder completeness and allocation-lifetime audit",
                         "ordering alone does not establish bitwise replay"]
    })).map_err(|e| e.to_string())?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        // multidevice-runner: a failed rank stops its peers' waits with the reason.
        if let Some(p) = peer::get() { p.abort(&error); }
        eprintln!("ad-binder: {error}"); std::process::exit(1);
    }
}

#[cfg(test)]
mod compressed_payload_tests {
    use super::*;
    #[test]
    fn compressed_payload_checks_length_and_hash() {
        use flate2::{write::GzEncoder, Compression};
        let dir = std::env::temp_dir();
        let name = format!("ad-binder-payload-test-{}.gz", std::process::id());
        let path = dir.join(&name);
        let file = File::create(&path).unwrap();
        let mut encoder = GzEncoder::new(file, Compression::fast());
        encoder.write_all(b"initial device state").unwrap();
        encoder.finish().unwrap();
        assert_eq!(gzip_bytes(&path, 20).unwrap(), b"initial device state");
        assert!(gzip_bytes(&path, 19).is_err());
        assert!(gzip_bytes(&path, 21).is_err());
        let correct = Blob::Gzip { path: name.clone(), sha256: digest(b"initial device state"), bytes: 20 };
        assert_eq!(blob(&dir, &correct).unwrap(), b"initial device state");
        let bad = Blob::Gzip { path: name, sha256: "00".repeat(32), bytes: 20 };
        assert!(blob(&dir, &bad).is_err());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn upload_relocation_changes_only_certified_addresses() {
        let allocations = HashMap::from([
            (7, submit::Allocation { id: 7, address: 0x1000, bytes: 128 }),
            (9, submit::Allocation { id: 9, address: 0x3000, bytes: 32 }),
        ]);
        let pointers = [
            plan::Patch { byte_offset: 11, target: plan::Slice { allocation: 9, offset: 31 } },
            plan::Patch { byte_offset: 1, target: plan::Slice { allocation: 7, offset: 8 } },
        ];
        let mut bytes = [0xaa; 20];
        patch_device_addresses(&mut bytes, &pointers, &allocations).unwrap();
        assert_eq!(&bytes[1..9], &0x1008_u64.to_le_bytes());
        assert_eq!(&bytes[11..19], &0x301f_u64.to_le_bytes());
        assert_eq!([bytes[0], bytes[9], bytes[10], bytes[19]], [0xaa; 4]);
    }

    #[test]
    fn upload_relocation_rejects_invalid_patches_without_partial_writes() {
        let allocations = HashMap::from([
            (7, submit::Allocation { id: 7, address: 0x1000, bytes: 16 }),
            (8, submit::Allocation { id: 8, address: u64::MAX, bytes: 2 }),
        ]);
        let patch = |byte_offset, allocation, offset| plan::Patch {
            byte_offset, target: plan::Slice { allocation, offset }
        };
        for pointers in [
            vec![patch(0, 7, 0), patch(7, 7, 1)],
            vec![patch(0, 7, 0), patch(9, 7, 0)],
            vec![patch(0, 7, 0), patch(8, 99, 0)],
            vec![patch(0, 7, 0), patch(8, 7, 16)],
            vec![patch(0, 7, 0), patch(8, 8, 1)],
            vec![patch(usize::MAX, 7, 0)],
        ] {
            let mut bytes = [0xaa; 16];
            assert!(patch_device_addresses(&mut bytes, &pointers, &allocations).is_err());
            assert_eq!(bytes, [0xaa; 16]);
        }
    }

    #[test]
    fn pointer_table_uses_current_allocations_and_checks_every_entry() {
        let allocations = HashMap::from([
            (7, submit::Allocation { id: 7, address: 0x1000, bytes: 16 }),
            (8, submit::Allocation { id: 8, address: u64::MAX, bytes: 2 }),
        ]);
        let slice = |allocation, offset| plan::Slice { allocation, offset };
        let table = live_pointer_table(&[slice(7, 8), slice(7, 0)], &allocations).unwrap();
        assert_eq!(&table[..8], &0x1008_u64.to_le_bytes());
        assert_eq!(&table[8..], &0x1000_u64.to_le_bytes());
        for entries in [vec![], vec![slice(99, 0)], vec![slice(7, 16)], vec![slice(8, 1)]] {
            assert!(live_pointer_table(&entries, &allocations).is_err());
        }
    }
}

#[cfg(test)]
mod physics_preflight_tests {
    use super::*;
    use serde_json::{json, Value};

    fn candidate() -> Value {
        json!({"schema":"ad.binder.plan/v1", "complete":true, "source_pin":PIN,
            "end_step_barrier_proven":true, "modules":[], "allocations":[], "carried":[1],
            "host_allocations":[{"id":2,"bytes":4,"origin":"initial_state",
                                "initial":{"encoding":"hex","bytes":"00000000","sha256":digest(&[0;4])}}],
            "host_carried":[2,9],
            "blocks":{"begin":[{"kind":"host_boundary","name":"physics_timetracker_state"},
                                  {"kind":"host_boundary","name":"update_state_time"}]},
            "physics":{"case":"hrrrv5","frame_state_id":9,"clock":{"year":2023,"month":3,"day":31,"seconds":43200},
                "dt_seconds":1200,"elapsed_seconds":0,"next_step":1,"last_time":null,
                "active":[true,true,false],"correct_rp6":false,"pbl_due":true,"alarms":{},"state_time_f32_bits":0,
                "state_xtime":"2023-03-31_12:00:00",
                "max_steps":6,"segments":[{"kind":"block","name":"begin"}]}})
    }
    fn check(v: Value) -> Result<(), String> { preflight(&serde_json::from_value(v).unwrap(), 6) }

    #[test]
    fn checkpoint_requires_all_carried_owners_and_current_extents() {
        assert_eq!(checkpoint_steps("1,6", 6).unwrap(), HashSet::from([1, 6]));
        for bad in ["", "0", "7", "1,1", "1,"] { assert!(checkpoint_steps(bad, 6).is_err()); }
        let mut value = candidate();
        let first = json!({"live":[{"id":1,"target":{"allocation":7,"offset":8},"bytes":4},
                                     {"id":8,"target":{"allocation":7,"offset":16},"bytes":4}],
                           "host_live":[2,9]});
        value["boundaries"] = json!([first]);
        let plan: Plan = serde_json::from_value(value.clone()).unwrap();
        validate_checkpoint_request(&plan, &HashSet::from([1])).unwrap();
        assert!(validate_checkpoint_request(&plan, &HashSet::from([2])).is_err());
        let live = HashMap::from([(7, submit::Allocation { id:7, address:0x1000, bytes:32 })]);
        let actual = HashMap::from([(1, submit::Allocation { id:1, address:0x1008, bytes:4 })]);
        let mut hosts = host_state::HostState::default();
        hosts.insert(2, vec![0;4]).unwrap();
        hosts.insert(9, vec![0;8]).unwrap();
        let bindings = checkpoint_bindings(&plan.boundaries[0], &actual, &live, &mut hosts).unwrap();
        assert_eq!(bindings.iter().map(|b| b.id).collect::<Vec<_>>(), [1,2,8,9]);
        let stale = HashMap::from([(1, submit::Allocation { id:1, address:0x1009, bytes:4 })]);
        assert!(checkpoint_bindings(&plan.boundaries[0], &stale, &live, &mut hosts).is_err());
        let mut missing = value.clone();
        missing["boundaries"][0]["host_live"] = json!([2]);
        let missing: Plan = serde_json::from_value(missing).unwrap();
        assert!(validate_checkpoint_request(&missing, &HashSet::from([1])).is_err());
        let mut overlap = value;
        overlap["boundaries"][0]["live"][1]["target"]["offset"] = json!(10);
        let overlap: Plan = serde_json::from_value(overlap).unwrap();
        assert!(checkpoint_bindings(&overlap.boundaries[0], &actual, &live, &mut hosts).is_err());
    }

    #[test]
    fn rrfs_copy_plan_reads_live_metadata_and_keeps_built_launch() {
        let slice = |offset| json!({"allocation":2,"offset":offset});
        let mut value = candidate();
        value["host_allocations"][0]["bytes"] = json!(136);
        let row = json!({"dst":{"allocation":1,"offset":0}, "src":{"allocation":1,"offset":0},
            "shape":[slice(0),slice(8),slice(16),slice(24)],
            "dst_stride":[slice(32),slice(40),slice(48),slice(56)],
            "src_stride":[slice(64),slice(72),slice(80),slice(88)],"fill":null});
        value["blocks"]["begin"][0] = json!({"kind":"rrfs_copy_plan_upload","stream":0,
            "destination":{"allocation":1,"offset":0},"rows":[row],"cache":slice(96)});
        assert!(check(value.clone()).is_ok());
        let mut bad = value.clone();
        bad["blocks"]["begin"][0]["rows"][0]["shape"][0] = json!(1);
        assert!(serde_json::from_value::<Plan>(bad).is_err());
        let mut bad = value.clone();
        bad["blocks"]["begin"][0]["cache"] = slice(8);
        assert!(check(bad).unwrap_err().contains("cache overlaps"));
        let mut bad = value.clone();
        bad["host_allocations"].as_array_mut().unwrap().push(json!({"id":10,"bytes":24,
            "origin":"initial_state", "initial":{"encoding":"hex","bytes":"00","sha256":digest(&[0])}}));
        bad["blocks"]["begin"][0]["cache"] = json!({"allocation":10,"offset":0});
        assert!(check(bad).unwrap_err().contains("cache must be"));
        let plan: Plan = serde_json::from_value(value).unwrap();
        let Operation::RrfsCopyPlanUpload { rows, .. } = &plan.blocks["begin"][0] else { panic!("wrong command") };
        let mut bytes = Vec::new();
        for n in [1_i64,1,1024,1024,0,0,1024,1,0,0,1,1024] {
            bytes.extend_from_slice(&n.to_ne_bytes());
        }
        bytes.extend_from_slice(&[0; copy_plan::CACHE_BYTES]);
        let mut hosts = host_state::HostState::default();
        hosts.insert(2, bytes).unwrap();
        let initial = copy_plan_rows(&hosts, rows).unwrap();
        let cached = copy_plan::encode_cache(&initial, plan::Slice { allocation: 1, offset: 0 }).unwrap();
        hosts.write_range(2, 96, &cached).unwrap();
        hosts.write_range(2, 16, &10_i64.to_ne_bytes()).unwrap();
        let current = copy_plan_rows(&hosts, rows).unwrap();
        assert_eq!(current[0].shape, [1,1,10,1024]);
        let (stored, _, table) = copy_plan::decode_cache(&hosts.get(2).unwrap()[96..136]).unwrap();
        assert_ne!(stored, copy_plan::launch(&current).unwrap());
        assert_eq!(table, plan::Slice { allocation: 1, offset: 0 });
        assert!(hosts.write_range(2, 135, &[0;2]).is_err());
    }

    #[test]
    fn native_library_geometry_must_reference_live_bounded_host_words() {
        let mut value = candidate();
        value["host_allocations"][0]["bytes"] = json!(16);
        let host = |offset| json!({"allocation":2,"offset":offset});
        let device = json!({"allocation":1,"offset":0});
        let common = json!({"stream":0,"library_path":"original.so","library_sha256":"0".repeat(64),
                            "workspace":device,"input":device,"output":device});
        let mut cub = common.clone();
        cub["kind"] = json!("native_cub_reduce"); cub["count"] = host(0);
        cub["op"] = json!(2); cub["dtype"] = json!(4);
        let mut segmented = common.clone();
        segmented["kind"] = json!("native_cub_segmented_reduce");
        segmented["segments"] = host(0); segmented["segment_size"] = host(8);
        segmented["op"] = json!(0); segmented["dtype"] = json!(4);
        let thrust = json!({"kind":"native_thrust_argsort","stream":0,"library_path":"original.so",
            "library_sha256":"0".repeat(64),"workspace":device,"input":device,
            "indices":device,"count":host(0),"dtype":4});
        let cublas = json!({"kind":"native_cublas_transpose","stream":0,"library_path":"original.so",
            "library_sha256":"0".repeat(64),"input":device,"output":device,"m":host(0),"n":host(8)});
        for (command, field) in [(cub, "count"), (segmented, "segments"),
                                  (thrust, "count"), (cublas, "m")] {
            value["blocks"]["begin"][0] = command;
            assert!(check(value.clone()).is_ok());
            let mut outside = value.clone();
            outside["blocks"]["begin"][0][field]["offset"] = json!(9);
            assert!(check(outside).unwrap_err().contains("outside host allocation"));
            let mut fixed = value.clone();
            fixed["blocks"]["begin"][0][field] = json!(4);
            assert!(serde_json::from_value::<Plan>(fixed).is_err());
        }
    }

    #[test]
    fn host_branches_check_both_arms_and_reject_cycles_and_bad_conditions() {
        let mut value = candidate();
        value["blocks"]["zero"] = json!([{"kind":"host_boundary","name":"update_state_time"}]);
        value["blocks"]["nonzero"] = json!([{"kind":"host_boundary","name":"update_state_time"}]);
        value["blocks"]["begin"][1] = json!({"kind":"host_branch","condition":{"allocation":2,"offset":0},
                                                 "if_zero":"zero","if_nonzero":"nonzero"});
        assert!(check(value.clone()).is_ok());
        let mut missing = value.clone();
        missing["blocks"]["begin"][1]["if_nonzero"] = json!("absent");
        assert!(check(missing).unwrap_err().contains("unknown host branch block"));
        let mut cycle = value.clone();
        cycle["blocks"]["zero"][0] = value["blocks"]["begin"][1].clone();
        assert!(check(cycle).unwrap_err().contains("cyclic host branch"));
        value["blocks"]["begin"][1]["condition"]["offset"] = json!(4);
        assert!(check(value).unwrap_err().contains("outside allocation"));
    }

    #[test]
    fn rk_control_checks_live_configuration_and_all_loop_blocks() {
        let mut value = candidate();
        value["host_allocations"][0]["bytes"] = json!(32);
        let s = |offset| json!({"allocation":2,"offset":offset});
        value["blocks"]["begin"][0] = json!({"kind":"rk_schedule", "dt_f64":s(0), "order_i64":s(8),
            "acoustic_substeps_i64":s(16), "dynamics_splits_i64":s(24), "body":"rk"});
        value["blocks"]["rk"] = json!([{"kind":"rk_splits","body":"stages"}]);
        value["blocks"]["stages"] = json!([{"kind":"rk_stages","body":"small"}]);
        value["blocks"]["small"] = json!([{"kind":"rk_substeps","body":"work"}]);
        value["blocks"]["work"] = json!([{"kind":"host_boundary","name":"update_state_time"}]);
        assert!(check(value.clone()).is_ok());
        let mut bad = value.clone();
        bad["blocks"]["begin"][0]["dt_f64"]["offset"] = json!(25);
        assert!(check(bad).unwrap_err().contains("outside allocation"));
        let mut cycle = value.clone();
        cycle["blocks"]["work"] = json!([{"kind":"rk_stages","body":"rk"}]);
        assert!(check(cycle).unwrap_err().contains("cyclic host branch"));
        let mut missing = value;
        missing["blocks"]["rk"][0]["body"] = json!("absent");
        assert!(check(missing).unwrap_err().contains("unknown host branch block"));
    }

    #[test]
    fn vinterp_requires_in_bounds_live_shape_words_and_single_stream() {
        let mut value = candidate();
        value["host_allocations"][0]["bytes"] = json!(40);
        let s = |offset| json!({"allocation":2,"offset":offset});
        value["blocks"]["begin"][0] = json!({"kind":"vinterp_ozone","stream":0,"module":"lw",
            "pmid":s(0),"pin":s(0),"ozone":s(0),"starts":s(0),"upper":s(0),"out":s(0),
            "pmid_shape":[s(0),s(8)],"pin_size":s(16),"ozone_shape":[s(24),s(32)]});
        assert!(check(value.clone()).is_ok());
        value["blocks"]["begin"][0]["ozone_shape"][1]["offset"] = json!(33);
        assert!(check(value.clone()).unwrap_err().contains("outside allocation"));
        value["blocks"]["begin"][0]["ozone_shape"][1]["offset"] = json!(32);
        value["blocks"]["begin"][0]["stream"] = json!(1);
        assert!(check(value).unwrap_err().contains("nonzero stream"));
    }

    #[test]
    fn radiation_unpack_checks_both_live_views_and_fill_sources() {
        let mut value = candidate();
        value["host_allocations"][0]["bytes"] = json!(32);
        let s = |offset| json!({"allocation":2,"offset":offset});
        let view = json!({"target":{"allocation":1,"offset":0},"shape":[s(0)],"strides":[s(8)]});
        value["blocks"]["begin"][0] = json!({"kind":"radiation_unpack","stream":0,"module":"rad",
            "rows":[{"dst":view.clone(),"src":view,"fill_f32":null}]});
        assert!(check(value.clone()).is_ok());
        value["blocks"]["begin"][0]["rows"][0]["fill_f32"] = s(16);
        assert!(check(value.clone()).unwrap_err().contains("selection"));
        value["blocks"]["begin"][0]["rows"][0]["src"] = Value::Null;
        assert!(check(value.clone()).is_ok());
        value["blocks"]["begin"][0]["rows"][0]["fill_f32"]["offset"] = json!(30);
        assert!(check(value).unwrap_err().contains("outside allocation"));
    }

    #[test]
    fn physics_plan_rejects_unclassified_host_and_upload() {
        let mut value = candidate();
        assert!(check(value.clone()).is_ok());
        value["blocks"]["begin"][0]["name"] = json!("unknown_host_work");
        assert!(check(value).unwrap_err().contains("native ABI"));
        let mut value = candidate();
        value["blocks"]["begin"][0] = json!({"kind":"upload","stream":0,"destination":{"allocation":1,"offset":0},
            "data":{"encoding":"hex","bytes":"00","sha256":""}});
        assert!(check(value.clone()).unwrap_err().contains("immutable"));
        value["blocks"]["begin"][0]["origin"] = json!("host_output");
        assert!(check(value).unwrap_err().contains("immutable"));
    }

    #[test]
    fn physics_requires_distinct_carried_host_state_and_live_view_bounds() {
        let mut value = candidate();
        value["host_carried"] = json!([]);
        assert!(check(value).unwrap_err().contains("host carried"));
        let mut value = candidate();
        value["host_carried"] = json!([1]);
        assert!(check(value).unwrap_err().contains("carried host"));
        let mut value = candidate();
        value["host_carried"] = json!([2]);
        assert!(check(value).unwrap_err().contains("frame"));
        let mut value = candidate();
        value["physics"]["frame_state_id"] = json!(2);
        assert!(check(value).unwrap_err().contains("overlaps"));
        let mut value = candidate();
        value["host_allocations"][0]["id"] = json!(1);
        value["allocations"] = json!([{"id":1,"bytes":4,"initial":null,"overwritten_before_read":true}]);
        assert!(check(value).unwrap_err().contains("identities overlap"));
        let mut value = candidate();
        value["blocks"]["begin"].as_array_mut().unwrap().push(json!({"kind":"allocate","stream":0,
            "id":2,"bytes":4,"asynchronous":false,"overwritten_before_read":true}));
        assert!(check(value).unwrap_err().contains("identities overlap"));
        let mut value = candidate();
        value["blocks"]["begin"][0] = json!({"kind":"host_upload","stream":0,
            "destination":{"allocation":1,"offset":0},"order":"c",
            "view":{"allocation":2,"offset":0,"shape":[2],"strides":[2],"item_bytes":2}});
        assert!(check(value.clone()).is_ok());
        value["blocks"]["begin"][0]["view"]["offset"] = json!(1);
        assert!(check(value).unwrap_err().contains("exceeds storage"));
    }
    #[test]
    fn physics_plan_rejects_other_streams_and_excess_steps() {
        let mut value = candidate();
        value["streams"] = json!([{"id":7,"flags":0}]);
        assert!(check(value).unwrap_err().contains("single-stream"));
        let mut value = candidate();
        value["blocks"]["begin"][0] = json!({"kind":"clear","stream":7,
            "destination":{"allocation":1,"offset":0},"bytes":4,"value":0});
        assert!(check(value).unwrap_err().contains("nonzero stream"));
        let mut value = candidate();
        value["physics"]["max_steps"] = json!(5);
        assert!(check(value).is_err());
    }

    #[test]
    fn physics_pointer_tables_require_the_identified_live_rule() {
        let mut value = candidate();
        value["blocks"]["begin"][0] = json!({"kind":"pointer_table_upload","stream":0,
            "destination":{"allocation":1,"offset":0}, "rule":"resident_v842_species",
            "entries":[{"allocation":1,"offset":0}]});
        assert!(check(value.clone()).is_ok());
        value["blocks"]["begin"][0]["rule"] = json!("unproved");
        assert!(check(value.clone()).unwrap_err().contains("pointer table"));
        value["blocks"]["begin"][0]["rule"] = json!("resident_v842_species");
        value["blocks"]["begin"][0]["entries"] = json!([]);
        assert!(check(value).unwrap_err().contains("pointer table"));
    }

    #[test]
    fn physics_fixed_arguments_require_classified_source_and_reject_unproved_metadata() {
        let kernel = |argument: Value| json!({"kind":"kernel", "stream":0, "module":"m",
            "function":"f", "grid":[1,1,1], "block":[1,1,1], "shared_bytes":0,
            "arguments":[argument]});
        let mut value = candidate();
        value["blocks"]["begin"][0] = kernel(json!({"kind":"bytes","bytes_hex":"10"}));
        assert!(check(value.clone()).unwrap_err().contains("lacks a source assertion"));
        value["blocks"]["begin"][0]["arguments"][0]["fixed_source"] = json!({
            "class":"host_output", "source":"forward.rs", "source_sha256":"00".repeat(32), "line":1});
        assert!(check(value.clone()).unwrap_err().contains("invalid physics fixed argument"));
        value["blocks"]["begin"][0]["arguments"][0]["fixed_source"]["class"] = json!("source_literal");
        assert!(check(value.clone()).is_ok());
        value["blocks"]["begin"][0]["arguments"][0]["frame_rule"] = json!("unproved");
        assert!(serde_json::from_value::<Plan>(value).is_err());
        let mut value = candidate();
        value["blocks"]["begin"][0] = kernel(json!({"kind":"patched","bytes_hex":"0000000000000000",
            "pointers":[]}));
        assert!(check(value).unwrap_err().contains("lacks a source assertion"));
    }

    #[test]
    fn physics_fixed_argument_source_hash_and_line_are_checked() {
        let base = std::env::temp_dir().join(format!("ad-binder-fixed-test-{}", std::process::id()));
        fs::create_dir_all(&base).unwrap();
        let source = base.join("source.txt");
        let contents = b"synthetic constant 16\n";
        fs::write(&source, contents).unwrap();
        let mut value = candidate();
        value["blocks"]["begin"][0] = json!({"kind":"kernel", "stream":0, "module":"m",
            "function":"f", "grid":[1,1,1], "block":[1,1,1], "shared_bytes":0,
            "arguments":[{"kind":"bytes","bytes_hex":"10","fixed_source":{
                "class":"source_literal", "source":"source.txt", "source_sha256":digest(contents), "line":1}}]});
        let plan: Plan = serde_json::from_value(value.clone()).unwrap();
        preflight(&plan, 6).unwrap();
        verify_fixed_argument_sources(&plan, &base).unwrap();
        value["blocks"]["begin"][0]["arguments"][0]["fixed_source"]["line"] = json!(2);
        let bad_line: Plan = serde_json::from_value(value).unwrap();
        assert!(verify_fixed_argument_sources(&bad_line, &base).unwrap_err().contains("line"));
        fs::write(&source, b"changed\n").unwrap();
        assert!(verify_fixed_argument_sources(&plan, &base).unwrap_err().contains("SHA-256 mismatch"));
        fs::remove_file(source).unwrap();
        fs::remove_dir(base).unwrap();
    }

    #[test]
    fn carried_views_follow_fields_as_pool_allocations_change() {
        let mut value = candidate();
        value["carried"] = json!([77, 88]);
        let first = json!([
            {"id":77,"target":{"allocation":7,"offset":4},"bytes":8},
            {"id":88,"target":{"allocation":7,"offset":24},"bytes":4}
        ]);
        let second = json!([
            {"id":77,"target":{"allocation":8,"offset":16},"bytes":8},
            {"id":88,"target":{"allocation":7,"offset":24},"bytes":4}
        ]);
        value["carried_views"] = json!([first, second, first, second, first, second]);
        let plan: Plan = serde_json::from_value(value.clone()).unwrap();
        validate_view_schedule(&plan, 6).unwrap();
        let live = HashMap::from([
            (7, submit::Allocation { id: 7, address: 0x1000, bytes: 32 }),
            (8, submit::Allocation { id: 8, address: 0x2000, bytes: 32 }),
        ]);
        let first_views = carried_views(&plan, 1, &live).unwrap();
        let second_views = carried_views(&plan, 2, &live).unwrap();
        assert_eq!(resolve_views(&first_views, &live).unwrap()[&77].address, 0x1004);
        assert_eq!(resolve_views(&second_views, &live).unwrap()[&77].address, 0x2010);
        let mut outside = second_views.clone();
        outside[0].target.offset = 25;
        assert!(resolve_views(&outside, &live).is_err());
        assert!(carried_views(&plan, 0, &live).is_err());
        let mut overlap = value.clone();
        overlap["carried_views"][0][1]["target"]["offset"] = json!(8);
        let overlap: Plan = serde_json::from_value(overlap).unwrap();
        assert!(validate_view_schedule(&overlap, 6).is_err());
        value["carried_views"][1][0]["bytes"] = json!(4);
        let wrong_size: Plan = serde_json::from_value(value).unwrap();
        assert!(validate_view_schedule(&wrong_size, 6).is_err());
    }

    #[test]
    fn short_replay_accepts_long_reference_but_checks_later_rows() {
        let plan: Plan = serde_json::from_value(candidate()).unwrap();
        let path = std::env::temp_dir().join(format!("ad-binder-reference-test-{}.jsonl", std::process::id()));
        let row = |step: usize, id: u64| format!("{}\n", json!({"event":"state_sha256",
            "step":step, "allocation":id, "bytes":4, "sha256":"00".repeat(32)}));
        let all: String = (1..=6).map(|step| format!("{}{}{}", row(step, 1), row(step, 2), row(step, 9))).collect();
        fs::write(&path, &all).unwrap();
        assert_eq!(read_references(&path, &plan, 2).unwrap().len(), 6);
        fs::write(&path, format!("{all}{}", row(5, 1))).unwrap();
        assert!(read_references(&path, &plan, 2).is_err());
        fs::write(&path, format!("{all}{}", row(6, 99))).unwrap();
        assert!(read_references(&path, &plan, 2).is_err());
        fs::remove_file(path).unwrap();
    }
}

#[cfg(test)]
mod p1_residue_and_held_pointer_tests {
    use super::{hex_bytes, masked_digest, plan::Operation};
    use sha2::{Digest, Sha256};
    #[test]
    fn residue_mask_zeroes_only_declared_ranges() {
        let mut view = vec![7_u8; 32];
        let masked = masked_digest(&mut view, &[(8, 8), (24, 8)]).unwrap();
        let mut expect = vec![7_u8; 32];
        expect[8..16].fill(0);
        expect[24..32].fill(0);
        assert_eq!(masked, hex_bytes(&Sha256::digest(&expect)));
        assert_eq!(&view[..8], &[7_u8; 8]);
        assert!(masked_digest(&mut [0_u8; 8], &[(4, 8)]).is_err());
    }
    #[test]
    fn held_upload_ranges_carry_pointer_patches() {
        let text = r#"{"kind":"held_upload_ranges","stream":0,"data":{"encoding":"hex","bytes":"00","sha256":"00"},
            "ranges":[{"destination":{"allocation":1,"offset":0},"source_offset":0,"bytes":1}],
            "origin":"held: land (milestone 1)","pointers":[{"byte_offset":0,"target":{"allocation":2,"offset":8}}]}"#;
        let Operation::HeldUploadRanges { pointers, .. } = serde_json::from_str::<Operation>(text).unwrap()
            else { panic!("wrong operation") };
        assert_eq!((pointers.len(), pointers[0].target.allocation, pointers[0].target.offset), (1, 2, 8));
        let old = text.replace(r#","pointers":[{"byte_offset":0,"target":{"allocation":2,"offset":8}}]"#, "");
        let Operation::HeldUploadRanges { pointers, .. } = serde_json::from_str::<Operation>(&old).unwrap()
            else { panic!("wrong operation") };
        assert!(pointers.is_empty());
    }
}

#[cfg(test)]
mod p1_address_view_tests {
    use super::{boundary_digest, forward_address_view, forward_address_view_tracked, mixed_reading};
    fn words(data: &[u8]) -> Vec<u64> { data.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect() }
    fn bytes(words: &[u64]) -> Vec<u8> { words.iter().flat_map(|w| w.to_le_bytes()).collect() }

    /// hrrrv5 pkg6s allocation 100: word 2 held a pointer into allocation 430 at step 4 (forward 0x7e57862b5c00,
    /// replay 0x78ac442b5c00); a 4-byte fill in step 5 cleared its low half. The lone replay high half
    /// 0x78ac00000000 is also allocation 453's replay base, so the plain rule turned it into 453's recorded base
    /// 0x7e5722000000. With the step-4 history it becomes 0x7e5700000000, what the forward holds.
    #[test]
    fn half_cleared_pointer_keeps_the_forward_half_not_a_live_pointer() {
        let ranges = [(0x78ac_0000_0000_u64, 0x78ac_0000_0000 + 35_007_488, 0x7e57_2200_0000_u64),
                      (0x78ac_4429_6400, 0x78ac_4429_6400 + 307_712, 0x7e57_8629_6400)];
        let (low, high) = (ranges[0].0, ranges[1].1);
        let step4 = bytes(&[0, 0x78ac_442b_5c00, 0x3b]);
        let (view, history, half) = forward_address_view_tracked(&step4, &ranges, low, high, None);
        assert_eq!(words(&view.unwrap().0), vec![0, 0x7e57_862b_5c00, 0x3b]);
        assert_eq!((history.len(), half), (1, 0));
        let step5 = bytes(&[0, 0x78ac_0000_0000, 0x3b]);
        // without history: the half is read as a live pointer into 453 (the defect)
        assert_eq!(words(&forward_address_view(&step5, &ranges, low, high).unwrap().0), vec![0, 0x7e57_2200_0000, 0x3b]);
        let (view, history5, half) = forward_address_view_tracked(&step5, &ranges, low, high, Some(&history));
        assert_eq!(words(&view.unwrap().0), vec![0, 0x7e57_0000_0000, 0x3b]);
        assert_eq!(half, 1);
        // step 6: the word is unchanged, the same half rule still applies
        let (view, _, half) = forward_address_view_tracked(&step5, &ranges, low, high, Some(&history5));
        assert_eq!((words(&view.unwrap().0), half), (vec![0, 0x7e57_0000_0000, 0x3b], 1));
        // a cleared high half keeps the forward's low half
        let low_kept = bytes(&[0, 0x442b_5c00, 0x3b]);
        let (view, _, half) = forward_address_view_tracked(&low_kept, &ranges, low, high, Some(&history));
        assert_eq!((words(&view.unwrap().0), half), (vec![0, 0x862b_5c00, 0x3b], 1));
        // a new full pointer (or any other value) takes the ordinary rule, never the half rule
        let fresh = bytes(&[0, 0x78ac_0000_0010, 0x3b]);
        let (view, history6, half) = forward_address_view_tracked(&fresh, &ranges, low, high, Some(&history5));
        assert_eq!((words(&view.unwrap().0), half), (vec![0, 0x7e57_2200_0010, 0x3b], 0));
        assert_eq!(history6.get(&1), Some(&(0x78ac_0000_0010, 0x7e57_2200_0010)));
        let zero = bytes(&[0, 0, 0x3b]);
        let (view, history0, half) = forward_address_view_tracked(&zero, &ranges, low, high, Some(&history));
        assert!(view.is_none() && history0.is_empty() && half == 0);
        // P1 sealer-output-steps: the low half rewritten with data, the old high half kept (no live address)
        let rewritten = bytes(&[0, 0x78ac_3a48_cf2f, 0x3b]);
        let (view, _, half) = forward_address_view_tracked(&rewritten, &ranges, low, high, Some(&history));
        assert_eq!((words(&view.unwrap().0), half), (vec![0, 0x7e57_3a48_cf2f, 0x3b], 1));
    }

    fn sha(data: &[u8]) -> String { super::hex_bytes(&<super::Sha256 as super::Digest>::digest(data)) }

    /// a test box sealed 18-step plan, allocation 454: one stale pointer (replay 0x7c8524162514 -> forward
    /// 0x798a34162514, allocation 447) and one byte-identical data word a replay allocation happens to cover. The
    /// view fails, the raw bytes fail, the per-word reading (pointer translated, data raw) gives the forward hash.
    #[test]
    fn mixed_block_reads_each_word_its_own_way() {
        let ranges = [(0x7c32_2f00_0000_u64, 0x7c32_2f00_0000 + 0x100_0000, 0x7e57_1000_0000_u64),
                      (0x7c85_2400_0000, 0x7c85_2400_0000 + 0x100_0000, 0x798a_3400_0000)];
        let (low, high) = (ranges[0].0, ranges[1].1);
        let replay = bytes(&[0, 0x7c85_2416_2514, 0x3b, 0x7c32_2f05_fd1b]);
        let forward = bytes(&[0, 0x798a_3416_2514, 0x3b, 0x7c32_2f05_fd1b]);
        let (view, history, _) = forward_address_view_tracked(&replay, &ranges, low, high, None);
        let (view, moved) = view.unwrap();
        assert_eq!((moved, history.len()), (2, 2));
        let expected = sha(&forward);
        assert_eq!(boundary_digest(&replay, Some(&view), Some(&expected)), (sha(&view), false));
        assert_eq!(mixed_reading(&replay, &view, &expected, None), Some((expected.clone(), vec![3])));
        // nothing reads right: no match, the block still fails
        let other = bytes(&[1, 0x798a_3416_2514, 0x3b, 0x7c32_2f05_fd1b]);
        assert_eq!(mixed_reading(&replay, &view, &sha(&other), None), None);
        // the pointer word was translated from the same replay value at the last boundary: only the data word is
        // in doubt (a test box proof18c2, allocation 446 step 18: 61 translated words, most proven at step 17)
        let mut previous = super::PointerHistory::new();
        previous.insert(1, (0x7c85_2416_2514, 0x798a_3416_2514));
        assert_eq!(mixed_reading(&replay, &view, &expected, Some(&previous)), Some((expected.clone(), vec![3])));
        // a look-alike word proven nowhere and a pointer proven: reverting the pointer is never needed
        previous.insert(3, (0x7c32_2f05_fd1b, 0x7e57_1005_fd1b));
        assert_eq!(mixed_reading(&replay, &view, &expected, Some(&previous)), None);
    }

    /// hrrrv5 pkg6s allocation 449: forward data words that only look like addresses (0x790b2bed3973 ...) are
    /// byte-identical in forward and replay; one that a replay allocation happens to cover must not fail the block.
    #[test]
    fn data_that_only_looks_like_an_address_passes_on_raw_bytes() {
        let ranges = [(0x790b_2bed_0000_u64, 0x790b_2bed_0000 + 0x10000, 0x7e57_3200_0000_u64)];
        let forward = bytes(&[0, 0x790b_2bed_3973, 0x3b]);
        let (view, history, _) = forward_address_view_tracked(&forward, &ranges, ranges[0].0, ranges[0].1, None);
        let view = view.unwrap().0;
        assert_eq!(history.len(), 1);
        // the forward hash is of these raw bytes: the view would fail, the raw bytes pass
        assert_eq!(boundary_digest(&forward, Some(&view), Some(&sha(&forward))), (sha(&forward), true));
        // a real pointer: the forward held the translated word, the view passes and raw is not used
        assert_eq!(boundary_digest(&forward, Some(&view), Some(&sha(&view))), (sha(&view), false));
        // neither matches: the view hash is kept and the block fails as before
        let other = bytes(&[1, 0x790b_2bed_3973, 0x3b]);
        assert_eq!(boundary_digest(&forward, Some(&view), Some(&sha(&other))), (sha(&view), false));
        // no translated word: raw hash, not flagged
        assert_eq!(boundary_digest(&forward, None, Some(&sha(&other))), (sha(&forward), false));
    }
    #[test]
    fn rewrites_only_words_inside_live_replay_allocations() {
        // replay allocations [0x1000, 0x1100] -> recorded 0x9000 and [0x5000, 0x5010] -> recorded 0x7000
        let ranges = [(0x1000_u64, 0x1100_u64, 0x9000_u64), (0x5000, 0x5010, 0x7000)];
        let words: [u64; 6] = [0x1000, 0x10ff, 0x1100, 0x1101, 0x5008, 0x4fff];
        let data: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let (view, moved) = forward_address_view(&data, &ranges, 0x1000, 0x5010).unwrap();
        let out: Vec<u64> = view.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        assert_eq!(out, vec![0x9000, 0x90ff, 0x9100, 0x1101, 0x7008, 0x4fff]);
        assert_eq!(moved, 4);
        let data: Vec<u8> = [7_u64, 0x2000].iter().flat_map(|w| w.to_le_bytes()).collect();
        assert!(forward_address_view(&data, &ranges, 0x1000, 0x5010).is_none());
    }
}

#[cfg(test)]
mod pinned_library_tests {
    use super::*;
    fn sha(b: &[u8]) -> String { format!("{:x}", Sha256::digest(b)) }
    #[test]
    fn a_pinned_library_is_found_by_name_and_sha_then_by_family_else_refused() {
        let root = std::env::temp_dir().join(format!("hexad-libs-{}", std::process::id()));
        let (a, b) = (root.join("a"), root.join("b"));
        fs::create_dir_all(&a).unwrap(); fs::create_dir_all(&b).unwrap();
        fs::write(a.join("cub.cpython-313-x86_64-linux-gnu.so"), b"other build").unwrap();
        fs::write(b.join("libcublas.so.13"), b"pinned").unwrap();
        let gone = Path::new("/nonexistent/recorder/libcublas.so.13");
        let (p, s) = resolve_pinned_library_in(gone, &sha(b"pinned"), &[a.clone(), b.clone()]).unwrap();
        assert_eq!((p, s), (b.join("libcublas.so.13"), sha(b"pinned")));
        let cub = Path::new("/nonexistent/recorder/cub.cpython-314-x86_64-linux-gnu.so");
        let (p, s) = resolve_pinned_library_in(cub, &sha(b"recorded build"), &[a.clone()]).unwrap();
        assert_eq!((p, s), (a.join("cub.cpython-313-x86_64-linux-gnu.so"), sha(b"other build")));
        assert!(resolve_pinned_library_in(gone, &sha(b"pinned"), &[a.clone()]).is_err());
        fs::remove_dir_all(&root).unwrap();
    }
    #[test]
    fn a_spec_beside_the_kernel_set_is_read_before_the_recorded_path() {
        let root = std::env::temp_dir().join(format!("hexad-kset-{}", std::process::id()));
        fs::create_dir_all(root.join("specs/srv/origin/x")).unwrap();
        fs::write(root.join("specs/srv/origin/x/k.json"), b"{}").unwrap();
        let kernels = root.join("kernels.json");
        assert_eq!(deriv::spec_path(&kernels, "/srv/origin/x/k.json"), root.join("specs/srv/origin/x/k.json"));
        assert_eq!(deriv::spec_path(&kernels, "/srv/origin/y/k.json"), PathBuf::from("/srv/origin/y/k.json"));
        fs::remove_dir_all(&root).unwrap();
    }
}
