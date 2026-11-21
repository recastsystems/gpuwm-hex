// SPDX-License-Identifier: Apache-2.0
//! Experimental native runner. It refuses incomplete observations and does not
//! establish that a producer's command list, frame rules, or census are complete.
mod cuda;
mod live_owner;
#[allow(dead_code)] mod reverse_machinery;
#[allow(dead_code)] mod device_graph;
mod step_profile;
mod mynn_fwd_shape;
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
mod cstore;
mod step_hold;
/// mem-oocore: VRAM / host budget with host spill (--vram-budget-gb, --host-budget-gb, --spill-dir).
mod spill;
mod replay_frame;
mod rk_schedule;
mod radiation_unpack;
mod ruc;
mod seaice;
mod sfclay;
mod staging;
mod submit;
mod tempo;
mod transport_copy;
mod vinterp;
mod word_reference;
mod xorshift;
mod peer;
mod peer_shm;
/// ranks diagnostic: AD_PEER_WATCH=alloc:offset:rows:stride:n:positive (see Execution::watch_range).
fn peer_watch() -> Option<(u64, usize, usize, usize, usize, bool)> {
    static W: std::sync::OnceLock<Option<(u64, usize, usize, usize, usize, bool)>> = std::sync::OnceLock::new();
    *W.get_or_init(|| { let v = std::env::var("AD_PEER_WATCH").ok()?; let p: Vec<&str> = v.split(':').collect();
        if p.len() != 6 { return None; }
        Some((p[0].parse().ok()?, p[1].parse().ok()?, p[2].parse().ok()?, p[3].parse().ok()?, p[4].parse().ok()?, p[5] == "1")) })
}
mod ranks;
// hex-central hunt-120-start: named words of the derivative memory after every command (AD_WORD_PROBE; off by default).
mod word_probe;
// hex-central hunt-60-repeat: the backward sweep's per-step, per-allocation scale-free digest (AD_REV_DIGEST; off by default).
mod rev_digest;
// Physics-on window stage A (ad/phys-window): U6's derivative entries and launch dot, reused as is.
#[allow(dead_code, unused)] #[path = "../e2e/deriv.rs"] mod deriv;
#[allow(dead_code, unused)] #[path = "../e2e/dev.rs"] mod dev;
#[allow(dead_code, unused)] #[path = "../e2e/dot.rs"] mod dot;
#[allow(dead_code, unused)] #[path = "../e2e/expr.rs"] mod expr;
#[allow(dead_code, unused)] #[path = "../e2e/recipes.rs"] mod recipes;
#[allow(dead_code, unused)] #[path = "../e2e/topo.rs"] mod topo;
mod addot;
// ad/held-rlw_cldprmc_coalesced: the per-launch dot of an ad.entry/v1 row on one real captured launch (ad-binder capture-dot).
mod capentry;
mod seam;
// Physics-on window lane L1: composed tangent + segment-checkpointed reverse (ad-binder window).
mod window;
// resident: x0-plant (a new start planted in memory, no moved package) and --x0-planted / --resident.
mod resident;
mod wrules;
#[allow(dead_code, unused)] #[path = "../e2e/held.rs"] mod held;
mod p8bridge;
mod schemes;
mod tempo_adw;
mod column_shard;
// ring-60km: the MYNN-EDMF ring plan and the tape growth that leaves its room (pure, CPU-tested).
mod ring_room;
mod card_room;
// 60 km tape: the MYNN-EDMF transient regions sized to the launch (AD_P3_STRIPE=1; pure layout, CPU-tested).
mod p3_stripe;
mod tempo_shard;
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

/// Preserve the provenance of certified upload pointers before relocation. A
/// later four-byte store can leave their high half behind before the first
/// boundary, so boundary-only address history is insufficient.
fn patch_upload_addresses(data: &mut [u8], patches: &[plan::Patch],
                          allocations: &HashMap<u64, submit::Allocation>,
                          destination: plan::Slice) -> Result<PointerHistory, String> {
    let mut captured = Vec::with_capacity(patches.len());
    for patch in patches {
        let end = patch.byte_offset.checked_add(8).ok_or("pointer patch offset overflow")?;
        let word = data.get(patch.byte_offset..end).ok_or("pointer patch outside upload")?;
        let offset = destination.offset.checked_add(patch.byte_offset).ok_or("upload pointer offset overflow")?;
        if offset % 8 == 0 {
            captured.push((offset / 8, patch.byte_offset, u64::from_le_bytes(word.try_into().unwrap())));
        }
    }
    patch_device_addresses(data, patches, allocations)?;
    Ok(captured.into_iter().map(|(index, offset, forward)| {
        let replay = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        (index, (replay, forward))
    }).collect())
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
                    | Operation::DescriptorTableUpload { stream, .. }
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
                        // continuation 17:40Z: [int(a.data.ptr) for a in arrays] as uint64, in order:
                        // cuda_physics_prep_v841.py:575-587 _device_pointer_table; rrfs/driver_rrfs.py:2657.
                        | "cuda_physics_prep_device_pointer_table" | "rrfs_stage_sources_rate_table") || entries.is_empty() {
                        return Err("unproved physics pointer table rule".into());
                    }
                }
                // S01: the only descriptor table rule is the limited-area scrub (5 words a row: address, rows, stride,
                // solve, pool bits); anything else is unproved.
                if let Operation::DescriptorTableUpload { rule, rows, .. } = command {
                    if rule != "solve_region_scrub_descriptors" || rows.is_empty() || rows.iter().any(|r| r.words.len() != 4) {
                        return Err("unproved descriptor table rule".into());
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
/// Package-local CIndexer proofs (ad-record, ad/host-recorder): the plan directory's `cindexer-pairs.json`, written by
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
    borrowed_context: bool,
    borrowed_allocations: HashSet<u64>,
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
    /// speed (storm): the digest arena is with a background boundary digest (a tangent step that keeps no host
    /// copy, see digest_arena_defer); its thread sends the arena back here when its readings are done.
    digest_arena_back: Vec<std::sync::mpsc::Receiver<pinned::PinnedBuf>>,
    /// speed (storm, round 3): set by the checkpointed reverse sweep (window.rs sweep_revolve), see digest_arenas_max.
    pub(crate) digest_lazy: bool,
    /// speed (storm, round 3): the pointer history of the boundary last restored while a background digest was
    /// still running (restore_history_lazy): the next boundary digest reads with it, in place of the last digest's.
    history_from: Option<HistSnap>,
    /// speed (storm, round 3): a background boundary digest found a boundary that differs; the next step end or
    /// restore joins the chain and refuses with that boundary's text.
    digest_failed: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// speed (storm, round 3): recomputed boundaries whose hash ran beside the card (window.json checkpointing).
    pub(crate) recompute_digests_beside: u64,
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
    staging_ahead: Option<staging::Ahead>,
    staging_prepared: bool,
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
    /// bl-forward wire (AD_MYNN_FWD_LANES): the workspace relayout kernel (module, function) and its scratch
    /// (pointer, bytes), made at the first re-shaped launch; see mynn_fwd_shape.rs "the workspace layout".
    mynn_relay: Option<(*mut c_void, *mut c_void, u64, usize)>,
    /// bl-forward wire: the clear the previous command made (allocation, offset, bytes), for the skip of the move before.
    mynn_last_clear: Option<(u64, usize, usize)>,
    /// bl-forward wire: launches relayed, moves before skipped (clear just before), moves before run (receipt only).
    mynn_relays: [u64; 3],
}

/// P1 (ad/phys): the carried bytes as the forward held them: every aligned 8-byte word pointing into a live
/// replay allocation [address, address + bytes] becomes the recorded base of that allocation plus the offset.
/// `ranges` is sorted by start: (start, end inclusive, recorded base). Returns None when no word moved.
/// P1 (ad/phys): SHA-256 of a forward-address view with the declared residue ranges zeroed.
// rank-proof reference-payload reading uses this outside tests (ad/final merge)
fn masked_digest(view: &mut [u8], zeroed: &[(usize, usize)]) -> Result<String, String> {
    for &(offset, bytes) in zeroed {
        let end = offset.checked_add(bytes).ok_or("residue range overflow")?;
        view.get_mut(offset..end).ok_or("residue range outside its allocation")?.fill(0);
    }
    Ok(hex_bytes(&Sha256::digest(&*view)))
}

/// F1: the same masked bytes, read without making another whole-boundary copy.
/// Validate in declaration order to preserve the first error, then hash each
/// unmasked slice and the union of zero ranges. No input or history is mutated.
fn masked_digest_reading(view: &[u8], zeroed: &[(usize, usize)]) -> Result<String, String> {
    let mut spans = Vec::with_capacity(zeroed.len());
    for &(offset, bytes) in zeroed {
        let end = offset.checked_add(bytes).ok_or("residue range overflow")?;
        view.get(offset..end).ok_or("residue range outside its allocation")?;
        if offset != end { spans.push((offset, end)); }
    }
    if spans.is_empty() { return Ok(hex_bytes(&Sha256::digest(view))); }
    spans.sort_unstable();
    static ZERO_BYTES: [u8; 64 << 10] = [0; 64 << 10];
    let mut hash = Sha256::new();
    let (mut cursor, mut i) = (0, 0);
    while i < spans.len() {
        let (start, mut end) = spans[i];
        i += 1;
        while i < spans.len() && spans[i].0 <= end {
            end = end.max(spans[i].1);
            i += 1;
        }
        hash.update(&view[cursor..start]);
        let mut left = end - start;
        while left > 0 {
            let n = left.min(ZERO_BYTES.len());
            hash.update(&ZERO_BYTES[..n]);
            left -= n;
        }
        cursor = end;
    }
    hash.update(&view[cursor..]);
    Ok(hex_bytes(&hash.finalize()))
}

fn address_range_prefilter_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    { std::is_x86_feature_detected!("avx2") }
    #[cfg(not(target_arch = "x86_64"))]
    { false }
}

/// F1: conservative unsigned envelope test for exactly 32 words. A false result
/// proves every ordinary translation would continue before inspecting ranges.
/// Half-history readings are checked separately by the caller before skipping.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn address_range_may_contain_pointer(data: &[u8], low: u64, high: u64) -> bool {
    use std::arch::x86_64::*;
    debug_assert_eq!(data.len(), 256);
    if low > high { return false; }
    let sign = _mm256_set1_epi64x(i64::MIN);
    let lo = _mm256_set1_epi64x((low ^ (1u64 << 63)) as i64);
    let hi = _mm256_set1_epi64x((high ^ (1u64 << 63)) as i64);
    let ones = _mm256_set1_epi64x(-1);
    let mut possible = _mm256_setzero_si256();
    for offset in (0..256).step_by(32) {
        let word = _mm256_xor_si256(_mm256_loadu_si256(data.as_ptr().add(offset).cast()), sign);
        let outside = _mm256_or_si256(_mm256_cmpgt_epi64(lo, word), _mm256_cmpgt_epi64(word, hi));
        possible = _mm256_or_si256(possible, _mm256_andnot_si256(outside, ones));
    }
    _mm256_movemask_epi8(possible) != 0
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn address_range_may_contain_pointer(_: &[u8], _: u64, _: u64) -> bool { true }

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
/// Bytes 5..7 of a word: what is left of a device address when its five low bytes are rewritten.
const TOP_BYTES: u64 = 0xffff_ff00_0000_0000;

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
    // F1 completion: history is sparse, but the former loop probed its hash
    // table once per word (153 million probes for allocation429). Determine
    // the history-only readings once, then merge them into the ordered scan.
    // Each reading depends only on this immutable word and its previous pair.
    let mut kept = Vec::new();
    let mut noted: Vec<(usize, u64, u64, u64)> = Vec::new();
    if let Some(previous) = previous {
        for (&index, &(replay, forward)) in previous {
            let Some(offset) = index.checked_mul(8) else { continue };
            let Some(end) = offset.checked_add(8) else { continue };
            let Some(chunk) = data.get(offset..end) else { continue };
            let word = u64::from_le_bytes(chunk.try_into().unwrap());
            // S01b (3 km LAM, 2026-11-07; ported onto the sparse-history scan): the word still holds the noted replay
            // pointer (the value a runner-built table wrote, or the last boundary translated), so it reads as the noted
            // forward address, whether or not a live replay allocation covers it now. The live-range rule reads a covered
            // word through whichever allocation the pool placed there: when the table entry's own array was freed and
            // another took its replay address, that is a different forward address. THE BREAKAGE THIS PREVENTS: pkg60lam
            // (3 km LAM) strict replay stopped at step 3 (allocation 419, solve_region_scrub descriptor pointer words),
            // and the failing step changed with where the replay's pool placed its arrays.
            if word == replay { noted.push((index, forward, replay, forward)); continue; }
            let reading = if word == replay { None }
                else if word == replay & HIGH_HALF && replay & !HIGH_HALF != 0 { Some(forward & HIGH_HALF) }
                else if word == replay & !HIGH_HALF && replay & HIGH_HALF != 0 { Some(forward & !HIGH_HALF) }
                // P1 sealer-output-steps: low half rewritten by a 4-byte data write, high half still the old
                // pointer's (and the word is no live replay address, which takes the ordinary rule below). Breakage:
                // the sealed 18-step hrrrv5 plan, strict replay, step 10, allocation 142 word 93: forward
                // 0x7e573a48cf2f, replay 0x77b53a48cf2f (both the old pointer's high half under the same 4 data bytes).
                else if word & HIGH_HALF == replay & HIGH_HALF && replay & HIGH_HALF != 0
                    && !inside_live(word, ranges, low, high) { Some((forward & HIGH_HALF) | (word & !HIGH_HALF)) }
                // longwin (2026-11-10): FIVE low bytes rewritten, bytes 5..7 still the old pointer's (and the word is
                // no live replay address). A byte array that ends inside the word does this: a 21-element bool array
                // written over a dead pointer table covers two words and five bytes of the third. Byte 5 is the top byte
                // of a device address (0x76.. in one process, 0x77.. in the next), so the word reads as the forward's
                // bytes 5..7 over the same five data bytes. THE BREAKAGE THIS PREVENTS: the 30-step 120 km package
                // pk120e30p-20230331, every window with the TEMPO derivative live: "state hash mismatch at step 9, 1
                // allocations, first 194" (word 1794: forward 0x770000000000, replay 0x760000000000 over the old land
                // table pointer 0x76e571a40c00 -> 0x778431a40c00), and allocation 202 word 2 at step 14. The model state
                // was equal at all 30 boundaries; the hash moved with where the process put its device memory, which
                // is why the same package passed with the TEMPO derivative held (top byte 0x77 in that process).
                // Shown on the two dumps without a card (lw_digest_emul.py: the rule as it was gives the failing
                // hashes 731a30db.. and fb9bfebe.., with this reading the recorded 45f6e4d0.. and f08df531..).
                else if word & TOP_BYTES == replay & TOP_BYTES && replay & TOP_BYTES != 0
                    && !inside_live(word, ranges, low, high) { Some((forward & TOP_BYTES) | (word & !TOP_BYTES)) }
                else { None };
            if let Some(value) = reading { kept.push((index, value, replay, forward)); }
        }
    }
    kept.sort_unstable_by_key(|x: &(usize, u64, u64, u64)| x.0);
    noted.sort_unstable_by_key(|x: &(usize, u64, u64, u64)| x.0);
    let prefilter = address_range_prefilter_available();
    // machine: every word is read on its own (this immutable word, its previous pair, the immutable ranges), so the
    // blocks are scanned in ranges on several threads for a large allocation and the per-range lists applied after.
    // THE BREAKAGE THIS FIXES: one thread scanned a 2.1 GB carried block for 1.6-2.0 s and copied it whole before the
    // translated bytes could be hashed (a test machine machine pair1-b, host_digest_prof: allocation 436 view 1.6-2.0 s of
    // a 2.5-2.9 s item), which is what a tangent pass's end waits for once the gathers are quick. The same words are
    // rewritten to the same values and the same history is returned. AD_DIGEST_SCAN_THREADS=1 keeps one thread.
    let scan = |b0: usize, b1: usize| -> (Vec<(usize, u64)>, Vec<(usize, (u64, u64))>, usize, usize) {
        let (mut writes, mut hist, mut moved, mut halves) = (Vec::new(), Vec::new(), 0_usize, 0_usize);
        let mut next_kept = kept.partition_point(|x| x.0 < b0 * 32);
        let mut next_noted = noted.partition_point(|x| x.0 < b0 * 32);
        for block_index in b0..b1 {
            let block = &data[block_index * 256..((block_index + 1) * 256).min(data.len())];
            let first = block_index * 32;
            if prefilter && block.len() == 256
                && kept.get(next_kept).map_or(true, |x| x.0 >= first + 32)
                && noted.get(next_noted).map_or(true, |x| x.0 >= first + 32)
                && !unsafe { address_range_may_contain_pointer(block, low, high) } {
                continue;
            }
            for (within, chunk) in block.chunks_exact(8).enumerate() {
                let index = first + within;
                let word = u64::from_le_bytes(chunk.try_into().unwrap());
                if let Some(&(at, value, replay, forward)) = noted.get(next_noted) {
                    if at == index {
                        if value != word { writes.push((index, value)); }
                        hist.push((index, (replay, forward)));
                        moved += 1;
                        next_noted += 1;
                        continue;
                    }
                }
                if let Some(&(at, value, replay, forward)) = kept.get(next_kept) {
                    if at == index {
                        if value != word { writes.push((index, value)); }
                        hist.push((index, (replay, forward)));
                        halves += 1;
                        next_kept += 1;
                        continue;
                    }
                }
                if word < low || word > high { continue; }
                let k = ranges.partition_point(|r| r.0 <= word);
                if k == 0 { continue; }
                let (start, end, recorded) = ranges[k - 1];
                if word > end { continue; }
                let forward = recorded + (word - start);
                writes.push((index, forward));
                hist.push((index, (word, forward)));
                moved += 1;
            }
        }
        (writes, hist, moved, halves)
    };
    let blocks = data.len().div_ceil(256);
    let threads = if data.len() < DIGEST_SCAN_PAR_BYTES { 1 } else { digest_scan_threads().min(blocks).max(1) };
    let parts: Vec<(Vec<(usize, u64)>, Vec<(usize, (u64, u64))>, usize, usize)> = if threads == 1 { vec![scan(0, blocks)] } else {
        let per = blocks.div_ceil(threads);
        std::thread::scope(|sc| {
            let scan = &scan;
            let hs: Vec<_> = (0..threads).map(|t| sc.spawn(move || scan((t * per).min(blocks), ((t + 1) * per).min(blocks)))).collect();
            hs.into_iter().map(|h| h.join().expect("address scan worker panicked")).collect()
        })
    };
    if parts.iter().any(|x| !x.0.is_empty()) {
        let mut bytes = vec![0_u8; data.len()];
        if threads == 1 { bytes.copy_from_slice(data); } else {
            let piece = data.len().div_ceil(threads).max(1);
            std::thread::scope(|sc| { for (d, s) in bytes.chunks_mut(piece).zip(data.chunks(piece)) { sc.spawn(move || d.copy_from_slice(s)); } });
        }
        for (writes, _, _, _) in &parts {
            for &(index, value) in writes { bytes[8 * index..8 * index + 8].copy_from_slice(&value.to_le_bytes()); }
        }
        out = Some(bytes);
    }
    for (_, hist, m, h) in parts {
        for (index, pair) in hist { history.insert(index, pair); }
        moved += m; halves += h;
    }
    (out.map(|bytes| (bytes, moved)), history, halves)
}
/// machine: blocks at least this large get the threaded address scan (see forward_address_view_tracked).
const DIGEST_SCAN_PAR_BYTES: usize = 64 << 20;
fn digest_scan_threads() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("AD_DIGEST_SCAN_THREADS").ok().and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8)).max(1))
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
    boundary_digest_pre2(data, view, expected, raw, None)
}
/// machine: boundary_digest_pre with the translated view's sha256 already computed too (`translated_pre`, hashed
/// beside the raw bytes of a large block); the same readings in the same order, so the same (hash, raw) result.
fn boundary_digest_pre2(data: &[u8], view: Option<&[u8]>, expected: Option<&str>, raw: Option<&str>, translated_pre: Option<&str>) -> (String, bool) {
    let raw_hash = || raw.map(str::to_owned).unwrap_or_else(|| hex_bytes(&Sha256::digest(data)));
    let Some(view) = view else { return (raw_hash(), false) };
    let translated = translated_pre.map(str::to_owned).unwrap_or_else(|| hex_bytes(&Sha256::digest(view)));
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
fn residue_masked_reading(data: &[u8], translated_view: Option<&[u8]>, ranges: &[(u64, u64, u64)], low: u64, high: u64, previous: Option<&PointerHistory>,
                          zeroed: &[(usize, usize)], masked_expected: &str, history: &mut PointerHistory)
                          -> Result<(String, Option<usize>), String> {
    residue_masked_reading_first(data, translated_view, ranges, low, high, previous, zeroed, masked_expected, history, &mut None)
}
/// speed (storm, round 3): residue_masked_reading, which also says in `first_missed` whether its first readings
/// both missed the recorded masked hash: the masked hash of the translated view (of the raw bytes when there is no
/// view), and, when that one missed and there is a view, the masked hash of the raw bytes. That is word for word the
/// test digest_blocks makes before it opens a recorded payload (`needs` there), which used to copy and hash the block
/// again for each of the two. Nothing else changes: same readings, same order, same result and history.
#[allow(clippy::too_many_arguments)]
fn residue_masked_reading_first(data: &[u8], translated_view: Option<&[u8]>, ranges: &[(u64, u64, u64)], low: u64, high: u64, previous: Option<&PointerHistory>,
                                zeroed: &[(usize, usize)], masked_expected: &str, history: &mut PointerHistory,
                                first_missed: &mut Option<bool>)
                                -> Result<(String, Option<usize>), String> {
    // F1: digest_blocks already normalized these exact bytes with this boundary's
    // immutable ranges/history. Reuse that reading; do not scan/copy GB blocks a
    // second time. Hash declared zero ranges without copying this immutable view.
    // speed storm3: a large block with a view takes its raw reading on a second thread beside the first one
    // (it was taken after it when the first missed: 0.7 s more of one thread for the 1.23 GB block 189 of the 3 km
    // storm box). It is used exactly where it was taken before, so the readings and their order are the same.
    let (first, raw_ahead) = match translated_view {
        Some(view) if data.len() >= DIGEST_OVERLAP_BYTES => std::thread::scope(|inner| {
            let raw = inner.spawn(|| masked_digest_reading(data, zeroed));
            let first = masked_digest_reading(view, zeroed);
            (first, Some(raw.join().unwrap()))
        }),
        _ => (masked_digest_reading(translated_view.unwrap_or(data), zeroed), None),
    };
    let mut masked = first?;
    // same raw-bytes fallback as boundary_digest: a byte-identical block passes on its raw bytes
    if masked != *masked_expected && translated_view.is_some() {
        let raw = match raw_ahead { Some(raw) => raw?, None => masked_digest_reading(data, zeroed)? };
        if raw == *masked_expected {
            masked = raw;
            // addr-s: the block was accepted on its raw bytes, so no word outside the left-out part is an address:
            // the notes the default view took there go. THE BREAKAGE THIS PREVENTS: 3 km block 192, a cell-edge
            // pair (two 32-bit cell numbers) inside a live replay block at step 1 kept its note, was read as the
            // noted address at step 2, and the block failed although every byte equalled the recording.
            history.retain(|index, _| zeroed.iter().any(|&(start, len)| 8 * index >= start && 8 * index + 8 <= start + len));
            // a development line 04b8a98f asks the same of this place with `history.clear()`; this line drops every note outside the
            // left-out part, which is what STORM's tests residue_raw_pass_leaves_no_history and (3ad1b788)
            // residue_raw_pass_refuses_one_wrong_data_word ask, and keeps the notes inside it (a development line 3 km block 192).
        }
    }
    // speed (storm, round 3): here `masked` is the recorded hash exactly when one of the two first readings gave it
    *first_missed = Some(masked != *masked_expected);
    // step6-rucland: the same half-word readings (subsets) as every other block (allocation 394, step 18).
    let mut found = None;
    if masked != *masked_expected {
        let alternates = ambiguous_half_words(data, ranges, low, high, previous);
        let n = alternates.len().min(12);
        // speed storm3: no copy of the block (one was made here even with no candidate word, and one more per
        // subset): the same subsets in the same order, hashed with the left-out bytes read as zero (first_mask_reading)
        let base = translated_view.unwrap_or(data);
        let mut masks: Vec<u32> = (1..(1u32 << n)).collect();
        masks.sort_by_key(|m| (std::cmp::Reverse(*m == (1u32 << n) - 1), m.count_ones(), *m));
        if !masks.is_empty() {
            let spans = zero_spans(base.len(), zeroed)?;
            let words: Vec<(usize, [u8; 8], [u8; 8])> = alternates.iter().take(n).map(|a| (a.0,
                zero_word(a.1.to_le_bytes(), a.0, &spans), zero_word(base[8 * a.0..8 * a.0 + 8].try_into().unwrap(), a.0, &spans))).collect();
            if let Some(mask) = first_mask_reading(base, &spans, &words, &masks, masked_expected) {
                let chosen: Vec<&(usize, u64)> = alternates.iter().take(n).enumerate()
                    .filter(|(k, _)| mask >> k & 1 == 1).map(|(_, a)| a).collect();
                masked = masked_expected.to_string();
                // keep the old pointer for the next boundary, as the unmasked path does
                for (index, _) in &chosen {
                    if let Some(entry) = previous.and_then(|p| p.get(index)) { history.insert(*index, *entry); }
                }
                found = Some(chosen.len());
            }
        }
    }
    // Rank-8 pk120, step 1, allocation 3598: 33 actual pointer-table words
    // need relocation, while two unchanged data words (0x7ce600000000 at
    // byte offsets 43760824 and 60771800) happen to lie in a replay allocation.
    // Neither the all-translated nor raw masked reading can match. Apply the
    // same proven/mixed reading as ordinary blocks, with only the declared
    // residue bytes zeroed. Never enlarge the exclusion or change its digest.
    if masked != *masked_expected {
        if let Some(translated) = translated_view {
            // speed storm3: no zero-filled copies of the block and no hash of each that was thrown away (two
            // 1.23 GB copies and two passes of one thread for block 189 of the 3 km storm box at every boundary whose
            // first two readings missed): mixed_reading_zeroed reads the bytes as they stand, left-out bytes as zero.
            if let Some((hashed, back)) = mixed_reading_zeroed(data, translated, zeroed, masked_expected, previous, Some(history))? {
                masked = hashed;
                for index in back { history.remove(&index); }
            }
        }
    }
    // addr-s: the one last reading (any_reading): every admissible reading of every word, decided by the recorded hash.
    if masked != *masked_expected {
        let (read, cost) = any_reading(data, translated_view.unwrap_or(data), ranges, low, high, previous, zeroed, masked_expected);
        println!("{}", serde_json::json!({"event":"address_any_reading","masked":true,"bytes":data.len(),
            "words_with_more_than_one_reading":cost.words,"hashes_tried":cost.hashes,"hash_bound":cost.bound,
            "matched_recorded_sha256":read.is_some(),"seconds":cost.seconds,"matched_by":cost.stage}));
        if let Some(words) = read {
            masked = masked_expected.to_string();
            for (index, _, note) in words { history.remove(&index); if let Some(entry) = note { history.insert(index, entry); } }
        }
    }
    Ok((masked, found))
}

/// A recorded payload chooses between existing address readings, never supplies
/// replacement model data. Every byte remains raw, relocated by a live allocation,
/// or the existing history half-word rule. The original hash authenticates the
/// payload before it can guide a reading. This removes the subset-count limit.
struct ReferenceReading {
    value: String, masked: String, history: PointerHistory,
    moved: usize, halves: usize, raw: bool,
}
#[allow(clippy::too_many_arguments)]
fn reference_reading(data: &[u8], viewed: &[u8], reference: &[u8], expected: &str,
                     zeroed: &[(usize, usize)], masked_expected: &str,
                     previous: Option<&PointerHistory>, current: &PointerHistory,
                     alternates: &[(usize, u64)]) -> Result<Option<ReferenceReading>, String> {
    if data.len() != viewed.len() || data.len() != reference.len() { return Err("reference reading size mismatch".into()); }
    verify_sha(reference, expected)?;
    let mut target = reference.to_vec();
    if masked_digest(&mut target, zeroed)? != masked_expected { return Err("reference reading residue hash mismatch".into()); }
    let mut raw = data.to_vec(); masked_digest(&mut raw, zeroed)?;
    let mut normal = viewed.to_vec(); masked_digest(&mut normal, zeroed)?;
    let mut output = viewed.to_vec(); let mut history = current.clone();
    let half: HashMap<usize, u64> = alternates.iter().copied().collect();
    let mut halves = 0;
    for index in 0..data.len() / 8 {
        let lo = 8 * index; let hi = lo + 8;
        // A residue does not prove any part of an overlapping pointer word.
        let covered = zeroed.iter().any(|&(start, bytes)| start < hi && lo < start.saturating_add(bytes));
        if normal[lo..hi] == target[lo..hi] {
            if covered { history.remove(&index); }
            continue;
        }
        let word = u64::from_le_bytes(data[lo..hi].try_into().unwrap());
        // Zero retains no address bits. A pointer on a 4 GiB boundary has a
        // zero low half, so clearing the whole word also looks like clearing
        // just its high half. Keep both readings until the payload decides.
        let proven = word != 0 && (previous.and_then(|p| p.get(&index)).is_some_and(|&(r, _)| r == word)
            || current.get(&index).is_some_and(|&(r, _)| r != word));
        if !proven && raw[lo..hi] == target[lo..hi] {
            output[lo..hi].copy_from_slice(&data[lo..hi]); history.remove(&index); continue;
        }
        if let Some(value) = half.get(&index) {
            let mut bytes = value.to_le_bytes();
            for &(start, count) in zeroed {
                let end = start.saturating_add(count);
                for at in lo.max(start)..hi.min(end) { bytes[at - lo] = 0; }
            }
            if bytes[..] == target[lo..hi] {
                output[lo..hi].copy_from_slice(&value.to_le_bytes());
                if let Some(entry) = previous.and_then(|p| p.get(&index)) { history.insert(index, *entry); }
                if covered { history.remove(&index); }
                halves += 1; continue;
            }
        }
        return Ok(None);
    }
    // Tail bytes cannot be read as aligned pointers.
    let tail = data.len() / 8 * 8;
    if normal[tail..] != target[tail..] { return Ok(None); }
    let value = hex_bytes(&Sha256::digest(&output));
    let moved = output.chunks_exact(8).zip(data.chunks_exact(8)).filter(|(a,b)| a != b).count();
    let masked = masked_digest(&mut output, zeroed)?;
    if masked != masked_expected { return Ok(None); }
    Ok(Some(ReferenceReading { value, masked, history, moved, halves, raw: moved == 0 }))
}

/// P1 a test machine (8x5090, 2026-11-04): most words in doubt (changed, not proven by the last boundary) a mixed block
/// may hold for the per-word reading.
const MIXED_MAX_WORDS: usize = 6;
/// Most doubtful words for the complement search (keep 0, 1 or 2 translated, revert the rest): 64 words, 2,081 hashes.
const MIXED_MAX_COMPLEMENT: usize = 64;

/// F220: what a complement search hashed, in bytes: in all, and along its longest sequential chain (the sum over
/// its phases of the largest single task, so its wall time with a free core per task).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct SearchCost { hashed: u64, longest: u64 }

/// F220: `task` for each of 0..count on every core (lowest index first, taken as workers free up); results in index order.
fn par_map<T: Send>(count: usize, task: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(8).min(count).max(1);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let parts: Vec<Vec<(usize, T)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers).map(|_| scope.spawn(|| {
            let mut mine = Vec::new();
            loop {
                let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if index >= count { break mine; }
                mine.push((index, task(index)));
            }
        })).collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut out: Vec<Option<T>> = (0..count).map(|_| None).collect();
    for (index, value) in parts.into_iter().flatten() { out[index] = Some(value); }
    out.into_iter().map(|v| v.unwrap()).collect()
}

/// The complement search of `mixed_reading`. `base` is the block with every doubtful word raw and every proven
/// word translated; the readings are keep none, then for each a: keep {a}, keep {a, b} for b > a (doubtful word
/// a, b translated from `viewed`). Returns the reverted doubtful words of the first reading that gives `expected`.
/// F220: a reading differs from the base only from its first kept word on, so the SHA-256 state at each doubtful
/// word is kept (one pass over the base), each keep {a} starts from the state at a (and keeps the state at every
/// later doubtful word), and each keep {a, b} starts from that state at b: only the tail after the first kept word
/// is hashed, and the a's, then the pairs, run on every core. Before, each of the 1 + n + n(n-1)/2 readings hashed
/// the whole block on one thread (allocation 454, 89.6 MB, 64 doubtful words, no match: 2,081 passes, ~186 GB).
/// The answer is the same reading as before: every reading earlier in the old order than a match is still hashed.
fn complement_search(base: &[u8], viewed: &[u8], doubt: &[usize], expected: &str) -> (Option<Vec<usize>>, SearchCost) {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    let n = doubt.len();
    debug_assert!(n <= 127 && doubt.windows(2).all(|w| w[0] < w[1]));
    let len = base.len() as u64;
    let reverted = |keep: &[usize]| -> Vec<usize> { doubt.iter().copied().filter(|i| !keep.contains(i)).collect() };
    let word = |i: usize| 8 * i..8 * i + 8;
    // keep none, one pass: the state at each doubtful word is the start of every reading whose first kept word it is.
    // Breakage it reads: allocation 454 step 3, 1 real pointer proven by the held pointer writes of the step, 7
    // doubtful lookalikes, all of them data.
    let mut hasher = Sha256::new();
    let mut at = Vec::with_capacity(n);
    let mut position = 0;
    for &i in doubt { hasher.update(&base[position..8 * i]); position = 8 * i; at.push(hasher.clone()); }
    hasher.update(&base[position..]);
    let mut cost = SearchCost { hashed: len, longest: len };
    if hex_bytes(&hasher.finalize()) == expected { return (Some(doubt.to_vec()), cost); }
    // the old order: (a, a) is keep {a}, (a, b) with b > a keep {a, b}, a first; the earliest match wins
    let order = |a: usize, b: usize| (a * 128 + b) as u64;
    let best = AtomicU64::new(u64::MAX);
    let singles: Vec<(Vec<Sha256>, u64)> = par_map(n, |a| {
        let mut h = at[a].clone();
        h.update(&viewed[word(doubt[a])]);
        let mut position = 8 * doubt[a] + 8;
        let mut states = Vec::with_capacity(n - a - 1);
        for &i in &doubt[a + 1..] { h.update(&base[position..8 * i]); position = 8 * i; states.push(h.clone()); }
        h.update(&base[position..]);
        if hex_bytes(&h.finalize()) == expected { best.fetch_min(order(a, a), Relaxed); }
        (states, len - 8 * doubt[a] as u64)
    });
    // longest tails first (smallest b)
    let mut pairs: Vec<(usize, usize)> = (0..n).flat_map(|a| (a + 1..n).map(move |b| (a, b))).collect();
    pairs.sort_by_key(|&(a, b)| (b, a));
    let tails: Vec<u64> = par_map(pairs.len(), |k| {
        let (a, b) = pairs[k];
        if order(a, b) > best.load(Relaxed) { return 0; }
        let mut h = singles[a].0[b - a - 1].clone();
        h.update(&viewed[word(doubt[b])]);
        h.update(&base[8 * doubt[b] + 8..]);
        if hex_bytes(&h.finalize()) == expected { best.fetch_min(order(a, b), Relaxed); }
        len - 8 * doubt[b] as u64
    });
    cost.hashed += singles.iter().map(|s| s.1).sum::<u64>() + tails.iter().sum::<u64>();
    cost.longest += singles.iter().map(|s| s.1).max().unwrap_or(0) + tails.iter().copied().max().unwrap_or(0);
    let found = match best.into_inner() {
        u64::MAX => None,
        code => {
            let (a, b) = ((code / 128) as usize, (code % 128) as usize);
            Some(if a == b { reverted(&[doubt[a]]) } else { reverted(&[doubt[a], doubt[b]]) })
        }
    };
    (found, cost)
}

/// speed storm3 (2026-11-11): the first mask, in the order given, whose reading of `base` hashes to `expected`.
/// A reading is `base` with, for each set bit b of the mask, the 8 bytes of word `words[b].0` replaced by `words[b].1`.
/// THE WASTE THIS REMOVES: the subset searches of mixed_reading and of the half-word rule hashed the whole block once
/// per subset on one thread (up to 63 and 4,095 passes), and the boundary checks run one after another in step order,
/// so a block that needed many subsets held up every later boundary: in the 59-step 3 km storm window block 153
/// (128.6 MB, 8 to 10 data words that look like replay addresses, at every boundary from step 16) kept one thread in
/// SHA-256 for seconds at a time while the card sat idle, and 13 to 17 of the 59 forward-sensitivity steps took 4 to
/// 14 s instead of 2.3 (speed/fast/storm3 o59-before, prof/before.txt). Every reading is the same up to the first
/// candidate word, so that part is hashed once and each reading hashes the rest only; the readings run on every core.
/// The answer is the one the one-thread loop gave: masks are handed out in the given order, a worker stops only at a
/// mask after the earliest match so far, and the earliest match wins.
fn first_mask_reading(base: &[u8], spans: &[(usize, usize)], words: &[(usize, [u8; 8], [u8; 8])], masks: &[u32], expected: &str) -> Option<u32> {
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    if masks.is_empty() { return None; }
    let mut order: Vec<usize> = (0..words.len()).collect();
    order.sort_by_key(|&b| words[b].0);
    if !order.windows(2).all(|w| words[w[0]].0 < words[w[1]].0) {
        // two candidates on one word (never seen): the plain loop, later bits written last, the left-out ranges zeroed after
        for &mask in masks {
            let mut alternate = base.to_vec();
            for (b, (index, set, _)) in words.iter().enumerate() {
                if mask >> b & 1 == 1 { alternate[8 * index..8 * index + 8].copy_from_slice(set); }
            }
            for &(start, end) in spans { alternate[start..end].fill(0); }
            if hex_bytes(&Sha256::digest(&alternate)) == expected { return Some(mask); }
        }
        return None;
    }
    let first = order.first().map_or(base.len(), |&b| 8 * words[b].0);
    let mut prefix = Sha256::new();
    feed_zeroed(&mut prefix, base, 0, first, spans);
    let read = |mask: u32| -> bool {
        let mut hasher = prefix.clone();
        let mut position = first;
        for &b in &order {
            let at = 8 * words[b].0;
            feed_zeroed(&mut hasher, base, position, at, spans);
            if mask >> b & 1 == 1 { hasher.update(words[b].1); } else { hasher.update(words[b].2); }
            position = at + 8;
        }
        feed_zeroed(&mut hasher, base, position, base.len(), spans);
        hex_bytes(&hasher.finalize()) == expected
    };
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(32).min(masks.len()).max(1);
    let (next, best) = (AtomicUsize::new(0), AtomicUsize::new(usize::MAX));
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let k = next.fetch_add(1, Relaxed);
                if k >= masks.len() || k > best.load(Relaxed) { break; }
                if read(masks[k]) { best.fetch_min(k, Relaxed); }
            });
        }
    });
    match best.into_inner() { usize::MAX => None, k => Some(masks[k]) }
}
/// speed storm3: the left-out ranges of a block as sorted, merged byte spans (start, end), refused as
/// masked_digest refuses them.
fn zero_spans(len: usize, zeroed: &[(usize, usize)]) -> Result<Vec<(usize, usize)>, String> {
    let mut spans = Vec::with_capacity(zeroed.len());
    for &(offset, bytes) in zeroed {
        let end = offset.checked_add(bytes).ok_or("residue range overflow")?;
        if end > len { return Err("residue range outside its allocation".into()); }
        if offset != end { spans.push((offset, end)); }
    }
    spans.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(spans.len());
    for (start, end) in spans {
        match merged.last_mut() { Some(last) if start <= last.1 => last.1 = last.1.max(end), _ => merged.push((start, end)) }
    }
    Ok(merged)
}
/// speed storm3: hash base[lo..hi] with the bytes inside `spans` (sorted, merged) read as zero: the bytes
/// masked_digest hashes of a zero-filled copy, with no copy.
fn feed_zeroed(hasher: &mut Sha256, base: &[u8], lo: usize, hi: usize, spans: &[(usize, usize)]) {
    static ZERO_BYTES: [u8; 64 << 10] = [0; 64 << 10];
    let mut cursor = lo;
    let mut k = spans.partition_point(|span| span.1 <= lo);
    while cursor < hi {
        match spans.get(k) {
            Some(&(start, end)) if start < hi => {
                if start > cursor { hasher.update(&base[cursor..start]); cursor = start; }
                let stop = end.min(hi);
                let mut left = stop - cursor;
                while left > 0 { let n = left.min(ZERO_BYTES.len()); hasher.update(&ZERO_BYTES[..n]); left -= n; }
                cursor = stop; k += 1;
            },
            _ => { hasher.update(&base[cursor..hi]); cursor = hi; },
        }
    }
}
/// speed storm3: the 8 bytes `word` of word `index` with the bytes inside `spans` zeroed.
fn zero_word(mut word: [u8; 8], index: usize, spans: &[(usize, usize)]) -> [u8; 8] {
    let (lo, hi) = (8 * index, 8 * index + 8);
    let mut k = spans.partition_point(|span| span.1 <= lo);
    while let Some(&(start, end)) = spans.get(k) {
        if start >= hi { break; }
        for at in start.max(lo)..end.min(hi) { word[at - lo] = 0; }
        k += 1;
    }
    word
}
/// speed storm3: the words of `viewed` that differ from `data` once the bytes inside `spans` are zeroed, in
/// order: what `(0..len / 8).filter(|i| viewed word != data word)` gives on zero-filled copies. A large block is
/// scanned in pieces on several cores.
fn changed_words(data: &[u8], viewed: &[u8], spans: &[(usize, usize)]) -> Vec<usize> {
    let words = data.len() / 8;
    let scan = |from: usize, to: usize| -> Vec<usize> {
        (from..to).filter(|&i| viewed[8 * i..8 * i + 8] != data[8 * i..8 * i + 8]
            && (spans.is_empty() || zero_word(viewed[8 * i..8 * i + 8].try_into().unwrap(), i, spans)
                                    != zero_word(data[8 * i..8 * i + 8].try_into().unwrap(), i, spans))).collect()
    };
    let threads = if data.len() < DIGEST_SCAN_PAR_BYTES { 1 } else { digest_scan_threads().min(words).max(1) };
    if threads == 1 { return scan(0, words); }
    let per = words.div_ceil(threads);
    std::thread::scope(|sc| {
        let scan = &scan;
        let hs: Vec<_> = (0..threads).map(|t| sc.spawn(move || scan((t * per).min(words), ((t + 1) * per).min(words)))).collect();
        hs.into_iter().flat_map(|h| h.join().expect("changed-word scan worker panicked")).collect()
    })
}
/// speed storm3: the reverted words of the last mixed readings that matched, by block length (newest first, at
/// most four a length). The subsets of a mixed reading give different bytes (each doubtful word differs between its
/// two readings), so at most one of them has the recorded SHA-256 and the order they are tried in cannot change the
/// answer: last boundary's answer is tried first, alone. THE WASTE THIS REMOVES: the same block needs the same
/// reading boundary after boundary (3 km storm box, blocks 189, 166, 144) and every boundary searched for it afresh.
static MIXED_MEMO: std::sync::Mutex<Vec<(usize, Vec<usize>)>> = std::sync::Mutex::new(Vec::new());
/// speed storm3: the subset search of the mixed reading: `masks` over the doubtful words `doubt` (bit b reverts
/// doubt[b] to its raw bytes), over `viewed` with the bytes inside `spans` read as zero. Returns the reverted words.
fn mixed_search(data: &[u8], viewed: &[u8], spans: &[(usize, usize)], doubt: &[usize], masks: &[u32], expected: &str) -> Option<Vec<usize>> {
    let words: Vec<(usize, [u8; 8], [u8; 8])> = doubt.iter().map(|&i| (i,
        zero_word(data[8 * i..8 * i + 8].try_into().unwrap(), i, spans),
        zero_word(viewed[8 * i..8 * i + 8].try_into().unwrap(), i, spans))).collect();
    let back = |mask: u32| -> Vec<usize> { (0..doubt.len()).filter(|b| mask & (1 << b) != 0).map(|b| doubt[b]).collect() };
    let remembered: Vec<u32> = MIXED_MEMO.lock().map(|memo| memo.iter().filter(|(len, _)| *len == data.len()).filter_map(|(_, set)| {
        let mut mask = 0u32;
        for index in set { mask |= 1 << doubt.iter().position(|d| d == index)?; }
        masks.contains(&mask).then_some(mask)
    }).collect()).unwrap_or_default();
    let found = first_mask_reading(viewed, spans, &words, &remembered, expected)
        .or_else(|| first_mask_reading(viewed, spans, &words, masks, expected))?;
    let set = back(found);
    if let Ok(mut memo) = MIXED_MEMO.lock() {
        memo.retain(|(len, old)| !(*len == data.len() && *old == set));
        memo.insert(0, (data.len(), set.clone()));
        let mut seen = 0;
        memo.retain(|(len, _)| { if *len == data.len() { seen += 1; seen <= 4 } else { true } });
        memo.truncate(256);
    }
    Some(set)
}

/// P1 a test machine: the forward hash of a block whose translated words are a mix of real stale pointers and stale data
/// that only looks like a replay address. Breakage: sealed 18-step hrrrv5 plan, a test machine: allocation 454 (89.6 MB,
/// not address-bearing) failed step 15 in the strict replay with 2 translated words and passed with 1 in the
/// continue replay, and the reverse at step 3; the step-3 dump against recording A's boundary payload differs in
/// exactly one word (replay 0x7c8524162514, forward 0x798a34162514, inside allocation 447), and the replay bytes
/// with that word set to the forward's give the forward hash, while 1,142 other 0x7xxx words are byte-identical
/// data, one of which a run's replay allocation can cover. Tries every subset of the changed words (fewest
/// first, at most MIXED_MAX_WORDS words) reverted to the raw bytes; returns (hash, reverted word indices) on a
/// match with `expected`, else None (the block keeps its view hash and fails as before).
fn mixed_reading(data: &[u8], viewed: &[u8], expected: &str, previous: Option<&PointerHistory>,
                 current: Option<&PointerHistory>) -> Option<(String, Vec<usize>)> {
    let changed: Vec<usize> = changed_words(data, viewed, &[]);
    // A word the last boundary translated from the same replay value passed that boundary's hash as a pointer and
    // is unchanged since, so it stays translated; only the others are in doubt. Breakage: a test machine continue replay
    // proof18c2, allocation 446 (34.4 MB) step 18: 61 translated words (the strict run: 59) after 84 at step 17
    // that all passed, too many for a search over every changed word.
    // F221: a word the half rule read this boundary (`current`, forward_address_view_tracked's history, holds the old
    // full pointer for it, so its replay value differs from the word) is proven by that same last-boundary pointer, so
    // it is not in doubt either. Breakage: a block with 3 half-cleared old pointers among lookalikes: keep-none
    // reverted the halves to the replay half, keep-1/2 restored at most two, and the block failed.
    let doubt: Vec<usize> = changed.iter().copied().filter(|i| {
        let word = u64::from_le_bytes(data[8 * i..8 * i + 8].try_into().unwrap());
        word == 0 || (!previous.and_then(|p| p.get(i)).is_some_and(|&(replay, _)| replay == word)
            && !current.and_then(|c| c.get(i)).is_some_and(|&(replay, _)| replay != word)
        )
    }).collect();
    let n = doubt.len();
    if n > MIXED_MAX_WORDS && n <= MIXED_MAX_COMPLEMENT {
        // Many doubtful words: the usual case is a few real stale pointers among byte-identical lookalikes, so
        // keep 0, 1 or 2 of them translated and revert every other one to raw (1 + n + n(n-1)/2 readings).
        let mut alternate = data.to_vec();
        for &i in &changed { if !doubt.contains(&i) { alternate[8 * i..8 * i + 8].copy_from_slice(&viewed[8 * i..8 * i + 8]); } }
        if let (Some(back), _) = complement_search(&alternate, viewed, &doubt, expected) { return Some((expected.to_string(), back)); }
        // multirank-2, from S01b 9e7e2577: the other way round: every doubtful word is a real pointer but 1 or 2, which
        // are data that only look like a replay address. Same complement search with the roles swapped (base = every
        // changed word translated, a reading takes its kept words raw from `data`), so it costs what F220's search
        // costs, not n(n-1)/2 whole-block hashes. THE BREAKAGE THIS PREVENTS: pkg60lam allocation 395 (TEMPO block,
        // 68.8 MB) at step 22 in about 1 replay in 4 (42 real pointers and one data word among 43 doubtful words); the
        // 2-rank pkg6-g4a TEMPO blocks 538/542 (43.8 MB) masked-unequal from step 3.
        let (kept, _) = complement_search(viewed, data, &doubt, expected);
        return kept.map(|others| (expected.to_string(), doubt.iter().copied().filter(|i| !others.contains(i)).collect()));
    }
    // g1 hunt (2026-11-11): MORE doubtful words than the complement search takes. The one reading left that costs
    // a single pass is tried: every proven word (noted by the runner's own table upload, or translated at the last
    // boundary and unchanged) stays translated and EVERY doubtful word is read as the data it is. THE BREAKAGE THIS
    // FIXES: the 2 h 54 min 60 km package pk60e29p, "state hash mismatch at step 6, 1 allocations, first 421" in most
    // replays and none in others. Step 6 uploads three pointer tables (51 words, all noted) into pool block 421
    // (2.1 GB), and the radiation step leaves data there that reads as 0x7xxx addresses; when the process's randomly
    // placed device memory covers them, 65 to 108 of them were taken for pointers (translated_address_words 116 to
    // 159 against 51 in a passing run). Above MIXED_MAX_COMPLEMENT no mixed reading was tried at all, so the block
    // kept the all-translated hash and failed. The reading is accepted only on the recording's own SHA-256.
    if n > MIXED_MAX_COMPLEMENT {
        let mut hasher = Sha256::new();
        let mut position = 0;
        for &i in &doubt { hasher.update(&viewed[position..8 * i]); hasher.update(&data[8 * i..8 * i + 8]); position = 8 * i + 8; }
        hasher.update(&viewed[position..]);
        return (hex_bytes(&hasher.finalize()) == expected).then(|| (expected.to_string(), doubt));
    }
    if n == 0 || n > MIXED_MAX_WORDS { return None; }
    // with every changed word in doubt, reverting all of them is the raw fallback boundary_digest already tried
    let top: u32 = if n == changed.len() { (1_u32 << n) - 1 } else { 1_u32 << n };
    let mut masks: Vec<u32> = (1..top).collect();
    masks.sort_by_key(|m| m.count_ones());
    // speed storm3: the same subsets, each hashed from the first doubtful word on, on every core (mixed_search)
    mixed_search(data, viewed, &[], &doubt, &masks, expected).map(|back| (expected.to_string(), back))
}
/// speed storm3: mixed_reading of a block with left-out ranges, without the two zero-filled copies of the block
/// when the search is the subset search (at most MIXED_MAX_WORDS doubtful words): the changed and doubtful words are
/// found on the bytes as they stand with the left-out bytes read as zero, and every reading is hashed that way. The
/// same words are in doubt, the same subsets are read and the same bytes hashed as on the copies. More doubtful words
/// than that take the copies and mixed_reading as before.
fn mixed_reading_zeroed(data: &[u8], viewed: &[u8], zeroed: &[(usize, usize)], expected: &str, previous: Option<&PointerHistory>,
                        current: Option<&PointerHistory>) -> Result<Option<(String, Vec<usize>)>, String> {
    let spans = zero_spans(data.len().min(viewed.len()), zeroed)?;
    let changed = changed_words(data, viewed, &spans);
    let doubt: Vec<usize> = changed.iter().copied().filter(|i| {
        let word = u64::from_le_bytes(zero_word(data[8 * i..8 * i + 8].try_into().unwrap(), *i, &spans));
        word == 0 || (!previous.and_then(|p| p.get(i)).is_some_and(|&(replay, _)| replay == word)
            && !current.and_then(|c| c.get(i)).is_some_and(|&(replay, _)| replay != word))
    }).collect();
    let n = doubt.len();
    if n == 0 { return Ok(None); }
    if n > MIXED_MAX_WORDS {
        let mut raw = data.to_vec();
        let mut translated = viewed.to_vec();
        for copy in [&mut raw, &mut translated] { for &(start, end) in &spans { copy[start..end].fill(0); } }
        return Ok(mixed_reading(&raw, &translated, expected, previous, current));
    }
    let top: u32 = if n == changed.len() { (1_u32 << n) - 1 } else { 1_u32 << n };
    let mut masks: Vec<u32> = (1..top).collect();
    masks.sort_by_key(|m| m.count_ones());
    Ok(mixed_search(data, viewed, &spans, &doubt, &masks, expected).map(|back| (expected.to_string(), back)))
}

/// g1 hunt (2026-11-11): data lookalikes and a half-rewritten old pointer in one block. Every proven word stays
/// translated, every doubtful word is read as data, except the doubtful words whose high half is still an old
/// pointer's (ambiguous_half_words: the last boundary held the full pointer there and a 4-byte store rewrote the low
/// half), which are read by the half rule; subsets of at most 4 such words, all of them first. THE BREAKAGE THIS
/// FIXES: pk60e29p, "state hash mismatch at step 11, 1 allocations, first 421" once step 6 passed. Step 11's
/// short-wave radiation stores a 4-byte count (0x00014001, plan step_11 command 154 reads it back) at
/// 421+2,010,661,632, over the low half of a pointer of the copy-plan table step 6 uploaded at 421+2,010,661,376;
/// the word (replay 0x714e00014001) lies inside a live replay allocation, so the plain view translated it as a
/// pointer, and the 84 data words of the w filter's table beside it were in doubt too (90 doubtful words): the
/// half readings were only tried with every lookalike translated, the lookalike readings only with the half word
/// raw. Returns (words read as data, words read by the half rule); accepted only on the recording's SHA-256.
fn lookalike_half_reading(data: &[u8], viewed: &[u8], expected: &str, previous: Option<&PointerHistory>,
                          current: Option<&PointerHistory>) -> Option<(Vec<usize>, Vec<usize>)> {
    let before = previous?;
    let doubt: Vec<usize> = (0..data.len() / 8).filter(|&i| viewed[8 * i..8 * i + 8] != data[8 * i..8 * i + 8]).filter(|i| {
        let word = u64::from_le_bytes(data[8 * i..8 * i + 8].try_into().unwrap());
        word == 0 || (!before.get(i).is_some_and(|&(replay, _)| replay == word)
            && !current.and_then(|c| c.get(i)).is_some_and(|&(replay, _)| replay != word))
    }).collect();
    let halves: Vec<(usize, u64)> = ambiguous_half_words(data, &[], 0, 0, previous).into_iter()
        .filter(|a| doubt.binary_search(&a.0).is_ok()).take(4).collect();
    let n = halves.len();
    if n == 0 { return None; }
    let mut masks: Vec<u32> = (1..(1u32 << n)).collect();
    masks.sort_by_key(|m| (std::cmp::Reverse(*m == (1u32 << n) - 1), m.count_ones(), *m));
    for mask in masks {
        let chosen: Vec<(usize, u64)> = halves.iter().enumerate().filter(|(k, _)| mask >> k & 1 == 1).map(|(_, a)| *a).collect();
        let mut hasher = Sha256::new();
        let mut position = 0;
        for &i in &doubt {
            hasher.update(&viewed[position..8 * i]);
            match chosen.iter().find(|c| c.0 == i) {
                Some(c) => hasher.update(c.1.to_le_bytes()),
                None => hasher.update(&data[8 * i..8 * i + 8]),
            }
            position = 8 * i + 8;
        }
        hasher.update(&viewed[position..]);
        if hex_bytes(&hasher.finalize()) == expected {
            let at: Vec<usize> = chosen.iter().map(|c| c.0).collect();
            return Some((doubt.into_iter().filter(|i| !at.contains(i)).collect(), at));
        }
    }
    None
}

/// addr-s (a development line, 2026-11-11): ONE last reading for a block whose fingerprint still differs after the readings
/// above. For every word it lists the admissible readings WITHOUT asking where this run's memory lies to rule one
/// out: plain data (the word as it stands), a live address by value (rewritten to the recorded block's address), and
/// what is left of an old address the last boundary's notes explain (the same word, its low half, high half or five
/// low bytes rewritten: the recorded top part with the rewritten bytes kept). Candidate blocks are then formed (a few
/// whole-block policies first, then single changes, then pairs) and one is accepted only when the SHA-256 of the
/// whole rebuilt block equals the recording's. THE BREAKAGE THIS PREVENTS: a sound recording stopping by chance,
/// because the guards `!inside_live` of the half-word and five-byte rules, and the note a word got at the last
/// boundary, chose a reading by where the card put this run's memory (120 km 2023-07-12 block 347 word 4874525,
/// replay 0x73363a48cf2f inside a live block; 3 km block 192, a cell-edge pair noted at step 1 and bound at step 2).
/// A real difference cannot pass: nothing is accepted except on equality with the recorded SHA-256, and when the
/// bound on hashes is passed the block stays a mismatch.
const ANY_READING_MAX_HASHES: usize = 4096;
/// addr-s: at most this many bytes are hashed by one search (a 125 MB block: 549 hashes; a 2.1 GB block: 30).
const ANY_READING_MAX_BYTES: u64 = 64 << 30;
fn any_reading_bound(len: usize) -> usize {
    ((ANY_READING_MAX_BYTES / len.max(1) as u64) as usize).clamp(8, ANY_READING_MAX_HASHES)
}
/// addr-s: one word with more than one reading: (value, note kept for the next boundary, from the notes).
#[derive(Debug, Clone, PartialEq)]
struct AnyWord { index: usize, raw: u64, alts: Vec<(u64, Option<(u64, u64)>, bool)> }
#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct AnyCost { words: usize, hashes: usize, bound: usize, seconds: f64, stage: &'static str }

/// addr-s order: the search of any_reading tries first the readings most likely to be right. Each word with more than
/// one reading gets a score of how much it looks like plain data and not like an address:
///   100  the word is zero (a zero is doubtful only through a note with a zero half);
///   +40  its low 32 bits are a small number (under 2^20), +20 more when the two 32-bit halves are within 4,096 of each
///        other (two neighbouring mesh numbers side by side: 0x744300007444 = 29763 and 29764);
///   +30  it has a note whose low half is zero (the noted block began on a 4 GiB line, so a zero written over half of
///        the word, or over all of it, cannot be told from the note);
///   +15  it has a note for an address no live block holds now (the noted block was freed);
///   +10  it has no note from the step before;
///    +5  it is not a multiple of 8 (every address the notes hold is).
/// The score orders the search and nothing else: a reading is accepted only when the whole block gives the recorded
/// SHA-256. THE BREAKAGE THIS PREVENTS: 3 km block 102 at step 1 (a development line round 6, <local path>): 557 to
/// 563 doubtful words of which 4 or 6 pairs of mesh numbers had to stay data; the search spent all 2,529 hashes on
/// single words in note order and stopped a right run 4 times of 4.
const ANY_ORDER_SMALL: u64 = 1 << 20;
const ANY_ORDER_NEAR: u64 = 1 << 12;
/// addr-s order: the highest score levels each tried as "every word at or above this level is data".
const ANY_ORDER_LEVELS: usize = 4;
/// addr-s order: the highest-scored words whose readings are then tried one and two at a time.
const ANY_ORDER_SUSPECTS: usize = 12;
fn data_likeness(w: &AnyWord, ranges: &[(u64, u64, u64)], low: u64, high: u64) -> u32 {
    if w.raw == 0 { return 100; }
    let (top, bottom) = (w.raw >> 32, w.raw & !HIGH_HALF);
    let mut score = 0;
    if bottom < ANY_ORDER_SMALL { score += 40; if top.abs_diff(bottom) < ANY_ORDER_NEAR { score += 20; } }
    let mut notes = w.alts.iter().filter(|a| a.2).filter_map(|a| a.1).peekable();
    if notes.peek().is_none() { score += 10; } else {
        let (mut zero_half, mut freed) = (false, false);
        for (replay, _) in notes {
            zero_half |= replay & !HIGH_HALF == 0;
            freed |= !inside_live(replay, ranges, low, high);
        }
        if zero_half { score += 30; }
        if freed { score += 15; }
    }
    if w.raw & 7 != 0 { score += 5; }
    score
}

fn admissible_readings(data: &[u8], ranges: &[(u64, u64, u64)], low: u64, high: u64,
                       previous: Option<&PointerHistory>) -> Vec<AnyWord> {
    let mut out: std::collections::BTreeMap<usize, AnyWord> = std::collections::BTreeMap::new();
    if let Some(previous) = previous {
        for (&index, &(replay, forward)) in previous {
            let Some(chunk) = data.get(8 * index..8 * index + 8) else { continue };
            let word = u64::from_le_bytes(chunk.try_into().unwrap());
            let mut values: Vec<u64> = Vec::new();
            if word == replay {
                values.push(forward);
                // addr-s order: a note whose low half (or five low bytes) is zero: a write of zeros over those bytes
                // leaves the word equal to its note, so "the old address with its low bytes rewritten" is one more
                // reading of a word that still equals such a note. THE BREAKAGE THIS PREVENTS: 3 km block 153 at step
                // 45 (a development line, <local path>): a live block began on a 4 GiB line
                // (0x7bf400000000 -> 0x75b4a4000000), the model wrote a 4-byte zero over the low half of word 16076672,
                // the recording held 0x75b400000000 and the check, which had only the whole address 0x75b4a4000000 to
                // offer, stopped a right run.
                if replay & !HIGH_HALF == 0 && forward & !HIGH_HALF != 0 { values.push(forward & HIGH_HALF); }
                if replay & !TOP_BYTES == 0 && forward & !TOP_BYTES != 0 { values.push(forward & TOP_BYTES); }
            } else {
                if word == replay & HIGH_HALF && replay & !HIGH_HALF != 0 { values.push(forward & HIGH_HALF); }
                if word == replay & !HIGH_HALF && replay & HIGH_HALF != 0 { values.push(forward & !HIGH_HALF); }
                if word & HIGH_HALF == replay & HIGH_HALF && replay & HIGH_HALF != 0 { values.push((forward & HIGH_HALF) | (word & !HIGH_HALF)); }
                if word & TOP_BYTES == replay & TOP_BYTES && replay & TOP_BYTES != 0 { values.push((forward & TOP_BYTES) | (word & !TOP_BYTES)); }
                // im p60: an old address with one to three of its five low bytes rewritten and the rest kept: the
                // recorded address with the same bytes rewritten. THE BREAKAGE THIS PREVENTS: the 60-step 12 km plan
                // p60 stopped its strict replay at step 28 ("state hash mismatch at step 28, 1 allocations, first
                // 790"): step 28 clears 1 byte at 790+103,754, byte 2 of a land pointer-table entry step 23 wrote at
                // 790+103,752 and nothing rewrote since (replay 0x72eb5b000000); the half and five-byte readings
                // replace whole halves or five bytes, and none gave the recording's word.
                let changed = word ^ replay;
                let mask = (0..8).map(|b| 0xff_u64 << (8 * b)).filter(|m| changed & m != 0).fold(0_u64, |a, m| a | m);
                if mask != 0 && mask & TOP_BYTES == 0 && mask.count_ones() <= 24 { values.push((forward & !mask) | (word & mask)); }
                if word & !HIGH_HALF == replay & !HIGH_HALF && replay & !HIGH_HALF != 0 && word & HIGH_HALF != replay & HIGH_HALF {
                    values.push((word & HIGH_HALF) | (forward & !HIGH_HALF));
                }
            }
            let mut alts: Vec<(u64, Option<(u64, u64)>, bool)> = Vec::new();
            for value in values {
                if value != word && !alts.iter().any(|a| a.0 == value) { alts.push((value, Some((replay, forward)), true)); }
            }
            if !alts.is_empty() { out.insert(index, AnyWord { index, raw: word, alts }); }
        }
    }
    for (index, chunk) in data.chunks_exact(8).enumerate() {
        let word = u64::from_le_bytes(chunk.try_into().unwrap());
        if word < low || word > high { continue; }
        let k = ranges.partition_point(|r| r.0 <= word);
        if k == 0 { continue; }
        let (start, end, recorded) = ranges[k - 1];
        if word > end { continue; }
        let forward = recorded + (word - start);
        if forward == word { continue; }
        let entry = out.entry(index).or_insert_with(|| AnyWord { index, raw: word, alts: Vec::new() });
        if !entry.alts.iter().any(|a| a.0 == forward) { entry.alts.push((forward, Some((word, forward)), false)); }
    }
    out.into_values().collect()
}

/// addr-s: see ANY_READING_MAX_HASHES. `viewed` is the default reading already tried, `zeroed` the left-out ranges
/// of a block with a masked fingerprint (empty otherwise). Returns the accepted reading of every word that had more
/// than one (word index, value, note for the next boundary) and what the search cost.
fn any_reading(data: &[u8], viewed: &[u8], ranges: &[(u64, u64, u64)], low: u64, high: u64,
               previous: Option<&PointerHistory>, zeroed: &[(usize, usize)], expected: &str)
               -> (Option<Vec<(usize, u64, Option<(u64, u64)>)>>, AnyCost) {
    let t0 = std::time::Instant::now();
    let left_out = |i: usize| zeroed.iter().any(|&(start, len)| 8 * i + 8 > start && 8 * i < start + len);
    let words: Vec<AnyWord> = admissible_readings(data, ranges, low, high, previous).into_iter().filter(|w| !left_out(w.index)).collect();
    let bound = any_reading_bound(data.len());
    let mut cost = AnyCost { words: words.len(), hashes: 0, bound, seconds: 0.0, stage: "" };
    if words.is_empty() || viewed.len() != data.len() { cost.seconds = t0.elapsed().as_secs_f64(); return (None, cost); }
    let masked: Option<Vec<u8>> = if zeroed.is_empty() { None } else {
        let mut copy = data.to_vec();
        if masked_digest(&mut copy, zeroed).is_err() { cost.seconds = t0.elapsed().as_secs_f64(); return (None, cost); }
        Some(copy)
    };
    let buffer: &[u8] = masked.as_deref().unwrap_or(data);
    let noted = |w: &AnyWord| w.alts.iter().find(|a| a.2).map(|a| a.0);
    let live = |w: &AnyWord| w.alts.iter().find(|a| !a.2).map(|a| a.0);
    // whole-block policies: the default already tried; the notes first, then live; the notes, else data; live, else
    // data (no note binds); every word data
    let mut bases: Vec<Vec<u64>> = Vec::new();
    for policy in 0..5 {
        let base: Vec<u64> = words.iter().map(|w| match policy {
            0 => u64::from_le_bytes(viewed[8 * w.index..8 * w.index + 8].try_into().unwrap()),
            1 => noted(w).unwrap_or(w.raw),
            2 => live(w).unwrap_or(w.raw),
            3 => noted(w).or(live(w)).unwrap_or(w.raw),
            _ => w.raw,
        }).collect();
        if !bases.contains(&base) { bases.push(base); }
    }
    // addr-s order (see data_likeness): after the whole-block policies, and before the single words in note order, the
    // readings most likely to be right. They take at most half of the hashes the policies leave, so the older search
    // keeps the other half. (1) Thresholds: every word at or above a score level read as data, on each policy, for the
    // ANY_ORDER_LEVELS highest levels. (2) The ANY_ORDER_SUSPECTS highest-scored words, each reading of one, then of
    // two of them, on each policy and then on each threshold reading.
    let policies = bases.len();
    let ordered_cap = (policies + bound.saturating_sub(policies) / 2).min(bound);
    let scores: Vec<u32> = words.iter().map(|w| data_likeness(w, ranges, low, high)).collect();
    let mut levels: Vec<u32> = scores.iter().copied().filter(|s| *s > 0).collect();
    levels.sort_unstable_by(|a, b| b.cmp(a));
    levels.dedup();
    levels.truncate(ANY_ORDER_LEVELS);
    'thresholds: for &level in &levels {
        for b in 0..policies {
            if bases.len() >= ordered_cap { break 'thresholds; }
            let base: Vec<u64> = (0..words.len()).map(|k| if scores[k] >= level { words[k].raw } else { bases[b][k] }).collect();
            if !bases.contains(&base) { bases.push(base); }
        }
    }
    let thresholds = bases.len();
    let mut trials: Vec<(usize, Vec<(usize, u64)>)> = (0..bases.len()).map(|b| (b, Vec::new())).collect();
    trials.truncate(bound);
    let options = |b: usize, k: usize| -> Vec<u64> {
        std::iter::once(words[k].raw).chain(words[k].alts.iter().map(|a| a.0)).filter(|v| *v != bases[b][k]).collect()
    };
    let mut suspects: Vec<usize> = (0..words.len()).filter(|&k| scores[k] > 0).collect();
    suspects.sort_by_key(|&k| (std::cmp::Reverse(scores[k]), k));
    suspects.truncate(ANY_ORDER_SUSPECTS);
    'ordered: for (b0, b1) in [(0, policies), (policies, thresholds)] {
        for b in b0..b1 { for &k in &suspects { for v in options(b, k) {
            if trials.len() >= ordered_cap { break 'ordered; }
            trials.push((b, vec![(k, v)]));
        } } }
        for b in b0..b1 { for (x, &k) in suspects.iter().enumerate() { for &j in &suspects[x + 1..] {
            for v in options(b, k) { for u in options(b, j) {
                if trials.len() >= ordered_cap { break 'ordered; }
                trials.push((b, vec![(k, v), (j, u)]));
            } }
        } } }
    }
    let ordered = trials.len();
    let tried: std::collections::HashSet<(usize, Vec<(usize, u64)>)> = trials[thresholds.min(ordered)..].iter().cloned().collect();
    // the older search, as it was: on each policy every single word in note order, then pairs
    let mut order: Vec<usize> = (0..words.len()).collect();
    order.sort_by_key(|&k| (!words[k].alts.iter().any(|a| a.2), k));
    'singles: for b in 0..policies {
        for &k in &order { for v in options(b, k) {
            if trials.len() >= bound { break 'singles; }
            let trial = (b, vec![(k, v)]);
            if !tried.contains(&trial) { trials.push(trial); }
        } }
    }
    'pairs: for b in 0..policies {
        for (x, &k) in order.iter().enumerate() { for &j in &order[x + 1..] {
            if trials.len() >= bound { break 'pairs; }
            for v in options(b, k) { for u in options(b, j) {
                if trials.len() >= bound { break 'pairs; }
                let trial = (b, vec![(k, v), (j, u)]);
                if !tried.contains(&trial) { trials.push(trial); }
            } }
        } }
    }
    let value_of = |t: &(usize, Vec<(usize, u64)>), k: usize| t.1.iter().find(|o| o.0 == k).map_or(bases[t.0][k], |o| o.1);
    let equal = |t: &(usize, Vec<(usize, u64)>)| -> bool {
        let mut hasher = Sha256::new();
        let mut position = 0;
        for (k, w) in words.iter().enumerate() {
            hasher.update(&buffer[position..8 * w.index]);
            hasher.update(value_of(t, k).to_le_bytes());
            position = 8 * w.index + 8;
        }
        hasher.update(&buffer[position..]);
        hex_bytes(&hasher.finalize()) == expected
    };
    let batch = std::thread::available_parallelism().map_or(8, |p| p.get()).clamp(1, 32);
    for chunk in trials.chunks(batch) {
        let hits = par_map(chunk.len(), |k| equal(&chunk[k]));
        if let Some(k) = hits.iter().position(|h| *h) {
            let at = cost.hashes + k;
            cost.stage = if at < policies { "whole_block_policy" } else if at < thresholds { "data_like_threshold" }
                else if at < ordered { "data_like_words" } else { "single_words_in_note_order" };
            cost.hashes += k + 1;
            cost.seconds = t0.elapsed().as_secs_f64();
            let read = words.iter().enumerate().map(|(n, w)| {
                let value = value_of(&chunk[k], n);
                (w.index, value, w.alts.iter().find(|a| a.0 == value && value != w.raw).and_then(|a| a.1))
            }).collect();
            return (Some(read), cost);
        }
        cost.hashes += chunk.len();
    }
    cost.seconds = t0.elapsed().as_secs_f64();
    (None, cost)
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
    /// lane tl-host-syncs: the boundaries before this one, finished (joined, compared, history filled) by this job's
    /// thread in step order before it read its own block (see digest_chain).
    earlier: Vec<DigestDone>,
    out: DigestOut,
    blocks: Vec<(u64, usize)>,
    residue: HashMap<(usize, u64), (Vec<(usize, usize)>, String)>,
    refs: Option<References>,
    ab: HashSet<u64>,
    bg_s: f64,
    /// a12 first-forward: how many of this boundary's blocks had their raw sha256 taken ahead (digest_raw_ahead).
    raw_ahead: usize,
}
/// a12 first-forward (default on; AD_DIGEST_RAW_AHEAD=0 keeps the old chain): how many threads of a chained boundary
/// digest take the raw sha256 of its large blocks while the job joins the boundary before it, and whether a checked,
/// deferred forward pass may have up to digest_arenas_max() arenas out, as a checkpointed sweep already may.
/// THE WASTE THIS REMOVES: the forward pass of a storm's first 3 km gradient is set by this chain, not by the card. A
/// forward step's launches take 0.36 s (fin-later, which compares no digest), but with one arena step s+1 cannot download
/// its end state before boundary s's job has read every block, and that job first joins boundary s-1, then hashes the
/// largest block (2,145,857,024 bytes) on one core: 1.47 to 1.74 s (a development line k12-prof host_digest_prof). So a forward
/// step costs that hash, 1.43 s median on back-1 g59-base and 2.06 s on fin-first (90.8 and 127.9 s for 59 steps), and
/// moves with the host's single-core speed, not with any setting of the reverse (none of them runs in the forward).
/// The raw sha256 of a block does not depend on the pointer history, so it is taken beside the join; with three arenas
/// three boundaries can be in that state at once. The same bytes are hashed, the same readings chosen in the same
/// order (boundary_digest_pre2 takes a raw hash already made, as for a large block before), the same history and
/// refusals, and no card word is read or written by a digest, so the answer bytes cannot move.
fn digest_raw_ahead() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("AD_DIGEST_RAW_AHEAD").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(4).min(16))
}
/// a12 first-forward: the spans of `spans` whose raw sha256 digest_blocks would take beside the address scan (a checked
/// boundary, at least DIGEST_OVERLAP_BYTES, not a large residue block read twice), largest first; an allocation that
/// appears more than once is left to digest_blocks.
fn raw_ahead_spans(step: usize, spans: &[(u64, usize, usize)], checked: bool,
                   residue: &HashMap<(usize, u64), (Vec<(usize, usize)>, String)>, ab: &HashSet<u64>) -> Vec<(u64, usize, usize)> {
    if !checked { return Vec::new(); }
    let mut seen: HashMap<u64, usize> = HashMap::new();
    for (id, _, _) in spans { *seen.entry(*id).or_default() += 1; }
    let mut v: Vec<(u64, usize, usize)> = spans.iter().copied()
        .filter(|(id, _, n)| *n >= DIGEST_OVERLAP_BYTES && seen[id] == 1 && !(residue.contains_key(&(step, *id)) && !ab.contains(id)))
        .collect();
    v.sort_by_key(|x| std::cmp::Reverse(x.2));
    v
}
/// lane tl-host-syncs: one finished deferred boundary digest, waiting for the main thread's digest_tail.
struct DigestDone { step: usize, mismatched: Result<Vec<u64>, String>, workers: usize, prof: Option<serde_json::Value>, bg_s: f64 }
/// lane tl-host-syncs: deferred boundary digests chain instead of joining at the next boundary. THE BREAKAGE THIS FIXES:
/// each boundary digest of the pk120 tangent pass takes 9-10 s on its background thread (block 429, 1.23 GB, 8.4 s of
/// readings; x20tls-I1), longer than a tangent step once the step's own host waits are gone, so finish_step waited for
/// the previous boundary's digest on the step path (1.1 s at step 3 already). Now the digest of boundary s+1 takes the
/// job of boundary s into its own thread, joins it there, runs its comparison (digest_post) and fills its pointer
/// history cell, then reads its own block with that history: the same readings, history and order as the
/// per-boundary join. The main thread joins the chain where it reads a history or ends the pass (digest_flush: every
/// restore, the pass end, the session end) and refuses there with the same text, naming the first boundary that
/// differs. AD_DIGEST_JOIN_STEP=1 keeps the join at every boundary.
/// speed (storm): a tangent step that keeps no host copy (with --checkpoints most steps) hashes its boundary beside
/// the next step too. THE DEFECT: the digest ran in the background only when the step's whole state had been captured
/// for storing; every other step downloaded its state into the digest arena and hashed it on the step path: 6.09 s of
/// a 10.26 s tangent step at the 3 km storm box (a test machine prof/ckprof forward_step_prof step 3: walk 4.16 s, finish_step
/// 6.09 s), against 0.003 s on a stored step. Now the arena goes to the digest thread with the same spans and comes
/// back when its readings are done; the next boundary that needs the arena waits for it there, so one arena is still
/// all the memory used and a step costs the longer of the step and the hash, not their sum. The same readings in the
/// same order through the same chain (digest_chain), so the same hashes, prints, history and refusals.
/// AD_DIGEST_ARENA_DEFER=0 keeps the hash on the step path (the pair's base leg).
fn digest_arena_defer() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AD_DIGEST_ARENA_DEFER").as_deref() != Ok("0"))
}
/// speed (storm, round 3): the boundary hash of a RECOMPUTED step runs beside the card, every check kept. THE
/// DEFECT: with --checkpoints the reverse sweep recomputes forward steps from a checkpoint, and each one downloaded its
/// end state and waited for its hash against the recording on the step path: 5.7 s of check for a 0.39 s step at the
/// 3 km storm box (a test machine prof/ckprof), 45 times a sweep in a 27-step window; and every restore joined the digest
/// chain. Now, inside a checkpointed sweep (digest_lazy), a recomputed boundary's bytes go to the digest chain in a
/// page-locked arena and the card goes on: at most this many arenas are out at once (AD_DIGEST_ARENAS, default 3; a
/// step that needs one more waits for the oldest), a restore no longer joins the chain (the restored boundary's own
/// history is handed to the next digest, restore_history_lazy), a boundary that differs raises digest_failed and the
/// next step end or restore refuses with the same text, and the sweep's end joins everything before the sweep counts.
/// The same readings of the same bytes in the same order through the same chain: the same hashes, prints, history
/// and refusals. AD_RECOMPUTE_HASH_BESIDE=0 keeps the hash on the step path (the pair's base leg).
pub(crate) fn recompute_hash_beside() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AD_RECOMPUTE_HASH_BESIDE").as_deref() != Ok("0"))
}
pub(crate) fn digest_arenas_max() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("AD_DIGEST_ARENAS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(3).clamp(1, 16))
}
fn digest_chain() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AD_DIGEST_JOIN_STEP").as_deref() != Ok("1"))
}
/// speed (merge, round 4): the body of a boundary digest thread, moved out of finish_step_host_hash line for line so
/// that the unit tests run the chain itself (digest_join_tests). It finishes the boundaries before this one in step
/// order (digest_chain), raises `failed` when one of them differs (digest_failed), then reads this boundary's blocks
/// with the history they left, or with the restored boundary's (`from`, restore_history_lazy). The caller sends the
/// arena back after it returns (bg_s, a receipt timing, now ends just before that send instead of just after it).
#[allow(clippy::too_many_arguments)]
fn digest_job_run(step: usize, prev: Option<DigestJob>, from: Option<HistSnap>, held: Vec<(u64, PointerHistory)>,
                  merged: HashMap<u64, PointerHistory>, failed: &std::sync::atomic::AtomicBool, bytes: &[u8],
                  spans: &[(u64, usize, usize)], ranges: &[(u64, u64, u64)], low: u64, high: u64,
                  residue: HashMap<(usize, u64), (Vec<(usize, usize)>, String)>, refs: Option<References>, ab: HashSet<u64>,
                  workers: usize, dprof: bool, raw_ahead: usize) -> DigestJobOut {
    // a12 first-forward: the large blocks' raw sha256 is taken beside the join of the boundary before (see
    // digest_raw_ahead); only when there is a join to wait for
    let ahead = if raw_ahead > 0 && prev.is_some() { raw_ahead_spans(step, spans, refs.is_some(), &residue, &ab) } else { Vec::new() };
    let next = std::sync::atomic::AtomicUsize::new(0);
    let raw_ready: std::sync::Mutex<HashMap<u64, String>> = std::sync::Mutex::new(HashMap::new());
    let (earlier, merged) = std::thread::scope(|sc| {
        for _ in 0..raw_ahead.min(ahead.len()) {
            sc.spawn(|| loop {
                let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(&(id, off, n)) = ahead.get(k) else { break };
                let h = hex_bytes(&Sha256::digest(&bytes[off..off + n]));
                raw_ready.lock().unwrap().insert(id, h);
            });
        }
        match prev {
            Some(job) => {
                let (done, m) = digest_finish(job);
                // speed (storm, round 3): every boundary before this one is finished here, so the restored
                // boundary's history is known (its cell is filled); a boundary that differed is flagged at once
                let mut m = match &from { Some(h) => h.get().clone(), None => m };
                for (id, words) in held { m.entry(id).or_default().extend(words); }
                if done.iter().any(|d| !matches!(&d.mismatched, Ok(x) if x.is_empty())) {
                    failed.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                (done, m)
            },
            None => (Vec::new(), merged),
        }
    });
    let raw_ready = raw_ready.into_inner().unwrap_or_default();
    let t = std::time::Instant::now();
    let buffers: Vec<(u64, &[u8])> = spans.iter().map(|&(id, off, n)| (id, &bytes[off..off + n])).collect();
    let out = digest_blocks_raw(step, &buffers, ranges, low, high, &merged, &residue, &ab, refs.as_ref(), workers, dprof, &raw_ready);
    let blocks: Vec<(u64, usize)> = buffers.iter().map(|(id, d)| (*id, d.len())).collect();
    drop(buffers);
    DigestJobOut { earlier, out, blocks, residue, refs, ab, bg_s: t.elapsed().as_secs_f64(), raw_ahead: raw_ready.len() }
}
/// Join a deferred job and finish it (and everything it finished before it): its done list in step order, and its
/// boundary pointer history (also set into the HostSnap cell).
fn digest_finish(job: DigestJob) -> (Vec<DigestDone>, HashMap<u64, PointerHistory>) {
    let t = std::time::Instant::now();
    let Ok(mut r) = job.handle.join() else {
        return (vec![DigestDone { step: job.step, mismatched: Err(format!("state hash at step {}: boundary digest thread panicked", job.step)), workers: job.workers, prof: None, bg_s: 0.0 }], HashMap::new());
    };
    let wait = t.elapsed().as_secs_f64();
    let mut done = std::mem::take(&mut r.earlier);
    let mismatched = digest_post(job.step, &r.blocks, &mut r.out, &r.residue, &r.ab, r.refs.as_ref());
    let history: HashMap<u64, PointerHistory> = r.blocks.iter().map(|(id, _)| *id).zip(std::mem::take(&mut r.out.histories))
        .filter(|(_, history)| !history.is_empty()).collect();
    let _ = job.cell.set(history.clone());
    let prof = job.dprof.then(|| digest_prof_json(job.step, &mut r.out.item_t, job.t_dl, r.out.workers_s, r.bg_s,
        Some(serde_json::json!({"deferred":true,"chained":digest_chain(),"background_s":r.bg_s,"join_wait_s":wait,"raw_hashed_ahead":r.raw_ahead}))));
    done.push(DigestDone { step: job.step, mismatched, workers: job.workers, prof, bg_s: r.bg_s });
    (done, history)
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
    digest_blocks_raw(step, buffers, ranges, low, high, previous, residue, address_bearing_ids, references, workers, dprof, &HashMap::new())
}
/// a12 first-forward: digest_blocks with the raw sha256 of some large blocks already taken (`raw_ready`, by allocation;
/// digest_job_run takes them beside its join). Such a block skips only its second thread: the same raw hash goes into
/// the same reading (boundary_digest_pre2), so the same values, history and refusals.
#[allow(clippy::too_many_arguments)]
fn digest_blocks_raw(step: usize, buffers: &[(u64, &[u8])], ranges: &[(u64, u64, u64)], low: u64, high: u64,
                 previous: &HashMap<u64, PointerHistory>, residue: &HashMap<(usize, u64), (Vec<(usize, usize)>, String)>,
                 address_bearing_ids: &HashSet<u64>, references: Option<&References>, workers: usize, dprof: bool,
                 raw_ready: &HashMap<u64, String>) -> DigestOut {
    let reference_payloads = std::env::var_os("AD_BINDER_REFERENCE_PAYLOADS").map(PathBuf::from);
    let t0 = std::time::Instant::now();
    let item_t: std::sync::Mutex<Vec<(u64, usize, f64, f64)>> = std::sync::Mutex::new(Vec::new());
    // S04: workers take the largest allocations first, so the longest single digest starts at once instead of
    // after the small ones (results land in their own slots: order of output unchanged).
    let mut order: Vec<usize> = (0..buffers.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(buffers[i].1.len()));
    if dprof {
        let sizes: Vec<_> = order.iter().take(8).map(|&i| {
            let (id, data) = buffers[i];
            serde_json::json!([id, data.len() / 8, previous.get(&id).map_or(0, |h| h.len())])
        }).collect();
        eprintln!("{}", serde_json::json!({"event":"host_digest_history","step":step,
            "address_range_prefilter":if address_range_prefilter_available() { "avx2_32_words" } else { "scalar" },
            "largest_id_words_history_entries":sizes}));
    }
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
                    // integ8 (U8/L1): an unchecked forward (no references: a Taylor row or any other pass at a state that was not recorded)
                    // compares no state hash, so its blocks get the address scan only (the pointer history the next
                    // boundary reads with, exactly as before) and no SHA-256: 1.0 s of the 1.3 s of every Taylor-row step.
                    let checked_step = references.is_some();
                    // F1: a large residue block always publishes the normalized
                    // digest and independently checks its masked digest. Its
                    // raw prehash cannot be used when a normalized view exists
                    // (expected=None below). Spend the second worker on the two
                    // required readings instead, joined before publication.
                    let overlap_residue = checked_step && data.len() >= DIGEST_OVERLAP_BYTES
                        && residue.contains_key(&(step, *id)) && !address_bearing_ids.contains(id);
                    let ((view, mut history, half), raw_pre, translated_pre) = if checked_step && !overlap_residue && data.len() >= DIGEST_OVERLAP_BYTES {
                        // a12 first-forward: the raw hash taken ahead, beside the join (digest_job_run)
                        if let Some(r) = raw_ready.get(id) {
                            let v = forward_address_view_tracked(data, &ranges, low, high, previous.get(id));
                            let t = v.0.as_ref().map(|x| hex_bytes(&Sha256::digest(&x.0)));
                            (v, Some(r.clone()), t)
                        } else {
                        std::thread::scope(|inner| {
                            let raw = inner.spawn(|| hex_bytes(&Sha256::digest(data)));
                            let v = forward_address_view_tracked(data, &ranges, low, high, previous.get(id));
                            // machine: the translated bytes are hashed here, beside the raw hash, not after it
                            let t = v.0.as_ref().map(|x| hex_bytes(&Sha256::digest(&x.0)));
                            (v, Some(raw.join().unwrap()), t)
                        })
                        }
                    } else { (forward_address_view_tracked(data, &ranges, low, high, previous.get(id)), None, None) };
                    let t_view = t_item.elapsed().as_secs_f64();
                    // residue blocks compare a masked digest below; the raw fallback is applied there
                    let expected = if residue.contains_key(&(step, *id)) { None }
                        else { references.and_then(|r| r.get(&(step, *id))).map(|e| e.1.as_str()) };
                    // speed (storm, round 3): whether the large residue block's two first masked readings both missed
                    let mut first_missed: Option<bool> = None;
                    let (mut value, raw, residue_ready) = if overlap_residue {
                        let (zeroed, masked_expected) = &residue[&(step, *id)];
                        let translated = view.as_ref().map(|v| v.0.as_slice());
                        std::thread::scope(|inner| {
                            let unmasked = inner.spawn(|| boundary_digest_pre(data, translated, None, None));
                            let masked = residue_masked_reading_first(data, translated, &ranges, low, high,
                                previous.get(id), zeroed, masked_expected, &mut history, &mut first_missed);
                            let (value, raw) = unmasked.join().unwrap();
                            if dprof {
                                eprintln!("{}", serde_json::json!({"event":"host_digest_required_overlap",
                                    "step":step,"allocation":id,"bytes":data.len(),"normalized":translated.is_some(),
                                    "required_readings":2,"joined_before_publication":true,"masked_initial_copy_bytes":0}));
                            }
                            (value, raw, Some(masked))
                        })
                    } else if checked_step {
                        let (value, raw) = boundary_digest_pre2(data, view.as_ref().map(|v| v.0.as_slice()), expected, raw_pre.as_deref(), translated_pre.as_deref());
                        (value, raw, None)
                    } else { (String::from("not computed (unchecked forward)"), false, None) };
                    // Optional authenticated bytes from this exact recording. Read only
                    // on mismatch; the model still executes every recorded operation.
                    if let (Some(root), Some(refs)) = (&reference_payloads, references) {
                        if !address_bearing_ids.contains(id) {
                            let (len, recorded_sha) = &refs[&(step, *id)];
                            let (zeroed, wanted): (&[(usize, usize)], &str) = residue.get(&(step, *id))
                                .map(|(z,h)| (z.as_slice(), h.as_str())).unwrap_or((&[], recorded_sha));
                            let normal = view.as_ref().map(|v| v.0.as_slice()).unwrap_or(data);
                            // speed (storm, round 3): each block's bytes are read once for this test. THE DEFECT: with
                            // recorded payloads given, every residue block was copied whole and hashed again here, as
                            // translated and (when that missed) raw, although residue_masked_reading had just taken the
                            // same two masked hashes of the same bytes: block 188 of the 3 km storm box (1.23 GB) cost
                            // 4.9 s at every boundary, most of a 5 to 6 s boundary check (a test machine prof/ckprof
                            // host_digest_prof). Now a large block's own reading answers (first_missed), and any other
                            // residue block is hashed in place (masked_digest_reading: the bytes masked_digest hashes
                            // of a copy, with no copy). The answer is the same in every case, an unreadable range
                            // included (both forms refuse the same ranges in the same order, and a refusal reads false;
                            // the large block's reading then stopped before it could answer, so the test is made here).
                            let needs = if zeroed.is_empty() { value != wanted } else {
                                match first_missed {
                                    Some(missed) => missed,
                                    None => masked_digest_reading(normal, zeroed).is_ok_and(|h| h != wanted)
                                        && masked_digest_reading(data, zeroed).is_ok_and(|h| h != wanted),
                                }
                            };
                            let file = root.join(format!("{recorded_sha}.gz"));
                            if needs && file.exists() {
                                let result = gzip_bytes(&file, *len).and_then(|reference| {
                                    let alternates = ambiguous_half_words(data, &ranges, low, high, previous.get(id));
                                    reference_reading(data, normal, &reference, recorded_sha, zeroed, wanted,
                                                      previous.get(id), &history, &alternates)
                                });
                                match result {
                                    Ok(Some(reading)) => {
                                        println!("{}", serde_json::json!({"event":"reference_address_reading","step":step,
                                            "allocation":id,"payload_sha256":recorded_sha,"translated_words":reading.moved,
                                            "half_words":reading.halves,"model_bytes_replaced":0}));
                                        let mut slot = slots[index].lock().unwrap();
                                        *slot.0 = reading.value; *slot.1 = reading.moved; *slot.2 = reading.history;
                                        *slot.3 = reading.halves; *slot.4 = reading.raw;
                                        *slot.5 = residue.contains_key(&(step, *id)).then(|| Ok((reading.masked, Some(reading.halves))));
                                        if dprof { item_t.lock().unwrap().push((*id, data.len(), t_view, t_item.elapsed().as_secs_f64())); }
                                        continue;
                                    },
                                    Err(error) => {
                                        let mut slot = slots[index].lock().unwrap();
                                        *slot.0 = format!("DEFECT: reference address reading: {error}");
                                        *slot.5 = residue.contains_key(&(step, *id)).then(|| Err(error));
                                        continue;
                                    },
                                    Ok(None) => {},
                                }
                            }
                        }
                    }
                    let mut half = half;
                    if !raw && expected.is_some_and(|e| e != value) {
                        let alternates = ambiguous_half_words(data, &ranges, low, high, previous.get(id));
                        // G2 (P1 milestone 2): each candidate is read as a half word or as it stands, in subsets (all of
                        // them first, then fewest first, at most 2^12 readings). THE BREAKAGE THIS PREVENTS: the
                        // radiation-live plan's TEMPO table (allocation 107, step 16) got 20 new full pointers in the
                        // same 4 GB region as old history entries, so they matched the half-word test and were all
                        // rewritten, while the one real half word (word 93) was outside every live allocation and left
                        // raw: no reading gave the forward hash.
                        let base: &[u8] = view.as_ref().map(|v| v.0.as_slice()).unwrap_or(data);
                        let n = alternates.len().min(12);
                        let mut masks: Vec<u32> = (1..(1u32 << n)).collect();
                        masks.sort_by_key(|m| (std::cmp::Reverse(*m == (1u32 << n) - 1), m.count_ones(), *m));
                        // speed storm3: the same subsets in the same order (first_mask_reading): no copy of the
                        // block per subset, each hashed from the first candidate word on, on every core
                        let words: Vec<(usize, [u8; 8], [u8; 8])> = alternates.iter().take(n)
                            .map(|a| (a.0, a.1.to_le_bytes(), base[8 * a.0..8 * a.0 + 8].try_into().unwrap())).collect();
                        if let Some(mask) = first_mask_reading(base, &[], &words, &masks, expected.unwrap()) {
                            let chosen: Vec<&(usize, u64)> = alternates.iter().take(n).enumerate()
                                .filter(|(k, _)| mask >> k & 1 == 1).map(|(_, a)| a).collect();
                            value = expected.unwrap().to_string(); half += chosen.len();
                            // keep the old pointer for the next boundary, so the same word gets the same reading
                            // (sealed 18-step hrrrv5 plan: allocation 163 word 93 passed step 12, failed step 13)
                            for (index, _) in &chosen {
                                if let Some(entry) = previous.get(id).and_then(|p| p.get(index)) {
                                    history.insert(*index, *entry);
                                }
                            }
                        }
                    }
                    // P1 a test machine: a block can hold both kinds of 0x7xxx word at once, a stale pointer that needs the
                    // forward's address and a stale data word that only looks like one and is byte-identical, so
                    // neither the all-translated view nor the raw bytes give the forward hash. Each changed word is
                    // then read either way (subsets of up to MIXED_MAX_WORDS words reverted to raw, fewest first).
                    let mut reverted = 0_usize;
                    if !raw && expected.is_some_and(|e| e != value) {
                        if let Some((viewed, _)) = view.as_ref() {
                            let mut halved: Vec<usize> = Vec::new();
                            let read = mixed_reading(data, viewed, expected.unwrap(), previous.get(id), Some(&history)).or_else(||
                                lookalike_half_reading(data, viewed, expected.unwrap(), previous.get(id), Some(&history))
                                    .map(|(back, at)| { halved = at; (expected.unwrap().to_string(), back) }));
                            // g1 hunt: name the words when a block is read with many data lookalikes, or is refused
                            // (one line a boundary and block, at most 256 words): byte offsets, replay words.
                            let words: Vec<usize> = match &read {
                                Some((_, back)) if back.len() > MIXED_MAX_WORDS => back.clone(),
                                Some(_) => Vec::new(),
                                None => (0..data.len() / 8).filter(|&i| viewed[8 * i..8 * i + 8] != data[8 * i..8 * i + 8]
                                    && !previous.get(id).and_then(|p| p.get(&i)).is_some_and(|&(replay, _)| replay.to_le_bytes() == data[8 * i..8 * i + 8])).collect(),
                            };
                            if !words.is_empty() {
                                let list: Vec<serde_json::Value> = words.iter().take(256).map(|&i| serde_json::json!([8 * i,
                                    format!("{:#x}", u64::from_le_bytes(data[8 * i..8 * i + 8].try_into().unwrap())),
                                    format!("{:#x}", u64::from_le_bytes(viewed[8 * i..8 * i + 8].try_into().unwrap()))])).collect();
                                println!("{}", serde_json::json!({"event":"address_lookalike_words","step":step,"allocation":id,
                                    "matched_recorded_sha256":read.is_some(),"words":words.len(),
                                    "half_rule_bytes":halved.iter().map(|i| 8 * i).collect::<Vec<usize>>(),
                                    "first_byte":8 * words[0],"last_byte":8 * words[words.len() - 1],
                                    "byte_replay_word_translated":list}));
                            }
                            if let Some((hashed, back)) = read {
                                value = hashed; reverted = back.len() + halved.len();
                                for index in &back { history.remove(index); }
                                // a half word keeps the old pointer for the next boundary, as the half readings above do
                                for index in &halved {
                                    if let Some(entry) = previous.get(id).and_then(|p| p.get(index)) { history.insert(*index, *entry); }
                                }
                                half += halved.len();
                            }
                        }
                    }
                    // addr-s: the one last reading (any_reading): every admissible reading of every word, without asking
                    // where this run's memory lies, decided by the recorded hash alone.
                    let mut any_count: Option<usize> = None;
                    if !raw && expected.is_some_and(|e| e != value) {
                        let default = view.as_ref().map(|v| v.0.as_slice()).unwrap_or(data);
                        let (read, cost) = any_reading(data, default, &ranges, low, high, previous.get(id), &[], expected.unwrap());
                        println!("{}", serde_json::json!({"event":"address_any_reading","step":step,"allocation":id,"bytes":data.len(),
                            "words_with_more_than_one_reading":cost.words,"hashes_tried":cost.hashes,"hash_bound":cost.bound,
                            "matched_recorded_sha256":read.is_some(),"seconds":cost.seconds,"matched_by":cost.stage}));
                        if let Some(words) = read {
                            value = expected.unwrap().to_string();
                            any_count = Some(words.iter().filter(|w| w.2.is_some()).count());
                            for (index, _, note) in words { history.remove(&index); if let Some(entry) = note { history.insert(index, entry); } }
                        }
                    }
                    let count = if raw { 0 } else if let Some(n) = any_count { n } else { view.as_ref().map(|v| v.1).unwrap_or(0).saturating_sub(reverted) };
                    if raw { history.clear(); }
                    // integ8: the excluded-residue block's masked digest (was the main thread's sequential loop below)
                    let res = if residue_ready.is_some() { residue_ready } else {
                        match residue.get(&(step, *id)) {
                            Some((zeroed, masked_expected)) if !address_bearing_ids.contains(id) =>
                                Some(residue_masked_reading(data, view.as_ref().map(|v| v.0.as_slice()), &ranges, low, high, previous.get(id), zeroed, masked_expected, &mut history)),
                            _ => None,
                        }
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
                let st = dev::sites::start();
                checked("cuMemHostAlloc staging", (self.cuda.host_alloc)(&mut h, bytes, 0))?;
                dev::sites::add("pin:cuMemHostAlloc staging", st);
                self.pinned_size.insert(h as usize, bytes);
                STAGING_ALLOCS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                h
            },
        };
        self.pinned.push(host);
        Ok(host)
    }
    /// lane tl-host-syncs: page-locked staging buffers made ahead on a helper thread (address, bytes), added to the pool
    /// exactly as buffers pinned_take made and a drain returned; Drop frees them the same way (cuMemFreeHost).
    pub(crate) fn adopt_pinned(&mut self, bufs: Vec<(usize, usize)>) {
        for (h, n) in bufs { self.pinned_size.insert(h, n); self.pinned_pool.push((h as *mut c_void, n)); }
    }
    /// Start once per Execution/session. The pool survives all TL/AD sweeps; CUDA addresses never cross contexts.
    unsafe fn start_staging_ahead(&mut self, sizes: Vec<usize>) {
        if self.staging_prepared { return; }
        self.staging_prepared = true;
        if std::env::var("AD_PIN_STAGING_AHEAD").as_deref() == Ok("0") { return; }
        match staging::Ahead::start(self.context as usize, self.cuda.ctx_set_current, self.cuda.host_alloc, self.cuda.host_free, sizes) {
            Ok(a) => self.staging_ahead = Some(a),
            Err(e) => eprintln!("{}", serde_json::json!({"event":"window_staging_ahead_fallback","error":e,"fallback":"inline page-locks"})),
        }
    }
    fn finish_staging_ahead(&mut self) {
        let Some(a) = self.staging_ahead.take() else { return; };
        match a.join() {
            Ok(mut b) => {
                eprintln!("{}", serde_json::json!({"event":"window_staging_ahead","buffers":b.buffers.len(),
                    "bytes":b.buffers.iter().map(|b| b.1).sum::<usize>(),"failed":b.failed,"first_error":b.first_error}));
                if b.failed != 0 { eprintln!("{}", serde_json::json!({"event":"window_staging_ahead_fallback",
                    "failed":b.failed,"first_error":b.first_error,"fallback":"inline page-locks for missing buffers"})); }
                self.adopt_pinned(std::mem::take(&mut b.buffers));
            },
            Err(e) => eprintln!("{}", serde_json::json!({"event":"window_staging_ahead_fallback","error":e,"fallback":"inline page-locks"})),
        }
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
    /// test (fresh contexts) got the root at 0x1000ce00000, congruent by luck, and wrote base + 0x30000; inside the
    /// host process (the host's own small allocations first) the driver sub-allocated it at 0x1000d093a00, the same
    /// rounding gave base + 0x2c600, and HexAD's forward_check refused 'state hash mismatch at step 2, allocation
    /// 427' (outer2 runs 7 and 8 and diagnostic X1: the in-host and CLI dumps of 427 differ in exactly that word).
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
        Ok(Self::from_parts(cuda, context, trace))
    }
    /// Opt-in live route. The caller keeps this context and stream alive and current.
    unsafe fn from_live(device: i32, context: *mut c_void, stream: *mut c_void) -> Result<Self, String> {
        let cuda = Cuda::load()?;
        checked("cuInit live", (cuda.init)(0))?;
        let mut current = ptr::null_mut();
        checked("cuCtxGetCurrent live", (cuda.ctx_get_current)(&mut current))?;
        if context.is_null() || current != context { return Err("live context is not current".into()); }
        let mut actual = -1;
        checked("cuCtxGetDevice live", (cuda.ctx_device)(&mut actual))?;
        if device != actual { return Err("live context/device mismatch".into()); }
        let mut ex = Self::from_parts(cuda, context, None);
        ex.borrowed_context = true;
        ex.streams.insert(0, stream);
        Ok(ex)
    }
    /// speed (merge, round 4): the struct as `new` fills it, around a loaded library and a context. The digest-join
    /// unit tests (digest_join_tests) pass a null context: they resolve libcuda's symbols and call nothing on a card.
    fn from_parts(cuda: Cuda, context: *mut c_void, trace: Option<BufWriter<File>>) -> Self {
        let mut streams = HashMap::new();
        streams.insert(0, ptr::null_mut());
        Self { kernel_override: HashMap::new(), borrowed_context: false, borrowed_allocations: HashSet::new(), mynn_relay: None, mynn_last_clear: None, mynn_relays: [0; 3], cuda, context, allocations: HashMap::new(), live_bytes: 0, live_peak: 0, host_state: host_state::HostState::default(), rk: None, streams, events: HashMap::new(),
                  modules: HashMap::new(), functions: HashMap::new(), cub_libraries: HashMap::new(),
                  thrust_libraries: HashMap::new(), cublas_libraries: HashMap::new(),
                  tempo_cache: tempo::Cache::default(), tempo_bindings: HashMap::new(),
                  digest_module: ptr::null_mut(), digest_function: ptr::null_mut(), digest_output: 0, digest_arena: None, digest_arena_back: Vec::new(), digest_lazy: false, history_from: None, digest_failed: Default::default(), recompute_digests_beside: 0, digest_s: 0.0,
                  word_module: ptr::null_mut(), word_function: ptr::null_mut(), word_output: 0,
                  pinned: Vec::new(),
                  submitter: submit::Submitter::default(), trace, address_bearing: HashSet::new(),
                  recorded_bases: HashMap::new(), raw_bases: HashMap::new(), residue: HashMap::new(), pointer_history: HashMap::new(),
                  held_pointer_writes: HashMap::new(), captured: None, digest_defer: false, unchecked_digest_skip: false, history_stale: false, digests_skipped: 0, digest_job: None, digest_bg_s: 0.0, ad: None, blob_cache: None,
                  pinned_pool: Vec::new(), pinned_size: HashMap::new(), staging_ahead: None, staging_prepared: false,
                  lw_regroup: false, lw_regroup_test: 0, lw_live_hist: None }
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

    /// ranks60 (fix E): `n` tables of other ranks entered the process-wide TEMPO dictionary here. Each one does
    /// what a miss does to the dictionary (the 64-entry clear, which also ends every live binding) and takes a place.
    fn tempo_foreign(&mut self, n: usize, step: usize) -> Result<(), String> {
        for _ in 0..n {
            if self.tempo_cache.clear_on_full() {
                let dropped = self.tempo_bindings.len();
                self.tempo_bindings.clear();
                self.log(serde_json::json!({"call":"tempo_cache_clear","step":step,"bindings_dropped":dropped,"by":"other rank"}))?;
            }
            self.tempo_cache.tick()?;
        }
        Ok(())
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
                // the source's 64-entry clear frees every cached pointer table, so every binding made before it
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
        let full_path = path(base, library_path);
        let key = (full_path.to_string_lossy().into_owned(), sha256.to_owned());
        if !self.cub_libraries.contains_key(&key) {
            let library = native_cub::NativeCub::open(&full_path, sha256)?;
            self.cub_libraries.insert(key.clone(), library);
        }
        Ok(&self.cub_libraries[&key])
    }
    unsafe fn thrust_library(&mut self, base: &Path, library_path: &str, sha256: &str)
        -> Result<&native_thrust::NativeThrust, String> {
        let full_path = path(base, library_path);
        let key = (full_path.to_string_lossy().into_owned(), sha256.to_owned());
        if !self.thrust_libraries.contains_key(&key) {
            let library = native_thrust::NativeThrust::open(&full_path, sha256)?;
            self.thrust_libraries.insert(key.clone(), library);
        }
        Ok(&self.thrust_libraries[&key])
    }
    unsafe fn cublas_library(&mut self, base: &Path, library_path: &str, sha256: &str)
        -> Result<&native_cublas::NativeCublas, String> {
        let full_path = path(base, library_path);
        let key = (full_path.to_string_lossy().into_owned(), sha256.to_owned());
        if !self.cublas_libraries.contains_key(&key) {
            let library = native_cublas::NativeCublas::open(&full_path, sha256)?;
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
            // central home-card: a refused package image is named, with this card's class (cuda::load_refusal).
            let status = (self.cuda.module_load)(&mut handle, image.as_ptr().cast());
            if status != 0 {
                return Err(cuda::load_refusal("cuModuleLoadData", status, &format!("plan module {} ({})", module.id, module.path), &self.cuda.card_class()));
            }
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
        // machine: the initial bytes of every allocation are read, inflated and SHA-256 verified ahead on worker
        // threads (in plan order, at most AD_PREPARE_AHEAD_GIB, default 8, decoded and not yet uploaded) while this
        // thread allocates and uploads in the same order as before. THE BREAKAGE THIS FIXES: one thread did all of it,
        // 5.7-5.9 s at 120 km with the card idle (a test machine machine pair1-b / pair2-a, mark prepared), and it grows
        // with the mesh. Same bytes, same checks, same order of device calls. AD_PREPARE_AHEAD=0 decodes inline.
        let jobs: Vec<usize> = plan.allocations.iter().enumerate().filter(|(_, a)| a.initial.is_some()).map(|(i, _)| i).collect();
        let slot_of: HashMap<usize, usize> = jobs.iter().enumerate().map(|(k, i)| (*i, k)).collect();
        let slots: Vec<std::sync::Mutex<Option<Result<Vec<u8>, String>>>> = jobs.iter().map(|_| std::sync::Mutex::new(None)).collect();
        let gate = std::sync::Mutex::new((0usize, 0usize, false)); // (next job, bytes decoded or decoding and not yet uploaded, stop)
        let (cv_room, cv_done) = (std::sync::Condvar::new(), std::sync::Condvar::new());
        let cap = std::env::var("AD_PREPARE_AHEAD_GIB").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(8.0).max(0.0) as usize * (1 << 30);
        let workers = if std::env::var("AD_PREPARE_AHEAD").as_deref() == Ok("0") { 0 }
                      else { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16).min(jobs.len()) };
        let mut kept: HashMap<u64, Vec<u8>> = HashMap::new(); // initial bytes of the allocations the relocation pass reads again
        let (mut wait_s, mut upload_s, mut initial_bytes) = (0f64, 0f64, 0usize);
        let t_prep = std::time::Instant::now();
        let uploaded: Result<(), String> = std::thread::scope(|sc| {
            for _ in 0..workers {
                sc.spawn(|| loop {
                    let k = {
                        let mut g = gate.lock().unwrap();
                        loop {
                            if g.2 || g.0 >= jobs.len() { return; }
                            let need = plan.allocations[jobs[g.0]].bytes;
                            if g.1 == 0 || g.1 + need <= cap { let k = g.0; g.0 += 1; g.1 += need; break k; }
                            g = cv_room.wait(g).unwrap();
                        }
                    };
                    let r = blob(base, plan.allocations[jobs[k]].initial.as_ref().unwrap());
                    *slots[k].lock().unwrap() = Some(r);
                    let _g = gate.lock().unwrap();
                    cv_done.notify_all();
                });
            }
            let r = (|| -> Result<(), String> {
                for (index, allocation) in plan.allocations.iter().enumerate() {
                    if allocation.bytes == 0 || self.allocations.contains_key(&allocation.id) {
                        return Err(format!("invalid or duplicate allocation {}", allocation.id));
                    }
                    let initial = match &allocation.initial {
                        Some(data) => {
                            let tw = std::time::Instant::now();
                            let data = if workers == 0 { blob(base, data)? } else {
                                let k = slot_of[&index];
                                let mut g = gate.lock().unwrap();
                                loop {
                                    if let Some(r) = slots[k].lock().unwrap().take() { break r?; }
                                    g = cv_done.wait(g).unwrap();
                                }
                            };
                            wait_s += tw.elapsed().as_secs_f64();
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
                        let tu = std::time::Instant::now();
                        checked("cuMemcpyHtoD", (self.cuda.upload)(address, data.as_ptr().cast(), data.len()))?;
                        upload_s += tu.elapsed().as_secs_f64(); initial_bytes += data.len();
                        if workers != 0 { let mut g = gate.lock().unwrap(); g.1 = g.1.saturating_sub(data.len()); cv_room.notify_all(); }
                        if !allocation.relocations.is_empty() { kept.insert(allocation.id, data); }
                    }
                    self.log(serde_json::json!({"call":"cuMemAlloc","id":allocation.id,"bytes":allocation.bytes,"address":address}))?;
                }
                Ok(())
            })();
            { let mut g = gate.lock().unwrap(); g.2 = true; cv_room.notify_all(); }
            r
        });
        uploaded?;
        eprintln!("{}", serde_json::json!({"event":"prepare_initial_state","allocations":plan.allocations.len(),"with_initial_bytes":jobs.len(),
            "bytes":initial_bytes,"decode_workers":workers,"ahead_cap_bytes":cap,"waited_for_decode_s":wait_s,"upload_s":upload_s,
            "wall_s":t_prep.elapsed().as_secs_f64()}));
        let recorded: HashMap<u64, u64> = plan.recorded_bases.iter().map(|[id, base]| (*id, *base)).collect();
        let mut relocated = 0_usize;
        for allocation in &plan.allocations {
            if allocation.relocations.is_empty() { continue; }
            let initial = match kept.remove(&allocation.id) { Some(d) => d,
                None => blob(base, allocation.initial.as_ref().ok_or("relocation without initial bytes")?)? };
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
        let mut probe_args: Option<Vec<submit::Argument>> = None;
        // bl-forward wire: the clear of the command just before (only the one just before counts), and this launch's relay.
        let last_clear = self.mynn_last_clear.take();
        let mut mynn_relay: Option<mynn_fwd_shape::Relay> = None;
        if let Some(ad) = self.ad.as_mut() {
            if let Operation::HeldUploadRanges { origin, .. } = op { ad.count(format!("held_upload_ranges:{origin}")); }
        }
        let ptr = |s: plan::Slice| submit::Slice { allocation: s.allocation, offset: s.offset };
        let native = match op {
            Operation::Kernel { stream, module, function, grid, block, shared_bytes, arguments } => {
                let handle = match self.kernel_override.get(function) { Some(f) => *f, None => self.function(module, function)? };
                let arguments = arguments.iter().map(|a| self.argument(a, step, frame.as_ref())).collect::<Result<Vec<_>,_>>()?;
                if self.ad.is_some() { ad_launch = Some((function.clone(), arguments.clone(), *grid, *block, *shared_bytes)); }
                if function == "solve_region_validate_v841" && peer::probe() { probe_args = Some(arguments.clone()); }
                // pk120 diagnostic (AD_BINDER_TRACE_KERNELS=1 with --trace): name each kernel before it is submitted, so a
                // native submit that fails (pkg120 step 1, CUDA 700/717 after the 42-row copy plan) names its kernel.
                if std::env::var("AD_BINDER_TRACE_KERNELS").as_deref() == Ok("1") {
                    self.log(serde_json::json!({"call":"kernel","function":function,"step":step,"grid":grid,"block":block}))?;
                }
                // bl-forward (AD_MYNN_FWD_LANES, default off): the MYNN-EDMF forward's entry and launch shape.
                let (handle, grid, block) = match crate::mynn_fwd_shape::get()? {
                    Some(shape) => match crate::mynn_fwd_shape::reshape(shape, function, &arguments, *grid, *block)? {
                        Some((entry, g, b)) => {
                            // THE BREAKAGE THIS PREVENTS: a --params carrier image replaces this function; the entry
                            // below would be read from the recorded image and bypass the carrier.
                            if self.kernel_override.contains_key(function) {
                                return Err(format!("AD_MYNN_FWD_LANES with a carrier image for {function}: not measured; refused"));
                            }
                            // bl-forward wire: the workspace moves to the entry's layout around the launch (below).
                            mynn_relay = mynn_fwd_shape::relay(shape, &arguments, |id| self.allocations.get(&id).map(|a| a.bytes))?;
                            (self.function(module, &entry)?, g, b)
                        },
                        None => (handle, *grid, *block),
                    },
                    None => (handle, *grid, *block),
                };
                Some((*stream, submit::Command::Kernel { function: handle, grid, block,
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
            Operation::PeerSend { stream, channel, source, bytes, .. } => {
                let p = peer::require("peer_send")?;
                let key = peer::key(step, *channel, &self.peer_scope());
                let address = self.address(*source, *bytes)?;
                // two-cards (AD_PEER_SHM=1): the bytes go to the shared slot on the command's stream, no drain
                if peer_shm::on() {
                    let s = peer_shm::get(&self.cuda)?;
                    if !s.has_forward(&key) { unsafe { peer_shm::fence(&self.cuda)?; } }
                    // fenced (=1): drained after the copy too, as the base download is synchronous (the sent bytes are read
                    // before any later command of this rank, on any stream, is issued)
                    if unsafe { s.send(b'f', self.stream(*stream)?, &key, address, *bytes)? } { unsafe { peer_shm::fence(&self.cuda)?; } }
                    else { p.note_resend(); }
                } else
                // a recompute of the base trajectory re-sends bytes its peer already holds: no sync, no download
                if p.has_forward(&key) { p.note_resend(); } else {
                    dev::sync(&self.cuda)?;
                    let data = dev::download(&self.cuda, address, *bytes)?;
                    p.put_forward(&key, &data)?;
                }
                None
            },
            Operation::PeerRecv { stream, channel, destination, bytes, .. } if peer_shm::on() => {
                // two-cards (AD_PEER_SHM=1): the command's stream waits for the slot's flag, then copies
                let address = self.address(*destination, *bytes)?;
                let key = peer::key(step, *channel, &self.peer_scope());
                let s = peer_shm::get(&self.cuda)?;
                unsafe { peer_shm::fence(&self.cuda)?; s.recv(b'f', self.stream(*stream)?, &key, address, *bytes)?; peer_shm::fence(&self.cuda)?; }
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
                let history = patch_upload_addresses(std::slice::from_raw_parts_mut(host.cast::<u8>(), n),
                    pointers, &self.allocations, *destination)?;
                checked("cuMemcpyHtoDAsync", (self.cuda.upload_async)(target, host, n, stream))?;
                if !history.is_empty() {
                    self.held_pointer_writes.entry(destination.allocation).or_default().extend(history);
                }
                self.log(serde_json::json!({"call":"cuMemcpyHtoDAsync","step":step,"bytes":n}))?;
                None
            },
            Operation::Upload { stream, destination, data, pointers, .. } => {
                let mut data = self.command_blob(base, data)?;
                if data.is_empty() { return Err("empty upload".into()); }
                let history = patch_upload_addresses(&mut data, pointers, &self.allocations, *destination)?;
                let target = self.address(*destination, data.len())?;
                let stream = self.stream(*stream)?;
                // adjoint-speed-2: from the pinned pool (returned at the next drain), not a fresh cuMemHostAlloc per command
                // (pkg18m2: ~170 cuMemHostAlloc/cuMemFreeHost pairs per step pass, both synchronizing). The same bytes are copied.
                let host = self.pinned_take(data.len())?;
                ptr::copy_nonoverlapping(data.as_ptr(), host.cast::<u8>(), data.len());
                checked("cuMemcpyHtoDAsync", (self.cuda.upload_async)(target, host, data.len(), stream))?;
                if !history.is_empty() {
                    self.held_pointer_writes.entry(destination.allocation).or_default().extend(history);
                }
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
                        // continuation 17:40Z: [int(a.data.ptr) for a in arrays] as uint64, in order:
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
            Operation::DescriptorTableUpload { stream, destination, rule, rows } => {
                if rule != "solve_region_scrub_descriptors" { return Err("unproved descriptor table rule".into()); }
                let pairs: Vec<(plan::Slice, &[u64])> = rows.iter().map(|r| (r.pointer, r.words.as_slice())).collect();
                let data = pointer_table::build_descriptors(&pairs, 5, &self.allocations)?;
                let target = self.address(*destination, data.len())?;
                let stream = self.stream(*stream)?;
                // runner overhead (0.6.2): the staging buffer comes from the window's page-locked pool (pinned_take, as
                // the pointer tables above), live until the next drain, which returns it to the pool after a sync. THE COST
                // THIS REMOVES: a new cuMemHostAlloc for every table and a cuMemFreeHost (which waits for the card) for each
                // at the drain: 472 of each a step at 12 km. The same bytes are uploaded from the same kind of memory.
                let host = self.pinned_take(data.len())?;
                ptr::copy_nonoverlapping(data.as_ptr(), host.cast::<u8>(), data.len());
                checked("cuMemcpyHtoDAsync descriptor table", (self.cuda.upload_async)(target, host, data.len(), stream))?;
                self.note_table_pointers(*destination, &data);
                self.log(serde_json::json!({"call":"cuMemcpyHtoDAsync", "source":"live_descriptor_table",
                                            "rule":rule, "step":step, "rows":rows.len()}))?;
                None
            },
            Operation::TempoPointerTableBind { stream, binding, destination, entries,
                                               expected_shape, device, optional, foreign_before, foreign_after } => {
                let data = tempo::pointer_table(entries, expected_shape.as_deref(), *device, *optional,
                                                &self.allocations)?;
                self.tempo_foreign(*foreign_before, step)?;
                self.bind_tempo(*stream, binding, *destination, *device, data, step)?;
                self.tempo_foreign(*foreign_after, step)?;
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
            Operation::DeviceToHost { stream, source, bytes, asynchronous, host_consumer, expected_hex, expected_zero_words, layer_groups, proof } => {
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
                    { let st = dev::sites::start(); checked("cuStreamSynchronize output", (self.cuda.stream_synchronize)(stream))?; dev::sites::add("plan:d2h-async-output", st); }
                } else {
                    { let st = dev::sites::start(); checked("cuMemcpyDtoH", (self.cuda.download)(host, source, *bytes))?; dev::sites::add("plan:d2h-sync-output", st); }
                }
                // Synchronised above (both paths): the bytes have landed.
                let got = unsafe { std::slice::from_raw_parts(host as *const u8, *bytes) };
                let verdict = judge_checked_read(got, expected.as_deref(), zero_words.as_deref(), layer_groups.is_some(), self.lw_regroup, self.lw_regroup_test);
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
                // dist47: a read that differs is refused in a sentence that says what the read is (checked_read_refusal):
                // the forecast's own fatal-condition words are not a host choice of the recording, and the refusal used
                // to tell the reader to record the package at a start no forecast runs from.
                // host-leftovers-2: the pooled buffer stays live until the next drain returns it to the pool (see the host
                // publication above); nothing reads it after this point, and the bytes copied are the same.
                if let CheckedRead::FollowLive { hist, recorded_equal } = verdict {
                    self.lw_live_hist = Some(hist);
                    self.log(serde_json::json!({"call":"lw_histogram_live","step":step,"bytes":bytes,"recorded_equal":recorded_equal}))?;
                    None
                } else {
                let (mismatch, detail) = match verdict { CheckedRead::Differs(d) => (true, d), _ => (false, String::new()) };
                if mismatch && peer::tolerate_d2h() {
                    eprintln!("{}", serde_json::json!({"event":"peer_checked_read_differs","step":step,"detail":detail,"allocation":source_slice.allocation,
                        "offset":source_slice.offset,"bytes":bytes,"host_consumer":host_consumer,"note":"logged, not refused (AD_PEER_TOLERATE_D2H=1)"}));
                } else if mismatch {
                    return Err(checked_read_refusal(step, &detail, source_slice.allocation, source_slice.offset, host_consumer, layer_groups.is_some(), proof.as_deref(), got));
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
                        { let st = dev::sites::start(); checked("cuEventSynchronize", (self.cuda.event_synchronize)(self.event(*id)?))?; dev::sites::add("plan:EventSynchronize", st); },
                    "StreamSynchronize" if *flags == 0 && *id == 0 =>
                        { let st = dev::sites::start(); checked("cuStreamSynchronize", (self.cuda.stream_synchronize)(stream_handle))?; dev::sites::add("plan:StreamSynchronize", st); },
                    "CtxSynchronize" if *flags == 0 && *id == 0 =>
                        { let st = dev::sites::start(); checked("cuCtxSynchronize", (self.cuda.context_synchronize)())?; dev::sites::add("plan:CtxSynchronize", st); },
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
        // S01c FD mode: the launch's own command, kept for the finite-difference re-submissions.
        let fd_cmd = match (self.ad.as_ref(), ad_launch.as_ref()) {
            (Some(ad), Some((f, ..))) if ad.fd_wanted(f, step) => native.clone(), _ => None };
        // THE BREAKAGE THIS PREVENTS: the FD re-submission re-issues the kept command alone, so the new entry would run
        // on the workspace in the recorded layout and read other words than the first submission did.
        if fd_cmd.is_some() && mynn_relay.is_some() {
            return Err("AD FD re-submission with AD_MYNN_FWD_LANES (an entry of another workspace layout): refused".into());
        }
        if let Some((stream, command)) = native {
            let stream = self.stream(stream)?;
            if let Some(r) = mynn_relay.as_ref() {
                let skip = last_clear.map_or(false, |c| r.covered_by(c));
                if !skip { self.mynn_relayout(r, mynn_fwd_shape::RECORDED_TILE, r.tile, stream)?; }
                self.mynn_relays[if skip { 1 } else { 2 }] += 1;
                if self.mynn_relays[0] < 4 {
                    eprintln!("{}", serde_json::json!({"event":"mynn_fwd_relay","step":step,"allocation":r.allocation,
                        "columns":r.columns,"tile":r.tile,"move_before": if skip { "skipped: clear just before" } else { "run" }}));
                }
            }
            let driver = submit::NativeDriver { launch: self.cuda.launch, copy: self.cuda.copy, clear: self.cuda.clear };
            // runner overhead (0.6.2): bound against the live table itself, not a copy of it made per command.
            self.submitter.submit_live(&driver, hex_step_validate, hex_step_submit, &self.allocations,
                                  &[command], &[], stream).map_err(|e| match op {
                                      Operation::Kernel { function, grid, block, .. } => format!("native submit: {e:?} (kernel {function}, grid {grid:?}, block {block:?})"),
                                      _ => format!("native submit: {e:?}") })?;
            if let Some(r) = mynn_relay.as_ref() {
                self.mynn_relayout(r, r.tile, mynn_fwd_shape::RECORDED_TILE, stream)?;
                self.mynn_relays[0] += 1;
            }
            self.log(serde_json::json!({"call":"hex_step_submit","step":step}))?;
        }
        if let Operation::Clear { destination, bytes, .. } = op {
            self.mynn_last_clear = Some((destination.allocation, destination.offset, *bytes));
        }
        if let Some(a) = probe_args { self.probe_validate(&a, step)?; }
        if let Some(w) = peer_watch() { self.watch_range(w, op, step)?; }
        if let (Some((f, a, g, b, sh)), Some(p)) = (ad_launch.as_ref(), ad_pending) {
            let mut ad = self.ad.take().ok_or("ad hook lost")?;
            let fd_stream = match fd_cmd.as_ref() { Some((s, _)) => Some(self.stream(*s)?), None => None };
            let fd_allocations: Vec<_> = if fd_cmd.is_some() { self.allocations.values().copied().collect() } else { Vec::new() };
            let (cuda, submitter) = (&self.cuda, &mut self.submitter);
            let mut rerun = || -> Result<(), String> {
                let (Some((_, c)), Some(st)) = (fd_cmd.as_ref(), fd_stream) else { return Err("FD mode: no forward command kept".into()) };
                let driver = submit::NativeDriver { launch: cuda.launch, copy: cuda.copy, clear: cuda.clear };
                submitter.submit(&driver, hex_step_validate, hex_step_submit, &fd_allocations, std::slice::from_ref(c), &[], st)
                    .map_err(|e| format!("FD native submit: {e:?}"))?;
                checked("cuCtxSynchronize FD", (cuda.context_synchronize)())
            };
            let r = ad.after(cuda, &self.allocations, step, f, a, *g, *b, *sh, p, &mut rerun);
            self.ad = Some(ad);
            r?;
        }
        Ok(())
    }

    /// bl-forward wire: move relay `r`'s whole 32-column groups of the workspace from tile `from` to tile `to` on `stream`
    /// (mynn_fwd_shape::commands), through a scratch of CHUNK_GROUPS groups made at first use.
    unsafe fn mynn_relayout(&mut self, r: &mynn_fwd_shape::Relay, from: u32, to: u32, stream: *mut c_void) -> Result<(), String> {
        let need = mynn_fwd_shape::scratch_bytes(r.stride);
        if self.mynn_relay.map_or(true, |(_, _, _, have)| have < need) {
            let (module, function) = match self.mynn_relay.take() {
                Some((m, f, p, _)) => {
                    checked("cuCtxSynchronize mynn relayout", (self.cuda.context_synchronize)())?;
                    checked("cuMemFree mynn relayout scratch", (self.cuda.free)(p))?;
                    (m, f)
                },
                None => {
                    let ptx = CString::new(mynn_fwd_shape::RELAYOUT_PTX).map_err(|e| e.to_string())?;
                    let (mut m, mut f) = (ptr::null_mut(), ptr::null_mut());
                    checked("cuModuleLoadData mynn relayout", (self.cuda.module_load)(&mut m, ptx.as_ptr().cast()))?;
                    let name = CString::new(mynn_fwd_shape::RELAYOUT_ENTRY).map_err(|e| e.to_string())?;
                    checked("cuModuleGetFunction mynn relayout", (self.cuda.module_function)(&mut f, m, name.as_ptr()))?;
                    (m, f)
                },
            };
            let mut p = 0u64;
            checked("cuMemAlloc mynn relayout scratch", (self.cuda.allocate)(&mut p, need))?;
            self.mynn_relay = Some((module, function, p, need));
            eprintln!("{}", serde_json::json!({"event":"mynn_fwd_relayout_ready","scratch_bytes":need,"stride":r.stride,
                "ptx_sha256":hex_bytes(&Sha256::digest(mynn_fwd_shape::RELAYOUT_PTX.as_bytes()))}));
        }
        let (_, function, scratch, bytes) = self.mynn_relay.ok_or("mynn relayout not made")?;
        if self.allocations.contains_key(&mynn_fwd_shape::SCRATCH_ID) {
            return Err(format!("plan allocation id {} is the relayout scratch's", mynn_fwd_shape::SCRATCH_ID));
        }
        let mut allocations: Vec<_> = self.allocations.values().copied().collect();
        allocations.push(submit::Allocation { id: mynn_fwd_shape::SCRATCH_ID, address: scratch, bytes });
        let commands = mynn_fwd_shape::commands(r, from, to, function);
        let driver = submit::NativeDriver { launch: self.cuda.launch, copy: self.cuda.copy, clear: self.cuda.clear };
        self.submitter.submit(&driver, hex_step_validate, hex_step_submit, &allocations, &commands, &[], stream)
            .map_err(|e| format!("mynn relayout native submit: {e:?} (allocation {}, {} columns, tile {from} -> {to})", r.allocation, r.columns))
    }

    /// ranks diagnostic: count the elements of rows x n (stride words apart) at `base` that fail the solve-region test.
    unsafe fn scan_field(&self, base: u64, rows: usize, stride: usize, n: usize, positive: bool) -> Result<serde_json::Value, String> {
        if rows == 0 || n == 0 { return Ok(serde_json::json!({"bad":0})); }
        let data = dev::download(&self.cuda, base, ((rows - 1) * stride + n) * 4)?;
        let (mut bad, mut zero, mut nonfin, mut neg) = (0usize, 0usize, 0usize, 0usize);
        let mut cols = std::collections::BTreeSet::new();
        for r in 0..rows { for c in 0..n {
            let v = f32::from_le_bytes(data[(r * stride + c) * 4..][..4].try_into().unwrap());
            if if positive { v.is_finite() && v > 0.0 } else { v.is_finite() } { continue; }
            bad += 1; cols.insert(c);
            if v == 0.0 { zero += 1 } else if !v.is_finite() { nonfin += 1 } else { neg += 1 }
        } }
        Ok(serde_json::json!({"bad":bad,"zero":zero,"nonfinite":nonfin,"negative":neg,"columns":cols.len(),
            "columns_first":cols.iter().take(48).collect::<Vec<_>>(),"columns_last":cols.iter().rev().take(12).collect::<Vec<_>>()}))
    }
    /// ranks diagnostic (AD_PEER_PROBE=1, never a proof leg): after a solve_region_validate_v841 launch read its flag;
    /// when it fired, decode the descriptor table (base, rows, stride, n, mode) and name each failing field.
    unsafe fn probe_validate(&mut self, a: &[submit::Argument], step: usize) -> Result<(), String> {
        let (Some(submit::Argument::Device(t)), Some(submit::Argument::Bytes(nb)), Some(submit::Argument::Device(fl))) = (a.first(), a.get(1), a.get(2)) else { return Ok(()) };
        let nf = i32::from_le_bytes(nb[..4].try_into().map_err(|_| "probe n_fields")?) as usize;
        dev::sync(&self.cuda)?;
        let fa = self.address(plan::Slice { allocation: fl.allocation, offset: fl.offset }, 4)?;
        if dev::download(&self.cuda, fa, 4)? == [0u8, 0, 0, 0] { return Ok(()); }
        let ta = self.address(plan::Slice { allocation: t.allocation, offset: t.offset }, nf * 40)?;
        let tab = dev::download(&self.cuda, ta, nf * 40)?;
        let w = |i: usize| i64::from_le_bytes(tab[i * 8..i * 8 + 8].try_into().unwrap());
        let mut rows = Vec::new();
        for f in 0..nf {
            let (base, nrows, stride, n, positive) = (w(f * 5) as u64, w(f * 5 + 1) as usize, w(f * 5 + 2) as usize, w(f * 5 + 3) as usize, w(f * 5 + 4) != 0);
            let owner = self.allocations.values().find(|x| base >= x.address && base < x.address + x.bytes as u64).map(|x| (x.id, base - x.address, x.bytes));
            let Some((id, off, ab)) = owner else { rows.push(serde_json::json!({"field":f,"base":format!("{base:#x}"),"where":"outside every live allocation"})); continue };
            if off as usize + ((nrows.max(1) - 1) * stride + n) * 4 > ab { rows.push(serde_json::json!({"field":f,"allocation":id,"offset":off,"where":"extent exceeds the allocation","rows":nrows,"stride":stride,"n":n})); continue; }
            let mut s = self.scan_field(base, nrows, stride, n, positive)?;
            if s["bad"] != 0 { s["field"] = f.into(); s["allocation"] = id.into(); s["offset"] = off.into(); s["rows"] = nrows.into(); s["stride"] = stride.into(); s["n"] = n.into(); s["positive"] = positive.into(); rows.push(s); }
        }
        eprintln!("{}", serde_json::json!({"event":"peer_probe_validate","step":step,"flag_allocation":fl.allocation,"flag_offset":fl.offset,"fields":nf,"bad_fields":rows}));
        Ok(())
    }
    /// ranks diagnostic (AD_PEER_WATCH=alloc:offset:rows:stride:n:positive, never a proof leg): after every command, scan
    /// that field and print when its failing-element count changes, with the command that changed it.
    unsafe fn watch_range(&mut self, w: (u64, usize, usize, usize, usize, bool), op: &Operation, step: usize) -> Result<(), String> {
        let (alloc, offset, rows, stride, n, positive) = w;
        let Some(a) = self.allocations.get(&alloc).copied() else { return Ok(()) };
        if offset + ((rows.max(1) - 1) * stride + n) * 4 > a.bytes { return Ok(()); }
        dev::sync(&self.cuda)?;
        let s = self.scan_field(a.address + offset as u64, rows, stride, n, positive)?;
        static LAST: std::sync::Mutex<Option<(u64, u64)>> = std::sync::Mutex::new(None);
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let k = SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let cur = (s["bad"].as_u64().unwrap_or(0), s["zero"].as_u64().unwrap_or(0));
        let mut last = LAST.lock().unwrap();
        if *last != Some(cur) {
            *last = Some(cur);
            let what = match op { Operation::Kernel { function, .. } => format!("kernel:{function}"), Operation::PeerSend { channel, .. } => format!("peer_send:{channel}"),
                Operation::PeerRecv { channel, .. } => format!("peer_recv:{channel}"), Operation::DeviceCopy { .. } => "device_copy".into(),
                Operation::Clear { .. } => "clear".into(), Operation::Upload { .. } => "upload".into(), Operation::Allocate { id, .. } => format!("allocate:{id}"), _ => "other".into() };
            eprintln!("{}", serde_json::json!({"event":"peer_watch","step":step,"command_seq":k,"after":what,"scan":s}));
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
        // speed (storm, round 3): a background digest found a boundary that differs: join the chain and refuse now
        if self.digest_failed.swap(false, std::sync::atomic::Ordering::Relaxed) { self.digest_flush()?; }
        // terrain-w-hash: the last boundary's deferred digest is joined (printed, compared) before this one starts
        // (its pointer history is this boundary's `previous`). lane tl-host-syncs: unless this one may defer too; then
        // finish_step_host_hash decides (a chained job joins it on its own thread, see digest_chain).
        if !(digest_chain() && self.digest_defer && references.is_some() && host_hash_requested()) {
            self.digest_flush()?;
        }
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
        // integ8 (S04 + ad/host-speed): carried views come from the window tangent's whole-state capture where it holds
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
        // speed (storm): the arena may be with an earlier boundary's background digest. A boundary that needs it
        // (no capture) waits for those readings here; a captured boundary does not wait and leaves it where it is.
        // speed (storm, round 3): inside a checkpointed sweep up to digest_arenas_max arenas may be out (oldest
        // first in digest_arena_back); elsewhere one, as before. One that has come back is taken first; at the bound
        // the step waits for the oldest (the chain finishes them in that order).
        if miss_total != 0 && self.digest_arena.as_ref().map_or(true, |a| a.len() < miss_total) {
            self.digest_arena = None;
            let mut i = 0;
            while self.digest_arena.is_none() && i < self.digest_arena_back.len() {
                match self.digest_arena_back[i].try_recv() {
                    Ok(a) => { self.digest_arena_back.remove(i); if a.len() >= miss_total { self.digest_arena = Some(a); } },
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => { self.digest_arena_back.remove(i); },
                    Err(std::sync::mpsc::TryRecvError::Empty) => i += 1,
                }
            }
            // a12 first-forward: a checked forward pass whose digests chain beside the card may have as many arenas out
            // as a checkpointed sweep (digest_raw_ahead); the sweep that follows it makes and keeps that many anyway
            let fwd_ahead = self.digest_defer && digest_chain() && references.is_some() && digest_raw_ahead() > 0;
            let bound = if self.digest_lazy || fwd_ahead { digest_arenas_max() } else { 1 };
            while self.digest_arena.is_none() && self.digest_arena_back.len() >= bound {
                let rx = self.digest_arena_back.remove(0);
                if let Ok(a) = rx.recv() { if a.len() >= miss_total { self.digest_arena = Some(a); } }
            }
        }
        let mut arena = match self.digest_arena.take() { Some(a) if a.len() >= miss_total => a, _ => pinned::PinnedBuf::new(&self.cuda, miss_total)? };
        // a12 bs-tape (backstep): inside a checkpointed sweep (a recomputed step whose hash runs beside the card) the arena
        // download goes on a copy stream of its own: it waits on the device for the step's last command, the legacy stream
        // waits on the device for the copy (so the next step's commands cannot change a byte before it is read), and the
        // digest thread waits for the copy before its first reading. THE WASTE THIS REMOVES: the host stood in a 10.5 GB
        // blocking download at the end of every recomputed 3 km step (about 0.2 s of the 0.92 s a recompute costs against
        // 0.58 s for a moved start, which skips the hash), unable to queue the next step's commands. The same bytes are read
        // in the same order by the same chain. AD_DIGEST_ASYNC=0 downloads on the step path as before.
        let async_dl = self.digest_lazy && self.digest_defer && miss_total != 0 && captured.is_none() && digest_arena_defer()
            && std::env::var("AD_BINDER_DIAGNOSTIC").as_deref() != Ok("1") && std::env::var("AD_DIGEST_ASYNC").as_deref() == Ok("1"); // no gain at 59 steps (bt-g59-t10): off unless 1
        let mut dl_event = 0usize;
        if async_dl {
            static DSTREAM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let mut st = DSTREAM.load(std::sync::atomic::Ordering::Relaxed) as *mut c_void;
            if st.is_null() { checked("cuStreamCreate digest", (self.cuda.stream_create)(&mut st, 1))?; DSTREAM.store(st as usize, std::sync::atomic::Ordering::Relaxed); }
            let (mut e0, mut e1) = (ptr::null_mut(), ptr::null_mut());
            checked("cuEventCreate digest", (self.cuda.event_create)(&mut e0, 2))?;
            checked("cuEventCreate digest", (self.cuda.event_create)(&mut e1, 2))?;
            checked("cuEventRecord digest start", (self.cuda.event_record)(e0, ptr::null_mut()))?;
            checked("cuStreamWaitEvent digest", (self.cuda.stream_wait_event)(st, e0, 0))?;
            for (id, p) in &place {
                if let Err((off, n)) = p {
                    if *n != 0 { checked("cuMemcpyDtoHAsync host digest", (self.cuda.download_async)(arena[*off..*off + *n].as_mut_ptr().cast(), actual[id].address, *n, st))?; }
                }
            }
            checked("cuEventRecord digest done", (self.cuda.event_record)(e1, st))?;
            checked("cuStreamWaitEvent legacy on digest", (self.cuda.stream_wait_event)(ptr::null_mut(), e1, 0))?;
            let _ = (self.cuda.event_destroy)(e0);
            dl_event = e1 as usize;
        } else {
        for (id, p) in &place {
            if let Err((off, n)) = p {
                if *n != 0 {
                    checked("cuMemcpyDtoH host digest", (self.cuda.download)(arena[*off..*off + *n].as_mut_ptr().cast(), actual[id].address, *n))?;
                }
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
        let diagnostic = std::env::var("AD_BINDER_DIAGNOSTIC").as_deref() == Ok("1");
        let defer_cap = self.digest_defer && miss_total == 0 && !diagnostic && captured.is_some();
        // speed (storm): no capture at all (a step that keeps no host copy): every view is in the arena.
        let defer_arena = self.digest_defer && !diagnostic && captured.is_none() && miss_total != 0 && digest_arena_defer();
        let defer_now = defer_cap || defer_arena;
        // lane tl-host-syncs: a deferred digest of the last boundary chains into this one's thread (digest_chain); any
        // other reading of this boundary needs its history now, so it is joined here as before.
        let prev = if defer_now && digest_chain() { self.digest_job.take() } else { self.digest_flush()?; None };
        // the previous boundary's translated words plus this step's relocated held words (held_pointer_writes)
        // speed (storm, round 3): after a lazy restore this boundary reads with the restored boundary's history
        let from = self.history_from.take();
        let mut merged = if prev.is_some() { HashMap::new() } else { match &from { Some(h) => h.get().clone(), None => self.pointer_history.clone() } };
        let held: Vec<(u64, PointerHistory)> = self.held_pointer_writes.drain().collect();
        if prev.is_none() { for (id, words) in held.iter().cloned() { merged.entry(id).or_default().extend(words); } }
        if defer_now {
            let data = if defer_cap { Some(captured.as_ref().unwrap().0.clone()) } else { None };
            // every view lies in the capture (a 0-byte view outside it reads as empty, as from the empty arena), or
            // (speed storm, defer_arena) every view lies in the arena at the offset it was downloaded to
            let spans: Vec<(u64, usize, usize)> = place.iter().map(|(id, p)| match p {
                Ok((off, n)) => (*id, *off, *n),
                Err((off, n)) => if defer_arena { (*id, *off, *n) } else { debug_assert_eq!(*n, 0); (*id, 0, 0) } }).collect();
            let arena_job = if defer_arena {
                let (tx, rx) = std::sync::mpsc::channel();
                self.digest_arena_back.push(rx);
                if self.digest_lazy { self.recompute_digests_beside += 1; }
                Some((arena, tx))
            } else { self.digest_arena = Some(arena); None };
            let residue: HashMap<(usize, u64), (Vec<(usize, usize)>, String)> =
                self.residue.iter().filter(|(k, _)| k.0 == step).map(|(k, v)| (*k, v.clone())).collect();
            let refs: Option<References> = references.map(|r| carried.iter()
                .filter_map(|id| r.get(&(step, *id)).map(|v| ((step, *id), v.clone()))).collect());
            let ab = self.address_bearing.clone();
            let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(DIGEST_BG_WORKERS).min(spans.len().max(1));
            let raw_ahead = digest_raw_ahead();
            let failed = self.digest_failed.clone();
            let (ev_sync, ev_destroy, ctx_set, ctx) = (self.cuda.event_synchronize, self.cuda.event_destroy, self.cuda.ctx_set_current, self.context as usize);
            let handle = std::thread::Builder::new().name("ad-boundary-digest".into()).spawn(move || {
                // a12 bs-tape: the arena's asynchronous download is complete before the first reading
                if dl_event != 0 {
                    if ctx_set(ctx as *mut c_void) != 0 || ev_sync(dl_event as *mut c_void) != 0 { panic!("step {step}: boundary digest could not wait for its download"); }
                    let _ = ev_destroy(dl_event as *mut c_void);
                }
                let bytes: &[u8] = match (&data, &arena_job) { (Some(d), _) => &d[..], (None, Some((a, _))) => &a[..], (None, None) => &[] };
                let out = digest_job_run(step, prev, from, held, merged, &failed, bytes, &spans, &ranges, low, high, residue, refs, ab, workers, dprof, raw_ahead);
                // speed (storm): the readings are done; the arena goes back for the next boundary's download
                if let Some((a, tx)) = arena_job { let _ = tx.send(a); }
                out
            }).map_err(|e| format!("step {step}: boundary digest thread: {e}"))?;
            self.digest_job = Some(DigestJob { step, handle, cell: std::sync::Arc::new(std::sync::OnceLock::new()), workers, dprof, t_dl });
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
        // Retain only failed extents for diagnosis. This does not enable diagnostic
        // admission or continuation: digest_tail still refuses the same mismatch.
        if !mismatched.is_empty() {
            if let Ok(dir) = std::env::var("AD_BINDER_FAILURE_DUMP") {
                let dir = Path::new(&dir).join(format!("step-{step}"));
                fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
                let listed: Vec<String> = ranges.iter().map(|r| format!("{:#x} {:#x} {:#x}", r.0, r.1, r.2)).collect();
                fs::write(dir.join("ranges.txt"), listed.join("\n")).map_err(|e| e.to_string())?;
                for (id, data) in &buffers {
                    if !mismatched.contains(id) { continue; }
                    fs::write(dir.join(format!("{id}.bin")), data).map_err(|e| e.to_string())?;
                    if let Some(h) = previous.get(id) {
                        let mut rows: Vec<String> = h.iter().map(|(i, (r, f))| format!("{i} {r:#x} {f:#x}")).collect();
                        rows.sort();
                        fs::write(dir.join(format!("{id}.history.txt")), rows.join(",")).map_err(|e| e.to_string())?;
                    }
                }
            }
        }
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
        // lane tl-host-syncs: the chain of boundaries in step order (digest_chain); the first that differs refuses.
        let (done, history) = digest_finish(job);
        self.pointer_history = history;
        let mut res = Ok(());
        for d in done {
            self.digest_bg_s += d.bg_s;
            if res.is_err() { continue; }
            res = d.mismatched.and_then(|m| self.digest_tail(d.step, m, d.workers, d.prof));
        }
        self.digest_s += t.elapsed().as_secs_f64();
        // speed (storm, round 3): the state on the card is the boundary last restored (restore_history_lazy)
        if let Some(h) = self.history_from.take() { if res.is_ok() { self.pointer_history = h.get().clone(); } }
        res
    }

    /// speed (storm, round 3): a restore inside a checkpointed sweep while a background digest is still running
    /// (see digest_arenas_max). Returns true when the restore must NOT join the chain: the restored boundary's history
    /// `h` is kept for the next boundary digest (history_from). A chain that has ended, or has flagged a boundary that
    /// differs, is joined here (so a refusal is raised now) and false is returned: the caller restores as before.
    pub(crate) fn restore_history_lazy(&mut self, h: &HistSnap) -> Result<bool, String> {
        if !self.digest_lazy || self.digest_job.is_none() { return Ok(false); }
        let failed = self.digest_failed.swap(false, std::sync::atomic::Ordering::Relaxed);
        if failed || self.digest_job.as_ref().is_some_and(|j| j.handle.is_finished()) { self.digest_flush()?; return Ok(false); }
        self.history_from = Some(h.clone());
        self.history_stale = false;
        Ok(true)
    }

    /// speed (merge, round 4): what window.rs restore_host does with the digest chain before it restores (its
    /// first two lines, moved here so the unit tests run them): inside a checkpointed sweep with the chain still
    /// running the restore does not join it (true); otherwise the chain is joined here and a boundary that differs
    /// refuses now.
    pub(crate) fn restore_join(&mut self, h: &HistSnap) -> Result<bool, String> {
        let lazy = self.restore_history_lazy(h)?;
        if !lazy { self.digest_flush()?; }
        Ok(lazy)
    }

    /// speed (storm, round 3): the end of a checkpointed sweep: every background digest is joined (compared,
    /// refused if one differs) and the arenas that were out come back; one is kept, as before the sweep.
    pub(crate) fn digest_lazy_end(&mut self) -> Result<(), String> {
        self.digest_lazy = false;
        let r = self.digest_flush();
        self.digest_failed.store(false, std::sync::atomic::Ordering::Relaxed);
        for rx in self.digest_arena_back.drain(..) {
            if let Ok(a) = rx.recv() { if self.digest_arena.is_none() { self.digest_arena = Some(a); } }
        }
        r
    }

    /// terrain-w-hash: the boundary history a HostSnap keeps: the current one, or (a deferred digest pending) the
    /// cell its join fills.
    pub(crate) fn history_snap(&self) -> HistSnap {
        match &self.digest_job {
            // speed (storm, round 3): restored lazily and no digest since: the restored boundary's history
            _ if self.history_from.is_some() => self.history_from.clone().unwrap(),
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
        // No detached staging worker may outlive this context, including errors during Win::new/session open.
        drop(self.staging_ahead.take());
        // terrain-w-hash: a deferred digest still running owns only host bytes; join it before the context goes.
        if let Some(job) = self.digest_job.take() { let _ = job.handle.join(); }
        unsafe {
            let mut previous = ptr::null_mut();
            if self.borrowed_context {
                // Never release caller resources, even if Rust drops outside the C close path.
                if (self.cuda.ctx_get_current)(&mut previous) != 0 || (self.cuda.ctx_set_current)(self.context) != 0 { return; }
                if (self.cuda.stream_synchronize)(self.streams[&0]) != 0 {
                    let _ = (self.cuda.ctx_set_current)(previous); return;
                }
            } else { let _ = (self.cuda.context_synchronize)(); }
            self.cub_libraries.clear();
            self.thrust_libraries.clear();
            self.cublas_libraries.clear();
            for host in self.pinned.drain(..) { let _ = (self.cuda.host_free)(host); }
            for (host, _) in self.pinned_pool.drain(..) { let _ = (self.cuda.host_free)(host); }
            for (_, event) in self.events.drain() { let _ = (self.cuda.event_destroy)(event); }
            for (id, stream) in self.streams.drain() { if id != 0 { let _ = (self.cuda.stream_destroy)(stream); } }
            for (id, allocation) in self.allocations.drain() {
                if !self.borrowed_allocations.contains(&id) { let _ = (self.cuda.free)(self.raw_bases.get(&id).copied().unwrap_or(allocation.address)); }
            }
            self.raw_bases.clear();
            for (_, module) in self.modules.drain() { let _ = (self.cuda.module_unload)(module); }
            if self.digest_output != 0 { let _ = (self.cuda.free)(self.digest_output); }
            if !self.digest_module.is_null() { let _ = (self.cuda.module_unload)(self.digest_module); }
            if self.word_output != 0 { let _ = (self.cuda.free)(self.word_output); }
            if !self.word_module.is_null() { let _ = (self.cuda.module_unload)(self.word_module); }
            if let Some((m, _, p, _)) = self.mynn_relay.take() {
                eprintln!("{}", serde_json::json!({"event":"mynn_fwd_relays","launches":self.mynn_relays[0],
                    "moves_before_skipped":self.mynn_relays[1],"moves_before_run":self.mynn_relays[2]}));
                let _ = (self.cuda.free)(p); let _ = (self.cuda.module_unload)(m);
            }
            if self.borrowed_context { let _ = (self.cuda.ctx_set_current)(previous); }
            else if !self.context.is_null() { let _ = (self.cuda.context_destroy)(self.context); }
        }
    }
}

/// dist47: the part of a refusal that says a checked read moved (the window names what the value is after it).
pub(crate) const READ_DIFFERS: &str = "differs from the proven value";
/// dist47: the part of a refusal that says the forecast itself stops at this start (see checked_read_refusal).
pub(crate) const FORECAST_STOPS: &str = "the forecast itself stops at this start";
/// dist47: the recorder's proof word of the forecast's deferred fatal-condition words (deferred_checks_v842.py).
pub(crate) const DEFERRED_CHECK_PROOF: &str = "raise_only_deferred_check_words";

/// dist47: what a checked read-back that has landed means, decided with no card.
/// `Same`: the bytes are the proven ones. `FollowLive`: the long-wave layer histogram of a forward that is not compared
/// with the recording (a moved start, a Taylor row): the window regroups the long-wave launches for the live histogram
/// (`recorded_equal` false) or, under AD_LW_REGROUP_TEST, for an equal one. `Differs`: every other read that moved, with
/// what moved; it is refused (checked_read_refusal).
#[derive(Debug, PartialEq)]
pub(crate) enum CheckedRead { Same, FollowLive { hist: Vec<i32>, recorded_equal: bool }, Differs(String) }
pub(crate) fn judge_checked_read(got: &[u8], expected: Option<&[u8]>, zero_words: Option<&[usize]>, lw_histogram: bool,
                                 follow_live: bool, regroup_test: u8) -> CheckedRead {
    let mut mismatch = expected.is_some_and(|want| got != want);
    if let Some(want) = zero_words {
        let zero: Vec<usize> = got.chunks_exact(4).enumerate().filter(|(_, w)| *w == [0u8; 4]).map(|(i, _)| i).collect();
        mismatch |= if want.is_empty() { zero.len() > 1 } else { zero.as_slice() != want };
    }
    if lw_histogram && follow_live && (mismatch || regroup_test != 0) {
        return CheckedRead::FollowLive { hist: got.chunks_exact(4).map(|w| i32::from_le_bytes(w.try_into().unwrap())).collect(), recorded_equal: !mismatch };
    }
    if mismatch { CheckedRead::Differs(mismatch_detail(got, expected, zero_words)) } else { CheckedRead::Same }
}
/// dist47: the refusal of a checked read that differs, by what the read is.
/// THE BREAKAGE THIS FIXES: the forecast's own fatal-condition words (eight a step at the 12 km storm cut, zero in every
/// forecast that runs) were refused as a host choice of the recording ("LW layer-group histogram: false ... record the
/// package at that state"): a minimisation's trial start at which the model itself stops was read as a long-wave
/// regroup the program had missed, and the reader was told to record a package no forecast can make.
pub(crate) fn checked_read_refusal(step: usize, detail: &str, allocation: u64, offset: usize, host_consumer: &str, lw_histogram: bool,
                                   proof: Option<&str>, got: &[u8]) -> String {
    let bytes = got.len();
    if proof == Some(DEFERRED_CHECK_PROOF) && bytes % 8 == 0 {
        let words: Vec<i64> = got.chunks_exact(8).map(|w| i64::from_le_bytes(w.try_into().unwrap())).collect();
        let set: Vec<String> = words.iter().enumerate().filter(|(_, v)| **v != 0).take(6).map(|(i, v)| format!("word {i}: {v}")).collect();
        return format!("step {step}: {FORECAST_STOPS}: {} of the {} check words the model reads at the end of the step are nonzero ({}); a \
forecast that runs reads zero in every one. Ask for a start nearer one that runs: a fresh program or a new recording stops at the same place, \
as the forecast command does. Each word is a fatal condition of one of the model's schemes (a native error branch, or a state that failed \
validation) (source allocation {allocation} offset {offset}, {bytes} bytes)",
                       words.iter().filter(|v| **v != 0).count(), words.len(), set.join(", "));
    }
    format!("checked device-to-host value {READ_DIFFERS} at step {step} ({detail}; source allocation {allocation} offset {offset}, {bytes} bytes, {host_consumer}; {})",
            if lw_histogram { "this read is the long-wave layer histogram" } else { "this read is not the long-wave layer histogram" })
}
/// dist47: a refusal a kept-loaded program survives (window.rs, the request loop): the forecast stops at the request's
/// start. The read is the last of its step, so the step has run to its end on the card and nothing of it is half done;
/// the next request restores the kept start as every request does.
pub(crate) fn refusal_leaves_program_loaded(error: &str) -> bool { error.contains(FORECAST_STOPS) }

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
        Some("window") => { let r = window::run(args.collect()); window::mark("run_returned"); return r; },
        // multicard: N rank processes over a run-time list of cards/hosts, and the per-host mailbox listener.
        Some("ranks") => return ranks::run(args.collect()),
        Some("peer-listen") => return peer::net_listen(args.collect()),
        Some("peer-ping") => return peer::ping(args.collect()),
        Some("peer-bench") => return peer_shm::bench(args.collect()),
        Some("capture-dot") => return capentry::run(args.collect()),
        Some("seam-dot") => return seam::run(args.collect()),
        Some("x0-plant") => return resident::plant_cmd(args.collect()),
        Some("resident-selftest") => return resident::selftest_cmd(args.collect()),
        Some("capdot") => return capdot::run(args.collect()),
        Some("--version") | Some("-V") => { println!("{}", version_line()); return Ok(()); },
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
    execution.ad = unsafe { addot::AdDot::from_env(&execution.cuda, &plan)? };
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

/// `ad-binder --version`: the crate version, the tree/ad commit this copy was synced from and whether the
/// rule/support fatbins were compiled in (an empty image makes `window` refuse at module load).
fn version_line() -> String {
    format!("ad-binder {} (hexad-phys; source {}; rule fatbins {})", env!("CARGO_PKG_VERSION"),
            env!("HEXAD_SOURCE_COMMIT"), env!("HEXAD_FATBINS"))
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
mod integrated_digest_chain_tests {
    use super::*;

    fn job(step: usize, word: u64, expected: u64, previous: Option<DigestJob>) -> DigestJob {
        let cell = std::sync::Arc::new(std::sync::OnceLock::new());
        let handle = std::thread::spawn(move || {
            let (earlier, history) = previous.map(digest_finish).unwrap_or_default();
            let data = word.to_le_bytes();
            let ranges = [(0x78ac_0000_0000, 0x78ac_0216_f000, 0x7e57_2200_0000),
                          (0x78ac_4429_6400, 0x78ac_442e_1600, 0x7e57_8629_6400)];
            let refs = HashMap::from([((step, 429), (8, hex_bytes(&Sha256::digest(expected.to_le_bytes()))))]);
            let residue = HashMap::new();
            let ab = HashSet::new();
            let out = digest_blocks(step, &[(429, &data)], &ranges, ranges[0].0, ranges[1].1,
                                    &history, &residue, &ab, Some(&refs), 1, true);
            DigestJobOut { earlier, out, blocks: vec![(429, 8)], residue, refs: Some(refs), ab, bg_s: 0.0, raw_ahead: 0 }
        });
        DigestJob { step, handle, cell, workers: 1, dprof: true, t_dl: 0.0 }
    }

    #[test]
    fn chained_boundaries_keep_pointer_history_and_receipt_keys() {
        let first = job(1, 0x78ac_442b_5c00, 0x7e57_862b_5c00, None);
        let first_cell = first.cell.clone();
        // The second word could be misread as a different live allocation without step 1's history.
        let second = job(2, 0x78ac_0000_0000, 0x7e57_0000_0000, Some(first));
        let second_cell = second.cell.clone();
        let (done, history) = digest_finish(second);
        assert_eq!(done.iter().map(|d| d.step).collect::<Vec<_>>(), vec![1, 2]);
        assert!(done.iter().all(|d| d.mismatched.as_ref().unwrap().is_empty()));
        assert_eq!(first_cell.get().unwrap()[&429][&0], (0x78ac_442b_5c00, 0x7e57_862b_5c00));
        assert_eq!(second_cell.get().unwrap(), &history);
        for d in done {
            let profile = d.prof.unwrap();
            assert_eq!(profile["step"], d.step);
            for key in ["download_s", "workers_s", "total_s", "residue_and_compare_s",
                        "slowest_id_bytes_view_s_item_s", "deferred", "chained", "background_s", "join_wait_s"] {
                assert!(profile.get(key).is_some(), "lost receipt key {key}");
            }
        }
    }

    #[test]
    fn later_boundary_does_not_erase_an_earlier_mismatch() {
        let first = job(1, 0x78ac_442b_5c00, 0x7e57_862b_5c01, None);
        let (done, _) = digest_finish(job(2, 0x78ac_0000_0000, 0x7e57_0000_0000, Some(first)));
        assert_eq!(done.len(), 2);
        assert_eq!(done[0].step, 1);
        assert_eq!(done[0].mismatched.as_ref().unwrap(), &[429]);
        assert!(done[1].mismatched.as_ref().unwrap().is_empty());
    }

    #[test]
    fn failed_background_digest_is_a_refusal() {
        let job = DigestJob { step: 3, handle: std::thread::spawn(|| panic!("injected digest failure")),
                              cell: Default::default(), workers: 1, dprof: false, t_dl: 0.0 };
        let (done, history) = digest_finish(job);
        assert!(history.is_empty());
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].mismatched.as_ref().unwrap_err(), "state hash at step 3: boundary digest thread panicked");
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
    use super::{boundary_digest, forward_address_view, forward_address_view_tracked, mixed_reading,
                hex_bytes, masked_digest, reference_reading};
    use sha2::{Digest, Sha256};
    fn words(data: &[u8]) -> Vec<u64> { data.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect() }
    fn bytes(words: &[u64]) -> Vec<u8> { words.iter().flat_map(|w| w.to_le_bytes()).collect() }

    #[test]
    fn uploaded_pointer_history_preserves_rewritten_data_half() {
        // Reduced pk120/r2 step-1 allocation 494 word 29: a certified upload
        // pointer precedes the first boundary, then its low half becomes data.
        let allocations = std::collections::HashMap::from([(868, super::submit::Allocation {
            id: 868, address: 0x7033_1000_0000, bytes: 0x100000,
        })]);
        let patches = [super::plan::Patch { byte_offset: 224,
            target: super::plan::Slice { allocation: 868, offset: 0x1000 } }];
        let mut upload = vec![0; 240];
        upload[224..232].copy_from_slice(&0x763a_1fca_1000_u64.to_le_bytes());
        // A pointer-shaped value without a certified patch is not history.
        upload[232..240].copy_from_slice(&0x763a_1fca_1010_u64.to_le_bytes());
        let previous = super::patch_upload_addresses(&mut upload, &patches, &allocations,
            super::plan::Slice { allocation: 494, offset: 8 }).unwrap();
        assert_eq!(previous.len(), 1);
        assert_eq!(previous[&29], (0x7033_1000_1000, 0x763a_1fca_1000));
        let mut data = vec![0; 256]; data[8..248].copy_from_slice(&upload);
        data[232..236].copy_from_slice(&0x3a48_cf2f_u32.to_le_bytes());
        let mut reference = data.clone();
        reference[232..240].copy_from_slice(&0x763a_3a48_cf2f_u64.to_le_bytes());
        let ranges = [(0x7033_2000_0000, 0x7033_4000_0000, 0x763a_1a00_0000)];
        let (low, high) = (ranges[0].0, ranges[0].1);
        let expected = hex_bytes(&Sha256::digest(&reference));
        let (view, history, _) = forward_address_view_tracked(&data, &ranges, low, high, Some(&previous));
        let view = view.unwrap().0;
        assert_ne!(view, reference);
        assert!(reference_reading(&data, &view, &reference, &expected, &[], &expected,
            None, &history, &[]).unwrap().is_none());
        let alternates = super::ambiguous_half_words(&data, &ranges, low, high, Some(&previous));
        let reading = reference_reading(&data, &view, &reference, &expected, &[], &expected,
            Some(&previous), &history, &alternates).unwrap().unwrap();
        assert_eq!(reading.value, expected); assert_eq!(reading.halves, 1);
        // A real change to the data half still fails against the same payload.
        data[232] ^= 1;
        let (wrong, history, _) = forward_address_view_tracked(&data, &ranges, low, high, Some(&previous));
        let alternates = super::ambiguous_half_words(&data, &ranges, low, high, Some(&previous));
        assert!(reference_reading(&data, &wrong.unwrap().0, &reference, &expected, &[], &expected,
            Some(&previous), &history, &alternates).unwrap().is_none());
    }

    #[test]
    fn zeroed_pointer_on_four_gib_boundary_has_two_valid_readings() {
        // pk120/r2 step-2 allocation 175 word 1856. A whole-word clear
        // must remain zero even when the historical pointer's low half was
        // already zero. A genuine high-half-only clear must still relocate.
        let previous = super::PointerHistory::from([
            (0, (0x7783_0000_0000, 0x763b_07a0_0000)),
            (1, (0x7783_1000_0010, 0x763b_17a0_0010)),
        ]);
        let data = bytes(&[0, 0x7783_1000_0010, 17]);
        let ranges = [(0x7783_1000_0000, 0x7783_1000_0100, 0x763b_17a0_0000)];
        let (view, history, _) = forward_address_view_tracked(&data, &ranges,
            ranges[0].0, ranges[0].1, Some(&previous));
        let view = view.unwrap().0;
        assert_eq!(words(&view), vec![0x07a0_0000, 0x763b_17a0_0010, 17]);
        let cleared = bytes(&[0, 0x763b_17a0_0010, 17]);
        for reference in [&cleared, &view] {
            let expected = sha(reference);
            let reading = reference_reading(&data, &view, reference, &expected, &[], &expected,
                Some(&previous), &history, &[]).unwrap().unwrap();
            assert_eq!(reading.value, expected);
            assert_eq!(mixed_reading(&data, &view, &expected, Some(&previous), Some(&history)).is_some(),
                reference == &cleared);
        }
        let bad = bytes(&[1, 0x763b_17a0_0010, 17]); let expected = sha(&bad);
        assert!(reference_reading(&data, &view, &bad, &expected, &[], &expected,
            Some(&previous), &history, &[]).unwrap().is_none());
    }

    #[test]
    fn reference_reading_many_pointers_keeps_three_data_words() {
        // Actual r8 failure shape: 220 pointers plus three raw data lookalikes.
        let ranges = [(0x740000000000, 0x740000100000, 0x710000000000)];
        let mut data = Vec::new(); let mut reference = Vec::new();
        for i in 0..223_u64 {
            let word = ranges[0].0 + 8 * i;
            data.extend(word.to_le_bytes());
            reference.extend((if [18, 79, 211].contains(&i) { word } else { ranges[0].2 + 8 * i }).to_le_bytes());
        }
        data.extend(0x123456789abcdef0_u64.to_le_bytes());
        reference.extend(0xeeeeeeee9abcdef0_u64.to_le_bytes());
        let zeroed = [(223 * 8 + 4, 4)];
        let expected = hex_bytes(&Sha256::digest(&reference));
        let masked = masked_digest(&mut reference.clone(), &zeroed).unwrap();
        let (view, history, _) = forward_address_view_tracked(&data, &ranges, ranges[0].0, ranges[0].1, None);
        let view = view.unwrap().0;
        let reading = reference_reading(&data, &view, &reference, &expected, &zeroed, &masked, None, &history, &[]).unwrap().unwrap();
        assert_eq!(reading.masked, masked); assert_eq!(reading.moved, 220);
        for i in [18, 79, 211] { assert!(!reading.history.contains_key(&i)); }
        // Wrong unmasked data cannot be repaired from the recorded payload.
        let mut wrong = data.clone(); wrong[223 * 8] ^= 1;
        let mut wrong_view = view.clone(); wrong_view[223 * 8] ^= 1;
        assert!(reference_reading(&wrong, &wrong_view, &reference, &expected, &zeroed, &masked, None, &history, &[]).unwrap().is_none());
        let mut corrupt_reference = reference.clone(); corrupt_reference[0] ^= 1;
        assert!(reference_reading(&data, &view, &corrupt_reference, &expected, &zeroed, &masked, None, &history, &[]).is_err());
    }

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

    /// longwin: pk120e30p-20230331 step 9, allocation 194 words 1792..1795 (a test machine <local path>):
    /// a 21-byte array of zeros written over a dead land pointer table. Words 0 and 1 are data now, word 2 keeps only
    /// the old pointer's byte 5, word 3 is still the whole pointer.
    #[test]
    fn five_rewritten_low_bytes_keep_the_forward_top_bytes() {
        let ranges = [(0x76e5_6d8b_0400_u64, 0x76e5_6d8b_0400 + 164_352, 0x7784_2d8b_0400_u64),
                      (0x76e5_6d8d_8600, 0x76e5_6d8d_8600 + 164_352, 0x7784_2d8d_8600),
                      (0x76e5_710e_0a00, 0x76e5_710e_0a00 + 164_352, 0x7784_310e_0a00),
                      (0x76e5_71a4_0c00, 0x76e5_71a4_0c00 + 164_352, 0x7784_31a4_0c00)];
        let (low, high) = (ranges[0].0, ranges[3].1);
        let step8 = bytes(&[0x76e5_6d8b_0400, 0x76e5_710e_0a00, 0x76e5_71a4_0c00, 0x76e5_6d8d_8600]);
        let (view, history, half) = forward_address_view_tracked(&step8, &ranges, low, high, None);
        assert_eq!(words(&view.unwrap().0), vec![0x7784_2d8b_0400, 0x7784_310e_0a00, 0x7784_31a4_0c00, 0x7784_2d8d_8600]);
        assert_eq!((history.len(), half), (4, 0));
        let step9 = bytes(&[0, 0, 0x7600_0000_0000, 0x76e5_6d8d_8600]);
        // without the history the lone top byte stays the replay's (the defect: the hash follows the process)
        assert_eq!(words(&forward_address_view(&step9, &ranges, low, high).unwrap().0), vec![0, 0, 0x7600_0000_0000, 0x7784_2d8d_8600]);
        let (view, history9, half) = forward_address_view_tracked(&step9, &ranges, low, high, Some(&history));
        assert_eq!((words(&view.unwrap().0), half), (vec![0, 0, 0x7700_0000_0000, 0x7784_2d8d_8600], 1));
        // the word stays: the same reading at the next boundary
        let (view, _, half) = forward_address_view_tracked(&step9, &ranges, low, high, Some(&history9));
        assert_eq!((words(&view.unwrap().0), half), (vec![0, 0, 0x7700_0000_0000, 0x7784_2d8d_8600], 1));
        // five data bytes that are not zero are kept under the forward's top bytes
        let data5 = bytes(&[0, 0, 0x7601_0203_0405, 0x76e5_6d8d_8600]);
        let (view, _, half) = forward_address_view_tracked(&data5, &ranges, low, high, Some(&history));
        assert_eq!((words(&view.unwrap().0), half), (vec![0, 0, 0x7701_0203_0405, 0x7784_2d8d_8600], 1));
        // another top byte is not the old pointer's: the word is plain data and stays as it is
        let other = bytes(&[0, 0, 0x7500_0000_0000, 0x76e5_6d8d_8600]);
        let (view, _, half) = forward_address_view_tracked(&other, &ranges, low, high, Some(&history));
        assert_eq!((words(&view.unwrap().0), half), (vec![0, 0, 0x7500_0000_0000, 0x7784_2d8d_8600], 0));
        // a word that is a live replay address takes the ordinary rule, never this one
        let live = bytes(&[0, 0, 0x76e5_6d8b_0408, 0x76e5_6d8d_8600]);
        let (view, _, half) = forward_address_view_tracked(&live, &ranges, low, high, Some(&history));
        assert_eq!((words(&view.unwrap().0), half), (vec![0, 0, 0x7784_2d8b_0408, 0x7784_2d8d_8600], 0));
    }

    fn sha(data: &[u8]) -> String { super::hex_bytes(&<super::Sha256 as super::Digest>::digest(data)) }

    /// Reduced saved pk120/r8 step-1 allocation3598 witness. Two genuine table
    /// pointers coexist with unchanged 0x7ce600000000 data and a declared hist
    /// residue. The old masked path had no mixed reading and rejected this.
    /// machine: the threaded address scan of a large block returns what the same scan returns for each of its
    /// halves read alone (words are read independently), and rewrites the same words.
    #[test]
    fn threaded_address_scan_matches_piecewise_scan() {
        let (lo, hi, rec) = (0x7000_0000_0000_u64, 0x7000_0010_0000_u64, 0x7100_0000_0000_u64);
        let ranges = [(lo, hi, rec)];
        let words = (super::DIGEST_SCAN_PAR_BYTES / 8) + 4096 + 3;
        let mut data = vec![0_u8; words * 8 + 5];
        let mut x = 0x9e37_79b9_7f4a_7c15_u64;
        let mut previous = super::PointerHistory::new();
        for i in 0..words {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            let w = match x % 97 { 0 => lo + (x >> 20) % (hi - lo), 1 => (lo + 0x40) & super::HIGH_HALF, 2 => 0x40, _ => x >> 3 };
            data[8 * i..8 * i + 8].copy_from_slice(&w.to_le_bytes());
            if x % 97 == 1 || x % 97 == 2 || x % 1021 == 5 { previous.insert(i, (lo + 0x40, rec + 0x40)); }
        }
        let (view, history, halves) = forward_address_view_tracked(&data, &ranges, lo, hi, Some(&previous));
        let (view, moved) = view.unwrap();
        // the same bytes cut at a 256-byte block boundary below the threaded size: each half takes the one-thread scan
        let cut = (super::DIGEST_SCAN_PAR_BYTES / 2 / 256) * 256;
        let shift = |p: &super::PointerHistory, by: usize, n: usize| -> super::PointerHistory {
            p.iter().filter(|(k, _)| **k >= by && **k < by + n).map(|(k, v)| (k - by, *v)).collect() };
        let (va, ha, xa) = forward_address_view_tracked(&data[..cut], &ranges, lo, hi, Some(&shift(&previous, 0, cut / 8)));
        let (vb, hb, xb) = forward_address_view_tracked(&data[cut..], &ranges, lo, hi, Some(&shift(&previous, cut / 8, words)));
        assert!(data[cut..].len() < super::DIGEST_SCAN_PAR_BYTES && data.len() >= super::DIGEST_SCAN_PAR_BYTES);
        let (va, ma) = va.unwrap(); let (vb, mb) = vb.unwrap();
        assert_eq!(moved, ma + mb);
        assert_eq!(halves, xa + xb);
        assert!(view[..cut] == va[..] && view[cut..] == vb[..]);
        let mut joined = ha.clone();
        for (k, v) in hb { joined.insert(k + cut / 8, v); }
        assert_eq!(history, joined);
        assert!(moved > 1000 && halves > 1000);
    }
    #[test]
    fn masked_residue_keeps_proven_pointers_and_raw_lookalikes() {
        let ranges = [(0x7ce3_f3cf_8a00, 0x7ce3_f3cf_8b00, 0x714f_c5af_8a00),
                      (0x7ce3_f714_b000, 0x7ce3_f714_b100, 0x714f_c7d4_b000),
                      (0x7ce6_0000_0000, 0x7ce6_0015_5c00, 0x7152_2680_0000)];
        let replay = bytes(&[0x7ce3_f3cf_8a00, 0x7ce3_f714_b000, 0x7ce6_0000_0000,
                            0x7ce6_0000_0000, 0x1234, 0x3b]);
        let forward = bytes(&[0x714f_c5af_8a00, 0x714f_c7d4_b000, 0x7ce6_0000_0000,
                             0x7ce6_0000_0000, 0, 0x3b]);
        let mut previous = super::PointerHistory::new();
        previous.insert(0, (0x7ce3_f3cf_8a00, 0x714f_c5af_8a00));
        previous.insert(1, (0x7ce3_f714_b000, 0x714f_c7d4_b000));
        let (view, mut history, _) = forward_address_view_tracked(&replay, &ranges,
            ranges[0].0, ranges[2].1, Some(&previous));
        let expected = sha(&forward);
        let view = view.unwrap().0;
        assert_ne!(super::masked_digest(&mut view.clone(), &[(32, 8)]).unwrap(), expected);
        assert_ne!(super::masked_digest(&mut replay.clone(), &[(32, 8)]).unwrap(), expected);
        // machine: the translated view goes in, as digest_blocks passes it (the mixed reading is read from it).
        let (actual, _) = super::residue_masked_reading(&replay, Some(&view), &ranges, ranges[0].0,
            ranges[2].1, Some(&previous), &[(32, 8)], &expected, &mut history).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(history, previous);
        // A real unmasked data error must still fail against the same digest.
        let mut wrong = replay.clone(); wrong[40] ^= 1;
        let (wrong_view, mut wrong_history, _) = forward_address_view_tracked(&wrong, &ranges,
            ranges[0].0, ranges[2].1, Some(&previous));
        let wrong_view = wrong_view.unwrap().0;
        let (actual, _) = super::residue_masked_reading(&wrong, Some(&wrong_view), &ranges, ranges[0].0,
            ranges[2].1, Some(&previous), &[(32, 8)], &expected, &mut wrong_history).unwrap();
        assert_ne!(actual, expected);
    }

    /// a test machine sealed 18-step plan, allocation 454: one stale pointer (replay 0x7c8524162514 -> forward
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
        assert_eq!(mixed_reading(&replay, &view, &expected, None, None), Some((expected.clone(), vec![3])));
        // nothing reads right: no match, the block still fails
        let other = bytes(&[1, 0x798a_3416_2514, 0x3b, 0x7c32_2f05_fd1b]);
        assert_eq!(mixed_reading(&replay, &view, &sha(&other), None, None), None);
        // the pointer word was translated from the same replay value at the last boundary: only the data word is
        // in doubt (a test machine proof18c2, allocation 446 step 18: 61 translated words, most proven at step 17)
        let mut previous = super::PointerHistory::new();
        previous.insert(1, (0x7c85_2416_2514, 0x798a_3416_2514));
        assert_eq!(mixed_reading(&replay, &view, &expected, Some(&previous), None), Some((expected.clone(), vec![3])));
        // a look-alike word proven nowhere and a pointer proven: reverting the pointer is never needed
        previous.insert(3, (0x7c32_2f05_fd1b, 0x7e57_1005_fd1b));
        assert_eq!(mixed_reading(&replay, &view, &expected, Some(&previous), None), None);
    }

    struct Rng(u64);
    impl Rng { fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 } }

    fn block(words: usize, n: usize, seed: u64) -> (Vec<u8>, Vec<u8>, Vec<usize>) {
        let mut rng = Rng(seed);
        let raw: Vec<u8> = (0..words).flat_map(|_| rng.next().to_le_bytes()).collect();
        let mut doubt: Vec<usize> = (0..n - 2).map(|k| k * (words - 2) / (n - 2)).collect();
        doubt.push(doubt[n / 2 - 1] + 1);
        doubt.push(words - 1);
        doubt.sort(); doubt.dedup();
        assert_eq!(doubt.len(), n);
        let mut viewed = raw.clone();
        for &i in &doubt { viewed[8 * i..8 * i + 8].copy_from_slice(&(0x7e57_0000_0000 | (rng.next() & 0xffff_ffff)).to_le_bytes()); }
        (raw, viewed, doubt)
    }

    #[test]
    fn mixed_reading_reverts_one_or_two_lookalikes_among_many() {
        let (raw, viewed, doubt) = block(4099, 64, 43);
        for back in [vec![doubt[5]], vec![doubt[5], doubt[50]], vec![doubt[62], doubt[63]]] {
            let mut want = viewed.clone();
            for &i in &back { want[8 * i..8 * i + 8].copy_from_slice(&raw[8 * i..8 * i + 8]); }
            let expected = sha(&want);
            assert_eq!(mixed_reading(&raw, &viewed, &expected, None, None), Some((expected.clone(), back.clone())), "{back:?}");
        }
        // three lookalikes: outside the search, as before
        let mut want = viewed.clone();
        for &i in &[doubt[1], doubt[2], doubt[3]] { want[8 * i..8 * i + 8].copy_from_slice(&raw[8 * i..8 * i + 8]); }
        assert_eq!(mixed_reading(&raw, &viewed, &sha(&want), None, None), None);
    }

    /// g1 hunt: more doubtful words than the complement search takes (pk60e29p allocation 421 at step 6: 51 table
    /// pointers the runner noted, and 65 to 108 data words a replay allocation happened to cover). Every doubtful word
    /// read as data, with the proven words translated, gives the recorded hash. One real pointer among the doubtful
    /// words, or one data byte off, is still refused.
    #[test]
    fn mixed_reading_reads_more_than_64_lookalikes_as_data() {
        let (raw, viewed, changed) = block(8209, 159, 47);
        let word = |b: &[u8], i: usize| u64::from_le_bytes(b[8 * i..8 * i + 8].try_into().unwrap());
        let proven: Vec<usize> = changed.iter().copied().step_by(3).take(51).collect();
        let doubt: Vec<usize> = changed.iter().copied().filter(|i| !proven.contains(i)).collect();
        assert_eq!((proven.len(), doubt.len()), (51, 108));
        let previous: super::PointerHistory = proven.iter().map(|&i| (i, (word(&raw, i), word(&viewed, i)))).collect();
        let mut want = viewed.clone();
        for &i in &doubt { want[8 * i..8 * i + 8].copy_from_slice(&raw[8 * i..8 * i + 8]); }
        let expected = sha(&want);
        assert_eq!(mixed_reading(&raw, &viewed, &expected, Some(&previous), None), Some((expected.clone(), doubt.clone())));
        // one of the doubtful words is a real pointer: not this reading
        let mut one = want.clone();
        one[8 * doubt[40]..8 * doubt[40] + 8].copy_from_slice(&viewed[8 * doubt[40]..8 * doubt[40] + 8]);
        assert_eq!(mixed_reading(&raw, &viewed, &sha(&one), Some(&previous), None), None);
        // one data byte differs from the recording: refused
        let mut off = want.clone();
        off[5] ^= 1;
        assert_eq!(mixed_reading(&raw, &viewed, &sha(&off), Some(&previous), None), None);
        // without the proof of the 51, all 159 are in doubt and the recording (51 translated) is not this reading
        assert_eq!(mixed_reading(&raw, &viewed, &expected, None, None), None);
    }

    /// g1 hunt: pk60e29p allocation 421 at step 11: data lookalikes and one old pointer whose low half a 4-byte
    /// store rewrote. Proven words translated, lookalikes as data, the half word by the half rule: the recorded hash.
    #[test]
    fn lookalike_half_reading_reads_a_half_word_among_lookalikes() {
        for (n, proven_n) in [(92_usize, 1_usize), (20, 3)] {
            let (raw, viewed, changed) = block(8209, n, 53);
            let word = |b: &[u8], i: usize| u64::from_le_bytes(b[8 * i..8 * i + 8].try_into().unwrap());
            let proven: Vec<usize> = changed.iter().copied().step_by(7).take(proven_n).collect();
            let doubt: Vec<usize> = changed.iter().copied().filter(|i| !proven.contains(i)).collect();
            let h = doubt[doubt.len() / 2];
            let mut previous: super::PointerHistory = proven.iter().map(|&i| (i, (word(&raw, i), word(&viewed, i)))).collect();
            // the last boundary held a full pointer at h: same high half as the word now, another low half
            let (old_replay, old_forward) = (word(&raw, h) ^ 0x10, 0x7e57_0000_0000_u64 | 0x4321);
            previous.insert(h, (old_replay, old_forward));
            let current: super::PointerHistory = changed.iter().map(|&i| (i, (word(&raw, i), word(&viewed, i)))).collect();
            let half_value = (old_forward & super::HIGH_HALF) | (word(&raw, h) & !super::HIGH_HALF);
            let mut want = viewed.clone();
            for &i in &doubt { want[8 * i..8 * i + 8].copy_from_slice(&raw[8 * i..8 * i + 8]); }
            want[8 * h..8 * h + 8].copy_from_slice(&half_value.to_le_bytes());
            let expected = sha(&want);
            let others: Vec<usize> = doubt.iter().copied().filter(|&i| i != h).collect();
            assert_eq!(super::lookalike_half_reading(&raw, &viewed, &expected, Some(&previous), Some(&current)), Some((others, vec![h])), "{n}");
            // one data byte differs from the recording: refused
            let mut off = want.clone();
            off[9] ^= 1;
            assert_eq!(super::lookalike_half_reading(&raw, &viewed, &sha(&off), Some(&previous), Some(&current)), None);
            // no old pointer at h: no half reading
            previous.remove(&h);
            assert_eq!(super::lookalike_half_reading(&raw, &viewed, &expected, Some(&previous), Some(&current)), None);
        }
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
mod f1_history_tests {
    use super::*;
    #[test]
    fn sparse_history_keeps_half_word_precedence_and_exact_counters() {
        let lo = 0x7800_0000_0000_u64;
        let recorded = 0x7e00_0000_4000_u64;
        let ranges = [(lo, lo + 0x1000, recorded)];
        let mut words = vec![0x1234_5678_9abc_def0_u64; 65538];
        words[1] = lo + 16;               // unchanged full pointer: ordinary translation
        words[32] = lo;                   // high half: history wins over the live range
        words[1024] = lo + 0x2000;        // rewritten low half, outside the live range
        words[2048] = lo + 30;            // rewritten low half inside the live range
        words[4096] = 0x88babe;           // unrelated replacement drops its old history
        words[32768] = lo + 0x1000;       // inclusive end of the live range
        words[65536] = 0x28;             // low half at a distant sparse position
        let previous: PointerHistory = [
            (65536, (lo + 0x28, recorded + 0x28)),
            (4096, (lo + 24, recorded + 24)),
            (2048, (lo + 24, recorded + 24)),
            (1024, (lo + 24, recorded + 24)),
            (32, (lo + 24, recorded + 24)),
            (1, (lo + 16, recorded + 16)),
            (usize::MAX, (lo, recorded)), // out-of-allocation history is never read
        ].into_iter().collect();
        let mut data: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        data.extend_from_slice(&[0xa5, 0x78, 0x0]); // partial tail stays unchanged
        let mut expected_words = words;
        expected_words[1] = recorded + 16;
        expected_words[32] = recorded & HIGH_HALF;
        expected_words[1024] = (recorded & HIGH_HALF) | 0x2000;
        expected_words[2048] = recorded + 30;
        expected_words[32768] = recorded + 0x1000;
        expected_words[65536] = (recorded + 0x28) & !HIGH_HALF;
        let mut expected: Vec<u8> = expected_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        expected.extend_from_slice(&[0xa5, 0x78, 0x0]);
        let expected_history: PointerHistory = [
            (1, (lo + 16, recorded + 16)), (32, previous[&32]),
            (1024, previous[&1024]), (2048, (lo + 30, recorded + 30)),
            (32768, (lo + 0x1000, recorded + 0x1000)), (65536, previous[&65536]),
        ].into_iter().collect();
        let (view, history, halves) = forward_address_view_tracked(&data, &ranges, lo, lo + 0x1000, Some(&previous));
        assert_eq!(view, Some((expected, 3)));
        assert_eq!(history, expected_history);
        assert_eq!(halves, 3);
        // An unchanged half reading still retains history/count, without creating a view.
        let unchanged = [(0, (lo + 0x40, 0x7e00_0000_0040))].into_iter().collect();
        let (view, history, halves) = forward_address_view_tracked(&0x40_u64.to_le_bytes(), &ranges, lo, lo + 0x1000, Some(&unchanged));
        assert_eq!(view, None);
        assert_eq!(history, unchanged);
        assert_eq!(halves, 1);
    }
}

#[cfg(test)]
mod f1_tests {
    use super::*;
    #[test]
    fn residue_reuses_exact_normalized_view_and_preserves_fallbacks() {
        let ranges = [(0x7800_0000_0000, 0x7800_0000_1000, 0x7e00_0000_0000)];
        let (lo, hi) = (ranges[0].0, ranges[0].1);
        let previous = [(0, (lo + 16, ranges[0].2 + 16))].into_iter().collect();
        // Half-cleared pointer, live pointer, ordinary data and unaligned residue.
        let data: Vec<u8> = [lo, lo + 24, 0x1234_u64].iter().flat_map(|v| v.to_le_bytes()).collect();
        let (view, history, _) = forward_address_view_tracked(&data, &ranges, lo, hi, Some(&previous));
        let original = view.as_ref().unwrap().0.clone();
        let zeroed = [(17, 3), (18, 4)];
        let expected = masked_digest(&mut original.clone(), &zeroed).unwrap();
        let mut h = history.clone();
        let got = residue_masked_reading(&data, Some(&original), &ranges, lo, hi, Some(&previous), &zeroed, &expected, &mut h).unwrap();
        assert_eq!(got, (expected.clone(), None));
        assert_eq!(h, history);
        assert_eq!(view.unwrap().0, original); // immutable view remains usable by other checks
        // Raw-byte fallback still wins when the translated word is data.
        let expected_raw = masked_digest(&mut data.clone(), &zeroed).unwrap();
        assert_eq!(residue_masked_reading(&data, Some(&original), &ranges, lo, hi, Some(&previous), &zeroed,
            &expected_raw, &mut h).unwrap(), (expected_raw, None));
        assert_eq!(residue_masked_reading(&data, None, &ranges, lo, hi, None, &[(data.len(), 1)],
            &expected, &mut h).unwrap_err(), "residue range outside its allocation");
        // Same defect rejection: a bad reference must not become equal after reuse.
        assert_ne!(residue_masked_reading(&data, Some(&original), &ranges, lo, hi, Some(&previous), &zeroed,
            &"0".repeat(64), &mut h).unwrap().0, "0".repeat(64));
    }
    /// a development line race C: a data word inside a live replay range is rewritten by the view, the residue block passes
    /// on its raw bytes, and the word must not stay in the pointer history. At the next boundary the same word holds
    /// new data with the same high half beside a real pointer; with the stale pair the half-word rule gave it the
    /// forward's high half (0x735c00001234) and the block could not pass.
    #[test]
    fn residue_raw_pass_leaves_no_history() {
        let ranges = [(0x100_1e80_0000_u64, 0x100_1f0c_3000_u64, 0x735c_33e0_0000_u64)];
        let (lo, hi) = (ranges[0].0, ranges[0].1);
        let zeroed = [(16, 8)];
        let data: Vec<u8> = [0x100_1e82_7304_u64, 7, 99].iter().flat_map(|v| v.to_le_bytes()).collect();
        let (view, mut history, _) = forward_address_view_tracked(&data, &ranges, lo, hi, None);
        let view = view.unwrap().0;
        assert_eq!(history.get(&0), Some(&(0x100_1e82_7304_u64, 0x735c_33e2_7304_u64)));
        // the forward held the same data word; its residue word differs
        let forward: Vec<u8> = [0x100_1e82_7304_u64, 7, 98].iter().flat_map(|v| v.to_le_bytes()).collect();
        let expected = masked_digest(&mut forward.clone(), &zeroed).unwrap();
        assert_eq!(residue_masked_reading(&data, Some(&view), &ranges, lo, hi, None, &zeroed, &expected, &mut history).unwrap(),
            (expected.clone(), None));
        assert!(history.is_empty());
        // next boundary: new low half under the same high half, outside every live range; word 1 is a real pointer
        let next: Vec<u8> = [0x100_0000_1234_u64, lo + 8, 5].iter().flat_map(|v| v.to_le_bytes()).collect();
        let forward: Vec<u8> = [0x100_0000_1234_u64, ranges[0].2 + 8, 6].iter().flat_map(|v| v.to_le_bytes()).collect();
        let (view, mut later, halves) = forward_address_view_tracked(&next, &ranges, lo, hi, Some(&history));
        assert_eq!(halves, 0);
        let expected = masked_digest(&mut forward.clone(), &zeroed).unwrap();
        assert_eq!(residue_masked_reading(&next, Some(&view.unwrap().0), &ranges, lo, hi, Some(&history), &zeroed, &expected,
            &mut later).unwrap(), (expected.clone(), None));
    }
    /// a development line fix-check (2026-11-11): the raw pass that drops the address readings is a pass on EQUAL bytes only.
    /// The same block with one planted wrong data word outside the declared residue (the replay holds 7 where the
    /// recording holds 8) is refused: no reading gives the recorded masked hash, and nothing leaves the history.
    /// THE BREAKAGE THIS PREVENTS: a later change that clears the history, or accepts the block, on anything weaker
    /// than the recorded hash would let a 3 km storm replay with a wrong model word pass allocation 189.
    #[test]
    fn residue_raw_pass_refuses_one_wrong_data_word() {
        let ranges = [(0x100_1e80_0000_u64, 0x100_1f0c_3000_u64, 0x735c_33e0_0000_u64)];
        let (lo, hi) = (ranges[0].0, ranges[0].1);
        let zeroed = [(16, 8)];
        let data: Vec<u8> = [0x100_1e82_7304_u64, 7, 99].iter().flat_map(|v| v.to_le_bytes()).collect();
        let (view, mut history, _) = forward_address_view_tracked(&data, &ranges, lo, hi, None);
        let view = view.unwrap().0;
        let before = history.clone();
        assert!(!before.is_empty());
        let forward: Vec<u8> = [0x100_1e82_7304_u64, 8, 98].iter().flat_map(|v| v.to_le_bytes()).collect();
        let expected = masked_digest(&mut forward.clone(), &zeroed).unwrap();
        let (masked, found) = residue_masked_reading(&data, Some(&view), &ranges, lo, hi, None, &zeroed, &expected, &mut history).unwrap();
        assert_ne!(masked, expected);
        assert_eq!(found, None);
        assert_eq!(history, before);
    }
}

#[cfg(test)]
mod f1_residue_overlap_tests {
    use super::*;
    #[test]
    fn required_large_residue_readings_join_with_exact_history_and_errors() {
        let id = 1;
        let ranges = [(0x7800_0000_0000_u64, 0x7800_0000_1000, 0x7e00_0000_0000)];
        let (low, high) = (ranges[0].0, ranges[0].1);
        let mut data = vec![7_u8; DIGEST_OVERLAP_BYTES];
        data[..8].copy_from_slice(&(low + 24).to_le_bytes());
        let zeroed = vec![(17, 37), (31, 29)];
        let (view, expected_history, expected_halves) = forward_address_view_tracked(&data, &ranges, low, high, None);
        let view = view.unwrap();
        let expected_unmasked = hex_bytes(&Sha256::digest(&view.0));
        let expected_masked = masked_digest(&mut view.0.clone(), &zeroed).unwrap();
        let refs = [((1, id), (data.len(), "unmasked residue differs by design".into()))].into_iter().collect();
        let residue = [((1, id), (zeroed, expected_masked.clone()))].into_iter().collect();
        let previous = HashMap::new();
        let ab = HashSet::new();
        let out = digest_blocks(1, &[(id, &data)], &ranges, low, high, &previous, &residue, &ab, Some(&refs), 1, false);
        assert_eq!(out.values, [expected_unmasked.clone()]);
        assert_eq!(out.moved, [view.1]);
        assert_eq!(out.halves, [expected_halves]);
        assert_eq!(out.histories, [expected_history.clone()]);
        assert_eq!(out.raw_equal, [false]);
        assert_eq!(out.residue_out, [Some(Ok((expected_masked, None)))]);
        // The masked check may fail, but its completed unmasked sibling is
        // still owned and joined. No incomplete reading escapes this result.
        let bad = [((1, id), (vec![(data.len(), 1)], "bad".into()))].into_iter().collect();
        let out = digest_blocks(1, &[(id, &data)], &ranges, low, high, &previous, &bad, &ab, Some(&refs), 1, false);
        assert_eq!(out.values, [expected_unmasked]);
        assert_eq!(out.histories, [expected_history]);
        assert_eq!(out.residue_out, [Some(Err("residue range outside its allocation".into()))]);
    }
}

#[cfg(test)]
mod f1_masked_stream_tests {
    use super::*;
    #[test]
    fn borrowed_mask_preserves_overlap_bytes_and_error_precedence() {
        let data: Vec<u8> = (0..(2 * (64 << 10) + 43)).map(|i| ((i * 31 + 7) & 255) as u8).collect();
        let n = data.len();
        let cases = [
            vec![], vec![(0, 0), (n, 0)], vec![(0, n)],
            vec![(17, 37), (31, 29), (4, 21), (60, 1)],
            vec![(n - 11, 11), (65_535, 3), (0, 1)],
            vec![(70_000, 60_000), (2, 100_000), (17, 1), (2, 100_000)],
        ];
        for ranges in cases {
            let mut oracle = data.clone();
            assert_eq!(masked_digest_reading(&data, &ranges), masked_digest(&mut oracle, &ranges), "{ranges:?}");
        }
        assert_eq!(masked_digest_reading(&[], &[(0, 0)]), masked_digest(&mut [], &[(0, 0)]));
        // The original first error wins even when a later range would overflow.
        for ranges in [
            vec![(n + 1, 0), (usize::MAX, 1)],
            vec![(usize::MAX, 1), (n + 1, 0)],
            vec![(3, 12), (n, 1)],
        ] {
            let mut oracle = data.clone();
            let expected = masked_digest(&mut oracle, &ranges);
            assert!(expected.is_err());
            assert_eq!(masked_digest_reading(&data, &ranges), expected, "{ranges:?}");
        }
        // The real residue reader takes the normalized and the raw fallback decision from the borrowed view, which
        // stays unchanged for the concurrent hash. A normalized pass keeps the boundary's address readings for the
        // next boundary (the half-word rules read with them: sealed 18-step hrrrv5 plan, allocation 163 word 93
        // passed step 12 and failed step 13 without). A raw pass drops them (a development line, 1b037c40): the masked hash
        // has just shown that the forward holds the replay's own bytes outside the residue, so word 1 (bytes 8..16,
        // outside the residue 17..60; the view holds 0x7e0000000020, the raw bytes do not) is data, not an address.
        // Until 2026-11-11 this test asked a raw pass to keep the history as well. That pinned the streamed-mask
        // change (a449bc5e) to the behaviour of the code it replaced and named no breakage; no boundary can need a
        // reading that the hash of the same boundary has disproved.
        let mut translated = data.clone();
        translated[8..16].copy_from_slice(&0x7e00_0000_0020_u64.to_le_bytes());
        let zeroed = [(17, 37), (31, 29)];
        let expected_normalized = masked_digest(&mut translated.clone(), &zeroed).unwrap();
        let expected_raw = masked_digest(&mut data.clone(), &zeroed).unwrap();
        assert_ne!(expected_normalized, expected_raw);
        let original = translated.clone();
        let mut saved_history = PointerHistory::new();
        saved_history.insert(1, (0x7800_0000_0020, 0x7e00_0000_0020));
        for (expected, kept) in [(expected_normalized, true), (expected_raw, false)] {
            let mut history = saved_history.clone();
            let result = residue_masked_reading(&data, Some(&translated), &[], 0, 0, None, &zeroed, &expected, &mut history);
            assert_eq!(result.unwrap(), (expected, None));
            assert_eq!(translated, original);
            assert_eq!(history, if kept { saved_history.clone() } else { PointerHistory::new() });
        }
    }
}

#[cfg(test)]
mod f1_range_prefilter_tests {
    use super::*;
    #[test]
    fn unsigned_envelope_preserves_boundary_lanes_history_and_trailing_bytes() {
        if !address_range_prefilter_available() {
            eprintln!("F1 AVX2 prefilter unavailable on this test CPU");
            return;
        }
        let sign = 1u64 << 63;
        // Test every SIMD lane at both inclusive ends and across the signed
        // boundary, including an unaligned load and an empty envelope.
        for (low, high) in [(0, 0), (u64::MAX, u64::MAX), (7, 13),
                            (sign - 1, sign + 1), (sign, sign), (u64::MAX, 0), (0, u64::MAX)] {
            let outside = if low != 0 { 0 } else { u64::MAX };
            for word in [0, 1, low.wrapping_sub(1), low, high, high.wrapping_add(1), sign - 1, sign, u64::MAX] {
                for at in 0..32 {
                    let mut storage = vec![0u8; 257];
                    for chunk in storage[1..].chunks_exact_mut(8) { chunk.copy_from_slice(&outside.to_le_bytes()); }
                    storage[1 + 8 * at..1 + 8 * at + 8].copy_from_slice(&word.to_le_bytes());
                    let scalar = storage[1..].chunks_exact(8).any(|c| {
                        let w = u64::from_le_bytes(c.try_into().unwrap());
                        w >= low && w <= high
                    });
                    assert_eq!(unsafe { address_range_may_contain_pointer(&storage[1..], low, high) }, scalar,
                               "lane {at}, word {word:#x}, range {low:#x}..={high:#x}");
                }
            }
        }
        let low = sign + 0x5000;
        let high = low + 0xfff;
        let ranges = [(low, high, 0x1000)];
        let old = (0x7800_1234_0000_0042u64, 0x7e00_5678_0000_0042u64);
        let previous = [(96usize, old)].into_iter().collect();
        let mut data = vec![0u8; 129 * 8 + 7];
        for (index, word) in [(31, low - 1), (32, low), (63, high), (64, high + 1),
                              (96, old.0 & HIGH_HALF), (128, low + 7)] {
            data[8 * index..8 * index + 8].copy_from_slice(&word.to_le_bytes());
        }
        data[129 * 8..].fill(0xa5);
        let mut expected = data.clone();
        for (index, word) in [(32, 0x1000u64), (63, 0x1fff), (96, old.1 & HIGH_HALF), (128, 0x1007)] {
            expected[8 * index..8 * index + 8].copy_from_slice(&word.to_le_bytes());
        }
        let (view, history, halves) = forward_address_view_tracked(&data, &ranges, low, high, Some(&previous));
        assert_eq!(view, Some((expected, 3)));
        assert_eq!(halves, 1);
        assert_eq!(history, [(32, (low, 0x1000)), (63, (high, 0x1fff)), (96, old),
                            (128, (low + 7, 0x1007))].into_iter().collect());
        // These are integer bit patterns only. No floating arithmetic or P7
        // conversion is introduced by the prefilter.
        let bits = [0u64, 1, sign, 0x7ff0_0000_0000_0000, 0xfff0_0000_0000_0000,
                    0x7ff8_1234_5678_9abc, 0xffff_ffff_ffff_ffff, 0x000f_ffff_ffff_ffff];
        let special: Vec<u8> = bits.iter().cycle().take(128).flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(forward_address_view_tracked(&special, &ranges, low, high, None),
                   (None, PointerHistory::new(), 0));
        eprintln!("F1 AVX2 prefilter exercised: unsigned lanes, inclusive bounds, history and trailing bytes exact");
    }
}

/// addr-s (a development line, 2026-11-11): the step check must not let this run's memory layout choose a word's reading.
/// These tests use only functions the base (830ca771) has, so the same module is run on the base as the "before".
#[cfg(test)]
mod addr_s_tests {
    use super::*;
    use sha2::{Digest, Sha256};
    fn bytes(words: &[u64]) -> Vec<u8> { words.iter().flat_map(|w| w.to_le_bytes()).collect() }
    fn sha(data: &[u8]) -> String { hex_bytes(&Sha256::digest(data)) }
    // Run B of a test machine <local path> (120 km, 2023-07-12, block 347, step 9):
    // the live replay block 0x733636000000..0x73363a964000 that covers the half-written word 0x73363a48cf2f, and a
    // live block that covers the index pairs 0x0000733800007337. Recorded bases as run A printed them (0x7e3b...).
    const LIVE: (u64, u64, u64) = (0x7336_3600_0000, 0x7336_3a96_4000, 0x7e3c_1000_0000);
    const PAIRS: (u64, u64, u64) = (0x7338_0000_0000, 0x7338_0001_0000, 0x7e3d_0000_0000);
    const PAIR: u64 = 0x0000_7338_0000_7337;
    const REAL: (u64, u64) = (0x7336_3600_0040, 0x7e3c_1000_0040);      // a real address, noted at the last boundary
    const FIVE: (u64, u64) = (0x7336_0000_2000, 0x7e3b_0000_2000);      // old address of word 4874434 (now 0x730000000000)
    const OLD: (u64, u64) = (0x7336_1234_5600, 0x7e3b_1234_5600);       // old address of word 4874525
    const HALF: u64 = 0x7336_3a48_cf2f;                                 // 0x3a48cf2f written over its low half
    const HALF_RECORDED: u64 = 0x7e3b_3a48_cf2f;

    /// The step check of one ordinary block: (equal to the recording, the notes kept for the next boundary).
    fn check(replay: &[u64], recorded: &[u64], previous: &[(usize, (u64, u64))]) -> (bool, PointerHistory) {
        let (data, want) = (bytes(replay), sha(&bytes(recorded)));
        let ranges = [LIVE, PAIRS];
        let refs: References = HashMap::from([((9_usize, 347_u64), (data.len(), want.clone()))]);
        let before: HashMap<u64, PointerHistory> = HashMap::from([(347_u64, previous.iter().copied().collect::<PointerHistory>())]);
        let out = digest_blocks(9, &[(347_u64, data.as_slice())], &ranges, ranges[0].0, ranges[1].1, &before,
                                &HashMap::new(), &HashSet::new(), Some(&refs), 1, false);
        (out.values[0] == want, out.histories[0].clone())
    }

    #[test]
    fn block_347_half_written_address_inside_a_live_block_as_dumped() {
        // as the two dumps show it: one real address, the five-byte word, the half-written word, two index pairs
        let replay = [REAL.0, 0x7300_0000_0000, HALF, PAIR, PAIR, 0x3ff0_0000_0000_0000, 7];
        let recorded = [REAL.1, 0x7e00_0000_0000, HALF_RECORDED, PAIR, PAIR, 0x3ff0_0000_0000_0000, 7];
        let previous = [(0, REAL), (1, FIVE), (2, OLD)];
        let (equal, notes) = check(&replay, &recorded, &previous);
        assert!(equal, "block 347 as dumped must read equal to the recording");
        assert_eq!(notes.get(&2), Some(&OLD), "the half-written word keeps its old address for the next boundary");
        assert!(!notes.contains_key(&3) && !notes.contains_key(&4), "index pairs are data: no note");
    }

    #[test]
    fn block_347_half_written_address_inside_a_live_block_with_six_pairs_and_a_new_address() {
        // run A's six index pairs, and one address the step wrote that no note knows yet
        let new = (0x7336_3700_0000_u64, 0x7e3c_1100_0000_u64);
        let replay = [REAL.0, 0x7300_0000_0000, HALF, PAIR, PAIR, PAIR + 2, PAIR + 2, PAIR + 4, PAIR + 4, new.0, 7];
        let recorded = [REAL.1, 0x7e00_0000_0000, HALF_RECORDED, PAIR, PAIR, PAIR + 2, PAIR + 2, PAIR + 4, PAIR + 4, new.1, 7];
        let previous = [(0, REAL), (1, FIVE), (2, OLD)];
        let (equal, notes) = check(&replay, &recorded, &previous);
        assert!(equal, "half-written address inside a live block, six index pairs and a new address in one block");
        assert_eq!(notes.get(&2), Some(&OLD));
        assert_eq!(notes.get(&9), Some(&new));
        assert!((3..9).all(|i| !notes.contains_key(&i)), "index pairs are data: no note");
        // SOUNDNESS: one planted data change (word 10: 7 -> 8) must still mismatch under every reading
        let mut planted = replay; planted[10] = 8;
        assert!(!check(&planted, &recorded, &previous).0, "a planted data change passed");
        // and one index pair off by one cell
        let mut planted = replay; planted[5] = PAIR + 3;
        assert!(!check(&planted, &recorded, &previous).0, "a planted change of an index pair passed");
        // and the half-written word with other data bytes
        let mut planted = replay; planted[2] = HALF ^ 0x100;
        assert!(!check(&planted, &recorded, &previous).0, "a planted change of the half-written word passed");
    }

    #[test]
    fn five_rewritten_low_bytes_inside_a_live_block_among_eight_index_pairs() {
        // an old address 0x734000001000 whose five low bytes were rewritten: 0x733638123456 falls inside LIVE. With it,
        // eight plain index pairs that look like addresses (more than two in one block).
        let old = (0x7340_0000_1000_u64, 0x7e61_0000_1000_u64);
        let word = 0x7336_3812_3456_u64;
        let mut replay = vec![REAL.0, word];
        let mut recorded = vec![REAL.1, 0x7e36_3812_3456];
        for k in 0..8 { replay.push(PAIR + 2 * (k / 2)); recorded.push(PAIR + 2 * (k / 2)); }
        replay.push(7); recorded.push(7);
        let previous = [(0, REAL), (1, old)];
        let (equal, notes) = check(&replay, &recorded, &previous);
        assert!(equal, "five rewritten low bytes inside a live block, among eight index pairs");
        assert_eq!(notes.get(&1), Some(&old));
        assert!((2..10).all(|i| !notes.contains_key(&i)), "index pairs are data: no note");
        // SOUNDNESS: one planted data change must still mismatch under every reading
        let mut planted = replay.clone(); planted[10] = 9;
        assert!(!check(&planted, &recorded, &previous).0, "a planted data change passed");
        let mut planted = replay.clone(); planted[1] = word + 1;
        assert!(!check(&planted, &recorded, &previous).0, "a planted change of the rewritten bytes passed");
    }

    #[test]
    fn plain_index_pairs_alone_stay_data() {
        // eight index pairs beside one real noted address (passes on the base too: kept as a guard)
        let mut replay = vec![REAL.0]; let mut recorded = vec![REAL.1];
        for k in 0..8 { replay.push(PAIR + 2 * (k / 2)); recorded.push(PAIR + 2 * (k / 2)); }
        let (equal, notes) = check(&replay, &recorded, &[(0, REAL)]);
        assert!(equal);
        assert_eq!(notes.len(), 1);
        let mut planted = replay.clone(); planted[4] = PAIR + 1;
        assert!(!check(&planted, &recorded, &[(0, REAL)]).0, "a planted change of an index pair passed");
    }

    /// a development line 3 km case in small: a block with a left-out part; an index pair inside a live block at step 1
    /// (accepted on the raw bytes); the same pair at step 2 beside three real addresses.
    #[test]
    fn list_word_read_as_an_address_at_one_boundary_does_not_bind_the_next() {
        let zeroed = [(8 * 40_usize, 8 * 16_usize)];
        let ranges = [LIVE, PAIRS];
        let (low, high) = (ranges[0].0, ranges[1].1);
        let masked = |words: &[u64]| masked_digest(&mut bytes(words), &zeroed).unwrap();
        let mut step1 = vec![3_u64; 64]; step1[5] = PAIR;
        let mut rec1 = step1.clone();
        for k in 40..56 { step1[k] = 0xdead_0000 + k as u64; rec1[k] = 0xbeef_0000 + k as u64; }
        let data1 = bytes(&step1);
        let (view1, mut notes1, _) = forward_address_view_tracked(&data1, &ranges, low, high, None);
        let view1 = view1.expect("the pair is read as an address by value").0;
        let got1 = residue_masked_reading(&data1, Some(&view1), &ranges, low, high, None, &zeroed, &masked(&rec1), &mut notes1).unwrap();
        assert_eq!(got1.0, masked(&rec1), "step 1 passes on its raw bytes");
        let mut step2 = step1.clone(); let mut rec2 = rec1.clone();
        for k in 10..13 { step2[k] = LIVE.0 + 0x100 * k as u64; rec2[k] = LIVE.2 + 0x100 * k as u64; }
        let data2 = bytes(&step2);
        // (a) with the notes step 1 left, (b) with the stale note forced back in
        let mut forced = notes1.clone(); forced.insert(5, (PAIR, PAIRS.2 + 0x7337));
        for (name, before) in [("notes of step 1", notes1.clone()), ("stale note forced", forced)] {
            let (view2, mut notes2, _) = forward_address_view_tracked(&data2, &ranges, low, high, Some(&before));
            let view2 = view2.unwrap().0;
            let got2 = residue_masked_reading(&data2, Some(&view2), &ranges, low, high, Some(&before), &zeroed, &masked(&rec2), &mut notes2).unwrap();
            assert_eq!(got2.0, masked(&rec2), "step 2, {name}: every byte outside the left-out part equals the recording");
            assert!(!notes2.contains_key(&5), "step 2, {name}: the pair keeps no note");
            assert!((10..13).all(|k| notes2.contains_key(&k)), "step 2, {name}: the three real addresses are noted");
            // SOUNDNESS: a planted data change outside the left-out part must still mismatch
            let mut planted = step2.clone(); planted[20] = 4;
            let data = bytes(&planted);
            let (view, mut notes, _) = forward_address_view_tracked(&data, &ranges, low, high, Some(&before));
            let got = residue_masked_reading(&data, Some(&view.unwrap().0), &ranges, low, high, Some(&before), &zeroed, &masked(&rec2), &mut notes).unwrap();
            assert_ne!(got.0, masked(&rec2), "step 2, {name}: a planted data change passed");
        }
    }

    /// a development line 3 km case on the RECORDED bytes of block 192 of pkg-lam-r300-e4 at steps 1 and 2 (a test machine
    /// <local path>, step2.bin; A3KM_DIR names the folder), at both list words: bytes
    /// 72560224 and 72560272. The replay holds exactly the recording's bytes with its real addresses at replay
    /// addresses; ONE word of the cell-edge list lies inside a made-up 8-byte live replay block.
    #[test]
    #[ignore = "needs A3KM_DIR: recorded bytes of a 3 km run, not carried by this crate"]
    fn a3km_block_192_recorded_bytes_pass_step_2_at_both_list_words() {
        const S1: &str = "bb9e7c7acbddc4c7eb95f77b82ae2944738e4d19e881c18e5c9513c07764ffb6";
        const S2: &str = "36f7f31d5dada3070a3df03dfca17f215937382851f7a016373588e83e80b292";
        const M1: &str = "0e822abf1cc56e478c148d8298eca598b67667c1c1c77f769d85069557c62a8c";
        const M2: &str = "8ae901a542895eebd5ee9b6fcf01315dde14a4a872e8a1b587178440c5953dfa";
        const ZEROED: [(usize, usize); 1] = [(27_560_448, 25_363_456)];
        const FWD: u64 = 0x75f0_0000_0000;
        const REPLAY: u64 = 0x7bf0_0000_0000;
        const SPAN: u64 = 0x0010_0000_0000;
        const NOTE_FWD: u64 = 0x7400_0000_0000;
        let dir = std::env::var("A3KM_DIR").expect("A3KM_DIR = folder with step1.bin and step2.bin");
        let load = |name: &str| std::fs::read(format!("{dir}/{name}")).expect(name);
        let word = |d: &[u8], i: usize| u64::from_le_bytes(d[8 * i..8 * i + 8].try_into().unwrap());
        let masked_out = |i: usize| 8 * i >= ZEROED[0].0 && 8 * i < ZEROED[0].0 + ZEROED[0].1;
        let is_ptr = |w: u64| (FWD..FWD + SPAN).contains(&w) && w & 0xffff_ffff >= 0x1_0000;
        let as_replay = |rec: &[u8]| -> Vec<u8> {
            let mut out = rec.to_vec();
            for i in 0..rec.len() / 8 {
                let w = word(rec, i);
                if is_ptr(w) { out[8 * i..8 * i + 8].copy_from_slice(&(w - FWD + REPLAY).to_le_bytes()); }
            }
            out
        };
        let (rec1, rec2) = (load("step1.bin"), load("step2.bin"));
        assert_eq!(sha(&rec1), S1); assert_eq!(sha(&rec2), S2);
        let (rep1, rep2) = (as_replay(&rec1), as_replay(&rec2));
        for w_byte in [72_560_224_usize, 72_560_272] {
            let i = w_byte / 8;
            let w = word(&rec1, i);
            assert!(!masked_out(i)); assert_eq!(word(&rec2, i), w, "the list word is the same at step 2");
            let mut ranges = vec![(w, w + 7, NOTE_FWD)];
            for k in 0..16_u64 { ranges.push((REPLAY + (k << 32) + 0x1_0000, REPLAY + (k << 32) + 0xffff_ffff, FWD + (k << 32) + 0x1_0000)); }
            ranges.sort_unstable();
            let (low, high) = (ranges[0].0, ranges[16].1);
            let (view1, mut notes1, _) = forward_address_view_tracked(&rep1, &ranges, low, high, None);
            let view1 = view1.expect("step 1: the view changes the list word").0;
            assert_eq!(word(&view1, i), NOTE_FWD, "step 1: the list word is read as an address by value");
            let got1 = residue_masked_reading(&rep1, Some(&view1), &ranges, low, high, None, &ZEROED, M1, &mut notes1).unwrap();
            assert_eq!(got1.0, M1, "byte {w_byte}: step 1 passes");
            let t = std::time::Instant::now();
            let (view2, mut notes2, _) = forward_address_view_tracked(&rep2, &ranges, low, high, Some(&notes1));
            let view2 = view2.expect("step 2 view").0;
            let got2 = residue_masked_reading(&rep2, Some(&view2), &ranges, low, high, Some(&notes1), &ZEROED, M2, &mut notes2).unwrap();
            println!("ADDR-S a3km byte {w_byte}: list word {w:#x}; step 2 masked {} (recorded {M2}); notes kept {}; note on the list word {:x?}; {:.2} s",
                     got2.0, notes2.len(), notes2.get(&i), t.elapsed().as_secs_f64());
            assert_eq!(got2.0, M2, "byte {w_byte}: step 2 must pass: every byte outside the left-out part equals the recording");
            assert!(!notes2.contains_key(&i), "byte {w_byte}: the list word keeps no note");
            // SOUNDNESS: one planted data change (the list word itself, one cell off) must still mismatch
            let mut planted = rep2.clone();
            planted[w_byte..w_byte + 8].copy_from_slice(&(w + 1).to_le_bytes());
            let (view, mut notes, _) = forward_address_view_tracked(&planted, &ranges, low, high, Some(&notes1));
            let got = residue_masked_reading(&planted, view.as_ref().map(|v| v.0.as_slice()), &ranges, low, high, Some(&notes1), &ZEROED, M2, &mut notes).unwrap();
            assert_ne!(got.0, M2, "byte {w_byte}: a planted change of the list word passed");
        }
    }
}

/// addr-s: the search itself (functions the base does not have): what it reports and where it stops.
#[cfg(test)]
mod addr_s_any_reading_tests {
    use super::*;
    use sha2::{Digest, Sha256};
    fn bytes(words: &[u64]) -> Vec<u8> { words.iter().flat_map(|w| w.to_le_bytes()).collect() }
    fn sha(data: &[u8]) -> String { hex_bytes(&Sha256::digest(data)) }
    const LIVE: (u64, u64, u64) = (0x7336_3600_0000, 0x7336_3a96_4000, 0x7e3c_1000_0000);

    #[test]
    fn the_bound_on_hashes_scales_with_the_block() {
        assert_eq!(any_reading_bound(4096), 4096);
        assert_eq!(any_reading_bound(125_671_424), 546);       // block 347 at 120 km
        assert_eq!(any_reading_bound(2_147_483_648), 32);      // a 2 GiB pool block at 60 km
        assert_eq!(any_reading_bound(1 << 40), 8);
    }

    #[test]
    fn an_old_address_with_one_low_byte_rewritten_has_the_recorded_reading() {
        // p60 step 28, block 790: a 1-byte clear over byte 2 of a pointer the table held since step 23
        let old = (0x72eb_5b12_3400_u64, 0x7b82_6c56_3400_u64);
        let previous = PointerHistory::from([(0, old)]);
        let data = bytes(&[0x72eb_5b00_3400]);
        let read = admissible_readings(&data, &[], 1, 0, Some(&previous));
        assert_eq!(read.len(), 1);
        assert!(read[0].alts.iter().any(|a| a.2 && a.0 == 0x7b82_6c00_3400), "{:?}", read[0].alts);
        // a top byte changed: not an old address with low bytes rewritten
        let data = bytes(&[0x73eb_5b00_3400]);
        let read = admissible_readings(&data, &[], 1, 0, Some(&previous));
        assert!(!read.iter().flat_map(|w| w.alts.iter()).any(|a| a.0 == 0x7b82_6c00_3400), "{read:?}");
    }

    #[test]
    fn readings_are_listed_without_asking_whether_the_word_is_live() {
        // an old address with its low half rewritten: the same two note readings whether or not the value is live
        let old = (0x7336_1234_5600_u64, 0x7e3b_1234_5600_u64);
        let previous = PointerHistory::from([(0, old)]);
        let data = bytes(&[0x7336_3a48_cf2f, 5]);
        let inside = admissible_readings(&data, &[LIVE], LIVE.0, LIVE.1, Some(&previous));
        let outside = admissible_readings(&data, &[], 1, 0, Some(&previous));
        assert_eq!(inside.len(), 1); assert_eq!(outside.len(), 1);
        let values = |w: &AnyWord, noted: bool| w.alts.iter().filter(|a| a.2 == noted).map(|a| a.0).collect::<Vec<u64>>();
        assert_eq!(values(&inside[0], true), vec![0x7e3b_3a48_cf2f, 0x7e36_3a48_cf2f]);
        assert_eq!(values(&outside[0], true), values(&inside[0], true));
        assert_eq!(values(&inside[0], false), vec![0x7e3c_1448_cf2f]);
        assert!(values(&outside[0], false).is_empty());
    }

    #[test]
    fn cost_is_reported_and_a_reading_out_of_reach_is_a_stop() {
        let replay: Vec<u64> = (0..20).map(|k| LIVE.0 + 0x1000 * k).chain([7]).collect();
        let data = bytes(&replay);
        let viewed = forward_address_view(&data, &[LIVE], LIVE.0, LIVE.1).unwrap().0;
        let recorded = |as_data: &[usize]| -> Vec<u8> {
            bytes(&replay.iter().enumerate().map(|(i, w)| if i == 20 || as_data.contains(&i) { *w } else { LIVE.2 + (*w - LIVE.0) }).collect::<Vec<u64>>())
        };
        // two of the twenty are data: found by a pair of changes
        let (read, cost) = any_reading(&data, &viewed, &[LIVE], LIVE.0, LIVE.1, None, &[], &sha(&recorded(&[4, 17])));
        let read = read.expect("two data words among twenty are within reach");
        assert_eq!(read.iter().filter(|w| w.2.is_none()).map(|w| w.0).collect::<Vec<usize>>(), vec![4, 17]);
        assert_eq!(cost.words, 20);
        assert!(cost.hashes > 2 && cost.hashes <= cost.bound, "{cost:?}");
        // three are data: out of reach, so the block stays a mismatch, inside the bound
        let (read, cost) = any_reading(&data, &viewed, &[LIVE], LIVE.0, LIVE.1, None, &[], &sha(&recorded(&[4, 9, 17])));
        assert!(read.is_none());
        assert!(cost.hashes >= 40 && cost.hashes <= cost.bound, "{cost:?}");
        assert_eq!(cost.bound, 4096);
        // a planted data change is never within reach
        let mut planted = replay.clone(); planted[20] = 8;
        let planted = bytes(&planted);
        let (read, _) = any_reading(&planted, &forward_address_view(&planted, &[LIVE], LIVE.0, LIVE.1).unwrap().0, &[LIVE], LIVE.0, LIVE.1, None, &[], &sha(&recorded(&[])));
        assert!(read.is_none());
    }

    #[test]
    fn data_like_pairs_among_many_addresses_are_tried_first() {
        // 3 km block 102 at step 1 in small: 560 words lie in a live block, six of them are pairs of small mesh numbers
        // (0x744300007444 = 29763 and 29764) that the recording holds as they are; none of the six has a note.
        const MESH: (u64, u64, u64) = (0x7443_0000_0000, 0x7443_4000_0000, 0x7e50_2000_0000);
        let ranges = [LIVE, MESH];
        let (low, high) = (LIVE.0, MESH.1);
        let data_at = [31_usize, 43, 44, 331, 343, 344];
        let mut run: Vec<u64> = (0..560_u64).map(|k| LIVE.0 + 0x1200 * k).collect();
        for (n, &i) in data_at.iter().enumerate() { run[i] = [0x7443_0000_7444_u64, 0x7443_0000_7448, 0x7443_0000_7109][n % 3]; }
        run.push(7);
        let moved = |w: u64| if w >= MESH.0 { MESH.2 + (w - MESH.0) } else { LIVE.2 + (w - LIVE.0) };
        let previous: PointerHistory = (0..380_usize).filter(|i| !data_at.contains(i)).map(|i| (i, (run[i], moved(run[i])))).collect();
        let recorded: Vec<u64> = run.iter().enumerate().map(|(i, &w)| if i == 560 || data_at.contains(&i) { w } else { moved(w) }).collect();
        let expected = sha(&bytes(&recorded));
        let data = bytes(&run);
        let viewed = forward_address_view_tracked(&data, &ranges, low, high, Some(&previous)).0.unwrap().0;
        assert_ne!(sha(&viewed), expected);
        let (read, cost) = any_reading(&data, &viewed, &ranges, low, high, Some(&previous), &[], &expected);
        let read = read.expect("six data pairs among 560 address words are within reach of the ordered search");
        assert_eq!(read.iter().filter(|w| w.2.is_none()).map(|w| w.0).collect::<Vec<usize>>(), data_at.to_vec());
        assert_eq!(cost.words, 560);
        assert_eq!(cost.stage, "data_like_threshold");
        assert!(cost.hashes <= 5 + 5 * ANY_ORDER_LEVELS, "{cost:?}");
        // a planted change of one real data word, or of one of the six pairs, is refused: the score orders the search,
        // only the recorded SHA-256 accepts
        for (at, value) in [(560_usize, 8_u64), (43, 0x7443_0000_7449)] {
            let mut planted = run.clone();
            planted[at] = value;
            let planted = bytes(&planted);
            let viewed = forward_address_view_tracked(&planted, &ranges, low, high, Some(&previous)).0.unwrap().0;
            let (read, cost) = any_reading(&planted, &viewed, &ranges, low, high, Some(&previous), &[], &expected);
            assert!(read.is_none(), "planted word {at} was accepted");
            assert!(cost.hashes > 100 && cost.hashes <= cost.bound, "{cost:?}");
        }
    }

    #[test]
    fn a_note_with_a_zero_low_half_is_read_as_a_half_and_a_plain_zero_as_data() {
        // 3 km block 153 at step 45 in small: a live block begins on a 4 GiB line, seven words hold its address and all
        // have the note. The model wrote a zero over the low half of word 39 (it still equals the note) and over all of
        // word 12 (the half rule reads that zero as the low half of the noted address).
        const LINE: (u64, u64, u64) = (0x7bf4_0000_0000, 0x7bf4_1000_0000, 0x75b4_a400_0000);
        let ranges = [LIVE, LINE];
        let (low, high) = (LIVE.0, LINE.1);
        let mut run: Vec<u64> = (0..40_u64).map(|k| LIVE.0 + 0x1200 * k).collect();
        for i in [3_usize, 9, 15, 21, 27, 33, 39] { run[i] = LINE.0; }
        run.push(7);
        let moved = |w: u64| if w >= LINE.0 { LINE.2 + (w - LINE.0) } else { LIVE.2 + (w - LIVE.0) };
        let mut previous: PointerHistory = (0..40_usize).map(|i| (i, (run[i], moved(run[i])))).collect();
        previous.insert(12, (LINE.0, LINE.2));
        run[12] = 0;
        let recorded: Vec<u64> = run.iter().enumerate().map(|(i, &w)| match i { 12 => 0, 39 => LINE.2 & HIGH_HALF, 40 => w, _ => moved(w) }).collect();
        let expected = sha(&bytes(&recorded));
        let data = bytes(&run);
        let viewed = forward_address_view_tracked(&data, &ranges, low, high, Some(&previous)).0.unwrap().0;
        assert_eq!(u64::from_le_bytes(viewed[8 * 12..8 * 13].try_into().unwrap()), 0xa400_0000);
        assert_eq!(u64::from_le_bytes(viewed[8 * 39..8 * 40].try_into().unwrap()), LINE.2);
        let (read, cost) = any_reading(&data, &viewed, &ranges, low, high, Some(&previous), &[], &expected);
        let read = read.expect("one half address and one plain zero under a zero-low-half note are within reach");
        assert_eq!(read.iter().find(|w| w.0 == 12).map(|w| (w.1, w.2)), Some((0, None)));
        assert_eq!(read.iter().find(|w| w.0 == 39).map(|w| (w.1, w.2)), Some((0x75b4_0000_0000, Some((LINE.0, LINE.2)))));
        assert_eq!(read.iter().find(|w| w.0 == 3).map(|w| w.1), Some(LINE.2));
        assert_eq!(cost.stage, "data_like_words");
        assert!(cost.hashes <= 120, "{cost:?}");
        // one real data word changed: refused
        let mut planted = run.clone();
        planted[40] = 8;
        let planted = bytes(&planted);
        let viewed = forward_address_view_tracked(&planted, &ranges, low, high, Some(&previous)).0.unwrap().0;
        let (read, _) = any_reading(&planted, &viewed, &ranges, low, high, Some(&previous), &[], &expected);
        assert!(read.is_none());
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// a development line worker chk-merge (2026-11-11): a development line test module fixproof_steps_test.rs (a test machine
// <local path>, round 5 worker fix-proof), brought here so the merged step check is judged on STORM's kept
// dumps of 3 km allocation 189 at step 11. The drive is STORM's text; added: a wrapper over several dumps and the asserts.
#[cfg(test)]
mod chk_storm_block189_dump_tests {
    use super::{ambiguous_half_words, forward_address_view_tracked, hex_bytes, masked_digest, reference_reading,
                residue_masked_reading, PointerHistory};
    use sha2::{Digest, Sha256};
    const ZEROED: [(usize, usize); 1] = [(352_313_856, 391_473_152)];
    const TABLE: (usize, usize) = (115_585_728, 115_585_797);
    fn sha(d: &[u8]) -> String { hex_bytes(&Sha256::digest(d)) }
    fn masked_out(i: usize) -> bool { let (lo, hi) = (8 * i, 8 * i + 8); ZEROED[0].0 < hi && lo < ZEROED[0].0 + ZEROED[0].1 }
    fn hexv(s: &str) -> u64 { u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap() }
    fn outside(h: &PointerHistory) -> PointerHistory { h.iter().filter(|(i, _)| !masked_out(**i)).map(|(i, p)| (*i, *p)).collect() }
    fn show(h: &PointerHistory) -> String {
        let mut rows: Vec<(usize, u64, u64)> = h.iter().map(|(i, (r, f))| (*i, *r, *f)).collect();
        rows.sort_unstable();
        let n = rows.len();
        let list: Vec<String> = rows.iter().take(40).map(|(i, r, f)| format!("word {i} (byte {}) {r:#x}->{f:#x}", 8 * i)).collect();
        format!("{n} entries [{}{}]", list.join("; "), if n > 40 { "; ..." } else { "" })
    }
    struct Out { pass: bool, masked: String, route: String, history: PointerHistory, view_words: usize }
    // the same order of calls as fixproof_block189::boundary (see the note on top of fixproof_unit_test.rs)
    fn boundary(data: &[u8], ranges: &[(u64, u64, u64)], previous: Option<&PointerHistory>, rec: &[u8], rec_sha: &str, m: &str) -> Out {
        let (low, high) = (ranges.first().map(|r| r.0).unwrap_or(u64::MAX), ranges.iter().map(|r| r.1).max().unwrap_or(0));
        let (view, mut history, _half) = forward_address_view_tracked(data, ranges, low, high, previous);
        let view_words = view.as_ref().map(|v| v.1).unwrap_or(0);
        let translated = view.as_ref().map(|v| v.0.as_slice());
        let masked = residue_masked_reading(data, translated, ranges, low, high, previous, &ZEROED, m, &mut history).unwrap();
        let normal = translated.unwrap_or(data);
        let m_view = masked_digest(&mut normal.to_vec(), &ZEROED).unwrap();
        let m_raw = masked_digest(&mut data.to_vec(), &ZEROED).unwrap();
        let mut route = if m_view == m { "passed on the address view".to_string() }
            else if m_raw == m { "passed on its plain bytes".to_string() }
            else { String::new() };
        if m_view != m && m_raw != m {
            let alternates = ambiguous_half_words(data, ranges, low, high, previous);
            match reference_reading(data, normal, rec, rec_sha, &ZEROED, m, previous, &history, &alternates).unwrap() {
                Some(r) => return Out { pass: r.masked == m, masked: r.masked,
                    route: format!("reference_reading gave a reading ({} words read as addresses, {} half words)", r.moved, r.halves),
                    history: r.history, view_words },
                None => { route = "reference_reading returned None (no reading)".to_string(); }
            }
        }
        Out { pass: masked.0 == m, masked: masked.0, route, history, view_words }
    }

    /// a development line chk-merge: a development line drive (above, unchanged but for this wrapper) run over every kept dump named in
    /// FP_DUMPS (comma list of folders holding 189.bin, 189.history.txt, ranges.txt), and ASSERTED: steps 1 to 10 pass
    /// and the stopped run's dumped block of step 11 gives the recorded masked hash. THE BREAKAGE THIS PREVENTS: the
    /// 3 km storm replay stopping at step 11 on block 189 (4 stops in 10 strict runs before a development line 04b8a98f).
    /// Ignored in the plain suite: it reads 12 files of 1.2 GB. Run: FP_DUMPS=... FP_DATA=... cargo test --release
    /// ... chk_storm_block189 -- --ignored --nocapture --test-threads 1
    #[test]
    #[ignore = "needs FP_DUMPS and FP_DATA: a development line kept bytes of 3 km block 189 (15 GB)"]
    fn storm_block_189_kept_dumps_pass_step_11() {
        let dumps = std::env::var("FP_DUMPS").expect("FP_DUMPS");
        let mut bad = Vec::new();
        for dump_dir in dumps.split(',').filter(|d| !d.trim().is_empty()) {
            let (steps, pass, equal) = drive(dump_dir.trim());
            if !(steps && pass && equal) { bad.push(dump_dir.to_string()); }
        }
        assert!(bad.is_empty(), "dumps that do not pass: {bad:?}");
    }
    fn drive(dump_dir: &str) -> (bool, bool, bool) {
        let data_dir = std::env::var("FP_DATA").expect("FP_DATA");
        let name = dump_dir.to_string();
        let code = std::env::var("FP_CODE").unwrap_or_default();
        let ranges: Vec<(u64, u64, u64)> = std::fs::read_to_string(format!("{dump_dir}/ranges.txt")).unwrap().lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| { let p: Vec<&str> = l.split_whitespace().collect(); (hexv(p[0]), hexv(p[1]), hexv(p[2])) }).collect();
        let dumped: PointerHistory = std::fs::read_to_string(format!("{dump_dir}/189.history.txt")).unwrap().split(',')
            .filter(|l| !l.trim().is_empty())
            .map(|l| { let p: Vec<&str> = l.split_whitespace().collect(); (p[0].parse::<usize>().unwrap(), (hexv(p[1]), hexv(p[2]))) }).collect();
        let table: PointerHistory = dumped.iter().filter(|(i, _)| **i >= TABLE.0 && **i <= TABLE.1).map(|(i, p)| (*i, *p)).collect();
        let carried: PointerHistory = dumped.iter().filter(|(i, _)| !(**i >= TABLE.0 && **i <= TABLE.1)).map(|(i, p)| (*i, *p)).collect();
        let two = |path: String| -> Vec<(usize, String)> { std::fs::read_to_string(path).unwrap().lines().filter(|l| !l.trim().is_empty())
            .map(|l| { let p: Vec<&str> = l.split_whitespace().collect(); (p[0].parse::<usize>().unwrap(), p[1].to_string()) }).collect() };
        let shas = two(format!("{data_dir}/steps-189.txt"));
        let maskeds = two(format!("{data_dir}/masked-189-fc-both-1.txt"));
        let get = |v: &Vec<(usize, String)>, k: usize| v.iter().find(|x| x.0 == k).unwrap().1.clone();
        println!("FPS inputs: run {name}, check code {code}; {} live allocations (the run's step-11 list, used for every step); dumped history: {} at the table, elsewhere (\"carried\"): {}",
            ranges.len(), table.len(), show(&carried));
        let mut hist: Option<PointerHistory> = None;
        let mut all_pass = true;
        let mut routes: Vec<String> = Vec::new();
        for k in 1..=10_usize {
            let rec = std::fs::read(format!("{data_dir}/rec{k:02}.bin")).unwrap();
            let (s, m) = (get(&shas, k), get(&maskeds, k));
            assert_eq!(sha(&rec), s, "recorded block of step {k}");
            let o = boundary(&rec, &ranges, hist.as_ref(), &rec, &s, &m);
            all_pass &= o.pass;
            let out = outside(&o.history);
            println!("FPS step {k:2} DRIVE (recorded block {}.. as the run's block): {} | masked hash {} the recorded {} | {} | words the address view took: {} | history kept, outside the left-out part: {} (inside it: {})",
                &s[..12], if o.pass { "PASSES" } else { "STOPS" }, if o.masked == m { "equal to" } else { "NOT equal to" }, &m[..12], o.route, o.view_words, show(&out), o.history.len() - out.len());
            routes.push(o.route.clone());
            hist = Some(o.history);
        }
        let h10 = hist.unwrap();
        let h10_out = outside(&h10);
        let carried_out = outside(&carried);
        let entry_equal = h10_out == carried_out;
        println!("FPS after step 10: this code holds, outside the left-out part: {}", show(&h10_out));
        println!("FPS the stopped run's dumped history without its table words: {}", show(&carried_out));
        println!("FPS the two are {}", if entry_equal { "EQUAL" } else { "NOT equal" });
        let rec11 = std::fs::read(format!("{data_dir}/rec11.bin")).unwrap();
        let (s11, m11) = (get(&shas, 11), get(&maskeds, 11));
        let dump = std::fs::read(format!("{dump_dir}/189.bin")).unwrap();
        let mut driven = h10.clone();
        for (i, p) in &table { driven.insert(*i, *p); }
        let same = outside(&driven) == outside(&dumped);
        let c = boundary(&dump, &ranges, Some(&driven), &rec11, &s11, &m11);
        println!("FPS step 11 REAL BYTES (the run's dumped block; history = what this code's steps 1 to 10 left plus the dumped table notes; outside the left-out part that history is {} the run's dumped history): {} | masked hash {} ({} the recorded {}) | {} | words the address view took: {}",
            if same { "EQUAL to" } else { "not equal to" }, if c.pass { "PASSES" } else { "STOPS (state hash mismatch)" }, c.masked, if c.masked == m11 { "equal to" } else { "NOT equal to" }, m11, c.route, c.view_words);
        let plain = routes.iter().filter(|r| r.as_str() == "passed on its plain bytes").count();
        println!("FPS SUMMARY run={name} code={code} steps_1_to_10_all_pass={all_pass} steps_passed_on_plain_bytes={plain}_of_10 history_after_step10_outside_leftout={} equals_dumped_carried={entry_equal} driven_history_equals_dumped={same} step11_pass={} step11_masked_equals_recorded={}",
            h10_out.len(), c.pass, c.masked == m11);
        (all_pass, c.pass, c.masked == m11)
    }
}

/// a12 first-forward: the raw sha256 of a large block taken beside the join (digest_raw_ahead). THE BREAKAGE THESE
/// PREVENT: a boundary whose large block was hashed ahead passing where the chain without it refuses, or reading with
/// another history. Host bytes only: no Execution, no libcuda, no card.
#[cfg(test)]
mod raw_ahead_tests {
    use super::*;

    const BIG: u64 = 9;
    const SMALL: u64 = 4;
    /// One boundary: a block of 64 MiB + 4 KiB (raw sha256 taken ahead) and a small one, in one arena.
    fn boundary(seed: u8) -> (Vec<u8>, Vec<(u64, usize, usize)>) {
        let n = DIGEST_OVERLAP_BYTES + 4096;
        let mut b: Vec<u8> = (0..n + 4096).map(|i| ((i as u32).wrapping_mul(2654435761) >> 13) as u8 ^ seed).collect();
        b[n + 7] ^= seed;
        (b, vec![(BIG, 0, n), (SMALL, n, 4096)])
    }
    fn recorded(step: usize, bytes: &[u8], spans: &[(u64, usize, usize)]) -> References {
        spans.iter().map(|&(id, off, n)| ((step, id), (n, hex_bytes(&Sha256::digest(&bytes[off..off + n]))))).collect()
    }
    /// Boundaries 5 and 6 as finish_step_host_hash chains them (6 joins 5 in its own thread); `bad6` flips one bit of
    /// boundary 6's large block after its line was recorded; `gate` holds boundary 5's job until boundary 6's has
    /// started, so 6's ahead hash runs while 5 is still out. Returns the chain's done list and boundary 6's history.
    fn chain(raw_ahead: usize, bad6: bool) -> (Vec<(usize, Result<Vec<u64>, String>, Option<serde_json::Value>)>, HashMap<u64, PointerHistory>) {
        let failed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (b5, s5) = boundary(5);
        let r5 = recorded(5, &b5, &s5);
        let (go, gate) = std::sync::mpsc::channel::<()>();
        let f5 = failed.clone();
        let h5 = std::thread::spawn(move || {
            let _ = gate.recv();
            digest_job_run(5, None, None, Vec::new(), HashMap::new(), &f5, &b5, &s5, &[], u64::MAX, 0, HashMap::new(), Some(r5), HashSet::new(), 2, true, raw_ahead)
        });
        let j5 = DigestJob { step: 5, handle: h5, cell: std::sync::Arc::new(std::sync::OnceLock::new()), workers: 2, dprof: true, t_dl: 0.0 };
        let (mut b6, s6) = boundary(6);
        let r6 = recorded(6, &b6, &s6);
        if bad6 { b6[12_345] ^= 1; }
        let f6 = failed.clone();
        let h6 = std::thread::spawn(move || {
            digest_job_run(6, Some(j5), None, Vec::new(), HashMap::new(), &f6, &b6, &s6, &[], u64::MAX, 0, HashMap::new(), Some(r6), HashSet::new(), 2, true, raw_ahead)
        });
        std::thread::sleep(std::time::Duration::from_millis(50));
        go.send(()).unwrap();
        let j6 = DigestJob { step: 6, handle: h6, cell: std::sync::Arc::new(std::sync::OnceLock::new()), workers: 2, dprof: true, t_dl: 0.0 };
        let (done, history) = digest_finish(j6);
        (done.into_iter().map(|d| (d.step, d.mismatched, d.prof)).collect(), history)
    }
    fn ahead_n(p: &Option<serde_json::Value>) -> u64 { p.as_ref().and_then(|v| v["raw_hashed_ahead"].as_u64()).unwrap() }

    #[test]
    fn the_large_block_hashed_ahead_gives_the_same_readings_and_the_same_refusal() {
        for bad in [false, true] {
            let (old, h_old) = chain(0, bad);
            let (new, h_new) = chain(4, bad);
            assert_eq!(old.len(), 2);
            assert_eq!(new.len(), 2);
            for (o, n) in old.iter().zip(&new) {
                assert_eq!(o.0, n.0);
                assert_eq!(o.1, n.1, "step {}: the same mismatch list with and without the ahead hash", o.0);
            }
            assert_eq!(h_old, h_new, "the same pointer history");
            // boundary 5 has no join to wait for (nothing ahead); boundary 6's large block, and only it, was hashed ahead
            assert_eq!((ahead_n(&old[0].2), ahead_n(&old[1].2)), (0, 0));
            assert_eq!((ahead_n(&new[0].2), ahead_n(&new[1].2)), (0, 1));
            assert_eq!(new[0].1, Ok(vec![]));
            assert_eq!(new[1].1, if bad { Ok(vec![BIG]) } else { Ok(vec![]) }, "a flipped bit in the block hashed ahead still refuses");
        }
    }
    #[test]
    fn only_checked_large_once_listed_non_residue_blocks_are_taken_ahead() {
        let n = DIGEST_OVERLAP_BYTES;
        let spans = vec![(1, 0, n - 1), (2, n, n), (3, 2 * n, 2 * n), (4, 4 * n, n), (4, 5 * n, n), (5, 6 * n, n)];
        let residue: HashMap<(usize, u64), (Vec<(usize, usize)>, String)> = [((7, 5), (vec![(0, 4)], String::new()))].into_iter().collect();
        let none = HashSet::new();
        // largest first; 1 is under the bar, 4 is listed twice, 5 is a large residue block (read twice in digest_blocks)
        assert_eq!(raw_ahead_spans(7, &spans, true, &residue, &none), vec![(3, 2 * n, 2 * n), (2, n, n)]);
        // an address-bearing residue block takes the ordinary large-block path, so it is taken ahead too
        let ab: HashSet<u64> = [5].into_iter().collect();
        assert_eq!(raw_ahead_spans(7, &spans, true, &residue, &ab), vec![(3, 2 * n, 2 * n), (2, n, n), (5, 6 * n, n)]);
        // an unchecked boundary hashes nothing
        assert!(raw_ahead_spans(7, &spans, false, &residue, &none).is_empty());
    }
}

/// speed (merge, round 4): the test owed for 6c51e35a (the boundary hash of a recomputed step beside the card). THE
/// BREAKAGE THESE PREVENT: a recomputed boundary that differs from the recording, hashed beside the card while a
/// restore no longer joins the chain, being counted because nothing joined it before the sweep's rows were returned.
/// They run the chain's own code (digest_job_run, digest_finish, digest_post, digest_flush, restore_history_lazy,
/// restore_join, digest_lazy_end, window.rs sweep_joined) on host bytes. No card is used: the Execution has no context;
/// libcuda.so.1 must be loadable because the struct holds its symbols.
#[cfg(test)]
mod digest_join_tests {
    use super::*;
    use std::sync::atomic::Ordering::Relaxed;

    const ID: u64 = 7;
    const REFUSAL: &str = "state hash mismatch at step 5, 1 allocations, first 7";

    fn ex_no_card() -> Execution {
        let cuda = unsafe { Cuda::load() }.expect("digest_join_tests need libcuda.so.1 to be loadable (no card is used)");
        Execution::from_parts(cuda, ptr::null_mut(), None)
    }
    fn block(seed: u8) -> Vec<u8> { (0..4096u32).map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed)).collect() }
    /// The recording's line for boundary `step`: the SHA-256 of the right bytes (no address words in them).
    fn recorded(step: usize, good: &[u8]) -> References {
        [((step, ID), (good.len(), hex_bytes(&Sha256::digest(good))))].into_iter().collect()
    }
    /// One boundary digest started as finish_step_host_hash starts it for a step that keeps no host copy: the job of
    /// the boundary before it goes into this one's thread (digest_chain), a lazily restored history is handed over
    /// (history_from). `gate`: the thread waits for a message first (a digest that is still running).
    fn start(ex: &mut Execution, step: usize, bytes: Vec<u8>, refs: References, gate: Option<std::sync::mpsc::Receiver<()>>) {
        let prev = ex.digest_job.take();
        let from = ex.history_from.take();
        let merged = if prev.is_some() { HashMap::new() } else { match &from { Some(h) => h.get().clone(), None => ex.pointer_history.clone() } };
        let failed = ex.digest_failed.clone();
        let handle = std::thread::spawn(move || {
            if let Some(g) = gate { let _ = g.recv(); }
            let spans = [(ID, 0, bytes.len())];
            digest_job_run(step, prev, from, Vec::new(), merged, &failed, &bytes, &spans, &[], u64::MAX, 0, HashMap::new(), Some(refs), HashSet::new(), 1, false, 0)
        });
        ex.digest_job = Some(DigestJob { step, handle, cell: std::sync::Arc::new(std::sync::OnceLock::new()), workers: 1, dprof: false, t_dl: 0.0 });
    }
    fn planted() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let good5 = block(5);
        let mut bad5 = good5.clone();
        bad5[1234] ^= 1; // one bit of the recomputed boundary of step 5 differs from the recording
        (good5, bad5, block(6))
    }

    #[test]
    fn a_recomputed_boundary_that_differs_reaches_the_sweep_end_join_and_the_sweep_is_refused() {
        let (good5, bad5, good6) = planted();
        let ready = HistSnap::Ready(HashMap::new());
        let mut ex = ex_no_card();

        // (1) the whole path of 6c51e35a. The sweep begins (window.rs sweep_revolve sets digest_lazy); the hash of
        // recomputed step 5 runs beside the card and is still running when the sweep restores a checkpoint.
        ex.digest_lazy = true;
        let (go, gate) = std::sync::mpsc::channel();
        start(&mut ex, 5, bad5.clone(), recorded(5, &good5), Some(gate));
        assert_eq!(ex.restore_join(&ready), Ok(true), "the restore must not join a chain that is still running");
        assert!(ex.digest_job.is_some() && ex.history_from.is_some());
        go.send(()).unwrap();
        // the next recomputed step chains step 5's digest into its own thread and reads with the restored history
        start(&mut ex, 6, good6.clone(), recorded(6, &good6), None);
        assert!(ex.history_from.is_none());
        // the sweep ends: every digest is joined, the boundary that differs refuses, and the sweep's rows do not count
        let joined = ex.digest_lazy_end();
        assert_eq!(joined, Err(REFUSAL.to_string()));
        assert!(!ex.digest_lazy && ex.digest_job.is_none() && !ex.digest_failed.load(Relaxed));
        assert_eq!(window::sweep_joined(Ok(vec![1, 2, 3]), joined), Err(REFUSAL.to_string()));
        // the sweep's own refusal comes first, as before
        assert_eq!(window::sweep_joined::<Vec<i32>>(Err("step 9: sweep".into()), Err(REFUSAL.to_string())), Err("step 9: sweep".to_string()));

        // (2) the early refusal: once step 6's thread has finished step 5 it raises the flag, and the next restore
        // joins the chain and refuses at once instead of going on to the sweep's end.
        ex.digest_lazy = true;
        start(&mut ex, 5, bad5.clone(), recorded(5, &good5), None);
        start(&mut ex, 6, good6.clone(), recorded(6, &good6), None);
        let t = std::time::Instant::now();
        while !ex.digest_failed.load(Relaxed) {
            assert!(t.elapsed().as_secs() < 60, "the chain never flagged the boundary that differs");
            std::thread::yield_now();
        }
        assert_eq!(ex.restore_join(&ready), Err(REFUSAL.to_string()));
        assert!(ex.digest_job.is_none() && ex.history_from.is_none());
        assert_eq!(ex.digest_lazy_end(), Ok(()));

        // (3) control: the same two boundaries without the planted bit pass the same joins, and the rows count
        ex.digest_lazy = true;
        let (go, gate) = std::sync::mpsc::channel();
        start(&mut ex, 5, good5.clone(), recorded(5, &good5), Some(gate));
        assert_eq!(ex.restore_join(&ready), Ok(true));
        go.send(()).unwrap();
        start(&mut ex, 6, good6.clone(), recorded(6, &good6), None);
        let joined = ex.digest_lazy_end();
        assert_eq!(joined, Ok(()));
        assert!(!ex.digest_failed.load(Relaxed));
        assert_eq!(window::sweep_joined(Ok(vec![1, 2, 3]), joined), Ok(vec![1, 2, 3]));
    }

    #[test]
    fn with_the_beside_path_off_a_boundary_that_differs_is_refused_in_its_own_step_and_a_restore_joins_the_chain() {
        let (good5, bad5, _) = planted();
        let refs = recorded(5, &good5);
        let ready = HistSnap::Ready(HashMap::new());
        let mut ex = ex_no_card();
        // AD_RECOMPUTE_HASH_BESIDE=0 (or no checkpointed sweep): sweep_revolve leaves digest_lazy false
        assert!(!ex.digest_lazy);

        // the hash on the step path (finish_step_host_hash, not deferred): readings, comparison and refusal in the step
        let (none, ab) = (HashMap::new(), HashSet::new());
        let mut out = digest_blocks(5, &[(ID, &bad5[..])], &[], u64::MAX, 0, &HashMap::new(), &none, &ab, Some(&refs), 1, false);
        let mismatched = digest_post(5, &[(ID, bad5.len())], &mut out, &none, &ab, Some(&refs));
        assert_eq!(mismatched, Ok(vec![ID]));
        assert_eq!(ex.digest_tail(5, mismatched.unwrap(), 1, None), Err(REFUSAL.to_string()));
        let mut out = digest_blocks(5, &[(ID, &good5[..])], &[], u64::MAX, 0, &HashMap::new(), &none, &ab, Some(&refs), 1, false);
        let mismatched = digest_post(5, &[(ID, good5.len())], &mut out, &none, &ab, Some(&refs));
        assert_eq!(mismatched, Ok(vec![]));
        assert_eq!(ex.digest_tail(5, mismatched.unwrap(), 1, None), Ok(()));

        // a deferred digest is pending when a restore comes: the restore is not lazy, it joins the chain (it waits for
        // it) and refuses there, as before 6c51e35a
        start(&mut ex, 5, bad5.clone(), refs.clone(), None);
        assert_eq!(ex.restore_join(&ready), Err(REFUSAL.to_string()));
        assert!(ex.digest_job.is_none() && ex.history_from.is_none() && !ex.digest_failed.load(Relaxed));
        // the sweep's end with the path off: nothing is pending and nothing is refused
        assert_eq!(ex.digest_lazy_end(), Ok(()));

        // control: the right bytes pass the same restore (joined, not lazy)
        start(&mut ex, 5, good5.clone(), refs, None);
        assert_eq!(ex.restore_join(&ready), Ok(false));
        assert!(ex.digest_job.is_none() && ex.history_from.is_none());
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// a development line worker cure-3km (round 6, 2026-11-11): a development line kept dump of 3 km block 136 at step 23 (a test machine
// <local path>), put through the readings of the step check in the order the check tries
// them for a block with a whole-block fingerprint and no recorded block at hand (finish_step_host_hash, the lines from
// forward_address_view_tracked to any_reading). A test only: no line of the check is changed. The lines between
// CURE-ANY-BEGIN and CURE-ANY-END are cut out when this module is appended to a source without the lane's fix (fold 4,
// e7888b1d): that copy is the control (r6/cure/kit/mk-control.sh).
#[cfg(test)]
mod cure_storm_block136_dump_tests {
    use super::*;
    use sha2::{Digest, Sha256};
    fn hexv(s: &str) -> u64 { u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap() }

    /// CURE_DUMPS: comma list of folders holding 136.bin, 136.history.txt, ranges.txt. CURE_SHA: the recorded SHA-256 of
    /// the block at that step. CURE_WANT: comma list, one of pass|miss|any per folder (default any): the test fails when
    /// a folder's outcome is not the one named. Ignored in the plain suite: each block is 117 MB and is not in the tree.
    #[test]
    #[ignore = "needs CURE_DUMPS and CURE_SHA: a development line kept bytes of 3 km block 136 at step 23"]
    fn storm_block_136_step_23_kept_dump() {
        let dumps = std::env::var("CURE_DUMPS").expect("CURE_DUMPS");
        let expected = std::env::var("CURE_SHA").expect("CURE_SHA");
        let code = std::env::var("CURE_CODE").unwrap_or_default();
        let want = std::env::var("CURE_WANT").unwrap_or_default();
        let wants: Vec<&str> = want.split(',').collect();
        let mut bad = Vec::new();
        for (k, dump_dir) in dumps.split(',').filter(|d| !d.trim().is_empty()).enumerate() {
            let gave = drive(dump_dir.trim(), &expected, &code);
            let w = wants.get(k).copied().unwrap_or("any");
            if (w == "pass" && !gave) || (w == "miss" && gave) { bad.push(format!("{dump_dir}: wanted {w}, gave_recorded_sha256={gave}")); }
        }
        assert!(bad.is_empty(), "outcomes not as named in CURE_WANT: {bad:?}");
    }

    fn drive(dump_dir: &str, expected: &str, code: &str) -> bool {
        // CURE_BLOCK: the block number in the file names of the dump (default 136)
        let block = std::env::var("CURE_BLOCK").unwrap_or_else(|_| "136".to_string());
        let ranges: Vec<(u64, u64, u64)> = std::fs::read_to_string(format!("{dump_dir}/ranges.txt")).unwrap().lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| { let p: Vec<&str> = l.split_whitespace().collect(); (hexv(p[0]), hexv(p[1]), hexv(p[2])) }).collect();
        let before: PointerHistory = std::fs::read_to_string(format!("{dump_dir}/{block}.history.txt")).unwrap().split(',')
            .filter(|l| !l.trim().is_empty())
            .map(|l| { let p: Vec<&str> = l.split_whitespace().collect(); (p[0].parse::<usize>().unwrap(), (hexv(p[1]), hexv(p[2]))) }).collect();
        let data = std::fs::read(format!("{dump_dir}/{block}.bin")).unwrap();
        let data: &[u8] = &data;
        let (low, high) = (ranges.first().map(|r| r.0).unwrap_or(u64::MAX), ranges.iter().map(|r| r.1).max().unwrap_or(0));
        let previous = Some(&before);
        println!("CURE136 inputs: dump {dump_dir}, check code {code}; block {block} of {} bytes, sha256 of its plain bytes {}; {} live blocks ({low:#x} to {high:#x}); notes of the step before: {}; recorded sha256 {expected}",
            data.len(), hex_bytes(&Sha256::digest(data)), ranges.len(), before.len());
        // 1. the address view and its fingerprint (or the plain bytes), as boundary_digest_pre2 decides
        let (view, mut history, half) = forward_address_view_tracked(data, &ranges, low, high, previous);
        let view_words = view.as_ref().map(|v| v.1).unwrap_or(0);
        let (mut value, raw) = boundary_digest_pre2(data, view.as_ref().map(|v| v.0.as_slice()), Some(expected), None, None);
        let mut reading = String::new();
        if value == expected { reading = if raw { "plain bytes".to_string() } else { "the address view (first reading)".to_string() }; }
        println!("CURE136 reading 1 (address view): {view_words} words read as addresses, {half} half words; sha256 {value}; plain bytes taken: {raw}; {}",
            if value == expected { "EQUAL to the recorded" } else { "not the recorded" });
        // 2. half-word subsets (the text of the check)
        if !raw && value != expected {
            let alternates = ambiguous_half_words(data, &ranges, low, high, previous);
            let base = view.as_ref().map(|v| v.0.clone()).unwrap_or_else(|| data.to_vec());
            let n = alternates.len().min(12);
            let mut masks: Vec<u32> = (1..(1u32 << n)).collect();
            masks.sort_by_key(|m| (std::cmp::Reverse(*m == (1u32 << n) - 1), m.count_ones(), *m));
            let mut tried = 0_usize;
            for mask in masks {
                let mut alternate = base.clone();
                let chosen: Vec<&(usize, u64)> = alternates.iter().take(n).enumerate()
                    .filter(|(k, _)| mask >> k & 1 == 1).map(|(_, a)| a).collect();
                for (index, word) in &chosen { alternate[8 * index..8 * index + 8].copy_from_slice(&word.to_le_bytes()); }
                let hashed = hex_bytes(&Sha256::digest(&alternate));
                tried += 1;
                if expected == hashed.as_str() {
                    value = hashed;
                    for (index, _) in &chosen { if let Some(entry) = before.get(index) { history.insert(*index, *entry); } }
                    reading = format!("half-word subsets ({} of {} candidate half words)", chosen.len(), alternates.len());
                    break;
                }
            }
            println!("CURE136 reading 2 (half-word subsets): {} candidate half words, {tried} hashes; {}",
                alternates.len(), if value == expected { "EQUAL to the recorded" } else { "not the recorded" });
        }
        // 3. mixed_reading, then lookalike_half_reading
        if !raw && value != expected {
            if let Some((viewed, _)) = view.as_ref() {
                let t0 = std::time::Instant::now();
                let mixed = mixed_reading(data, viewed, expected, previous, Some(&history));
                let mixed_s = t0.elapsed().as_secs_f64();
                println!("CURE136 reading 3a (mixed_reading): {} in {mixed_s:.1} s",
                    match &mixed { Some((_, back)) => format!("EQUAL to the recorded, {} words put back to plain", back.len()), None => "no reading".to_string() });
                if let Some((hashed, back)) = mixed {
                    value = hashed; reading = format!("mixed_reading ({} words put back to plain)", back.len());
                } else {
                    let t1 = std::time::Instant::now();
                    let look = lookalike_half_reading(data, viewed, expected, previous, Some(&history));
                    println!("CURE136 reading 3b (lookalike_half_reading): {} in {:.1} s",
                        match &look { Some((back, at)) => format!("EQUAL to the recorded, {} words put back to plain, {} half words", back.len(), at.len()), None => "no reading".to_string() },
                        t1.elapsed().as_secs_f64());
                    if let Some((back, at)) = look {
                        value = expected.to_string();
                        reading = format!("lookalike_half_reading ({} words put back to plain, {} half words)", back.len(), at.len());
                    }
                }
            }
        }
        // CURE-ANY-BEGIN
        // 4. the lane's one last reading
        if !raw && value != expected {
            let default = view.as_ref().map(|v| v.0.as_slice()).unwrap_or(data);
            let (read, cost) = any_reading(data, default, &ranges, low, high, previous, &[], expected);
            println!("CURE136 reading 4 (any_reading): words_with_more_than_one_reading {} hashes_tried {} hash_bound {} seconds {:.3} matched_recorded_sha256 {} matched_by [{}]",
                cost.words, cost.hashes, cost.bound, cost.seconds, read.is_some(), cost.stage);
            if let Some(words) = read {
                value = expected.to_string();
                reading = format!("address_any_reading ({} words with more than one reading, {} hashes of a bound of {}, {} words kept as addresses)",
                    cost.words, cost.hashes, cost.bound, words.iter().filter(|w| w.2.is_some()).count());
            }
        }
        // CURE-ANY-END
        let gave = value == expected;
        println!("CURE136 SUMMARY dump={dump_dir} code={code} gives_recorded_sha256={gave} reading=[{}] final_sha256={value}",
            if gave { reading.as_str() } else { "none: every reading this code has misses; the check stops here (state hash mismatch)" });
        gave
    }
}

#[cfg(test)]
mod checked_read_tests {
    use super::*;
    fn le32(v: &[i32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
    fn le64(v: &[i64]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
    /// A moved start whose long-wave layer histogram differs from the recorded one is followed (regrouped), not refused;
    /// the same read on a forward that is compared with the recording is refused and says which read it is.
    #[test]
    fn moved_start_with_a_differing_lw_histogram_is_regrouped() {
        let recorded = le32(&[0, 0, 2890, 0]);
        let live = le32(&[0, 3, 2887, 0]);
        assert_eq!(judge_checked_read(&live, Some(&recorded[..]), None, true, true, 0),
                   CheckedRead::FollowLive { hist: vec![0, 3, 2887, 0], recorded_equal: false });
        // the recorded histogram on a moved start: nothing to regroup, the read passes
        assert_eq!(judge_checked_read(&recorded, Some(&recorded[..]), None, true, true, 0), CheckedRead::Same);
        // a forward compared with the recording does not follow
        let CheckedRead::Differs(d) = judge_checked_read(&live, Some(&recorded[..]), None, true, false, 0) else { panic!("followed on a checked forward") };
        let e = checked_read_refusal(7, &d, 165, 0, "checked_value", true, Some("lw_layer_histogram_from_launches"), &live);
        assert!(e.contains(READ_DIFFERS) && e.contains("this read is the long-wave layer histogram") && e.contains("at step 7"), "{e}");
        assert!(!refusal_leaves_program_loaded(&e));
    }
    /// The forecast's own fatal-condition words are never followed, on a moved start either; the refusal says the forecast
    /// stops, names the words, tells the reader what to do, and is the one refusal a kept-loaded program survives.
    #[test]
    fn forecast_check_words_refuse_the_start_in_plain_words() {
        let want = le64(&[0; 8]);
        let got = le64(&[0, 0, 0, 0, 0, 0, 1, 1]);
        let CheckedRead::Differs(d) = judge_checked_read(&got, Some(&want[..]), None, false, true, 0) else { panic!("not refused") };
        assert!(d.starts_with("2 of 16 words differ"), "{d}");
        let e = checked_read_refusal(2, &d, 132, 13312, "checked_value", false, Some(DEFERRED_CHECK_PROOF), &got);
        assert!(e.starts_with("step 2: the forecast itself stops at this start: 2 of the 8 check words"), "{e}");
        assert!(e.contains("(word 6: 1, word 7: 1)") && e.contains("Ask for a start nearer one that runs"), "{e}");
        // what to do comes before what the words are: a client that keeps the first 400 characters of an answer keeps it,
        // with as many words named as the sentence ever names (six)
        let six = checked_read_refusal(1234, &d, 132, 13312, "checked_value", false, Some(DEFERRED_CHECK_PROOF), &le64(&[-104, 7, 7, 7, 7, 7, 7, 7]));
        let at = six.find("same place").expect("the instruction");
        assert!(at + "same place".len() <= 400 && six.contains("(word 0: -104, word 1: 7, word 2: 7, word 3: 7, word 4: 7, word 5: 7)"), "{at}: {six}");
        assert!(!e.contains("record the package") && !e.contains(READ_DIFFERS) && !e.contains("LW layer-group"), "{e}");
        assert!(refusal_leaves_program_loaded(&e));
        // any other checked read that moved keeps the host-choice refusal, and says it is not the long-wave histogram
        let e2 = checked_read_refusal(2, &d, 132, 13312, "checked_value", false, Some("recorded_two_recordings_equal"), &got);
        assert!(e2.contains(READ_DIFFERS) && e2.contains("this read is not the long-wave layer histogram"), "{e2}");
        assert!(!refusal_leaves_program_loaded(&e2));
        // zero-word reads: the set of zero words moved
        assert!(matches!(judge_checked_read(&le32(&[0, 5, 0]), None, Some(&[0usize][..]), false, true, 0), CheckedRead::Differs(_)));
        assert_eq!(judge_checked_read(&le32(&[0, 5, 7]), None, Some(&[0usize][..]), false, true, 0), CheckedRead::Same);
    }
    /// The plan's proof word reaches the runner (older plans without it still read).
    #[test]
    fn plan_read_back_carries_its_proof() {
        let op = |proof: &str| -> Operation { serde_json::from_str(&format!(r#"{{"kind":"device_to_host","stream":0,"source":{{"allocation":132,"offset":13312}},
            "bytes":64,"asynchronous":true,"host_consumer":"checked_value","expected_hex":"00"{proof}}}"#)).unwrap() };
        let Operation::DeviceToHost { proof, .. } = op(r#","proof":"raise_only_deferred_check_words","site":"x.py:1""#) else { panic!() };
        assert_eq!(proof.as_deref(), Some(DEFERRED_CHECK_PROOF));
        let Operation::DeviceToHost { proof, .. } = op("") else { panic!() };
        assert_eq!(proof, None);
    }
}
