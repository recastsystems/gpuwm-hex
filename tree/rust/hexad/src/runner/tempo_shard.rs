// SPDX-License-Identifier: Apache-2.0
//! TEMPO TL/AD column sharding. The dycore stays in the caller's context.
//! Each helper thread owns its CUDA context, modules, arena and failure ledger.
//! Trajectory slices, derivative vectors and immutable tables cross cards.
//! Tape stays local.
//! All helpers are joined before the caller can reuse a snapshot or result.
use crate::{column_shard::{self, Range}, cuda::{checked, Cuda, PeerCopy3D, PeerCopyEndpoint}, dev::{self, Arena, Ledger}, tempo_adw::{Snap, TempoAdw}};
use serde_json::json;
use std::{ffi::c_void, ptr, sync::mpsc, thread, time::Instant};

struct Job { snap: Snap, range: Range, input: u64, output: u64, adjoint: bool, peer_blob: bool }
struct Done { differing: u64, compared: usize, tiles: usize, parts: [f64; 4], receipt: serde_json::Value }
struct Worker { tx: mpsc::Sender<Job>, rx: mpsc::Receiver<Result<Done, String>>, join: thread::JoinHandle<()> }

pub struct Pool { ordinals: Vec<i32>, workers: Vec<Worker>, primary: usize, local_ledger: Ledger, peer_blob: bool }

/// Copy operations are queued into one stream. Events include the copy-engine
/// work and launch gaps; host wall includes queueing and the completion wait.
struct Transfers<'a> {
    cuda: &'a Cuda, here: *mut c_void, main: *mut c_void,
    stream: *mut c_void, begin: *mut c_void, end: *mut c_void,
    bytes: usize, calls: usize,
}
impl<'a> Transfers<'a> {
    unsafe fn new(cuda: &'a Cuda, here: *mut c_void, main: *mut c_void) -> Result<Self, String> {
        let mut t = Self { cuda, here, main, stream: ptr::null_mut(), begin: ptr::null_mut(), end: ptr::null_mut(), bytes: 0, calls: 0 };
        // Nonblocking stream. Explicit joins below connect it to the kernels'
        // default stream; there is no reliance on legacy stream ordering.
        checked("shard stream", (cuda.stream_create)(&mut t.stream, 1))?;
        checked("shard begin event", (cuda.event_create)(&mut t.begin, 0))?;
        checked("shard end event", (cuda.event_create)(&mut t.end, 0))?;
        Ok(t)
    }
    unsafe fn start(&mut self) -> Result<Instant, String> {
        self.bytes = 0; self.calls = 0;
        checked("shard copy begin", (self.cuda.event_record)(self.begin, self.stream))?;
        Ok(Instant::now())
    }
    unsafe fn copy(&mut self, src: u64, dst: u64, bytes: usize, back: bool) -> Result<(), String> {
        if bytes == 0 { return Ok(()); }
        let (sc, dc) = if back { (self.here, self.main) } else { (self.main, self.here) };
        if self.here == self.main {
            checked("shard local copy", (self.cuda.copy)(dst, src, bytes, self.stream))?;
        } else {
            checked("shard peer copy", (self.cuda.copy_peer_async)(dst, dc, src, sc, bytes, self.stream))?;
        }
        self.bytes += bytes; self.calls += 1;
        Ok(())
    }
    unsafe fn rectangle(&mut self, full: u64, packed: u64, levels: usize, columns: usize,
                        range: Range, width: usize, back: bool) -> Result<(), String> {
        let rows = column_shard::rows(full, packed, levels, columns, range, width, back)?;
        let Some(&(src, dst, bytes)) = rows.first() else { return Ok(()); };
        // One pitched transfer per field instead of one driver call per level.
        // The checked rows above also prove both exclusive allocation endpoints.
        let (sp, dp) = if back { (bytes, columns * width) } else { (columns * width, bytes) };
        let (sc, dc) = if back { (self.here, self.main) } else { (self.main, self.here) };
        let copy = PeerCopy3D {
            src: PeerCopyEndpoint { device: src, context: sc, pitch: sp, memory_type: 2, ..Default::default() },
            dst: PeerCopyEndpoint { device: dst, context: dc, pitch: dp, memory_type: 2, ..Default::default() },
            width_bytes: bytes, height: levels, depth: 1,
        };
        let total = bytes.checked_mul(levels).ok_or("DEFECT shard copy byte count overflow")?;
        checked("shard pitched copy", (self.cuda.copy_peer_3d_async)(&copy, self.stream))?;
        self.bytes += total; self.calls += 1;
        Ok(())
    }
    unsafe fn finish(&self, t: Instant) -> Result<serde_json::Value, String> {
        checked("shard copy end", (self.cuda.event_record)(self.end, self.stream))?;
        checked("shard copy join", (self.cuda.stream_synchronize)(self.stream))?;
        let mut ms = 0f32;
        checked("shard copy elapsed", (self.cuda.event_elapsed)(&mut ms, self.begin, self.end))?;
        Ok(json!({"bytes":self.bytes,"calls":self.calls,"device_ms":ms,"wall_ms":t.elapsed().as_secs_f64()*1000.0}))
    }
}
impl Drop for Transfers<'_> {
    fn drop(&mut self) { unsafe {
        // Even an enqueue failure must not leave copies using an arena that the
        // caller is about to reset. Cleanup never replaces the original error.
        if !self.stream.is_null() { let _ = (self.cuda.stream_synchronize)(self.stream); }
        if !self.end.is_null() { let _ = (self.cuda.event_destroy)(self.end); }
        if !self.begin.is_null() { let _ = (self.cuda.event_destroy)(self.begin); }
        if !self.stream.is_null() { let _ = (self.cuda.stream_destroy)(self.stream); }
    } }
}

/// A packed snapshot retains null carriers. Rebuild tables with addresses in
/// the owning context, never copy a table of primary-context pointers.
unsafe fn snapshot(arena: &mut Arena, s: &Snap, range: Range, identity: bool, peer_blob: bool,
                   transfer: &mut Transfers<'_>) -> Result<Snap, String> {
    let mut out = s.clone();
    out.columns = range.count;
    // The generated driver tests `first && sub == 0 && c == 0`.
    // Packed local column zero is global zero only in the first range.
    if range.first != 0 { out.flags[4] = 0; }
    let mut field = |p: u64, levels: usize| -> Result<u64, String> {
        if p == 0 { return Ok(0); }
        let n = levels.checked_mul(range.count).and_then(|v| v.checked_mul(4)).ok_or("DEFECT shard field size overflow")?;
        let dst = arena.take(n)?;
        transfer.rectangle(p, dst, levels, s.columns, range, 4, false)?;
        Ok(dst)
    };
    for j in 0..18 { out.fsnap[j] = field(s.fsnap[j], s.nz)?; }
    for j in 0..10 { out.ssnap[j] = field(s.ssnap[j], 1)?; }
    out.fpost = [0; 18]; out.spost = [0; 10]; out.ppost = [0; 4];
    if identity {
        out.fpost[0] = field(s.fpost[0], s.nz)?;
        for j in 4..17 { out.fpost[j] = field(s.fpost[j], s.nz)?; }
        for j in 0..10 { out.spost[j] = field(s.spost[j], 1)?; }
        for j in 0..4 { if j == 0 || s.flags[6] != 0 { out.ppost[j] = field(s.ppost[j], s.nz)?; } }
    }
    // Both modes retain the source for this call only. Peer mode reads its
    // const TempoBlob through the already-enabled peer mapping. All workers
    // are joined before the caller may reuse source storage, including errors.
    // This is not a cache: mutations between calls remain visible.
    if transfer.here != transfer.main && !peer_blob {
        out.blob = arena.take(s.blob_bytes)?;
        transfer.copy(s.blob, out.blob, s.blob_bytes, false)?;
    }
    out.ftab = arena.take(20 * 8)?; out.stab = arena.take(12 * 8)?;
    Ok(out)
}

unsafe fn tables(cuda: &Cuda, s: &Snap) -> Result<(), String> {
    let mut f = vec![0u8; 20 * 8]; let mut v = vec![0u8; 12 * 8];
    for (j, p) in s.fsnap.iter().enumerate() { f[j * 8..j * 8 + 8].copy_from_slice(&p.to_le_bytes()); }
    for (j, p) in s.ssnap.iter().enumerate() { v[j * 8..j * 8 + 8].copy_from_slice(&p.to_le_bytes()); }
    dev::upload(cuda, s.ftab, &f)?; dev::upload(cuda, s.stab, &v)
}

unsafe fn run(cuda: &Cuda, t: &TempoAdw, arena: &mut Arena, ledger: &mut Ledger,
              transfer: &mut Transfers<'_>, job: Job, ordinal: i32) -> Result<Done, String> {
    let mark = arena.used;
    let mut phase = "scatter";
    let mut completed_scatter = None;
    let mut completed_gather = None;
    let (range, levels, adjoint) = (job.range, job.snap.nz, job.adjoint);
    let result = (|| {
        let wall = Instant::now();
        let start = transfer.start()?;
        let s = snapshot(arena, &job.snap, job.range, !job.adjoint, job.peer_blob, transfer)?;
        let rows = 18 * s.nz + 10;
        let bytes = rows.checked_mul(s.columns).and_then(|n| n.checked_mul(8)).ok_or("DEFECT shard vector size overflow")?;
        let input = arena.take(bytes)?; let output = arena.take(bytes)?;
        transfer.rectangle(job.input, input, rows, job.snap.columns, job.range, 8, false)?;
        let scatter = transfer.finish(start)?;
        completed_scatter = Some(scatter.clone());
        phase = "compute";
        let table_start = Instant::now();
        tables(cuda, &s)?;
        let table_upload_ms = table_start.elapsed().as_secs_f64() * 1000.0;
        let compute = Instant::now();
        let (tok, compared, tiles) = t.apply(cuda, arena, &s, input, output, job.adjoint, !job.adjoint, ledger)?;
        // Checks remain ON on every card. Flush before returning any result.
        ledger.flush(cuda)?;
        dev::sync(cuda)?;
        let compute_ms = compute.elapsed().as_secs_f64() * 1000.0;
        let differing = match tok {
            Some(tok) => ledger.value(tok).and_then(|v| v.first().copied()).ok_or("DEFECT shard identity count missing")? as u64,
            None => 0,
        };
        let start = transfer.start()?;
        phase = "gather";
        transfer.rectangle(job.output, output, rows, job.snap.columns, job.range, 8, true)?;
        let gather = transfer.finish(start)?;
        completed_gather = Some(gather.clone());
        Ok(Done { differing, compared, tiles, parts: t.parts.get(), receipt: json!({
            "device":ordinal,"first":job.range.first,"columns":s.columns,"levels":s.nz,"scatter":scatter,"gather":gather,
            "first_call_main":s.flags[4],
            "constant_bytes":if transfer.here != transfer.main && !job.peer_blob { s.blob_bytes } else { 0 },
            "peer_blob_reads":transfer.here != transfer.main && job.peer_blob,
            "peer_blob_mapped_bytes":if transfer.here != transfer.main && job.peer_blob { s.blob_bytes } else { 0 },
            "table_upload_bytes":256,"table_upload_wall_ms":table_upload_ms,
            "compute_ms":compute_ms,"wall_ms":wall.elapsed().as_secs_f64()*1000.0,"arena_peak_bytes":arena.peak,
            "identity_differing_words":differing,"identity_compared_words":compared,
            "flags":ledger.take_report()}) })
    })();
    // Synchronize even on error before reusing memory. No CPU physics fallback.
    let sync = dev::sync(cuda);
    if let Err(ref error) = result {
        eprintln!("{}", json!({"event":"tempo_shard_refusal","device":ordinal,"first":range.first,
            "columns":range.count,"levels":levels,"pass":if adjoint {"ad"} else {"tl"},
            "phase":phase,"error":error,"completed_scatter":completed_scatter,
            "completed_gather":completed_gather,"current_copy_phase_submitted_bytes":transfer.bytes,
            "synchronization_error":sync.as_ref().err()}));
    }
    arena.used = mark;
    result.and_then(|value| sync.map(|_| value))
}

impl Pool {
    pub unsafe fn from_env(cuda: &Cuda, primary: *mut c_void, image: &str, tlf: Option<&str>) -> Result<Option<Self>, String> {
        let text = match std::env::var("AD_TEMPO_SHARD_DEVICES") { Ok(s) => s, Err(std::env::VarError::NotPresent) => return Ok(None),
            Err(e) => return Err(format!("DEFECT AD_TEMPO_SHARD_DEVICES: {e}")) };
        let ordinals = column_shard::devices(&text)?;
        let peer_blob = match std::env::var("AD_TEMPO_SHARD_BLOB_TRANSPORT") {
            Err(std::env::VarError::NotPresent) => false,
            Ok(v) if v == "copy" => false,
            Ok(v) if v == "peer" => true,
            _ => return Err("DEFECT AD_TEMPO_SHARD_BLOB_TRANSPORT must be copy or peer".into()),
        };
        let mut main_device = 0;
        checked("shard primary device", (cuda.ctx_device)(&mut main_device))?;
        let mut first = 0;
        checked("shard first device", (cuda.device_get)(&mut first, ordinals[0]))?;
        if first != main_device { return Err("DEFECT first shard ordinal must be the dycore device".into()); }
        for &ordinal in &ordinals[1..] {
            let mut device = 0;
            checked("shard device", (cuda.device_get)(&mut device, ordinal))?;
            if device == main_device {
                // Same-card contexts (proof mode): copies between contexts of one device
                // need no peer mapping; a peer-read blob would, so refuse that combination.
                if peer_blob { return Err("DEFECT peer blob transport needs distinct cards".into()); }
                continue;
            }
            for (a, b) in [(main_device, device), (device, main_device)] {
                let mut can = 0;
                checked("shard peer capability", (cuda.device_can_access_peer)(&mut can, a, b))?;
                if can == 0 { return Err(format!("DEFECT TEMPO sharding requires peer copies between devices {a} and {b}")); }
            }
        }
        let gib = match std::env::var("AD_TEMPO_SHARD_ARENA_GIB") {
            Ok(v) => v.parse::<usize>().map_err(|_| "DEFECT AD_TEMPO_SHARD_ARENA_GIB must be an integer")?,
            Err(std::env::VarError::NotPresent) => 16,
            Err(e) => return Err(format!("DEFECT AD_TEMPO_SHARD_ARENA_GIB: {e}")),
        };
        let bytes = gib.checked_mul(1usize << 30).filter(|&n| n > 0).ok_or("DEFECT shard arena size overflow or zero")?;
        let mut pool = Self { ordinals: ordinals.clone(), workers: Vec::new(),
            primary: primary as usize, local_ledger: Ledger::new(cuda, 8 << 20)?, peer_blob };
        for &ordinal in &ordinals[1..] {
            let (tx, rx) = mpsc::channel::<Job>();
            let (answer, result) = mpsc::channel();
            let (ready, startup) = mpsc::sync_channel(1);
            let (image, tlf, main) = (image.to_owned(), tlf.map(str::to_owned), primary as usize);
            let main_device = main_device;
            let join = thread::spawn(move || {
                let mut context = ptr::null_mut();
                let loaded = Cuda::load();
                let cuda = match loaded { Ok(c) => c, Err(e) => { let _ = ready.send(Err(e)); return; } };
                let init = (|| {
                    checked("shard cuInit", (cuda.init)(0))?;
                    let mut device = 0;
                    checked("shard cuDeviceGet", (cuda.device_get)(&mut device, ordinal))?;
                    checked("shard cuCtxCreate", (cuda.context_create)(&mut context, 4, device))?;
                    if device != main_device {
                        let status = (cuda.context_enable_peer)(main as *mut c_void, 0);
                        if status != 704 { checked("shard peer enable", status)?; }
                    }
                    let mut tempo = TempoAdw::load(&cuda, &image)?;
                    if let Some(path) = tlf { tempo.load_tlf(&cuda, &path)?; }
                    let arena = Arena::new(&cuda, bytes)?;
                    let ledger = Ledger::new(&cuda, 8 << 20)?;
                    let transfer = Transfers::new(&cuda, context, main as *mut c_void)?;
                    Ok::<_, String>((tempo, arena, ledger, transfer))
                })();
                match init {
                    Err(e) => { let _ = ready.send(Err(format!("DEFECT shard device {ordinal}: {e}"))); },
                    Ok((tempo, mut arena, mut ledger, mut transfer)) => {
                        if ready.send(Ok(context as usize)).is_ok() {
                            while let Ok(job) = rx.recv() {
                                let done = run(&cuda, &tempo, &mut arena, &mut ledger, &mut transfer, job, ordinal);
                                let failed = done.is_err();
                                if answer.send(done).is_err() || failed { break; }
                            }
                        }
                    },
                }
                if !context.is_null() { let _ = (cuda.context_synchronize)(); let _ = (cuda.context_destroy)(context); }
            });
            // Add before receiving so failure tears down every started thread.
            pool.workers.push(Worker { tx, rx: result, join });
            let peer = startup.recv().map_err(|_| "DEFECT shard startup thread exited")??;
            let mut device = 0;
            checked("shard device", (cuda.device_get)(&mut device, ordinal))?;
            if device != main_device {
                let status = (cuda.context_enable_peer)(peer as *mut c_void, 0);
                if status != 704 { checked("primary peer enable", status)?; }
            }
        }
        eprintln!("{}", json!({"event":"tempo_shards_loaded","visible_ordinals":ordinals,"worker_arena_bytes":bytes,
            "image":image,"tlf":tlf,"transport":"peer_async","dycore_device":main_device,
            "blob_transport":if peer_blob { "peer_read" } else { "peer_copy" }}));
        Ok(Some(pool))
    }

    pub unsafe fn apply(&mut self, cuda: &Cuda, t: &TempoAdw, arena: &mut Arena, s: &Snap,
                         input: u64, output: u64, adjoint: bool, ledger: &mut Ledger)
        -> Result<(Option<u64>, usize, usize), String> {
        // F08: a PBL subgrid-radii launch runs on the main card's taped route (its snapshot of bl is not split by range).
        if s.blsnap != 0 { return t.apply(cuda, arena, s, input, output, adjoint, !adjoint, ledger); }
        // The dycore is stopped at this dependency boundary. Use its card for
        // one shard while helpers process the other disjoint column ranges.
        let ranges = column_shard::partition(s.columns, self.ordinals.len())?;
        eprintln!("{}", json!({"event":"tempo_shards_dispatch","pass":if adjoint {"ad"} else {"tl"},
            "columns":s.columns,"levels":s.nz,"ranges":self.ordinals.iter().zip(&ranges)
                .map(|(d,r)| json!({"device":d,"first":r.first,"columns":r.count})).collect::<Vec<_>>()}));
        // Snapshot/input production must finish before peer streams read it.
        let wall = Instant::now();
        dev::sync(cuda)?;
        let ready_wait_ms = wall.elapsed().as_secs_f64() * 1000.0;
        let peer_blob = self.peer_blob;
        let job = |range| Job { snap: s.clone(), range, input, output, adjoint, peer_blob };
        let mut sent = 0;
        let mut error = None;
        for (worker, &range) in self.workers.iter().zip(&ranges[1..]) {
            if worker.tx.send(job(range)).is_err() { error = Some("DEFECT TEMPO shard worker stopped".to_owned()); break; }
            sent += 1;
        }
        let mut results = Vec::new();
        if error.is_none() {
            let local = (|| {
                let primary = self.primary as *mut c_void;
                let mut transfer = Transfers::new(cuda, primary, primary)?;
                run(cuda, t, arena, &mut self.local_ledger, &mut transfer,
                    job(ranges[0]), self.ordinals[0])
            })();
            match local { Ok(r) => results.push(r), Err(e) => error = Some(e) }
        }
        // Drain every submitted job, including after a dispatch failure.
        // No early return can leave a helper writing freed primary buffers.
        for worker in &self.workers[..sent] {
            match worker.rx.recv() {
                Ok(Ok(r)) => results.push(r),
                Ok(Err(e)) => { if error.is_none() { error = Some(e); } },
                Err(_) => { if error.is_none() { error = Some("DEFECT TEMPO shard reply missing".into()); } },
            }
        }
        if let Some(e) = error { return Err(e); }
        let differing = results.iter().map(|r| r.differing).sum::<u64>();
        let compared = results.iter().map(|r| r.compared).sum();
        let tiles = results.iter().map(|r| r.tiles).sum();
        let token = if !adjoint {
            let (address, token) = ledger.reserve(cuda, 8, dev::LedgerItem::Count)?;
            dev::upload(cuda, address, &differing.to_le_bytes())?;
            Some(token)
        } else { None };
        // Parts are concurrent: report each in the shard receipt, never add
        // them and label that sum an elapsed time.
        t.parts.set([0.0; 4]);
        eprintln!("{}", json!({"event":"window_tempo_shards","pass":if adjoint {"ad"} else {"tl"},
            "wall_ms":wall.elapsed().as_secs_f64()*1000.0,"cards":self.ordinals,"columns":s.columns,"levels":s.nz,
            "primary_device":self.ordinals[0],"leased_cards":self.ordinals,
            "global_first_call_main":s.flags[4],
            "input_ready_wait_ms":ready_wait_ms,
            "shards":results.iter().map(|r| &r.receipt).collect::<Vec<_>>(),
            "parts_ms":results.iter().map(|r| r.parts).collect::<Vec<_>>(),
            "dycore_overlap":false}));
        Ok((token, compared, tiles))
    }
}
impl Drop for Pool {
    fn drop(&mut self) {
        let mut joins = Vec::new();
        for Worker { tx, rx, join } in self.workers.drain(..) { drop(tx); drop(rx); joins.push(join); }
        for join in joins { let _ = join.join(); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn driver_peer_copy_layout() {
        // CUDA 13 cuda.h: CUDA_MEMCPY3D_PEER_v1, checked by the C ABI probe
        // retained in receipts/cuda-peer-layout.log. This host is 64-bit.
        assert_eq!(std::mem::size_of::<PeerCopyEndpoint>(), 88);
        assert_eq!(std::mem::size_of::<PeerCopy3D>(), 200);
        assert_eq!(std::mem::offset_of!(PeerCopyEndpoint, memory_type), 32);
        assert_eq!(std::mem::offset_of!(PeerCopyEndpoint, device), 48);
        assert_eq!(std::mem::offset_of!(PeerCopyEndpoint, context), 64);
        assert_eq!(std::mem::offset_of!(PeerCopyEndpoint, pitch), 72);
        assert_eq!(std::mem::offset_of!(PeerCopy3D, dst), 88);
        assert_eq!(std::mem::offset_of!(PeerCopy3D, width_bytes), 176);
    }
}
