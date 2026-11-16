// SPDX-License-Identifier: Apache-2.0
//! two-cards: the same-host rank exchange without host waits (AD_PEER_SHM=1; off by default: the file mailbox
//! of peer.rs is the base and stays the default).
//!
//! The file mailbox moves every halo copy through the host: the sending rank drains its card (cuCtxSynchronize),
//! copies the bytes to pageable memory, writes and renames a tmpfs file; the receiving rank polls for the file (backing
//! off to 2 ms), reads it, drains its card, uploads and drains again. A 2-rank 120 km plan has 43 sends and 42 receives
//! a step (pk120-r2-e-r0, step_02), so a backward step carries about 85 such host round trips per rank, each of which
//! empties the card's queue.
//!
//! This path keeps the same messages and names (peer.rs keys: step, channel, RK scope) but moves the bytes and the
//! ordering onto the cards, as hex's own forecast ranks do ("halo packets cross between processes through shared host
//! memory registered for DMA in both, ordered on the cards by flags in that memory"):
//!   send:    cuMemcpyDtoHAsync(slot <- source) ; cuStreamWriteValue32(slot.ready = 1)        (the sender's stream)
//!   receive: cuStreamWaitValue32(slot.ready == 1) ; cuMemcpyHtoDAsync(destination <- slot)   (the receiver's stream)
//! No host thread waits for a card or for a peer on the step path.
//!
//! Slots. One shared table per message kind (f forward, t tangent, a cotangent) in AD_PEER_DIR, open addressing on the
//! 64-bit hash of the message name; whichever rank meets a name first claims its entry (CAS on the shared word) and its
//! data range (CAS bump in the table head), the other finds it. So no rank relies on the other's order of messages.
//! Data lives in 1 GiB chunk files of the same directory, mapped and page-locked (cuMemHostRegister, device-mapped) by
//! each rank once. Forward messages stay for the whole run: every recompute of the base trajectory (seed passes, the
//! reverse's restore + redo, anchors, segments) reads the same slot again and the sender never re-sends (as the file
//! mailbox's "written once"). Tangent and cotangent slots are reused pass by pass: begin_tl / begin_ad is a barrier
//! (each rank drains its card, then rank 0 clears that kind's table between two file barriers). THE BREAKAGE THE
//! BARRIER PREVENTS: a sender of pass e+1 overwriting a slot whose pass-e copy the peer's card has not yet read.
//!
//! What it refuses: a rank on another host (AD_PEER_SEND: that host cannot see this host's memory); a peer copy inside
//! a stream capture (as the file path does). A dead peer leaves a card waiting on a flag that never comes, so a
//! watchdog thread ends the process when the job's ABORT file appears or the launcher is gone.
use std::ffi::{c_char, c_int, c_void, CString};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::*};
use std::sync::{Mutex, OnceLock};
use std::path::{Path, PathBuf};

pub const CAP: usize = 1 << 17;          // entries a kind holds (a 59-step tangent pass of a 2-rank plan needs about 5,000)
const ENTRY: usize = 64;
const HEAD: usize = 64;
pub const CHUNK: u64 = 1 << 30;           // bytes of one data chunk file
const KINDS: [u8; 3] = [b'f', b't', b'a'];

/// On only with the rank configuration and AD_PEER_SHM=1 or 2.
/// 1 (fenced): every copy keeps the base mailbox's context drains (before and after a send, before and after a receive), so a
///   producer or consumer of a halo buffer on any stream of the rank, joined to the others or not, gets the order the
///   base gave it. Only the file write, the file read and the pageable copies go.
/// 2 (stream-ordered): no drains; the copies are ordered on the stream that issues them (the window walk's legacy
///   stream, the forward command's own stream). Correct only if every side stream that touches a halo buffer is
///   joined to that stream by an event; run 2 of the card plan tests it (same sha256 as the file mailbox, twice).
pub fn on() -> bool { mode() != 0 }
pub fn fenced() -> bool { mode() == 1 }
fn mode() -> u8 {
    static M: OnceLock<u8> = OnceLock::new();
    *M.get_or_init(|| if crate::peer::get().is_none() { 0 } else {
        match std::env::var("AD_PEER_SHM").as_deref() { Ok("1") => 1, Ok("2") => 2, _ => 0 } })
}
/// AD_PEER_SHM=1 only: the base mailbox's context drain at this point of a copy (nothing when stream-ordered).
/// After a send it stands for the base's synchronous download: without it a later write to the sent buffer from a
/// non-blocking side stream (not ordered behind the copy) could change the bytes before the copy reads them.
pub unsafe fn fence(cuda: &crate::cuda::Cuda) -> Result<(), String> {
    if fenced() { crate::cuda::checked("cuCtxSynchronize (peer shm fence)", (cuda.context_synchronize)())?; }
    Ok(())
}

fn fnv(b: &[u8]) -> u64 { let mut h = 0xcbf29ce484222325u64; for x in b { h ^= *x as u64; h = h.wrapping_mul(0x100000001b3); } h }

/// One kind's shared table: head (bump u64 at 0) and CAP entries of {key u64, off u64, bytes u64, ready u32, ...}.
pub struct Table { pub base: *mut u8, pub dev: u64 }
unsafe impl Send for Table {}
unsafe impl Sync for Table {}
pub struct Chunk { pub host: *mut u8, pub dev: u64 }
unsafe impl Send for Chunk {}
unsafe impl Sync for Chunk {}

/// The engine moves bytes and flags: the card (stream-ordered driver calls) in the runner, the host in the CPU test.
pub trait Engine: Sync {
    /// Make `host` (len bytes, shared mapping) reachable by the engine; return its device address.
    unsafe fn register(&self, host: *mut u8, len: usize) -> Result<u64, String>;
    unsafe fn send(&self, stream: *mut c_void, slot_host: *mut u8, slot_dev: u64, src: u64, bytes: usize, ready_host: *mut u32, ready_dev: u64) -> Result<(), String>;
    unsafe fn recv(&self, stream: *mut c_void, slot_host: *mut u8, slot_dev: u64, dst: u64, bytes: usize, ready_host: *mut u32, ready_dev: u64) -> Result<(), String>;
    unsafe fn drain(&self) -> Result<(), String>;
}

pub struct Shm<E: Engine> {
    pub dir: PathBuf, pub rank: usize, pub ranks: usize, pub engine: E,
    tables: Vec<Table>,
    chunks: Mutex<std::collections::HashMap<(u8, u64), Chunk>>,
    sent_f: Mutex<std::collections::HashSet<u64>>,
    seen: Mutex<std::collections::HashSet<u64>>,
    pub stats: Mutex<[u64; 4]>,   // sends, receives, bytes out, bytes in
    pass: Mutex<[u64; 3]>,
}

unsafe fn map_file(path: &Path, len: u64) -> Result<*mut u8, String> {
    extern "C" { fn mmap(a: *mut c_void, l: usize, p: c_int, f: c_int, fd: c_int, o: i64) -> *mut c_void; }
    use std::os::unix::io::AsRawFd;
    let f = std::fs::OpenOptions::new().read(true).write(true).create(true).open(path).map_err(|e| format!("peer shm open {}: {e}", path.display()))?;
    // both ranks may size it; the size is the same, so the second call changes nothing
    if f.metadata().map_err(|e| e.to_string())?.len() < len { f.set_len(len).map_err(|e| format!("peer shm size {}: {e}", path.display()))?; }
    let p = mmap(std::ptr::null_mut(), len as usize, 3, 1, f.as_raw_fd(), 0);   // PROT_READ|PROT_WRITE, MAP_SHARED
    if p as isize == -1 || p.is_null() { return Err(format!("peer shm mmap {}", path.display())); }
    Ok(p.cast())
}

impl<E: Engine> Shm<E> {
    pub unsafe fn open(dir: PathBuf, rank: usize, ranks: usize, engine: E) -> Result<Self, String> {
        std::fs::create_dir_all(&dir).map_err(|e| format!("peer shm {}: {e}", dir.display()))?;
        let mut tables = Vec::new();
        for k in KINDS {
            let len = (HEAD + CAP * ENTRY) as u64;
            let base = map_file(&dir.join(format!("shm.{}.table", k as char)), len)?;
            let dev = engine.register(base, len as usize)?;
            tables.push(Table { base, dev });
        }
        Ok(Shm { dir, rank, ranks, engine, tables, chunks: Default::default(), sent_f: Default::default(), seen: Default::default(),
                 stats: Mutex::new([0; 4]), pass: Mutex::new([0; 3]) })
    }
    fn ki(kind: u8) -> usize { KINDS.iter().position(|k| *k == kind).unwrap() }
    unsafe fn entry(&self, kind: u8, i: usize) -> *mut u8 { self.tables[Self::ki(kind)].base.add(HEAD + i * ENTRY) }
    unsafe fn chunk(&self, kind: u8, c: u64) -> Result<(*mut u8, u64), String> {
        let mut g = self.chunks.lock().unwrap();
        if let Some(x) = g.get(&(kind, c)) { return Ok((x.host, x.dev)); }
        let host = map_file(&self.dir.join(format!("shm.{}.d{c}", kind as char)), CHUNK)?;
        let dev = self.engine.register(host, CHUNK as usize)?;
        g.insert((kind, c), Chunk { host, dev });
        Ok((host, dev))
    }
    /// The entry of `name` (claimed by whichever rank comes first) and its data range.
    /// Returns (slot host, slot device, ready host, ready device).
    pub unsafe fn slot(&self, kind: u8, name: &str, bytes: usize) -> Result<(*mut u8, u64, *mut u32, u64), String> {
        if bytes as u64 > CHUNK { return Err(format!("peer shm {name}: {bytes} bytes exceed a chunk")); }
        let h = fnv(name.as_bytes()) | 1;
        let t = &self.tables[Self::ki(kind)];
        let bump = &*(t.base as *const AtomicU64);
        let mut i = (h as usize) % CAP;
        for _ in 0..CAP {
            let e = self.entry(kind, i);
            let key = &*(e as *const AtomicU64);
            let off = &*(e.add(8) as *const AtomicU64);
            let len = &*(e.add(16) as *const AtomicU64);
            let k = key.load(Acquire);
            let mine = if k == 0 { key.compare_exchange(0, h, AcqRel, Acquire).is_ok() } else { false };
            if mine {
                // claim a data range that does not cross a chunk end (CAS on the shared bump)
                let b = ((bytes as u64) + 255) & !255;
                let mut cur = bump.load(Acquire);
                let start = loop {
                    let s = if (cur % CHUNK) + b > CHUNK { (cur / CHUNK + 1) * CHUNK } else { cur };
                    match bump.compare_exchange(cur, s + b, AcqRel, Acquire) { Ok(_) => break s, Err(x) => cur = x }
                };
                off.store(start, Release);
                len.store(bytes as u64, Release);
            } else if key.load(Acquire) != h { i = (i + 1) % CAP; continue; }
            // the claimer sets the length right after its claim: wait for it (host words, nanoseconds)
            let mut spins = 0u64;
            while len.load(Acquire) == 0 { std::hint::spin_loop(); spins += 1; if spins > 4_000_000_000 { return Err(format!("peer shm {name}: entry never completed")); } }
            if len.load(Acquire) != bytes as u64 {
                return Err(format!("peer shm {name}: the peer's entry holds {} bytes, this copy moves {bytes}", len.load(Acquire)));
            }
            let o = off.load(Acquire);
            let (ch, cd) = self.chunk(kind, o / CHUNK)?;
            let r = e.add(24) as *mut u32;
            let rdev = t.dev + (HEAD + i * ENTRY + 24) as u64;
            return Ok((ch.add((o % CHUNK) as usize), cd + o % CHUNK, r, rdev));
        }
        Err(format!("peer shm: table {} full ({CAP} entries)", kind as char))
    }
    fn name(&self, kind: u8, key: &str) -> String {
        let e = self.pass.lock().unwrap()[Self::ki(kind)];
        format!("{}{e}.{key}", kind as char)
    }
    /// The send half of one copy: `src` (a device address, or a host address for the CPU engine) to the slot of `key`.
    pub unsafe fn send(&self, kind: u8, stream: *mut c_void, key: &str, src: u64, bytes: usize) -> Result<bool, String> {
        let name = self.name(kind, key);
        if kind == b'f' && !self.sent_f.lock().unwrap().insert(fnv(name.as_bytes())) { return Ok(false); }  // written once
        let (sh, sd, rh, rd) = self.slot(kind, &name, bytes)?;
        // THE BREAKAGE THIS PREVENTS (as put_tl): two executions of one copy under one scope would overwrite a slot
        if kind != b'f' && !self.seen.lock().unwrap().insert(fnv(name.as_bytes())) {
            return Err(format!("peer shm: {name} sent twice in one pass (two executions of one copy share a scope key)"));
        }
        let _ = rh;
        self.engine.send(stream, sh, sd, src, bytes, rh, rd)?;
        let mut s = self.stats.lock().unwrap(); s[0] += 1; s[2] += bytes as u64;
        Ok(true)
    }
    pub fn has_forward(&self, key: &str) -> bool { self.sent_f.lock().unwrap().contains(&fnv(self.name(b'f', key).as_bytes())) }
    pub unsafe fn recv(&self, kind: u8, stream: *mut c_void, key: &str, dst: u64, bytes: usize) -> Result<(), String> {
        let name = self.name(kind, key);
        let (sh, sd, rh, rd) = self.slot(kind, &name, bytes)?;
        self.engine.recv(stream, sh, sd, dst, bytes, rh, rd)?;
        let mut s = self.stats.lock().unwrap(); s[1] += 1; s[3] += bytes as u64;
        Ok(())
    }
    fn wait_files(&self, tag: &str) -> Result<(), String> {
        std::fs::write(self.dir.join(format!("shm.bar.{tag}.{}", self.rank)), b"").map_err(|e| e.to_string())?;
        let t = std::time::Instant::now();
        for r in 0..self.ranks {
            let p = self.dir.join(format!("shm.bar.{tag}.{r}"));
            while !p.exists() {
                if self.dir.join("ABORT").exists() { return Err(format!("peer shm barrier {tag}: a peer aborted")); }
                if t.elapsed().as_secs() > 1800 { return Err(format!("peer shm barrier {tag}: rank {r} never came")); }
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
        }
        Ok(())
    }
    /// Start pass `e` of kind t or a on every rank (both ranks run the same passes in the same order).
    pub unsafe fn begin(&self, kind: u8, e: u64) -> Result<(), String> {
        self.engine.drain()?;
        let tag = format!("{}{e}", kind as char);
        self.wait_files(&format!("{tag}.A"))?;
        if self.rank == 0 {
            let t = &self.tables[Self::ki(kind)];
            std::ptr::write_bytes(t.base, 0, HEAD + CAP * ENTRY);
            std::sync::atomic::fence(SeqCst);
        }
        self.wait_files(&format!("{tag}.B"))?;
        self.pass.lock().unwrap()[Self::ki(kind)] = e;
        self.seen.lock().unwrap().clear();
        Ok(())
    }
}

// ---- the card engine ----------------------------------------------------------------------------------------------
type WV = unsafe extern "C" fn(*mut c_void, u64, u32, u32) -> i32;
type HDP = unsafe extern "C" fn(*mut u64, *mut c_void, u32) -> i32;
pub struct Card {
    host_register: unsafe extern "C" fn(*mut c_void, usize, u32) -> i32,
    device_pointer: HDP,
    write_value: WV,
    wait_value: WV,
    d2h: unsafe extern "C" fn(*mut c_void, u64, usize, *mut c_void) -> i32,
    h2d: unsafe extern "C" fn(u64, *const c_void, usize, *mut c_void) -> i32,
    sync: unsafe extern "C" fn() -> i32,
}
extern "C" { fn dlopen(f: *const c_char, flags: c_int) -> *mut c_void; fn dlsym(h: *mut c_void, s: *const c_char) -> *mut c_void; }
unsafe fn sym<T: Copy>(lib: *mut c_void, name: &str) -> Result<T, String> {
    let c = CString::new(name).unwrap();
    let a = dlsym(lib, c.as_ptr());
    if a.is_null() { return Err(format!("peer shm: CUDA symbol {name} missing")); }
    Ok(std::mem::transmute_copy(&a))
}
impl Card {
    pub unsafe fn load(cuda: &crate::cuda::Cuda) -> Result<Self, String> {
        let lib = dlopen(c"libcuda.so.1".as_ptr(), 2);
        if lib.is_null() { return Err("peer shm: cannot load libcuda.so.1".into()); }
        Ok(Card { host_register: cuda.host_register, device_pointer: sym(lib, "cuMemHostGetDevicePointer_v2")?,
                  write_value: sym(lib, "cuStreamWriteValue32_v2")?, wait_value: sym(lib, "cuStreamWaitValue32_v2")?,
                  d2h: cuda.download_async, h2d: cuda.upload_async, sync: cuda.context_synchronize })
    }
}
impl Engine for Card {
    unsafe fn register(&self, host: *mut u8, len: usize) -> Result<u64, String> {
        // CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP
        crate::cuda::checked("cuMemHostRegister (peer shm)", (self.host_register)(host.cast(), len, 3))?;
        let mut d = 0u64;
        crate::cuda::checked("cuMemHostGetDevicePointer (peer shm)", (self.device_pointer)(&mut d, host.cast(), 0))?;
        Ok(d)
    }
    unsafe fn send(&self, s: *mut c_void, sh: *mut u8, _sd: u64, src: u64, bytes: usize, _rh: *mut u32, rd: u64) -> Result<(), String> {
        crate::cuda::checked("cuMemcpyDtoHAsync (peer shm send)", (self.d2h)(sh.cast(), src, bytes, s))?;
        // flags 0: a memory barrier before the write, so the bytes are visible before the flag
        crate::cuda::checked("cuStreamWriteValue32 (peer shm send)", (self.write_value)(s, rd, 1, 0))
    }
    unsafe fn recv(&self, s: *mut c_void, sh: *mut u8, _sd: u64, dst: u64, bytes: usize, _rh: *mut u32, rd: u64) -> Result<(), String> {
        // CU_STREAM_WAIT_VALUE_EQ
        crate::cuda::checked("cuStreamWaitValue32 (peer shm receive)", (self.wait_value)(s, rd, 1, 1))?;
        crate::cuda::checked("cuMemcpyHtoDAsync (peer shm receive)", (self.h2d)(dst, sh.cast(), bytes, s))
    }
    unsafe fn drain(&self) -> Result<(), String> { crate::cuda::checked("cuCtxSynchronize (peer shm barrier)", (self.sync)()) }
}

// ---- the runner's instance ----------------------------------------------------------------------------------------
static SHM: OnceLock<Result<Shm<Card>, String>> = OnceLock::new();
/// The process's card exchange (first use maps the tables; the CUDA context must be current).
pub fn get(cuda: &crate::cuda::Cuda) -> Result<&'static Shm<Card>, String> {
    SHM.get_or_init(|| unsafe {
        let p = crate::peer::require("peer shm")?;
        if p.remote_hosts() > 0 { return Err("AD_PEER_SHM=1 is one host only: a rank on another host (AD_PEER_SEND) cannot see this host's memory".into()); }
        let s = Shm::open(p.dir().to_path_buf(), p.rank, p.ranks, Card::load(cuda)?)?;
        watchdog(p.dir().to_path_buf());
        Ok(s)
    }).as_ref().map_err(|e| e.clone())
}
pub fn started() -> Option<&'static Shm<Card>> { SHM.get().and_then(|r| r.as_ref().ok()) }
/// A card waiting on a flag of a dead peer never returns: end the process when the job is over.
fn watchdog(dir: PathBuf) {
    let launcher: Option<u32> = std::env::var("AD_PEER_LAUNCHER").ok().and_then(|v| v.parse().ok());
    std::thread::spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_millis(500));
        if let Ok(x) = std::fs::read_to_string(dir.join("ABORT")) { eprintln!("peer shm watchdog: a peer aborted: {x}"); std::process::exit(70); }
        if let Some(p) = launcher { if !Path::new(&format!("/proc/{p}")).exists() { eprintln!("peer shm watchdog: the launcher (pid {p}) is gone"); std::process::exit(70); } }
    });
}
/// begin_tl / begin_ad: the pass barrier (always run when AD_PEER_SHM=1, so both ranks meet it whether or not their
/// card exchange has started; a rank that has not started has nothing in flight to drain).
pub fn begin(kind: u8, e: u64) -> Result<(), String> {
    if !on() { return Ok(()); }
    match started() {
        Some(s) => unsafe { s.begin(kind, e) },
        None => {
            // no exchange yet in this process: meet the barrier without a card
            let p = crate::peer::require("peer shm")?;
            let tmp = BarrierOnly { dir: p.dir().to_path_buf(), rank: p.rank, ranks: p.ranks };
            tmp.begin(kind, e)
        }
    }
}
struct BarrierOnly { dir: PathBuf, rank: usize, ranks: usize }
impl BarrierOnly {
    fn wait(&self, tag: &str) -> Result<(), String> {
        std::fs::write(self.dir.join(format!("shm.bar.{tag}.{}", self.rank)), b"").map_err(|e| e.to_string())?;
        for r in 0..self.ranks { let p = self.dir.join(format!("shm.bar.{tag}.{r}")); while !p.exists() {
            if self.dir.join("ABORT").exists() { return Err(format!("peer shm barrier {tag}: a peer aborted")); }
            std::thread::sleep(std::time::Duration::from_micros(200)); } }
        Ok(())
    }
    fn begin(&self, kind: u8, e: u64) -> Result<(), String> {
        let tag = format!("{}{e}", kind as char);
        self.wait(&format!("{tag}.A"))?;
        if self.rank == 0 {
            // the table may exist (another rank started): clear it as the started path does
            let p = self.dir.join(format!("shm.{}.table", kind as char));
            if p.exists() { unsafe { let b = map_file(&p, (HEAD + CAP * ENTRY) as u64)?; std::ptr::write_bytes(b, 0, HEAD + CAP * ENTRY); } }
        }
        self.wait(&format!("{tag}.B"))?;
        if let Some(s) = started() { s.pass.lock().unwrap()[Shm::<Card>::ki(kind)] = e; }
        Ok(())
    }
}

// ---- the CPU engine and test --------------------------------------------------------------------------------------
/// Host copies and host flags: the same tables, slots, names and barrier as the card engine (the CPU test's engine).
pub struct Host;
impl Engine for Host {
    unsafe fn register(&self, host: *mut u8, _len: usize) -> Result<u64, String> { Ok(host as u64) }
    unsafe fn send(&self, _s: *mut c_void, sh: *mut u8, _sd: u64, src: u64, bytes: usize, rh: *mut u32, _rd: u64) -> Result<(), String> {
        std::ptr::copy_nonoverlapping(src as *const u8, sh, bytes);
        (*(rh as *const AtomicU32)).store(1, Release);
        Ok(())
    }
    unsafe fn recv(&self, _s: *mut c_void, sh: *mut u8, _sd: u64, dst: u64, bytes: usize, rh: *mut u32, _rd: u64) -> Result<(), String> {
        let t = std::time::Instant::now();
        while (*(rh as *const AtomicU32)).load(Acquire) != 1 {
            std::hint::spin_loop();
            if t.elapsed().as_secs() > 60 { return Err("host engine: flag never set".into()); }
        }
        std::ptr::copy_nonoverlapping(sh, dst as *mut u8, bytes);
        Ok(())
    }
    unsafe fn drain(&self) -> Result<(), String> { Ok(()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Real 3 km words: potential temperature and water vapour of 1,000 cells x 60 levels of package
    /// k3_qv, narrowed to float32), in fixtures/speed16-k3-th-qv.f32.
    fn real_words() -> Vec<f32> {
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "/src/runner/fixtures/speed16-k3-th-qv.f32");
        let b = std::fs::read(p).expect("fixtures/speed16-k3-th-qv.f32");
        assert_eq!(b.len(), 2 * 1000 * 60 * 4);
        b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }
    /// The message sizes of one step of the 2-rank 120 km plan (pk120-r2-e-r0 step_02, bytes of its peer copies, in
    /// plan order), cut to 12 for test time and scaled by 1/8.
    const SIZES: [usize; 12] = [869188, 455952, 81916, 6609592, 303968, 869188, 455952, 869188, 81916, 303968, 455952, 869188];
    fn message(words: &[f32], step: usize, i: usize, bytes: usize) -> Vec<u8> {
        let n = bytes / 4; let k0 = (step * 7919 + i * 104729) % words.len();
        (0..n).flat_map(|j| words[(k0 + j) % words.len()].to_le_bytes()).collect()
    }
    /// Two ranks (threads, each with its own mapping of the shared files) run a tangent-like pass (rank 0 sends, rank 1
    /// receives) and a cotangent-like pass (back), through the unchanged file mailbox and through the shared slots.
    /// Returns, per rank, every received message of both routes. `plant`: rank 0's shared-slot send of message 0 of
    /// step 1 carries word 0 times 1.001 (the planted error).
    fn exchange(plant: bool) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let words = std::sync::Arc::new(real_words());
        let dir = std::env::temp_dir().join(format!("speed16-shm-{}-{}", std::process::id(), plant));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut hs = Vec::new();
        for rank in 0..2usize {
            let (dir, words) = (dir.clone(), words.clone());
            hs.push(std::thread::spawn(move || {
                let filed = crate::peer::make_for_test(rank, 2, dir.join("files"));
                let shm = unsafe { Shm::open(dir.join("shm"), rank, 2, Host).unwrap() };
                let (mut via_file, mut via_shm) = (Vec::new(), Vec::new());
                filed.begin_tl();
                unsafe { shm.begin(b't', 1).unwrap(); }
                for step in 1..=3usize {
                    for (i, b) in SIZES.iter().map(|b| b / 8 / 4 * 4).enumerate() {
                        let key = crate::peer::key(step, 500 + i as u64, "Some(0)|Some(1)|Some(2)");
                        if rank == 0 {
                            let m = message(&words, step, i, b);
                            filed.put_tl(&key, &m).unwrap();
                            let mut s = m.clone();
                            if plant && step == 1 && i == 0 { let w = f32::from_le_bytes(s[0..4].try_into().unwrap()) * 1.001; s[0..4].copy_from_slice(&w.to_le_bytes()); }
                            unsafe { shm.send(b't', std::ptr::null_mut(), &key, s.as_ptr() as u64, b).unwrap(); }
                        } else {
                            via_file.push(filed.get_tl(&key, b).unwrap());
                            let mut d = vec![0u8; b];
                            unsafe { shm.recv(b't', std::ptr::null_mut(), &key, d.as_mut_ptr() as u64, b).unwrap(); }
                            via_shm.push(d);
                        }
                    }
                }
                // the cotangent pass: rank 1 sends back what rank 0 adds (reverse order)
                filed.begin_ad();
                unsafe { shm.begin(b'a', 1).unwrap(); }
                for step in (1..=3usize).rev() {
                    for (i, b) in SIZES.iter().map(|b| b / 8 / 4 * 4).enumerate().rev() {
                        let key = crate::peer::key(step, 500 + i as u64, "Some(0)|Some(1)|Some(2)");
                        if rank == 1 {
                            let m = message(&words, step + 100, i, b);
                            filed.put_ad(&key, &m).unwrap();
                            unsafe { shm.send(b'a', std::ptr::null_mut(), &key, m.as_ptr() as u64, b).unwrap(); }
                        } else {
                            via_file.push(filed.get_ad(&key, b).unwrap());
                            let mut d = vec![0u8; b];
                            unsafe { shm.recv(b'a', std::ptr::null_mut(), &key, d.as_mut_ptr() as u64, b).unwrap(); }
                            via_shm.push(d);
                        }
                    }
                }
                (via_file, via_shm)
            }));
        }
        let mut file_all = Vec::new(); let mut shm_all = Vec::new();
        for h in hs { let (f, s) = h.join().unwrap(); file_all.extend(f); shm_all.extend(s); }
        let _ = std::fs::remove_dir_all(&dir);
        (file_all, shm_all)
    }
    fn differing_words(a: &[Vec<u8>], b: &[Vec<u8>]) -> usize {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| { assert_eq!(x.len(), y.len()); x.chunks_exact(4).zip(y.chunks_exact(4)).filter(|(p, q)| p != q).count() }).sum()
    }
    #[test]
    fn shm_exchange_equals_file_mailbox_on_real_3km_words() {
        let (f, s) = exchange(false);
        assert_eq!(f.len(), 2 * 3 * SIZES.len());
        assert_eq!(differing_words(&f, &s), 0, "the shared-slot route moved different words than the file mailbox");
    }
    #[test]
    fn planted_x1001_error_is_caught() {
        let (f, s) = exchange(true);
        assert_eq!(differing_words(&f, &s), 1, "a x1.001 word planted in one shared-slot send must show as exactly one differing word");
    }
    #[test]
    fn forward_slots_are_written_once_and_reread() {
        let dir = std::env::temp_dir().join(format!("speed16-shm-fwd-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let words = real_words();
        let shm0 = unsafe { Shm::open(dir.clone(), 0, 2, Host).unwrap() };
        let shm1 = unsafe { Shm::open(dir.clone(), 1, 2, Host).unwrap() };
        let m = message(&words, 1, 0, 4096);
        let key = crate::peer::key(1, 7, "-");
        assert!(unsafe { shm0.send(b'f', std::ptr::null_mut(), &key, m.as_ptr() as u64, 4096).unwrap() });
        // a recompute re-sends: skipped, the slot keeps the first bytes
        let other = message(&words, 2, 0, 4096);
        assert!(!unsafe { shm0.send(b'f', std::ptr::null_mut(), &key, other.as_ptr() as u64, 4096).unwrap() });
        assert!(shm0.has_forward(&key));
        for _ in 0..3 {
            let mut d = vec![0u8; 4096];
            unsafe { shm1.recv(b'f', std::ptr::null_mut(), &key, d.as_mut_ptr() as u64, 4096).unwrap(); }
            assert_eq!(d, m);
        }
        // a size disagreement is refused, not read
        let mut d = vec![0u8; 8192];
        assert!(unsafe { shm1.recv(b'f', std::ptr::null_mut(), &key, d.as_mut_ptr() as u64, 8192) }.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

// ---- ad-binder peer-bench: one step's halo pattern of a real rank plan, file mailbox against shared slots ----------
/// ad-binder peer-bench --ops FILE --data WORDS.f32 --steps N --transport file|shm|none [--work-bytes B] [--plant 1]
/// [--out FILE]. Run as one process per rank with AD_PEER_RANK, AD_PEER_RANKS, AD_PEER_DIR (and AD_PEER_SHM=1 for
/// shm). FILE holds this rank's peer copies of one plan step in plan order, one per line: `send|recv CHANNEL BYTES`
/// (from a split plan's blocks). Each step runs the tangent pattern (send = this rank's bytes to the peer, recv = the
/// peer's bytes into this rank) and then the cotangent pattern in reverse order (recv sends its cotangent and zeroes
/// it, send receives it, as window.rs does; the add kernel is a plain copy on both routes). Before every copy a memset
/// of B bytes stands for the card work between halo rounds. Every received word of both patterns is compared with the
/// words the peer sent (real words from WORDS.f32, a different window per step and channel); --plant 1 multiplies the
/// first word of the first send of step 1 by 1.001 before sending. The steps are timed with the card drained at both
/// ends; nothing is timed while words are compared.
pub fn bench(args: Vec<String>) -> Result<(), String> {
    use crate::cuda::{checked, Cuda};
    let (mut ops_f, mut data_f, mut steps, mut transport, mut work, mut plant, mut out) = (None, None, 5usize, "file".to_string(), 0usize, false, None);
    let mut it = args.into_iter();
    while let Some(k) = it.next() {
        let v = it.next().ok_or_else(|| format!("peer-bench: {k} needs a value"))?;
        match k.as_str() {
            "--ops" => ops_f = Some(v), "--data" => data_f = Some(v), "--steps" => steps = v.parse().map_err(|_| "--steps")?,
            "--transport" => transport = v, "--work-bytes" => work = v.parse().map_err(|_| "--work-bytes")?,
            "--plant" => plant = v == "1", "--out" => out = Some(v), _ => return Err(format!("peer-bench: unknown {k}")),
        }
    }
    let p = crate::peer::require("peer-bench")?;
    // shm = AD_PEER_SHM=2 (stream-ordered); shmf = AD_PEER_SHM=1 (fenced: the base's drains around each copy)
    if transport == "shm" && !(on() && !fenced()) { return Err("peer-bench --transport shm needs AD_PEER_SHM=2".into()); }
    if transport == "shmf" && !fenced() { return Err("peer-bench --transport shmf needs AD_PEER_SHM=1".into()); }
    if transport != "shm" && transport != "shmf" && on() { return Err("peer-bench: AD_PEER_SHM is set but --transport is not shm or shmf".into()); }
    let ops: Vec<(bool, u64, usize)> = std::fs::read_to_string(ops_f.ok_or("--ops")?).map_err(|e| e.to_string())?.lines()
        .filter(|l| !l.trim().is_empty()).map(|l| { let w: Vec<&str> = l.split_whitespace().collect();
            (w[0] == "send", w[1].parse().unwrap(), w[2].parse().unwrap()) }).collect();
    let words: Vec<f32> = std::fs::read(data_f.ok_or("--data")?).map_err(|e| e.to_string())?.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let msg = |tag: usize, step: usize, ch: u64, bytes: usize| -> Vec<u8> {
        let k0 = (tag * 1_000_003 + step * 7919 + ch as usize * 104729) % words.len();
        (0..bytes / 4).flat_map(|j| words[(k0 + j) % words.len()].to_le_bytes()).collect()
    };
    unsafe {
        let cuda = Cuda::load()?;
        checked("cuInit", (cuda.init)(0))?;
        let mut device = 0; checked("cuDeviceGet", (cuda.device_get)(&mut device, 0))?;
        let mut ctx = std::ptr::null_mut(); checked("cuCtxCreate", (cuda.context_create)(&mut ctx, 0, device))?;
        let alloc = |b: usize| -> Result<u64, String> { let mut d = 0u64; checked("cuMemAlloc", (cuda.allocate)(&mut d, b.max(256)))?; Ok(d) };
        let bufs: Vec<u64> = ops.iter().map(|o| alloc(o.2)).collect::<Result<_, _>>()?;
        let tmp = alloc(ops.iter().map(|o| o.2).max().unwrap_or(256))?;
        let wbuf = if work > 0 { alloc(work)? } else { 0 };
        let shm = if on() { Some(get(&cuda)?) } else { None };
        let tr = if transport == "shmf" { "shm" } else { transport.as_str() };
        let null = std::ptr::null_mut();
        let sync = || checked("cuCtxSynchronize", (cuda.context_synchronize)());
        let (mut t_tl, mut t_ad, mut bad, mut checked_words) = (Vec::new(), Vec::new(), 0usize, 0usize);
        let peer_rank = 1 - p.rank;
        let first_send = ops.iter().position(|x| x.0);
        // one tangent pass and one cotangent sweep over all steps, as in a window
        if transport != "none" { p.begin_tl(); }
        for step in 1..=steps {
            for (i, o) in ops.iter().enumerate() {
                if o.0 {
                    let mut m = msg(p.rank, step, o.1, o.2);
                    if plant && step == 1 && first_send == Some(i) {
                        let w = f32::from_le_bytes(m[0..4].try_into().unwrap()) * 1.001; m[0..4].copy_from_slice(&w.to_le_bytes());
                    }
                    checked("cuMemcpyHtoD", (cuda.upload)(bufs[i], m.as_ptr().cast(), o.2))?;
                }
            }
            sync()?;
            let t = std::time::Instant::now();
            for (i, o) in ops.iter().enumerate() {
                if work > 0 { checked("cuMemsetD8Async", (cuda.clear)(wbuf, (step + i) as u8, work, null))?; }
                let key = crate::peer::key(step, o.1, "bench");
                match (tr, o.0) {
                    ("file", true) => { sync()?; let mut d = vec![0u8; o.2]; checked("cuMemcpyDtoH", (cuda.download)(d.as_mut_ptr().cast(), bufs[i], o.2))?; p.put_tl(&key, &d)?; },
                    ("file", false) => { let d = p.get_tl(&key, o.2)?; sync()?; checked("cuMemcpyHtoD", (cuda.upload)(bufs[i], d.as_ptr().cast(), o.2))?; sync()?; },
                    ("shm", true) => { fence(&cuda)?; shm.unwrap().send(b't', null, &key, bufs[i], o.2)?; fence(&cuda)?; },
                    ("shm", false) => { fence(&cuda)?; shm.unwrap().recv(b't', null, &key, bufs[i], o.2)?; fence(&cuda)?; },
                    _ => {},
                }
            }
            sync()?;
            t_tl.push(t.elapsed().as_secs_f64());
            if transport != "none" {
                for (i, o) in ops.iter().enumerate().filter(|(_, o)| !o.0) {
                    let mut d = vec![0u8; o.2]; checked("cuMemcpyDtoH", (cuda.download)(d.as_mut_ptr().cast(), bufs[i], o.2))?;
                    let want = msg(peer_rank, step, o.1, o.2);
                    bad += d.chunks_exact(4).zip(want.chunks_exact(4)).filter(|(a, b)| a != b).count(); checked_words += o.2 / 4;
                }
            }
        }
        if transport != "none" { p.begin_ad(); }
        for step in (1..=steps).rev() {
            for (i, o) in ops.iter().enumerate() {
                if !o.0 { let m = msg(p.rank + 10, step, o.1, o.2); checked("cuMemcpyHtoD", (cuda.upload)(bufs[i], m.as_ptr().cast(), o.2))?; }
            }
            sync()?;
            let t = std::time::Instant::now();
            for (i, o) in ops.iter().enumerate().rev() {
                if work > 0 { checked("cuMemsetD8Async", (cuda.clear)(wbuf, (step + i) as u8, work, null))?; }
                let key = crate::peer::key(step, o.1, "bench");
                match (tr, o.0) {
                    // the receiving rank of the forward copy sends its cotangent and zeroes it; the sending rank takes it
                    ("file", false) => { sync()?; let mut d = vec![0u8; o.2]; checked("cuMemcpyDtoH", (cuda.download)(d.as_mut_ptr().cast(), bufs[i], o.2))?; p.put_ad(&key, &d)?;
                                         checked("cuMemsetD8Async", (cuda.clear)(bufs[i], 0, o.2, null))?; },
                    ("file", true) => { let d = p.get_ad(&key, o.2)?; sync()?; checked("cuMemcpyHtoD", (cuda.upload)(tmp, d.as_ptr().cast(), o.2))?; sync()?;
                                        checked("cuMemcpyDtoDAsync", (cuda.copy)(bufs[i], tmp, o.2, null))?; },
                    ("shm", false) => { fence(&cuda)?; shm.unwrap().send(b'a', null, &key, bufs[i], o.2)?; fence(&cuda)?; checked("cuMemsetD8Async", (cuda.clear)(bufs[i], 0, o.2, null))?; },
                    ("shm", true) => { fence(&cuda)?; shm.unwrap().recv(b'a', null, &key, tmp, o.2)?; fence(&cuda)?; checked("cuMemcpyDtoDAsync", (cuda.copy)(bufs[i], tmp, o.2, null))?; },
                    _ => {},
                }
            }
            sync()?;
            t_ad.push(t.elapsed().as_secs_f64());
            if transport != "none" {
                for (i, o) in ops.iter().enumerate().filter(|(_, o)| o.0) {
                    let mut d = vec![0u8; o.2]; checked("cuMemcpyDtoH", (cuda.download)(d.as_mut_ptr().cast(), bufs[i], o.2))?;
                    let want = msg(peer_rank + 10, step, o.1, o.2);
                    bad += d.chunks_exact(4).zip(want.chunks_exact(4)).filter(|(a, b)| a != b).count(); checked_words += o.2 / 4;
                }
            }
        }
        let med = |v: &Vec<f64>| { let mut s = v.clone(); s.sort_by(|a, b| a.partial_cmp(b).unwrap()); s.get(s.len() / 2).copied().unwrap_or(0.0) };
        let doc = serde_json::json!({"rank":p.rank,"ranks":p.ranks,"transport":transport,"steps":steps,"ops_per_step":ops.len(),
            "sends_per_step":ops.iter().filter(|o| o.0).count(),"bytes_per_step":ops.iter().map(|o| o.2).sum::<usize>(),"work_bytes_per_copy":work,
            "planted":plant,"tangent_step_s":t_tl,"cotangent_step_s":t_ad,"tangent_median_s":med(&t_tl),"cotangent_median_s":med(&t_ad),
            "words_checked":checked_words,"words_differing":bad,"peer":p.stats_json()});
        println!("{doc}");
        if let Some(o) = out { std::fs::write(o, doc.to_string()).map_err(|e| e.to_string())?; }
        if bad > 0 { return Err(format!("peer-bench: {bad} received words differ from the words sent")); }
    }
    Ok(())
}
