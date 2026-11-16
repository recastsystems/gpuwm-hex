// SPDX-License-Identifier: Apache-2.0
//! multidevice-runner (adjoint-speed round 5): the TL+AD runner as one process per rank, one card per process.
//!
//! A 2-rank (or N-rank) recording of the decomposed forward is split by tools/multidevice/split_plan.py into one plan per
//! rank: each rank keeps its own allocations and commands, and every device copy that crosses a rank boundary (the halo
//! DMA between one rank's send packet and the other rank's receive packet, partition_samehost_transport_v841._receive)
//! becomes a `PeerSend` in the source rank's plan and a `PeerRecv` in the destination rank's plan, under one channel id.
//! Each process runs the ordinary single-device runner on its own card (CUDA_VISIBLE_DEVICES), so module loads, the
//! tape, the shadow arena, TEMPO caches and every native library stay per card: the tape is sharded per card by
//! construction.
//!
//! The exchange is through a mailbox directory (AD_PEER_DIR, on tmpfs): one file per message, written to a temporary
//! name and renamed, so a reader never sees a partial message. Message names:
//!   f.<key>          the forward bytes of one copy (written once; every later forward of the same base trajectory - the
//!                    seed passes, the reverse's restore + redo, its anchors and segments - reads the same file, so a
//!                    recompute never waits on the peer and the two ranks may segment their reverse differently);
//!   t<e>.<key>       the tangent words of one copy in tangent pass e;
//!   a<e>.<key>       the cotangent words of one copy in reverse sweep e (sent by the receiving rank, which zeroes its
//!                    own: the exact transpose of dst <- src is src += dst_bar, dst_bar = 0);
//!   r.<name>.<rank>  a rank's partial sums (the global dot is the rank-ordered sum, the same in every rank).
//! <key> = s<step>.c<channel>.k<scope>: the step, the copy's channel and a hash of the live RK scope (split, stage,
//! small step), which names the loop iteration, so both ranks name the same execution of the same copy.
//! Deadlock freedom: sends never block; a receive waits only for a message its peer produces at the same point of the
//! global command order (forward and tangent: before it; reverse: the transpose order), which both ranks reach.
use std::{path::PathBuf, sync::{atomic::{AtomicU64, Ordering}, Mutex, OnceLock}, time::{Duration, Instant}};

pub struct Peer {
    pub rank: usize,
    pub ranks: usize,
    dir: PathBuf,
    timeout: Duration,
    tl: AtomicU64,
    ad: AtomicU64,
    pub stats: Mutex<Stats>,
    seen_tl: Mutex<std::collections::HashSet<String>>,
    /// multicard: the mailbox listeners of the OTHER hosts (AD_PEER_SEND). Every message this rank writes is
    /// also sent to each of them, so ranks on different hosts exchange halos (one mailbox per host, see net_listen).
    remotes: Vec<Mutex<Remote>>,
}
pub struct Remote { addr: String, stream: Option<std::net::TcpStream> }
#[derive(Default, Clone, Debug)]
pub struct Stats { pub sent: u64, pub received: u64, pub bytes_out: u64, pub bytes_in: u64, pub wait_s: f64, pub io_s: f64, pub replayed_reads: u64,
    /// ranks: the receive wait by message kind [forward, tangent, cotangent, reduce]: seconds and messages; how many
    /// receives waited more than 5 ms, and the longest one (where a rank window's halo time goes; read-only counters).
    pub wait_kind_s: [f64; 4], pub recv_kind: [u64; 4], pub waits_over_5ms: u64, pub wait_max_s: f64 }

static PEER: OnceLock<Option<Peer>> = OnceLock::new();
static HOST_UNCHECKED: OnceLock<std::collections::HashSet<u64>> = OnceLock::new();
/// The plan's peer_host_unchecked list (set once at plan load; empty for every plan that is not a rank plan).
/// ranks: AD_PEER_TOLERATE_GUARDS=1 in a rank process. A rank-local mesh has rows the forward never solves (the
/// outer halo and the clamped out-of-rank references); a derivative entry's guard that fires on such a row then skips the
/// row and is counted in the receipt (peer_guards_tolerated) instead of refusing the step. Off: the refusal stands.
pub fn tolerate_guards() -> bool { get().is_some() && std::env::var("AD_PEER_TOLERATE_GUARDS").as_deref() == Ok("1") }
/// ranks: AD_PEER_TOLERATE_D2H=1 in a rank process: a checked read-back that differs from the proven value is logged
/// (peer_checked_read_differs), not refused. The owned allocations' step hashes still prove the forward.
pub fn tolerate_d2h() -> bool { get().is_some() && std::env::var("AD_PEER_TOLERATE_D2H").as_deref() == Ok("1") }
pub fn probe() -> bool { std::env::var("AD_PEER_PROBE").as_deref() == Ok("1") }
pub fn set_host_unchecked(ids: &[u64]) { let _ = HOST_UNCHECKED.set(ids.iter().copied().collect()); }
pub fn host_unchecked(id: u64) -> bool { get().is_some() && HOST_UNCHECKED.get().is_some_and(|s| s.contains(&id)) }

/// two-cards: a mailbox for the CPU test of peer_shm.rs (two ranks as threads of one test process).
#[cfg(test)]
pub(crate) fn make_for_test(rank: usize, ranks: usize, dir: PathBuf) -> Peer { std::fs::create_dir_all(&dir).unwrap(); make(rank, ranks, dir, Duration::from_secs(60)) }
fn make(rank: usize, ranks: usize, dir: PathBuf, timeout: Duration) -> Peer {
    let remotes = std::env::var("AD_PEER_SEND").ok().map(|v| v.split(',').filter(|a| !a.is_empty())
        .map(|a| Mutex::new(Remote { addr: a.to_string(), stream: None })).collect()).unwrap_or_default();
    Peer { rank, ranks, dir, timeout, tl: AtomicU64::new(0), ad: AtomicU64::new(0),
           stats: Mutex::new(Stats::default()), seen_tl: Mutex::new(Default::default()), remotes }
}

/// The process's rank configuration: AD_PEER_RANK, AD_PEER_RANKS, AD_PEER_DIR (all three, or none: the ordinary
/// one-process runner), AD_PEER_TIMEOUT_S (default 1800).
pub fn get() -> Option<&'static Peer> {
    PEER.get_or_init(|| {
        let rank = std::env::var("AD_PEER_RANK").ok()?.parse().ok()?;
        let ranks = std::env::var("AD_PEER_RANKS").ok()?.parse().ok()?;
        let dir = PathBuf::from(std::env::var("AD_PEER_DIR").ok()?);
        let timeout = std::env::var("AD_PEER_TIMEOUT_S").ok().and_then(|v| v.parse().ok()).unwrap_or(1800u64);
        std::fs::create_dir_all(&dir).ok()?;
        Some(make(rank, ranks, dir, Duration::from_secs(timeout)))
    }).as_ref()
}
/// A plan with peer commands needs the rank configuration (THE BREAKAGE THIS PREVENTS: a rank plan run by the
/// one-process runner would skip its halo exchanges and replay a wrong forward).
pub fn require(what: &str) -> Result<&'static Peer, String> {
    get().ok_or_else(|| format!("{what}: a rank plan's peer command needs AD_PEER_RANK, AD_PEER_RANKS and AD_PEER_DIR"))
}

fn launcher_pid() -> Option<u32> {
    static L: OnceLock<Option<u32>> = OnceLock::new();
    *L.get_or_init(|| std::env::var("AD_PEER_LAUNCHER").ok().and_then(|v| v.parse().ok()))
}
fn fnv(s: &str) -> u64 { let mut h = 0xcbf29ce484222325u64; for b in s.bytes() { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); } h }
/// The name of one execution of one copy (see the module text).
pub fn key(step: usize, channel: u64, scope: &str) -> String { format!("s{step}.c{channel}.k{:016x}", fnv(scope)) }

impl Peer {
    fn path(&self, name: &str) -> PathBuf { self.dir.join(name) }
    fn write(&self, name: &str, data: &[u8]) -> Result<(), String> {
        let t = Instant::now();
        let tmp = self.path(&format!(".{name}.tmp{}", self.rank));
        std::fs::write(&tmp, data).map_err(|e| format!("peer write {name}: {e}"))?;
        std::fs::rename(&tmp, self.path(name)).map_err(|e| format!("peer rename {name}: {e}"))?;
        for r in &self.remotes { send_frame(&mut r.lock().unwrap(), name, data, self.timeout)?; }
        let mut s = self.stats.lock().unwrap();
        s.sent += 1; s.bytes_out += data.len() as u64; s.io_s += t.elapsed().as_secs_f64();
        Ok(())
    }
    /// A forward message is written once: the base trajectory is deterministic, so a recompute's bytes are the same.
    pub fn put_forward(&self, key: &str, data: &[u8]) -> Result<(), String> {
        let name = format!("f.{key}");
        if self.path(&name).exists() { self.stats.lock().unwrap().replayed_reads += 1; return Ok(()); }
        self.write(&name, data)
    }
    pub fn has_forward(&self, key: &str) -> bool { self.path(&format!("f.{key}")).exists() }
    pub fn note_resend(&self) { self.stats.lock().unwrap().replayed_reads += 1; }
    pub fn get_forward(&self, key: &str, bytes: usize) -> Result<Vec<u8>, String> { self.read(&format!("f.{key}"), bytes) }
    /// Tangent pass e's message. A second send of one name in one pass is refused (THE BREAKAGE THIS PREVENTS: two
    /// executions of one copy under one scope would make the receiver read the wrong one).
    pub fn put_tl(&self, key: &str, data: &[u8]) -> Result<(), String> {
        let name = format!("t{}.{key}", self.tl.load(Ordering::SeqCst));
        if !self.seen_tl.lock().unwrap().insert(name.clone()) {
            return Err(format!("peer: {name} sent twice in one tangent pass (two executions of one copy share a scope key)"));
        }
        self.write(&name, data)
    }
    pub fn get_tl(&self, key: &str, bytes: usize) -> Result<Vec<u8>, String> { self.read(&format!("t{}.{key}", self.tl.load(Ordering::SeqCst)), bytes) }
    pub fn put_ad(&self, key: &str, data: &[u8]) -> Result<(), String> { self.write(&format!("a{}.{key}", self.ad.load(Ordering::SeqCst)), data) }
    pub fn get_ad(&self, key: &str, bytes: usize) -> Result<Vec<u8>, String> { self.read(&format!("a{}.{key}", self.ad.load(Ordering::SeqCst)), bytes) }
    /// Start the next tangent pass / reverse sweep (both ranks run the same passes in the same order).
    pub fn begin_tl(&self) -> u64 {
        self.seen_tl.lock().unwrap().clear();
        let e = self.tl.fetch_add(1, Ordering::SeqCst) + 1;
        self.shm_begin(b't', e); e
    }
    pub fn begin_ad(&self) -> u64 { let e = self.ad.fetch_add(1, Ordering::SeqCst) + 1; self.shm_begin(b'a', e); e }
    /// two-cards (AD_PEER_SHM=1 only): every rank meets the pass barrier of the shared slots (peer_shm.rs).
    /// A rank that cannot meet it stops the job (its peers stop at their next wait).
    fn shm_begin(&self, kind: u8, e: u64) {
        if let Err(x) = crate::peer_shm::begin(kind, e) { self.abort(&x); eprintln!("peer shm: {x}"); std::process::exit(70); }
    }
    pub fn dir(&self) -> &std::path::Path { &self.dir }
    pub fn remote_hosts(&self) -> usize { self.remotes.len() }
    fn read(&self, name: &str, bytes: usize) -> Result<Vec<u8>, String> {
        let t = Instant::now();
        let p = self.path(name);
        let mut nap = Duration::from_micros(20);
        loop {
            match std::fs::read(&p) {
                Ok(d) => {
                    if d.len() != bytes { return Err(format!("peer message {name}: {} bytes, the command moves {bytes}", d.len())); }
                    let mut s = self.stats.lock().unwrap();
                    let dt = t.elapsed().as_secs_f64();
                    s.received += 1; s.bytes_in += d.len() as u64; s.wait_s += dt;
                    let kind = match name.as_bytes().first() { Some(b'f') => 0, Some(b't') => 1, Some(b'a') => 2, _ => 3 };
                    s.wait_kind_s[kind] += dt; s.recv_kind[kind] += 1;
                    if dt > 0.005 { s.waits_over_5ms += 1; }
                    if dt > s.wait_max_s { s.wait_max_s = dt; }
                    return Ok(d);
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    if t.elapsed() > self.timeout {
                        return Err(format!("peer message {name}: none after {} s (rank {} of {})", self.timeout.as_secs(), self.rank, self.ranks));
                    }
                    if let Ok(x) = std::fs::read_to_string(self.dir.join("ABORT")) { return Err(format!("peer message {name}: a peer aborted: {x}")); }
                    // A rank started by `ad-binder ranks` on the launcher's host stops waiting once the launcher is gone
                    // (AD_PEER_LAUNCHER), instead of waiting out the whole timeout for a peer nobody will restart.
                    if let Some(p) = launcher_pid() { if !std::path::Path::new(&format!("/proc/{p}")).exists() {
                        return Err(format!("peer message {name}: the launcher (pid {p}) is gone"));
                    } }
                    std::thread::sleep(nap);
                    nap = (nap * 2).min(Duration::from_millis(2));
                }
                Err(e) => return Err(format!("peer read {name}: {e}")),
            }
        }
    }
    /// The rank-ordered sum of every rank's `vals` (the same value in every rank).
    pub fn reduce_sum(&self, name: &str, vals: &[f64]) -> Result<Vec<f64>, String> {
        let b: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.write(&format!("r.{name}.{}", self.rank), &b)?;
        let mut out = vec![0f64; vals.len()];
        for r in 0..self.ranks {
            let d = self.read(&format!("r.{name}.{r}"), b.len())?;
            for (o, c) in out.iter_mut().zip(d.chunks_exact(8)) { *o += f64::from_le_bytes(c.try_into().unwrap()); }
        }
        Ok(out)
    }
    /// Stop the peers' waits with a reason (a rank that fails writes it; the others refuse at their next receive).
    pub fn abort(&self, why: &str) {
        let msg = format!("rank {}: {why}", self.rank);
        let _ = std::fs::write(self.dir.join("ABORT"), &msg);
        for r in &self.remotes { if let Ok(mut r) = r.lock() { let _ = send_frame(&mut r, "ABORT", msg.as_bytes(), Duration::from_secs(5)); } }
    }
    pub fn stats_json(&self) -> serde_json::Value {
        let s = self.stats.lock().unwrap().clone();
        let v = serde_json::json!({"rank":self.rank,"ranks":self.ranks,"messages_sent":s.sent,"messages_received":s.received,
            "bytes_out":s.bytes_out,"bytes_in":s.bytes_in,"receive_wait_s":s.wait_s,"write_s":s.io_s,
            "receive_wait_by_kind_s":{"forward":s.wait_kind_s[0],"tangent":s.wait_kind_s[1],"cotangent":s.wait_kind_s[2],"reduce":s.wait_kind_s[3]},
            "received_by_kind":{"forward":s.recv_kind[0],"tangent":s.recv_kind[1],"cotangent":s.recv_kind[2],"reduce":s.recv_kind[3]},
            "receives_waiting_over_5ms":s.waits_over_5ms,"longest_receive_wait_s":s.wait_max_s,
            "forward_resends_skipped":s.replayed_reads,"tangent_passes":self.tl.load(Ordering::SeqCst),
            "reverse_sweeps":self.ad.load(Ordering::SeqCst),"remote_hosts":self.remotes.len()});
        // two-cards: the shared-slot counters, only when AD_PEER_SHM=1 (off: the receipt is the base one)
        if let Some(sh) = crate::peer_shm::started() { let x = *sh.stats.lock().unwrap();
            let mut v = v; v["transport"] = serde_json::json!(if crate::peer_shm::fenced() { "shm fenced (AD_PEER_SHM=1)" } else { "shm stream-ordered (AD_PEER_SHM=2)" });
            v["shm"] = serde_json::json!({"sends":x[0],"receives":x[1],"bytes_out":x[2],"bytes_in":x[3]}); return v; }
        v
    }
}

// multicard: the cross-host mailbox. Ranks on one host share one tmpfs mailbox (AD_PEER_DIR); each host runs
// one listener (ad-binder peer-listen) that writes every message it receives into its host's mailbox under the same
// name (temporary name + rename, as a local write), and every rank sends each message it writes to the listener of
// every other host (AD_PEER_SEND). Names are written once (module text), so no ordering between hosts is needed and a
// received message is never sent on. Frame: b"ADPM", u32 name length, name, u64 data length, data (little endian).
// The send is synchronous (returns after the bytes are in the socket), so a rank that exits has handed every message
// it wrote to the kernel. THE BREAKAGE THIS PREVENTS: a rank on a second host would wait forever on a halo written
// into the first host's tmpfs.
fn send_frame(r: &mut Remote, name: &str, data: &[u8], timeout: Duration) -> Result<(), String> {
    use std::io::Write;
    if r.stream.is_none() {
        let t = Instant::now();
        loop {
            match std::net::TcpStream::connect(&r.addr) {
                Ok(s) => { let _ = s.set_nodelay(true); r.stream = Some(s); break; }
                Err(_) if t.elapsed() < timeout => std::thread::sleep(Duration::from_millis(50)),
                Err(e) => return Err(format!("peer net connect {}: {e}", r.addr)),
            }
        }
    }
    let s = r.stream.as_mut().unwrap();
    let mut head = Vec::with_capacity(16 + name.len());
    head.extend_from_slice(b"ADPM"); head.extend_from_slice(&(name.len() as u32).to_le_bytes());
    head.extend_from_slice(name.as_bytes()); head.extend_from_slice(&(data.len() as u64).to_le_bytes());
    s.write_all(&head).and_then(|_| s.write_all(data)).and_then(|_| s.flush())
        .map_err(|e| format!("peer net send {name} to {}: {e}", r.addr))
}

/// ad-binder peer-ping --bytes B --rounds R [--out DIR]: the exchange alone, run as ranks under `ad-binder ranks` (no
/// card work): R ping-pong round trips of B bytes between rank 0 and every other rank, then R messages of B bytes from
/// every rank to every other rank at once, then one rank-ordered reduce. Every received word is checked. It measures
/// what a halo of B bytes costs over the mapping (same-host tmpfs or the TCP host mailbox) before a model leg uses it.
pub fn ping(args: Vec<String>) -> Result<(), String> {
    let (mut bytes, mut rounds, mut out) = (1usize << 20, 20usize, None);
    let mut it = args.into_iter();
    while let Some(k) = it.next() {
        let v = it.next().ok_or_else(|| format!("peer-ping: {k} needs a value"))?;
        match k.as_str() {
            "--bytes" => bytes = v.parse().map_err(|_| "peer-ping --bytes")?,
            "--rounds" => rounds = v.parse().map_err(|_| "peer-ping --rounds")?,
            "--out" => out = Some(PathBuf::from(v)),
            _ => return Err(format!("peer-ping: unknown {k}")),
        }
    }
    let p = require("peer-ping")?;
    let fill = |r: usize, i: usize| -> Vec<u8> { (0..bytes).map(|j| (j as u64).wrapping_mul(31).wrapping_add((r * 7919 + i * 104729) as u64) as u8).collect() };
    let mut rtt = Vec::new();
    for peer in 1..p.ranks {
        for i in 0..rounds {
            let t = Instant::now();
            if p.rank == 0 {
                p.put_forward(&format!("ping.{peer}.{i}"), &fill(0, i))?;
                if p.get_forward(&format!("pong.{peer}.{i}"), bytes)? != fill(peer, i) { return Err(format!("peer-ping: pong {i} from rank {peer} differs")); }
                rtt.push(t.elapsed().as_secs_f64());
            } else if p.rank == peer {
                if p.get_forward(&format!("ping.{peer}.{i}"), bytes)? != fill(0, i) { return Err(format!("peer-ping: ping {i} differs")); }
                p.put_forward(&format!("pong.{peer}.{i}"), &fill(peer, i))?;
            }
        }
    }
    let t = Instant::now();
    for i in 0..rounds { p.put_forward(&format!("bulk.{}.{i}", p.rank), &fill(p.rank, i))?; }
    for r in (0..p.ranks).filter(|r| *r != p.rank) {
        for i in 0..rounds { if p.get_forward(&format!("bulk.{r}.{i}"), bytes)? != fill(r, i) { return Err(format!("peer-ping: bulk {i} from rank {r} differs")); } }
    }
    let bulk_s = t.elapsed().as_secs_f64();
    let sum = p.reduce_sum("ping", &[p.rank as f64 + 0.5])?;
    let want: f64 = (0..p.ranks).map(|r| r as f64 + 0.5).sum();
    if sum[0] != want { return Err(format!("peer-ping: reduce {} != {want}", sum[0])); }
    rtt.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let moved = (bytes * rounds * (p.ranks - 1)) as f64;
    let doc = serde_json::json!({"rank":p.rank,"ranks":p.ranks,"bytes":bytes,"rounds":rounds,
        "rtt_median_s":rtt.get(rtt.len() / 2),"rtt_min_s":rtt.first(),"rtt_max_s":rtt.last(),
        "bulk_s":bulk_s,"bulk_in_mb_per_s":moved / bulk_s / 1e6,"reduce":sum[0],"words_checked":"all","peer":p.stats_json()});
    println!("{doc}");
    if let Some(o) = out { std::fs::create_dir_all(&o).map_err(|e| e.to_string())?; std::fs::write(o.join("ping.json"), doc.to_string()).map_err(|e| e.to_string())?; }
    Ok(())
}

/// ad-binder peer-listen --dir MAILBOX --addr HOST:PORT: one host's mailbox listener, for the whole job (the launcher
/// stops it by PID). Prints "listening <addr>" once bound.
pub fn net_listen(args: Vec<String>) -> Result<(), String> {
    use std::io::{Read, Write};
    let (mut dir, mut addr, mut watch, mut idle) = (None, None, None, None);
    let mut it = args.into_iter();
    while let Some(k) = it.next() {
        let v = it.next().ok_or_else(|| format!("peer-listen: {k} needs a value"))?;
        match k.as_str() {
            "--dir" => dir = Some(PathBuf::from(v)), "--addr" => addr = Some(v),
            "--watch-pid" => watch = Some(v.parse::<u32>().map_err(|_| "peer-listen --watch-pid")?),
            "--idle-exit-s" => idle = Some(v.parse::<u64>().map_err(|_| "peer-listen --idle-exit-s")?),
            _ => return Err(format!("peer-listen: unknown {k}")),
        }
    }
    let dir = dir.ok_or("peer-listen needs --dir")?;
    let addr = addr.ok_or("peer-listen needs --addr")?;
    std::fs::create_dir_all(&dir).map_err(|e| format!("peer-listen {}: {e}", dir.display()))?;
    let l = std::net::TcpListener::bind(&addr).map_err(|e| format!("peer-listen bind {addr}: {e}"))?;
    println!("listening {}", l.local_addr().map(|a| a.to_string()).unwrap_or(addr));
    let _ = std::io::stdout().flush();
    let seq = std::sync::Arc::new(AtomicU64::new(0));
    // Lifetime: a listener ends with its job even when the launcher cannot stop it. --watch-pid: exit when that process
    // (the launcher, same host) is gone; --idle-exit-s: exit after that long with no open sender connection (remote
    // hosts). Either way the mailbox goes too. THE BREAKAGE THIS PREVENTS: a launcher killed by timeout left listeners
    // and mailboxes on both hosts (seen 2026-11-09 on a test machine and a test machine).
    let open = std::sync::Arc::new(AtomicU64::new(0));
    let last = std::sync::Arc::new(Mutex::new(Instant::now()));
    {
        let (open, last, dir) = (open.clone(), last.clone(), dir.clone());
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_secs(1));
            let gone = watch.is_some_and(|p| !std::path::Path::new(&format!("/proc/{p}")).exists());
            let idle_out = idle.is_some_and(|s| open.load(Ordering::SeqCst) == 0 && last.lock().unwrap().elapsed() > Duration::from_secs(s));
            if gone || idle_out {
                let _ = std::fs::remove_dir_all(&dir);
                eprintln!("peer-listen: exit ({})", if gone { "launcher gone" } else { "idle" });
                std::process::exit(0);
            }
        });
    }
    for c in l.incoming() {
        let mut c = c.map_err(|e| format!("peer-listen accept: {e}"))?;
        let (dir, seq, open, last) = (dir.clone(), seq.clone(), open.clone(), last.clone());
        open.fetch_add(1, Ordering::SeqCst);
        std::thread::spawn(move || {
            let r = (|| -> Result<(), String> {
                loop {
                    let mut h = [0u8; 8];
                    if c.read_exact(&mut h).is_err() { return Ok(()); } // the sender closed
                    if &h[..4] != b"ADPM" { return Err("bad frame magic".into()); }
                    let n = u32::from_le_bytes(h[4..8].try_into().unwrap()) as usize;
                    let mut name = vec![0u8; n];
                    c.read_exact(&mut name).map_err(|e| e.to_string())?;
                    let name = String::from_utf8(name).map_err(|e| e.to_string())?;
                    if name.contains('/') || name.starts_with('.') { return Err(format!("bad name {name}")); }
                    let mut l8 = [0u8; 8];
                    c.read_exact(&mut l8).map_err(|e| e.to_string())?;
                    let mut data = vec![0u8; u64::from_le_bytes(l8) as usize];
                    c.read_exact(&mut data).map_err(|e| e.to_string())?;
                    let tmp = dir.join(format!(".{name}.net{}", seq.fetch_add(1, Ordering::SeqCst)));
                    std::fs::write(&tmp, &data).map_err(|e| format!("write {name}: {e}"))?;
                    std::fs::rename(&tmp, dir.join(&name)).map_err(|e| format!("rename {name}: {e}"))?;
                }
            })();
            if let Err(e) = r { eprintln!("peer-listen: {e}"); }
            *last.lock().unwrap() = Instant::now();
            open.fetch_sub(1, Ordering::SeqCst);
        });
    }
    Ok(())
}

#[cfg(test)]
mod peer_tests {
    use super::*;
    fn mk(dir: &std::path::Path, rank: usize) -> Peer { make(rank, 2, dir.to_owned(), Duration::from_secs(5)) }
    #[test]
    fn forward_tangent_reverse_messages_pair_across_ranks_and_reduce_in_rank_order() {
        let dir = std::env::temp_dir().join(format!("peer-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (a, b) = (mk(&dir, 0), mk(&dir, 1));
        let k = key(3, 17, "Some(1).None.Some(2)");
        assert_ne!(k, key(3, 17, "Some(1).None.Some(3)"));
        a.put_forward(&k, &[1, 2, 3, 4]).unwrap();
        a.put_forward(&k, &[9, 9, 9, 9]).unwrap(); // a recompute: the first bytes stay
        assert_eq!(b.get_forward(&k, 4).unwrap(), vec![1, 2, 3, 4]);
        assert!(b.get_forward(&k, 8).is_err());
        a.begin_tl(); b.begin_tl();
        a.put_tl(&k, &[5; 8]).unwrap();
        assert!(a.put_tl(&k, &[5; 8]).is_err(), "a second send in one pass is refused");
        assert_eq!(b.get_tl(&k, 8).unwrap(), vec![5; 8]);
        a.begin_tl(); b.begin_tl();
        let d2 = dir.clone();
        let k2 = k.clone();
        let h = std::thread::spawn(move || { std::thread::sleep(Duration::from_millis(30)); mk(&d2, 0).write(&format!("t2.{k2}"), &[7; 8]).unwrap(); });
        assert_eq!(b.get_tl(&k, 8).unwrap(), vec![7; 8], "a receive waits for its own pass's message");
        h.join().unwrap();
        b.begin_ad(); a.begin_ad();
        b.put_ad(&k, &[3; 4]).unwrap();
        assert_eq!(a.get_ad(&k, 4).unwrap(), vec![3; 4]);
        let d3 = dir.clone();
        let ha = std::thread::spawn(move || mk(&d3, 0).reduce_sum("dot1", &[1.0, 2.0]).unwrap());
        let rb = b.reduce_sum("dot1", &[0.5, 0.25]).unwrap();
        assert_eq!(ha.join().unwrap(), rb);
        assert_eq!(rb, vec![1.5, 2.25]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// multicard: two hosts = two mailboxes joined only by the TCP listener; a message written on one host is
    /// read on the other, and the reduce is the same rank-ordered sum.
    #[test]
    fn messages_cross_hosts_through_the_listener() {
        let base = std::env::temp_dir().join(format!("peer-net-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let (da, db) = (base.join("a"), base.join("b"));
        std::fs::create_dir_all(&da).unwrap(); std::fs::create_dir_all(&db).unwrap();
        let free = |d: std::path::PathBuf| { let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap(); let a = l.local_addr().unwrap();
            drop(l); let s = a.to_string(); let s2 = s.clone();
            std::thread::spawn(move || net_listen(vec!["--dir".into(), d.display().to_string(), "--addr".into(), s2]).unwrap()); s };
        let (la, lb) = (free(da.clone()), free(db.clone()));
        let mut a = make(0, 2, da.clone(), Duration::from_secs(5));
        a.remotes = vec![Mutex::new(Remote { addr: lb, stream: None })];
        let mut b = make(1, 2, db.clone(), Duration::from_secs(5));
        b.remotes = vec![Mutex::new(Remote { addr: la, stream: None })];
        let k = key(1, 5, "Some(0)");
        a.put_forward(&k, &[1, 2, 3, 4]).unwrap();
        assert_eq!(b.get_forward(&k, 4).unwrap(), vec![1, 2, 3, 4]);
        b.begin_ad(); a.begin_ad();
        b.put_ad(&k, &[6; 16]).unwrap();
        assert_eq!(a.get_ad(&k, 16).unwrap(), vec![6; 16]);
        let ha = std::thread::spawn(move || a.reduce_sum("dot", &[1.0, 2.0]).unwrap());
        let rb = b.reduce_sum("dot", &[0.5, 0.25]).unwrap();
        assert_eq!(ha.join().unwrap(), rb);
        assert_eq!(rb, vec![1.5, 2.25]);
        b.abort("test");
        let t = Instant::now();
        while !da.join("ABORT").exists() { assert!(t.elapsed() < Duration::from_secs(5)); std::thread::sleep(Duration::from_millis(5)); }
        let _ = std::fs::remove_dir_all(&base);
    }
}
