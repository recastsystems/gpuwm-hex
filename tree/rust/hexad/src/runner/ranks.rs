// SPDX-License-Identifier: Apache-2.0
//! multicard: N ranks over M cards (and H hosts), chosen at run time.
//!
//!   ad-binder ranks --ranks N --cards LIST --pkg PREFIX [--port P] [--advertise ADDR] -- <window|replay> ARGS...
//!
//! Rank k runs the ordinary one-device runner (`ad-binder <sub> ARGS`) in PREFIX-r<k> (its rank plan, written by
//! tools/multidevice/split_plan.py) on the card LIST assigns it, as its own process with its own CUDA context, and the
//! halo exchange between ranks goes through the peer mailbox (peer.rs). LIST is comma-separated, each item `CARD` (this
//! host) or `HOST:CARD`:
//!   one item         every rank on that card (correctness runs, or ranks that fit one card together);
//!   N items          rank k on item k;
//!   M items, M | N   ranks in blocks of N/M per card (neighbouring ranks share a card).
//! Ranks on one host share a tmpfs mailbox; with more than one host every host runs a listener (`peer-listen`) and every
//! rank sends what it writes to the other hosts' listeners (AD_PEER_SEND). A HOST of `localhost`, `127.0.0.1` or this
//! host's name runs on this machine but is still its own mailbox host, so `--cards localhost:0,127.0.0.1:0` puts two
//! ranks on one card joined only by TCP, the exact exchange two hosts use. Remote hosts are reached with ssh and must
//! have the binder and PREFIX-r<k> at the same paths. `--out X`, `--bstore X` and AD_RESULT_DUMP get a per-rank `/r<k>`.
//! The launcher writes X/ranks.json (mapping, return codes, wall, every rank's dot row and peer stats) when --out is set.
use std::{collections::BTreeMap, path::PathBuf, process::{Child, Command, Stdio}, time::{Duration, Instant}};
use serde_json::{json, Value};

#[derive(Clone, Debug, PartialEq)]
pub struct Slot { pub host: Option<String>, pub card: u32 }

/// Rank -> (host, card). THE BREAKAGE THIS PREVENTS: a rank count that does not divide over the cards would put an
/// uneven number of ranks (and of tape) on one card without anyone asking for it.
pub fn map_ranks(ranks: usize, list: &str) -> Result<Vec<Slot>, String> {
    if ranks == 0 { return Err("ranks: --ranks must be at least 1".into()); }
    let mut cards = Vec::new();
    for item in list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (host, card) = match item.rsplit_once(':') { Some((h, c)) => (Some(h.to_string()), c), None => (None, item) };
        let card = card.parse::<u32>().map_err(|_| format!("ranks: card `{item}` is not CARD or HOST:CARD"))?;
        cards.push(Slot { host, card });
    }
    let m = cards.len();
    if m == 0 { return Err("ranks: --cards is empty".into()); }
    if m != 1 && m != ranks && ranks % m != 0 {
        return Err(format!("ranks: {ranks} ranks do not divide over {m} cards (give 1, {ranks}, or a divisor of {ranks} cards)"));
    }
    Ok((0..ranks).map(|k| if m == 1 { cards[0].clone() } else if m == ranks { cards[k].clone() } else { cards[k / (ranks / m)].clone() }).collect())
}

fn is_this_machine(h: &str) -> bool {
    if matches!(h, "localhost" | "127.0.0.1" | "::1") { return true; }
    let me = Command::new("hostname").output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
    !me.is_empty() && me == h
}
/// The address ssh itself reaches HOST at (`ssh -G`), so a listener is dialled where ssh started it. THE BREAKAGE THIS
/// PREVENTS: a test machine /etc/hosts maps a test machine to a stale <private address> while ssh's config reaches <private address>,
/// so rank 0 dialled a dead address forever.
fn ssh_hostname(h: &str) -> String {
    Command::new("ssh").args(["-G", h]).output().ok().and_then(|o| String::from_utf8_lossy(&o.stdout).lines()
        .find_map(|l| l.strip_prefix("hostname ").map(|x| x.trim().to_string()))).unwrap_or_else(|| h.to_string())
}
fn sh_quote(s: &str) -> String { format!("'{}'", s.replace('\'', "'\\''")) }
fn rank_args(args: &[String], k: usize) -> Vec<String> {
    let mut out = Vec::with_capacity(args.len());
    let mut it = args.iter();
    while let Some(a) = it.next() {
        out.push(a.replace("{rank}", &k.to_string()));
        if a == "--out" || a == "--bstore" { if let Some(v) = it.next() { out.push(format!("{v}/r{k}")); } }
    }
    out
}

struct Host { name: Option<String>, local: bool, mailbox: String, listen_port: u16, connect: String, listener: Option<(Child, Option<String>)> }

pub fn run(args: Vec<String>) -> Result<(), String> {
    let split = args.iter().position(|a| a == "--").ok_or("usage: ad-binder ranks --ranks N --cards LIST --pkg PREFIX [--port P] [--advertise ADDR] -- <window|replay> ARGS...")?;
    let (opts, sub) = (&args[..split], &args[split + 1..]);
    if sub.is_empty() { return Err("ranks: nothing after --".into()); }
    let mut o = BTreeMap::new();
    let mut it = opts.iter();
    while let Some(k) = it.next() {
        if !matches!(k.as_str(), "--ranks" | "--cards" | "--pkg" | "--port" | "--advertise" | "--bin") { return Err(format!("ranks: unknown option {k}")); }
        o.insert(k.clone(), it.next().ok_or_else(|| format!("ranks: {k} needs a value"))?.clone());
    }
    let n: usize = o.get("--ranks").ok_or("ranks: --ranks N")?.parse().map_err(|_| "ranks: --ranks N")?;
    let slots = map_ranks(n, o.get("--cards").ok_or("ranks: --cards LIST")?)?;
    let prefix = o.get("--pkg").ok_or("ranks: --pkg PREFIX (rank k runs in PREFIX-r<k>)")?.clone();
    let port: u16 = o.get("--port").map(|p| p.parse().map_err(|_| "ranks: --port")).transpose()?.unwrap_or(41000 + (std::process::id() % 2000) as u16 * 4);
    // The binder path every host runs: --bin, else the path this was invoked by (not current_exe, which resolves
    // symlinks to a path the other host may not have; THE BREAKAGE THIS PREVENTS: a test machine ~/lanes is a symlink into
    // $HOME/weather/..., which a test machine does not have, so the remote listener refused a binder it had).
    let argv0 = std::env::args().next().unwrap_or_default();
    let exe = match o.get("--bin") { Some(b) => b.clone(), None if argv0.starts_with('/') => argv0,
        None => std::env::current_exe().map_err(|e| e.to_string())?.display().to_string() };
    let out = sub.windows(2).find(|w| w[0] == "--out").map(|w| PathBuf::from(&w[1]));
    if let Some(o) = &out { std::fs::create_dir_all(o).map_err(|e| format!("ranks: {}: {e}", o.display()))?; }
    // hosts in first-appearance order; each gets its own mailbox
    let mut hosts: Vec<Host> = Vec::new();
    let mut host_of = Vec::with_capacity(n);
    let tag = format!("ad-ranks-{}", std::process::id());
    let advertise = o.get("--advertise").cloned().unwrap_or_else(|| Command::new("hostname").arg("-I").output()
        .map(|x| String::from_utf8_lossy(&x.stdout).split_whitespace().next().unwrap_or("127.0.0.1").to_string()).unwrap_or("127.0.0.1".into()));
    for s in &slots {
        let i = match hosts.iter().position(|h| h.name == s.host) {
            Some(i) => i,
            None => {
                let i = hosts.len();
                let local = s.host.as_deref().map_or(true, is_this_machine);
                let lp = port + i as u16;
                let connect = match (&s.host, local) { (_, true) => format!("127.0.0.1:{lp}"), (Some(h), false) => format!("{}:{lp}", ssh_hostname(h)), _ => unreachable!() };
                hosts.push(Host { name: s.host.clone(), local, mailbox: format!("/dev/shm/{tag}-h{i}"), listen_port: lp, connect, listener: None });
                i
            }
        };
        host_of.push(i);
    }
    let any_remote = hosts.iter().any(|h| !h.local);
    if any_remote { for h in hosts.iter_mut().filter(|h| h.local) { h.connect = format!("{advertise}:{}", h.listen_port); } }
    let multi = hosts.len() > 1;
    let mut kids: Vec<Child> = Vec::with_capacity(n);
    let t0 = Instant::now();
    // Everything that starts processes runs inside `launched`; the cleanup below always runs after it, so an early
    // error never leaves a listener or a mailbox behind (THE BREAKAGE THIS PREVENTS: a refused remote listener left the
    // local listener running and both mailboxes on tmpfs).
    let launched = (|| -> Result<Vec<Option<i32>>, String> {
    // listeners first, so no message is sent before its host can take it
    for h in hosts.iter_mut() {
        if h.local { std::fs::create_dir_all(&h.mailbox).map_err(|e| format!("ranks: {}: {e}", h.mailbox))?; }
        if !multi { continue; }
        let bind = format!("0.0.0.0:{}", h.listen_port);
        if h.local {
            let mut c = Command::new(&exe).args(["peer-listen", "--dir", &h.mailbox, "--addr", &bind, "--watch-pid", &std::process::id().to_string()]).stdout(Stdio::piped()).stderr(Stdio::inherit())
                .spawn().map_err(|e| format!("ranks: peer-listen: {e}"))?;
            let mut line = String::new();
            use std::io::BufRead;
            std::io::BufReader::new(c.stdout.take().unwrap()).read_line(&mut line).map_err(|e| e.to_string())?;
            if !line.starts_with("listening") { return Err(format!("ranks: listener for host {} did not start: {line}", h.listen_port)); }
            h.listener = Some((c, None));
        } else {
            let host = h.name.clone().unwrap();
            let remote = format!("test -x {exe} || {{ echo 'DEFECT {exe} missing on {host}'; exit 3; }}; mkdir -p {mb} || exit 3; setsid nohup {exe} peer-listen --dir {mb} --addr {bind} --idle-exit-s 1800 < /dev/null > {mb}.listen.log 2>&1 & echo pid $!",
                mb = sh_quote(&h.mailbox), exe = sh_quote(&exe), host = host, bind = bind);
            // the listener runs detached (setsid, stdin from /dev/null) so this ssh returns once it is in the background
            // (THE BREAKAGE THIS PREVENTS: the launcher hung in ssh while the background listener held the session).
            let r = Command::new("ssh").args(["-o", "BatchMode=yes", &host, &remote]).stdin(Stdio::null()).output().map_err(|e| format!("ranks: ssh {host}: {e}"))?;
            let txt = String::from_utf8_lossy(&r.stdout).to_string();
            let pid = txt.split_whitespace().skip_while(|w| *w != "pid").nth(1).map(str::to_string)
                .ok_or_else(|| format!("ranks: remote listener on {host} did not start: {txt} {}", String::from_utf8_lossy(&r.stderr)))?;
            let dummy = Command::new("true").spawn().map_err(|e| e.to_string())?;
            h.listener = Some((dummy, Some(pid)));
        }
    }
    // ranks (R3): every rank process sizes its host tiers from MemAvailable at the same moment, so N local ranks
    // together plan for N times the machine (two 120 km ranks took 145 GiB; eight 60 km ranks would ask about 580 GiB).
    // Unless the caller gave one, each local rank gets --host-budget-gb = 0.8 x MemAvailable / local ranks.
    let local_ranks = (0..n).filter(|k| hosts[host_of[*k]].local).count().max(1);
    let avail_gib = std::fs::read_to_string("/proc/meminfo").ok().and_then(|m| m.lines().find(|l| l.starts_with("MemAvailable"))
        .and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())).unwrap_or(0.0) / 1048576.0;
    let host_budget = (sub.first().map(String::as_str) == Some("window") && !sub.iter().any(|a| a == "--host-budget-gb") && avail_gib > 0.0)
        .then(|| (0.8 * avail_gib / local_ranks as f64).floor().max(8.0));
    let pass: Vec<(String, String)> = std::env::vars().filter(|(k, _)| k.starts_with("AD_") || k.starts_with("CUDA_MODULE") || k == "BINDER_BOUNDARY_PAYLOADS").collect();
    for k in 0..n {
        let h = &hosts[host_of[k]];
        let send: Vec<&str> = hosts.iter().enumerate().filter(|(i, _)| *i != host_of[k]).map(|(_, x)| x.connect.as_str()).collect();
        let mut env: Vec<(String, String)> = pass.iter().filter(|(k, _)| !k.starts_with("AD_PEER_")).cloned().collect();
        if let Some(e) = env.iter_mut().find(|(k, _)| k == "AD_RESULT_DUMP") { e.1 = format!("{}/r{k}", e.1); }
        for e in env.iter_mut() { if e.1.contains("{rank}") { e.1 = e.1.replace("{rank}", &k.to_string()); } }
        env.extend([("CUDA_DEVICE_ORDER".into(), "PCI_BUS_ID".into()), ("CUDA_VISIBLE_DEVICES".into(), slots[k].card.to_string()),
                    ("AD_PEER_RANK".into(), k.to_string()), ("AD_PEER_RANKS".into(), n.to_string()),
                    ("AD_PEER_DIR".into(), h.mailbox.clone()), ("AD_PEER_SEND".into(), send.join(","))]);
        if h.local { env.push(("AD_PEER_LAUNCHER".into(), std::process::id().to_string())); }
        let mut argv = rank_args(sub, k);
        if let (Some(b), true) = (host_budget, h.local) { argv.extend(["--host-budget-gb".to_string(), format!("{b}")]); }
        let cwd = format!("{prefix}-r{k}");
        let (so, se) = match &out {
            Some(o) => (Stdio::from(std::fs::File::create(o.join(format!("r{k}.stdout"))).map_err(|e| e.to_string())?),
                        Stdio::from(std::fs::File::create(o.join(format!("r{k}.stderr"))).map_err(|e| e.to_string())?)),
            None => (Stdio::inherit(), Stdio::inherit()) };
        let child = if h.local {
            Command::new(&exe).args(&argv).current_dir(&cwd).envs(env.iter().cloned()).stdout(so).stderr(se).spawn()
        } else {
            let envs: Vec<String> = env.iter().map(|(k, v)| format!("{k}={}", sh_quote(v))).collect();
            let qa: Vec<String> = argv.iter().map(|a| sh_quote(a)).collect();
            let remote = format!("cd {} && exec env {} {} {}", sh_quote(&cwd), envs.join(" "), sh_quote(&exe), qa.join(" "));
            Command::new("ssh").args(["-o", "BatchMode=yes", h.name.as_deref().unwrap(), &remote]).stdin(Stdio::null()).stdout(so).stderr(se).spawn()
        };
        match child {
            Ok(c) => kids.push(c),
            Err(e) => { for c in kids.iter_mut() { let _ = c.kill(); let _ = c.wait(); } return Err(format!("ranks: rank {k}: {e}")); }
        }
    }
    // wait; the first rank to fail stops the others through every mailbox's ABORT
    let mut rc: Vec<Option<i32>> = vec![None; n];
    let mut aborted = false;
    while rc.iter().any(Option::is_none) {
        for k in 0..n {
            if rc[k].is_some() { continue; }
            if let Some(s) = kids[k].try_wait().map_err(|e| e.to_string())? {
                rc[k] = Some(s.code().unwrap_or(-1));
                if rc[k] != Some(0) && !aborted {
                    aborted = true;
                    for h in &hosts {
                        let msg = format!("launcher: rank {k} exited {:?}", rc[k]);
                        if h.local { let _ = std::fs::write(format!("{}/ABORT", h.mailbox), &msg); }
                        else { let _ = Command::new("ssh").args(["-o", "BatchMode=yes", h.name.as_deref().unwrap(), &format!("echo {} > {}/ABORT", sh_quote(&msg), sh_quote(&h.mailbox))]).status(); }
                    }
                }
            }
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    Ok(rc)
    })();
    let wall = t0.elapsed().as_secs_f64();
    let mut mailbox_bytes = 0u64;
    for h in hosts.iter_mut() {
        if let Some((mut c, pid)) = h.listener.take() {
            match pid {
                None => { let _ = c.kill(); let _ = c.wait(); }
                Some(p) => { let _ = c.wait(); let _ = Command::new("ssh").args(["-o", "BatchMode=yes", h.name.as_deref().unwrap(), &format!("kill {p}; rm -rf {mb} {mb}.listen.log", mb = sh_quote(&h.mailbox))]).status(); }
            }
        }
        if h.local {
            if let Ok(rd) = std::fs::read_dir(&h.mailbox) { for e in rd.flatten() { mailbox_bytes += e.metadata().map(|m| m.len()).unwrap_or(0); } }
            let _ = std::fs::remove_dir_all(&h.mailbox);
        }
    }
    let rc = launched?;
    let mut rows = Vec::new();
    for k in 0..n {
        let w: Value = out.as_ref().and_then(|o| std::fs::read(o.join(format!("r{k}/window.json"))).ok()).and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(Value::Null);
        let dot = w["dots"].get(0).map(|d| json!({"rel_aligned":d["rel_aligned"],"y_Mx":d["y_Mx"],"x_MTy":d["x_MTy"],"pass":d["pass"],
            "result_sha256":d["reverse"][0]["result_sha256"]})).unwrap_or(Value::Null);
        rows.push(json!({"rank":k,"host":slots[k].host,"card":slots[k].card,"mailbox_host":host_of[k],"rc":rc[k],"dot":dot,"peer":w["peer"],"wall_s":w["wall_s"]}));
    }
    let receipt = json!({"schema":"ad.ranks/v1","ranks":n,"cards":o.get("--cards"),"pkg":prefix,"hosts":hosts.iter().map(|h| json!({"host":h.name,"local":h.local,"listener":h.connect})).collect::<Vec<_>>(),
        "transport":if multi { "tcp between host mailboxes" } else { "one tmpfs mailbox" },"argv":sub,"wall_s":wall,"local_mailbox_bytes":mailbox_bytes,"rank_rows":rows});
    if let Some(o) = &out { std::fs::write(o.join("ranks.json"), serde_json::to_vec_pretty(&receipt).unwrap()).map_err(|e| e.to_string())?; }
    eprintln!("{}", json!({"event":"ranks_done","wall_s":wall,"rc":rc}));
    if rc.iter().all(|r| *r == Some(0)) { Ok(()) } else { Err(format!("ranks: return codes {rc:?}")) }
}

#[cfg(test)]
mod ranks_tests {
    use super::*;
    #[test]
    fn ranks_map_onto_cards_and_hosts() {
        let l = |h: Option<&str>, c| Slot { host: h.map(str::to_string), card: c };
        assert_eq!(map_ranks(2, "0").unwrap(), vec![l(None, 0), l(None, 0)]);
        assert_eq!(map_ranks(2, "0,1").unwrap(), vec![l(None, 0), l(None, 1)]);
        assert_eq!(map_ranks(4, "0,1").unwrap(), vec![l(None, 0), l(None, 0), l(None, 1), l(None, 1)]);
        assert_eq!(map_ranks(2, "a test machine:0,a test machine:0").unwrap(), vec![l(Some("a test machine"), 0), l(Some("a test machine"), 0)]);
        assert!(map_ranks(3, "0,1").is_err());
        assert!(map_ranks(2, "x").is_err());
        assert_eq!(rank_args(&["window".into(), "--out".into(), "/o".into(), "--bstore".into(), "/b".into(), "--steps".into(), "2".into()], 1),
                   vec!["window", "--out", "/o/r1", "--bstore", "/b/r1", "--steps", "2"]);
    }
}
