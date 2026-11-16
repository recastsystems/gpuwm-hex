//! rerun-tape (AD_TAPE_PREDICT=1, off unless asked): the share decisions of a taped re-run, taken from the
//! same step of the gradient before (a book kept in a file) and checked once at the chunk end.
//!
//! THE WASTE THIS REMOVES: a taped re-run decides, launch by launch, whether each pre-launch span and each post span is
//! shared with an equal earlier copy or copied. Each decision waits on a blocking readback of a device compare (about
//! 1,550 a 3 km step: cluster and post collects, block-compare reads), so the card drains after every launch and the
//! host plans the next launch with the card idle. The re-run is 0.87 to 0.90 s a step against a forward step of 0.30 to
//! 0.36 s (BACKSTEP.md, fin-first and fin-later).
//!
//! WHY THE SAME STEP OF THE GRADIENT BEFORE, NOT THE STEP TAPED BEFORE: in fin-first and fin-later (3 km, 59 steps, the
//! second with its start moved one fit step) no two consecutive steps of one gradient had the same share counters (0 of
//! 58; even the command count moves, 3,676 to 3,839), while the same step of the two gradients had the same commands,
//! cluster checks, shared spans, shared bytes, block checks and block spans at all 59 steps. Only the block-compare
//! flags moved between the gradients (equal bytes differ at every step, a median of 157 and at most 800 64-KiB blocks
//! of about 450,000). So the cluster and post decisions are predicted from the gradient before, and the block flags are
//! read for real unless AD_TAPE_PREDICT_SITES names `pairs`.
//!
//! HOW: the device compares are queued exactly as before, their result words are copied on the device into a log in
//! stream order, and the host takes the decisions from the book at once. At the chunk end the log is read back once and
//! every predicted decision is compared with the real one. All equal: every host decision was the real one, so every
//! device command (takes, copies, recon tables, launches) is the one the unchanged path issues, in the same order, and
//! the tape holds the same words at the same offsets. One differs: the step is restored and taped again with every
//! decision read for real (a miss costs one re-run more, never a word).
//!
//! This file is pure host logic (no CUDA), so it is tested on the CPU on its own: `rustc --test tape_predict.rs`.

use std::collections::HashMap;

/// Where a decision is read: the pre-launch cluster checks, the pre-launch block compares, the post-launch checks.
pub const SITE_PRE_WORDS: u8 = 0;
pub const SITE_PRE_PAIRS: u8 = 1;
pub const SITE_POST_WORDS: u8 = 2;
/// the rolling-copy block compares of a MYNN-EDMF launch in the taped re-run (schemes.rs prepare_impl), one list a launch.
pub const SITE_ROLL_PAIRS: u8 = 3;

/// (step, FNV-1a of the kernel name, occurrence of that kernel in the step, site). A key the book has never seen is
/// read for real, not missed.
pub type Key = (u32, u64, u32, u8);

pub fn name_hash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() { h ^= *b as u64; h = h.wrapping_mul(0x100000001b3); }
    h
}

/// The sites named in AD_TAPE_PREDICT_SITES (comma list of pre, pairs, post), as a bit mask; default pre and post.
pub fn sites_mask(spec: Option<&str>) -> u8 {
    match spec {
        None => (1 << SITE_PRE_WORDS) | (1 << SITE_POST_WORDS) | (1 << SITE_ROLL_PAIRS),
        Some(s) => s.split(',').map(str::trim).fold(0u8, |m, x| m | match x { "pre" => 1 << SITE_PRE_WORDS, "pairs" => 1 << SITE_PRE_PAIRS, "post" => 1 << SITE_POST_WORDS, "roll" => 1 << SITE_ROLL_PAIRS, _ => 0 }),
    }
}

/// One predicted read whose real answer is still on the device: its key, where its result words sit in the device
/// log (byte offset), how many decisions, whether they are block-compare flags (u32, nonzero = differ) or word-count
/// slots ([u64; 3], first word zero = equal), and the decisions handed to the host.
#[derive(Clone, Debug)]
pub struct Pending { pub key: Key, pub off: usize, pub n: usize, pub pairs: bool, pub pred: Vec<bool> }

#[derive(Default, Debug)]
pub struct Book {
    /// The real decisions of every key seen in a held attempt, and in how many gradients in a row each key gave them.
    last: HashMap<Key, (Vec<bool>, u32)>,
    /// The real decisions of the attempt under way: read for real, or predicted and then checked.
    cur: HashMap<Key, Vec<bool>>,
    pub pending: Vec<Pending>,
    /// Kernel occurrences of the step under way (by name hash).
    seen: HashMap<u64, u32>,
    step: u32,
    /// The current command's (name hash, occurrence).
    cmd: (u64, u32),
    /// Predict only keys that gave the same decisions in at least this many gradients in a row (AD_TAPE_PREDICT_AGREE,
    /// default 1).
    pub need: u32,
    /// Sites that may be predicted (sites_mask).
    pub sites: u8,
    /// Predicting in this attempt (false: every decision is read for real and recorded).
    pub active: bool,
    pub stats: Stats,
}

#[derive(Default, Debug, Clone)]
pub struct Stats { pub predicted_reads: u64, pub predicted_decisions: u64, pub real_reads: u64, pub misses: u64, pub miss_keys: u64,
                   pub steps_held: u64, pub log_bytes: u64 }

/// The equal/differ decisions of `n` word-count slots ([u64; 3] little-endian each): equal when the first word is 0.
pub fn decode_words(raw: &[u8], n: usize) -> Vec<bool> {
    (0..n).map(|k| u64::from_le_bytes(raw[24 * k..24 * k + 8].try_into().unwrap()) == 0).collect()
}
/// The differ flags of `n` block compares (u32 little-endian each, nonzero = the blocks differ).
pub fn decode_pairs(raw: &[u8], n: usize) -> Vec<bool> {
    (0..n).map(|k| u32::from_le_bytes(raw[4 * k..4 * k + 4].try_into().unwrap()) != 0).collect()
}
/// Device bytes one read takes in the log (8-byte aligned).
pub fn log_bytes(n: usize, pairs: bool) -> usize { (if pairs { 4 * n } else { 24 * n }).div_ceil(8) * 8 }

const MAGIC: &[u8; 8] = b"s16book1";

impl Book {
    pub fn new(need: u32, sites: u8) -> Self { Self { need: need.max(1), sites, ..Default::default() } }
    /// Step `step`'s taping begins (each attempt).
    pub fn begin(&mut self, step: u32, active: bool) {
        self.cur.clear(); self.pending.clear(); self.seen.clear(); self.cmd = (0, 0); self.step = step; self.active = active;
    }
    /// The next taped launch of the step is `name`.
    pub fn set_cmd(&mut self, name: &str) {
        let h = name_hash(name);
        let e = self.seen.entry(h).or_insert(0);
        self.cmd = (h, *e);
        *e += 1;
    }
    pub fn key(&self, site: u8) -> Key { (self.step, self.cmd.0, self.cmd.1, site) }
    /// The decisions to hand the host for `n` reads at `site`, when the book holds them and they held long enough.
    pub fn predict(&self, site: u8, n: usize) -> Option<Vec<bool>> {
        if !self.active || n == 0 || self.sites & (1 << site) == 0 { return None; }
        let (v, agree) = self.last.get(&self.key(site))?;
        (v.len() == n && *agree >= self.need).then(|| v.clone())
    }
    /// A read made for real: its decisions go into the attempt's record.
    pub fn record_real(&mut self, site: u8, bits: Vec<bool>) {
        self.stats.real_reads += 1;
        let k = self.key(site);
        self.cur.insert(k, bits);
    }
    /// A predicted read: its real words were copied to `off` in the device log.
    pub fn queue(&mut self, site: u8, off: usize, n: usize, pairs: bool, pred: Vec<bool>) {
        self.stats.predicted_reads += 1; self.stats.predicted_decisions += n as u64; self.stats.log_bytes += log_bytes(n, pairs) as u64;
        let key = self.key(site);
        self.pending.push(Pending { key, off, n, pairs, pred });
    }
    /// Check every pending prediction against the log read back from the device (`log` holds the bytes from offset
    /// 0). Returns the keys whose prediction was wrong (empty: every decision held). The real decisions go into the
    /// attempt's record either way.
    pub fn verify(&mut self, log: &[u8]) -> Vec<Key> {
        let mut bad = Vec::new();
        for p in std::mem::take(&mut self.pending) {
            let raw = &log[p.off..p.off + log_bytes(p.n, p.pairs).min(log.len() - p.off)];
            let real = if p.pairs { decode_pairs(raw, p.n) } else { decode_words(raw, p.n) };
            if real != p.pred { bad.push(p.key); }
            self.cur.insert(p.key, real);
        }
        if !bad.is_empty() { self.stats.misses += 1; self.stats.miss_keys += bad.len() as u64; }
        bad
    }
    /// The step's taping is over. `held`: every decision of the attempt was the real one (or read for real), so its
    /// record goes into the book for this step. Otherwise the attempt's record is dropped (the redo records).
    pub fn end_step(&mut self, held: bool) {
        self.pending.clear();
        self.active = false;
        if !held { self.cur.clear(); return; }
        self.stats.steps_held += 1;
        // this step's old keys go; its new record comes in, counting the gradients in a row each key held
        let step = self.step;
        let old: HashMap<Key, (Vec<bool>, u32)> = self.last.iter().filter(|(k, _)| k.0 == step).map(|(k, v)| (*k, v.clone())).collect();
        self.last.retain(|k, _| k.0 != step);
        for (k, v) in self.cur.drain() {
            let agree = match old.get(&k) { Some((o, a)) if *o == v => a.saturating_add(1), _ => 1 };
            self.last.insert(k, (v, agree));
        }
    }
    pub fn keys(&self) -> usize { self.last.len() }
    /// The book as bytes (for AD_TAPE_PREDICT_BOOK): magic, entry count, then each entry's step, name hash,
    /// occurrence, site, agree, decision count and the decisions packed 8 to a byte. Entries in key order.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut keys: Vec<&Key> = self.last.keys().collect();
        keys.sort();
        let mut out = Vec::with_capacity(16 + keys.len() * 32);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(keys.len() as u64).to_le_bytes());
        for k in keys {
            let (v, a) = &self.last[k];
            out.extend_from_slice(&k.0.to_le_bytes()); out.extend_from_slice(&k.1.to_le_bytes());
            out.extend_from_slice(&k.2.to_le_bytes()); out.push(k.3);
            out.extend_from_slice(&a.to_le_bytes()); out.extend_from_slice(&(v.len() as u32).to_le_bytes());
            let mut packed = vec![0u8; v.len().div_ceil(8)];
            for (i, &b) in v.iter().enumerate() { if b { packed[i / 8] |= 1 << (i % 8); } }
            out.extend_from_slice(&packed);
        }
        out
    }
    /// The book from to_bytes(); a file that does not parse is refused (the caller then starts with an empty book).
    pub fn load_bytes(&mut self, b: &[u8]) -> Result<usize, String> {
        let bad = || "tape predict book: truncated or not a book".to_string();
        if b.len() < 16 || &b[..8] != MAGIC { return Err(bad()); }
        let n = u64::from_le_bytes(b[8..16].try_into().unwrap()) as usize;
        let mut at = 16usize;
        let mut take = |k: usize| -> Result<&[u8], String> { if at + k > b.len() { return Err(bad()); } let s = &b[at..at + k]; at += k; Ok(s) };
        let mut last = HashMap::with_capacity(n);
        for _ in 0..n {
            let step = u32::from_le_bytes(take(4)?.try_into().unwrap());
            let h = u64::from_le_bytes(take(8)?.try_into().unwrap());
            let occ = u32::from_le_bytes(take(4)?.try_into().unwrap());
            let site = take(1)?[0];
            let agree = u32::from_le_bytes(take(4)?.try_into().unwrap());
            let len = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
            let packed = take(len.div_ceil(8))?;
            let v: Vec<bool> = (0..len).map(|i| packed[i / 8] >> (i % 8) & 1 == 1).collect();
            last.insert((step, h, occ, site), (v, agree));
        }
        if at != b.len() { return Err(bad()); }
        self.last = last;
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn words(bits: &[bool]) -> Vec<u8> {
        bits.iter().flat_map(|&eq| { let mut w = [0u8; 24]; if !eq { w[0] = 7; w[8] = 3; } w }).collect()
    }
    fn pairs(bits: &[bool]) -> Vec<u8> { bits.iter().flat_map(|&d| (d as u32 * 5).to_le_bytes()).collect() }
    type L = Vec<(&'static str, Vec<bool>, Vec<bool>, Vec<bool>)>;

    /// One taped step: launches (name, decisions per site), read for real or predicted. The device answers with `real`.
    fn step(b: &mut Book, s: u32, active: bool, launches: &L) -> (Vec<Key>, usize) {
        b.begin(s, active);
        let mut log: Vec<u8> = Vec::new();
        let mut predicted = 0;
        for (name, pre, blk, post) in launches {
            b.set_cmd(name);
            for (site, real, is_pairs) in [(SITE_PRE_WORDS, pre, false), (SITE_PRE_PAIRS, blk, true), (SITE_POST_WORDS, post, false)] {
                if real.is_empty() { continue; }
                match b.predict(site, real.len()) {
                    Some(p) => {
                        let off = log.len();
                        log.extend(if is_pairs { pairs(real) } else { words(real) });
                        log.resize(off + log_bytes(real.len(), is_pairs), 0);
                        b.queue(site, off, real.len(), is_pairs, p);
                        predicted += 1;
                    },
                    None => b.record_real(site, real.clone()),
                }
            }
        }
        let bad = b.verify(&log);
        b.end_step(bad.is_empty());
        (bad, predicted)
    }
    fn launches(flip: Option<(usize, u8, usize)>) -> L {
        let mut v = vec![("acoustic", vec![true, false, true, true], vec![false, false, true], vec![true, true]),
                         ("transport", vec![false, false], vec![], vec![true, false]),
                         ("acoustic", vec![true, true, true, false], vec![true, false, false], vec![false, true])];
        if let Some((i, site, k)) = flip {
            let t = match site { SITE_PRE_WORDS => &mut v[i].1, SITE_PRE_PAIRS => &mut v[i].2, _ => &mut v[i].3 };
            t[k] = !t[k];
        }
        v
    }
    fn all() -> u8 { sites_mask(Some("pre,pairs,post")) }
    /// A gradient: steps 3, 2, 1 (the walk-back order), each with its own launches; a miss is redone for real.
    fn gradient(b: &mut Book, flips: &[(u32, (usize, u8, usize))]) -> (usize, usize) {
        let (mut misses, mut predicted) = (0, 0);
        for s in [3u32, 2, 1] {
            let f = flips.iter().find(|x| x.0 == s).map(|x| x.1);
            let l = launches(f);
            let (bad, p) = step(b, s, true, &l);
            predicted += p;
            if !bad.is_empty() { misses += 1; let (bad2, _) = step(b, s, false, &l); assert!(bad2.is_empty()); }
        }
        (misses, predicted)
    }

    #[test]
    fn off_reads_everything_for_real() {
        let mut b = Book::new(1, all());
        for _ in 0..3 { let (bad, pred) = step(&mut b, 1, false, &launches(None)); assert!(bad.is_empty()); assert_eq!(pred, 0); }
        assert_eq!(b.stats.predicted_reads, 0);
    }
    #[test]
    fn the_second_gradient_predicts_every_read() {
        let mut b = Book::new(1, all());
        assert_eq!(gradient(&mut b, &[]), (0, 0), "the first gradient has no book: every read is real");
        assert_eq!(gradient(&mut b, &[]), (0, 24), "every read of the second gradient is predicted");
    }
    #[test]
    fn default_sites_leave_the_block_flags_real() {
        let mut b = Book::new(1, sites_mask(None));
        gradient(&mut b, &[]);
        assert_eq!(gradient(&mut b, &[]), (0, 18));
        // a block flag that moves is read, never missed
        assert_eq!(gradient(&mut b, &[(2, (0, SITE_PRE_PAIRS, 1))]), (0, 18));
    }
    #[test]
    fn planted_flip_is_caught_at_every_site() {
        for site in [SITE_PRE_WORDS, SITE_PRE_PAIRS, SITE_POST_WORDS] {
            for i in [0usize, 2] {
                let mut b = Book::new(1, all());
                gradient(&mut b, &[]);
                let mut bb = Book::new(1, all());
                std::mem::swap(&mut bb.last, &mut b.last);
                let (bad, _) = step(&mut bb, 2, true, &launches(Some((i, site, 1))));
                assert_eq!(bad.len(), 1, "site {site} launch {i}: the flipped decision must be caught");
                assert_eq!(bad[0].3, site);
                assert_eq!(bad[0].0, 2);
                // the redo reads for real; the next gradient predicts from the redo's record and holds
                assert!(step(&mut bb, 2, false, &launches(Some((i, site, 1)))).0.is_empty());
                let (bad, pred) = step(&mut bb, 2, true, &launches(Some((i, site, 1))));
                assert!(bad.is_empty()); assert_eq!(pred, 8);
            }
        }
    }
    #[test]
    fn steps_are_kept_apart() {
        // step 2's decisions never predict step 1's
        let mut b = Book::new(1, all());
        step(&mut b, 2, true, &launches(None));
        assert_eq!(step(&mut b, 1, true, &launches(None)).1, 0);
    }
    #[test]
    fn occurrences_are_kept_apart() {
        let mut b = Book::new(1, all());
        step(&mut b, 1, true, &launches(None));
        let mut l = launches(None);
        l.swap(0, 2);
        assert!(!step(&mut b, 1, true, &l).0.is_empty());
    }
    #[test]
    fn a_new_command_list_reads_for_real() {
        let mut b = Book::new(1, all());
        step(&mut b, 1, true, &launches(None));
        let l: L = vec![("radiation", vec![true, false], vec![], vec![true])];
        let (bad, pred) = step(&mut b, 1, true, &l);
        assert!(bad.is_empty()); assert_eq!(pred, 0);
    }
    #[test]
    fn agree_two_waits_for_two_equal_gradients() {
        let mut b = Book::new(2, all());
        assert_eq!(gradient(&mut b, &[]).1, 0);
        assert_eq!(gradient(&mut b, &[]).1, 0);
        assert_eq!(gradient(&mut b, &[]).1, 24);
        let f = [(1u32, (1usize, SITE_POST_WORDS, 0usize))];
        assert_eq!(gradient(&mut b, &f).0, 1);
        // the moved key starts its count again: one read fewer is predicted in the next gradient
        assert_eq!(gradient(&mut b, &f), (0, 23));
    }
    #[test]
    fn the_book_survives_a_file() {
        let mut b = Book::new(1, all());
        gradient(&mut b, &[]);
        let bytes = b.to_bytes();
        let mut c = Book::new(1, all());
        assert_eq!(c.load_bytes(&bytes).unwrap(), b.keys());
        assert_eq!(c.to_bytes(), bytes);
        assert_eq!(gradient(&mut c, &[]), (0, 24));
        assert!(Book::new(1, all()).load_bytes(&bytes[..bytes.len() - 1]).is_err());
        assert!(Book::new(1, all()).load_bytes(b"not a book at all").is_err());
    }
    #[test]
    fn decoders_read_the_device_layout() {
        assert_eq!(decode_words(&words(&[true, false, true]), 3), vec![true, false, true]);
        assert_eq!(decode_pairs(&pairs(&[false, true]), 2), vec![false, true]);
        assert_eq!(log_bytes(3, true), 16);
        assert_eq!(log_bytes(2, false), 48);
        assert_eq!(sites_mask(None), 0b1101);
        assert_eq!(sites_mask(Some("pairs")), 0b010);
    }
}
