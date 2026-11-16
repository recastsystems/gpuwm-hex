// SPDX-License-Identifier: Apache-2.0
//! speed, piece ckpt60 (2026-11-11): the content store of step boundaries (--bstore-content, with --checkpoints).
//!
//! A stored boundary is the whole live device state. THE WASTE THIS REMOVES (60 km, 29 steps, a test machine
//! <local path> test machine/c60-d1/window.json): each boundary is 26.55 GB; the changed-chunk files hold 17.7 GB of
//! it and the leg wrote 454 GB of them, 252 s storing and 396 s loading, and every load rebuilt a whole 26.55 GB copy in
//! host memory beside the copies the sweep already held (boundary 0, the window end twice, the step's start and end
//! states, a read-ahead and its file, two page-locked staging buffers).
//!
//! Three things replace that:
//! - `State`: a checkpoint is a list of pieces (1 MiB of one allocation each) in host memory. A piece whose bytes equal
//!   the same piece of the state captured before it is not stored again: both states point at one copy. A piece of zeros
//!   is not stored at all. The bytes a state gives back are exactly the bytes it was built from (lossless), so restore
//!   writes the same words.
//! - `Truth`: what the redo check compares a re-made end state with. The check needs no copy of the stored end state,
//!   only to know whether the re-made bytes are the same bytes: a truth is the SHA-256 of every piece, plus the bytes
//!   themselves of the allocations that have declared residue ranges at that step (those are compared word by word as
//!   before, so residue words are still counted apart). A piece that differs in one bit has another SHA-256: a wrong
//!   re-made state is refused as before; what is lost is the count of differing words (the refusal names the pieces).
//! - everything moves through one small page-locked buffer, a window (1 GiB) at a time: capture, restore, truth and
//!   check never hold a whole state in host memory.
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::Arc;

pub const PIECE: usize = 1 << 20;
pub type Hash = [u8; 32];
static ZEROS: [u8; PIECE] = [0; PIECE];
/// Bytes held by live pieces now, their peak, and every byte ever copied into a piece.
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static MADE: AtomicU64 = AtomicU64::new(0);
pub fn live_bytes() -> usize { LIVE.load(Relaxed) }
pub fn peak_bytes() -> usize { PEAK.load(Relaxed) }
pub fn made_bytes() -> u64 { MADE.load(Relaxed) }
fn threads() -> usize { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).clamp(1, 32) }
/// f over every item on all cores, results in the items' order.
fn par_map<I: Sync, T: Send>(items: &[I], f: impl Fn(&I) -> T + Sync) -> Vec<T> {
    if items.len() < 8 { return items.iter().map(&f).collect(); }
    let per = items.len().div_ceil(threads()).max(1);
    std::thread::scope(|sc| {
        let f = &f;
        let hs: Vec<_> = items.chunks(per).map(|ch| sc.spawn(move || ch.iter().map(f).collect::<Vec<T>>())).collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

pub struct Piece(Box<[u8]>);
impl Piece {
    fn new(src: &[u8]) -> Arc<Piece> {
        let now = LIVE.fetch_add(src.len(), Relaxed) + src.len();
        PEAK.fetch_max(now, Relaxed);
        MADE.fetch_add(src.len() as u64, Relaxed);
        Arc::new(Piece(src.into()))
    }
}
impl Drop for Piece {
    fn drop(&mut self) {
        LIVE.fetch_sub(self.0.len(), Relaxed);
        // The piece's pages go back to the system now. THE WASTE THIS PREVENTS (box fast-1, 8-step leg cs8, 1 s samples
        // of the program's resident memory): the allocator keeps a freed 1 MiB block in its own heaps (blocks under its
        // 32 MiB map threshold are not unmapped), so the store's dropped states stayed resident and the leg's memory
        // only grew: 255 GiB early in the sweep, 273 GiB late, with fewer states held. The whole pages inside the block
        // are released here; the block itself is then freed as usual (its first and last page are left alone).
        #[cfg(target_os = "linux")]
        {
            let (a, n) = (self.0.as_ptr() as usize, self.0.len());
            let (lo, hi) = ((a + 8191) & !4095, (a + n).saturating_sub(4096) & !4095);
            if hi > lo { unsafe { madvise(lo as *mut std::ffi::c_void, hi - lo, 4); } }
        }
    }
}
#[cfg(target_os = "linux")]
extern "C" { fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32; }

/// How a state is cut into pieces. `index` is a boundary's index: (allocation, byte offset in the whole-state layout,
/// bytes), ascending by allocation. Pieces are numbered in that order; the last piece of an allocation may be short.
#[derive(Clone)]
pub struct Layout {
    pub index: Vec<(u64, usize, usize)>,
    /// first piece of each allocation, and the piece count after the last
    first: Vec<usize>,
    /// of each piece: (position of its allocation in `index`, its number inside the allocation)
    at: Vec<(u32, u32)>,
    pub total: usize,
}
impl Layout {
    pub fn new(index: &[(u64, usize, usize)]) -> Layout {
        let mut first = Vec::with_capacity(index.len() + 1);
        let mut at = Vec::new();
        for (i, &(_, _, n)) in index.iter().enumerate() {
            first.push(at.len());
            for k in 0..n.div_ceil(PIECE) { at.push((i as u32, k as u32)); }
        }
        first.push(at.len());
        Layout { index: index.to_vec(), first, at, total: index.iter().map(|x| x.2).sum() }
    }
    pub fn pieces(&self) -> usize { self.at.len() }
    /// Piece p: (position of its allocation in `index`, byte offset inside the allocation, bytes).
    pub fn piece(&self, p: usize) -> (usize, usize, usize) {
        let (i, k) = (self.at[p].0 as usize, self.at[p].1 as usize);
        (i, k * PIECE, (self.index[i].2 - k * PIECE).min(PIECE))
    }
    /// The pieces in windows of at most `cap` bytes: (first piece, piece after the last, bytes).
    pub fn windows(&self, cap: usize) -> Vec<(usize, usize, usize)> {
        let (mut v, mut a, mut bytes) = (Vec::new(), 0usize, 0usize);
        for p in 0..self.pieces() {
            let n = self.piece(p).2;
            if bytes + n > cap && p > a { v.push((a, p, bytes)); a = p; bytes = 0; }
            bytes += n;
        }
        if self.pieces() > a { v.push((a, self.pieces(), bytes)); }
        v
    }
    /// The pieces [a, b) as runs inside allocations, in order: (position of the allocation, byte offset inside it,
    /// bytes). A window's buffer holds its runs one after the other.
    pub fn runs(&self, a: usize, b: usize) -> Vec<(usize, usize, usize)> {
        let mut v: Vec<(usize, usize, usize)> = Vec::new();
        for p in a..b {
            let (i, off, n) = self.piece(p);
            match v.last_mut() { Some(l) if l.0 == i && l.1 + l.2 == off => l.2 += n, _ => v.push((i, off, n)) }
        }
        v
    }
    /// Of the pieces [a, b) in a window buffer: (piece, offset in the buffer, bytes).
    fn in_window(&self, a: usize, b: usize) -> Vec<(usize, usize, usize)> {
        let mut at = 0usize;
        (a..b).map(|p| { let n = self.piece(p).2; let r = (p, at, n); at += n; r }).collect()
    }
}

/// One whole state: its pieces. None is a piece of zeros.
pub struct State {
    pub lay: Layout,
    pieces: Vec<Option<Arc<Piece>>>,
    /// of this state's bytes: copied into new pieces, shared with the reference state, zeros
    pub new_bytes: usize,
    pub shared_bytes: usize,
    pub zero_bytes: usize,
    /// speed live (--bstore-live 1): bytes of live allocations this state leaves out (the plan's non-carried
    /// allocations at an intermediate boundary; window.rs live_index). 0: the state holds every live allocation, and
    /// only then has it whole bytes to give (`whole`).
    pub left_out_bytes: usize,
}
/// Builds a state from its windows, fed in order. Every piece that equals the same piece (same allocation, same size,
/// same number) of `reference` is shared with it.
pub struct Builder {
    lay: Layout,
    reference: Option<Arc<State>>,
    /// per allocation: the first piece of the same allocation in the reference, when it has the same size there
    rfirst: Vec<Option<usize>>,
    pieces: Vec<Option<Arc<Piece>>>,
    by: [usize; 3],
}
impl Builder {
    pub fn new(lay: Layout, reference: Option<Arc<State>>) -> Builder {
        let rmap: HashMap<u64, usize> = reference.as_ref().map(|r| r.lay.index.iter().enumerate().map(|(i, x)| (x.0, i)).collect()).unwrap_or_default();
        let rfirst = lay.index.iter().map(|&(id, _, n)| reference.as_ref().and_then(|r| rmap.get(&id).filter(|&&i| r.lay.index[i].2 == n).map(|&i| r.lay.first[i]))).collect();
        Builder { pieces: Vec::with_capacity(lay.pieces()), lay, reference, rfirst, by: [0; 3] }
    }
    /// The pieces [a, b), whose bytes `buf` holds one after the other.
    pub fn feed(&mut self, a: usize, b: usize, buf: &[u8]) {
        assert_eq!(a, self.pieces.len(), "content store: windows are fed in order");
        let items = self.lay.in_window(a, b);
        let (lay, rfirst, reference) = (&self.lay, &self.rfirst, self.reference.as_deref());
        // kind: 0 new, 1 shared, 2 zeros
        let out = par_map(&items, |&(p, o, len)| {
            let x = &buf[o..o + len];
            let (i, k) = (lay.at[p].0 as usize, lay.at[p].1 as usize);
            if let (Some(f), Some(r)) = (rfirst[i], reference) { if let Some(q) = &r.pieces[f + k] { if q.0[..] == *x { return (Some(q.clone()), 1u8, len); } } }
            if *x == ZEROS[..len] { (None, 2, len) } else { (Some(Piece::new(x)), 0, len) }
        });
        for (p, kind, len) in out { self.by[kind as usize] += len; self.pieces.push(p); }
    }
    pub fn finish(self) -> State {
        assert_eq!(self.pieces.len(), self.lay.pieces(), "content store: a state is built from all its windows");
        State { lay: self.lay, pieces: self.pieces, new_bytes: self.by[0], shared_bytes: self.by[1], zero_bytes: self.by[2], left_out_bytes: 0 }
    }
}
impl State {
    /// The pieces [a, b) into `buf`, one after the other, on all cores.
    pub fn fill(&self, a: usize, b: usize, buf: &mut [u8]) {
        let mut slices: Vec<(&mut [u8], &Option<Arc<Piece>>)> = Vec::with_capacity(b - a);
        let mut rest: &mut [u8] = buf;
        for p in a..b {
            let (x, t) = std::mem::take(&mut rest).split_at_mut(self.lay.piece(p).2);
            rest = t;
            slices.push((x, &self.pieces[p]));
        }
        let per = slices.len().div_ceil(threads()).max(1);
        std::thread::scope(|sc| {
            let mut s = slices;
            while !s.is_empty() {
                let rest = s.split_off(s.len().min(per));
                let mine = std::mem::replace(&mut s, rest);
                sc.spawn(move || for (d, p) in mine { match p { Some(p) => d.copy_from_slice(&p.0), None => d.fill(0) } });
            }
        });
    }
    /// The whole state as one buffer in the layout of its index (the dots and the checks outside the sweep).
    pub fn whole(&self) -> Vec<u8> {
        let mut v = vec![0u8; self.lay.total];
        self.fill(0, self.lay.pieces(), &mut v);
        v
    }
    /// The whole state to a file, piece after piece: the bytes `whole` gives.
    pub fn write(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::io::BufWriter::with_capacity(8 << 20, std::fs::File::create(path)?);
        for p in 0..self.lay.pieces() {
            match &self.pieces[p] { Some(q) => f.write_all(&q.0)?, None => f.write_all(&ZEROS[..self.lay.piece(p).2])? }
        }
        f.flush()
    }
    /// The bytes of the allocation at position `i` of the index.
    pub fn alloc(&self, i: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(self.lay.index[i].2);
        for p in self.lay.first[i]..self.lay.first[i + 1] {
            match &self.pieces[p] { Some(q) => v.extend_from_slice(&q.0), None => v.extend_from_slice(&ZEROS[..self.lay.piece(p).2]) }
        }
        v
    }
    /// The truth of this state (see Truth), read from its pieces. `exact`: positions of the allocations kept as bytes.
    pub fn truth(&self, exact: &[usize]) -> Truth {
        let ps: Vec<usize> = (0..self.lay.pieces()).collect();
        let hashes = par_map(&ps, |&p| -> Hash { match &self.pieces[p] { Some(q) => Sha256::digest(&q.0[..]).into(), None => Sha256::digest(&ZEROS[..self.lay.piece(p).2]).into() } });
        Truth { lay: self.lay.clone(), hashes, exact: exact.iter().map(|&i| (i, self.alloc(i))).collect() }
    }
}

/// What a re-made state is compared with: the SHA-256 of every piece, and the bytes of the `exact` allocations.
pub struct Truth {
    pub lay: Layout,
    hashes: Vec<Hash>,
    exact: HashMap<usize, Vec<u8>>,
}
impl Truth {
    pub fn exact(&self, i: usize) -> Option<&[u8]> { self.exact.get(&i).map(|v| &v[..]) }
    pub fn exact_bytes(&self) -> usize { self.exact.values().map(|v| v.len()).sum() }
}
/// Builds a truth from the windows of a state, fed in order.
pub struct TruthBuilder { lay: Layout, hashes: Vec<Hash>, exact: HashMap<usize, Vec<u8>> }
impl TruthBuilder {
    pub fn new(lay: Layout, exact: &[usize]) -> TruthBuilder {
        let exact = exact.iter().map(|&i| (i, Vec::with_capacity(lay.index[i].2))).collect();
        TruthBuilder { hashes: Vec::with_capacity(lay.pieces()), lay, exact }
    }
    pub fn feed(&mut self, a: usize, b: usize, buf: &[u8]) {
        assert_eq!(a, self.hashes.len(), "content store: windows are fed in order");
        let items = self.lay.in_window(a, b);
        let h = par_map(&items, |&(_, o, len)| -> Hash { Sha256::digest(&buf[o..o + len]).into() });
        self.hashes.extend(h);
        if !self.exact.is_empty() { for &(p, o, len) in &items { if let Some(v) = self.exact.get_mut(&(self.lay.at[p].0 as usize)) { v.extend_from_slice(&buf[o..o + len]); } } }
    }
    pub fn finish(self) -> Truth {
        assert_eq!(self.hashes.len(), self.lay.pieces(), "content store: a truth is built from all its windows");
        Truth { lay: self.lay, hashes: self.hashes, exact: self.exact }
    }
}
/// Compares a re-made state, fed in the windows of the truth's layout, with a truth. Only the allocations marked in
/// `want` (by position) are compared. An allocation the truth holds as bytes is gathered whole for the word-by-word
/// compare; every other piece is compared by its SHA-256.
pub struct Checker<'a> { truth: &'a Truth, want: Vec<bool>, pub bad: Vec<usize>, pub live: HashMap<usize, Vec<u8>>, fed: usize }
impl<'a> Checker<'a> {
    pub fn new(truth: &'a Truth, want: Vec<bool>) -> Checker<'a> {
        let live = truth.exact.keys().filter(|&&i| want[i]).map(|&i| (i, Vec::with_capacity(truth.lay.index[i].2))).collect();
        Checker { truth, want, bad: Vec::new(), live, fed: 0 }
    }
    pub fn feed(&mut self, a: usize, b: usize, buf: &[u8]) {
        assert_eq!(a, self.fed, "content store: windows are fed in order");
        self.fed = b;
        let (t, want) = (self.truth, &self.want);
        let items = t.lay.in_window(a, b);
        let bad = par_map(&items, |&(p, o, len)| -> bool {
            let i = t.lay.at[p].0 as usize;
            if !want[i] || t.exact.contains_key(&i) { return false; }
            let h: Hash = Sha256::digest(&buf[o..o + len]).into();
            h != t.hashes[p]
        });
        self.bad.extend(items.iter().zip(&bad).filter(|x| *x.1).map(|x| x.0 .0));
        if !self.live.is_empty() { for &(p, o, len) in &items { if let Some(v) = self.live.get_mut(&(t.lay.at[p].0 as usize)) { v.extend_from_slice(&buf[o..o + len]); } } }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn words(n: usize, seed: u32) -> Vec<u8> { (0..n).map(|i| ((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 13) as u8).collect() }
    fn build(index: &[(u64, usize, usize)], data: &[u8], reference: Option<Arc<State>>, cap: usize) -> State {
        let lay = Layout::new(index);
        let mut b = Builder::new(lay.clone(), reference);
        let mut at = 0;
        for (a, z, n) in lay.windows(cap) { b.feed(a, z, &data[at..at + n]); at += n; }
        assert_eq!(at, data.len());
        b.finish()
    }
    fn check(t: &Truth, want: Vec<bool>, data: &[u8], cap: usize) -> (Vec<usize>, HashMap<usize, Vec<u8>>) {
        let mut c = Checker::new(t, want);
        let mut at = 0;
        for (a, z, n) in t.lay.windows(cap) { c.feed(a, z, &data[at..at + n]); at += n; }
        (c.bad, c.live)
    }
    #[test]
    fn a_state_gives_back_its_bytes_shares_equal_pieces_and_its_truth_refuses_one_changed_bit() {
        // three allocations: one larger than two pieces, one of zeros, one small; then an empty one
        let index = vec![(3u64, 0usize, 2 * PIECE + 4096), (7, 2 * PIECE + 4096, PIECE), (9, 3 * PIECE + 4096, 12), (11, 3 * PIECE + 4108, 0)];
        let total = 3 * PIECE + 4096 + 12;
        let mut a = words(total, 1);
        a[2 * PIECE + 4096..3 * PIECE + 4096].fill(0);
        for cap in [PIECE, 2 * PIECE + 100, 64 * PIECE] {
            let sa = build(&index, &a, None, cap);
            assert_eq!((sa.lay.total, sa.zero_bytes, sa.shared_bytes, sa.new_bytes), (total, PIECE, 0, total - PIECE));
            assert_eq!(sa.whole(), a);
        }
        let sa = Arc::new(build(&index, &a, None, 2 * PIECE));
        assert_eq!(sa.lay.runs(0, sa.lay.pieces()), vec![(0, 0, 2 * PIECE + 4096), (1, 0, PIECE), (2, 0, 12)]);
        // the next state: one word of the first allocation's second piece changed, and a new allocation in the middle
        let index2 = vec![(3u64, 0usize, 2 * PIECE + 4096), (5, 2 * PIECE + 4096, 64), (7, 2 * PIECE + 4160, PIECE), (9, 3 * PIECE + 4160, 12)];
        let mut b = Vec::new();
        b.extend_from_slice(&a[..2 * PIECE + 4096]); b.extend_from_slice(&words(64, 5)); b.extend_from_slice(&a[2 * PIECE + 4096..]);
        b[PIECE + 8] ^= 1;
        let sb = build(&index2, &b, Some(sa.clone()), PIECE + 5000);
        assert_eq!((sb.new_bytes, sb.shared_bytes, sb.zero_bytes), (PIECE + 64, PIECE + 4096 + 12, PIECE));
        assert_eq!(sb.whole(), b);
        assert_eq!(sb.alloc(0), b[..2 * PIECE + 4096].to_vec());
        assert_eq!(sb.alloc(2), vec![0u8; PIECE]);
        // a window of the state is the window it was built from
        let w = sb.lay.windows(PIECE + 5000);
        let mut buf = vec![0xffu8; w[1].2];
        sb.fill(w[1].0, w[1].1, &mut buf);
        assert_eq!(buf, b[w[0].2..w[0].2 + w[1].2].to_vec());
        // dropping the first state keeps the shared pieces alive for the second
        drop(sa);
        assert_eq!(sb.whole(), b);
        // the file of a state is its bytes
        let f = std::env::temp_dir().join(format!("cstore-test-{}.bin", std::process::id()));
        sb.write(&f).unwrap();
        assert_eq!(std::fs::read(&f).unwrap(), b);
        let _ = std::fs::remove_file(&f);
        // the truth, read from the pieces and built from windows, is the same and accepts the state's own bytes
        let t = sb.truth(&[3]);
        let lay = Layout::new(&index2);
        let mut tb = TruthBuilder::new(lay.clone(), &[3]);
        let mut at = 0;
        for (x, z, n) in lay.windows(3 * PIECE) { tb.feed(x, z, &b[at..at + n]); at += n; }
        let t2 = tb.finish();
        assert_eq!(t.hashes, t2.hashes);
        assert_eq!(t.exact(3), Some(&b[3 * PIECE + 4160..]));
        assert_eq!(t2.exact(3), t.exact(3));
        assert_eq!(t.exact_bytes(), 12);
        let (bad, live) = check(&t, vec![true; 4], &b, PIECE);
        assert!(bad.is_empty());
        assert_eq!(live[&3], b[3 * PIECE + 4160..].to_vec());
        // one changed bit anywhere is refused, and the piece is named
        let mut c = b.clone();
        c[2 * PIECE + 4096 + 3] ^= 0x10;
        assert_eq!(check(&t2, vec![true; 4], &c, 2 * PIECE).0, vec![3]);
        let mut c = b.clone();
        c[2 * PIECE + 4160 + 77] = 1;
        assert_eq!(check(&t, vec![true; 4], &c, 64 * PIECE).0, vec![4]);
        // an allocation that is not asked for is not compared; an exact one comes back as bytes, not as a piece
        assert!(check(&t, vec![true, true, false, true], &c, PIECE).0.is_empty());
        let mut c = b.clone();
        *c.last_mut().unwrap() ^= 1;
        let (bad, live) = check(&t, vec![true; 4], &c, PIECE);
        assert!(bad.is_empty());
        assert_ne!(Some(&live[&3][..]), t.exact(3));
    }
}
