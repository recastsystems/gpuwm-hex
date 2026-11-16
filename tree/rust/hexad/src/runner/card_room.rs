// SPDX-License-Identifier: Apache-2.0
//! s60-cardmem: how the card's free room is shared out when the reverse starts. No arithmetic on weather fields.
//!
//! THE BREAKAGE THIS PREVENTS: the reverse's two big users of card memory were sized by fixed reserves that know nothing
//! of the card. The boundary layer's (MYNN-EDMF) reverse budget kept 12 GiB free (16 GiB for its first launch's tangent
//! budget) and was capped at 32 GiB, the tape arena kept 12 GiB free, and the boundary layer's ring is the TEMPO arena,
//! whose size is a command-line number. At 60 km on a 96 GB card (six-3 b8-head) that left the boundary layer a 4.29 GB
//! ring (11.3 wraps a launch) while the ring that keeps its 761 slots busy is 15.2 GB (six-2 L8-r5,
//! ring_bytes_for_slots), and a gradient-only run's first launch a 4.41 GB budget: 305 to 457 slots, not 761 (six-2
//! G29-f6). On a 179 GB class card the same reserves would leave tens of GB idle while a step's tape (82 GB in one
//! arena, six-3 t8-rev90) still went through the host in chunks.
//!
//! The rule (`split`): the room is what the card has free when the reverse starts less what must stay free, and what
//! must stay free is measured, not fixed: the largest family of cold blocks (scheme scratch kept in host memory and pulled
//! onto the card while its scheme runs: one family runs at a time, so the card must hold the largest of them beside
//! everything resident), what the reverse itself allocates after it starts (a gradient-only run makes its boundary-layer
//! transient regions then: 9.77 GB in G29-f6), and 1/64 of the card for the driver. The boundary layer takes the room
//! first, up to the ring that keeps all its slots busy or the ring that never wraps, whichever is smaller (past that, more
//! ring only adds laps, which cost nothing: ring-60km, gen_whole.py mw_persist); before its first counted launch that ring
//! is not known and it takes at most 1/8 of the card. The tape arena takes what is left, up to the largest step's tape.
//! What the reverse allocates and the largest cold family are known only once a run has reversed a step, so a first run
//! shares nothing out; AD_CARD_ROOM_FILE carries the measurements to the next run of the same package on the same card.
//!
//! Where bytes live is the only change: the ring plan, the slots and the tape chunking never change a word.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Arena granule the tape and the ring grow by (the driver's large page).
pub const GRANULE: u64 = 2 << 20;

/// What the driver keeps for itself on a card of `total` bytes: 1/64 of it (1.56 GB on a 96 GB card budget).
pub fn head(total: u64) -> u64 { total / 64 }

/// The ring the boundary layer wants: the smaller of the ring that keeps every slot busy (`full`) and the ring that never
/// wraps (`nowrap`); 0 for either means not known; both unknown: 1/8 of the card.
pub fn bl_target(full: u64, nowrap: u64, total: u64) -> u64 {
    match (full, nowrap) { (0, 0) => total / 8, (0, n) => n, (f, 0) => f, (f, n) => f.min(n) }
}

/// What the split needs to know besides the card's free and total bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Need {
    /// the largest cold family (live now or measured by an earlier run, the larger)
    pub cold_peak: u64,
    /// device bytes the reverse allocated after it started, as an earlier run measured them; None: not known
    pub rev_growth: Option<u64>,
    /// the boundary layer's ring now (the lent TEMPO arena)
    pub bl_held: u64,
    /// the ring that keeps every slot busy, the ring that never wraps (0: not known)
    pub bl_full: u64,
    pub bl_nowrap: u64,
    /// the tape arena now, and the largest step's tape (0: not known)
    pub tape_held: u64,
    pub tape_step: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Split {
    /// what stays free: largest cold family + the reverse's own growth + head (u64::MAX: growth not known)
    pub reserve: u64,
    /// free less the reserve
    pub room: u64,
    /// the boundary layer's ring target
    pub bl_target: u64,
    /// bytes the boundary layer's ring grows by (whole granules)
    pub bl_add: u64,
    /// bytes the tape arena grows by (whole granules)
    pub tape_add: u64,
    /// tape arena after the growth
    pub tape_after: u64,
    /// tape_after / the largest step's tape (0 when it is not known)
    pub tape_share: f64,
    /// room nobody took
    pub left: u64,
}

/// Share the room: the boundary layer first, up to its target less what its ring holds; then the tape, up to the largest
/// step's tape less what the arena holds. Growth not known: nothing is shared (room 0).
pub fn split(free: u64, total: u64, n: &Need) -> Split {
    let reserve = match n.rev_growth { Some(g) => n.cold_peak.saturating_add(g).saturating_add(head(total)), None => u64::MAX };
    let room = free.saturating_sub(reserve);
    let bl_target = bl_target(n.bl_full, n.bl_nowrap, total);
    let bl_add = room.min(bl_target.saturating_sub(n.bl_held)) / GRANULE * GRANULE;
    let rest = room - bl_add;
    let tape_add = rest.min(n.tape_step.saturating_sub(n.tape_held)) / GRANULE * GRANULE;
    let tape_after = n.tape_held + tape_add;
    let tape_share = if n.tape_step > 0 { tape_after as f64 / n.tape_step as f64 } else { 0.0 };
    Split { reserve, room, bl_target, bl_add, tape_add, tape_after, tape_share, left: rest - tape_add }
}

/// The boundary layer at one reverse launch: whether `nontape` bytes of slot growth and a ring of `ring_need` bytes fit
/// the card. `free`: the card's free bytes now (cold blocks excluded, as cuMemGetInfo reports them under a budget);
/// `cold_others`: the largest cold family other than the boundary layer's own; `bl_cold_now`: the boundary layer's own
/// cold blocks (on the card while it runs; the slot growth adds to them); `ring_have`: the part of the ring already held
/// that a larger ring can keep. The ring's growth is device memory (never cold, so it is not copied to the host when the
/// next family's blocks come in) with 1/8 slack.
pub fn bl_fits(free: u64, total: u64, cold_others: u64, bl_cold_now: u64, nontape: u64, ring_need: u64, ring_have: u64) -> bool {
    let grow = ring_need.saturating_sub(ring_have);
    let grow = grow + grow / 8;
    let keep = cold_others.max(bl_cold_now + nontape) + head(total);
    grow.saturating_add(keep) <= free
}

/// The most slots (at most `s_cap`, in steps of 8 and `s_cap` itself) whose ring that keeps them all busy (its cost
/// `cost_of(s)`: 0 when the ring held already holds it) and slot growth `nontape_of(s)` fit (`bl_fits`). None: none fits.
pub fn bl_choose(s_cap: usize, free: u64, total: u64, cold_others: u64, bl_cold_now: u64, ring_have: u64,
                 cost_of: &dyn Fn(usize) -> u64, nontape_of: &dyn Fn(usize) -> u64) -> Option<usize> {
    let mut best: Option<usize> = None;
    let mut s = 1usize;
    while s <= s_cap.max(1) {
        if bl_fits(free, total, cold_others, bl_cold_now, nontape_of(s), cost_of(s), ring_have) { best = Some(s); }
        s = if s < s_cap && s + 8 > s_cap { s_cap } else { s + 8 };
    }
    best
}

// What a run measured, for AD_CARD_ROOM_FILE. Kept here so the scheme loader and the window need no new fields.
static BL_FULL: AtomicU64 = AtomicU64::new(0);
static BL_NOWRAP: AtomicU64 = AtomicU64::new(0);
static TAPE_STEP: AtomicU64 = AtomicU64::new(0);
static COLD_PEAK: AtomicU64 = AtomicU64::new(0);
static REV_GROWTH: AtomicU64 = AtomicU64::new(0);
/// device bytes when the reverse started, +1 (0: not started)
static REV_DEV0: AtomicU64 = AtomicU64::new(0);
/// 60 km tape: device bytes the boundary layer's ring of its own holds now (its nodes and bars blocks, the retired
/// ones included; 0 while the ring is the lent arena).
static BL_OWN: AtomicU64 = AtomicU64::new(0);
/// 60 km tape: BL_OWN when the reverse started (it is inside the start's device bytes already, so only what the
/// ring of its own adds after the start is taken out of the growth)
static BL_OWN0: AtomicU64 = AtomicU64::new(0);
pub fn note_bl(full: u64, nowrap: u64) { BL_FULL.fetch_max(full, Relaxed); BL_NOWRAP.fetch_max(nowrap, Relaxed); }
pub fn note_tape_step(bytes: u64) { TAPE_STEP.fetch_max(bytes, Relaxed); }
/// 60 km tape: the boundary layer's own ring now holds `bytes` of device memory (schemes, after it is made or its
/// retired blocks are freed; only when AD_CARD_ROOM_FILE is set, the one place the growth is used).
pub fn note_bl_own(bytes: u64) { BL_OWN.store(bytes, Relaxed); }
pub fn bl_own_now() -> u64 { BL_OWN.load(Relaxed) }
/// The reverse starts with `dev` device bytes held (after the room is shared out: what it adds is not growth).
pub fn note_rev_start(dev: u64) {
    if REV_DEV0.compare_exchange(0, dev + 1, Relaxed, Relaxed).is_ok() { BL_OWN0.store(BL_OWN.load(Relaxed), Relaxed); }
}
/// The reverse's own growth at one sample: device bytes now less those at the reverse start (`d0`) less what the boundary
/// layer's ring of its own added since the start (`bl_own` now less `bl_own0` then; a ring that shrank adds nothing).
///
/// 60 km tape. THE BREAKAGE THIS PREVENTS: the ring of its own that the boundary layer makes during the reverse
/// (when the room left beside the lent arena lets a fresh ring keep 5/4 as many groups in flight) was counted as the
/// reverse's growth, so the next run kept that much of the card free and gave the lent ring nothing, then made the same
/// ring of its own again, with the lent arena idle beside it. a test machine card 1, a test program (AD_P3_STRIPE=1 with
/// the room words): st-R1 measured rev_growth 11,886,527,260 bytes with a 10,956,570,624-byte ring of its own in all 48
/// launches; st-R2 then gave the lent ring 56,623,104 bytes (room 58,044,420) and ran the same 10.96 GB ring of its own
/// beside the idle 6.44 GB lent arena. That ring is what the split sizes (the boundary layer's target), not something
/// the card must keep free for it, so it is not growth.
pub fn growth(dev: u64, d0: u64, bl_own: u64, bl_own0: u64) -> u64 { dev.saturating_sub(d0).saturating_sub(bl_own.saturating_sub(bl_own0)) }
/// During the reverse: device bytes now and the largest cold family now.
pub fn note_rev_now(dev: u64, cold_peak: u64) {
    COLD_PEAK.fetch_max(cold_peak, Relaxed);
    let d0 = REV_DEV0.load(Relaxed);
    if d0 > 0 { REV_GROWTH.fetch_max(growth(dev, d0 - 1, BL_OWN.load(Relaxed), BL_OWN0.load(Relaxed)), Relaxed); }
}

/// Measurements: (bl_full, bl_nowrap, tape_step, cold_peak, rev_growth; None: not measured).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Known { pub bl_full: u64, pub bl_nowrap: u64, pub tape_step: u64, pub cold_peak: u64, pub rev_growth: Option<u64> }
impl Known {
    fn max(self, o: Known) -> Known {
        Known { bl_full: self.bl_full.max(o.bl_full), bl_nowrap: self.bl_nowrap.max(o.bl_nowrap), tape_step: self.tape_step.max(o.tape_step),
                cold_peak: self.cold_peak.max(o.cold_peak), rev_growth: match (self.rev_growth, o.rev_growth) { (Some(a), Some(b)) => Some(a.max(b)), (a, b) => a.or(b) } }
    }
}
pub fn seen() -> Known {
    let started = REV_DEV0.load(Relaxed) > 0;
    Known { bl_full: BL_FULL.load(Relaxed), bl_nowrap: BL_NOWRAP.load(Relaxed), tape_step: TAPE_STEP.load(Relaxed),
            cold_peak: COLD_PEAK.load(Relaxed), rev_growth: (started && TAPE_STEP.load(Relaxed) > 0).then(|| REV_GROWTH.load(Relaxed)) }
}

/// The file a run reads its earlier measurements from and writes its own to (AD_CARD_ROOM_FILE; one per package, card
/// class and scheme image, as the count cache folder is). Plain text, one `key bytes` pair a line.
pub fn file() -> Option<std::path::PathBuf> { std::env::var("AD_CARD_ROOM_FILE").ok().filter(|v| !v.is_empty()).map(std::path::PathBuf::from) }
pub fn parse(text: &str) -> Known {
    let g = |k: &str| text.lines().filter_map(|l| { let mut w = l.split_whitespace(); (w.next() == Some(k)).then(|| w.next()).flatten() })
        .filter_map(|v| v.parse::<u64>().ok()).max();
    Known { bl_full: g("bl_full_bytes").unwrap_or(0), bl_nowrap: g("bl_nowrap_bytes").unwrap_or(0), tape_step: g("tape_step_bytes").unwrap_or(0),
            cold_peak: g("cold_peak_bytes").unwrap_or(0), rev_growth: g("rev_growth_bytes") }
}
pub fn render(k: &Known) -> String {
    let mut s = format!("bl_full_bytes {}\nbl_nowrap_bytes {}\ntape_step_bytes {}\ncold_peak_bytes {}\n", k.bl_full, k.bl_nowrap, k.tape_step, k.cold_peak);
    if let Some(g) = k.rev_growth { s += &format!("rev_growth_bytes {g}\n"); }
    s + "# s60-cardmem: the boundary layer's ring that keeps every slot busy and that never wraps, the largest step's tape (sum of its chunks), the largest cold family, the device bytes the reverse allocated after it started: the largest any run measured\n"
}
/// What the file holds and this run measured, each the larger.
pub fn known() -> Known {
    let f = file().and_then(|p| std::fs::read_to_string(p).ok()).map(|s| parse(&s)).unwrap_or_default();
    seen().max(f)
}
/// Write what is known (never lowers a value the file holds).
pub fn save() {
    let Some(p) = file() else { return };
    let k = known();
    if k == Known::default() { return; }
    let tmp = p.with_extension("tmp");
    if std::fs::write(&tmp, render(&k)).is_ok() { let _ = std::fs::rename(&tmp, &p); }
}

/// AD_CARD_ROOM=1: at the reverse start the boundary layer's lent ring and then the tape arena take the room (`split`),
/// and the reverse scratch is sized to its measured need (AD_REV_SCRATCH_FIT=max).
pub fn on() -> bool { std::env::var("AD_CARD_ROOM").as_deref() == Ok("1") }
/// AD_P3_ROOM_RULE=1: the boundary layer's reverse launches size slots and ring by `bl_choose`.
pub fn bl_on() -> bool { std::env::var("AD_P3_ROOM_RULE").as_deref() == Ok("1") }
/// AD_CARD_RELEASE=1: retired blocks no later step reads are freed when the reverse starts.
pub fn release_on() -> bool { std::env::var("AD_CARD_RELEASE").as_deref() == Ok("1") }

#[cfg(test)]
mod tests {
    use super::*;
    const GB: f64 = 1e9;

    // 60 km, 96 GB card (six-3, --vram-budget-gb 93: cuMemGetInfo total 99,857,989,632 bytes), from the run files:
    // b8-head window_tape_sized card_free_bytes 12,291,650,336; window_scratch_grown 13,553,893,376 against the max
    // sizing of the same need (9,190,900,736 + 256 MiB + 64 MiB, 2 MiB-rounded = 9,527,361,536); the cold families of the
    // census before the reverse (ruc 5,450,498,048 + 3,667,918,848); what the reverse allocated after it started (the
    // censuses after each reverse step, largest 87,915,384,828, against 87,566,339,296 before); the lent
    // TEMPO ring 4 GiB; L8-r5 / pbl-rg8 largest ring_bytes_for_slots 15,199,480,200 and nodes_total 2,799,605,812 (x 24
    // bytes a node); t8-rev90 step 8 in one arena 82,014,107,168; ring-60km's tape after the MYNN bind 10,340,007,936
    // against 15,032,385,536.
    const TOTAL96: u64 = 99_857_989_632;
    const FREE_B8: u64 = 12_291_650_336;
    const SCRATCH_SLACK: u64 = 13_553_893_376 - 9_527_361_536;
    const RUC: u64 = 5_450_498_048 + 3_667_918_848;
    const GROW_B8: u64 = 87_915_384_828 - 87_566_339_296;
    const RING4: u64 = 4 << 30;
    const BL_FULL: u64 = 15_199_480_200;
    const BL_NOWRAP: u64 = 2_799_605_812 * 24;
    const TAPE_STEP: u64 = 82_014_107_168;
    const TAPE_RING60: u64 = 10_340_007_936;
    const TAPE_2X: u64 = 15_032_385_536;
    fn need(tape_held: u64) -> Need {
        Need { cold_peak: RUC, rev_growth: Some(GROW_B8), bl_held: RING4, bl_full: BL_FULL, bl_nowrap: BL_NOWRAP, tape_held, tape_step: TAPE_STEP }
    }

    #[test]
    fn the_60km_96gb_card_gives_its_room_to_the_boundary_layer() {
        assert_eq!(SCRATCH_SLACK, 4_026_531_840);
        // the program as it ran (tape doubled by the MYNN bind), with the scratch cut to its need
        let s = split(FREE_B8 + SCRATCH_SLACK, TOTAL96, &need(TAPE_2X));
        assert_eq!(s.reserve, RUC + GROW_B8 + TOTAL96 / 64);
        assert_eq!(s.room, FREE_B8 + SCRATCH_SLACK - s.reserve);
        assert_eq!(s.bl_target, BL_FULL, "15.2 GB keeps 761 slots busy; never wrapping would take 67.2 GB");
        assert_eq!(s.bl_add, s.room / GRANULE * GRANULE, "the whole room goes to the ring: it is still short of its target");
        assert_eq!(s.tape_add, 0);
        println!("96 GB, tape as it ran (2x): room {:.2} GB -> ring {:.2} GB ({} nodes), tape {:.2} GB = {:.1} % of step 8",
                 s.room as f64 / GB, (RING4 + s.bl_add) as f64 / GB, (RING4 + s.bl_add) / 24, s.tape_after as f64 / GB, 100.0 * s.tape_share);
        // with ring-60km's tape (need + 1/8, not 2x): 4.69 GB more room
        let r = split(FREE_B8 + SCRATCH_SLACK + (TAPE_2X - TAPE_RING60), TOTAL96, &need(TAPE_RING60));
        assert!(r.bl_add <= BL_FULL - RING4);
        assert_eq!(r.bl_add + r.tape_add + r.left, r.room);
        println!("96 GB, ring-60km tape: room {:.2} GB -> ring {:.2} GB ({} nodes), tape {:.2} GB = {:.1} % of step 8, left {:.2} GB",
                 r.room as f64 / GB, (RING4 + r.bl_add) as f64 / GB, (RING4 + r.bl_add) / 24, r.tape_after as f64 / GB, 100.0 * r.tape_share, r.left as f64 / GB);
    }

    #[test]
    fn a_179gb_card_holds_the_ring_and_a_whole_step_of_tape() {
        // the same resident set on a 179 GiB card run with --vram-budget-gb 175 (total 175 GiB): everything but the tape
        // (t8-rev90 census 72,533,953,760) less the scratch slack, and ring-60km's tape
        let total: u64 = 175 << 30;
        let resident = 72_533_953_760 - SCRATCH_SLACK;
        let free = total - resident - TAPE_RING60;
        let s = split(free, total, &need(TAPE_RING60));
        assert_eq!(s.bl_add, (BL_FULL - RING4) / GRANULE * GRANULE, "the ring gets its whole target");
        assert!(TAPE_STEP - s.tape_after < GRANULE, "the tape holds a whole step (to the granule)");
        assert!(s.tape_share > 0.999 && s.tape_share <= 1.0);
        assert_eq!(s.bl_add + s.tape_add + s.left, s.room);
        println!("179 GiB (budget 175): room {:.2} GB -> ring {:.2} GB, tape {:.2} GB = {:.1} % of step 8, left {:.2} GB",
                 s.room as f64 / GB, (RING4 + s.bl_add) as f64 / GB, s.tape_after as f64 / GB, 100.0 * s.tape_share, s.left as f64 / GB);
        // the program notes the ring for every resident slot (1,504, not the 761 that ran): about twice the 761-slot ring
        // (an approximation for this test: ring_for_slots is the largest sum of that many consecutive groups)
        let mut n2 = need(TAPE_RING60);
        n2.bl_full = 2 * BL_FULL;
        let s2 = split(free, total, &n2);
        assert_eq!(s2.bl_add, (2 * BL_FULL - RING4) / GRANULE * GRANULE);
        assert!(s2.tape_share > 0.95);
        println!("179 GiB, 1,504-slot ring: ring {:.2} GB, tape {:.2} GB = {:.1} % of step 8, left {:.2} GB",
                 (RING4 + s2.bl_add) as f64 / GB, s2.tape_after as f64 / GB, 100.0 * s2.tape_share, s2.left as f64 / GB);
        // and on 96 GB with the same set the tape gets nothing: the ring is short of its target
        let t = split(TOTAL96 - resident - TAPE_RING60, TOTAL96, &need(TAPE_RING60));
        assert_eq!(t.tape_add, 0);
    }

    #[test]
    fn the_29_step_legs_match_what_ran() {
        // six-2, same program family, 29 steps: L29-fill (8 GiB ring) thrashed its land scheme (162 to 219 s a step) with
        // 6,537,826,144 free at the reverse start; L29-r6 (6 GiB) ran (land 1.24 s) with 8,685,309,792 free. Both hold the
        // 13.55 GB scratch and the doubled tape; growth after the start 349,045,532 (L29-r6 censuses, as b8-head).
        let n = |ring: u64| Need { cold_peak: RUC, rev_growth: Some(GROW_B8), bl_held: ring, bl_full: BL_FULL, bl_nowrap: BL_NOWRAP, tape_held: TAPE_2X, tape_step: TAPE_STEP };
        let fill = split(6_537_826_144, TOTAL96, &n(8 << 30));
        let r6 = split(8_685_309_792, TOTAL96, &n(6 << 30));
        assert_eq!((fill.room, r6.room), (0, 0), "neither leaves room the rule would give away");
        println!("L29-fill short of the reserve by {:.2} GB, L29-r6 by {:.2} GB (the 93 GiB budget sits 2.12 GB under the card)",
                 (fill.reserve - 6_537_826_144) as f64 / GB, (r6.reserve - 8_685_309_792) as f64 / GB);
        // L29-r6 with the scratch cut to its need: the ring may grow
        let r6f = split(8_685_309_792 + SCRATCH_SLACK, TOTAL96, &n(6 << 30));
        assert!(r6f.bl_add > 0 && r6f.tape_add == 0);
        println!("L29-r6 with the scratch fitted: room {:.2} GB -> ring {:.2} GB", r6f.room as f64 / GB, ((6u64 << 30) + r6f.bl_add) as f64 / GB);
        // G29-f6 (gradient only): free 27,353,142,376 at the start, but the reverse then made 12,671,944,740 bytes
        // (largest census 85,176,791,996 against 72,504,847,256 at the start) and its land family is 8,300,527,616
        let g = Need { cold_peak: 8_300_527_616, rev_growth: Some(85_176_791_996 - 72_504_847_256), bl_held: 6 << 30, bl_full: BL_FULL, bl_nowrap: BL_NOWRAP, tape_held: 7_516_192_768, tape_step: TAPE_STEP };
        let gs = split(27_353_142_376, TOTAL96, &g);
        assert!(gs.bl_add > 0 && gs.tape_add == 0);
        println!("G29-f6 next run: room {:.2} GB -> ring {:.2} GB, tape {:.2} GB = {:.1} % of step 8",
                 gs.room as f64 / GB, ((6u64 << 30) + gs.bl_add) as f64 / GB, gs.tape_after as f64 / GB, 100.0 * gs.tape_share);
    }

    #[test]
    fn a_first_run_shares_nothing_and_an_unmeasured_ring_takes_an_eighth() {
        let mut n = need(TAPE_RING60);
        n.rev_growth = None;
        let s = split(50 << 30, TOTAL96, &n);
        assert_eq!((s.room, s.bl_add, s.tape_add, s.tape_after), (0, 0, 0, TAPE_RING60));
        let mut m = need(TAPE_RING60);
        m.bl_full = 0; m.bl_nowrap = 0;
        assert_eq!(bl_target(0, 0, TOTAL96), TOTAL96 / 8);
        let s = split(60 << 30, TOTAL96, &m);
        assert_eq!(s.bl_add, (TOTAL96 / 8 - RING4) / GRANULE * GRANULE);
        // an unknown step tape never grows the tape
        let mut t = need(TAPE_RING60);
        t.tape_step = 0;
        let s = split(60 << 30, TOTAL96, &t);
        assert_eq!((s.tape_add, s.tape_share), (0, 0.0));
        assert_eq!(s.bl_add + s.left, s.room);
    }

    #[test]
    fn no_wrap_caps_the_target_when_it_is_the_smaller() {
        // a 3 km-like launch: the whole launch's tape is under the ring that keeps the slots busy
        assert_eq!(bl_target(15 << 30, 3 << 30, TOTAL96), 3 << 30);
        assert_eq!(bl_target(0, 3 << 30, TOTAL96), 3 << 30);
        assert_eq!(bl_target(15 << 30, 0, TOTAL96), 15 << 30);
        let n = Need { cold_peak: RUC, rev_growth: Some(0), bl_held: 1 << 30, bl_full: 15 << 30, bl_nowrap: 3 << 30, tape_held: 8 << 30, tape_step: 20 << 30 };
        let s = split(40 << 30, TOTAL96, &n);
        assert_eq!(s.bl_add, 2 << 30);
        assert_eq!(s.tape_add, (s.room - s.bl_add).min(12 << 30) / GRANULE * GRANULE);
    }

    #[test]
    fn a_full_card_gives_nothing_and_never_underflows() {
        let s = split(RUC, TOTAL96, &need(TAPE_2X));
        assert_eq!((s.room, s.bl_add, s.tape_add, s.left), (0, 0, 0, 0));
        let n = Need { cold_peak: u64::MAX, rev_growth: Some(u64::MAX), bl_held: u64::MAX, bl_full: 0, bl_nowrap: 0, tape_held: u64::MAX, tape_step: 0 };
        let s = split(0, 0, &n);
        assert_eq!((s.room, s.bl_add, s.tape_add), (0, 0, 0));
        // a tape already larger than the step does not grow
        let mut m = need(TAPE_STEP + (1 << 30));
        m.bl_full = RING4;
        assert_eq!(split(90 << 30, TOTAL96, &m).tape_add, 0);
    }

    #[test]
    fn growth_comes_in_whole_granules_only() {
        for extra in [0u64, 1, GRANULE - 1, GRANULE, GRANULE + 1, 7 * GRANULE + 12345] {
            let n = Need { cold_peak: RUC, rev_growth: Some(0), bl_held: RING4, bl_full: RING4, bl_nowrap: 0, tape_held: 1 << 30, tape_step: 100 << 30 };
            let s = split(RUC + head(TOTAL96) + extra, TOTAL96, &n);
            assert_eq!((s.bl_add, s.tape_add % GRANULE), (0, 0));
            assert_eq!(s.tape_add, extra / GRANULE * GRANULE);
            assert_eq!(s.bl_add + s.tape_add + s.left, s.room);
            let n2 = Need { bl_full: 64 << 30, ..n };
            let s2 = split(RUC + head(TOTAL96) + extra, TOTAL96, &n2);
            assert_eq!(s2.bl_add % GRANULE, 0);
            assert_eq!(s2.bl_add + s2.tape_add + s2.left, s2.room);
        }
    }

    #[test]
    fn the_room_file_round_trips_and_never_lowers() {
        let k = Known { bl_full: BL_FULL, bl_nowrap: BL_NOWRAP, tape_step: 94_930_000_000, cold_peak: RUC, rev_growth: Some(9_768_534_016) };
        assert_eq!(parse(&render(&k)), k);
        let k0 = Known { rev_growth: None, ..k };
        assert_eq!(parse(&render(&k0)), k0);
        assert_eq!(parse("bl_full_bytes 5\nbl_full_bytes 9\njunk\ntape_step_bytes x\n"), Known { bl_full: 9, ..Known::default() });
        assert_eq!(parse(""), Known::default());
        assert_eq!(k0.max(Known { rev_growth: Some(3), bl_full: 1, ..Known::default() }), Known { rev_growth: Some(3), ..k0 });
    }

    #[test]
    fn the_launch_rule_keeps_the_largest_cold_family_free() {
        // G29-f6 (gradient only, 96 GB) at a reverse launch: free about 16.1 GB (its budget 7.60 GB = free + 4.37 GB of
        // its own cold blocks - 12 GiB), the land family 8.30 GB, MYNN's own cold blocks 4.37 GB, a lent 6 GiB ring; per
        // slot 8 lanes x 881,376 B arena + 1 MiB store, with need()'s half again (an approximation, for this test only);
        // the ring that keeps s slots busy about s x 15.2 GB / 761.
        let (free, cold_others, p3_cold, ring6) = (16_100_000_000u64, 8_300_527_616u64, 4_373_610_496u64, 6u64 << 30);
        let per_slot = (8 * 881_376 + (1 << 20)) * 3 / 2;
        let nontape = |s: usize| (s.saturating_sub(457) as u64) * per_slot as u64;
        let ring = |s: usize| BL_FULL / 761 * s as u64;
        // the lent ring costs nothing while it holds the ring, a larger ring is made whole (own_held 0)
        let cost = |s: usize| if ring(s) <= ring6 { 0 } else { ring(s) };
        let s = bl_choose(1504, free, TOTAL96, cold_others, p3_cold, 0, &cost, &nontape).unwrap();
        assert!(ring(s) <= ring6, "on this card the slots stay inside the lent ring");
        assert!(!bl_fits(free, TOTAL96, cold_others, p3_cold, nontape(s + 8), cost(s + 8), 0));
        println!("G29-f6-like launch with the 6 GiB lent ring: {s} slots all busy (ring {:.2} GB); 457 ran", ring(s) as f64 / GB);
        // the same launch after AD_CARD_ROOM grew the lent ring by 6 GB at the reverse start (free 6 GB less)
        let ring12 = ring6 + 6_000_000_000;
        let cost12 = |s: usize| if ring(s) <= ring12 { 0 } else { ring(s) };
        let s12 = bl_choose(1504, free - 6_000_000_000, TOTAL96, cold_others, p3_cold, 0, &cost12, &nontape).unwrap();
        println!("  with a {:.2} GB lent ring: {s12} slots all busy (ring {:.2} GB)", ring12 as f64 / GB, ring(s12) as f64 / GB);
        assert!(s12 >= s);
        // the reserve is never given away: with free equal to what must stay free nothing beyond the held ring fits
        let keep = cold_others.max(p3_cold) + head(TOTAL96);
        assert!(bl_fits(keep, TOTAL96, cold_others, p3_cold, 0, ring6, ring6));
        assert!(!bl_fits(keep, TOTAL96, cold_others, p3_cold, 0, ring6 + (1 << 20), ring6));
    }

    // 60 km tape: a test machine card 1, a test program, legs st-R1 and st-R2 (AD_P3_STRIPE=1 AD_CARD_ROOM=1
    // AD_P3_ROOM_RULE=1 AD_CARD_RELEASE=1 AD_REV_SCRATCH_FIT=max, one new room file), read in their stderr
    // (window_tape_sized card_room, room_rule): free at the reverse start 22,623,269,664 in both; st-R1 wrote rev_growth
    // 11,886,527,260; every one of the 48 boundary-layer launches of both legs ran a ring of its own of 10,956,570,624
    // bytes (ring_lent false), the lent arena 6,442,450,944 beside it.
    const FREE_STR: u64 = 22_623_269_664;
    const GROW_STR1: u64 = 11_886_527_260;
    const OWN_STR1: u64 = 10_956_570_624;
    fn need_str(rev_growth: u64) -> Need {
        Need { cold_peak: 9_118_416_896, rev_growth: Some(rev_growth), bl_held: 6_442_450_944, bl_full: 28_742_372_088, bl_nowrap: 67_228_878_072,
               tape_held: 11_669_602_304, tape_step: 89_362_386_480 }
    }

    #[test]
    fn the_ring_of_its_own_is_not_the_reverses_growth() {
        // st-R2 as it ran: the receipt's reserve, room and ring growth, to the byte
        let s = split(FREE_STR, TOTAL96, &need_str(GROW_STR1));
        assert_eq!((s.reserve, s.room, s.bl_add, s.tape_add, s.left), (22_565_225_244, 58_044_420, 56_623_104, 0, 1_421_316));
        // the growth with the ring of its own taken out (the receipt's own_ring_bytes: what min(nodes, bars) holds, so this
        // is at most the growth the fixed program measures)
        let d0 = 75_000_000_000u64;
        let g = growth(d0 + GROW_STR1, d0, OWN_STR1, 0);
        assert_eq!(g, 929_956_636);
        assert_eq!(growth(d0 + GROW_STR1, d0, 0, 0), GROW_STR1, "no ring of its own: the growth as before");
        assert_eq!(growth(d0 - 5, d0, OWN_STR1, 0), 0, "never underflows");
        assert_eq!(growth(d0 + 5, d0, OWN_STR1, 0), 0);
        // a ring of its own already held at the reverse start is inside d0: only what it adds after the start comes out
        assert_eq!(growth(d0 + GROW_STR1, d0, OWN_STR1, OWN_STR1), GROW_STR1, "a ring held at the start is not taken out twice");
        assert_eq!(growth(d0 + GROW_STR1, d0, OWN_STR1, 4_000_000_000), GROW_STR1 - (OWN_STR1 - 4_000_000_000));
        assert_eq!(growth(d0 + GROW_STR1, d0, 1, OWN_STR1), GROW_STR1, "a ring that shrank adds nothing");
        // the next run's share-out with that growth: the room goes to the lent ring (still short of its 28.7 GB target)
        let f = split(FREE_STR, TOTAL96, &need_str(g));
        assert_eq!(f.reserve, 9_118_416_896 + 929_956_636 + TOTAL96 / 64);
        assert_eq!(f.room, 11_014_615_044);
        assert_eq!(f.bl_add, 11_014_242_304);
        assert_eq!(f.tape_add, 0);
        let ring = 6_442_450_944 + f.bl_add;
        assert_eq!(ring, 17_456_693_248);
        assert!(ring > OWN_STR1 + 6_000_000_000, "one ring larger than the ring of its own that ran, from the same card");
        println!("st-R2 as it ran: room {} -> lent ring {}; with the ring of its own not counted as growth: growth {} room {} -> lent ring {} \
                  (the ring that ran: {} of its own, {} lent and idle)", s.room, 6_442_450_944u64 + s.bl_add, g, f.room, ring, OWN_STR1, 6_442_450_944u64);
    }
}
