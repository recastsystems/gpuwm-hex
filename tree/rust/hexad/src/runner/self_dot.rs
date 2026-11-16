//! central hunt-3km-slot (2026-11-11, the central line): the arithmetic of the self form of the window dot
//! (AD_DOT_SELF=1, Window::self_y), moved here word for word so it can be tested on the processor, and two named
//! diagnostics that are OFF by default. Neither changes what a run calls valid: valid_derivative still reads the
//! unfloored dot (dots[].rel_aligned) against the program's bar (DotBar: 4 float32 roundings of the dot by default,
//! AD_DOT_BAR), and a run with either switch says so in window.json dot_form.
//!
//! * AD_DOT_SELF_FLOOR=<k>:<unit>=<c>[,<unit>=<c>...] (needs AD_DOT_SELF=1; one process). Unit u of the self dot votes
//!   v_u = W_u * sum(S_u^2). The unfloored form sets W_u = 1 / sum(S_u^2), so every unit votes 1 however much its
//!   backward cancels. c_u is the unit's measured absolute backward terms per vote: abs_terms_ad of a run with
//!   AD_DOT_SELF_ONLY=<u> and AD_DOT_ALLOC=1 (whose vote is 1; 3 km unit 184 read 426.46). The floored weight is
//!       W_u = 1 / max(sum(S_u^2), sum(S_u^2) * c_u / k),   so   v_u = min(1, k / c_u),
//!   and no unit brings more than k votes of absolute terms, so no more than about k float32 roundings (k * 2^-24
//!   votes) of rounding. It is the tester's floor W = 1 / max(sum S^2, (k 2^-24 sum|terms|)^2) with the unit's
//!   sum |terms| taken from a measured run: that number is a reverse-side quantity and a single pass cannot know it
//!   before its reverse. Units not named are not floored. The floored dot costs one more reverse (first seed only),
//!   seeded with the floored y; it is printed as window.json self_floor beside the unfloored dot, and every unit's
//!   decision is in self_dot.floor.rows. Its cost: an error inside a floored unit is shrunk by v_u (test
//!   floor_shrinks_an_error_inside_a_floored_unit), which is one reason it is never the valid reading.
//! * AD_DOT_ROUNDINGS=1: every dot reading also prints miss_roundings = (x_MTy - y_Mx) / (2^-24 * abs_terms), the
//!   signed miss in float32 roundings of the dot's absolute terms (needs AD_DOT_ALLOC=1 for abs_terms; null
//!   otherwise). A unit with a large c_u inflates abs_terms and so shrinks this number for every error, a real one
//!   included (test roundings_alone_can_hide_a_plant_behind_a_cancelling_unit): read it beside a planted-error row.
use serde_json::{json, Value};
use std::collections::BTreeMap;

/// One float32 rounding, 2^-24.
pub const ROUNDING: f64 = 1.0 / 16_777_216.0;

#[derive(Clone, Debug, PartialEq)]
pub struct FloorSpec { pub k: f64, pub c: BTreeMap<u64, f64> }

impl FloorSpec {
    pub fn parse(spec: &str) -> Result<FloorSpec, String> {
        let want = "AD_DOT_SELF_FLOOR wants <k>:<unit>=<c>[,<unit>=<c>...] (c = abs_terms_ad of an AD_DOT_SELF_ONLY=<unit> AD_DOT_ALLOC=1 run)";
        let (k, list) = spec.split_once(':').ok_or(want)?;
        let k: f64 = k.trim().parse().map_err(|_| format!("AD_DOT_SELF_FLOOR: k {k:?} is not a number; {want}"))?;
        if !(k > 0.0 && k.is_finite()) { return Err(format!("AD_DOT_SELF_FLOOR: k must be a positive finite number, not {k}")); }
        let mut c = BTreeMap::new();
        for it in list.split(',').map(str::trim).filter(|x| !x.is_empty()) {
            let (u, v) = it.split_once('=').ok_or(format!("AD_DOT_SELF_FLOOR: {it:?}: {want}"))?;
            let u: u64 = u.trim().parse().map_err(|_| format!("AD_DOT_SELF_FLOOR: unit {u:?} is not a number"))?;
            let v: f64 = v.trim().parse().map_err(|_| format!("AD_DOT_SELF_FLOOR: c {v:?} of unit {u} is not a number"))?;
            if !(v > 0.0 && v.is_finite()) { return Err(format!("AD_DOT_SELF_FLOOR: c of unit {u} must be positive and finite, not {v}")); }
            if c.insert(u, v).is_some() { return Err(format!("AD_DOT_SELF_FLOOR: unit {u} is named twice")); }
        }
        if c.is_empty() { return Err(format!("AD_DOT_SELF_FLOOR names no unit; {want}")); }
        Ok(FloorSpec { k, c })
    }
    pub fn from_env() -> Result<Option<FloorSpec>, String> {
        match std::env::var("AD_DOT_SELF_FLOOR") { Ok(v) if !v.trim().is_empty() => Ok(Some(Self::parse(&v)?)), _ => Ok(None) }
    }
    /// (measured c_u if the unit is named, its vote v_u = min(1, k / c_u); 1 when not named).
    pub fn vote(&self, unit: u64) -> (Option<f64>, f64) {
        match self.c.get(&unit) { Some(&c) => (Some(c), (self.k / c).min(1.0)), None => (None, 1.0) }
    }
}

pub fn roundings_on() -> bool { std::env::var("AD_DOT_ROUNDINGS").as_deref() == Ok("1") }

/// The signed miss x_MTy - y_Mx in float32 roundings of the absolute terms; None without terms.
pub fn roundings(y_mx: f64, x_mty: f64, abs_terms: f64) -> Option<f64> {
    (abs_terms > 0.0 && abs_terms.is_finite() && y_mx.is_finite() && x_mty.is_finite()).then(|| (x_mty - y_mx) / (ROUNDING * abs_terms))
}

/// float32 epsilon, 2^-23: one single-precision rounding of the dot, relative to its size. The window check's verdict
/// is stated in these roundings (hexcore.dot_bar, the door table's thresholds.self_dot_bar): reading
/// r = (x_MTy - y_Mx) / (2^-23 * max(|y_Mx|, |x_MTy|)), both sums binary64 (win_dot2_f32 widens each word), so
/// |r| = rel / 2^-23 for rel = |a - b| / max(|a|, |b|), the self form's rel_aligned.
pub const EPS32: f64 = 1.0 / 8_388_608.0;
/// The bar a bare runner judges each dot by when the door names none (AD_DOT_BAR unset): 4 roundings, the door
/// table's thresholds.self_dot_bar.each_within_roundings (a door test reads both and fails if they differ).
pub const DEFAULT_BAR_ROUNDINGS: f64 = 4.0;

/// The signed reading in float32 roundings of the dot's size (the verdict's unit); None when a side is not finite.
pub fn dot_roundings(y_mx: f64, x_mty: f64) -> Option<f64> {
    if !(y_mx.is_finite() && x_mty.is_finite()) { return None; }
    if y_mx == x_mty { return Some(0.0); }
    Some((x_mty - y_mx) / (EPS32 * y_mx.abs().max(x_mty.abs())))
}

/// The bar valid_derivative reads each dot against, from AD_DOT_BAR (the door sets it from its table on every window):
/// `roundings:<k>` (each dot within k roundings of the dot; default k = DEFAULT_BAR_ROUNDINGS) or `relative:<r>` (the
/// strict setting, 1e-7). THE BREAKAGE THIS PREVENTS: a hidden 1e-7 here, 0.84 of one rounding, read valid false on
/// rounding alone (the released build: 3 km seed 1 at 1.406e-07, the README command at 2.046e-07 on one install and
/// 8.575e-08 on another) while the door judged by another number.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DotBar { Roundings(f64), Relative(f64) }

impl DotBar {
    pub fn parse(spec: &str) -> Result<DotBar, String> {
        let want = "AD_DOT_BAR wants roundings:<k> or relative:<r>";
        let (kind, v) = spec.trim().split_once(':').ok_or(want)?;
        let v: f64 = v.trim().parse().map_err(|_| format!("AD_DOT_BAR: {v:?} is not a number; {want}"))?;
        if !(v > 0.0 && v.is_finite()) { return Err(format!("AD_DOT_BAR: {v} must be positive and finite")); }
        match kind.trim() { "roundings" => Ok(DotBar::Roundings(v)), "relative" => Ok(DotBar::Relative(v)), _ => Err(want.into()) }
    }
    pub fn from_env() -> Result<DotBar, String> {
        match std::env::var("AD_DOT_BAR") { Ok(v) if !v.trim().is_empty() => Self::parse(&v), _ => Ok(DotBar::Roundings(DEFAULT_BAR_ROUNDINGS)) }
    }
    /// The largest rel_aligned a dot may read (the two settings in one number: k roundings is k * 2^-23 relative).
    pub fn rel(&self) -> f64 { match *self { DotBar::Roundings(k) => k * EPS32, DotBar::Relative(r) => r } }
    pub fn passes(&self, rel_aligned: f64) -> bool { rel_aligned.is_finite() && rel_aligned <= self.rel() * (1.0 + 1e-12) }
    pub fn receipt(&self) -> Value {
        match *self {
            DotBar::Roundings(k) => json!({"setting":"roundings","each_within_roundings":k,"rel":self.rel(),
                "rounding":"2^-23 * max(|y_Mx|, |x_MTy|) for the self form; 2^-23 * max(abs_terms_tl, abs_terms_ad) for a random y (dots[].bar_rel, F14); both sums binary64"}),
            DotBar::Relative(r) => json!({"setting":"strict","rel":r,"each_within_roundings":r / EPS32}),
        }
    }
}

/// A word of y under the floor: v times the unfloored word, in binary64, rounded once.
pub fn floor_word(y: f32, v: f64) -> f32 { if y == 0.0 { y } else { (y as f64 * v) as f32 } }

/// y (bytes) with each floored unit (first byte, words, v) scaled word by word as floor_word does.
pub fn apply_floor(y: &mut [u8], scale: &[(usize, usize, f64)]) {
    for &(lo, n, v) in scale {
        for k in 0..n {
            let o = lo + 4 * k;
            let w = f32::from_le_bytes(y[o..o + 4].try_into().unwrap());
            y[o..o + 4].copy_from_slice(&floor_word(w, v).to_le_bytes());
        }
    }
}

pub struct FloorOut {
    pub k: f64,
    /// [unit, weighted words, sum S^2, c_u (null: not named), v_u, unfloored term of y_Mx, floored term]
    pub rows: Vec<Value>,
    /// (first byte, words, v) of every unit with v < 1: apply_floor turns the unfloored y into the floored one.
    pub scale: Vec<(usize, usize, f64)>,
    /// <y_floor, M x>, the units' floored terms summed in unit order.
    pub y_mx: f64,
    /// units the switch names that cast no vote in this dot
    pub named_not_voting: Vec<u64>,
}

pub struct SelfOut {
    pub y: Vec<u8>, pub p: f64, pub pcomp: (f64, f64), pub used: u64, pub skipped: u64, pub words: u64,
    /// (unit, weighted words, sum S^2), largest sum first
    pub per: Vec<(u64, u64, f64)>,
    /// (unit, first byte of its words in y, bytes, its own term of <y, M x>)
    pub parts: Vec<(u64, usize, usize, f64)>,
    /// AD_DOT_SELF_DUMP lines (window_self_unit_words), in unit order
    pub dumps: Vec<Value>,
    pub floor: Option<FloorOut>,
}

/// The self y over the units (number, first byte in the arena, words): s is the tangent result, g the standard y
/// direction (only its nonzero words are weighted), prim the primal at the window end when it has the arena layout.
/// With floor = None every number is the one Window::self_y computed before this module (test
/// no_switch_is_bit_identical_to_the_old_loop), and the one the wide candidate's threaded self_y computes (test
/// joined_is_bit_identical_to_wide_and_to_the_3km_function).
///
/// central wide-tools join (2026-11-11): the wide candidate (ad/central-wide b2e8ebf9, "speed outside") runs the two
/// passes over each unit's words on worker threads, one unit per thread, and folds the units in unit order. THE
/// BREAKAGE THAT CHANGE FIXED: both passes ran on one thread (10.5 GB of tangent and of weights at 3 km: most of the
/// 17.1 s before the mark fill_y, a test machine win12f BUDGET.md). It is carried here so the move into this module does not
/// drop it. A unit's sums are taken in its own word order by one thread, so y, y_Mx and its compensated form keep
/// their bits. Units that overlap (possible only with AD_DOT_SELF_FIELDS) keep one thread, in unit order. The floor's
/// term of a unit is taken from that unit's own y words just after they are written, as the one-thread loop read them.
#[allow(clippy::too_many_arguments)]
pub fn self_weights(s: &[u8], g: &[u8], units: &[(u64, usize, usize)], only: Option<&[u64]>, dump: Option<u64>, prim: Option<&[u8]>,
                    floor: Option<&FloorSpec>) -> SelfOut {
    self_weights_pow2(s, g, units, only, dump, prim, floor, 0)
}

/// self_weights with the 60 km hunt's AD_DOT_SELF_YPOW2=k (ad/central-hunt-60-repeat e708d47a): every word of y is
/// rev_digest::self_y_word(S, sum S^2, k), y times 2^k exactly; k = 0 is self_weights to the bit (rev_digest's own
/// test, and joined_ypow2_is_bit_identical_to_the_60km_loop here). Window::self_y passes rev_digest::ypow2().
#[allow(clippy::too_many_arguments)]
pub fn self_weights_pow2(s: &[u8], g: &[u8], units: &[(u64, usize, usize)], only: Option<&[u64]>, dump: Option<u64>, prim: Option<&[u8]>,
                         floor: Option<&FloorSpec>, ypow2: i32) -> SelfOut {
    let threads = if units_disjoint(units) { std::thread::available_parallelism().map(|t| t.get()).unwrap_or(4).min(64) } else { 1 };
    self_weights_threads(s, g, units, only, dump, prim, floor, ypow2, threads)
}

/// True when no two units' byte ranges meet (the wide candidate's test, over every unit, AD_DOT_SELF_ONLY or not).
pub fn units_disjoint(units: &[(u64, usize, usize)]) -> bool {
    let mut v: Vec<(usize, usize)> = units.iter().map(|u| (u.1, u.1 + 4 * u.2)).collect(); v.sort_unstable(); v.windows(2).all(|x| x[0].1 <= x[1].0)
}

/// self_weights_pow2 with a set number of worker threads. More than one thread is sound only when units_disjoint(units).
#[allow(clippy::too_many_arguments)]
pub fn self_weights_threads(s: &[u8], g: &[u8], units: &[(u64, usize, usize)], only: Option<&[u64]>, dump: Option<u64>, prim: Option<&[u8]>,
                            floor: Option<&FloorSpec>, ypow2: i32, threads: usize) -> SelfOut {
    assert!(threads <= 1 || units_disjoint(units), "self_weights_threads: overlapping units need one thread");
    let mut y = vec![0u8; s.len()];
    let (mut p, mut used, mut skipped, mut words) = (0f64, 0u64, 0u64, 0u64);
    let mut pcomp = (0f64, 0f64);
    let mut per: Vec<(u64, u64, f64)> = Vec::new();
    let mut parts: Vec<(u64, usize, usize, f64)> = Vec::new();
    let mut dumps: Vec<Value> = Vec::new();
    let (mut frows, mut fscale, mut fy, mut fvoted) = (Vec::new(), Vec::new(), 0f64, Vec::new());
    let ybase = y.as_mut_ptr() as usize;
    // (0 = not used, 1 = skipped, 2 = used; its term of <y, M x>; the same compensated; weighted words; sum of squares;
    //  its floored term (= its term when the floor is off or does not lower its vote))
    type UnitRow = (u8, f64, (f64, f64), u64, f64, f64);
    let one = move |id: &u64, lo: usize, n: usize| -> UnitRow {
        const NO: UnitRow = (0, 0.0, (0.0, 0.0), 0, 0.0, 0.0);
        if only.is_some_and(|o| !o.contains(id)) { return NO; }
        if lo + 4 * n > s.len() { return (1, 0.0, (0.0, 0.0), 0, 0.0, 0.0); }
        let rd = |b: &[u8], k: usize| f32::from_le_bytes(b[lo + 4 * k..lo + 4 * k + 4].try_into().unwrap());
        let (mut ss, mut c) = (0f64, 0u64);
        for k in 0..n { let (sv, gv) = (rd(s, k), rd(g, k)); if gv != 0.0 && sv.is_finite() && sv != 0.0 { ss += sv as f64 * sv as f64; c += 1; } }
        if c == 0 || !(ss > 0.0) || !ss.is_finite() { return NO; }
        let (mut pf, mut ok) = (0f64, true); let mut pfc = (0f64, 0f64);
        // this unit's own words of y: no other unit's range meets them (disjoint), or this is the only thread
        let yu = unsafe { std::slice::from_raw_parts_mut((ybase + lo) as *mut u8, 4 * n) };
        for k in 0..n { let (sv, gv) = (rd(s, k), rd(g, k)); if gv != 0.0 && sv.is_finite() && sv != 0.0 {
            let yv = crate::rev_digest::self_y_word(sv, ss, ypow2);
            if !yv.is_finite() { ok = false; break; }
            yu[4 * k..4 * k + 4].copy_from_slice(&yv.to_le_bytes()); pf += yv as f64 * sv as f64; super::neumaier(&mut pfc, yv as f64 * sv as f64); } }
        if !ok { for b in yu.iter_mut() { *b = 0; } return (1, 0.0, (0.0, 0.0), 0, 0.0, 0.0); }
        let mut pfl = pf;
        if let Some(fs) = floor { let v = fs.vote(*id).1; if v < 1.0 {
            pfl = 0.0;
            for k in 0..n { let yv = f32::from_le_bytes(yu[4 * k..4 * k + 4].try_into().unwrap()); if yv != 0.0 { pfl += floor_word(yv, v) as f64 * rd(s, k) as f64; } }
        } }
        (2, pf, pfc, c, ss, pfl)
    };
    let res: Vec<UnitRow> = {
        let next = std::sync::atomic::AtomicUsize::new(0);
        let slots: Vec<std::sync::Mutex<Option<UnitRow>>> = (0..units.len()).map(|_| std::sync::Mutex::new(None)).collect();
        let (next, slots, one) = (&next, &slots, &one);
        std::thread::scope(|sc| {
            for _ in 0..threads.min(units.len()).max(1) {
                sc.spawn(move || loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= units.len() { break; }
                    let v = one(&units[i].0, units[i].1, units[i].2);
                    *slots[i].lock().unwrap() = Some(v);
                });
            }
        });
        slots.iter().map(|m| m.lock().unwrap().take().unwrap()).collect()
    };
    for ((id, lo, n), r) in units.iter().zip(&res) {
        let (lo, n) = (*lo, *n);
        if r.0 == 1 { skipped += 1; }
        if r.0 != 2 { continue; }
        let (pf, pfc, c, ss, pfl) = (r.1, r.2, r.3, r.4, r.5);
        let rd = |b: &[u8], k: usize| f32::from_le_bytes(b[lo + 4 * k..lo + 4 * k + 4].try_into().unwrap());
        p += pf; super::neumaier(&mut pcomp, pfc.0 + pfc.1); used += 1; words += c; per.push((*id, c, ss)); parts.push((*id, lo, 4 * n, pf));
        if let Some(nmax) = dump { if c <= nmax {
            let rows: Vec<Value> = (0..n).filter_map(|k| { let (sv, gv) = (rd(s, k), rd(g, k));
                if gv != 0.0 && sv.is_finite() && sv != 0.0 { Some(json!([k, sv, gv, prim.map(|b| rd(b, k))])) } else { None } }).collect();
            dumps.push(json!({"event":"window_self_unit_words","unit":id,"first_byte":lo,"words":c,"primal_layout":prim.is_some(),"rows":rows}));
        } }
        if let Some(fs) = floor {
            let (cu, v) = fs.vote(*id);
            if v < 1.0 { fscale.push((lo, n, v)); }
            fy += pfl; fvoted.push(*id);
            frows.push(json!([id, c, ss, cu, v, pf, pfl]));
        }
    }
    per.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    let floor = floor.map(|fs| FloorOut { k: fs.k, rows: frows, scale: fscale, y_mx: fy,
                                         named_not_voting: fs.c.keys().copied().filter(|u| !fvoted.contains(u)).collect() });
    SelfOut { y, p, pcomp, used, skipped, words, per, parts, dumps, floor }
}

/// window.json self_dot.floor: the switch, k, and every voting unit's decision.
pub fn floor_receipt(f: &FloorOut) -> Value {
    json!({"switch":"AD_DOT_SELF_FLOOR","k":f.k,"rule":"v_u = min(1, k / c_u); W_u = v_u / sum(S_u^2); c_u = measured abs terms per vote (AD_DOT_SELF_ONLY=<u> AD_DOT_ALLOC=1 abs_terms_ad); a unit not named keeps v_u = 1",
           "y_Mx_floored":f.y_mx,"floored_units":f.scale.len(),"named_not_voting":f.named_not_voting,
           "fields":"[unit, weighted words, sum S^2, c_u (null: not named), v_u, unfloored term of y_Mx, floored term]","rows":f.rows,
           "reading":"window.json self_floor (one more reverse, first seed only); never the valid reading"})
}

/// window.json dot_form: which form of the dot this run read, and that the valid rule is unchanged.
pub fn form_receipt(self_on: bool, fields: Option<&str>, only: Option<&str>, floor: Option<&FloorSpec>, roundings: bool) -> Value {
    let form = if !self_on { "standard: the random y direction (no AD_DOT_SELF)" }
               else if fields.is_some() { "self: one unit per prognostic field at the window end (AD_DOT_SELF=1; the package's field map, or AD_DOT_SELF_FIELDS)" }
               else { "self: one unit per memory allocation (AD_DOT_SELF=1 with AD_DOT_SELF_UNITS=pool, or no field map at the window end)" };
    json!({"form":form,"self_fields":fields,"self_only":only,
           "floor":floor.map(|f| json!({"switch":"AD_DOT_SELF_FLOOR","k":f.k,"c":f.c,"reading":"self_floor, beside the unfloored dot"})),
           "roundings":roundings,
           "valid_rule":"unchanged: valid_derivative reads dots[].rel_aligned of the dot as computed without AD_DOT_SELF_FLOOR, against the program's bar (DotBar, AD_DOT_BAR; valid_derivative_rule.dot_bar); AD_DOT_SELF_FLOOR and AD_DOT_ROUNDINGS only add readings"})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bar_is_four_roundings_of_the_dot_by_default_and_strict_on_request() {
        let d = DotBar::parse("roundings:4").unwrap();
        assert_eq!(d, DotBar::Roundings(DEFAULT_BAR_ROUNDINGS));
        assert!((d.rel() - 4.0 / 8_388_608.0).abs() < 1e-20);
        // recorded readings (card runs, 15 Nov 2026): rounding passes, real terms fail
        for rel in [3.2041e-08, 3.4309e-08, 1.4060e-07, 4.5157e-08, 2.9965e-08, 5.4327e-08, 8.575e-08, 2.046e-07] { assert!(d.passes(rel), "{rel}"); }
        for rel in [8.406e-05, 2.211e-05, 1.532e-05, 3.416e-05, 1.92e-06, 2.25e-06] { assert!(!d.passes(rel), "{rel}"); }
        let s = DotBar::parse("relative:1e-7").unwrap();
        assert!(s.passes(8.575e-08) && !s.passes(2.046e-07) && !s.passes(1.4060e-07));
        assert!(DotBar::parse("roundings").is_err() && DotBar::parse("percent:4").is_err() && DotBar::parse("roundings:-1").is_err());
        assert!(!d.passes(f64::NAN));
        // the reading is signed and is rel / 2^-23 in size
        let r = dot_roundings(12.0, 12.0 * (1.0 - 1.4060e-07)).unwrap();
        assert!(r < 0.0 && (r.abs() - 1.4060e-07 / EPS32).abs() < 1e-6, "{r}");
        assert_eq!(dot_roundings(5.0, 5.0), Some(0.0));
        assert!(dot_roundings(1.0, f64::INFINITY).is_none());
    }

    // ---- the loop of Window::self_y at fold 9 (b0da827a737d), copied verbatim but for its inputs and outputs ----
    #[allow(clippy::type_complexity)]
    fn old_self_loop(s: &[u8], g: &[u8], units: &[(u64, usize, usize)], only: Option<Vec<u64>>, dump: Option<u64>, prim: Option<&[u8]>)
                     -> (Vec<u8>, f64, (f64, f64), u64, u64, u64, Vec<(u64, u64, f64)>, Vec<(u64, usize, usize, f64)>, Vec<Value>) {
        let mut y = vec![0u8; s.len()];
        let (mut p, mut used, mut skipped, mut words) = (0f64, 0u64, 0u64, 0u64);
        let mut pcomp = (0f64, 0f64);
        let mut per: Vec<(u64, u64, f64)> = Vec::new();
        let mut parts: Vec<(u64, usize, usize, f64)> = Vec::new();
        let mut dumps = Vec::new();
        for (id, lo, n) in units {
            let (lo, n) = (*lo, *n);
            if only.as_ref().is_some_and(|o| !o.contains(id)) { continue; }
            if lo + 4 * n > s.len() { skipped += 1; continue; }
            let rd = |b: &[u8], k: usize| f32::from_le_bytes(b[lo + 4 * k..lo + 4 * k + 4].try_into().unwrap());
            let (mut ss, mut c) = (0f64, 0u64);
            for k in 0..n { let (sv, gv) = (rd(s, k), rd(g, k)); if gv != 0.0 && sv.is_finite() && sv != 0.0 { ss += sv as f64 * sv as f64; c += 1; } }
            if c == 0 || !(ss > 0.0) || !ss.is_finite() { continue; }
            let (mut pf, mut ok) = (0f64, true); let mut pfc = (0f64, 0f64);
            for k in 0..n { let (sv, gv) = (rd(s, k), rd(g, k)); if gv != 0.0 && sv.is_finite() && sv != 0.0 {
                let yv = (sv as f64 / ss) as f32;
                if !yv.is_finite() { ok = false; break; }
                y[lo + 4 * k..lo + 4 * k + 4].copy_from_slice(&yv.to_le_bytes()); pf += yv as f64 * sv as f64; super::super::neumaier(&mut pfc, yv as f64 * sv as f64); } }
            if !ok { for b in &mut y[lo..lo + 4 * n] { *b = 0; } skipped += 1; continue; }
            p += pf; super::super::neumaier(&mut pcomp, pfc.0 + pfc.1); used += 1; words += c; per.push((*id, c, ss)); parts.push((*id, lo, 4 * n, pf));
            if let Some(nmax) = dump { if c <= nmax {
                let rows: Vec<Value> = (0..n).filter_map(|k| { let (sv, gv) = (rd(s, k), rd(g, k));
                    if gv != 0.0 && sv.is_finite() && sv != 0.0 { Some(json!([k, sv, gv, prim.map(|b| rd(b, k))])) } else { None } }).collect();
                dumps.push(json!({"event":"window_self_unit_words","unit":id,"first_byte":lo,"words":c,"primal_layout":prim.is_some(),"rows":rows}));
            } }
        }
        per.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        (y, p, pcomp, used, skipped, words, per, parts, dumps)
    }

    struct Lcg(u64);
    impl Lcg {
        /// uniform in [-1, 1)
        fn next(&mut self) -> f64 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (self.0 >> 11) as f64 / (1u64 << 52) as f64 - 1.0 }
    }
    fn bytes(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
    fn words(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }

    #[test]
    fn no_switch_is_bit_identical_to_the_old_loop() {
        let mut r = Lcg(7);
        let n = 4096usize;
        let mut s: Vec<f32> = (0..n).map(|_| (r.next() * 10f64.powi((r.next() * 20.0) as i32)) as f32).collect();
        let mut g: Vec<f32> = (0..n).map(|_| if r.next() < -0.6 { 0.0 } else { r.next() as f32 * 1e-20 }).collect();
        s[5] = f32::NAN; s[6] = f32::INFINITY; s[7] = 0.0;
        // unit 9 (words 3000..3002): one denormal tangent word, so y = S / sum(S^2) overflows float32 (the !ok branch)
        s[3000] = 1e-40; g[3000] = 1.0; s[3001] = 0.0; s[3002] = 0.0;
        // unit 11: no weighted word
        for k in 3100..3110 { g[k] = 0.0; }
        let (sb, gb) = (bytes(&s), bytes(&g));
        let prim = bytes(&(0..n).map(|k| k as f32).collect::<Vec<_>>());
        let units: Vec<(u64, usize, usize)> = vec![(1, 0, 1000), (2, 4000, 1), (3, 4004, 2), (4, 4012, 900), (9, 12000, 3), (11, 12400, 10),
                                                   (5, 4 * 2000, 1000), (6, 4 * 4090, 10) /* runs past the end: skipped */, (7, 4 * 3500, 590)];
        for (only, dump, pr) in [(None, None, None), (Some(vec![2u64, 4, 9]), Some(4u64), Some(&prim[..])), (None, Some(2), None)] {
            let old = old_self_loop(&sb, &gb, &units, only.clone(), dump, pr);
            for fl in [None, Some(FloorSpec { k: 4.0, c: BTreeMap::from([(99u64, 1e3)]) })] {
                let new = self_weights(&sb, &gb, &units, only.as_deref(), dump, pr, fl.as_ref());
                assert_eq!(new.y, old.0);
                assert_eq!(new.p.to_bits(), old.1.to_bits());
                assert_eq!((new.pcomp.0.to_bits(), new.pcomp.1.to_bits()), (old.2 .0.to_bits(), old.2 .1.to_bits()));
                assert_eq!((new.used, new.skipped, new.words), (old.3, old.4, old.5));
                assert_eq!(new.per.iter().map(|x| (x.0, x.1, x.2.to_bits())).collect::<Vec<_>>(), old.6.iter().map(|x| (x.0, x.1, x.2.to_bits())).collect::<Vec<_>>());
                assert_eq!(new.parts.iter().map(|x| (x.0, x.1, x.2, x.3.to_bits())).collect::<Vec<_>>(), old.7.iter().map(|x| (x.0, x.1, x.2, x.3.to_bits())).collect::<Vec<_>>());
                assert_eq!(new.dumps, old.8);
                if let Some(f) = &new.floor { assert!(f.scale.is_empty()); assert_eq!(f.named_not_voting, vec![99]); assert_eq!(f.y_mx.to_bits(), new.parts.iter().fold(0f64, |a, x| a + x.3).to_bits()); }
            }
            if only.is_none() { assert!(old.4 >= 2, "the test must reach the skipped and the overflow branches"); }
        }
    }

    // ---- the loop of Window::self_y in the wide candidate (ad/central-wide b2e8ebf9, window.rs blob 1f837d1f), copied
    // verbatim but for its inputs and outputs (AD_DOT_SELF_DUMP comes in as `dump`, its lines are collected, not
    // printed, and the timing mark is left out) ----
    #[allow(clippy::type_complexity)]
    fn wide_self_loop(s: Vec<u8>, g: Vec<u8>, units: Vec<(u64, usize, usize)>, only: Option<Vec<u64>>, dump: Option<u64>, prim: Option<&[u8]>)
                      -> (Vec<u8>, f64, (f64, f64), u64, u64, u64, Vec<(u64, u64, f64)>, Vec<(u64, usize, usize, f64)>, Vec<Value>) {
        use super::super::neumaier;
        let mut y = vec![0u8; s.len()];
        let (mut p, mut used, mut skipped, mut words) = (0f64, 0u64, 0u64, 0u64);
        let mut pcomp = (0f64, 0f64);
        let mut per: Vec<(u64, u64, f64)> = Vec::new();
        let mut parts: Vec<(u64, usize, usize, f64)> = Vec::new();
        let mut dumps = Vec::new();
        let disjoint = { let mut v: Vec<(usize, usize)> = units.iter().map(|u| (u.1, u.1 + 4 * u.2)).collect(); v.sort_unstable(); v.windows(2).all(|x| x[0].1 <= x[1].0) };
        let threads = if disjoint { std::thread::available_parallelism().map(|t| t.get()).unwrap_or(4).min(64) } else { 1 };
        let ybase = y.as_mut_ptr() as usize;
        let (sref, gref, onlyr) = (&s[..], &g[..], &only);
        type UnitRow = (u8, f64, (f64, f64), u64, f64);
        let one = move |id: &u64, lo: usize, n: usize| -> UnitRow {
            const NO: UnitRow = (0, 0.0, (0.0, 0.0), 0, 0.0);
            if onlyr.as_ref().is_some_and(|o| !o.contains(id)) { return NO; }
            if lo + 4 * n > sref.len() { return (1, 0.0, (0.0, 0.0), 0, 0.0); }
            let rd = |b: &[u8], k: usize| f32::from_le_bytes(b[lo + 4 * k..lo + 4 * k + 4].try_into().unwrap());
            let (mut ss, mut c) = (0f64, 0u64);
            for k in 0..n { let (sv, gv) = (rd(sref, k), rd(gref, k)); if gv != 0.0 && sv.is_finite() && sv != 0.0 { ss += sv as f64 * sv as f64; c += 1; } }
            if c == 0 || !(ss > 0.0) || !ss.is_finite() { return NO; }
            let (mut pf, mut ok) = (0f64, true); let mut pfc = (0f64, 0f64);
            let yu = unsafe { std::slice::from_raw_parts_mut((ybase + lo) as *mut u8, 4 * n) };
            for k in 0..n { let (sv, gv) = (rd(sref, k), rd(gref, k)); if gv != 0.0 && sv.is_finite() && sv != 0.0 {
                let yv = (sv as f64 / ss) as f32;
                if !yv.is_finite() { ok = false; break; }
                yu[4 * k..4 * k + 4].copy_from_slice(&yv.to_le_bytes()); pf += yv as f64 * sv as f64; neumaier(&mut pfc, yv as f64 * sv as f64); } }
            if !ok { for b in yu.iter_mut() { *b = 0; } return (1, 0.0, (0.0, 0.0), 0, 0.0); }
            (2, pf, pfc, c, ss)
        };
        let res: Vec<UnitRow> = {
            let next = std::sync::atomic::AtomicUsize::new(0);
            let slots: Vec<std::sync::Mutex<Option<UnitRow>>> = (0..units.len()).map(|_| std::sync::Mutex::new(None)).collect();
            let (next, slots, units, one) = (&next, &slots, &units, &one);
            std::thread::scope(|sc| {
                for _ in 0..threads.min(units.len()).max(1) {
                    sc.spawn(move || loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= units.len() { break; }
                        let v = one(&units[i].0, units[i].1, units[i].2);
                        *slots[i].lock().unwrap() = Some(v);
                    });
                }
            });
            slots.iter().map(|m| m.lock().unwrap().take().unwrap()).collect()
        };
        for ((id, lo, n), r) in units.iter().zip(&res) {
            let (lo, n) = (*lo, *n);
            if r.0 == 1 { skipped += 1; }
            if r.0 != 2 { continue; }
            let (pf, pfc, c, ss) = (r.1, r.2, r.3, r.4);
            let rd = |b: &[u8], k: usize| f32::from_le_bytes(b[lo + 4 * k..lo + 4 * k + 4].try_into().unwrap());
            p += pf; neumaier(&mut pcomp, pfc.0 + pfc.1); used += 1; words += c; per.push((*id, c, ss)); parts.push((*id, lo, 4 * n, pf));
            if let Some(nmax) = dump { if c <= nmax {
                let rows: Vec<Value> = (0..n).filter_map(|k| { let (sv, gv) = (rd(&s, k), rd(&g, k));
                    if gv != 0.0 && sv.is_finite() && sv != 0.0 { Some(json!([k, sv, gv, prim.map(|b| rd(b, k))])) } else { None } }).collect();
                dumps.push(json!({"event":"window_self_unit_words","unit":id,"first_byte":lo,"words":c,"primal_layout":prim.is_some(),"rows":rows}));
            } }
        }
        per.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        (y, p, pcomp, used, skipped, words, per, parts, dumps)
    }

    // ---- self_weights of the 3 km hunt branch (ad/central-hunt-3km-slot 151be949, self_dot.rs), copied verbatim but for
    // its name: the one-thread form with the floor ----
    #[allow(clippy::too_many_arguments)]
    fn hunt3km_self_weights(s: &[u8], g: &[u8], units: &[(u64, usize, usize)], only: Option<&[u64]>, dump: Option<u64>, prim: Option<&[u8]>,
                            floor: Option<&FloorSpec>) -> SelfOut {
        let mut y = vec![0u8; s.len()];
        let (mut p, mut used, mut skipped, mut words) = (0f64, 0u64, 0u64, 0u64);
        let mut pcomp = (0f64, 0f64);
        let mut per: Vec<(u64, u64, f64)> = Vec::new();
        let mut parts: Vec<(u64, usize, usize, f64)> = Vec::new();
        let mut dumps: Vec<Value> = Vec::new();
        let (mut frows, mut fscale, mut fy, mut fvoted) = (Vec::new(), Vec::new(), 0f64, Vec::new());
        for (id, lo, n) in units {
            let (lo, n) = (*lo, *n);
            if only.is_some_and(|o| !o.contains(id)) { continue; }
            if lo + 4 * n > s.len() { skipped += 1; continue; }
            let rd = |b: &[u8], k: usize| f32::from_le_bytes(b[lo + 4 * k..lo + 4 * k + 4].try_into().unwrap());
            let (mut ss, mut c) = (0f64, 0u64);
            for k in 0..n { let (sv, gv) = (rd(s, k), rd(g, k)); if gv != 0.0 && sv.is_finite() && sv != 0.0 { ss += sv as f64 * sv as f64; c += 1; } }
            if c == 0 || !(ss > 0.0) || !ss.is_finite() { continue; }
            let (mut pf, mut ok) = (0f64, true); let mut pfc = (0f64, 0f64);
            for k in 0..n { let (sv, gv) = (rd(s, k), rd(g, k)); if gv != 0.0 && sv.is_finite() && sv != 0.0 {
                let yv = (sv as f64 / ss) as f32;
                if !yv.is_finite() { ok = false; break; }
                y[lo + 4 * k..lo + 4 * k + 4].copy_from_slice(&yv.to_le_bytes()); pf += yv as f64 * sv as f64; super::super::neumaier(&mut pfc, yv as f64 * sv as f64); } }
            if !ok { for b in &mut y[lo..lo + 4 * n] { *b = 0; } skipped += 1; continue; }
            p += pf; super::super::neumaier(&mut pcomp, pfc.0 + pfc.1); used += 1; words += c; per.push((*id, c, ss)); parts.push((*id, lo, 4 * n, pf));
            if let Some(nmax) = dump { if c <= nmax {
                let rows: Vec<Value> = (0..n).filter_map(|k| { let (sv, gv) = (rd(s, k), rd(g, k));
                    if gv != 0.0 && sv.is_finite() && sv != 0.0 { Some(json!([k, sv, gv, prim.map(|b| rd(b, k))])) } else { None } }).collect();
                dumps.push(json!({"event":"window_self_unit_words","unit":id,"first_byte":lo,"words":c,"primal_layout":prim.is_some(),"rows":rows}));
            } }
            if let Some(fs) = floor {
                let (cu, v) = fs.vote(*id);
                let mut pfl = pf;
                if v < 1.0 {
                    pfl = 0.0;
                    for k in 0..n { let yv = rd(&y, k); if yv != 0.0 { pfl += floor_word(yv, v) as f64 * rd(s, k) as f64; } }
                    fscale.push((lo, n, v));
                }
                fy += pfl; fvoted.push(*id);
                frows.push(json!([id, c, ss, cu, v, pf, pfl]));
            }
        }
        per.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        let floor = floor.map(|fs| FloorOut { k: fs.k, rows: frows, scale: fscale, y_mx: fy,
                                             named_not_voting: fs.c.keys().copied().filter(|u| !fvoted.contains(u)).collect() });
        SelfOut { y, p, pcomp, used, skipped, words, per, parts, dumps, floor }
    }

    fn same_out(a: &SelfOut, b: &SelfOut, what: &str) {
        assert_eq!(a.y, b.y, "{what}: y");
        assert_eq!(a.p.to_bits(), b.p.to_bits(), "{what}: p");
        assert_eq!((a.pcomp.0.to_bits(), a.pcomp.1.to_bits()), (b.pcomp.0.to_bits(), b.pcomp.1.to_bits()), "{what}: pcomp");
        assert_eq!((a.used, a.skipped, a.words), (b.used, b.skipped, b.words), "{what}: counts");
        assert_eq!(a.per.iter().map(|x| (x.0, x.1, x.2.to_bits())).collect::<Vec<_>>(), b.per.iter().map(|x| (x.0, x.1, x.2.to_bits())).collect::<Vec<_>>(), "{what}: per");
        assert_eq!(a.parts.iter().map(|x| (x.0, x.1, x.2, x.3.to_bits())).collect::<Vec<_>>(), b.parts.iter().map(|x| (x.0, x.1, x.2, x.3.to_bits())).collect::<Vec<_>>(), "{what}: parts");
        assert_eq!(a.dumps, b.dumps, "{what}: dumps");
        match (&a.floor, &b.floor) {
            (None, None) => {}
            (Some(f), Some(h)) => {
                assert_eq!(f.k.to_bits(), h.k.to_bits(), "{what}: floor k");
                assert_eq!(f.rows, h.rows, "{what}: floor rows");
                assert_eq!(f.scale.iter().map(|x| (x.0, x.1, x.2.to_bits())).collect::<Vec<_>>(), h.scale.iter().map(|x| (x.0, x.1, x.2.to_bits())).collect::<Vec<_>>(), "{what}: floor scale");
                assert_eq!(f.y_mx.to_bits(), h.y_mx.to_bits(), "{what}: floor y_mx");
                assert_eq!(f.named_not_voting, h.named_not_voting, "{what}: named_not_voting");
            }
            _ => panic!("{what}: one side has a floor and the other has not"),
        }
    }

    /// central wide-tools join: the joined self_weights computes, bit for bit, what the wide candidate's threaded loop
    /// computes (no floor) and what the 3 km branch's one-thread self_weights computes (with and without a floor), for
    /// disjoint units (many threads) and for overlapping units (one thread, unit order), at 1, 2, 7 and 64 threads too.
    #[test]
    fn joined_is_bit_identical_to_wide_and_to_the_3km_function() {
        let mut r = Lcg(19);
        let n = 4096usize;
        let mut s: Vec<f32> = (0..n).map(|_| (r.next() * 10f64.powi((r.next() * 20.0) as i32)) as f32).collect();
        let mut g: Vec<f32> = (0..n).map(|_| if r.next() < -0.6 { 0.0 } else { r.next() as f32 * 1e-20 }).collect();
        s[5] = f32::NAN; s[6] = f32::INFINITY; s[7] = 0.0;
        s[3000] = 1e-40; g[3000] = 1.0; s[3001] = 0.0; s[3002] = 0.0;
        for k in 3100..3110 { g[k] = 0.0; }
        let (sb, gb) = (bytes(&s), bytes(&g));
        let prim = bytes(&(0..n).map(|k| k as f32).collect::<Vec<_>>());
        let disjoint: Vec<(u64, usize, usize)> = vec![(1, 0, 1000), (2, 4000, 1), (3, 4004, 2), (4, 4012, 900), (9, 12000, 3), (11, 12400, 10),
                                                      (5, 4 * 2000, 1000), (6, 4 * 4090, 10), (7, 4 * 3500, 590)];
        // unit 13 writes word 3000 (inside unit 9) before unit 9's overflow zeroes words 3000..3002; unit 12 sits inside unit 1
        let overlap: Vec<(u64, usize, usize)> = vec![(1, 0, 1000), (12, 400, 50), (13, 4 * 2999, 3), (9, 12000, 3), (4, 4012, 900), (14, 4 * 1000, 20)];
        assert!(units_disjoint(&disjoint) && !units_disjoint(&overlap));
        let floors = [None, Some(FloorSpec { k: 4.0, c: BTreeMap::from([(99u64, 1e3)]) }),
                      Some(FloorSpec { k: 2.0, c: BTreeMap::from([(1u64, 50.0), (4, 3.0), (12, 9.0), (13, 400.0), (7, 1.5), (99, 8.0)]) })];
        let mut floored_seen = 0;
        for (units, parallel) in [(&disjoint, true), (&overlap, false)] {
            for (only, dump, pr) in [(None, None, None), (Some(vec![2u64, 4, 9, 12, 13]), Some(4u64), Some(&prim[..])), (None, Some(2), None)] {
                let old = old_self_loop(&sb, &gb, units, only.clone(), dump, pr);
                let wide = wide_self_loop(sb.clone(), gb.clone(), units.clone(), only.clone(), dump, pr);
                assert_eq!(wide.0, old.0); assert_eq!(wide.1.to_bits(), old.1.to_bits());
                for fl in &floors {
                    let h = hunt3km_self_weights(&sb, &gb, units, only.as_deref(), dump, pr, fl.as_ref());
                    let mut runs = vec![("default", self_weights(&sb, &gb, units, only.as_deref(), dump, pr, fl.as_ref()))];
                    for t in if parallel { vec![1usize, 2, 7, 64] } else { vec![1usize] } {
                        runs.push(("threads", self_weights_threads(&sb, &gb, units, only.as_deref(), dump, pr, fl.as_ref(), 0, t)));
                    }
                    for (what, j) in &runs {
                        same_out(j, &h, what);
                        assert_eq!(j.y, wide.0); assert_eq!(j.p.to_bits(), wide.1.to_bits());
                        assert_eq!((j.pcomp.0.to_bits(), j.pcomp.1.to_bits()), (wide.2 .0.to_bits(), wide.2 .1.to_bits()));
                        assert_eq!((j.used, j.skipped, j.words), (wide.3, wide.4, wide.5));
                        assert_eq!(j.per.iter().map(|x| (x.0, x.1, x.2.to_bits())).collect::<Vec<_>>(), wide.6.iter().map(|x| (x.0, x.1, x.2.to_bits())).collect::<Vec<_>>());
                        assert_eq!(j.parts.iter().map(|x| (x.0, x.1, x.2, x.3.to_bits())).collect::<Vec<_>>(), wide.7.iter().map(|x| (x.0, x.1, x.2, x.3.to_bits())).collect::<Vec<_>>());
                        assert_eq!(j.dumps, wide.8);
                        if let Some(f) = &j.floor { floored_seen += f.scale.len(); }
                    }
                }
            }
        }
        assert!(floored_seen > 0, "the test must reach a unit whose vote the floor lowers");
    }

    // ---- the loop of Window::self_y in the 60 km hunt branch (ad/central-hunt-60-repeat e708d47a), copied verbatim but for
    // its inputs and outputs (ypow2 comes in as an argument, dump lines are collected, not printed) ----
    #[allow(clippy::type_complexity)]
    fn hunt60_self_loop(s: Vec<u8>, g: Vec<u8>, units: Vec<(u64, usize, usize)>, only: Option<Vec<u64>>, dump: Option<u64>, prim: Option<&[u8]>, ypow2: i32)
                        -> (Vec<u8>, f64, (f64, f64), u64, u64, u64, Vec<(u64, u64, f64)>, Vec<(u64, usize, usize, f64)>, Vec<Value>) {
        use super::super::neumaier;
        let mut y = vec![0u8; s.len()];
        let (mut p, mut used, mut skipped, mut words) = (0f64, 0u64, 0u64, 0u64);
        let mut pcomp = (0f64, 0f64);
        let mut per: Vec<(u64, u64, f64)> = Vec::new();
        let mut parts: Vec<(u64, usize, usize, f64)> = Vec::new();
        let mut dumps = Vec::new();
        for (id, lo, n) in &units {
            let (lo, n) = (*lo, *n);
            if only.as_ref().is_some_and(|o| !o.contains(id)) { continue; }
            if lo + 4 * n > s.len() { skipped += 1; continue; }
            let rd = |b: &[u8], k: usize| f32::from_le_bytes(b[lo + 4 * k..lo + 4 * k + 4].try_into().unwrap());
            let (mut ss, mut c) = (0f64, 0u64);
            for k in 0..n { let (sv, gv) = (rd(&s, k), rd(&g, k)); if gv != 0.0 && sv.is_finite() && sv != 0.0 { ss += sv as f64 * sv as f64; c += 1; } }
            if c == 0 || !(ss > 0.0) || !ss.is_finite() { continue; }
            let (mut pf, mut ok) = (0f64, true); let mut pfc = (0f64, 0f64);
            for k in 0..n { let (sv, gv) = (rd(&s, k), rd(&g, k)); if gv != 0.0 && sv.is_finite() && sv != 0.0 {
                let yv = crate::rev_digest::self_y_word(sv, ss, ypow2);
                if !yv.is_finite() { ok = false; break; }
                y[lo + 4 * k..lo + 4 * k + 4].copy_from_slice(&yv.to_le_bytes()); pf += yv as f64 * sv as f64; neumaier(&mut pfc, yv as f64 * sv as f64); } }
            if !ok { for b in &mut y[lo..lo + 4 * n] { *b = 0; } skipped += 1; continue; }
            p += pf; neumaier(&mut pcomp, pfc.0 + pfc.1); used += 1; words += c; per.push((*id, c, ss)); parts.push((*id, lo, 4 * n, pf));
            if let Some(nmax) = dump { if c <= nmax {
                let rows: Vec<Value> = (0..n).filter_map(|k| { let (sv, gv) = (rd(&s, k), rd(&g, k));
                    if gv != 0.0 && sv.is_finite() && sv != 0.0 { Some(json!([k, sv, gv, prim.map(|b| rd(b, k))])) } else { None } }).collect();
                dumps.push(json!({"event":"window_self_unit_words","unit":id,"first_byte":lo,"words":c,"primal_layout":prim.is_some(),"rows":rows}));
            } }
        }
        per.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        (y, p, pcomp, used, skipped, words, per, parts, dumps)
    }

    /// central wide-tools join: with AD_DOT_SELF_YPOW2=k the joined self_weights_pow2 computes, bit for bit, what the 60 km
    /// branch's loop computes, at k = 0, 20 and -30, many threads and one; and at k = 0 it is self_weights.
    #[test]
    fn joined_ypow2_is_bit_identical_to_the_60km_loop() {
        let mut r = Lcg(23);
        let n = 4096usize;
        let mut s: Vec<f32> = (0..n).map(|_| (r.next() * 10f64.powi((r.next() * 20.0) as i32)) as f32).collect();
        let g: Vec<f32> = (0..n).map(|_| if r.next() < -0.6 { 0.0 } else { r.next() as f32 * 1e-20 }).collect();
        s[5] = f32::NAN; s[6] = f32::INFINITY; s[7] = 0.0;
        let (sb, gb) = (bytes(&s), bytes(&g));
        let prim = bytes(&(0..n).map(|k| k as f32).collect::<Vec<_>>());
        let units: Vec<(u64, usize, usize)> = vec![(1, 0, 1000), (2, 4000, 1), (3, 4004, 2), (4, 4012, 900), (5, 4 * 2000, 1000), (7, 4 * 3500, 590), (6, 4 * 4090, 10)];
        for k in [0i32, 20, -30] {
            for (only, dump, pr) in [(None, None, None), (Some(vec![2u64, 4, 7]), Some(4u64), Some(&prim[..]))] {
                let h = hunt60_self_loop(sb.clone(), gb.clone(), units.clone(), only.clone(), dump, pr, k);
                let mut runs = vec![self_weights_pow2(&sb, &gb, &units, only.as_deref(), dump, pr, None, k)];
                for t in [1usize, 7] { runs.push(self_weights_threads(&sb, &gb, &units, only.as_deref(), dump, pr, None, k, t)); }
                if k == 0 { runs.push(self_weights(&sb, &gb, &units, only.as_deref(), dump, pr, None)); }
                for j in &runs {
                    assert_eq!(j.y, h.0, "k {k}: y"); assert_eq!(j.p.to_bits(), h.1.to_bits(), "k {k}: p");
                    assert_eq!((j.pcomp.0.to_bits(), j.pcomp.1.to_bits()), (h.2 .0.to_bits(), h.2 .1.to_bits()));
                    assert_eq!((j.used, j.skipped, j.words), (h.3, h.4, h.5));
                    assert_eq!(j.per.iter().map(|x| (x.0, x.1, x.2.to_bits())).collect::<Vec<_>>(), h.6.iter().map(|x| (x.0, x.1, x.2.to_bits())).collect::<Vec<_>>());
                    assert_eq!(j.parts.iter().map(|x| (x.0, x.1, x.2, x.3.to_bits())).collect::<Vec<_>>(), h.7.iter().map(|x| (x.0, x.1, x.2, x.3.to_bits())).collect::<Vec<_>>());
                    assert_eq!(j.dumps, h.8);
                }
            }
        }
        let p0 = self_weights_pow2(&sb, &gb, &units, None, None, None, None, 0).p;
        let p20 = self_weights_pow2(&sb, &gb, &units, None, None, None, None, 20).p;
        assert!((p20 / p0 / 2f64.powi(20) - 1.0).abs() < 1e-6, "y times 2^20 scales <y, M x> by 2^20: {p0} {p20}");
    }

    #[test]
    fn floor_spec_parses_and_refuses() {
        let f = FloorSpec::parse("4:184=426.4578460077008, 176=2e3").unwrap();
        assert_eq!(f.k, 4.0);
        assert_eq!(f.vote(184), (Some(426.4578460077008), 4.0 / 426.4578460077008));
        assert_eq!(f.vote(176), (Some(2e3), 2e-3));
        assert_eq!(f.vote(5), (None, 1.0));
        assert_eq!(FloorSpec::parse("1000:7=3").unwrap().vote(7).1, 1.0, "a unit that cancels less than k is not floored");
        for bad in ["4", "4:", "x:1=2", "0:1=2", "-1:1=2", "inf:1=2", "4:1", "4:a=2", "4:1=0", "4:1=nan", "4:1=2,1=3"] {
            assert!(FloorSpec::parse(bad).is_err(), "{bad} must be refused");
        }
    }

    #[test]
    fn roundings_reads_the_miss_in_float32_roundings_of_the_terms() {
        let two = roundings(86.0, 86.0 + 2.0 * ROUNDING * 200.0, 200.0).unwrap();
        assert!((two - 2.0).abs() < 1e-6, "{two}");
        assert!(roundings(86.0, 86.0 - 3.0 * ROUNDING * 200.0, 200.0).unwrap() < 0.0, "signed: x side smaller reads minus");
        assert!(roundings(1.0, 2.0, 0.0).is_none());
        assert!(roundings(1.0, f64::NAN, 3.0).is_none());
        // the stored 3 km 58-step direction-4 reading (<local path> test machine/storm-f58-usual-s4/window.json)
        let r = roundings(85.99999998853897, 85.99998747794108, 197.66521760138684).unwrap();
        assert!((r + 1.0617).abs() < 1e-3, "{r}");
    }

    // ---- a small synthetic linear operator with float32 words, as the window's check sees one ----
    /// S = M x: float32 products, float32 running sum in column order (the forward).
    fn tl32(m: &[Vec<f32>], x: &[f32]) -> Vec<f32> { m.iter().map(|row| { let mut s = 0f32; for (a, b) in row.iter().zip(x) { s += a * b; } s }).collect() }
    /// xbar = M^T y: float32 products, float32 running sums over the rows in reverse order (the backward).
    fn ad32(m: &[Vec<f32>], y: &[f32], nin: usize) -> Vec<f32> { let mut xb = vec![0f32; nin]; for i in (0..m.len()).rev() { for (j, v) in xb.iter_mut().enumerate() { *v += m[i][j] * y[i]; } } xb }
    /// the same in binary64 from the float32 words (exact for the small integers of the exact test)
    fn tl64(m: &[Vec<f32>], x: &[f32]) -> Vec<f32> { m.iter().map(|row| row.iter().zip(x).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() as f32).collect() }
    fn ad64(m: &[Vec<f32>], y: &[f32], nin: usize) -> Vec<f64> { (0..nin).map(|j| (0..m.len()).map(|i| m[i][j] as f64 * y[i] as f64).sum()).collect() }
    /// <x, xbar> as the window sums it: binary64 products of the float32 words; and sum |terms|
    fn xdot<T: Copy + Into<f64>>(x: &[f32], xb: &[T]) -> (f64, f64) {
        x.iter().zip(xb).fold((0.0, 0.0), |a, (u, v)| { let t = *u as f64 * (*v).into(); (a.0 + t, a.1 + t.abs()) })
    }

    /// Three ordinary units (12, 12 and 6 rows) and one 1-row unit T (row 30) whose tangent is a small difference of
    /// two large terms, so float32 rounding of its forward is large against its own size: the 3 km unit 184's shape.
    /// `small` sets the size of T's other terms and so how much T's backward cancels.
    fn operator(small: f32) -> (Vec<Vec<f32>>, Vec<f32>, Vec<(u64, usize, usize)>) {
        let (nin, nout) = (48usize, 31usize);
        let mut r = Lcg(11);
        let mut m: Vec<Vec<f32>> = (0..nout).map(|_| (0..nin).map(|_| r.next() as f32).collect()).collect();
        let mut x: Vec<f32> = (0..nin).map(|_| r.next() as f32).collect();
        x[0] = 0.712_345_6; x[1] = x[0] * (1.0 + 3e-6);
        m[30] = (0..nin).map(|j| match j { 0 => 1000.0, 1 => -1000.0, _ => small * r.next() as f32 }).collect();
        let units = vec![(0u64, 0usize, 12usize), (1, 48, 12), (2, 96, 6), (30, 120, 1)];
        (m, x, units)
    }
    /// T's other terms in the main tests: its backward then cancels some hundreds-fold, as 3 km unit 184's did (426)
    const SMALL: f32 = 8.0;
    fn plant(m: &[Vec<f32>], rows: std::ops::Range<usize>, f: f32) -> Vec<Vec<f32>> {
        m.iter().enumerate().map(|(i, r)| if rows.contains(&i) { r.iter().map(|v| v * f).collect() } else { r.clone() }).collect()
    }
    /// (y_Mx, x_MTy, abs terms, rel) of one self dot: y from self_weights over S, backward through m_ad in float32.
    fn read32(s: &[f32], m_ad: &[Vec<f32>], x: &[f32], units: &[(u64, usize, usize)], only: Option<&[u64]>, floor: Option<&FloorSpec>) -> (f64, f64, f64, f64, SelfOut) {
        let sb = bytes(s); let g = bytes(&vec![1f32; s.len()]);
        let o = self_weights(&sb, &g, units, only, None, None, floor);
        let (y, ymx) = match &o.floor { Some(f) => { let mut y = o.y.clone(); apply_floor(&mut y, &f.scale); (words(&y), f.y_mx) }, None => (words(&o.y), o.p) };
        let (xmty, abs) = xdot(x, &ad32(m_ad, &y, x.len()));
        (ymx, xmty, abs, super::super::rel(ymx, xmty), o)
    }
    /// the measured c_u of one unit: abs terms of its backward alone (its vote is 1), as AD_DOT_SELF_ONLY=<u> AD_DOT_ALLOC=1 reads it
    fn measure_c(s: &[f32], m: &[Vec<f32>], x: &[f32], units: &[(u64, usize, usize)], u: u64) -> f64 { read32(s, m, x, units, Some(&[u]), None).2 }

    #[test]
    fn an_exact_operator_reads_zero_in_every_form() {
        let mut r = Lcg(3);
        let (nin, nout) = (40usize, 23usize);
        let m: Vec<Vec<f32>> = (0..nout).map(|_| (0..nin).map(|_| (r.next() * 8.0).round() as f32).collect()).collect();
        let x: Vec<f32> = (0..nin).map(|_| (r.next() * 8.0).round() as f32).collect();
        let units = vec![(0u64, 0usize, 10usize), (1, 40, 10), (2, 80, 2), (3, 88, 1)];
        let s = tl64(&m, &x);
        let floor = FloorSpec { k: 0.5, c: BTreeMap::from([(2u64, 50.0), (3, 7.0)]) };
        for fl in [None, Some(&floor)] {
            let sb = bytes(&s); let g = bytes(&vec![1f32; nout]);
            let o = self_weights(&sb, &g, &units, None, None, None, fl);
            let (y, ymx) = match &o.floor { Some(f) => { assert_eq!(f.scale.len(), 2); let mut y = o.y.clone(); apply_floor(&mut y, &f.scale); (words(&y), f.y_mx) }, None => (words(&o.y), o.p) };
            let (xmty, abs) = xdot(&x, &ad64(&m, &y, nin));
            let rel = super::super::rel(ymx, xmty);
            assert!(rel <= 1e-14, "exact operator, floor {:?}: rel {rel}", fl.is_some());
            assert!(roundings(ymx, xmty, abs).unwrap().abs() < 1e-6);
        }
    }

    #[test]
    fn a_rounding_level_unit_no_longer_decides_the_floored_result() {
        let (m, x, units) = operator(SMALL);
        let s = tl32(&m, &x);
        let c_t = measure_c(&s, &m, &x, &units, 30);
        let (_, _, abs_u, rel_u, _) = read32(&s, &m, &x, &units, None, None);
        let (_, _, _, rel_wo, _) = read32(&s, &m, &x, &units, Some(&[0, 1, 2]), None);
        let floor = FloorSpec { k: 4.0, c: BTreeMap::from([(30u64, c_t)]) };
        let (_, _, abs_f, rel_f, o) = read32(&s, &m, &x, &units, None, Some(&floor));
        eprintln!("c_T {c_t} | unfloored rel {rel_u:e} abs {abs_u} | without T {rel_wo:e} | floored rel {rel_f:e} abs {abs_f}");
        assert!(c_t > 100.0, "T must cancel hard in its backward: c_T {c_t}");
        assert!(rel_u > 1e-6, "T decides the unfloored dot (more than 10x the 1e-7 bar): {rel_u:e}");
        assert!(rel_wo < 1e-7, "the three ordinary units alone are inside the bar: {rel_wo:e}");
        assert!(rel_f < rel_u / 100.0, "T no longer decides: floored {rel_f:e} against unfloored {rel_u:e}");
        assert!(rel_f < 2.0 * rel_wo.max(1e-7), "the floored dot reads as the ordinary units do: {rel_f:e} against {rel_wo:e}");
        // the floor decision of every voting unit is printed: T floored to k / c_T, the others not named and kept
        let f = o.floor.as_ref().unwrap();
        assert_eq!(f.rows.len(), 4);
        assert_eq!(f.scale.len(), 1);
        let t = f.rows.iter().find(|r| r[0] == 30).unwrap();
        assert_eq!(t[3].as_f64(), Some(c_t));
        assert!((t[4].as_f64().unwrap() - 4.0 / c_t).abs() < 1e-15);
        assert!(f.rows.iter().filter(|r| r[0] != 30).all(|r| r[3].is_null() && r[4].as_f64() == Some(1.0)));
        let rec = floor_receipt(f);
        assert_eq!(rec["floored_units"], 1);
        // in roundings of the terms, the unfloored miss is small: T's backward brings its own huge terms
        assert!(roundings(0.0, rel_u * 4.0, abs_u).unwrap() < 10.0);
    }

    #[test]
    fn a_planted_error_reads_far_over_the_bar_in_every_form() {
        let (m, x, units) = operator(SMALL);
        let s = tl32(&m, &x);
        let c_t = measure_c(&s, &m, &x, &units, 30);
        let floor = FloorSpec { k: 4.0, c: BTreeMap::from([(30u64, c_t)]) };
        let m_bad = plant(&m, 0..12, 1.001);   // the transpose of unit 0's rows wrong by 0.1 percent
        let clean_u = read32(&s, &m, &x, &units, None, None);
        let clean_f = read32(&s, &m, &x, &units, None, Some(&floor));
        let bad_u = read32(&s, &m_bad, &x, &units, None, None);
        let bad_f = read32(&s, &m_bad, &x, &units, None, Some(&floor));
        let (ru, rf) = (roundings(bad_u.0, bad_u.1, bad_u.2).unwrap(), roundings(bad_f.0, bad_f.1, bad_f.2).unwrap());
        let (cu, cf) = (roundings(clean_u.0, clean_u.1, clean_u.2).unwrap(), roundings(clean_f.0, clean_f.1, clean_f.2).unwrap());
        eprintln!("planted: unfloored rel {:e} ({ru:.1} roundings), floored rel {:e} ({rf:.1} roundings); clean {cu:.2} and {cf:.2} roundings", bad_u.3, bad_f.3);
        assert!(bad_u.3 > 1e-5 && bad_f.3 > 1e-5, "100x the bar in both forms");
        assert!(ru.abs() > 10.0 * cu.abs().max(1.0) && rf.abs() > 10.0 * cf.abs().max(1.0), "and far outside the clean reading's roundings in both forms");
        assert!(ru > 0.0 && rf > 0.0, "one-sided: the planted backward is 0.1 percent larger, so x_MTy - y_Mx > 0 in both forms");
    }

    #[test]
    fn roundings_alone_can_hide_a_plant_behind_a_cancelling_unit() {
        // T cancels about 1e5-fold here: its terms swamp the dot's absolute terms, and the unfloored roundings reading
        // of a 0.1 percent plant falls under one rounding. The floored form still shows the plant. This pins why
        // miss_roundings is read beside a planted-error row and is not a verdict.
        let (m, x, units) = operator(0.001);
        let s = tl32(&m, &x);
        let c_t = measure_c(&s, &m, &x, &units, 30);
        let floor = FloorSpec { k: 4.0, c: BTreeMap::from([(30u64, c_t)]) };
        let m_bad = plant(&m, 0..12, 1.001);
        let bad_u = read32(&s, &m_bad, &x, &units, None, None);
        let bad_f = read32(&s, &m_bad, &x, &units, None, Some(&floor));
        let (ru, rf) = (roundings(bad_u.0, bad_u.1, bad_u.2).unwrap(), roundings(bad_f.0, bad_f.1, bad_f.2).unwrap());
        eprintln!("c_T {c_t}: planted unfloored {ru:.2} roundings (rel {:e}), floored {rf:.1} roundings (rel {:e})", bad_u.3, bad_f.3);
        assert!(c_t > 1e4);
        assert!(ru.abs() < 2.0, "the hazard: {ru}");
        assert!(rf.abs() > 100.0 && bad_f.3 > 1e-5);
    }

    #[test]
    fn floor_shrinks_an_error_inside_a_floored_unit() {
        // The cost of the floor, pinned: a wrong transpose inside a floored unit shows only v_u of its size.
        let (m, x, units) = operator(SMALL);
        let s = tl32(&m, &x);
        let c_t = measure_c(&s, &m, &x, &units, 30);
        let floor = FloorSpec { k: 4.0, c: BTreeMap::from([(30u64, c_t)]) };
        let m_bad = plant(&m, 30..31, 1.001);
        let u = read32(&s, &m_bad, &x, &units, None, None);
        let f = read32(&s, &m_bad, &x, &units, None, Some(&floor));
        let (du, df) = (u.1 - u.0, f.1 - f.0);
        eprintln!("plant inside T: unfloored miss {du:e}, floored miss {df:e}, v_T {}", 4.0 / c_t);
        assert!(du > 5e-4, "unfloored: about 1e-3 of T's one vote");
        assert!(df < du * 8.0 / c_t, "floored: shrunk toward v_T times it");
    }

    #[test]
    fn form_receipt_names_the_form_and_keeps_the_valid_rule() {
        let f = FloorSpec::parse("4:184=426.46").unwrap();
        let r = form_receipt(true, None, None, Some(&f), true);
        assert!(r["form"].as_str().unwrap().starts_with("self: one unit per memory allocation"));
        assert_eq!(r["floor"]["c"]["184"], 426.46);
        assert!(r["valid_rule"].as_str().unwrap().starts_with("unchanged"));
        assert!(form_receipt(true, Some("m.json:59"), None, None, false)["form"].as_str().unwrap().contains("prognostic field"));
        assert!(form_receipt(false, None, None, None, true)["form"].as_str().unwrap().starts_with("standard"));
    }
}
