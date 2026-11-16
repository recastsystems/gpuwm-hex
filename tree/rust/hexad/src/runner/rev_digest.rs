// SPDX-License-Identifier: Apache-2.0
//! hex-central, hunt-60-repeat (2026-11-11): a digest of the cotangent the backward sweep holds, per shadow allocation
//! (the "start blocks" of the dot's per-block print), after every step walked. OFF unless AD_REV_DIGEST is set; with
//! it unset nothing here runs and no byte of a window changes.
//!
//! THE BREAKAGE IT LOCATES. A window's check number moves when nothing changes but the size of the test direction by
//! an exact power of two (60 km, 29 steps: 7 of 10 pairs; only the backward side moves). A backward sweep is linear in
//! its seed, and a power of two changes no digit of a single-precision product or sum, so two such runs must hold
//! cotangents that are word for word 2^k times each other. They stop doing so when a word underflows (it falls under
//! the smallest normal float32, 1.18e-38, and loses digits or is flushed to zero: 25 of the 89 derivative kernels of
//! the 60 km kernel set are built with --ftz=true), overflows, or is compared with a fixed size. The stored runs show
//! WHEN this first becomes visible (the norm of the cotangent entering the cloud scheme, step 23 to 19 walking back
//! from 29) but not WHERE: a sum of squares hides a damaged word until later steps bring it back to ordinary size.
//!
//! WHAT IT PRINTS. One `window_rev_digest` line on stderr per point: the seed before the sweep (`at` "seed"), the
//! arena after each step walked (`at` "step"), and, when asked, after each command walked of chosen steps (`at`
//! "command"). Per allocation: words, nonzero words, subnormal words, non-finite words, the binary exponent of the
//! largest and of the smallest nonzero word, a hash, and a normalised sum of squares.
//! The hash and the sum are SCALE-FREE: each word enters as (sign, 23 mantissa bits, exponent below the block's
//! largest), so two blocks that are word for word 2^k times each other give the same hash, the same sum and exponents
//! that differ by exactly k. A subnormal word is normalised first, so one that is still the exact image of its twin
//! hashes the same and one that lost digits does not. `tree/ad/phys/window/central/hunt60/rev_digest_cmp.py` reads two
//! runs' stderr and names the first point and block that depart.
//!
//! SWITCHES (all off by default):
//!   AD_REV_DIGEST=1 | all        every shadow allocation (26.5 GB comes down per point at 60 km: seconds a step)
//!   AD_REV_DIGEST=only:361,373   those allocations only
//!   AD_REV_DIGEST_CMD=23,22      also after every command walked back in those steps; needs the only: form
//!   AD_DOT_SELF_YPOW2=k          the self check's seed y is multiplied by 2^k (exactly, in binary64 before the one
//!                                rounding to float32). The check is linear in y, so both of its sums carry the factor
//!                                2^k and the check number must not move; where it does, the backward sweep is not
//!                                proportional to its seed at that size. Same x, same tangent, same y up to the power
//!                                of two: the one experiment that separates the backward side from the forward side.
use serde_json::{json, Value};

/// What one block of float32 words holds. `emax`/`emin` are the binary exponents e of the largest and smallest finite
/// nonzero |word| = (1 + m / 2^23) * 2^e (subnormals normalised: e goes down to -149).
#[derive(Clone, Debug, PartialEq)]
pub struct BlockDigest {
    pub words: u64,
    pub nonzero: u64,
    pub subnormal: u64,
    pub nonfinite: u64,
    pub emax: Option<i32>,
    pub emin: Option<i32>,
    pub hash: u64,
    /// sum over finite nonzero words of (word / 2^emax)^2, binary64, in word order
    pub nsq: f64,
}

/// (negative, exponent e, 23 mantissa bits m) of a finite nonzero float32: |v| = (1 + m / 2^23) * 2^e.
/// A subnormal is normalised (its leading one becomes the implicit one), so e runs from -149 to 127.
#[inline]
fn parts(bits: u32) -> (bool, i32, u32) {
    let (neg, ef, frac) = (bits >> 31 != 0, ((bits >> 23) & 0xff) as i32, bits & 0x7f_ffff);
    if ef != 0 { return (neg, ef - 127, frac); }
    let p = 31 - frac.leading_zeros() as i32;          // position of the leading one, 0..=22
    let shift = 23 - p;                                // 1..=23
    (neg, -126 - shift, (frac << shift) & 0x7f_ffff)
}

#[inline]
fn mix(h: u64, token: u64) -> u64 { (h.rotate_left(5) ^ token).wrapping_mul(0x9E37_79B9_7F4A_7C15) }

/// The digest of one block (little-endian float32 words; a tail shorter than a word is ignored).
pub fn digest(bytes: &[u8]) -> BlockDigest {
    let rd = |c: &[u8]| u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    let (mut nonzero, mut subnormal, mut nonfinite) = (0u64, 0u64, 0u64);
    let (mut emax, mut emin): (Option<i32>, Option<i32>) = (None, None);
    for c in bytes.chunks_exact(4) {
        let b = rd(c);
        let mag = b & 0x7fff_ffff;
        if mag == 0 { continue; }
        if mag >= 0x7f80_0000 { nonfinite += 1; continue; }
        nonzero += 1;
        if mag < 0x0080_0000 { subnormal += 1; }
        let e = parts(b).1;
        emax = Some(emax.map_or(e, |x: i32| x.max(e)));
        emin = Some(emin.map_or(e, |x: i32| x.min(e)));
    }
    let top = emax.unwrap_or(0);
    let (mut h, mut nsq) = (0x243F_6A88_85A3_08D3u64, 0f64);
    for c in bytes.chunks_exact(4) {
        let b = rd(c);
        let mag = b & 0x7fff_ffff;
        if mag == 0 { h = mix(h, 1 | ((b >> 31) as u64) << 2); continue; }
        if mag >= 0x7f80_0000 { h = mix(h, 2 | (b as u64) << 8); continue; }
        let (neg, e, m) = parts(b);
        let down = (top - e) as u64;                   // 0..=276
        h = mix(h, 3 | (neg as u64) << 2 | down << 8 | (m as u64) << 40);
        let v = (1.0 + m as f64 / 8_388_608.0) * (2f64).powi(e - top);
        nsq += v * v;
    }
    BlockDigest { words: (bytes.len() / 4) as u64, nonzero, subnormal, nonfinite, emax, emin, hash: h, nsq }
}

/// `digest` of every block, on up to 16 worker threads; results in the blocks' order.
pub fn digest_many(blocks: &[Vec<u8>]) -> Vec<BlockDigest> {
    let threads = std::thread::available_parallelism().map(|t| t.get()).unwrap_or(4).min(16).min(blocks.len()).max(1);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let slots: Vec<std::sync::Mutex<Option<BlockDigest>>> = (0..blocks.len()).map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= blocks.len() { break; }
                *slots[i].lock().unwrap() = Some(digest(&blocks[i]));
            });
        }
    });
    slots.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect()
}

#[derive(Clone, Debug, PartialEq)]
pub enum Which { All, Only(Vec<u64>) }

#[derive(Clone, Debug, PartialEq)]
pub struct Config { pub which: Which, pub cmd_steps: Vec<usize> }
impl Config {
    pub fn wants(&self, id: u64) -> bool { match &self.which { Which::All => true, Which::Only(v) => v.contains(&id) } }
}

/// The switches as words. Unset, empty or "0" is OFF (None). A word that cannot be read is refused: a diagnostic run
/// that silently digests nothing costs a card hour.
pub fn parse(digest: Option<&str>, cmd: Option<&str>) -> Result<Option<Config>, String> {
    let d = digest.map(str::trim).unwrap_or("");
    let c = cmd.map(str::trim).unwrap_or("");
    if d.is_empty() || d == "0" {
        if !c.is_empty() { return Err("AD_REV_DIGEST_CMD is set but AD_REV_DIGEST is not: set AD_REV_DIGEST=only:<allocation,...>".into()); }
        return Ok(None);
    }
    let which = if d == "1" || d == "all" { Which::All } else if let Some(l) = d.strip_prefix("only:") {
        let ids: Result<Vec<u64>, _> = l.split(',').map(|x| x.trim().parse::<u64>()).collect();
        let ids = ids.map_err(|_| format!("AD_REV_DIGEST={d}: only: wants allocation numbers separated by commas"))?;
        if ids.is_empty() { return Err(format!("AD_REV_DIGEST={d}: no allocation named")); }
        Which::Only(ids)
    } else { return Err(format!("AD_REV_DIGEST={d}: wants 1, all, or only:<allocation,...>")); };
    let cmd_steps = if c.is_empty() { Vec::new() } else {
        if which == Which::All { return Err("AD_REV_DIGEST_CMD needs AD_REV_DIGEST=only:<allocation,...>: every allocation after every command is the whole arena down a thousand times a step".into()); }
        let s: Result<Vec<usize>, _> = c.split(',').map(|x| x.trim().parse::<usize>()).collect();
        s.map_err(|_| format!("AD_REV_DIGEST_CMD={c}: wants step numbers separated by commas"))?
    };
    Ok(Some(Config { which, cmd_steps }))
}

/// The process's switches, read once. An unreadable word is an error at every call (the window refuses at its start).
pub fn config() -> Result<&'static Option<Config>, String> {
    static C: std::sync::OnceLock<Result<Option<Config>, String>> = std::sync::OnceLock::new();
    C.get_or_init(|| parse(std::env::var("AD_REV_DIGEST").ok().as_deref(), std::env::var("AD_REV_DIGEST_CMD").ok().as_deref()))
        .as_ref().map_err(Clone::clone)
}

/// True when the commands walked back in `step` are digested one by one.
pub fn cmd_step(step: usize) -> bool { matches!(config(), Ok(Some(c)) if c.cmd_steps.contains(&step)) }

/// AD_DOT_SELF_YPOW2 as a word: unset, empty or "0" is 0 (the seed as it always was).
pub fn parse_ypow2(v: Option<&str>) -> Result<i32, String> {
    let s = v.map(str::trim).unwrap_or("");
    if s.is_empty() { return Ok(0); }
    let k: i32 = s.parse().map_err(|_| format!("AD_DOT_SELF_YPOW2={s}: wants a whole number"))?;
    if !(-120..=120).contains(&k) { return Err(format!("AD_DOT_SELF_YPOW2={k}: outside -120..=120 (a float32 has 254 binades)")); }
    Ok(k)
}
pub fn ypow2() -> Result<i32, String> {
    static K: std::sync::OnceLock<Result<i32, String>> = std::sync::OnceLock::new();
    K.get_or_init(|| parse_ypow2(std::env::var("AD_DOT_SELF_YPOW2").ok().as_deref())).clone()
}

/// One word of the self check's seed: y = S / sum(S^2), times 2^k when k is not 0. k = 0 is the expression the window
/// always used, to the bit. The product with 2^k is exact in binary64, so the one rounding to float32 is the same
/// rounding at every k, until the result leaves the normal float32 range.
#[inline]
pub fn self_y_word(sv: f32, ss: f64, k: i32) -> f32 {
    if k == 0 { (sv as f64 / ss) as f32 } else { (sv as f64 / ss * (2f64).powi(k)) as f32 }
}

/// The stderr line of one point. The hash is 16 hex digits (a JSON number would lose its low bits).
pub fn event(at: &str, step: usize, cmd: Option<(usize, &str)>, tag: &Value, rows: &[(u64, BlockDigest)]) -> Value {
    let r: Vec<Value> = rows.iter().map(|(id, d)| json!([id, d.words, d.nonzero, d.subnormal, d.nonfinite, d.emax, d.emin, format!("{:016x}", d.hash), d.nsq])).collect();
    json!({"event":"window_rev_digest","at":at,"step":step,
           "segment_command":cmd.map(|c| c.0),"command":cmd.map(|c| c.1),"tag":tag,
           "fields":"[allocation, words, nonzero, subnormal, nonfinite, emax, emin, scale-free hash, scale-free sum of squares]",
           "subnormal_words":rows.iter().map(|x| x.1.subnormal).sum::<u64>(),"nonzero_words":rows.iter().map(|x| x.1.nonzero).sum::<u64>(),
           "rows":r})
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
    fn field(seed: u64, n: usize) -> Vec<f32> {
        // ordinary cotangent-like words: signs, 30 binades of size, exact zeros
        let mut s = seed;
        (0..n).map(|i| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            if i % 7 == 3 { return 0.0; }
            let m = 1.0 + ((s >> 40) as f32) / 16_777_216.0;
            let e = ((s >> 20) % 30) as i32 - 15;
            let v = m * (2f32).powi(e);
            if s & 1 == 0 { v } else { -v } }).collect()
    }

    #[test]
    fn a_power_of_two_changes_exponents_and_nothing_else() {
        let a = field(1, 4099);
        let da = digest(&bytes(&a));
        for k in [1i32, 20, -20, 21] {
            let b: Vec<f32> = a.iter().map(|x| x * (2f32).powi(k)).collect();
            let db = digest(&bytes(&b));
            assert_eq!(db.hash, da.hash, "k {k}");
            assert_eq!(db.nsq.to_bits(), da.nsq.to_bits(), "k {k}");
            assert_eq!((db.nonzero, db.subnormal, db.nonfinite, db.words), (da.nonzero, 0, 0, da.words));
            assert_eq!(db.emax.unwrap() - da.emax.unwrap(), k);
            assert_eq!(db.emin.unwrap() - da.emin.unwrap(), k);
        }
        assert!(da.nonzero > 3000 && da.nonzero < da.words);
    }

    #[test]
    fn one_word_off_by_one_rounding_step_is_seen_and_so_is_its_place() {
        let a = field(2, 1000);
        let da = digest(&bytes(&a));
        let mut b = a.clone();
        let i = b.iter().position(|x| *x != 0.0).unwrap();
        b[i] = f32::from_bits(b[i].to_bits() + 1);
        assert_ne!(digest(&bytes(&b)).hash, da.hash);
        // the same words in another order are another block
        let mut c = a.clone();
        let j = (i + 1..c.len()).find(|&j| c[j] != c[i]).unwrap();
        c.swap(i, j);
        assert_ne!(digest(&bytes(&c)).hash, da.hash);
        // a word flushed to zero is seen, and counted
        let mut z = a.clone(); z[i] = 0.0;
        let dz = digest(&bytes(&z));
        assert_ne!(dz.hash, da.hash);
        assert_eq!(dz.nonzero, da.nonzero - 1);
    }

    #[test]
    fn a_subnormal_that_is_still_exact_hashes_as_its_twin_and_one_that_lost_digits_does_not() {
        // 1.5 * 2^-120 and 1.25 * 2^-100: normal; times 2^-20 the first is subnormal but exact (two mantissa bits)
        let a = [1.5f32 * (2f32).powi(-120), -1.25 * (2f32).powi(-100), 0.0];
        let b: Vec<f32> = a.iter().map(|x| x * (2f32).powi(-20)).collect();
        let (da, db) = (digest(&bytes(&a)), digest(&bytes(&b)));
        assert_eq!((da.subnormal, db.subnormal), (0, 1));
        assert_eq!(db.hash, da.hash);
        assert_eq!(db.emin, Some(-140));
        assert_eq!(db.emax.unwrap() - da.emax.unwrap(), -20);
        // a full 24-bit mantissa cannot survive 20 binades under the normal range: the digest departs
        let full = f32::from_bits(((-120i32 + 127) as u32) << 23 | 0x7f_ffff);
        let c = [full, -1.25 * (2f32).powi(-100), 0.0];
        let d: Vec<f32> = c.iter().map(|x| x * (2f32).powi(-20)).collect();
        assert_ne!(d[0] * (2f32).powi(20), c[0]);
        assert_ne!(digest(&bytes(&d)).hash, digest(&bytes(&c)).hash);
        // flushed to zero (what an --ftz=true kernel does with it): seen as a lost word
        let f = [0.0f32, d[1], 0.0];
        let df = digest(&bytes(&f));
        assert_ne!(df.hash, digest(&bytes(&c)).hash);
        assert_eq!(df.nonzero, 1);
    }

    #[test]
    fn smallest_and_largest_words_and_non_finite_words() {
        let a = [f32::from_bits(1), f32::MAX, f32::INFINITY, f32::NAN, -0.0, 0.0, f32::MIN_POSITIVE];
        let d = digest(&bytes(&a));
        assert_eq!((d.words, d.nonzero, d.subnormal, d.nonfinite), (7, 3, 1, 2));
        assert_eq!((d.emax, d.emin), (Some(127), Some(-149)));
        assert_eq!(parts(f32::from_bits(1).to_bits()), (false, -149, 0));
        assert_eq!(parts((-f32::MIN_POSITIVE).to_bits()), (true, -126, 0));
        assert_eq!(parts(f32::from_bits(0x0040_0001).to_bits()), (false, -127, 2));
        let e = digest(&[]);
        assert_eq!((e.words, e.nonzero, e.emax, e.nsq), (0, 0, None, 0.0));
        // -0 and +0 are different words
        assert_ne!(digest(&bytes(&[0.0, 1.0])).hash, digest(&bytes(&[-0.0, 1.0])).hash);
    }

    #[test]
    fn many_blocks_come_back_in_order() {
        let blocks: Vec<Vec<u8>> = (0..37u64).map(|i| bytes(&field(i, 50 + 13 * i as usize))).collect();
        let d = digest_many(&blocks);
        assert_eq!(d.len(), 37);
        for (b, x) in blocks.iter().zip(&d) { assert_eq!(*x, digest(b)); }
        assert!(digest_many(&[]).is_empty());
    }

    #[test]
    fn off_unless_asked_and_bad_words_are_refused() {
        assert_eq!(parse(None, None), Ok(None));
        assert_eq!(parse(Some(""), None), Ok(None));
        assert_eq!(parse(Some("0"), Some("")), Ok(None));
        assert_eq!(parse(Some("1"), None), Ok(Some(Config { which: Which::All, cmd_steps: vec![] })));
        assert_eq!(parse(Some("all"), None).unwrap().unwrap().which, Which::All);
        let c = parse(Some("only:361, 373"), Some("23,22")).unwrap().unwrap();
        assert_eq!(c, Config { which: Which::Only(vec![361, 373]), cmd_steps: vec![23, 22] });
        assert!(c.wants(373) && !c.wants(350));
        assert!(parse(Some("1"), None).unwrap().unwrap().wants(350));
        assert!(parse(Some("361"), None).is_err());
        assert!(parse(Some("only:"), None).is_err());
        assert!(parse(Some("only:a"), None).is_err());
        assert!(parse(Some("1"), Some("23")).is_err());
        assert!(parse(None, Some("23")).is_err());
        assert!(parse(Some("only:361"), Some("x")).is_err());
        assert_eq!(parse_ypow2(None), Ok(0));
        assert_eq!(parse_ypow2(Some("")), Ok(0));
        assert_eq!(parse_ypow2(Some("20")), Ok(20));
        assert_eq!(parse_ypow2(Some("-3")), Ok(-3));
        assert!(parse_ypow2(Some("121")).is_err());
        assert!(parse_ypow2(Some("two")).is_err());
    }

    #[test]
    fn the_seed_word_is_unchanged_at_zero_and_exactly_scaled_while_it_stays_normal() {
        let ss = 2.3972812312972375e18f64;   // the largest sum of squares of the 60 km direction-1 run at s
        for sv in [1.5e9f32, -3.25e-4, 7.0e-12, 2.9e-20] {
            assert_eq!(self_y_word(sv, ss, 0).to_bits(), ((sv as f64 / ss) as f32).to_bits());
            let (y0, y20, y40) = (self_y_word(sv, ss, 0), self_y_word(sv, ss, 20), self_y_word(sv, ss, 40));
            assert!(y0.is_normal());
            assert_eq!(y20, y0 * (2f32).powi(20));
            assert_eq!(y40, y20 * (2f32).powi(20));
        }
    }

    /// The defect this hunt names, on one word: a tangent word of ordinary size in a block whose sum of squares is
    /// carried by a few giant words gets a SUBNORMAL seed word, which has lost digits; 2^20 times larger (the size the
    /// seed has when the test direction is 2^20 times smaller) it has not. So the two seeds are not 2^20 times each other.
    #[test]
    fn a_small_seed_word_loses_digits_and_a_larger_seed_does_not() {
        let ss = 2.3972812312972375e18f64;
        let sv = 1.234_567_9e-21f32;
        let (y0, y20, y40) = (self_y_word(sv, ss, 0), self_y_word(sv, ss, 20), self_y_word(sv, ss, 40));
        assert!(y0 != 0.0 && !y0.is_normal(), "the seed word at the natural size is subnormal");
        assert!(y20.is_normal() && y40.is_normal());
        assert_eq!(y40, y20 * (2f32).powi(20), "two normal sizes are exact images of each other");
        assert_ne!(y0 * (2f32).powi(20), y20, "the subnormal one is not");
        // and the digest of the three seeds says so: the larger two agree, the natural one departs
        let big = 1.5e9f32;
        let seed = |k: i32| bytes(&[self_y_word(big, ss, k), self_y_word(sv, ss, k)]);
        let (d0, d20, d40) = (digest(&seed(0)), digest(&seed(20)), digest(&seed(40)));
        assert_eq!(d20.hash, d40.hash);
        assert_ne!(d0.hash, d20.hash);
        assert_eq!((d0.subnormal, d20.subnormal), (1, 0));
    }

    #[test]
    fn the_line_carries_every_field_and_a_hex_hash() {
        let d = digest(&bytes(&[1.0, -2.0, 0.0]));
        let e = event("step", 23, None, &json!({"seed":1,"repeat":0}), &[(361, d.clone())]);
        assert_eq!(e["event"], "window_rev_digest");
        assert_eq!((e["at"].as_str(), e["step"].as_u64(), e["segment_command"].is_null()), (Some("step"), Some(23), true));
        let r = e["rows"][0].as_array().unwrap();
        assert_eq!(r.len(), 9);
        assert_eq!((r[0].as_u64(), r[1].as_u64(), r[2].as_u64(), r[5].as_i64(), r[6].as_i64()), (Some(361), Some(3), Some(2), Some(1), Some(0)));
        assert_eq!(r[7].as_str().unwrap(), format!("{:016x}", d.hash));
        assert_eq!(r[8].as_f64().unwrap(), 1.25);
        let c = event("command", 23, Some((58, "fct_edge_residual_v841")), &Value::Null, &[]);
        assert_eq!((c["segment_command"].as_u64(), c["command"].as_str()), (Some(58), Some("fct_edge_residual_v841")));
    }
}
