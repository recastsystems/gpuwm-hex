// SPDX-License-Identifier: Apache-2.0
//! hex-central, hunt-120-start (2026-11-11): named words of the derivative memory, printed bit for bit after every
//! command of chosen steps, in the forward (tangent) sweep and in the backward sweep. OFF unless AD_WORD_PROBE is
//! set; with it unset nothing here runs, no word of a window changes and no receipt key is added. It only reads.
//!
//! THE BREAKAGE IT LOCATES. At 120 km over 3 hours the check number of one test direction moves between 4.5e-08 and
//! 1.09e-06 when nothing changes but the size of the direction by 3 or 13 (a factor 2 moves nothing). Two blocks of
//! 437 (the physics copies of temperature and Exner pressure at the window start) carry 96 to 99 percent of the
//! movement, and inside them a dozen words in six model columns carry half. Read from the stored words: in each of
//! those columns two neighbouring levels hold backward values of opposite sign and nearly equal size (a vertical
//! difference with a large gain), and all heavy words of one column move between sizes by ONE common factor of 1e-06
//! to 1e-05. So the rounding is made in one number per column that the sweep hands on, not where the two blocks are
//! written. The stored dumps hold the two ends of the window only, so they cannot say which kernel makes that number.
//!
//! WHAT IT PRINTS. One `window_word_probe` line on stderr per point: after each command of the asked steps (`at`
//! "command"; only the stripes whose bits changed since their last print), at the end of every step walked (`at`
//! "step", every stripe), and at the seed of the backward sweep (`at` "seed"). A stripe is `count` float32 words of
//! one allocation's derivative memory, `stride` words apart from `first`: with the level-major layout of the hex
//! fields (word = level * cells + cell) one stripe is one model column of one field. Words are printed as 8 hex
//! digits each, so two runs can be compared exactly. `tree/ad/phys/window/central/hunt120/word_probe_cmp.py` reads two
//! runs' stderr (sizes s and r s) and names, per stripe, the first command after which the larger run is no longer
//! r times the smaller one (forward) or 1/r times it (backward, self check) to the asked tolerance.
//!
//! SWITCHES (all off by default):
//!   AD_WORD_PROBE=<file.json>      {"stripes":[{"name":"col38573.th","allocation":391,"first_word":12204333,
//!                                  "stride_words":40962,"count":24}, ...]}
//!   AD_WORD_PROBE_STEPS=1,2        steps whose commands are probed one by one (step ends are always probed)
//!   AD_WORD_PROBE_CMDS=350:4700    forward sweep only: the plan command numbers probed in those steps (default all)
//!
//! COST. Each word is one 4-byte read after a drain. 8 columns of 24 levels in 45 fields is 8,640 words a point; a
//! 120 km step has 6,440 commands. Keep the stripes to the columns in question and the steps to one or two.
use serde_json::{json, Value};

/// At most this many words in all stripes together: a probe is a few columns, not a dump.
pub const MAX_WORDS: u64 = 200_000;

#[derive(Clone, Debug, PartialEq)]
pub struct Stripe { pub name: String, pub alloc: u64, pub first: u64, pub stride: u64, pub count: u64 }
impl Stripe {
    /// The stripe must lie inside its allocation (`bytes` long). A stripe that reads past the end would print another
    /// allocation's words under this name, so it is refused by name.
    pub fn check(&self, bytes: usize) -> Result<(), String> {
        let last = self.first + (self.count - 1) * self.stride;
        if (last + 1) * 4 > bytes as u64 {
            return Err(format!("AD_WORD_PROBE: stripe {} reaches word {last} of allocation {}, which has {} words", self.name, self.alloc, bytes / 4));
        }
        Ok(())
    }
    /// The word numbers of the stripe inside its allocation, in print order.
    pub fn words(&self) -> impl Iterator<Item = u64> + '_ { (0..self.count).map(move |k| self.first + k * self.stride) }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Config { pub stripes: Vec<Stripe>, pub cmd_steps: Vec<usize>, pub cmd_range: Option<(usize, usize)> }
impl Config {
    /// True when the command `index` of `step` is probed. The range is in plan command numbers, which only the
    /// forward sweep has; the backward sweep numbers its commands inside a segment, so there every command counts.
    pub fn wants_cmd(&self, sweep: &str, step: usize, index: usize) -> bool {
        if !self.cmd_steps.contains(&step) { return false; }
        match (sweep, self.cmd_range) { ("tangent", Some((lo, hi))) => index >= lo && index <= hi, _ => true }
    }
}

/// The stripe file as text. Every stripe needs a name (unique), an allocation, a first word, a stride of at least 1
/// when it has more than one word, and a count of at least 1. A file that cannot be read this way is refused: a
/// diagnostic run that silently probes nothing costs a card hour.
pub fn parse_spec(text: &str) -> Result<Vec<Stripe>, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("AD_WORD_PROBE: not JSON: {e}"))?;
    let list = v["stripes"].as_array().ok_or("AD_WORD_PROBE: no \"stripes\" list")?;
    if list.is_empty() { return Err("AD_WORD_PROBE: the \"stripes\" list is empty".into()); }
    let (mut out, mut total): (Vec<Stripe>, u64) = (Vec::with_capacity(list.len()), 0);
    for (i, s) in list.iter().enumerate() {
        let name = s["name"].as_str().ok_or_else(|| format!("AD_WORD_PROBE: stripe {i} has no name"))?.to_string();
        let num = |k: &str| s[k].as_u64().ok_or_else(|| format!("AD_WORD_PROBE: stripe {name}: {k} is not a whole number"));
        let (alloc, first, count) = (num("allocation")?, num("first_word")?, num("count")?);
        let stride = if count > 1 { num("stride_words")? } else { s["stride_words"].as_u64().unwrap_or(1) };
        if count == 0 { return Err(format!("AD_WORD_PROBE: stripe {name}: count is 0")); }
        if count > 1 && stride == 0 { return Err(format!("AD_WORD_PROBE: stripe {name}: stride_words is 0")); }
        if name.is_empty() || name.contains(char::is_whitespace) { return Err(format!("AD_WORD_PROBE: stripe {i}: a name is one word without spaces")); }
        if out.iter().any(|x: &Stripe| x.name == name) { return Err(format!("AD_WORD_PROBE: stripe name {name} is used twice")); }
        total += count;
        if total > MAX_WORDS { return Err(format!("AD_WORD_PROBE: more than {MAX_WORDS} words asked (at stripe {name}): a probe is a few columns")); }
        out.push(Stripe { name, alloc, first, stride, count });
    }
    Ok(out)
}

/// AD_WORD_PROBE_STEPS as a word: step numbers separated by commas; unset or empty is no step.
pub fn parse_steps(v: Option<&str>) -> Result<Vec<usize>, String> {
    let s = v.map(str::trim).unwrap_or("");
    if s.is_empty() { return Ok(Vec::new()); }
    s.split(',').map(|x| x.trim().parse::<usize>()).collect::<Result<Vec<_>, _>>()
        .map_err(|_| format!("AD_WORD_PROBE_STEPS={s}: wants step numbers separated by commas"))
}

/// AD_WORD_PROBE_CMDS as a word: lo:hi, both included; unset or empty is every command.
pub fn parse_range(v: Option<&str>) -> Result<Option<(usize, usize)>, String> {
    let s = v.map(str::trim).unwrap_or("");
    if s.is_empty() { return Ok(None); }
    let bad = || format!("AD_WORD_PROBE_CMDS={s}: wants lo:hi with lo <= hi");
    let (a, b) = s.split_once(':').ok_or_else(bad)?;
    let (lo, hi) = (a.trim().parse::<usize>().map_err(|_| bad())?, b.trim().parse::<usize>().map_err(|_| bad())?);
    if lo > hi { return Err(bad()); }
    Ok(Some((lo, hi)))
}

/// The three switches as words. `spec` is the text of the stripe file, None when AD_WORD_PROBE is unset, empty or "0".
pub fn parse(spec: Option<&str>, steps: Option<&str>, cmds: Option<&str>) -> Result<Option<Config>, String> {
    let (st, rg) = (parse_steps(steps)?, parse_range(cmds)?);
    let Some(text) = spec else {
        if !st.is_empty() || rg.is_some() { return Err("AD_WORD_PROBE_STEPS or AD_WORD_PROBE_CMDS is set but AD_WORD_PROBE is not: name the stripe file".into()); }
        return Ok(None);
    };
    if rg.is_some() && st.is_empty() { return Err("AD_WORD_PROBE_CMDS needs AD_WORD_PROBE_STEPS: the range is inside the steps named there".into()); }
    Ok(Some(Config { stripes: parse_spec(text)?, cmd_steps: st, cmd_range: rg }))
}

/// The process's switches, read once. An unreadable word or file is an error at every call (the window refuses at its
/// start, before the tangent pass).
pub fn config() -> Result<&'static Option<Config>, String> {
    static C: std::sync::OnceLock<Result<Option<Config>, String>> = std::sync::OnceLock::new();
    C.get_or_init(|| {
        let path = std::env::var("AD_WORD_PROBE").ok().map(|p| p.trim().to_string()).filter(|p| !p.is_empty() && p != "0");
        let text = match &path { Some(p) => Some(std::fs::read_to_string(p).map_err(|e| format!("AD_WORD_PROBE {p}: {e}"))?), None => None };
        parse(text.as_deref(), std::env::var("AD_WORD_PROBE_STEPS").ok().as_deref(), std::env::var("AD_WORD_PROBE_CMDS").ok().as_deref())
    }).as_ref().map_err(Clone::clone)
}

/// True when the switch is on and this command is probed (the window asks before it builds a command label).
pub fn cmd_on(sweep: &str, step: usize, index: usize) -> bool { matches!(config(), Ok(Some(c)) if c.wants_cmd(sweep, step, index)) }

/// The bits each stripe had at its last print, and how many points were visited (printed or not): two runs of one
/// command visit the same points in the same order, so the count lines their prints up. One process, one sweep at a time.
#[derive(Default)]
pub struct Seen { last: Vec<Option<Vec<u32>>>, points: u64 }
impl Seen {
    /// The stripes to print at this point: all of them at a step end or a seed (`all`), else those whose bits differ
    /// from their last print. What is returned is remembered as printed.
    pub fn take(&mut self, rows: Vec<Vec<u32>>, all: bool) -> Vec<(usize, Vec<u32>)> {
        self.points += 1;
        if self.last.len() < rows.len() { self.last.resize(rows.len(), None); }
        let mut out = Vec::new();
        for (i, w) in rows.into_iter().enumerate() {
            if all || self.last[i].as_ref() != Some(&w) { self.last[i] = Some(w.clone()); out.push((i, w)); }
        }
        out
    }
}

/// Float32 words as 8 hex digits each, no separator.
pub fn hex(words: &[u32]) -> String { let mut s = String::with_capacity(words.len() * 8); for w in words { s.push_str(&format!("{w:08x}")); } s }

/// The stderr line of one point, or None when no stripe changed at a command.
pub fn event(cfg: &Config, n: u64, sweep: &str, at: &str, step: usize, cmd: Option<(usize, &str)>, rows: &[(usize, Vec<u32>)], tag: &Value) -> Option<Value> {
    if rows.is_empty() { return None; }
    let r: Vec<Value> = rows.iter().map(|(i, w)| json!([cfg.stripes[*i].name, hex(w)])).collect();
    Some(json!({"event":"window_word_probe","n":n,"sweep":sweep,"at":at,"step":step,"index":cmd.map(|c| c.0),"command":cmd.map(|c| c.1),"tag":tag,"rows":r}))
}

/// One point: remember and build the line. `rows` holds every stripe of `cfg`, in order.
pub fn point(cfg: &Config, sweep: &str, at: &str, step: usize, cmd: Option<(usize, &str)>, rows: Vec<Vec<u32>>, tag: &Value) -> Option<Value> {
    static SEEN: std::sync::Mutex<Seen> = std::sync::Mutex::new(Seen { last: Vec::new(), points: 0 });
    let (take, n) = { let mut s = SEEN.lock().unwrap_or_else(|e| e.into_inner()); let t = s.take(rows, at != "command"); (t, s.points) };
    event(cfg, n, sweep, at, step, cmd, &take, tag)
}

/// How far two runs of one stripe are from exact proportion. `a` is the stripe at size s, `b` at size r s. Forward
/// sweep: b should be r a. Backward sweep of the self check: the seed is S / sum(S^2), so b should be a / r.
/// Returns (largest |dep|, the dep of the largest word) over the words whose |a| is at least 1e-3 of the stripe's
/// largest, with dep = scaled b / a - 1 in binary64; None when the stripe is all zero or not finite in `a`.
/// The compare tool (word_probe_cmp.py) uses this same definition; the test below pins one case in both.
#[allow(dead_code)]
pub fn departure(a: &[u32], b: &[u32], r: f64, backward: bool) -> Option<(f64, f64)> {
    let fa: Vec<f64> = a.iter().map(|w| f32::from_bits(*w) as f64).collect();
    let fb: Vec<f64> = b.iter().map(|w| f32::from_bits(*w) as f64).collect();
    let top = fa.iter().filter(|x| x.is_finite()).fold(0f64, |m, x| m.max(x.abs()));
    if !(top > 0.0) || fa.len() != fb.len() { return None; }
    let (mut worst, mut at_top) = (0f64, 0f64);
    for (x, y) in fa.iter().zip(&fb) {
        if !x.is_finite() || x.abs() < 1e-3 * top { continue; }
        let dep = if backward { y * r / x - 1.0 } else { y / (r * x) - 1.0 };
        if dep.abs() > worst { worst = dep.abs(); }
        if x.abs() == top { at_top = dep; }
    }
    Some((worst, at_top))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPEC: &str = r#"{"stripes":[{"name":"c5.th","allocation":391,"first_word":5,"stride_words":10,"count":3},
                                      {"name":"c5.tsq","allocation":431,"first_word":105,"stride_words":10,"count":2}]}"#;

    #[test]
    fn off_unless_set_and_refuses_half_set_switches() {
        assert_eq!(parse(None, None, None).unwrap(), None);
        assert_eq!(parse(None, Some(""), Some(" ")).unwrap(), None);
        assert!(parse(None, Some("1"), None).unwrap_err().contains("AD_WORD_PROBE is not"));
        assert!(parse(None, None, Some("3:9")).unwrap_err().contains("AD_WORD_PROBE is not"));
        assert!(parse(Some(SPEC), None, Some("3:9")).unwrap_err().contains("needs AD_WORD_PROBE_STEPS"));
        assert!(parse(Some(SPEC), Some("one"), None).unwrap_err().contains("step numbers"));
        assert!(parse(Some(SPEC), Some("1"), Some("9:3")).unwrap_err().contains("lo <= hi"));
    }

    #[test]
    fn spec_is_read_and_bad_specs_are_refused_by_name() {
        let c = parse(Some(SPEC), Some("1, 2"), Some("350:4700")).unwrap().unwrap();
        assert_eq!(c.stripes.len(), 2);
        assert_eq!(c.stripes[0], Stripe { name: "c5.th".into(), alloc: 391, first: 5, stride: 10, count: 3 });
        assert_eq!(c.stripes[0].words().collect::<Vec<_>>(), vec![5, 15, 25]);
        assert_eq!((c.cmd_steps.clone(), c.cmd_range), (vec![1, 2], Some((350, 4700))));
        assert!(parse_spec("{}").unwrap_err().contains("no \"stripes\""));
        assert!(parse_spec(r#"{"stripes":[]}"#).unwrap_err().contains("empty"));
        assert!(parse_spec(r#"{"stripes":[{"name":"a","allocation":1,"first_word":0,"stride_words":0,"count":2}]}"#).unwrap_err().contains("stride_words is 0"));
        assert!(parse_spec(r#"{"stripes":[{"name":"a","allocation":1,"first_word":0,"count":0}]}"#).unwrap_err().contains("count is 0"));
        assert!(parse_spec(r#"{"stripes":[{"name":"a b","allocation":1,"first_word":0,"count":1}]}"#).unwrap_err().contains("one word"));
        assert!(parse_spec(r#"{"stripes":[{"name":"a","allocation":1,"first_word":0,"count":1},{"name":"a","allocation":2,"first_word":0,"count":1}]}"#).unwrap_err().contains("used twice"));
        assert!(parse_spec(r#"{"stripes":[{"name":"a","allocation":1,"first_word":0,"stride_words":1,"count":200001}]}"#).unwrap_err().contains("more than"));
        // a single word needs no stride
        assert_eq!(parse_spec(r#"{"stripes":[{"name":"a","allocation":1,"first_word":7,"count":1}]}"#).unwrap()[0].words().collect::<Vec<_>>(), vec![7]);
    }

    #[test]
    fn a_stripe_past_its_allocation_is_refused() {
        let s = Stripe { name: "c".into(), alloc: 9, first: 5, stride: 10, count: 3 };   // last word 25
        assert!(s.check(26 * 4).is_ok());
        let e = s.check(25 * 4).unwrap_err();
        assert!(e.contains("stripe c") && e.contains("word 25") && e.contains("allocation 9"), "{e}");
    }

    #[test]
    fn commands_are_chosen_by_step_and_forward_range() {
        let c = parse(Some(SPEC), Some("1"), Some("350:4700")).unwrap().unwrap();
        assert!(c.wants_cmd("tangent", 1, 350) && c.wants_cmd("tangent", 1, 4700));
        assert!(!c.wants_cmd("tangent", 1, 349) && !c.wants_cmd("tangent", 1, 4701) && !c.wants_cmd("tangent", 2, 400));
        // the backward sweep numbers commands inside a segment: the plan range does not apply there
        assert!(c.wants_cmd("reverse", 1, 3) && !c.wants_cmd("reverse", 2, 3));
        let all = parse(Some(SPEC), Some("1"), None).unwrap().unwrap();
        assert!(all.wants_cmd("tangent", 1, 0) && all.wants_cmd("tangent", 1, 999_999));
    }

    #[test]
    fn only_changed_stripes_are_printed_at_commands_and_all_at_step_ends() {
        let c = parse(Some(SPEC), Some("1"), None).unwrap().unwrap();
        let mut seen = Seen::default();
        let a = vec![vec![1u32, 2, 3], vec![7u32, 8]];
        let first = seen.take(a.clone(), false);
        assert_eq!(first.len(), 2, "nothing was printed before, so both stripes are new");
        assert!(seen.take(a.clone(), false).is_empty(), "same bits: nothing to print");
        let mut b = a.clone(); b[1][0] = 0x8000_0007;                    // one bit of one word
        let ch = seen.take(b.clone(), false);
        assert_eq!(ch, vec![(1usize, vec![0x8000_0007u32, 8])]);
        assert_eq!(seen.take(b.clone(), true).len(), 2, "a step end prints every stripe");
        assert!(event(&c, 4, "tangent", "command", 1, Some((357, "prep_mass_v841_f32")), &[], &Value::Null).is_none());
        let e = event(&c, seen.points, "reverse", "command", 1, Some((12, "rrfs_mynnedmf_run_l16")), &ch, &json!({"seed":1})).unwrap();
        assert_eq!(e["n"].as_u64(), Some(4), "four points were visited, printed or not");
        assert_eq!(e["rows"], json!([["c5.tsq", "8000000700000008"]]));
        assert_eq!((e["sweep"].as_str(), e["at"].as_str(), e["step"].as_u64(), e["index"].as_u64()), (Some("reverse"), Some("command"), Some(1), Some(12)));
        assert_eq!(hex(&[0x3f80_0000, 0, 0xffff_ffff]), "3f80000000000000ffffffff");
    }

    /// The rounding effect the probe is built to find, on the stored words. Two neighbouring levels of one column
    /// (level 14 and 13 of cell 38573, block 375 of the 3-hour 120 km window, size 1: the two x_a values in
    /// <local path> test machine.txt) enter a kernel as single-precision words and leave it as
    /// gain * (upper - lower): the form a level-to-level temperature difference takes in a boundary-layer scheme.
    /// At 3 or 13 times the size every input word is still its twin to one rounding (6e-08), but the difference is
    /// not: the two inputs are close, so their rounding is 20 times larger against what is left. The probe's measure
    /// must show the inputs as clean and the output as departed, which names the kernel between them.
    #[test]
    fn a_level_difference_amplifies_single_precision_rounding_and_the_probe_shows_where() {
        let (x14, x13) = (-1.2692729e-04f32, -1.1495597e-04f32);
        let run = |r: f32| -> (Vec<u32>, Vec<u32>) {
            let (a, b) = (x14 * r, x13 * r);                             // the direction at size r, rounded to float32
            let gain = 49290.613f32;                                     // the stored backward value at that word
            let out = gain * (a - b);                                    // the kernel: a level difference with a large gain
            (vec![a.to_bits(), b.to_bits()], vec![out.to_bits()])
        };
        let ((in1, out1), (in3, out3), (in13, out13)) = (run(1.0), run(3.0), run(13.0));
        let cancel = (x14.abs() + x13.abs()) as f64 / (x14 - x13).abs() as f64;
        assert!(cancel > 15.0 && cancel < 25.0, "the two stored words cancel by a factor near 20: {cancel}");
        let mut seen_amplified = 0;
        for (inp, out, r) in [(&in3, &out3, 3.0f64), (&in13, &out13, 13.0)] {
            let (din, _) = departure(&in1, inp, r, false).unwrap();
            let (dout, _) = departure(&out1, out, r, false).unwrap();
            assert!(din <= 6.0e-08, "each input word is its twin to one rounding: {din}");
            assert!(dout <= cancel * 1.3e-07, "the difference departs by no more than the cancellation allows: {dout}");
            if dout >= 4.0 * din.max(1.5e-08) { seen_amplified += 1; }
        }
        assert!(seen_amplified >= 1, "at size 3 or 13 the difference departs by several times its inputs");
        // a factor 2 changes no rounding at all: the stored control (r3-scale120/PROOF.txt) in one line
        let (in2, out2) = run(2.0);
        assert_eq!(departure(&in1, &in2, 2.0, false).unwrap().0, 0.0);
        assert_eq!(departure(&out1, &out2, 2.0, false).unwrap().0, 0.0);
        // the same definition serves the backward sweep (values scale by 1 / r there): the stored pair of that word
        let (a, b) = (-4.9290613e+04f32, -4.9290492e+04f32 / 3.0);
        let (d, top) = departure(&[a.to_bits()], &[b.to_bits()], 3.0, true).unwrap();
        assert!(d > 2.0e-06 && d < 3.0e-06 && top < 0.0, "the stored word departs by 2.5e-06: {d} {top}");
        // pinned in word_probe_cmp.py --selftest as well, so the two definitions cannot drift apart
        let pin = departure(&[0x3f80_0000, 0x4000_0000], &[0x4040_0001, 0x40c0_0000], 3.0, false).unwrap();
        assert!((pin.0 - 7.947285970052083e-08).abs() < 1e-15 && pin.1 == 0.0, "{pin:?}");
    }
}
