// SPDX-License-Identifier: Apache-2.0
//! step-hold (`ad-binder window ... --checkpoints N --bstore-ram-gib G --step-hold FILE`, default off).
//!
//! Every intermediate step-start state of a checkpointed window is held, so the reverse sweep re-runs no step between
//! the steps it reverses. A held state is not the whole live state (8,385,001,472 bytes at the 3 km storm box): it is
//! the spans the next step reads, as a hold set names them (tools/step_hold/mkset.py, from a development line nr-probe):
//!   * `const`: allocations whose bytes are the same at every step boundary of the recording. Boundary 0 holds them
//!     (whole); nothing writes them, so no restore puts them back. Checked against boundary 0 after every sweep.
//!   * `drop`: [allocation, offset, bytes] ranges every step writes before it reads them (nr-probe poisoned them with
//!     NaN and with 0.5 and the next steps stayed equal word for word). A restore leaves them as the card has them.
//!   * everything else live at the boundary is held, an allocation the set does not know included.
//! With the set C2-0527 a held state is 2,679,377,352 bytes; with C1-0537 4,498,619,560 (both are mkset.py's
//! arithmetic over the recording's allocation sizes). The reverse step's redo check still compares every held word of
//! the re-run end state with the held state of the next boundary, and a run with a set that leaves out a word some
//! step reads refuses there (or at the constant check), never answers.
//! Boundary 0 and the window end stay whole: the dots, the directions and the window-end dumps read them.

use std::collections::{HashMap, HashSet};

/// A parsed hold set (the JSON mkset.py writes).
pub(crate) struct HoldSet {
    pub name: String,
    pub sha256: String,
    /// allocation -> bytes, as the recording had them: a live allocation of another size refuses (a set made for
    /// another plan).
    pub sizes: HashMap<u64, usize>,
    pub consts: HashSet<u64>,
    /// allocation -> its dropped byte ranges [lo, hi), ascending and disjoint.
    pub drop: HashMap<u64, Vec<(usize, usize)>>,
    pub drop_bytes: usize,
}

impl HoldSet {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        use sha2::Digest;
        let doc: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| format!("--step-hold: {e}"))?;
        let name = doc["name"].as_str().unwrap_or("").to_owned();
        let mut sizes = HashMap::new();
        if let Some(m) = doc["sizes"].as_object() {
            for (k, v) in m {
                let id: u64 = k.parse().map_err(|_| format!("--step-hold: sizes key {k} is not an allocation"))?;
                sizes.insert(id, v.as_u64().ok_or_else(|| format!("--step-hold: sizes[{k}] is not a byte count"))? as usize);
            }
        }
        let consts: HashSet<u64> = doc["const"].as_array().ok_or("--step-hold: no const list")?.iter()
            .map(|v| v.as_u64().ok_or_else(|| "--step-hold: const holds a non-allocation".to_string())).collect::<Result<_, _>>()?;
        let mut drop: HashMap<u64, Vec<(usize, usize)>> = HashMap::new();
        let mut drop_bytes = 0usize;
        for r in doc["drop"].as_array().ok_or("--step-hold: no drop list")? {
            let (Some(id), Some(o), Some(n)) = (r[0].as_u64(), r[1].as_u64(), r[2].as_u64()) else { return Err(format!("--step-hold: drop entry {r} is not [allocation, offset, bytes]")) };
            if consts.contains(&id) { return Err(format!("--step-hold: allocation {id} is both constant and dropped from")); }
            if let Some(&z) = sizes.get(&id) { if (o + n) as usize > z { return Err(format!("--step-hold: drop [{id}, {o}, {n}] runs past the allocation's {z} bytes")); } }
            if n > 0 { drop.entry(id).or_default().push((o as usize, (o + n) as usize)); drop_bytes += n as usize; }
        }
        for (id, rs) in drop.iter_mut() {
            rs.sort_unstable();
            if rs.windows(2).any(|w| w[0].1 > w[1].0) { return Err(format!("--step-hold: drop ranges of allocation {id} overlap")); }
        }
        let sha256 = format!("{:x}", sha2::Sha256::digest(bytes));
        Ok(Self { name, sha256, sizes, consts, drop, drop_bytes })
    }

    /// The held spans of a boundary whose live allocations are `live` (allocation, bytes; ascending allocation):
    /// (allocation, offset in it, bytes), ascending. An allocation the set sizes differently refuses.
    pub fn spans(&self, live: &[(u64, usize)]) -> Result<Vec<(u64, usize, usize)>, String> {
        let mut out = Vec::new();
        for &(id, bytes) in live {
            if let Some(&z) = self.sizes.get(&id) {
                if z != bytes { return Err(format!("--step-hold: allocation {id} holds {bytes} bytes, the hold set {z}: the set is for another plan")); }
            }
            if self.consts.contains(&id) { continue; }
            let mut at = 0usize;
            for &(lo, hi) in self.drop.get(&id).map(|v| v.as_slice()).unwrap_or(&[]) {
                if hi > bytes { return Err(format!("--step-hold: a drop range of allocation {id} runs past its {bytes} bytes")); }
                if lo > at { out.push((id, at, lo - at)); }
                at = at.max(hi);
            }
            if bytes > at { out.push((id, at, bytes - at)); }
        }
        Ok(out)
    }
}

/// The held bytes' layout of `spans`: the boundary index (allocation, offset in the held bytes, bytes), one entry per
/// span, and each entry's offset inside its allocation (Boundary::sub).
pub(crate) fn layout(spans: &[(u64, usize, usize)]) -> (Vec<(u64, usize, usize)>, Vec<usize>) {
    let mut at = 0usize;
    let mut index = Vec::with_capacity(spans.len());
    let mut sub = Vec::with_capacity(spans.len());
    for &(id, o, n) in spans { index.push((id, at, n)); sub.push(o); at += n; }
    (index, sub)
}

/// An allocation's declared residue ranges (byte offset from its start, bytes) seen from its span [sub, sub + n): the
/// parts inside the span, measured from the span's start.
pub(crate) fn shift_ranges(r: &[(usize, usize)], sub: usize, n: usize) -> Vec<(usize, usize)> {
    r.iter().filter_map(|&(o, l)| {
        let (a, b) = (o.max(sub), (o + l).min(sub + n));
        (a < b).then(|| (a - sub, b - a))
    }).collect()
}

/// AD_PLANT_HOLD=<boundary>[:<factor>] (TEST ONLY): the held state of that boundary has one float32 word scaled by
/// the factor (default 1.001) right after it is stored. THE BREAKAGE THIS PREVENTS: a held state that lost or changed
/// a word would restore a wrong start state; with nothing planted no run could show that the redo check refuses it. The
/// sweep meets it first at the step that starts from that boundary (its re-run end differs from the next held state),
/// so a planted run must stop there with window_redo_differs. A run with it set is never valid (window.json
/// planted_error from the store on).
pub(crate) fn plant_parse(v: &str) -> Option<(usize, f64)> {
    let mut it = v.splitn(2, ':');
    let s = it.next()?.trim().parse::<usize>().ok()?;
    let f = match it.next() { None => 1.001, Some(x) => x.trim().parse::<f64>().ok()? };
    (s >= 1 && f.is_finite() && f != 1.0).then_some((s, f))
}
pub(crate) fn plant() -> &'static Option<(usize, f64)> {
    static P: std::sync::OnceLock<Option<(usize, f64)>> = std::sync::OnceLock::new();
    P.get_or_init(|| plant_parse(&std::env::var("AD_PLANT_HOLD").ok()?))
}
/// Scales the first finite float32 word of magnitude 1e-30 or more in `words` by `f`; (word index, bits before, after).
pub(crate) fn plant_word(words: &mut [u8], f: f64) -> Option<(usize, u32, u32)> {
    let (i, b) = words.chunks_exact(4).enumerate().find_map(|(i, c)| {
        let b = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        let v = f32::from_bits(b);
        (v.is_finite() && v.abs() >= 1e-30).then_some((i, b))
    })?;
    let a = ((f32::from_bits(b) as f64) * f) as f32;
    let a = if a.to_bits() == b { f32::from_bits(b ^ 1) } else { a };
    words[4 * i..4 * i + 4].copy_from_slice(&a.to_le_bytes());
    Some((i, b, a.to_bits()))
}
