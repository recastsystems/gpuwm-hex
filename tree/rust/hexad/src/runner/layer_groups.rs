// SPDX-License-Identifier: Apache-2.0
//! Live host rules for the RRTMG longwave wrapper's layer grouping.
//!
//! These rules mirror rrtmg_lw_v842.py:_layer_groups and _lw_wrapper_chunk.
//! The caller must obtain the histogram, optional counts, optional indices,
//! and free-memory count from the current replay, never from a recording.
//! Merely having these rules does not establish a complete physics replay.

use std::collections::BTreeMap;

pub const HISTOGRAM_BINS: usize = 1024;
const WRAPPER_WORDS_PER_LAYER: usize = 5 * 140 + 12 + 2 + 16 + 2 * 12 + 8;
const WRAPPER_BUDGET_BYTES: usize = 1 << 30;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GroupIndices {
    /// A view into the live, stable device argsort result.
    SortedView { offset: usize, length: usize },
    /// The overflow path constructs a new device array from these live indices.
    Explicit(Vec<i32>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerGroup {
    pub layers: i64,
    pub indices: GroupIndices,
    pub members: Option<Vec<i32>>,
}

/// Reproduce _layer_groups after the histogram and stable argsort have run.
/// `counts` is required only when the overflow bin is nonzero. `sorted` is
/// required only when the caller requests host members on the normal path.
pub fn groups(nc: usize, nm: i32, histogram: &[i32], counts: Option<&[i32]>,
              sorted: Option<&[i32]>, need_members: bool) -> Result<Vec<LayerGroup>, String> {
    if nc == 0 { return Ok(Vec::new()); }
    if nc > i32::MAX as usize { return Err("column indices exceed int32".into()); }
    if histogram.len() != HISTOGRAM_BINS + 1 { return Err("histogram size mismatch".into()); }
    let base = nm.checked_add(1).ok_or("layer base exceeds int32")?;
    let total = histogram.iter().try_fold(0_usize, |sum, &size| {
        let size = usize::try_from(size).map_err(|_| "negative histogram count")?;
        sum.checked_add(size).ok_or("histogram count overflow")
    })?;
    if total != nc { return Err("histogram total differs from column count".into()); }
    if histogram[HISTOGRAM_BINS] != 0 {
        let counts = counts.ok_or("overflow requires live counts")?;
        if counts.len() != nc { return Err("counts size mismatch".into()); }
        let mut found: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
        for (column, &nl) in counts.iter().enumerate() {
            found.entry(nl).or_default().push(column as i32);
        }
        return Ok(found.into_iter().map(|(nl, indices)| LayerGroup {
            layers: i64::from(nl),
            members: need_members.then(|| indices.clone()),
            indices: GroupIndices::Explicit(indices),
        }).collect());
    }
    let sorted = if need_members {
        let values = sorted.ok_or("host members require live stable-sort indices")?;
        if values.len() != nc { return Err("stable-sort indices size mismatch".into()); }
        Some(values)
    } else { None };
    let mut result = Vec::new();
    let mut offset = 0_usize;
    for (bin, &size) in histogram[..HISTOGRAM_BINS].iter().enumerate() {
        if size == 0 { continue; }
        let length = size as usize;
        let end = offset.checked_add(length).ok_or("group offset overflow")?;
        let members = sorted.map(|s| s[offset..end].to_vec());
        result.push(LayerGroup { layers: i64::from(base) + bin as i64,
            indices: GroupIndices::SortedView { offset, length }, members });
        offset = end;
    }
    Ok(result)
}

/// Reproduce _lw_wrapper_chunk using the device's current free-byte count.
pub fn wrapper_chunk(ncol: usize, nlayers: i64, free_bytes: usize) -> Result<usize, String> {
    let layers = usize::try_from(nlayers.max(1)).map_err(|_| "layer count overflow")?;
    let per_column = 4_usize.checked_mul(WRAPPER_WORDS_PER_LAYER)
        .and_then(|v| v.checked_mul(layers))
        .ok_or("longwave column size overflow")?;
    let budget = WRAPPER_BUDGET_BYTES.min(free_bytes / 8);
    Ok(ncol.max(1).min(256_usize.max(budget / per_column)))
}

/// G2 (P1 milestone 2): the wrapper batches rrtmg_lwrad_device launches for one call, rebuilt from the live
/// histogram: _layer_groups' groups (ascending layer count), each split into batches of at most `chunk` columns
/// (rrtmg_lw_v842.py:1157-1160). `chunk` is the forward's _lw_wrapper_chunk width; the replay takes it from the
/// recorded batches (their largest), never from this process's free memory, because the plan's launches were
/// recorded at that width.
pub fn batches(nc: usize, nm: i32, histogram: &[i32], chunk: usize) -> Result<Vec<[i64; 2]>, String> {
    if chunk == 0 { return Err("wrapper chunk is zero".into()); }
    let mut out = Vec::new();
    for group in groups(nc, nm, histogram, None, None, false)? {
        let length = match group.indices { GroupIndices::SortedView { length, .. } => length,
                                           GroupIndices::Explicit(ref v) => v.len() };
        let mut first = 0;
        while first < length { let n = chunk.min(length - first); out.push([group.layers, n as i64]); first += n; }
    }
    Ok(out)
}

/// G2: check a plan's recorded LW batches against the live histogram (the replay's read-back).
pub fn check_recorded(nc: usize, nm: i32, histogram: &[i32], recorded: &[[i64; 2]]) -> Result<(), String> {
    let chunk = recorded.iter().map(|b| b[1]).max().ok_or("no recorded LW batches")?;
    let chunk = usize::try_from(chunk).map_err(|_| "negative recorded batch")?;
    let live = batches(nc, nm, histogram, chunk)?;
    if live != recorded {
        return Err(format!("LW wrapper batches from the live histogram {live:?} differ from the plan's {recorded:?}"));
    }
    Ok(())
}

/// S13: column orders that let a plan's recorded LW batches compute a live grouping the plan did not record.
///
/// `live_order` is the live stable argsort of the live counts, so the live histogram's bins cut it into the live
/// groups (ascending layer count). Each recorded batch [layers, n] reads `n` order entries at its start and runs
/// them at `layers`. Every pass fills every batch with live columns of the batch's own layer count (pass p, slot
/// k of layer L's capacity cap_L: column (p * cap_L + k) mod m_L), so a column is only ever computed at its live
/// layer count and repeats are the same column's same words. Passes = max over L of ceil(m_L / cap_L).
/// Refused: the overflow bin, a live layer count with no recorded batch, a recorded batch with no live column.
/// `test` (AD_LW_REGROUP_TEST, diagnostic; 0 = off): 1: every pass is rotated by m_L / 2 and one more pass covers
/// only the first half of each group (repeats); its step hashes differ in the cupy pool scratch the batches leave;
/// 2: the rotated passes, then the unmodified live order as the last pass: the reference step hashes must hold
/// (a test machine S13-T2: 5 LW calls in 7 steps, every hash equal); 3: the unmodified live order first, then one rotated
/// pass; the window compares the LW output arrays of the two (S13-T3: 632,814 words, 0 differ).
pub fn regroup_passes(nc: usize, nm: i32, live_hist: &[i32], live_order: &[i32], recorded: &[[i64; 2]], test: u8)
                      -> Result<Vec<Vec<i32>>, String> {
    let test_mode = test;
    let test = test_mode != 0;
    if live_hist.len() != HISTOGRAM_BINS + 1 { return Err("histogram size mismatch".into()); }
    if live_hist[HISTOGRAM_BINS] != 0 { return Err("live histogram overflow bin is not regrouped".into()); }
    if live_order.len() != nc { return Err("live order size mismatch".into()); }
    let base = i64::from(nm) + 1;
    let mut live: BTreeMap<i64, &[i32]> = BTreeMap::new();
    let mut offset = 0usize;
    for (bin, &size) in live_hist[..HISTOGRAM_BINS].iter().enumerate() {
        let size = usize::try_from(size).map_err(|_| "negative histogram count")?;
        if size == 0 { continue; }
        let end = offset.checked_add(size).filter(|&e| e <= nc).ok_or("histogram total exceeds column count")?;
        live.insert(base + bin as i64, &live_order[offset..end]);
        offset = end;
    }
    if offset != nc { return Err("histogram total differs from column count".into()); }
    let mut cap: BTreeMap<i64, usize> = BTreeMap::new();
    let mut total = 0usize;
    for b in recorded {
        let n = usize::try_from(b[1]).ok().filter(|&n| n > 0).ok_or("recorded batch is empty")?;
        *cap.entry(b[0]).or_default() += n;
        total += n;
    }
    if total != nc { return Err(format!("recorded batches cover {total} of {nc} columns")); }
    for (l, cols) in &live {
        if !cap.contains_key(l) { return Err(format!("live layer count {l} ({} columns) has no recorded batch", cols.len())); }
    }
    for l in cap.keys() {
        if !live.contains_key(l) { return Err(format!("recorded layer count {l} has no live column")); }
    }
    // the live grouping is the recorded one (then the batches' segments are the live order's own)
    let same = live.len() == cap.len() && live.iter().all(|(l, c)| cap.get(l) == Some(&c.len()));
    let mut passes = live.iter().map(|(l, c)| c.len().div_ceil(cap[l])).max().unwrap_or(1);
    if test { passes += 1; }
    let mut out = Vec::with_capacity(passes);
    for p in 0..passes {
        let mut order = Vec::with_capacity(nc);
        let mut used: BTreeMap<i64, usize> = BTreeMap::new();
        for b in recorded {
            let (cols, c) = (live[&b[0]], cap[&b[0]]);
            let m = cols.len();
            let k0 = used.entry(b[0]).or_default();
            for k in *k0..*k0 + b[1] as usize {
                let j = if test_mode == 3 && same && p == 0 { usize::MAX }
                        else if test_mode == 3 && same { (k + m / 2) % m }
                        else if test_mode == 2 && same && p + 1 == passes { usize::MAX }
                        else if test && p + 1 == passes { k % (m / 2).max(1) }
                        else if test { (p * c + k + m / 2) % m }
                        else { (p * c + k) % m };
                order.push(if j == usize::MAX { live_order[order.len()] } else { cols[j] });
            }
            *k0 += b[1] as usize;
        }
        out.push(order);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    /// S13: one column moves between the recorded groups (the 2200 Pa NINT tie): two passes, each column only at
    /// its live layer count, every live column covered, repeats only of the same group's columns.
    #[test]
    fn regroup_one_column_moved_up_and_down() {
        let (nc, nm) = (10usize, 59);
        let recorded = [[64i64, 7], [65, 3]];
        for (n64, n65) in [(6usize, 4usize), (8, 2)] {
            let mut h = vec![0; HISTOGRAM_BINS + 1];
            h[4] = n64 as i32; h[5] = n65 as i32;
            let order: Vec<i32> = (0..nc as i32).rev().collect();
            let passes = regroup_passes(nc, nm, &h, &order, &recorded, 0).unwrap();
            assert_eq!(passes.len(), 2);
            let (c64, c65) = (&order[..n64], &order[n64..]);
            let mut covered = std::collections::BTreeSet::new();
            for p in &passes {
                assert_eq!(p.len(), nc);
                assert!(p[..7].iter().all(|c| c64.contains(c)));
                assert!(p[7..].iter().all(|c| c65.contains(c)));
                covered.extend(p.iter().copied());
            }
            assert_eq!(covered.len(), nc);
        }
        // the recorded grouping itself: one pass, the live order unchanged
        let mut h = vec![0; HISTOGRAM_BINS + 1];
        h[4] = 7; h[5] = 3;
        let order: Vec<i32> = (0..10).collect();
        assert_eq!(regroup_passes(nc, nm, &h, &order, &recorded, 0).unwrap(), vec![order.clone()]);
        let t2 = regroup_passes(nc, nm, &h, &order, &recorded, 2).unwrap();
        assert_eq!(t2.len(), 2);
        assert_ne!(t2[0], order);
        assert_eq!(t2[1], order);
        // test mode: rotated full pass, then a partial pass of repeats
        let t = regroup_passes(nc, nm, &h, &order, &recorded, 1).unwrap();
        assert_eq!(t.len(), 2);
        assert_ne!(t[0], order);
        assert_eq!(t[0].iter().copied().collect::<std::collections::BTreeSet<_>>().len(), nc);
        assert_eq!(t[1], vec![0, 1, 2, 0, 1, 2, 0, 7, 7, 7]);
    }
    #[test]
    fn regroup_refuses_what_the_batches_cannot_compute() {
        let recorded = [[64i64, 7], [65, 3]];
        let order: Vec<i32> = (0..10).collect();
        let mut h = vec![0; HISTOGRAM_BINS + 1];
        h[4] = 7; h[6] = 3;
        assert!(regroup_passes(10, 59, &h, &order, &recorded, 0).unwrap_err().contains("live layer count 66"));
        let mut h = vec![0; HISTOGRAM_BINS + 1];
        h[4] = 10;
        assert!(regroup_passes(10, 59, &h, &order, &recorded, 0).unwrap_err().contains("recorded layer count 65"));
        let mut h = vec![0; HISTOGRAM_BINS + 1];
        h[4] = 7; h[HISTOGRAM_BINS] = 3;
        assert!(regroup_passes(10, 59, &h, &order, &recorded, 0).is_err());
    }
    #[test]
    fn normal_groups_are_live_views_and_preserve_member_order() {
        let mut h = vec![0; HISTOGRAM_BINS + 1];
        h[0] = 2; h[3] = 1;
        let sorted = [2, 0, 1];
        let result = groups(3, 59, &h, None, Some(&sorted), true).unwrap();
        assert_eq!(result, vec![
            LayerGroup { layers: 60, indices: GroupIndices::SortedView { offset: 0, length: 2 }, members: Some(vec![2, 0]) },
            LayerGroup { layers: 63, indices: GroupIndices::SortedView { offset: 2, length: 1 }, members: Some(vec![1]) },
        ]);
        assert!(groups(3, 59, &h, None, None, true).is_err());
        assert!(groups(3, 59, &h, None, None, false).unwrap().iter().all(|g| g.members.is_none()));
    }
    #[test]
    fn overflow_uses_live_counts_and_sorts_group_keys() {
        let mut h = vec![0; HISTOGRAM_BINS + 1];
        h[0] = 1; h[HISTOGRAM_BINS] = 3;
        let result = groups(4, 59, &h, Some(&[60, 5000, 2, 5000]), None, true).unwrap();
        assert_eq!(result.iter().map(|g| g.layers).collect::<Vec<_>>(), [2, 60, 5000]);
        assert_eq!(result[2].indices, GroupIndices::Explicit(vec![1, 3]));
        assert_eq!(result[2].members, Some(vec![1, 3]));
        assert!(groups(4, 59, &h, None, None, false).is_err());
    }
    #[test]
    fn empty_and_invalid_inputs_fail_closed() {
        assert!(groups(0, 1, &[], None, None, false).unwrap().is_empty());
        assert!(groups(1, 1, &[], None, None, false).is_err());
        let mut h = vec![0; HISTOGRAM_BINS + 1];
        h[0] = -1;
        assert!(groups(1, 1, &h, None, None, false).is_err());
    }
    /// G2 (P1 milestone 2) acceptance 3: the Rust port reproduces the recorded LW groups at every radiation step
    /// of the 18-step hrrrv5 x1.2562 forecast (recorder observer observe_layer_groups.py on the radiation-live
    /// recording r18a: the counts each call grouped, the groups and their members the pinned Python returned, and
    /// every _lw_wrapper_chunk call with its memGetInfo reading). Groups, members (stable order), batches and the
    /// chunk all match; a histogram with one column moved is refused by check_recorded.
    #[test]
    fn recorded_hrrrv5_radiation_steps_reproduce_lw_groups_and_chunks() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!("fixtures/lw_layer_groups_x1_2562_hrrrv5_g2.json")).unwrap();
        let calls = fixture["calls"].as_array().unwrap();
        assert_eq!(calls.len(), 12);
        for call in calls {
            let step = call["step"].as_i64().unwrap();
            let nm = call["nm"].as_i64().unwrap() as i32;
            let counts: Vec<i32> = call["counts"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect();
            let nc = counts.len();
            let mut hist = vec![0; HISTOGRAM_BINS + 1];
            for &c in &counts { let b = c - (nm + 1); hist[if (0..HISTOGRAM_BINS as i32).contains(&b) { b as usize } else { HISTOGRAM_BINS }] += 1; }
            let mut sorted: Vec<i32> = (0..nc as i32).collect();
            sorted.sort_by_key(|&c| counts[c as usize]);  // stable, as cp.argsort(kind='stable')
            let got = groups(nc, nm, &hist, None, Some(&sorted), true).unwrap();
            let want = call["groups"].as_array().unwrap();
            assert_eq!(got.len(), want.len(), "step {step}");
            let mut recorded = Vec::new();
            for (g, w) in got.iter().zip(want) {
                let members: Vec<i32> = w[1].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect();
                assert_eq!(g.layers, w[0].as_i64().unwrap(), "step {step}");
                assert_eq!(g.members.as_ref().unwrap(), &members, "step {step} members");
                recorded.push([g.layers, members.len() as i64]);
            }
            assert_eq!(batches(nc, nm, &hist, nc).unwrap(), recorded, "step {step}");
            check_recorded(nc, nm, &hist, &recorded).unwrap();
            let mut moved = hist.clone();
            let first = moved.iter().position(|&n| n > 0).unwrap();
            moved[first] -= 1; moved[first + 1] += 1;
            assert!(check_recorded(nc, nm, &moved, &recorded).is_err(), "step {step}: a moved column must refuse");
        }
        for c in fixture["chunks"].as_array().unwrap() {
            let (ncol, nl, result) = (c["ncol"].as_u64().unwrap() as usize, c["nlayers"].as_i64().unwrap(), c["result"].as_u64().unwrap() as usize);
            let free = c["free"].as_array().unwrap();
            assert_eq!(free.len(), 1);
            assert_eq!(wrapper_chunk(ncol, nl, free[0].as_u64().unwrap() as usize).unwrap(), result, "step {}", c["step"]);
        }
    }
    #[test]
    fn chunk_uses_current_free_memory() {
        assert_eq!(wrapper_chunk(0, 60, 0).unwrap(), 1);
        assert_eq!(wrapper_chunk(10000, 60, 0).unwrap(), 256);
        assert_eq!(wrapper_chunk(100, 0, usize::MAX).unwrap(), 100);
        assert_eq!(wrapper_chunk(10000, 60, 8 * (1 << 30)).unwrap(), (1 << 30) / (4 * 762 * 60));
    }
    #[test]
    fn real_hrrrv5_x1_2562_counts_give_the_recorded_lw_groups() {
        // mpas_layer_counts on the real captured inputs (held-mpas_layer_counts, hrrrv5_x1.2562, nm = 59, nc = 2562):
        // counts take only 64 and 65 (top pressure 1874-2220 Pa, tie at 2200 Pa). The groups must be the two
        // longwave launch shapes the live forward ran (held-rlw_inatm recapture: 2499x64 + 63x65 at step 1,
        // 2496x64 + 66x65 at step 6, 2525x64 + 37x65 at step 12; shared capture step 3: 2501 + 61).
        for (step, n64, n65) in [(1, 2499, 63), (3, 2501, 61), (6, 2496, 66), (12, 2525, 37)] {
            let mut h = vec![0; HISTOGRAM_BINS + 1];
            h[64 - 60] = n64; h[65 - 60] = n65;
            let g = groups(2562, 59, &h, None, None, false).unwrap();
            assert_eq!(g, vec![
                LayerGroup { layers: 64, indices: GroupIndices::SortedView { offset: 0, length: n64 as usize }, members: None },
                LayerGroup { layers: 65, indices: GroupIndices::SortedView { offset: n64 as usize, length: n65 as usize }, members: None },
            ], "step {step}");
        }
    }
}
