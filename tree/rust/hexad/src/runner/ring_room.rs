// SPDX-License-Identifier: Apache-2.0
//! ring-60km: the MYNN-EDMF ring tape plan (layout 3) and the room it can be given. No arithmetic on weather fields.
//!
//! The reverse launch packs its column groups in order into one ring of `ring` nodes; a block starts group g once
//! vstart[g] + size[g] <= vstart[low] + ring, low = the first group not yet done (gen_whole.py mw_persist). So the groups
//! in flight at any time are the ones whose tape fits in one ring length from `low`, whatever the number of laps: a
//! launch split into batches whose tape fits the ring once has the same groups in flight at each batch's start and fewer
//! at its end, so it is not used. What raises the groups in flight is ring room. These functions compute the plan the
//! kernel reads, the groups the ring keeps in flight, and the ring that keeps every slot busy, from the counts.

/// Each group's tape in nodes: the sum over its `lanes` consecutive columns of count + 16 (a negative count is 0 + 16),
/// as the count pass's largest-group figure and the kernel's per-group prefix take it.
pub fn group_sizes(counts: &[i64], lanes: usize) -> Vec<u64> {
    let lanes = lanes.max(1);
    counts.chunks(lanes).map(|g| g.iter().map(|&u| u.max(0) as u64 + 16).sum()).collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingPlan {
    /// virtual start of each group, then the virtual end (groups + 1 values, monotone)
    pub vstart: Vec<u64>,
    /// vstart mod ring: where the group's tape starts in the ring
    pub phys: Vec<u64>,
    /// whole laps of the ring the launch makes (the receipt's ring_wraps)
    pub wraps: u64,
}

/// The in-order packing: group g takes [vstart[g], vstart[g] + size[g]) of a virtual tape; a group that would cross the
/// ring end starts the next lap instead. Groups keep their order, and each column's tape inside its group keeps the
/// kernel's prefix order, so the walk-back order of every column is that of the old inline loop.
pub fn ring_plan(sizes: &[u64], ring: u64) -> Result<RingPlan, String> {
    if ring == 0 { return Err("DEFECT MYNN ring of 0 nodes".into()); }
    if let Some((g, s)) = sizes.iter().enumerate().find(|(_, &s)| s > ring) {
        return Err(format!("DEFECT MYNN ring of {ring} nodes is smaller than group {g} ({s} nodes)"));
    }
    let mut vstart = Vec::with_capacity(sizes.len() + 1);
    let mut phys = Vec::with_capacity(sizes.len());
    let mut v = 0u64;
    for &sz in sizes {
        let off = v % ring;
        if off + sz > ring { v += ring - off; }
        vstart.push(v);
        phys.push(v % ring);
        v += sz;
    }
    vstart.push(v);
    Ok(RingPlan { vstart, phys, wraps: v / ring })
}

/// The bytes the kernel reads as mw_ring: vstart (groups + 1), phys (groups), then a zeroed u32 done flag per group.
pub fn ring_bytes(p: &RingPlan) -> Vec<u8> {
    let groups = p.phys.len();
    let mut rb: Vec<u8> = Vec::with_capacity((2 * groups + 1) * 8 + groups * 4 + 8);
    for x in p.vstart.iter().chain(p.phys.iter()) { rb.extend_from_slice(&x.to_le_bytes()); }
    rb.resize(rb.len() + groups * 4, 0u8);
    rb
}

/// Groups the ring lets be in flight, (fewest, mean) over every `low` of the walk: from `low`, the groups g >= low with
/// vstart[g] + size[g] <= vstart[low] + ring. Slots are not counted here: in flight is the lesser of this and the slots.
pub fn ring_inflight(p: &RingPlan, sizes: &[u64], ring: u64) -> (usize, f64) {
    let n = sizes.len();
    if n == 0 { return (0, 0.0); }
    let (mut hi, mut least, mut sum) = (0usize, usize::MAX, 0usize);
    for low in 0..n {
        if hi < low { hi = low; }
        while hi < n && p.vstart[hi] + sizes[hi] <= p.vstart[low] + ring { hi += 1; }
        let k = hi - low;
        // the last groups of a launch are few because the launch ends, not because the ring is short
        if hi < n { least = least.min(k); }
        sum += k;
    }
    (if least == usize::MAX { n } else { least }, sum as f64 / n as f64)
}

/// The ring that lets `slots` consecutive groups be in flight wherever `low` is: the largest sum of `slots` consecutive
/// group sizes plus the largest group (a lap end wastes less than one group inside any span of at most one ring).
pub fn ring_for_slots(sizes: &[u64], slots: usize) -> u64 {
    let n = sizes.len();
    if n == 0 { return 0; }
    let s = slots.clamp(1, n);
    let mut win: u64 = sizes[..s].iter().sum();
    let mut best = win;
    for g in s..n { win = win + sizes[g] - sizes[g - s]; best = best.max(win); }
    best + sizes.iter().copied().max().unwrap_or(0)
}

/// The tape arena a tangent bind that ran out of room grows to. THE BREAKAGE: the old rule grew to at least twice the
/// arena (an exhausted take names only where it ends, not the bind's total), so a 60 km run started at 7 GiB went to
/// 14 GiB for a MYNN-EDMF bind that named 9,190,900,736 bytes (six-1 pbl-base window_tl_tape_grown), and those 4.7 GB
/// were the card the boundary layer's ring could not get (pbl-ta12b: a 12 GiB ring left 13.8 MB of room, launches of 287
/// and 454 s). Now the arena grows to the named need plus 1/8, and at least 1/8 over what it was, and the bind runs again
/// until it fits (the caller's loop), so it settles within 1/8 of the bind's real total. `double` keeps the old rule.
/// MiB-rounded; never above `max` unless the need is (as before).
pub fn tape_grow_want(need: usize, have: usize, max: usize, double: bool) -> usize {
    let floor = if double { 2 * have } else { have + have / 8 };
    (need + need / 8).max(floor).min(max.max(need)).div_ceil(1 << 20) << 20
}

#[cfg(test)]
mod tests {
    use super::*;

    // the inline loop schemes.rs ran before this module (5fe6e3741), kept word for word as the reference
    fn old_inline(pers_counted: &[i64], lanes: usize, cols_here: usize, pnu: u64) -> (Vec<u8>, u64) {
        let groups = cols_here.div_ceil(lanes);
        let mut vs: Vec<u64> = Vec::with_capacity(groups + 1);
        let mut ph: Vec<u64> = Vec::with_capacity(groups);
        let mut v = 0u64;
        for g in 0..groups {
            let sz: u64 = pers_counted[g * lanes..((g + 1) * lanes).min(cols_here)].iter().map(|&u| u.max(0) as u64 + 16).sum();
            let off = v % pnu;
            if off + sz > pnu { v += pnu - off; }
            vs.push(v); ph.push(v % pnu); v += sz;
        }
        vs.push(v);
        let mut rb: Vec<u8> = Vec::with_capacity((2 * groups + 1) * 8 + groups * 4 + 8);
        for x in vs.iter().chain(ph.iter()) { rb.extend_from_slice(&x.to_le_bytes()); }
        rb.resize(rb.len() + groups * 4, 0u8);
        (rb, v / pnu.max(1))
    }

    // 60 km-like counts: mean about 67,000 nodes, a tail to about 3 x, some columns with no tape (-1)
    fn counts(n: usize, seed: u64) -> Vec<i64> {
        let mut x = seed | 1;
        (0..n).map(|i| {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            if i % 97 == 5 { -1 } else { 20_000 + (x % 60_000) as i64 + if x % 23 == 0 { 140_000 } else { 0 } }
        }).collect()
    }

    #[test]
    fn plan_bytes_and_wraps_match_the_old_loop() {
        for (n, lanes) in [(29_216usize, 8usize), (17_762, 8), (2_562, 32), (13, 8), (1, 8)] {
            let c = counts(n, n as u64 * 7 + lanes as u64);
            let sizes = group_sizes(&c, lanes);
            let largest = *sizes.iter().max().unwrap();
            for ring in [largest, largest + 1, 178_956_800, 357_913_770, 536_870_741, sizes.iter().sum::<u64>() + largest] {
                let p = ring_plan(&sizes, ring).unwrap();
                let (rb, wraps) = old_inline(&c, lanes, n, ring);
                assert_eq!(ring_bytes(&p), rb, "n {n} lanes {lanes} ring {ring}");
                assert_eq!(p.wraps, wraps);
            }
        }
    }

    #[test]
    fn every_column_once_in_order_and_inside_the_ring() {
        let (n, lanes) = (29_216usize, 8usize);
        let c = counts(n, 11);
        let sizes = group_sizes(&c, lanes);
        for ring in [*sizes.iter().max().unwrap(), 178_956_800u64, 357_913_770] {
            let p = ring_plan(&sizes, ring).unwrap();
            assert_eq!(p.phys.len(), sizes.len());
            let mut seen = vec![0u8; n];
            let mut last_end = 0u64;
            for (g, &sz) in sizes.iter().enumerate() {
                // groups keep their order and never overlap in the virtual tape
                assert!(p.vstart[g] >= last_end);
                last_end = p.vstart[g] + sz;
                assert!(p.vstart[g + 1] >= last_end);
                // never across the ring end
                assert_eq!(p.phys[g], p.vstart[g] % ring);
                assert!(p.phys[g] + sz <= ring);
                // each column of the group, in its prefix order (the kernel's tape start: gbase + exclusive prefix)
                let lo = g * lanes;
                let mut pre = 0u64;
                for col in lo..(lo + lanes).min(n) {
                    seen[col] += 1;
                    let need = c[col].max(0) as u64 + 16;
                    assert!(p.phys[g] + pre + need <= ring);
                    pre += need;
                }
                assert_eq!(pre, sz);
            }
            assert!(seen.iter().all(|&s| s == 1));
        }
    }

    #[test]
    fn ring_for_slots_keeps_that_many_in_flight() {
        let lanes = 8;
        for (n, seed) in [(29_216usize, 3u64), (17_762, 5), (800, 9)] {
            let sizes = group_sizes(&counts(n, seed), lanes);
            for slots in [1usize, 7, 64, 333, 761, 1504] {
                let ring = ring_for_slots(&sizes, slots);
                let p = ring_plan(&sizes, ring).unwrap();
                let (least, _) = ring_inflight(&p, &sizes, ring);
                assert!(least >= slots.min(sizes.len()), "n {n} slots {slots}: ring {ring} keeps only {least}");
            }
        }
    }

    #[test]
    fn inflight_counts_by_brute_force() {
        let sizes = group_sizes(&counts(3_000, 21), 8);
        for ring in [*sizes.iter().max().unwrap(), 2_000_000u64, 9_000_000] {
            let p = ring_plan(&sizes, ring).unwrap();
            let n = sizes.len();
            let (mut least, mut sum) = (usize::MAX, 0usize);
            for low in 0..n {
                let k = (low..n).take_while(|&g| p.vstart[g] + sizes[g] <= p.vstart[low] + ring).count();
                if low + k < n { least = least.min(k); }
                sum += k;
            }
            let (l, m) = ring_inflight(&p, &sizes, ring);
            assert_eq!(l, if least == usize::MAX { n } else { least });
            assert!((m - sum as f64 / n as f64).abs() < 1e-9);
        }
    }

    #[test]
    fn refuses_a_ring_smaller_than_a_group() {
        assert!(ring_plan(&[10, 30, 5], 29).is_err());
        assert!(ring_plan(&[10], 0).is_err());
        assert!(ring_plan(&[], 1).unwrap().vstart == vec![0]);
    }

    #[test]
    fn tape_growth_settles_near_the_need() {
        let gib = 1usize << 30;
        // six-1 pbl-base: 7 GiB arena, a MYNN-EDMF bind that named 9,190,900,736 bytes, --tape-max-gib 14
        assert_eq!(tape_grow_want(9_190_900_736, 7 * gib, 14 * gib, true), 14 * gib);
        let w = tape_grow_want(9_190_900_736, 7 * gib, 14 * gib, false);
        assert_eq!(w, 10_340_007_936);
        assert!(w >= 9_190_900_736 + 9_190_900_736 / 8 && w < 9_190_900_736 + 9_190_900_736 / 8 + (1 << 20));
        // a bind whose failing take names an end just past the arena each time still settles within 1/8 of its total
        for total in [9_900_000_000usize, 12_000_000_000, 14 * gib] {
            let (mut have, mut tries, mut named) = (7 * gib, 0, 9_190_900_736usize);
            loop {
                have = tape_grow_want(named, have, 14 * gib, false);
                tries += 1;
                if have >= total { break; }
                named = have + 1;
                assert!(tries < 12, "total {total}: {tries} re-binds");
            }
            assert!(have <= (total + total / 8 + (1 << 20)).max(14 * gib), "total {total}: arena {have}");
        }
        // never above the cap unless the need is
        assert_eq!(tape_grow_want(3 * gib, 2 * gib, 3 * gib, false), 3 * gib);
        assert_eq!(tape_grow_want(5 * gib, 2 * gib, 3 * gib, false), 5 * gib);
    }
}
