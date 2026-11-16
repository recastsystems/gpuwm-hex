// SPDX-License-Identifier: Apache-2.0
//! 60 km tape (AD_P3_STRIPE=1, off by default): the boundary layer's (MYNN-EDMF) two transient regions sized to
//! the launch's columns, not to the whole mesh. No arithmetic on weather fields: only where the bytes live.
//!
//! THE BREAKAGE THIS PREVENTS: a deferred MYNN-EDMF bind takes a float64 shadow region and a seed snapshot region of
//! one float64 per float32 word of every active argument, and every argument's extent is the whole mesh (ims:ime) while
//! a launch holds a batch of its columns. At 60 km (163,842 columns, 26 surface and 58 sixty-level arrays: 3,506 rows)
//! each region is 4.60 GB, 9.19 GB a launch, for launches of 29,216 or 17,762 columns: the reverse scratch peak of the
//! fullest card (s60-cardmem: 9,190,900,736 bytes). About 7.3 GB of it is never read or written.
//!
//! THE LAYOUT. Every array is rows x ni float32 words with the column index fastest, and the launch's entries read and
//! write only the elements [i0, i0 + cols) of every row (columns are independent, G5; the MYNN helper-local stripes in
//! schemes.rs copy in, run on and copy out only those elements and keep the same words). The kernel finds element
//! (row k, column i) of an array at `base + (k * ni + i) * 8`, with ni and i0 from the launch's own scalars, which do not
//! change. So the arrays keep their kernel addresses' arithmetic and only their bases move: arrays are put in bands of
//! `q = floor(ni * 8 / wp)` arrays (wp = cols * 8 rounded up to 256 bytes), array t of a band at
//! `base = band + t * wp - i0 * 8`. Row k of its stripe is then `band + k * ni * 8 + t * wp .. + cols * 8`, inside
//! row k's pitch and apart from the other arrays of the band. The words outside the stripes are virtual: nothing
//! reads or writes them, so the window moves only the stripes (`stripes`) between its float32 shadow and the region.
//! Same words by construction: the same entries on the same element values at different addresses. Shown on a card.

/// Bytes one shadow word takes (float64), as schemes.rs's W.
pub const WB: usize = 8;

/// The banded layout of one launch's active segments.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    /// Byte offset in the region of row 0 of segment j's stripe (j in the order the rows were given).
    pub place: Vec<usize>,
    /// Region bytes (one region; the snapshot region has the same layout).
    pub total: usize,
    /// What the whole-mesh layout takes for the same segments (one region).
    pub full: usize,
    /// Arrays a band holds.
    pub per_band: usize,
}

fn up256(b: usize) -> usize { b.div_ceil(256) * 256 }

/// The banded layout for segments of `rows[j]` rows of `ni` float32 words, of which a launch uses [i0, i0 + cols).
/// None when the geometry does not fit or the bands would save nothing (then the whole-mesh layout stays).
pub fn plan(ni: usize, i0: usize, cols: usize, rows: &[usize]) -> Option<Plan> {
    if ni == 0 || cols == 0 || i0.checked_add(cols)? > ni || rows.is_empty() || rows.iter().any(|&r| r == 0) { return None; }
    let pitch = ni * WB;
    let wp = up256(cols * WB);
    let q = pitch / wp;
    if q < 2 { return None; }
    let full: usize = rows.iter().map(|&r| up256(r * pitch)).sum();
    // tallest first, so a band's height (its first array's rows) wastes the least
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by(|&a, &b| rows[b].cmp(&rows[a]).then(a.cmp(&b)));
    let mut place = vec![0usize; rows.len()];
    let mut at = 0usize;
    for band in order.chunks(q) {
        let h = rows[band[0]];
        for (t, &j) in band.iter().enumerate() { place[j] = at + t * wp; }
        // the band's last row needs only its arrays' stripes, not the whole pitch
        at += up256((h - 1) * pitch + band.len() * wp);
    }
    let total = at.max(256);
    if total >= full { return None; }
    Some(Plan { place, total, full, per_band: q })
}

/// The kernel base (element 0 of row 0) of an array whose stripe row 0 sits at `region + place`.
pub fn vbase(region: u64, place: usize, i0: usize) -> u64 { region.wrapping_add(place as u64).wrapping_sub((i0 * WB) as u64) }

/// The stripes of one active segment [lo, hi) of its allocation whose kernel base is `base`: (lo, hi, shadow address)
/// of each row's [i0, i0 + cols) words, in row order. The window moves these between its shadow and the region.
pub fn stripes(lo: usize, hi: usize, base: u64, ni: usize, i0: usize, cols: usize) -> Vec<(usize, usize, u64)> {
    let rows = (hi - lo) / (ni * 4);
    (0..rows).map(|k| {
        let e = k * ni + i0;
        (lo + e * 4, lo + (e + cols) * 4, base.wrapping_add((e * WB) as u64))
    }).collect()
}

/// The window's moves for a list of segments (allocation, lo, hi, shadow address), as (first, end, pitch bytes): segments
/// [first, end) are rows of one allocation, of one width, `pitch` bytes apart in the allocation and 2 x pitch in the
/// shadow (one float64 a float32 word), with pitch >= width, so one pitched launch moves them; a run of one has pitch 0.
pub fn pitched_runs(segs: &[(u64, usize, usize, u64)]) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < segs.len() {
        let (al, lo, hi, d) = segs[i];
        let w = hi.saturating_sub(lo);
        let mut j = i + 1;
        let mut pitch = 0usize;
        if w > 0 && w % 4 == 0 && j < segs.len() && segs[j].0 == al && segs[j].2.saturating_sub(segs[j].1) == w && segs[j].1 > lo {
            let p = segs[j].1 - lo;
            if p % 4 == 0 && p >= w && segs[j].3 == d.wrapping_add(2 * p as u64) {
                pitch = p;
                while j < segs.len() && segs[j].0 == al && segs[j].2.saturating_sub(segs[j].1) == w
                      && segs[j].1 == segs[j - 1].1 + p && segs[j].3 == segs[j - 1].3.wrapping_add(2 * p as u64) { j += 1; }
            }
        }
        out.push((i, j, pitch));
        i = j;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 60 km, from the MYNN-EDMF row (schemes.json rrfs_mynnedmf_run_l16: 26 active (ime-ims+1)*(jme-jms+1) arrays,
    /// 58 active (ime-ims+1)*(kme-kms+1)*(jme-jms+1) arrays) and the mesh (163,842 columns, kme-kms+1 = 60: 3,506 rows,
    /// 4,596 MB a region, the 4.60 GB of the window.rs note). Launches: five of 29,216 columns and one of 17,762.
    const NI60: usize = 163_842;
    fn rows60(nk: usize) -> Vec<usize> {
        // a mixed order (the row's arguments interleave them), so the tallest-first sort is exercised
        let mut v = Vec::new();
        let mut ones = 0;
        for j in 0..84 { if ones < 26 && j % 3 == 1 { v.push(1); ones += 1; } else { v.push(nk); } }
        assert_eq!(v.iter().filter(|&&r| r == 1).count(), 26);
        assert_eq!(v.iter().filter(|&&r| r == nk).count(), 58);
        v
    }
    fn launches60() -> Vec<(usize, usize)> {
        let mut l: Vec<(usize, usize)> = (0..5).map(|b| (b * 29_216, 29_216)).collect();
        l.push((5 * 29_216, 17_762));
        assert_eq!(l.iter().map(|x| x.1).sum::<usize>(), NI60);
        l
    }

    /// Every byte the kernel can touch, for every launch: inside the region, and no two arrays share a byte.
    fn check_disjoint(ni: usize, i0: usize, cols: usize, rows: &[usize], p: &Plan) {
        let mut iv: Vec<(usize, usize)> = Vec::new();
        for (j, &r) in rows.iter().enumerate() {
            let base = vbase(1 << 40, p.place[j], i0);
            for k in 0..r {
                let a = (base + ((k * ni + i0) * WB) as u64 - (1u64 << 40)) as usize;
                iv.push((a, a + cols * WB));
            }
        }
        iv.sort();
        for w in iv.windows(2) { assert!(w[0].1 <= w[1].0, "stripes overlap: {:?} {:?}", w[0], w[1]); }
        assert!(iv.last().unwrap().1 <= p.total, "a stripe ends at {} past the region's {}", iv.last().unwrap().1, p.total);
    }

    #[test]
    fn sixty_km_layout_fits_and_frees_the_regions() {
        let rows = rows60(60);
        let mut worst = 0usize;
        for (i0, cols) in launches60() {
            let p = plan(NI60, i0, cols, &rows).expect("a 60 km launch gets bands");
            check_disjoint(NI60, i0, cols, &rows, &p);
            assert!(p.full >= 3_506 * NI60 * WB && p.full < 3_506 * NI60 * WB + 84 * 256, "whole-mesh region {}", p.full);
            worst = worst.max(p.total);
            eprintln!("60 km launch i0 {i0} cols {cols}: {} arrays a band, region {} bytes against {} ({:.1} %)",
                      p.per_band, p.total, p.full, 100.0 * p.total as f64 / p.full as f64);
        }
        let full = plan(NI60, 0, 29_216, &rows).unwrap().full;
        eprintln!("two regions, largest launch: {} -> {} bytes, {} freed", 2 * full, 2 * worst, 2 * (full - worst));
        assert!(2 * (full - worst) > 7_000_000_000, "should free about 7.3 GB");
    }

    #[test]
    fn whole_mesh_launch_keeps_the_old_layout() {
        assert_eq!(plan(1000, 0, 1000, &[3, 1]), None);
        assert_eq!(plan(1000, 10, 995, &[3, 1]), None);
        assert_eq!(plan(1000, 0, 600, &[3]), None);
    }

    // ---------------------------------------------------------------------------------------------------------------
    // The data path, both ways, on 60 km columns: the window's float32 shadow, the region, the snapshot region, the
    // entries (an emulation that reads only its own columns and mixes rows and arrays of a column, as MYNN-EDMF's do),
    // and back. The whole-mesh path is the head's (one segment = the array, the region at its old layout); the
    // banded path uses plan, vbase and stripes exactly as schemes.rs does. The window's words after all six launches
    // must be the same bytes. Ten levels instead of sixty keep it at 0.3 GB of host memory; the layout test above
    // covers sixty.

    struct Dev { base: u64, mem: Vec<f64> }
    impl Dev {
        fn ix(&self, a: u64) -> usize {
            assert!(a >= self.base && (a - self.base) as usize % WB == 0, "address {a:#x} below the region or mid-word");
            let i = ((a - self.base) as usize) / WB;
            assert!(i < self.mem.len(), "address {a:#x} past the region (a fault on a card)");
            i
        }
    }

    /// The emulated entries: for each column of the launch, every element of every array becomes a fixed function of
    /// that column's snapshot (all arrays, neighbouring rows), written to the shadow. `bump` plants an error on one word.
    fn entries(sh: &mut Dev, sn: &Dev, bases: &[(u64, u64)], rows: &[usize], ni: usize, i0: usize, cols: usize, bump: Option<(usize, usize, usize)>) {
        for c in i0..i0 + cols {
            for (j, &r) in rows.iter().enumerate() {
                for k in 0..r {
                    let me = sn.mem[sn.ix(bases[j].1 + ((k * ni + c) * WB) as u64)];
                    let jn = (j + 1) % rows.len();
                    let kn = (k + 1).min(rows[jn] - 1);
                    let nb = sn.mem[sn.ix(bases[jn].1 + ((kn * ni + c) * WB) as u64)];
                    let mut v = 0.75 * me + 0.125 * nb + 1e-3 * (c as f64).sin();
                    if bump == Some((j, k, c)) { v *= 1.001; }
                    let a = sh.ix(bases[j].0 + ((k * ni + c) * WB) as u64);
                    sh.mem[a] = v;
                }
            }
        }
    }

    fn run(banded: bool, ni: usize, rows: &[usize], launches: &[(usize, usize)], bump: Option<(usize, usize, usize, usize)>) -> Vec<Vec<f32>> {
        // the window's float32 shadow, one allocation an array (real alloc boundaries do not matter to the layout)
        let mut win: Vec<Vec<f32>> = rows.iter().enumerate().map(|(j, &r)| (0..r * ni).map(|w| (((w * 2_654_435_761 + j * 97) % 10_007) as f32) * 1e-4 - 0.5).collect()).collect();
        for (li, &(i0, cols)) in launches.iter().enumerate() {
            let (region, snapr) = (0x7f00_0000_0000u64, 0x7e00_0000_0000u64);
            let (bytes, bases, segs): (usize, Vec<(u64, u64)>, Vec<Vec<(usize, usize, u64)>>) = if banded {
                let p = plan(ni, i0, cols, rows).expect("bands");
                let bases: Vec<(u64, u64)> = (0..rows.len()).map(|j| (vbase(region, p.place[j], i0), vbase(snapr, p.place[j], i0))).collect();
                let segs = rows.iter().enumerate().map(|(j, &r)| stripes(0, r * ni * 4, bases[j].0, ni, i0, cols)).collect();
                (p.total, bases, segs)
            } else {
                let mut at = 0usize;
                let mut bases = Vec::new();
                let mut segs = Vec::new();
                for &r in rows { bases.push((region + at as u64, snapr + at as u64)); segs.push(vec![(0, r * ni * 4, region + at as u64)]); at += up256(r * ni * WB); }
                (at, bases, segs)
            };
            let mut sh = Dev { base: region, mem: vec![0.0; bytes / WB] }; // zeroed at bind
            // the window's segment list (allocation = array index), as active_segments gives it
            let flat: Vec<(u64, usize, usize, u64)> = segs.iter().enumerate().flat_map(|(j, s)| s.iter().map(move |&(lo, hi, d)| (j as u64, lo, hi, d))).collect();
            // widen (window xfer): the banded path through pitched_runs and the pitched kernel, the head one segment a launch
            xfer(&mut win, &mut sh, &flat, true, banded);
            // the seed snapshot (p3: per segment, or the whole region when banded)
            let sn = Dev { base: snapr, mem: sh.mem.clone() };
            let b = bump.filter(|x| x.0 == li).map(|x| (x.1, x.2, x.3));
            entries(&mut sh, &sn, &bases, rows, ni, i0, cols, b);
            // narrow (window xfer back)
            xfer(&mut win, &mut sh, &flat, false, banded);
        }
        win
    }

    /// The window's xfer: win_widen/narrow one segment a launch, or (pitched) win_widen2/narrow2 over pitched_runs, as
    /// the kernels in window.cu move elements: run row k, word c is source word lo/4 + k * pw + c, region word
    /// (d - base)/8 + k * pw + c.
    fn xfer(win: &mut [Vec<f32>], sh: &mut Dev, flat: &[(u64, usize, usize, u64)], widen: bool, pitched: bool) {
        let mut mv = |al: usize, w: usize, a: u64| { let i = sh.ix(a); if widen { sh.mem[i] = win[al][w] as f64; } else { win[al][w] = sh.mem[i] as f32; } };
        let runs: Vec<(usize, usize, usize)> = if pitched { pitched_runs(flat) } else { (0..flat.len()).map(|i| (i, i + 1, 0)).collect() };
        let mut moved = 0usize;
        for (i, j, pitch) in runs {
            let (al, lo, hi, d) = flat[i];
            let n = (hi - lo) / 4;
            let (rows, pw) = if j - i >= 2 { (j - i, pitch / 4) } else { (1, 0) };
            for k in 0..rows { for c in 0..n { mv(al as usize, lo / 4 + k * pw + c, d + ((k * pw + c) * WB) as u64); } }
            moved += j - i;
        }
        assert_eq!(moved, flat.len());
    }

    #[test]
    fn pitched_runs_group_a_banded_arrays_rows_only() {
        let rows = rows60(10);
        let p = plan(NI60, 29_216, 29_216, &rows).unwrap();
        let mut flat = Vec::new();
        for (j, &r) in rows.iter().enumerate() {
            let base = vbase(1 << 40, p.place[j], 29_216);
            for (lo, hi, d) in stripes(0, r * NI60 * 4, base, NI60, 29_216, 29_216) { flat.push((j as u64, lo, hi, d)); }
        }
        let runs = pitched_runs(&flat);
        assert_eq!(runs.len(), rows.len(), "one launch an array");
        for (i, j, pitch) in &runs { assert_eq!(j - i, rows[flat[*i].0 as usize]); if j - i > 1 { assert_eq!(*pitch, NI60 * 4); } }
        // the head's segments (one an array, regions 256-rounded) never coalesce into a wrong run
        let mut at = 0u64;
        let head: Vec<(u64, usize, usize, u64)> = rows.iter().enumerate().map(|(j, &r)| { let s = (j as u64, 0, r * NI60 * 4, at); at += up256(r * NI60 * WB) as u64; s }).collect();
        assert!(pitched_runs(&head).iter().all(|r| r.1 - r.0 == 1));
        eprintln!("60 km, ten levels: {} stripes in {} pitched launches", flat.len(), runs.len());
    }

    fn same_bytes(a: &[Vec<f32>], b: &[Vec<f32>]) -> usize {
        a.iter().zip(b).map(|(x, y)| x.iter().zip(y).filter(|(p, q)| p.to_bits() != q.to_bits()).count()).sum()
    }

    #[test]
    fn sixty_km_columns_same_bytes_and_a_planted_error_fails() {
        let rows = rows60(10);
        let l = launches60();
        let head = run(false, NI60, &rows, &l, None);
        let banded = run(true, NI60, &rows, &l, None);
        let diff = same_bytes(&head, &banded);
        let words: usize = head.iter().map(|v| v.len()).sum();
        eprintln!("60 km columns, 6 launches, {words} window words: {diff} differ (banded against the head's layout)");
        assert_eq!(diff, 0);
        // planted: x1.001 on one word of launch 3 (array 41, ten rows; row 2, the launch's first column)
        let planted = run(true, NI60, &rows, &l, Some((3, 41, 2, 3 * 29_216)));
        let d2 = same_bytes(&head, &planted);
        eprintln!("planted x1.001: {d2} words differ");
        assert!(d2 > 0, "the planted x1.001 error must fail the comparison");
    }
}
