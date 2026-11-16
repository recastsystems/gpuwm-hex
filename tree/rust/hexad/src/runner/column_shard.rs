// SPDX-License-Identifier: Apache-2.0
//! Checked, contiguous column ownership. No arithmetic on weather fields.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Range { pub first: usize, pub count: usize }

pub fn partition(columns: usize, cards: usize) -> Result<Vec<Range>, String> {
    if cards == 0 || columns < cards {
        return Err("DEFECT column sharding needs at least one column per card".into());
    }
    let mut first = 0;
    Ok((0..cards).map(|i| {
        let count = columns / cards + usize::from(i < columns % cards);
        let r = Range { first, count };
        first += count;
        r
    }).collect())
}

/// mynn: as `partition`, with the first card (the dycore card) taking `primary` (0.0 to below 1.0) of an equal share,
/// never fewer than `floor` columns (one launch group), and the other cards sharing the rest evenly. THE BREAKAGE: at
/// x1.163842 the dycore card is full, so its quarter of every MYNN-EDMF launch ran on scratch the cold tier keeps
/// moving on and off the card, on a 4 GiB lent tape ring, while the other three cards had 50 GB free each.
pub fn partition_primary(columns: usize, cards: usize, primary: f64, floor: usize) -> Result<Vec<Range>, String> {
    if cards < 2 || !(0.0..1.0).contains(&primary) { return partition(columns, cards); }
    if columns < cards { return Err("DEFECT column sharding needs at least one column per card".into()); }
    let equal = columns / cards;
    let first = (((equal as f64) * primary).round() as usize).max(floor).min(equal).max(1);
    let mut out = vec![Range { first: 0, count: first }];
    out.extend(partition(columns - first, cards - 1)?.into_iter().map(|r| Range { first: r.first + first, count: r.count }));
    Ok(out)
}

pub fn devices(text: &str) -> Result<Vec<i32>, String> {
    devices_with(text, std::env::var("AD_SHARD_SAME_CARD_CONTEXTS").as_deref() == Ok("1"))
}

/// `repeat` (AD_SHARD_SAME_CARD_CONTEXTS=1, correctness proofs only): an ordinal may
/// repeat, giving each shard its own CUDA context on that card. Columns, copies and
/// kernels follow exactly the multi-card path; only the hardware is shared.
pub fn devices_with(text: &str, repeat: bool) -> Result<Vec<i32>, String> {
    let mut out = Vec::new();
    for field in text.split(',') {
        let d = field.trim().parse::<i32>().map_err(|_| "DEFECT shard devices must be comma-separated visible ordinals")?;
        if d < 0 || (!repeat && out.contains(&d)) { return Err("DEFECT shard devices must be distinct nonnegative ordinals (AD_SHARD_SAME_CARD_CONTEXTS=1 allows repeats for proofs)".into()); }
        out.push(d);
    }
    if out.len() < 2 { return Err("DEFECT sharding requires at least two cards".into()); }
    Ok(out)
}

/// (source, destination, bytes) for an exact rectangle, packed on the shard.
/// `back` reverses the copy without changing column ownership or touching padding.
pub fn rows(full: u64, packed: u64, levels: usize, columns: usize, range: Range,
            width: usize, back: bool) -> Result<Vec<(u64, u64, usize)>, String> {
    if ![4, 8].contains(&width) || range.count == 0 || range.first > columns
        || range.count > columns - range.first {
        return Err("DEFECT invalid column rectangle".into());
    }
    let pitch = columns.checked_mul(width).ok_or("DEFECT column pitch overflow")?;
    let bytes = range.count.checked_mul(width).ok_or("DEFECT shard pitch overflow")?;
    let offset = range.first.checked_mul(width).ok_or("DEFECT column offset overflow")?;
    let address = |base: u64, row: usize, stride: usize, off: usize| {
        row.checked_mul(stride).and_then(|x| x.checked_add(off))
            .and_then(|x| base.checked_add(x as u64)).ok_or("DEFECT column address overflow")
    };
    // Check exclusive endpoints as well as starts, before returning any transfers.
    if levels > 0 {
        address(full, levels - 1, pitch, offset)?.checked_add(bytes as u64).ok_or("DEFECT full extent overflow")?;
        address(packed, levels - 1, bytes, 0)?.checked_add(bytes as u64).ok_or("DEFECT packed extent overflow")?;
    }
    (0..levels).map(|k| {
        let a = address(full, k, pitch, offset)?;
        let b = address(packed, k, bytes, 0)?;
        Ok(if back { (b, a, bytes) } else { (a, b, bytes) })
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uneven_decks_cover_each_column_once() {
        for n in [2562, 40962] {
            for cards in [1, 2, 3, 4] {
                let p = partition(n, cards).unwrap();
                let mut owners = vec![0; n];
                for r in &p { for c in r.first..r.first + r.count { owners[c] += 1; } }
                assert!(owners.iter().all(|&x| x == 1));
                assert!(p.iter().map(|r| r.count).max().unwrap() - p.iter().map(|r| r.count).min().unwrap() <= 1);
            }
        }
    }
    #[test]
    fn primary_share_covers_each_column_once() {
        for n in [2562, 29216, 17762, 163842] {
            for cards in [2, 3, 4, 8] {
                for share in [0.0, 0.05, 0.5, 0.999] {
                    let p = partition_primary(n, cards, share, 8).unwrap();
                    assert_eq!(p.len(), cards);
                    let mut next = 0;
                    for r in &p { assert_eq!(r.first, next); assert!(r.count >= 1); next += r.count; }
                    assert_eq!(next, n);
                    assert!(p[0].count <= n / cards && p[0].count >= 8.min(n / cards));
                }
                assert_eq!(partition_primary(n, cards, 1.0, 8).unwrap(), partition(n, cards).unwrap());
            }
        }
    }
    #[test]
    fn roundtrip_preserves_words_and_guards() {
        // Odd count, both element widths, missing carriers handled by the caller.
        // Include signed zero, infinities and NaN payloads: this is a byte copy.
        for width in [4, 8] {
            let (nc, nz) = (11, 3);
            let mut source = vec![0u8; nc * nz * width];
            for (i, v) in source.iter_mut().enumerate() { *v = (i * 197 + 131) as u8; }
            source[..8].copy_from_slice(&0x7ff8_1234_8000_0000u64.to_le_bytes());
            let mut output = vec![0xa5u8; source.len() + 32];
            for r in partition(nc, 4).unwrap() {
                let mut packed = vec![0x5au8; nz * r.count * width + 16];
                for (s, d, n) in rows(0, 0, nz, nc, r, width, false).unwrap() {
                    packed[d as usize..d as usize + n].copy_from_slice(&source[s as usize..s as usize + n]);
                }
                assert!(packed[nz * r.count * width..].iter().all(|&x| x == 0x5a));
                for (s, d, n) in rows(16, 0, nz, nc, r, width, true).unwrap() {
                    output[d as usize..d as usize + n].copy_from_slice(&packed[s as usize..s as usize + n]);
                }
            }
            assert_eq!(&output[16..16 + source.len()], &source);
            assert!(output[..16].iter().chain(output[16 + source.len()..].iter()).all(|&x| x == 0xa5));
        }
    }
    #[test]
    fn rejects_invalid_ownership_and_overflow() {
        for text in ["", "0", "0,0", "0,-1", "0,x", "0,"] { assert!(devices(text).is_err()); }
        assert_eq!(devices("0, 1,2,3").unwrap(), vec![0,1,2,3]);
        assert_eq!(devices_with("0,0,0", true).unwrap(), vec![0,0,0]);
        assert!(devices_with("0,0", false).is_err());
        assert!(partition(3, 4).is_err());
        assert!(partition(3, 0).is_err());
        assert!(rows(0, 0, 3, 11, Range { first: 10, count: 2 }, 4, false).is_err());
        assert!(rows(u64::MAX - 4, 0, 1, 2, Range { first: 0, count: 2 }, 4, false).is_err());
        assert!(rows(0, 0, usize::MAX, 2, Range { first: 0, count: 2 }, 8, false).is_err());
    }
}
