// SPDX-License-Identifier: Apache-2.0
//! Radiation KISS xorshift jump table from live draw and g-point counts.
//! This host rule does not establish the provenance of either input.
//!
//! Source: hexcore/mpasphys/rrtmg_lw_v842.py:1039-1095. Both shortwave
//! and longwave call the same implementation. Seed values themselves are
//! carried state and must never be replaced by this table.

fn apply(columns: &[u32; 32], mut value: u32) -> u32 {
    let mut output = 0;
    let mut index = 0;
    while value != 0 {
        if value & 1 != 0 { output ^= columns[index]; }
        value >>= 1;
        index += 1;
    }
    output
}

fn compose(outer: &[u32; 32], inner: &[u32; 32]) -> [u32; 32] {
    std::array::from_fn(|i| apply(outer, inner[i]))
}

pub fn draws_per_gpoint(overlap: i32, layers: i32) -> Result<u64, String> {
    if layers < 0 { return Err("negative radiation layer count".into()); }
    if overlap == 3 { return Ok(1); }
    let multiplier = if overlap >= 4 { 2u64 } else { 1u64 };
    Ok((layers as u64) * multiplier)
}

pub fn jump_table(draws: u64, gpoints: usize) -> Result<Vec<u32>, String> {
    let words = gpoints.checked_mul(32).ok_or("xorshift table size overflow")?;
    let mut output = Vec::new();
    output.try_reserve_exact(words).map_err(|_| "xorshift table allocation failed")?;
    let identity: [u32; 32] = std::array::from_fn(|i| 1u32 << i);
    let mut power = identity;
    let mut base = std::array::from_fn(|i| {
        let mut value = 1u32 << i;
        value ^= value << 13;
        value ^= value >> 17;
        value ^= value << 5;
        value
    });
    let mut count = draws;
    while count != 0 {
        if count & 1 != 0 { power = compose(&base, &power); }
        base = compose(&base, &base);
        count >>= 1;
    }
    let mut current = identity;
    for _ in 0..gpoints {
        output.extend_from_slice(&current);
        current = compose(&power, &current);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn small_table_matches_direct_repeated_steps() {
        let table = jump_table(3, 8).unwrap();
        for bit in 0..32 {
            let mut value = 1u32 << bit;
            for row in 0..8 {
                assert_eq!(table[row * 32 + bit], value);
                for _ in 0..3 { value ^= value << 13; value ^= value >> 17; value ^= value << 5; }
            }
        }
    }
    #[test]
    fn live_counts_and_overflow_are_checked() {
        assert_eq!(draws_per_gpoint(2, 60).unwrap(), 60);
        assert_eq!(draws_per_gpoint(3, 60).unwrap(), 1);
        assert_eq!(draws_per_gpoint(4, 60).unwrap(), 120);
        assert!(draws_per_gpoint(1, -1).is_err());
        assert!(jump_table(1, usize::MAX).is_err());
        assert!(jump_table(0, 0).unwrap().is_empty());
    }
}
