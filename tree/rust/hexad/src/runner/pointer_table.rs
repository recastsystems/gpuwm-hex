// SPDX-License-Identifier: Apache-2.0
//! Build an ordered table from current allocation bindings.
//! The caller must prove the source order and that this table is required.
use super::{plan::Slice, submit::Allocation};
use std::collections::HashMap;

pub fn build(entries: &[Slice], allocations: &HashMap<u64, Allocation>) -> Result<Vec<u8>, String> {
    if entries.is_empty() { return Err("empty live pointer table".into()); }
    let bytes = entries.len().checked_mul(8).ok_or("pointer table size overflow")?;
    let mut table = Vec::with_capacity(bytes);
    for target in entries {
        let allocation = allocations.get(&target.allocation)
            .ok_or_else(|| format!("unknown pointer target {}", target.allocation))?;
        if target.offset >= allocation.bytes { return Err("pointer target outside allocation".into()); }
        let offset = u64::try_from(target.offset).map_err(|_| "pointer target offset overflow")?;
        let address = allocation.address.checked_add(offset).ok_or("device address overflow")?;
        table.extend_from_slice(&address.to_le_bytes());
    }
    Ok(table)
}

/// S01: rows of (live address of `pointer`, then the literal words), each row `row_words` 8-byte words long.
pub fn build_descriptors(rows: &[(Slice, &[u64])], row_words: usize,
                         allocations: &HashMap<u64, Allocation>) -> Result<Vec<u8>, String> {
    if rows.is_empty() { return Err("empty descriptor table".into()); }
    let mut table = Vec::with_capacity(rows.len() * row_words * 8);
    for (pointer, words) in rows {
        if words.len() + 1 != row_words { return Err(format!("descriptor row of {} words, not {row_words}", words.len() + 1)); }
        let address = build(std::slice::from_ref(pointer), allocations)?;
        table.extend_from_slice(&address);
        for w in words.iter() { table.extend_from_slice(&w.to_le_bytes()); }
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn addresses_use_live_allocations_and_reject_stale_or_oob_bindings() {
        let alloc = HashMap::from([(1, Allocation { id: 1, address: 0x1234, bytes: 16 })]);
        let entries = [Slice { allocation: 1, offset: 8 }, Slice { allocation: 1, offset: 0 }];
        let output = build(&entries, &alloc).unwrap();
        assert_eq!(&output[0..8], &0x123cu64.to_le_bytes());
        assert_eq!(&output[8..16], &0x1234u64.to_le_bytes());
        assert!(build(&[Slice { allocation: 2, offset: 0 }], &alloc).is_err());
        assert!(build(&[Slice { allocation: 1, offset: 16 }], &alloc).is_err());
    }
    #[test]
    fn descriptor_rows_take_the_live_address_then_the_literal_words() {
        let alloc = HashMap::from([(1, Allocation { id: 1, address: 0x1000, bytes: 64 })]);
        let w = [3u64, 2788, 2787, 0x3f80_0000];
        let rows = [(Slice { allocation: 1, offset: 8 }, &w[..])];
        let out = build_descriptors(&rows, 5, &alloc).unwrap();
        assert_eq!(out.len(), 40);
        assert_eq!(&out[0..8], &0x1008u64.to_le_bytes());
        assert_eq!(&out[8..16], &3u64.to_le_bytes());
        assert_eq!(&out[32..40], &0x3f80_0000u64.to_le_bytes());
        assert!(build_descriptors(&rows, 4, &alloc).is_err());
        let w2 = [1u64, 1, 0, 0];
        let stale = [(Slice { allocation: 2, offset: 0 }, &w2[..])];
        assert!(build_descriptors(&stale, 5, &alloc).is_err());
    }
}
