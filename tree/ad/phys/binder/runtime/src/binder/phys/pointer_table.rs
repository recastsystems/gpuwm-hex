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
}
