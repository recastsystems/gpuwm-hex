// SPDX-License-Identifier: Apache-2.0
//! Fenced, disjoint physical owners supplied to a checkpoint consumer.
//!
//! The caller keeps the allocations and frame snapshot alive for the entire
//! checkpoint operation. Addresses are process-local and are never persisted.
use std::collections::HashSet;

pub const DEVICE: u32 = 1;
pub const HOST: u32 = 2;
pub const DRY_FRAME_OWNER: u64 = 2_000_001;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Binding {
    pub id: u64,
    pub bytes: u64,
    pub memory: u32,
    pub address: u64,
}

pub fn validate(bindings: &mut [Binding]) -> Result<(), &'static str> {
    if bindings.is_empty() { return Err("empty checkpoint binding"); }
    let mut ids = HashSet::new();
    for binding in bindings.iter() {
        if binding.id == 0 || !ids.insert(binding.id) || binding.bytes == 0 || binding.address == 0
            || binding.address.checked_add(binding.bytes).is_none()
            || !matches!(binding.memory, HOST | DEVICE) {
            return Err("invalid checkpoint binding");
        }
    }
    for memory in [HOST, DEVICE] {
        let mut ranges: Vec<_> = bindings.iter().filter(|b| b.memory == memory)
            .map(|b| (b.address, b.address + b.bytes)).collect();
        ranges.sort_unstable();
        if ranges.windows(2).any(|x| x[0].1 > x[1].0) {
            return Err("overlapping checkpoint owners");
        }
    }
    bindings.sort_unstable_by_key(|binding| binding.id);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fenced_owners_are_unique_disjoint_and_sorted() {
        let mut owners = [Binding { id: 2, bytes: 8, memory: HOST, address: 100 },
                          Binding { id: 1, bytes: 8, memory: DEVICE, address: 100 }];
        validate(&mut owners).unwrap();
        assert_eq!((owners[0].id, owners[1].id), (1, 2));
        let mut overlap = [Binding { id: 1, bytes: 8, memory: DEVICE, address: 100 },
                           Binding { id: 2, bytes: 8, memory: DEVICE, address: 107 }];
        assert!(validate(&mut overlap).is_err());
        let mut duplicate = [owners[0], owners[0]];
        assert!(validate(&mut duplicate).is_err());
    }
}
