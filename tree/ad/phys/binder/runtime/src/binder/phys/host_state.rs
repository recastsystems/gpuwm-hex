// SPDX-License-Identifier: Apache-2.0
//! Live CPU storage for the physics binder. Initial bytes are replay inputs;
//! subsequent values are produced only by live host rules or device downloads.
use super::host_array::{self, Order, View};
use super::poison::HostWrite;
use std::collections::HashMap;

#[derive(Default, Clone)]
pub struct HostState { allocations: HashMap<u64, Vec<u8>> }

impl HostState {
    pub fn insert(&mut self, id: u64, bytes: Vec<u8>) -> Result<(), String> {
        if bytes.is_empty() || self.allocations.contains_key(&id) {
            return Err(format!("empty or duplicate host allocation {id}"));
        }
        self.allocations.insert(id, bytes);
        Ok(())
    }

    pub fn get(&self, id: u64) -> Result<&[u8], String> {
        self.allocations.get(&id).map(Vec::as_slice).ok_or_else(|| format!("unknown host allocation {id}"))
    }

    pub fn sizes(&self) -> HashMap<u64, usize> {
        self.allocations.iter().map(|(&id, value)| (id, value.len())).collect()
    }

    /// Return process-local owner addresses for one synchronous callback.
    /// The caller must not alter the allocation map or retain the pointers
    /// after that callback. Independently established owner IDs are required.
    pub fn binding_addresses(&mut self, ids: &[u64]) -> Result<Vec<(u64, usize, u64)>, String> {
        let mut seen = std::collections::HashSet::new();
        for id in ids {
            if *id == 0 || !seen.insert(*id) || !self.allocations.contains_key(id) {
                return Err(format!("unknown or duplicate host checkpoint owner {id}"));
            }
        }
        ids.iter().map(|id| {
            let value = self.allocations.get_mut(id).ok_or("missing host checkpoint owner")?;
            Ok((*id, value.len(), value.as_mut_ptr() as u64))
        }).collect()
    }

    /// Check all host poison writes before changing any live storage.
    pub fn validate_poison(&self, writes: &[HostWrite]) -> Result<(), String> {
        let mut ranges = Vec::with_capacity(writes.len());
        for write in writes {
            let end = write.offset.checked_add(write.bytes).ok_or("host poison extent overflow")?;
            if write.bytes == 0 || write.pattern.is_empty() || end > self.get(write.allocation)?.len() {
                return Err("host poison extent exceeds storage".into());
            }
            ranges.push((write.allocation, write.offset, end));
        }
        ranges.sort_unstable();
        if ranges.windows(2).any(|p| p[0].0 == p[1].0 && p[0].2 > p[1].1) {
            return Err("overlapping host poison writes".into());
        }
        Ok(())
    }

    pub fn inject_poison(&mut self, writes: &[HostWrite]) -> Result<(), String> {
        self.validate_poison(writes)?;
        for write in writes {
            let bytes = &mut self.allocations.get_mut(&write.allocation).ok_or("unknown host allocation")?
                [write.offset..write.offset + write.bytes];
            let mut chunks = bytes.chunks_exact_mut(write.pattern.len());
            for chunk in &mut chunks { chunk.copy_from_slice(write.pattern); }
            let tail = chunks.into_remainder();
            tail.copy_from_slice(&write.pattern[..tail.len()]);
        }
        Ok(())
    }

    /// Replace a same-sized logical allocation with bytes computed from a
    /// live host rule. The caller owns the identity and provenance check.
    pub fn replace(&mut self, id: u64, bytes: &[u8]) -> Result<(), String> {
        let allocation = self.allocations.get_mut(&id).ok_or_else(|| format!("unknown host allocation {id}"))?;
        if allocation.len() != bytes.len() { return Err("logical host allocation size changed".into()); }
        allocation.copy_from_slice(bytes);
        Ok(())
    }

    pub fn write_range(&mut self, id: u64, offset: usize, bytes: &[u8]) -> Result<(), String> {
        let end = offset.checked_add(bytes.len()).ok_or("host write overflow")?;
        let allocation = self.allocations.get_mut(&id).ok_or_else(|| format!("unknown host allocation {id}"))?;
        allocation.get_mut(offset..end).ok_or("host write outside allocation")?.copy_from_slice(bytes);
        Ok(())
    }

    /// TEMPO clears its shared first-call SAVE flag immediately after its
    /// main microphysics launch succeeds. The producer must place this rule
    /// on every successful main path and bind the actual shared owner.
    pub fn finish_tempo_main(&mut self, id: u64, offset: usize) -> Result<(), String> {
        let allocation = self.allocations.get_mut(&id).ok_or_else(|| format!("unknown host allocation {id}"))?;
        let flag = allocation.get_mut(offset).ok_or("TEMPO first-call flag outside allocation")?;
        if *flag > 1 { return Err("invalid TEMPO first-call boolean".into()); }
        *flag = 0;
        Ok(())
    }

    pub fn pack(&self, id: u64, view: &View, order: Order) -> Result<Vec<u8>, String> {
        host_array::pack(self.get(id)?, view, order).map_err(str::to_owned)
    }

    pub fn publish(&mut self, id: u64, view: &View, order: Order, bytes: &[u8]) -> Result<(), String> {
        let allocation = self.allocations.get_mut(&id).ok_or_else(|| format!("unknown host allocation {id}"))?;
        host_array::unpack(allocation, view, order, bytes).map_err(str::to_owned)
    }
}

/// Count different four-byte words, including a final partial word. The
/// reference is used only for checking and is never inserted into live state.
pub fn differing_words(actual: &[u8], reference: &[u8]) -> Result<u64, String> {
    if actual.len() != reference.len() { return Err("host reference length differs".into()); }
    u64::try_from(actual.chunks(4).zip(reference.chunks(4)).filter(|(a, b)| a != b).count())
        .map_err(|_| "host differing word count overflow".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn live_state_rejects_duplicate_and_publishes_only_the_selected_words() {
        let mut state = HostState::default();
        state.insert(7, (0..12).collect()).unwrap();
        assert!(state.insert(7, vec![9]).is_err());
        let view = View { offset: 2, shape: vec![2], strides: vec![4], item_bytes: 2 };
        assert_eq!(state.pack(7, &view, Order::C).unwrap(), [2,3,6,7]);
        state.publish(7, &view, Order::C, &[21,22,23,24]).unwrap();
        assert_eq!(state.get(7).unwrap(), [0,1,21,22,4,5,23,24,8,9,10,11]);
        assert!(state.publish(7, &view, Order::C, &[1]).is_err());
        assert_eq!(state.get(7).unwrap(), [0,1,21,22,4,5,23,24,8,9,10,11]);
        assert!(state.replace(7, &[0]).is_err());
        assert_eq!(state.get(7).unwrap(), [0,1,21,22,4,5,23,24,8,9,10,11]);
        state.replace(7, &[5;12]).unwrap();
        assert_eq!(state.get(7).unwrap(), &[5;12]);
    }
    #[test]
    fn word_counts_include_a_final_partial_word() {
        assert_eq!(differing_words(&[0;9], &[0,1,1,0,0,0,0,0,2]).unwrap(), 2);
        assert!(differing_words(&[0;4], &[0;3]).is_err());
    }
    #[test]
    fn tempo_first_call_clears_only_its_valid_shared_flag() {
        let mut state = HostState::default();
        state.insert(11, vec![7, 1, 8]).unwrap();
        state.finish_tempo_main(11, 1).unwrap();
        assert_eq!(state.get(11).unwrap(), &[7, 0, 8]);
        state.finish_tempo_main(11, 1).unwrap();
        assert_eq!(state.get(11).unwrap(), &[7, 0, 8]);
        assert!(state.finish_tempo_main(11, 0).is_err());
        assert!(state.finish_tempo_main(11, 3).is_err());
        assert!(state.finish_tempo_main(12, 0).is_err());
        assert_eq!(state.get(11).unwrap(), &[7, 0, 8]);
    }
    #[test]
    fn checkpoint_bindings_address_live_unique_owners() {
        let mut state = HostState::default();
        state.insert(4, vec![1, 2]).unwrap();
        state.insert(9, vec![3]).unwrap();
        assert!(state.binding_addresses(&[4, 4]).is_err());
        assert!(state.binding_addresses(&[4, 8]).is_err());
        let bindings = state.binding_addresses(&[9, 4]).unwrap();
        assert_eq!((bindings[0].0, bindings[0].1), (9, 1));
        unsafe { *(bindings[1].2 as *mut u8) = 8; }
        assert_eq!(state.get(4).unwrap(), &[8, 2]);
    }
}
