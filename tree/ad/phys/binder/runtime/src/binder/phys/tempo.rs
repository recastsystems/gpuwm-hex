// SPDX-License-Identifier: Apache-2.0
//! TEMPO's pointer words, constructed from live device views.
//! Source: rrfs/tempo_v303.py:4548-4574, source pin 16251589f19aec0012fd164d1ea060126208aed1.
//! This rule does not establish which named arrays, presence flags, cache
//! entries or call sites are live. The plan producer must establish them.
use super::{plan::Slice, submit::Allocation};
use serde::Deserialize;
use std::collections::HashMap;

#[path = "tempo_selection.rs"]
pub mod selection;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct View {
    /// A null target is valid only for an empty array.
    pub target: Option<Slice>,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub strides: Vec<i64>,
    pub device: u64,
}

fn address(view: &View, expected_shape: Option<&[usize]>, device: u64,
           allocations: &HashMap<u64, Allocation>) -> Result<u64, String> {
    if view.dtype != "<f4" || view.device != device || view.shape.len() != view.strides.len() {
        return Err("TEMPO view dtype, rank or device differs from source".into());
    }
    if let Some(shape) = expected_shape {
        if view.shape != shape { return Err("TEMPO view shape differs from source".into()); }
    }
    let empty = view.shape.contains(&0);
    let mut bytes = 4usize;
    // CuPy treats empty arrays as C-contiguous and ignores strides on axes
    // of extent one, just as NumPy does. All other axes must be C ordered.
    if !empty {
        for (&extent, &stride) in view.shape.iter().zip(&view.strides).rev() {
            if extent > 1 && (stride < 0 || usize::try_from(stride).ok() != Some(bytes)) {
                return Err("TEMPO view is not C-contiguous".into());
            }
            bytes = bytes.checked_mul(extent).ok_or("TEMPO view extent overflow")?;
        }
    } else { bytes = 0; }
    match view.target {
        None if bytes == 0 => Ok(0),
        None => Err("nonempty TEMPO view has a null pointer".into()),
        Some(target) => {
            let allocation = allocations.get(&target.allocation).ok_or("unknown TEMPO allocation")?;
            let end = target.offset.checked_add(bytes).ok_or("TEMPO view extent overflow")?;
            if end > allocation.bytes { return Err("TEMPO view outside allocation".into()); }
            let offset = u64::try_from(target.offset).map_err(|_| "TEMPO offset overflow")?;
            allocation.address.checked_add(offset).ok_or_else(|| "TEMPO address overflow".into())
        }
    }
}

/// Returns pointer words only. The source caches them by (device, words) and
/// uploads only on a miss; callers must derive cache membership live.
pub fn pointer_table(entries: &[Option<View>], expected_shape: Option<&[usize]>,
                     device: u64, optional: bool, allocations: &HashMap<u64, Allocation>)
                     -> Result<Vec<u8>, String> {
    let size = entries.len().checked_mul(8).ok_or("TEMPO table size overflow")?;
    let mut table = Vec::with_capacity(size);
    for entry in entries {
        let word = match entry {
            None if optional => 0,
            None => return Err("required TEMPO view is absent".into()),
            Some(view) => address(view, expected_shape, device, allocations)?,
        };
        table.extend_from_slice(&word.to_le_bytes());
    }
    Ok(table)
}

/// The forward starts its process with an empty dictionary; the caller must
/// establish that start condition for its plan. Entries retain the allocation
/// containing their table until the source's 64-entry clear takes place.
#[derive(Default, Clone)]
pub struct Cache {
    entries: HashMap<(u64, Vec<u8>), Slice>,
}
impl Cache {
    pub fn get(&self, device: u64, words: &[u8]) -> Option<Slice> {
        self.entries.get(&(device, words.to_vec())).copied()
    }
    pub fn clear_on_full(&mut self) -> bool {
        if self.entries.len() >= 64 { self.entries.clear(); true } else { false }
    }
    pub fn insert(&mut self, device: u64, words: Vec<u8>, target: Slice) -> Result<(), String> {
        if words.is_empty() || words.len() % 8 != 0 || self.entries.len() >= 64 {
            return Err("invalid TEMPO cache insertion".into());
        }
        let end = target.offset.checked_add(words.len()).ok_or("TEMPO cache target overflow")?;
        for ((_, other_words), other) in &self.entries {
            let other_end = other.offset.checked_add(other_words.len()).ok_or("TEMPO cache target overflow")?;
            if target.allocation == other.allocation && target.offset < other_end && other.offset < end {
                return Err("TEMPO cache tables overlap".into());
            }
        }
        let key = (device, words);
        if self.entries.contains_key(&key) {
            return Err("duplicate TEMPO cache insertion".into());
        }
        self.entries.insert(key, target);
        Ok(())
    }
    pub fn owns_allocation(&self, id: u64) -> bool {
        self.entries.values().any(|slice| slice.allocation == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn view() -> View {
        View { target: Some(Slice { allocation: 1, offset: 4 }), dtype: "<f4".into(),
               shape: vec![2, 3], strides: vec![12, 4], device: 0 }
    }
    #[test]
    fn live_pointer_and_optional_entries_are_preserved() {
        let allocations = HashMap::from([(1, Allocation { id: 1, address: 0x1000, bytes: 28 })]);
        let bytes = pointer_table(&[Some(view()), None], Some(&[2, 3]), 0, true, &allocations).unwrap();
        assert_eq!(&bytes[..8], &0x1004u64.to_le_bytes());
        assert_eq!(&bytes[8..], &0u64.to_le_bytes());
        assert!(pointer_table(&[None], None, 0, false, &allocations).is_err());
        assert!(pointer_table(&[Some(view())], Some(&[3, 2]), 0, true, &allocations).is_err());
        assert!(pointer_table(&[Some(view())], None, 1, true, &allocations).is_err());
    }
    #[test]
    fn rejects_bad_views_and_handles_degenerate_axes() {
        let allocations = HashMap::from([(1, Allocation { id: 1, address: 16, bytes: 28 })]);
        let mut x = view(); x.strides[0] = 4;
        assert!(pointer_table(&[Some(x)], None, 0, false, &allocations).is_err());
        let mut x = view(); x.dtype = "<f8".into();
        assert!(pointer_table(&[Some(x)], None, 0, false, &allocations).is_err());
        let mut x = view(); x.target = Some(Slice { allocation: 1, offset: 8 });
        assert!(pointer_table(&[Some(x)], None, 0, false, &allocations).is_err());
        let mut x = view(); x.shape = vec![1, 3]; x.strides = vec![-77, 4];
        assert!(pointer_table(&[Some(x)], None, 0, false, &allocations).is_ok());
        let mut x = view(); x.shape = vec![0, 3]; x.strides = vec![-77, 4]; x.target = None;
        assert_eq!(pointer_table(&[Some(x)], None, 0, false, &allocations).unwrap(), vec![0; 8]);
    }
    #[test]
    fn cache_uses_live_words_and_clears_only_on_the_next_miss() {
        let mut cache = Cache::default();
        for i in 0..64u64 {
            let words = i.to_le_bytes().to_vec();
            assert!(!cache.clear_on_full());
            cache.insert(0, words, Slice { allocation: i + 1, offset: 0 }).unwrap();
        }
        assert_eq!(cache.get(0, &42u64.to_le_bytes()).unwrap().allocation, 43);
        assert!(cache.get(1, &42u64.to_le_bytes()).is_none());
        assert!(cache.clear_on_full());
        assert!(cache.get(0, &42u64.to_le_bytes()).is_none());
        cache.insert(1, vec![0; 8], Slice { allocation: 7, offset: 0 }).unwrap();
        assert!(cache.owns_allocation(7));
        assert!(cache.insert(1, 1u64.to_le_bytes().to_vec(),
                             Slice { allocation: 7, offset: 4 }).is_err());
    }
}
