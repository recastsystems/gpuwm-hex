// SPDX-License-Identifier: Apache-2.0
//! Provisional injection plan for device and host storage. Complete
//! allocation discovery and the real two-boundary physics proof remain
//! independent requirements; a supplied live set is not a census proof.
use super::census::{self, Extent, PoisonWrite, WordKind};
use super::{host_state::HostState, plan, submit};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExtentSpec { allocation: u64, offset: u64, bytes: u64, kind: Kind }
#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Kind { Float, Float64, Integer, IntegerBytes }
impl From<&ExtentSpec> for Extent {
    fn from(v: &ExtentSpec) -> Self {
        Extent { allocation: v.allocation, offset: v.offset, bytes: v.bytes,
            kind: match v.kind { Kind::Float => WordKind::Float, Kind::Float64 => WordKind::Float64,
                Kind::Integer => WordKind::Integer, Kind::IntegerBytes => WordKind::IntegerBytes } }
    }
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Mode { Noncarried, Bite { allocation: u64, offset: u64 }, HostBite { allocation: u64, offset: u64 } }

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum HostKind { Float32, Float64, Integer }
impl HostKind {
    fn pattern(self) -> &'static [u8] {
        match self {
            Self::Float32 => &[0x00, 0x00, 0xc0, 0x7f],
            Self::Float64 => &[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f],
            Self::Integer => &[0xef, 0xbe, 0xad, 0xde],
        }
    }
    fn alignment(self) -> usize { match self { Self::Float32 => 4, Self::Float64 => 8, Self::Integer => 1 } }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HostExtentSpec { allocation: u64, offset: usize, bytes: usize, kind: HostKind }

#[derive(Clone, Copy, Debug)]
pub struct HostWrite {
    pub allocation: u64,
    pub offset: usize,
    pub bytes: usize,
    pub pattern: &'static [u8],
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoisonPlan {
    /// The boundary immediately before this one-based next step.
    pub next_step: usize,
    live: Vec<ExtentSpec>,
    carried: Vec<ExtentSpec>,
    /// Every physical host allocation must be covered exactly, including
    /// noncarried storage. The logical frame is always wholly carried and is
    /// excluded because its backing Rust objects are not this byte buffer.
    #[serde(default)] host_live: Vec<HostExtentSpec>,
    mode: Mode,
}
impl PoisonPlan {
    pub fn validate_bindings(&self, allocations: &HashMap<u64, submit::Allocation>,
                             carried_views: &[plan::CarriedView], host_state: &HostState,
                             host_carried: &[u64], frame_id: u64) -> Result<(), String> {
        let mut expected = Vec::with_capacity(carried_views.len());
        for view in carried_views {
            expected.push((view.target.allocation,
                           u64::try_from(view.target.offset).map_err(|_| "carried offset overflow")?,
                           u64::try_from(view.bytes).map_err(|_| "carried size overflow")?));
        }
        let mut supplied: Vec<_> = self.carried.iter().map(|v| (v.allocation, v.offset, v.bytes)).collect();
        expected.sort_unstable();
        supplied.sort_unstable();
        if supplied != expected { return Err("poison carried extents differ from replay".into()); }
        for extent in self.live.iter().chain(&self.carried) {
            let allocation = allocations.get(&extent.allocation)
                .ok_or_else(|| format!("unknown poison allocation {}", extent.allocation))?;
            let end = extent.offset.checked_add(extent.bytes).ok_or("poison extent overflow")?;
            if end > allocation.bytes as u64 { return Err("poison extent exceeds allocation".into()); }
        }
        let mut sizes = host_state.sizes();
        if sizes.remove(&frame_id).is_none() || !host_carried.contains(&frame_id) {
            return Err("the live logical frame must be carried".into());
        }
        let mut expected_host: HashSet<_> = host_carried.iter().copied().collect();
        if expected_host.len() != host_carried.len() { return Err("duplicate carried host allocation".into()); }
        expected_host.remove(&frame_id);
        if expected_host.iter().any(|id| !sizes.contains_key(id)) {
            return Err("unknown carried host allocation".into());
        }
        let mut ranges: HashMap<u64, Vec<(usize, usize)>> = HashMap::new();
        for extent in &self.host_live {
            let size = sizes.get(&extent.allocation).ok_or("unknown host poison allocation")?;
            let end = extent.offset.checked_add(extent.bytes).ok_or("host poison extent overflow")?;
            let alignment = extent.kind.alignment();
            if extent.bytes == 0 || end > *size || extent.offset % alignment != 0 || extent.bytes % alignment != 0 {
                return Err("invalid typed host poison extent".into());
            }
            ranges.entry(extent.allocation).or_default().push((extent.offset, end));
        }
        if ranges.len() != sizes.len() { return Err("host poison live set misses an allocation".into()); }
        for (id, size) in sizes {
            let regions = ranges.get_mut(&id).ok_or("host poison live set misses an allocation")?;
            regions.sort_unstable();
            let mut cursor = 0;
            for &(start, end) in regions.iter() {
                if start != cursor { return Err("host poison extents overlap or leave a gap".into()); }
                cursor = end;
            }
            if cursor != size { return Err("host poison live set misses allocation bytes".into()); }
        }
        Ok(())
    }
    pub fn writes(&self, steps: usize) -> Result<Vec<PoisonWrite>, String> {
        if self.next_step == 0 || self.next_step.checked_add(1).is_none_or(|end| end > steps) {
            return Err("poison boundary must leave at least two executed steps".into());
        }
        let live: Vec<_> = self.live.iter().map(Extent::from).collect();
        let carried: Vec<_> = self.carried.iter().map(Extent::from).collect();
        // Always validate both complete supplied extent sets, even for a bite.
        let noncarried = census::noncarried_poison(&live, &carried).map_err(str::to_owned)?;
        match self.mode {
            Mode::Noncarried => Ok(noncarried),
            Mode::Bite { allocation, offset } => Ok(vec![census::bite(&carried, allocation, offset).map_err(str::to_owned)?]),
            Mode::HostBite { .. } => Ok(Vec::new()),
        }
    }

    pub fn host_writes(&self, host_carried: &[u64], frame_id: u64) -> Result<Vec<HostWrite>, String> {
        let carried: HashSet<_> = host_carried.iter().copied().collect();
        match self.mode {
            Mode::Noncarried => Ok(self.host_live.iter().filter(|v| !carried.contains(&v.allocation))
                .map(|v| HostWrite { allocation: v.allocation, offset: v.offset, bytes: v.bytes,
                                      pattern: v.kind.pattern() }).collect()),
            Mode::Bite { .. } => Ok(Vec::new()),
            Mode::HostBite { allocation, offset } => {
                if allocation == frame_id || !carried.contains(&allocation) {
                    return Err("host bite must target physical carried host storage".into());
                }
                let offset = usize::try_from(offset).map_err(|_| "host bite offset overflow")?;
                let region = self.host_live.iter().find(|v| v.allocation == allocation && v.offset <= offset
                    && v.offset.checked_add(v.bytes).is_some_and(|end| offset < end))
                    .ok_or("host bite target is outside the live set")?;
                let pattern = region.kind.pattern();
                let end = offset.checked_add(pattern.len()).ok_or("host bite extent overflow")?;
                if offset % pattern.len() != 0 || end > region.offset + region.bytes {
                    return Err("unaligned host bite or bite exceeds typed extent".into());
                }
                Ok(vec![HostWrite { allocation, offset, bytes: pattern.len(), pattern }])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn plan(mode: &str, next_step: usize) -> PoisonPlan {
        serde_json::from_str(&format!(r#"{{"next_step":{next_step},
          "live":[{{"allocation":1,"offset":0,"bytes":16,"kind":"float"}}],
          "carried":[{{"allocation":1,"offset":4,"bytes":8,"kind":"float"}}],
          "host_live":[{{"allocation":2,"offset":0,"bytes":12,"kind":"float32"}},
                       {{"allocation":3,"offset":0,"bytes":8,"kind":"float64"}},
                       {{"allocation":4,"offset":0,"bytes":5,"kind":"integer"}}],
          "mode":{mode}}}"#)).unwrap()
    }

    fn hosts() -> HostState {
        let mut state = HostState::default();
        state.insert(2, vec![1;12]).unwrap();
        state.insert(3, vec![2;8]).unwrap();
        state.insert(4, vec![3;5]).unwrap();
        state.insert(9, vec![4;103]).unwrap();
        state
    }
    #[test]
    fn proves_only_supplied_noncarried_intervals() {
        let p = plan(r#"{"kind":"noncarried"}"#, 1);
        let writes = p.writes(2).unwrap();
        assert_eq!(writes.len(), 2);
        assert_eq!((writes[0].extent.offset, writes[0].extent.bytes, writes[0].word), (0, 4, 0x7fc0_0000));
        assert_eq!((writes[1].extent.offset, writes[1].extent.bytes, writes[1].word), (12, 4, 0x7fc0_0000));
        assert!(p.writes(1).is_err());
    }
    #[test]
    fn bite_requires_a_word_inside_a_supplied_carried_extent() {
        let p = plan(r#"{"kind":"bite","allocation":1,"offset":4}"#, 2);
        assert_eq!(p.writes(3).unwrap()[0].extent.bytes, 4);
        assert!(p.writes(2).is_err());
        let wrong = plan(r#"{"kind":"bite","allocation":1,"offset":0}"#, 1);
        assert!(wrong.writes(2).is_err());
    }

    #[test]
    fn poison_must_match_complete_carried_extents() {
        let p = plan(r#"{"kind":"noncarried"}"#, 1);
        let live = HashMap::from([(1, submit::Allocation { id: 1, address: 0x1000, bytes: 16 })]);
        let mut view = plan::CarriedView { id: 77, target: plan::Slice { allocation: 1, offset: 4 }, bytes: 8 };
        p.validate_bindings(&live, &[view], &hosts(), &[2,9], 9).unwrap();
        view.bytes = 12;
        assert!(p.validate_bindings(&live, &[view], &hosts(), &[2,9], 9).is_err());
        view.bytes = 8;
        view.target.offset = 0;
        assert!(p.validate_bindings(&live, &[view], &hosts(), &[2,9], 9).is_err());
    }

    #[test]
    fn host_poison_covers_noncarried_state_and_refuses_an_incomplete_inventory() {
        let p = plan(r#"{"kind":"noncarried"}"#, 1);
        let live = HashMap::from([(1, submit::Allocation { id: 1, address: 0x1000, bytes: 16 })]);
        let view = plan::CarriedView { id: 77, target: plan::Slice { allocation: 1, offset: 4 }, bytes: 8 };
        let mut state = hosts();
        p.validate_bindings(&live, &[view], &state, &[2,9], 9).unwrap();
        let writes = p.host_writes(&[2,9], 9).unwrap();
        state.inject_poison(&writes).unwrap();
        assert_eq!(state.get(2).unwrap(), &[1;12]);
        assert!(f64::from_le_bytes(state.get(3).unwrap().try_into().unwrap()).is_nan());
        assert_eq!(state.get(4).unwrap(), &[0xef,0xbe,0xad,0xde,0xef]);
        assert_eq!(state.get(9).unwrap(), &[4;103]);
        let mut incomplete = plan(r#"{"kind":"noncarried"}"#, 1);
        incomplete.host_live.pop();
        assert!(incomplete.validate_bindings(&live, &[view], &state, &[2,9], 9).is_err());
        incomplete.host_live[0].offset = 4;
        assert!(incomplete.validate_bindings(&live, &[view], &state, &[2,9], 9).is_err());
    }

    #[test]
    fn host_bite_targets_real_carried_storage_not_frame_shadow_bytes() {
        let p = plan(r#"{"kind":"host_bite","allocation":2,"offset":4}"#, 1);
        assert!(p.writes(2).unwrap().is_empty());
        let mut state = hosts();
        state.inject_poison(&p.host_writes(&[2,9], 9).unwrap()).unwrap();
        assert!(f32::from_le_bytes(state.get(2).unwrap()[4..8].try_into().unwrap()).is_nan());
        let frame = plan(r#"{"kind":"host_bite","allocation":9,"offset":0}"#, 1);
        assert!(frame.host_writes(&[2,9], 9).is_err());
        let other = plan(r#"{"kind":"host_bite","allocation":3,"offset":0}"#, 1);
        assert!(other.host_writes(&[2,9], 9).is_err());
    }
}
