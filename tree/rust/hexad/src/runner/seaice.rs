// SPDX-License-Identifier: Apache-2.0
//! Live host pointer construction from the pinned sea-ice driver. This rule
//! needs independent source, allocation-binding and forecast equality checks.
use super::{plan::Slice, submit::Allocation};
use std::collections::HashMap;

// hexcore/mpasphys/seaice_v842.py:28 and :890-897 at the required source pin.
pub const FIELDS: [&str; 53] = [
    "t3d", "qv3d", "p8w3d", "dz8w", "sr", "glw", "qgh", "swdown", "rainbl", "snoalb2d",
    "xice", "rib", "chs", "tslb", "emiss", "albedo", "albsi", "z02d", "snow", "tsk", "snowc",
    "snowh2d", "cqs2", "acsnow", "acsnom", "sfcrunoff", "potevp", "snopcx", "hfx", "lh", "qfx",
    "znt", "grdflx", "qsfc", "chs2", "noahres", "snowsi", "icedepth", "gsw", "sr_input", "rainncv",
    "raincv", "tsk_sea", "cpm", "cpm_sea", "chs_sea", "chs2_sea", "cqs2_sea", "hfx_sea", "lh_sea",
    "qfx_sea", "qgh_sea", "qsfc_sea",
];
pub const SEA_FLUX_FIELDS: [&str; 9] = ["cpm_sea", "chs_sea", "chs2_sea", "cqs2_sea", "hfx_sea",
                                         "lh_sea", "qfx_sea", "qgh_sea", "qsfc_sea"];

fn address(target: Slice, allocations: &HashMap<u64, Allocation>) -> Result<u64, String> {
    let allocation = allocations.get(&target.allocation)
        .ok_or_else(|| format!("unknown pointer target {}", target.allocation))?;
    if target.offset >= allocation.bytes { return Err("pointer target outside allocation".into()); }
    let offset = u64::try_from(target.offset).map_err(|_| "pointer target offset overflow")?;
    allocation.address.checked_add(offset).ok_or_else(|| "device address overflow".into())
}

pub fn pointer_table(arrays: &HashMap<String, Slice>, dummy: Slice,
                     allocations: &HashMap<u64, Allocation>) -> Result<Vec<u8>, String> {
    if arrays.keys().any(|name| !FIELDS.contains(&name.as_str())) {
        return Err("unknown seaice pointer field".into());
    }
    let fallback = address(dummy, allocations)?;
    let mut bytes = Vec::with_capacity(FIELDS.len() * 8);
    for field in FIELDS {
        let pointer = match arrays.get(field) {
            Some(target) => address(*target, allocations)?,
            None if SEA_FLUX_FIELDS.contains(&field) => 0,
            None => fallback,
        };
        bytes.extend_from_slice(&pointer.to_le_bytes());
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn live_array_dummy_and_absent_flux_are_bound_in_source_order() {
        let allocations = HashMap::from([(1, Allocation { id: 1, address: 0x1000, bytes: 16 }),
                                         (2, Allocation { id: 2, address: 0x2000, bytes: 8 })]);
        let dummy = Slice { allocation: 2, offset: 0 };
        let mut arrays = HashMap::from([("t3d".into(), Slice { allocation: 1, offset: 8 }),
                                        ("cpm_sea".into(), Slice { allocation: 1, offset: 0 })]);
        let bytes = pointer_table(&arrays, dummy, &allocations).unwrap();
        let word = |index: usize| u64::from_le_bytes(bytes[8*index..8*index+8].try_into().unwrap());
        assert_eq!(bytes.len(), 53 * 8);
        assert_eq!(word(0), 0x1008);
        assert_eq!(word(1), 0x2000);
        assert_eq!(word(44), 0x1000);
        assert_eq!(word(45), 0);
        arrays.insert("not_a_field".into(), dummy);
        assert!(pointer_table(&arrays, dummy, &allocations).is_err());
        assert!(pointer_table(&HashMap::new(), Slice { allocation: 2, offset: 8 }, &allocations).is_err());
    }
}
