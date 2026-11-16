// SPDX-License-Identifier: Apache-2.0
//! RUC boundary pointer assembly from live named device views and flags.
//! Source: resident_ruc_rrfs.py:94-127, ruc_lsm_8c9f1a8.py:36.
//! This rule does not establish view ownership, flag contents or scheduling.
use super::{plan::Slice, submit::Allocation};
use serde::Deserialize;
use std::collections::HashMap;

#[path = "ruc_layout.rs"]
mod layout;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dtype { I32, F32 }

#[derive(Clone, Copy)]
struct Field { name: &'static str, dtype: Dtype, optional: bool }

#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct View {
    /// An absent allocation is valid only for an empty, null device view.
    pub target: Option<Slice>,
    pub bytes: usize,
    pub dtype: Dtype,
    pub contiguous: bool,
    /// None indicates a scalar rather than the array required by the source.
    pub last_dimension: Option<usize>,
}

fn address(view: View, required_dtype: Dtype, n: usize,
           allocations: &HashMap<u64, Allocation>) -> Result<u64, String> {
    if view.dtype != required_dtype || !view.contiguous || view.last_dimension != Some(n) {
        return Err("RUC view dtype, contiguity or final extent differs from source".into());
    }
    if view.bytes < n.checked_mul(4).ok_or("RUC extent overflow")? || view.bytes % 4 != 0 {
        return Err("RUC view is too small for its final extent".into());
    }
    match view.target {
        None if view.bytes == 0 && n == 0 => Ok(0),
        None => Err("nonempty RUC view has a null pointer".into()),
        Some(target) => {
            let allocation = allocations.get(&target.allocation).ok_or("unknown RUC allocation")?;
            let end = target.offset.checked_add(view.bytes).ok_or("RUC view extent overflow")?;
            if end > allocation.bytes { return Err("RUC view outside allocation".into()); }
            let offset = u64::try_from(target.offset).map_err(|_| "RUC view offset overflow")?;
            allocation.address.checked_add(offset).ok_or_else(|| "RUC device address overflow".into())
        }
    }
}

pub fn pointer_table(entry: &str, arguments: &HashMap<String, View>, flags: &HashMap<String, View>,
                     tile_stop: Option<View>, allocations: &HashMap<u64, Allocation>)
                     -> Result<Vec<u8>, String> {
    let fields = match entry {
        "land_ruc_land" => layout::LAND,
        "ice_ruc_ice" => layout::ICE,
        "diags_sfcdiags_ruclsm" => layout::DIAGNOSTICS,
        "init_rucinit" => layout::INITIALIZE,
        "drivers_seaice_prepare" => layout::ICE_PREPARE,
        "drivers_seaice_finish" => layout::ICE_FINISH,
        "drivers_correct_tsk_over_seaice" => layout::ICE_TSK,
        "drivers_driver_forcing" => layout::FORCING,
        _ => return Err("unknown RUC entry".into()),
    };
    if arguments.len() != fields.len() || flags.len() != fields.iter().filter(|f| f.optional).count() {
        return Err("RUC named array or optional flag count differs from source".into());
    }
    let first = arguments.get(fields[0].name).ok_or("missing first RUC argument")?;
    let n = first.last_dimension.ok_or("RUC argument has no final dimension")?;
    let mut pointers = Vec::with_capacity(fields.len() + flags.len() + 3);
    for field in fields {
        let view = *arguments.get(field.name).ok_or("missing RUC named argument")?;
        pointers.push(address(view, field.dtype, n, allocations)?);
    }
    for field in fields.iter().filter(|field| field.optional) {
        let view = *flags.get(field.name).ok_or("missing RUC optional presence flag")?;
        if view.bytes != n.checked_mul(4).ok_or("RUC flag extent overflow")? {
            return Err("RUC optional presence flag has wrong size".into());
        }
        pointers.push(address(view, Dtype::I32, n, allocations)?);
    }
    match (entry, tile_stop) {
        ("land_ruc_land", Some(view)) if view.bytes == 8 => {
            pointers.push(address(view, Dtype::I32, 2, allocations)?);
        }
        ("land_ruc_land", _) => return Err("land RUC tile-stop array is missing or invalid".into()),
        (_, None) => (),
        (_, Some(_)) => return Err("unexpected RUC tile-stop array".into()),
    }
    if entry == "land_ruc_land" || entry == "ice_ruc_ice" { pointers.extend([0, 0]); }
    Ok(pointers.into_iter().flat_map(u64::to_le_bytes).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_order_flags_tile_and_nulls_and_bad_inputs() {
        let alloc = HashMap::from([(1, Allocation { id: 1, address: 0x1000, bytes: 4096 })]);
        let mut fields = HashMap::new();
        let mut flags = HashMap::new();
        for (i, field) in layout::LAND.iter().enumerate() {
            let view = View { target: Some(Slice { allocation: 1, offset: i * 8 }),
                              bytes: 8, dtype: field.dtype, contiguous: true, last_dimension: Some(2) };
            fields.insert(field.name.into(), view);
            if field.optional { flags.insert(field.name.into(), View { dtype: Dtype::I32, ..view }); }
        }
        let tile = View { target: Some(Slice { allocation: 1, offset: 2000 }), bytes: 8,
                          dtype: Dtype::I32, contiguous: true, last_dimension: Some(2) };
        let bytes = pointer_table("land_ruc_land", &fields, &flags, Some(tile), &alloc).unwrap();
        let words: Vec<_> = bytes.chunks_exact(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect();
        assert_eq!(words.len(), layout::LAND.len() + flags.len() + 3);
        assert_eq!(words[0], 0x1000);
        assert_eq!(words[layout::LAND.len() + flags.len()], 0x1000 + 2000);
        assert_eq!(&words[words.len()-2..], &[0, 0]);
        assert!(pointer_table("land_ruc_land", &fields, &flags, None, &alloc).is_err());
        fields.get_mut(layout::LAND[0].name).unwrap().bytes = 9000;
        assert!(pointer_table("land_ruc_land", &fields, &flags, Some(tile), &alloc).is_err());
    }
}
