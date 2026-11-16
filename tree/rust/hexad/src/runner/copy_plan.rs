// SPDX-License-Identifier: Apache-2.0
//! RRFS glue-plan table construction from live array descriptors.
//! The plan producer must establish each descriptor from the current forecast;
//! this routine does not itself certify host ownership or a physics replay.
use super::{plan::Slice, submit::Allocation};
use serde::Deserialize;
use std::collections::HashMap;

/// Bind every descriptor word to current host state. For a copy the source
/// and its strides are present and fill is the source's literal zero. For a
/// fill, source/strides are absent and fill points to the live cast word.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RowSpec {
    pub dst: Slice,
    pub src: Option<Slice>,
    pub dst_stride: [Slice; 4],
    pub src_stride: Option<[Slice; 4]>,
    pub shape: [Slice; 4],
    pub fill: Option<Slice>,
}

/// Materialized inputs to `CopyPlan._add` (glue_plan.py:155). Strides are
/// in four-byte words and dimensions include the rank-four padding.
pub struct Row {
    pub dst: Slice,
    pub src: Option<Slice>,
    pub dst_stride: [i64; 4],
    pub src_stride: [i64; 4],
    pub shape: [i64; 4],
    pub fill: u32,
    pub mode: i32,
}

#[derive(Debug, Eq, PartialEq)]
pub struct Launch { pub grid: [u32; 3], pub block: [u32; 3], pub tiled: bool }

/// A logical host owner encoding the already-built table's launch state.
/// It is carried and checkpointed just like the source's `_grid`/`_tiled`.
pub const CACHE_BYTES: usize = 40;
pub fn encode_cache(rows: &[Row], table: Slice) -> Result<[u8; CACHE_BYTES], String> {
    let state = launch(rows)?;
    let table_bytes = u64::try_from(rows.len()).map_err(|_| "copy-plan row count overflow")?
        .checked_mul(128).ok_or("copy-plan table size overflow")?;
    let mut out = [0; CACHE_BYTES];
    out[..4].copy_from_slice(b"ADCP");
    out[4..8].copy_from_slice(&state.grid[0].to_le_bytes());
    out[8..12].copy_from_slice(&state.grid[1].to_le_bytes());
    out[12..16].copy_from_slice(&u32::from(state.tiled).to_le_bytes());
    out[16..24].copy_from_slice(&table_bytes.to_le_bytes());
    out[24..32].copy_from_slice(&table.allocation.to_le_bytes());
    out[32..40].copy_from_slice(&u64::try_from(table.offset)
        .map_err(|_| "copy-plan table offset overflow")?.to_le_bytes());
    Ok(out)
}
pub fn decode_cache(bytes: &[u8]) -> Result<(Launch, usize, Slice), String> {
    if bytes.len() != CACHE_BYTES || &bytes[..4] != b"ADCP" {
        return Err("copy-plan launch cache is unbuilt or invalid".into());
    }
    let word = |i| u32::from_le_bytes(bytes[i..i+4].try_into().unwrap());
    let (x, y, tiled) = (word(4), word(8), word(12));
    let table_bytes = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    if !(1..=1024).contains(&x) || !(1..=65535).contains(&y) || tiled > 1
        || table_bytes != u64::from(y) * 128 {
        return Err("invalid copy-plan cached launch".into());
    }
    let allocation = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
    let offset = usize::try_from(u64::from_le_bytes(bytes[32..40].try_into().unwrap()))
        .map_err(|_| "copy-plan table offset overflow")?;
    Ok((Launch { grid: [x, y, 1], block: [256, 1, 1], tiled: tiled != 0 },
        usize::try_from(table_bytes).map_err(|_| "copy-plan cache size overflow")?,
        Slice { allocation, offset }))
}

/// Recompute the kernel choice and launch geometry from the same descriptors
/// as `CopyPlan.build`; recorded launch dimensions are not frame inputs.
pub fn launch(rows: &[Row]) -> Result<Launch, String> {
    if rows.is_empty() || rows.len() > 65535 { return Err("invalid copy-plan row count".into()); }
    let mut largest = 0_i64;
    let mut tiled = false;
    for row in rows {
        if row.shape.iter().any(|&n| n <= 0) { return Err("invalid copy-plan shape".into()); }
        if !matches!((row.mode, row.src), (0, Some(_)) | (1, None))
            || (row.mode == 1 && row.src_stride != [0; 4]) {
            return Err("invalid copy-plan mode and source".into());
        }
        let count = row.shape.iter().try_fold(1_i64, |n, &d| n.checked_mul(d))
            .ok_or("copy-plan element count overflow")?;
        largest = largest.max(count);
        let mut order = [0, 1, 2, 3];
        order.sort_by(|&a, &b| (row.shape[a] > 1).cmp(&(row.shape[b] > 1))
            .then_with(|| row.dst_stride[b].unsigned_abs().cmp(&row.dst_stride[a].unsigned_abs())));
        tiled |= row.mode == 0 && count >= 1_048_576
            && row.shape[order[0]] == 1 && row.shape[order[1]] == 1
            && row.dst_stride[order[3]] == 1 && row.src_stride[order[2]] == 1
            && row.shape[order[2]] > 1 && row.shape[order[3]] > 1;
    }
    let x = ((largest - 1) / 256 + 1).clamp(1, 1024);
    Ok(Launch { grid: [u32::try_from(x).map_err(|_| "copy-plan grid overflow")?,
                       u32::try_from(rows.len()).map_err(|_| "copy-plan grid overflow")?, 1],
                block: [256, 1, 1], tiled })
}

fn view_address(target: Slice, shape: &[i64; 4], strides: &[i64; 4],
                allocations: &HashMap<u64, Allocation>) -> Result<u64, String> {
    let allocation = allocations.get(&target.allocation)
        .ok_or_else(|| format!("unknown copy-plan target {}", target.allocation))?;
    let mut low = 0_i128;
    let mut high = 0_i128;
    for (&extent, &stride) in shape.iter().zip(strides) {
        let displacement = i128::from(extent - 1).checked_mul(i128::from(stride))
            .and_then(|value| value.checked_mul(4)).ok_or("copy-plan span overflow")?;
        low = low.checked_add(displacement.min(0)).ok_or("copy-plan span overflow")?;
        high = high.checked_add(displacement.max(0)).ok_or("copy-plan span overflow")?;
    }
    let offset = i128::try_from(target.offset).map_err(|_| "copy-plan offset overflow")?;
    let size = i128::try_from(allocation.bytes).map_err(|_| "copy-plan size overflow")?;
    let first = offset.checked_add(low).ok_or("copy-plan span overflow")?;
    let last = offset.checked_add(high).and_then(|value| value.checked_add(4))
        .ok_or("copy-plan span overflow")?;
    if first < 0 || last > size {
        return Err("copy-plan view exceeds its allocation".into());
    }
    let offset = u64::try_from(target.offset).map_err(|_| "copy-plan offset overflow")?;
    allocation.address.checked_add(offset).ok_or_else(|| "copy-plan device address overflow".into())
}

pub fn table(rows: &[Row], allocations: &HashMap<u64, Allocation>) -> Result<Vec<u8>, String> {
    if rows.is_empty() || rows.len() > 65535 { return Err("invalid copy-plan row count".into()); }
    let mut bytes = Vec::with_capacity(rows.len().checked_mul(128).ok_or("copy-plan length overflow")?);
    for row in rows {
        if row.shape.iter().any(|&n| n <= 0) { return Err("invalid copy-plan shape".into()); }
        let count = row.shape.iter().try_fold(1_i64, |n, &d| n.checked_mul(d))
            .ok_or("copy-plan element count overflow")?;
        let dst = view_address(row.dst, &row.shape, &row.dst_stride, allocations)?;
        let src = match (row.mode, row.src) {
            (0, Some(src)) => view_address(src, &row.shape, &row.src_stride, allocations)?,
            (1, None) if row.src_stride == [0; 4] => 0,
            _ => return Err("invalid copy-plan mode and source".into()),
        };
        let mut order = [0, 1, 2, 3];
        order.sort_by(|&a, &b| (row.shape[a] > 1).cmp(&(row.shape[b] > 1))
            .then_with(|| row.dst_stride[b].unsigned_abs().cmp(&row.dst_stride[a].unsigned_abs())));
        bytes.extend_from_slice(&dst.to_le_bytes());
        bytes.extend_from_slice(&src.to_le_bytes());
        for &k in &order { bytes.extend_from_slice(&row.dst_stride[k].to_le_bytes()); }
        for &k in &order { bytes.extend_from_slice(&row.src_stride[k].to_le_bytes()); }
        for &k in &order { bytes.extend_from_slice(&row.shape[k].to_le_bytes()); }
        bytes.extend_from_slice(&count.to_le_bytes());
        bytes.extend_from_slice(&row.fill.to_le_bytes());
        bytes.extend_from_slice(&row.mode.to_le_bytes());
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn broadcast_transpose_is_encoded_from_live_allocation_addresses() {
        let allocations = HashMap::from([(1, Allocation { id: 1, address: 0x1000, bytes: 256 }),
                                         (2, Allocation { id: 2, address: 0x2000, bytes: 64 })]);
        let mut row = Row { dst: Slice { allocation: 1, offset: 0 },
            src: Some(Slice { allocation: 2, offset: 8 }), dst_stride: [0, 0, 1, 3],
            src_stride: [0, 0, 1, 0], shape: [1, 1, 3, 4], fill: 0, mode: 0 };
        let data = table(std::slice::from_ref(&row), &allocations).unwrap();
        assert_eq!(data.len(), 128);
        let word = |n: usize| i64::from_le_bytes(data[n*8..n*8+8].try_into().unwrap());
        assert_eq!((word(0), word(1)), (0x1000, 0x2008));
        assert_eq!((word(2), word(3), word(4), word(5)), (0, 0, 3, 1));
        assert_eq!((word(10), word(11), word(12), word(13), word(14)), (1, 1, 4, 3, 12));
        row.dst.offset = 255;
        assert!(table(&[row], &allocations).is_err());
    }

    #[test]
    fn launch_geometry_and_tiled_branch_follow_live_descriptors() {
        let mut row = Row { dst: Slice { allocation: 1, offset: 0 },
            src: Some(Slice { allocation: 2, offset: 0 }), dst_stride: [0, 0, 1024, 1],
            src_stride: [0, 0, 1, 1024], shape: [1, 1, 1024, 1024], fill: 0, mode: 0 };
        let spec = launch(std::slice::from_ref(&row)).unwrap();
        assert_eq!((spec.grid, spec.block, spec.tiled), ([1024, 1, 1], [256, 1, 1], true));
        row.shape[2] = 10;
        let spec = launch(std::slice::from_ref(&row)).unwrap();
        assert_eq!((spec.grid, spec.tiled), ([40, 1, 1], false));
        row.mode = 1;
        assert!(launch(&[row]).is_err());
    }
    #[test]
    fn cached_launch_survives_changes_to_input_descriptors() {
        let mut row = Row { dst: Slice { allocation: 1, offset: 0 },
            src: Some(Slice { allocation: 2, offset: 0 }), dst_stride: [0, 0, 1024, 1],
            src_stride: [0, 0, 1, 1024], shape: [1, 1, 1024, 1024], fill: 0, mode: 0 };
        let target = Slice { allocation: 7, offset: 16 };
        let snapshot = encode_cache(std::slice::from_ref(&row), target).unwrap();
        row.shape[2] = 10;
        let (cached, bytes, table) = decode_cache(&snapshot).unwrap();
        assert_eq!((cached.grid, cached.tiled, bytes), ([1024, 1, 1], true, 128));
        assert_eq!((table.allocation, table.offset), (7, 16));
        assert_ne!(cached, launch(&[row]).unwrap());
        assert!(decode_cache(&[0; CACHE_BYTES]).is_err());
        let mut corrupt = snapshot;
        corrupt[16] ^= 1;
        assert!(decode_cache(&corrupt).is_err());
    }
}
