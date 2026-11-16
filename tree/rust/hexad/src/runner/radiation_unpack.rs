// SPDX-License-Identifier: Apache-2.0
//! Live frame for the pinned RRFS radiation unpack kernel.
//! View owners, shapes, fill inputs and call selection need independent proof.
use super::{plan::Slice, submit::{Allocation, Argument}};
use std::collections::HashMap;

const ROSTER: usize = 23;
pub struct View { pub target: Option<Slice>, pub shape: Vec<i64>, pub strides: Vec<i64> }
pub struct Row { pub dst: View, pub src: Option<View>, pub fill: u32 }
pub struct Launch { pub grid: [u32; 3], pub block: [u32; 3], pub arguments: Vec<Argument> }

fn view(view: &View, allocations: &HashMap<u64, Allocation>) -> Result<(i64, i64, [i64; 2], Argument), String> {
    if !(1..=2).contains(&view.shape.len()) || view.shape.len() != view.strides.len()
        || view.shape.iter().any(|&n| n < 0) || view.strides.iter().any(|s| s % 4 != 0) {
        return Err("invalid radiation unpack view".into());
    }
    let n = view.shape.iter().try_fold(1_i64, |a, &b| a.checked_mul(b))
        .ok_or("radiation unpack element count overflow")?;
    let width = *view.shape.last().ok_or("missing radiation unpack width")?;
    let strides = if view.shape.len() == 1 { [0, view.strides[0] / 4] }
                  else { [view.strides[0] / 4, view.strides[1] / 4] };
    if n == 0 { return Ok((n, width, strides, Argument::Bytes(0_u64.to_le_bytes().to_vec()))); }
    let target = view.target.ok_or("nonempty radiation unpack view has null pointer")?;
    let allocation = allocations.get(&target.allocation).ok_or("unknown radiation unpack allocation")?;
    let offset = i128::try_from(target.offset).map_err(|_| "radiation unpack offset overflow")?;
    let mut low = 0_i128;
    let mut high = 0_i128;
    for (&extent, &stride) in view.shape.iter().zip(&view.strides) {
        let delta = i128::from(extent - 1).checked_mul(i128::from(stride)).ok_or("radiation unpack span overflow")?;
        low = low.checked_add(delta.min(0)).ok_or("radiation unpack span overflow")?;
        high = high.checked_add(delta.max(0)).ok_or("radiation unpack span overflow")?;
    }
    let start = offset.checked_add(low).ok_or("radiation unpack span overflow")?;
    let end = offset.checked_add(high).and_then(|n| n.checked_add(4)).ok_or("radiation unpack span overflow")?;
    let address = allocation.address.checked_add(u64::try_from(target.offset).map_err(|_| "offset overflow")?)
        .ok_or("radiation unpack address overflow")?;
    if start < 0 || end > allocation.bytes as i128 || address % 4 != 0 {
        return Err("radiation unpack view exceeds or misaligns allocation".into());
    }
    Ok((n, width, strides, Argument::Device(super::submit::Slice {
        allocation: target.allocation, offset: target.offset })))
}

pub fn prepare(rows: &[Row], allocations: &HashMap<u64, Allocation>) -> Result<Option<Launch>, String> {
    if rows.len() > ROSTER { return Err("radiation unpack exceeds argument roster".into()); }
    let mut largest = 0_i64;
    let mut arguments = Vec::with_capacity(ROSTER * 9);
    for row in rows {
        let (n, width, ds, dst) = view(&row.dst, allocations)?;
        let (src, ss) = if let Some(src) = &row.src {
            if src.shape != row.dst.shape || row.fill != 0 { return Err("radiation unpack copy mismatch".into()); }
            let (_, _, strides, pointer) = view(src, allocations)?;
            (pointer, strides)
        } else { (Argument::Bytes(0_u64.to_le_bytes().to_vec()), [0,0]) };
        largest = largest.max(n);
        arguments.push(dst);
        arguments.push(src);
        arguments.extend([ds[0],ds[1],ss[0],ss[1],n,width].into_iter()
            .map(|value| Argument::Bytes(value.to_le_bytes().to_vec())));
        arguments.push(Argument::Bytes(row.fill.to_le_bytes().to_vec()));
    }
    for _ in rows.len()..ROSTER {
        arguments.extend((0..8).map(|_| Argument::Bytes(0_u64.to_le_bytes().to_vec())));
        arguments.push(Argument::Bytes(0_u32.to_le_bytes().to_vec()));
    }
    if largest == 0 { return Ok(None); }
    let x = u32::try_from((largest - 1) / 256 + 1).map_err(|_| "radiation unpack grid overflow")?;
    Ok(Some(Launch { grid:[x,rows.len() as u32,1], block:[256,1,1], arguments }))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn allocation(id: u64, address: u64, bytes: usize) -> Allocation { Allocation { id, address, bytes } }
    fn v(id: u64, offset: usize, shape: &[i64], strides: &[i64]) -> View {
        View { target:Some(Slice { allocation:id, offset }), shape:shape.to_vec(), strides:strides.to_vec() }
    }
    #[test]
    fn copies_and_live_fills_use_view_metadata_and_pad_the_roster() {
        let a = HashMap::from([(1, allocation(1,0x1000,128)),(2, allocation(2,0x2000,128))]);
        let rows = [Row { dst:v(1,0,&[3,2],&[8,4]), src:Some(v(2,0,&[3,2],&[4,12])), fill:0 },
                    Row { dst:v(1,32,&[3],&[4]), src:None, fill:1.5_f32.to_bits() }];
        let launch = prepare(&rows,&a).unwrap().unwrap();
        assert_eq!((launch.grid,launch.block,launch.arguments.len()),([1,2,1],[256,1,1],207));
        assert!(matches!(&launch.arguments[2], Argument::Bytes(b) if b == &2_i64.to_le_bytes()));
        assert!(matches!(&launch.arguments[3], Argument::Bytes(b) if b == &1_i64.to_le_bytes()));
        assert!(matches!(&launch.arguments[17], Argument::Bytes(b) if b == &1.5_f32.to_le_bytes()));
    }
    #[test]
    fn validates_bounds_shapes_and_empty_rows_before_launch() {
        let a = HashMap::from([(1, allocation(1,0x1000,16))]);
        let empty = Row { dst:v(1,0,&[0],&[4]), src:None, fill:0 };
        assert!(prepare(&[empty],&a).unwrap().is_none());
        let bad = Row { dst:v(1,12,&[2],&[4]), src:None, fill:0 };
        assert!(prepare(&[bad],&a).is_err());
        let bad = Row { dst:v(1,0,&[2],&[3]), src:None, fill:0 };
        assert!(prepare(&[bad],&a).is_err());
        let bad = Row { dst:v(1,0,&[2],&[4]), src:Some(v(1,0,&[1],&[4])), fill:0 };
        assert!(prepare(&[bad],&a).is_err());
    }
}
