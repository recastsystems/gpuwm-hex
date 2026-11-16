// SPDX-License-Identifier: Apache-2.0
//! Live arguments for the pinned RRFS post-transport copy.
//! Producers still prove owner, dtype, selected rows, and current metadata lifetime.
use super::{plan::Slice, submit::{Allocation, Argument}};
use std::collections::HashMap;

pub struct View { pub target: Option<Slice>, pub shape: Vec<i64>, pub strides: Vec<i64> }
pub struct Row { pub dst: View, pub src: View }
pub struct Launch { pub grid: [u32; 3], pub block: [u32; 3], pub arguments: Vec<Argument> }

fn view(v: &View, allocations: &HashMap<u64, Allocation>) -> Result<(i64, Argument), String> {
    if v.shape.len() > 4 || v.shape.len() != v.strides.len() || v.shape.iter().any(|&n| n < 0)
        || v.strides.iter().any(|s| s % 4 != 0) {
        return Err("invalid transport copy view".into());
    }
    let count = v.shape.iter().try_fold(1_i64, |a, &b| a.checked_mul(b))
        .ok_or("transport copy element count overflow")?;
    let Some(target) = v.target else {
        if count == 0 { return Ok((0, Argument::Bytes(0_u64.to_le_bytes().to_vec()))); }
        return Err("nonempty transport copy view has null pointer".into());
    };
    let a = allocations.get(&target.allocation).ok_or("unknown transport copy allocation")?;
    let offset = i128::try_from(target.offset).map_err(|_| "transport copy offset overflow")?;
    let address = a.address.checked_add(u64::try_from(target.offset).map_err(|_| "transport copy offset overflow")?)
        .ok_or("transport copy address overflow")?;
    if address % 4 != 0 || target.offset >= a.bytes {
        return Err("transport copy view exceeds or misaligns allocation".into());
    }
    if count != 0 {
        let mut low = 0_i128;
        let mut high = 0_i128;
        for (&extent, &stride) in v.shape.iter().zip(&v.strides) {
            let delta = i128::from(extent - 1).checked_mul(i128::from(stride))
                .ok_or("transport copy span overflow")?;
            low = low.checked_add(delta.min(0)).ok_or("transport copy span overflow")?;
            high = high.checked_add(delta.max(0)).ok_or("transport copy span overflow")?;
        }
        let start = offset.checked_add(low).ok_or("transport copy span overflow")?;
        let end = offset.checked_add(high).and_then(|n| n.checked_add(4)).ok_or("transport copy span overflow")?;
        if start < 0 || end > a.bytes as i128 {
            return Err("transport copy view exceeds or misaligns allocation".into());
        }
    }
    Ok((count, Argument::Device(super::submit::Slice { allocation: target.allocation, offset: target.offset })))
}

pub fn prepare(rows: &[Row], allocations: &HashMap<u64, Allocation>) -> Result<Launch, String> {
    if rows.len() != 12 { return Err("transport copy needs twelve live rows".into()); }
    let mut largest = 0_i64;
    let mut arguments = Vec::with_capacity(12 * 14);
    for row in rows {
        if row.dst.shape != row.src.shape { return Err("transport copy shape mismatch".into()); }
        let (count, dst) = view(&row.dst, allocations)?;
        let (_, src) = view(&row.src, allocations)?;
        largest = largest.max(count);
        let pad = 4 - row.dst.shape.len();
        let mut shape = [1_i64; 4];
        let mut ds = [0_i64; 4];
        let mut ss = [0_i64; 4];
        shape[pad..].copy_from_slice(&row.dst.shape);
        for (i, &stride) in row.dst.strides.iter().enumerate() { ds[pad + i] = stride / 4; }
        for (i, &stride) in row.src.strides.iter().enumerate() { ss[pad + i] = stride / 4; }
        let mut order = [0_usize, 1, 2, 3];
        order.sort_by_key(|&k| (shape[k] > 1, -i128::from(ds[k]).abs()));
        arguments.push(dst);
        arguments.push(src);
        let metadata = std::iter::once(count).chain(order[1..].iter().map(|&k| shape[k]))
            .chain(order.iter().map(|&k| ds[k])).chain(order.iter().map(|&k| ss[k]));
        arguments.extend(metadata.map(|x| Argument::Bytes(x.to_le_bytes().to_vec())));
    }
    let grid_x = if largest == 0 { 1 } else { ((largest - 1) / 256 + 1).min(1024) as u32 };
    Ok(Launch { grid: [grid_x, 12, 1], block: [256, 1, 1], arguments })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn v(id:u64, off:usize, shape:&[i64], strides:&[i64]) -> View {
        View { target:Some(Slice { allocation:id, offset:off }), shape:shape.to_vec(), strides:strides.to_vec() }
    }
    fn rows() -> Vec<Row> { (0..12).map(|_| Row {
        dst:v(1,0,&[2,3],&[12,4]), src:v(2,0,&[2,3],&[4,8]) }).collect() }
    #[test]
    fn derives_live_stable_traversal_and_geometry() {
        let a=HashMap::from([(1,Allocation{id:1,address:0x1000,bytes:24}),
                             (2,Allocation{id:2,address:0x2000,bytes:24})]);
        let l=prepare(&rows(),&a).unwrap();
        assert_eq!((l.grid,l.block,l.arguments.len()),([1,12,1],[256,1,1],168));
        let words:Vec<_>=l.arguments[2..14].iter().map(|x| match x {
            Argument::Bytes(b)=>i64::from_le_bytes(b.as_slice().try_into().unwrap()), _=>panic!() }).collect();
        assert_eq!(words,vec![6,1,2,3,0,0,3,1,0,0,1,2]);
    }
    #[test]
    fn validates_shape_span_roster_and_strides() {
        let a=HashMap::from([(1,Allocation{id:1,address:0x1000,bytes:24}),
                             (2,Allocation{id:2,address:0x2000,bytes:24})]);
        let mut r=rows(); r[0].src.shape[1]=4; assert!(prepare(&r,&a).is_err());
        let mut r=rows(); r[0].dst.strides[0]=100; assert!(prepare(&r,&a).is_err());
        let mut r=rows(); r[0].dst.strides[0]=3; assert!(prepare(&r,&a).is_err());
        let mut r=rows(); r.pop(); assert!(prepare(&r,&a).is_err());
    }
}
