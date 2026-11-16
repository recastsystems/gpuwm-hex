// SPDX-License-Identifier: Apache-2.0
//! Byte-preserving host views for live physics pool storage.
//!
//! The underlying storage must come from the current replay state. A captured
//! upload or a previously observed host result is not a storage provider.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct View {
    pub offset: i64,
    pub shape: Vec<usize>,
    pub strides: Vec<i64>,
    pub item_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order { C, Fortran }

/// Validate a view without visiting each element, and return its packed size.
/// This checks bounds only; publication additionally checks for aliasing.
pub fn packed_size(storage_bytes: usize, view: &View) -> Result<usize, &'static str> {
    if view.item_bytes == 0 || view.shape.len() != view.strides.len() {
        return Err("invalid host view descriptor");
    }
    let elements = view.shape.iter().try_fold(1_usize, |n, &d| n.checked_mul(d))
        .ok_or("host view element count overflow")?;
    let bytes = elements.checked_mul(view.item_bytes).ok_or("host view size overflow")?;
    if elements == 0 { return Ok(0); }
    let (mut lower, mut upper) = (i128::from(view.offset), i128::from(view.offset));
    for (&extent, &stride) in view.shape.iter().zip(&view.strides) {
        let delta = i128::try_from(extent - 1).map_err(|_| "host view extent overflow")?
            .checked_mul(i128::from(stride)).ok_or("host view address overflow")?;
        lower = lower.checked_add(delta.min(0)).ok_or("host view address overflow")?;
        upper = upper.checked_add(delta.max(0)).ok_or("host view address overflow")?;
    }
    let end = upper.checked_add(i128::try_from(view.item_bytes).map_err(|_| "host item size overflow")?)
        .ok_or("host view address overflow")?;
    if lower < 0 || end > i128::try_from(storage_bytes).map_err(|_| "host storage size overflow")? {
        return Err("host view exceeds storage");
    }
    Ok(bytes)
}

/// The views selected by device_pools_v842.py:prepared, including the
/// [:count, ..., 0] slice and the reversed profile axes. `lower` is the live
/// MPAS variable's lower-bound vector and `count` is the live solve extent.
pub fn prepared_view(host: &View, lower: &[i64], count: usize) -> Result<View, &'static str> {
    let ndim = host.shape.len();
    if !(2..=4).contains(&ndim) || host.strides.len() != ndim || lower.len() != ndim
        || lower[0] != 1 || host.item_bytes == 0 {
        return Err("invalid prepared host descriptor");
    }
    let j_axis = if ndim == 4 { 2 } else { ndim - 1 };
    if host.shape[j_axis] != 1 { return Err("prepared host j dimension must be one"); }
    let n = count.min(host.shape[0]);
    let axes: &[usize] = match ndim { 2 => &[0], 3 => &[1, 0], 4 => &[3, 1, 0], _ => unreachable!() };
    let shape = axes.iter().map(|&axis| if axis == 0 { n } else { host.shape[axis] }).collect();
    let strides = axes.iter().map(|&axis| host.strides[axis]).collect();
    Ok(View { offset: host.offset, shape, strides, item_bytes: host.item_bytes })
}

/// Copy the current view in C or Fortran element order without interpreting
/// floating-point values, preserving NaNs and all other bit patterns.
pub fn pack(storage: &[u8], view: &View, order: Order) -> Result<Vec<u8>, &'static str> {
    if view.item_bytes == 0 || view.shape.len() != view.strides.len() {
        return Err("invalid host view descriptor");
    }
    let elements = view.shape.iter().try_fold(1_usize, |n, &d| n.checked_mul(d))
        .ok_or("host view element count overflow")?;
    let bytes = elements.checked_mul(view.item_bytes).ok_or("host view size overflow")?;
    let mut result = Vec::with_capacity(bytes);
    for linear in 0..elements {
        let mut rem = linear;
        let mut address = i128::from(view.offset);
        for dimension in 0..view.shape.len() {
            let axis = match order { Order::C => view.shape.len() - 1 - dimension, Order::Fortran => dimension };
            let coordinate = rem % view.shape[axis];
            rem /= view.shape[axis];
            address = address.checked_add((coordinate as i128) * i128::from(view.strides[axis]))
                .ok_or("host view address overflow")?;
        }
        let start = usize::try_from(address).map_err(|_| "negative host view address")?;
        let end = start.checked_add(view.item_bytes).ok_or("host view address overflow")?;
        result.extend_from_slice(storage.get(start..end).ok_or("host view exceeds storage")?);
    }
    Ok(result)
}

/// Publish a freshly downloaded live device value into its host view. `order`
/// is the memory order of the downloaded value. Every destination is checked
/// before writing; overlapping element views are refused until their NumPy
/// assignment order has independently been established.
pub fn unpack(storage: &mut [u8], view: &View, order: Order, input: &[u8]) -> Result<(), &'static str> {
    if view.item_bytes == 0 || view.shape.len() != view.strides.len() {
        return Err("invalid host view descriptor");
    }
    let elements = view.shape.iter().try_fold(1_usize, |n, &d| n.checked_mul(d))
        .ok_or("host view element count overflow")?;
    let bytes = elements.checked_mul(view.item_bytes).ok_or("host view size overflow")?;
    if input.len() != bytes { return Err("download size differs from host view"); }
    let mut destinations = Vec::with_capacity(elements);
    for linear in 0..elements {
        let mut rem = linear;
        let mut address = i128::from(view.offset);
        for dimension in 0..view.shape.len() {
            let axis = match order { Order::C => view.shape.len() - 1 - dimension, Order::Fortran => dimension };
            let coordinate = rem % view.shape[axis];
            rem /= view.shape[axis];
            address = address.checked_add((coordinate as i128) * i128::from(view.strides[axis]))
                .ok_or("host view address overflow")?;
        }
        let start = usize::try_from(address).map_err(|_| "negative host view address")?;
        let end = start.checked_add(view.item_bytes).ok_or("host view address overflow")?;
        if end > storage.len() { return Err("host view exceeds storage"); }
        destinations.push((start, linear));
    }
    destinations.sort_unstable_by_key(|&(start, _)| start);
    if destinations.windows(2).any(|pair| pair[0].0 + view.item_bytes > pair[1].0) {
        return Err("overlapping host view elements");
    }
    for (start, linear) in destinations {
        let source = linear * view.item_bytes;
        storage[start..start + view.item_bytes].copy_from_slice(&input[source..source + view.item_bytes]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packs_strided_and_reversed_views_in_both_orders() {
        let bytes: Vec<u8> = (0..24).collect();
        let view = View { offset: 10, shape: vec![2, 3], strides: vec![8, -2], item_bytes: 2 };
        assert_eq!(pack(&bytes, &view, Order::C).unwrap(), [10,11,8,9,6,7,18,19,16,17,14,15]);
        assert_eq!(pack(&bytes, &view, Order::Fortran).unwrap(), [10,11,18,19,8,9,16,17,6,7,14,15]);
    }
    #[test]
    fn prepared_axes_follow_the_live_source_layout() {
        let host = View { offset: 0, shape: vec![4, 3, 1, 2], strides: vec![1, 4, 12, 12], item_bytes: 1 };
        let view = prepared_view(&host, &[1,1,1,1], 2).unwrap();
        assert_eq!(view.shape, [2,3,2]);
        assert_eq!(view.strides, [12,4,1]);
        let data: Vec<u8> = (0..24).collect();
        assert_eq!(pack(&data, &view, Order::C).unwrap(), [0,1,4,5,8,9,12,13,16,17,20,21]);
        assert!(prepared_view(&host, &[0,1,1,1], 2).is_err());
    }
    #[test]
    fn invalid_extents_fail_closed_and_empty_views_are_empty() {
        let view = View { offset: 3, shape: vec![2], strides: vec![2], item_bytes: 2 };
        assert!(pack(&[0;4], &view, Order::C).is_err());
        let empty = View { offset: 0, shape: vec![3,0], strides: vec![5,1], item_bytes: 4 };
        assert!(pack(&[], &empty, Order::C).unwrap().is_empty());
        let scalar = View { offset: 1, shape: vec![], strides: vec![], item_bytes: 2 };
        assert_eq!(pack(&[1,2,3], &scalar, Order::C).unwrap(), [2,3]);
    }
    #[test]
    fn publishes_without_touching_other_words_and_refuses_overlap_atomically() {
        let view = View { offset: 10, shape: vec![2, 3], strides: vec![8, -2], item_bytes: 2 };
        let input: Vec<u8> = (40..52).collect();
        let mut data = vec![9; 24];
        unpack(&mut data, &view, Order::Fortran, &input).unwrap();
        assert_eq!(pack(&data, &view, Order::Fortran).unwrap(), input);
        assert_eq!(data[0..6], [9; 6]);
        let before = data.clone();
        let alias = View { offset: 0, shape: vec![2], strides: vec![1], item_bytes: 2 };
        assert!(unpack(&mut data, &alias, Order::C, &[1,2,3,4]).is_err());
        assert_eq!(data, before);
        assert!(unpack(&mut data, &view, Order::C, &[1,2]).is_err());
        assert_eq!(data, before);
    }
}
