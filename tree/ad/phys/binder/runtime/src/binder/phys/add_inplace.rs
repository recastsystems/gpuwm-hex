// SPDX-License-Identifier: Apache-2.0
//! Frame for cuda_driver.py's live add-in-place call.
//! The producer must independently prove view ownership and call selection.

pub struct Launch { pub grid: [u32; 3], pub block: [u32; 3], pub bytes: usize,
                    pub nlevels: i32, pub nowners: i32 }

pub fn prepare(target: [i64; 2], increment: [i64; 2]) -> Result<Launch, String> {
    if target != increment { return Err("add-inplace shape mismatch".into()); }
    if target.iter().any(|&n| n < 1) { return Err("add-inplace shape must be nonempty".into()); }
    let nlevels = i32::try_from(target[0]).map_err(|_| "add-inplace levels exceed int32")?;
    let nowners = i32::try_from(target[1]).map_err(|_| "add-inplace owners exceed int32")?;
    // The archived kernel multiplies these dimensions as signed C ints.
    let count = nlevels.checked_mul(nowners).ok_or("add-inplace count exceeds int32")?;
    let bytes = usize::try_from(count).map_err(|_| "add-inplace count overflow")?
        .checked_mul(4).ok_or("add-inplace byte count overflow")?;
    let x = u32::try_from((i64::from(count) + 127) / 128).map_err(|_| "add-inplace grid overflow")?;
    Ok(Launch { grid:[x,1,1], block:[128,1,1], bytes, nlevels, nowners })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uses_both_current_shapes_for_frame_and_geometry() {
        let x=prepare([60,2562],[60,2562]).unwrap();
        assert_eq!((x.nlevels,x.nowners,x.bytes,x.grid,x.block),(60,2562,614880,[1201,1,1],[128,1,1]));
        assert!(prepare([60,2562],[59,2562]).is_err());
        assert!(prepare([0,3],[0,3]).is_err());
        assert!(prepare([i32::MAX as i64,2],[i32::MAX as i64,2]).is_err());
    }
}
