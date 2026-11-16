// SPDX-License-Identifier: Apache-2.0
//! Host launch frame for `vinterp_ozn_device` in the pinned hex source.
//!
//! Values here come from the current live view shapes, never from a trace.
//! The plan producer must still prove the views, their contiguous storage and
//! the surrounding initialisation/copies for this particular call.

#[derive(Debug, PartialEq, Eq)]
pub struct Frame {
    pub nc: i32,
    pub nl: i32,
    pub nz: i32,
    pub grid: [u32; 3],
    /// Required bytes for pmid, pin, ozone, starts, upper and out.
    pub bytes: [usize; 6],
}

pub fn prepare(pmid: [i64; 2], pin: i64, ozone: [i64; 2]) -> Result<Option<Frame>, String> {
    let [nc, nl] = pmid;
    if nc < 0 || nl < 0 || pin < 0 || ozone != [nc, pin] {
        return Err("vinterp view shape mismatch".into());
    }
    // The source returns before creating the level-chain launches for an
    // empty pmid, after checking the pin and ozone shapes.
    if nc == 0 || nl == 0 { return Ok(None); }
    if pin == 0 { return Err("vinterp empty pressure levels are unsafe to launch".into()); }
    let nc = i32::try_from(nc).map_err(|_| "vinterp nc exceeds int32")?;
    let nl = i32::try_from(nl).map_err(|_| "vinterp nl exceeds int32")?;
    let nz = i32::try_from(pin).map_err(|_| "vinterp nz exceeds int32")?;
    let (c, l, z) = (nc as usize, nl as usize, nz as usize);
    let words = |a: usize, b: usize| a.checked_mul(b).and_then(|n| n.checked_mul(4))
        .ok_or_else(|| "vinterp array extent overflow".to_owned());
    Ok(Some(Frame {
        nc, nl, nz, grid: [((nc as u32) + 127) / 128, 1, 1],
        bytes: [words(c,l)?, words(z,1)?, words(c,z)?, words(l,1)?, words(c,1)?, words(c,l)?],
    }))
}

impl Frame {
    pub fn scalar_bytes(&self, k: i32) -> Result<[[u8; 4]; 4], String> {
        if k < 0 || k >= self.nl { return Err("vinterp level outside live shape".into()); }
        Ok([self.nc.to_le_bytes(), self.nl.to_le_bytes(), self.nz.to_le_bytes(), k.to_le_bytes()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn live_shapes_define_all_dimensions_geometry_and_descending_levels() {
        let frame = prepare([2562, 59], 59, [2562, 59]).unwrap().unwrap();
        assert_eq!(frame.grid, [21, 1, 1]);
        assert_eq!(frame.bytes, [604632,236,604632,236,10248,604632]);
        let levels: Vec<i32> = (0..frame.nl).rev().collect();
        assert_eq!(levels.first(), Some(&58));
        assert_eq!(levels.last(), Some(&0));
        assert_eq!(frame.scalar_bytes(58).unwrap()[0], 2562_i32.to_le_bytes());
        assert!(frame.scalar_bytes(59).is_err());
        assert!(frame.scalar_bytes(-1).is_err());
    }
    #[test]
    fn empty_and_invalid_shapes_fail_closed() {
        assert_eq!(prepare([0, 5], 3, [0, 3]).unwrap(), None);
        assert_eq!(prepare([5, 0], 3, [5, 3]).unwrap(), None);
        assert!(prepare([5, 3], 2, [5, 3]).is_err());
        assert!(prepare([5, 3], 0, [5, 0]).is_err());
        assert!(prepare([-1, 3], 2, [-1, 2]).is_err());
        assert!(prepare([i64::from(i32::MAX) + 1, 1], 1, [i64::from(i32::MAX) + 1, 1]).is_err());
    }
}
