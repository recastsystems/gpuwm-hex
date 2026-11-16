// SPDX-License-Identifier: Apache-2.0
//! Source-selected scalar casts from current host storage, never trace values.
//!
//! Ownership, publication, selected control flow and scalar provenance are
//! obligations of the plan producer. Unsupported conversions fail closed.
use serde::Deserialize;

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Cast {
    I32, I64, U32, U64, F32, F64,
    I64ToI32, U64ToU32, F64ToF32, BoolToI32,
}

impl Cast {
    pub fn source_bytes(self) -> usize {
        match self {
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::BoolToI32 => 1,
            _ => 8,
        }
    }
}

pub fn evaluate(cast: Cast, source: &[u8]) -> Result<Vec<u8>, String> {
    if source.len() != cast.source_bytes() { return Err("host scalar source width differs".into()); }
    match cast {
        Cast::I32 | Cast::I64 | Cast::U32 | Cast::U64 | Cast::F32 | Cast::F64 => Ok(source.to_vec()),
        Cast::I64ToI32 => {
            let n = i64::from_le_bytes(source.try_into().map_err(|_| "invalid host integer")?);
            let v = i32::try_from(n).map_err(|_| "host integer does not fit i32")?;
            Ok(v.to_le_bytes().to_vec())
        },
        Cast::U64ToU32 => {
            let n = u64::from_le_bytes(source.try_into().map_err(|_| "invalid host integer")?);
            let v = u32::try_from(n).map_err(|_| "host integer does not fit u32")?;
            Ok(v.to_le_bytes().to_vec())
        },
        Cast::F64ToF32 => {
            let n = f64::from_le_bytes(source.try_into().map_err(|_| "invalid host float")?);
            Ok((n as f32).to_le_bytes().to_vec())
        },
        Cast::BoolToI32 => match source[0] {
            0 => Ok(0_i32.to_le_bytes().to_vec()),
            1 => Ok(1_i32.to_le_bytes().to_vec()),
            _ => Err("invalid host boolean".into()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checks_width_bounds_and_boolean_representation() {
        assert_eq!(evaluate(Cast::I64ToI32, &(-37_i64).to_le_bytes()).unwrap(), (-37_i32).to_le_bytes());
        assert!(evaluate(Cast::I64ToI32, &(i64::from(i32::MAX) + 1).to_le_bytes()).is_err());
        assert!(evaluate(Cast::U64ToU32, &(u64::from(u32::MAX) + 1).to_le_bytes()).is_err());
        assert_eq!(evaluate(Cast::BoolToI32, &[1]).unwrap(), 1_i32.to_le_bytes());
        assert!(evaluate(Cast::BoolToI32, &[2]).is_err());
        assert!(evaluate(Cast::F32, &[0, 0]).is_err());
    }
}
