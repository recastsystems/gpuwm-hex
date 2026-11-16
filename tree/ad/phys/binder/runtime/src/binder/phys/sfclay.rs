// SPDX-License-Identifier: Apache-2.0
//! Frame values for the pinned resident MYNN-SFC driver.
//!
//! Source: hexcore/mpasphys/rrfs/resident_sfclay_rrfs.py:36-58, 121-125
//! and sf_mynnsfc_v600.py:14. This rule alone is not a replay proof.
use serde::Deserialize;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub restart: bool,
    pub cycling: bool,
    pub lsm_scheme: String,
    pub sfcflux_water: i32,
    pub sfcflux_land: i32,
}

pub fn resident_options(config: &Config) -> Vec<u8> {
    let lsm = match config.lsm_scheme.as_str() {
        "sf_noaa" => 1, "sf_noahmp" => 2, "sf_ruc" => 3, _ => 4,
    };
    let words: [i32; 14] = [
        0, 0, 0, 0, config.sfcflux_water, config.sfcflux_land,
        0, 1, i32::from(config.restart), i32::from(config.cycling),
        0, 1, lsm, 0,
    ];
    words.into_iter().flat_map(i32::to_le_bytes).collect()
}

pub fn initflag(config: &Config, step: u64) -> Result<i32, String> {
    if step == 0 { return Err("invalid MYNN-SFC step".into()); }
    Ok(i32::from(!(config.restart || config.cycling || step > 1)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn matches_observed_hrrrv5_option_upload_and_live_flags() {
        let mut config = Config { restart: false, cycling: false, lsm_scheme: "sf_ruc".into(),
                                  sfcflux_water: 0, sfcflux_land: 1 };
        let bytes = resident_options(&config);
        // Independent observed upload SHA-256, never an input to replay.
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)),
                   "c74cbac65d1cb2710262972de6ffd3c42b52e5bd4d7ba9905568c11482e1f16f");
        assert_eq!(bytes.len(), 56);
        assert_eq!(initflag(&config, 1).unwrap(), 1);
        assert_eq!(initflag(&config, 2).unwrap(), 0);
        assert!(initflag(&config, 0).is_err());
        config.cycling = true;
        assert_eq!(initflag(&config, 1).unwrap(), 0);
        config.cycling = false;
        config.restart = true;
        assert_eq!(initflag(&config, 1).unwrap(), 0);
        config.lsm_scheme = "other".into();
        assert_eq!(i32::from_le_bytes(resident_options(&config)[48..52].try_into().unwrap()), 4);
    }
}
