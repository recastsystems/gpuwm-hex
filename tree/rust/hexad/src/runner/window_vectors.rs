// SPDX-License-Identifier: Apache-2.0
//! Explicit, package-bound vectors for a comparison across allocation layouts.
//! These change the test inputs only. They do not change any model operation.
use std::{collections::BTreeMap, fs, path::Path};
use serde_json::Value;

pub struct Segment { pub allocation: u64, pub offset: usize, pub data: Vec<u8> }
pub struct Pair { pub seed: u64, pub x: Vec<Segment>, pub y: Vec<Segment>, pub sha256: String }

impl Pair {
    pub fn load(path: &Path, plan_sha256: &str, sizes: &BTreeMap<u64, usize>) -> Result<Self, String> {
        let raw=fs::read(path).map_err(|e|format!("vector gate {}: {e}",path.display()))?;
        let doc: Value=serde_json::from_slice(&raw).map_err(|e|e.to_string())?;
        if doc["schema"]!="ad.window-vectors/v1" || doc["plan_sha256"]!=plan_sha256 {
            return Err("DEFECT: vector gate schema or package hash mismatch".into());
        }
        let seed=doc["seed"].as_u64().ok_or("vector gate seed")?;
        let base=path.parent().unwrap_or(Path::new("."));
        let read=|side:&str| -> Result<Vec<Segment>,String> {
            let mut segments=Vec::new();
            for row in doc[side].as_array().ok_or_else(||format!("vector gate {side} segments"))? {
                let allocation=row["allocation"].as_u64().ok_or("vector allocation")?;
                let offset=row["offset"].as_u64().ok_or("vector offset")? as usize;
                let file=base.join(row["file"].as_str().ok_or("vector file")?);
                let data=fs::read(&file).map_err(|e|format!("{}: {e}",file.display()))?;
                let want=row["sha256"].as_str().ok_or("vector sha256")?;
                if crate::exec::digest(&data)!=want {return Err(format!("DEFECT: vector hash mismatch {}",file.display()));}
                let size=*sizes.get(&allocation).ok_or("vector allocation has no shadow")?;
                if offset%4!=0 || data.len()%4!=0 || data.is_empty() || offset.checked_add(data.len()).map_or(true,|n|n>size) {
                    return Err(format!("DEFECT: vector range outside allocation {allocation}"));
                }
                segments.push(Segment{allocation,offset,data});
            }
            segments.sort_by_key(|s|(s.allocation,s.offset));
            for p in segments.windows(2) {
                if p[0].allocation==p[1].allocation && p[0].offset+p[0].data.len()>p[1].offset {
                    return Err(format!("DEFECT: overlapping {side} vector segments"));
                }
            }
            Ok(segments)
        };
        Ok(Self{seed,x:read("x")?,y:read("y")?,sha256:crate::exec::digest(&raw)})
    }
    pub fn side(&self, seed:u64) -> Result<&[Segment],String> {
        if seed==0x5eed_0000_0000+2*self.seed+1 {Ok(&self.x)}
        else if seed==0x5eed_0000_0000+2*self.seed+2 {Ok(&self.y)}
        else {Err("DEFECT: vector gate requested a different random seed".into())}
    }
}
