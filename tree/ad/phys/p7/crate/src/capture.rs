// SPDX-License-Identifier: Apache-2.0
//! Read an ad.hex.launch/v1 capture: arguments, complete backing allocations
//! before and after the launch, the original image and source.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub struct Launch {
    pub dir: PathBuf,
    pub manifest: Value,
    pub pre: Vec<Vec<u8>>,
    pub post: Vec<Vec<u8>>,
    pub image: Vec<u8>,
}

fn hex(b: &[u8]) -> String { b.iter().map(|x| format!("{x:02x}")).collect() }

fn object(dir: &Path, o: &Value) -> Result<Vec<u8>, String> {
    let data = if o.get("encoding").and_then(Value::as_str) == Some("raw-concat-v1") {
        let mut v = Vec::new();
        for c in o["chunks"].as_array().ok_or("chunks")? {
            v.extend(std::fs::read(dir.join(c["path"].as_str().ok_or("chunk path")?)).map_err(|e| e.to_string())?);
        }
        v
    } else {
        std::fs::read(dir.join(o["path"].as_str().ok_or("object path")?)).map_err(|e| e.to_string())?
    };
    let want = o["sha256"].as_str().ok_or("sha")?;
    if hex(&Sha256::digest(&data)) != want { return Err(format!("object hash mismatch in {}", dir.display())); }
    Ok(data)
}

impl Launch {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let manifest: Value = serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        let mut pre = Vec::new(); let mut post = Vec::new();
        for a in manifest["allocations"].as_array().ok_or("allocations")? {
            pre.push(object(dir, &a["pre"])?);
            post.push(if a["post"]["sha256"] == a["pre"]["sha256"] { pre.last().unwrap().clone() } else { object(dir, &a["post"])? });
        }
        let image = object(dir, &manifest["image"])?;
        Ok(Self { dir: dir.to_path_buf(), manifest, pre, post, image })
    }
    pub fn arg(&self, name: &str) -> Result<&Value, String> {
        self.manifest["arguments"].as_array().unwrap().iter().find(|a| a["name"] == name)
            .ok_or(format!("argument {name}"))
    }
    pub fn scalar_bytes(&self, name: &str) -> Result<[u8; 4], String> {
        let h = self.arg(name)?["value_hex"].as_str().ok_or("value_hex")?;
        let mut b = [0u8; 4];
        for i in 0..4 { b[i] = u8::from_str_radix(&h[2 * i..2 * i + 2], 16).map_err(|e| e.to_string())?; }
        Ok(b)
    }
    pub fn i32(&self, name: &str) -> Result<i32, String> { Ok(i32::from_le_bytes(self.scalar_bytes(name)?)) }
    pub fn f32(&self, name: &str) -> Result<f32, String> { Ok(f32::from_le_bytes(self.scalar_bytes(name)?)) }
    pub fn alloc_of(&self, name: &str) -> Result<usize, String> {
        Ok(self.arg(name)?["allocation"].as_u64().ok_or("allocation")? as usize)
    }
    pub fn geometry(&self) -> ([u32; 3], [u32; 3]) {
        let g = &self.manifest["geometry"];
        let v = |k: &str| { let a = g[k].as_array().unwrap(); [a[0].as_u64().unwrap() as u32, a[1].as_u64().unwrap() as u32, a[2].as_u64().unwrap() as u32] };
        (v("grid"), v("block"))
    }
}

pub fn f32s(b: &[u8]) -> Vec<f32> { crate::num::par_vec(b.len() / 4, |i| f32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap())) }
pub fn f64s(b: &[u8]) -> Vec<f64> { crate::num::par_vec(b.len() / 8, |i| f64::from_le_bytes(b[8 * i..8 * i + 8].try_into().unwrap())) }
pub fn bytes_f32(v: &[f32]) -> Vec<u8> { crate::num::par_vec(4 * v.len(), |i| v[i / 4].to_le_bytes()[i % 4]) }
pub fn bytes_f64(v: &[f64]) -> Vec<u8> { crate::num::par_vec(8 * v.len(), |i| v[i / 8].to_le_bytes()[i % 8]) }
pub fn words_differ(a: &[u8], b: &[u8]) -> usize {
    let n = a.len().min(b.len()) / 4;
    crate::num::par_map_sum(n, |i| if a[4 * i..4 * i + 4] != b[4 * i..4 * i + 4] { 1.0 } else { 0.0 }) as usize + (a.len() as isize - b.len() as isize).unsigned_abs() / 4
}
