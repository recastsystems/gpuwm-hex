// SPDX-License-Identifier: Apache-2.0
//! Captured hex launches (schema ad.hex.launch/v1): manifest, pre/post allocations, argument binding.
use crate::cuda::{DBuf, KArg};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub fn sha(b: &[u8]) -> String { let mut h = Sha256::new(); h.update(b); format!("{:x}", h.finalize()) }

pub struct Cap { pub dir: PathBuf, pub m: Value, pub pre: Vec<Vec<u8>>, pub post: Vec<Vec<u8>> }

impl Cap {
    pub fn load(dir: &Path) -> Cap {
        let m: Value = serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).expect("manifest")).unwrap();
        let mut pre = vec![]; let mut post = vec![];
        for a in m["allocations"].as_array().unwrap() {
            for (k, v) in [("pre", &mut pre), ("post", &mut post)] {
                let p = dir.join(a[k]["path"].as_str().unwrap());
                let b = std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
                assert_eq!(sha(&b), a[k]["sha256"].as_str().unwrap(), "sha {}", p.display());
                v.push(b);
            }
        }
        Cap { dir: dir.to_path_buf(), m, pre, post }
    }
    pub fn args(&self) -> &Vec<Value> { self.m["arguments"].as_array().unwrap() }
    pub fn arg(&self, name: &str) -> &Value { self.args().iter().find(|a| a["name"] == name).unwrap_or_else(|| panic!("arg {name}")) }
    pub fn alloc(&self, name: &str) -> usize { self.arg(name)["allocation"].as_u64().unwrap() as usize }
    pub fn offset(&self, name: &str) -> usize { self.arg(name)["offset"].as_u64().unwrap() as usize }
    pub fn count(&self, name: &str) -> usize { self.arg(name)["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).product() }
    pub fn scalar(&self, name: &str) -> Vec<u8> {
        let h = self.arg(name)["value_hex"].as_str().unwrap();
        (0..h.len() / 2).map(|i| u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap()).collect()
    }
    pub fn scalar_i32(&self, name: &str) -> i32 { let b = self.scalar(name); i32::from_le_bytes([b[0], b[1], b[2], b[3]]) }
    pub fn scalar_f32(&self, name: &str) -> f32 { let b = self.scalar(name); f32::from_le_bytes([b[0], b[1], b[2], b[3]]) }
    pub fn grid(&self) -> [u32; 3] { g3(&self.m["geometry"]["grid"]) }
    pub fn block(&self) -> [u32; 3] { g3(&self.m["geometry"]["block"]) }
    pub fn pre_f32(&self, name: &str) -> Vec<f32> { f32s(&self.pre[self.alloc(name)][self.offset(name)..], self.count(name)) }
    pub fn post_f32(&self, name: &str) -> Vec<f32> { f32s(&self.post[self.alloc(name)][self.offset(name)..], self.count(name)) }
    pub fn pre_i32(&self, name: &str) -> Vec<i32> { i32s(&self.pre[self.alloc(name)][self.offset(name)..], self.count(name)) }
    pub fn post_i32(&self, name: &str) -> Vec<i32> { i32s(&self.post[self.alloc(name)][self.offset(name)..], self.count(name)) }
    pub fn image(&self) -> Vec<u8> {
        let b = std::fs::read(self.dir.join(self.m["image"]["path"].as_str().unwrap())).unwrap();
        assert_eq!(sha(&b), self.m["image"]["sha256"].as_str().unwrap());
        b
    }
}

fn g3(v: &Value) -> [u32; 3] { let a = v.as_array().unwrap(); [a[0].as_u64().unwrap() as u32, a[1].as_u64().unwrap() as u32, a[2].as_u64().unwrap() as u32] }
pub fn f32s(b: &[u8], n: usize) -> Vec<f32> { (0..n).map(|i| f32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap())).collect() }
pub fn i32s(b: &[u8], n: usize) -> Vec<i32> { (0..n).map(|i| i32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap())).collect() }
pub fn bytes_f32(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
pub fn bytes_f64(v: &[f64]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
pub fn f64s(b: &[u8], n: usize) -> Vec<f64> { (0..n).map(|i| f64::from_le_bytes(b[8 * i..8 * i + 8].try_into().unwrap())).collect() }

/// One device copy of every allocation of a capture, at its pre state.
pub struct Dev { pub bufs: Vec<DBuf> }
impl Dev {
    pub fn new(c: &Cap) -> Dev { let bufs = c.pre.iter().map(|b| { let d = DBuf::new(b.len()); d.upload_at(0, b); d }).collect(); Dev { bufs } }
    pub fn reset(&self, c: &Cap) { for (d, b) in self.bufs.iter().zip(&c.pre) { d.upload_at(0, b); } }
    pub fn ptr(&self, c: &Cap, name: &str) -> u64 { self.bufs[c.alloc(name)].ptr + c.offset(name) as u64 }
    pub fn put_f32(&self, c: &Cap, name: &str, v: &[f32]) { self.bufs[c.alloc(name)].upload_at(c.offset(name), &bytes_f32(v)); }
    pub fn get_f32(&self, c: &Cap, name: &str) -> Vec<f32> { let n = c.count(name); f32s(&self.bufs[c.alloc(name)].download_at(c.offset(name), 4 * n), n) }
    pub fn get_i32(&self, c: &Cap, name: &str) -> Vec<i32> { let n = c.count(name); i32s(&self.bufs[c.alloc(name)].download_at(c.offset(name), 4 * n), n) }
    /// The forward's own argument list, in manifest order.
    pub fn fwd_args(&self, c: &Cap) -> Vec<KArg> {
        c.args().iter().map(|a| if a["kind"] == "device" { KArg::Ptr(self.ptr(c, a["name"].as_str().unwrap())) } else { KArg::Bytes(c.scalar(a["name"].as_str().unwrap())) }).collect()
    }
}
