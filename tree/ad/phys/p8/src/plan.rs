// SPDX-License-Identifier: Apache-2.0
//! Read-only view of a sealed hex physics plan (ad.binder.plan/v1, the hrrrv5
//! milestone plan of ad0-l01-binder-phys). Walks the cadence in order, decodes
//! kernel arguments and tracks the pointer tables written by the plan.
use serde_json::Value;
use std::collections::HashMap;

pub struct Plan {
    pub v: Value,
    pub host: HashMap<u64, Vec<u8>>,
    pub bytes: HashMap<u64, usize>,
    /// Sizes of allocations made by `allocate` commands inside steps.
    pub later: HashMap<u64, usize>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Slice { pub alloc: u64, pub off: usize }
pub fn slice(v: &Value) -> Slice {
    Slice { alloc: v["allocation"].as_u64().unwrap_or(u64::MAX), off: v["offset"].as_u64().unwrap_or(0) as usize }
}

#[derive(Clone, Debug)]
pub enum Arg {
    Ptr(Slice),
    /// CuPy CArray by value: origin, size, shape and byte strides.
    CArray { at: Slice, size: i64, shape: Vec<i64>, strides: Vec<i64> },
    /// Any other by-value bytes with relocated pointers.
    Packed { bytes: Vec<u8>, ptrs: Vec<(usize, Slice)> },
    Scalar(Vec<u8>),
    Frame(String),
}
pub fn hex(s: &str) -> Vec<u8> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap_or(0)).collect()
}
pub fn le_i64(b: &[u8], at: usize) -> i64 { i64::from_le_bytes(b[at..at + 8].try_into().unwrap()) }

impl Plan {
    pub fn load(path: &str) -> Result<Self, String> {
        let raw = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let v: Value = serde_json::from_slice(&raw).map_err(|e| format!("{path}: {e}"))?;
        let mut host = HashMap::new();
        for h in v["host_allocations"].as_array().into_iter().flatten() {
            let init = &h["initial"];
            if init["encoding"] == "hex" { host.insert(h["id"].as_u64().unwrap(), hex(init["bytes"].as_str().unwrap_or(""))); }
        }
        let mut bytes = HashMap::new();
        for a in v["allocations"].as_array().into_iter().flatten() {
            bytes.insert(a["id"].as_u64().unwrap(), a["bytes"].as_u64().unwrap() as usize);
        }
        let mut later = HashMap::new();
        for (_, ops) in v["blocks"].as_object().into_iter().flatten() {
            for o in ops.as_array().into_iter().flatten() {
                if o["kind"] == "allocate" { later.insert(o["id"].as_u64().unwrap(), o["bytes"].as_u64().unwrap() as usize); }
            }
        }
        Ok(Plan { v, host, bytes, later })
    }
    /// (step number from 1, block name) in cadence order.
    pub fn steps(&self) -> Vec<(usize, String)> {
        let mut out = Vec::new();
        for (i, s) in self.v["cadence"].as_array().into_iter().flatten().enumerate() {
            for b in s.as_array().into_iter().flatten() { out.push((i + 1, b.as_str().unwrap().to_string())); }
        }
        out
    }
    pub fn ops(&self, block: &str) -> &[Value] { self.v["blocks"][block].as_array().map(|v| v.as_slice()).unwrap_or(&[]) }
    pub fn host_i64(&self, s: &Value) -> Result<i64, String> {
        let s = slice(s);
        let h = self.host.get(&s.alloc).ok_or_else(|| format!("no host allocation {}", s.alloc))?;
        if s.off + 8 > h.len() { return Err("host word outside allocation".into()); }
        Ok(le_i64(h, s.off))
    }
    pub fn host_u32(&self, s: &Value) -> Result<u32, String> {
        let s = slice(s);
        let h = self.host.get(&s.alloc).ok_or_else(|| format!("no host allocation {}", s.alloc))?;
        Ok(u32::from_le_bytes(h.get(s.off..s.off + 4).ok_or("host word outside allocation")?.try_into().unwrap()))
    }
}

pub fn args(op: &Value) -> Vec<Arg> {
    op["arguments"].as_array().into_iter().flatten().map(|a| match a["kind"].as_str().unwrap_or("") {
        "device" => Arg::Ptr(slice(&a["target"])),
        "patched" => {
            let b = hex(a["bytes_hex"].as_str().unwrap_or(""));
            let ptrs: Vec<(usize, Slice)> = a["pointers"].as_array().into_iter().flatten()
                .map(|p| (p["byte_offset"].as_u64().unwrap() as usize, slice(&p["target"]))).collect();
            if ptrs.len() == 1 && ptrs[0].0 == 0 && b.len() >= 16 && (b.len() - 16) % 16 == 0 {
                let n = (b.len() - 16) / 16;
                Arg::CArray { at: ptrs[0].1, size: le_i64(&b, 8),
                              shape: (0..n).map(|k| le_i64(&b, 16 + 8 * k)).collect(),
                              strides: (0..n).map(|k| le_i64(&b, 16 + 8 * (n + k))).collect() }
            } else { Arg::Packed { bytes: b, ptrs } }
        },
        "bytes" => Arg::Scalar(hex(a["bytes_hex"].as_str().unwrap_or(""))),
        k => Arg::Frame(k.to_string()),
    }).collect()
}

/// Byte extent [lo, hi) of a CArray relative to its origin.
pub fn carray_span(shape: &[i64], strides: &[i64], item: i64) -> (i64, i64) {
    let (mut lo, mut hi) = (0i64, 0i64);
    if shape.iter().any(|&n| n == 0) { return (0, 0); }
    for (&n, &s) in shape.iter().zip(strides) { let d = (n - 1) * s; lo += d.min(0); hi += d.max(0); }
    (lo, hi + item)
}

/// Element sizes of the array operands of a CuPy elementwise kernel, from its
/// name (cupy_<op>__<dtypes>); python scalars ("float", "int") are by value.
pub fn cupy_items(function: &str) -> Vec<i64> {
    let Some((_, d)) = function.split_once("__") else { return vec![] };
    d.split('_').filter_map(|t| match t {
        "float32" | "int32" | "uint32" => Some(4), "float64" | "int64" | "uint64" => Some(8),
        "bool" | "int8" | "uint8" => Some(1), "float16" | "int16" => Some(2), _ => None,
    }).collect()
}
