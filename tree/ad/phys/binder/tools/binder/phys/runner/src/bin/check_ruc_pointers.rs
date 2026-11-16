// SPDX-License-Identifier: Apache-2.0
//! Compare the runtime RUC pointer rule to observed forward upload hashes.
use flate2::read::GzDecoder;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::{BTreeSet, HashMap}, env, fs::File,
          io::{BufRead, BufReader, ErrorKind}};

mod plan {
    #[derive(Clone, Copy, Debug, serde::Deserialize)]
    pub struct Slice { pub allocation: u64, pub offset: usize }
}
mod submit {
    pub struct Allocation { pub id: u64, pub address: u64, pub bytes: usize }
}
#[path = "../../../../../../runtime/src/binder/phys/ruc.rs"]
mod ruc;

#[derive(Deserialize)]
struct Array {
    base: u64, allocation_id: u64, bytes: usize, offset: usize,
    dtype: String, shape: Vec<usize>, strides: Vec<i64>,
}
#[derive(Deserialize)]
struct Named { name: String, view: Array }
#[derive(Deserialize)]
struct Observation {
    schema: String, complete: bool, step: u64, entry: String, n: usize,
    clock_ns: u64, thread_id: u64, bytes: usize, sha256: String,
    fields: Vec<Named>, flags: Vec<Named>, tile_stop: Option<Array>,
}

fn view(array: &Array, allocations: &mut HashMap<u64, submit::Allocation>)
        -> Result<ruc::View, Box<dyn std::error::Error>> {
    let dtype = match array.dtype.as_str() {
        "<i4" | "=i4" => ruc::Dtype::I32,
        "<f4" | "=f4" => ruc::Dtype::F32,
        _ => return Err("unexpected RUC view dtype".into()),
    };
    if array.shape.len() != array.strides.len() { return Err("invalid RUC view rank".into()); }
    let mut bytes = 4usize;
    for &dimension in &array.shape { bytes = bytes.checked_mul(dimension).ok_or("RUC size overflow")?; }
    let mut stride = 4i64;
    let mut contiguous = true;
    if array.shape.iter().any(|&dimension| dimension == 0) {
        contiguous = true;
    } else {
        for (&dimension, &actual) in array.shape.iter().zip(&array.strides).rev() {
            if dimension > 1 && actual != stride { contiguous = false; }
            stride = stride.checked_mul(i64::try_from(dimension)?).ok_or("RUC stride overflow")?;
        }
    }
    if let Some(prior) = allocations.get(&array.allocation_id) {
        if prior.address != array.base || prior.bytes != array.bytes {
            return Err("inconsistent RUC allocation identity".into());
        }
    } else {
        allocations.insert(array.allocation_id, submit::Allocation {
            id: array.allocation_id, address: array.base, bytes: array.bytes,
        });
    }
    Ok(ruc::View { target: Some(plan::Slice { allocation: array.allocation_id, offset: array.offset }),
                   bytes, dtype, contiguous, last_dimension: array.shape.last().copied() })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let path = args.next().ok_or("usage: check_ruc_pointers INPUT.gz [--allow-partial]")?;
    let allow_partial = match args.next().as_deref() {
        None => false, Some("--allow-partial") => true, _ => return Err("invalid argument".into()),
    };
    if args.next().is_some() { return Err("too many arguments".into()); }
    let mut reader = BufReader::new(GzDecoder::new(File::open(path)?));
    let mut line = String::new();
    let mut results = Vec::new();
    let mut steps = BTreeSet::new();
    let stream_closed = loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break true,
            Ok(_) if line.ends_with('\n') => {},
            Ok(_) if allow_partial => break false,
            Err(e) if allow_partial && e.kind() == ErrorKind::UnexpectedEof => break false,
            Ok(_) => return Err("incomplete RUC observation".into()),
            Err(e) => return Err(e.into()),
        }
        let observed: Observation = serde_json::from_str(&line)?;
        if observed.schema != "ad.binder.phys-ruc-pointers/v1" || observed.complete || observed.step == 0 {
            return Err("invalid RUC observation".into());
        }
        let mut allocations = HashMap::new();
        let mut arguments = HashMap::new();
        let mut flags = HashMap::new();
        for field in &observed.fields {
            let value = view(&field.view, &mut allocations)?;
            if value.last_dimension != Some(observed.n) || arguments.insert(field.name.clone(), value).is_some() {
                return Err("invalid or duplicate RUC argument".into());
            }
        }
        for flag in &observed.flags {
            if flags.insert(flag.name.clone(), view(&flag.view, &mut allocations)?).is_some() {
                return Err("duplicate RUC presence flag".into());
            }
        }
        let tile = observed.tile_stop.as_ref().map(|a| view(a, &mut allocations)).transpose()?;
        let bytes = ruc::pointer_table(&observed.entry, &arguments, &flags, tile, &allocations)?;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        if bytes.len() != observed.bytes || digest != observed.sha256 {
            return Err(format!("RUC table mismatch in observation {}", results.len()).into());
        }
        results.push(serde_json::json!({"step": observed.step, "entry": observed.entry,
            "clock_ns": observed.clock_ns, "thread_id": observed.thread_id,
            "bytes": bytes.len(), "sha256": digest}));
        steps.insert(observed.step);
    };
    if results.is_empty() { return Err("no RUC observations".into()); }
    println!("{}", serde_json::json!({"observations": results.len(), "steps": steps,
        "stream_closed": stream_closed, "mismatches": 0, "checks": results,
        "allocation_ownership_proved": false, "native_physics_replay": false}));
    Ok(())
}
