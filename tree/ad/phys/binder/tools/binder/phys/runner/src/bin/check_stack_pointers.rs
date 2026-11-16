// SPDX-License-Identifier: Apache-2.0
//! Compare the native live pointer-table rule to observed CuPy stack inputs.
use flate2::read::GzDecoder;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::{BTreeSet, HashMap}, env, fs::File,
          io::{BufRead, BufReader, ErrorKind}};

mod plan {
    #[derive(Clone, Copy, Debug)]
    pub struct Slice { pub allocation: u64, pub offset: usize }
}
mod submit {
    pub struct Allocation { pub id: u64, pub address: u64, pub bytes: usize }
}
#[path = "../../../../../../runtime/src/binder/phys/pointer_table.rs"]
mod pointer_table;

#[derive(Deserialize)]
struct Array { base: u64, allocation_id: u64, bytes: usize, offset: usize }
#[derive(Deserialize)]
struct Observation {
    schema: String, complete: bool, success: bool, step: u64, thread_id: u64,
    enter_ns: u64, exit_ns: u64, inputs: Vec<Array>,
    pointer_bytes: usize, pointer_sha256: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let path = args.next().ok_or("usage: check_stack_pointers INPUT.gz [--allow-partial]")?;
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
            Ok(_) => return Err("incomplete stack observation".into()),
            Err(e) => return Err(e.into()),
        }
        let observed: Observation = serde_json::from_str(&line)?;
        if observed.schema != "ad.binder.phys-stack-inputs/v1" || observed.complete ||
           !observed.success || observed.step == 0 || observed.enter_ns > observed.exit_ns {
            return Err("invalid stack observation".into());
        }
        let mut allocations: HashMap<u64, submit::Allocation> = HashMap::new();
        let mut entries = Vec::new();
        for array in &observed.inputs {
            if let Some(prior) = allocations.get(&array.allocation_id) {
                if prior.address != array.base || prior.bytes != array.bytes {
                    return Err("inconsistent stack allocation identity".into());
                }
            } else {
                allocations.insert(array.allocation_id, submit::Allocation {
                    id: array.allocation_id, address: array.base, bytes: array.bytes });
            }
            entries.push(plan::Slice { allocation: array.allocation_id, offset: array.offset });
        }
        let bytes = pointer_table::build(&entries, &allocations)?;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        if bytes.len() != observed.pointer_bytes || digest != observed.pointer_sha256 {
            return Err(format!("stack pointer mismatch in observation {}", results.len()).into());
        }
        results.push(serde_json::json!({"step": observed.step, "thread_id": observed.thread_id,
            "enter_ns": observed.enter_ns, "exit_ns": observed.exit_ns,
            "pointer_bytes": bytes.len(), "pointer_sha256": digest}));
        steps.insert(observed.step);
    };
    if results.is_empty() { return Err("no stack observations".into()); }
    println!("{}", serde_json::json!({"observations": results.len(), "steps": steps,
        "stream_closed": stream_closed, "mismatches": 0, "checks": results,
        "path_selection_proved": false, "allocation_ownership_proved": false,
        "native_physics_replay": false}));
    Ok(())
}
