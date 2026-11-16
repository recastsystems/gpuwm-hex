// SPDX-License-Identifier: Apache-2.0
//! Compare the native TEMPO pointer rule to same-run source observations.
use flate2::read::GzDecoder;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::{BTreeSet, HashMap}, env, fs::File,
          io::{BufRead, BufReader, ErrorKind}};

mod plan {
    use serde::Deserialize;
    #[derive(Clone, Copy, Debug, Deserialize)]
    pub struct Slice { pub allocation: u64, pub offset: usize }
}
mod submit {
    pub struct Allocation { pub id: u64, pub address: u64, pub bytes: usize }
}
#[path = "../../../../../../runtime/src/binder/phys/tempo.rs"]
mod tempo;

#[derive(Deserialize)]
struct Array {
    base: u64, allocation_id: Option<u64>, bytes: usize, offset: usize,
    dtype: String, shape: Vec<usize>, strides: Vec<i64>,
}
#[derive(Deserialize)]
struct Observation {
    schema: String, complete: bool, success: bool, step: u64, thread_id: u64,
    enter_ns: u64, exit_ns: u64, shape: Option<Vec<usize>>, device: u64, optional: bool,
    inputs: Vec<Option<Array>>, bytes: usize, sha256: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let path = args.next().ok_or("usage: check_tempo_pointers INPUT.gz [--allow-partial]")?;
    let allow_partial = match args.next().as_deref() {
        None => false, Some("--allow-partial") => true, _ => return Err("invalid argument".into()),
    };
    if args.next().is_some() { return Err("too many arguments".into()); }
    let mut reader = BufReader::new(GzDecoder::new(File::open(path)?));
    let mut line = String::new();
    let mut checks = Vec::new();
    let mut steps = BTreeSet::new();
    let stream_closed = loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break true,
            Ok(_) if line.ends_with('\n') => {},
            Ok(_) if allow_partial => break false,
            Err(e) if allow_partial && e.kind() == ErrorKind::UnexpectedEof => break false,
            Ok(_) => return Err("incomplete TEMPO observation".into()),
            Err(e) => return Err(e.into()),
        }
        let observed: Observation = serde_json::from_str(&line)?;
        if observed.schema != "ad.binder.phys-tempo-pointers/v1" || observed.complete ||
           !observed.success || observed.step == 0 || observed.enter_ns > observed.exit_ns {
            return Err("invalid TEMPO pointer observation".into());
        }
        let mut allocations: HashMap<u64, submit::Allocation> = HashMap::new();
        let mut entries = Vec::with_capacity(observed.inputs.len());
        for (index, item) in observed.inputs.iter().enumerate() {
            let Some(array) = item else { entries.push(None); continue; };
            // The check builds a local identity even when the observer has no
            // recorded allocation ID; only address and extent are compared.
            let id = u64::try_from(index + 1)?;
            let target = if array.base == 0 && array.bytes == 0 && array.offset == 0 {
                None
            } else {
                allocations.insert(id, submit::Allocation { id, address: array.base, bytes: array.bytes });
                Some(plan::Slice { allocation: id, offset: array.offset })
            };
            let _observed_allocation_id = array.allocation_id;
            entries.push(Some(tempo::View { target, dtype: array.dtype.clone(),
                shape: array.shape.clone(), strides: array.strides.clone(), device: observed.device }));
        }
        let bytes = tempo::pointer_table(&entries, observed.shape.as_deref(), observed.device,
                                         observed.optional, &allocations)?;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        if bytes.len() != observed.bytes || digest != observed.sha256 {
            return Err(format!("TEMPO pointer mismatch in observation {}", checks.len()).into());
        }
        checks.push(serde_json::json!({"step": observed.step, "thread_id": observed.thread_id,
            "enter_ns": observed.enter_ns, "exit_ns": observed.exit_ns,
            "bytes": bytes.len(), "sha256": digest}));
        steps.insert(observed.step);
    };
    if checks.is_empty() { return Err("no TEMPO observations".into()); }
    println!("{}", serde_json::json!({"observations": checks.len(), "steps": steps,
        "stream_closed": stream_closed, "mismatches": 0, "checks": checks,
        "cache_selection_proved": false, "allocation_ownership_proved": false,
        "native_physics_replay": false}));
    Ok(())
}
