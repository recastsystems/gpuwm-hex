// SPDX-License-Identifier: Apache-2.0
//! Compare the runtime's live-input table rule with observed forward checksums.
//! This is an offline check; it is not a native forecast replay or a census.
use flate2::read::GzDecoder;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::HashMap, env, fs::File, io::{BufRead, BufReader, ErrorKind}};

mod plan {
    #[derive(Clone, Copy, Debug, serde::Deserialize)]
    pub struct Slice { pub allocation: u64, pub offset: usize }
}
mod submit {
    pub struct Allocation { pub id: u64, pub address: u64, pub bytes: usize }
}
#[path = "../../../../../../runtime/src/binder/phys/copy_plan.rs"]
mod copy_plan;

#[derive(Deserialize)]
struct Input {
    dst: u64, src: u64, dst_stride: [i64; 4], src_stride: [i64; 4],
    shape: [i64; 4], fill: u32, mode: i32,
}
#[derive(Deserialize)]
struct Observation {
    schema: String, complete: bool, step: u64, bytes: usize, sha256: String,
    inputs: Vec<Input>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let path = args.next().ok_or("usage: check_copy_plan INPUT.gz [--allow-partial]")?;
    let allow_partial = match args.next().as_deref() {
        None => false,
        Some("--allow-partial") => true,
        _ => return Err("invalid argument".into()),
    };
    if args.next().is_some() { return Err("too many arguments".into()); }
    // Preserve the observed addresses only for this offline comparison. In the
    // replay, actual allocation bounds and live addresses come from the runtime.
    let alloc = HashMap::from([(1, submit::Allocation { id: 1, address: 0, bytes: usize::MAX })]);
    let mut reader = BufReader::new(GzDecoder::new(File::open(path)?));
    let mut line = String::new();
    let (mut checked, mut steps, mut rows) = (0_u64, std::collections::BTreeSet::new(), 0_u64);
    let mut launches = Vec::new();
    let stream_closed = loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break true,
            Ok(_) if line.ends_with('\n') => {},
            Ok(_) if allow_partial => break false,
            Err(e) if allow_partial && e.kind() == ErrorKind::UnexpectedEof => break false,
            Ok(_) => return Err("incomplete observation row".into()),
            Err(e) => return Err(e.into()),
        }
        let observed: Observation = serde_json::from_str(&line)?;
        if observed.schema != "ad.binder.phys-copy-plan-input/v1" || observed.complete || observed.step == 0
            || observed.inputs.is_empty() { return Err("invalid observation".into()); }
        let mut inputs = Vec::with_capacity(observed.inputs.len());
        for input in observed.inputs {
            let dst = usize::try_from(input.dst)?;
            let src = if input.src == 0 { None } else {
                Some(plan::Slice { allocation: 1, offset: usize::try_from(input.src)? })
            };
            inputs.push(copy_plan::Row { dst: plan::Slice { allocation: 1, offset: dst }, src,
                dst_stride: input.dst_stride, src_stride: input.src_stride, shape: input.shape,
                fill: input.fill, mode: input.mode });
        }
        let actual = copy_plan::table(&inputs, &alloc)?;
        let digest = format!("{:x}", Sha256::digest(&actual));
        if actual.len() != observed.bytes || digest != observed.sha256 {
            return Err(format!("table mismatch at observation {checked}").into());
        }
        let launch = copy_plan::launch(&inputs)?;
        let cached_bytes = copy_plan::encode_cache(&inputs, plan::Slice { allocation: 2, offset: 0 })?;
        let (cached, table_bytes, table) = copy_plan::decode_cache(&cached_bytes)?;
        if cached != launch || table_bytes != actual.len() || (table.allocation, table.offset) != (2, 0) {
            return Err(format!("cached launch mismatch at observation {checked}").into());
        }
        launches.push(serde_json::json!({"step": observed.step, "sha256": digest,
            "grid": launch.grid, "block": launch.block, "tiled": launch.tiled}));
        checked += 1;
        rows += u64::try_from(inputs.len())?;
        steps.insert(observed.step);
    };
    if checked == 0 { return Err("no complete observations".into()); }
    println!("{}", serde_json::json!({ "observations": checked, "rows": rows,
        "steps": steps, "stream_closed": stream_closed, "mismatches": 0, "launches": launches,
        "cached_launches_checked": checked,
        "allocation_ownership_proved": false, "native_physics_replay": false }));
    Ok(())
}
