// SPDX-License-Identifier: Apache-2.0
//! Check the replay's live host packing rule against same-run upload hashes.
//! The source files contain initial inputs only, never recorded host outputs.
#[path = "../../../../../../runtime/src/binder/phys/host_array.rs"]
mod host_array;
use flate2::read::GzDecoder;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{env, fs::File, io::Read, path::{Component, Path}};

#[derive(Deserialize)]
struct Input { schema: String, complete: bool, checks: Vec<Check> }
#[derive(Deserialize)]
struct Check {
    sequence: u64, owner: String, initial_path: String, initial_bytes: usize,
    initial_sha256: String, offset: i64, shape: Vec<usize>, strides: Vec<i64>,
    item_bytes: usize, upload_bytes: usize, upload_sha256: String,
}
fn digest(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let root = args.next().ok_or("usage: check_initial_host_views ROOT CHECKS.json")?;
    let input = args.next().ok_or("missing checks")?;
    if args.next().is_some() { return Err("too many arguments".into()); }
    let spec: Input = serde_json::from_reader(File::open(input)?)?;
    if spec.schema != "ad.binder.phys-initial-host-view-checks/v1" || spec.complete || spec.checks.is_empty() {
        return Err("invalid partial check set".into());
    }
    let mut results = Vec::with_capacity(spec.checks.len());
    let (mut matched, mut different) = (0_usize, 0_usize);
    for check in spec.checks {
        let relative = Path::new(&check.initial_path);
        if relative.components().any(|part| !matches!(part, Component::Normal(_))) {
            return Err("initial input path is not relative".into());
        }
        let mut raw = Vec::new();
        GzDecoder::new(File::open(Path::new(&root).join(relative))?)
            .take(u64::try_from(check.initial_bytes)?.checked_add(1).ok_or("input length overflow")?)
            .read_to_end(&mut raw)?;
        if raw.len() != check.initial_bytes || digest(&raw) != check.initial_sha256 {
            return Err(format!("initial input length or hash differs for {}", check.owner).into());
        }
        let view = host_array::View { offset: check.offset, shape: check.shape,
                                     strides: check.strides, item_bytes: check.item_bytes };
        if host_array::packed_size(raw.len(), &view)? != check.upload_bytes {
            return Err(format!("input view size differs for {}", check.owner).into());
        }
        let c = digest(&host_array::pack(&raw, &view, host_array::Order::C)?);
        let order = if c == check.upload_sha256 { Some("c") }
                    else if digest(&host_array::pack(&raw, &view, host_array::Order::Fortran)?) == check.upload_sha256 {
                        Some("fortran")
                    } else { None };
        if order.is_some() { matched += 1 } else { different += 1 }
        results.push(serde_json::json!({"sequence":check.sequence, "owner":check.owner,
            "bytes":check.upload_bytes, "matching_order":order, "upload_sha256":check.upload_sha256}));
    }
    println!("{}", serde_json::json!({"schema":"ad.binder.phys-initial-host-view-results/v1",
        "complete":false, "checks":results.len(), "matching_initial_view":matched,
        "different_from_initial":different, "results":results,
        "ownership_proved":false, "intervening_host_mutations_proved":false,
        "native_physics_replay":false}));
    Ok(())
}
