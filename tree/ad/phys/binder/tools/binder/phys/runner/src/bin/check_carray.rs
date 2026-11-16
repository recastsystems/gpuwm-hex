// SPDX-License-Identifier: Apache-2.0
//! Compare native CArray encoding to independently observed view descriptors.
use serde::Deserialize;
use std::{collections::HashMap, io::{self, BufRead}};
mod plan {
    use serde::Deserialize;
    #[derive(Clone, Copy, Debug, Deserialize)]
    pub struct Slice { pub allocation: u64, pub offset: usize }
}
mod submit {
    pub struct Allocation { pub id: u64, pub address: u64, pub bytes: usize }
}
#[path = "../../../../../../runtime/src/binder/phys/carray.rs"]
mod carray;

#[derive(Deserialize)]
struct Case { base: u64, bytes: usize, offset: usize, shape: Vec<i64>,
              strides: Vec<i64>, item_bytes: usize, captured_hex: String }

fn hex(bytes: &[u8]) -> String { bytes.iter().map(|b| format!("{b:02x}")).collect() }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut count = 0;
    let mut errors = 0;
    for line in io::stdin().lock().lines() {
        let case: Case = serde_json::from_str(&line?)?;
        let allocation = submit::Allocation { id: 1, address: case.base, bytes: case.bytes };
        let allocations = HashMap::from([(1, allocation)]);
        let target = plan::Slice { allocation: 1, offset: case.offset };
        let mut actual = carray::encode(Some(target), &case.shape, &case.strides, case.item_bytes, &allocations)?;
        actual[..8].copy_from_slice(&case.base.checked_add(case.offset as u64).ok_or("address overflow")?.to_le_bytes());
        if hex(&actual) != case.captured_hex { errors += 1; }
        count += 1;
    }
    println!("{}", serde_json::json!({"complete":false,"carray_checks":count,
        "differing_carray_arguments":errors,"ownership_proven":false,"native_physics_replay":false}));
    if count == 0 || errors != 0 { std::process::exit(1); }
    Ok(())
}
