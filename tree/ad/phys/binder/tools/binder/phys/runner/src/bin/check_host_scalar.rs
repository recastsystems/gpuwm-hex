// SPDX-License-Identifier: Apache-2.0
//! Exercise the native live-scalar cast against an independent CPU oracle.
use serde::Deserialize;
use std::io::{self, BufRead};
#[path = "../../../../../../runtime/src/binder/phys/host_scalar.rs"]
mod host_scalar;

#[derive(Deserialize)]
struct Case { cast: host_scalar::Cast, input: String, expected: Option<String> }
fn decode(text: &str) -> Result<Vec<u8>, String> {
    if text.len() % 2 != 0 || !text.is_ascii() { return Err("invalid hex".into()); }
    (0..text.len()).step_by(2).map(|i| u8::from_str_radix(&text[i..i+2], 16)
        .map_err(|_| "invalid hex".to_owned())).collect()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (mut cases, mut differences) = (0_u64, 0_u64);
    for line in io::stdin().lock().lines() {
        let case: Case = serde_json::from_str(&line?)?;
        let got = host_scalar::evaluate(case.cast, &decode(&case.input)?);
        let matches = match (got, case.expected) {
            (Err(_), None) => true,
            (Ok(bytes), Some(expected)) => bytes == decode(&expected)?,
            _ => false,
        };
        cases += 1;
        differences += u64::from(!matches);
    }
    println!("{}", serde_json::json!({"complete": false, "cases": cases, "differing_cases": differences,
                                      "native_physics_replay": false}));
    if cases == 0 || differences != 0 { std::process::exit(1); }
    Ok(())
}
