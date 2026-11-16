// SPDX-License-Identifier: Apache-2.0
//! Check real launch words through the actual native live-scalar conversion.
#[path = "../../../../../../runtime/src/binder/phys/host_scalar.rs"]
mod host_scalar;
use serde::Deserialize;
use std::{collections::{HashMap, HashSet}, env, fs};

#[derive(Deserialize)]
struct Input { input_dimensions: HashMap<String, i64>, checks: Vec<Check> }
#[derive(Deserialize)]
struct Check { step: usize, ordinal: u64, argument: usize, kernel: String,
               owner: String, reference_hex: String }

fn hex(bytes: &[u8]) -> String { bytes.iter().map(|x| format!("{x:02x}")).collect() }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).ok_or("need prepared ordinary launch checks")?;
    let input: Input = serde_json::from_slice(&fs::read(path)?)?;
    if input.checks.is_empty() { return Err("no launch checks".into()); }
    let mut keys = HashSet::new();
    let mut steps = HashSet::new();
    for check in &input.checks {
        if !(1..=6).contains(&check.step) || check.kernel.is_empty()
                || !keys.insert((check.ordinal, check.argument)) {
            return Err("invalid or repeated launch check".into());
        }
        let value = input.input_dimensions.get(&check.owner).ok_or("unknown live mesh owner")?;
        let current = host_scalar::evaluate(host_scalar::Cast::I64ToI32, &value.to_le_bytes())?;
        if check.reference_hex != hex(&current) {
            return Err(format!("step {} launch {} argument {}: mesh scalar differs",
                               check.step, check.ordinal, check.argument).into());
        }
        steps.insert(check.step);
    }
    if steps.len() != 6 { return Err("launch checks miss a step".into()); }
    println!("{}", serde_json::json!({"complete": false, "native_physics_replay": false,
        "steps": steps.len(), "launch_words_checked": input.checks.len(), "differences": 0}));
    Ok(())
}
