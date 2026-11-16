// SPDX-License-Identifier: Apache-2.0
//! Check actual ordinary launch words and boundary hashes using the runner's
//! live TEMPO state update. Input owner and complete replay remain unproved.
#[path = "../../../../../../runtime/src/binder/phys/host_array.rs"]
mod host_array;
// The checked rule does not use the poison machinery. HostState's unrelated
// poison method still needs this matching type in the standalone checker.
mod poison {
    pub struct HostWrite { pub allocation: u64, pub offset: usize, pub bytes: usize, pub pattern: &'static [u8] }
}
#[path = "../../../../../../runtime/src/binder/phys/host_state.rs"]
mod host_state;
#[path = "../../../../../../runtime/src/binder/phys/host_scalar.rs"]
mod host_scalar;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::HashSet, env, fs};

#[derive(Deserialize)]
struct Input { initial_flag: u8, initial_boundary_sha256: String, checks: Vec<Check>,
               complete: bool, native_physics_replay: bool }
#[derive(Deserialize)]
struct Check { step: usize, ordinal: u64, argument: usize, reference_hex: String, boundary_sha256: String }

fn hex(bytes: &[u8]) -> String { bytes.iter().map(|x| format!("{x:02x}")).collect() }
fn scalar_hash(flag: u8) -> Result<String, String> {
    let value = match flag { 0 => b"bool:0", 1 => b"bool:1", _ => return Err("invalid flag".into()) };
    Ok(hex(&Sha256::digest(value)))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).ok_or("need prepared ordinary TEMPO checks")?;
    let input: Input = serde_json::from_slice(&fs::read(path)?)?;
    if input.complete || input.native_physics_replay || input.checks.len() != 6 {
        return Err("expected six incomplete observational checks".into());
    }
    let mut state = host_state::HostState::default();
    state.insert(1, vec![input.initial_flag])?;
    if scalar_hash(state.get(1)?[0])? != input.initial_boundary_sha256 {
        return Err("initial TEMPO boundary differs".into());
    }
    let mut ordinals = HashSet::new();
    for (index, check) in input.checks.iter().enumerate() {
        if check.step != index + 1 || !ordinals.insert(check.ordinal) || check.argument != 13 {
            return Err("invalid TEMPO launch sequence or source argument".into());
        }
        let current = host_scalar::evaluate(host_scalar::Cast::BoolToI32, state.get(1)?)?;
        if hex(&current) != check.reference_hex { return Err("TEMPO launch word differs".into()); }
        state.finish_tempo_main(1, 0)?;
        if scalar_hash(state.get(1)?[0])? != check.boundary_sha256 {
            return Err("TEMPO boundary differs".into());
        }
    }
    println!("{}", serde_json::json!({"complete": false, "native_physics_replay": false,
        "steps": input.checks.len(), "launch_words_checked": input.checks.len(),
        "boundary_hashes_checked": input.checks.len() + 1, "differences": 0}));
    Ok(())
}
