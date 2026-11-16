// SPDX-License-Identifier: Apache-2.0
//! Check the actual native frame rule against independently observed launches.
#[path = "../../../../../../runtime/src/binder/phys/vinterp.rs"]
mod vinterp;
use serde::Deserialize;
use std::io::{self, BufRead};

#[derive(Deserialize)]
struct Case { pmid: [i64; 2], pin: i64, ozone: [i64; 2], k: i32,
              scalar_hex: [String; 4], grid: Vec<u32>, block: Vec<u32> }
fn hex(bytes: &[u8]) -> String { bytes.iter().map(|b| format!("{b:02x}")).collect() }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (mut count, mut scalar_differences, mut geometry_differences) = (0, 0, 0);
    for line in io::stdin().lock().lines() {
        let case: Case = serde_json::from_str(&line?)?;
        let live = vinterp::prepare(case.pmid, case.pin, case.ozone)?.ok_or("empty observed launch")?;
        for (actual, expected) in live.scalar_bytes(case.k)?.iter().zip(case.scalar_hex.iter()) {
            if hex(actual) != *expected { scalar_differences += 1; }
        }
        if case.grid != [live.grid[0]] || case.block != [128] { geometry_differences += 1; }
        count += 1;
    }
    println!("{}", serde_json::json!({"complete":false,"launches_checked":count,
        "scalar_words_checked":count * 4,"scalar_words_different":scalar_differences,
        "geometry_different":geometry_differences,"view_ownership_proved":false,
        "native_physics_replay":false}));
    if count == 0 || scalar_differences != 0 || geometry_differences != 0 { std::process::exit(1); }
    Ok(())
}
