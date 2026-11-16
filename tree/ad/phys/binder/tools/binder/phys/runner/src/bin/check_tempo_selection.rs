// SPDX-License-Identifier: Apache-2.0
//! Exercise native TEMPO named selection with independently supplied cases.
use serde::Deserialize;
use std::io::{self, BufRead};
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
struct Case { carriers: tempo::selection::Carriers, flags: [i32; 2] }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for line in io::stdin().lock().lines() {
        let case: Case = serde_json::from_str(&line?)?;
        let output = match tempo::selection::select(&case.carriers, case.flags) {
            Ok(tables) => serde_json::json!({"ok": true, "tables": tables.iter().map(|table|
                serde_json::json!({"name":table.name, "shape":table.shape, "device":table.device,
                    "optional":table.optional, "entries":table.entries.iter().map(|view|
                        view.as_ref().and_then(|v| v.target).map(|s| s.allocation))
                        .collect::<Vec<_>>() })).collect::<Vec<_>>()}),
            Err(_) => serde_json::json!({"ok": false}),
        };
        println!("{output}");
    }
    Ok(())
}
