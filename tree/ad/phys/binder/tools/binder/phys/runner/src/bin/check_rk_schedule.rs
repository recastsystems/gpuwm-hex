// SPDX-License-Identifier: Apache-2.0
//! Compare the native schedule with independently evaluated pinned source.
#[path = "../../../../../../runtime/src/binder/phys/rk_schedule.rs"]
mod rk_schedule;
use serde::Deserialize;
use std::io::{self, BufRead};

#[derive(Deserialize)]
struct Expected { stage: i64, count: i64, large_f64: String, acoustic_f64: String,
                  large_f32: String, acoustic_f32: String }
#[derive(Deserialize)]
struct Case { dt: f64, order: i64, acoustic_substeps: i64, dynamics_splits: i64,
              expected: Option<Vec<Expected>> }
fn hex(bytes: &[u8]) -> String { bytes.iter().map(|b| format!("{b:02x}")).collect() }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (mut cases, mut stages, mut differences, mut refusals) = (0, 0, 0, 0);
    for line in io::stdin().lock().lines() {
        let case: Case = serde_json::from_str(&line?)?;
        let actual = rk_schedule::Schedule::new(case.dt, case.order, case.acoustic_substeps, case.dynamics_splits);
        match (actual, case.expected) {
            (Err(_), None) => refusals += 1,
            (Ok(schedule), Some(expected)) if expected.len() == 3 => {
                for (stage, expected) in schedule.stages.iter().zip(expected) {
                    stages += 1;
                    differences += usize::from(stage.stage != expected.stage || stage.acoustic_steps != expected.count
                        || hex(&stage.large_timestep.to_le_bytes()) != expected.large_f64
                        || hex(&stage.acoustic_timestep.to_le_bytes()) != expected.acoustic_f64
                        || hex(&(stage.large_timestep as f32).to_le_bytes()) != expected.large_f32
                        || hex(&(stage.acoustic_timestep as f32).to_le_bytes()) != expected.acoustic_f32);
                }
            },
            _ => differences += 1,
        }
        cases += 1;
    }
    println!("{}", serde_json::json!({"complete":false,"cases":cases,"stages":stages,
        "refusals":refusals,"differences":differences,"native_physics_replay":false}));
    if cases == 0 || differences != 0 { std::process::exit(1); }
    Ok(())
}
