// SPDX-License-Identifier: Apache-2.0
//! Check ordinary-forward launch words against the live native RK schedule.
#[path = "../../../../../../runtime/src/binder/phys/rk_schedule.rs"]
mod rk_schedule;
use serde::Deserialize;
use std::{env, fs};

#[derive(Deserialize)]
struct Input {
    dt: f64,
    order: i64,
    acoustic_substeps: i64,
    dynamics_splits: i64,
    steps: Vec<Step>,
}
#[derive(Deserialize)]
struct Step { step: usize, events: Vec<Event> }
#[derive(Deserialize)]
struct Event { kernel: String, words: Vec<String> }

fn hex(bytes: &[u8]) -> String { bytes.iter().map(|x| format!("{x:02x}")).collect() }

fn expect(events: &[Event], pos: &mut usize, kernel: &str, words: &[Vec<u8>],
          count: &mut usize) -> Result<(), String> {
    let event = events.get(*pos).ok_or_else(|| format!("missing {kernel} launch"))?;
    if event.kernel != kernel || event.words.len() != words.len() {
        return Err(format!("event {}: expected {kernel}, got {}", *pos, event.kernel));
    }
    for (observed, expected) in event.words.iter().zip(words) {
        if *observed != hex(expected) { return Err(format!("{kernel}: launch word differs")); }
        *count += 1;
    }
    *pos += 1;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).ok_or("need prepared ordinary launch checks")?;
    let input: Input = serde_json::from_slice(&fs::read(path)?)?;
    if input.steps.is_empty() { return Err("no completed steps".into()); }
    let schedule = rk_schedule::Schedule::new(input.dt, input.order, input.acoustic_substeps,
                                               input.dynamics_splits)?;
    let (mut events_checked, mut words_checked) = (0, 0);
    for (index, step) in input.steps.iter().enumerate() {
        if step.step != index + 1 { return Err("nonconsecutive step boundaries".into()); }
        let mut pos = 0;
        for _split in 0..schedule.dynamics_splits {
            for stage in &schedule.stages {
                let stage_word = i32::try_from(stage.stage)?.to_le_bytes().to_vec();
                let count_word = i32::try_from(stage.acoustic_steps)?.to_le_bytes().to_vec();
                let large_word = (stage.large_timestep as f32).to_le_bytes().to_vec();
                let acoustic_word = (stage.acoustic_timestep as f32).to_le_bytes().to_vec();
                for small in 1..=stage.acoustic_steps {
                    let small_word = i32::try_from(small)?.to_le_bytes().to_vec();
                    expect(&step.events, &mut pos, "acoustic_ru_v841",
                           &[small_word.clone(), acoustic_word.clone()], &mut words_checked)?;
                    expect(&step.events, &mut pos, "acoustic_prepare_v841",
                           &[small_word], &mut words_checked)?;
                    expect(&step.events, &mut pos, "acoustic_column_solve_v841",
                           &[acoustic_word.clone()], &mut words_checked)?;
                }
                expect(&step.events, &mut pos, "recover_cells_f32",
                       &[stage_word, large_word], &mut words_checked)?;
                expect(&step.events, &mut pos, "recover_edges_f32",
                       &[count_word.clone()], &mut words_checked)?;
                expect(&step.events, &mut pos, "recover_interfaces_f32",
                       &[count_word], &mut words_checked)?;
            }
        }
        if pos != step.events.len() { return Err("unexpected additional launch".into()); }
        events_checked += pos;
    }
    println!("{}", serde_json::json!({"complete": false, "native_physics_replay": false,
        "steps": input.steps.len(), "launches_checked": events_checked,
        "launch_words_checked": words_checked, "differences": 0}));
    Ok(())
}
