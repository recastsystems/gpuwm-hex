// SPDX-License-Identifier: Apache-2.0
//! Compare live forward grouping observations with the same native host rule
//! used by replay. Observation data is a check, not a replay input.
use flate2::read::GzDecoder;
use serde::Deserialize;
use std::{collections::BTreeSet, env, fs::File, io::{BufRead, BufReader}};

#[path = "../../../../../../runtime/src/binder/phys/layer_groups.rs"]
mod layer_groups;

#[derive(Deserialize)]
struct ObservedGroup { layers: i64, indices: Vec<i32>, members: Option<Vec<i32>> }
#[derive(Deserialize)]
struct Observation {
    schema: String, complete: bool, step: u64, nm: i32, need_members: bool,
    counts: Vec<i32>, groups: Vec<ObservedGroup>,
}
#[derive(Deserialize)]
struct ChunkObservation {
    schema: String, complete: bool, step: u64, ncol: usize, nlayers: i64,
    memory_info_calls: Vec<[usize; 2]>, result: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let path = args.next().ok_or("usage: check_layer_groups OBSERVATION.gz [CHUNKS.gz]")?;
    let chunks = args.next();
    if args.next().is_some() { return Err("too many arguments".into()); }
    let reader = BufReader::new(GzDecoder::new(File::open(path)?));
    let (mut observations, mut overflow, mut count) = (0_u64, 0_u64, 0_u64);
    let mut steps = BTreeSet::new();
    for line in reader.lines() {
        let observed: Observation = serde_json::from_str(&line?)?;
        if observed.schema != "ad.binder.phys-layer-groups-observation/v1" || observed.complete {
            return Err("invalid observation schema".into());
        }
        let mut histogram = vec![0_i32; layer_groups::HISTOGRAM_BINS + 1];
        let base = i64::from(observed.nm) + 1;
        for &value in &observed.counts {
            let index = i64::from(value) - base;
            let bin = if (0..layer_groups::HISTOGRAM_BINS as i64).contains(&index) {
                index as usize
            } else { layer_groups::HISTOGRAM_BINS };
            histogram[bin] = histogram[bin].checked_add(1).ok_or("histogram overflow")?;
        }
        if histogram[layer_groups::HISTOGRAM_BINS] != 0 { overflow += 1; }
        // Stable sorted indices are the values checked by this offline tool;
        // actual replay obtains its sorted indices from live device work.
        let mut sorted: Vec<i32> = (0..observed.counts.len()).map(i32::try_from)
            .collect::<Result<_, _>>()?;
        sorted.sort_by_key(|&i| observed.counts[i as usize]);
        let actual = layer_groups::groups(observed.counts.len(), observed.nm, &histogram,
            Some(&observed.counts), Some(&sorted), observed.need_members)?;
        if actual.len() != observed.groups.len() { return Err("group count mismatch".into()); }
        for (computed, expected) in actual.iter().zip(&observed.groups) {
            let indices: &[i32] = match &computed.indices {
                layer_groups::GroupIndices::SortedView { offset, length } => &sorted[*offset..offset + length],
                layer_groups::GroupIndices::Explicit(values) => values,
            };
            if computed.layers != expected.layers || indices != expected.indices || computed.members != expected.members {
                return Err(format!("group mismatch in observation {observations}").into());
            }
            count += 1;
        }
        steps.insert(observed.step);
        observations += 1;
    }
    if observations == 0 { return Err("no grouping observations".into()); }
    let mut chunk_observations = 0_u64;
    if let Some(path) = chunks {
        let reader = BufReader::new(GzDecoder::new(File::open(path)?));
        for line in reader.lines() {
            let observed: ChunkObservation = serde_json::from_str(&line?)?;
            if observed.schema != "ad.binder.phys-layer-chunk-observation/v1" || observed.complete
                || observed.memory_info_calls.len() != 1 { return Err("invalid chunk observation".into()); }
            let actual = layer_groups::wrapper_chunk(observed.ncol, observed.nlayers,
                                                      observed.memory_info_calls[0][0])?;
            if actual != observed.result { return Err(format!("chunk mismatch at step {}", observed.step).into()); }
            chunk_observations += 1;
        }
        if chunk_observations == 0 { return Err("no chunk observations".into()); }
    }
    println!("{}", serde_json::json!({"observations": observations, "groups": count,
        "overflow_observations": overflow, "steps": steps, "mismatches": 0,
        "chunk_observations": chunk_observations,
        "native_physics_replay": false, "check_values_used_as_replay_inputs": false}));
    Ok(())
}
