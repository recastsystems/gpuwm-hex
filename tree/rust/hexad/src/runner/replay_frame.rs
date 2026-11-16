// SPDX-License-Identifier: Apache-2.0
//! Live physics clock, host scalar values and driver-slot expansion.
//! This does not certify the supplied allocations, command blocks or census.
use super::{cadence::{self, FrozenCase}, frames::{Alarm, Calendar, Frame, FrameMachine, Manager, update_state_time}, sfclay};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClockSpec { pub year: i32, pub month: u32, pub day: u32, pub seconds: u32 }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlarmSpec { pub period: u64, pub next_ring: u64 }
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Segment {
    Block { name: String },
    DriverSlots { blocks: BTreeMap<String, String> },
    /// One recorded block per step (pool rotation gives each step its own device targets;
    /// the fixed-length physics plan records each step, as the passed dry plan does).
    StepBlocks { blocks: Vec<String> },
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicsPlan {
    pub case: String,
    /// A distinct logical carried allocation for the live host frame.
    pub frame_state_id: u64,
    pub clock: ClockSpec,
    pub dt_seconds: u64,
    pub elapsed_seconds: u64,
    pub next_step: u64,
    pub last_time: Option<u64>,
    pub active: [bool; 3],
    pub correct_rp6: bool,
    pub pbl_due: bool,
    pub alarms: BTreeMap<String, AlarmSpec>,
    pub state_time_f32_bits: u32,
    pub state_xtime: String,
    #[serde(default)] pub mynn_sfc: Option<sfclay::Config>,
    /// Output stream intervals (seconds) whose end-of-step write reads the physics pools
    /// (runtime_v842.py:870 physics_pool_readers_due). Used by the tempo_radii_due_i32 scalar.
    #[serde(default)] pub pool_reader_output_seconds: Vec<u64>,
    pub max_steps: usize,
    /// Steps of the RECORDED forecast, when the plan is a prefix of a longer recording (a 60-step pair resealed to
    /// 29 steps). Absent: the plan holds the whole recorded run (run_steps = max_steps). See run_steps_of.
    #[serde(default)] pub run_steps: Option<usize>,
    pub segments: Vec<Segment>,
}
#[derive(Clone)]
struct Pending {
    frame: Frame,
    after: FrameMachine,
    clock_before: Calendar,
    calendar_done: bool,
    time_done: bool,
}
#[derive(Clone)]
pub struct ReplayFrame {
    case: FrozenCase,
    machine: FrameMachine,
    time_f32_bits: u32,
    xtime: String,
    pbl_due: bool,
    mynn_sfc: Option<sfclay::Config>,
    pending: Option<Pending>,
    /// The recorded forecast's last step (run_steps_of): the step whose end reads the physics pools.
    run_steps: usize,
    pool_reader_output_seconds: Vec<u64>,
}
/// g1 step29: the step count of the RECORDED forecast. The plan's own word (physics.run_steps) wins; for a
/// package sealed before that word existed, AD_PLAN_RUN_STEPS names it on the command line; else the plan holds the
/// whole run. A value below the plan's length cannot be a recorded run that contains the plan: refused.
/// FIXES: pk60e29p-20230331 (29 steps resealed from a 60-step recording), every 29-step replay: "state hash mismatch
/// at step 29, 2 allocations, first 395". The recorded model computes TEMPO's effective radii only when something
/// reads them (tempo_v303.py radii_due: radiation next step, a due output, or the RUN's last step). Step 29 of a
/// 60-step run is none of these, so the recording left the three radii profile buffers at their cleared zeros
/// (allocation 395 bytes 39,322,112.. and 78,644,224.., allocation 424 bytes 0.., 38,666,712 bytes each). The replay
/// took the plan's length for the run's length, set tempo_radii_due_i32 to 1 at step 29 and wrote radii there.
pub(crate) fn run_steps_of(plan_word: Option<usize>, switch: Option<&str>, max_steps: usize) -> Result<usize, String> {
    let run = match (plan_word, switch) {
        (Some(n), _) => n,
        (None, Some(v)) => v.trim().parse::<usize>().map_err(|_| "AD_PLAN_RUN_STEPS must be a step count".to_string())?,
        (None, None) => max_steps,
    };
    if run < max_steps {
        return Err(format!("recorded run of {run} steps is shorter than the plan's {max_steps} steps"));
    }
    Ok(run)
}
impl ReplayFrame {
    pub fn new(plan: &PhysicsPlan) -> Result<Self, String> {
        let case = match plan.case.as_str() {
            "hrrrv5" => FrozenCase::Hrrrv5,
            "convection_permitting" => FrozenCase::ConvectionPermitting,
            _ => return Err("unknown physics configuration".into()),
        };
        if plan.max_steps == 0 || plan.segments.is_empty() || plan.next_step == 0 {
            return Err("empty physics schedule or invalid first step".into());
        }
        if !f32::from_bits(plan.state_time_f32_bits).is_finite() { return Err("invalid initial state Time".into()); }
        let clock = Calendar { year: plan.clock.year, month: plan.clock.month,
            day: plan.clock.day, seconds: plan.clock.seconds };
        clock.validate().map_err(str::to_owned)?;
        if plan.state_xtime.is_empty() || !plan.state_xtime.is_ascii() {
            return Err("invalid initial state xtime".into());
        }
        let alarms = plan.alarms.iter().map(|(k, a)|
            (k.clone(), Alarm { period: a.period, next_ring: a.next_ring })).collect();
        let switch = std::env::var("AD_PLAN_RUN_STEPS").ok();
        let run_steps = run_steps_of(plan.run_steps, switch.as_deref(), plan.max_steps)?;
        if run_steps != plan.max_steps {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| println!("{}", serde_json::json!({"event":"physics_run_steps","plan_steps":plan.max_steps,
                "run_steps":run_steps,"source":if plan.run_steps.is_some() { "plan physics.run_steps" } else { "AD_PLAN_RUN_STEPS" }})));
        }
        Ok(Self { case, machine: FrameMachine { clock,
            manager: Manager { dt: plan.dt_seconds, alarms, last_time: plan.last_time },
            elapsed_seconds: plan.elapsed_seconds, next_step: plan.next_step,
            active: plan.active, correct_rp6: plan.correct_rp6 },
            time_f32_bits: plan.state_time_f32_bits, xtime: plan.state_xtime.clone(),
            pbl_due: plan.pbl_due, mynn_sfc: plan.mynn_sfc.clone(), pending: None,
            run_steps, pool_reader_output_seconds: plan.pool_reader_output_seconds.clone() })
    }
    pub fn begin(&mut self, step: usize) -> Result<(), String> {
        if self.pending.is_some() { return Err("previous physics frame is unfinished".into()); }
        if self.machine.next_step != step as u64 { return Err("physics step counter differs".into()); }
        let mut after = self.machine.clone();
        let frame = after.next().map_err(str::to_owned)?;
        self.pending = Some(Pending { frame, after, clock_before: self.machine.clock,
            calendar_done: false, time_done: false });
        Ok(())
    }
    /// integ8: the blocks of steps `step`, `step + 1`, ... (at most `k`; `step` is the next step to begin), read on a
    /// clone that advances its machine as the step's host boundaries do. Blob prefetch lookahead only, never executed;
    /// it stops at the first step it cannot tell.
    pub fn blocks_ahead(&self, plan: &PhysicsPlan, step: usize, k: usize) -> Vec<Vec<String>> {
        let mut g = self.clone();
        let mut out = Vec::new();
        for i in 0..k {
            if g.begin(step + i).is_err() { break; }
            match g.blocks(plan) { Ok(b) => out.push(b), Err(_) => break }
            let p = g.pending.take().unwrap();
            g.machine = p.after;
            g.pbl_due = p.frame.pbl_due;
        }
        out
    }
    pub fn blocks(&self, plan: &PhysicsPlan) -> Result<Vec<String>, String> {
        let pending = self.pending.as_ref().ok_or("physics frame not started")?;
        let mut result = Vec::new();
        for segment in &plan.segments {
            match segment {
                Segment::Block { name } => result.push(name.clone()),
                Segment::StepBlocks { blocks } => {
                    let step = usize::try_from(pending.frame.step).map_err(|_| "step overflow")?;
                    result.push(blocks.get(step.wrapping_sub(1)).ok_or("no recorded block for this step")?.clone());
                },
                Segment::DriverSlots { blocks } => {
                    for call in cadence::calls(self.case, &pending.frame.flags, pending.frame.pbl_due).map_err(str::to_owned)? {
                        result.push(blocks.get(call.operation)
                            .ok_or_else(|| format!("missing physics slot block {}", call.operation))?.clone());
                    }
                },
            }
        }
        if result.is_empty() { return Err("empty live physics schedule".into()); }
        Ok(result)
    }
    pub fn host_call(&mut self, name: &str) -> Result<(), String> {
        let pending = self.pending.as_mut().ok_or("physics frame not started")?;
        match name {
            "physics_timetracker_state" if !pending.calendar_done => {
                pending.calendar_done = true;
                self.machine = pending.after.clone();
            },
            // The pinned outer loop writes level 1 from level 1 and advances
            // the clock immediately afterward (runtime_v842.py:2100-2103).
            "update_state_time" if pending.calendar_done && !pending.time_done => {
                let dt = u32::try_from(self.machine.manager.dt).map_err(|_| "timestep overflow")?;
                let next = update_state_time(pending.clock_before, self.time_f32_bits, dt, false)
                    .map_err(str::to_owned)?;
                self.time_f32_bits = next.time_f32_bits;
                self.xtime = next.xtime;
                pending.time_done = true;
            },
            _ => return Err(format!("unknown, repeated or out-of-order host boundary {name}")),
        }
        Ok(())
    }
    pub fn scalar(&self, name: &str) -> Result<Vec<u8>, String> {
        let p = self.pending.as_ref().ok_or("physics frame not started")?;
        let f = &p.frame;
        let word = match name {
            "itimestep_i32" => i32::try_from(f.step).map_err(|_| "step overflow")?.to_le_bytes().to_vec(),
            "elapsed_f32" => (f.elapsed_seconds as f32).to_le_bytes().to_vec(),
            "elapsed_f64" => (f.elapsed_seconds as f64).to_le_bytes().to_vec(),
            "julian_f32" => f.julian_f32_bits.to_le_bytes().to_vec(),
            "year_i32" => f.year.to_le_bytes().to_vec(),
            "month_i32" => (f.month as i32).to_le_bytes().to_vec(),
            "day_i32" => (f.day as i32).to_le_bytes().to_vec(),
            "day_of_year_i32" => (f.day_of_year as i32).to_le_bytes().to_vec(),
            "leapyear_i32" => i32::from(f.leapyear).to_le_bytes().to_vec(),
            "pbl_due_i32" => i32::from(f.pbl_due).to_le_bytes().to_vec(),
            // TEMPO effective radii have a reader (tempo_v303.py:4940 radii_due, :5022): the NEXT step runs
            // radiation, or something reads the pools at the end of THIS step (runtime_v842.py:870: the run's
            // last step or a due output stream). No moving or cycling publication in the binder recipe.
            // The run's last step is the RECORDED run's (run_steps), which a prefix plan does not reach.
            "tempo_radii_due_i32" => {
                let mut peek = p.after.clone();
                let next = peek.next().map_err(str::to_owned)?;
                let end = p.after.elapsed_seconds;
                let output = self.pool_reader_output_seconds.iter().any(|&s| s > 0 && end % s == 0);
                let last = usize::try_from(f.step).map_err(|_| "step overflow")? == self.run_steps;
                i32::from(next.flags.radiation_lw || next.flags.radiation_sw || last || output).to_le_bytes().to_vec()
            },
            "state_time_f32" => self.time_f32_bits.to_le_bytes().to_vec(),
            "dt_f32" => (self.machine.manager.dt as f32).to_le_bytes().to_vec(),
            "current_date" => f.date.as_bytes().to_vec(),
            "state_xtime" => self.xtime.as_bytes().to_vec(),
            "mynn_sfc_resident_options" => sfclay::resident_options(
                self.mynn_sfc.as_ref().ok_or("MYNN-SFC configuration unavailable")?),
            "mynn_sfc_initflag_i32" => sfclay::initflag(
                self.mynn_sfc.as_ref().ok_or("MYNN-SFC configuration unavailable")?, f.step)?.to_le_bytes().to_vec(),
            _ if name.starts_with("flag_") => {
                let flag = match &name[5..] {
                    "radiation_lw" => f.flags.radiation_lw, "radiation_sw" => f.flags.radiation_sw,
                    "convection" => f.flags.convection, "greeness" => f.flags.greeness,
                    "surface" => f.flags.surface, "cam_abs" => f.flags.cam_abs,
                    "cam_restart" => f.flags.cam_restart, "rain_bucket" => f.flags.rain_bucket,
                    "radiation_bucket" => f.flags.radiation_bucket, "diagnostics" => f.flags.diagnostics,
                    _ => return Err(format!("unknown physics flag {name}")),
                };
                i32::from(flag).to_le_bytes().to_vec()
            },
            _ if name.starts_with("scheme_dt_f32:") => {
                let slot = &name[14..];
                let dt = self.machine.manager.scheme_timestep(slot, self.machine.correct_rp6).map_err(str::to_owned)?;
                (dt as f32).to_le_bytes().to_vec()
            },
            _ => return Err(format!("unknown physics frame scalar {name}")),
        };
        Ok(word)
    }
    pub fn finish(&mut self) -> Result<(), String> {
        let p = self.pending.as_ref().ok_or("physics frame not started")?;
        if !p.calendar_done || !p.time_done { return Err("required physics host boundary missing".into()); }
        self.pbl_due = p.frame.pbl_due;
        self.pending = None;
        Ok(())
    }

    /// Canonical bytes for the live framework clock, alarm phases and state
    /// time. This logical allocation is for checkpointing and comparison;
    /// it does not claim to enumerate the remaining carried physics state.
    pub fn snapshot(&self) -> Result<Vec<u8>, String> {
        if self.pending.is_some() { return Err("cannot snapshot an unfinished physics frame".into()); }
        let mut out = b"ADFRAME1".to_vec();
        let mut word = |bytes: &[u8]| out.extend_from_slice(bytes);
        let m = &self.machine;
        word(&m.clock.year.to_le_bytes());
        for v in [m.clock.month, m.clock.day, m.clock.seconds] { word(&v.to_le_bytes()); }
        for v in [m.elapsed_seconds, m.next_step, m.manager.dt] { word(&v.to_le_bytes()); }
        word(&u32::from(m.manager.last_time.is_some()).to_le_bytes());
        word(&m.manager.last_time.unwrap_or(0).to_le_bytes());
        let active = m.active.iter().enumerate().fold(0_u32, |bits, (i, &v)| bits | (u32::from(v) << i));
        for v in [active, u32::from(m.correct_rp6), u32::from(self.pbl_due)] { word(&v.to_le_bytes()); }
        word(&u32::try_from(m.manager.alarms.len()).map_err(|_| "too many alarms")?.to_le_bytes());
        for (name, alarm) in &m.manager.alarms {
            if !name.is_ascii() || name.is_empty() { return Err("invalid alarm name".into()); }
            word(&u32::try_from(name.len()).map_err(|_| "alarm name too long")?.to_le_bytes());
            word(name.as_bytes());
            word(&alarm.period.to_le_bytes());
            word(&alarm.next_ring.to_le_bytes());
        }
        word(&self.time_f32_bits.to_le_bytes());
        word(&u32::try_from(self.xtime.len()).map_err(|_| "xtime too long")?.to_le_bytes());
        word(self.xtime.as_bytes());
        Ok(out)
    }

    /// Restore only at the same completed boundary. Parse and validate the
    /// complete snapshot before changing any frame state.
    pub fn restore_snapshot(&mut self, bytes: &[u8]) -> Result<(), String> {
        if self.pending.is_some() { return Err("cannot restore an unfinished physics frame".into()); }
        struct Reader<'a> { bytes: &'a [u8], pos: usize }
        impl<'a> Reader<'a> {
            fn take(&mut self, len: usize) -> Result<&'a [u8], String> {
                let end = self.pos.checked_add(len).ok_or("frame offset overflow")?;
                let result = self.bytes.get(self.pos..end).ok_or("truncated frame snapshot")?;
                self.pos = end;
                Ok(result)
            }
            fn u32(&mut self) -> Result<u32, String> {
                Ok(u32::from_le_bytes(self.take(4)?.try_into().map_err(|_| "frame u32")?))
            }
            fn u64(&mut self) -> Result<u64, String> {
                Ok(u64::from_le_bytes(self.take(8)?.try_into().map_err(|_| "frame u64")?))
            }
            fn text(&mut self) -> Result<String, String> {
                let len = usize::try_from(self.u32()?).map_err(|_| "frame text length")?;
                let text = self.take(len)?;
                if text.is_empty() || !text.is_ascii() { return Err("invalid frame text".into()); }
                String::from_utf8(text.to_vec()).map_err(|_| "invalid frame text".into())
            }
        }
        let mut input = Reader { bytes, pos: 0 };
        if input.take(8)? != b"ADFRAME1" { return Err("invalid frame snapshot magic".into()); }
        let clock = Calendar { year: i32::from_le_bytes(input.take(4)?.try_into().map_err(|_| "frame year")?),
            month: input.u32()?, day: input.u32()?, seconds: input.u32()? };
        clock.validate().map_err(str::to_owned)?;
        let elapsed = input.u64()?;
        let next_step = input.u64()?;
        let dt = input.u64()?;
        let last_flag = input.u32()?;
        let last_value = input.u64()?;
        let last_time = match last_flag {
            0 if last_value == 0 => None,
            1 => Some(last_value),
            _ => return Err("invalid last-time frame encoding".into()),
        };
        let active = input.u32()?;
        let correct = input.u32()?;
        let pbl = input.u32()?;
        let expected_active = self.machine.active.iter().enumerate()
            .fold(0_u32, |bits, (i, &value)| bits | (u32::from(value) << i));
        if clock != self.machine.clock || elapsed != self.machine.elapsed_seconds
            || next_step != self.machine.next_step || dt != self.machine.manager.dt
            || active != expected_active || correct != u32::from(self.machine.correct_rp6)
            || pbl > 1 {
            return Err("checkpoint changed physics replay boundary or configuration".into());
        }
        let count = usize::try_from(input.u32()?).map_err(|_| "alarm count overflow")?;
        if count != self.machine.manager.alarms.len() { return Err("checkpoint alarm count differs".into()); }
        let mut alarms = BTreeMap::new();
        for _ in 0..count {
            let name = input.text()?;
            let alarm = Alarm { period: input.u64()?, next_ring: input.u64()? };
            if self.machine.manager.alarms.get(&name).is_none_or(|old| old.period != alarm.period)
                || alarms.insert(name, alarm).is_some() {
                return Err("checkpoint alarm identity or period differs".into());
            }
        }
        let time = input.u32()?;
        let xtime = input.text()?;
        if !f32::from_bits(time).is_finite() || input.pos != bytes.len() {
            return Err("invalid time or trailing frame snapshot bytes".into());
        }
        self.machine.manager.alarms = alarms;
        self.machine.manager.last_time = last_time;
        self.time_f32_bits = time;
        self.xtime = xtime;
        self.pbl_due = pbl != 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn observed_plan(case: &str, sample: &str) -> (PhysicsPlan, Vec<Value>) {
        let rows: Vec<Value> = sample.lines().map(|s| serde_json::from_str(s).unwrap()).collect();
        let manager = &rows[0];
        let clock = &rows[1]["clock_before"];
        let number = |v: &Value| v.as_str().unwrap().parse::<u64>().unwrap();
        let alarms: BTreeMap<String, Value> = manager["alarms_before"].as_object().unwrap().iter()
            .map(|(key, v)| (key.clone(), json!({"period":number(&v["period"]),"next_ring":number(&v["next_ring"])})))
            .collect();
        let a = &manager["active"];
        let p: PhysicsPlan = serde_json::from_value(json!({
            "case":case,"frame_state_id":999,"clock":{"year":clock["year"],"month":clock["month"],"day":clock["day"],"seconds":number(&clock["seconds"])},
            "dt_seconds":number(&manager["dt"]),"elapsed_seconds":number(&clock["elapsed"]),
            "next_step":1,"last_time":null,"active":[a["radiation_lw"],a["radiation_sw"],a["convection"]],
            "correct_rp6":manager["correct_rp6"],"pbl_due":true,"alarms":alarms,"state_time_f32_bits":0,
            "state_xtime":clock["stamp"],
            "max_steps":72,"segments":[{"kind":"block","name":"start"}] })).unwrap();
        (p, rows)
    }
    fn verify(case: &str, sample: &str) {
        let (plan, rows) = observed_plan(case, sample);
        let mut frame = ReplayFrame::new(&plan).unwrap();
        assert!(frame.snapshot().unwrap().starts_with(b"ADFRAME1"));
        for step in 1..=72 {
            let m = &rows[(step - 1) * 3];
            let c = &rows[(step - 1) * 3 + 1];
            frame.begin(step).unwrap();
            assert!(frame.snapshot().is_err());
            assert_eq!(frame.blocks(&plan).unwrap(), ["start"]);
            assert_eq!(frame.scalar("current_date").unwrap(), c["result"]["current_date"].as_str().unwrap().as_bytes());
            let hex: String = frame.scalar("julian_f32").unwrap().iter().map(|v| format!("{v:02x}")).collect();
            assert_eq!(hex, c["result"]["julian_f32_le"]);
            for key in ["radiation_lw","radiation_sw","convection","greeness","surface","cam_abs","cam_restart","rain_bucket","radiation_bucket","diagnostics"] {
                let actual = i32::from_le_bytes(frame.scalar(&format!("flag_{key}")).unwrap().try_into().unwrap());
                assert_eq!(actual != 0, m["flags"][key].as_bool().unwrap());
            }
            assert!(frame.finish().is_err());
            assert!(frame.host_call("update_state_time").is_err());
            frame.host_call("physics_timetracker_state").unwrap();
            assert!(frame.host_call("physics_timetracker_state").is_err());
            frame.host_call("update_state_time").unwrap();
            frame.finish().unwrap();
            assert!(frame.snapshot().unwrap().starts_with(b"ADFRAME1"));
        }
    }
    #[test]
    fn hrrrv5_observed_live_frames() {
        verify("hrrrv5", include_str!("../../tests/data/binder/hrrrv5.frames.sample.jsonl"));
    }
    #[test]
    fn convection_observed_live_frames() {
        verify("convection_permitting", include_str!("../../tests/data/binder/convection_permitting.frames.sample.jsonl"));
    }
    /// g1 step29: tempo_radii_due_i32 at every step of a `steps`-step plan of the hrrrv5 sample.
    fn radii_due(plan: &PhysicsPlan, steps: usize) -> Vec<i32> {
        let mut frame = ReplayFrame::new(plan).unwrap();
        (1..=steps).map(|step| {
            frame.begin(step).unwrap();
            let word = i32::from_le_bytes(frame.scalar("tempo_radii_due_i32").unwrap().try_into().unwrap());
            frame.host_call("physics_timetracker_state").unwrap();
            frame.host_call("update_state_time").unwrap();
            frame.finish().unwrap();
            word
        }).collect()
    }
    /// From the sample's own rows: whether step `step` + 1 runs radiation.
    fn radiation_next(rows: &[Value], step: usize) -> bool {
        let m = &rows[step * 3];
        m["flags"]["radiation_lw"].as_bool().unwrap() || m["flags"]["radiation_sw"].as_bool().unwrap()
    }
    #[test]
    fn prefix_plan_last_step_is_not_the_recorded_runs_last_step() {
        let (mut plan, rows) = observed_plan("hrrrv5", include_str!("../../tests/data/binder/hrrrv5.frames.sample.jsonl"));
        // a prefix length whose next step runs no radiation, so only the last-step rule can ask for the radii
        let n = (2..40).find(|&s| !radiation_next(&rows, s)).unwrap();
        // the whole recorded run in the plan: its last step reads the pools, the radii are due (as before the fix)
        plan.max_steps = n; plan.run_steps = None;
        let whole = radii_due(&plan, n);
        assert_eq!(whole[n - 1], 1);
        // the same plan as a prefix of a longer recording: step n is an ordinary step, nothing reads the radii.
        // Without the fix the frame has no run length and this word is 1 (pk60e29p step 29).
        plan.run_steps = Some(n + 31);
        let prefix = radii_due(&plan, n);
        assert_eq!(prefix[n - 1], 0);
        // nothing else moves: every earlier step's word is the same, and radiation next step still asks for them
        assert_eq!(prefix[..n - 1], whole[..n - 1]);
        for s in 1..n { assert_eq!(prefix[s - 1] != 0, radiation_next(&rows, s), "step {s}"); }
        // a run that ends inside the plan cannot be the recorded one
        plan.run_steps = Some(n - 1);
        assert!(ReplayFrame::new(&plan).is_err());
    }
    #[test]
    fn run_steps_plan_word_wins_and_bad_switch_is_refused() {
        assert_eq!(run_steps_of(None, None, 29).unwrap(), 29);
        assert_eq!(run_steps_of(None, Some("60"), 29).unwrap(), 60);
        assert_eq!(run_steps_of(Some(60), Some("31"), 29).unwrap(), 60);
        assert!(run_steps_of(None, Some("28"), 29).is_err());
        assert!(run_steps_of(None, Some("sixty"), 29).is_err());
        assert!(run_steps_of(Some(28), None, 29).is_err());
    }
    #[test]
    fn unknown_scalars_and_hosts_are_rejected() {
        let (plan, _) = observed_plan("hrrrv5", include_str!("../../tests/data/binder/hrrrv5.frames.sample.jsonl"));
        let mut frame = ReplayFrame::new(&plan).unwrap();
        frame.begin(1).unwrap();
        assert!(frame.scalar("unknown").is_err());
        assert!(frame.host_call("unimplemented_physics").is_err());
    }

    #[test]
    fn checkpoint_restores_live_frame_and_rejects_invalid_snapshots_atomically() {
        let (plan, _) = observed_plan("hrrrv5", include_str!("../../tests/data/binder/hrrrv5.frames.sample.jsonl"));
        let mut frame = ReplayFrame::new(&plan).unwrap();
        frame.begin(1).unwrap();
        let initial = frame.snapshot();
        assert!(initial.is_err());
        assert!(frame.restore_snapshot(b"ADFRAME1").is_err());
        frame.host_call("physics_timetracker_state").unwrap();
        frame.host_call("update_state_time").unwrap();
        frame.finish().unwrap();
        let saved = frame.snapshot().unwrap();
        frame.pbl_due = !frame.pbl_due;
        frame.machine.manager.last_time = None;
        frame.machine.manager.alarms.values_mut().next().unwrap().next_ring += 1;
        frame.time_f32_bits = 4.0_f32.to_bits();
        assert_ne!(frame.snapshot().unwrap(), saved);
        frame.restore_snapshot(&saved).unwrap();
        assert_eq!(frame.snapshot().unwrap(), saved);
        let mut wrong_boundary = saved.clone();
        wrong_boundary[32] ^= 1; // next_step
        let mut wrong_magic = saved.clone();
        wrong_magic[0] ^= 1;
        let mut trailing = saved.clone();
        trailing.push(0);
        for invalid in [&wrong_boundary[..], &wrong_magic[..], &trailing[..], &saved[..saved.len()-1]] {
            assert!(frame.restore_snapshot(invalid).is_err());
            assert_eq!(frame.snapshot().unwrap(), saved);
        }
    }

    fn verify_snapshot(case: &str, sample: &str, expected: &str) {
        let (plan, _) = observed_plan(case, sample);
        let mut frame = ReplayFrame::new(&plan).unwrap();
        let initial_len = frame.snapshot().unwrap().len();
        for row in expected.lines().map(|line| serde_json::from_str::<Value>(line).unwrap()) {
            let step = row["step"].as_u64().unwrap() as usize;
            frame.begin(step).unwrap();
            frame.host_call("physics_timetracker_state").unwrap();
            frame.host_call("update_state_time").unwrap();
            frame.finish().unwrap();
            let bytes = frame.snapshot().unwrap();
            assert_eq!(bytes.len(), initial_len);
            let hex: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
            assert_eq!(hex, row["snapshot_hex"].as_str().unwrap(), "frame step {step}");
        }
    }
    #[test]
    fn hrrrv5_frame_state_matches_completed_observations() {
        verify_snapshot("hrrrv5", include_str!("../../tests/data/binder/hrrrv5.frames.sample.jsonl"),
            include_str!("../../tests/data/binder/hrrrv5.frame-state.sample.jsonl"));
    }
    #[test]
    fn convection_frame_state_matches_completed_observations() {
        verify_snapshot("convection_permitting", include_str!("../../tests/data/binder/convection_permitting.frames.sample.jsonl"),
            include_str!("../../tests/data/binder/convection_permitting.frame-state.sample.jsonl"));
    }
}
