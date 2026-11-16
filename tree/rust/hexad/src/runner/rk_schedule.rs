// SPDX-License-Identifier: Apache-2.0
//! Live RK and acoustic schedule from hexcore/integration.py:109-164.
//! The caller supplies current host configuration, not recorded launch values.

#[derive(Clone, Copy, Debug)]
pub struct Stage {
    pub stage: i64,
    pub large_timestep: f64,
    pub acoustic_timestep: f64,
    pub acoustic_steps: i64,
}

#[derive(Clone, Copy, Debug)]
pub struct Schedule {
    pub full_timestep: f64,
    pub dynamics_splits: i64,
    pub stages: [Stage; 3],
}

impl Schedule {
    pub fn new(dt: f64, order: i64, acoustic_substeps: i64, dynamics_splits: i64)
        -> Result<Self, String>
    {
        if dt <= 0.0 { return Err("config_dt must be positive".into()); }
        if dynamics_splits < 1 { return Err("config_dynamics_split_steps must be positive".into()); }
        if acoustic_substeps < 1 { return Err("config_number_of_sub_steps must be positive".into()); }
        let dt_dyn = dt / dynamics_splits as f64;
        let half = (acoustic_substeps / 2).max(1);
        let (large, acoustic_dt, counts) = match order {
            3 => ([dt_dyn / 3.0, dt_dyn / 2.0, dt_dyn],
                  [dt_dyn / 3.0, dt_dyn / acoustic_substeps as f64,
                   dt_dyn / acoustic_substeps as f64],
                  [1, half, acoustic_substeps]),
            2 => ([dt_dyn / 2.0, dt_dyn / 2.0, dt_dyn],
                  [dt_dyn / acoustic_substeps as f64; 3],
                  [half, half, acoustic_substeps]),
            _ => return Err("config_time_integration_order must be 2 or 3".into()),
        };
        Ok(Self { full_timestep: dt, dynamics_splits,
            stages: std::array::from_fn(|i| Stage { stage: i as i64 + 1,
                large_timestep: large[i], acoustic_timestep: acoustic_dt[i], acoustic_steps: counts[i] }) })
    }
}

/// Frame variables exist only in their source loop scope.
#[derive(Clone)]
pub struct Scope {
    pub schedule: Schedule,
    pub split: Option<i64>,
    pub stage: Option<Stage>,
    pub small_step: Option<i64>,
}
impl Scope {
    pub fn scalar(&self, name: &str) -> Result<Vec<u8>, String> {
        let word = match name {
            "full_dt_f32" => (self.schedule.full_timestep as f32).to_le_bytes().to_vec(),
            "split_i32" => i32::try_from(self.split.ok_or("split loop not active")?)
                .map_err(|_| "split index overflow")?.to_le_bytes().to_vec(),
            "stage_i32" => i32::try_from(self.stage.ok_or("RK stage loop not active")?.stage)
                .map_err(|_| "stage index overflow")?.to_le_bytes().to_vec(),
            "acoustic_steps_i32" => i32::try_from(self.stage.ok_or("RK stage loop not active")?.acoustic_steps)
                .map_err(|_| "acoustic count overflow")?.to_le_bytes().to_vec(),
            "small_step_i32" => i32::try_from(self.small_step.ok_or("acoustic loop not active")?)
                .map_err(|_| "acoustic step overflow")?.to_le_bytes().to_vec(),
            "large_dt_f32" => (self.stage.ok_or("RK stage loop not active")?.large_timestep as f32)
                .to_le_bytes().to_vec(),
            "acoustic_dt_f32" => (self.stage.ok_or("RK stage loop not active")?.acoustic_timestep as f32)
                .to_le_bytes().to_vec(),
            _ => return Err(format!("unknown RK scalar {name}")),
        };
        Ok(word)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stages_follow_live_order_and_odd_substep_counts() {
        let s = Schedule::new(1200.0, 3, 5, 2).unwrap();
        assert_eq!(s.stages.map(|x| x.acoustic_steps), [1, 2, 5]);
        assert_eq!(s.stages.map(|x| x.large_timestep), [200.0, 300.0, 600.0]);
        assert_eq!(s.stages.map(|x| x.acoustic_timestep), [200.0, 120.0, 120.0]);
        assert_eq!(Schedule::new(1200.0, 2, 1, 1).unwrap().stages.map(|x| x.acoustic_steps), [1, 1, 1]);
    }
    #[test]
    fn invalid_configuration_and_out_of_scope_values_refuse() {
        assert!(Schedule::new(0.0, 3, 6, 1).is_err());
        assert!(Schedule::new(1.0, 1, 6, 1).is_err());
        assert!(Schedule::new(1.0, 2, 0, 1).is_err());
        assert!(Schedule::new(1.0, 2, 1, 0).is_err());
        let s = Scope { schedule: Schedule::new(1.0, 2, 1, 1).unwrap(),
                        split: None, stage: None, small_step: None };
        assert!(s.scalar("small_step_i32").is_err());
        assert!(s.scalar("stage_i32").is_err());
        assert!(s.scalar("unknown").is_err());
    }
}
