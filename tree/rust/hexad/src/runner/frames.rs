// SPDX-License-Identifier: Apache-2.0
//! Partial integral-second Gregorian clock and physics alarms. The forecast
//! binder must separately establish the initial state and every other frame
//! scalar; fractional intervals and non-Gregorian calendars are not accepted.
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Calendar {
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub seconds: u32,
}

fn leap(year: i32) -> bool { year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) }
fn month_days(year: i32, month: u32) -> Option<u32> {
    Some(match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => if leap(year) { 29 } else { 28 },
        _ => return None,
    })
}

impl Calendar {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.year < 1 || self.day == 0 || self.day > month_days(self.year, self.month).ok_or("month")?
            || self.seconds >= 86_400 { return Err("invalid Gregorian date"); }
        Ok(())
    }
    pub fn advance(&mut self, dt: u32) -> Result<(), &'static str> {
        self.validate()?;
        let total = u64::from(self.seconds) + u64::from(dt);
        self.seconds = (total % 86_400) as u32;
        for _ in 0..total / 86_400 {
            self.day += 1;
            if self.day > month_days(self.year, self.month).ok_or("month")? {
                self.day = 1;
                self.month += 1;
                if self.month == 13 {
                    self.month = 1;
                    self.year = self.year.checked_add(1).ok_or("year overflow")?;
                }
            }
        }
        Ok(())
    }
    pub fn words(&self) -> Result<(String, u32, bool), &'static str> {
        self.validate()?;
        let h = self.seconds / 3600;
        let minute = (self.seconds % 3600) / 60;
        let s = self.seconds % 60;
        let day_of_year = (1..self.month).map(|m| month_days(self.year, m).unwrap()).sum::<u32>() + self.day;
        // Match the pinned RKIND operations in framework_calendar_v842.py.
        let utc_hour = (h as f32 + (minute as f32 / 60.0_f32)) + (s as f32 / 3600.0_f32);
        let julian = (day_of_year - 1) as f32 + utc_hour / 24.0_f32;
        Ok((format!("{:04}-{:02}-{:02}_{:02}:{:02}:{:02}", self.year, self.month, self.day, h, minute, s),
            julian.to_bits(), leap(self.year)))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Alarm { pub period: u64, pub next_ring: u64 }

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Flags {
    pub radiation_lw: bool, pub radiation_sw: bool, pub convection: bool,
    pub greeness: bool, pub surface: bool, pub cam_abs: bool, pub cam_restart: bool,
    pub rain_bucket: bool, pub radiation_bucket: bool, pub diagnostics: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Manager {
    pub dt: u64,
    pub alarms: BTreeMap<String, Alarm>,
    pub last_time: Option<u64>,
}

/// Calendar and manager words used by the ported framework at a step's
/// `physics_timetracker_state` boundary. The flag order is the eight arguments
/// written by FrameworkPhysics.calendar_state, not a native-library ABI.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Frame {
    pub step: u64,
    pub elapsed_seconds: u64,
    pub date: String,
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub day_of_year: u32,
    pub julian_f32_bits: u32,
    pub leapyear: bool,
    pub flags: Flags,
    pub manager_flag_words: [i32; 8],
    pub pbl_due: bool,
}

/// Host pool time words for the integral-second Gregorian route. The source
/// word comes from the live state, not from the ordinary-forward recording.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateTime {
    pub xtime: String,
    pub time_f32_bits: u32,
}

pub fn update_state_time(clock: Calendar, source_f32_bits: u32, dt_seconds: u32,
                         clock_advanced: bool) -> Result<StateTime, &'static str> {
    let source = f32::from_bits(source_f32_bits);
    let dt = dt_seconds as f32;
    if !source.is_finite() || f64::from(dt) != f64::from(dt_seconds) {
        return Err("unsupported state time or timestep");
    }
    let sum = source + dt;
    if !sum.is_finite() { return Err("state time overflow"); }
    let mut target = clock;
    if !clock_advanced { target.advance(dt_seconds)?; }
    Ok(StateTime { xtime: target.words()?.0, time_f32_bits: sum.to_bits() })
}

/// Mutable frame state is part of a checkpoint, including alarm phases. This
/// deliberately supports only the integral-second Gregorian configurations
/// observed in the two samples. Other configurations must be refused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrameMachine {
    pub clock: Calendar,
    pub manager: Manager,
    pub elapsed_seconds: u64,
    pub next_step: u64,
    pub active: [bool; 3],
    pub correct_rp6: bool,
}

impl FrameMachine {
    /// Produce live frame values and advance the clock exactly once. A failed
    /// operation leaves every state word unchanged, so it cannot partially
    /// advance a replay that has to be refused.
    pub fn next(&mut self) -> Result<Frame, &'static str> {
        if self.next_step == 0 || self.manager.dt == 0 { return Err("invalid frame state"); }
        let dt: u32 = self.manager.dt.try_into().map_err(|_| "dt exceeds u32")?;
        let elapsed_after = self.elapsed_seconds.checked_add(self.manager.dt).ok_or("time overflow")?;
        let step_after = self.next_step.checked_add(1).ok_or("step overflow")?;
        let mut clock = self.clock;
        let mut manager = self.manager.clone();
        let (flags, pbl_due) = manager.advance(self.elapsed_seconds, self.active, self.correct_rp6)?;
        let (date, julian_f32_bits, leapyear) = clock.words()?;
        let day_of_year = (1..clock.month).map(|m| month_days(clock.year, m).unwrap()).sum::<u32>() + clock.day;
        let manager_flag_words = [flags.radiation_lw, flags.radiation_sw, flags.convection,
            flags.cam_abs, flags.cam_restart, flags.rain_bucket, flags.radiation_bucket,
            flags.diagnostics].map(i32::from);
        let frame = Frame { step: self.next_step, elapsed_seconds: self.elapsed_seconds,
            date, year: clock.year, month: clock.month, day: clock.day, day_of_year,
            julian_f32_bits, leapyear, flags, manager_flag_words, pbl_due };
        clock.advance(dt)?;
        self.clock = clock;
        self.manager = manager;
        self.elapsed_seconds = elapsed_after;
        self.next_step = step_after;
        Ok(frame)
    }
}

impl Manager {
    /// PhysicsManager.scheme_dt at the pinned driver. The selected slot
    /// receives the named interval or the dynamics timestep. Alarm phases do
    /// not affect the interval, but the RP6 choice affects land-surface dt.
    pub fn scheme_timestep(&self, slot: &str, correct_rp6: bool) -> Result<u64, &'static str> {
        if self.dt == 0 { return Err("zero timestep"); }
        let key = match slot {
            "radiation_lw" | "radiation_sw" | "convection" | "pbl" => Some(slot),
            "oml" | "seaice" | "gwdo" | "ngw" => Some("pbl"),
            "land_surface" if correct_rp6 => Some("pbl"),
            _ => None,
        };
        let dt = key.and_then(|k| self.alarms.get(k)).map_or(self.dt, |a| a.period);
        if dt == 0 { return Err("zero scheme timestep"); }
        Ok(dt)
    }

    fn ring(&mut self, key: &str, now: u64, ahead: bool) -> Result<bool, &'static str> {
        let Some(alarm) = self.alarms.get_mut(key) else { return Ok(false); };
        if alarm.period == 0 { return Err("zero alarm period"); }
        let target = now.checked_add(if ahead { self.dt } else { 0 }).ok_or("time overflow")?;
        if alarm.next_ring > target { return Ok(false); }
        let periods = (target - alarm.next_ring) / alarm.period + 1;
        alarm.next_ring = alarm.next_ring.checked_add(periods.checked_mul(alarm.period).ok_or("alarm overflow")?)
            .ok_or("alarm overflow")?;
        Ok(true)
    }
    pub fn advance(&mut self, now: u64, active: [bool; 3], correct_rp6: bool) -> Result<(Flags, bool), &'static str> {
        if self.dt == 0 || self.last_time.is_some_and(|last| now <= last) { return Err("non-increasing time"); }
        self.last_time = Some(now);
        let mut slot = |name: &str, enabled: bool| -> Result<bool, &'static str> {
            Ok(enabled && (!self.alarms.contains_key(name) || self.ring(name, now, false)?))
        };
        let radiation_lw = slot("radiation_lw", active[0])?;
        let radiation_sw = slot("radiation_sw", active[1])?;
        let convection = slot("convection", active[2])?;
        let greeness = self.ring("greeness", now, false)?;
        let surface = self.ring("surface", now, false)?;
        let cam_abs = self.ring("cam_abs", now, false)?;
        let cam_restart = self.ring("cam_restart", now, true)?;
        let rain_bucket = active[2] && self.ring("rain_bucket", now, true)?;
        let radiation_bucket = (active[0] || active[1]) && self.ring("radiation_bucket", now, true)?;
        self.ring("diagnostics", now, false)?;
        let target = now.checked_add(self.dt).ok_or("time overflow")?;
        let diagnostics = self.alarms.get("diagnostics").is_some_and(|a| a.next_ring <= target);
        let pbl_due = if correct_rp6 { self.ring("pbl", now, false)? } else { true };
        Ok((Flags { radiation_lw, radiation_sw, convection, greeness, surface, cam_abs,
                    cam_restart, rain_bucket, radiation_bucket, diagnostics }, pbl_due))
    }
}
