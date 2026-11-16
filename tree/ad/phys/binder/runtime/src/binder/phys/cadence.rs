// SPDX-License-Identifier: Apache-2.0
//! Driver-level slot selection for the two frozen, exact-mode physics cases.
//!
//! This is a host scheduling rule, not the CUDA command list for each slot.
//! It must be bound to the measured case configuration before being used.
//! Source: hex pin, mpasphys/driver_v842.py:1187-1359 and
//! mpasphys/rrfs/driver_rrfs.py:806-818.  The supplied flags are live manager
//! outputs; recorded slot samples must never supply a replay's schedule.
use super::frames::Flags;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrozenCase { Hrrrv5, ConvectionPermitting }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend { CalendarFrame, HexCuda }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Call {
    pub operation: &'static str,
    pub backend: Backend,
}

/// Slot order at a step in the two pinned configurations, excluding the
/// underlying CUDA operations and any host work outside the driver calls.
/// Reject a frame that would require an unimplemented configuration branch.
pub fn calls(case: FrozenCase, flags: &Flags, pbl_due: bool) -> Result<Vec<Call>, &'static str> {
    if flags.surface { return Err("surface stream branch is not bound for frozen cases"); }
    if matches!(case, FrozenCase::Hrrrv5) && flags.convection {
        return Err("convection flag set in frozen hrrrv5 case");
    }
    // Both frozen configurations use exact mode with PBL due every step.
    // Correct-mode precipitation windows and held PBL calls need a separate
    // configured plan, not this frozen-case specialization.
    if !pbl_due { return Err("PBL skip branch is not bound for frozen cases"); }

    let mut result = vec![Call { operation: "physics_timetracker_state", backend: Backend::CalendarFrame }];
    let mut cuda = |operation| result.push(Call { operation, backend: Backend::HexCuda });
    if flags.greeness { cuda("physics_update_surface"); }
    if flags.radiation_lw || flags.radiation_sw { cuda("o3climatology_from_MPAS"); }
    cuda("allocate_forall_physics");
    cuda("MPAS_to_physics");
    if flags.radiation_lw || flags.radiation_sw {
        cuda("allocate_cloudiness");
        cuda("cloudiness");
    }
    if flags.radiation_sw {
        cuda("allocate_radiation_sw");
        cuda("radiation_sw");
    }
    if flags.radiation_lw {
        cuda("allocate_radiation_lw");
        cuda("radiation_lw");
    }
    cuda("deallocate_cloudiness");
    cuda("deallocate_radiation_sw");
    cuda("deallocate_radiation_lw");
    cuda("allocate_sfclayer");
    cuda("surface_layer");
    cuda("deallocate_sfclayer");
    cuda("allocate_lsm");
    cuda("land_surface");
    cuda("deallocate_lsm");
    cuda("allocate_seaice");
    cuda("seaice");
    cuda("deallocate_seaice");
    cuda("atmphys_sfc_diagnostics");
    cuda("allocate_pbl");
    cuda("pbl");
    cuda("deallocate_pbl");
    cuda("allocate_gwdo");
    cuda("gwdo");
    cuda("deallocate_gwdo");
    cuda("update_convection_step1");
    if flags.convection {
        cuda("allocate_convection");
        cuda("convection");
        cuda("deallocate_convection");
    }
    if matches!(case, FrozenCase::ConvectionPermitting) { cuda("update_convection_step2"); }
    cuda("deallocate_forall_physics");
    cuda("physics_get_tend");
    cuda("post_transport_prepare");
    if matches!(case, FrozenCase::ConvectionPermitting) { cuda("pre_microphysics"); }
    cuda("microphysics");
    if matches!(case, FrozenCase::ConvectionPermitting) { cuda("post_microphysics"); }
    Ok(result)
}
