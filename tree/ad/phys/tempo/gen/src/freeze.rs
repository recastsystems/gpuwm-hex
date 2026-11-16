// SPDX-License-Identifier: Apache-2.0
//! The TEMPO closure freeze (`tempo-ad-gen --freeze closure`): statement-level points in the pinned forward
//! source (sha256 bb99eb4f..) where the active copy drops the derivative of a closure quantity and keeps its
//! value bit for bit (`ad_frz`: TL tangent 0, no tape node, so REV pushes nothing). This is the coordinator
//! ruling of 2026-11-05 08:35Z (freeze from the trajectory), applied as for the FCT limiter, the MYNN-EDMF
//! closure and the GSL drag. Every comparison still decides on the forward value; branches stay the executed
//! ones. What stays live: the masses, temperature, vapour, saturation ratios, the process tendencies'
//! explicit dependence on them, sedimentation transport and the depletion limiter.
//!
//! Each point is (forward source line, 1-based; a text that line must contain; insert before or after it;
//! freeze group bit; the statement). The generator refuses to run when a line does not hold its text, so a
//! changed source can never be frozen at the wrong statement. The groups compile behind AD_FRZ_MASK (default:
//! all), so a bisection build is `-DAD_FRZ_MASK=..` with no regeneration.
/// The production mask (groups 1, 2, 6 and 7: the number re-derivations and the snow, cloud-ice and graupel deposition
/// coefficients). Chosen by
/// the lane's 18-step traces; the other groups are kept as selectable options, the diagnostic groups (8-23) only
/// ever by an explicit --freeze-mask.
pub const DEFAULT_MASK: &str = "0xc6u";
pub const SOURCE_SHA256: &str = "bb99eb4f9aaf85642e70a482fffea5ad595b496e225873cd764646c78828cfc6";

#[derive(Clone, Copy, PartialEq)]
pub enum At { Before, After }

/// Freeze groups (bit, name, what).
pub const GROUPS: &[(u32, &str, &str)] = &[
    (0, "slope", "size-distribution slopes lam/ilam, median volume diameters mvd, intercepts n0 and the warm-rain mean sizes xdc/dc_b"),
    (1, "clamp_number", "number re-derived from mass at the size clamps (hit_limit branches of the check_and_update routines)"),
    (2, "number_rederive", "the n1d re-derivation at the end of every check_and_update (number from mass and the slope)"),
    (3, "nucleation", "DeMott and Koop ice nucleation, the Cooper fallback, droplet activation"),
    (4, "evap_closure", "the Srivastava-Coen t1_evap / t1_subl curvature factors"),
    (5, "snow_moments", "the Field et al. snow moments other than the mass moment smob"),
    (6, "snow_deposition_coefficient", "the snow deposition/sublimation rate coefficient prs_sde/ssati (moments, ventilation, curvature); ssati live"),
    (7, "ice_graupel_deposition_coefficient", "the cloud-ice and graupel deposition/sublimation coefficients pri_ide/ssati, prg_gde/ssati; ssati live"),
    (8, "diag_warm_rain", "diagnostic: warm_rain tendencies"),
    (9, "diag_rain_snow_graupel", "diagnostic: rain_snow_rain_graupel tendencies"),
    (10, "diag_ice_nucleation", "diagnostic: ice_nucleation tendencies"),
    (11, "diag_ice_processes", "diagnostic: ice_processes tendencies"),
    (12, "diag_riming", "diagnostic: riming tendencies"),
    (13, "diag_melting", "diagnostic: melting tendencies"),
    (14, "diag_condensation", "diagnostic: cloud_condensation tendencies"),
    (15, "diag_rain_evaporation", "diagnostic: rain_evaporation tendencies"),
    (16, "diag_depletion_ratio", "diagnostic: check_over_depletion ratios"),
    (17, "diag_aerosol_scavenging", "diagnostic: aerosol_scavenging tendencies"),
    (18, "diag_ice_18", "diagnostic: ice_processes subset: ice deposition pri_ide/pni_ide/prs_ide"),
    (19, "diag_ice_19", "diagnostic: ice_processes subset: snow deposition prs_sde"),
    (20, "diag_ice_20", "diagnostic: ice_processes subset: graupel deposition prg_gde/png_gde"),
    (21, "diag_ice_21", "diagnostic: ice_processes subset: ice-to-snow autoconversion prs_iau/pni_iau"),
    (22, "diag_ice_22", "diagnostic: ice_processes subset: snow collecting ice prs_sci/pni_sci"),
    (23, "diag_ice_23", "diagnostic: ice_processes subset: rain-ice collection pri_rci..pbg_rci"),
];

pub const POINTS: &[(usize, &str, At, u32, &str)] = &[
    // ---- group 0: slopes, mvd, n0, mean sizes
    // cloud_check_and_update: lamc after the size clamps (feeds nc, nc1d), and the second lamc (ilamc, mvd_c)
    (7035, "nc[k] = PA2(ccg, 1, nu_c) * PA1(ocg2, nu_c) * rc[k]", At::Before, 0, "if (AD_FRZ_ON(0)) lamc = ad_frz(lamc);"),
    (7048, "lamc = tempo_pow((nc[k] * PS(am_r)", At::After, 0, "if (AD_FRZ_ON(0)) lamc = ad_frz(lamc);"),
    // rain_check_and_update: lamr and mvd_r after the clamps (feed nr1d, ilamr)
    (7100, "nr1d[k] = PA1(crg, 2) * PS(org3) * qr1d[k]", At::Before, 0, "if (AD_FRZ_ON(0)) { lamr = ad_frz(lamr); mvd_r[k] = ad_frz(mvd_r[k]); }"),
    // ice_check_and_update: lami after the clamps (feeds ni1d, ilami)
    (7148, "ni1d[k] = tmax(PS(r2) / rho[k],", At::Before, 0, "if (AD_FRZ_ON(0)) lami = ad_frz(lami);"),
    // graupel_check_and_update: lamg (both size-distribution arms) and mvd_g
    (7224, "ng1d[k] = PA2(cgg, 2, 1) * PS(ogg3) * qg1d[k]", At::Before, 0, "if (AD_FRZ_ON(0)) lamg = ad_frz(lamg);"),
    (7236, "ng[k] = PA2(cgg, 2, 1) * PS(ogg3) * rg[k] * tpowi(lamg, 3)", At::Before, 0, "if (AD_FRZ_ON(0)) lamg = ad_frz(lamg);"),
    (7239, "ilamg[k] = 1.0f / lamg;", At::Before, 0, "if (AD_FRZ_ON(0)) lamg = ad_frz(lamg);"),
    // intercepts in the process routines (rain evaporation, warm rain, riming, ice processes, melting, scavenging)
    (5771, "n0_r = nr[k] * PS(org2) * tempo_pow(lamr, PA1(cre, 2));", At::After, 0, "if (AD_FRZ_ON(0)) n0_r = ad_frz(n0_r);"),
    (5874, "n0_r = nr[k] * PS(org2) * tempo_pow(lamr, PA1(cre, 2));", At::After, 0, "if (AD_FRZ_ON(0)) n0_r = ad_frz(n0_r);"),
    (6222, "n0_r = nr[k] * PS(org2) * tempo_pow(lamr, PA1(cre, 2));", At::After, 0, "if (AD_FRZ_ON(0)) n0_r = ad_frz(n0_r);"),
    (6378, "n0_r = nr[k] * PS(org2) * tempo_pow(lamr, PA1(cre, 2));", At::After, 0, "if (AD_FRZ_ON(0)) n0_r = ad_frz(n0_r);"),
    (5965, "n0_g = ng[k] * PS(ogg2) * tempo_pow((1.0f / ilamg[k]), PA2(cge, 2, 1));", At::After, 0, "if (AD_FRZ_ON(0)) n0_g = ad_frz(n0_g);"),
    (6254, "n0_g = ng[k] * PS(ogg2) * tempo_pow((1.0f / ilamg[k]), PA2(cge, 2, 1));", At::After, 0, "if (AD_FRZ_ON(0)) n0_g = ad_frz(n0_g);"),
    (6318, "n0_melt = ng[k] * PS(ogg2) * p2;", At::After, 0, "if (AD_FRZ_ON(0)) { n0_g = ad_frz(n0_g); n0_melt = ad_frz(n0_melt); }"),
    (6324, "n0_melt = (1.e-4f / rg[k]) * PS(ogg2) * tempo_pow(lamg, PA2(cge, 2, 1));", At::After, 0, "if (AD_FRZ_ON(0)) n0_melt = ad_frz(n0_melt);"),
    (6403, "n0_g = ng[k] * PS(ogg2) * tempo_pow((1.0f / ilamg[k]), PA2(cge, 2, 1));", At::After, 0, "if (AD_FRZ_ON(0)) n0_g = ad_frz(n0_g);"),
    // warm_rain autoconversion: mean drop size xdc and dc_b = (xdc^3 dc_g^3 - xdc^6)^(1/6) (singular as its base -> 0)
    (5851, "xdc = tmax(d0c * 1.e6f, (tempo_pow((rc[k] / (am_r * nc[k])), PS(obmr))) * 1.e6f);", At::After, 0, "if (AD_FRZ_ON(0)) xdc = ad_frz(xdc);"),
    (5854, "(1.0f / 6.0f));", At::After, 0, "if (AD_FRZ_ON(0)) dc_b = ad_frz(dc_b);"),
    // ---- group 1: number from mass at the size clamps
    (7035, "nc[k] = PA2(ccg, 1, nu_c) * PA1(ocg2, nu_c) * rc[k]", At::After, 1, "if (AD_FRZ_ON(1)) nc[k] = ad_frz(nc[k]);"),
    (7084, "nr[k] = PA1(crg, 2) * PS(org3) * rr[k] * tpowi(lamr, 3) / PS(am_r);", At::After, 1, "if (AD_FRZ_ON(1)) nr[k] = ad_frz(nr[k]);"),
    (7092, "nr[k] = PA1(crg, 2) * PS(org3) * rr[k] * tpowi(lamr, 3) / PS(am_r);", At::After, 1, "if (AD_FRZ_ON(1)) nr[k] = ad_frz(nr[k]);"),
    (7097, "nr[k] = PA1(crg, 2) * PS(org3) * rr[k] * tpowi(lamr, 3) / PS(am_r);", At::After, 1, "if (AD_FRZ_ON(1)) nr[k] = ad_frz(nr[k]);"),
    (7134, "ni[k] = tmin(PS(max_ni), PA1(cig, 1) * PS(oig2) * ri[k]", At::After, 1, "if (AD_FRZ_ON(1)) ni[k] = ad_frz(ni[k]);"),
    (7141, "ni[k] = tmin(PS(max_ni), PA1(cig, 1) * PS(oig2) * ri[k]", At::After, 1, "if (AD_FRZ_ON(1)) ni[k] = ad_frz(ni[k]);"),
    (7145, "ni[k] = PA1(cig, 1) * PS(oig2) * ri[k] / PS(am_i) * tpowi(lami, 3);", At::After, 1, "if (AD_FRZ_ON(1)) ni[k] = ad_frz(ni[k]);"),
    (7208, "ng[k] = PA2(cgg, 2, 1) * PS(ogg3) * rg[k] * tpowi(lamg, 3)", At::After, 1, "if (AD_FRZ_ON(1)) ng[k] = ad_frz(ng[k]);"),
    (7216, "ng[k] = PA2(cgg, 2, 1) * PS(ogg3) * rg[k] * tpowi(lamg, 3)", At::After, 1, "if (AD_FRZ_ON(1)) ng[k] = ad_frz(ng[k]);"),
    (7221, "ng[k] = PA2(cgg, 2, 1) * PS(ogg3) * rg[k] * tpowi(lamg, 3)", At::After, 1, "if (AD_FRZ_ON(1)) ng[k] = ad_frz(ng[k]);"),
    (7236, "ng[k] = PA2(cgg, 2, 1) * PS(ogg3) * rg[k] * tpowi(lamg, 3)", At::After, 1, "if (AD_FRZ_ON(1)) ng[k] = ad_frz(ng[k]);"),
    // ---- group 2: the n1d re-derivation at the end of each check
    (7039, "PS(nt_c_max) / rho[k]));", At::After, 2, "if (AD_FRZ_ON(2)) nc1d[k] = ad_frz(nc1d[k]);"),
    (7100, "nr1d[k] = PA1(crg, 2) * PS(org3) * qr1d[k]", At::After, 2, "if (AD_FRZ_ON(2)) nr1d[k] = ad_frz(nr1d[k]);"),
    (7149, "tmin(PA1(cig, 1) * PS(oig2) * qi1d[k] / PS(am_i) * tpowi(lami, 3), PS(max_ni) / rho[k]));", At::After, 2, "if (AD_FRZ_ON(2)) ni1d[k] = ad_frz(ni1d[k]);"),
    (7224, "ng1d[k] = PA2(cgg, 2, 1) * PS(ogg3) * qg1d[k]", At::After, 2, "if (AD_FRZ_ON(2)) ng1d[k] = ad_frz(ng1d[k]);"),
    // ---- group 3: nucleation
    (5573, "prob_h = tmin(1.0f - tempo_exp(-j_rate * PS(ar_volume) * C.global_dt), 1.0f);", At::After, 3, "if (AD_FRZ_ON(3)) prob_h = ad_frz(prob_h);"),
    (5656, "xni = xni * rho / PS(rho_not0) * 1000.0f;", At::After, 3, "if (AD_FRZ_ON(3)) xni = ad_frz(xni);"),
    (5706, "xnc = tmax(PS(nt_c_min), activate_cloud_number(P, temp[k], w1d[k], nwfa[k], nullptr));", At::After, 3, "if (AD_FRZ_ON(3)) xnc = ad_frz(xnc);"),
    (6107, "xnc = tmin(PS(icenuc_max), PS(tno) * tempo_exp(PS(ato) * (t0 - temp[k])));", At::After, 3, "if (AD_FRZ_ON(3)) xnc = ad_frz(xnc);"),
    // ---- group 4: evaporation / sublimation closure factors
    (5721, "- 5.0f * alphsc * alphsc * alphsc * xsat * xsat * xsat) / (1.0f + gamsc);", At::After, 4, "if (AD_FRZ_ON(4)) t1_evap = ad_frz(t1_evap);"),
    (5765, "- 5.0f * alphsc * alphsc * alphsc * xsat * xsat * xsat) / (1.0f + gamsc);", At::After, 4, "if (AD_FRZ_ON(4)) t1_evap = ad_frz(t1_evap);"),
    (6148, "- 5.0f * alphsc * alphsc * alphsc * xsat * xsat * xsat) / (1.0f + gamsc);", At::After, 4, "if (AD_FRZ_ON(4)) t1_subl[k] = ad_frz(t1_subl[k]);"),
    // ---- group 5: snow moments (all but the mass moment smob), at the end of snow_moments
    (5346, "#undef TEMPO_SNOW_AB", At::Before, 5, "  if (AD_FRZ_ON(5)) { *smoc = ad_frz(*smoc); if (ns) *ns = ad_frz(*ns); if (smo0) *smo0 = ad_frz(*smo0); if (smo1) *smo1 = ad_frz(*smo1); if (smo2) *smo2 = ad_frz(*smo2); if (smoe) *smoe = ad_frz(*smoe); if (smof) *smof = ad_frz(*smof); if (smog) *smog = ad_frz(*smog); if (smoz) *smoz = ad_frz(*smoz); }"),
    // ---- group 6: the snow deposition / sublimation coefficient. prs_sde = K * ssati with K = c_snow t1_subl diffu rvs
    // (t1_qs_sd smo1 + t2_qs_sd rhof2 vsc2 smof): K (the snow size-distribution moments, the ventilation and the
    // curvature factors) is frozen and the ice supersaturation ssati stays live, before the executed rate_max / -rs/dt
    // caps. Measured (lane tempo-freeze, 8-step bisection): this one term carries the step 7-8 theta growth.
    (6245, "* (PS(t1_qs_sd) * smo1[k] + PS(t2_qs_sd) * rhof2[k] * vsc2[k] * smof[k]);", At::After, 6, "if (AD_FRZ_ON(6)) T.prs_sde[k] = ad_lin(T.prs_sde[k], ssati[k]);"),
    // the same coefficient in melting (sublimation of snow above 0 C, ssati < 0)
    (6309, "* (PS(t1_qs_sd) * smo1[k] + PS(t2_qs_sd) * rhof2[k] * vsc2[k] * smof[k]);", At::After, 6, "if (AD_FRZ_ON(6)) T.prs_sde[k] = ad_lin(T.prs_sde[k], ssati[k]);"),
    // ---- group 7: the cloud-ice and graupel deposition / sublimation coefficients, the same way (ssati live):
    // pri_ide = K ni ilami ssati (ice_processes), prg_gde = K n0_g (..) ssati (ice_processes and melting).
    (6185, "* ni[k] * ilami[k];", At::After, 7, "if (AD_FRZ_ON(7)) T.pri_ide[k] = ad_lin(T.pri_ide[k], ssati[k]);"),
    (6259, "+ t2_qg_sd * vsc2[k] * rhof2[k] * tempo_pow_with(bg, PA2(cge, 11, idx[k])));", At::After, 7, "if (AD_FRZ_ON(7)) T.prg_gde[k] = ad_lin(T.prg_gde[k], ssati[k]);"),
    (6346, "+ t2_qg_sd * vsc2[k] * rhof2[k] * p11);", At::After, 7, "if (AD_FRZ_ON(7)) T.prg_gde[k] = ad_lin(T.prg_gde[k], ssati[k]);"),
    // ---- diagnostic groups 8-17 (bisection only, never in the default mask): every tendency a process routine
    // writes, frozen right after its call in tempo_main_column; 16: the depletion-limiter ratios.
    (8196, "T);", At::After, 8, "if (AD_FRZ_ON(8)) { TEMPO_LEVELS(k, nz) { T.pnc_rcw[k] = ad_frz(T.pnc_rcw[k]); T.pnc_wau[k] = ad_frz(T.pnc_wau[k]); T.pnr_rcr[k] = ad_frz(T.pnr_rcr[k]); T.pnr_wau[k] = ad_frz(T.pnr_wau[k]); T.prr_rcw[k] = ad_frz(T.prr_rcw[k]); T.prr_wau[k] = ad_frz(T.prr_wau[k]); } }"),
    (8198, "W.rg, W.ng, W.ilamg, W.idx_bg, T);", At::After, 9, "if (AD_FRZ_ON(9)) { TEMPO_LEVELS(k, nz) { T.pbg_rcg[k] = ad_frz(T.pbg_rcg[k]); T.pbg_rcs[k] = ad_frz(T.pbg_rcs[k]); T.png_rcg[k] = ad_frz(T.png_rcg[k]); T.png_rcs[k] = ad_frz(T.png_rcs[k]); T.pnr_rcg[k] = ad_frz(T.pnr_rcg[k]); T.pnr_rcs[k] = ad_frz(T.pnr_rcs[k]); T.prg_rcg[k] = ad_frz(T.prg_rcg[k]); T.prg_rcs[k] = ad_frz(T.prg_rcs[k]); T.prr_rcg[k] = ad_frz(T.prr_rcg[k]); T.prr_rcs[k] = ad_frz(T.prr_rcs[k]); T.prs_rcs[k] = ad_frz(T.prs_rcs[k]); } }"),
    (8200, "W.ni, W.smo0, W.rc, W.nc, W.rr, W.nr, W.ilamr, T);", At::After, 10, "if (AD_FRZ_ON(10)) { TEMPO_LEVELS(k, nz) { T.pbg_rfz[k] = ad_frz(T.pbg_rfz[k]); T.png_rfz[k] = ad_frz(T.png_rfz[k]); T.pni_iha[k] = ad_frz(T.pni_iha[k]); T.pni_inu[k] = ad_frz(T.pni_inu[k]); T.pni_rfz[k] = ad_frz(T.pni_rfz[k]); T.pni_wfz[k] = ad_frz(T.pni_wfz[k]); T.pnr_rfz[k] = ad_frz(T.pnr_rfz[k]); T.prg_rfz[k] = ad_frz(T.prg_rfz[k]); T.pri_iha[k] = ad_frz(T.pri_iha[k]); T.pri_inu[k] = ad_frz(T.pri_inu[k]); T.pri_rfz[k] = ad_frz(T.pri_rfz[k]); T.pri_wfz[k] = ad_frz(T.pri_wfz[k]); } }"),
    (8203, "W.smo1, W.rr, W.nr, W.ilamr, W.mvd_r, W.l_qg, W.rg, W.ng, W.ilamg, W.idx_bg, T);", At::After, 11, "if (AD_FRZ_ON(11)) { TEMPO_LEVELS(k, nz) { T.pbg_rci[k] = ad_frz(T.pbg_rci[k]); T.png_gde[k] = ad_frz(T.png_gde[k]); T.png_rci[k] = ad_frz(T.png_rci[k]); T.pni_iau[k] = ad_frz(T.pni_iau[k]); T.pni_ide[k] = ad_frz(T.pni_ide[k]); T.pni_rci[k] = ad_frz(T.pni_rci[k]); T.pni_sci[k] = ad_frz(T.pni_sci[k]); T.pnr_rci[k] = ad_frz(T.pnr_rci[k]); T.prg_gde[k] = ad_frz(T.prg_gde[k]); T.prg_rci[k] = ad_frz(T.prg_rci[k]); T.pri_ide[k] = ad_frz(T.pri_ide[k]); T.pri_rci[k] = ad_frz(T.pri_rci[k]); T.prr_rci[k] = ad_frz(T.prr_rci[k]); T.prs_iau[k] = ad_frz(T.prs_iau[k]); T.prs_ide[k] = ad_frz(T.prs_ide[k]); T.prs_sci[k] = ad_frz(T.prs_sci[k]); T.prs_sde[k] = ad_frz(T.prs_sde[k]); } }"),
    (8205, "W.smo0, W.smob, W.smoc, W.smoe, W.vtboost, W.l_qg, W.rg, W.ng, W.ilamg, W.idx_bg, T);", At::After, 12, "if (AD_FRZ_ON(12)) { TEMPO_LEVELS(k, nz) { T.pbg_gcw[k] = ad_frz(T.pbg_gcw[k]); T.pbg_scw[k] = ad_frz(T.pbg_scw[k]); T.pnc_gcw[k] = ad_frz(T.pnc_gcw[k]); T.pnc_scw[k] = ad_frz(T.pnc_scw[k]); T.png_scw[k] = ad_frz(T.png_scw[k]); T.pni_ihm[k] = ad_frz(T.pni_ihm[k]); T.prg_gcw[k] = ad_frz(T.prg_gcw[k]); T.prg_ihm[k] = ad_frz(T.prg_ihm[k]); T.prg_scw[k] = ad_frz(T.prg_scw[k]); T.pri_ihm[k] = ad_frz(T.pri_ihm[k]); T.prs_ihm[k] = ad_frz(T.prs_ihm[k]); T.prs_scw[k] = ad_frz(T.prs_scw[k]); } }"),
    (8207, "W.l_qs, W.rs, W.smof, W.smo0, W.smo1, W.l_qg, W.rg, W.ng, W.ilamg, W.idx_bg, T);", At::After, 13, "if (AD_FRZ_ON(13)) { TEMPO_LEVELS(k, nz) { T.pbg_gml[k] = ad_frz(T.pbg_gml[k]); T.png_gde[k] = ad_frz(T.png_gde[k]); T.pnr_gml[k] = ad_frz(T.pnr_gml[k]); T.pnr_sml[k] = ad_frz(T.pnr_sml[k]); T.prg_gcw[k] = ad_frz(T.prg_gcw[k]); T.prg_gde[k] = ad_frz(T.prg_gde[k]); T.prr_gml[k] = ad_frz(T.prr_gml[k]); T.prr_rcw[k] = ad_frz(T.prr_rcw[k]); T.prr_sml[k] = ad_frz(T.prr_sml[k]); T.prs_scw[k] = ad_frz(T.prs_scw[k]); T.prs_sde[k] = ad_frz(T.prs_sde[k]); } }"),
    (8210, "W.ilamg, W.idx_bg, T);", At::After, 17, "if (AD_FRZ_ON(17)) { TEMPO_LEVELS(k, nz) { T.pna_gca[k] = ad_frz(T.pna_gca[k]); T.pna_rca[k] = ad_frz(T.pna_rca[k]); T.pna_sca[k] = ad_frz(T.pna_sca[k]); T.pnd_gcd[k] = ad_frz(T.pnd_gcd[k]); T.pnd_rcd[k] = ad_frz(T.pnd_rcd[k]); T.pnd_scd[k] = ad_frz(T.pnd_scd[k]); } }"),
    (8277, "W.nwfa, W.qv, W.qvs, W.l_qc, W.rc, W.nc, T);", At::After, 14, "if (AD_FRZ_ON(14)) { TEMPO_LEVELS(k, nz) { T.pnc_wcd[k] = ad_frz(T.pnc_wcd[k]); T.prw_vcd[k] = ad_frz(T.prw_vcd[k]); } }"),
    (8303, "W.qv, W.qvs, W.l_qr, W.rr, W.nr, W.ilamr, T);", At::After, 15, "if (AD_FRZ_ON(15)) { TEMPO_LEVELS(k, nz) { T.pnr_rev[k] = ad_frz(T.pnr_rev[k]); T.prv_rev[k] = ad_frz(T.prv_rev[k]); } }"),
    (7338, "ratio = rate_max / sump;", At::After, 16, "if (AD_FRZ_ON(16)) ratio = ad_frz(ratio);"),
    (7352, "ratio = rate_max / sump;", At::After, 16, "if (AD_FRZ_ON(16)) ratio = ad_frz(ratio);"),
    (7366, "ratio = rate_max / sump;", At::After, 16, "if (AD_FRZ_ON(16)) ratio = ad_frz(ratio);"),
    (7378, "ratio = rate_max / sump;", At::After, 16, "if (AD_FRZ_ON(16)) ratio = ad_frz(ratio);"),
    (7392, "ratio = rate_max / sump;", At::After, 16, "if (AD_FRZ_ON(16)) ratio = ad_frz(ratio);"),
    (7404, "ratio = rate_max / sump;", At::After, 16, "if (AD_FRZ_ON(16)) ratio = ad_frz(ratio);"),
    (8203, "W.smo1, W.rr, W.nr, W.ilamr, W.mvd_r, W.l_qg, W.rg, W.ng, W.ilamg, W.idx_bg, T);", At::After, 18, "if (AD_FRZ_ON(18)) { TEMPO_LEVELS(k, nz) { T.pri_ide[k] = ad_frz(T.pri_ide[k]); T.pni_ide[k] = ad_frz(T.pni_ide[k]); T.prs_ide[k] = ad_frz(T.prs_ide[k]); } }"),
    (8203, "W.smo1, W.rr, W.nr, W.ilamr, W.mvd_r, W.l_qg, W.rg, W.ng, W.ilamg, W.idx_bg, T);", At::After, 19, "if (AD_FRZ_ON(19)) { TEMPO_LEVELS(k, nz) { T.prs_sde[k] = ad_frz(T.prs_sde[k]); } }"),
    (8203, "W.smo1, W.rr, W.nr, W.ilamr, W.mvd_r, W.l_qg, W.rg, W.ng, W.ilamg, W.idx_bg, T);", At::After, 20, "if (AD_FRZ_ON(20)) { TEMPO_LEVELS(k, nz) { T.prg_gde[k] = ad_frz(T.prg_gde[k]); T.png_gde[k] = ad_frz(T.png_gde[k]); } }"),
    (8203, "W.smo1, W.rr, W.nr, W.ilamr, W.mvd_r, W.l_qg, W.rg, W.ng, W.ilamg, W.idx_bg, T);", At::After, 21, "if (AD_FRZ_ON(21)) { TEMPO_LEVELS(k, nz) { T.prs_iau[k] = ad_frz(T.prs_iau[k]); T.pni_iau[k] = ad_frz(T.pni_iau[k]); } }"),
    (8203, "W.smo1, W.rr, W.nr, W.ilamr, W.mvd_r, W.l_qg, W.rg, W.ng, W.ilamg, W.idx_bg, T);", At::After, 22, "if (AD_FRZ_ON(22)) { TEMPO_LEVELS(k, nz) { T.prs_sci[k] = ad_frz(T.prs_sci[k]); T.pni_sci[k] = ad_frz(T.pni_sci[k]); } }"),
    (8203, "W.smo1, W.rr, W.nr, W.ilamr, W.mvd_r, W.l_qg, W.rg, W.ng, W.ilamg, W.idx_bg, T);", At::After, 23, "if (AD_FRZ_ON(23)) { TEMPO_LEVELS(k, nz) { T.pri_rci[k] = ad_frz(T.pri_rci[k]); T.pnr_rci[k] = ad_frz(T.pnr_rci[k]); T.png_rci[k] = ad_frz(T.png_rci[k]); T.pni_rci[k] = ad_frz(T.pni_rci[k]); T.prr_rci[k] = ad_frz(T.prr_rci[k]); T.prg_rci[k] = ad_frz(T.prg_rci[k]); T.pbg_rci[k] = ad_frz(T.pbg_rci[k]); } }"),
];

/// Emitted once after the active-type header when freezing.
pub const HELPER: &str = r#"
// ===================== closure freeze (tempo-ad-gen --freeze closure) =====================
// ad_frz: the value bit for bit, no derivative (TL 0, no tape node, so no cotangent). Groups by bit.
#ifndef AD_FRZ_MASK
#define AD_FRZ_MASK 0xffffffffu
#endif
#define AD_FRZ_ON(b) ((((unsigned)(AD_FRZ_MASK)) >> (b)) & 1u)
namespace adr {
template<class T> __device__ __forceinline__ AR<T> ad_frz(const AR<T>& x) { return ad_mk(x.v, 0); }
// ad_lin(y, x): the value of y bit for bit, with y = c x and c = y/x frozen: the derivative keeps only x's
// (one node, partial y/x; none when x is a constant or 0).
template<class T, class U> __device__ __forceinline__ AR<T> ad_lin(const AR<T>& y, const AR<U>& x) {
  const double xv = (double)x.v; return ad_mk(y.v, (x.i && xv != 0.0) ? ad_node1(x.i, (double)y.v / xv) : 0); }
}
using adr::ad_frz;
"#;

/// Trajectory-gated deposition freeze (`tempo-ad-gen --freeze closure --freeze-gate R6[,R7]`, lane S11, 2026-11-07).
/// Groups 6 and 7 freeze the deposition/sublimation coefficient K = prs_sde/ssati (pri_ide, prg_gde) everywhere. The
/// growth they remove lives in near-empty columns: there the relative tangent of the species mass r is huge (gains of
/// 1e7 at qi 3e-11), so the size-distribution moments inside K swing the vapour-heat balance by orders of magnitude.
/// Where the species has mass, K is a smooth power-law function of r and freezing it biases the tangent (j4-repose: the
/// column snow/graupel tangents 4x low, rain 3-5x high at cell 914; 15 % on J4 at 3 h). The gate freezes K only where the
/// forward's own species mass is below a bound (kg m-3). It reads the primal value only (no ad_bit: the branch log is
/// unchanged), so TL and AD take the same decision and the operator stays linear in the direction. Without
/// --freeze-gate the output is byte for byte the ungated generator's.
/// (point source line, freeze group, species mass array the gate reads)
pub const GATE_POINTS: &[(usize, u32, &str)] = &[
    (6245, 6, "rs"), (6309, 6, "rs"),   // snow deposition / sublimation (ice_processes, melting)
    (6185, 7, "ri"),                    // cloud-ice deposition (ice_processes)
    (6259, 7, "rg"), (6346, 7, "rg"),   // graupel deposition / sublimation (ice_processes, melting)
];

/// The gated statement of point j, or None when the point is not gated.
pub fn gated_stmt(j: usize) -> Option<String> {
    let (ln, _txt, _at, bit, stmt) = POINTS[j];
    let (_, _, sp) = GATE_POINTS.iter().find(|(l, b, _)| *l == ln && *b == bit)?;
    let on = format!("if (AD_FRZ_ON({bit}))");
    if !stmt.starts_with(&on) { return None; }
    Some(format!("if (AD_FRZ_ON({bit}) && AD_FRZ_LOW{bit}({sp}[k])){}", &stmt[on.len()..]))
}

/// Emitted after HELPER when gating.
pub const GATE_HELPER: &str = r#"
// ===================== trajectory gate (tempo-ad-gen --freeze-gate, lane S11) =====================
// Groups 6/7 freeze the deposition coefficient only where the forward's species mass (kg m-3) is below the bound.
// Primal comparison only: no ad_bit, so the branch log and its replay are unchanged.
#define AD_FRZ_LOW6(r) ((double)((r).v) < (double)(AD_FRZ_GATE_R6))
#define AD_FRZ_LOW7(r) ((double)((r).v) < (double)(AD_FRZ_GATE_R7))
"#;
