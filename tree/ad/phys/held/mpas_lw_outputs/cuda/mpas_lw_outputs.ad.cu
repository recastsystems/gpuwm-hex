// SPDX-License-Identifier: Apache-2.0
// Derivative entry (ad.entry/v1) of mpas_lw_outputs, the RRTMG LW driver tail that maps the solver fluxes and heating
// rate back to MPAS (module_ra_rrtmg_lw.F:12198-12254; driver mpas_atmphys_driver_radiation_lw.F _to_MPAS:613-719).
// Forward: captured source 683d7a07 lines 9633-9647 (hexcore/mpasphys/rrtmg_lw_v842.py _DRIVER_SOURCE 588-602), image
// 34da9e09, launched once per LW wrapper batch from rrtmg_lw_v842.py:1239-1241, grid ceil(nc/32), block 32.
// Per column c (row strides: m 12*nm, a 12*nl, u/d/uc/dc nl+1, hr nl, tendency nm, radii 3*nm, flux 11):
//   flux[0]=d[0] flux[1]=u[nl] flux[2]=uc[nl]-u[nl] flux[3]=u[nl] flux[4]=uc[nl] flux[5]=d[nl] flux[6]=dc[nl]
//   flux[7]=u[0] flux[8]=uc[0] flux[9]=d[0] flux[10]=dc[0]
//   tendency[k] = (hr[k]/86400)/m[11][k]                                  k < nm   (m field 11 = exner)
//   radii[j][k] = (icloud>0 && m[7][k]>0) ? a[9+j][k] : 0                  k < nm, j = 0..2  (m field 7 = cloud fraction)
// Active: m (fields 7 and 11 read; 7 only through the mask), a (fields 9..11 read), u, d, uc, dc, hr (in);
// tendency, radii, flux (out, full overwrite of the c < nc rows). nc, nm, nl, icloud are integer launch scalars.
// Frozen coefficient (the one limiter-like branch, forward 9639): the cloud mask M = (icloud>0 && cf>0) is a step
// function of cf. It is taken from the trajectory (primal m field 7) in __tl and __ad, so d radii / d cf = 0 and radii
// is linear in a under the frozen mask. That is the exact derivative everywhere except where a perturbation moves a
// cf word across 0 (a jump the size of the radius; see the RESULT hazard list).
// Precision (the glue-row rule, as the live schemes): __aug repeats the forward word for word in f32 (__f*_rn, built
// --fmad=false --ftz=false) and is bitwise to the forward image. __tl/__ad recompute the f32 primal word q = hr/86400
// as the forward does, form the local partials in double (dt/dhr = 1/(86400 e), dt/de = -q/e^2), and round once to
// f32 per written word. One thread per column, as the forward; columns own disjoint words, so both entries are
// deterministic. failure[0] |= 1 on exner == 0 or a non-finite partial.
#define MLO_SU(a, b) __fsub_rn((a), (b))
#define MLO_DV(a, b) __fdiv_rn((a), (b))
#define MF(arr, f, k) arr[(long long)c * 12 * nm + (long long)(f) * nm + (k)]
#define AF(arr, f, k) arr[(long long)c * 12 * nl + (long long)(f) * nl + (k)]
#define LV(arr, l) arr[(long long)c * (nl + 1) + (l)]
#define HR(arr, k) arr[(long long)c * nl + (k)]
#define TN(arr, k) arr[(long long)c * nm + (k)]
#define RD(arr, j, k) arr[(long long)c * 3 * nm + (long long)(j) * nm + (k)]
#define FX(arr, i) arr[(long long)c * 11 + (i)]

__device__ __forceinline__ void mlo_flag(int* failure, double v) { if (!isfinite(v)) atomicOr(failure, 1); }
__device__ __forceinline__ bool mlo_mask(int icloud, float cf) { return icloud > 0 && cf > 0.0f; }

extern "C" __global__ void mpas_lw_outputs__aug(
    int nc, int nm, int nl, int icloud, const float* __restrict__ m, const float* __restrict__ a,
    const float* __restrict__ u, const float* __restrict__ d, const float* __restrict__ uc, const float* __restrict__ dc,
    const float* __restrict__ hr, float* __restrict__ tendency, float* __restrict__ radii, float* __restrict__ flux,
    int* __restrict__ failure)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    FX(flux, 0) = LV(d, 0); FX(flux, 1) = LV(u, nl); FX(flux, 2) = MLO_SU(LV(uc, nl), LV(u, nl));
    FX(flux, 3) = LV(u, nl); FX(flux, 4) = LV(uc, nl); FX(flux, 5) = LV(d, nl); FX(flux, 6) = LV(dc, nl);
    FX(flux, 7) = LV(u, 0); FX(flux, 8) = LV(uc, 0); FX(flux, 9) = LV(d, 0); FX(flux, 10) = LV(dc, 0);
    for (int k = 0; k < nm; ++k) {
        TN(tendency, k) = MLO_DV(MLO_DV(HR(hr, k), 86400.0f), MF(m, 11, k));
        if (!isfinite(TN(tendency, k))) atomicOr(failure, 1);
        const bool mask = mlo_mask(icloud, MF(m, 7, k));
        for (int j = 0; j < 3; ++j) RD(radii, j, k) = mask ? AF(a, 9 + j, k) : 0.0f;
    }
}

extern "C" __global__ void mpas_lw_outputs__tl(
    int nc, int nm, int nl, int icloud, const float* __restrict__ m, const float* __restrict__ hr,
    const float* __restrict__ d_m, const float* __restrict__ d_a,
    const float* __restrict__ d_u, const float* __restrict__ d_d, const float* __restrict__ d_uc,
    const float* __restrict__ d_dc, const float* __restrict__ d_hr,
    float* __restrict__ o_tendency, float* __restrict__ o_radii, float* __restrict__ o_flux,
    int* __restrict__ failure)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    FX(o_flux, 0) = LV(d_d, 0); FX(o_flux, 1) = LV(d_u, nl);
    FX(o_flux, 2) = (float)((double)LV(d_uc, nl) - (double)LV(d_u, nl));
    FX(o_flux, 3) = LV(d_u, nl); FX(o_flux, 4) = LV(d_uc, nl); FX(o_flux, 5) = LV(d_d, nl); FX(o_flux, 6) = LV(d_dc, nl);
    FX(o_flux, 7) = LV(d_u, 0); FX(o_flux, 8) = LV(d_uc, 0); FX(o_flux, 9) = LV(d_d, 0); FX(o_flux, 10) = LV(d_dc, 0);
    for (int k = 0; k < nm; ++k) {
        const float e = MF(m, 11, k);
        const float q = MLO_DV(HR(hr, k), 86400.0f);
        const double re = 1.0 / (double)e;
        const double t = ((double)HR(d_hr, k) / 86400.0 - (double)q * re * (double)MF(d_m, 11, k)) * re;
        if (e == 0.0f) atomicOr(failure, 1);
        mlo_flag(failure, t);
        TN(o_tendency, k) = (float)t;
        const bool mask = mlo_mask(icloud, MF(m, 7, k));   // frozen from the trajectory
        for (int j = 0; j < 3; ++j) RD(o_radii, j, k) = mask ? AF(d_a, 9 + j, k) : 0.0f;
    }
}

// Reverse. Seeds of the full-overwrite outputs (tendency, radii, flux) are read and consumed (set 0); the inputs
// accumulate. flux[0] and flux[9] both copy d[0], and flux[1], flux[3] and -flux[2] all read u[nl]: their seeds are
// summed in double and rounded once into the accumulated word.
extern "C" __global__ void mpas_lw_outputs__ad(
    int nc, int nm, int nl, int icloud, const float* __restrict__ m, const float* __restrict__ hr,
    float* __restrict__ b_m, float* __restrict__ b_a,
    float* __restrict__ b_u, float* __restrict__ b_d, float* __restrict__ b_uc, float* __restrict__ b_dc,
    float* __restrict__ b_hr,
    float* __restrict__ b_tendency, float* __restrict__ b_radii, float* __restrict__ b_flux,
    int* __restrict__ failure)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    double s[11];
    for (int i = 0; i < 11; ++i) { s[i] = (double)FX(b_flux, i); FX(b_flux, i) = 0.0f; }
    // nl >= 1 always (the wrapper's layer groups are 64 or 65 layers), so level 0 and level nl are distinct words.
    LV(b_d, 0) = (float)((double)LV(b_d, 0) + (s[0] + s[9]));
    LV(b_u, nl) = (float)((double)LV(b_u, nl) + (s[1] + s[3] - s[2]));
    LV(b_uc, nl) = (float)((double)LV(b_uc, nl) + (s[2] + s[4]));
    LV(b_d, nl) = (float)((double)LV(b_d, nl) + s[5]);
    LV(b_dc, nl) = (float)((double)LV(b_dc, nl) + s[6]);
    LV(b_u, 0) = (float)((double)LV(b_u, 0) + s[7]);
    LV(b_uc, 0) = (float)((double)LV(b_uc, 0) + s[8]);
    LV(b_dc, 0) = (float)((double)LV(b_dc, 0) + s[10]);
    for (int k = 0; k < nm; ++k) {
        const float e = MF(m, 11, k);
        const float q = MLO_DV(HR(hr, k), 86400.0f);
        const double re = 1.0 / (double)e;
        const double g = (double)TN(b_tendency, k);
        TN(b_tendency, k) = 0.0f;
        const double bh = g * re / 86400.0, be = -g * (double)q * re * re;
        if (e == 0.0f) atomicOr(failure, 1);
        mlo_flag(failure, bh); mlo_flag(failure, be);
        HR(b_hr, k) = (float)((double)HR(b_hr, k) + bh);
        MF(b_m, 11, k) = (float)((double)MF(b_m, 11, k) + be);
        const bool mask = mlo_mask(icloud, MF(m, 7, k));   // frozen from the trajectory: d radii / d cf = 0
        for (int j = 0; j < 3; ++j) {
            const double r = (double)RD(b_radii, j, k);
            RD(b_radii, j, k) = 0.0f;
            if (mask) AF(b_a, 9 + j, k) = (float)((double)AF(b_a, 9 + j, k) + r);
        }
    }
}
#undef MF
#undef AF
#undef LV
#undef HR
#undef TN
#undef RD
#undef FX
