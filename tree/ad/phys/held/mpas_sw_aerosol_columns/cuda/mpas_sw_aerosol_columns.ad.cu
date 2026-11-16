// SPDX-License-Identifier: Apache-2.0
// Derivative entry (ad.entry/v1) of mpas_sw_aerosol_columns, the column pass of the RRTMG SW aerosol optics
// (MPAS module_ra_rrtmg_sw_aerosols.F:70-451; production path MPASShortwave._aerosol_passes,
// hexcore/mpasphys/rrtmg_sw_v842.py:440-442 launches it with grid ceil(nc/64), block 64).
// Forward: captured source 3c499c7b lines 3850-3877 (rrtmg_sw_v842.py CUDA_SOURCE 3531-3558), image c22c485e.
// Per column c (row strides: p/t/qv/dz/zs nz, spectral 14):
//   aodopt==1:  aod2[c] = aodval                                   (written, then read below)
//   a = angopt==1 ? angval : ang2[c];  e = fminf(2.5, fmaxf(-0.3, a));  exponent_out[c] = e
//   has3d: return (the 3-D optical-depth path reads nothing else and writes nothing else)
//   spectral[c][b] = angopt==3 ? sum_j (L_j(rh_s) * RAOD[type-1][j][b]) * aod2[c]       (Lagrange in surface rh)
//                              : aod2[c] * rate_b(e)                                     (Angstrom band integral)
//   zs[c][0] = ht + dz0/2;  zs[c][k] = zs[c][k-1] + (dz[k-1]+dz[k])/2;  htoa = zs[nz-1] + dz[nz-1]/2
//   denom[c] = 2500 (exp(-ht/2500) - exp(-htoa/2500))
// rh_s = clamp(0, 99, 100 e_v / e_s) of the level-0 p, t, qv (mpas_sw_rh).
// Active: p, t, qv, dz, ht, ang2 (in); aod2 (inout: read when aodopt==2, overwritten with a constant when aodopt==1);
// exponent_out (out); spectral, zs, denom (out, written only when has3d == 0). type is an integer index (inactive).
// nc, nz, has3d, aodopt, angopt are integer launch scalars and aodval, angval float launch parameters (inactive).
// Frozen coefficients (every limiter-like branch is evaluated on the trajectory, primal words, in __tl and __ad):
//   F1 the exponent clamp fminf(2.5, fmaxf(-0.3, a)): d e/d a = 1 strictly inside (-0.3, 2.5), else 0.
//   F2 the rh clamp fminf(99, fmaxf(0, r)): d rh/d r = 1 strictly inside (0, 99), else 0.
//   F3 rv = fmaxf(0, qv): d rv/d qv = 1 for qv > 0, else 0.
//   F4 the Lagrange window ii/imin/imax (an index chosen from rh): fixed; the polynomial is differentiated in rh.
//   F5 the rate select (e < .999 || e > 1.001): the near-1 branch is a constant (the e -> 1 limit), so d rate/d e = 0
//      inside the band, the analytic derivative of the power-law integral outside it.
// Precision (the glue-row rule, as the live schemes): __aug repeats the forward word for word in f32 (__f*_rn and the
// forward's own glibc-transcribed gfk_exp/gfk_pow/gfk_log from the generated prelude; built --fmad=false --ftz=false)
// and is bitwise to the forward image. __tl/__ad recompute the f32 primal words the forward makes (tc, e_s, e_v, r,
// rh, e, htoa), form the local partials in double and round once to f32 per written word; the AD accumulates each
// input word's contributions in double and adds them with one rounding. One thread per column, as the forward;
// columns own disjoint words, so both entries are deterministic. failure[0] |= 1 on a non-finite partial.
#include "mpas_sw_aerosol_columns.prelude.cuh"

#define MSAC_ARGS_S int nc, int nz, int has3d, int aodopt, int angopt, float aodval, float angval
#define MSAC_ARGS_P const float* __restrict__ p, const float* __restrict__ t, const float* __restrict__ qv, \
    const float* __restrict__ dz, const float* __restrict__ ht, const int* __restrict__ type, \
    const float* aod2, const float* __restrict__ ang2

__device__ __forceinline__ void msac_flag(int* failure, double v) { if (!isfinite(v)) atomicOr(failure, 1); }
__device__ __forceinline__ float msac_clamp(float a) { return fminf(2.5f, fmaxf(-.3f, a)); }
__device__ __forceinline__ bool msac_pass(float a) { return a > -.3f && a < 2.5f; }   // F1

// Surface rh and its partials (F2, F3). The f32 words are the forward's (mpas_sw_rh), op for op.
struct MsacRh { float rh; double cp, ct, cq; };
__device__ MsacRh msac_rh(float pp, float tt, float qq) {
    float tc = FSUB(tt, 273.15f), rv = fmaxf(0.f, qq);
    float es = FMUL(6.112f, gfk_exp(FDIV(FMUL(17.6f, tc), FADD(tc, 243.5f))));
    float e = FDIV(FMUL(FMUL(.01f, rv), pp), FADD(rv, .62197f));
    float r = FDIV(FMUL(100.f, e), es);
    MsacRh o; o.rh = fminf(99.f, fmaxf(0.f, r));
    const double m = (r > 0.f && r < 99.f) ? 1.0 : 0.0;                           // F2
    const double dtc = (double)tc + 243.5, des_dt = (double)es * 17.6 * 243.5 / (dtc * dtc);
    const double drv = (double)rv + 0.62197;
    o.cp = m * 100.0 / (double)es * 0.01 * (double)rv / drv;
    o.cq = qq > 0.f ? m * 100.0 / (double)es * 0.01 * (double)pp * 0.62197 / (drv * drv) : 0.0;   // F3
    o.ct = -m * 100.0 * (double)e * des_dt / ((double)es * (double)es);
    return o;
}
// Lagrange sum S = sum_j L_j(rh) T_j and dS/drh over the window the forward picks (F4).
__device__ void msac_lagrange(int type, int b, float rh, double& S, double& dS) {
    int ii = 0; while (ii < 8 && rh > MPAS_RHS[ii]) ii++;
    const int imin = max(0, ii - 3), imax = min(7, ii + 2);
    S = 0.0; dS = 0.0;
    for (int jj = imin; jj <= imax; jj++) {
        const double xj = MPAS_RHS[jj], T = MPAS_RAOD[type - 1][jj][b];
        double L = 1.0, Lp = 0.0;
        for (int k = imin; k <= imax; k++) if (k != jj) {
            const double w = 1.0 / (xj - (double)MPAS_RHS[k]), f = ((double)rh - (double)MPAS_RHS[k]) * w;
            Lp = Lp * f + L * w; L *= f;
        }
        S += L * T; dS += Lp * T;
    }
}
// Forward f32 rate word of band b and d rate/d e in double (F5).
__device__ float msac_rate(float e, int b, double& drate) {
    const float U = MPAS_UPPER[b], Lo = MPAS_LOWER[b];
    if (e < .999f || e > 1.001f) {
        float one = FSUB(1.f, e);
        float rate = FDIV(FMUL(gfk_pow(.55f, e), FSUB(gfk_pow(U, one), gfk_pow(Lo, one))), FMUL(one, FSUB(U, Lo)));
        const double o = 1.0 - (double)e, lu = log((double)U), ll = log((double)Lo), l55 = log((double).55f);
        const double P = exp((double)e * l55), Pu = exp(o * lu), Pl = exp(o * ll), W = (double)U - (double)Lo;
        const double Q = Pu - Pl, D = o * W;
        drate = (P * l55 * Q - P * (Pu * lu - Pl * ll)) / D + P * Q * W / (D * D);
        return rate;
    }
    drate = 0.0;
    return FMUL(FDIV(.55f, FSUB(U, Lo)), gfk_log(FDIV(U, Lo)));
}

// The forward, word for word (captured source 3c499c7b:3850-3877), plus the failure flag.
extern "C" __global__ void mpas_sw_aerosol_columns__aug(MSAC_ARGS_S,
    const float* p, const float* t, const float* qv, const float* dz, const float* ht, const int* type,
    float* aod2, const float* ang2, float* exponent_out, float* spectral, float* zs, float* denom, int* failure)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= nc) return;
    if (aodopt == 1) aod2[c] = aodval;
    float exponent = angopt == 1 ? fminf(2.5f, fmaxf(-.3f, angval)) : fminf(2.5f, fmaxf(-.3f, ang2[c]));
    exponent_out[c] = exponent;
    if (has3d) return;
    const long long base = (long long)c * nz;
    float rh_surface = mpas_sw_rh(p[base], t[base], qv[base]);
    for (int b = 0; b < 14; b++) {
        float value;
        if (angopt == 3) value = mpas_sw_spectral(MPAS_RAOD, type[c], b, rh_surface, aod2[c], true);
        else {
            float rate;
            if (exponent < .999f || exponent > 1.001f) { float one = FSUB(1.f, exponent);
                rate = FDIV(FMUL(gfk_pow(.55f, exponent), FSUB(gfk_pow(MPAS_UPPER[b], one), gfk_pow(MPAS_LOWER[b], one))), FMUL(one, FSUB(MPAS_UPPER[b], MPAS_LOWER[b])));
            } else rate = FMUL(FDIV(.55f, FSUB(MPAS_UPPER[b], MPAS_LOWER[b])), gfk_log(FDIV(MPAS_UPPER[b], MPAS_LOWER[b])));
            value = FMUL(aod2[c], rate);
        }
        spectral[(long long)c * 14 + b] = value;
        if (!isfinite(value)) atomicOr(failure, 1);
    }
    float z = FADD(ht[c], FMUL(.5f, dz[base])); zs[base] = z;
    for (int k = 1; k < nz; k++) { z = FADD(z, FMUL(.5f, FADD(dz[base + k - 1], dz[base + k]))); zs[base + k] = z; }
    float htoa = FADD(z, FMUL(.5f, dz[base + nz - 1]));
    denom[c] = FMUL(2500.f, FSUB(gfk_exp(FDIV(-ht[c], 2500.f)), gfk_exp(FDIV(-htoa, 2500.f))));
}

// Forward primal htoa (f32, the forward's running-sum order).
__device__ __forceinline__ float msac_htoa(int nz, long long base, const float* ht, const float* dz, int c) {
    float z = FADD(ht[c], FMUL(.5f, dz[base]));
    for (int k = 1; k < nz; k++) z = FADD(z, FMUL(.5f, FADD(dz[base + k - 1], dz[base + k])));
    return FADD(z, FMUL(.5f, dz[base + nz - 1]));
}

extern "C" __global__ void mpas_sw_aerosol_columns__tl(MSAC_ARGS_S, MSAC_ARGS_P,
    const float* __restrict__ d_p, const float* __restrict__ d_t, const float* __restrict__ d_qv,
    const float* __restrict__ d_dz, const float* __restrict__ d_ht, float* d_aod2, const float* __restrict__ d_ang2,
    float* __restrict__ o_exponent, float* __restrict__ o_spectral, float* __restrict__ o_zs, float* __restrict__ o_denom,
    int* __restrict__ failure)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= nc) return;
    float A; double dA;
    if (aodopt == 1) { d_aod2[c] = 0.0f; A = aodval; dA = 0.0; }        // overwritten with a constant
    else { A = aod2[c]; dA = (double)d_aod2[c]; }                       // read only: the tangent passes through
    const float a = angopt == 1 ? angval : ang2[c];
    const float e = msac_clamp(a);
    const double de = (angopt != 1 && msac_pass(a)) ? (double)d_ang2[c] : 0.0;   // F1
    o_exponent[c] = (float)de;
    if (has3d) return;
    const long long base = (long long)c * nz;
    double drh = 0.0; float rh = 0.f;
    if (angopt == 3) {
        const MsacRh R = msac_rh(p[base], t[base], qv[base]); rh = R.rh;
        drh = R.cp * (double)d_p[base] + R.ct * (double)d_t[base] + R.cq * (double)d_qv[base];
    }
    for (int b = 0; b < 14; b++) {
        double v;
        if (angopt == 3) { double S, dS; msac_lagrange(type[c], b, rh, S, dS); v = (double)A * dS * drh + S * dA; }
        else { double dr; const float rate = msac_rate(e, b, dr); v = (double)rate * dA + (double)A * dr * de; }
        msac_flag(failure, v);
        o_spectral[(long long)c * 14 + b] = (float)v;
    }
    double dzr = (double)d_ht[c] + 0.5 * (double)d_dz[base];
    o_zs[base] = (float)dzr;
    for (int k = 1; k < nz; k++) { dzr += 0.5 * ((double)d_dz[base + k - 1] + (double)d_dz[base + k]); o_zs[base + k] = (float)dzr; }
    const double dhtoa = dzr + 0.5 * (double)d_dz[base + nz - 1];
    const float htoa = msac_htoa(nz, base, ht, dz, c);
    const double dd = -exp(-(double)ht[c] / 2500.0) * (double)d_ht[c] + exp(-(double)htoa / 2500.0) * dhtoa;
    msac_flag(failure, dd); msac_flag(failure, dzr);
    o_denom[c] = (float)dd;
}

// Reverse. Seeds of the written outputs (exponent_out; spectral, zs, denom when has3d == 0) are read and consumed
// (set 0); words the forward does not write keep their seed (identity). Inputs accumulate, one rounding per word.
// aod2 is in place: overwritten (aodopt==1) its seed is consumed and the pre-launch word gets nothing; read
// (aodopt==2) it gets the spectral contributions on top of its pass-through seed.
extern "C" __global__ void mpas_sw_aerosol_columns__ad(MSAC_ARGS_S, MSAC_ARGS_P,
    float* __restrict__ b_p, float* __restrict__ b_t, float* __restrict__ b_qv, float* __restrict__ b_dz,
    float* __restrict__ b_ht, float* b_aod2, float* __restrict__ b_ang2,
    float* __restrict__ b_exponent, float* __restrict__ b_spectral, float* __restrict__ b_zs, float* __restrict__ b_denom,
    int* __restrict__ failure)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= nc) return;
    const float A = aodopt == 1 ? aodval : aod2[c];
    const float a = angopt == 1 ? angval : ang2[c];
    const float e = msac_clamp(a);
    double bE = (double)b_exponent[c]; b_exponent[c] = 0.0f;
    double bA = 0.0;
    if (!has3d) {
        const long long base = (long long)c * nz;
        double bRH = 0.0; MsacRh R; R.rh = 0.f; R.cp = R.ct = R.cq = 0.0;
        if (angopt == 3) R = msac_rh(p[base], t[base], qv[base]);
        for (int b = 0; b < 14; b++) {
            const long long i = (long long)c * 14 + b;
            const double s = (double)b_spectral[i]; b_spectral[i] = 0.0f;
            if (angopt == 3) { double S, dS; msac_lagrange(type[c], b, R.rh, S, dS); bRH += s * (double)A * dS; bA += s * S; }
            else { double dr; const float rate = msac_rate(e, b, dr); bA += s * (double)rate; bE += s * (double)A * dr; }
        }
        const float htoa = msac_htoa(nz, base, ht, dz, c);
        const double sd = (double)b_denom[c]; b_denom[c] = 0.0f;
        const double bhtoa = sd * exp(-(double)htoa / 2500.0);
        double bht = -sd * exp(-(double)ht[c] / 2500.0);
        double Anext = 0.0;                                   // adjoint of zs[k+1] (all its uses)
        for (int k = nz - 1; k >= 0; k--) {
            const double Ak = (double)b_zs[base + k] + Anext + (k == nz - 1 ? bhtoa : 0.0); b_zs[base + k] = 0.0f;
            const double g = 0.5 * Ak + (k < nz - 1 ? 0.5 * Anext : 0.0) + (k == nz - 1 ? 0.5 * bhtoa : 0.0);
            msac_flag(failure, g);
            b_dz[base + k] = (float)((double)b_dz[base + k] + g);
            Anext = Ak;
        }
        bht += Anext;                                         // zs[0] = ht + dz0/2
        msac_flag(failure, bht);
        b_ht[c] = (float)((double)b_ht[c] + bht);
        if (angopt == 3) {
            const double gp = R.cp * bRH, gt = R.ct * bRH, gq = R.cq * bRH;
            msac_flag(failure, gp); msac_flag(failure, gt); msac_flag(failure, gq);
            b_p[base] = (float)((double)b_p[base] + gp);
            b_t[base] = (float)((double)b_t[base] + gt);
            b_qv[base] = (float)((double)b_qv[base] + gq);
        }
    }
    msac_flag(failure, bE); msac_flag(failure, bA);
    if (angopt != 1 && msac_pass(a)) b_ang2[c] = (float)((double)b_ang2[c] + bE);   // F1
    if (aodopt == 1) b_aod2[c] = 0.0f;
    else if (!has3d) b_aod2[c] = (float)((double)b_aod2[c] + bA);
}
