// SPDX-License-Identifier: Apache-2.0
// Derivative entry (ad.entry/v1) of rlw_rtrn_final, the RRTMG LW flux/heating-rate finisher (MPAS module_ra_rrtmg_lw.F
// rtrnmc tail, 3096-3533). Forward: captured source 683d7a07 lines 2192-2232 (gpuwm/core/kernels/rrtmg_lw_chain.cu),
// image 34da9e09, launched from hexcore/mpasphys/rrtmg_lw_v842.py:882-886.
// Per column c, levels l = 0..nl (row stride nl+1):
//   u1[l] = fluxfac*u0[l]   d1[l] = fluxfac*d0[l]   F[l] = u1[l]-d1[l]        (same for the clear-sky pair -> Fc)
//   htr[l] = heatfac*(F[l]-F[l+1]) / (pz[l]-pz[l+1])  for l < nl;   htr[nl] = 0      (same for htrc with Fc)
// Active: pz (in), totuflux/totdflux/totuclfl/totdclfl (inout, full overwrite), fnet/fnetc/htr/htrc (out, full
// overwrite). fluxfac, heatfac are configuration constants (pi*2e4 and g*86400/(cp*100)): inactive.
// No limiter, clamp, branch on state or table: the operator is linear in the fluxes and rational in pz, so no
// frozen-coefficient treatment is needed (none is applied). The only singular point is dp = pz[l]-pz[l+1] = 0.
// Precision (the glue-row rule, as the live schemes): __aug repeats the forward word for word in f32 (__f*_rn,
// built --fmad=false --ftz=false) and is bitwise to the forward image. __tl/__ad recompute the f32 primal words
// exactly as the forward does (u1, F, N = heatfac*(F[l]-F[l+1]), dp), form the local partials in double, and round
// once to f32 per written word. One thread per column, as the forward; columns are disjoint and no data word is
// written by two threads, so both entries are deterministic. failure[0] |= 1 on a non-finite partial or dp == 0.
#define RRF_MU(a, b) __fmul_rn((a), (b))
#define RRF_SU(a, b) __fsub_rn((a), (b))
#define RRF_DV(a, b) __fdiv_rn((a), (b))
#define LV(a, l) a[(long long)col * (nl + 1) + (l)]

__device__ __forceinline__ void rrf_flag(int* failure, double v) { if (!isfinite(v)) atomicOr(failure, 1); }

extern "C" __global__ void rlw_rtrn_final__aug(
    int ncol, int nl, const float* __restrict__ pz, float fluxfac, float heatfac,
    float* __restrict__ totuflux, float* __restrict__ totdflux, float* __restrict__ totuclfl, float* __restrict__ totdclfl,
    float* __restrict__ fnet, float* __restrict__ fnetc, float* __restrict__ htr, float* __restrict__ htrc,
    int* __restrict__ failure)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    LV(totuflux, 0) = RRF_MU(LV(totuflux, 0), fluxfac);
    LV(totdflux, 0) = RRF_MU(LV(totdflux, 0), fluxfac);
    LV(fnet, 0) = RRF_SU(LV(totuflux, 0), LV(totdflux, 0));
    LV(totuclfl, 0) = RRF_MU(LV(totuclfl, 0), fluxfac);
    LV(totdclfl, 0) = RRF_MU(LV(totdclfl, 0), fluxfac);
    LV(fnetc, 0) = RRF_SU(LV(totuclfl, 0), LV(totdclfl, 0));
    for (int lev = 1; lev <= nl; ++lev) {
        LV(totuflux, lev) = RRF_MU(LV(totuflux, lev), fluxfac);
        LV(totdflux, lev) = RRF_MU(LV(totdflux, lev), fluxfac);
        LV(fnet, lev) = RRF_SU(LV(totuflux, lev), LV(totdflux, lev));
        LV(totuclfl, lev) = RRF_MU(LV(totuclfl, lev), fluxfac);
        LV(totdclfl, lev) = RRF_MU(LV(totdclfl, lev), fluxfac);
        LV(fnetc, lev) = RRF_SU(LV(totuclfl, lev), LV(totdclfl, lev));
        int l0 = lev - 1;
        LV(htr, l0) = RRF_DV(RRF_MU(heatfac, RRF_SU(LV(fnet, l0), LV(fnet, lev))), RRF_SU(LV(pz, l0), LV(pz, lev)));
        LV(htrc, l0) = RRF_DV(RRF_MU(heatfac, RRF_SU(LV(fnetc, l0), LV(fnetc, lev))), RRF_SU(LV(pz, l0), LV(pz, lev)));
        if (!isfinite(LV(htr, l0)) || !isfinite(LV(htrc, l0))) atomicOr(failure, 1);
    }
    LV(htr, nl) = 0.0f;
    LV(htrc, nl) = 0.0f;
}

// Net flux word F[l] of the f32 forward, from the pre-launch words u, d.
__device__ __forceinline__ float rrf_net(float u, float d, float fluxfac) { return RRF_SU(RRF_MU(u, fluxfac), RRF_MU(d, fluxfac)); }

extern "C" __global__ void rlw_rtrn_final__tl(
    int ncol, int nl, const float* __restrict__ pz, float fluxfac, float heatfac,
    const float* __restrict__ totuflux, const float* __restrict__ totdflux,
    const float* __restrict__ totuclfl, const float* __restrict__ totdclfl,
    const float* __restrict__ d_pz,
    const float* __restrict__ d_totuflux, const float* __restrict__ d_totdflux,
    const float* __restrict__ d_totuclfl, const float* __restrict__ d_totdclfl,
    float* __restrict__ o_totuflux, float* __restrict__ o_totdflux, float* __restrict__ o_totuclfl, float* __restrict__ o_totdclfl,
    float* __restrict__ o_fnet, float* __restrict__ o_fnetc, float* __restrict__ o_htr, float* __restrict__ o_htrc,
    int* __restrict__ failure)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    const double ff = (double)fluxfac, hf = (double)heatfac;
    double dF_prev = 0.0, dFc_prev = 0.0;
    float F_prev = 0.0f, Fc_prev = 0.0f;
    for (int l = 0; l <= nl; ++l) {
        const double du = ff * (double)LV(d_totuflux, l), dd = ff * (double)LV(d_totdflux, l);
        const double duc = ff * (double)LV(d_totuclfl, l), ddc = ff * (double)LV(d_totdclfl, l);
        LV(o_totuflux, l) = (float)du; LV(o_totdflux, l) = (float)dd;
        LV(o_totuclfl, l) = (float)duc; LV(o_totdclfl, l) = (float)ddc;
        const double dF = du - dd, dFc = duc - ddc;
        LV(o_fnet, l) = (float)dF; LV(o_fnetc, l) = (float)dFc;
        const float F = rrf_net(LV(totuflux, l), LV(totdflux, l), fluxfac);
        const float Fc = rrf_net(LV(totuclfl, l), LV(totdclfl, l), fluxfac);
        if (l >= 1) {
            const int l0 = l - 1;
            const float dp = RRF_SU(LV(pz, l0), LV(pz, l));
            const float N = RRF_MU(heatfac, RRF_SU(F_prev, F)), Nc = RRF_MU(heatfac, RRF_SU(Fc_prev, Fc));
            const double rdp = 1.0 / (double)dp;
            const double ddp = (double)LV(d_pz, l0) - (double)LV(d_pz, l);
            const double t = (hf * (dF_prev - dF) - (double)N * rdp * ddp) * rdp;   // dN/dp - N*ddp/dp^2
            const double tc = (hf * (dFc_prev - dFc) - (double)Nc * rdp * ddp) * rdp;
            if (dp == 0.0f) atomicOr(failure, 1);
            rrf_flag(failure, t); rrf_flag(failure, tc);
            LV(o_htr, l0) = (float)t; LV(o_htrc, l0) = (float)tc;
        }
        dF_prev = dF; dFc_prev = dFc; F_prev = F; Fc_prev = Fc;
    }
    LV(o_htr, nl) = 0.0f;
    LV(o_htrc, nl) = 0.0f;
}

// Reverse. Seeds of the inout fluxes arrive in s_* (the binder's snapshot); their cotangents accumulate into the
// cleared shadows b_*. Seeds of the full-overwrite outputs (fnet, fnetc, htr, htrc) are read and consumed (set 0).
// pz accumulates. Per level l: bF[l] = s_fnet[l] + heatfac*(bN[l] - bN[l-1]) with bN[l] = s_htr[l]/dp[l] (l < nl,
// 0 otherwise); bpz[l] += bdp[l] - bdp[l-1] with bdp[l] = -s_htr[l]*N[l]/dp[l]^2; bu0[l] += fluxfac*(s_u[l] + bF[l]);
// bd0[l] += fluxfac*(s_d[l] - bF[l]). s_htr[nl] multiplies the constant 0 and is dropped.
extern "C" __global__ void rlw_rtrn_final__ad(
    int ncol, int nl, const float* __restrict__ pz, float fluxfac, float heatfac,
    const float* __restrict__ totuflux, const float* __restrict__ totdflux,
    const float* __restrict__ totuclfl, const float* __restrict__ totdclfl,
    float* __restrict__ b_pz,
    const float* __restrict__ s_totuflux, const float* __restrict__ s_totdflux,
    const float* __restrict__ s_totuclfl, const float* __restrict__ s_totdclfl,
    float* __restrict__ b_totuflux, float* __restrict__ b_totdflux, float* __restrict__ b_totuclfl, float* __restrict__ b_totdclfl,
    float* __restrict__ b_fnet, float* __restrict__ b_fnetc, float* __restrict__ b_htr, float* __restrict__ b_htrc,
    int* __restrict__ failure)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    const double ff = (double)fluxfac, hf = (double)heatfac;
    double bN_prev = 0.0, bNc_prev = 0.0, bdp_prev = 0.0;
    float F = rrf_net(LV(totuflux, 0), LV(totdflux, 0), fluxfac);
    float Fc = rrf_net(LV(totuclfl, 0), LV(totdclfl, 0), fluxfac);
    for (int l = 0; l <= nl; ++l) {
        double bN = 0.0, bNc = 0.0, bdp = 0.0;
        float F_next = 0.0f, Fc_next = 0.0f;
        if (l < nl) {
            F_next = rrf_net(LV(totuflux, l + 1), LV(totdflux, l + 1), fluxfac);
            Fc_next = rrf_net(LV(totuclfl, l + 1), LV(totdclfl, l + 1), fluxfac);
            const float dp = RRF_SU(LV(pz, l), LV(pz, l + 1));
            const float N = RRF_MU(heatfac, RRF_SU(F, F_next)), Nc = RRF_MU(heatfac, RRF_SU(Fc, Fc_next));
            const double rdp = 1.0 / (double)dp;
            const double g = (double)LV(b_htr, l), gc = (double)LV(b_htrc, l);
            bN = g * rdp; bNc = gc * rdp;
            bdp = -(g * (double)N + gc * (double)Nc) * rdp * rdp;
            if (dp == 0.0f) atomicOr(failure, 1);
            rrf_flag(failure, bN); rrf_flag(failure, bNc); rrf_flag(failure, bdp);
        }
        const double bF = (double)LV(b_fnet, l) + hf * (bN - bN_prev);
        const double bFc = (double)LV(b_fnetc, l) + hf * (bNc - bNc_prev);
        LV(b_pz, l) = (float)((double)LV(b_pz, l) + (bdp - bdp_prev));
        LV(b_totuflux, l) = (float)((double)LV(b_totuflux, l) + ff * ((double)LV(s_totuflux, l) + bF));
        LV(b_totdflux, l) = (float)((double)LV(b_totdflux, l) + ff * ((double)LV(s_totdflux, l) - bF));
        LV(b_totuclfl, l) = (float)((double)LV(b_totuclfl, l) + ff * ((double)LV(s_totuclfl, l) + bFc));
        LV(b_totdclfl, l) = (float)((double)LV(b_totdclfl, l) + ff * ((double)LV(s_totdclfl, l) - bFc));
        LV(b_fnet, l) = 0.0f; LV(b_fnetc, l) = 0.0f; LV(b_htr, l) = 0.0f; LV(b_htrc, l) = 0.0f;
        bN_prev = bN; bNc_prev = bNc; bdp_prev = bdp; F = F_next; Fc = Fc_next;
    }
}
#undef LV
