// SPDX-License-Identifier: Apache-2.0
// Derivative entry (ad.entry/v1) of mpas_sw_gt_aod, the Thompson water/ice-friendly aerosol optical depth that feeds
// the RRTMG SW aerosol optics on hrrrv5 (MPAS module_mp_thompson_aerosols.F:23-210 gt_aod; driver
// mpas_atmphys_driver_radiation_sw.F:529-583). Forward: captured source 3c499c7b, device function aer3_gt_aod at
// 3406-3454 (RSLF 3379-3398, NINT 3400-3403, tables 3358-3377), kernel 3917-3922 (hexcore/mpasphys/rrtmg_sw_v842.py
// 3038-3134 and 3598-3603), image c22c485e, launched once per SW radiation step from rrtmg_sw_v842.py:461-462
// (thompson_aerosols), grid ceil(nc/64), block 64, all nc = 2562 columns in one launch.
// Per point i = c*nz + k:
//   rhoa = p/(287 t); t_idx = clamp(NINT(10.999 - 0.0333 t), 1, 4)
//   qvsat = RSLF(p, t): x = max(-80, t - 273.16), esl = min(poly8(x), 0.15 p), qvsat = 0.622 esl/(p - esl)
//   rh = clamp(100 qv/qvsat, 10.1, 98); (i1, i2) = bracket of rh in rh_arr; g(r) = r/(100 - r)
//   rh_f = clamp((g(rh) - g(a1))/(g(a2) - g(a1)), 0, 1)
//   bext_w/i = L[i1][t_idx] + (L[i2][t_idx] - L[i1][t_idx]) rh_f
//   aod3[i] = bext_w clamp(nwfa, 1, 99999e6) dz rhoa + bext_i clamp(nifa, 0.01, 9999e6) dz rhoa
//   aod2[c] = sum_k aod3 (f32, k ascending); type[c] = 1 if xland == 1, 3 if xland == 2 (else unwritten)
// Active: p, t, qv, dz, nwfa, nifa (in); aod3, aod2 (out, full overwrite of the c < nc rows). Inactive: xland (in, only
// selects the integer aerosol type) and type (out, integer). nc, nz are integer launch scalars.
// Frozen coefficients (every limiter-like select, taken from the trajectory in __tl/__ad; a clamp counts as the
// executed identity when it leaves the value unchanged, i.e. on the closed interval):
//   F1 t_idx = clamp(NINT(...)): frozen index (piecewise constant; the table column jumps at t ~ 225.2/255.2/285.3 K)
//   F2 x = max(-80, t - 273.16): slope 1 iff t - 273.16 >= -80, else 0
//   F3 esl = min(poly, 0.15 p): d esl = poly'(x) dx iff poly <= 0.15 p, else 0.15 dp
//   F4 rh = clamp(rh0, 10.1, 98): slope 1 iff 10.1 <= rh0 <= 98, else 0
//   F5 (i1, i2) bracket: frozen index (the interpolant is continuous across brackets; kinks at the rh_arr nodes)
//   F6 rh_f = clamp(., 0, 1): slope 1 iff 0 <= rh_f0 <= 1 (true by construction of the bracket)
//   F7 nwfa clamp [1, 99999e6], F8 nifa clamp [0.01, 9999e6]: slope 1 inside, 0 when the clamp fires
// Precision (the glue-row rule, as the live schemes): __aug repeats the forward word for word in f32 (__f*_rn, built
// --fmad=false --ftz=false) and is bitwise to the forward image. __tl/__ad recompute the f32 primal words exactly as the
// forward (the same ga_eval body), form the six local partials dA/d{p,t,qv,dz,nwfa,nifa} in double from those words (one
// shared function, so TL and AD are transposes by construction), and round once to f32 per written word. The aod2
// tangent is the double column sum of the double point tangents, rounded once; the AD adds the aod2 seed to every aod3
// seed of the column in double. One thread per column, as the forward; columns own disjoint words, so both entries are
// deterministic. failure[0] |= 1 on a non-finite partial or primal, or a non-positive p - esl, qvsat (F4 active) or t.
// __probe is harness-only (range statistics): per-point frozen-branch bits and the primal words behind them.
#define FADD(a, b) __fadd_rn((a), (b))
#define FSUB(a, b) __fsub_rn((a), (b))
#define FMUL(a, b) __fmul_rn((a), (b))
#define FDIV(a, b) __fdiv_rn((a), (b))
#define GA_NRH 8

// Verbatim from capture source 3c499c7b:3358-3377 (AER3_GT_RH_ARR, AER3_GT_LOOKUP; [rh][t][species]).
__device__ const float GA_RH_ARR[GA_NRH] = {
    10.f, 60.f, 70.f, 80.f, 85.f, 90.f, 95.f, 99.8f};
__device__ const float GA_LOOKUP[GA_NRH][4][2] = {
    {{5.73936E-15f, 2.63577E-12f}, {5.73936E-15f, 2.63577E-12f},
     {5.73936E-15f, 2.63577E-12f}, {5.73936E-15f, 2.63577E-12f}},
    {{6.93515E-15f, 2.72095E-12f}, {6.93168E-15f, 2.72092E-12f},
     {6.92570E-15f, 2.72091E-12f}, {6.91833E-15f, 2.72087E-12f}},
    {{7.24707E-15f, 2.77219E-12f}, {7.23809E-15f, 2.77222E-12f},
     {7.23108E-15f, 2.77201E-12f}, {7.21800E-15f, 2.77111E-12f}},
    {{8.95130E-15f, 2.87263E-12f}, {9.01582E-15f, 2.87252E-12f},
     {9.13216E-15f, 2.87241E-12f}, {9.16219E-15f, 2.87211E-12f}},
    {{1.06695E-14f, 2.96752E-12f}, {1.06370E-14f, 2.96726E-12f},
     {1.05999E-14f, 2.96702E-12f}, {1.05443E-14f, 2.96603E-12f}},
    {{1.37908E-14f, 3.15081E-12f}, {1.37172E-14f, 3.15020E-12f},
     {1.36362E-14f, 3.14927E-12f}, {1.35287E-14f, 3.14817E-12f}},
    {{2.26019E-14f, 3.66798E-12f}, {2.24435E-14f, 3.66540E-12f},
     {2.23254E-14f, 3.66173E-12f}, {2.20496E-14f, 3.65796E-12f}},
    {{4.41983E-13f, 7.50091E-11f}, {3.93335E-13f, 6.79097E-11f},
     {3.45569E-13f, 6.07845E-11f}, {2.96971E-13f, 5.36085E-11f}}};

// RSLF polynomial coefficients, capture source 3c499c7b:3381-3385.
#define GA_C0 .611583699E03f
#define GA_C1 .444606896E02f
#define GA_C2 .143177157E01f
#define GA_C3 .264224321E-1f
#define GA_C4 .299291081E-3f
#define GA_C5 .203154182E-5f
#define GA_C6 .702620698E-8f
#define GA_C7 .379534310E-11f
#define GA_C8 -.321582393E-13f

struct GaProbe { int bits; float rh0, qvsat, esl0, cap, rhf0, t; };

// One point. The f32 body is the forward aer3_gt_aod (3406-3454, with aer3_rslf 3379-3398 inlined and the nested
// subexpressions named; every operation and operand order is unchanged). With D, g[0..5] = dA/d(p, t, qv, dz, nwfa,
// nifa) in double, under the frozen branches F1-F8 of the trajectory.
template <bool D>
static __device__ __forceinline__ float ga_eval(float p, float dz8w, float t, float qv, float nwfa, float nifa,
                                                double* g, int* bad, GaProbe* pr)
{
    const int rind = GA_NRH;
    float rhoa = FDIV(p, FMUL(287.0f, t));
    int t_idx = max(1, min((int)roundf(FSUB(10.999f, FMUL(0.0333f, t))), 4));                       // F1
    // RSLF(p, t), capture 3379-3398
    float xr = FSUB(t, 273.16f);
    float x = fmaxf(-80.0f, xr);                                                                       // F2
    float esl = FADD(GA_C7, FMUL(x, GA_C8));
    esl = FADD(GA_C6, FMUL(x, esl));
    esl = FADD(GA_C5, FMUL(x, esl));
    esl = FADD(GA_C4, FMUL(x, esl));
    esl = FADD(GA_C3, FMUL(x, esl));
    esl = FADD(GA_C2, FMUL(x, esl));
    esl = FADD(GA_C1, FMUL(x, esl));
    esl = FADD(GA_C0, FMUL(x, esl));
    const float esl0 = esl, cap = FMUL(p, 0.15f);
    esl = fminf(esl, cap);                                                                             // F3
    float qvsat = FDIV(FMUL(0.622f, esl), FSUB(p, esl));
    float rh0 = FMUL(FDIV(qv, qvsat), 100.0f);
    float rh = fminf(98.0f, fmaxf(10.1f, rh0));                                                        // F4
    int i1, i2;                                                                                        // F5
    if (rh < 60.0f) {
        i1 = 1;
        i2 = 2;
    } else {
        int idx;
        if (rh >= 60.0f && rh < 80.0f) {
            idx = (int)roundf(FADD(FMUL(0.1f, rh), -4.0f));
        } else {
            idx = min(rind, (int)roundf(FADD(FMUL(0.2f, rh), -12.0f)));
        }
        float rh_d = FSUB(rh, GA_RH_ARR[idx - 1]);
        if (rh_d < 0.0f) {
            i1 = idx - 1;
            i2 = idx;
        } else {
            i1 = idx;
            i2 = idx + 1;
            if (i2 > rind) {
                i2 = rind;
                i1 = rind - 1;
            }
        }
    }
    float a1 = GA_RH_ARR[i1 - 1];
    float a2 = GA_RH_ARR[i2 - 1];
    const float gr = FDIV(rh, FSUB(100.0f, rh));
    const float g1 = FDIV(a1, FSUB(100.0f, a1));
    const float den = FSUB(FDIV(a2, FSUB(100.0f, a2)), g1);
    const float rhf0 = FDIV(FSUB(gr, g1), den);
    float rh_f = fmaxf(0.0f, fminf(1.0f, rhf0));                                                       // F6
    float l1w = GA_LOOKUP[i1 - 1][t_idx - 1][0];
    float l2w = GA_LOOKUP[i2 - 1][t_idx - 1][0];
    float l1i = GA_LOOKUP[i1 - 1][t_idx - 1][1];
    float l2i = GA_LOOKUP[i2 - 1][t_idx - 1][1];
    const float dw = FSUB(l2w, l1w), di = FSUB(l2i, l1i);
    float unit_bext1 = FADD(l1w, FMUL(dw, rh_f));
    float unit_bext3 = FADD(l1i, FMUL(di, rh_f));
    const float ntw = fmaxf(1.0f, fminf(99999.E6f, nwfa));                                             // F7
    float aod_wfa = FMUL(FMUL(FMUL(unit_bext1, ntw), dz8w), rhoa);
    const float nti = fmaxf(0.01f, fminf(9999.E6f, nifa));                                             // F8
    float aod_ifa = FMUL(FMUL(FMUL(unit_bext3, nti), dz8w), rhoa);
    const float aod = FADD(aod_wfa, aod_ifa);
    const bool ax = xr >= -80.0f;
    const bool aesl = esl0 <= cap;
    const bool arh = rh0 >= 10.1f && rh0 <= 98.0f;
    const bool arf = rhf0 >= 0.0f && rhf0 <= 1.0f;
    const bool anw = nwfa >= 1.0f && nwfa <= 99999.E6f;
    const bool ani = nifa >= 0.01f && nifa <= 9999.E6f;
    if (pr) {
        pr->bits = (ax ? 1 : 0) | (aesl ? 2 : 0) | (rh0 < 10.1f ? 4 : 0) | (rh0 > 98.0f ? 8 : 0) | (arf ? 16 : 0)
                 | (nwfa < 1.0f ? 32 : 0) | (nwfa > 99999.E6f ? 64 : 0) | (nifa < 0.01f ? 128 : 0)
                 | (nifa > 9999.E6f ? 256 : 0) | (t_idx << 12) | (i1 << 16);
        pr->rh0 = rh0; pr->qvsat = qvsat; pr->esl0 = esl0; pr->cap = cap; pr->rhf0 = rhf0; pr->t = t;
    }
    if (D) {
        const double P = p, T = t, QV = qv, DZ = dz8w, RHOA = rhoa, QS = qvsat, ES = esl, RH = rh, X = x;
        const double B1 = unit_bext1, B3 = unit_bext3, NW = ntw, NI = nti;
        const double dA_dB1 = NW * DZ * RHOA, dA_dB3 = NI * DZ * RHOA;
        const double dA_drhoa = (B1 * NW + B3 * NI) * DZ;
        const double dA_ddz = (B1 * NW + B3 * NI) * RHOA;
        const double dA_dnw = anw ? B1 * DZ * RHOA : 0.0;
        const double dA_dni = ani ? B3 * DZ * RHOA : 0.0;
        const double dA_drhf = arf ? dA_dB1 * (double)dw + dA_dB3 * (double)di : 0.0;
        const double om = 100.0 - RH;
        const double dA_drh = arh ? dA_drhf * (100.0 / (om * om)) / (double)den : 0.0;
        const double dA_dqv = dA_drh * 100.0 / QS;
        const double dA_dqs = -dA_drh * 100.0 * QV / (QS * QS);
        const double pe = P - ES, k622 = (double)0.622f;
        const double dA_desl = dA_dqs * (k622 * P / (pe * pe));
        double desl_dt = 0.0, desl_dp = 0.0;
        if (aesl) {
            if (ax) {   // poly'(x), Horner in double on the f32 coefficients
                double d = 8.0 * (double)GA_C8;
                d = 7.0 * (double)GA_C7 + X * d;
                d = 6.0 * (double)GA_C6 + X * d;
                d = 5.0 * (double)GA_C5 + X * d;
                d = 4.0 * (double)GA_C4 + X * d;
                d = 3.0 * (double)GA_C3 + X * d;
                d = 2.0 * (double)GA_C2 + X * d;
                desl_dt = (double)GA_C1 + X * d;
            }
        } else {
            desl_dp = (double)0.15f;
        }
        g[0] = dA_drhoa / (287.0 * T) + dA_dqs * (-k622 * ES / (pe * pe)) + dA_desl * desl_dp;
        g[1] = -dA_drhoa * P / (287.0 * T * T) + dA_desl * desl_dt;
        g[2] = dA_dqv;
        g[3] = dA_ddz;
        g[4] = dA_dnw;
        g[5] = dA_dni;
        bool ok = pe > 0.0 && T > 0.0 && (!arh || QS > 0.0) && isfinite(aod);
        for (int j = 0; j < 6; ++j) ok = ok && isfinite(g[j]);
        if (!ok) *bad = 1;
    }
    return aod;
}

extern "C" __global__ void mpas_sw_gt_aod__aug(
    int nc, int nz, const float* __restrict__ p, const float* __restrict__ t, const float* __restrict__ qv,
    const float* __restrict__ dz, const float* __restrict__ nwfa, const float* __restrict__ nifa,
    const float* __restrict__ xland, float* __restrict__ aod3, float* __restrict__ aod2, int* __restrict__ type,
    int* __restrict__ failure)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    if (xland[c] == 1.f) type[c] = 1; else if (xland[c] == 2.f) type[c] = 3;
    float sum = 0.f;
    for (int k = 0; k < nz; k++) {
        int i = c * nz + k;
        aod3[i] = ga_eval<false>(p[i], dz[i], t[i], qv[i], nwfa[i], nifa[i], nullptr, nullptr, nullptr);
        if (!isfinite(aod3[i])) atomicOr(failure, 1);
        sum = FADD(sum, aod3[i]);
    }
    aod2[c] = sum;
}

extern "C" __global__ void mpas_sw_gt_aod__tl(
    int nc, int nz, const float* __restrict__ p, const float* __restrict__ t, const float* __restrict__ qv,
    const float* __restrict__ dz, const float* __restrict__ nwfa, const float* __restrict__ nifa,
    const float* __restrict__ d_p, const float* __restrict__ d_t, const float* __restrict__ d_qv,
    const float* __restrict__ d_dz, const float* __restrict__ d_nwfa, const float* __restrict__ d_nifa,
    float* __restrict__ o_aod3, float* __restrict__ o_aod2, int* __restrict__ failure)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    double col = 0.0;
    int bad = 0;
    for (int k = 0; k < nz; k++) {
        const long long i = (long long)c * nz + k;
        double g[6];
        ga_eval<true>(p[i], dz[i], t[i], qv[i], nwfa[i], nifa[i], g, &bad, nullptr);
        double s = 0.0;   // +0.0 start: a zero tangent writes +0.0 bits
        s += g[0] * (double)d_p[i];
        s += g[1] * (double)d_t[i];
        s += g[2] * (double)d_qv[i];
        s += g[3] * (double)d_dz[i];
        s += g[4] * (double)d_nwfa[i];
        s += g[5] * (double)d_nifa[i];
        o_aod3[i] = (float)s;
        col += s;
    }
    o_aod2[c] = (float)col;
    if (bad || !isfinite(col)) atomicOr(failure, 1);
}

// Reverse. Seeds of the full-overwrite outputs (aod3, aod2) are read and consumed (set 0); the inputs accumulate.
// aod2 = sum_k aod3, so every point of the column takes the aod2 seed on top of its own, summed in double.
extern "C" __global__ void mpas_sw_gt_aod__ad(
    int nc, int nz, const float* __restrict__ p, const float* __restrict__ t, const float* __restrict__ qv,
    const float* __restrict__ dz, const float* __restrict__ nwfa, const float* __restrict__ nifa,
    float* __restrict__ b_p, float* __restrict__ b_t, float* __restrict__ b_qv,
    float* __restrict__ b_dz, float* __restrict__ b_nwfa, float* __restrict__ b_nifa,
    float* __restrict__ b_aod3, float* __restrict__ b_aod2, int* __restrict__ failure)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    const double s2 = (double)b_aod2[c];
    b_aod2[c] = 0.0f;
    int bad = 0;
    for (int k = 0; k < nz; k++) {
        const long long i = (long long)c * nz + k;
        const double b = (double)b_aod3[i] + s2;
        b_aod3[i] = 0.0f;
        double g[6];
        ga_eval<true>(p[i], dz[i], t[i], qv[i], nwfa[i], nifa[i], g, &bad, nullptr);
        b_p[i] = (float)((double)b_p[i] + g[0] * b);
        b_t[i] = (float)((double)b_t[i] + g[1] * b);
        b_qv[i] = (float)((double)b_qv[i] + g[2] * b);
        b_dz[i] = (float)((double)b_dz[i] + g[3] * b);
        b_nwfa[i] = (float)((double)b_nwfa[i] + g[4] * b);
        b_nifa[i] = (float)((double)b_nifa[i] + g[5] * b);
        if (!isfinite(b)) bad = 1;
    }
    if (bad) atomicOr(failure, 1);
}

// Harness-only: frozen-branch bits and primal words per point (range statistics). Not part of the ABI.
extern "C" __global__ void mpas_sw_gt_aod__probe(
    int nc, int nz, const float* __restrict__ p, const float* __restrict__ t, const float* __restrict__ qv,
    const float* __restrict__ dz, const float* __restrict__ nwfa, const float* __restrict__ nifa,
    int* __restrict__ bits, float* __restrict__ words)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    for (int k = 0; k < nz; k++) {
        const long long i = (long long)c * nz + k;
        GaProbe pr;
        ga_eval<false>(p[i], dz[i], t[i], qv[i], nwfa[i], nifa[i], nullptr, nullptr, &pr);
        bits[i] = pr.bits;
        float* w = words + i * 6;
        w[0] = pr.rh0; w[1] = pr.qvsat; w[2] = pr.esl0; w[3] = pr.cap; w[4] = pr.rhf0; w[5] = pr.t;
    }
}
