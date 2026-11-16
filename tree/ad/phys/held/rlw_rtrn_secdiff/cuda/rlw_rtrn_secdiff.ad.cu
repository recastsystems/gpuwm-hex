// SPDX-License-Identifier: Apache-2.0
// Derivative entry (ad.entry/v1: aug / tl / ad) for rlw_rtrn_secdiff, RRTMG LW (MPAS-A v8.4.2 module_ra_rrtmg_lw.F
// rtrnmc, diffusivity angle), launched from hexcore/mpasphys/rrtmg_lw_v842.py:817-820.  Forward: captured source
// 683d7a07 (lines 1862-1884), image 34da9e09 (hrrrv5_x1.2562 captures, steps 1 and 3).
//
// Forward, per column col < ncol, per band ibnd = 1..16 (forward lines 1871-1883):
//   ibnd in {1, 4, 10..16}:  sd = 1.66f                                       (constant, zero tangent)
//   otherwise:               sd = a0[b] + a1[b] * expf(a2[b] * pwvcm)          (f32, __f*_rn, glibc expf)
//                            if (sd > 1.80f) sd = 1.80f;  if (sd < 1.50f) sd = 1.50f;
//   secdiff[col*16 + b] = sd          (full overwrite of the 16 words of the row)
// Active: pwvcm_v (in) -> secdiff (out).  a0/a1/a2 are the RRTMG band constants (rrtmg_lw_v842.py K table, uploaded
// once, never written by any launch): inactive, as in the P7 spec (tree/ad/phys/p7/specs/rlw_rtrn_secdiff.json).
//
// Precision (the glue-row rule): __aug repeats the forward word for word in f32 (__f*_rn and the forward's own glibc
// 2.39 expf transcription, built --fmad=false --ftz=false), so its outputs are bitwise the forward image's.  __tl/__ad
// form the local partials in double from the f32 primal words (d sd / d pwvcm = a1 * a2 * e, e the f32 exp word) and
// round once to f32 per written word.
//
// Limiter-like clamps, frozen-coefficient treatment (the branch the forward took is the branch differentiated):
//   C1 (forward line 1879) if (sd > 1.80f) sd = 1.80f;   C2 (forward line 1880) if (sd < 1.50f) sd = 1.50f;
//   The pwvcm tangent passes iff 1.50f <= raw <= 1.80f, raw the f32 word before the clamps, recomputed bitwise here
//   (the output word equals the raw word, ties included).  A clamped word is the constant 1.80 or 1.50: zero tangent.
//   A NaN raw word (the forward writes NaN, both tests false) passes neither test and raises the failure flag.
// Adjoint: one thread per column sums its band contributions in fixed band order in double, rounds once, adds into
// b_pwvcm[col]; the 16 secdiff cotangent words of the row are consumed (zeroed).  Disjoint words per thread, no
// atomics, so the adjoint is deterministic.

// ---- verbatim from the captured forward source 683d7a07 (lines 92-94, 100, 186-193, 220-235, 292-302, 324-359; GFK_ renamed SD_) -----------------------
#define FADD(a, b) __fadd_rn((a), (b))
#define DADD(a, b) __dadd_rn((a), (b))
#define DSUB(a, b) __dsub_rn((a), (b))
#define DMUL(a, b) __dmul_rn((a), (b))
#define RLW_AD(a, b) __fadd_rn((a), (b))
#define RLW_MU(a, b) __fmul_rn((a), (b))
#define NBNDLW 16

__device__ const unsigned long long SD_EXP2F_TAB[32] = {
    0x3ff0000000000000ULL, 0x3fefd9b0d3158574ULL, 0x3fefb5586cf9890fULL,
    0x3fef9301d0125b51ULL, 0x3fef72b83c7d517bULL, 0x3fef54873168b9aaULL,
    0x3fef387a6e756238ULL, 0x3fef1e9df51fdee1ULL, 0x3fef06fe0a31b715ULL,
    0x3feef1a7373aa9cbULL, 0x3feedea64c123422ULL, 0x3feece086061892dULL,
    0x3feebfdad5362a27ULL, 0x3feeb42b569d4f82ULL, 0x3feeab07dd485429ULL,
    0x3feea47eb03a5585ULL, 0x3feea09e667f3bcdULL, 0x3fee9f75e8ec5f74ULL,
    0x3feea11473eb0187ULL, 0x3feea589994cce13ULL, 0x3feeace5422aa0dbULL,
    0x3feeb737b0cdc5e5ULL, 0x3feec49182a3f090ULL, 0x3feed503b23e255dULL,
    0x3feee89f995ad3adULL, 0x3feeff76f2fb5e47ULL, 0x3fef199bdd85529cULL,
    0x3fef3720dcef9069ULL, 0x3fef5818dcfba487ULL, 0x3fef7c97337b9b5fULL,
    0x3fefa4afa2a490daULL, 0x3fefd0765b6e4540ULL };
#define SD_EXP2F_P0 0x1.c6af84b912394p-5
#define SD_EXP2F_P1 0x1.ebfce50fac4f3p-3
#define SD_EXP2F_P2 0x1.62e42ff0c52d6p-1
#define SD_EXP2F_SHIFT 0x1.8p+52

__device__ __forceinline__ float sd_d2f_rn(double y)          // = gfk_d2f_rn
{
    double a = fabs(y);
    if (a > 0.0 && a < 1.1754943508222875e-38) {
        double scaled = rint(a * 7.1362384635297994e+44);
        unsigned int m = (unsigned int)scaled;
        unsigned int s = (__double_as_longlong(y) < 0LL) ? 0x80000000u : 0u;
        return __uint_as_float(s | m);
    }
    return __double2float_rn(y);
}

__device__ __forceinline__ float sd_exp(float x)               // = gfk_exp = rlw_exp (glibc 2.39 e_expf.c)
{
    unsigned int abstop = (__float_as_uint(x) >> 20) & 0x7ffu;
    if (abstop >= ((__float_as_uint(88.0f)) >> 20)) {
        if (__float_as_uint(x) == 0xff800000u) return 0.0f;
        if (abstop >= (0x7f800000u >> 20)) return FADD(x, x);
        if (x > __int_as_float(0x42b17218)) return __int_as_float(0x7f800000);
        if (x < -__int_as_float(0x42cff1b4)) return 0.0f;
    }
    double xd = (double)x;
    const double invln2n = 0x1.71547652b82fep+0 * 32.0;
    double kd = __fma_rn(invln2n, xd, SD_EXP2F_SHIFT);
    unsigned long long ki = (unsigned long long)__double_as_longlong(kd);
    kd = DSUB(kd, SD_EXP2F_SHIFT);
    double r = __fma_rn(invln2n, xd, -kd);
    unsigned long long t = SD_EXP2F_TAB[ki & 31ULL];
    t += ki << (52 - 5);
    double s = __longlong_as_double((long long)t);
    double z = __fma_rn(SD_EXP2F_P0 / 32.0 / 32.0 / 32.0, r,
                        SD_EXP2F_P1 / 32.0 / 32.0);
    double r2 = DMUL(r, r);
    double y = __fma_rn(SD_EXP2F_P2 / 32.0, r, 1.0);
    y = __fma_rn(z, r2, y);
    return sd_d2f_rn(DMUL(y, s));
}
// ---- end verbatim --------------------------------------------------------------------------------------------------

__device__ __forceinline__ bool sd_const_band(int ibnd) { return ibnd == 1 || ibnd == 4 || ibnd >= 10; }

extern "C" __global__ void rlw_rtrn_secdiff__aug(
    int ncol, const float* __restrict__ pwvcm_v, const float* __restrict__ a0, const float* __restrict__ a1,
    const float* __restrict__ a2, float* __restrict__ secdiff, int* failure)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    float pwvcm = pwvcm_v[col];
    bool bad = !isfinite(pwvcm);
    for (int ibnd = 1; ibnd <= NBNDLW; ++ibnd) {
        float sd;
        if (sd_const_band(ibnd)) {
            sd = 1.66f;
        } else {
            sd = RLW_AD(a0[ibnd - 1], RLW_MU(a1[ibnd - 1], sd_exp(RLW_MU(a2[ibnd - 1], pwvcm))));
            if (sd > 1.80f) sd = 1.80f;
            if (sd < 1.50f) sd = 1.50f;
            bad |= !isfinite(sd);
        }
        secdiff[(long long)col * NBNDLW + (ibnd - 1)] = sd;
    }
    if (bad) atomicExch(failure, 1);
}

// d sd_b / d pwvcm on the frozen branch (0 for constant or clamped bands).  *bad set for a non-finite word.
__device__ __forceinline__ double sd_partial(int ibnd, float pwvcm, const float* a0, const float* a1, const float* a2,
                                             bool* bad)
{
    if (sd_const_band(ibnd)) return 0.0;
    float m = RLW_MU(a2[ibnd - 1], pwvcm);
    float e = sd_exp(m);
    float raw = RLW_AD(a0[ibnd - 1], RLW_MU(a1[ibnd - 1], e));
    if (!isfinite(raw) || !isfinite(e)) { *bad = true; return 0.0; }
    if (!(raw >= 1.50f && raw <= 1.80f)) return 0.0;              // C1 / C2 clamped: frozen constant
    return (double)a1[ibnd - 1] * (double)a2[ibnd - 1] * (double)e;
}

extern "C" __global__ void rlw_rtrn_secdiff__tl(
    int ncol, const float* __restrict__ pwvcm_v, const float* __restrict__ a0, const float* __restrict__ a1,
    const float* __restrict__ a2, const float* __restrict__ d_pwvcm, float* __restrict__ d_secdiff, int* failure)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    float pwvcm = pwvcm_v[col];
    bool bad = !isfinite(pwvcm);
    double dp = (double)d_pwvcm[col];
    for (int ibnd = 1; ibnd <= NBNDLW; ++ibnd) {
        double g = sd_partial(ibnd, pwvcm, a0, a1, a2, &bad);
        // + 0.0f maps the -0.0 of a negative partial times a zero direction to +0.0, so TL(0) is exact zero bits
        float t = __fadd_rn((float)(g * dp), 0.0f);
        bad |= !isfinite(t);
        d_secdiff[(long long)col * NBNDLW + (ibnd - 1)] = t;
    }
    if (bad) atomicExch(failure, 1);
}

extern "C" __global__ void rlw_rtrn_secdiff__ad(
    int ncol, const float* __restrict__ pwvcm_v, const float* __restrict__ a0, const float* __restrict__ a1,
    const float* __restrict__ a2, float* __restrict__ b_pwvcm, float* __restrict__ b_secdiff, int* failure)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    float pwvcm = pwvcm_v[col];
    bool bad = !isfinite(pwvcm);
    double acc = 0.0;
    for (int ibnd = 1; ibnd <= NBNDLW; ++ibnd) {
        long long w = (long long)col * NBNDLW + (ibnd - 1);
        double y = (double)b_secdiff[w];
        bad |= !isfinite(y);
        double g = sd_partial(ibnd, pwvcm, a0, a1, a2, &bad);
        if (g != 0.0) acc += g * y;
        b_secdiff[w] = 0.0f;                                        // full overwrite: seed consumed
    }
    float r = (float)((double)b_pwvcm[col] + acc);
    bad |= !isfinite(r);
    b_pwvcm[col] = r;
    if (bad) atomicExch(failure, 1);
}
