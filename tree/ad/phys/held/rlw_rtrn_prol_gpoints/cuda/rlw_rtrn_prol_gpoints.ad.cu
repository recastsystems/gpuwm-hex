// SPDX-License-Identifier: Apache-2.0
// Derivative entry (ad.entry/v1: aug / tl / ad) for rlw_rtrn_prol_gpoints, RRTMG LW (MPAS-A v8.4.2
// module_ra_rrtmg_lw.F rtrnmc, cloudy-layer prologue over the McICA g-points), launched from
// hexcore/mpasphys/rrtmg_lw_v842.py:826. Forward: captured source 683d7a07, lines 9145-9182; image 34da9e09
// (hrrrv5_x1.2562 captures: shared root steps 1 and 3, recaptured steps 1, 6 and 12).
//
// Forward, per word w = (col, lay, ig), col < ncol, lay = 1..nl, ig = 1..140 (one block per (col, lay)):
//   cf = cldfmc[w], tau = taucmc[w], s = secdiff[col, ngb[ig] - 1]
//   cf == 1.0f (line 9162):  od = s*tau (9164);  T = rlw_exp(-od) (9167);  A = 1 - T (9168)
//                            odcld[w] = od;  efclfrac[w] = A*cf (9169);  layer flag set
//   otherwise:               odcld[w] = 0;  efclfrac[w] = 0 (9172-9173)
//   icldlyr[col, lay] |= 1 when any g-point of the layer is cloudy (__ballot_sync + atomicOr, 9176-9178).
// Active: cldfmc, taucmc, secdiff (in) -> odcld, efclfrac (out, full overwrite of every word of the launch's
// (ncol, nl, 140) range). Inactive: ncol, nl (scalars), ngb (band of each g-point, a table), icldlyr (integer flag).
//
// Precision (the glue-row rule, as the rlw_rtrn_secdiff entry): __aug repeats the forward word for word in f32
// (__f*_rn and the forward's own glibc 2.39 expf transcription, built --fmad=false --ftz=false), so its outputs
// and icldlyr are bitwise the forward image's. __tl/__ad form the local partials in double from the f32 primal
// words (d od = tau ds + s dtau; d efclfrac = cf T d od + A dcf, T the f32 exp word as double) and round once to
// f32 per written word; +0.0f maps -0.0 to +0.0 so TL(0) is exact zero bits.
//
// Frozen-coefficient treatment: the McICA mask test cf == 1.0f (9162) is a step in cf, so it is frozen on the
// trajectory. The cloudy branch differentiates A*cf as A dcf + cf dA; the clear branch has zero derivative in
// every argument (its outputs are the constant 0). cldfmc is 0 or 1 by construction (mpas_lw_mcica_gpoint
// thresholds the random stream against cloud fraction), so its tangent is formal: upstream it is zero almost
// everywhere.
//
// Adjoint: secdiff[col, b] fans in from every cloudy (lay, ig) of band b in the column. One block per (col, lay)
// is the forward geometry; the lay == 1 block of each column does the whole column in a fixed order (per g-point
// double accumulators in shared memory, each owned by one thread, layers ascending; then one thread per band sums
// its g-points in ascending ig, rounds once and adds into b_secdiff). The other blocks return at once. No atomics,
// disjoint words per thread for b_cldfmc and b_taucmc: the adjoint is deterministic. Every output cotangent word
// in the launch's range is consumed (zeroed).
//
// Refusal (failure slot 0): a non-finite od or efclfrac word on the cloudy branch (non-finite tau or secdiff),
// or a non-finite written tangent or adjoint word.
// ---- verbatim from the captured forward source 683d7a07 (lines 92-94, 100, 186-193, 220-235, 292-302, 324-359; GFK_ renamed PG_) -----------------------
#define FADD(a, b) __fadd_rn((a), (b))
#define DADD(a, b) __dadd_rn((a), (b))
#define DSUB(a, b) __dsub_rn((a), (b))
#define DMUL(a, b) __dmul_rn((a), (b))
#define RLW_AD(a, b) __fadd_rn((a), (b))
#define RLW_MU(a, b) __fmul_rn((a), (b))
#define NBNDLW 16

__device__ const unsigned long long PG_EXP2F_TAB[32] = {
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
#define PG_EXP2F_P0 0x1.c6af84b912394p-5
#define PG_EXP2F_P1 0x1.ebfce50fac4f3p-3
#define PG_EXP2F_P2 0x1.62e42ff0c52d6p-1
#define PG_EXP2F_SHIFT 0x1.8p+52

__device__ __forceinline__ float pg_d2f_rn(double y)          // = gfk_d2f_rn
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

__device__ __forceinline__ float pg_exp(float x)               // = gfk_exp = rlw_exp (glibc 2.39 e_expf.c)
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
    double kd = __fma_rn(invln2n, xd, PG_EXP2F_SHIFT);
    unsigned long long ki = (unsigned long long)__double_as_longlong(kd);
    kd = DSUB(kd, PG_EXP2F_SHIFT);
    double r = __fma_rn(invln2n, xd, -kd);
    unsigned long long t = PG_EXP2F_TAB[ki & 31ULL];
    t += ki << (52 - 5);
    double s = __longlong_as_double((long long)t);
    double z = __fma_rn(PG_EXP2F_P0 / 32.0 / 32.0 / 32.0, r,
                        PG_EXP2F_P1 / 32.0 / 32.0);
    double r2 = DMUL(r, r);
    double y = __fma_rn(PG_EXP2F_P2 / 32.0, r, 1.0);
    y = __fma_rn(z, r2, y);
    return pg_d2f_rn(DMUL(y, s));
}
// ---- end verbatim --------------------------------------------------------------------------------------------------
#define RLW_SU(a, b) __fsub_rn((a), (b))
#define NGPTLW 140
#define PG_W(a) a[((long long)col * nl + (lay - 1)) * NGPTLW + (ig - 1)]

extern "C" __global__ void rlw_rtrn_prol_gpoints__aug(
    int ncol, int nl, const float* __restrict__ cldfmc, const float* __restrict__ taucmc,
    const float* __restrict__ secdiff, const int* __restrict__ ngb, float* __restrict__ odcld,
    float* __restrict__ efclfrac, int* __restrict__ icldlyr, int* failure)
{
    long long tid = blockIdx.x;
    if (tid >= (long long)ncol * nl) return;
    int col = (int)(tid / nl);
    int lay = (int)(tid % nl) + 1;
    int flag = 0;
    bool bad = false;
    for (int ig = threadIdx.x + 1; ig <= NGPTLW; ig += blockDim.x) {
        if (PG_W(cldfmc) == 1.0f) {
            int ib = ngb[ig - 1];
            float od = RLW_MU(secdiff[(long long)col * NBNDLW + (ib - 1)], PG_W(taucmc));
            PG_W(odcld) = od;
            float transcld = pg_exp(-od);
            float abscld = RLW_SU(1.0f, transcld);
            float e = RLW_MU(abscld, PG_W(cldfmc));
            PG_W(efclfrac) = e;
            bad |= !isfinite(od) || !isfinite(e);
            flag = 1;
        } else {
            PG_W(odcld) = 0.0f;
            PG_W(efclfrac) = 0.0f;
        }
    }
    unsigned int cloudy = __ballot_sync(0xffffffff, flag);
    if (threadIdx.x % 32 == 0 && cloudy)
        atomicOr(&icldlyr[(long long)col * nl + (lay - 1)], 1);
    if (bad) atomicExch(failure, 1);
}

extern "C" __global__ void rlw_rtrn_prol_gpoints__tl(
    int ncol, int nl, const float* __restrict__ cldfmc, const float* __restrict__ taucmc,
    const float* __restrict__ secdiff, const int* __restrict__ ngb,
    const float* __restrict__ d_cldfmc, const float* __restrict__ d_taucmc, const float* __restrict__ d_secdiff,
    float* __restrict__ d_odcld, float* __restrict__ d_efclfrac, int* failure)
{
    long long tid = blockIdx.x;
    if (tid >= (long long)ncol * nl) return;
    int col = (int)(tid / nl);
    int lay = (int)(tid % nl) + 1;
    bool bad = false;
    for (int ig = threadIdx.x + 1; ig <= NGPTLW; ig += blockDim.x) {
        if (PG_W(cldfmc) == 1.0f) {                                  // frozen: the executed branch
            int ib = ngb[ig - 1];
            long long si = (long long)col * NBNDLW + (ib - 1);
            float s = secdiff[si], tau = PG_W(taucmc), cf = PG_W(cldfmc);
            float od = RLW_MU(s, tau);
            float transcld = pg_exp(-od);
            float abscld = RLW_SU(1.0f, transcld);
            double dod = (double)d_secdiff[si] * (double)tau + (double)s * (double)PG_W(d_taucmc);   // od = s*tau
            double dab = (double)transcld * dod;                                                   // A = 1 - exp(-od)
            double def = dab * (double)cf + (double)abscld * (double)PG_W(d_cldfmc);                // efclfrac = A*cf
            float to = __fadd_rn((float)dod, 0.0f), te = __fadd_rn((float)def, 0.0f);
            bad |= !isfinite(to) || !isfinite(te) || !isfinite(od);
            PG_W(d_odcld) = to;
            PG_W(d_efclfrac) = te;
        } else {
            PG_W(d_odcld) = 0.0f;
            PG_W(d_efclfrac) = 0.0f;
        }
    }
    if (bad) atomicExch(failure, 1);
}

extern "C" __global__ void rlw_rtrn_prol_gpoints__ad(
    int ncol, int nl, const float* __restrict__ cldfmc, const float* __restrict__ taucmc,
    const float* __restrict__ secdiff, const int* __restrict__ ngb,
    float* __restrict__ b_cldfmc, float* __restrict__ b_taucmc, float* __restrict__ b_secdiff,
    float* __restrict__ b_odcld, float* __restrict__ b_efclfrac, int* failure)
{
    __shared__ double acc[NGPTLW];                // per g-point cotangent of s, summed over the column's layers
    long long tid = blockIdx.x;
    if (tid >= (long long)ncol * nl) return;
    if ((int)(tid % nl) != 0) return;              // the lay == 1 block owns the whole column
    int col = (int)(tid / nl);
    bool bad = false;
    for (int ig = threadIdx.x + 1; ig <= NGPTLW; ig += blockDim.x) acc[ig - 1] = 0.0;
    for (int lay = 1; lay <= nl; ++lay) {
        for (int ig = threadIdx.x + 1; ig <= NGPTLW; ig += blockDim.x) {
            if (PG_W(cldfmc) == 1.0f) {
                int ib = ngb[ig - 1];
                float s = secdiff[(long long)col * NBNDLW + (ib - 1)], tau = PG_W(taucmc), cf = PG_W(cldfmc);
                float od = RLW_MU(s, tau);
                float transcld = pg_exp(-od);
                float abscld = RLW_SU(1.0f, transcld);
                double bef = (double)PG_W(b_efclfrac), bod = (double)PG_W(b_odcld);
                double bodt = bod + (double)transcld * (bef * (double)cf);       // efclfrac = (1 - exp(-od)) cf
                float bc = __fadd_rn(PG_W(b_cldfmc), (float)(bef * (double)abscld));
                float bt = __fadd_rn(PG_W(b_taucmc), (float)(bodt * (double)s));
                bad |= !isfinite(bc) || !isfinite(bt) || !isfinite(od);
                PG_W(b_cldfmc) = bc;
                PG_W(b_taucmc) = bt;
                acc[ig - 1] += bodt * (double)tau;
            }
            PG_W(b_odcld) = 0.0f;                                                   // both outputs fully written
            PG_W(b_efclfrac) = 0.0f;
        }
    }
    __syncthreads();
    for (int b = threadIdx.x + 1; b <= NBNDLW; b += blockDim.x) {
        double sum = 0.0;
        for (int ig = 1; ig <= NGPTLW; ++ig) if (ngb[ig - 1] == b) sum += acc[ig - 1];
        float v = __fadd_rn(b_secdiff[(long long)col * NBNDLW + (b - 1)], (float)sum);
        bad |= !isfinite(v);
        b_secdiff[(long long)col * NBNDLW + (b - 1)] = v;
    }
    if (bad) atomicExch(failure, 1);
}
#undef PG_W
