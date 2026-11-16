// SPDX-License-Identifier: Apache-2.0
// Derivative entry (ad.entry/v1: aug / tl / ad) for mpas_lw_mcica_seeds, RRTMG LW McICA (MPAS-A v8.4.2
// module_ra_rrtmg_lw.F:2432-2440 KISS seeds, 2252-2697 exponential-overlap weights; wrapper 12175-12185, IRNG=0,
// seed=150), launched from hexcore/mpasphys/rrtmg_lw_v842.py:1110 (_lw_mcica_wrapper).  Forward: captured source
// 683d7a07 (lines 9551-9568), image 34da9e09 (hrrrv5_x1.2562 captures, steps 1, 3, 6, 12).
//
// Forward, per wrapper column c < nc (a = abase + c*12*nl, A(f,k) = a[f*nl+k]):
//   seeds:   seed[k] = (u32)(int)((p - (float)(int)p) * 1e9),  p = A(0,k)*100 (Pa), k < 4;  then permuteseed KISS draws;
//            seeds[c*4+k] = seed[k]                                                        (integer McICA state)
//   overlap < 4: return (weights is a one-word dummy, not written).
//   overlap >= 4: L = 2500, or (idcor == 1) L = 1000 (1.4315 + 2.1219 exp(-(lat - am3)^2 / 25.584^2)), lat = surface[c*6+5];
//            weights[c*nl+k] = k ? exp(-(A(2,k) - A(2,k-1)) / L) : 0   (A(2,k) the layer-centre height, m)
// Active: abase field 2 (heights) and surface word 5 (xlat) -> weights.  Every other abase word (pressure A(0,k) only
// reaches the integer seeds) has zero derivative.  seeds are integers: inactive, saved (the trajectory's McICA sample
// state; see SAVED below).
//
// SAVED (the hold reason "McICA samples must be saved").  The seeds are a piecewise-constant function of the pressure
// words: they jump whenever (p*100 - int(p*100))*1e9 crosses an integer, i.e. at every ulp of p for real pressures
// (measured: one-ulp pressure moves change the seeds of every column; RESULT.md).  The true derivative is zero almost
// everywhere and the function is discontinuous at ulp scale, so the only usable linearisation is the frozen one: the
// trajectory's seeds, written by __aug bitwise to the forward, are the McICA samples of the linearisation, and TL/AD
// never recompute them from a perturbed state.  A nonlinear difference quotient through this kernel (Taylor, FD) must
// hold the seeds (or the cloud masks downstream in mpas_lw_mcica_gpoint) at the trajectory's, or it resamples McICA
// and does not converge.
//
// Precision (the glue-row rule, as the live radiation entries): __aug repeats the forward word for word in f32
// (__f*_rn and the forward's own glibc 2.39 expf transcription, built --fmad=false --ftz=false --prec-div=true
// --prec-sqrt=true), so its seeds and weights are bitwise the forward image's.  __tl/__ad form the local partials in
// double from the f32 primal words (dz, inverse, w, delta, e) and round once to f32 per written word.
//
// Limiter-like elements, frozen treatment (the branch the forward took is the branch differentiated):
//   - the float->int truncations of the seeds (line 9556): piecewise constant -> zero derivative, seeds frozen (SAVED);
//   - julday > 181 (line 9561): an integer selector, inactive;
//   - k ? ... : 0 (line 9567): index predicate, the k = 0 word is the constant 0 (tangent +0, cotangent consumed);
//   - overlap < 4 early return (line 9558): integer selector; no float output, so no derivative words at all.
// The kernel has no clamp or limiter on a float value.

// ---- verbatim from the captured forward source 683d7a07 (lines 92-95, 186-194, 219-236, 292-303, 323-359,
// 9464-9473, 9293; names kept) -----------------------------------------------------------------------------
#define RLW_AD(a, b) __fadd_rn((a), (b))
#define RLW_SU(a, b) __fsub_rn((a), (b))
#define RLW_MU(a, b) __fmul_rn((a), (b))
#define RLW_DV(a, b) __fdiv_rn((a), (b))

#define FADD(a, b) __fadd_rn((a), (b))
#define FSUB(a, b) __fsub_rn((a), (b))
#define FMUL(a, b) __fmul_rn((a), (b))
#define FDIV(a, b) __fdiv_rn((a), (b))
#define FSQRT(a)   __fsqrt_rn(a)
#define DADD(a, b) __dadd_rn((a), (b))
#define DSUB(a, b) __dsub_rn((a), (b))
#define DMUL(a, b) __dmul_rn((a), (b))
#define DDIV(a, b) __ddiv_rn((a), (b))

// e_exp2f_data.c, shared by expf / exp2f / powf.  EXP2F_TABLE_BITS = 5.
__device__ const unsigned long long GFK_EXP2F_TAB[32] = {
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
#define GFK_EXP2F_P0 0x1.c6af84b912394p-5
#define GFK_EXP2F_P1 0x1.ebfce50fac4f3p-3
#define GFK_EXP2F_P2 0x1.62e42ff0c52d6p-1
#define GFK_EXP2F_SHIFT 0x1.8p+52
#define GFK_EXP2F_SHIFT_SCALED (0x1.8p+52 / 32.0)

__device__ float gfk_d2f_rn(double y)
{
    double a = fabs(y);
    if (a > 0.0 && a < 1.1754943508222875e-38) {   /* 0x1p-126 */
        double scaled = rint(a * 7.1362384635297994e+44);   /* 2^149 */
        unsigned int m = (unsigned int)scaled;
        unsigned int s = (__double_as_longlong(y) < 0LL) ? 0x80000000u : 0u;
        return __uint_as_float(s | m);
    }
    return __double2float_rn(y);
}


// glibc 2.39 sysdeps/ieee754/flt-32/e_expf.c
__device__ float gfk_exp(float x)
{
    unsigned int abstop = (__float_as_uint(x) >> 20) & 0x7ffu;
    if (abstop >= ((__float_as_uint(88.0f)) >> 20)) {
        if (__float_as_uint(x) == 0xff800000u) return 0.0f;
        if (abstop >= (0x7f800000u >> 20)) return FADD(x, x);
        if (x > __int_as_float(0x42b17218)) return __int_as_float(0x7f800000);
        if (x < -__int_as_float(0x42cff1b4)) return 0.0f;
    }
    // The reference expf on an x86-64 host with FMA is the FMA build of this
    // algorithm, and in that build the product InvLn2N*x, used only by the
    // add that forms kd and the subtract that forms r, is fused into both:
    // r keeps the product's rounding error.  Spelled here with __fma_rn, and
    // the three polynomial multiply-adds likewise.  MEASURED on box W1 (Xeon
    // Platinum 8559C, glibc 2.39) over all 2^32 inputs
    // (tools/noah_wrf461_oracle/expf_variant_probe.c): this form equals the
    // host expf on every input; the separately rounded form it replaces
    // differed on two (0x4202422f and 0xc27c65d9, 1 ULP each), and fusing
    // only the polynomial did not move either.  gfk_exp2_core, which powf
    // also uses, is left as it was: powf's reduction has no product to fuse.
    double xd = (double)x;
    const double invln2n = 0x1.71547652b82fep+0 * 32.0;
    double kd = __fma_rn(invln2n, xd, GFK_EXP2F_SHIFT);
    unsigned long long ki = (unsigned long long)__double_as_longlong(kd);
    kd = DSUB(kd, GFK_EXP2F_SHIFT);
    double r = __fma_rn(invln2n, xd, -kd);
    unsigned long long t = GFK_EXP2F_TAB[ki & 31ULL];
    t += ki << (52 - 5);
    double s = __longlong_as_double((long long)t);
    double z = __fma_rn(GFK_EXP2F_P0 / 32.0 / 32.0 / 32.0, r,
                        GFK_EXP2F_P1 / 32.0 / 32.0);
    double r2 = DMUL(r, r);
    double y = __fma_rn(GFK_EXP2F_P2 / 32.0, r, 1.0);
    y = __fma_rn(z, r2, y);
    return gfk_d2f_rn(DMUL(y, s));
}

#define rlw_exp gfk_exp
#define A(f,k) a[(f)*nl+(k)]

// MPAS: module_ra_rrtmg_lw.F:2705-2737.
__device__ float mpas_kiss(unsigned int* seed){
    // MPAS: module_ra_rrtmg_lw.F:2705-2737. Unsigned implements wrap/ISHFT.
    seed[0]=69069u*seed[0]+1327217885u;
    seed[1]^=seed[1]<<13;seed[1]^=seed[1]>>17;seed[1]^=seed[1]<<5;
    seed[2]=18000u*(seed[2]&65535u)+(seed[2]>>16);
    seed[3]=30903u*(seed[3]&65535u)+(seed[3]>>16);
    int kiss=(int)(seed[0]+seed[1]+(seed[2]<<16)+seed[3]);
    return RLW_AD(RLW_MU((float)kiss,2.328306e-10f),0.5f);
}
// ---- end verbatim --------------------------------------------------------------------------------------------------

// Range-overlap test for the refusal of aliased views (alias_group null in the spec: every view distinct).
__device__ __forceinline__ bool mcs_overlap(const void* a, long long na, const void* b, long long nb)
{
    const char *pa = (const char*)a, *pb = (const char*)b;
    return pa < pb + nb && pb < pa + na;
}
__device__ __forceinline__ bool mcs_aliased(int nc, int nl, int overlap, const void* surface, const void* abase,
                                            const void* weights)
{
    long long ns = (long long)nc * 6 * 4, na = (long long)nc * 12 * nl * 4;
    long long nw = overlap >= 4 ? (long long)nc * nl * 4 : 4;
    return mcs_overlap(surface, ns, abase, na) || mcs_overlap(surface, ns, weights, nw) ||
           mcs_overlap(abase, na, weights, nw);
}

// The forward's decorrelation length (rrtmg_lw_v842.py:515-521, capture 683d7a07:9560-9564), word for word, plus the
// two f32 words the derivative needs (delta and the exp word e).
__device__ __forceinline__ float mcs_length(int idcor, int julday, float lat, float* delta_o, float* e_o)
{
    float length = 2500.0f;
    *delta_o = 0.0f; *e_o = 0.0f;
    if (idcor == 1) {
        float am3 = RLW_MU(julday > 181 ? (-28.0f / 365.0f) : (28.0f / 365.0f),
                           (float)(julday - (julday > 181 ? 272 : 91)));
        float delta = RLW_SU(lat, am3);
        float e = rlw_exp(RLW_DV(-RLW_MU(delta, delta), (-25.584f * -25.584f)));
        length = RLW_MU(RLW_AD(1.4315f, RLW_MU(2.1219f, e)), 1000.0f);
        *delta_o = delta; *e_o = e;
    }
    return length;
}

// d inverse / d lat on the f32 primal words, in double (0 when idcor != 1).
//   inverse = 1/L, L = 1000 (1.4315 + 2.1219 e), e = exp(q), q = -delta^2 / C, C = 25.584^2 (the forward's f32 word),
//   delta = lat - am3:  d inverse/d lat = -(1/L^2) * 1000 * 2.1219 * e * (-2 delta / C)
__device__ __forceinline__ double mcs_dinv_dlat(int idcor, float length, float delta, float e)
{
    if (idcor != 1) return 0.0;
    const double C = (double)(-25.584f * -25.584f);
    double L = (double)length;
    return -(1.0 / (L * L)) * 1000.0 * 2.1219 * (double)e * (-2.0 * (double)delta / C);
}

// One layer's weight, the forward's f32 words (k >= 1):  w_k = exp(-(z_k - z_{k-1}) * inverse).
// Partials (frozen at these words):  dw/dz_k = -w*inverse,  dw/dz_{k-1} = +w*inverse,  dw/dinverse = -w*dz.
__device__ __forceinline__ float mcs_weight(const float* a, int nl, int k, float inverse, float* dz_o)
{
    float dz = RLW_SU(A(2, k), A(2, k - 1));
    *dz_o = dz;
    return rlw_exp(RLW_MU(-dz, inverse));
}

// ------------------------------------------------------------------------------------------------------------------
// __aug: the forward (capture 683d7a07:9551-9568) word for word (seeds and weights bitwise to the forward image), plus
// the refusal flag: a non-finite or out-of-int-range pressure word among the four seed layers (the float->int
// conversion of such a word is implementation-defined), a non-finite weight, or aliased views.
extern "C" __global__ void mpas_lw_mcica_seeds__aug(int nc, int nl, int overlap, int idcor, int julday, int permuteseed,
    const float* surface, const float* abase, unsigned int* seeds, float* weights, int* failure)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= nc) return;
    if (c == 0 && mcs_aliased(nc, nl, overlap, surface, abase, weights)) atomicExch(failure, 1);
    const float* a = abase + (long long)c * 12 * nl;
    bool bad = false;
    unsigned int seed[4];
    for (int k = 0; k < 4; ++k) {
        seed[k] = 0u;
        if (k < nl) {
            float p = RLW_MU(A(0, k), 100.0f);
            bad |= !isfinite(p) || fabsf(p) >= 2.1e9f;
            seed[k] = (unsigned int)(int)RLW_MU(RLW_SU(p, (float)(int)p), 1.e9f);
        }
    }
    for (int j = 0; j < permuteseed; ++j) mpas_kiss(seed);
    for (int k = 0; k < 4; ++k) seeds[c * 4 + k] = seed[k];
    if (overlap < 4) { if (bad) atomicExch(failure, 1); return; }
    float delta, e;
    float length = mcs_length(idcor, julday, surface[c * 6 + 5], &delta, &e);
    float inverse = RLW_DV(1.0f, length);
    for (int k = 0; k < nl; ++k) {
        float w = 0.0f, dz;
        if (k) w = mcs_weight(a, nl, k, inverse, &dz);
        bad |= !isfinite(w);
        weights[(long long)c * nl + k] = w;
    }
    bad |= !isfinite(inverse);
    if (bad) atomicExch(failure, 1);
}

// __tl: seeds are integers (the McICA sample state), so they carry no tangent: they are the trajectory's, saved by
// __aug, and neither derivative entry reads or writes them.  For overlap < 4 the forward writes no float word (weights
// is a one-word dummy), so TL writes nothing.  For overlap >= 4, per column, every one of the nl weight words is
// written (k = 0 is the constant 0: tangent +0).  Local partials in double from the f32 primal words, one rounding
// per written word.
extern "C" __global__ void mpas_lw_mcica_seeds__tl(int nc, int nl, int overlap, int idcor, int julday, int permuteseed,
    const float* surface, const float* abase, const float* d_surface, const float* d_abase, float* d_weights,
    int* failure)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= nc) return;
    if (overlap < 4) return;
    if (c == 0 && (mcs_aliased(nc, nl, overlap, surface, abase, d_weights) ||
                   mcs_aliased(nc, nl, overlap, d_surface, d_abase, d_weights))) atomicExch(failure, 1);
    const float* a = abase + (long long)c * 12 * nl;
    const float* da = d_abase + (long long)c * 12 * nl;
    float delta, e;
    float length = mcs_length(idcor, julday, surface[c * 6 + 5], &delta, &e);
    float inverse = RLW_DV(1.0f, length);
    double dinv = idcor == 1 ? mcs_dinv_dlat(idcor, length, delta, e) * (double)d_surface[c * 6 + 5] : 0.0;
    bool bad = !isfinite(inverse) || !isfinite(dinv);
    d_weights[(long long)c * nl] = 0.0f;
    for (int k = 1; k < nl; ++k) {
        float dz, w = mcs_weight(a, nl, k, inverse, &dz);
        double ddz = (double)da[2 * nl + k] - (double)da[2 * nl + k - 1];
        double t = (double)w * (-(ddz * (double)inverse) - (double)dz * dinv);
        // + 0.0f maps the -0.0 of a zero direction to +0.0, so TL(0) is exact zero bits
        float tw = __fadd_rn((float)t, 0.0f);
        bad |= !isfinite(w) || !isfinite(tw);
        d_weights[(long long)c * nl + k] = tw;
    }
    if (bad) atomicExch(failure, 1);
}

// __ad: the transpose.  One thread per column; it owns its column's field-2 abase words, its latitude word and its
// nl weight cotangent words, so there are no atomics and the result is deterministic.  Each z_k cotangent is the sum
// of the two layer terms that read it, formed in double in fixed order (layer k, then layer k+1) and added once; the
// latitude cotangent goes through d inverse / d lat.  The nl weight cotangent words of the column are consumed.
// For overlap < 4 there is no float output: nothing is read, written or consumed.
extern "C" __global__ void mpas_lw_mcica_seeds__ad(int nc, int nl, int overlap, int idcor, int julday, int permuteseed,
    const float* surface, const float* abase, float* b_surface, float* b_abase, float* b_weights, int* failure)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= nc) return;
    if (overlap < 4) return;
    if (c == 0 && (mcs_aliased(nc, nl, overlap, surface, abase, b_weights) ||
                   mcs_aliased(nc, nl, overlap, b_surface, b_abase, b_weights))) atomicExch(failure, 1);
    const float* a = abase + (long long)c * 12 * nl;
    float* ba = b_abase + (long long)c * 12 * nl;
    float* bw = b_weights + (long long)c * nl;
    float delta, e;
    float length = mcs_length(idcor, julday, surface[c * 6 + 5], &delta, &e);
    float inverse = RLW_DV(1.0f, length);
    bool bad = !isfinite(inverse);
    double binv = 0.0;
    // g_k = ybar_k * w_k; z_k gets -g_k*inverse (from layer k) and +g_{k+1}*inverse (from layer k+1).
    for (int k = 0; k < nl; ++k) {
        double acc = 0.0;
        if (k >= 1) {
            float dz, w = mcs_weight(a, nl, k, inverse, &dz);
            double y = (double)bw[k];
            bad |= !isfinite(y) || !isfinite(w);
            double g = y * (double)w;
            if (g != 0.0) { acc -= g * (double)inverse; binv -= g * (double)dz; }
        }
        if (k + 1 < nl) {
            float dz, w = mcs_weight(a, nl, k + 1, inverse, &dz);
            double g = (double)bw[k + 1] * (double)w;
            if (g != 0.0) acc += g * (double)inverse;
        }
        if (acc != 0.0) {
            float r = (float)((double)ba[2 * nl + k] + acc);
            bad |= !isfinite(r);
            ba[2 * nl + k] = r;
        }
    }
    if (idcor == 1 && binv != 0.0) {
        double bl = binv * mcs_dinv_dlat(idcor, length, delta, e);
        float r = (float)((double)b_surface[c * 6 + 5] + bl);
        bad |= !isfinite(r);
        b_surface[c * 6 + 5] = r;
    }
    for (int k = 0; k < nl; ++k) bw[k] = 0.0f;              // full overwrite (k = 0 constant included): consumed
    if (bad) atomicExch(failure, 1);
}
