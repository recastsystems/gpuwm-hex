// SPDX-License-Identifier: Apache-2.0
// Derivative entry (ad.entry/v1: aug / tl / ad) for mpas_sw_mcica_seeds, RRTMG SW McICA (MPAS-A v8.4.2
// module_ra_rrtmg_sw.F:1745-1748 KISS seeds, 1403-2012 exponential-overlap weights, irng = 0, permuteseed = 1),
// launched from hexcore/mpasphys/rrtmg_sw_v842.py:3364.  Forward: captured source 3c499c7b (lines 3683-3698),
// image c22c485e (hrrrv5_x1.2562 captures, steps 1, 3, 6, 12).
//
// Forward, per wrapper column c < nc (X(a,k) = a[c*nl+k]):
//   seeds:   sd[j] = (u32)(int)((pm - (float)(int)pm) * 1e9),  pm = play(c,j)*100 (Pa), j < 4;  then permuteseed KISS
//            draws;  seeds[c*4+j] = sd[j]                                                 (integer McICA state)
//   icld not 4 or 5: nothing else (weights is a one-word dummy, not written).  hrrrv5: icld = 2 (maximum-random).
//   icld 4 or 5: zo = 2500, or (idcor == 1) zo = 1000 (1.4315 + 2.1219 exp(-(lat - am3)^2 / 654.541...)), lat =
//            latitude[c] (degrees);  weights[c*nl+k] = exp(-(hgt(c,k) - hgt(c,k-1)) / zo) for k = 1..nl-1.
//            The k = 0 word is NOT written (unlike the LW twin, which writes 0): it keeps whatever it held.
// Active: hgt (all words of the nc*nl block) and latitude (idcor == 1) -> weights.  play only reaches the integer
// seeds: zero derivative, not bound.  seeds are integers: inactive, saved (see SAVED).
//
// SAVED (the McICA hold reason).  The seeds are a piecewise-constant function of the play words: they jump whenever
// (pm - int(pm))*1e9 crosses an integer, i.e. at about every ulp of a real pressure (measured: RESULT.md).  The true
// derivative is zero almost everywhere and the function is discontinuous at ulp scale, so the only usable
// linearisation is the frozen one: the trajectory's seeds, written by __aug bitwise to the forward, are the McICA
// samples of the linearisation; TL/AD never recompute them.  A nonlinear difference quotient through this kernel
// must hold the seeds (or the cloud masks downstream in mpas_sw_mcica_gpoint) at the trajectory's.
//
// Precision (the glue-row rule, as the live radiation entries): __aug repeats the forward word for word in f32
// (FMUL/FSUB/FADD/FDIV = __f*_rn and the forward's own glibc 2.39 expf transcription gfk_exp, built --fmad=false
// --ftz=false --prec-div=true --prec-sqrt=true), so its seeds and weights are bitwise the forward image's.  __tl/__ad
// form the local partials in double from the f32 primal words (dz, inverse, w, d, e) and round once to f32 per
// written word.
//
// Limiter-like elements, frozen treatment (the branch the forward took is the branch differentiated):
//   - the float->int truncations of the seeds (line 3687): piecewise constant -> zero derivative, seeds frozen;
//   - the KISS permutation (line 3688): integer arithmetic;
//   - icld == 4 || icld == 5 (line 3690): integer selector; otherwise no float output and no derivative words;
//   - idcor == 1, julday > 181 (lines 3692): integer selectors, inactive;
//   - k from 1 (line 3696): the k = 0 weight word is untouched by the forward, so its tangent and cotangent pass
//     through unchanged (identity), which TL and AD implement by not touching it.
// The kernel has no clamp or limiter on a float value.

// ---- verbatim from the captured forward source 3c499c7b (lines 184-192, 217-234, 290-300, 321-357, 3588-3594;
// names kept) -------------------------------------------------------------------------------------------------------
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

// MPAS module_ra_rrtmg_sw.F:2016-2048, signed final KISS integer.
__device__ float mpas_kiss(unsigned int& s1,unsigned int& s2,unsigned int& s3,unsigned int& s4) {
 s1=69069u*s1+1327217885u;s2^=s2<<13;s2^=s2>>17;s2^=s2<<5;
 s3=18000u*(s3&65535u)+(s3>>16);s4=30903u*(s4&65535u)+(s4>>16);
 int kiss=(int)(s1+s2+(s3<<16)+s4);
 return FADD(FMUL((float)kiss,2.328306e-10f),.5f);
}
// ---- end verbatim --------------------------------------------------------------------------------------------------

// Range-overlap test for the refusal of aliased views (alias_group null in the spec: every view distinct).
__device__ __forceinline__ bool mcs_overlap(const void* a, long long na, const void* b, long long nb)
{
    const char *pa = (const char*)a, *pb = (const char*)b;
    return pa < pb + nb && pb < pa + na;
}
__device__ __forceinline__ long long mcs_wbytes(int nc, int nl, int icld)
{
    return (icld == 4 || icld == 5) ? (long long)nc * nl * 4 : 4;
}
// hgt / latitude / weights (and play / seeds when given): pairwise disjoint.
__device__ __forceinline__ bool mcs_aliased(int nc, int nl, int icld, const void* hgt, const void* lat,
                                            const void* weights, const void* play = nullptr,
                                            const void* seeds = nullptr)
{
    const void* p[5] = {hgt, lat, weights, play, seeds};
    long long n[5] = {(long long)nc * nl * 4, (long long)nc * 4, mcs_wbytes(nc, nl, icld), (long long)nc * nl * 4,
                      (long long)nc * 16};
    for (int i = 0; i < 5; ++i)
        for (int j = i + 1; j < 5; ++j)
            if (p[i] && p[j] && mcs_overlap(p[i], n[i], p[j], n[j])) return true;
    return false;
}

// The forward's decorrelation length (capture 3c499c7b:3691-3693), word for word, plus the two f32 words the
// derivative needs (d = lat - am3 and the exp word e).
__device__ __forceinline__ float mcs_length(int idcor, int julday, float lat, float* d_o, float* e_o)
{
    float zo = 2500.f;
    *d_o = 0.0f; *e_o = 0.0f;
    if (idcor == 1) {
        float am3 = FMUL(julday > 181 ? -0.07671232521533966f : 0.07671232521533966f,
                         (float)(julday - (julday > 181 ? 272 : 91)));
        float d = FSUB(lat, am3);
        float e = gfk_exp(FDIV(-FMUL(d, d), 654.54107666015625f));
        zo = FMUL(FADD(1.4315f, FMUL(2.1219f, e)), 1.e3f);
        *d_o = d; *e_o = e;
    }
    return zo;
}

// d inverse / d lat on the f32 primal words, in double (0 when idcor != 1).
//   inverse = 1/zo, zo = 1000 (1.4315 + 2.1219 e), e = exp(q), q = -d^2 / C, C = 654.54107666015625 (the forward's
//   f32 word), d = lat - am3:  d inverse/d lat = -(1/zo^2) * 1000 * 2.1219 * e * (-2 d / C)
__device__ __forceinline__ double mcs_dinv_dlat(int idcor, float zo, float d, float e)
{
    if (idcor != 1) return 0.0;
    const double C = (double)654.54107666015625f;
    double L = (double)zo;
    return -(1.0 / (L * L)) * 1000.0 * 2.1219 * (double)e * (-2.0 * (double)d / C);
}

// One layer's weight, the forward's f32 words (k >= 1):  w_k = exp(-(z_k - z_{k-1}) * inverse).
// Partials (frozen at these words):  dw/dz_k = -w*inverse,  dw/dz_{k-1} = +w*inverse,  dw/dinverse = -w*dz.
__device__ __forceinline__ float mcs_weight(const float* z, int k, float inverse, float* dz_o)
{
    float dz = FSUB(z[k], z[k - 1]);
    *dz_o = dz;
    return gfk_exp(FMUL(-dz, inverse));
}

// ------------------------------------------------------------------------------------------------------------------
// __aug: the forward (capture 3c499c7b:3683-3698) word for word (seeds and weights bitwise to the forward image),
// plus the refusal flag: nl < 4 (the forward then converts an uninitialised word, line 3687), a non-finite or
// out-of-int-range seed-layer pressure word (the float->int conversion of such a word is implementation-defined), a
// non-finite inverse length or weight, or aliased views.
extern "C" __global__ void mpas_sw_mcica_seeds__aug(int nc, int nl, int icld, int idcor, int julday, int permuteseed,
    const float* play, const float* hgt, const float* latitude, unsigned int* seeds, float* weights, int* failure)
{
    int c = blockDim.x * blockIdx.x + threadIdx.x; if (c >= nc) return;
    if (c == 0 && mcs_aliased(nc, nl, icld, hgt, latitude, weights, play, seeds)) atomicExch(failure, 1);
    bool bad = nl < 4;
    unsigned int sd[4];
    for (int j = 0; j < 4; ++j) {
        float pm = 0.0f;
        if (j < nl) pm = FMUL(play[(long long)c * nl + j], 100.f);
        bad |= !isfinite(pm) || fabsf(pm) >= 2.1e9f;
        sd[j] = (unsigned int)(int)FMUL(FSUB(pm, (float)(int)pm), 1000000000.f);
    }
    for (int i = 0; i < permuteseed; ++i) mpas_kiss(sd[0], sd[1], sd[2], sd[3]);
    for (int j = 0; j < 4; ++j) seeds[c * 4 + j] = sd[j];
    if (icld == 4 || icld == 5) {
        float d, e;
        float zo = mcs_length(idcor, julday, latitude[c], &d, &e);
        float inverse = FDIV(1.f, zo);
        bad |= !isfinite(inverse);
        const float* z = hgt + (long long)c * nl;
        for (int k = 1; k < nl; ++k) {
            float dz, w = mcs_weight(z, k, inverse, &dz);
            bad |= !isfinite(w);
            weights[(long long)c * nl + k] = w;
        }
    }
    if (bad) atomicExch(failure, 1);
}

// __tl: seeds are integers (the McICA sample state), so they carry no tangent: they are the trajectory's, saved by
// __aug, and neither derivative entry reads or writes them.  For icld not 4 or 5 the forward writes no float word
// (weights is a one-word dummy), so TL writes nothing.  For icld 4 or 5, per column, the nl-1 weight words k >= 1 are
// written; the k = 0 word, which the forward does not write, keeps its tangent (identity).  Local partials in double
// from the f32 primal words, one rounding per written word.
extern "C" __global__ void mpas_sw_mcica_seeds__tl(int nc, int nl, int icld, int idcor, int julday, int permuteseed,
    const float* hgt, const float* latitude, const float* d_hgt, const float* d_latitude, float* d_weights,
    int* failure)
{
    int c = blockDim.x * blockIdx.x + threadIdx.x; if (c >= nc) return;
    if (!(icld == 4 || icld == 5)) return;
    if (c == 0 && (mcs_aliased(nc, nl, icld, hgt, latitude, d_weights) ||
                   mcs_aliased(nc, nl, icld, d_hgt, d_latitude, d_weights))) atomicExch(failure, 1);
    const float* z = hgt + (long long)c * nl;
    const float* dzt = d_hgt + (long long)c * nl;
    float d, e;
    float zo = mcs_length(idcor, julday, latitude[c], &d, &e);
    float inverse = FDIV(1.f, zo);
    double dinv = idcor == 1 ? mcs_dinv_dlat(idcor, zo, d, e) * (double)d_latitude[c] : 0.0;
    bool bad = !isfinite(inverse) || !isfinite(dinv);
    for (int k = 1; k < nl; ++k) {
        float dz, w = mcs_weight(z, k, inverse, &dz);
        double ddz = (double)dzt[k] - (double)dzt[k - 1];
        double t = (double)w * (-(ddz * (double)inverse) - (double)dz * dinv);
        // + 0.0f maps the -0.0 of a zero direction to +0.0, so TL(0) is exact zero bits
        float tw = __fadd_rn((float)t, 0.0f);
        bad |= !isfinite(w) || !isfinite(tw);
        d_weights[(long long)c * nl + k] = tw;
    }
    if (bad) atomicExch(failure, 1);
}

// __ad: the transpose.  One thread per column; it owns its column's nl height words, its latitude word and its nl
// weight cotangent words, so there are no atomics and the result is deterministic.  Each z_k cotangent is the sum of
// the two layer terms that read it, formed in double in fixed order (layer k, then layer k+1) and added once; the
// latitude cotangent goes through d inverse / d lat.  The nl-1 overwritten weight cotangent words (k >= 1) are
// consumed; the k = 0 word (not written by the forward: identity) is left as it is.  For icld not 4 or 5 there is no
// float output: nothing is read, written or consumed.
extern "C" __global__ void mpas_sw_mcica_seeds__ad(int nc, int nl, int icld, int idcor, int julday, int permuteseed,
    const float* hgt, const float* latitude, float* b_hgt, float* b_latitude, float* b_weights, int* failure)
{
    int c = blockDim.x * blockIdx.x + threadIdx.x; if (c >= nc) return;
    if (!(icld == 4 || icld == 5)) return;
    if (c == 0 && (mcs_aliased(nc, nl, icld, hgt, latitude, b_weights) ||
                   mcs_aliased(nc, nl, icld, b_hgt, b_latitude, b_weights))) atomicExch(failure, 1);
    const float* z = hgt + (long long)c * nl;
    float* bz = b_hgt + (long long)c * nl;
    float* bw = b_weights + (long long)c * nl;
    float d, e;
    float zo = mcs_length(idcor, julday, latitude[c], &d, &e);
    float inverse = FDIV(1.f, zo);
    bool bad = !isfinite(inverse);
    double binv = 0.0;
    // g_k = ybar_k * w_k; z_k gets -g_k*inverse (from layer k) and +g_{k+1}*inverse (from layer k+1).
    for (int k = 0; k < nl; ++k) {
        double acc = 0.0;
        if (k >= 1) {
            float dz, w = mcs_weight(z, k, inverse, &dz);
            double y = (double)bw[k];
            bad |= !isfinite(y) || !isfinite(w);
            double g = y * (double)w;
            if (g != 0.0) { acc -= g * (double)inverse; binv -= g * (double)dz; }
        }
        if (k + 1 < nl) {
            float dz, w = mcs_weight(z, k + 1, inverse, &dz);
            double g = (double)bw[k + 1] * (double)w;
            if (g != 0.0) acc += g * (double)inverse;
        }
        if (acc != 0.0) {
            float r = (float)((double)bz[k] + acc);
            bad |= !isfinite(r);
            bz[k] = r;
        }
    }
    if (idcor == 1 && binv != 0.0) {
        double bl = binv * mcs_dinv_dlat(idcor, zo, d, e);
        float r = (float)((double)b_latitude[c] + bl);
        bad |= !isfinite(r);
        b_latitude[c] = r;
    }
    for (int k = 1; k < nl; ++k) bw[k] = 0.0f;              // overwritten words consumed; k = 0 kept (identity)
    if (bad) atomicExch(failure, 1);
}
