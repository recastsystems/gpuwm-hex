typedef float real;
// ======================================================================
// THIRD-PARTY NOTICE.  Parts of this file are hand transcriptions of
// third-party work.  ArWen distributes the file under the Apache License
// 2.0; the notices below belong to the transcribed parts and are kept
// here because their own licences require it.  Full texts are in the
// repository NOTICE and in the licenses/ directory.
//
// For the two libm grants the text also sits beside the code, in
// gpuwm/core/kernels/LICENSE-third-party.txt.
//
//   Arm optimized-routines -- the logf, expf, exp2f and powf cores and
//   their data tables github.com/ARM-software/optimized-routines:
//   math/logf.c, math/expf.c, math/exp2f.c, math/powf.c and the matching
//   math/*_data.c, published August 2017 and imported into glibc for
//   2.27/2.28 by their own author; transcribed here from glibc 2.39
//   sysdeps/ieee754/flt-32/.
//
//       Copyright (c) 2017-2018, Arm Limited.
//       SPDX-License-Identifier: MIT
//
//   Taken under the MIT branch of Arm's grant.  MIT requires the
//   copyright notice above and its permission notice to travel with every
//   copy; the permission notice is reproduced in full in the files named
//   above.
//
//   NO FDLIBM CODE REMAINS IN THIS FILE.  It carried transcriptions of
//   FDLIBM's expm1f (s_expm1f.c) and lgammaf reduction (e_lgammaf_r.c)
//   through ArWen 2.6.5; both were deleted at 2.6.6 with the gamma block
//   that was their only caller, so the Sun notice that had to travel with
//   them travels no longer -- there is nothing here for it to attach to.
//   The notice is unchanged and still reproduced, for the twelve files
//   that DO carry FDLIBM code -- nine .cu in this directory and three
//   Python modules under gpuwm/core/ -- in the repository NOTICE, in
//   licenses/LICENSE-FDLIBM-SunPro.txt and in
//   gpuwm/core/kernels/LICENSE-third-party.txt.
//
//   Nothing else in this file is a transcription of anything.  gfk_d2f_rn
//   is ArWen's own subnormal-rounding countermeasure and gfk_tgamma is
//   ArWen's own correctly rounded gamma (see its block below and
//   docs/gf_gamma_known_delta.md); both are Apache-2.0 original work.
// ======================================================================
// glibc 2.39 float32 transcendentals, shared device code.
//
// PROVENANCE.  Lifted VERBATIM from gpuwm/core/kernels/gf.cu on 2026-08-28,
// where it was lines 235-243 (the FP contraction pins) and 538-1141 (the
// transcriptions).  Not a re-derivation and not a tidy-up: the text is the
// text that gf's 396 parity tests already grade at max_ulp 0 against the
// live glibc 2.39 sweep fixtures gpuwm/data/gf/oracle/gf-libm-*.csv.
//
// WHY IT EXISTS.  gfk_log/gfk_exp/gfk_pow are glibc's own words, not CUDA's.
// CUDA's expf/powf/tgammaf are DIFFERENT functions -- gf.cu's header records
// the measured divergence and keeps a negative control proving it is real --
// so any kernel graded bitwise against a gfortran/glibc oracle must call
// these and not the builtins.  New Tiedtke (cu_physics=16) needs exactly
// gfk_log, gfk_exp and gfk_pow: its whole libm surface is 9 exp, 1 log,
// 10 sqrt and four pow forms (**t13, **0.5777, **0.2, **0.5).  sqrtf is
// correctly rounded on both sides and needs nothing.
//
// This would have been the THIRD transcription in the tree -- gf.cu's
// gfk_*, and noahmp_leaves.cu's r_log/r_exp/r_pow, which gf.cu:40 records
// it was renamed from.  Two copies is a duplicate; three is a drift hazard.
//
// The contraction pins travel WITH the block because the block uses all
// nine of them 294 times and they are what makes its rounding reproducible
// (project numerics rule: __fmaf_rn/__fmul_rn/__fadd_rn are NVIDIA-guaranteed never
// merged, so pinning to the fused form is free).  gf.cu KEEPS its own
// copies at 235-238; identical redefinition of an object-like or
// function-like macro is legal C++ and nothing else in gf.cu moves.
//
// MEASURED, before and after the lift (compile-only, no launch, sm_120 /
// NVRTC 13.0, tools-side probe): the assembled gf source changes by 346
// chars and the cubin by 796 bytes of 4,639,695 -- entirely the ELF
// section-name table (.nv.constant3 <-> .nv.constant4, .nv.global.init
// reordered).  All seven gf entry points keep byte-identical
// local_size_bytes, num_regs and const_size_bytes.

#define FADD(a, b) __fadd_rn((a), (b))
#define FSUB(a, b) __fsub_rn((a), (b))
#define FMUL(a, b) __fmul_rn((a), (b))
#define FDIV(a, b) __fdiv_rn((a), (b))
#define FSQRT(a)   __fsqrt_rn(a)
#define DADD(a, b) __dadd_rn((a), (b))
#define DSUB(a, b) __dsub_rn((a), (b))
#define DMUL(a, b) __dmul_rn((a), (b))
#define DDIV(a, b) __ddiv_rn((a), (b))

// ==========================================================================
// glibc 2.39 float32 transcendentals (rule 3)
// ==========================================================================
// e_logf_data.c
__device__ const double GFK_LOGF_INVC[16] = {
    0x1.661ec79f8f3bep+0, 0x1.571ed4aaf883dp+0, 0x1.49539f0f010bp+0,
    0x1.3c995b0b80385p+0, 0x1.30d190c8864a5p+0, 0x1.25e227b0b8eap+0,
    0x1.1bb4a4a1a343fp+0, 0x1.12358f08ae5bap+0, 0x1.0953f419900a7p+0,
    0x1p+0,               0x1.e608cfd9a47acp-1, 0x1.ca4b31f026aap-1,
    0x1.b2036576afce6p-1, 0x1.9c2d163a1aa2dp-1, 0x1.886e6037841edp-1,
    0x1.767dcf5534862p-1 };
__device__ const double GFK_LOGF_LOGC[16] = {
    -0x1.57bf7808caadep-2, -0x1.2bef0a7c06ddbp-2, -0x1.01eae7f513a67p-2,
    -0x1.b31d8a68224e9p-3, -0x1.6574f0ac07758p-3, -0x1.1aa2bc79c81p-3,
    -0x1.a4e76ce8c0e5ep-4, -0x1.1973c5a611cccp-4, -0x1.252f438e10c1ep-5,
     0x0p+0,                0x1.aa5aa5df25984p-5,  0x1.c5e53aa362eb4p-4,
     0x1.526e57720db08p-3,  0x1.bc2860d22477p-3,   0x1.1058bc8a07ee1p-2,
     0x1.4043057b6ee09p-2 };
#define GFK_LOGF_LN2 0x1.62e42fefa39efp-1
#define GFK_LOGF_A0 (-0x1.00ea348b88334p-2)
#define GFK_LOGF_A1 (0x1.5575b0be00b6ap-2)
#define GFK_LOGF_A2 (-0x1.ffffef20a4123p-2)

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

// e_powf_log2_data.c.  POWF_SCALE is 1.0 (TOINT_INTRINSICS = 0 on x86-64).
__device__ const double GFK_POWF_INVC[16] = {
    0x1.661ec79f8f3bep+0, 0x1.571ed4aaf883dp+0, 0x1.49539f0f010bp+0,
    0x1.3c995b0b80385p+0, 0x1.30d190c8864a5p+0, 0x1.25e227b0b8eap+0,
    0x1.1bb4a4a1a343fp+0, 0x1.12358f08ae5bap+0, 0x1.0953f419900a7p+0,
    0x1p+0,               0x1.e608cfd9a47acp-1, 0x1.ca4b31f026aap-1,
    0x1.b2036576afce6p-1, 0x1.9c2d163a1aa2dp-1, 0x1.886e6037841edp-1,
    0x1.767dcf5534862p-1 };
__device__ const double GFK_POWF_LOGC[16] = {
    -0x1.efec65b963019p-2, -0x1.b0b6832d4fca4p-2, -0x1.7418b0a1fb77bp-2,
    -0x1.39de91a6dcf7bp-2, -0x1.01d9bf3f2b631p-2, -0x1.97c1d1b3b7afp-3,
    -0x1.2f9e393af3c9fp-3, -0x1.960cbbf788d5cp-4, -0x1.a6f9db6475fcep-5,
     0x0p+0,                0x1.338ca9f24f53dp-4,  0x1.476a9543891bap-3,
     0x1.e840b4ac4e4d2p-3,  0x1.40645f0c6651cp-2,  0x1.88e9c2c1b9ff8p-2,
     0x1.ce0a44eb17bccp-2 };
__device__ const double GFK_POWF_A[5] = {
     0x1.27616c9496e0bp-2, -0x1.71969a075c67ap-2,  0x1.ec70a6ca7baddp-2,
    -0x1.7154748bef6c8p-1,  0x1.71547652ab82bp+0 };

// glibc 2.39 sysdeps/ieee754/flt-32/e_logf.c
__device__ float gfk_log(float x)
{
    unsigned int ix = __float_as_uint(x);
    if (ix == 0x3f800000u) return 0.0f;
    if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u) {
        if (ix * 2u == 0u) return __int_as_float(0xff800000);
        if (ix == 0x7f800000u) return x;
        if ((ix & 0x80000000u) || ix * 2u >= 0xff000000u)
            return __int_as_float(0x7fc00000);
        ix = __float_as_uint(FMUL(x, 8388608.0f));   /* 0x1p23f */
        ix -= 23u << 23;
    }
    unsigned int tmp = ix - 0x3f330000u;
    int i = (int)((tmp >> 19) & 15u);
    int k = (int)tmp >> 23;
    unsigned int iz = ix - (tmp & 0xff800000u);
    double z = (double)__uint_as_float(iz);
    double r = DSUB(DMUL(z, GFK_LOGF_INVC[i]), 1.0);
    double y0 = DADD(GFK_LOGF_LOGC[i], DMUL((double)k, GFK_LOGF_LN2));
    double r2 = DMUL(r, r);
    double y = DADD(DMUL(GFK_LOGF_A1, r), GFK_LOGF_A2);
    y = DADD(DMUL(GFK_LOGF_A0, r2), y);
    y = DADD(DMUL(y, r2), DADD(y0, r));
    return __double2float_rn(y);
}

// Round a double to binary32, INCLUDING into the subnormal range.  On this
// toolchain `__double2float_rn` flushes a subnormal result to zero (CuPy
// appends -ftz=true and the compiler emits the flush after the conversion),
// while glibc's expf/powf do produce subnormals.  The correctly rounded
// subnormal is recovered exactly: m * 2^-149 scaling is exact in binary64
// over this band and rint rounds ties to even.  Same function, same
// reasoning, as noahmp_leaves.cu::nmp_d2f_rn -- the sm_120 FP32-DAZ
// countermeasure this repo has already proven.
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

// The 32-entry exp2 core shared by glibc's expf, exp2f and powf.
__device__ double gfk_exp2_core(double xd, double shift,
                                double p0, double p1, double p2,
                                unsigned long long sign_bias)
{
    double kd = DADD(xd, shift);
    unsigned long long ki = (unsigned long long)__double_as_longlong(kd);
    kd = DSUB(kd, shift);
    double r = DSUB(xd, kd);
    unsigned long long t = GFK_EXP2F_TAB[ki & 31ULL];
    t += (ki + sign_bias) << (52 - 5);
    double s = __longlong_as_double((long long)t);
    double z = DADD(DMUL(p0, r), p1);
    double r2 = DMUL(r, r);
    double y = DADD(DMUL(p2, r), 1.0);
    y = DADD(DMUL(z, r2), y);
    return DMUL(y, s);
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

// e_exp2f.c (exp2f), s_expm1f.c (expm1f) and e_lgammaf_r.c (lgammaf) were
// transcribed here through 2.6.5 and are DELETED at 2.6.6.  Their only
// caller anywhere in this tree was the earlier gamma this file used to end
// with, which used all three on its way to Gamma; with nothing else calling
// them they had no reason to stay.  That earlier gamma is
// gone, and ArWen's own gamma that replaced it evaluates no logarithm, no
// exponential and no exp2 at all, so all three functions became dead to the
// physics and survived only as slots of the gf_libm_unary_probe TEST
// kernel.  Dead
// transcriptions are not decoration to keep: gfk_lgamma_pos was the ONLY
// transcription of glibc's e_lgammaf_r.c anywhere in the repository, so
// deleting it retires that file from the distribution outright, and exp2f
// and expm1f leave THIS translation unit (mynn_pbl.cu and
// mynn_dmp_sibling.cu still carry their own expm1f for tanhf).  The
// 32-entry exp2 CORE above stays -- gfk_exp and gfk_pow are live physics
// and both call it.
//
// The live float32 surface of this header is now exactly gfk_log, gfk_exp,
// gfk_pow, gfk_d2f_rn and gfk_tgamma.

// glibc 2.39 sysdeps/ieee754/flt-32/e_powf.c log2_inline
__device__ double gfk_powf_log2(unsigned int ix)
{
    unsigned int tmp = ix - 0x3f330000u;
    int i = (int)((tmp >> 19) & 15u);
    unsigned int top = tmp & 0xff800000u;
    unsigned int iz = ix - top;
    int k = (int)top >> 23;
    double z = (double)__uint_as_float(iz);
    double r = DSUB(DMUL(z, GFK_POWF_INVC[i]), 1.0);
    double y0 = DADD(GFK_POWF_LOGC[i], (double)k);
    double r2 = DMUL(r, r);
    double y = DADD(DMUL(GFK_POWF_A[0], r), GFK_POWF_A[1]);
    double p = DADD(DMUL(GFK_POWF_A[2], r), GFK_POWF_A[3]);
    double r4 = DMUL(r2, r2);
    double q = DADD(DMUL(GFK_POWF_A[4], r), y0);
    q = DADD(DMUL(p, r2), q);
    return DADD(DMUL(y, r4), q);
}

__device__ int gfk_checkint(unsigned int iy)
{
    int e = (int)(iy >> 23 & 0xffu);
    if (e < 0x7f) return 0;
    if (e > 0x7f + 23) return 2;
    if (iy & ((1u << (0x7f + 23 - e)) - 1u)) return 0;
    if (iy & (1u << (0x7f + 23 - e))) return 1;
    return 2;
}

__device__ bool gfk_zeroinfnan(unsigned int ix)
{
    return 2u * ix - 1u >= 2u * 0x7f800000u - 1u;
}

// glibc 2.39 sysdeps/ieee754/flt-32/e_powf.c, full special-case surface:
// the beta-shape powers reach kratio == 0 and kratio == 1 on every column
// (powf(0, +y) and powf(+0-adjacent bases), so the zero/int paths are live.
__device__ float gfk_pow(float x, float y)
{
    unsigned int sign_bias = 0u;
    unsigned int ix = __float_as_uint(x);
    unsigned int iy = __float_as_uint(y);
    if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u || gfk_zeroinfnan(iy)) {
        if (gfk_zeroinfnan(iy)) {
            if (2u * iy == 0u) return 1.0f;
            if (ix == 0x3f800000u) return 1.0f;
            if (2u * ix > 2u * 0x7f800000u || 2u * iy > 2u * 0x7f800000u)
                return FADD(x, y);
            if (2u * ix == 2u * 0x3f800000u) return 1.0f;
            if ((2u * ix < 2u * 0x3f800000u) == !(iy & 0x80000000u))
                return 0.0f;
            return FMUL(y, y);
        }
        if (gfk_zeroinfnan(ix)) {
            float x2 = FMUL(x, x);
            if ((ix & 0x80000000u) && gfk_checkint(iy) == 1) x2 = -x2;
            return (iy & 0x80000000u) ? FDIV(1.0f, x2) : x2;
        }
        if (ix & 0x80000000u) {
            int yint = gfk_checkint(iy);
            if (yint == 0) return __int_as_float(0x7fc00000);
            if (yint == 1) sign_bias = 1u << (5 + 11);
            ix &= 0x7fffffffu;
        }
        if (ix < 0x00800000u) {
            ix = __float_as_uint(FMUL(x, 8388608.0f)) & 0x7fffffffu;
            ix -= 23u << 23;
        }
    }
    double logx = gfk_powf_log2(ix);
    double ylogx = DMUL((double)y, logx);
    unsigned int hi = (unsigned int)
        (((unsigned long long)__double_as_longlong(ylogx) >> 47) & 0xffffULL);
    if (hi >= (unsigned int)
            (((unsigned long long)__double_as_longlong(126.0) >> 47) & 0xffffULL)) {
        if (ylogx > 0x1.fffffffd1d571p+6)
            return sign_bias ? __int_as_float(0xff800000)
                             : __int_as_float(0x7f800000);
        if (ylogx <= -150.0) return sign_bias ? -0.0f : 0.0f;
    }
    return gfk_d2f_rn(
        gfk_exp2_core(ylogx, GFK_EXP2F_SHIFT_SCALED,
                      GFK_EXP2F_P0, GFK_EXP2F_P1, GFK_EXP2F_P2,
                      (unsigned long long)sign_bias));
}

// ==========================================================================
// gfk_tgamma -- ArWen's own float32 gamma.  ORIGINAL WORK, Apache-2.0.
// ==========================================================================
//
// WHAT THIS REPLACED.  Until 2.6.5 this file ended with ArWen's earlier
// gamma, gfk_gamma_product / gfk_gammaf_positive / gfk_tgamma, which
// returned the same float32 words as glibc 2.39's tgammaf and so carried
// that function's rounding errors.  Like the code below, it was this
// project's own work under the project's licence.  2.6.6 replaced it with
// the correctly rounded gamma below; the measurement is further down and
// in docs/gf_gamma_known_delta.md.  Nothing from the earlier gamma
// survives below: no constant, no branch structure, no algorithm.  This
// code evaluates no lgamma, performs no exponentiation, and calls
// nothing.
//
// HOW IT WAS DERIVED.  Gamma on [1,2) is a degree-9 polynomial on each of 16
// equal segments; every other positive argument reduces to [1,2) by the
// functional equation Gamma(z+1) = z*Gamma(z); negative arguments use the
// reflection formula Gamma(x)*Gamma(1-x) = pi/sin(pi x).  The 160
// coefficients in GFK_TG_C were generated from Stirling's asymptotic series
// (DLMF 5.11.1) evaluated in 113-bit arithmetic with the standard Bernoulli
// numbers B2..B24.  GFK_TG_SP holds the Taylor coefficients of sin(pi r),
// c_k = (-1)^k pi^(2k+1)/(2k+1)!, in closed form.  Classical mathematics; no
// implementation of any kind was consulted, and no third-party source was
// read, quoted or fitted against.  The audit trail is lineage-docs/gam-03*.
//
// IT IS CORRECTLY ROUNDED, AND THE REFERENCE IS NOT.  MEASURED against
// libquadmath's 113-bit tgammaq -- an oracle unrelated to glibc -- over all
// 59,768,833 float32 arguments of [0.25, 36], the interval that covers every
// value this scheme can reach:
//
//     this code       0 arguments not correctly rounded
//     glibc 2.39      23,575,230 arguments not correctly rounded (39.44 %),
//                     worst 6 ULP.  tgammaf(4.0f) returns 6.00000048, not 6.
//
// A table fitted to or copied from glibc would carry glibc's own error, ten
// orders of magnitude larger than this code's ~5.6e-17 relative agreement
// with true Gamma.  That is the structural evidence the coefficients are not
// derived from the reference.
//
// >>> DELIBERATE DIVERGENCE FROM WRF.  READ docs/gf_gamma_known_delta.md. <<<
//
// gfortran binds WRF's F2008 gamma() intrinsic to glibc's tgammaf, so this
// kernel no longer reproduces WRF's fzu bit for bit and the deep mass flux
// xmb moves by up to 7.3 per cent on converged columns (median 1.9).  That is
// the amplification recorded at gf.cu:50-56, not an accuracy loss.  MEASURED
// over every reachable (alpha, beta) -- all 53,687,093 float32 tunning values
// of the three drafts -- fzu changes on 68.17 per cent of them: 98.39 per cent
// of the set within 4 ULP, worst 12 (draft 1, beta=2.5), worst relative
// 8.39e-7.  The committed 216-column fixture spans only the first 4 ULP, which
// is why the gate below reads 4 and this bound reads 12.  The scheme's own
// xk = (xaa0-aa1)/mbdt cancellation is what turns any of it into per cent.
//
// AND OURS IS THE CLOSER ONE.  Graded against the exact 113-bit
// Gamma(a+b)/(Gamma(a)Gamma(b)) rounded once to float32, over the same
// 53,687,093: this code is worst 4 ULP from the true fzu and glibc is worst
// 11; ours is strictly closer on 49.95 per cent of the set and strictly
// further on 9.59; mean relative error 3.83e-8 against glibc's 9.67e-8.
// Neither is exact -- rounding three gammas and a multiply and a divide to
// float32 costs up to 4 ULP on its own -- so xmb is not determined to better
// than tens of per cent by ANY float32 build of this scheme, WRF's included.
//
// The divergence is NOT new to ArWen and it is NOT unbounded.  MEASURED, this
// code returns the same word as gpuwm/verify/gf_deep_ref.py::_tgammaf -- the
// float32 CPU authority's own gamma model -- on ALL 59,768,833 arguments of
// [0.25, 36], so the CPU and CUDA paths now agree bitwise where before they
// did not, and the divergence from WRF is exactly the one the CPU suite has
// carried, gated and documented since the port landed
// (tests/test_gf_deep_parity.py::test_fzu_is_the_one_measured_divergence,
// budget 4 ULP; this code lands at 4).
//
// TO GET WRF'S ANSWER BACK, pin fzu.  gfd_get_zu_zd_pdf takes fzu_override
// and gf_deep_stage exposes it in scin; passing the oracle's captured word
// makes the whole chain bitwise against WRF again, which is exactly how the
// CPU suite reaches max_ulp 0 today.  No build flag is needed and none is
// provided -- restoring glibc's bits at run time would mean shipping 22.5 MB
// of measured glibc deviation, which is the thing this change removes.
// ==========================================================================

// Gamma on [1,2): 16 equal segments, degree-9 polynomial in u = r - centre.
__device__ const double GFK_TG_C[16][10] = {
  { 0x1.f73ed01940522p-1, -0x1.092fd20dd784cp-1, 0x1.d1a2ea66d2aefp-1, -0x1.966a1d7f2b9d9p-1, 0x1.af057a1fa6b33p-1, -0x1.a114e80ce35b1p-1, 0x1.99b28f9ddcee8p-1, -0x1.8e73dca5f882p-1, 0x1.843285f62ee7dp-1, -0x1.78e83c533d8b1p-1 },   /* centre 1.031250000 */
  { 0x1.e865a5b755fb9p-1, -0x1.a6b50f60b5c6ep-2, 0x1.8e9a9675e8147p-1, -0x1.39251063e4e66p-1, 0x1.41a834d107f5ep-1, -0x1.237aaf45da8d8p-1, 0x1.0f0e4f41772eap-1, -0x1.f11209307f1cp-2, 0x1.c8cbfdb620ef9p-2, -0x1.a24012d73ac3cp-2 },   /* centre 1.093750000 */
  { 0x1.dcac35f2a7419p-1, -0x1.49cf184c91f8ep-2, 0x1.5ac61acea578ep-1, -0x1.e5d43093961e1p-2, 0x1.e91903e412f2ap-2, -0x1.9eedaece7d029p-2, 0x1.6f07e745bdddep-2, -0x1.3e2e7165ba686p-2, 0x1.14c7245554af4p-2, -0x1.df8a55df66eep-3 },   /* centre 1.156250000 */
  { 0x1.d3aa3cecb6cdp-1, -0x1.f0b8c2384c507p-3, 0x1.327dd5130ef72p-1, -0x1.7a2070b357f21p-2, 0x1.7a7cfacdf879dp-2, -0x1.2c07bb9dae965p-2, 0x1.fb753b1f31c65p-3, -0x1.a0d97e4269587p-3, 0x1.58589453b97bbp-3, -0x1.1b0d3d42d932bp-3 },   /* centre 1.218750000 */
  { 0x1.cd0ebb0c4e488p-1, -0x1.5fa4609a59d2cp-3, 0x1.13236e09cf181p-1, -0x1.2616a66cb748ap-2, 0x1.29fb9224d7007p-2, -0x1.b7c019bda7132p-3, 0x1.658f76dab2a86p-3, -0x1.16bef3649d39bp-3, 0x1.b6aee3baf8284p-4, -0x1.570897c5886ebp-4 },   /* centre 1.281250000 */
  { 0x1.c89aaab6c10fdp-1, -0x1.b8d4972a0d9cp-4, 0x1.f59ee44fdc796p-2, -0x1.c6c163713c6cbp-3, 0x1.dd53c7e7fbca8p-3, -0x1.45d61e648b785p-3, 0x1.00706690f6116p-3, -0x1.7bbc7df8eac4cp-4, 0x1.1d809aa6ec9fp-4, -0x1.a9b3bfeb56d76p-5 },   /* centre 1.343750000 */
  { 0x1.c61d286fe74edp-1, -0x1.8f960eacc3e79p-5, 0x1.d035f977bf7ffp-2, -0x1.5afe653a01c68p-3, 0x1.850c11888a092p-3, -0x1.e6f81933b909dp-4, 0x1.7607930824288p-4, -0x1.06f53c3fa90a8p-4, 0x1.7afbb95743636p-5, -0x1.0de3bdb94b909p-5 },   /* centre 1.406250000 */
  { 0x1.c5709f063f61ep-1, 0x1.8e787a2ac1f9ep-8, 0x1.b3f656a41419p-2, -0x1.026537ebbdf28p-3, 0x1.42e2dcb386c76p-3, -0x1.6dedd3608424ap-4, 0x1.1533b414d61ep-4, -0x1.718dbd05eec3dp-5, 0x1.001e8923ca168p-5, -0x1.5cf9f00355d61p-6 },   /* centre 1.468750000 */
  { 0x1.c678adaa16db2p-1, 0x1.dade0522ce28dp-5, 0x1.9f502b04aa917p-2, -0x1.7061698453edp-4, 0x1.111adaec147d2p-3, -0x1.138a886268f36p-4, 0x1.a17ae1c392bd6p-5, -0x1.0707922c8c43ep-5, 0x1.60077de104a49p-6, -0x1.cb5c7571211ddp-7 },   /* centre 1.531250000 */
  { 0x1.c920953aa5d66p-1, 0x1.b9496a6874861p-4, 0x1.9116c2e5de8d1p-2, -0x1.e346a34c78d6ep-5, 0x1.d74c383999e27p-4, -0x1.9de756ede0cd1p-5, 0x1.3f8131690fd1ap-5, -0x1.7a888b14ec653p-6, 0x1.eb8b5d6cd0026p-7, -0x1.334a276679165p-7 },   /* centre 1.593750000 */
  { 0x1.cd5a098928442p-1, 0x1.3fb8f1d0abf2cp-3, 0x1.8867f0f1d270ap-2, -0x1.0654798b0415fp-5, 0x1.9f3d067f9d585p-4, -0x1.3412ca3fa8323p-5, 0x1.f164f1fbed563p-6, -0x1.12caa9ef9524ep-6, 0x1.5c618b73bc8a7p-7, -0x1.a14084fcb5d75p-8 },   /* centre 1.656250000 */
  { 0x1.d31c4db6ff586p-1, 0x1.a140aba605e6ap-3, 0x1.849a7d111fe3ep-2, -0x1.0665ee5c25becp-7, 0x1.75d82f748db66p-4, -0x1.c1f8dbdb02d95p-6, 0x1.8a422b1844434p-6, -0x1.917ecef6748cdp-7, 0x1.f5201988eff97p-8, -0x1.1f187b1036facp-8 },   /* centre 1.718750000 */
  { 0x1.da6389f09f623p-1, 0x1.0131e5b57cf1ap-2, 0x1.853176b1f3b73p-2, 0x1.c5c9000d6e1cep-7, 0x1.581177702069bp-4, -0x1.3d027a323721fp-6, 0x1.3ec6f9466489fp-6, -0x1.263da5167e69p-7, 0x1.6dbe71bbe090cp-8, -0x1.8fd143df66799p-9 },   /* centre 1.781250000 */
  { 0x1.e3304db941633p-1, 0x1.3217c4a579d7cp-2, 0x1.89d2fbae9b259p-2, 0x1.1807b533d9092p-5, 0x1.43ae1d6087604p-4, -0x1.a0c1119bc32ap-7, 0x1.077745c6a4217p-6, -0x1.aecbc52480399p-8, 0x1.0f00dd6253d47p-8, -0x1.194b848e9be9ep-9 },   /* centre 1.843750000 */
  { 0x1.ed87357995087p-1, 0x1.63cf26c2a3f66p-2, 0x1.924179f06fc8cp-2, 0x1.b66aa0070a6b4p-5, 0x1.370c0b9271813p-4, -0x1.d632bc804cea1p-8, 0x1.be2e3854ef50bp-7, -0x1.391015cd2774dp-8, 0x1.9823259db4371p-9, -0x1.8f35002269e0fp-10 },   /* centre 1.906250000 */
  { 0x1.f970ac84d0a49p-1, 0x1.96cf0f1b4e4dp-2, 0x1.9e56bf5311f1ap-2, 0x1.2815a841a3351p-4, 0x1.30fa25285e4fep-4, -0x1.3d45167c1dcbap-9, 0x1.83e8c0f79bf95p-7, -0x1.bf53a20fad88p-9, 0x1.38d7f76fe48b2p-9, -0x1.1d0e83e2ca312p-10 },   /* centre 1.968750000 */
};

// sin(pi*r) for |r| <= 1/2:  c_k = (-1)^k pi^(2k+1) / (2k+1)!  (Taylor).
// Truncation bound at |r| = 1/2 is 1.81e-23.  Reached only for x < 0.
__device__ const double GFK_TG_SP[13] = {
  0x1.921fb54442d18p+1,   /* 3.1415926535897932385 */
  -0x1.4abbce625be53p+2,   /* -5.1677127800499700292 */
  0x1.466bc6775aae2p+1,   /* 2.5501640398773454439 */
  -0x1.32d2cce62bd86p-1,   /* -0.59926452932079207689 */
  0x1.50783487ee782p-4,   /* 0.082145886611128228799 */
  -0x1.e3074fde8871fp-8,   /* -0.0073704309457143507773 */
  0x1.e8f434d018d63p-12,   /* 0.00046630280576761256442 */
  -0x1.6fadb9f155744p-16,   /* -2.1915353447830215827e-05 */
  0x1.aaec32af93359p-21,   /* 7.9520540014755127848e-07 */
  -0x1.8a404211f9547p-26,   /* -2.294842899726987311e-08 */
  0x1.2877020d52cfp-31,   /* 5.3926646626081284894e-10 */
  -0x1.7215f879e1ac9p-37,   /* -1.0518471716932064455e-11 */
  0x1.859c594ba4573p-43    /* 1.7302192458361107612e-13 */
};

#define GFK_TG_PI   0x1.921fb54442d18p+1
#define GFK_TG_QNAN __uint_as_float(0x7fc00000u)

// Gamma(r) for r in [1,2).  All arithmetic pinned to binary64 add/mul; the
// two subtractions below are exact (Sterbenz), so j and u are exact.
__device__ double gfk_tgamma_poly(double r)
{
    double t = DMUL(DSUB(r, 1.0), 16.0);                 /* [0,16), exact */
    int j = (int)t;
    if (j > 15) j = 15;                                  /* r = nextbelow(2) */
    double u = DSUB(r, DMUL((double)(33 + 2 * j), 0.03125));   /* exact */
    const double *c = GFK_TG_C[j];
    double p = c[9];
    p = DADD(DMUL(p, u), c[8]);
    p = DADD(DMUL(p, u), c[7]);
    p = DADD(DMUL(p, u), c[6]);
    p = DADD(DMUL(p, u), c[5]);
    p = DADD(DMUL(p, u), c[4]);
    p = DADD(DMUL(p, u), c[3]);
    p = DADD(DMUL(p, u), c[2]);
    p = DADD(DMUL(p, u), c[1]);
    p = DADD(DMUL(p, u), c[0]);
    return p;
}

// Gamma(v) in binary64 for 0 < v < 43 (the caller's x < 36 guard bounds the
// reflected argument 1-x below 43, so the recurrence runs at most 41 times).
__device__ double gfk_tgamma_pos(double v)
{
    if (v >= 1.0) {
        int n = (int)v;                                  /* 1 .. 42 */
        double r = DSUB(v, (double)(n - 1));             /* [1,2), exact */
        double p = gfk_tgamma_poly(r);
        for (int i = 1; i < n; i++)
            p = DMUL(p, DSUB(v, (double)i));
        return p;
    }
    return DDIV(gfk_tgamma_poly(DADD(v, 1.0)), v);       /* Gamma(v)=Gamma(v+1)/v */
}

// sin(pi*x) for |x| < 2^23.  k and the subtraction are exact.
__device__ double gfk_tgamma_sinpi(double x)
{
    long long k = (long long)(x < 0.0 ? DSUB(x, 0.5) : DADD(x, 0.5));
    double r  = DSUB(x, (double)k);                      /* [-1/2,1/2], exact */
    double r2 = DMUL(r, r);
    double s  = GFK_TG_SP[12];
    for (int i = 11; i >= 0; i--) s = DADD(DMUL(s, r2), GFK_TG_SP[i]);
    s = DMUL(s, r);
    return (k & 1) ? -s : s;
}

// The correctly-rounded float32 gamma.
//
// The 36.0f overflow guard is arithmetic, not inherited: MEASURED, the last
// float32 with a finite Gamma is 35.0401001, and binary64 holds Gamma out to
// x = 171.6, so ANY threshold in [35.041, 171] gives the same float32 answer
// on every argument.  36 is taken as the smallest integer above the overflow
// point, which is also what bounds the recurrence trip count above at 42.
// glibc picks the same number for the same arithmetic reason; the choice is
// free within a 136-wide interval and carries no expression.
__device__ float gfk_tgamma(float x)
{
    unsigned int ix = __float_as_uint(x);
    unsigned int ax = ix & 0x7fffffffu;

    if (ax >= 0x7f800000u) {                    /* inf / nan */
        if (ix == 0xff800000u) return GFK_TG_QNAN;      /* -inf  -> qNaN */
        return FADD(x, x);                              /* +inf  -> +inf
                                                           NaN   -> same, quieted */
    }
    if (ix < 0x80000000u) {                     /* x >= +0 */
        if (ix == 0u) return FDIV(1.0f, x);             /* +0 -> +inf */
        if (x >= 36.0f)
            return FMUL(FMUL(x, 0x1p127f), 0x1p127f);   /* -> +inf */
        return gfk_d2f_rn(gfk_tgamma_pos((double)x));
    }
    /* x < 0: reflection  Gamma(x) = pi / (sin(pi x) * Gamma(1-x)) */
    if (ix == 0x80000000u) return FDIV(1.0f, x);        /* -0 -> -inf */
    if (ax >= 0x4b000000u) return GFK_TG_QNAN;          /* |x| >= 2^23: every
                                                           float is an integer,
                                                           so every one is a pole */
    {
        double s = gfk_tgamma_sinpi((double)x);
        if (s == 0.0) return GFK_TG_QNAN;               /* negative integer: pole */
        if (x <= -42.0f)                                /* |Gamma| < 2^-150 below
                                                           here; measured, the
                                                           last non-zero float32
                                                           result is at -41.000042 */
            return (s < 0.0) ? -0.0f : 0.0f;
        {
            double g1 = gfk_tgamma_pos(DSUB(1.0, (double)x));   /* 1-x in (1,43) */
            return gfk_d2f_rn(DDIV(GFK_TG_PI, DMUL(s, g1)));
        }
    }
}

// WOOF's own float32 exp, contracted form: gfk_exp_fma.
//
// Requires glibc_flt32.cuh ahead of it in the assembled source (the loader's
// _EXTRA_HEADERS lists both, in that order).  It holds no table of its own:
// the scale 2**(k/32) comes from that header's gfk_exp2_core called with a
// zero polynomial (which returns its table scale exactly), and it reuses the
// polynomial constants GFK_EXP2F_P0..P2, GFK_EXP2F_SHIFT and gfk_d2f_rn.
//
// WHY A SECOND FORM.  gfk_exp evaluates every a*b+c of the algorithm as a
// separate multiply and add.  The reference a WRF column oracle calls on an
// x86-64 host with FMA (every oracle host here: W1's Xeon Platinum 8559C,
// the RTX 4090 workstation) is the C library's run-time-selected FMA build of expf, whose
// multiply-adds are fused.  The two forms differ at exactly two float32
// inputs over the whole line -- x = 0x4202422F (32.5646324) and
// x = 0xC27C65D9 (-63.0994606), 1 ULP each -- measured by
// tools/myjpbl_wrf461_oracle/libm_sweep.py over all 2**32 inputs on W1.
// gfk_exp_fma fuses the same five sites (the shift add, the reduced
// argument, both polynomial halves and their combination) with explicit
// __fma_rn, so it is unaffected by --fmad and matches the reference at every
// input (same sweep).  The MYJ PBL calls it; kernels/mynn_pbl.cu carries the
// same contracted form privately as mynn_glibc_expf.

__device__ float gfk_exp_fma(float x)
{
    unsigned int ix = __float_as_uint(x);
    unsigned int abstop = (ix >> 20) & 0x7ffu;
    if (abstop >= ((__float_as_uint(88.0f)) >> 20)) {
        if (ix == 0xff800000u) return 0.0f;
        if (abstop >= (0x7f800000u >> 20)) return __fadd_rn(x, x);
        if (x > __int_as_float(0x42b17218)) return __int_as_float(0x7f800000);
        if (x < -__int_as_float(0x42cff1b4)) return 0.0f;
    }
    const double inv_ln2_n = 0x1.71547652b82fep+0 * 32.0;
    double xd = (double)x;
    double kd = __fma_rn(inv_ln2_n, xd, GFK_EXP2F_SHIFT);
    kd = __dsub_rn(kd, GFK_EXP2F_SHIFT);
    double r = __fma_rn(inv_ln2_n, xd, -kd);
    // kd is an integer-valued double, so kd + SHIFT inside the core
    // rebuilds the fused shift sum above bit for bit (the same table index
    // and exponent bits); with a zero
    // polynomial the core returns 1.0 * scale, the scale exactly.
    double s = gfk_exp2_core(kd, GFK_EXP2F_SHIFT, 0.0, 0.0, 0.0, 0ULL);
    double z = __fma_rn(GFK_EXP2F_P0 / 32.0 / 32.0 / 32.0, r,
                        GFK_EXP2F_P1 / 32.0 / 32.0);
    double r2 = __dmul_rn(r, r);
    double y = __fma_rn(GFK_EXP2F_P2 / 32.0, r, 1.0);
    y = __dmul_rn(__fma_rn(z, r2, y), s);
    return gfk_d2f_rn(y);
}

// ======================================================================
// THIRD-PARTY NOTICE. Parts of this file are hand transcriptions of
// third-party work. ArWen distributes the file under the Apache License
// 2.0; the notices below belong to the transcribed parts and are kept
// here because their own licences require it. Full texts are in the
// repository NOTICE, licenses/, and beside the code in
// gpuwm/core/kernels/LICENSE-third-party.txt.
//
// Arm optimized-routines: glibc_sinf / glibc_cosf and their data tables.
// github.com/ARM-software/optimized-routines: math/sinf.c, math/cosf.c,
// math/sincosf.h, math/sincosf_data.c, published in 2018 and imported
// into glibc in August 2018; transcribed here from glibc 2.43
// sysdeps/ieee754/flt-32/s_sinf.c, s_cosf.c, s_sincosf.h,
// sincosf_poly.h and s_sincosf_data.c (x86 FMA IFUNC variant).
//
// Copyright (c) 2018-2024, Arm Limited.
// Copyright (c) 2018-2019, Arm Limited.
// SPDX-License-Identifier: MIT
//
// Taken under the MIT branch of Arm's grant, as glibc_flt32.cuh is.
// The algorithm, constants and tables reproduce Arm's work; glibc's
// integration macros, aliases and IFUNC wrapper are not transcribed.
// Arm 5e8389113b47 (2018), MIT relicensing 11253b0b9d6b,
// current fe09d4e7ed62. The full permission notice is reproduced in
// licenses/LICENSE-Arm-optimized-routines-MIT.txt and the files above.
//
// CORE-MATH MIT transcriptions from sysdeps/ieee754/flt-32/:
// e_asinf.c and s_asincosf_data.c (src/binary32/asin/asinf.c, bc385c2);
// e_acosf.c (src/binary32/acos/acosf.c, 56dd347);
// s_atanf.c (src/binary32/atan/atanf.c, a8066a5);
// s_tanf.c (src/binary32/tan/tanf.c, 59d21d7).
// Copyright (c) 2022-2024 Alexei Sibidanov.
// Copyright (c) 2023-2024 Alexei Sibidanov.
// Full text: licenses/LICENSE-CORE-MATH-MIT.txt.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
// No SunPro FDLIBM code is used.
// ======================================================================
// Target: Ubuntu GLIBC 2.43-2ubuntu2.4, x86-64 AVX2+FMA, FE_TONEAREST.
// sinf/cosf use __sinf_fma/__cosf_fma. The other four scalar functions
// have no CPU IFUNC. Floating exceptions and errno are not device outputs.
// Arithmetic is pinned to the selected libm machine instructions, not CUDA
// trig builtins. Use -std=c++17. No FADD/FMUL macros are defined here.
#ifndef GPUWM_GLIBC_TRIG_FLT32_CUH
#define GPUWM_GLIBC_TRIG_FLT32_CUH
typedef unsigned int gt_u32;
typedef unsigned long long gt_u64;
typedef long long gt_i64;

__device__ inline float gt_d2f(double x) { return __double2float_rn(x); }
__device__ inline double gt_identity(double x) { return x; }
__device__ inline gt_u64 gt_as_u64(double x) { return (gt_u64)__double_as_longlong(x); }
__device__ inline double gt_abs(double x) {
  return __longlong_as_double((long long)(gt_as_u64(x) & 0x7fffffffffffffffULL));
}
__device__ inline float gt_absf(float x) { return __uint_as_float(__float_as_uint(x) & 0x7fffffffU); }
__device__ inline double gt_sign(double x, double y) {
  return __longlong_as_double((long long)((gt_as_u64(x) & 0x7fffffffffffffffULL) | (gt_as_u64(y) & 0x8000000000000000ULL)));
}
__device__ inline float gt_signf(float x, float y) {
  return __uint_as_float((__float_as_uint(x)&0x7fffffffU) | (__float_as_uint(y)&0x80000000U));
}
__device__ inline float gt_invalid(float x) {
  gt_u32 t=__float_as_uint(x);
  return __uint_as_float((t & 0x7fffffffU)>0x7f800000U ? t|0x00400000U : 0xffc00000U);
}
__device__ inline double gt_roundeven(double x) {
  // This helper only receives |x| < 2^28 * 2/pi. Conversion pins RN.
  return (double)__double2ll_rn(x);
}
__device__ inline float gt_tiny_fmaf(float a,float b,float x) {
  // Only called by the three tiny-input paths below, where the fused
  // correction is less than half an ULP in RN. Preserve exponent-zero
  // inputs by bits because CUDA FTZ flushes even an explicit FMA's result.
  // All normal inputs retain glibc's explicit fused operation.
  if ((__float_as_uint(x)&0x7f800000U)==0) return x;
  return __fmaf_rn(a,b,x);
}

// s_sincosf_data.c and sincosf_poly.h.
struct gt_sincos_t { double sign[4], hpi_inv, hpi, c0,c1,c2,c3,c4,s1,s2,s3; };
__device__ const gt_sincos_t gt_sincos_table[2] = {
 {{1,-1,-1,1},0x1.45F306DC9C883p+23,0x1.921FB54442D18p0,
  0x1p0,-0x1.ffffffd0c621cp-2,0x1.55553e1068f19p-5,-0x1.6c087e89a359dp-10,0x1.99343027bf8c3p-16,
  -0x1.555545995a603p-3,0x1.1107605230bc4p-7,-0x1.994eb3774cf24p-13},
 {{1,-1,-1,1},0x1.45F306DC9C883p+23,0x1.921FB54442D18p0,
  -0x1p0,0x1.ffffffd0c621cp-2,-0x1.55553e1068f19p-5,0x1.6c087e89a359dp-10,-0x1.99343027bf8c3p-16,
  -0x1.555545995a603p-3,0x1.1107605230bc4p-7,-0x1.994eb3774cf24p-13}
};
__device__ const gt_u32 gt_inv_pio4[24] = {
 0xa2,0xa2f9,0xa2f983,0xa2f9836e,0xf9836e4e,0x836e4e44,0x6e4e4415,0x4e441529,
 0x441529fc,0x1529fc27,0x29fc2757,0xfc2757d1,0x2757d1f5,0x57d1f534,0xd1f534dd,0xf534ddc0,
 0x34ddc0db,0xddc0db62,0xc0db6295,0xdb629599,0x6295993c,0x95993c43,0x993c4390,0x3c439041
};
__device__ inline float gt_sinf_poly(double x,double x2,const gt_sincos_t* p,int n) {
 if (!(n&1)) {
  double x3=__dmul_rn(x,x2);
  double s1=__fma_rn(x2,p->s3,p->s2);
  double x7=__dmul_rn(x3,x2);
  double s=__fma_rn(x3,p->s1,x);
  return gt_d2f(__fma_rn(x7,s1,s));
 }
 double x4=__dmul_rn(x2,x2);
 double c2=__fma_rn(x2,p->c4,p->c3);
 double c1=__fma_rn(x2,p->c1,p->c0);
 double x6=__dmul_rn(x4,x2);
 double c=__fma_rn(x4,p->c2,c1);
 return gt_d2f(__fma_rn(x6,c2,c));
}
__device__ inline double gt_reduce_large(gt_u32 xi,int* np) {
 const gt_u32* arr=&gt_inv_pio4[(xi>>26)&15];
 int shift=(xi>>23)&7;
 xi=((xi&0xffffffU)|0x800000U)<<shift;
 gt_u64 r0=(gt_u32)(xi*arr[0]);
 gt_u64 r1=(gt_u64)xi*arr[4], r2=(gt_u64)xi*arr[8];
 r0=((r2>>32)|(r0<<32))+r1;
 gt_u64 n=(r0+(1ULL<<61))>>62;
 r0-=n<<62;
 *np=(int)n;
 return __dmul_rn((double)(gt_i64)r0,0x1.921FB54442D18p-62);
}
__device__ inline float gt_sincos(float y,int cosine) {
 gt_u32 yi=__float_as_uint(y), top=(yi>>20)&0x7ffU;
 if (top>=0x7f8U) return gt_invalid(y);
 if (top<0x398U) return cosine ? 1.0f : y;
 double x=(double)y;
 const gt_sincos_t* p=&gt_sincos_table[0];
 int n=0, sign=0;
 if (top>=0x3f4U) {
  if (top<0x42fU) {
   double r=__dmul_rn(x,p->hpi_inv);
   n=((int)r+0x800000)>>24;
   x=__fma_rn(-(double)n,p->hpi,x);
  } else {
   x=gt_reduce_large(yi,&n);
   sign=(int)(yi>>31);
  }
  int q=(n+sign)&3;
  if (q&2) p=&gt_sincos_table[1];
  double x2=__dmul_rn(x,x);
  return gt_sinf_poly(__dmul_rn(x,gt_sincos_table[0].sign[q]),x2,p,n^cosine);
 }
 return gt_sinf_poly(x,__dmul_rn(x,x),p,cosine);
}
__device__ float glibc_sinf(float x) { return gt_sincos(x,0); }
__device__ float glibc_cosf(float x) { return gt_sincos(x,1); }

__device__ const double gt_asincos_c0[12] = {
 0x1.555555555529cp-3,0x1.333333337e0ddp-4,0x1.6db6db3b4465ep-5,
 0x1.f1c72e13ac306p-6,0x1.6e89cebe06bc4p-6,0x1.1c6dcf5289094p-6,
 0x1.c6dbbcc7c6315p-7,0x1.8f8dc2615e996p-7,0x1.a5833b7bf15e8p-8,
 0x1.43f44ace1665cp-6,-0x1.0fb17df881c73p-6,0x1.07520c026b2d6p-5
};
__device__ const double gt_asincos_c1[12] = {
 0x1.6a09e667f3bcbp+0,0x1.e2b7dddff2db9p-4,0x1.b27247ab42dbcp-6,
 0x1.02995cc4e0744p-7,0x1.5ffb0276ec8eap-9,0x1.033885a928decp-10,
 0x1.911f2be23f8c7p-12,0x1.4c3c55d2437fdp-13,0x1.af477e1d7b461p-15,
 0x1.abd6bdff67dcbp-15,-0x1.1717e86d0fa28p-16,0x1.6ff526de46023p-16
};
// s_tanf.c rbig: limb form of glibc's unsigned 128-bit multiplies.
__device__ const gt_u64 gt_tan_ipi[4] = {
 0xfe5163abdebbc562ULL,0xdb6295993c439041ULL,0xfc2757d1f534ddc0ULL,0xa2f9836e4e441529ULL
};
__device__ inline double gt_rbig(gt_u32 u,int* q) {
 gt_u64 m=(u&0x7fffffU)|0x800000U;
 gt_u64 h=__umul64hi(m,gt_tan_ipi[0]);
 gt_u64 p1l=m*gt_tan_ipi[1]+h;
 h=__umul64hi(m,gt_tan_ipi[1])+(p1l<h);
 gt_u64 p2l=m*gt_tan_ipi[2]+h;
 h=__umul64hi(m,gt_tan_ipi[2])+(p2l<h);
 gt_u64 p3l=m*gt_tan_ipi[3]+h;
 gt_u64 p3h=__umul64hi(m,gt_tan_ipi[3])+(p3l<h);
 int s=(int)((u>>23)&255)-127-23;
 gt_u64 a; gt_u32 i;
 if(s<64) { i=(gt_u32)((p3h<<s)|(p3l>>(64-s))); a=(p3l<<s)|(p2l>>(64-s)); }
 else if(s==64) { i=(gt_u32)p3l; a=p2l; }
 else { i=(gt_u32)((p3l<<(s-64))|(p2l>>(128-s))); a=(p2l<<(s-64))|(p1l>>(128-s)); }
 gt_u32 sign=0U-(u>>31);
 i-= (gt_u32)((gt_i64)a>>63);
 i=(i^sign)-sign;
 *q=(int)i;
 gt_i64 az=(gt_i64)(a^(gt_u64)(gt_i64)(int)sign);
 return __dmul_rn((double)az,0x1p-64);
}

// glibc-2.43 sysdeps/ieee754/flt-32/e_asinf.c
__device__ const double gt_asinf_b[] = {0x1.0000000000005p+0, 0x1.55557aeca105dp-3, 0x1.3314ec3db7d12p-4, 0x1.775738a5a6f92p-5, 0x1.5d5f7ce1c8538p-8, 0x1.605c6d58740fp-2, -0x1.5728b732d73c6p+1, 0x1.f152170f151ebp+3, -0x1.f962ea3ca992ep+5, 0x1.71971e17375ap+7, -0x1.860512b4ba23p+8, 0x1.26a3b8d4bdb14p+9, -0x1.36f2ea5698b51p+9, 0x1.b3d722aebfa2ep+8, -0x1.6cf89703b1289p+7, 0x1.1518af6a65e2dp+5};

__device__ static float gt_asinf_as_special(float x)
{
  gt_u32 ax = __float_as_uint(x) << 1;
  if (ax > (0xffu << 24))
    return gt_invalid(x); // Preserve the input NaN sign and payload.
  return gt_invalid(0.0);
}


__device__ static double gt_asinf_poly12(double z, const double *c)
{
  double z2 = __dmul_rn(z, z);
  double z4 = __dmul_rn(z2, z2);
  double c0 = __dadd_rn(c[0], __dmul_rn(z, c[1]));
  double c2 = __dadd_rn(c[2], __dmul_rn(z, c[3]));
  double c4 = __dadd_rn(c[4], __dmul_rn(z, c[5]));
  double c6 = __dadd_rn(c[6], __dmul_rn(z, c[7]));
  double c8 = __dadd_rn(c[8], __dmul_rn(z, c[9]));
  double c10 = __dadd_rn(c[10], __dmul_rn(z, c[11]));
  c0 = __dadd_rn(c0, __dmul_rn(c2, z2));
  c4 = __dadd_rn(c4, __dmul_rn(c6, z2));
  c8 = __dadd_rn(c8, __dmul_rn(z2, c10));
  c0 = __dadd_rn(c0, __dmul_rn(z4, __dadd_rn(c4, __dmul_rn(z4, c8))));
  return c0;
}


__device__ float glibc_asinf(float x)
{
  const double pi2 = 0x1.921fb54442d18p+0;
  double xs = x;
  double r;
  gt_u32 ax = __float_as_uint(x) << 1;
  if (ax > (0x7f << 24))
    return gt_asinf_as_special(x);
  if (ax < 0x7ec29000u)
  {
    if (ax < (115 << 24))
      return gt_tiny_fmaf(x, 0x1p-25f, x);
    /* table gt_asinf_b */;
    double z = xs;
    double z2 = __dmul_rn(z, z);
    double z4 = __dmul_rn(z2, z2);
    double z8 = __dmul_rn(z4, z4);
    double z16 = __dmul_rn(z8, z8);
    r = __dmul_rn(z, __dadd_rn(__dadd_rn(__dadd_rn(__dadd_rn(gt_asinf_b[0], __dmul_rn(z2, gt_asinf_b[1])), __dmul_rn(z4, __dadd_rn(gt_asinf_b[2], __dmul_rn(z2, gt_asinf_b[3])))), __dmul_rn(z8, __dadd_rn(__dadd_rn(gt_asinf_b[4], __dmul_rn(z2, gt_asinf_b[5])), __dmul_rn(z4, __dadd_rn(gt_asinf_b[6], __dmul_rn(z2, gt_asinf_b[7])))))), __dmul_rn(z16, __dadd_rn(__dadd_rn(__dadd_rn(gt_asinf_b[8], __dmul_rn(z2, gt_asinf_b[9])), __dmul_rn(z4, __dadd_rn(gt_asinf_b[10], __dmul_rn(z2, gt_asinf_b[11])))), __dmul_rn(z8, __dadd_rn(__dadd_rn(gt_asinf_b[12], __dmul_rn(z2, gt_asinf_b[13])), __dmul_rn(z4, __dadd_rn(gt_asinf_b[14], __dmul_rn(z2, gt_asinf_b[15])))))))));
    float ub = gt_d2f(r);
    float lb = gt_d2f(__dsub_rn(r, __dmul_rn(z, 0x1.efa8ebp-31)));
    if (ub == lb)
      return ub;
  }
  if (ax < (0x7eu << 24))
  {
    double z = xs;
    double z2 = __dmul_rn(z, z);
    double c0 = gt_asinf_poly12(z2, gt_asincos_c0);
    r = __dadd_rn(z, __dmul_rn(__dmul_rn(z, z2), c0));
  }
  else
  {
    if (ax == 0x7e55688au)
      return __fadd_rn(gt_signf(0x1.75b8a2p-1f, x), gt_signf(0x1p-26f, x));
    if (ax == 0x7e107434u)
      return __fadd_rn(gt_signf(0x1.1f4b64p-1f, x), gt_signf(0x1p-26f, x));
    double bx = gt_abs(xs);
    double z = __dsub_rn(1.0, bx);
    double s = __dsqrt_rn(z);
    r = __dsub_rn(pi2, __dmul_rn(s, gt_asinf_poly12(z, gt_asincos_c1)));
    r = gt_sign(r, xs);
  }
  return gt_d2f(r);
}


// glibc-2.43 sysdeps/ieee754/flt-32/e_acosf.c
__device__ const double gt_acosf_o[] = {0, 0x1.921fb54442d18p+1};
__device__ const double gt_acosf_b[] = {0x1.fffffffd9ccb8p-1, 0x1.5555c94838007p-3, 0x1.32ded4b7c20fap-4, 0x1.8566df703309ep-5, -0x1.980c959bec9a3p-6, 0x1.56fbb04998344p-1, -0x1.403d8e4c49f52p+2, 0x1.b06c3e9f311eap+4, -0x1.9ea97c4e2c21fp+6, 0x1.200b8261cc61bp+8, -0x1.2274c2799a5c7p+9, 0x1.a558a59cc19d3p+9, -0x1.aca4b6a529ffp+9, 0x1.228744703f813p+9, -0x1.d7dbb0b322228p+7, 0x1.5c2018c0c0105p+5};

__device__ static float gt_acosf_as_special(float x)
{
  const float pih = gt_d2f(0x1.921fb6p+1);
  const float pil = -0x1p-24f;
  gt_u32 t = __float_as_uint(x);
  if (t == (0x7fu << 23))
    return 0.0f;
  if (t == (0x17fu << 23))
    return __fadd_rn(pih, pil);
  gt_u32 ax = t << 1;
  if (ax > (0xffu << 24))
    return gt_invalid(x); // Preserve the input NaN sign and payload.
  return gt_invalid(0.0);
}


__device__ inline static double gt_acosf_poly12(double z, const double *c)
{
  double z2 = __dmul_rn(z, z);
  double z4 = __dmul_rn(z2, z2);
  double c0 = __dadd_rn(c[0], __dmul_rn(z, c[1]));
  double c2 = __dadd_rn(c[2], __dmul_rn(z, c[3]));
  double c4 = __dadd_rn(c[4], __dmul_rn(z, c[5]));
  double c6 = __dadd_rn(c[6], __dmul_rn(z, c[7]));
  double c8 = __dadd_rn(c[8], __dmul_rn(z, c[9]));
  double c10 = __dadd_rn(c[10], __dmul_rn(z, c[11]));
  c0 = __dadd_rn(c0, __dmul_rn(c2, z2));
  c4 = __dadd_rn(c4, __dmul_rn(c6, z2));
  c8 = __dadd_rn(c8, __dmul_rn(z2, c10));
  c0 = __dadd_rn(c0, __dmul_rn(z4, __dadd_rn(c4, __dmul_rn(z4, c8))));
  return c0;
}


__device__ float glibc_acosf(float x)
{
  double pi2 = 0x1.921fb54442d18p+0;
  /* table gt_acosf_o */;
  double xs = x;
  double r;
  gt_u32 t = __float_as_uint(x);
  gt_u32 ax = t << 1;
  if (ax >= (0x7f << 24))
    return gt_acosf_as_special(x);
  if (ax < 0x7ec2a1dcu)
  {
    /* table gt_acosf_b */;
    if (ax <= 0x40000000u)
      return gt_d2f(gt_identity(pi2));
    double z = xs;
    double z2 = __dmul_rn(z, z);
    double z4 = __dmul_rn(z2, z2);
    double z8 = __dmul_rn(z4, z4);
    double z16 = __dmul_rn(z8, z8);
    r = __dmul_rn(z, __dadd_rn(__dadd_rn(__dadd_rn(__dadd_rn(gt_acosf_b[0], __dmul_rn(z2, gt_acosf_b[1])), __dmul_rn(z4, __dadd_rn(gt_acosf_b[2], __dmul_rn(z2, gt_acosf_b[3])))), __dmul_rn(z8, __dadd_rn(__dadd_rn(gt_acosf_b[4], __dmul_rn(z2, gt_acosf_b[5])), __dmul_rn(z4, __dadd_rn(gt_acosf_b[6], __dmul_rn(z2, gt_acosf_b[7])))))), __dmul_rn(z16, __dadd_rn(__dadd_rn(__dadd_rn(gt_acosf_b[8], __dmul_rn(z2, gt_acosf_b[9])), __dmul_rn(z4, __dadd_rn(gt_acosf_b[10], __dmul_rn(z2, gt_acosf_b[11])))), __dmul_rn(z8, __dadd_rn(__dadd_rn(gt_acosf_b[12], __dmul_rn(z2, gt_acosf_b[13])), __dmul_rn(z4, __dadd_rn(gt_acosf_b[14], __dmul_rn(z2, gt_acosf_b[15])))))))));
    float ub = gt_d2f(__dsub_rn(0x1.921fb54574191p+0, r));
    float lb = gt_d2f(__dsub_rn(0x1.921fb543118ap+0, r));
    if (ub == lb)
      return ub;
  }
  if (ax < (0x7eu << 24))
  {
    if (t == 0x328885a3u)
      return __fadd_rn(0x1.921fb6p+0f, 0x1p-25f);
    if (t == 0x39826222u)
      return __fadd_rn(0x1.920f6ap+0f, 0x1p-25f);
    double x2 = __dmul_rn(xs, xs);
    r = __dsub_rn(__dsub_rn(pi2, xs), __dmul_rn(__dmul_rn(xs, x2), gt_acosf_poly12(x2, gt_asincos_c0)));
  }
  else
  {
    double bx = gt_abs(xs);
    double z = __dsub_rn(1.0, bx);
    double s = gt_sign(__dsqrt_rn(z), xs);
    r = __dadd_rn(gt_acosf_o[t >> 31], __dmul_rn(s, gt_acosf_poly12(z, gt_asincos_c1)));
  }
  return gt_d2f(r);
}


// glibc-2.43 sysdeps/ieee754/flt-32/s_atanf.c
__device__ const double gt_atanf_cn[] = {0x1.51eccde075d67p-2, 0x1.a76bb5637f2f2p-1, 0x1.81e0eed20de88p-1, 0x1.376c8ca67d11dp-2, 0x1.aec7b69202ac6p-5, 0x1.9561899acc73ep-9, 0x1.bf9fa5b67e6p-16};
__device__ const double gt_atanf_cd[] = {0x1.51eccde075d66p-2, 0x1.dfbdd7b392d28p-1, 0x1p+0, 0x1.fd22bf0e89b54p-2, 0x1.d91ff8b576282p-4, 0x1.653ea99fc9bbp-7, 0x1.1e7fcc202340ap-12};

__device__ float glibc_atanf(float x)
{
  const double pi2 = 0x1.921fb54442d18p+0;
  gt_u32 t = __float_as_uint(x);
  int e = (t >> 23) & 0xff;
  bool gt = e >= 127;
  gt_u32 ta = t & 0x7fffffff;
  if (ta >= 0x4c700518u)
  {
    if (ta > 0x7f800000u)
      return gt_invalid(x); // Preserve the input NaN sign and payload.
    return gt_d2f(gt_sign(pi2, (double) x));
  }
  if (e < (127 - 13))
  {
    if (e < (127 - 25))
    {
      if (!(t << 1))
        return x;
      return gt_tiny_fmaf(-x, gt_absf(x), x);
    }
    return __fmaf_rn(__fmul_rn(-0x1.555556p-2f, x), __fmul_rn(x, x), x);
  }
  double z = x;
  if (gt)
    z = __ddiv_rn(1, z);
  double z2 = __dmul_rn(z, z);
  double z4 = __dmul_rn(z2, z2);
  double z8 = __dmul_rn(z4, z4);
  /* table gt_atanf_cn */;
  /* table gt_atanf_cd */;
  double cn0 = __dadd_rn(gt_atanf_cn[0], __dmul_rn(z2, gt_atanf_cn[1]));
  double cn2 = __dadd_rn(gt_atanf_cn[2], __dmul_rn(z2, gt_atanf_cn[3]));
  double cn4 = __dadd_rn(gt_atanf_cn[4], __dmul_rn(z2, gt_atanf_cn[5]));
  double cn6 = gt_atanf_cn[6];
  cn0 = __dadd_rn(cn0, __dmul_rn(z4, cn2));
  cn4 = __dadd_rn(cn4, __dmul_rn(z4, cn6));
  cn0 = __dadd_rn(cn0, __dmul_rn(z8, cn4));
  cn0 = __dmul_rn(cn0, z);
  double cd0 = __dadd_rn(gt_atanf_cd[0], __dmul_rn(z2, gt_atanf_cd[1]));
  double cd2 = __dadd_rn(gt_atanf_cd[2], __dmul_rn(z2, gt_atanf_cd[3]));
  double cd4 = __dadd_rn(gt_atanf_cd[4], __dmul_rn(z2, gt_atanf_cd[5]));
  double cd6 = gt_atanf_cd[6];
  cd0 = __dadd_rn(cd0, __dmul_rn(z4, cd2));
  cd4 = __dadd_rn(cd4, __dmul_rn(z4, cd6));
  cd0 = __dadd_rn(cd0, __dmul_rn(z8, cd4));
  double r = __ddiv_rn(cn0, cd0);
  if (!gt)
    return gt_d2f(r);
  r = __dadd_rn(__dsub_rn(gt_sign(0x1.0fdaa22168c23p-7, z), r), gt_sign(0x1.9p0, z));
  return gt_d2f(r);
}


// glibc-2.43 sysdeps/ieee754/flt-32/s_tanf.c
__device__ const double gt_tanf_cn[] = {0x1.921fb54442d18p+0, -0x1.fd226e573289fp-2, 0x1.b7a60c8dac9f6p-6, -0x1.725beb40f33e5p-13};
__device__ const double gt_tanf_cd[] = {0x1p+0, -0x1.2395347fb829dp+0, 0x1.2313660f29c36p-3, -0x1.9a707ab98d1c1p-9};
__device__ const double gt_tanf_s[] = {0, 1};
__device__ const struct 
    {
      float arg;
      float rh;
      float rl;
    } gt_tanf_st[] = {{0x1.143ec4p+0f, 0x1.ddf9f6p+0f, -0x1.891d24p-52f}, {0x1.ada6aap+27f, 0x1.e80304p-3f, 0x1.419f46p-58f}, {0x1.af61dap+48f, 0x1.60d1c8p-2f, -0x1.2d6c3ap-55f}, {0x1.0088bcp+52f, 0x1.ca1edp+0f, 0x1.f6053p-53f}, {0x1.f90dfcp+72f, 0x1.597f9cp-1f, 0x1.925978p-53f}, {0x1.cc4e22p+85f, -0x1.f33584p+1f, 0x1.d7254ap-51f}, {0x1.a6ce12p+86f, -0x1.c5612ep-1f, -0x1.26c33ep-53f}, {0x1.6a0b76p+102f, -0x1.e42a1ep+0f, -0x1.1dc906p-52f}};

__device__ inline static double gt_tanf_rltl(float z, int *q)
{
  double x = z;
  double idl = __dmul_rn(-0x1.b1bbead603d8bp-32, x);
  double idh = __dmul_rn(0x1.45f306ep-1, x);
  double id = gt_roundeven(idh);
  *q = (gt_i64) id;
  return __dadd_rn(__dsub_rn(idh, id), idl);
}


__device__ float glibc_tanf(float x)
{
  gt_u32 t = __float_as_uint(x);
  int e = (t >> 23) & 0xff;
  int i;
  double z;
  if (e < (127 + 28))
  {
    if (e < 115)
    {
      if (e < 102)
        return gt_tiny_fmaf(x, gt_absf(x), x);
      float x2 = __fmul_rn(x, x);
      return __fmaf_rn(x, __fmul_rn(0x1.555556p-2f, x2), x);
    }
    z = gt_tanf_rltl(x, &i);
  }
  else
    if (e < 0xff)
    z = gt_rbig(t, &i);
  else
  {
    if (t << 9)
      return gt_invalid(x); // Preserve the input NaN sign and payload.
    return gt_invalid(x);
  }
  double z2 = __dmul_rn(z, z);
  double z4 = __dmul_rn(z2, z2);
  /* table gt_tanf_cn */;
  /* table gt_tanf_cd */;
  /* table gt_tanf_s */;
  double n = __dadd_rn(gt_tanf_cn[0], __dmul_rn(z2, gt_tanf_cn[1]));
  double n2 = __dadd_rn(gt_tanf_cn[2], __dmul_rn(z2, gt_tanf_cn[3]));
  n = __dadd_rn(n, __dmul_rn(z4, n2));
  double d = __dadd_rn(gt_tanf_cd[0], __dmul_rn(z2, gt_tanf_cd[1]));
  double d2 = __dadd_rn(gt_tanf_cd[2], __dmul_rn(z2, gt_tanf_cd[3]));
  d = __dadd_rn(d, __dmul_rn(z4, d2));
  n = __dmul_rn(n, z);
  double s0 = gt_tanf_s[i & 1];
  double s1 = gt_tanf_s[1 - (i & 1)];
  double r1 = __ddiv_rn(__dsub_rn(__dmul_rn(n, s1), __dmul_rn(d, s0)), __dadd_rn(__dmul_rn(n, s0), __dmul_rn(d, s1)));
  gt_u64 tail = (gt_as_u64(r1) + 7) & ((~0ULL) >> 35);
  if (tail <= 14)
  {
    /* table gt_tanf_st */;
    gt_u32 ax = t & ((~0u) >> 1);
    gt_u32 sgn = t >> 31;
    for (int j = 0; j < 8; j++)
    {
      if (__float_as_uint(gt_tanf_st[j].arg) == ax)
      {
        if (sgn)
          return __fsub_rn(-gt_tanf_st[j].rh, gt_tanf_st[j].rl);
        else
          return __fadd_rn(gt_tanf_st[j].rh, gt_tanf_st[j].rl);
      }
    }

  }
  return gt_d2f(r1);
}


#endif

// Correctly rounded binary32 tanh, from glibc's CORE-MATH implementation.
// Source revision: 0e0be3ed803714285831b14a0032bc91b93e9027, s_tanhf.c.
// This matches the native SFIRE oracle's glibc 2.43. Legacy glibc 2.39
// tanhf differs by one ULP on ordinary TG arguments below one.
/*
Copyright (c) 2022-2024 Alexei Sibidanov.

The original version was copied from the CORE-MATH project,
src/binary32/tanh/tanhf.c, revision b1ecd83.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#ifndef GPUWM_SFIRE_LIBM_CUH
#define GPUWM_SFIRE_LIBM_CUH
__device__ float sfire_tanhf(float x)
{
    double z=(double)x;
    unsigned ux=__float_as_uint(x);
    int e=(ux>>23)&0xff;
    if(e==0xff) {
        if(ux<<9) return x+x;
        return ux>>31 ? -1.0f : 1.0f;
    }
    if(e<115) {
        if(e<102) {
            if((ux<<1)==0) return x;
            return __fmaf_rn(-x,fabsf(x),x);
        }
        float x2=x*x;
        return __fmaf_rn(x,-0x1.555556p-2f*x2,x);
    }
    if((ux<<1)>(0x41102cb3u<<1))
        return copysignf(1.0f,x)-copysignf(0x1p-25f,x);
    double z2=z*z,z4=z2*z2,z8=z4*z4;
    const double cn[]={
        0x1p+0,0x1.30877b8b72d33p-3,0x1.694aa09ae9e5ep-8,
        0x1.4101377abb729p-14,0x1.e0392b1db0018p-22,0x1.2533756e546f7p-30,
        0x1.d62e5abe6ae8ap-41,0x1.b06be534182dep-54};
    const double cd[]={
        0x1p+0,0x1.ed99131b0ebeap-2,0x1.0d27ed6c95a69p-5,
        0x1.7cbdaca0e9fccp-11,0x1.b4e60b892578ep-18,0x1.a6f707c5c71abp-26,
        0x1.35a8b6e2cd94cp-35,0x1.ca8230677aa01p-47};
    double n0=cn[0]+z2*cn[1],n2=cn[2]+z2*cn[3];
    double n4=cn[4]+z2*cn[5],n6=cn[6]+z2*cn[7];
    n0=n0+z4*n2;n4=n4+z4*n6;n0=n0+z8*n4;
    double d0=cd[0]+z2*cd[1],d2=cd[2]+z2*cd[3];
    double d4=cd[4]+z2*cd[5],d6=cd[6]+z2*cd[7];
    d0=d0+z4*d2;d4=d4+z4*d6;d0=d0+z8*d4;
    return __double2float_rn(__ddiv_rn(z*n0,d0));
}
#endif

// log10f: CORE-MATH src/binary32/log10/log10f.c (revision bc385c2) as
// glibc 2.43 carries it in sysdeps/ieee754/flt-32/e_log10f.c (no x86-64
// IFUNC variant), transcribed line for line; correctly rounded.
//   Copyright (c) 2022-2023 Alexei Sibidanov.
//   SPDX-License-Identifier: MIT.  Permission is hereby granted, free of
//   charge, to any person obtaining a copy of this software and associated
//   documentation files (the "Software"), to deal in the Software without
//   restriction, including without limitation the rights to use, copy,
//   modify, merge, publish, distribute, sublicense, and/or sell copies of
//   the Software, and to permit persons to whom the Software is furnished to
//   do so, subject to the following conditions: The above copyright notice
//   and this permission notice shall be included in all copies or
//   substantial portions of the Software.  THE SOFTWARE IS PROVIDED "AS IS",
//   WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.
//   Full text: licenses/LICENSE-CORE-MATH-MIT.txt.
__device__ const double ubm_l10_tr[65] = {
  0x1p+0,         0x1.f81f82p-1,  0x1.f07c1fp-1,  0x1.e9131acp-1,
  0x1.e1e1e1ep-1, 0x1.dae6077p-1, 0x1.d41d41dp-1, 0x1.cd85689p-1,
  0x1.c71c71cp-1, 0x1.c0e0704p-1, 0x1.bacf915p-1, 0x1.b4e81b5p-1,
  0x1.af286bdp-1, 0x1.a98ef6p-1,  0x1.a41a41ap-1, 0x1.9ec8e95p-1,
  0x1.999999ap-1, 0x1.948b0fdp-1, 0x1.8f9c19p-1,  0x1.8acb90fp-1,
  0x1.8618618p-1, 0x1.8181818p-1, 0x1.7d05f41p-1, 0x1.78a4c81p-1,
  0x1.745d174p-1, 0x1.702e05cp-1, 0x1.6c16c17p-1, 0x1.6816817p-1,
  0x1.642c859p-1, 0x1.605816p-1,  0x1.5c9882cp-1, 0x1.58ed231p-1,
  0x1.5555555p-1, 0x1.51d07ebp-1, 0x1.4e5e0a7p-1, 0x1.4afd6ap-1,
  0x1.47ae148p-1, 0x1.446f865p-1, 0x1.4141414p-1, 0x1.3e22cbdp-1,
  0x1.3b13b14p-1, 0x1.3813814p-1, 0x1.3521cfbp-1, 0x1.323e34ap-1,
  0x1.2f684bep-1, 0x1.2c9fb4ep-1, 0x1.29e412ap-1, 0x1.27350b9p-1,
  0x1.2492492p-1, 0x1.21fb781p-1, 0x1.1f7047ep-1, 0x1.1cf06aep-1,
  0x1.1a7b961p-1, 0x1.1811812p-1, 0x1.15b1e5fp-1, 0x1.135c811p-1,
  0x1.1111111p-1, 0x1.0ecf56cp-1, 0x1.0c9715p-1,  0x1.0a6810ap-1,
  0x1.0842108p-1, 0x1.0624dd3p-1, 0x1.041041p-1,  0x1.0204081p-1,
  0.5};
__device__ const double ubm_l10_tl[65] = {
  -0x1.d45fd6237ebe3p-47, 0x1.b947689311b6ep-8, 0x1.b5e909c96d7d5p-7,
  0x1.45f4f59ed2165p-6,   0x1.af5f92cbd8f1ep-6, 0x1.0ba01a606de8cp-5,
  0x1.3ed119b9a2b7bp-5,   0x1.714834298eec2p-5, 0x1.a30a9d98357fbp-5,
  0x1.d41d512670813p-5,   0x1.02428c0f65519p-4, 0x1.1a23444eecc3ep-4,
  0x1.31b30543f4cb4p-4,   0x1.48f3ed39bfd04p-4, 0x1.5fe8049a0e423p-4,
  0x1.769140a6aa008p-4,   0x1.8cf1836c98cb3p-4, 0x1.a30a9d55541a1p-4,
  0x1.b8de4d1ee823ep-4,   0x1.ce6e4202ca2e6p-4, 0x1.e3bc1accace07p-4,
  0x1.f8c9683b5abd4p-4,   0x1.06cbd68ca9a6ep-3, 0x1.11142f19df73p-3,
  0x1.1b3e71fa7a97fp-3,   0x1.254b4d37a46e3p-3, 0x1.2f3b6912cbf07p-3,
  0x1.390f683115886p-3,   0x1.42c7e7fffc5a8p-3, 0x1.4c65808c78d3cp-3,
  0x1.55e8c50751c55p-3,   0x1.5f52445dec3d8p-3, 0x1.68a288c3f12p-3,
  0x1.71da17bdf0d19p-3,   0x1.7af973608afd9p-3, 0x1.84011952a2579p-3,
  0x1.8cf1837a7ea6p-3,    0x1.95cb2891e43d6p-3, 0x1.9e8e7b0f869ep-3,
  0x1.a73beaa5db18dp-3,   0x1.afd3e394558d3p-3, 0x1.b856cf060d9f1p-3,
  0x1.c0c5134de1ffcp-3,   0x1.c91f1371bc99fp-3, 0x1.d1652ffcd3f53p-3,
  0x1.d997c6f635e75p-3,   0x1.e1b733ab90f3bp-3, 0x1.e9c3ceadac856p-3,
  0x1.f1bdeec43a305p-3,   0x1.f9a5e7a5fa3fep-3, 0x1.00be05ac02f2bp-2,
  0x1.04a054d81a2d4p-2,   0x1.087a0835957fbp-2, 0x1.0c4b457099517p-2,
  0x1.101431aa1fe51p-2,   0x1.13d4f08b98dd8p-2, 0x1.178da53edb892p-2,
  0x1.1b3e71e9f9d58p-2,   0x1.1ee777defdeedp-2, 0x1.2288d7b48e23bp-2,
  0x1.2622b0f52e49fp-2,   0x1.29b522a4c6314p-2, 0x1.2d404b0e30f8p-2,
  0x1.30c4478f3fbe5p-2,   0x1.34413509f7915p-2};
__device__ const float ubm_l10_st[16] = {
  0x1p+0f,        0x1.4p+3f,      0x1.9p+6f,       0x1.f4p+9f,
  0x1.388p+13f,   0x1.86ap+16f,   0x1.e848p+19f,   0x1.312dp+23f,
  0x1.7d784p+26f, 0x1.dcd65p+29f, 0x1.2a05f2p+33f, 0.0f,
  0.0f,           0.0f,           0.0f,            0.0f};
__device__ const double ubm_l10_b[3] = {
  0x1.bcb7b15c5a2f8p-2, -0x1.bcbb1dbb88ebap-3, 0x1.2871c39d521c6p-3};
__device__ const double ubm_l10_c[7] = {
  0x1.bcb7b1526e50ep-2,  -0x1.bcb7b1526e53dp-3, 0x1.287a7636f3fa2p-3,
  -0x1.bcb7b146a14b3p-4, 0x1.63c627d5219cbp-4,  -0x1.2880736c8762dp-4,
  0x1.fc1ecf913961ap-5};
__device__ float ubm_log10f_special(float x)
{
  unsigned int ux = __float_as_uint(x);
  if (ux == 0x7f800000u)
    return x; /* +inf */
  unsigned int ax = ux << 1;
  if (ax == 0u)
    return __uint_as_float(0xff800000u); /* +-0: __math_divzerof (1) = -inf */
  if (ax > 0xff000000u)
    return x + x; /* nan */
  return __uint_as_float(0x7fc00000u); /* x < 0: __math_invalidf */
}
__device__ float ubm_log10f(float x)
{
  unsigned int ux = __float_as_uint(x);
  if (ux < (1u << 23) || ux >= 0x7f800000u)
    {
      if (ux == 0 || ux >= 0x7f800000u)
        return ubm_log10f_special(x);
      /* subnormal */
      int n = __clz(ux) - 8;
      ux <<= n;
      ux -= n << 23;
    }
  unsigned int m = ux & ((1u << 23) - 1), j = (m + (1u << (23 - 7))) >> (23 - 6);
  double ix = ubm_l10_tr[j], l = ubm_l10_tl[j];
  int e = ((int) ux >> 23) - 127;
  unsigned int je = e + 1;
  je = (je * 0x4d104d4u) >> 28;
  if (ux == __float_as_uint(ubm_l10_st[je]))
    return (float) je;
  double tz = __longlong_as_double(((long long) m | ((long long) 1023 << 23)) << (52 - 23));
  double z = __dsub_rn(__dmul_rn(tz, ix), 1.0), z2 = __dmul_rn(z, z);
  double r = __dadd_rn(
      __dadd_rn(__dadd_rn(__dmul_rn((double) e, 0x1.34413509f79ffp-2), l),
                __dmul_rn(z, ubm_l10_b[0])),
      __dmul_rn(z2, __dadd_rn(ubm_l10_b[1], __dmul_rn(z, ubm_l10_b[2]))));
  float ub = __double2float_rn(r), lb = __double2float_rn(__dadd_rn(r, 0x1.b008p-34));
  if (ub != lb)
    {
      const double *c = ubm_l10_c;
      double t456 = __dadd_rn(__dadd_rn(c[4], __dmul_rn(z, c[5])), __dmul_rn(z2, c[6]));
      double t23 = __dadd_rn(c[2], __dmul_rn(z, c[3]));
      double t01 = __dadd_rn(c[0], __dmul_rn(z, c[1]));
      double f = __dmul_rn(z, __dadd_rn(t01, __dmul_rn(z2, __dadd_rn(t23, __dmul_rn(z2, t456)))));
      f = __dsub_rn(f, __dmul_rn(0x1.0cee0ed4ca7e9p-54, (double) e));
      f = __dadd_rn(f, __dsub_rn(l, ubm_l10_tl[0]));
      double el = __dmul_rn((double) e, 0x1.34413509f7ap-2);
      r = __dadd_rn(el, f);
      ub = __double2float_rn(r);
      tz = r;
      if (!((((unsigned long long) __double_as_longlong(tz)) & ((1 << 28) - 1))))
        {
          double dr = __dadd_rn(__dsub_rn(el, r), f);
          r = __dadd_rn(r, __dmul_rn(dr, 32.0));
          ub = __double2float_rn(r);
        }
    }
  return ub;
}




#define SI_COMPILE_ONE
#define SI_SEAICE_ORIGINAL_MATH_AVAILABLE
// SPDX-License-Identifier: BSD-3-Clause
// Hand rules for RUC sea-ice helpers and driver boundaries at hex pin
// 16251589f19aec0012fd164d1ea060126208aed1. Whole-kernel composition is
// performed by the scheme reconcile.
// The binder retains the complete pre-launch state until both derivatives end.
// Pointer-table entries and strides follow the pinned oracle wrappers. See the
// corresponding specs for extent, non-alias, launch and failure-slot contracts.

// The device has one int32 refusal slot per source site. The host maps each
// nonzero slot to the full source id recorded in the corresponding spec.
using SeaiceFailure = int;

__device__ __forceinline__ bool si_finite(float x) {
    return (__float_as_uint(x) & 0x7f800000u) != 0x7f800000u;
}
__device__ __forceinline__ void si_fail(SeaiceFailure *f, int c,
                                        unsigned long long field,
                                        unsigned long long element,
                                        unsigned long long source) {
    (void)c; (void)field; (void)element;
    int slot;
    switch (source) {
      case 4711: case 4736: case 5007: case 5026: case 5064: slot=0; break;
      case 4725: case 5008: case 5028: case 5065: slot=1; break;
      case 4726: case 5009: case 5033: case 5066: slot=2; break;
      case 5046: slot=3; break;
      default: slot=0; break;
    }
    atomicOr(f+slot,1);
}
__device__ __forceinline__ float *si_f(unsigned long long *p, int k) {
    return reinterpret_cast<float *>(p[k]);
}
__device__ __forceinline__ int *si_i(unsigned long long *p, int k) {
    return reinterpret_cast<int *>(p[k]);
}
__device__ __forceinline__ bool si_check(float x, SeaiceFailure *f, int c,
                                         unsigned long long field,
                                         unsigned long long element,
                                         unsigned long long source) {
    if (si_finite(x)) return true;
    si_fail(f, c, field, element, source); return false;
}
// Each slot is [false count, true count]. Counts use exact integer addition.
__device__ __forceinline__ void si_branch(unsigned int *record, int slot, bool arm) {
    atomicAdd(record + 2 * slot + int(arm), 1u);
}
// Some source-ledger entries name both an if statement and its predicate.
// Record both without evaluating a short-circuited subpredicate early.
__device__ __forceinline__ void si_branch_pair(unsigned int *record, int slot, bool arm) {
    if (record) { si_branch(record,slot,arm); si_branch(record,slot+1,arm); }
}

// Building blocks for the ice kernel. These do not constitute an implementation
// of oracle_ice_ruc_ice: its caller must supply its executed iteration index and
// accumulate table cotangents in a fixed order. Table slots are zero based here.
struct SiQsnPoint { int slot; float weight, dr_dtn; };
__device__ __forceinline__ bool si_qsn_point(float tn, SiQsnPoint &q) {
    if (!si_finite(tn)) return false;
    float r=__fadd_rn(__fdiv_rn(__fsub_rn(tn,173.15f),0.05f),1.f);
    // Conversion outside the representable int range would be invalid.
    if (!si_finite(r) || r<=-2147483648.f || r>=2147483648.f) return false;
    int i=int(r); q.dr_dtn=__fdiv_rn(1.f,0.05f);
    if (i<1) {i=1; r=1.f; q.dr_dtn=0.f;}
    if (i>5000) {i=5000; r=5001.f; q.dr_dtn=0.f;}
    q.slot=i-1; q.weight=__fsub_rn(r,float(i)); return true;
}
__device__ __forceinline__ bool si_qsn_tl(float tn,float dtn,const float *table,
                                          const float *dtable,float &out) {
    SiQsnPoint q; if(!si_qsn_point(tn,q)||!si_finite(dtn))return false;
    float a=table[q.slot],b=table[q.slot+1],da=dtable[q.slot],db=dtable[q.slot+1];
    if(!si_finite(a)||!si_finite(b)||!si_finite(da)||!si_finite(db))return false;
    float dw=__fmul_rn(q.dr_dtn,dtn);
    out=__fadd_rn(__fadd_rn(__fmul_rn(__fsub_rn(db,da),q.weight),
                             __fmul_rn(__fsub_rn(b,a),dw)),da);
    return si_finite(out);
}
// The caller consumes bo and adds b0/b1 to its deterministic table workspace.
__device__ __forceinline__ bool si_qsn_ad(float tn,const float *table,float bo,
                                          int &slot,double &btn,double &b0,double &b1) {
    SiQsnPoint q; if(!si_qsn_point(tn,q)||!si_finite(bo))return false;
    float a=table[q.slot],b=table[q.slot+1]; if(!si_finite(a)||!si_finite(b))return false;
    slot=q.slot; double bd=double(bo)*double(q.weight);
    b0=double(bo)-bd; b1=bd;
    btn=double(bo)*double(__fsub_rn(b,a))*double(q.dr_dtn);
    return si_finite(float(btn))&&si_finite(float(b0))&&si_finite(float(b1));
}

// Compact per-call form for composing and testing ice_qsn. `slot` is the
// 1-based selected table index saved on the actual forward call; a changed
// index is refused rather than silently extrapolating a captured table pair.
__device__ __forceinline__ bool si_qsn_compact(float tn,float a,float b,int slot,
                                               SiQsnPoint &q) {
    return si_qsn_point(tn,q) && q.slot+1==slot && si_finite(a) && si_finite(b);
}
__device__ __forceinline__ bool si_qsn_compact_tl(float tn,float a,float b,int slot,
                         float dtn,float da,float db,float &dy) {
    SiQsnPoint q;if(!si_qsn_compact(tn,a,b,slot,q)||!si_finite(dtn)||
       !si_finite(da)||!si_finite(db))return false;
    float dw=__fmul_rn(q.dr_dtn,dtn);
    dy=__fadd_rn(__fadd_rn(__fmul_rn(__fsub_rn(db,da),q.weight),
                           __fmul_rn(__fsub_rn(b,a),dw)),da);
    return si_finite(dy);
}
__device__ __forceinline__ bool si_qsn_compact_ad(float tn,float a,float b,int slot,
                         float by,double &btn,double &ba,double &bb) {
    SiQsnPoint q;if(!si_qsn_compact(tn,a,b,slot,q)||!si_finite(by))return false;
    double w=double(by)*double(q.weight);ba=double(by)-w;bb=w;
    btn=double(by)*double(__fsub_rn(b,a))*double(q.dr_dtn);
    return si_finite(float(btn))&&si_finite(float(ba))&&si_finite(float(bb));
}
#if !defined(SI_COMPILE_ONE) || defined(SI_QSN_AUG)
extern "C" __global__ void seaice_ice_qsn__aug(const float *tn,const float *a,
 const float *b,const int *slot,int n,float *y,SeaiceFailure *fail) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;SiQsnPoint q;
    if(!si_qsn_compact(tn[c],a[c],b[c],slot[c],q)){si_fail(fail,c,1,c,4736);return;}
    float v=__fadd_rn(__fmul_rn(__fsub_rn(b[c],a[c]),q.weight),a[c]);
    if(!si_check(v,fail,c,5,c,4736))return;y[c]=v;
}
#endif

#if !defined(SI_COMPILE_ONE) || defined(SI_QSN_TL)
extern "C" __global__ void seaice_ice_qsn__tl(const float *tn,const float *a,
 const float *b,const int *slot,int n,const float *dtn,const float *da,const float *db,
 float *dy,SeaiceFailure *fail,unsigned int *branches) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
    float r=__fadd_rn(__fdiv_rn(__fsub_rn(tn[c],173.15f),.05f),1.f);
    if(!si_finite(r)||r<=-2147483648.f||r>=2147483648.f){si_fail(fail,c,1,c,4736);return;}
    int i=int(r);si_branch(branches,0,i>=1);if(i<1)i=1;si_branch(branches,1,i<=5000);
    float v;if(!si_qsn_compact_tl(tn[c],a[c],b[c],slot[c],dtn[c],da[c],db[c],v)){
       si_fail(fail,c,1,c,4736);return;}dy[c]=v;
}
#endif
#if !defined(SI_COMPILE_ONE) || defined(SI_QSN_AD)
extern "C" __global__ void seaice_ice_qsn__ad(const float *tn,const float *a,
 const float *b,const int *slot,int n,float *btn,float *ba,float *bb,
 float *by,SeaiceFailure *fail) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;double x,y,z;
    if(!si_qsn_compact_ad(tn[c],a[c],b[c],slot[c],by[c],x,y,z)){
       si_fail(fail,c,1,c,4736);return;}
    float rx=float(double(btn[c])+x),ry=float(double(ba[c])+y),rz=float(double(bb[c])+z);
    if(!si_check(rx,fail,c,1,c,4736)||!si_check(ry,fail,c,2,c,4736)||
       !si_check(rz,fail,c,3,c,4736))return;
    // The output is overwritten by the forward. Consume its separate seed
    // before accumulating into the input cotangents (COMMON-AD 74).
    by[c]=0.f;btn[c]=rx;ba[c]=ry;bb[c]=rz;
}
#endif

// ice_vilka's continuous output depends on the final integer index only. The
// parent records and validates every executed loop iteration before calling it.
struct SiVilkaPoint { float a,b,t1,f,den,rn,num,pp; };
// Replay the initial estimate and each actually executed iteration. visits[0]
// is the first table index; each later element is the index at loop entry.
// The caller records the list in forward order and rejects a different path.
__device__ __forceinline__ bool si_vilka_replay(float tn,float d1,float d2,
        const float *table,const int *visits,int nvisits,int &final_i) {
    if(!si_finite(tn)||!si_finite(d1)||!si_finite(d2)||nvisits<2)return false;
    float r=__fadd_rn(__fdiv_rn(__fsub_rn(tn,1.7315e2f),.05f),1.f);
    if(!si_finite(r)||r<=-2147483648.f||r>=2147483648.f)return false;
    int i=int(r);if(i>5000||i<1||visits[0]!=i)return false;
    float a=table[i-1],b=table[i];
    if(!si_finite(a)||!si_finite(b))return false;
    float t1=__fadd_rn(173.1f,__fmul_rn(float(i),.05f));
    float f=__fsub_rn(__fadd_rn(t1,__fmul_rn(d1,a)),d2);
    float den=__fadd_rn(.05f,__fmul_rn(d1,__fsub_rn(b,a)));
    if(!si_finite(f)||!si_finite(den)||den==0.f)return false;
    float proposal=__fsub_rn(float(i),__fdiv_rn(f,den));
    if(!si_finite(proposal)||proposal<=-2147483648.f||proposal>=2147483648.f)return false;
    i=int(proposal);if(i>5000||i<1)return false;
    for(int step=1;step<nvisits;++step) {
        if(i!=visits[step])return false;
        a=table[i-1];b=table[i];if(!si_finite(a)||!si_finite(b))return false;
        t1=__fadd_rn(173.1f,__fmul_rn(float(i),.05f));
        f=__fsub_rn(__fadd_rn(t1,__fmul_rn(d1,a)),d2);
        den=__fadd_rn(.05f,__fmul_rn(d1,__fsub_rn(b,a)));
        if(!si_finite(f)||!si_finite(den)||den==0.f)return false;
        float rn=__fdiv_rn(f,den);
        if(!si_finite(rn)||rn<=-2147483648.f||rn>=2147483648.f)return false;
        int change=int(rn);long long next=(long long)i-(long long)change;
        if(next>5000||next<1)return false;
        if(next==i){if(step!=nvisits-1)return false;final_i=i;return true;}
        i=int(next);
    }
    return false;
}
__device__ __forceinline__ bool si_vilka_point(int i,float d1,float d2,float pp,
                                               const float *table,SiVilkaPoint &q) {
    if(i>5000||i<1||!si_finite(d1)||!si_finite(d2)||!si_finite(pp)||pp==0.f)return false;
    q.a=table[i-1];q.b=table[i];q.pp=pp;
    if(!si_finite(q.a)||!si_finite(q.b))return false;
    q.t1=__fadd_rn(173.1f,__fmul_rn(float(i),.05f));
    q.f=__fsub_rn(__fadd_rn(q.t1,__fmul_rn(d1,q.a)),d2);
    q.den=__fadd_rn(.05f,__fmul_rn(d1,__fsub_rn(q.b,q.a)));
    if(!si_finite(q.den)||q.den==0.f)return false;
    q.rn=__fdiv_rn(q.f,q.den);
    q.num=__fadd_rn(q.a,__fmul_rn(__fsub_rn(q.a,q.b),q.rn));
    return si_finite(q.f)&&si_finite(q.rn)&&si_finite(q.num);
}
__device__ __forceinline__ bool si_vilka_tl(int i,float d1,float d2,float pp,
        const float *table,float dd1,float dd2,float dpp,float da,float db,
        float &dqs,float &dts) {
    SiVilkaPoint q;if(!si_vilka_point(i,d1,d2,pp,table,q))return false;
    if(!si_finite(dd1)||!si_finite(dd2)||!si_finite(dpp)||!si_finite(da)||!si_finite(db))return false;
    float df=__fsub_rn(__fadd_rn(__fmul_rn(dd1,q.a),__fmul_rn(d1,da)),dd2);
    float dden=__fadd_rn(__fmul_rn(dd1,__fsub_rn(q.b,q.a)),__fmul_rn(d1,__fsub_rn(db,da)));
    float drn=__fdiv_rn(__fsub_rn(df,__fmul_rn(q.rn,dden)),q.den);
    float dnum=__fadd_rn(da,__fadd_rn(__fmul_rn(__fsub_rn(da,db),q.rn),__fmul_rn(__fsub_rn(q.a,q.b),drn)));
    dts=__fmul_rn(-.05f,drn);
    dqs=__fdiv_rn(__fsub_rn(dnum,__fmul_rn(__fdiv_rn(q.num,pp),dpp)),pp);
    return si_finite(dts)&&si_finite(dqs);
}
__device__ __forceinline__ bool si_vilka_ad(int i,float d1,float d2,float pp,
        const float *table,float bqs,float bts,double &bd1,double &bd2,double &bpp,
        double &ba,double &bb) {
    SiVilkaPoint q;if(!si_vilka_point(i,d1,d2,pp,table,q)||!si_finite(bqs)||!si_finite(bts))return false;
    double bnum=double(bqs)/double(pp);
    bpp=-bnum*double(__fdiv_rn(q.num,pp));
    double brn=double(-.05f)*double(bts)+bnum*double(__fsub_rn(q.a,q.b));
    ba=bnum+bnum*double(q.rn); bb=-bnum*double(q.rn);
    double bf=brn/double(q.den), bden=-bf*double(q.rn);
    bd1=bf*double(q.a)+bden*double(__fsub_rn(q.b,q.a));
    bd2=-bf;
    ba+=bf*double(d1)-bden*double(d1);
    bb+=bden*double(d1);
    return si_finite(float(bd1))&&si_finite(float(bd2))&&si_finite(float(bpp))&&
           si_finite(float(ba))&&si_finite(float(bb));
}
__device__ __forceinline__ bool si_vilka_executed_tl(float tn,float d1,float d2,float pp,
        const float *table,const int *visits,int nvisits,float dtn,float dd1,float dd2,
        float dpp,float da,float db,float &dqs,float &dts) {
    int i;if(!si_finite(dtn)||!si_vilka_replay(tn,d1,d2,table,visits,nvisits,i))return false;
    return si_vilka_tl(i,d1,d2,pp,table,dd1,dd2,dpp,da,db,dqs,dts);
}
__device__ __forceinline__ bool si_vilka_executed_ad(float tn,float d1,float d2,float pp,
        const float *table,const int *visits,int nvisits,float bqs,float bts,
        double &bd1,double &bd2,double &bpp,double &ba,double &bb) {
    int i;if(!si_vilka_replay(tn,d1,d2,table,visits,nvisits,i))return false;
    // tn and intermediate selected entries influence integer choices only;
    // on a fixed executed path their cotangent contribution is zero.
    return si_vilka_ad(i,d1,d2,pp,table,bqs,bts,bd1,bd2,bpp,ba,bb);
}

// Standalone per-call replay. `visits` is row-major [n,stride], including the
// initial estimate followed by each loop-entry index, exactly as executed.
// Table cotangents are returned per call for a later deterministic scatter.
#if !defined(SI_COMPILE_ONE) || defined(SI_VILKA_AUG)
extern "C" __global__ void seaice_ice_vilka__aug(const float *tn,const float *d1,
 const float *d2,const float *pp,const float *table,const int *visits,
 const int *nvisits,int stride,int n,float *qs,float *ts,SeaiceFailure *fail) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
 if(stride<2||nvisits[c]<2||nvisits[c]>stride){si_fail(fail,c,1,c,4711);return;}
 int i;SiVilkaPoint q;
 if(!si_vilka_replay(tn[c],d1[c],d2[c],table,visits+(unsigned long long)c*stride,nvisits[c],i)||
    !si_vilka_point(i,d1[c],d2[c],pp[c],table,q)){
   si_fail(fail,c,1,c,4711);return;}
 float x=__fdiv_rn(q.num,pp[c]),y=__fsub_rn(q.t1,__fmul_rn(.05f,q.rn));
 if(!si_check(x,fail,c,1,c,4726)||!si_check(y,fail,c,2,c,4725))return;
 qs[c]=x;ts[c]=y;
}
#endif
#if !defined(SI_COMPILE_ONE) || defined(SI_VILKA_TL)
extern "C" __global__ void seaice_ice_vilka__tl(const float *tn,const float *d1,
 const float *d2,const float *pp,const float *table,const int *visits,
 const int *nvisits,int stride,int n,const float *vtn,const float *vd1,const float *vd2,
 const float *vpp,const float *va,const float *vb,float *vqs,float *vts,
 SeaiceFailure *fail,unsigned int *branches) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;float x,y;
 if(stride<2||nvisits[c]<2||nvisits[c]>stride||!si_vilka_executed_tl(tn[c],d1[c],d2[c],pp[c],
    table,visits+(unsigned long long)c*stride,nvisits[c],vtn[c],vd1[c],vd2[c],vpp[c],va[c],vb[c],x,y)){
   si_fail(fail,c,1,c,4711);return;}
 // A successful replay has taken the in-range arm after the initial guess
 // and after each loop body. The continuation comparison is true for all
 // loop visits except the final one. Preserve short-circuit evaluation.
 si_branch(branches,0,false);si_branch(branches,1,false);si_branch(branches,2,false);
 for(int step=1;step<nvisits[c];++step) {
   si_branch(branches,3,false);si_branch(branches,4,false);si_branch(branches,5,false);
   si_branch(branches,6,step!=nvisits[c]-1);
 }
 vqs[c]=x;vts[c]=y;
}
#endif
#if !defined(SI_COMPILE_ONE) || defined(SI_VILKA_AD)
extern "C" __global__ void seaice_ice_vilka__ad(const float *tn,const float *d1,
 const float *d2,const float *pp,const float *table,const int *visits,
 const int *nvisits,int stride,int n,float *bqs,float *bts,float *btn,float *bd1,
 float *bd2,float *bpp,float *ba,float *bb,SeaiceFailure *fail) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;double x,y,z,a,b;
 if(stride<2||nvisits[c]<2||nvisits[c]>stride||!si_vilka_executed_ad(tn[c],d1[c],d2[c],pp[c],
    table,visits+(unsigned long long)c*stride,nvisits[c],bqs[c],bts[c],x,y,z,a,b)){
   si_fail(fail,c,1,c,4711);return;}
 float rx=float(double(bd1[c])+x),ry=float(double(bd2[c])+y),
       rz=float(double(bpp[c])+z),ra=float(double(ba[c])+a),rb=float(double(bb[c])+b);
 if(!si_check(btn[c],fail,c,1,c,4711)||!si_check(rx,fail,c,2,c,4726)||
    !si_check(ry,fail,c,3,c,4726)||!si_check(rz,fail,c,4,c,4726)||
    !si_check(ra,fail,c,5,c,4726)||!si_check(rb,fail,c,6,c,4726))return;
 bqs[c]=0.f;bts[c]=0.f;
 bd1[c]=rx;bd2[c]=ry;bpp[c]=rz;ba[c]=ra;bb[c]=rb;
}
#endif

// One layer of ice_icetmp's material-property update (pinned lines 3839-44).
// The caller owns the surrounding executed nzs loop and reverse ordering.
struct SiIceMaterial { float temp,den,rho,heat,capacity,diffusivity; };
__device__ __forceinline__ bool si_ice_material(float ts,SiIceMaterial &v) {
    if(!si_finite(ts))return false;
    v.temp=__fsub_rn(ts,273.15f);
    v.den=__fsub_rn(1.f,__fmul_rn(0.000165f,v.temp));
    if(!si_finite(v.den)||v.den==0.f)return false;
    v.rho=__fdiv_rn(917.6f,v.den);
    v.heat=__fadd_rn(2115.85f,__fmul_rn(7.7948f,v.temp));
    v.capacity=__fmul_rn(v.heat,v.rho);
    if(!si_finite(v.capacity)||v.capacity==0.f)return false;
    v.diffusivity=__fdiv_rn(2.260872f,v.capacity);
    return si_finite(v.rho)&&si_finite(v.heat)&&si_finite(v.diffusivity);
}
__device__ __forceinline__ bool si_ice_material_tl(float ts,float dts,
        float &dtemp,float &drho,float &dcapacity,float &ddiffusivity) {
    SiIceMaterial v;if(!si_ice_material(ts,v)||!si_finite(dts))return false;
    dtemp=dts;
    float dden=__fmul_rn(-0.000165f,dts);
    drho=__fdiv_rn(__fmul_rn(-v.rho,dden),v.den);
    float dheat=__fmul_rn(7.7948f,dts);
    dcapacity=__fadd_rn(__fmul_rn(dheat,v.rho),__fmul_rn(v.heat,drho));
    ddiffusivity=__fdiv_rn(__fmul_rn(-v.diffusivity,dcapacity),v.capacity);
    return si_finite(drho)&&si_finite(dcapacity)&&si_finite(ddiffusivity);
}
__device__ __forceinline__ bool si_ice_material_ad(float ts,float btemp,float brho,
        float bcapacity,float bdiffusivity,float &bts) {
    SiIceMaterial v;if(!si_ice_material(ts,v)||!si_finite(btemp)||!si_finite(brho)||
        !si_finite(bcapacity)||!si_finite(bdiffusivity))return false;
    double bc=double(bcapacity)-double(bdiffusivity)*double(v.diffusivity)/double(v.capacity);
    double br=double(brho)+bc*double(v.heat), bh=bc*double(v.rho);
    double bd=-br*double(v.rho)/double(v.den);
    bts=float(double(btemp)+double(7.7948f)*bh+double(-0.000165f)*bd);
    return si_finite(bts);
}

// One or more fields of the snow-mosaic blend in ice_icetmp, lines 3964-3979
// and 3982-3983. The parent records the snow_mosaic branch and handles the
// separate albedo/emissivity selections and the sublim=eeta assignment.
// Bare and snowy are the saved pre-blend values; the parent supplies one
// element for each scalar or temperature layer and retains all contributions
// to bsnowfrac in double until the complete call has been reversed.
__device__ __forceinline__ bool si_ice_mosaic_tl(
        int n,float snowfrac,float dsnowfrac,const float *bare,const float *snowy,
        const float *dbare,const float *dsnowy,float *out,float *dout) {
    if(n<0||!si_finite(snowfrac)||!si_finite(dsnowfrac))return false;
    float w=__fsub_rn(1.f,snowfrac);
    for(int k=0;k<n;++k) {
        if(!si_finite(bare[k])||!si_finite(snowy[k])||!si_finite(dbare[k])||
           !si_finite(dsnowy[k]))return false;
        out[k]=__fadd_rn(__fmul_rn(bare[k],w),__fmul_rn(snowy[k],snowfrac));
        dout[k]=float(double(w)*double(dbare[k])+double(snowfrac)*double(dsnowy[k])+
                      (double(snowy[k])-double(bare[k]))*double(dsnowfrac));
        if(!si_finite(out[k])||!si_finite(dout[k]))return false;
    }
    return true;
}
__device__ __forceinline__ bool si_ice_mosaic_ad(
        int n,float snowfrac,const float *bare,const float *snowy,const double *bout,
        double *bbare,double *bsnowy,double &bsnowfrac) {
    if(n<0||!si_finite(snowfrac))return false;
    float w=__fsub_rn(1.f,snowfrac);
    for(int k=n-1;k>=0;--k) {
        if(!si_finite(bare[k])||!si_finite(snowy[k]))return false;
        bbare[k]+=double(w)*bout[k];bsnowy[k]+=double(snowfrac)*bout[k];
        bsnowfrac+=(double(snowy[k])-double(bare[k]))*bout[k];
    }
    return true;
}

// ice_sice lines 4093-4104: the forward elimination of the layer system.
// This is a compose building block, not an implementation of ice_sice. Arrays
// are the zero-based views of the one-based Fortran arrays. The parent keeps
// the complete pre-call tso and records c and r, including the initial slot.
// The adjoint's double work arrays must survive until all uses of the same
// input have been reversed and are rounded exactly once by the caller.
__device__ __forceinline__ bool si_sice_eliminate_tl(
        int nzs,const float *tso,const float *diff,const float *dz,
        const float *dtso,const float *ddiff,const float *ddz,
        float *c,float *r,float *dc,float *dr) {
    if(nzs<3)return false;
    c[0]=0.f;r[0]=tso[nzs-1];dc[0]=0.f;dr[0]=dtso[nzs-1];
    for(int k=1;k<=nzs-2;++k) {
        int kn=nzs-k, t=kn-1, a=2*kn-4, b=a+1;
        float x1=__fmul_rn(dz[a],diff[t-1]);
        float x2=__fmul_rn(dz[b],diff[t]);
        float dx1=__fadd_rn(__fmul_rn(ddz[a],diff[t-1]),__fmul_rn(dz[a],ddiff[t-1]));
        float dx2=__fadd_rn(__fmul_rn(ddz[b],diff[t]),__fmul_rn(dz[b],ddiff[t]));
        float dt1=__fsub_rn(tso[t-1],tso[t]),dt2=__fsub_rn(tso[t],tso[t+1]);
        float ft=__fsub_rn(__fadd_rn(tso[t],__fmul_rn(x1,dt1)),__fmul_rn(x2,dt2));
        float dft=__fsub_rn(__fadd_rn(dtso[t],__fadd_rn(__fmul_rn(dx1,dt1),
                    __fmul_rn(x1,__fsub_rn(dtso[t-1],dtso[t])))),
                    __fadd_rn(__fmul_rn(dx2,dt2),__fmul_rn(x2,__fsub_rn(dtso[t],dtso[t+1]))));
        float den=__fsub_rn(__fadd_rn(__fadd_rn(1.f,x1),x2),__fmul_rn(x2,c[k-1]));
        float dden=__fsub_rn(__fadd_rn(dx1,dx2),
                     __fadd_rn(__fmul_rn(dx2,c[k-1]),__fmul_rn(x2,dc[k-1])));
        if(!si_finite(den)||den==0.f)return false;
        c[k]=__fdiv_rn(x1,den);
        r[k]=__fdiv_rn(__fadd_rn(ft,__fmul_rn(x2,r[k-1])),den);
        dc[k]=__fdiv_rn(__fsub_rn(dx1,__fmul_rn(c[k],dden)),den);
        dr[k]=__fdiv_rn(__fsub_rn(__fadd_rn(dft,__fadd_rn(__fmul_rn(dx2,r[k-1]),
                         __fmul_rn(x2,dr[k-1]))),__fmul_rn(r[k],dden)),den);
        if(!si_finite(c[k])||!si_finite(r[k])||!si_finite(dc[k])||!si_finite(dr[k]))return false;
    }
    return true;
}

__device__ __forceinline__ bool si_sice_eliminate_ad(
        int nzs,const float *tso,const float *diff,const float *dz,
        const float *c,const float *r,double *btso,double *bdiff,double *bdz,
        double *bc,double *br) {
    if(nzs<3)return false;
    for(int k=nzs-2;k>=1;--k) {
        int kn=nzs-k, t=kn-1, a=2*kn-4, b=a+1;
        float x1=__fmul_rn(dz[a],diff[t-1]),x2=__fmul_rn(dz[b],diff[t]);
        float den=__fsub_rn(__fadd_rn(__fadd_rn(1.f,x1),x2),__fmul_rn(x2,c[k-1]));
        if(!si_finite(den)||den==0.f)return false;
        double ac=bc[k], ar=br[k];bc[k]=0.;br[k]=0.;
        double bf=ar/double(den);
        double bd=-ac*double(c[k])/double(den)-ar*double(r[k])/double(den);
        double bx1=ac/double(den)+bd+bf*double(__fsub_rn(tso[t-1],tso[t]));
        double bx2=bd-bd*double(c[k-1])+bf*double(r[k-1])-
                   bf*double(__fsub_rn(tso[t],tso[t+1]));
        bc[k-1]-=bd*double(x2);br[k-1]+=bf*double(x2);
        btso[t-1]+=bf*double(x1);
        btso[t]+=bf;btso[t]-=bf*double(x1);btso[t]-=bf*double(x2);
        btso[t+1]+=bf*double(x2);
        bdz[a]+=bx1*double(diff[t-1]);bdiff[t-1]+=bx1*double(dz[a]);
        bdz[b]+=bx2*double(diff[t]);bdiff[t]+=bx2*double(dz[b]);
    }
    btso[nzs-1]+=br[0];br[0]=0.;bc[0]=0.;
    return true;
}

// ice_sice lines 4112-4127. This hand-written expression graph preserves the
// forward's float operations and their grouping; the derivative sweep uses
// double precision so repeated contributions are added before a final round.
// It is a building block, not an entry point or a complete ice_sice pair.
// Input slots: thdifice(1), capice(1), dzstop, tkms, cp, rho, conflx,
// delt, emiss, stbolt, soilt, d1, d2, rnet, rainf, cw, prcpms, qkms,
// tabs, qvatm, qvg, patm. Output slots: d9, r211, aa1, bb, pp.
enum { SI_SC_NINPUT=22, SI_SC_NOUTPUT=5, SI_SC_NNODE=128 };
struct SiSiceClosureTape {
    float v[SI_SC_NNODE];
    double da[SI_SC_NNODE],db[SI_SC_NNODE];
    int a[SI_SC_NNODE],b[SI_SC_NNODE],out[SI_SC_NOUTPUT],n;
    bool valid, tabs_arm;
};
__device__ __forceinline__ int si_sc_leaf(SiSiceClosureTape &t,float v) {
    if(t.n>=SI_SC_NNODE||!si_finite(v)){t.valid=false;return 0;}
    int k=t.n++;t.v[k]=v;t.a[k]=t.b[k]=-1;t.da[k]=t.db[k]=0.;return k;
}
__device__ __forceinline__ int si_sc_op(SiSiceClosureTape &t,int a,int b,int op) {
    if(!t.valid||t.n>=SI_SC_NNODE){t.valid=false;return 0;}
    float x=t.v[a],y=t.v[b],v;double da,db;
    if(op==0){v=__fadd_rn(x,y);da=1.;db=1.;}
    else if(op==1){v=__fsub_rn(x,y);da=1.;db=-1.;}
    else if(op==2){v=__fmul_rn(x,y);da=double(y);db=double(x);}
    else {if(y==0.f){t.valid=false;return 0;}v=__fdiv_rn(x,y);
          da=1./double(y);db=-(double(x)/double(y))/double(y);}
    if(!si_finite(v)){t.valid=false;return 0;}
    int k=t.n++;t.v[k]=v;t.a[k]=a;t.b[k]=b;t.da[k]=da;t.db[k]=db;return k;
}
__device__ __forceinline__ bool si_sice_closure_record(
        const float *x,SiSiceClosureTape &t) {
    t.n=0;t.valid=true;
    for(int i=0;i<SI_SC_NINPUT;++i)si_sc_leaf(t,x[i]);
    const int half=si_sc_leaf(t,.5f),one=si_sc_leaf(t,1.f),
              xls=si_sc_leaf(t,2.85e6f),freeze=si_sc_leaf(t,273.15f),
              kpa=si_sc_leaf(t,1.e3f);
    // Each named operation follows the corresponding intrinsic in the pin.
    int d9=si_sc_op(t,si_sc_op(t,0,1,2),2,2);
    int d10=si_sc_op(t,si_sc_op(t,3,4,2),5,2);
    int r211=si_sc_op(t,si_sc_op(t,half,6,2),7,3);
    int r21=si_sc_op(t,si_sc_op(t,r211,4,2),5,2);
    int dz2=si_sc_op(t,2,2,2);
    int r22=si_sc_op(t,half,si_sc_op(t,si_sc_op(t,0,7,2),dz2,2),3);
    int tn2=si_sc_op(t,10,10,2);
    int r6=si_sc_op(t,si_sc_op(t,si_sc_op(t,8,9,2),half,2),
                    si_sc_op(t,tn2,tn2,2),2);
    int r7=si_sc_op(t,r6,10,3),d11=si_sc_op(t,13,r6,0);
    int rain=si_sc_op(t,si_sc_op(t,14,15,2),16,2);
    int part=si_sc_op(t,si_sc_op(t,one,11,1),r22,0);
    int den=si_sc_op(t,si_sc_op(t,si_sc_op(t,si_sc_op(t,
             si_sc_op(t,d9,part,2),d10,0),r21,0),r7,0),rain,0);
    int fkq=si_sc_op(t,17,5,2),r210=si_sc_op(t,r211,5,2);
    int aa=si_sc_op(t,si_sc_op(t,xls,si_sc_op(t,fkq,r210,0),2),den,3);
    int n0=si_sc_op(t,d10,18,2),n1=si_sc_op(t,r21,10,2);
    int n2=si_sc_op(t,xls,si_sc_op(t,si_sc_op(t,19,fkq,2),
                                      si_sc_op(t,r210,20,2),0),2);
    int n3=si_sc_op(t,d9,si_sc_op(t,12,si_sc_op(t,r22,10,2),0),2);
    // ruc_max(a,b) selects b when equal, including at exactly 273.15 K.
    t.tabs_arm=!(273.15f>x[18]);int warm=t.tabs_arm?18:freeze;
    int n4=si_sc_op(t,rain,warm,2);
    int num=si_sc_op(t,si_sc_op(t,si_sc_op(t,si_sc_op(t,
              si_sc_op(t,n0,n1,0),n2,0),d11,0),n3,0),n4,0);
    int bb=si_sc_op(t,num,den,3),pp=si_sc_op(t,21,kpa,2);
    int aa1=si_sc_op(t,aa,pp,3);
    t.out[0]=d9;t.out[1]=r211;t.out[2]=aa1;t.out[3]=bb;t.out[4]=pp;
    return t.valid;
}
__device__ __forceinline__ bool si_sice_closure_tl(
        const SiSiceClosureTape &t,const double *dx,double *dy) {
    if(!t.valid)return false;double d[SI_SC_NNODE];
    for(int i=0;i<t.n;++i) {
        if(i<SI_SC_NINPUT)d[i]=dx[i];
        else if(t.a[i]<0)d[i]=0.;
        else d[i]=t.da[i]*d[t.a[i]]+t.db[i]*d[t.b[i]];
    }
    for(int i=0;i<SI_SC_NOUTPUT;++i)dy[i]=d[t.out[i]];
    return true;
}
__device__ __forceinline__ bool si_sice_closure_ad(
        const SiSiceClosureTape &t,const double *by,double *bx) {
    if(!t.valid)return false;double b[SI_SC_NNODE];
    for(int i=0;i<t.n;++i)b[i]=0.;
    for(int i=0;i<SI_SC_NOUTPUT;++i)b[t.out[i]]+=by[i];
    for(int i=t.n-1;i>=SI_SC_NINPUT;--i)if(t.a[i]>=0) {
        b[t.a[i]]+=t.da[i]*b[i];b[t.b[i]]+=t.db[i]*b[i];
    }
    for(int i=0;i<SI_SC_NINPUT;++i)bx[i]+=b[i];
    return true;
}

// ice_sice lines 4147-4193, excluding diagnostics. `pow_value` is the exact
// value returned by the pinned gfk_pow call at line 4152; it is an active
// input, not a frozen coefficient. The caller must replay that call and chain
// its cotangent to both patm and rovcp. The two q1 arms and the MYJ arm are
// selected from the executed primal, and the max tie selects tabs.
// Inputs: tn,soilt,tabs,stbolt,emiss,glw,tkms,cp,rho,qkms,qvatm,qsg,qvg,
// rnet,thdifice(1),capice(1),dzstop,tso(1),tso(2),r211,delt,zsmain(2),
// rainf,cw,prcpms,qgold,pow_value. Outputs: dew,eeta,qfx,hfx,s,evapl,prcpl,
// fltot. The caller must merge duplicate inputs before rounding cotangents.
enum { SI_SF_NINPUT=27, SI_SF_NOUTPUT=8 };
struct SiSiceFluxTape {
    SiSiceClosureTape g;
    int out[SI_SF_NOUTPUT];
    bool nonpositive,myj,warm;
};
__device__ __forceinline__ int si_sc_neg(SiSiceClosureTape &t,int a) {
    if(!t.valid||t.n>=SI_SC_NNODE){t.valid=false;return 0;}
    int k=t.n++;t.v[k]=-t.v[a];t.a[k]=t.b[k]=a;t.da[k]=-1.;t.db[k]=0.;return k;
}
__device__ __forceinline__ bool si_sice_flux_record(
        const float *x,bool myj,SiSiceFluxTape &s) {
    SiSiceClosureTape &t=s.g;t.n=0;t.valid=true;s.myj=myj;
    for(int i=0;i<SI_SF_NINPUT;++i)si_sc_leaf(t,x[i]);
    const int zero=si_sc_leaf(t,0.f),one=si_sc_leaf(t,1.f),
              half=si_sc_leaf(t,.5f),thousand=si_sc_leaf(t,1.e3f),
              milli=si_sc_leaf(t,1.e-3f),xls=si_sc_leaf(t,2.85e6f),
              freeze=si_sc_leaf(t,273.15f);
    int hbase=si_sc_op(t,si_sc_op(t,si_sc_op(t,si_sc_neg(t,6),7,2),8,2),
                       si_sc_op(t,2,1,1),2);
    int hfx=si_sc_op(t,hbase,26,2);
    int ras=si_sc_op(t,8,milli,2);
    int qbase=si_sc_op(t,si_sc_neg(t,9),ras,2);
    int q1=si_sc_op(t,qbase,si_sc_op(t,10,11,1),2);
    s.nonpositive=t.valid && t.v[q1]<=0.f;
    int dew=zero,eeta1,eeta;
    if(s.nonpositive) {
        if(myj) {
            int aq=si_sc_op(t,10,si_sc_op(t,one,10,0),3);
            int bq=si_sc_op(t,11,si_sc_op(t,one,11,0),3);
            eeta1=si_sc_op(t,si_sc_op(t,qbase,si_sc_op(t,aq,bq,1),2),thousand,2);
        } else {
            dew=si_sc_op(t,9,si_sc_op(t,10,11,1),2);
            eeta1=si_sc_op(t,si_sc_neg(t,8),dew,2);
        }
        eeta=si_sc_op(t,si_sc_neg(t,8),dew,2);
    } else {
        if(myj) {
            int aq=si_sc_op(t,10,si_sc_op(t,one,10,0),3);
            int bq=si_sc_op(t,12,si_sc_op(t,one,12,0),3);
            eeta1=si_sc_op(t,si_sc_op(t,qbase,si_sc_op(t,aq,bq,1),2),thousand,2);
        } else eeta1=si_sc_op(t,q1,thousand,2);
        eeta=si_sc_op(t,q1,thousand,2);
    }
    int qfx=si_sc_op(t,xls,eeta1,2);
    int ground=si_sc_op(t,si_sc_op(t,si_sc_op(t,14,15,2),16,2),
                        si_sc_op(t,17,18,1),2);
    int heat=si_sc_op(t,si_sc_op(t,si_sc_op(t,7,8,2),19,2),
            si_sc_op(t,si_sc_op(t,si_sc_op(t,15,21,2),half,2),20,3),0);
    int change=si_sc_op(t,heat,si_sc_op(t,1,0,1),2);
    int moisture=si_sc_op(t,si_sc_op(t,si_sc_op(t,xls,8,2),19,2),
                          si_sc_op(t,11,25,1),2);
    int xsum=si_sc_op(t,change,moisture,0);
    s.warm=!(273.15f>x[2]);int warm=s.warm?2:freeze;
    int rain=si_sc_op(t,si_sc_op(t,22,23,2),24,2);
    int flux=si_sc_op(t,xsum,si_sc_op(t,rain,si_sc_op(t,warm,1,1),2),1);
    int net=si_sc_op(t,si_sc_op(t,si_sc_op(t,si_sc_op(t,13,
              si_sc_op(t,xls,eeta,2),1),hbase,1),ground,1),flux,1);
    int fltot=si_sc_op(t,net,net,1);
    s.out[0]=dew;s.out[1]=eeta;s.out[2]=qfx;s.out[3]=hfx;
    s.out[4]=ground;s.out[5]=eeta;s.out[6]=24;s.out[7]=fltot;
    return t.valid;
}
__device__ __forceinline__ bool si_sice_flux_tl(
        const SiSiceFluxTape &s,const double *dx,double *dy) {
    const SiSiceClosureTape &t=s.g;if(!t.valid)return false;double d[SI_SC_NNODE];
    for(int i=0;i<t.n;++i) {
        if(i<SI_SF_NINPUT)d[i]=dx[i];
        else if(t.a[i]<0)d[i]=0.;
        else d[i]=t.da[i]*d[t.a[i]]+t.db[i]*d[t.b[i]];
    }
    for(int i=0;i<SI_SF_NOUTPUT;++i)dy[i]=d[s.out[i]];
    return true;
}
__device__ __forceinline__ bool si_sice_flux_ad(
        const SiSiceFluxTape &s,const double *by,double *bx) {
    const SiSiceClosureTape &t=s.g;if(!t.valid)return false;double b[SI_SC_NNODE];
    for(int i=0;i<t.n;++i)b[i]=0.;
    for(int i=0;i<SI_SF_NOUTPUT;++i)b[s.out[i]]+=by[i];
    for(int i=t.n-1;i>=SI_SF_NINPUT;--i)if(t.a[i]>=0) {
        b[t.a[i]]+=t.da[i]*b[i];b[t.b[i]]+=t.db[i]*b[i];
    }
    for(int i=0;i<SI_SF_NINPUT;++i)bx[i]+=b[i];
    return true;
}

// Vertical-coordinate construction in ice_ruc_ice, lines 3648-3666.
// The first entries of zmain/zhalf are overwritten with zero. The remaining
// entries depend on zs and the two dtdzs entries per interior level depend
// on the time step and both adjacent layer spacings. All arrays use C indices.
struct SiIceGeoTape {
    int n;
    float dt;
    const float *zs;
    float *zmain,*zhalf,*x,*dtdzs,*dtdzs2;
};
__device__ __forceinline__ bool si_ice_geo_record(SiIceGeoTape &t) {
    if(t.n<3)return false;
    t.zmain[0]=0.f;t.zhalf[0]=0.f;
    for(int k=1;k<t.n;++k) {
        t.zmain[k]=t.zs[k];
        t.zhalf[k]=__fmul_rn(.5f,__fadd_rn(t.zmain[k-1],t.zmain[k]));
    }
    for(int k=1;k<t.n-1;++k) {
        float dh=__fsub_rn(t.zhalf[k+1],t.zhalf[k]);
        float dl=__fsub_rn(t.zmain[k],t.zmain[k-1]);
        float dr=__fsub_rn(t.zmain[k+1],t.zmain[k]);
        if(dh==0.f||dl==0.f||dr==0.f)return false;
        float x=__fdiv_rn(__fdiv_rn(t.dt,2.f),dh);
        t.x[k-1]=x;t.dtdzs[2*(k-1)]=__fdiv_rn(x,dl);
        t.dtdzs2[k-1]=x;t.dtdzs[2*(k-1)+1]=__fdiv_rn(x,dr);
        if(!si_finite(x)||!si_finite(t.dtdzs[2*(k-1)])||
           !si_finite(t.dtdzs[2*(k-1)+1]))return false;
    }
    return true;
}
__device__ __forceinline__ bool si_ice_geo_tl(
        const SiIceGeoTape &t,double ddt,const double *dzs,
        double *dmain,double *dhalf,double *ddtdzs,double *ddtdzs2) {
    if(t.n<3)return false;
    dmain[0]=0.;dhalf[0]=0.;
    for(int k=1;k<t.n;++k) {
        dmain[k]=dzs[k];dhalf[k]=.5*(dmain[k-1]+dmain[k]);
    }
    for(int k=1;k<t.n-1;++k) {
        float dh=__fsub_rn(t.zhalf[k+1],t.zhalf[k]);
        float dl=__fsub_rn(t.zmain[k],t.zmain[k-1]);
        float dr=__fsub_rn(t.zmain[k+1],t.zmain[k]);
        if(dh==0.f||dl==0.f||dr==0.f)return false;
        double dx=(.5*ddt-double(t.x[k-1])*(dhalf[k+1]-dhalf[k]))/double(dh);
        int a=2*(k-1),b=a+1;
        ddtdzs[a]=(dx-double(t.dtdzs[a])*(dmain[k]-dmain[k-1]))/double(dl);
        ddtdzs[b]=(dx-double(t.dtdzs[b])*(dmain[k+1]-dmain[k]))/double(dr);
        ddtdzs2[k-1]=dx;
    }
    return true;
}
// bmain and bhalf contain the cotangents of the explicit coordinate outputs
// on entry and serve as caller-owned reverse scratch. Output cotangents of
// the overwritten coordinate and spacing slots are consumed and zeroed.
__device__ __forceinline__ bool si_ice_geo_ad(
        const SiIceGeoTape &t,double &bdt,double *bzs,
        double *bmain,double *bhalf,double *bdtdzs,double *bdtdzs2) {
    if(t.n<3)return false;
    for(int k=t.n-2;k>=1;--k) {
        float dh=__fsub_rn(t.zhalf[k+1],t.zhalf[k]);
        float dl=__fsub_rn(t.zmain[k],t.zmain[k-1]);
        float dr=__fsub_rn(t.zmain[k+1],t.zmain[k]);
        if(dh==0.f||dl==0.f||dr==0.f)return false;
        int a=2*(k-1),b=a+1;
        double bx=bdtdzs2[k-1]+bdtdzs[a]/double(dl)+bdtdzs[b]/double(dr);
        double bl=-bdtdzs[a]*double(t.dtdzs[a])/double(dl);
        double br=-bdtdzs[b]*double(t.dtdzs[b])/double(dr);
        double bh=-bx*double(t.x[k-1])/double(dh);
        bdt+=.5*bx/double(dh);
        bmain[k]+=bl-br;bmain[k-1]-=bl;bmain[k+1]+=br;
        bhalf[k+1]+=bh;bhalf[k]-=bh;
        bdtdzs[a]=0.;bdtdzs[b]=0.;bdtdzs2[k-1]=0.;
    }
    for(int k=t.n-1;k>=1;--k) {
        double b=.5*bhalf[k];bhalf[k]=0.;bmain[k]+=b;bmain[k-1]+=b;
    }
    for(int k=1;k<t.n;++k){bzs[k]+=bmain[k];bmain[k]=0.;}
    bmain[0]=0.;bhalf[0]=0.;
    return true;
}

// Sea-ice column inputs and precipitation partition from ice_ruc_ice,
// pinned lines 3593-3646 and 3671-3676. This is an internal building block;
// the caller selects the sea-ice column and separately supplies its geometry,
// saturation table and the post-icetmp copyback. For an absent optional
// graupel field it supplies a finite placeholder and sets has_graupel=false.
// Input slots: tabs,qvatm,qcatm,p8w,z3d,rho,rainbl,rainncv,frzfrac,
// snowncv,graupelncv,dt,flqc,flhc,cp,snow,snowh,snowc,rhosnf.
// Outputs: tabs,qvatm,qcatm,patm,conflx,rho,prcpms,newsnms,snowrat,
// grauprat,icerat,curat,precipfr,qkms,tkms,snwe,snhei,snowfrac,rhosnfall,
// rhosn. frpcpn and has_graupel are passive recorded configuration choices.
enum { SI_ICE_PRE_NINPUT=19, SI_ICE_PRE_NOUTPUT=20 };
struct SiIcePreTape {
    SiSiceClosureTape g;
    int out[SI_ICE_PRE_NOUTPUT],precip_arm;
    bool frozen_sum,positive_snow_density;
};
// ruc_min and ruc_max select their right argument on a finite tie.
__device__ __forceinline__ int si_sc_max(SiSiceClosureTape &t,int a,int b) {
    return t.v[a]>t.v[b]?a:b;
}
__device__ __forceinline__ int si_sc_min(SiSiceClosureTape &t,int a,int b) {
    return t.v[a]<t.v[b]?a:b;
}
// Source min/max choose their second operand on equality.
__device__ __forceinline__ int si_sc_max_record(SiSiceClosureTape &t,int a,int b,
                                               unsigned int *record,int slot) {
    const bool first=t.v[a]>t.v[b];
    if(record&&t.valid)si_branch(record,slot,first);
    return first?a:b;
}
__device__ __forceinline__ int si_sc_min_record(SiSiceClosureTape &t,int a,int b,
                                               unsigned int *record,int slot) {
    const bool first=t.v[a]<t.v[b];
    if(record&&t.valid)si_branch(record,slot,first);
    return first?a:b;
}
__device__ __forceinline__ bool si_ice_pre_record(
        const float *x,bool frpcpn,bool has_graupel,SiIcePreTape &s,
        unsigned int *branches=nullptr) {
    SiSiceClosureTape &t=s.g;t.n=0;t.valid=true;
    for(int i=0;i<SI_ICE_PRE_NINPUT;++i)si_sc_leaf(t,x[i]);
    int zero=si_sc_leaf(t,0.f),one=si_sc_leaf(t,1.f),half=si_sc_leaf(t,.5f);
    int milli=si_sc_leaf(t,1.e-3f),thousand=si_sc_leaf(t,1.e3f);
    int pscale=si_sc_leaf(t,1.e-5f),humidity=si_sc_leaf(t,.84f);
    int default_density=si_sc_leaf(t,300.f);
    int patm=si_sc_op(t,3,pscale,2),conflx=si_sc_op(t,4,half,2);
    int liquid=zero,frozen=zero,snowrat=zero,grauprat=zero,icerat=zero,curat=zero;
    s.precip_arm=0;s.frozen_sum=false;
    if(frpcpn) {
        int nc_liq=si_sc_op(t,7,si_sc_op(t,one,8,1),2);
        int nc_frozen=si_sc_op(t,7,8,2),cu_liq=zero,cu_frozen=zero;
        int conv=si_sc_op(t,6,7,1);
        const bool frozen_fraction=x[8]>0.f;
        const bool cold=frozen_fraction&&x[0]<273.f;
        if(branches){si_branch(branches,48,cold);si_branch(branches,49,frozen_fraction);
                     if(frozen_fraction)si_branch(branches,50,x[0]<273.f);}
        if(cold) {
            s.precip_arm=1;
            cu_liq=si_sc_max_record(t,zero,si_sc_op(t,conv,si_sc_op(t,one,8,1),2),branches,83);
            cu_frozen=si_sc_max_record(t,zero,si_sc_op(t,conv,8,2),branches,84);
        } else {
            si_branch_pair(branches,51,x[0]<273.f);
            if(x[0]<273.f) {
                s.precip_arm=2;cu_frozen=si_sc_max_record(t,zero,conv,branches,85);
            } else {
                s.precip_arm=3;cu_liq=si_sc_max_record(t,zero,conv,branches,86);
            }
        }
        int liquid_sum=si_sc_op(t,nc_liq,cu_liq,0);
        int frozen_sum=si_sc_op(t,nc_frozen,cu_frozen,0);
        liquid=si_sc_op(t,si_sc_op(t,liquid_sum,11,3),milli,2);
        frozen=si_sc_op(t,si_sc_op(t,frozen_sum,11,3),milli,2);
        s.frozen_sum=t.valid&&t.v[frozen_sum]>0.f;
        si_branch_pair(branches,54,s.frozen_sum);
        if(s.frozen_sum) {
            int graup=has_graupel?10:zero;
            int a=si_sc_op(t,9,frozen_sum,3),b=si_sc_op(t,graup,frozen_sum,3);
            int c=si_sc_op(t,si_sc_op(t,si_sc_op(t,nc_frozen,9,1),graup,1),frozen_sum,3);
            int d=si_sc_op(t,cu_frozen,frozen_sum,3);
            snowrat=si_sc_min_record(t,one,si_sc_max_record(t,zero,a,branches,88),branches,87);
            grauprat=si_sc_min_record(t,one,si_sc_max_record(t,zero,b,branches,90),branches,89);
            icerat=si_sc_min_record(t,one,si_sc_max_record(t,zero,c,branches,92),branches,91);
            curat=si_sc_min_record(t,one,si_sc_max_record(t,zero,d,branches,94),branches,93);
        }
    } else {
        si_branch_pair(branches,56,x[0]<=273.15f);
        if(x[0]<=273.15f) {
            s.precip_arm=4;frozen=si_sc_op(t,si_sc_op(t,6,11,3),milli,2);snowrat=one;
        } else {
            s.precip_arm=5;liquid=si_sc_op(t,si_sc_op(t,6,11,3),milli,2);
        }
    }
    int precip=si_sc_op(t,si_sc_op(t,frozen,11,2),thousand,2);
    int qkms=si_sc_op(t,12,5,3);
    int tkms=si_sc_op(t,si_sc_op(t,13,5,3),
                       si_sc_op(t,14,si_sc_op(t,one,si_sc_op(t,humidity,1,2),0),2),3);
    int snwe=si_sc_op(t,15,milli,2);
    s.positive_snow_density=x[15]>0.f&&x[16]>0.f;
    if(branches){si_branch(branches,63,s.positive_snow_density);
                 si_branch(branches,64,x[15]>0.f);
                 if(x[15]>0.f)si_branch(branches,65,x[16]>0.f);}
    int rhosn=s.positive_snow_density?si_sc_op(t,15,16,3):default_density;
    s.out[0]=0;s.out[1]=1;s.out[2]=2;s.out[3]=patm;s.out[4]=conflx;s.out[5]=5;
    s.out[6]=liquid;s.out[7]=frozen;s.out[8]=snowrat;s.out[9]=grauprat;
    s.out[10]=icerat;s.out[11]=curat;s.out[12]=precip;s.out[13]=qkms;
    s.out[14]=tkms;s.out[15]=snwe;s.out[16]=16;s.out[17]=17;s.out[18]=18;s.out[19]=rhosn;
    return t.valid;
}
__device__ __forceinline__ bool si_ice_pre_tl(
        const SiIcePreTape &s,const double *dx,double *dy) {
    const SiSiceClosureTape &t=s.g;if(!t.valid)return false;double d[SI_SC_NNODE];
    for(int i=0;i<t.n;++i) {
        if(i<SI_ICE_PRE_NINPUT)d[i]=dx[i];
        else if(t.a[i]<0)d[i]=0.;
        else d[i]=t.da[i]*d[t.a[i]]+t.db[i]*d[t.b[i]];
    }
    for(int i=0;i<SI_ICE_PRE_NOUTPUT;++i)dy[i]=d[s.out[i]];
    return true;
}
__device__ __forceinline__ bool si_ice_pre_ad(
        const SiIcePreTape &s,const double *by,double *bx) {
    const SiSiceClosureTape &t=s.g;if(!t.valid)return false;double b[SI_SC_NNODE];
    for(int i=0;i<t.n;++i)b[i]=0.;
    for(int i=0;i<SI_ICE_PRE_NOUTPUT;++i)b[s.out[i]]+=by[i];
    for(int i=t.n-1;i>=SI_ICE_PRE_NINPUT;--i)if(t.a[i]>=0) {
        b[t.a[i]]+=t.da[i]*b[i];b[t.b[i]]+=t.db[i]*b[i];
    }
    for(int i=0;i<SI_ICE_PRE_NINPUT;++i)bx[i]+=b[i];
    return true;
}

// The opening snow-depth, material, and moisture-budget calculations of
// ice_snowseaice (pinned lines 4306-4353). This is a fragment,
// not the complete snow routine. The caller preserves the old snhei and snwe:
// both are read before snhei is overwritten, and snwe can be overwritten.
// Inputs: old snhei, rhosn, old snwe, rho, qkms, qvatm, qsg, delt.
// Outputs: deltsn, snth, rhocsn, thdifsn, ras, new snhei, beta, new snwe,
// epot, epdt, snwepr. Each discrete choice follows the pinned comparison.
enum { SI_SN_PRE_NINPUT=8, SI_SN_PRE_NOUTPUT=11 };
struct SiSnowPrepTape {
    SiSiceClosureTape g;
    int out[SI_SN_PRE_NOUTPUT];
    bool thick,halve,positive_epdt,dryout,epdt_max_right;
};
__device__ __forceinline__ bool si_snow_prep_record(const float *x,SiSnowPrepTape &s, unsigned int *branches=nullptr) {
    SiSiceClosureTape &t=s.g;t.n=0;t.valid=true;
    for(int i=0;i<SI_SN_PRE_NINPUT;++i)si_sc_leaf(t,x[i]);
    int zero=si_sc_leaf(t,0.f),one=si_sc_leaf(t,1.f),half=si_sc_leaf(t,.5f);
    int thousand=si_sc_leaf(t,1.e3f),milli=si_sc_leaf(t,1.e-3f);
    int five=si_sc_leaf(t,.05f),cent=si_sc_leaf(t,.01f),heat=si_sc_leaf(t,2090.f);
    int conductivity=si_sc_leaf(t,.265f),floor=si_sc_leaf(t,1.e-8f);
    int deltsn=si_sc_op(t,si_sc_op(t,five,thousand,2),1,3);
    int snth=si_sc_op(t,si_sc_op(t,cent,thousand,2),1,3);
    int threshold=si_sc_op(t,deltsn,snth,0);
    s.thick=t.valid&&t.v[0]>=t.v[threshold];s.halve=false;
    si_branch_pair(branches,3,s.thick);
    if(s.thick) {
        int remainder=si_sc_op(t,si_sc_op(t,0,deltsn,1),snth,1);
        s.halve=t.valid&&t.v[remainder]<t.v[snth];
        si_branch_pair(branches,5,s.halve);
        if(s.halve)deltsn=si_sc_op(t,half,si_sc_op(t,0,snth,1),2);
    }
    int ras=si_sc_op(t,3,milli,2),rhocsn=si_sc_op(t,heat,1,2);
    int thdifsn=si_sc_op(t,conductivity,rhocsn,3);
    int snhei=si_sc_op(t,si_sc_op(t,2,thousand,2),1,3);
    int epot=si_sc_op(t,si_sc_neg(t,4),si_sc_op(t,5,6,1),2);
    int epdt=si_sc_op(t,si_sc_op(t,epot,ras,2),7,2);
    s.positive_epdt=t.valid&&t.v[epdt]>0.f;
    s.dryout=s.positive_epdt&&t.v[2]<=t.v[epdt];
    if(branches&&t.valid) {
        si_branch(branches,8,s.dryout);si_branch(branches,9,s.positive_epdt);
        if(s.positive_epdt)si_branch(branches,10,t.v[2]<=t.v[epdt]);
    }
    s.epdt_max_right=false;int beta=one,snwe=2;
    if(s.dryout) {
        // ruc_max(floor,epdt) takes epdt when the two compare equal.
        s.epdt_max_right=!(t.v[floor]>t.v[epdt]);
        if(branches&&t.valid)si_branch(branches,104,!s.epdt_max_right);
        int den=s.epdt_max_right?epdt:floor;
        beta=si_sc_op(t,2,den,3);snwe=zero;
    }
    s.out[0]=deltsn;s.out[1]=snth;s.out[2]=rhocsn;s.out[3]=thdifsn;
    s.out[4]=ras;s.out[5]=snhei;s.out[6]=beta;s.out[7]=snwe;
    s.out[8]=epot;s.out[9]=epdt;s.out[10]=2;
    return t.valid;
}
__device__ __forceinline__ bool si_snow_prep_tl(
        const SiSnowPrepTape &s,const double *dx,double *dy) {
    const SiSiceClosureTape &t=s.g;if(!t.valid)return false;double d[SI_SC_NNODE];
    for(int i=0;i<t.n;++i) {
        if(i<SI_SN_PRE_NINPUT)d[i]=dx[i];
        else if(t.a[i]<0)d[i]=0.;
        else d[i]=t.da[i]*d[t.a[i]]+t.db[i]*d[t.b[i]];
    }
    for(int i=0;i<SI_SN_PRE_NOUTPUT;++i)dy[i]=d[s.out[i]];
    return true;
}
__device__ __forceinline__ bool si_snow_prep_ad(
        const SiSnowPrepTape &s,const double *by,double *bx) {
    const SiSiceClosureTape &t=s.g;if(!t.valid)return false;double b[SI_SC_NNODE];
    for(int i=0;i<t.n;++i)b[i]=0.;
    for(int i=0;i<SI_SN_PRE_NOUTPUT;++i)b[s.out[i]]+=by[i];
    for(int i=t.n-1;i>=SI_SN_PRE_NINPUT;--i)if(t.a[i]>=0) {
        b[t.a[i]]+=t.da[i]*b[i];b[t.b[i]]+=t.db[i]*b[i];
    }
    for(int i=0;i<SI_SN_PRE_NINPUT;++i)bx[i]+=b[i];
    return true;
}

// Snow-to-ice layer coupling in ice_snowseaice, lines 4367-4425. The
// unchanged elimination inputs cprev/rprev are cotso(nzs-1)/rhtso(nzs-1),
// cprev2/rprev2 are the preceding pair. Derivatives include snhei in each
// selected formula; the branch comparisons themselves have zero tangent.
// In the no-snow arm, slots whose source locals have not been initialized are
// marked invalid. The caller must not consume those slots in that arm.
enum { SI_SN_COUPLE_NINPUT=22, SI_SN_COUPLE_NOUTPUT=12 };
struct SiSnowCoupleTape {
    SiSiceClosureTape g;
    int out[SI_SN_COUPLE_NOUTPUT],mask,arm,ilnb;
};
__device__ __forceinline__ bool si_snow_couple_record(const float *x,SiSnowCoupleTape &s, unsigned int *branches=nullptr) {
    SiSiceClosureTape &t=s.g;t.n=0;t.valid=true;
    for(int k=0;k<SI_SN_COUPLE_NINPUT;++k)si_sc_leaf(t,x[k]);
    int zero=si_sc_leaf(t,0.f),one=si_sc_leaf(t,1.f),two=si_sc_leaf(t,2.f);
    int half=si_sc_leaf(t,.5f),freeze=si_sc_leaf(t,273.15f);
    // x: snhei,snth,deltsn,delt,zshalf2,zshalf3,zsmain2,thdifsn,
    // dtdzs1,dtdzs2,diff1,diff2,tso1,tso2,tso3,soilt,old_soilt1,
    // cprev,rprev,cprev2,rprev2,old_tsnav.
    int clast=zero,rlast=zero,cprev=17,rprev=18,cotsn=zero,rhtsn=zero;
    int tsnav=21,soilt1=16,tsob=zero,snprim=zero,fsn=one,fso=zero;
    s.ilnb=0;s.arm=0;
    s.mask=(1<<2)|(1<<3)|(1<<6)|(1<<7)|(1<<10)|(1<<11);
    int threshold=si_sc_op(t,2,1,0);
    const bool thick=t.v[0]>=t.v[1];si_branch_pair(branches,13,thick);
    if(thick) {
        s.mask=(1<<SI_SN_COUPLE_NOUTPUT)-1;
        const bool one_layer=t.v[0]<=t.v[threshold];si_branch_pair(branches,15,one_layer);
        if(one_layer) {
            s.arm=2;s.ilnb=1;
            snprim=si_sc_max_record(t,1,0,branches,105); // right on tie
            soilt1=12;tsob=12;
            int xsn=si_sc_op(t,si_sc_op(t,3,two,3),
                                 si_sc_op(t,4,si_sc_op(t,half,snprim,2),0),3);
            int ddz=si_sc_op(t,xsn,snprim,3),x1=si_sc_op(t,ddz,7,2);
            int x2=si_sc_op(t,8,10,2);
            int ft=si_sc_op(t,si_sc_op(t,12,si_sc_op(t,x1,si_sc_op(t,15,12,1),2),0),
                                  si_sc_op(t,x2,si_sc_op(t,12,13,1),2),1);
            int den=si_sc_op(t,si_sc_op(t,si_sc_op(t,one,x1,0),x2,0),si_sc_op(t,x2,cprev,2),1);
            clast=si_sc_op(t,x1,den,3);rlast=si_sc_op(t,si_sc_op(t,ft,si_sc_op(t,x2,rprev,2),0),den,3);
            cotsn=clast;rhtsn=rlast;
            tsnav=si_sc_op(t,si_sc_op(t,half,si_sc_op(t,15,12,0),2),freeze,1);
        } else {
            s.arm=3;s.ilnb=2;snprim=2;tsob=16;
            int depth=si_sc_op(t,0,2,1);
            int xsn=si_sc_op(t,si_sc_op(t,3,two,3),si_sc_op(t,half,0,2),3);
            int xsn1=si_sc_op(t,si_sc_op(t,3,two,3),
                                  si_sc_op(t,4,si_sc_op(t,half,depth,2),0),3);
            int x1=si_sc_op(t,si_sc_op(t,xsn,2,3),7,2);
            int x11=si_sc_op(t,si_sc_op(t,xsn1,depth,3),7,2);
            int x2=si_sc_op(t,8,10,2);
            int ft=si_sc_op(t,si_sc_op(t,12,si_sc_op(t,x11,si_sc_op(t,16,12,1),2),0),
                                  si_sc_op(t,x2,si_sc_op(t,12,13,1),2),1);
            int den=si_sc_op(t,si_sc_op(t,si_sc_op(t,one,x11,0),x2,0),si_sc_op(t,x2,cprev,2),1);
            clast=si_sc_op(t,x11,den,3);rlast=si_sc_op(t,si_sc_op(t,ft,si_sc_op(t,x2,rprev,2),0),den,3);
            int ft2=si_sc_op(t,si_sc_op(t,16,si_sc_op(t,x1,si_sc_op(t,15,16,1),2),0),
                                   si_sc_op(t,x11,si_sc_op(t,16,12,1),2),1);
            int den2=si_sc_op(t,si_sc_op(t,si_sc_op(t,one,x1,0),x11,0),si_sc_op(t,x11,clast,2),1);
            cotsn=si_sc_op(t,x1,den2,3);rhtsn=si_sc_op(t,si_sc_op(t,ft2,si_sc_op(t,x11,rlast,2),0),den2,3);
            int avg=si_sc_op(t,si_sc_op(t,si_sc_op(t,15,16,0),2,2),
                                  si_sc_op(t,si_sc_op(t,16,12,0),depth,2),0);
            tsnav=si_sc_op(t,si_sc_op(t,si_sc_op(t,half,0,3),avg,2),freeze,1);
        }
    }
    const bool thin_left=t.v[0]<t.v[1],thin_right=thin_left&&t.v[0]>0.f;
    if(branches&&t.valid) {
        si_branch(branches,17,thin_right);si_branch(branches,18,thin_left);
        if(thin_left)si_branch(branches,19,t.v[0]>0.f);
    }
    if(thin_right) {
        s.arm=1;s.mask=(1<<SI_SN_COUPLE_NOUTPUT)-1;
        snprim=si_sc_op(t,0,6,0);fsn=si_sc_op(t,0,snprim,3);fso=si_sc_op(t,one,fsn,1);
        soilt1=12;tsob=13;
        int xsn=si_sc_op(t,si_sc_op(t,3,two,3),
                          si_sc_op(t,si_sc_op(t,5,6,1),si_sc_op(t,half,snprim,2),0),3);
        int material=si_sc_op(t,si_sc_op(t,fsn,7,2),si_sc_op(t,fso,10,2),0);
        int x1=si_sc_op(t,si_sc_op(t,xsn,snprim,3),material,2);
        int x2=si_sc_op(t,9,11,2);
        int ft=si_sc_op(t,si_sc_op(t,13,si_sc_op(t,x1,si_sc_op(t,15,13,1),2),0),
                              si_sc_op(t,x2,si_sc_op(t,13,14,1),2),1);
        int den=si_sc_op(t,si_sc_op(t,si_sc_op(t,one,x1,0),x2,0),si_sc_op(t,x2,19,2),1);
        cprev=si_sc_op(t,x1,den,3);rprev=si_sc_op(t,si_sc_op(t,ft,si_sc_op(t,x2,20,2),0),den,3);
        clast=cprev;rlast=rprev;cotsn=clast;rhtsn=rlast;
        tsnav=si_sc_op(t,si_sc_op(t,half,si_sc_op(t,15,12,0),2),freeze,1);
    }
    s.out[0]=clast;s.out[1]=rlast;s.out[2]=cprev;s.out[3]=rprev;
    s.out[4]=cotsn;s.out[5]=rhtsn;s.out[6]=tsnav;s.out[7]=soilt1;
    s.out[8]=tsob;s.out[9]=snprim;s.out[10]=fsn;s.out[11]=fso;
    return t.valid;
}
__device__ __forceinline__ bool si_snow_couple_tl(
        const SiSnowCoupleTape &s,const double *dx,double *dy) {
    const SiSiceClosureTape &t=s.g;if(!t.valid)return false;double d[SI_SC_NNODE];
    for(int i=0;i<t.n;++i) {
        if(i<SI_SN_COUPLE_NINPUT)d[i]=dx[i];
        else if(t.a[i]<0)d[i]=0.;
        else d[i]=t.da[i]*d[t.a[i]]+t.db[i]*d[t.b[i]];
    }
    for(int i=0;i<SI_SN_COUPLE_NOUTPUT;++i)dy[i]=(s.mask&(1<<i))?d[s.out[i]]:0.;
    return true;
}
__device__ __forceinline__ bool si_snow_couple_ad(
        const SiSnowCoupleTape &s,const double *by,double *bx) {
    const SiSiceClosureTape &t=s.g;if(!t.valid)return false;double b[SI_SC_NNODE];
    for(int i=0;i<t.n;++i)b[i]=0.;
    for(int i=0;i<SI_SN_COUPLE_NOUTPUT;++i)if(s.mask&(1<<i))b[s.out[i]]+=by[i];
    for(int i=t.n-1;i>=SI_SN_COUPLE_NINPUT;--i)if(t.a[i]>=0) {
        b[t.a[i]]+=t.da[i]*b[i];b[t.b[i]]+=t.db[i]*b[i];
    }
    for(int i=0;i<SI_SN_COUPLE_NINPUT;++i)bx[i]+=b[i];
    return true;
}

// Surface-equation coefficients in ice_snowseaice, pinned lines 4428-4476.
// This pair is a composition building block: the caller supplies the current
// inputs on each visit and must chain the following ice_vilka call separately.
// A negative snow depth leaves the native coefficient locals undefined; the
// pair refuses that execution rather than manufacturing a derivative.
// Inputs are snhei,snth,deltsn,qkms,qvatm,qsg,capice1,tkms,cprev,rprev,
// soilt,diff1,zsmain1,zsmain2,cp,rho,conflx,delt,emiss,stbolt,rnet,
// cotsn,rhtsn,snprim,fsn,fso,thdifsn,rhocsn,rainf,cw,prcpms,rhonewsn,
// newsnow,xlv,tabs,qvg,patm,clast,rlast,beta,snoh.
// Outputs are aa1,bb,pp,tdenom,d1sn,d2sn,d9sn,r22sn,r21,r210,d10,epot,
// r211,rhcs,dzstop,d11. All named intermediates remain active.
enum { SI_SN_CLOSURE_NINPUT=41, SI_SN_CLOSURE_NOUTPUT=16 };
struct SiSnowClosureTape {
    SiSiceClosureTape g;
    int out[SI_SN_CLOSURE_NOUTPUT],arm;
    bool warm,cold;
};
__device__ __forceinline__ bool si_snow_closure_record(const float *x,SiSnowClosureTape &s, unsigned int *branches=nullptr) {
    SiSiceClosureTape &t=s.g;t.n=0;t.valid=true;s.arm=-1;
    for(int i=0;i<SI_SN_CLOSURE_NINPUT;++i)si_sc_leaf(t,x[i]);
    int one=si_sc_leaf(t,1.f),half=si_sc_leaf(t,.5f),freeze=si_sc_leaf(t,273.15f);
    int melt=si_sc_leaf(t,3.35e5f),thousand=si_sc_leaf(t,1.e3f);
    int snow_heat=si_sc_leaf(t,2090.f);
    int epot=si_sc_op(t,si_sc_neg(t,3),si_sc_op(t,4,5,1),2);
    int dzstop=si_sc_op(t,one,si_sc_op(t,13,12,1),3);
    int d9=si_sc_op(t,si_sc_op(t,11,6,2),dzstop,2);
    int d10=si_sc_op(t,si_sc_op(t,7,14,2),15,2);
    int r211=si_sc_op(t,si_sc_op(t,half,16,2),17,3);
    int r21=si_sc_op(t,si_sc_op(t,r211,14,2),15,2);
    int r22=si_sc_op(t,half,si_sc_op(t,si_sc_op(t,11,17,2),
                                     si_sc_op(t,dzstop,dzstop,2),2),3);
    int r6=si_sc_op(t,si_sc_op(t,si_sc_op(t,18,19,2),half,2),
                    si_sc_op(t,si_sc_op(t,10,10,2),si_sc_op(t,10,10,2),2),2);
    int r7=si_sc_op(t,r6,10,3),d11=si_sc_op(t,20,r6,0);
    int d1=0,d2=0,d9sn=0,r22sn=0;
    const bool thick=x[0]>=x[1];si_branch_pair(branches,20,thick);
    if(thick) {
        const bool one_layer=x[0]<=__fadd_rn(x[2],x[1]);si_branch_pair(branches,22,one_layer);
        if(one_layer) {s.arm=2;d1=37;d2=38;}
        else {s.arm=3;d1=21;d2=22;}
        d9sn=si_sc_op(t,si_sc_op(t,26,27,2),23,3);
        r22sn=si_sc_op(t,si_sc_op(t,si_sc_op(t,23,23,2),half,2),
                          si_sc_op(t,26,17,2),3);
    }
    const bool thin_left=x[0]<x[1],thin=thin_left&&x[0]>0.f;
    if(branches) {
        si_branch(branches,24,thin);si_branch(branches,25,thin_left);
        if(thin_left)si_branch(branches,26,x[0]>0.f);
    }
    if(thin) {
        s.arm=1;d1=8;d2=9;
        int thermal=si_sc_op(t,si_sc_op(t,si_sc_op(t,24,26,2),27,2),
                               si_sc_op(t,si_sc_op(t,25,11,2),6,2),0);
        int diffusivity=si_sc_op(t,si_sc_op(t,24,26,2),si_sc_op(t,25,11,2),0);
        d9sn=si_sc_op(t,thermal,23,3);
        r22sn=si_sc_op(t,si_sc_op(t,si_sc_op(t,23,23,2),half,2),
                          si_sc_op(t,diffusivity,17,2),3);
    }
    si_branch_pair(branches,27,x[0]==0.f);
    if(x[0]==0.f) {s.arm=0;d1=8;d2=9;d9sn=d9;r22sn=r22;}
    if(s.arm<0)return false;
    int rain=si_sc_op(t,si_sc_op(t,28,29,2),30,2);
    int new_snow_capacity=si_sc_op(t,snow_heat,31,2);
    int news=si_sc_op(t,si_sc_op(t,new_snow_capacity,32,2),17,3);
    int part=si_sc_op(t,si_sc_op(t,one,d1,1),r22sn,0);
    int den=si_sc_op(t,si_sc_op(t,si_sc_op(t,si_sc_op(t,
             si_sc_op(t,si_sc_op(t,d9sn,part,2),d10,0),r21,0),r7,0),rain,0),news,0);
    int fkq=si_sc_op(t,3,15,2),r210=si_sc_op(t,r211,15,2);
    int xls=si_sc_op(t,33,melt,0),betaf=si_sc_op(t,39,fkq,2);
    int aa=si_sc_op(t,si_sc_op(t,xls,si_sc_op(t,betaf,r210,0),2),den,3);
    s.warm=!(273.15f>x[34]);s.cold=!(273.15f<x[34]);
    if(branches) {si_branch(branches,106,!s.warm);si_branch(branches,107,!s.cold);}
    int warm=s.warm?34:freeze,cold=s.cold?34:freeze;
    int n0=si_sc_op(t,d10,34,2),n1=si_sc_op(t,r21,10,2);
    int n2=si_sc_op(t,xls,si_sc_op(t,si_sc_op(t,4,betaf,2),
                                      si_sc_op(t,r210,35,2),0),2);
    int n3=si_sc_op(t,d9sn,si_sc_op(t,d2,si_sc_op(t,r22sn,10,2),0),2);
    int n4=si_sc_op(t,rain,warm,2),n5=si_sc_op(t,news,cold,2);
    int num=si_sc_op(t,si_sc_op(t,si_sc_op(t,si_sc_op(t,si_sc_op(t,
              si_sc_op(t,n0,n1,0),n2,0),d11,0),n3,0),n4,0),n5,0);
    int bb=si_sc_op(t,num,den,3),pp=si_sc_op(t,36,thousand,2);
    int aa1=si_sc_op(t,aa,pp,3);
    bb=si_sc_op(t,bb,si_sc_op(t,40,den,3),1);
    s.out[0]=aa1;s.out[1]=bb;s.out[2]=pp;s.out[3]=den;
    s.out[4]=d1;s.out[5]=d2;s.out[6]=d9sn;s.out[7]=r22sn;
    s.out[8]=r21;s.out[9]=r210;s.out[10]=d10;s.out[11]=epot;
    s.out[12]=r211;s.out[13]=6;s.out[14]=dzstop;s.out[15]=d11;
    return t.valid;
}
__device__ __forceinline__ bool si_snow_closure_tl(
        const SiSnowClosureTape &s,const double *dx,double *dy) {
    const SiSiceClosureTape &t=s.g;if(!t.valid)return false;double d[SI_SC_NNODE];
    for(int i=0;i<t.n;++i) {
        if(i<SI_SN_CLOSURE_NINPUT)d[i]=dx[i];
        else if(t.a[i]<0)d[i]=0.;
        else d[i]=t.da[i]*d[t.a[i]]+t.db[i]*d[t.b[i]];
    }
    for(int i=0;i<SI_SN_CLOSURE_NOUTPUT;++i)dy[i]=d[s.out[i]];
    return true;
}
__device__ __forceinline__ bool si_snow_closure_ad(
        const SiSnowClosureTape &s,const double *by,double *bx) {
    const SiSiceClosureTape &t=s.g;if(!t.valid)return false;double b[SI_SC_NNODE];
    for(int i=0;i<t.n;++i)b[i]=0.;
    for(int i=0;i<SI_SN_CLOSURE_NOUTPUT;++i)b[s.out[i]]+=by[i];
    for(int i=t.n-1;i>=SI_SN_CLOSURE_NINPUT;--i)if(t.a[i]>=0) {
        b[t.a[i]]+=t.da[i]*b[i];b[t.b[i]]+=t.db[i]*b[i];
    }
    for(int i=0;i<SI_SN_CLOSURE_NINPUT;++i)bx[i]+=b[i];
    return true;
}

// Melt selection and the non-melting water update, ice_snowseaice lines
// 4524-4540 and 4589-4596. The melt arm produces soiltfrac for a subsequent
// ice_qsn call. In the no-melt arm it leaves the old surface temperature and
// returns the updated water/evaporation state. The caller chains qsn at
// soiltfrac before evaluating the melt-energy fragment below.
// Inputs: soilt,snwepr,beta,old_epot,ras,delt,snhei,snowfrac,old_snwe,
// qvatm,qsg,qkms. Outputs: soiltfrac,epot,snwe. The recorded `melt` flag is
// the executed branch, including the exact ordering of the comparisons.
enum { SI_SM_SELECT_NINPUT=12, SI_SM_SELECT_NOUTPUT=3 };
struct SiSnowMeltSelectTape {
    SiSiceClosureTape g;
    int out[SI_SM_SELECT_NOUTPUT];
    bool melt,has_snow;
};
__device__ __forceinline__ bool si_snow_melt_select_record(
        const float *x,SiSnowMeltSelectTape &s, unsigned int *branches=nullptr) {
    SiSiceClosureTape &t=s.g;t.n=0;t.valid=true;
    for(int i=0;i<SI_SM_SELECT_NINPUT;++i)si_sc_leaf(t,x[i]);
    int zero=si_sc_leaf(t,0.f),one=si_sc_leaf(t,1.f);
    int freeze=si_sc_leaf(t,273.15f),icecap=si_sc_leaf(t,271.4f);
    s.melt=false;
    if(branches)si_branch(branches,50,x[0]>273.15f);
    if(x[0]>273.15f) {
        int evaporated=si_sc_op(t,si_sc_op(t,si_sc_op(t,2,3,2),4,2),5,2);
        int available=si_sc_op(t,1,evaporated,1);
        const bool available_positive=t.valid && t.v[available]>0.f;
        if(branches&&t.valid) {
            si_branch(branches,51,available_positive);
            if(available_positive)si_branch(branches,52,x[6]>0.f);
        }
        s.melt=available_positive && x[6]>0.f;
    }
    if(branches)si_branch(branches,49,s.melt);
    s.has_snow=x[6]!=0.f;
    if(!s.melt)si_branch_pair(branches,68,s.has_snow);
    int frac=0,epot=3,snwe=8;
    if(s.melt) {
        int capped=si_sc_min_record(t,icecap,0,branches,117);
        frac=si_sc_op(t,si_sc_op(t,7,freeze,2),
                        si_sc_op(t,si_sc_op(t,one,7,1),capped,2),0);
    } else if(s.has_snow) {
        epot=si_sc_op(t,si_sc_neg(t,11),si_sc_op(t,9,10,1),2);
        int lost=si_sc_op(t,si_sc_op(t,si_sc_op(t,2,epot,2),4,2),5,2);
        snwe=si_sc_max_record(t,zero,si_sc_op(t,1,lost,1),branches,131);
    }
    s.out[0]=frac;s.out[1]=epot;s.out[2]=snwe;
    return t.valid;
}
__device__ __forceinline__ bool si_snow_graph_tl(const SiSiceClosureTape &t,
        int ni,int no,const int *out,const double *dx,double *dy) {
    if(!t.valid)return false;double d[SI_SC_NNODE];
    for(int i=0;i<t.n;++i) {
        if(i<ni)d[i]=dx[i];else if(t.a[i]<0)d[i]=0.;
        else d[i]=t.da[i]*d[t.a[i]]+t.db[i]*d[t.b[i]];
    }
    for(int i=0;i<no;++i)dy[i]=d[out[i]];
    return true;
}
__device__ __forceinline__ bool si_snow_graph_ad(const SiSiceClosureTape &t,
        int ni,int no,const int *out,const double *by,double *bx) {
    if(!t.valid)return false;double b[SI_SC_NNODE];
    for(int i=0;i<t.n;++i)b[i]=0.;
    for(int i=0;i<no;++i)b[out[i]]+=by[i];
    for(int i=t.n-1;i>=ni;--i)if(t.a[i]>=0) {
        b[t.a[i]]+=t.da[i]*b[i];b[t.b[i]]+=t.db[i]*b[i];
    }
    for(int i=0;i<ni;++i)bx[i]+=b[i];
    return true;
}
__device__ __forceinline__ bool si_snow_melt_select_tl(
        const SiSnowMeltSelectTape &s,const double *dx,double *dy) {
    return si_snow_graph_tl(s.g,SI_SM_SELECT_NINPUT,SI_SM_SELECT_NOUTPUT,s.out,dx,dy);
}
__device__ __forceinline__ bool si_snow_melt_select_ad(
        const SiSnowMeltSelectTape &s,const double *by,double *bx) {
    return si_snow_graph_ad(s.g,SI_SM_SELECT_NINPUT,SI_SM_SELECT_NOUTPUT,s.out,by,bx);
}

// Physics-to-MPAS scalar hand-off for an active sea-ice column, source lines
// 3714-3731. The threshold and the pre-call snow amount control branches;
// their derivatives are zero on an executed path. Other columns and the
// array copies are handled by the caller. Snow water converts metres to mm.
// Inputs: qvg,emissl,old_snow,snwe,snhei,sflx,snowfrac,xice,rhosnfall,
// sfcevp,qfx,dt,xice_threshold,znt. Outputs: qsfc,emissl,emiss,snow,
// snowh,grdflx,snowc,rhosnf,sfcevp,z0.
enum { SI_ICE_POST_NINPUT=14, SI_ICE_POST_NOUTPUT=10 };
struct SiIcePostTape {
    SiSiceClosureTape g;int out[SI_ICE_POST_NOUTPUT];
    bool snowless,scale_snowfrac;
};
__device__ __forceinline__ bool si_ice_post_record(const float *x,SiIcePostTape &s) {
    SiSiceClosureTape &t=s.g;t.n=0;t.valid=true;
    for(int i=0;i<SI_ICE_POST_NINPUT;++i)si_sc_leaf(t,x[i]);
    int one=si_sc_leaf(t,1.f),minus=si_sc_leaf(t,-1.f);
    int thousand=si_sc_leaf(t,1000.f),emissice=si_sc_leaf(t,.98f);
    int qsfc=si_sc_op(t,0,si_sc_op(t,one,0,0),3);
    s.snowless=x[2]==0.f;
    int emiss=s.snowless?emissice:1;
    int snow=si_sc_op(t,3,thousand,2),grd=si_sc_op(t,minus,5,2);
    s.scale_snowfrac=x[6]>0.f&&x[7]>=x[12];
    int snowc=s.scale_snowfrac?si_sc_op(t,6,7,2):6;
    int sfcevp=si_sc_op(t,9,si_sc_op(t,10,11,2),0);
    s.out[0]=qsfc;s.out[1]=emiss;s.out[2]=emiss;s.out[3]=snow;
    s.out[4]=4;s.out[5]=grd;s.out[6]=snowc;s.out[7]=8;
    s.out[8]=sfcevp;s.out[9]=13;return t.valid;
}
__device__ __forceinline__ bool si_ice_post_tl(
        const SiIcePostTape &s,const double *dx,double *dy) {
    return si_snow_graph_tl(s.g,SI_ICE_POST_NINPUT,SI_ICE_POST_NOUTPUT,s.out,dx,dy);
}
__device__ __forceinline__ bool si_ice_post_ad(
        const SiIcePostTape &s,const double *by,double *bx) {
    return si_snow_graph_ad(s.g,SI_ICE_POST_NINPUT,SI_ICE_POST_NOUTPUT,s.out,by,bx);
}

// The executed argument and result of an existing forward math helper.
// The caller replays the unchanged forward helper and saves its result.
// `arg` is checked bitwise at the exact call site. Per COMMON-AD section 57,
// the local rule uses y for exp and 1-y*y for tanh, evaluated from the saved
// result; no library source is copied. These records are not tunable controls.
struct SiIceUnaryRecord {float arg,value;};
__device__ __forceinline__ int si_sc_unary_record(SiSiceClosureTape &t,int a,
                                                  const SiIceUnaryRecord &u,bool exp_op) {
    if(!t.valid||t.n>=SI_SC_NNODE||!si_finite(u.arg)||!si_finite(u.value)||
       __float_as_uint(t.v[a])!=__float_as_uint(u.arg)) {
        t.valid=false;return 0;
    }
    double y=double(u.value),slope=exp_op?y:1.-y*y;
    int k=t.n++;t.v[k]=u.value;t.a[k]=t.b[k]=a;t.da[k]=slope;t.db[k]=0.;return k;
}

// Snow compaction, snowfall mixing and coverage in ice_icetmp, lines
// 3808-17 and 3851-90. Inputs: delt,c1sn,c2sn,tsnav,rhosn,snhei,snwe,
// snowfrac,newsnms,tabs,snowrat,grauprat,icerat,curat,prcpms,snowfallac,
// old_rhosnfall. Unary records 0..3 are exp(compaction), exp(growth),
// tanh(snow density), tanh(graupel density), respectively; only records on
// the executed arm are read. The caller owns the unary replay and its proof.
// Outputs: snhei_crit,snhei_crit_newsn,rhosn,snwe,snhei,snowfrac,newsn,
// rhonewsn,newsnowratio,snowfracnewsn,rainf,rhosnfall. The tape also records
// the exact snow_mosaic / keep_snow_albedo decisions for the optics fragment.
enum { SI_ICE_SNOWSTATE_NINPUT=17, SI_ICE_SNOWSTATE_NOUTPUT=12 };
struct SiIceSnowStateTape {
    SiSiceClosureTape g;
    int out[SI_ICE_SNOWSTATE_NOUTPUT];
    bool compact,compact_growth,new_snow,has_snow,mosaic,keep_snow_albedo;
};
__device__ __forceinline__ bool si_icetmp_snowstate_record(
        const float *x,const SiIceUnaryRecord *unary,SiIceSnowStateTape &s,
        unsigned int *branches=nullptr) {
    SiSiceClosureTape &t=s.g;t.n=0;t.valid=true;
    for(int i=0;i<SI_ICE_SNOWSTATE_NINPUT;++i)si_sc_leaf(t,x[i]);
    int zero=si_sc_leaf(t,0.f),one=si_sc_leaf(t,1.f),two=si_sc_leaf(t,2.f);
    int low=si_sc_leaf(t,58.8f),high=si_sc_leaf(t,500.f),water=si_sc_leaf(t,1000.f);
    int hundred=si_sc_leaf(t,100.f),h125=si_sc_leaf(t,125.f),h17=si_sc_leaf(t,17.f);
    int h35=si_sc_leaf(t,3.5f),h8=si_sc_leaf(t,8.f),h3600=si_sc_leaf(t,3600.f);
    int milli=si_sc_leaf(t,1.e-3f),critc=si_sc_leaf(t,.01601f),critnew=si_sc_leaf(t,.0005f);
    int c0081=si_sc_leaf(t,.0081f),c008=si_sc_leaf(t,.08f);
    int c015=si_sc_leaf(t,.15f),c033=si_sc_leaf(t,.3333f);
    int k276=si_sc_leaf(t,276.65f),k274=si_sc_leaf(t,274.15f);
    int crit=si_sc_op(t,si_sc_op(t,critc,water,2),4,3);
    int crit2=si_sc_op(t,si_sc_op(t,critnew,water,2),4,3);
    int threshold=si_sc_op(t,si_sc_op(t,c0081,water,2),4,3);
    int rho=4,snow=6,depth=5,frac=x[5]==0.f?zero:7;
    int newsn=zero,rhonewsn=hundred,fall=16,ratio=zero,fracnew=zero,rain=zero;
    si_branch_pair(branches,2,x[5]==0.f);
    s.compact=t.valid&&x[5]>t.v[threshold];s.compact_growth=false;
    si_branch_pair(branches,10,s.compact);
    if(s.compact) {
        int cold=si_sc_min_record(t,zero,3,branches,66);
        int arg=si_sc_op(t,si_sc_op(t,c008,cold,2),
                            si_sc_op(t,si_sc_op(t,2,4,2),milli,2),1);
        if(!unary)return false;
        int ex=si_sc_unary_record(t,arg,unary[0],true);
        int bsn=si_sc_op(t,si_sc_op(t,si_sc_op(t,0,h3600,3),1,2),ex,2);
        int growth=si_sc_op(t,si_sc_op(t,bsn,snow,2),hundred,2);
        s.compact_growth=t.valid&&t.v[growth]>1.e-4f;
        si_branch_pair(branches,12,s.compact_growth);
        if(s.compact_growth) {
            int ex2=si_sc_unary_record(t,growth,unary[1],true);
            int mixed=si_sc_op(t,si_sc_op(t,rho,si_sc_op(t,ex2,one,1),2),growth,3);
            rho=si_sc_min_record(t,si_sc_max_record(t,low,mixed,branches,68),high,branches,67);
        }
    }
    s.mosaic=t.valid&&t.v[frac]<.75f;
    si_branch_pair(branches,14,s.mosaic);
    newsn=si_sc_op(t,8,0,2);s.new_snow=t.valid&&t.v[newsn]>0.f;
    si_branch_pair(branches,16,s.new_snow);
    if(s.new_snow) {
        int sum=si_sc_op(t,snow,newsn,0);
        ratio=si_sc_min_record(t,one,si_sc_op(t,newsn,sum,3),branches,69);
        int a=si_sc_op(t,si_sc_op(t,k276,9,1),c015,2);
        int b=si_sc_op(t,si_sc_op(t,k274,9,1),c033,2);
        if(!unary)return false;
        int tanh_s=si_sc_unary_record(t,a,unary[2],false);
        int tanh_g=si_sc_unary_record(t,b,unary[3],false);
        int fresh=si_sc_min_record(t,h125,si_sc_op(t,water,
                    si_sc_max_record(t,h8,si_sc_op(t,h17,tanh_s,2),branches,71),3),branches,70);
        int graupel=si_sc_min_record(t,high,si_sc_op(t,water,
                    si_sc_max_record(t,two,si_sc_op(t,h35,tanh_g,2),branches,73),3),branches,72);
        int terms=si_sc_op(t,si_sc_op(t,si_sc_op(t,si_sc_op(t,fresh,10,2),
                         si_sc_op(t,graupel,11,2),0),si_sc_op(t,fresh,12,2),0),
                         si_sc_op(t,graupel,13,2),0);
        fall=si_sc_min_record(t,high,si_sc_max_record(t,low,terms,branches,75),branches,74);rhonewsn=fall;
        int numerator=si_sc_op(t,si_sc_op(t,rho,snow,2),si_sc_op(t,rhonewsn,newsn,2),0);
        rho=si_sc_min_record(t,si_sc_max_record(t,low,si_sc_op(t,numerator,sum,3),branches,77),high,branches,76);
    }
    si_branch_pair(branches,19,x[14]>0.f);
    if(x[14]>0.f)rain=one;
    si_branch_pair(branches,21,s.new_snow);
    if(s.new_snow)snow=si_sc_max_record(t,zero,si_sc_op(t,snow,newsn,0),branches,78);
    depth=si_sc_op(t,si_sc_op(t,snow,water,2),rho,3);
    newsn=si_sc_op(t,si_sc_op(t,newsn,water,2),rhonewsn,3);
    s.has_snow=t.valid&&t.v[depth]>0.f;s.keep_snow_albedo=false;
    si_branch_pair(branches,23,s.has_snow);
    if(s.has_snow) {
        frac=si_sc_min_record(t,one,si_sc_op(t,depth,si_sc_op(t,two,crit,2),3),branches,79);
        const bool fresh=t.valid&&t.v[newsn]>0.f;
        si_branch_pair(branches,25,fresh);
        if(fresh)fracnew=si_sc_min_record(t,one,si_sc_op(t,15,crit2,3),branches,80);
        const bool thin=t.valid&&t.v[frac]<.75f;
        si_branch_pair(branches,27,thin);
        if(thin)s.mosaic=true;
        const bool left=t.valid&&t.v[fracnew]>.99f;
        const bool keep=left&&t.v[fall]<450.f;
        if(branches) {si_branch(branches,29,keep);si_branch(branches,30,left);
                      if(left)si_branch(branches,31,t.v[fall]<450.f);}
        if(keep){s.keep_snow_albedo=true;s.mosaic=false;}
    }
    s.out[0]=crit;s.out[1]=crit2;s.out[2]=rho;s.out[3]=snow;s.out[4]=depth;
    s.out[5]=frac;s.out[6]=newsn;s.out[7]=rhonewsn;s.out[8]=ratio;
    s.out[9]=fracnew;s.out[10]=rain;s.out[11]=fall;
    return t.valid;
}
__device__ __forceinline__ bool si_icetmp_snowstate_tl(
        const SiIceSnowStateTape &s,const double *dx,double *dy) {
    return si_snow_graph_tl(s.g,SI_ICE_SNOWSTATE_NINPUT,SI_ICE_SNOWSTATE_NOUTPUT,s.out,dx,dy);
}
__device__ __forceinline__ bool si_icetmp_snowstate_ad(
        const SiIceSnowStateTape &s,const double *by,double *bx) {
    return si_snow_graph_ad(s.g,SI_ICE_SNOWSTATE_NINPUT,SI_ICE_SNOWSTATE_NOUTPUT,s.out,by,bx);
}

// Radiation and albedo before the nested ice_sice/ice_snowseaice calls in
// ice_icetmp, source lines 3832-34, 3846, 3895-3915, 3942-55 and 4000-08.
// Inputs: gsw,old_alb,alb_snow_free,alb_snow,tice1,soilt,stbolt,glw,
// old_emiss,snowfrac,snhei. The caller passes the recorded mosaic and
// keep_snow_albedo flags; these comparisons are not differentiable controls.
// Outputs: albice,gswin,bare_gsw,bare_rnet,active_gsw,active_rnet,alb,
// emiss,snfr. bare_gsw/rnet are valid only when bare_valid is set. The
// active pair is the snowy call when has_snow, otherwise the sole bare call.
// Snow density, snow coverage, and post-call mosaic blending are separate.
enum { SI_ICE_OPTICS_NINPUT=11, SI_ICE_OPTICS_NOUTPUT=9 };
struct SiIceOpticsTape {
    SiSiceClosureTape g;
    int out[SI_ICE_OPTICS_NOUTPUT];
    bool has_snow,mosaic,bare_valid,keep_snow_albedo;
};
__device__ __forceinline__ bool si_icetmp_optics_record(
        const float *x,bool mosaic,bool keep_snow_albedo,SiIceOpticsTape &s,
        unsigned int *branches=nullptr) {
    SiSiceClosureTape &t=s.g;t.n=0;t.valid=true;
    for(int i=0;i<SI_ICE_OPTICS_NINPUT;++i)si_sc_leaf(t,x[i]);
    int zero=si_sc_leaf(t,0.f),one=si_sc_leaf(t,1.f);
    int tenth=si_sc_leaf(t,.1f),twentieth=si_sc_leaf(t,.05f);
    int ten=si_sc_leaf(t,10.f),low=si_sc_leaf(t,263.15f);
    int freeze=si_sc_leaf(t,273.15f),factor=si_sc_leaf(t,.15f);
    int emissn=si_sc_leaf(t,.98f);
    int gswin=si_sc_op(t,0,si_sc_op(t,one,1,1),3);
    int alb_lo=si_sc_op(t,2,twentieth,1);
    int change=si_sc_op(t,si_sc_op(t,tenth,si_sc_op(t,4,ten,0),2),ten,3);
    int albice=si_sc_min_record(t,2,si_sc_max_record(t,alb_lo,si_sc_op(t,2,change,1),branches,65),branches,64);
    int t3=si_sc_op(t,si_sc_op(t,si_sc_op(t,6,5,2),5,2),5,2);
    int upflux=si_sc_op(t,t3,5,2);
    int bare_gsw=zero,bare_rnet=zero,gswnew=0,rnet=zero,alb=1,emiss=8,snfr=one;
    s.has_snow=x[10]>0.f;s.mosaic=s.has_snow&&mosaic;
    s.bare_valid=!s.has_snow||s.mosaic;s.keep_snow_albedo=s.has_snow&&keep_snow_albedo;
    if(s.has_snow) {
        int keep=s.keep_snow_albedo?one:zero,albsn;
        si_branch_pair(branches,34,s.mosaic);
        if(s.mosaic) {albsn=3;emiss=emissn;}
        else {
            int a=si_sc_op(t,albice,si_sc_op(t,si_sc_op(t,3,albice,1),9,2),0);
            albsn=si_sc_max_record(t,si_sc_op(t,keep,3,2),si_sc_min_record(t,a,3,branches,82),branches,81);
            int e=si_sc_op(t,emissn,si_sc_op(t,si_sc_op(t,emissn,emissn,1),9,2),0);
            emiss=si_sc_max_record(t,si_sc_op(t,keep,emissn,2),si_sc_min_record(t,e,emissn,branches,84),branches,83);
        }
        const bool left=t.valid&&t.v[albsn]<x[3];
        const bool selected=t.valid&&(left||s.keep_snow_albedo);
        if(branches) {si_branch(branches,38,selected);si_branch(branches,39,left);
                      if(!left)si_branch(branches,40,s.keep_snow_albedo);}
        if(selected)alb=albsn;
        else {
            int warm=si_sc_op(t,si_sc_op(t,si_sc_op(t,factor,albsn,2),
                                 si_sc_op(t,5,low,1),2),si_sc_op(t,freeze,low,1),3);
            int candidate=si_sc_op(t,albsn,warm,1);
            alb=si_sc_min_record(t,albsn,si_sc_max_record(t,candidate,si_sc_op(t,albsn,tenth,1),branches,86),branches,85);
        }
        si_branch_pair(branches,41,s.mosaic);
        if(s.mosaic) {
            bare_gsw=si_sc_op(t,gswin,si_sc_op(t,one,albice,1),2);
            bare_rnet=si_sc_op(t,bare_gsw,si_sc_op(t,emissn,si_sc_op(t,7,upflux,1),2),0);
        }
        gswnew=si_sc_op(t,gswin,si_sc_op(t,one,alb,1),2);
        rnet=si_sc_op(t,gswnew,si_sc_op(t,emiss,si_sc_op(t,7,upflux,1),2),0);
        snfr=s.mosaic?one:9;
    } else {
        // The native comparison leaves the original gsw untouched on a tie.
        const bool differs=t.valid&&x[1]!=t.v[albice];
        si_branch_pair(branches,60,differs);
        if(differs)gswnew=si_sc_op(t,gswin,si_sc_op(t,one,albice,1),2);
        alb=albice;
        rnet=si_sc_op(t,gswnew,si_sc_op(t,emiss,si_sc_op(t,7,upflux,1),2),0);
        bare_gsw=gswnew;bare_rnet=rnet;
    }
    s.out[0]=albice;s.out[1]=gswin;s.out[2]=bare_gsw;s.out[3]=bare_rnet;
    s.out[4]=gswnew;s.out[5]=rnet;s.out[6]=alb;s.out[7]=emiss;s.out[8]=snfr;
    return t.valid;
}
__device__ __forceinline__ bool si_icetmp_optics_tl(
        const SiIceOpticsTape &s,const double *dx,double *dy) {
    return si_snow_graph_tl(s.g,SI_ICE_OPTICS_NINPUT,SI_ICE_OPTICS_NOUTPUT,s.out,dx,dy);
}
__device__ __forceinline__ bool si_icetmp_optics_ad(
        const SiIceOpticsTape &s,const double *by,double *bx) {
    return si_snow_graph_ad(s.g,SI_ICE_OPTICS_NINPUT,SI_ICE_OPTICS_NOUTPUT,s.out,by,bx);
}

// Post-nested-call optical blend on the executed snow-mosaic arm,
// ice_icetmp source lines 3980-81. Inputs are albice,alb,alb_snow_free,
// snowfrac; keep_snow_albedo is the recorded comparison outcome.
enum { SI_ICE_OPTICS_BLEND_NINPUT=4, SI_ICE_OPTICS_BLEND_NOUTPUT=2 };
struct SiIceOpticsBlendTape {SiSiceClosureTape g;int out[SI_ICE_OPTICS_BLEND_NOUTPUT];};
__device__ __forceinline__ bool si_icetmp_optics_blend_record(
        const float *x,bool keep_snow_albedo,SiIceOpticsBlendTape &s,
        unsigned int *branches=nullptr) {
    SiSiceClosureTape &t=s.g;t.n=0;t.valid=true;
    for(int i=0;i<SI_ICE_OPTICS_BLEND_NINPUT;++i)si_sc_leaf(t,x[i]);
    int keep=si_sc_leaf(t,keep_snow_albedo?1.f:0.f),emissn=si_sc_leaf(t,.98f);
    int candidate=si_sc_op(t,0,si_sc_op(t,si_sc_op(t,1,2,1),3,2),0);
    s.out[0]=si_sc_max_record(t,si_sc_op(t,keep,1,2),si_sc_min_record(t,candidate,1,branches,88),branches,87);
    int e=si_sc_op(t,emissn,si_sc_op(t,si_sc_op(t,emissn,emissn,1),3,2),0);
    s.out[1]=si_sc_max_record(t,si_sc_op(t,keep,emissn,2),si_sc_min_record(t,e,emissn,branches,90),branches,89);
    return t.valid;
}
__device__ __forceinline__ bool si_icetmp_optics_blend_tl(
        const SiIceOpticsBlendTape &s,const double *dx,double *dy) {
    return si_snow_graph_tl(s.g,SI_ICE_OPTICS_BLEND_NINPUT,SI_ICE_OPTICS_BLEND_NOUTPUT,s.out,dx,dy);
}
__device__ __forceinline__ bool si_icetmp_optics_blend_ad(
        const SiIceOpticsBlendTape &s,const double *by,double *bx) {
    return si_snow_graph_ad(s.g,SI_ICE_OPTICS_BLEND_NINPUT,SI_ICE_OPTICS_BLEND_NOUTPUT,s.out,by,bx);
}

// Energy balance on the executed melt arm, lines 4541-4577. qsn is the
// active result of ice_qsn(soiltfrac), so its caller must propagate through
// that interpolation and the selected table entries. This fragment does not
// silently replace it by a constant. Inputs: soiltfrac,qsn,pp,stbolt,tnold,
// emiss,glw,qkms,qvatm,ras,xlvm,rho,beta,d10,tabs,snhei,snth,thdifsn,
// rhocsn,tsob,snprim,fsn,fso,diff1,rhcs,r21,d9sn,r22sn,r210,qgold,rnet,
// rhonewcsn,newsnow,delt,rainf,cw,prcpms,old_dew. Outputs: qsg,epot,dew,eeta,qfx,
// hfx,soh,x,snoh,upflux,xinet. The branch is legal only for positive snow.
enum { SI_SM_ENERGY_NINPUT=38, SI_SM_ENERGY_NOUTPUT=11 };
struct SiSnowMeltEnergyTape {
    SiSiceClosureTape g;
    int out[SI_SM_ENERGY_NOUTPUT];
    bool nonpositive,thick;
};
__device__ __forceinline__ bool si_snow_melt_energy_record(
        const float *x,SiSnowMeltEnergyTape &s, unsigned int *branches=nullptr) {
    SiSiceClosureTape &t=s.g;t.n=0;t.valid=true;
    if(!(x[15]>0.f))return false;
    for(int i=0;i<SI_SM_ENERGY_NINPUT;++i)si_sc_leaf(t,x[i]);
    int zero=si_sc_leaf(t,0.f),half=si_sc_leaf(t,.5f);
    int thousand=si_sc_leaf(t,1.e3f),freeze=si_sc_leaf(t,273.15f);
    int qsg=si_sc_op(t,1,2,3);
    int t3=si_sc_op(t,si_sc_op(t,si_sc_op(t,3,4,2),4,2),4,2);
    int upflux=si_sc_op(t,si_sc_op(t,t3,half,2),si_sc_op(t,4,0,0),2);
    int xinet=si_sc_op(t,5,si_sc_op(t,6,upflux,1),2);
    int epot=si_sc_op(t,si_sc_neg(t,7),si_sc_op(t,8,qsg,1),2);
    int q1=si_sc_op(t,epot,9,2);s.nonpositive=t.valid&&t.v[q1]<=0.f;
    si_branch_pair(branches,53,s.nonpositive);
    int dew=37,eeta,qfx;
    if(s.nonpositive) {
        dew=si_sc_neg(t,epot);
        qfx=si_sc_op(t,si_sc_op(t,10,11,2),dew,2);
        eeta=si_sc_op(t,qfx,10,3);
    } else {
        eeta=si_sc_op(t,si_sc_op(t,q1,12,2),thousand,2);
        qfx=si_sc_op(t,si_sc_neg(t,10),eeta,2);
    }
    int hfx=si_sc_op(t,13,si_sc_op(t,14,0,1),2);
    s.thick=x[15]>=x[16];si_branch_pair(branches,55,s.thick);int thermal;
    if(s.thick) thermal=si_sc_op(t,17,18,2);
    else thermal=si_sc_op(t,si_sc_op(t,si_sc_op(t,21,17,2),18,2),
                            si_sc_op(t,si_sc_op(t,22,23,2),24,2),0);
    int soh=si_sc_op(t,si_sc_op(t,thermal,si_sc_op(t,0,19,1),2),20,3);
    int heat=si_sc_op(t,si_sc_op(t,25,si_sc_op(t,26,27,2),0),
                       si_sc_op(t,0,4,1),2);
    int moisture=si_sc_op(t,si_sc_op(t,10,28,2),si_sc_op(t,qsg,29,1),2);
    int balance=si_sc_op(t,heat,moisture,0);
    int cold=si_sc_min_record(t,freeze,14,branches,118),warm=si_sc_max_record(t,freeze,14,branches,119);
    int news=si_sc_op(t,si_sc_op(t,31,32,2),33,3);
    int rain=si_sc_op(t,si_sc_op(t,34,35,2),36,2);
    int snoh=si_sc_op(t,si_sc_op(t,si_sc_op(t,si_sc_op(t,si_sc_op(t,
        30,qfx,0),hfx,0),si_sc_op(t,news,si_sc_op(t,cold,0,1),2),0),soh,1),balance,1);
    snoh=si_sc_op(t,snoh,si_sc_op(t,rain,si_sc_op(t,warm,0,1),2),0);
    snoh=si_sc_max_record(t,zero,snoh,branches,120);
    s.out[0]=qsg;s.out[1]=epot;s.out[2]=dew;s.out[3]=eeta;s.out[4]=qfx;
    s.out[5]=hfx;s.out[6]=soh;s.out[7]=balance;s.out[8]=snoh;
    s.out[9]=upflux;s.out[10]=xinet;
    return t.valid;
}
__device__ __forceinline__ bool si_snow_melt_energy_tl(
        const SiSnowMeltEnergyTape &s,const double *dx,double *dy) {
    return si_snow_graph_tl(s.g,SI_SM_ENERGY_NINPUT,SI_SM_ENERGY_NOUTPUT,s.out,dx,dy);
}
__device__ __forceinline__ bool si_snow_melt_energy_ad(
        const SiSnowMeltEnergyTape &s,const double *by,double *bx) {
    return si_snow_graph_ad(s.g,SI_SM_ENERGY_NINPUT,SI_SM_ENERGY_NOUTPUT,s.out,by,bx);
}

// Snow melt/refreezing and remaining water, lines 4578-4588. Inputs:
// snoh,snwepr,delt,beta,epot,ras,soilt,snhei. Outputs: smelt,snoh,snodif,
// rsm,snwe. Bounds use ruc_min/ruc_max comparison order, including ties.
enum { SI_SM_WATER_NINPUT=8, SI_SM_WATER_NOUTPUT=5 };
struct SiSnowMeltWaterTape {
    SiSiceClosureTape g;
    int out[SI_SM_WATER_NOUTPUT];
    bool retain;
};
__device__ __forceinline__ bool si_snow_melt_water_record(
        const float *x,SiSnowMeltWaterTape &s, unsigned int *branches=nullptr) {
    SiSiceClosureTape &t=s.g;t.n=0;t.valid=true;
    for(int i=0;i<SI_SM_WATER_NINPUT;++i)si_sc_leaf(t,x[i]);
    int zero=si_sc_leaf(t,0.f),one=si_sc_leaf(t,1.f),melt=si_sc_leaf(t,3.35e5f);
    int milli=si_sc_leaf(t,1.e-3f),thousand=si_sc_leaf(t,1.e3f);
    int freeze=si_sc_leaf(t,273.15f),meltlimit=si_sc_leaf(t,5.6e-8f);
    int low=si_sc_leaf(t,.08f),high=si_sc_leaf(t,.18f);
    int tenth=si_sc_leaf(t,.10f),ratio=si_sc_leaf(t,.13f);
    int loss=si_sc_op(t,si_sc_op(t,3,4,2),5,2);
    int available=si_sc_op(t,si_sc_op(t,1,2,3),loss,1);
    int smelt=si_sc_op(t,si_sc_op(t,0,melt,3),milli,2);
    smelt=si_sc_min_record(t,smelt,available,branches,121);smelt=si_sc_max_record(t,zero,smelt,branches,122);
    int warm=si_sc_max_record(t,one,si_sc_op(t,6,freeze,1),branches,124);
    smelt=si_sc_min_record(t,smelt,si_sc_op(t,meltlimit,warm,2),branches,123);
    smelt=si_sc_min_record(t,smelt,available,branches,125);
    int newheat=si_sc_op(t,si_sc_op(t,smelt,melt,2),thousand,2);
    int snodif=si_sc_max_record(t,zero,si_sc_op(t,0,newheat,1),branches,126);
    int fraction=si_sc_min_record(t,high,si_sc_max_record(t,low,
                    si_sc_op(t,si_sc_op(t,1,tenth,3),ratio,2),branches,128),branches,127);
    s.retain=x[7]>.01f;si_branch_pair(branches,64,s.retain);
    int rsm=zero,rsm_rate=zero;
    if(s.retain) {
        // The source forms fl(fl(rate*delt)/delt). Differentiate both
        // executed operations together. The FMA retains the dependence on
        // delt when the rounded product is not exactly rate*delt; separately
        // rounding the two opposing slopes can erase that dependence.
        int rate=si_sc_op(t,fraction,smelt,2);
        rsm=si_sc_op(t,rate,2,2);
        if(!t.valid||t.n>=SI_SC_NNODE||t.v[2]==0.f){t.valid=false;return false;}
        float quotient=__fdiv_rn(t.v[rsm],t.v[2]);
        if(!si_finite(quotient)){t.valid=false;return false;}
        double delt=double(t.v[2]);
        double slope=fma(double(t.v[rate]),delt,-double(t.v[rsm]))/(delt*delt);
        rsm_rate=t.n++;
        t.v[rsm_rate]=quotient;t.a[rsm_rate]=rate;t.b[rsm_rate]=2;
        t.da[rsm_rate]=1.;t.db[rsm_rate]=slope;
    } else rsm_rate=si_sc_op(t,zero,2,3);
    smelt=si_sc_max_record(t,zero,si_sc_op(t,smelt,rsm_rate,1),branches,129);
    int snwe=si_sc_max_record(t,zero,si_sc_op(t,1,
                  si_sc_op(t,si_sc_op(t,smelt,loss,0),2,2),1),branches,130);
    s.out[0]=smelt;s.out[1]=newheat;s.out[2]=snodif;s.out[3]=rsm;s.out[4]=snwe;
    return t.valid;
}
__device__ __forceinline__ bool si_snow_melt_water_tl(
        const SiSnowMeltWaterTape &s,const double *dx,double *dy) {
    return si_snow_graph_tl(s.g,SI_SM_WATER_NINPUT,SI_SM_WATER_NOUTPUT,s.out,dx,dy);
}
__device__ __forceinline__ bool si_snow_melt_water_ad(
        const SiSnowMeltWaterTape &s,const double *by,double *bx) {
    return si_snow_graph_ad(s.g,SI_SM_WATER_NINPUT,SI_SM_WATER_NOUTPUT,s.out,by,bx);
}

// Post-melt snow density, diagnostics and humidity in ice_snowseaice,
// pinned lines 4599-4631. `qsn` is the result of the separate active ice_qsn
// call at the current soilt; the caller must chain its derivative into soilt
// and the selected table entries. The previous rhocsn/thdifsn and tsnav are
// inputs because the forward leaves them unchanged on some executed arms.
// Inputs: smelt,rsm,snwe,rhosn,snhei,soilt,soilt1,tso1,deltsn,old_tsnav,
// patm,qsn,qkms,qvatm,old_snom,delt,old_rhocsn,old_thdifsn.
// Outputs: rhosn,rhocsn,thdifsn,snweprint,snheiprint,tsnav,pp,qsg,epot,
// dew,snom. `ilnb` is the recorded passive layer count.
enum { SI_SN_POST_NINPUT=18, SI_SN_POST_NOUTPUT=11 };
struct SiSnowPostTape {
    SiSiceClosureTape g;
    int out[SI_SN_POST_NOUTPUT];
    bool refreeze,depth_positive,two_layers,dew;
};
__device__ __forceinline__ bool si_snow_post_record(
        const float *x,int ilnb,SiSnowPostTape &s, unsigned int *branches=nullptr) {
    SiSiceClosureTape &t=s.g;t.n=0;t.valid=true;
    for(int i=0;i<SI_SN_POST_NINPUT;++i)si_sc_leaf(t,x[i]);
    int zero=si_sc_leaf(t,0.f),half=si_sc_leaf(t,.5f),thousand=si_sc_leaf(t,1.e3f);
    int low=si_sc_leaf(t,58.8f),high=si_sc_leaf(t,500.f),heat=si_sc_leaf(t,2090.f);
    int cond=si_sc_leaf(t,.265f),freeze=si_sc_leaf(t,273.15f);
    int density=3,rhocsn=16,thdifsn=17;
    const bool melt_positive=x[0]>0.f,both=melt_positive&&x[1]>0.f;
    if(branches) {
        si_branch(branches,70,both);si_branch(branches,71,melt_positive);
        if(melt_positive)si_branch(branches,72,x[1]>0.f);
    }
    if(both)si_branch_pair(branches,73,x[2]<=x[1]);
    s.refreeze=both&&!(x[2]<=x[1]);
    if(s.refreeze) {
        int numerator=si_sc_op(t,si_sc_op(t,3,si_sc_op(t,2,1,1),2),
                                  si_sc_op(t,thousand,1,2),0);
        int mixed=si_sc_op(t,numerator,2,3);
        density=si_sc_min_record(t,si_sc_max_record(t,low,mixed,branches,133),high,branches,132);
        rhocsn=si_sc_op(t,heat,density,2);thdifsn=si_sc_op(t,cond,rhocsn,3);
    }
    int snheiprint=si_sc_op(t,si_sc_op(t,2,thousand,2),density,3);
    int tsnav=9;s.depth_positive=x[4]>0.f;s.two_layers=s.depth_positive&&ilnb>1;
    si_branch_pair(branches,79,s.depth_positive);
    if(s.depth_positive)si_branch_pair(branches,81,ilnb>1);
    if(s.two_layers) {
        int depth=si_sc_op(t,4,8,1);
        int avg=si_sc_op(t,si_sc_op(t,si_sc_op(t,5,6,0),8,2),
                               si_sc_op(t,si_sc_op(t,6,7,0),depth,2),0);
        tsnav=si_sc_op(t,si_sc_op(t,si_sc_op(t,half,4,3),avg,2),freeze,1);
    } else if(s.depth_positive) {
        tsnav=si_sc_op(t,si_sc_op(t,half,si_sc_op(t,5,7,0),2),freeze,1);
    }
    int pp=si_sc_op(t,10,thousand,2),qsg=si_sc_op(t,11,pp,3);
    int epot=si_sc_op(t,si_sc_neg(t,12),si_sc_op(t,13,qsg,1),2);
    s.dew=t.valid&&t.v[epot]<0.f;si_branch_pair(branches,83,s.dew);
    int dew=s.dew?si_sc_neg(t,epot):zero;
    int snom=si_sc_op(t,14,si_sc_op(t,si_sc_op(t,0,15,2),thousand,2),0);
    s.out[0]=density;s.out[1]=rhocsn;s.out[2]=thdifsn;s.out[3]=2;
    s.out[4]=snheiprint;s.out[5]=tsnav;s.out[6]=pp;s.out[7]=qsg;
    s.out[8]=epot;s.out[9]=dew;s.out[10]=snom;
    return t.valid;
}
__device__ __forceinline__ bool si_snow_post_tl(
        const SiSnowPostTape &s,const double *dx,double *dy) {
    const SiSiceClosureTape &t=s.g;if(!t.valid)return false;double d[SI_SC_NNODE];
    for(int i=0;i<t.n;++i) {
        if(i<SI_SN_POST_NINPUT)d[i]=dx[i];
        else if(t.a[i]<0)d[i]=0.;
        else d[i]=t.da[i]*d[t.a[i]]+t.db[i]*d[t.b[i]];
    }
    for(int i=0;i<SI_SN_POST_NOUTPUT;++i)dy[i]=d[s.out[i]];
    return true;
}
__device__ __forceinline__ bool si_snow_post_ad(
        const SiSnowPostTape &s,const double *by,double *bx) {
    const SiSiceClosureTape &t=s.g;if(!t.valid)return false;double b[SI_SC_NNODE];
    for(int i=0;i<t.n;++i)b[i]=0.;
    for(int i=0;i<SI_SN_POST_NOUTPUT;++i)b[s.out[i]]+=by[i];
    for(int i=t.n-1;i>=SI_SN_POST_NINPUT;--i)if(t.a[i]>=0) {
        b[t.a[i]]+=t.da[i]*b[i];b[t.b[i]]+=t.db[i]*b[i];
    }
    for(int i=0;i<SI_SN_POST_NINPUT;++i)bx[i]+=b[i];
    return true;
}

// Heat and moisture fluxes from ice_snowseaice lines 4632-4657. `pow_value`
// is the active result of the pinned pressure/rovcp power call; its caller
// chains that result back to pressure and rovcp. The MYJ choice is passive.
// Inputs: stbolt,tnold,soilt,emiss,glw,tkms,cp,rho,tabs,pow_value,qkms,
// ras,qvatm,qsg,qvg,beta,xlv,old_dew. Outputs: upflux,xinet,hft,hfx,q1,
// dew,eeta,qfx,sublim,xlvm. The separate old_dew input matters on the MYJ
// condensation arm, where the source does not overwrite it before reuse.
enum { SI_SN_FLUX_NINPUT=18, SI_SN_FLUX_NOUTPUT=10 };
struct SiSnowFluxTape {
    SiSiceClosureTape g;
    int out[SI_SN_FLUX_NOUTPUT];
    bool condensation,myj;
};
__device__ __forceinline__ bool si_snow_flux_record(
        const float *x,bool myj,SiSnowFluxTape &s, unsigned int *branches=nullptr) {
    SiSiceClosureTape &t=s.g;t.n=0;t.valid=true;s.myj=myj;
    for(int i=0;i<SI_SN_FLUX_NINPUT;++i)si_sc_leaf(t,x[i]);
    int one=si_sc_leaf(t,1.f),half=si_sc_leaf(t,.5f),thousand=si_sc_leaf(t,1.e3f);
    int melt=si_sc_leaf(t,3.35e5f),xlvm=si_sc_op(t,16,melt,0);
    int t3=si_sc_op(t,si_sc_op(t,si_sc_op(t,0,1,2),1,2),1,2);
    int upflux=si_sc_op(t,si_sc_op(t,t3,half,2),si_sc_op(t,2,1,0),2);
    int xinet=si_sc_op(t,3,si_sc_op(t,4,upflux,1),2);
    int hft=si_sc_op(t,si_sc_op(t,si_sc_op(t,si_sc_neg(t,5),6,2),7,2),
                       si_sc_op(t,8,2,1),2);
    int hfx=si_sc_op(t,hft,9,2);
    int qbase=si_sc_op(t,si_sc_neg(t,10),11,2);
    int q1=si_sc_op(t,qbase,si_sc_op(t,12,13,1),2);
    s.condensation=t.valid&&t.v[q1]<0.f;si_branch_pair(branches,85,s.condensation);
    if(branches)si_branch(branches,s.condensation?87:88,myj);
    int dew=17,eeta,qfx,sublim;
    if(s.condensation) {
        int heat_eeta;
        if(myj) {
            int qa=si_sc_op(t,12,si_sc_op(t,one,12,0),3);
            int qb=si_sc_op(t,13,si_sc_op(t,one,13,0),3);
            heat_eeta=si_sc_op(t,si_sc_op(t,qbase,si_sc_op(t,qa,qb,1),2),thousand,2);
        } else {
            dew=si_sc_op(t,10,si_sc_op(t,12,13,1),2);
            heat_eeta=si_sc_op(t,si_sc_neg(t,7),dew,2);
        }
        qfx=si_sc_op(t,xlvm,heat_eeta,2);
        eeta=si_sc_op(t,si_sc_neg(t,7),dew,2);sublim=eeta;
    } else {
        int heat_eeta;
        if(myj) {
            int qa=si_sc_op(t,12,si_sc_op(t,one,12,0),3);
            int qb=si_sc_op(t,14,si_sc_op(t,one,14,0),3);
            heat_eeta=si_sc_op(t,si_sc_op(t,si_sc_op(t,qbase,15,2),
                                             si_sc_op(t,qa,qb,1),2),thousand,2);
        } else heat_eeta=si_sc_op(t,si_sc_op(t,q1,15,2),thousand,2);
        qfx=si_sc_op(t,xlvm,heat_eeta,2);
        eeta=si_sc_op(t,si_sc_op(t,q1,15,2),thousand,2);sublim=eeta;
    }
    s.out[0]=upflux;s.out[1]=xinet;s.out[2]=hft;s.out[3]=hfx;s.out[4]=q1;
    s.out[5]=dew;s.out[6]=eeta;s.out[7]=qfx;s.out[8]=sublim;s.out[9]=xlvm;
    return t.valid;
}
__device__ __forceinline__ bool si_snow_flux_tl(
        const SiSnowFluxTape &s,const double *dx,double *dy) {
    const SiSiceClosureTape &t=s.g;if(!t.valid)return false;double d[SI_SC_NNODE];
    for(int i=0;i<t.n;++i) {
        if(i<SI_SN_FLUX_NINPUT)d[i]=dx[i];
        else if(t.a[i]<0)d[i]=0.;
        else d[i]=t.da[i]*d[t.a[i]]+t.db[i]*d[t.b[i]];
    }
    for(int i=0;i<SI_SN_FLUX_NOUTPUT;++i)dy[i]=d[s.out[i]];
    return true;
}
__device__ __forceinline__ bool si_snow_flux_ad(
        const SiSnowFluxTape &s,const double *by,double *bx) {
    const SiSiceClosureTape &t=s.g;if(!t.valid)return false;double b[SI_SC_NNODE];
    for(int i=0;i<t.n;++i)b[i]=0.;
    for(int i=0;i<SI_SN_FLUX_NOUTPUT;++i)b[s.out[i]]+=by[i];
    for(int i=t.n-1;i>=SI_SN_FLUX_NINPUT;--i)if(t.a[i]>=0) {
        b[t.a[i]]+=t.da[i]*b[i];b[t.b[i]]+=t.db[i]*b[i];
    }
    for(int i=0;i<SI_SN_FLUX_NINPUT;++i)bx[i]+=b[i];
    return true;
}

// Final conductive flux and energy diagnostic in ice_snowseaice,
// pinned lines 4658-4697. `old_s` is retained on the bare-ice arm because
// the original writes snflx there but leaves s unchanged. The snow depth used
// for this arm is the input depth, before the later snwe/rhosn update.
// Inputs: snhei,snth,thdifsn,rhocsn,soilt,tsob,snprim,fsn,fso,diff1,rhcs,
// d9sn,old_s,snwe,rhosn,r21,r22sn,tnold,xlv,r210,qsg,qgold,rhonewsn,
// newsnow,delt,tabs,rainf,cw,prcpms,rnet,hft,eeta,snoh,old_tsnav,alb,
// emiss,znt. Outputs: s,snflx,new_snhei,x,icemelt,fltot,tsnav,alb,emiss,znt.
enum { SI_SN_ENERGY_NINPUT=37, SI_SN_ENERGY_NOUTPUT=10 };
struct SiSnowEnergyTape {
    SiSiceClosureTape g;
    int out[SI_SN_ENERGY_NOUTPUT],arm;
    bool warm,cold,now_bare;
};
__device__ __forceinline__ bool si_snow_energy_record(
        const float *x,SiSnowEnergyTape &s, unsigned int *branches=nullptr) {
    SiSiceClosureTape &t=s.g;t.n=0;t.valid=true;
    for(int i=0;i<SI_SN_ENERGY_NINPUT;++i)si_sc_leaf(t,x[i]);
    int thousand=si_sc_leaf(t,1.e3f),freeze=si_sc_leaf(t,273.15f);
    int heat=si_sc_leaf(t,2090.f),melt=si_sc_leaf(t,3.35e5f);
    int alb_bare=si_sc_leaf(t,.55f),emiss_bare=si_sc_leaf(t,.98f);
    int znt_bare=si_sc_leaf(t,.011f),xls=si_sc_op(t,18,melt,0);
    int surface=12,snflx;
    const bool thick=x[0]>=x[1];si_branch_pair(branches,89,thick);
    if(!thick&&branches) {
        const bool left=x[0]<x[1];
        si_branch(branches,91,left&&x[0]>0.f);si_branch(branches,92,left);
        if(left)si_branch(branches,93,x[0]>0.f);
    }
    if(thick) {
        s.arm=2;
        surface=si_sc_op(t,si_sc_op(t,si_sc_op(t,2,3,2),
                                      si_sc_op(t,4,5,1),2),6,3);
        snflx=surface;
    } else if(x[0]<x[1]&&x[0]>0.f) {
        s.arm=1;
        int mix=si_sc_op(t,si_sc_op(t,si_sc_op(t,7,2,2),3,2),
                            si_sc_op(t,si_sc_op(t,8,9,2),10,2),0);
        surface=si_sc_op(t,si_sc_op(t,mix,si_sc_op(t,4,5,1),2),6,3);
        snflx=surface;
    } else {
        s.arm=0;snflx=si_sc_op(t,11,si_sc_op(t,4,5,1),2);
    }
    int snhei=si_sc_op(t,si_sc_op(t,13,thousand,2),14,3);
    int term0=si_sc_op(t,si_sc_op(t,15,si_sc_op(t,11,16,2),0),
                          si_sc_op(t,4,17,1),2);
    int term1=si_sc_op(t,si_sc_op(t,xls,19,2),si_sc_op(t,20,21,1),2);
    int balance=si_sc_op(t,term0,term1,0);
    s.warm=!(273.15f>x[25]);s.cold=!(273.15f<x[25]);
    if(branches) {si_branch(branches,134,!s.cold);si_branch(branches,135,!s.warm);}
    int warm=s.warm?25:freeze,cold=s.cold?25:freeze;
    int news=si_sc_op(t,si_sc_op(t,si_sc_op(t,heat,22,2),23,2),24,3);
    int rain=si_sc_op(t,si_sc_op(t,26,27,2),28,2);
    balance=si_sc_op(t,si_sc_op(t,balance,
                  si_sc_op(t,news,si_sc_op(t,cold,4,1),2),1),
                  si_sc_op(t,rain,si_sc_op(t,warm,4,1),2),1);
    int energy=si_sc_op(t,si_sc_op(t,si_sc_op(t,si_sc_op(t,
                  si_sc_op(t,29,30,1),si_sc_op(t,xls,31,2),1),
                  surface,1),32,1),balance,1);
    int fltot=si_sc_op(t,energy,energy,1);
    s.now_bare=t.valid&&t.v[snhei]==0.f;si_branch_pair(branches,102,s.now_bare);
    int tsnav=s.now_bare?si_sc_op(t,4,freeze,1):33;
    int alb=s.now_bare?alb_bare:34,emiss=s.now_bare?emiss_bare:35;
    int znt=s.now_bare?znt_bare:36;
    s.out[0]=surface;s.out[1]=snflx;s.out[2]=snhei;s.out[3]=balance;
    s.out[4]=energy;s.out[5]=fltot;s.out[6]=tsnav;s.out[7]=alb;
    s.out[8]=emiss;s.out[9]=znt;
    return t.valid;
}
__device__ __forceinline__ bool si_snow_energy_tl(
        const SiSnowEnergyTape &s,const double *dx,double *dy) {
    const SiSiceClosureTape &t=s.g;if(!t.valid)return false;double d[SI_SC_NNODE];
    for(int i=0;i<t.n;++i) {
        if(i<SI_SN_ENERGY_NINPUT)d[i]=dx[i];
        else if(t.a[i]<0)d[i]=0.;
        else d[i]=t.da[i]*d[t.a[i]]+t.db[i]*d[t.b[i]];
    }
    for(int i=0;i<SI_SN_ENERGY_NOUTPUT;++i)dy[i]=d[s.out[i]];
    return true;
}
__device__ __forceinline__ bool si_snow_energy_ad(
        const SiSnowEnergyTape &s,const double *by,double *bx) {
    const SiSiceClosureTape &t=s.g;if(!t.valid)return false;double b[SI_SC_NNODE];
    for(int i=0;i<t.n;++i)b[i]=0.;
    for(int i=0;i<SI_SN_ENERGY_NOUTPUT;++i)b[s.out[i]]+=by[i];
    for(int i=t.n-1;i>=SI_SN_ENERGY_NINPUT;--i)if(t.a[i]>=0) {
        b[t.a[i]]+=t.da[i]*b[i];b[t.b[i]]+=t.db[i]*b[i];
    }
    for(int i=0;i<SI_SN_ENERGY_NINPUT;++i)bx[i]+=b[i];
    return true;
}

// ice_snowseaice lines 4491-4523: reconstruct the snow/ice temperatures
// following the selected snow-depth arm. The caller retains c, r and the
// reconstructed temperatures; pass has one byte per ice layer. Branches are
// recorded from the primal comparisons, including ruc_min's right-hand tie.
// This is a composition building block, not the complete snow routine.
struct SiSnowBackInput {
    int n;
    float snhei,snth,deltsn,fso,soilt,cotsn,rhtsn;
    const float *c,*r;
};
struct SiSnowBackTape {
    int arm; // 0: no snow, 1: shallow, 2: one layer, 3: two layers
    bool pass_soilt1;
    unsigned char *pass;
    float *tso;
    float soilt1,tsob;
};
__device__ __forceinline__ float si_snow_ice_min(float x,unsigned char &pass) {
    pass=!(271.4f<x); return pass?x:271.4f;
}
__device__ __forceinline__ bool si_snow_back_record(
        const SiSnowBackInput &in,SiSnowBackTape &t) {
    const int n=in.n;if(n<3)return false;
    t.pass_soilt1=false;
    if(in.snhei>=in.snth) {
        if(in.snhei>__fadd_rn(in.deltsn,in.snth)) {
            t.arm=3;
            float q=__fadd_rn(in.rhtsn,__fmul_rn(in.cotsn,in.soilt));
            t.pass_soilt1=!(273.15f<q);
            t.soilt1=t.pass_soilt1?q:273.15f;
            t.tso[0]=si_snow_ice_min(__fadd_rn(in.r[n-1],
                                __fmul_rn(in.c[n-1],t.soilt1)),t.pass[0]);
            t.tsob=t.soilt1;
        } else {
            t.arm=2;
            t.tso[0]=si_snow_ice_min(__fadd_rn(in.r[n-1],
                                __fmul_rn(in.c[n-1],in.soilt)),t.pass[0]);
            t.soilt1=t.tso[0];t.tsob=t.tso[0];
        }
    } else if(in.snhei>0.f && in.snhei<in.snth) {
        t.arm=1;
        t.tso[1]=si_snow_ice_min(__fadd_rn(in.r[n-2],
                               __fmul_rn(in.c[n-2],in.soilt)),t.pass[1]);
        t.tso[0]=si_snow_ice_min(__fadd_rn(t.tso[1],
                      __fmul_rn(__fsub_rn(in.soilt,t.tso[1]),in.fso)),t.pass[0]);
        t.soilt1=t.tso[0];t.tsob=t.tso[1];
    } else {
        t.arm=0;t.tso[0]=si_snow_ice_min(in.soilt,t.pass[0]);
        t.soilt1=t.tso[0];t.tsob=t.tso[0];
    }
    int start=t.arm==1?2:1;
    for(int k=start;k<n;++k) {
        int q=n-k-1;
        t.tso[k]=si_snow_ice_min(__fadd_rn(in.r[q],
                         __fmul_rn(in.c[q],t.tso[k-1])),t.pass[k]);
    }
    return true;
}
__device__ __forceinline__ bool si_snow_back_tl(
        const SiSnowBackInput &in,const SiSnowBackTape &t,
        double dfso,double dsoilt,double dcotsn,double drhtsn,
        const double *dc,const double *dr,double *dtso,
        double &dsoilt1,double &dtsob) {
    const int n=in.n;if(n<3)return false;
    if(t.arm==3) {
        dsoilt1=t.pass_soilt1?(drhtsn+dcotsn*double(in.soilt)+
                                    double(in.cotsn)*dsoilt):0.;
        dtso[0]=t.pass[0]?(dr[n-1]+dc[n-1]*double(t.soilt1)+
                                    double(in.c[n-1])*dsoilt1):0.;
        dtsob=dsoilt1;
    } else if(t.arm==2) {
        dtso[0]=t.pass[0]?(dr[n-1]+dc[n-1]*double(in.soilt)+
                                    double(in.c[n-1])*dsoilt):0.;
        dsoilt1=dtso[0];dtsob=dtso[0];
    } else if(t.arm==1) {
        dtso[1]=t.pass[1]?(dr[n-2]+dc[n-2]*double(in.soilt)+
                                    double(in.c[n-2])*dsoilt):0.;
        dtso[0]=t.pass[0]?(dtso[1]+(dsoilt-dtso[1])*double(in.fso)+
                  double(__fsub_rn(in.soilt,t.tso[1]))*dfso):0.;
        dsoilt1=dtso[0];dtsob=dtso[1];
    } else {
        dtso[0]=t.pass[0]?dsoilt:0.;dsoilt1=dtso[0];dtsob=dtso[0];
    }
    int start=t.arm==1?2:1;
    for(int k=start;k<n;++k) {
        int q=n-k-1;
        dtso[k]=t.pass[k]?(dr[q]+dc[q]*double(t.tso[k-1])+
                                      double(in.c[q])*dtso[k-1]):0.;
    }
    return true;
}
// btso is caller-owned reverse scratch. The routine consumes it and accumulates
// cotangents into the caller's input buffers, avoiding atomics on shared fields.
__device__ __forceinline__ bool si_snow_back_ad(
        const SiSnowBackInput &in,const SiSnowBackTape &t,
        double &bfso,double &bsoilt,double &bcotsn,double &brhtsn,
        double *bc,double *br,double *btso,double bsoilt1,double btsob) {
    const int n=in.n;if(n<3)return false;
    int start=t.arm==1?2:1;
    for(int k=n-1;k>=start;--k) {
        int q=n-k-1;double b=btso[k];btso[k]=0.;
        if(t.pass[k]) {
            br[q]+=b;bc[q]+=b*double(t.tso[k-1]);btso[k-1]+=b*double(in.c[q]);
        }
    }
    if(t.arm==3) {
        double b=btso[0];btso[0]=0.;
        if(t.pass[0]) {br[n-1]+=b;bc[n-1]+=b*double(t.soilt1);
                        bsoilt1+=b*double(in.c[n-1]);}
        bsoilt1+=btsob;
        if(t.pass_soilt1) {brhtsn+=bsoilt1;bcotsn+=bsoilt1*double(in.soilt);
                            bsoilt+=bsoilt1*double(in.cotsn);}
    } else if(t.arm==2) {
        double b=btso[0]+bsoilt1+btsob;btso[0]=0.;
        if(t.pass[0]) {br[n-1]+=b;bc[n-1]+=b*double(in.soilt);
                        bsoilt+=b*double(in.c[n-1]);}
    } else if(t.arm==1) {
        double b=btso[0]+bsoilt1;btso[0]=0.;btso[1]+=btsob;
        if(t.pass[0]) {
            btso[1]+=b*(1.-double(in.fso));bsoilt+=b*double(in.fso);
            bfso+=b*double(__fsub_rn(in.soilt,t.tso[1]));
        }
        b=btso[1];btso[1]=0.;
        if(t.pass[1]) {br[n-2]+=b;bc[n-2]+=b*double(in.soilt);
                        bsoilt+=b*double(in.c[n-2]);}
    } else {
        double b=btso[0]+bsoilt1+btsob;btso[0]=0.;
        if(t.pass[0])bsoilt+=b;
    }
    return true;
}

// ice_sice lines 4139-4145: reconstruct the column from the eliminated
// system. The parent records pass[k] from the executed min arm and keeps the
// reconstructed values until reverse. ruc_min(a,b) returns b on a tie.
// This is a composition building block, not a complete ice_sice derivative.
__device__ __forceinline__ bool si_sice_backsub_tl(
        int nzs,float ts1,float dts1,const float *c,const float *r,
        const float *dc,const float *dr,float *tso,float *dtso,
        unsigned char *pass) {
    if(nzs<1||!si_finite(ts1)||!si_finite(dts1))return false;
    pass[0]=!(271.4f<ts1);
    tso[0]=pass[0]?ts1:271.4f;dtso[0]=pass[0]?dts1:0.f;
    for(int k=1;k<nzs;++k) {
        int q=nzs-k-1;
        float v=__fadd_rn(r[q],__fmul_rn(c[q],tso[k-1]));
        float dv=__fadd_rn(dr[q],__fadd_rn(__fmul_rn(dc[q],tso[k-1]),
                                             __fmul_rn(c[q],dtso[k-1])));
        if(!si_finite(v)||!si_finite(dv))return false;
        pass[k]=!(271.4f<v);
        tso[k]=pass[k]?v:271.4f;dtso[k]=pass[k]?dv:0.f;
    }
    return true;
}

__device__ __forceinline__ bool si_sice_backsub_ad(
        int nzs,const float *c,const float *tso,const unsigned char *pass,
        double *btso,double *bc,double *br,double &bts1) {
    if(nzs<1)return false;
    for(int k=nzs-1;k>=1;--k) {
        int q=nzs-k-1;
        double b=btso[k];btso[k]=0.;
        if(pass[k]) {
            br[q]+=b;bc[q]+=b*double(tso[k-1]);btso[k-1]+=b*double(c[q]);
        }
    }
    if(pass[0])bts1+=btso[0];
    btso[0]=0.;
    return true;
}

// Full single-call ice_sice arithmetic, composed from the layer and closure
// rules above. This device interface is private to the RUC composition: the
// caller owns all variable-length work buffers, saves the exact gfk_pow result
// and the complete Vilka index trace, and retains the unmodified input arrays.
// It is not yet the FRAMEWORK launch entry or a complete component proof.
// Scalar slots are ordinary continuous inputs; the selected table entries are
// supplied separately to the tangent and returned separately by the adjoint.
enum SiSiceInputSlot {
 SI_X_DELT,SI_X_CONFLX,SI_X_PRCPMS,SI_X_RAINF,SI_X_PATM,SI_X_QVATM,
 SI_X_GLW,SI_X_EMISS,SI_X_RNET,SI_X_QKMS,SI_X_TKMS,SI_X_RHO,
 SI_X_CP,SI_X_ROVCP,SI_X_CW,SI_X_STBOLT,SI_X_TABS,SI_X_SOILT,
 SI_X_QVG,SI_X_QSG,SI_X_ZMAIN0,SI_X_ZMAIN1,SI_SICE_NX
};
enum SiSiceOutputSlot {
 SI_Y_QVG,SI_Y_QSG,SI_Y_QCG,SI_Y_SOILT,SI_Y_DEW,SI_Y_EETA,
 SI_Y_QFX,SI_Y_HFX,SI_Y_S,SI_Y_EVAPL,SI_Y_PRCPL,SI_Y_FLTOT,SI_SICE_NY
};
struct SiSiceInput {
 const float *x,*old_tso,*diff,*capacity,*dz,*table;
 const int *visits;
 int nzs,nvisits;
 bool myj;
 float pow_value; // exact saved gfk_pow result, differentiated below
};
struct SiSiceRecord {
 float *c,*r,*new_tso;
 unsigned char *pass;
 float dzstop,denz,qs,ts,pow_base,y[SI_SICE_NY];
 int table_index;
 SiVilkaPoint vilka;
 SiSiceClosureTape closure;
 SiSiceFluxTape flux;
};

__device__ __forceinline__ bool si_sice_record(const SiSiceInput &in,SiSiceRecord &t) {
 const int n=in.nzs;const float *x=in.x;
 if(n<3||in.nvisits<2||!si_finite(in.pow_value))return false;
 for(int k=0;k<SI_SICE_NX;++k)if(!si_finite(x[k]))return false;
 t.denz=__fsub_rn(x[SI_X_ZMAIN1],x[SI_X_ZMAIN0]);
 if(t.denz==0.f||x[SI_X_PATM]<=0.f)return false;
 t.dzstop=__fdiv_rn(1.f,t.denz);
 t.pow_base=__fdiv_rn(__fmul_rn(100000.f,.00001f),x[SI_X_PATM]);
 if(!si_finite(t.dzstop)||!si_finite(t.pow_base)||t.pow_base<=0.f)return false;
 t.c[0]=0.f;t.r[0]=in.old_tso[n-1];
 for(int k=1;k<=n-2;++k) {
   int kn=n-k,j=kn-1,a=2*kn-4,b=a+1;
   float x1=__fmul_rn(in.dz[a],in.diff[j-1]);
   float x2=__fmul_rn(in.dz[b],in.diff[j]);
   float ft=__fsub_rn(__fadd_rn(in.old_tso[j],
            __fmul_rn(x1,__fsub_rn(in.old_tso[j-1],in.old_tso[j]))),
            __fmul_rn(x2,__fsub_rn(in.old_tso[j],in.old_tso[j+1])));
   float den=__fsub_rn(__fadd_rn(__fadd_rn(1.f,x1),x2),__fmul_rn(x2,t.c[k-1]));
   if(!si_finite(den)||den==0.f)return false;
   t.c[k]=__fdiv_rn(x1,den);
   t.r[k]=__fdiv_rn(__fadd_rn(ft,__fmul_rn(x2,t.r[k-1])),den);
   if(!si_finite(t.c[k])||!si_finite(t.r[k]))return false;
 }
 float cx[SI_SC_NINPUT]={in.diff[0],in.capacity[0],t.dzstop,x[SI_X_TKMS],
  x[SI_X_CP],x[SI_X_RHO],x[SI_X_CONFLX],x[SI_X_DELT],x[SI_X_EMISS],
  x[SI_X_STBOLT],x[SI_X_SOILT],t.c[n-2],t.r[n-2],x[SI_X_RNET],
  x[SI_X_RAINF],x[SI_X_CW],x[SI_X_PRCPMS],x[SI_X_QKMS],x[SI_X_TABS],
  x[SI_X_QVATM],x[SI_X_QVG],x[SI_X_PATM]};
 if(!si_sice_closure_record(cx,t.closure))return false;
 const float aa=t.closure.v[t.closure.out[2]],bb=t.closure.v[t.closure.out[3]],
             pp=t.closure.v[t.closure.out[4]];
 if(!si_vilka_replay(x[SI_X_SOILT],aa,bb,in.table,in.visits,in.nvisits,t.table_index)||
    !si_vilka_point(t.table_index,aa,bb,pp,in.table,t.vilka))return false;
 t.qs=__fdiv_rn(t.vilka.num,pp);
 t.ts=__fsub_rn(t.vilka.t1,__fmul_rn(.05f,t.vilka.rn));
 t.pass[0]=!(271.4f<t.ts);t.new_tso[0]=t.pass[0]?t.ts:271.4f;
 for(int k=1;k<n;++k) {
   int q=n-k-1;
   float v=__fadd_rn(t.r[q],__fmul_rn(t.c[q],t.new_tso[k-1]));
   if(!si_finite(v))return false;
   t.pass[k]=!(271.4f<v);t.new_tso[k]=t.pass[k]?v:271.4f;
 }
 float fx[SI_SF_NINPUT]={x[SI_X_SOILT],t.new_tso[0],x[SI_X_TABS],x[SI_X_STBOLT],
  x[SI_X_EMISS],x[SI_X_GLW],x[SI_X_TKMS],x[SI_X_CP],x[SI_X_RHO],x[SI_X_QKMS],
  x[SI_X_QVATM],t.qs,t.qs,x[SI_X_RNET],in.diff[0],in.capacity[0],t.dzstop,
  t.new_tso[0],t.new_tso[1],t.closure.v[t.closure.out[1]],x[SI_X_DELT],
  x[SI_X_ZMAIN1],x[SI_X_RAINF],x[SI_X_CW],x[SI_X_PRCPMS],x[SI_X_QSG],in.pow_value};
 if(!si_sice_flux_record(fx,in.myj,t.flux))return false;
 t.y[SI_Y_QVG]=t.qs;t.y[SI_Y_QSG]=t.qs;t.y[SI_Y_QCG]=0.f;
 t.y[SI_Y_SOILT]=t.new_tso[0];
 for(int k=0;k<SI_SF_NOUTPUT;++k)t.y[SI_Y_DEW+k]=t.flux.g.v[t.flux.out[k]];
 return si_finite(t.qs)&&si_finite(t.ts);
}

// Tangent uses double accumulators to transpose consistently when several
// references in the native signature name the same state slot.
__device__ __forceinline__ bool si_sice_tl(const SiSiceInput &in,const SiSiceRecord &t,
 const double *dx,const double *dold,const double *ddiff,const double *dcap,
 const double *ddz,double dtable0,double dtable1,double *dc,double *dr,
 double *dnew,double *dy) {
 const int n=in.nzs;const float *x=in.x;dc[0]=0.;dr[0]=dold[n-1];
 for(int k=1;k<=n-2;++k) {
  int kn=n-k,j=kn-1,a=2*kn-4,b=a+1;
  float x1=__fmul_rn(in.dz[a],in.diff[j-1]),x2=__fmul_rn(in.dz[b],in.diff[j]);
  float den=__fsub_rn(__fadd_rn(__fadd_rn(1.f,x1),x2),__fmul_rn(x2,t.c[k-1]));
  double dx1=ddz[a]*double(in.diff[j-1])+double(in.dz[a])*ddiff[j-1];
  double dx2=ddz[b]*double(in.diff[j])+double(in.dz[b])*ddiff[j];
  double dft=dold[j]+dx1*double(__fsub_rn(in.old_tso[j-1],in.old_tso[j]))+
    double(x1)*(dold[j-1]-dold[j])-dx2*double(__fsub_rn(in.old_tso[j],in.old_tso[j+1]))-
    double(x2)*(dold[j]-dold[j+1]);
  double dden=dx1+dx2-dx2*double(t.c[k-1])-double(x2)*dc[k-1];
  dc[k]=(dx1-double(t.c[k])*dden)/double(den);
  dr[k]=(dft+dx2*double(t.r[k-1])+double(x2)*dr[k-1]-double(t.r[k])*dden)/double(den);
 }
 double ddzstop=-double(t.dzstop)/double(t.denz)*(dx[SI_X_ZMAIN1]-dx[SI_X_ZMAIN0]);
 double cx[SI_SC_NINPUT]={ddiff[0],dcap[0],ddzstop,dx[SI_X_TKMS],dx[SI_X_CP],
  dx[SI_X_RHO],dx[SI_X_CONFLX],dx[SI_X_DELT],dx[SI_X_EMISS],dx[SI_X_STBOLT],
  dx[SI_X_SOILT],dc[n-2],dr[n-2],dx[SI_X_RNET],dx[SI_X_RAINF],dx[SI_X_CW],
  dx[SI_X_PRCPMS],dx[SI_X_QKMS],dx[SI_X_TABS],dx[SI_X_QVATM],dx[SI_X_QVG],dx[SI_X_PATM]};
 double cy[SI_SC_NOUTPUT];if(!si_sice_closure_tl(t.closure,cx,cy))return false;
 const SiVilkaPoint &q=t.vilka;
 double df=cy[2]*double(q.a)+double(t.closure.v[t.closure.out[2]])*dtable0-cy[3];
 double dd=cy[2]*double(__fsub_rn(q.b,q.a))+
            double(t.closure.v[t.closure.out[2]])*(dtable1-dtable0);
 double drn=(df-double(q.rn)*dd)/double(q.den);
 double dnum=dtable0+(dtable0-dtable1)*double(q.rn)+double(__fsub_rn(q.a,q.b))*drn;
 double dqs=(dnum-double(t.qs)*cy[4])/double(q.pp),dts=-double(.05f)*drn;
 dnew[0]=t.pass[0]?dts:0.;
 for(int k=1;k<n;++k) {int j=n-k-1;dnew[k]=t.pass[k]?
   dr[j]+dc[j]*double(t.new_tso[k-1])+double(t.c[j])*dnew[k-1]:0.;}
 double dpow=double(in.pow_value)*(log(double(t.pow_base))*dx[SI_X_ROVCP]-
                    double(x[SI_X_ROVCP])/double(x[SI_X_PATM])*dx[SI_X_PATM]);
 double fx[SI_SF_NINPUT]={dx[SI_X_SOILT],dnew[0],dx[SI_X_TABS],dx[SI_X_STBOLT],
  dx[SI_X_EMISS],dx[SI_X_GLW],dx[SI_X_TKMS],dx[SI_X_CP],dx[SI_X_RHO],dx[SI_X_QKMS],
  dx[SI_X_QVATM],dqs,dqs,dx[SI_X_RNET],ddiff[0],dcap[0],ddzstop,dnew[0],dnew[1],cy[1],
  dx[SI_X_DELT],dx[SI_X_ZMAIN1],dx[SI_X_RAINF],dx[SI_X_CW],dx[SI_X_PRCPMS],dx[SI_X_QSG],dpow};
 double fy[SI_SF_NOUTPUT];if(!si_sice_flux_tl(t.flux,fx,fy))return false;
 dy[SI_Y_QVG]=dqs;dy[SI_Y_QSG]=dqs;dy[SI_Y_QCG]=0.;dy[SI_Y_SOILT]=dnew[0];
 for(int k=0;k<SI_SF_NOUTPUT;++k)dy[SI_Y_DEW+k]=fy[k];
 return true;
}

// bx, bold, bdiff, bcap and bdz are accumulation destinations; bc, br and
// bnew are scratch. Table cotangents are per call; the owner must reduce them
// in a deterministic order at the two selected indices.
__device__ __forceinline__ bool si_sice_ad(const SiSiceInput &in,const SiSiceRecord &t,
 const double *by,const double *by_tso,double *bx,double *bold,double *bdiff,
 double *bcap,double *bdz,double &btable0,double &btable1,
 double *bc,double *br,double *bnew) {
 const int n=in.nzs;const float *x=in.x;
 for(int k=0;k<n;++k){bc[k]=0.;br[k]=0.;bnew[k]=by_tso[k];}
 double fy[SI_SF_NOUTPUT],fx[SI_SF_NINPUT]={};
 for(int k=0;k<SI_SF_NOUTPUT;++k)fy[k]=by[SI_Y_DEW+k];
 if(!si_sice_flux_ad(t.flux,fy,fx))return false;
 bx[SI_X_SOILT]+=fx[0];bnew[0]+=fx[1]+fx[17]+by[SI_Y_SOILT];bnew[1]+=fx[18];
 bx[SI_X_TABS]+=fx[2];bx[SI_X_STBOLT]+=fx[3];bx[SI_X_EMISS]+=fx[4];bx[SI_X_GLW]+=fx[5];
 bx[SI_X_TKMS]+=fx[6];bx[SI_X_CP]+=fx[7];bx[SI_X_RHO]+=fx[8];bx[SI_X_QKMS]+=fx[9];
 bx[SI_X_QVATM]+=fx[10];bx[SI_X_RNET]+=fx[13];bdiff[0]+=fx[14];bcap[0]+=fx[15];
 double bdzstop=fx[16],br211=fx[19];bx[SI_X_DELT]+=fx[20];bx[SI_X_ZMAIN1]+=fx[21];
 bx[SI_X_RAINF]+=fx[22];bx[SI_X_CW]+=fx[23];bx[SI_X_PRCPMS]+=fx[24];bx[SI_X_QSG]+=fx[25];
 bx[SI_X_ROVCP]+=fx[26]*double(in.pow_value)*log(double(t.pow_base));
 bx[SI_X_PATM]-=fx[26]*double(in.pow_value)*double(x[SI_X_ROVCP])/double(x[SI_X_PATM]);
 double bqs=by[SI_Y_QVG]+by[SI_Y_QSG]+fx[11]+fx[12],bts=0.;
 if(!si_sice_backsub_ad(n,t.c,t.new_tso,t.pass,bnew,bc,br,bts))return false;
 const SiVilkaPoint &q=t.vilka;double bnum=bqs/double(q.pp);
 double bpp=-bnum*double(t.qs);
 double brn=-double(.05f)*bts+bnum*double(__fsub_rn(q.a,q.b));
 double ba=bnum+bnum*double(q.rn),bb=-bnum*double(q.rn);
 double bf=brn/double(q.den),bden=-bf*double(q.rn);
 double aa=double(t.closure.v[t.closure.out[2]]);
 double baa=bf*double(q.a)+bden*double(__fsub_rn(q.b,q.a));
 double bbb=-bf;ba+=bf*aa-bden*aa;bb+=bden*aa;
 btable0+=ba;btable1+=bb;
 double cy[SI_SC_NOUTPUT]={0.,br211,baa,bbb,bpp},cx[SI_SC_NINPUT]={};
 if(!si_sice_closure_ad(t.closure,cy,cx))return false;
 bdiff[0]+=cx[0];bcap[0]+=cx[1];bdzstop+=cx[2];bx[SI_X_TKMS]+=cx[3];
 bx[SI_X_CP]+=cx[4];bx[SI_X_RHO]+=cx[5];bx[SI_X_CONFLX]+=cx[6];bx[SI_X_DELT]+=cx[7];
 bx[SI_X_EMISS]+=cx[8];bx[SI_X_STBOLT]+=cx[9];bx[SI_X_SOILT]+=cx[10];
 bc[n-2]+=cx[11];br[n-2]+=cx[12];bx[SI_X_RNET]+=cx[13];bx[SI_X_RAINF]+=cx[14];
 bx[SI_X_CW]+=cx[15];bx[SI_X_PRCPMS]+=cx[16];bx[SI_X_QKMS]+=cx[17];
 bx[SI_X_TABS]+=cx[18];bx[SI_X_QVATM]+=cx[19];bx[SI_X_QVG]+=cx[20];bx[SI_X_PATM]+=cx[21];
 if(!si_sice_eliminate_ad(n,in.old_tso,in.diff,in.dz,t.c,t.r,bold,bdiff,bdz,bc,br))return false;
 double bz=-bdzstop*double(t.dzstop)/double(t.denz);
 bx[SI_X_ZMAIN0]-=bz;bx[SI_X_ZMAIN1]+=bz;
 return true;
}

// A column-local composition tape. This is an internal hand-rule workspace,
// not a FRAMEWORK launch entry. The owner supplies disjoint node/leaf storage,
// one double work array of at least node_capacity elements, and deterministic
// per-column table adjoints. No cross-column atomic accumulation occurs here.
struct SiColumnNode { float value; double da,db; int a,b; };
struct SiColumnLeaf { int node,field,element; };
struct SiColumnGraph {
    SiColumnNode *node; SiColumnLeaf *leaf;
    int count,capacity,nleaf,leaf_capacity;
    bool valid;
};
__device__ __forceinline__ int si_col_node(SiColumnGraph &g,float v,
        int a,int b,double da,double db) {
    if(!g.valid||g.count>=g.capacity||!si_finite(v)) {g.valid=false;return 0;}
    int i=g.count++;g.node[i]={v,da,db,a,b};return i;
}
__device__ __forceinline__ int si_col_const(SiColumnGraph &g,float v) {
    return si_col_node(g,v,-1,-1,0.,0.);
}
__device__ __forceinline__ int si_col_var(SiColumnGraph &g,float v,int f,int e) {
    if(g.nleaf>=g.leaf_capacity){g.valid=false;return 0;}
    int i=si_col_const(g,v);if(g.valid)g.leaf[g.nleaf++]={i,f,e};return i;
}
__device__ __forceinline__ int si_col_op(SiColumnGraph &g,int a,int b,int op) {
    if(!g.valid||a<0||b<0||a>=g.count||b>=g.count){g.valid=false;return 0;}
    float x=g.node[a].value,y=g.node[b].value,v;double da,db;
    if(op==0){v=__fadd_rn(x,y);da=1.;db=1.;}
    else if(op==1){v=__fsub_rn(x,y);da=1.;db=-1.;}
    else if(op==2){v=__fmul_rn(x,y);da=double(y);db=double(x);}
    else {if(y==0.f){g.valid=false;return 0;}v=__fdiv_rn(x,y);
          da=1./double(y);db=-(double(x)/double(y))/double(y);}
    return si_col_node(g,v,a,b,da,db);
}
__device__ __forceinline__ int si_col_min(SiColumnGraph &g,int a,int b) {
    return g.node[a].value<g.node[b].value?a:b;
}
__device__ __forceinline__ int si_col_min_record(SiColumnGraph &g,int a,int b,
                                                unsigned int *record,int slot) {
    const bool first=g.node[a].value<g.node[b].value;
    if(record&&g.valid)si_branch(record,slot,first);
    return first?a:b;
}
// Import a manually specified fragment while preserving its executed float
// values and local slopes. Each fragment input can alias any previous node.
__device__ __forceinline__ bool si_col_import(SiColumnGraph &g,
        const SiSiceClosureTape &t,int ni,const int *input,int no,
        const int *fragment_output,int *output,int mask=-1) {
    if(!g.valid||!t.valid||ni<0||ni>t.n||t.n>SI_SC_NNODE)return false;
    int map[SI_SC_NNODE];
    for(int i=0;i<ni;++i) {
        int k=input[i];if(k<0||k>=g.count||
           __float_as_uint(g.node[k].value)!=__float_as_uint(t.v[i]))return false;
        map[i]=k;
    }
    for(int i=ni;i<t.n;++i) {
        if(t.a[i]<0)map[i]=si_col_const(g,t.v[i]);
        else {
            if(t.a[i]>=i||t.b[i]<0||t.b[i]>=i)return false;
            map[i]=si_col_node(g,t.v[i],map[t.a[i]],map[t.b[i]],t.da[i],t.db[i]);
        }
    }
    for(int i=0;i<no;++i) {
        if(!(mask&(1<<i))){output[i]=-1;continue;}
        int k=fragment_output[i];if(k<0||k>=t.n)return false;output[i]=map[k];
    }
    return g.valid;
}
__device__ __forceinline__ bool si_col_tl(const SiColumnGraph &g,
        const double *const *direction,double *d) {
    if(!g.valid)return false;
    for(int i=0;i<g.count;++i)d[i]=0.;
    for(int i=0;i<g.nleaf;++i) {
        SiColumnLeaf x=g.leaf[i];d[x.node]=direction[x.field][x.element];
    }
    for(int i=0;i<g.count;++i)if(g.node[i].a>=0) {
        SiColumnNode x=g.node[i];d[i]=x.da*d[x.a]+x.db*d[x.b];
    }
    return true;
}
__device__ __forceinline__ bool si_col_ad(const SiColumnGraph &g,
        const int *out,const double *seed,int nout,double *b,double *const *result) {
    if(!g.valid)return false;
    for(int i=0;i<g.count;++i)b[i]=0.;
    for(int i=0;i<nout;++i) {if(out[i]<0||out[i]>=g.count)return false;b[out[i]]+=seed[i];}
    for(int i=g.count-1;i>=0;--i)if(g.node[i].a>=0) {
        SiColumnNode x=g.node[i];b[x.a]+=x.da*b[i];b[x.b]+=x.db*b[i];
    }
    for(int i=0;i<g.nleaf;++i) {
        SiColumnLeaf x=g.leaf[i];result[x.field][x.element]+=b[x.node];
    }
    return true;
}

enum SiSnowColumnField {
    SI_SNOW_SCALAR,SI_SNOW_TSO,SI_SNOW_DIFF,SI_SNOW_CAP,
    SI_SNOW_ZMAIN,SI_SNOW_ZHALF,SI_SNOW_DZ,SI_SNOW_TABLE,SI_SNOW_NFIELD
};

// Import one executed child column into its parent without detaching the
// material properties or other inputs computed by the parent. `input[f][e]`
// is the parent node for child field f, element e. Table reads instead become
// individual parent leaves so repeated selected entries accumulate during the
// parent reverse sweep. The caller owns `map`, sized to child.count.
__device__ __forceinline__ bool si_col_import_column(
        SiColumnGraph &parent,const SiColumnGraph &child,
        const int *const *input,const int *length,int *map,
        const int *child_out,int nout,int *parent_out) {
    if(!parent.valid||!child.valid||!input||!length||!map||
       nout<0||(nout&&(!child_out||!parent_out)))return false;
    for(int i=0;i<child.count;++i)map[i]=-1;
    for(int i=0;i<child.nleaf;++i) {
        SiColumnLeaf leaf=child.leaf[i];
        if(leaf.node<0||leaf.node>=child.count||map[leaf.node]>=0||
           leaf.field<0||leaf.field>=SI_SNOW_NFIELD||leaf.element<0||
           leaf.element>=length[leaf.field])return false;
        float value=child.node[leaf.node].value;int p;
        if(leaf.field==SI_SNOW_TABLE) {
            p=si_col_var(parent,value,SI_SNOW_TABLE,leaf.element);
        } else {
            if(!input[leaf.field])return false;
            p=input[leaf.field][leaf.element];
            if(p<0||p>=parent.count||
               __float_as_uint(parent.node[p].value)!=__float_as_uint(value))return false;
        }
        if(!parent.valid)return false;
        map[leaf.node]=p;
    }
    for(int i=0;i<child.count;++i) {
        if(map[i]>=0)continue;
        SiColumnNode node=child.node[i];
        if(node.a<0)map[i]=si_col_const(parent,node.value);
        else {
            if(node.a>=i||node.b<0||node.b>=i||map[node.a]<0||map[node.b]<0)
                return false;
            map[i]=si_col_node(parent,node.value,map[node.a],map[node.b],node.da,node.db);
        }
        if(!parent.valid)return false;
    }
    for(int i=0;i<nout;++i) {
        if(child_out[i]<0||child_out[i]>=child.count)return false;
        parent_out[i]=map[child_out[i]];
    }
    return parent.valid;
}
enum SiSnowColumnScalar {
    SN_DELT,SN_CONFLX,SN_RHONEWSN,SN_SNHEI_CRIT,SN_PRCPMS,SN_RAINF,
    SN_NEWSNOW,SN_SNHEI,SN_SNWE,SN_SNOWFRAC,SN_RHOSN,SN_PATM,SN_QVATM,
    SN_QCATM,SN_GLW,SN_GSW,SN_EMISS,SN_RNET,SN_QKMS,SN_TKMS,SN_RHO,
    SN_XLV,SN_CP,SN_ROVCP,SN_CW,SN_STBOLT,SN_TABS,SN_SOILT,SN_SOILT1,
    SN_TSNAV,SN_QVG,SN_QSG,SN_QCG,SN_SNOM,SN_ALB,SN_ZNT,SN_DEW,SN_S,
    SI_SNOW_NSCALAR
};
enum SiSnowColumnOutput {
    SN_OUT_SNHEI,SN_OUT_SNWE,SN_OUT_SNOWFRAC,SN_OUT_RHOSN,SN_OUT_EMISS,
    SN_OUT_ALB,SN_OUT_ZNT,SN_OUT_SNWEPRINT,SN_OUT_SNHEIPRINT,SN_OUT_RSM,
    SN_OUT_DEW,SN_OUT_SOILT,SN_OUT_SOILT1,SN_OUT_TSNAV,SN_OUT_QVG,
    SN_OUT_QSG,SN_OUT_QCG,SN_OUT_SMELT,SN_OUT_SNOH,SN_OUT_SNFLX,SN_OUT_SNOM,
    SN_OUT_EETA,SN_OUT_QFX,SN_OUT_HFX,SN_OUT_S,SN_OUT_SUBLIM,SN_OUT_PRCPL,
    SN_OUT_FLTOT,SI_SNOW_NOUTPUT
};
struct SiSnowColumnInput {
    const float *scalar,*tso,*diff,*cap,*zmain,*zhalf,*dz,*table;
    const int *vilka_visits;
    int nzs,nddzs,vilka_nvisits,ilnb;
    bool myj;
    float pow_value; // saved result of the actual forward gfk_pow call
    int *vilka_record,*vilka_record_count,vilka_record_capacity;
};
struct SiSnowColumnTape {
    SiColumnGraph graph;
    int *c,*r,*tso; // caller-allocated int arrays, each of length nzs
    int out[SI_SNOW_NOUTPUT],ilnb;
    unsigned char written[SI_SNOW_NOUTPUT],tso_written;
    bool melt;
};
__device__ __forceinline__ bool si_col_values(const SiColumnGraph &g,
        const int *index,int n,float *value) {
    for(int i=0;i<n;++i){if(index[i]<0||index[i]>=g.count)return false;value[i]=g.node[index[i]].value;}
    return true;
}
// Index decisions are discrete; selected table values and interpolation weight
// stay active. Independent leaves for repeated table indices are reduced by
// si_col_ad in recorded order into the same per-column table workspace.
__device__ __forceinline__ bool si_col_qsn(SiColumnGraph &g,int tn,
        const float *table,int &out) {
    SiQsnPoint q;if(!si_qsn_point(g.node[tn].value,q))return false;
    int a=si_col_var(g,table[q.slot],SI_SNOW_TABLE,q.slot);
    int b=si_col_var(g,table[q.slot+1],SI_SNOW_TABLE,q.slot+1);
    int weight;
    if(q.dr_dtn==0.f)weight=si_col_const(g,q.weight);
    else {
        int r=si_col_op(g,si_col_op(g,tn,si_col_const(g,173.15f),1),
                            si_col_const(g,.05f),3);
        r=si_col_op(g,r,si_col_const(g,1.f),0);
        weight=si_col_op(g,r,si_col_const(g,float(q.slot+1)),1);
    }
    out=si_col_op(g,si_col_op(g,si_col_op(g,b,a,1),weight,2),a,0);
    return g.valid;
}
__device__ __forceinline__ bool si_vilka_record(float tn,float d1,float d2,
        float pp,const float *table,int *visits,int capacity,int &count,
        float &qs,float &ts) {
    count=0;
    if(!table||!visits||capacity<2||!si_finite(tn)||!si_finite(d1)||
       !si_finite(d2)||!si_finite(pp)||pp==0.f)return false;
    float r=__fadd_rn(__fdiv_rn(__fsub_rn(tn,1.7315e2f),.05f),1.f);
    if(!si_finite(r)||r<=-2147483648.f||r>=2147483648.f)return false;
    int i=int(r);
    // The original indexes the table here before its first range check. An
    // out-of-range access is undefined, so AD refuses before reading it.
    if(i<1||i>5000)return false;
    visits[count++]=i;
    float a=table[i-1],b=table[i];
    if(!si_finite(a)||!si_finite(b))return false;
    float t1=__fadd_rn(173.1f,__fmul_rn(float(i),.05f));
    float f=__fsub_rn(__fadd_rn(t1,__fmul_rn(d1,a)),d2);
    float den=__fadd_rn(.05f,__fmul_rn(d1,__fsub_rn(b,a)));
    if(!si_finite(f)||!si_finite(den)||den==0.f)return false;
    float proposal=__fsub_rn(float(i),__fdiv_rn(f,den));
    if(!si_finite(proposal)||proposal<=-2147483648.f||proposal>=2147483648.f)return false;
    i=int(proposal);
    if(i<1||i>5000)return false;
    for(;;) {
        if(count>=capacity)return false;
        visits[count++]=i;
        a=table[i-1];b=table[i];if(!si_finite(a)||!si_finite(b))return false;
        t1=__fadd_rn(173.1f,__fmul_rn(float(i),.05f));
        f=__fsub_rn(__fadd_rn(t1,__fmul_rn(d1,a)),d2);
        den=__fadd_rn(.05f,__fmul_rn(d1,__fsub_rn(b,a)));
        if(!si_finite(f)||!si_finite(den)||den==0.f)return false;
        float rn=__fdiv_rn(f,den);
        if(!si_finite(rn)||rn<=-2147483648.f||rn>=2147483648.f)return false;
        long long next=(long long)i-(long long)int(rn);
        if(next<1||next>5000)return false;
        if(next==i) {
            ts=__fsub_rn(t1,__fmul_rn(.05f,rn));
            qs=__fdiv_rn(__fadd_rn(a,__fmul_rn(__fsub_rn(a,b),rn)),pp);
            return si_finite(qs)&&si_finite(ts);
        }
        i=int(next);
    }
}


__device__ __forceinline__ bool si_col_vilka(SiColumnGraph &g,int tn,int d1,
        int d2,int pp,const float *table,const int *visits,int nvisits,
        int *capture,int *capture_count,int capture_capacity,int &qs,int &ts) {
    if(!visits) {
        if(!capture||!capture_count)return false;
        float saved_q,saved_t;int count=0;
        if(!si_vilka_record(g.node[tn].value,g.node[d1].value,g.node[d2].value,
                            g.node[pp].value,table,capture,capture_capacity,
                            count,saved_q,saved_t))return false;
        *capture_count=count;visits=capture;nvisits=count;
    }
    int i;if(!si_vilka_replay(g.node[tn].value,g.node[d1].value,g.node[d2].value,
                               table,visits,nvisits,i))return false;
    int a=si_col_var(g,table[i-1],SI_SNOW_TABLE,i-1);
    int b=si_col_var(g,table[i],SI_SNOW_TABLE,i);
    int half=si_col_const(g,.05f);
    int t1=si_col_const(g,__fadd_rn(173.1f,__fmul_rn(float(i),.05f)));
    int f=si_col_op(g,si_col_op(g,t1,si_col_op(g,d1,a,2),0),d2,1);
    int den=si_col_op(g,half,si_col_op(g,d1,si_col_op(g,b,a,1),2),0);
    int rn=si_col_op(g,f,den,3);
    ts=si_col_op(g,t1,si_col_op(g,half,rn,2),1);
    qs=si_col_op(g,si_col_op(g,a,si_col_op(g,si_col_op(g,a,b,1),rn,2),0),pp,3);
    return g.valid;
}

// Hand composition of ice_snowseaice. Arrays and the two selected table
// entries remain active; the caller supplies the saved forward math result and
// Vilka's complete visit sequence. A false result is a refusal, not an
// alternate physical path. This internal interface still needs its ABI binder
// and captured-column component proof before registration as complete.
#ifndef SI_FRAME_DEFINED
#define SI_FRAME_DEFINED
// G4 (2026-11-06): the augmented primal's fragment tapes live in a per-column frame of the caller's global scratch
// arena, not on the thread stack. On the stack they made a 49,616-byte frame that the driver reserves for every
// resident thread of the card (13.1 GiB on an RTX PRO 6000). The host sets si_frame_base (column c's frame is
// si_frame_base + c * si_frame_stride, c the launch's global thread index, which every __aug kernel uses as its column)
// and sizes the arena from si_frame_bytes. Storage only: every value and slope is computed exactly as before.
enum { SI_FRAME_SLOT=3712, SI_FRAME_NSLOT=15 };
__device__ unsigned char *si_frame_base;
__device__ unsigned long long si_frame_stride;
__device__ const unsigned long long si_frame_bytes=(unsigned long long)SI_FRAME_SLOT*SI_FRAME_NSLOT;
template<class T> __device__ __forceinline__ T &si_frame_slot(int slot) {
    static_assert(sizeof(T)<=SI_FRAME_SLOT&&alignof(T)<=16,"G4 frame slot too small");
    unsigned long long c=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    return *reinterpret_cast<T*>(si_frame_base+c*si_frame_stride+(unsigned long long)slot*SI_FRAME_SLOT);
}
#endif
__device__ __forceinline__ bool si_snow_column_record(
        const SiSnowColumnInput &in,SiSnowColumnTape &t, unsigned int *branches=nullptr) {
    if(in.nzs<3||in.nddzs<2*(in.nzs-2)||!in.scalar||!in.tso||!in.diff||
       !in.cap||!in.zmain||!in.zhalf||!in.dz||!in.table||
       (!in.vilka_visits&&(!in.vilka_record||!in.vilka_record_count))||
       !t.c||!t.r||!t.tso||!t.graph.node||!t.graph.leaf)return false;
    SiColumnGraph &g=t.graph;g.count=0;g.nleaf=0;g.valid=true;
    int x[SI_SNOW_NSCALAR];
    for(int i=0;i<SI_SNOW_NSCALAR;++i)x[i]=si_col_var(g,in.scalar[i],SI_SNOW_SCALAR,i);
    for(int i=0;i<in.nzs;++i)t.tso[i]=si_col_var(g,in.tso[i],SI_SNOW_TSO,i);
    int zero=si_col_const(g,0.f),one=si_col_const(g,1.f);
    int z0=si_col_var(g,in.zmain[0],SI_SNOW_ZMAIN,0);
    int z1=si_col_var(g,in.zmain[1],SI_SNOW_ZMAIN,1);
    int zh1=si_col_var(g,in.zhalf[1],SI_SNOW_ZHALF,1);
    int zh2=si_col_var(g,in.zhalf[2],SI_SNOW_ZHALF,2);
    int diff0=si_col_var(g,in.diff[0],SI_SNOW_DIFF,0);
    int diff1=si_col_var(g,in.diff[1],SI_SNOW_DIFF,1);
    int cap0=si_col_var(g,in.cap[0],SI_SNOW_CAP,0);
    int dz0=si_col_var(g,in.dz[0],SI_SNOW_DZ,0);
    int dz1=si_col_var(g,in.dz[1],SI_SNOW_DZ,1);
    float v[SI_SM_ENERGY_NINPUT+8];
    int p0[]={x[SN_SNHEI],x[SN_RHOSN],x[SN_SNWE],x[SN_RHO],x[SN_QKMS],
              x[SN_QVATM],x[SN_QSG],x[SN_DELT]};
    SiSnowPrepTape &prep=si_frame_slot<SiSnowPrepTape>(0);int p[SI_SN_PRE_NOUTPUT];
    if(!si_col_values(g,p0,SI_SN_PRE_NINPUT,v)||!si_snow_prep_record(v,prep,branches)||
       !si_col_import(g,prep.g,SI_SN_PRE_NINPUT,p0,SI_SN_PRE_NOUTPUT,prep.out,p))return false;
    int deltsn=p[0],snth=p[1],rhocsn=p[2],thdifsn=p[3],ras=p[4];
    int snhei=p[5],beta=p[6],snwe=p[7],epot=p[8],snwepr=p[10];
    const int n=in.nzs;t.c[0]=zero;t.r[0]=t.tso[n-1];
    for(int k=1;k<=n-2;++k) {
        int kn=n-k,j=kn-1,a=2*kn-4,b=a+1;
        int da=si_col_var(g,in.dz[a],SI_SNOW_DZ,a);
        int db=si_col_var(g,in.dz[b],SI_SNOW_DZ,b);
        int dl=si_col_var(g,in.diff[j-1],SI_SNOW_DIFF,j-1);
        int dr=si_col_var(g,in.diff[j],SI_SNOW_DIFF,j);
        int x1=si_col_op(g,da,dl,2),x2=si_col_op(g,db,dr,2);
        int f=si_col_op(g,si_col_op(g,t.tso[j],si_col_op(g,x1,
                    si_col_op(g,t.tso[j-1],t.tso[j],1),2),0),
                    si_col_op(g,x2,si_col_op(g,t.tso[j],t.tso[j+1],1),2),1);
        int den=si_col_op(g,si_col_op(g,si_col_op(g,one,x1,0),x2,0),
                             si_col_op(g,x2,t.c[k-1],2),1);
        t.c[k]=si_col_op(g,x1,den,3);
        t.r[k]=si_col_op(g,si_col_op(g,f,si_col_op(g,x2,t.r[k-1],2),0),den,3);
    }
    int u0[]={snhei,snth,deltsn,x[SN_DELT],zh1,zh2,z1,thdifsn,dz0,dz1,
              diff0,diff1,t.tso[0],t.tso[1],t.tso[2],x[SN_SOILT],x[SN_SOILT1],
              t.c[n-2],t.r[n-2],t.c[n-3],t.r[n-3],x[SN_TSNAV]};
    SiSnowCoupleTape &couple=si_frame_slot<SiSnowCoupleTape>(1);int u[SI_SN_COUPLE_NOUTPUT];
    if(!si_col_values(g,u0,SI_SN_COUPLE_NINPUT,v)||!si_snow_couple_record(v,couple,branches)||
       !si_col_import(g,couple.g,SI_SN_COUPLE_NINPUT,u0,SI_SN_COUPLE_NOUTPUT,
                      couple.out,u,couple.mask))return false;
    // Only the one- and two-layer arms assign ilnb in the forward.  The
    // shallow-snow and no-snow arms retain its incoming value.
    t.ilnb=couple.arm>=2?couple.ilnb:in.ilnb;
    t.c[n-2]=u[2];t.r[n-2]=u[3];
    t.c[n-1]=u[0]<0?zero:u[0];t.r[n-1]=u[1]<0?zero:u[1];
    int cotsn=u[4]<0?zero:u[4],rhtsn=u[5]<0?zero:u[5];
    int tsnav=u[6],soilt1=u[7],tsob=u[8]<0?zero:u[8];
    int snprim=u[9]<0?zero:u[9],fsn=u[10],fso=u[11];
    int c0[]={snhei,snth,deltsn,x[SN_QKMS],x[SN_QVATM],x[SN_QSG],cap0,
        x[SN_TKMS],t.c[n-2],t.r[n-2],x[SN_SOILT],diff0,z0,z1,x[SN_CP],
        x[SN_RHO],x[SN_CONFLX],x[SN_DELT],x[SN_EMISS],x[SN_STBOLT],x[SN_RNET],
        cotsn,rhtsn,snprim,fsn,fso,thdifsn,rhocsn,x[SN_RAINF],x[SN_CW],
        x[SN_PRCPMS],x[SN_RHONEWSN],x[SN_NEWSNOW],x[SN_XLV],x[SN_TABS],
        x[SN_QVG],x[SN_PATM],t.c[n-1],t.r[n-1],beta,zero};
    SiSnowClosureTape &closure=si_frame_slot<SiSnowClosureTape>(2);int c[SI_SN_CLOSURE_NOUTPUT];
    if(!si_col_values(g,c0,SI_SN_CLOSURE_NINPUT,v)||!si_snow_closure_record(v,closure,branches)||
       !si_col_import(g,closure.g,SI_SN_CLOSURE_NINPUT,c0,SI_SN_CLOSURE_NOUTPUT,
                      closure.out,c))return false;
    int qs,soilt;
    if(!si_col_vilka(g,x[SN_SOILT],c[0],c[1],c[2],in.table,
                     in.vilka_visits,in.vilka_nvisits,in.vilka_record,
                     in.vilka_record_count,in.vilka_record_capacity,qs,soilt))return false;
    int qvg=qs,qsg=qs;
    int icecap=si_col_const(g,271.4f),freeze=si_col_const(g,273.15f);
    float h=g.node[snhei].value,th=g.node[snth].value;
    const bool thick=h>=th;si_branch_pair(branches,33,thick);
    if(!thick&&branches) {
        const bool positive=h>0.f;
        si_branch(branches,37,positive&&h<th);si_branch(branches,38,positive);
        if(positive)si_branch(branches,39,h<th);
    }
    if(thick) {
        const bool two_layers=h>__fadd_rn(g.node[deltsn].value,th);
        si_branch_pair(branches,35,two_layers);
        if(two_layers) {
            soilt1=si_col_min_record(g,freeze,si_col_op(g,rhtsn,si_col_op(g,cotsn,soilt,2),0),branches,108);
            t.tso[0]=si_col_min_record(g,icecap,si_col_op(g,t.r[n-1],si_col_op(g,t.c[n-1],soilt1,2),0),branches,109);
            tsob=soilt1;
        } else {
            t.tso[0]=si_col_min_record(g,icecap,si_col_op(g,t.r[n-1],si_col_op(g,t.c[n-1],soilt,2),0),branches,110);
            soilt1=t.tso[0];tsob=t.tso[0];
        }
    } else if(h>0.f&&h<th) {
        t.tso[1]=si_col_min_record(g,icecap,si_col_op(g,t.r[n-2],si_col_op(g,t.c[n-2],soilt,2),0),branches,111);
        t.tso[0]=si_col_min_record(g,icecap,si_col_op(g,t.tso[1],si_col_op(g,
                              si_col_op(g,soilt,t.tso[1],1),fso,2),0),branches,112);
        soilt1=t.tso[0];tsob=t.tso[1];
    } else {
        t.tso[0]=si_col_min_record(g,icecap,soilt,branches,113);
        soilt1=si_col_min_record(g,icecap,soilt,branches,114);tsob=t.tso[0];
    }
    const bool loop_positive=h>0.f,loop_thin=loop_positive&&h<th;
    if(branches) {
        si_branch(branches,40,loop_thin);si_branch(branches,41,loop_positive);
        if(loop_positive)si_branch(branches,42,h<th);
    }
    int first=loop_thin?2:1;
    for(int k=first;k<n;++k) {
        int j=n-k-1;
        t.tso[k]=si_col_min_record(g,icecap,si_col_op(g,t.r[j],si_col_op(g,t.c[j],t.tso[k-1],2),0),branches,loop_thin?115:116);
    }
    int m0[]={soilt,snwepr,beta,c[11],ras,x[SN_DELT],snhei,x[SN_SNOWFRAC],
              snwe,x[SN_QVATM],qsg,x[SN_QKMS]};
    SiSnowMeltSelectTape &select=si_frame_slot<SiSnowMeltSelectTape>(3);int m[SI_SM_SELECT_NOUTPUT];
    if(!si_col_values(g,m0,SI_SM_SELECT_NINPUT,v)||!si_snow_melt_select_record(v,select,branches)||
       !si_col_import(g,select.g,SI_SM_SELECT_NINPUT,m0,SI_SM_SELECT_NOUTPUT,
                      select.out,m))return false;
    t.melt=select.melt;int smelt=zero,snoh=zero,rsm=zero;
    if(select.melt) {
        int qfrac;if(!si_col_qsn(g,m[0],in.table,qfrac))return false;
        int xlvm=si_col_op(g,x[SN_XLV],si_col_const(g,3.35e5f),0);
        int rhonewcsn=si_col_op(g,si_col_const(g,2090.f),x[SN_RHONEWSN],2);
        int e0[]={m[0],qfrac,c[2],x[SN_STBOLT],x[SN_SOILT],x[SN_EMISS],x[SN_GLW],
            x[SN_QKMS],x[SN_QVATM],ras,xlvm,x[SN_RHO],beta,c[10],x[SN_TABS],
            snhei,snth,thdifsn,rhocsn,tsob,snprim,fsn,fso,diff0,cap0,c[8],
            c[6],c[7],c[9],x[SN_QSG],x[SN_RNET],rhonewcsn,x[SN_NEWSNOW],
            x[SN_DELT],x[SN_RAINF],x[SN_CW],x[SN_PRCPMS],x[SN_DEW]};
        SiSnowMeltEnergyTape &energy=si_frame_slot<SiSnowMeltEnergyTape>(4);int e[SI_SM_ENERGY_NOUTPUT];
        if(!si_col_values(g,e0,SI_SM_ENERGY_NINPUT,v)||!si_snow_melt_energy_record(v,energy,branches)||
           !si_col_import(g,energy.g,SI_SM_ENERGY_NINPUT,e0,SI_SM_ENERGY_NOUTPUT,
                          energy.out,e))return false;
        int w0[]={e[8],snwepr,x[SN_DELT],beta,e[1],ras,soilt,snhei};
        SiSnowMeltWaterTape &water=si_frame_slot<SiSnowMeltWaterTape>(5);int w[SI_SM_WATER_NOUTPUT];
        if(!si_col_values(g,w0,SI_SM_WATER_NINPUT,v)||!si_snow_melt_water_record(v,water,branches)||
           !si_col_import(g,water.g,SI_SM_WATER_NINPUT,w0,SI_SM_WATER_NOUTPUT,
                          water.out,w))return false;
        smelt=w[0];snoh=w[1];rsm=w[3];snwe=w[4];soilt=m[0];epot=e[1];
    } else {epot=m[1];snwe=m[2];}
    int qpost;if(!si_col_qsn(g,soilt,in.table,qpost))return false;
    int z0s[]={smelt,rsm,snwe,x[SN_RHOSN],snhei,soilt,soilt1,t.tso[0],deltsn,
               tsnav,x[SN_PATM],qpost,x[SN_QKMS],x[SN_QVATM],x[SN_SNOM],
               x[SN_DELT],rhocsn,thdifsn};
    SiSnowPostTape &post=si_frame_slot<SiSnowPostTape>(6);int z[SI_SN_POST_NOUTPUT];
    if(!si_col_values(g,z0s,SI_SN_POST_NINPUT,v)||!si_snow_post_record(v,t.ilnb,post,branches)||
       !si_col_import(g,post.g,SI_SN_POST_NINPUT,z0s,SI_SN_POST_NOUTPUT,post.out,z))return false;
    int rhosn=z[0];rhocsn=z[1];thdifsn=z[2];tsnav=z[5];qsg=z[7];epot=z[8];
    float patm=g.node[x[SN_PATM]].value;
    float base=__fdiv_rn(__fmul_rn(100000.f,.00001f),patm);
    if(!(patm>0.f)||!(base>0.f)||!si_finite(in.pow_value))return false;
    int power=si_col_node(g,in.pow_value,x[SN_PATM],x[SN_ROVCP],
        -double(in.pow_value)*double(g.node[x[SN_ROVCP]].value)/double(patm),
         double(in.pow_value)*log(double(base)));
    int f0[]={x[SN_STBOLT],x[SN_SOILT],soilt,x[SN_EMISS],x[SN_GLW],x[SN_TKMS],
              x[SN_CP],x[SN_RHO],x[SN_TABS],power,x[SN_QKMS],ras,x[SN_QVATM],
              qsg,qvg,beta,x[SN_XLV],z[9]};
    SiSnowFluxTape &flux=si_frame_slot<SiSnowFluxTape>(7);int f[SI_SN_FLUX_NOUTPUT];
    if(!si_col_values(g,f0,SI_SN_FLUX_NINPUT,v)||!si_snow_flux_record(v,in.myj,flux,branches)||
       !si_col_import(g,flux.g,SI_SN_FLUX_NINPUT,f0,SI_SN_FLUX_NOUTPUT,flux.out,f))return false;
    int y0[]={snhei,snth,thdifsn,rhocsn,soilt,tsob,snprim,fsn,fso,diff0,cap0,
              c[6],x[SN_S],snwe,rhosn,c[8],c[7],x[SN_SOILT],x[SN_XLV],c[9],
              qsg,x[SN_QSG],x[SN_RHONEWSN],x[SN_NEWSNOW],x[SN_DELT],x[SN_TABS],
              x[SN_RAINF],x[SN_CW],x[SN_PRCPMS],x[SN_RNET],f[2],f[6],snoh,
              tsnav,x[SN_ALB],x[SN_EMISS],x[SN_ZNT]};
    SiSnowEnergyTape &final=si_frame_slot<SiSnowEnergyTape>(8);int y[SI_SN_ENERGY_NOUTPUT];
    if(!si_col_values(g,y0,SI_SN_ENERGY_NINPUT,v)||!si_snow_energy_record(v,final,branches)||
       !si_col_import(g,final.g,SI_SN_ENERGY_NINPUT,y0,SI_SN_ENERGY_NOUTPUT,
                      final.out,y))return false;
    int output[SI_SNOW_NOUTPUT]={y[2],snwe,x[SN_SNOWFRAC],rhosn,y[8],y[7],y[9],
        z[3],z[4],rsm,f[5],soilt,soilt1,y[6],qvg,qsg,zero,smelt,snoh,y[1],
        z[10],f[6],f[7],f[3],y[0],f[8],x[SN_PRCPMS],y[5]};
    for(int i=0;i<SI_SNOW_NOUTPUT;++i)t.out[i]=output[i];
    // Record assignments, including writes of unchanged values. Graph-node
    // equality alone cannot distinguish those writes from pass-throughs.
    for(int i=0;i<SI_SNOW_NOUTPUT;++i)t.written[i]=1;
    t.written[SN_OUT_SNOWFRAC]=0;
    t.written[SN_OUT_SNWE]=prep.dryout||select.melt||select.has_snow;
    t.written[SN_OUT_RHOSN]=post.refreeze;
    t.written[SN_OUT_EMISS]=t.written[SN_OUT_ALB]=t.written[SN_OUT_ZNT]=final.now_bare;
    t.written[SN_OUT_S]=final.arm!=0;
    t.written[SN_OUT_TSNAV]=(couple.arm!=0)||post.depth_positive||final.now_bare;
    t.tso_written=1;
    return g.valid;
}

// Column-local bare-ice composition on the same deterministic graph as the
// snow calculation. Field numbers use SiSnowColumnField; the zhalf field is
// unused by this call. Table leaves are reduced in recorded order by the
// caller, including repeated references to the same entry.
struct SiSiceColumnInput {
    const float *scalar,*tso,*diff,*cap,*dz,*table;
    const int *vilka_visits;
    int nzs,nddzs,vilka_nvisits;
    bool myj;
    float pow_value;
    int *vilka_record,*vilka_record_count,vilka_record_capacity;
};
struct SiSiceColumnTape {
    SiColumnGraph graph;
    int *c,*r,*tso;
    int out[SI_SICE_NY];
    unsigned char written[SI_SICE_NY],tso_written;
};
__device__ __forceinline__ bool si_sice_column_record(
        const SiSiceColumnInput &in,SiSiceColumnTape &t,
        unsigned int *branches=nullptr) {
    if(in.nzs<3||in.nddzs<2*(in.nzs-2)||!in.scalar||!in.tso||!in.diff||
       !in.cap||!in.dz||!in.table||
       (!in.vilka_visits&&(!in.vilka_record||!in.vilka_record_count))||!t.c||!t.r||!t.tso||
       !t.graph.node||!t.graph.leaf)return false;
    SiColumnGraph &g=t.graph;g.count=0;g.nleaf=0;g.valid=true;
    for(int i=0;i<SI_SICE_NY;++i)t.written[i]=1;
    t.tso_written=1;
    int x[SI_SICE_NX];
    for(int i=0;i<SI_SICE_NX;++i)x[i]=si_col_var(g,in.scalar[i],SI_SNOW_SCALAR,i);
    const int n=in.nzs;
    for(int i=0;i<n;++i)t.tso[i]=si_col_var(g,in.tso[i],SI_SNOW_TSO,i);
    int zero=si_col_const(g,0.f),one=si_col_const(g,1.f);
    int diff0=si_col_var(g,in.diff[0],SI_SNOW_DIFF,0);
    int cap0=si_col_var(g,in.cap[0],SI_SNOW_CAP,0);
    int dzstop=si_col_op(g,one,si_col_op(g,x[SI_X_ZMAIN1],x[SI_X_ZMAIN0],1),3);
    t.c[0]=zero;t.r[0]=t.tso[n-1];
    for(int k=1;k<=n-2;++k) {
        int kn=n-k,j=kn-1,a=2*kn-4,b=a+1;
        int da=si_col_var(g,in.dz[a],SI_SNOW_DZ,a);
        int db=si_col_var(g,in.dz[b],SI_SNOW_DZ,b);
        int dl=si_col_var(g,in.diff[j-1],SI_SNOW_DIFF,j-1);
        int dr=si_col_var(g,in.diff[j],SI_SNOW_DIFF,j);
        int x1=si_col_op(g,da,dl,2),x2=si_col_op(g,db,dr,2);
        int f=si_col_op(g,si_col_op(g,t.tso[j],si_col_op(g,x1,
                    si_col_op(g,t.tso[j-1],t.tso[j],1),2),0),
                    si_col_op(g,x2,si_col_op(g,t.tso[j],t.tso[j+1],1),2),1);
        int den=si_col_op(g,si_col_op(g,si_col_op(g,one,x1,0),x2,0),
                             si_col_op(g,x2,t.c[k-1],2),1);
        t.c[k]=si_col_op(g,x1,den,3);
        t.r[k]=si_col_op(g,si_col_op(g,f,si_col_op(g,x2,t.r[k-1],2),0),den,3);
    }
    int q0[]={diff0,cap0,dzstop,x[SI_X_TKMS],x[SI_X_CP],x[SI_X_RHO],
        x[SI_X_CONFLX],x[SI_X_DELT],x[SI_X_EMISS],x[SI_X_STBOLT],x[SI_X_SOILT],
        t.c[n-2],t.r[n-2],x[SI_X_RNET],x[SI_X_RAINF],x[SI_X_CW],x[SI_X_PRCPMS],
        x[SI_X_QKMS],x[SI_X_TABS],x[SI_X_QVATM],x[SI_X_QVG],x[SI_X_PATM]};
    float v[SI_SF_NINPUT];SiSiceClosureTape &closure=si_frame_slot<SiSiceClosureTape>(9);int q[SI_SC_NOUTPUT];
    if(!si_col_values(g,q0,SI_SC_NINPUT,v)||!si_sice_closure_record(v,closure)||
       !si_col_import(g,closure,SI_SC_NINPUT,q0,SI_SC_NOUTPUT,closure.out,q))return false;
    int qs,ts;
    if(!si_col_vilka(g,x[SI_X_SOILT],q[2],q[3],q[4],in.table,
                     in.vilka_visits,in.vilka_nvisits,in.vilka_record,
                     in.vilka_record_count,in.vilka_record_capacity,qs,ts))return false;
    int icecap=si_col_const(g,271.4f);
    if(!g.valid)return false;
    if(branches)si_branch(branches,20,g.node[icecap].value<g.node[ts].value);
    t.tso[0]=si_col_min(g,icecap,ts);
    for(int k=1;k<n;++k) {
        int j=n-k-1;
        int uncapped=si_col_op(g,t.r[j],si_col_op(g,t.c[j],t.tso[k-1],2),0);
        if(!g.valid)return false;
        if(branches)si_branch(branches,21,g.node[icecap].value<g.node[uncapped].value);
        t.tso[k]=si_col_min(g,icecap,uncapped);
    }
    float patm=g.node[x[SI_X_PATM]].value;
    float base=__fdiv_rn(__fmul_rn(100000.f,.00001f),patm);
    if(!(patm>0.f)||!(base>0.f)||!si_finite(in.pow_value))return false;
    int power=si_col_node(g,in.pow_value,x[SI_X_PATM],x[SI_X_ROVCP],
        -double(in.pow_value)*double(g.node[x[SI_X_ROVCP]].value)/double(patm),
        double(in.pow_value)*log(double(base)));
    int f0[]={x[SI_X_SOILT],t.tso[0],x[SI_X_TABS],x[SI_X_STBOLT],
        x[SI_X_EMISS],x[SI_X_GLW],x[SI_X_TKMS],x[SI_X_CP],x[SI_X_RHO],
        x[SI_X_QKMS],x[SI_X_QVATM],qs,qs,x[SI_X_RNET],diff0,cap0,dzstop,
        t.tso[0],t.tso[1],q[1],x[SI_X_DELT],x[SI_X_ZMAIN1],x[SI_X_RAINF],
        x[SI_X_CW],x[SI_X_PRCPMS],x[SI_X_QSG],power};
    SiSiceFluxTape &flux=si_frame_slot<SiSiceFluxTape>(10);int f[SI_SF_NOUTPUT];
    if(!si_col_values(g,f0,SI_SF_NINPUT,v)||!si_sice_flux_record(v,in.myj,flux)||
       !si_col_import(g,flux.g,SI_SF_NINPUT,f0,SI_SF_NOUTPUT,flux.out,f))return false;
    if(branches) {
        // Both source-ledger rows 8 and 9 denote q1 <= 0. The max/min
        // comparisons count their first operand on true and the second on
        // false, exactly like ruc_max/ruc_min, including equal operands.
        si_branch(branches,8,flux.nonpositive);
        si_branch(branches,9,flux.nonpositive);
        si_branch(branches,19,!closure.tabs_arm);
        si_branch(branches,22,!flux.warm);
    }
    t.out[SI_Y_QVG]=qs;t.out[SI_Y_QSG]=qs;t.out[SI_Y_QCG]=zero;
    t.out[SI_Y_SOILT]=t.tso[0];
    for(int i=0;i<SI_SF_NOUTPUT;++i)t.out[SI_Y_DEW+i]=f[i];
    return g.valid;
}

#ifdef SI_SEAICE_ORIGINAL_MATH_AVAILABLE
// Standalone helper calls evaluate the same pinned power device function as
// the forward. The graph keeps patm and rovcp active; the value is recorded,
// not promoted to a constant input or replaced by a CUDA builtin.
__device__ __forceinline__ bool si_sice_column_record_aug(
        SiSiceColumnInput in,SiSiceColumnTape &t,unsigned int *branches=nullptr) {
    if(!in.scalar||!si_finite(in.scalar[SI_X_PATM])||in.scalar[SI_X_PATM]==0.f||
       !si_finite(in.scalar[SI_X_ROVCP]))return false;
    float base=__fdiv_rn(__fmul_rn(100000.f,.00001f),in.scalar[SI_X_PATM]);
    if(!si_finite(base)||base<=0.f)return false;
    in.pow_value=gfk_pow(base,in.scalar[SI_X_ROVCP]);
    return si_finite(in.pow_value)&&si_sice_column_record(in,t,branches);
}
__device__ __forceinline__ bool si_snow_column_record_aug(
        SiSnowColumnInput in,SiSnowColumnTape &t, unsigned int *branches=nullptr) {
    if(!in.scalar||!si_finite(in.scalar[SN_PATM])||in.scalar[SN_PATM]==0.f||
       !si_finite(in.scalar[SN_ROVCP]))return false;
    float base=__fdiv_rn(__fmul_rn(100000.f,.00001f),in.scalar[SN_PATM]);
    if(!si_finite(base)||base<=0.f)return false;
    in.pow_value=gfk_pow(base,in.scalar[SN_ROVCP]);
    return si_finite(in.pow_value)&&si_snow_column_record(in,t,branches);
}
#endif

// ice_icetmp composes the material properties, snow state, optical state and
// the actually executed bare/snow columns in one column-local tape. Scalar
// slots include every real argument read or returned by these expressions;
// integer flags, recorded math results and iteration visits are passive.
enum SiIceTmpScalar {
 IT_DELT,IT_CONFLX,IT_PRCPMS,IT_NEWSNMS,IT_SNOWRAT,IT_GRAUPRAT,
 IT_ICERAT,IT_CURAT,IT_PATM,IT_TABS,IT_QVATM,IT_QCATM,IT_RHO,IT_GLW,
 IT_GSW,IT_EMISS,IT_QKMS,IT_TKMS,IT_ALB,IT_ZNT,IT_ALB_SNOW,
 IT_ALB_SNOW_FREE,IT_CP,IT_ROVCP,IT_G0,IT_LV,IT_STBOLT,IT_CW,IT_C1SN,
 IT_C2SN,IT_SOILT,IT_SOILT1,IT_TSNAV,IT_RHOSN,IT_RHONEWSN,IT_RHOSNFALL,
 IT_SNWE,IT_SNHEI,IT_SNOWFRAC,IT_RSM,IT_SNWEPRINT,IT_SNHEIPRINT,IT_DEW,
 IT_QVG,IT_QSG,IT_QCG,IT_SMELT,IT_SNOH,IT_SNFLX,IT_SNOM,IT_SNOWFALLAC,
 IT_ACSNOW,IT_EDIR,IT_EC,IT_ETT,IT_EETA,IT_QFX,IT_HFX,IT_S,IT_SUBLIM,
 IT_EVAPL,IT_PRCPL,IT_FLTOT,IT_RUNOFF1,IT_RUNOFF2,IT_INFILTR,IT_SMF,
 SI_ICETMP_NSCALAR
};
// This recorder is compiled only in a translation unit that first assembles
// the pinned forward's math prelude byte-for-byte. The caller must define
// SI_SEAICE_ORIGINAL_MATH_AVAILABLE after that prelude. Never replace these
// calls with CUDA builtins: they do not have the forward's rounding behavior.
// The values are used in the rules below while their arguments remain active.
#ifdef SI_SEAICE_ORIGINAL_MATH_AVAILABLE
struct SiIceMathRecord {SiIceUnaryRecord unary[4];float power;unsigned mask;};
__device__ __forceinline__ bool si_icetmp_capture_math(
        const float *x,SiIceMathRecord &record) {
 if(!x)return false;
 for(int i=0;i<SI_ICETMP_NSCALAR;++i)if(!si_finite(x[i]))return false;
 for(int i=0;i<4;++i){record.unary[i].arg=0.f;record.unary[i].value=0.f;}
 record.mask=0u;record.power=0.f;
 float rho=x[IT_RHOSN];if(rho==0.f)return false;
 float threshold=__fdiv_rn(__fmul_rn(.0081f,1000.f),rho);
 if(!si_finite(threshold))return false;
 if(x[IT_SNHEI]>threshold) {
   float cold=0.f<x[IT_TSNAV]?0.f:x[IT_TSNAV];
   float a=__fsub_rn(__fmul_rn(.08f,cold),
                     __fmul_rn(__fmul_rn(x[IT_C2SN],rho),.001f));
   float y=gfk_exp_fma(a);
   if(!si_finite(a)||!si_finite(y))return false;
   record.unary[0]={a,y};record.mask|=1u;
   float bsn=__fmul_rn(__fmul_rn(__fdiv_rn(x[IT_DELT],3600.f),x[IT_C1SN]),y);
   float growth=__fmul_rn(__fmul_rn(bsn,x[IT_SNWE]),100.f);
   if(!si_finite(growth))return false;
   if(growth>1.e-4f) {
     y=gfk_exp_fma(growth);if(!si_finite(y))return false;
     record.unary[1]={growth,y};record.mask|=2u;
   }
 }
 float newsn=__fmul_rn(x[IT_NEWSNMS],x[IT_DELT]);
 if(!si_finite(newsn))return false;
 if(newsn>0.f) {
   float a=__fmul_rn(__fsub_rn(276.65f,x[IT_TABS]),.15f);
   float b=__fmul_rn(__fsub_rn(274.15f,x[IT_TABS]),.3333f);
   float y=sfire_tanhf(a),z=sfire_tanhf(b);
   if(!si_finite(a)||!si_finite(b)||!si_finite(y)||!si_finite(z))return false;
   record.unary[2]={a,y};record.unary[3]={b,z};record.mask|=12u;
 }
 if(x[IT_PATM]==0.f)return false;
 float base=__fdiv_rn(__fmul_rn(100000.f,.00001f),x[IT_PATM]);
 if(!si_finite(base)||base<=0.f)return false;
 record.power=gfk_pow(base,x[IT_ROVCP]);
 return si_finite(record.power);
}
#endif
struct SiIceTmpInput {
 const float *scalar,*tso,*zmain,*zhalf,*dz,*table;
 const SiIceUnaryRecord *unary;
 const int *bare_visits,*snow_visits;
 int nzs,nddzs,ntable,bare_nvisits,snow_nvisits,iland,isice;
 bool myj;
 float bare_pow,snow_pow;
 int *bare_record,*snow_record,*bare_record_count,*snow_record_count;
 int bare_record_capacity,snow_record_capacity;
};
// Scratch sizes: each layer array nzs, dz_nodes nddzs, child_map at least
// child.capacity. Both graphs and all scratch storage are caller-owned.
struct SiIceTmpTape {
 SiColumnGraph graph,child;
 int *tso,*tice,*cap,*diff,*zmain,*zhalf,*dz,*bare_tso;
 float *cap_value,*diff_value;
 int *child_c,*child_r,*child_tso,*child_map;
 int out[SI_ICETMP_NSCALAR],iland;
 unsigned char written[SI_ICETMP_NSCALAR],tso_written;
 bool has_snow,mosaic,keep_snow_albedo;
};
__device__ __forceinline__ int si_icetmp_blend(
        SiColumnGraph &g,int bare,int snowy,int frac,int one) {
 return si_col_op(g,si_col_op(g,bare,si_col_op(g,one,frac,1),2),
                     si_col_op(g,snowy,frac,2),0);
}
__device__ __forceinline__ bool si_icetmp_column_record(
        const SiIceTmpInput &in,SiIceTmpTape &t,unsigned int *branches=nullptr) {
 const int n=in.nzs,nd=in.nddzs;
 if(n<3||nd<2*(n-2)||in.ntable<5001||!in.scalar||!in.tso||!in.zmain||
    !in.zhalf||!in.dz||!in.table||!t.graph.node||!t.graph.leaf||
    !t.child.node||!t.child.leaf||!t.tso||!t.tice||!t.cap||!t.diff||
    !t.zmain||!t.zhalf||!t.dz||!t.bare_tso||!t.cap_value||!t.diff_value||
    !t.child_c||!t.child_r||!t.child_tso||!t.child_map)return false;
 SiColumnGraph &g=t.graph;g.count=0;g.nleaf=0;g.valid=true;
 int x[SI_ICETMP_NSCALAR];
 for(int k=0;k<SI_ICETMP_NSCALAR;++k) {
   x[k]=si_col_var(g,in.scalar[k],SI_SNOW_SCALAR,k);t.out[k]=x[k];t.written[k]=0;
 }
 t.tso_written=1;
 int zero=si_col_const(g,0.f),one=si_col_const(g,1.f);
 int k273=si_col_const(g,273.15f),k917=si_col_const(g,917.6f);
 int kexp=si_col_const(g,.000165f),kheat=si_col_const(g,2115.85f);
 int kslope=si_col_const(g,7.7948f),kcond=si_col_const(g,2.260872f);
 for(int k=0;k<n;++k) {
   t.tso[k]=si_col_var(g,in.tso[k],SI_SNOW_TSO,k);
   t.zmain[k]=si_col_var(g,in.zmain[k],SI_SNOW_ZMAIN,k);
   t.zhalf[k]=si_col_var(g,in.zhalf[k],SI_SNOW_ZHALF,k);
   t.tice[k]=si_col_op(g,t.tso[k],k273,1);
   int den=si_col_op(g,one,si_col_op(g,kexp,t.tice[k],2),1);
   int rho=si_col_op(g,k917,den,3);
   int heat=si_col_op(g,kheat,si_col_op(g,kslope,t.tice[k],2),0);
   t.cap[k]=si_col_op(g,heat,rho,2);t.diff[k]=si_col_op(g,kcond,t.cap[k],3);
   if(!g.valid)return false;
   t.cap_value[k]=g.node[t.cap[k]].value;t.diff_value[k]=g.node[t.diff[k]].value;
 }
 for(int k=0;k<nd;++k)t.dz[k]=si_col_var(g,in.dz[k],SI_SNOW_DZ,k);
 int s0[]={x[IT_DELT],x[IT_C1SN],x[IT_C2SN],x[IT_TSNAV],x[IT_RHOSN],
   x[IT_SNHEI],x[IT_SNWE],x[IT_SNOWFRAC],x[IT_NEWSNMS],x[IT_TABS],
   x[IT_SNOWRAT],x[IT_GRAUPRAT],x[IT_ICERAT],x[IT_CURAT],x[IT_PRCPMS],
   x[IT_SNOWFALLAC],x[IT_RHOSNFALL]};
 float values[SI_ICETMP_NSCALAR];SiIceSnowStateTape &state=si_frame_slot<SiIceSnowStateTape>(11);int st[SI_ICE_SNOWSTATE_NOUTPUT];
 if(!si_col_values(g,s0,SI_ICE_SNOWSTATE_NINPUT,values)||
    !si_icetmp_snowstate_record(values,in.unary,state,branches)||
    !si_col_import(g,state.g,SI_ICE_SNOWSTATE_NINPUT,s0,SI_ICE_SNOWSTATE_NOUTPUT,
                   state.out,st))return false;
 t.has_snow=state.has_snow;t.mosaic=state.has_snow&&state.mosaic;
 t.keep_snow_albedo=state.keep_snow_albedo;t.iland=state.has_snow?in.isice:in.iland;
 const int always[]={IT_RHONEWSN,IT_SNHEI,IT_SMELT,IT_RSM,IT_INFILTR,IT_SMF,IT_ALB};
 for(unsigned k=0;k<sizeof(always)/sizeof(always[0]);++k)t.written[always[k]]=1;
 t.written[IT_RHOSN]=state.compact_growth||state.new_snow;
 t.written[IT_RHOSNFALL]=state.new_snow;
 t.written[IT_SNWE]=state.new_snow;
 t.written[IT_SNOWFRAC]=(in.scalar[IT_SNHEI]==0.f)||state.has_snow;
 t.written[IT_EMISS]=state.has_snow;
 t.out[IT_RHOSN]=st[2];t.out[IT_SNWE]=st[3];t.out[IT_SNHEI]=st[4];
 t.out[IT_SNOWFRAC]=st[5];t.out[IT_RHONEWSN]=st[7];t.out[IT_RHOSNFALL]=st[11];
 t.out[IT_SMELT]=zero;t.out[IT_RSM]=zero;t.out[IT_INFILTR]=zero;t.out[IT_SMF]=zero;
 int o0[]={x[IT_GSW],x[IT_ALB],x[IT_ALB_SNOW_FREE],x[IT_ALB_SNOW],
   t.tice[0],x[IT_SOILT],x[IT_STBOLT],x[IT_GLW],x[IT_EMISS],st[5],st[4]};
 SiIceOpticsTape &optics=si_frame_slot<SiIceOpticsTape>(12);int opt[SI_ICE_OPTICS_NOUTPUT];
 if(!si_col_values(g,o0,SI_ICE_OPTICS_NINPUT,values)||
    !si_icetmp_optics_record(values,state.mosaic,state.keep_snow_albedo,optics,branches)||
    !si_col_import(g,optics.g,SI_ICE_OPTICS_NINPUT,o0,SI_ICE_OPTICS_NOUTPUT,
                   optics.out,opt))return false;
 t.out[IT_ALB]=opt[6];t.out[IT_EMISS]=opt[7];
 int scalar_ids[SI_SNOW_NSCALAR];float scalar_values[SI_SNOW_NSCALAR];
 const int *field_ids[SI_SNOW_NFIELD]={scalar_ids,t.tso,t.diff,t.cap,
                                       t.zmain,t.zhalf,t.dz,nullptr};
 int lengths[SI_SNOW_NFIELD]={SI_SNOW_NSCALAR,n,n,n,n,n,nd,in.ntable};
 int bare[SI_SICE_NY];
 if(!t.has_snow||t.mosaic) {
   if(!in.bare_visits&&(!in.bare_record||!in.bare_record_count))return false;
   int bare_emiss=t.mosaic?si_col_const(g,.98f):opt[7];
   int a[SI_SICE_NX]={x[IT_DELT],x[IT_CONFLX],x[IT_PRCPMS],st[10],x[IT_PATM],
     x[IT_QVATM],x[IT_GLW],bare_emiss,opt[3],x[IT_QKMS],x[IT_TKMS],x[IT_RHO],
     x[IT_CP],x[IT_ROVCP],x[IT_CW],x[IT_STBOLT],x[IT_TABS],x[IT_SOILT],
     x[IT_QVG],x[IT_QSG],t.zmain[0],t.zmain[1]};
   for(int k=0;k<SI_SICE_NX;++k)scalar_ids[k]=a[k];
   if(!si_col_values(g,a,SI_SICE_NX,scalar_values))return false;
   SiSiceColumnInput child_in={scalar_values,in.tso,t.diff_value,t.cap_value,
     in.dz,in.table,in.bare_visits,n,nd,in.bare_nvisits,in.myj,in.bare_pow,
     in.bare_record,in.bare_record_count,in.bare_record_capacity};
   SiSiceColumnTape child={};child.graph=t.child;child.c=t.child_c;
   child.r=t.child_r;child.tso=t.child_tso;
   if(!si_sice_column_record(child_in,child)||
      !si_col_import_column(g,child.graph,field_ids,lengths,t.child_map,
                            child.out,SI_SICE_NY,bare))return false;
   for(int k=0;k<n;++k)t.bare_tso[k]=t.child_map[child.tso[k]];
 }
 if(t.has_snow) {
   si_branch_pair(branches,51,t.mosaic);
   if(!in.snow_visits&&(!in.snow_record||!in.snow_record_count))return false;
   int a[SI_SNOW_NSCALAR]={x[IT_DELT],x[IT_CONFLX],st[7],st[0],x[IT_PRCPMS],
     st[10],st[6],st[4],st[3],opt[8],st[2],x[IT_PATM],x[IT_QVATM],x[IT_QCATM],
     x[IT_GLW],opt[4],opt[7],opt[5],x[IT_QKMS],x[IT_TKMS],x[IT_RHO],x[IT_LV],
     x[IT_CP],x[IT_ROVCP],x[IT_CW],x[IT_STBOLT],x[IT_TABS],x[IT_SOILT],
     x[IT_SOILT1],x[IT_TSNAV],x[IT_QVG],x[IT_QSG],x[IT_QCG],x[IT_SNOM],
     opt[6],x[IT_ZNT],x[IT_DEW],x[IT_S]};
   for(int k=0;k<SI_SNOW_NSCALAR;++k)scalar_ids[k]=a[k];
   if(!si_col_values(g,a,SI_SNOW_NSCALAR,scalar_values))return false;
   SiSnowColumnInput child_in={scalar_values,in.tso,t.diff_value,t.cap_value,
     in.zmain,in.zhalf,in.dz,in.table,in.snow_visits,n,nd,in.snow_nvisits,0,
     in.myj,in.snow_pow,in.snow_record,in.snow_record_count,in.snow_record_capacity};
   SiSnowColumnTape child={};child.graph=t.child;child.c=t.child_c;
   child.r=t.child_r;child.tso=t.child_tso;int snow[SI_SNOW_NOUTPUT];
   if(!si_snow_column_record(child_in,child)||
      !si_col_import_column(g,child.graph,field_ids,lengths,t.child_map,
                            child.out,SI_SNOW_NOUTPUT,snow))return false;
   for(int k=0;k<n;++k)t.tso[k]=t.child_map[child.tso[k]];
   const int dest[]={IT_SNHEI,IT_SNWE,IT_SNOWFRAC,IT_RHOSN,IT_EMISS,IT_ALB,
     IT_ZNT,IT_SNWEPRINT,IT_SNHEIPRINT,IT_RSM,IT_DEW,IT_SOILT,IT_SOILT1,
     IT_TSNAV,IT_QVG,IT_QSG,IT_QCG,IT_SMELT,IT_SNOH,IT_SNFLX,IT_SNOM,
     IT_EETA,IT_QFX,IT_HFX,IT_S,IT_SUBLIM,IT_PRCPL,IT_FLTOT};
   for(int k=0;k<SI_SNOW_NOUTPUT;++k)if(k!=SN_OUT_SNOWFRAC) {
     t.out[dest[k]]=snow[k];t.written[dest[k]]|=child.written[k];
   }
   const int snow_written[]={IT_EDIR,IT_EC,IT_ETT,IT_RUNOFF1,IT_RUNOFF2,
                             IT_SNOWFALLAC,IT_ACSNOW};
   for(unsigned k=0;k<sizeof(snow_written)/sizeof(snow_written[0]);++k)
     t.written[snow_written[k]]=1;
   t.out[IT_EDIR]=si_col_op(g,snow[SN_OUT_EETA],si_col_const(g,1.e-3f),2);
   t.out[IT_EC]=zero;t.out[IT_ETT]=zero;t.out[IT_RUNOFF1]=snow[SN_OUT_SMELT];
   t.out[IT_RUNOFF2]=zero;
   si_branch_pair(branches,53,t.mosaic);
   if(t.mosaic) {
     int frac=st[5];
     for(int k=0;k<n;++k)t.tso[k]=si_icetmp_blend(g,t.bare_tso[k],t.tso[k],frac,one);
     const int target[]={IT_DEW,IT_SOILT,IT_QVG,IT_QSG,IT_QCG,IT_EETA,IT_QFX,
                         IT_HFX,IT_S,IT_PRCPL,IT_FLTOT};
     const int from[]={SI_Y_DEW,SI_Y_SOILT,SI_Y_QVG,SI_Y_QSG,SI_Y_QCG,SI_Y_EETA,
                       SI_Y_QFX,SI_Y_HFX,SI_Y_S,SI_Y_PRCPL,SI_Y_FLTOT};
     for(int k=0;k<11;++k) {
       t.out[target[k]]=si_icetmp_blend(g,bare[from[k]],t.out[target[k]],frac,one);
       t.written[target[k]]=1;
     }
     t.written[IT_SUBLIM]=1;
     t.out[IT_SUBLIM]=t.out[IT_EETA];
     t.out[IT_RUNOFF1]=si_icetmp_blend(g,zero,t.out[IT_RUNOFF1],frac,one);
     t.out[IT_RUNOFF2]=si_icetmp_blend(g,zero,t.out[IT_RUNOFF2],frac,one);
     int b0[]={opt[0],t.out[IT_ALB],x[IT_ALB_SNOW_FREE],frac};
     SiIceOpticsBlendTape &blend=si_frame_slot<SiIceOpticsBlendTape>(13);int b[SI_ICE_OPTICS_BLEND_NOUTPUT];
     if(!si_col_values(g,b0,SI_ICE_OPTICS_BLEND_NINPUT,values)||
        !si_icetmp_optics_blend_record(values,state.keep_snow_albedo,blend,branches)||
        !si_col_import(g,blend.g,SI_ICE_OPTICS_BLEND_NINPUT,b0,
                       SI_ICE_OPTICS_BLEND_NOUTPUT,blend.out,b))return false;
     t.out[IT_ALB]=b[0];t.out[IT_EMISS]=b[1];
   }
   const bool snow_gone=g.node[t.out[IT_SNHEI]].value==0.f;
   si_branch_pair(branches,58,snow_gone);
   if(snow_gone) {
     t.out[IT_ALB]=x[IT_ALB_SNOW_FREE];t.iland=in.isice;
   } else {
     int two=si_col_const(g,2.f);
     const int candidate=si_col_op(g,t.out[IT_SNHEI],si_col_op(g,two,st[0],2),3);
     if(branches&&g.valid)si_branch(branches,91,g.node[one].value<g.node[candidate].value);
     t.out[IT_SNOWFRAC]=si_col_min(g,one,candidate);
   }
   t.out[IT_SNOWFALLAC]=si_col_op(g,x[IT_SNOWFALLAC],st[6],0);
   t.out[IT_ACSNOW]=t.out[IT_SNOWFALLAC];
 } else {
   for(int k=0;k<n;++k)t.tso[k]=t.bare_tso[k];
   const int target[]={IT_QVG,IT_QSG,IT_QCG,IT_SOILT,IT_DEW,IT_EETA,IT_QFX,
                       IT_HFX,IT_S,IT_EVAPL,IT_PRCPL,IT_FLTOT};
   for(int k=0;k<SI_SICE_NY;++k){t.out[target[k]]=bare[k];t.written[target[k]]=1;}
   const int bare_written[]={IT_SNWEPRINT,IT_SNHEIPRINT,IT_EDIR,IT_EC,
                             IT_ETT,IT_RUNOFF1,IT_RUNOFF2};
   for(unsigned k=0;k<sizeof(bare_written)/sizeof(bare_written[0]);++k)
     t.written[bare_written[k]]=1;
   t.out[IT_SNWEPRINT]=zero;t.out[IT_SNHEIPRINT]=zero;
   t.out[IT_EDIR]=si_col_op(g,bare[SI_Y_EETA],si_col_const(g,1.e-3f),2);
   t.out[IT_EC]=zero;t.out[IT_ETT]=zero;t.out[IT_RUNOFF1]=x[IT_PRCPMS];
   t.out[IT_RUNOFF2]=zero;
 }
 return g.valid;
}
#ifdef SI_SEAICE_ORIGINAL_MATH_AVAILABLE
// Saves the values actually returned by the original math helpers; all
// continuous arguments are still leaves of the derivative graph.
__device__ __forceinline__ bool si_icetmp_column_record_aug(
        SiIceTmpInput in,SiIceTmpTape &t,unsigned int *branches=nullptr) {
 SiIceMathRecord math;
 if(!si_icetmp_capture_math(in.scalar,math))return false;
 in.unary=math.unary;in.bare_pow=math.power;in.snow_pow=math.power;
 return si_icetmp_column_record(in,t,branches);
}
#endif
// The scalar boundary of ice_ruc_ice, in source argument order where
// possible. Local slots are sibling-state inputs for arena/stack values which
// the pinned forward can pass into ice_icetmp before overwriting them. In a
// cold start the source initializes the corresponding arena slots to zero.
// Inputs and outputs which the forward leaves unchanged keep the same node.
enum SiRucIceScalar {
 RI_DT,RI_TABS,RI_QVATM,RI_QCATM,RI_P8W_KMS,RI_P8W_1,RI_Z3D,RI_RHO,
 RI_RAINBL,RI_RAINNCV,RI_FRZFRAC,RI_SNOWNCV,RI_GRAUPELNCV,RI_FLQC,RI_FLHC,
 RI_CP,RI_ROVCP,RI_G0,RI_LV,RI_STBOLT,RI_GLW,RI_GSW,RI_ALB,RI_ALBBCK,
 RI_SOILT,RI_SOILT1,RI_TSNAV,RI_SNOW,RI_SNOWH,RI_SNOWC,RI_QSFC,RI_QSG,
 RI_QVG,RI_QCG,RI_HFX,RI_QFX,RI_LH,RI_DEW,RI_SFCRUNOFF,RI_ACRUNOFF,
 RI_SFCEVP,RI_GRDFLX,RI_SNOWFALLAC,RI_ACSNOW,RI_SNOM,RI_PRECIPFR,
 RI_RHOSNF,RI_CHKLOWQ,RI_ZNT,RI_Z0,RI_SNOALB,RI_EMISS,RI_XICE,
 RI_XICE_THRESHOLD,RI_LOCAL_SMELT,RI_LOCAL_SNOH,RI_LOCAL_SNFLX,
 RI_LOCAL_EDIR,RI_LOCAL_EC,RI_LOCAL_ETT,RI_LOCAL_SUBLIM,RI_LOCAL_SFLX,
 RI_LOCAL_SMF,RI_LOCAL_EVAPL,RI_LOCAL_PRCPL,RI_LOCAL_BUDGET,
 RI_LOCAL_RUNOFF1,RI_LOCAL_RUNOFF2,RI_LOCAL_INFILTR,RI_LOCAL_RSM,
 RI_LOCAL_SNWEPRINT,RI_LOCAL_SNHEIPRINT,SI_RUC_NSCALAR
};
struct SiRucIceInput {
 const float *scalar,*tso,*zs,*table;
 const SiIceUnaryRecord *unary;
 const int *bare_visits,*snow_visits;
 int nzs,ntable,bare_nvisits,snow_nvisits,isice;
 bool cold_start,frpcpn,has_graupel,myj;
 float bare_pow,snow_pow;
 int *bare_record,*snow_record,*bare_record_count,*snow_record_count;
 int bare_record_capacity,snow_record_capacity;
};
// The caller owns the parent, icetmp and nested-column graphs, their leaves,
// and all scratch arrays. map has at least ice.graph.capacity integers;
// tso/zmain/zhalf have nzs elements and dz has 2*(nzs-2) elements.
struct SiRucIceTape {
 SiColumnGraph graph;
 SiIceTmpTape ice;
 int *tso,*zmain,*zhalf,*dz,*map;
 float *tso_value,*zmain_value,*zhalf_value,*dz_value;
 int out[SI_RUC_NSCALAR];
 unsigned char written[SI_RUC_NSCALAR],tso_written;
 bool active,cold_start,reset_soilt1,clamp_qcg,reset_qvg,scale_snowfrac;
};
__device__ __forceinline__ bool si_ruc_ice_column_record(
        const SiRucIceInput &in,SiRucIceTape &t,unsigned int *branches=nullptr) {
 const int n=in.nzs,nd=2*(n-2);
 if(n<3||in.ntable<5001||!in.scalar||!in.tso||!in.zs||!in.table||
    !t.graph.node||!t.graph.leaf||!t.tso||!t.zmain||!t.zhalf||!t.dz||
    !t.map||!t.tso_value||!t.zmain_value||!t.zhalf_value||!t.dz_value)
    return false;
 SiColumnGraph &g=t.graph;g.count=0;g.nleaf=0;g.valid=true;
 int x[SI_RUC_NSCALAR];
 // pk120 (x1.40962): the RI_LOCAL_* slots are arena/stack values the forward passes into ice_icetmp and overwrites
 // there before any read (runoff2 = 0 at L1171/L1225, smelt = 0, ...), so an incoming value is dead. Outside a cold
 // start that arena word is whatever the slot held, and a non-finite one made the first leaf invalid: pk120e4 step 2,
 // RI_LOCAL_RUNOFF2 on one of 1,707 sea-ice columns, 'oracle_ice_ruc_ice augmented primal refused (failure code 1)'.
 // Such a dead non-finite input enters as 0 (its slope is 0 either way). If the forward did read it, the augmented
 // primal would no longer equal the forward's post state, and the runner's word-for-word compare refuses.
 for(int i=0;i<SI_RUC_NSCALAR;++i){float v=in.scalar[i];if(i>=RI_LOCAL_SMELT&&!si_finite(v))v=0.f;
   x[i]=si_col_var(g,v,SI_SNOW_SCALAR,i);t.out[i]=x[i];t.written[i]=0;}
 for(int k=0;k<n;++k)t.tso[k]=si_col_var(g,in.tso[k],SI_SNOW_TSO,k);
 t.active=in.scalar[RI_XICE]>=in.scalar[RI_XICE_THRESHOLD];
 t.tso_written=t.active;
 t.cold_start=t.active&&in.cold_start;
 t.reset_soilt1=t.clamp_qcg=t.reset_qvg=t.scale_snowfrac=false;
 if(!g.valid)return false;
 si_branch_pair(branches,12,t.active);
 if(in.cold_start)si_branch_pair(branches,21,t.active);
 si_branch_pair(branches,43,t.active);
 if(!t.active)return true;
 int zero=si_col_const(g,0.f),one=si_col_const(g,1.f);
 int half=si_col_const(g,.5f),freeze=si_col_const(g,273.15f);
 int soilt1=x[RI_SOILT1],tsnav=x[RI_TSNAV],qcg=x[RI_QCG];
 if(t.cold_start) {
   t.reset_soilt1=in.scalar[RI_SOILT1]<170.f||in.scalar[RI_SOILT1]>400.f;
   if(branches){bool left=in.scalar[RI_SOILT1]<170.f;
     si_branch(branches,23,t.reset_soilt1);si_branch(branches,24,left);
     if(!left)si_branch(branches,25,in.scalar[RI_SOILT1]>400.f);}
   int mean=si_col_op(g,half,si_col_op(g,x[RI_SOILT],t.tso[0],0),2);
   if(t.reset_soilt1) {
     si_branch_pair(branches,26,in.scalar[RI_SNOWC]>0.f);
     soilt1=in.scalar[RI_SNOWC]>0.f?mean:t.tso[0];
   }
   tsnav=si_col_op(g,mean,freeze,1);
   int qsat;if(!si_col_qsn(g,x[RI_SOILT],in.table,qsat))return false;
   int p=si_col_op(g,x[RI_P8W_KMS],si_col_const(g,.01f),2);
   int qsg=si_col_op(g,qsat,p,3);
   t.clamp_qcg=in.scalar[RI_QCG]<0.f||in.scalar[RI_QCG]>.1f;
   if(branches){bool left=in.scalar[RI_QCG]<0.f;
     si_branch(branches,29,t.clamp_qcg);si_branch(branches,30,left);
     if(!left)si_branch(branches,31,in.scalar[RI_QCG]>.1f);}
   if(t.clamp_qcg)qcg=zero;
   t.reset_qvg=in.scalar[RI_QVG]<=0.f||in.scalar[RI_QVG]>.1f;
   if(branches){bool left=in.scalar[RI_QVG]<=0.f;
     si_branch(branches,33,t.reset_qvg);si_branch(branches,34,left);
     if(!left)si_branch(branches,35,in.scalar[RI_QVG]>.1f);}
   int qvg=t.reset_qvg?qsg:x[RI_QVG];
   t.out[RI_QSFC]=si_col_op(g,qvg,si_col_op(g,one,qvg,0),3);
   const int reset[]={RI_SNOM,RI_SNOWFALLAC,RI_ACSNOW,RI_PRECIPFR,RI_DEW,
     RI_SFCRUNOFF,RI_ACRUNOFF,RI_LOCAL_SMELT,RI_LOCAL_SNOH,RI_LOCAL_SNFLX,
     RI_LOCAL_EDIR,RI_LOCAL_EC,RI_LOCAL_ETT,RI_LOCAL_SUBLIM,RI_LOCAL_SFLX,
     RI_LOCAL_SMF,RI_LOCAL_EVAPL,RI_LOCAL_PRCPL,RI_LOCAL_RUNOFF1,
     RI_LOCAL_RUNOFF2};
   for(unsigned k=0;k<sizeof(reset)/sizeof(reset[0]);++k){t.out[reset[k]]=zero;t.written[reset[k]]=1;}
   t.written[RI_SOILT1]=t.reset_soilt1;t.written[RI_TSNAV]=1;
   t.written[RI_QSG]=1;t.written[RI_QSFC]=1;
   t.written[RI_QCG]=t.clamp_qcg;t.written[RI_QVG]=t.reset_qvg;
   t.written[RI_RHOSNF]=1;t.written[RI_CHKLOWQ]=1;
   t.out[RI_RHOSNF]=si_col_const(g,-1000.f);t.out[RI_CHKLOWQ]=one;
 }
 const int always[]={RI_PRECIPFR,RI_ZNT,RI_SNOALB,RI_QVG,RI_QSG,RI_QSFC,
   RI_Z0,RI_EMISS,RI_SNOW,RI_SNOWH,RI_GRDFLX,RI_SNOWC,RI_RHOSNF,RI_SFCEVP};
 for(unsigned k=0;k<sizeof(always)/sizeof(always[0]);++k)t.written[always[k]]=1;
 int preinput[]={x[RI_TABS],x[RI_QVATM],x[RI_QCATM],x[RI_P8W_KMS],x[RI_Z3D],
   x[RI_RHO],x[RI_RAINBL],x[RI_RAINNCV],x[RI_FRZFRAC],x[RI_SNOWNCV],
   x[RI_GRAUPELNCV],x[RI_DT],x[RI_FLQC],x[RI_FLHC],x[RI_CP],x[RI_SNOW],
   x[RI_SNOWH],x[RI_SNOWC],t.out[RI_RHOSNF]};
 float pv[SI_ICETMP_NSCALAR];int pre[SI_ICE_PRE_NOUTPUT];SiIcePreTape &prep=si_frame_slot<SiIcePreTape>(14);
 if(!si_col_values(g,preinput,SI_ICE_PRE_NINPUT,pv)||
    !si_ice_pre_record(pv,in.frpcpn,in.has_graupel,prep,branches)||
    !si_col_import(g,prep.g,SI_ICE_PRE_NINPUT,preinput,SI_ICE_PRE_NOUTPUT,
                   prep.out,pre))return false;
 t.out[RI_PRECIPFR]=pre[12];
 t.zmain[0]=zero;t.zhalf[0]=zero;
 for(int k=1;k<n;++k){
   t.zmain[k]=si_col_var(g,in.zs[k],SI_SNOW_ZMAIN,k);
   t.zhalf[k]=si_col_op(g,half,si_col_op(g,t.zmain[k-1],t.zmain[k],0),2);
 }
 for(int k=1;k<n-1;++k){
   int z=si_col_op(g,si_col_op(g,x[RI_DT],si_col_const(g,2.f),3),
                      si_col_op(g,t.zhalf[k+1],t.zhalf[k],1),3);
   t.dz[2*(k-1)]=si_col_op(g,z,si_col_op(g,t.zmain[k],t.zmain[k-1],1),3);
   t.dz[2*(k-1)+1]=si_col_op(g,z,si_col_op(g,t.zmain[k+1],t.zmain[k],1),3);
 }
 int qsat;if(!si_col_qsn(g,x[RI_SOILT],in.table,qsat))return false;
 int patmb=si_col_op(g,x[RI_P8W_1],si_col_const(g,.01f),2);
 int qvg=si_col_op(g,qsat,patmb,3);
 int icecap=si_col_const(g,271.4f);
 for(int k=0;k<n;++k) {
   if(branches&&g.valid)si_branch(branches,95,g.node[icecap].value<g.node[t.tso[k]].value);
   t.tso[k]=si_col_min(g,icecap,t.tso[k]);
 }
 int it[SI_ICETMP_NSCALAR];for(int i=0;i<SI_ICETMP_NSCALAR;++i)it[i]=zero;
 it[IT_DELT]=x[RI_DT];it[IT_CONFLX]=pre[4];it[IT_PRCPMS]=pre[6];
 it[IT_NEWSNMS]=pre[7];it[IT_SNOWRAT]=pre[8];it[IT_GRAUPRAT]=pre[9];
 it[IT_ICERAT]=pre[10];it[IT_CURAT]=pre[11];it[IT_PATM]=pre[3];
 it[IT_TABS]=x[RI_TABS];it[IT_QVATM]=x[RI_QVATM];it[IT_QCATM]=x[RI_QCATM];
 it[IT_RHO]=x[RI_RHO];it[IT_GLW]=x[RI_GLW];it[IT_GSW]=x[RI_GSW];
 it[IT_EMISS]=si_col_const(g,.98f);it[IT_QKMS]=pre[13];it[IT_TKMS]=pre[14];
 it[IT_ALB]=x[RI_ALB];it[IT_ZNT]=si_col_const(g,.011f);
 it[IT_ALB_SNOW]=si_col_const(g,.75f);it[IT_ALB_SNOW_FREE]=x[RI_ALBBCK];
 it[IT_CP]=x[RI_CP];it[IT_ROVCP]=x[RI_ROVCP];it[IT_G0]=x[RI_G0];
 it[IT_LV]=x[RI_LV];it[IT_STBOLT]=x[RI_STBOLT];it[IT_CW]=si_col_const(g,4.183e6f);
 it[IT_C1SN]=si_col_const(g,.026f);it[IT_C2SN]=si_col_const(g,21.f);
 it[IT_SOILT]=x[RI_SOILT];it[IT_SOILT1]=soilt1;it[IT_TSNAV]=tsnav;
 it[IT_RHOSN]=pre[19];it[IT_RHONEWSN]=si_col_const(g,200.f);
 it[IT_RHOSNFALL]=pre[18];it[IT_SNWE]=pre[15];it[IT_SNHEI]=pre[16];
 it[IT_SNOWFRAC]=pre[17];it[IT_RSM]=t.out[RI_LOCAL_RSM];
 it[IT_SNWEPRINT]=t.out[RI_LOCAL_SNWEPRINT];it[IT_SNHEIPRINT]=t.out[RI_LOCAL_SNHEIPRINT];
 it[IT_DEW]=t.out[RI_DEW];it[IT_QVG]=qvg;it[IT_QSG]=qvg;it[IT_QCG]=qcg;
 it[IT_SMELT]=t.out[RI_LOCAL_SMELT];it[IT_SNOH]=t.out[RI_LOCAL_SNOH];
 it[IT_SNFLX]=t.out[RI_LOCAL_SNFLX];it[IT_SNOM]=t.out[RI_SNOM];
 it[IT_SNOWFALLAC]=t.out[RI_SNOWFALLAC];it[IT_ACSNOW]=t.out[RI_ACSNOW];
 it[IT_EDIR]=t.out[RI_LOCAL_EDIR];it[IT_EC]=t.out[RI_LOCAL_EC];
 it[IT_ETT]=t.out[RI_LOCAL_ETT];it[IT_EETA]=x[RI_QFX];it[IT_QFX]=x[RI_LH];
 it[IT_HFX]=x[RI_HFX];it[IT_S]=t.out[RI_LOCAL_SFLX];it[IT_SUBLIM]=t.out[RI_LOCAL_SUBLIM];
 it[IT_EVAPL]=t.out[RI_LOCAL_EVAPL];it[IT_PRCPL]=t.out[RI_LOCAL_PRCPL];
 it[IT_FLTOT]=t.out[RI_LOCAL_BUDGET];it[IT_RUNOFF1]=t.out[RI_LOCAL_RUNOFF1];
 it[IT_RUNOFF2]=t.out[RI_LOCAL_RUNOFF2];it[IT_INFILTR]=t.out[RI_LOCAL_INFILTR];
 it[IT_SMF]=t.out[RI_LOCAL_SMF];
 if(!si_col_values(g,it,SI_ICETMP_NSCALAR,pv))return false;
 for(int k=0;k<n;++k){t.tso_value[k]=g.node[t.tso[k]].value;
   t.zmain_value[k]=g.node[t.zmain[k]].value;t.zhalf_value[k]=g.node[t.zhalf[k]].value;}
 for(int k=0;k<nd;++k)t.dz_value[k]=g.node[t.dz[k]].value;
 SiIceTmpInput child={pv,t.tso_value,t.zmain_value,t.zhalf_value,t.dz_value,
   in.table,in.unary,in.bare_visits,in.snow_visits,n,nd,in.ntable,
   in.bare_nvisits,in.snow_nvisits,in.isice,in.isice,in.myj,in.bare_pow,in.snow_pow,
   in.bare_record,in.snow_record,in.bare_record_count,in.snow_record_count,
   in.bare_record_capacity,in.snow_record_capacity};
#ifdef SI_SEAICE_ORIGINAL_MATH_AVAILABLE
 if(in.unary) {
   if(!si_icetmp_column_record(child,t.ice))return false;
 } else if(!si_icetmp_column_record_aug(child,t.ice))return false;
#else
 if(!si_icetmp_column_record(child,t.ice))return false;
#endif
 const int *fields[SI_SNOW_NFIELD]={it,t.tso,nullptr,nullptr,t.zmain,t.zhalf,t.dz,nullptr};
 int lengths[SI_SNOW_NFIELD]={SI_ICETMP_NSCALAR,n,0,0,n,n,nd,in.ntable};
 int result[SI_ICETMP_NSCALAR];
 if(!si_col_import_column(g,t.ice.graph,fields,lengths,t.map,t.ice.out,
                          SI_ICETMP_NSCALAR,result))return false;
 for(int k=0;k<n;++k)t.tso[k]=t.map[t.ice.tso[k]];
 const int dst[]={RI_ALB,RI_ZNT,RI_SOILT,RI_SOILT1,RI_TSNAV,RI_DEW,RI_QVG,
   RI_QSG,RI_QCG,RI_SNOM,RI_SNOWFALLAC,RI_ACSNOW,RI_QFX,RI_LH,RI_HFX};
 const int src[]={IT_ALB,IT_ZNT,IT_SOILT,IT_SOILT1,IT_TSNAV,IT_DEW,IT_QVG,
   IT_QSG,IT_QCG,IT_SNOM,IT_SNOWFALLAC,IT_ACSNOW,IT_EETA,IT_QFX,IT_HFX};
 for(unsigned k=0;k<sizeof(dst)/sizeof(dst[0]);++k) {
   t.out[dst[k]]=result[src[k]];t.written[dst[k]]|=t.ice.written[src[k]];
 }
 const int local_dst[]={RI_LOCAL_SMELT,RI_LOCAL_SNOH,RI_LOCAL_SNFLX,RI_LOCAL_EDIR,
   RI_LOCAL_EC,RI_LOCAL_ETT,RI_LOCAL_SUBLIM,RI_LOCAL_SFLX,RI_LOCAL_SMF,
   RI_LOCAL_EVAPL,RI_LOCAL_PRCPL,RI_LOCAL_BUDGET,RI_LOCAL_RUNOFF1,RI_LOCAL_RUNOFF2,
   RI_LOCAL_INFILTR,RI_LOCAL_RSM,RI_LOCAL_SNWEPRINT,RI_LOCAL_SNHEIPRINT};
 const int local_src[]={IT_SMELT,IT_SNOH,IT_SNFLX,IT_EDIR,IT_EC,IT_ETT,IT_SUBLIM,
   IT_S,IT_SMF,IT_EVAPL,IT_PRCPL,IT_FLTOT,IT_RUNOFF1,IT_RUNOFF2,IT_INFILTR,
   IT_RSM,IT_SNWEPRINT,IT_SNHEIPRINT};
 for(unsigned k=0;k<sizeof(local_dst)/sizeof(local_dst[0]);++k) {
   t.out[local_dst[k]]=result[local_src[k]];
   t.written[local_dst[k]]|=t.ice.written[local_src[k]];
 }
 t.out[RI_SNOALB]=it[IT_ALB_SNOW];t.out[RI_Z0]=result[IT_ZNT];
 t.out[RI_QSFC]=si_col_op(g,result[IT_QVG],si_col_op(g,one,result[IT_QVG],0),3);
 si_branch_pair(branches,76,in.scalar[RI_SNOW]==0.f);
 t.out[RI_EMISS]=in.scalar[RI_SNOW]==0.f?si_col_const(g,.98f):result[IT_EMISS];
 t.out[RI_SNOW]=si_col_op(g,result[IT_SNWE],si_col_const(g,1000.f),2);
 t.out[RI_SNOWH]=result[IT_SNHEI];
 t.out[RI_GRDFLX]=si_col_op(g,si_col_const(g,-1.f),result[IT_S],2);
 t.scale_snowfrac=g.node[result[IT_SNOWFRAC]].value>0.f;
 if(branches){si_branch(branches,79,t.scale_snowfrac&&t.active);
               si_branch(branches,80,t.scale_snowfrac);
               if(t.scale_snowfrac)si_branch(branches,81,t.active);}
 t.out[RI_SNOWC]=t.scale_snowfrac?si_col_op(g,result[IT_SNOWFRAC],x[RI_XICE],2):result[IT_SNOWFRAC];
 t.out[RI_RHOSNF]=result[IT_RHOSNFALL];
 t.out[RI_SFCEVP]=si_col_op(g,x[RI_SFCEVP],si_col_op(g,result[IT_EETA],x[RI_DT],2),0);
 return g.valid;
}

// The threshold is a predicate only: its executed-arm derivative is zero.
#if !defined(SI_COMPILE_ONE) || defined(SI_PREP_TL)
extern "C" __global__ void oracle_drivers_seaice_prepare__tl(
    unsigned long long *p, unsigned long long *d, int nc, SeaiceFailure *fail,
    unsigned int *branches) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    const int n = si_i(p, 0)[c];
    const float threshold = si_f(p, 1)[c];
    if (!si_check(threshold, fail, c, 2, c, 5007)) return;
    for (int i = 0; i < n; ++i) {
        const int k = c + i * nc;
        const float x = si_f(p, 2)[k], a = si_f(p, 3)[k], e = si_f(p, 4)[k];
        const float dx = si_f(d, 2)[k], da = si_f(d, 3)[k], de = si_f(d, 4)[k];
        if (!si_check(x, fail, c, 3, k, 5007) || !si_check(a, fail, c, 4, k, 5008) ||
            !si_check(e, fail, c, 5, k, 5009) || !si_check(dx, fail, c, 3, k, 5007) ||
            !si_check(da, fail, c, 4, k, 5008) || !si_check(de, fail, c, 5, k, 5009)) return;
        float va = da, ve = de;
        const bool left = x >= threshold, right = left && x < 1.0f;
        si_branch(branches,0,right); si_branch(branches,1,left);
        if (left) si_branch(branches,2,x < 1.0f);
        if (right) {
            if (x == 0.0f) { si_fail(fail, c, 3, k, 5008); return; }
            const float inv = __fdiv_rn(1.0f, x);
            const float na = __fsub_rn(a, __fmul_rn(0.08f, __fsub_rn(1.0f, x)));
            const float ne = __fsub_rn(e, __fmul_rn(0.98f, __fsub_rn(1.0f, x)));
            va = __fsub_rn(__fmul_rn(__fadd_rn(da, __fmul_rn(0.08f, dx)), inv),
                           __fmul_rn(__fmul_rn(__fdiv_rn(na, x), inv), dx));
            ve = __fsub_rn(__fmul_rn(__fadd_rn(de, __fmul_rn(0.98f, dx)), inv),
                           __fmul_rn(__fmul_rn(__fdiv_rn(ne, x), inv), dx));
        }
        if (!si_check(va, fail, c, 6, k, 5008) || !si_check(ve, fail, c, 7, k, 5009)) return;
        si_f(d, 5)[k] = va; si_f(d, 6)[k] = ve;
    }
}
#endif

#if !defined(SI_COMPILE_ONE) || defined(SI_PREP_AD)
extern "C" __global__ void oracle_drivers_seaice_prepare__ad(
    unsigned long long *p, unsigned long long *b, int nc, SeaiceFailure *fail) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    const int n = si_i(p, 0)[c];
    const float threshold = si_f(p, 1)[c];
    if (!si_check(threshold, fail, c, 2, c, 5007)) return;
    for (int i = n - 1; i >= 0; --i) {
        const int k = c + i * nc;
        const float x = si_f(p, 2)[k], a = si_f(p, 3)[k], e = si_f(p, 4)[k];
        const float ba = si_f(b, 5)[k], be = si_f(b, 6)[k];
        if (!si_check(x, fail, c, 3, k, 5007) || !si_check(a, fail, c, 4, k, 5008) ||
            !si_check(e, fail, c, 5, k, 5009) || !si_check(ba, fail, c, 6, k, 5008) ||
            !si_check(be, fail, c, 7, k, 5009)) return;
        double ax = 0.; float aa = ba, ae = be;
        if (x >= threshold && x < 1.0f) {
            if (x == 0.0f) { si_fail(fail, c, 3, k, 5008); return; }
            const float inv = __fdiv_rn(1.f, x);
            const float na = __fsub_rn(a, __fmul_rn(0.08f, __fsub_rn(1.f, x)));
            const float ne = __fsub_rn(e, __fmul_rn(0.98f, __fsub_rn(1.f, x)));
            aa = __fmul_rn(ba, inv); ae = __fmul_rn(be, inv);
            ax = double(ba)*double(__fsub_rn(__fmul_rn(0.08f, inv), __fmul_rn(__fdiv_rn(na, x), inv))) +
                 double(be)*double(__fsub_rn(__fmul_rn(0.98f, inv), __fmul_rn(__fdiv_rn(ne, x), inv)));
        }
        if (!si_check(float(ax), fail, c, 3, k, 5008) || !si_check(aa, fail, c, 4, k, 5008) ||
            !si_check(ae, fail, c, 5, k, 5009)) return;
        float rx=float(double(si_f(b,2)[k])+ax),ra=__fadd_rn(si_f(b,3)[k],aa);
        float re=__fadd_rn(si_f(b,4)[k],ae);
        if(!si_check(rx,fail,c,3,k,5008)||!si_check(ra,fail,c,4,k,5008)||
           !si_check(re,fail,c,5,k,5009))return;
        si_f(b, 5)[k] = 0.f; si_f(b, 6)[k] = 0.f;
        si_f(b, 2)[k]=rx;si_f(b, 3)[k]=ra;si_f(b, 4)[k]=re;
    }
}
#endif

__device__ __forceinline__ bool si_mix_tl(unsigned long long *p, unsigned long long *d,
                                          int pi, int si, int k, float x, float dx,
                                          SeaiceFailure *f, int c) {
    float a=si_f(p,pi)[k], s=si_f(p,si)[k], da=si_f(d,pi)[k], ds=si_f(d,si)[k];
    if (!si_check(a,f,c,pi+1,k,5028) || !si_check(s,f,c,si+1,k,5028) ||
        !si_check(da,f,c,pi+1,k,5028) || !si_check(ds,f,c,si+1,k,5028)) return false;
    float z=__fadd_rn(__fadd_rn(__fmul_rn(x,da),__fmul_rn(__fsub_rn(1.f,x),ds)),
                      __fmul_rn(dx,__fsub_rn(a,s)));
    if (!si_check(z,f,c,pi+1,k,5028)) return false;
    si_f(d,pi)[k]=z; return true;
}
__device__ __forceinline__ bool si_constmix_tl(unsigned long long *p, unsigned long long *d,
                                               int pi,int k,float x,float dx,float s,
                                               SeaiceFailure *f,int c) {
    float a=si_f(p,pi)[k], da=si_f(d,pi)[k];
    if (!si_check(a,f,c,pi+1,k,5046) || !si_check(da,f,c,pi+1,k,5046)) return false;
    float z=__fadd_rn(__fmul_rn(x,da),__fmul_rn(dx,__fsub_rn(a,s)));
    if (!si_check(z,f,c,pi+1,k,5046)) return false;
    si_f(d,pi)[k]=z; return true;
}
#if !defined(SI_COMPILE_ONE) || defined(SI_FINISH_TL)
extern "C" __global__ void oracle_drivers_seaice_finish__tl(
    unsigned long long *p, unsigned long long *d, int nc, SeaiceFailure *fail,
    unsigned int *branches) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x; if (c>=nc) return;
    const int n=si_i(p,0)[c], config=si_i(p,2)[c], mynn=si_i(p,3)[c];
    if (!config) return;
    const float threshold=config?si_f(p,1)[c]:0.f;
    if (!si_check(threshold,fail,c,2,c,5026)) return;
    for(int i=0;i<n;++i) {
        int k=c+i*nc; float x=si_f(p,4)[k];
        if (!si_check(x,fail,c,5,k,5026)) return;
        const bool left=x>=threshold, right=left && x<1.f;
        si_branch(branches,0,right); si_branch(branches,1,left);
        if(left) si_branch(branches,2,x<1.f);
        if (!right) continue;
        float dx=si_f(d,4)[k]; if (!si_check(dx,fail,c,5,k,5026)) return;
        if (!si_mix_tl(p,d,5,6,k,x,dx,fail,c) || !si_mix_tl(p,d,7,8,k,x,dx,fail,c)) return;
        if (mynn) {
            if (!si_mix_tl(p,d,9,10,k,x,dx,fail,c)) return;
        } else {
            // The original cqs value is overwritten by a blend of the OLD chs.
            float a=si_f(p,13)[k], s=si_f(p,14)[k], da=si_f(d,13)[k], ds=si_f(d,14)[k];
            if (!si_check(a,fail,c,14,k,5033) || !si_check(s,fail,c,15,k,5033) ||
                !si_check(da,fail,c,14,k,5033) || !si_check(ds,fail,c,15,k,5033)) return;
            float z=__fadd_rn(__fadd_rn(__fmul_rn(x,da),__fmul_rn(__fsub_rn(1.f,x),ds)),
                              __fmul_rn(dx,__fsub_rn(a,s)));
            if (!si_check(z,fail,c,10,k,5033)) return;
            si_f(d,9)[k]=z;
        }
        for (int j=11;j<=29;j+=2) if (!si_mix_tl(p,d,j,j+1,k,x,dx,fail,c)) return;
        if (!si_constmix_tl(p,d,31,k,x,dx,0.08f,fail,c) ||
            !si_constmix_tl(p,d,32,k,x,dx,0.98f,fail,c)) return;
    }
}
#endif
// Reverse uses distinct output-seed and input-accumulator tables. Only seeds
// for locations the original forward overwrote are consumed and zeroed.
__device__ __forceinline__ bool si_mix_ad(unsigned long long *p,unsigned long long *bo,
                                          unsigned long long *bi,
                                          int pi,int si,int k,float x,double &bx,
                                          SeaiceFailure *f,int c) {
    float a=si_f(p,pi)[k],s=si_f(p,si)[k],v=si_f(bo,pi)[k];
    if (!si_check(a,f,c,pi+1,k,5028) || !si_check(s,f,c,si+1,k,5028) ||
        !si_check(v,f,c,pi+1,k,5028) || !si_check(si_f(bi,pi)[k],f,c,pi+1,k,5028) ||
        !si_check(si_f(bi,si)[k],f,c,si+1,k,5028)) return false;
    double xx=bx+double(v)*double(__fsub_rn(a,s));
    float pp=float(double(si_f(bi,pi)[k])+double(x)*double(v));
    float ss=float(double(si_f(bi,si)[k])+double(__fsub_rn(1.f,x))*double(v));
    if(!si_check(float(xx),f,c,5,k,5028)||!si_check(pp,f,c,pi+1,k,5028)||
       !si_check(ss,f,c,si+1,k,5028))return false;
    bx=xx;si_f(bo,pi)[k]=0.f;si_f(bi,pi)[k]=pp;si_f(bi,si)[k]=ss;
    return true;
}
__device__ __forceinline__ bool si_constmix_ad(unsigned long long *p,unsigned long long *bo,
                                               unsigned long long *bi,
                                               int pi,int k,float x,double &bx,float s,
                                               SeaiceFailure *f,int c) {
    float a=si_f(p,pi)[k],v=si_f(bo,pi)[k];
    if (!si_check(a,f,c,pi+1,k,5046) || !si_check(v,f,c,pi+1,k,5046) ||
        !si_check(si_f(bi,pi)[k],f,c,pi+1,k,5046)) return false;
    double xx=bx+double(v)*double(__fsub_rn(a,s));
    float pp=float(double(si_f(bi,pi)[k])+double(x)*double(v));
    if(!si_check(float(xx),f,c,5,k,5046)||!si_check(pp,f,c,pi+1,k,5046))return false;
    bx=xx;si_f(bo,pi)[k]=0.f;si_f(bi,pi)[k]=pp;return true;
}
__device__ __forceinline__ bool si_finish_passthrough(unsigned long long *bo,
        unsigned long long *bi,int pi,int k,SeaiceFailure *f,int c) {
    float seed=si_f(bo,pi)[k],prior=si_f(bi,pi)[k];
    if(!si_check(seed,f,c,pi+1,k,5028)||!si_check(prior,f,c,pi+1,k,5028))return false;
    float result=float(double(prior)+double(seed));
    if(!si_check(result,f,c,pi+1,k,5028))return false;
    si_f(bi,pi)[k]=result;return true;
}
#if !defined(SI_COMPILE_ONE) || defined(SI_FINISH_AD)
extern "C" __global__ void oracle_drivers_seaice_finish__ad(
    unsigned long long *p,unsigned long long *bo,unsigned long long *bi,
    int nc,SeaiceFailure *fail) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=nc)return;
    const int n=si_i(p,0)[c],config=si_i(p,2)[c],mynn=si_i(p,3)[c];
    const float threshold=config?si_f(p,1)[c]:0.f;
    if(config&&!si_check(threshold,fail,c,2,c,5026))return;
    for(int i=n-1;i>=0;--i) {
        int k=c+i*nc; float x=0.f;
        if(config){x=si_f(p,4)[k];if(!si_check(x,fail,c,5,k,5026))return;}
        if(!config||!(x>=threshold && x<1.f)) {
            for(int j=5;j<=29;j+=2)if(!si_finish_passthrough(bo,bi,j,k,fail,c))return;
            if(!si_finish_passthrough(bo,bi,31,k,fail,c)||
               !si_finish_passthrough(bo,bi,32,k,fail,c))return;
            continue;
        }
        double bx=0.;
        if(!si_constmix_ad(p,bo,bi,32,k,x,bx,0.98f,fail,c) ||
           !si_constmix_ad(p,bo,bi,31,k,x,bx,0.08f,fail,c))return;
        for(int j=29;j>=11;j-=2) {
            if(j==13 && !mynn) {
                // The old chs values feed both the chs and overwritten cqs
                // outputs. Accumulate their cotangents before one f32 write.
                float a=si_f(p,13)[k],s=si_f(p,14)[k];
                float v=si_f(bo,13)[k],u=si_f(bo,9)[k],
                      old_p=si_f(bi,13)[k],old_s=si_f(bi,14)[k];
                if(!si_check(a,fail,c,14,k,5033)||!si_check(s,fail,c,15,k,5033)||
                   !si_check(v,fail,c,14,k,5033)||!si_check(u,fail,c,10,k,5033)||
                   !si_check(old_p,fail,c,14,k,5033)||!si_check(old_s,fail,c,15,k,5033))return;
                bx+=double(v)*double(__fsub_rn(a,s));
                bx+=double(u)*double(__fsub_rn(a,s));
                float pp=float(double(old_p)+double(x)*double(v)+double(x)*double(u));
                double omx=double(__fsub_rn(1.f,x));
                float ss=float(double(old_s)+omx*double(v)+omx*double(u));
                if(!si_check(float(bx),fail,c,5,k,5033)||!si_check(pp,fail,c,14,k,5033)||
                   !si_check(ss,fail,c,15,k,5033))return;
                si_f(bo,9)[k]=0.f;si_f(bo,13)[k]=0.f;
                si_f(bi,13)[k]=pp;si_f(bi,14)[k]=ss;
            } else if(!si_mix_ad(p,bo,bi,j,j+1,k,x,bx,fail,c))return;
        }
        if(mynn) {
            if(!si_mix_ad(p,bo,bi,9,10,k,x,bx,fail,c))return;
        }
        if(!si_mix_ad(p,bo,bi,7,8,k,x,bx,fail,c) ||
           !si_mix_ad(p,bo,bi,5,6,k,x,bx,fail,c))return;
        if(!si_check(float(bx),fail,c,5,k,5026))return;
        float accumulated=float(double(si_f(bi,4)[k])+bx);
        if(!si_check(accumulated,fail,c,5,k,5026))return;
        si_f(bi,4)[k]=accumulated;
    }
}
#endif

#if !defined(SI_COMPILE_ONE) || defined(SI_CORRECT_TL)
extern "C" __global__ void oracle_drivers_correct_tsk_over_seaice__tl(
    unsigned long long *p,unsigned long long *d,int nc,SeaiceFailure *fail,
    unsigned int *branches) {
    int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=nc)return;
    int ims=si_i(p,0)[c],ime=si_i(p,1)[c],jms=si_i(p,2)[c];
    int its=si_i(p,4)[c],ite=si_i(p,5)[c],jts=si_i(p,6)[c],jte=si_i(p,7)[c];
    float t=si_f(p,8)[c]; if(!si_check(t,fail,c,9,c,5064))return;
    for(int j=jts;j<=jte;++j) for(int i=its;i<=ite;++i) {
        int k=c+((j-jms)*(ime-ims+1)+(i-ims))*nc;
        float x=si_f(p,9)[k],v=si_f(p,10)[k],dv=si_f(d,10)[k];
        if(!si_check(x,fail,c,10,k,5064)||!si_check(v,fail,c,11,k,5065)||
           !si_check(dv,fail,c,11,k,5065))return;
        bool sea=true,ice=true;
        const bool left=x>=t, right=left && x<=1.f;
        si_branch(branches,0,right); si_branch(branches,1,left);
        if(left) si_branch(branches,2,x<=1.f);
        if(right) {
            sea=(v>271.4f); // ruc_max(a,b) selects b when a == b
            si_branch(branches,3,sea);
            bool a=x<0.2f, b=a && v<253.15f;
            si_branch(branches,4,b); si_branch(branches,5,a);
            if(a) si_branch(branches,6,v<253.15f);
            float vi=b?253.15f:v;
            bool d1=x<0.1f, d2=d1 && vi<263.15f;
            si_branch(branches,7,d2); si_branch(branches,8,d1);
            if(d1) si_branch(branches,9,vi<263.15f);
            if(b || d2) ice=false;
        }
        si_f(d,11)[k]=sea?dv:0.f; si_f(d,12)[k]=ice?dv:0.f;
    }
}
#endif
#if !defined(SI_COMPILE_ONE) || defined(SI_CORRECT_AD)
extern "C" __global__ void oracle_drivers_correct_tsk_over_seaice__ad(
    unsigned long long *p,unsigned long long *b,int nc,SeaiceFailure *fail) {
    int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=nc)return;
    int ims=si_i(p,0)[c],ime=si_i(p,1)[c],jms=si_i(p,2)[c];
    int its=si_i(p,4)[c],ite=si_i(p,5)[c],jts=si_i(p,6)[c],jte=si_i(p,7)[c];
    float t=si_f(p,8)[c]; if(!si_check(t,fail,c,9,c,5064))return;
    for(int j=jte;j>=jts;--j) for(int i=ite;i>=its;--i) {
        int k=c+((j-jms)*(ime-ims+1)+(i-ims))*nc;
        float x=si_f(p,9)[k],v=si_f(p,10)[k],bs=si_f(b,11)[k],bi=si_f(b,12)[k];
        if(!si_check(x,fail,c,10,k,5064)||!si_check(v,fail,c,11,k,5065)||
           !si_check(bs,fail,c,12,k,5065)||!si_check(bi,fail,c,13,k,5066))return;
        bool sea=true,ice=true;
        if(x>=t && x<=1.f) {
            sea=(v>271.4f);
            if(x<0.2f && v<253.15f)ice=false;
            if(x<0.1f && v<263.15f)ice=false;
        }
        double r=double(sea?bs:0.f)+double(ice?bi:0.f);
        float accumulated=float(double(si_f(b,10)[k])+r);
        if(!si_check(accumulated,fail,c,11,k,5065))return;
        si_f(b,11)[k]=0.f;si_f(b,12)[k]=0.f;si_f(b,10)[k]=accumulated;
    }
}
#endif

// Hand entries for the frozen ad-gen copy plans. Launch and argument order
// match each opspec's generated contract, with int32 source-site failure slots
// as required by COMMON-AD section 35.4. The binder validates n, one-block
// reverse geometry, disjoint spans and immutable masks before every launch.
// Reverse workspace has n floats. Compile with --ftz=false so the rounded
// addition preserves subnormals, as the frozen generator's mpas_f32 rule does.
// The enclosing scheduler runs pattern/field packing only for spp_lsm == 1;
// masks are recorded per element. The TSO pack input is the clamped value.
#define SI_NO_MASK
#define SI_WITH_MASK const int *executed_mask,
#define SI_ALL_ELEMENTS true
#define SI_MASK_ELEMENT (executed_mask[i] != 0)
#define SI_PACK_ENTRIES_FOR(name, mask_param, selected, partial) \
extern "C" __global__ void name##__aug(int n, mask_param const float *src, \
                                        float *dst, SeaiceFailure *fail) { \
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; \
 if((partial) && !si_finite(dst[i])) atomicOr(fail,1); \
 if(!(selected))return; \
 float x=src[i]; if(!si_finite(x))atomicOr(fail,1); dst[i]=x; \
} \
extern "C" __global__ void name##__tl(int n, mask_param const float *src, \
 float *dst,const float *dsrc,float *ddst,SeaiceFailure *fail) { \
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; \
 if((partial) && (!si_finite(dst[i]) || !si_finite(ddst[i]))) atomicOr(fail,1); \
 if(!(selected))return; \
 float x=src[i],dx=dsrc[i]; \
 if(!si_finite(x)||!si_finite(dx))atomicOr(fail,1); ddst[i]=dx; \
} \
extern "C" __global__ void name##__ad(int n, mask_param const float *src, \
 float *dst,float *bsrc,float *bdst,float *workspace,SeaiceFailure *fail) { \
 if(gridDim.x!=1 || gridDim.y!=1 || gridDim.z!=1 || \
    blockDim.y!=1 || blockDim.z!=1) { \
   if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && \
      threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0) atomicOr(fail,1); \
   return; \
 } \
 for(long long i=threadIdx.x;i<n;i+=blockDim.x) { \
   workspace[i]=bdst[i]; \
   if(!si_finite(bdst[i])||!si_finite(dst[i]))atomicOr(fail,1); \
 } \
 __syncthreads(); \
 for(long long i=threadIdx.x;i<n;i+=blockDim.x) if(selected)bdst[i]=0.f; \
 __syncthreads(); \
 for(long long i=threadIdx.x;i<n;i+=blockDim.x) { \
   if(!si_finite(src[i])||!si_finite(bsrc[i]))atomicOr(fail,1); \
   if(selected)bsrc[i]=__fadd_rn(bsrc[i],workspace[i]); \
   if(!si_finite(bsrc[i]))atomicOr(fail,1); \
 } \
}
#if !defined(SI_COMPILE_ONE) || defined(SI_PACK_ENTRIES)
SI_PACK_ENTRIES_FOR(seaice_tsk_to_ice, SI_NO_MASK, SI_ALL_ELEMENTS, false)
SI_PACK_ENTRIES_FOR(seaice_tsk_to_sea, SI_NO_MASK, SI_ALL_ELEMENTS, false)
SI_PACK_ENTRIES_FOR(seaice_pattern_pack, SI_NO_MASK, SI_ALL_ELEMENTS, false)
SI_PACK_ENTRIES_FOR(seaice_field_pack, SI_NO_MASK, SI_ALL_ELEMENTS, false)
SI_PACK_ENTRIES_FOR(seaice_field_unpack, SI_WITH_MASK, SI_MASK_ELEMENT, true)
SI_PACK_ENTRIES_FOR(seaice_tso_pack, SI_WITH_MASK, SI_MASK_ELEMENT, true)
SI_PACK_ENTRIES_FOR(seaice_tso_unpack, SI_WITH_MASK, SI_MASK_ELEMENT, true)
#endif
#undef SI_PACK_ENTRIES_FOR
#undef SI_MASK_ELEMENT
#undef SI_ALL_ELEMENTS
#undef SI_WITH_MASK
#undef SI_NO_MASK

#define SI_SEAICE_COMPOSE_HAVE_BASE
// Column-local composition adapters for the RUC sea-ice hand graph.
//
// These adapters consume a graph recorded with the actual forward unary
// values and Vilka visits. They do not launch a kernel or claim a proof.
// Each caller supplies distinct scalar-output and tso-output cotangent arrays,
// graph work storage disjoint from the field buffers, and field lengths. Each
// graph tape records the forward's actual scalar and tso write masks. The
// same physical input may be bound to more than one leaf; the reverse adds
// every such contribution in recorded order. This is important for repeated
// table reads and for aliased forward inputs.
#ifndef SI_SEAICE_COMPOSE_HAVE_BASE
#ifndef SI_COMPILE_ONE
#define SI_COMPILE_ONE
#define SI_SEAICE_COMPOSE_UNDEF_ONE
#endif
#include "seaice.ad.cu"
#ifdef SI_SEAICE_COMPOSE_UNDEF_ONE
#undef SI_COMPILE_ONE
#undef SI_SEAICE_COMPOSE_UNDEF_ONE
#endif
#endif

__device__ __forceinline__ bool si_comp_finite(double x) {
    return x==x && x<=1.7976931348623157e308 && x>=-1.7976931348623157e308;
}

__device__ __forceinline__ bool si_comp_check(
        const SiColumnGraph &g,const int *scalar_out,int nscalar,
        const int *tso_out,int nzs,const int *width) {
    if(!g.valid||!g.node||!g.leaf||!scalar_out||!tso_out||!width||
       g.count<0||g.count>g.capacity||g.nleaf<0||g.nleaf>g.leaf_capacity||
       nscalar<0||nzs<0)return false;
    for(int i=0;i<nscalar;++i)
        if(scalar_out[i]<0||scalar_out[i]>=g.count)return false;
    for(int i=0;i<nzs;++i)
        if(tso_out[i]<0||tso_out[i]>=g.count)return false;
    for(int i=0;i<g.count;++i) {
        SiColumnNode x=g.node[i];
        if(!si_finite(x.value)||!si_comp_finite(x.da)||!si_comp_finite(x.db)||
           (x.a>=0&&(x.a>=i||x.b<0||x.b>=i))||(x.a<0&&x.b>=0))return false;
    }
    for(int i=0;i<g.nleaf;++i) {
        SiColumnLeaf x=g.leaf[i];
        if(x.node<0||x.node>=g.count||x.field<0||x.field>=SI_SNOW_NFIELD||
           x.element<0||x.element>=width[x.field])return false;
    }
    return true;
}

// The output arrays are independent scalar slots even when two slots refer
// to the same graph node. Derivative work is accumulated in double and each
// public launch is responsible for rounding to its declared buffer dtype.
__device__ __forceinline__ bool si_comp_tl(
        const SiColumnGraph &g,const int *scalar_out,int nscalar,
        const int *tso_out,int nzs,const int *width,
        const double *const *direction,double *work,
        double *dscalar,double *dtso) {
    if(!direction||!work||!dscalar||!dtso||
       !si_comp_check(g,scalar_out,nscalar,tso_out,nzs,width))return false;
    for(int i=0;i<g.nleaf;++i) {
        SiColumnLeaf x=g.leaf[i];
        if(!direction[x.field]||!si_comp_finite(direction[x.field][x.element]))return false;
    }
    if(!si_col_tl(g,direction,work))return false;
    for(int i=0;i<g.count;++i)if(!si_comp_finite(work[i]))return false;
    for(int i=0;i<nscalar;++i)dscalar[i]=work[scalar_out[i]];
    for(int i=0;i<nzs;++i)dtso[i]=work[tso_out[i]];
    return true;
}

// Seed all outputs before clearing them: two forward outputs can represent
// the same intermediate. Forward-written slots are consumed; slots outside the
// executed write mask retain their exact incoming seed bits. Input accumulators
// are separate from output seeds, including for pass-through identities.
__device__ __forceinline__ bool si_comp_ad(
        const SiColumnGraph &g,const int *scalar_out,int nscalar,
        const int *tso_out,int nzs,const int *width,
        const unsigned char *scalar_written,unsigned char tso_written,
        double *bscalar,double *btso,double *work,double *const *result) {
    if(!result||!work||!bscalar||!btso||!scalar_written||tso_written>1||
       !si_comp_check(g,scalar_out,nscalar,tso_out,nzs,width))return false;
    for(int i=0;i<g.nleaf;++i) {
        SiColumnLeaf x=g.leaf[i];
        if(!result[x.field]||!si_comp_finite(result[x.field][x.element]))return false;
    }
    for(int i=0;i<nscalar;++i)
        if(scalar_written[i]>1||!si_comp_finite(bscalar[i]))return false;
    for(int i=0;i<nzs;++i)if(!si_comp_finite(btso[i]))return false;
    for(int i=0;i<g.count;++i)work[i]=0.;
    for(int i=0;i<nscalar;++i)work[scalar_out[i]]+=bscalar[i];
    for(int i=0;i<nzs;++i)work[tso_out[i]]+=btso[i];
    for(int i=g.count-1;i>=0;--i)if(g.node[i].a>=0) {
        SiColumnNode x=g.node[i];
        work[x.a]+=x.da*work[i];work[x.b]+=x.db*work[i];
    }
    for(int i=0;i<g.count;++i)if(!si_comp_finite(work[i]))return false;
    for(int i=0;i<nscalar;++i)if(scalar_written[i])bscalar[i]=0.;
    if(tso_written)for(int i=0;i<nzs;++i)btso[i]=0.;
    for(int i=0;i<g.nleaf;++i) {
        SiColumnLeaf x=g.leaf[i];
        double v=result[x.field][x.element]+work[x.node];
        if(!si_comp_finite(v))return false;
        result[x.field][x.element]=v;
    }
    return true;
}

#define SI_SEAICE_COLUMN_PAIR(name, Tape, count) \
__device__ __forceinline__ bool name##__tl(const Tape &t,int nzs, \
        const int *width,const double *const *d,double *work, \
        double *out,double *tso) { \
    return si_comp_tl(t.graph,t.out,count,t.tso,nzs,width,d,work,out,tso); \
} \
__device__ __forceinline__ bool name##__ad(const Tape &t,int nzs, \
        const int *width,double *out,double *tso,double *work, \
        double *const *result) { \
    return si_comp_ad(t.graph,t.out,count,t.tso,nzs,width,t.written, \
                      t.tso_written,out,tso,work,result); \
}

SI_SEAICE_COLUMN_PAIR(si_ice_sice, SiSiceColumnTape, SI_SICE_NY)
SI_SEAICE_COLUMN_PAIR(si_ice_snowseaice, SiSnowColumnTape, SI_SNOW_NOUTPUT)
SI_SEAICE_COLUMN_PAIR(si_ice_icetmp, SiIceTmpTape, SI_ICETMP_NSCALAR)
SI_SEAICE_COLUMN_PAIR(si_ice_ruc_ice, SiRucIceTape, SI_RUC_NSCALAR)
#undef SI_SEAICE_COLUMN_PAIR

#define SI_BUILD_ONE
#define SI_BUILD_RUC
// SPDX-License-Identifier: BSD-3-Clause
// Launch adapters for the whole RUC sea-ice helper graphs. Compile this file
// after the pinned forward's own math prelude and with
// SI_SEAICE_ORIGINAL_MATH_AVAILABLE defined. The prelude remains forward
// source: this file differentiates the values returned by that prelude.
//
// Each Input and Tape is a per-call POD containing device pointers. The
// caller allocates the graph nodes, graph leaves, layer scratch and visit
// storage required by the corresponding record function in seaice.ad.cu,
// then initializes the PODs on the device. The graph, output-node indices,
// write masks and layer indices are saved by __aug until both sweeps finish.
// No layer or iteration bound is hard-coded in these launches.
//
// A view names one derivative field. Scalars are the first N views, one per
// scalar slot; remaining views name SiSnowColumnField 1..7. The last field
// is the shared table. The producer writes a per-column double table partial
// in field_work; seaice_table_reduce__ad must then run in increasing column
// order. Input accumulator views and output-seed views must be disjoint.
#ifndef SI_SEAICE_ORIGINAL_MATH_AVAILABLE
#error "prepend the pinned forward math prelude and define SI_SEAICE_ORIGINAL_MATH_AVAILABLE"
#endif


struct SiLaunchView {
    unsigned long long address;
    long long column_stride_bytes, element_stride_bytes;
    int width;
    int shadow_f64;
};
__device__ __forceinline__ char *si_launch_address(const SiLaunchView &v,int c,int i) {
    return reinterpret_cast<char*>(v.address)+c*v.column_stride_bytes+i*v.element_stride_bytes;
}
__device__ __forceinline__ double si_launch_read(const SiLaunchView &v,int c,int i) {
    char *p=si_launch_address(v,c,i);
    return v.shadow_f64?*reinterpret_cast<double*>(p):double(*reinterpret_cast<float*>(p));
}
__device__ __forceinline__ void si_launch_write(const SiLaunchView &v,int c,int i,double x) {
    char *p=si_launch_address(v,c,i);
    if(v.shadow_f64)*reinterpret_cast<double*>(p)=x;
    else *reinterpret_cast<float*>(p)=float(x);
}
__device__ __forceinline__ const SiLaunchView &si_launch_field(
        const SiLaunchView *v,int ns,int field,int element) {
    return v[field==0?element:ns+field-1];
}
__device__ __forceinline__ int si_launch_element(int field,int element) {
    return field==0?0:element;
}
__device__ __forceinline__ bool si_launch_same(const SiLaunchView &a,int ac,int ai,
                                               const SiLaunchView &b,int bc,int bi) {
    return si_launch_address(a,ac,ai)==si_launch_address(b,bc,bi);
}
__device__ bool si_launch_sweep(const SiColumnGraph &g,const int *outputs,int nsout,
        const int *tso_out,int nz,const unsigned char *written,unsigned char tso_written,
        const int *width,const SiLaunchView *in,const SiLaunchView *out,int c,
        double *graph_work,int graph_stride,double *field_work,int field_stride,
        double *output_work,int output_stride,bool reverse) {
    if(!in||!out||!graph_work||!field_work||!output_work||
       graph_stride<g.count||output_stride<nsout+nz)return false;
    const int ns=width[0];int total=0;
    double *fv[SI_SNOW_NFIELD];
    for(int f=0;f<SI_SNOW_NFIELD;++f) {
        if(width[f]<0)return false;
        fv[f]=field_work+total;total+=width[f];
    }
    if(field_stride<total)return false;
    for(int f=0;f<SI_SNOW_NFIELD;++f)for(int k=0;k<width[f];++k) {
        const SiLaunchView &v=si_launch_field(in,ns,f,k);
        int e=si_launch_element(f,k),col=f==SI_SNOW_TABLE?0:c;
        if(!v.address||e<0||e>=v.width||(v.shadow_f64!=0&&v.shadow_f64!=1))return false;
        bool duplicate=false;
        if(reverse&&f!=SI_SNOW_TABLE) {
            // Alias groups, including scalar/array aliases, contribute once
            // to the existing accumulator and are summed before rounding.
            for(int pf=0;pf<=f&&!duplicate;++pf)for(int pk=0;pk<width[pf];++pk) {
                if(pf==f&&pk>=k)break;
                if(pf==SI_SNOW_TABLE)continue;
                const SiLaunchView &prior=si_launch_field(in,ns,pf,pk);
                int pe=si_launch_element(pf,pk);
                if(si_launch_same(v,col,e,prior,c,pe)) {
                    if(v.shadow_f64!=prior.shadow_f64)return false;
                    duplicate=true;break;
                }
            }
        }
        fv[f][k]=(reverse&&(f==SI_SNOW_TABLE||duplicate))?0.:si_launch_read(v,col,e);
    }
    double *scalar=output_work,*tso=output_work+nsout;
    for(int k=0;k<nsout+nz;++k) {
        int slot=k<nsout?k:nsout,elem=k<nsout?0:k-nsout;
        if(!out[slot].address||elem>=out[slot].width||
           (out[slot].shadow_f64!=0&&out[slot].shadow_f64!=1))return false;
        if(reverse)output_work[k]=si_launch_read(out[slot],c,elem);
    }
    if(reverse) {
        if(!si_comp_ad(g,outputs,nsout,tso_out,nz,width,written,tso_written,
                       scalar,tso,graph_work,fv))return false;
        for(int k=0;k<nsout+nz;++k) {
            int slot=k<nsout?k:nsout,elem=k<nsout?0:k-nsout;
            si_launch_write(out[slot],c,elem,output_work[k]);
        }
        for(int f=0;f<SI_SNOW_TABLE;++f)for(int k=0;k<width[f];++k) {
            const SiLaunchView &v=si_launch_field(in,ns,f,k);
            int e=si_launch_element(f,k);bool duplicate=false;
            for(int pf=0;pf<=f&&!duplicate;++pf)for(int pk=0;pk<width[pf];++pk) {
                if(pf==f&&pk>=k)break;
                const SiLaunchView &prior=si_launch_field(in,ns,pf,pk);
                if(si_launch_same(v,c,e,prior,c,si_launch_element(pf,pk))) {duplicate=true;break;}
            }
            if(duplicate)continue;
            double sum=fv[f][k];
            for(int nf=f;nf<SI_SNOW_TABLE;++nf)for(int nk=0;nk<width[nf];++nk) {
                if(nf==f&&nk<=k)continue;
                const SiLaunchView &next=si_launch_field(in,ns,nf,nk);
                if(si_launch_same(v,c,e,next,c,si_launch_element(nf,nk)))sum+=fv[nf][nk];
            }
            si_launch_write(v,c,e,sum);
        }
    } else {
        const double *direction[SI_SNOW_NFIELD];
        for(int f=0;f<SI_SNOW_NFIELD;++f)direction[f]=fv[f];
        if(!si_comp_tl(g,outputs,nsout,tso_out,nz,width,direction,graph_work,scalar,tso))return false;
        for(int k=0;k<nsout+nz;++k) {
            int slot=k<nsout?k:nsout,elem=k<nsout?0:k-nsout;
            si_launch_write(out[slot],c,elem,output_work[k]);
        }
    }
    return true;
}

#define SI_WHOLE_SWEEPS(name,Tape,NSOUT,W0,W1,W2,W3,W4,W5,W6) \
extern "C" __global__ void name##__tl(const Tape *t,int n,int nz,int nd, \
 const SiLaunchView *in,const SiLaunchView *out,double *graph,int gs, \
 double *fields,int fs,double *outputs,int os,int *failure) { \
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return; \
 int width[SI_SNOW_NFIELD]={W0,W1,W2,W3,W4,W5,W6,5001}; \
 if(!si_launch_sweep(t[c].graph,t[c].out,NSOUT,t[c].tso,nz,t[c].written, \
 t[c].tso_written,width,in,out,c,graph+(long long)c*gs,gs,fields+(long long)c*fs,fs, \
 outputs+(long long)c*os,os,false))atomicOr(failure,1); \
} \
extern "C" __global__ void name##__ad(const Tape *t,int n,int nz,int nd, \
 const SiLaunchView *in,const SiLaunchView *out,double *graph,int gs, \
 double *fields,int fs,double *outputs,int os,int *failure) { \
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return; \
 int width[SI_SNOW_NFIELD]={W0,W1,W2,W3,W4,W5,W6,5001}; \
 if(!si_launch_sweep(t[c].graph,t[c].out,NSOUT,t[c].tso,nz,t[c].written, \
 t[c].tso_written,width,in,out,c,graph+(long long)c*gs,gs,fields+(long long)c*fs,fs, \
 outputs+(long long)c*os,os,true))atomicOr(failure,1); \
}
#if !defined(SI_BUILD_ONE) || defined(SI_BUILD_SICE)
SI_WHOLE_SWEEPS(seaice_ice_sice,SiSiceColumnTape,SI_SICE_NY,SI_SICE_NX,nz,nz,1,0,0,nd)
#endif
#if !defined(SI_BUILD_ONE) || defined(SI_BUILD_SNOW)
SI_WHOLE_SWEEPS(seaice_ice_snowseaice,SiSnowColumnTape,SI_SNOW_NOUTPUT,SI_SNOW_NSCALAR,nz,nz,nz,nz,nz,nd)
#endif
#if !defined(SI_BUILD_ONE) || defined(SI_BUILD_ICETMP)
SI_WHOLE_SWEEPS(seaice_ice_icetmp,SiIceTmpTape,SI_ICETMP_NSCALAR,SI_ICETMP_NSCALAR,nz,0,0,nz,nz,nd)
#endif
#if !defined(SI_BUILD_ONE) || defined(SI_BUILD_RUC)
SI_WHOLE_SWEEPS(seaice_ice_ruc_ice,SiRucIceTape,SI_RUC_NSCALAR,SI_RUC_NSCALAR,nz,0,0,nz,0,0)
#endif
#undef SI_WHOLE_SWEEPS

#define SI_WHOLE_AUG(name,Input,Tape,NSOUT,record,extra) \
extern "C" __global__ void name##__aug(const Input *inputs,Tape *tapes, \
 int n,int nz,float *scalar_out,float *tso_out,int *aux, \
 unsigned int *branches,int *failure) { \
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return; \
 if(inputs[c].nzs!=nz||!record(inputs[c],tapes[c],branches)) {atomicOr(failure,1);return;} \
 for(int k=0;k<NSOUT;++k)scalar_out[(long long)c*NSOUT+k]=tapes[c].graph.node[tapes[c].out[k]].value; \
 for(int k=0;k<nz;++k)tso_out[(long long)c*nz+k]=tapes[c].graph.node[tapes[c].tso[k]].value; \
 extra \
}
#if !defined(SI_BUILD_ONE) || defined(SI_BUILD_SICE)
SI_WHOLE_AUG(seaice_ice_sice,SiSiceColumnInput,SiSiceColumnTape,SI_SICE_NY,si_sice_column_record_aug,)
#endif
#if !defined(SI_BUILD_ONE) || defined(SI_BUILD_SNOW)
SI_WHOLE_AUG(seaice_ice_snowseaice,SiSnowColumnInput,SiSnowColumnTape,SI_SNOW_NOUTPUT,si_snow_column_record_aug,)
#endif
#if !defined(SI_BUILD_ONE) || defined(SI_BUILD_ICETMP)
SI_WHOLE_AUG(seaice_ice_icetmp,SiIceTmpInput,SiIceTmpTape,SI_ICETMP_NSCALAR,si_icetmp_column_record_aug,if(aux)aux[c]=tapes[c].iland;)
#endif
#if !defined(SI_BUILD_ONE) || defined(SI_BUILD_RUC)
// pk120: a dead non-finite RI_LOCAL_* input entered the graph as 0 (si_ruc_ice_column_record); where the slot passes
// through unchanged (its output is still leaf k: scalar leaves are nodes 0..NSCALAR-1 in order), the augmented primal
// writes the incoming word back, so it equals the forward's post state word for word (pk120e4 step 2: 2 words).
SI_WHOLE_AUG(seaice_ice_ruc_ice,SiRucIceInput,SiRucIceTape,SI_RUC_NSCALAR,si_ruc_ice_column_record,if(aux)aux[c]=tapes[c].active;for(int k=RI_LOCAL_SMELT;k<SI_RUC_NSCALAR;++k)if(!si_finite(inputs[c].scalar[k])&&tapes[c].out[k]==k)scalar_out[(long long)c*SI_RUC_NSCALAR+k]=inputs[c].scalar[k];)
#endif
#undef SI_WHOLE_AUG

#define SI_SEAICE_TABLE_REDUCE_HAVE_BASE
// SPDX-License-Identifier: BSD-3-Clause
// Deterministic adjoint reduction for an ice lookup table shared by columns.
// The producer writes one double partial per column and table entry, zeroing
// unused entries. The output is accumulated once, in increasing column order.
// Input/output storage must not overlap; the caller supplies a failure slot.
#ifndef SI_SEAICE_TABLE_REDUCE_HAVE_BASE
#ifndef SI_COMPILE_ONE
#define SI_COMPILE_ONE
#define SI_SEAICE_TABLE_REDUCE_UNDEF_ONE
#endif
#include "seaice.ad.cu"
#ifdef SI_SEAICE_TABLE_REDUCE_UNDEF_ONE
#undef SI_COMPILE_ONE
#undef SI_SEAICE_TABLE_REDUCE_UNDEF_ONE
#endif
#endif

__device__ __forceinline__ bool si_table_finite_double(double x) {
    return x==x && x<=1.7976931348623157e308 && x>=-1.7976931348623157e308;
}

extern "C" __global__ void seaice_table_reduce__ad(
        const double *partials,int columns,int width,int row_stride,int offset,
        float *table_bar,SeaiceFailure *failure) {
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(!failure)return;
    if(!partials||!table_bar||columns<0||width<=0||row_stride<width||
       offset<0||offset>row_stride-width) {
        if(k==0)si_fail(failure,0,1,0,4711);return;
    }
    if(k>=width)return;
    double sum=double(table_bar[k]);
    if(!si_table_finite_double(sum)){si_fail(failure,k,1,k,4711);return;}
    for(int c=0;c<columns;++c) {
        double v=partials[(unsigned long long)c*row_stride+offset+k];
        if(!si_table_finite_double(v)){si_fail(failure,c,1,k,4711);return;}
        sum+=v;
        if(!si_table_finite_double(sum)){si_fail(failure,c,1,k,4711);return;}
    }
    float result=float(sum);
    if(!si_finite(result)){si_fail(failure,k,1,k,4711);return;}
    table_bar[k]=result;
}

struct IceTmpRow {
 float x[67],tso[9],zmain[9],zhalf[9],dz[14];SiIceUnaryRecord unary[4];
 float expected[67],expected_tso[9],bare_pow,snow_pow;
 int bv[32],sv[32],bn,sn,myj,iland,isice,out_iland,col,call;
};
struct RucRow {float x[72],tso[9],zs[9],expected[72],expected_tso[9];IceTmpRow child;int col,active;};
static_assert(sizeof(RucRow)==1756,"captured row");
extern "C" __global__ void ruc_launch_construct(const RucRow *rows,const float *table,int n,
 SiColumnNode *pn,SiColumnLeaf *pl,SiColumnNode *inn,SiColumnLeaf *il,SiColumnNode *cn,SiColumnLeaf *cl,
 int *map,int *cmap,int *layer,float *flayer,int *rec,int *counts,SiRucIceInput *input,SiRucIceTape *tape) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;const RucRow &r=rows[c];
 SiRucIceInput x={r.x,r.tso,r.zs,table,nullptr,nullptr,nullptr,9,5001,0,0,15,true,true,true,false,0.f,0.f};
 x.bare_record=rec+c*64;x.snow_record=x.bare_record+32;x.bare_record_count=counts+c*2;
 x.snow_record_count=x.bare_record_count+1;x.bare_record_capacity=32;x.snow_record_capacity=32;
 x.bare_record_count[0]=-1;x.snow_record_count[0]=-1;
 SiRucIceTape t={};t.graph.node=pn+c*4096;t.graph.leaf=pl+c*2048;t.graph.capacity=4096;t.graph.leaf_capacity=2048;
 int *q=layer+c*145;t.tso=q;q+=9;t.zmain=q;q+=9;t.zhalf=q;q+=9;t.dz=q;q+=14;t.map=map+c*4096;
 float *f=flayer+c*59;t.tso_value=f;f+=9;t.zmain_value=f;f+=9;t.zhalf_value=f;f+=9;t.dz_value=f;f+=14;
 SiIceTmpTape &i=t.ice;i.graph.node=inn+c*4096;i.graph.leaf=il+c*1024;i.graph.capacity=4096;i.graph.leaf_capacity=1024;
 i.child.node=cn+c*2048;i.child.leaf=cl+c*512;i.child.capacity=2048;i.child.leaf_capacity=512;
 i.tso=q;q+=9;i.tice=q;q+=9;i.cap=q;q+=9;i.diff=q;q+=9;i.zmain=q;q+=9;i.zhalf=q;q+=9;i.dz=q;q+=14;
 i.bare_tso=q;q+=9;i.child_c=q;q+=9;i.child_r=q;q+=9;i.child_tso=q;i.child_map=cmap+c*2048;
 i.cap_value=f;f+=9;i.diff_value=f;input[c]=x;tape[c]=t;
}
extern "C" __global__ void ruc_launch_validate(int n,const RucRow *r,const int *rec,const int *cnt,const int *active,int *failure) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;const IceTmpRow &i=r[c].child;
 if(active[c]!=r[c].active){atomicOr(failure,2);return;}if(!active[c])return;
 if((i.bn?cnt[c*2]!=i.bn:cnt[c*2]!=-1)||(i.sn?cnt[c*2+1]!=i.sn:cnt[c*2+1]!=-1)){atomicOr(failure,4);return;}
 for(int k=0;k<i.bn;++k)if(rec[c*64+k]!=i.bv[k])atomicOr(failure,8);
 for(int k=0;k<i.sn;++k)if(rec[c*64+32+k]!=i.sv[k])atomicOr(failure,8);
}
extern "C" __global__ void ruc_launch_views(const unsigned long long *src,SiLaunchView *dst,int output) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(output){if(i>=73)return;int w=i==72?9:1;dst[i]={src[i],4LL*w,4,w,0};return;}
 if(i>=79)return;int j=i<72?(i<5?i:i-1):(i==72?71:(i==75?72:(i==78?73:-1)));
 int w=i<72?1:(i==72||i==75?9:(i==78?5001:0));dst[i]={j<0?0:src[j],i==78?0:4LL*w,4,w,0};
}
__device__ unsigned int ruc_launch_bits(const SiLaunchView &v,int c,int e) {return *reinterpret_cast<unsigned int*>(si_launch_address(v,c,e));}
extern "C" __global__ void ruc_launch_seed_check(const SiLaunchView *out,const SiRucIceTape *t,int n,unsigned int *old,int phase,int *failure) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
 for(int k=0;k<81;++k){int slot=k<72?k:72,e=k<72?0:k-72;unsigned int b=ruc_launch_bits(out[slot],c,e);
  if(!phase)old[c*81+k]=b;else if(b!=((k<72?t[c].written[k]:t[c].tso_written)?0U:old[c*81+k]))atomicOr(failure,16);
 }
}
