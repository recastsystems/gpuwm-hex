// SPDX-License-Identifier: Apache-2.0
// Private autodiff program (ad/held-oracle_drivers_correct_tsk_over_seaice). Whole-launch tangent-linear and adjoint
// entries of hex 16251589 oracle_drivers_correct_tsk_over_seaice (tree/src/hexcore/mpasphys/rrfs/ruc_lsm_8c9f1a8.py:5436,
// captured .cu e6fd85f0 device body drivers_correct_tsk_over_seaice L6354-L6372, launch L6735-L6752; RRFS-MPAS 1d6e370e
// mpas_atmphys_lsm_shared.F:22-57), written by hand from the forward CUDA source.
//
// Forward, per thread c < nc, j = jts..jte, i = its..ite (element (i,j) of every array is word
// c + (i-ims)*nc + (j-jms)*nc*(ime-ims+1)):
//   tsk_sea = tsk; tsk_ice = tsk                                                   (L42, L43)
//   if (xice >= xice_thresh && xice <= 1) {                                        (L45)
//     tsk_sea = ruc_max(tsk_sea, 271.4)        ruc_max(a,b) = a > b ? a : b        (L47)
//     if (xice < 0.2 && tsk_ice < 253.15) tsk_ice = 253.15                         (L51)
//     if (xice < 0.1 && tsk_ice < 263.15) tsk_ice = 263.15                         (L52, reads the L51 result)
//   }
// Linearisation (frozen-coefficient treatment of the three clamps: each switch is evaluated on the trajectory
// primal and frozen, so the derivative is the exact Jacobian of the branch the forward took):
//   d tsk_sea = (ice && !(tsk > 271.4)) ? 0 : d tsk       (tie tsk == 271.4: ruc_max returns the constant b -> 0)
//   d tsk_ice = (ice && (L51 fired || L52 fired)) ? 0 : d tsk
//   xice, xice_thresh enter only the switch predicates: the outputs are piecewise constant in them, so their
//   derivative is exactly zero wherever it exists (inactive). The 8 int32 extents (ims..jte) are inactive.
// Precision: float32 throughout, built with --fmad=false --ftz=false (as the live schemes). There is no arithmetic
// in the TL: every derivative word is a copy of d tsk or 0, so TL is exact; the AD adds at most two terms per word
// with __fadd_rn.
// tsk_sea and tsk_ice are fully overwritten on the loop range: TL writes d_out = J d_in; AD accumulates J^T b_out
// into b_tsk and clears b_out. The forward never writes tsk, xice or xice_thresh, so the live (post-launch) primal
// read through the table equals the pre-launch primal; the entries refuse (slot 1) if an output aliases an input or
// the two outputs alias each other (then the forward result depends on write order).
//
// Pointer table (layout::ICE_TSK, ruc_layout.rs:323): 0 ims, 1 ime, 2 jms, 3 jme, 4 its, 5 ite, 6 jts, 7 jte (int32),
// 8 xice_thresh, 9 xice, 10 tsk (inputs), 11 tsk_sea, 12 tsk_ice (outputs). The derivative table is the dispatcher
// mirror of the same table (indirect.mirror): word k points at the shadow of the forward pointer k.
//
// Failure slots: 0 nonfinite tangent/cotangent or primal tsk; 1 an output pointer equals an input pointer or the
// other output.
// AD_GAIN_PROBE (diagnostic image only, never the bound image): the TL entry also reduces ||x_tsk||, ||TL x|| per
// output, switch counts and the primal margins to every switch, and prints one ADGAIN line per TL launch.

#ifndef AD_GAIN_PROBE
#define AD_GAIN_PROBE 0
#endif
#if AD_GAIN_PROBE
#include <cstdio>
#endif

namespace cts {
enum { IMS = 0, IME, JMS, JME, ITS, ITE, JTS, JTE, THR = 8, XICE = 9, TSK = 10, SEA = 11, ICE = 12, NPTR = 13 };
__device__ __forceinline__ bool fin(float v) { return isfinite(v); }
__device__ __forceinline__ bool aliased(const unsigned long long *p) {
  return p[SEA] == p[TSK] || p[ICE] == p[TSK] || p[SEA] == p[ICE] || p[SEA] == p[XICE] || p[ICE] == p[XICE]
      || p[SEA] == p[THR] || p[ICE] == p[THR];
}
struct Sw { bool ice, sea_clamp, l51, l52; };
// The three switches, on the trajectory primal, exactly as the forward evaluates them (float32 compares).
__device__ __forceinline__ Sw switches(float xice, float thr, float tsk) {
  Sw s{false, false, false, false};
  s.ice = (xice >= thr) && (xice <= 1.f);                                   // L45
  if (s.ice) {
    s.sea_clamp = !(tsk > 271.4f);                                          // L47: ruc_max(tsk, 271.4) took the constant
    float ti = tsk;
    if ((xice < 0.2f) && (ti < 253.15f)) { ti = 253.15f; s.l51 = true; }    // L51
    if ((xice < 0.1f) && (ti < 263.15f)) { ti = 263.15f; s.l52 = true; }    // L52
  }
  return s;
}
struct Geo { int ims, ime, jms, jme, its, ite, jts, jte; float thr; };
__device__ __forceinline__ Geo geo(const unsigned long long *p, int c) {
  Geo g;
  g.ims = ((const int *)p[IMS])[c]; g.ime = ((const int *)p[IME])[c]; g.jms = ((const int *)p[JMS])[c]; g.jme = ((const int *)p[JME])[c];
  g.its = ((const int *)p[ITS])[c]; g.ite = ((const int *)p[ITE])[c]; g.jts = ((const int *)p[JTS])[c]; g.jte = ((const int *)p[JTE])[c];
  g.thr = ((const float *)p[THR])[c];
  return g;
}
__device__ __forceinline__ long long word(const Geo &g, int c, int nc, int i, int j) {
  return c + (long long)(i - g.ims) * nc + (long long)(j - g.jms) * nc * (long long)(g.ime - g.ims + 1);
}
#if AD_GAIN_PROBE
// x2: sum of x_tsk^2 over the loop words; t2[0] sum of d tsk_sea^2, t2[1] sum of d tsk_ice^2.
// margins (trajectory primal): ice columns |tsk - 271.4|, |tsk - 253.15| where xice < 0.2, |tsk - 263.15| where
// xice < 0.1; all columns |xice - thr|, |xice - 1|, |xice - 0.2|, |xice - 0.1|.
__device__ double g_x2, g_t2[2];
__device__ float g_m[7] = {3.4e38f, 3.4e38f, 3.4e38f, 3.4e38f, 3.4e38f, 3.4e38f, 3.4e38f};
__device__ unsigned int g_blocks, g_words, g_ice, g_sea, g_l51, g_l52, g_tie;
#endif
}

extern "C" __global__ void oracle_drivers_correct_tsk_over_seaice__tl(const unsigned long long *ptrs, const unsigned long long *dptrs,
                                                                     int nc, int *failures) {
  using namespace cts;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
#if AD_GAIN_PROBE
  double x2 = 0, t2s = 0, t2i = 0;
  float m[7] = {3.4e38f, 3.4e38f, 3.4e38f, 3.4e38f, 3.4e38f, 3.4e38f, 3.4e38f};
  unsigned int nw = 0, ni = 0, ns = 0, n51 = 0, n52 = 0, ntie = 0;
#endif
  if (c < nc) {
    if (c == 0 && aliased(ptrs)) atomicOr(&failures[1], 1);
    const Geo g = geo(ptrs, c);
    const float *xice = (const float *)ptrs[XICE], *tsk = (const float *)ptrs[TSK];
    const float *t_tsk = (const float *)dptrs[TSK];
    float *t_sea = (float *)dptrs[SEA], *t_ice = (float *)dptrs[ICE];
    for (int j = g.jts; j <= g.jte; ++j) {
      for (int i = g.its; i <= g.ite; ++i) {
        const long long w = word(g, c, nc, i, j);
        const float x = t_tsk[w], ts = tsk[w], xi = xice[w];   // read the incoming tangent before any output word
        const Sw s = switches(xi, g.thr, ts);
        if (!fin(x) || !fin(ts)) atomicOr(&failures[0], 1);
        const float dsea = s.sea_clamp ? 0.f : x;             // L42 + L47
        const float dice = (s.l51 || s.l52) ? 0.f : x;        // L43 + L51/L52
        t_sea[w] = dsea;
        t_ice[w] = dice;
#if AD_GAIN_PROBE
        x2 += (double)x * x; t2s += (double)dsea * dsea; t2i += (double)dice * dice; nw += 1;
        m[3] = fminf(m[3], fabsf(xi - g.thr)); m[4] = fminf(m[4], fabsf(xi - 1.f));
        m[5] = fminf(m[5], fabsf(xi - 0.2f)); m[6] = fminf(m[6], fabsf(xi - 0.1f));
        if (s.ice) {
          ni += 1; ns += s.sea_clamp; n51 += s.l51; n52 += s.l52; ntie += (ts == 271.4f);
          m[0] = fminf(m[0], fabsf(ts - 271.4f));
          if (xi < 0.2f) m[1] = fminf(m[1], fabsf(ts - 253.15f));
          if (xi < 0.1f) m[2] = fminf(m[2], fabsf(ts - 263.15f));
        }
#endif
      }
    }
  }
#if AD_GAIN_PROBE
  if (nw) {
    atomicAdd(&g_x2, x2); atomicAdd(&g_t2[0], t2s); atomicAdd(&g_t2[1], t2i);
    atomicAdd(&g_words, nw); atomicAdd(&g_ice, ni); atomicAdd(&g_sea, ns); atomicAdd(&g_l51, n51); atomicAdd(&g_l52, n52); atomicAdd(&g_tie, ntie);
    for (int k = 0; k < 7; ++k) atomicMin((int *)&g_m[k], __float_as_int(m[k]));   // nonnegative floats order as int bits
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned int done = atomicAdd(&g_blocks, 1u) + 1u;
    if (done == gridDim.x * gridDim.y * gridDim.z) {
      __threadfence();
      printf("ADGAIN {\"kernel\":\"oracle_drivers_correct_tsk_over_seaice\",\"x2\":%.17g,\"t2_sea\":%.17g,\"t2_ice\":%.17g,\"words\":%u,"
             "\"ice_cells\":%u,\"sea_clamped\":%u,\"ice_l51\":%u,\"ice_l52\":%u,\"sea_ties\":%u,"
             "\"margin_tsk_271_4\":%.9g,\"margin_tsk_253_15\":%.9g,\"margin_tsk_263_15\":%.9g,"
             "\"margin_xice_thr\":%.9g,\"margin_xice_1\":%.9g,\"margin_xice_0_2\":%.9g,\"margin_xice_0_1\":%.9g}\n",
             g_x2, g_t2[0], g_t2[1], g_words, g_ice, g_sea, g_l51, g_l52, g_tie, g_m[0], g_m[1], g_m[2], g_m[3], g_m[4], g_m[5], g_m[6]);
      g_x2 = 0.0; g_t2[0] = 0.0; g_t2[1] = 0.0; g_words = g_ice = g_sea = g_l51 = g_l52 = g_tie = 0u;
      for (int k = 0; k < 7; ++k) g_m[k] = 3.4e38f;
      g_blocks = 0u;
    }
  }
#endif
}

extern "C" __global__ void oracle_drivers_correct_tsk_over_seaice__ad(const unsigned long long *ptrs, const unsigned long long *bptrs,
                                                                     int nc, int *failures) {
  using namespace cts;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= nc) return;
  if (c == 0 && aliased(ptrs)) atomicOr(&failures[1], 1);
  const Geo g = geo(ptrs, c);
  const float *xice = (const float *)ptrs[XICE], *tsk = (const float *)ptrs[TSK];
  float *b_tsk = (float *)bptrs[TSK], *b_sea = (float *)bptrs[SEA], *b_ice = (float *)bptrs[ICE];
  for (int j = g.jte; j >= g.jts; --j) {
    for (int i = g.ite; i >= g.its; --i) {
      const long long w = word(g, c, nc, i, j);
      const float ts = tsk[w];
      const Sw s = switches(xice[w], g.thr, ts);
      // the forward overwrote both output words: consume their cotangents and clear them
      const float ys = b_sea[w], yi = b_ice[w];
      b_sea[w] = 0.f; b_ice[w] = 0.f;
      if (!fin(ys) || !fin(yi) || !fin(ts)) atomicOr(&failures[0], 1);
      // L43 + L51/L52 reverse, then L42 + L47 reverse (reverse statement order)
      if (!(s.l51 || s.l52)) b_tsk[w] = __fadd_rn(b_tsk[w], yi);
      if (!s.sea_clamp) b_tsk[w] = __fadd_rn(b_tsk[w], ys);
    }
  }
}
