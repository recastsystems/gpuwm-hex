// SPDX-License-Identifier: Apache-2.0
// Private autodiff program (ad/held-oracle_drivers_driver_forcing). Whole-launch tangent-linear and adjoint
// entries of hex 16251589 oracle_drivers_driver_forcing (tree/src/hexcore/mpasphys/rrfs/ruc_lsm_8c9f1a8.py:5454,
// device body drivers_driver_forcing at :5076; RRFS-MPAS 1d6e370e mpas_atmphys_driver_seaice.F L610-L639),
// written by hand from the forward CUDA source.
//
// Forward, per thread c < nc and i = 1..ncells[c] (element (i) of every array is word c + (i-1)*nc):
//   sr_p     = micro_on ? sr : 0
//   rainbl_p = ((0 + rainncv) if micro_on) + raincv if conv_on          (__fadd_rn)
//   swdown_p = gsw / (1 - sfc_albedo)                                    (__fsub_rn, __fdiv_rn)
// Precision: float32, explicitly rounded intrinsics only, built with --fmad=false --ftz=false (as the live schemes).
// micro_on, conv_on and ncells are integer configuration read from the forward's own pointer table: inactive.
// No clamp, limiter or min/max: no frozen-coefficient treatment is needed (none exists in the forward).
// Every output is fully overwritten: TL writes d_out = J d_in; AD accumulates J^T b_out into the inputs and clears
// b_out. Inputs are read-only in the forward, so the live (post-launch) primal through the table equals the
// pre-launch primal; the entries refuse (slot 1) if an output pointer equals an input pointer.
//
// Pointer table (layout::FORCING, ruc_layout.rs:339): 0 ncells, 1 micro_on, 2 conv_on (int32), 3 sr, 4 rainncv,
// 5 raincv, 6 gsw, 7 sfc_albedo (inputs), 8 sr_p, 9 rainbl_p, 10 swdown_p (outputs). The derivative table is the
// dispatcher mirror of the same table (indirect.mirror): word k points at the shadow of the forward pointer k.
//
// Failure slots: 0 nonfinite input, cotangent or 1 - sfc_albedo == 0 (swdown_p and its derivative undefined);
//                1 an output pointer equals an input pointer (in-place AD would be wrong).
// AD_GAIN_PROBE (diagnostic image only, never the bound image): the TL entry also reduces ||x_in||, ||TL x|| per
// path and the primal range of 1 - sfc_albedo, and prints one ADGAIN line per TL launch.

#ifndef AD_GAIN_PROBE
#define AD_GAIN_PROBE 0
#endif
#if AD_GAIN_PROBE
#include <cstdio>
#endif

namespace odf {
enum { NCELLS = 0, MICRO = 1, CONV = 2, SR = 3, RAINNCV = 4, RAINCV = 5, GSW = 6, ALB = 7, SR_P = 8, RAINBL_P = 9, SWDOWN_P = 10, NPTR = 11 };
__device__ __forceinline__ bool fin(float v) { return isfinite(v); }
__device__ __forceinline__ bool aliased(const unsigned long long *p) {
  for (int o = SR_P; o <= SWDOWN_P; ++o)
    for (int k = SR; k <= ALB; ++k) if (p[o] == p[k]) return true;
  return false;
}
#if AD_GAIN_PROBE
// x2: sum over active input words of x^2 (sr, rainncv, raincv, gsw, alb); t2: sum of TL output^2 (sr_p, rainbl_p,
// swdown_p, swdown_p from gsw only, swdown_p from alb only); range: min/max (1-a), max 1/(1-a), max |s/(1-a)|
__device__ double g_x2[5], g_t2[5];
__device__ float g_rmin = 3.4e38f, g_rmax, g_inv, g_sens;
__device__ unsigned int g_blocks, g_cells, g_micro, g_conv;
#endif
}

extern "C" __global__ void oracle_drivers_driver_forcing__tl(const unsigned long long *ptrs, const unsigned long long *dptrs,
                                                            int nc, int *failures) {
  using namespace odf;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
#if AD_GAIN_PROBE
  double x2[5] = {0, 0, 0, 0, 0}, t2[5] = {0, 0, 0, 0, 0};
  float rmin = 3.4e38f, rmax = 0.f, inv = 0.f, sens = 0.f;
  unsigned int cells = 0, nm = 0, ncv = 0;
#endif
  if (c < nc) {
    if (c == 0 && aliased(ptrs)) atomicOr(&failures[1], 1);
    const int ncells = ((const int *)ptrs[NCELLS])[c], micro_on = ((const int *)ptrs[MICRO])[c], conv_on = ((const int *)ptrs[CONV])[c];
    const float *gsw = (const float *)ptrs[GSW], *alb = (const float *)ptrs[ALB];
    const float *t_sr = (const float *)dptrs[SR], *t_rainncv = (const float *)dptrs[RAINNCV], *t_raincv = (const float *)dptrs[RAINCV];
    const float *t_gsw = (const float *)dptrs[GSW], *t_alb = (const float *)dptrs[ALB];
    float *t_sr_p = (float *)dptrs[SR_P], *t_rainbl_p = (float *)dptrs[RAINBL_P], *t_swdown_p = (float *)dptrs[SWDOWN_P];
    for (int i = 1; i <= ncells; ++i) {
      const long long w = c + (long long)(i - 1) * nc;
      // read every incoming tangent before any output word is written
      const float x_sr = micro_on ? t_sr[w] : 0.f, x_rn = micro_on ? t_rainncv[w] : 0.f, x_rc = conv_on ? t_raincv[w] : 0.f;
      const float x_g = t_gsw[w], x_a = t_alb[w];
      const float den = __fsub_rn(1.f, alb[w]);                 // L638
      const float s = __fdiv_rn(gsw[w], den);                   // primal swdown_p, rounded as the forward rounds it
      float d_rb = 0.f;                                          // L611
      if (micro_on) d_rb = __fadd_rn(d_rb, x_rn);                // L622
      if (conv_on) d_rb = __fadd_rn(d_rb, x_rc);                 // L631
      // L638: d(g/(1-a)) = (dg + s*da) / (1-a)
      const float d_sw = __fdiv_rn(__fadd_rn(x_g, __fmul_rn(s, x_a)), den);
      if (den == 0.f || !fin(s) || !fin(x_sr) || !fin(x_rn) || !fin(x_rc) || !fin(x_g) || !fin(x_a)) atomicOr(&failures[0], 1);
      t_sr_p[w] = x_sr;                                          // L610/L621
      t_rainbl_p[w] = d_rb;
      t_swdown_p[w] = d_sw;
#if AD_GAIN_PROBE
      const float sg = __fdiv_rn(x_g, den), sa = __fdiv_rn(__fmul_rn(s, x_a), den);
      x2[0] += (double)t_sr[w] * t_sr[w]; x2[1] += (double)t_rainncv[w] * t_rainncv[w]; x2[2] += (double)t_raincv[w] * t_raincv[w];
      x2[3] += (double)x_g * x_g; x2[4] += (double)x_a * x_a;
      t2[0] += (double)x_sr * x_sr; t2[1] += (double)d_rb * d_rb; t2[2] += (double)d_sw * d_sw; t2[3] += (double)sg * sg; t2[4] += (double)sa * sa;
      rmin = fminf(rmin, den); rmax = fmaxf(rmax, den); inv = fmaxf(inv, fabsf(__fdiv_rn(1.f, den))); sens = fmaxf(sens, fabsf(__fdiv_rn(s, den)));
      cells += 1; nm += micro_on != 0; ncv += conv_on != 0;
#endif
    }
  }
#if AD_GAIN_PROBE
  for (int k = 0; k < 5; ++k) { if (x2[k] != 0.0) atomicAdd(&g_x2[k], x2[k]); if (t2[k] != 0.0) atomicAdd(&g_t2[k], t2[k]); }
  if (cells) {
    atomicAdd(&g_cells, cells); atomicAdd(&g_micro, nm); atomicAdd(&g_conv, ncv);
    // nonnegative floats order as their int bits (1 - albedo < 0 would be bad input: clamped to 0 here, probe only)
    atomicMin((int *)&g_rmin, rmin < 0.f ? 0 : __float_as_int(rmin)); atomicMax((int *)&g_rmax, __float_as_int(rmax));
    atomicMax((int *)&g_inv, __float_as_int(inv)); atomicMax((int *)&g_sens, __float_as_int(sens));
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned int done = atomicAdd(&g_blocks, 1u) + 1u;
    if (done == gridDim.x * gridDim.y * gridDim.z) {
      __threadfence();
      printf("ADGAIN {\"x2\":[%.17g,%.17g,%.17g,%.17g,%.17g],\"t2\":[%.17g,%.17g,%.17g,%.17g,%.17g],\"one_minus_albedo_min\":%.9g,"
             "\"one_minus_albedo_max\":%.9g,\"max_inv\":%.9g,\"max_swdown_over_den\":%.9g,\"cells\":%u,\"micro_cells\":%u,\"conv_cells\":%u}\n",
             g_x2[0], g_x2[1], g_x2[2], g_x2[3], g_x2[4], g_t2[0], g_t2[1], g_t2[2], g_t2[3], g_t2[4],
             g_rmin, g_rmax, g_inv, g_sens, g_cells, g_micro, g_conv);
      for (int k = 0; k < 5; ++k) { g_x2[k] = 0.0; g_t2[k] = 0.0; }
      g_rmin = 3.4e38f; g_rmax = 0.f; g_inv = 0.f; g_sens = 0.f; g_cells = 0u; g_micro = 0u; g_conv = 0u; g_blocks = 0u;
    }
  }
#endif
}

extern "C" __global__ void oracle_drivers_driver_forcing__ad(const unsigned long long *ptrs, const unsigned long long *bptrs,
                                                            int nc, int *failures) {
  using namespace odf;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= nc) return;
  if (c == 0 && aliased(ptrs)) atomicOr(&failures[1], 1);
  const int ncells = ((const int *)ptrs[NCELLS])[c], micro_on = ((const int *)ptrs[MICRO])[c], conv_on = ((const int *)ptrs[CONV])[c];
  const float *gsw = (const float *)ptrs[GSW], *alb = (const float *)ptrs[ALB];
  float *b_sr = (float *)bptrs[SR], *b_rainncv = (float *)bptrs[RAINNCV], *b_raincv = (float *)bptrs[RAINCV];
  float *b_gsw = (float *)bptrs[GSW], *b_alb = (float *)bptrs[ALB];
  float *b_sr_p = (float *)bptrs[SR_P], *b_rainbl_p = (float *)bptrs[RAINBL_P], *b_swdown_p = (float *)bptrs[SWDOWN_P];
  for (int i = ncells; i >= 1; --i) {
    const long long w = c + (long long)(i - 1) * nc;
    const float den = __fsub_rn(1.f, alb[w]);
    const float s = __fdiv_rn(gsw[w], den);
    // the forward overwrote every output word: consume its cotangent and clear it
    const float y_srp = b_sr_p[w], y_rb = b_rainbl_p[w], y_sw = b_swdown_p[w];
    b_sr_p[w] = 0.f; b_rainbl_p[w] = 0.f; b_swdown_p[w] = 0.f;
    if (den == 0.f || !fin(s) || !fin(y_srp) || !fin(y_rb) || !fin(y_sw)) atomicOr(&failures[0], 1);
    // L638 reverse: q = y_sw / (1-a); b_gsw += q; b_alb += s*q
    const float q = __fdiv_rn(y_sw, den);
    b_gsw[w] = __fadd_rn(b_gsw[w], q);
    b_alb[w] = __fadd_rn(b_alb[w], __fmul_rn(s, q));
    if (conv_on) b_raincv[w] = __fadd_rn(b_raincv[w], y_rb);     // L631
    if (micro_on) {                                              // L621/L622
      b_rainncv[w] = __fadd_rn(b_rainncv[w], y_rb);
      b_sr[w] = __fadd_rn(b_sr[w], y_srp);
    }
  }
}
