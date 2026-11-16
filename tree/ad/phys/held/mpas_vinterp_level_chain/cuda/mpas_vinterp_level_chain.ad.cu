// SPDX-License-Identifier: Apache-2.0
// Derivative entry (ad.entry/v1: tl / ad, plus a harness-only __check) for mpas_vinterp_level_chain, the RRTMG LW
// ozone-climatology vertical interpolation (MPAS module_ra_rrtmg_vinterp.F:114-165): launched once per model level
// k = nl-1 .. 0 by hexcore/mpasphys/rrtmg_lw_v842.py:988 (vinterp_ozn_device; kernel at rrtmg_lw_v842.py:325, capture
// source sha256 683d7a07 line 9370). Written by hand from the forward CUDA source.
//
// Forward (one thread per column c, level k of this launch; f32 __f*_rn, RLW_* macros):
//   p = pmid[c,k];  u = upper[c];  for kk = starts[k] .. nz-2: if pin[kk] < p <= pin[kk+1] { u = kk; break }
//   upper[c] = u;  warp-min of u -> atomicMin(starts[k-1]) when k > 0           (integer search state)
//   p <  pin[0]    : out[c,k] = (oz[c,0]*p) / pin[0]                              (top: scale to zero)
//   p >  pin[nz-1] : out[c,k] = oz[c,nz-1]                                        (bottom: constant clamp)
//   otherwise      : dpu = p - pin[u], dpl = pinl - p (pinl = pin[u+1], 0 if u+1 >= nz; ozl likewise)
//                    out[c,k] = (oz[c,u]*dpl + ozl*dpu) / (dpl + dpu)             (linear in p between levels)
// Only out[c,k] is a float output of the launch; every other word of `out` is untouched (identity).
//
// Active: pmid (tangent in / adjoint accumulate), ozone (tangent in / adjoint accumulate), out (inout: the launch's
// level-k words are overwritten, the rest pass through). Inactive: pin (the fixed climatology pressure coordinate,
// pools['atm_input']['pin'], read from the init file and never written by the model), the integer search state
// starts / upper (written by the forward, not differentiable), and the scalars nc, nl, nz, k.
//
// Frozen coefficients (limiter-like selection, documented): the branch (top / bottom / interior) and the bracket
// index u are piecewise selections of the forward. The TL/AD recompute them from the PRE-launch primal words
// (pmid, pin, starts, upper) exactly as the forward does and differentiate the selected piece only; the selection
// is frozen at the trajectory. The map is continuous at every knot (top: oz0*p/pin0 -> oz0 at p = pin0; bottom:
// interior -> ozl at p = pin[nz-1]; interior pieces meet at pin levels), so the frozen derivative is the one-sided
// derivative of a C0 piecewise-linear map. The bottom clamp gives d out / d p = 0 (frozen-coefficient clamp).
//
// Precision (the glue-row rule): the f32 primal words the partials need (dpu, dpl, their sum) are formed as the
// forward forms them (__fsub_rn / __fadd_rn, built --fmad=false --ftz=false; __check reproduces out, upper and
// starts bitwise from the same helper). Local partials in double from those f32 words; one f32 rounding per written
// derivative word. One thread owns one column's words (pmid[c,k], ozone[c,*], out[c,k]), so no atomics: the adjoint
// is deterministic. Primal arguments are the pre-launch words (the window binds primal without `post` to its
// pre-launch copy), so the in-place integer updates of the forward do not reach the derivative.
// Refusal: a nonfinite p, coefficient, seed or written word, or a bracket index outside [0, nz-1] on the interior
// branch, sets failure slot 0 (source rrtmg_lw_v842.py:988).

struct MvlPiece { int kind; int u; float out; double cp, c0, c1; int j0, j1; bool bad; };
// kind 0 top, 1 bottom, 2 interior.  d out = cp*dp + c0*d oz[j0] + c1*d oz[j1]  (j < 0: no term).

__device__ __forceinline__ MvlPiece mvl_piece(int nl, int nz, int k, int c, const float *pmid, const float *pin,
                                              const float *ozone, const int *starts, const int *upper) {
  MvlPiece q; q.c0 = q.c1 = q.cp = 0.0; q.j0 = q.j1 = -1; q.bad = false;
  float p = pmid[(long long)c * nl + k];
  int u = upper[c];
  for (int kk = starts[k]; kk < nz - 1; ++kk) if (pin[kk] < p && p <= pin[kk + 1]) { u = kk; break; }
  q.u = u;
  const float *oz = ozone + (long long)c * nz;
  if (p < pin[0]) {
    q.kind = 0;
    q.out = __fdiv_rn(__fmul_rn(oz[0], p), pin[0]);
    q.cp = (double)oz[0] / (double)pin[0];
    q.c0 = (double)p / (double)pin[0]; q.j0 = 0;
  } else if (p > pin[nz - 1]) {
    q.kind = 1;
    q.out = oz[nz - 1];
    q.c0 = 1.0; q.j0 = nz - 1;
  } else {
    q.kind = 2;
    if (u < 0 || u >= nz) { q.bad = true; q.out = 0.0f; return q; }
    bool has = u + 1 < nz;
    float lower_pin = has ? pin[u + 1] : 0.0f;
    float lower_ozone = has ? oz[u + 1] : 0.0f;
    float dpu = __fsub_rn(p, pin[u]), dpl = __fsub_rn(lower_pin, p);
    float den = __fadd_rn(dpl, dpu);
    q.out = __fdiv_rn(__fadd_rn(__fmul_rn(oz[u], dpl), __fmul_rn(lower_ozone, dpu)), den);
    double D = (double)den;
    // N = oz_u*dpl + ozl*dpu, D = dpl + dpu; d dpu/dp = +1, d dpl/dp = -1, so dD/dp = 0 and dN/dp = ozl - oz_u.
    q.cp = ((double)lower_ozone - (double)oz[u]) / D;
    q.c0 = (double)dpl / D; q.j0 = u;
    if (has) { q.c1 = (double)dpu / D; q.j1 = u + 1; }
  }
  q.bad |= !isfinite(p) || !isfinite(q.cp) || !isfinite(q.c0) || !isfinite(q.c1);
  return q;
}

// Harness only (not in launch_abi): the forward's words from the shared helper, out of place for out/upper/starts.
extern "C" __global__ void mpas_vinterp_level_chain__check(int nc, int nl, int nz, int k, const float *pmid,
    const float *pin, const float *ozone, const int *starts, const int *upper, float *out_o, int *upper_o, int *starts_o,
    int *failure) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int u = 0x7fffffff;
  if (c < nc) {
    MvlPiece q = mvl_piece(nl, nz, k, c, pmid, pin, ozone, starts, upper);
    u = q.u; upper_o[c] = u; out_o[(long long)c * nl + k] = q.out;
    if (q.bad || !isfinite(q.out)) atomicExch(failure, 1);
  }
  if (k == 0) return;
  for (int offset = 16; offset > 0; offset >>= 1) u = min(u, __shfl_down_sync(0xffffffffu, u, offset));
  if ((threadIdx.x & 31) == 0) atomicMin(starts_o + k - 1, u);
}

extern "C" __global__ void mpas_vinterp_level_chain__tl(int nc, int nl, int nz, int k, const float *pmid,
    const float *pin, const float *ozone, const int *starts, const int *upper,
    const float *d_pmid, const float *d_ozone, float *d_out, int *failure) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= nc) return;
  MvlPiece q = mvl_piece(nl, nz, k, c, pmid, pin, ozone, starts, upper);
  long long jo = (long long)c * nl + k;
  const float *doz = d_ozone + (long long)c * nz;
  double dp = (double)d_pmid[jo];
  double v = q.cp * dp;
  if (q.j0 >= 0) v += q.c0 * (double)doz[q.j0];
  if (q.j1 >= 0) v += q.c1 * (double)doz[q.j1];
  float t = (float)v;
  d_out[jo] = t;
  if (q.bad || !isfinite(dp) || !isfinite(t)) atomicExch(failure, 1);
}

extern "C" __global__ void mpas_vinterp_level_chain__ad(int nc, int nl, int nz, int k, const float *pmid,
    const float *pin, const float *ozone, const int *starts, const int *upper,
    float *b_pmid, float *b_ozone, float *b_out, int *failure) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= nc) return;
  MvlPiece q = mvl_piece(nl, nz, k, c, pmid, pin, ozone, starts, upper);
  long long jo = (long long)c * nl + k;
  float *boz = b_ozone + (long long)c * nz;
  double y = (double)b_out[jo];
  b_out[jo] = 0.0f;                                   // the level-k word is a full overwrite: its seed is consumed
  bool bad = q.bad || !isfinite(y);
  float t;
  if (q.kind != 1) { t = (float)((double)b_pmid[jo] + q.cp * y); b_pmid[jo] = t; bad |= !isfinite(t); }
  if (q.j0 >= 0) { t = (float)((double)boz[q.j0] + q.c0 * y); boz[q.j0] = t; bad |= !isfinite(t); }
  if (q.j1 >= 0) { t = (float)((double)boz[q.j1] + q.c1 * y); boz[q.j1] = t; bad |= !isfinite(t); }
  if (bad) atomicExch(failure, 1);
}
