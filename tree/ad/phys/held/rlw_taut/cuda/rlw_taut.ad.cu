// SPDX-License-Identifier: Apache-2.0
// Held-kernel lane rlw_taut (ad/held-rlw_taut, private autodiff program): augmented forward, tangent and adjoint of
// the RRTMG LW launch rlw_taut, written by hand from the forward CUDA source
//   gpuwm-hex 16251589 tree/src/hexcore/mpasphys/rrtmg_lw_v842.py:808-814 (Python launch; rrtmg_lw_v842.py sha256
//   f69f4f17), kernel source captured as objects/683d7a07...cu lines 1847-1860, store at 1858-1859 (image 34da9e09):
//     taut[tid] = RLW_AD(taug[tid], taua[collay*NBNDLW + (ngb[ig]-1)]),   RLW_AD(a,b) = __fadd_rn(a,b)
//   with tid = (col*nl + lay)*NGPTLW + ig, collay = tid / NGPTLW, ig = tid % NGPTLW, NGPTLW = 140, NBNDLW = 16.
// RRTMG v4.85 rrtmg_lw_rad.f90 "taut = taug + taua(ngb)" (the clear-sky-plus-aerosol optical depth per g-point).
//
// The map is affine (linear) in the two active inputs and contains no limiter, clamp or branch on a real value:
//   d taut[tid] = d taug[tid] + d taua[collay, ngb[ig]-1].
// ngb is an integer table (fixed, inactive); ncol and nl are launch constants. No frozen-coefficient treatment is
// needed: there is nothing to freeze (documented in the spec's "branches": the only branch is the integer bound test).
//
// ABI (ad.entry/v1, spec rlw_taut.spec.json), the forward's own launch geometry (one thread per (col, lay, ig)):
//   __aug: repeats the forward exactly (identity proof against the captured image).
//   __tl : writes only the output tangent dtaut (taut is write-only in the forward: `cp.empty`, never read).
//   __ad : accumulates both input cotangents and consumes the write-only output cotangent (sets it to +0).
//          taua's cotangent is a GATHER over the g-points of one band. To keep the adjoint deterministic (bitwise
//          across runs and processes, no atomics), the thread of the FIRST g-point of each band (lowest ig with that
//          ngb value) owns the whole (col, lay, band) group: it walks ig ascending over every g-point with that band,
//          adds each cotangent to taug's, sums them in float64 in that fixed order, zeroes them, and adds the sum to
//          taua's cotangent with one rounding. The other threads of the group do nothing, so every derivative word
//          has exactly one writer. ngb need not be contiguous (it is, in RRTMG: 140 g-points in 16 runs).
// Aliasing: the forward's taut is a fresh `cp.empty` allocation (rrtmg_lw_v842.py:808), so taut never aliases taug or
// taua; the entries still handle taut's shadow == taug's shadow exactly (in-place form: the taug word passes
// through unchanged, d/d taug = 1) and record it in failure slot 2 so a plan that ever aliases them is named.
//
// Precision (as the live P3/P4/P5 schemes and the e2e entries, COMMON-AD 39.4): derivative arithmetic in float64,
// one rounding per stored derivative word; +0.0 before each store so the zero direction has no negative zeros.
// Failure slots: 0 non-finite primal/tangent/adjoint word, 1 ngb outside 1..NBNDLW (would read outside taua's row),
// 2 taut shadow aliases taug shadow (handled exactly, named only).

#define NBNDLW 16
#define NGPTLW 140

__device__ __forceinline__ void flag(int* failure, int slot) { if (failure) atomicExch(failure + slot, 1); }

extern "C" __global__ void rlw_taut__aug(
    int ncol, int nl, const float* __restrict__ taug, const float* __restrict__ taua, const int* __restrict__ ngb,
    float* __restrict__ taut, int* failure) {
  long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (long long)ncol * nl * NGPTLW) return;
  int ig = (int)(tid % NGPTLW);
  long long collay = tid / NGPTLW;
  int b = ngb[ig] - 1;
  if (b < 0 || b >= NBNDLW) { flag(failure, 1); return; }
  const float g = taug[tid], a = taua[collay * NBNDLW + b];
  const float v = __fadd_rn(g, a);
  taut[tid] = v;
  if (!isfinite(g) || !isfinite(a) || !isfinite(v)) flag(failure, 0);
}

extern "C" __global__ void rlw_taut__tl(
    int ncol, int nl, const int* __restrict__ ngb,
    const float* dtaug, const float* dtaua, float* dtaut, int* failure) {
  long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (long long)ncol * nl * NGPTLW) return;
  int ig = (int)(tid % NGPTLW);
  long long collay = tid / NGPTLW;
  int b = ngb[ig] - 1;
  if (b < 0 || b >= NBNDLW) { flag(failure, 1); return; }
  const double g = (double)dtaug[tid], a = (double)dtaua[collay * NBNDLW + b];
  const float v = (float)(g + a + 0.0);
  dtaut[tid] = v;   // in-place form (dtaut == dtaug) is the same per-thread read-then-write
  if (dtaut == dtaug) flag(failure, 2);
  if (!isfinite(g) || !isfinite(a) || !isfinite(v)) flag(failure, 0);
}

extern "C" __global__ void rlw_taut__ad(
    int ncol, int nl, const int* __restrict__ ngb,
    float* btaug, float* btaua, float* btaut, int* failure) {
  long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (long long)ncol * nl * NGPTLW) return;
  const int ig = (int)(tid % NGPTLW);
  const long long collay = tid / NGPTLW;
  const long long row = collay * NGPTLW;
  const int band = ngb[ig];
  if (band < 1 || band > NBNDLW) { flag(failure, 1); return; }
  for (int j = 0; j < ig; ++j) if (ngb[j] == band) return;   // not the first g-point of its band: owned elsewhere
  const bool alias = (btaut == btaug);
  if (alias) flag(failure, 2);
  double sum = 0.0;
  for (int j = ig; j < NGPTLW; ++j) {
    if (ngb[j] != band) continue;
    const long long t = row + j;
    const double c = (double)btaut[t];
    if (!isfinite(c)) flag(failure, 0);
    sum += c;
    if (!alias) {
      const float g = (float)((double)btaug[t] + c + 0.0);
      btaug[t] = g;
      btaut[t] = 0.0f;
      if (!isfinite(g)) flag(failure, 0);
    }
    // alias: the word is both the post-launch taut cotangent and the pre-launch taug cotangent; d taut/d taug = 1,
    // so it stays as it is (no clear: the clear and the accumulate cancel exactly).
  }
  const long long ia = collay * NBNDLW + (band - 1);
  const float a = (float)((double)btaua[ia] + sum + 0.0);
  btaua[ia] = a;
  if (!isfinite(a)) flag(failure, 0);
}
