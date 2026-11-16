// SPDX-License-Identifier: Apache-2.0
// Held-kernel lane mpas_pressure_mb (ad/held-mpas_pressure_mb, private autodiff program): augmented forward, tangent
// and adjoint of the RRTMG LW launch mpas_pressure_mb, written by hand from the forward CUDA source
//   gpuwm-hex 16251589 tree/src/hexcore/mpasphys/rrtmg_lw_v842.py:609-612 (rrtmg_lw_v842.py sha256 f69f4f17), kernel
//   source captured as objects/683d7a07...cu lines 9653-9657 (image 34da9e09):
//     if (i < n) out[i] = __fdiv_rn(p[i], 100.f);        // MPAS mpas_atmphys_driver_radiation_lw.F:490-492, Pa -> hPa
// Launch sites (all one thread per word, grid ceil(n/128) x 128): rrtmg_lw_v842.py:1408, rrfs/radiation_rrfs.py:550,
// rrfs/driver_rrfs.py:2101, rrfs/resident_radiation_rrfs.py:272. Output feeds vinterp_ozn_device (the o3input == 2
// ozone climatology interpolation, mpas_vinterp_level_chain). hrrrv5 x1.2562: 2 launches per radiation step,
// n = 151,158 (2,562 cells x 59 levels).
//
// The map is linear in its one active input, with a constant coefficient and no limiter, clamp or real-valued branch:
//   d out[i] = d p[i] / 100.
// n is a launch constant. No frozen-coefficient treatment is needed: there is nothing to freeze (the only branch is the
// integer bound test i < n, recorded in the spec's "branches").
//
// ABI (ad.entry/v1, spec mpas_pressure_mb.spec.json), the forward's own launch geometry:
//   __aug: repeats the forward exactly (identity proof against the captured image).
//   __tl : writes only the output tangent dout (out is write-only in the forward: `cp.empty_like` / a scratch buffer
//          at every launch site, never read).
//   __ad : accumulates the p cotangent and consumes the write-only out cotangent (sets it to +0). One writer per word:
//          thread i owns p[i] and out[i] and nothing else, so the adjoint is deterministic with no atomics.
// Aliasing: every launch site passes a fresh output allocation, so out never aliases p (on the captures p and out are
// different allocations). The entries still handle out's shadow == p's shadow exactly (in place: TL w <- w/100; AD the
// word is both the post-launch out cotangent and the pre-launch p cotangent, so w <- w/100 with no clear), and record it
// in failure slot 1 so a plan that ever aliases them is named.
//
// Precision (as the live P3/P4/P5 schemes and the e2e entries, COMMON-AD 39.4): derivative arithmetic in float64, one
// rounding per stored derivative word; +0.0 before each store so the zero direction has no negative zeros. The
// coefficient is applied as a float64 division by 100 (the exact derivative of the forward's division; 0.01 is not a
// binary number, so a multiply by 0.01 would carry the representation error of the coefficient).
// Failure slots: 0 non-finite primal/tangent/adjoint word, 1 out shadow aliases p shadow (handled exactly, named only).

__device__ __forceinline__ void flag(int* failure, int slot) { if (failure) atomicExch(failure + slot, 1); }

extern "C" __global__ void mpas_pressure_mb__aug(int n, const float* __restrict__ p, float* __restrict__ out,
                                                 int* failure) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float x = p[i];
  const float v = __fdiv_rn(x, 100.f);
  out[i] = v;
  if (!isfinite(x) || !isfinite(v)) flag(failure, 0);
}

extern "C" __global__ void mpas_pressure_mb__tl(int n, const float* dp, float* dout, int* failure) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double x = (double)dp[i];
  const float v = (float)(x / 100.0 + 0.0);
  dout[i] = v;   // in-place form (dout == dp) is the same per-thread read-then-write
  if (dout == dp) flag(failure, 1);
  if (!isfinite(x) || !isfinite(v)) flag(failure, 0);
}

extern "C" __global__ void mpas_pressure_mb__ad(int n, float* bp, float* bout, int* failure) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double c = (double)bout[i];
  if (!isfinite(c)) flag(failure, 0);
  if (bout == bp) {
    // alias: one word is both the post-launch out cotangent and the pre-launch p cotangent: w <- w / 100, no clear.
    flag(failure, 1);
    const float g = (float)(c / 100.0 + 0.0);
    bp[i] = g;
    if (!isfinite(g)) flag(failure, 0);
    return;
  }
  const float g = (float)((double)bp[i] + c / 100.0 + 0.0);
  bp[i] = g;
  bout[i] = 0.0f;
  if (!isfinite(g)) flag(failure, 0);
}
