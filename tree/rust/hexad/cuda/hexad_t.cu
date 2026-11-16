// SPDX-License-Identifier: Apache-2.0
// libhexad-phys: the increment <-> runner direction maps of
// libhexad-phys C ABI v2, from alloc-map.json (tree/ad/phys/host/mk_alloc_map.py).
//
// T0   : the host Increment(t0) -> runner tangent at boundary 0, every copy the forward reads. It is the linearisation of
//        the hex write-back (MPAS atm_init_coupled_diagnostics as hexcore.driver.load_mpas_initial_state transcribes it):
//          theta_m = theta (1 + rvord qv);  rho_zz = rho / zz;  ru = 0.5 u (rho_zz1 + rho_zz2);
//          rw = w Irho Izz - sum_slots sign (zb + sgn(flux) c3 zb3) flux Izz;  rho_p = rho_zz - rho_base;
//          rtheta_p = theta_m rho_p + rho_base (theta_m - theta_base);  rho_theta = rtheta_p + rtheta_base;
//          exner = (zz rgas/p0 rho_theta)^rcv;  pressure_p = zz rgas (exner rtheta_p + rtheta_base (exner - exner_base)).
// T0^T : its exact transpose.
// Ts   : runner tangent at boundary s -> Increment(t_s), a read-only projection from the canonical copies.
// Ts^T : its transpose, ADDED into the runner cotangent.
// Everything in float64 work buffers, plan layout (word = k * n + h). Every output element has exactly one writer and
// every sum has a fixed order: deterministic, no atomics. The float32 arena is touched only by t_put / t_sum.
#include <stdint.h>

extern "C" {

#define IDX (blockIdx.x * (int64_t)blockDim.x + threadIdx.x)
#define STRIDE ((int64_t)gridDim.x * blockDim.x)

// ---- generic moves ---------------------------------------------------------------------------------------------
// work[k*nh + h] = scale * field[k*ls + h*hs]   (field float64 or float32)
__global__ void t_inc_get(double* work, const void* field, int f64, int64_t nlev, int64_t nh, int64_t ls, int64_t hs, double scale) {
  int64_t n = nlev * nh;
  for (int64_t i = IDX; i < n; i += STRIDE) {
    int64_t k = i / nh, h = i - k * nh, j = k * ls + h * hs;
    double v = f64 ? ((const double*)field)[j] : (double)((const float*)field)[j];
    work[i] = scale * v;
  }
}
// field[k*ls + h*hs] (=, +=) scale * work[k*nh + h]
__global__ void t_inc_put(void* field, int f64, const double* work, int64_t nlev, int64_t nh, int64_t ls, int64_t hs, double scale, int add) {
  int64_t n = nlev * nh;
  for (int64_t i = IDX; i < n; i += STRIDE) {
    int64_t k = i / nh, h = i - k * nh, j = k * ls + h * hs;
    double v = scale * work[i];
    if (f64) { double* p = (double*)field; p[j] = add ? p[j] + v : v; }
    else { float* p = (float*)field; p[j] = add ? (float)((double)p[j] + v) : (float)v; }
  }
}
// dst[i] (=, +=) scale * work[i]; dst float32 arena words (rounded once) or a float64 mirror
__global__ void t_put(void* dst, int f64, const double* work, int64_t n, double scale, int add) {
  for (int64_t i = IDX; i < n; i += STRIDE) {
    double v = scale * work[i];
    if (f64) { double* p = (double*)dst; p[i] = add ? p[i] + v : v; }
    else { float* p = (float*)dst; p[i] = add ? (float)((double)p[i] + v) : (float)v; }
  }
}
// work[i] = scale * sum_{c < m} src_c[i], copies summed in their listed order
__global__ void t_sum(double* work, const uint64_t* srcs, int m, int f64, int64_t n, double scale) {
  for (int64_t i = IDX; i < n; i += STRIDE) {
    double s = 0.0;
    for (int c = 0; c < m; c++) s += f64 ? ((const double*)srcs[c])[i] : (double)((const float*)srcs[c])[i];
    work[i] = scale * s;
  }
}
// float32 primal words -> float64
__global__ void t_widen(double* dst, const float* src, int64_t n) { for (int64_t i = IDX; i < n; i += STRIDE) dst[i] = (double)src[i]; }
__global__ void t_zero(double* x, int64_t n) { for (int64_t i = IDX; i < n; i += STRIDE) x[i] = 0.0; }
__global__ void t_axpy(double* y, const double* x, double a, int64_t n) { for (int64_t i = IDX; i < n; i += STRIDE) y[i] += a * x[i]; }
// counter-based N(0,1)-like (sum of 4 uniforms, centred, unit variance) for tests
__device__ static uint64_t mix(uint64_t z) { z += 0x9e3779b97f4a7c15ULL; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; return z ^ (z >> 31); }
__global__ void t_randn(double* x, int64_t n, uint64_t seed) {
  for (int64_t i = IDX; i < n; i += STRIDE) {
    double s = 0.0;
    for (int r = 0; r < 4; r++) s += (double)(mix(seed * 0x100000001b3ULL + 4 * (uint64_t)i + r) >> 11) * (1.0 / 9007199254740992.0);
    x[i] = (s - 2.0) * 1.7320508075688772;
  }
}
// fixed-order partial dots (1024 blocks x 256 threads), summed in block order on the host
__global__ void t_dot(const double* a, const double* b, int64_t n, double* part) {
  __shared__ double s[256];
  double v = 0.0;
  for (int64_t i = IDX; i < n; i += STRIDE) v += a[i] * b[i];
  s[threadIdx.x] = v; __syncthreads();
  for (int w = 128; w > 0; w >>= 1) { if (threadIdx.x < w) s[threadIdx.x] += s[threadIdx.x + w]; __syncthreads(); }
  if (threadIdx.x == 0) part[blockIdx.x] = s[0];
}

// ---- T0 (tangent) -----------------------------------------------------------------------------------------------
// per (k, c): thermodynamic part. pr: primal theta_m, qv, rho_p, rtheta_p, exner (float64, [nl][nc]).
__global__ void t0_cell(int64_t nl, int64_t nc, double rvord, double rgas, double rcv,
    const double* thm, const double* qv, const double* rp, const double* rtp, const double* ex,
    const double* zz, const double* rbase, const double* rtbase,
    const double* dth, const double* dqv, const double* drho,
    double* dthm, double* drz, double* drp, double* drtp, double* drth, double* dex, double* dpp) {
  int64_t n = nl * nc;
  for (int64_t i = IDX; i < n; i += STRIDE) {
    double a = 1.0 + rvord * qv[i], th = thm[i] / a;
    double t_thm = dth[i] * a + th * rvord * dqv[i];
    double t_rz = drho[i] / zz[i];
    double t_rtp = t_thm * rp[i] + thm[i] * t_rz + rbase[i] * t_thm;
    double rth = rtp[i] + rtbase[i];
    double t_ex = rcv * ex[i] / rth * t_rtp;
    double t_pp = zz[i] * rgas * (t_ex * rtp[i] + ex[i] * t_rtp + rtbase[i] * t_ex);
    dthm[i] = t_thm; drz[i] = t_rz; drp[i] = t_rz; drtp[i] = t_rtp; drth[i] = t_rtp; dex[i] = t_ex; dpp[i] = t_pp;
  }
}
// per (k, e): dru = 0.5 du (rz1 + rz2) + 0.5 u (drz1 + drz2)
__global__ void t0_ru(int64_t nl, int64_t ne, int64_t nc, const int* coe, const double* rz, const double* u,
                      const double* du, const double* drz, double* dru) {
  int64_t n = nl * ne;
  for (int64_t i = IDX; i < n; i += STRIDE) {
    int64_t k = i / ne, e = i - k * ne;
    int64_t c1 = coe[2 * e], c2 = coe[2 * e + 1];
    dru[i] = 0.5 * du[i] * (rz[k * nc + c1] + rz[k * nc + c2]) + 0.5 * u[i] * (drz[k * nc + c1] + drz[k * nc + c2]);
  }
}
// interface interpolants of level fields at (k, c), k in 1..nl-1
__device__ static double interp(const double* f, const double* fzm, const double* fzp, int64_t k, int64_t c, int64_t nc) {
  return fzm[k] * f[k * nc + c] + fzp[k] * f[(k - 1) * nc + c];
}
// per (k, c), k in 0..nl: drw. sgnf[(k*nc + c)*maxe + j] = sign of the primal flux (the terrain term's frozen branch).
__global__ void t0_rw(int64_t nl, int64_t nc, int64_t ne, int maxe, const int* eoc, const int* noe,
                      const double* fzm, const double* fzp, const double* zz, const double* rz, const double* w,
                      const double* zbc, const double* zb3c, const double* signs, const double* sgnf,
                      const double* dw, const double* drz, const double* dru, double* drw) {
  int64_t nw = nl + 1, n = nw * nc;
  for (int64_t i = IDX; i < n; i += STRIDE) {
    int64_t k = i / nc, c = i - k * nc;
    if (k == 0 || k == nl) { drw[i] = 0.0; continue; }
    double iz = interp(zz, fzm, fzp, k, c, nc), ir = interp(rz, fzm, fzp, k, c, nc), dir = interp(drz, fzm, fzp, k, c, nc);
    double v = dw[i] * ir * iz + w[i] * dir * iz;
    for (int j = 0; j < noe[c]; j++) {
      int64_t e = eoc[c * maxe + j];
      double df = fzm[k] * dru[k * ne + e] + fzp[k] * dru[(k - 1) * ne + e];
      int64_t m = (k * nc + c) * maxe + j;
      v -= signs[c * maxe + j] * (zbc[m] + sgnf[m] * zb3c[m]) * df * iz;
    }
    drw[i] = v;
  }
}

// ---- T0^T (cotangent) ------------------------------------------------------------------------------------------
// per (k, c), k in 0..nl: bar_dw += bar_drw Irho Izz (interior); bar_drz_rw[k] = w[k] Izz[k] fzm[k] b[k] + w[k+1] Izz[k+1] fzp[k+1] b[k+1]
__global__ void t0_rw_T_cell(int64_t nl, int64_t nc, const double* fzm, const double* fzp, const double* zz, const double* rz,
                             const double* w, const double* brw, double* bdw, double* bdrz) {
  int64_t nw = nl + 1, n = nw * nc;
  for (int64_t i = IDX; i < n; i += STRIDE) {
    int64_t k = i / nc, c = i - k * nc;
    if (k > 0 && k < nl) bdw[i] += brw[i] * interp(rz, fzm, fzp, k, c, nc) * interp(zz, fzm, fzp, k, c, nc);
    if (k < nl) {   // level k of rho_zz feeds interface k (fzm) and k + 1 (fzp)
      double s = 0.0;
      if (k >= 1) s += w[k * nc + c] * interp(zz, fzm, fzp, k, c, nc) * fzm[k] * brw[k * nc + c];
      if (k + 1 <= nl - 1) s += w[(k + 1) * nc + c] * interp(zz, fzm, fzp, k + 1, c, nc) * fzp[k + 1] * brw[(k + 1) * nc + c];
      bdrz[k * nc + c] += s;
    }
  }
}
// per (k, e), k in 0..nl-1: bar_dru from the terrain term of both cells of e at interfaces k (fzm) and k + 1 (fzp)
__global__ void t0_rw_T_edge(int64_t nl, int64_t nc, int64_t ne, int maxe, const int* coe, const int* eoc, const int* noe,
                             const double* fzm, const double* fzp, const double* zz,
                             const double* zbc, const double* zb3c, const double* signs, const double* sgnf,
                             const double* brw, double* bdru) {
  int64_t n = nl * ne;
  for (int64_t i = IDX; i < n; i += STRIDE) {
    int64_t k = i / ne, e = i - k * ne;
    double s = 0.0;
    for (int side = 0; side < 2; side++) {
      int64_t c = coe[2 * e + side];
      int j = -1;
      for (int q = 0; q < noe[c]; q++) if (eoc[c * maxe + q] == e) { j = q; break; }
      if (j < 0) continue;
      for (int t = 0; t < 2; t++) {
        int64_t kk = k + t;            // interface: t = 0 uses fzm[k] dru[k]; t = 1 uses fzp[k+1] dru[k]
        if (kk < 1 || kk > nl - 1) continue;
        double coef = t == 0 ? fzm[kk] : fzp[kk];
        int64_t m = (kk * nc + c) * maxe + j;
        double iz = interp(zz, fzm, fzp, kk, c, nc);
        s -= signs[c * maxe + j] * (zbc[m] + sgnf[m] * zb3c[m]) * coef * iz * brw[kk * nc + c];
      }
    }
    bdru[i] += s;
  }
}
// per (k, e): bar_du += 0.5 (rz1 + rz2) bar_dru
__global__ void t0_ru_T_edge(int64_t nl, int64_t ne, int64_t nc, const int* coe, const double* rz, const double* bdru, double* bdu) {
  int64_t n = nl * ne;
  for (int64_t i = IDX; i < n; i += STRIDE) {
    int64_t k = i / ne, e = i - k * ne;
    bdu[i] += 0.5 * (rz[k * nc + coe[2 * e]] + rz[k * nc + coe[2 * e + 1]]) * bdru[i];
  }
}
// per (k, c): bar_drz += sum over the cell's edges 0.5 u bar_dru (edges in edgesOnCell order)
__global__ void t0_ru_T_cell(int64_t nl, int64_t nc, int64_t ne, int maxe, const int* eoc, const int* noe,
                             const double* u, const double* bdru, double* bdrz) {
  int64_t n = nl * nc;
  for (int64_t i = IDX; i < n; i += STRIDE) {
    int64_t k = i / nc, c = i - k * nc;
    double s = 0.0;
    for (int j = 0; j < noe[c]; j++) { int64_t e = eoc[c * maxe + j]; s += 0.5 * u[k * ne + e] * bdru[k * ne + e]; }
    bdrz[i] += s;
  }
}
// per (k, c): the thermodynamic transpose. Inputs are the summed copy cotangents (bthm, brz, brp, brtp, brth, bex, bpp)
// plus bdrz (already holding the ru / rw contributions). Outputs ADD into bdth, bdqv, bdrho.
__global__ void t0_cell_T(int64_t nl, int64_t nc, double rvord, double rgas, double rcv,
    const double* thm, const double* qv, const double* rp, const double* rtp, const double* ex,
    const double* zz, const double* rbase, const double* rtbase,
    const double* bthm, const double* brz_in, const double* brp, const double* brtp, const double* brth, const double* bex, const double* bpp,
    double* bdth, double* bdqv, double* bdrho) {
  int64_t n = nl * nc;
  for (int64_t i = IDX; i < n; i += STRIDE) {
    double a = 1.0 + rvord * qv[i], th = thm[i] / a;
    double rth = rtp[i] + rtbase[i];
    double b_ex = bex[i] + zz[i] * rgas * (rtp[i] + rtbase[i]) * bpp[i];
    double b_rtp = brtp[i] + brth[i] + zz[i] * rgas * ex[i] * bpp[i] + rcv * ex[i] / rth * b_ex;
    double b_thm = bthm[i] + (rp[i] + rbase[i]) * b_rtp;
    double b_rz = brz_in[i] + brp[i] + thm[i] * b_rtp;
    bdrho[i] += b_rz / zz[i];
    bdth[i] += a * b_thm;
    bdqv[i] += th * rvord * b_thm;
  }
}

// ---- Ts / Ts^T --------------------------------------------------------------------------------------------------
// Ts: (t_thm, t_qv) -> dth;  t_rz -> drho = t_rz zz. thm, qv: float32 primal words of boundary s.
__global__ void ts_cell(int64_t n, double rvord, const float* thm, const float* qv, const double* zz,
                        const double* t_thm, const double* t_qv, const double* t_rz, double* dth, double* drho) {
  for (int64_t i = IDX; i < n; i += STRIDE) {
    double a = 1.0 + rvord * (double)qv[i], th = (double)thm[i] / a;
    dth[i] = (t_thm[i] - th * rvord * t_qv[i]) / a;
    drho[i] = t_rz[i] * zz[i];
  }
}
// Ts^T: g_th, g_rho -> bar_thm, bar_qv (added to the g_qv already there), bar_rz
__global__ void ts_cell_T(int64_t n, double rvord, const float* thm, const float* qv, const double* zz,
                          const double* g_th, const double* g_rho, double* b_thm, double* b_qv, double* b_rz) {
  for (int64_t i = IDX; i < n; i += STRIDE) {
    double a = 1.0 + rvord * (double)qv[i], th = (double)thm[i] / a;
    b_thm[i] = g_th[i] / a;
    b_qv[i] += -g_th[i] * th * rvord / a;
    b_rz[i] = g_rho[i] * zz[i];
  }
}
// sign of the primal terrain flux per (k, c, slot): fzm[k] ru[k, e] + fzp[k] ru[k-1, e], ru float32 primal [nl][ne]
__global__ void t_flux_sign(int64_t nl, int64_t nc, int64_t ne, int maxe, const int* eoc, const int* noe,
                            const float* fzm32, const float* fzp32, const float* ru, double* sgnf) {
  int64_t n = (nl + 1) * nc;
  for (int64_t i = IDX; i < n; i += STRIDE) {
    int64_t k = i / nc, c = i - k * nc;
    for (int j = 0; j < maxe; j++) {
      double s = 0.0;
      if (k >= 1 && k <= nl - 1 && j < noe[c]) {
        int64_t e = eoc[c * maxe + j];
        float f = fzm32[k] * ru[k * ne + e] + fzp32[k] * ru[(k - 1) * ne + e];   // the forward's float32 expression
        s = copysignf(1.0f, f);
      }
      sgnf[i * maxe + j] = s;
    }
  }
}

}  // extern "C"
