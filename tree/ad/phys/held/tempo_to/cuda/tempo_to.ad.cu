// SPDX-License-Identifier: Apache-2.0
// Held-kernel lane tempo_to (ad/held-tempo_to, private autodiff program): tangent and adjoint of the hrrrv5 TEMPO
// launch tempo_to, written by hand from the forward CUDA source
//   gpuwm-hex 16251589 tree/src/hexcore/mpasphys/rrfs/tempo_v303.py:4485-4513 (pkg18 module 477158923b00908c),
//   RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_interface.F:1195-1270,1354-1456,1572-1616:
// the TEMPO carriers back to the MPAS pool (copies), theta_m, the dry-mass diagnostics (exner, pressure_p,
// rtheta_p, dtheta_dt_mp, rt_diabatic_tend), the microphysics tendencies, the surface-pressure diagnostic and the
// composite-reflectivity running maximum.
//
// ABI (ad.entry/v1, spec tempo_to.spec.json): one thread per column, the forward's own launch geometry.
//   pm, pmesh, pd, pf     : primal pointer tables mirrored onto PRE-LAUNCH copies of every target (deriv.rs
//                           `primal` + `indirect.mirror`, added on this branch); null entries stay null. The
//                           tendency table t needs no primal: every t word is linear in its own old value.
//   gm, gmesh, gd, gt, gf : tangent (TL) or adjoint (AD) pointer tables mirrored onto the sweep shadows.
//   pcmax/pcomp, gcmax/gcomp : composite reflectivity running max and its source (plain float32 arrays).
//   fail                  : refusal slots (0: mpas_powf base <= 0 at :4497, 1: dz1 + dz2 == 0 at :4510-4511).
// Tangent: the shadows hold the pre-launch tangent on entry and the post-launch tangent on exit (in place).
// Adjoint: the shadows hold the post-launch cotangent on entry and the pre-launch cotangent on exit (in place).
//
// Precision (as the live P3/P4/P5 schemes and the e2e entries): the primal is recomputed in float32 in the
// forward's own order of operations (strict build, --fmad=false), so every branch is the executed one; every
// derivative coefficient and product is float64; derivative storage is the sweep's (float32 in the window), one
// rounding per stored word. Adding +0.0 before each store keeps the zero direction free of negative zeros.
//
// Frozen-coefficient (executed-branch) treatment of the limiter-like selections, as COMMON-AD's max rule:
//   * tmax(0, m0) in rthmpten (:4503): slope 1 where !(0 > m0) (the forward picks m0 there, ties included), else 0;
//   * compmax = tmax(compmax, comp) (:4508): the derivative follows the operand the forward selected
//     (compmax where compmax > comp, comp otherwise: ties go to comp, as in the forward);
//   * mp_top_level / aero / hail are integer switches: inactive, taken from the launch.
// mpas_powf(X, rcv) (:4497): d/dX = rcv * X^rcv / X with X^rcv in float64 (pow of the float32 base); the
// forward's own float32 power is not re-entered (the two differ below float32 rounding of the result).
#include <math.h>

#define G(p, i) ((p) ? (double)(p)[i] : 0.0)

__device__ __forceinline__ void st(float* p, long long i, double v) { if (p) p[i] = (float)(v + 0.0); }
__device__ __forceinline__ void acc(float* p, long long i, double v) { if (p) p[i] = (float)((double)p[i] + v + 0.0); }
__device__ __forceinline__ float tmaxf_(float a, float b) { return a > b ? a : b; }
__device__ __forceinline__ void refuse(unsigned int* fail, int slot) { if (fail) atomicExch(fail + slot, 1u); }

__constant__ int TT_MAPS[13] = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

// Primal of one level, recomputed in the forward's float32 order (k < mp_top_level).
struct Lev {
  float m13o, m0o, f0, f4, m14, d4, d3, zz, m13n, d5, S, X, crp;
  double d0, dd0, A, B, P;
};
__device__ __forceinline__ void lev(Lev& L, const float* const* pm, const float* const* pmesh, const float* const* pd,
                                    const float* const* pf, long long x, float rvord, float rd, float p0, float rcv,
                                    unsigned int* fail) {
  L.m13o = pm[13][x]; L.m0o = pm[0][x];
  L.f0 = pf[0][x]; L.f4 = pf[4][x]; L.m14 = pm[14][x]; L.d4 = pd[4][x]; L.d3 = pd[3][x]; L.zz = pmesh[0][x];
  L.A = 1.0 + (double)rvord * (double)L.m0o;                         // d6tmp = m13o / (1 + rvord m0o)
  L.m13n = L.f0 * (1.0f + rvord * L.f4);                             // :4495
  L.B = 1.0 + (double)rvord * (double)L.f4;                          // m0 is f4 after :4494
  L.d5 = L.m14 * L.m13n - L.d4;                                      // :4497
  L.S = L.d5 + L.d4;
  L.crp = rd / p0;
  L.X = L.zz * L.crp * L.S;
  if (!(L.X > 0.0f)) { refuse(fail, 0); L.d0 = 0.0; L.dd0 = 0.0; }
  else { L.d0 = pow((double)L.X, (double)rcv); L.dd0 = (double)rcv * L.d0 / (double)L.X; }
  L.P = L.d0 * (double)L.d5 + (L.d0 - (double)L.d3) * (double)L.d4;   // :4498 bracket
}

// Column part (:4509-4512), primal.
struct Col { float dz1, dz2, W, R, E, m14a, zza, f4a, m14b, zzb, f4b; };
__device__ __forceinline__ void col(Col& C, const float* const* pm, const float* const* pmesh, const float* const* pf,
                                    int nc, int c, unsigned int* fail) {
  const float* z = pmesh[1];
  C.dz1 = z[nc + c] - z[c]; C.dz2 = z[2 * nc + c] - z[nc + c];
  C.m14a = pm[14][c]; C.zza = pmesh[0][c]; C.f4a = pf[4][c];
  C.m14b = pm[14][nc + c]; C.zzb = pmesh[0][nc + c]; C.f4b = pf[4][nc + c];
  const float rho1 = C.m14a * C.zza * (1.0f + C.f4a), rho2 = C.m14b * C.zzb * (1.0f + C.f4b);
  C.W = C.dz1 + C.dz2; C.R = rho2 - rho1;
  if (C.W == 0.0f) refuse(fail, 1);
  C.E = rho1 - 0.5f * (rho2 - rho1) * C.dz1 / (C.dz1 + C.dz2);
}

extern "C" __global__ void tempo_to__tl(
    const float* const* pm, const float* const* pmesh, const float* const* pd, const float* const* pf,
    float* const* gm, float* const* gmesh, float* const* gd, float* const* gt, float* const* gf,
    int nz, int nc, int aero, int hail, float dt, float rv, float rd, float rvord, float p0, float rcv, float grav,
    int mpt, const float* pcmax, const float* pcomp, float* gcmax, float* gcomp, unsigned int* fail) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= nc) return;
  const double DT = (double)dt;
  const float cr = rv / rd;
  for (int k = 0; k < nz; ++k) {
    const long long x = (long long)k * nc + c;
    if (k < mpt) {
      Lev L; lev(L, pm, pmesh, pd, pf, x, rvord, rd, p0, rcv, fail);
      const double tm13o = G(gm[13], x), tm0o = G(gm[0], x);
      const double t_d7tmp = tm13o;                                                         // d7 = m13 (old)
      const double t_d6tmp = tm13o / L.A - (double)L.m13o * rvord * tm0o / (L.A * L.A);       // d6 = m13/(1+rvord m0)
      const double tf0 = G(gf[0], x), tf4 = G(gf[4], x), tf5 = G(gf[5], x), tf6 = G(gf[6], x);
      st(gm[0], x, tf4); st(gm[1], x, tf5); st(gm[2], x, tf6);                              // m[0..2] = f[4..6]
      const double tm13n = tf0 * (1.0 + (double)rvord * L.f4) + (double)L.f0 * rvord * tf4;  // m13 = f0 (1 + rvord f4)
      st(gm[13], x, tm13n);
      st(gd[7], x, (G(gm[13], x) - t_d7tmp) / DT);                                         // d7 = (m13 - d7)/dt
      const double tq = G(gm[13], x) / L.B - (double)L.m13n * rvord * G(gm[0], x) / (L.B * L.B);
      st(gd[6], x, (tq - t_d6tmp) / DT);                                                   // d6 = (m13/(1+rvord m0) - d6)/dt
      const double tm14 = G(gm[14], x), td4 = G(gd[4], x);
      st(gd[5], x, tm14 * (double)L.m13n + (double)L.m14 * G(gm[13], x) - td4);           // d5 = m14 m13 - d4
      const double td5 = G(gd[5], x), tS = td5 + G(gd[4], x), tzz = G(gmesh[0], x);
      const double tX = (double)L.crp * (tzz * (double)L.S + (double)L.zz * tS);
      st(gd[0], x, L.dd0 * tX);                                                            // d0 = X^rcv
      const double td0 = G(gd[0], x), td3 = G(gd[3], x);
      const double tP = td0 * (double)L.d5 + L.d0 * td5 + (td0 - td3) * (double)L.d4 + (L.d0 - (double)L.d3) * G(gd[4], x);
      st(gd[2], x, (double)rd * (G(gmesh[0], x) * L.P + (double)L.zz * tP));                // d2 = zz rd (d0 d5 + (d0-d3) d4)
    } else {
      st(gm[1], x, 0.0); st(gm[2], x, 0.0); st(gd[7], x, 0.0);
    }
    for (int j = 3; j < 13; ++j) if (j < 8 || (j < 11 ? aero : hail)) st(gm[j], x, G(gf[TT_MAPS[j]], x));
    if (k < mpt) {
      const float m0 = pf[4][x];                       // m[0] after :4494
      const float m13 = pf[0][x] * (1.0f + rvord * pf[4][x]);
      const float mx = tmaxf_(0.0f, m0);
      const double H = (0.0f > m0) ? 0.0 : 1.0;
      const double C = 1.0 + (double)cr * (double)mx;
      const double tt = G(gm[13], x) / C - (double)m13 * cr * H * G(gm[0], x) / (C * C);
      st(gt[0], x, (tt - G(gt[0], x)) / DT);
      for (int j = 0; j < 6; ++j) st(gt[j + 1], x, (G(gm[j], x) - G(gt[j + 1], x)) / DT);
    } else {
      for (int j = 0; j < 7; ++j) st(gt[j], x, 0.0);
    }
    for (int j = 6; j < 8; ++j) st(gt[j + 1], x, (G(gm[j], x) - G(gt[j + 1], x)) / DT);
    if (aero) {
      st(gt[9], x, (G(gm[8], x) - G(gt[9], x)) / DT);
      st(gt[10], x, (G(gm[10], x) - G(gt[10], x)) / DT);
      st(gt[11], x, (G(gm[9], x) - G(gt[11], x)) / DT);
    }
  }
  if (pcmax) { if (!(pcmax[c] > pcomp[c])) st(gcmax, c, G(gcomp, c)); }       // tmax(compmax, comp), executed operand
  Col K; col(K, pm, pmesh, pf, nc, c, fail);
  const double tdz1 = G(gmesh[1], nc + c) - G(gmesh[1], c), tdz2 = G(gmesh[1], 2 * nc + c) - G(gmesh[1], nc + c);
  const double trho1 = (G(gm[14], c) * K.zza + (double)K.m14a * G(gmesh[0], c)) * (1.0 + (double)K.f4a) + (double)K.m14a * K.zza * G(gf[4], c);
  const double trho2 = (G(gm[14], nc + c) * K.zzb + (double)K.m14b * G(gmesh[0], nc + c)) * (1.0 + (double)K.f4b) + (double)K.m14b * K.zzb * G(gf[4], nc + c);
  const double W = K.W, R = K.R, dz1 = K.dz1;
  const double tQ = 0.5 * ((trho2 - trho1) * dz1 / W + R * tdz1 / W - R * dz1 * (tdz1 + tdz2) / (W * W));
  const double tE = trho1 - tQ;
  const double td8a = 0.5 * (double)grav * (tdz1 * (double)K.E + dz1 * tE);
  const double td8o = G(gd[8], c);                                                       // d9 = d8 (old)
  const double td8 = (td8a + G(gd[2], c)) + G(gd[1], c);                                 // d8 = (d8 + d2) + d1
  st(gd[8], c, td8);
  st(gd[9], c, (G(gd[8], c) - td8o) / DT);                                               // d9 = (d8 - d9)/dt
}

extern "C" __global__ void tempo_to__ad(
    const float* const* pm, const float* const* pmesh, const float* const* pd, const float* const* pf,
    float* const* gm, float* const* gmesh, float* const* gd, float* const* gt, float* const* gf,
    int nz, int nc, int aero, int hail, float dt, float rv, float rd, float rvord, float p0, float rcv, float grav,
    int mpt, const float* pcmax, const float* pcomp, float* gcmax, float* gcomp, unsigned int* fail) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= nc) return;
  const double DT = (double)dt;
  const float cr = rv / rd;
  // ---- column part, reversed (it ran last)
  {
    Col K; col(K, pm, pmesh, pf, nc, c, fail);
    const double W = K.W, R = K.R, dz1 = K.dz1;
    const double a9 = G(gd[9], c);
    const double a8 = G(gd[8], c) + a9 / DT;                 // d9 = (d8 - d9tmp)/dt
    const double a9tmp = (0.0 - a9) / DT;
    acc(gd[2], c, a8); acc(gd[1], c, a8);                     // d8 = (d8a + d2) + d1
    const double a8a = a8;                                    // d8a = 0.5 grav dz1 E
    double adz1 = 0.5 * (double)grav * (double)K.E * a8a;
    const double aE = 0.5 * (double)grav * dz1 * a8a;
    double arho1 = aE; const double aQ = 0.0 - aE;            // E = rho1 - Q, Q = 0.5 R dz1 / W
    const double aR = aQ * 0.5 * dz1 / W;
    const double arho2 = aR; arho1 -= aR;
    adz1 += aQ * (0.5 * R / W - 0.5 * R * dz1 / (W * W));
    const double adz2 = (0.0 - aQ) * 0.5 * R * dz1 / (W * W);
    acc(gm[14], c, arho1 * K.zza * (1.0 + (double)K.f4a)); acc(gmesh[0], c, arho1 * K.m14a * (1.0 + (double)K.f4a));
    acc(gf[4], c, arho1 * K.m14a * K.zza);
    acc(gm[14], nc + c, arho2 * K.zzb * (1.0 + (double)K.f4b)); acc(gmesh[0], nc + c, arho2 * K.m14b * (1.0 + (double)K.f4b));
    acc(gf[4], nc + c, arho2 * K.m14b * K.zzb);
    acc(gmesh[1], nc + c, adz1); acc(gmesh[1], c, 0.0 - adz1);
    acc(gmesh[1], 2 * nc + c, adz2); acc(gmesh[1], nc + c, 0.0 - adz2);
    st(gd[8], c, a9tmp);                                      // d9tmp = d8 (old); d8 itself was overwritten
    st(gd[9], c, 0.0);                                        // d9's old value is never read
    if (pcmax) { if (!(pcmax[c] > pcomp[c])) { acc(gcomp, c, G(gcmax, c)); st(gcmax, c, 0.0); } }
  }
  // ---- levels, last to first
  for (int k = nz - 1; k >= 0; --k) {
    const long long x = (long long)k * nc + c;
    if (aero) {
      double a;
      a = G(gt[11], x); acc(gm[9], x, a / DT); st(gt[11], x, (0.0 - a) / DT);
      a = G(gt[10], x); acc(gm[10], x, a / DT); st(gt[10], x, (0.0 - a) / DT);
      a = G(gt[9], x); acc(gm[8], x, a / DT); st(gt[9], x, (0.0 - a) / DT);
    }
    for (int j = 7; j >= 6; --j) { const double a = G(gt[j + 1], x); acc(gm[j], x, a / DT); st(gt[j + 1], x, (0.0 - a) / DT); }
    if (k < mpt) {
      for (int j = 5; j >= 0; --j) { const double a = G(gt[j + 1], x); acc(gm[j], x, a / DT); st(gt[j + 1], x, (0.0 - a) / DT); }
      const float m0 = pf[4][x];
      const float m13 = pf[0][x] * (1.0f + rvord * pf[4][x]);
      const float mx = tmaxf_(0.0f, m0);
      const double H = (0.0f > m0) ? 0.0 : 1.0;
      const double C = 1.0 + (double)cr * (double)mx;
      const double a = G(gt[0], x), g = a / DT;
      st(gt[0], x, (0.0 - a) / DT);
      acc(gm[13], x, g / C);
      if (H != 0.0) acc(gm[0], x, (0.0 - g) * (double)m13 * cr / (C * C));
    } else {
      for (int j = 0; j < 7; ++j) st(gt[j], x, 0.0);
    }
    for (int j = 12; j >= 3; --j) if (j < 8 || (j < 11 ? aero : hail)) { acc(gf[TT_MAPS[j]], x, G(gm[j], x)); st(gm[j], x, 0.0); }
    if (k < mpt) {
      Lev L; lev(L, pm, pmesh, pd, pf, x, rvord, rd, p0, rcv, fail);
      double a;
      // d2 = zz rd P
      a = G(gd[2], x); st(gd[2], x, 0.0);
      acc(gmesh[0], x, a * (double)rd * L.P);
      const double aP = a * (double)L.zz * (double)rd;
      acc(gd[0], x, aP * ((double)L.d5 + (double)L.d4));
      acc(gd[5], x, aP * L.d0);
      acc(gd[3], x, (0.0 - aP) * (double)L.d4);
      acc(gd[4], x, aP * (L.d0 - (double)L.d3));
      // d0 = X^rcv, X = zz crp S, S = d5 + d4
      a = G(gd[0], x); st(gd[0], x, 0.0);
      const double aX = a * L.dd0;
      acc(gmesh[0], x, aX * (double)L.crp * (double)L.S);
      const double aS = aX * (double)L.zz * (double)L.crp;
      acc(gd[5], x, aS); acc(gd[4], x, aS);
      // d5 = m14 m13 - d4
      a = G(gd[5], x); st(gd[5], x, 0.0);
      acc(gm[14], x, a * (double)L.m13n); acc(gm[13], x, a * (double)L.m14); acc(gd[4], x, 0.0 - a);
      // d6 = (m13/(1+rvord m0) - d6tmp)/dt
      a = G(gd[6], x); st(gd[6], x, (0.0 - a) / DT);
      { const double g = a / DT; acc(gm[13], x, g / L.B); acc(gm[0], x, (0.0 - g) * (double)L.m13n * rvord / (L.B * L.B)); }
      // d7 = (m13 - d7tmp)/dt
      a = G(gd[7], x); st(gd[7], x, (0.0 - a) / DT); acc(gm[13], x, a / DT);
      // m13 = f0 (1 + rvord f4)
      a = G(gm[13], x); st(gm[13], x, 0.0);
      acc(gf[0], x, a * (1.0 + (double)rvord * L.f4)); acc(gf[4], x, a * (double)L.f0 * rvord);
      // m[2..0] = f[6..4]
      acc(gf[6], x, G(gm[2], x)); st(gm[2], x, 0.0);
      acc(gf[5], x, G(gm[1], x)); st(gm[1], x, 0.0);
      acc(gf[4], x, G(gm[0], x)); st(gm[0], x, 0.0);
      // d6tmp = m13o/(1+rvord m0o); d6's old value is never read
      a = G(gd[6], x); st(gd[6], x, 0.0);
      acc(gm[13], x, a / L.A); acc(gm[0], x, (0.0 - a) * (double)L.m13o * rvord / (L.A * L.A));
      // d7tmp = m13o; d7's old value is never read
      a = G(gd[7], x); st(gd[7], x, 0.0); acc(gm[13], x, a);
    } else {
      st(gm[1], x, 0.0); st(gm[2], x, 0.0); st(gd[7], x, 0.0);
    }
  }
}
