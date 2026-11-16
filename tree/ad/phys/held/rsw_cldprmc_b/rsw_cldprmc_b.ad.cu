// SPDX-License-Identifier: Apache-2.0
// Held-kernel lane rsw_cldprmc_b (ad/held-rsw_cldprmc_b, private autodiff program): tangent and adjoint of the
// hrrrv5 RRTMG SW McICA cloud-optics launch rsw_cldprmc_b, written by hand from the forward CUDA source
//   hexcore/mpasphys/rrtmg_sw_v842.py:2188 (captured translation unit 3c499c7ba1022106, lines 2264-2475 and
//   2507-2538; image c22c485e0c80715f), a port of MPAS-A v8.4.2 module_ra_rrtmg_sw.F:2088-2509 (cldprmc_sw).
// Per (col, lay, ig) the forward first copies taormc = taucmc_in; where the McICA subcolumn is cloudy (and
// inflag >= 2) it overwrites taormc, taucmc, ssacmc and asmcmc from the water paths (clwp, ciwp, cswp) and the
// per-layer effective radii (relq, reic, resn) through table interpolations and the delta-scaling of each species:
//   tau0_k = P_k ext_k(r_k); a_k = 1 - f_k w_k; ssa'_k = w_k (1 - f_k) / a_k; tau_k = a_k tau0_k;
//   scat_k = ssa'_k tau_k; taormc = sum tau0_k; taucmc = sum tau_k; ssacmc = sum scat_k / taucmc;
//   asmcmc = sum scat_k (g_k - f_k) / (1 - f_k) / sum scat_k     (k = liq, ice, and snow when iceflag == 5)
// with f_k = g_k^2 (liquid, iceflag 1/2) or min(fdelta_k + 0.5 / w_k, g_k) (iceflag >= 3, snow).
//
// ABI (ad.entry/v1, spec rsw_cldprmc_b.spec.json). Geometry: one thread per (col, lay), ceil(ncol*nl/128) x 128,
// NOT the forward's (col, lay, ig) geometry: the adjoint of the per-layer radii reicmc/relqmc/resnmc sums over the
// 112 g-points of one (col, lay), and doing that sum in one thread in a fixed ig order keeps the reverse
// deterministic (bitwise across runs) with no atomics. Every forward word is independent of every other (no word
// reads a word another thread writes; the early returns are fatal paths, refused below), so the regrouping is exact.
//   primal arrays (pre-launch copies): tab ngb cldfmc ciwpmc clwpmc cswpmc reicmc relqmc resnmc taucmc (taucmc's
//                 pre-launch value is a gate operand and the non-cloudy taormc).
//   TL: t_ciwp t_clwp t_cswp t_reic t_relq t_resn are read (io `in`); t_taor t_tau t_ssa t_asm are the pre-launch
//       tangents on entry and the post-launch tangents on exit (io `inout`). Non-cloudy: t_taor = t_tau (the
//       copy), the other three pass through. Cloudy: all four are written from the 4 x 6 executed Jacobian.
//   AD: the transpose. Non-cloudy: b_tau += b_taor, b_taor consumed, b_ssa and b_asm pass through. Cloudy: the four
//       output cotangents are consumed (the pre-launch taucmc/ssacmc/asmcmc enter only through the gate, so their
//       derivative there is 0) and b_clwp b_ciwp b_cswp (per word) and b_relq b_reic b_resn (per layer, summed over
//       the 112 g-points in one thread) accumulate (io `accumulate`).
//   fail: refusal slots, one per fatal path of the forward (it sets err_flag and returns; the host raises), and
//       one for configurations whose forward has no derivative here (see slot 4).
//
// Precision (as the live P3/P4/P5 schemes and the e2e entries): the primal is recomputed in float32 in the
// forward's own order of operations with the same add.rn/sub.rn/mul.rn and __fdiv_rn (strict build,
// --fmad=false --ftz=false --prec-div=true), so every branch, index and coefficient is the executed one; every
// derivative coefficient and product is float64; derivative storage is the sweep's (float32 in the window), one
// rounding per stored word; +0.0 is added before each store so the zero direction has no negative zeros.
//
// Frozen-coefficient (executed-branch) treatment, per forward line (captured .cu 3c499c7b):
//   * 2286 gate cldfmc >= 1e-20 && (cwp >= 1e-20 || taucmc_in >= 1e-20): trajectory value. Not cloudy (and
//     inflag < 0, 2304) -> only the taormc copy (2282). cldfmc is a McICA 0/1 step: zero derivative by design.
//   * 2317 ciwp + cswp == 0 / 2372 cswp > 0 / 2395 clwp == 0 zero-path shortcuts: frozen, the species'
//     coefficients are 0 there (one-sided: a perturbation that creates condensate is not seen).
//   * 2338-2341, 2353-2356, 2375-2378, 2400-2403 table index (int)factor and the clamps 43->42, 46->45, 0->1,
//     58->57: index frozen; slope = the forward's float32 table difference SU(T[i], T[i-1]) times d fint/d r (1/3
//     ice and snow, 1 liquid); the clamped ends are linear extrapolations and keep their slope.
//   * 2331 gice >= 1 -> 1 - 1e-6 (iceflag 1), 2367 / 2389 forw > g -> forw = g, 2409 fint < 0 && ssa > 1 ->
//     ssa = table node: limiter-like clamps, frozen on the executed side (the clamped value's own derivative:
//     constant -> 0, forw = g -> d forw = d g).
//   * 2451-2453 taucmc == 0 / scatice == 0 / scatsno == 0 -> cldmin: frozen constant (derivative 0 on that word).
//   * 2308 inflag == 1, 2333/2337/2349/2352/2365/2368 ice, 2374/2387/2390 snow, 2399/2412 liquid range and
//     optics checks: fatal paths (err_flag, host raises) -> refusal slots 0-3.
//   * inflag == 0 (2289-2301, delta-scaling of supplied optics, an fsfcmc path), iceflag 0 with ice and liqflag != 1
//     with liquid (the forward reads uninitialised coefficients there): no derivative written -> refusal slot 4.
//     hrrrv5 runs inflag 5, iceflag 5, liqflag 1.
//   * inflag / iceflag / liqflag / ngb are integer switches (launch values); tab is a constant table (inactive).

#define NGPT 112
#define T_EXTLIQ1 87366
#define T_SSALIQ1 88178
#define T_ASYLIQ1 88990
#define T_EXTICE2 89802
#define T_SSAICE2 90404
#define T_ASYICE2 91006
#define T_EXTICE3 91608
#define T_SSAICE3 92252
#define T_ASYICE3 92896
#define T_FDLICE3 93540

// The forward's arithmetic (captured .cu 1265-1300): binary32 round-to-nearest, gradual underflow. The forward's
// NaN-payload rewrite only changes NaN bit patterns, which no branch below reads, so it is not repeated.
static __device__ __forceinline__ float AD(float a, float b) { float r; asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
static __device__ __forceinline__ float SU(float a, float b) { float r; asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
static __device__ __forceinline__ float MU(float a, float b) { float r; asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
static __device__ __forceinline__ float DV(float a, float b) { return __fdiv_rn(a, b); }
static __device__ __forceinline__ bool EQ0(float x) { return (__float_as_uint(x) & 0x7fffffffu) == 0u; }

__device__ __forceinline__ void refuse(unsigned int* fail, int slot) { if (fail) atomicExch(fail + slot, 1u); }

// One species' executed optics: float32 values and float64 derivatives in its radius.
struct Sp { float ext, w, g, f; double dext, dw, dg, df; };

__device__ __forceinline__ void sp_zero(Sp& s) { s.ext = s.w = s.g = s.f = 0.0f; s.dext = s.dw = s.dg = s.df = 0.0; }

// Fu tables (iceflag >= 3 ice, and snow), captured .cu 2352-2369 / 2374-2391. Returns false on a fatal path.
__device__ __forceinline__ bool fu(Sp& s, float r, const float* __restrict__ tab, int c) {
  if (r < 5.0f || r > 140.0f) return false;                                                // 2352 / 2374
  const float factor = DV(SU(r, 2.0f), 3.0f);
  int index = (int)factor;
  if (index == 46) index = 45;
  const float fint = SU(factor, (float)index);
  const float* e3 = tab + T_EXTICE3 + (size_t)46 * c;
  const float* s3 = tab + T_SSAICE3 + (size_t)46 * c;
  const float* a3 = tab + T_ASYICE3 + (size_t)46 * c;
  const float* f3 = tab + T_FDLICE3 + (size_t)46 * c;
  const float De = SU(e3[index], e3[index - 1]), Ds = SU(s3[index], s3[index - 1]);
  const float Da = SU(a3[index], a3[index - 1]), Df = SU(f3[index], f3[index - 1]);
  s.ext = AD(e3[index - 1], MU(fint, De));
  s.w = AD(s3[index - 1], MU(fint, Ds));
  s.g = AD(a3[index - 1], MU(fint, Da));
  const float fdelta = AD(f3[index - 1], MU(fint, Df));
  if (fdelta < 0.0f || fdelta > 1.0f) return false;                                       // 2365 / 2387
  s.dext = (double)De / 3.0; s.dw = (double)Ds / 3.0; s.dg = (double)Da / 3.0;
  const double dfd = (double)Df / 3.0;
  s.f = AD(fdelta, DV(0.5f, s.w));                                                         // 2366 / 2388
  s.df = dfd - 0.5 / ((double)s.w * (double)s.w) * s.dw;
  if (s.f > s.g) { s.f = s.g; s.df = s.dg; }                  // 2367 / 2389: frozen limiter, forw = g
  if (s.ext < 0.0f || s.w > 1.0f || s.w < 0.0f || s.g > 1.0f || s.g < 0.0f) return false; // 2368 / 2390
  return true;
}

// One species' delta-scaled contributions and their float64 derivatives in (P, r).
struct Ct { double t0P, t0r, tP, tr, sP, sr; };   // d tau0, d tau, d scat  / d P, d r

__device__ __forceinline__ void contrib(Ct& o, const Sp& s, float P, float& tau0f, float& tauf, float& scatf) {
  // forward float32 (captured .cu 2432-2447): tau0 = P ext; ssa' = w (1-f) / (1 - f w); tau = (1 - f w) tau0
  tau0f = MU(P, s.ext);
  const float a = SU(1.0f, MU(s.f, s.w));
  const float ssap = DV(MU(s.w, SU(1.0f, s.f)), a);
  tauf = MU(a, tau0f);
  scatf = MU(ssap, tauf);
  const double A = a, Wp = ssap, T0 = tau0f, T = tauf, w = s.w, f = s.f, p = P;
  const double da = -(s.df * w + f * s.dw);
  const double num = w * (1.0 - f), dnum = s.dw * (1.0 - f) - w * s.df;
  const double dssap = (dnum * A - num * da) / (A * A);
  o.t0P = s.ext; o.t0r = p * s.dext;
  o.tP = A * (double)s.ext; o.tr = da * T0 + A * p * s.dext;
  o.sP = Wp * o.tP; o.sr = dssap * T + Wp * o.tr;
}

// The executed 4 x 6 Jacobian of one word. Outputs: 0 taormc, 1 taucmc, 2 ssacmc, 3 asmcmc. Inputs: 0 clwp,
// 1 ciwp, 2 cswp, 3 relq, 4 reic, 5 resn. Returns 1 where the forward overwrites the four outputs, 0 where it only
// copies taormc = taucmc_in (non-cloudy, inflag < 0, and every refused word).
__device__ __forceinline__ int optics(
    double J[4][6], int inflag, int iceflag, int liqflag, long long w, long long cl, int ig,
    const float* __restrict__ tab, const int* __restrict__ ngb, const float* __restrict__ cldfmc,
    const float* __restrict__ ciwpmc, const float* __restrict__ clwpmc, const float* __restrict__ cswpmc,
    const float* __restrict__ reicmc, const float* __restrict__ relqmc, const float* __restrict__ resnmc,
    const float* __restrict__ taucmc, unsigned int* fail) {
  const float cldmin = 1.0e-20f;
  const float ciwp = ciwpmc[w], clwp = clwpmc[w], cswp = cswpmc[w];
  const float cwp = AD(AD(ciwp, clwp), cswp);                                              // 2284
  if (!(cldfmc[w] >= cldmin && (cwp >= cldmin || taucmc[w] >= cldmin))) return 0;          // 2286
  if (inflag == 0) { refuse(fail, 4); return 0; }                                          // 2289
  if (inflag < 0) return 0;                                                                // 2304
  if (inflag == 1) { refuse(fail, 0); return 0; }                                          // 2305
  const int c = ngb[ig] - 16;                                                              // 2315
  Sp sp[3];
  sp_zero(sp[0]); sp_zero(sp[1]); sp_zero(sp[2]);
  // ice (2312-2370)
  const float radice = reicmc[cl];
  if (EQ0(AD(ciwp, cswp))) {
    // 2317: all ice coefficients 0
  } else if (iceflag == 1) {
    const float upper[14] = {3250.f,4000.f,4650.f,5150.f,6150.f,7700.f,8050.f,12850.f,16000.f,22650.f,29000.f,38000.f,50000.f,2600.f};
    const float cb[5] = {1.e-5f,1.1e-4f,1.240e-2f,3.779e-2f,4.666e-1f};
    const float db[5] = {0.f,1.405e-5f,6.867e-4f,1.284e-3f,2.050e-5f};
    const float eb[5] = {.7661f,.7730f,.7865f,.8172f,.9595f};
    const float fb[5] = {5.851e-4f,5.665e-4f,7.204e-4f,7.463e-4f,1.076e-4f};
    const int icx = upper[c] > 14300.f ? 0 : upper[c] > 7700.f ? 1 : upper[c] > 5300.f ? 2 : upper[c] > 4000.f ? 3 : 4;
    Sp& s = sp[1];
    s.ext = AD(.003448f, DV(2.431f, radice)); s.dext = -2.431 / ((double)radice * (double)radice);
    s.w = SU(SU(1.f, cb[icx]), MU(db[icx], radice)); s.dw = -(double)db[icx];
    s.g = AD(eb[icx], MU(fb[icx], radice)); s.dg = fb[icx];
    if (s.g >= 1.f) { s.g = SU(1.f, 1.e-6f); s.dg = 0.0; }                                // 2331: frozen limiter
    s.f = MU(s.g, s.g); s.df = 2.0 * (double)s.g * s.dg;
    if (s.ext < 0.f || s.w > 1.f || s.w < 0.f || s.g > 1.f || s.g < 0.f) { refuse(fail, 1); return 0; }   // 2333
  } else if (iceflag == 2) {
    if (radice < 5.0f || radice > 131.0f) { refuse(fail, 1); return 0; }                  // 2337
    Sp& s = sp[1];
    const float factor = DV(SU(radice, 2.0f), 3.0f);
    int index = (int)factor;
    if (index == 43) index = 42;
    const float fint = SU(factor, (float)index);
    const float* e2 = tab + T_EXTICE2 + (size_t)43 * c;
    const float* s2 = tab + T_SSAICE2 + (size_t)43 * c;
    const float* a2 = tab + T_ASYICE2 + (size_t)43 * c;
    const float De = SU(e2[index], e2[index - 1]), Ds = SU(s2[index], s2[index - 1]), Da = SU(a2[index], a2[index - 1]);
    s.ext = AD(e2[index - 1], MU(fint, De)); s.w = AD(s2[index - 1], MU(fint, Ds)); s.g = AD(a2[index - 1], MU(fint, Da));
    s.dext = (double)De / 3.0; s.dw = (double)Ds / 3.0; s.dg = (double)Da / 3.0;
    s.f = MU(s.g, s.g); s.df = 2.0 * (double)s.g * s.dg;
    if (s.ext < 0.0f || s.w > 1.0f || s.w < 0.0f || s.g > 1.0f || s.g < 0.0f) { refuse(fail, 1); return 0; }   // 2349
  } else if (iceflag >= 3) {
    if (!fu(sp[1], radice, tab, c)) { refuse(fail, 1); return 0; }                        // 2351-2369
  } else {
    refuse(fail, 4); return 0;                       // iceflag 0 with ice: the forward reads uninitialised coefficients
  }
  // snow (2372-2392)
  if (cswp > 0.0f && iceflag == 5) {
    if (!fu(sp[2], resnmc[cl], tab, c)) { refuse(fail, 2); return 0; }
  }
  // liquid (2394-2414)
  if (EQ0(clwp)) {
    // 2395: all liquid coefficients 0
  } else if (liqflag == 1) {
    const float radliq = relqmc[cl];
    if (radliq < 1.5f || radliq > 60.0f) { refuse(fail, 3); return 0; }                   // 2399
    int index = (int)SU(radliq, 1.5f);
    if (index == 0) index = 1;
    if (index == 58) index = 57;
    const float fint = SU(SU(radliq, 1.5f), (float)index);
    const float* el = tab + T_EXTLIQ1 + (size_t)58 * c;
    const float* sl = tab + T_SSALIQ1 + (size_t)58 * c;
    const float* al = tab + T_ASYLIQ1 + (size_t)58 * c;
    Sp& s = sp[0];
    const float De = SU(el[index], el[index - 1]), Ds = SU(sl[index], sl[index - 1]), Da = SU(al[index], al[index - 1]);
    s.ext = AD(el[index - 1], MU(fint, De)); s.dext = De;
    s.w = AD(sl[index - 1], MU(fint, Ds)); s.dw = Ds;
    if (fint < 0.0f && s.w > 1.0f) { s.w = sl[index - 1]; s.dw = 0.0; }                     // 2409: frozen limiter
    s.g = AD(al[index - 1], MU(fint, Da)); s.dg = Da;
    s.f = MU(s.g, s.g); s.df = 2.0 * (double)s.g * s.dg;
    if (s.ext < 0.0f || s.w > 1.0f || s.w < 0.0f || s.g > 1.0f || s.g < 0.0f) { refuse(fail, 3); return 0; }   // 2412
  } else {
    refuse(fail, 4); return 0;                       // liqflag != 1 with liquid: uninitialised coefficients
  }
  // combination (2416-2474)
  const int ns = iceflag < 5 ? 2 : 3;
  const float P[3] = {clwp, ciwp, cswp};
  Ct ct[3]; float t0f[3], tf[3], sf[3];
  for (int k = 0; k < 3; ++k) {
    if (k < ns) contrib(ct[k], sp[k], P[k], t0f[k], tf[k], sf[k]);
    else { t0f[k] = tf[k] = sf[k] = 0.0f; ct[k].t0P = ct[k].t0r = ct[k].tP = ct[k].tr = ct[k].sP = ct[k].sr = 0.0; }
  }
  float tauc = ns == 2 ? AD(tf[0], tf[1]) : AD(AD(tf[0], tf[1]), tf[2]);                 // 2430 / 2448
  bool tclamp = false;
  if (EQ0(tauc)) { tauc = cldmin; tclamp = true; }                                         // 2451
  if (EQ0(sf[1])) { sf[1] = cldmin; ct[1].sP = ct[1].sr = 0.0; }                           // 2452
  if (EQ0(sf[2])) { sf[2] = cldmin; ct[2].sP = ct[2].sr = 0.0; }                           // 2453 (unused if ns == 2)
  // asymmetry terms: term_k = scat_k h_k, h_k = (g_k - f_k) / (1 - f_k)   (2460-2473)
  double S = 0.0, Tt = 0.0, dSP[3], dSr[3], dTP[3], dTr[3];
  for (int k = 0; k < 3; ++k) dSP[k] = dSr[k] = dTP[k] = dTr[k] = 0.0;
  for (int k = 0; k < ns; ++k) {
    const Sp& s = sp[k];
    const double g = s.g, f = s.f, om = 1.0 - f;
    const double h = (g - f) / om;
    const double dh = ((s.dg - s.df) * om + (g - f) * s.df) / (om * om);
    const double sc = sf[k];
    S += sc; Tt += sc * h;
    dSP[k] = ct[k].sP; dSr[k] = ct[k].sr;
    dTP[k] = h * ct[k].sP; dTr[k] = h * ct[k].sr + sc * dh;
  }
  const double tau = tauc;
  for (int k = 0; k < 3; ++k) {
    const double dtP = tclamp ? 0.0 : ct[k].tP, dtr = tclamp ? 0.0 : ct[k].tr;
    J[0][k] = ct[k].t0P;                       J[0][3 + k] = ct[k].t0r;
    J[1][k] = dtP;                             J[1][3 + k] = dtr;
    J[2][k] = dSP[k] / tau - S * dtP / (tau * tau);
    J[2][3 + k] = dSr[k] / tau - S * dtr / (tau * tau);
    J[3][k] = dTP[k] / S - Tt * dSP[k] / (S * S);
    J[3][3 + k] = dTr[k] / S - Tt * dSr[k] / (S * S);
  }
  return 1;
}

#define CLD_PRIMAL_PARAMS \
    int ncol, int nl, int inflag, int iceflag, int liqflag, \
    const float* __restrict__ tab, const int* __restrict__ ngb, const float* __restrict__ cldfmc, \
    const float* __restrict__ ciwpmc, const float* __restrict__ clwpmc, const float* __restrict__ cswpmc, \
    const float* __restrict__ reicmc, const float* __restrict__ relqmc, const float* __restrict__ resnmc, \
    const float* __restrict__ taucmc
#define CLD_PRIMAL_ARGS tab, ngb, cldfmc, ciwpmc, clwpmc, cswpmc, reicmc, relqmc, resnmc, taucmc

// Derivative pointers are not __restrict__: the window binds them into one float32 shadow of allocation 0 (the
// forward's McICA slab holds cldfmc, the three paths and taucmc/ssacmc/asmcmc/fsfcmc at disjoint offsets).
extern "C" __global__ void rsw_cldprmc_b__tl(
    CLD_PRIMAL_PARAMS,
    const float* t_ciwp, const float* t_clwp, const float* t_cswp,
    const float* t_reic, const float* t_relq, const float* t_resn,
    float* t_taor, float* t_tau, float* t_ssa, float* t_asm, unsigned int* fail) {
  const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (long long)ncol * nl) return;
  const long long cl = t;                       // col * nl + lay
  const double tr[3] = {(double)t_relq[cl], (double)t_reic[cl], (double)t_resn[cl]};
  for (int ig = 0; ig < NGPT; ++ig) {
    const long long w = cl * NGPT + ig;
    double J[4][6];
    if (!optics(J, inflag, iceflag, liqflag, w, cl, ig, CLD_PRIMAL_ARGS, fail)) {
      t_taor[w] = t_tau[w] + 0.0f;              // 2282: taormc = taucmc_in; the other three pass through
      continue;
    }
    const double x[6] = {(double)t_clwp[w], (double)t_ciwp[w], (double)t_cswp[w], tr[0], tr[1], tr[2]};
    double y[4];
    for (int o = 0; o < 4; ++o) {
      double s = 0.0;
      for (int i = 0; i < 6; ++i) s += J[o][i] * x[i];
      y[o] = s;
    }
    t_taor[w] = (float)y[0] + 0.0f;
    t_tau[w] = (float)y[1] + 0.0f;
    t_ssa[w] = (float)y[2] + 0.0f;
    t_asm[w] = (float)y[3] + 0.0f;
  }
}

extern "C" __global__ void rsw_cldprmc_b__ad(
    CLD_PRIMAL_PARAMS,
    float* b_ciwp, float* b_clwp, float* b_cswp,
    float* b_reic, float* b_relq, float* b_resn,
    float* b_taor, float* b_tau, float* b_ssa, float* b_asm, unsigned int* fail) {
  const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (long long)ncol * nl) return;
  const long long cl = t;
  double ar[3] = {0.0, 0.0, 0.0};
  bool any = false;
  for (int ig = 0; ig < NGPT; ++ig) {           // fixed order: the radius cotangents are deterministic
    const long long w = cl * NGPT + ig;
    double J[4][6];
    if (!optics(J, inflag, iceflag, liqflag, w, cl, ig, CLD_PRIMAL_ARGS, fail)) {
      b_tau[w] = (float)((double)b_tau[w] + (double)b_taor[w]) + 0.0f;   // transpose of 2282 taormc = taucmc_in
      b_taor[w] = 0.0f;
      continue;
    }
    const double g[4] = {(double)b_taor[w], (double)b_tau[w], (double)b_ssa[w], (double)b_asm[w]};
    b_taor[w] = 0.0f; b_tau[w] = 0.0f; b_ssa[w] = 0.0f; b_asm[w] = 0.0f;   // consumed: overwritten words
    double xb[6];
    for (int i = 0; i < 6; ++i) {
      double s = 0.0;
      for (int o = 0; o < 4; ++o) s += J[o][i] * g[o];
      xb[i] = s;
    }
    b_clwp[w] = (float)((double)b_clwp[w] + xb[0]) + 0.0f;
    b_ciwp[w] = (float)((double)b_ciwp[w] + xb[1]) + 0.0f;
    b_cswp[w] = (float)((double)b_cswp[w] + xb[2]) + 0.0f;
    ar[0] += xb[3]; ar[1] += xb[4]; ar[2] += xb[5];
    any = true;
  }
  if (any) {
    b_relq[cl] = (float)((double)b_relq[cl] + ar[0]) + 0.0f;
    b_reic[cl] = (float)((double)b_reic[cl] + ar[1]) + 0.0f;
    b_resn[cl] = (float)((double)b_resn[cl] + ar[2]) + 0.0f;
  }
}
