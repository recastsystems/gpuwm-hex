// SPDX-License-Identifier: Apache-2.0
// Held-kernel lane rlw_cldprmc_coalesced (ad/held-rlw_cldprmc_coalesced, private autodiff program): tangent and
// adjoint of the hrrrv5 RRTMG LW McICA cloud-optics launch rlw_cldprmc_coalesced, written by hand from the forward
// CUDA source
//   gpuwm-hex 16251589 tree/src/hexcore/mpasphys/rrtmg_lw_v842.py:736 (captured translation unit
//   683d7a070b22b058, lines 2323-2454; image 34da9e095e716797), a port of MPAS-A v8.4.2
//   module_ra_rrtmg_lw.F:2775-3036 (cldprmc).
// Per (col, lay, ig) the forward overwrites, where the McICA subcolumn is cloudy,
//   taucmc = (ciwp * abscoice(reic) + clwp * abscoliq(relq)) + cswp * abscosno(resn)
// with abscoice / abscosno / abscoliq table interpolations in the effective radii, and sets ncb_flag[col] = 1
// when an ncbands-writing branch runs. Nothing else is written (err_flag only on the fatal paths).
//
// ABI (ad.entry/v1, spec rlw_cldprmc_coalesced.spec.json). Geometry: one thread per (col, lay), ceil(ncol*nl/128)
// x 128, NOT the forward's (col, ig) geometry: the adjoint of the per-column radii reicmc/relqmc/resnmc sums over
// the 140 g-points of one (col, lay), and doing that sum in one thread in a fixed ig order keeps the reverse
// deterministic (bitwise across runs) with no atomics. Every forward layer is independent of the others (the only
// cross-layer effect, the early return on a fatal path, is a refusal below), so the regrouping is exact.
//   primal arrays (pre-launch copies, deriv.rs `primal`): cldfmc ciwpmc clwpmc cswpmc reicmc relqmc resnmc, the
//                 tables absice0..3 absliq1, ngb, and taucmc (its pre-launch value is a gate operand).
//   TL: t_ciwp t_clwp t_cswp t_reic t_relq t_resn are read (io `in`); t_tau is the pre-launch tangent on entry and
//       the post-launch tangent on exit (io `inout`); t_ncb (the int words of ncb_flag, float32-width shadow) is
//       set to 0 on every word the forward writes (an integer result carries no derivative).
//   AD: b_tau holds the post-launch cotangent on entry; every word the forward overwrote is consumed (zeroed: the
//       pre-launch taucmc enters only through the gate), the rest pass through (identity). b_ciwp b_clwp b_cswp
//       b_reic b_relq b_resn accumulate (io `accumulate`); b_ncb written words are consumed (zeroed).
//   fail: refusal slots, one per fatal path of the forward (it sets err_flag and returns; the host raises).
//
// Precision (as the live P3/P4/P5 schemes and the e2e entries): the primal is recomputed in float32 in the
// forward's own order of operations with the same __f*_rn intrinsics (strict build, --fmad=false --ftz=false),
// so every branch, index and coefficient is the executed one; every derivative coefficient and product is
// float64; derivative storage is the sweep's (float32 in the window), one rounding per stored word; +0.0 is added
// before each store so the zero direction has no negative zeros.
//
// Frozen-coefficient (executed-branch) treatment, per forward line (captured .cu 683d7a07):
//   * 2355-2357 gate cldfmc >= 1e-20 && (cwp >= 1e-20 || taucmc_in >= 1e-20): a step in cldfmc / cwp / taucmc_in,
//     taken from the trajectory. Not cloudy -> taucmc passes through (identity), no sensitivity to the paths.
//     Cloudy -> taucmc is overwritten: d(out)/d(taucmc_in) = 0.
//   * 2364 ice_plus_snow == 0 -> abscoice = abscosno = 0: frozen, so d taucmc/d ciwp = 0 there (one-sided: the
//     forward value is continuous but its right derivative would be abscoice(reic)); 2410 cswp > 0 gates snow
//     and 2427 clwp == 0 gates liquid the same way.
//   * 2386-2388, 2400-2402, 2417-2419, 2436-2439 table index (int)factor and the clamps index==43->42, 46->45,
//     58->57, 0->1: the index is frozen; the slope is the forward's own table difference RLW_SU(T[i+1], T[i])
//     times d fint / d radius (1/3 for ice and snow, 1 for liquid). Piecewise linear, continuous at the nodes; the
//     clamped end (fint in [1, 1.5) for radliq >= 59.5) is a linear extrapolation and keeps its slope.
//   * 2368, 2372, 2381, 2395, 2412, 2433 radius range checks and 2359 inflag == 1: fatal paths of the forward
//     (err_flag, host raises) -> refusal slots 0-3. inflag == 0 writes nothing (identity everywhere).
//   * iceflag / liqflag / inflag / ngb / icb are integer switches (launch values, inactive); the tables absice*,
//     absliq1 and absliq0 are constants (inactive).

#define NGPTLW 140
#define RLW_AD(a, b) __fadd_rn((a), (b))
#define RLW_SU(a, b) __fsub_rn((a), (b))
#define RLW_MU(a, b) __fmul_rn((a), (b))
#define RLW_DV(a, b) __fdiv_rn((a), (b))
#define AB2(tab, ld, i, b) tab[((i) - 1) + ((b) - 1) * (ld)]

__device__ __forceinline__ void refuse(unsigned int* fail, int slot) { if (fail) atomicExch(fail + slot, 1u); }

// One (col, lay, ig) word: the executed coefficients of taucmc in its six active inputs.
struct Opt {
  double ci, cl, cs;   // d taucmc / d ciwp, clwp, cswp  (= abscoice, abscoliq, abscosno as float32 values)
  double ri, rl, rs;   // d taucmc / d reic, relq, resn (= path * table slope * d fint / d radius)
  bool ncb;            // the forward writes ncb_flag[col] here
};

// Primal of one word in the forward's float32 order (captured .cu 2353-2449). Returns true where the forward
// overwrites taucmc (the derivative is then the Opt coefficients), false where it leaves it (identity).
__device__ __forceinline__ bool optics(
    Opt& o, int inflag, int iceflag, int liqflag, long long w, long long cl, int ig,
    const float* __restrict__ cldfmc, const float* __restrict__ ciwpmc, const float* __restrict__ clwpmc,
    const float* __restrict__ cswpmc, const float* __restrict__ reicmc, const float* __restrict__ relqmc,
    const float* __restrict__ resnmc, const float* __restrict__ absice1, const float* __restrict__ absice2,
    const float* __restrict__ absice3, const float* __restrict__ absice0, const float* __restrict__ absliq1,
    float absliq0, const int* __restrict__ ngb, const float* __restrict__ taucmc, unsigned int* fail) {
  const float cldmin = 1.e-20f;
  const int icb[16] = {1,2,3,3,3,4,4,4,5,5,5,5,5,5,5,5};
  const float ciwp = ciwpmc[w], clwp = clwpmc[w], cswp = cswpmc[w];
  const float cwp = RLW_AD(RLW_AD(ciwp, clwp), cswp);                                     // 2353
  if (!(cldfmc[w] >= cldmin && (cwp >= cldmin || taucmc[w] >= cldmin))) return false;     // 2355-2357
  if (inflag == 0) return false;                                                          // 2358: nothing written
  if (inflag == 1) { refuse(fail, 0); return false; }                                     // 2359
  o.ci = o.cl = o.cs = o.ri = o.rl = o.rs = 0.0; o.ncb = false;
  const float radice = reicmc[cl];                                                        // 2361
  float abscoice = 0.0f, abscosno = 0.0f;
  double sice = 0.0, ssno = 0.0, sliq = 0.0;     // d abs / d radius
  const float ice_plus_snow = RLW_AD(ciwp, cswp);                                         // 2363
  if (ice_plus_snow == 0.0f) {                                                            // 2364
    abscoice = 0.0f; abscosno = 0.0f;
  } else if (iceflag == 0) {                                                              // 2367
    if (radice < 10.0f) { refuse(fail, 1); return false; }
    abscoice = RLW_AD(absice0[0], RLW_DV(absice0[1], radice));
    sice = -(double)absice0[1] / ((double)radice * (double)radice);
  } else if (iceflag == 1) {                                                              // 2371
    if (radice < 13.0f || radice > 130.0f) { refuse(fail, 1); return false; }
    o.ncb = true;
    const int ib = icb[ngb[ig - 1] - 1];
    abscoice = RLW_AD(AB2(absice1, 2, 1, ib), RLW_DV(AB2(absice1, 2, 2, ib), radice));
    sice = -(double)AB2(absice1, 2, 2, ib) / ((double)radice * (double)radice);
  } else if (iceflag == 2) {                                                              // 2380
    if (radice < 5.0f || radice > 131.0f) { refuse(fail, 1); return false; }
    o.ncb = true;
    const float factor = RLW_DV(RLW_SU(radice, 2.0f), 3.0f);
    int index = (int)factor;
    if (index == 43) index = 42;
    const float fint = RLW_SU(factor, (float)index);
    const int ib = ngb[ig - 1];
    const float dtab = RLW_SU(AB2(absice2, 43, index + 1, ib), AB2(absice2, 43, index, ib));
    abscoice = RLW_AD(AB2(absice2, 43, index, ib), RLW_MU(fint, dtab));
    sice = (double)dtab / 3.0;
  } else {                                                                                // 2394: iceflag >= 3
    if (radice < 5.0f || radice > 140.0f) { refuse(fail, 1); return false; }
    o.ncb = true;
    const float factor = RLW_DV(RLW_SU(radice, 2.0f), 3.0f);
    int index = (int)factor;
    if (index == 46) index = 45;
    const float fint = RLW_SU(factor, (float)index);
    const int ib = ngb[ig - 1];
    const float dtab = RLW_SU(AB2(absice3, 46, index + 1, ib), AB2(absice3, 46, index, ib));
    abscoice = RLW_AD(AB2(absice3, 46, index, ib), RLW_MU(fint, dtab));
    sice = (double)dtab / 3.0;
  }
  if (cswp > 0.0f && iceflag == 5) {                                                      // 2410
    const float radsno = resnmc[cl];
    if (radsno < 5.0f || radsno > 140.0f) { refuse(fail, 2); return false; }
    o.ncb = true;
    const float factor = RLW_DV(RLW_SU(radsno, 2.0f), 3.0f);
    int index = (int)factor;
    if (index == 46) index = 45;
    const float fint = RLW_SU(factor, (float)index);
    const int ib = ngb[ig - 1];
    const float dtab = RLW_SU(AB2(absice3, 46, index + 1, ib), AB2(absice3, 46, index, ib));
    abscosno = RLW_AD(AB2(absice3, 46, index, ib), RLW_MU(fint, dtab));
    ssno = (double)dtab / 3.0;
  }
  float abscoliq;
  if (clwp == 0.0f) {                                                                     // 2427
    abscoliq = 0.0f;
  } else if (liqflag == 0) {
    abscoliq = absliq0;
  } else {                                                                                // 2431: liqflag == 1
    const float radliq = relqmc[cl];
    if (radliq < 2.5f || radliq > 60.0f) { refuse(fail, 3); return false; }
    int index = (int)RLW_SU(radliq, 1.5f);
    if (index == 0) index = 1;
    if (index == 58) index = 57;
    const float fint = RLW_SU(RLW_SU(radliq, 1.5f), (float)index);
    const int ib = ngb[ig - 1];
    const float dtab = RLW_SU(AB2(absliq1, 58, index + 1, ib), AB2(absliq1, 58, index, ib));
    abscoliq = RLW_AD(AB2(absliq1, 58, index, ib), RLW_MU(fint, dtab));
    sliq = (double)dtab;
  }
  // 2446-2449: taucmc = (ciwp abscoice + clwp abscoliq) + cswp abscosno
  o.ci = (double)abscoice; o.cl = (double)abscoliq; o.cs = (double)abscosno;
  o.ri = (double)ciwp * sice; o.rl = (double)clwp * sliq; o.rs = (double)cswp * ssno;
  return true;
}

#define CLD_PRIMAL_PARAMS \
    int ncol, int nl, int inflag, int iceflag, int liqflag, \
    const float* __restrict__ cldfmc, const float* __restrict__ ciwpmc, const float* __restrict__ clwpmc, \
    const float* __restrict__ cswpmc, const float* __restrict__ reicmc, const float* __restrict__ relqmc, \
    const float* __restrict__ resnmc, const float* __restrict__ absice1, const float* __restrict__ absice2, \
    const float* __restrict__ absice3, const float* __restrict__ absice0, const float* __restrict__ absliq1, \
    float absliq0, const int* __restrict__ ngb, const float* __restrict__ taucmc
#define CLD_PRIMAL_ARGS cldfmc, ciwpmc, clwpmc, cswpmc, reicmc, relqmc, resnmc, absice1, absice2, absice3, \
    absice0, absliq1, absliq0, ngb, taucmc

extern "C" __global__ void rlw_cldprmc_coalesced__tl(
    CLD_PRIMAL_PARAMS,
    const float* __restrict__ t_ciwp, const float* __restrict__ t_clwp, const float* __restrict__ t_cswp,
    const float* __restrict__ t_reic, const float* __restrict__ t_relq, const float* __restrict__ t_resn,
    float* __restrict__ t_tau, float* t_ncb, unsigned int* fail) {
  const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (long long)ncol * nl) return;
  const int col = (int)(t / nl);
  const long long cl = t;                       // col * nl + (lay - 1)
  const double tri = (double)t_reic[cl], trl = (double)t_relq[cl], trs = (double)t_resn[cl];
  bool ncb = false;
  for (int ig = 1; ig <= NGPTLW; ++ig) {
    const long long w = cl * NGPTLW + (ig - 1);
    Opt o;
    if (!optics(o, inflag, iceflag, liqflag, w, cl, ig, CLD_PRIMAL_ARGS, fail)) continue;
    const double d = ((o.ci * (double)t_ciwp[w] + o.cl * (double)t_clwp[w]) + o.cs * (double)t_cswp[w])
                   + ((o.ri * tri + o.rl * trl) + o.rs * trs);
    t_tau[w] = (float)d + 0.0f;
    ncb = ncb || o.ncb;
  }
  if (ncb) t_ncb[col] = 0.0f;                   // every writer stores the same 0 (benign identical stores)
}

extern "C" __global__ void rlw_cldprmc_coalesced__ad(
    CLD_PRIMAL_PARAMS,
    float* __restrict__ b_ciwp, float* __restrict__ b_clwp, float* __restrict__ b_cswp,
    float* __restrict__ b_reic, float* __restrict__ b_relq, float* __restrict__ b_resn,
    float* __restrict__ b_tau, float* b_ncb, unsigned int* fail) {
  const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (long long)ncol * nl) return;
  const int col = (int)(t / nl);
  const long long cl = t;
  double ari = 0.0, arl = 0.0, ars = 0.0;
  bool ncb = false, any = false;
  for (int ig = 1; ig <= NGPTLW; ++ig) {        // fixed order: the radius cotangents are deterministic
    const long long w = cl * NGPTLW + (ig - 1);
    Opt o;
    if (!optics(o, inflag, iceflag, liqflag, w, cl, ig, CLD_PRIMAL_ARGS, fail)) continue;
    const double g = (double)b_tau[w];
    b_tau[w] = 0.0f;                            // consumed: the pre-launch taucmc enters only through the gate
    b_ciwp[w] = (float)((double)b_ciwp[w] + o.ci * g) + 0.0f;
    b_clwp[w] = (float)((double)b_clwp[w] + o.cl * g) + 0.0f;
    b_cswp[w] = (float)((double)b_cswp[w] + o.cs * g) + 0.0f;
    ari += o.ri * g; arl += o.rl * g; ars += o.rs * g;
    ncb = ncb || o.ncb; any = true;
  }
  if (any) {
    b_reic[cl] = (float)((double)b_reic[cl] + ari) + 0.0f;
    b_relq[cl] = (float)((double)b_relq[cl] + arl) + 0.0f;
    b_resn[cl] = (float)((double)b_resn[cl] + ars) + 0.0f;
  }
  if (ncb) b_ncb[col] = 0.0f;
}
