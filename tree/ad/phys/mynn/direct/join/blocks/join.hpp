#pragma once
// The join reads back no bar of a frozen input: the block reverses send those to a sink (numeric.hpp, DIRECT_FROZEN_BAR).
#define DIRECT_SINK_FROZEN
// The whole-column MYNN-EDMF backward with no tape: every block's transpose joined in the
// reverse order of the native column (rrfs_mynnedmf_run_l16, production recipe of
// gen_whole.py at 20a441f67, FREEZE='closure').
// One serial owner per column. Float primals P, double bars. Fixed arrays only: no tape,
// no node count, no arena, no heap. The caller fills ColumnFrame with the cut-entry words
// of the forward (recomputed in the backward) and seeds ColumnBars with the output bars;
// column_reverse() turns them into the bars of the column's inputs.
// Blocks (each checked by its own lane against the native tape of its cut):
//   pre/post conversions, prediction, heat solves + repair
//   U V NC NI NWFA NIFA solves
//   turbulence production
//   cloud PDF
//   ESS gradients, driver thermodynamics and surface, dissipation heating
// Glue written here: the qkw interpolation of f_mym_length, the cloud/thermo blend after the
// plumes, the stage order, the version bookkeeping and the frozen-edge stops.
//   plume cloud-blend tail, plume values held                (mdir-cloud-plume, plume_tail.hpp)
// The plume block is the production B9 form: stage M4 runs the updraft integration on frozen
// inputs and f_dmp_mf's outputs are frozen after the call, so only the cloud-blend tail
// (F90:7544-7713) is live; it overwrites cldfra_bl1, qc_bl1, vt1, vq1 where the plume carries
// condensate. A frame marked plume_live (an image whose plume integration is live, as gen-A may
// be) is refused (Status::Plume): that transpose is not written.
#include "column.hpp"
#include "solves.hpp"
#include "ess.hpp"
#include "driver.hpp"
#include "dissipation.hpp"
#include "turbprod.hpp"
#include "cloud.hpp"
#include "plume_tail.hpp"
#ifdef JOIN_PBLH
// the boundary-layer height the column RETURNS (what the surface layer and the other schemes
// read at the next step) answers the column's inputs. blocks/pblhjac.hpp, the mirror of f_get_pblh, runs here on the
// driver cut's own words: the height reads thv1, zw1, dz1, ust and xland, all formed from the driver frame, so the
// capture writes no new word and the frame is unchanged. Inside the column every use of the height stays held
// (generator form: fast-closure/pknock.py mode O on freeze mask 0x7c9).
#include "pblhjac.hpp"
#endif
#ifdef JOIN_CLOSURE
// the closure live on the tape-free route is built ON the mixing block, not beside it: the
// same ten words and the same Jacobian (MixJac), with the level-2 functions folded into it by the capture, a third row
// (qdiv), and the turbulence-energy terms' bars of sm, sh, qdiv taken back through it (closure_prod_reverse below).
// The length stays held (the ranking: 1.010 against 1.023 of the real wind change).
#ifndef JOIN_MIX
#error "JOIN_CLOSURE extends the mixing block: build with -DJOIN_MIX -DJOIN_CLOSURE"
#endif
#endif
#ifndef DIRECT_FROZEN_BAR_IS_SINK
#error "include join.hpp before any block header: its stages zero and read only the live bars"
#endif
namespace direct { namespace join {

// JOIN_STAGE: empty on the CPU; the card build may set __noinline__ so each stage keeps its own stack frame
// (the frame the driver reserves is then the deepest stage's, not the sum of inlined ones). Same operations either way.
#ifndef JOIN_STAGE
#define JOIN_STAGE
#endif
enum class Status : int { Ok = 0, Geometry = 1, Plume = 2, Block = 3, Mix = 4 };

#ifdef JOIN_MIX
template<class P> struct MixJac {
  int on = 0, bad = 0; double t = 0.0;  // on: written by a mixing capture; bad: mirror mismatch at some level; t: rule threshold
  P qtw[MAX], sm2[MAX], sh2[MAX];       // qtw(k); level-2 sm, sh (the capture's scratch for the mirror)
  double jac[MAX * 20];                 // per level k: d ksm / d word (10), then d ksh / d word (10); words: mixjac::Word
#ifdef JOIN_CLOSURE
  double jq[MAX * 10];                  // per level k: d qdiv / d word (10); zero on the level-2.5 arm and where the rule holds
#endif
};
#endif
#ifdef JOIN_PMOM
// ---- The plume's transport of momentum, live (gen_whole.py --plume-live mom; bl-live-reverse, 2026-11-14).
// In that form the three terms the wind solves read from the plume (s_awu/s_awv, sub_u/sub_v, det_u/det_v) are a LINEAR
// function of the environment wind u1 (v1) whose coefficients are forward values with no derivative. The forward with
// captures writes those coefficients here (capture_gen.py, layout with -DJOIN_PMOM); pmom_reverse() is the transpose.
// For plume ip = 1..8 and level k (the native statements, coefficients in brackets held):
//   upu(2,ip)   = (u(1)*[dz2] + u(2)*[dz1]) / [dz1+dz2]                                 where bit ip of `started`
//   un          = upu(k,ip)*[om] + u(k)*[en]                                            k = 2..n-2 as far as the plume ran
//   envm(k)    += (0.5*(un + upu(k,ip)) - u(k)) * [dc]                                  envm starts as u
//   upu(k+1,ip) = un                                                                    where bit ip of pass[k]
//   s_awu(k)    = [adj] * sum_ip ([c1]*upu(k,ip)) * [psig]
//   sub_u(k)    = (([cs] * ([rho(k+1)]*u(k+1) - [rho(k)]*u(k))) / [dzi]) / [rhoi]
//   det_u(k)    = (([cdet] * (envm(k) - u(k))) * [ea]) * [psig]
// A word the forward did not reach stays zero and gives no bar. One coefficient set serves u and v.
constexpr int NUPM = 8;
template<class P> struct PlumeMom {
  int on = 0, started = 0, adj_set = 0;
  int pass[MAX];
  P dz1 = P(0), dz2 = P(0), psig = P(0), adj = P(0);
  P om[NUPM * MAX], en[NUPM * MAX], c1[NUPM * MAX];
  double dc[NUPM * MAX];
  P cs[MAX], dzi[MAX], rhoi[MAX], rho[MAX + 1], ea[MAX];
};
// fu[k] (k = 1..n), sub[k], det[k] (k = 1..n-1): bars of s_awu, sub_u, det_u from the wind solve; adds to ub[1..n].
template<class P> inline void pmom_reverse(const PlumeMom<P>& M, int n, const double* fu, const double* sub, const double* det, double* ub, bool planted) {
  if (!M.on) return;
  const double psig = double(M.psig), adj = M.adj_set ? double(M.adj) : 1.0, cdet = double(P(1.f) / P(45.f));
  double em[MAX]{}, upb[MAX + 1]{};
  for (int k = 1; k < n; ++k) {
    const double g = ((det[k] * psig) * double(M.ea[k])) * cdet; em[k] += g; ub[k] -= g;
    if (M.dzi[k] != P(0) && M.rhoi[k] != P(0)) {
      const double t = ((sub[k] / double(M.rhoi[k])) / double(M.dzi[k])) * double(M.cs[k]) * ((planted && k == 2) ? 1.001 : 1.0);
      ub[k + 1] += t * double(M.rho[k + 1]); ub[k] -= t * double(M.rho[k]);
    }
  }
  for (int k = 1; k <= n; ++k) ub[k] += em[k];
  const P den = r6_add(M.dz1, M.dz2);
  for (int ip = 0; ip < NUPM; ++ip) {
    const P* om = M.om + ip * MAX; const P* en = M.en + ip * MAX; const P* c1 = M.c1 + ip * MAX; const double* dc = M.dc + ip * MAX;
    for (int k = 1; k <= n; ++k) upb[k] = ((fu[k] * adj) * psig) * double(c1[k]);
    upb[n + 1] = 0.0;
    for (int k = n - 1; k >= 2; --k) {
      double unb = ((M.pass[k] >> ip) & 1) ? upb[k + 1] : 0.0;
      const double w = em[k] * dc[k];
      unb += 0.5 * w; upb[k] += 0.5 * w; ub[k] -= w;
      upb[k] += unb * double(om[k]); ub[k] += unb * double(en[k]);
    }
    if (((M.started >> ip) & 1) && den != P(0)) { const double t = upb[2] / double(den); ub[1] += t * double(M.dz2); ub[2] += t * double(M.dz1); }
  }
}
#endif
#ifdef JOIN_PHEAT
#ifndef JOIN_PMOM
#error "JOIN_PHEAT shares the plume-momentum capture (started, pass, c1, cs, dzi, rhoi, rho, ea, psig, adj): build with -DJOIN_PMOM"
#endif
// ---- The plume's transport of heat and moisture, live (gen_whole.py --plume-live momheat; full/closure, 2026-11-15).
// Same held-coefficient form as the momentum above. For plume ip and level k (coefficients in brackets held; the plume's
// own condensation, buoyancy, speed and area read upthl, upqt frozen):
//   upthl(2,ip)   = (thl(1)*[dz2] + thl(2)*[dz1]) / [dz1+dz2] + [exc_heat]              where bit ip of `started`
//   thln          = upthl(k,ip)*[om] + thl(k)*[en]              om = 1 - entexp, en = entexp (the momentum uses entexm)
//   envm_thl(k)  += (0.5*(thln + upthl(k,ip)) - thl(k)) * [dc]  dc = detrate*aratio*min(dzp,dzpmax); envm starts as thl
//   upthl(k+1,ip) = thln                                        where bit ip of pass[k]
//   s_awthl(k)    = [adj] * sum_ip ([c1]*upthl(k,ip)) * [psig]
//   sub_thl(k)    = (([cs] * ([rho(k+1)]*thl(k+1) - [rho(k)]*thl(k))) / [dzi]) / [rhoi]
//   det_thl(k)    = (([cdet] * (envm_thl(k) - thl(k))) * [ea]) * [psig]
// Moisture has two environment inputs, total water qt (what the plume carries) and vapour qv (what subsides and what the
// detrained air is compared with):
//   upqt(2,ip), qtn, upqt(k+1,ip)   as the heat lines, on qt
//   qv_ent        = 0.5*([g1]*qtn + [g2]*upqt(k,ip))            g1 = (qt_ent - qc_ent > 0), g2 = (upqt - upqc > 0); qc held
//   envm_sqv(k)  += (qv_ent - qv(k)) * [dc]                     envm starts as qv
//   s_awqt(k)     = [adj] * sum_ip ([c1]*upqt(k,ip)) * [psig];  s_awqv(k) = s_awqt(k) - [s_awqc(k)]
//   sub_sqv(k), det_sqv(k)          as the heat lines, on qv and envm_sqv
template<class P> struct PlumeHeat {
  int on = 0;
  int g1[MAX], g2[MAX];                 // per level: bit ip set where the first / second max() of qv_ent keeps its argument
  P om[NUPM * MAX], en[NUPM * MAX];
  double dc[NUPM * MAX];
};
// fl[k] (k = 1..n): bar of s_awthl; sub[k], det[k] (k = 1..n-1): bars of sub_thl, det_thl; adds to tb[1..n] (bar of thl).
template<class P> inline void pheat_reverse(const PlumeMom<P>& M, const PlumeHeat<P>& H, int n, const double* fl, const double* sub, const double* det, double* tb, bool planted) {
  if (!M.on || !H.on) return;
  const double psig = double(M.psig), adj = M.adj_set ? double(M.adj) : 1.0, cdet = double(P(1.f) / P(45.f));
  double em[MAX]{}, upb[MAX + 1]{};
  for (int k = 1; k < n; ++k) {
    const double g = ((det[k] * psig) * double(M.ea[k])) * cdet; em[k] += g; tb[k] -= g;
    if (M.dzi[k] != P(0) && M.rhoi[k] != P(0)) {
      const double t = ((sub[k] / double(M.rhoi[k])) / double(M.dzi[k])) * double(M.cs[k]) * ((planted && k == 2) ? 1.001 : 1.0);
      tb[k + 1] += t * double(M.rho[k + 1]); tb[k] -= t * double(M.rho[k]);
    }
  }
  for (int k = 1; k <= n; ++k) tb[k] += em[k];
  const P den = r6_add(M.dz1, M.dz2);
  for (int ip = 0; ip < NUPM; ++ip) {
    const P* om = H.om + ip * MAX; const P* en = H.en + ip * MAX; const P* c1 = M.c1 + ip * MAX; const double* dc = H.dc + ip * MAX;
    for (int k = 1; k <= n; ++k) upb[k] = ((fl[k] * adj) * psig) * double(c1[k]);
    upb[n + 1] = 0.0;
    for (int k = n - 1; k >= 2; --k) {
      double tnb = ((M.pass[k] >> ip) & 1) ? upb[k + 1] : 0.0;
      const double w = em[k] * dc[k];
      tnb += 0.5 * w; upb[k] += 0.5 * w; tb[k] -= w;
      upb[k] += tnb * double(om[k]); tb[k] += tnb * double(en[k]);
    }
    if (((M.started >> ip) & 1) && den != P(0)) { const double t = upb[2] / double(den); tb[1] += t * double(M.dz2); tb[2] += t * double(M.dz1); }
  }
}
// fq[k] (k = 1..n): bar of s_awqt PLUS bar of s_awqv; sub[k], det[k] (k = 1..n-1): bars of sub_sqv, det_sqv;
// adds to wb[1..n] (bar of total water qt) and vb[1..n] (bar of vapour qv).
template<class P> inline void pmoist_reverse(const PlumeMom<P>& M, const PlumeHeat<P>& H, int n, const double* fq, const double* sub, const double* det, double* wb, double* vb, bool planted) {
  if (!M.on || !H.on) return;
  const double psig = double(M.psig), adj = M.adj_set ? double(M.adj) : 1.0, cdet = double(P(1.f) / P(45.f));
  double em[MAX]{}, upb[MAX + 1]{};
  for (int k = 1; k < n; ++k) {
    const double g = ((det[k] * psig) * double(M.ea[k])) * cdet; em[k] += g; vb[k] -= g;
    if (M.dzi[k] != P(0) && M.rhoi[k] != P(0)) {
      const double t = ((sub[k] / double(M.rhoi[k])) / double(M.dzi[k])) * double(M.cs[k]);
      vb[k + 1] += t * double(M.rho[k + 1]); vb[k] -= t * double(M.rho[k]);
    }
  }
  for (int k = 1; k <= n; ++k) vb[k] += em[k];
  const P den = r6_add(M.dz1, M.dz2);
  for (int ip = 0; ip < NUPM; ++ip) {
    const P* om = H.om + ip * MAX; const P* en = H.en + ip * MAX; const P* c1 = M.c1 + ip * MAX; const double* dc = H.dc + ip * MAX;
    for (int k = 1; k <= n; ++k) upb[k] = ((fq[k] * adj) * psig) * double(c1[k]);
    upb[n + 1] = 0.0;
    for (int k = n - 1; k >= 2; --k) {
      double qnb = ((M.pass[k] >> ip) & 1) ? upb[k + 1] : 0.0;
      const double w = em[k] * dc[k] * ((planted && k == 3) ? 1.001 : 1.0);
      vb[k] -= w;
      if ((H.g1[k] >> ip) & 1) qnb += 0.5 * w;
      if ((H.g2[k] >> ip) & 1) upb[k] += 0.5 * w;
      upb[k] += qnb * double(om[k]); wb[k] += qnb * double(en[k]);
    }
    if (((M.started >> ip) & 1) && den != P(0)) { const double t = upb[2] / double(den); wb[1] += t * double(M.dz2); wb[2] += t * double(M.dz1); }
  }
}
#endif

template<class P> struct ColumnFrame {
  int n = 0; P dt = P(0); int dheat = 1; int plume_active = 0; int plume_live = 0;
  int ktop = -1;                        // the plume's top, zero-based (native ktop - 1); -1: no plume
  P tail[15 * 59 + 1];                  // tail inputs per level a,plth,plqt,plqc,w,dz,tk,p,qt,ex,thl,cldfra,qcbl,vt,vq; landsea
  int plant = 0;                        // test knob only: plants one x1.001 coefficient (0 in production)
  P pre[4 * MAX];                       // driver qv,qc,qi,qs (mixing ratios) at pre_run
  P drv[DNI];                           // driver_cut inputs (dz,sqv,sqc,sqi,sqs,th,ex,rho; ts,qfx,hfx,ust,xland)
  P cloud[CNI]; M3Col<P> ccfg{};        // f_mym_condensation cut inputs
  P blend[8 * MAX];                     // qc_bl,qi_bl,sqc,sqi,sqs,th,ex,sqv after the plumes
  P ess[ENI];                           // f_mym_level2 inputs
  P qkwx[2 * MAX];                      // qke (pre-predict), dz for f_mym_length's qkw
  P prod[TPNI]; double qdiv[MAX];       // production cut (closure frozen)
  P pred[11 * MAX + 2];                 // f_mym_predict inputs
  P diss[3 * MAX];                      // qke (post-predict), el, pressure
  P heat[HNI];                          // f_mynn_tendencies heat cut
  P solve[6][solves::NI];               // f_mynn_tendencies other solves
  P post[8 * MAX];                      // post_run: qv,qc,qi,qs mixing + dqv..dqs specific
#ifdef JOIN_PMOM
  PlumeMom<P> pm{};                     // the plume's momentum coefficients (last: every other offset as before)
#endif
#ifdef JOIN_MIX
  MixJac<P> mx{};                       // the mixing's stability Jacobian (last: every other offset as before)
#endif
#ifdef JOIN_PHEAT
  PlumeHeat<P> ph{};                    // the plume's heat and moisture coefficients (last: every other offset as before)
#endif
};

// Additive bars of every live version. Name suffix 0 = before prediction, 1 = after.
struct ColumnBars {
  double dz[MAX], u[MAX], v[MAX], th[MAX], tk[MAX], p[MAX], ex[MAX], rho[MAX];
  double qnc[MAX], qni[MAX], qnwfa[MAX], qnifa[MAX];
  double qvm[MAX], qcm[MAX], qim[MAX], qsm[MAX];          // driver mixing ratios
  double sqv[MAX], sqc[MAX], sqi[MAX], sqs[MAX];          // specific contents
  double zw[MAX + 1], qvr[MAX], qcr[MAX], qir[MAX], qsr[MAX], sqw[MAX], thl[MAX], thva[MAX];
  double qc_bl[MAX], qi_bl[MAX], cf_bl[MAX], vt[MAX], vq[MAX], sgm[MAX];
  double thlv[MAX], thvb[MAX], qc_tot[MAX], qi_tot[MAX], thl_tot[MAX];
  double dtl[MAX], dqw[MAX], dtv[MAX], gm[MAX], gh[MAX], qkw[MAX];
  double pdk[MAX], pdt[MAX], pdq[MAX], pdc[MAX], tcd[MAX], qcd[MAX];
  double qke0[MAX], qke1[MAX], qsq0[MAX], qsq1[MAX], tsq1[MAX], cov1[MAX], diss[MAX];
  double du[MAX], dv[MAX], dth[MAX], dqv[MAX], dqc[MAX], dqi[MAX], dqs[MAX];
  double dqnc[MAX], dqni[MAX], dqnwfa[MAX], dqnifa[MAX];
  double rqv[MAX], rqc[MAX], rqi[MAX], rqs[MAX];          // post_run outputs
  double ts, qfx, hfx, ust, xland, ps, wspd, uoce, voce;
  double flqv, flq, th_sfc, flt, fltv, rmol, rmolh, pmz, phh;
#ifdef JOIN_PBLH
  double pblh;                                             // the returned height (seeded from the output's bar; last: every other offset as before)
#endif
};

#ifdef JOIN_MIX
// ---- The mixing live (gen_whole.py --freeze closure --mix-live q:1,s:1 [--weakshear T]; bl-fast-live, 2026-11-14).
// In that form the diffusivities keep their derivative (k = 2..n; el held, the forward value unchanged):
//   dfm(k) = ((el(k)*qtw(k)) * ksm(k)) / dzk,  dfh(k) = ((el(k)*qkw(k)) * ksh(k)) / dzk,  dfq(k) = dfm(k),  dzk = 0.5*(dz(k)+dz(k-1))
// with ksm, ksh the final stability functions of the turbulence loop. qtw and qkw answer the turbulence energy through the
// same interpolation (qpe = 0 at closure 2.6, so qtw's derivative is qkw's); ksm, ksh answer ten live words through the
// 2 x 10 Jacobian the forward with captures writes (mixjac.hpp), zero where the weak-shear rule holds the stability path
// (gm(k) < T, decided by the capture on the forward's gm). Where the rule holds, ksm and ksh carry NO derivative at all:
// their answer to the turbulence energy (word 0 of the Jacobian) is held with their answer to shear and stratification, as
// the generated statement holds it (mw_ksm = mw_frz(mw_ksm)); only the el*qkw factor of the diffusivity stays live there.
// The six solves, the heat solves and the prediction compute the
// bars of dfm, dfh, dfq; mix_reverse() takes them back to qkw, gm, gh, u, v, dz, zw and the cloud fraction, whose later
// stages (rev_turb_prod's qkw, rev_turb_ess, rev_plume) carry them on. Frame words: prod (qkw gm gh dtl dqw dz el sm sh)
// and MixJac. A frame whose mirror of the stability block did not give the forward's sm, sh (bad), or whose diffusivities
// as re-formed here are not the words the solves read, is refused (Status::Mix): the column goes to the taped route.
#ifndef JOIN_MIX_PLANT
#define JOIN_MIX_PLANT 0   // card test image only (-DJOIN_MIX_PLANT=1): the x1.001 plant of plant 60 in every column
#endif
template<class P> inline void mix_dfs(const ColumnFrame<P>& F, int k, P& dfm, P& dfh, P& elqt, P& elq, P& dzk) {
  const P* x = F.prod;
  dzk = r6_mul(P(0.5f), r6_add(x[5 * MAX + k], x[5 * MAX + k - 1]));
  elqt = r6_mul(x[6 * MAX + k], F.mx.qtw[k]); elq = r6_mul(x[6 * MAX + k], x[k]);
  dfm = r6_div(r6_mul(elqt, x[7 * MAX + k]), dzk); dfh = r6_div(r6_mul(elq, x[8 * MAX + k]), dzk);
}
// the guard: the diffusivities re-formed from the frame are the words the solves and the prediction read (bitwise)
template<class P> inline bool mix_frame_ok(const ColumnFrame<P>& F) {
  if (!F.mx.on || F.mx.bad) return false;
  for (int k = 2; k <= F.n; ++k) {
    P dfm, dfh, a, b, c; mix_dfs(F, k, dfm, dfh, a, b, c);
    if (!(dfm == F.solve[0][solves::DF * MAX + k]) || !(dfm == F.solve[1][solves::DF * MAX + k]) ||
        !(dfh == F.solve[2][solves::DF * MAX + k]) || !(dfh == F.solve[3][solves::DF * MAX + k]) || !(dfh == F.solve[4][solves::DF * MAX + k]) ||
        !(dfh == F.solve[5][solves::DF * MAX + k]) || !(dfh == F.heat[DFH * MAX + k]) || !(dfm == F.pred[3 * MAX + k])) return false;
  }
  return true;
}
// gm_bar / gh_bar: bars of dfm (or dfq, the same word) and dfh at levels 1..n (level 1 is the constant zero: no bar).
template<class P> inline void mix_reverse(const ColumnFrame<P>& F, const double* gm_bar, const double* gh_bar, ColumnBars& B) {
  const int n = F.n;
  for (int k = 2; k <= n; ++k) {
    const double gm = gm_bar ? gm_bar[k] : 0.0, gh = gh_bar ? gh_bar[k] : 0.0;
    if (gm == 0.0 && gh == 0.0) continue;
    P dfm, dfh, elqt, elq, dzk; mix_dfs(F, k, dfm, dfh, elqt, elq, dzk);
    const P* x = F.prod; const double el = double(x[6 * MAX + k]), ksm = double(x[7 * MAX + k]), ksh = double(x[8 * MAX + k]);
    const double bnm = gm / double(dzk) * (((F.plant == 60 || JOIN_MIX_PLANT) && k == n / 2) ? 1.001 : 1.0), bnh = gh / double(dzk);  // numerators (plant 60: test only)
    // quotient rule. The quotient is taken in double from the float numerator and denominator, as the generated rule
    // takes it (mw_ad_rules.inc r6_div: mw_quot(a.d, b.d, (double)a.v / (double)b.v, b.v)), not the forward's rounded
    // dfm, dfh. with the rounded words the bar of dz was one float rounding off the taped reverse's on
    // this term (12 km cut, one launch: dz 6.5e-09 of the field's largest word, every other array 1e-12 or less).
    const double qm = double(r6_mul(elqt, x[7 * MAX + k])) / double(dzk), qh = double(r6_mul(elq, x[8 * MAX + k])) / double(dzk);
    const double bdzk = -(gm * qm + gh * qh) / double(dzk);
    B.dz[k] += 0.5 * bdzk; B.dz[k - 1] += 0.5 * bdzk;
    B.qkw[k] += (bnm * ksm + bnh * ksh) * el;                                  // qtw (dfm) and qkw (dfh): one chain
    const double bs[2] = {bnm * double(elqt), bnh * double(elq)};
    const double* Jk = F.mx.jac + 20 * k; double g[10];
    for (int i = 0; i < 10; ++i) g[i] = bs[0] * Jk[i] + bs[1] * Jk[10 + i];
    B.qkw[k] += g[0]; B.gm[k] += g[1]; B.gh[k] += g[2];
    B.u[k] += g[3]; B.u[k - 1] -= g[3]; B.v[k] += g[4]; B.v[k - 1] -= g[4];
    B.dz[k] += g[5]; B.dz[k - 1] += g[6]; B.zw[k] += g[7]; B.cf_bl[k] += g[8]; B.cf_bl[k - 1] += g[9];
  }
}
#endif

#ifdef JOIN_CLOSURE
// ---- The closure live in the turbulence-energy terms (generator freeze groups 2 and 4 off, el held; the weak-shear rule).
// pdk, pdt, pdq, pdc answer sm(k), sh(k) and qdiv; turbprod_reverse returns their bars (kb: sm, sh, qdiv, stride MAX) and
// this takes them to the ten words through the transposed Jacobian of the capture (rows zero where the rule holds).
// The diffusivities' own bars of sm, sh go the same way in mix_reverse (the same Jacobian, level-2 folded in).
template<class P> inline void closure_prod_reverse(const ColumnFrame<P>& F, const double* kb, ColumnBars& B) {
  const int n = F.n;
  for (int k = 2; k <= n; ++k) {
    const double pl = (F.plant == 61 && k == n / 2) ? 1.001 : 1.0;                         // plant 61: test only
    const double bs = kb[k] * pl, bh = kb[MAX + k] * pl, bq = kb[2 * MAX + k] * pl;
    if (bs == 0.0 && bh == 0.0 && bq == 0.0) continue;
    const double* Jk = F.mx.jac + 20 * k; const double* Jq = F.mx.jq + 10 * k; double g[10];
    for (int i = 0; i < 10; ++i) g[i] = bs * Jk[i] + bh * Jk[10 + i] + bq * Jq[i];
    B.qkw[k] += g[0]; B.gm[k] += g[1]; B.gh[k] += g[2];
    B.u[k] += g[3]; B.u[k - 1] -= g[3]; B.v[k] += g[4]; B.v[k - 1] -= g[4];
    B.dz[k] += g[5]; B.dz[k - 1] += g[6]; B.zw[k] += g[7]; B.cf_bl[k] += g[8]; B.cf_bl[k - 1] += g[9];
  }
}
#endif

inline void add(double* d, const double* s, int n, int off = 0) { for (int k = 1; k <= n; ++k) d[k] += s[off + k]; }
inline void put(double* d, const double* s, int n, int off = 0) { for (int k = 1; k <= n; ++k) d[off + k] = s[k]; }
inline void clr(double* d, int n) { for (int k = 0; k <= n + 1 && k < MAX; ++k) d[k] = 0.0; }

// ---- post_run: mixing-ratio tendencies from specific tendencies and the original mixing ratios
template<class P> JOIN_STAGE bool rev_post(const ColumnFrame<P>& F, ColumnBars& B) {
  const int n = F.n; double s[4 * MAX]{}, b[8 * MAX]{};
  put(s, B.rqv, n, 0); put(s, B.rqc, n, MAX); put(s, B.rqi, n, 2 * MAX); put(s, B.rqs, n, 3 * MAX);
  post_reverse(n, F.dt, F.post, s, b, F.plant == 1);
  add(B.qvm, b, n, 0); add(B.qcm, b, n, MAX); add(B.qim, b, n, 2 * MAX); add(B.qsm, b, n, 3 * MAX);
  add(B.dqv, b, n, 4 * MAX); add(B.dqc, b, n, 5 * MAX); add(B.dqi, b, n, 6 * MAX); add(B.dqs, b, n, 7 * MAX);
  clr(B.rqv, n); clr(B.rqc, n); clr(B.rqi, n); clr(B.rqs, n);
  return true;
}

// ---- f_mynn_tendencies: four heat/moisture solves with repair, then U V NC NI NWFA NIFA.
// Frozen inputs (dfh, dfm, the plume fluxes, pblh) get no bar; sqc2/sqi2 enter the number
// solves only through comparisons (bits), so the two groups share no bar.
template<class P> JOIN_STAGE bool rev_tend_heat(const ColumnFrame<P>& F, ColumnBars& B) {
  const int n = F.n;
  double s[HNO]{}, b[HNI];   // b: only the fields read back below are zeroed and accumulated (heat_reverse sends the rest to a sink)
  { const int lf[13] = {DZ, RHO, TK, QV, EX, TH, THL, SQV, SQC, SQI, TCD, QCD, HEAT}; for (int f = 0; f < 13; ++f) for (int k = 0; k < MAX; ++k) b[lf[f] * MAX + k] = 0.0; for (int i = NHF * MAX; i < HNI; ++i) b[i] = 0.0; }
#ifdef JOIN_MIX
  // under JOIN_MIX heat_reverse ADDS the bar of dfh into b (numeric.hpp DIRECT_MIX_BAR) and mix_reverse reads
  // it back, so it is a field read back below and starts at zero like them. THE BREAKAGE: it was left out of the list
  // above, and on the card b is carved from the thread's workspace, which is not cleared and is kept across launches: the
  // bar of dfh started from the last launch's (or from whatever the card memory held). 12 km cut, first tape-free
  // launch: output bars 1e8 and more times the taped reverse's, growing every step (route_compare.py).
  for (int k = 0; k < MAX; ++k) b[DFH * MAX + k] = 0.0;
#endif
#ifdef JOIN_PHEAT
  // the bars of the plume's heat and vapour terms, read back by pheat_reverse / pmoist_reverse: fields read back, so they
  // start at zero (on the card b is carved from a workspace that is not cleared; see JOIN_MIX above)
  { const int pf[6] = {FAH, FAV, SUBH, DETH, SUBV, DETV}; for (int f = 0; f < 6; ++f) for (int k = 0; k < MAX; ++k) b[pf[f] * MAX + k] = 0.0; }
#endif
  put(s, B.dth, n, MAX); put(s, B.dqv, n, 2 * MAX); put(s, B.dqc, n, 3 * MAX); put(s, B.dqi, n, 4 * MAX); put(s, B.dqs, n, 5 * MAX);
  if (n < 3 || n > 59) return false;
  HeatState<P> hs; heat_forward(n, F.dt, F.heat, static_cast<P*>(nullptr), hs); heat_reverse(n, F.dt, F.heat, hs, s, b, F.plant == 2);
  double* to[14] = {B.dz, B.rho, B.tk, B.qvr, B.ex, B.th, B.thl, B.sqv, B.sqc, B.sqi, nullptr, B.tcd, B.qcd, B.diss};
  for (int j = 0; j < 14; ++j) if (to[j]) add(to[j], b, n, j * MAX);
#ifdef JOIN_MIX
  mix_reverse(F, nullptr, b + DFH * MAX, B);
#endif
#ifdef JOIN_PHEAT
  // mixqt = 0: the vapour solve reads s_awqv (= s_awqt - held s_awqc), so its bar is the bar of the plume's total water flux
  pheat_reverse(F.pm, F.ph, n, b + FAH * MAX, b + SUBH * MAX, b + DETH * MAX, B.thl, F.plant == 40);
  pmoist_reverse(F.pm, F.ph, n, b + FAV * MAX, b + SUBV * MAX, b + DETV * MAX, B.sqw, B.sqv, F.plant == 41);
#endif
  B.ps += b[NHF * MAX]; B.flt += b[NHF * MAX + 1]; B.flqv += b[NHF * MAX + 3];
  return true;
}
template<class P> JOIN_STAGE bool rev_tend_solve(const ColumnFrame<P>& F, ColumnBars& B, int j) {
  const int n = F.n;
  double* q[6] = {B.u, B.v, B.qnc, B.qni, B.qnwfa, B.qnifa};
  double* d[6] = {B.du, B.dv, B.dqnc, B.dqni, B.dqnwfa, B.dqnifa};
  double s[solves::NO], b[solves::NI];   // s is written whole by put(); b: only the fields read back below are zeroed
  { const int lf[6] = {solves::DZ, solves::RHO, solves::Q, solves::ZW, solves::TK, solves::QV}; for (int f = 0; f < 6; ++f) for (int k = 0; k < MAX; ++k) b[lf[f] * MAX + k] = 0.0; for (int i = solves::PS; i < solves::NI; ++i) b[i] = 0.0; }
#ifdef JOIN_MIX
  for (int k = 0; k < MAX; ++k) b[solves::DF * MAX + k] = 0.0;   // the bar of dfm / dfh, read back by mix_reverse (as DFH above)
#endif
#ifdef JOIN_PMOM
  // the bars of the plume's momentum terms, read back by pmom_reverse (numeric.hpp DIRECT_PMOM_BAR). The merge
  // of the plume momentum onto the sink form left them in the sink and not cleared: pmom_reverse read uncleared words.
  { const int pf[3] = {solves::FU, solves::SUB, solves::DET}; for (int f = 0; f < 3; ++f) for (int k = 0; k < MAX; ++k) b[pf[f] * MAX + k] = 0.0; }
#endif
  put(s, d[j], n);
  if (n < 2 || n >= MAX - 1) return false;
  solves::State<P> ss; solves::forward(n, j, F.dt, F.solve[j], static_cast<P*>(nullptr), ss); solves::reverse(n, j, F.dt, F.solve[j], ss, s, b, F.plant == 20 + j);
#ifdef JOIN_PMOM
  if (j < 2) pmom_reverse(F.pm, n, b + solves::FU * MAX, b + solves::SUB * MAX, b + solves::DET * MAX, q[j], F.plant == 30 + j);
#endif
#ifdef JOIN_MIX
  mix_reverse(F, j < 2 ? b + solves::DF * MAX : nullptr, j < 2 ? nullptr : b + solves::DF * MAX, B);
#endif
  add(B.dz, b, n, solves::DZ * MAX); add(B.rho, b, n, solves::RHO * MAX); add(q[j], b, n, solves::Q * MAX);
  add(B.zw, b, n, solves::ZW * MAX); add(B.tk, b, n, solves::TK * MAX); add(B.qvr, b, n, solves::QV * MAX);
  B.ps += b[solves::PS]; B.ust += b[solves::UST]; B.wspd += b[solves::WSPD];
  (j == 1 ? B.voce : B.uoce) += b[solves::OCE];
  return true;
}
template<class P> bool rev_tend(const ColumnFrame<P>& F, ColumnBars& B) {
  const int n = F.n;
  if (!rev_tend_heat(F, B)) return false;
  for (int j = 0; j < 6; ++j) if (!rev_tend_solve(F, B, j)) return false;
  double* d[6] = {B.du, B.dv, B.dqnc, B.dqni, B.dqnwfa, B.dqnifa};
  clr(B.dth, n); clr(B.dqv, n); clr(B.dqc, n); clr(B.dqi, n); clr(B.dqs, n);
  for (int j = 0; j < 6; ++j) clr(d[j], n);
  return true;
}

// ---- dissipation heating: post-prediction qke and pressure; el held
template<class P> JOIN_STAGE bool rev_diss(const ColumnFrame<P>& F, ColumnBars& B) {
  const int n = F.n; double b[3 * MAX]{};
  dissipation_reverse(n, F.dheat, F.diss, B.diss, b, F.plant == 3);
  add(B.qke1, b, n, 0); add(B.p, b, n, 2 * MAX); clr(B.diss, n);
  return true;
}

// ---- f_mym_predict: qke and qsq implicit solves, algebraic tsq and cov (closure 2.6)
template<class P> JOIN_STAGE bool rev_pred(const ColumnFrame<P>& F, ColumnBars& B) {
  const int n = F.n; double s[4 * MAX + 4]{}, b[11 * MAX + 2]{};
  for (int k = 1; k <= n; ++k) { s[k - 1] = B.qke1[k]; s[n + k - 1] = B.qsq1[k]; }
  for (int k = 1; k < n; ++k) { s[2 * n + k - 1] = B.tsq1[k]; s[3 * n - 1 + k - 1] = B.cov1[k]; }
  s[3 * n - 2] += B.tsq1[n]; s[4 * n - 3] += B.cov1[n];      // top copies fold into n-1
  if (!predict_reverse(n, 1, F.dt, F.pred, s, b, F.plant == 4)) return false;
  add(B.dz, b, n, 0); add(B.rho, b, n, MAX);                  // el, dfq, s_aw: frozen (dfq live under JOIN_MIX)
#ifdef JOIN_MIX
  mix_reverse(F, b + 3 * MAX, nullptr, B);
#endif
  add(B.pdk, b, n, 5 * MAX); add(B.pdq, b, n, 6 * MAX); add(B.pdt, b, n, 7 * MAX); add(B.pdc, b, n, 8 * MAX);
  add(B.qke0, b, n, 9 * MAX); add(B.qsq0, b, n, 10 * MAX);
  B.ust += b[11 * MAX]; B.pmz += b[11 * MAX + 1];
  clr(B.qke1, n); clr(B.qsq1, n); clr(B.tsq1, n); clr(B.cov1, n);
  return true;
}

// ---- f_mym_turbulence: production (closure frozen) -> qkw of f_mym_length -> ESS gradients
// (two calls, so the card's stage gate can judge ESS against its own noise; rev_turb runs both, in this order)
template<class P> JOIN_STAGE bool rev_turb_prod(const ColumnFrame<P>& F, ColumnBars& B) {
  const int n = F.n;
  { double s[TPNO]{}, b[TPNB]{};
    put(s, B.pdk, n, 0); put(s, B.pdt, n, MAX); put(s, B.pdq, n, 2 * MAX); put(s, B.pdc, n, 3 * MAX);
    put(s, B.tcd, n, 4 * MAX); put(s, B.qcd, n, 5 * MAX);
#ifdef JOIN_CLOSURE
    double kb[3 * MAX]{};
    turbprod_reverse(n, P(2.6f), 0, F.prod, F.qdiv, s, b, F.plant == 50, kb);
    closure_prod_reverse(F, kb, B);
#else
    turbprod_reverse(n, P(2.6f), 0, F.prod, F.qdiv, s, b, F.plant == 50);
#endif
    add(B.qkw, b, n, 0); add(B.gm, b, n, MAX); add(B.gh, b, n, 2 * MAX); add(B.dtl, b, n, 3 * MAX);
    add(B.dqw, b, n, 4 * MAX); add(B.dz, b, n, 5 * MAX);
    clr(B.pdk, n); clr(B.pdt, n); clr(B.pdq, n); clr(B.pdc, n); clr(B.tcd, n); clr(B.qcd, n);
  }
  { // qkw(1) = sqrt(max(qke(1), qkemin)); qkw(k) = sqrt(max(qke(k)*abk + qke(k-1)*afk, qkemin)),
    // afk = dz(k)/(dz(k)+dz(k-1)), abk = 1-afk (f_mym_length, mixlength 1). Native rule weights.
    const P* qke = F.qkwx; const P* dz = F.qkwx + MAX; const P qkemin = P(1.e-5f);
    for (int k = n; k >= 2; --k) {
      const double g = B.qkw[k]; if (g == 0.0) continue;
      const P den = r6_add(dz[k], dz[k - 1]), afk = r6_div(dz[k], den), abk = r6_sub(P(1), afk);
      const P t1 = r6_mul(qke[k], abk), t2 = r6_mul(qke[k - 1], afk), a = r6_add(t1, t2);
      if (!(a > qkemin)) continue;                               // max: derivative on the qkemin arm is 0
      const P r = mynn_sqrt(a); const double ga = g / (2.0 * double(r)) * ((F.plant == 5 && k == n / 2) ? 1.001 : 1.0);
      B.qke0[k] += ga * double(abk); B.qke0[k - 1] += ga * double(afk);
      double gab = ga * double(qke[k]), gaf = ga * double(qke[k - 1]);
      gaf -= gab;                                                // abk = 1 - afk
      const double tq = gaf / double(den), w = double(dz[k]) / double(den);
      B.dz[k] += tq; B.dz[k] -= tq * w; B.dz[k - 1] -= tq * w;
    }
    { const double g = B.qkw[1];
      if (g != 0.0 && qke[1] > qkemin) B.qke0[1] += g / (2.0 * double(mynn_sqrt(qke[1]))); }
    clr(B.qkw, n);
  }
  return true;
}
template<class P> JOIN_STAGE bool rev_turb_ess(const ColumnFrame<P>& F, ColumnBars& B) {
  const int n = F.n;
  { double s[ENO]{}, b[ENI]{};
    put(s, B.dtl, n, 0); put(s, B.dqw, n, MAX); put(s, B.dtv, n, 2 * MAX); put(s, B.gm, n, 3 * MAX); put(s, B.gh, n, 4 * MAX);
    ess_reverse(n, F.ess, s, b, F.plant == 51);
    double* to[17] = {B.dz, B.zw, B.u, B.v, B.thl, B.thlv, B.th, B.p, B.ex, B.qke0, B.cf_bl, nullptr, nullptr, B.sqw, B.thvb, B.vt, B.vq};
    for (int j = 0; j < 17; ++j) if (to[j]) add(to[j], b, n, j * MAX);
    B.xland += b[17 * MAX + 1];
    clr(B.dtl, n); clr(B.dqw, n); clr(B.dtv, n); clr(B.gm, n); clr(B.gh, n);
  }
  return true;
}
template<class P> bool rev_turb(const ColumnFrame<P>& F, ColumnBars& B) {
  if (!rev_turb_prod(F, B)) return false;
  return rev_turb_ess(F, B);
}

// ---- the blend after the plumes (module_bl_mynnedmf.F90 1161-1171 in the generated text):
// qc_tot = max(qc_bl, sqc); qi_tot = max(qi_bl, sqi+sqs);
// thlv = (th - xlvcp/ex*qc_tot - xlscp/ex*qi_tot) * (1 + p608*sqv); thl_tot = the same bracket;
// thv = th * ((1 + p608*sqv) - (qc_tot + qi_tot)).
template<class P> JOIN_STAGE bool rev_blend(const ColumnFrame<P>& F, ColumnBars& B) {
  using namespace ess_const; const int n = F.n; const P* x = F.blend;
  auto X = [&](int j, int k) { return x[j * MAX + k]; };
  for (int k = n; k >= 1; --k) {
    const P qcb = X(0, k), qib = X(1, k), sqc = X(2, k), sqi = X(3, k), sqs = X(4, k), th = X(5, k), ex = X(6, k), sqv = X(7, k);
    const P qct = mx(qcb, sqc), sis = r6_add(sqi, sqs), qit = mx(qib, sis);
    const P cl = r6_div(P(xlvcp), ex), ci = r6_div(P(xlscp), ex);
    const P m1 = r6_mul(cl, qct), m2 = r6_mul(ci, qit), s1 = r6_sub(th, m1), br = r6_sub(s1, m2);
    const P pv = r6_mul(P(p608), sqv), fv = r6_add(P(1), pv), qs2 = r6_add(qct, qit), fw = r6_sub(fv, qs2);
    double gth = 0, gex = 0, gsqv = 0, gqct = B.qc_tot[k], gqit = B.qi_tot[k];
    double gbr = B.thl_tot[k], gfv = 0;
    { const double g = B.thlv[k]; gbr += g * double(fv); gfv += g * double(br); }
    { const double g = B.thvb[k]; gth += g * double(fw); const double gfw = g * double(th); gfv += gfw; gqct -= gfw; gqit -= gfw; }
    gsqv += gfv * double(p608) * ((F.plant == 6 && k == n / 2) ? 1.001 : 1.0);
    gth += gbr; const double gs1 = gbr, gm2 = -gbr; const double gneg1 = -gs1;
    double gcl = gneg1 * double(qct); gqct += gneg1 * double(cl);
    double gci = gm2 * double(qit); gqit += gm2 * double(ci);
    { const double t = gcl / double(ex); gex -= t * (double(xlvcp) / double(ex)); }
    { const double t = gci / double(ex); gex -= t * (double(xlscp) / double(ex)); }
    if (qcb > sqc) B.qc_bl[k] += gqct; else B.sqc[k] += gqct;
    if (qib > sis) B.qi_bl[k] += gqit; else { B.sqi[k] += gqit; B.sqs[k] += gqit; }
    B.th[k] += gth; B.ex[k] += gex; B.sqv[k] += gsqv;
  }
  clr(B.qc_tot, n); clr(B.qi_tot, n); clr(B.thlv, n); clr(B.thl_tot, n); clr(B.thvb, n);
  return true;
}

// ---- f_dmp_mf's cloud-blend tail with the plume values held (B9: M4 plume, outputs frozen after the call).
// Bars of the post-tail cldfra_bl1, qc_bl1, vt1, vq1 become bars of the live environment (dz1, tk1, pres1,
// qt1 = sqw1, ex1, thl1) and of the pre-tail cldfra_bl1, qc_bl1, vt1, vq1. ktop < 0: the tail is identity.
template<class P> JOIN_STAGE bool rev_plume(const ColumnFrame<P>& F, ColumnBars& B) {
  const int n = F.n; if (F.ktop < 0) return true;
  double s[4 * 59]{}, b[15 * 59 + 1]{};
  for (int k = 1; k <= n; ++k) { const int o = 4 * (k - 1); s[o] = B.cf_bl[k]; s[o + 1] = B.qc_bl[k]; s[o + 2] = B.vt[k]; s[o + 3] = B.vq[k]; }
  plume::tail_reverse(n, F.ktop, F.tail, s, b, static_cast<P*>(nullptr), F.plant == 7);
  for (int k = 1; k <= n; ++k) { const double* q = b + 15 * (k - 1);
    B.dz[k] += q[5]; B.tk[k] += q[6]; B.p[k] += q[7]; B.sqw[k] += q[8]; B.ex[k] += q[9]; B.thl[k] += q[10];
    B.cf_bl[k] = q[11]; B.qc_bl[k] = q[12]; B.vt[k] = q[13]; B.vq[k] = q[14]; }
  return true;
}

// ---- f_mym_condensation (cloud PDF, case 2)
template<class P> JOIN_STAGE bool rev_cloud(const ColumnFrame<P>& F, ColumnBars& B) {
  const int n = F.n; double s[CNO]{}, b[CNI]{};
  put(s, B.qc_bl, n, 0); put(s, B.qi_bl, n, MAX); put(s, B.cf_bl, n, 2 * MAX);
  put(s, B.vt, n, 3 * MAX); put(s, B.vq, n, 4 * MAX); put(s, B.sgm, n, 5 * MAX);
  if (!cloud_backward(n, F.cloud, F.ccfg, s, b, static_cast<P*>(nullptr), F.plant == 8)) return false;
  if (F.plant == 80) { int im = 0; for (int i = 0; i < CNI; ++i) if (std::fabs(b[i]) > std::fabs(b[im])) im = i; b[im] *= 1.001; }  // test knob: plant at the largest cloud bar
  double* to[18] = {B.zw, B.dz, B.th, B.ex, B.p, B.sqw, B.qsq0, nullptr, nullptr, B.sqi, B.sqs, B.sqc, nullptr, nullptr, B.thl, B.sqv, nullptr, nullptr};
  for (int j = 0; j < 18; ++j) if (to[j]) add(to[j], b, n, j * MAX);
  clr(B.qc_bl, n); clr(B.qi_bl, n); clr(B.cf_bl, n); clr(B.vt, n); clr(B.vq, n); clr(B.sgm, n);
  return true;
}

#ifdef JOIN_PBLH
// ---- f_get_pblh: the bar of the returned height to thv1, zw1, dz1 and ust (six words, pblhjac.hpp). The profile words are
// re-formed from the driver frame by the driver's own statements (the zw sum, driver_thermo's thv). The mirror's height
// must be the forward's own (the cloud cut's pblh1 word, read after f_get_pblh) to within float rounding of the two
// builds' tanh: otherwise its branches were not the forward's and the column is refused (the taped route).
template<class P> JOIN_STAGE bool rev_pblh(const ColumnFrame<P>& F, ColumnBars& B) {
  const double g = B.pblh; B.pblh = 0.0;
  if (g == 0.0) return true;
  const int n = F.n; const P* x = F.drv;
  P zw[MAX + 2], thv[MAX + 1];
  zw[1] = P(0);
  for (int k = 1; k <= n; ++k) {
    zw[k + 1] = r6_add(zw[k], x[k]);
    P a[6], o[7]; for (int q = 0; q < 6; ++q) a[q] = x[(q + 1) * MAX + k];
    driver_thermo<P, false>(a, o, nullptr, nullptr); thv[k] = o[6];
  }
  const pblhjac::Out o = pblhjac::get_pblh(1, n, [&](int k) { return float(thv[k]); }, [&](int k) { return float(zw[k]); },
                                           [&](int k) { return float(x[k]); }, float(x[8 * MAX + 3]), float(x[8 * MAX + 4]));
  const double fw = double(F.ccfg.pblh1), dv = double(o.pblh) - fw;
  if (!((dv < 0 ? -dv : dv) <= 2e-6 * (fw < 0 ? -fw : fw))) return false;
  pblhjac::reverse(o, g * (F.plant == 90 ? 1.001 : 1.0), B.thva, B.zw, B.dz, B.ust);   // plant 90: test only
  return true;
}
#endif

// ---- driver thermodynamics, heights and surface terms (stfunc 0, the column's module constant)
template<class P> JOIN_STAGE bool rev_driver(const ColumnFrame<P>& F, ColumnBars& B) {
  const int n = F.n; double s[DNO]{}, b[DNI]{};
  for (int k = 1; k <= n + 1; ++k) s[k] = B.zw[k];
  put(s, B.qvr, n, MAX); put(s, B.qcr, n, 2 * MAX); put(s, B.qir, n, 3 * MAX); put(s, B.qsr, n, 4 * MAX);
  put(s, B.sqw, n, 5 * MAX); put(s, B.thl, n, 6 * MAX); put(s, B.thva, n, 7 * MAX);
  const double t[9] = {B.flqv, B.flq, B.th_sfc, B.flt, B.fltv, B.rmol, B.rmolh, B.pmz, B.phh};
  for (int j = 0; j < 9; ++j) s[8 * MAX + j] = t[j];
  driver_reverse(n, F.drv, s, b, F.plant == 9);
  double* to[8] = {B.dz, B.sqv, B.sqc, B.sqi, B.sqs, B.th, B.ex, B.rho};
  for (int j = 0; j < 8; ++j) add(to[j], b, n, j * MAX);
  B.ts += b[8 * MAX]; B.qfx += b[8 * MAX + 1]; B.hfx += b[8 * MAX + 2]; B.ust += b[8 * MAX + 3]; B.xland += b[8 * MAX + 4];
  for (int k = 0; k <= n + 1 && k <= MAX; ++k) B.zw[k] = 0.0;
  clr(B.qvr, n); clr(B.qcr, n); clr(B.qir, n); clr(B.qsr, n); clr(B.sqw, n); clr(B.thl, n); clr(B.thva, n);
  B.flqv = B.flq = B.th_sfc = B.flt = B.fltv = B.rmol = B.rmolh = B.pmz = B.phh = 0.0;
  return true;
}

// ---- pre_run: specific contents from the driver's mixing ratios
template<class P> JOIN_STAGE bool rev_pre(const ColumnFrame<P>& F, ColumnBars& B) {
  const int n = F.n; double s[4 * MAX]{}, b[4 * MAX]{};
  put(s, B.sqv, n, 0); put(s, B.sqc, n, MAX); put(s, B.sqi, n, 2 * MAX); put(s, B.sqs, n, 3 * MAX);
  pre_reverse(n, F.pre, s, b, F.plant == 10);
  add(B.qvm, b, n, 0); add(B.qcm, b, n, MAX); add(B.qim, b, n, 2 * MAX); add(B.qsm, b, n, 3 * MAX);
  clr(B.sqv, n); clr(B.sqc, n); clr(B.sqi, n); clr(B.sqs, n);
  return true;
}

// The stages in reverse order of the forward. A hook lets a test read the bars between stages.
enum Stage : int { Post, Tend, Diss, Pred, Turb, Blend, Plume, Cloud, Driver, Pre, NSTAGE };
template<class P, class Hook> Status column_reverse(const ColumnFrame<P>& F, ColumnBars& B, Hook&& hook) {
  if (F.n < 4 || F.n > 59) return Status::Geometry;
  if (F.plume_live && F.plume_active) return Status::Plume;
#ifdef JOIN_PLUME_REFUSE
  // for a form whose updraft plume keeps its whole derivative (generator freeze group 8 off). A column in which
  // any plume started (the started bits of the plume-momentum capture) or carries area goes to the taped route; the others,
  // where the plume returns zeros with no derivative, are reversed here. Needs JOIN_PMOM (the capture of the started bits).
#ifndef JOIN_PMOM
#error "JOIN_PLUME_REFUSE reads the plume-momentum capture: build with -DJOIN_PMOM"
#endif
  if (F.plume_active || F.pm.started) return Status::Plume;
#endif
#ifdef JOIN_MIX
  if (!mix_frame_ok(F)) return Status::Mix;
#endif
  if (!rev_post(F, B)) return Status::Block;   hook(Post);
  if (!rev_tend(F, B)) return Status::Block;   hook(Tend);
  if (!rev_diss(F, B)) return Status::Block;   hook(Diss);
  if (!rev_pred(F, B)) return Status::Block;   hook(Pred);
  if (!rev_turb(F, B)) return Status::Block;   hook(Turb);
  if (!rev_blend(F, B)) return Status::Block;  hook(Blend);
  if (!rev_plume(F, B)) return Status::Block;  hook(Plume);
  if (!rev_cloud(F, B)) return Status::Block;  hook(Cloud);
#ifdef JOIN_PBLH
  if (!rev_pblh(F, B)) return Status::Block;
#endif
  if (!rev_driver(F, B)) return Status::Block; hook(Driver);
  if (!rev_pre(F, B)) return Status::Block;    hook(Pre);
  return Status::Ok;
}
template<class P> Status column_reverse(const ColumnFrame<P>& F, ColumnBars& B) { return column_reverse(F, B, [](int) {}); }
}}
