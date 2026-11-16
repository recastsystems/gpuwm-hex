// The tape-free MYNN-EDMF backward of rrfs_mynnedmf_run_l16 for the binder (AD_P3_DIRECT=1), items 2 of the bl-card page:
// the gather from the 91 MwBind rows into the joined backward's ledger, join/blocks/join.hpp's column_reverse, and the
// scatter of the ledger into the input bars. The frame (the forward's cut-entry words) comes from the capture image
// (join/card/capture_gen.py, entry mynn_capture_l1), one ColumnFrame<float> a column, written just before this launch.
// Built with NVRTC in the card form to ship, m150o0e (bl-card page): join/card/nvrtc_build.py with
// JOIN_BLOCKS=<arena_rewrite.py JW_MIN_WORDS=150 JW_OBJECTS=0 output> and -DJW_ARENA -DCARD_FLOAT_VIA_DOUBLE_ALL.
// The interface is code/join_test.cpp's seed_bars / finish_bars (the CPU route checked against the native tape on every
// word of 1,000 real columns a mesh), written for the program's arrays: in-place convention, ybar = the shadow snapshot,
// xbar = the shadow, which the binder has just set equal to ybar, so words this column does not touch keep identity bars.
// Grid-stride over the batch's columns; each thread owns wbytes of workspace (jw arena).
// Status a column (failures[4c]): 0 ok; 1 geometry; 2 live plume (refused); 3 block; 5 launch switches differ from the
// recipe the joined backward was proven for (gen_whole.py closure freeze, 3 km launch flags); 6 initflag launch
// (f_mym_initialize is not in the join); 7 frame not written by the capture; +64 workspace too small.
#include "join.hpp"
#include "abi.cuh"
#ifndef JW_ARENA
namespace jw { struct NoOver { __device__ int operator*() const { return 0; } };
               __device__ inline void begin(char*, long long) {} __device__ inline NoOver over() { return NoOver{}; } }
#endif
using namespace direct;
using namespace direct::join;
typedef float CP;

// MwBind rows, in abi.json's float_arrays order (pinned to 20a441f67; the binder refuses a row in another order)
enum MwField : int {W_dx=0, W_xland=1, W_ps=2, W_ts=3, W_qsfc=4, W_ust=5, W_ch=6, W_hfx=7, W_qfx=8, W_wspd=9, W_znt=10, W_uoce=11,
 W_voce=12, W_dz=13, W_u=14, W_v=15, W_w=16, W_th=17, W_tt=18, W_p=19, W_exner=20, W_rho=21, W_qv=22, W_qc=23, W_qi=24, W_qs=25,
 W_nc=26, W_ni=27, W_nifa=28, W_nwfa=29, W_nbca=30, W_qoz=31, W_rthraten=32, W_pblh=33, W_cldfra_bl=34, W_qc_bl=35, W_qi_bl=36,
 W_maxwidth=37, W_maxmf=38, W_ztop_plume=39, W_excess_h=40, W_excess_q=41, W_maxwidth_dd=42, W_maxmf_dd=43, W_maxtkeprod=44,
 W_cldtop_cooling=45, W_ent_eff=46, W_qke=47, W_qke_adv=48, W_tsq=49, W_qsq=50, W_cov=51, W_el_pbl=52, W_rublten=53, W_rvblten=54,
 W_rthblten=55, W_rqvblten=56, W_rqcblten=57, W_rqiblten=58, W_rqsblten=59, W_rncblten=60, W_rniblten=61, W_rnifablten=62,
 W_rnwfablten=63, W_rnbcablten=64, W_rqozblten=65, W_edmf_a=66, W_edmf_w=67, W_edmf_qt=68, W_edmf_thl=69, W_edmf_ent=70, W_edmf_qc=71,
 W_sub_thl=72, W_sub_sqv=73, W_det_thl=74, W_det_sqv=75, W_exch_h=76, W_exch_m=77, W_dqke=78, W_qwt=79, W_qshear=80, W_qbuoy=81,
 W_qdiss=82, W_sh3d=83, W_sm3d=84, W_pattern_spp=85, W_chem3d=86, W_settle3d=87, W_vd3d=88, W_frp_mean=89, W_emis_ant_no=90};

extern "C" {
// version 2 (frames from the capture image, ledger and workspace from the binder): version, launch bytes, fields, required
// stages, complete stages, threads a block, frame bytes (the capture image must write the same), ledger bytes a column
__device__ unsigned long long mynn_direct_info[8] = {2, sizeof(MwLaunch), 91, 8191, 8191, 32, sizeof(ColumnFrame<float>), sizeof(ColumnBars)};
}

struct Geo { int ims, ime, jms, jme, kms, kme; long long ni, nk; };
__device__ inline long long at3(const Geo& g, int i, int k, int j) { return (i - g.ims) + (long long)(k - g.kms) * g.ni + (long long)(j - g.jms) * g.ni * g.nk; }
__device__ inline long long at2(const Geo& g, int i, int j) { return (i - g.ims) + (long long)(j - g.jms) * g.ni; }

// ledger row of each seeded output / passed-through input (join_test.cpp outs(), maps(), SC)
__device__ inline double* row(ColumnBars& B, int w) {
  switch (w) {
  case W_rublten: return B.du; case W_rvblten: return B.dv; case W_rthblten: return B.dth; case W_rqvblten: return B.rqv;
  case W_rqcblten: return B.rqc; case W_rqiblten: return B.rqi; case W_rqsblten: return B.rqs; case W_rncblten: return B.dqnc;
  case W_rniblten: return B.dqni; case W_rnifablten: return B.dqnifa; case W_rnwfablten: return B.dqnwfa; case W_qke: return B.qke1;
  case W_tsq: return B.tsq1; case W_qsq: return B.qsq1; case W_cov: return B.cov1; case W_qc_bl: return B.qc_bl; case W_qi_bl: return B.qi_bl;
  case W_cldfra_bl: return B.cf_bl; case W_qke_adv: return B.qke1;
  case W_dz: return B.dz; case W_u: return B.u; case W_v: return B.v; case W_th: return B.th; case W_tt: return B.tk; case W_p: return B.p;
  case W_exner: return B.ex; case W_rho: return B.rho; case W_qv: return B.qvm; case W_qc: return B.qcm; case W_qi: return B.qim;
  case W_qs: return B.qsm; case W_nc: return B.qnc; case W_ni: return B.qni; case W_nifa: return B.qnifa; case W_nwfa: return B.qnwfa;
  case W_ts: return &B.ts; case W_qfx: return &B.qfx; case W_hfx: return &B.hfx; case W_ust: return &B.ust; case W_xland: return &B.xland;
  case W_ps: return &B.ps; case W_wspd: return &B.wspd; case W_uoce: return &B.uoce; case W_voce: return &B.voce;
  default: return nullptr; }
}
__constant__ int OUTS[19] = {W_rublten, W_rvblten, W_rthblten, W_rqvblten, W_rqcblten, W_rqiblten, W_rqsblten, W_rncblten, W_rniblten,
                             W_rnifablten, W_rnwfablten, W_qke, W_tsq, W_qsq, W_cov, W_qc_bl, W_qi_bl, W_cldfra_bl, W_qke_adv};
__constant__ int MAPS[16] = {W_dz, W_u, W_v, W_th, W_tt, W_p, W_exner, W_rho, W_qv, W_qc, W_qi, W_qs, W_nc, W_ni, W_nifa, W_nwfa};
__constant__ int SCS[9] = {W_ts, W_qfx, W_hfx, W_ust, W_xland, W_ps, W_wspd, W_uoce, W_voce};
// the driver overwrites these (join_test.cpp OVER): their input bars are zero on levels 1..n (one word for a surface array)
__constant__ int OVER2[11] = {W_pblh, W_maxwidth, W_maxmf, W_ztop_plume, W_excess_h, W_excess_q, W_maxwidth_dd, W_maxmf_dd, W_maxtkeprod,
                              W_cldtop_cooling, W_ent_eff};
__constant__ int OVER3[39] = {W_cldfra_bl, W_qc_bl, W_qi_bl, W_qke, W_qke_adv, W_tsq, W_qsq, W_cov, W_el_pbl, W_rublten, W_rvblten,
  W_rthblten, W_rqvblten, W_rqcblten, W_rqiblten, W_rqsblten, W_rncblten, W_rniblten, W_rnifablten, W_rnwfablten, W_edmf_a, W_edmf_w,
  W_edmf_qt, W_edmf_thl, W_edmf_ent, W_edmf_qc, W_sub_thl, W_sub_sqv, W_det_thl, W_det_sqv, W_exch_h, W_exch_m, W_dqke, W_qwt, W_qshear,
  W_qbuoy, W_qdiss, W_sh3d, W_sm3d};

extern "C" __global__ void __launch_bounds__(32) mynn_whole_ad_direct(const MwLaunch* __restrict__ L, const unsigned char* __restrict__ frames,
    unsigned long long frame_bytes, unsigned char* __restrict__ bars, char* __restrict__ ws, int wbytes,
    int batch_first, int batch_count, unsigned long long workspace_stride, int ids, int ide, int jds, int jde, int kds, int kde, int ims, int ime,
    int jms, int jme, int kms, int kme, int its, int ite, int jts, int jte, int kts, int kte, int f_qc, int f_qi, int f_qs, int f_qoz, int f_nc,
    int f_ni, int f_nifa, int f_nwfa, int f_nbca, int initflag, int do_restart, int do_dacycling, int icloud_bl, float delt, int spp_pbl,
    int bl_mynn_tkeadvect, int bl_mynn_tkebudget, int bl_mynn_cloudpdf, int bl_mynn_mixlength, float bl_mynn_closure, int bl_mynn_stfunc,
    int bl_mynn_topdown, int bl_mynn_scaleaware, int bl_mynn_dheat_opt, int bl_mynn_edmf, int bl_mynn_edmf_dd, int bl_mynn_edmf_mom,
    int bl_mynn_edmf_tke, int bl_mynn_output, int bl_mynn_mixscalars, int bl_mynn_mixaerosols, int bl_mynn_mixnumcon, int bl_mynn_cloudmix,
    int bl_mynn_mixqt, int bl_mynn_ess, int mix_chem, int nchem, int ndvel, int enh_mix, int present_enh_mix) {
  const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x, T = (long long)gridDim.x * blockDim.x;
  char* const mine = ws + t * (long long)wbytes;
  // the switches of the 3 km launch the join was proven for (code/build.py's wrapper call, bltime.py's recorded scalars)
  const bool recipe = f_qc == 1 && f_qi == 1 && f_qs == 1 && f_qoz == 0 && f_nc == 1 && f_ni == 1 && f_nifa == 1 && f_nwfa == 1 && f_nbca == 0
      && icloud_bl == 1 && spp_pbl == 0 && bl_mynn_tkeadvect == 0 && bl_mynn_tkebudget == 0 && bl_mynn_cloudpdf == 2 && bl_mynn_mixlength == 1
      && bl_mynn_closure == 2.5999999046325684f && bl_mynn_stfunc == 1 && bl_mynn_topdown == 0 && bl_mynn_scaleaware == 1
      && bl_mynn_dheat_opt == 1 && bl_mynn_edmf == 1 && bl_mynn_edmf_dd == 0 && bl_mynn_edmf_mom == 1 && bl_mynn_edmf_tke == 0
      && bl_mynn_output == 0 && bl_mynn_mixscalars == 0 && bl_mynn_mixaerosols == 1 && bl_mynn_mixnumcon == 1 && bl_mynn_cloudmix == 1
      && bl_mynn_mixqt == 0 && bl_mynn_ess == 1 && mix_chem == 0 && enh_mix == 0 && do_restart == 0;
  const Geo g{ims, ime, jms, jme, kms, kme, (long long)(ime - ims + 1), (long long)(kme - kms + 1)};
  const int n = kte;
  const bool geo = kts == 1 && n >= 4 && n <= 59 && kms <= 1 && kme >= n && L->col_first >= 0 && L->col_count >= 0;
  for (long long c = t; c < L->col_count; c += T) {
    int code = !geo ? 1 : !recipe ? 5 : initflag != 0 ? 6 : 0;
    const ColumnFrame<CP>& F = *reinterpret_cast<const ColumnFrame<CP>*>(frames + c * frame_bytes);
    if (!code && F.n != n) code = 7;
    const long long index = (long long)batch_first + L->col_first + c;
    const int i = its + int(index % (ite - its + 1)), j = jts + int(index / (ite - its + 1));
    ColumnBars& B = *reinterpret_cast<ColumnBars*>(bars + c * (unsigned long long)sizeof(ColumnBars));
    if (!code) {
      { double* z = reinterpret_cast<double*>(&B); for (int q = 0; q < int(sizeof(ColumnBars) / 8); ++q) z[q] = 0.0; }
      for (int q = 0; q < 19; ++q) { const double* y = L->b[OUTS[q]].ybar; if (!y) continue; double* r = row(B, OUTS[q]);
        for (int k = 1; k <= n; ++k) r[k] += y[at3(g, i, k, j)]; }
      for (int q = 0; q < 16; ++q) { const double* y = L->b[MAPS[q]].ybar; if (!y) continue; double* r = row(B, MAPS[q]);
        for (int k = 1; k <= n; ++k) r[k] += y[at3(g, i, k, j)]; }
      for (int q = 0; q < 9; ++q) { const double* y = L->b[SCS[q]].ybar; if (y) *row(B, SCS[q]) += y[at2(g, i, j)]; }
#ifdef JOIN_PBLH
      // the bar of the height the column returns (its input bar stays zero below: the driver overwrites it)
      { const double* y = L->b[W_pblh].ybar; if (y) B.pblh += y[at2(g, i, j)]; }
#endif
      jw::begin(mine, wbytes);
      code = int(column_reverse(F, B));
      if (*jw::over()) code |= 64;
    }
    if (!code) {
      for (int q = 0; q < 11; ++q) { double* x = L->b[OVER2[q]].xbar; if (x) x[at2(g, i, j)] = 0.0; }
      for (int q = 0; q < 39; ++q) { double* x = L->b[OVER3[q]].xbar; if (x) for (int k = 1; k <= n; ++k) x[at3(g, i, k, j)] = 0.0; }
      for (int q = 0; q < 16; ++q) { double* x = L->b[MAPS[q]].xbar; if (!x) continue; const double* r = row(B, MAPS[q]);
        for (int k = 1; k <= n; ++k) x[at3(g, i, k, j)] = r[k]; }
      { double* x = L->b[W_qke].xbar; if (x) for (int k = 1; k <= n; ++k) x[at3(g, i, k, j)] = B.qke0[k]; }
      { double* x = L->b[W_qsq].xbar; if (x) for (int k = 1; k <= n; ++k) x[at3(g, i, k, j)] = B.qsq0[k]; }
      for (int q = 0; q < 9; ++q) { double* x = L->b[SCS[q]].xbar; if (x) x[at2(g, i, j)] = *row(B, SCS[q]); }
    }
    if (L->failures) { L->failures[4 * c] = code; L->failures[4 * c + 1] = 0; L->failures[4 * c + 2] = 0; L->failures[4 * c + 3] = 0; }
  }
  (void)batch_count; (void)workspace_stride; (void)delt; (void)ids; (void)ide; (void)jds; (void)jde; (void)kds; (void)kde;
  (void)jms; (void)jme; (void)do_dacycling; (void)nchem; (void)ndvel; (void)present_enh_mix;
}
