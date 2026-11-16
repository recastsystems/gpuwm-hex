// SPDX-License-Identifier: Apache-2.0
// ad/held-mpas_init_rwgt (autodiff, private): TL and AD entries of the RRTMG LW setup kernel mpas_init_rwgt.
//
// Forward: gpuwm-hex 16251589 tree/src/hexcore/mpasphys/rrtmg_lw_v842.py:2326-2340 (launch :2399-2400, inside
// initialize_device :2375, called once per loaded LW module by load_tables_device :1292), captured source
// 683d7a07.cu:10536-10550, image 34da9e09. Port of MPAS-A v8.4.2 module_ra_rrtmg_lw.F:8082-8113 (rrtmg_lw_ini).
//
//   rwgt[band*16+g] = wt[g] / sum_{old in group ngm[band*16+g]} wt[old]   (bands with ngc < 16)
//   rwgt[band*16+g] = 1                                                  (bands 3 and 5, ngc = 16)
//
// Every word it reads (mpas_static_ngc/ngn/ngm/wt) is a compiled-in __device__ const table; its one argument rwgt is
// written in full (256 words, overwrite) and never read. So the launch map is F(rwgt_pre) = c, a constant:
//   TL: d rwgt_post = 0            (the shadow words of rwgt are overwritten with +0.0)
//   AD: a rwgt_pre  = 0            (the cotangent of the overwritten words is dropped: in place, io inout)
// This is the exact derivative (no approximation, no frozen coefficient): there is no limiter, clamp or real-valued
// branch on state. The only branch, `ngc[band] < 16` (683d7a07.cu:10541), tests a constant table.
// Precision: each stored derivative word is +0.0 (bit pattern 0, no negative zero), the same rule as the live schemes
// (one rounding per stored word, +0.0 added before each store: 0 + 0.0 = +0.0).
// Geometry "forward" (grid 1, block 1, as the forward): thread 0 writes all 256 words, as the forward does; any other
// thread returns (683d7a07.cu:10538), so a wider launch is harmless.
// No failure slot: whatever the incoming tangent (TL) or cotangent (AD) of rwgt holds, NaN and Inf included, it is
// overwritten exactly (the forward ignores rwgt's pre-content), so there is nothing to refuse. A slot would only abort
// a window leg (window.rs:497-498 and :791-792 refuse the step when any slot fires) for a word this map discards.
#define NRWGT 256

extern "C" __global__ void mpas_init_rwgt__tl(float* __restrict__ d_rwgt) {
  if (threadIdx.x || blockIdx.x) return;
  for (int i = 0; i < NRWGT; ++i) d_rwgt[i] = 0.0f + 0.0f;
}

extern "C" __global__ void mpas_init_rwgt__ad(float* __restrict__ a_rwgt) {
  if (threadIdx.x || blockIdx.x) return;
  for (int i = 0; i < NRWGT; ++i) a_rwgt[i] = 0.0f + 0.0f;
}
