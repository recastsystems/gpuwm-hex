#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# ad/rung3-params: parameter carrier for the P3 MYNN-EDMF whole launch (gen_whole.py output, exact executed branch).
# Parameter slot 0 = the TKE dissipation coefficient of the qke prediction (f_mym_predict: b1l = b1 * 0.25 *
# (el(k+1) + el(k)), b1 = 24, MYNN level-2.5 B1). Only this occurrence becomes the parameter; the closure's B1
# (stability functions, constexpr a1/g2 ...) stays the constant. The same occurrence is parameterised in the
# forward, so the finite differences and the derivative see the same parameter.
#   derivative unit: MynnColumn gets a member MR mw_p_b1, set per column before the driver call:
#     MODE 0 (TL): F(p, D(mw_par_t[0], &ctx)) if the parameter tangent is nonzero, else F(p) (no dependence: the
#                  tangent run is then the unchanged one);
#     MODE 1/2 (taped / count): F(p, D(ctx.leaf(), &ctx)); after the reverse the leaf's bar is added to
#                  mw_par_bar[column] (one thread per column, one launch per column: deterministic).
#   forward: the captured source with mw_parf(0, b1) at the same occurrence (module globals, literal by default).
# usage: gen_params_p3.py <mynn_whole.gen.cu> <captured forward .cu> <out_deriv.cu> <out_forward.cu>
import hashlib, json, sys
gen_p, fwd_p, out_d, out_f = sys.argv[1:5]
def die(m): sys.exit("gen_params_p3: " + m)
def sub1(s, old, new, what):
    n = s.count(old)
    if n != 1: die(f"{what}: matched {n}, expected 1")
    return s.replace(old, new, 1)
GLOB = """
// ===== rung3: parameter carrier (ad/rung3-params) =====
#define MW_NPAR 8
#define MW_PAR_COLS 4096
__device__ int mw_par_on[MW_NPAR];
__device__ float mw_par_val[MW_NPAR];
__device__ double mw_par_t[MW_NPAR];
__device__ double mw_par_bar[MW_NPAR * MW_PAR_COLS];
__device__ int mw_par_cols_overflow;
__device__ __forceinline__ float mw_parf(int k, float lit) { return mw_par_on[k] ? mw_par_val[k] : lit; }
"""
B1 = "static constexpr float b1 = 24.0f;"
OCC = "b1l = r6_mul(r6_mul(b1, 0.25f), r6_add(el(((k) + (1))), el(k)));"
g = open(gen_p).read(); f = open(fwd_p).read()
if g.count(B1) != 1 or f.count(B1) != 1: die("b1 = 24 declaration")
# derivative unit
g = sub1(g, '\n#include "mw_types.cuh"\n', '\n#include "mw_types.cuh"\n' + GLOB, "include")
g = sub1(g, "int lane_sub = 0; unsigned lane_mask = 0u;", "int lane_sub = 0; unsigned lane_mask = 0u;\nMR mw_p_b1;", "member")
g = sub1(g, OCC, OCC.replace("r6_mul(b1, 0.25f)", "r6_mul(mw_p_b1, 0.25f)"), "derivative occurrence")
g = sub1(g, "  column.active_i = active_i; column.active_j = active_j;\n",
         "  column.active_i = active_i; column.active_j = active_j;\n"
         "  int mw_pid = -1;\n"
         "  { const float pv = mw_parf(0, 24.0f);\n"
         "    if constexpr(MODE==0) { column.mw_p_b1 = mw_par_t[0] != 0.0 ? F(pv, D(mw_par_t[0], &ctx)) : F(pv); }\n"
         "    else { mw_pid = ctx.leaf(); column.mw_p_b1 = F(pv, D(mw_pid, &ctx)); } }\n", "column init")
g = sub1(g, "    ctx.reverse();\n",
         "    ctx.reverse();\n"
         "    if (mw_pid >= 0) { if (index < MW_PAR_COLS) mw_par_bar[index] += ctx.bars[mw_pid]; else mw_par_cols_overflow = 1; }\n", "reverse collect")
# forward
f = GLOB.replace("__device__ double mw_par_t[MW_NPAR];\n", "").replace("__device__ double mw_par_bar[MW_NPAR * MW_PAR_COLS];\n", "").replace("__device__ int mw_par_cols_overflow;\n", "") + f
f = sub1(f, OCC, OCC.replace("r6_mul(b1, 0.25f)", "r6_mul(mw_parf(0, b1), 0.25f)"), "forward occurrence")
open(out_d, "w").write(g); open(out_f, "w").write(f)
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
print(json.dumps({"deriv_in": sha(open(gen_p).read()), "deriv_out": sha(g), "forward_in": sha(open(fwd_p).read()), "forward_out": sha(f),
                  "param": {"slot": 0, "name": "mynn_qke_dissipation_b1", "literal": "24.0f"}}))
