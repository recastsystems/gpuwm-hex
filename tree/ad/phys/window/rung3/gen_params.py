#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# ad/rung3-params: parameter carrier for the P4 derivative modules (MYNN surface layer, GSL drag).
# Post-processes the generated P4 sources (tree/ad/phys/p4/gen-out/*.p4.cu, output of p4-gen) and makes a
# small set of recorded constants into parameters:
#   * the verbatim forward reads p4_parf(k, literal): p4_par_on[k] ? p4_par_val[k] : literal (module globals,
#     zero-initialised, so the default build computes exactly the literal: same float op on the same value);
#   * the derivative copy reads p4_parA(k, literal): TL = ADF(value, p4_par_t[k]) (tangent of parameter k,
#     module global, 0 by default); REV = a tape leaf with array id 8 + k whose adjoint the sweep adds to
#     p4_par_bar[k * P4_PAR_COLS + column] (one thread per column, fixed sweep order: deterministic; the host sums
#     columns in index order in float64).
# Every textual pattern must match its expected count, or generation stops.
import hashlib, json, sys

PARAMS = {
    "mynnsfc": [
        # slot, name, verbatim pattern (unique), literal
        (0, "coare30_charnock", "czc = __fadd_rn(0x1.6872b00000000p-7f,__fmul_rn(0x1.cac0840000000p-8f,", "0x1.6872b00000000p-7f"),
    ],
    "ugwp": [
        (0, "tofd_beljaars_a1", "a1=0.00026615161f*(var_temp*var_temp)", "0.00026615161f"),
        (1, "mesoscale_gwd_xlinv", "xlinv=coefm*8.84e-6f;", "8.84e-6f"),
        (2, "flow_blocking_cd", "taufb[1]=0.5f*1.92e-4f*rhobar", "1.92e-4f"),
    ],
}
HEAD = """// ===== rung3: parameter carrier (ad/rung3-params) =====
#define P4_NPAR 8
#define P4_PAR_COLS 4096
__device__ int p4_par_on[P4_NPAR];
__device__ float p4_par_val[P4_NPAR];
__device__ __forceinline__ float p4_parf(int k, float lit) { return p4_par_on[k] ? p4_par_val[k] : lit; }
"""
ADHEAD = """
// ===== rung3: parameter carrier, derivative side =====
#if defined(P4_MODE_TL)
__device__ double p4_par_t[P4_NPAR];
__device__ __forceinline__ ADF p4_parA(int k, float lit) { return ADF(p4_parf(k, lit), p4_par_t[k]); }
#else
__device__ double p4_par_bar[P4_NPAR * P4_PAR_COLS];
__device__ int p4_par_cols_overflow;
__device__ __forceinline__ ADF p4_parA(int k, float lit) { return p4_leaf(p4_parf(k, lit), 8 + k, 0); }
#endif
"""
def die(m): sys.exit("gen_params: " + m)
def sub1(s, old, new, what):
    n = s.count(old)
    if n != 1: die(f"{what}: matched {n}, expected 1")
    return s.replace(old, new, 1)

def main(fam, src_path, out_path):
    src = open(src_path).read()
    mark = "// ===== p4: derivative types ====="
    if src.count(mark) != 1: die("derivative-types marker")
    verb, deriv = src.split(mark)
    for slot, name, pat, lit in PARAMS[fam]:
        if pat.count(lit) != 1: die(f"{name}: literal not unique in pattern")
        verb = sub1(verb, pat, pat.replace(lit, f"p4_parf({slot}, {lit})"), f"{name} verbatim")
        dpat = pat.replace("float", "ADF")
        deriv = sub1(deriv, dpat, dpat.replace(lit, f"p4_parA({slot}, {lit})"), f"{name} derivative")
    e1 = "// ===== p4: entries part 1"
    if deriv.count(e1) != 1: die("entries part 1 marker")
    deriv = deriv.replace(e1, ADHEAD + e1, 1)
    if fam == "ugwp":
        old = "if (p.x <= -2) { TT *b = bin[-2 - p.x];"
        new = ("if (p.x <= -2) { if (p.x <= -10) { if (i < P4_PAR_COLS) p4_par_bar[(long long)(-10 - p.x) * P4_PAR_COLS + i] += a;"
               " else p4_par_cols_overflow = 1; continue; } TT *b = bin[-2 - p.x];")
        deriv = sub1(deriv, old, new, "ugwp sweep")
    else:
        deriv = sub1(deriv, "__device__ __forceinline__ void p4_sweep(const int2 *ab, const double2 *w, double *adj, int cnt, long long stride, int t) {",
                     "__device__ __forceinline__ void p4_sweep(const int2 *ab, const double2 *w, double *adj, int cnt, long long stride, int t, int c) {", "sweep signature")
        deriv = sub1(deriv, "        if (p.x <= -2) continue;\n",
                     "        if (p.x <= -2) { if (p.x <= -10) { if (c < P4_PAR_COLS) p4_par_bar[(long long)(-10 - p.x) * P4_PAR_COLS + c] += a;"
                     " else p4_par_cols_overflow = 1; } continue; }\n", "sweep leaf")
        deriv = sub1(deriv, "p4_sweep(ab, w, adj, cnt, stride, t);", "p4_sweep(ab, w, adj, cnt, stride, t, c);", "sweep call")
    out = HEAD + verb + mark + deriv
    open(out_path, "w").write(out)
    sha = lambda b: hashlib.sha256(b.encode()).hexdigest()
    print(json.dumps({"family": fam, "in": src_path, "in_sha256": sha(src), "out": out_path, "out_sha256": sha(out),
                      "params": [{"slot": s, "name": n, "literal": l} for s, n, _, l in PARAMS[fam]]}))

if __name__ == "__main__":
    if len(sys.argv) != 4: die("usage: gen_params.py <mynnsfc|ugwp> <gen-out/x.p4.cu> <out.cu>")
    main(*sys.argv[1:])
