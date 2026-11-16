#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""x20 MYNN M3: patch a generated P3 unit (gen_whole.py output, e.g. b8d132a0) so the tape entries (MynnTapeF:
mynn_whole_ad, mynn_whole_count) run f_mym_condensation (bl_mynn_cloudpdf = 2) through the M3 stage (m3_cond.cuh,
proof hook m3_swap.cuh). The tangent entry is unchanged. Build time only.
Usage: patch_m3.py IN.gen.cu OUT.gen.cu"""
import hashlib, sys
src, out = sys.argv[1], sys.argv[2]
s = open(src).read()
inc = '#include "mw_types.cuh"\n'
assert s.count(inc) == 1
s = s.replace(inc, inc + '#include "m3_cond.cuh"\n#include "m3_swap.cuh"\n', 1)
sig = '__device__ __forceinline__ void f_mym_condensation(int kts, int kte, MR dx,'
i = s.index(sig)
assert s.count(sig) == 1
anchor = 'pattern_spp_pbl1.rebase({kts});\n'
j = s.index(anchor, i) + len(anchor)
assert s.index('\n}\n', i) > j, 'anchor outside f_mym_condensation'
hook = ('#ifndef MW_M3_OFF\n'
        'if constexpr (mw_same<MR, MynnTapeF>::v) { if (bl_mynn_cloudpdf == 2) { m3_tape_swap(kts, kte, xland, dz, zw, qc, qi, qs, p, '
        'exner, tsq, qsq, cov, qw, th, pattern_spp_pbl1, qc_bl1, qi_bl1, cldfra_bl1, vt, vq, sgm, pblh1, closure, spp_pbl); return; } }\n'
        '#endif\n')
s = s[:j] + hook + s[j:]
open(out, 'w').write(s)
print(hashlib.sha256(open(src, 'rb').read()).hexdigest()[:8], '->', hashlib.sha256(s.encode()).hexdigest()[:8])
