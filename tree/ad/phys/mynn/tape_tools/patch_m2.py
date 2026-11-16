# SPDX-License-Identifier: Apache-2.0
# x20 MYNN stage M2 swap-in (proof of the fused stage inside today's taped kernel): the frozen routines
# f_get_pblh and f_scale_aware run on primal floats (cuda/mynn_m2_pbl.cuh) and hand back constant carriers, so the
# record and count passes make no nodes for them. Usage: patch_m2.py in.gen.cu out.gen.cu
import sys, re
src, dst = sys.argv[1:3]
s = open(src).read()
inc = '#include "mw_types.cuh"\n'
assert s.count(inc) == 1, 'mw_types include'
s = s.replace(inc, inc + '#include "mynn_m2_pbl.cuh"\n')
n = 0
for old, new in [
    ('f_get_pblh(kts, kte, pblh, thv1, qke1, ust, zw1, dz1, xland, kpbl);',
     'm2_get_pblh_mr(kts, kte, pblh, thv1, ust, zw1, dz1, xland, kpbl);'),
    ('f_scale_aware(dx, pblh, psig_bl, psig_shcu);', 'm2_scale_aware_mr(dx, pblh, psig_bl, psig_shcu);')]:
    c = s.count(old)
    assert c == 2, (old, c)
    s = s.replace(old, new); n += c
open(dst, 'w').write(s)
print('patched call sites', n)
