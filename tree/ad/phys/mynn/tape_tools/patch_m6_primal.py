#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# x20 MYNN stage M6 (length-frozen), 2026-11-09: patch a generated mynn_whole.gen.cu (gen_whole.py --freeze closure) so
# the frozen outputs of f_mym_level2 (sm, sh) and f_mym_length/f_boulac_length (el) are computed primal-only
# (cuda/stage_m6_length.cuh) instead of taped and swept at zero adjoint. Live outputs and call signatures unchanged.
# usage: patch_m6_primal.py IN.gen.cu OUT.gen.cu
import re, sys
src = open(sys.argv[1]).read()
freeze = re.search(r'"f_mym_level2"', src) is None  # gen-info is not in the source; check the freeze statements instead
assert 'mw_frz_arr(sm); mw_frz_arr(sh);' in src and 'mw_frz_arr(el);' in src, "not a --freeze closure source"

# 1) include the stage inside MynnColumn, right before the generated f_mym_level2 definition
m = re.search(r'^__device__ __\w+ void f_mym_level2\(', src, re.M)
assert m
src = src[:m.start()] + '#include "stage_m6_length.cuh"\n' + src[m.start():]

# 2) f_mym_level2: drop the ri statements of the gradient loops (ri feeds only sm/sh) and replace the closure loop
m = re.search(r'^__device__ __\w+ void f_mym_level2\(.*\) \{$', src, re.M)
s = m.end(); d = 1; e = s
while d:
    c = src[e]; d += (c == '{') - (c == '}'); e += 1
body = src[s:e]
ri_stmt = 'ri(k) = r6_neg(r6_div(gh(k), mynn_max((MR)(duz), (MR)(1.0e-10f))));\n'
assert body.count(ri_stmt) == 2, body.count(ri_stmt)
body = body.replace(ri_stmt, '')
i = body.index('{ const int mynn_lo2 = (((kts) + (1)))')
j = body.index('\n', body.index('} mynn_lane_sync(); k = mynn_lo2', i)) + 1
assert 'sh(k) = ' in body[i:j] and 'sm(k) = ' in body[i:j]
body = body[:i] + 'm6_level2_closure(kts, kte, gm, gh, sm, sh); // x20 M6: closure primal-only\n' + body[j:]
src = src[:s] + body + src[e:]

# 3) both f_mym_length call sites (f_mym_initialize, f_mym_turbulence) go through the stage
n = len(re.findall(r'^f_mym_length\(', src, re.M))
assert n == 2, n
src = re.sub(r'^f_mym_length\(', 'm6_length(', src, flags=re.M)
open(sys.argv[2], 'w').write(src)
print("patched: level2 closure + %d length call sites" % n)
