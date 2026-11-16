#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# x20 mynn-M9 (2026-11-09): splice the generated M9 stage (gen_m9.py) into a gen_whole.py output so the taped P3
# entries (MODE 1 record+reverse, MODE 2 count) take stage M9 as one external node instead of its ~125k tape nodes.
# The tangent entry (MODE 0) and every other stage are unchanged. Build-time only.
# usage: patch_m9.py IN.gen.cu STAGE.cuh OUT.gen.cu
import re, sys
src, stage, dst = sys.argv[1:4]
s = open(src).read()
st = open(stage).read()
structs = open(stage + '.structs').read()

# 1) the stage's parameter structs, before the column template (MynnArray is defined above it)
a = 'template<int MYNN_LANES, class MR=float, class MRD=double> struct MynnColumn {'
assert s.count(a) == 1
s = s.replace(a, structs + a, 1)

# 2) the stage member functions, right before f_mynn_tendencies, as noinline members
m = re.search(r'^__device__ __forceinline__ void f_mynn_tendencies\((.*)\) \{$', s, re.M)
assert m
params = [p.strip().split(' ')[-1] for p in re.split(r',\s*(?![^<]*>)', m.group(1))]
st2 = st.replace('__device__ void m9_', '__device__ __noinline__ void m9_').replace('__device__ bool m9_', '__device__ __noinline__ bool m9_')
s = s[:m.start()] + st2 + '\n' + s[m.start():]

# 3) the switch at the top of f_mynn_tendencies (taped carriers only; mixqt > 0 keeps the tape: f_qsat_blend path
#    not generated; kts == 1 because the inlined callees index from 1; one lane per column)
m = re.search(r'^__device__ __forceinline__ void f_mynn_tendencies\((.*)\) \{$', s, re.M)
sw = ('\n#ifdef M9_EXT\nif constexpr (mw_same<MR, MynnTapeF>::v) { if (bl_mynn_mixqt == 0 && kts == 1 && lane_width == 1) { '
      'm9_ext_record(%s); return; } }\n#endif\n') % ', '.join(params)
s = s[:m.end()] + sw + s[m.end():]

# 4) the reverse sweep stops at the stage marker and hands it to the column
a = '    ctx.reverse();\n'
assert s.count(a) == 1
hook = ('#ifdef M9_EXT\n'
        '    { long long m9_t0 = clock64(), m9_ts = 0; int mw_i = ctx.used - 1;\n'
        '      while ((mw_i = ctx.reverse_until(mw_i)) >= 0) { const long long m9_t1 = clock64();\n'
        '        if (!column.m9_ext_reverse(ctx, mw_i)) ctx.fail(3); m9_ts += clock64() - m9_t1; --mw_i; }\n'
        '#ifdef M9_PROF\n'
        '      if ((col_local & 63) == 0) printf("M9PROF col %d nodes %d sweep_cyc %lld m9_rev_cyc %lld\\n", col_local, ctx.used, clock64() - m9_t0 - m9_ts, m9_ts);\n'
        '#endif\n'
        '      (void)m9_t0; (void)m9_ts; }\n'
        '#else\n')
base_prof = ('#ifdef M9_PROF\n'
             '    { const long long m9_t0 = clock64();\n' + a +
             '      if ((col_local & 63) == 0) printf("M9PROF col %d nodes %d sweep_cyc %lld m9_rev_cyc 0\\n", col_local, ctx.used, clock64() - m9_t0); }\n'
             '#else\n' + a + '#endif\n')
s = s.replace(a, hook + base_prof + '#endif\n', 1)
open(dst, 'w').write(s)

# 5) the tape: reserve a run of slots (marker + record + output leaves) and a sweep that stops at a marker
tp = sys.argv[4] if len(sys.argv) > 4 else None
if tp:
    t = open(tp).read()
    a = '    __device__ int leaf() { return push(-1, 0.0, -1, 0.0); }'
    assert t.count(a) == 1
    t = t.replace(a, a + '''
    // x20 mynn-M9: n consecutive slots for an external stage (marker, record, output leaves). Count mode only counts.
    __device__ int ext_reserve(int n) {
        if (used + n > capacity || used < 0 || used >= (1 << 29) - n) { fail(0); return -1; }
        const int id = used; used += n; return id;
    }''', 1)
    a = '    __device__ void reverse() {'
    i = t.index(a)
    j = t.index('\n    }\n', i) + 7
    body = t[i:j]
    rb = body.replace('__device__ void reverse() {', '__device__ int reverse_until(int start) {')
    rb = rb.replace('for (int i = used - 1; i >= 0; --i) {', 'for (int i = start; i >= 0; --i) {')
    rb = rb.replace('            MynnAdNode n = nodes[i];\n', '            MynnAdNode n = nodes[i];\n            if (n.left == -2) return i;   // x20 mynn-M9: external stage marker (bar 1.0)\n', 1)
    assert rb.count('if (n.left == -2)') == 1
    rb = rb.rstrip('\n')
    assert rb.endswith('}')
    rb = rb[:-1] + '    return -1;\n    }\n'
    t = t[:j] + rb + t[j:]
    open(tp, 'w').write(t)
