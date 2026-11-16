#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""x20 MYNN stage M4 (updraft plume, f_dmp_mf + f_condensation_edmf): primal-only plume in the TL/AD.

Build time only. Post-processes the generated MYNN derivative TU (gen_whole.py output, --freeze closure).

Finding behind it (lane mynn-M4-plume, 2026-11-09): under --freeze closure every OUTPUT of f_dmp_mf is frozen after
the call, but f_dmp_mf also writes four live arrays inside its own body (the shallow-cumulus cloud blend,
module_bl_mynnedmf.F90:7620-7712): vt1, vq1 (buoyancy coefficients, read by f_mym_turbulence), qc_bl1, cldfra_bl1
(launch outputs). They read the plume's internal edmf_a/w/th/qt/qc and mf_at_cb before the after-call freeze, so the
derivative of the whole 8-updraft integration still reaches the operator through them, and the whole plume is taped,
counted and swept. In columns without cloudy plume levels that work is all zero-adjoint.

Stage M4 here: the plume integration (everything in f_dmp_mf before the edmf_th1 loop) runs on frozen copies of its
inputs (mw_frz: primal bitwise, no derivative, no tape node), so it is primal recompute only: no nodes, no count, no
sweep. The cloud-blend tail keeps its live dependence on the ENVIRONMENT (thl1, qt1, tk1, pres1, ex1, dz1 and the
incoming qc_bl1, cldfra_bl1, vt1, vq1) with the plume quantities (edmf_*, mf_at_cb) frozen, consistent with the
11-05 freeze ruling's "updraft plume frozen". This CHANGES the operator in cloudy plume levels (it drops d/d(plume)
in vt1/vq1/qc_bl1/cldfra_bl1); it is graded on dot and Taylor, not words. The primal (forward) is unchanged bitwise.

Usage: m4_plume.py IN.gen.cu OUT.gen.cu  > info.json
"""
import json, re, sys
# bl-live-reverse (2026-11-14): --live LIST names inputs the plume region keeps LIVE (not shadowed by a frozen copy).
# For a gen_whole.py --plume-live mom source it is u1,v1: the plume's transport of momentum answers the environment
# wind with coefficients that are all built from the frozen copies of every other input.
LIVE = []
if '--live' in sys.argv:
    _i = sys.argv.index('--live'); LIVE = [x for x in sys.argv[_i + 1].split(',') if x]; del sys.argv[_i:_i + 2]
src, dst = sys.argv[1:3]
s = open(src).read()
# Breakage this prevents: a --plume-live mom source through M4 with u1, v1 shadowed is a momentum-frozen image under a
# momentum-live name (the tape-free backward would then differ from the taped one in every plume column).
MOMHEAT = '(gen_whole.py --plume-live momheat)' in s  # mom plus heat and moisture; its live views are declared in the source
if ('(gen_whole.py --plume-live mom)' in s or MOMHEAT) and sorted(LIVE) != ['u1', 'v1']:
    raise SystemExit('m4_plume.py: source is --plume-live mom; give --live u1,v1')
if LIVE and '(gen_whole.py --plume-live mom)' not in s and not MOMHEAT:
    raise SystemExit('m4_plume.py: --live is for a --plume-live mom source (its momentum lines hold every coefficient)')
# lane GROUND (2026-11-10). Breakage this prevents: M4 runs the plume on frozen copies of its inputs, so on a
# gen_whole.py --freeze closure-plume source it would hand back a plume-frozen image under a live-plume name.
if '(gen_whole.py --freeze closure-plume)' in s:
    raise SystemExit('m4_plume.py: source is --freeze closure-plume (live plume); stage M4 freezes the plume. Compose variant A.')
lines = s.split('\n')
# locate the derivative f_dmp_mf
i0 = [i for i, l in enumerate(lines) if re.match(r'^__device__ __(forceinline|noinline)__ void f_dmp_mf\(', l)]
assert len(i0) == 1, i0
i0 = i0[0]
sig = lines[i0]
params = re.match(r'^.*?f_dmp_mf\((.*)\) \{$', sig).group(1)
ps = []
depth = 0; cur = ''
for ch in params:
    if ch == '<': depth += 1
    if ch == '>': depth -= 1
    if ch == ',' and depth == 0: ps.append(cur.strip()); cur = ''
    else: cur += ch
ps.append(cur.strip())
arrays, scalars = [], []
for p in ps:
    t, n = p.rsplit(' ', 1)
    if t.startswith('MynnArray<MR'): arrays.append(n)
    elif t == 'MR': scalars.append(n)
# function end: first line '}' at column 0 after i0
nxt = next(i for i in range(i0 + 1, len(lines)) if lines[i].startswith('__device__ ') or lines[i].startswith('// RRFS ') or lines[i].startswith('template'))
iend = max(i for i in range(i0 + 1, nxt) if lines[i] == '}')
body = list(range(i0 + 1, iend))
# outputs of f_dmp_mf (frozen after the call by gen_whole.py) and the four live in-body writes: never shadowed
written = set()
for i in body:
    m = re.match(r'^(\w+)\(.*?\) = ', lines[i])
    if m: written.add(m.group(1))
    m = re.match(r'^(\w+) = ', lines[i])
    for n in arrays:
        if re.search(r'\b' + n + r'\b(\.slice\([^;]*?\))?\.linear\([^)]*\) = ', lines[i]): written.add(n)
    if m: written.add(m.group(1))
# inputs only: read-only params (shadowing an output would lose its write)
ins_a = [a for a in arrays if a not in written and a not in LIVE]
for _n in LIVE:
    if _n not in arrays or _n in written: raise SystemExit('--live %s: not a read-only array input of f_dmp_mf' % _n)
ins_s = [a for a in scalars if a not in written]
# start: after the rebase block that follows 'MynnFrame mynn_frame(arena);'
fr = next(i for i in body if lines[i] == 'MynnFrame mynn_frame(arena);')
st = fr + 1
while lines[st].endswith('.rebase({kts});') or re.match(r'^\w+\.rebase\(', lines[st]): st += 1
# a --plume-live momheat source declares live views of thl1, qt1, qv1 here; the frozen copies and their names start after them
nlv = 0
while re.match(r'^MynnArray<MR, 1> (\w+)_lv = \1;$', lines[st]): st += 1; nlv += 1
if MOMHEAT != (nlv == 3): raise SystemExit('m4_plume.py: momheat source and its live views do not agree (%d views)' % nlv)
# end: the loop header that holds the first edmf_th1(k) = line
th = next(i for i in body if lines[i].startswith('edmf_th1(k) = '))
en = th - 1
while not lines[en].startswith('{ const int mynn_lo'): en -= 1
# guard: no member access or call-with-same-name of a shadowed token inside the region
reg = '\n'.join(lines[st:en])
for n in ins_a + ins_s:
    if re.search(r'[.>]\s*' + n + r'\b', reg): raise SystemExit('member access to ' + n)
# guard: the region never writes a shadowed input
for n in ins_a + ins_s:
    if re.search(r'^\s*' + n + r'(\(.*?\))? = ', reg, re.M): raise SystemExit('region writes input ' + n)
pre = ['// x20 M4 (m4_plume.py): plume integration on frozen inputs (primal only, no tape nodes).']
for a in ins_a: pre.append('MynnArray<MR, %d> %s_m4f = mw_m4_frzcopy<mynn_tile>(arena, %s);' % (
    int(re.search(r'MynnArray<MR, (\d)>', next(p for p in ps if p.endswith(' ' + a))).group(1)), a, a))
for a in ins_s: pre.append('MR %s_m4f = mw_frz(%s);' % (a, a))
for a in ins_a + ins_s: pre.append('#define %s %s_m4f' % (a, a))
post = ['// x20 M4: end of the frozen plume; the cloud-blend tail reads the live environment.']
for a in ins_a + ins_s: post.append('#undef %s' % a)
lines = lines[:en] + post + lines[en:]
lines = lines[:st] + pre + lines[st:]
helper = r'''
// x20 M4 (m4_plume.py): a frozen copy of an input array, same index range, in the column's arena.
template<int TILE, class T, int R> __device__ __forceinline__ MynnArray<T, R> mw_m4_frzcopy(MynnArena& arena, MynnArray<T, R> a) {
    if (!a.data) return a;
    MynnIndices<R> lo, hi; bool empty = false;
    for (int d = 0; d < R; ++d) { lo.value[d] = a.lo[d]; hi.value[d] = a.hi[d]; if (a.hi[d] < a.lo[d]) empty = true; }
    if (empty) return a;
    MynnArray<T, R> r = mynn_array_any<T, R, TILE>(arena, lo, hi);
    if constexpr (R == 1) { for (int k = a.lo[0]; k <= a.hi[0]; ++k) r(k) = mw_frz(a(k)); }
    else { static_assert(R == 2, "m4 copy rank"); for (int c = a.lo[1]; c <= a.hi[1]; ++c) for (int k = a.lo[0]; k <= a.hi[0]; ++k) r(k, c) = mw_frz(a(k, c)); }
    return r;
}
'''
out = '\n'.join(lines)
# helper goes right after the mynn_array_any template definition (it needs it, and MynnArena)
k = out.index('template<class T,int R,int TILE> __device__ MynnArray<T,R> mynn_array_any(')
k = out.index('\n}\n', k) + 3
out = out[:k] + helper + out[k:]
open(dst, 'w').write(out)
json.dump({'stage': 'M4', 'live_inputs': LIVE, 'live_views': nlv, 'frozen_inputs_arrays': ins_a, 'frozen_inputs_scalars': ins_s,
           'region_lines': [st, en], 'not_shadowed_written': sorted(written & set(arrays + scalars))}, sys.stdout, indent=1)
