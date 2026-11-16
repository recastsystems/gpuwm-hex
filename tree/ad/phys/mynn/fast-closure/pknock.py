#!/usr/bin/env python3
# knockouts of the boundary-layer height's uses in a height-live source (generator mask with bit 1 clear).
#   pknock.py BASE.gen.cu OUT.gen.cu KEEP      KEEP = letters of the sites that stay live (others read a frozen copy):
#     A  f_mym_level2: the weight of clam (zw against pblh + 100, 0.3 pblh)
#     C  f_mym_turbulence: the weight of the Prandtl limit (zw / max(pblh, 200))
#     D  f_mym_condensation: every weight formed from pblh (pblh2 = max(pblh1, 10) and the pblh1 - 100 weight)
#     O  OUTPUT ONLY: the height is frozen for every use inside the column right after f_get_pblh and its live value is put
#        back at the end of f_mynnedmf, so only the column's pblh output (what the surface layer and the other schemes read at
#        the next step) answers. With O the letters A, C, D are ignored (those uses read the frozen value).
#     N  none of A, C, D (the length, the scale-aware weights, the plume and the output keep what the source gives them)
# Always frozen here: the qpe weight (dead at closure 2.6). Left as the source has it: the tendencies' aerosol weight (stage M9 parses
# that routine and takes no frozen copy; it moves the aerosol number over water only) and the plume's uses (stage M4 freezes them).
import sys, re
src, out, keep = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(src).read()
def fn_span(name):
    m = re.search(r'^__device__[^\n]*\b%s\(' % name, s, re.M); a = m.start()
    n = re.search(r'^__device__', s[a + 10:], re.M); return a, (a + 10 + n.start()) if n else len(s)
def rep(fn, old, new, count=1):
    global s
    a, b = fn_span(fn); body = s[a:b]
    if body.count(old) != count: raise SystemExit('%s: %r found %d times, wanted %d' % (fn, old[:70], body.count(old), count))
    s = s[:a] + body.replace(old, new) + s[b:]
if 'O' in keep:
    a, b = fn_span('f_mynnedmf'); body = s[a:b]
    NL = chr(10); anc = 'sm1.rebase({kts});' + NL; frz = 'if (MW_FRZ_ON(1)) { pblh = mw_frz(pblh); }'
    if body.count(anc) != 1 or body.count(frz) != 2: raise SystemExit('O: anchors %d %d' % (body.count(anc), body.count(frz)))
    body = body.replace(anc, anc + 'MR mw_pblh_live = pblh;' + NL).replace(frz, 'mw_pblh_live = pblh; pblh = mw_frz(pblh);')
    e = body.rfind(NL + '}' + NL) + 1
    if e <= 0: raise SystemExit('O: end of f_mynnedmf not found')
    body = body[:e] + 'pblh = mw_pblh_live;' + NL + body[e:]
    s = s[:a] + body + s[b:]
    open(out, 'w').write(s); print('pknock keep=O (output only)'); sys.exit(0)
A_OLD = 'wt = mynn_min((MR)(r6_div(mynn_max((MR)(r6_sub(zw(k), r6_add(pblh, hundred))), (MR)(zero)), mynn_max((MR)(r6_mul(p3, pblh)), (MR)(400.f)))), (MR)(one));'
C_OLD = 'wt = mynn_min((MR)(r6_div(zw(k), mynn_max((MR)(pblh), (MR)(200.f)))), (MR)(one));'
B_OLD = 'pblh2 = mynn_max((MR)(pblh), (MR)(200.f));'
D1 = 'pblh2 = mynn_max((MR)(pblh1), (MR)(ten));'
D2 = 'r6_sub(zagl, r6_sub(pblh1, 100.f))'
if 'A' not in keep: rep('f_mym_level2', A_OLD, A_OLD.replace('pblh', 'mw_frz(pblh)'))
if 'C' not in keep: rep('f_mym_turbulence', C_OLD, C_OLD.replace('(MR)(pblh)', '(MR)(mw_frz(pblh))'))
rep('f_mym_turbulence', B_OLD, B_OLD.replace('(MR)(pblh)', '(MR)(mw_frz(pblh))'))
if 'D' not in keep:
    rep('f_mym_condensation', D1, D1.replace('(MR)(pblh1)', '(MR)(mw_frz(pblh1))'))
    rep('f_mym_condensation', D2, D2.replace('pblh1', 'mw_frz(pblh1)'))
open(out, 'w').write(s)
print('pknock keep=%s' % keep)
