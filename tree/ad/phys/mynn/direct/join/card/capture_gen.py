#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""the card forward with captures, item 1 of the bl-card page.

Writes the source of `mynn_capture_l1`, the hex MYNN-EDMF forward (the source the 3 km program's P3 row binds,
sha256 70da4af7...) with one lane a column, whose probes write the cut-entry words of the joined backward straight
into a ColumnFrame<float> per column (no tape, no jprobe::Frame). The probe points and anchors are code/probes.py's
(the points the CPU oracle and the CPU route with no tape use); where each word goes is fill_frame()'s mapping in
code/join_test.cpp, written out below. Byte offsets come from card/frame_layout.cpp (the CPU build of join.hpp).
Probes only read model values: the forward's own arithmetic is unchanged, and with jcap null (every other entry)
no probe writes. The caller zeroes each frame before the launch (as fill_frame starts from ColumnFrame{}).

usage: capture_gen.py FWD.cu LAYOUT.json OUT.cu
"""
import json, re, sys, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import probes                                    # P: point -> scope, anchor, occurrence, where

src, lay, out = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3])
u = src.read_text(); FR = lay['frame']; K = lay['const']; MAX = K['MAX']; SNI = K['SNI']
N1 = 'jcap_n'                                    # n = kte, set by the capture entry
B = {'1': '1', 'n': N1, 'n+1': '(' + N1 + '+1)', '2': '2'}


def prof(member, slot, var, lo='1', hi='n', base=None):
    off = (FR[member] if base is None else base) + 4 * slot * MAX
    return ('for(int jk=max(%s,(%s).lo[0]);jk<=min(%s,(%s).hi[0]);++jk)jcf(%dULL+4ULL*jk,(float)(%s(jk)));'
            % (B[lo], var, B[hi], var, off, var))


def scal(off, var, kind='f'):
    return {'f': 'jcf(%dULL,(float)(%s));', 'd': 'jcd(%dULL,(double)(%s));', 'i': 'jci(%dULL,(int)(%s));'}[kind] % (off, var)


def solve(j, field):
    return FR['solve'] + 4 * (j * SNI + K[field] * MAX)


M = {}  # point -> code
M['D_pre'] = ''.join(prof('pre', j, v) for j, v in enumerate('qv1 qc1 qi1 qs1'.split()))
M['D_post'] = ''.join(prof('post', j, v) for j, v in enumerate('qv1 qc1 qi1 qs1 rqvblten1 rqcblten1 rqiblten1 rqsblten1'.split()))
M['M_drv'] = ''.join(prof('drv', j, v) for j, v in enumerate('dz1 sqv1 sqc1 sqi1 sqs1 th1 ex1 rho1'.split())) + \
    ''.join(scal(FR['drv'] + 4 * (8 * MAX + j), v) for j, v in enumerate('ts qfx hfx ust xland'.split()))
M['M_cond'] = ''.join(prof('cloud', j, v, hi='n+1' if j == 0 else 'n') for j, v in enumerate(
    'zw1 dz1 th1 ex1 pres1 sqw1 qsq1 tsq1 cov1 sqi1 sqs1 sqc1 pattern_spp_pbl1 sgm1 thl1 sqv1 sh1 el1'.split())) + \
    scal(FR['ccfg_pblh1'], 'pblh') + scal(FR['ccfg_xland'], 'xland')
M['M_dmpx'] = ('{int jpa=0;for(int jk=max(1,(edmf_a1).lo[0]);jk<=min(%s,(edmf_a1).hi[0]);++jk)if((double)(edmf_a1(jk))>0.0)jpa=1;'
               'jci(%dULL,jpa);}' % (N1, FR['plume_active']))
TAIL = 'edmf_a1 edmf_th1 edmf_qt1 edmf_qc1 edmf_w1 dz1 tk1 pres1 qt1 ex1 thl1 cldfra_bl1 qc_bl1 vt1 vq1'.split()
M['M_tail'] = ''.join('for(int jk=max(1,(%s).lo[0]);jk<=min(%s,(%s).hi[0]);++jk)jcf(%dULL+4ULL*(15ULL*(jk-1)+%d),(float)(%s(jk)));'
                      % (v, N1, v, FR['tail'], j, v) for j, v in enumerate(TAIL)) + \
    'jcf(%dULL+4ULL*15ULL*%s,(float)(landsea));jci(%dULL,((int)((double)(ktop)))-1);' % (FR['tail'], N1, FR['ktop'])
M['M_blend'] = ''.join(prof('blend', j, v) for j, v in enumerate('qc_bl1 qi_bl1 sqc1 sqi1 sqs1 th1 ex1 sqv1'.split()))
M['T_l2'] = ''.join(prof('ess', j, v, hi='n+1' if j == 1 else 'n') for j, v in enumerate(
    'dz zw u v thl thlv theta p exner qke cldfra_bl1 edmf_a1 edmf_w1 qw thv vt vq'.split())) + \
    scal(FR['ess'] + 4 * 17 * MAX, 'pblh') + scal(FR['ess'] + 4 * (17 * MAX + 1), 'xland') + \
    prof('qkwx', 1, 'dz') + scal(FR['prod'] + 4 * (5 * MAX + 1), 'dz(1)')
M['T_len'] = prof('qkwx', 0, 'qke')
M['M_pred'] = ''.join(prof('pred', j, v, hi='n+1' if j == 4 else 'n') for j, v in enumerate(
    'dz1 rho1 el1 dfq1 s_aw1 pdk1 pdq1 pdt1 pdc1 qke1 qsq1'.split())) + \
    scal(FR['pred'] + 4 * 11 * MAX, 'ust') + scal(FR['pred'] + 4 * (11 * MAX + 1), 'pmz')
M['M_heatq'] = ''.join(prof('diss', j, v) for j, v in enumerate('qke1 el1 pres1'.split()))
HEAT = probes.HEAT.split()
assert len(HEAT) == 29
M['N_ent'] = ''.join(prof('heat', j, v) for j, v in enumerate(HEAT)) + \
    ''.join(scal(FR['heat'] + 4 * (K['NHF'] * MAX + j), v) for j, v in enumerate('psfc flt flqc flqv'.split()))
Q = 'u v qnc qni qnwfa qnifa'.split(); FU = 's_awu1 s_awv1 s_awqnc1 s_awqni1 s_awqnwfa1 s_awqnifa1'.split()
FD = 'sd_awu1 sd_awv1 sd_awqnc1 sd_awqni1 sd_awqnwfa1 sd_awqnifa1'.split()
for j in range(6):
    c = prof(None, 0, 'dz', base=solve(j, 'DZ')) + prof(None, 0, 'rho', base=solve(j, 'RHO')) + \
        prof(None, 0, 'dfm' if j < 2 else 'dfh', base=solve(j, 'DF')) + prof(None, 0, 's_aw1', hi='n+1', base=solve(j, 'SA')) + \
        prof(None, 0, 'sd_aw1', hi='n+1', base=solve(j, 'SD')) + prof(None, 0, Q[j], base=solve(j, 'Q')) + \
        prof(None, 0, FU[j], hi='n+1', base=solve(j, 'FU')) + prof(None, 0, FD[j], hi='n+1', base=solve(j, 'FD'))
    if j == 0: c += prof(None, 0, 'sub_u', base=solve(j, 'SUB')) + prof(None, 0, 'det_u', base=solve(j, 'DET'))
    if j == 1: c += prof(None, 0, 'sub_v', base=solve(j, 'SUB')) + prof(None, 0, 'det_v', base=solve(j, 'DET'))
    if j == 2: c += prof(None, 0, 'sqc', base=solve(j, 'COLD'))
    if j == 3: c += prof(None, 0, 'sqi', base=solve(j, 'COLD'))
    c += prof(None, 0, 'zw', hi='n+1', base=solve(j, 'ZW')) + prof(None, 0, 'tk', base=solve(j, 'TK')) + prof(None, 0, 'qv', base=solve(j, 'QV'))
    for fld, v in (('PS', 'psfc'), ('UST', 'ust'), ('WSPD', 'wspd'), ('OCE', 'voce' if j == 1 else 'uoce'), ('PBLH', 'pblh'), ('LAND', 'xland')):
        c += scal(FR['solve'] + 4 * (j * SNI + K[fld]), v)      # K[fld] is the word's index in the solve input
    M['N_ent'] += c
M['N_c2'] = prof(None, 0, 'sqc2', base=solve(2, 'CNEW')) + prof(None, 0, 'sqi2', base=solve(3, 'CNEW'))
# production inputs at level k inside f_mym_turbulence's loop (fill_frame packs levels 2..n)
PROD = 'qkw gm gh dtl dqw dz el sm sh'.split()
prodc = 'if(k>=2&&k<=%s){' % N1 + \
    ''.join('jcf(%dULL+4ULL*k,(float)(%s(k)));' % (FR['prod'] + 4 * j * MAX, v) for j, v in enumerate(PROD)) + \
    ''.join('jcf(%dULL+4ULL*k,(float)(%s));' % (FR['prod'] + 4 * (9 + j) * MAX, v) for j, v in enumerate('gamt gamq gamv'.split())) + \
    'jcd(%dULL+8ULL*k,(double)(qdiv));' % FR['qdiv'] + \
    ''.join('jcf(%dULL+4ULL*k,(float)(%s(k)));' % (FR['prod'] + 4 * (12 + j) * MAX, v) for j, v in enumerate('tkeprod_dn tkeprod_up'.split())) + '}'

unused = sorted(set(probes.P) - set(M))   # oracle-only points (D_end, M_dmp, M_turb, M_tend, M_tendx, T_lenx)


def func_span(u, name):
    m = re.search(r'\n[^\n]*\b' + name + r'\([^\n]*\) \{', u)
    assert m, name
    a = m.start() + 1; b = m.end(); depth = 1
    while depth: depth += (u[b] == '{') - (u[b] == '}'); b += 1
    return a, m.end(), b


edits = []
for p, code in M.items():
    d = probes.P[p]; a, bodystart, b = func_span(u, d['scope'])
    if d['where'] == 'after-sig':
        edits.append((bodystart, '\n' + code + '\n')); continue
    body = u[a:b]; i = -1
    for _ in range(d['occ'] + 1):
        i = body.index(d['anchor'], i + 1)
    pos = a + i
    if d['where'] == 'after':
        pos = u.index(';\n', pos) + 2
    edits.append((pos, code + '\n'))
a, _, b = func_span(u, 'f_mym_turbulence'); i = u.index('elq = r6_mul(el(k), qkw(k));', a); assert i < b
edits.append((i, prodc + '\n'))
# a layout built with -DJOIN_MIX has the mixing block (join.hpp MixJac). The probes below write
# the level-2 sm, sh right after f_mym_level2 in f_mym_turbulence (read back by the next probe), and at the production
# probe of the loop qtw(k) and the 2 x 10 Jacobian of the final stability functions (blocks/mixjac.hpp, inlined at the top
# of the source), zero where the weak-shear rule holds: (double)gm(k) < MIX_WS_T (environment MIX_WS_T, default 1e-4,
# the generator's --weakshear threshold; "none" = no rule). A level where the mirror does not give the forward's own
# sm(k), sh(k) bit for bit sets `bad`, and the joined backward refuses the column (Status::Mix: the taped route).
MIX = 'mx' in FR
if MIX:
    import os
    WS = os.environ.get('MIX_WS_T', '1e-4')
    WST = '-1.0' if WS == 'none' else repr(float(WS))
    CLOS = 'mx_jq' in FR                        # a layout built with -DJOIN_CLOSURE (the closure live)
    a, _, b = func_span(u, 'f_mym_turbulence'); fbody = u[a:b]
    l2 = fbody.index('f_mym_level2(kts, kte, bl_mynn_ess'); l2e = fbody.index(';\n', l2) + 2
    edits.append((a + l2e, ('if(jcap){for(int jk=2;jk<=min(%s,(sm).hi[0]);++jk){jcf(%dULL+4ULL*jk,(float)(sm(jk)));jcf(%dULL+4ULL*jk,(float)(sh(jk)));}}\n'
                            % (N1, FR['mx_sm2'], FR['mx_sh2']))))
    mixc = ('if(jcap&&k>=2&&k<=%s){jcf(%dULL+4ULL*k,(float)(qtw(k)));'
            'const float jsm2=*reinterpret_cast<const float*>(jcap+%dULL+4ULL*k),jsh2=*reinterpret_cast<const float*>(jcap+%dULL+4ULL*k);'
            'mixjac::In jx{qkw(k),gm(k),gh(k),u(k),u(((k)-(1))),v(k),v(((k)-(1))),dz(k),dz(((k)-(1))),zw(k),cldfra_bl1(k),cldfra_bl1(((k)-(1))),'
            'el(k),jsm2,jsh2,pblh,edmf_a1(k),edmf_a1(((k)-(1))),edmf_w1(k),edmf_w1(((k)-(1))),edmf_a_dd1(k),edmf_w_dd1(k),tkeprod_dn(k),closure,tke_budget};'
            'const mixjac::Con jc{b1,a1,a2,c1,e1c,e2c,e3c,e4c,e5c};mixjac::F jsm,jsh;'
            + ('const int jok=mixjac::stab(jx,jc,jsm,jsh);' if not CLOS else
               # the level-2 functions live (their mirror must give the forward's level-2 words) and the qdiv row
               'mixjac::F jl2m,jl2h;mixjac::D jqd;const mixjac::L2Con jl2c{g1,g2,b1,a1,a2,c1,c2,c5,((ckmod)==(1))?1:0};'
               'mixjac::level2(gm(k),gh(k),jl2c,jl2m,jl2h);'
               'if(__float_as_uint(jl2m.v)!=__float_as_uint(jsm2)||__float_as_uint(jl2h.v)!=__float_as_uint(jsh2))jcor(%dULL,2);'
               'const int jok=mixjac::stab3(jx,jc,jl2m,jl2h,jsm,jsh,jqd);' % FR['mx_bad']) +
            'if(!jok||__float_as_uint(jsm.v)!=__float_as_uint((float)(sm(k)))||__float_as_uint(jsh.v)!=__float_as_uint((float)(sh(k))))jcor(%dULL,1);'
            'const bool jhold=(double)(gm(k))<(double)(%s);'
            'for(int ji=0;ji<10;++ji){jcd(%dULL+8ULL*(20ULL*k+ji),jhold?0.0:jsm.d[ji]);jcd(%dULL+8ULL*(20ULL*k+10+ji),jhold?0.0:jsh.d[ji]);'
            + ('jcd(%dULL+8ULL*(10ULL*k+ji),jhold?0.0:jqd.d[ji]);' % FR['mx_jq'] if CLOS else '') + '}}')
    mixc = mixc % (N1, FR['mx_qtw'], FR['mx_sm2'], FR['mx_sh2'], FR['mx_bad'], WST, FR['mx_jac'], FR['mx_jac'])
    i = u.index('elq = r6_mul(el(k), qkw(k));', a); assert i < b
    edits.append((i, mixc + '\n'))
# a layout built with -DJOIN_PMOM has the plume-momentum block (join.hpp PlumeMom). The probes
# below write its coefficients: the values gen_whole.py --plume-live mom holds (mw_frz) on the plume's momentum lines.
PMOM = 'pm' in FR
if PMOM:
    assert FR['pm_nup'] == 8
    a, _, b = func_span(u, 'f_dmp_mf'); fbody = u[a:b]
    # The plume constants the transpose holds (F42). join.hpp's pmom_reverse (and the heat and moisture blocks) take
    # cdet as 1/45 and have no term for the plume's pressure-gradient (shear) term dxsa*pgfac*(uk - ukm1), which the
    # forward's wind lines carry. THE BREAKAGE: a forward with pgfac != 0 or another cdet gives a joined backward that is
    # the transpose of no tangent, silently (its probes still match). Refused by name before any byte is written.
    def plume_const(name):
        hits = re.findall(r'constexpr float %s = ([^;]+);' % name, fbody)
        if len(hits) != 1:
            sys.exit('capture_gen: REFUSED: f_dmp_mf defines the plume constant %s %d times (want 1); the plume-momentum '
                     'transpose holds it and cannot be checked against this forward' % (name, len(hits)))
        txt = re.sub(r'(?<=[0-9.])f\b', '', hits[0])
        if not re.fullmatch(r'[0-9.()/*+\- e]+', txt):
            sys.exit('capture_gen: REFUSED: the plume constant %s = %s is not a literal this check reads' % (name, hits[0]))
        return hits[0], eval(txt, {'__builtins__': {}})
    import struct
    f32 = lambda x: struct.unpack('f', struct.pack('f', x))[0]
    (pg_txt, pg), (cd_txt, cd) = plume_const('pgfac'), plume_const('cdet')
    if pg != 0.0:
        sys.exit('capture_gen: REFUSED: the forward sets pgfac = %s; the plume-momentum transpose (join.hpp pmom_reverse) '
                 'has no term for dxsa*pgfac*(uk - ukm1), so its backward would leave that term out' % pg_txt)
    if f32(cd) != f32(f32(1.0) / f32(45.0)):
        sys.exit('capture_gen: REFUSED: the forward sets cdet = %s; the plume transposes (join.hpp) hold cdet = 1/45' % cd_txt)
    IPK = '((unsigned long long)(ip-1)*%dULL+(unsigned long long)k)' % MAX

    def aft(line, code):
        assert fbody.count(line + '\n') == 1, (fbody.count(line + '\n'), line[:70])
        edits.append((a + fbody.index(line + '\n') + len(line) + 1, code + '\n'))
    aft('upu(2, ip) = r6_div(r6_add(r6_mul(u1(kts), dz1(((kts) + (1)))), r6_mul(u1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));',
        'jcor(%dULL,1<<(ip-1));jcf(%dULL,(float)(dz1(kts)));jcf(%dULL,(float)(dz1(((kts)+(1)))));' % (FR['pm_started'], FR['pm_dz1'], FR['pm_dz2']))
    aft('un = r6_add(r6_add(r6_mul(upu(k, ip), r6_sub(one, entexm)), r6_mul(u1(k), entexm)), r6_mul(r6_mul(dxsa, pgfac), r6_sub(uk, ukm1)));',
        'jcf(%dULL+4ULL*%s,(float)(r6_sub(one, entexm)));jcf(%dULL+4ULL*%s,(float)(entexm));' % (FR['pm_om'], IPK, FR['pm_en'], IPK))
    aft('envm_u(k) = r6_add(envm_u(k), r6_mul(r6_mul(r6_mul(r6_sub(r6_mul(p5, r6_add(un, upu(k, ip))), u1(k)), detrateuv), aratio), mynn_min((float)(dzp), (float)(dzpmax))));',
        'jcd(%dULL+8ULL*%s,((double)(detrateuv)*(double)(aratio))*(double)(mynn_min((float)(dzp), (float)(dzpmax))));' % (FR['pm_dc'], IPK))
    aft('upu(((k) + (1)), ip) = un;', 'jcor(%dULL+4ULL*k,1<<(ip-1));' % FR['pm_pass'])
    aft('s_awu1(k) = r6_add(s_awu1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upu(k, ip)), psig_w));',
        'jcf(%dULL+4ULL*%s,(float)(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip))));jcf(%dULL,(float)(psig_w));' % (FR['pm_c1'], IPK, FR['pm_psig']))
    aft('for (int mynn_element=0; mynn_element < (s_awu1).size(); ++mynn_element) { s_awu1.linear(mynn_element) = r6_mul(s_awu1.linear(mynn_element), adjustment); }',
        'jcf(%dULL,(float)(adjustment));jci(%dULL,1);' % (FR['pm_adj'], FR['pm_adj_set']))
    aft('sub_u1(kts) = r6_div(r6_div(r6_mul(r6_mul(r6_mul(0.5f, envi_w(kts)), envi_a(kts)), r6_sub(r6_mul(rho1(((kts) + (1))), u1(((kts) + (1)))), r6_mul(rho1(kts), u1(kts)))), dzi(kts)), rhoi(kts));',
        'jcf(%dULL+4ULL,(float)(r6_mul(r6_mul(0.5f, envi_w(kts)), envi_a(kts))));jcf(%dULL+4ULL,(float)(dzi(kts)));jcf(%dULL+4ULL,(float)(rhoi(kts)));'
        'jcf(%dULL+4ULL,(float)(rho1(kts)));jcf(%dULL+8ULL,(float)(rho1(((kts)+(1)))));' % (FR['pm_cs'], FR['pm_dzi'], FR['pm_rhoi'], FR['pm_rho'], FR['pm_rho']))
    aft('sub_u1(k) = r6_div(r6_div(r6_mul(r6_mul(r6_mul(r6_mul(0.5f, r6_add(envi_w(k), envi_w(((k) - (1))))), 0.5f), r6_add(envi_a(k), envi_a(((k) - (1))))), r6_sub(r6_mul(rho1(((k) + (1))), u1(((k) + (1)))), r6_mul(rho1(k), u1(k)))), dzi(k)), rhoi(k));',
        'jcf(%dULL+4ULL*k,(float)(r6_mul(r6_mul(r6_mul(0.5f, r6_add(envi_w(k), envi_w(((k) - (1))))), 0.5f), r6_add(envi_a(k), envi_a(((k) - (1)))))));jcf(%dULL+4ULL*k,(float)(dzi(k)));'
        'jcf(%dULL+4ULL*k,(float)(rhoi(k)));jcf(%dULL+4ULL*k,(float)(rho1(k)));jcf(%dULL+4ULL*(k+1),(float)(rho1(((k)+(1)))));' % (FR['pm_cs'], FR['pm_dzi'], FR['pm_rhoi'], FR['pm_rho'], FR['pm_rho']))
    aft('det_u1(k) = r6_mul(r6_mul(r6_mul(cdet, r6_sub(envm_u(k), u1(k))), envi_a(k)), psig_w);',
        'jcf(%dULL+4ULL*k,(float)(envi_a(k)));jcf(%dULL,(float)(psig_w));' % (FR['pm_ea'], FR['pm_psig']))
# a layout built with -DJOIN_PHEAT has the plume heat and moisture block (join.hpp PlumeHeat). The
# probes write the coefficients gen_whole.py --plume-live momheat holds on the plume's heat and moisture lines; started, pass,
# c1, cs, dzi, rhoi, rho, ea, psig, adj are the plume-momentum block's.
PHEAT = 'ph' in FR
if PHEAT:
    assert PMOM, 'JOIN_PHEAT needs JOIN_PMOM'
    aft('thln = r6_add(r6_mul(upthl(k, ip), r6_sub(one, entexp)), r6_mul(thl1(k), entexp));',
        'jcf(%dULL+4ULL*%s,(float)(r6_sub(one, entexp)));jcf(%dULL+4ULL*%s,(float)(entexp));' % (FR['ph_om'], IPK, FR['ph_en'], IPK))
    aft('envm_thl(k) = r6_add(envm_thl(k), r6_mul(r6_mul(r6_mul(r6_sub(r6_mul(p5, r6_add(thl_ent, upthl(k, ip))), thl1(k)), detrate), aratio), mynn_min((float)(dzp), (float)(dzpmax))));',
        'jcd(%dULL+8ULL*%s,((double)(detrate)*(double)(aratio))*(double)(mynn_min((float)(dzp), (float)(dzpmax))));' % (FR['ph_dc'], IPK))
    # mynn_max(a, b) is a > b ? a : b, so the derivative follows the second argument unless zero > it
    aft('qv_ent = r6_mul(p5, r6_add(mynn_max((float)(zero), (float)(r6_sub(qt_ent, qc_ent))), mynn_max((float)(zero), (float)(r6_sub(upqt(k, ip), upqc(k, ip))))));',
        'if(!((float)(zero)>(float)(r6_sub(qt_ent, qc_ent))))jcor(%dULL+4ULL*k,1<<(ip-1));if(!((float)(zero)>(float)(r6_sub(upqt(k, ip), upqc(k, ip)))))jcor(%dULL+4ULL*k,1<<(ip-1));' % (FR['ph_g1'], FR['ph_g2']))
for pos, text in sorted(edits, key=lambda e: -e[0]):
    u = u[:pos] + text + u[pos:]

# frame pointer and writers on the column object (null in every other entry: no write)
anchor = 'int lane_sub = 0; unsigned lane_mask = 0u;'
assert u.count(anchor) == 1
u = u.replace(anchor, anchor + '\nunsigned char* jcap = nullptr; int jcap_n = 0;'
              '\n__device__ __forceinline__ void jcf(unsigned long long o, float v) const { if (jcap && lane_sub == 0) *reinterpret_cast<float*>(jcap + o) = v; }'
              '\n__device__ __forceinline__ void jcd(unsigned long long o, double v) const { if (jcap && lane_sub == 0) *reinterpret_cast<double*>(jcap + o) = v; }'
              '\n__device__ __forceinline__ void jci(unsigned long long o, int v) const { if (jcap && lane_sub == 0) *reinterpret_cast<int*>(jcap + o) = v; }'
              + ('\n__device__ __forceinline__ void jcor(unsigned long long o, int v) const { if (jcap && lane_sub == 0) *reinterpret_cast<int*>(jcap + o) |= v; }' if (PMOM or MIX) else ''))
# the capture entry: rrfs_mynnedmf_column with the frame arguments appended; the three plain entries are dropped
head = 'template<int MYNN_LANES> __device__ __forceinline__ void rrfs_mynnedmf_column('
s = u.index(head); e = u.index(') {', s); depth = 1; b = e + 3
while depth: depth += (u[b] == '{') - (u[b] == '}'); b += 1
fn = u[s:b]
params = fn[len(head):fn.index(') {')]
cap = fn.replace('rrfs_mynnedmf_column(', 'rrfs_mynnedmf_capture(', 1).replace(params + ') {', params + ',unsigned char* jcap_frames,unsigned long long jcap_bytes) {', 1)
mark = 'MynnColumn<MYNN_LANES> column{arena};'
assert cap.count(mark) == 1
hdr = ('column.jcap=jcap_frames+(unsigned long long)column_index*jcap_bytes;column.jcap_n=kte;'
       'column.jci(%dULL,kte);column.jcf(%dULL,delt);column.jci(%dULL,1);column.jci(%dULL,-1);column.jcf(%dULL,2.5999999046325684f);'
       % (FR['n'], FR['dt'], FR['dheat'], FR['ktop'], FR['ccfg_closure']))
if PMOM: hdr += 'column.jci(%dULL,1);' % FR['pm_on']
if PHEAT: hdr += 'column.jci(%dULL,1);' % FR['ph_on']
# CAP_PLUME_LIVE=1 marks every frame plume_live, for a form whose updraft plume keeps its derivative (generator
# freeze group 8 off): the joined backward then refuses a column whose plume is active (Status::Plume, the taped route) and
# reverses the others, where the plume returns zeros and has no derivative.
import os as _os
if _os.environ.get('CAP_PLUME_LIVE') == '1': hdr += 'column.jci(%dULL,1);' % FR['plume_live']
if MIX: hdr += 'column.jci(%dULL,1);column.jcd(%dULL,%s);' % (FR['mx_on'], FR['mx_t'], WST)
cap = cap.replace(mark, mark + hdr)
names = [q.replace('*', ' ').split()[-1] for q in params.split(',')]
entry = ('extern "C" __global__ void __launch_bounds__(32) mynn_capture_l1(%s,unsigned char* jcap_frames,unsigned long long jcap_bytes) '
         '{ rrfs_mynnedmf_capture<1>(%s,jcap_frames,jcap_bytes); }\n'
         'extern "C" __device__ unsigned long long mynn_capture_info[4]={1,%d,%d,32};\n' % (params, ','.join(names), FR['sizeof'], len(names)))
u = u[:b] + '\n' + cap + '\n' + u[b:]
u, ndrop = re.subn(r'^extern "C" __global__[^\n]*void rrfs_mynnedmf_run_l(1|4|16)\([^\n]*\n', '', u, flags=re.M)
assert ndrop == 3, ndrop
u = u + '\n// bl-wire capture entry (capture_gen.py)\n' + entry
if MIX:   # the mirror, inlined after the forward's value functions, before the column object (its members hold the constants)
    mj = (Path(__file__).resolve().parents[1] / 'blocks' / 'mixjac.hpp').read_text().replace('#pragma once\n', '')
    mdef = ('#define MJ_HD __device__ __forceinline__\n#define MJ_FADD(a,b) r6_add((float)(a),(float)(b))\n#define MJ_FSUB(a,b) r6_sub((float)(a),(float)(b))\n'
            '#define MJ_FMUL(a,b) r6_mul((float)(a),(float)(b))\n#define MJ_FDIV(a,b) r6_div((float)(a),(float)(b))\n'
            '#define MJ_DADD(a,b) mynn_dadd((double)(a),(double)(b))\n#define MJ_DSUB(a,b) mynn_dsub((double)(a),(double)(b))\n'
            '#define MJ_DMUL(a,b) mynn_dmul((double)(a),(double)(b))\n#define MJ_DDIV(a,b) mynn_ddiv((double)(a),(double)(b))\n#define MJ_DSQRT(a) mynn_sqrt((double)(a))\n'
            '#define MJ_FSQRT(a) mynn_sqrt((float)(a))\n')
    k = u.index('template<int MYNN_LANES> struct MynnColumn {')
    u = u[:k] + '// the stability block\'s forward-mode mirror (blocks/mixjac.hpp)\n' + mdef + mj + '\n' + u[k:]
out.write_text(u)
print(json.dumps({'source_sha256': hashlib.sha256(src.read_bytes()).hexdigest(), 'out_sha256': hashlib.sha256(u.encode()).hexdigest(),
                  'plume_momentum': PMOM, 'plume_heat': PHEAT, 'plume_live': _os.environ.get('CAP_PLUME_LIVE') == '1', 'mixing': MIX, 'closure': bool(MIX and CLOS), 'points': sorted(M) + ['T_prod (loop)'], 'oracle_only_points': unused, 'frame_bytes': FR['sizeof'], 'entry_args': len(names) + 2}))
