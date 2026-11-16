#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# x20 mynn-M9 (2026-11-09): build-time generator for the fused reverse of MYNN-EDMF stage M9
# (f_mynn_tendencies with f_tridiag2 and f_moisture_check inlined), read from the generated forward-carrier
# source mynn_whole.gen.cu (gen_whole.py output). Emits register/arena-resident __device__ code, no node tape:
#   m9_fwd  : the stage forward on plain float (the forward's own r6_*/mynn_* routines, so bitwise the forward);
#   m9_rev  : block-recompute reverse. The stage is cut at solve boundaries; at each cut the forward keeps only
#             the arrays that are live across the cut AND rewritten later (snapshots); each block is reversed by
#             re-running its own forward from that state (in-block value stack only for in-block overwrites and
#             branch bits), then sweeping its statements backwards. Partials are the tape rules' (mw_ad_rules.inc):
#             float primal, double derivative arithmetic from (double) primal values, executed-branch min/max.
#   m9_ext_record / m9_ext_reverse : drop-in for the taped P3 kernel (one external node replaces the stage's nodes).
# usage: gen_m9.py IN.gen.cu OUT.cuh
import re, sys

SRC = open(sys.argv[1]).read()
OUT = sys.argv[2]

def func_body(name):
    m = re.search(r'^__device__ __\w+ void %s\((.*)\) \{$' % name, SRC, re.M)
    assert m, name
    s = m.end(); d = 1; e = s
    while d:
        c = SRC[e]; d += (c == '{') - (c == '}'); e += 1
    return m.group(1), SRC[s:e - 1]

def parse_params(ps):
    out = []
    for p in [x.strip() for x in re.split(r',\s*(?![^<]*>)', ps)]:
        m = re.match(r'(MynnArray<MR, 1>|MR&|MR|int&|int|bool) (\w+)$', p)
        assert m, p
        out.append((m.group(1), m.group(2)))
    return out

# ---------------------------------------------------------------- expressions
TOK = re.compile(r'\s*(?:(\d+\.\d*(?:[eE][-+]?\d+)?f?|\d*\.\d+(?:[eE][-+]?\d+)?f?|\d+[eE][-+]?\d+f?|\d+\.?f?)|(\w+)|(<=|>=|==|!=|&&|\|\||[-+*/(),<>!{}?:.]))')
def tokenize(s):
    out = []; i = 0
    while i < len(s):
        m = TOK.match(s, i)
        if not m or m.end() == i:
            if s[i:].strip() == '': break
            raise SyntaxError(s[i:i + 40])
        out.append(m.group(1) or m.group(2) or m.group(3)); i = m.end()
    return out

class P:
    def __init__(s, toks): s.t = toks; s.i = 0
    def peek(s, k=0): return s.t[s.i + k] if s.i + k < len(s.t) else None
    def eat(s, x=None):
        v = s.t[s.i]
        if x is not None and v != x: raise SyntaxError('want %s got %s at %s' % (x, v, ' '.join(s.t[max(0, s.i - 5):s.i + 5])))
        s.i += 1; return v
    def expr(s): return s.orr()
    def orr(s):
        a = s.andd()
        while s.peek() == '||': s.eat(); a = ('||', a, s.andd())
        return a
    def andd(s):
        a = s.cmp()
        while s.peek() == '&&': s.eat(); a = ('&&', a, s.cmp())
        return a
    def cmp(s):
        a = s.add()
        while s.peek() in ('<', '>', '<=', '>=', '==', '!='):
            o = s.eat(); a = (o, a, s.add())
        return a
    def add(s):
        a = s.mul()
        while s.peek() in ('+', '-'): o = s.eat(); a = (o, a, s.mul())
        return a
    def mul(s):
        a = s.un()
        while s.peek() in ('*', '/', '%'): o = s.eat(); a = (o, a, s.un())
        return a
    def un(s):
        if s.peek() == '-': s.eat(); return ('neg', s.un())
        if s.peek() == '!': s.eat(); return ('!', s.un())
        return s.prim()
    def prim(s):
        t = s.peek()
        if t == '(':
            s.eat()
            if s.peek() == 'MR' and s.peek(1) == ')':
                s.eat(); s.eat(); s.eat('('); e = s.expr(); s.eat(')'); return e   # (MR)(e)
            e = s.expr(); s.eat(')'); return e
        if re.match(r'[\d.]', t): s.eat(); return ('num', t)
        name = s.eat()
        if name in ('true', 'false'): return ('num', name)
        if name == 'mynn_ipow':
            s.eat('<'); n = s.eat(); s.eat('>'); s.eat('('); a = s.expr(); s.eat(')'); return ('call', 'ipow' + n, [a])
        if s.peek() == '(':
            s.eat(); args = []
            if s.peek() != ')':
                args.append(s.expr())
                while s.peek() == ',': s.eat(); args.append(s.expr())
            s.eat(')')
            return ('ref', name, args)   # function call or array element, resolved later
        if s.peek() == '.':
            s.eat(); f = s.eat(); s.eat('('); args = []
            if s.peek() != ')': args.append(s.expr())
            s.eat(')'); return ('meth', name, f, args)
        return ('var', name)

def rawx(s):
    m = re.match(r"^(\w+)\.(lo|hi)\[0\]$", s)
    return ("raw", s, m.group(1)) if m else pexpr(s)

def pexpr(s):
    p = P(tokenize(s)); e = p.expr(); assert p.i == len(p.t), s; return e

# ---------------------------------------------------------------- statements
LOOPH = re.compile(r'^\{ const int mynn_lo(\d+) = \((.*)\), mynn_hi\1 = \((.*)\), mynn_st\1 = \((.*)\); const int mynn_n\1 = .*?for \(int mynn_it\1 = lane_sub; .*?\) \{ (\w+) = mynn_lo\1 \+ mynn_it\1\*mynn_st\1;$')
LOOPE = re.compile(r'^\} mynn_lane_sync\(\); (\w+) = mynn_lo(\d+) \+ mynn_n\2\*mynn_st\2;.*\}$')
FORH = re.compile(r'^for \((\w+)=(.*); \(+(.*?)\)+ > 0 \? \1 <= \((.*)\) : \1 >= \(.*\); \1 \+= \(+(.*?)\)+\) \{$')
FILL = re.compile(r'^mynn_lane_sync\(\); for \(int mynn_element=lane_sub; mynn_element < \((.*)\)\.size\(\); mynn_element \+= lane_width\) \{ (.*)\.linear\(mynn_element\) = (.*); \} mynn_lane_sync\(\);$')
SLICE = re.compile(r'^(\w+)\.slice\(MynnRange\{(.*), (.*)\}\)$')

def clean(body):
    lines = []; skip = 0
    for l in body.split('\n'):
        l = l.strip()
        if not l or l.startswith('//'): continue
        if l.startswith('#ifdef'): skip += 1; continue
        if l.startswith('#endif'): skip -= 1; continue
        if skip: continue
        if l.startswith('MynnFrame') or re.match(r'^\w+\.rebase\(', l) or l.startswith('; /*'): continue
        lines.append(l)
    return lines

def parse_block(lines, i, decl):
    out = []
    while i < len(lines):
        l = lines[i]
        if l == '}' or l == '} else {' or LOOPE.match(l): return out, i
        m = re.match(r'^(MR|int|bool) (\w+);$', l)
        if m: decl['scal'][m.group(2)] = m.group(1); i += 1; continue
        m = re.match(r'^MynnArray<MR, 1> (\w+) = mynn_array_any<MR, 1, mynn_tile>\(arena, \{(.*)\}, \{(.*)\}\);$', l)
        if m: decl['arr'][m.group(1)] = (m.group(2), m.group(3)); i += 1; continue
        m = re.match(r'^constexpr (float|int) (\w+) = (.*);$', l)
        if m: decl['const'][m.group(2)] = (m.group(1), m.group(3)); i += 1; continue
        m = LOOPH.match(l)
        if m:
            body, i = parse_block(lines, i + 1, decl)
            me = LOOPE.match(lines[i]); assert me and me.group(2) == m.group(1), lines[i]
            out.append(('loop', m.group(5), pexpr(m.group(2)), pexpr(m.group(3)), int(m.group(4)), body)); i += 1; continue
        m = FORH.match(l)
        if m:
            body, i = parse_block(lines, i + 1, decl); assert lines[i] == '}', lines[i]
            st = int(m.group(5).replace('-(1)', '-1').replace('(', '').replace(')', ''))
            out.append(('loop', m.group(1), pexpr(m.group(2)), pexpr(m.group(4)), st, body)); i += 1; continue
        m = FILL.match(l)
        if m:
            dst = m.group(2); src = m.group(3); ms = SLICE.match(dst)
            if ms: name, lo, hi = ms.group(1), rawx(ms.group(2)), rawx(ms.group(3))
            else: name, lo, hi = dst, None, None
            ms = re.match(r'^(\w+)\.linear\(mynn_element\)$', src)
            out.append(('fill', name, lo, hi, ('arr', ms.group(1)) if ms else pexpr(src))); i += 1; continue
        m = re.match(r'^if \((.*)\) \{ (.*;) \}$', l)
        if m:
            sub, _ = parse_block([m.group(2), '}'], 0, decl)
            out.append(('if', pexpr(m.group(1)), sub, [])); i += 1; continue
        m = re.match(r'^if \((.*)\) \{$', l)
        if m:
            th, i = parse_block(lines, i + 1, decl); el = []
            if lines[i] == '} else {':
                el, i = parse_block(lines, i + 1, decl)
            assert lines[i] == '}', lines[i]
            out.append(('if', pexpr(m.group(1)), th, el)); i += 1; continue
        if l == 'return;': i += 1; continue
        m = re.match(r'^(f_\w+)\((.*)\);$', l)
        if m:
            out.append(('call', m.group(1), [a.strip() for a in m.group(2).split(',')])); i += 1; continue
        m = re.match(r'^(\w+(?:\(.*?\))?) = (.*);$', l)
        if m:
            lhs = pexpr(m.group(1)); out.append(('set', lhs, pexpr(m.group(2)))); i += 1; continue
        raise SyntaxError(l)
    return out, i

def parse_func(name):
    ps, body = func_body(name)
    decl = {'scal': {}, 'arr': {}, 'const': {}}
    st, i = parse_block(clean(body), 0, decl)
    return parse_params(ps), st, decl

# ---------------------------------------------------------------- inlining with renaming
def rename_expr(e, ren):
    if e is None: return None
    t = e[0]
    if t == 'var': return ('var', ren.get(e[1], e[1])) if not isinstance(ren.get(e[1]), tuple) else ren[e[1]]
    if t == 'num': return e
    if t == 'ref': return ('ref', ren.get(e[1], e[1]), [rename_expr(a, ren) for a in e[2]])
    if t == 'call': return ('call', e[1], [rename_expr(a, ren) for a in e[2]])
    if t == 'meth': return ('meth', ren.get(e[1], e[1]), e[2], [rename_expr(a, ren) for a in e[3]])
    if t in ('neg', '!'): return (t, rename_expr(e[1], ren))
    if t == 'arr': return ('arr', ren.get(e[1], e[1]))
    if t == 'raw':
        n = ren.get(e[2], e[2]); return ('raw', n + e[1][len(e[2]):], n)
    return (t, rename_expr(e[1], ren), rename_expr(e[2], ren))

def rename_stmts(sts, ren):
    out = []
    for s in sts:
        k = s[0]
        if k == 'set': out.append(('set', rename_expr(s[1], ren), rename_expr(s[2], ren)))
        elif k == 'if': out.append(('if', rename_expr(s[1], ren), rename_stmts(s[2], ren), rename_stmts(s[3], ren)))
        elif k == 'loop': out.append(('loop', ren.get(s[1], s[1]), rename_expr(s[2], ren), rename_expr(s[3], ren), s[4], rename_stmts(s[5], ren)))
        elif k == 'fill': out.append(('fill', ren.get(s[1], s[1]), rename_expr(s[2], ren), rename_expr(s[3], ren), rename_expr(s[4], ren)))
        elif k == 'call': out.append(('call', s[1], [ren.get(a, a) for a in s[2]]))
        else: raise ValueError(k)
    return out

top_params, top, top_decl = parse_func('f_mynn_tendencies')
callees = {}
for cn, pre in (('f_tridiag2', 'tri_'), ('f_moisture_check', 'mc_')):
    callees[cn] = (parse_func(cn), pre)

LOCAL_ARR = dict(top_decl['arr']); LOCAL_SCAL = dict(top_decl['scal']); LOCAL_CONST = dict(top_decl['const'])
def inline(sts):
    out = []
    for s in sts:
        if s[0] == 'call':
            (cps, cst, cdecl), pre = callees[s[1]]
            ren = {}
            for (ty, pn), a in zip(cps, s[2]): ren[pn] = a
            for n, (lo, hi) in cdecl['arr'].items(): ren[n] = pre + n; LOCAL_ARR[pre + n] = (lo, hi)
            for n, ty in cdecl['scal'].items(): ren[n] = pre + n; LOCAL_SCAL[pre + n] = ty
            for n, v in cdecl['const'].items(): ren[n] = pre + n; LOCAL_CONST[pre + n] = v
            out += inline(rename_stmts(cst, ren))
        elif s[0] == 'if':
            out.append(('if', s[1], inline(s[2]), inline(s[3])))
        elif s[0] == 'loop':
            out.append(s[:5] + (inline(s[5]),))
        else: out.append(s)
    return out
SPEC = {'bl_mynn_mixqt': '0'}   # specialisation: the f_qsat_blend path is not generated; the caller keeps the tape there
def cfold(e):
    if e is None: return None
    t = e[0]
    if t == 'var' and e[1] in SPEC: return ('num', SPEC[e[1]])
    if t in ('num', 'var', 'arr', 'meth', 'raw'): return e
    if t == 'ref': return ('ref', e[1], [cfold(a) for a in e[2]])
    if t == 'call': return ('call', e[1], [cfold(a) for a in e[2]])
    if t in ('neg', '!'): return (t, cfold(e[1]))
    a, b = cfold(e[1]), cfold(e[2])
    if a[0] == 'num' and b[0] == 'num' and t in ('<', '>', '<=', '>=', '==', '!='):
        r = eval('%s %s %s' % (a[1].rstrip('f'), t, b[1].rstrip('f'))); return ('num', 'true' if r else 'false')
    if t == '&&' and ('num', 'false') in (a, b): return ('num', 'false')
    if t == '||' and ('num', 'true') in (a, b): return ('num', 'true')
    return (t, a, b)
def fold(sts):
    out = []
    for s in sts:
        if s[0] == 'if':
            c = cfold(s[1])
            if c == ('num', 'false'): out += fold(s[3]); continue
            if c == ('num', 'true'): out += fold(s[2]); continue
            out.append(('if', c, fold(s[2]), fold(s[3])))
        elif s[0] == 'loop': out.append(s[:5] + (fold(s[5]),))
        else: out.append(s)
    return out
top = fold(top)
# the debug block (debug_code is a static constexpr false in the forward) carries no field arithmetic
top = [s for s in top if not (s[0] == 'if' and s[1] == ('var', 'debug_code'))]
top = inline(top)

PARR = [n for t, n in top_params if t == 'MynnArray<MR, 1>']
PSCAL = [n for t, n in top_params if t == 'MR']
PINT = [(t, n) for t, n in top_params if t in ('int', 'bool')]
assert not [n for t, n in top_params if t in ('MR&', 'int&')]
FARR = set(PARR) | set(LOCAL_ARR)
FSCAL = set(PSCAL) | {n for n, t in LOCAL_SCAL.items() if t == 'MR'}
ISCAL = {n for n, t in LOCAL_SCAL.items() if t != 'MR'} | {n for t, n in PINT}
FUNCS = {'r6_add', 'r6_sub', 'r6_mul', 'r6_div', 'r6_neg', 'mynn_max', 'mynn_min', 'mynn_exp', 'f_qsat_blend'}
# frozen under --freeze closure (gen-info.json): the tape gives them no derivative, so the stage keeps no adjoint
FROZEN = set()
if '--frozen-closure' in sys.argv:
    FROZEN = {'dfm', 'dfh', 'dfq', 'pblh'} | {n for n in PARR if re.match(r'(s_aw|sd_aw|sub_|det_)', n)}
# lane GROUND (2026-11-10): as gen_m9_rounded.py. Under gen_whole.py --freeze closure-plume the plume arrays carry a
# derivative, so the stage keeps their adjoint; --frozen-closure on such a source would drop it in the reverse only
# (a wrong adjoint, a failed dot) and is refused.
SRC_FREEZE = (re.search(r'\(gen_whole\.py --freeze ([^)\s]+)\)', SRC) or [None, 'none'])[1]
FRZ_NAME = 'closure'
if '--frozen-closure-plume' in sys.argv:
    if '--frozen-closure' in sys.argv: raise SystemExit('--frozen-closure and --frozen-closure-plume are two modes: give one')
    FROZEN = {'dfm', 'dfh', 'dfq', 'pblh'}; FRZ_NAME = 'closure-plume'
if '--frozen-closure' in sys.argv and SRC_FREEZE == 'closure-plume':
    raise SystemExit('source is --freeze closure-plume (live plume): use --frozen-closure-plume, not --frozen-closure')
if '--frozen-closure-plume' in sys.argv and SRC_FREEZE == 'none':
    raise SystemExit('source is --freeze none: dfm dfh dfq pblh are live there, --frozen-closure-plume would drop their adjoint')

def is_float_expr(e):
    t = e[0]
    if t == 'num': return 'f' in e[1] or ('.' in e[1] and e[1] not in ('true', 'false')) or 'e' in e[1]
    if t == 'var': return e[1] in FSCAL or (e[1] not in ISCAL and e[1] not in ('kts', 'kte', 'k'))
    if t == 'ref': return e[1] in FARR or e[1] in FUNCS
    if t == 'call': return True
    if t == 'neg': return is_float_expr(e[1])
    return False

# ---------------------------------------------------------------- read/write sets (textual, flattened)
def reads(e, acc):
    if e is None: return
    t = e[0]
    if t == 'var': acc.append(e[1])
    elif t == 'ref':
        if e[1] in FARR: acc.append(e[1])
        for a in e[2]: reads(a, acc)
    elif t == 'call':
        for a in e[2]: reads(a, acc)
    elif t == 'arr': acc.append(e[1])
    elif t == 'meth': acc.append(e[1])
    elif t == 'raw': pass   # array bounds only, no value
    elif t in ('neg', '!'): reads(e[1], acc)
    elif t != 'num': reads(e[1], acc); reads(e[2], acc)

def target(lhs):
    return lhs[1]

def accesses(sts, acc):
    """textual (name, 'r'|'w') sequence, branches flattened in order"""
    for s in sts:
        k = s[0]
        if k == 'set':
            r = []; reads(s[2], r)
            if s[1][0] == 'ref':
                for a in s[1][2]: reads(a, r)
            acc += [(n, 'r') for n in r]; acc.append((target(s[1]), 'w'))
        elif k == 'if':
            r = []; reads(s[1], r); acc += [(n, 'r') for n in r]
            accesses(s[2], acc); accesses(s[3], acc)
        elif k == 'loop':
            r = []; reads(s[2], r); reads(s[3], r); acc += [(n, 'r') for n in r]
            acc.append((s[1], 'w')); accesses(s[5], acc)
        elif k == 'fill':
            r = []; reads(s[2], r); reads(s[3], r); reads(s[4], r); acc += [(n, 'r') for n in r]
            acc.append((s[1], 'w'))
    return acc

CONSTS = set(LOCAL_CONST)
def is_var(n): return n in FARR or n in FSCAL or n in ISCAL

# ---------------------------------------------------------------- blocks: cut where no solve temporary is live
TEMPS = {'a', 'b', 'c', 'd', 'x', 'upcont', 'dncont', 'k'} | {n for n in LOCAL_ARR if n.startswith(('tri_', 'mc_'))} | \
        {n for n in LOCAL_SCAL if n.startswith(('tri_', 'mc_'))} | \
        {n for n, t in LOCAL_SCAL.items() if t == 'MR' and n not in ('onoff', 'ustovwsp', 'rhosfc', 'dztop')}
# moisture_check keeps dqv2 (the last level's value) for its column mass fix after the loop: one block
ACC = [accesses([s], []) for s in top]

# ---- element coverage: an array counts as rewritten (killed) only once writes cover its whole range kts..kte
# (endpoints symbolic: (0, n) = level n with kts == 1, (1, n) = kte + n). Reads before the kill are upward exposed.
def endpoint(e):
    t = e[0]
    if t == 'num' and re.match(r'^\d+$', e[1]): return (0, int(e[1]))
    if t == 'var' and e[1] == 'kts': return (0, 1)
    if t == 'var' and e[1] == 'kte': return (1, 0)
    if t == 'raw': return (0, 1) if e[1].endswith('.lo[0]') else (1, 0)
    if t in ('+', '-'):
        a = endpoint(e[1]); b = e[2]
        if a and b[0] == 'num' and re.match(r'^\d+$', b[1]): return (a[0], a[1] + (int(b[1]) if t == '+' else -int(b[1])))
    return None
def interval(e, env):
    p = endpoint(e)
    if p: return (p, p)
    if e[0] == 'var' and e[1] in env: return env[e[1]]
    if e[0] in ('+', '-') and e[1][0] == 'var' and e[1][1] in env and e[2][0] == 'num' and re.match(r'^\d+$', e[2][1]):
        lo, hi = env[e[1][1]]; c = int(e[2][1]) * (1 if e[0] == '+' else -1)
        return ((lo[0], lo[1] + c), (hi[0], hi[1] + c))
    return None
def full_of(n):
    if n in LOCAL_ARR:
        lo, hi = LOCAL_ARR[n]; a = endpoint(pexpr(lo)); b = endpoint(pexpr(hi))
        if a and b: return (a, b)
    return ((0, 1), (1, 0))
def covers(ivs, full):
    ivs = sorted(ivs); cur = None
    for lo, hi in ivs:
        if cur is None:
            if lo > full[0]: return False
            cur = hi
        else:
            if lo > (cur[0], cur[1] + 1): break
            cur = max(cur, hi)
        if cur >= full[1]: return True
    return cur is not None and cur >= full[1]
def refs_of(e, acc):
    if isinstance(e, tuple):
        if e and e[0] == 'ref' and e[1] in FARR: acc.append((e[1], e[2][0]))
        elif e and e[0] == 'var': acc.append((e[1], None))
        elif e and e[0] == 'arr': acc.append((e[1], 'ALL'))
        for y in e[1:]:
            if isinstance(y, (tuple, list)): refs_of(y, acc)
    elif isinstance(e, list):
        for y in e: refs_of(y, acc)
def offset_of(ix, var):
    if ix[0] == 'var' and ix[1] == var: return 0
    if ix[0] in ('+', '-') and ix[1] == ('var', var) and ix[2][0] == 'num' and re.match(r'^\d+$', ix[2][1]):
        return int(ix[2][1]) * (1 if ix[0] == '+' else -1)
    return None
def fa_cov(sts, names, env=None, st=None, loop=None):
    """upward-exposed reads per element: name -> 'r' when some element is read before the suffix writes it.
    loop = (var, start, end, step, names written as v(var) unconditionally in the body, names written so far)"""
    if env is None: env = {}
    if st is None: st = {'dec': {}, 'cov': {}}
    dec, cv = st['dec'], st['cov']
    def covered(n, iv):
        return iv is not None and covers(cv.get(n, []), iv)
    def rd(e):
        acc = []; refs_of(e, acc)
        for n, ix in acc:
            if n not in names or n in dec: continue
            if n not in FARR:
                if n not in cv: dec[n] = 'r'
                continue
            if ix == 'ALL': iv = full_of(n)
            else: iv = interval(ix, env)
            ok = covered(n, iv)
            if not ok and loop and ix not in (None, 'ALL'):
                var, s0, e0, stp, bw, done = loop
                o = offset_of(ix, var)
                if o is not None and n in bw:
                    if o == 0 and n in done: ok = True
                    elif stp > 0 and o < 0: ok = covered(n, ((s0[0], s0[1] + o), (s0[0], s0[1] - 1)))
                    elif stp < 0 and o > 0: ok = covered(n, ((s0[0], s0[1] + 1), (s0[0], s0[1] + o)))
            if not ok: dec[n] = 'r'
    def wr(n, iv):
        if n in FARR:
            if iv is not None: cv.setdefault(n, []).append(iv)
        else: cv[n] = [((0, 0), (0, 0))]
    for s in sts:
        k = s[0]
        if k == 'set':
            rd(s[2])
            if s[1][0] == 'ref':
                rd(s[1][2])
                if loop and offset_of(s[1][2][0], loop[0]) == 0: loop[5].add(s[1][1])
                wr(s[1][1], interval(s[1][2][0], env))
            else:
                wr(s[1][1], None)
                if s[1][1] in ISCAL:
                    pe = endpoint(s[2])
                    if pe: env[s[1][1]] = (pe, pe)
                    else: env.pop(s[1][1], None)
        elif k == 'if':
            rd(s[1])
            s1 = {'dec': dict(dec), 'cov': {n: list(v) for n, v in cv.items()}}
            s2 = {'dec': dict(dec), 'cov': {n: list(v) for n, v in cv.items()}}
            l1 = loop and loop[:5] + (set(loop[5]),); l2 = loop and loop[:5] + (set(loop[5]),)
            e1, e2 = dict(env), dict(env)
            fa_cov(s[2], names, e1, s1, l1); fa_cov(s[3], names, e2, s2, l2)
            for n in list(env):
                if e1.get(n) != env[n] or e2.get(n) != env[n]: env.pop(n)
            for n in set(e1) & set(e2):
                if e1[n] == e2[n]: env[n] = e1[n]
            for n in set(s1['dec']) | set(s2['dec']): dec.setdefault(n, 'r')
            for n in set(s1['cov']) & set(s2['cov']):
                if s1['cov'][n] == s2['cov'][n]: cv[n] = s1['cov'][n]
            if loop: loop[5].update(l1[5] & l2[5])
        elif k == 'loop':
            rd([s[2], s[3]]); wr(s[1], None)
            env.pop(s[1], None)
            a, b = endpoint(s[2]), endpoint(s[3])
            env2 = dict(env)
            if a and b:
                env2[s[1]] = (min(a, b), max(a, b))
                bw = {x[1][1] for x in s[5] if x[0] == 'set' and x[1][0] == 'ref' and offset_of(x[1][2][0], s[1]) == 0}
                fa_cov(s[5], names, env2, st, (s[1], a, b, s[4], bw, set()))
            else:
                env2.pop(s[1], None); fa_cov(s[5], names, env2, st, None)
        elif k == 'fill':
            rd(s[4])
            if s[2] is None: wr(s[1], full_of(s[1]))
            else:
                a, b = endpoint(s[2]), endpoint(s[3]); wr(s[1], (a, b) if a and b else None)
    return dec
def first_access(names, frm):
    res = {}
    for j in range(frm, len(top)):
        for n, rw in ACC[j]:
            if n in names and n not in res: res[n] = rw
    return res
cuts = [0]
for i in range(1, len(top)):
    fa = fa_cov(top[i:], TEMPS)
    if all(rw == 'w' for rw in fa.values()): cuts.append(i)
blocks = [(cuts[j], cuts[j + 1] if j + 1 < len(cuts) else len(top)) for j in range(len(cuts))]
written_from = []
for b0, b1 in blocks:
    w = set()
    for j in range(b0, len(top)):
        w |= {n for n, rw in ACC[j] if rw == 'w'}
    written_from.append(w)
# restore set of block b: the variables block b reads before writing them (its upward-exposed reads) that some
# block >= b rewrites. Storage: one copy per value version, taken at the first cut that needs that version.
snaps = []; ver_at = []; versions = []
blk_w = []
for b0, b1 in blocks:
    ww = set()
    for j in range(b0, b1): ww |= {n for n, rw in ACC[j] if rw == 'w'}
    blk_w.append(ww)
last = {}
for bi, ((b0, b1), w) in enumerate(zip(blocks, written_from)):
    fa = fa_cov(top[b0:b1], {n for n in FARR | FSCAL | ISCAL})
    need = sorted(n for n, rw in fa.items() if rw == 'r' and n in w and n not in TEMPS and is_var(n))
    snaps.append(need)
    take = []
    for n in need:
        if n not in last or any(n in blk_w[j] for j in range(last[n], bi)):
            versions.append((n, bi)); last[n] = bi; take.append(n)
    ver_at.append(take)
src_ver = []   # (block, var) -> version block
for bi, need in enumerate(snaps):
    m = {}
    for n in need:
        m[n] = max(vb for vn, vb in versions if vn == n and vb <= bi)
    src_ver.append(m)

# ---------------------------------------------------------------- emission helpers
WRAP = {'r6_add', 'r6_sub', 'r6_mul', 'r6_div', 'r6_neg', 'mynn_exp'}
def cx(e):
    """primal C expression (float/int), same routines as the forward"""
    t = e[0]
    if t == 'num': return e[1]
    if t == 'var': return e[1]
    if t == 'raw': return e[1]
    if t == 'ref':
        if e[1] in FARR: return '%s(%s)' % (e[1], ', '.join(cx(a) for a in e[2]))
        assert e[1] in FUNCS, e[1]
        if e[1] in ('mynn_max', 'mynn_min'): return '%s((float)(%s), (float)(%s))' % (e[1], cx(e[2][0]), cx(e[2][1]))
        return '%s%s(%s)' % ('m9w_' if e[1] in WRAP else '', e[1], ', '.join(cx(a) for a in e[2]))
    if t == 'call': return 'mynn_ipow<%s>(%s)' % (e[1][4:], cx(e[2][0]))
    if t == 'meth':
        assert e[2] == 'size' or e[2] in ('lo', 'hi'), e
        return '%s.%s(%s)' % (e[1], e[2], ', '.join(cx(a) for a in e[3]))
    if t == 'neg': return '(-(%s))' % cx(e[1])
    if t == '!': return '(!(%s))' % cx(e[1])
    return '((%s) %s (%s))' % (cx(e[1]), t, cx(e[2]))

def bar_of(n): return n + '_b'

ACTIVE = set()
def active(n): return n in ACTIVE

class Em:
    def __init__(s): s.lines = []; s.tmp = 0; s.ind = 1
    def w(s, l): s.lines.append('    ' * s.ind + l)
    def t(s, pre='t'): s.tmp += 1; return '%s%d' % (pre, s.tmp)

def V(n):
    return n[3] if n[0] == "sel" or len(n) == 4 else n[2]

def fwd_prim(em, e):
    """emit primal temporaries for the reverse of e; returns (node tree with temp names)"""
    t = e[0]
    if t in ('num', 'var') or (t == 'ref' and e[1] in FARR):
        return ('leaf', e, cx(e))
    if t == 'neg':
        a = fwd_prim(em, e[1]); v = em.t(); em.w('const float %s = m9w_r6_neg(%s);' % (v, V(a))); return ('neg', a, v)
    if t == 'call':
        a = fwd_prim(em, e[2][0]); v = em.t(); em.w('const float %s = mynn_ipow<%s>(%s);' % (v, e[1][4:], V(a))); return (e[1], a, v)
    if t == 'ref':
        f = e[1]
        if f == 'mynn_exp':
            a = fwd_prim(em, e[2][0]); v = em.t(); em.w('const float %s = m9w_mynn_exp(%s);' % (v, V(a))); return ('exp', a, v)
        if f == 'r6_neg':
            a = fwd_prim(em, e[2][0]); v = em.t(); em.w('const float %s = m9w_r6_neg(%s);' % (v, V(a))); return ('neg', a, v)
        assert f in ('r6_add', 'r6_sub', 'r6_mul', 'r6_div', 'mynn_max', 'mynn_min'), f
        a = fwd_prim(em, e[2][0]); b = fwd_prim(em, e[2][1]); v = em.t()
        if f in ('mynn_max', 'mynn_min'):
            sel = em.t('s'); em.w('const bool %s = (float)(%s) %s (float)(%s);' % (sel, V(a), '>' if f == 'mynn_max' else '<', V(b)))
            em.w('const float %s = %s ? (float)(%s) : (float)(%s);' % (v, sel, V(a), V(b))); return ('sel', a, b, v, sel)
        em.w('const float %s = m9w_%s(%s, %s);' % (v, f, V(a), V(b))); return (f, a, b, v)
    raise ValueError(e)

def back(em, node, gb):
    """propagate double adjoint expression gb (a C identifier) into the leaves (tape rules)"""
    k = node[0]
    if k == 'leaf':
        e = node[1]
        if e[0] == 'var' and active(e[1]): em.w('%s += %s;' % (bar_of(e[1]), gb))
        elif e[0] == 'ref' and e[1] in FARR and active(e[1]):
            em.w('%s(%s) += %s;' % (bar_of(e[1]), ', '.join(cx(a) for a in e[2]), gb))
        return
    if not has_active(node): return
    if k == 'neg': g = em.t('g'); em.w('const double %s = -%s;' % (g, gb)); back(em, node[1], g); return
    if k.startswith('ipow'):
        n = int(k[4:]); a = node[1]
        assert n == 2
        g = em.t('g'); em.w('const double %s = %s * (2.0*(double)%s);' % (g, gb, V(a))); back(em, a, g); return
    if k == 'exp':
        g = em.t('g'); em.w('const double %s = %s * (double)%s;' % (g, gb, node[2])); back(em, node[1], g); return
    if k == 'sel':
        a, b, v, sel = node[1:]
        em.w('if (%s) {' % sel); em.ind += 1; back(em, a, gb); em.ind -= 1; em.w('} else {'); em.ind += 1; back(em, b, gb); em.ind -= 1; em.w('}')
        return
    a, b = node[1], node[2]
    if k == 'r6_add': back(em, a, gb); back(em, b, gb); return
    if k == 'r6_sub':
        back(em, a, gb)
        if has_active(b): g = em.t('g'); em.w('const double %s = -%s;' % (g, gb)); back(em, b, g)
        return
    if k == 'r6_mul':
        if has_active(a): g = em.t('g'); em.w('const double %s = %s * (double)%s;' % (g, gb, V(b))); back(em, a, g)
        if has_active(b): g = em.t('g'); em.w('const double %s = %s * (double)%s;' % (g, gb, V(a))); back(em, b, g)
        return
    if k == 'r6_div':
        # mw_quot: tq = (1/den)*bar; a += tq; b += w*(-tq), w = (double)a/(double)b, den = (double)b
        tq = em.t('g'); em.w('const double %s = (1.0 / (double)%s) * %s;' % (tq, V(b), gb))
        back(em, a, tq)
        if has_active(b):
            g = em.t('g'); em.w('const double %s = ((double)%s / (double)%s) * (-%s);' % (g, V(a), V(b), tq)); back(em, b, g)
        return
    raise ValueError(k)

def has_active(node):
    k = node[0]
    if k == 'leaf':
        e = node[1]
        return (e[0] == 'var' and active(e[1])) or (e[0] == 'ref' and e[1] in FARR and active(e[1]))
    if k == 'sel': return has_active(node[1]) or has_active(node[2])
    return any(has_active(c) for c in node[1:] if isinstance(c, tuple))

def expr_active(e):
    r = []; reads(e, r); return any(active(n) for n in r)

def cond_float(e):
    r = []; reads(e, r)
    return any(n in FARR or n in FSCAL for n in r)

# ---------------------------------------------------------------- push analysis inside one block
def push_set(sts):
    """statements ids whose target needs its old value saved: every write except a block-first write of a
    variable whose first access in the block is that write, outside any loop for scalars"""
    first = {}
    acc = accesses(sts, [])
    for n, rw in acc:
        if n not in first: first[n] = rw
    return first

def gen_block(em, sts, mode, first, depth=0, seen=None):
    """mode 'f' plain forward, 'p' forward with pushes, 'r' reverse"""
    if seen is None: seen = set()
    seq = sts if mode != 'r' else list(reversed(sts))
    for s in seq:
        k = s[0]
        if k == 'set':
            tn = target(s[1]); lhs = cx(s[1])
            isint = tn in ISCAL
            if mode == 'f':
                em.w('%s = %s;' % (lhs, cx(s[2])))
            elif mode == 'p':
                if s_needs_push(s, first, depth):
                    em.w('M9_PUSH(%s);' % (('__int_as_float(%s)' % lhs) if isint else lhs))
                em.w('%s = %s;' % (lhs, cx(s[2])))
            else:
                em.w('{')
                em.ind += 1
                if s_needs_push(s, first, depth):
                    em.w(('%s = __float_as_int(M9_POP());' if isint else '%s = M9_POP();') % lhs)
                if not isint and active(tn):
                    gb = em.t('eb')
                    bl = bar_of(tn) + (('(%s)' % ', '.join(cx(a) for a in s[1][2])) if s[1][0] == 'ref' else '')
                    em.w('const double %s = %s; %s = 0.0;' % (gb, bl, bl))
                    if expr_active(s[2]):
                        node = fwd_prim(em, s[2]); back(em, node, gb)
                em.ind -= 1
                em.w('}')
        elif k == 'if':
            fl = cond_float(s[1])
            if mode == 'f':
                em.w('if (%s) {' % cx(s[1])); em.ind += 1; gen_block(em, s[2], mode, first, depth); em.ind -= 1
                if s[3]: em.w('} else {'); em.ind += 1; gen_block(em, s[3], mode, first, depth); em.ind -= 1
                em.w('}')
            elif mode == 'p':
                c = em.t('c'); em.w('{ const bool %s = %s;' % (c, cx(s[1])))
                em.w('if (%s) {' % c); em.ind += 1; gen_block(em, s[2], mode, first, depth); em.ind -= 1
                if s[3]: em.w('} else {'); em.ind += 1; gen_block(em, s[3], mode, first, depth); em.ind -= 1
                em.w('}')
                # the branch bit goes on after the branch's own pushes, so the reverse pops it first
                if fl: em.w('M9_PUSHB(%s);' % c)
                em.w('}')
            else:
                c = em.t('c'); em.w('{ const bool %s = %s;' % (c, 'M9_POPB()' if fl else cx(s[1])))
                em.w('if (%s) {' % c); em.ind += 1; gen_block(em, s[2], mode, first, depth); em.ind -= 1
                if s[3]: em.w('} else {'); em.ind += 1; gen_block(em, s[3], mode, first, depth); em.ind -= 1
                em.w('} }')
        elif k == 'loop':
            v, lo, hi, st, body = s[1:]
            n = em.t('n')
            em.w('{ const int %s_lo = %s, %s_hi = %s; const int %s = mynn_trip(%s_lo, %s_hi, %d);' % (n, cx(lo), n, cx(hi), n, n, n, st))
            if mode == 'p': em.w('M9_PUSH(__int_as_float(%s));' % v)
            if mode in ('f', 'p'):
                em.w('for (int it = 0; it < %s; ++it) { %s = %s_lo + it*(%d);' % (n, v, n, st))
                em.ind += 1; gen_block(em, body, mode, first, depth + 1); em.ind -= 1
                em.w('}')
                em.w('%s = %s_lo + %s*(%d); }' % (v, n, n, st))
            else:
                em.w('for (int it = %s - 1; it >= 0; --it) { %s = %s_lo + it*(%d);' % (n, v, n, st))
                em.ind += 1; gen_block(em, body, mode, first, depth + 1); em.ind -= 1
                em.w('}')
                em.w('%s = __float_as_int(M9_POP()); }' % v)
        elif k == 'fill':
            name, lo, hi, src = s[1:]
            los = cx(lo) if lo else '%s.lo[0]' % name; his = cx(hi) if hi else '%s.hi[0]' % name
            srcs = ('%s.linear(q - %s)' % (src[1], los)) if src[0] == 'arr' else cx(src)
            if src[0] == 'arr': srcs = '%s(q)' % src[1]
            nf = first.get(name) == 'w' and depth == 0
            if mode == 'f':
                em.w('for (int q = %s; q <= %s; ++q) %s(q) = %s;' % (los, his, name, srcs))
            elif mode == 'p':
                em.w('for (int q = %s; q <= %s; ++q) { %s%s(q) = %s; }' % (los, his, '' if fill_fresh(s, first, depth) else 'M9_PUSH(%s(q)); ' % name, name, srcs))
            else:
                em.w('for (int q = %s; q >= %s; --q) {' % (his, los)); em.ind += 1
                if not fill_fresh(s, first, depth): em.w('%s(q) = M9_POP();' % name)
                if active(name):
                    em.w('const double eb = %s(q); %s(q) = 0.0;' % (bar_of(name), bar_of(name)))
                    if src[0] == 'arr' and active(src[1]): em.w('%s(q) += eb;' % bar_of(src[1]))
                em.ind -= 1; em.w('}')

FIRST_WRITE_STMTS = set()
def s_needs_push(s, first, depth):
    return id(s) not in FIRST_WRITE_STMTS
def fill_fresh(s, first, depth):
    return id(s) in FIRST_WRITE_STMTS

def mark_fresh(sts, depth=0, seen=None, first=None):
    """a write needs no push when it is the block's first access to that variable, at top level of the block
    (scalars) or anywhere outside a nested re-write for arrays written by one statement per element"""
    for s in sts:
        k = s[0]
        if k == 'set':
            tn = target(s[1])
            r = []; reads(s[2], r)
            if s[1][0] == 'ref':
                for a in s[1][2]: reads(a, r)
            for n in r: seen.add(n)
            if tn not in seen and first.get(tn) == 'w' and (depth == 0 or tn in FARR) and tn not in ISCAL:
                FIRST_WRITE_STMTS.add(id(s))
            seen.add(tn)
        elif k == 'if':
            r = []; reads(s[1], r); seen.update(r)
            # a variable first written inside a branch is fresh in that branch; the join is conservative
            s1 = set(seen); mark_fresh(s[2], depth, s1, first)
            s2 = set(seen); mark_fresh(s[3], depth, s2, first)
            seen |= s1 | s2
        elif k == 'loop':
            seen.add(s[1]); mark_fresh(s[5], depth + 1, seen, first)
        elif k == 'fill':
            if s[1] not in seen and first.get(s[1]) == 'w' and depth == 0: FIRST_WRITE_STMTS.add(id(s))
            r = []; reads(s[4], r); seen.update(r); seen.add(s[1])

# A loop body that writes an array element and later (in the same body) re-writes the same array must push the
# second write; mark_fresh only exempts the first textual write per array per block, so re-writes push. Arrays
# written once per element by a loop (a(k) = ...) are fresh for every iteration. That is exact for the stage's
# loops: each writes v(k) at its own k, never v(k +- 1) before a read of it (checked below).
def check_loop_writes(sts):
    for s in sts:
        if s[0] == 'loop':
            for b in s[5]:
                if b[0] == 'set' and b[1][0] == 'ref' and id(b) in FIRST_WRITE_STMTS:
                    assert cx(b[1][2][0]) == s[1], ('fresh write off the loop index', cx(b[1]))
                    acc = []
                    def walk(x):
                        if isinstance(x, tuple):
                            if x and x[0] == 'ref' and x[1] == b[1][1] and cx(x[2][0]) != s[1]: acc.append(cx(x))
                            for y in x: walk(y)
                        elif isinstance(x, list):
                            for y in x: walk(y)
                    walk(s[5])
                    assert not acc, ('fresh loop write with other indices read', cx(b[1]), acc)
            check_loop_writes(s[5])
        elif s[0] == 'if':
            check_loop_writes(s[2]); check_loop_writes(s[3])

# ---------------------------------------------------------------- activity: every float variable but the frozen
for n in FARR | FSCAL:
    if n not in FROZEN: ACTIVE.add(n)

for bi, (b0, b1) in enumerate(blocks):
    sts = top[b0:b1]
    fa = {}
    for n, rw in accesses(sts, []):
        if n not in fa: fa[n] = rw
    mark_fresh(sts, 0, set(), fa)
    check_loop_writes(sts)

# stack bound per block: pushes per statement times trips (<= kte - kts + 2 per loop level)
def bound(sts, depth=0):
    tot = [0, 0, 0]   # coefficients of 1, L, L^2
    for s in sts:
        k = s[0]
        if k == 'set':
            if id(s) not in FIRST_WRITE_STMTS: tot[depth] += 1
        elif k == 'if':
            if cond_float(s[1]): tot[depth] += 1
            a = bound(s[2], depth); b = bound(s[3], depth)
            for j in range(3): tot[j] += max(a[j], b[j])
        elif k == 'loop':
            tot[depth] += 1
            a = bound(s[5], depth + 1)
            for j in range(3): tot[j] += a[j]
        elif k == 'fill':
            if id(s) not in FIRST_WRITE_STMTS: tot[depth + 1] += 1
    return tot
bounds = [bound(top[b0:b1]) for b0, b1 in blocks]
B0 = max(b[0] for b in bounds); B1 = max(b[1] for b in bounds); B2 = max(b[2] for b in bounds)
assert B2 == 0

# ---------------------------------------------------------------- write the stage
W = []
w = W.append
written_params = sorted({n for s in top for n, rw in accesses([s], []) if rw == 'w' and n in PARR})
w('// SPDX-License-Identifier: Apache-2.0')
w('// GENERATED by gen_m9.py from %s (x20 mynn-M9). Do not edit: regenerate.' % sys.argv[1].split('/')[-1])
w('// Stage M9 of the MYNN-EDMF c6d08709 reverse: f_mynn_tendencies (+ f_tridiag2, f_moisture_check inlined).')
w('// Blocks %d (cuts at top-level statements %s); stored value versions: %s.' % (len(blocks), cuts, versions))
w('// In-block value stack bound: %d + %d*(kte-kts+2) words. Written params (stage outputs): %s.' % (B0, B1, ', '.join(written_params)))
w('// Frozen (no adjoint kept, --freeze %s): %s.' % (FRZ_NAME, ', '.join(sorted(FROZEN & (set(PARR) | set(PSCAL))))))
w('#define M9_NPARR %d' % len(PARR))
w('#define M9_NPSCAL %d' % len(PSCAL))
w('#define M9_NPINT %d' % len(PINT))
w('#define M9_STK(L) (%d + %d*(L))' % (B0 + 4, B1 + 1))
w('// The stage calls the forward float routines through noinline wrappers: inlined, the glibc-exact r6_* bodies')
w('// took the AD entry past 50 GB in NVRTC/ptxas. Same routines, same results.')
for f_ in ('r6_add', 'r6_sub', 'r6_mul', 'r6_div'):
    w('__device__ __noinline__ float m9w_%s(float a, float b) { return %s(a, b); }' % (f_, f_))
w('__device__ __noinline__ float m9w_r6_neg(float a) { return r6_neg(a); }')
w('__device__ __noinline__ float m9w_mynn_exp(float a) { return mynn_exp(a); }')
w('struct M9P {')
for n in PARR: w('    MynnArray<float, 1> %s;' % n)
for n in PSCAL: w('    float %s;' % n)
for t, n in PINT: w('    %s %s;' % (t, n))
w('};')
w('struct M9B {')
for n in PARR:
    if n not in FROZEN: w('    MynnArray<double, 1> %s;' % n)
for n in PSCAL:
    if n not in FROZEN: w('    double %s;' % n)
w('};')
open(OUT + '.structs', 'w').write('\n'.join(W) + '\n')

# member functions (inside MynnColumn): one noinline function per block over a shared state struct, so the
# compiler sees small functions (one function for the whole stage took ptxas past 40 GB)
M = []
A = "*static_cast<%s*>(arena.allocate(sizeof(%s), sizeof(%s)))"
SF = []
for n in PARR: SF.append(('MynnArray<float, 1>', n))
for n in PSCAL: SF.append(('float', n))
for t, n in PINT: SF.append((t, n))
for n in LOCAL_ARR: SF.append(('MynnArray<float, 1>', n))
for n, t in LOCAL_SCAL.items(): SF.append(('float' if t == 'MR' else t, n))
for n in list(PARR) + list(LOCAL_ARR):
    if n not in FROZEN: SF.append(('MynnArray<double, 1>', bar_of(n)))
for n in list(PSCAL) + [n for n, t in LOCAL_SCAL.items() if t == 'MR']:
    if n not in FROZEN: SF.append(('double', bar_of(n)))
SF += [('MynnArray<float, 1>', 'm9_stk'), ('MynnArray<int, 1>', 'm9_bit'), ('int', 'm9_sp'), ('int', 'm9_bp'), ('int', 'm9_L'), ('bool', 'm9_ok')]
with open(OUT + '.structs', 'a') as f:
    f.write('struct M9S {\n' + ''.join('    %s %s;\n' % (t, n) for t, n in SF) + '};\n')

def used_names(sts):
    u = set()
    for n, rw in accesses(sts, []): u.add(n)
    def walk(x):
        if isinstance(x, tuple):
            if len(x) >= 2 and x[0] in ('var', 'ref', 'arr', 'meth') and isinstance(x[1], str): u.add(x[1])
            if len(x) >= 3 and x[0] == 'raw': u.add(x[2])
            for y in x: walk(y)
        elif isinstance(x, list):
            for y in x: walk(y)
    walk(sts)
    return u
def bind(em, sts=None):
    u = used_names(sts) if sts is not None else None
    for t, n in SF:
        base = n[:-2] if n.endswith('_b') else n
        if u is None or n.startswith('m9_') or base in u or n in ('kts', 'kte'):
            em.w('auto& %s = S.%s;' % (n, n))
    for n, (ty, v) in LOCAL_CONST.items(): em.w('constexpr %s %s = %s;' % (ty, n, v))

def init(em, with_bars):
    for n in PARR: em.w('S.%s = P.%s;' % (n, n))
    for n in PSCAL: em.w('S.%s = P.%s;' % (n, n))
    for t, n in PINT: em.w('S.%s = P.%s;' % (n, n))
    em.w('const int kts = P.kts, kte = P.kte;')
    for n, (lo, hi) in LOCAL_ARR.items():
        em.w('S.%s = mynn_array_any<float, 1, mynn_tile>(arena, {%s}, {%s});' % (n, cx(pexpr(lo)), cx(pexpr(hi))))
    for n, t in LOCAL_SCAL.items(): em.w('S.%s = 0;' % n)
    if with_bars:
        for n in PARR:
            if n not in FROZEN: em.w('S.%s = B.%s;' % (bar_of(n), n))
        for n in PSCAL:
            if n not in FROZEN: em.w('S.%s = B.%s;' % (bar_of(n), n))
        for n, (lo, hi) in LOCAL_ARR.items():
            if n not in FROZEN: em.w('S.%s = mynn_array_any<double, 1, mynn_tile>(arena, {%s}, {%s});' % (bar_of(n), cx(pexpr(lo)), cx(pexpr(hi))))
        for n, t in LOCAL_SCAL.items():
            if t == 'MR' and n not in FROZEN: em.w('S.%s = 0.0;' % bar_of(n))
        em.w('S.m9_L = kte - kts + 2;')
        em.w('S.m9_stk = mynn_array_any<float, 1, mynn_tile>(arena, {0}, {M9_STK(S.m9_L) - 1});')
        em.w('S.m9_bit = mynn_array_any<int, 1, mynn_tile>(arena, {0}, {(M9_STK(S.m9_L) + 31) / 32});')
        em.w('S.m9_sp = 0; S.m9_bp = 0; S.m9_ok = true;')

M.append('#define M9_PUSH(v) do { if (m9_sp < M9_STK(m9_L)) m9_stk(m9_sp) = (v); else m9_ok = false; ++m9_sp; } while (0)')
M.append('#define M9_POP() (m9_stk(--m9_sp))')
M.append('#define M9_PUSHB(c) do { const int w_ = m9_bp >> 5, s_ = m9_bp & 31; if (s_ == 0) m9_bit(w_) = 0; if (c) m9_bit(w_) |= 1 << s_; ++m9_bp; } while (0)')
M.append('#define M9_POPB() (--m9_bp, (m9_bit(m9_bp >> 5) >> (m9_bp & 31)) & 1)')
for bi, (b0, b1) in enumerate(blocks):
    em = Em()
    M.append('// block %d forward: top-level statements %d..%d' % (bi, b0, b1 - 1))
    M.append('__device__ __noinline__ void m9_bf_%d(M9S& S) {' % bi)
    bind(em, top[b0:b1]); gen_block(em, top[b0:b1], 'f', {})
    M += em.lines; M.append('}')
    em = Em()
    M.append('// block %d reverse: re-run with the in-block stack, then sweep backwards' % bi)
    M.append('__device__ __noinline__ void m9_br_%d(M9S& S) {' % bi)
    bind(em, top[b0:b1]); em.w('m9_sp = 0; m9_bp = 0;')
    gen_block(em, top[b0:b1], 'p', {}); gen_block(em, top[b0:b1], 'r', {})
    M += em.lines; M.append('}')

em = Em()
M.append('// Stage forward on plain float: bitwise the forward routine (same r6_*/mynn_* calls, same order).')
M.append('__device__ void m9_fwd(M9P& P) {')
M.append('    MynnFrame mynn_frame(arena);')
em.w('M9S& S = %s;   // in the column arena, not the thread frame' % (A % ('M9S', 'M9S', 'M9S')))
init(em, False)
for bi in range(len(blocks)): em.w('m9_bf_%d(S);' % bi)
M += em.lines; M.append('}')

em = Em()
M.append('// Fused reverse of stage M9. P: primal entry state (modified in place: it is the stage\'s own working copy).')
M.append('// B: adjoints, inout. On entry B holds the adjoints of the stage outputs (written params) and whatever the')
M.append('// caller has accumulated for the inputs; on exit it holds the adjoints with respect to the entry state.')
M.append('// Returns false only if the in-block stack bound is exceeded (a guard for a regenerated stage).')
M.append('__device__ bool m9_rev(M9P& P, M9B& B) {')
M.append('    MynnFrame mynn_frame(arena);')
em.w('M9S& S = %s;   // in the column arena, not the thread frame' % (A % ('M9S', 'M9S', 'M9S')))
init(em, True)
em.w('{')
em.ind += 1
for t, n in SF:
    if n not in ('kts', 'kte'): em.w('auto& %s = S.%s;' % (n, n))
for n, vb in versions:
    if n in FARR:
        em.w('MynnArray<float, 1> sn%d_%s = mynn_array_any<float, 1, mynn_tile>(arena, {%s.lo[0]}, {%s.hi[0]});' % (vb, n, n, n))
    else:
        em.w('%s sn%d_%s;' % ('float' if n in FSCAL else 'int', vb, n))
em.w('// ---- forward over the blocks, keeping one copy per value version the reverse needs')
for bi, (b0, b1) in enumerate(blocks):
    for n in ver_at[bi]:
        if n in FARR: em.w('for (int q = %s.lo[0]; q <= %s.hi[0]; ++q) sn%d_%s(q) = %s(q);' % (n, n, bi, n, n))
        else: em.w('sn%d_%s = %s;' % (bi, n, n))
    em.w('m9_bf_%d(S);' % bi)
em.w('// ---- reverse, block by block: restore what the block reads, re-run it with its in-block stack, sweep it')
for bi in reversed(range(len(blocks))):
    for n in snaps[bi]:
        vb = src_ver[bi][n]
        if n in FARR: em.w('for (int q = %s.lo[0]; q <= %s.hi[0]; ++q) %s(q) = sn%d_%s(q);' % (n, n, n, vb, n))
        else: em.w('%s = sn%d_%s;' % (n, vb, n))
    em.w('m9_br_%d(S);' % bi)
em.ind -= 1
em.w('}')
for n in PSCAL:
    if n not in FROZEN: em.w('B.%s = S.%s;' % (n, bar_of(n)))
em.w('return S.m9_ok;')
M += em.lines; M.append('}')
M.append('#undef M9_PUSH')
M.append('#undef M9_POP')
M.append('#undef M9_PUSHB')
M.append('#undef M9_POPB')

# ---------------------------------------------------------------- drop-in for the taped P3 kernel
# Record (MODE 1/2, MR = MynnTapeF): one marker node, then record slots (value, id pairs of every input word and
# the int params), then one leaf per output word. Reverse: ctx.reverse stops at the marker (kind 4), the column
# calls m9_ext_reverse, which rebuilds P from the record, seeds B from the output leaves, runs m9_rev and adds B
# into the recorded input ids.
sig = ', '.join(('%s %s' % (t, n)) for t, n in top_params)
R = []
R.append('__device__ void m9_ext_record(%s) {' % sig)
R.append('  if constexpr (mw_same<MR, MynnTapeF>::v) {')
R.append('    MynnAdContext* ctx = mw_ctxp[threadIdx.x & 31];')
R.append('    MynnFrame mynn_frame(arena);')
R.append('    M9P& P = %s;' % (A % ('M9P', 'M9P', 'M9P')))
for n in PARR:
    R.append('    %s.rebase({kts});' % n)
nwords = ' + '.join('%s.size()' % n for n in PARR)
R.append('    const int m9_nw = %s + %d;' % (nwords, len(PSCAL)))
R.append('    const int m9_nint = 4 + %d + 2*%d;' % (len(PINT), len(PARR)))
R.append('    const int m9_nrec = (2*m9_nw + m9_nint + 3) / 4;   // 16-byte node slots: 4 words each')
outw = ' + '.join('%s.size()' % n for n in written_params)
R.append('    const int m9_nout = %s;' % outw)
R.append('    const int m9_base = ctx->ext_reserve(1 + m9_nrec + m9_nout);')
R.append('    if (m9_base < 0) return;')
R.append('    const bool rec = !ctx->count_only;')
R.append('    int* iw = rec ? reinterpret_cast<int*>(ctx->nodes + m9_base + 1) : nullptr;')
R.append('    float* fw = rec ? reinterpret_cast<float*>(iw + m9_nint) : nullptr;')
R.append('    int* idw = rec ? reinterpret_cast<int*>(fw + m9_nw) : nullptr;')
R.append('    int o = 0;')
R.append('    if (rec) { iw[0] = m9_nw; iw[1] = m9_nint; iw[2] = m9_nout; iw[3] = m9_nrec;')
for j, (t, n) in enumerate(PINT): R.append('      iw[%d] = (int)%s;' % (4 + j, n))
for j, n in enumerate(PARR): R.append('      iw[%d] = %s.lo[0]; iw[%d] = %s.hi[0];' % (4 + len(PINT) + 2 * j, n, 5 + len(PINT) + 2 * j, n))
R.append('    }')
for n in PARR:
    R.append('    P.%s = mynn_array_any<float, 1, mynn_tile>(arena, {%s.lo[0]}, {%s.hi[0]});' % (n, n, n))
    R.append('    for (int q = %s.lo[0]; q <= %s.hi[0]; ++q, ++o) { const MR& e = %s(q); P.%s(q) = e.v; if (rec) { fw[o] = e.v; idw[o] = e.d.id; } }' % (n, n, n, n))
for n in PSCAL:
    R.append('    P.%s = %s.v; if (rec) { fw[o] = %s.v; idw[o] = %s.d.id; } ++o;' % (n, n, n, n))
for t, n in PINT: R.append('    P.%s = %s;' % (n, n))
R.append('    m9_fwd(P);')
R.append('    if (rec) { ctx->nodes[m9_base] = {-2, -2, (double)m9_nrec}; ctx->bars[m9_base] = 1.0;   // marker: left == -2')
R.append('      for (int s = 1; s <= m9_nrec + m9_nout; ++s) { ctx->bars[m9_base + s] = 0.0; } }')
R.append('    int leaf = m9_base + 1 + m9_nrec;')
for n in written_params:
    R.append('    for (int q = %s.lo[0]; q <= %s.hi[0]; ++q, ++leaf) { if (rec) ctx->nodes[leaf] = {-1, -1, 0.0}; %s(q) = MR(P.%s(q), MynnLinear(leaf, ctx)); }' % (n, n, n, n))
R.append('  }')
R.append('}')
R.append('')
R.append('__device__ bool m9_ext_reverse(MynnAdContext& ctx, int m9_base) {')
R.append('  MynnFrame mynn_frame(arena);')
R.append('  const int m9_nrec = (int)ctx.nodes[m9_base].weight;')
R.append('  const int* iw = reinterpret_cast<const int*>(ctx.nodes + m9_base + 1);')
R.append('  const int m9_nw = iw[0], m9_nint = iw[1], m9_nout = iw[2];')
R.append('  const float* fw = reinterpret_cast<const float*>(iw + m9_nint);')
R.append('  const int* idw = reinterpret_cast<const int*>(fw + m9_nw);')
R.append('  M9P& P = %s; M9B& B = %s;' % (A % ('M9P', 'M9P', 'M9P'), A % ('M9B', 'M9B', 'M9B')))
for j, (t, n) in enumerate(PINT): R.append('  P.%s = (%s)iw[%d];' % (n, t, 4 + j))
R.append('  int o = 0;')
for j, n in enumerate(PARR):
    lo = 'iw[%d]' % (4 + len(PINT) + 2 * j); hi = 'iw[%d]' % (5 + len(PINT) + 2 * j)
    R.append('  P.%s = mynn_array_any<float, 1, mynn_tile>(arena, {%s}, {%s});' % (n, lo, hi))
    R.append('  for (int q = %s; q <= %s; ++q) P.%s(q) = fw[o++];' % (lo, hi, n))
    if n not in FROZEN: R.append('  B.%s = mynn_array_any<double, 1, mynn_tile>(arena, {%s}, {%s});' % (n, lo, hi))
for n in PSCAL:
    R.append('  P.%s = fw[o++];' % n)
    if n not in FROZEN: R.append('  B.%s = 0.0;' % n)
R.append('  int leaf = m9_base + 1 + m9_nrec;')
for n in written_params:
    R.append('  for (int q = B.%s.lo[0]; q <= B.%s.hi[0]; ++q, ++leaf) B.%s(q) = ctx.bars[leaf];' % (n, n, n))
R.append('  const bool ok = m9_rev(P, B);')
R.append('  o = 0;')
for n in PARR:
    if n in FROZEN: R.append('  o += P.%s.size();' % n)
    else: R.append('  for (int q = B.%s.lo[0]; q <= B.%s.hi[0]; ++q, ++o) { const int id = idw[o]; if (id >= 0) ctx.bars[id] += B.%s(q); }' % (n, n, n))
for n in PSCAL:
    if n in FROZEN: R.append('  ++o;')
    else: R.append('  { const int id = idw[o++]; if (id >= 0) ctx.bars[id] += B.%s; }' % n)
R.append('  return ok;')
R.append('}')
M += R
open(OUT, 'w').write('\n'.join(M) + '\n')

print('blocks', len(blocks), 'cuts', cuts)
print('restores', [x for x in snaps if x]); print('versions', versions)
print('stack bound', B0, '+', B1, '*L')
print('written params', written_params)
print('params: arrays', len(PARR), 'scalars', len(PSCAL), 'ints', len(PINT))
print('local arrays', len(LOCAL_ARR), 'local scalars', len(LOCAL_SCAL))
