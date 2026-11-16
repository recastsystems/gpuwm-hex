#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The tangent and taped-reverse entries of every New Tiedtke
launch, generated from the forward's own NVRTC text (hexcore.mpasphys.ntiedtke_v842.cuda_source(), the image the
recording runs), the way the RUC land entries were made.

Mechanical, no hand edits of the physics:
  * the forward region (from nt_geometry_ok to the end: the 20 stages, ntiedtke_midlevel and hex's four
    mpas_ntiedtke_* kernels) is re-typed over a scalar R: `float` -> R, and each f32 operation the forward calls
    (mpas_nt_add/sub/mul/div/sqrt/neg, fmaxf, fminf, fabsf, sqrtf, gfk_exp/log/pow, mpas_nt_int, mpas_nt_nan) -> an r_*
    overload whose primal word is that same forward function (nt_ad.cuh);
  * every kernel becomes `__device__ void NAME__R(...)`, included twice: namespace tl (R = AF) and adr (R = AR);
  * per kernel, a tangent entry NAME__tl and a taped reverse NAME__ad bind the forward's arguments, shadow every
    column array of the launch thread's own column in a workspace (f32 arrays as R words, int arrays as copies,
    the launch-report words as dummies), run the re-typed body, and write tangents / adjoints.
The pre-launch state is the trajectory (the binder's primal binding); TL and AD never write a primal word.

usage: gen_nt.py FORWARD.cu TABLES.json OUTDIR
  TABLES.json: the engine's own name tables (nt_tables.py): level/surface float/int names, aliases, level bases.
Writes OUTDIR/nt_R.inc, nt_entries.inc, nt_kernels.json (argument classes, rows, workspace layout per kernel)."""
import hashlib, json, re, sys

src_path, tab_path, out = sys.argv[1:4]
SRC = open(src_path).read()
SHA = hashlib.sha256(SRC.encode()).hexdigest()
T = json.load(open(tab_path))
L = SRC.split('\n')

start = next((i for i, s in enumerate(L) if s.startswith('__device__ __forceinline__ bool nt_geometry_ok(')),
             next(i for i, s in enumerate(L) if s.startswith('extern "C" __global__')))
region = '\n'.join(L[start:])

# ---------------------------------------------------------------- kernels and their parameters
KRE = re.compile(r'extern "C" __global__ void (\w+)\(')
def params_of(text, pos):
    depth, i = 1, pos
    while depth:
        depth += (text[i] == '(') - (text[i] == ')'); i += 1
    body = text[pos:i - 1]
    body = re.sub(r'/\*.*?\*/', '', body, flags=re.S); body = re.sub(r'//[^\n]*', '', body)
    out, depth, cur = [], 0, ''
    for ch in body:
        if ch == ',' and depth == 0: out.append(cur); cur = ''; continue
        depth += (ch == '(') - (ch == ')'); cur += ch
    out.append(cur)
    ps = []
    for p in out:
        p = ' '.join(p.split())
        m = re.match(r'^(const )?(float|int|bool) (\*)?\s*(?:__restrict__ )?(\w+)$', p.replace('* ', '*').replace('*__restrict__ ', '*'))
        if not m:
            m2 = re.match(r'^(const\s+)?(float|int|bool)\s*(\*)?\s*(?:__restrict__\s+)?(\w+)$', p)
            m = m2
        if not m: raise SystemExit(f'cannot parse parameter {p!r}')
        ps.append({'const': bool(m.group(1)), 'type': m.group(2), 'ptr': bool(m.group(3)), 'name': m.group(4)})
    return ps

kernels = []
for m in KRE.finditer(region):
    kernels.append((m.group(1), params_of(region, m.end())))
names = [k for k, _ in kernels]

# ---------------------------------------------------------------- argument classes
LEVEL = set(T['level_f']) | set(T['level_i'])
SURF = set(T['surface_f']) | set(T['surface_i'])
GEOM_DUMMY = {'geom_report': 64, 'order_report': 64, 'ticket': 1}
RO_NONCOLUMN = {'llo3_mask', 'mpas_llo3_start'}
HEX = {   # hex's own kernels in ntiedtke_v842.DRIVER_CUDA: (count scalar, per-pointer rows, written non-column int)
    'mpas_ntiedtke_forcing': ('size', {}, {}),
    'mpas_ntiedtke_dx': ('ncol', {}, {}),
    'rrfs_convection_forcing': ('n', {}, {}),   # driver_rrfs.py _CONVECTION_FORCING_CUDA: the driver's forcing fold
    'mpas_ntiedtke_init_state': ('1', {'out': '18'}, {}),
    'mpas_ntiedtke_ascent_start': ('ncol', {'klab': 'klev+2', 'pqen': 'klev+2', 'pqsen': 'klev+2', 'pgeo': 'klev+2'},
                                   {'start': 1}),
}
def classify(kname, ps):
    pn = [p['name'] for p in ps]
    if kname in HEX:
        count, rows, dummies = HEX[kname]
    else:
        count = 'ncol'
        lev = 'nz' if 'nz' in pn else 'klev'
        base = T['base'].get(kname, 1)
        rows, dummies = {}, dict(GEOM_DUMMY)
        for p in ps:
            if not p['ptr'] or p['name'] in RO_NONCOLUMN or p['name'] in GEOM_DUMMY: continue
            n = p['name']
            if n == 'scr': rows[n] = f'10*({lev}+2)'
            elif n == 'scr_i': rows[n] = f'({lev}+2)'
            else:
                t = T['resolve'].get(f'{kname}:{n}', n)
                if t in LEVEL: rows[n] = f'({lev}+{2 - (1 if base == 0 else 0)})'
                elif t in SURF: rows[n] = '1'
                else: raise SystemExit(f'{kname}: no class for {n} (resolves to {t})')
    args = []
    owner = {}
    for p in ps:
        a = dict(p)
        if not p['ptr']: a['cls'] = 'scalar'
        elif p['name'] in RO_NONCOLUMN: a['cls'] = 'ro'
        elif p['name'] in dummies: a['cls'] = 'dummy'; a['words'] = dummies[p['name']]
        elif p['type'] == 'float':
            a['cls'] = 'F'; a['rows'] = rows.get(p['name'], '1')
        else:
            a['cls'] = 'I'; a['rows'] = rows.get(p['name'], '1')
        # storage aliases inside one launch (ntiedtke_post_conversion: pqc = qcf, pqi = qif, pu = uf, pv = vf)
        if a['cls'] in ('F', 'I') and kname not in HEX:
            tgt = T['resolve'].get(f'{kname}:{p["name"]}', p['name'])
            if tgt in owner: a['alias_of'] = owner[tgt]
            else: owner[tgt] = p['name']
        args.append(a)
    for a in args:
        if a['cls'] == 'F' and 'alias_of' not in a:
            group = [b for b in args if b['name'] == a['name'] or b.get('alias_of') == a['name']]
            a['written'] = any(not b['const'] for b in group)
    return count, args

K = {}
for kname, ps in kernels:
    count, args = classify(kname, ps)
    K[kname] = {'count': count, 'args': args}

# ---------------------------------------------------------------- the re-typed region
RENAME = (('mpas_nt_add', 'r_add'), ('mpas_nt_sub', 'r_sub'), ('mpas_nt_mul', 'r_mul'), ('mpas_nt_div', 'r_div'),
          ('mpas_nt_sqrt', 'r_sqrt'), ('mpas_nt_neg', 'r_neg'), ('mpas_nt_int', 'r_ntint'), ('mpas_nt_nan', 'r_nan'),
          ('fmaxf', 'r_max'), ('fminf', 'r_min'), ('fabsf', 'r_fabs'), ('sqrtf', 'r_sqrtf'),
          ('gfk_exp', 'r_exp'), ('gfk_log', 'r_log'), ('gfk_pow', 'r_pow'),
          ('__fadd_rn', 'r_fadd'), ('__fsub_rn', 'r_fsub'), ('__fmul_rn', 'r_fmul'), ('__fdiv_rn', 'r_fdiv'))
def retype(s):
    s = KRE.sub(lambda m: f'__device__ void {m.group(1)}__R(', s)
    s = re.sub(r'\bfloat\b', 'R', s)
    for a, b in RENAME:
        s = re.sub(r'\b' + a + r'\s*\(', b + '(', s)
    return s
hdr = (f'// GENERATED by gen_nt.py from the forward New Tiedtke source sha256 {SHA}\n'
       f'// re-typed region: lines {start + 1}-{len(L)}. Do not edit.\n')
open(f'{out}/nt_R.inc', 'w', newline='\n').write(hdr + retype(region) + '\n')

# ---------------------------------------------------------------- entries
def decl(a):
    if a['cls'] == 'scalar': return f"{a['type']} {a['name']}"
    c = 'const ' if (a['const'] or a['cls'] in ('F', 'ro')) else ''
    return f"{c}{a['type']} *{a['name']}"
def body_arg(a):
    if a['cls'] == 'scalar': return a['name']
    if a['cls'] == 'ro': return a['name']
    if a['cls'] == 'dummy': return f"ntd_{a['name']}"
    if a['cls'] in ('F', 'I'): return f"nts_{a.get('alias_of', a['name'])}"
    raise SystemExit(a)
def layout(k, word):
    """Workspace regions in a fixed order; the same text is the spec's byte expression."""
    regs = []
    for a in k['args']:
        if a['cls'] == 'F' and 'alias_of' not in a: regs.append(('F', a['name'], f"{word}*({a['rows']})*({k['count']})"))
    return regs
E = [hdr, '#ifndef NT_DT\n#define NT_DT float   // derivative word type in memory (double in the processor proofs)\n#endif\n']
for kname, k in K.items():
    A = k['args']; W = k['count']
    F = [a for a in A if a['cls'] == 'F' and 'alias_of' not in a]
    Iw = [a for a in A if a['cls'] == 'I' and 'alias_of' not in a]
    D = [a for a in A if a['cls'] == 'dummy']
    k['F'] = [{'name': a['name'], 'rows': a['rows'], 'written': a['written']} for a in F]
    k['I'] = [{'name': a['name'], 'rows': a['rows']} for a in Iw]
    k['dummy'] = [{'name': a['name'], 'words': a['words']} for a in D]
    fwd = ', '.join(decl(a) for a in A)
    call = ', '.join(body_arg(a) for a in A)
    nslot = ' + '.join(f"({a['rows']})" for a in F) or '0'
    k['nslot'] = nslot
    def common(word, ns):
        s = [f' const int nt_t = blockIdx.x * blockDim.x + threadIdx.x; const int nt_W = {W}; if (nt_t >= nt_W) return;',
             ' unsigned char *nt_q = ws;']
        for a in F:
            s.append(f" {ns}::R *nts_{a['name']} = ({ns}::R *)nt_q; nt_q += (size_t){word} * (size_t)({a['rows']}) * (size_t)nt_W;")
        return s
    def tail_regions():
        s = []
        for a in Iw:
            s.append(f" int *nts_{a['name']} = (int *)nt_q; nt_q += (size_t)4 * (size_t)({a['rows']}) * (size_t)nt_W;")
            s.append(f" for (int nt_r = 0; nt_r < ({a['rows']}); ++nt_r) {{ const size_t nt_w = (size_t)nt_r * nt_W + nt_t; nts_{a['name']}[nt_w] = {a['name']}[nt_w]; }}")
        for a in D:
            s.append(f" int *ntd_{a['name']} = (int *)nt_q; nt_q += (size_t)4 * {a['words']};")
        return s
    ws_int = ' + '.join([f"4*({a['rows']})*({W})" for a in Iw] + [f"4*{a['words']}" for a in D]) or '0'
    # ---- tangent
    tin = [f"const NT_DT *ntt_{a['name']}" for a in F]
    tout = [f"NT_DT *nto_{a['name']}" for a in F if a['written']]
    sig = ', '.join([fwd] + tin + tout + ['unsigned char *ws', 'int *failure'])
    E.append(f'extern "C" __global__ void {kname}__tl({sig}) {{')
    E += common(16, 'tl')
    for a in F:
        E.append(f" for (int nt_r = 0; nt_r < ({a['rows']}); ++nt_r) {{ const size_t nt_w = (size_t)nt_r * nt_W + nt_t; nts_{a['name']}[nt_w] = AF({a['name']}[nt_w], (double)ntt_{a['name']}[nt_w]); }}")
    E += tail_regions()
    E.append(f' tl::{kname}__R({call});')
    for a in F:
        if not a['written']: continue
        E.append(f" for (int nt_r = 0; nt_r < ({a['rows']}); ++nt_r) {{ const size_t nt_w = (size_t)nt_r * nt_W + nt_t; const AF nt_x = nts_{a['name']}[nt_w];"
                 f" const NT_DT nt_o = (NT_DT)nt_x.d + (NT_DT)0; nto_{a['name']}[nt_w] = nt_o; if (isfinite(nt_x.v) && !isfinite(nt_o)) failure[0] = 1; }}")
    E.append('}')
    k['tl_ws'] = f"16*({nslot})*({W}) + {ws_int}"
    # ---- taped reverse
    bin_ = [f"const NT_DT *ntb_{a['name']}" for a in F if a['written']]
    bacc = [f"NT_DT *nta_{a['name']}" for a in F]
    sig = ', '.join([fwd] + bin_ + bacc + ['int nt_cap', 'unsigned char *ws', 'int *failure'])
    E.append(f'extern "C" __global__ void {kname}__ad({sig}) {{')
    E += common(8, 'adr')
    E.append(' NtTapeE *nttape = (NtTapeE *)nt_q; nt_q += (size_t)24 * (size_t)nt_cap * (size_t)nt_W;')
    E.append(' double *ntadj = (double *)nt_q; nt_q += (size_t)8 * (size_t)nt_cap * (size_t)nt_W;')
    E.append(f' const int nt_nslot = {nslot};')
    E.append(' double *ntadjin = (double *)nt_q; nt_q += (size_t)8 * (size_t)nt_nslot * (size_t)nt_W;')
    E.append(' int *ntcnt = (int *)nt_q; nt_q += (size_t)4 * (size_t)nt_W; int *ntovf = (int *)nt_q; nt_q += (size_t)4 * (size_t)nt_W;')
    E.append(' g_nt.tape = nttape; g_nt.adj = ntadj; g_nt.adjin = ntadjin; g_nt.cnt = ntcnt; g_nt.ovf = ntovf; g_nt.cap = nt_cap; g_nt.W = nt_W;')
    E.append(' ntcnt[nt_t] = 0; ntovf[nt_t] = 0; for (int nt_s = 0; nt_s < nt_nslot; ++nt_s) ntadjin[(size_t)nt_s * nt_W + nt_t] = 0.0;')
    E.append(' int nt_off = 0;')
    for a in F:
        E.append(f" for (int nt_r = 0; nt_r < ({a['rows']}); ++nt_r) {{ const size_t nt_w = (size_t)nt_r * nt_W + nt_t; nts_{a['name']}[nt_w] = AR({a['name']}[nt_w], -(2 + nt_off + nt_r)); }} nt_off += ({a['rows']});")
    E += tail_regions()
    E.append(f' adr::{kname}__R({call});')
    E.append(' const int nt_n = ntcnt[nt_t] < nt_cap ? ntcnt[nt_t] : nt_cap; for (int nt_e = 0; nt_e < nt_n; ++nt_e) ntadj[(size_t)nt_e * nt_W + nt_t] = 0.0;')
    for a in F:
        if not a['written']: continue
        E.append(f" for (int nt_r = 0; nt_r < ({a['rows']}); ++nt_r) {{ const size_t nt_w = (size_t)nt_r * nt_W + nt_t; nt_acc(nt_t, nts_{a['name']}[nt_w].id, (double)ntb_{a['name']}[nt_w]); }}")
    E.append(' nt_sweep(nt_t);')
    E.append(' nt_off = 0;')
    for a in F:
        E.append(f" for (int nt_r = 0; nt_r < ({a['rows']}); ++nt_r) {{ const size_t nt_w = (size_t)nt_r * nt_W + nt_t; const double nt_g = ntadjin[(size_t)(nt_off + nt_r) * nt_W + nt_t];"
                 f" if (nt_g != 0.0) {{ const NT_DT nt_o = nta_{a['name']}[nt_w] + (NT_DT)nt_g; nta_{a['name']}[nt_w] = nt_o; if (!isfinite(nt_o)) failure[0] = 1; }} }} nt_off += ({a['rows']});")
    E.append(' if (ntovf[nt_t]) failure[0] = 2;')
    E.append('}')
    k['ad_ws'] = f"8*({nslot})*({W}) + 32*nt_cap*({W}) + 8*({nslot})*({W}) + 8*({W}) + {ws_int}"
    # tape census entry (host / card: tape length per column, no reverse) is the AD entry with cap large and n read back
open(f'{out}/nt_entries.inc', 'w', newline='\n').write('\n'.join(E) + '\n')
json.dump({'source_sha256': SHA, 'region_first_line': start + 1, 'kernels': K}, open(f'{out}/nt_kernels.json', 'w'), indent=1)
print('kernels', len(K), ' '.join(K))
