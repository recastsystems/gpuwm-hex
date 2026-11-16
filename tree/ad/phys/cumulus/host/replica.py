#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A processor replica of hex's New Tiedtke call (hexcore/mpasphys/ntiedtke_v842.py
NewTiedtkeV842.__call__ + _pipeline + the engine's _nt_walk), running the forward kernels' processor form
(libnthost.so, the same generated build as the proofs) on real 12 km columns, and writing every launch as an nthost
bundle (pre = the arguments before the launch, post = after). It stands in for the card capture while the card is
queued; it is checked against the capture when that lands.

Inputs: the case's own start (init.nc, 15:00Z): temperature from theta and the Exner function, moisture, the
reconstructed cell winds, full pressure, layer depths, density from the gas law, interface pressure by MPAS's fzm/fzp
weights, the start's w, xland, dx = len_disp / meshDensity^0.25. Not in a start file, set to zero: surface heat and
moisture fluxes and the dynamics/boundary-layer/radiation forcing (so these columns exercise the deep and mid-level
triggers and their closures, not the shallow trigger, which needs an upward surface buoyancy flux).
usage: replica.py LIB nt_kernels.json tables.json INIT.nc OUTDIR [LEN_DISP]"""
import ctypes, json, math, os, struct, sys
import numpy as np
import netCDF4

lib_path, kj, tj, init, out = sys.argv[1:6]
len_disp = float(sys.argv[6]) if len(sys.argv) > 6 else 12000.0
os.makedirs(out, exist_ok=True)
lib = ctypes.CDLL(lib_path)
lib.nt_call.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint),
                        ctypes.c_uint, ctypes.c_uint]
K = json.load(open(kj))['kernels']; T = json.load(open(tj))

d = netCDF4.Dataset(init)
f32 = np.float32
def v(name): return np.asarray(d.variables[name][0] if d.variables[name].ndim == 3 or name in ('xland', 'surface_pressure') else d.variables[name][:])
theta = v('theta').astype(f32); ncol, nz = theta.shape
p = (v('pressure_p') + v('pressure_base')).astype(f32)
exner = np.power(p / f32(1e5), f32(287.0 / 1004.5)).astype(f32)
t = (theta * exner).astype(f32)
qv = v('qv').astype(f32); qc = v('qc').astype(f32); qi = v('qi').astype(f32)
u = v('uReconstructZonal').astype(f32); vv = v('uReconstructMeridional').astype(f32)
zgrid = np.asarray(d.variables['zgrid'][:]).astype(f32)
dz = (zgrid[:, 1:] - zgrid[:, :-1]).astype(f32)
rho = (p / (f32(287.0) * t * (f32(1.0) + f32(0.608) * qv))).astype(f32)
fzm = np.asarray(d.variables['fzm'][:]).astype(f32); fzp = np.asarray(d.variables['fzp'][:]).astype(f32)
p8w = np.zeros((ncol, nz + 1), f32); p8w[:, 0] = v('surface_pressure').astype(f32)
p8w[:, 1:nz] = fzm[1:nz] * p[:, 1:nz] + fzp[1:nz] * p[:, 0:nz - 1]
p8w[:, nz] = np.maximum(f32(2.0) * p[:, nz - 1] - p8w[:, nz - 1], f32(1.0))
w = np.asarray(d.variables['w'][0]).astype(f32)
xland = v('xland').astype(f32)
dens = np.asarray(d.variables['meshDensity'][:]).astype(f32)
LV = lambda a: np.ascontiguousarray(a.T)          # (level, cell), bottom first
fields = {'t3d': LV(t), 'qv3d': LV(qv), 'qc3d': LV(qc), 'qi3d': LV(qi), 'u3d': LV(u), 'v3d': LV(vv), 'pcps': LV(p),
          'dz8w': LV(dz), 'rho3d': LV(rho), 'exner': LV(exner), 'p8w': LV(p8w), 'w': LV(w),
          'xland': xland.copy(), 'hfx': np.zeros(ncol, f32), 'qfx': np.zeros(ncol, f32)}
zero_lev = np.zeros((nz, ncol), f32)

# ---- the workspace (engine NtWorkspace: (nz+2, ncol) level arrays, (ncol) surface arrays, cutypen's scratch)
glob = {}
for key, tgt in T['resolve'].items():
    st, nm = key.split(':')
    if (st, nm) in (('ntiedtke_cududvn', 'pmfu'), ('ntiedtke_cududvn', 'pmfd')): continue
    glob.setdefault(nm, tgt)
res = lambda n: glob.get(n, n)
WS = {}
for names, dt in ((T['level_f'], np.float32), (T['level_i'], np.int32)):
    for n in names:
        if res(n) == n: WS[n] = np.zeros((nz + 2, ncol), dt)
for names, dt in ((T['surface_f'], np.float32), (T['surface_i'], np.int32)):
    for n in names:
        if res(n) == n: WS[n] = np.zeros(ncol, dt)
scr = np.zeros((10, nz + 2, ncol), f32); scr_i = np.zeros((1, nz + 2, ncol), np.int32)
def bind(name, base, stage=None):
    tgt = T['resolve'].get(f'{stage}:{name}', res(name)) if stage else res(name)
    a = WS[tgt]
    if a.ndim == 1 or base == 1: return a
    return a[1:]
geom = np.full(21, -1, np.int32); order = np.full(21, -1, np.int32); ticket = np.zeros(1, np.int32)
mask = np.zeros(1, np.int32); start = np.zeros(1, np.int32)
tpb = 32; nblocks = (ncol + tpb - 1) // tpb
initflag = 0
SC = {'ncol': ncol, 'nz': nz, 'klev': nz, 'stepcu': 1, 'itimestep': initflag, 'ktopm2': 2, 'tiedtke_closure': 0,
      'dt': f32(60.0), 'delt': f32(60.0), 'ztmst': f32(60.0), 'cp': f32(1004.5), 'rd': f32(287.0), 'rv': f32(461.6),
      'xlv': f32(2.5e6), 'xls': f32(2.85e6), 'xlf': f32(3.5e5), 'grav': f32(9.80616), 'expect_tpb': tpb,
      'expect_nblocks': nblocks, 'len_disp': f32(len_disp)}

def raw(val, typ):
    return struct.unpack('<I', struct.pack('<i' if typ == 'int' else '<f', int(val) if typ == 'int' else float(val)))[0]
def ev(expr, sc):
    return int(eval(expr, {'__builtins__': {}}, {k: int(x) if isinstance(x, (int, np.integer)) else x for k, x in sc.items()}))
nbundle = [0]
def launch(kname, arrays, sc, gx, bx):
    """arrays: param name -> numpy view (pointer params); sc: scalar values. Writes the launch's bundle."""
    k = K[kname]; ptrs, raws, slots = [], [], []
    W = ev(k['count'], sc)
    for a in k['args']:
        if a['cls'] == 'scalar':
            ptrs.append(None); raws.append(raw(sc[a['name']], a['type'])); continue
        arr = arrays[a['name']]
        assert arr.flags.c_contiguous or (arr.base is not None), a['name']
        ptrs.append(arr.ctypes.data); raws.append(0)
    def words(a, arr):
        if a['cls'] in ('F', 'I'):
            n = ev(a['rows'], sc) * W
            flat = np.frombuffer((ctypes.c_char * (4 * n)).from_address(arr.ctypes.data), dtype='<u4', count=n)
            return flat.copy()
        n = min(64, arr.size)
        return np.frombuffer((ctypes.c_char * (4 * n)).from_address(arr.ctypes.data), dtype='<u4', count=n).copy()
    pre = [None if a['cls'] == 'scalar' else words(a, arrays[a['name']]) for a in k['args']]
    P = (ctypes.c_void_p * len(ptrs))(*ptrs); R = (ctypes.c_uint * len(raws))(*raws)
    rc = lib.nt_call(kname.encode(), len(ptrs), P, R, gx, bx)
    assert rc == 0, kname
    post = [None if a['cls'] == 'scalar' else words(a, arrays[a['name']]) for a in k['args']]
    path = os.path.join(out, f'{nbundle[0]:02d}-{kname}.bin'); nbundle[0] += 1
    with open(path, 'wb') as f:
        f.write(b'NTB1'); f.write(kname.encode().ljust(64, b'\0')); f.write(struct.pack('<IIi', gx, bx, len(ptrs)))
        for a, rw, pw, qw in zip(k['args'], raws, pre, post):
            if a['cls'] == 'scalar':
                f.write(struct.pack('<iiq', 0, 1 if a['type'] == 'int' else 0, 0)); f.write(struct.pack('<I', rw))
            else:
                f.write(struct.pack('<iiq', 1, 1 if a['type'] == 'int' else 0, len(pw))); f.write(b'\0\0\0\0')
                f.write(pw.tobytes()); f.write(qw.tobytes())

def stage(name):
    base = T['base'].get(name, 1); arrays = {}
    for a in K[name]['args']:
        n = a['name']
        if a['cls'] == 'scalar': continue
        if n == 'scr': arrays[n] = scr
        elif n == 'scr_i': arrays[n] = scr_i
        elif n == 'llo3_mask': arrays[n] = mask
        elif n == 'mpas_llo3_start': arrays[n] = start
        elif n == 'geom_report': arrays[n] = geom
        elif n == 'order_report': arrays[n] = order
        elif n == 'ticket': arrays[n] = ticket
        else: arrays[n] = bind(n, base, name)
    launch(name, arrays, SC, nblocks, tpb)

# ---- hex convection_from_MPAS: the forcing fold and dx (zero forcing here), then __call__'s fills and binds
ten = {n: zero_lev.copy() for n in ('rthdynten', 'rthblten', 'rthratenlw', 'rthratensw', 'rqvdynten', 'rqvblten')}
thften = np.empty((nz, ncol), f32); qvften = np.empty((nz, ncol), f32)
size = nz * ncol
launch('mpas_ntiedtke_forcing', dict(ten, exner=fields['exner'], thften=thften, qvften=qvften), {'size': size}, (size + 127) // 128, 128)
dx = np.empty(ncol, f32)
launch('mpas_ntiedtke_dx', {'meshDensity': dens, 'dx': dx}, {'ncol': ncol, 'len_disp': SC['len_disp']}, (ncol + 127) // 128, 128)
fields.update(thften=thften, qvften=qvften, dx=dx)
for n in ('t3d', 'qv3d', 'qc3d', 'qi3d', 'u3d', 'v3d', 'pcps', 'dz8w', 'rho3d', 'exner', 'qvften', 'thften'):
    bind(n, 0)[:nz, :] = fields[n]
for n in ('p8w', 'w'):
    bind(n, 0)[:nz + 1, :] = fields[n]
for n in ('xland', 'hfx', 'qfx', 'dx'):
    bind(n, 1)[:] = fields[n]
# ---- the engine's _nt_walk (dry_mass_coupling False), with hex's ascent latch
for n in ('pcte', 'ptenu', 'ptenv'): bind(n, 1).fill(0)
stage('ntiedtke_prep'); stage('ntiedtke_convert')
bind('zqq', 1)[...] = bind('pqte', 1); bind('ztt', 1)[...] = bind('ptte', 1)
for st in T['call_order']:
    if st in ('ntiedtke_prep', 'ntiedtke_convert'): continue
    if st == 'ntiedtke_cuascn':
        start.fill(0)
        launch('mpas_ntiedtke_ascent_start', {n: bind(n, 1) for n in ('ldcum', 'ktype', 'klab', 'pqen', 'pqsen', 'pgeo')} | {'start': start},
               {'ncol': ncol, 'klev': nz, 'grav': SC['grav']}, nblocks, tpb)
    if st == 'ntiedtke_cududvn':
        bind('ztenu', 1)[...] = bind('pvom', 1); bind('ztenv', 1)[...] = bind('pvol', 1)
    stage(st)
kt = bind('ktype', 1)
rc = bind('raincv', 1)
summary = {'ncol': ncol, 'nz': nz, 'bundles': nbundle[0], 'ktype_counts': {int(x): int((kt == x).sum()) for x in np.unique(kt)},
           'ldcum': int(bind('ldcum', 1).sum()), 'raincv_nonzero': int((rc != 0).sum()), 'raincv_max': float(rc.max()),
           'rthcuten_absmax': float(np.abs(bind('rthcuten', 0)[:nz]).max()), 'geom_report': geom.tolist(),
           'start': int(start[0]), 'len_disp': len_disp, 'inputs': init}
json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
print(json.dumps(summary))
