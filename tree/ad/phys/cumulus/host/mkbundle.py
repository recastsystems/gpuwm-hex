#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One captured New Tiedtke launch (ad.hex.launch/v1 manifest, the capture
observer) -> an nthost bundle: every argument in the forward's order, each pointer's words from the thread's column
layout (rows x W, rows from nt_kernels.json evaluated on the launch's own scalars), pre and post.
usage: mkbundle.py LAUNCH_DIR nt_kernels.json OUT.bin [--cols N]   (--cols: keep the first N columns only)"""
import hashlib, json, os, re, struct, sys
import numpy as np

ldir, kj, outp = sys.argv[1:4]
ncols_keep = int(sys.argv[sys.argv.index('--cols') + 1]) if '--cols' in sys.argv else None
m = json.load(open(os.path.join(ldir, 'manifest.json')))
K = json.load(open(kj))
k = K['kernels'][m['kernel']]
root = ldir
def objpath(p):
    for base in (ldir, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(ldir.rstrip('/')))))):
        q = os.path.join(base, p)
        if os.path.exists(q): return q
    raise SystemExit('object not found ' + p)
cache = {}
def alloc(i, which):
    key = (i, which)
    if key in cache: return cache[key]
    a = m['allocations'][i][which]
    if a.get('encoding') == 'raw-concat-v1':
        data = b''.join(open(objpath(c['path']), 'rb').read() for c in a['chunks'])
    else:
        data = open(objpath(a['path']), 'rb').read()
    assert len(data) == a['bytes'], (i, which)
    cache[key] = data
    return data
args = {a['name']: a for a in m['arguments']}
scal = {}
for a in m['arguments']:
    if a['kind'] == 'scalar':
        raw = bytes.fromhex(a['value_hex'])
        scal[a['name']] = struct.unpack('<i' if a['dtype'] in ('<i4', 'i32') else '<f', raw)[0]
def ev(expr):
    return int(eval(expr, {'__builtins__': {}}, dict(scal)))
W = ev(k['count'])
keep = W if ncols_keep is None else min(W, ncols_keep)
slots = []
for a in k['args']:
    c = args[a['name']]
    if a['cls'] == 'scalar':
        raw = bytes.fromhex(c['value_hex'])
        if a['name'] == k['count'] and keep != W: raw = struct.pack('<i', keep)
        slots.append((0, 1 if a['type'] == 'int' else 0, 0, raw, b'', b''))
        continue
    pre, post = alloc(c['allocation'], 'pre'), alloc(c['allocation'], 'post')
    off = c['offset']
    if a['cls'] in ('F', 'I'):
        rows = ev(a['rows'])
        n = rows * W
        P = np.frombuffer(pre, dtype='<u4', count=n, offset=off).reshape(rows, W)[:, :keep]
        Q = np.frombuffer(post, dtype='<u4', count=n, offset=off).reshape(rows, W)[:, :keep]
        pw, qw = np.ascontiguousarray(P).tobytes(), np.ascontiguousarray(Q).tobytes()
        nw = rows * keep
    else:   # ro / dummy: a few words
        nw = min(64, (len(pre) - off) // 4)
        pw = pre[off:off + 4 * nw]; qw = post[off:off + 4 * nw]
    slots.append((1, 1 if a['type'] == 'int' else 0, nw, b'\0\0\0\0', pw, qw))
with open(outp, 'wb') as f:
    f.write(b'NTB1'); f.write(m['kernel'].encode().ljust(64, b'\0'))
    f.write(struct.pack('<IIi', m['geometry']['grid'][0], m['geometry']['block'][0], len(slots)))
    for kind, dt, nw, raw, pw, qw in slots:
        f.write(struct.pack('<iiq', kind, dt, nw)); f.write(raw); f.write(pw); f.write(qw)
src = m.get('source') or {}
print(json.dumps({'kernel': m['kernel'], 'step': m['step'], 'W': W, 'kept': keep, 'slots': len(slots),
                  'source_sha256': src.get('sha256'), 'generator_source_sha256': K['source_sha256']}))
