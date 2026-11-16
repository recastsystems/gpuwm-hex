#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# step6-rucland: add the oracle_land_ruc_land row (kind ruc_land) and its two images to an L2 scheme bundle.
# Usage: ruc_land_row.py <bundle in> <bundle out> <entry fatbin> <bind cubin> <entry generated dir (land_R_*.inc, land_R_fields.json)>
# The row is data read from the entry's own generated sources (gen_land_R.py output, committed with the entry), so the
# slots it names are exactly the slots the entry reads:
#   fields          land_R_fields.json: the 70 float fields (name, slot), sized at bind time by the entry's __sizes;
#   cell_slots      every `((int*)ptrs[K])[c]`, `((float*)ptrs[K])[c]` and `FArray<int> x{((int*)ptrs[K])+c, ...}` of the
#                   entry prologue and call: one word per cell (the bind checks the 1-D fields are one word per cell too);
#   tile_slot 119   the tile stop pair (int[2], read whole);  ilnb_slot 120  ilnb (int per cell, optional);
#   untouched 121   snowtemp_calls, a diagnostic counter the derivative entries never touch.
import hashlib, json, os, re, shutil, sys

src, dst, fatbin, bind, gen = sys.argv[1:6]
pre = open(os.path.join(gen, 'land_R_entry_pre.inc')).read()
call = open(os.path.join(gen, 'land_R_entry_call.inc')).read()
fields = json.load(open(os.path.join(gen, 'land_R_fields.json')))['fields']
fk = {k for _, k in fields}
cells = set()
for pat in (r'\(\(int\*\)ptrs\[(\d+)\]\)\[c\]', r'\(\(float\*\)ptrs\[(\d+)\]\)\[c\]', r'FArray<int> \w+\{\(\(int\*\)ptrs\[(\d+)\]\)\+c'):
    for t in (pre, call):
        cells.update(int(m) for m in re.findall(pat, t))
assert not cells & fk, cells & fk
assert {119, 120, 121}.isdisjoint(cells | fk)
assert 'ptrs[119]' in pre and 'ptrs[120]' in pre, 'tile stop / ilnb slots moved'
named = cells | fk | {119, 120, 121}
other = sorted(set(int(m) for m in re.findall(r'ptrs\[(\d+)\]', pre + call)) - named)
assert not other, f'entry reads slots the row would not name: {other}'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()

if os.path.exists(dst): sys.exit(f'{dst} exists')
shutil.copytree(src, dst, symlinks=False)
os.makedirs(os.path.join(dst, 'ruc_land'))
shutil.copy(fatbin, os.path.join(dst, 'ruc_land', 'oracle_land_ruc_land.ad.fatbin'))
shutil.copy(bind, os.path.join(dst, 'ruc_land', 'ruc_land_bind-sm_120.cubin'))
t = json.load(open(os.path.join(dst, 'schemes.json')))
assert not any(r['function'] == 'oracle_land_ruc_land' for r in t['rows'])
t['images']['ruc_land'] = 'ruc_land/oracle_land_ruc_land.ad.fatbin'
t['images']['ruc_land_bind'] = 'ruc_land/ruc_land_bind-sm_120.cubin'
t['rows'].append({
    'function': 'oracle_land_ruc_land', 'unit': 'RUC-land', 'kind': 'ruc_land', 'p8_class': 'entry', 'fixed': {},
    'lane': 'ad/step6-rucland (entry: ad/held-oracle_land_ruc_land 4aa1b7fb)',
    'forward': 'hexcore/mpasphys/rrfs/ruc_lsm_8c9f1a8.py:5102 (RRFS-MPAS 1d6e370e module_ruc_land.F90:104-5872), captured source e6fd85f0',
    'entry_image_sha256': sha(fatbin), 'bind_image_sha256': sha(bind),
    'args': [
        {'name': 'ptrs', 'pos': 0, 'device': True, 'dtype': '<u8', 'extent': '122', 'role': 'pointer-table', 'active': False},
        {'name': 'tables', 'pos': 1, 'device': True, 'dtype': '<f4', 'extent': None, 'role': 'const', 'active': False},
        {'name': 'work', 'pos': 2, 'device': True, 'dtype': '<f4', 'extent': None, 'role': 'scratch-arena', 'active': False},
        {'name': 'error', 'pos': 3, 'device': True, 'dtype': '<i4', 'extent': None, 'role': 'error', 'active': False},
        {'name': 'nc', 'pos': 4, 'device': False, 'dtype': '<i4', 'role': 'scalar', 'active': False}],
    'ruc_land': {'pointer_slots': 122, 'nwork': 619, 'fields': fields, 'cell_slots': sorted(cells),
                 'tile_slot': 119, 'ilnb_slot': 120, 'untouched_slots': [121],
                 'precision': 'float32 primal recomputed by the forward\'s own operations (generated re-typing); partials and '
                              'accumulation in double, each written tangent / cotangent word rounded once to float32 (+0.f)',
                 'frozen': 'every comparison, goto, int conversion, table index, min/max/fabs choice and iteration count on the '
                           'pre-launch primal words (executed arm only); tables, ruc_tbq_table and the scalars dt, cp, rovcp, g0, '
                           'lv, stbolt, xice_threshold inactive',
                 'in_place': 'tangent: tan[k] <- J tan[k] over the 70 float fields; adjoint: bar[k] <- J^T bar[k]; untouched '
                             'words pass through'}})
json.dump(t, open(os.path.join(dst, 'schemes.json'), 'w'), indent=1)
with open(os.path.join(dst, 'SHA256SUMS'), 'w') as o:
    for root, _, fs in sorted(os.walk(dst)):
        for f in sorted(fs):
            if f.startswith('SHA256SUMS'): continue
            q = os.path.join(root, f)
            o.write(f'{sha(q)}  {os.path.relpath(q, dst)}\n')
print(json.dumps({'cell_slots': sorted(cells), 'fields': len(fields), 'out': dst}))
