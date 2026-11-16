#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The ad.entry/v1 specs and kernel-set rows of every New Tiedtke launch, from
nt_kernels.json (gen_nt.py) and the tape census (nthost dot receipts: the largest tape a column needed per kernel).
Each entry binds the forward's arguments in its own order (scalars; pointers as the pre-launch primal), then the
derivative words: tangent in for every f32 column array, tangent out for those the launch writes; adjoint in (the
seed) for the written arrays and adjoint accumulate for every array; one workspace (shadows and, for the reverse,
the tape); failure slots (1: a non-finite derivative word where the primal is finite; 2: tape over capacity).
usage: mkspec_nt.py nt_kernels.json CENSUS.json OUT_DIR [GROUP]
  CENSUS.json: {"kernel": max tape entries per column, ...}; the capacity is 1.5 x that + 4096."""
import json, os, sys
K = json.load(open(sys.argv[1])); census = json.load(open(sys.argv[2])); out = sys.argv[3]
group = sys.argv[4] if len(sys.argv) > 4 else 'cumulus/linear'
os.makedirs(os.path.join(out, 'specs'), exist_ok=True)
CU = 'tree/ad/phys/cumulus/cuda/nt_ad.cu'
IMAGE = 'nt_ad.fatbin'  # every row's entries are in this one image (cubins/nt_ad.fatbin): derivative.shared_image
def shape(a, W):
    r = a['rows'].strip()
    return f'[{r},{W}]' if r != '1' else f'[{W}]'
rows_out = []
for name, k in K['kernels'].items():
    W = k['count']; F = {f['name']: f for f in k['F']}
    fid = {n: i + 1 for i, n in enumerate(F)}
    cap = int(1.5 * census.get(name, 0)) + 4096
    args, fwd = [], []
    for a in k['args']:
        n = a['name']
        if a['cls'] == 'scalar':
            args.append({'name': n, 'role': 'const', 'field_id': 0, 'dtype': 'i32' if a['type'] == 'int' else 'f32',
                         'shape': '[]', 'active': False, 'inactive_reason': 'launch scalar (extent, constant or time step)',
                         'overwrite': 'none', 'alias_group': None, 'weight': 'none', 'pointer_depth': 0})
            fwd.append({'bind': 'scalar', 'arg': n}); continue
        d = {'name': n, 'dtype': 'f32' if a['type'] == 'float' else 'i32', 'pointer_depth': 1, 'weight': 'none',
             'alias_group': None}
        if a['cls'] == 'F' and 'alias_of' not in a:
            d.update(role='inout' if a['written'] else 'in', field_id=fid[n], shape=shape(a, W), active=True,
                     inactive_reason=None, overwrite='partial' if a['written'] else 'none')
        elif a['cls'] == 'F':
            d.update(role='inout', field_id=fid[a['alias_of']], shape=shape(a, W), active=True, inactive_reason=None,
                     overwrite='partial', alias_group=a['alias_of'],
                     alias_note=f"the same storage as {a['alias_of']} (engine NT_ALIASES); its derivative words are {a['alias_of']}'s")
        elif a['cls'] == 'I':
            d.update(role='in' if a['const'] else 'inout', field_id=0, shape=shape(a, W), active=False, overwrite='none',
                     inactive_reason='integer state (cloud type, levels, labels, flags): frozen at the recorded value; the entries read a copy and never write it')
        elif a['cls'] == 'ro':
            d.update(role='in', field_id=0, shape='[1]', active=False, overwrite='none',
                     inactive_reason='launch-wide integer word (slab latch / report mask): frozen at the recorded value')
        else:
            d.update(role='inout', field_id=0, shape=f"[{a['words']}]", active=False, overwrite='none',
                     inactive_reason='launch report words (geometry, order ticket, ascent start): the entries write a workspace copy, never these')
        if a['cls'] == 'F' and 'alias_of' in a: pass
        args.append(d)
        fwd.append({'bind': 'primal', 'arg': n})
    tl_ws = k['tl_ws']; ad_ws = k['ad_ws'].replace('nt_cap', str(cap))
    fail = lambda e: {'bind': 'failure', 'slots': 4, 'source_ids': [f'{CU}:{name}__{e}:nonfinite', f'{CU}:{name}__{e}:tape-capacity']}
    tl = fwd + [{'bind': 'tangent', 'field_id': fid[n], 'io': 'in', 'arg': n} for n in F] \
             + [{'bind': 'tangent', 'field_id': fid[n], 'io': 'out', 'arg': n} for n, f in F.items() if f['written']] \
             + [{'bind': 'workspace', 'name': 'nt_tl_ws', 'bytes': tl_ws}, fail('tl')]
    ad = fwd + [{'bind': 'adjoint', 'field_id': fid[n], 'io': 'in', 'arg': n} for n, f in F.items() if f['written']] \
             + [{'bind': 'adjoint', 'field_id': fid[n], 'io': 'accumulate', 'arg': n} for n in F] \
             + [{'bind': 'scalar', 'value': cap, 'dtype': '<i4', 'name': 'nt_cap'},
                {'bind': 'workspace', 'name': 'nt_ad_ws', 'bytes': ad_ws}, fail('ad')]
    spec = {
        'kernel': name, 'engine': 'hex', 'file': 'tree/src/hexcore/mpasphys/ntiedtke_v842.py (cuda_source)',
        'route': 'generated: the forward NVRTC text re-typed over AF (tangent) / AR (taped reverse) by tree/ad/phys/cumulus/gen/gen_nt.py',
        'mode': 'exact executed-branch linearisation along the recorded path',
        'forward_source_sha256': K['source_sha256'],
        'why': 'NOAA 12 km hrrrv5 runs cu_ntiedtke every step; no kernel set had a row for its launches, so the window refused any 12 km recording with it',
        'branch_rule': 'every comparison, mpas_nt_int truncation (INTEGER zcbase), fmaxf/fminf arm (first operand on a tie), fabs sign, level index, cloud type (ktype), ldcum, loddraf, klab and the slab latch (mpas_llo3_start) are taken on the pre-launch primal words as the forward takes them; only the executed arm is differentiated',
        'precision': 'partials in double from the f32 primal words; tangents and adjoints accumulate in double, rounded once per written word (+0: no -0.0 words)',
        'primal': 'trajectory = the pre-launch state; the entries never write a primal word (column arrays are shadowed in the workspace)',
        'tape': {'entries_per_column_measured_max': census.get(name, 0), 'capacity': cap, 'entry_bytes': 32},
        'args': args, 'branches': [],
        'launch_abi': {'schema': 'ad.entry/v1', 'entries': {
            'tl': {'function': f'{name}__tl', 'geometry': 'forward', 'args': tl},
            'ad': {'function': f'{name}__ad', 'geometry': 'forward', 'args': ad}}}}
    json.dump(spec, open(os.path.join(out, 'specs', name + '.json'), 'w', newline='\n'), indent=1)
    rowargs = [{'name': a['name'], 'kind': 'device' if a['cls'] != 'scalar' else 'scalar',
                'dtype': ('<f4' if a['type'] == 'float' else '<i4'), 'shape': [], 'strides_bytes': None,
                'pointer_depth': 0 if a['cls'] == 'scalar' else 1, 'value_hex': None} for a in k['args']]
    inactive = not any(f['written'] for f in k['F'])
    rows_out.append({'function': name, 'class': 'inactive-entry' if inactive else 'rowed', 'gate_ready': True,
                     'derivative': {'row': f'cumulus-{name}', 'group': group, 'shared_image': IMAGE,
                                    'spec': {'path': f'specs/{name}.json'},
                                    'cuda': {'path': CU}}, 'arguments': rowargs})
json.dump(rows_out, open(os.path.join(out, 'rows.json'), 'w', newline='\n'), indent=1)
print('specs', len(rows_out))
