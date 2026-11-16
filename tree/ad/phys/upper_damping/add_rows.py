#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A kernel set = BASE + the two upper-damping rows (spec, image, kernels.json row).
usage: add_rows.py [--copy] BASE_KSET OUT_KSET BUILD_DIR   (BUILD_DIR holds specs/, rows.json and the two .fatbin; build.sh)
Without --copy every file is a link (a working set beside its base); with --copy every file is a real file, the
form a download holds (its unpacker takes no links)."""
import hashlib, json, os, shutil, sys
argv = sys.argv[1:]
copy = '--copy' in argv
if copy: argv.remove('--copy')
base, out, build = argv[:3]
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
def put(src, dst):
    if os.path.lexists(dst): os.remove(dst)
    if copy: shutil.copyfile(os.path.realpath(src), dst)
    else: os.symlink(os.path.realpath(src), dst)
os.makedirs(os.path.join(out, 'cubins'), exist_ok=True); os.makedirs(os.path.join(out, 'specs'), exist_ok=True)
for top in sorted(os.listdir(base)):
    if top in ('cubins', 'specs', 'kernels.json'): continue
    s, d = os.path.join(base, top), os.path.join(out, top)
    if os.path.lexists(d): continue
    if not copy: os.symlink(os.path.realpath(s), d)
    elif os.path.isdir(s): shutil.copytree(os.path.realpath(s), d)
    else: shutil.copyfile(os.path.realpath(s), d)
for sub in ('cubins', 'specs'):
    for f in sorted(os.listdir(os.path.join(base, sub))):
        d = os.path.join(out, sub, f)
        if not os.path.lexists(d): put(os.path.join(base, sub, f), d)
kj = json.load(open(os.path.join(base, 'kernels.json')))
have = {k['function'] for k in kj['kernels']}
rows = json.load(open(os.path.join(build, 'rows.json')))
for r in rows:
    fn = r['function']
    if fn in have: raise SystemExit(f'{fn} already has a row in {base}')
    put(os.path.join(build, 'specs', fn + '.json'), os.path.join(out, 'specs', fn + '.json'))
    put(os.path.join(build, fn + '.fatbin'), os.path.join(out, 'cubins', fn + '.fatbin'))
    r['derivative']['spec']['sha256'] = sha(os.path.join(out, 'specs', fn + '.json'))
    r['derivative']['fatbin_sha256'] = sha(os.path.join(out, 'cubins', fn + '.fatbin'))
    kj['kernels'].append(r)
kj['upper_damping'] = {r['function']: r['derivative']['fatbin_sha256'] for r in rows}
json.dump(kj, open(os.path.join(out, 'kernels.json'), 'w', newline='\n'), indent=1)
print('kset', out, 'rows', len(kj['kernels']), kj['upper_damping'])
