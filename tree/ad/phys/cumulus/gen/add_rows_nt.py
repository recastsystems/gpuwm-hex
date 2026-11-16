#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A kernel set = BASE + the New Tiedtke rows (spec, row, and the one image every row names: cubins/nt_ad.fatbin).
usage: add_rows_nt.py [--copy] BASE_KSET OUT_KSET BUILD_DIR SPEC_DIR   (BUILD_DIR: nt_ad.fatbin; SPEC_DIR: specs/, rows.json)
Without --copy every file is a link (a working set beside its base); with --copy every file is a real file, the
form a download holds (its unpacker takes no links)."""
import hashlib, json, os, shutil, sys
argv = sys.argv[1:]
copy = '--copy' in argv
if copy: argv.remove('--copy')
base, out, build, specd = argv[:4]
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
rows = json.load(open(os.path.join(specd, 'rows.json')))
img = os.path.realpath(os.path.join(build, 'nt_ad.fatbin')); img_sha = sha(img)
names = {r['derivative'].get('shared_image') for r in rows}
if len(names) != 1 or None in names: raise SystemExit(f'the rows do not name one image: {sorted(map(str, names))}')
image = names.pop()
put(img, os.path.join(out, 'cubins', image))
added = {}
for r in rows:
    fn = r['function']
    if fn in have: raise SystemExit(f'{fn} already has a row in {base}')
    put(os.path.join(specd, 'specs', fn + '.json'), os.path.join(out, 'specs', fn + '.json'))
    r['derivative']['spec']['sha256'] = sha(os.path.join(out, 'specs', fn + '.json'))
    r['derivative']['fatbin_sha256'] = img_sha
    kj['kernels'].append(r); added[fn] = r['class']
kj['cumulus_ntiedtke'] = {'image': image, 'image_sha256': img_sha, 'rows': added}
json.dump(kj, open(os.path.join(out, 'kernels.json'), 'w', newline='\n'), indent=1)
print('kset', out, 'rows', len(kj['kernels']), 'added', len(added), image, img_sha[:16])
