#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""L2 (ad/window-L2-bindings): copy the captured longwave launches the P7 rows are proved on (manifest plus only
the objects it names, dereferenced) into OUT, so the captured-launch harness can run on another node.
usage: p7_pack.py CAPTURE_ROOT OUT kernel/step/launch ..."""
import json, os, shutil, sys
root, out = sys.argv[1], sys.argv[2]
n = 0
for rel in sys.argv[3:]:
    src, dst = os.path.join(root, rel), os.path.join(out, rel)
    os.makedirs(os.path.join(dst, "objects"), exist_ok=True)
    m = json.load(open(os.path.join(src, "manifest.json")))
    shutil.copy(os.path.join(src, "manifest.json"), dst)
    paths = set()
    def walk(v):
        if isinstance(v, dict):
            if isinstance(v.get("path"), str) and v["path"].startswith("objects/"): paths.add(v["path"])
            for c in v.get("chunks", []) or []:
                if isinstance(c, dict) and str(c.get("path", "")).startswith("objects/"): paths.add(c["path"])
            for x in v.values(): walk(x)
        elif isinstance(v, list):
            for x in v: walk(x)
    walk(m)
    for p in paths:
        d = os.path.join(dst, p)
        if not os.path.exists(d): shutil.copyfile(os.path.realpath(os.path.join(src, p)), d); n += 1
print("objects", n)
