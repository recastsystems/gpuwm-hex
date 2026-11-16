#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# held-rsw_taumol_b: list the files one captured launch references (manifest, image, source, pre/post objects and chunks),
# relative to ROOT, for tar -h --files-from.  refs.py ROOT step/launch-NNNNNN ...
import json, os, sys
root = sys.argv[1]
seen = set()
def add(p):
    if p not in seen: seen.add(p); print(p)
for rel in sys.argv[2:]:
    m = json.load(open(os.path.join(root, rel, "manifest.json")))
    add(os.path.join(rel, "manifest.json"))
    for k in ("image", "source"): add(os.path.join(rel, m[k]["path"]))
    for a in m["allocations"]:
        for side in ("pre", "post"):
            o = a[side]
            for c in (o.get("chunks") or [o]): add(os.path.join(rel, c["path"]))
