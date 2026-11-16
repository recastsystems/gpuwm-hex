#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# held-rsw_taumol_b: one line per captured rsw_taumol_b launch (ncol, nlayers, geometry, source/image, input hashes).
import json, struct, sys, glob, os
for d in sys.argv[1:]:
    for man in sorted(glob.glob(os.path.join(d, "*", "launch-*", "manifest.json")), key=lambda p: (int(p.split("/")[-3]), p)):
        m = json.load(open(man)); a = {x["name"]: x for x in m["arguments"]}
        i = lambda n: struct.unpack("<i", bytes.fromhex(a[n]["value_hex"]))[0]
        print(json.dumps({"capture": man[:-14], "step": m["step"], "launch": m["launch_index"], "ncol": i("ncol"), "nlayers": i("nlayers"),
                          "tab": a["tab"]["shape"], "grid": m["geometry"]["grid"], "block": m["geometry"]["block"],
                          "source": m["source"]["sha256"][:8], "image": m["image"]["sha256"][:8],
                          "in": {n: a[n]["sha256"][:8] for n in ("laytrop", "jp", "colh2o", "fac00")}}))
