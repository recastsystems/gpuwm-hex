#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# held-rsw_taumol_b: switch / range hazard census of the copied SW band body (rsw_spec .. rsw_taumol_b) with line numbers
# in the captured source 3c499c7b.  p7_rsw_taumol_hazards.py SOURCE.cu OUT.json
import json, re, sys
src = open(sys.argv[1]).read().split("\n")
lo = next(i for i, l in enumerate(src) if l.startswith("struct RswSpec"))
hi = next(i for i, l in enumerate(src) if "void rsw_taumol_b(int ncol" in l) + 30
PAT = [
    ("division (derivative 1/b, -a/b^2: b = speccomb = colA + strrat*colB -> 0 gives inf/NaN)", r"\bDV\("),
    ("limiter-like clamp specparm >= RSW_C_ONEMINUS (frozen: zero derivative when it fires)", r"RSW_C_ONEMINUS"),
    ("eta bin js = 1 + (int)specmult (trajectory bin; d fs = d specmult)", r"\(int\)specmult|truncf\("),
    ("lower/upper switch lay <= laytrop (integer, trajectory)", r"lay <= laytrop|if \(lower\)"),
    ("table row index from jp/jt/jt1/js (integer; out of range reads another table's words)", r"int ind[01] ="),
    ("self/foreign continuum index inds/indf (integer; reads sc[inds], fc[indf])", r"sc\[inds|fc\[indf"),
    ("Rayleigh eta interpolation ra[ig + 8*(js-1)], ra[ig + 8*js] (band 24, js in 1..8 needed)", r"ra\[ig"),
    ("band from ngb[iw]; any value other than 16..28 takes the default (band 29) arm", r"int jb = ngb|default: \{"),
    ("g-point offset ig = iw - k (assumes ngb is the RRTMG table)", r"int ig = iw -"),
    ("output store (one writer per word)", r"taug\[out\]|taur\[out\]"),
]
out = []
for what, pat in PAT:
    lines = [i + 1 for i in range(lo, min(hi, len(src))) if re.search(pat, src[i])]
    out.append({"hazard": what, "pattern": pat, "lines": lines, "count": len(lines)})
json.dump({"source_sha256": "3c499c7ba10221067f80e300b1003e382a66245027ae997f254005f80b74e700", "range": [lo + 1, hi], "hazards": out}, open(sys.argv[2], "w"), indent=1)
for h in out: print(h["count"], h["hazard"][:70], h["lines"][:40])
