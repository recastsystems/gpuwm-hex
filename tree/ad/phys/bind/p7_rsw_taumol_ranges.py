#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# held-rsw_taumol_b: range census of the hazards on captured launches (host numpy, no GPU). For each launch: integer index
# ranges (laytrop, jp, jt, jt1, indself, indfor, ngb), minimum column amounts, the eta-binning specparm per rsw_spec site
# (float64 re-evaluation of the float32 expression: how close it gets to the RSW_C_ONEMINUS clamp, and the js range), and
# the output ranges (taug, taur max, non-finite words).  rsw_taumol_ranges.py SOURCE.cu OUT.json LAUNCH_DIR...
import hashlib, json, os, re, struct, sys
import numpy as np

src = open(sys.argv[1]).read()
C = {m.group(1): struct.unpack("<f", struct.pack("<I", int(m.group(2))))[0]
     for m in re.finditer(r"#define (RSW_C_\w+) __uint_as_float\((\d+)u\)", src)}
SITES = [  # band, colA, colB, strrat, mult, lower?  (rsw_spec calls in rsw_taumol_body)
    (16, "colh2o", "colch4", C["RSW_C_KG16_STRRAT1"], 8, True), (17, "colh2o", "colco2", C["RSW_C_KG17_STRRAT"], 8, True),
    (17, "colh2o", "colco2", C["RSW_C_KG17_STRRAT"], 4, False), (18, "colh2o", "colch4", C["RSW_C_KG18_STRRAT"], 8, True),
    (19, "colh2o", "colco2", C["RSW_C_KG19_STRRAT"], 8, True), (21, "colh2o", "colco2", C["RSW_C_KG21_STRRAT"], 8, True),
    (21, "colh2o", "colco2", C["RSW_C_KG21_STRRAT"], 4, False), (22, "colh2o", "colo2", 1.6 * C["RSW_C_KG22_STRRAT"], 8, True),
    (24, "colh2o", "colo2", C["RSW_C_KG24_STRRAT"], 8, True), (28, "colo3", "colo2", C["RSW_C_KG28_STRRAT"], 8, True),
    (28, "colo3", "colo2", C["RSW_C_KG28_STRRAT"], 4, False)]
oneminus = C["RSW_C_ONEMINUS"]
res = {"oneminus": oneminus, "launches": []}
for cap in sys.argv[3:]:
    m = json.load(open(os.path.join(cap, "manifest.json")))
    a = {x["name"]: x for x in m["arguments"]}
    sc = lambda n: struct.unpack("<i", bytes.fromhex(a[n]["value_hex"]))[0]
    ncol, nl = sc("ncol"), sc("nlayers")

    def arr(n, side="pre", dt=None, count=None):
        al = m["allocations"][a[n]["allocation"]][side]
        b = b"".join(open(os.path.join(cap, c["path"]), "rb").read() for c in al["chunks"]) if "chunks" in al else open(os.path.join(cap, al["path"]), "rb").read()
        dt = dt or (np.int32 if a[n]["dtype"] == "<i4" else np.float32)
        return np.frombuffer(b, dtype=dt, count=count if count is not None else int(np.prod(a[n]["shape"])), offset=a[n]["offset"])
    r = {"capture": cap, "ncol": ncol, "nlayers": nl}
    lt = arr("laytrop", count=ncol)
    r["int"] = {n: [int(v.min()), int(v.max())] for n, v in [("laytrop", lt)] + [(k, arr(k, count=ncol * nl)) for k in ("jp", "jt", "jt1", "indself", "indfor")] + [("ngb", arr("ngb", count=112))]}
    f = {k: arr(k, count=ncol * nl).astype(np.float64).reshape(ncol, nl) for k in ("colh2o", "colco2", "colch4", "colo2", "colo3", "colmol")}
    r["min_amount"] = {k: float(v.min()) for k, v in f.items()}
    lower = (np.arange(1, nl + 1)[None, :] <= lt[:, None])
    # indices as the body reads them: indself only in lower arms (rsw_selffor / inline), indfor in both arms
    ins, inf = arr("indself", count=ncol * nl).reshape(ncol, nl), arr("indfor", count=ncol * nl).reshape(ncol, nl)
    r["int_read"] = {"indself_lower": [int(ins[lower].min()), int(ins[lower].max())], "indfor_lower": [int(inf[lower].min()), int(inf[lower].max())],
                     "indfor_upper": [int(inf[~lower].min()), int(inf[~lower].max())], "indself_upper": [int(ins[~lower].min()), int(ins[~lower].max())]}
    sites = []
    for band, A, B, s, mult, low in SITES:
        w = lower if low else ~lower
        if not w.any(): continue
        comb = f[A] + s * f[B]; sp = f[A][w] / comb[w]
        js = 1 + np.floor(mult * np.minimum(sp, oneminus)).astype(int)
        sites.append({"band": band, "arm": "lower" if low else "upper", "mult": mult, "words": int(w.sum()),
                      "speccomb_min": float(comb[w].min()), "specparm_max": float(sp.max()), "specparm_min": float(sp.min()),
                      "clamp_fires": int((sp >= oneminus).sum()), "js": [int(js.min()), int(js.max())]})
    r["spec_sites"] = sites
    tg, tr = arr("taug", "post", count=112 * ncol * nl), arr("taur", "post", count=112 * ncol * nl)
    r["out"] = {"taug_max": float(tg.max()), "taug_min": float(tg.min()), "taur_max": float(tr.max()),
                "nonfinite": int((~np.isfinite(tg)).sum() + (~np.isfinite(tr)).sum())}
    res["launches"].append(r)
    print(json.dumps({k: r[k] for k in ("ncol", "int", "int_read", "min_amount", "out")}))
    for s_ in sites: print("  ", json.dumps(s_))
json.dump(res, open(sys.argv[2], "w"), indent=1)
