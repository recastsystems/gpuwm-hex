#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Test tooling (never in a step loop): sort the columns of one captured longwave
call into weather regimes, from that call's own rlw_setcoef inputs, so the P7
dot and FD checks can be graded per regime. Writes {"classes": {name: [col,...]}}.

Regimes (layer 0 is the bottom layer; checked):
  moist / dry            precipitable water >= 40 mm / <= 5 mm
  near_freezing          lowest-layer temperature within 2 K of 273.15 K
  saturated / unsaturated  max RH below 300 hPa >= 0.95 / < 0.60 (ice below 273.15 K)
  bl_stable / bl_unstable  skin below lowest-layer air by > 1 K (night-like) /
                           skin above it by > 2 K (day-like)
Usage: colclass.py CAPTURE_ROOT STEP LAUNCH OUT.json
"""
import json, sys
from pathlib import Path
import numpy as np

root, step, launch, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
d = root / "rlw_setcoef" / step / f"launch-{int(launch):06d}"
m = json.loads((d / "manifest.json").read_text())

def alloc_bytes(i):
    o = m["allocations"][i]["pre"]
    if o.get("encoding") == "raw-concat-v1":
        return b"".join((d / c["path"]).read_bytes() for c in o["chunks"])
    return (d / o["path"]).read_bytes()

args = {a["name"]: a for a in m["arguments"]}
cache = {}
def arr(n):
    a = args[n]
    if a["allocation"] not in cache: cache[a["allocation"]] = alloc_bytes(a["allocation"])
    cnt = int(np.prod(a["shape"]))
    return np.frombuffer(cache[a["allocation"]], dtype="<f4", count=cnt, offset=a.get("offset", 0)).reshape(a["shape"]).astype(np.float64)

pavel, tavel, coldry, wkl, tb = arr("pavel"), arr("tavel"), arr("coldry"), arr("wkl"), arr("tbound_v")
ncol, nl = pavel.shape
assert np.all(pavel[:, 0] > pavel[:, -1]), "layer 0 is not the bottom"
h2o = wkl[:, 0, :]
pw_mm = (h2o * 18.015 / 6.02214e23).sum(axis=1) * 10.0
vmr = h2o / coldry
e = vmr * pavel / (1.0 + vmr)
tc = tavel - 273.15
es = np.where(tavel >= 273.15, 6.112 * np.exp(17.67 * tc / (tavel - 29.65)), 6.112 * np.exp(22.46 * tc / (tavel - 0.53)))
rh = np.where(pavel >= 300.0, e / es, 0.0)
rhmax = rh.max(axis=1)
dts = tb - tavel[:, 0]
cls = {
    "all": np.arange(ncol),
    "moist": np.where(pw_mm >= 40)[0], "dry": np.where(pw_mm <= 5)[0],
    "near_freezing": np.where(np.abs(tavel[:, 0] - 273.15) <= 2)[0],
    "saturated": np.where(rhmax >= 0.95)[0], "unsaturated": np.where(rhmax < 0.60)[0],
    "bl_stable": np.where(dts < -1)[0], "bl_unstable": np.where(dts > 2)[0],
}
summary = {k: len(v) for k, v in cls.items()}
stats = {"pw_mm": [float(pw_mm.min()), float(np.median(pw_mm)), float(pw_mm.max())],
         "t_lowest_K": [float(tavel[:, 0].min()), float(tavel[:, 0].max())],
         "rhmax": [float(rhmax.min()), float(rhmax.max())], "skin_minus_air_K": [float(dts.min()), float(dts.max())]}
out.write_text(json.dumps({"step": step, "launch": int(launch), "ncol": ncol, "nlayers": nl, "counts": summary,
                           "stats": stats, "classes": {k: [int(x) for x in v] for k, v in cls.items()}}))
print(json.dumps({"step": step, "launch": launch, "ncol": ncol, "nl": nl, "counts": summary, "stats": stats}))
