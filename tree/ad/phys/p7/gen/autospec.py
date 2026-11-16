#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build-time: classify each radiation launch for gen_kernel.py and write a spec.
Active inputs = const float pointer params whose captured leading dimension is the
column count; outputs = writable float pointer params with that leading dimension.
Tables (other shapes) stay inactive. Refuses bodies with shared memory, barriers,
atomics or shuffles, writable non-column float arrays, and kernels where several
threads read one column input (thread-private reads are what the per-thread
adjoint needs for a deterministic, race-free reverse)."""
import json, re, sys, hashlib
from pathlib import Path

cap_root, kernel, step, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
man = json.loads((cap_root / kernel / step / "launch-000000" / "manifest.json").read_text())
src = (cap_root / kernel / step / "launch-000000" / man["source"]["path"]).read_text()
m = re.search(r'extern "C" __global__ void ' + kernel + r"\s*\(", src)
i = m.end() - 1; d = 0; j = i
while True:
    if src[j] == "(": d += 1
    elif src[j] == ")":
        d -= 1
        if d == 0: break
    j += 1
params = [" ".join(p.split()) for p in re.sub(r"//[^\n]*", "", src[i + 1:j]).split(",")]
b0 = src.index("{", j); d = 0; k = b0
while True:
    if src[k] == "{": d += 1
    elif src[k] == "}":
        d -= 1
        if d == 0: break
    k += 1
body = src[b0:k + 1]
why = [t for t in ("__shared__", "__syncthreads", "atomic", "__shfl") if t in body]
args = {a["name"]: a for a in man["arguments"]}
scal = {n: int.from_bytes(bytes.fromhex(a["value_hex"]), "little", signed=True)
        for n, a in args.items() if a["kind"] == "scalar" and a["dtype"] == "<i4"}
geom = man["geometry"]; threads = geom["grid"][0] * geom["grid"][1] * geom["block"][0]
ain, aout, inactive, ncol_name = [], [], [], None
for p in params:
    name = re.findall(r"(\w+)\s*$", p)[0]
    a = args.get(name)
    if a is None or a["kind"] != "device" or a["dtype"] != "<f4": continue
    lead = a["shape"][0] if a["shape"] else None
    col = [n for n, v in scal.items() if v == lead and n in ("ncol", "nc", "ng", "n", "ncols")]
    if col: ncol_name = col[0]
    if "const" in p.split(name)[0]:
        (ain if col else inactive).append(name)
    else:
        if not col: why.append(f"writable non-column array {name}")
        aout.append(name)
ncol = scal.get(ncol_name, 0)
if ncol and threads > 2 * ncol and threads // ncol > 1:
    why.append(f"{threads} threads for {ncol} columns: column inputs read by several threads")
spec = {"kernel": kernel, "source_sha256": man["source"]["sha256"], "active_in": ain, "active_out": aout,
        "inactive_float_inputs": inactive, "columns_arg": ncol_name, "tape": 4096, "writes": 2048,
        "refused": why, "threads": threads, "ncol": ncol}
out.write_text(json.dumps(spec, indent=1))
print(json.dumps({"kernel": kernel, "eligible": not why, "why": why, "in": len(ain), "out": len(aout), "threads": threads, "ncol": ncol}))
