#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# held-rsw_taumol_b (autodiff, private): the tangent against the REAL forward on one captured launch. numpy + the driver
# API via ctypes, no other dependency.
#   F = the captured image (rsw_taumol_b, captured source 3c499c7b) on the captured pre-launch allocations.
#   dp = u |p| on the 14 active inputs (u iid uniform(-1, 1), seed 7); p + h dp is rounded to float32 and the REALIZED
#   step (fl(p + h dp) - p) / h is what the tangent gets, so input rounding is not counted as tangent error.
#   FD(h) = (F(p + h dp) - F(p)) / h in float64 from the float32 outputs; TL = rsw_taumol_b__tl of the bound image.
#   Timed per h: one tangent launch and one forward launch (launch + synchronize, host clock).
#   Reported per h and per output: ||FD - TL|| / ||TL||, and the share of words whose |FD - TL| > 1e-2 |TL| + 1e-30
#   (bin-crossing words: the eta bin js or the clamp switched between p and p + h dp).
# Usage: CUDA_VISIBLE_DEVICES=<card> python3 -I rsw_taumol_fd.py <launch dir> <bound cubin> <out.json>
import ctypes as C, hashlib, json, os, sys, time
import numpy as np

cap, cubin, out = sys.argv[1:4]
ACTIVE = ["colh2o", "colco2", "colch4", "colo2", "colo3", "colmol", "fac00", "fac01", "fac10", "fac11",
          "selffac", "selffrac", "forfac", "forfrac"]
cu = C.CDLL("libcuda.so.1")
cu.cuMemAlloc_v2.argtypes = [C.POINTER(C.c_uint64), C.c_size_t]
cu.cuMemcpyHtoD_v2.argtypes = [C.c_uint64, C.c_void_p, C.c_size_t]
cu.cuMemcpyDtoH_v2.argtypes = [C.c_void_p, C.c_uint64, C.c_size_t]
cu.cuLaunchKernel.argtypes = [C.c_void_p] + [C.c_uint] * 7 + [C.c_void_p] * 3


def ck(r, what):
    if r != 0: raise SystemExit(f"{what}: CUDA error {r}")


ck(cu.cuInit(0), "init"); dev = C.c_int(); ck(cu.cuDeviceGet(C.byref(dev), 0), "dev")
ctx = C.c_void_p(); ck(cu.cuDevicePrimaryCtxRetain(C.byref(ctx), dev), "ctx"); ck(cu.cuCtxSetCurrent(ctx), "set")
m = json.load(open(os.path.join(cap, "manifest.json")))


def obj(o):
    if "chunks" in o:
        b = b"".join(open(os.path.join(cap, c["path"]), "rb").read() for c in o["chunks"])
    else:
        b = open(os.path.join(cap, o["path"]), "rb").read()
    if hashlib.sha256(b).hexdigest() != o["sha256"]: raise SystemExit("object sha256 differs")
    return b


def func(path, name):
    mod = C.c_void_p(); data = open(path, "rb").read() if isinstance(path, str) else path
    ck(cu.cuModuleLoadData(C.byref(mod), C.c_char_p(data)), "module")
    f = C.c_void_p(); ck(cu.cuModuleGetFunction(C.byref(f), mod, name.encode()), name); return f


def alloc(b):
    p = C.c_uint64(); ck(cu.cuMemAlloc_v2(C.byref(p), max(len(b), 8)), "alloc")
    ck(cu.cuMemcpyHtoD_v2(p.value, b, len(b)), "h2d"); return p.value


def down(p, n):
    b = C.create_string_buffer(n); ck(cu.cuMemcpyDtoH_v2(b, p, n), "d2h"); return b.raw


pre = [obj(a["pre"]) for a in m["allocations"]]
args = {a["name"]: a for a in m["arguments"]}
order = [a["name"] for a in m["arguments"]]
import struct
ncol = struct.unpack("<i", bytes.fromhex(args["ncol"]["value_hex"]))[0]
nl = struct.unpack("<i", bytes.fromhex(args["nlayers"]["value_hex"]))[0]
n2, nout = ncol * nl, 112 * ncol * nl


def view(name, dtype):
    a = args[name]; b = pre[a["allocation"]]; o = a["offset"]; n = n2 if name in ACTIVE else None
    return np.frombuffer(b, dtype=dtype, count=n, offset=o)


base = {n: view(n, np.float32).astype(np.float32) for n in ACTIVE}
g, blk = m["geometry"]["grid"], m["geometry"]["block"]
fwd = func(obj(m["image"]), "rsw_taumol_b")
tl = func(cubin, "rsw_taumol_b__tl")


def launch(f, vals, grid, block):
    holders = []
    for v in vals:
        holders.append(C.c_int(v[0]) if isinstance(v, tuple) else C.c_uint64(v))
    arr = (C.c_void_p * len(holders))(*[C.cast(C.byref(h), C.c_void_p) for h in holders])
    ck(cu.cuLaunchKernel(f, grid[0], grid[1], grid[2], block[0], block[1], block[2], 0, None, arr, None), "launch")
    ck(cu.cuCtxSynchronize(), "sync")


def run_forward(inputs):
    ptr = {}
    vals = []
    for nme in order:
        a = args[nme]
        if a["kind"] == "scalar":
            vals.append((struct.unpack("<i", bytes.fromhex(a["value_hex"]))[0],)); continue
        b = bytearray(pre[a["allocation"]])
        if nme in inputs:
            o = a["offset"]; b[o:o + 4 * n2] = inputs[nme].astype(np.float32).tobytes()
        ptr[nme] = alloc(bytes(b)); vals.append(ptr[nme] + 0)
    return ptr, vals


def outputs(ptr):
    return {k: np.frombuffer(down(ptr[k] + args[k]["offset"], 4 * nout), dtype=np.float32).astype(np.float64) for k in ("taug", "taur")}


p0, v0 = run_forward({})
launch(fwd, v0, g, blk); F0 = outputs(p0)
rng = np.random.default_rng(7)
u = {n: rng.uniform(-1, 1, n2) for n in ACTIVE}
res = {"capture": cap, "ncol": ncol, "nlayers": nl, "bound_cubin_sha256": hashlib.sha256(open(cubin, "rb").read()).hexdigest(), "h": []}
for h in (1e-2, 1e-3, 1e-4, 1e-5):
    xp = {n: (base[n].astype(np.float64) + h * u[n] * np.abs(base[n].astype(np.float64))).astype(np.float32) for n in ACTIVE}
    dx = {n: (xp[n].astype(np.float64) - base[n].astype(np.float64)) / h for n in ACTIVE}
    p1, v1 = run_forward(xp); launch(fwd, v1, g, blk); F1 = outputs(p1)
    # tangent: forward args on fresh copies of the pre state, then d-inputs, then dtaug/dtaur
    pt, vt = run_forward({})
    dptr = [alloc(dx[n].astype(np.float64).tobytes()) for n in ACTIVE]
    dout = [alloc(bytes(8 * nout)) for _ in range(2)]
    t0 = time.perf_counter(); launch(tl, vt + dptr + dout, g, blk); tl_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter(); launch(fwd, v1, g, blk); fwd_ms = (time.perf_counter() - t0) * 1e3
    T = {k: np.frombuffer(down(dout[i], 8 * nout), dtype=np.float64) for i, k in enumerate(("taug", "taur"))}
    aug = outputs(pt)
    row = {"h": h, "augmented_primal_differing_words": int(sum((aug[k] != F0[k]).sum() for k in aug)),
           "tl_launch_ms": round(tl_ms, 3), "forward_launch_ms": round(fwd_ms, 3)}  # launch + sync, host clock
    for k in ("taug", "taur"):
        fd = (F1[k] - F0[k]) / h; t = T[k]; e = fd - t
        far = np.abs(e) > 1e-2 * np.abs(t) + 1e-30
        mask = ~far
        row[k] = {"rel_err": float(np.linalg.norm(e) / max(np.linalg.norm(t), 1e-300)),
                  "rel_err_without_crossings": float(np.linalg.norm(e[mask]) / max(np.linalg.norm(t[mask]), 1e-300)),
                  "far_words": int(far.sum()), "words": int(t.size), "tl_norm": float(np.linalg.norm(t)),
                  "nonfinite_tl": int((~np.isfinite(t)).sum())}
    res["h"].append(row)
    print(json.dumps(row))
json.dump(res, open(out, "w"), indent=1)
