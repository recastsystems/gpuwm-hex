#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build time only: compile one CUDA source to a cubin with NVRTC (ctypes, no other dependency).
  nvrtc_cubin.py LIBNVRTC SOURCE.cu OUT.cubin OPTION...
Writes OUT.cubin and OUT.cubin.json (options, sha256 of source and image, compile seconds, NVRTC version, log)."""
import ctypes as C, hashlib, json, sys, time

lib, src, out, *opts = sys.argv[1:]
nv = C.CDLL(lib)
text = open(src, "rb").read()
prog = C.c_void_p()


def ck(r, what):
    if r != 0:
        nv.nvrtcGetErrorString.restype = C.c_char_p
        raise SystemExit(f"{what}: {nv.nvrtcGetErrorString(r).decode()}")


ck(nv.nvrtcCreateProgram(C.byref(prog), text, src.encode(), 0, None, None), "nvrtcCreateProgram")
copts = (C.c_char_p * len(opts))(*[o.encode() for o in opts])
t0 = time.time()
r = nv.nvrtcCompileProgram(prog, len(opts), copts)
dt = time.time() - t0
n = C.c_size_t()
nv.nvrtcGetProgramLogSize(prog, C.byref(n))
log = C.create_string_buffer(n.value)
nv.nvrtcGetProgramLog(prog, log)
if r != 0:
    sys.stderr.write(log.value.decode(errors="replace")[-20000:])
    raise SystemExit(f"nvrtcCompileProgram failed ({r})")
ck(nv.nvrtcGetCUBINSize(prog, C.byref(n)), "nvrtcGetCUBINSize")
buf = C.create_string_buffer(n.value)
ck(nv.nvrtcGetCUBIN(prog, buf), "nvrtcGetCUBIN")
img = buf.raw[:n.value]
open(out, "wb").write(img)
maj, mnr = C.c_int(), C.c_int()
nv.nvrtcVersion(C.byref(maj), C.byref(mnr))
rec = {"source": src, "source_sha256": hashlib.sha256(text).hexdigest(), "image": out,
       "image_sha256": hashlib.sha256(img).hexdigest(), "image_bytes": len(img), "options": opts,
       "nvrtc": f"{maj.value}.{mnr.value}", "compile_s": round(dt, 2), "log": log.value.decode(errors="replace")[-4000:]}
json.dump(rec, open(out + ".json", "w"), indent=1)
print(json.dumps({k: rec[k] for k in ("image_sha256", "image_bytes", "compile_s", "nvrtc")}))
