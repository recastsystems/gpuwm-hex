#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# compile the captured hex MYNN-EDMF forward source (sha256 70da4af7...) with the P3 harness
# options (strict: fmad off, ftz off, IEEE div and sqrt), for a home card. Build time only.
# Usage: nvrtc_build.py LIBNVRTC SRC ARCH OUT.cubin [extra NVRTC options...]
import ctypes, sys, time, hashlib, json
lib, src, arch, out = sys.argv[1:5]; extra = sys.argv[5:]
n = ctypes.CDLL(lib)
opts = ["--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true",
        "--split-compile=4", "--ptxas-options=--split-compile=4", "--gpu-architecture=" + arch] + extra
prog = ctypes.c_void_p(); code = open(src, "rb").read()
assert n.nvrtcCreateProgram(ctypes.byref(prog), code, b"mynn_fwd.cu", 0, None, None) == 0
arr = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
t0 = time.time(); rc = n.nvrtcCompileProgram(prog, len(opts), arr)
ls = ctypes.c_size_t(); n.nvrtcGetProgramLogSize(prog, ctypes.byref(ls))
log = ctypes.create_string_buffer(ls.value); n.nvrtcGetProgramLog(prog, log)
open(out + ".log", "w").write(log.value.decode(errors="replace"))
if rc != 0: sys.stderr.write(log.value.decode(errors="replace")[-4000:]); sys.exit(1)
cs = ctypes.c_size_t(); assert n.nvrtcGetCUBINSize(prog, ctypes.byref(cs)) == 0
buf = ctypes.create_string_buffer(cs.value); assert n.nvrtcGetCUBIN(prog, buf) == 0
open(out, "wb").write(buf.raw)
r = {"source_sha256": hashlib.sha256(code).hexdigest(), "options": opts, "seconds": round(time.time() - t0, 1),
     "bytes": len(buf.raw), "sha256": hashlib.sha256(buf.raw).hexdigest()}
open(out + ".build.json", "w").write(json.dumps(r, indent=1) + "\n"); print(json.dumps(r))
