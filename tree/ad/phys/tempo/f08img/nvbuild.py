#!/usr/bin/env python3
"""tempo-image: NVRTC build of a split2 source with the release's split/build.py options (no card needed).
nvbuild.py LIBNVRTC SOURCE OUTPUT"""
import ctypes as c, hashlib, json, sys, time
from pathlib import Path
libp, source, output = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
lib = c.CDLL(libp)
major, minor = c.c_int(), c.c_int(); lib.nvrtcVersion(c.byref(major), c.byref(minor))
prog = c.c_void_p(); data = source.read_bytes()
name = sys.argv[4] if len(sys.argv) > 4 else str(source)
assert lib.nvrtcCreateProgram(c.byref(prog), data, name.encode(), 0, None, None) == 0
opts = ['--std=c++17', '--fmad=false', '--ftz=false', '--prec-div=true', '--prec-sqrt=true',
        '-arch=sm_120', '--device-as-default-execution-space', '--maxrregcount=128', '--ptxas-options=-v']
arr = (c.c_char_p * len(opts))(*(x.encode() for x in opts))
t = time.monotonic(); rc = lib.nvrtcCompileProgram(prog, len(opts), arr)
n = c.c_size_t(); lib.nvrtcGetProgramLogSize(prog, c.byref(n))
log = c.create_string_buffer(n.value); lib.nvrtcGetProgramLog(prog, log)
output.with_suffix('.build.log').write_bytes(log.value)
r = {'nvrtc': [major.value, minor.value], 'lib': libp, 'options': opts, 'rc': rc, 'seconds': round(time.monotonic() - t, 1),
     'source': str(source), 'source_sha256': hashlib.sha256(data).hexdigest(), 'program_name': name}
if rc == 0:
    assert lib.nvrtcGetCUBINSize(prog, c.byref(n)) == 0
    cub = c.create_string_buffer(n.value); assert lib.nvrtcGetCUBIN(prog, cub) == 0
    output.write_bytes(cub.raw); r.update(cubin_sha256=hashlib.sha256(cub.raw).hexdigest(), bytes=n.value)
output.with_suffix('.build.json').write_text(json.dumps(r, indent=2) + '\n')
print(json.dumps(r), flush=True)
sys.exit(rc)
