"""Build a card image of the joined MYNN backward with NVRTC (no CUDA toolkit needed; no card touched).
usage: NVRTC=/path/libnvrtc.so.13 LD_LIBRARY_PATH=/path [JOIN_BLOCKS=dir] python3 nvrtc_build.py SRC.cu OUT_STEM ARCH [extra -D options...]
Writes OUT_STEM.cubin, OUT_STEM.log (ptxas -v resource lines) and OUT_STEM.json (receipt)."""
import ctypes as ct, hashlib, json, os, re, sys, time
from pathlib import Path

H = Path(__file__).resolve().parent


def build(library, name, src, opts):
    nv = ct.CDLL(str(Path(library).resolve()))
    prog = ct.c_void_p()
    nv.nvrtcCreateProgram.argtypes = [ct.POINTER(ct.c_void_p), ct.c_char_p, ct.c_char_p, ct.c_int, ct.c_void_p, ct.c_void_p]
    nv.nvrtcCompileProgram.argtypes = [ct.c_void_p, ct.c_int, ct.POINTER(ct.c_char_p)]
    nv.nvrtcDestroyProgram.argtypes = [ct.POINTER(ct.c_void_p)]
    for f in ("nvrtcGetProgramLogSize", "nvrtcGetCUBINSize"):
        getattr(nv, f).argtypes = [ct.c_void_p, ct.POINTER(ct.c_size_t)]
    for f in ("nvrtcGetProgramLog", "nvrtcGetCUBIN"):
        getattr(nv, f).argtypes = [ct.c_void_p, ct.c_void_p]
    if nv.nvrtcCreateProgram(ct.byref(prog), src.encode(), name.encode(), 0, None, None):
        raise RuntimeError("nvrtc create")
    try:
        o = (ct.c_char_p * len(opts))(*(x.encode() for x in opts))
        rc = nv.nvrtcCompileProgram(prog, len(opts), o)
        n = ct.c_size_t(); nv.nvrtcGetProgramLogSize(prog, ct.byref(n))
        buf = ct.create_string_buffer(max(n.value, 1)); nv.nvrtcGetProgramLog(prog, buf)
        log = buf.value.decode(errors="replace")
        if rc:
            return rc, b"", log
        nv.nvrtcGetCUBINSize(prog, ct.byref(n)); cub = ct.create_string_buffer(n.value); nv.nvrtcGetCUBIN(prog, cub)
        return 0, cub.raw, log
    finally:
        nv.nvrtcDestroyProgram(ct.byref(prog))


def main():
    src_path, stem, arch = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    blocks = Path(os.environ.get("JOIN_BLOCKS", H.parent / "blocks"))   # the arena form: build/arena_blocks (arena_rewrite.py)
    opts = ["--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true", f"-arch={arch}",
            "--device-as-default-execution-space", "--ptxas-options=-v", "-I" + str(H / "nvrtc_std"), "-I" + str(blocks),
            "-I" + str(src_path.parent)] + sys.argv[4:]
    t = time.monotonic()
    rc, cub, log = build(os.environ["NVRTC"], src_path.name, src_path.read_text(), opts)
    stem.with_suffix(".log").write_text(log)
    res = re.findall(r"ptxas[^\n]*", log)
    rec = dict(source=str(src_path), source_sha256=hashlib.sha256(src_path.read_bytes()).hexdigest(), options=opts, rc=rc,
               seconds=round(time.monotonic() - t, 1), cubin_bytes=len(cub),
               cubin_sha256=hashlib.sha256(cub).hexdigest() if cub else None, resources=res)
    if not rc:
        stem.with_suffix(".cubin").write_bytes(cub)
    stem.with_suffix(".json").write_text(json.dumps(rec, indent=1) + "\n")
    print(("PASS" if rc == 0 else "FAIL") + f" nvrtc {arch} {src_path.name}: rc {rc}, {rec['seconds']} s, cubin {len(cub)} B")
    for line in res:
        print("  " + line)
    if rc:
        print(log[-6000:])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
