#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rebuild a shipped kernel image for other card families from the PTX the image itself carries.

For a kernel whose source is lost (its record names an edited file that was never kept, or a commit whose file no longer
gives the shipped bytes), the image is still rebuildable when it carries its own PTX: that PTX is what the shipped
machine code was assembled from. This tool
  1. reads the image's PTX entry (zstd-compressed in a CUDA 13 fatbin) and the ptxas options the image records;
  2. CONTROL: assembles that PTX for the image's own architecture with the named ptxas and refuses unless the result
     equals the shipped machine code byte for byte (so the PTX, the assembler and the options are the shipped ones);
  3. assembles the same PTX, with only its .target line lowered, for each architecture asked for;
  4. packs portable code (the lowest target) plus every machine code, the shipped one unchanged, into one image.
THE BREAKAGE THIS PREVENTS (F11, 15 Nov 2026): on an RTX 4090 the 12 km example's tangent and adjoint stopped at
CUDA_ERROR_NO_BINARY_FOR_GPU for smagorinsky_v841_f32, vector_momentum_v841_f32 and w_filter_lap2_v841_f32: each was
shipped with Blackwell code only, and no source in any tree gave the shipped bytes, so no Ada code could be built.
Each carries compute_120 PTX; through ptxas 13.5.37 with the recorded "--fmad false" it gives the shipped sm_120 code
byte for byte, and sm_80, sm_86, sm_89 and sm_90 code from the same PTX.

Usage: retarget_from_ptx.py IMAGE.fatbin --ptxas PTXAS --fatbinary FATBINARY --arch 80,86,89,90 --out OUT.fatbin
Needs the zstd command (the image's PTX is zstd frames).
"""
import argparse
import os
import struct
import subprocess
import sys
import tempfile

FATBIN_MAGIC = 0xBA55ED50


def entries(blob):
    out, off = [], 0
    while off + 16 <= len(blob):
        magic, _ver, header, size = struct.unpack_from("<IHHQ", blob, off)
        if magic != FATBIN_MAGIC:
            break
        p, end = off + header, off + header + size
        while p + 64 <= end:
            kind, _u, eh, psize = struct.unpack_from("<HHIQ", blob, p)
            arch = struct.unpack_from("<I", blob, p + 28)[0]
            opt_off, opt_len = struct.unpack_from("<II", blob, p + 64) if eh >= 72 else (0, 0)
            options = blob[p + opt_off:p + opt_off + opt_len].decode(errors="replace") if opt_len else ""
            raw_size = struct.unpack_from("<Q", blob, p + 56)[0] if eh >= 64 else 0
            packed = struct.unpack_from("<I", blob, p + 16)[0]
            out.append({"kind": {1: "ptx", 2: "elf"}.get(kind, kind), "arch": arch, "payload": blob[p + eh:p + eh + psize],
                        "options": options.strip(), "raw_size": raw_size, "packed": packed})
            p += eh + psize
        off = (end + 7) & ~7
    return out


def ptx_text(entry):
    body = entry["payload"]
    if body[:4] == b"\x28\xb5\x2f\xfd":   # zstd frame
        body = subprocess.run(["zstd", "-d", "-q", "-c"], input=body, capture_output=True).stdout
    if entry["raw_size"]:
        body = body[:entry["raw_size"]]
    return body.rstrip(b"\x00").decode()


def ptxas_options(recorded):
    """The image records ptxas options as words ("--fmad false"); give them back as ptxas flags."""
    words, flags = recorded.split(), []
    for i in range(0, len(words) - 1, 2):
        flags.append(f"{words[i]}={words[i + 1]}")
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--ptxas", required=True)
    ap.add_argument("--fatbinary", required=True)
    ap.add_argument("--arch", default="80,86,89,90")
    ap.add_argument("--out", required=True)
    o = ap.parse_args()
    ents = entries(open(o.image, "rb").read())
    ptx = [e for e in ents if e["kind"] == "ptx"]
    elf = [e for e in ents if e["kind"] == "elf"]
    if len(ptx) != 1 or len(elf) != 1 or ptx[0]["arch"] != elf[0]["arch"]:
        sys.exit(f"refusing: {o.image} is not one PTX and one machine code of the same architecture")
    own, text, flags = elf[0]["arch"], ptx_text(ptx[0]), ptxas_options(ptx[0]["options"])
    if f".target sm_{own}\n" not in text:
        sys.exit(f"refusing: the PTX does not target sm_{own}")
    with tempfile.TemporaryDirectory() as d:
        def assemble(arch):
            src, dst = os.path.join(d, f"k{arch}.ptx"), os.path.join(d, f"k{arch}.cubin")
            open(src, "w").write(text.replace(f".target sm_{own}\n", f".target sm_{arch}\n"))
            subprocess.run([o.ptxas, f"-arch=sm_{arch}", "-m64", *flags, src, "-o", dst], check=True)
            return src, dst
        _, control = assemble(own)
        if open(control, "rb").read() != elf[0]["payload"]:
            sys.exit(f"refusing: sm_{own} code from the image's own PTX ({o.ptxas} {' '.join(flags)}) is not the shipped "
                     "code byte for byte")
        print(f"control: sm_{own} from the image's PTX equals the shipped code ({len(elf[0]['payload'])} B)")
        arches = sorted(int(a) for a in o.arch.split(","))
        built = {a: assemble(a) for a in arches}
        images = [f"--image3=kind=ptx,sm={arches[0]},file={built[arches[0]][0]}"]
        images += [f"--image3=kind=elf,sm={a},file={built[a][1]}" for a in arches]
        images += [f"--image3=kind=elf,sm={own},file={control}"]
        subprocess.run([o.fatbinary, "--64", f"--create={o.out}", *images], check=True)
    print(f"{o.out}: portable compute_{arches[0]} code, sm_{', sm_'.join(map(str, arches))} and the shipped sm_{own} code")


if __name__ == "__main__":
    main()
