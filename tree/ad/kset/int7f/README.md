# kset-int7f, rebuildable from this tree (S09, 2026-11-07)

`python3 tree/ad/kset/int7f/build_kset_int7f.py --out DIR` rebuilds the integ7 kernel set (kernels.json 9f51b2e6) from
this repository alone: 150 derivative images, 153 spec files, the derived stores and kernels.json. The spec paths are
rewritten into DIR; every spec record is checked against its file. DIR/REBUILD.json is the receipt.

- `recipe.json`: per image, the toolchain that built the image now in kset-int7f (read from the ptxas note in each
  cubin: 107 x 13.5.37, 38 x 13.2.86, 3 x 13.0.88, 2 x 13.4.92), working directory, source, exact flags (including
  `-Xptxas -v` where the original passed it: it is recorded in the fatbin), image sha256 and the sha256 of every
  source and header it reads.
- `toolchains.json`: NVIDIA PyPI wheels pinned by sha256 (nvcc, nvvm, crt, runtime; cccl for 13.2.86). nvvm must be
  pinned too: `nvidia-cuda-nvcc==13.2.86` alone pulls nvvm 13.5 and emits PTX 9.5 that ptxas 13.2 refuses.
- `Containerfile.ubuntu-24.04`: host for the 13.0.88 rows (CUDA 13.0 headers do not parse against glibc >= 2.41).
  podman or docker. `oracle_diags_sfcdiags_ruclsm` compiles at its original path
  (/work/ad/held/oracle_diags_sfcdiags_ruclsm/src) because its file-scope tables are named by a hash of the path.
- `src/`: 116 sources and headers that lived only in lane folders outside the repo (ad1-*/out/patch, ad2-hrrrv5-glue,
  two lane worktrees, the g3 captured forward sources). G8's five adjoints are at `tree/ad/kset/g8`
  (merged from ad/g8-adjoint-speed 0e390cc). `.gitattributes` keeps every input byte-exact.
- `--toolchain 13.2.86` builds every row with 13.2.86 instead.

Result from a fresh checkout (RTX 5090 workstation): 149 of 150 images byte-equal. The one that is not, rlw_rtrn_march_coalesced,
is not byte-reproducible by nvcc itself (13.5.37 cicc output changes run to run with address-space layout); its
rebuilt images give the same TL/AD words as c5760da3 on all 26 captured launches (`md_dot.py --hash-only 1`).
