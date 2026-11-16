#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Generate and build the New Tiedtke derivative entries (card image and processor harness).
#   build.sh FORWARD_SOURCE OUT [card|host|all]
# FORWARD_SOURCE: the forward NVRTC text the recording runs (hexcore.mpasphys.ntiedtke_v842.cuda_source(), or the
# capture's source object). ENGINE_PY: a python with the gpuwm engine (its New Tiedtke name tables). NVCC: a cu13 nvcc.
# Card image: sm_120 + sm_89 + compute_80 PTX, the strict forward's numerics (--fmad=false --ftz=false, IEEE div/sqrt).
set -euo pipefail
SRC=$1; OUT=$2; WHAT=${3:-all}
H=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
cat "$SRC" > "$OUT/nt-forward.cu"
# the driver forcing fold (driver_rrfs.py _CONVECTION_FORCING_CUDA, a module of its own) shares the image: FOLD_SRC
if [ -n "${FOLD_SRC:-}" ]; then printf "
" >> "$OUT/nt-forward.cu"; cat "$FOLD_SRC" >> "$OUT/nt-forward.cu"; fi
cp "$H/cuda/nt_ad.cuh" "$H/cuda/nt_ad.cu" "$H/cuda/hostshim.h" "$H/host/nthost.cpp" "$OUT/"
"${ENGINE_PY:?python with the gpuwm engine}" -I "$H/gen/nt_tables.py" "$OUT/tables.json"
python3 -I "$H/gen/gen_nt.py" "$OUT/nt-forward.cu" "$OUT/tables.json" "$OUT"
python3 -I "$H/gen/gen_host.py" "$OUT/nt_kernels.json" "$OUT/nt_host.inc"
cd "$OUT"
if [ "$WHAT" = card ] || [ "$WHAT" = all ]; then
  "${NVCC:?a cu13 nvcc}" -std=c++17 --threads 3 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_80,code=compute_80 -fatbin -O3 --fmad=false --ftz=false --prec-div=true --prec-sqrt=true \
    -o nt_ad.fatbin nt_ad.cu
fi
if [ "$WHAT" = host ] || [ "$WHAT" = all ]; then
  for P in 0 1 2; do
    g++ -std=c++17 -O2 -ffp-contract=off -fno-fast-math -fexcess-precision=standard -w -DNT_PLANT=$P -o nthost-p$P nthost.cpp
  done
fi
sha256sum nt-forward.cu nt_R.inc nt_entries.inc nt_kernels.json nt_ad.* nthost-p* 2>/dev/null | tee build.txt
