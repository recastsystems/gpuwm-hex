#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Build the two upper-damping derivative images and specs into OUT (processors only; NVCC = a cu13 nvcc).
#   build.sh OUT
set -euo pipefail
OUT=$1; H=$(cd "$(dirname "$0")" && pwd)
NVCC=${NVCC:?a cu13 nvcc}
FLAGS="-gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_80,code=compute_80 -O3 --fmad=false --ftz=false --prec-div=true"
mkdir -p "$OUT"; python3 "$H/mkspecs.py" "$OUT"
for FN in upper_cam_f32 upper_rayleigh_f32; do
  $NVCC -fatbin $FLAGS -o "$OUT/$FN.fatbin" "$H/cuda/$FN.ad.cu"
done
sha256sum "$OUT"/*.fatbin "$OUT"/specs/*.json "$H"/cuda/*.cu | tee "$OUT/build.txt"
