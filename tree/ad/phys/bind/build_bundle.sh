#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# L2 bindings (ad/window-L2-bindings): the ad.scheme.rows/v1 bundle the P1 runner loads with AD_SCHEMES=<OUT>.
#   1. derivative images of P3 and P4 from their committed recipes (build_images.sh: NVRTC 13.2, the forward's
#      strict options, sm_120), unless IMG already holds them;
#   2. P8's rule kernels (p8/cuda/p8.cu) and the two L2 rules (bind/cuda/l2_rules.cu), nvcc 13.2, P8's strict recipe;
#   3. schemes.json from the captures, P3's gen-info, P4-ENTRIES.json and P8's table, checked against PLAN.
# Build time only. usage: NVRTC=<libnvrtc 13.2> NVCC=<nvcc 13.2> build_bundle.sh IMG OUT PLAN
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); phys=$(cd "$here/.." && pwd)
IMG=${1:?image scratch}; OUT=${2:?bundle dir}; PLAN=${3:?plan.json}
NVCC=${NVCC:?nvcc 13.2}; C=${CAPTURES:?capture root (the recorder's captures)}
[ -s "$IMG/p3/rel.2.cubin" ] && [ -s "$IMG/p4/cubins/ugwp.rev.sm_120.cubin" ] || bash "$here/build_images.sh" "$IMG"
mkdir -p "$OUT/p3" "$OUT/p4" "$OUT/p6"
# P6 sea ice: the whole-launch image from its own recipe (p6/build.sh, NVRTC), the bind kernels, and the Vilka
# saturation table the forward module carries (from ad2-ruc-seaice, pinned by sha256).
P6_TABLE=${P6_TABLE:?the forward sea-ice module's Vilka saturation table (sha256 below)}
P6_TABLE_SHA256=${P6_TABLE_SHA256:-7dd3e5e565bb25c8807b1cada5e8a14ec33e71b9654aed4f4f9b49b98c78e7fb}
echo "$P6_TABLE_SHA256  $P6_TABLE" | sha256sum -c --quiet
[ -s "$IMG/p6/p6_seaice_sm_120.cubin" ] || NVRTC_LIB=${NVRTC:?libnvrtc} sh "$phys/p6/build.sh" "$IMG/p6" sm_120
flags=(--fmad=false --prec-div=true --prec-sqrt=true -O3)
"$NVCC" -cubin -arch=sm_120 "${flags[@]}" -o "$OUT/p8-sm_120.cubin" "$phys/p8/cuda/p8.cu" &
"$NVCC" -cubin -arch=sm_120 "${flags[@]}" -o "$OUT/l2_rules-sm_120.cubin" "$here/cuda/l2_rules.cu" &
"$NVCC" -cubin -arch=sm_89 "${flags[@]}" -o "$OUT/l2_rules-sm_89.cubin" "$here/cuda/l2_rules.cu" &
"$NVCC" -cubin -arch=sm_120 "${flags[@]}" -o "$OUT/p6/p6_bind-sm_120.cubin" "$here/cuda/p6_bind.cu" &
wait
cp "$IMG/p6/p6_seaice_sm_120.cubin" "$P6_TABLE" "$OUT/p6/"
cp "$IMG"/p3/rel.{0,1,2}.cubin "$IMG/p3/gen-info.json" "$OUT/p3/"
cp "$IMG"/p4/cubins/{mynnsfc,ugwp}.{tl,rev}.sm_120.cubin "$OUT/p4/"
python3 "$here/gen_rows.py" "$C/hrrrv5_x1.2562" "$IMG/p3/gen-info.json" "$phys/p4/P4-ENTRIES.json" "$OUT" --plan "$PLAN"
(cd "$OUT" && sha256sum schemes.json *.cubin p3/* p4/* p6/* > SHA256SUMS)
echo "bundle $OUT"
