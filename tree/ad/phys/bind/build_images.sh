#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# L2 bindings (ad/window-L2-bindings): derivative images of P3 (MYNN-EDMF whole launch) and P4 (MYNN surface layer,
# GSL drag) from their committed recipes, NVRTC 13.2, the forward strict options, sm_120. Build time only.
# usage: NVRTC=<libnvrtc.so.13 (13.2)> build_images.sh SCRATCH
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); phys=$(cd "$here/.." && pwd)
S=${1:?scratch}; NVRTC=${NVRTC:?libnvrtc 13.2}; C=${CAPTURES:?capture root (the recorder's captures)}
export PATH=$HOME/.cargo/bin:$PATH CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-8}
mkdir -p "$S/p3" "$S/p4"
SRC=$C/hrrrv5_mynn_dd_c6/objects/70da4af74192b47fef6dd002101d64a22659b38a93c496a7832ec49071689727.cu
python3 "$phys/mynn/gen_whole.py" "$SRC" "$S/p3/mynn_whole.gen.cu" > "$S/p3/gen-info.json"
(cd "$phys/mynn/harness" && CARGO_TARGET_DIR="$S/target-p3" cargo build --release --offline -q)
( "$S/target-p3/release/p3-mynn" build --src "$S/p3/mynn_whole.gen.cu" --include "$phys/mynn/cuda" --nvrtc "$NVRTC" \
    --arch sm_120 --out "$S/p3/rel" > "$S/p3/build.json" 2> "$S/p3/build.log" ) &
p3=$!
( bash "$phys/p4/build.sh" "$C/hrrrv5_x1.2562" "$S/p4" "$NVRTC" sm_120 > "$S/p4/build.out" 2>&1 ) &
p4=$!
rc=0; wait $p3 || rc=1; wait $p4 || rc=$((rc+2))
sha256sum "$S"/p3/rel.*.cubin "$S"/p4/cubins/*.cubin > "$S/images.sha256" || true
echo "rc=$rc"; exit $rc
