#!/usr/bin/env bash
# ad/phys P4: generate and compile the MYNN surface-layer and GSL drag derivative modules.
# usage: build.sh <capture-root> <scratch> <libnvrtc.so.13> <arch...>
# capture-root: the hrrrv5_x1.2562 capture (ad.hex.launch/v1); sources are sha256-pinned below.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cap=$1 scratch=$2 lib=$3; shift 3
MS=9c87dce88bee3f9595bf84410375caed4884c51b4767c4e8367d77cfdf8f7eb9   # mynnsfc_columns/from/to/tables source
US=e1af1284c8b51bd0e871d00208f894ce0125a32a55999dbe43eda27316886730   # ugwp_oro_run/dx/diff source
export CARGO_TARGET_DIR=$scratch/target CARGO_BUILD_JOBS=4
(cd "$here/gen" && cargo build --release --offline -q)
(cd "$here/harness" && cargo build --release --offline -q)
mkdir -p "$here/gen-out"
"$scratch/target/release/p4-gen" mynnsfc "$cap/mynnsfc_columns/1/launch-000000/objects/$MS.cu" $MS \
    "$here/cuda/p4_ad.cuh" "$here/cuda/mynnsfc_entries.cuh" "$here/gen-out/mynnsfc.p4.cu"
"$scratch/target/release/p4-gen" ugwp "$cap/ugwp_oro_run/1/launch-000000/objects/$US.cu" $US \
    "$here/cuda/p4_ad.cuh" "$here/cuda/ugwp_entries.cuh" "$here/gen-out/ugwp.p4.cu"
# G4: the generated ugwp body keeps its column arrays in the frame arena (gen/p4_frame.py, storage only).
python3 "$here/gen/p4_frame.py" --body "$here/gen-out/ugwp.p4.cu"
for arch in "$@"; do
    "$scratch/target/release/p4-proof" build "$here/gen-out" "$scratch/cubins" "$lib" "$arch" > "$scratch/build.$arch.log" 2>&1 \
        || { grep -n " error" "$scratch/build.$arch.log" | head -40; exit 1; }
    grep -E '^"(mynnsfc|ugwp)"' "$scratch/build.$arch.log" | cut -c1-40
done
