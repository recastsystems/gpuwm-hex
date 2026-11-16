#!/bin/bash
# P3: regenerate, build and prove the whole-launch MYNN-EDMF c6d08709 tangent/adjoint on captured hrrrv5 launches.
# Usage: run_p3.sh SCRATCH   (heavy outputs go to SCRATCH; nothing here touches the forward tree)
set -euo pipefail
U=$(cd "$(dirname "$0")" && pwd)
S=${1:?scratch dir}
C=${CAPTURES:?capture root (the recorder's captures)}
NVRTC=${NVRTC:?libnvrtc.so.13}
SRC=$C/hrrrv5_mynn_dd_c6/objects/70da4af74192b47fef6dd002101d64a22659b38a93c496a7832ec49071689727.cu
mkdir -p "$S/gen" "$S/runs"
# 1. generate (build time, pinned source hash checked inside)
python3 "$U/gen_whole.py" "$SRC" "$S/gen/mynn_whole.gen.cu" > "$S/gen/gen-info.json"
# 2. Rust harness (libcuda + NVRTC only)
(cd "$U/harness" && CARGO_TARGET_DIR="$S/target" cargo build --release -j 4 --offline)
B=$S/target/release/p3-mynn
# 3. three entries, three NVRTC programs in parallel, the forward image's own options
"$B" build --src "$S/gen/mynn_whole.gen.cu" --include "$U/cuda" --nvrtc "$NVRTC" --arch sm_120 --out "$S/gen/rel" > "$S/gen/build.json"
# 4. proofs on every required capture
caps=""
for d in hrrrv5_x1.2562/rrfs_mynnedmf_run_l16/1/launch-000000 hrrrv5_x1.2562/rrfs_mynnedmf_run_l16/2/launch-000000 \
         hrrrv5-convective/hrrrv5-convective/rrfs_mynnedmf_run_l16/59/launch-000000 \
         hrrrv5-convective/hrrrv5-convective/rrfs_mynnedmf_run_l16/60/launch-000000 \
         hrrrv5_mynn_dd_c6/rrfs_mynnedmf_run_l16/1/launch-000000 hrrrv5_mynn_dd_c6/rrfs_mynnedmf_run_l16/1/launch-000001 \
         hrrrv5_mynn_dd_c6_winter/rrfs_mynnedmf_run_l16/1/launch-000000 hrrrv5_mynn_dd_c6_winter/rrfs_mynnedmf_run_l16/1/launch-000001; do
  caps="$caps --capture $C/$d"
done
P3_EPS=1,0.3,0.1,0.03,0.01,0.003,1e-3,3e-4,1e-4 P3_ARENA_X=12 P3_STACK=32768 \
  "$B" prove --cubin "$S/gen/rel" --gen-info "$S/gen/gen-info.json" $caps --out "$S/runs/all.json" \
  --tests replay,identity,dot,fd --seeds 2 --field-dots --budget-gb 4
