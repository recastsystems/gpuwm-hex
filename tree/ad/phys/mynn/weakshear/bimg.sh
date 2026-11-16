#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Build boundary-layer derivative images of one composed source folder (integrated/compose.sh OUT). Processors only, no
# card: a reverse or count build takes 39 to 43 GB of host memory and about 18 minutes; run one at a time.
#   P3_MYNN=<harness binary p3-mynn> NVRTC=<libnvrtc.so.13> bimg.sh SRC_DIR ENTRIES [DEFINE...]
#   ENTRIES 0 = the tangent; 1,2 = the taped reverse and the tape count (built with M9_EXT)
# Same builder and options as the row's own boundary-layer images: sm_120, MW_TILE=8.
set -u
S=$(cd "${1:?composed source folder}" && pwd); E=${2:?entries: 0 or 1,2}; shift 2
: "${P3_MYNN:?the harness binary (harness/, cargo build --release)}" "${NVRTC:?libnvrtc.so.13}"
D=""; [ "$E" != 0 ] && D="--define M9_EXT"; for d in "$@"; do D="$D --define $d"; done
O=$S/rel-$E; mkdir -p "$O"; t0=$(date +%s)
env CUDA_VISIBLE_DEVICES= LD_LIBRARY_PATH="$(dirname "$NVRTC")" P3_ENTRIES=$E nice -n 10 "$P3_MYNN" build \
  --src "$S/mynn_whole.gen.cu" --include "$S/inc" --nvrtc "$NVRTC" --arch sm_120 --define MW_TILE=8 $D \
  --out "$O/rel" > "$O/build.json" 2> "$O/build.err"
rc=$?
echo "rc=$rc wall=$(( $(date +%s) - t0 ))s entries $E defines:$D" > "$O/build.rc"
( cd "$O" && sha256sum rel.*.cubin > SHA256SUMS 2>/dev/null )
cat "$O/build.rc"; exit $rc
