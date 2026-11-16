#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# The five boundary-layer images of the tape-free route's default form, from the captured MYNN-EDMF forward
# (processors only, no card). The form: the turbulence closure live (level-2 and level-2.5 stability functions, the
# turbulence-energy terms), the plume's momentum, heat and moisture live with held coefficients, the boundary-layer
# height the column returns live (its uses inside the column held), the weak-shear rule at 1e-5 s-2, the mixing length
# held. README.md beside this script says what was shown with it.
#   NVRTC=<libnvrtc.so.13> P3_MYNN=<harness binary> build-kit.sh CAPTURED.cu OUT [STEP...]
#   STEP: src (the generated and composed source), join (the joined backward and its capture forward), tangent,
#   reverse (the taped reverse, entry 1), count (the tape count, entry 2), kit (OUT/kit with its SHA256SUMS).
#   Default: every step in that order. A reverse or count build takes 39 to 43 GB of host memory and about 18
#   minutes; they run one at a time.
set -u
M=$(cd "$(dirname "$0")/.." && pwd); J=$M/direct/join
CAP=$(readlink -f "${1:?the captured MYNN-EDMF forward source}"); mkdir -p "${2:?OUT}"; O=$(cd "$2" && pwd); shift 2
: "${NVRTC:?libnvrtc.so.13}"; NVD=$(dirname "$NVRTC")
WS=1e-5; MASK=0x7c9; DEFS="-DJOIN_PMOM -DJOIN_MIX -DJOIN_CLOSURE -DJOIN_PHEAT -DJOIN_PBLH"
export CUDA_VISIBLE_DEVICES=
t0=$(date +%s); say() { echo "$(date -u +%H:%M:%SZ) +$(( $(date +%s) - t0 ))s $*"; }
for s in ${*:-src join tangent reverse count kit}; do case $s in
src)
  python3 "$M/gen_whole.py" --freeze closure --freeze-mask $MASK --weakshear $WS --plume-live momheat --fused m8 "$CAP" \
    "$O/p.gen.cu" > "$O/gen-info.json" 2> "$O/gen.err" || { say "generator failed"; exit 2; }
  python3 "$M/fast-closure/pknock.py" "$O/p.gen.cu" "$O/base.gen.cu" O > "$O/pknock.out" || { say "pknock failed"; exit 2; }
  rm -rf "$O/src"; bash "$M/fast-closure/compose-m4l.sh" "$O/base.gen.cu" "$O/src" > "$O/comp.out" 2> "$O/comp.err" \
    || { say "compose failed"; exit 2; }
  cp "$O/gen-info.json" "$O/src/gen-info.json"
  say "src: $(sha256sum "$O/p.gen.cu" "$O/base.gen.cu" "$O/src/mynn_whole.gen.cu" | cut -c1-8 | tr '\n' ' ')";;
join)
  g++ -std=c++17 $DEFS -I"$J/blocks" "$J/card/frame_layout.cpp" -o "$O/fl" && "$O/fl" > "$O/layout.json" || { say "layout failed"; exit 2; }
  MIX_WS_T=$WS python3 "$J/card/capture_gen.py" "$CAP" "$O/layout.json" "$O/capture.cu" > "$O/capture.json" || { say "capture_gen failed"; exit 2; }
  rm -rf "$O/blocks"; ( cd "$J/card" && JW_MIN_WORDS=150 JW_OBJECTS=0 python3 arena_rewrite.py ../blocks "$O/blocks" ) > "$O/arena.log" 2>&1 \
    || { say "arena_rewrite failed"; exit 2; }
  ( cd "$J/card" && JOIN_BLOCKS="$O/blocks" NVRTC="$NVRTC" LD_LIBRARY_PATH="$NVD" python3 nvrtc_build.py ../../mynn_direct.cu \
      "$O/direct120" sm_120 -DJW_ARENA -DCARD_FLOAT_VIA_DOUBLE_ALL $DEFS ) > "$O/dir.log" 2>&1 & pd=$!
  ( cd "$M" && LD_LIBRARY_PATH="$NVD" python3 bl-forward/nvrtc_build.py "$NVRTC" "$O/capture.cu" sm_120 "$O/cap120.cubin" ) \
      > "$O/cap.json" 2> "$O/cap.err" & pc=$!
  wait $pd; rd=$?; wait $pc; rc=$?
  [ $rd = 0 ] && [ $rc = 0 ] || { say "join images failed: direct rc $rd, capture rc $rc"; exit 2; }
  say "join: $(cd "$O" && sha256sum direct120.cubin cap120.cubin | cut -c1-8 | tr '\n' ' ')$(grep -o '"frame_bytes": [0-9]*' "$O/capture.json")";;
tangent|reverse|count)
  : "${P3_MYNN:?the harness binary p3-mynn}"
  e=$( case $s in tangent) echo 0;; reverse) echo 1;; count) echo 2;; esac )
  NVRTC="$NVRTC" P3_MYNN="$P3_MYNN" bash "$M/weakshear/bimg.sh" "$O/src" $e || { say "$s image failed"; exit 2; }
  say "$s: $(cut -c1-8 "$O/src/rel-$e/SHA256SUMS" | tr '\n' ' ')";;
kit)
  K=$O/kit; rm -rf "$K"; mkdir -p "$K/p3"
  cp -p "$O/direct120.cubin" "$O/cap120.cubin" "$K/" && cp -p "$O/src/rel-0/rel.0.cubin" "$O/src/rel-1/rel.1.cubin" \
    "$O/src/rel-2/rel.2.cubin" "$O/src/gen-info.json" "$K/p3/" || { say "kit: an image is missing"; exit 2; }
  ( cd "$K" && sha256sum direct120.cubin cap120.cubin p3/rel.0.cubin p3/rel.1.cubin p3/rel.2.cubin p3/gen-info.json > SHA256SUMS )
  say "kit:"; cat "$K/SHA256SUMS";;
*) echo "step $s?"; exit 2;;
esac; done
