#!/bin/sh
# ad/phys P6 build: assemble the whole-launch sea-ice derivative module and the
# Rust proof driver. Build time only; nothing here runs in a step loop.
#   NVRTC_LIB=/path/libnvrtc.so.13 ./build.sh <out-dir> [sm_120|sm_89]
# The composite source is the 11-02 ad2-ruc-seaice whole-launch source (forward
# math prelude of hex pin 16251589 + seaice derivative + launch adapters,
# sha256 recorded in README.md) followed by cuda/p6_seaice.cu.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out=${1:?out dir}; arch=${2:-sm_120}
mkdir -p "$out"
cat "$here/cuda/seaice_ruc_ice_whole.cu" "$here/cuda/p6_seaice.cu" > "$out/p6_seaice_all.cu"
cp "$here/cuda/seaice.ad.cu" "$out/"
( cd "$here/tools" && rustc --edition=2021 -O -Awarnings nvrtc_compile.rs -o "$out/nvrtc_compile" )
( cd "$out" && AD_NVRTC_LIB="$NVRTC_LIB" LD_LIBRARY_PATH="$(dirname "$NVRTC_LIB")" AD_ARCH="--gpu-architecture=$arch" \
    ./nvrtc_compile p6_seaice_all.cu "p6_seaice_$arch.cubin" ) &
( cd "$here" && CARGO_TARGET_DIR="$out/target" cargo build --release --offline -q ) &
wait
sha256sum "$out/p6_seaice_all.cu" "$out/p6_seaice_$arch.cubin" "$out/target/release/ad-phys-p6"
