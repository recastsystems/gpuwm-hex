#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# tree/ad/phys/tempo/f08img/make.sh LIBNVRTC_13.4.92 OUTDIR: the taped TEMPO image (latest/aux/split2.cubin) and its F08 form.
#   1. split2.cubin from split2.main.cu as shipped: must hash to ce8d3a67... (the release's image, byte for byte);
#   2. split2-f08.cubin from the same source with coltest/f08_subgrid.py applied (the PBL subgrid-radii path, g_tempo_bl).
# NVRTC 13.4.92 (pip nvidia-cuda-nvrtc==13.4.92) and the published recipe's options (284234ec tree/ad/phys/tempo/split/
# build.py: strict arithmetic, sm_120, --maxrregcount=128). nvcc, another NVRTC, or no register cap gives other bytes.
set -eu
here=$(cd "$(dirname "$0")" && pwd); L=$1; O=$2; mkdir -p "$O"
echo "a15591d1e43c422dd41388b204791276e849a27df78c53ba9ce7cd4ddc2aa548  $here/split2.main.cu" | sha256sum -c -
cp "$here/split2.main.cu" "$O/split2-f08.main.cu"
python3 "$here/../coltest/f08_subgrid.py" "$O/split2-f08.main.cu"
python3 "$here/nvbuild.py" "$L" "$here/split2.main.cu" "$O/split2.cubin" split2.main.cu &
python3 "$here/nvbuild.py" "$L" "$O/split2-f08.main.cu" "$O/split2-f08.cubin" split2.main.cu &
wait
echo "ce8d3a67170843fd835cca96956d208b958dfe5b8a3c93d916770d74d41a65d5  $O/split2.cubin" | sha256sum -c -
sha256sum "$O/split2-f08.main.cu" "$O/split2-f08.cubin"
