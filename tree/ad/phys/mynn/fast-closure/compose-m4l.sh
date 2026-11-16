#!/bin/bash
# full/closure (from the ranking lane): compose a freeze-group source as variant L (no M6 M2 M3 M10)
# and then apply stage M4 (the plume on frozen copies of its inputs, as the shipped form has it; --live u1,v1 for a
# --plume-live mom source), then M9, as integrated/compose.sh does for B.   rk-compose.sh BASE.gen.cu OUTDIR
set -eu
T=$(cd "$(dirname "$0")/.." && pwd); IN=$(readlink -f $1); O=$2
mkdir -p $O; cd $O; cp $IN s4.gen.cu
M4L=""; if grep -q -m1 "(gen_whole.py --plume-live mom\(heat\)\?)" $IN; then M4L="--live u1,v1"; fi
python3 $T/cuda/m4_plume.py $M4L s4.gen.cu s4m.gen.cu > m4-info.json; mv s4m.gen.cu s4.gen.cu
python3 $T/m9/gen_m9.py s4.gen.cu m9_stage.cuh > m9-info.txt
python3 $T/m9/patch_m9.py s4.gen.cu m9_stage.cuh mynn_whole.gen.cu
rm -rf inc; cp -a $T/cuda inc; cp $T/fused/m3_cond.cuh $T/fused/m3_swap.cuh inc/
python3 $T/integrated/tape_int.py $T/cuda/mynn_tape.cuh inc/mynn_tape.cuh
sha256sum $IN s4.gen.cu mynn_whole.gen.cu | cut -c1-8,65- > SHAS
