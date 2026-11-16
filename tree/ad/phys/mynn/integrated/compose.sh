#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# x20 mynn-integrated (2026-11-09): compose the MYNN stage lanes into one P3 source + include dir (build time only).
# usage: compose.sh BASE.gen.cu OUTDIR VARIANT      VARIANT: A (M8 M6 M2 M3 M10 M9) | Ainl (A, M9 float wrappers inlined) | B (A + M4)
# a12 tp-pbl (2026-11-13): VARIANT L (M8 M9 only) for a gen_whole.py --freeze none source (the whole closure live), and
# VARIANT auto: the stages follow the source's own freeze mode (gen_whole.py header line): closure -> B, closure-plume -> A,
# none -> L. Breakage the check below prevents: M6, M2, M3, M10 and M4 are written for outputs the source FREEZES (M6: sm,
# sh, el primal-only; M2: pblh, psig primal-only; M3: d/dpblh zero in the reverse; M10: no reverse of exch_m/exch_h; M4:
# the plume on frozen inputs). Applied to a source that keeps those outputs live they hand back an image whose tangent
# or adjoint drops a live derivative under the live source's name. M8 (record of the true inputs by tape id) and M9 (no
# --frozen-closure flag: every adjoint kept) are exact for any freeze mode. Before this, a --freeze none source failed in
# patch_m6_primal.py ("not a --freeze closure source") and a closure-plume source in m4_plume.py under B: each stage
# refused rightly, but no variant built the live source without it.
# BASE = gen_whole.py --freeze closure --fused m8 output (the f6 recipe + --fused m8; the M8 lane gm8c = bf9516c0).
# Order: M6 M2 M3 M10 [M4] are local call-site/body swaps (any order); M9 last (it rewrites the AD entry sweep line).
# Not composed into the image: M1 (taped swap measured +0.3 % at pk120, changes summation order), M7 (taped swap records
# the same Jacobian rows, no speedup until the stage is inlined into a tapeless kernel), M5 (never runs: edmf_dd = 0).
set -eu
T=$(cd "$(dirname "$0")/.." && pwd); IN=$(readlink -f $1); O=$2; V=${3:-A}
FZ=$(grep -o -m1 '(gen_whole.py --freeze [^)]*)' $IN | sed 's/.*--freeze //; s/)$//' || true); FZ=${FZ:-none}
# a freeze-group source (gen_whole.py --freeze closure --freeze-mask M) keeps live every group whose bit is clear. Set A's
# primal-only stages need groups 0 1 2 3 7 frozen (M2, M3: launch pblh and pblh; M6: level-2 sm/sh and el; M10: exch_m/h),
# set B also group 8 (M4: the updraft plume). A mask without them is composed as L.
MASK=$(grep -o -m1 '^#define MW_FRZ_MASK 0x[0-9a-f]*u' $IN | sed 's/.* 0x//; s/u$//' || true)
if [ -n "$MASK" ]; then
  M=$((16#$MASK))
  if [ $((M & 0x8F)) -ne $((0x8F)) ]; then FZ=closure-groups-live
  elif [ $((M & 0x18F)) -ne $((0x18F)) ]; then FZ=closure-plume; fi
fi
if [ "$V" = auto ]; then case $FZ in closure) V=B;; closure-plume) V=A;; none|closure-groups-live) V=L;; *) echo "compose: unknown freeze mode $FZ" >&2; exit 2;; esac; fi
case "$V:$FZ" in
  A:closure|Ainl:closure|B:closure|A:closure-plume|Ainl:closure-plume|L:none|L:closure|L:closure-plume|L:closure-groups-live) ;;
  *) echo "compose: variant $V applies stages that freeze outputs the source ($FZ) keeps live; use VARIANT auto (closure B, closure-plume A, none L)" >&2; exit 2;;
esac
echo "compose: source freeze $FZ, variant $V" >&2
# COMPOSE_PLAN_ONLY=1: print the choice above and stop (test_compose_variants.sh part 1; no stage runs, nothing written).
[ "${COMPOSE_PLAN_ONLY:-0}" = 1 ] && exit 0
mkdir -p $O; cd $O
if [ "$V" = L ]; then
  cp $IN s1.gen.cu; cp s1.gen.cu s2.gen.cu; cp s2.gen.cu s3.gen.cu; cp s3.gen.cu s4.gen.cu     # M6 M2 M3 M10 not applied
else
python3 $T/tape_tools/patch_m6_primal.py $IN s1.gen.cu
python3 $T/tape_tools/patch_m2.py s1.gen.cu s2.gen.cu
python3 $T/fused/patch_m3.py s2.gen.cu s3.gen.cu
python3 $T/tape_tools/m10_swap.py s3.gen.cu s4.gen.cu
fi
# bl-live-reverse (2026-11-14): a gen_whole.py --plume-live mom source keeps u1, v1 live through M4 (the plume's momentum
# transport answers the wind; every coefficient is from the frozen copies). Variant A on such a source would leave the
# plume's heat and moisture live into the cloud-blend tail, which is not the form its name says: refused below.
M4L=""; if grep -q -m1 "(gen_whole.py --plume-live mom)" $IN; then M4L="--live u1,v1"; fi
if [ -n "$M4L" ] && [ "$V" != B ]; then echo "compose: a --plume-live mom source is composed as B (M4 with u1, v1 live)" >&2; exit 2; fi
if [ "$V" = B ]; then python3 $T/cuda/m4_plume.py $M4L s4.gen.cu s4m.gen.cu > m4-info.json; mv s4m.gen.cu s4.gen.cu; fi
python3 $T/m9/gen_m9.py s4.gen.cu m9_stage.cuh > m9-info.txt
if [ "$V" = Ainl ]; then sed -i -e "s/__device__ __noinline__ float m9w_/__device__ __forceinline__ float m9w_/" m9_stage.cuh m9_stage.cuh.structs; fi
python3 $T/m9/patch_m9.py s4.gen.cu m9_stage.cuh mynn_whole.gen.cu
rm -rf inc; cp -a $T/cuda inc; cp $T/fused/m3_cond.cuh $T/fused/m3_swap.cuh inc/
python3 $T/integrated/tape_int.py $T/cuda/mynn_tape.cuh inc/mynn_tape.cuh
sha256sum $IN s1.gen.cu s2.gen.cu s3.gen.cu s4.gen.cu mynn_whole.gen.cu inc/mynn_tape.cuh | cut -c1-8,65- > SHAS
for w in M9_EXT "m6_length(" m2_get_pblh_mr m3_tape_swap m10_post_run_tape mw_m8_record_in; do grep -c "$w" mynn_whole.gen.cu || true; done
