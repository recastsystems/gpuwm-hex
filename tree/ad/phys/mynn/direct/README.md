# Direct (tape-free) MYNN-EDMF reverse for the binder

Off by default. Without `AD_P3_DIRECT=1` the binder runs the taped P3 adjoint exactly as before (the runner's `p3()`).

With `AD_P3_DIRECT=1 AD_P3_DIRECT_IMAGE=<direct.cubin> AD_P3_DIRECT_FWD=<capture.cubin>`, each MYNN-EDMF adjoint launch
(`rrfs_mynnedmf_run_l16`, mode 1) runs the runner's `p3_direct()` instead, in
batches of `AD_P3_DIRECT_BATCH` columns (default 16384):

1. **Forward with captures** (`join/card/capture_gen.py`, entry `mynn_capture_l1`). The hex forward source the 3 km
   program's P3 row binds (sha256 70da4af7...), one lane a column. Probes at code/probes.py's points (the CPU oracle's
   and the CPU route's) write the joined backward's cut-entry words straight into one `ColumnFrame<float>` a column, by
   byte offset (`join/card/frame_layout.cpp`, the CPU build of join.hpp). It runs on a copy of the launch's pre-state,
   with a zeroed workspace (as the plan clears the forward's), so neither the pre-state nor the plan's buffers change.
2. **The joined backward** (`mynn_direct.cu`, entry `mynn_whole_ad_direct`). Gathers the output bars from the 91 MwBind
   rows (ybar = the shadow snapshot) into `ColumnBars`, runs `join/blocks/join.hpp`'s `column_reverse`, and scatters
   the input bars into the shadow (xbar). Words the column does not touch keep identity bars, words the driver overwrites
   get zero. Built in the form that ships
   (m150o0e: `arena_rewrite.py JW_MIN_WORDS=150 JW_OBJECTS=0`, `-DJW_ARENA -DCARD_FLOAT_VIA_DOUBLE_ALL`): 21,072 B of
   stack a thread, 48 KiB of workspace a launched thread (`AD_P3_DIRECT_WS`), shared-memory carveout 0.

It refuses, by name and before any bar is accepted: an image whose metadata differs (`mynn_direct_info` version 2,
mask 8191 of 8191; `mynn_capture_info` frame bytes equal to the backward's), parameter bars, helper-card shards,
launch switches other than the 3 km recipe the join was proven for (code 5), an `initflag` launch (code 6:
`f_mym_initialize` is not in the join), a frame the capture did not write (code 7), a live plume (code 2: the 60 km
gen-A image), and a workspace too small (code +64). Tangent launches stay taped.

Build (no card, NVRTC 12.8 or later for sm_120; `V` the folder of libnvrtc):
```
cd join && g++ -std=c++17 -Iblocks card/frame_layout.cpp -o OUT/fl && OUT/fl > OUT/layout.json
python3 card/capture_gen.py <fwd source 70da4af7...cu> OUT/layout.json OUT/capture.cu
LD_LIBRARY_PATH=$V python3 <bl-forward nvrtc_build.py> $V/libnvrtc.so.13 OUT/capture.cu sm_120 OUT/cap120.cubin --ptxas-options=-v
cd card && JW_MIN_WORDS=150 JW_OBJECTS=0 python3 arena_rewrite.py ../blocks OUT/blocks-m150o0
JOIN_BLOCKS=OUT/blocks-m150o0 NVRTC=$V/libnvrtc.so.13 LD_LIBRARY_PATH=$V python3 nvrtc_build.py ../../mynn_direct.cu OUT/direct120 sm_120 -DJW_ARENA -DCARD_FLOAT_VIA_DOUBLE_ALL
```
The capture image takes about 11 minutes (one entry; ptxas), the backward about 35 s.

The 3 km storm row ships five images of one linear model (`../fast-closure/build-kit.sh` builds them from the
captured forward; `join/README.md` says what each define differentiates): the capture forward, this backward, and
the tangent, taped reverse and tape count of the same form, which the engine uses for the columns this backward
refuses. Processor tests: `join/code/run_cpu_tests.sh`.
