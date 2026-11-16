# cumulus: tangent and adjoint rows for every New Tiedtke launch

NOAA's hrrrv5 deck runs `cu_ntiedtke` (New Tiedtke, MPAS 44c0849 /
MMM-physics a4baf7f3, BSD-3; hex's port in `hexcore/mpasphys/ntiedtke_v842.py` over the engine's
`gpuwm/core/kernels/ntiedtke.cu`) on its 12 km meshes. The window refuses any launch without a derivative row, so
no 12 km recording with the scheme could be reversed. This folder gives every launch of the scheme a row.

## How the rows are made (as the RUC land rows were)

- `gen/nt_tables.py` (engine venv): the engine's own name tables: which parameter names are level or surface arrays,
  float or int, each stage's level base, and the storage each (stage, parameter) resolves to.
- `gen/gen_nt.py FORWARD.cu TABLES.json OUT`: reads the forward's exact NVRTC text (`cuda_source()`, or the capture's
  source object) and re-types its scheme region (the 20 stages, `ntiedtke_midlevel` and hex's four `mpas_ntiedtke_*`
  kernels) over a scalar R. `float` becomes R; each f32 operation the forward calls becomes an `r_*` overload whose
  primal word is that same forward function (`cuda/nt_ad.cuh`). Nothing in the physics is edited by hand.
  - R = AF (tangent: f32 primal word + double tangent), R = AR (taped reverse: f32 primal word + tape slot).
  - Per kernel a tangent entry `NAME__tl` and a taped reverse `NAME__ad` bind the forward's arguments in its own
    order, shadow the launch thread's own column of every array in a workspace (f32 arrays as R words, int arrays as
    copies, the launch-report words as dummies), run the re-typed body and write tangents or adjoints.
  - The trajectory is the pre-launch state. The entries never write a primal word.
- Frozen at the recorded path: every comparison, `mpas_nt_int` truncation (the INTEGER cloud-base pressure), fmaxf /
  fminf arm (the first operand on a tie), fabs sign, level index, cloud type, ldcum, loddraf, klab and the slab latch
  (`mpas_llo3_start`) are decided on the primal words as the forward decides them. Integer state is never a carrier,
  so the linear model is the derivative along the recorded branches and integer counts.
- Precision: partials in double from the f32 words; tangents and adjoints accumulate in double and are rounded once
  per written word.
- Storage aliases inside one launch (post_conversion: pqc = qcf, pqi = qif, pu = uf, pv = vf, the engine's
  NT_ALIASES) share one shadow; the alias argument gets no derivative binding of its own.

## Files

- `cuda/nt_ad.cuh` carriers and tape; `cuda/nt_ad.cu` the image's translation unit; `cuda/hostshim.h` the processor
  form of the CUDA intrinsics used by the proofs (IEEE f32, no contraction, no flush to zero = strict).
- `gen/gen_nt.py`, `gen/nt_tables.py`, `gen/gen_host.py` (processor dispatcher), `gen/mkspec_nt.py` (ad.entry/v1
  specs and kernel-set rows from the generated layout and the tape census), `gen/add_rows_nt.py` (a kernel set =
  base + these rows; one image holds every entry).
- `host/nthost.cpp` processor proofs on a captured launch; `host/mkbundle.py` capture manifest -> bundle.
- `build.sh FORWARD.cu OUT [card|host|all]`.
- `kset/`: the 26 rows (`rows.json`) and their ad.entry/v1 specs, as `gen/mkspec_nt.py` wrote them. Every row names
  the one image (`derivative.shared_image`: `nt_ad.fatbin`), which a kernel set holds once in `cubins/`.

## Proofs (processors)

`nthost BUNDLE fwd`: the forward's host form on the captured pre-state against the captured post-state.
`nthost BUNDLE dot SEED`: the tangent's and the taped reverse's primal words against the post-state; the tape census;
the transpose check per column (whole, and with the in/out identity removed), on real 12 km columns. `nthost-p1` /
`nthost-p2` carry a planted 1e-3 error in the reverse's / the tangent's product rule and must fail.

## Results (2026-11-15, round 1, processors; evidence/)

Real 12 km columns: the 12 km case's own 15:00Z start (init.nc, 2,890 columns, 59 levels) through `host/replica.py`,
the processor replica of hex's call (surface fluxes and the dynamics/boundary-layer/radiation forcing are zero in a
start file: deep and mid-level triggers, not the shallow one). 1,246 columns convect (ktype 1: 221, 2: 53, 3: 972).
23 launches a step (forcing fold, dx, 20 stages, the ascent latch), each through `nthost` (`evidence/proof-rep1.log`):
- tangent and taped reverse primal words = the forward's, 0 differing words on every launch;
- transpose check per column, two seeds: worst 5.1e-14 (cuascn), most 1e-16 to 1e-15; 0 columns over 1e-10;
- planted 1e-3 product-rule errors (reverse, tangent) seen on every launch with a live product (20 of 23);
- tangent against centred differences of the forward on columns whose integer outputs do not move: median falls to the
  f32 floor at eps 1e-4 on most launches; the misses that do not shrink with eps are words sitting exactly on a kink
  (an earlier stage's clamp left a tie: adjust's pmfude_rate where zerate = 0, updraft_scale's zmfub on its 1e-10 floor).
Tape census (largest per column): cutypen 33,610, cuascn 8,766, cuinin 5,234, cuddrafn 4,461, the rest under 2,300.
Kernel set: `gen/add_rows_nt.py [--copy] BASE_KSET OUT BUILD kset` (26 rows; image nt_ad.fatbin 7eab4ec6).

## The forward text the image is of

The image linearises one forward text: `cuda_source()` followed by the driver's forcing fold, sha256 59e5f609...
(every spec's `forward_source_sha256`). `evidence/build-rel.txt` is the build from this tree: the same image,
byte for byte, as the first build. If the forward text of New Tiedtke or of the fold changes, the image and the
rows are rebuilt with `build.sh` and `gen/mkspec_nt.py`, and the kernel set that holds them gets a new version.
