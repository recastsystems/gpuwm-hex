# The joined (tape-free) MYNN-EDMF column backward

`blocks/join.hpp` is the whole-column backward of the MYNN-EDMF boundary layer (`rrfs_mynnedmf_run_l16`, the code at
commit c6d08709) with no tape: every stage's transpose joined in the reverse order of the native column. Float
primals, double bars, one thread a column. On the card it is `direct120.cubin` (built from `../mynn_direct.cu` with
`card/nvrtc_build.py`), and it runs from the values the boundary layer's forward kept for it (`cap120.cubin`, the
captured forward with probes, written by `card/capture_gen.py` at the offsets `card/frame_layout.cpp` prints). This is
the 3 km storm row's default boundary-layer route (`--boundary-layer tape-free`); a column it refuses is reversed on
the taped route by the engine, with the generated images of the same linear model.

What the default form differentiates, each part under its own define (all five are on in the shipped images):

| define | stage | block |
|---|---|---|
| `JOIN_MIX` | the diffusivities' dependence on the turbulence energy and the level-2.5 stability functions | `mixjac.hpp` |
| `JOIN_CLOSURE` | the level-2 stability functions and the stability functions inside the turbulence-energy terms; the weak-shear rule (an interface whose forecast squared shear is under the capture's `MIX_WS_T`, 1e-5 s-2 in the shipped form, keeps its stability functions' value) | `mixjac.hpp` |
| `JOIN_PMOM` | the plume's momentum transport (flux, subsidence, detrainment) with the plume's coefficients held | `join.hpp` |
| `JOIN_PHEAT` | the plume's transport of liquid-water potential temperature and total water, with held coefficients | `join.hpp` |
| `JOIN_PBLH` | the boundary-layer height the column returns (read by the surface layer at the next step), through the mirror of `f_get_pblh` | `pblhjac.hpp` |

Held at the forecast's values: the mixing length, the plume's own speed, area, condensation and top, and the
height's uses inside the column. The generated images of the same form (tangent, taped reverse, tape count) come from
`../../gen_whole.py --freeze closure --freeze-mask 0x7c9 --weakshear 1e-5 --plume-live momheat --fused m8`, then
`../../fast-closure/pknock.py ... O` and `../../fast-closure/compose-m4l.sh`. `../../fast-closure/build-kit.sh` builds
all five images; `../../fast-closure/test_form.sh` pins their sources.

Tests (processors only): `code/run_cpu_tests.sh` runs every stage's transpose against the forward statements it
reverses (dot identities and central differences, with a planted error each must see) and the workspace test of the
arena form the card runs (no stage may read a word of the workspace it did not write).
