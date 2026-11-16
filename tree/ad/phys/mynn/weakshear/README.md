# The weak-shear rule of the exact boundary-layer form (`gen_whole.py --weakshear`)

The three images of the kit below are built from this option (where the door table offers a route for them, the
sensitivity command lays them over the row's scheme bundle: `--boundary-layer guarded`). It is off unless asked
for: without it the generator writes byte for byte what it wrote before (`integrated/test_weakshear.sh`).

## What it is

The exact boundary-layer derivative (MYNN-EDMF with nothing held: `--freeze none --fused m8 --l25-quot exact`)
carries the slope of the two stability functions sm and sh through the Richardson number
`ri = -gh / max(gm, 1e-10)`. That slope goes as 1 / gm, with gm the squared wind shear between two model levels. In a
calm column it is hundreds of times larger than at an ordinary interface, and over a long window the linear model
follows it where the real model, bounded by the functions' own limits, does not. On a 3 km storm box one fog column
ran the 59-step tangent of a +1 g/kg nudge to +3.6e-02 where the real model moves by under 1e-04 whichever way.

`--weakshear [T]` (T in s-2, default 1e-4): in `f_mym_turbulence`, before `elq = el(k) qkw(k)`, where the forecast's
own squared shear gm(k) of the interface is under T, sm(k), sh(k), qdiv, gamt, gamq and gamv keep their value and
carry no derivative. One decision from the forward value alone, so the tangent, the taped reverse and the count
entries take the same decision from the same bits. No size of any derivative is capped. The forecast is unchanged.

## What else it holds where it binds

The hold comes after the scheme's own floors on sm and sh, so at a held interface it also removes:

- the slope of the plume floor `0.04 min(10 mfmax, 1)` and of the cloud floor `0.04 min(cldavg, 0.5)` through sm
  and sh (and of the `tkeprod_dn` and closure-2.7 floors);
- the slope of sm and sh in the turbulence energy and the mixing length, not only in the stratification and shear;
- through qdiv, the turbulence energy's slope of `elh = elq qdiv` in the production terms.

What stays live there: `elq = el qkw`, and the solves. The size of these held parts was not measured.

## How much it holds

On a 12 km cut of the storm box (2,890 columns, 60 s steps) at 1e-4 s-2: 43 per cent of the ten lowest interfaces
and 78 per cent of all interfaces, 3 and 5 per cent of the two lowest. On the 3 km box: 0.8, 2.0, 8.5 and 19.5 per
cent of the four lowest. The threshold is in s-2 on layers 20 to 60 m deep and was chosen on the 3 km box; on another
vertical grid or a coarse mesh it needs its own runs. 3e-5 and 3e-4 were run as tangents at 3 km only.

## How to build it

```
python3 gen_whole.py --freeze none --fused m8 --l25-quot exact --weakshear CAPTURED.cu BASE.gen.cu > gen-info.json
bash integrated/compose.sh BASE.gen.cu OUT auto        # BASE d2d0319a, OUT/mynn_whole.gen.cu 793084cd at 1e-4
P3_MYNN=<harness binary> NVRTC=<libnvrtc.so.13> bash weakshear/bimg.sh OUT 0      # the tangent
P3_MYNN=<harness binary> NVRTC=<libnvrtc.so.13> bash weakshear/bimg.sh OUT 1,2    # the taped reverse and the count
bash weakshear/mkbundle.sh BASE_BUNDLE OUT_BUNDLE IMAGE_DIR                        # a scheme bundle with the three
```

A reverse or count build takes 39 to 43 GB of host memory and about 18 minutes on processors; one at a time.
`kit-bl-weakshear-849608c8.SHA256SUMS` lists the kit the shipped images are: tangent a90522ce, taped reverse
849608c8, count 4c0bd4d6, source 793084cd. The builder does not give the same image bytes for the same source, so an
image is judged by its outputs, not its bytes; the source is byte-equal (`integrated/test_weakshear.sh`, part 4). Two
include files of the kit (`mynn_tape.cuh`, `mw_m8_predict.cuh`) are an earlier copy of the ones `compose.sh` writes
here; they differ only by a precision option that is off by default and then gives the same tokens. A taped reverse
built with this tree's own include files has not been built.

## What was shown, and what was not

Shown, all on the engine's own route:

- 3 km, 12 steps, a recording of another storm day: the gradient read 1.0031 of the real model's change along a
  box-wide low-level wind nudge (1.0057 with nothing held and no rule, 1.3674 with the closure held); six other
  directions unmoved.
- 3 km, 59 steps: the one-column runaway is gone (-1.8e-08 at the start word where the unguarded form reads
  +3.6e-02), and the tangent-against-reverse consistency check reads 5.5e-10 in a seeded field direction and 6.8e-07
  in the storm direction with the cloud images of the builds before the refined ones.
- 12 km cut, 15 steps: the consistency check stays under the 1e-07 bar with three cloud image sets and three seeds
  (5.0e-08, 1.6e-08, 5.3e-09, 6.0e-09, 3.2e-08), where the same form without the rule fails (1.4e-07, 1.5e-07).
- 12 km cut, four places, 2 and 12 steps: the tangent's wind answer stays within 0.033 of the unguarded form's at
  three of them; at a weak-shear place the small cross-wind response reads about 0.13 under it (0.835 against 0.964).

Not shown:

- 59 steps at 3 km with the refined, repaired cloud images.
- Other directions. On the 12 km cut over 15 minutes a near-surface measure's own descent direction read 0.67 to
  0.71 of the real change for surface wind and 0.35 to 0.44 for surface temperature.
- The rule on the tape-free route, on the `--mix-live` path (refused), other meshes, longer windows.
