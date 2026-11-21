# gpuwm-hex

MPAS-Atmosphere v8.4.2 and NOAA's hrrrv5 (RRFS-MPAS) configuration on NVIDIA GPUs, with the tangent-linear and
adjoint of the exact-mode hrrrv5 forecast. gpuwm-hex 0.6.2 ports the whole model to
CUDA: the dynamical core, both MPAS physics suites and their scheme options,
initialization and static fields, the incremental analysis update, limited-area
runs, streams, restart and diagnostics (the scheme list and what is not yet in
are in `tree/README.md`). MPAS's own physics is the default; ArWen's physics
stays one flag away (`--physics-suite arwen_mpas_column`), and hex adds
capabilities MPAS does not have: one-command global forecasts, local time
stepping, an experimental storm-following region, and a tangent-linear and
adjoint model of its own forecast step. Forecasts run on one consumer CUDA card
(larger meshes take more memory or more cards, as `tree/README.md` says); the
12 km adjoint example runs on a 32 GB card, and the 3 km and 120 km adjoint
examples need a 96 GB card.

This is the repository. The distribution it builds lives in [`tree/`](tree/),
and its user documentation is [`tree/README.md`](tree/README.md) (the
capability contract: every claim with its scope) and the
[user manual](tree/docs/manual/index.md). This page says what the project is,
what it is matched against and how that is checked.

**Install:** `pip install "gpuwm-hex[gpu]"` for forecasts, then `gpuwm-hex doctor`
to see what the install can reach, and `gpuwm fetch-geog` once for the geography
every static build needs, `gpuwm-hex global` included (about 2.5 GiB to
download, about 31 GiB unpacked). For the derivative commands install
`pip install "gpuwm-hex[ad]"` instead, in a Python 3.14 environment of its own
(not on top of `[gpu]`). Either way pip resolves the engine it requires, gpuwm 3.0
(`gpuwm>=3.0.0,<3.1`).

---

## What 0.6.2 claims, and how far

Each line is measured, and each carries its scope. Nothing outside the scope
is claimed. [`tree/README.md`](tree/README.md) gives the detail.

Kernels compile with fast numerics by default (FMA contraction, flush-to-zero); `--numerics strict` is the opt-in for bitwise identity to native MPAS-A, and every identity claim here was measured with it.

| Claim | Scope |
|---|---|
| The dynamical core is byte-identical to native MPAS-A v8.4.2 built with the pinned strict recipe (GNU Fortran, single precision, no FP contraction or vectorised reassociation). | `x1.40962`, 55 levels, real terrain, dry, 5 days, two dates; the same mesh with water vapour and with five condensates transported, on 1, 2 and 4 ranks of one card; `x1.163842` with water vapour; one dry step on `x1.2562`. Hex and the reference run on the same host C library. |
| A 24-hour forecast of the `convection_permitting` suite with all ten of its physics schemes on the GPU is bitwise identical to native MPAS-A v8.4.2: every field native writes in this forecast, hex writes too, 650,457,072 values in 13,938 written fields, 0 different. | `x1.2562`, one process, 2023-03-31 12Z, exact mode, strict numerics. |
| Every other MPAS-A v8.4.2 physics scheme in this release is bitwise identical to native on real inputs captured from a native run, and a forecast with it on was compared with native, map by map. | Per scheme. A whole-forecast match is claimed only for the forecast above. |
| Correct mode is the default on every route. On the MPAS suites it fixes 5 defects found in native MPAS-A v8.4.2; on the ArWen route it applies the corrections the engine carries (WB4, WB5); `--physics-mode exact` reproduces native bit for bit. | Each fix is checked against its own native referee build. No combined referee is claimed. |
| Real-data initialization and the static builder match native field for field. IAU, limited-area boundaries, streams, restart and the diagnostics match native on real captured inputs. | Per unit, on the cases in `tree/README.md`. A few static words where native reads undefined memory are labelled declared residue, never identical. |
| `gpuwm-hex forecast --analysis` starts a forecast from an MPAS analysis bundle (direct insertion, IAU, cycling restart, remap). | Measured on the ArWen physics route. IAU also runs across a restart, on the explicit MPAS suite route, on same-host multi-card ranks and in member rosters; IAU on multi-card local time stepping is refused by name. |
| `gpuwm-hex global` runs a global forecast in one command. | Rows are defined at 120, 60, 30 and 15 km, on ArWen physics and on the Morrison suite (Morrison, Grell-Freitas, YSU, revised MM5, Noah, RRTMGP); the measured runs are the 120 km rows. |
| A storm-following 3 km region, experimental: `gpuwm-hex moving`. | ArWen physics, three 48-hour tropical cyclone runs; the balanced remap between moves is the default. |
| RRFS-MPAS runs end to end from a pip install: the GSL fork's dycore and table-admitted physics run on the GPU by default. `hrrrv5`, `rrfs_convection_permitting`, `rrfs_mesoscale_reference`, and RRFS New Tiedtke use the ported framework without a native library. Native comparison backends remain explicit site builds. | With every physics slot on the native library (`--physics-default-backend native-cpu`), a one-hour `hrrrv5` run on `x1.2562`, one process, is bitwise to the native fork build in every field hex writes; each GPU port is bitwise as a module replay. RRFS initial conditions and hourly boundaries come from `gpuwm-hex init --physics-suite hrrrv5`, the fork's init ported to Rust ([RRFS init](tree/docs/rrfs-init.md)). |
| A whole `hrrrv5` (RRFS-MPAS) forecast with every physics slot on the GPU is bitwise identical to the native fork build in every history, diagnostics and restart word hex writes, at 8 ranks and on one card. | `x1.40962` 120 km, 59 levels, 2023-03-31 12Z, 24 h, strict numerics: 8 same-host ranks on one card against 8 MPI ranks on the same partition, 1,958,644,100 words, 0 different; one card against one rank, 786,309,080 words, 0 different. The reference is the pinned 44c0849 strict build with four labelled source lines where native reads values it never set; it is rebuildable and always called patched (`tree/README.md`). |
| A plain `--physics-suite hrrrv5` forecast is NOAA's hrrrv5 deck (RRFS v2's configuration), and its one-hour forecast on a limited-area mesh, exact mode with strict numerics, is bitwise identical to the native fork build in every history, diagnostics and restart word hex writes. | HRRR 2024-05-21 15Z start, one process on one RTX 5090, once each, through the plain door: a 12 km cut of 2,890 cells with NOAA's 12 km cumulus on, 0 of 86,751,014 words different in 27 of 27 files; a 3 km storm box of 38,278 cells, 0 of 624,931,668 words in 11 of 11. The same 12 km hour on one RTX 4090: 0 of 86,751,014. The reference is the patched 44c0849 strict build, as above. |
| Native MPAS restart files restart the dynamics on every MPAS physics suite. With `--numerics strict` a restart reproduces native's own restart; on the default route it continues the run exactly, which native's restart does not. | Strict: 0 of 540,860,379 words against native restarted from its own file (`x1.40962`, one card). Default: 0 of 1,208,871,000 words against the uninterrupted run (`x1.40962`, 8 ranks) and 0 of 452,696,265 (`x1.163842`, 8 ranks, restarted at +24 h). |
| The tangent-linear and adjoint model of NOAA's exact hrrrv5 configuration (the exact-mode forecast step, `--physics-mode exact`; not the correct-mode forecast `gpuwm-hex forecast` runs by default, whose gradient is refused in one sentence), dynamics and physics, on the GPU, in four commands: `gpuwm-hex tangent`, `adjoint-test`, `sensitivity` and `param-grad`. `adjoint-test` checks the adjoint against the tangent-linear model (within 4 single-precision roundings of the dot on every direction, with no one-signed lean over 8 or more) and the gradient against the forecast's own response (within 1 per cent along each resolved direction), and prints PASS or FAIL; the readings of this release are in [the limits](tree/docs/differentiable/LIMITS.md). | A 12 km cut of the storm box (2,890 cells, 15 steps) that runs on a 32 GB card, a 3 km storm-box example (38,278 cells, a 15-minute window of 59 steps) and a 120 km global example (`x1.40962`, `hrrrv5`) of three steps (36 minutes), the last two on a 96 GB card; NVIDIA cards of compute capability 12.0 only (GeForce RTX 50 series, RTX PRO Blackwell), Python 3.14. On the 12 km example `adjoint-test` ends PASS (the bump and the steepest descent inside; near-ground moisture named as about 2 to 4 per cent short); on the 3 km example it ends PASS on its default route (the bump and the steepest descent inside) and FAIL on the exact route; on the 120 km rows, over a 1-step window, PASS. The checks are consistency and short-window checks: how far in time the gradient predicts the forecast was measured separately, by mesh and window, in the table "How far the gradient can be trusted" of [the differentiable engine](tree/docs/differentiable/README.md). A measure built on radar reflectivity does not pass the gradient check and is refused by name. Held by name: on the 12 km row the boundary layer's seven species outputs and the cumulus scheme; on the 12 km row and the 120 km default row the transport limiter's coefficients (the 3 km row differentiates the limiter on windows up to 30 minutes, the 120 km full-physics row on its window); on the 12 km and 3 km rows the boundary layer's mixing length and the plume's own speed, area and condensation, the lateral-boundary driving data (zero tangent) and, on the 3 km row's default cloud route, two riming terms; on the 120 km default row the boundary layer's turbulence closure and whole plume, and radiation and cloud fraction (the full-physics row differentiates the plume, its speed on a ramp); on both 120 km rows the mountain-drag closure. The 120 km full-physics row runs a floored boundary layer by default. [What each row differentiates](tree/docs/differentiable/README.md). |

Named gaps, each with its reason, are listed in
[`tree/README.md`](tree/README.md#not-in-this-release).

---

## The two relationships that define this project

### To native MPAS-Atmosphere v8.4.2: matched wherever a match is reachable

The referee for anything that ports MPAS is native MPAS itself: the pinned
v8.4.2 source (commit `f34984b2`) built with a strict GNU recipe, run on the
same inputs, compared word for word. "Identical" here always means every
stored word, with no tolerance. Where hex cannot match (a value native reads
from undefined memory, a field hex does not write yet), the gap is counted and
named, never averaged away.

The reference is rebuildable from this repository:
[`tree/tools/native_dry_reference/`](tree/tools/native_dry_reference/) builds the
dycore reference, [`tree/tools/native_physics_reference/`](tree/tools/native_physics_reference/)
the physics reference, and [`tree/verification/gates/`](tree/verification/gates/)
holds the gate decks every change to the port must pass before it lands: the
dry dycore, both MPAS suites on the native backend, and the all-GPU forecast
above. A claim whose reference cannot be rebuilt is not made.

Native MPAS has defects of its own. Matching it exactly found real ones.
**Correct mode**, the default on every route, fixes the ones this release
carries; `--physics-mode exact` puts every one of them back. A fix that a
scheme does not carry yet is never a refusal: the run keeps native behaviour
for it and names the gap at start-up and in its receipt.

**ArWen's physics is judged by observations, not by MPAS.** It is a different
physics implementation, so matching MPAS was never its standard; its referee
is skill against MRMS, Stage-IV, ASOS and radiosondes, and its measured
differences from native MPAS are registered in
[`tree/docs/declared-divergences.md`](tree/docs/declared-divergences.md). A
bias that is wrong against observations is still wrong.

### To `gpuwm` (ArWen): a one-way dependency

- **The MPAS physics ports**, the default physics since 0.4.6, are this
  project's own (`tree/src/hexcore/mpasphys/`). They compile with device maths
  headers read from the installed `gpuwm` at run time.
- **ArWen's physics** (`--physics-suite arwen_mpas_column`) is reached through
  the engine's column-batch seam, so gpuwm-hex and `gpuwm` share one
  implementation of it. Sixteen engine files are pinned by SHA-256;
  `gpuwm-hex doctor` reports whether the installed engine still matches.
- **The Rust field engines** (`rw_mpas_init`, `rw_mpas_static`,
  `rw_mpas_convert`, `rw_mpas_mesh`, `rw_mpas_lbc`, `rw_wrfbatch` and the
  others) are built in `gpuwm`'s Rust workspace and reach a user through its
  bridge bundle (`gpuwm fetch-bridges`). Several of them, among them
  `rw_mpas_geometry`, `rw_mpas_stream` and the mesh generator `rw_mpas_mesh`,
  ship in the gpuwm-hex wheel instead, pinned by SHA-256, because no released
  gpuwm carries the versions this release needs (`tree/README.md`,
  Requirements, lists them). `gpuwm-hex mesh-plan` makes a mesh with it: the
  3 km CONUS globe, 1,959,938 cells, grid and static in 110 s on a 24-core
  host ([mesh rows](tree/docs/mesh-rows.md)).
- `gpuwm` does not import `hexcore`, so the two can be separate repositories.

---

## Layout

```
LICENSE          Apache-2.0
NOTICE           every upstream, its terms and what derives from it
README.md        this page
.github/         the CPU-tier workflow and the publish workflow
tree/            the gpuwm-hex distribution
  src/hexcore/     the port (`import hexcore`)
    mpasphys/        the MPAS-A v8.4.2 physics driver and scheme ports
  tools/           native reference recipes, gate runner, comparators
  tests/           the three-tier battery
  verification/    gate decks, schemas, manifests and vertical specs
  evidence/        see EVIDENCE.md
  docs/            the manual and the door pages
  licenses/        every upstream licence text
```

The measurements quoted in `tree/README.md` that carry an `evidence/...` reference name the receipt they came from.
The receipts themselves are not carried here; `tree/evidence/EVIDENCE.md` says
why and what an `evidence/...` reference means.

## Running the battery

Three tiers, split by what a machine has to own. Each names what it is missing
rather than failing somewhere unrecognisable. Tier 1 needs nothing but Python:

```sh
cd tree
PYTHONPATH=src python -m pytest -q -m "not gpu and not bigcard and not assets" \
    $(grep -v '^#' tools/battery/cpu_files.txt | grep -v '^$')
```

`tools/battery/README.md` covers all three. `GPUWM_HEX_NO_LOCAL_GPU=1` bans
device contact outright. The gate decks need the native references and a
card; `tree/verification/gates/README.md` says how to lay them out and run
them.

## Licence

Apache-2.0, the same licence `gpuwm` ships under, except where an upstream's
own terms govern a ported portion. MPAS-Atmosphere (LANS and UCAR) is
BSD-3-Clause, and its terms travel with every copy. The physics ports also
carry NCAR MMM-physics (BSD-3-Clause), WRF physics as MPAS distributes it (the
WRF public-domain notice) and RRTMG (AER, BSD-3-Clause), and the RRFS stage
ports NOAA GSL physics (UGWP, MYNN-EDMF, MYNN-SFC, RUC, GFL, smoke), which the
project treats as works of the US Government. `NOTICE` names every upstream and its terms,
`tree/THIRD-PARTY-NOTICES.txt` lists the derived files, and `tree/licenses/`
holds the texts, including those of the Rust crates the wheel's compiled
helpers link; all required permissive texts ship in the wheel and sdist.

The UCAR Noah-MP v5.0 port is withheld from this public export: UCAR's
terms (**not** Apache-2.0) forbid selling it or a work that contains it. Its
upstream pin and attribution remain in `NOTICE` section 4. The public wheel
contains no Fortran runtime. An opt-in cannot restore an excluded port. The
Noah-MP that hex runs is the ArWen route's separate WRF-lineage v4.x, which is not
MPAS-bitwise.

This is **not** the version available from LANS and UCAR, and neither they nor
their contributors endorse it. Results from gpuwm-hex are not results from
MPAS-Atmosphere.
