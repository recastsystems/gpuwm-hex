# 11. New in 0.6.2

Each section says what the feature does, whether it is on by default, the one switch that changes it, and what has
been shown. The [CHANGELOG](../../CHANGELOG.md) lists every change, including the fixes; this chapter covers the
features that change how a run is set up.

## 11.1 Generated meshes use geometry 2

Every door that generates a mesh (for example `mesh-plan --generate`, `country` and the zoom globes) now calls
the mesh generator `rw_mpas_mesh` with geometry 2 unless the request names another. Geometry 2 grows the cell
spacing at one rate per cell outward from each fine zone, builds every spacing that was asked for, rather than
stepping down a halving ladder, and produces no mesh with an obtuse triangle. For the same request it gives a few per
cent more columns; the plan the door prints counts them, so look at the plan before the run.

- `--geometry 1`, or `"generator": {"geometry": 1}` in a mesh row, builds the halving ladder and the same bytes as
  0.6.1.
- A registered country row or region keeps the mesh it was made with.
- The generator uses at most 32 threads when `RAYON_NUM_THREADS` is not set. `RW_MPAS_MESH_MAX_THREADS` changes that
  cap, and `RAYON_NUM_THREADS` overrides it. The mesh written does not depend on the thread count.

The generator is a new engine bridge. It ships inside the wheel, and is published alone as the release
`engine-bridges-v0.6.2`.

## 11.2 Runs on several cards: seven speed settings, on by default

Seven settings speed up runs on several ranks. Each is on unless it is set to `0`. None of them changes a stored
value, so a run with all seven set to `0` writes the same bytes as before they existed.

| Setting | What it does |
|---|---|
| `GPUWM_HEX_LIMITER_BATCH` | The monotonic limiter handles 7 species per halo exchange, so 13 tracers take two exchanges a step instead of thirteen. A number sets the batch size. |
| `GPUWM_HEX_HALO_PARALLEL_RECV` | Each peer's halo receive runs on its own stream. |
| `GPUWM_HEX_HALO_TABLE_CACHE` | Halo descriptor tables are kept between steps and uploaded from pinned memory. |
| `GPUWM_HEX_PUBLISH_LEAN` | Host copies of the pools are downloaded only when something reads them. |
| `GPUWM_HEX_PUBLISH_FAST` | Those downloads are transposed on the card. |
| `GPUWM_HEX_RESTART_RELEASE` | A same-host resume frees the whole-globe copy once every rank has its share. |
| `GPUWM_HEX_DIRECT_SHARD_RESTORE` | A resume from rank shards, on the same ranks and partition, restores each rank from its own shard. Rank 0 no longer assembles the globe. |

Cold kernel compiles now size their worker pool by free host memory as well as by CPU count: 1 GiB per worker, within
half of the free memory. A cold compile on a busy host no longer takes 13 to 15 GB. `GPUWM_KERNEL_PLAN_WORKERS`
still sets the pool size directly.

## 11.3 Ranks on several hosts

```sh
gpuwm-hex forecast ... --physics-suite hrrrv5 --hosts hostA:0-7,hostB:0-7
```

`--hosts` runs one rank process for each listed card, across machines, on the MPAS-physics route. Halos are
exchanged through NCCL. The command runs on the first host, which is rank 0 and writes every output. Every other
host is started over `ssh` and must be able to read the same inputs at the same paths. Card lists are written
`a-b`, or joined with `+`.

- A checkpoint is sealed as rank shards on every host. A run resumes across the hosts from those shards, and a host
  that rejoins resumes from its own disk.
- A launcher that starts every host itself (a batch scheduler, for example) sets `GPUWM_HEX_MULTIHOST_EXTERNAL`,
  together with `MASTER_ADDR`, `MASTER_PORT` and `GPUWM_HEX_MULTIHOST_TOKEN`.
- These are off by default: `GPUWM_HEX_PER_RANK_OUTPUT=1`, where each rank writes its own cells, and
  `GPUWM_HEX_PER_RANK_START=1` with `GPUWM_HEX_START_SHARDS=DIR`, where no whole-globe start is held in memory. The
  per-rank start is refused by name on limited-area meshes.

## 11.4 Updraft helicity on several ranks

Updraft helicity from several same-host ranks is now the same, word for word, as from one card. Each rank sums a
seam cell's vertex terms in the order one card uses. Before this release the two differed in the last place at the
seams. `GPUWM_HEX_UH_RANK_ORDER=native` keeps native MPAS's per-rank order.

## 11.5 Hail size (HAILCAST)

HAILCAST runs by default wherever the mesh has cells finer than 4 km. The forecast writes HRRR's hail fields to its
diagnostics stream, under HRRR's names and in HRRR's units. Each is the largest value since the last whole-hour
frame, so the hourly frame holds HRRR's hourly maximum.

| Field | Unit | What |
|---|---|---|
| `HAILCAST_DIAM_MAX`, `HAILCAST_DIAM_MEAN`, `HAILCAST_DIAM_STD`, `HAILCAST_DHAIL1` to `HAILCAST_DHAIL5` | mm | HAILCAST's hailstone model (HRRR v4's by default) |
| `HAIL_MAXK1`, `HAIL_MAX2D` | m | the largest hail in the size distribution at the lowest level and anywhere in the column (TEMPO's own on the RRFS route; gpuwm's `hail_psd` with MPAS-A Thompson and WSM6) |

- `GPUWM_HEX_HAILCAST=0` turns it off, and `=1` runs it in every cell. `GPUWM_HEX_HAILCAST_VERSION=wrf_461` uses WRF
  4.6.1's hailstone model.
- The hailstone model comes from the installed gpuwm. If gpuwm's hail kernels are not the expected ones, the
  forecast logs one line and carries on without these fields.
- Runs on several cards, and runs resumed from a checkpoint, give the same fields as one card and an uninterrupted
  run.
- They are not in the default map set. `gpuwm-hex render --products hail_max_hailcast,hail_max_surface,hail_max_column`
  draws them with any gpuwm of the 3.0 line: the converter the wheel bundles, which the door runs before any gpuwm
  bridge, maps `HAILCAST_DIAM_MAX` in mm and `HAIL_MAXK1` / `HAIL_MAX2D` in m.
- It costs time where it runs: on a 3 km storm box over eastern Iowa (38,278 cells, one RTX PRO 6000) the step loop
  took 20.8 s a forecast hour with it and 18.2 to 18.4 s with `GPUWM_HEX_HAILCAST=0`.

## 11.6 Surface and boundary-layer closures on hrrrv5 (opt-in)

Three optional closures act on the RUC land surface and on MYNN-EDMF in the `hrrrv5` suite. All are off by default.
At most one can run at a time.

| Option | Door setting | Environment setting | Needs |
|---|---|---|---|
| Surface-flux network | `--physics config_surface_flux_net=r4_a50` | `GPUWM_HEX_SURFACE_FLUX_NET=r4_a50` | PyTorch |
| Neural boundary layer | `--physics config_neural_pbl=v1` (or `v2`) | `GPUWM_HEX_NEURAL_PBL=v1` | PyTorch |
| its dry-air gate (turns v1 into v2) | `--physics config_neural_pbl_dry_gate=true` | `GPUWM_HEX_NEURAL_PBL_DRY_GATE=1` | |
| 8-parameter closure | `--physics config_autophysics_pbl=v2`, `v3_bowen` or `v2_warm_night` | `GPUWM_HEX_AUTOPHYSICS_PBL=...` | CuPy only |

What each one changes:

- **The surface-flux network and the 8-parameter closure** act after each RUC land step. They shift the sensible
  and latent heat fluxes (and the moisture flux with them). For the surface diagnostics and MYNN-EDMF only, they also
  shift the skin temperature and surface humidity; that shift is undone before the next land step, so RUC's soil
  heat balance never carries it.
- **The neural boundary layer** does the same at the surface. After MYNN-EDMF it also adds a vertical
  redistribution of theta, qv, u and v over the lowest ten levels. The redistribution conserves every column's
  mass-weighted sums.
- **The 8-parameter closure** is eight fitted numbers in a closed-form law, with no network. `v3_bowen` keeps only
  the daytime Bowen-ratio law. `v2_warm_night` applies v2's nocturnal cooling on warm nights only.

The weights and parameter sets ship in the wheel (`hexcore/data/closures/`). Their SHA-256 sums are listed in
`closures.v1.json` and checked before the first step; a file whose sum differs is refused. The two networks need
PyTorch, from the `torch` extra. A run that names a network without PyTorch installed is refused by name.

Install `[torch]` in a Python 3.14 environment of its own, as the README asks for `[ad]`, not on top of `[gpu]`.
The `torch` extra brings the `ad` extra, and with it the CUDA 13 CuPy (`cupy-cuda13x`):

- it needs a driver that serves CUDA 13; on a driver that serves only CUDA 12 it stops at its first device call
  (`cudaErrorInsufficientDriver`);
- installed over a `[gpu]` environment it leaves two CuPy packages there (`cupy-cuda12x` and `cupy-cuda13x`), because
  pip does not remove one when it installs the other; `gpuwm-hex doctor` reports that as a gap.

```sh
python3.14 -m venv hextorch && . hextorch/bin/activate   # Debian and Ubuntu: sudo apt install python3-venv first
pip install "gpuwm-hex[torch]"
gpuwm-hex doctor
```

NOTICE section 20 says what each option was trained or fitted on.

Combinations that cannot work are refused by name at the door, before any step:

```
gpuwm-hex: autophysics_pbl v2 and surface_flux_net r4_a50 cannot run together: both shift RUC's land fluxes and skin temperature on the same slot. Turn one off
gpuwm-hex: config_neural_pbl_dry_gate (GPUWM_HEX_NEURAL_PBL_DRY_GATE) gates the neural boundary layer; this run has none (config_neural_pbl / GPUWM_HEX_NEURAL_PBL is off)
gpuwm-hex: config_autophysics_pbl='v9': not off or one of ['v2', 'v2_warm_night', 'v3_bowen'] (GPUWM_HEX_AUTOPHYSICS_PBL takes the same)
gpuwm-hex: --physics config_surface_flux_net=off and GPUWM_HEX_SURFACE_FLUX_NET=r4_a50 name two selections for one run; give one
```

A route without RUC land (or without MYNN-EDMF, for the neural boundary layer), a slot that is not on the CUDA
resident route, and the MPAS-A v8.4.2 suites are also refused by name.

What has been run on this release: on the published 120 km global mesh `x1.40962`, started from GFS 2026-11-19 12Z (built
with `gpuwm-hex init-source --source gfs --physics-suite hrrrv5`), 3 hours, one RTX PRO 6000, from a fresh install
of the wheel with an empty home directory:

- With every option off, the run was the same as 0.6.2 without these options: 0 of 558,064,324 stored words
  differed.
- Each of the six settings (`r4_a50`, `v1`, `v1` with the dry-air gate, `v2`, `v3_bowen`, `v2_warm_night`) ran to the
  end in about 20 seconds. The median change in land 2 m temperature at +3 h ranged from 0.07 K (`v3_bowen`) to
  0.69 K (neural boundary layer `v1`).
- The 8-parameter closure with its gain set to 0 gave 0 different words from the plain run, so the hooks change
  nothing by themselves.
- `GPUWM_HEX_AUTOPHYSICS_PBL=v2` and `--physics config_autophysics_pbl=v2` gave the same words.
- The largest changes, 8 to 9 K in 2 m temperature, were in a few cells far from the conditions the options were
  trained on: snow in the Arctic polar night, and the high mountains of Asia.

These options were developed and scored on 3 km runs over the United States. This release shows that each one is
wired correctly and runs. It does not claim forecast skill for them. Judge them against observations on your own
cases before using them.

## 11.7 A learned column correction (opt-in)

A trained network can add a correction to the model's tendencies at every step:

```sh
python3.14 -m venv hextorch && . hextorch/bin/activate
pip install "gpuwm-hex[torch]"
GPUWM_HEX_LEARNED_CORRECTION=/path/to/checkpoint.pt gpuwm-hex forecast ...
```

The `torch` extra goes in an environment of its own, for the reason in 11.6: it brings the CUDA 13 CuPy, which needs
a driver that serves CUDA 13 and, over a `[gpu]` environment, leaves two CuPy packages.

At each step, the network reads the step-start state (17 features per level, built from theta, water vapour,
pressure, the cell winds, their vertical gradients, the height above ground, the solar angle, latitude, terrain, the
land mask and the cell size). It returns four rates: theta in K/s, qv in kg/kg/s, and u and v in m/s2, at cells. The
model adds them through its external-tendency slot, before the edge projection and the coupling. The rates are
capped by tanh at 10 K/day, 5 g/kg/day and 15 m/s/day. They are applied at full weight below 150 hPa,
taper to zero at 100 hPa, and are zero in a limited-area mesh's boundary rows.

- **The checkpoint** is a PyTorch `state_dict` (or a dict with a `"model"` key holding one) for the network that
  `GPUWM_HEX_LEARNED_CORRECTION_ARCH` names: `PhysConvNetV2`, the default, or `ColumnNet`. Both are defined in
  `hexcore.ad.learned_correction`, built for the mesh's level count, and loaded with `weights_only=True`.
- `GPUWM_HEX_LEARNED_CORRECTION_SCALE` multiplies the rates (default 1).
- **No weights ship with the package.** A checkpoint you name is your own.
- It runs on one card only, and more than one rank is refused: the slot's wind terms on several cards have not yet
  been shown to match one card.
- With the setting unset, nothing is imported and the forecast is the plain model, bit for bit.

What has been run on this release: on the same case and card, from a fresh install:

- A network whose output weights are all zero gave the plain run's words: 0 of 558,064,324 different.
- A freshly initialised `PhysConvNetV2` ran to the end, with rates up to 3e-5 K/s, and gave the same words with
  `--dt auto` as without it.
- A missing file, an unknown network name and two ranks were each refused by name.
- It also ran together with the surface-flux network.

## 11.8 A start from ECMWF IFS with GFS above 50 hPa

`--source ifs-gfs-top` (on `init-source` and the doors that take a start row) starts from ECMWF IFS open data at
0.25 degrees below 50 hPa, and from NCEP GFS 0.25-degree data above 50 hPa. Sea ice comes from the GFS analysis, as
in the `ifs` row. ECMWF's open data has 13 pressure levels and stops at 50 hPa, below the model top, so the plain
`ifs` row extrapolates the column above that. This row fills the column from GFS instead. It is used only when it is
named. The ECMWF part is CC-BY-4.0 and the GFS part is US government work.

## 11.9 The derivative engine

The tangent-linear and adjoint commands gain a simplified inner model (`--inner-model simple`), a cloud-sample seed
rule for radiation (`--physics config_radt_mcica_seed=cell`), recordings as several rank threads on one card, the
guarded boundary layer on rows made for `--mesh`, and the external-tendency slot that 11.7 uses. The
[CHANGELOG](../../CHANGELOG.md) section *Derivative engine* lists each change, and the README section *The
differentiable engine* gives the commands and the checks.

## 11.10 Grell-Freitas convection with hrrrv5, on the GPU

`--physics config_convection_scheme=cu_grell_freitas` with `--physics-suite hrrrv5` runs MPAS-A's Grell-Freitas
scheme on the card, beside TEMPO, MYNN-EDMF, RUC and UGWP. It is the port MPAS-A's `convection_permitting` suite
uses; the RRFS fork compiles the same scheme file. The run's record names convection `cu_grell_freitas` on the
`hex-cuda` backend. This pairing has not been compared value for value with native RRFS-MPAS.

Run on this release from a fresh install (one RTX PRO 6000; one hour of the 10 km, 800 km region around central
Missouri, 35,043 cells). With hrrrv5 the region builds its own static and asks for a geography root that holds MODIS
15 s land use and LAI:

```sh
gpuwm fetch-geog --root ~/.local/share/gpuwm/WPS_GEOG --datasets lai_modis_30s,modis_landuse_20class_15s
gpuwm-hex regional --point 38.5,-93.0 --radius-km 800 --fine-dx-m 10000 --background-km 80 --source hrrr --cycle 2024-05-21T15 --hours 1 --geog ~/.local/share/gpuwm/WPS_GEOG --forecast-arg=--physics-suite=hrrrv5 --forecast-arg=--physics=config_convection_scheme=cu_grell_freitas
```

A convection choice with no GPU route beside hrrrv5's other schemes (`cu_tiedtke`, `cu_kain_fritsch`) stops at
the door, before any download (`gpuwm-hex regional` and `global` read it before they build the mesh), and
names the choices that run on the GPU with the run's other schemes, as `--physics` flags.
