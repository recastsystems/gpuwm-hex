# Limits of the differentiable engine in 0.6.0

Every limit here was measured, and each states where. Nothing on this page is a limit nobody has reproduced. When a
limit is lifted it leaves this page in the release that lifts it. [GUIDE.md](GUIDE.md) says what the tangent and
adjoint are; the README's *The differentiable engine* has every number with its run.

## Hardware

- **One card family.** The adjoint commands run only on NVIDIA cards of compute capability 12.0 (GeForce RTX 50
  series, RTX PRO Blackwell). On every other family they stop before any download and say so (one RTX 4090 was
  refused so). The derivative
  images, part of the kernel set and the example recordings are compiled for that family, and a recording replays
  only on the family it was recorded on. Forward forecasts are not limited this way (README, *Requirements*).
- **One card a gradient.** Splitting one gradient over several cards (`--cards`) is experimental: a split gradient
  has not been shown equal to the one-card gradient (on the 3 km box its wind part read 7.6 per cent off), and the
  command says so before it runs.
- **Card memory.** 28,700 MiB free for the 12 km example, 73,156 MiB for the 120 km example (80,000 MiB on its
  full-physics row), 96,600 MiB for the 3 km storm box. A card whose whole memory is smaller is refused before any
  download; free memory is checked after the row's mesh file (2 MB for the 12 km cut) and before the rest.
- **Host memory on the 12 km example.** 47,104 MiB available. The README command peaked at 41,022 MiB on the run
  `examples/differentiable/expected.json` gives (up to 44,613 MiB on a 512 GiB host) and the release check `adjoint-test --directions all --dot-directions 8` at 46,681 MiB (one RTX 5090 host).
- **Host memory on the 3 km storm box.** Every command states its need on this host before it downloads the
  example and stops in one sentence on a host with less: 258,057 MiB for a first `sensitivity` with its answer
  check on a large host. `adjoint-test` keeps every step boundary, in host memory up to 0.4 of what is available,
  so its need follows the host: it held 262,306 MiB and 254,885 MiB on its two card runs (512 GiB hosts), and
  420,360 MiB (450,866 MiB asked) on the run that read PASS at the row's five step sizes. On a
  host too small for that share the command gives the runner a smaller store in memory (`--bstore-ram-gib`, said
  in its output) and writes the rest to the run folder, so it runs from 102,948 MiB available at 59 steps, with
  up to about 587 GiB in the run folder, and is refused under that. On a 499 GiB host (2026-11-16, the 0.6.0
  candidate) it stated 303,410 MiB, kept 19 of 60 step boundaries in host memory, held 278,824 MiB and read PASS
  with the same digits. It has not been run on a smaller host, and the time the extra files cost there is not
  measured. A first `sensitivity --steps 59` with its answer check on the same host held 240,043 MiB, its reverse
  sweep 191.3 s and 630.4 s in all.

## What the linear model is of

- **Exact mode's boundary layer, not the default forecast's.** The derivative is of MYNN-EDMF c6d08709, the
  version NOAA's RRFS-MPAS 44c0849 pins, and of the MYNN surface layer v6.0.0: what `--physics-mode exact` runs.
  A plain `hrrrv5` forecast runs correct mode, with MYNN-EDMF v6.1.1 and the surface layer v6.1.0; a gradient of
  that forecast is refused before anything is downloaded, and the derivative of v6.1.1 is planned for the next
  build. c6d08709 leaves the latent
  heat of the scheme's own cloud mixing out of the temperature tendency: on one 3 km state of 38,278 columns that
  term is over 0.1 K/h in 8.2 per cent of the columns and over 1 K/h in 1.6 per cent. In cloud-topped boundary
  layers a sensitivity is that of the older scheme. The surface layers differ only at a run's first step.
- **Held pieces** (derivative taken as zero; each receipt names them in `row_what`): on the 12 km and 3 km rows the
  mixing length and the plume's own speed, area and condensation; the transport limiter's coefficients on the 12 km
  row and the 120 km default row, and on the 3 km row only on windows longer than 30 minutes (on the 3 km default
  route the limiter is live, its exact derivative, on windows up to 30 minutes, the example's 59 steps among them;
  a longer window holds it, runs the `gated` cloud route and says so); on the 120 km default row also radiation and
  cloud fraction (64 kernel kinds) and the boundary layer's turbulence closure and plume; the mountain-drag closure on both 120 km rows (the full-physics
  row differentiates the limiter and the plume, the plume's speed on a ramp). The lateral boundaries of a limited-area window are held at the recorded values.
- **The 3 km default cloud route is `number`.** It holds the snow riming ratio where riming is slow and the riming
  fall-speed boost where a trace of cloud water would carry it to its cap, lets rain drop number and cloud droplet
  number follow the forecast, and holds snow and cloud-ice deposition only where it would add the species' own mass
  within 200 s. `--cloud-derivative gated` holds both numbers at their forecast values; `--cloud-derivative shipped`
  runs the derivative of the builds before.
- **Number concentrations.** Ice, rain, cloud-droplet and graupel number carry no tangent at the window's end
  (measured on the 12 km cut at 2 and 15 steps, along two directions, where they had values). The aerosol numbers
  and graupel volume carry tangent and pass. A sensitivity to those four numbers reads zero.
- **No cumulus derivative.** No example recording has a cumulus scheme in it; on the 12 km row a plan recorded with
  New Tiedtke on runs its 26 launch kinds as identity in both sweeps, named in `window.json`.
- **Composite reflectivity** is not a measure the gradient can follow (the forecast itself jumps); the commands
  refuse it by name.

## What the checks show, and what they do not

- **Near-ground moisture on the 12 km example is about 2 to 4 per cent short.** At ten step sizes of both signs the
  real change read 1.010 to 1.049 of the gradient's prediction (centred, 1.021 to 1.036: outside the 0.01 bar,
  2.6e-02 from 1, when read on centred differences). The cause is the
  boundary-layer plume's own speed and condensation, which the default route holds; with them live (a form not in
  this release) the centred readings were 0.998 to 1.013. The first command's five step sizes print this direction
  UNRESOLVED and its PASS does not cover it; the command names this limit under its directions.
- **The 120 km example tests on a 1-step window.** Its `adjoint-test` runs 1 step (12 minutes) on eight step
  sizes a factor of two apart and passes (the bump and near-ground moisture inside the bar); the steepest descent
  (1.008 to 1.047 of the real change) and the smooth draw stay UNRESOLVED. Over the row's 3 steps (36 minutes) only the
  bump resolves (`--steps 3` ends NOT PASSED (UNRESOLVED)), because the real model's change per unit step moves by 1 to
  6 per cent between neighbouring sizes. **Over 2 steps (24 minutes) the steepest descent reads OUTSIDE the bar:** it
  settles at 0.977 to 0.987 of the real change at the four smallest sizes (1.74e-02 from 1), so `--steps 2` fails;
  centred (each size at both signs) it reads 0.986 to 0.992 there. A gap of about 1 per cent along minus the
  gradient sits under the 1-step window's roughness and is not yet explained;
  at 2 steps the bump reads inside the bar and near-ground moisture is unresolved (0.944 to 0.980).
- **The 120 km full-physics row's own boundary-layer images fail its adjoint-test along the steepest descent**
  (`--boundary-layer taped`; the default, `floored`, passes: descent 1.4e-03 from 1, rc 0). On the same 1-step window
  (`--row hrrrv5-full-x1.40962 --boundary-layer taped`) the checks pass, the whole-state dot reads 1.0e-10 and the bump and near-ground
  moisture are inside the bar, but descent reads OUTSIDE (5.84e-02 from 1): 0.925 to 0.984 of the real change
  one-sided, and centred (each size at both signs, which cancels the curvature) 0.949 and 0.954 at the two largest
  sizes before the forecast's roughness takes over (0.971 to 1.084). The transpose is exact, so the tangent misses
  about 5 per cent of the real response along this direction. The command exits 1 and names this limit. The term
  is the boundary layer's live mixing (the diffusivities follow the turbulence speed): with the default row's
  boundary-layer images (mixing and plume held) the same window and direction read 1.030 and 1.035 centred, the
  default row's land derivative or the limiter held change nothing, and the closure-live images of the 3 km and
  12 km default route read 0.832 and 0.836. The gap sits in six coastal desert columns of West Africa (13 to 25 N,
  12 to 16 W) at the two lowest model levels, where the tangent moves theta_m by 7 to 31 K per unit step and the
  real model by -8 to +3 K. Its cause is the turbulence-energy path of the diffusivities (dK = K dq/q, dq = dqke/2q),
  unbounded where the turbulence speed q goes to zero. The default floors it with the factor q*q/(q*q + 0.5*0.5) of
  the forecast's own q: on the same window and direction the centred readings go to 0.995 and 1.000 (a floor of
  0.1 m/s gives 0.954 and 0.959), and the row's adjoint-test passes, exit status 0.
- **The window's consistency on further seeded directions, 12 km, 15 steps.** On the default route of this build seeds 1
  to 8 read +0.63, +0.33, -0.87, -0.25, +0.96, -0.34, +0.11 and +2.26 single-precision roundings of the dot
  (7.49e-08, 3.98e-08, 1.03e-07, 3.03e-08, 1.14e-07, 4.00e-08, 1.34e-08, 2.69e-07), each inside the bar of 4, with
  no one-signed drift: 5 of 8 positive, mean +0.35, standard error 0.34 (`adjoint-test --dot-directions 8`, one
  window, one RTX 5090, exit status 0; seeds 3 and 4 read the same through `--directions all --seed N`, and seed 1
  through the README command). Before the 12 km row's boundary-layer species hold, seed 3 read 7.16e-07 to
  9.80e-07, over the bar. On the exact route seeds 1 to 3 pass and seed 4 reads 1.74e-07.
  With the transport limiter's derivative live (not this row's default), twelve seeded directions read within 4 roundings: worst 3.38, mean -0.15, standard error 0.38, no one-signed lean.
- **Held on the 12 km row.** The boundary layer's seven species outputs (its tendencies of cloud water, cloud ice
  and snow, and of cloud-droplet, ice and the two aerosol numbers) are held in both sweeps: the part of a sensitivity
  that passes through the boundary layer's mixing of those species is zero. The cumulus scheme (New Tiedtke) is held by name: with
  its derivative live the 12 km window check misses by tens to hundreds of roundings, growing with the window. The
  12 km example's recorded forecast runs no cumulus scheme.
- **The exact route on the 12 km example fails** (`--boundary-layer taped`): the gradient's steepest descent reads
  0.954 to 0.959 of the real change and near-ground moisture 0.850 to 0.882. That is the held boundary layer.
- **3 km: `adjoint-test` ends PASS on the default route and FAIL on the exact route.** The
  default route on the storm example (number cloud route, live transport limiter, three double-arithmetic
  dynamics images): the window and whole-state dots passed (1.317e-08; 1.786e-07 relative, -1.50 roundings), the
  first window's finite-difference and Taylor checks read 3.653e-04 (bar 1e-02), and at the row's own step
  sizes (0.3, 0.2, 0.1, 0.06 and 0.03) the bump 0.9987 to 0.9996 (inside, 3.65e-04 from 1) and the steepest
  descent 0.9842 to 0.9981 (inside, 1.92e-03 from 1). At sizes 1 to 0.01 the descent read 0.9332 to 1.0044
  (unresolved): at 0.01 it moves the measure by only about 5 single-precision spacings of it. The exact route: the dots passed (2.825e-08, 1.313e-08), the finite-difference and
  Taylor checks read 1.017e-02, the bump 1.0102 to 1.0133 (unresolved), and the steepest descent 0.873 to 0.971
  (outside, 2.86e-02 from 1). Each was read once, on one 96 GB card. Two directions (the bump and the descent) are shown inside the bar
  at 3 km, on the default route only. In a storm direction (a 1 K warm bump under an updraft) the dot test reads 1.7e-06 to 2.2e-06 (14 to 18
  single-precision roundings of the dot), over the bar of 4.
- **Storm measures (rain water, cloud water, vertical velocity, and any measure placed on a storm cell) are not
  covered by any PASS of this release.** The PASSes above are for the README's measures (low-level temperature and moisture averaged over circles of 100 to 150 km radius).
  Read with `adjoint-test --metric` on measures placed on the storm itself, on the sealed wheel, 16 Nov:
  - 12 km example, 15 steps, a 60 km circle on the radar's 58 dBZ band at 41.85N 92.92W. **Rain water (`qr`,
    levels 1-20) fails** the first window's finite-difference check: 0.118 against 0.01 (exit 1). Read again at
    twelve step sizes of both signs (1 to 0.003), the real model's change carries an offset of about -3.4e-08 kg/kg
    (0.03 per cent of the measure) at every small step of either sign, so the one-sided ratios the command prints
    scatter from 0.40 to 2.93 at steps under 0.03. Read centred (each size at both signs, which cancels the offset
    and the curvature), the bump is 1.036 and 1.038 of the gradient's prediction at steps 0.1 and 0.03, 1.016 at
    0.01 and 0.99 at 0.003; at 1 and 0.3 it is 0.83 to 0.85, where rain answers a 1 K bump nonlinearly.
    **Updraft (`w`, levels 12-30) is NOT PASSED (UNRESOLVED):** the bump 0.877 to 1.009 and the steepest descent
    0.972 to 1.052 over steps 1 to 0.01; its first-window checks pass (7.2e-03).
  - 3 km example, 59 steps, a 40 km circle on a 66 dBZ cell at 39.85N 91.11W. **Updraft (`w`, levels 12-30) fails**
    along the steepest descent: 1.008 to 1.023 of the gradient's prediction (1.8e-02 from 1, bar 1e-02); the bump
    reads 0.953 to 0.998 (unresolved); the first-window checks pass (2.0e-03). **Low-level temperature (`theta_m`,
    levels 1-10) on the same cell fails too:** the first window's finite-difference check 0.245 against 0.01, the
    steepest descent 1.022 to 1.025 of the gradient's prediction (outside), the bump unresolved (0.60 to 1.68). So a
    temperature measure is covered only as the README poses it, not on a storm cell. Rain water was not read through
    the command at 3 km.
  - The transpose holds on every one of these windows (window dot 2e-08 to 2.5e-06, bar 2e-05): the backward code is
    the transpose of the tangent. What is not shown is that the tangent follows the real model's rain and updraft
    to 1 per cent. On the 12 km row the cloud scheme's derivative is the earlier builds' form (rain and droplet
    number held) and the transport limiter is held; the 3 km row runs the `number` cloud route with the limiter
    live. A map of a storm measure from this release shows where and with what sign the measure responds; its size
    is not shown to 1 per cent. `sensitivity`, `tangent` and `adjoint-test` print the row's measured line before
    they run when a measure is `qr`, `qc` or `w` (on a mesh named with `--mesh`, that it has not been read there).
- **On the 12 km example the whole-state dot is scored on memory allocations, not the model's fields.** The
  example's plan holds 15 step boundaries, so its 15-step window end has no field map; every 12 km command scores
  the dot on the window's memory allocations (134 of them) and prints "scored on memory allocations, not the model's
  fields" with the reason. The 3 km example scores the 12 model fields. Scoring the 12 km example on its fields needs
  the row recorded one step longer, which is planned for 0.6.1.
- **The limiter's own scratch words.** With the transport limiter live, the tangent and the adjoint disagree on
  the limiter's scratch memory (relative 5.85e-06, measured on the engine before this release); the model's fields
  pass on every window measured. The shipped checks are not affected: the 12 km and 120 km default rows hold the
  limiter, the 3 km example scores its fields (-1.50 roundings), and the 120 km full-physics row leaves that work
  memory out of the dot and proves on its plan that it is never read. A check you score on memory allocations
  (`AD_DOT_SELF_UNITS=pool`) on a window with the limiter live may read it. The fix is planned for 0.6.1.
- **On NOAA's deck the cloud scheme's boundary-layer subgrid radii are not in the shipped cloud images.** The
  runner can take that launch down the taped route, but only with a cloud image built with it, and the images this
  release ships are not: a plan you record on NOAA's hrrrv5 deck (with `config_tempo_ml_for_bl_nc`) is refused at
  the first launch that runs the path, naming the image's missing part. The shipped example plans do not run it.
  Planned for 0.6.1.
- **The per-species check misses on rain at 12 km.** The runner's per-species form (`AD_DOT_PLANES=scalars`, [CHECKS.md](CHECKS.md)),
  read on the 12 km cut over 14 steps of 60 s with the transport limiter held: the rain plane's own dot is off by
  -2.2e-05 of itself (-42.9 roundings) and -1.7e-05 (-33.7) at two sizes on test direction 1, and +8.0e-06 (+18.8)
  on direction 2, against the bar of 4; on the same windows snow read +1.5 to +7.1 roundings and graupel -3.9 to
  -5.5. So the per-species check does not pass on the 12 km example in this release. The commands' whole-state dot
  scores the species together and does not run this form.
- **The whole-state dot test is of one form.** Its start direction is scaled word by word by the state, so words
  that are zero in the start state are not perturbed, and its end weights are the tangent's own result. A test with
  an independent random end weight and an additive start direction is not run by this release.
- **Consistency is not accuracy: a window is good to about 3 hours at 120 km and 60 km.** Measured as the share of
  the real model's change that the tangent-linear model explains (all levels, 0.7 as the line). At 120 km, on three
  dates, with the mountain-drag closure held as the default row holds it: at 3 hours temperature 0.78 to 0.82, wind
  0.90 to 0.91, moisture 0.76 to 0.79; at 6 hours temperature 0.61 to 0.64 and moisture 0.54 to 0.59. With the
  closure differentiated (`param-grad`, rows made with `--suite hrrrv5-exact`) about 1 hour. At 60 km: temperature
  0.65 and 0.87 in two places at 2 h 54 min, and 0.63 and 0.81 on two dates at 3 hours, so in one place of two the
  temperature is already under the line near 3 hours; at 6 hours 0.15 and 0.53. At 3 km: 14 min 45 s for
  temperature after a low-level warm bump, and not for moisture at any window tried. Moisture and vertical-wind
  sensitivities at 3 km are not validated. Six hours is not claimed on any mesh. The shipped 120 km example is three
  steps (36 minutes); the 3-hour figures are measurements of the engine on longer recordings, not a command of this
  release.
- **Near-ground wind on the exact route** reads about 1.37 times the real change along a box-wide low-level wind
  nudge (3 km, 12 steps): the held boundary-layer mixing. The default route differentiates the closure but has not
  been run along that nudge through the command. `--boundary-layer guarded` read 1.003 along it over 12 steps; over
  59 steps it has run through the command on a 96 GB card (answer check 5.34e-08 against 1e-07, no one-word spike,
  1,478 s) and has not been compared with the real model. It stays a named option: the exact route still holds the
  boundary layer.

## Recordings

- A plan recorded with `--init` has not passed the strict check; the receipt runs it.
- A plan replays only on the card family that recorded it.
- The 6 km gradient route (`--gradient-mesh 6km`) is shown on one storm, one measure and one window only; any other
  is refused by name. **No example this release ships runs it:** the 6 km plan that ships is of 2024-05-21 15Z, the
  3 km storm example is of 2023-03-31 18Z, and on the example's own day three of the route's four readings were
  under its bar of 0.99 (cosine with the 3 km gradient 0.992 for temperature and 0.986 for moisture on one measure,
  0.80 and 0.96 on a second), so there it stays refused. It runs only with your own 3 km recording of
  2024-05-21 15Z given with `--plan`.

## The variational interface

- `libhexad-phys` (the C library) ships with its header, but no example in this release drives it end to end.
- One session is one recorded trajectory: a new outer loop records a new plan.
- None of an assimilation system ships: no observation operators, no background-error model, no minimiser.
