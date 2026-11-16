#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""P3 build-time generator: whole-launch tangent and adjoint of hex's MYNN-EDMF c6d08709
launch `rrfs_mynnedmf_run_l16`, by type substitution of the exact captured forward source.

Build time only. Nothing here runs in a forecast, tangent or reverse loop.
Input : the captured NVRTC source of the forward image (sha256 pinned below, hex pin 16251589).
Output: one CUDA translation unit with three entries
          mynn_whole_tl     forward tangent (primal is the captured float arithmetic, tangent in double)
          mynn_whole_count  tape-sizing pass (node count per column)
          mynn_whole_ad     taped forward + reverse sweep (adjoint)
The primal routines are the captured ones (r6_*, mynn_d*, library wrappers), so the primal of each entry is
bitwise the forward's by construction; the derivative rule per routine is in mynn_ad_rules.inc.
Branch policy: exact executed branch. Every float/double comparison (min/max, thresholds, limiters) decides on
the primal value; the tangent and adjoint follow the arm the forward took. Loop and iteration counts are integer
control and replay exactly. Uninitialised arena words read by the forward carry no derivative.
"""
import hashlib, json, re, sys
from pathlib import Path

CAPTURE = '70da4af74192b47fef6dd002101d64a22659b38a93c496a7832ec49071689727'
TOP = 'f_mynnedmf_driver'

argv = sys.argv[1:]
MIX_LIVE_GIVEN = '--mix-live' in argv or '--bl-live' in argv
BL_LIVE_GIVEN = '--bl-live' in argv
# strike physics bl (2026-11-11): --bl-live NAME is ONE named option for a tested pair of live bounded forms of the two
# boundary-layer holds (the mixing closure and the plume's upward speed). It expands to the options below and to nothing else,
# so the output is byte for byte the one of the expanded command.
#   q1-ramp   --freeze closure-plume --plume-w-smooth ramp:0.5:1.5 --mix-live q:1
#             the diffusivities follow the turbulence speed whole (stability functions and length scale held); the plume's
#             upward speed answers with a factor that is 0 below 0.5 m/s and 1 above 1.5 m/s
#   q1        --freeze closure-plume --plume-hold w --mix-live q:1      (the mixing form alone, the plume's speed held)
# Rows, checks and what was dropped: plume/strike-bl/RESULT.md. Off by default.
BL_LIVE = {'q1-ramp': ['--freeze', 'closure-plume', '--plume-w-smooth', 'ramp:0.5:1.5', '--mix-live', 'q:1'],
           'q1': ['--freeze', 'closure-plume', '--plume-hold', 'w', '--mix-live', 'q:1']}
if '--bl-live' in argv:
    i = argv.index('--bl-live')
    if argv[i + 1] not in BL_LIVE: raise SystemExit('--bl-live ' + '|'.join(BL_LIVE))
    if any(x in argv for x in ('--freeze', '--plume-hold', '--plume-w-smooth', '--mix-live')):
        raise SystemExit('--bl-live sets --freeze, the plume option and --mix-live itself: give it alone (with --fused, --frame)')
    argv[i:i + 2] = BL_LIVE[argv[i + 1]]
FREEZE = 'closure'
if '--freeze' in argv:
    i = argv.index('--freeze'); FREEZE = argv[i + 1]; del argv[i:i + 2]
# lane GROUND (2026-11-10): --freeze closure-plume is the closure list minus the two plume routines only: the outputs
# of f_dmp_mf and f_ddmp_mf (edmf_*, s_aw*, sd_aw*, sub_*, det_*, maxwidth, maxmf, ztop, excess_h/q, tkeprod_up/dn)
# keep their derivative; pblh, the stability functions, the mixing length, the diffusivities, exch_m/h, the launch
# inputs and f_topdown_cloudrad stay frozen as under closure. It is the same set of freeze points as
# --freeze closure --freeze-mask 0x7cff (groups 8 and 9 off), written without the group wrappers.
# Breakage it is for: with the plume frozen the 2-step tangent explains 0.14 / 0.02 of the real theta_m change at
# the two lowest levels of plume columns (measured on real columns). NOT --freeze none:
# that also frees the closure whose tangent was 1e4-5e6x the primal (ruling 2026-11-05).
# The stage M4 swap (cuda/m4_plume.py) re-freezes the plume and refuses a closure-plume source.
if FREEZE not in ('none', 'closure', 'closure+cloud', 'closure-plume'):
    raise SystemExit('--freeze none|closure|closure+cloud|closure-plume')
# lane GROUND (2026-11-11): --plume-hold LIST (default: nothing held; needs --freeze closure-plume) holds the derivative
# of named terms inside the live updraft plume (f_dmp_mf). The forward value is untouched (mw_frz keeps it bitwise).
#   w     the updraft velocity wn after its limiter: the plume's mass flux, entrainment rate and top no longer respond;
#         what the plume carries (thl, qt, u, v of the air it entrains) still does
#   winv  only the 1/upw in the buoyancy term of the velocity equation (bcoeff*b/upw)
#   b     only the buoyancy b that drives the velocity equation
#   adj   only the surface-flux limiter `adjustment` that rescales every s_aw array
# Breakage it is for: with the plume fully live the 30-step tangent of case b6h-dx-0331p reaches 107 K at the plume top
# in single columns by 6 h where the real model changes by 0.3 K. The velocity equation
# divides by upw, which goes to zero at the plume top, so d(wn)/d(upw) and d(wn)/d(b) grow like 1/upw and 1/upw^2
# there: the true derivative of where the plume stops, which the real model takes as a whole level or not at all.
PLUME_HOLD = []
if '--plume-hold' in argv:
    i = argv.index('--plume-hold'); PLUME_HOLD = [x for x in argv[i + 1].split(',') if x]; del argv[i:i + 2]
if any(x not in ('w', 'winv', 'b', 'adj') for x in PLUME_HOLD):
    raise SystemExit('--plume-hold w|winv|b|adj (comma separated)')
if PLUME_HOLD and FREEZE != 'closure-plume':
    raise SystemExit('--plume-hold needs --freeze closure-plume')
# bl-live-reverse (2026-11-14): --plume-live mom (default: not given; needs --freeze closure, no group mask) keeps ONE
# part of the updraft plume live under the closure freeze: the plume's transport of momentum. The plume's own wind
# (upu, upv: its start value, what it entrains, what it detrains into the environment mean) and the three terms the
# wind solves read from it (the mass-flux momentum flux s_awu/s_awv, the subsidence term sub_u/sub_v and the detrainment
# term det_u/det_v) answer the environment wind u1, v1. Every coefficient of that answer (entrainment, plume area and
# speed, density, layer depths, the flux adjustment, the scale-aware factor) is the forward's value with no derivative
# (mw_frz), so the plume's momentum terms are a LINEAR function of u1, v1 with held coefficients, and everything else the
# plume returns stays frozen after the call exactly as under --freeze closure. The forward value is untouched.
# Breakage it is for: with s_awu/s_awv frozen the wind solves keep the downward (compensating) half of the mass-flux
# transport, -s_aw*u, live and drop the upward half, so the 12-step response of near-ground wind on the 12 km storm cut
# reads 6.458 where the real forecast reads 3.881 (low-wind lane, e3-12.score.json); with the plume live 3.857.
# Why not --freeze closure-plume --plume-hold w: that form has no tape-free backward (the whole plume integration would
# need a hand transpose); this one is linear with held coefficients, so its transpose needs only those coefficients.
PLUME_LIVE = None
if '--plume-live' in argv:
    i = argv.index('--plume-live'); PLUME_LIVE = argv[i + 1]; del argv[i:i + 2]
    if PLUME_LIVE not in ('mom', 'momheat'): raise SystemExit('--plume-live mom|momheat')
    if FREEZE != 'closure': raise SystemExit('--plume-live mom|momheat needs --freeze closure (the rest of the plume stays frozen)')
# lane GROUND (2026-11-11, round 7): --plume-w-smooth FORM (default: not given; needs --freeze closure-plume; not with
# --plume-hold w) is the smooth replacement of the hold `w`. The derivative of the updraft velocity wn after its limiter
# is KEPT and multiplied by a factor phi of the FORWARD STATE ONLY (never of the perturbation, so the operator is the
# same for every direction and its transpose exists). m = min(wn, upw(k)) is the smaller of the new speed and the speed
# one level below, which is the divisor of the velocity equation (bcoeff*b/upw); phi goes to zero with m:
#   sq:W0        phi = m*m / (m*m + W0*W0)        (W0 in m/s, the stated floor: 1/upw becomes at most 1/(2 W0),
#                                                  1/upw^2 at most 1/W0^2; phi -> 1 for a fast plume)
#   ifs:W0:FRAC  phi = FRAC where m >= W0, else 0 (the published practice read by the blocker hunt, IFS Cy47r1 Part II
#                                                  3.9.7: the speed's perturbation is 0 below 0.1 m/s and 0.1 of it above)
# The forward value is untouched: mw_scl keeps a.v bitwise and scales a.d only. Off by default: without the option the
# output is byte for byte the one before this commit.
# Breakage it is for: the same as --plume-hold w (107 K at the plume top by 6 h), without giving
# the plume's mass flux, entrainment and top no response at all (the hold).
PLUME_W_SMOOTH = None
if '--plume-w-smooth' in argv:
    i = argv.index('--plume-w-smooth'); PLUME_W_SMOOTH = argv[i + 1].split(':'); del argv[i:i + 2]
    if PLUME_W_SMOOTH[0] == 'sq' and len(PLUME_W_SMOOTH) == 2:
        _w0 = float(PLUME_W_SMOOTH[1]); MW_WPHI = 'return m*m / (m*m + (%r)*(%r));' % (_w0, _w0)
    elif PLUME_W_SMOOTH[0] == 'ifs' and len(PLUME_W_SMOOTH) == 3:
        _w0 = float(PLUME_W_SMOOTH[1]); _fr = float(PLUME_W_SMOOTH[2]); MW_WPHI = 'return m >= %r ? %r : 0.0;' % (_w0, _fr)
    elif PLUME_W_SMOOTH[0] == 'sq2' and len(PLUME_W_SMOOTH) == 2:
        # strike physics bl (2026-11-11): the square of the sq ratio: next to nothing below W0/2, most of the response in a fast plume
        _w0 = float(PLUME_W_SMOOTH[1]); MW_WPHI = 'const double r = m*m / (m*m + (%r)*(%r)); return r*r;' % (_w0, _w0)
    elif PLUME_W_SMOOTH[0] == 'ramp' and len(PLUME_W_SMOOTH) == 3:
        # strike physics bl (2026-11-11): 0 below W0, 1 above W1, linear between (continuous, a factor of the forward state only)
        _w0 = float(PLUME_W_SMOOTH[1]); _w1 = float(PLUME_W_SMOOTH[2])
        if not _w1 > _w0: raise SystemExit('--plume-w-smooth ramp:W0:W1 needs W1 > W0')
        MW_WPHI = 'return m <= %r ? 0.0 : (m >= %r ? 1.0 : (m - %r) / (%r - %r));' % (_w0, _w1, _w0, _w1, _w0)
    else:
        raise SystemExit('--plume-w-smooth sq:W0 | sq2:W0 | ramp:W0:W1 | ifs:W0:FRAC')
    if not _w0 > 0.0: raise SystemExit('--plume-w-smooth: the floor W0 must be positive')
    if FREEZE != 'closure-plume': raise SystemExit('--plume-w-smooth needs --freeze closure-plume')
    if 'w' in PLUME_HOLD: raise SystemExit('--plume-w-smooth replaces --plume-hold w: give one of them')
# S10 (2026-11-07): --frame inline1 (the default) emits every void helper that has exactly one call site as
# __forceinline__, as the forward image does for f_mynnedmf_driver and f_mynnedmf. Their ~100-array by-value
# parameter blocks and the register save areas around their calls leave the thread stack, which the driver reserves
# for every resident thread of the card (26.7 KB/thread = 7.2 GiB on a 188-SM card before S10). Call shape only:
# the same operations in the same order (--fmad=false), so every word is unchanged.
# --frame noinline reproduces the previous output byte for byte.
FRAME = 'inline1'
if '--frame' in argv:
    i = argv.index('--frame'); FRAME = argv[i + 1]; del argv[i:i + 2]
if FRAME not in ('inline1', 'noinline'):
    raise SystemExit('--frame inline1|noinline')
# lane S11 (2026-11-07): --freeze-mask M selects freeze groups (MW_FRZ_GROUPS below; bisection and the gated freeze);
# --gate-q Q gates groups 4 and 5 (the level-2.5 stability functions and the diffusivities, frozen where formed) on the
# forward's own turbulence: frozen only at levels where qkw = sqrt(qke) (m/s) is below Q, live where the turbulence is
# resolved. The gate reads the primal value only (no mw_rec), so TL, AD and the tape count take the same decision and
# the operator stays linear in the direction. Neither option given: the output is byte for byte the ungated one.
# x20 (2026-11-09): --fused m8 replaces the taped f_mym_predict (stage M8: implicit qke/qsq, algebraic tsq/cov) in the
# adjoint and count entries by one external tape record and a hand reverse (cuda/mw_m8_predict.cuh): the record keeps
# the stage's true inputs, the reverse recomputes every intermediate and runs the transposed Thomas solve. The tangent
# entry and every unsupported deck keep the generated routine. Not given: the output is byte for byte the default one.
FUSED = []
if '--fused' in argv:
    i = argv.index('--fused'); FUSED = [x for x in argv[i + 1].split(',') if x]; del argv[i:i + 2]
if any(x not in ('m8',) for x in FUSED):
    raise SystemExit('--fused m8')
FRZ_MASK = None; GATE_Q = None
if '--freeze-mask' in argv:
    i = argv.index('--freeze-mask'); FRZ_MASK = int(argv[i + 1], 0); del argv[i:i + 2]
if '--gate-q' in argv:
    i = argv.index('--gate-q'); GATE_Q = argv[i + 1]; float(GATE_Q); del argv[i:i + 2]
GROUPED = FRZ_MASK is not None or GATE_Q is not None
# a12 truephys-term (2026-11-13): --l2-quot exact|generic. The level-2 stability functions of f_mym_level2,
#   sh = shc (rfc - rf) / (1 - rf)   and   sm = smc (rf1 - rf) / (rf2 - rf) sh   (module_bl_mynnedmf.F90:1791-1792),
# (and the flux Richardson number rf of F90:1789 they are formed from, whose 1 - (2 ri - ri3) / (2 sqrt D) cancels on the
# stable side) took their derivative through the generic quotient rule (a' - q b') / b. Where the shear is near zero under an
# unstable gradient the Richardson number is large and negative (down to -gh / 1e-10, the routine's own floor on the
# shear), so |rf| is large; the numerator and the denominator then carry the same tangent -drf, and the rule subtracts
# two terms of size |drf| to leave one of size |drf| / rf^2. The tangent and the adjoint each lose about 2 log10 |rf|
# digits, in different orders, and q is only as good as the float32 words it is formed from.
# Breakage it prevents (term-work/l25 on the live source 69d27e49, 2026-11-13): the adjoint against the tangent of these
# two statements differed by up to 1.25e-06 of the size of its terms in double, the tangent against a central difference
# in double by up to 1.96 of that size. exact (the default whenever the level-2 output is live: --freeze none, or a
# --freeze-mask with bit 2 clear) emits mw_l2_sh / mw_l2_sm (cuda/mw_ad_rules.inc): the value is the same float32
# operations in the same order (bitwise), the derivative the closed-form partials (dsh/drf = shc (rfc - 1) / (1 - rf)^2,
# dsm/drf = smc sh (rf1 - rf2) / (rf2 - rf)^2, and the plain ones in shc, rfc, smc, rf1, rf2, sh), with no cancellation.
# generic reproduces the earlier output byte for byte; a source whose level-2 output is frozen after the call
# (closure, closure-plume, closure+cloud, a mask with bit 2 set) keeps generic, where the derivative is zero anyway.
L2_QUOT = None
if '--l2-quot' in argv:
    i = argv.index('--l2-quot'); L2_QUOT = argv[i + 1]; del argv[i:i + 2]
    if L2_QUOT not in ('exact', 'generic'): raise SystemExit('--l2-quot exact|generic')
if L2_QUOT is None:
    L2_QUOT = 'exact' if (FREEZE == 'none' or (GROUPED and not ((FRZ_MASK >> 2) & 1))) else 'generic'
# a12 truephys lead (2026-11-13): --l25-quot live|hold (default live). hold keeps the derivative of the two level-2.5
# stability quotients of f_mym_turbulence, sm = q3sq a1 (e3 - 3 c1 e4) / eden (module_bl_mynnedmf.F90:2874) and
# sh = q3sq a2 a2fac (e2 + 3 c1 e5c gmel) / eden (F90:2878), out of the tangent and the adjoint (mw_frz: the value is
# bitwise the forward's). Everything else of the closure stays live: the level-2 functions, the q3sq < q2sq branch
# (qdiv), the length, the diffusivities.
# Breakage it prevents (3 km storm box, phys-3, 2026-11-13): with these two statements live the adjoint check of the
# live boundary layer misses (12 steps 2.17e-07, 21 steps 4.21e-06 with --l2-quot exact; bar 1e-07); held it reads
# 1.55e-08 at 12 steps and the 5-minute low-level moisture score stays at 0.649 (fully live 0.743, closure frozen 0.203).
# Reproduces the measured source byte for byte (base 0f49b8b4, composed 204c803e).
L25_QUOT = 'live'
if '--l25-quot' in argv:
    i = argv.index('--l25-quot'); L25_QUOT = argv[i + 1]; del argv[i:i + 2]
    if L25_QUOT not in ('live', 'hold', 'exact'): raise SystemExit('--l25-quot live|hold|exact')
# fullphys mynn-exact (2026-11-13): --l25-quot exact emits the two statements as mw_l25_sm / mw_l25_sh
# (cuda/mw_ad_rules.inc): the value is the statement's own double operations (bitwise), the derivative one node with
# closed-form partials in q3sq, ghel, gmel, a2fac (and a2 a2fac for sh), its q3sq-partial from the degree-0
# homogeneity in (q3sq, ghel, gmel) where that form has the smaller rounding bound. Breakage it prevents: the generic
# quotient rule takes the q3sq-derivative as the difference of two terms of size 2 s / q3sq that cancel to a value of
# relative size ghel / q3sq, so in a well-mixed layer the tangent and the adjoint of these statements disagree by up to
# 4.1e-09 of the statement's terms (stages_test/l25, live source 9cc87a5b); exact keeps them within 1e-15.
# Off by default (live stays the generic rule) until the 3 km window check passes with it.
if GROUPED and FREEZE == 'none':
    raise SystemExit('--freeze-mask/--gate-q need --freeze closure')
# bl-guard (2026-11-14): --weakshear [T] (T in s-2, default 1e-4; not given: the output is byte for byte the one before
# this option) is the weak-shear rule of the exact form (--freeze none). In f_mym_turbulence, for each interface k, after
# the stability functions have their final value and before the diffusivities and the turbulence production terms are
# formed, where the forecast's own squared shear of the interface gm(k) is under T the two stability functions sm(k),
# sh(k) and the terms formed with them (qdiv, gamt, gamq, gamv) keep their value and carry no derivative (mw_frz); the
# true derivative everywhere else. ONE decision from the forward value alone (mw_pv reads the value, no derivative, no
# branch bit), so the tangent, the taped reverse and the count entries take the same decision from the same bits. No size
# of any derivative is capped. WHAT ELSE IT HOLDS where it binds, because the hold comes after the scheme's own floors on
# sm and sh: the slope of the plume floor (0.04 min(10 mfmax, 1)) and of the cloud floor (0.04 min(cldavg, 0.5)) through
# sm and sh, the slope of sm and sh in the turbulence energy and the mixing length, and through qdiv the turbulence
# energy's slope of elh = elq qdiv in the production terms. What stays live there: elq = el qkw, and the solves.
# Breakage it prevents: the scheme forms the Richardson number as -gh / max(gm, 1e-10); its slope in the temperature
# difference goes as 1 / gm and in the wind difference as 1 / gm^(3/2). In a calm column (3 km storm box, cell 21685, a
# fog column with gm 3e-06 to 4e-05 s-2 at its four lowest interfaces) the exact form's tangent of a +1 g/kg nudge reads
# +3.6e-02 after 59 steps where the real model moves by -3e-05 to -8e-05 whichever way; with this rule -1.8e-08, and the
# 12-step near-ground wind stays at 1.003 of the real change (1.006 with nothing held, 1.37 with the closure held).
# What was shown and what was not: weakshear/README.md. Byte test: integrated/test_weakshear.sh.
WEAKSHEAR = None
if '--weakshear' in argv:
    i = argv.index('--weakshear'); del argv[i]
    WEAKSHEAR = 1.0e-4
    if i < len(argv):
        try:
            WEAKSHEAR = float(argv[i]); del argv[i]
        except ValueError:
            pass
    if not 0.0 < WEAKSHEAR < float('inf'):   # inf would hold every interface under the exact form's name
        raise SystemExit('--weakshear T: the threshold T (s-2) must be a finite positive number')
    if BL_LIVE_GIVEN or (FREEZE != 'none' and not MIX_LIVE_GIVEN and not GROUPED):  # the rule also with the freeze groups (the closure-live forms)
        raise SystemExit('--weakshear is the rule of the exact form (--freeze none) or of the live mixing (--freeze closure '
                         '--mix-live with the stability path s): under --freeze closure alone the stability functions are held already')
if GROUPED and FRZ_MASK is None:
    FRZ_MASK = 0x7fff
# strike physics bl (2026-11-11): --mix-live SPEC (default: not given; needs --freeze closure or closure-plume, not with
# --freeze-mask/--gate-q) is the bounded live form of the mixing hold. The diffusivities K = el * q * S / dz (dfm, dfh, dfq;
# el the mixing length, q the turbulence speed sqrt(qke), S the level-2.5 stability function sm or sh) are held at the
# forecast's values under --freeze closure. With this option K's derivative is KEPT along two named paths, each multiplied
# by a factor of the FORWARD STATE ONLY (never of the perturbation: one linear operator for every direction, its transpose
# exists). The mixing length, the level-2 functions, pblh and the counter-gradient terms stay held, and so does S inside
# the turbulence-energy budget (pdk, pdt, pdq, pdc): only the diffusivities answer.
#   q:FQ[:Q0]   the turbulence-energy path, dK = K dq/q, times FQ (0..1); with Q0 (m/s) also times q*q/(q*q + Q0*Q0),
#               which takes the response away where the turbulence speed goes to zero (dq = dqke / 2q is singular there)
#   s:FS[:u|st] the stability path, dK = K dS/S (S answers the local gradients gh, gm and q), times FS (0..1);
#               `u` only where the layer is unstable (gh > 0), `st` only where it is stable; default everywhere
#   l:FL[:t]   the length-scale path, dK = K del/el (el answers q, the stratification and the plume), times FL (0..1). The
#               length scale stays held everywhere else (the stability functions, the turbulence-energy budget, the output
#               el_pbl); with `t` it is also live in the production terms formed in the turbulence routine (pdk, pdt, pdq, pdc).
#               The source must then be composed with f_mym_length taped at its call in f_mym_turbulence
#               (plume/strike-bl/compose_l.sh: stage M6 keeps the level-2 closure and the initialisation call primal-only).
# SPEC is one or more, comma separated, e.g. `q:1`, `s:0.1`, `q:1,s:0.1`, `q:1,l:1`. A path not named is held.
# The forward value is untouched: K is formed again from the same operands (mw_scl and mw_frz keep values bitwise).
# Breakage it is for: the fully live closure runs away (tangent 1e4-5e6 times the primal, ruling 2026-11-05) and so does
# the closure live where q >= 1.0 or 0.5 m/s (0.8777 / -16.44 / -8043.5), while the hold leaves the
# near-ground derivative at 0.46 of the real model's change at 6 h where 0.75 is possible (measured).
MIX_LIVE = None
if '--mix-live' in argv:
    i = argv.index('--mix-live'); _spec = argv[i + 1]; del argv[i:i + 2]
    MIX_LIVE = {'spec': _spec, 'fq': 0.0, 'q0': None, 'fs': 0.0, 'smode': 'a', 'fl': 0.0, 'lmode': 'k'}
    for _t in [x for x in _spec.split(',') if x]:
        _p = _t.split(':')
        if _p[0] == 'q' and len(_p) in (2, 3):
            MIX_LIVE['fq'] = float(_p[1]); MIX_LIVE['q0'] = float(_p[2]) if len(_p) == 3 else None
        elif _p[0] == 's' and len(_p) in (2, 3) and (len(_p) == 2 or _p[2] in ('a', 'u', 'st')):
            MIX_LIVE['fs'] = float(_p[1]); MIX_LIVE['smode'] = _p[2] if len(_p) == 3 else 'a'
        elif _p[0] == 'l' and len(_p) in (2, 3) and (len(_p) == 2 or _p[2] == 't'):
            MIX_LIVE['fl'] = float(_p[1]); MIX_LIVE['lmode'] = 't' if len(_p) == 3 else 'k'
        else:
            raise SystemExit('--mix-live q:FQ[:Q0][,s:FS[:u|st]][,l:FL[:t]]')
    if not (0.0 <= MIX_LIVE['fq'] <= 1.0 and 0.0 <= MIX_LIVE['fs'] <= 1.0 and 0.0 <= MIX_LIVE['fl'] <= 1.0) or MIX_LIVE['fq'] + MIX_LIVE['fs'] + MIX_LIVE['fl'] == 0.0:
        raise SystemExit('--mix-live: FQ, FS and FL are fractions in 0..1 and one of them must be above 0')
    if MIX_LIVE['q0'] is not None and not MIX_LIVE['q0'] > 0.0: raise SystemExit('--mix-live: the floor Q0 must be positive')
    if FREEZE not in ('closure', 'closure-plume'): raise SystemExit('--mix-live needs --freeze closure or closure-plume')
    if GROUPED: raise SystemExit('--mix-live is not combined with --freeze-mask/--gate-q')
    # bl-fast-live (2026-11-14): --weakshear with --mix-live holds the stability path of the diffusivities (mw_ksm, mw_ksh)
    # where the forecast's own gm(k) is under T (the exact form's rule restricted to what is live here: sm, sh, qdiv and the
    # counter-gradient terms are held anyway). The turbulence-energy path stays live. Without s there is nothing to hold.
    if WEAKSHEAR is not None and not MIX_LIVE['fs'] > 0.0:
        raise SystemExit('--weakshear with --mix-live needs the stability path (s:FS): without it the functions are held already')
# lane GROUND (2026-11-11, worker mix): the gate combines with the live plume as it stands; no code path was added.
#   --freeze closure-plume --plume-hold w --freeze-mask 0x37ff --gate-q Q --fused m8
# is the best bundle (plume live, its upward speed held) with the diffusivities live where the forecast own turbulence
# speed qkw is at or above Q: groups 4, 5 and 13 are frozen only below Q, and bits 11 and 14 are OFF in the mask because
# the after-call freeze of dfm/dfq/dfh (groups 11, 14) would otherwise re-freeze what the gate left live. sm/sh/el after
# f_mym_turbulence (group 6), the mixing length (3), level-2 sm/sh (2), pblh (1) and the launch inputs (0) stay frozen.
# Bits 8 and 9 mean nothing under closure-plume (the plume routines have no after-call freeze there). Off unless asked:
# none of these options given, the output is byte for byte the default (plume/ready/gencheck.sh, 5 of 5).
# Tangent image only so far; the adjoint and count entries of this source were not built or run.
MW_FRZ_GROUPS = {0: 'launch inputs el_pbl/sh3d/sm3d/pblh', 1: 'pblh, psig_bl, psig_shcu after f_get_pblh/f_scale_aware',
                 2: 'level-2 sm/sh after f_mym_level2', 3: 'mixing length el after f_mym_length',
                 4: 'level-2.5/3 sm/sh/qdiv/gamt/gamq/gamv before use (gated by --gate-q)',
                 5: 'dfm/dfq where formed (gated by --gate-q)', 6: 'sh/sm/el after f_mym_turbulence',
                 7: 'exch_m/exch_h after f_retrieve_exchange_coeffs', 8: 'updraft plume (f_dmp_mf)',
                 9: 'downdraft plume (f_ddmp_mf)', 10: 'top-down cloud radiation (f_topdown_cloudrad)',
                 11: 'dfm/dfq after f_mym_turbulence', 12: 'subgrid cloud closure + surface similarity (closure+cloud)',
                 13: 'dfh where formed (gated by --gate-q)', 14: 'dfh after f_mym_turbulence'}
# (masks built before the 13/14 split, 2026-11-07 ~10:50Z, carried dfh in bits 5 and 11; gq10/l45/b* were built so)
AFTER_BIT = {'f_get_pblh': 1, 'f_scale_aware': 1, 'f_mym_level2': 2, 'f_mym_length': 3, 'f_retrieve_exchange_coeffs': 7,
             'f_dmp_mf': 8, 'f_ddmp_mf': 9, 'f_topdown_cloudrad': 10, 'f_mym_condensation': 12}
def _after_bit(callee, name):
    if callee == 'f_mym_turbulence': return 14 if name == 'dfh' else 11 if name in ('dfm', 'dfq') else 6
    if callee == 'f_dmp_mf' and name in ('qc_bl1', 'cldfra_bl1'): return 12
    return AFTER_BIT[callee]
def _wrap(bit, code, gate=False):
    if not GROUPED: return code
    g = ' && MW_GATE_LO(qkw(k))' if (gate and GATE_Q is not None) else ''
    return 'if (MW_FRZ_ON(%d)%s) { %s }' % (bit, g, code)
# --correct IDS (physics-correct, 15 Nov): differentiate correct mode's forward, the captured source with the fork
# corrections IDS applied by the forward module's own text pair (hexcore.mpasphys.rrfs.bl_mynnedmf_c6d08709._MY1,
# repeated here so this build-time tool needs no hexcore import). Off by default: the output is then byte for byte the
# exact-mode generation. The pin check stays on the captured input.
CORRECT = []
if '--correct' in argv:
    i = argv.index('--correct'); CORRECT = [x for x in argv[i + 1].split(',') if x]; del argv[i:i + 2]
    if set(CORRECT) - {'MY1', 'MY2'}: raise SystemExit('--correct knows MY1 and MY2 only')
_MY1 = ('qvflux = mynn_max((float)(qvflux), (float)(r6_div(mynn_min((float)(r6_sub(r6_mul(0.9f, sqv(kts)), '
        '1e-8f)), (float)(zero)), dtz(kts))));',
        'qvflux = mynn_max((float)(qvflux), (float)(r6_div(mynn_min((float)(r6_sub(1e-8f, r6_mul(0.9f, sqv(kts)))), '
        '(float)(zero)), r6_mul(r6_mul(dtz(kts), rhosfc), rhoinv(kts)))));')
_MY2 = ('rhosfc = r6_div(psfc, r6_mul(r_d, r6_add(tk(kts), r6_mul(p608, qv(kts)))));',
        'rhosfc = r6_div(psfc, r6_mul(r6_mul(r_d, tk(kts)), r6_add(one, r6_mul(p608, qv(kts)))));')
src, out = Path(argv[0]), Path(argv[1])
s = src.read_text()
if hashlib.sha256(s.encode()).hexdigest() != CAPTURE:
    raise SystemExit('captured MYNN source hash mismatch')
if 'MY1' in CORRECT:
    if s.count(_MY1[0]) != 1: raise SystemExit('--correct MY1: the dew limiter line is not in the captured source once')
    s = s.replace(*_MY1)
if 'MY2' in CORRECT:
    if s.count(_MY2[0]) != 1: raise SystemExit('--correct MY2: the surface density line is not in the captured source once')
    s = s.replace(*_MY2)

start = s.index('template<int MYNN_LANES> struct MynnColumn {')
end = s.index('\n};', start) + 3
pre, body, tail = s[:start], s[start:end], s[end:]

# ---- 1. closure of the top helper inside MynnColumn -------------------------------------------
ms = list(re.finditer(r'^__device__[^\n]*?\b(f_\w+)\(', body, re.M))
funcs = {m.group(1): body[m.start():(ms[i + 1].start() if i + 1 < len(ms) else len(body) - 3)]
         for i, m in enumerate(ms)}
todo, seen = [TOP], set()
while todo:
    n = todo.pop()
    if n in seen:
        continue
    if n not in funcs:
        raise SystemExit('missing helper ' + n)
    seen.add(n)
    todo.extend(sorted(set(re.findall(r'\b(f_\w+)\s*\(', funcs[n])) - seen))
body = body[:ms[0].start()] + ''.join(funcs[m.group(1)] for m in ms if m.group(1) in seen) + '\n};\n'

# ---- 2. type substitution (float -> MR, double -> MRD) -----------------------------------------
body = body.replace('template<int MYNN_LANES> struct MynnColumn {',
                    'template<int MYNN_LANES, class MR=float, class MRD=double> struct MynnColumn {', 1)
body = body.replace('int lane_sub = 0; unsigned lane_mask = 0u;',
                    'int lane_sub = 0; unsigned lane_mask = 0u;\n'
                    'template<class Q> __device__ Q mynn_lane_bcast(Q v, int) const { return v; }', 1)
new = []
for line in body.splitlines():
    if line.startswith('template<int MYNN_LANES, class MR='):
        new.append(line); continue
    if 'constexpr float' in line or 'mynn_lane_bcast(float' in line or 'mynn_lane_bcast(double' in line:
        new.append(line); continue
    line = re.sub(r'\(float\)\(', '(MR)(', line)
    line = line.replace('mynn_array<float, ', 'mynn_array_any<float, ')
    line = re.sub(r'\bfloat\b', 'MR', line)
    if 'constexpr double' not in line:
        line = re.sub(r'\(double\)\(', '(MRD)(', line)
        line = re.sub(r'\bdouble\b', 'MRD', line)
    new.append(line)
body = '\n'.join(new).replace('mynn_array<MR, ', 'mynn_array_any<MR, ')
body, n_noinline = re.subn(r'^__device__ __forceinline__ (void f_\w+\()', r'__device__ __noinline__ \1', body, flags=re.M)
# debug builds only (MW_DEBUG): the first thread of a launch announces each helper it enters
body = re.sub(r'^(__device__ __noinline__ (?:void|MR) (f_\w+)\(.*\) \{)$',
              lambda m: m.group(1) + '\n#ifdef MW_BOUNDS\nif (mw_bounds_rec) mw_bounds_rec[8*(blockIdx.x*blockDim.x+threadIdx.x)+7] = ' + str(sorted(seen).index(m.group(2)) + 1) + ';\n#endif\n#ifdef MW_DEBUG\nif (blockIdx.x == 0 && threadIdx.x == 0) printf("enter ' + m.group(2) + '\\n' + '");\n#endif',
              body, flags=re.M)
if 'mynn_array<MRD' in body:
    raise SystemExit('double automatic arrays are not handled')

# ---- 2a. trajectory-frozen closure (ruling 2026-11-05 08:35Z: coefficients that are discontinuous or singular
#          functions of the state are FROZEN from the forward trajectory in TL/AD, no derivative through them).
#          mw_frz(x) keeps the primal value bitwise and drops the derivative (tangent 0, no tape node), so the
#          primal of every entry is unchanged; the TL/AD become the tendency equations (implicit vertical diffusion,
#          TKE/variance prediction, mass-flux transport of the environment) with frozen coefficients.
#          Switches and clamps: every comparison decides on the forward value (exact executed branch, unchanged);
#          a clamped arm carries the bound's zero derivative.
#          Breakage it prevents: MYNN-EDMF exact-branch tangents 1e4-5e6x primal at launch 1 (sh3d 4.8e6, exch_h
#          1.5e5, sm3d 5.6e4, el_pbl 1.9e4; WINDOW-RUN-2026-11-05b step 2), 4.9e21x held over 18 steps, wrong-sign
#          J2u Taylor ratio with MYNN-EDMF live. --freeze none reproduces the exact-branch output b809ef96.
def _params(sig):
    out, depth, cur = [], 0, ''
    for ch in sig:
        if ch in '<(': depth += 1
        elif ch in '>)': depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur.strip()); cur = ''
        else:
            cur += ch
    if cur.strip(): out.append(cur.strip())
    return out

def _sig(name):
    m = re.search(r'^__device__ __noinline__ (?:void|MR) ' + name + r'\((.*)\) \{$', body, re.M)
    if not m: raise SystemExit('signature not found ' + name)
    return [(p.rsplit(None, 1)[0], p.rsplit(None, 1)[1]) for p in _params(m.group(1))]

# callee -> parameter names whose values are frozen right after every call (outputs of the closure routines)
FREEZE_AFTER = {}
FREEZE_STMT = []      # (function, exact line, 'before'|'after', code, expected count)
if FREEZE != 'none':
    FREEZE_AFTER = {
        'f_get_pblh': ['pblh'],                                   # PBL height (kpbl is integer control)
        'f_scale_aware': ['psig_bl', 'psig_shcu'],                # scale-aware switches of pblh and dx
        'f_mym_level2': ['sm', 'sh'],                             # level-2 stability functions
        'f_mym_length': ['el'],                                   # mixing length (resets, limiters)
        'f_mym_turbulence': ['sh', 'sm', 'el'] if MIX_LIVE else ['sh', 'sm', 'el', 'dfm', 'dfh', 'dfq'],  # stability functions, diffusivities
        'f_retrieve_exchange_coeffs': ['km1', 'kh1'],             # exch_m, exch_h
        'f_dmp_mf': ['edmf_a1', 'edmf_w1', 'edmf_qt1', 'edmf_thl1', 'edmf_ent1', 'edmf_qc1', 'edmf_qv1', 'edmf_u1',
                     'edmf_v1', 's_aw1', 's_awthl1', 's_awqt1', 's_awqv1', 's_awqc1', 's_awu1', 's_awv1', 's_awqke1',
                     's_awqsq1', 's_awqnc1', 's_awqni1', 's_awqnwfa1', 's_awqnifa1', 's_awqnbca1', 'sub_thl1',
                     'sub_sqv1', 'sub_u1', 'sub_v1', 'det_thl1', 'det_sqv1', 'det_sqc1', 'det_u1', 'det_v1',
                     's_awchem1', 's_awscalars1', 'maxwidth', 'maxmf', 'ztop', 'excess_h', 'excess_q',
                     'tkeprod_up'],                               # updraft plume: activation, top, fluxes
        'f_ddmp_mf': ['edmf_a_dd', 'edmf_w_dd', 'edmf_qt_dd', 'edmf_thl_dd', 'edmf_ent_dd', 'edmf_qc_dd', 'sd_aw',
                      'sd_awthl', 'sd_awqt', 'sd_awqv', 'sd_awqc', 'sd_awqi', 'sd_awqnc', 'sd_awqni', 'sd_awqnwfa',
                      'sd_awqnifa', 'sd_awu', 'sd_awv', 'sd_awqke', 'tkeprod_dn', 'maxmf_dd', 'maxwidth_dd'],
        'f_topdown_cloudrad': ['tkeprod_dn', 'maxtkeprod', 'cldtop_cooling', 'ent_eff'],
    }
    FREEZE_STMT = [
        # launch inputs carrying the closure state of the previous step (sh3d, sm3d, el_pbl, pblh)
        ('f_mynnedmf', 'sm1.rebase({kts});', 'after',
         _wrap(0, 'mw_frz_arr(el1); mw_frz_arr(sh1); mw_frz_arr(sm1); pblh = mw_frz(pblh);'), 1),
        # level-2.5/3 stability functions and their counter-gradient terms, frozen at their final value before use
        ('f_mym_turbulence', 'elq = r6_mul(el(k), qkw(k));', 'before',
         (('MR mw_ksm = sm(k); MR mw_ksh = sh(k); '
           + ('{ const double mw_fs = mw_mixs(mw_spv(gh(k))); mw_ksm = mw_scl(mw_ksm, mw_fs); mw_ksh = mw_scl(mw_ksh, mw_fs); } '
              if MIX_LIVE['fs'] > 0.0 else 'mw_ksm = mw_frz(mw_ksm); mw_ksh = mw_frz(mw_ksh); ')
           + (('if (mw_pv(gm(k)) < (double)(%r)) { mw_ksm = mw_frz(mw_ksm); mw_ksh = mw_frz(mw_ksh); } ' % WEAKSHEAR) if WEAKSHEAR is not None else '')
           + (('MR mw_kel = el(k); ' + (('mw_kel = mw_scl(mw_kel, %r); el(k) = mw_frz(el(k)); ' % MIX_LIVE['fl']) if MIX_LIVE['lmode'] == 'k' else ''))
              if MIX_LIVE['fl'] > 0.0 else '')) if MIX_LIVE else '') +
         _wrap(4, 'sm(k) = mw_frz(sm(k)); sh(k) = mw_frz(sh(k)); qdiv = mw_frz(qdiv); gamt = mw_frz(gamt); '
         'gamq = mw_frz(gamq); gamv = mw_frz(gamv);', gate=True), 1),
        # diffusivities, frozen where they are formed (before the SPP perturbation and the TKE budget terms)
        ('f_mym_turbulence', 'dfq(k) = dfm(k);', 'after',
         (_wrap(5, 'dfm(k) = mw_frz(dfm(k)); dfq(k) = mw_frz(dfq(k));', gate=True) + ' ' + _wrap(13, 'dfh(k) = mw_frz(dfh(k));', gate=True))
         if GROUPED else
         (('{ const double mw_fq = mw_mixq(mw_spv(qkw(k))); ' if MIX_LIVE['fq'] > 0.0 else '{ ') +
          'dfm(k) = r6_div(r6_mul(r6_mul(mw_kel, %s), mw_ksm), dzk); dfh(k) = r6_div(r6_mul(r6_mul(mw_kel, %s), mw_ksh), dzk); dfq(k) = dfm(k); }'
          % (('mw_scl(qtw(k), mw_fq)', 'mw_scl(qkw(k), mw_fq)') if MIX_LIVE['fq'] > 0.0 else ('mw_frz(qtw(k))', 'mw_frz(qkw(k))')))
         if (MIX_LIVE and MIX_LIVE['fl'] > 0.0) else
         (('{ const double mw_fq = mw_mixq(mw_spv(qkw(k))); dfm(k) = r6_div(r6_mul(mw_scl(elqt, mw_fq), mw_ksm), dzk); '
           'dfh(k) = r6_div(r6_mul(mw_scl(elq, mw_fq), mw_ksh), dzk); dfq(k) = dfm(k); }') if MIX_LIVE['fq'] > 0.0 else
          ('dfm(k) = r6_div(r6_mul(mw_frz(elqt), mw_ksm), dzk); dfh(k) = r6_div(r6_mul(mw_frz(elq), mw_ksh), dzk); dfq(k) = dfm(k);'))
         if MIX_LIVE else 'dfm(k) = mw_frz(dfm(k)); dfh(k) = mw_frz(dfh(k)); dfq(k) = mw_frz(dfq(k));', 1),
    ]
    if FREEZE == 'closure-plume':
        # live plume: no after-call freeze of the updraft and downdraft plume routines (see the note at --freeze)
        del FREEZE_AFTER['f_dmp_mf'], FREEZE_AFTER['f_ddmp_mf']
        if 'w' in PLUME_HOLD:
            FREEZE_STMT.append(('f_dmp_mf', 'wn = mynn_min((MR)(mynn_max((MR)(zero), (MR)(wn))), (MR)(3.0f));', 'after', 'wn = mw_frz(wn);', 1))
        if PLUME_W_SMOOTH:
            FREEZE_STMT.append(('f_dmp_mf', 'wn = mynn_min((MR)(mynn_max((MR)(zero), (MR)(wn))), (MR)(3.0f));', 'after',
                                'wn = mw_scl(wn, mw_wphi(mw_spv(wn), mw_spv(upw(k, ip))));', 1))
        if 'b' in PLUME_HOLD:
            FREEZE_STMT.append(('f_dmp_mf', 'b = r6_mul(grav, r6_sub(r6_div(thvn, thvk), one));', 'after', 'b = mw_frz(b);', 1))
        if 'adj' in PLUME_HOLD:
            FREEZE_STMT.append(('f_dmp_mf', 'adjustment = mynn_max((MR)(r6_div(r6_div(r6_mul(fluxportion, flt2), dz1(kts)), flx1)), (MR)(0.01f));', 'after',
                                'adjustment = mw_frz(adjustment);', 1))
        if 'winv' in PLUME_HOLD:
            for _old, _new in (('r6_div(r6_mul(bcoeff, b), mynn_max((MR)(upw(k, ip)), (MR)(0.2f)))', 'r6_div(r6_mul(bcoeff, b), mw_frz(mynn_max((MR)(upw(k, ip)), (MR)(0.2f))))'),
                               ('r6_div(r6_mul(bcoeff, b), upw(k, ip))', 'r6_div(r6_mul(bcoeff, b), mw_frz(upw(k, ip)))')):
                if body.count(_old) != 1: raise SystemExit('plume-hold winv anchor count %r: %d' % (_old, body.count(_old)))
                body = body.replace(_old, _new)
    if PLUME_LIVE in ('mom', 'momheat'):
        pass  # the plume momentum also with the freeze groups (the closure-live form t1w)
        for _n in ('s_awu1', 's_awv1', 'sub_u1', 'sub_v1', 'det_u1', 'det_v1'): FREEZE_AFTER['f_dmp_mf'].remove(_n)
        _pm = []
        for _x in 'uv':
            _U, _UP, _N, _EM, _K = _x + '1', 'up' + _x, _x + 'n', 'envm_' + _x, _x + 'k'
            _pm += [
                ('%s(2, ip) = r6_div(r6_add(r6_mul(%s(kts), dz1(((kts) + (1)))), r6_mul(%s(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));' % (_UP, _U, _U),
                 '%s(2, ip) = r6_div(r6_add(r6_mul(%s(kts), mw_frz(dz1(((kts) + (1))))), r6_mul(%s(((kts) + (1))), mw_frz(dz1(kts)))), mw_frz(r6_add(dz1(kts), dz1(((kts) + (1))))));' % (_UP, _U, _U)),
                ('%s = r6_add(r6_add(r6_mul(%s(k, ip), r6_sub(one, entexm)), r6_mul(%s(k), entexm)), r6_mul(r6_mul(dxsa, pgfac), r6_sub(%s, %sm1)));' % (_N, _UP, _U, _K, _K),
                 '%s = r6_add(r6_add(r6_mul(%s(k, ip), mw_frz(r6_sub(one, entexm))), r6_mul(%s(k), mw_frz(entexm))), mw_frz(r6_mul(r6_mul(dxsa, pgfac), r6_sub(%s, %sm1))));' % (_N, _UP, _U, _K, _K)),
                ('%s(k) = r6_add(%s(k), r6_mul(r6_mul(r6_mul(r6_sub(r6_mul(p5, r6_add(%s, %s(k, ip))), %s(k)), detrateuv), aratio), mynn_min((MR)(dzp), (MR)(dzpmax))));' % (_EM, _EM, _N, _UP, _U),
                 '%s(k) = r6_add(%s(k), r6_mul(r6_mul(r6_mul(r6_sub(r6_mul(p5, r6_add(%s, %s(k, ip))), %s(k)), mw_frz(detrateuv)), mw_frz(aratio)), mw_frz(mynn_min((MR)(dzp), (MR)(dzpmax)))));' % (_EM, _EM, _N, _UP, _U)),
                ('s_aw%s(k) = r6_add(s_aw%s(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), %s(k, ip)), psig_w));' % (_U, _U, _UP),
                 's_aw%s(k) = r6_add(s_aw%s(k), r6_mul(r6_mul(mw_frz(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip))), %s(k, ip)), mw_frz(psig_w)));' % (_U, _U, _UP)),
                ('{ s_aw%s.linear(mynn_element) = r6_mul(s_aw%s.linear(mynn_element), adjustment); }' % (_U, _U),
                 '{ s_aw%s.linear(mynn_element) = r6_mul(s_aw%s.linear(mynn_element), mw_frz(adjustment)); }' % (_U, _U)),
                ('sub_%s(kts) = r6_div(r6_div(r6_mul(r6_mul(r6_mul(0.5f, envi_w(kts)), envi_a(kts)), r6_sub(r6_mul(rho1(((kts) + (1))), %s(((kts) + (1)))), r6_mul(rho1(kts), %s(kts)))), dzi(kts)), rhoi(kts));' % (_U, _U, _U),
                 'sub_%s(kts) = r6_div(r6_div(r6_mul(mw_frz(r6_mul(r6_mul(0.5f, envi_w(kts)), envi_a(kts))), r6_sub(r6_mul(mw_frz(rho1(((kts) + (1)))), %s(((kts) + (1)))), r6_mul(mw_frz(rho1(kts)), %s(kts)))), mw_frz(dzi(kts))), mw_frz(rhoi(kts)));' % (_U, _U, _U)),
                ('sub_%s(k) = r6_div(r6_div(r6_mul(r6_mul(r6_mul(r6_mul(0.5f, r6_add(envi_w(k), envi_w(((k) - (1))))), 0.5f), r6_add(envi_a(k), envi_a(((k) - (1))))), r6_sub(r6_mul(rho1(((k) + (1))), %s(((k) + (1)))), r6_mul(rho1(k), %s(k)))), dzi(k)), rhoi(k));' % (_U, _U, _U),
                 'sub_%s(k) = r6_div(r6_div(r6_mul(mw_frz(r6_mul(r6_mul(r6_mul(0.5f, r6_add(envi_w(k), envi_w(((k) - (1))))), 0.5f), r6_add(envi_a(k), envi_a(((k) - (1)))))), r6_sub(r6_mul(mw_frz(rho1(((k) + (1)))), %s(((k) + (1)))), r6_mul(mw_frz(rho1(k)), %s(k)))), mw_frz(dzi(k))), mw_frz(rhoi(k)));' % (_U, _U, _U)),
                ('det_%s(k) = r6_mul(r6_mul(r6_mul(cdet, r6_sub(%s(k), %s(k))), envi_a(k)), psig_w);' % (_U, _EM, _U),
                 'det_%s(k) = r6_mul(r6_mul(r6_mul(cdet, r6_sub(%s(k), %s(k))), mw_frz(envi_a(k))), mw_frz(psig_w));' % (_U, _EM, _U)),
            ]
        # the plume's only other read of the wind: the surface wind speed that sets its width and number. Held, so the
        # plume's size, speed and what it carries of heat and moisture do not answer the wind (they are frozen after the
        # call anyway; this keeps them frozen inside the cloud-blend tail too, which reads them before that freeze).
        _pm.append(('wspd_pbl = mynn_sqrt(mynn_max((MR)(r6_add(mynn_ipow<2>(u1(kts)), mynn_ipow<2>(v1(kts)))), (MR)(0.01f)));',
                    'wspd_pbl = mw_frz(mynn_sqrt(mynn_max((MR)(r6_add(mynn_ipow<2>(u1(kts)), mynn_ipow<2>(v1(kts)))), (MR)(0.01f))));'))
        for _old, _new in _pm:
            if body.count(_old) != 1: raise SystemExit('plume-live mom anchor count %r: %d' % (_old[:60], body.count(_old)))
            body = body.replace(_old, _new)
    if PLUME_LIVE == 'momheat':
        # full/closure (2026-11-15): --plume-live momheat = mom plus the plume's transport of HEAT and MOISTURE, in the same
        # held-coefficient form. The plume's liquid-water potential temperature and total water (upthl, upqt: start value,
        # what they entrain) and what the heat and moisture solves read from them (the mass-flux fluxes s_awthl1, s_awqt1,
        # s_awqv1, the subsidence terms sub_thl1, sub_sqv1 and the detrainment terms det_thl1, det_sqv1 through envm_thl,
        # envm_sqv) answer the environment thl1, qt1, qv1. Everything else the plume computes from upthl, upqt (its
        # condensation, buoyancy, speed, area, entrainment, top, cloud values) reads them frozen, so every coefficient of
        # the answer is the forward's value and the plume's heat and moisture terms are LINEAR in thl1, qt1, qv1.
        # Stage M4 (cuda/m4_plume.py) is NEEDED after this: it runs the plume on frozen copies of its inputs, and the
        # lines below read the live environment through the views thl1_lv, qt1_lv, qv1_lv declared here.
        # Breakage it is for: with the plume's heat and moisture fluxes frozen the linear model keeps a low-level
        # temperature or vapour change near the ground where the forecast lifts it to the plume's upper levels: at 3 km
        # the 15-minute gradient reads 1.37 (temperature) and 1.82 (vapour) of the real change (scoreboard rows 5, 6).
        for _n in ('s_awthl1', 's_awqt1', 's_awqv1', 'sub_thl1', 'sub_sqv1', 'det_thl1', 'det_sqv1'): FREEZE_AFTER['f_dmp_mf'].remove(_n)
        _bl = body.split('\n')
        _f0 = [i for i, l in enumerate(_bl) if re.match(r'^__device__ __\w+__ void f_dmp_mf\(', l)]
        if len(_f0) != 1: raise SystemExit('plume-live momheat: f_dmp_mf found %d times' % len(_f0))
        _f1 = next(i for i in range(_f0[0] + 1, len(_bl)) if _bl[i].startswith('__device__ '))
        def _one(pred, what):
            h = [i for i in range(_f0[0], _f1) if pred(_bl[i])]
            if len(h) != 1: raise SystemExit('plume-live momheat anchor %s: %d' % (what, len(h)))
            return h[0]
        def _lv(line, names):
            for n in names:
                line, c = re.subn(r'\b%s\b' % n, n + '_lv', line)
                if c < 1: raise SystemExit('plume-live momheat: no %s in %r' % (n, line[:70]))
            return line
        for _pre, _names in (('upthl(2, ip) = ', ['thl1']), ('upqt(2, ip) = ', ['qt1']),
                             ('qtn = r6_add(r6_mul(upqt(k, ip), ', ['qt1']), ('thln = r6_add(r6_mul(upthl(k, ip), ', ['thl1']),
                             ('envm_thl(k) = r6_add(envm_thl(k), ', ['thl1']), ('envm_sqv(k) = r6_add(envm_sqv(k), ', ['qv1']),
                             ('sub_thl1(kts) = ', ['thl1']), ('sub_thl1(k) = ', ['thl1']), ('sub_sqv1(kts) = ', ['qv1']), ('sub_sqv1(k) = ', ['qv1']),
                             ('det_thl1(k) = ', ['thl1']), ('det_sqv1(k) = ', ['qv1'])):
            _i = _one(lambda l, p=_pre: l.startswith(p), _pre); _bl[_i] = _lv(_bl[_i], _names)
        for _sub, _names in (('{ envm_thl.slice(MynnRange{kts, kte}).linear(mynn_element) = thl1.slice(', ['thl1']),
                             ('{ envm_sqv.slice(MynnRange{kts, kte}).linear(mynn_element) = qv1.slice(', ['qv1'])):
            _i = _one(lambda l, p=_sub: p in l, _sub); _bl[_i] = _lv(_bl[_i], _names)
        # the plume's own readers of upthl, upqt, thln, qtn: frozen
        _i = _one(lambda l: l.startswith('f_condensation_edmf(qtn, thln, '), 'condensation call')
        _bl[_i] = _bl[_i].replace('f_condensation_edmf(qtn, thln, ', 'f_condensation_edmf(mw_frz(qtn), mw_frz(thln), ', 1)
        def _frz_reads(line, name):
            out, j, c = '', 0, 0
            while True:
                m = re.compile(r'\b%s\(' % name).search(line, j)
                if not m: return out + line[j:], c
                d, e = 1, m.end()
                while d: d += {'(': 1, ')': -1}.get(line[e], 0); e += 1
                out += line[j:m.start()] + 'mw_frz(' + line[m.start():e] + ')'; j = e; c += 1
        for _pre, _name in (('edmf_qt1(k) = r6_add(edmf_qt1(k), ', 'upqt'), ('edmf_thl1(k) = r6_add(edmf_thl1(k), ', 'upthl')):
            _h = [i for i in range(_f0[0], _f1) if _bl[i].startswith(_pre)]
            if len(_h) != 3: raise SystemExit('plume-live momheat: %s lines %d' % (_pre, len(_h)))
            for _i in _h:
                _bl[_i], _c = _frz_reads(_bl[_i], _name)
                if _c < 1: raise SystemExit('plume-live momheat: no %s read in %r' % (_name, _bl[_i][:70]))
        # no other line of f_dmp_mf may read the live carriers
        _ok = ('upthl(', 'upqt(', 'qtn = ', 'thln = ', 'qt_ent = qtn;', 'thl_ent = thln;', 'envm_thl(k) = ', 'envm_sqv(k) = ', 'qv_ent = ', 's_awthl1(k) = ', 's_awqt1(k) = ',
               'f_condensation_edmf(mw_frz(qtn), mw_frz(thln), ', 'edmf_qt1(k) = ', 'edmf_thl1(k) = ', 'MR ', 'MynnArray<MR, 2> ')
        _zero = re.compile(r'^mynn_lane_sync\(\); for \(int mynn_element=lane_sub; mynn_element < \((upthl|upqt)\)\.size\(\); mynn_element \+= lane_width\) \{ (upthl|upqt)\.linear\(mynn_element\) = zero; \} mynn_lane_sync\(\);$')
        _bad = [_bl[_i] for _i in range(_f0[0] + 1, _f1)
                if re.search(r'\b(upthl|upqt|thln|qtn|thl_ent|qt_ent|qv_ent)\b', _bl[_i]) and not _bl[_i].startswith(_ok) and not _bl[_i].startswith('//') and not _zero.match(_bl[_i])]
        if _bad: raise SystemExit('plume-live momheat: unlisted readers of the plume heat or moisture:\n' + '\n'.join(l[:400] for l in _bad))
        # the live views of the environment, after the rebase block of f_dmp_mf (M4 puts its frozen copies after them)
        _fr = next(i for i in range(_f0[0], _f1) if _bl[i] == 'MynnFrame mynn_frame(arena);') + 1
        while re.match(r'^\w+\.rebase\(', _bl[_fr]): _fr += 1
        _bl[_fr:_fr] = ['MynnArray<MR, 1> %s_lv = %s;' % (n, n) for n in ('thl1', 'qt1', 'qv1')]
        body = '\n'.join(_bl)
    if FREEZE == 'closure+cloud':
        # subgrid cloud closure and the surface-layer similarity functions as well
        FREEZE_AFTER['f_mym_condensation'] = ['qc_bl1', 'qi_bl1', 'cldfra_bl1', 'vt', 'vq', 'sgm']
        FREEZE_AFTER['f_dmp_mf'] = FREEZE_AFTER['f_dmp_mf'] + ['qc_bl1', 'cldfra_bl1']
        _cond = [l for l in body.splitlines() if l.startswith('f_mym_condensation(')]
        assert len(_cond) == 1, len(_cond)
        FREEZE_STMT.append(('f_mynnedmf', _cond[0], 'before',
                            _wrap(12, 'rmol = mw_frz(rmol); rmolh = mw_frz(rmolh); pmz = mw_frz(pmz); phh = mw_frz(phh); '
                            'phi_m = mw_frz(phi_m);'), 1))
FROZEN = {'freeze': FREEZE, 'after_call': {}, 'statements': []}
if FREEZE != 'none':
    sigs = {f: _sig(f) for f in FREEZE_AFTER}
    fn, hits, out_lines = None, {}, []
    for line in body.splitlines():
        m = re.match(r'^__device__ __noinline__ (?:void|MR) (f_\w+)\(', line)
        if m: fn = m.group(1)
        pre_code, post_code = [], []
        for (f, exact, where, code, _) in FREEZE_STMT:
            if f == fn and line == exact:
                (pre_code if where == 'before' else post_code).append(code)
                hits[(f, exact)] = hits.get((f, exact), 0) + 1
        cm = re.match(r'^(f_\w+)\((.*)\);$', line)
        if cm and cm.group(1) in FREEZE_AFTER:
            callee, args = cm.group(1), _params(cm.group(2))
            ps = sigs[callee]
            if len(args) != len(ps): raise SystemExit('arity ' + callee)
            code, cbits = [], []
            for name in FREEZE_AFTER[callee]:
                k = [i for i, (_, n) in enumerate(ps) if n == name]
                if len(k) != 1: raise SystemExit('param %s of %s' % (name, callee))
                t, a = ps[k[0]][0], args[k[0]]
                if not re.fullmatch(r'\w+', a):
                    raise SystemExit('frozen output %s of %s is not a variable: %s' % (name, callee, a))
                if MIX_LIVE and MIX_LIVE['fl'] > 0.0 and callee == 'f_mym_length' and fn == 'f_mym_turbulence' and name == 'el':
                    code.append(('mw_scl_arr(%s, %r);' % (a, MIX_LIVE['fl'])) if MIX_LIVE['lmode'] == 't' else '/* el keeps its derivative to the diffusivities (--mix-live l) */')
                elif t.startswith('MynnArray<'): code.append('mw_frz_arr(%s);' % a)
                elif t in ('MR&', 'MRD&'): code.append('%s = mw_frz(%s);' % (a, a))
                else: raise SystemExit('frozen output %s of %s has type %s' % (name, callee, t))
                cbits.append(_after_bit(callee, name))
            if GROUPED:
                for b in sorted(set(cbits)):
                    post_code.append(_wrap(b, ' '.join(c for c, cb in zip(code, cbits) if cb == b)))
            else:
                post_code.append(' '.join(code))
            FROZEN['after_call'].setdefault(callee, {'params': FREEZE_AFTER[callee], 'call_sites_in': []})['call_sites_in'].append(fn)
        elif re.match(r'^\s*(f_\w+)\(', line) and re.match(r'^\s*(f_\w+)\(', line).group(1) in FREEZE_AFTER:
            raise SystemExit('unparsed call site: ' + line[:120])
        out_lines.extend(pre_code); out_lines.append(line); out_lines.extend(post_code)
    for (f, exact, where, code, n) in FREEZE_STMT:
        if hits.get((f, exact), 0) != n:
            raise SystemExit('freeze anchor count %s %r: %d' % (f, exact, hits.get((f, exact), 0)))
        FROZEN['statements'].append({'function': f, where: exact[:80], 'code': code})
    for callee in FREEZE_AFTER:
        if callee not in FROZEN['after_call']: raise SystemExit('no call site of ' + callee)
    body = '\n'.join(out_lines)

any_array = r'''
// Automatic arrays of the generated column: same arena discipline as the forward (warp tile of 32 columns,
// lane = column within the tile). Elements start at zero with no derivative; an arena overrun is a refusal.
__device__ unsigned long long mw_arena_limit;
__device__ unsigned long long* mw_arena_hw;
template<class T,int R,int TILE> __device__ MynnArray<T,R> mynn_array_any(MynnArena& arena, MynnIndices<R> lo, MynnIndices<R> hi) {
    MynnArray<T,R> result; int count=1;
    for (int d=0; d<R; ++d) { result.lo[d]=lo.value[d]; result.hi[d]=hi.value[d]; result.stride[d]=count*TILE; count *= result.extent(d); }
    { const unsigned m = __activemask(); if (__match_any_sync(m, arena.offset) != m && mw_fail) mw_fail[(blockIdx.x*blockDim.x+threadIdx.x)*4+3] = 1; }
    result.data = static_cast<T*>(arena.allocate(sizeof(T)*count, sizeof(T)));
    if (mw_arena_hw) { unsigned long long* h = mw_arena_hw + (blockIdx.x*blockDim.x+threadIdx.x); if (arena.offset > *h) *h = arena.offset; }
    if (arena.offset > mw_arena_limit) {
        if (mw_fail) mw_fail[(blockIdx.x*blockDim.x+threadIdx.x)*4+3] = 1;
#ifdef MW_DEBUG
        printf("arena overflow thread %d offset %llu count %d\n", (int)(blockIdx.x*blockDim.x+threadIdx.x), arena.offset, count);
#endif
        __trap();
    }
    for (int q=0; q<count; ++q) result.data[q*TILE] = T();
    return result;
}
'''

MW_BOUNDS_DECL = """__device__ int* mw_fail;
// warp-tile interleave guard: co-active columns allocate at one offset, else the column is refused (arena slot)
#define MW_LOCKSTEP_CHECK(ar) { const unsigned m_ = __activemask(); if (__match_any_sync(m_, (ar).offset) != m_ && mw_fail) mw_fail[(blockIdx.x*blockDim.x+threadIdx.x)*4+3] = 1; }
#ifdef MW_BOUNDS
__device__ int* mw_bounds_rec;
// debug build only: record the first out-of-bounds index per thread and keep the access inside the array
__device__ __noinline__ void mw_bounds_check(int* ix, const int* lo, const int* hi, int R) {
    for (int d=0; d<R; ++d) if (ix[d] < lo[d] || ix[d] > hi[d]) {
        if (mw_bounds_rec) {
            int* b = mw_bounds_rec + 8*(blockIdx.x*blockDim.x+threadIdx.x);
            if (b[0]++ == 0) { b[1] = R; b[2] = d; b[3] = ix[d]; b[4] = lo[d]; b[5] = hi[d]; b[6] = b[7]; }
        }
        ix[d] = ix[d] < lo[d] ? lo[d] : hi[d];
    }
}
#endif
"""
MW_BOUNDS_ACC = """        int ix[] = {int(indices)...};
#ifdef MW_BOUNDS
        mw_bounds_check(ix, lo, hi, R);
#endif
        int offset = 0;"""
# ---- 2b. bounds check of every array access in MW_BOUNDS debug builds (reports, does not change values) -----
_acc = """        int ix[] = {int(indices)...};
        int offset = 0;"""
assert _acc in pre, 'MynnArray accessor not found'
pre = pre.replace(_acc, MW_BOUNDS_ACC, 1)
pre = pre.replace("template<class T, int R> struct MynnArray {", MW_BOUNDS_DECL + "template<class T, int R> struct MynnArray {", 1)

# ---- 2c. MW_INT_ZERO experiment builds only: integer automatic arrays start at zero (the forward leaves them as the
#          arena's previous bytes; with derivative carriers those bytes are pointer and double words)
_ia = """    result.data = static_cast<T*>(arena.allocate(sizeof(T)*count, sizeof(T)));
    return result;
}"""
assert _ia in pre, 'integer arena allocator not found'
# mynn-tape (2026-11-08): the warp tile (columns per warp, the arena interleave width) is MW_TILE, a build-time
# constant (default 32, the forward's own); an image built with -DMW_TILE=8 runs 8-thread blocks, each block one
# warp owning its own 8-column arena tile, so lockstep allocation still holds. Logical offsets are unchanged, so every
# word is unchanged. The image reports MW_TILE through mw_sizes and the host launches that many threads per block.
_tile = 'static constexpr int lane_width = MYNN_LANES; static constexpr int mynn_tile = 32 / MYNN_LANES;'
assert body.count(_tile) == 1, 'mynn_tile definition not found'
body = body.replace(_tile, 'static constexpr int lane_width = MYNN_LANES; static constexpr int mynn_tile = MW_TILE / MYNN_LANES;', 1)
pre = '#ifndef MW_TILE\n#define MW_TILE 32\n#endif\n' + pre
pre = pre.replace(_ia, """    MW_LOCKSTEP_CHECK(arena);
    result.data = static_cast<T*>(arena.allocate(sizeof(T)*count, sizeof(T)));
#ifdef MW_INT_ZERO
    for (int q=0; q<count; ++q) result.data[q*TILE] = T();
#endif
    return result;
}""", 1)

# ---- 2d. a12 truephys-term: exact derivative form of the level-2 stability quotients (--l2-quot, above) ---------
L2_RF = 'r6_mul(ri1, r6_sub(r6_add(ri(k), ri2), mynn_sqrt(r6_add(r6_sub(mynn_ipow<2>(ri(k)), r6_mul(ri3, ri(k))), ri4))))'
L2_SH = 'sh(k) = r6_div(r6_mul(shc, r6_sub(rfc, rf)), r6_sub(one, rf));'
L2_SM = 'sm(k) = r6_mul(r6_div(r6_mul(smc, r6_sub(rf1, rf)), r6_sub(rf2, rf)), sh(k));'
if L2_QUOT == 'exact':
    for _old, _new in ((L2_RF, 'mw_l2_rf(ri1, ri(k), ri2, ri3, ri4)'), (L2_SH, 'sh(k) = mw_l2_sh(shc, rfc, rf, one);'), (L2_SM, 'sm(k) = mw_l2_sm(smc, rf1, rf2, rf, sh(k));')):
        if body.count(_old) != 1: raise SystemExit('--l2-quot exact: anchor count %r: %d' % (_old, body.count(_old)))
        body = body.replace(_old, _new)

# ---- 2e. a12 truephys lead: hold of the level-2.5 stability quotients (--l25-quot hold, above) -------------------
L25_SM = 'sm(k) = mynn_ddiv(mynn_dmul(mynn_dmul(q3sq, a1), mynn_dsub(e3, mynn_dmul(r6_mul(three, c1), e4))), eden);' + chr(10)
L25_SH = 'mynn_dmul(q3sq, r6_mul(a2, a2fac)))), eden);' + chr(10)
L25_SM_X = 'sm(k) = mw_l25_sm(q3sq, ghel, gmel, a2fac, e2, e3, e4, eden, a1, r6_mul(three, c1), e1c, e2c, e3c, e4c, e5c);'
L25_SH_F = ('sh(k) = mynn_ddiv(mynn_dmul((tke_budget == 1 ? mynn_dmul(q3sq, r6_mul(a2, a2fac)) : mynn_dadd(mynn_dmul(r6_mul(r6_mul(three, c1), e5c), gmel), e2)), '
            '(tke_budget == 1 ? mynn_dadd(mynn_dmul(r6_mul(r6_mul(three, c1), e5c), gmel), e2) : mynn_dmul(q3sq, r6_mul(a2, a2fac)))), eden);')
L25_SH_X = 'sh(k) = mw_l25_sh(tke_budget, q3sq, ghel, gmel, a2fac, e2, e3, e4, eden, r6_mul(r6_mul(three, c1), e5c), r6_mul(a2, a2fac), e1c, e2c, e3c, e4c, e5c);'
if L25_QUOT == 'exact':
    if FREEZE != 'none': raise SystemExit('--l25-quot exact needs --freeze none')
    for _old, _new in ((L25_SM.rstrip(chr(10)), L25_SM_X), (L25_SH_F, L25_SH_X)):
        if body.count(_old) != 1: raise SystemExit('--l25-quot exact: anchor count %r: %d' % (_old[:60], body.count(_old)))
        body = body.replace(_old, _new)
# ---- 2f. bl-guard: the weak-shear rule (--weakshear, above), before the diffusivities of f_mym_turbulence ---------
WS_ANCHOR = 'elq = r6_mul(el(k), qkw(k));'
if WEAKSHEAR is not None and (FREEZE == 'none' or GROUPED):  # also with the freeze groups
    _ws = ('{ const bool mw_ws_hold = mw_pv(gm(k)) < (double)(%r); '
           'if (mw_ws_hold) { sm(k) = mw_frz(sm(k)); sh(k) = mw_frz(sh(k)); qdiv = mw_frz(qdiv); gamt = mw_frz(gamt); '
           'gamq = mw_frz(gamq); gamv = mw_frz(gamv); } }' % WEAKSHEAR)
    _fn, _hits, _out = None, 0, []
    for _line in body.splitlines():
        _m = re.match(r'^__device__ __noinline__ (?:void|MR) (f_\w+)\(', _line)
        if _m: _fn = _m.group(1)
        if _fn == 'f_mym_turbulence' and _line == WS_ANCHOR:
            _out.append('// bl-guard weak shear (gen_whole.py --weakshear %r): sm, sh and their terms carry no derivative where gm(k) < %r s-2' % (WEAKSHEAR, WEAKSHEAR))
            _out.append(_ws); _hits += 1
        _out.append(_line)
    if _hits != 1: raise SystemExit('--weakshear: anchor %r in f_mym_turbulence found %d times' % (WS_ANCHOR, _hits))
    body = '\n'.join(_out)
    FROZEN['statements'].append({'function': 'f_mym_turbulence', 'before': WS_ANCHOR, 'code': _ws})
if L25_QUOT == 'hold':
    if FREEZE != 'none': raise SystemExit('--l25-quot hold needs --freeze none')
    for _old, _new in ((L25_SM, 'sm(k) = mw_frz(sm(k)); // a12 truephys-card: hold of F90:2874 sm (level 2.5, over eden)'),
                       (L25_SH, 'sh(k) = mw_frz(sh(k)); // a12 truephys-card: hold of F90:2878 sh (level 2.5, over eden)')):
        if body.count(_old) != 1: raise SystemExit('--l25-quot hold: anchor count %r: %d' % (_old, body.count(_old)))
        body = body.replace(_old, _old + _new + chr(10))

# ---- 3. the launch wrapper: arrays, scalars and the driver call, read from the captured kernel -----
wrap = tail[tail.index('template<int MYNN_LANES> __device__ __forceinline__ void rrfs_mynnedmf_column('):]
sig = re.search(r'rrfs_mynnedmf_column\((.*?)\)\s*\{', wrap, re.S).group(1)
params = [p.strip() for p in sig.split(',')]
call = re.search(r'column\.' + TOP + r'\((.*?)\);\n', wrap, re.S).group(1)
arrays = []   # (name, dtype, rank, [(lo, hi, stride)])
for m in re.finditer(r'MynnArray<(float|int),(\d)> a_(\w+);a_\3\.data=\3;\n((?:a_\3\.lo\[\d\]=[^\n]*\n)+)', wrap):
    dims = re.findall(r'a_\w+\.lo\[(\d)\]=([^;]+);a_\w+\.hi\[\1\]=([^;]+);a_\w+\.stride\[\1\]=([^;]+);', m.group(4))
    arrays.append((m.group(3), m.group(1), int(m.group(2)), [(lo, hi, st) for _, lo, hi, st in dims]))
names = {a[0] for a in arrays}
scalars = [p for p in params if not any(p.endswith('* ' + n) or p.endswith('*' + n) for n in names)
           and p.split()[-1] not in ('workspace', 'error_flags')]
farr = [a for a in arrays if a[1] == 'float']
iarr = [a for a in arrays if a[1] == 'int']
assert len(iarr) == 1 and iarr[0][0] == 'kpbl', iarr

def col_extent_expr(dims):
    terms = [f'(({hi})-({lo})+1)' for lo, hi, st in dims if lo.strip() not in ('ims', 'jms')]
    return '*'.join(terms) if terms else '1'

def build_view(a, T, base):
    """Emit code making MynnArray<T,R> a_<name> backed by this column's packed storage `base`,
    plus the global-offset function for packed element q."""
    name, dt, rank, dims = a
    present = f'L->b[{farr.index(a)}].x' if dt == 'float' else 'L->kpbl_in'
    lines = [f'MynnArray<{T},{rank}> a_{name}; a_{name}.data = {present} ? {base} : nullptr;']
    cnt = '1'
    for d, (lo, hi, st) in enumerate(dims):
        lines.append(f'a_{name}.lo[{d}]={lo}; a_{name}.hi[{d}]={hi};')
        if lo.strip() in ('ims', 'jms'):
            lines.append(f'a_{name}.stride[{d}]=0;')
        else:
            lines.append(f'a_{name}.stride[{d}]={cnt};')
            cnt = f'({cnt})*(({hi})-({lo})+1)'
    return '\n'.join(lines)

def gofs(a):
    """Global element offset of packed element q of this column (forward's layout)."""
    name, dt, rank, dims = a
    parts, rem = [], 'q'
    code = [f'unsigned long long g=0; int r=q;']
    for d, (lo, hi, st) in enumerate(dims):
        if lo.strip() == 'ims':
            code.append(f'g += (unsigned long long)(active_i-({lo}))*(unsigned long long)({st});')
        elif lo.strip() == 'jms':
            code.append(f'g += (unsigned long long)(active_j-({lo}))*(unsigned long long)({st});')
        else:
            code.append(f'{{ int e=(({hi})-({lo})+1); g += (unsigned long long)(r%e)*(unsigned long long)({st}); r/=e; }}')
    return ' '.join(code)

NF = len(farr)
pre_kernel = []
pre_kernel.append(f'#define MW_NF {NF}')
pre_kernel.append('struct MwBind { const float* x; float* y; const double* tin; double* tout; const double* ybar; double* xbar; };')
pre_kernel.append('''struct MwLaunch {
  MwBind b[MW_NF];
  const int* kpbl_in; int* kpbl_out; int* error_flags;
  unsigned char* arena; unsigned long long arena_col_bytes;
  unsigned char* store; unsigned long long store_col_elems;
  int* leaf; int* kstore;
  MynnAdNode* nodes; double* bars; long long capacity; long long* used;
  int* failures;
  int col_first; int col_count; int mode; int layout;
};''')

# per-array: packed offset prefix sums computed at run time
ext = '\n'.join(f'const int ext_{a[0]} = {col_extent_expr(a[3])};' for a in farr)
offs, acc = [], '0'
for a in farr:
    offs.append(f'const unsigned long long off_{a[0]} = {acc};')
    acc = f'off_{a[0]} + (unsigned long long)ext_{a[0]}'
offs.append(f'const unsigned long long npack = {acc};')
offs = '\n'.join(offs)

scalar_params = ', '.join(scalars)
views = '\n'.join(build_view(a, 'F', f'colstore + off_{a[0]}') for a in farr)
kview = build_view(iarr[0], 'int', 'kcol')
load_lines, store_lines, seed_lines, leafout_lines = [], [], [], []
for i, a in enumerate(farr):
    n = a[0]
    load_lines.append(f'''if (L->b[{i}].x) for (int q=0; q<ext_{n}; ++q) {{ {gofs(a)}
  const MwBind& B=L->b[{i}]; F& e=colstore[off_{n}+q]; float xv=B.x[g];
  if constexpr(MODE==0) e = F(xv, D(B.tin ? B.tin[g] : 0.0, &ctx));
  else {{ int id=ctx.leaf(); if constexpr(MODE==1) leafs[off_{n}+q]=id; e = F(xv, D(id, &ctx)); }} }}''')
    store_lines.append(f'''if (L->b[{i}].x) for (int q=0; q<ext_{n}; ++q) {{ {gofs(a)}
  const MwBind& B=L->b[{i}]; const F& e=colstore[off_{n}+q];
  if (B.y) B.y[g]=e.v;
  if constexpr(MODE==0) {{ if (B.tout) B.tout[g]=e.d.v; }} }}''')
    seed_lines.append(f'''if (L->b[{i}].x) for (int q=0; q<ext_{n}; ++q) {{ {gofs(a)}
  const MwBind& B=L->b[{i}]; if (B.ybar) ctx.seed(colstore[off_{n}+q].d.id, B.ybar[g]); }}''')
    leafout_lines.append(f'''if (L->b[{i}].x) for (int q=0; q<ext_{n}; ++q) {{ {gofs(a)}
  const MwBind& B=L->b[{i}]; int id=leafs[off_{n}+q]; if (B.xbar) B.xbar[g] = id>=0 ? ctx.bars[id] : 0.0; }}''')
kp = iarr[0]
kload = f'''for (int q=0; q<{col_extent_expr(kp[3])}; ++q) {{ {gofs(kp)} kcol[q]=L->kpbl_in[g]; }}'''
kstore = f'''for (int q=0; q<{col_extent_expr(kp[3])}; ++q) {{ {gofs(kp)} if (L->kpbl_out) L->kpbl_out[g]=kcol[q]; }}'''

call_g = call  # same names: a_<name> views, scalars, message/error_flag
kernel = f'''
// mynn-adjoint-v2 (2026-11-08): col_local is the column of the launch; slot_local the thread's slot of the column
// buffers (store, leaf, kstore, arena). The one-shot entries pass both = their thread index, as before. The persistent
// entries (layout 2) give a block a fixed slot of MW_TILE columns and loop it over column groups; pb_base/pb_cap are
// then the column's tape start and node count (+16) in the block's tape region.
template<class F,class D,int MODE> __device__ __noinline__ void mynn_whole_column(const MwLaunch* L, int col_local, int slot_local, unsigned long long pb_base, long long pb_cap, {scalar_params}) {{
  if (col_local >= L->col_count) return;
  const int column_index = L->col_first + col_local;
#ifdef MW_BENCH_REPLICATE
  const int index = batch_first + column_index % batch_count;
#else
  if (column_index >= batch_count) return;
  const int index = batch_first + column_index;
#endif
  const int active_i = its + index%(ite-its+1);
  const int active_j = jts + index/(ite-its+1);
{ext}
{offs}
  if (npack > L->store_col_elems) {{ L->failures[col_local*4+3] = 1; return; }}
#ifdef MW_DEBUG
  if (col_local == 0) printf("mode %d: npack %llu\\n", MODE, npack);
#endif
  F* colstore = reinterpret_cast<F*>(L->store) + (unsigned long long)slot_local*L->store_col_elems;
  int* leafs = L->leaf ? L->leaf + (unsigned long long)slot_local*L->store_col_elems : nullptr;
  int* kcol = L->kstore + (unsigned long long)slot_local*8;
  int* lf = L->failures + col_local*4;
  // mynn-tape (2026-11-08): layout 1 = packed tape. L->used then holds col_count+1 node offsets, the exclusive prefix
  // sums of each column's measured node count (+16) from the count pass, so a column's tape is its own count and not
  // the tile's largest (pkg18m2: mean 116k against max 300-385k nodes). layout 0 = every column L->capacity nodes.
  const bool mw_packed = MODE==1 && L->layout == 1;
  const bool mw_pers = MODE==1 && (L->layout == 2 || L->layout == 3);
  const unsigned long long mw_base = mw_pers ? pb_base : mw_packed ? (unsigned long long)L->used[col_local] : (unsigned long long)col_local*L->capacity;
  const long long mw_cap = mw_pers ? pb_cap : mw_packed ? L->used[col_local+1] - L->used[col_local] : L->capacity;
  MynnAdContext ctx{{MODE==1 ? L->nodes + mw_base : nullptr,
                     MODE==1 ? L->bars + mw_base : nullptr,
                     MODE==2 ? 2147483647 : (int)mw_cap, 0, lf, MODE==2 ? 1 : 0}};
{chr(10).join(load_lines)}
{kload}
{views}
{kview}
  MynnArena arena{{L->arena + (unsigned long long)(slot_local - slot_local%MW_TILE)*L->arena_col_bytes, 0, MW_TILE, slot_local%MW_TILE}};
  MynnColumn<1, F, typename MwDouble<F>::type> column{{arena}};
  column.lane_sub = 0; column.lane_mask = 1u << (threadIdx.x & 31);
  column.active_i = active_i; column.active_j = active_j;
  MynnText message; int error_flag = 0;
#ifdef MW_DEBUG
  if (col_local == 0) printf("mode %d: inputs loaded, calling driver\\n", MODE);
#endif
  column.{TOP}({call_g});
#ifdef MW_DEBUG
  if (col_local == 0) printf("mode %d: driver returned, arena offset %llu\\n", MODE, arena.offset);
#endif
{chr(10).join(store_lines)}
{kstore}
  if (L->error_flags) L->error_flags[index] = error_flag;
  if constexpr(MODE==2) {{ if (L->used) L->used[col_local] = ctx.used; }}
  if constexpr(MODE==1) {{
{chr(10).join(seed_lines)}
    ctx.reverse();
{chr(10).join(leafout_lines)}
  }}
}}
template<class F> struct MwDouble;
template<> struct MwDouble<MynnTangentF> {{ using type = MynnTangentD; }};
template<> struct MwDouble<MynnTapeF> {{ using type = MynnTapeD; }};
'''
argnames = ', '.join(p.split()[-1] for p in scalars)

pers = f"""
// mynn-adjoint-v2 (2026-11-08): persistent entries. The runner launches as many blocks of MW_TILE threads as the memory
// budget and the card hold; each block owns one slot of MW_TILE columns (arena tile, store, leaf, kstore and, for the
// adjoint, a tape region of L->capacity nodes) and takes column groups of MW_TILE consecutive columns from mw_sched[0]
// until none is left. Per group it zeroes its arena tile and store (what the one-shot launch's host memset did) and
// runs the same column routine; columns are independent (G5), so the words are those of the one-shot entries.
// count_p: mw_sched[1] = the largest group tape (sum over the group of count + 16), the region the adjoint needs.
// ad_p (layout 2): a column's tape starts at block * capacity + the group's exclusive prefix of count + 16 (the
// count pass's L->used); a group whose sum exceeds the region fails its columns' overflow slot.
// ad_p (layout 3, mynn-adjoint-v2 ring tape): L->nodes/L->bars are ONE ring of L->capacity nodes shared by every block.
// The runner packs the groups in order from the count pass (exact: each group takes its own sum of count + 16, not the
// largest group's) and uploads mw_ring = vstart[ngroups + 1] (virtual start, monotone, a group never straddles the ring
// end), phys[ngroups] (vstart mod capacity) and done[ngroups] (u32, zeroed). Groups are taken in order; a block starts
// group g once vstart[g] + its sum <= vstart[low] + capacity, low = mw_sched[2] = the first group not yet done, so it
// never writes ring nodes an unfinished group still reads. The group holding `low` never waits, so the blocks always
// progress (every block that holds a group is resident). After its sweep a group sets done[g] and moves low forward.
__device__ unsigned long long* mw_sched;
__device__ unsigned long long* mw_ring;
__device__ __forceinline__ void mw_zero_bytes(unsigned char* p, unsigned long long n) {{
  unsigned long long* q = reinterpret_cast<unsigned long long*>(p);
  const unsigned long long w = n / 8ULL;
  for (unsigned long long k = threadIdx.x; k < w; k += blockDim.x) q[k] = 0ULL;
  for (unsigned long long k = w * 8ULL + threadIdx.x; k < n; k += blockDim.x) p[k] = 0;
}}
template<class F, class D, int MODE> __device__ __forceinline__ void mw_persist(const MwLaunch* L, {scalar_params}) {{
  const unsigned m = MW_TILE == 32 ? 0xffffffffu : ((1u << MW_TILE) - 1u);
  const int lane = threadIdx.x;
  const int slot_local = blockIdx.x * MW_TILE + lane;
  const int ngroups = (L->col_count + MW_TILE - 1) / MW_TILE;
  unsigned char* ar = L->arena + (unsigned long long)blockIdx.x * MW_TILE * L->arena_col_bytes;
  unsigned char* st = L->store + (unsigned long long)blockIdx.x * MW_TILE * L->store_col_elems * sizeof(F);
  for (;;) {{
    int g = 0;
    if (lane == 0) g = (int)atomicAdd(mw_sched, 1ULL);
    g = __shfl_sync(m, g, 0);
    if (g >= ngroups) break;
    const int col_local = g * MW_TILE + lane;
    mw_zero_bytes(ar, (unsigned long long)MW_TILE * L->arena_col_bytes);
    mw_zero_bytes(st, (unsigned long long)MW_TILE * L->store_col_elems * sizeof(F));
    __threadfence_block();
    __syncwarp(m);
    if constexpr (MODE == 1) {{
      const long long need = col_local < L->col_count ? L->used[col_local] + 16 : 0;
      long long inc = need;
      for (int d = 1; d < MW_TILE; d <<= 1) {{ const long long v = __shfl_up_sync(m, inc, d, MW_TILE); if (lane >= d) inc += v; }}
      const long long tot = __shfl_sync(m, inc, MW_TILE - 1, MW_TILE);
      unsigned long long gbase = (unsigned long long)blockIdx.x * (unsigned long long)L->capacity;
      if (L->layout == 3 && tot <= L->capacity) {{
        if (lane == 0) {{
          const unsigned long long vg = mw_ring[g];
          for (;;) {{
            const unsigned long long lo = *((volatile unsigned long long*)(mw_sched + 2));
            if (vg + (unsigned long long)tot <= mw_ring[lo] + (unsigned long long)L->capacity) break;
            __nanosleep(256);
          }}
          __threadfence();
        }}
        __syncwarp(m);
        gbase = mw_ring[ngroups + 1 + g];
      }}
      if (tot > L->capacity) {{ if (col_local < L->col_count) L->failures[col_local * 4] = 1; }}
      else mynn_whole_column<F, D, 1>(L, col_local, slot_local, gbase + (unsigned long long)(inc - need), need, {argnames});
      if (L->layout == 3) {{
        __syncwarp(m);
        if (lane == 0) {{
          unsigned* done = reinterpret_cast<unsigned*>(mw_ring + 2 * ngroups + 1);
          __threadfence();
          atomicExch(done + g, 1u);
          unsigned long long lo = atomicAdd(mw_sched + 2, 0ULL);
          while (lo < (unsigned long long)ngroups && atomicAdd(done + lo, 0u) != 0u) {{
            const unsigned long long was = atomicCAS(mw_sched + 2, lo, lo + 1);
            lo = was == lo ? lo + 1 : was;
          }}
        }}
      }}
    }} else {{
      mynn_whole_column<F, D, MODE>(L, col_local, slot_local, 0ULL, 0LL, {argnames});
      if constexpr (MODE == 2) {{
        __syncwarp(m);
        long long need = col_local < L->col_count ? L->used[col_local] + 16 : 0;
        for (int d = MW_TILE / 2; d > 0; d >>= 1) need += __shfl_down_sync(m, need, d, MW_TILE);
        if (lane == 0) atomicMax(mw_sched + 1, (unsigned long long)need);
      }}
    }}
    __syncwarp(m);
  }}
}}
"""
entries = pers + f'''
#if !defined(MW_ENTRY) || MW_ENTRY == 0
extern "C" __global__ void __launch_bounds__(32) mynn_whole_tl(const MwLaunch* L, {scalar_params}) {{ const int c = blockIdx.x*blockDim.x+threadIdx.x; mynn_whole_column<MynnTangentF, MynnTangent, 0>(L, c, c, 0ULL, 0LL, {argnames}); }}
extern "C" __global__ void __launch_bounds__(32) mynn_whole_tl_p(const MwLaunch* L, {scalar_params}) {{ mw_persist<MynnTangentF, MynnTangent, 0>(L, {argnames}); }}
#endif
#if !defined(MW_ENTRY) || MW_ENTRY == 1
extern "C" __global__ void __launch_bounds__(32) mynn_whole_ad(const MwLaunch* L, {scalar_params}) {{ const int c = blockIdx.x*blockDim.x+threadIdx.x; mynn_whole_column<MynnTapeF, MynnLinear, 1>(L, c, c, 0ULL, 0LL, {argnames}); }}
extern "C" __global__ void __launch_bounds__(32) mynn_whole_ad_p(const MwLaunch* L, {scalar_params}) {{ mw_persist<MynnTapeF, MynnLinear, 1>(L, {argnames}); }}
#endif
#if !defined(MW_ENTRY) || MW_ENTRY == 2
extern "C" __global__ void __launch_bounds__(32) mynn_whole_count(const MwLaunch* L, {scalar_params}) {{ const int c = blockIdx.x*blockDim.x+threadIdx.x; mynn_whole_column<MynnTapeF, MynnLinear, 2>(L, c, c, 0ULL, 0LL, {argnames}); }}
extern "C" __global__ void __launch_bounds__(32) mynn_whole_count_p(const MwLaunch* L, {scalar_params}) {{ mw_persist<MynnTapeF, MynnLinear, 2>(L, {argnames}); }}
#endif
extern "C" __global__ void mw_sizes(unsigned long long* o) {{ o[0]=sizeof(MynnTangentF); o[1]=sizeof(MynnTapeF); o[2]=sizeof(MynnAdNode); o[3]=sizeof(MwLaunch); o[4]=MW_TILE;
#ifdef MW_BENCH_REPLICATE
  o[5]=1;
#endif
}}
'''
# MwDouble must precede its use
kernel = kernel.replace('template<class F> struct MwDouble;\ntemplate<> struct MwDouble<MynnTangentF> { using type = MynnTangentD; };\ntemplate<> struct MwDouble<MynnTapeF> { using type = MynnTapeD; };\n', '')
traits = '''template<class F> struct MwDouble;
template<> struct MwDouble<MynnTangentF> { using type = MynnTangentD; };
template<> struct MwDouble<MynnTapeF> { using type = MynnTapeD; };
'''
INLINED = []
STATIC_TABLES = 0
if FRAME == 'inline1':
    # The captured glibc-exact library keeps seven lookup tables as function-scope const arrays (hex_glibc_log10f's
    # tr/tl of 65 doubles and its st powers of ten, b, c, and two rational-approximation tables), which every call
    # rebuilds in the thread's local frame (1,104 B of the derivative kernels' frame). static const keeps the same
    # words in global memory, built once. This copy of the prefix only: the forward image is not touched.
    pre, STATIC_TABLES = re.subn(r'^(\s+)const (double \w+\[\]\s*=|union\s*$)', r'\1static const \2', pre, flags=re.M)
    if STATIC_TABLES != 7:
        raise SystemExit('function-scope library tables: %d, expected 7' % STATIC_TABLES)
    for m in re.finditer(r'^__device__ __noinline__ void (f_\w+)\(', body, re.M):
        if len(re.findall(r'\b' + m.group(1) + r'\(', body + kernel)) == 2:   # its definition and one call site (the top helper's is in the kernel)
            INLINED.append(m.group(1))
    for n in INLINED:
        body, k = re.subn(r'^__device__ __noinline__ void ' + n + r'\(', '__device__ __forceinline__ void ' + n + '(', body, flags=re.M)
        assert k == 1, n
M8_SIG = ('int kts, int kte, MR closure, MR delt, MynnArray<MR, 1> dz, MR ust, MR flt, MR flq, MR pmz, MR phh, '
          'MynnArray<MR, 1> el, MynnArray<MR, 1> dfq, MynnArray<MR, 1> rho, MynnArray<MR, 1> pdk, MynnArray<MR, 1> pdt, '
          'MynnArray<MR, 1> pdq, MynnArray<MR, 1> pdc, MynnArray<MR, 1> qke, MynnArray<MR, 1> tsq, MynnArray<MR, 1> qsq, '
          'MynnArray<MR, 1> cov, MynnArray<MR, 1> s_aw1, MynnArray<MR, 1> s_awqke1, MynnArray<MR, 1> s_awqsq1, int bl_mynn_edmf, '
          'int bl_mynn_edmf_tke, MynnArray<MR, 1> qwt1, MynnArray<MR, 1> qdiss1, int tke_budget')
if 'm8' in FUSED:
    import re as _re
    _m = _re.search(r'^__device__ (__forceinline__|__noinline__) void f_mym_predict\((.*)\) \{$', body, _re.M)
    if not _m or _m.group(2) != M8_SIG:
        raise SystemExit('f_mym_predict signature changed; --fused m8 needs a review')
    _names = [p.rsplit(None, 1)[1] for p in _params(M8_SIG)]
    _arr = [p.rsplit(None, 1)[1] for p in _params(M8_SIG) if p.startswith('MynnArray<')]
    _fargs = []
    for _n in _names:
        if _n in _arr: _fargs.append('mw_f_' + _n)
        elif _n in ('kts', 'kte', 'bl_mynn_edmf', 'bl_mynn_edmf_tke', 'tke_budget'): _fargs.append(_n)
        else: _fargs.append(_n + '.v')
    _disp = ('__device__ __forceinline__ void f_mym_predict(' + M8_SIG + ') {\n'
             '// x20 M8 (gen_whole.py --fused m8): the adjoint and count entries record one external tape record\n'
             '// (cuda/mw_m8_predict.cuh); the primal is this routine on float copies (bitwise the forward).\n'
             'if constexpr (mw_m8_is_tape<MR>::v) {\n'
             'if (mw_m8_ok(kts, kte, closure, delt, bl_mynn_edmf, bl_mynn_edmf_tke, tke_budget)) {\n'
             'MynnFrame mynn_frame(arena);\n'
             + ''.join('%s.rebase({kts});\n' % a for a in _arr) +
             'const int mw_mk = mw_m8_record_in(kte, bl_mynn_edmf, delt, ust, pmz, dz, rho, el, dfq, s_aw1, pdk, pdq, pdt, pdc, qke, qsq);\n'
             + ''.join('MynnArray<float, 1> mw_f_%s = mw_m8_f32<mynn_tile>(arena, %s);\n' % (a, a) for a in _arr) +
             'MynnColumn<1, float, double> mw_fc{arena};\n'
             'mw_fc.lane_sub = lane_sub; mw_fc.lane_mask = lane_mask; mw_fc.active_i = active_i; mw_fc.active_j = active_j;\n'
             'mw_fc.f_mym_predict_taped(' + ', '.join(_fargs) + ');\n'
             'mw_m8_record_out(mw_mk, kte, pdk, pdt, pdq, pdc, qke, tsq, qsq, cov, mw_f_pdk, mw_f_pdt, mw_f_pdq, mw_f_pdc, '
             'mw_f_qke, mw_f_tsq, mw_f_qsq, mw_f_cov);\n'
             'return;\n}\n}\n'
             'f_mym_predict_taped(' + ', '.join(_names) + ');\n}\n')
    body = body[:_m.start()] + _disp + body[_m.start():_m.end()].replace('void f_mym_predict(', 'void f_mym_predict_taped(', 1) + body[_m.end():]
    _rv = '    ctx.reverse();\n'
    assert kernel.count(_rv) == 1
    kernel = kernel.replace(_rv, '    ctx.scr = L->arena + (unsigned long long)(slot_local - slot_local%MW_TILE)*L->arena_col_bytes; '
                            'ctx.scr_cap = L->arena_col_bytes; ctx.scr_lane = slot_local%MW_TILE;\n' + _rv, 1)
header = ('// GENERATED by tree/ad/phys/mynn/gen_whole.py from the captured hex MYNN-EDMF c6d08709 source\n'
          f'// sha256 {CAPTURE} (hex pin 16251589). Do not edit; regenerate.\n'
          '// Upstream notices of the captured source are carried in the prefix below.\n'
          + ('' if FREEZE == 'none' else '// Closure frozen from the forward trajectory in TL/AD (gen_whole.py --freeze %s).\n' % FREEZE)
          + ('' if not CORRECT else '// Correct mode: fork corrections %s applied to the captured source (gen_whole.py --correct).\n'
             % ','.join(CORRECT))
          + ('' if not PLUME_LIVE else '// Live plume momentum (gen_whole.py --plume-live %s): s_awu/v, sub_u/v, det_u/v answer u1, v1 with held coefficients; the rest of the plume frozen after the call.\n' % PLUME_LIVE)
          + ('' if not PLUME_HOLD else '// Plume holds (gen_whole.py --plume-hold %s): derivative held, forward value untouched.\n' % ','.join(PLUME_HOLD))
          + ('' if not PLUME_W_SMOOTH else '// Smooth plume speed (gen_whole.py --plume-w-smooth %s): derivative of wn times a factor of the forward state only, forward value untouched.\n' % ':'.join(PLUME_W_SMOOTH))
          + ('' if not MIX_LIVE else '// Bounded live mixing (gen_whole.py --mix-live %s): derivative of the diffusivities kept times factors of the forward state only, forward value untouched.\n' % MIX_LIVE['spec'])
          + ('' if not FUSED else '// Fused stages (gen_whole.py --fused %s): external tape records, hand reverse.\n' % ','.join(FUSED))
          + ('' if L2_QUOT != 'exact' else '// Level-2 quotients (gen_whole.py --l2-quot exact): closed-form partials of rf, sh and sm, value bitwise.\n')
          + ('' if L25_QUOT != 'exact' else '// Level-2.5 quotients (gen_whole.py --l25-quot exact): closed-form partials of sm and sh, value bitwise.\n')
          + ('' if WEAKSHEAR is None else '// Weak-shear rule (gen_whole.py --weakshear %r): stability functions held where the squared shear is under it, true derivative elsewhere.\n' % WEAKSHEAR)
          + ('' if FRAME == 'noinline' else '// Frame (gen_whole.py --frame %s): %d one-call-site helpers inlined, %d library tables static.\n' % (FRAME, len(INLINED), STATIC_TABLES))
          + ('' if not GROUPED else '// Freeze groups (gen_whole.py --freeze-mask 0x%x%s, lane S11).\n#define MW_FRZ_MASK 0x%xu\n'
             '#define MW_FRZ_ON(b) (((MW_FRZ_MASK) >> (b)) & 1u)\n' % (FRZ_MASK, '' if GATE_Q is None else ' --gate-q ' + GATE_Q, FRZ_MASK)
             + ('' if GATE_Q is None else '#define MW_GATE_Q %s\n#define MW_GATE_LO(x) (mw_pv(x) < (double)(MW_GATE_Q))\n' % GATE_Q)))
MW_PV = '''
// Primal value of a carrier (lane S11 gate): the comparison reads no derivative and records no branch bit.
template<class T> __device__ __forceinline__ double mw_pv(T x) { return (double)x; }
__device__ __forceinline__ double mw_pv(MynnTangentF a) { return (double)a.v; }
__device__ __forceinline__ double mw_pv(MynnTangentD a) { return a.v; }
__device__ __forceinline__ double mw_pv(MynnTapeF a) { return (double)a.v; }
__device__ __forceinline__ double mw_pv(MynnTapeD a) { return a.v; }
''' if (GROUPED or WEAKSHEAR is not None) else ''
MW_SCL = '''
// Smooth plume speed (gen_whole.py --plume-w-smooth): mw_spv is the forward value of a carrier (no derivative read, no
// branch bit); mw_wphi the factor, a function of forward values only; mw_scl keeps the value bitwise and scales the
// derivative (one linear tape node in the adjoint and count entries, the same in both).
template<class T> __device__ __forceinline__ double mw_spv(T x) { return (double)x; }
__device__ __forceinline__ double mw_spv(MynnTangentF a) { return (double)a.v; }
__device__ __forceinline__ double mw_spv(MynnTangentD a) { return a.v; }
__device__ __forceinline__ double mw_spv(MynnTapeF a) { return (double)a.v; }
__device__ __forceinline__ double mw_spv(MynnTapeD a) { return a.v; }
__device__ __forceinline__ double mw_wphi(double wn, double wb) { const double m = wn < wb ? wn : wb; if (!(m > 0.0)) return 0.0; %s }
template<class T> __device__ __forceinline__ T mw_scl(T x, double) { return x; }
__device__ __forceinline__ MynnTangentF mw_scl(MynnTangentF a, double f) { return MynnTangentF(a.v, mynn_ad_depend(a.d * f, a.d.depends)); }
__device__ __forceinline__ MynnTangentD mw_scl(MynnTangentD a, double f) { return MynnTangentD(a.v, mynn_ad_depend(a.d * f, a.d.depends)); }
__device__ __forceinline__ MynnTapeF mw_scl(MynnTapeF a, double f) { return MynnTapeF(a.v, mynn_ad_depend(a.d * f, a.d.depends)); }
__device__ __forceinline__ MynnTapeD mw_scl(MynnTapeD a, double f) { return MynnTapeD(a.v, mynn_ad_depend(a.d * f, a.d.depends)); }
''' % MW_WPHI if PLUME_W_SMOOTH else ''
if MIX_LIVE:
    # strike physics bl: the same value/scale helpers without the plume factor when --plume-w-smooth is not given
    if not PLUME_W_SMOOTH:
        MW_SCL = '''
// Bounded live mixing (gen_whole.py --mix-live): mw_spv is the forward value of a carrier (no derivative read, no branch
// bit); mw_scl keeps the value bitwise and scales the derivative (one linear tape node in the adjoint and count entries).
template<class T> __device__ __forceinline__ double mw_spv(T x) { return (double)x; }
__device__ __forceinline__ double mw_spv(MynnTangentF a) { return (double)a.v; }
__device__ __forceinline__ double mw_spv(MynnTangentD a) { return a.v; }
__device__ __forceinline__ double mw_spv(MynnTapeF a) { return (double)a.v; }
__device__ __forceinline__ double mw_spv(MynnTapeD a) { return a.v; }
template<class T> __device__ __forceinline__ T mw_scl(T x, double) { return x; }
__device__ __forceinline__ MynnTangentF mw_scl(MynnTangentF a, double f) { return MynnTangentF(a.v, mynn_ad_depend(a.d * f, a.d.depends)); }
__device__ __forceinline__ MynnTangentD mw_scl(MynnTangentD a, double f) { return MynnTangentD(a.v, mynn_ad_depend(a.d * f, a.d.depends)); }
__device__ __forceinline__ MynnTapeF mw_scl(MynnTapeF a, double f) { return MynnTapeF(a.v, mynn_ad_depend(a.d * f, a.d.depends)); }
__device__ __forceinline__ MynnTapeD mw_scl(MynnTapeD a, double f) { return MynnTapeD(a.v, mynn_ad_depend(a.d * f, a.d.depends)); }
'''
    _q0 = MIX_LIVE['q0']
    _mq = ('return %r;' % MIX_LIVE['fq']) if _q0 is None else ('if (!(q > 0.0)) return 0.0; return %r * q*q / (q*q + (%r)*(%r));' % (MIX_LIVE['fq'], _q0, _q0))
    _ms = {'a': 'return %r;', 'u': 'return gh > 0.0 ? %r : 0.0;', 'st': 'return gh > 0.0 ? 0.0 : %r;'}[MIX_LIVE['smode']] % MIX_LIVE['fs']
    MW_SCL = MW_SCL + '''
// Bounded live mixing (gen_whole.py --mix-live %s): the factor on the turbulence-energy path of the diffusivities (of the
// forward turbulence speed q only) and on the stability path (of the forward sign of gh only).
__device__ __forceinline__ double mw_mixq(double q) { %s }
__device__ __forceinline__ double mw_mixs(double gh) { %s }
''' % (MIX_LIVE['spec'], _mq, _ms)
    if MIX_LIVE['fl'] > 0.0:
        MW_SCL = MW_SCL + '''// Scale the derivative of every element of an array, values kept bitwise (--mix-live l:FL:t).
template<class T, int R> __device__ __noinline__ void mw_scl_arr(MynnArray<T, R> a, double f) {
    if (!a.data) return;
    const int n = a.size();
    for (int q = 0; q < n; ++q) a.linear(q) = mw_scl(a.linear(q), f);
}
'''
MW_PV = MW_PV + MW_SCL
if FUSED:
    outtxt = (header + pre + '\n#define MW_EXT_REV 1\n#include "mw_types.cuh"\n' + MW_PV + any_array + '\n#include "mw_m8_predict.cuh"\n'
              + body + '\n#define MW_M8_PART2\n#include "mw_m8_predict.cuh"\n#undef MW_M8_PART2\n'
              + '\n'.join(pre_kernel) + '\n' + traits + kernel + entries)
else:
    outtxt = (header + pre + '\n#include "mw_types.cuh"\n' + MW_PV + any_array + '\n' + body + '\n'
              + '\n'.join(pre_kernel) + '\n' + traits + kernel + entries)
out.write_text(outtxt)
info = {'top': TOP, 'helpers': sorted(seen), 'noinline_helpers': n_noinline, 'float_arrays': [a[0] for a in farr], 'int_arrays': [a[0] for a in iarr],
        'scalars': scalars, 'output': str(out), 'output_sha256': hashlib.sha256(outtxt.encode()).hexdigest(),
        'source_sha256': CAPTURE, 'frozen': FROZEN, 'frame': FRAME, 'inlined_helpers': INLINED, 'static_tables': STATIC_TABLES}
if CORRECT:
    info['corrections'] = CORRECT
if FUSED:
    info['fused'] = FUSED
if PLUME_LIVE:
    info['plume_live'] = PLUME_LIVE
if PLUME_HOLD:
    info['plume_hold'] = PLUME_HOLD
if PLUME_W_SMOOTH:
    info['plume_w_smooth'] = ':'.join(PLUME_W_SMOOTH)
if MIX_LIVE:
    info['mix_live'] = MIX_LIVE['spec']
if GROUPED:
    info['freeze_groups'] = {'mask': hex(FRZ_MASK), 'gate_q': GATE_Q, 'groups': MW_FRZ_GROUPS}
if L2_QUOT == 'exact':
    info['l2_quot'] = L2_QUOT
if L25_QUOT in ('hold', 'exact'):
    info['l25_quot'] = L25_QUOT
if WEAKSHEAR is not None:
    info['weakshear'] = WEAKSHEAR
print(json.dumps(info))
