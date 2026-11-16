#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""F08 (15 Nov): the TEMPO PBL subgrid radii in the taped whole-launch derivative (tempo_adw_record).

tempo_warp_steps runs `if(ml_bl && has_bl && re) tempo_subgrid_cloud_radii(...)` after tempo_main_column on the last
substep of a radii launch (NOAA's hrrrv5 deck: config_tempo_ml_for_bl_nc = .true., bl = qc_bl, cldfrac_bl from the
boundary layer). The record entry did not, so the runner refused such a launch (NOAA 3 km window, step 59 of 59).

This inserts the same call, with the forward's own arguments, into the record entry: bl is read from a module global
g_tempo_bl that the runner points at its snapshot of bl ([2][nz][columns]) on exactly those launches (0 otherwise, so
every other launch records the same graph as before). bl enters as constants: the TEMPO input vector has no bl slots,
so its slope is held (the radii still answer qc, qr, qi, qs, w and the work state through the subgrid path).

  f08_subgrid.py FILE [FILE ...]     edits each file in place (the generator's driver_warp.cuh, truephys split2.main.cu,
                                     or a generated split2-*.main.cu); refuses if an anchor is missing or already patched.
"""
import sys

GLOBAL_ANCHOR = 'extern "C" __global__ void __launch_bounds__(AD_BLOCK) tempo_adw_record('
GLOBAL_TEXT = (
    '// F08: the PBL subgrid radii. The runner points g_tempo_bl at its snapshot of the launch\'s bl ([2][nz][columns]:\n'
    '// qc_bl, cldfrac_bl) on a launch that takes tempo_warp_steps\' `ml_bl && has_bl && re` path, and at 0 otherwise.\n'
    '__device__ const float* g_tempo_bl = nullptr;\n'
)
CALL_ANCHOR = (
    '                           st.f[0], st.f[1], st.f[2], st.f[3]);\n'
    '    ad_sync();\n'
    '    // tempo_step_diags, in its order of operations (lead lane).\n'
)
CALL_TEXT = (
    '                           st.f[0], st.f[1], st.f[2], st.f[3]);\n'
    '    ad_sync();\n'
    '    // F08: tempo_warp_steps\' PBL subgrid radii, the forward\'s own call and arguments (b = bl[x], b + nz =\n'
    '    // bl[nz * columns + x], nt 1, c0 0). bl enters as constants (no input slots): its slope is held.\n'
    '    if (re && g_tempo_bl) {\n'
    '      AR<float> qb[TEMPO_NZP], fb[TEMPO_NZP];\n'
    '      TEMPO_LEVELS(k, nz) {\n'
    '        const long long x = (long long)(k - 1) * columns + c;\n'
    '        qb[k - 1] = ad_mk((ad_prim)g_tempo_bl[x], 0);\n'
    '        fb[k - 1] = ad_mk((ad_prim)g_tempo_bl[(long long)nz * columns + x], 0);\n'
    '      }\n'
    '      adt::tempo_subgrid_cloud_radii(P, ctx, nz, W, st.f[5], st.f[6], st.f[7], st.f[8],\n'
    '                                     (presence & 1) ? st.f[12] : nullptr, st.f[2], qb, fb, 1, 0);\n'
    '      ad_sync();\n'
    '    }\n'
    '    // tempo_step_diags, in its order of operations (lead lane).\n'
)


def main():
    for path in sys.argv[1:]:
        src = open(path).read()
        if 'g_tempo_bl' in src:
            sys.exit(f'{path}: already has g_tempo_bl')
        if src.count(GLOBAL_ANCHOR) != 1 or src.count(CALL_ANCHOR) != 1:
            sys.exit(f'{path}: anchors found {src.count(GLOBAL_ANCHOR)} / {src.count(CALL_ANCHOR)}, expected 1 / 1')
        src = src.replace(GLOBAL_ANCHOR, GLOBAL_TEXT + GLOBAL_ANCHOR).replace(CALL_ANCHOR, CALL_TEXT)
        open(path, 'w').write(src)
        print(f'{path}: patched')


if __name__ == '__main__':
    main()
