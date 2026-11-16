#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The engine's own New Tiedtke name tables (gpuwm.core.ntiedtke, read only) as JSON for
gen_nt.py: which parameter names are level or surface arrays, float or int, each stage's level base (0: the stage
gets arr[1:], 1: the whole (nz+2, ncol) allocation), and the storage each (stage, parameter) resolves to, so storage
aliases inside one launch share one shadow. usage (engine venv): nt_tables.py OUT.json"""
import json, sys
import gpuwm.core.ntiedtke as n

out = {'level_f': list(n.NT_LEVEL_F), 'level_i': list(n.NT_LEVEL_I), 'surface_f': list(n.NT_SURFACE_F),
       'surface_i': list(n.NT_SURFACE_I), 'base': {}, 'resolve': {}, 'call_order': list(n.NT_CALL_ORDER),
       'engine_file': n.__file__}
for stage, sig in n.NT_STAGE_SIGNATURE.items():
    out['base'][stage] = n.nt_level_base(stage)
    for a in sig:
        if a in ('scr', 'scr_i'):
            continue
        try:
            out['resolve'][f'{stage}:{a}'] = n.nt_resolve(n.NT_STAGE_ALIASES.get((stage, a), a))
        except Exception:
            pass
json.dump(out, open(sys.argv[1], 'w'), indent=1)
print('stages', len(out['base']))
