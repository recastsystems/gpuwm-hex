#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ad.scheme.rows/v1 row of the P6 RUC sea-ice whole launch (oracle_ice_ruc_ice), as data.

The helper-scalar order (NAMES), the pointer-table slot of each (SLOTS) and the scratch-arena row of the
local_* helpers (LOCAL_WORK) are P6's own tables (tree/ad/phys/p6/src/main.rs NAMES/SLOTS/LOCAL_WORK), the ones
its replay, augmented-primal identity and dot proofs use. schemes.rs (kind p6_seaice) binds them to a live
launch: the pointer table is read at bind time and every word resolves to (allocation, offset) in live storage.

Used by gen_rows.py (fresh bundle) and, as a CLI, to extend an existing bundle's schemes.json:
  p6_row.py IN_SCHEMES.json OUT_SCHEMES.json
"""
import json, sys

NAMES = ["dt", "tabs", "qvatm", "qcatm", "p8w_kms", "p8w_1", "z3d", "rho", "rainbl", "rainncv", "frzfrac", "snowncv",
         "graupelncv", "flqc", "flhc", "cp", "rovcp", "g0", "lv", "stbolt", "glw", "gsw", "alb", "albbck", "soilt", "soilt1",
         "tsnav", "snow", "snowh", "snowc", "qsfc", "qsg", "qvg", "qcg", "hfx", "qfx", "lh", "dew", "sfcrunoff", "acrunoff",
         "sfcevp", "grdflx", "snowfallac", "acsnow", "snom", "precipfr", "rhosnf", "chklowq", "znt", "z0", "snoalb", "emiss",
         "xice", "xice_threshold", "local_smelt", "local_snoh", "local_snflx", "local_edir", "local_ec", "local_ett",
         "local_sublim", "local_sflx", "local_smf", "local_evapl", "local_prcpl", "local_budget", "local_runoff1",
         "local_runoff2", "local_infiltr", "local_rsm", "local_snweprint", "local_snheiprint"]
SLOTS = [3, 19, 20, 21, 18, 18, 17, 22, 12, 10, 13, 9, 8, 28, 29, 45, 46, 47, 48, 49, 23, 24, 30, 34, 51, 52, 53, 54, 55,
         56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 16, 15, 26, 31, 32, 33, 25, 43, 44]
LOCAL_WORK = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 12, 9, 10, -1, -1, -1, -1]
assert len(NAMES) == 72 and len(SLOTS) == 54 and len(LOCAL_WORK) == 18

IMAGES = {"p6": "p6/p6_seaice_sm_120.cubin", "p6_bind": "p6/p6_bind-sm_120.cubin", "p6_table": "p6/vilka-table.bin"}


def p6_row():
    return {
        "function": "oracle_ice_ruc_ice", "unit": "P6", "kind": "p6_seaice", "p8_class": "entry", "fixed": {},
        "args": [
            {"name": "ptrs", "pos": 0, "device": True, "dtype": "<u8", "extent": "98", "role": "pointer-table", "active": False},
            {"name": "tables", "pos": 1, "device": True, "dtype": "<f4", "extent": None, "role": "const", "active": False},
            {"name": "work", "pos": 2, "device": True, "dtype": "<f4", "extent": None, "role": "scratch-arena", "active": False},
            {"name": "error", "pos": 3, "device": True, "dtype": "<i4", "extent": None, "role": "error", "active": False},
            {"name": "nc", "pos": 4, "device": False, "dtype": "<i4", "role": "scalar", "active": False}],
        "p6": {
            "names": NAMES, "slots": SLOTS, "local_work": LOCAL_WORK, "pointer_slots": 98, "nz": 9,
            "tso_slot": 50, "zs_slot": 11, "xice_slot": 43, "threshold_slot": 44,
            "flags": {"ktau": 4, "restart": 6, "cycling": 7, "isice": 42, "frpcpn": 14, "graupel": 93, "myj": 40},
            "table_words": 5001,
            "columns": "sea-ice columns, xice >= xice_threshold in the pre state (as the forward and P6's driver select them)",
            "in_place": "tangent: the shadow of every helper output word (72 scalars, tso) takes the TL output; zs and the table are inputs only and keep theirs. Adjoint: the transpose (output words zeroed, then every input word += its accumulated cotangent; zs keeps its cotangent and gains)",
            "frozen": "branch-frozen linearization recorded by the augmented primal at the launch's own pre state: the 271.4 K ice cap tso = min(271.4, tso) takes the executed arm (a tie keeps tso, so its derivative passes), the snow and precipitation switches and the Vilka table visits likewise (08:35Z ruling: freeze from the trajectory)",
            "unmodeled": "arena scratch rows the forward writes besides LOCAL_WORK (launch-local stack temporaries) are not bound: they keep their incoming tangent, as under a held launch"},
        "source": "tree/ad/phys/p6 (cuda/seaice_ruc_ice_whole.cu + cuda/p6_seaice.cu, NVRTC, --fmad=false --ftz=false) + tree/ad/phys/bind/cuda/p6_bind.cu"}


def extend(table):
    table["rows"] = [r for r in table["rows"] if r.get("function") != "oracle_ice_ruc_ice"] + [p6_row()]
    table["images"].update(IMAGES)
    return table


if __name__ == "__main__":
    t = extend(json.load(open(sys.argv[1])))
    json.dump(t, open(sys.argv[2], "w"), indent=1)
    print(json.dumps({"rows": len(t["rows"]), "images": sorted(t["images"])}))
