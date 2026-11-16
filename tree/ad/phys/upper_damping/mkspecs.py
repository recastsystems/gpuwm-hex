#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The ad.entry/v1 specs of the two upper-damping rows (upper_cam_f32, upper_rayleigh_f32).
usage: mkspecs.py OUT_DIR   (writes OUT/specs/<fn>.json and OUT/rows.json, the kernels.json rows add_rows.py appends)"""
import json, os, sys

# The forward text both rows are written against: hexcore.cuda_upper_damping_v841._CUDA_SOURCE (the shared
# float helpers and the two kernels). tests/test_ad_cumulus_rows.py reads it against the build.
FORWARD = "803b7ff5234a38237c5089fbb5eb31f3ebd0d593e969d2a12ccb4c2c9b4b23dc"

def arr(name, role, field, shape, overwrite='none'):
    return {"name": name, "role": role, "field_id": field, "dtype": "f32", "shape": shape, "active": True,
            "inactive_reason": None, "overwrite": overwrite, "alias_group": None, "weight": "none", "pointer_depth": 1}

def const(name, dtype='i32', why='fixed discrete dimension, launch extent or level count'):
    return {"name": name, "role": "const", "field_id": 0, "dtype": dtype, "shape": "[]", "active": False,
            "inactive_reason": why, "overwrite": "none", "alias_group": None, "weight": "none", "pointer_depth": 0}

def failure(fn, line):
    return {"bind": "failure", "slots": 1, "source_ids": [f"ad/phys/upper_damping/cuda/{fn}.ad.cu:{line}:nonfinite"]}

CAM = {
    "kernel": "upper_cam_f32", "engine": "hex", "forward_source_sha256": FORWARD, "file": "tree/src/hexcore/cuda_upper_damping_v841.py", "line": 5,
    "route": "hand", "mode": "exact",
    "why": "NOAA's hrrrv5 deck (config_mpas_cam_coef 2.0 on 8 levels) runs this launch every step; no kernel set had a row, so the window refused any recording on the deck",
    "args": [const("nl"), const("nc"), const("solved"), const("levels"),
             const("length", "f32", "mesh length scale (dcEdge-derived), fixed geometry"),
             const("coefficient", "f32", "namelist config_mpas_cam_coef"),
             arr("values", "inout", 1, "[nl,nc]", "partial")],
    "branches": [{"file_line": "cuda_upper_damping_v841.py:10", "predicate": "max(values, floor(k))",
                  "policy": "executed-arm", "tie": "values == floor takes the floor arm (derivative 0)",
                  "replay": "recompute floor(k) from the fixed scalars; compare with the bound primal values"}],
    "polynomial_degree": 1, "polynomial_reason": "piecewise identity: values or a constant",
    "launch_abi": {"schema": "ad.entry/v1", "entries": {
        "tl": {"function": "upper_cam_f32__tl", "geometry": "forward", "args": [
            {"bind": "scalar", "arg": "nl"}, {"bind": "scalar", "arg": "nc"}, {"bind": "scalar", "arg": "solved"},
            {"bind": "scalar", "arg": "levels"}, {"bind": "scalar", "arg": "length"}, {"bind": "scalar", "arg": "coefficient"},
            {"bind": "primal", "arg": "values"},
            {"bind": "tangent", "field_id": 1, "io": "in", "arg": "values"},
            {"bind": "tangent", "field_id": 1, "io": "out", "arg": "values"},
            failure("upper_cam_f32", 25)]},
        "ad": {"function": "upper_cam_f32__ad", "geometry": "forward", "args": [
            {"bind": "scalar", "arg": "nl"}, {"bind": "scalar", "arg": "nc"}, {"bind": "scalar", "arg": "solved"},
            {"bind": "scalar", "arg": "levels"}, {"bind": "scalar", "arg": "length"}, {"bind": "scalar", "arg": "coefficient"},
            {"bind": "primal", "arg": "values"},
            {"bind": "adjoint", "field_id": 1, "io": "accumulate", "arg": "values"},
            {"bind": "adjoint", "field_id": 1, "io": "in", "arg": "values"},
            failure("upper_cam_f32", 39)]}}}}

RAY = {
    "kernel": "upper_rayleigh_f32", "engine": "hex", "forward_source_sha256": FORWARD, "file": "tree/src/hexcore/cuda_upper_damping_v841.py", "line": 12,
    "route": "hand", "mode": "exact",
    "why": "NOAA's hrrrv5 deck (config_rayleigh_damp_u true) runs this launch at every RK stage; no kernel set had a row",
    "args": [const("nl"), const("ne"), const("solved"), const("levels"),
             const("days", "f32", "namelist config_rayleigh_damp_u_timescale_days"),
             arr("tendency", "inout", 1, "[nl,ne]", "partial"),
             arr("rho", "in", 2, "[nl,ne]"), arr("velocity", "in", 3, "[nl,ne]")],
    "branches": [],
    "polynomial_degree": 2, "polynomial_reason": "rho_edge times u, times a fixed per-level constant",
    "launch_abi": {"schema": "ad.entry/v1", "entries": {
        "tl": {"function": "upper_rayleigh_f32__tl", "geometry": "forward", "args": [
            {"bind": "scalar", "arg": "nl"}, {"bind": "scalar", "arg": "ne"}, {"bind": "scalar", "arg": "solved"},
            {"bind": "scalar", "arg": "levels"}, {"bind": "scalar", "arg": "days"},
            {"bind": "primal", "arg": "tendency"}, {"bind": "primal", "arg": "rho"}, {"bind": "primal", "arg": "velocity"},
            {"bind": "tangent", "field_id": 2, "io": "in", "arg": "rho"},
            {"bind": "tangent", "field_id": 3, "io": "in", "arg": "velocity"},
            {"bind": "tangent", "field_id": 1, "io": "in", "arg": "tendency"},
            {"bind": "tangent", "field_id": 1, "io": "out", "arg": "tendency"},
            failure("upper_rayleigh_f32", 23)]},
        "ad": {"function": "upper_rayleigh_f32__ad", "geometry": "forward", "args": [
            {"bind": "scalar", "arg": "nl"}, {"bind": "scalar", "arg": "ne"}, {"bind": "scalar", "arg": "solved"},
            {"bind": "scalar", "arg": "levels"}, {"bind": "scalar", "arg": "days"},
            {"bind": "primal", "arg": "tendency"}, {"bind": "primal", "arg": "rho"}, {"bind": "primal", "arg": "velocity"},
            {"bind": "adjoint", "field_id": 2, "io": "accumulate", "arg": "rho"},
            {"bind": "adjoint", "field_id": 3, "io": "accumulate", "arg": "velocity"},
            {"bind": "adjoint", "field_id": 1, "io": "accumulate", "arg": "tendency"},
            {"bind": "adjoint", "field_id": 1, "io": "in", "arg": "tendency"},
            failure("upper_rayleigh_f32", 39)]}}}}

def row_args(spec):
    out = []
    for a in spec["args"]:
        dev = a["pointer_depth"] == 1
        out.append({"name": a["name"], "kind": "device" if dev else "scalar",
                    "dtype": "<f4" if a["dtype"] == "f32" else "<i4", "shape": [], "strides_bytes": None,
                    "pointer_depth": a["pointer_depth"], "value_hex": None})
    return out

out = sys.argv[1]
os.makedirs(os.path.join(out, "specs"), exist_ok=True)
rows = []
for spec in (CAM, RAY):
    fn = spec["kernel"]
    json.dump(spec, open(os.path.join(out, "specs", fn + ".json"), "w", newline="\n"), indent=1)
    rows.append({"function": fn, "class": "rowed", "gate_ready": True,
                 "derivative": {"row": "upper-damping-" + fn, "group": "upper-damping", "spec": {"path": f"specs/{fn}.json"},
                                "cuda": {"path": f"ad/phys/upper_damping/cuda/{fn}.ad.cu"}},
                 "arguments": row_args(spec)})
json.dump(rows, open(os.path.join(out, "rows.json"), "w", newline="\n"), indent=1)
print("specs", [r["function"] for r in rows])
