#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""L2 bindings (ad/window-L2-bindings): write the ad.scheme.rows/v1 table (schemes.json) the P1 runner's
loader (schemes.rs) binds to live plan launches. Build time only; nothing here runs in a step.

Sources (all read-only):
  * P3 MYNN-EDMF: the captured hrrrv5 launch manifest (argument order, kinds, dtypes, shapes) and P3's
    gen-info.json (float array order of the MwLaunch slots, scalar order of the entries).
  * P4 surface layer + GSL drag: the captured launch manifests and P4-ENTRIES.json (roles, entries, hand-off).
  * P8: the rule table (tree/ad/phys/p8/src/rules.rs, kernel_class) for the command rows; each row carries the
    class P8's table gives it, and the runner refuses the table if they disagree.
Extents are expressions in the launch's own scalars (ime-ims+1, nz, n, ...), so a row binds any launch of the
function; a dimension that is not a launch scalar is recorded in "fixed" and checked at bind time.
With --plan, every row is checked against every launch of the sealed plan (argument count and kinds) and the
counts are written into the table.

usage: gen_rows.py CAPTURE_ROOT P3_GEN_INFO P4_ENTRIES OUT_DIR [--plan PLAN.json]
"""
import collections, hashlib, json, os, sys

cap, p3info_path, p4e_path, out = sys.argv[1:5]
plan_path = sys.argv[sys.argv.index("--plan") + 1] if "--plan" in sys.argv else None

def manifest(k):
    return json.load(open(os.path.join(cap, k, "1", "launch-000000", "manifest.json")))

def scalars_of(m):
    v = {}
    for a in m["arguments"]:
        if a["kind"] != "device" and a.get("value_hex"):
            b = bytes.fromhex(a["value_hex"])
            d = a.get("dtype", "")
            if d in ("<i4", "<u4") and len(b) == 4: v[a["name"]] = int.from_bytes(b, "little", signed=d == "<i4")
            elif d in ("<i8", "<u8") and len(b) == 8: v[a["name"]] = int.from_bytes(b, "little", signed=d == "<i8")
    return v

def args_of(m, roles, symbol_dims, fixed):
    rows = []
    for i, a in enumerate(m["arguments"]):
        name = a["name"]
        role, active = roles.get(name, ("in", False))
        if a["kind"] == "device":
            dims = []
            # pk120-recording: a device array whose element count is a scalar of the launch (ugwp_oro_diff: [59, 2562]
            # with n = 151,158) takes that scalar as its extent, never the capture mesh's literal dims. Breakage: x1.40962
            # bound 59*2562 words of shadow for n = 59*40962 and the per-launch dot ZERO-FAILed at launches 1622/1623
            # (177,772 nonzero adjoint words from a zero cotangent, 904,135 words differing between two reverses).
            whole = 1
            for d in a["shape"]: whole *= d
            if len(a["shape"]) > 1 and whole in symbol_dims and not all(symbol_dims.get(d) for d in a["shape"]):
                rows.append({"name": name, "pos": i, "device": True, "dtype": a["dtype"], "extent": symbol_dims[whole],
                             "capture_shape": a["shape"], "role": role, "active": active})
                continue
            for d in a["shape"]:
                sym = symbol_dims.get(d)
                dims.append(sym if sym else str(d))
                if not sym and d not in (1,) and d > 64:
                    fixed.setdefault("_literal_dims", []).append([name, d])
            rows.append({"name": name, "pos": i, "device": True, "dtype": a["dtype"], "extent": "*".join(dims) or "1",
                         "capture_shape": a["shape"], "role": role, "active": active})
        elif a.get("value_hex") is None:
            # a null pointer in the forward (absent array): no storage, no derivative
            rows.append({"name": name, "pos": i, "device": True, "dtype": a.get("dtype") or "<f4", "extent": None,
                         "role": "absent", "active": False})
        else:
            rows.append({"name": name, "pos": i, "device": False, "dtype": a["dtype"], "role": "scalar", "active": False})
    return rows

rows, notes = [], []
images = {"p8": "p8-sm_120.cubin", "l2_rules": "l2_rules-sm_120.cubin"}

# ------------------------------------------------------------------ P3 MYNN-EDMF whole launch
m = manifest("rrfs_mynnedmf_run_l16")
info = json.load(open(p3info_path))
sv = scalars_of(m)
ni, nk, nj = sv["ime"] - sv["ims"] + 1, sv["kme"] - sv["kms"] + 1, sv["jme"] - sv["jms"] + 1
symbols = {ni: "(ime-ims+1)", nk: "(kme-kms+1)"}
if nj not in symbols: symbols[nj] = "(jme-jms+1)"
roles = {n: ("inout", True) for n in info["float_arrays"]}
roles.update({"kpbl": ("inout", False), "error_flags": ("out", False), "workspace": ("scratch", False)})
fixed = {}
p3args = args_of(m, roles, symbols, fixed)
ws = next(a for a in p3args if a["name"] == "workspace"); ws["extent"] = "workspace_stride*batch_count"
for a in p3args:
    if a["device"] and a["extent"]:
        env = dict(sv, ime=sv["ime"], ims=sv["ims"])
        n = eval(a["extent"], {}, env) if a["name"] != "workspace" else sv["workspace_stride"] * sv["batch_count"]
        want = 1
        for d in a["capture_shape"]: want *= d
        assert n == want, (a["name"], a["extent"], n, want)
names = {a["name"] for a in p3args}
missing = [n for n in info["float_arrays"] if n not in names]
assert not missing, missing
absent = [a["name"] for a in p3args if a["role"] == "absent"]
rows.append({"function": "rrfs_mynnedmf_run_l16", "unit": "P3", "kind": "p3_mynn", "p8_class": "entry",
             "source_sha256": info["source_sha256"], "generator_output_sha256": info["output_sha256"],
             "float_arrays": info["float_arrays"], "int_arrays": info["int_arrays"], "scalars": info["scalars"],
             "args": p3args, "fixed": {}, "absent_in_capture": absent,
             "entries": {"tl": "mynn_whole_tl (rel.0)", "ad": "mynn_whole_count then mynn_whole_ad (rel.2, rel.1), per tile",
                         "launch_struct": "MwLaunch: per float array {x, y, tin, tout, ybar, xbar}, then kpbl in/out, error flags, arena, store, leaf, kstore, nodes, bars, capacity, used, failures, col_first, col_count, mode"},
             "tile": "columns per entry launch = tiles.p3_columns (multiple of the 32-column warp tile); arena per tile = tile x workspace_stride x tiles.p3_arena_x bytes; tape per tile = tile x (max tape nodes per column of the tile + 16) x 32 bytes",
             "pre_state": "x = the launch's pre-state snapshot of every array (taken before the forward); the entries recompute the column from it; the composed reverse must restore the same pre state (checkpoint) before the adjoint",
             "derivative_width": "float64 (tin/tout/ybar/xbar are double)"})
images.update({"p3_tl": "p3/rel.0.cubin", "p3_ad": "p3/rel.1.cubin", "p3_count": "p3/rel.2.cubin"})

# ------------------------------------------------------------------ P4 surface layer + GSL drag
p4e = json.load(open(p4e_path))
P4 = {
    "mynnsfc_from_mpas": ("p4_from", {"f": ("inout", True), "sea": ("out", True), "density": ("const", False), "sst": ("in", True), "ice": ("in-branch", False)}),
    "mynnsfc_columns": ("p4_columns", {"f": ("inout", True), "options": ("const", False), "vegtype": ("const", False), "errors": ("out", False)}),
    "mynnsfc_to_mpas": ("p4_to", {"f": ("inout", True), "sea": ("in", True), "ice": ("in", True)}),
    "ugwp_oro_run": ("p4_run", {"layers": ("in", True), "prsi0": ("in", True), "surface": ("in", True), "kpbl0": ("const", False),
                                "out": ("out-whole", True), "sout": ("out-whole", True), "status": ("out", False)}),
    "ugwp_oro_diff": ("p4_diff", {"before": ("in", True), "after": ("in", True), "diff": ("out", True)}),
    "ugwp_oro_dx": ("p4_dx", {"density": ("const", False), "dx": ("out-geometry", True)}),
}
for fn, (kind, roles) in P4.items():
    m = manifest(fn)
    sv = scalars_of(m)
    sym = {}
    for s in ("n", "nc"):
        if s in sv: sym[sv[s]] = s
    if "nz" in sv: sym[sv["nz"]] = "nz"; sym.setdefault(sv["nz"] + 1, "(nz+1)")
    fixed = {}
    a = args_of(m, roles, sym, fixed)
    lit = fixed.pop("_literal_dims", [])
    rows.append({"function": fn, "unit": "P4", "kind": kind, "p8_class": "entry", "args": a, "fixed": {},
                 "literal_dims": lit, "p4_entries": p4e["kernels"].get(fn, {}),
                 "pre_state": "every primal pointer of the entries is a private copy of the launch's pre-state snapshot; mynnsfc_columns' adjoint needs f as the PRE state and ugwp_oro_run's adjoint recomputes each column from layers/prsi0/surface: the composed reverse hands them the restored pre state",
                 "tile": "adjoints with a tape (mynnsfc_columns, ugwp_oro_run): one count pass, then column chunks sized so 32 x tape nodes x chunk <= tiles.p4_tape_budget_bytes (128-column chunks for the surface layer, 64 for GSL drag)",
                 "derivative_width": "float64 (__tl64 / __ad64 entries)"})
m = manifest("mynnsfc_tables")
rows.append({"function": "mynnsfc_tables", "unit": "P4", "kind": "inactive", "p8_class": "inactive",
             "why": "builds constant psi tables (P4-ENTRIES: inactive); writes only an error word; the derivative modules build their own tables at load",
             "args": args_of(m, {}, {}, {}), "fixed": {}})
images.update({"p4_mynnsfc_tl": "p4/mynnsfc.tl.sm_120.cubin", "p4_mynnsfc_rev": "p4/mynnsfc.rev.sm_120.cubin",
               "p4_ugwp_tl": "p4/ugwp.tl.sm_120.cubin", "p4_ugwp_rev": "p4/ugwp.rev.sm_120.cubin"})

# ------------------------------------------------------------------ TEMPO order (ad/held-tempo_order_scan)
# tempo_order_scan (tempo_v303.py:4040-4042): in-place exclusive scan of 128 uint32 bucket counts, one pointer argument.
# Recorded sm_120 SASS (module 477158923b00908c): 127 LDG.E + 128 STG.E inside bytes 0-511 of that pointer, IADD only,
# no float opcode. The row binds every launch as INACTIVE: TL and AD are the identity on every float word.
rows.append({"function": "tempo_order_scan", "unit": "P5-order", "kind": "inactive", "p8_class": "inactive",
             "why": "COMMON-AD 87.1 column-order permutation: exclusive prefix scan of 128 uint32 bucket counts in place (tempo_v303.py:4040-4042); one device pointer argument, no float word read or written; TL and AD are the identity on every float word (empty active set)",
             "args": [{"name": "hist", "pos": 0, "device": True, "dtype": "<u4", "extent": "128", "capture_shape": [128], "role": "inout", "active": False}], "fixed": {}})
# ------------------------------------------------------------------ P6 RUC sea ice (whole launch; bind/p6_row.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p6_row
rows.append(p6_row.p6_row())
images.update(p6_row.IMAGES)

# ------------------------------------------------------------------ P8 command rules
P8 = [("cupy_copy__float32_float32", "p8_view", "rule", "view copy: d_out = d_in; b_in += b_out, b_out = 0 (broadcast source: copy-row gather)"),
      ("cupy_copy__float32_float64", "p8_view", "rule", "view copy, cast exact"),
      ("cupy_copy__float64_float64", "p8_view", "rule", "view copy"),
      ("cupy_copy__int32_float64", "p8_zero_out", "inactive", "integer source: the float64 output's derivative is zero in both directions"),
      ("cupy_maximum__float_float32_float32", "p8_max", "rule", "max(c, x): executed arm of the pre-launch x (x wins only if x > c and not NaN)"),
      ("cupy_copy_where__float32_bool_float32", "p8_where", "rule", "copy_where: executed arm of the forward's own mask"),
      ("cupy_multiply__float32_float_float32", "p8_scale", "rule", "x * c, c a host constant: d_out = c d_x; b_x += c b_out, b_out = 0")]
# step6-sw: no cupy_concatenate_same_size row. Its float32 (SW radiation input stack, module 5396101c) and integer
# (module 82af2f9c) instantiations share the name; the window binds it per module from the recorded source
# (wrules::concatenate) and schemes.rs refuses a by-name row.
for fn, kind, cls, what in P8:
    r = {"function": fn, "unit": "P8", "kind": kind, "p8_class": cls, "rule": what, "args": [], "fixed": {}}
    rows.append(r)

# ------------------------------------------------------------------ check against the plan
verified = None
if plan_path:
    p = json.load(open(plan_path))
    by = {r["function"]: r for r in rows}
    counts, bad, mods = collections.Counter(), [], collections.defaultdict(set)
    for s in p["cadence"]:
        for b in s:
            for op in p["blocks"][b]:
                if op.get("kind") != "kernel" or op["function"] not in by: continue
                f = op["function"]; r = by[f]; counts[f] += 1; mods[f].add(op.get("module"))
                if r["args"]:
                    if len(op["arguments"]) != len(r["args"]): bad.append([f, "argument count", len(op["arguments"]), len(r["args"])])
                    for ra, pa in zip(r["args"], op["arguments"]):
                        if ra["device"] and ra["role"] != "absent" and pa["kind"] != "device": bad.append([f, ra["name"], pa["kind"]])
    # concatenate: the module's recorded source must be the long long instantiation
    src_dir = os.path.dirname(plan_path)
    for mod in mods.get("cupy_concatenate_same_size", []):
        found = None
        for s in p["cadence"]:
            for b in s:
                for op in p["blocks"][b]:
                    if op.get("function") == "cupy_concatenate_same_size" and op.get("module") == mod:
                        for a in op["arguments"]:
                            for key in ("source", "fixed_source"):
                                v = (a.get(key) or {}).get("source", "")
                                if v.startswith("recording/sources/"): found = v
                        break
        t = None
        if found:
            for line in open(os.path.join(src_dir, found)):
                line = line.strip()
                if line.startswith("typedef ") and line.endswith(" T;"): t = line[8:-3]; break
        if t not in ("long long", "long", "int", "unsigned long long", "unsigned int", "bool"): bad.append(["cupy_concatenate_same_size", "module type", mod, t])
    verified = {"plan": plan_path, "launches": dict(counts), "modules": {k: sorted(v) for k, v in mods.items()}, "mismatches": bad}
    if bad: print("MISMATCH", bad[:20]); sys.exit(1)

table = {"schema": "ad.scheme.rows/v1", "lane": "ad/window-L2-bindings", "images": images,
         "tiles": {"p3_columns": 128, "p3_arena_x": 12, "p4_tape_budget_bytes": 2 << 30}, "rows": rows,
         "verified_on": verified}
os.makedirs(out, exist_ok=True)
json.dump(table, open(os.path.join(out, "schemes.json"), "w"), indent=1)
print(json.dumps({"rows": len(rows), "verified": verified and verified["launches"]}))
