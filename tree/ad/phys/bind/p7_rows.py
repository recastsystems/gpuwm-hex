#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""L2 bindings (ad/window-L2-bindings): rows for P7's five checked longwave launches, with each per-thread tape
sized from the launch's own layer count (P9 hand-off: a fixed tape fails past about 81 layers).

  rows SPEC_DIR CHECK_DIR OUT.json   write p7-rows.json. The tape and write-log sizes are linear in the layer count
                                      (one column per thread, every layer recorded once): the check lane's receipts
                                      measured the maxima at 64 and 65 layers on the same kernels, which fixes
                                      per-layer and base counts exactly; the row adds 10 % plus 64 entries of margin.
                                      rlw_taumol_batched records one (column, layer) at a time (tape reset per layer,
                                      P7_TMAX 512 per thread), so its tape does not grow with the layer count.
  size ROWS.json CAPTURE_DIR OUT.json write the harness spec for one captured launch with tape/writes from its layer count.
Build time only.
"""
import glob, json, math, os, sys

def rows(spec_dir, check_dir, out):
    meas = {}
    for f in glob.glob(os.path.join(check_dir, "*.json")):
        d = json.load(open(f))
        st = d.get("tape_status_fail_maxentries_maxwrites")
        if not st or st[0] != 0: continue
        cap = d["capture"]
        m = json.load(open(os.path.join(cap, "manifest.json")))
        sc = {a["name"]: int.from_bytes(bytes.fromhex(a["value_hex"]), "little", signed=True)
              for a in m["arguments"] if a.get("kind") == "scalar" and a.get("dtype") == "<i4"}
        layers = next((sc[k] for k in ("nlayers", "nl", "nlay") if k in sc), None)
        meas.setdefault(d["launch"], {})[layers] = (st[1], st[2], f)
    out_rows = []
    for sp in sorted(glob.glob(os.path.join(spec_dir, "*.json"))):
        s = json.load(open(sp)); k = s["kernel"]
        mk = meas.get(k, {})
        ls = sorted(l for l in mk if l is not None)
        if len(ls) >= 2:
            (l0, (e0, w0, _)), (l1, (e1, w1, _)) = (ls[0], mk[ls[0]]), (ls[-1], mk[ls[-1]])
            ea, wa = (e1 - e0) // (l1 - l0), (w1 - w0) // (l1 - l0)
            assert ea * (l1 - l0) == e1 - e0 and wa * (l1 - l0) == w1 - w0, (k, mk)
            eb, wb = e0 - ea * l0, w0 - wa * l0
        else:  # no layer dependence measured (rlw_rtrn_secdiff: one value per column)
            e0, w0, _ = mk[None] if None in mk else list(mk.values())[0]
            ea, wa, eb, wb = 0, 0, e0, w0
        out_rows.append({"function": k, "unit": "P7", "kind": "p7_launch", "harness": "p7-launch (tree/ad/phys/p7/crate)",
                         "spec": os.path.relpath(sp, os.path.dirname(os.path.dirname(spec_dir))),
                         "active_in": s["active_in"], "active_out": s["active_out"],
                         "tape": {"entries_per_layer": ea, "entries_base": eb, "writes_per_layer": wa, "writes_base": wb,
                                  "margin": "ceil(1.10 * n) + 64", "measured": {str(l): {"entries": v[0], "writes": v[1], "receipt": os.path.basename(v[2])} for l, v in mk.items()}},
                         "layer_scalars": ["nlayers", "nl", "nlay"],
                         "plan_binding": "named gap, owner P7: radiation is a held upload in pkg18, so these launches are not in the plan's command stream yet"})
    out_rows.append({"function": "rlw_taumol_batched", "unit": "P7", "kind": "p7_taumol", "harness": "p7-taumol (tree/ad/phys/p7/crate)",
                     "tape": {"per_column_layer": 512, "grows_with_layers": False,
                              "why": "the tape is reset for every (column, layer) inside the thread (taumol_ad_kernels.cuh), P7_TMAX 512"},
                     "plan_binding": "named gap, owner P7: radiation is a held upload in pkg18"})
    json.dump({"schema": "ad.scheme.rows/v1", "lane": "ad/window-L2-bindings", "unit": "P7", "rows": out_rows}, open(out, "w"), indent=1)
    print(json.dumps([[r["function"], r["tape"]] for r in out_rows], indent=0)[:3000])

def size(rows_path, cap, out):
    t = json.load(open(rows_path))
    m = json.load(open(os.path.join(cap, "manifest.json")))
    k = m["kernel"]
    r = next(x for x in t["rows"] if x["function"] == k)
    sc = {a["name"]: int.from_bytes(bytes.fromhex(a["value_hex"]), "little", signed=True)
          for a in m["arguments"] if a.get("kind") == "scalar" and a.get("dtype") == "<i4"}
    layers = next((sc[s] for s in r["layer_scalars"] if s in sc), 0)
    tp = r["tape"]
    need_e = tp["entries_per_layer"] * layers + tp["entries_base"]
    need_w = tp["writes_per_layer"] * layers + tp["writes_base"]
    spec_dir = os.path.dirname(os.path.abspath(rows_path))
    base = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), r["spec"])))
    base["tape"] = math.ceil(1.10 * need_e) + 64
    base["writes"] = math.ceil(1.10 * need_w) + 64
    base["sized_from"] = {"layers": layers, "entries_needed": need_e, "writes_needed": need_w, "row": rows_path}
    json.dump(base, open(out, "w"), indent=1)
    print(json.dumps(base["sized_from"] | {"tape": base["tape"], "writes": base["writes"]}))

if __name__ == "__main__":
    {"rows": rows, "size": size}[sys.argv[1]](*sys.argv[2:])
