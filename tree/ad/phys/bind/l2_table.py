#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""L2 bindings: per-unit table of an AD_DOT_* run with AD_SCHEMES (dots.jsonl[.gz] plus the summary if present).
Prints, per unit (P3, P4, P8): launches with a row, launches dotted, statuses, worst rel, zero-direction and
reverse-run words over the checked launches, primal identity words (P3), summed TL / AD entry seconds; then
the held / unbound launches grouped by P8's class (from the summary's untested reasons).
usage: l2_table.py DOTS.jsonl[.gz] [SUMMARY.json] [SCHEMES.json]"""
import collections, gzip, json, sys
p = sys.argv[1]
f = gzip.open(p, "rt") if p.endswith(".gz") else open(p)
units = collections.defaultdict(lambda: {"launches": 0, "dotted": 0, "status": collections.Counter(), "max_rel": 0.0, "max_ns": 0.0,
                                         "zero_tl": 0, "zero_ad": 0, "rev": 0, "checked": 0, "ident": 0, "tl_s": 0.0, "ad_s": 0.0,
                                         "fns": collections.Counter(), "first_bad": None})
steps = []
for line in f:
    try: r = json.loads(line)
    except Exception: continue
    if r.get("event") == "ad_dot_step": steps.append(r); continue
    if "function" not in r or r.get("row") != "ad.scheme.rows/v1" and r.get("unit") is None: continue
    u = units[r.get("unit") or "?"]
    u["launches"] += 1; u["fns"][r["function"]] += 1
    s = r["status"]; u["status"][s] += 1
    if s != "INACTIVE": u["dotted"] += 1
    if s not in ("PASS", "INACTIVE") and u["first_bad"] is None: u["first_bad"] = {k: r.get(k) for k in ("function", "step", "status", "rel", "norm_scaled", "error", "zero_tl_nonzero_words", "zero_ad_nonzero_words", "reverse_differing_words", "identity_differing_words")}
    for k, src in (("max_rel", "rel"), ("max_ns", "norm_scaled")):
        x = r.get(src)
        if isinstance(x, (int, float)) and x > u[k]: u[k] = x
    if r.get("checks"):
        u["checked"] += 1; u["zero_tl"] += r.get("zero_tl_nonzero_words") or 0; u["zero_ad"] += r.get("zero_ad_nonzero_words") or 0
        u["rev"] += r.get("reverse_differing_words") or 0
    u["ident"] += r.get("identity_differing_words") or 0
    u["tl_s"] += (r.get("tl_ms") or 0) / 1e3; u["ad_s"] += (r.get("ad_ms") or 0) / 1e3
print("unit\tlaunches_with_row\tdotted\tstatus\tmax_rel\tmax_norm_scaled\tchecked\tzero_tl_words\tzero_ad_words\treverse_diff_words\tidentity_diff_words\ttl_s\tad_s\ts_per_dotted_launch")
for k in sorted(units):
    u = units[k]
    print("%s\t%d\t%d\t%s\t%.3g\t%.3g\t%d\t%d\t%d\t%d\t%d\t%.1f\t%.1f\t%.3f" % (k, u["launches"], u["dotted"], ",".join("%s:%d" % x for x in sorted(u["status"].items())),
          u["max_rel"], u["max_ns"], u["checked"], u["zero_tl"], u["zero_ad"], u["rev"], u["ident"], u["tl_s"], u["ad_s"], (u["tl_s"] + u["ad_s"]) / max(u["dotted"], 1)))
for k in sorted(units):
    print("#", k, "functions", dict(units[k]["fns"]), "first non-PASS", json.dumps(units[k]["first_bad"]))
print("# steps", [(s["step"], round(s["elapsed_s"], 1), s.get("arena_peak_bytes"), s.get("scheme_scratch_bytes")) for s in steps])
if len(sys.argv) > 2:
    s = json.load(open(sys.argv[2]))
    groups = collections.defaultdict(lambda: [0, []])
    for fn, v in s["kernels"].items():
        if isinstance(v.get("status"), dict): continue
        why = v.get("why", "")
        key = why.split("P8 table: ")[1].split(":")[0] if "P8 table: " in why else why
        if "P8 table: pending" in why: key = "pending " + why.split("P8 table: pending: ")[1].split(" ")[0]
        groups[key][0] += v["launches"]; groups[key][1].append("%s %d" % (fn, v["launches"]))
    print("# held / unbound launches by class")
    for k, (n, fs) in sorted(groups.items(), key=lambda x: -x[1][0]): print("#  %-60s %6d  %s" % (k[:60], n, "; ".join(sorted(fs))[:400]))
    print("# commands", json.dumps(s.get("commands")))
    print("# totals", json.dumps(s.get("totals")), "wall_s", round(s.get("wall_s", 0), 1))
if len(sys.argv) > 3:
    t = json.load(open(sys.argv[3]))
    want = (t.get("verified_on") or {}).get("launches", {})
    have = collections.Counter()
    for u in units.values(): have.update(u["fns"])
    miss = {k: (v, have.get(k, 0)) for k, v in want.items() if have.get(k, 0) != v}
    print("# row coverage vs plan launches (steps run):", "all launches seen" if not miss else json.dumps(miss))
