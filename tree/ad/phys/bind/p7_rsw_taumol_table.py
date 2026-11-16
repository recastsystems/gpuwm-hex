#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# held-rsw_taumol_b: one table row per captured launch from the two capdot processes (runs/<tag>-r1, -r2).
import glob, json, os, sys
runs = sys.argv[1]
tags = sorted({os.path.basename(p)[:-3] for p in glob.glob(os.path.join(runs, "*-r1"))},
              key=lambda t: (t.startswith("shared"), int(t.split("-s")[-1].split("-")[0]) if "-s" in t else int(t[1:].split("-")[0]), t))
def ncol(d): return d["forward"]["post_words"]
print("| launch | ncol x nl | forward post (differing / words) | augmented primal (differing / words) | dot rel seed 1 / 2 | norm-scaled seed 1 / 2 | zero dir TL / AD | 2nd reverse | two processes (result_sha256) | gain raw | gain relative | tape high water | write coverage | wall s |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
worst = 0; allpass = True
for t in tags:
    a, b = (json.load(open(os.path.join(runs, f"{t}-r{r}", "capdot.json"))) for r in (1, 2))
    man = json.load(open(os.path.join(a["capture"], "manifest.json")))
    import struct
    sc = {x["name"]: struct.unpack("<i", bytes.fromhex(x["value_hex"]))[0] for x in man["arguments"] if x["kind"] == "scalar"}
    dd = [x["dot"] for x in a["dots"]]
    ent = dd[0]["entry"]
    aug = ent["tangent"]["augmented_primal_differing_words_taug"] + ent["tangent"]["augmented_primal_differing_words_taur"]
    rel = [x["rel"] for x in dd]; worst = max(worst, *rel)
    zero = (sum(x["zero_tl_nonzero_words"] for x in dd), sum(x["zero_ad_nonzero_words"] for x in dd))
    rev = sum(x["reverse_differing_words"] for x in dd)
    same = a["result_sha256"] == b["result_sha256"] and [x["shadows_sha256"] for x in a["dots"]] == [x["shadows_sha256"] for x in b["dots"]]
    g = {x["direction"]: x["gain"] for x in a["gains"]}
    wc = a["write_coverage"]
    cov = "complete" if all(v["left_unwritten"] == 0 and v["rerun_equals_first_forward"] for v in wc.values()) else json.dumps(wc)
    ok = a["status"] == b["status"] == "PASS" and same and aug == 0 and zero == (0, 0) and rev == 0 and a["forward"]["post_differing_words"] == 0
    allpass &= ok
    print(f"| {t} | {sc['ncol']} x {sc['nlayers']} | {a['forward']['post_differing_words']} / {a['forward']['post_words']:,} | {aug} / {ent['tangent']['augmented_primal_words']:,} | "
          f"{rel[0]:.2e} / {rel[1]:.2e} | {dd[0]['norm_scaled']:.1e} / {dd[1]['norm_scaled']:.1e} | {zero[0]} / {zero[1]} | {'bitwise' if rev == 0 else rev} | "
          f"{'bitwise ' + a['result_sha256'][:8] if same else 'DIFFER'} | {g['raw']:.3g} | {g['relative']:.3f} | {ent['adjoint']['tape_high_water']} | {cov} | {a['wall_s']:.1f} / {b['wall_s']:.1f} |")
print(f"\n{len(tags)} launches, {'all PASS' if allpass else 'NOT all pass'}, worst rel {worst:.3g}")
