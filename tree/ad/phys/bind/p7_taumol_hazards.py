#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Range/switch hazards of rlw_taumol_batched: every line of the functions gen_taumol.py copies (the 16 rlw_gbandN
# bodies, the 8 helpers) and the kernel itself that holds an int cast, fmodf, a comparison on a primal value,
# a clamp, a division, exp/log/pow, or a table index. Usage: hazards.py SOURCE.cu OUT.json
import json, re, sys
src = open(sys.argv[1]).read(); lines = src.split("\n")
def span(name, kw="__device__"):
    m = re.search(r"(?:__device__|__global__)[^;{()]*?\b" + name + r"\s*\(", src)
    i = src.index("{", m.end()); d = 0; j = i
    while True:
        c = src[j]
        if c == "{": d += 1
        elif c == "}":
            d -= 1
            if d == 0: break
        j += 1
    return src.count("\n", 0, m.start()) + 1, src.count("\n", 0, j) + 1
names = [f"rlw_gband{b}" for b in range(1, 17)] + ["rlw_pow4", "rlw_b35_spec", "rlw_b35_facs", "rlw_b35_major", "rlw_b35_bmajor",
        "rlw_specfac_0609", "rlw_ninemajor_0609", "rlw_nine_major_b1316", "rlw_taumol_batched"]
pats = {"int_cast": r"\(int\)", "fmod": r"fmodf", "compare": r"(?<![<>=!])(<|>|<=|>=)(?![<>=])\s*[-0-9.a-z_(]", "clamp": r"\b(fminf|fmaxf|min|max)\s*\(",
        "divide": r"RLW_DV|(?<![/*])/(?![/*])", "transcendental": r"\b(expf|logf|powf|rlw_pow)\b"}
out = {}
for n in names:
    a, b = span(n)
    hits = []
    for k in range(a, b + 1):
        t = lines[k - 1]; code = t.split("//")[0]
        kinds = [p for p, r in pats.items() if re.search(r, code)]
        if "specparm" in code and ("<" in code or ">" in code): kinds.append("specparm_switch")
        if "laytrop" in code and ("<" in code or ">" in code): kinds.append("laytrop_switch")
        if kinds and not code.strip().startswith(("for", "#")): hits.append({"line": k, "kinds": sorted(set(kinds)), "code": code.strip()[:160]})
    out[n] = {"lines": [a, b], "hits": hits}
json.dump(out, open(sys.argv[2], "w"), indent=1)
for n, v in out.items():
    c = {}
    for h in v["hits"]:
        for kk in h["kinds"]: c[kk] = c.get(kk, 0) + 1
    print(n, v["lines"], c)
