#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ad/held-rlw_taumol_batched: the rlw_taumol_batched row (ad.scheme.rows/v1, kind p7_taumol) as data, added to a
copy of an L2 scheme bundle. Build time only.

  p7_taumol_row.py BUNDLE_IN BUNDLE_OUT IMAGE GEN_SOURCE CAPTURE_MANIFEST

BUNDLE_OUT gets every file of BUNDLE_IN (hard links when possible, never edited in place), the P7 image as
p7/taumol.sm_120.cubin and a schemes.json with images.p7_taumol and the row. The row's arguments are the forward's
(hexcore/mpasphys/rrtmg_lw_v842.py:795, captured source 683d7a07 kernel rlw_taumol_batched), checked against the
capture manifest's argument names, kinds and dtypes; extents are in the launch's own ncol and nl.
"""
import hashlib, json, os, shutil, sys

PIN = "683d7a070b22b0586a646f9e1542ce73c396a801767cd92046f4749183efd606"
ARGS = [  # name, device, dtype, extent, role, active
    ("ncol", False, "<i4", None, "scalar", False),
    ("nl", False, "<i4", None, "scalar", False),
    ("laytrop_v", True, "<i4", "ncol", "in-branch", False),           # tropopause layer per column (setcoef): switch
    ("fs", True, "<f4", "33*ncol*nl", "in", True),                    # 33 slots: fac00..11, column amounts, ratios, self/foreign/minor factors
    ("isv", True, "<i4", "6*ncol*nl", "in-branch", False),            # jp, jt, jt1, indself, indfor, indminor: table indices
    ("wx", True, "<f4", "4*ncol*nl", "in", True),                     # cross-section amounts (CCl4, CFC11, CFC12, CFC22)
    ("chi_mls", True, "<f4", "413", "const", False),                  # MLS reference ratios (static table)
    ("oneminus", False, "<f4", None, "scalar", False),
    ("bandptrs", True, "<u8", "16", "table", False),                  # 16 band pointer tables -> k-coefficient tables (static)
    ("taug", True, "<f4", "140*ncol*nl", "out", True),                # gas optical depth, every g-point
    ("fracs", True, "<f4", "140*ncol*nl", "out", True),               # Planck fractions, every g-point
]

def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()

def main(bin_, bout, image, gen, manifest):
    m = json.load(open(manifest))
    if m["source"]["sha256"] != PIN: raise SystemExit(f"capture source {m['source']['sha256']} is not the pinned {PIN}")
    got = [(a["name"], a["kind"] == "device", a["dtype"]) for a in m["arguments"]]
    want = [(n, d, t if n != "bandptrs" else got[8][2]) for n, d, t, *_ in ARGS]
    if got != want: raise SystemExit(f"capture arguments {got} differ from the row {want}")
    if os.path.exists(bout): raise SystemExit(f"{bout} exists; the bundle is written once")
    shutil.copytree(bin_, bout, copy_function=lambda s, d: os.link(s, d) if os.stat(s).st_dev == os.stat(os.path.dirname(d) or ".").st_dev else shutil.copy2(s, d))
    os.unlink(os.path.join(bout, "schemes.json"))
    t = json.load(open(os.path.join(bin_, "schemes.json")))
    os.makedirs(os.path.join(bout, "p7"), exist_ok=True)
    shutil.copy2(image, os.path.join(bout, "p7", "taumol.sm_120.cubin"))
    t["images"]["p7_taumol"] = "p7/taumol.sm_120.cubin"
    if any(r["function"] == "rlw_taumol_batched" for r in t["rows"]): raise SystemExit("bundle already has a rlw_taumol_batched row")
    t["rows"].append({
        "function": "rlw_taumol_batched", "unit": "P7", "kind": "p7_taumol", "p8_class": "entry",
        "source": "hexcore/mpasphys/rrtmg_lw_v842.py:795 (RRTMG-LW v4.85 taumol, all 16 bands, one launch per layer group)",
        "source_sha256": PIN, "generator": "tree/ad/phys/p7/gen/gen_taumol.py", "generated_sha256": sha(gen),
        "image_sha256": sha(image),
        "nvrtc_options": ["-std=c++17", "--ftz=false", "--fmad=false", "--prec-div=true", "--prec-sqrt=true", "-arch=sm_120"],
        "args": [{"name": n, "pos": i, "device": d, "dtype": t_, **({"extent": e} if e else {}), "role": r, "active": a}
                 for i, (n, d, t_, e, r, a) in enumerate(ARGS)],
        "fixed": {},
        "entries": {"tangent": "rlw_taumol_batched__tl (forward geometry; value float with the forward's rounding, tangent float64)",
                    "adjoint": "rlw_taumol_batched__ad (grid ceil(ncol*nl/64) x 64; per-thread tape P7_TMAX 512, reset per g-point)"},
        "precision": "float32 primal identical to the forward (augmented primal checked word for word); float64 derivative storage and arithmetic",
        "frozen": ["laytrop_v and isv (integer indices jp/jt/jt1/indself/indfor/indminor from rlw_setcoef)",
                   "eta binning: js = 1 + (int)specmult with fs = fmodf(specmult, 1) (the fraction is differentiated, the bin is the trajectory's)",
                   "the oneminus clamp `if (specparm >= oneminus) specparm = oneminus` (clamped: zero derivative, the limit is a constant)",
                   "the specparm < 0.125 / > 0.875 stencil choice (band 12 inline; bands 3-5 via rlw_b35_facs, 7 and 9 via rlw_specfac_0609 / rlw_ninemajor_0609, 13, 15 and 16 via rlw_nine_major_b1316)",
                   "the ratn2o > 1.5 (band 3), ratco2 > 3.0 (bands 6, 7) and pp < 250 (band 1) corrections",
                   "lay <= laytrop (lower/upper atmosphere)"],
        "hazards": "hazards.json beside the bundle row (line numbers in the captured source 683d7a07)",
        "plan_binding": "bindable: --native-schemes 1 binds it live through the float32-shadow adapter wherever a plan issues the launch. "
                        "pkg18 does not: radiation is a held upload there (named gap, owner P1/P7: the LW wrapper's host batch logic in the Rust plan and a re-seal with radiation live)",
    })
    json.dump(t, open(os.path.join(bout, "schemes.json"), "w"), indent=1)
    with open(os.path.join(bout, "SHA256SUMS"), "w") as f:
        for root, _, files in os.walk(bout):
            for n in sorted(files):
                if n == "SHA256SUMS": continue
                p = os.path.join(root, n); f.write(f"{sha(p)}  {os.path.relpath(p, bout)}\n")
    print(json.dumps({"bundle": bout, "rows": len(t["rows"]), "image": sha(image), "generated": sha(gen)}))

if __name__ == "__main__":
    main(*sys.argv[1:6])
