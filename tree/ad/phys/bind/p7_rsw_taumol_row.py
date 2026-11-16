#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ad/held-rsw_taumol_b: the rsw_taumol_b row (ad.scheme.rows/v1, kind p7_rsw_taumol) as data, added to a copy of an
L2 scheme bundle. Build time only.

  p7_rsw_taumol_row.py BUNDLE_IN BUNDLE_OUT IMAGE GEN_SOURCE CAPTURE_MANIFEST

BUNDLE_OUT gets every file of BUNDLE_IN (hard links when possible, never edited in place), the image as
p7/rsw_taumol.sm_120.cubin and a schemes.json with images.p7_rsw_taumol and the row. The row's arguments are the
forward's (hexcore/mpasphys/rrtmg_sw_v842.py:1712, captured source 3c499c7b kernel rsw_taumol_b), checked against the
capture manifest's argument names, kinds and dtypes; extents are in the launch's own ncol and nlayers.
"""
import hashlib, json, os, shutil, sys

PIN = "3c499c7ba10221067f80e300b1003e382a66245027ae997f254005f80b74e700"
L2 = "ncol*nlayers"
ARGS = [  # name, device, dtype, extent, role, active
    ("ncol", False, "<i4", None, "scalar", False),
    ("nlayers", False, "<i4", None, "scalar", False),
    ("laytrop", True, "<i4", "ncol", "in-branch", False),       # tropopause layer per column (rsw_setcoef_b): lower/upper switch
    ("tab", True, "<f4", "104303", "const", False),             # packed k-coefficient, Rayleigh and cloud tables (static)
    ("ngb", True, "<i4", "112", "table", False),                # g-point -> band (16..29)
    ("jp", True, "<i4", L2, "in-branch", False),                # pressure index (rsw_setcoef_b)
    ("jt", True, "<i4", L2, "in-branch", False),                # temperature index, reference pressure jp
    ("jt1", True, "<i4", L2, "in-branch", False),               # temperature index, reference pressure jp + 1
    ("colh2o", True, "<f4", L2, "in", True),                    # column amounts (molecules/cm2), per layer
    ("colco2", True, "<f4", L2, "in", True),
    ("colch4", True, "<f4", L2, "in", True),
    ("colo2", True, "<f4", L2, "in", True),
    ("colo3", True, "<f4", L2, "in", True),
    ("colmol", True, "<f4", L2, "in", True),
    ("fac00", True, "<f4", L2, "in", True),                     # pressure/temperature interpolation factors
    ("fac01", True, "<f4", L2, "in", True),
    ("fac10", True, "<f4", L2, "in", True),
    ("fac11", True, "<f4", L2, "in", True),
    ("selffac", True, "<f4", L2, "in", True),                   # water vapour self-continuum factor and fraction
    ("selffrac", True, "<f4", L2, "in", True),
    ("indself", True, "<i4", L2, "in-branch", False),
    ("forfac", True, "<f4", L2, "in", True),                    # foreign-continuum factor and fraction
    ("forfrac", True, "<f4", L2, "in", True),
    ("indfor", True, "<i4", L2, "in-branch", False),
    ("taug", True, "<f4", "112*ncol*nlayers", "out", True),     # gas optical depth, every g-point
    ("taur", True, "<f4", "112*ncol*nlayers", "out", True),     # Rayleigh optical depth, every g-point
]


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(bin_, bout, image, gen, manifest):
    m = json.load(open(manifest))
    if m["source"]["sha256"] != PIN: raise SystemExit(f"capture source {m['source']['sha256']} is not the pinned {PIN}")
    got = [(a["name"], a["kind"] == "device", a["dtype"]) for a in m["arguments"]]
    want = [(n, d, t) for n, d, t, *_ in ARGS]
    if got != want: raise SystemExit(f"capture arguments {got} differ from the row {want}")
    if os.path.exists(bout): raise SystemExit(f"{bout} exists; the bundle is written once")
    shutil.copytree(bin_, bout, copy_function=lambda s, d: os.link(s, d) if os.stat(s).st_dev == os.stat(os.path.dirname(d) or ".").st_dev else shutil.copy2(s, d))
    os.unlink(os.path.join(bout, "schemes.json"))
    if os.path.exists(os.path.join(bout, "SHA256SUMS")): os.unlink(os.path.join(bout, "SHA256SUMS"))
    t = json.load(open(os.path.join(bin_, "schemes.json")))
    os.makedirs(os.path.join(bout, "p7"), exist_ok=True)
    shutil.copy2(image, os.path.join(bout, "p7", "rsw_taumol.sm_120.cubin"))
    t["images"]["p7_rsw_taumol"] = "p7/rsw_taumol.sm_120.cubin"
    if any(r["function"] == "rsw_taumol_b" for r in t["rows"]): raise SystemExit("bundle already has a rsw_taumol_b row")
    t["rows"].append({
        "function": "rsw_taumol_b", "unit": "P7", "kind": "p7_rsw_taumol", "p8_class": "entry",
        "source": "hexcore/mpasphys/rrtmg_sw_v842.py:1712 (RRTMG-SW taumol_sw, MPAS module_ra_rrtmg_sw.F:3201-4666, bands 16-29, one launch per column batch)",
        "source_sha256": PIN, "generator": "tree/ad/phys/p7/gen/gen_rsw_taumol.py", "generated_sha256": sha(gen),
        "image_sha256": sha(image),
        "nvrtc_options": ["-std=c++17", "--ftz=false", "--fmad=false", "--prec-div=true", "--prec-sqrt=true", "-arch=sm_120"],
        "args": [{"name": n, "pos": i, "device": d, "dtype": t_, **({"extent": e} if e else {}), "role": r, "active": a}
                 for i, (n, d, t_, e, r, a) in enumerate(ARGS)],
        "fixed": {},
        "entries": {"tangent": "rsw_taumol_b__tl (forward geometry; value float through the forward's own rounding helpers, tangent float64)",
                    "adjoint": "rsw_taumol_b__ad (grid ceil(ncol*nlayers/64) x 64; per-thread tape P7SW_TMAX 96, reset per g-point; 112 g-points in ascending order)"},
        "precision": "float32 primal identical to the forward (augmented primal checked word for word); float64 derivative storage and arithmetic",
        "frozen": ["laytrop, ngb, jp/jt/jt1, indself/indfor (integer inputs from rsw_setcoef_b: table indices and switches)",
                   "eta binning: js = 1 + (int)specmult with fs = specmult - truncf(specmult) (the fraction is differentiated, the bin is the trajectory's)",
                   "the limiter-like clamp `if (specparm >= RSW_C_ONEMINUS) specparm = RSW_C_ONEMINUS` in rsw_spec (clamped: zero derivative, the limit is a constant)",
                   "lay <= laytrop (lower/upper atmosphere)",
                   "the k-coefficient table tab and the band constants RSW_C_* are constants"],
        "hazards": "hazards.json beside the bundle row (line numbers in the captured source 3c499c7b)",
        "plan_binding": "bindable: --native-schemes 1 binds it live through the float32-shadow adapter wherever a plan issues the launch. "
                        "pkg18 does not: radiation is a held upload there (named gap, owner P1/P7: the radiation wrappers' host batch logic in the Rust plan and a re-seal with radiation live)",
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
