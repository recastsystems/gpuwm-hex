#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# S09 (private autodiff): rebuild the kset-int7f kernel set from this tree alone.
#
#   python3 tree/ad/kset/int7f/build_kset_int7f.py --out DIR [--toolchain 13.2.86] [--tc ~/.cache/ad-kset-tc]
#
# recipe.json names, for each of the 150 derivative images, the nvcc toolchain that built the image now in kset-int7f
# (read from the image itself: the ptxas version note in every cubin), the directory it ran in, the source, the exact
# flags and the image sha256. Every source and header is in this tree (tree/ad/..., G8 at tree/ad/kset/g8, the rest
# vendored under src/ with their sha256 in the recipe). Toolchains are the NVIDIA PyPI wheels pinned by sha256
# (toolchains.json), one venv per version. Rows marked "host" run in the Containerfile.ubuntu-24.04 image (the CUDA
# 13.0 headers need glibc <= 2.39); a row with "mount_as" is compiled at that absolute path (its file-scope tables are
# named by a hash of the path). The kernel set is assembled in DIR: cubins/, specs/ (spec paths rewritten into DIR),
# derived/, derived.pv/, kernels.json; every spec record is checked against its file (the binder's admission).
# --toolchain V builds every row with V instead (the 13.2.86 acceptance set). DIR/REBUILD.json is the receipt.
import argparse, glob, hashlib, json, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../../.."))
IMAGE = "ad-kset-host-ubuntu2404"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find_nvcc(d):
    hit = glob.glob(f"{d}/lib/python3*/site-packages/nvidia/cu13/bin/nvcc")
    return hit[0] if hit else None


def toolchain(tc, v, pins):
    d = os.path.join(tc, v)
    nv = find_nvcc(d)
    if nv and os.path.exists(f"{d}/.ok"):
        return d, nv
    subprocess.run([sys.executable, "-m", "venv", "--clear", d], check=True)
    req = f"{d}/requirements.txt"
    with open(req, "w") as f:
        for p in pins[v]:
            f.write(f"{p['package']}=={p['version']} " + " ".join(f"--hash=sha256:{w['sha256']}" for w in p["wheels"]) + "\n")
    subprocess.run([f"{d}/bin/pip", "install", "-q", "--no-deps", "--require-hashes", "-r", req], check=True)
    nv = find_nvcc(d)
    ver = subprocess.run([nv, "--version"], capture_output=True, text=True).stdout
    if f"V{v}" not in ver:
        raise SystemExit(f"toolchain {v}: nvcc reports {ver.strip().splitlines()[-2:]}")
    open(f"{d}/.ok", "w").write(ver)
    return d, nv


def engine():
    for e in ("podman", "docker"):
        if shutil.which(e):
            return e
    raise SystemExit("a row needs the ubuntu-24.04 host image and neither podman nor docker is installed")


def image(e):
    probe = [e, "image", "exists", IMAGE] if e == "podman" else [e, "image", "inspect", IMAGE]
    if subprocess.run(probe, capture_output=True).returncode:
        subprocess.run([e, "build", "-q", "-t", IMAGE, "-f", f"{HERE}/Containerfile.ubuntu-24.04", HERE], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tc", default=os.path.expanduser("~/.cache/ad-kset-tc"))
    ap.add_argument("--toolchain", help="build every row with this nvcc version (e.g. 13.2.86) instead of the recipe's")
    ap.add_argument("--only", help="comma-separated functions")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    t0 = time.time()
    rec = json.load(open(f"{HERE}/recipe.json"))
    pins = json.load(open(f"{HERE}/toolchains.json"))
    rows = [r for r in rec["rows"] if not a.only or r["function"] in a.only.split(",")]
    # 1. every source byte is the one the image was built from
    bad = [(r["function"], p) for r in rows for p, h in r["source_sha256"].items() if sha(f"{REPO}/{p}") != h]
    if bad:
        raise SystemExit(f"{len(bad)} source file(s) differ from the recipe: {bad[:5]}")
    out = os.path.abspath(a.out)
    os.makedirs(f"{out}/cubins", exist_ok=True)
    vers = sorted({a.toolchain or r["toolchain"] for r in rows})
    tcs = {v: toolchain(a.tc, v, pins) for v in vers}
    on_host = lambda r: bool(r.get("mount_as") or (not a.toolchain and r.get("host")))
    eng = engine() if any(on_host(r) for r in rows) else None
    if eng:
        image(eng)

    def build(r):
        v = a.toolchain or r["toolchain"]
        f = r["function"]
        tgt = f"{out}/cubins/{f}.fatbin"
        tcd, nv = tcs[v]
        t = time.time()
        if on_host(r):
            rel = os.path.relpath(nv, tcd)
            src = f"{r['mount_as']}/{r['src']}" if r.get("mount_as") else r["src"]
            wd = r.get("mount_as") or f"/repo/{r['cwd']}"
            cmd = [eng, "run", "--rm", "--network=none", "-v", f"{REPO}:/repo:ro", "-v", f"{tcd}:/tc:ro",
                   "-v", f"{out}/cubins:/out"]
            if r.get("mount_as"):
                cmd += ["-v", f"{REPO}/{r['cwd']}:{r['mount_as']}:ro"]
            cmd += ["-w", wd, IMAGE, f"/tc/{rel}"] + r["flags"] + ["-o", f"/out/{f}.fatbin", src]
            p = subprocess.run(cmd, capture_output=True, text=True)
        else:
            env = dict(os.environ, PATH=os.path.dirname(nv) + ":" + os.environ.get("PATH", ""))
            p = subprocess.run([nv] + r["flags"] + ["-o", tgt, r["src"]], cwd=f"{REPO}/{r['cwd']}",
                               capture_output=True, text=True, env=env)
        h = sha(tgt) if p.returncode == 0 and os.path.exists(tgt) else None
        return {"function": f, "toolchain": v, "in_host_image": on_host(r), "rc": p.returncode, "sha256": h,
                "want": r["fatbin_sha256"], "byte_equal": h == r["fatbin_sha256"],
                "byte_reproducible": r.get("byte_reproducible", True), "seconds": round(time.time() - t, 2),
                "stderr": p.stderr[-2000:] if p.returncode else ""}

    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(build, rows))
    failed = [x for x in res if x["rc"]]
    # 2. assemble the kernel set beside the cubins
    kj = json.load(open(f"{HERE}/kernels.json"))
    for d in ("derived", "derived.pv"):
        os.makedirs(f"{out}/{d}", exist_ok=True)
        for fn in os.listdir(f"{HERE}/{d}"):
            shutil.copyfile(f"{HERE}/{d}/{fn}", f"{out}/{d}/{fn}")
            if fn.endswith(".bin") and sha(f"{out}/{d}/{fn}") != fn[:-4]:
                raise SystemExit(f"derived {fn} is not its sha256")
    stale, checked = [], 0
    for e in kj["kernels"]:
        d = e.get("derivative")
        if not (isinstance(d, dict) and isinstance(d.get("spec"), dict) and d["spec"].get("path")):
            continue
        p = d["spec"]["path"]
        dst = f"{out}/specs{p}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(f"{HERE}/specs{p}", dst)
        d["spec"]["path"] = dst
        checked += 1
        if d["spec"].get("sha256") != sha(dst):
            stale.append(e["function"])
    raw = json.dumps(kj, indent=1).encode()
    open(f"{out}/kernels.json", "wb").write(raw)
    cub = {f: sha(f"{out}/cubins/{f}") for f in sorted(os.listdir(f"{out}/cubins"))}
    summary = {"kset": rec["kset"], "repo": REPO, "out": out, "toolchain_override": a.toolchain, "toolchains": vers,
               "rows": len(res), "built": len(res) - len(failed), "byte_equal": sum(x["byte_equal"] for x in res),
               "differ": [x["function"] for x in res if x["rc"] == 0 and not x["byte_equal"]],
               "failed": [x["function"] for x in failed], "spec_records_checked": checked,
               "stale_spec_records": stale, "kernels_json_sha256": hashlib.sha256(raw).hexdigest(),
               "cubins_list_sha256": hashlib.sha256(json.dumps(cub, sort_keys=True).encode()).hexdigest(),
               "seconds": round(time.time() - t0, 1), "host": os.uname().nodename, "jobs": a.jobs}
    json.dump(dict(summary, images=res), open(f"{out}/REBUILD.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    for x in failed:
        print("FAILED", x["function"], x["stderr"][-400:], file=sys.stderr)
    sys.exit(1 if failed or stale else 0)


if __name__ == "__main__":
    main()
