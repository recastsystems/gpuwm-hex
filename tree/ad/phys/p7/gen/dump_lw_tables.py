# Test-fixture preparation only (outside any step loop): rebuild the RRTMG-LW
# band coefficient tables exactly as hex builds them, and dump them to files so
# the Rust taumol harness can bind the captured launch's table pointers.
import json, sys, hashlib, numpy as np
hexsrc, table, decoder, out = sys.argv[1:5]
sys.path.insert(0, hexsrc)
import cupy as cp
from hexcore.mpasphys import rrtmg_lw_v842 as lw
K = lw.load_tables_device(table, decoder=decoder)
cp.cuda.runtime.deviceSynchronize()
index = {"bands": {}, "chi": None}
for band in range(1, 17):
    arrays, _ = K["band"][band]
    names = lw._legacy.GPU_BAND_TABS[band]
    rows = []
    for i, (n, a) in enumerate(zip(names, arrays)):
        h = cp.asnumpy(a).astype("<f4")
        fn = f"b{band:02d}_{i}.bin"
        h.tofile(f"{out}/{fn}")
        rows.append({"name": n, "file": fn, "words": int(h.size), "sha256": hashlib.sha256(h.tobytes()).hexdigest()})
    index["bands"][str(band)] = rows
chi = cp.asnumpy(K["chi"]).astype("<f4"); chi.tofile(f"{out}/chi.bin")
index["chi"] = {"file": "chi.bin", "words": int(chi.size), "sha256": hashlib.sha256(chi.tobytes()).hexdigest()}
index["oneminus_hex"] = np.float32(K["oneminus"]).tobytes().hex()
json.dump(index, open(f"{out}/index.json", "w"), indent=1)
print("bands", sum(len(v) for v in index["bands"].values()), "chi", chi.size, index["oneminus_hex"])
