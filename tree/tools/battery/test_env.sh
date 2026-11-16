#!/bin/sh
# Build the CPU test environment for this tree, the same way ci.yml does, into a venv of its own.
#
#   sh tools/battery/test_env.sh <venv-dir> [python]
#
# then, from tree/:
#
#   GPUWM_HEX_NO_LOCAL_GPU=1 <venv-dir>/bin/python -m pytest -q -p no:cacheprovider -rfEs tests/...
#
# THE BREAKAGE THIS PREVENTS: run in a shared venv, the battery imported an older installed gpuwm (2.9.12 of
# 2026-10-31, without gpuwm.hip_backend, gpuwm.gpu_smi, gpuwm.backend) or an installed gpuwm-hex 0.4.x, and the
# door tests failed by the hundred on ModuleNotFoundError, which hid the failures that are real. Here the engine is
# the one pyproject.toml declares (resolved from PyPI, as a user gets it), and gpuwm-hex is this tree, installed
# editable with no dependencies, so `hexcore` is tree/src for the battery and for every subprocess it starts.
# Nothing is uploaded or published. No GPU is touched: GPUWM_HEX_NO_LOCAL_GPU=1 skips the CUDA tier by name.
set -eu
venv=${1:?usage: test_env.sh <venv-dir> [python]}
py=${2:-python3}
tree=$(cd "$(dirname "$0")/../.." && pwd)
cd "$tree"
"$py" -m venv "$venv"
pip="$venv/bin/python -m pip"
$pip install -q --upgrade pip
# The runtime dependencies and the test runner, exactly as ci.yml names them.
$pip install -q "psutil>=5.9" "numpy>=2.0" "rapidgzip>=0.16.0" "numcodecs>=0.12,<0.17" "netCDF4>=1.6" \
    "scipy>=1.11" "pytest>=8.0" "setuptools>=77"
# The engine this tree declares, read from pyproject.toml so the pin has one home.
engine=$("$venv/bin/python" -c "import tomllib,pathlib;d=tomllib.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8'));print(next(s for s in d['project']['dependencies'] if s.split('>=')[0].strip()=='gpuwm'))")
echo "declared engine: $engine"
$pip install -q "$engine"
# This tree, editable, without resolving the engine again.
$pip install -q --no-deps -e .
# The regional contract's host-preparation helper, as ci.yml builds it (tests that run it look for this binary).
cargo=$(command -v cargo || echo "$HOME/.cargo/bin/cargo")
if [ -x "$cargo" ]; then
    "$cargo" build -q --release --locked --manifest-path rust/host-prep/Cargo.toml
else
    echo "cargo not found: the host-prep helper is not built; tests that run it will say so" >&2
fi
"$venv/bin/python" - <<'EOF'
import hexcore, gpuwm, importlib.metadata as m, pathlib
src = pathlib.Path("src").resolve()
where = pathlib.Path(hexcore.__file__).resolve()
print("hexcore from", where.parent)
print("gpuwm", m.version("gpuwm"), "from", pathlib.Path(gpuwm.__file__).parent)
assert src in where.parents, f"hexcore resolves outside this tree: {where}"
EOF
