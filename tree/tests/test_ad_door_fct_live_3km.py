"""F12: the 3 km storm row runs the transport limiter live (its exact derivative) and holds it past fct_live_max_s.

The held limiter put the 3 km adjoint-test's bump 1.1 per cent off the real model at every step size (dist49 at3: 1.0110
to 1.0142, outside the 1 per cent bar); the limiter's exact derivative grows on windows past about 30 minutes at 3 km.
"""
import json
from pathlib import Path

import hexcore.ad_doors as d

STORM = "hrrrv5-q3.96.38278.n40.60w91.30"


def _words(argv, flag):
    return argv[argv.index(flag, 2) + 1]


def _toy(tmp_path: Path, live_max):
    (tmp_path / "k").write_text("x")
    row = {"asset_args": {"--kernels": "kernels"}, "asset_env": {}, "window_args": ["--fct-frozen", "0", "--anchors", "2"],
           "dt_s": 15.0}
    if live_max is not None:
        row["fct_live_max_s"] = live_max
    table = {"doors": {"adjoint-test": {"window_args": []}}}
    return table, row, {"kernels": tmp_path / "k"}


def test_live_inside_the_limit_held_past_it(tmp_path):
    table, row, assets = _toy(tmp_path, 1800)
    for steps, want in ((59, "0"), (120, "0"), (121, "1"), (239, "1")):
        argv, _ = d.window_command(table, row, assets, "adjoint-test", Path("/bin/true"), steps, tmp_path / "o", [])
        assert _words(argv, "--fct-frozen") == want, steps


def test_no_limit_leaves_the_row_word(tmp_path):
    table, row, assets = _toy(tmp_path, None)
    argv, _ = d.window_command(table, row, assets, "adjoint-test", Path("/bin/true"), 239, tmp_path / "o", [])
    assert _words(argv, "--fct-frozen") == "0"


def test_the_storm_row_runs_the_limiter_live():
    table = d.load_table()
    row = table["rows"][STORM]
    w = row["window_args"]
    assert w[w.index("--fct-frozen") + 1] == "0"
    assert row["fct_live_max_s"] == 1800 and row["dt_s"] == 15.0
    assert "FCT limiter coefficients held" not in row["what"]
