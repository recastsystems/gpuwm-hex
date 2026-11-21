"""Settings the hex column route accepts and cannot honour are refused by name (refused_settings).

Mechanism from fork jacobbieker/woof PR 302 (Apache-2.0); the rows are the ones checked against this engine.
"""
from __future__ import annotations

import pytest

from hexcore.physics_suites import PhysicsSelectionRefusal, load_table, resolve_physics


@pytest.mark.parametrize("overrides", [{"slope_rad": "1"}, {"topo_shading": "1"}, {"shadlen": "25000"}])
def test_a_setting_hex_cannot_honour_is_refused_by_name(overrides):
    with pytest.raises(PhysicsSelectionRefusal) as caught:
        resolve_physics("arwen_mpas_column", overrides)
    (key,) = overrides
    assert key in str(caught.value)
    assert "neighbour terrain" in str(caught.value) or "topo_shading" in str(caught.value)


@pytest.mark.parametrize("overrides", [{"slope_rad": "0"}, {"topo_shading": "0"}])
def test_the_off_value_passes(overrides):
    resolve_physics("arwen_mpas_column", overrides)


def test_every_row_says_why_and_what_runs_instead():
    for key, row in load_table()["refused_settings"]["settings"].items():
        assert row["reason"].strip() and row["follow_up"].strip(), key
        assert ("refused_values" in row) != ("inactive_values" in row), key
