"""A storm measure (rain water, cloud water, vertical velocity) prints the row's measured line before it runs."""
import argparse

from hexcore import ad_doors


def _args(**kw):
    base = {"row": None, "mesh": None, "metric": "theta_m"}
    base.update(kw)
    return argparse.Namespace(**base)


def test_storm_measure_line_on_the_example_rows():
    table = ad_doors.load_table()
    block = table["storm_measures"]
    assert set(block["metrics"]) == {"qr", "qc", "w"}
    for row in ("hrrrv5-q12.96.2890.n40.60w91.30", "hrrrv5-q3.96.38278.n40.60w91.30"):
        assert row in table["rows"] and row in block["rows"]
        for metric in ("qr", "w"):
            lines = ad_doors.storm_measure_lines(table, _args(row=row, metric=metric), "adjoint-test", [metric])
            assert len(lines) == 1 and "no PASS of this release covers it" in lines[0] and "LIMITS.md" in lines[0]
    twelve = ad_doors.storm_measure_lines(table, _args(row="hrrrv5-q12.96.2890.n40.60w91.30"), "sensitivity",
                                          ["qr", "w", "theta_m"])
    assert "0.118" in twelve[0] and "(qr, w)" in twelve[0]


def test_no_line_for_the_readme_measures():
    table = ad_doors.load_table()
    for metric in ("theta_m", "theta", "qv", None):
        assert ad_doors.storm_measure_lines(table, _args(row="hrrrv5-q12.96.2890.n40.60w91.30"), "sensitivity",
                                            [metric]) == []
    assert ad_doors.storm_measure_lines(table, _args(mesh="x1.2562", metric="theta_m"), "sensitivity", ["theta_m"]) == []


def test_a_named_mesh_says_it_has_no_reading():
    table = ad_doors.load_table()
    lines = ad_doors.storm_measure_lines(table, _args(mesh="x1.2562", metric="qr"), "sensitivity", ["qr"])
    assert len(lines) == 1 and "no PASS of this release covers it" in lines[0] and "a mesh named with --mesh (x1.2562)" in lines[0]


def test_a_row_without_a_reading_says_so():
    table = ad_doors.load_table()
    other = next(r for r in table["rows"] if r not in table["storm_measures"]["rows"])
    lines = ad_doors.storm_measure_lines(table, _args(row=other, metric="w"), "adjoint-test", ["w"])
    assert lines and "has not been read on row" in lines[0]
