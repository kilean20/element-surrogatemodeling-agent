"""The corpus runner: every planted defect is caught by the check meant for it, and clean
fields raise no findings (the M1 acceptance for the readiness checks)."""

import pytest

from element_surrogate.corpus import CHECKS, run_corpus


@pytest.fixture(scope="module")
def report():
    return run_corpus()


def test_every_planted_defect_is_caught_by_its_check(report):
    planted = [row for row in report["cases"] if row["planted"]]
    assert len(planted) == report["planted"] == 12
    assert report["missed"] == []
    for row in planted:
        assert set(row["expected"]) <= set(row["flagged_by"]), row["case"]
        assert set(row["expected_observations"]) <= set(row["observed_by"]), row["case"]


def test_clean_fields_raise_no_findings_and_every_check_runs(report):
    assert report["false_alarms"] == {}
    assert report["errors"] == {}
    clean = [row for row in report["cases"] if not row["planted"]]
    assert {row["case"] for row in clean} == {"clean quadrupole", "clean bend"}
    assert all(set(row["checks"]) == set(CHECKS) for row in report["cases"])


def test_exact_zero_and_invariant_vacuum_fields_are_not_corruption(report):
    controls = report["exact_vacuum_nodes"]
    assert len(controls) == 3
    assert all(not r["node_findings"] and not r["harmonic_findings"] for r in controls)
    assert all(
        r["max_curl_t_per_m"] < 1e-12 and r["max_abs_divergence_t_per_m"] < 1e-12 for r in controls
    )
