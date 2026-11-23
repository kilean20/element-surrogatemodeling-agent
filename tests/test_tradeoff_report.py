"""Unselected validation evidence for manufactured 4D/5D/6D goals."""

from pathlib import Path

import numpy as np
import pytest

from element_surrogate.benchmark.protocol import file_hash, read_json, write_json
from element_surrogate.benchmark.report import build_report, build_tradeoff_report
from element_surrogate.benchmark.study import create_study, lock_selection, run_study
from tests.benchmark_fixture import approval, make_case


def prepare(tmp_path, dimension=4, *, infeasible=True):
    path, p, _ = make_case(tmp_path / "case", dimension, configurations=1)
    for candidate in p["candidates"]:
        candidate["max_steps"] = 1
    p["selection"].update(
        policy="accuracy_with_latency_budget",
        max_median_us_per_row=1e-12 if infeasible else 1e6,
        latency_bound="exclusive",
        budget_rationale="Manufactured provisional budget",
    )
    write_json(path, p)
    approvals = [
        approval(path.parent, p, g)
        for g in ("description_target", "region_ranges", "compute_budget")
    ]
    study = create_study(path, tmp_path / "study", approvals)["study_path"]
    (path.parent / "test.npz").rename(path.parent / "withheld.npz")
    assert run_study(study)["status"] == "completed"
    return Path(study), p


@pytest.mark.parametrize("dimension", [4, 5, 6])
def test_exploration_with_infeasible_ceiling_needs_no_lock_or_test(tmp_path, dimension):
    study, p = prepare(tmp_path, dimension)
    before = {str(f): file_hash(f) for f in study.parent.rglob("*") if f.is_file()}
    with pytest.raises(ValueError, match="no complete configuration"):
        lock_selection(study)
    report = build_tradeoff_report(study, tmp_path / "exploration")
    assert report == build_tradeoff_report(study, tmp_path / "exploration", check=True)
    assert not report["test_opened"] and not report["selection_lock_created"]
    assert not report["selection_lock_present"]
    assert not (study.parent / "test").exists()
    assert all(file_hash(f) == digest for f, digest in before.items())
    assert not (study.parent / "selection.json").exists()
    comparison = read_json(report["comparison_path"])
    assert comparison["coordinate_units"] == [c["unit"] for c in p["coordinates"]]
    for configuration in comparison["configurations"]:
        metrics = [
            read_json(source["trial_path"])["metrics"] for source in configuration["source_trials"]
        ]
        for name in ("rmse", "setting_mean_rmse", "p99_absolute"):
            np.testing.assert_array_equal(
                configuration["mean_seed_" + name], np.mean([m[name] for m in metrics], axis=0)
            )
        np.testing.assert_array_equal(
            configuration["worst_seed_p99_absolute"],
            np.max([m["p99_absolute"] for m in metrics], axis=0),
        )
    text = Path(report["report_path"]).read_text(encoding="utf-8")
    assert "no feasible model is asserted" in text
    assert "Mean seed P99" in text


def test_exploration_preserves_existing_lock_and_ignores_test_results(tmp_path):
    study, _ = prepare(tmp_path, infeasible=False)
    lock = lock_selection(study)
    lock_path = Path(lock["selection_path"])
    before = file_hash(lock_path)
    (study.parent / "test").mkdir()
    tripwire = study.parent / "test/result.json"
    tripwire.write_text("This is deliberately not JSON: exploration must not read it.")
    report = build_tradeoff_report(study, tmp_path / "exploration")
    assert report["selection_lock_present"] and not report["test_opened"]
    assert file_hash(lock_path) == before
    # The selected-model report correctly does inspect final results; the
    # separate exploratory route must not accidentally call that code path.
    with pytest.raises(ValueError):
        build_report(study, tmp_path / "selected")


def test_exploratory_report_registered_as_managed_scientific_operation():
    from element_surrogate import jobs
    from element_surrogate.tools import REGISTRY

    assert "build_benchmark_tradeoff_report" in jobs.TOOL_OPERATIONS
    operation = REGISTRY["build_benchmark_tradeoff_report"]
    assert operation.input_file_args == ("path",)
    assert not operation.read_only
