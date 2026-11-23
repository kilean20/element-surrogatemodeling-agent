"""Manufactured selection controls; no facility models or timing constants."""

import copy

import pytest

from element_surrogate.benchmark.protocol import read_json, validate_protocol, write_json
from element_surrogate.benchmark.report import build_report
from element_surrogate.benchmark.study import choose, create_study, lock_selection, run_study
from tests.benchmark_fixture import approval, make_case


def case_rows():
    p = {
        "candidates": [],
        "selection": {
            "policy": "accuracy_with_latency_budget",
            "max_median_us_per_row": 4.0,
            "budget_rationale": "Synthetic application constraint",
            "relative_accuracy_tie": 0.0,
        },
    }
    rows = []
    for name, score, cost in (
        ("fast", 2.0, 1.0),
        ("boundary", 1.0, 4.0),
        ("slow", 0.1, 8.0),
        ("dominated", 3.0, 2.0),
    ):
        p["candidates"].append({"id": name, "architecture": name, "seeds": [1, 2]})
        for seed in (1, 2):
            rows.append(
                {
                    "candidate_id": name,
                    "architecture": name,
                    "seed": seed,
                    "primary_score": score,
                    "cost_us_per_row": cost,
                }
            )
    return p, rows


def test_budget_is_applied_before_accuracy_and_frontier_preserves_tradeoffs():
    p, rows = case_rows()
    result = choose(p, rows)
    assert result["selected"]["candidate_id"] == "boundary"
    assert result["latency_eligible_candidate_ids"] == ["fast", "boundary", "dominated"]
    assert [r["candidate_id"] for r in result["pareto_frontier"]] == ["fast", "boundary", "slow"]
    p["selection"]["max_median_us_per_row"] = 3.999
    assert choose(p, rows)["selected"]["candidate_id"] == "fast"
    p["selection"]["max_median_us_per_row"] = 0.9
    with pytest.raises(ValueError, match="no complete configuration"):
        choose(p, rows)


def test_budget_seed_aggregation_and_relative_accuracy_tie():
    p, rows = case_rows()
    p["selection"]["relative_accuracy_tie"] = 1.0
    assert choose(p, rows)["selected"]["candidate_id"] == "fast"
    # Aggregate cost is the median across required seed timings, not the luckiest seed.
    for row in rows:
        if row["candidate_id"] == "boundary":
            row["cost_us_per_row"] = 2.0 if row["seed"] == 1 else 8.0
    assert "boundary" not in choose(p, rows)["latency_eligible_candidate_ids"]


def test_explicit_accuracy_first_and_legacy_result_shape():
    p, rows = case_rows()
    p["selection"] = {"relative_accuracy_tie": 0.0}
    legacy = choose(p, rows)
    assert legacy["selected"]["candidate_id"] == "slow"
    assert set(legacy) == {"selected", "configurations", "architecture_finalists", "selection_rule"}
    p["selection"]["policy"] = "accuracy_first"
    assert choose(p, rows)["selected"] == legacy["selected"]


@pytest.mark.parametrize(
    "change", ["zero", "nan", "negative", "missing", "unknown", "ignored", "rationale"]
)
def test_invalid_budget_protocols_are_refused(tmp_path, change):
    _, p, _ = make_case(tmp_path)
    p["selection"].update(
        policy="accuracy_with_latency_budget",
        max_median_us_per_row=2.0,
        budget_rationale="Synthetic use",
    )
    if change in ("zero", "nan", "negative"):
        p["selection"]["max_median_us_per_row"] = {"zero": 0, "nan": float("nan"), "negative": -1}[
            change
        ]
    elif change == "missing":
        del p["selection"]["max_median_us_per_row"]
    elif change == "unknown":
        p["selection"]["policy"] = "made_up"
    elif change == "ignored":
        p["selection"]["policy"] = "accuracy_first"
    else:
        p["selection"]["budget_rationale"] = ""
    with pytest.raises(ValueError):
        validate_protocol(p)


@pytest.mark.parametrize("bound", ["inclusive", "exclusive"])
def test_budget_study_report_and_immutable_lock(tmp_path, bound):
    path, p, _ = make_case(tmp_path / "case", configurations=1)
    p["selection"].update(
        policy="accuracy_with_latency_budget",
        max_median_us_per_row=1e6,
        budget_rationale="Loose synthetic smoke-test ceiling",
        latency_bound=bound,
    )
    validate_protocol(p)
    write_json(path, p)
    approvals = [
        approval(path.parent, p, gate)
        for gate in ("description_target", "region_ranges", "compute_budget")
    ]
    study = create_study(path, tmp_path / "study", approvals)["study_path"]
    # Fit/lock/report do not need the test file.
    (path.parent / "test.npz").rename(path.parent / "withheld.npz")
    assert run_study(study)["status"] == "completed"
    lock = lock_selection(study)
    assert lock_selection(study) == lock
    report = build_report(study, tmp_path / "report")
    assert report == build_report(study, tmp_path / "report", check=True)
    from pathlib import Path

    text = Path(report["report_path"]).read_text()
    assert "inference ceiling" in text and "Pareto frontier" in text
    assert "Loose synthetic" in text
    assert ("strictly below" if bound == "exclusive" else "at or below") in text
    # A changed protocol/ceiling cannot silently reuse the locked study.
    changed = copy.deepcopy(read_json(study))
    changed["protocol"]["selection"]["max_median_us_per_row"] *= 2
    write_json(study, changed)
    with pytest.raises(ValueError):
        lock_selection(study)


def test_exclusive_bound_excludes_equality_without_changing_legacy_default():
    p, rows = case_rows()
    implicit = choose(p, rows)
    p["selection"]["latency_bound"] = "inclusive"
    assert choose(p, rows)["selected"] == implicit["selected"]
    p["selection"]["latency_bound"] = "exclusive"
    strict = choose(p, rows)
    assert strict["selected"]["candidate_id"] == "fast"
    assert "boundary" not in strict["latency_eligible_candidate_ids"]
    p["selection"]["max_median_us_per_row"] = 1.0
    with pytest.raises(ValueError, match="no complete configuration"):
        choose(p, rows)


@pytest.mark.parametrize("bound", ["unrecognized", "", None])
def test_invalid_latency_bound_is_not_silently_ignored(tmp_path, bound):
    _, p, _ = make_case(tmp_path)
    p["selection"].update(
        policy="accuracy_with_latency_budget",
        max_median_us_per_row=2.0,
        budget_rationale="Synthetic use",
        latency_bound=bound,
    )
    with pytest.raises(ValueError, match="latency_bound"):
        validate_protocol(p)


def test_latency_bound_requires_cost_policy(tmp_path):
    _, p, _ = make_case(tmp_path)
    p["selection"]["latency_bound"] = "exclusive"
    with pytest.raises(ValueError, match="latency ceiling requires"):
        validate_protocol(p)


@pytest.mark.parametrize("bound, expected_seed", [("inclusive", 1), ("exclusive", 2)])
def test_checkpoint_selection_respects_bound_without_discarding_replicates(bound, expected_seed):
    p, rows = case_rows()
    p["selection"]["latency_bound"] = bound
    for row in rows:
        if row["candidate_id"] == "boundary":
            row["cost_us_per_row"] = 4.0 if row["seed"] == 1 else 1.0
            row["primary_score"] = 0.1 if row["seed"] == 1 else 0.2
    result = choose(p, rows)
    assert result["selected"]["candidate_id"] == "boundary"
    assert result["selected"]["seed"] == expected_seed
    configuration = next(r for r in result["configurations"] if r["candidate_id"] == "boundary")
    assert configuration["seed_count"] == 2
    assert configuration["mean_score"] == pytest.approx(0.15)
    check = next(r for r in result["latency_eligibility"] if r["candidate_id"] == "boundary")
    assert check["representative_seed"] == expected_seed


def test_fast_configuration_median_cannot_hide_an_over_budget_selected_checkpoint():
    p, rows = case_rows()
    p["selection"]["latency_bound"] = "exclusive"
    for row in rows:
        if row["candidate_id"] == "boundary":
            row["cost_us_per_row"] = 6.0 if row["seed"] == 1 else 1.0
            row["primary_score"] = 0.1 if row["seed"] == 1 else 0.2
    result = choose(p, rows)
    assert result["selected"]["candidate_id"] == "boundary"
    assert result["selected"]["seed"] == 2
    assert result["selected"]["cost_us_per_row"] < 4.0
    # All declared seeds still contribute; this is constrained checkpoint
    # selection, not exclusion of an inconvenient replicate from the statistics.
    configuration = next(r for r in result["configurations"] if r["candidate_id"] == "boundary")
    assert configuration["mean_score"] == pytest.approx(0.15)
    assert configuration["median_us_per_row"] == 3.5
