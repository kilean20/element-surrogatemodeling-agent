"""Scientific and workflow controls for the benchmark layer."""

import copy
import csv
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from element_surrogate.benchmark.protocol import (
    file_hash,
    load_protocol,
    protocol_identity,
    read_json,
    validate_protocol,
    write_json,
)
from element_surrogate.benchmark.report import (
    _scatter,
    build_report,
    prepare_metric_crosscheck,
    render_card,
    score_documentation,
    verify_vendor,
)
from element_surrogate.benchmark.scoring import (
    load_cohort,
    score_predictions,
    summary,
    write_predictions,
)
from element_surrogate.benchmark.study import (
    choose,
    create_study,
    evaluate_test,
    lock_selection,
    run_study,
    start_study,
    study_status,
)
from element_surrogate.regression import RegressionMetrics
from tests.benchmark_fixture import approval, make_case


def test_tradeoff_plot_retains_fast_and_slow_candidates_beyond_legend_limit():
    # The fast candidates deliberately have worse accuracy: accuracy-only
    # truncation would hide them and falsely narrow the observed cost range.
    rows = [(f"candidate_{i:02d}", 1.0 + i, 40.0 - i) for i in range(30)]
    root = ET.fromstring(_scatter(rows[::-1], "Tradeoff", "Time", "Error", focus="candidate_13"))
    circles = list(root.iter("{http://www.w3.org/2000/svg}circle"))
    assert {e.attrib["data-candidate-id"] for e in circles} == {r[0] for r in rows}
    assert len(circles) == 30
    selected = [e for e in circles if e.attrib["data-selected"] == "true"]
    assert len(selected) == 1 and selected[0].attrib["data-candidate-id"] == "candidate_13"
    named = {
        e.attrib["data-candidate-label"] for e in root.iter() if "data-candidate-label" in e.attrib
    }
    assert len(named) <= 12
    assert {"candidate_00", "candidate_29", "candidate_13"} <= named
    assert all(e.find("{http://www.w3.org/2000/svg}title") is not None for e in circles)


def test_tradeoff_plot_handles_one_point_and_escapes_candidate_identity():
    name = 'one<&"point'
    root = ET.fromstring(_scatter([(name, 1.0, 0.0)], "Example", "Time", "Error", focus=name))
    circles = list(root.iter("{http://www.w3.org/2000/svg}circle"))
    assert len(circles) == 1 and circles[0].attrib["data-candidate-id"] == name
    assert circles[0].attrib["data-selected"] == "true"


@pytest.mark.parametrize("dimension", [4, 5, 6])
def test_trained_neural_study_and_guarded_test_report(tmp_path, dimension):
    path, p, approvals = make_case(tmp_path / "case", dimension, configurations=1)
    created = create_study(path, tmp_path / "study", approvals)
    study = created["study_path"]
    assert not created["test_opened"]
    # Fitting succeeds with the test file physically absent.
    test_path = path.parent / "test.npz"
    test_path.rename(path.parent / "withheld.npz")
    assert run_study(study)["status"] == "completed"
    locked = lock_selection(study)
    assert not locked["test_used_for_selection"]
    trial = read_json(locked["selected"]["trial_path"])
    best = min(trial["history"], key=lambda r: r["validation_primary_score"])
    assert trial["best_step"] == best["step"]
    assert trial["metrics"]["primary_score"] == best["validation_primary_score"]
    initial_score = trial["history"][0]["validation_primary_score"]
    assert any(
        r["step"] > 0 and r["validation_primary_score"] != initial_score for r in trial["history"]
    )
    with np.load(locked["selected"]["checkpoint_path"], allow_pickle=False) as saved:
        if trial["best_step"] > 0:
            assert np.linalg.norm(saved["w2"]) > 0
        else:
            # Optimization was exercised, but the pre-optimization base won.
            assert best["step"] == 0
    assert trial["timing"]["p10_us_per_row"] <= trial["timing"]["p90_us_per_row"]
    with pytest.raises(ValueError, match="release"):
        evaluate_test(study, [])
    validation_report = build_report(study, tmp_path / "validation_report")
    assert validation_report["split"] == "validation"
    (path.parent / "withheld.npz").rename(test_path)
    release = approval(path.parent, p, "release")
    result = evaluate_test(study, [release])
    assert result["metrics"]["rows"] == 23
    if dimension >= 5:
        assert result["metrics"]["max_absolute"][-1] == 0
    assert evaluate_test(study, [release]) == result
    report = build_report(study, tmp_path / "test_report")
    assert build_report(study, tmp_path / "test_report", check=True) == report
    for plot in report["plot_paths"]:
        assert ET.parse(plot).getroot().tag.endswith("svg")
    assert not report["full_card_validated"]
    cross = prepare_metric_crosscheck(study, tmp_path / "crosscheck", maximum_rows=7)
    assert cross["exported_rows"] == 7 and cross["source_rows"] == 23
    with open(cross["csv_path"], newline="") as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == 7 and len({r["sample_id"] for r in records}) == 7
    assert not cross["external_evaluator_run"]
    # Resume never retrains already completed trials or mutates the lock.
    original = file_hash(locked["selection_path"])
    assert run_study(study)["status"] == "completed"
    assert lock_selection(study)["selection_sha256"] == original


@pytest.mark.parametrize(
    "mutation", ["units", "groups", "scale", "arrival", "seeds", "leakage", "budget", "native"]
)
def test_malformed_or_inconsistent_protocols_refused(tmp_path, mutation):
    _, p, _ = make_case(tmp_path)
    if mutation == "units":
        p["coordinates"][0]["unit"] = ""
    elif mutation == "groups":
        p["dataset"]["splits"]["test"]["groups"] = p["dataset"]["splits"]["train"]["groups"]
    elif mutation == "scale":
        p["coordinates"][0]["error_scale"] = float("nan")
    elif mutation == "arrival":
        p["coordinates"].append({**p["coordinates"][0], "name": "t", "role": "arrival"})
    elif mutation == "seeds":
        p["candidates"][0]["seeds"] = [1, 1]
    elif mutation == "leakage":
        p["inputs"].append("target")
    elif mutation == "budget":
        p["candidates"][0]["max_steps"] = -1
    else:
        p["physics"]["native_planes"] = False
    with pytest.raises(ValueError):
        validate_protocol(p)


def test_user_approval_and_source_changes_refused(tmp_path):
    path, p, approvals = make_case(tmp_path / "case")
    with pytest.raises(ValueError, match="approval"):
        create_study(path, tmp_path / "unapproved", [])
    altered = copy.deepcopy(p)
    altered["timing"]["batch_size"] += 1
    write_json(path, altered)
    with pytest.raises(ValueError, match="approval"):
        create_study(path, tmp_path / "mismatch", approvals)
    write_json(path, p)
    (path.parent / "adapter.py").write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        create_study(path, tmp_path / "changed", approvals)


def test_dimensional_units_group_weighting_quantiles_and_batch_pooling(tmp_path):
    _, p, _ = make_case(tmp_path)
    truth = np.zeros((4, 4))
    prediction = np.array([[0, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0], [12, 0, 0, 0]], float)
    groups = np.array(["a", "a", "a", "b"])
    metrics = summary(p, prediction, truth, groups)
    assert metrics["rmse"][0] == pytest.approx(np.sqrt(40))
    assert metrics["setting_mean_rmse"][0] == pytest.approx((np.sqrt(16 / 3) + 12) / 2)
    assert metrics["p99_absolute"][0] == 12
    assert metrics["quantile_method"] == "inverted_cdf"
    first = RegressionMetrics(("mm", "mrad", "mm", "mrad"))
    first.update(prediction, truth)
    second = RegressionMetrics(first.coordinate_units)
    second.update(prediction[:1], truth[:1])
    second.update(prediction[1:], truth[1:])
    assert first.summary(quantile_method="inverted_cdf") == second.summary(
        quantile_method="inverted_cdf"
    )


@pytest.mark.parametrize(
    "damage",
    ["missing", "duplicate", "nonfinite", "units", "identity", "shape", "object", "memory"],
)
def test_prediction_join_and_artifact_contract(tmp_path, damage):
    path, p, _ = make_case(tmp_path)
    cohort = load_cohort(p, path.parent, "validation")
    target = tmp_path / "prediction.npz"
    checksum = "a" * 64
    write_predictions(target, p, "validation", cohort, cohort["targets"], checksum)
    with np.load(target, allow_pickle=False) as data:
        values = {name: data[name] for name in data.files}
    order = np.arange(len(cohort["targets"]))[::-1]
    values["sample_id"], values["prediction"] = (
        values["sample_id"][order],
        values["prediction"][order],
    )
    np.savez(target, **values)
    assert score_predictions(p, path.parent, "validation", target, checksum)["primary_score"] == 0
    if damage == "missing":
        values["sample_id"] = values["sample_id"][:-1]
    elif damage == "duplicate":
        values["sample_id"][0] = values["sample_id"][1]
    elif damage == "nonfinite":
        values["prediction"][0, 0] = np.nan
    elif damage == "units":
        values["coordinate_units"] = np.array(["m"] * 4)
    elif damage == "identity":
        values["protocol_sha256"] = "f" * 64
    elif damage == "shape":
        values["prediction"] = values["prediction"][:, :3]
    elif damage == "object":
        values["prediction"] = np.array([object()], dtype=object)
    else:
        p["resources"]["max_npz_uncompressed_bytes"] = 20
    np.savez(target, **values)
    with pytest.raises(ValueError):
        score_predictions(p, path.parent, "validation", target, checksum)


def test_selection_groups_seeds_and_hyperparameters(tmp_path):
    _, p, _ = make_case(tmp_path)
    p["candidates"] = []
    rows = []
    for name, architecture, scores, cost in (
        ("lucky", "a", [0.1, 0.9], 1),
        ("stable", "b", [0.4, 0.4], 2),
        ("bad_ridge", "b", [0.8, 0.8], 1),
    ):
        p["candidates"].append({"id": name, "architecture": architecture, "seeds": [0, 1]})
        rows.extend(
            {
                "candidate_id": name,
                "architecture": architecture,
                "seed": seed,
                "primary_score": score,
                "cost_us_per_row": cost,
            }
            for seed, score in enumerate(scores)
        )
    assert choose(p, rows)["selected"]["candidate_id"] == "stable"
    # One additional bad ridge configuration does not dilute a good configuration.
    assert len(choose(p, rows)["architecture_finalists"]) == 2
    with pytest.raises(ValueError, match="incomplete"):
        choose(p, rows[:-1])
    p["selection"]["relative_accuracy_tie"] = 0.3
    assert choose(p, rows)["selected"]["candidate_id"] == "lucky"


def test_budget_timeout_and_incomplete_policy(tmp_path):
    path, p, _ = make_case(tmp_path / "case", configurations=1)
    p["candidates"][0]["max_wall_seconds"] = 0.02
    write_json(path, p)
    approvals = [
        approval(path.parent, p, gate)
        for gate in ("description_target", "region_ranges", "compute_budget")
    ]
    study = create_study(path, tmp_path / "study", approvals)["study_path"]
    status = run_study(study)
    assert {r["status"] for r in status["trials"]} == {"timed_out"}
    with pytest.raises(ValueError, match="incomplete"):
        lock_selection(study)


def test_background_start_no_duplicate_launch_and_mcp_registry(tmp_path):
    path, _, approvals = make_case(tmp_path / "case", configurations=1)
    study = create_study(path, tmp_path / "study", approvals)["study_path"]
    launched = start_study(study)
    assert launched["launched"]
    assert not start_study(study)["launched"]
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = study_status(study)
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert status["status"] == "completed", status
    from element_surrogate.tools import REGISTRY

    assert not REGISTRY["start_benchmark_study"].read_only
    assert REGISTRY["validate_benchmark_protocol"].read_only


def test_vendor_hashes_and_incomplete_rubric_refused(tmp_path):
    record = verify_vendor()
    assert len(record["sha256"]) == 4
    rubric = tmp_path / "rubric.yaml"
    rubric.write_text("benchmark_name: test\n")
    with pytest.raises(ValueError):
        score_documentation(rubric, tmp_path / "score.md")


def test_locked_artifacts_and_generated_card_changes_refused(tmp_path):
    path, _, approvals = make_case(tmp_path / "case", configurations=1)
    study = create_study(path, tmp_path / "study", approvals)["study_path"]
    run_study(study)
    locked = lock_selection(study)
    report = build_report(study, tmp_path / "report")
    card = Path(report["card_path"])
    card.write_text(
        card.read_text().replace("Benchmark results (from metrics.json)", "Altered results")
    )
    with pytest.raises(ValueError):
        render_card(study, tmp_path / "report", check=True)
    checkpoint = Path(locked["selected"]["checkpoint_path"])
    with checkpoint.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checkpoint"):
        lock_selection(study)


def test_protocol_round_trip_is_identified(tmp_path):
    path, p, _ = make_case(tmp_path)
    assert protocol_identity(load_protocol(path)) == protocol_identity(p)
    with pytest.raises(ValueError, match="test scoring"):
        score_predictions(p, path.parent, "test", tmp_path / "absent.npz", "a" * 64)


def test_fast_eligible_configuration_is_not_hidden_by_family_grouping(tmp_path):
    _, p, _ = make_case(tmp_path)
    rows = []
    for candidate, score, cost in (
        (p["candidates"][0], 1.0, 10.0),
        (p["candidates"][1], 1.005, 1.0),
    ):
        rows += [
            {
                "candidate_id": candidate["id"],
                "architecture": candidate["architecture"],
                "seed": seed,
                "primary_score": score,
                "cost_us_per_row": cost,
            }
            for seed in candidate["seeds"]
        ]
    assert choose(p, rows)["selected"]["candidate_id"] == "wide"


def test_benchmark_protocol_and_creation_through_mcp(tmp_path):
    import asyncio

    from element_surrogate.mcp_server import build_server
    from element_surrogate.runs import RunStore

    path, p, approvals = make_case(tmp_path / "case", configurations=1)
    server = build_server(RunStore(tmp_path / "records"))
    result = asyncio.run(server.call_tool("validate_benchmark_protocol", {"path": str(path)}))
    assert not result.is_error
    assert result.structured_content["protocol_sha256"] == protocol_identity(p)
    result = asyncio.run(
        server.call_tool(
            "create_benchmark_study",
            {
                "path": str(path),
                "output_dir": str(tmp_path / "study"),
                "approval_paths": approvals,
            },
        )
    )
    assert not result.is_error
    study = result.structured_content["study_path"]
    created_run_id = result.structured_content["run_id"]
    status = asyncio.run(server.call_tool("benchmark_study_status", {"path": study}))
    assert status.structured_content["status"] == "created"
    from mcp.server.mcpserver.exceptions import ToolError

    with pytest.raises(ToolError, match="side effects"):
        asyncio.run(server.call_tool("replay_run", {"run_id": created_run_id}))
    with pytest.raises(ToolError, match="release"):
        asyncio.run(
            server.call_tool(
                "evaluate_benchmark_test",
                {
                    "path": study,
                    "approval_paths": [],
                },
            )
        )


def test_exclusive_record_and_duplicate_json_keys(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "immutable.json"

    def write(value):
        try:
            write_json(path, {"value": value}, exclusive=True)
            return True
        except FileExistsError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(write, [1, 2])) == 1
    assert read_json(path)["value"] in (1, 2)
    path.write_text('{"same":1,"same":2}')
    with pytest.raises(ValueError, match="duplicate JSON"):
        read_json(path)


def test_single_seed_does_not_claim_zero_variability(tmp_path):
    _, p, _ = make_case(tmp_path)
    p["candidates"] = [{"id": "one", "architecture": "control", "seeds": [0]}]
    rows = [
        {
            "candidate_id": "one",
            "architecture": "control",
            "seed": 0,
            "primary_score": 1.0,
            "cost_us_per_row": 1.0,
        }
    ]
    assert choose(p, rows)["configurations"][0]["std_score"] is None


def test_generic_replay_never_reexecutes_mutating_tool(tmp_path):
    from element_surrogate.runs import RunStore, Tool, record_call, replay

    calls = []

    def action():
        calls.append(1)
        return {"calls": len(calls)}

    store = RunStore(tmp_path)
    tool = Tool("mutating", action, "1", read_only=False)
    run_id = record_call(store, tool, {})["run_id"]
    with pytest.raises(ValueError, match="side effects"):
        replay(store, {tool.name: tool}, run_id)
    assert calls == [1]


def test_open_status_reader_does_not_block_atomic_replace(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from element_surrogate.file_io import artifact_guard

    path = tmp_path / "status.json"
    write_json(path, {"version": 1})
    entered, release, writer_started = Event(), Event(), Event()

    def reader():
        with artifact_guard(path):
            with path.open() as stream:
                entered.set()
                assert release.wait(10)
                return stream.read()

    def writer():
        writer_started.set()
        write_json(path, {"version": 2})

    with ThreadPoolExecutor(max_workers=2) as pool:
        read = pool.submit(reader)
        assert entered.wait(10)
        write = pool.submit(writer)
        assert writer_started.wait(10)
        assert not write.done()
        release.set()
        assert '"version": 1' in read.result()
        write.result()
    assert read_json(path) == {"version": 2}
