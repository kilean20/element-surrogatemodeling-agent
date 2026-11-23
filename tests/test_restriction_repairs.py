"""Meaningful controls for the restrictions observed in frozen native cases."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from element_surrogate import jobs, readiness
from element_surrogate.benchmark.protocol import read_json, write_json
from element_surrogate.card_frontmatter import compose_metrics, metric_coverage, split_card
from element_surrogate.fields import FieldMap
from tests.test_scientific_jobs import campaign, request, wait


@pytest.mark.parametrize("constant", [0.0, 0.125])
@pytest.mark.parametrize("backend", ["eager", "bounded"])
def test_constant_storage_has_no_fabricated_order_or_nonfinite_json(tmp_path, constant, backend):
    a = np.linspace(-0.02, 0.02, 5)
    field = FieldMap(a, a, a, np.full((5, 5, 5, 3), constant))
    result = readiness.storage_order(
        field, backend=backend, working_memory_bytes=256 * 1024, scratch_directory=tmp_path
    )
    json.dumps(result, allow_nan=False)
    assert result["best"] is None and result["as_read_rank"] is None
    assert result["as_read_over_best"] is None and result["identifiable_candidates"] == 0
    assert not result["findings"]
    assert "zero" in result["ratio_undefined_reason"]
    assert not list(tmp_path.iterdir())


def test_exact_zero_alternative_still_identifies_positive_as_read():
    from element_surrogate.region import Region

    rows = [
        {"as_read": True, "relative_residual": 0.5},
        {
            "as_read": False,
            "relative_residual": 0.0,
            "axes_slowest_to_fastest": ["x", "y", "z"],
            "columns_hold": ["By", "Bx", "Bz"],
        },
    ]
    result = readiness._storage_order_result(rows, {}, Region("circle", 0.01), 5, 1.01)
    assert len(result["findings"]) == 1
    assert result["as_read_over_best"] is None
    json.dumps(result, allow_nan=False)


def test_model_yaml_composition_preserves_spaced_labels_and_body():
    bundle = {"results": [{"benchmark": "Readable beam case", "metric": "x_rmse"}]}
    raw = (
        "---\nname: Example\nmetrics:\n- Readable beam case/x_rmse\n---\n\n"
        "## Evaluation\nOriginal body.\n"
    )
    result = compose_metrics(raw, bundle, ["training/loss", "Readable beam case/x_rmse"])
    front, body = split_card(result)
    assert front["metrics"] == ["training/loss", "Readable beam case/x_rmse"]
    assert body == split_card(raw)[1]
    assert metric_coverage(result, bundle)["pass"]
    assert not metric_coverage(result.replace("Readable beam case/x_rmse", "wrong/metric"), bundle)[
        "pass"
    ]
    with pytest.raises(ValueError, match="duplicate YAML"):
        split_card("---\nmetrics: []\nmetrics: []\n---\n")
    assert not metric_coverage(
        raw.replace("- Readable", "- Readable beam case/x_rmse\n- Readable"), bundle
    )["pass"]


def generation_pilot(tmp_path, *, time_policy="active"):
    from tests.test_workflow_tools import file_case
    from tests.workflow_fixture import approvals, manufactured_transport

    transport, _, plan = file_case(tmp_path)
    plan.update(settings=manufactured_transport()[3], max_wall_seconds=150)
    write_json(tmp_path / "plan.json", plan)
    decisions = approvals(tmp_path / "decisions", plan)
    path, _ = campaign(tmp_path, seconds=180)
    spec = request(tmp_path / "plan.json", maximum=40)
    spec.update(
        time_policy=time_policy,
        operation="generate_training_data",
        arguments={
            "transport_path": str(transport),
            "qualification_path": str(tmp_path / "qualification.json"),
            "plan_path": str(tmp_path / "plan.json"),
            "output_dir": str(tmp_path / "particles"),
            "approval_paths": [str(p) for p in decisions],
            "max_batches": 1,
        },
    )
    handle = jobs.submit_request(path, spec)
    first = wait(path, handle)
    assert first["status"] == "incomplete"
    inspection = tmp_path / "inspection.json"
    jobs.inspect_recovery(path, handle["job_id"], inspection)
    allocation = {
        "schema": "element_surrogate.generation_continuation.v1",
        "max_batches": None,
        "max_wall_seconds": 90,
        "rationale": "Proceed from the completed capacity pilot under the original plan",
    }
    write_json(tmp_path / "continuation.json", allocation)
    return path, handle, first, inspection, allocation


def test_actual_pilot_to_complete_with_conserved_identity_cost_and_no_fork(tmp_path):
    from element_surrogate.data import verify_generation

    path, handle, first, inspection, allocation = generation_pilot(tmp_path)
    old = verify_generation(tmp_path / "particles")
    original_request = read_json(Path(first["job_directory"]) / "request.json")["request"]
    changed_execution = deepcopy(original_request)
    changed_execution["arguments"]["max_batches"] = 3
    with pytest.raises(ValueError, match="same scientific work"):
        jobs.submit_request(path, changed_execution)
    successor = jobs.continue_generation_phase(
        path, handle["job_id"], inspection, tmp_path / "continuation.json"
    )
    assert successor["job_id"] != handle["job_id"]
    assert (
        jobs.continue_generation_phase(
            path, handle["job_id"], inspection, tmp_path / "continuation.json"
        )["job_id"]
        == successor["job_id"]
    )
    final = wait(path, successor)
    assert final["status"] == "complete", final
    current = verify_generation(tmp_path / "particles")
    assert sum(current["survivors"].values()) == 128
    assert current["shards"][0] == old["shards"][0]
    assert final["predecessor_job_id"] == handle["job_id"]
    assert (
        jobs.job_status(path, handle["job_id"])["charged_job_wall_seconds"]
        == first["charged_job_wall_seconds"]
    )
    assert (
        jobs.campaign_status(path)["total_job_wall_seconds"]
        >= first["charged_job_wall_seconds"] + final["charged_job_wall_seconds"]
    )
    allocation["max_batches"] = 3
    write_json(tmp_path / "fork.json", allocation)
    with pytest.raises(ValueError, match="already has a successor"):
        jobs.continue_generation_phase(path, handle["job_id"], inspection, tmp_path / "fork.json")
    jobs.inspect_recovery(path, handle["job_id"], tmp_path / "old_parent_inspection.json")
    with pytest.raises(ValueError, match="has a successor"):
        jobs.resume(
            path, handle["job_id"], tmp_path / "old_parent_inspection.json", "must not fork"
        )


def test_phase_refuses_new_scientific_plan_and_excess_original_budget(tmp_path):
    path, handle, first, inspection, allocation = generation_pilot(tmp_path)
    allocation["max_wall_seconds"] = 151
    write_json(tmp_path / "excess.json", allocation)
    with pytest.raises(ValueError, match="original cumulative"):
        jobs.continue_generation_phase(path, handle["job_id"], inspection, tmp_path / "excess.json")
    allocation.update(max_wall_seconds=80, seed=9)
    write_json(tmp_path / "changed.json", allocation)
    with pytest.raises(ValueError, match="execution allocation only"):
        jobs.continue_generation_phase(
            path, handle["job_id"], inspection, tmp_path / "changed.json"
        )
    plan = read_json(tmp_path / "plan.json")
    plan["seed"] += 1
    write_json(tmp_path / "plan.json", plan)
    with pytest.raises(ValueError, match="changed since"):
        jobs.continue_generation_phase(
            path, handle["job_id"], inspection, tmp_path / "continuation.json"
        )


def test_state_flags_include_starting_and_queued():
    for state in jobs.PENDING | jobs.TERMINAL:
        handle = jobs._handle(
            "fixture.json",
            {
                "id": "a" * 32,
                "status": state,
                "folder": "fixture",
                "result_path": "result.json",
                "request_sha256": "0" * 64,
            },
        )
        assert handle["is_pending"] == (state in jobs.PENDING)
        assert handle["is_terminal"] == (state in jobs.TERMINAL)


def test_real_zero_update_affine_control(tmp_path):
    pytest.importorskip("torch")
    from element_surrogate.benchmark.study import create_study, lock_selection, run_study
    from element_surrogate.data import generate
    from element_surrogate.modeling import prepare_generated_study
    from tests.test_physical_workflow import study_request
    from tests.workflow_fixture import approvals, qualified_plan

    transport, qualification, plan = qualified_plan()
    generate(
        transport,
        qualification,
        plan,
        tmp_path / "data",
        approvals(tmp_path / "gen_approvals", plan),
    )
    req = study_request(4)
    req["candidates"] = [req["candidates"][1]]
    req["candidates"][0].update(max_steps=0, seeds=[3], kind="deterministic")
    pp = prepare_generated_study(tmp_path / "data", tmp_path / "benchmark", req)["protocol_path"]
    protocol = read_json(pp)
    study = create_study(pp, tmp_path / "study", approvals(tmp_path / "approvals", protocol))[
        "study_path"
    ]
    run_study(study)
    selected = lock_selection(study)
    lock = read_json(Path(study).parent / "selection.json")
    assert lock["selected"]["best_step"] == 0
    assert selected
    trial = next((Path(study).parent / "trials").rglob("trial.json"))
    saved = read_json(trial)
    assert saved["steps"] == 0 and saved["history"] == [
        {"step": 0, "validation_primary_score": saved["metrics"]["primary_score"]}
    ]
    for value in (-1, True):
        invalid = deepcopy(protocol)
        invalid["candidates"][0]["max_steps"] = value
        from element_surrogate.benchmark.protocol import validate_protocol

        with pytest.raises(ValueError, match="integer >= 0"):
            validate_protocol(invalid)


def test_continuation_preserves_original_calendar_deadline(tmp_path):
    path, handle, first, inspection, allocation = generation_pilot(tmp_path, time_policy="calendar")
    successor = jobs.continue_generation_phase(
        path, handle["job_id"], inspection, tmp_path / "continuation.json"
    )
    last = wait(path, successor)
    assert last["deadline_unix"] == first["deadline_unix"]
    assert last["max_wall_seconds"] == allocation["max_wall_seconds"]
    assert last["status"] == "complete"


def test_domain_incomparable_dominating_and_unknown_visuals(tmp_path):
    from dataclasses import replace

    from element_surrogate.domain_report import build_domain_report
    from element_surrogate.region import Region
    from element_surrogate.transport import qualify_transport
    from tests.workflow_fixture import manufactured_transport

    transport, refinements, ranges, settings = manufactured_transport()
    paths = []
    for i, (w, h) in enumerate([(0.010, 0.014), (0.014, 0.010), (0.014, 0.014)]):
        candidate = replace(transport, trajectory_region=Region("rectangle", w, half_height_m=h))
        q = qualify_transport(
            candidate,
            ranges,
            settings,
            refinements,
            random_probes=0,
            seed=7,
            component_limits=[1e-8] * 4,
            path_limit_m=1e-9,
        )
        p = tmp_path / f"q{i}.json"
        write_json(p, q)
        paths.append(p)
    result = build_domain_report(paths, tmp_path / "dominating")
    assert result["recommendation"]["radius_m"] == 0.014
    assert result["recommendation"]["half_height_m"] == 0.014
    assert len(result["maximal_passing_candidates"]) == 1
    q = read_json(paths[-1])
    q["accepted"] = False
    for row in q["reports"]:
        row["changes"][-1]["max_component_change"] = None
    write_json(paths[-1], q)
    result = build_domain_report(paths, tmp_path / "unknown")
    assert result["recommendation"] is None
    svg = Path(result["plot_path"]).read_text(encoding="utf-8")
    assert "not computed" in svg and "w=10, h=14" in svg and "w=14, h=10" in svg
    assert "all gates pass" not in svg


@pytest.mark.parametrize("shape", ["circle", "square", "rectangle"])
def test_all_implemented_shapes_render_their_actual_dimensions(tmp_path, shape):
    from dataclasses import replace

    from element_surrogate.domain_report import build_domain_report
    from element_surrogate.region import Region
    from element_surrogate.transport import qualify_transport
    from tests.workflow_fixture import manufactured_transport

    t, r, bounds, settings = manufactured_transport()
    region = Region(shape, 0.014, half_height_m=0.012 if shape == "rectangle" else None)
    t = replace(t, trajectory_region=region)
    q = qualify_transport(
        t,
        bounds,
        settings,
        r,
        random_probes=0,
        seed=7,
        component_limits=[1e-8] * 4,
        path_limit_m=1e-9,
    )
    path = tmp_path / "q.json"
    write_json(path, q)
    result = build_domain_report([path], tmp_path / "report")
    svg = Path(result["plot_path"]).read_text(encoding="utf-8")
    expected = {"circle": "r=14", "square": "w=14, h=14", "rectangle": "w=14, h=12"}[shape]
    assert expected in svg
