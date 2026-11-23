"""File interfaces, background generation, exact reference/frame and private decisions."""

import time
from dataclasses import replace
from pathlib import Path

import pytest

from element_surrogate.benchmark.protocol import protocol_identity, read_json, write_json
from element_surrogate.decisions import list_decisions, record_decision
from element_surrogate.domain_report import build_domain_report
from element_surrogate.field_store import save_mapped_field
from element_surrogate.generation_job import generation_status, start_generation
from element_surrogate.region import Region
from element_surrogate.tools import REGISTRY, trace_reference_orbit
from element_surrogate.transport import frame_dict, qualify_transport
from element_surrogate.workflow import load_transport, qualify_from_files
from tests.workflow_fixture import approvals, manufactured_transport, readiness_for_test


def file_case(directory):
    transport, levels, ranges, settings = manufactured_transport()
    stored = save_mapped_field(transport.grid, directory / "map", block_nodes=10)
    reference_path = directory / "reference.npz"
    reference = trace_reference_orbit(
        stored["path"],
        str(reference_path),
        2.0,
        1.0,
        frame_dict(transport.entry),
        frame_dict(transport.exit),
        levels[-1],
    )
    spec = {
        "schema": "element_surrogate.transport.v1",
        "native_planes": True,
        "field": {"path": stored["path"], "sha256": stored["sha256"]},
        "reference": {
            "path": str(reference_path),
            "sha256": reference["saved_reference"]["sha256"],
            "rigidity_tm": 2.0,
            "field_scale": 1.0,
        },
        "boundary_convention": "declared_native",
        "entry_frame": frame_dict(transport.entry),
        "exit_frame": frame_dict(transport.exit),
        "codec": {"dimension": 4, "longitudinal": "path_m"},
        "support": {"shape": "circle", "radius_m": 0.04},
        "trajectory_region": {"shape": "circle", "radius_m": 0.02},
        "integration": levels[-1],
    }
    path = directory / "transport.json"
    write_json(path, spec)
    request = {
        "ranges": ranges,
        "settings": settings,
        "refinements": levels,
        "random_probes": 0,
        "seed": 17,
        "component_limits": [1e-8] * 4,
        "path_limit_m": 1e-9,
    }
    write_json(directory / "qualification_request.json", request)
    qualify_from_files(
        path, directory / "qualification_request.json", directory / "qualification.json"
    )
    model, _ = load_transport(path)
    qualification = read_json(directory / "qualification.json")
    assessment = readiness_for_test(model)
    plan = {
        "schema": "element_surrogate.generation.v2",
        "transport": model.description(),
        "qualification_sha256": protocol_identity(qualification),
        "field_readiness": assessment,
        "field_readiness_sha256": protocol_identity(assessment),
        "ranges": ranges,
        "settings": settings,
        "seed": 102,
        "batch_size": 17,
        "max_proposals_per_setting": 100,
        "max_wall_seconds": 60.0,
        "material_boundary_evidence": "synthetic vacuum restriction; no mechanical claim",
    }
    write_json(directory / "plan.json", plan)
    return path, spec, plan


def test_fixed_reference_file_roundtrip_and_mismatched_native_plane_rejected(tmp_path):
    path, spec, _ = file_case(tmp_path)
    transport, _ = load_transport(path)
    assert transport.reference_path_m > 0
    spec["exit_frame"]["origin_m"][2] += 0.01
    write_json(path, spec)
    with pytest.raises(ValueError, match="planes differ"):
        load_transport(path)


def test_background_generation_and_duplicate_launch_refusal(tmp_path):
    path, _, plan = file_case(tmp_path)
    decisions = approvals(tmp_path / "decisions", plan)
    started = start_generation(
        path, tmp_path / "qualification.json", tmp_path / "plan.json", tmp_path / "data", decisions
    )
    assert started["status"] == "scheduled"
    with pytest.raises(ValueError, match="already scheduled or running"):
        start_generation(
            path,
            tmp_path / "qualification.json",
            tmp_path / "plan.json",
            tmp_path / "data",
            decisions,
        )
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        status = generation_status(tmp_path / "data")
        if status["status"] not in ("scheduled", "running"):
            break
        time.sleep(0.05)
    assert status["status"] == "complete", status
    assert status["generation"]["surviving_rows"] == 128


def test_private_decisions_do_not_become_approval_files(tmp_path):
    statement = "Use the smaller numerical region; the aperture is unknown."
    event = record_decision(
        tmp_path,
        actor="synthetic-user",
        user_statement=statement,
        scope="region",
        source_reference="synthetic test transcript",
    )
    saved = read_json(event["decision_path"])
    assert saved["user_statement"] == statement and saved["not_an_approval_record"]
    assert list_decisions(tmp_path)["events"] == [saved]
    assert saved["schema"] != "element_surrogate.approval.v1"


def test_size_report_uses_the_same_criteria_and_does_not_claim_aperture(tmp_path):
    transport, levels, ranges, settings = manufactured_transport()
    paths = []
    for i, radius in enumerate((0.001, 0.01, 0.02)):
        candidate = replace(transport, trajectory_region=Region("circle", radius))
        report = qualify_transport(
            candidate,
            ranges,
            settings,
            levels,
            random_probes=0,
            seed=1,
            component_limits=[1e-8] * 4,
            path_limit_m=1e-9,
        )
        path = tmp_path / f"qualification_{i}.json"
        write_json(path, report)
        paths.append(str(path))
    result = build_domain_report(paths, tmp_path / "report")
    assert result["recommendation"]["radius_m"] == 0.02
    assert not result["selection_objective_needed"]
    assert "already supplied decision or delegation" in result["authority"]
    import xml.etree.ElementTree as ET

    ET.parse(result["plot_path"])
    text = Path(result["report_path"]).read_text()
    assert "no mechanical aperture" in text


def test_new_tools_have_single_registry_definitions_and_correct_mutation_annotations():
    for name in (
        "trace_reference_orbit",
        "qualify_element_transport",
        "start_training_data_generation",
        "prepare_neural_surrogate_study",
        "export_native_surrogate",
        "record_user_decision",
    ):
        assert name in REGISTRY and not REGISTRY[name].read_only
    for name in ("validate_training_data_plan", "training_data_status", "read_user_decisions"):
        assert name in REGISTRY and REGISTRY[name].read_only


def test_generation_supervisor_enforces_a_hard_budget(tmp_path):
    path, _, plan = file_case(tmp_path)
    plan["max_wall_seconds"] = 0.05
    write_json(tmp_path / "plan.json", plan)
    decisions = approvals(tmp_path / "decisions", plan)
    start_generation(
        path, tmp_path / "qualification.json", tmp_path / "plan.json", tmp_path / "data", decisions
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        status = generation_status(tmp_path / "data")
        if status["status"] not in ("scheduled", "running"):
            break
        time.sleep(0.05)
    assert status["status"] == "timed_out", status
