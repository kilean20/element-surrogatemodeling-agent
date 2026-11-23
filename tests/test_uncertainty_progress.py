"""Unknown facts allow scoped progress; analysis notes cannot waive physics checks."""

import asyncio
import copy
from dataclasses import replace

import pytest

from element_surrogate.benchmark.protocol import protocol_identity, read_json, write_json
from element_surrogate.contracts import StaticMagneticCodec
from element_surrogate.data import SCHEMA, generate
from element_surrogate.field_readiness import assess
from element_surrogate.mcp_server import build_server
from element_surrogate.runs import RunStore
from element_surrogate.transport import qualify_transport
from element_surrogate.uncertainty import SCHEMA as REVIEW_SCHEMA
from element_surrogate.uncertainty import review
from tests.workflow_fixture import approvals, manufactured_transport, readiness_for_test


def request(status="confirmed_unavailable", *, question=None):
    return {
        "schema": REVIEW_SCHEMA,
        "purpose": "Manufactured research scope; not a user approval",
        "items": [
            {
                "id": "geometry",
                "topic": "Mechanical geometry not available",
                "information_status": status,
                "analysis": "Use actual field/stencil/path evidence; no physical clearance claim.",
                "evidence": ["Synthetic field and supplied task scope, not physical bore data"],
                "alternatives": ["Qualified numerical map", "Physical assessment if facts appear"],
                "action": "continue_scoped",
                "plan": "Continue field-specific numerical qualification and research generation.",
                "authority_reference": "Manufactured task delegation; not real-user authority",
                "continue_work": ["numerical research"],
                "unavailable_outputs": ["physical aperture certification"],
                "limitations": ["Numerical survivors do not establish mechanical clearance."],
                "question": question,
            }
        ],
    }


def call_review(tmp_path, value):
    path = tmp_path / "request.json"
    write_json(path, value)
    return review(path, tmp_path / "review.json")


@pytest.mark.parametrize("status", ["confirmed_unavailable", "supplied", "derived", "delegated"])
def test_answered_fact_does_not_trigger_repeated_question(tmp_path, status):
    q = {"kind": "fact", "status": "unasked", "text": "Can you supply the geometry?"}
    result = call_review(tmp_path, request(status, question=q))
    assert "Do not ask the same" in result["next_actions"][0]["question_guidance"]
    assert result["next_actions"][0]["continue_work"] == ["numerical research"]
    assert not result["generation_eligibility_changed"] and not result["approval_created"]
    assert not result["prerequisite_for_existing_workflows"]


@pytest.mark.parametrize("status", ["not_supplied", "conflicting"])
def test_important_question_offered_once_without_blanket_hold(tmp_path, status):
    q = {"kind": "fact", "status": "unasked", "text": "Is this radius or diameter?"}
    result = call_review(tmp_path, request(status, question=q))
    assert "Offer this useful question once" in result["next_actions"][0]["question_guidance"]
    assert result["next_actions"][0]["unavailable_outputs"] == ["physical aperture certification"]


@pytest.mark.parametrize("state", ["pending", "answered", "declined"])
def test_pending_or_finished_questions_not_reissued(tmp_path, state):
    q = {"kind": "fact", "status": state, "text": "Can you supply a source record?"}
    result = call_review(tmp_path, request("asked_pending", question=q))
    assert "Offer this" not in result["next_actions"][0]["question_guidance"]


def test_unavailable_fact_can_lead_to_a_genuinely_new_scope_decision(tmp_path):
    q = {"kind": "scope_decision", "status": "unasked", "text": "Accept numerical-only use?"}
    assert (
        "Offer this"
        in call_review(tmp_path, request(question=q))["next_actions"][0]["question_guidance"]
    )


def test_existing_delegation_does_not_require_new_scope_approval(tmp_path):
    q = {"kind": "scope_decision", "status": "unasked", "text": "Choose a numerical domain?"}
    result = call_review(tmp_path, request("delegated", question=q))
    assert "Reuse existing delegation" in result["next_actions"][0]["question_guidance"]


def test_immutable_review_retains_reasoning_and_empty_case_is_valid(tmp_path):
    value = request()
    result = call_review(tmp_path, value)
    assert read_json(result["review_path"])["review"] == value
    with pytest.raises((ValueError, FileExistsError)):
        review(tmp_path / "request.json", tmp_path / "review.json")
    empty = tmp_path / "empty.json"
    write_json(empty, {"schema": REVIEW_SCHEMA, "purpose": "No unresolved decisions", "items": []})
    assert review(empty, tmp_path / "empty_review.json")["next_actions"] == []


def test_real_mcp_review_is_logged_but_creates_no_approval(tmp_path):
    path = tmp_path / "request.json"
    write_json(path, request())
    store = RunStore(tmp_path / "project")
    server = build_server(store)
    result = asyncio.run(
        server.call_tool(
            "write_scientific_uncertainty_review",
            {"request_path": str(path), "output_path": str(tmp_path / "review.json")},
        )
    )
    assert not result.is_error
    data = result.structured_content
    assert not data["approval_created"]
    assert len(store.list_runs()) == 1
    assert store.load(data["run_id"])["tool"] == "write_scientific_uncertainty_review"
    assert not list((tmp_path / "project").rglob("*approval*"))


def numerical_case():
    transport, levels, ranges, settings = manufactured_transport()
    values = transport.grid.b_t.copy()
    # Deliberate small raw curl inconsistency: fit the supplied numerical field,
    # without claiming it is an exact current-free vacuum solution.
    values[..., 1] += 0.01 * transport.grid.x_m[:, None, None]
    transport = replace(
        transport, grid=transport.grid.with_values(values), actual_p0_kg_m_s=None, mass_kg=None
    )
    qualification = qualify_transport(
        transport,
        ranges,
        settings,
        levels,
        random_probes=0,
        seed=17,
        component_limits=[1e-8] * 4,
        path_limit_m=1e-9,
    )
    assert qualification["accepted"]
    initial = readiness_for_test(transport)
    interpretation = copy.deepcopy(initial["review"])
    for item in interpretation["categories"].values():
        item["rationale"] = (
            "No source-solver, current calibration, material mask or mechanical geometry supplied. "
            "Measured raw inconsistency retained; only scoped numerical-field research qualified."
        )
    for item in interpretation["issue_resolutions"].values():
        item["rationale"] = (
            "Retain raw numerical limitations/invariance observations. This is not proof of an "
            "exact physical vacuum field, physical aperture or calibrated source current."
        )
    assessment = assess(initial["evidence"], interpretation)
    plan = {
        "schema": SCHEMA,
        "transport": transport.description(),
        "qualification_sha256": protocol_identity(qualification),
        "field_readiness": assessment,
        "field_readiness_sha256": protocol_identity(assessment),
        "ranges": ranges,
        "settings": settings,
        "seed": 103,
        "batch_size": 17,
        "max_proposals_per_setting": 100,
        "max_wall_seconds": 60,
        "material_boundary_evidence": "Unknown geometry; declared numerical path region only.",
    }
    return transport, qualification, plan


def test_unknown_geometry_source_quality_and_mass_allow_actual_4d_generation(tmp_path):
    transport, qualification, plan = numerical_case()
    assert plan["field_readiness"]["generation_eligible"]
    assert plan["field_readiness"]["evidence"]["diagnostics"]["maxwell"]
    before = protocol_identity(plan)
    call_review(tmp_path, request())
    assert protocol_identity(plan) == before
    result = generate(transport, qualification, plan, tmp_path / "data", approvals(tmp_path, plan))
    assert result["status"] == "complete" and result["surviving_rows"] == 128
    assert transport.actual_p0_kg_m_s is None and transport.mass_kg is None


def test_advisory_note_cannot_override_existing_numerical_or_authority_guards(tmp_path):
    transport, qualification, plan = numerical_case()
    note = call_review(tmp_path, request())
    with pytest.raises(ValueError):
        generate(transport, qualification, plan, tmp_path / "not_approved", [note["review_path"]])
    rejected = copy.deepcopy(plan)
    interpretation = copy.deepcopy(rejected["field_readiness"]["review"])
    interpretation["categories"]["grid_integrity"]["status"] = "rejected"
    rejected["field_readiness"] = assess(rejected["field_readiness"]["evidence"], interpretation)
    rejected["field_readiness_sha256"] = protocol_identity(rejected["field_readiness"])
    with pytest.raises(ValueError, match="unresolved or rejected"):
        generate(
            transport, qualification, rejected, tmp_path / "rejected", approvals(tmp_path, rejected)
        )
    assert not (tmp_path / "rejected").exists()


def test_partial_progress_does_not_fabricate_time_or_disable_species_check(tmp_path):
    value = request()
    item = value["items"][0]
    item.update(
        topic="Actual momentum and mass unavailable",
        action="continue_partial",
        continue_work=["field and transverse trajectory analysis"],
        unavailable_outputs=["physical arrival time"],
        plan="Explain path/transverse alternatives; do not silently substitute time.",
    )
    result = call_review(tmp_path, value)
    assert result["next_actions"][0]["unavailable_outputs"] == ["physical arrival time"]
    transport, *_ = manufactured_transport()
    with pytest.raises(ValueError, match="actual positive momentum and rest mass"):
        replace(
            transport, codec=StaticMagneticCodec(6, "time_s"), actual_p0_kg_m_s=None, mass_kg=None
        )
