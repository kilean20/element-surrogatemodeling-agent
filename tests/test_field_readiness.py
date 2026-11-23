"""Meaningful readiness refusal, scope, diagnostic and recovery controls."""

import copy
from dataclasses import replace

import pytest

from element_surrogate.benchmark.protocol import file_hash, protocol_identity
from element_surrogate.data import generate
from element_surrogate.field_readiness import (
    CATEGORIES,
    REVIEW_SCHEMA,
    assess,
    collect_evidence,
    validate_assessment,
)
from element_surrogate.region import Region
from tests.workflow_fixture import (
    approvals,
    manufactured_transport,
    qualified_plan,
    readiness_for_test,
)


def test_integration_qualification_alone_cannot_generate(tmp_path):
    transport, qualification, plan = qualified_plan()
    del plan["field_readiness"]
    with pytest.raises(ValueError, match="field readiness"):
        generate(transport, qualification, plan, tmp_path / "data", approvals(tmp_path, plan))
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("status", ["unresolved", "rejected"])
def test_unresolved_scientific_category_blocks_proposals(status, tmp_path):
    transport, qualification, plan = qualified_plan()
    original = plan["field_readiness"]
    review = copy.deepcopy(original["review"])
    review["categories"]["source_and_description"]["status"] = status
    plan["field_readiness"] = assess(original["evidence"], review)
    plan["field_readiness_sha256"] = protocol_identity(plan["field_readiness"])
    assert not plan["field_readiness"]["generation_eligible"]
    with pytest.raises(ValueError, match="unresolved or rejected"):
        generate(transport, qualification, plan, tmp_path / "data", approvals(tmp_path, plan))
    assert not (tmp_path / "data").exists()


def test_no_missing_category_or_finding_can_be_silently_ignored():
    transport, *_ = manufactured_transport()
    original = readiness_for_test(transport)
    assert original["evidence"]["issues"]  # intended invariant planes/zero-axis warnings
    review = copy.deepcopy(original["review"])
    review["issue_resolutions"].pop(next(iter(review["issue_resolutions"])))
    with pytest.raises(ValueError, match="every readiness finding"):
        assess(original["evidence"], review)
    review = copy.deepcopy(original["review"])
    del review["categories"][CATEGORIES[0]]
    with pytest.raises(ValueError, match="all readiness categories"):
        assess(original["evidence"], review)


def test_evidence_mutation_unsupported_claim_and_stale_implementation_refused():
    transport, *_ = manufactured_transport()
    original = readiness_for_test(transport)
    changed = copy.deepcopy(original)
    changed["evidence"]["diagnostics"]["maxwell"]["reading"] = "fabricated pass"
    with pytest.raises(ValueError, match="identity"):
        validate_assessment(transport, changed)
    changed = copy.deepcopy(original)
    changed["evidence"]["implementation_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="implementation"):
        validate_assessment(transport, changed)
    changed = copy.deepcopy(original)
    changed["review"]["categories"][CATEGORIES[0]]["evidence_refs"] = ["absent CAD certificate"]
    with pytest.raises(ValueError, match="available evidence"):
        assess(original["evidence"], changed["review"])


def test_values_reconstruction_and_support_changes_invalidate_assessment():
    transport, *_ = manufactured_transport()
    assessment = readiness_for_test(transport)
    values = transport.grid.b_t.copy()
    values[5, 5, 5, 0] += 0.2
    for changed in (
        replace(transport, grid=transport.grid.with_values(values)),
        replace(transport, reconstruction="trilinear"),
        replace(transport, support=Region("circle", 0.035)),
        replace(transport, trajectory_region=Region("circle", 0.018)),
    ):
        with pytest.raises(ValueError, match="scope changed"):
            validate_assessment(changed, assessment)


def test_real_defect_diagnostics_remain_visible_and_cannot_be_unreviewed():
    transport, *_ = manufactured_transport()
    values = transport.grid.b_t.copy()
    values[5, 5, 10, 0] += 1.0
    damaged = replace(transport, grid=transport.grid.with_values(values))
    evidence = collect_evidence(damaged, shells=2)
    assert evidence["diagnostics"]["harmonic_nodes"]["findings"]
    assert evidence["issues"]
    review = {
        "schema": REVIEW_SCHEMA,
        "evidence_sha256": protocol_identity(evidence),
        "categories": {
            k: {
                "status": "reviewed",
                "rationale": "Synthetic test only",
                "evidence_refs": ["maxwell"],
            }
            for k in CATEGORIES
        },
        "issue_resolutions": {},
    }
    with pytest.raises(ValueError, match="every readiness finding"):
        assess(evidence, review)


def test_benign_findings_require_reason_but_do_not_force_a_physical_false_alarm():
    transport, *_ = manufactured_transport()
    assessment = readiness_for_test(transport)
    validate_assessment(transport, assessment)
    assert assessment["generation_eligible"]
    assert "not a physical-validity" in assessment["scope"]


def test_external_provenance_changes_invalidate_review(tmp_path):
    transport, *_ = manufactured_transport()
    assessment = readiness_for_test(transport)
    source_note = tmp_path / "source_description.md"
    source_note.write_text("Synthetic source description, not physical geometry")
    review = copy.deepcopy(assessment["review"])
    review["external_evidence"] = [
        {"id": "source_note", "path": str(source_note), "sha256": file_hash(source_note)}
    ]
    review["categories"]["source_and_description"]["evidence_refs"] = ["source_note"]
    assessment = assess(assessment["evidence"], review)
    validate_assessment(transport, assessment)
    source_note.write_text("changed excitation or geometry")
    with pytest.raises(ValueError, match="external readiness evidence changed"):
        validate_assessment(transport, assessment)


def test_readiness_through_mcp_with_file_generation_refusal(tmp_path):
    from element_surrogate.benchmark.protocol import read_json, write_json
    from element_surrogate.mcp_server import build_server
    from element_surrogate.runs import RunStore
    from tests.test_mcp_server import _call
    from tests.test_workflow_tools import file_case

    path, _, plan = file_case(tmp_path)
    server = build_server(RunStore(tmp_path / "project"))
    evidence_path = tmp_path / "readiness.json"
    result = _call(
        server,
        "collect_field_readiness",
        {
            "path": str(path),
            "output_path": str(evidence_path),
            "shells": 2,
        },
    )
    evidence = read_json(evidence_path)
    assert result["evidence_sha256"] == protocol_identity(evidence)
    review = copy.deepcopy(plan["field_readiness"]["review"])
    review["evidence_sha256"] = protocol_identity(evidence)
    review_path, assessment_path = tmp_path / "review.json", tmp_path / "assessment.json"
    write_json(review_path, review)
    assessed = _call(
        server,
        "assess_field_readiness",
        {
            "path": str(path),
            "evidence_path": str(evidence_path),
            "review_path": str(review_path),
            "output_path": str(assessment_path),
        },
    )
    assert assessed["generation_eligible"]
    plan["field_readiness"] = read_json(assessment_path)
    plan["field_readiness_sha256"] = assessed["assessment_sha256"]
    write_json(tmp_path / "plan.json", plan)
    validated = _call(
        server,
        "validate_training_data_plan",
        {
            "path": str(path),
            "qualification_path": str(tmp_path / "qualification.json"),
            "plan_path": str(tmp_path / "plan.json"),
        },
    )
    assert validated["valid"]
