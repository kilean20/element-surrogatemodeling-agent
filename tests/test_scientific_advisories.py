"""Scientific context advisories preserve results and real failure semantics."""

import asyncio
import copy

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from element_surrogate.advisories import readiness_warnings
from element_surrogate.benchmark.protocol import protocol_identity, read_json, write_json
from element_surrogate.field_readiness import assess, assess_from_files
from element_surrogate.mcp_server import build_server
from element_surrogate.runs import RunStore
from element_surrogate.uncertainty import review
from tests.test_uncertainty_progress import request
from tests.test_workflow_tools import file_case
from tests.workflow_fixture import manufactured_transport, readiness_for_test


def test_pending_notice_does_not_infer_a_missing_user_answer():
    warning = readiness_warnings("evidence.json")[0]
    assert warning["blocking"] is False
    assert "existing case decisions" in warning["recommended_action"]
    assert "current is provenance" in warning["recommended_action"]


def test_reviewed_limits_visible_without_changing_eligibility():
    transport, *_ = manufactured_transport()
    assessment = readiness_for_test(transport)
    before = protocol_identity(assessment)
    warnings = readiness_warnings("assessment.json", assessment)
    assert warnings and not warnings[0]["blocking"]
    assert "do not re-ask" in warnings[0]["recommended_action"]
    assert assessment["generation_eligible"] and protocol_identity(assessment) == before


def test_clean_completed_scope_has_no_unnecessary_advisory():
    assert (
        readiness_warnings(
            "assessment.json", {"generation_eligible": True, "review": {"issue_resolutions": {}}}
        )
        == []
    )


@pytest.mark.parametrize("status", ["unresolved", "rejected"])
def test_assessment_returns_warning_without_raising_or_granting_eligibility(tmp_path, status):
    transport_path, _, plan = file_case(tmp_path)
    original = plan["field_readiness"]
    revised = copy.deepcopy(original["review"])
    revised["categories"]["material_and_domain"]["status"] = status
    write_json(tmp_path / "evidence.json", original["evidence"])
    write_json(tmp_path / "review.json", revised)
    result = assess_from_files(
        transport_path,
        tmp_path / "evidence.json",
        tmp_path / "review.json",
        tmp_path / "assessment.json",
    )
    assert result["generation_eligible"] is False
    assert result["warnings"][0]["code"] == "READINESS_NOT_ELIGIBLE"
    assert not result["warnings"][0]["blocking"]
    saved = read_json(result["assessment_path"])
    assert saved == assess(original["evidence"], revised)


@pytest.mark.parametrize(
    "status, question_status", [("confirmed_unavailable", "answered"), ("asked_pending", "pending")]
)
def test_uncertainty_advisory_keeps_question_pacing_and_no_approval(
    tmp_path, status, question_status
):
    value = request(
        status,
        question={"kind": "fact", "status": question_status, "text": "Is geometry available?"},
    )
    write_json(tmp_path / "request.json", value)
    result = review(tmp_path / "request.json", tmp_path / "review.json")
    warning = result["warnings"][0]
    assert not warning["blocking"] and "Offer this" not in warning["recommended_action"]
    assert not result["approval_created"] and not result["generation_eligibility_changed"]
    assert read_json(result["review_path"])["warnings"] == result["warnings"]


def test_real_mcp_collects_evidence_with_advisory_but_invalid_arguments_remain_errors(tmp_path):
    transport, *_ = file_case(tmp_path)
    server = build_server(RunStore(tmp_path / "client"))
    arguments = {"path": str(transport), "output_path": str(tmp_path / "raw.json"), "shells": 1}
    result = asyncio.run(server.call_tool("collect_field_readiness", arguments))
    assert not result.is_error
    assert result.structured_content["warnings"][0]["code"] == "READINESS_INTERPRETATION_PENDING"
    assert read_json(tmp_path / "raw.json")["diagnostics"]["maxwell"]
    arguments.update(shells=0, output_path=str(tmp_path / "invalid.json"))
    with pytest.raises(ToolError, match="positive explicit Maxwell shell count"):
        asyncio.run(server.call_tool("collect_field_readiness", arguments))
    assert not (tmp_path / "invalid.json").exists()


def test_malformed_uncertainty_input_is_not_disguised_as_a_warning(tmp_path):
    write_json(tmp_path / "invalid.json", {"schema": "invalid"})
    with pytest.raises(ValueError, match="uncertainty_review"):
        review(tmp_path / "invalid.json", tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()
