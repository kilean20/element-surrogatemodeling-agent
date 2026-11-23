"""Actionable scientific advisories; never approvals or numerical overrides."""

from __future__ import annotations


def _warning(code, message, action, evidence_ref):
    return {
        "code": code,
        "message": message,
        "recommended_action": action,
        "evidence_ref": str(evidence_ref),
        "blocking": False,
    }


def readiness_warnings(evidence_ref, assessment=None):
    """Expose interpretation work without inferring missing user facts."""
    if assessment is None:
        return [
            _warning(
                "READINESS_INTERPRETATION_PENDING",
                "Raw diagnostics are available; scientific interpretation is still needed.",
                "Inspect the evidence and existing case decisions. Ask only useful unanswered "
                "questions; unavailable physical facts can support a qualified numerical scope. "
                "Continue independent authorized analysis and record the scientific review. "
                "Calculation current is provenance, not a required map input or calibration task.",
                evidence_ref,
            )
        ]
    if not assessment["generation_eligible"]:
        return [
            _warning(
                "READINESS_NOT_ELIGIBLE",
                "The assessment completed; generation eligibility is false.",
                "Inspect unresolved/rejected dispositions and diagnostic errors in the assessment. "
                "The agent can review unknown facts as retained numerical limitations when "
                "defensible; a new user answer is not always required. Continue independent "
                "analysis. Actual computation/source/support failures still need correction; "
                "this advisory does not change the generation decision.",
                evidence_ref,
            )
        ]
    retained = [
        key
        for key, row in assessment["review"].get("issue_resolutions", {}).items()
        if row["status"] == "accepted_numerical_limitation"
    ]
    if retained:
        return [
            _warning(
                "NUMERICAL_LIMITATIONS_RETAINED",
                f"The reviewed numerical scope retains {len(retained)} measured limitations.",
                "Continue within the assessed scope and report its limitations. Reuse recorded "
                "answers/delegation; do not re-ask unavailable facts or imply "
                "physical certification.",
                evidence_ref,
            )
        ]
    return []


def uncertainty_warning(item, question_guidance, evidence_ref):
    """Make the caller's unresolved/partial scope visible, without inventing a question."""
    if (
        item["information_status"] in {"supplied", "derived", "delegated"}
        and not item["unavailable_outputs"]
    ):
        return None
    return _warning(
        "SCIENTIFIC_UNCERTAINTY",
        f"{item['topic']}: {item['information_status']}.",
        question_guidance + " Follow the recorded plan and communicate its retained limits.",
        evidence_ref,
    )
