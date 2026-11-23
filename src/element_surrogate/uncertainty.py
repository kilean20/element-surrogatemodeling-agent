"""Record reasoned uncertainty decisions without adding a scientific gate.

The client supplies the analysis. This helper does not infer physical truth,
authenticate a user, approve work or override numerical qualification.
"""

from __future__ import annotations

from pathlib import Path

from element_surrogate.advisories import uncertainty_warning
from element_surrogate.benchmark.protocol import file_hash, read_json, write_json

SCHEMA = "element_surrogate.uncertainty_review.v1"
STATES = {
    "supplied",
    "derived",
    "not_supplied",
    "asked_pending",
    "confirmed_unavailable",
    "delegated",
    "conflicting",
}
ACTIONS = {
    "continue_scoped",
    "continue_partial",
    "ask_and_continue",
    "analyze_alternatives",
    "needs_definition",
}


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"nonempty {name} required")


def _texts(value, name):
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of text summaries")
    for item in value:
        _text(item, name)


def _question_guidance(item):
    question = item.get("question")
    if question is None:
        return "No question recorded; follow the reasoned scope decision."
    if not isinstance(question, dict) or set(question) != {"kind", "status", "text"}:
        raise ValueError("question needs kind, status and text, or null")
    _text(question["text"], "question text")
    if question["kind"] not in {"fact", "scope_decision"}:
        raise ValueError("question kind must be fact or scope_decision")
    if question["status"] not in {"unasked", "pending", "answered", "declined"}:
        raise ValueError("explicit question status required")
    if question["status"] in {"answered", "declined"}:
        return "Do not repeat an answered/declined question; use the recorded analysis."
    if question["status"] == "pending":
        return "Keep the question pending without repetition; continue independent scoped work."
    state = item["information_status"]
    if question["kind"] == "fact" and state in {
        "supplied",
        "derived",
        "confirmed_unavailable",
        "delegated",
    }:
        return "Do not ask the same known/unavailable fact again; analyze the feasible scope."
    if question["kind"] == "scope_decision" and state == "delegated":
        return "Reuse existing delegation within its recorded scope; ask only for new authority."
    return "Offer this useful question once; do not stop work independent of its answer."


def review(request_path, output_path):
    """Write an immutable advisory analysis, preserving claims and alternatives."""
    request = read_json(request_path)
    if not isinstance(request, dict) or request.get("schema") != SCHEMA:
        raise ValueError("uncertainty_review.v1 request required")
    _text(request.get("purpose"), "purpose")
    items = request.get("items")
    if not isinstance(items, list):
        raise ValueError("items must be a list; an empty review is permitted")
    summaries, identifiers, warnings = [], set(), []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each uncertainty needs an analysis record")
        for name in ("id", "topic", "analysis", "plan", "authority_reference"):
            _text(item.get(name), name)
        if item["id"] in identifiers:
            raise ValueError("unique uncertainty IDs required within this review")
        identifiers.add(item["id"])
        if item.get("information_status") not in STATES:
            raise ValueError("explicit information_status required")
        if item.get("action") not in ACTIONS:
            raise ValueError("explicit scoped action required")
        for name in (
            "evidence",
            "alternatives",
            "continue_work",
            "unavailable_outputs",
            "limitations",
        ):
            _texts(item.get(name), name)
        guidance = _question_guidance(item)
        warning = uncertainty_warning(item, guidance, Path(request_path).resolve())
        if warning is not None:
            warnings.append(warning)
        summaries.append(
            {
                "id": item["id"],
                "information_status": item["information_status"],
                "action": item["action"],
                "continue_work": item["continue_work"],
                "unavailable_outputs": item["unavailable_outputs"],
                "limitations": item["limitations"],
                "question_guidance": guidance,
            }
        )
    report = {
        "schema": SCHEMA,
        "request_path": str(Path(request_path).resolve()),
        "request_sha256": file_hash(request_path),
        "review": request,
        "next_actions": summaries,
        "warnings": warnings,
        "scope": "Client scientific judgment, not an automatic correctness assessment.",
        "generation_eligibility_changed": False,
        "approval_created": False,
        "prerequisite_for_existing_workflows": False,
    }
    write_json(output_path, report, exclusive=True)
    return {
        "review_path": str(Path(output_path).resolve()),
        "review_sha256": file_hash(output_path),
        "next_actions": summaries,
        "warnings": warnings,
        "generation_eligibility_changed": False,
        "approval_created": False,
        "prerequisite_for_existing_workflows": False,
        "scope": report["scope"],
    }
