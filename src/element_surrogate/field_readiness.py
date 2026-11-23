"""Bound field diagnostics and explicit scientific interpretation to generation.

Completeness and identity guards for trusted local research, not authentication,
a physical-validity certificate or a substitute for the user's decisions.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np

from element_surrogate.benchmark.protocol import file_hash, protocol_identity, read_json, write_json
from element_surrogate.diagnostic_blocks import DEFAULT_WORKING_BYTES, validate_budget
from element_surrogate.readiness import field_ends, harmonic_nodes, maxwell_report, node_defects
from element_surrogate.runs import implementation_identity, sha256_json

CATEGORIES = (
    "source_and_description",
    "units_frame_polarity",
    "grid_integrity",
    "vacuum_applicability_and_consistency",
    "field_coverage",
    "material_and_domain",
)
EVIDENCE_SCHEMA = "element_surrogate.field_readiness_evidence.v1"
REVIEW_SCHEMA = "element_surrogate.field_readiness_review.v1"
SCHEMA = "element_surrogate.field_readiness_assessment.v1"


def _safe(value):
    """Retain undefined/unbounded statistics explicitly in valid JSON."""
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"undefined_numeric": str(value)}
    return value


def readiness_scope(transport):
    description = transport.description()
    return {
        "field": description["field"],
        "support": description["support"],
        "trajectory_region": description["trajectory_region"],
    }


def _issues(value, path=""):
    result = []
    if isinstance(value, dict):
        if "diagnostic_error" in value or "undefined_numeric" in value:
            result.append({"path": path, "finding": value})
        for key, item in value.items():
            if key in ("findings", "observations"):
                for index, finding in enumerate(item):
                    result.append(
                        {
                            "path": f"{path}.{key}.{index}",
                            "finding": finding,
                            "issue_type": "observation" if key == "observations" else "finding",
                        }
                    )
            else:
                result.extend(_issues(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.extend(_issues(item, f"{path}.{index}"))
    return result


def collect_evidence(
    transport,
    *,
    shells,
    end_tolerance=None,
    working_memory_bytes=DEFAULT_WORKING_BYTES,
    scratch_directory=None,
):
    """Compute diagnostics on raw values in the declared reconstruction support.

    Interpretation of units, source trust, geometry and vacuum applicability is
    deliberately not inferred from numerical convergence or a small residual.
    """
    if type(shells) is not int or shells < 1:
        raise ValueError("positive explicit Maxwell shell count required")
    if end_tolerance is not None and (not math.isfinite(end_tolerance) or end_tolerance <= 0):
        raise ValueError("positive declared field-end tolerance required")
    validate_budget(working_memory_bytes)
    options = {"working_memory_bytes": working_memory_bytes, "scratch_directory": scratch_directory}
    from element_surrogate.transport import verify_transport_source

    verify_transport_source(transport)
    scope = readiness_scope(transport)
    reports = {}
    for name, call in (
        ("grid_integrity", lambda: node_defects(transport.grid, **options)),
        ("maxwell", lambda: maxwell_report(transport.grid, transport.support, shells, **options)),
        ("harmonic_nodes", lambda: harmonic_nodes(transport.grid, transport.support, **options)),
        (
            "field_ends",
            lambda: field_ends(transport.grid, transport.support, end_tolerance, **options),
        ),
    ):
        try:
            reports[name] = _safe(call())
        except (
            ValueError,
            FloatingPointError,
            ZeroDivisionError,
            MemoryError,
            OSError,
            sqlite3.Error,
        ) as exc:
            reports[name] = {"diagnostic_error": f"{type(exc).__name__}: {exc}"}
    verify_transport_source(transport)
    if readiness_scope(transport) != scope:
        raise ValueError("field changed during readiness collection")
    issues = {protocol_identity(issue): issue for issue in _issues(reports)}
    return {
        "schema": EVIDENCE_SCHEMA,
        "scope": scope,
        "implementation_sha256": sha256_json(implementation_identity()),
        "source_summary": {
            "grid": transport.grid.grid_summary(),
            "source": None if transport.grid.source is None else transport.grid.source.as_dict(),
        },
        "diagnostics": reports,
        "issues": issues,
        "review_categories": list(CATEGORIES),
        "limits": (
            "Raw diagnostics, not physical certification. "
            "Undefined statistics require explicit treatment. Computation/resource "
            "errors block generation until a successful evidence collection replaces them."
        ),
    }


def _reason(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("explicit scientific rationale required")


def _eligibility(evidence, review):
    if evidence.get("schema") != EVIDENCE_SCHEMA or review.get("schema") != REVIEW_SCHEMA:
        raise ValueError("field readiness evidence/review schema required")
    if review.get("evidence_sha256") != protocol_identity(evidence):
        raise ValueError("readiness evidence identity differs from review")
    actual_issues = {protocol_identity(i): i for i in _issues(evidence["diagnostics"])}
    if evidence.get("issues") != actual_issues:
        raise ValueError("readiness finding inventory changed")
    categories = review.get("categories", {})
    if set(categories) != set(CATEGORIES):
        raise ValueError("all readiness categories require an explicit disposition")
    eligible = True
    references = {"source_summary", *evidence["diagnostics"]}
    for item in review.get("external_evidence", []):
        if not item.get("id") or item["id"] in references:
            raise ValueError("unique external readiness evidence ID required")
        if not Path(item["path"]).is_absolute():
            raise ValueError("absolute external readiness evidence path required")
        if file_hash(item["path"]) != item["sha256"]:
            raise ValueError("external readiness evidence changed")
        references.add(item["id"])
    for row in categories.values():
        if row.get("status") not in {"reviewed", "not_applicable", "unresolved", "rejected"}:
            raise ValueError("explicit readiness category status required")
        _reason(row.get("rationale"))
        refs = row.get("evidence_refs", [])
        if not refs or not set(refs) <= references:
            raise ValueError("readiness interpretation must cite available evidence")
        eligible &= row["status"] in {"reviewed", "not_applicable"}
    resolutions = review.get("issue_resolutions", {})
    if set(resolutions) != set(actual_issues):
        raise ValueError("every readiness finding/error requires an explicit disposition")
    for issue_id, row in resolutions.items():
        if row.get("status") not in {
            "resolved",
            "accepted_numerical_limitation",
            "explained_observation",
            "unresolved",
            "rejected",
        }:
            raise ValueError("explicit readiness finding status required")
        _reason(row.get("rationale"))
        if "diagnostic_error" in actual_issues[issue_id].get("finding", {}):
            # Missing computation is not a measurable field limitation. A new
            # successful evidence collection is required before generation.
            eligible = False
        if (
            row["status"] == "explained_observation"
            and actual_issues[issue_id].get("issue_type") != "observation"
        ):
            raise ValueError("only contextual observations may be explained as observations")
        refs = row.get("evidence_refs", [])
        if not refs or not set(refs) <= references:
            raise ValueError("finding treatment must cite available evidence")
        eligible &= row["status"] in {
            "resolved",
            "accepted_numerical_limitation",
            "explained_observation",
        }
    return bool(eligible)


def assess(evidence, review):
    """Bind caller's interpretation to computed evidence; create no approval."""
    return {
        "schema": SCHEMA,
        "evidence": evidence,
        "review": review,
        "generation_eligible": _eligibility(evidence, review),
        "scope": (
            "reviewed numerical study; not a physical-validity certificate "
            "or authenticated approval"
        ),
    }


def validate_assessment(transport, assessment, *, require_eligible=True):
    if assessment.get("schema") != SCHEMA:
        raise ValueError("bound field readiness assessment required")
    evidence = assessment["evidence"]
    if evidence["scope"] != readiness_scope(transport):
        raise ValueError("field readiness source/reconstruction/support scope changed")
    if evidence["implementation_sha256"] != sha256_json(implementation_identity()):
        raise ValueError("field readiness implementation changed")
    eligible = _eligibility(evidence, assessment["review"])
    if assessment.get("generation_eligible") is not eligible:
        raise ValueError("readiness disposition was modified")
    if require_eligible and not eligible:
        raise ValueError("unresolved or rejected field readiness; generation refused")


def collect_from_files(
    transport_path,
    output_path,
    *,
    shells,
    end_tolerance=None,
    working_memory_bytes=DEFAULT_WORKING_BYTES,
    scratch_directory=None,
) -> dict:
    from element_surrogate.advisories import readiness_warnings
    from element_surrogate.workflow import load_transport

    transport, _ = load_transport(transport_path)
    scratch = Path(output_path).resolve().parent if scratch_directory is None else scratch_directory
    evidence = collect_evidence(
        transport,
        shells=shells,
        end_tolerance=end_tolerance,
        working_memory_bytes=working_memory_bytes,
        scratch_directory=scratch,
    )
    write_json(output_path, evidence, exclusive=True)
    return {
        "evidence_path": str(Path(output_path).resolve()),
        "evidence_sha256": protocol_identity(evidence),
        "issues": evidence["issues"],
        "review_categories": list(CATEGORIES),
        "warnings": readiness_warnings(Path(output_path).resolve()),
    }


def assess_from_files(transport_path, evidence_path, review_path, output_path) -> dict:
    from element_surrogate.advisories import readiness_warnings
    from element_surrogate.workflow import load_transport

    transport, _ = load_transport(transport_path)
    assessment = assess(read_json(evidence_path), read_json(review_path))
    validate_assessment(transport, assessment, require_eligible=False)
    write_json(output_path, assessment, exclusive=True)
    return {
        "assessment_path": str(Path(output_path).resolve()),
        "assessment_sha256": protocol_identity(assessment),
        "generation_eligible": assessment["generation_eligible"],
        "scope": assessment["scope"],
        "warnings": readiness_warnings(Path(output_path).resolve(), assessment),
    }
