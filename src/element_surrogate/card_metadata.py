"""Explicit rights and documentation checks; no licence or identity is inferred."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

SCHEMA = Path(__file__).parent / "schemas/card_request.json"
ROLES = ("field", "particle", "model")


@lru_cache(maxsize=1)
def software_version_pattern():
    import yaml

    path = (
        Path(__file__).parent / "_vendor/genesis/datacard-generator/scripts/genesis_datacard.yaml"
    )
    return yaml.safe_load(path.read_text("utf-8"))["slots"]["version"]["pattern"]


def software_identity(name: str, version: str) -> dict:
    """Faithful identity even when the pinned optional version slot is narrower."""
    if not all(isinstance(v, str) and v.strip() for v in (name, version)):
        raise ValueError("software name and exact version must be nonempty strings")
    result = {
        "name": name,
        "identifier": {"type": "local", "value": f"{name}@{version}"},
        "relationship": "used_to_create",
    }
    if re.fullmatch(software_version_pattern(), version):
        result["version"] = version
    return result


def software_preflight(front):
    findings = []
    creators = front.get("discoverability", {}).get("datacard", {}).get("created_by", [])
    for i, contribution in enumerate(creators or []):
        if not isinstance(contribution, dict):
            continue
        creator = contribution.get("creator")
        if not isinstance(creator, dict):
            continue
        software = creator.get("software")
        if not isinstance(software, dict):
            continue
        version = software.get("version")
        if version is not None and not re.fullmatch(software_version_pattern(), str(version)):
            findings.append(
                {
                    "category": "software_schema_compatibility",
                    "path": f"discoverability.datacard.created_by.{i}.creator.software.version",
                    "message": (
                        "Pinned optional numeric version slot cannot encode this exact version."
                    ),
                    "exact_version": version,
                    "suggested_software_identity": software_identity(
                        software.get("name", "software"), str(version)
                    ),
                    "request_modified": False,
                }
            )
    return findings


def normalize_heading(value: str) -> str:
    """One normalization rule for supplied keys and pinned template headings."""
    return " ".join(value.strip().lower().rstrip(". ").split())


def model_sections(supplied: dict) -> tuple[dict, dict]:
    headings = json.loads(
        (Path(__file__).parent / "_vendor/genesis/model_card_headings.json").read_text("utf-8")
    )["headings"]
    allowed = {normalize_heading(h) for h in headings}
    result, sources = {}, {}
    for key, value in supplied.get("sections", {}).items():
        normalized = normalize_heading(key)
        if normalized not in allowed:
            raise ValueError(f"unknown model section {key!r}; use a pinned template heading")
        if normalized in result:
            raise ValueError(f"colliding model section headings: {sources[normalized]!r}, {key!r}")
        result[normalized], sources[normalized] = value, key
    return result, {
        "supplied_to_normalized": {original: key for key, original in sources.items()},
        "narrative_sections_applied": sorted(set(result) - {"model license"}),
        "controlled_sections": {"model license": "rendered from explicit artifact rights"}
        if "model license" in result
        else {},
        "unused_sections": [],
    }


def schema_document() -> dict[str, Any]:
    value = json.loads(SCHEMA.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("card metadata schema must be an object")
    return value


def inspect_request(request: dict) -> dict:
    """Check metadata before simulation-dependent card generation.

    A confirmation is a recorded caller attestation, never authenticated owner
    identity or institutional release authority. Rights remain separate for the
    original field, generated labels and trained weights.
    """
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise ValueError("install element-surrogate[cards] for metadata validation") from exc

    findings: list[dict[str, str]] = []

    def issue(code, path, message):
        findings.append({"code": code, "path": path, "message": message, "severity": "error"})

    for error in Draft202012Validator(schema_document()).iter_errors(request):
        path = ".".join(map(str, error.absolute_path)) or "request"
        issue("request_shape", path, error.message)
    if findings:
        return {
            "schema": "element_surrogate.card_metadata_review.v1",
            "valid": False,
            "findings": findings,
            "roles": {},
            "release_metadata_ready": False,
        }

    from element_surrogate.model_provenance import input_findings

    section_review = None
    try:
        _, section_review = model_sections(request.get("model", {}))
    except ValueError as exc:
        issue("model_sections", "model.sections", str(exc))

    for finding in input_findings(request.get("model", {})):
        issue("physical_provenance", finding["path"], finding["message"])
    roles = {}
    for role in ROLES:
        supplied = (
            request.get("model", {})
            if role == "model"
            else request.get("datasets", {}).get(role, {})
        )
        front = supplied.get("frontmatter", {})
        rights = supplied.get("rights_confirmation", {})
        confirmed = rights.get("status") == "confirmed"
        prefix = "model" if role == "model" else "datasets." + role
        if confirmed and rights["subject"] != role:
            issue(
                "rights_subject_mismatch",
                prefix + ".rights_confirmation.subject",
                "rights confirmation must describe this " + role + " artifact",
            )
            confirmed = False
        claim = (
            front.get("license") if role == "model" else front.get("reusability", {}).get("license")
        )
        claim_path = (
            prefix + ".frontmatter." + ("license" if role == "model" else "reusability.license")
        )
        if claim is not None:
            if not confirmed:
                issue(
                    "license_rights_unconfirmed",
                    claim_path,
                    "omit a licence claim until this artifact's rights are confirmed",
                )
            else:
                expected = (
                    rights["terms"].get("spdx_id", rights["terms"]["name"])
                    if role == "model"
                    else rights["terms"]
                )
                if claim != expected:
                    issue(
                        "rights_terms_mismatch",
                        claim_path,
                        "licence claim must match this artifact's explicit terms",
                    )
        if role != "model" and front.get("supports_reusability", "No") == "Yes":
            if not confirmed:
                issue(
                    "reuse_rights_unconfirmed",
                    prefix + ".rights_confirmation",
                    role + " rights are unconfirmed; propose supports_reusability=No "
                    "or supply explicit rights evidence",
                )
            elif not rights["downstream_use_permitted"]:
                issue(
                    "reuse_not_permitted",
                    prefix + ".frontmatter.supports_reusability",
                    "a reusable claim requires attested downstream-use permission",
                )
            elif front.get("reusability", {}).get("license") != rights["terms"]:
                issue(
                    "rights_terms_mismatch",
                    prefix + ".frontmatter.reusability.license",
                    "card licence must match the terms bound to this rights confirmation",
                )
        discover = front.get("discoverability", {})
        missing = (
            []
            if role == "model"
            else [
                key
                for key in (
                    "contact",
                    "authors",
                    "sponsor_organizations",
                    "research_organizations",
                    "sensitivity",
                    "dataset_publisher",
                )
                if key not in discover
                or discover[key] is None
                or (
                    key not in ("sponsor_organizations", "research_organizations")
                    and not discover[key]
                )
            ]
        )
        confirmation = supplied.get("capability_confirmation", {})
        capabilities = all(
            isinstance(confirmation.get(key), str) and confirmation[key].strip()
            for key in ("actor", "user_statement", "source_reference")
        )
        roles[role] = {
            "rights_confirmed": confirmed,
            "downstream_use_permitted": rights.get("downstream_use_permitted")
            if confirmed
            else None,
            "rights_state": rights.get("status", "not_supplied"),
            "metadata_missing": missing,
            "capabilities_confirmed": capabilities if role != "model" else None,
            "software_schema_findings": software_preflight(front) if role != "model" else [],
        }
    ready = not findings and all(value["rights_confirmed"] for value in roles.values())
    ready = ready and all(
        not roles[kind]["metadata_missing"]
        and roles[kind]["capabilities_confirmed"]
        and not roles[kind]["software_schema_findings"]
        for kind in ("field", "particle")
    )
    return {
        "schema": "element_surrogate.card_metadata_review.v1",
        "valid": not findings,
        "findings": findings,
        "roles": roles,
        "model_section_coverage": section_review,
        "release_metadata_ready": bool(ready),
        "confirmation_scope": "caller attestation; identity and release approval not established",
    }


def require_valid_request(request: dict) -> dict:
    review = inspect_request(request)
    if not review["valid"]:
        raise ValueError("; ".join(f["path"] + ": " + f["message"] for f in review["findings"]))
    return review


def review_model_structure(markdown: str, required_headings: list[str]) -> dict:
    """Check the complete pinned template separately from evaluation/registry checks."""

    def normalize(value):
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    present = {
        normalize(heading) for heading in re.findall(r"^##\s+(.+?)\s*$", markdown, re.MULTILINE)
    }
    missing = [heading for heading in required_headings if normalize(heading) not in present]
    return {
        "template_structure_pass": not missing,
        "missing_headings": missing,
        "required_heading_count": len(required_headings),
        "registry_validation_performed": False,
        "scope": "pinned model-template headings only; evaluation and owner facts are separate",
    }


def classify_upstream_data_validation(frontmatter: dict, log: str, passed: bool) -> dict:
    """Describe the pinned Draft publisher issue without changing upstream validation."""
    discover = frontmatter.get("discoverability", {})
    errors = [line for line in log.splitlines() if "[ERROR]" in line]
    human = {
        "contact",
        "authors",
        "sponsor_organizations",
        "research_organizations",
        "sensitivity",
        "dataset_publisher",
    }
    classified = []
    for line in errors:
        missing = re.search(r"'([^']+)' is a required property in /discoverability(?:\s|$)", line)
        category = "missing_user_metadata" if missing and missing[1] in human else "schema_other"
        if "/software/version" in line and "does not match" in line:
            category = "software_schema_compatibility"
        classified.append({"category": category, "raw_finding": line})
    known = bool(
        not passed
        and discover.get("release_status") == "Draft"
        and not discover.get("dataset_publisher")
        and errors
        and all(
            "'dataset_publisher' is a required property in /discoverability" in line
            for line in errors
        )
    )
    return {
        "upstream_schema_pass": passed,
        "known_draft_publisher_precondition_only": known,
        "upstream_errors_reported": len(errors),
        "classified_findings": classified,
        "upstream_schema_modified": False,
        "scope": "raw pinned LinkML result retained; issue classification is not conformance",
    }


def model_rights_text(supplied: dict) -> str:
    """Render a licence statement from explicit rights facts, not placeholder prose."""
    rights = supplied.get("rights_confirmation", {})
    if rights.get("status") != "confirmed":
        return (
            "Trained-weight rights remain pending explicit confirmation. No licence is "
            "inherited from the source field, generated labels or software. No release "
            "authority is supplied by this draft."
        )
    terms = rights["terms"]
    permission = "permitted" if rights["downstream_use_permitted"] else "not permitted"
    return (
        f"Supplied weight terms: {terms['name']}"
        + (f" ({terms['spdx_id']})" if terms.get("spdx_id") else "")
        + f". Downstream use is {permission} under the caller-attested terms. "
        + f"Source: {rights['source_reference']}. Statement: {rights['user_statement']}. "
        + "This records metadata; it does not authenticate the actor or approve publication."
    )
