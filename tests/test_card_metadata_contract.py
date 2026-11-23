"""Rights and missing-fact controls independent of simulation or training."""

from copy import deepcopy

from element_surrogate.card_metadata import (
    inspect_request,
    require_valid_request,
    review_model_structure,
)


def supplied(role="particle", permitted=True):
    terms = {"spdx_id": "CC0-1.0", "name": "Synthetic fixture CC0 terms"}
    return {
        "frontmatter": {"supports_reusability": "Yes", "reusability": {"license": terms}},
        "rights_confirmation": {
            "status": "confirmed",
            "subject": role,
            "terms": deepcopy(terms),
            "downstream_use_permitted": permitted,
            "actor": "synthetic owner fixture",
            "user_statement": "These terms apply to this synthetic dataset.",
            "source_reference": "manufactured owner statement; not real authorization",
        },
    }


def test_reusable_claim_requires_explicit_subject_rights_not_a_license_string():
    metadata = supplied()
    metadata.pop("rights_confirmation")
    request = {"datasets": {"particle": metadata}}
    result = inspect_request(request)
    assert not result["valid"]
    assert any(f["code"] == "reuse_rights_unconfirmed" for f in result["findings"])
    import pytest

    with pytest.raises(ValueError, match="particle.*rights"):
        require_valid_request(request)


def test_pending_rights_remain_valid_draft_with_default_no_reuse():
    request = {"datasets": {"particle": {"rights_confirmation": {"status": "pending"}}}}
    result = inspect_request(request)
    assert result["valid"]
    assert not result["roles"]["particle"]["rights_confirmed"]
    assert not result["roles"]["model"]["rights_confirmed"]
    assert not result["release_metadata_ready"]


def test_malformed_nested_metadata_returns_findings_before_generation():
    request = {"datasets": {"field": {"frontmatter": {"discoverability": "unknown"}}}}
    result = inspect_request(request)
    assert not result["valid"]
    assert result["findings"][0]["code"] == "request_shape"


def test_confirmed_reuse_is_bound_to_subject_terms_and_permission():
    request = {"datasets": {"particle": supplied()}}
    assert inspect_request(request)["valid"]
    wrong_subject = deepcopy(request)
    wrong_subject["datasets"]["particle"]["rights_confirmation"]["subject"] = "field"
    assert not inspect_request(wrong_subject)["valid"]
    wrong_terms = deepcopy(request)
    wrong_terms["datasets"]["particle"]["frontmatter"]["reusability"]["license"]["name"] = (
        "Other terms"
    )
    assert not inspect_request(wrong_terms)["valid"]
    prohibited = {"datasets": {"particle": supplied(permitted=False)}}
    assert not inspect_request(prohibited)["valid"]


def test_field_rights_are_not_inherited_by_generated_labels_or_weights():
    request = {
        "datasets": {
            "field": supplied("field"),
            "particle": {
                "frontmatter": {
                    "supports_reusability": "Yes",
                    "reusability": {
                        "license": {"spdx_id": "CC0-1.0", "name": "Synthetic fixture CC0 terms"}
                    },
                }
            },
        }
    }
    result = inspect_request(request)
    assert not result["valid"]
    assert result["roles"]["field"]["rights_confirmed"]
    assert not result["roles"]["particle"]["rights_confirmed"]
    assert not result["roles"]["model"]["rights_confirmed"]


def test_explicit_empty_sponsors_are_known_empty_missing_or_null_are_unknown():
    request = {
        "datasets": {"field": {"frontmatter": {"discoverability": {"sponsor_organizations": []}}}}
    }
    result = inspect_request(request)
    assert "sponsor_organizations" not in result["roles"]["field"]["metadata_missing"]
    request["datasets"]["field"]["frontmatter"]["discoverability"]["sponsor_organizations"] = None
    assert "sponsor_organizations" in inspect_request(request)["roles"]["field"]["metadata_missing"]
    request["datasets"]["field"]["frontmatter"]["discoverability"].clear()
    assert "sponsor_organizations" in inspect_request(request)["roles"]["field"]["metadata_missing"]


def test_model_structure_checks_all_required_headings_without_registry_claim():
    headings = ["Model description", "Evaluation results", "Model license"]
    complete = (
        "# Fixture\n\n## Model description\nText.\n## Evaluation results\nNumbers.\n"
        "## Model license\nUnknown.\n"
    )
    assert review_model_structure(complete, headings)["template_structure_pass"]
    failed = review_model_structure(complete.replace("## Model license", "## Unrelated"), headings)
    assert not failed["template_structure_pass"]
    assert failed["missing_headings"] == ["Model license"]
    assert not failed["registry_validation_performed"]
