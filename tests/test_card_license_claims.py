"""A licence claim cannot contradict pending or artifact-specific rights."""

from copy import deepcopy

from element_surrogate.card_metadata import inspect_request
from tests.test_card_metadata_contract import supplied


def test_model_frontmatter_cannot_claim_a_license_while_weight_rights_are_pending():
    request = {
        "model": {
            "frontmatter": {"license": "CC0-1.0"},
            "rights_confirmation": {"status": "pending"},
        }
    }
    result = inspect_request(request)
    assert not result["valid"]
    assert any(item["code"] == "license_rights_unconfirmed" for item in result["findings"])


def test_model_frontmatter_license_matches_explicit_weight_terms():
    model = supplied("model")
    model["frontmatter"] = {"license": "CC0-1.0"}
    assert inspect_request({"model": model})["valid"]
    wrong = deepcopy(model)
    wrong["frontmatter"]["license"] = "Apache-2.0"
    assert not inspect_request({"model": wrong})["valid"]


def test_dataset_license_is_not_free_to_disagree_when_reuse_flag_is_no():
    particle = supplied("particle")
    particle["frontmatter"]["supports_reusability"] = "No"
    particle["rights_confirmation"] = {"status": "pending"}
    assert not inspect_request({"datasets": {"particle": particle}})["valid"]
