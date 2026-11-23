"""The same current physics eligibility applies to both native label origins."""

from copy import deepcopy

import pytest

pytest.importorskip("torch")

from element_surrogate import neural, theory
from element_surrogate.benchmark.protocol import protocol_identity
from element_surrogate.candidate_scope import validate_candidate_scope
from tests.test_canonical import configuration as canonical_configuration
from tests.test_equivariant_features import setup
from tests.workflow_fixture import manufactured_transport


@pytest.mark.parametrize("family", ["reflection", "canonical", "theory"])
def test_current_evidence_and_transport_are_required_before_a_new_fit(family):
    if family == "canonical":
        config, _ = canonical_configuration()
        key = "canonical_contract"
        description = config[key]["scope"]["transport"]
    elif family == "reflection":
        config, _, _ = setup()
        key = "symmetry_assessment"
        description = config[key]["scope"]["transport"]
    else:
        config, _, _ = setup()
        config["model_family"] = "mlp"
        config["reflection_group"] = [[1] * 6]
        config.pop("symmetry_assessment")
        config.pop("symmetry_evidence_sha256")
        config["base"] = "theory_drift"
        key = "theory_contract"
        t, *_ = manufactured_transport()
        config[key] = theory.assess(t, "drift", "Manufactured current native-base eligibility")
        description = t.description()
    validate_candidate_scope(config, description)
    old = deepcopy(config)
    old[key]["implementation_sha256"] = "0" * 64
    if family == "reflection":
        old["symmetry_evidence_sha256"] = protocol_identity(old[key])
    neural._configuration(old)  # Archived inference can retain its historical evidence.
    with pytest.raises(ValueError, match="implementation"):
        validate_candidate_scope(old, description)
    with pytest.raises(ValueError, match="transport"):
        validate_candidate_scope(config, {**description, "coordinates": "another contract"})


@pytest.mark.parametrize("mode", ["mean_std", "zero_centered_rms"])
def test_canonical_normalization_cannot_silently_ignore_a_generic_option(mode):
    config, _ = canonical_configuration()
    config["input_normalization"] = mode
    with pytest.raises(ValueError, match="canonical_contract"):
        neural._configuration(config)
