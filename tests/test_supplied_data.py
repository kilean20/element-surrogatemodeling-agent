"""Provided labels retain their identity/history without fictitious generation."""

from copy import deepcopy
from pathlib import Path

import pytest

pytest.importorskip("torch")

from element_surrogate.benchmark.protocol import read_json, write_json
from element_surrogate.data import generate
from element_surrogate.modeling import prepare_generated_study
from element_surrogate.supplied_data import SCHEMA, prepare
from tests.test_physical_workflow import study_request
from tests.workflow_fixture import approvals, qualified_plan


def supplied_request(tmp_path, dimension=4, continuous_coupling=False):
    transport, qualification, plan = qualified_plan(
        dimension, continuous_coupling=continuous_coupling
    )
    generate(
        transport, qualification, plan, tmp_path / "source", approvals(tmp_path / "approvals", plan)
    )
    request = study_request(dimension)
    p = read_json(
        prepare_generated_study(tmp_path / "source", tmp_path / "original_protocol", request)[
            "protocol_path"
        ]
    )
    p["dataset"].pop("generation_record_path")
    for v in p["dataset"]["splits"].values():
        v["path"] = str((tmp_path / "original_protocol" / v["path"]).resolve())
    path = tmp_path / "provided_request.json"
    body = {
        "schema": SCHEMA,
        "protocol": p,
        "label_provenance": {
            "description": (
                "Manufactured supplied label snapshot. Its producer study remains separate."
            ),
            "qualification_status": "source_declared",
            "split_history": {
                "train": "previously_used",
                "validation": "previously_used",
                "test": "unknown",
            },
        },
    }
    write_json(path, body)
    return path, body


def test_supplied_train_and_validation_only_no_new_generation(tmp_path, monkeypatch):
    path, body = supplied_request(tmp_path)
    from element_surrogate.benchmark import scoring

    old = scoring.load_cohort
    used = []

    def tracked(p, base, split):
        assert split != "test"
        used.append(split)
        return old(p, base, split)

    monkeypatch.setattr(scoring, "load_cohort", tracked)
    result = prepare(path, tmp_path / "new_study")
    assert used == ["train", "validation"]
    assert not result["new_training_labels_generated"] and not result["test_arrays_read"]
    p = read_json(result["protocol_path"])
    assert "generation_record_path" not in p["dataset"]
    assert p["dataset"]["identity_sha256"] == body["protocol"]["dataset"]["identity_sha256"]
    assert p["dataset"]["supplied_label_provenance"]["split_history"]["test"] == "unknown"
    assert all(Path(i["path"]).is_absolute() for i in p["dataset"]["splits"].values())
    assert all(c["configuration"]["output_normalization"] == "train_rms" for c in p["candidates"])


def test_history_units_and_identity_errors_stop_before_output(tmp_path):
    path, body = supplied_request(tmp_path)
    cases = []
    bad = deepcopy(body)
    bad["label_provenance"].pop("split_history")
    cases.append((bad, "prior-use"))
    bad = deepcopy(body)
    bad["protocol"]["coordinates"][0]["unit"] = "mm"
    cases.append((bad, "m/rad"))
    bad = deepcopy(body)
    bad["protocol"]["dataset"]["splits"]["train"]["sha256"] = "0" * 64
    cases.append((bad, "SHA256"))
    bad = deepcopy(body)
    bad["protocol"]["dataset"]["generation_record_path"] = "invented.json"
    cases.append((bad, "masquerade"))
    for i, (value, message) in enumerate(cases):
        write_json(path, value)
        with pytest.raises(ValueError, match=message):
            prepare(path, tmp_path / f"rejected_{i}")
        assert not (tmp_path / f"rejected_{i}").exists()


def test_supplied_reference_sample_checks_controls_status_units_and_scope(tmp_path):
    from element_surrogate.supplied_data import check_reference

    path, body = supplied_request(tmp_path)
    t, _, _ = qualified_plan(4)
    p = read_json(prepare(path, tmp_path / "imported")["protocol_path"])
    request = {"split": "validation", "sample_rows": 5, "seed": 21, "component_limits": [1e-12] * 4}
    result = check_reference(t, p, tmp_path, request)
    assert result["within_limits"] and result["unsupported_rows"] == 0
    assert result["sample_rows"] == 5 and not result["new_training_labels_generated"]
    assert check_reference(t, p, tmp_path, request)["sample_ids"] == result["sample_ids"]
    bad = deepcopy(p)
    bad["coordinates"][0]["unit"] = "mm"
    with pytest.raises(ValueError, match="coordinates"):
        check_reference(t, bad, tmp_path, request)
    bad = deepcopy(p)
    bad["physics"]["transport_description"]["coordinates"] = "another convention"
    with pytest.raises(ValueError, match="transport differs"):
        check_reference(t, bad, tmp_path, request)
    with pytest.raises(ValueError, match="train/validation"):
        check_reference(t, p, tmp_path, {**request, "split": "test"})
    with pytest.raises(ValueError, match="limits"):
        check_reference(t, p, tmp_path, {**request, "component_limits": [0] * 4})


def test_supplied_structured_candidate_uses_the_same_current_full_scope(tmp_path):
    from element_surrogate import symmetry
    from element_surrogate.benchmark.protocol import protocol_identity
    from tests.test_equivariant_features import group

    path, body = supplied_request(tmp_path)
    t, _, plan = qualified_plan(4)
    config = body["protocol"]["candidates"][0]["configuration"]
    evidence = symmetry.assess(
        t,
        plan["ranges"],
        plan["settings"],
        group(4),
        [1e-8] * 4,
        random_probes=0,
        seed=77,
        rationale="Manufactured supplied-label reflection eligibility",
    )
    assert evidence["supported"]
    config.update(
        model_family="reflection_features",
        input_normalization="zero_centered_rms",
        reflection_group=group(4),
        symmetry_assessment=evidence,
        symmetry_evidence_sha256=protocol_identity(evidence),
    )
    body["source_contract"] = plan
    write_json(path, body)
    assert prepare(path, tmp_path / "current_structured")["rows"]["train"] > 0
    evidence["implementation_sha256"] = "0" * 64
    config["symmetry_evidence_sha256"] = protocol_identity(evidence)
    write_json(path, body)
    with pytest.raises(ValueError, match="implementation"):
        prepare(path, tmp_path / "old_structured")
    assert not (tmp_path / "old_structured").exists()
