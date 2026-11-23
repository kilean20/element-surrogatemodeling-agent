"""Raw supplied-intake API contracts must work without generated-scope leftovers."""

from copy import deepcopy

import numpy as np
import pytest

from element_surrogate.benchmark.protocol import file_hash, read_json, write_json
from element_surrogate.native_scope import export_physics, mask, validate


def declaration(tmp_path, interval=None):
    scope = {
        "transport": {"identity": "manufactured fixed native map"},
        "ranges": [[-0.01, 0.01]] * 4,
        "settings": [{"id": "nominal", "effective_rigidity_tm": 2.0, "field_scale": 1.0}],
    }
    if interval is not None:
        scope["effective_coupling_range"] = interval
    source = tmp_path / "source.json"
    write_json(source, {"source_contract": scope})
    return {
        "coordinates": [{"name": str(i)} for i in range(4)],
        "physics": {"native_planes": True, "transport_description": scope["transport"]},
        "dataset": {
            "supplied_source_contract": scope,
            "supplied_import_request": {"path": str(source), "sha256": file_hash(source)},
        },
    }


def test_bound_source_resolves_scope_without_changing_frozen_protocol(tmp_path):
    p = declaration(tmp_path, [-0.5, 0.5])
    before = deepcopy(p)
    result = export_physics(p)
    assert p == before
    assert result["qualified_ranges"] == p["dataset"]["supplied_source_contract"]["ranges"]
    assert result["effective_coupling_range"] == [-0.5, 0.5]
    assert result["input_scope_provenance"]["kind"] == "source_declared"
    assert not result["input_scope_provenance"]["new_physical_qualification"]
    assert "generation_record_path" not in p["dataset"]
    a = np.array([[0, 0, 0, 0, 2, 0.4], [0, 0, 0, 0, 2, 1.1], [0, 0, 0, 0, 0, 0]])
    assert mask(a, result, 4)[1].tolist() == [True, False, False]


def test_discrete_settings_do_not_become_a_continuous_interval(tmp_path):
    p = declaration(tmp_path)
    result = export_physics(p)
    assert result["effective_coupling_range"] is None
    assert mask([[0, 0, 0, 0, 2, 1], [0, 0, 0, 0, 2, 0.9]], result, 4)[1].tolist() == [True, False]


def test_missing_unbound_changed_and_conflicting_scope_fail(tmp_path):
    p = declaration(tmp_path)
    no_scope = deepcopy(p)
    no_scope["dataset"].pop("supplied_source_contract")
    with pytest.raises(ValueError, match="scope is missing"):
        export_physics(no_scope)
    unbound = deepcopy(p)
    unbound["dataset"].pop("supplied_import_request")
    with pytest.raises(ValueError, match="not bound"):
        export_physics(unbound)
    other_transport = deepcopy(p)
    other_transport["physics"]["transport_description"] = {"identity": "different"}
    with pytest.raises(ValueError, match="another transport"):
        export_physics(other_transport)
    conflict = deepcopy(p)
    conflict["physics"]["qualified_ranges"] = [[-0.02, 0.02]] * 4
    with pytest.raises(ValueError, match="conflicting"):
        export_physics(conflict)
    write_json(tmp_path / "source.json", {"source_contract": {}})
    with pytest.raises(ValueError, match="changed"):
        export_physics(p)


@pytest.mark.parametrize("broken", ["ranges", "empty", "rigidity", "interval"])
def test_invalid_runtime_scope_is_rejected(broken):
    p = {
        "qualified_ranges": [[-1, 1]] * 4,
        "qualified_settings": [{"effective_rigidity_tm": -2, "field_scale": 1}],
    }
    if broken == "ranges":
        p["qualified_ranges"][0] = [1, -1]
    elif broken == "empty":
        p["qualified_settings"] = []
    elif broken == "rigidity":
        p["qualified_settings"][0]["effective_rigidity_tm"] = 0
    else:
        p["effective_coupling_range"] = [0, float("inf")]
    with pytest.raises(ValueError):
        validate(p, 4)


def test_complete_legacy_contract_is_preserved_and_incomplete_contract_fails_early(tmp_path):
    from element_surrogate.export import ElementSurrogate
    from element_surrogate.public_interface import export_interface

    legacy = {
        "qualified_ranges": [[-1, 1]] * 4,
        "qualified_settings": [{"effective_rigidity_tm": -2, "field_scale": 1}],
    }
    assert export_physics({"physics": legacy, "dataset": {}, "coordinates": [0] * 4}) == legacy
    write_json(
        tmp_path / "contract.json",
        {
            "schema": "element_surrogate.native_model.v1",
            "coordinates": [{"unit": "m"}] * 4,
            "physics": {"native_planes": True},
            "files": {},
        },
    )
    with pytest.raises(ValueError, match="scope is missing"):
        ElementSurrogate(tmp_path)
    with pytest.raises(ValueError, match="scope is missing"):
        export_interface(
            tmp_path / "contract.json", tmp_path / "absent_request.json", tmp_path / "public.json"
        )
    assert not (tmp_path / "public.json").exists()


@pytest.mark.parametrize("dimension,continuous", [(4, False), (4, True), (5, True), (6, True)])
def test_supplied_export_load_and_public_prediction_from_source_only(
    tmp_path, dimension, continuous
):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from element_surrogate.benchmark.scoring import load_cohort
    from element_surrogate.benchmark.study import (
        create_study,
        evaluate_test,
        lock_selection,
        run_study,
    )
    from element_surrogate.export import ElementSurrogate, export_selected
    from element_surrogate.physics_review import review
    from element_surrogate.public_interface import PublicElementSurrogate, export_interface
    from element_surrogate.supplied_data import prepare
    from tests.test_physics_review import declared_review
    from tests.test_supplied_data import supplied_request
    from tests.workflow_fixture import approvals, qualified_plan

    path, body = supplied_request(tmp_path, dimension, continuous)
    _, _, plan = qualified_plan(dimension, continuous_coupling=continuous)
    # Emulate the actual user-supplied protocol, not a generated protocol's hidden fields.
    for key in ("qualified_ranges", "qualified_settings", "effective_coupling_range"):
        body["protocol"]["physics"].pop(key, None)
    assert "qualified_ranges" not in body["protocol"]["physics"]
    body["source_contract"] = deepcopy(plan)
    candidate = body["protocol"]["candidates"][1]
    candidate.update(kind="deterministic", seeds=[3], max_steps=0)
    body["protocol"]["candidates"] = [candidate]
    write_json(path, body)
    prepared = prepare(path, tmp_path / "prepared")
    p = read_json(prepared["protocol_path"])
    # This boundary regression does not claim a physics-model study. Preserve the
    # real pre-fit review gate and explicitly defer unmeasured manufactured topics.
    p["physics"]["modeling_review"] = review(p, declared_review(), tmp_path)
    write_json(prepared["protocol_path"], p)
    assert "generation_record_path" not in p["dataset"]
    assert "qualified_ranges" not in p["physics"]
    study = create_study(
        prepared["protocol_path"], tmp_path / "study", approvals(tmp_path / "decisions", p)
    )
    run_study(study["study_path"])
    lock_selection(study["study_path"])
    release = approvals(tmp_path / "release_approvals", p, ("release",))
    evaluate_test(study["study_path"], release)
    exported = export_selected(study["study_path"], tmp_path / "model", release)
    contract = read_json(exported["contract_path"])
    assert contract["physics"]["qualified_ranges"] == plan["ranges"]
    assert contract["physics"]["effective_coupling_range"] == plan.get("effective_coupling_range")
    assert not contract["physics"]["input_scope_provenance"]["new_physical_qualification"]
    cohort = load_cohort(p, tmp_path / "prepared", "validation")
    torch_model = ElementSurrogate(tmp_path / "model", backend="torch")
    onnx_model = ElementSurrogate(tmp_path / "model", backend="onnx")
    expected = torch_model.predict(cohort["inputs"], numerical_support_established=True)
    np.testing.assert_allclose(
        onnx_model.predict(cohort["inputs"], numerical_support_established=True),
        expected,
        atol=1e-12,
    )
    outside = cohort["inputs"][:1].copy()
    outside[0, 0] = plan["ranges"][0][1] + 1
    with pytest.raises(ValueError, match="outside"):
        onnx_model.predict(outside, numerical_support_established=True)
    request = {
        "state_units": ["mm", "mrad", "mm", "mrad"]
        + (["mm", "1"] if dimension == 6 else ["1"] if dimension == 5 else []),
        "control_bounds": {
            key: [min(s[key] for s in plan["settings"]), max(s[key] for s in plan["settings"])]
            for key in ("effective_rigidity_tm", "field_scale")
        },
    }
    write_json(tmp_path / "public_request.json", request)
    export_interface(
        tmp_path / "model/contract.json", tmp_path / "public_request.json", tmp_path / "public.json"
    )
    public = PublicElementSurrogate(tmp_path / "model", tmp_path / "public.json")
    factors = np.array(
        [1e-3] * 4 + ([1e-3, 1] if dimension == 6 else [1] if dimension == 5 else [])
    )
    controls = {
        "effective_rigidity_tm": cohort["inputs"][:, -2],
        "field_scale": cohort["inputs"][:, -1],
    }
    actual = public.predict(
        cohort["inputs"][:, :dimension] / factors, controls, numerical_support_established=True
    )
    np.testing.assert_allclose(actual, expected / factors, atol=1e-9)
