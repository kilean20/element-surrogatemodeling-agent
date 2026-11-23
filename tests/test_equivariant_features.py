"""Physical symmetry eligibility, exact algebra, learning and portable export."""

from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from element_surrogate import equivariant, neural
from element_surrogate.benchmark.protocol import protocol_identity
from element_surrogate.symmetry import assess
from tests.test_neural import configuration
from tests.workflow_fixture import manufactured_transport


def group(d, kind="both"):
    h = [-1, -1, 1, 1] + [1] * (d - 2)
    v = [1, 1, -1, -1] + [1] * (d - 2)
    identity = [1] * (d + 2)
    if kind == "horizontal":
        return [identity, h]
    if kind == "central":
        return [identity, (np.array(h) * v).tolist()]
    return [identity, h, v, (np.array(h) * v).tolist()]


def setup(d=4, base="none", kind="both"):
    t, _, ranges, _ = manufactured_transport(d)
    signs = group(d, kind)
    evidence = assess(
        t,
        ranges,
        [{"id": "control", "effective_rigidity_tm": 2.0, "field_scale": 1.0}],
        signs,
        [1e-8] * d,
        random_probes=0,
        seed=31,
        rationale="Manufactured paired-plane reflection test",
    )
    assert evidence["supported"]
    config = configuration(d, base)
    config.update(
        model_family="reflection_features",
        input_normalization="zero_centered_rms",
        output_normalization="train_rms",
        reflection_group=signs,
        symmetry_assessment=evidence,
        symmetry_evidence_sha256=protocol_identity(evidence),
    )
    if base == "train_polynomial":
        config.update(base_degree=3, max_base_features=500)
    x = np.random.default_rng(17).uniform(np.array(ranges)[:, 0], np.array(ranges)[:, 1], (64, d))
    x = np.column_stack([x, np.full(64, 2.0), np.ones(64)])
    y = x[:, :d].copy()
    y[:, 0] += 0.13 * x[:, 1] + 30 * x[:, 0] ** 3
    y[:, 2] -= 0.17 * x[:, 3]
    if d == 6:
        y[:, 4] += 0.01 * x[:, 1] ** 2
    return config, x, y


@pytest.mark.parametrize("dimension", [4, 5, 6])
@pytest.mark.parametrize("kind", ["both", "horizontal", "central"])
def test_feature_algebra_static_invariants_and_one_network_call(tmp_path, dimension, kind):
    config, x, y = setup(dimension, kind=kind)
    model = neural.build_adapter(x, y, config, 23)
    for _ in range(8):
        model.step()
    calls = []
    hook = model.module.network.register_forward_hook(lambda *args: calls.append(1))
    result = model.predict(x[:7])
    hook.remove()
    assert len(calls) == 1
    for signs in config["reflection_group"]:
        transformed = model.predict(x * signs)
        np.testing.assert_allclose(
            transformed, model.predict(x) * np.asarray(signs[:dimension]), atol=1e-14, rtol=1e-13
        )
    if dimension > 4:
        np.testing.assert_array_equal(model.predict(x)[:, -1], x[:, dimension - 1])
    if dimension == 6:
        shifted = x.copy()
        shifted[:, 4] += 2
        np.testing.assert_allclose(
            model.predict(shifted)[:, 4] - model.predict(x)[:, 4], 2, atol=1e-14
        )
        np.testing.assert_array_equal(model.predict(shifted)[:, :4], model.predict(x)[:, :4])
    file = tmp_path / "model.bin"
    model.save(file)
    restored = neural.load_adapter(file, config)
    np.testing.assert_array_equal(restored.predict(x[:7]), result)


@pytest.mark.parametrize("base", ["train_affine", "train_polynomial"])
def test_fitted_base_obeys_the_same_characters(base):
    config, x, y = setup(base=base)
    model = neural.build_adapter(x, y, config, 23)
    assert np.max(np.abs(model.predict(x) - y)) < (1e-6 if base == "train_affine" else 1e-12)
    for signs in config["reflection_group"]:
        np.testing.assert_allclose(
            model.predict(x * signs), model.predict(x) * signs[:4], atol=1e-14
        )
    config["base_degree"] = 5
    config["max_base_features"] = 500
    config["base"] = "train_polynomial"
    _, mask = equivariant.base_layout(config)
    assert int(mask.sum()) == 272


def test_feature_training_improves_error_without_changing_normalization():
    config, x, y = setup()
    model = neural.build_adapter(x, y, config, 23)
    initial = np.mean((model.predict(x) - y) ** 2)
    scales = model.module.output_scale.clone()
    for _ in range(150):
        model.step()
    assert np.mean((model.predict(x) - y) ** 2) < initial / 20
    assert torch.equal(scales, model.module.output_scale)


def test_missing_or_rejected_evidence_and_mean_shift_are_not_ignored():
    config, x, y = setup()
    bad = deepcopy(config)
    bad.pop("symmetry_assessment")
    with pytest.raises((ValueError, TypeError)):
        neural.build_adapter(x, y, bad, 2)
    bad = deepcopy(config)
    bad["input_normalization"] = "mean_std"
    with pytest.raises(ValueError, match="zero_centered_rms"):
        neural.build_adapter(x, y, bad, 2)
    bad = deepcopy(config)
    bad["symmetry_assessment"]["supported"] = False
    with pytest.raises(ValueError):
        neural.build_adapter(x, y, bad, 2)


@pytest.mark.parametrize("dimension", [4, 5, 6])
@pytest.mark.parametrize("base", ["none", "train_polynomial"])
def test_onnx_matches_one_pass_feature_map(tmp_path, dimension, base):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    config, x, y = setup(dimension, base=base)
    model = neural.build_adapter(x, y, config, 23)
    for _ in range(3):
        model.step()
    path = tmp_path / "map.onnx"
    torch.onnx.export(
        model.module.eval(),
        (torch.as_tensor(x[:2], dtype=torch.float64),),
        str(path),
        dynamo=False,
        opset_version=17,
        input_names=["inputs"],
        output_names=["outputs"],
        dynamic_axes={"inputs": {0: "batch"}, "outputs": {0: "batch"}},
    )
    runtime = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for n in (1, 7, len(x)):
        np.testing.assert_allclose(
            runtime.run(None, {"inputs": x[:n]})[0], model.predict(x[:n]), atol=1e-12, rtol=1e-12
        )


def test_saved_basis_cannot_change_the_symmetry_contract(tmp_path):
    config, x, y = setup()
    model = neural.build_adapter(x, y, config, 23)
    path = tmp_path / "map.bin"
    model.save(path)
    saved = torch.load(path, weights_only=True)
    saved["state_dict"]["invariant_powers"][0, 0] += 1
    torch.save(saved, path)
    with pytest.raises(ValueError, match="characters/basis"):
        neural.load_adapter(path, config)


def test_real_asymmetry_can_use_a_breaking_branch_without_a_false_guarantee(tmp_path):
    from dataclasses import replace

    config, x, _ = setup(base="train_affine")
    t, _, ranges, _ = manufactured_transport()
    b = t.grid.b_t.copy()
    b[..., 0] += 0.02
    t = replace(t, grid=t.grid.with_values(b))
    measured = assess(
        t,
        ranges,
        [{"id": "control", "effective_rigidity_tm": 2.0, "field_scale": 1.0}],
        config["reflection_group"],
        [1e-8] * 4,
        random_probes=0,
        seed=31,
        rationale="Manufactured real dipole offset breaks quadrupole reflection",
    )
    assert not measured["supported"]
    config.update(
        symmetry_assessment=measured, symmetry_evidence_sha256=protocol_identity(measured)
    )
    with pytest.raises(ValueError):
        neural.build_adapter(x, x[:, :4], config, 3)
    config.update(symmetry_breaking=True, breaking_hidden_widths=[8])
    target = t.run(x[:, :4], 2.0, 1.0)[0]
    model = neural.build_adapter(x, target, config, 3)
    initial = np.mean((model.predict(x) - target) ** 2)
    for _ in range(150):
        model.step()
    assert np.mean((model.predict(x) - target) ** 2) < initial / 10
    assert not model.normalization_report["complete_reflection_equivariance_enforced"]
    assert model.normalization_report["network_evaluations_per_prediction"] == 2
    assert "no complete-map reflection guarantee" in neural.forward_description(config)
    path = tmp_path / "breaking.bin"
    model.save(path)
    np.testing.assert_array_equal(neural.load_adapter(path, config).predict(x), model.predict(x))
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    path = tmp_path / "breaking.onnx"
    torch.onnx.export(
        model.module.eval(),
        (torch.as_tensor(x[:2], dtype=torch.float64),),
        str(path),
        dynamo=False,
        opset_version=17,
        input_names=["inputs"],
        output_names=["outputs"],
        dynamic_axes={"inputs": {0: "batch"}, "outputs": {0: "batch"}},
    )
    runtime = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    np.testing.assert_allclose(
        runtime.run(None, {"inputs": x[:7]})[0], model.predict(x[:7]), atol=1e-12, rtol=1e-12
    )
