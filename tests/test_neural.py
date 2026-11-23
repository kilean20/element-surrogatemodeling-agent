"""Optional neural backend: actual learning and exact static-magnetic invariants."""

import copy

import numpy as np
import pytest

pytest.importorskip("torch")

from element_surrogate.neural import build_adapter, load_adapter


def configuration(dimension, base="train_affine"):
    return {
        "schema": "element_surrogate.static_magnetic_mlp.v1",
        "dimension": dimension,
        "hidden_widths": [12, 12],
        "base": base,
        "dtype": "float64",
        "device": "cpu",
        "threads": 1,
        "batch_size": 24,
        "inference_batch_size": 32,
        "learning_rate": 0.01,
        "loss_scales": [1.0] * dimension,
    }


@pytest.mark.parametrize("dimension", [4, 5, 6])
def test_polynomial_residual_base_native_invariants_restore_and_feature_budget(tmp_path, dimension):
    x = np.random.default_rng(83).uniform(-0.4, 0.4, (160, dimension + 2))
    x[:, -2] = 2.0
    y = x[:, :dimension].copy()
    y[:, 0] += 0.3 * x[:, 1] ** 2 + 0.2 * x[:, 0] * x[:, 2]
    if dimension == 6:
        y[:, 4] += 0.2 * x[:, 0] ** 2 + 0.1 * x[:, 5]
    config = configuration(dimension, "train_polynomial")
    config.update(base_degree=2, max_base_features=40)
    model = build_adapter(x, y, config, 31)
    np.testing.assert_allclose(model.predict(x), y, atol=1e-13, rtol=0)
    model.save(tmp_path / "polynomial.bin")
    restored = load_adapter(tmp_path / "polynomial.bin", config)
    np.testing.assert_array_equal(restored.predict(x), model.predict(x))
    if dimension > 4:
        np.testing.assert_array_equal(restored.predict(x)[:, -1], x[:, dimension - 1])
    if dimension == 6:
        moved = x.copy()
        moved[:, 4] += 1.0
        np.testing.assert_allclose(
            restored.predict(moved)[:, 4] - restored.predict(x)[:, 4], 1, atol=1e-15, rtol=0
        )
    wrong = copy.deepcopy(config)
    wrong["max_base_features"] = 1
    with pytest.raises(ValueError, match="feature budget"):
        build_adapter(x, y, wrong, 31)


@pytest.mark.parametrize("dimension", [4, 5, 6])
@pytest.mark.parametrize("base", ["none", "train_affine"])
def test_neural_training_restore_and_static_invariants(tmp_path, dimension, base):
    x = np.random.default_rng(8).uniform(-0.3, 0.3, (80, dimension + 2))
    x[:, -2] = 2.0
    y = x[:, :dimension].copy()
    y[:, 0] += 0.2 * x[:, 1] - x[:, 0] ** 3 + 0.03 * x[:, -1]
    y[:, 1] -= 0.1 * x[:, 0]
    if dimension == 6:
        y[:, 4] += 0.3 * x[:, 1] ** 2 + 0.2 * x[:, 5]
    config = configuration(dimension, base)
    model = build_adapter(x, y, config, 17)
    initial = np.mean((model.predict(x) - y) ** 2)
    for _ in range(2000):
        model.step()
    final = np.mean((model.predict(x) - y) ** 2)
    assert final < initial / 2
    checkpoint = tmp_path / "model.bin"
    model.save(checkpoint)
    restored = load_adapter(checkpoint, config)
    np.testing.assert_array_equal(restored.predict(x), model.predict(x))
    if dimension > 4:
        np.testing.assert_array_equal(restored.predict(x)[:, -1], x[:, dimension - 1])
    if dimension == 6:
        moved = x.copy()
        moved[:, 4] += 1.25
        difference = restored.predict(moved) - restored.predict(x)
        np.testing.assert_allclose(difference[:, 4], 1.25, atol=1e-15, rtol=0)
        np.testing.assert_array_equal(difference[:, :4], 0 * difference[:, :4])


def test_reflection_projection_is_exact_and_requires_evidence():
    x = np.random.default_rng(6).uniform(-1, 1, (12, 6))
    x[:, :4] *= 0.001
    x[:, -2:] = [2.0, 1.0]
    from element_surrogate.benchmark.protocol import protocol_identity
    from element_surrogate.symmetry import assess
    from tests.workflow_fixture import manufactured_transport

    transport, _, ranges, _ = manufactured_transport()
    y = transport.run(x[:, :4], 2.0, 1.0)[0]
    config = configuration(4)
    signs = np.array([-1, -1, -1, -1, 1, 1])
    config["reflection_group"] = [[1] * 6, signs.tolist()]
    with pytest.raises(ValueError, match="evidence"):
        build_adapter(x, y, config, 1)
    evidence = assess(
        transport,
        ranges,
        [{"id": "test", "effective_rigidity_tm": 2.0, "field_scale": 1.0}],
        config["reflection_group"],
        [1e-12] * 4,
        random_probes=0,
        seed=0,
        rationale="Manufactured central reflection control; finite paired tracking evidence",
    )
    config["symmetry_assessment"] = evidence
    config["symmetry_evidence_sha256"] = protocol_identity(evidence)
    model = build_adapter(x, y, config, 1)
    for _ in range(3):
        model.step()
    np.testing.assert_array_equal(model.predict(x * signs), model.predict(x) * signs[:4])
    wrong = copy.deepcopy(config)
    wrong["reflection_group"] = [[1] * 6, [-1, 1, 1, 1, 1, 1], [1, -1, 1, 1, 1, 1]]
    with pytest.raises(ValueError, match="closed"):
        build_adapter(x, y, wrong, 1)


def test_nonconserved_labels_and_checkpoint_configuration_refused(tmp_path):
    x = np.zeros((8, 7))
    x[:, -2] = 2.0
    y = np.zeros((8, 5))
    y[:, 4] = 1
    config = configuration(5)
    with pytest.raises(ValueError, match="conserve"):
        build_adapter(x, y, config, 1)
    model = build_adapter(x, np.zeros((8, 5)), config, 1)
    model.save(tmp_path / "model.bin")
    wrong = copy.deepcopy(config)
    wrong["learning_rate"] *= 2
    with pytest.raises(ValueError, match="configuration"):
        load_adapter(tmp_path / "model.bin", wrong)


def test_effective_strength_reduction_is_exact():
    x = np.random.default_rng(17).uniform(-0.2, 0.2, (20, 6))
    x[:, -2] = 2.0
    model = build_adapter(x, x[:, :4], configuration(4), 12)
    for _ in range(5):
        model.step()
    scaled = x.copy()
    scaled[:, -2:] *= 2
    np.testing.assert_array_equal(model.predict(scaled), model.predict(x))


def test_unsupported_hyperparameter_is_not_silently_ignored():
    config = configuration(4)
    config["activation"] = "relu"
    with pytest.raises(ValueError, match="unknown neural configuration"):
        build_adapter(np.ones((4, 6)), np.zeros((4, 4)), config, 1)
