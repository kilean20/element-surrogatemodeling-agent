import copy

import numpy as np
import pytest

from element_surrogate.scaling import (
    mechanical_to_slopes,
    output_scales,
    scaled_canonical_form,
    slopes_to_mechanical,
)


def test_slope_momentum_chart_and_curved_metric_are_distinct():
    slopes = np.array([[0.3, -0.4], [0.01, 0.02]])
    delta = np.array([0.2, -0.1])
    metric = np.array([1.2, 0.8])
    momentum = slopes_to_mechanical(slopes, delta, metric)
    np.testing.assert_allclose(mechanical_to_slopes(momentum, delta, metric), slopes, atol=1e-16)
    assert not np.allclose(slopes, momentum)
    np.testing.assert_allclose(slopes_to_mechanical(slopes / metric[:, None], delta), momentum)
    with pytest.raises(ValueError, match="forward"):
        mechanical_to_slopes([[1.0, 1.0]])
    with pytest.raises(ValueError):
        slopes_to_mechanical([[0.0, 0.0]], longitudinal_metric=-1)


def test_scaled_canonical_form_requires_weights_for_unequal_pair_products():
    scales = np.array([2.0, 3.0, 4.0, 5.0])
    form = scaled_canonical_form(scales)
    j = scaled_canonical_form(np.ones(4))
    # Canonical rotation mixes the two transverse pairs.
    angle = 0.37
    c, s = np.cos(angle), np.sin(angle)
    matrix = np.array([[c, 0, s, 0], [0, c, 0, s], [-s, 0, c, 0], [0, -s, 0, c]])
    normalized = matrix * scales[None, :] / scales[:, None]
    np.testing.assert_allclose(normalized.T @ form @ normalized, form, atol=4e-15)
    assert np.max(np.abs(normalized.T @ j @ normalized - j)) > 0.1


@pytest.mark.parametrize("dimension", [4, 5, 6])
def test_output_scale_default_is_raw_train_rms_and_zero_fallback(dimension):
    rng = np.random.default_rng(17)
    x = rng.normal(size=(24, dimension + 2))
    y = x[:, :dimension] * 3
    if dimension == 6:
        y[:, 4] = x[:, 4] + 0.25
    scale, report = output_scales(x, y, dimension)
    expected = y[:, : len(scale)].copy()
    if dimension == 6:
        expected[:, 4] -= x[:, 4]
    np.testing.assert_allclose(scale, np.sqrt(np.mean(expected**2, axis=0)))
    assert not report["loss_targets_used"] and not report["validation_or_test_used"]
    zeros = np.zeros_like(y)
    x[:, 0] = 0
    scale, report = output_scales(x, zeros, dimension)
    assert scale[0] == 1 and report["degenerate_native_unit_fallback_components"] == [0]


def config(d, base="none", mode="train_rms"):
    return dict(
        schema="element_surrogate.static_magnetic_mlp.v1",
        dimension=d,
        hidden_widths=[8, 8],
        base=base,
        dtype="float64",
        device="cpu",
        threads=1,
        batch_size=32,
        inference_batch_size=32,
        learning_rate=0.002,
        loss_scales=[2e-6] * d,
        output_normalization=mode,
    )


@pytest.mark.parametrize("dimension", [4, 5, 6])
def test_normalized_model_restore_and_static_invariants(tmp_path, dimension):
    pytest.importorskip("torch")
    from element_surrogate.neural import build_adapter, load_adapter

    rng = np.random.default_rng(19)
    x = rng.normal(0, 0.002, (64, dimension + 2))
    x[:, -2:] = [1.0, 1.0]
    y = x[:, :dimension].copy()
    y[:, 0] += 0.3 * x[:, 1]
    if dimension == 6:
        y[:, 4] += 0.2 * x[:, 0] ** 2
    c = config(dimension)
    model = build_adapter(x, y, c, 7)
    initial = np.mean((model.predict(x) - y) ** 2)
    for _ in range(400):
        model.step()
    assert np.mean((model.predict(x) - y) ** 2) < initial / 2
    p = tmp_path / "model.pt"
    model.save(p)
    restored = load_adapter(p, c)
    np.testing.assert_array_equal(restored.predict(x), model.predict(x))
    if dimension > 4:
        np.testing.assert_array_equal(restored.predict(x)[:, -1], x[:, dimension - 1])
    if dimension == 6:
        shifted = x.copy()
        shifted[:, 4] += 10
        np.testing.assert_allclose(restored.predict(shifted)[:, 4] - restored.predict(x)[:, 4], 10)


def test_normalization_independent_of_loss_units_and_affine_base(tmp_path):
    torch = pytest.importorskip("torch")
    from element_surrogate.neural import build_adapter, load_adapter

    rng = np.random.default_rng(31)
    x = rng.normal(0, 0.002, (128, 6))
    x[:, -2:] = [1.0, 1.0]
    y = x[:, :4] * 1.2
    y[:, 0] += 0.3 * x[:, 1]
    a = build_adapter(x, y, config(4), 13)
    c = config(4)
    c["loss_scales"] = [1e-5] * 4
    b = build_adapter(x, y, c, 13)
    affine = build_adapter(x, y, config(4, "train_affine"), 13)
    np.testing.assert_array_equal(a.module.output_scale, b.module.output_scale)
    np.testing.assert_array_equal(a.module.output_scale, affine.module.output_scale)
    np.testing.assert_allclose(affine.predict(x), y, atol=1e-17)
    # Changing native units and their error scales leaves normalized training identical.
    units = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1.0, 1.0])
    c = config(4)
    c["loss_scales"] = (np.array(c["loss_scales"]) * units[:4]).tolist()
    converted = build_adapter(x * units, y * units[:4], c, 13)
    for _ in range(20):
        a.step()
        converted.step()
    np.testing.assert_allclose(converted.predict(x * units) / units[:4], a.predict(x), atol=2e-17)
    # Omitted key is legacy behavior and needs no new output buffer.
    legacy = config(4)
    legacy.pop("output_normalization")
    old = build_adapter(x, y, legacy, 13)
    assert "output_scale" not in old.module.state_dict()
    p = tmp_path / "legacy.pt"
    old.save(p)
    saved = torch.load(p, weights_only=True)
    saved.pop("normalization_json")
    torch.save(saved, p)
    np.testing.assert_array_equal(load_adapter(p, legacy).predict(x), old.predict(x))
    wrong = copy.deepcopy(c)
    wrong["output_normalization"] = "explicit"
    wrong["output_scales"] = [0.0] * 4
    with pytest.raises(ValueError, match="positive"):
        build_adapter(x, y, wrong, 13)
