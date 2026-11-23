from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from element_surrogate import neural, theory
from element_surrogate.bend import PathFrame
from tests.workflow_fixture import manufactured_transport


def config(d, base, contract):
    return {
        "schema": "element_surrogate.static_magnetic_mlp.v1",
        "dimension": d,
        "hidden_widths": [8],
        "base": base,
        "theory_contract": contract,
        "dtype": "float64",
        "device": "cpu",
        "threads": 1,
        "batch_size": 16,
        "inference_batch_size": 16,
        "learning_rate": 0.001,
        "loss_scales": [0.01] * d,
    }


@pytest.mark.parametrize("d", [4, 5, 6])
def test_exact_drift_base_and_zero_residual_saved_roundtrip(tmp_path, d):
    transport, _, ranges, _ = manufactured_transport(d, gradient=0.0)
    contract = theory.assess(transport, "drift", "Manufactured zero field")
    rng = np.random.default_rng(7)
    states = rng.uniform(np.array(ranges)[:, 0], np.array(ranges)[:, 1], (16, d))
    inputs = np.column_stack([states, np.full(16, 2.0), np.ones(16)])
    targets, status, _ = transport.run(states, 2.0, 1.0)
    assert status.reached_exit.all()
    base = theory.predict_torch(torch.tensor(inputs), contract, d).numpy()
    truth = targets.copy()
    if d == 6:
        truth[:, 4] -= inputs[:, 4]
    np.testing.assert_allclose(base, truth[:, : base.shape[1]], atol=1e-11, rtol=1e-10)
    configuration = config(d, "theory_drift", contract)
    model = neural.build_adapter(inputs, targets, configuration, 3)
    np.testing.assert_allclose(model.predict(inputs), targets, atol=1e-11, rtol=1e-10)
    model.save(tmp_path / "model.pt")
    restored = neural.load_adapter(tmp_path / "model.pt", configuration)
    np.testing.assert_array_equal(model.predict(inputs), restored.predict(inputs))


def test_quad_linear_limit_and_signed_strength():
    from scipy.linalg import expm

    transport, _, ranges, settings = manufactured_transport(5)
    contract = theory.assess(
        transport,
        "paraxial_quadrupole",
        "Test declared ideal base",
        gradient_t_m=3.0,
        ranges=ranges,
        settings=settings,
    )
    state = np.array([[0.001, 0.002, -0.001, -0.002, 0.03, 2.0, 1.0]])
    for scale in (0.0, 1.0, -1.0):
        state[:, -1] = scale
        result = theory.predict_torch(torch.tensor(state), contract, 5).numpy()[0]
        k = 3.0 * scale / 2.0 / 1.03
        expected = np.concatenate(
            [
                expm(np.array([[0, 1], [-k, 0]]) * contract["length_m"]) @ state[0, :2],
                expm(np.array([[0, 1], [k, 0]]) * contract["length_m"]) @ state[0, 2:4],
            ]
        )
        np.testing.assert_allclose(result, expected, atol=1e-15)


def test_ineligible_native_geometry_and_dimension_refused():
    transport, _, _, _ = manufactured_transport()
    old = transport.exit
    shifted = PathFrame(
        old.origin_m + np.array([0.001, 0, 0]), old.tangent, old.horizontal, old.vertical
    )
    with pytest.raises(ValueError, match="aligned straight"):
        theory.assess(replace(transport, exit=shifted), "drift", "Test")
    six, _, _, _ = manufactured_transport(6)
    with pytest.raises(ValueError, match="4D/5D"):
        theory.assess(six, "paraxial_quadrupole", "Test", 3.0)


def test_scaled_matrix_kernel_and_modified_contract_refusal():
    from copy import deepcopy

    from scipy.linalg import expm

    transport, _, ranges, settings = manufactured_transport(5)
    contract = theory.assess(
        transport,
        "paraxial_quadrupole",
        "Scaled kernel fixture",
        3000.0,
        ranges=ranges,
        settings=settings,
    )
    assert contract["matrix_squarings"] > 0
    state = np.array([[0.001, 0.002, -0.001, -0.002, 0.03, 2.0, 1.0]])
    result = theory.predict_torch(torch.tensor(state), contract, 5).numpy()[0]
    k = 3000.0 / 2.0 / 1.03
    expected = np.concatenate(
        [
            expm(np.array([[0, 1], [-k, 0]]) * contract["length_m"]) @ state[0, :2],
            expm(np.array([[0, 1], [k, 0]]) * contract["length_m"]) @ state[0, 2:4],
        ]
    )
    np.testing.assert_allclose(result, expected, atol=1e-12, rtol=1e-12)
    wrong = deepcopy(contract)
    wrong["length_m"] *= 2
    with pytest.raises(ValueError, match="native span"):
        theory.validate(wrong, "paraxial_quadrupole", 5)
    wrong = deepcopy(contract)
    wrong["maximum_abs_kappa_l2"] *= 2
    with pytest.raises(ValueError, match="declared controls"):
        theory.validate(wrong, "paraxial_quadrupole", 5)
