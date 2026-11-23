"""A declared static zero-field limit is enforced only on a qualified native base."""

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from element_surrogate import neural, symmetry, theory
from element_surrogate.benchmark.protocol import protocol_identity
from tests.test_equivariant_features import setup
from tests.workflow_fixture import manufactured_transport


@pytest.mark.parametrize("dimension", [4, 5, 6])
@pytest.mark.parametrize("family", ["mlp", "reflection_features"])
def test_zero_field_is_the_native_theoretical_map_after_training_and_reload(
    tmp_path, dimension, family
):
    config, x, y = setup(dimension)
    t, _, ranges, _ = manufactured_transport(dimension)
    settings = [
        {"id": "off", "effective_rigidity_tm": 2.0, "field_scale": 0.0},
        {"id": "on", "effective_rigidity_tm": 2.0, "field_scale": 1.0},
    ]
    measured = symmetry.assess(
        t,
        ranges,
        settings,
        config["reflection_group"],
        [1e-8] * dimension,
        random_probes=0,
        seed=47,
        rationale="Manufactured static zero-field and powered limits",
        effective_coupling_range=[0.0, 0.5],
    )
    config.update(
        model_family=family,
        residual_gate="coupling",
        symmetry_assessment=measured,
        symmetry_evidence_sha256=protocol_identity(measured),
    )
    kind = "drift" if dimension == 6 else "paraxial_quadrupole"
    config["base"] = "theory_drift" if dimension == 6 else "theory_quad"
    config["theory_contract"] = theory.assess(
        t,
        kind,
        "Manufactured native-span limiting case",
        ranges=ranges,
        settings=settings,
        effective_coupling_range=[0.0, 0.5],
        **({"gradient_t_m": 3.0} if dimension < 6 else {}),
    )
    model = neural.build_adapter(x, y, config, 47)
    for _ in range(5):
        model.step()
    zero = x.copy()
    zero[:, -1] = 0
    exact = theory.predict_torch(
        torch.as_tensor(zero, dtype=torch.float64), config["theory_contract"], dimension
    ).numpy()
    expected = zero[:, :dimension].copy()
    if dimension == 6:
        expected[:, 4] += exact[:, 4]
    expected[:, :4] = exact[:, :4]
    np.testing.assert_allclose(model.predict(zero), expected, atol=1e-14, rtol=1e-14)
    path = tmp_path / "gated.bin"
    model.save(path)
    np.testing.assert_array_equal(
        neural.load_adapter(path, config).predict(zero), model.predict(zero)
    )


def test_a_fitted_base_cannot_claim_an_exact_zero_field_limit():
    config, x, y = setup(base="train_affine")
    config["residual_gate"] = "coupling"
    with pytest.raises(ValueError, match="qualified native theoretical base"):
        neural.build_adapter(x, y, config, 47)
