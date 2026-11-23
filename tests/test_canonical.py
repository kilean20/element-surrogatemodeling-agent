from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from element_surrogate import canonical, neural, symplectic
from tests.workflow_fixture import manufactured_transport


def configuration():
    transport, _, ranges, settings = manufactured_transport(4, gradient=0.0)
    contract = canonical.assess(
        transport,
        ranges,
        settings,
        q_scale_m=0.01,
        p_scale=0.03,
        rationale="Manufactured zero-normal test field",
    )
    config = {
        "schema": "element_surrogate.static_magnetic_mlp.v1",
        "dimension": 4,
        "model_family": "canonical_shear",
        "canonical_contract": contract,
        "shear_pairs": 3,
        "hidden_widths": [8],
        "base": "none",
        "dtype": "float64",
        "device": "cpu",
        "threads": 1,
        "batch_size": 16,
        "inference_batch_size": 16,
        "learning_rate": 0.0001,
        "loss_scales": [0.01] * 4,
    }
    rng = np.random.default_rng(1)
    states = rng.uniform(np.array(ranges)[:, 0], np.array(ranges)[:, 1], (16, 4))
    control = settings[0]
    x = np.column_stack(
        [states, np.full(16, control["effective_rigidity_tm"]), np.full(16, control["field_scale"])]
    )
    return config, x


def test_exact_shear_jacobian_save_reload_and_native_onnx(tmp_path):
    config, x = configuration()
    adapter = neural.build_adapter(x, x[:, :4], config, 3)
    with torch.no_grad():
        for pair in adapter.module.pairs:
            for potential in pair:
                potential.amplitude.uniform_(-0.01, 0.01)
    report = symplectic.diagnose(
        lambda states: adapter.predict(np.column_stack([states, x[: len(states), 4:]])),
        lambda s, e: np.zeros(len(s)),
        x[:2, :4],
        x[0, 4],
        x[0, 5],
        q_scale_m=0.01,
        p_scale=0.03,
        relative_steps=[1e-3, 5e-4, 2.5e-4],
    )
    assert report["refinements"][-1]["max_relative_two_form_defect"] < 1e-8
    adapter.step()
    adapter.save(tmp_path / "model.pt")
    restored = neural.load_adapter(tmp_path / "model.pt", config)
    np.testing.assert_array_equal(adapter.predict(x), restored.predict(x))
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    torch.onnx.export(
        restored.module,
        torch.tensor(x[:2]),
        str(tmp_path / "model.onnx"),
        input_names=["inputs"],
        output_names=["outputs"],
        dynamic_axes={"inputs": {0: "rows"}, "outputs": {0: "rows"}},
        opset_version=17,
        dynamo=False,
    )
    session = ort.InferenceSession(str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"])
    np.testing.assert_allclose(
        session.run(None, {"inputs": x})[0], restored.predict(x), atol=1e-12, rtol=1e-12
    )


def test_nonzero_normal_field_and_reflection_average_ineligible():
    transport, _, ranges, settings = manufactured_transport(4)
    data = transport.grid.b_t.copy()
    data[..., 2] = 0.01
    assessment = canonical.assess(
        replace(transport, grid=transport.grid.with_values(data)),
        ranges,
        settings,
        q_scale_m=0.01,
        p_scale=0.03,
        rationale="Test unsupported canonical chart",
    )
    assert not assessment["eligible"]
    data[..., 2] = np.nan
    with pytest.raises(ValueError, match="non-finite field"):
        canonical.assess(
            replace(transport, grid=transport.grid.with_values(data)),
            ranges,
            settings,
            q_scale_m=0.01,
            p_scale=0.03,
            rationale="Invalid test field",
        )
    config, x = configuration()
    config["canonical_contract"] = assessment
    with pytest.raises(ValueError, match="zero-normal-field"):
        neural.build_adapter(x, x[:, :4], config, 3)
    config, x = configuration()
    config["reflection_group"] = [[1] * 6, [-1] * 4 + [1] * 2]
    with pytest.raises(ValueError):
        neural._configuration(config)


def test_full_derivative_canonical_jacobian():
    config, _ = configuration()
    model = canonical.module(config, 0.0, 1.0)
    with torch.no_grad():
        for pair in model.pairs:
            for potential in pair:
                potential.amplitude.uniform_(-0.02, 0.02)
    state = torch.tensor([0.1, 0.02, -0.1, 0.03], dtype=torch.float64, requires_grad=True)

    def mapping(v):
        q = v[[0, 2]] * 0.01
        p = v[[1, 3]] * 0.03
        slopes = p / torch.sqrt(1 - torch.sum(p * p))
        native = torch.stack([q[0], slopes[0], q[1], slopes[1], v[0] * 0 + 2.0, v[0] * 0 + 1.0])[
            None, :
        ]
        out = model(native)[0]
        s = out[[1, 3]]
        momentum = s / torch.sqrt(1 + torch.sum(s * s))
        return torch.stack([out[0] / 0.01, momentum[0] / 0.03, out[2] / 0.01, momentum[1] / 0.03])

    jac = torch.autograd.functional.jacobian(mapping, state).detach().numpy()
    j = symplectic.form(np.zeros(1), np.zeros(1), 0.01, 0.03)[0]
    np.testing.assert_allclose(jac.T @ j @ jac, j, atol=2e-14, rtol=2e-14)
