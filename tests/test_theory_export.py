import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from element_surrogate import neural, theory
from tests.test_theory import config
from tests.workflow_fixture import manufactured_transport


@pytest.mark.parametrize(
    "kind,d,arrival", [("drift", 6, "time_s"), ("paraxial_quadrupole", 5, "path_m")]
)
def test_theory_base_export_and_strength_derivative_at_zero(tmp_path, kind, d, arrival):
    transport, _, ranges, settings = manufactured_transport(d, longitudinal=arrival, gradient=0.0)
    contract = theory.assess(
        transport,
        kind,
        "Synthetic portable base test",
        3.0 if kind == "paraxial_quadrupole" else None,
        ranges=ranges,
        settings=settings,
    )
    states = np.tile(np.mean(ranges, axis=1), (3, 1))
    states[:, 0] = 0.001
    states[:, 1] = 0.002
    x = np.column_stack([states, np.full(3, 2.0), [-1.0, 0.0, 1.0]])
    y = theory.predict_torch(torch.tensor(x), contract, d).numpy()
    if d == 6:
        y[:, 4] += x[:, 4]
    if d > 4:
        y = np.column_stack([y, x[:, d - 1]])
    configuration = config(d, "theory_drift" if kind == "drift" else "theory_quad", contract)
    model = neural.build_adapter(x, y, configuration, 3)
    check = torch.tensor(x, requires_grad=True)
    model.module(check).sum().backward()
    assert torch.isfinite(check.grad).all()
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    torch.onnx.export(
        model.module,
        torch.tensor(x),
        str(tmp_path / "model.onnx"),
        input_names=["inputs"],
        output_names=["outputs"],
        opset_version=17,
        dynamo=False,
    )
    session = ort.InferenceSession(str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"])
    np.testing.assert_allclose(
        session.run(None, {"inputs": x})[0], model.predict(x), atol=1e-12, rtol=1e-12
    )
