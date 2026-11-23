"""Worked example generates all numerical payloads locally from a stated field."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


def test_manufactured_example_generates_field_model_and_portable_outputs(tmp_path):
    from element_surrogate.quadrupole_example import run

    result = run(tmp_path / "example", steps=3, seed=73)
    assert result["train_rows"] + result["validation_rows"] == 192
    assert result["updates"] == 3
    assert result["onnx_max_change"] < 1e-12
    assert np.max(result["reflection_max_changes_native"]) < 1e-12
    assert "Manufactured software example" in result["scope"]
