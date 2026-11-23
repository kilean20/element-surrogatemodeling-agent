"""Public conversions compose with the native API instead of duplicating physics."""

import numpy as np
import pytest

from element_surrogate import public_interface as public
from element_surrogate.benchmark.protocol import read_json, write_json


def fixture(tmp_path, monkeypatch, units):
    contract = {
        "schema": "element_surrogate.native_model.v1",
        "coordinates": [{"unit": unit} for unit in units],
        "physics": {
            "qualified_ranges": [[-2, 2] for _ in units],
            "qualified_settings": [{"effective_rigidity_tm": 2.0, "field_scale": 1.0}],
            "effective_coupling_range": [0.45, 0.55],
        },
    }
    write_json(tmp_path / "contract.json", contract)

    class Native:
        def __init__(self, directory, backend):
            self.contract = read_json(directory / "contract.json")
            self.last = None

        def predict(self, inputs, numerical_support_established=False):
            if not numerical_support_established:
                raise ValueError("support required")
            self.last = inputs
            return inputs[:, : len(units)] * 2

        def predict_checked(self, inputs, transport):
            return self.predict(inputs, True), np.zeros(len(inputs), dtype=int)

    monkeypatch.setattr(public, "ElementSurrogate", Native)
    request = {
        "state_units": ["mm", "mrad", "mm", "mrad"]
        + (
            ["ns", "1"]
            if len(units) == 6 and units[4] == "s"
            else ["mm", "1"]
            if len(units) == 6
            else ["1"]
            if len(units) == 5
            else []
        ),
        "fixed_controls": {"effective_rigidity_tm": 2.0},
        "default_controls": {"field_scale": 1.0},
        "control_bounds": {"effective_rigidity_tm": [2, 2], "field_scale": [0.9, 1.1]},
    }
    write_json(tmp_path / "request.json", request)
    public.export_interface(
        tmp_path / "contract.json", tmp_path / "request.json", tmp_path / "public.json"
    )
    return public.PublicElementSurrogate(tmp_path, tmp_path / "public.json"), request


@pytest.mark.parametrize(
    "units",
    [
        ["m", "rad", "m", "rad"],
        ["m", "rad", "m", "rad", "1"],
        ["m", "rad", "m", "rad", "s", "1"],
        ["m", "rad", "m", "rad", "m", "1"],
    ],
)
def test_conversion_delegates_native_and_preserves_controls(tmp_path, monkeypatch, units):
    model, _ = fixture(tmp_path, monkeypatch, units)
    states = np.ones((3, len(units)))
    with pytest.raises(ValueError, match="support"):
        model.predict(states)
    np.testing.assert_allclose(
        model.predict(states, numerical_support_established=True), states * 2
    )
    np.testing.assert_allclose(model.native.last[:, : len(units)], states * model.factors)
    np.testing.assert_allclose(model.native.last[:, -2:], [[2, 1]] * 3)
    assert model.predict(np.empty((0, len(units))), numerical_support_established=True).shape == (
        0,
        len(units),
    )


def test_raw_controls_refuse_even_if_ratio_could_match_native(tmp_path, monkeypatch):
    model, _ = fixture(tmp_path, monkeypatch, ["m", "rad", "m", "rad"])
    with pytest.raises(ValueError, match="fixed"):
        model.predict(np.zeros((1, 4)), {"effective_rigidity_tm": 4})
    with pytest.raises(ValueError, match="bounds"):
        model.predict(np.zeros((1, 4)), {"field_scale": 2}, numerical_support_established=True)
    prediction, status = model.predict_checked(np.zeros((2, 4)), None, {"field_scale": [1, 2]})
    assert status.tolist() == [0, -1]
    assert np.isnan(prediction[1]).all()
    with pytest.raises(ValueError, match="unknown"):
        model.predict(np.zeros((1, 4)), {"actual_p0": 1})


def test_manifest_binding_and_incompatible_units(tmp_path, monkeypatch):
    model, request = fixture(tmp_path, monkeypatch, ["m", "rad", "m", "rad", "s", "1"])
    request["state_units"][4] = "mm"
    with pytest.raises(ValueError, match="incompatible"):
        public._validate(model.native.contract, request)
    write_json(tmp_path / "contract.json", {**model.native.contract, "changed": True})
    with pytest.raises(ValueError, match="match"):
        public.PublicElementSurrogate(tmp_path, tmp_path / "public.json")
