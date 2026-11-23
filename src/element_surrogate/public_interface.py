"""Unit/control view of a released native map; no orbit or placement transform."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from element_surrogate.benchmark.protocol import file_hash, read_json, write_json
from element_surrogate.export import ElementSurrogate
from element_surrogate.native_scope import validate as validate_native_scope

CONTROLS = ("effective_rigidity_tm", "field_scale")
UNITS = {
    "m": ("length", 1.0),
    "mm": ("length", 1e-3),
    "um": ("length", 1e-6),
    "rad": ("angle", 1.0),
    "mrad": ("angle", 1e-3),
    "urad": ("angle", 1e-6),
    "s": ("time", 1.0),
    "ns": ("time", 1e-9),
    "ps": ("time", 1e-12),
    "1": ("dimensionless", 1.0),
}


def _validate(native, request):
    allowed = {"state_units", "fixed_controls", "default_controls", "control_bounds"}
    if not isinstance(request, dict) or set(request) - allowed:
        raise ValueError("public request needs state_units and explicit control_bounds")
    units = request.get("state_units", [])
    coordinates = native["coordinates"]
    if not isinstance(units, list) or len(units) != len(coordinates):
        raise ValueError("one public unit per native coordinate required")
    factors = []
    for unit, coordinate in zip(units, coordinates, strict=True):
        target = coordinate["unit"]
        if (
            not isinstance(unit, str)
            or unit not in UNITS
            or target not in UNITS
            or UNITS[unit][0] != UNITS[target][0]
        ):
            raise ValueError("incompatible unit; coordinate definition changes need another map")
        factors.append(UNITS[unit][1] / UNITS[target][1])
    fixed, defaults = request.get("fixed_controls", {}), request.get("default_controls", {})
    bounds = request.get("control_bounds", {})
    if not all(isinstance(value, dict) for value in (fixed, defaults, bounds)):
        raise ValueError("control declarations must be objects")
    if set(bounds) != set(CONTROLS) or (set(fixed) | set(defaults)) - set(CONTROLS):
        raise ValueError("bounds for both effective controls and no unknown controls required")
    if set(fixed) & set(defaults):
        raise ValueError("fixed and default controls must be disjoint")
    for key in CONTROLS:
        box = np.asarray(bounds[key], dtype=float)
        if box.shape != (2,) or not np.isfinite(box).all() or box[0] > box[1]:
            raise ValueError("finite ordered control bounds required")
        for declaration in (fixed, defaults):
            if key in declaration:
                value = np.asarray(declaration[key], dtype=float)
                if value.shape != () or not np.isfinite(value) or not box[0] <= value <= box[1]:
                    raise ValueError("fixed/default value outside control bounds")
    return np.asarray(factors)


def export_interface(native_contract_path, request_path, output_path):
    """Bind explicit units/raw bounds to an immutable native contract."""
    contract = read_json(native_contract_path)
    if contract.get("schema") != "element_surrogate.native_model.v1":
        raise ValueError("native model contract required")
    validate_native_scope(contract.get("physics", {}), len(contract["coordinates"]))
    request = read_json(request_path)
    _validate(contract, request)
    manifest = {
        "schema": "element_surrogate.public_interface.v1",
        "native_contract_sha256": file_hash(native_contract_path),
        "request": request,
        "scope": "compatible units and effective controls; native physics/reference unchanged",
    }
    write_json(output_path, manifest, exclusive=True)
    return {
        "interface_path": str(Path(output_path).resolve()),
        "native_contract_sha256": manifest["native_contract_sha256"],
    }


class PublicElementSurrogate:
    """A state-only convenience API with explicit effective control arguments.

    These controls change magnetic coupling. They do not redefine charge,
    reference frames, actual beam momentum, energy or arrival coordinates.
    """

    def __init__(self, directory, interface_path, *, backend="onnx"):
        self.native = ElementSurrogate(directory, backend=backend)
        manifest = read_json(interface_path)
        if manifest.get("schema") != "element_surrogate.public_interface.v1" or manifest[
            "native_contract_sha256"
        ] != file_hash(Path(directory) / "contract.json"):
            raise ValueError("public interface does not match native contract")
        self.request = manifest["request"]
        self.factors = _validate(self.native.contract, self.request)

    def _inputs(self, states, controls):
        states = np.asarray(states, dtype=float)
        if (
            states.ndim != 2
            or states.shape[1] != len(self.factors)
            or not np.isfinite(states).all()
        ):
            raise ValueError("finite state matrix in the declared public units required")
        controls = {} if controls is None else controls
        if not isinstance(controls, dict) or set(controls) - set(CONTROLS):
            raise ValueError("unknown effective control")
        fixed = self.request.get("fixed_controls", {})
        if set(controls) & set(fixed):
            raise ValueError("fixed controls cannot be overridden")
        values = {**self.request.get("default_controls", {}), **fixed, **controls}
        if set(values) != set(CONTROLS):
            raise ValueError("supply every non-fixed effective control without a default")
        columns, inside = [], np.ones(len(states), dtype=bool)
        for key in CONTROLS:
            column = np.asarray(values[key], dtype=float)
            if column.ndim == 0:
                column = np.full(len(states), float(column))
            if column.shape != (len(states),) or not np.isfinite(column).all():
                raise ValueError("effective control must be finite scalar or one value per row")
            low, high = self.request["control_bounds"][key]
            inside &= (column >= low) & (column <= high)
            columns.append(column)
        return np.column_stack([states * self.factors, *columns]), inside

    def predict(self, states, controls=None, *, numerical_support_established=False):
        inputs, inside = self._inputs(states, controls)
        if not inside.all():
            raise ValueError("effective control outside the public raw-control bounds")
        return (
            self.native.predict(inputs, numerical_support_established=numerical_support_established)
            / self.factors
        )

    def predict_checked(self, states, transport, controls=None):
        inputs, inside = self._inputs(states, controls)
        prediction = np.full((len(inputs), len(self.factors)), np.nan)
        status = np.full(len(inputs), -1, dtype=int)
        if inside.any():
            prediction[inside], status[inside] = self.native.predict_checked(
                inputs[inside], transport
            )
        return prediction / self.factors, status
