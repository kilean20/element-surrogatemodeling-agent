"""Reference orbits through a magnet (knowledge entry ``reference-orbit-in-bending-magnets``).

The reference orbit is the design particle's actual path through the field: the particle
that starts on the entry plane with zero transverse state. It is found in one of two ways,
which must not be confused:

* ``trace_reference``: the field strength is fixed and the design particle is tracked;
* ``fit_reference``: the field is scaled until the design particle leaves the exit plane
  with zero horizontal slope, as a simulator that fits its bend for a symmetric exit does;
  every other exit coordinate's closure is then checked, not assumed.

A particle with another rigidity is tracked through the same scaled field: rigidity never
triggers a refit. Along the orbit, the transverse frame at path length ``s`` has the orbit's
tangent, horizontal = up x tangent and vertical = tangent x horizontal. Reporting a
particle in the frame of the reference at a station means tracking it to that station's
plane (``lorentz.track`` with ``orbit.frame_at(s)`` as the exit), not sampling it at equal
path length.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import brentq

from element_surrogate.adaptive import track_adaptive
from element_surrogate.bend import PathFrame
from element_surrogate.contracts import array_digest, field_identity
from element_surrogate.lorentz import Field, track
from element_surrogate.readings import tracking_field

_UP = np.array([0.0, 1.0, 0.0])


def _advance(field_map, *args, step_m, integration_options=None, **kwargs):
    if step_m is not None:
        if integration_options:
            raise ValueError("adaptive integration options require step_m=None")
        return track(field_map, *args, step_m=step_m, **kwargs)
    if integration_options is None:
        raise ValueError("adaptive reference tracing requires explicit integration tolerances")
    return track_adaptive(field_map, *args, **integration_options, **kwargs)


@dataclass
class ReferenceOrbit:
    """The traced design particle: path lengths ``s_m`` and laboratory states ``(r, t)``,
    the rigidity and field scale it was traced with, how it was found, and its state on the
    nominal exit plane."""

    s_m: np.ndarray
    lab_states: np.ndarray
    rigidity_tm: float
    field_scale: float
    method: str
    exit_state: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    _curve: CubicHermiteSpline = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.s_m = np.asarray(self.s_m, dtype=np.float64)
        self.lab_states = np.asarray(self.lab_states, dtype=np.float64)
        if self.s_m.ndim != 1 or self.s_m.size < 2 or self.lab_states.shape != (self.s_m.size, 6):
            raise ValueError("an orbit needs path lengths and matching (r, t) states")
        if self.s_m[0] != 0.0 or not np.all(np.diff(self.s_m) > 0):
            raise ValueError("path lengths must start at 0 and increase")
        if not np.isfinite(self.s_m).all() or not np.isfinite(self.lab_states).all():
            raise ValueError("reference arrays must be finite")
        if not np.isfinite([self.rigidity_tm, self.field_scale]).all() or self.rigidity_tm == 0:
            raise ValueError("finite reference configuration required")
        directions = self.lab_states[:, 3:]
        self._curve = CubicHermiteSpline(self.s_m, self.lab_states[:, :3], directions, axis=0)

    @property
    def length_m(self) -> float:
        return float(self.s_m[-1])

    def save(self, path: str | Path) -> None:
        """Save numeric states and identities without pickle, in the user's project."""
        if "field" not in self.metadata:
            raise ValueError("reference has no verified field/reconstruction identity")
        meta = {
            **self.metadata,
            "schema": "element_surrogate.reference.v1",
            "rigidity_tm": self.rigidity_tm,
            "field_scale": self.field_scale,
            "method": self.method,
            "exit_state": self.exit_state.tolist(),
            "arrays_sha256": array_digest(self.s_m, self.lab_states),
        }
        with Path(path).open("wb") as handle:
            np.savez(
                handle,
                s_m=self.s_m,
                lab_states=self.lab_states,
                metadata_json=np.array(json.dumps(meta, sort_keys=True)),
            )

    @classmethod
    def load(cls, path, field_map, *, expected_rigidity_tm, expected_field_scale):
        """Refuse a changed field, reconstruction, scale or reference configuration."""
        field_map = tracking_field(field_map)
        with np.load(path, allow_pickle=False) as stored:
            s, states = stored["s_m"], stored["lab_states"]
            meta = json.loads(str(stored["metadata_json"]))
        if meta.get("schema") != "element_surrogate.reference.v1" or meta.get(
            "arrays_sha256"
        ) != array_digest(s, states):
            raise ValueError("invalid or changed reference arrays")
        if (
            meta["field"] != field_identity(field_map)
            or meta["rigidity_tm"] != expected_rigidity_tm
            or meta["field_scale"] != expected_field_scale
        ):
            raise ValueError("reference field/reconstruction/configuration mismatch")
        return cls(
            s,
            states,
            meta["rigidity_tm"],
            meta["field_scale"],
            meta["method"],
            np.asarray(meta["exit_state"]),
            meta,
        )

    def frame_at(self, s_m: float) -> PathFrame:
        """The orbit's point and transverse frame at path length ``s_m``."""
        if not 0.0 <= s_m <= self.length_m:
            raise ValueError("the station lies outside the orbit")
        point = np.asarray(self._curve(s_m))
        tangent = np.asarray(self._curve.derivative()(s_m))
        tangent = tangent / np.linalg.norm(tangent)
        horizontal = np.cross(_UP, tangent)
        if np.linalg.norm(horizontal) < 1e-8:
            raise ValueError("a vertical tangent has no horizontal axis")
        horizontal /= np.linalg.norm(horizontal)
        return PathFrame(point, tangent, horizontal, np.cross(tangent, horizontal))

    def summary(self, nominal: Any | None = None) -> dict[str, Any]:
        """Length, exit state and, given a nominal path (``bend.NominalBend``), the orbit's
        largest horizontal and vertical departure from it."""
        out: dict[str, Any] = {
            "method": self.method,
            "rigidity_tm": self.rigidity_tm,
            "field_scale": self.field_scale,
            "length_m": self.length_m,
            "exit_state_on_nominal_plane": {
                "x_m": float(self.exit_state[0]),
                "xp_rad": float(self.exit_state[1]),
                "y_m": float(self.exit_state[2]),
                "yp_rad": float(self.exit_state[3]),
            },
        }
        if nominal is not None:
            offsets = nominal.path_coordinates(self.lab_states[:, :3])
            worst = int(np.argmax(np.abs(offsets[:, 1])))
            out["departure_from_nominal_path"] = {
                "max_abs_horizontal_m": float(np.abs(offsets[worst, 1])),
                "at_nominal_s_m": float(offsets[worst, 0]),
                "max_abs_vertical_m": float(np.abs(offsets[:, 2]).max()),
            }
        return out


def trace_reference(
    field_map: Field,
    rigidity_tm: float,
    entry: PathFrame,
    exit: PathFrame,
    *,
    step_m: float | None,
    field_scale: float = 1.0,
    max_path_m: float | None = None,
    integration_options: dict | None = None,
    plane_tolerance_m: float | None = None,
) -> ReferenceOrbit:
    """The design particle tracked at a fixed field strength."""
    field_map = tracking_field(field_map)
    samples: list[tuple[float, np.ndarray]] = []
    result = _advance(
        field_map,
        np.zeros((1, 4)),
        rigidity_tm,
        entry,
        exit,
        step_m=step_m,
        integration_options=integration_options,
        field_scale=field_scale,
        max_path_m=max_path_m,
        **({} if plane_tolerance_m is None else {"plane_tolerance_m": plane_tolerance_m}),
        observer=lambda s, u: samples.append((s, u)),
    )
    if not result.reached_exit[0]:
        raise ValueError(f"the design particle did not reach the exit: {result.summary()}")
    s = np.array([v[0] for v in samples])
    states = np.array([v[1] for v in samples])
    keep = np.r_[True, np.diff(s) > 1e-14]
    return ReferenceOrbit(
        s[keep],
        states[keep],
        float(rigidity_tm),
        float(field_scale),
        "traced at a fixed field strength",
        result.states[0],
        {
            "field": field_identity(field_map),
            "frame_policy": "upright-up-cross-tangent",
            "input_coordinates": "plane-position-m,direction-ratios",
            "entry_frame": {
                k: getattr(entry, k).tolist()
                for k in ("origin_m", "tangent", "horizontal", "vertical")
            },
            "nominal_exit_frame": {
                k: getattr(exit, k).tolist()
                for k in ("origin_m", "tangent", "horizontal", "vertical")
            },
            "integration": {"step_m": step_m, "adaptive": integration_options},
            "explicit_plane_tolerance_m": plane_tolerance_m,
        },
    )


def fit_reference(
    field_map: Field,
    rigidity_tm: float,
    entry: PathFrame,
    exit: PathFrame,
    *,
    scale_bracket: tuple[float, float],
    step_m: float | None,
    position_tolerance_m: float,
    slope_tolerance_rad: float,
    max_path_m: float | None = None,
    integration_options: dict | None = None,
) -> ReferenceOrbit:
    """The field scale, within ``scale_bracket``, at which the design particle leaves the exit
    plane with zero horizontal slope, and the orbit traced at it. The other exit coordinates
    (x, y, y') must close within the tolerances; otherwise the fit is refused, because a
    symmetric exit was assumed and not found."""

    field_map = tracking_field(field_map)

    def slope(scale: float) -> float:
        result = _advance(
            field_map,
            np.zeros((1, 4)),
            rigidity_tm,
            entry,
            exit,
            step_m=step_m,
            integration_options=integration_options,
            field_scale=scale,
            max_path_m=max_path_m,
        )
        if not result.reached_exit[0]:
            stopped = [name for name, count in result.summary().items() if count]
            raise ValueError(f"at field scale {scale} the design particle stops: {stopped}")
        return float(result.states[0, 1])

    scale = brentq(slope, *scale_bracket, xtol=1e-13)
    orbit = trace_reference(
        field_map,
        rigidity_tm,
        entry,
        exit,
        step_m=step_m,
        field_scale=scale,
        max_path_m=max_path_m,
        integration_options=integration_options,
    )
    x, xp, y, yp = orbit.exit_state
    if max(abs(x), abs(y)) > position_tolerance_m or max(abs(xp), abs(yp)) > slope_tolerance_rad:
        raise ValueError(
            f"the horizontal exit slope was fitted but the exit does not close: {orbit.exit_state}"
        )
    orbit.method = (
        "field scale fitted for zero horizontal exit slope; position and vertical closure checked"
    )
    return orbit
