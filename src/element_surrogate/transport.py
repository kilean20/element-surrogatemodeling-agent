"""One native-plane static-magnetic map and its empirical qualification.

Qualification is a finite-cohort refinement result. Generation is permitted only
for the identical declared box/settings/configuration and uses strict stencil
support on every query. Neither property certifies an unknown material aperture
or an absolute physical-field error.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from element_surrogate.benchmark.protocol import file_hash, protocol_identity, read_json
from element_surrogate.bend import PathFrame
from element_surrogate.contracts import StaticMagneticCodec, array_digest, field_identity
from element_surrogate.domain import SupportedField, advance
from element_surrogate.fields import FieldMap
from element_surrogate.lorentz import Status
from element_surrogate.readings import reconstruct
from element_surrogate.region import Region


@dataclass
class Transport:
    """A declared element, with fixed reference and native reporting planes."""

    grid: FieldMap
    entry: PathFrame
    exit: PathFrame
    reference_path_m: float
    reference_sha256: str
    codec: StaticMagneticCodec
    support: Region
    trajectory_region: Region
    integration: dict
    reconstruction: str = "total-degree-two"
    actual_p0_kg_m_s: float | None = None
    mass_kg: float | None = None

    def __post_init__(self):
        if not np.isfinite(self.reference_path_m) or self.reference_path_m <= 0:
            raise ValueError("positive finite reference path required")
        if len(self.reference_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.reference_sha256
        ):
            raise ValueError("reference SHA256 required")
        if self.codec.dimension == 6 and self.codec.longitudinal != "path_m":
            if (
                self.actual_p0_kg_m_s is None
                or self.mass_kg is None
                or not np.isfinite([self.actual_p0_kg_m_s, self.mass_kg]).all()
                or min(self.actual_p0_kg_m_s, self.mass_kg) <= 0
            ):
                raise ValueError("arrival time needs actual positive momentum and rest mass")
        if self.integration.get("method") not in ("fixed-rk4", "adaptive-rk4"):
            raise ValueError("explicit supported integration method required")
        if "field_scale" in self.integration or "region" in self.integration:
            raise ValueError(
                "setting scale and declared region must not be overridden by integration options"
            )
        reconstruct(self.grid, self.reconstruction)

    def description(self):
        return {
            "field": field_identity(reconstruct(self.grid, self.reconstruction)),
            "reference_sha256": self.reference_sha256,
            "entry": frame_dict(self.entry),
            "exit": frame_dict(self.exit),
            "reference_path_m": self.reference_path_m,
            "codec": {"dimension": self.codec.dimension, "longitudinal": self.codec.longitudinal},
            "support": self.support.as_dict(),
            "trajectory_region": self.trajectory_region.as_dict(),
            "integration": self.integration,
            "reconstruction": self.reconstruction,
            "actual_p0_kg_m_s": self.actual_p0_kg_m_s,
            "mass_kg": self.mass_kg,
            "native_planes": True,
            "coordinates": (
                "plane positions in m, direction ratios, optional relative arrival, delta_p/p0"
            ),
            "rigidity_interpretation": (
                "setting effective rigidity independent of actual_p0; multiplied by 1+delta_p/p0"
            ),
        }

    def run(self, states, effective_rigidity_tm, field_scale):
        """Return labels and all statuses; failed labels remain NaN.

        The supplied setting rigidity is effective p0/q. It may encode an
        amplitude error; actual momentum for time stays a separate parameter.
        Reference and native planes are never refitted for a setting.
        """
        transverse, delta = self.codec.decode(states)
        if (
            not np.isfinite([effective_rigidity_tm, field_scale]).all()
            or effective_rigidity_tm == 0
        ):
            raise ValueError("finite nonzero signed effective rigidity and finite scale required")
        supported = SupportedField(self.grid, self.support, self.reconstruction)
        result = advance(
            supported,
            transverse,
            effective_rigidity_tm * (1 + delta),
            self.entry,
            self.exit,
            {**self.integration, "field_scale": float(field_scale)},
            region=self.trajectory_region,
        )
        output = np.full_like(np.asarray(states, dtype=float), np.nan)
        ok = result.reached_exit
        if np.any(ok):
            output[ok] = self.codec.encode_exit(
                np.asarray(states)[ok],
                result.states[ok],
                result.path_length_m[ok],
                self.reference_path_m,
                actual_p0_kg_m_s=self.actual_p0_kg_m_s,
                mass_kg=self.mass_kg,
            )
            invalid = ok & ~np.isfinite(output).all(axis=1)
            result.status[invalid] = Status.NON_FINITE
            output[invalid] = np.nan
        return output, result, supported.summary()


def frame_dict(frame):
    return {
        k: getattr(frame, k).tolist() for k in ("origin_m", "tangent", "horizontal", "vertical")
    }


def frame_from_dict(value):
    return PathFrame(
        **{
            k: np.asarray(value[k], dtype=float)
            for k in ("origin_m", "tangent", "horizontal", "vertical")
        }
    )


def verify_transport_source(transport):
    """Check the supplied source bytes, including mapped members, without copying arrays."""
    from pathlib import Path

    source = transport.grid.source
    if source is None:
        return
    path = Path(source.path).resolve()
    if file_hash(path) != source.sha256:
        raise ValueError("transport source bytes changed")
    if source.format == "mapped_field":
        metadata = read_json(path)
        for item in metadata["files"].values():
            member = (path.parent / item["name"]).resolve()
            if not member.is_relative_to(path.parent) or file_hash(member) != item["sha256"]:
                raise ValueError("transport mapped-source member changed")


def _validate_ranges(transport, ranges, settings):
    box = np.asarray(ranges, dtype=float)
    if (
        box.shape != (transport.codec.dimension, 2)
        or not np.isfinite(box).all()
        or np.any(box[:, 1] < box[:, 0])
    ):
        raise ValueError("one finite increasing or constant interval per coordinate required")
    if transport.codec.dimension > 4 and box[-1, 0] <= -1:
        raise ValueError("delta_p/p0 must exceed -1")
    if not settings or len({s["id"] for s in settings}) != len(settings):
        raise ValueError("unique nonempty settings required")
    for s in settings:
        if (
            not isinstance(s["id"], str)
            or not s["id"]
            or not np.isfinite([s["effective_rigidity_tm"], s["field_scale"]]).all()
            or s["effective_rigidity_tm"] == 0
        ):
            raise ValueError("finite setting values and text ID required")
    couplings = [s["field_scale"] / s["effective_rigidity_tm"] for s in settings]
    if not np.isfinite(couplings).all() or len(set(couplings)) != len(couplings):
        raise ValueError("setting groups must have distinct finite amplitude/rigidity couplings")
    return box


def scope_description(transport, ranges, settings, effective_coupling_range=None):
    box = _validate_ranges(transport, ranges, settings)
    physics_settings = [
        {k: s[k] for k in ("id", "effective_rigidity_tm", "field_scale")} for s in settings
    ]
    interval = None
    if effective_coupling_range is not None:
        limits = np.asarray(effective_coupling_range, dtype=float)
        values = [s["field_scale"] / s["effective_rigidity_tm"] for s in settings]
        if (
            limits.shape != (2,)
            or not np.isfinite(limits).all()
            or limits.tolist() != [min(values), max(values)]
        ):
            raise ValueError("declared coupling range must match the tested endpoint settings")
        interval = limits.tolist()
    return {
        "transport": transport.description(),
        "ranges": box.tolist(),
        "settings": physics_settings,
        "effective_coupling_range": interval,
    }


def qualify_transport(
    transport,
    ranges,
    settings,
    refinements,
    *,
    random_probes,
    seed,
    component_limits,
    path_limit_m,
    effective_coupling_range=None,
):
    """Compare three or more declared resolutions on corners and random probes.

    Both last adjacent changes must meet supplied component/path limits and all
    statuses must agree. Report each setting, including losses. This is empirical
    support for a finite cohort, never a mathematical global-error bound.
    """
    box = _validate_ranges(transport, ranges, settings)
    d = transport.codec.dimension
    limits = np.asarray(component_limits, dtype=float)
    if (
        limits.shape != (d,)
        or not np.isfinite(limits).all()
        or np.any(limits <= 0)
        or not np.isfinite(path_limit_m)
        or path_limit_m <= 0
    ):
        raise ValueError("positive component and path accuracy limits required")
    if (
        type(random_probes) is not int
        or random_probes < 0
        or type(seed) is not int
        or seed < 0
        or len(refinements) < 3
    ):
        raise ValueError(
            "nonnegative probe count and at least three refinement configurations required"
        )
    _check_refinements(refinements)
    if transport.integration != refinements[-1]:
        raise ValueError("generation must use the finest qualified configuration")
    verify_transport_source(transport)
    initial_scope = scope_description(transport, ranges, settings, effective_coupling_range)
    corners = np.array(np.meshgrid(*[[lo, hi] for lo, hi in box], indexing="ij")).reshape(d, -1).T
    random = np.random.default_rng(seed).uniform(box[:, 0], box[:, 1], (random_probes, d))
    states = np.unique(np.vstack([corners, random, box.mean(axis=1)[None]]), axis=0)
    reports = []
    for setting in settings:
        results = []
        for integration in refinements:
            model = replace(transport, integration=integration)
            outputs, result, audit = model.run(
                states, setting["effective_rigidity_tm"], setting["field_scale"]
            )
            results.append((outputs, result, audit))
        common = np.logical_and.reduce([r[1].reached_exit for r in results])
        status_changed = int(
            np.any(np.stack([r[1].status for r in results]) != results[0][1].status, axis=0).sum()
        )
        changes = []
        for a, b in zip(results[:-1], results[1:], strict=True):
            error = np.max(np.abs(a[0][common] - b[0][common]), axis=0) if common.any() else None
            path_error = (
                float(np.max(np.abs(a[1].path_length_m[common] - b[1].path_length_m[common])))
                if common.any()
                else None
            )
            changes.append(
                {
                    "max_component_change": None if error is None else error.tolist(),
                    "max_path_change_m": path_error,
                    "within_limits": bool(
                        error is not None and np.all(error <= limits) and path_error <= path_limit_m
                    ),
                }
            )
        reports.append(
            {
                "setting_id": setting["id"],
                "common_survivors": int(common.sum()),
                "status_changes": status_changed,
                "status_counts": [r[1].summary() for r in results],
                "changes": changes,
                "stencil_audits": [r[2] for r in results],
                "accepted": status_changed == 0 and all(c["within_limits"] for c in changes[-2:]),
            }
        )
    scope = scope_description(transport, ranges, settings, effective_coupling_range)
    verify_transport_source(transport)
    if scope != initial_scope:
        raise ValueError("transport or field values changed during qualification")
    from element_surrogate.runs import implementation_identity, sha256_json

    return {
        "schema": "element_surrogate.transport_qualification.v1",
        "scope": scope,
        "scope_sha256": protocol_identity(scope),
        "implementation_sha256": sha256_json(implementation_identity()),
        "probe_states_sha256": array_digest(states),
        "probe_design": {
            "corners": True,
            "centre": True,
            "random_probes": random_probes,
            "seed": seed,
            "numpy": np.__version__,
        },
        "probe_count_per_setting": len(states),
        "refinements": refinements,
        "component_limits": limits.tolist(),
        "path_limit_m": path_limit_m,
        "reports": reports,
        "accepted": all(r["accepted"] for r in reports),
        "scope_limit": (
            "finite-cohort empirical integration refinement; physical field "
            "accuracy and material aperture unverified"
        ),
    }


def _check_refinements(refinements):
    """Refine the controlling resolutions; reject coarsening disguised as refinement."""
    methods = {r.get("method") for r in refinements}
    if len(methods) != 1 or not methods <= {"fixed-rk4", "adaptive-rk4"}:
        raise ValueError("one supported refinement method required")
    keys = (
        ("step_m",) if "fixed-rk4" in methods else ("position_tolerance_m", "direction_tolerance")
    )
    for key in keys:
        values = np.array([r[key] for r in refinements], float)
        if not np.isfinite(values).all() or np.any(values <= 0) or np.any(np.diff(values) >= 0):
            raise ValueError("strictly decreasing positive refinement controls required")
    allowed = set(keys) | {"max_step_m", "max_turn_rad"}
    fixed = {k: v for k, v in refinements[0].items() if k not in allowed}
    if any({k: v for k, v in r.items() if k not in allowed} != fixed for r in refinements):
        raise ValueError("non-refinement controls changed between resolutions")
    for key in ("max_step_m", "max_turn_rad"):
        if any(key in r for r in refinements):
            values = np.array([r[key] for r in refinements], float)
            if not np.isfinite(values).all() or np.any(values <= 0) or np.any(np.diff(values) > 0):
                raise ValueError("step/turn caps may not increase during refinement")
