"""Error-controlled path-length tracking, independent of any facility step size.

Two half RK4 steps are accepted only when their difference from one full step
fits the component tolerances. The raw difference is used conservatively, not
divided by 15: a reconstructed field may jump at stencil switches. Grid/switch
and curvature limits supplement this estimator; tolerance refinement remains
necessary. This is a transparent CPU reference, not a high-throughput generator.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from element_surrogate.bend import PathFrame
from element_surrogate.lorentz import Field, Observer, Status, TrackResult, to_lab, to_plane
from element_surrogate.readings import tracking_field
from element_surrogate.region import Region


class _Stop(Exception):
    def __init__(self, status: Status):
        self.status = status


def _grid_limit(field: Field, r: np.ndarray, t: np.ndarray) -> float:
    grid: Any = getattr(field, "field", field)
    if not hasattr(grid, "x_m"):
        return math.inf
    limit = math.inf
    for k, axis in enumerate((grid.x_m, grid.y_m, grid.z_m)):
        if abs(t[k]) < 1e-15:
            continue
        limit = min(limit, 0.5 * float(np.min(np.diff(axis))) / abs(t[k]))
        nearest = hasattr(field, "describe") and field.describe().get("centre") == "nearest"
        switches = (axis[:-1] + axis[1:]) * 0.5 if nearest else axis
        distances = (switches - r[k]) / t[k]
        # At a face, advance into the next stencil. The local error check still
        # handles its jump; this cap alone is not an exact curved-path event solve.
        ahead = distances[distances > 128 * np.finfo(float).eps * max(1.0, abs(r[k]))]
        if ahead.size:
            limit = min(limit, float(ahead.min()))
    return limit


def track_adaptive(
    field: Field,
    states: np.ndarray,
    rigidity_tm: float | np.ndarray,
    entry: PathFrame,
    exit: PathFrame,
    *,
    position_tolerance_m: float,
    direction_tolerance: float,
    relative_tolerance: float = 0.0,
    max_step_m: float | None = None,
    max_turn_rad: float = 0.1,
    field_scale: float = 1.0,
    max_path_m: float | None = None,
    plane_tolerance_m: float = 1e-11,
    max_steps: int = 200000,
    region: Region | None = None,
    observer: Observer | None = None,
) -> TrackResult:
    """Track with per-particle adaptive steps and explicit numerical stop statuses.

    Position and direction tolerances control local full/half-step discrepancies,
    not certified final errors. Tighten both and caps to establish convergence.
    No renormalization hides direction-norm drift. A rejected stage outside
    support/region is retried at a smaller step before declaring a loss.
    """
    field = tracking_field(field)
    s0 = np.asarray(states, dtype=float)
    if s0.ndim != 2 or s0.shape[1] != 4 or not np.isfinite(s0).all():
        raise ValueError("finite states of shape (N,4) required")
    n = len(s0)
    brho = np.broadcast_to(np.asarray(rigidity_tm, dtype=float), (n,))
    span = float(np.linalg.norm(exit.origin_m - entry.origin_m))
    budget = 10 * span if max_path_m is None else max_path_m
    cap = budget / 8 if max_step_m is None else max_step_m
    positive = [
        position_tolerance_m,
        direction_tolerance,
        cap,
        max_turn_rad,
        plane_tolerance_m,
        budget,
    ]
    if (
        not np.isfinite(positive).all()
        or min(positive) <= 0
        or not math.isfinite(relative_tolerance)
        or relative_tolerance < 0
        or not math.isfinite(field_scale)
        or not np.isfinite(brho).all()
        or np.any(brho == 0)
        or max_steps < 1
    ):
        raise ValueError("invalid tolerances, budgets, scale or signed rigidity")
    if observer is not None and n != 1:
        raise ValueError("an observer follows one particle")
    r0, t0 = to_lab(entry, s0)
    out = np.full((n, 4), np.nan)
    statuses = np.full(n, int(Status.STEP_LIMIT), dtype=np.int8)
    path = np.zeros(n)
    steps = np.zeros(n, dtype=int)
    norm_error = np.zeros(n)
    rejected = np.zeros(n, dtype=int)
    min_used, max_used = math.inf, 0.0
    for i in range(n):
        v = np.r_[r0[i], t0[i]]
        h = cap
        if observer is not None:
            observer(0.0, v.copy())

        def rhs(w, particle_rigidity=brho[i]):
            if not np.isfinite(w).all():
                raise _Stop(Status.NON_FINITE)
            if not field.contains(w[None, :3])[0]:
                raise _Stop(Status.OUTSIDE_FIELD)
            if region is not None and not region.contains(w[None, :3])[0]:
                raise _Stop(Status.OUTSIDE_REGION)
            b = field.sample(w[None, :3])[0] * field_scale
            if not np.isfinite(b).all():
                raise _Stop(Status.NON_FINITE)
            return np.r_[w[3:], np.cross(w[3:], b) / particle_rigidity]

        def rk(w, ds):
            a = rhs(w)
            b = rhs(w + 0.5 * ds * a)
            c = rhs(w + 0.5 * ds * b)
            d = rhs(w + ds * c)
            result = w + ds * (a + 2 * b + 2 * c + d) / 6
            rhs(result)  # endpoints must also be supported
            return result

        for _ in range(max_steps):
            try:
                initial_rhs = rhs(v)
            except _Stop as stop:
                statuses[i] = stop.status
                break
            distance = float((v[:3] - exit.origin_m) @ exit.tangent)
            along = float(v[3:] @ exit.tangent)
            if abs(distance) <= plane_tolerance_m:
                if along <= 0:
                    statuses[i] = Status.WRONG_EXIT_DIRECTION
                    break
                ds = -distance / along
                corrected = v + ds * initial_rhs
                try:
                    rhs(corrected)
                except _Stop as stop:
                    statuses[i] = stop.status
                    break
                v = corrected
                path[i] += ds
                out[i] = to_plane(exit, v[None, :3], v[None, 3:])[0]
                statuses[i] = Status.REACHED_EXIT
                if observer is not None:
                    observer(float(path[i]), v.copy())
                break
            if distance > plane_tolerance_m:
                statuses[i] = Status.WRONG_EXIT_DIRECTION
                break
            if path[i] >= budget:
                statuses[i] = Status.PATH_LIMIT
                break
            curvature = float(np.linalg.norm(initial_rhs[3:]))
            h = min(
                h,
                cap,
                budget - path[i],
                _grid_limit(field, v[:3], v[3:]),
                max_turn_rad / curvature if curvature else math.inf,
                0.9 * abs(distance) / np.linalg.norm(v[3:]),
            )
            minimum = 64 * np.finfo(float).eps * max(1.0, span, float(np.linalg.norm(v[:3])))
            if h < minimum:
                statuses[i] = Status.STEP_LIMIT
                break
            try:
                coarse = rk(v, h)
                fine = rk(rk(v, h / 2), h / 2)
            except _Stop as stop:
                rejected[i] += 1
                h *= 0.5
                if h < minimum or stop.status == Status.NON_FINITE:
                    statuses[i] = stop.status
                    break
                continue
            scale = np.r_[np.full(3, position_tolerance_m), np.full(3, direction_tolerance)]
            # Use displacement, not absolute lab origin, in relative position scales.
            relative = np.r_[np.abs(fine[:3] - v[:3]), np.maximum(np.abs(v[3:]), np.abs(fine[3:]))]
            error = float(np.max(np.abs(fine - coarse) / (scale + relative_tolerance * relative)))
            factor = 2.0 if error == 0 else float(np.clip(0.9 * error ** (-0.2), 0.2, 2.0))
            if error <= 1:
                v = fine
                path[i] += h
                steps[i] += 1
                norm_error[i] = max(norm_error[i], abs(np.linalg.norm(v[3:]) - 1))
                min_used, max_used = min(min_used, h), max(max_used, h)
                if observer is not None:
                    observer(float(path[i]), v.copy())
            else:
                rejected[i] += 1
            h *= factor
    return TrackResult(
        out,
        statuses,
        path,
        steps,
        norm_error,
        {
            "method": "RK4-step-doubling-v1",
            "rejected_trials": int(rejected.sum()),
            "min_accepted_step_m": min_used if math.isfinite(min_used) else None,
            "max_accepted_step_m": max_used,
            "position_tolerance_m": position_tolerance_m,
            "direction_tolerance": direction_tolerance,
            "relative_tolerance": relative_tolerance,
            "switch_policy": "grid-face cap plus local discrepancy control; refine independently",
        },
    )


def adaptive_convergence(
    field, states, rigidity_tm, entry, exit, tolerances, **options
) -> dict[str, Any]:
    """Compare successively tighter position/direction tolerances and every status.

    No fixed-step order is assigned to adaptive tolerance studies.
    """
    tol = np.asarray(tolerances, dtype=float)
    if (
        tol.ndim != 2
        or tol.shape[1] != 2
        or len(tol) < 2
        or not np.isfinite(tol).all()
        or np.any(tol <= 0)
        or np.any(np.diff(tol, axis=0) >= 0)
    ):
        raise ValueError(
            "at least two strictly decreasing position/direction tolerance pairs required"
        )
    results = [
        track_adaptive(
            field,
            states,
            rigidity_tm,
            entry,
            exit,
            position_tolerance_m=p,
            direction_tolerance=d,
            **options,
        )
        for p, d in tol
    ]
    common = np.logical_and.reduce([r.reached_exit for r in results])
    changes = []
    for a, b in zip(results[:-1], results[1:], strict=True):
        changes.append(
            {
                "max_component_change_m_rad": np.max(
                    np.abs(a.states[common] - b.states[common]), axis=0
                ).tolist()
                if common.any()
                else None,
                "max_path_change_m": float(
                    np.max(np.abs(a.path_length_m[common] - b.path_length_m[common]))
                )
                if common.any()
                else None,
            }
        )
    return {
        "tolerances_m_direction": tol.tolist(),
        "common_survivors": int(common.sum()),
        "status_counts": [r.summary() for r in results],
        "changed_status_particles": int(
            np.any(np.stack([r.status for r in results]) != results[0].status, axis=0).sum()
        ),
        "changes": changes,
        "integration": [r.diagnostics for r in results],
        "interpretation": "empirical refinement, not a certified absolute-error bound",
    }
