"""Lorentz-force tracking through a static magnetic field (knowledge entry
``lorentz-force-tracking``).

A particle's state is its position ``r`` and unit direction ``t``. With the signed rigidity
``B rho = p / q`` in T*m, the equations of motion in path length ``s`` are

    dr/ds = t,    dt/ds = t x B(r) / (B rho),

integrated with fourth-order Runge-Kutta at a fixed step. The direction's norm is not
renormalized but monitored: its drift measures the integration error. RK4 is not
symplectic, and its error in a meshed field is measured (``step_convergence``), never
assumed. Only static magnetic fields are handled: no electric field, space charge,
scattering or material.

Particles start on an entry plane and end on an exit plane, each a ``bend.PathFrame``
(origin, tangent, horizontal, vertical axes). A state is ``(x, x', y, y')``: offsets in
metres along the plane's horizontal and vertical axes, and slopes ``(t . h) / (t . tangent)``
and ``(t . v) / (t . tangent)``. Slopes are not canonical momenta. The last steps shorten
so that a particle lands on the exit plane within ``plane_tolerance_m``; no state is
extrapolated across a step. A particle that leaves the field's support, leaves a
``region``, produces non-finite values, exceeds its path limit or meets the exit plane
going backwards stops with that status.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dcfield
from enum import IntEnum
from typing import Any, Protocol

import numpy as np

from element_surrogate.bend import PathFrame
from element_surrogate.readings import tracking_field
from element_surrogate.region import Region


class Field(Protocol):
    """A magnetic field in tesla at points in metres, and where it is defined."""

    def sample(self, points_m: np.ndarray) -> np.ndarray: ...

    def contains(self, points_m: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class UniformField:
    """A uniform field everywhere: an analytic control."""

    b_t: tuple[float, float, float]

    def sample(self, points_m: np.ndarray) -> np.ndarray:
        p = np.asarray(points_m, dtype=np.float64)
        return np.broadcast_to(np.asarray(self.b_t, dtype=np.float64), p.shape).copy()

    def contains(self, points_m: np.ndarray) -> np.ndarray:
        return np.ones(np.asarray(points_m).shape[:-1], dtype=bool)


class Status(IntEnum):
    REACHED_EXIT = 0
    OUTSIDE_FIELD = 1
    OUTSIDE_REGION = 2
    PATH_LIMIT = 3
    NON_FINITE = 4
    WRONG_EXIT_DIRECTION = 5
    STEP_LIMIT = 6


@dataclass
class TrackResult:
    """Per particle: the exit state (NaN unless the exit was reached), the status, the path
    length, the steps taken and the largest drift of the direction's norm from 1."""

    states: np.ndarray
    status: np.ndarray
    path_length_m: np.ndarray
    steps: np.ndarray
    max_direction_norm_error: np.ndarray
    diagnostics: dict[str, Any] = dcfield(default_factory=dict)

    @property
    def reached_exit(self) -> np.ndarray:
        result: np.ndarray = self.status == Status.REACHED_EXIT
        return result

    def summary(self) -> dict[str, int]:
        return {code.name.lower(): int(np.count_nonzero(self.status == code)) for code in Status}


def straight_frame(z_m: float) -> PathFrame:
    """The plane ``z = z_m`` of a straight magnet, with horizontal x and vertical y."""
    return PathFrame(
        np.array([0.0, 0.0, z_m]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    )


def to_lab(frame: PathFrame, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positions and unit directions of states ``(x, x', y, y')`` on a plane."""
    s = np.asarray(states, dtype=np.float64)
    r = frame.origin_m + s[:, 0, None] * frame.horizontal + s[:, 2, None] * frame.vertical
    t = frame.tangent + s[:, 1, None] * frame.horizontal + s[:, 3, None] * frame.vertical
    return r, t / np.linalg.norm(t, axis=1)[:, None]


def to_plane(frame: PathFrame, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """States ``(x, x', y, y')`` of points on a plane and their directions."""
    q = r - frame.origin_m
    along = t @ frame.tangent
    return np.stack(
        [
            q @ frame.horizontal,
            (t @ frame.horizontal) / along,
            q @ frame.vertical,
            (t @ frame.vertical) / along,
        ],
        axis=-1,
    )


Observer = Callable[[float, np.ndarray], None]


def track(
    field: Field,
    states: np.ndarray,
    rigidity_tm: float | np.ndarray,
    entry: PathFrame,
    exit: PathFrame,
    *,
    step_m: float,
    field_scale: float = 1.0,
    max_path_m: float | None = None,
    plane_tolerance_m: float = 1.0e-11,
    region: Region | None = None,
    observer: Observer | None = None,
) -> TrackResult:
    """Track particles from ``entry`` to ``exit`` through ``field_scale * field``.

    ``states`` is ``(N, 4)`` on the entry plane; ``rigidity_tm`` is the signed ``p / q``, one
    value or one per particle. ``max_path_m`` (default: ten times the distance between the
    planes' origins) only stops particles that loop. ``observer(s, u)`` receives every
    accepted state ``u = (r, t)`` of a single particle.
    """
    field = tracking_field(field)
    s0 = np.asarray(states, dtype=np.float64)
    if s0.ndim != 2 or s0.shape[1] != 4 or not np.isfinite(s0).all():
        raise ValueError("states must be finite, shape (N, 4)")
    n = s0.shape[0]
    brho = np.broadcast_to(np.asarray(rigidity_tm, dtype=np.float64), (n,)).copy()
    if not np.isfinite(brho).all() or np.any(brho == 0.0):
        raise ValueError("rigidity must be finite and non-zero")
    if not (math.isfinite(step_m) and step_m > 0 and plane_tolerance_m > 0):
        raise ValueError("step_m and plane_tolerance_m must be positive")
    if not math.isfinite(field_scale):
        raise ValueError("field_scale must be finite")
    if observer is not None and n != 1:
        raise ValueError("an observer follows one particle")
    if max_path_m is None:
        max_path_m = 10.0 * float(np.linalg.norm(exit.origin_m - entry.origin_m))
    if not max_path_m > 0:
        raise ValueError("max_path_m must be positive")

    r, t = to_lab(entry, s0)
    u = np.concatenate([r, t], axis=1)
    status = np.full(n, -1, dtype=np.int8)
    path = np.zeros(n)
    steps = np.zeros(n, dtype=np.int64)
    norm_error = np.zeros(n)
    out = np.full((n, 4), np.nan)
    if observer is not None:
        observer(0.0, u[0].copy())

    def stop(ids: np.ndarray, mask: np.ndarray, code: Status) -> None:
        hit = ids[mask & (status[ids] < 0)]
        status[hit] = code

    def rhs(v: np.ndarray, ids: np.ndarray) -> np.ndarray:
        p = v[:, :3]
        stop(ids, ~np.isfinite(v).all(axis=1), Status.NON_FINITE)
        stop(ids, ~field.contains(p), Status.OUTSIDE_FIELD)
        if region is not None:
            stop(ids, ~region.contains(p), Status.OUTSIDE_REGION)
        result = np.full_like(v, np.nan)
        ok = status[ids] < 0
        if ok.any():
            b = field.sample(p[ok]) * field_scale
            good = np.asarray(np.isfinite(b).all(axis=1))
            stop(ids[ok], ~good, Status.NON_FINITE)
            local = np.flatnonzero(ok)[good]
            result[local, :3] = v[local, 3:]
            result[local, 3:] = np.cross(v[local, 3:], b[good]) / brho[ids[local], None]
        return result

    def finish(ids: np.ndarray) -> None:
        if region is not None:
            inside = region.contains(u[ids, :3])
            status[ids[~inside]] = Status.OUTSIDE_REGION
            ids = ids[inside]
        along = u[ids, 3:] @ exit.tangent
        backwards = ~np.isfinite(along) | (along <= 0)
        status[ids[backwards]] = Status.WRONG_EXIT_DIRECTION
        ids = ids[~backwards]
        # The last accepted step can end just outside the field's support.
        supported = np.asarray(field.contains(u[ids, :3]))
        status[ids[~supported]] = Status.OUTSIDE_FIELD
        ids = ids[supported]
        if not ids.size:
            return
        # The remaining distance is within the plane tolerance: cover it with one Euler step
        # (position along t, direction turned by the field there), an error of its square.
        correction = -((u[ids, :3] - exit.origin_m) @ exit.tangent) / (u[ids, 3:] @ exit.tangent)
        turn = np.cross(u[ids, 3:], field.sample(u[ids, :3]) * field_scale) / brho[ids, None]
        u[ids, :3] += correction[:, None] * u[ids, 3:]
        u[ids, 3:] += correction[:, None] * turn
        path[ids] += correction
        supported = field.contains(u[ids, :3])
        status[ids[~supported]] = Status.OUTSIDE_FIELD
        ids = ids[supported]
        if region is not None:
            inside = region.contains(u[ids, :3])
            status[ids[~inside]] = Status.OUTSIDE_REGION
            ids = ids[inside]
        out[ids] = to_plane(exit, u[ids, :3], u[ids, 3:])
        status[ids] = Status.REACHED_EXIT
        if observer is not None:
            observer(float(path[0]), u[0].copy())

    for _ in range(math.ceil(max_path_m / step_m) + 128):
        ids = np.flatnonzero(status < 0)
        if not ids.size:
            break
        distance = (u[ids, :3] - exit.origin_m) @ exit.tangent
        near = np.abs(distance) <= plane_tolerance_m
        finish(ids[near])
        ids, distance = ids[~near], distance[~near]
        stop(ids, max_path_m - path[ids] <= 1e-14, Status.PATH_LIMIT)
        stop(ids, distance > plane_tolerance_m, Status.STEP_LIMIT)  # crossed without landing
        keep = status[ids] < 0
        ids, distance = ids[keep], distance[keep]
        if not ids.size:
            continue
        # Near the plane, step 90% of the way, so the field is never read beyond it.
        h = np.minimum(step_m, 0.9 * (-distance) / np.linalg.norm(u[ids, 3:], axis=1))
        h = np.minimum(h, max_path_m - path[ids])[:, None]
        v = u[ids]
        k1 = rhs(v, ids)
        k2 = rhs(v + 0.5 * h * k1, ids)
        k3 = rhs(v + 0.5 * h * k2, ids)
        k4 = rhs(v + h * k3, ids)
        trial = v + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        stop(ids, ~np.isfinite(trial).all(axis=1), Status.NON_FINITE)
        stop(ids, ~field.contains(trial[:, :3]), Status.OUTSIDE_FIELD)
        if region is not None:
            stop(ids, ~region.contains(trial[:, :3]), Status.OUTSIDE_REGION)
        ok = status[ids] < 0
        accepted = ids[ok]
        u[accepted] = trial[ok]
        path[accepted] += h[ok, 0]
        steps[accepted] += 1
        norm_error[accepted] = np.maximum(
            norm_error[accepted], np.abs(np.linalg.norm(trial[ok, 3:], axis=1) - 1.0)
        )
        if observer is not None and accepted.size:
            observer(float(path[0]), u[0].copy())
    status[status < 0] = Status.STEP_LIMIT
    return TrackResult(out, status, path, steps, norm_error)


def step_convergence(
    field: Field,
    states: np.ndarray,
    rigidity_tm: float | np.ndarray,
    entry: PathFrame,
    exit: PathFrame,
    steps_m: list[float],
    **options: Any,
) -> dict[str, Any]:
    """Track the same particles with each step (largest first) and report how much the exit
    states change between successive steps, and the observed order of that change. RK4 in a
    smooth field reads about 4; a mesh reading that is only continuous (trilinear) lowers
    it where steps cross cell faces. Successive-step differences are diagnostics,
    not certified absolute error bounds."""
    if (
        len(steps_m) < 2
        or not np.isfinite(steps_m).all()
        or min(steps_m) <= 0
        or not np.all(np.diff(steps_m) < 0)
    ):
        raise ValueError("give positive finite strictly decreasing steps, largest first")
    results = [track(field, states, rigidity_tm, entry, exit, step_m=h, **options) for h in steps_m]
    reached = np.logical_and.reduce([r.reached_exit for r in results])
    if not reached.any():
        raise ValueError("no particle reached the exit at every step")
    changes = []
    for coarse, fine in zip(results[:-1], results[1:], strict=True):
        difference = np.abs(coarse.states[reached] - fine.states[reached])
        changes.append(difference.max(axis=0))
    rows = []
    for k, change in enumerate(changes):
        row: dict[str, Any] = {
            "from_step_m": steps_m[k],
            "to_step_m": steps_m[k + 1],
            "max_change": {
                "x_m": float(change[0]),
                "xp_rad": float(change[1]),
                "y_m": float(change[2]),
                "yp_rad": float(change[3]),
            },
        }
        if k > 0:
            ratio = steps_m[k - 1] / steps_m[k]
            if math.isclose(ratio, steps_m[k] / steps_m[k + 1], rel_tol=1e-10):
                with np.errstate(divide="ignore", invalid="ignore"):
                    orders = np.log(changes[k - 1] / change) / math.log(ratio)
                row["observed_order"] = [float(o) if np.isfinite(o) else None for o in orders]
            else:
                row["observed_order"] = [None] * 4
                row["order_note"] = "omitted: unequal refinement ratios"
        rows.append(row)
    return {
        "particles": int(np.asarray(states).shape[0]),
        "reached_at_every_step": int(reached.sum()),
        "steps": rows,
        "max_direction_norm_error": [float(r.max_direction_norm_error.max()) for r in results],
        "status_counts": [r.summary() for r in results],
        "changed_status_particles": int(
            sum(np.any(np.stack([r.status for r in results]) != results[0].status, axis=0))
        ),
        "interpretation": "successive-step differences, not certified absolute error bounds",
    }
