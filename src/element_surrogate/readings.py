"""Mesh readings: how a field is read between the nodes of its grid.

A tracker's agreement with a simulator means something only when both read the mesh the
same way (knowledge entry ``lorentz-force-tracking``). ``FieldMap.sample`` is the trilinear
reading: linear in each coordinate within a cell, continuous across faces. This module adds
the triquadratic reading: a three-point Lagrange stencil along each axis, the tensor product
of the three.

The stencil's centre can be the nearest node (``centre="nearest"``), which makes the
reading jump where the nearest node changes, half-way between nodes; or the node below
(``centre="lower"``), which makes it jump at the nodes. At a nearest-node jump a query that
lands exactly half-way must pick one side (``tie``); a tracker whose steps land there depends
on that choice. Near a grid face the stencil shifts inward to stay on the grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from element_surrogate.fields import FieldMap

CENTRES = ("nearest", "lower")
TIES = ("upper", "lower")


@dataclass(frozen=True)
class TotalDegreeTwo:
    """Uniform-grid, nearest-centre degree-two Taylor reconstruction (19 nodes).

    Half-cell ties select the higher index; boundary centres shift inward. This
    reproduces degree-two polynomials, not arbitrary Maxwell-consistent fields.
    It may jump when its centre changes; integrators must refine across switches.
    """

    field: FieldMap

    def __post_init__(self) -> None:
        if min(self.field.shape) < 3:
            raise ValueError("total-degree-two requires three nodes per axis")
        for axis in (self.field.x_m, self.field.y_m, self.field.z_m):
            if not np.allclose(np.diff(axis), axis[1] - axis[0], rtol=1e-9, atol=0):
                raise ValueError("total-degree-two v1 requires a uniform grid")

    def contains(self, points_m: np.ndarray) -> np.ndarray:
        return self.field.contains(points_m)

    def _centres_offsets(self, points_m: np.ndarray):
        p = np.asarray(points_m, dtype=float)
        if p.ndim < 1 or p.shape[-1] != 3 or not np.isfinite(p).all():
            raise ValueError("finite points of shape (...,3) required")
        if not self.contains(p).all():
            raise ValueError("points lie outside the grid")
        centres, offsets = [], []
        for k, axis in enumerate((self.field.x_m, self.field.y_m, self.field.z_m)):
            q = np.clip((p[..., k] - axis[0]) / (axis[1] - axis[0]), 0, len(axis) - 1)
            c = np.clip(np.floor(q + 0.5).astype(int), 1, len(axis) - 2)
            centres.append(c)
            offsets.append(q - c)
        return centres, offsets

    def stencil_nodes(self, points_m):
        """Every node actually read, including nodes with a zero coefficient."""
        centres, _ = self._centres_offsets(points_m)
        return np.stack(
            [np.stack([centres[k] + d[k] for k in range(3)], -1) for d in total_degree_offsets()],
            axis=-2,
        )

    def sample(self, points_m: np.ndarray) -> np.ndarray:
        centres, offsets = self._centres_offsets(points_m)

        def node(delta: np.ndarray) -> np.ndarray:
            return cast(np.ndarray, self.field.b_t[tuple(centres[k] + delta[k] for k in range(3))])

        middle = node(np.zeros(3, dtype=int))
        out = middle.copy()
        directions = np.eye(3, dtype=int)
        for a in range(3):
            plus, minus = node(directions[a]), node(-directions[a])
            v = offsets[a][..., None]
            out += 0.5 * v * (plus - minus) + 0.5 * v * v * (plus - 2 * middle + minus)
            for b in range(a):
                mixed = sum(
                    sa * sb * node(sa * directions[a] + sb * directions[b])
                    for sa in (-1, 1)
                    for sb in (-1, 1)
                )
                out += 0.25 * (offsets[a] * offsets[b])[..., None] * mixed
        return out

    def describe(self) -> dict[str, str]:
        return {
            "reading": "total-degree-two-v1",
            "centre": "nearest",
            "tie": "upper",
            "boundary": "inward",
            "coefficients": "centred-finite-differences",
        }


def reconstruct(field: FieldMap, method: str = "total-degree-two"):
    """Select an explicit reconstruction; never silently downgrade its support."""
    if method == "total-degree-two":
        return TotalDegreeTwo(field)
    if method == "trilinear":
        return TrilinearReading(field)
    if method == "tensor-quadratic":
        return QuadraticReading(field)
    raise ValueError("unknown reconstruction")


@dataclass(frozen=True)
class TrilinearReading:
    """Explicit trilinear tracking choice; raw-grid readiness remains unchanged."""

    field: FieldMap

    def contains(self, points_m):
        return self.field.contains(points_m)

    def stencil_nodes(self, points_m):
        return self.field.interpolation_stencil(points_m)[0]

    def sample(self, points_m):
        return self.field.sample(points_m)

    def describe(self):
        return {"reading": "trilinear-v1", "centre": "cell"}


def tracking_field(field):
    """A raw grid gets the tracking default; an explicit reconstruction is preserved."""
    return reconstruct(field) if isinstance(field, FieldMap) else field


def _stencil(
    axis: np.ndarray, q: np.ndarray, centre: str, tie: str
) -> tuple[np.ndarray, np.ndarray]:
    """Centre node indices and the three Lagrange weights of each query coordinate."""
    lower = np.clip(np.searchsorted(axis, q, side="right") - 1, 0, axis.size - 2)
    if centre == "nearest":
        upper_distance = axis[lower + 1] - q
        lower_distance = q - axis[lower]
        upper_wins = upper_distance < lower_distance
        if tie == "upper":
            upper_wins |= upper_distance == lower_distance
        index = np.where(upper_wins, lower + 1, lower)
    else:
        index = lower
    index = np.clip(index, 1, axis.size - 2)
    a, b, c = axis[index - 1], axis[index], axis[index + 1]
    weights = np.stack(
        [
            (q - b) * (q - c) / ((a - b) * (a - c)),
            (q - a) * (q - c) / ((b - a) * (b - c)),
            (q - a) * (q - b) / ((c - a) * (c - b)),
        ],
        axis=-1,
    )
    return index, weights


@dataclass(frozen=True)
class QuadraticReading:
    """A field map read with a triquadratic stencil; a ``lorentz.Field``."""

    field: FieldMap
    centre: str = "nearest"
    tie: str = "upper"

    def __post_init__(self) -> None:
        if self.centre not in CENTRES:
            raise ValueError(f"centre must be one of {CENTRES}")
        if self.tie not in TIES:
            raise ValueError(f"tie must be one of {TIES}")
        if min(self.field.shape) < 3:
            raise ValueError("a quadratic reading needs at least three nodes along every axis")

    def contains(self, points_m: np.ndarray) -> np.ndarray:
        return self.field.contains(points_m)

    def _stencils(self, points_m):
        p = np.asarray(points_m, dtype=np.float64)
        if p.ndim < 1 or p.shape[-1] != 3 or not np.isfinite(p).all():
            raise ValueError("finite points of shape (...,3) required")
        if not self.contains(p).all():
            raise ValueError("points lie outside the grid")
        axes = (self.field.x_m, self.field.y_m, self.field.z_m)
        stencils = [
            _stencil(axis, np.clip(p[..., k], axis[0], axis[-1]), self.centre, self.tie)
            for k, axis in enumerate(axes)
        ]
        return stencils

    def stencil_nodes(self, points_m):
        stencils = self._stencils(points_m)
        return np.stack(
            [np.stack([stencils[k][0] + d[k] for k in range(3)], -1) for d in tensor_offsets()],
            axis=-2,
        )

    def sample(self, points_m: np.ndarray) -> np.ndarray:
        (ix, wx), (iy, wy), (iz, wz) = self._stencils(points_m)
        out = np.zeros(np.asarray(points_m).shape[:-1] + (3,))
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    weight = wx[..., a] * wy[..., b] * wz[..., c]
                    out += weight[..., None] * self.field.b_t[ix + a - 1, iy + b - 1, iz + c - 1]
        return out

    def describe(self) -> dict[str, str]:
        return {"reading": "triquadratic", "centre": self.centre, "tie": self.tie}


def total_degree_offsets():
    """Centre, six axial neighbours and twelve two-axis diagonals (19 reads)."""
    out = [np.zeros(3, dtype=int)]
    directions = np.eye(3, dtype=int)
    for a in range(3):
        out.extend([directions[a], -directions[a]])
        for b in range(a):
            out.extend(sa * directions[a] + sb * directions[b] for sa in (-1, 1) for sb in (-1, 1))
    return out


def tensor_offsets():
    """The 27 nodes of the tensor-product reconstruction."""
    return [np.array([a, b, c]) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
