"""Synthetic field maps with exact answers, and defects planted into them.

Each clean field is the field of magnetic point sources ("charges") placed outside the
grid: ``B(r) = sum_i s_i (r - r_i) / |r - r_i|^3``. Away from its sources such a field
has zero divergence and zero curl exactly, so every vacuum-Maxwell residual a tool
reports on it is the tool's own numerical error. Rows of sources around the axis give
multipole fields with realistic fringes; their parameters are test inputs, not the
properties of any real magnet.

The planting functions return a modified copy and say what they changed, so a test
can check that a readiness tool finds exactly the planted defect.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from element_surrogate.bend import NominalBend
from element_surrogate.fields import GAUSS_TO_T, FieldMap

Source = tuple[tuple[float, float, float], float]


def point_source_field(
    sources: Sequence[Source],
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
    label: str = "field",
) -> FieldMap:
    """Field of point sources ``((x, y, z) in m, strength in T*m^2)`` on a grid.

    Raises ``ValueError`` if a source lies inside the grid's bounding box, where the
    field would be singular and no longer a vacuum field.
    """
    x_m, y_m, z_m = (np.asarray(a, dtype=np.float64) for a in (x_m, y_m, z_m))
    lows = np.array([x_m[0], y_m[0], z_m[0]])
    highs = np.array([x_m[-1], y_m[-1], z_m[-1]])
    positions = np.array([p for p, _ in sources], dtype=np.float64).reshape(-1, 3)
    strengths = np.array([s for _, s in sources], dtype=np.float64)
    for p in positions:
        if np.all(p >= lows) and np.all(p <= highs):
            raise ValueError(f"source at {tuple(p)} lies inside the grid")
    nodes = np.stack(np.meshgrid(x_m, y_m, z_m, indexing="ij"), axis=-1).reshape(-1, 3)
    b = np.zeros_like(nodes)
    chunk = max(1, 2**22 // max(1, nodes.shape[0]))  # sources per pass, bounding memory
    for k in range(0, positions.shape[0], chunk):
        d = nodes[None, :, :] - positions[k : k + chunk, None, :]
        r2 = np.einsum("snc,snc->sn", d, d)
        weight = strengths[k : k + chunk, None] / (r2 * np.sqrt(r2))
        b += np.einsum("sn,snc->nc", weight, d)
    shape = (x_m.size, y_m.size, z_m.size, 3)
    return FieldMap(x_m, y_m, z_m, b.reshape(shape), label=label)


def multipole_sources(
    order: int, radius_m: float, length_m: float, strength: float, per_row: int
) -> list[Source]:
    """Rows of sources parallel to z on a circle of ``radius_m``: ``2 * order`` rows of
    alternating sign, centred on z = 0 and ``length_m`` long. ``order`` 1 gives a dipole-like
    field and 2 a quadrupole-like one, each with a fringe at both ends."""
    if order < 1 or per_row < 1:
        raise ValueError("order and per_row must be at least 1")
    zs = np.linspace(-0.5 * length_m, 0.5 * length_m, per_row)
    sources: list[Source] = []
    for k in range(2 * order):
        angle = np.pi * (k + 0.5) / order
        sign = 1.0 if k % 2 == 0 else -1.0
        for z in zs:
            sources.append(
                (
                    (radius_m * np.cos(angle), radius_m * np.sin(angle), float(z)),
                    sign * strength / per_row,
                )
            )
    return sources


def bend_sources(
    bend: NominalBend,
    half_gap_m: float,
    half_width_m: float,
    strength: float,
    along: int,
    across: int,
) -> list[Source]:
    """Two sheets of sources at ``y = +-half_gap_m`` over the arc of ``bend`` (its arms carry
    none), ``half_width_m`` to each side of it: a bending magnet whose poles follow the arc
    in the grid's Cartesian frame. ``strength`` > 0 gives ``By`` > 0 between the sheets."""
    if along < 2 or across < 1:
        raise ValueError("need at least two sources along the arc and one across it")
    s_values = bend.arm_m + np.linspace(0.0, bend.arc_length_m, along)
    offsets = np.linspace(-half_width_m, half_width_m, across) if across > 1 else np.zeros(1)
    each = strength / (along * across)
    sources: list[Source] = []
    for s in s_values:
        frame = bend.frame_at(float(s))
        for u in offsets:
            base = frame.origin_m + u * frame.horizontal
            for sign in (-1.0, 1.0):
                p = base + sign * half_gap_m * frame.vertical
                sources.append(((float(p[0]), float(p[1]), float(p[2])), -sign * each))
    return sources


@dataclass(frozen=True)
class Planted:
    """A field with a planted defect and a description of where it is."""

    field: FieldMap
    kind: str
    where: dict[str, Any]


def plant_zeroed_nodes(field: FieldMap, nodes: Sequence[tuple[int, int, int]]) -> Planted:
    """Set the field to exactly zero at the given nodes, as a masked export would."""
    b = field.b_t.copy()
    for node in nodes:
        b[node] = 0.0
    return Planted(field.with_values(b), "zeroed_nodes", {"nodes": [list(n) for n in nodes]})


def plant_duplicated_plane(field: FieldMap, axis: int, index: int) -> Planted:
    """Copy grid plane ``index`` onto plane ``index + 1`` along ``axis`` (0, 1, 2 = x, y, z)."""
    b = np.moveaxis(field.b_t.copy(), axis, 0)
    b[index + 1] = b[index]
    return Planted(
        field.with_values(np.moveaxis(b, 0, axis)),
        "duplicated_plane",
        {"axis": axis, "planes": [index, index + 1]},
    )


def plant_noisy_shell(
    field: FieldMap, inner_radius_m: float, amplitude_t: float, seed: int
) -> Planted:
    """Add random noise to every node farther than ``inner_radius_m`` from the z axis.

    Noise has no divergence- or curl-free structure, so it imitates a region whose
    values are not a vacuum field (for example a badly sampled or badly exported shell).
    """
    rng = np.random.default_rng(seed)
    r = np.hypot(field.x_m[:, None], field.y_m[None, :])
    outer = (r > inner_radius_m)[:, :, None, None]
    noise = rng.normal(0.0, amplitude_t, field.b_t.shape)
    b = np.where(outer, field.b_t + noise, field.b_t)
    return Planted(
        field.with_values(b),
        "noisy_shell",
        {"inner_radius_m": inner_radius_m, "amplitude_t": amplitude_t},
    )


def plant_gauss_as_tesla(field: FieldMap) -> Planted:
    """The values a reader returns when a file holds gauss but declares tesla: every value
    is 1/GAUSS_TO_T times too large. Maxwell's equations cannot see a scale factor; only a
    comparison with the described strength can."""
    return Planted(field.with_values(field.b_t / GAUSS_TO_T), "gauss_as_tesla", {"factor": 1.0e4})


def plant_truncation(field: FieldMap, keep_z_m: tuple[float, float]) -> Planted:
    """Keep only the planes with ``keep_z_m[0] <= z <= keep_z_m[1]``, as an export whose box
    ends inside the fringe field would."""
    keep = np.flatnonzero((field.z_m >= keep_z_m[0]) & (field.z_m <= keep_z_m[1]))
    window = field.window(slice(None), slice(None), slice(int(keep[0]), int(keep[-1]) + 1))
    return Planted(
        window, "truncated", {"z_range_m": [float(window.z_m[0]), float(window.z_m[-1])]}
    )


def plant_storage_mismatch(
    field: FieldMap,
    nesting: tuple[int, int, int] = (0, 1, 2),
    columns: tuple[int, int, int] = (0, 1, 2),
    signs: tuple[int, int, int] = (1, 1, 1),
) -> Planted:
    """What a reader that assumes x slowest, z fastest and columns (Bx, By, Bz) sees when the
    file was written with the grid axes nested as ``nesting`` (slowest to fastest) and its
    columns holding the components ``columns`` with ``signs``."""
    true = field.b_t
    stored = np.stack([signs[c] * true[..., columns[c]] for c in range(3)], axis=-1)
    rows = np.transpose(stored, (*nesting, 3)).reshape(-1, 3)
    names = ("x", "y", "z")
    components = ("Bx", "By", "Bz")
    return Planted(
        field.with_values(rows.reshape(*field.shape, 3)),
        "storage_mismatch",
        {
            "axes_slowest_to_fastest": [names[a] for a in nesting],
            "columns_hold": [
                ("-" if signs[c] < 0 else "") + components[columns[c]] for c in range(3)
            ],
        },
    )
