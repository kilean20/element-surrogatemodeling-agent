"""Readiness checks on a field map: is it a physical vacuum field, and are its nodes sound?

Physics first: where there is no current and no magnetized material, a static field
has zero divergence and zero curl at every point, right up to any material surface. A
residual in a vacuum cell is therefore numerical or a processing error (sampling,
interpolation, export, solver convergence, units, masks), never a property of the
magnet. Across a material surface the normal component of B is continuous and the
tangential component jumps, so a cell or node at one is outside these tests.

The checks report numbers, where they occur, and how they change with grid spacing;
they apply no thresholds carried over from other magnets. Two scales recur:

* truncation error of a smooth field, which grows as the spacing squared (observed
  order about 2 on coarsening);
* node-level error in the data itself (rounding, export interpolation, solver noise),
  which does not grow on coarsening. Exported maps usually carry such a floor
  everywhere, and a defect is judged against the map's own floor.

Checks that can decide something on their own add ``findings``: each names the check, a
kind, what was found and where, and the criterion that decided it. A criterion is a
statistic of the map itself or a tolerance the caller states, never a number carried over
from another magnet.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from element_surrogate.diagnostic_blocks import DEFAULT_WORKING_BYTES, validate_budget
from element_surrogate.fields import FieldMap
from element_surrogate.region import Region


def finding(check: str, kind: str, summary: str, criterion: str, **where: Any) -> dict[str, Any]:
    """One finding of a check, in the form every check uses."""
    return {
        "check": check,
        "kind": kind,
        "summary": summary,
        "criterion": criterion,
        "where": where,
    }


METHOD = (
    "raw-vector, cell-centred finite-volume divergence and curl on the map's own grid with its "
    "actual cell widths; no ideal magnet subtracted"
)


@dataclass(frozen=True)
class CellResiduals:
    """Residuals at cell centres, shape ``(nx-1, ny-1, nz-1)``; the curl vector adds a last
    axis of 3."""

    divergence_t_per_m: np.ndarray
    curl_vector_t_per_m: np.ndarray
    derivative_norm_t_per_m: np.ndarray

    @property
    def curl_t_per_m(self) -> np.ndarray:
        result: np.ndarray = np.linalg.norm(self.curl_vector_t_per_m, axis=-1)
        return result


def cell_derivatives(field: FieldMap) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """d/dx, d/dy, d/dz of each field component at cell centres, shape ``(nx-1, ny-1, nz-1, 3)``.

    Each is the difference across the cell's two faces normal to that axis, averaged
    over the cell's four edges parallel to it.
    """
    b = field.b_t
    dx = np.diff(field.x_m)[:, None, None, None]
    dy = np.diff(field.y_m)[None, :, None, None]
    dz = np.diff(field.z_m)[None, None, :, None]
    face_x = (b[1:] - b[:-1]) / dx
    d_dx = 0.25 * (
        face_x[:, :-1, :-1] + face_x[:, 1:, :-1] + face_x[:, :-1, 1:] + face_x[:, 1:, 1:]
    )
    face_y = (b[:, 1:] - b[:, :-1]) / dy
    d_dy = 0.25 * (
        face_y[:-1, :, :-1] + face_y[1:, :, :-1] + face_y[:-1, :, 1:] + face_y[1:, :, 1:]
    )
    face_z = (b[:, :, 1:] - b[:, :, :-1]) / dz
    d_dz = 0.25 * (face_z[:-1, :-1] + face_z[1:, :-1] + face_z[:-1, 1:] + face_z[1:, 1:])
    return d_dx, d_dy, d_dz


def cell_residuals(field: FieldMap) -> CellResiduals:
    d_dx, d_dy, d_dz = cell_derivatives(field)
    divergence = d_dx[..., 0] + d_dy[..., 1] + d_dz[..., 2]
    curl = np.stack(
        (d_dy[..., 2] - d_dz[..., 1], d_dz[..., 0] - d_dx[..., 2], d_dx[..., 1] - d_dy[..., 0]),
        axis=-1,
    )
    norm = np.sqrt(np.sum(d_dx * d_dx + d_dy * d_dy + d_dz * d_dz, axis=-1))
    return CellResiduals(divergence, curl, norm)


def coarsen(field: FieldMap, step: int = 2) -> FieldMap:
    """Every ``step``-th node along each axis: the same data, ``step`` times more coarsely."""
    s = slice(None, None, step)
    return FieldMap(
        field.x_m[s], field.y_m[s], field.z_m[s], field.b_t[s, s, s], field.label, field.source
    )


def _stats(values: np.ndarray, scale: float | None = None) -> dict[str, Any]:
    values = np.abs(values[np.isfinite(values)])
    if values.size == 0:
        return {"cells": 0, "rms": None, "p99": None, "max": None, "p99_relative": None}
    p99 = float(np.quantile(values, 0.99))
    return {
        "cells": int(values.size),
        "rms": float(np.sqrt(np.mean(values * values))),
        "p99": p99,
        "max": float(values.max()),
        "p99_relative": p99 / scale if scale else None,
    }


def _order(fine: float | None, coarse: float | None) -> float | None:
    if not fine or not coarse:
        return None
    return float(np.log2(coarse / fine))


def _child_mean(fine: np.ndarray, coarse_shape: tuple[int, int, int]) -> np.ndarray:
    """Mean of the eight fine cells inside each cell of the grid coarsened by two."""
    mx, my, mz = coarse_shape
    block = fine[: 2 * mx, : 2 * my, : 2 * mz]
    result: np.ndarray = block.reshape(mx, 2, my, 2, mz, 2, *fine.shape[3:]).mean(axis=(1, 3, 5))
    return result


def _children(coarse_mask: np.ndarray, fine_shape: tuple[int, ...]) -> np.ndarray:
    """Fine cells inside the selected cells of the grid coarsened by two."""
    m = coarse_mask.repeat(2, 0).repeat(2, 1).repeat(2, 2)
    out = np.zeros(fine_shape, dtype=bool)
    sx, sy, sz = (min(a, b) for a, b in zip(m.shape, fine_shape, strict=True))
    out[:sx, :sy, :sz] = m[:sx, :sy, :sz]
    return out


ORDER_READING = (
    "pointwise order compares each coarse cell with the mean of the eight fine cells inside it; "
    "same_volume order compares the 99th percentiles over the same volume. Measured on synthetic "
    "fields: truncation of a smooth field reads pointwise 2 and same-volume 2 or somewhat less "
    "(less where the residual changes a lot across one coarse cell, as next to the axis); "
    "independent node errors read about 0.5 and -1; a jump between neighbouring nodes reads about "
    "0 and -1 for the quantity it affects. When the two disagree in other ways (for example "
    "pointwise above 2 with same-volume below 0) the residual is not of one simple kind and the "
    "orders alone do not identify its cause. Cells at a material surface are outside the test."
)


STENCIL_READING = (
    "higher_order_stencil compares node-centred divergence and curl from fourth-order and "
    "second-order central differences at the same nodes (uniform axes only). Truncation of a "
    "smooth field falls with the higher order (ratio well below 1); independent node errors "
    "rise by the stencils' noise gains, sqrt(130)/12 over sqrt(2)/2, a ratio of about 1.34; an "
    "error the higher order cannot remove reads about 1."
)


def _node_residuals(field: FieldMap, order: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Node-centred |divergence| and |curl| from central differences of ``order`` 2 or 4 at
    the nodes at least two from every face, shape ``(nx - 4, ny - 4, nz - 4)``. Fourth order
    needs uniform axes; returns None otherwise."""
    b = field.b_t
    inner = (slice(2, -2),) * 3
    derivative = []
    for axis, values in enumerate((field.x_m, field.y_m, field.z_m)):
        steps = np.diff(values)
        if order == 4 and not np.allclose(steps, steps[0], rtol=1e-9, atol=0.0):
            return None

        def shifted(k: int, axis: int = axis, size: int = values.size) -> np.ndarray:
            index = list(inner)
            index[axis] = slice(2 + k, size - 2 + k)
            return b[tuple(index)]

        if order == 2:
            span = (values[2:] - values[:-2])[1:-1]
            shape = [1, 1, 1, 1]
            shape[axis] = span.size
            derivative.append((shifted(1) - shifted(-1)) / span.reshape(shape))
        else:
            h = float(steps[0])
            derivative.append(
                (-shifted(2) + 8 * shifted(1) - 8 * shifted(-1) + shifted(-2)) / (12 * h)
            )
    d_dx, d_dy, d_dz = derivative
    divergence = np.abs(d_dx[..., 0] + d_dy[..., 1] + d_dz[..., 2])
    curl = np.sqrt(
        (d_dy[..., 2] - d_dz[..., 1]) ** 2
        + (d_dz[..., 0] - d_dx[..., 2]) ** 2
        + (d_dx[..., 1] - d_dy[..., 0]) ** 2
    )
    return divergence, curl


def _eager_maxwell_report(
    field: FieldMap, region: Region | None = None, shells: int = 10
) -> dict[str, Any]:
    """Vacuum-Maxwell residuals in shells of a region (default: the largest circle about the
    z axis that the grid covers), with two cause-finding comparisons in each shell: the grid
    coarsened by two, and a fourth-order stencil against a second-order one."""
    if shells < 1:
        raise ValueError("shells must be at least 1")
    region = region if region is not None else Region.covering(field)
    res = cell_residuals(field)
    inside = region.cell_mask(field)
    if not inside.any():
        raise ValueError("no cell lies inside the region")
    scale_values = res.derivative_norm_t_per_m[inside]
    scale = float(np.sqrt(np.mean(scale_values[np.isfinite(scale_values)] ** 2)))

    coarse_field = coarsen(field)
    can_coarsen = min(coarse_field.shape) >= 2
    if can_coarsen:
        coarse = cell_residuals(coarse_field)
        shape3 = coarse.divergence_t_per_m.shape
        mean_div = _child_mean(res.divergence_t_per_m, shape3)
        mean_curl = np.linalg.norm(_child_mean(res.curl_vector_t_per_m, shape3), axis=-1)

    radial = region.cell_radial(field)
    edges = np.linspace(0.0, region.radius_m, shells + 1)
    stencils = None
    if min(field.shape) >= 5:
        second, fourth = _node_residuals(field, 2), _node_residuals(field, 4)
        if second is not None and fourth is not None:
            node_radial, along = region.node_measures(field)
            node_radial = np.broadcast_to(node_radial, field.shape)[2:-2, 2:-2, 2:-2]
            along = np.broadcast_to(along, field.shape)[2:-2, 2:-2, 2:-2]
            stencils = (second, fourth, node_radial, along)
    rows = []
    for k, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        lower = (radial >= low) if k == 0 else (radial > low)
        mask = inside & lower & (radial <= high * (1 + 1e-12))
        row: dict[str, Any] = {
            "outer_radius_m": float(high),
            "divergence_t_per_m": _stats(res.divergence_t_per_m[mask], scale),
            "curl_t_per_m": _stats(res.curl_t_per_m[mask], scale),
            "coarsened_by_2": None,
        }
        if can_coarsen:
            c_inside = region.cell_mask(coarse_field, outer_m=high)
            c_radial = region.cell_radial(coarse_field)
            c_lower = (c_radial >= low) if k == 0 else (c_radial > low)
            c_mask = c_inside & c_lower
            f_mask = _children(c_mask, res.divergence_t_per_m.shape)
            orders: dict[str, Any] = {}
            for key, fine, coarse_v, child in (
                ("divergence", res.divergence_t_per_m, coarse.divergence_t_per_m, mean_div),
                ("curl", res.curl_t_per_m, coarse.curl_t_per_m, mean_curl),
            ):
                c_p99 = _stats(coarse_v[c_mask])["p99"]
                orders[key] = {
                    "pointwise_order": _order(_stats(child[c_mask])["p99"], c_p99),
                    "same_volume_order": _order(_stats(fine[f_mask])["p99"], c_p99),
                }
            row["coarsened_by_2"] = {"coarse_cells": int(c_mask.sum()), **orders}
        row["higher_order_stencil"] = None
        if stencils is not None:
            second, fourth, node_radial, along = stencils
            lower_n = (node_radial >= low) if k == 0 else (node_radial > low)
            nodes = along & lower_n & (node_radial <= high * (1 + 1e-12))
            ratios: dict[str, Any] = {"nodes": int(nodes.sum())}
            for key, i in (("divergence", 0), ("curl", 1)):
                p2 = _stats(second[i][nodes])["p99"]
                p4 = _stats(fourth[i][nodes])["p99"]
                ratios[f"{key}_p99_fourth_over_second"] = p4 / p2 if p2 else None
            row["higher_order_stencil"] = ratios
        rows.append(row)
    return {
        "method": METHOD,
        "region": region.as_dict(),
        "derivative_scale_rms_t_per_m": scale,
        "whole_region": {
            "divergence_t_per_m": _stats(res.divergence_t_per_m[inside], scale),
            "curl_t_per_m": _stats(res.curl_t_per_m[inside], scale),
        },
        "shells": rows,
        "reading_the_order": ORDER_READING,
        "reading_the_stencil": STENCIL_READING,
        "reading_relative": "p99_relative divides by derivative_scale_rms_t_per_m over the region",
    }


def _laplacian(
    b: np.ndarray, axes: tuple[np.ndarray, ...], step: int
) -> tuple[np.ndarray, np.ndarray]:
    """Discrete Laplacian of each component using neighbours ``step`` nodes away, and the sum
    of the centre-node stencil weights (``sum over axes of 2 / (h+ h-)``).

    Both are full-grid arrays; they are complete only on nodes at least ``step`` nodes from
    every face, which the caller selects.
    """
    lap = np.zeros(b.shape[:3] + (3,))
    coeff = np.zeros(b.shape[:3])
    for a, axis in enumerate(axes):
        n = axis.size
        centre = slice(step, n - step)
        plus = slice(2 * step, n)
        minus = slice(0, n - 2 * step)
        h_plus = axis[plus] - axis[centre]
        h_minus = axis[centre] - axis[minus]
        shape = [1, 1, 1]
        shape[a] = -1
        hp = h_plus.reshape(shape)
        hm = h_minus.reshape(shape)
        bm = np.moveaxis(b, a, 0)
        f0 = np.moveaxis(bm[centre], 0, a)
        fp = np.moveaxis(bm[plus], 0, a)
        fn = np.moveaxis(bm[minus], 0, a)
        second = (
            2.0
            * (fp * hm[..., None] - f0 * (hp + hm)[..., None] + fn * hp[..., None])
            / ((hp * hm * (hp + hm))[..., None])
        )
        index = [slice(None)] * 3
        index[a] = centre
        lap[tuple(index)] += second
        coeff[tuple(index)] += np.broadcast_to(2.0 / (hp * hm), f0.shape[:3])
    return lap, coeff


Node = tuple[int, int, int]


def _clusters(nodes: list[Node]) -> list[list[Node]]:
    """Group nodes that touch (26-neighbourhood) into connected clusters."""
    remaining = set(nodes)
    groups = []
    while remaining:
        stack = [remaining.pop()]
        group = []
        while stack:
            node = stack.pop()
            group.append(node)
            for d in itertools.product((-1, 0, 1), repeat=3):
                neighbour = (node[0] + d[0], node[1] + d[1], node[2] + d[2])
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    stack.append(neighbour)
        groups.append(group)
    return groups


def _local_median(values: np.ndarray, reach: int) -> np.ndarray:
    """Median of the finite values within ``reach`` nodes along each axis of every node (a
    cube of side ``2 * reach + 1``); NaN marks values to leave out."""
    side = 2 * reach + 1
    padded = np.pad(values, reach, constant_values=np.nan)
    out = np.full(values.shape, np.nan)
    for i in range(values.shape[0]):
        window = np.lib.stride_tricks.sliding_window_view(padded[i : i + side], (side,) * 3)[0]
        finite = np.isfinite(values[i])
        if finite.any():
            with np.errstate(all="ignore"):
                out[i][finite] = np.nanmedian(window[finite].reshape(-1, side**3), axis=1)
    return out


def _eager_harmonic_nodes(
    field: FieldMap,
    region: Region | None = None,
    top: int = 100,
    examples: int = 10,
    fence_iqr: float = 3.0,
) -> dict[str, Any]:
    """Nodes whose value disagrees with what their neighbours imply for a vacuum field,
    judged against the map's own floor and their own neighbourhood.

    In vacuum each Cartesian component of B is harmonic. Dividing a node's discrete
    Laplacian by its stencil weight gives its *implied node error*: how much the value
    would have to change to satisfy the discrete Laplace equation, in tesla. The *order*
    ``log2(|L_2h| / |L_h|)`` compares neighbours two nodes away with neighbours one node
    away: about +2 where the Laplacian is truncation of a smooth field, about -2 where it is
    node-level error in controlled examples. The *floor* is the median implied correction;
    its order is a diagnostic signature, not a unique cause or absolute uncertainty.
    Harmonic errors in strength, polarity or units can leave this residual unchanged.
    Individually inconsistent values can stand far above it. The ``top``
    highest-ranked nodes are grouped into touching clusters.

    A node is *flagged* when its implied error is node-level (order below 0) and stands out
    from its neighbourhood: its *local ratio*, the implied error divided by the median over
    the nodes within two of it (the reach of the order's stencil), lies beyond the far-out
    fence of the map's own local ratios, the upper quartile plus ``fence_iqr`` interquartile
    ranges in log scale (3 is Tukey's far-out fence). Exported maps carry node errors whose
    size varies across the map and whose tail is far heavier than Gaussian noise's, so no
    noise model is assumed: the map's own distribution sets the fence. Flagged nodes are
    wrong values (masked, missing, corrupted) or a material surface; each touching cluster
    of them is one finding (the worst ``examples`` are listed). A region of wrong values
    many nodes deep shows at its boundary; ``maxwell_report`` shows its extent.

    Only nodes at least two nodes from every face (and inside ``region``, if given) count.
    """
    axes = (field.x_m, field.y_m, field.z_m)
    if min(field.shape) < 5:
        raise ValueError("the grid needs at least five nodes along every axis")
    lap1, w1 = _laplacian(field.b_t, axes, 1)
    lap2, _ = _laplacian(field.b_t, axes, 2)
    evaluated = np.zeros(field.shape, dtype=bool)
    evaluated[2:-2, 2:-2, 2:-2] = True
    if region is not None:
        evaluated &= region.node_mask(field)
    if not evaluated.any():
        raise ValueError("no node is evaluated: the region is empty or too close to the faces")
    l1 = np.linalg.norm(lap1, axis=-1)
    l2 = np.linalg.norm(lap2, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        error = np.where(evaluated, l1 / w1, np.nan)
        order = np.where(evaluated, np.log2(l2 / l1), np.nan)
    values = error[evaluated]
    floor = float(np.median(values))
    finite_orders = order[evaluated & np.isfinite(order)]
    floor_order = float(np.median(finite_orders)) if finite_orders.size else math.nan

    eligible = np.flatnonzero(evaluated)
    flat = eligible[np.lexsort((eligible, -error.ravel()[eligible]))[: max(top, examples)]]
    ranked: list[Node] = []
    for k in flat:
        i, j, m = np.unravel_index(k, field.shape)
        ranked.append((int(i), int(j), int(m)))

    def describe(node: Node) -> dict[str, Any]:
        return {
            "node": list(node),
            "position_m": [
                float(field.x_m[node[0]]),
                float(field.y_m[node[1]]),
                float(field.z_m[node[2]]),
            ],
            "implied_error_t": float(error[node]),
            "ratio_to_floor": float(error[node] / floor) if floor > 0 else None,
            "local_ratio": float(local[node]) if np.isfinite(local[node]) else None,
            "node_field_t": float(np.linalg.norm(field.b_t[node])),
            "order": float(order[node]) if np.isfinite(order[node]) else None,
        }

    clusters: list[dict[str, Any]] = []
    for group in _clusters(ranked[:top]):
        idx = np.array(group)
        errs = np.array([error[n] for n in group])
        lo, hi = idx.min(axis=0), idx.max(axis=0)
        clusters.append(
            {
                "nodes": len(group),
                "node_range": [lo.tolist(), hi.tolist()],
                "position_range_m": [
                    [float(field.x_m[lo[0]]), float(field.y_m[lo[1]]), float(field.z_m[lo[2]])],
                    [float(field.x_m[hi[0]]), float(field.y_m[hi[1]]), float(field.z_m[hi[2]])],
                ],
                "max_implied_error_t": float(errs.max()),
                "max_ratio_to_floor": float(errs.max() / floor) if floor > 0 else None,
            }
        )
    clusters.sort(key=lambda c: c["max_implied_error_t"], reverse=True)  # worst first

    with np.errstate(divide="ignore", invalid="ignore"):
        local = error / _local_median(error, 2)
        logs = np.log10(local[evaluated & np.isfinite(local) & (local > 0.0)])
    if logs.size:
        q1, q3 = np.percentile(logs, [25.0, 75.0])
        fence = float(10.0 ** (q3 + fence_iqr * (q3 - q1)))
    else:
        fence = math.inf  # No positive finite baseline: report undefined, not a false defect.
    with np.errstate(invalid="ignore"):
        flagged_mask = evaluated & (local > fence) & (order < 0.0)
    flagged = [(int(i), int(j), int(m)) for i, j, m in np.argwhere(flagged_mask)]
    findings = []
    criterion = (
        f"implied error above {fence:.3g} x the median of the nodes within two (the far-out "
        f"fence, {fence_iqr:g} interquartile ranges above the upper quartile of the map's own "
        "local ratios in log scale) and order below 0"
    )
    flagged_clusters = sorted(_clusters(flagged), key=lambda g: -max(local[n] for n in g))
    for group in flagged_clusters[:examples]:
        idx = np.array(group)
        lo, hi = idx.min(axis=0), idx.max(axis=0)
        findings.append(
            finding(
                "harmonic_nodes",
                "wrong_values_or_material_surface",
                f"{len(group)} touching node(s) far above the map's floor",
                criterion,
                node_range=[lo.tolist(), hi.tolist()],
                position_range_m=[
                    [float(field.x_m[lo[0]]), float(field.y_m[lo[1]]), float(field.z_m[lo[2]])],
                    [float(field.x_m[hi[0]]), float(field.y_m[hi[1]]), float(field.z_m[hi[2]])],
                ],
                max_local_ratio=float(max(local[n] for n in group)),
                max_ratio_to_floor=float(max(error[n] for n in group) / floor)
                if floor > 0
                else None,
            )
        )
    return {
        "method": (
            "discrete Laplacian of each component on the map's own grid; implied node error = "
            "|Laplacian| / stencil weight; order = log2(|L_2h| / |L_h|)"
        ),
        "region": region.as_dict() if region is not None else None,
        "evaluated_nodes": int(values.size),
        "absolute_field_uncertainty_calibrated": False,
        "interpretation": (
            "local harmonic consistency; cannot certify absolute scale, "
            "polarity or material clearance"
        ),
        "floor": {
            "median_implied_error_t": floor,
            "median_order": floor_order,
            "regime": (
                "order not identifiable from zero/non-finite residual ratios"
                if not math.isfinite(floor_order)
                else "node-level error in the data"
                if floor_order < 0
                else "truncation of a smooth field"
            ),
        },
        "implied_error_t": {
            "p99": float(np.quantile(values, 0.99)),
            "p999": float(np.quantile(values, 0.999)),
            "max": float(values.max()),
        },
        "worst_nodes": [describe(n) for n in ranked[:examples]],
        "clusters_of_top_nodes": {"top": min(top, len(ranked)), "clusters": clusters},
        "local_ratio_fence": fence,
        "flagged_nodes": len(flagged),
        "flagged_clusters": len(flagged_clusters),
        "findings": findings,
        "reading": (
            "Judge nodes by ratio_to_floor, not by the sign of their order: when the floor itself "
            "is node-level error (median order below 0), every node reads negative. A cluster far "
            "above the floor is a region of wrong values (masked, missing, corrupted) or a "
            "material surface. For independent node errors of standard deviation s per component "
            "on a uniform grid, the median implied error is about 1.66 s."
        ),
    }


def _eager_node_defects(field: FieldMap, max_examples: int = 10) -> dict[str, Any]:
    """Separate non-finite defects from contextual zero/duplicate observations.

    Reports non-finite values; zero vectors next to a non-zero node (a masked or missing
    node, or the edge of a zeroed region); and grid planes identical to the next plane
    along an axis, excluding planes that are entirely zero.
    """
    b = field.b_t
    nonfinite = np.argwhere(~np.isfinite(b).all(axis=-1))

    zero = ~np.any(b, axis=-1)
    padded = np.pad(zero, 1, constant_values=True)
    all_neighbours_zero = (
        padded[:-2, 1:-1, 1:-1]
        & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1]
        & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2]
        & padded[1:-1, 1:-1, 2:]
    )
    zero_next_to_nonzero = np.argwhere(zero & ~all_neighbours_zero)

    duplicated = []
    for axis in range(3):
        planes = np.moveaxis(b, axis, 0)
        for i in range(planes.shape[0] - 1):
            if np.any(planes[i]) and np.array_equal(planes[i], planes[i + 1]):
                duplicated.append({"axis": "xyz"[axis], "planes": [i, i + 1]})

    return _node_defects_result(
        [len(nonfinite), len(zero_next_to_nonzero)],
        [nonfinite[:max_examples].tolist(), zero_next_to_nonzero[:max_examples].tolist()],
        duplicated,
    )


_AXES = ("x", "y", "z")
_COMPONENTS = ("Bx", "By", "Bz")
# An overall sign flip leaves Maxwell's equations unchanged, so four sign classes suffice.
_SIGN_CLASSES = ((1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1))


def _renest(field: FieldMap, assumed: tuple[int, ...], nesting: tuple[int, ...]) -> FieldMap:
    """The field the file would hold if its values nested the grid axes as ``nesting``
    (slowest to fastest), given that the reader assumed the nesting ``assumed``. The columns
    stay as stored."""
    stored = np.transpose(field.b_t, (*assumed, 3)).reshape(-1, 3)
    dims = [field.shape[a] for a in nesting]
    return field.with_values(np.transpose(stored.reshape(*dims, 3), (*np.argsort(nesting), 3)))


def _relative_residual(
    jacobian: np.ndarray, scale: float, components: tuple[int, ...], signs: tuple[int, ...]
) -> float | None:
    """The larger of the median |divergence| and median |curl| when stored column ``col``
    holds component ``components[col]`` with sign ``signs[col]``, divided by ``scale``, the
    median field-derivative norm (the same for every assignment). ``jacobian[cell, col, a]``
    is the derivative of stored column ``col`` along axis ``a``. Medians, because a wrong
    storage order scrambles the field everywhere while a defect is local: a 99th percentile
    would let one bad shell decide."""
    j = np.empty_like(jacobian)
    for column, component in enumerate(components):
        j[:, component, :] = signs[column] * jacobian[:, column, :]
    divergence = j[:, 0, 0] + j[:, 1, 1] + j[:, 2, 2]
    curl = np.sqrt(
        (j[:, 2, 1] - j[:, 1, 2]) ** 2
        + (j[:, 0, 2] - j[:, 2, 0]) ** 2
        + (j[:, 1, 0] - j[:, 0, 1]) ** 2
    )
    worst = max(float(np.median(np.abs(divergence))), float(np.median(curl)))
    return worst / scale if scale > 0 else None


def _eager_storage_order(
    field: FieldMap, region: Region | None = None, report: int = 5, tie_ratio: float = 1.01
) -> dict[str, Any]:
    """Test the reader's storage assumption against Maxwell's equations.

    Re-reads the same numbers under every nesting of the grid axes, every assignment of
    the columns to (Bx, By, Bz) and every relative sign, and ranks the interpretations by
    their relative vacuum-Maxwell residual (the larger of the median divergence and median
    curl over the region, divided by the median field derivative). The interpretation a
    vacuum field supports has the smallest residual; medians keep a local defect from
    deciding a global question. Interpretations that a symmetry of the magnet makes
    equivalent tie; residuals within ``tie_ratio`` of the lowest count as tied, and a finding
    is made when the reading as read is not among them.
    """
    region = region if region is not None else Region.covering(field)
    assumed = (
        field.source.assumed_storage
        if field.source is not None
        else {"axes_slowest_to_fastest": list(_AXES), "components": list(_COMPONENTS)}
    )
    base = tuple(_AXES.index(a) for a in assumed["axes_slowest_to_fastest"])
    window = region.node_window(field)
    results: list[dict[str, Any]] = []
    for nesting in itertools.permutations(range(3)):
        # Derivatives are linear in the field, so each nesting's are computed once and every
        # assignment of columns and signs is a relabelling of them.
        candidate = _renest(field, base, nesting).window(*window)
        inside = region.cell_mask(candidate)
        jacobian = np.stack(cell_derivatives(candidate), axis=-1)[inside]
        scale = float(np.median(np.sqrt(np.sum(jacobian**2, axis=(1, 2)))))
        for components in itertools.permutations(range(3)):
            for signs in _SIGN_CLASSES:
                results.append(
                    {
                        "axes_slowest_to_fastest": [_AXES[a] for a in nesting],
                        "columns_hold": [
                            ("-" if s < 0 else "") + _COMPONENTS[c]
                            for c, s in zip(components, signs, strict=True)
                        ],
                        "relative_residual": _relative_residual(jacobian, scale, components, signs),
                        "as_read": nesting == base
                        and components == (0, 1, 2)
                        and signs == (1, 1, 1),
                    }
                )
    return _storage_order_result(results, assumed, region, report, tie_ratio)


def _storage_order_result(results, assumed, region, report, tie_ratio):
    """Rank identifiable candidates, retaining undefined zero-scale scores."""
    results.sort(key=lambda r: (r["relative_residual"] is None, r["relative_residual"] or 0))
    as_read = next(r for r in results if r["as_read"])
    defined = [r for r in results if r["relative_residual"] is not None]
    best = defined[0] if defined else None
    a = as_read["relative_residual"]
    b = best["relative_residual"] if best is not None else None
    ratio = a / b if a is not None and b is not None and b > 0 else None
    worse = a is not None and b is not None and a > tie_ratio * b
    findings = []
    if worse:
        findings.append(
            finding(
                "storage_order",
                "storage_order",
                "another reading has a smaller vacuum-Maxwell residual: axes "
                f"{best['axes_slowest_to_fastest']}, columns {best['columns_hold']}",
                f"as-read residual above {tie_ratio:g} x the lowest",
                as_read_over_best=ratio,
                as_read_relative_residual=a,
                best_relative_residual=b,
            )
        )
    reason = (
        "as-read derivative scale is zero; ordering is not identifiable by this diagnostic"
        if a is None
        else "lowest residual is exactly zero; a multiplicative ratio is undefined"
        if b == 0
        else None
    )
    return {
        "reader_assumed": assumed,
        "region": region.as_dict(),
        "candidates_tested": len(results),
        "as_read_relative_residual": a,
        "as_read_rank": results.index(as_read) + 1 if a is not None else None,
        "best": best,
        "as_read_over_best": ratio,
        "ratio_undefined_reason": reason,
        "identifiable_candidates": len(defined),
        "lowest": results[:report],
        "findings": findings,
        "reading": (
            "Compare only candidates with a defined derivative-normalized residual. "
            "Null scores/ratios are undefined, not zero error. A constant field cannot "
            "identify storage order. Ties can arise from physical symmetry; a smaller "
            "alternative residual is evidence to investigate the reader, not automatic "
            "permission to change its metadata. No ordering is certified from this test alone."
        ),
    }


def storage_order(
    field,
    region=None,
    report=5,
    tie_ratio=1.01,
    *,
    working_memory_bytes=DEFAULT_WORKING_BYTES,
    scratch_directory=None,
    backend="auto",
):
    if _bounded_requested(field, working_memory_bytes, backend, 2048):
        from element_surrogate.bounded_readiness import storage_order as blocked

        return blocked(
            field,
            region,
            report,
            tie_ratio,
            working_memory_bytes=working_memory_bytes,
            scratch_directory=scratch_directory,
        )
    return _eager_storage_order(field, region, report, tie_ratio)


def _eager_field_ends(
    field: FieldMap, region: Region | None = None, tolerance: float | None = None
) -> dict[str, Any]:
    """Whether the map holds its whole field along z: the field at the two end planes and
    the integral it misses beyond them.

    The profile is the largest field magnitude over each plane's nodes in ``region``
    (default: the whole plane). Beyond an end that is still decaying, the missing integral is
    extrapolated from the last two planes in two ways, exponential and power-law in the
    distance from the profile's centroid (a magnet's far field decays as a power law), and
    the larger is reported as a fraction of the profile's integral. With ``tolerance`` (the
    caller's accuracy target for integrals), an end is a finding when that fraction exceeds
    it, or, at an end that is not decaying, when its field exceeds ``tolerance`` of the peak.
    """
    mask = region.node_mask(field) if region is not None else np.ones(field.shape, dtype=bool)
    magnitude = np.where(mask, np.linalg.norm(field.b_t, axis=-1), -np.inf)
    profile = magnitude.max(axis=(0, 1))
    return _field_end_result(field, region, tolerance, profile)


def node_floor(field: FieldMap, region: Region | None = None) -> float | None:
    """The map's node-level floor: the median implied node error (see ``harmonic_nodes``)
    over the nodes at least two from every face and inside ``region``; None if there are none."""
    if min(field.shape) < 5:
        return None
    lap1, w1 = _laplacian(field.b_t, (field.x_m, field.y_m, field.z_m), 1)
    evaluated = np.zeros(field.shape, dtype=bool)
    evaluated[2:-2, 2:-2, 2:-2] = True
    if region is not None:
        evaluated &= region.node_mask(field)
    if not evaluated.any():
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        error = np.linalg.norm(lap1, axis=-1) / w1
    return float(np.median(error[evaluated]))


def _interpolation_error(other: FieldMap, region: Region | None) -> float | None:
    """RMS error of reading ``other`` by trilinear interpolation, estimated by reading it from
    every second node onto the nodes between and dividing by four (the error of a second-order
    reading falls as the spacing squared)."""
    if min(other.shape) < 3:
        return None
    coarse = other.window(slice(None, None, 2), slice(None, None, 2), slice(None, None, 2))
    keep = [
        values <= axis[-1] + 1e-12
        for values, axis in zip(
            (other.x_m, other.y_m, other.z_m), (coarse.x_m, coarse.y_m, coarse.z_m), strict=True
        )
    ]
    nodes = np.ones(other.shape, dtype=bool)
    nodes &= keep[0][:, None, None] & keep[1][None, :, None] & keep[2][None, None, :]
    on_coarse = np.zeros(other.shape, dtype=bool)
    on_coarse[::2, ::2, ::2] = True
    nodes &= ~on_coarse
    if region is not None:
        nodes &= region.node_mask(other)
    if not nodes.any():
        return None
    X, Y, Z = np.meshgrid(other.x_m, other.y_m, other.z_m, indexing="ij")
    points = np.stack([X[nodes], Y[nodes], Z[nodes]], axis=-1)
    error = coarse.sample(points) - other.b_t[nodes]
    return float(np.sqrt(np.mean(np.sum(error**2, axis=-1)))) / 4.0


def compare_exports(
    field: FieldMap,
    other: FieldMap,
    region: Region | None = None,
    tolerance: float | None = None,
    shells: int = 4,
) -> dict[str, Any]:
    """Compare two exports of one field solve (or two maps that should hold the same field)
    at the first map's nodes.

    Exports of one solve hold the same field, so where they differ, and whether their
    vacuum-Maxwell residuals agree, separates errors of the solve itself (shared) from
    errors of an export (in one only). ``other`` is read at ``field``'s nodes inside
    ``region`` (default: the largest circle about the z axis that ``field``'s grid covers)
    and inside ``other``'s grid: directly where the grids coincide, else by trilinear
    interpolation, whose own error is estimated and reported.

    Reported: the least-squares scale ``s`` minimizing ``|a - s b|`` (its sign is the
    relative polarity), the vector RMS difference with the polarity matched and with the
    best scale, per component, the largest difference and where it is, each map's node-level
    floor, and both maps' Maxwell residuals in shells of the region. Findings: an opposite
    polarity; with ``tolerance`` (relative), a scale beyond it, or a difference beyond it plus
    the interpolation error.
    """
    region = region if region is not None else Region.covering(field)
    # One grid when every node coincides to a millionth of the spacing (text exports print
    # coordinates to a few parts in 10^9 of a metre).
    same_grid = field.shape == other.shape and all(
        np.allclose(a, b, rtol=0.0, atol=1e-6 * float(np.min(np.diff(a))))
        for a, b in zip(
            (field.x_m, field.y_m, field.z_m), (other.x_m, other.y_m, other.z_m), strict=True
        )
    )
    mask = region.node_mask(field).copy()
    for axis, (mine, theirs) in enumerate(
        zip((field.x_m, field.y_m, field.z_m), (other.x_m, other.y_m, other.z_m), strict=True)
    ):
        inside = (mine >= theirs[0] - 1e-12) & (mine <= theirs[-1] + 1e-12)
        shape = [1, 1, 1]
        shape[axis] = mine.size
        mask &= inside.reshape(shape)
    if not mask.any():
        raise ValueError("no node of the first map lies inside the region and the other grid")
    a = field.b_t[mask]
    if same_grid:
        b = other.b_t[mask]
        interpolation = None
    else:
        X, Y, Z = np.meshgrid(field.x_m, field.y_m, field.z_m, indexing="ij")
        b = other.sample(np.stack([X[mask], Y[mask], Z[mask]], axis=-1))
        interpolation = _interpolation_error(other, region)
    scale = float(np.sum(a * b) / np.sum(b * b))
    polarity = 1.0 if scale >= 0.0 else -1.0
    rms_a = float(np.sqrt(np.mean(np.sum(a * a, axis=-1))))
    matched = a - polarity * b
    fitted = a - scale * b
    norms = np.linalg.norm(matched, axis=-1)
    worst = int(np.argmax(norms))
    where = np.argwhere(mask)[worst]
    vector_rms = float(np.sqrt(np.mean(norms**2)))
    best_rms = float(np.sqrt(np.mean(np.sum(fitted**2, axis=-1))))

    def residuals(m: FieldMap) -> list[dict[str, Any]] | None:
        try:
            report = maxwell_report(m, region, shells)
        except ValueError:
            return None
        return [
            {
                "outer_radius_m": r["outer_radius_m"],
                "curl_p99_relative": r["curl_t_per_m"]["p99_relative"],
            }
            for r in report["shells"]
        ]

    findings = []
    if polarity < 0:
        findings.append(
            finding(
                "compare_exports",
                "opposite_polarity",
                "the two maps hold the field with opposite signs",
                "sign of the least-squares scale",
                scale=scale,
            )
        )
    if tolerance is not None:
        if abs(abs(scale) - 1.0) > tolerance:
            findings.append(
                finding(
                    "compare_exports",
                    "scale_differs",
                    f"the maps differ by a factor {abs(scale):.6g} (another excitation or unit?)",
                    f"| |s| - 1 | above the stated tolerance {tolerance:g}",
                    scale=scale,
                )
            )
        if best_rms > tolerance * rms_a + (interpolation or 0.0):
            findings.append(
                finding(
                    "compare_exports",
                    "exports_differ",
                    f"after the best scale the maps differ by {best_rms / rms_a:.3g} of the RMS",
                    f"RMS difference above the stated tolerance {tolerance:g} of the field RMS "
                    "plus the interpolation error",
                )
            )
    return {
        "region": region.as_dict(),
        "same_grid": same_grid,
        "nodes_compared": int(mask.sum()),
        "reading_of_other": "its own nodes" if same_grid else "trilinear interpolation",
        "interpolation_error_rms_t": interpolation,
        "scale": scale,
        "polarity": polarity,
        "field_rms_t": rms_a,
        "difference_with_polarity_matched": {
            "vector_rms_t": vector_rms,
            "relative_to_field_rms": vector_rms / rms_a,
            "component_rms_t": np.sqrt(np.mean(matched**2, axis=0)).tolist(),
            "component_max_abs_t": np.max(np.abs(matched), axis=0).tolist(),
            "largest_at_m": [
                float(field.x_m[where[0]]),
                float(field.y_m[where[1]]),
                float(field.z_m[where[2]]),
            ],
        },
        "difference_with_best_scale_relative": best_rms / rms_a,
        "node_floor_t": {"field": node_floor(field, region), "other": node_floor(other, region)},
        "maxwell_shells": {"field": residuals(field), "other": residuals(other)},
        "findings": findings,
        "reading": (
            "Compare the difference with both floors and with the interpolation error: a "
            "difference far above all three is a real disagreement. Residuals that both maps "
            "share come from the solve; residuals in one map only come from its export."
        ),
    }


HARMONIC_METHOD = (
    "discrete Laplacian of each component on the map's own grid; "
    "implied node error = |Laplacian| / stencil weight; order = log2(|L_2h| / |L_h|)"
)
HARMONIC_INTERPRETATION = (
    "local harmonic consistency; cannot certify absolute scale, polarity or material clearance"
)
HARMONIC_READING = (
    "Judge nodes by ratio_to_floor, not by the sign of their order: when the floor itself "
    "is node-level error (median order below 0), every node reads negative. "
    "A cluster far above the floor is a region of wrong values (masked, missing, corrupted) "
    "or a material surface. For independent node errors of standard deviation s per component "
    "on a uniform grid, the median implied error is about 1.66 s."
)


def _node_defects_result(counts, examples_by_kind, duplicated):
    findings = []
    observations = []
    if counts[0]:
        findings.append(
            finding(
                "node_defects",
                "non_finite_values",
                f"{counts[0]} node(s) hold NaN or infinity",
                "non-finite values do not define a finite magnetic field for tracking",
                examples=[list(p) for p in examples_by_kind[0]],
            )
        )
    if counts[1]:
        observations.append(
            finding(
                "node_defects",
                "zero_vectors",
                f"{counts[1]} zero vector(s) next to non-zero nodes",
                (
                    "context required: a symmetry zero, zeroed region or missing "
                    "sample can give this pattern"
                ),
                examples=[list(p) for p in examples_by_kind[1]],
            )
        )
    for item in duplicated:
        observations.append(
            finding(
                "node_defects",
                "duplicated_plane",
                f"planes {item['planes']} along {item['axis']} are identical",
                (
                    "context required: translational invariance, symmetry or dupl"
                    "icated export values can give identical planes"
                ),
                **item,
            )
        )
    return {
        "non_finite_nodes": {
            "count": int(counts[0]),
            "examples": [list(p) for p in examples_by_kind[0]],
        },
        "zero_nodes_next_to_nonzero": {
            "count": int(counts[1]),
            "examples": [list(p) for p in examples_by_kind[1]],
        },
        "duplicated_planes": duplicated,
        "findings": findings,
        "observations": observations,
    }


def _field_end_result(field, region, tolerance, profile):
    planes = np.isfinite(profile)
    z, p = field.z_m[planes], profile[planes]
    if z.size < 3:
        raise ValueError("the region holds fewer than three planes")
    peak = float(p.max())
    if not peak > 0.0:
        # This is a measured result, not missing computation. Relative tail and
        # centroid statistics are undefined; source/description review must say
        # whether zero is intended. A zero profile cannot establish the field
        # outside this grid or distinguish switched-off from missing field data.
        return {
            "region": region.as_dict() if region is not None else None,
            "profile": "largest |B| over each plane's nodes in the region",
            "peak_t": 0.0,
            "centroid_z_m": None,
            "ends": {
                name: {
                    "z_m": float(z[index]),
                    "field_t": 0.0,
                    "fraction_of_peak": None,
                    "decaying": False,
                    "missing_integral_fraction": None,
                }
                for name, index in (("low_z", 0), ("high_z", -1))
            },
            "tolerance": tolerance,
            "findings": [],
            "observations": [
                finding(
                    "field_ends",
                    "zero_profile",
                    "the evaluated field profile is identically zero",
                    "largest field magnitude is zero in every evaluated plane; "
                    "relative tail and centroid are undefined",
                )
            ],
            "reading": (
                "Zero measured field is contextual: establish intended zero versus missing "
                "field from source/description evidence. No outside-field or tail certificate."
            ),
        }
    total = float(np.sum(0.5 * (p[1:] + p[:-1]) * np.diff(z)))
    centroid = float(np.sum(0.5 * (p[1:] * z[1:] + p[:-1] * z[:-1]) * np.diff(z)) / total)

    ends = {}
    findings = []
    for name, last, previous in (("low_z", 0, 1), ("high_z", -1, -2)):
        p_end, p_prev = float(p[last]), float(p[previous])
        d_end, d_prev = abs(z[last] - centroid), abs(z[previous] - centroid)
        decaying = 0.0 < p_end < p_prev
        tail = None
        if decaying:
            exponential = p_end * abs(z[last] - z[previous]) / math.log(p_prev / p_end)
            power = math.log(p_prev / p_end) / math.log(d_end / d_prev)
            algebraic = p_end * d_end / (power - 1.0) if power > 1.0 else math.inf
            tail = max(exponential, algebraic)
        fraction = None if tail is None else tail / total
        ends[name] = {
            "z_m": float(z[last]),
            "field_t": p_end,
            "fraction_of_peak": p_end / peak,
            "decaying": decaying,
            "missing_integral_fraction": fraction,
        }
        if tolerance is None:
            continue
        unbounded = fraction is None or math.isinf(fraction)
        if (not unbounded and fraction is not None and fraction > tolerance) or (
            unbounded and p_end / peak > tolerance
        ):
            findings.append(
                finding(
                    "field_ends",
                    "truncated",
                    f"the map ends at z = {z[last]:.6g} m before its field has decayed",
                    f"missing integral (or, if not decaying, end field) above {tolerance:g} of "
                    "the integral (or peak)",
                    end=name,
                    fraction_of_peak=p_end / peak,
                    missing_integral_fraction=fraction,
                )
            )
    return {
        "region": region.as_dict() if region is not None else None,
        "profile": "largest |B| over each plane's nodes in the region",
        "peak_t": peak,
        "centroid_z_m": centroid,
        "ends": ends,
        "tolerance": tolerance,
        "findings": findings,
        "reading": (
            "missing_integral_fraction estimates what the map misses beyond each end; judge it "
            "against the accuracy the surrogate needs. Without a tolerance nothing is flagged."
        ),
    }


def _bounded_requested(field, working_memory_bytes, backend, bytes_per_node):
    validate_budget(working_memory_bytes)
    if backend not in {"auto", "bounded", "eager"}:
        raise ValueError("diagnostic backend must be auto, bounded or eager")
    estimated = math.prod(field.shape) * bytes_per_node
    return backend == "bounded" or (backend == "auto" and estimated > working_memory_bytes)


def maxwell_report(
    field,
    region=None,
    shells=10,
    *,
    working_memory_bytes=DEFAULT_WORKING_BYTES,
    scratch_directory=None,
    backend="auto",
):
    """Raw finite-volume Maxwell evidence; exact blocked execution for large grids.

    working_memory_bytes bounds numerical working arrays, not mapped pages/RSS.
    Disk scratch defaults to the caller's current project directory. Source values,
    grid spacing, shell inclusion and global linear quantiles remain unchanged.
    """
    if _bounded_requested(field, working_memory_bytes, backend, 2048):
        from element_surrogate.bounded_readiness import maxwell_report as blocked

        return blocked(
            field,
            region,
            shells,
            working_memory_bytes=working_memory_bytes,
            scratch_directory=scratch_directory,
        )
    return _eager_maxwell_report(field, region, shells)


def harmonic_nodes(
    field,
    region=None,
    top=100,
    examples=10,
    fence_iqr=3.0,
    *,
    working_memory_bytes=DEFAULT_WORKING_BYTES,
    scratch_directory=None,
    backend="auto",
):
    """Harmonic-node evidence with exact disk statistics and cross-block components.

    Equal-error examples use global node-index order, independently of block size.
    Only evaluated nodes are ranked. The working budget is not a fixed RSS cap.
    """
    if _bounded_requested(field, working_memory_bytes, backend, 4096):
        from element_surrogate.bounded_readiness import harmonic_nodes as blocked

        return blocked(
            field,
            region,
            top,
            examples,
            fence_iqr,
            working_memory_bytes=working_memory_bytes,
            scratch_directory=scratch_directory,
        )
    return _eager_harmonic_nodes(field, region, top, examples, fence_iqr)


def node_defects(
    field,
    max_examples=10,
    *,
    working_memory_bytes=DEFAULT_WORKING_BYTES,
    scratch_directory=None,
    backend="auto",
):
    """All-node integrity counts/contextual observations with bounded examples."""
    if _bounded_requested(field, working_memory_bytes, backend, 512):
        from element_surrogate.bounded_readiness import node_defects as blocked

        return blocked(
            field,
            max_examples,
            working_memory_bytes=working_memory_bytes,
            scratch_directory=scratch_directory,
        )
    return _eager_node_defects(field, max_examples)


def field_ends(
    field,
    region=None,
    tolerance=None,
    *,
    working_memory_bytes=DEFAULT_WORKING_BYTES,
    scratch_directory=None,
    backend="auto",
):
    """The same end-profile/tail evidence, reducing the profile block by block."""
    if _bounded_requested(field, working_memory_bytes, backend, 128):
        from element_surrogate.bounded_readiness import field_ends as blocked

        return blocked(
            field,
            region,
            tolerance,
            working_memory_bytes=working_memory_bytes,
            scratch_directory=scratch_directory,
        )
    return _eager_field_ends(field, region, tolerance)
