"""Exact full-resolution diagnostics with bounded vector working arrays."""

from __future__ import annotations

import heapq
import itertools
import math
import sqlite3

import numpy as np

from element_surrogate import readiness as ref
from element_surrogate.diagnostic_blocks import Workspace, expanded, node_window
from element_surrogate.fields import FieldMap
from element_surrogate.region import Region


def _cell_window(field, core):
    return field.window(*(slice(s.start, s.stop + 1) for s in core))


def _stat_append(series, values):
    series.append(np.abs(values[np.isfinite(values)]))


def _node_derivatives(field, core, order):
    derivatives = []
    for axis, values in enumerate((field.x_m, field.y_m, field.z_m)):

        def shifted(k, axis=axis):
            index = list(core)
            s = core[axis]
            index[axis] = slice(s.start + k, s.stop + k)
            return field.b_t[tuple(index)]

        if order == 2:
            s = core[axis]
            span = values[s.start + 1 : s.stop + 1] - values[s.start - 1 : s.stop - 1]
            shape = [1, 1, 1, 1]
            shape[axis] = span.size
            derivatives.append((shifted(1) - shifted(-1)) / span.reshape(shape))
        else:
            h = float(values[1] - values[0])
            derivatives.append(
                (-shifted(2) + 8 * shifted(1) - 8 * shifted(-1) + shifted(-2)) / (12 * h)
            )
    x, y, z = derivatives
    return (
        np.abs(x[..., 0] + y[..., 1] + z[..., 2]),
        np.sqrt(
            (y[..., 2] - z[..., 1]) ** 2
            + (z[..., 0] - x[..., 2]) ** 2
            + (x[..., 1] - y[..., 0]) ** 2
        ),
    )


def maxwell_report(field, region, shells, *, working_memory_bytes, scratch_directory):
    if type(shells) is not int or shells < 1:
        raise ValueError("shells must be at least 1")
    region = region if region is not None else Region.covering(field)
    with Workspace(working_memory_bytes, scratch_directory) as work:
        edges = np.linspace(0.0, region.radius_m, shells + 1)
        whole = [work.scalar_series(), work.scalar_series()]
        fine = [[work.scalar_series() for _ in range(2)] for _ in range(shells)]
        scale_count, scale_sum, compensation = 0, 0.0, 0.0
        for core in work.cores(tuple(n - 1 for n in field.shape), bytes_per_node=2048):
            window = _cell_window(field, core)
            inside = region.cell_mask(window)
            if not inside.any():
                continue
            result = ref.cell_residuals(window)
            norm = result.derivative_norm_t_per_m[inside]
            norm = norm[np.isfinite(norm)]
            term = float(np.sum(norm * norm)) - compensation
            total = scale_sum + term
            compensation = (total - scale_sum) - term
            scale_sum = total
            scale_count += int(norm.size)
            div, curl = result.divergence_t_per_m, result.curl_t_per_m
            _stat_append(whole[0], div[inside])
            _stat_append(whole[1], curl[inside])
            radial = region.cell_radial(window)
            for k, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
                lower = radial >= low if k == 0 else radial > low
                mask = inside & lower & (radial <= high * (1 + 1e-12))
                _stat_append(fine[k][0], div[mask])
                _stat_append(fine[k][1], curl[mask])
        if not whole[0].count:
            raise ValueError("no cell lies inside the region")
        scale = math.sqrt(scale_sum / scale_count) if scale_count else math.nan
        rows = [
            {
                "outer_radius_m": float(high),
                "divergence_t_per_m": a.stats(scale),
                "curl_t_per_m": b.stats(scale),
                "coarsened_by_2": None,
                "higher_order_stencil": None,
            }
            for high, (a, b) in zip(edges[1:], fine, strict=True)
        ]

        coarse_grid = ref.coarsen(field)
        if min(coarse_grid.shape) >= 2:
            # coarse, mean-of-eight fine children, and every fine cell in that volume
            groups = [[[work.scalar_series() for _ in range(3)] for _ in range(2)] for _ in rows]
            counts = [0] * shells
            for core in work.cores(tuple(n - 1 for n in coarse_grid.shape), bytes_per_node=16384):
                window = _cell_window(coarse_grid, core)
                inside = region.cell_mask(window)
                if not inside.any():
                    continue
                coarse = ref.cell_residuals(window)
                fwindow = field.window(*(slice(2 * s.start, 2 * s.stop + 1) for s in core))
                child = ref.cell_residuals(fwindow)
                mean_div = ref._child_mean(
                    child.divergence_t_per_m, coarse.divergence_t_per_m.shape
                )
                mean_curl = np.linalg.norm(
                    ref._child_mean(child.curl_vector_t_per_m, coarse.divergence_t_per_m.shape),
                    axis=-1,
                )
                radial = region.cell_radial(window)
                for k, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
                    lower = radial >= low if k == 0 else radial > low
                    selected = region.cell_mask(window, outer_m=high) & lower
                    if not selected.any():
                        continue
                    counts[k] += int(selected.sum())
                    children = ref._children(selected, child.divergence_t_per_m.shape)
                    for c, (cvalues, average, fvalues) in enumerate(
                        (
                            (coarse.divergence_t_per_m, mean_div, child.divergence_t_per_m),
                            (coarse.curl_t_per_m, mean_curl, child.curl_t_per_m),
                        )
                    ):
                        _stat_append(groups[k][c][0], cvalues[selected])
                        _stat_append(groups[k][c][1], average[selected])
                        _stat_append(groups[k][c][2], fvalues[children])
            for row, count, metrics in zip(rows, counts, groups, strict=True):
                order = {"coarse_cells": count}
                for key, (cvalues, average, fvalues) in zip(
                    ("divergence", "curl"), metrics, strict=True
                ):
                    cp = cvalues.stats()["p99"]
                    order[key] = {
                        "pointwise_order": ref._order(average.stats()["p99"], cp),
                        "same_volume_order": ref._order(fvalues.stats()["p99"], cp),
                    }
                row["coarsened_by_2"] = order

        uniform = all(
            np.allclose(np.diff(a), a[1] - a[0], rtol=1e-9, atol=0.0)
            for a in (field.x_m, field.y_m, field.z_m)
        )
        if min(field.shape) >= 5 and uniform:
            nodal = [[[work.scalar_series() for _ in range(2)] for _ in range(2)] for _ in rows]
            counts = [0] * shells
            for core in work.cores(
                field.shape,
                lower=(2, 2, 2),
                upper=tuple(n - 2 for n in field.shape),
                bytes_per_node=2048,
            ):
                window = node_window(field, core)
                inside = region.node_mask(window)
                if not inside.any():
                    continue
                second, fourth = (
                    _node_derivatives(field, core, 2),
                    _node_derivatives(field, core, 4),
                )
                radial, along = region.node_measures(window)
                for k, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
                    lower = radial >= low if k == 0 else radial > low
                    selected = np.broadcast_to(
                        along & lower & (radial <= high * (1 + 1e-12)), window.shape
                    )
                    counts[k] += int(selected.sum())
                    for c in range(2):
                        _stat_append(nodal[k][c][0], second[c][selected])
                        _stat_append(nodal[k][c][1], fourth[c][selected])
            for row, count, metrics in zip(rows, counts, nodal, strict=True):
                ratio = {"nodes": count}
                for key, (second, fourth) in zip(("divergence", "curl"), metrics, strict=True):
                    p2, p4 = second.stats()["p99"], fourth.stats()["p99"]
                    ratio[f"{key}_p99_fourth_over_second"] = p4 / p2 if p2 else None
                row["higher_order_stencil"] = ratio
        return {
            "method": ref.METHOD,
            "region": region.as_dict(),
            "derivative_scale_rms_t_per_m": scale,
            "whole_region": {
                "divergence_t_per_m": whole[0].stats(scale),
                "curl_t_per_m": whole[1].stats(scale),
            },
            "shells": rows,
            "reading_the_order": ref.ORDER_READING,
            "reading_the_stencil": ref.STENCIL_READING,
            "reading_relative": (
                "p99_relative divides by derivative_scale_rms_t_per_m over the region"
            ),
            "execution": work.description(),
        }


def _laplacian_at(field, core, step):
    shape = tuple(s.stop - s.start for s in core)
    lap, weight = np.zeros(shape + (3,)), np.zeros(shape)
    centre = field.b_t[core]
    for axis, values in enumerate((field.x_m, field.y_m, field.z_m)):
        s = core[axis]
        plus, minus = list(core), list(core)
        plus[axis], minus[axis] = (
            slice(s.start + step, s.stop + step),
            slice(s.start - step, s.stop - step),
        )
        span = [1, 1, 1]
        span[axis] = s.stop - s.start
        hp = (values[s.start + step : s.stop + step] - values[s]).reshape(span)
        hm = (values[s] - values[s.start - step : s.stop - step]).reshape(span)
        second = (
            2.0
            * (
                field.b_t[tuple(plus)] * hm[..., None]
                - centre * (hp + hm)[..., None]
                + field.b_t[tuple(minus)] * hp[..., None]
            )
            / ((hp * hm * (hp + hm))[..., None])
        )
        lap += second
        weight += np.broadcast_to(2.0 / (hp * hm), shape)
    return lap, weight


class _DiskComponents:
    """26-neighbor components with disk union/find, bounded SQLite page cache."""

    def __init__(self, work, shape):
        self.shape = shape
        self.path = work.path / "components.sqlite"
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=OFF")
        self.db.execute("PRAGMA synchronous=OFF")  # only disposable scratch, never evidence
        self.db.execute(f"PRAGMA cache_size=-{max(64, min(16384, work.budget // 16384))}")
        self.db.execute(
            "CREATE TABLE nodes(id INTEGER PRIMARY KEY,parent INTEGER,"
            "i INTEGER,j INTEGER,k INTEGER,r REAL,e REAL)"
        )
        self.count = 0

    def root(self, key):
        while True:
            parent = self.db.execute("SELECT parent FROM nodes WHERE id=?", (key,)).fetchone()[0]
            if parent == key:
                return key
            grand = self.db.execute("SELECT parent FROM nodes WHERE id=?", (parent,)).fetchone()[0]
            self.db.execute("UPDATE nodes SET parent=? WHERE id=?", (grand, key))
            key = parent

    def append(self, positions, ratios, errors):
        sy, sz = self.shape[1:]
        for (i, j, k), ratio, error in zip(positions, ratios, errors, strict=True):
            i, j, k = int(i), int(j), int(k)
            key = (i * sy + j) * sz + k
            self.db.execute(
                "INSERT INTO nodes VALUES(?,?,?,?,?,?,?)",
                (key, key, i, j, k, float(ratio), float(error)),
            )
            neighbors = [
                ((i + a) * sy + j + b) * sz + k + c
                for a, b, c in itertools.product((-1, 0, 1), repeat=3)
                if (a, b, c) != (0, 0, 0)
            ]
            # Evaluated harmonic nodes are at least two from every global face.
            present = self.db.execute(
                "SELECT id FROM nodes WHERE id IN (" + ",".join("?" for _ in neighbors) + ")",
                neighbors,
            ).fetchall()
            roots = {key, *(self.root(row[0]) for row in present)}
            root = min(roots)
            for item in roots:
                if item != root:
                    self.db.execute("UPDATE nodes SET parent=? WHERE id=?", (root, item))
            self.count += 1

    def summarize(self, examples):
        self.db.execute(
            "CREATE TABLE groups(id INTEGER PRIMARY KEY,n INTEGER,"
            "i0 INTEGER,j0 INTEGER,k0 INTEGER,i1 INTEGER,j1 INTEGER,k1 INTEGER,r REAL,e REAL)"
        )
        for key, i, j, k, ratio, error in self.db.execute("SELECT id,i,j,k,r,e FROM nodes"):
            root = self.root(key)
            self.db.execute(
                """INSERT INTO groups VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET n=n+1,
                i0=min(i0,excluded.i0),j0=min(j0,excluded.j0),k0=min(k0,excluded.k0),
                i1=max(i1,excluded.i1),j1=max(j1,excluded.j1),k1=max(k1,excluded.k1),r=max(r,excluded.r),e=max(e,excluded.e)""",
                (root, 1, i, j, k, i, j, k, ratio, error),
            )
        count = self.db.execute("SELECT count(*) FROM groups").fetchone()[0]
        best = []
        for row in self.db.execute("SELECT * FROM groups"):
            key = (row[8], -row[0])
            if len(best) < examples:
                heapq.heappush(best, (key, row))
            elif examples and key > best[0][0]:
                heapq.heapreplace(best, (key, row))
        return count, [row for _, row in sorted(best, reverse=True)]

    def close(self):
        self.db.close()


def harmonic_nodes(
    field, region, top, examples, fence_iqr, *, working_memory_bytes, scratch_directory
):
    if min(field.shape) < 5:
        raise ValueError("the grid needs at least five nodes along every axis")
    if any(type(v) is not int or v < 0 for v in (top, examples)):
        raise ValueError("nonnegative top/examples required")
    if max(top, examples) * 256 > working_memory_bytes:
        raise ValueError("top/examples exceed the working-memory budget")
    with Workspace(working_memory_bytes, scratch_directory) as work:
        error, order, local = (work.mapped(field.shape) for _ in range(3))
        for array in (error, order, local):
            array.fill(np.nan)
        values, orders, logs = (work.scalar_series() for _ in range(3))
        ranked = []
        keep = max(top, examples)
        bounds = {"lower": (2, 2, 2), "upper": tuple(n - 2 for n in field.shape)}
        for core in work.cores(field.shape, bytes_per_node=2048, **bounds):
            window = node_window(field, core)
            valid = np.ones(window.shape, bool) if region is None else region.node_mask(window)
            if not valid.any():
                continue
            lap1, weight = _laplacian_at(field, core, 1)
            lap2, _ = _laplacian_at(field, core, 2)
            l1, l2 = np.linalg.norm(lap1, axis=-1), np.linalg.norm(lap2, axis=-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                err = np.where(valid, l1 / weight, np.nan)
                power = np.where(valid, np.log2(l2 / l1), np.nan)
            error[core], order[core] = err, power
            values.append(err[valid])
            orders.append(power[valid & np.isfinite(power)])
            if keep:
                pos = np.argwhere(valid)
                pos += np.array([s.start for s in core])
                ids = np.ravel_multi_index(pos.T, field.shape)
                errs = err[valid]
                chosen = np.lexsort((ids, -errs))[:keep]
                ranked = sorted(
                    ranked + [(float(errs[q]), int(ids[q])) for q in chosen],
                    key=lambda x: (-x[0], x[1]),
                )[:keep]
        if not values.count:
            raise ValueError("no node is evaluated: the region is empty or too close to the faces")
        floor, p99, p999 = map(float, values.quantiles([0.5, 0.99, 0.999]))
        floor_order = float(orders.quantiles(0.5)) if orders.count else math.nan
        for core in work.cores(field.shape, bytes_per_node=8192, **bounds):
            err = error[core]
            finite = np.isfinite(err)
            if not finite.any():
                continue
            outer = expanded(core, field.shape, 2)
            neighborhood = np.lib.stride_tricks.sliding_window_view(error[outer], (5, 5, 5))
            med = np.full(err.shape, np.nan)
            with np.errstate(all="ignore"):
                med[finite] = np.nanmedian(neighborhood[finite].reshape(-1, 125), axis=1)
                ratio = err / med
            local[core] = ratio
            baseline = finite & np.isfinite(ratio) & (ratio > 0.0)
            logs.append(np.log10(ratio[baseline]))
        if logs.count:
            q1, q3 = logs.quantiles([0.25, 0.75])
            with np.errstate(over="ignore"):
                fence = float(10.0 ** (q3 + fence_iqr * (q3 - q1)))
        else:
            fence = math.inf
        ranked_nodes = [
            tuple(int(v) for v in np.unravel_index(key, field.shape)) for _, key in ranked
        ]
        axes = (field.x_m, field.y_m, field.z_m)

        def describe(node):
            return {
                "node": list(node),
                "position_m": [float(a[i]) for a, i in zip(axes, node, strict=True)],
                "implied_error_t": float(error[node]),
                "ratio_to_floor": float(error[node] / floor) if floor > 0 else None,
                "local_ratio": float(local[node]) if np.isfinite(local[node]) else None,
                "node_field_t": float(np.linalg.norm(field.b_t[node])),
                "order": float(order[node]) if np.isfinite(order[node]) else None,
            }

        clusters = []
        for group in ref._clusters(ranked_nodes[:top]):
            idx = np.array(group)
            lo, hi = idx.min(axis=0), idx.max(axis=0)
            worst = max(float(error[n]) for n in group)
            clusters.append(
                {
                    "nodes": len(group),
                    "node_range": [lo.tolist(), hi.tolist()],
                    "position_range_m": [
                        [float(a[i]) for a, i in zip(axes, lo, strict=True)],
                        [float(a[i]) for a, i in zip(axes, hi, strict=True)],
                    ],
                    "max_implied_error_t": worst,
                    "max_ratio_to_floor": worst / floor if floor > 0 else None,
                }
            )
        clusters.sort(key=lambda c: c["max_implied_error_t"], reverse=True)
        components = _DiskComponents(work, field.shape)
        try:
            for core in work.cores(field.shape, bytes_per_node=1024, **bounds):
                ratio, power = local[core], order[core]
                flagged = (ratio > fence) & (power < 0.0)
                if not flagged.any():
                    continue
                pos = np.argwhere(flagged) + np.array([s.start for s in core])
                components.append(pos, ratio[flagged], error[core][flagged])
            cluster_count, groups = components.summarize(examples)
            flagged_count = components.count
        finally:
            components.close()
        work.account_existing((work.path / "components.sqlite").stat().st_size)
        criterion = (
            f"implied error above {fence:.3g} x the median of the nodes within two (the far-out "
            f"fence, {fence_iqr:g} interquartile ranges above the upper quartile of the map's own "
            "local ratios in log scale) and order below 0"
        )
        findings = []
        for _, count, i0, j0, k0, i1, j1, k1, ratio, worst in groups:
            lo, hi = (i0, j0, k0), (i1, j1, k1)
            findings.append(
                ref.finding(
                    "harmonic_nodes",
                    "wrong_values_or_material_surface",
                    f"{count} touching node(s) far above the map's floor",
                    criterion,
                    node_range=[list(lo), list(hi)],
                    position_range_m=[
                        [float(a[i]) for a, i in zip(axes, lo, strict=True)],
                        [float(a[i]) for a, i in zip(axes, hi, strict=True)],
                    ],
                    max_local_ratio=float(ratio),
                    max_ratio_to_floor=float(worst / floor) if floor > 0 else None,
                )
            )
        return {
            "method": ref.HARMONIC_METHOD,
            "region": region.as_dict() if region is not None else None,
            "evaluated_nodes": values.count,
            "absolute_field_uncertainty_calibrated": False,
            "interpretation": ref.HARMONIC_INTERPRETATION,
            "floor": {
                "median_implied_error_t": floor,
                "median_order": floor_order,
                "regime": "order not identifiable from zero/non-finite residual ratios"
                if not math.isfinite(floor_order)
                else "node-level error in the data"
                if floor_order < 0
                else "truncation of a smooth field",
            },
            "implied_error_t": {"p99": p99, "p999": p999, "max": values.maximum},
            "worst_nodes": [describe(n) for n in ranked_nodes[:examples]],
            "clusters_of_top_nodes": {"top": min(top, len(ranked_nodes)), "clusters": clusters},
            "local_ratio_fence": fence,
            "flagged_nodes": flagged_count,
            "flagged_clusters": cluster_count,
            "findings": findings,
            "reading": ref.HARMONIC_READING,
            "execution": work.description(),
        }


def node_defects(field, max_examples, *, working_memory_bytes, scratch_directory):
    with Workspace(working_memory_bytes, scratch_directory) as work:
        counts = [0, 0]
        examples = [[], []]
        same = [np.ones(n - 1, bool) for n in field.shape]
        nonzero = [np.zeros(n - 1, bool) for n in field.shape]
        for core in work.cores(field.shape, bytes_per_node=1024):
            b = field.b_t[core]
            bad, zero = ~np.isfinite(b).all(axis=-1), ~np.any(b, axis=-1)
            all_zero_neighbors = np.ones(zero.shape, bool)
            for axis in range(3):
                for delta in (-1, 1):
                    source = list(core)
                    s = core[axis]
                    lo, hi = max(s.start, -delta), min(s.stop, field.shape[axis] - delta)
                    if hi <= lo:
                        continue
                    source[axis] = slice(lo + delta, hi + delta)
                    select = [slice(None)] * 3
                    select[axis] = slice(lo - s.start, hi - s.start)
                    all_zero_neighbors[tuple(select)] &= ~np.any(field.b_t[tuple(source)], axis=-1)
                s = core[axis]
                hi = min(s.stop, field.shape[axis] - 1)
                if hi > s.start:
                    left, right = list(core), list(core)
                    left[axis] = slice(s.start, hi)
                    right[axis] = slice(s.start + 1, hi + 1)
                    reduce = tuple(a for a in range(3) if a != axis)
                    equal = (
                        np.equal(field.b_t[tuple(left)], field.b_t[tuple(right)])
                        .all(axis=-1)
                        .all(axis=reduce)
                    )
                    active = np.any(field.b_t[tuple(left)], axis=-1).any(axis=reduce)
                    same[axis][s.start : hi] &= equal
                    nonzero[axis][s.start : hi] |= active
            for q, mask in enumerate((bad, zero & ~all_zero_neighbors)):
                counts[q] += int(mask.sum())
                pos = np.argwhere(mask) + np.array([s.start for s in core])
                examples[q] = sorted(
                    examples[q] + [tuple(map(int, p)) for p in pos[:max_examples]]
                )[:max_examples]
        duplicated = [
            {"axis": "xyz"[axis], "planes": [int(i), int(i + 1)]}
            for axis in range(3)
            for i in np.flatnonzero(same[axis] & nonzero[axis])
        ]
        result = ref._node_defects_result(counts, examples, duplicated)
        result["execution"] = work.description()
        return result


def field_ends(field, region, tolerance, *, working_memory_bytes, scratch_directory):
    with Workspace(working_memory_bytes, scratch_directory) as work:
        profile = np.full(field.shape[2], -np.inf)
        for core in work.cores(field.shape, bytes_per_node=256):
            window = node_window(field, core)
            mask = region.node_mask(window) if region is not None else np.ones(window.shape, bool)
            p = np.where(mask, np.linalg.norm(window.b_t, axis=-1), -np.inf).max(axis=(0, 1))
            profile[core[2]] = np.maximum(profile[core[2]], p)
        result = ref._field_end_result(field, region, tolerance, profile)
        result["execution"] = work.description()
        return result


def _renested_window(field, assumed, nesting, core):
    """Gather exactly the requested source indices without whole-source renesting."""
    indices = np.meshgrid(*(np.arange(s.start, s.stop) for s in core), indexing="ij")
    dims = tuple(field.shape[a] for a in nesting)
    offset = np.ravel_multi_index(tuple(indices[a] for a in nesting), dims)
    source = np.unravel_index(offset, tuple(field.shape[a] for a in assumed))
    physical = tuple(source[assumed.index(a)] for a in range(3))
    axes = [a[s] for a, s in zip((field.x_m, field.y_m, field.z_m), core, strict=True)]
    return FieldMap(*axes, field.b_t[physical], field.label, field.source)


def storage_order(field, region, report, tie_ratio, *, working_memory_bytes, scratch_directory):
    """Exhaustive, exact medians; no full-source permutation or sampled substitute."""
    region = region if region is not None else Region.covering(field)
    assumed = (
        field.source.assumed_storage
        if field.source is not None
        else {"axes_slowest_to_fastest": list(ref._AXES), "components": list(ref._COMPONENTS)}
    )
    base = tuple(ref._AXES.index(a) for a in assumed["axes_slowest_to_fastest"])
    window = region.node_window(field)
    lower = tuple(s.start for s in window)
    upper = tuple(s.stop - 1 for s in window)
    if any(a >= b for a, b in zip(lower, upper, strict=True)):
        raise ValueError("storage-order region must contain cells")
    results = []
    with Workspace(working_memory_bytes, scratch_directory) as work:
        for nesting in itertools.permutations(range(3)):
            jac = work.scalar_series()
            norms = work.scalar_series()
            for core in work.cores(field.shape, lower=lower, upper=upper, bytes_per_node=4096):
                nodes = tuple(slice(s.start, s.stop + 1) for s in core)
                candidate = _renested_window(field, base, nesting, nodes)
                inside = region.cell_mask(candidate)
                if not inside.any():
                    continue
                values = np.stack(ref.cell_derivatives(candidate), axis=-1)[inside]
                jac.append(values)
                norms.append(np.sqrt(np.sum(values**2, axis=(1, 2))))
            if not norms.count:
                raise ValueError("no cell lies inside storage-order region")
            scale = float(norms.quantiles(0.5))
            norms.discard()
            jac.file.flush()
            mapped = np.memmap(jac.path, mode="r", dtype=np.float64, shape=(jac.count // 9, 3, 3))
            try:
                for components in itertools.permutations(range(3)):
                    for signs in ref._SIGN_CLASSES:
                        div, curl = work.scalar_series(), work.scalar_series()
                        for begin in range(0, len(mapped), max(1, work.budget // 1024)):
                            values = mapped[begin : begin + max(1, work.budget // 1024)]
                            j = np.empty_like(values)
                            for column, component in enumerate(components):
                                j[:, component, :] = signs[column] * values[:, column, :]
                            div.append(np.abs(j[:, 0, 0] + j[:, 1, 1] + j[:, 2, 2]))
                            curl.append(
                                np.sqrt(
                                    (j[:, 2, 1] - j[:, 1, 2]) ** 2
                                    + (j[:, 0, 2] - j[:, 2, 0]) ** 2
                                    + (j[:, 1, 0] - j[:, 0, 1]) ** 2
                                )
                            )
                        worst = max(float(div.quantiles(0.5)), float(curl.quantiles(0.5)))
                        results.append(
                            {
                                "axes_slowest_to_fastest": [ref._AXES[a] for a in nesting],
                                "columns_hold": [
                                    ("-" if s < 0 else "") + ref._COMPONENTS[c]
                                    for c, s in zip(components, signs, strict=True)
                                ],
                                "relative_residual": worst / scale if scale > 0 else None,
                                "as_read": nesting == base
                                and components == (0, 1, 2)
                                and signs == (1, 1, 1),
                            }
                        )
                        div.discard()
                        curl.discard()
            finally:
                mapped._mmap.close()
            jac.discard()
        result = ref._storage_order_result(results, assumed, region, report, tie_ratio)
        result["execution"] = work.description()
        return result
