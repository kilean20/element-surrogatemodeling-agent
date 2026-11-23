"""Bounded working arrays and owned disk scratch for exact grid diagnostics.

The budget controls numerical working arrays, not total RSS: source/scratch
file pages remain OS-managed and reclaimable. Scratch defaults to the caller's
project directory rather than an implicitly RAM-backed system temporary area.
"""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

DEFAULT_WORKING_BYTES = 256 * 1024**2
MIN_WORKING_BYTES = 256 * 1024


def node_window(field, core):
    """Coordinate-only view for region masks; singleton core axes are allowed.

    Never use this view for interpolation/differentiation: a FieldMap retains
    its requirement for at least two nodes on every axis.
    """
    axes = [
        axis[index] for axis, index in zip((field.x_m, field.y_m, field.z_m), core, strict=True)
    ]
    return SimpleNamespace(
        x_m=axes[0],
        y_m=axes[1],
        z_m=axes[2],
        shape=tuple(len(a) for a in axes),
        b_t=field.b_t[core],
    )


def validate_budget(working_memory_bytes):
    if type(working_memory_bytes) is not int or working_memory_bytes < MIN_WORKING_BYTES:
        raise ValueError(f"working_memory_bytes must be an integer >= {MIN_WORKING_BYTES}")
    return working_memory_bytes


def blocks(shape, max_nodes, *, lower=None, upper=None):
    """Disjoint core boxes; preserve contiguous runs along the last storage axis."""
    low = (0, 0, 0) if lower is None else tuple(lower)
    high = tuple(shape) if upper is None else tuple(upper)
    size = tuple(b - a for a, b in zip(low, high, strict=True))
    if min(size) <= 0:
        return
    z = min(size[2], max(1, max_nodes))
    y = min(size[1], max(1, max_nodes // z))
    x = min(size[0], max(1, max_nodes // (y * z)))
    for i in range(low[0], high[0], x):
        for j in range(low[1], high[1], y):
            for k in range(low[2], high[2], z):
                yield (
                    slice(i, min(i + x, high[0])),
                    slice(j, min(j + y, high[1])),
                    slice(k, min(k + z, high[2])),
                )


def expanded(core, shape, halo):
    return tuple(
        slice(max(0, s.start - halo), min(n, s.stop + halo))
        for s, n in zip(core, shape, strict=True)
    )


def local_core(core, outer):
    return tuple(
        slice(s.start - o.start, s.stop - o.start) for s, o in zip(core, outer, strict=True)
    )


class Workspace:
    def __init__(self, working_memory_bytes=DEFAULT_WORKING_BYTES, scratch_directory=None):
        self.budget = validate_budget(working_memory_bytes)
        self.parent = Path.cwd() if scratch_directory is None else Path(scratch_directory)
        self.maps = []
        self.series = []
        self.bytes = 0
        self.peak_bytes = 0
        self.tiles = 0

    def __enter__(self):
        self.parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix=".element-diagnostics-", dir=self.parent)
        self.path = Path(self.temp.name)
        return self

    def reserve(self, count):
        if count > max(0, shutil.disk_usage(self.path).free - (16 << 20)):
            raise OSError(
                "insufficient disk scratch for exact bounded diagnostics; "
                "supply a larger scratch_directory"
            )
        self.account_existing(count)

    def account_existing(self, count):
        """Account for bytes already written by a disk-backed library."""
        self.bytes += count
        self.peak_bytes = max(self.peak_bytes, self.bytes)

    def mapped(self, shape, dtype=np.float64):
        count = math.prod(shape) * np.dtype(dtype).itemsize
        self.reserve(count)
        path = self.path / f"array-{len(self.maps)}.bin"
        result = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
        self.maps.append(result)
        return result

    def scalar_series(self):
        value = ScalarSeries(self, self.path / f"values-{len(self.series)}.bin")
        self.series.append(value)
        return value

    def cores(self, shape, *, bytes_per_node=512, lower=None, upper=None):
        for core in blocks(shape, max(1, self.budget // bytes_per_node), lower=lower, upper=upper):
            self.tiles += 1
            yield core

    def description(self):
        return {
            "backend": "bounded-blocks-v1",
            "working_memory_budget_bytes": self.budget,
            "core_blocks": self.tiles,
            "peak_scratch_bytes": self.peak_bytes,
            "percentiles": "exact NumPy linear quantiles on owned disk-backed scratch",
            "memory_scope": (
                "working arrays; mapped resident pages and runtime baseline are not a fixed RSS cap"
            ),
        }

    def __exit__(self, *_):
        for value in self.series:
            value.close()
        for value in self.maps:
            value.flush()
            value._mmap.close()
        self.maps.clear()
        self.temp.cleanup()


class ScalarSeries:
    """Streaming reductions and exact percentiles; never partition source data."""

    def __init__(self, workspace, path):
        self.workspace, self.path = workspace, path
        self.file = path.open("xb")
        self.count, self.sum_squares, self.compensation = 0, 0.0, 0.0
        self.maximum = None

    def append(self, values):
        data = np.asarray(values, dtype=np.float64).ravel()
        if not data.size:
            return
        self.workspace.reserve(data.nbytes)
        data.tofile(self.file)
        self.count += int(data.size)
        term = float(np.sum(data * data))
        # Compensated addition of block reductions; infinities preserve NumPy semantics.
        if math.isfinite(term) and math.isfinite(self.sum_squares):
            corrected = term - self.compensation
            total = self.sum_squares + corrected
            self.compensation = (total - self.sum_squares) - corrected
            self.sum_squares = total
        else:
            self.sum_squares += term
            self.compensation = 0.0
        peak = float(np.max(data))
        self.maximum = peak if self.maximum is None else float(np.maximum(self.maximum, peak))

    def quantiles(self, q):
        if not self.count:
            return np.full(np.shape(q), np.nan)
        self.file.flush()
        values = np.memmap(self.path, mode="r+", dtype=np.float64, shape=(self.count,))
        try:
            # The entire selected sample remains on disk; no global ndarray copy.
            result = np.quantile(values, q, overwrite_input=True, method="linear")
            return np.array(result, copy=True)
        finally:
            values._mmap.close()

    def stats(self, scale=None):
        if not self.count:
            return {"cells": 0, "rms": None, "p99": None, "max": None, "p99_relative": None}
        p99 = float(self.quantiles(0.99))
        return {
            "cells": self.count,
            "rms": math.sqrt(self.sum_squares / self.count),
            "p99": p99,
            "max": self.maximum,
            "p99_relative": p99 / scale if scale else None,
        }

    def close(self):
        if not self.file.closed:
            self.file.close()

    def discard(self):
        """Release a finished owned series; useful for sequential exact statistics."""
        self.close()
        if self.path.exists():
            self.workspace.bytes -= self.path.stat().st_size
            self.path.unlink()
