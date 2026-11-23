"""Field maps and the files they come from.

A ``FieldMap`` holds one magnetic field on a Cartesian grid, in metres and tesla, with
the ``FieldSource`` it was read from. Readers convert from the file's units once and
record what they converted from and which storage order they assumed. The grid is the
file's own: whether the assumed storage order is right, how the grid relates to the
beam's frame, and the field's polarity are established by readiness checks
(``readiness.storage_order`` first) and recorded with them, not asserted by a reader.

Readers are registered in ``READERS``; ``load_field_file`` detects the format when it
is not given. Registered: TRACK Cartesian tables (``sol3d``, ``sol3c`` and ``mag3d``
files, including ``FLD3D`` dipole maps), CST text exports and CST HDF5 exports.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from element_surrogate.runs import sha256_file

CM_TO_M = 1.0e-2
GAUSS_TO_T = 1.0e-4

# The storage order a reader assumes: grid axes from slowest- to fastest-varying index,
# and the order of the vector components in each row.
TRACK_STORAGE = {"axes_slowest_to_fastest": ["x", "y", "z"], "components": ["Bx", "By", "Bz"]}


@dataclass(frozen=True)
class FieldSource:
    """Where a field map came from: the file, its hash and format, the slot within it, the
    units the reader converted from, and the storage order the reader assumed."""

    path: str
    sha256: str
    format: str
    slot: str
    source_units: dict[str, str]
    assumed_storage: dict[str, list[str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "format": self.format,
            "slot": self.slot,
            "source_units": dict(self.source_units),
            "assumed_storage": {k: list(v) for k, v in self.assumed_storage.items()},
        }


@dataclass(frozen=True)
class FieldMap:
    """A magnetic field on a Cartesian grid: axes in metres, ``b_t[ix, iy, iz, component]`` in
    tesla, components ordered (Bx, By, Bz). ``source`` is ``None`` for fields built in code."""

    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    b_t: np.ndarray
    label: str = "field"
    source: FieldSource | None = None

    def __post_init__(self) -> None:
        for name in ("x_m", "y_m", "z_m"):
            axis = np.asarray(getattr(self, name), dtype=np.float64)
            if axis.ndim != 1 or axis.size < 2:
                raise ValueError(f"{name} must be a 1-D axis with at least two nodes")
            if not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0.0):
                raise ValueError(f"{name} must be strictly increasing")
            object.__setattr__(self, name, axis)
        b = np.asarray(self.b_t, dtype=np.float64)
        expected = (self.x_m.size, self.y_m.size, self.z_m.size, 3)
        if b.shape != expected:
            raise ValueError(f"b_t has shape {b.shape}, expected {expected}")
        object.__setattr__(self, "b_t", b)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.x_m.size, self.y_m.size, self.z_m.size

    def with_values(self, b_t: np.ndarray) -> FieldMap:
        """The same grid, label and source with different field values."""
        return replace(self, b_t=b_t)

    def window(self, ix: slice, iy: slice, iz: slice) -> FieldMap:
        """The nodes within index ranges along x, y and z, with the same label and source."""
        return replace(
            self, x_m=self.x_m[ix], y_m=self.y_m[iy], z_m=self.z_m[iz], b_t=self.b_t[ix, iy, iz]
        )

    def contains(self, points_m: np.ndarray) -> np.ndarray:
        """Whether each point ``(..., 3)`` lies inside the grid, where the map is defined."""
        p = np.asarray(points_m, dtype=np.float64)
        inside = np.ones(p.shape[:-1], dtype=bool)
        for axis, values in enumerate((self.x_m, self.y_m, self.z_m)):
            inside &= (p[..., axis] >= values[0] - 1e-12) & (p[..., axis] <= values[-1] + 1e-12)
        return inside

    def interpolation_stencil(self, points_m):
        """Trilinear indices and weights; shared by sampling and support auditing."""
        p = np.asarray(points_m, dtype=np.float64)
        if p.ndim < 1 or p.shape[-1] != 3 or not np.isfinite(p).all():
            raise ValueError("finite points of shape (...,3) required")
        cells, weights = [], []
        for k, axis in enumerate((self.x_m, self.y_m, self.z_m)):
            q = p[..., k]
            if np.any(q < axis[0] - 1e-12) or np.any(q > axis[-1] + 1e-12):
                raise ValueError("points lie outside the grid")
            q = np.clip(q, axis[0], axis[-1])
            i = np.clip(np.searchsorted(axis, q, side="right") - 1, 0, len(axis) - 2)
            cells.append(i)
            weights.append((q - axis[i]) / (axis[i + 1] - axis[i]))
        nodes, coefficients = [], []
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    delta = (a, b, c)
                    nodes.append(np.stack([cells[k] + delta[k] for k in range(3)], -1))
                    coefficients.append(
                        np.prod(
                            np.stack(
                                [weights[k] if delta[k] else 1 - weights[k] for k in range(3)], -1
                            ),
                            axis=-1,
                        )
                    )
        return np.stack(nodes, -2), np.stack(coefficients, -1)

    def sample(self, points_m: np.ndarray) -> np.ndarray:
        """Trilinear field in tesla; support auditing uses the same index selector."""
        nodes, weights = self.interpolation_stencil(points_m)
        out = np.zeros(np.asarray(points_m).shape[:-1] + (3,))
        for k in range(8):
            indices = nodes[..., k, :]
            out += weights[..., k, None] * self.b_t[tuple(indices[..., a] for a in range(3))]
        return out

    def grid_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {"nodes": list(self.shape)}
        for name, axis in (("x", self.x_m), ("y", self.y_m), ("z", self.z_m)):
            steps = np.diff(axis)
            summary[f"{name}_range_m"] = [float(axis[0]), float(axis[-1])]
            summary[f"{name}_step_m"] = [float(steps.min()), float(steps.max())]
            summary[f"{name}_uniform"] = bool(np.allclose(steps, steps[0], rtol=1.0e-9, atol=0.0))
        return summary

    def field_summary(self) -> dict[str, Any]:
        from element_surrogate.diagnostic_blocks import blocks

        all_finite, all_zero, maximum = True, True, 0.0
        components = np.zeros(3)
        for core in blocks(self.shape, 65536):
            values = self.b_t[core]
            finite = np.isfinite(values)
            safe = np.where(finite, values, 0.0)
            all_finite &= bool(finite.all())
            all_zero &= not bool(np.any(safe))
            maximum = max(maximum, float(np.linalg.norm(safe, axis=-1).max()))
            components = np.maximum(components, np.abs(safe).max(axis=(0, 1, 2)))
        return {
            "all_finite": all_finite,
            "max_abs_b_t": maximum,
            "max_abs_component_t": components.tolist(),
            "all_zero": all_zero,
        }


@dataclass(frozen=True)
class FieldFile:
    """A field file read from disk: one ``FieldMap`` per field slot, what the reader found
    (``integrity``), and format-specific details such as header lines."""

    path: str
    sha256: str
    format: str
    slots: list[FieldMap]
    integrity: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def slot(self, index: int | None) -> FieldMap:
        """One slot. With several slots, the caller must choose: some files hold an all-zero
        slot, and a default would silently analyse the wrong field."""
        if index is None:
            if len(self.slots) != 1:
                labels = [s.label for s in self.slots]
                empty = self.integrity.get("all_zero_slots", [])
                raise ValueError(f"the file has slots {labels} (all zero: {empty}); choose one")
            return self.slots[0]
        if not 0 <= index < len(self.slots):
            raise ValueError(f"slot {index} does not exist; the file has {len(self.slots)}")
        return self.slots[index]

    def summary(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "format": self.format,
            **self.details,
            "integrity": self.integrity,
            "slots": [
                {"label": s.label, "grid": s.grid_summary(), "field": s.field_summary()}
                for s in self.slots
            ],
        }


def _axis_line(line: str) -> tuple[float, float, int]:
    parts = line.split()
    if len(parts) != 3:
        raise ValueError(f"expected an axis header 'low high count', got {line!r}")
    return float(parts[0]), float(parts[1]), int(parts[2])


def _track_header(path: Path) -> list[str]:
    with path.open("rb") as handle:
        header = [handle.readline().decode("ascii").strip() for _ in range(5)]
    if any(not line for line in header):
        raise ValueError(f"{path}: fewer than five header lines")
    return header


def is_track_table(path: Path) -> bool:
    """Whether the file starts like a TRACK Cartesian table: five header lines, the last
    three of the form ``low high count``."""
    try:
        header = _track_header(path)
        for line in header[2:]:
            _axis_line(line)
    except (ValueError, UnicodeDecodeError):
        return False
    return True


_TRACK_FIELD_TYPES = {0: "E", 1: "B", 2: "E and B"}


def _track_declarations(header: list[str]) -> dict[str, Any]:
    """The first two header lines as the TRACK ``mag3d`` manual documents them: the field type
    (0 E, 1 B, 2 E and B) and symmetry flags in x, y and z. A first line in another form
    (multi-slot tables use one) is kept verbatim and not interpreted."""
    tokens = header[0].split()
    declared: dict[str, Any] = {"field_type_line": header[0]}
    if len(tokens) == 1 and tokens[0] in {"0", "1", "2"}:
        code = int(tokens[0])
        declared["field_type"] = {"code": code, "meaning": _TRACK_FIELD_TYPES[code]}
    else:
        declared["field_type"] = None
    flags = header[1].split()
    declared["symmetry_flags"] = (
        [int(f) for f in flags]
        if len(flags) == 3 and all(f.lstrip("-").isdigit() for f in flags)
        else None
    )
    return declared


def read_track_table(path: str | Path) -> FieldFile:
    """Read a TRACK Cartesian field table (the ``sol3d``, ``sol3c`` and ``mag3d`` layout).

    Layout, as the TRACK ``mag3d`` manual documents it: a field-type line (0 E, 1 B,
    2 E and B), a line of symmetry flags in x, y and z, then three lines giving
    ``low high count`` in cm for x, y and z; then the data, x varying slowest and z
    fastest, holding Bx, By, Bz in gauss for each field slot (3 columns per slot).
    Type-2 tables interleave an electric row before each magnetic row.

    Reported in ``integrity`` rather than hidden: the declared field type and whether it
    agrees with the rows found, the symmetry flags (this reader does not unfold a
    symmetric half-map), an interleaved zero row per node, a duplicated final data row
    (dropped from the field), and slots that are entirely zero. An electric-only table,
    or anything else that does not match the header, raises ``ValueError``.
    """
    file_path = Path(path)
    header = _track_header(file_path)
    declared = _track_declarations(header)
    if declared["field_type"] is not None and declared["field_type"]["code"] == 0:
        raise ValueError(f"{file_path}: the header declares an electric field only")
    axes_cm = [np.linspace(low, high, count) for low, high, count in map(_axis_line, header[2:])]
    nodes = axes_cm[0].size * axes_cm[1].size * axes_cm[2].size

    raw = np.loadtxt(file_path, skiprows=5, dtype=np.float64, ndmin=2)
    rows, columns = raw.shape
    if columns % 3 != 0:
        raise ValueError(f"{file_path}: {columns} columns is not a whole number of (Bx, By, Bz)")

    duplicate_terminal_row = False
    if rows in (nodes + 1, 2 * nodes + 1) and np.array_equal(raw[-1], raw[-2]):
        duplicate_terminal_row = True
        raw = raw[:-1]
        rows -= 1
    if rows == nodes:
        rows_per_node, magnetic = 1, raw
    elif rows == 2 * nodes:
        if np.any(raw[0::2]):
            raise ValueError(
                f"{file_path}: two rows per node, but the first row of each is not zero"
            )
        rows_per_node, magnetic = 2, raw[1::2]
    else:
        raise ValueError(f"{file_path}: header declares {nodes} nodes, found {rows} data rows")

    sha = sha256_file(file_path)
    resolved = str(file_path.resolve())
    units = {"coordinates": "cm", "field": "gauss"}
    n_slots = columns // 3
    shaped = magnetic.reshape(axes_cm[0].size, axes_cm[1].size, axes_cm[2].size, n_slots, 3)
    x, y, z = (axis * CM_TO_M for axis in axes_cm)
    slots = []
    for s in range(n_slots):
        label = f"slot{s}"
        source = FieldSource(resolved, sha, "track_table", label, units, TRACK_STORAGE)
        slots.append(FieldMap(x, y, z, shaped[..., s, :] * GAUSS_TO_T, label=label, source=source))
    integrity = {
        "nodes": nodes,
        "data_rows_found": int(rows + duplicate_terminal_row),
        "columns": columns,
        "field_slots": n_slots,
        "rows_per_node": rows_per_node,
        "interleaved_zero_rows": rows_per_node == 2,
        "duplicate_terminal_row": duplicate_terminal_row,
        "all_finite": bool(np.isfinite(magnetic).all()),
        "all_zero_slots": [slot.label for slot in slots if not np.any(slot.b_t)],
        "declared_field_type": declared["field_type"],
        "declared_type_agrees_with_rows": (
            None
            if declared["field_type"] is None
            else (declared["field_type"]["code"] == 2) == (rows_per_node == 2)
        ),
        "symmetry_flags": declared["symmetry_flags"],
        "symmetry_note": (
            None
            if not declared["symmetry_flags"] or not any(declared["symmetry_flags"])
            else "nonzero symmetry flags: the file may hold only part of the field, which this "
            "reader does not unfold"
        ),
    }
    details = {
        "format_name": "TRACK Cartesian field table",
        "format_reference": "TRACK v3.9 manual, element mag3d (field file format)",
        "source_units": units,
        "assumed_storage": TRACK_STORAGE,
        "header_lines": header,
    }
    return FieldFile(resolved, sha, "track_table", slots, integrity, details)


# Units found in exports, as factors to metres and tesla. A magnetizing field H (A/m) is
# not converted: B = mu0 H holds only outside magnetic material, which a reader cannot know.
_LENGTH_UNITS = {"m": 1.0, "cm": 1.0e-2, "mm": 1.0e-3, "um": 1.0e-6}
_B_UNITS = {
    "T": 1.0,
    "V.s/m^2": 1.0,
    "Vs/m^2": 1.0,
    "mT": 1.0e-3,
    "G": GAUSS_TO_T,
    "gauss": GAUSS_TO_T,
}
_H_UNITS = {"A/m"}
_CST_COLUMN = re.compile(r"(\S+)\s*\[([^\]]*)\]")


def _length_factor(unit: str, where: str) -> float:
    if unit not in _LENGTH_UNITS:
        raise ValueError(f"{where}: unknown length unit {unit!r}; known: {sorted(_LENGTH_UNITS)}")
    return _LENGTH_UNITS[unit]


def _b_factor(unit: str, where: str) -> float:
    if unit in _H_UNITS:
        raise ValueError(
            f"{where}: the export holds H in {unit}; export B instead, or convert it yourself "
            "where B = mu0 H holds (outside magnetic material)"
        )
    if unit not in _B_UNITS:
        raise ValueError(
            f"{where}: unknown magnetic-field unit {unit!r}; known: {sorted(_B_UNITS)}"
        )
    return _B_UNITS[unit]


def _cst_text_columns(path: Path) -> list[tuple[str, str]]:
    with path.open("rb") as handle:
        first = handle.readline().decode("ascii").strip()
        second = handle.readline().decode("ascii").strip()
    columns = _CST_COLUMN.findall(first)
    if len(columns) < 6 or [c[0] for c in columns[:3]] != ["x", "y", "z"]:
        raise ValueError(f"{path}: the first line does not name columns x, y, z and a field")
    if not second or set(second) != {"-"}:
        raise ValueError(f"{path}: the second line is not a line of dashes")
    return columns


def is_cst_text(path: Path) -> bool:
    """Whether the file starts like a CST text export: a line naming columns ``x [unit]``,
    ``y [unit]``, ``z [unit]`` and the field components, then a line of dashes."""
    try:
        _cst_text_columns(path)
    except (ValueError, UnicodeDecodeError):
        return False
    return True


def _component(name: str) -> tuple[int, str]:
    """The component (0, 1, 2 for x, y, z) and part ("re", "im" or "") a field column holds,
    from names such as ``x``, ``Bx``, ``BxRe`` or ``HyIm``."""
    stem, part = name, ""
    for suffix in ("Re", "Im"):
        if stem.endswith(suffix):
            stem, part = stem[: -len(suffix)], suffix.lower()
    stem = stem[1:] if len(stem) == 2 and stem[0] in "BH" else stem
    if stem not in ("x", "y", "z"):
        raise ValueError(f"cannot tell which component the column {name!r} holds")
    return "xyz".index(stem), part


def _nesting(coords: np.ndarray) -> list[str] | None:
    """Grid axes from slowest to fastest varying in row order, or None when the rows do not
    follow one nesting."""
    changes = (np.diff(coords, axis=0) != 0).sum(axis=0)
    order = np.argsort(changes)
    counts = [np.unique(coords[:, a]).size for a in order]
    expected = [counts[0] - 1, counts[0] * counts[1] - 1, coords.shape[0] - 1]
    if [int(changes[a]) for a in order] != expected:
        return None
    return ["xyz"[a] for a in order]


def _place_rows(
    coords: np.ndarray, values: np.ndarray, where: str
) -> tuple[list[np.ndarray], np.ndarray, int]:
    """Put each row's vector at the grid node its coordinates name. Returns the axes, the
    values on the grid and the number of exactly duplicated rows dropped; rows that do not
    fill a Cartesian product, or that give one node two different values, raise."""
    axes = [np.unique(coords[:, a]) for a in range(3)]
    index = np.stack([np.searchsorted(axes[a], coords[:, a]) for a in range(3)], axis=-1)
    flat: np.ndarray = np.asarray(np.ravel_multi_index(index.T, tuple(a.size for a in axes)))
    order = np.argsort(flat, kind="stable")
    flat_sorted = flat[order]
    first = np.ones(flat_sorted.size, dtype=bool)
    first[1:] = flat_sorted[1:] != flat_sorted[:-1]
    repeated = order[~first]
    kept = order[first]
    if repeated.size:
        earlier = order[np.flatnonzero(~first) - 1]
        if not np.array_equal(values[repeated], values[earlier]):
            raise ValueError(f"{where}: some nodes appear twice with different values")
    nodes = int(np.prod([a.size for a in axes]))
    if kept.size != nodes:
        raise ValueError(
            f"{where}: {kept.size} distinct nodes do not fill the {nodes}-node Cartesian product "
            "of the coordinate values"
        )
    grid = np.empty((nodes, 3))
    grid[flat[kept]] = values[kept]
    return axes, grid.reshape(*(a.size for a in axes), 3), int(repeated.size)


def read_cst_text(path: str | Path) -> FieldFile:
    """Read a CST text field export.

    Layout, as found in such files: a line naming the columns with their units in
    brackets (``x [mm]  y [mm]  z [mm]  x [V.s/m^2] ...``), a line of dashes, then one row
    per node holding its coordinates and field. Each row is placed by its own coordinates,
    so no storage order is assumed; the order the rows follow is recorded. Imaginary parts
    of a complex export must be zero. An H-field export raises (see ``_b_factor``).
    """
    file_path = Path(path)
    where = str(file_path)
    columns = _cst_text_columns(file_path)
    coord_units = {c[1] for c in columns[:3]}
    if len(coord_units) != 1:
        raise ValueError(f"{where}: coordinate columns use different units {sorted(coord_units)}")
    coord_unit = coord_units.pop()
    to_m = _length_factor(coord_unit, where)
    field_units = {c[1] for c in columns[3:]}
    if len(field_units) != 1:
        raise ValueError(f"{where}: field columns use different units {sorted(field_units)}")
    field_unit = field_units.pop()
    to_t = _b_factor(field_unit, where)
    parts = [_component(name) for name, _ in columns[3:]]
    real = {c: k for k, (c, part) in enumerate(parts) if part in ("", "re")}
    imaginary = [k for k, (_, part) in enumerate(parts) if part == "im"]
    if sorted(real) != [0, 1, 2] or len(real) + len(imaginary) != len(parts):
        raise ValueError(
            f"{where}: the field columns {[c[0] for c in columns[3:]]} are not one vector"
        )

    raw = np.loadtxt(file_path, skiprows=2, dtype=np.float64, ndmin=2)
    if raw.shape[1] != len(columns):
        raise ValueError(f"{where}: {raw.shape[1]} data columns, header names {len(columns)}")
    coords = raw[:, :3]
    field_columns = raw[:, 3:]
    imaginary_zero = None if not imaginary else not np.any(field_columns[:, imaginary])
    if imaginary_zero is False:
        raise ValueError(f"{where}: non-zero imaginary parts; this reader handles static fields")
    values = field_columns[:, [real[c] for c in range(3)]]
    nesting = _nesting(coords)
    axes, grid, duplicates = _place_rows(coords, values, where)

    sha = sha256_file(file_path)
    resolved = str(file_path.resolve())
    units = {"coordinates": coord_unit, "field": field_unit}
    storage = {
        "axes_slowest_to_fastest": nesting if nesting is not None else ["x", "y", "z"],
        "components": ["Bx", "By", "Bz"],
        "placement": ["each row by its own coordinates"],
    }
    source = FieldSource(resolved, sha, "cst_text", "B", units, storage)
    x, y, z = (a * to_m for a in axes)
    field_map = FieldMap(x, y, z, grid * to_t, label="B", source=source)
    integrity = {
        "nodes": int(grid.shape[0] * grid.shape[1] * grid.shape[2]),
        "data_rows_found": int(raw.shape[0]),
        "duplicated_rows_dropped": duplicates,
        "row_nesting_slowest_to_fastest": nesting,
        "all_finite": bool(np.isfinite(raw).all()),
        "imaginary_parts": None if imaginary_zero is None else "present, all zero",
        "all_zero_slots": [] if np.any(grid) else ["B"],
    }
    details = {
        "format_name": "CST text field export",
        "format_reference": "layout as found in the file (column names with units, dashes, rows)",
        "columns": [f"{name} [{unit}]" for name, unit in columns],
        "source_units": units,
        "assumed_storage": storage,
    }
    return FieldFile(resolved, sha, "cst_text", [field_map], integrity, details)


_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
CST_HDF5_STORAGE = {"axes_slowest_to_fastest": ["z", "y", "x"], "components": ["Bx", "By", "Bz"]}


def _attr_text(value: Any) -> str:
    return value.decode("ascii").strip() if isinstance(value, bytes) else str(value).strip()


def _cst_hdf5_units(h5, where):
    """Validate unit metadata before either CST reader touches dataset payloads."""
    if "B-Field" not in h5:
        raise ValueError(f"{where}: CST export requires B-Field, not H-Field")
    units = set()
    for name in [*(f"Mesh line {axis}" for axis in "xyz"), "B-Field"]:
        if name not in h5:
            raise ValueError(f"{where}: required CST dataset {name!r} is missing")
        node = h5[name]
        if "unit" not in node.attrs or not _attr_text(node.attrs["unit"]):
            kind = "magnetic-field" if name == "B-Field" else "coordinate"
            raise ValueError(
                f"{where}: dataset {name!r} requires a nonempty 'unit' attribute "
                f"declaring its {kind} unit; no units are assumed"
            )
        unit = _attr_text(node.attrs["unit"])
        if name == "B-Field":
            _b_factor(unit, where)
            field_unit = unit
        else:
            _length_factor(unit, where)
            units.add(unit)
    if len(units) != 1:
        raise ValueError(f"{where}: mesh lines use different units {sorted(units)}")
    return units.pop(), field_unit


def is_cst_hdf5(path: Path) -> bool:
    """Whether the file is HDF5 holding the CST layout: ``Mesh line x``, ``Mesh line y`` and
    ``Mesh line z`` datasets and a ``B-Field`` or ``H-Field`` dataset."""
    with path.open("rb") as handle:
        if handle.read(8) != _HDF5_SIGNATURE:
            return False
    import h5py  # only the HDF5 reader needs it

    with h5py.File(path, "r") as h5:
        names = set(h5.keys())
    return {"Mesh line x", "Mesh line y", "Mesh line z"} <= names and bool(
        names & {"B-Field", "H-Field"}
    )


def read_cst_hdf5(path: str | Path) -> FieldFile:
    """Read a CST HDF5 field export.

    Layout, as found in such files: datasets ``Mesh line x``, ``Mesh line y`` and ``Mesh
    line z`` (each with a ``unit`` attribute) give the axes; ``B-Field`` holds one compound
    vector ``(x, y, z)`` per node with a ``unit`` attribute, indexed ``[iz, iy, ix]`` (x
    fastest). That index order is assumed and recorded, and its shape must match the axes;
    the storage-order test checks it. Complex components must have zero imaginary parts.
    """
    import h5py  # only the HDF5 reader needs it

    file_path = Path(path)
    where = str(file_path)
    with h5py.File(file_path, "r") as h5:
        coord_unit, field_unit = _cst_hdf5_units(h5, where)
        axes, coord_units = [], set()
        for name in "xyz":
            node = h5[f"Mesh line {name}"]
            coord_units.add(_attr_text(node.attrs["unit"]))
            axes.append(np.asarray(node[:], dtype=np.float64))
        if len(coord_units) != 1:
            raise ValueError(f"{where}: mesh lines use different units {sorted(coord_units)}")
        coord_unit = coord_units.pop()
        to_m = _length_factor(coord_unit, where)
        data = h5["B-Field"]
        to_t = _b_factor(field_unit, where)
        if data.dtype.names != ("x", "y", "z"):
            raise ValueError(f"{where}: B-Field holds {data.dtype.names}, expected (x, y, z)")
        expected = tuple(a.size for a in axes[::-1])
        if data.shape != expected:
            raise ValueError(
                f"{where}: B-Field has shape {data.shape}, expected (nz, ny, nx) {expected}"
            )
        complex_parts = data.dtype["x"].names
        grid = np.empty((*(a.size for a in axes), 3))
        imaginary_zero = True
        chunk = max(1, 2**22 // max(1, expected[1] * expected[2]))
        for k in range(0, expected[0], chunk):
            block = data[k : k + chunk]
            for c, name in enumerate("xyz"):
                values = block[name]
                if complex_parts is not None:
                    if complex_parts != ("re", "im"):
                        raise ValueError(f"{where}: component {name} holds {complex_parts}")
                    imaginary_zero &= not np.any(values["im"])
                    values = values["re"]
                grid[:, :, k : k + len(block), c] = np.transpose(values, (2, 1, 0))
        root_attrs = {key: _attr_text(value) for key, value in h5.attrs.items()}
    if not imaginary_zero:
        raise ValueError(f"{where}: non-zero imaginary parts; this reader handles static fields")
    if any(a.size < 2 or not np.all(np.diff(a) > 0) for a in axes):
        raise ValueError(f"{where}: a mesh line is not strictly increasing")

    sha = sha256_file(file_path)
    resolved = str(file_path.resolve())
    units = {"coordinates": coord_unit, "field": field_unit}
    source = FieldSource(resolved, sha, "cst_hdf5", "B-Field", units, CST_HDF5_STORAGE)
    x, y, z = (a * to_m for a in axes)
    field_map = FieldMap(x, y, z, grid * to_t, label="B-Field", source=source)
    integrity = {
        "nodes": int(np.prod(expected)),
        "all_finite": bool(np.isfinite(grid).all()),
        "imaginary_parts": None if complex_parts is None else "present, all zero",
        "all_zero_slots": [] if np.any(grid) else ["B-Field"],
    }
    details = {
        "format_name": "CST HDF5 field export",
        "format_reference": "layout as found in the file (mesh lines and a B-Field compound)",
        "root_attributes": root_attrs,
        "source_units": units,
        "assumed_storage": CST_HDF5_STORAGE,
    }
    return FieldFile(resolved, sha, "cst_hdf5", [field_map], integrity, details)


Detector = Callable[[Path], bool]
Reader = Callable[[str | Path], FieldFile]


def _is_mapped_field(path):
    from element_surrogate.field_store import is_mapped_field

    return is_mapped_field(path)


def _read_mapped_field(path):
    from element_surrogate.field_store import read_mapped_field

    return read_mapped_field(path)


# Registered readers: format name -> (detector, reader). Registering a reader here makes
# every tool able to use it.
READERS: dict[str, tuple[Detector, Reader]] = {
    "mapped_field": (_is_mapped_field, _read_mapped_field),
    "track_table": (is_track_table, read_track_table),
    "cst_text": (is_cst_text, read_cst_text),
    "cst_hdf5": (is_cst_hdf5, read_cst_hdf5),
}


def detect_format(path: str | Path) -> str:
    """The registered format whose detector accepts the file; raises when none or several do."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"not a file: {path}")
    matches = [name for name, (detect, _) in READERS.items() if detect(file_path)]
    if len(matches) != 1:
        found = "no registered format" if not matches else f"several formats {matches}"
        raise ValueError(f"{path}: {found} matches; give format, one of {sorted(READERS)}")
    return matches[0]


def load_field_file(path: str | Path, format: str | None = None) -> FieldFile:
    """Read a field file with the named reader, or with the one whose detector accepts it."""
    name = format if format is not None else detect_format(path)
    if name not in READERS:
        raise ValueError(f"unknown format {name!r}; registered: {sorted(READERS)}")
    return READERS[name][1](path)
