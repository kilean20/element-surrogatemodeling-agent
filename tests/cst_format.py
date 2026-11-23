"""Write synthetic fields in the CST text and HDF5 export layouts, for reader tests."""

from pathlib import Path

import h5py
import numpy as np

from element_surrogate.fields import FieldMap


def write_cst_text(
    path: Path,
    field: FieldMap,
    nesting: str = "xyz",
    field_columns: tuple[str, ...] = ("x", "y", "z"),
    field_unit: str = "V.s/m^2",
    imaginary: np.ndarray | None = None,
    shuffle_seed: int | None = None,
    repeat_rows: tuple[int, ...] = (),
) -> Path:
    """Write ``field`` with coordinates in mm, rows nested as ``nesting`` (slowest first),
    optionally shuffled, with some rows repeated, and with imaginary columns if given."""
    axes = {"x": field.x_m, "y": field.y_m, "z": field.z_m}
    order = ["xyz".index(a) for a in nesting]
    values = np.transpose(field.b_t, (*order, 3)).reshape(-1, 3)
    grids = np.meshgrid(*(axes[a] for a in nesting), indexing="ij")
    coords = np.empty((values.shape[0], 3))
    for k, a in enumerate(nesting):
        coords[:, "xyz".index(a)] = grids[k].ravel() * 1.0e3
    columns = [values]
    names = [f"{c} [{field_unit}]" for c in field_columns]
    if imaginary is not None:
        columns.append(np.broadcast_to(imaginary, values.shape))
        names = [f"B{c}{part} [{field_unit}]" for part in ("Re", "Im") for c in "xyz"]
    rows = np.concatenate([coords, *columns], axis=1)
    if shuffle_seed is not None:
        rows = rows[np.random.default_rng(shuffle_seed).permutation(rows.shape[0])]
    if repeat_rows:
        rows = np.concatenate([rows, rows[list(repeat_rows)]])
    header = "  ".join(["x [mm]", "y [mm]", "z [mm]", *names])
    lines = [header, "-" * len(header)]
    lines += [" ".join(f"{v:.17g}" for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def write_cst_hdf5(
    path: Path,
    field: FieldMap,
    field_unit: str = "V.s/m^2",
    dataset: str = "B-Field",
    complex_parts: bool = False,
    imaginary: float = 0.0,
) -> Path:
    """Write ``field`` with mesh lines in mm and the vectors indexed [iz, iy, ix]."""
    with h5py.File(path, "w") as h5:
        for name, axis in (("x", field.x_m), ("y", field.y_m), ("z", field.z_m)):
            line = h5.create_dataset(f"Mesh line {name}", data=axis * 1.0e3)
            line.attrs["unit"] = b"mm"
            line.attrs["type"] = 0
        native = np.transpose(field.b_t, (2, 1, 0, 3))
        if complex_parts:
            part = np.dtype([("re", "<f8"), ("im", "<f8")])
            vector = np.dtype([("x", part), ("y", part), ("z", part)])
            data = np.empty(native.shape[:3], dtype=vector)
            for c, name in enumerate("xyz"):
                data[name]["re"] = native[..., c]
                data[name]["im"] = imaginary
        else:
            vector = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
            data = np.empty(native.shape[:3], dtype=vector)
            for c, name in enumerate("xyz"):
                data[name] = native[..., c]
        out = h5.create_dataset(dataset, data=data)
        out.attrs["unit"] = field_unit.encode("ascii")
        out.attrs["type"] = 0
    return path
