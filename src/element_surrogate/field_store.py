"""Exact float64 memory-mapped field storage; no resident whole-field copy.

This is a local conversion/cache format, not a solver mesh, compression model or
permission to redistribute a field. Conversion accepts an already read field;
the original text/CST readers still have their documented ingestion memory cost.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from element_surrogate.benchmark.protocol import file_hash, read_json, write_json
from element_surrogate.fields import FieldFile, FieldMap, FieldSource

SCHEMA = "element_surrogate.mapped_field.v1"


def convert_cst_hdf5(path, directory, *, block_nodes):
    """Stream a complete CST Cartesian export directly into the exact mapped format.

    No source cropping, downsampling, interpolation or whole-field array. An
    incomplete conversion never acquires the requested destination name.
    """
    import h5py

    from element_surrogate.diagnostic_blocks import blocks
    from element_surrogate.fields import (
        CST_HDF5_STORAGE,
        _attr_text,
        _b_factor,
        _cst_hdf5_units,
        _length_factor,
    )

    if type(block_nodes) is not int or block_nodes < 1:
        raise ValueError("positive conversion block_nodes required")
    source_path, directory = Path(path).resolve(), Path(directory).resolve()
    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    before = file_hash(source_path)
    with h5py.File(source_path, "r") as h5:
        _, field_unit = _cst_hdf5_units(h5, str(source_path))
        axes = []
        units = set()
        for name in "xyz":
            node = h5[f"Mesh line {name}"]
            unit = _attr_text(node.attrs["unit"])
            units.add(unit)
            axis = np.asarray(node[:], dtype=np.float64) * _length_factor(unit, str(source_path))
            if (
                axis.ndim != 1
                or len(axis) < 2
                or not np.isfinite(axis).all()
                or not np.all(np.diff(axis) > 0)
            ):
                raise ValueError(
                    "CST mesh lines must be finite, increasing and contain at least two nodes"
                )
            axes.append(axis)
        if len(units) != 1:
            raise ValueError("CST mesh lines use different units")
        field = h5["B-Field"]
        unit = field_unit
        factor = _b_factor(unit, str(source_path))
        shape = tuple(len(a) for a in axes)
        if field.shape != shape[::-1] or field.dtype.names != ("x", "y", "z"):
            raise ValueError("CST B-Field must be an (nz,ny,nx) compound (x,y,z)")
        for name in "xyz":
            if field.dtype[name].names not in (None, ("re", "im")):
                raise ValueError("CST complex components require re/im parts")
        needed = int(np.prod(shape)) * 24 + sum(a.nbytes for a in axes) + (16 << 20)
        if shutil.disk_usage(directory.parent).free < needed:
            raise OSError("insufficient destination disk for complete mapped CST field")
        with tempfile.TemporaryDirectory(
            prefix=".element-conversion-", dir=directory.parent
        ) as tmp:
            staging = Path(tmp)
            files = {}
            for key, axis in zip(("x_m", "y_m", "z_m"), axes, strict=True):
                target = staging / f"{key}.npy"
                np.save(target, axis, allow_pickle=False)
                files[key] = {"name": target.name, "sha256": file_hash(target)}
            target = staging / "b_t.npy"
            mapped = np.lib.format.open_memmap(
                target, mode="w+", dtype=np.float64, shape=shape + (3,)
            )
            try:
                for core in blocks(field.shape, block_nodes):
                    raw = field[core]
                    for c, name in enumerate("xyz"):
                        values = raw[name]
                        if values.dtype.names is not None:
                            if np.any(values["im"]):
                                raise ValueError(
                                    "non-zero imaginary parts; only static fields supported"
                                )
                            values = values["re"]
                        converted = np.asarray(values, dtype=np.float64) * factor
                        if not np.isfinite(converted).all():
                            raise ValueError("nonfinite field cannot enter the mapped store")
                        mapped[core[2], core[1], core[0], c] = np.transpose(converted, (2, 1, 0))
                mapped.flush()
            finally:
                mapped._mmap.close()
            if file_hash(source_path) != before:
                raise ValueError("source changed during CST conversion")
            files["b_t"] = {"name": target.name, "sha256": file_hash(target)}
            source = FieldSource(
                str(source_path),
                before,
                "cst_hdf5",
                "B-Field",
                {"coordinates": units.pop(), "field": unit},
                CST_HDF5_STORAGE,
            )
            write_json(
                staging / "field.json",
                {
                    "schema": SCHEMA,
                    "shape": list(shape + (3,)),
                    "dtype": "float64",
                    "units": {"position": "m", "field": "T"},
                    "files": files,
                    "source": source.as_dict(),
                },
                exclusive=True,
            )
            # Same filesystem: publish only the completed, hash-verified directory.
            staging.rename(directory)
    return {
        "path": str(directory / "field.json"),
        "sha256": file_hash(directory / "field.json"),
        "field_nodes": int(np.prod(shape)),
        "max_conversion_block_nodes": block_nodes,
        "source_retained": True,
        "ingestion": "bounded CST HDF5 blocks; complete native resolution",
    }


def save_mapped_field(grid, directory, *, block_nodes) -> dict:
    """Create complete NPY members with bounded float64 blocks, then the manifest."""
    if type(block_nodes) is not int or block_nodes < 1:
        raise ValueError("positive conversion block_nodes required")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    files = {}
    for key, values in zip(("x_m", "y_m", "z_m"), (grid.x_m, grid.y_m, grid.z_m), strict=True):
        path = directory / f"{key}.npy"
        np.save(path, values, allow_pickle=False)
        files[key] = {"name": path.name, "sha256": file_hash(path)}
    path = directory / "b_t.npy"
    shape = (*grid.shape, 3)
    mapped = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=shape)
    # Slabs along z bound temporary arrays independently of x/y slab size.
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for k in range(0, grid.shape[2], block_nodes):
                block = grid.b_t[i, j, k : k + block_nodes]
                if not np.isfinite(block).all():
                    raise ValueError("nonfinite field cannot enter the mapped store")
                mapped[i, j, k : k + block_nodes] = block
    mapped.flush()
    del mapped
    files["b_t"] = {"name": path.name, "sha256": file_hash(path)}
    manifest = {
        "schema": SCHEMA,
        "shape": list(shape),
        "dtype": "float64",
        "units": {"position": "m", "field": "T"},
        "files": files,
        "source": None if grid.source is None else grid.source.as_dict(),
    }
    write_json(directory / "field.json", manifest, exclusive=True)
    return {
        "path": str(directory / "field.json"),
        "sha256": file_hash(directory / "field.json"),
        "field_nodes": int(np.prod(grid.shape)),
        "max_conversion_block_nodes": block_nodes,
        "source_retained": True,
    }


def is_mapped_field(path):
    if path.suffix.lower() != ".json":
        return False
    try:
        return read_json(path).get("schema") == SCHEMA
    except (ValueError, UnicodeError, AttributeError):
        return False


def read_mapped_field(path):
    path = Path(path).resolve()
    metadata = read_json(path)
    if (
        metadata.get("schema") != SCHEMA
        or metadata.get("dtype") != "float64"
        or metadata.get("units") != {"position": "m", "field": "T"}
        or set(metadata["files"]) != {"x_m", "y_m", "z_m", "b_t"}
    ):
        raise ValueError("unsupported mapped-field contract")
    arrays = {}
    for key, item in metadata["files"].items():
        member = (path.parent / item["name"]).resolve()
        if (
            not member.is_relative_to(path.parent)
            or member.name != item["name"]
            or file_hash(member) != item["sha256"]
        ):
            raise ValueError("mapped-field member changed or leaves its directory")
        arrays[key] = np.load(member, mmap_mode="r", allow_pickle=False)
        if arrays[key].dtype != np.float64:
            raise ValueError("mapped-field values must be exact float64")
    if list(arrays["b_t"].shape) != metadata["shape"]:
        raise ValueError("mapped field shape changed")
    source = FieldSource(
        str(path),
        file_hash(path),
        "mapped_field",
        "B",
        {"coordinates": "m", "field": "T"},
        {"axes_slowest_to_fastest": ["x", "y", "z"], "components": ["Bx", "By", "Bz"]},
    )
    grid = FieldMap(**arrays, label="B", source=source)
    # Verify finiteness in bounded rows instead of allocating a whole-field mask.
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for k in range(0, grid.shape[2], 4096):
                if not np.isfinite(grid.b_t[i, j, k : k + 4096]).all():
                    raise ValueError("nonfinite mapped-field values")
    return FieldFile(
        str(path),
        file_hash(path),
        "mapped_field",
        [grid],
        {"all_finite": True, "verified_member_hashes": True},
        {
            "format_name": "exact float64 memory-mapped field",
            "original_source": metadata.get("source"),
        },
    )
