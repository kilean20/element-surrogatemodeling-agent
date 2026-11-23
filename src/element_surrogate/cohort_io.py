"""Checked shard descriptors shared by benchmark ingestion and scale diagnostics."""

import zipfile
from pathlib import Path

import numpy as np

from element_surrogate.benchmark.protocol import file_hash, read_json


def descriptor_members(path):
    path = Path(path).resolve()
    descriptor = read_json(path)
    if descriptor.get("schema") != "element_surrogate.cohort_shards.v1":
        raise ValueError("unsupported cohort descriptor; expected cohort_shards.v1 JSON")
    members, seen = [], set()
    for member in descriptor.get("members", []):
        source = Path(member["path"])
        source = (path.parent / source).resolve() if not source.is_absolute() else source.resolve()
        count = member["rows"]
        if source in seen or type(count) is not int or count < 1:
            raise ValueError("duplicate cohort member or invalid prefix count")
        if file_hash(source) != member["sha256"]:
            raise ValueError("cohort member changed: SHA256 mismatch")
        seen.add(source)
        members.append((source, count))
    if not members or type(descriptor.get("rows")) is not int:
        raise ValueError("nonempty cohort descriptor with integer rows required")
    if sum(n for _, n in members) != descriptor["rows"]:
        raise ValueError("cohort descriptor row count differs from members")
    return descriptor, members


def uncompressed_bytes(path):
    with zipfile.ZipFile(path) as archive:
        return sum(item.file_size for item in archive.infolist())


def require_native_metadata(metadata, dimension):
    units = metadata.get("coordinate_units", [])
    names = metadata.get("coordinate_names", [])
    if len(units) != dimension or len(names) != dimension:
        raise ValueError("native coordinate names/units and matching dimension required")
    if units[:4] != ["m", "rad", "m", "rad"] or names[:4] != ["x", "xp", "y", "yp"]:
        raise ValueError("scale proposal requires native m/rad plane coordinates")
    if dimension > 4 and (units[-1] != "1" or names[-1] != "delta_p"):
        raise ValueError("native momentum deviation must be delta_p, unit 1")
    if dimension == 6 and units[4] not in ("m", "s"):
        raise ValueError("native arrival coordinate must use m or s")
    if metadata.get("input_units") != units + ["T*m", "1"]:
        raise ValueError("cohort input units must include effective rigidity and scale")
    if metadata.get("input_names") != names + ["effective_rigidity_tm", "field_scale"]:
        raise ValueError("cohort input names differ from native state/control contract")


def native_batches(path, dimension, *, batch_rows, working_memory_bytes, protocol_path=None):
    """Read all declared training rows, bounded by a shard and numeric batch.

    ZIP arrays are decompressed per shard; an oversize shard is refused before
    decompression, not silently sampled. This bounds working arrays, not RSS.
    """
    if type(batch_rows) is not int or batch_rows < 1:
        raise ValueError("positive batch_rows required")
    if type(working_memory_bytes) is not int or working_memory_bytes < 1:
        raise ValueError("positive working_memory_bytes required")
    path = Path(path).resolve()
    if path.stat().st_size > working_memory_bytes:
        raise ValueError("cohort file exceeds working-memory allowance; use smaller shards")
    if not zipfile.is_zipfile(path):
        descriptor, members = descriptor_members(path)
        if descriptor.get("split") != "train":
            raise ValueError("scale statistics require an explicitly declared train descriptor")
        metadata = descriptor.get("metadata", {})
    else:
        descriptor, metadata, members = None, {}, [(path, None)]
    expected_rows = None
    if protocol_path is not None:
        from element_surrogate.benchmark.protocol import load_protocol

        protocol = load_protocol(protocol_path)
        item = protocol["dataset"]["splits"]["train"]
        expected = (Path(protocol_path).resolve().parent / item["path"]).resolve()
        if path != expected or file_hash(path) != item["sha256"]:
            raise ValueError("cohort is not the frozen protocol training split")
        if len(protocol["coordinates"]) != dimension:
            raise ValueError("cohort dimension differs from frozen protocol")
        expected_rows = item["rows"]
        if descriptor is not None:
            from element_surrogate.benchmark.scoring import _metadata

            _metadata(metadata, protocol)
            for key, value in (
                ("input_names", protocol["inputs"]),
                ("input_units", protocol["input_units"]),
            ):
                if metadata.get(key) != value:
                    raise ValueError(f"cohort {key} differs from frozen protocol")
    # A descriptor supplies units. A legacy numeric NPZ may rely on its caller's
    # native-unit contract; never claim that unprovided units were verified.
    if descriptor is not None:
        require_native_metadata(metadata, dimension)
    rows = 0
    for member_index, (source, prefix) in enumerate(members):
        expected_hash = (
            descriptor["members"][member_index]["sha256"] if descriptor else file_hash(source)
        )
        if file_hash(source) != expected_hash:
            raise ValueError("cohort member changed before reading")
        # Reserve headroom for loaded arrays plus numeric normalization scratch.
        if 4 * uncompressed_bytes(source) > working_memory_bytes:
            raise ValueError(
                "cohort shard exceeds working-memory allowance; reshard or increase it"
            )
        with np.load(source, allow_pickle=False) as data:
            if descriptor is None and any(k in data for k in ("coordinate_units", "input_units")):
                embedded = {
                    k: np.asarray(data[k]).tolist()
                    for k in ("coordinate_names", "coordinate_units", "input_names", "input_units")
                    if k in data
                }
                require_native_metadata(embedded, dimension)
            if protocol_path is not None and descriptor is None:
                from element_surrogate.benchmark.scoring import _metadata

                _metadata(data, protocol)
                for key, value in (
                    ("input_names", protocol["inputs"]),
                    ("input_units", protocol["input_units"]),
                ):
                    if key not in data or np.asarray(data[key]).tolist() != value:
                        raise ValueError(f"cohort {key} differs from frozen protocol")
            x, y = data["inputs"], data["targets"]
            if x.dtype.kind not in "fiu" or y.dtype.kind not in "fiu":
                raise ValueError("numeric inputs and targets required; pickle is never enabled")
            n = len(x) if prefix is None else prefix
            if x.ndim != 2 or y.ndim != 2 or x.shape[1] != dimension + 2:
                raise ValueError("matching native input dimension required")
            if y.shape != (len(x), dimension) or not 0 < n <= len(x):
                raise ValueError("matching targets and valid cohort prefix required")
            for key, value in metadata.items():
                if key in data and np.asarray(data[key]).tolist() != value:
                    raise ValueError(f"cohort member metadata mismatch: {key}")
            for start in range(0, n, batch_rows):
                a, b = x[start : min(start + batch_rows, n)], y[start : min(start + batch_rows, n)]
                if not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError("finite training inputs and targets required")
                rows += len(a)
                yield a, b
        if file_hash(source) != expected_hash:
            raise ValueError("cohort member changed during scale calculation")
    declared = descriptor["rows"] if descriptor is not None else rows
    if rows != declared or (expected_rows is not None and rows != expected_rows):
        raise ValueError("training row count differs from descriptor or frozen protocol")
