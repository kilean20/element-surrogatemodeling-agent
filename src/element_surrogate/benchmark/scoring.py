"""ID-joined dimensional errors for explicitly bounded local NPZ cohorts."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

from element_surrogate.regression import RegressionMetrics

from .protocol import file_hash, protocol_identity


def _npz(path, protocol):
    path = Path(path)
    maximum = protocol["resources"]["max_npz_uncompressed_bytes"]
    if path.stat().st_size > maximum:
        raise ValueError("NPZ exceeds approved memory budget")
    if path.suffix.lower() == ".json":
        from element_surrogate.cohort_io import descriptor_members

        descriptor, checked_members = descriptor_members(path)
        # Bound all decompressed source members plus selected concatenation before reading.
        total_bytes = 0
        for source, _ in checked_members:
            with zipfile.ZipFile(source) as archive:
                total_bytes += sum(i.file_size for i in archive.infolist())
        if total_bytes * 3 > maximum:
            raise ValueError(
                "sharded cohort exceeds approved decompression/concatenation memory budget"
            )
        chunks = []
        for source, count in checked_members:
            with np.load(source, allow_pickle=False) as data:
                if len(data["sample_id"]) < count:
                    raise ValueError("cohort prefix exceeds source member")
                chunks.append(
                    {k: data[k][:count] for k in ("inputs", "targets", "sample_id", "setting_id")}
                )
        if not chunks or sum(len(c["sample_id"]) for c in chunks) != descriptor["rows"]:
            raise ValueError("cohort descriptor row count differs from members")
        return {
            **{k: np.asarray(v) for k, v in descriptor["metadata"].items()},
            **{k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]},
        }
    with zipfile.ZipFile(path) as archive:
        if sum(i.file_size for i in archive.infolist()) > maximum:
            raise ValueError("NPZ uncompressed arrays exceed approved memory budget")
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _text_ids(value, label, rows):
    a = np.asarray(value)
    if a.ndim != 1 or len(a) != rows or a.dtype.kind not in "iuUS":
        raise ValueError(f"{label}: integer or text vector required")
    if a.dtype.kind in "iu" and np.any(a < 0):
        raise ValueError(f"{label}: negative ID")
    return a.astype(str)


def _metadata(data, protocol):
    expected = {
        "dataset_identity_sha256": protocol["dataset"]["identity_sha256"],
        "coordinate_names": [c["name"] for c in protocol["coordinates"]],
        "coordinate_units": [c["unit"] for c in protocol["coordinates"]],
    }
    for name, value in expected.items():
        if name not in data or np.asarray(data[name]).tolist() != value:
            raise ValueError(f"artifact {name} does not match protocol")


def load_cohort(protocol, base, split):
    """Internal reader. Public test access is guarded in the study API."""
    item = protocol["dataset"]["splits"][split]
    path = (Path(base) / item["path"]).resolve()
    if file_hash(path) != item["sha256"]:
        raise ValueError(f"{split} cohort changed")
    data = _npz(path, protocol)
    _metadata(data, protocol)
    rows, dim = item["rows"], len(protocol["coordinates"])
    if np.asarray(data.get("input_names")).tolist() != protocol["inputs"]:
        raise ValueError("cohort inference inputs differ from protocol")
    if np.asarray(data.get("input_units")).tolist() != protocol["input_units"]:
        raise ValueError("cohort input units differ from protocol")
    for key, columns in (("inputs", len(protocol["inputs"])), ("targets", dim)):
        a = np.asarray(data[key])
        if a.shape != (rows, columns) or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
            raise ValueError(f"{split}: complete finite {key} required")
    ids = _text_ids(data["sample_id"], "sample_id", rows)
    if len(np.unique(ids)) != rows:
        raise ValueError("duplicate truth sample IDs")
    settings = _text_ids(data["setting_id"], "setting_id", rows)
    if set(settings) != set(item["groups"]):
        raise ValueError("cohort groups differ from the frozen split")
    data.update(sample_id=ids, setting_id=settings)
    order = np.argsort(ids)
    for key in ("inputs", "targets", "sample_id", "setting_id"):
        data[key] = data[key][order]
    return data


def summary(protocol, prediction, truth, settings):
    accumulator = RegressionMetrics(tuple(c["unit"] for c in protocol["coordinates"]))
    accumulator.update(prediction, truth)
    result = accumulator.summary(quantile_method="inverted_cdf")
    absolute = np.abs(np.asarray(prediction) - truth)
    result["p95_absolute"] = np.quantile(absolute, 0.95, axis=0, method="inverted_cdf").tolist()
    result["p999_absolute"] = np.quantile(absolute, 0.999, axis=0, method="inverted_cdf").tolist()
    settings = np.asarray(settings)
    if settings.shape != (len(truth),):
        raise ValueError("one setting ID per truth row required")
    groups, inverse, counts = np.unique(settings, return_inverse=True, return_counts=True)
    per_group = np.stack(
        [
            np.sqrt(np.bincount(inverse, weights=absolute[:, j] ** 2) / counts)
            for j in range(absolute.shape[1])
        ],
        axis=1,
    )
    group_metrics = [
        {"setting_id": str(group), "rows": int(counts[i]), "rmse": per_group[i].tolist()}
        for i, group in enumerate(groups)
    ]
    balanced = np.mean([r["rmse"] for r in group_metrics], axis=0)
    scales = np.array([c["error_scale"] for c in protocol["coordinates"]])
    selected = (
        result["rmse"] if protocol["selection"]["primary"] == "worst_scaled_rmse" else balanced
    )
    result.update(
        coordinate_names=[c["name"] for c in protocol["coordinates"]],
        setting_count=len(group_metrics),
        setting_metrics=group_metrics,
        setting_mean_rmse=balanced.tolist(),
        primary_score=float(np.max(np.asarray(selected) / scales)),
        coverage={
            "n_truth": len(truth),
            "n_scored": len(prediction),
            "n_truth_without_prediction": 0,
            "n_predictions_without_truth": 0,
        },
    )
    return result


def prediction_metadata(protocol, split, checkpoint_sha256):
    return {
        "protocol_sha256": protocol_identity(protocol),
        "split": split,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_identity_sha256": protocol["dataset"]["identity_sha256"],
        "coordinate_names": [c["name"] for c in protocol["coordinates"]],
        "coordinate_units": [c["unit"] for c in protocol["coordinates"]],
    }


def write_predictions(path, protocol, split, cohort, prediction, checkpoint_sha256):
    with Path(path).open("wb") as handle:
        np.savez_compressed(
            handle,
            sample_id=cohort["sample_id"],
            prediction=prediction,
            **prediction_metadata(protocol, split, checkpoint_sha256),
        )


def score_predictions(protocol, base, split, prediction_path, checkpoint_sha256):
    """Public scorer for train/validation. Test scoring requires the locked study API."""
    if split not in ("train", "validation"):
        raise ValueError("test scoring requires the locked study and release approval")
    return _score_predictions(protocol, base, split, prediction_path, checkpoint_sha256)


def _score_predictions(protocol, base, split, prediction_path, checkpoint_sha256):
    """Internal scoring after the study's access/identity checks."""
    truth = load_cohort(protocol, base, split)
    data = _npz(prediction_path, protocol)
    _metadata(data, protocol)
    for key, value in prediction_metadata(protocol, split, checkpoint_sha256).items():
        if np.asarray(data.get(key)).tolist() != value:
            raise ValueError(f"prediction identity mismatch: {key}")
    ids = _text_ids(data["sample_id"], "prediction IDs", len(truth["sample_id"]))
    if len(np.unique(ids)) != len(ids) or set(ids) != set(truth["sample_id"]):
        raise ValueError("predictions must cover each truth ID exactly once")
    predictions = np.asarray(data["prediction"])
    if predictions.shape != truth["targets"].shape or predictions.dtype.kind not in "fiu":
        raise ValueError("prediction shape differs from coordinate contract")
    order, target_order = np.argsort(ids), np.argsort(truth["sample_id"])
    result = summary(
        protocol,
        predictions[order],
        truth["targets"][target_order],
        truth["setting_id"][target_order],
    )
    result.update(
        schema="element_surrogate.benchmark.metrics.v1",
        split=split,
        protocol_sha256=protocol_identity(protocol),
        checkpoint_sha256=checkpoint_sha256,
        predictions_sha256=file_hash(prediction_path),
        cohort_sha256=protocol["dataset"]["splits"][split]["sha256"],
        dataset_identity_sha256=protocol["dataset"]["identity_sha256"],
    )
    return result
