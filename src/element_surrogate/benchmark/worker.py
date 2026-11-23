"""Isolated deadline-controlled process for a trusted project-local model adapter."""

from __future__ import annotations

import importlib.util
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .protocol import check_approvals, file_hash, read_json, write_json
from .scoring import _score_predictions, load_cohort, score_predictions, summary, write_predictions
from .study import _guard, _read_study


def _module(path):
    spec = importlib.util.spec_from_file_location("element_surrogate_user_adapter", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load declared adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ("build_adapter", "load_adapter"):
        if not callable(getattr(module, name, None)):
            raise ValueError(f"adapter module needs {name}")
    return module


def _output(value, rows, dim):
    predicted = np.asarray(value)
    if (
        predicted.shape != (rows, dim)
        or predicted.dtype.kind not in "fiu"
        or not np.isfinite(predicted).all()
    ):
        raise ValueError("adapter must return complete finite output coordinates")
    return predicted.astype(np.float64)


def _predict(model, inputs, dim):
    return _output(model.predict(inputs), len(inputs), dim)


def _synchronize(model, timing):
    if timing["synchronization"] == "adapter":
        method = getattr(model, "synchronize", None)
        if not callable(method):
            raise ValueError("timing contract requires adapter.synchronize()")
        method()


def _timing(model, inputs, expected, p):
    contract = p["timing"]
    batch = inputs[: contract["batch_size"]]
    golden = expected[: len(batch)]
    for _ in range(contract["warmup_calls"]):
        _predict(model, batch, len(p["coordinates"]))
        _synchronize(model, contract)
    durations = []
    for _ in range(contract["repetitions"]):
        _synchronize(model, contract)
        start = time.perf_counter_ns()
        raw = model.predict(batch)
        _synchronize(model, contract)
        durations.append((time.perf_counter_ns() - start) / 1000 / len(batch))
        actual = _output(raw, len(batch), len(p["coordinates"]))
        if np.any(np.abs(actual - golden) > np.array(contract["golden_atol"])):
            raise ValueError("timed full-map predictions differ from untimed predictions")
    return {
        "contract": contract,
        "raw_us_per_row": durations,
        "median_us_per_row": float(np.median(durations)),
        "p10_us_per_row": float(np.quantile(durations, 0.1)),
        "p90_us_per_row": float(np.quantile(durations, 0.9)),
        "measured_host": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def _execute(job_path):
    job = read_json(job_path)
    directory = Path(job["directory"])
    study_directory, record, p = _read_study(job["study_path"])
    module = _module(Path(record["base"]) / p["reproduction"]["adapter_path"])
    dim = len(p["coordinates"])
    dtype = p["timing"]["input_dtype"]
    if job["mode"] == "fit":
        candidate, seed = job["candidate"], job["seed"]
        train, validation = [load_cohort(p, record["base"], s) for s in ("train", "validation")]
        model = module.build_adapter(
            train["inputs"].astype(dtype), train["targets"], candidate["configuration"], seed
        )
        inputs = validation["inputs"].astype(dtype)
        checkpoint = directory / "model.bin"
        saved_prediction = _predict(model, inputs, dim)
        initial_metrics = summary(
            p, saved_prediction, validation["targets"], validation["setting_id"]
        )
        best_score, best_step = initial_metrics["primary_score"], 0
        history = [{"step": 0, "validation_primary_score": best_score}]
        model.save(checkpoint)
        if (
            not checkpoint.is_file()
            or checkpoint.stat().st_size > p["resources"]["max_checkpoint_bytes"]
        ):
            raise ValueError("adapter must save one bounded initial checkpoint")
        for step in range(1, candidate["max_steps"] + 1):
            model.step()
            predicted = _predict(model, inputs, dim)
            metrics = summary(p, predicted, validation["targets"], validation["setting_id"])
            history.append({"step": step, "validation_primary_score": metrics["primary_score"]})
            if metrics["primary_score"] < best_score:
                best_score, best_step = metrics["primary_score"], step
                model.save(checkpoint)
                if (
                    not checkpoint.is_file()
                    or checkpoint.stat().st_size > p["resources"]["max_checkpoint_bytes"]
                ):
                    raise ValueError(
                        "adapter must save one bounded checkpoint at the supplied path"
                    )
                saved_prediction = predicted.copy()
        best = module.load_adapter(checkpoint, candidate["configuration"])
        reloaded = _predict(best, inputs, dim)
        if np.any(np.abs(reloaded - saved_prediction) > np.array(p["timing"]["golden_atol"])):
            raise ValueError("strict saved/reloaded model prediction check failed")
        checksum = file_hash(checkpoint)
        prediction_path = directory / "validation.npz"
        write_predictions(prediction_path, p, "validation", validation, reloaded, checksum)
        metrics = score_predictions(p, record["base"], "validation", prediction_path, checksum)
        timing = _timing(best, inputs, reloaded, p)
        if file_hash(checkpoint) != checksum:
            raise ValueError("checkpoint mutated during inference")
        _read_study(job["study_path"])
        write_json(
            directory / "trial.json",
            {
                "status": "complete",
                "candidate": candidate,
                "candidate_id": candidate["id"],
                "seed": seed,
                "steps": candidate["max_steps"],
                "best_step": best_step,
                "protocol_sha256": record["protocol_sha256"],
                "checkpoint_sha256": checksum,
                "metrics": metrics,
                "timing": timing,
                "history": history,
                "adapter_sha256": p["reproduction"]["adapter_sha256"],
            },
        )
    elif job["mode"] == "test":
        check_approvals(p, job["approval_paths"], ("release",))
        if file_hash(study_directory / "selection.json") != job["selection_sha256"]:
            raise ValueError("selection lock changed before test evaluation")
        lock = read_json(study_directory / "selection.json")
        selected = job["selected"]
        if (
            lock["selected"] != selected
            or file_hash(selected["checkpoint_path"]) != selected["checkpoint_sha256"]
        ):
            raise ValueError("selected checkpoint identity mismatch")
        candidate = next(c for c in p["candidates"] if c["id"] == selected["candidate_id"])
        test = load_cohort(p, record["base"], "test")
        model = module.load_adapter(Path(selected["checkpoint_path"]), candidate["configuration"])
        predicted = _predict(model, test["inputs"].astype(dtype), dim)
        prediction_path = directory / "test.npz"
        write_predictions(
            prediction_path, p, "test", test, predicted, selected["checkpoint_sha256"]
        )
        metrics = _score_predictions(
            p, record["base"], "test", prediction_path, selected["checkpoint_sha256"]
        )
        write_json(
            directory / "result.json",
            {
                "metrics": metrics,
                "selection_sha256": job["selection_sha256"],
                "release_approvals": [
                    {"path": str(path), "sha256": file_hash(path)} for path in job["approval_paths"]
                ],
            },
        )
    else:
        raise ValueError("unsupported worker mode")


def execute(job_path):
    job = read_json(job_path)
    directory = Path(job["directory"])
    with _guard(directory):
        if job["mode"] == "fit" and read_json(directory / "trial.json")["status"] != "running":
            raise ValueError("trial was already inspected or completed; stale worker refused")
        try:
            return _execute(job_path)
        except Exception as exc:
            name = "trial.json" if job["mode"] == "fit" else "worker_error.json"
            write_json(
                directory / name,
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-10000:],
                },
            )
            raise


def main():
    try:
        execute(Path(sys.argv[1]))
    except Exception as exc:
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
