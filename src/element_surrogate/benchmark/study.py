"""Persistent bounded studies, validation-only selection and guarded final scoring."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from element_surrogate.file_io import open_guard
from element_surrogate.runs import implementation_identity, sha256_json

from .protocol import (
    check_approvals,
    file_hash,
    load_protocol,
    protocol_identity,
    read_json,
    validate_protocol,
    write_json,
)
from .scoring import _score_predictions, load_cohort, score_predictions


@contextmanager
def _guard(directory):
    """OS-released advisory lock; a killed parent does not leave a stale lock token."""
    path = Path(directory) / "study.guard"
    with open_guard(path) as handle:
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("another process owns this study") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _read_study(path):
    path = Path(path).resolve()
    record = read_json(path)
    if record.get("schema") != "element_surrogate.benchmark.study.v1":
        raise ValueError("unsupported study record")
    p = validate_protocol(record["protocol"])
    if protocol_identity(p) != record["protocol_sha256"]:
        raise ValueError("study protocol changed")
    if sha256_json(implementation_identity()) != record["implementation_sha256"]:
        raise ValueError("study implementation changed; create a new study")
    adapter = Path(record["base"]) / p["reproduction"]["adapter_path"]
    if file_hash(adapter) != p["reproduction"]["adapter_sha256"]:
        raise ValueError("training adapter changed")
    for item in record["approvals"]:
        if file_hash(item["path"]) != item["sha256"]:
            raise ValueError("approval record changed")
    check_approvals(
        p,
        [r["path"] for r in record["approvals"]],
        ("description_target", "region_ranges", "compute_budget"),
    )
    return path.parent, record, p


def create_study(protocol_path, output_dir, approval_paths) -> dict[str, Any]:
    p = load_protocol(protocol_path)
    from element_surrogate.physics_review import require_before_fit

    require_before_fit(p)
    base = Path(protocol_path).resolve().parent
    approvals = check_approvals(
        p, approval_paths, ("description_target", "region_ranges", "compute_budget")
    )
    if file_hash(base / p["reproduction"]["adapter_path"]) != p["reproduction"]["adapter_sha256"]:
        raise ValueError("adapter source hash mismatch")
    train, validation = [load_cohort(p, base, s) for s in ("train", "validation")]
    if set(train["sample_id"]) & set(validation["sample_id"]):
        raise ValueError("train/validation sample ID overlap")
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    record = {
        "schema": "element_surrogate.benchmark.study.v1",
        "protocol": p,
        "protocol_sha256": protocol_identity(p),
        "base": str(base),
        "approvals": approvals,
        "implementation_sha256": sha256_json(implementation_identity()),
    }
    write_json(directory / "study.json", record, exclusive=True)
    write_json(directory / "status.json", {"status": "created", "trials": []})
    return {
        "study_path": str(directory / "study.json"),
        "protocol_sha256": record["protocol_sha256"],
        "candidate_configurations": len(p["candidates"]),
        "planned_trials": sum(len(c["seeds"]) for c in p["candidates"]),
        "test_opened": False,
    }


def _trial_path(directory, candidate, seed):
    return directory / "trials" / candidate["id"] / str(seed)


def run_study(study_path) -> dict[str, Any]:
    """Run each declared local adapter in a worker with a parent-enforced deadline.

    Adapter code is trusted user code, not an OS sandbox. Only train and
    validation paths are supplied to fitting. Failed/incomplete trials are kept.
    """
    directory, record, p = _read_study(study_path)
    with _guard(directory):
        status = read_json(directory / "status.json")
        deadline = status.get(
            "deadline_unix", time.time() + p["resources"]["max_study_wall_seconds"]
        )
        status.update(status="running", deadline_unix=deadline)
        write_json(directory / "status.json", status)
        for candidate in p["candidates"]:
            for seed in candidate["seeds"]:
                target = _trial_path(directory, candidate, seed)
                result_path = target / "trial.json"
                if result_path.exists():
                    prior = read_json(result_path)
                    if prior["status"] in ("complete", "failed", "timed_out", "interrupted"):
                        continue
                    raise ValueError(
                        "interrupted worker needs inspection; no duplicate trial is launched"
                    )
                remaining = deadline - time.time()
                if remaining <= 0:
                    status.update(status="budget_exhausted")
                    write_json(directory / "status.json", status)
                    return study_status(study_path)
                target.mkdir(parents=True, exist_ok=False)
                job = {
                    "study_path": str(Path(study_path).resolve()),
                    "candidate": candidate,
                    "seed": seed,
                    "mode": "fit",
                    "directory": str(target),
                }
                write_json(target / "job.json", job)
                write_json(
                    result_path,
                    {"status": "running", "candidate_id": candidate["id"], "seed": seed},
                )
                environment = dict(os.environ)
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                    environment[name] = str(p["timing"]["threads"])
                command = [
                    sys.executable,
                    "-m",
                    "element_surrogate.benchmark.worker",
                    str(target / "job.json"),
                ]
                try:
                    completed = subprocess.run(
                        command,
                        env=environment,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=min(remaining, candidate["max_wall_seconds"]),
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    )
                    result = read_json(result_path)
                    if completed.returncode or result.get("status") != "complete":
                        write_json(
                            result_path,
                            {
                                **result,
                                "status": "failed",
                                "worker_exit_code": completed.returncode,
                            },
                        )
                except subprocess.TimeoutExpired:
                    # subprocess.run kills and waits for this worker before returning.
                    write_json(
                        result_path,
                        {"status": "timed_out", "candidate_id": candidate["id"], "seed": seed},
                    )
                status["trials"] = _trial_summaries(directory, p)
                write_json(directory / "status.json", status)
        status.update(status="completed")
        write_json(directory / "status.json", status)
    return study_status(study_path)


def _trial_summaries(directory, p):
    rows = []
    for candidate in p["candidates"]:
        for seed in candidate["seeds"]:
            path = _trial_path(directory, candidate, seed) / "trial.json"
            row = read_json(path) if path.exists() else {"status": "not_started"}
            rows.append(
                {
                    "candidate_id": candidate["id"],
                    "seed": seed,
                    "status": row["status"],
                    "trial_path": str(path),
                }
            )
    return rows


def study_status(study_path) -> dict[str, Any]:
    directory, _, p = _read_study(study_path)
    record = read_json(directory / "status.json")
    return {
        "study_path": str(Path(study_path).resolve()),
        "status": record["status"],
        "trials": _trial_summaries(directory, p),
        "selection_locked": (directory / "selection.json").exists(),
    }


def start_study(study_path) -> dict[str, Any]:
    """Launch a hidden coordinator; duplicate tool calls never launch another study."""
    from element_surrogate.jobs import start_existing

    initial = read_json(study_path)
    managed = start_existing(
        "run_benchmark_study",
        Path(study_path).parent,
        {
            "study_path": str(Path(study_path).resolve()),
        },
        initial["protocol"]["resources"]["max_study_wall_seconds"],
    )
    if managed is not None:
        return managed
    directory, _, _ = _read_study(study_path)
    with _guard(directory):
        status = read_json(directory / "status.json")
        if status["status"] != "created":
            return {**study_status(study_path), "launched": False}
        status.update(status="scheduled")
        write_json(directory / "status.json", status)
    # The scheduled state prevents duplicate starts; release the OS guard before
    # the coordinator acquires it, avoiding a startup race.
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "element_surrogate.benchmark.study",
                str(Path(study_path).resolve()),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except OSError:
        with _guard(directory):
            status.update(status="launch_failed")
            write_json(directory / "status.json", status)
        raise
    return {
        "study_path": str(Path(study_path).resolve()),
        "launched": True,
        "coordinator_pid": process.pid,
        "status_path": str(directory / "status.json"),
    }


def _validated_trials(directory, record, p):
    valid, incomplete = [], []
    for candidate in p["candidates"]:
        rows = []
        for seed in candidate["seeds"]:
            folder = _trial_path(directory, candidate, seed)
            path = folder / "trial.json"
            trial = read_json(path) if path.exists() else {"status": "not_started"}
            if trial["status"] != "complete":
                incomplete.append(
                    {"candidate_id": candidate["id"], "seed": seed, "reason": trial["status"]}
                )
                continue
            if (
                trial["protocol_sha256"] != record["protocol_sha256"]
                or trial["candidate"] != candidate
                or trial["seed"] != seed
            ):
                raise ValueError("mixed trial configuration or protocol")
            model = folder / "model.bin"
            if file_hash(model) != trial["checkpoint_sha256"]:
                raise ValueError("trial checkpoint changed")
            result = score_predictions(
                p,
                record["base"],
                "validation",
                folder / "validation.npz",
                trial["checkpoint_sha256"],
            )
            if result != trial["metrics"]:
                raise ValueError("trial metrics changed or do not reproduce")
            cost = trial["timing"]["median_us_per_row"]
            if not np.isfinite(cost) or cost <= 0 or trial["timing"]["contract"] != p["timing"]:
                raise ValueError("invalid or mixed timing contract")
            if not 0 <= trial["best_step"] <= trial["steps"] <= candidate["max_steps"]:
                raise ValueError("trial exceeded step budget")
            rows.append(
                {
                    "candidate_id": candidate["id"],
                    "architecture": candidate["architecture"],
                    "seed": seed,
                    "best_step": trial["best_step"],
                    "primary_score": result["primary_score"],
                    "cost_us_per_row": cost,
                    "checkpoint_sha256": trial["checkpoint_sha256"],
                    "checkpoint_path": str(model),
                    "trial_path": str(path),
                    "trial_sha256": file_hash(path),
                }
            )
        if len(rows) == len(candidate["seeds"]):
            valid.extend(rows)
    if incomplete and p["selection"]["incomplete_policy"] == "refuse":
        raise ValueError("selection refuses incomplete/failed configurations")
    return valid, incomplete


def configuration_summaries(p, trials):
    """Aggregate complete matched replicates without selecting a deployment model."""
    configurations = []
    for candidate in p["candidates"]:
        rows = [r for r in trials if r["candidate_id"] == candidate["id"]]
        if not rows:
            continue
        if len(rows) != len(candidate["seeds"]) or {r["seed"] for r in rows} != set(
            candidate["seeds"]
        ):
            raise ValueError("incomplete or duplicate seeds")
        configurations.append(
            {
                "candidate_id": candidate["id"],
                "architecture": candidate["architecture"],
                "mean_score": float(np.mean([r["primary_score"] for r in rows])),
                "std_score": float(np.std([r["primary_score"] for r in rows], ddof=1))
                if len(rows) > 1
                else None,
                "median_us_per_row": float(np.median([r["cost_us_per_row"] for r in rows])),
                "seed_count": len(rows),
            }
        )
    if not configurations:
        raise ValueError("no complete configuration")
    return configurations


def latency_eligibility(p, configurations, trials):
    """Keep all seed statistics; find the best measured latency-feasible checkpoint."""
    rule = p["selection"]
    if rule.get("policy") != "accuracy_with_latency_budget":
        return None
    ceiling = rule["max_median_us_per_row"]
    exclusive = rule.get("latency_bound", "inclusive") == "exclusive"

    def passes(value):
        return value < ceiling if exclusive else value <= ceiling

    checks = []
    for configuration in configurations:
        rows = [r for r in trials if r["candidate_id"] == configuration["candidate_id"]]
        feasible = [r for r in rows if passes(r["cost_us_per_row"])]
        representative = (
            min(feasible, key=lambda r: (r["primary_score"], r["seed"])) if feasible else None
        )
        aggregate_ok = passes(configuration["median_us_per_row"])
        checks.append(
            {
                "candidate_id": configuration["candidate_id"],
                "configuration_median_us_per_row": configuration["median_us_per_row"],
                "feasible_seeds": [r["seed"] for r in feasible],
                "representative_seed": representative["seed"] if representative else None,
                "representative_median_us_per_row": representative["cost_us_per_row"]
                if representative
                else None,
                "configuration_passes": aggregate_ok,
                "representative_passes": representative is not None,
                "eligible": aggregate_ok and representative is not None,
            }
        )
    return checks


def pareto_frontier(configurations):
    """Observed mean accuracy/configuration-median cost frontier, not a confidence bound."""
    return sorted(
        (
            r
            for r in configurations
            if not any(
                other["mean_score"] <= r["mean_score"]
                and other["median_us_per_row"] <= r["median_us_per_row"]
                and (
                    other["mean_score"] < r["mean_score"]
                    or other["median_us_per_row"] < r["median_us_per_row"]
                )
                for other in configurations
            )
        ),
        key=lambda r: (r["median_us_per_row"], r["mean_score"], r["candidate_id"]),
    )


def choose(p, trials):
    """Configurations are validation hyperparameters; stochastic seeds are replicates."""
    configurations = configuration_summaries(p, trials)
    architectures = {}
    for row in configurations:
        name = row["architecture"]
        previous = architectures.get(name)
        if previous is None or (row["mean_score"], row["candidate_id"]) < (
            previous["mean_score"],
            previous["candidate_id"],
        ):
            architectures[name] = row
    rule = p["selection"]
    eligible_cost = configurations
    checks = latency_eligibility(p, configurations, trials)
    if checks is not None:
        eligible_ids = {r["candidate_id"] for r in checks if r["eligible"]}
        eligible_cost = [r for r in configurations if r["candidate_id"] in eligible_ids]
        if not eligible_cost:
            raise ValueError(
                "no complete configuration meets the declared inference latency budget"
            )
    best = min(r["mean_score"] for r in eligible_cost)
    eligible = [
        r for r in eligible_cost if r["mean_score"] <= best * (1 + rule["relative_accuracy_tie"])
    ]
    selected = min(
        eligible, key=lambda r: (r["median_us_per_row"], r["mean_score"], r["candidate_id"])
    )
    winner_rows = [r for r in trials if r["candidate_id"] == selected["candidate_id"]]
    if checks is not None:
        check = next(r for r in checks if r["candidate_id"] == selected["candidate_id"])
        winner_rows = [r for r in winner_rows if r["seed"] in check["feasible_seeds"]]
    winner = min(
        winner_rows,
        key=lambda r: (r["primary_score"], r["seed"]),
    )
    result = {
        "selected": winner,
        "configurations": configurations,
        "architecture_finalists": list(architectures.values()),
        "selection_rule": rule,
    }
    # Preserve byte-level legacy selection records when no policy was declared.
    # New studies expose the full frontier, including configurations outside a cap.
    if "policy" in rule:
        result["pareto_frontier"] = pareto_frontier(configurations)
        result["latency_eligible_candidate_ids"] = [r["candidate_id"] for r in eligible_cost]
        if checks is not None:
            result["latency_eligibility"] = checks
    return result


def lock_selection(study_path) -> dict[str, Any]:
    directory, record, p = _read_study(study_path)
    with _guard(directory):
        trials, exclusions = _validated_trials(directory, record, p)
        selection = {
            "schema": "element_surrogate.benchmark.selection.v1",
            "protocol_sha256": record["protocol_sha256"],
            "trials": trials,
            "excluded": exclusions,
            "test_used_for_selection": False,
            **choose(p, trials),
        }
        path = directory / "selection.json"
        if path.exists():
            if read_json(path) != selection:
                raise ValueError("validation selection changed after locking")
        else:
            write_json(path, selection, exclusive=True)
    return {
        "selection_path": str(path),
        "selection_sha256": file_hash(path),
        "selected": selection["selected"],
        "excluded": exclusions,
        "test_used_for_selection": False,
    }


def evaluate_test(study_path, approval_paths) -> dict[str, Any]:
    directory, record, p = _read_study(study_path)
    check_approvals(p, approval_paths, ("release",))
    if not (directory / "selection.json").exists():
        raise ValueError("lock validation selection before final test access")
    lock_selection(study_path)  # revalidates source, predictions, metrics and weights
    selected = read_json(directory / "selection.json")["selected"]
    with _guard(directory):
        target = directory / "test"
        if (target / "result.json").exists():
            result = read_json(target / "result.json")
            if result["selection_sha256"] != file_hash(directory / "selection.json"):
                raise ValueError("test result belongs to another selection")
            reproduced = _score_predictions(
                p, record["base"], "test", target / "test.npz", selected["checkpoint_sha256"]
            )
            if result["metrics"] != reproduced:
                raise ValueError("test result changed")
            return {"test_result_path": str(target / "result.json"), "metrics": reproduced}
        if target.exists():
            raise ValueError("incomplete final evaluation needs inspection")
        target.mkdir()
        # Check group/ID disjointness only after test access is authorized.
        test = load_cohort(p, record["base"], "test")
        for split in ("train", "validation"):
            if set(test["sample_id"]) & set(load_cohort(p, record["base"], split)["sample_id"]):
                raise ValueError("test sample ID overlap")
        job = {
            "study_path": str(Path(study_path).resolve()),
            "mode": "test",
            "directory": str(target),
            "selected": selected,
            "selection_sha256": file_hash(directory / "selection.json"),
            "approval_paths": [str(Path(path).resolve()) for path in approval_paths],
        }
        write_json(target / "job.json", job)
        completed = subprocess.run(
            [sys.executable, "-m", "element_surrogate.benchmark.worker", str(target / "job.json")],
            timeout=p["resources"]["max_test_wall_seconds"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if completed.returncode or not (target / "result.json").exists():
            raise ValueError("final evaluation failed; inspect its worker record")
    return evaluate_test(study_path, approval_paths)


def main():
    path = Path(sys.argv[1]).resolve()
    try:
        run_study(path)
    except Exception as exc:
        try:
            with _guard(path.parent):
                status = read_json(path.parent / "status.json")
                write_json(
                    path.parent / "status.json",
                    {**status, "status": "failed", "error": f"{type(exc).__name__}: {exc}"},
                )
        except ValueError:
            pass  # Another coordinator owns the state; never overwrite it.
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
