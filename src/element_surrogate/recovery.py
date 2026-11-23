"""Explicit recovery of verified artifacts under the original wall deadline.

Locks test process ownership, not PID existence or elapsed-time guesses. A
receipt describes current bytes; it is inspection evidence, not user approval.
"""

import time
from contextlib import ExitStack
from pathlib import Path

from element_surrogate.benchmark.protocol import file_hash, read_json, write_json
from element_surrogate.benchmark.study import _guard, _read_study, _trial_summaries


def _snapshot(directory):
    """Metadata/checkpoint identities without opening sealed particle arrays."""
    return [
        {"path": str(p.relative_to(directory)), "sha256": file_hash(p)}
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.suffix not in (".guard", ".tmp") and ".inspection" not in p.parts
    ]


def _study_inspection(path):
    directory, _, p = _read_study(path)
    status = read_json(directory / "status.json")
    trials = _trial_summaries(directory, p)
    with ExitStack() as stack:
        stack.enter_context(_guard(directory))
        for row in trials:
            folder = Path(row["trial_path"]).parent
            if folder.exists():
                stack.enter_context(_guard(folder))
        # Recheck bytes only while coordinators and workers are excluded.
        status = read_json(directory / "status.json")
        snapshot = _snapshot(directory)
    return {
        "schema": "element_surrogate.recovery.v1",
        "kind": "benchmark",
        "path": str(Path(path).resolve()),
        "snapshot": snapshot,
        "status": status,
        "trials": trials,
        "deadline_unix": status.get("deadline_unix"),
        "selection_locked": (directory / "selection.json").exists(),
        "policy": "Preserve interrupted fits; only untouched trials resume. No budget extension.",
    }


def inspect_study(path, output_path) -> dict:
    record = _study_inspection(path)
    output = Path(output_path).resolve()
    if output.is_relative_to(Path(path).resolve().parent):
        raise ValueError("inspection receipt must be outside the study directory")
    write_json(output, record, exclusive=True)
    return {
        "inspection_path": str(output),
        "sha256": file_hash(output),
        "remaining_wall_seconds": max(0, (record["deadline_unix"] or time.time()) - time.time()),
        "trials": record["trials"],
        "policy": record["policy"],
    }


def resume_study(path, inspection_path, rationale) -> dict:
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("explicit recovery rationale required")
    receipt = read_json(inspection_path)
    current = _study_inspection(path)
    if current != receipt:
        raise ValueError("stale recovery inspection; inspect current artifacts again")
    if current["selection_locked"]:
        raise ValueError("selected study cannot be resumed")
    deadline = current["deadline_unix"]
    if deadline is None or deadline <= time.time():
        raise ValueError("original deadline absent or exhausted; a new approved study is required")
    directory = Path(path).resolve().parent
    with ExitStack() as stack:
        stack.enter_context(_guard(directory))
        for row in current["trials"]:
            folder = Path(row["trial_path"]).parent
            if folder.exists():
                stack.enter_context(_guard(folder))
        if _snapshot(directory) != current["snapshot"]:
            raise ValueError("artifacts changed after inspection")
        for row in current["trials"]:
            if row["status"] == "running" or (
                row["status"] == "not_started" and Path(row["trial_path"]).parent.exists()
            ):
                trial = Path(row["trial_path"])
                write_json(
                    trial,
                    {
                        **(read_json(trial) if trial.exists() else {}),
                        "status": "interrupted",
                        "recovery_rationale": rationale,
                        "optimizer_resume_available": False,
                    },
                )
        status = read_json(directory / "status.json")
        history = status.get("recovery_history", []) + [
            {
                "inspection_sha256": file_hash(inspection_path),
                "rationale": rationale,
                "timestamp_unix": time.time(),
            }
        ]
        write_json(
            directory / "status.json", {**status, "status": "created", "recovery_history": history}
        )
    from element_surrogate.benchmark.study import start_study
    from element_surrogate.jobs import find_campaign

    if find_campaign(directory) is not None:
        return {
            "study_path": str(Path(path).resolve()),
            "status": "recovery_prepared",
            "managed_resume_required": True,
            "next_step": (
                "inspect and resume the existing scientific job; original deadlines "
                "and interrupted-fit dispositions remain"
            ),
        }
    return start_study(path)


def _generation_inspection(directory):
    from element_surrogate.data import verify_generation
    from element_surrogate.generation_job import _verify
    from element_surrogate.workflow import validate_generation_files

    directory = Path(directory).resolve()
    supervisor = directory / ".supervisor"
    supervisor.mkdir(exist_ok=True)
    with _guard(supervisor), _guard(directory):
        job = read_json(directory / "job.json")
        _verify(job)
        validate_generation_files(
            job["transport_path"], job["qualification_path"], job["plan_path"]
        )
        generation = (
            verify_generation(directory) if (directory / "generation.json").exists() else None
        )
        return {
            "schema": "element_surrogate.recovery.v1",
            "kind": "generation",
            "path": str(directory),
            "job": job,
            "status": read_json(directory / "job_status.json"),
            "snapshot": _snapshot(directory),
            "generation_status": generation["status"] if generation else "not_started",
            "policy": (
                "Verify committed shards; replay only uncommitted batches. Or"
                "iginal wall deadline includes downtime."
            ),
        }


def inspect_generation(directory, output_path) -> dict:
    directory, output = Path(directory).resolve(), Path(output_path).resolve()
    if output.is_relative_to(directory):
        raise ValueError("inspection receipt must be outside generation directory")
    record = _generation_inspection(directory)
    write_json(output, record, exclusive=True)
    return {
        "inspection_path": str(output),
        "sha256": file_hash(output),
        "generation_status": record["generation_status"],
        "policy": record["policy"],
        "remaining_wall_seconds": max(0, record["job"]["deadline_unix"] - time.time()),
    }


def resume_generation(directory, inspection_path, rationale) -> dict:
    import uuid

    from element_surrogate.generation_job import _launch

    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("explicit recovery rationale required")
    current = _generation_inspection(directory)
    if current != read_json(inspection_path):
        raise ValueError("stale recovery inspection; inspect current artifacts again")
    if current["generation_status"] in (
        "complete",
        "invalidated",
        "proposal_budget_exhausted",
        "wall_budget_exhausted",
    ):
        raise ValueError("terminal generation requires a new approved campaign")
    job = current["job"]
    if job["deadline_unix"] <= time.time():
        raise ValueError("original generation deadline exhausted; new approved campaign required")
    directory = Path(directory).resolve()
    with _guard(directory / ".supervisor"), _guard(directory):
        if _snapshot(directory) != current["snapshot"]:
            raise ValueError("artifacts changed after inspection")
        job = {
            **job,
            "job_id": uuid.uuid4().hex,
            "recovered_from": current["job"]["job_id"],
            "recovery_rationale": rationale,
            "inspection_sha256": file_hash(inspection_path),
        }
        launch_dir = directory / "launches"
        launch_dir.mkdir(exist_ok=True)
        path = launch_dir / (job["job_id"] + ".json")
        write_json(path, job, exclusive=True)
        write_json(directory / "job.json", job)
        write_json(
            directory / "job_status.json",
            {
                "status": "scheduled",
                "job_path": str(path),
                "job_id": job["job_id"],
                "deadline_unix": job["deadline_unix"],
            },
        )
    return _launch(path)
