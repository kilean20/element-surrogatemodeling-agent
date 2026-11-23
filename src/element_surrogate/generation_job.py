"""Background generation with a supervising process deadline.

The generation child commits only complete shards. A timeout terminates and
waits for that child; committed batches remain resumable. Partial/orphan output
is inspected or verified, never silently counted. No unmanaged grandchildren.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

from element_surrogate.benchmark.protocol import check_approvals, file_hash, read_json, write_json
from element_surrogate.benchmark.study import _guard
from element_surrogate.data import generation_summary
from element_surrogate.file_io import artifact_guard
from element_surrogate.workflow import generate_from_files, validate_generation_files


def start_generation(
    transport_path, qualification_path, plan_path, output_dir, approval_paths
) -> dict:
    """Start one locally supervised job after exact preflight and user attestations."""
    from element_surrogate.jobs import start_existing

    managed = start_existing(
        "generate_training_data",
        output_dir,
        {
            "transport_path": str(Path(transport_path).resolve()),
            "qualification_path": str(Path(qualification_path).resolve()),
            "plan_path": str(Path(plan_path).resolve()),
            "output_dir": str(Path(output_dir).resolve()),
            "approval_paths": [str(Path(p).resolve()) for p in approval_paths],
        },
        read_json(plan_path)["max_wall_seconds"],
    )
    if managed is not None:
        return managed
    validate_generation_files(transport_path, qualification_path, plan_path)
    plan = read_json(plan_path)
    check_approvals(plan, approval_paths, ("description_target", "region_ranges", "compute_budget"))
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / "job_status.json"
    job_path = directory / "job.json"
    with artifact_guard(directory / "launch", create=True):
        if status_path.exists():
            old_job = read_json(job_path)
            _verify(old_job)
            expected_paths = [
                str(Path(p).resolve())
                for p in (transport_path, qualification_path, plan_path, *approval_paths)
            ]
            if [r["path"] for r in old_job["inputs"]] != expected_paths:
                raise ValueError("generation directory belongs to another job specification")
            previous = read_json(status_path)
            if previous["status"] in ("scheduled", "running"):
                raise ValueError("generation job already scheduled or running; inspect status")
            if previous["status"] not in ("complete", "paused_at_batch_boundary"):
                raise ValueError("failed/interrupted job needs inspection before a new launch")
            if previous["status"] == "complete":
                return generation_status(directory)
            raise ValueError("paused job requires explicit inspection and resume")
        inputs = [
            Path(p).resolve()
            for p in (transport_path, qualification_path, plan_path, *approval_paths)
        ]
        job = {
            "transport_path": str(inputs[0]),
            "qualification_path": str(inputs[1]),
            "plan_path": str(inputs[2]),
            "output_dir": str(directory),
            "approval_paths": [str(p) for p in inputs[3:]],
            "inputs": [{"path": str(p), "sha256": file_hash(p)} for p in inputs],
            "max_wall_seconds": plan["max_wall_seconds"],
            "deadline_unix": time.time() + plan["max_wall_seconds"],
            "job_id": uuid.uuid4().hex,
        }
        launch_dir = directory / "launches"
        launch_dir.mkdir(exist_ok=True)
        immutable_path = launch_dir / (job["job_id"] + ".json")
        write_json(immutable_path, job, exclusive=True)
        write_json(job_path, job)
        write_json(
            status_path,
            {
                "status": "scheduled",
                "job_path": str(immutable_path),
                "job_id": job["job_id"],
                "deadline_unix": job["deadline_unix"],
            },
        )
    return _launch(immutable_path)


def _launch(job_path) -> dict:
    job = read_json(job_path)
    directory = Path(job["output_dir"])
    status_path = directory / "job_status.json"
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "element_surrogate.generation_job", "supervise", str(job_path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except Exception as error:
        try:
            with _guard(directory):
                current = read_json(status_path)
                if current.get("job_id") == job["job_id"]:
                    write_json(
                        status_path, {**current, "status": "launch_failed", "error": str(error)}
                    )
        except ValueError:
            pass  # Another child owns generation; never overwrite its launch status.
        raise
    return {"job_path": str(job_path), "supervisor_pid": process.pid, "status": "scheduled"}


def generation_status(directory) -> dict:
    directory = Path(directory).resolve()
    result: dict = read_json(directory / "job_status.json")
    if (directory / "generation.json").exists():
        result["generation"] = generation_summary(directory)
    return result


def _verify(job):
    if any(file_hash(r["path"]) != r["sha256"] for r in job["inputs"]):
        raise ValueError("generation job input changed after scheduling")


def _active(job):
    status = read_json(Path(job["output_dir"]) / "job_status.json")
    if status.get("job_id") != job["job_id"]:
        raise ValueError("stale generation launch refused")
    if time.time() >= job["deadline_unix"]:
        raise ValueError("original generation deadline exhausted")


def execute(path):
    job = read_json(path)
    try:
        _verify(job)
        _active(job)
        result = generate_from_files(
            **{
                k: job[k]
                for k in (
                    "transport_path",
                    "qualification_path",
                    "plan_path",
                    "output_dir",
                    "approval_paths",
                )
            },
            execution_check=lambda: _active(job),
        )
        _verify(job)
        write_json(Path(job["output_dir"]) / "launches" / (job["job_id"] + "_result.json"), result)
    except Exception:
        write_json(
            Path(job["output_dir"]) / "launches" / (job["job_id"] + "_failure.json"),
            {"error": traceback.format_exc()[-8000:]},
        )
        raise


def _supervise(path):
    job = read_json(path)
    directory = Path(job["output_dir"])
    status_path = directory / "job_status.json"
    start = time.monotonic()
    if read_json(status_path).get("job_id") != job["job_id"]:
        return
    try:
        _verify(job)
        # Check the lease before writing: an old scheduled supervisor cannot
        # replace the status of an explicitly resumed launch.
        if read_json(status_path).get("job_id") != job["job_id"]:
            return
        remaining = job["deadline_unix"] - time.time()
        if remaining <= 0:
            raise subprocess.TimeoutExpired("original generation deadline", 0)
        write_json(
            status_path,
            {**read_json(status_path), "status": "running", "supervisor_pid": os.getpid()},
        )
        completed = subprocess.run(
            [sys.executable, "-m", "element_surrogate.generation_job", "execute", str(path)],
            timeout=remaining,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if completed.returncode:
            raise ValueError(
                "generation child failed; inspect its launch failure record and committed shards"
            )
        result_path = directory / "launches" / (job["job_id"] + "_result.json")
        result = read_json(result_path)
        write_json(
            status_path,
            {
                **read_json(status_path),
                "status": result["status"],
                "job_result_path": str(result_path),
                "wall_seconds": time.monotonic() - start,
            },
        )
    except subprocess.TimeoutExpired:
        # subprocess.run kills and waits for its direct child before raising.
        write_json(
            status_path,
            {
                **read_json(status_path),
                "status": "timed_out",
                "wall_seconds": time.monotonic() - start,
                "inspection": "verify committed shards before explicitly resuming",
            },
        )
    except Exception:
        write_json(
            status_path,
            {
                **read_json(status_path),
                "status": "failed",
                "error": traceback.format_exc()[-8000:],
                "wall_seconds": time.monotonic() - start,
            },
        )


def supervise(path):
    job = read_json(path)
    directory = Path(job["output_dir"]) / ".supervisor"
    directory.mkdir(exist_ok=True)
    with _guard(directory):
        _supervise(path)


if __name__ == "__main__":
    mode, path = sys.argv[1:]
    if mode == "execute":
        execute(path)
    elif mode == "supervise":
        supervise(path)
    else:
        raise ValueError("unsupported generation job mode")
