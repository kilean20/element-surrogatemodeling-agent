import time
from pathlib import Path

import pytest

from element_surrogate import recovery
from element_surrogate.benchmark.protocol import read_json, write_json
from element_surrogate.benchmark.study import _guard, create_study
from tests.benchmark_fixture import make_case


def study(tmp_path):
    p, _, approvals = make_case(tmp_path / "case")
    path = Path(create_study(p, tmp_path / "study", approvals)["study_path"])
    write_json(
        path.parent / "status.json",
        {"status": "failed", "trials": [], "deadline_unix": time.time() + 300},
    )
    trial = path.parent / "trials" / "small" / "1"
    # Use the actual declared identity rather than introducing another trial.
    protocol = read_json(path)["protocol"]
    trial = (
        path.parent
        / "trials"
        / protocol["candidates"][0]["id"]
        / str(protocol["candidates"][0]["seeds"][0])
    )
    trial.mkdir(parents=True)
    write_json(trial / "trial.json", {"status": "running"})
    (trial / "model.bin").write_bytes(b"partial optimizer-free fixture")
    return path, trial


def test_active_worker_refuses_inspection(tmp_path):
    path, trial = study(tmp_path)
    with _guard(trial), pytest.raises(ValueError, match="owns this study"):
        recovery.inspect_study(path, tmp_path / "inspection.json")


def test_inspected_resume_preserves_partial_fit_and_deadline(tmp_path, monkeypatch):
    from element_surrogate.benchmark import study as module

    path, trial = study(tmp_path)
    old_deadline = read_json(path.parent / "status.json")["deadline_unix"]
    recovery.inspect_study(path, tmp_path / "inspection.json")
    monkeypatch.setattr(module, "start_study", lambda p: {"launched": True})
    result = recovery.resume_study(
        path, tmp_path / "inspection.json", "Resume only untouched tests"
    )
    assert result["launched"]
    assert read_json(trial / "trial.json")["status"] == "interrupted"
    assert (trial / "model.bin").read_bytes() == b"partial optimizer-free fixture"
    assert read_json(path.parent / "status.json")["deadline_unix"] == old_deadline
    with pytest.raises(ValueError, match="stale"):
        recovery.resume_study(path, tmp_path / "inspection.json", "again")


def test_exhausted_budget_refuses_resume(tmp_path):
    path, _ = study(tmp_path)
    write_json(path.parent / "status.json", {"status": "failed", "deadline_unix": 1, "trials": []})
    recovery.inspect_study(path, tmp_path / "inspection.json")
    with pytest.raises(ValueError, match="deadline"):
        recovery.resume_study(path, tmp_path / "inspection.json", "test")


def test_generation_recovery_bound_to_bytes_and_original_deadline(tmp_path, monkeypatch):
    from element_surrogate import generation_job
    from tests.test_workflow_tools import file_case
    from tests.workflow_fixture import approvals

    path, _, plan = file_case(tmp_path)
    monkeypatch.setattr(generation_job, "_launch", lambda p: {"job_path": str(p)})
    started = generation_job.start_generation(
        path,
        tmp_path / "qualification.json",
        tmp_path / "plan.json",
        tmp_path / "data",
        approvals(tmp_path / "decisions", plan),
    )
    original = read_json(started["job_path"])
    recovery.inspect_generation(tmp_path / "data", tmp_path / "inspect.json")
    resumed = recovery.resume_generation(
        tmp_path / "data", tmp_path / "inspect.json", "Recover before first batch"
    )
    job = read_json(resumed["job_path"])
    assert job["deadline_unix"] == original["deadline_unix"]
    assert job["job_id"] != original["job_id"]
    with pytest.raises(ValueError, match="stale generation"):
        generation_job._active(original)
    with pytest.raises(ValueError, match="stale recovery"):
        recovery.resume_generation(tmp_path / "data", tmp_path / "inspect.json", "again")
