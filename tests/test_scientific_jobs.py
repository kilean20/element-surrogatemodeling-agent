"""Lifecycle/resources are scientific controls; no facility payloads or live agents."""

import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import pytest

from element_surrogate import jobs
from element_surrogate.benchmark.protocol import read_json, write_json
from element_surrogate.project import initialize


def campaign(tmp_path, *, seconds=120):
    initialize(tmp_path, "manufactured resource test")
    policy = {
        "schema": jobs.CAMPAIGN_SCHEMA,
        "limits": {
            "threads": 1,
            "working_memory_bytes": 256 << 20,
            "scratch_bytes": 10 << 20,
            "max_total_job_wall_seconds": seconds,
        },
        "user_authority": {
            "actor": "synthetic fixture",
            "user_statement": "Test-only explicit budget",
            "source_reference": "manufactured acceptance fixture",
        },
    }
    path = tmp_path / "policy.json"
    write_json(path, policy)
    return jobs.configure(tmp_path, path)["campaign_path"], policy


def request(path, *, maximum=30):
    return {
        "schema": jobs.JOB_SCHEMA,
        "operation": "inspect_file",
        "arguments": {"path": str(path)},
        "resources": {"threads": 1, "working_memory_bytes": 256 << 20, "scratch_bytes": 0},
        "max_wall_seconds": maximum,
        "time_policy": "active",
    }


def wait(path, handle):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        row = jobs.job_status(path, handle["job_id"])
        if row["status"] in jobs.TERMINAL | {"interrupted"} and not jobs._alive(
            row.get("supervisor")
        ):
            return row
        time.sleep(0.1)
    pytest.fail("fixture job did not terminate within test-only allowance")


def test_policy_explicit_immutable_and_multi_day_budget(tmp_path):
    path, policy = campaign(tmp_path, seconds=172800)
    assert jobs.campaign_status(path)["limits"]["max_total_job_wall_seconds"] == 172800
    assert jobs.configure(tmp_path, tmp_path / "policy.json")["campaign_path"] == path
    policy["limits"]["threads"] = 2
    write_json(tmp_path / "policy.json", policy)
    with pytest.raises(ValueError, match="immutable"):
        jobs.configure(tmp_path, tmp_path / "policy.json")
    # This validates duration representation, not a claimed two-day execution.


def test_managed_job_detaches_from_exiting_client_and_queues_without_duplicate(tmp_path):
    path, _ = campaign(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("manufactured source")
    spec = request(source)
    write_json(tmp_path / "job.json", spec)
    code = (
        "from element_surrogate.jobs import submit; import json; print(json.dumps(submit("
        + repr(path)
        + ","
        + repr(str(tmp_path / "job.json"))
        + ")))"
    )
    client = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    handle = json.loads(client.stdout)
    assert jobs.submit_request(path, spec)["job_id"] == handle["job_id"]
    changed = deepcopy(spec)
    changed["max_wall_seconds"] += 1
    with pytest.raises(ValueError, match="same scientific work"):
        jobs.submit_request(path, changed)
    other = tmp_path / "second.txt"
    other.write_text("independent operation")
    second = jobs.submit_request(path, request(other))
    first_row, second_row = wait(path, handle), wait(path, second)
    assert first_row["status"] == second_row["status"] == "complete", (first_row, second_row)
    assert first_row["completed_unix"] <= second_row["attempt_started_unix"]
    assert read_json(first_row["result_path"])["size_bytes"] == source.stat().st_size
    assert first_row["charged_job_wall_seconds"] > 0
    assert jobs.campaign_status(path)["reserved_resources"]["threads"] == 0


def test_reject_overcommitted_defaults_outputs_and_unknown_operation(tmp_path, monkeypatch):
    path, _ = campaign(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("fixture")
    monkeypatch.setattr(jobs, "_launch", lambda p, identifier: {"job_id": identifier})
    spec = request(source)
    spec["resources"]["threads"] = 2
    with pytest.raises(ValueError, match="resources exceed"):
        jobs.submit_request(path, spec)
    spec = request(source)
    spec["operation"] = "ordinary_shell"
    with pytest.raises(ValueError, match="catalog"):
        jobs.submit_request(path, spec)
    spec = request(source)
    spec.update(
        operation="collect_field_readiness",
        arguments={"path": str(source), "shells": 2, "output_path": str(tmp_path / "out.json")},
    )
    spec["resources"]["working_memory_bytes"] = 1
    with pytest.raises(ValueError, match="working-array target"):
        jobs.submit_request(path, spec)
    spec.update(
        operation="qualify_element_transport",
        arguments={
            "path": str(source),
            "request_path": str(source),
            "output_path": str(tmp_path.parent / "outside.json"),
        },
    )
    with pytest.raises(ValueError, match="outputs must stay"):
        jobs.submit_request(path, spec)


def test_changed_inputs_queued_cancellation_and_exhausted_campaign(tmp_path, monkeypatch):
    path, _ = campaign(tmp_path, seconds=30)
    source = tmp_path / "source.txt"
    source.write_text("fixture")
    monkeypatch.setattr(jobs, "_launch", lambda p, identifier: {"job_id": identifier})
    handle = jobs.submit_request(path, request(source))
    row = jobs.job_status(path, handle["job_id"])
    spec = read_json(Path(row["job_directory"]) / "request.json")
    source.write_text("changed fixture")
    with pytest.raises(ValueError, match="changed since"):
        jobs._verify_inputs(spec)
    campaign_record, _, _ = jobs._context(path)
    ledger = {
        "jobs": {handle["job_id"]: {**row, "charged_job_wall_seconds": 30, "status": "failed"}}
    }
    next_row = {**row, "id": "b" * 32, "submitted_unix": time.time() + 1}
    assert not jobs._admit(campaign_record, ledger, next_row)
    assert next_row["status"] == "refused_budget"
    pending = {**row, "cancel_requested": True}
    assert not jobs._admit(campaign_record, {"jobs": {}}, pending)
    assert pending["status"] == "cancelled"


def test_process_identity_and_owned_descendant_cleanup():
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    identity = jobs._identity(process.pid)
    try:
        assert jobs._alive(identity)
        # A reused PID has a different stable start identity. Linux wall creation
        # time is informational because clock corrections can move it.
        key = "kernel_start_ticks" if "kernel_start_ticks" in identity else "created_unix"
        assert not jobs._alive({**identity, key: identity[key] - 1})
        jobs._stop_tree(identity)
        process.wait(timeout=5)
        assert not jobs._alive(identity)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_uncertain_lost_observers_are_not_a_free_retry(tmp_path, monkeypatch):
    path, _ = campaign(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("fixture")
    monkeypatch.setattr(jobs, "_launch", lambda p, identifier: {"job_id": identifier})
    handle = jobs.submit_request(path, request(source))
    _, ledger_path, _ = jobs._context(path)
    ledger = read_json(ledger_path)
    ledger["jobs"][handle["job_id"]].update(status="running", worker=None, supervisor=None)
    write_json(ledger_path, ledger)
    row = jobs.job_status(path, handle["job_id"])
    assert row["accounting_uncertain"] and row["charged_job_wall_seconds"] == 30
    receipt = tmp_path / "inspection.json"
    jobs.inspect_recovery(path, handle["job_id"], receipt)
    with pytest.raises(ValueError, match="no verified original"):
        jobs.resume(path, handle["job_id"], receipt, "fixture recovery")


def test_configured_heavy_tools_require_managed_path(tmp_path, monkeypatch):
    campaign(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ELEMENT_SURROGATE_MANAGED_JOB", raising=False)
    from element_surrogate.tools import inspect_file

    with pytest.raises(ValueError, match="start_scientific_job"):
        inspect_file(str(tmp_path / "policy.json"))
    assert "collect_field_readiness" in jobs.catalog()["operations"]
    # This guard cooperates with the client, not a claim over arbitrary Python.


def test_partial_generation_resumes_same_id_budget_and_committed_bytes(tmp_path):
    from element_surrogate.data import verify_generation
    from tests.test_workflow_tools import file_case
    from tests.workflow_fixture import approvals, manufactured_transport

    transport, _, plan = file_case(tmp_path)
    plan["settings"] = manufactured_transport()[3]
    write_json(tmp_path / "plan.json", plan)
    decisions = approvals(tmp_path / "decisions", plan)
    path, _ = campaign(tmp_path, seconds=180)
    spec = request(tmp_path / "plan.json", maximum=120)
    spec.update(
        operation="generate_training_data",
        arguments={
            "transport_path": str(transport),
            "qualification_path": str(tmp_path / "qualification.json"),
            "plan_path": str(tmp_path / "plan.json"),
            "output_dir": str(tmp_path / "particles"),
            "approval_paths": [str(p) for p in decisions],
            "max_batches": 1,
        },
    )
    handle = jobs.submit_request(path, spec)
    first = wait(path, handle)
    assert first["status"] == "incomplete", first
    committed = verify_generation(tmp_path / "particles")
    assert sum(committed["survivors"].values()) == 17
    inspection = tmp_path / "resume.json"
    jobs.inspect_recovery(path, handle["job_id"], inspection)
    resumed = jobs.resume(
        path, handle["job_id"], inspection, "continue synthetic committed batches"
    )
    assert resumed["job_id"] == handle["job_id"]
    second = wait(path, handle)
    current = verify_generation(tmp_path / "particles")
    assert current["shards"][0] == committed["shards"][0]
    assert sum(current["survivors"].values()) == 32
    assert len(current["shards"]) == 2
    assert second["charged_job_wall_seconds"] > first["charged_job_wall_seconds"]
    assert second["max_wall_seconds"] == first["max_wall_seconds"]
    assert second["result_path"] != first["result_path"] and Path(first["result_path"]).exists()
    with pytest.raises(ValueError, match="stale inspection"):
        jobs.resume(path, handle["job_id"], inspection, "must reject old receipt")


def test_cancel_running_owned_numerical_tree_before_releasing_resources(tmp_path):
    from tests.test_workflow_tools import file_case
    from tests.workflow_fixture import approvals, manufactured_transport

    transport, _, plan = file_case(tmp_path)
    plan.update(
        settings=manufactured_transport()[3], batch_size=20000, max_proposals_per_setting=20000
    )
    write_json(tmp_path / "plan.json", plan)
    decisions = approvals(tmp_path / "decisions", plan)
    path, _ = campaign(tmp_path, seconds=180)
    spec = request(tmp_path / "plan.json", maximum=120)
    spec.update(
        operation="generate_training_data",
        arguments={
            "transport_path": str(transport),
            "qualification_path": str(tmp_path / "qualification.json"),
            "plan_path": str(tmp_path / "plan.json"),
            "output_dir": str(tmp_path / "particles"),
            "approval_paths": [str(p) for p in decisions],
        },
    )
    handle = jobs.submit_request(path, spec)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        row = jobs.job_status(path, handle["job_id"])
        if row["status"] == "running":
            break
        time.sleep(0.1)
    assert row["status"] == "running", row
    identity = row["worker"]
    jobs.cancel(path, handle["job_id"], "synthetic cancellation fault injection")
    final = wait(path, handle)
    assert final["status"] == "cancelled", final
    assert not jobs._alive(identity)
    assert jobs.campaign_status(path)["reserved_resources"]["threads"] == 0
