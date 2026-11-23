"""Manufactured diagnostic successors; no scientific payloads or live LLM clients."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from element_surrogate import jobs
from element_surrogate.benchmark.protocol import file_hash, read_json, write_json
from element_surrogate.benchmark.study import _guard
from element_surrogate.diagnostic_retry import inspect_retry, retry
from element_surrogate.project import initialize
from element_surrogate.runs import implementation_identity, sha256_json


def fixture(tmp_path, monkeypatch):
    initialize(tmp_path, "manufactured diagnostic retry")
    policy = {
        "schema": jobs.CAMPAIGN_SCHEMA,
        "limits": {
            "threads": 1,
            "working_memory_bytes": 1024,
            "scratch_bytes": 1024,
            "max_total_job_wall_seconds": 120,
            "max_calendar_seconds": 600,
        },
        "user_authority": {
            "actor": "fixture",
            "user_statement": "synthetic explicit budget",
            "source_reference": "manufactured test",
        },
    }
    write_json(tmp_path / "policy.json", policy)
    campaign = jobs.configure(tmp_path, tmp_path / "policy.json")["campaign_path"]
    source = tmp_path / "source.txt"
    source.write_text("manufactured immutable input")
    operation = SimpleNamespace(func=lambda path: None, input_file_args=("path",))
    monkeypatch.setattr(jobs, "_operation", lambda name: operation)

    def launch(path, identifier):
        _, ledger, _ = jobs._context(path)
        return jobs._handle(path, read_json(ledger)["jobs"][identifier])

    monkeypatch.setattr(jobs, "_launch", launch)
    request = {
        "schema": jobs.JOB_SCHEMA,
        "operation": "inspect_file",
        "arguments": {"path": str(source)},
        "resources": {"threads": 1, "working_memory_bytes": 1024, "scratch_bytes": 1024},
        "max_wall_seconds": 30,
        "time_policy": "active",
    }
    handle = jobs.submit_request(campaign, request)
    _, ledger_path, _ = jobs._context(campaign)
    ledger = read_json(ledger_path)
    row = ledger["jobs"][handle["job_id"]]
    row.update(status="timed_out", accounting_uncertain=True, charged_job_wall_seconds=32.0)
    folder = Path(row["folder"])
    spec = read_json(folder / "request.json")
    spec["implementation"] = {"version": "manufactured-predecessor"}
    spec["request_sha256"] = sha256_json(
        {k: spec[k] for k in ("request", "inputs", "implementation")}
    )
    row["request_sha256"] = spec["request_sha256"]
    write_json(folder / "request.json", spec)
    write_json(
        folder / "verified_inputs.json", [{"path": str(source), "sha256": file_hash(source)}]
    )
    write_json(ledger_path, ledger)
    inspection = tmp_path / "retry_inspection.json"
    result = inspect_retry(campaign, handle["job_id"], inspection)
    allocation = result["request_template"]
    allocation.update(
        max_wall_seconds=12,
        fresh_output_root="studies/fresh_retry",
        rationale="Explicit further synthetic allocation, old cost retained",
    )
    allocation_path = tmp_path / "retry_allocation.json"
    write_json(allocation_path, allocation)
    return campaign, handle["job_id"], ledger_path, inspection, allocation_path, source


def test_successor_retains_parent_cost_policy_and_inputs(tmp_path, monkeypatch):
    campaign, parent, ledger, inspection, allocation, source = fixture(tmp_path, monkeypatch)
    original_policy = Path(campaign).read_bytes()
    before = read_json(ledger)["jobs"][parent]
    child = retry(campaign, parent, inspection, allocation)
    record = read_json(ledger)
    after = record["jobs"][parent]
    assert after == {**before, "successor_job_id": child["job_id"]}
    successor = record["jobs"][child["job_id"]]
    assert successor["predecessor_job_id"] == parent
    assert successor["charged_job_wall_seconds"] == 0
    assert successor["max_wall_seconds"] == 12
    assert successor["status"] == "queued"
    assert Path(campaign).read_bytes() == original_policy
    spec = read_json(Path(child["job_directory"]) / "request.json")
    assert spec["request"]["expected_input_sha256"] == {str(source): file_hash(source)}
    assert spec["implementation"] == implementation_identity()
    assert retry(campaign, parent, inspection, allocation)["job_id"] == child["job_id"]
    assert len(read_json(ledger)["jobs"]) == 2


@pytest.mark.parametrize(
    "kind", ["changed_input", "wrong_implementation", "existing_output", "budget", "stale"]
)
def test_retry_refusals_preserve_ledger(tmp_path, monkeypatch, kind):
    campaign, parent, ledger, inspection, allocation, source = fixture(tmp_path, monkeypatch)
    value = read_json(allocation)
    if kind == "changed_input":
        source.write_text("changed")
    elif kind == "wrong_implementation":
        value["acknowledge_implementation_sha256"] = "0" * 64
    elif kind == "existing_output":
        (tmp_path / "studies/fresh_retry").mkdir(parents=True)
    elif kind == "budget":
        value["max_wall_seconds"] = 100
    else:
        record = read_json(ledger)
        record["jobs"][parent]["charged_job_wall_seconds"] += 1
        write_json(ledger, record)
    write_json(allocation, value)
    before = ledger.read_bytes()
    with pytest.raises(ValueError):
        retry(campaign, parent, inspection, allocation)
    assert ledger.read_bytes() == before
    if kind != "existing_output":
        assert not (tmp_path / "studies/fresh_retry").exists()


def test_guard_and_stateful_work_are_not_bypassed(tmp_path, monkeypatch):
    campaign, parent, ledger, _, _, _ = fixture(tmp_path, monkeypatch)
    folder = read_json(ledger)["jobs"][parent]["folder"]
    with _guard(folder), pytest.raises(ValueError, match="another process owns"):
        inspect_retry(campaign, parent, tmp_path / "held.json")
    value = read_json(ledger)
    value["jobs"][parent]["operation"] = "generate_training_data"
    write_json(ledger, value)
    with pytest.raises(ValueError, match="state/checkpoint"):
        inspect_retry(campaign, parent, tmp_path / "stateful.json")


def test_retry_cannot_fork_a_successor(tmp_path, monkeypatch):
    campaign, parent, ledger, inspection, allocation, _ = fixture(tmp_path, monkeypatch)
    retry(campaign, parent, inspection, allocation)
    changed = read_json(allocation)
    changed["fresh_output_root"] = "studies/second_fork"
    write_json(tmp_path / "fork.json", changed)
    before = ledger.read_bytes()
    with pytest.raises(ValueError, match="successor"):
        retry(campaign, parent, inspection, tmp_path / "fork.json")
    assert ledger.read_bytes() == before
    assert not (tmp_path / "studies/second_fork").exists()


@pytest.mark.parametrize(
    "kind", ["campaign_calendar", "job_calendar", "managed_output", "metadata"]
)
def test_retry_retains_calendar_and_request_identity(tmp_path, monkeypatch, kind):
    campaign, parent, ledger, inspection, allocation, _ = fixture(tmp_path, monkeypatch)
    record = read_json(ledger)
    if kind == "campaign_calendar":
        monkeypatch.setattr(jobs.time, "time", lambda: read_json(campaign)["deadline_unix"] + 1)
    elif kind == "job_calendar":
        # A separately inspected calendar-bounded predecessor, not a reset timer.
        record["jobs"][parent]["deadline_unix"] = 1
        write_json(ledger, record)
        inspect_retry(campaign, parent, tmp_path / "calendar_inspection.json")
        inspection = tmp_path / "calendar_inspection.json"
    elif kind == "managed_output":
        value = read_json(allocation)
        value["fresh_output_root"] = str(Path(record["jobs"][parent]["folder"]).parent / "new")
        write_json(allocation, value)
    else:
        record["jobs"][parent]["resources"]["threads"] = 2
        write_json(ledger, record)
        with pytest.raises(ValueError, match="execution metadata"):
            inspect_retry(campaign, parent, tmp_path / "tampered_inspection.json")
        return
    before = ledger.read_bytes()
    with pytest.raises(ValueError):
        retry(campaign, parent, inspection, allocation)
    assert ledger.read_bytes() == before
