"""Manufactured executor visibility and at-most-once charge controls."""

import os
from copy import deepcopy

import pytest

from element_surrogate import execution_context, jobs
from element_surrogate.benchmark.protocol import file_hash, read_json, write_json
from element_surrogate.benchmark.study import _guard
from element_surrogate.executor_binding import bind_executor
from element_surrogate.project import initialize


def test_identity_uses_current_observer_context():
    identity = jobs._identity(os.getpid())
    assert identity["executor_context"] == execution_context.current_context()
    assert jobs._alive(identity)
    if identity["executor_context"]["platform"].startswith("linux"):
        assert identity["kernel_start_ticks"] > 0


def test_foreign_namespace_is_unobservable_not_dead(monkeypatch):
    actual = execution_context.current_context()
    foreign = {**actual, "pid_namespace": "manufactured foreign namespace"}
    identity = {"pid": os.getpid(), "created_unix": 0, "executor_context": foreign}
    with pytest.raises(execution_context.ExecutorContextError, match="persistent MCP"):
        jobs._alive(identity)


def test_linux_counter_ignores_wall_creation_time_and_detects_reused_pid(monkeypatch):
    identity = jobs._identity(os.getpid())
    if "kernel_start_ticks" not in identity:
        pytest.skip("Linux kernel-counter control")
    identity["created_unix"] -= 60
    assert jobs._alive(identity)
    identity["kernel_start_ticks"] -= 1
    assert not jobs._alive(identity)


def attempt():
    return {
        "status": "running",
        "attempts": [],
        "charged_job_wall_seconds": 3.0,
        "max_wall_seconds": 30.0,
        "active_attempt_id": "manufactured-attempt",
        "active_attempt_charge_kind": None,
        "attempt_started_unix": 1.0,
    }


def test_observed_finalization_is_at_most_once():
    row = attempt()
    jobs._finalize_attempt(row, "complete", 2.0)
    before = deepcopy(row)
    jobs._finalize_attempt(row, "complete", 2.0)
    assert row == before
    assert row["charged_job_wall_seconds"] == 5.0
    assert len(row["attempts"]) == 1


def test_conservative_charge_is_not_billed_again_or_turned_into_pass():
    row = attempt()
    row.update(charged_job_wall_seconds=30.0, active_attempt_charge_kind="conservative")
    assert jobs._cost(row) == 30.0
    jobs._finalize_attempt(row, "complete", 2.0)
    assert row["charged_job_wall_seconds"] == 30.0
    assert row["status"] == "interrupted"
    assert row["attempts"][0]["job_wall_seconds"] == 2.0
    assert row["attempts"][0]["charge_kind"] == "previously_charged_conservatively"


def legacy_fixture(tmp_path):
    initialize(tmp_path, "manufactured legacy executor")
    policy = {
        "schema": jobs.CAMPAIGN_SCHEMA,
        "limits": {
            "threads": 1,
            "working_memory_bytes": 1024,
            "scratch_bytes": 1024,
            "max_total_job_wall_seconds": 120,
        },
        "user_authority": {
            "actor": "fixture",
            "user_statement": "synthetic budget",
            "source_reference": "manufactured unit control",
        },
    }
    request = tmp_path / "policy.json"
    write_json(request, policy)
    campaign = jobs.configure(tmp_path, request)["campaign_path"]
    _, ledger_path, directory = jobs._context(campaign)
    directory.mkdir()
    folder = directory / ("1" * 32)
    folder.mkdir()
    live = jobs._identity(os.getpid())
    owner = {k: live[k] for k in ("pid", "created_unix")}
    ledger = {
        "jobs": {
            "1" * 32: {
                "id": "1" * 32,
                "folder": str(folder),
                "status": "timed_out",
                "launch_owner": owner,
                "worker": {"pid": 99999999, "created_unix": 0},
                "charged_job_wall_seconds": 32.0,
                "max_wall_seconds": 30.0,
                "accounting_uncertain": True,
                "attempts": [],
            }
        }
    }
    write_json(ledger_path, ledger)
    binding = tmp_path / "binding.json"
    value = {
        "schema": "element_surrogate.executor_binding.v1",
        "expected_ledger_sha256": file_hash(ledger_path),
        "legacy_launch_owner": owner,
        "rationale": "Manufactured context adoption",
    }
    write_json(binding, value)
    return campaign, ledger_path, binding, value, folder


def test_legacy_access_requires_inspected_adoption(tmp_path):
    campaign, ledger, binding, _, _ = legacy_fixture(tmp_path)
    before = ledger.read_bytes()
    with pytest.raises(execution_context.ExecutorContextError, match="legacy campaign"):
        jobs.campaign_status(campaign)
    assert ledger.read_bytes() == before
    policy_before = open(campaign, "rb").read()
    result = bind_executor(campaign, binding)
    assert result["prior_charges_retained"]
    row = read_json(ledger)["jobs"]["1" * 32]
    assert row["charged_job_wall_seconds"] == 32.0
    assert row["status"] == "timed_out" and row["accounting_uncertain"]
    assert row["worker"]["executor_context"] == execution_context.current_context()
    assert open(campaign, "rb").read() == policy_before
    after = ledger.read_bytes()
    assert bind_executor(campaign, binding)["already_applied"]
    assert ledger.read_bytes() == after


def test_legacy_adoption_rejects_changed_state_or_unobserved_owner(tmp_path):
    campaign, ledger, binding, request, _ = legacy_fixture(tmp_path)
    request["legacy_launch_owner"]["created_unix"] -= 1
    write_json(binding, request)
    before = ledger.read_bytes()
    with pytest.raises(execution_context.ExecutorContextError, match="not observably alive"):
        bind_executor(campaign, binding)
    assert ledger.read_bytes() == before
    request["expected_ledger_sha256"] = "0" * 64
    write_json(binding, request)
    with pytest.raises(ValueError, match="stale"):
        bind_executor(campaign, binding)
    assert ledger.read_bytes() == before


def test_legacy_adoption_refuses_pending_job_and_held_guard(tmp_path):
    campaign, ledger, binding, request, folder = legacy_fixture(tmp_path)
    with _guard(folder), pytest.raises(ValueError, match="another process owns"):
        bind_executor(campaign, binding)
    value = read_json(ledger)
    value["jobs"]["1" * 32]["status"] = "queued"
    write_json(ledger, value)
    request["expected_ledger_sha256"] = file_hash(ledger)
    write_json(binding, request)
    before = ledger.read_bytes()
    with pytest.raises(execution_context.ExecutorContextError, match="pending jobs"):
        bind_executor(campaign, binding)
    assert ledger.read_bytes() == before


def test_quiescent_handoff_does_not_treat_old_pids_as_local(tmp_path):
    campaign, ledger, binding, request, _ = legacy_fixture(tmp_path)
    value = read_json(ledger)
    current = execution_context.current_context()
    value["executor_context"] = {**current, "pid_namespace": "other observable executor"}
    row = value["jobs"]["1" * 32]
    row.update(status="complete", accounting_uncertain=False)
    write_json(ledger, value)
    request.pop("legacy_launch_owner")
    request["expected_ledger_sha256"] = file_hash(ledger)
    write_json(binding, request)
    bind_executor(campaign, binding)
    after = read_json(ledger)["jobs"]["1" * 32]
    assert after["charged_job_wall_seconds"] == 32
    assert not jobs._alive(after["worker"])
    assert not jobs._alive(after["launch_owner"])


def test_uncertain_campaign_cannot_move_to_another_live_namespace(tmp_path):
    campaign, ledger, binding, request, _ = legacy_fixture(tmp_path)
    value = read_json(ledger)
    value["executor_context"] = {**execution_context.current_context(), "pid_namespace": "other"}
    write_json(ledger, value)
    request.pop("legacy_launch_owner")
    request["expected_ledger_sha256"] = file_hash(ledger)
    write_json(binding, request)
    before = ledger.read_bytes()
    with pytest.raises(execution_context.ExecutorContextError, match="uncertain ownership"):
        bind_executor(campaign, binding)
    assert ledger.read_bytes() == before


def test_proven_local_reboot_retains_allowance_conservatively(tmp_path, monkeypatch):
    campaign, ledger, binding, request, _ = legacy_fixture(tmp_path)
    current = {
        **execution_context.current_context(),
        "machine_id_source": "machine-id",
        "boot_id": "current manufactured boot",
    }
    monkeypatch.setattr("element_surrogate.executor_binding.current_context", lambda: current)
    value = read_json(ledger)
    value["executor_context"] = {**current, "boot_id": "previous manufactured boot"}
    value["jobs"]["1" * 32].update(status="running", charged_job_wall_seconds=3)
    write_json(ledger, value)
    request.pop("legacy_launch_owner")
    request["expected_ledger_sha256"] = file_hash(ledger)
    write_json(binding, request)
    before_policy = open(campaign, "rb").read()
    bind_executor(campaign, binding)
    row = read_json(ledger)["jobs"]["1" * 32]
    assert row["status"] == "interrupted" and row["accounting_uncertain"]
    assert row["charged_job_wall_seconds"] == 30
    assert open(campaign, "rb").read() == before_policy
