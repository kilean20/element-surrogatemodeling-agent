"""A real Linux PID namespace must not mutate another executor's campaign."""

import json
import os
import shutil
import subprocess
import sys
import time

import pytest

from element_surrogate import jobs
from element_surrogate.benchmark.protocol import read_json, write_json
from element_surrogate.project import initialize


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux PID namespace")
def test_separate_pid_namespace_cannot_refresh_or_submit(tmp_path, monkeypatch):
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        pytest.skip("bwrap is not installed")
    prefix = [bwrap, "--unshare-pid", "--dev-bind", "/", "/", "--proc", "/proc", "--"]
    probe = subprocess.run(
        [*prefix, sys.executable, "-c", "print('namespace ready')"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if probe.returncode:
        pytest.skip("unprivileged PID namespace unavailable: " + probe.stderr[:180])
    initialize(tmp_path, "manufactured namespace control")
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
            "source_reference": "manufactured process visibility control",
        },
    }
    write_json(tmp_path / "policy.json", policy)
    campaign = jobs.configure(tmp_path, tmp_path / "policy.json")["campaign_path"]
    monkeypatch.setattr(jobs, "_launch", lambda path, identifier: {"job_id": identifier})
    request = {
        "schema": jobs.JOB_SCHEMA,
        "operation": "inspect_file",
        "arguments": {"path": str(tmp_path / "policy.json")},
        "resources": {"threads": 1, "working_memory_bytes": 1024, "scratch_bytes": 1024},
        "max_wall_seconds": 30,
        "time_policy": "active",
    }
    handle = jobs.submit_request(campaign, request)
    _, ledger_path, _ = jobs._context(campaign)
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        ledger = read_json(ledger_path)
        ledger["jobs"][handle["job_id"]].update(
            status="running",
            supervisor=jobs._identity(os.getpid()),
            worker=jobs._identity(worker.pid),
            attempt_started_monotonic=time.monotonic(),
            attempt_started_unix=time.time(),
            attempt_boot_id=ledger["executor_context"]["boot_id"],
        )
        write_json(ledger_path, ledger)
        before = ledger_path.read_bytes()
        folders = sorted(p.name for p in ledger_path.parent.rglob("request.json"))
        code = """
import json, sys
from element_surrogate import jobs
from element_surrogate.execution_context import ExecutorContextError, current_context
campaign, request = sys.argv[1], json.loads(sys.argv[2])
for name, action in [('status', lambda: jobs.campaign_status(campaign)),
                     ('submit', lambda: jobs.submit_request(campaign, request))]:
    try:
        action()
    except ExecutorContextError as error:
        assert 'persistent MCP' in str(error)
        print(json.dumps({'operation': name, 'refused': True, 'context': current_context()}))
    else:
        raise AssertionError(name + ' accepted a foreign namespace')
"""
        result = subprocess.run(
            [*prefix, sys.executable, "-c", code, str(campaign), json.dumps(request)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        receipts = [json.loads(line) for line in result.stdout.splitlines()]
        assert len(receipts) == 2 and all(row["refused"] for row in receipts)
        assert all(
            row["context"]["pid_namespace"] != ledger["executor_context"]["pid_namespace"]
            for row in receipts
        )
        assert ledger_path.read_bytes() == before
        assert sorted(p.name for p in ledger_path.parent.rglob("request.json")) == folders
        assert worker.poll() is None
    finally:
        worker.terminate()
        worker.wait(timeout=10)
