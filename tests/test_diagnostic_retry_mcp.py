"""Real stdio/MCP adoption and successor execution on manufactured evidence."""

import os
import sys
import time
from pathlib import Path

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from element_surrogate import jobs
from element_surrogate.benchmark.protocol import file_hash, read_json, write_json
from element_surrogate.mcp_server import PROJECT_ENV
from element_surrogate.project import initialize


async def round_trip(root, campaign, parent, binding, source):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "element_surrogate.mcp_server"],
        env={**os.environ, PROJECT_ENV: str(root)},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            bound = await session.call_tool(
                "bind_scientific_executor",
                {"campaign_path": str(campaign), "request_path": str(binding)},
            )
            assert not bound.is_error, bound.content
            assert bound.structured_content["prior_charges_retained"]
            inspection = root / "inspection.json"
            inspected = await session.call_tool(
                "inspect_scientific_diagnostic_retry",
                {"campaign_path": str(campaign), "job_id": parent, "output_path": str(inspection)},
            )
            assert not inspected.is_error, inspected.content
            allocation = inspected.structured_content["request_template"]
            allocation.update(
                max_wall_seconds=30,
                fresh_output_root="studies/retry",
                rationale="Manufactured explicitly allocated retry; retain predecessor cost",
            )
            write_json(root / "allocation.json", allocation)
            arguments = {
                "campaign_path": str(campaign),
                "job_id": parent,
                "inspection_path": str(inspection),
                "request_path": str(root / "allocation.json"),
            }
            successor = await session.call_tool("retry_scientific_diagnostic", arguments)
            assert not successor.is_error, successor.content
            child = successor.structured_content["job_id"]
            repeated = await session.call_tool("retry_scientific_diagnostic", arguments)
            assert not repeated.is_error and repeated.structured_content["job_id"] == child
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                result = await session.call_tool(
                    "scientific_job_status", {"campaign_path": str(campaign), "job_id": child}
                )
                assert not result.is_error, result.content
                row = result.structured_content
                if row["is_terminal"] and not jobs._alive(row.get("supervisor")):
                    break
                await anyio.sleep(0.1)
            else:
                raise AssertionError("manufactured diagnostic did not terminate")
            assert row["status"] == "complete", row
            assert read_json(row["result_path"])["size_bytes"] == source.stat().st_size
            return child


def test_real_mcp_adoption_and_new_diagnostic_execution(tmp_path, monkeypatch):
    initialize(tmp_path, "manufactured MCP recovery control")
    policy = {
        "schema": jobs.CAMPAIGN_SCHEMA,
        "limits": {
            "threads": 1,
            "working_memory_bytes": 256 << 20,
            "scratch_bytes": 1024,
            "max_total_job_wall_seconds": 120,
            "max_calendar_seconds": 600,
        },
        "user_authority": {
            "actor": "fixture",
            "user_statement": "synthetic explicit budget",
            "source_reference": "manufactured MCP integration control",
        },
    }
    write_json(tmp_path / "policy.json", policy)
    campaign = Path(jobs.configure(tmp_path, tmp_path / "policy.json")["campaign_path"])
    source = tmp_path / "source.txt"
    source.write_text("immutable manufactured bytes")
    request = {
        "schema": jobs.JOB_SCHEMA,
        "operation": "inspect_file",
        "arguments": {"path": str(source)},
        "resources": {"threads": 1, "working_memory_bytes": 256 << 20, "scratch_bytes": 1024},
        "max_wall_seconds": 30,
        "time_policy": "active",
    }
    with monkeypatch.context() as patch:
        patch.setattr(jobs, "_launch", lambda path, identifier: {"job_id": identifier})
        parent = jobs.submit_request(campaign, request)["job_id"]
    _, ledger_path, _ = jobs._context(campaign)
    ledger = read_json(ledger_path)
    owner = jobs._identity(os.getpid())
    owner = {k: owner[k] for k in ("pid", "created_unix")}
    row = ledger["jobs"][parent]
    row.update(
        status="timed_out",
        accounting_uncertain=True,
        charged_job_wall_seconds=32,
        launch_owner=owner,
    )
    write_json(
        Path(row["folder"]) / "verified_inputs.json",
        [{"path": str(source), "sha256": file_hash(source)}],
    )
    ledger.pop("executor_context")
    write_json(ledger_path, ledger)
    binding = tmp_path / "binding.json"
    write_json(
        binding,
        {
            "schema": "element_surrogate.executor_binding.v1",
            "expected_ledger_sha256": file_hash(ledger_path),
            "legacy_launch_owner": owner,
            "rationale": "Manufactured same-observer legacy adoption",
        },
    )
    policy_before = campaign.read_bytes()
    child = anyio.run(round_trip, tmp_path, campaign, parent, binding, source)
    after = read_json(ledger_path)
    assert set(after["jobs"]) == {parent, child}
    assert after["jobs"][parent]["charged_job_wall_seconds"] == 32
    assert after["jobs"][parent]["status"] == "timed_out"
    assert after["jobs"][parent]["successor_job_id"] == child
    assert after["jobs"][child]["charged_job_wall_seconds"] > 0
    assert sum(row["charged_job_wall_seconds"] for row in after["jobs"].values()) < 120
    assert campaign.read_bytes() == policy_before
