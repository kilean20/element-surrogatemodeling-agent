"""End-to-end test: start the real server over stdio, as an agent client does, and call it."""

import os
import sys

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from element_surrogate.mcp_server import PROJECT_ENV
from element_surrogate.runs import RunStore


async def _session_calls(project, field):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "element_surrogate.mcp_server"],
        env={**os.environ, PROJECT_ENV: str(project)},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            summary = await session.call_tool("inspect_file", {"path": str(field)})
            replayed = await session.call_tool(
                "replay_run", {"run_id": summary.structured_content["run_id"]}
            )
            missing = await session.call_tool("inspect_file", {"path": str(field) + ".absent"})
            return tools, summary, replayed, missing


def test_stdio_round_trip_writes_a_replayable_record(tmp_path):
    project = tmp_path / "project"
    field = tmp_path / "field.txt"
    field.write_text("0 0 0 0.0 0.0 1.0\n", encoding="utf-8")

    tools, summary, replayed, missing = anyio.run(_session_calls, project, field)

    assert {"inspect_file", "list_runs", "replay_run"} <= {tool.name for tool in tools.tools}
    assert not summary.is_error
    assert summary.structured_content["size_bytes"] == field.stat().st_size
    assert replayed.structured_content["reproduced"] is True
    store = RunStore(project)
    records = [store.load(identifier) for identifier in store.list_runs()]
    assert len(records) == 2
    assert records[0]["run_id"] == summary.structured_content["run_id"]
    assert records[1]["status"] == "error" and records[1]["error_stage"] == "input_validation"
    # An input error reaches the client as a tool error that carries its message.
    assert missing.is_error
    assert ".absent" in " ".join(getattr(c, "text", "") for c in missing.content)
