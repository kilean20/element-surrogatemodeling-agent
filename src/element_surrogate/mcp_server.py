"""MCP server: serves the tool registry to agent clients and records every tool call.

Every registered tool (``element_surrogate.tools.REGISTRY``) becomes an MCP tool with the
function's signature as its schema and its docstring as its description, so a tool is
defined once. Run it over stdio with ``element-surrogate-mcp``. Run records go to the
project folder named by the ``ELEMENT_SURROGATE_PROJECT`` environment variable, or to the
current directory when it is unset.
"""

from __future__ import annotations

import functools
import inspect
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from element_surrogate import __version__
from element_surrogate.runs import RunStore, Tool, record_call, replay
from element_surrogate.tools import REGISTRY

PROJECT_ENV = "ELEMENT_SURROGATE_PROJECT"

_READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True)


def project_dir() -> Path:
    return Path(os.environ.get(PROJECT_ENV) or Path.cwd())


def _recorded_handler(registered: Tool, runs: RunStore) -> Callable[..., dict[str, Any]]:
    """A handler with the tool's own signature that records the call. Input errors (a missing
    file, a slot that must be chosen) go back to the client as tool errors with their message."""

    @functools.wraps(registered.func)
    def handler(**arguments: Any) -> dict[str, Any]:
        try:
            return record_call(runs, registered, arguments)
        except (ValueError, OSError) as exc:
            raise ToolError(str(exc)) from exc

    return handler


def build_server(store: RunStore | None = None) -> MCPServer:
    """Build the server. Tests pass their own store; the command line uses the project folder."""
    runs = store if store is not None else RunStore(project_dir())
    server = MCPServer(
        name="element-surrogate",
        version=__version__,
        instructions=(
            "Tools for building a neural-network surrogate of one magnet's transfer map from "
            "its 3D field map. Calls are recorded; generic replay accepts read-only tools. "
            "Mutating workflows use explicit status/check/resume paths. "
            "Initialize or restore the explicit private project before changing its plan; "
            "keep client-attested decisions distinct from scientific gate approvals. "
            "On a new field file, run check_storage_order first. "
            "Pinned Genesis guidance and validators are packaged; use read_genesis_guidance. "
            "Exact zeros or invariant planes are contextual observations, "
            "not proof of corrupt fields. "
            "Unavailable physical facts do not automatically stop numerical research. "
            "Use available evidence, existing authority and a reasoned scoped decision; "
            "the optional uncertainty-review tool overrides no scientific check or approval."
        ),
    )
    for registered in REGISTRY.values():
        server.add_tool(
            _recorded_handler(registered, runs),
            name=registered.name,
            description=inspect.getdoc(registered.func),
            annotations=ToolAnnotations(
                read_only_hint=registered.read_only,
                destructive_hint=False,
                idempotent_hint=registered.read_only,
            ),
        )

    @server.tool(annotations=_READ_ONLY)
    def list_runs() -> dict[str, Any]:
        """List the ids of the run records in this project, oldest first."""
        return {"run_ids": runs.list_runs()}

    @server.tool(annotations=_READ_ONLY)
    def replay_run(run_id: str) -> dict[str, Any]:
        """Re-execute a recorded read-only call and check its result identity.
        Changed inputs are reported. Mutating calls are refused; use their
        explicit status/check/resume paths instead of launching duplicate work."""
        try:
            return replay(runs, REGISTRY, run_id).as_dict()
        except (KeyError, ValueError, OSError) as exc:
            raise ToolError(f"cannot replay run {run_id}: {exc}") from exc

    return server


def main() -> None:
    build_server().run("stdio")


if __name__ == "__main__":
    main()
