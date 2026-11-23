"""Local process observability, distinct from authentication and compute authority."""

from __future__ import annotations

import hashlib
import os
import socket
import sys
from pathlib import Path


class ExecutorContextError(ValueError):
    """The caller cannot safely interpret the executor's process identities."""


def current_context():
    """Identify the local observer context; these are not security attestations."""
    machine = socket.gethostname()
    source = "hostname"
    if sys.platform.startswith("linux"):
        try:
            machine = Path("/etc/machine-id").read_text().strip()
            source = "machine-id"
        except OSError:
            pass
        try:
            namespace = os.readlink("/proc/self/ns/pid")
            boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        except OSError as error:
            raise ExecutorContextError(
                "Linux process namespace/boot identity is not observable"
            ) from error
        if not machine or not namespace or not boot:
            raise ExecutorContextError("empty Linux executor identity")
    else:
        namespace = None
        boot = None
    return {
        "schema": "element_surrogate.executor_context.v1",
        "platform": sys.platform,
        "machine_id_sha256": hashlib.sha256(machine.encode()).hexdigest(),
        "machine_id_source": source,
        "pid_namespace": namespace,
        "boot_id": boot,
    }


def require_context(expected):
    actual = current_context()
    if expected != actual:
        raise ExecutorContextError(
            "scientific executor context differs: process IDs are not observable here. "
            "Use the configured persistent MCP service. No ledger mutation is permitted; "
            "an executor handoff requires its separate inspected operation."
        )
    return actual


def kernel_start_ticks(pid):
    """Linux /proc start counter, unaffected by wall-clock corrections."""
    text = Path(f"/proc/{pid}/stat").read_text()
    first, tail = text.split("(", 1)
    if int(first.strip()) != pid:
        raise ExecutorContextError("process stat identity differs")
    fields = tail.rsplit(")", 1)[1].split()
    return int(fields[19])  # field 22; fields here start with state (field 3)
