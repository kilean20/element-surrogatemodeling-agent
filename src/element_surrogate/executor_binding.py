"""Inspected executor binding without changing a campaign's authority or limits."""

from __future__ import annotations

import os
import time
from contextlib import ExitStack

from element_surrogate import jobs
from element_surrogate.benchmark.protocol import file_hash, read_json
from element_surrogate.benchmark.study import _guard
from element_surrogate.execution_context import ExecutorContextError, current_context
from element_surrogate.file_io import artifact_guard
from element_surrogate.runs import sha256_json

SCHEMA = "element_surrogate.executor_binding.v1"


def _same_owner(left, right):
    return left is not None and all(left.get(k) == right.get(k) for k in ("pid", "created_unix"))


def _local_reboot(previous, current):
    return bool(
        previous
        and previous.get("machine_id_source") == current.get("machine_id_source") == "machine-id"
        and previous["machine_id_sha256"] == current["machine_id_sha256"]
        and previous.get("boot_id")
        and previous["boot_id"] != current.get("boot_id")
    )


def bind_executor(campaign_path, request_path):
    """Adopt a proven legacy origin or hand off an inactive local campaign.

    Request rationale is caller-attested, not authenticated human approval.
    A local reboot is distinguished from an unobservable live PID namespace.
    This is a cooperating local-executor protocol, not host authentication.
    """
    request = read_json(request_path)
    if request.get("schema") != SCHEMA or set(request) - {
        "schema",
        "expected_ledger_sha256",
        "rationale",
        "legacy_launch_owner",
    }:
        raise ValueError("executor_binding.v1 request required")
    if not isinstance(request.get("rationale"), str) or not request["rationale"].strip():
        raise ValueError("executor binding rationale required")
    campaign, ledger_path, _ = jobs._context(campaign_path, check_executor=False)
    current = current_context()
    request_sha = sha256_json(request)
    with artifact_guard(ledger_path.with_name("campaign_mutation"), create=True):
        ledger = read_json(ledger_path)
        previous = ledger.get("executor_context")
        history = ledger.get("executor_binding_history", [])
        if history and history[-1]["request_sha256"] == request_sha and previous == current:
            return {"executor_context": current, "already_applied": True}
        if file_hash(ledger_path) != request.get("expected_ledger_sha256"):
            raise ValueError("stale executor binding inspection; ledger changed")
        if previous == current:
            return {"executor_context": current, "already_current": True}
        charges = {k: row["charged_job_wall_seconds"] for k, row in ledger["jobs"].items()}
        pending = [row for row in ledger["jobs"].values() if row["status"] in jobs.PENDING]
        reboot = _local_reboot(previous, current)
        if pending and not reboot:
            raise ExecutorContextError(
                "executor has pending jobs; reconnect to its persistent service"
            )
        if (
            previous is not None
            and not reboot
            and any(row.get("accounting_uncertain") for row in ledger["jobs"].values())
        ):
            raise ExecutorContextError(
                "uncertain ownership cannot be handed to another live context"
            )
        owner = request.get("legacy_launch_owner")
        if previous is None:
            if not isinstance(owner, dict) or set(owner) != {"pid", "created_unix"}:
                raise ValueError("legacy adoption requires its live recorded launch owner")
            if type(owner["pid"]) is not int or not jobs._alive(owner):
                raise ExecutorContextError("legacy launch owner is not observably alive")
            if current["pid_namespace"] is not None:
                if os.readlink(f"/proc/{owner['pid']}/ns/pid") != current["pid_namespace"]:
                    raise ExecutorContextError("legacy launch owner belongs to another namespace")
            if not any(
                _same_owner(row.get("launch_owner"), owner) for row in ledger["jobs"].values()
            ):
                raise ValueError("live owner does not match any recorded legacy job launcher")
        with ExitStack() as stack:
            for row in ledger["jobs"].values():
                stack.enter_context(_guard(row["folder"]))
            if reboot:
                for row in pending:
                    if row["status"] in jobs.ACTIVE:
                        row.update(
                            charged_job_wall_seconds=max(
                                row["charged_job_wall_seconds"], row["max_wall_seconds"]
                            ),
                            accounting_uncertain=True,
                            active_attempt_charge_kind="conservative",
                        )
                    row.update(status="interrupted", error="previous local boot ended")
            for row in ledger["jobs"].values():
                proven_legacy = previous is None and _same_owner(row.get("launch_owner"), owner)
                for key in ("launch_owner", "supervisor", "worker"):
                    identity = row.get(key)
                    if not identity:
                        continue
                    if previous is None:
                        if proven_legacy:
                            identity["executor_context"] = current
                        else:
                            identity["context_unverified"] = True
                    else:
                        identity["inactive_verified_in_context"] = current
                if previous is None and not proven_legacy:
                    row["legacy_executor_unverified"] = True
            history.append(
                {
                    "request_sha256": request_sha,
                    "rationale": request["rationale"],
                    "previous_context": previous,
                    "new_context": current,
                    "legacy_launch_owner": owner,
                    "local_reboot": reboot,
                    "unix": time.time(),
                    "authority_authenticated": False,
                }
            )
            ledger.update(executor_context=current, executor_binding_history=history)
            assert all(
                ledger["jobs"][k]["charged_job_wall_seconds"] >= v for k, v in charges.items()
            )
            jobs._save_ledger(ledger_path, ledger)
    return {
        "executor_context": current,
        "binding_request_sha256": request_sha,
        "policy_sha256": campaign["policy_sha256"],
        "deadline_unix": campaign["deadline_unix"],
        "prior_charges_retained": True,
        "ledger_path": str(ledger_path),
    }
