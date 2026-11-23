"""Persistent scientific jobs with shared admission and explicit time semantics.

This controls cooperating jobs, not arbitrary user shell code. Reservations name
numerical working arrays/scratch, not fixed process RSS. Process identities and
owned locks prevent releasing capacity merely because an RPC/client vanished.
"""

from __future__ import annotations

import inspect
import math
import os
import re
import subprocess
import sys
import time
import traceback
import uuid
from copy import deepcopy
from pathlib import Path

import psutil

from element_surrogate.benchmark.protocol import file_hash, read_json, write_json
from element_surrogate.execution_context import (
    ExecutorContextError,
    current_context,
    kernel_start_ticks,
    require_context,
)
from element_surrogate.file_io import artifact_guard
from element_surrogate.project_paths import LEGACY_STATE_DIRNAME, STATE_DIRNAME, state_directory
from element_surrogate.runs import RunStore, implementation_identity, record_call, sha256_json

CAMPAIGN_SCHEMA = "element_surrogate.scientific_campaign.v1"
JOB_SCHEMA = "element_surrogate.scientific_job.v1"
ACTIVE = {"starting", "running", "stopping"}
TERMINAL = {
    "complete",
    "incomplete",
    "failed",
    "timed_out",
    "cancelled",
    "refused_budget",
    "interrupted",
}
PENDING = ACTIVE | {"queued"}
TOOL_OPERATIONS = {
    "inspect_file",
    "read_field_file",
    "check_storage_order",
    "check_maxwell",
    "check_harmonic_nodes",
    "scan_node_defects",
    "check_field_ends",
    "analyze_harmonics",
    "check_mirror_symmetry",
    "analyze_field_frame",
    "collect_field_readiness",
    "qualify_element_transport",
    "trace_reference_orbit",
    "fit_reference_orbit",
    "prepare_mapped_field_store",
    "prepare_genesis_release_cards",
    "introspect_card_dataset",
    "analyze_generalized_gradients",
    "compare_field_exports",
    "check_tracking_convergence",
    "check_adaptive_tracking_convergence",
    "validate_training_data_plan",
    "assess_field_readiness",
    "assess_map_symmetry",
    "assess_native_theory_base",
    "analyze_transverse_symplecticity",
    "assess_canonical_shear_coordinates",
    "prepare_training_benchmark_cohorts",
    "prepare_neural_surrogate_study",
    "prepare_supplied_neural_surrogate_study",
    "review_model_physics",
    "assess_supplied_label_reference",
    "prepare_supplied_genesis_release_cards",
    "evaluate_benchmark_test",
    "export_native_surrogate",
    "export_public_surrogate_interface",
    "prepare_genesis_dataset_draft",
    "build_training_data_request",
    "summarize_generation_capacity",
    "compare_with_description",
    "propose_training_coordinate_scales",
    "check_track_domain_isolation",
    "build_benchmark_report",
    "build_benchmark_tradeoff_report",
    "prepare_benchmark_metric_crosscheck",
}
INTERNAL_OPERATIONS = {"generate_training_data", "run_benchmark_study"}
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _positive(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"positive finite {name} required")
    return value


def _resources(value):
    result = {}
    for key, minimum in (("threads", 1), ("working_memory_bytes", 0), ("scratch_bytes", 0)):
        if type(value.get(key)) is not int or value[key] < minimum:
            raise ValueError(f"integer resource {key} >= {minimum} required")
        result[key] = value[key]
    return result


def configure(project_dir, request_path):
    from element_surrogate.project import state_paths

    root, project = state_paths(project_dir)
    policy = read_json(request_path)
    if policy.get("schema") != CAMPAIGN_SCHEMA:
        raise ValueError("scientific campaign schema required")
    limits = _resources(policy["limits"])
    limits["max_total_job_wall_seconds"] = _positive(
        policy["limits"].get("max_total_job_wall_seconds"), "total job-wall budget"
    )
    calendar = policy["limits"].get("max_calendar_seconds")
    if calendar is not None:
        _positive(calendar, "calendar budget")
    authority = policy.get("user_authority", {})
    if not all(
        isinstance(authority.get(k), str) and authority[k].strip()
        for k in ("actor", "user_statement", "source_reference")
    ):
        raise ValueError("record the supplied user/delegation authority; no tool creates it")
    target = state_directory(root) / "campaign.json"
    identity = sha256_json(policy)
    with artifact_guard(target.with_name("campaign_configuration"), create=True):
        if target.exists():
            prior = read_json(target)
            if prior["policy_sha256"] != identity:
                raise ValueError("campaign policy is immutable; do not silently increase a budget")
        else:
            prior = {
                "schema": CAMPAIGN_SCHEMA,
                "project_id": project["id"],
                "project_dir": str(root),
                "policy": policy,
                "policy_sha256": identity,
                "created_unix": time.time(),
                "deadline_unix": None if calendar is None else time.time() + calendar,
            }
            write_json(target, prior, exclusive=True)
    ledger = target.with_name("campaign_ledger.json")
    with artifact_guard(ledger.with_name("campaign_mutation"), create=True):
        if not ledger.exists():
            write_json(ledger, {"jobs": {}, "executor_context": current_context()}, exclusive=True)
        else:
            _require_ledger_context(read_json(ledger))
    return {
        "campaign_path": str(target),
        "policy_sha256": identity,
        "limits": limits,
        "authority_authenticated": False,
        "scope": "managed jobs only; not arbitrary shell code",
    }


def find_campaign(directory):
    path = Path(directory).resolve()
    if path.is_file():
        path = path.parent
    for root in (path, *path.parents):
        # Ancestors of a study/input file are not necessarily project roots.
        # An unrelated file named records must not shadow the real campaign.
        if not any(
            (root / name / "campaign.json").is_file()
            for name in (STATE_DIRNAME, LEGACY_STATE_DIRNAME)
        ):
            continue
        candidate = state_directory(root) / "campaign.json"
        if candidate.is_file():
            return str(candidate)
    return None


def _context(campaign_path, *, check_executor=True):
    path = Path(campaign_path).resolve()
    campaign = read_json(path)
    if (
        campaign.get("schema") != CAMPAIGN_SCHEMA
        or sha256_json(campaign["policy"]) != campaign["policy_sha256"]
    ):
        raise ValueError("campaign policy identity changed")
    if path != state_directory(campaign["project_dir"]) / "campaign.json":
        raise ValueError("campaign location differs from project")
    ledger = path.with_name("campaign_ledger.json")
    if check_executor:
        _require_ledger_context(read_json(ledger))
    return campaign, ledger, path.parent / "jobs"


def _require_ledger_context(ledger):
    expected = ledger.get("executor_context")
    if expected is None:
        raise ExecutorContextError(
            "legacy campaign has no verified executor context; use inspected legacy adoption "
            "before any process probe or state mutation"
        )
    return require_context(expected)


def _identity(pid):
    process = psutil.Process(pid)
    context = current_context()
    identity = {"pid": pid, "created_unix": process.create_time(), "executor_context": context}
    if context["platform"].startswith("linux"):
        try:
            identity["kernel_start_ticks"] = kernel_start_ticks(pid)
        except FileNotFoundError as error:
            raise psutil.NoSuchProcess(pid) from error
    return identity


def _alive(identity):
    if not identity:
        return False
    if identity.get("context_unverified"):
        raise ExecutorContextError("legacy process identity has an unverified observer context")
    inactive = identity.get("inactive_verified_in_context")
    if inactive is not None:
        require_context(inactive)
        return False
    expected = identity.get("executor_context")
    if expected is not None:
        actual = current_context()
        if (
            expected.get("boot_id") is not None
            and expected["boot_id"] != actual.get("boot_id")
            and expected["machine_id_sha256"] == actual["machine_id_sha256"]
            and expected.get("machine_id_source") == actual.get("machine_id_source") == "machine-id"
        ):
            return False
        require_context(expected)
    try:
        process = psutil.Process(identity["pid"])
        same = (
            kernel_start_ticks(identity["pid"]) == identity["kernel_start_ticks"]
            if "kernel_start_ticks" in identity
            else process.create_time() == identity["created_unix"]
        )
        return same and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, FileNotFoundError, ProcessLookupError):
        return False
    except (psutil.AccessDenied, PermissionError) as error:
        raise ExecutorContextError(
            "owned process is not observable; reservation retained"
        ) from error


def _job_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{32}", value):
        raise ValueError("job id from the returned handle required")
    return value


def _elapsed(row):
    start = row.get("attempt_start_monotonic")
    if start is None:
        return 0.0
    boot = row.get("attempt_boot_id")
    changed = (
        boot != current_context().get("boot_id")
        if boot is not None
        else row.get("boot_time") != psutil.boot_time()
    )
    if changed:
        return row.get("last_elapsed_seconds", 0.0)
    return max(row.get("last_elapsed_seconds", 0.0), time.monotonic() - start)


def _cost(row):
    uncharged_active = (
        row["status"] in ACTIVE and row.get("active_attempt_charge_kind") != "conservative"
    )
    return row["charged_job_wall_seconds"] + (_elapsed(row) if uncharged_active else 0)


def _finalize_attempt(row, status, elapsed, *, error=None, **details):
    """Record an observed end without billing an already charged attempt twice."""
    identifier = row.get("active_attempt_id") or str(row.get("attempt_start_monotonic"))
    if any(item.get("attempt_id") == identifier for item in row["attempts"]):
        return
    conservative = row.get("active_attempt_charge_kind") == "conservative"
    if conservative:
        status = "interrupted"
    else:
        row["charged_job_wall_seconds"] += elapsed
    row["attempts"].append(
        {
            "attempt_id": identifier,
            "started_unix": row.get("attempt_started_unix"),
            "job_wall_seconds": elapsed,
            "status": status,
            "charge_kind": "previously_charged_conservatively" if conservative else "observed",
            **details,
        }
    )
    row.update(status=status, last_elapsed_seconds=elapsed, completed_unix=time.time())
    if error:
        row["error"] = error


def _refresh(ledger):
    for row in ledger["jobs"].values():
        if row["status"] == "queued" and not any(
            _alive(row.get(k)) for k in ("supervisor", "worker", "launch_owner")
        ):
            row.update(
                status="interrupted",
                error="queued job lost its launcher before acquiring numerical resources",
            )
            continue
        if (
            row["status"] not in ACTIVE
            or _alive(row.get("supervisor"))
            or _alive(row.get("worker"))
        ):
            continue
        # With both observers lost the exact end time is unknowable. Never forgive
        # work or renew a budget: conservatively book the full remaining allowance.
        row["charged_job_wall_seconds"] = max(
            row["charged_job_wall_seconds"], row["max_wall_seconds"]
        )
        row.update(
            status="interrupted",
            accounting_uncertain=True,
            active_attempt_charge_kind="conservative",
            error="supervisor and worker disappeared; full job allowance charged conservatively",
        )


def _save_ledger(path, value):
    # The caller owns a separate mutation lock; JSON publication has its own
    # short reader/writer lock, so Windows replacement waits for open readers.
    write_json(path, value)


def _operation(name):
    from element_surrogate.tools import REGISTRY

    if name in INTERNAL_OPERATIONS:
        return None
    if name not in TOOL_OPERATIONS or name not in REGISTRY:
        raise ValueError("choose a scientific operation from scientific_job_catalog")
    return REGISTRY[name]


def catalog():
    from element_surrogate.tools import REGISTRY

    return {
        "operations": {
            name: str(inspect.signature(REGISTRY[name].func))
            for name in sorted(TOOL_OPERATIONS & REGISTRY.keys())
        },
        "internal_operations": {
            "generate_training_data": (
                "transport_path, qualification_path, plan_path, output_dir, approval_paths"
            ),
            "run_benchmark_study": "study_path",
        },
        "request_schema": JOB_SCHEMA,
        "resources": ["threads", "working_memory_bytes", "scratch_bytes"],
        "time_policy": "required: calendar or active; explicit positive max_wall_seconds",
        "working_memory_scope": (
            "numerical working arrays, not fixed RSS; source/mapped pages measured separately"
        ),
    }


def _bindings(name, arguments, root, expected):
    if not isinstance(arguments, dict) or not isinstance(expected, dict):
        raise ValueError("arguments and expected_input_sha256 must be objects")
    if any(
        not isinstance(k, str) or not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
        for k, v in expected.items()
    ):
        raise ValueError("expected input hashes must be absolute paths to lowercase SHA256 values")
    operation = _operation(name)
    if operation:
        bound = inspect.signature(operation.func).bind(**arguments)
        bound.apply_defaults()
        arguments.update(bound.arguments)
        names = operation.input_file_args
    elif name == "generate_training_data":
        from element_surrogate.workflow import generate_from_files

        inspect.signature(generate_from_files).bind(**arguments)
        names = ("transport_path", "qualification_path", "plan_path", "approval_paths")
    else:
        if set(arguments) != {"study_path"}:
            raise ValueError("run_benchmark_study requires study_path")
        names = ("study_path",)
    result = []
    for key in names:
        if key not in arguments:
            continue
        values = arguments[key] if isinstance(arguments[key], list) else [arguments[key]]
        normalized = []
        for value in values:
            path = (root / value).resolve()
            stat = path.stat()
            if not path.is_file():
                raise ValueError("scientific input must be a file")
            normalized.append(str(path))
            result.append(
                {
                    "path": str(path),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "expected_sha256": expected.get(str(path)),
                }
            )
        arguments[key] = normalized if isinstance(arguments[key], list) else normalized[0]
    for key in ("output_path", "output_dir"):
        if key in arguments:
            output = (root / arguments[key]).resolve()
            if not output.is_relative_to(root):
                raise ValueError("managed outputs must stay in the campaign project")
            arguments[key] = str(output)
    return result


def submit(campaign_path, request_path):
    request = read_json(request_path)
    return submit_request(campaign_path, request, request_path=str(Path(request_path).resolve()))


def start_existing(operation, directory, arguments, maximum):
    """Compatibility starters share an explicitly configured campaign if present."""
    if os.environ.get("ELEMENT_SURROGATE_MANAGED_JOB"):
        return None
    path = find_campaign(directory)
    if path is None:
        return None
    campaign, _, _ = _context(path)
    resources = campaign["policy"].get("operation_resources", {}).get(operation)
    if resources is None:
        raise ValueError(
            "configured campaign: use start_scientific_job with explicit "
            "resources, or supply operation_resources in the original campaign "
            "policy"
        )
    return submit_request(
        path,
        {
            "schema": JOB_SCHEMA,
            "operation": operation,
            "arguments": arguments,
            "resources": resources,
            "time_policy": "calendar",
            "max_wall_seconds": maximum,
        },
    )


def submit_request(
    campaign_path, request, *, request_path=None, _continuation=None, _diagnostic_retry=None
):
    if _continuation is not None and _diagnostic_retry is not None:
        raise ValueError("choose one successor mechanism")
    campaign, ledger_path, jobs = _context(campaign_path)
    request = deepcopy(request)
    if request.get("schema") != JOB_SCHEMA or request.get("time_policy") not in (
        "active",
        "calendar",
    ):
        raise ValueError("scientific job schema and explicit active/calendar time policy required")
    maximum = _positive(request.get("max_wall_seconds"), "job wall budget")
    resources = _resources(request["resources"])
    limits = campaign["policy"]["limits"]
    if any(resources[k] > limits[k] for k in resources):
        raise ValueError("job resources exceed the configured campaign")
    if maximum > limits["max_total_job_wall_seconds"]:
        raise ValueError("job maximum exceeds the total campaign job-wall budget")
    arguments = request["arguments"]
    bindings = _bindings(
        request["operation"],
        arguments,
        Path(campaign["project_dir"]),
        request.get("expected_input_sha256", {}),
    )
    if (
        "working_memory_bytes" in arguments
        and arguments["working_memory_bytes"] > resources["working_memory_bytes"]
    ):
        raise ValueError("operation working-array target exceeds its reservation")
    if request["operation"] == "run_benchmark_study":
        study = read_json(arguments["study_path"])
        if study["protocol"]["timing"]["threads"] > resources["threads"]:
            raise ValueError("benchmark numerical threads exceed the job reservation")
    implementation = implementation_identity()
    key = sha256_json({"request": request, "inputs": bindings, "implementation": implementation})
    scientific_arguments = deepcopy(arguments)
    if request["operation"] == "generate_training_data":
        scientific_arguments.pop("max_batches", None)  # execution phase, not a new dataset
    semantic = sha256_json(
        {"operation": request["operation"], "arguments": scientific_arguments, "inputs": bindings}
    )
    jobs.mkdir(exist_ok=True)
    with artifact_guard(ledger_path.with_name("campaign_mutation"), create=True):
        ledger = read_json(ledger_path)
        _refresh(ledger)
        predecessor = None
        if _continuation is not None:
            predecessor = ledger["jobs"][_continuation["predecessor_id"]]
            if predecessor.get("successor_job_id"):
                child = ledger["jobs"][predecessor["successor_job_id"]]
                if child.get("continuation_request_sha256") == _continuation["request_sha256"]:
                    return _handle(campaign_path, child)
                raise ValueError("generation phase already has a successor; inspect that job")
            if sha256_json(predecessor) != _continuation["predecessor_sha256"]:
                raise ValueError("generation predecessor changed after inspection")
            if file_hash(_continuation["manifest_path"]) != _continuation["manifest_sha256"]:
                raise ValueError("generation checkpoint changed after inspection")
            if any(_alive(predecessor.get(k)) for k in ("supervisor", "worker")):
                raise ValueError("generation phase still has an active owner")
            lineage = set(_continuation["lineage_ids"])
        elif _diagnostic_retry is not None:
            predecessor = ledger["jobs"][_diagnostic_retry["predecessor_id"]]
            if predecessor.get("successor_job_id"):
                child = ledger["jobs"][predecessor["successor_job_id"]]
                if (
                    child.get("diagnostic_retry_request_sha256")
                    == _diagnostic_retry["request_sha256"]
                ):
                    return _handle(campaign_path, child)
                raise ValueError("job already has a successor; inspect that job")
            if sha256_json(predecessor) != _diagnostic_retry["predecessor_sha256"]:
                raise ValueError("diagnostic predecessor changed after inspection")
            if any(_alive(predecessor.get(k)) for k in ("supervisor", "worker")):
                raise ValueError("diagnostic still has an active owner")
            if campaign["deadline_unix"] is not None and time.time() >= campaign["deadline_unix"]:
                raise ValueError("original campaign calendar deadline exhausted")
            if (
                predecessor["deadline_unix"] is not None
                and time.time() >= predecessor["deadline_unix"]
            ):
                raise ValueError("original diagnostic calendar deadline exhausted")
            costs = sum(_cost(row) for row in ledger["jobs"].values())
            reserved = sum(
                max(0, row["max_wall_seconds"] - _cost(row))
                for row in ledger["jobs"].values()
                if row["status"] in ACTIVE
            )
            if maximum > limits["max_total_job_wall_seconds"] - costs - reserved:
                raise ValueError(
                    "remaining original campaign cannot cover diagnostic retry allocation"
                )
            lineage = set(_diagnostic_retry["lineage_ids"])
        else:
            lineage = set()
        for row in ledger["jobs"].values():
            if row["request_sha256"] == key:
                return _handle(campaign_path, row)
            if row["semantic_sha256"] == semantic and row["id"] not in lineage:
                raise ValueError(
                    "same scientific work already exists; inspect its job, do not "
                    "duplicate or reset its budget"
                )
        if _diagnostic_retry is not None:
            Path(_diagnostic_retry["output_root"]).mkdir(parents=True, exist_ok=False)
        identifier = uuid.uuid4().hex
        folder = jobs / identifier
        folder.mkdir()
        spec = {
            "request": request,
            "inputs": bindings,
            "request_sha256": key,
            "caller_request_path": request_path,
            "campaign_policy_sha256": campaign["policy_sha256"],
            "implementation": implementation,
        }
        write_json(folder / "request.json", spec, exclusive=True)
        now = time.time()
        row = {
            "id": identifier,
            "status": "queued",
            "operation": request["operation"],
            "request_sha256": key,
            "semantic_sha256": semantic,
            "resources": resources,
            "max_wall_seconds": maximum,
            "time_policy": request["time_policy"],
            "submitted_unix": now,
            "deadline_unix": now + maximum if request["time_policy"] == "calendar" else None,
            "charged_job_wall_seconds": 0.0,
            "attempts": [],
            "cancel_requested": False,
            "launch_owner": _identity(os.getpid()),
            "folder": str(folder),
            "result_path": str(folder / "attempt_0001_result.json"),
        }
        if predecessor is not None and _continuation is not None:
            row.update(
                predecessor_job_id=predecessor["id"],
                continuation_request_sha256=_continuation["request_sha256"],
                generation_manifest_sha256=_continuation["manifest_sha256"],
                continuation_rationale=_continuation["rationale"],
                lineage_job_ids=_continuation["lineage_ids"],
            )
            if predecessor["deadline_unix"] is not None:
                row["deadline_unix"] = min(row["deadline_unix"], predecessor["deadline_unix"])
            predecessor["successor_job_id"] = identifier
        elif predecessor is not None:
            row.update(
                predecessor_job_id=predecessor["id"],
                diagnostic_retry_request_sha256=_diagnostic_retry["request_sha256"],
                diagnostic_retry_inspection_sha256=_diagnostic_retry["inspection_sha256"],
                predecessor_implementation_sha256=_diagnostic_retry[
                    "original_implementation_sha256"
                ],
                retry_rationale=_diagnostic_retry["rationale"],
                lineage_job_ids=_diagnostic_retry["lineage_ids"],
            )
            if predecessor["deadline_unix"] is not None:
                row["deadline_unix"] = min(row["deadline_unix"], predecessor["deadline_unix"])
            predecessor["successor_job_id"] = identifier
        ledger["jobs"][identifier] = row
        _save_ledger(ledger_path, ledger)
    return _launch(campaign_path, identifier)


def _handle(campaign_path, row):
    return {
        "campaign_path": str(Path(campaign_path).resolve()),
        "job_id": row["id"],
        "status": row["status"],
        "is_pending": row["status"] in PENDING,
        "is_terminal": row["status"] in TERMINAL,
        "job_directory": row["folder"],
        "result_path": row["result_path"],
        "request_sha256": row["request_sha256"],
    }


def _launch(campaign_path, identifier):
    campaign, ledger_path, jobs_root = _context(campaign_path)
    with (jobs_root / identifier / "supervisor.log").open("ab") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "element_surrogate.jobs",
                "supervise",
                str(Path(campaign_path).resolve()),
                identifier,
            ],
            cwd=campaign["project_dir"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=os.name != "nt",
            creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
            if os.name == "nt"
            else 0,
        )
    with artifact_guard(ledger_path.with_name("campaign_mutation"), create=True):
        ledger = read_json(ledger_path)
        row = ledger["jobs"][identifier]
        try:
            row["supervisor"] = _identity(process.pid)
        except psutil.NoSuchProcess:
            pass  # A fast terminal worker may already have completed.
        _save_ledger(ledger_path, ledger)
    return _handle(campaign_path, row)


def campaign_status(campaign_path):
    campaign, path, _ = _context(campaign_path)
    with artifact_guard(path.with_name("campaign_mutation"), create=True):
        ledger = read_json(path)
        _refresh(ledger)
        _save_ledger(path, ledger)
    rows = list(ledger["jobs"].values())
    return {
        "campaign_path": str(Path(campaign_path).resolve()),
        "policy_sha256": campaign["policy_sha256"],
        "limits": campaign["policy"]["limits"],
        "deadline_unix": campaign["deadline_unix"],
        "total_job_wall_seconds": sum(_cost(r) for r in rows),
        "reserved_resources": {
            k: sum(r["resources"][k] for r in rows if r["status"] in ACTIVE)
            for k in ("threads", "working_memory_bytes", "scratch_bytes")
        },
        "jobs": [
            {
                **_handle(campaign_path, r),
                "charged_job_wall_seconds": _cost(r),
                "operation": r["operation"],
            }
            for r in rows
        ],
        "scope": (
            "managed numerical jobs; working-array/scratch reservations are not process RSS caps"
        ),
    }


def job_status(campaign_path, job_id):
    _job_id(job_id)
    campaign_status(campaign_path)
    _, path, _ = _context(campaign_path)
    row = read_json(path)["jobs"][job_id]
    result = {
        **_handle(campaign_path, row),
        **{k: v for k, v in row.items() if k not in {"folder", "semantic_sha256"}},
        "elapsed_current_attempt_seconds": _elapsed(row) if row["status"] in ACTIVE else 0,
    }
    if row["operation"] == "generate_training_data":
        spec = read_json(Path(row["folder"]) / "request.json")
        manifest = Path(spec["request"]["arguments"]["output_dir"]) / "generation.json"
        if manifest.exists():
            generation = read_json(manifest)
            result["generation_progress"] = {
                "status": generation["status"],
                "surviving_rows": sum(generation["survivors"].values()),
                "proposed_rows": sum(generation["attempts"].values()),
                "survivors_by_setting": generation["survivors"],
                "committed_batches": len(generation["shards"]),
                "scope": (
                    "manifest progress only; full shard verification occurs on "
                    "completion/recovery/capacity report"
                ),
            }
    return result


def cancel(campaign_path, job_id, rationale):
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("cancellation rationale required")
    _, path, _ = _context(campaign_path)
    with artifact_guard(path.with_name("campaign_mutation"), create=True):
        ledger = read_json(path)
        row = ledger["jobs"][_job_id(job_id)]
        row.update(cancel_requested=True, cancellation_rationale=rationale)
        _save_ledger(path, ledger)
    if row["status"] in ACTIVE and not _alive(row.get("supervisor")) and _alive(row.get("worker")):
        _stop_tree(row["worker"])
        with artifact_guard(path.with_name("campaign_mutation"), create=True):
            ledger = read_json(path)
            row = ledger["jobs"][job_id]
            elapsed = _elapsed(row)
            _finalize_attempt(row, "cancelled", elapsed, orphan_cleanup=True)
            _save_ledger(path, ledger)
    return job_status(campaign_path, job_id)


def inspect_recovery(campaign_path, job_id, output_path):
    row = job_status(campaign_path, job_id)
    if row["status"] in ACTIVE or _alive(row.get("worker")) or _alive(row.get("supervisor")):
        raise ValueError(
            "job still has an active owner; cancel/wait or reconnect, never duplicate it"
        )
    output = Path(output_path).resolve()
    if output.is_relative_to(Path(row["job_directory"])):
        raise ValueError("save inspection outside the job directory")
    inputs = _verify_inputs(read_json(Path(row["job_directory"]) / "request.json"))
    receipt = {
        "schema": "element_surrogate.scientific_job_inspection.v1",
        "campaign_path": str(Path(campaign_path).resolve()),
        "job_id": job_id,
        "status_sha256": sha256_json(row),
        "inputs": inputs,
        "remaining_job_wall_seconds": max(
            0, row["max_wall_seconds"] - row["charged_job_wall_seconds"]
        ),
        "accounting_uncertain": row.get("accounting_uncertain", False),
    }
    write_json(output, receipt, exclusive=True)
    return {"inspection_path": str(output), "inspection_sha256": file_hash(output), **receipt}


def resume(campaign_path, job_id, inspection_path, rationale):
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("recovery rationale required")
    inspection = read_json(inspection_path)
    row = job_status(campaign_path, job_id)
    if inspection.get("job_id") != job_id or inspection.get("campaign_path") != str(
        Path(campaign_path).resolve()
    ):
        raise ValueError("inspection belongs to a different job/campaign")
    if sha256_json(row) != inspection["status_sha256"]:
        raise ValueError("stale inspection; job state changed")
    if row.get("successor_job_id"):
        raise ValueError("generation phase has a successor; inspect/resume that job instead")
    if row["status"] not in {"failed", "cancelled", "incomplete", "interrupted"}:
        raise ValueError("job is complete or its original deadline is exhausted")
    if (
        row.get("accounting_uncertain")
        or row["charged_job_wall_seconds"] >= row["max_wall_seconds"]
    ):
        raise ValueError("no verified original job budget remains")
    if row["deadline_unix"] is not None and time.time() >= row["deadline_unix"]:
        raise ValueError("original calendar deadline exhausted")
    spec = read_json(Path(row["job_directory"]) / "request.json")
    if _verify_inputs(spec) != inspection["inputs"]:
        raise ValueError("inputs changed after inspection")
    if row["operation"] == "run_benchmark_study":
        # run_study itself refuses any partially running fit. First apply the
        # study-specific inspection/recovery disposition, preserving its budget.
        from element_surrogate.benchmark.study import study_status

        state = study_status(spec["request"]["arguments"]["study_path"])
        if any(t.get("status") == "running" for t in state.get("trials", [])):
            raise ValueError("inspect/dispose interrupted benchmark fits before managed recovery")
    _, path, _ = _context(campaign_path)
    with artifact_guard(path.with_name("campaign_mutation"), create=True):
        ledger = read_json(path)
        current = ledger["jobs"][job_id]
        if (
            current["status"] != row["status"]
            or current.get("successor_job_id")
            or _alive(current.get("worker"))
            or _alive(current.get("supervisor"))
        ):
            raise ValueError("job ownership changed after inspection")
        history = current.get("recovery_history", [])
        history.append(
            {
                "inspection_sha256": file_hash(inspection_path),
                "rationale": rationale,
                "unix": time.time(),
            }
        )
        current.update(
            status="queued",
            cancel_requested=False,
            recovery_history=history,
            result_path=str(
                Path(current["folder"]) / f"attempt_{len(current['attempts']) + 1:04d}_result.json"
            ),
            launch_owner=_identity(os.getpid()),
        )
        current.pop("worker", None)
        current.pop("supervisor", None)
        current.pop("attempt_start_monotonic", None)
        _save_ledger(path, ledger)
    return _launch(campaign_path, job_id)


def continue_generation_phase(campaign_path, job_id, inspection_path, request_path):
    """Link a clean generation checkpoint to an explicit further allocation.

    Scientific inputs, committed IDs, original generation cap, campaign costs
    and any original calendar deadline remain. Fault recovery is separate.
    """
    from element_surrogate.data import verify_generation

    allocation = read_json(request_path)
    if allocation.get("schema") != "element_surrogate.generation_continuation.v1":
        raise ValueError("generation_continuation.v1 request required")
    allowed = {"schema", "max_batches", "max_wall_seconds", "rationale", "resources"}
    if set(allocation) - allowed:
        raise ValueError("continuation changes execution allocation only, not scientific inputs")
    batches = allocation.get("max_batches")
    if batches is not None and (type(batches) is not int or batches < 1):
        raise ValueError("max_batches must be null or a positive integer")
    maximum = _positive(allocation.get("max_wall_seconds"), "phase wall budget")
    rationale = allocation.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("continuation rationale required")
    row = job_status(campaign_path, job_id)
    _, ledger_path, _ = _context(campaign_path)
    ledger = read_json(ledger_path)
    prior = ledger["jobs"][job_id]
    allocation_sha = sha256_json(allocation)
    if prior.get("successor_job_id"):
        child = ledger["jobs"][prior["successor_job_id"]]
        if child.get("continuation_request_sha256") == allocation_sha:
            return _handle(campaign_path, child)
        raise ValueError("generation phase already has a successor; inspect that job")
    if row["operation"] != "generate_training_data" or row["status"] != "incomplete":
        raise ValueError(
            "only a clean paused generation phase can continue; inspect fault recovery otherwise"
        )
    if row.get("accounting_uncertain") or any(_alive(row.get(k)) for k in ("worker", "supervisor")):
        raise ValueError("verified accounting and inactive generation ownership required")
    inspection = read_json(inspection_path)
    if (
        inspection.get("job_id") != job_id
        or inspection.get("campaign_path") != str(Path(campaign_path).resolve())
        or inspection.get("status_sha256") != sha256_json(row)
    ):
        raise ValueError("current inspection for this generation phase required")
    spec = read_json(Path(row["job_directory"]) / "request.json")
    if _verify_inputs(spec) != inspection["inputs"]:
        raise ValueError("generation inputs changed after inspection")
    if file_hash(row["result_path"]) != row.get("result_sha256"):
        raise ValueError("previous generation result changed")
    result = read_json(row["result_path"])
    if result.get("status") != "paused_at_batch_boundary":
        raise ValueError("previous execution did not pause at a clean batch boundary")
    arguments = deepcopy(spec["request"]["arguments"])
    directory = Path(arguments["output_dir"])
    manifest = directory / "generation.json"
    record = verify_generation(directory)
    plan = read_json(arguments["plan_path"])
    if record["plan_sha256"] != sha256_json(plan) or record["status"] != "paused_at_batch_boundary":
        raise ValueError("checkpoint differs from the original scientific plan or is complete")
    lineage = [*prior.get("lineage_job_ids", []), job_id]
    if any(ledger["jobs"][i].get("accounting_uncertain") for i in lineage):
        raise ValueError("generation lineage has uncertain accounting")
    charged = sum(_cost(ledger["jobs"][i]) for i in lineage)
    if maximum > plan["max_wall_seconds"] - charged:
        raise ValueError("continuation exceeds the original cumulative generation allowance")
    if row["deadline_unix"] is not None and time.time() >= row["deadline_unix"]:
        raise ValueError("original calendar deadline exhausted")
    arguments["max_batches"] = batches
    request = deepcopy(spec["request"])
    request.update(
        arguments=arguments,
        max_wall_seconds=maximum,
        resources=allocation.get("resources", request["resources"]),
        predecessor_job_id=job_id,
    )
    return submit_request(
        campaign_path,
        request,
        request_path=str(Path(request_path).resolve()),
        _continuation={
            "predecessor_id": job_id,
            "predecessor_sha256": sha256_json(prior),
            "manifest_path": str(manifest),
            "manifest_sha256": file_hash(manifest),
            "lineage_ids": lineage,
            "request_sha256": allocation_sha,
            "rationale": rationale,
        },
    )


def managed_context_required(operation_name):
    """Guard only public heavy tools in configured projects; not arbitrary code."""
    if operation_name in TOOL_OPERATIONS and not os.environ.get("ELEMENT_SURROGATE_MANAGED_JOB"):
        if find_campaign(Path.cwd()) is not None:
            raise ValueError(
                "configured campaign: submit this operation with "
                "start_scientific_job; see scientific_job_catalog"
            )


def _verify_inputs(spec, *, require_same_implementation=True):
    if require_same_implementation and implementation_identity() != spec["implementation"]:
        raise ValueError(
            "installed implementation changed since scheduling; do not resume under different code"
        )
    actual = []
    for row in spec["inputs"]:
        path = Path(row["path"])
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (row["size"], row["mtime_ns"]):
            raise ValueError("scientific input changed since scheduling: " + str(path))
        digest = file_hash(path)
        if row["expected_sha256"] is not None and digest != row["expected_sha256"]:
            raise ValueError("scientific input hash differs: " + str(path))
        actual.append({"path": str(path), "sha256": digest})
    return actual


def _stop_tree(identity):
    if not _alive(identity):
        return
    root = psutil.Process(identity["pid"])
    children = root.children(recursive=True)
    processes = [*reversed(children), root]
    for process in processes:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=1)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=2)
    if alive:
        raise RuntimeError("owned numerical processes did not exit; reservation retained")


def _admit(campaign, ledger, row):
    now = time.time()
    if row["cancel_requested"]:
        row["status"] = "cancelled"
        return False
    if (campaign["deadline_unix"] is not None and now >= campaign["deadline_unix"]) or (
        row["deadline_unix"] is not None and now >= row["deadline_unix"]
    ):
        row.update(status="timed_out", error="explicit calendar deadline exhausted")
        return False
    active = [r for r in ledger["jobs"].values() if r["status"] in ACTIVE]
    remaining = campaign["policy"]["limits"]["max_total_job_wall_seconds"] - sum(
        _cost(r) for r in ledger["jobs"].values()
    )
    needed = row["max_wall_seconds"] - row["charged_job_wall_seconds"]
    reserved = sum(max(0, r["max_wall_seconds"] - _cost(r)) for r in active)
    if needed > remaining - reserved:
        if not active:
            row.update(
                status="refused_budget",
                error="remaining campaign budget cannot cover the declared job allowance",
            )
        return False
    if any(
        r["status"] == "queued" and r["submitted_unix"] < row["submitted_unix"]
        for r in ledger["jobs"].values()
    ):
        return False
    if any(
        sum(r["resources"][key] for r in active) + value > campaign["policy"]["limits"][key]
        for key, value in row["resources"].items()
    ):
        return False
    row.update(
        status="starting",
        attempt_start_monotonic=time.monotonic(),
        attempt_started_unix=now,
        boot_time=psutil.boot_time(),
        attempt_boot_id=current_context().get("boot_id"),
        active_attempt_id=uuid.uuid4().hex,
        active_attempt_charge_kind=None,
        last_elapsed_seconds=0.0,
    )
    return True


def supervise(campaign_path, job_id):
    from element_surrogate.benchmark.study import _guard

    campaign, path, jobs = _context(campaign_path)
    folder = jobs / _job_id(job_id)
    with _guard(folder):
        while True:
            with artifact_guard(path.with_name("campaign_mutation"), create=True):
                ledger = read_json(path)
                row = ledger["jobs"][job_id]
                row["supervisor"] = _identity(os.getpid())
                _refresh(ledger)
                admitted = _admit(campaign, ledger, row) if row["status"] == "queued" else False
                _save_ledger(path, ledger)
            if admitted:
                break
            if row["status"] != "queued":
                return
            time.sleep(0.2)
        environment = dict(os.environ)
        environment.update({key: str(row["resources"]["threads"]) for key in THREAD_ENV})
        environment.update(
            ELEMENT_SURROGATE_MANAGED_JOB=job_id,
            ELEMENT_SURROGATE_NUMERICAL_THREADS=str(row["resources"]["threads"]),
            ELEMENT_SURROGATE_CAMPAIGN=str(Path(campaign_path).resolve()),
        )
        worker = None
        observed_children = {}
        status, error = "failed", None
        try:
            with (folder / "worker.log").open("ab") as log:
                worker = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "element_surrogate.jobs",
                        "execute",
                        str(Path(campaign_path).resolve()),
                        job_id,
                    ],
                    cwd=campaign["project_dir"],
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=os.name != "nt",
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            identity = _identity(worker.pid)
            while True:
                with artifact_guard(path.with_name("campaign_mutation"), create=True):
                    ledger = read_json(path)
                    row = ledger["jobs"][job_id]
                    row.update(
                        status="running", worker=identity, last_elapsed_seconds=_elapsed(row)
                    )
                    try:
                        process = psutil.Process(worker.pid)
                        descendants = process.children(recursive=True)
                        observed_children.update({p.pid: _identity(p.pid) for p in descendants})
                        rss = sum(
                            p.memory_info().rss for p in [process, *descendants] if p.is_running()
                        )
                        row["peak_tree_rss_bytes"] = max(row.get("peak_tree_rss_bytes", 0), rss)
                    except psutil.Error:
                        pass
                    wall_expired = _cost(row) >= row["max_wall_seconds"]
                    calendar_expired = any(
                        t is not None and time.time() >= t
                        for t in (row["deadline_unix"], campaign["deadline_unix"])
                    )
                    total_expired = (
                        sum(_cost(r) for r in ledger["jobs"].values())
                        >= campaign["policy"]["limits"]["max_total_job_wall_seconds"]
                    )
                    stop = (
                        row["cancel_requested"] or wall_expired or calendar_expired or total_expired
                    )
                    _save_ledger(path, ledger)
                if stop:
                    status = "cancelled" if row["cancel_requested"] else "timed_out"
                    _stop_tree(identity)
                    worker.wait()
                    break
                if worker.poll() is not None:
                    if worker.returncode == 0:
                        result = read_json(row["result_path"])
                        status = (
                            "incomplete"
                            if row["operation"] == "generate_training_data"
                            and result.get("status") != "complete"
                            else "complete"
                        )
                    else:
                        error = "scientific worker failed; inspect failure.json and worker.log"
                    break
                time.sleep(0.2)
        except Exception:
            error = traceback.format_exc()[-8000:]
            if worker is not None:
                _stop_tree(_identity(worker.pid) if worker.poll() is None else None)
                worker.wait()
        # A worker returning or crashing must not free resources while an
        # observed numerical descendant remains alive.
        for child in observed_children.values():
            _stop_tree(child)
        with artifact_guard(path.with_name("campaign_mutation"), create=True):
            ledger = read_json(path)
            row = ledger["jobs"][job_id]
            elapsed = _elapsed(row)
            _finalize_attempt(row, status, elapsed, error=error)
            if row["status"] in {"complete", "incomplete"}:
                row["result_sha256"] = file_hash(row["result_path"])
            _save_ledger(path, ledger)


def execute(campaign_path, job_id):
    campaign, _, jobs = _context(campaign_path)
    folder = jobs / _job_id(job_id)
    spec = read_json(folder / "request.json")
    request = spec["request"]
    try:
        inputs = _verify_inputs(spec)
        write_json(folder / "verified_inputs.json", inputs)
        name, arguments = request["operation"], request["arguments"]
        if name == "generate_training_data":
            from element_surrogate.workflow import generate_from_files

            result = generate_from_files(**arguments)
        elif name == "run_benchmark_study":
            from element_surrogate.benchmark.study import run_study

            result = run_study(**arguments)
        else:
            operation = _operation(name)
            store = RunStore(campaign["project_dir"])
            result = record_call(store, operation, arguments)
        if _verify_inputs(spec) != inputs:
            raise ValueError("scientific inputs changed during work")
        _, ledger_path, _ = _context(campaign_path)
        row = read_json(ledger_path)["jobs"][job_id]
        write_json(row["result_path"], result, exclusive=True)
    except Exception:
        write_json(folder / "failure.json", {"error": traceback.format_exc()[-12000:]})
        raise


if __name__ == "__main__":
    mode, campaign_path, identifier = sys.argv[1:]
    {"execute": execute, "supervise": supervise}[mode](campaign_path, identifier)
