"""Fresh diagnostic allocations retain failed work and the original campaign cap."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from element_surrogate import jobs
from element_surrogate.benchmark.protocol import file_hash, read_json, write_json
from element_surrogate.benchmark.study import _guard
from element_surrogate.runs import implementation_identity, sha256_json

SCHEMA = "element_surrogate.diagnostic_retry.v1"
INSPECTION = "element_surrogate.diagnostic_retry_inspection.v1"
# These operations inspect immutable inputs and may publish fresh derived outputs.
# Mutable fitting/generation/test-state operations keep their own recovery protocols.
OPERATIONS = {
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
    "analyze_generalized_gradients",
    "compare_field_exports",
    "check_tracking_convergence",
    "check_adaptive_tracking_convergence",
    "introspect_card_dataset",
    "validate_training_data_plan",
    "compare_with_description",
    "trace_reference_orbit",
    "fit_reference_orbit",
    "prepare_mapped_field_store",
    "assess_field_readiness",
    "assess_map_symmetry",
    "assess_native_theory_base",
    "analyze_transverse_symplecticity",
    "assess_canonical_shear_coordinates",
}


def _checked_parent(campaign_path, job_id):
    _, ledger_path, _ = jobs._context(campaign_path)
    raw = read_json(ledger_path)["jobs"][jobs._job_id(job_id)]
    if raw["operation"] not in OPERATIONS:
        raise ValueError(
            "operation needs its state/checkpoint-specific recovery, not diagnostic replay"
        )
    row = jobs.job_status(campaign_path, job_id)
    if row["operation"] not in OPERATIONS:
        raise ValueError(
            "operation needs its state/checkpoint-specific recovery, not diagnostic replay"
        )
    if row["status"] not in {"failed", "cancelled", "timed_out", "interrupted", "refused_budget"}:
        raise ValueError("only a terminal failed diagnostic can receive a new allocation")
    if any(jobs._alive(row.get(k)) for k in ("supervisor", "worker")):
        raise ValueError("diagnostic has an active owner; do not duplicate it")
    return row


def _original_inputs(row, spec):
    identity = sha256_json(
        {
            "request": spec["request"],
            "inputs": spec["inputs"],
            "implementation": spec["implementation"],
        }
    )
    if identity != spec["request_sha256"] or identity != row["request_sha256"]:
        raise ValueError("predecessor request/implementation identity changed")
    if any(
        row[key] != spec["request"][key]
        for key in ("operation", "resources", "max_wall_seconds", "time_policy")
    ):
        raise ValueError("predecessor execution metadata differs from its frozen request")
    actual = jobs._verify_inputs(spec, require_same_implementation=False)
    verified = Path(row["job_directory"]) / "verified_inputs.json"
    if verified.exists():
        if actual != read_json(verified):
            raise ValueError("scientific input differs from the predecessor's verified payload")
    elif any(item["expected_sha256"] is None for item in spec["inputs"]):
        raise ValueError(
            "original payload hash was never verified; diagnostic replay cannot attest it"
        )
    return actual


def inspect_retry(campaign_path, job_id, output_path):
    campaign, _, _ = jobs._context(campaign_path)
    row = _checked_parent(campaign_path, job_id)
    output = Path(output_path).resolve()
    folder = Path(row["job_directory"])
    if not output.is_relative_to(Path(campaign["project_dir"])) or output.is_relative_to(folder):
        raise ValueError("save retry inspection inside the project, outside the predecessor job")
    with _guard(folder):
        row = _checked_parent(campaign_path, job_id)
        spec = read_json(folder / "request.json")
        inputs = _original_inputs(row, spec)
        campaign, ledger_path, _ = jobs._context(campaign_path)
        ledger = read_json(ledger_path)
        record = {
            "schema": INSPECTION,
            "campaign_path": str(Path(campaign_path).resolve()),
            "job_id": job_id,
            "status_sha256": sha256_json(row),
            "predecessor_sha256": sha256_json(ledger["jobs"][job_id]),
            "inputs": inputs,
            "original_implementation_sha256": sha256_json(spec["implementation"]),
            "current_implementation_sha256": sha256_json(implementation_identity()),
            "prior_charge_seconds": row["charged_job_wall_seconds"],
            "policy_sha256": campaign["policy_sha256"],
            "deadline_unix": campaign["deadline_unix"],
            "remaining_campaign_job_wall_seconds": max(
                0,
                campaign["policy"]["limits"]["max_total_job_wall_seconds"]
                - sum(jobs._cost(item) for item in ledger["jobs"].values()),
            ),
            "original_job_calendar_deadline_unix": row["deadline_unix"],
            "request_template": {
                "schema": SCHEMA,
                "max_wall_seconds": None,
                "resources": row["resources"],
                "fresh_output_root": None,
                "acknowledge_implementation_sha256": sha256_json(implementation_identity()),
                "rationale": None,
            },
            "scope": (
                "A new diagnostic attempt, not same-job resume or a refund; "
                "no passed credit for predecessor."
            ),
        }
        write_json(output, record, exclusive=True)
    return {"inspection_path": str(output), "inspection_sha256": file_hash(output), **record}


def retry(campaign_path, job_id, inspection_path, request_path):
    allocation = read_json(request_path)
    if allocation.get("schema") != SCHEMA or set(allocation) != {
        "schema",
        "max_wall_seconds",
        "resources",
        "fresh_output_root",
        "acknowledge_implementation_sha256",
        "rationale",
    }:
        raise ValueError("diagnostic_retry.v1 allocation required")
    jobs._positive(allocation["max_wall_seconds"], "new diagnostic wall allocation")
    if not isinstance(allocation["rationale"], str) or not allocation["rationale"].strip():
        raise ValueError("diagnostic retry rationale required")
    campaign, ledger_path, jobs_root = jobs._context(campaign_path)
    request_sha = sha256_json(allocation)
    prior = read_json(ledger_path)["jobs"][jobs._job_id(job_id)]
    if prior.get("successor_job_id"):
        child = read_json(ledger_path)["jobs"][prior["successor_job_id"]]
        if child.get("diagnostic_retry_request_sha256") == request_sha:
            return jobs._handle(campaign_path, child)
        raise ValueError("diagnostic already has a successor; inspect that job")
    row = _checked_parent(campaign_path, job_id)
    with _guard(row["job_directory"]):
        row = _checked_parent(campaign_path, job_id)
        inspection = read_json(inspection_path)
        if (
            inspection.get("schema") != INSPECTION
            or inspection.get("campaign_path") != str(Path(campaign_path).resolve())
            or inspection.get("job_id") != job_id
            or inspection.get("status_sha256") != sha256_json(row)
        ):
            raise ValueError("current retry inspection for this diagnostic required")
        current_sha = sha256_json(implementation_identity())
        if (
            inspection["current_implementation_sha256"] != current_sha
            or allocation["acknowledge_implementation_sha256"] != current_sha
        ):
            raise ValueError("inspect and acknowledge the current implementation identity")
        spec = read_json(Path(row["job_directory"]) / "request.json")
        actual = _original_inputs(row, spec)
        if actual != inspection["inputs"]:
            raise ValueError("diagnostic inputs changed after inspection")
        raw_root = allocation["fresh_output_root"]
        if not isinstance(raw_root, str) or not raw_root.strip():
            raise ValueError("fresh output root required")
        root = (Path(campaign["project_dir"]) / raw_root).resolve()
        if (
            not root.is_relative_to(Path(campaign["project_dir"]))
            or root.is_relative_to(jobs_root)
            or root.exists()
        ):
            raise ValueError("diagnostic outputs require a new root within the project")
        arguments = deepcopy(spec["request"]["arguments"])
        for key in ("output_path", "output_dir"):
            if arguments.get(key) is not None:
                name = Path(arguments[key]).name if key == "output_path" else "output"
                arguments[key] = str(root / name)
        request = deepcopy(spec["request"])
        request.update(
            arguments=arguments,
            resources=allocation["resources"],
            max_wall_seconds=allocation["max_wall_seconds"],
            predecessor_job_id=job_id,
            expected_input_sha256={item["path"]: item["sha256"] for item in actual},
        )
        return jobs.submit_request(
            campaign_path,
            request,
            request_path=str(Path(request_path).resolve()),
            _diagnostic_retry={
                "predecessor_id": job_id,
                "predecessor_sha256": inspection["predecessor_sha256"],
                "request_sha256": request_sha,
                "inspection_sha256": file_hash(inspection_path),
                "original_implementation_sha256": inspection["original_implementation_sha256"],
                "lineage_ids": [*prior.get("lineage_job_ids", []), job_id],
                "rationale": allocation["rationale"],
                "output_root": str(root),
            },
        )
