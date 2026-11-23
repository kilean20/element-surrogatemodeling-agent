"""Bounded, deterministic, resumable training-label generation.

Immutable NPZ shards retain all proposals/statuses and separately identify the
requested survivors. A manifest publishes a batch only after both shards are
complete. Resume checks identities and hashes. This CPU reference is scalable
in storage and memory; it is not a GPU/compiled throughput claim.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import numpy as np

from element_surrogate.benchmark.protocol import (
    check_approvals,
    file_hash,
    protocol_identity,
    read_json,
    write_json,
)
from element_surrogate.benchmark.study import _guard
from element_surrogate.contracts import StaticMagneticCodec, coordinate_metadata
from element_surrogate.field_readiness import validate_assessment
from element_surrogate.file_io import publish_file
from element_surrogate.runs import implementation_identity, sha256_json
from element_surrogate.transport import scope_description, verify_transport_source

SCHEMA = "element_surrogate.generation.v2"


def validate_plan(transport, qualification, plan):
    """Fail before proposals if ranges, numerical qualification or budgets disagree."""
    if plan.get("schema") != SCHEMA:
        raise ValueError("unsupported generation plan")
    verify_transport_source(transport)
    scope = scope_description(
        transport, plan["ranges"], plan["settings"], plan.get("effective_coupling_range")
    )
    if plan.get("transport") != scope["transport"]:
        raise ValueError("generation transport declaration differs from actual transport")
    if (
        qualification.get("schema") != "element_surrogate.transport_qualification.v1"
        or qualification.get("accepted") is not True
        or qualification.get("scope") != scope
        or qualification.get("scope_sha256") != protocol_identity(scope)
    ):
        raise ValueError(
            "accepted qualification for exactly this transport/range/settings required"
        )
    if qualification.get("implementation_sha256") != sha256_json(implementation_identity()):
        raise ValueError("qualification implementation changed")
    if plan["qualification_sha256"] != protocol_identity(qualification):
        raise ValueError("qualification receipt changed")
    assessment = plan.get("field_readiness", {})
    validate_assessment(transport, assessment)
    if plan.get("field_readiness_sha256") != protocol_identity(assessment):
        raise ValueError("field readiness assessment identity changed")
    for key in ("batch_size", "max_proposals_per_setting", "seed"):
        if type(plan[key]) is not int or plan[key] < (0 if key == "seed" else 1):
            raise ValueError(f"invalid generation budget {key}")
    if not np.isfinite(plan["max_wall_seconds"]) or plan["max_wall_seconds"] <= 0:
        raise ValueError("positive generation wall budget required")
    if (
        not isinstance(plan.get("material_boundary_evidence"), str)
        or not plan["material_boundary_evidence"].strip()
    ):
        raise ValueError("declare supplied material limits or explicitly unknown material boundary")
    for s in plan["settings"]:
        if (
            s["split"] not in ("train", "validation", "test")
            or type(s["surviving_rows"]) is not int
            or not 0 < s["surviving_rows"] <= plan["max_proposals_per_setting"]
        ):
            raise ValueError("setting split and achievable positive survivor budget required")
    return scope


def _write_shard(path, arrays):
    """Complete publication; an orphan from an interrupted commit must match exactly."""
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != set(arrays) or any(
                not np.array_equal(stored[k], v, equal_nan=True)
                if np.asarray(v).dtype.kind == "f"
                else not np.array_equal(stored[k], v)
                for k, v in arrays.items()
            ):
                raise ValueError("uncommitted generation shard differs; inspect before resuming")
        return
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        publish_file(temporary, path, exclusive=True)
    finally:
        temporary.unlink(missing_ok=True)


def verify_generation(directory):
    """Verify committed bytes without loading proposal or test arrays."""
    directory = Path(directory).resolve()
    record = read_json(directory / "generation.json")
    if (
        record.get("schema") != "element_surrogate.generation.record.v1"
        or protocol_identity(record["plan"]) != record["plan_sha256"]
    ):
        raise ValueError("generation plan identity changed")
    counts = {s["id"]: 0 for s in record["plan"]["settings"]}
    attempts = {k: 0 for k in counts}
    for shard in record["shards"]:
        counts[shard["setting_id"]] += shard["survivors"]
        attempts[shard["setting_id"]] += shard["proposals"]
        for kind in ("proposals", "survivors"):
            member = shard["files"][kind]
            path = (directory / member["name"]).resolve()
            if not path.is_relative_to(directory) or file_hash(path) != member["sha256"]:
                raise ValueError("committed generation shard changed or leaves its directory")
    if counts != record["survivors"] or attempts != record["attempts"]:
        raise ValueError("generation counts disagree with shard ledger")
    return record


def generate(
    transport,
    qualification,
    plan,
    output_dir,
    approval_paths,
    *,
    max_batches=None,
    execution_check=None,
) -> dict:
    """Generate after three bound user attestations, or resume the same plan.

    ``max_batches`` is an optional scheduling boundary, never an accuracy or
    survivor requirement. Deadlines are checked between bounded tracking batches;
    hard process-level deadlines belong to a supervising caller.
    """
    validate_plan(transport, qualification, plan)
    approvals = check_approvals(
        plan, approval_paths, ("description_target", "region_ranges", "compute_budget")
    )
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    identity = protocol_identity(plan)
    if max_batches is not None and (type(max_batches) is not int or max_batches < 1):
        raise ValueError("max_batches must be positive")
    with _guard(directory):
        if execution_check is not None:
            execution_check()
        path = directory / "generation.json"
        if path.exists():
            record = verify_generation(directory)
            if record["plan_sha256"] != identity or record["approvals"] != approvals:
                raise ValueError("cannot resume with changed plan or approvals")
        else:
            record = {
                "schema": "element_surrogate.generation.record.v1",
                "plan": plan,
                "plan_sha256": identity,
                "approvals": approvals,
                "shards": [],
                "survivors": {s["id"]: 0 for s in plan["settings"]},
                "attempts": {s["id"]: 0 for s in plan["settings"]},
                "elapsed_seconds": 0.0,
                "status": "created",
            }
            write_json(path, record, exclusive=True)
        if record["status"] == "complete":
            return generation_summary(directory, record)
        if record["status"] == "invalidated":
            raise ValueError(
                "generation invalidated by changed source/configuration; "
                "inspect and use a new campaign"
            )
        started = time.monotonic()
        prior_elapsed = record["elapsed_seconds"]
        batches = 0
        box = np.asarray(plan["ranges"], float)

        def finish(status) -> dict:
            try:
                validate_plan(transport, qualification, plan)
            except (ValueError, OSError) as error:
                record["status"] = "invalidated"
                record["invalidation_reason"] = str(error)
                record["elapsed_seconds"] = prior_elapsed + time.monotonic() - started
                write_json(path, record)
                raise
            record["status"] = status
            record["elapsed_seconds"] = prior_elapsed + time.monotonic() - started
            write_json(path, record)
            return generation_summary(directory, record)

        for index, setting in enumerate(plan["settings"]):
            name = setting["id"]
            while record["survivors"][name] < setting["surviving_rows"]:
                record["elapsed_seconds"] = prior_elapsed + time.monotonic() - started
                if record["elapsed_seconds"] >= plan["max_wall_seconds"]:
                    return finish("wall_budget_exhausted")
                proposed = record["attempts"][name]
                count = min(plan["batch_size"], plan["max_proposals_per_setting"] - proposed)
                if count <= 0:
                    return finish("proposal_budget_exhausted")
                rng = np.random.default_rng(np.random.SeedSequence([plan["seed"], index, proposed]))
                batch_started = time.monotonic()
                states = rng.uniform(box[:, 0], box[:, 1], (count, transport.codec.dimension))
                tracking_started = time.monotonic()
                targets, result, audit = transport.run(
                    states, setting["effective_rigidity_tm"], setting["field_scale"]
                )
                tracking_seconds = time.monotonic() - tracking_started
                kept = np.flatnonzero(result.reached_exit)[
                    : setting["surviving_rows"] - record["survivors"][name]
                ]
                ids = np.array([f"{index}:{j}" for j in range(proposed, proposed + count)])
                inputs = np.column_stack(
                    [
                        states,
                        np.full(count, setting["effective_rigidity_tm"]),
                        np.full(count, setting["field_scale"]),
                    ]
                )
                base = f"setting_{index:06d}_proposal_{proposed:012d}"
                all_arrays = {
                    "inputs": inputs,
                    "targets": targets,
                    "sample_id": ids,
                    "status": result.status,
                    "path_length_m": result.path_length_m,
                    "setting_id": np.full(count, name),
                }
                selected = {k: v[kept] for k, v in all_arrays.items()}
                _write_shard(directory / (base + "_proposals.npz"), all_arrays)
                _write_shard(directory / (base + "_survivors.npz"), selected)
                files = {
                    kind: {
                        "name": base + f"_{kind}.npz",
                        "sha256": file_hash(directory / (base + f"_{kind}.npz")),
                    }
                    for kind in ("proposals", "survivors")
                }
                record["shards"].append(
                    {
                        "setting_id": name,
                        "split": setting["split"],
                        "proposals": count,
                        "survivors": len(kept),
                        "unused_survivors": int(result.reached_exit.sum()) - len(kept),
                        "status_counts": result.summary(),
                        "stencil_audit": audit,
                        "files": files,
                    }
                )
                record["survivors"][name] += len(kept)
                record["attempts"][name] += count
                record["status"] = "generating"
                record["elapsed_seconds"] = prior_elapsed + time.monotonic() - started
                write_json(path, record)
                # Observational timing is separate from deterministic particle
                # shards. A crash after commit may leave an explicitly untimed
                # batch; never regenerate labels just to fill that evidence gap.
                metrics = directory / "batch_metrics"
                metrics.mkdir(exist_ok=True)
                write_json(
                    metrics / (base + ".json"),
                    {
                        "plan_sha256": identity,
                        "shard_sha256": protocol_identity(record["shards"][-1]),
                        "tracking_seconds": tracking_seconds,
                        "batch_seconds": time.monotonic() - batch_started,
                    },
                    exclusive=True,
                )
                batches += 1
                if max_batches is not None and batches >= max_batches:
                    return finish("paused_at_batch_boundary")
        return finish("complete")


def generation_summary(directory, record=None) -> dict:
    record = verify_generation(directory) if record is None else record
    return {
        "generation_path": str(Path(directory).resolve() / "generation.json"),
        "plan_sha256": record["plan_sha256"],
        "status": record["status"],
        "surviving_rows": sum(record["survivors"].values()),
        "proposed_rows": sum(record["attempts"].values()),
        "survivors_by_setting": record["survivors"],
        "elapsed_seconds": record["elapsed_seconds"],
        "shards": len(record["shards"]),
    }


def export_benchmark_cohorts(directory, output_dir, *, maximum_rows_per_split) -> dict:
    """Publish bounded cohort *descriptors*, without opening particle/test arrays.

    Benchmark fitting opens only train/validation members. Test member arrays are
    opened by the existing locked-study, release-approved evaluation path. A
    descriptor pins every member's bytes and exact prefix count before selection.
    """
    directory = Path(directory).resolve()
    record = verify_generation(directory)
    if (
        record["status"] != "complete"
        or type(maximum_rows_per_split) is not int
        or maximum_rows_per_split < 1
    ):
        raise ValueError("complete generation and explicit positive cohort row cap required")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    transport = record["plan"]["transport"]
    coordinates = coordinate_metadata(StaticMagneticCodec(**transport["codec"]))
    names = [c["name"] for c in coordinates]
    units = [c["unit"] for c in coordinates]
    metadata = {
        "input_names": names + ["effective_rigidity_tm", "field_scale"],
        "input_units": units + ["T*m", "1"],
        "coordinate_names": names,
        "coordinate_units": units,
        "dataset_identity_sha256": record["plan_sha256"],
    }
    results = {}
    for split in ("train", "validation", "test"):
        selected = [s for s in record["shards"] if s["split"] == split]
        groups = sorted({s["setting_id"] for s in selected})
        if not groups or maximum_rows_per_split < len(groups):
            raise ValueError("row cap must permit at least one survivor per split setting")
        allocation = {
            g: maximum_rows_per_split // len(groups) + int(i < maximum_rows_per_split % len(groups))
            for i, g in enumerate(groups)
        }
        members = []
        for shard in selected:
            count = min(allocation[shard["setting_id"]], shard["survivors"])
            if count:
                source = shard["files"]["survivors"]
                members.append(
                    {
                        "path": str(directory / source["name"]),
                        "sha256": source["sha256"],
                        "rows": count,
                    }
                )
                allocation[shard["setting_id"]] -= count
        descriptor = {
            "schema": "element_surrogate.cohort_shards.v1",
            "metadata": metadata,
            "members": members,
            "rows": sum(m["rows"] for m in members),
            "split": split,
            "selection": "first surviving proposals per setting under a declared balanced cap",
        }
        path = output_dir / f"{split}.json"
        if path.exists():
            if read_json(path) != descriptor:
                raise ValueError("cohort descriptor already exists with different contents")
        else:
            write_json(path, descriptor, exclusive=True)
        results[split] = {
            "path": str(path),
            "sha256": file_hash(path),
            "rows": descriptor["rows"],
            "groups": groups,
        }
    return {
        "dataset_identity_sha256": record["plan_sha256"],
        "splits": results,
        "test_arrays_opened": False,
        "metadata": metadata,
    }
