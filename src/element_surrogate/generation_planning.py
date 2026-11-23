"""Computed generation contracts and measured capacity, without changing physics."""

from pathlib import Path

from element_surrogate.benchmark.protocol import file_hash, protocol_identity, read_json, write_json
from element_surrogate.contracts import coordinate_metadata
from element_surrogate.data import SCHEMA, validate_plan, verify_generation
from element_surrogate.workflow import load_transport


def build_request(
    transport_path,
    qualification_path,
    readiness_path,
    request_path,
    output_path,
    population_policy_path=None,
):
    request = read_json(request_path)
    required = {
        "schema",
        "settings",
        "seed",
        "batch_size",
        "max_proposals_per_setting",
        "max_wall_seconds",
        "material_boundary_evidence",
    }
    if set(request) != required or request["schema"] != "element_surrogate.generation_request.v1":
        raise ValueError(
            "explicit generation_request.v1 settings, seed and budgets required; "
            "no guessed transport payload"
        )
    if any(set(s) != {"id", "split", "surviving_rows"} for s in request["settings"]):
        raise ValueError("request settings name qualified IDs, split and survivor count only")
    ids = [s["id"] for s in request["settings"]]
    if len(ids) != len(set(ids)):
        raise ValueError("unique setting IDs required")
    qualification = read_json(qualification_path)
    qualified = qualification["scope"]["settings"]
    if ids != [s["id"] for s in qualified]:
        raise ValueError("request must preserve all qualified settings in their order")
    assessment = read_json(readiness_path)
    transport, _ = load_transport(transport_path)
    plan = {k: v for k, v in request.items() if k not in {"schema", "settings"}}
    plan.update(
        schema=SCHEMA,
        transport=transport.description(),
        qualification_sha256=protocol_identity(qualification),
        field_readiness=assessment,
        field_readiness_sha256=protocol_identity(assessment),
        ranges=qualification["scope"]["ranges"],
        settings=[{**q, **s} for q, s in zip(qualified, request["settings"], strict=True)],
    )
    coupling = qualification["scope"].get("effective_coupling_range")
    if coupling is not None:
        plan["effective_coupling_range"] = coupling
    validate_plan(transport, qualification, plan)
    population = None
    if population_policy_path is not None:
        from element_surrogate.population import check_plan

        policy = read_json(population_policy_path)
        counts, floor_met = check_plan(plan, policy)
        population = {
            "policy_sha256": protocol_identity(policy),
            "planned_survivors_by_split": counts,
            "planned_floor_met": floor_met,
        }
    write_json(output_path, plan, exclusive=True)
    return {
        "plan_path": str(Path(output_path).resolve()),
        "plan_sha256": protocol_identity(plan),
        "coordinates": coordinate_metadata(transport.codec),
        "ranges": plan["ranges"],
        "settings": plan["settings"],
        "native_planes": True,
        "total_requested_survivors": sum(s["surviving_rows"] for s in plan["settings"]),
        "authority_created": False,
        "approval_subject": "exact resulting plan_sha256",
        "population_policy": population,
    }


def capacity_report(directory, output_path):
    """Describe measured batches; never call an extrapolation scale acceptance."""
    directory = Path(directory).resolve()
    record = verify_generation(directory)
    measured, missing = {}, []
    for shard in record["shards"]:
        name = shard["files"]["proposals"]["name"].removesuffix("_proposals.npz")
        path = directory / "batch_metrics" / (name + ".json")
        if not path.exists():
            missing.append(name)
            continue
        row = read_json(path)
        if (
            row["shard_sha256"] != protocol_identity(shard)
            or row["plan_sha256"] != record["plan_sha256"]
        ):
            raise ValueError("timing receipt is not bound to the committed batch")
        if (
            min(row["tracking_seconds"], row["batch_seconds"]) <= 0
            or row["tracking_seconds"] > row["batch_seconds"]
        ):
            raise ValueError("invalid batch timing receipt")
        measured.setdefault(shard["setting_id"], []).append((shard, row))
    settings = []
    for setting in record["plan"]["settings"]:
        rows = measured.get(setting["id"], [])
        kept = sum(s["survivors"] for s, _ in rows)
        proposed = sum(s["proposals"] for s, _ in rows)
        elapsed = sum(t["batch_seconds"] for _, t in rows)
        costs = [t["batch_seconds"] / s["survivors"] for s, t in rows if s["survivors"]]
        remaining = setting["surviving_rows"] - record["survivors"][setting["id"]]
        settings.append(
            {
                "setting_id": setting["id"],
                "measured_batches": len(rows),
                "proposed_rows": proposed,
                "retained_survivors": kept,
                "measured_batch_seconds": elapsed,
                "retained_survivors_per_second": kept / elapsed if elapsed else None,
                "observed_batch_size_range": [
                    min(s["proposals"] for s, _ in rows),
                    max(s["proposals"] for s, _ in rows),
                ]
                if rows
                else None,
                "remaining_requested_rows": remaining,
                "remaining_batch_time_projection_seconds": [
                    remaining * min(costs),
                    remaining * max(costs) if all(s["survivors"] for s, _ in rows) else None,
                ]
                if len(costs) >= 2
                else None,
                "projection_status": "conditional on measured batch size, setting and yield"
                if len(costs) >= 2
                else "inconclusive: fewer than two survivor-bearing batches",
            }
        )
    projections = [
        [0.0, 0.0]
        if s["remaining_requested_rows"] == 0
        else s["remaining_batch_time_projection_seconds"]
        for s in settings
    ]
    complete = all(p is not None and p[1] is not None for p in projections)
    low = sum(p[0] for p in projections if p is not None)
    known_high = sum(p[1] for p in projections if p is not None and p[1] is not None)
    result = {
        "schema": "element_surrogate.generation_capacity.v1",
        "generation_path": str(directory / "generation.json"),
        "generation_sha256": file_hash(directory / "generation.json"),
        "plan_sha256": record["plan_sha256"],
        "settings": settings,
        "committed_batches_without_timing": missing,
        "generation_active_seconds": record["elapsed_seconds"],
        "remaining_measured_batch_seconds_range": [low, known_high if complete else None],
        "all_remaining_settings_calibrated": complete,
        "resource_prompt_after_seconds": 12 * 3600,
        "resource_question_required": low > 12 * 3600 or known_high > 12 * 3600,
        "scope": (
            "measured batch tracking/publication; startup, source validation, "
            "scheduling, OS cache and future changes are additional costs; "
            "projections are not confidence intervals or scale acceptance"
        ),
        "million_row_acceptance": "requires actual verified surviving rows; no extrapolated pass",
    }
    write_json(output_path, result, exclusive=True)
    return result
