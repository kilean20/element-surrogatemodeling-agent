"""Explicit survivor requirements, distinct from statistical sample adequacy."""

from pathlib import Path

from element_surrogate.benchmark.protocol import protocol_identity, read_json, write_json


def validate_policy(policy):
    required = {"schema", "purpose", "minimum_survivors", "count_basis", "rationale"}
    if (
        set(policy) != required
        or policy["schema"] != "element_surrogate.population_policy.v1"
        or policy["purpose"] not in ("pilot", "full")
        or type(policy["minimum_survivors"]) is not int
        or policy["minimum_survivors"] < 1
        or policy["count_basis"] not in ("total", "train")
        or not isinstance(policy["rationale"], str)
        or not policy["rationale"].strip()
    ):
        raise ValueError(
            "explicit purpose, positive survivor floor, count basis and rationale required"
        )


def planned_counts(plan):
    counts = dict.fromkeys(("train", "validation", "test"), 0)
    seen = set()
    for setting in plan["settings"]:
        split, size = setting["split"], setting["surviving_rows"]
        if setting["id"] in seen or split not in counts or type(size) is not int or size < 1:
            raise ValueError(
                "unique settings with positive retained-survivor split quotas required"
            )
        seen.add(setting["id"])
        counts[split] += size
    if not seen:
        raise ValueError("at least one generation setting required")
    return counts


def check_plan(plan, policy):
    validate_policy(policy)
    counts = planned_counts(plan)
    amount = sum(counts.values()) if policy["count_basis"] == "total" else counts["train"]
    floor_met = amount >= policy["minimum_survivors"]
    if policy["purpose"] == "full" and not floor_met:
        raise ValueError("full generation plan is below the declared retained-survivor minimum")
    return counts, floor_met


def assess(plan_path, policy_path, output_path, generation_dir=None):
    plan, policy = read_json(plan_path), read_json(policy_path)
    planned, planned_floor = check_plan(plan, policy)
    actual, status = None, "not_started"
    if generation_dir is not None:
        from element_surrogate.data import verify_generation

        record = verify_generation(generation_dir)
        if record["plan_sha256"] != protocol_identity(plan):
            raise ValueError("generation belongs to a different population plan")
        status = record["status"]
        actual = dict.fromkeys(planned, 0)
        for setting in plan["settings"]:
            actual[setting["split"]] += record["survivors"][setting["id"]]
    retained = (
        None
        if actual is None
        else sum(actual.values())
        if policy["count_basis"] == "total"
        else actual["train"]
    )
    verified = retained is not None and retained >= policy["minimum_survivors"]
    result = {
        "schema": "element_surrogate.population_assessment.v1",
        "plan_sha256": protocol_identity(plan),
        "policy_sha256": protocol_identity(policy),
        "purpose": policy["purpose"],
        "minimum_survivors": policy["minimum_survivors"],
        "count_basis": policy["count_basis"],
        "planned_survivors_by_split": planned,
        "planned_floor_met": planned_floor,
        "verified_survivors_by_split": actual,
        "generation_status": status,
        "verified_floor_met": verified,
        "full_population_requirement_met": bool(
            policy["purpose"] == "full" and status == "complete" and verified
        ),
        "sample_adequacy": (
            "Not established by count: evaluate setting/domain coverage, representative "
            "data-size learning curves, optimizer convergence and independent errors."
        ),
        "scope": "Counts verified committed survivors, not proposals or projected generation.",
    }
    write_json(output_path, result, exclusive=True)
    return {**result, "assessment_path": str(Path(output_path).resolve())}
