"""Conditional work-time estimates from explicit measured observations.

Observation hashes establish identity, not authenticity of measurements. Clients
must obtain quantities/times from actual pilots or a supplied simulator's records.
No ideal hardware scaling or scientific adequacy is inferred by this arithmetic.
"""

import math
from pathlib import Path

from element_surrogate.benchmark.protocol import file_hash, protocol_identity, read_json, write_json


def _number(value, name, *, zero=False):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        or (not zero and value == 0)
    ):
        raise ValueError(f"finite {'nonnegative' if zero else 'positive'} {name} required")
    return value


def estimate(request_path, output_path):
    request = read_json(request_path)
    if (
        set(request) != {"schema", "phases", "overheads_seconds", "resource_prompt_after_hours"}
        or request["schema"] != "element_surrogate.workload_request.v1"
    ):
        raise ValueError("workload_request.v1 phases, overheads and resource threshold required")
    threshold = _number(request["resource_prompt_after_hours"], "resource threshold") * 3600
    if not isinstance(request["phases"], list) or not request["phases"]:
        raise ValueError("at least one explicitly planned phase required")
    names, phases, sources = set(), [], []
    for phase in request["phases"]:
        if set(phase) != {
            "name",
            "work_units",
            "work_unit",
            "measurement_paths",
            "context",
            "assumptions",
        }:
            raise ValueError("phase requires work quantity, observations, context and assumptions")
        name = phase["name"]
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("distinct nonempty phase names required")
        names.add(name)
        units = _number(phase["work_units"], "remaining work units", zero=True)
        if not isinstance(phase["work_unit"], str) or not phase["work_unit"].strip():
            raise ValueError("explicit work unit required")
        if not isinstance(phase["context"], dict) or not phase["context"]:
            raise ValueError("hardware, batch and numerical context must be declared")
        if not isinstance(phase["assumptions"], str) or not phase["assumptions"].strip():
            raise ValueError("extrapolation assumptions required")
        costs, quantities, times, zero_yield = [], [], [], False
        if not isinstance(phase["measurement_paths"], list):
            raise ValueError("measurement_paths must be a list")
        seen_paths = set()
        for path in phase["measurement_paths"]:
            resolved = str(Path(path).resolve())
            if resolved in seen_paths:
                raise ValueError("duplicate observation is not a repeated measurement")
            seen_paths.add(resolved)
            observation = read_json(path)
            if (
                set(observation)
                != {
                    "schema",
                    "work_unit",
                    "completed_units",
                    "elapsed_seconds",
                    "context",
                    "included_costs",
                    "source_artifacts",
                }
                or observation["schema"] != "element_surrogate.workload_observation.v1"
            ):
                raise ValueError("explicit workload_observation.v1 required")
            if (
                observation["work_unit"] != phase["work_unit"]
                or observation["context"] != phase["context"]
            ):
                raise ValueError("measurement work unit/hardware/numerical context differs")
            quantity = _number(observation["completed_units"], "observed quantity", zero=True)
            elapsed = _number(observation["elapsed_seconds"], "observed elapsed seconds")
            if (
                not isinstance(observation["included_costs"], str)
                or not observation["included_costs"].strip()
            ):
                raise ValueError("measurement must describe included preparation/validation costs")
            if (
                not isinstance(observation["source_artifacts"], list)
                or not observation["source_artifacts"]
            ):
                raise ValueError("observations require identity-matched source receipts")
            for source in observation["source_artifacts"]:
                if (
                    set(source) != {"path", "sha256"}
                    or file_hash(source["path"]) != source["sha256"]
                ):
                    raise ValueError("workload observation source identity differs")
            sources.append({"path": str(Path(path).resolve()), "sha256": file_hash(path)})
            if quantity:
                costs.append(elapsed / quantity)
                quantities.append(quantity)
                times.append(elapsed)
            else:
                zero_yield = True
        if units == 0:
            bounds, point, state = [0.0, 0.0], 0.0, "no_remaining_work"
        elif len(costs) < 2:
            bounds, point, state = None, None, "needs_two_positive_observations"
        elif zero_yield:
            bounds, point, state = (
                [units * min(costs), None],
                None,
                "zero_yield_observation_unbounded",
            )
        else:
            bounds = [units * min(costs), units * max(costs)]
            point = units * sum(times) / sum(quantities)
            state = "conditional_measured_projection"
        phases.append(
            {
                "name": name,
                "work_units": units,
                "work_unit": phase["work_unit"],
                "context": phase["context"],
                "assumptions": phase["assumptions"],
                "positive_observations": len(costs),
                "zero_yield_observed": zero_yield,
                "projected_elapsed_seconds_range": bounds,
                "projected_elapsed_seconds_point": point,
                "status": state,
            }
        )
    overheads = request["overheads_seconds"]
    if overheads is not None:
        if not isinstance(overheads, list) or len(overheads) != 2:
            raise ValueError("overheads must be an explicit [low, high] seconds pair or null")
        for value in overheads:
            _number(value, "overhead", zero=True)
        if overheads[1] < overheads[0]:
            raise ValueError("overhead range must be ordered")
    bounds = [p["projected_elapsed_seconds_range"] for p in phases]
    known_low = sum(b[0] for b in bounds if b is not None)
    work_complete = all(b is not None and b[1] is not None for b in bounds)
    work_high = sum(b[1] for b in bounds) if work_complete else None
    total_range = [
        known_low + (overheads[0] if overheads is not None else 0),
        work_high + overheads[1] if work_complete and overheads is not None else None,
    ]
    known_upper = sum(b[1] for b in bounds if b is not None and b[1] is not None)
    question = (
        known_upper + (overheads[1] if overheads is not None else 0) > threshold
        or total_range[0] > threshold
    )
    if total_range[1] is not None and total_range[1] > threshold:
        question = True
    result = {
        "schema": "element_surrogate.workload_estimate.v1",
        "request_sha256": protocol_identity(request),
        "phases": phases,
        "source_observations": sources,
        "projected_remaining_elapsed_seconds_range": total_range,
        "complete_estimate": work_complete and overheads is not None,
        "queue_transfer_startup_seconds_range": overheads,
        "resource_prompt_after_seconds": threshold,
        "resource_question_required": bool(question),
        "resource_question": (
            "A measured projection exceeds the elapsed-time threshold. "
            "Is a faster CPU/GPU server "
            "or allocated HPC resource available? Reuse an existing answer; authorized "
            "local continuation remains possible."
        )
        if question
        else None,
        "limitations": (
            "Observed min/max costs are conditional ranges, not confidence intervals. "
            "Included work must cover validation/checkpoints/final fits; unlisted work "
            "and unknown queue/transfer time remain additional. Context and receipt hashes "
            "do not authenticate execution. Do not assume linear device/worker scaling."
        ),
    }
    write_json(output_path, result, exclusive=True)
    return {**result, "estimate_path": str(Path(output_path).resolve())}
