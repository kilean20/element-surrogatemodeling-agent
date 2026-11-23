"""Measured finite-cohort equivariance tied to a native transport declaration.

A supported assessment permits a declared numerical approximation; it is not a
proof of exact field symmetry or authenticated scientific approval.
"""

from __future__ import annotations

import numpy as np

from element_surrogate.benchmark.protocol import protocol_identity, read_json, write_json
from element_surrogate.contracts import array_digest
from element_surrogate.runs import implementation_identity, sha256_json
from element_surrogate.transport import scope_description, verify_transport_source

SCHEMA = "element_surrogate.map_symmetry.v1"


def validate_group(group, dimension):
    signs = np.asarray(group)
    if (
        signs.ndim != 2
        or signs.shape[1] != dimension + 2
        or not np.isin(signs, (-1, 1)).all()
        or len(signs) < 1
    ):
        raise ValueError("finite transverse reflection sign group required")
    rows = {tuple(row) for row in signs.tolist()}
    if len(rows) != len(signs) or (1,) * (dimension + 2) not in rows:
        raise ValueError("unique group elements including identity required")
    if any(np.any(row[4:] != 1) or row[0] != row[1] or row[2] != row[3] for row in signs):
        raise ValueError("spatial reflections act on transverse position/slope pairs only")
    if any(tuple(a * b) not in rows for a in signs for b in signs):
        raise ValueError("reflection group must be closed under composition")
    return signs.astype(int)


def assess(
    transport,
    ranges,
    settings,
    group,
    component_limits,
    *,
    random_probes,
    seed,
    rationale,
    effective_coupling_range=None,
):
    d = transport.codec.dimension
    signs = validate_group(group, d)
    scope = scope_description(transport, ranges, settings, effective_coupling_range)
    box = np.asarray(scope["ranges"])
    limits = np.asarray(component_limits, float)
    if limits.shape != (d,) or not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("one explicit positive symmetry limit per native output required")
    if type(random_probes) is not int or random_probes < 0 or type(seed) is not int or seed < 0:
        raise ValueError("nonnegative explicit probe count and seed required")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("scientific symmetry rationale required")
    for row in signs:
        if np.any((row[:d] == -1) & (box[:, 0] != -box[:, 1])):
            raise ValueError("reflection must preserve the declared input box")
    corners = np.array(np.meshgrid(*box, indexing="ij")).reshape(d, -1).T
    random = np.random.default_rng(seed).uniform(box[:, 0], box[:, 1], (random_probes, d))
    states = np.unique(np.vstack((corners, random, box.mean(axis=1)[None])), axis=0)
    reports = []
    verify_transport_source(transport)
    for setting in settings:
        control = (setting["effective_rigidity_tm"], setting["field_scale"])
        nominal, base, _ = transport.run(states, *control)
        for row in signs:
            if np.all(row == 1):
                continue
            actual, result, _ = transport.run(states * row[:d], *control)
            common = base.reached_exit & result.reached_exit
            changes = (
                np.max(np.abs(actual[common] - nominal[common] * row[:d]), axis=0)
                if common.any()
                else None
            )
            changed = int(np.count_nonzero(base.status != result.status))
            reports.append(
                {
                    "setting_id": setting["id"],
                    "signs": row.tolist(),
                    "common_survivors": int(common.sum()),
                    "status_changes": changed,
                    "max_component_difference": None if changes is None else changes.tolist(),
                    "supported": bool(common.any() and changed == 0 and np.all(changes <= limits)),
                }
            )
    verify_transport_source(transport)
    if scope_description(transport, ranges, settings, effective_coupling_range) != scope:
        raise ValueError("transport changed during symmetry assessment")
    record = {
        "schema": SCHEMA,
        "scope": scope,
        "group": signs.tolist(),
        "component_limits": limits.tolist(),
        "seed": seed,
        "random_probes": random_probes,
        "probe_count": len(states),
        "probe_sha256": array_digest(states),
        "reports": reports,
        "implementation_sha256": sha256_json(implementation_identity()),
        "supported": bool(reports and all(r["supported"] for r in reports)),
        "rationale": rationale,
        "limits": (
            "Finite paired-map evidence only; not proof of exact physical"
            " symmetry. Exact projection is an adopted numerical model as"
            "sumption."
        ),
    }
    return record


def validate_measurement(record, group, dimension, *, expected_scope=None, require_current=False):
    """Check measured evidence, including truthful negative/unsupported results."""
    signs = validate_group(group, dimension)
    if (
        not isinstance(record, dict)
        or record.get("schema") != SCHEMA
        or record.get("group") != signs.tolist()
    ):
        raise ValueError("bound symmetry measurement record/group required")
    limits = np.asarray(record["component_limits"], float)
    if limits.shape != (dimension,) or not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("positive finite measured symmetry component limits required")
    expected = {
        (s["id"], tuple(row))
        for s in record["scope"]["settings"]
        for row in signs.tolist()
        if not all(v == 1 for v in row)
    }
    reports = record["reports"]
    actual = {(r["setting_id"], tuple(r["signs"])) for r in reports}
    if not expected or actual != expected or len(reports) != len(expected):
        raise ValueError("incomplete measured symmetry settings/group coverage")
    outcomes = []
    for row in reports:
        count, changed = row["common_survivors"], row["status_changes"]
        if type(count) is not int or count < 0 or type(changed) is not int or changed < 0:
            raise ValueError("invalid measured symmetry survivor/status counts")
        difference = row["max_component_difference"]
        if count:
            difference = np.asarray(difference, float)
            if (
                difference.shape != (dimension,)
                or not np.isfinite(difference).all()
                or np.any(difference < 0)
            ):
                raise ValueError("finite nonnegative measured component differences required")
        elif difference is not None:
            raise ValueError("no-survivor symmetry result cannot have component differences")
        supported = bool(count and changed == 0 and np.all(difference <= limits))
        if row["supported"] is not supported:
            raise ValueError("unsupported measured symmetry disposition")
        outcomes.append(supported)
    if record["supported"] is not bool(all(outcomes)):
        raise ValueError("unsupported measured symmetry disposition")
    if expected_scope is not None and record["scope"] != expected_scope:
        raise ValueError("symmetry source/reference/region/range/settings scope differs")
    if require_current and record["implementation_sha256"] != sha256_json(
        implementation_identity()
    ):
        raise ValueError("symmetry assessment implementation changed")


def validate(record, group, dimension, *, expected_scope=None, require_current=False):
    """Check a linked measured record; do not grade a rationale as physical truth."""
    validate_measurement(
        record, group, dimension, expected_scope=expected_scope, require_current=require_current
    )
    signs = validate_group(group, dimension)
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise ValueError("bound symmetry assessment required")
    if record.get("group") != signs.tolist() or record.get("supported") is not True:
        raise ValueError("bound symmetry assessment does not support this group")
    if not isinstance(record.get("rationale"), str) or not record["rationale"].strip():
        raise ValueError("bound symmetry rationale required")
    limits = np.asarray(record.get("component_limits"), float)
    if limits.shape != (dimension,) or not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("invalid measured symmetry limits")
    reports = record.get("reports", [])
    settings = record["scope"]["settings"]
    expected_pairs = {
        (s["id"], tuple(row))
        for s in settings
        for row in signs.tolist()
        if not all(v == 1 for v in row)
    }
    actual_pairs = {(r["setting_id"], tuple(r["signs"])) for r in reports}
    if not expected_pairs or actual_pairs != expected_pairs or len(reports) != len(expected_pairs):
        raise ValueError("incomplete measured symmetry settings/group coverage")
    for row in reports:
        difference = np.asarray(row["max_component_difference"], float)
        if (
            row.get("supported") is not True
            or row["common_survivors"] < 1
            or row["status_changes"] != 0
            or difference.shape != (dimension,)
            or not np.isfinite(difference).all()
            or np.any(difference > limits)
        ):
            raise ValueError("unsupported measured symmetry disposition")
    if expected_scope is not None and record["scope"] != expected_scope:
        raise ValueError("symmetry source/reference/region/range/settings scope differs")
    if require_current and record["implementation_sha256"] != sha256_json(
        implementation_identity()
    ):
        raise ValueError("symmetry assessment implementation changed")


def validate_inputs(record, inputs):
    box = np.asarray(record["scope"]["ranges"], float)
    d = len(box)
    if not np.all((inputs[:, :d] >= box[:, 0]) & (inputs[:, :d] <= box[:, 1])):
        raise ValueError("training inputs leave the symmetry-assessed box")
    coupling = inputs[:, -1] / inputs[:, -2]
    interval = record["scope"].get("effective_coupling_range")
    if interval is None:
        values = [
            s["field_scale"] / s["effective_rigidity_tm"] for s in record["scope"]["settings"]
        ]
        inside = np.isin(coupling, values)
    else:
        inside = (coupling >= interval[0]) & (coupling <= interval[1])
    if not inside.all():
        raise ValueError("training controls leave the symmetry-assessed scope")


def from_files(transport_path, request_path, output_path) -> dict:
    from element_surrogate.workflow import load_transport

    transport, _ = load_transport(transport_path)
    record = assess(transport, **read_json(request_path))
    write_json(output_path, record, exclusive=True)
    return {
        "assessment_path": str(output_path),
        "assessment_sha256": protocol_identity(record),
        "supported": record["supported"],
        "reports": record["reports"],
        "limits": record["limits"],
    }
