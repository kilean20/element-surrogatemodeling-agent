"""Scientific contracts and caller-attested approvals, independent of model framework.

Approval records are consistency guards for the trusted local workflow. They do
not authenticate a human or sandbox a training adapter. Clients must obtain the
decision from their user; no tool in this package manufactures approval records.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

from element_surrogate.file_io import publish_file, read_utf8

SCHEMA = "element_surrogate.benchmark.v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def protocol_identity(protocol):
    return hashlib.sha256(canonical(protocol).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(read_utf8(path), parse_constant=invalid, object_pairs_hook=unique)


def write_json(path, value, *, exclusive=False):
    """Atomic JSON; exclusive records cannot be replaced by a competing writer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        publish_file(temporary, path, exclusive=exclusive)
    finally:
        temporary.unlink(missing_ok=True)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, minimum=1):
    _require(type(value) is int and value >= minimum, f"{name}: integer >= {minimum} required")


def _positive(value, name, *, zero=False):
    import math

    _require(
        type(value) in (int, float)
        and math.isfinite(value)
        and (value >= 0 if zero else value > 0),
        f"{name}: finite {'nonnegative' if zero else 'positive'} value required",
    )


def _text(value, name):
    _require(isinstance(value, str) and bool(value.strip()), f"{name}: nonempty text required")


def _digest(value, name):
    _require(
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
        f"{name}: SHA256 required",
    )


def validate_protocol(p):
    """Validate explicit study choices; no previous magnet's thresholds are defaults."""
    try:
        _require(p["schema"] == SCHEMA, "unsupported benchmark schema")
        _text(p["name"], "name")
        coordinates = p["coordinates"]
        _require(len(coordinates) in (4, 5, 6), "4D, 5D or 6D outputs required")
        names = []
        for c in coordinates:
            for key in ("name", "unit", "definition", "role"):
                _text(c[key], f"coordinate {key}")
            _positive(c["error_scale"], "coordinate error_scale")
            names.append(c["name"])
        _require(len(set(names)) == len(names), "unique coordinate names required")
        roles = [c["role"] for c in coordinates]
        _require(
            roles[:4] == ["position", "slope", "position", "slope"],
            "first four coordinates must declare position/slope pairs",
        )
        _require(
            sorted(roles[4:])
            == (
                [] if len(roles) == 4 else ["energy"] if len(roles) == 5 else ["arrival", "energy"]
            ),
            "declare energy/arrival roles",
        )
        _text(p["coordinate_convention"], "coordinate_convention")
        inputs = p["inputs"]
        _require(isinstance(inputs, list) and bool(inputs), "declare inference input columns")
        _require(len(set(inputs)) == len(inputs), "unique input names required")
        for name in inputs:
            _text(name, "input name")
        _require(len(p["input_units"]) == len(inputs), "one unit for every input column required")
        for unit in p["input_units"]:
            _text(unit, "input unit")
        _require(not set(inputs) & set(p["forbidden_inputs"]), "forbidden input leakage")
        for key in (
            "field_sha256",
            "reference_sha256",
            "reconstruction_sha256",
            "integration_qualification_sha256",
            "region_sha256",
        ):
            _digest(p["physics"][key], key)
        _require(p["physics"]["native_planes"] is True, "native element planes required")
        _text(p["physics"]["qualification_scope"], "qualification_scope")
        _text(p["physics"]["boundary_convention"], "boundary_convention")
        dataset = p["dataset"]
        _require(
            set(dataset["splits"]) == {"train", "validation", "test"},
            "three declared splits required",
        )
        _digest(dataset["identity_sha256"], "dataset identity")
        _text(dataset["access_and_reuse"], "access_and_reuse")
        split_unit = dataset.get("split_unit", "setting")
        _require(split_unit in ("setting", "particle"), "split_unit must be setting or particle")
        groups = set()
        for split in ("train", "validation", "test"):
            item = dataset["splits"][split]
            _text(item["path"], "cohort path")
            _digest(item["sha256"], "cohort hash")
            _integer(item["rows"], "cohort rows")
            _require(
                isinstance(item["groups"], list) and bool(item["groups"]), "split groups required"
            )
            _require(all(isinstance(g, str) and g for g in item["groups"]), "groups use text IDs")
            current = set(item["groups"])
            _require(
                len(current) == len(item["groups"])
                and (split_unit == "particle" or not groups & current),
                "split setting groups must be unique and disjoint",
            )
            groups |= current
        resources = p["resources"]
        _integer(resources["max_rows_per_split"], "max_rows_per_split")
        _integer(resources["max_npz_uncompressed_bytes"], "max_npz_uncompressed_bytes")
        _integer(resources["max_checkpoint_bytes"], "max_checkpoint_bytes")
        _positive(resources["max_study_wall_seconds"], "max_study_wall_seconds")
        _positive(resources["max_test_wall_seconds"], "max_test_wall_seconds")
        _require(
            all(i["rows"] <= resources["max_rows_per_split"] for i in dataset["splits"].values()),
            "cohort exceeds approved row budget",
        )
        seen = set()
        candidates = p["candidates"]
        _require(isinstance(candidates, list) and bool(candidates), "candidates required")
        architecture_seeds = {}
        for item in candidates:
            identifier = item["id"]
            _text(identifier, "candidate id")
            _require(
                all(c.isalnum() or c in "-_" for c in identifier), "safe candidate id required"
            )
            _require(identifier not in seen, "duplicate candidate id")
            seen.add(identifier)
            _text(item["architecture"], "architecture")
            _text(item["rationale"], "rationale")
            _text(item["forward_description"], "forward_description")
            _require(item["kind"] in ("stochastic", "deterministic"), "candidate kind required")
            _require(isinstance(item["configuration"], dict), "configuration mapping required")
            canonical(item["configuration"])
            seeds = item["seeds"]
            _require(
                isinstance(seeds, list) and bool(seeds) and len(set(seeds)) == len(seeds),
                "explicit distinct seeds required",
            )
            for seed in seeds:
                _integer(seed, "seed", 0)
            _require(
                item["kind"] != "deterministic" or len(seeds) == 1,
                "deterministic hyperparameters are not repeated stochastic seeds",
            )
            signature = (item["kind"], tuple(sorted(seeds)))
            previous = architecture_seeds.setdefault(item["architecture"], signature)
            _require(previous == signature, "same architecture configurations need matched seeds")
            _integer(item["max_steps"], "max_steps", 0)
            _positive(item["max_wall_seconds"], "max_wall_seconds")
        rule = p["selection"]
        _require(
            rule["primary"] in ("worst_scaled_rmse", "worst_scaled_setting_mean_rmse"),
            "unsupported primary score",
        )
        _positive(rule["relative_accuracy_tie"], "relative_accuracy_tie", zero=True)
        _require(rule["relative_accuracy_tie"] <= 1, "relative accuracy tie must be <= 1")
        policy = rule.get("policy", "accuracy_first")
        _require(
            policy in ("accuracy_first", "accuracy_with_latency_budget"),
            "unsupported accuracy/cost selection policy",
        )
        if policy == "accuracy_with_latency_budget":
            _positive(rule["max_median_us_per_row"], "max_median_us_per_row")
            _text(rule["budget_rationale"], "latency budget rationale")
            _require(
                rule.get("latency_bound", "inclusive") in ("inclusive", "exclusive"),
                "latency_bound must be inclusive or exclusive",
            )
        else:
            _require(
                "max_median_us_per_row" not in rule and "latency_bound" not in rule,
                "latency ceiling requires accuracy_with_latency_budget policy",
            )
        _require(
            rule["incomplete_policy"] in ("refuse", "exclude_configuration"),
            "incomplete policy required",
        )
        timing = p["timing"]
        for name in ("batch_size", "repetitions", "threads"):
            _integer(timing[name], name)
        _integer(timing["warmup_calls"], "warmup_calls", 0)
        _require(
            timing["batch_size"] <= dataset["splits"]["validation"]["rows"],
            "timing batch exceeds validation cohort",
        )
        _require(timing["input_dtype"] in ("float32", "float64"), "timing dtype required")
        _require(timing["scope"] == "adapter_predict_call", "time the full declared adapter call")
        for name in ("hardware", "device", "included_operations", "excluded_operations"):
            _text(timing[name], name)
        _require(
            timing["synchronization"] in ("synchronous_cpu", "adapter"), "synchronization required"
        )
        _require(
            timing["device"] == "cpu" or timing["synchronization"] == "adapter",
            "accelerator requires explicit adapter synchronization",
        )
        tolerance = timing["golden_atol"]
        _require(len(tolerance) == len(coordinates), "per-coordinate golden tolerances required")
        for t in tolerance:
            _positive(t, "golden_atol", zero=True)
        _text(p["reproduction"]["adapter_path"], "adapter_path")
        _digest(p["reproduction"]["adapter_sha256"], "adapter_sha256")
        _text(p["reproduction"]["software"], "software environment")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"malformed benchmark protocol: {exc}") from exc
    return p


def load_protocol(path):
    return validate_protocol(read_json(path))


def check_approvals(protocol, paths, gates):
    """Require protocol-bound user attestations; never accept a generic approved flag."""
    records = [read_json(path) for path in paths]
    for gate in gates:
        matching = [
            r
            for r in records
            if r.get("schema") == "element_surrogate.approval.v1"
            and r.get("protocol_sha256") == protocol_identity(protocol)
            and r.get("gate") == gate
            and r.get("decision") == "approved"
            and isinstance(r.get("actor"), str)
            and r["actor"].strip()
            and isinstance(r.get("user_statement"), str)
            and r["user_statement"].strip()
        ]
        _require(len(matching) == 1, f"one user approval for {gate} and this protocol is required")
    return [{"path": str(Path(path).resolve()), "sha256": file_hash(path)} for path in paths]
