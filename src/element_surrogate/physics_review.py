"""Bind physics considerations to an actual finite candidate study.

Records expose skipped opportunities; they do not certify physical truth or
force a model family. Numerical eligibility still belongs to the tested tools.
"""

from copy import deepcopy
from pathlib import Path

from element_surrogate.benchmark.protocol import file_hash, protocol_identity, read_json, write_json

TOPICS = (
    "controls",
    "reference_and_coordinates",
    "symmetry",
    "theoretical_base",
    "limiting_cases",
    "canonical_structure",
    "scaling_and_metrics",
)
SCHEMA = "element_surrogate.physics_modeling_review.v1"


def require_before_fit(protocol):
    result = summarize(protocol)
    if protocol.get("physics", {}).get("physics_review_required") and result["status"] == "absent":
        raise ValueError(
            "record the physics modeling review before fitting; explicit "
            "reasoned deferrals are permitted and reported"
        )
    return result


def design_identity(protocol):
    material = {
        k: deepcopy(protocol[k])
        for k in (
            "physics",
            "coordinates",
            "inputs",
            "input_units",
            "dataset",
            "candidates",
            "selection",
            "timing",
        )
    }
    material["physics"].pop("modeling_review", None)
    return protocol_identity(material)


def summarize(protocol):
    record = protocol.get("physics", {}).get("modeling_review")
    if record is None:
        return {
            "status": "absent",
            "unresolved_topics": list(TOPICS),
            "scope": (
                "Declared candidate comparison only; physics opportunities have not "
                "been accounted for."
            ),
        }
    if record.get("schema") != SCHEMA or record.get("design_sha256") != design_identity(protocol):
        raise ValueError("physics review differs from this candidate/data/coordinate design")
    recomputed = review(protocol, {"topics": record["topics"]}, Path.cwd())
    if protocol_identity(recomputed) != protocol_identity(record):
        raise ValueError("physics-review coverage or dispositions were changed")
    for entry in record["topics"].values():
        for evidence in entry["evidence"]:
            if file_hash(evidence["path"]) != evidence["sha256"]:
                raise ValueError("physics-review source evidence changed")
    return {
        "status": record["coverage_status"],
        "unresolved_topics": record["unresolved_topics"],
        "scope": (
            "Recorded finite-study decisions, not complete physical validation "
            "or a global model optimum."
        ),
    }


def review(protocol, request, base_path):
    topics = request.get("topics", {})
    if set(topics) != set(TOPICS):
        raise ValueError("record every physics topic, including unknown or inapplicable ones")
    candidates = {c["id"]: c for c in protocol["candidates"]}
    output, unresolved = {}, []
    for topic in TOPICS:
        row = deepcopy(topics[topic])
        disposition = row.get("disposition")
        if disposition not in (
            "compared",
            "established_contract",
            "measured_not_supported",
            "not_applicable",
            "deferred",
        ):
            raise ValueError("unknown physics disposition: " + topic)
        if not isinstance(row.get("rationale"), str) or not row["rationale"].strip():
            raise ValueError("physics disposition requires its scientific rationale")
        ids = row.get("candidate_ids", [])
        if not isinstance(ids, list) or any(i not in candidates for i in ids):
            raise ValueError("physics review names an absent candidate")
        if disposition == "compared" and len(ids) < 2:
            raise ValueError("a physics comparison needs at least two actual candidates")
        if len(set(ids)) != len(ids):
            raise ValueError("a physics comparison requires distinct candidate IDs")
        evidence = []
        for item in row.get("evidence", []):
            path = Path(item["path"])
            path = (Path(base_path) / path).resolve() if not path.is_absolute() else path.resolve()
            if file_hash(path) != item["sha256"]:
                raise ValueError("physics evidence SHA256 mismatch")
            evidence.append({"path": str(path), "sha256": item["sha256"]})
        if (
            disposition in ("established_contract", "measured_not_supported", "not_applicable")
            and not evidence
        ):
            raise ValueError("an established/measured disposition requires bound evidence")
        if topic == "symmetry":
            if disposition == "established_contract":
                raise ValueError(
                    "symmetry needs a measured comparison/disposition, not parity alone"
                )
            structured = [
                candidates[i]["configuration"]
                for i in ids
                if len(candidates[i]["configuration"].get("reflection_group", [])) > 1
            ]
            if disposition == "compared":
                plain = [
                    i
                    for i in ids
                    if len(candidates[i]["configuration"].get("reflection_group", [])) < 2
                ]
                if not structured or not plain:
                    raise ValueError(
                        "symmetry comparison needs structured and unconstrained controls"
                    )
                for config in structured:
                    from element_surrogate.neural import _configuration

                    _configuration(config)
            if disposition == "measured_not_supported":
                maps = [
                    read_json(e["path"])
                    for e in evidence
                    if Path(e["path"]).suffix.lower() == ".json"
                ]
                maps = [m for m in maps if m.get("schema") == "element_surrogate.map_symmetry.v1"]
                if not maps or any(m.get("supported") is not False for m in maps):
                    raise ValueError("field parity alone cannot reject map-level symmetry")
                from element_surrogate.symmetry import validate_measurement

                for measured in maps:
                    validate_measurement(
                        measured,
                        measured["group"],
                        len(protocol["coordinates"]),
                        require_current=True,
                    )
                transport = protocol["physics"].get("transport_description")
                if transport is not None and any(
                    m["scope"]["transport"] != transport for m in maps
                ):
                    raise ValueError("rejected symmetry evidence belongs to another transport")
        if topic == "theoretical_base" and disposition == "compared":
            if not any(
                candidates[i]["configuration"].get("base", "").startswith("theory_") for i in ids
            ):
                raise ValueError("theoretical-base comparison has no theoretical-base candidate")
        if topic == "canonical_structure" and disposition == "compared":
            if not any(
                candidates[i]["configuration"].get("model_family") == "canonical_shear" for i in ids
            ):
                raise ValueError("canonical comparison has no canonical candidate")
        if disposition == "deferred":
            unresolved.append(topic)
        row["evidence"] = evidence
        row["candidate_ids"] = ids
        output[topic] = row
    record = {
        "schema": SCHEMA,
        "design_sha256": design_identity(protocol),
        "topics": output,
        "coverage_status": "incomplete" if unresolved else "recorded",
        "unresolved_topics": unresolved,
        "scope": (
            "Explicit evidence and finite-study dispositions. Omission reasons "
            "are recorded claims, not independently authenticated scientific "
            "truth."
        ),
    }
    return record


def from_files(protocol_path, request_path, output_dir):
    from element_surrogate.benchmark.protocol import load_protocol, validate_protocol

    p = load_protocol(protocol_path)
    for item in p["dataset"]["splits"].values():
        item["path"] = str((Path(protocol_path).resolve().parent / item["path"]).resolve())
    p["physics"]["modeling_review"] = review(
        p, read_json(request_path), Path(request_path).resolve().parent
    )
    validate_protocol(p)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "physics_review.json", p["physics"]["modeling_review"], exclusive=True)
    write_json(output / "protocol.json", p, exclusive=True)
    return {
        "protocol_path": str(output / "protocol.json"),
        "protocol_sha256": protocol_identity(p),
        "review": summarize(p),
        "user_approvals_created": False,
    }
