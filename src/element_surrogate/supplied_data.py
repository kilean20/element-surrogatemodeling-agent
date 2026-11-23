"""Prepare native neural studies from existing, explicitly described labels.

No labels, qualifications or user approvals are generated. The ordinary
benchmark remains responsible for split checks, fitting and locked evaluation.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np

from element_surrogate.benchmark.protocol import (
    file_hash,
    protocol_identity,
    read_json,
    validate_protocol,
    write_json,
)

SCHEMA = "element_surrogate.supplied_neural_study.v1"


def prepare(request_path, output_dir):
    from element_surrogate import neural
    from element_surrogate.benchmark.scoring import load_cohort
    from element_surrogate.cohort_io import require_native_metadata

    request_path = Path(request_path).resolve()
    request = read_json(request_path)
    if request.get("schema") != SCHEMA:
        raise ValueError("unsupported supplied-data neural study request")
    p = deepcopy(request["protocol"])
    p["physics"].pop("modeling_review", None)
    p["physics"]["physics_review_required"] = True
    provenance = deepcopy(request["label_provenance"])
    if not isinstance(provenance.get("description"), str) or not provenance["description"].strip():
        raise ValueError("describe the supplied label origin and limitations")
    history = provenance.get("split_history", {})
    if set(history) != {"train", "validation", "test"} or any(
        v not in ("unused", "previously_used", "unknown") for v in history.values()
    ):
        raise ValueError("explicit prior-use history required for each supplied split")
    if provenance.get("qualification_status") != "source_declared":
        raise ValueError(
            "this importer records source_declared label provenance, not a new "
            "physics qualification"
        )
    if "generation_record_path" in p["dataset"]:
        raise ValueError("supplied labels must not masquerade as locally generated rows")
    d = len(p["coordinates"])
    metadata = {
        "coordinate_names": [v["name"] for v in p["coordinates"]],
        "coordinate_units": [v["unit"] for v in p["coordinates"]],
        "input_names": p["inputs"],
        "input_units": p["input_units"],
    }
    require_native_metadata(metadata, d)
    for split, item in p["dataset"]["splits"].items():
        path = Path(item["path"])
        path = (request_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
        if file_hash(path) != item["sha256"]:
            raise ValueError("supplied split SHA256 mismatch: " + split)
        item["path"] = str(path)
    for candidate in p["candidates"]:
        config = candidate["configuration"]
        if config.get("model_family", "mlp") in ("mlp", "reflection_features"):
            config.setdefault("output_normalization", "train_rms")
        if neural._configuration(config) != d:
            raise ValueError("candidate differs from supplied label dimension")
        from element_surrogate.candidate_scope import validate_candidate_scope

        validate_candidate_scope(
            config, p["physics"].get("transport_description"), request.get("source_contract")
        )
        candidate["proposed_forward_description"] = candidate["forward_description"]
        candidate["forward_description"] = neural.forward_description(config)
    p["dataset"]["supplied_label_provenance"] = provenance
    if "source_contract" in request:
        p["dataset"]["supplied_source_contract"] = deepcopy(request["source_contract"])
    p["dataset"]["supplied_import_request"] = {
        "path": str(request_path),
        "sha256": file_hash(request_path),
    }
    p["reproduction"]["adapter_path"] = str(Path(neural.__file__).resolve())
    p["reproduction"]["adapter_sha256"] = file_hash(neural.__file__)
    if "physics_review" in request:
        from element_surrogate.physics_review import review

        p["physics"]["modeling_review"] = review(p, request["physics_review"], request_path.parent)
    validate_protocol(p)
    cohorts = {
        split: load_cohort(p, request_path.parent, split) for split in ("train", "validation")
    }
    if set(cohorts["train"]["sample_id"]) & set(cohorts["validation"]["sample_id"]):
        raise ValueError("supplied training and validation IDs overlap")
    if p["dataset"].get("split_unit", "setting") == "setting":
        if set(cohorts["train"]["setting_id"]) & set(cohorts["validation"]["setting_id"]):
            raise ValueError("supplied setting holdout overlaps")
    for split, cohort in cohorts.items():
        if d > 4 and not np.array_equal(cohort["inputs"][:, d - 1], cohort["targets"][:, d - 1]):
            raise ValueError("supplied static-magnetic labels do not conserve momentum")
        for candidate in p["candidates"]:
            neural._configuration(candidate["configuration"], cohort["inputs"])
        item = p["dataset"]["splits"][split]
        if file_hash(item["path"]) != item["sha256"]:
            raise ValueError("supplied data changed during verification")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", p, exclusive=True)
    receipt = {
        "schema": "element_surrogate.supplied_data_import.v1",
        "request_path": str(request_path),
        "request_sha256": file_hash(request_path),
        "protocol_sha256": protocol_identity(p),
        "verified_rows": {k: len(v["sample_id"]) for k, v in cohorts.items()},
        "test_arrays_read": False,
        "label_generation_performed": False,
        "physics_qualification_performed": False,
        "label_provenance": provenance,
    }
    write_json(output / "supplied_data_import.json", receipt, exclusive=True)
    from element_surrogate.physics_review import summarize

    return {
        "protocol_path": str(output / "protocol.json"),
        "protocol_sha256": protocol_identity(p),
        "import_receipt_path": str(output / "supplied_data_import.json"),
        "rows": receipt["verified_rows"],
        "test_arrays_read": False,
        "new_training_labels_generated": False,
        "physics_review": summarize(p),
    }


def check_reference(transport, protocol, base, request):
    """Compare a declared sample of existing train/validation labels with tracking."""
    from element_surrogate.benchmark.scoring import load_cohort
    from element_surrogate.contracts import coordinate_metadata
    from element_surrogate.transport import verify_transport_source

    split, count, seed = request["split"], request["sample_rows"], request["seed"]
    if (
        split not in ("train", "validation")
        or type(count) is not int
        or count < 1
        or type(seed) is not int
        or seed < 0
    ):
        raise ValueError("explicit train/validation sample count and seed required")
    d = transport.codec.dimension
    expected = coordinate_metadata(transport.codec)
    if [(v["name"], v["unit"]) for v in expected] != [
        (v["name"], v["unit"]) for v in protocol["coordinates"]
    ]:
        raise ValueError("supplied coordinates differ from reference transport")
    if protocol["physics"].get("transport_description") != transport.description():
        raise ValueError("reference transport differs from supplied study declaration")
    limits = np.asarray(request["component_limits"], float)
    if limits.shape != (d,) or not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("positive finite native component comparison limits required")
    cohort = load_cohort(protocol, base, split)
    if count > len(cohort["inputs"]):
        raise ValueError("requested comparison sample exceeds supplied cohort")
    chosen = np.sort(
        np.random.default_rng(seed).choice(len(cohort["inputs"]), count, replace=False)
    )
    x, y = cohort["inputs"][chosen], cohort["targets"][chosen]
    actual = np.empty_like(y)
    status_changes = 0
    controls, inverse = np.unique(x[:, -2:], axis=0, return_inverse=True)
    verify_transport_source(transport)
    for i, (rigidity, scale) in enumerate(controls):
        which = inverse == i
        values, result, _ = transport.run(x[which, :d], float(rigidity), float(scale))
        actual[which] = values
        status_changes += int(np.count_nonzero(~result.reached_exit))
    verify_transport_source(transport)
    difference = np.abs(actual - y)
    maximum = np.max(difference, axis=0) if np.isfinite(difference).all() else None
    return {
        "schema": "element_surrogate.supplied_label_reference.v1",
        "dataset_identity_sha256": protocol["dataset"]["identity_sha256"],
        "split": split,
        "cohort_sha256": protocol["dataset"]["splits"][split]["sha256"],
        "transport": transport.description(),
        "sample_rows": count,
        "sample_ids": cohort["sample_id"][chosen].tolist(),
        "seed": seed,
        "sampled_control_count": len(controls),
        "component_limits": limits.tolist(),
        "max_component_difference": None if maximum is None else maximum.tolist(),
        "unsupported_rows": status_changes,
        "within_limits": bool(
            maximum is not None and status_changes == 0 and np.all(maximum <= limits)
        ),
        "new_training_labels_generated": False,
        "scope": (
            "Finite diagnostic comparison of supplied rows, not complete label "
            "certification, field accuracy or a new training dataset."
        ),
    }


def check_reference_files(transport_path, protocol_path, request_path, output_path):
    from element_surrogate.benchmark.protocol import load_protocol
    from element_surrogate.workflow import load_transport

    transport, _ = load_transport(transport_path)
    result = check_reference(
        transport,
        load_protocol(protocol_path),
        Path(protocol_path).resolve().parent,
        read_json(request_path),
    )
    write_json(output_path, result, exclusive=True)
    return {
        "output_path": str(Path(output_path).resolve()),
        "sha256": file_hash(output_path),
        **{
            k: result[k]
            for k in (
                "sample_rows",
                "sampled_control_count",
                "max_component_difference",
                "unsupported_rows",
                "within_limits",
                "new_training_labels_generated",
                "scope",
            )
        },
    }


def card_source(protocol, transport):
    """An explicit provided-label interpretation view, never a generation ledger."""
    declaration = protocol["dataset"].get("supplied_source_contract")
    if not isinstance(declaration, dict) or declaration.get("transport") != transport.description():
        raise ValueError("provided-label source contract for exactly this transport required")
    bounds = np.asarray(declaration.get("ranges"), float)
    if (
        bounds.shape != (transport.codec.dimension, 2)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] > bounds[:, 1])
    ):
        raise ValueError("provided-label native range contract required for cards")
    settings = declaration.get("settings", [])
    if not settings or any(
        not np.isfinite([s["effective_rigidity_tm"], s["field_scale"]]).all() for s in settings
    ):
        raise ValueError("provided-label declared controls required for cards")
    imported = protocol["dataset"].get("supplied_import_request", {})
    if file_hash(imported["path"]) != imported["sha256"]:
        raise ValueError("provided-label import declaration changed")
    return {
        "schema": "element_surrogate.supplied_label_card_source.v1",
        "status": "supplied_snapshot",
        "plan": deepcopy(declaration),
        "plan_sha256": protocol["dataset"]["identity_sha256"],
        "source_record_path": imported["path"],
        "source_record_sha256": imported["sha256"],
        "label_provenance": protocol["dataset"]["supplied_label_provenance"],
        "locally_generated": False,
    }
