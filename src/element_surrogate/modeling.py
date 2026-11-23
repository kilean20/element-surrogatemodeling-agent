"""Connect qualified generated labels to the existing benchmark authority."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from element_surrogate import neural
from element_surrogate.benchmark.protocol import (
    SCHEMA,
    file_hash,
    protocol_identity,
    validate_protocol,
    write_json,
)
from element_surrogate.candidate_scope import validate_candidate_scope
from element_surrogate.contracts import StaticMagneticCodec, coordinate_metadata
from element_surrogate.data import export_benchmark_cohorts, verify_generation
from element_surrogate.field_readiness import SCHEMA as READINESS_SCHEMA


def prepare_generated_study(generation_dir, output_dir, request) -> dict:
    """Freeze explicit candidates, target and timing; never invent approval records.

    Sizes/seeds/budgets/ties come from the request. Neural symmetry/base choices
    need a rationale; a reflection group additionally pins its evidence. Current
    built-in adapters handle a static grid, its scalar amplitude and effective
    rigidity. A non-scaling family of solved fields needs a separate adapter.
    """
    record = verify_generation(generation_dir)
    request = deepcopy(request)
    for candidate in request["candidates"]:
        config = candidate["configuration"]
        if config.get("model_family", "mlp") in ("mlp", "reflection_features"):
            config.setdefault("output_normalization", "train_rms")
    assessment = record["plan"].get("field_readiness", {})
    if (
        assessment.get("schema") != READINESS_SCHEMA
        or assessment.get("generation_eligible") is not True
    ):
        raise ValueError("generated neural study requires the recorded field readiness assessment")
    if protocol_identity(assessment) != record["plan"].get("field_readiness_sha256"):
        raise ValueError("generated readiness assessment changed")
    output_dir = Path(output_dir).resolve()
    cohorts = export_benchmark_cohorts(
        generation_dir,
        output_dir / "cohorts",
        maximum_rows_per_split=request["maximum_rows_per_split"],
    )
    description = record["plan"]["transport"]
    coordinates = coordinate_metadata(StaticMagneticCodec(**description["codec"]))
    scales = request["error_scales"]
    if len(scales) != len(coordinates):
        raise ValueError("one declared accuracy scale per native coordinate required")
    for c, scale in zip(coordinates, scales, strict=True):
        c["error_scale"] = scale
    candidates = []
    for candidate in request["candidates"]:
        dimension = neural._configuration(candidate["configuration"])
        if dimension != len(coordinates):
            raise ValueError("candidate dimension differs from generated label coordinates")
        config = candidate["configuration"]
        validate_candidate_scope(config, description, record["plan"])
        candidates.append(
            {
                **candidate,
                "proposed_forward_description": candidate["forward_description"],
                "forward_description": neural.forward_description(candidate["configuration"]),
            }
        )
    p = {
        "schema": SCHEMA,
        "name": request["name"],
        "coordinates": coordinates,
        "coordinate_convention": description["coordinates"]
        + "; "
        + description["rigidity_interpretation"],
        "inputs": cohorts["metadata"]["input_names"],
        "input_units": cohorts["metadata"]["input_units"],
        "forbidden_inputs": ["targets", "status", "path_length_m", "survival_flag"],
        "physics": {
            "field_sha256": protocol_identity(description["field"]),
            "reference_sha256": description["reference_sha256"],
            "reconstruction_sha256": protocol_identity(description["field"]["reconstruction"]),
            "integration_qualification_sha256": record["plan"]["qualification_sha256"],
            "field_readiness_sha256": record["plan"]["field_readiness_sha256"],
            "region_sha256": protocol_identity(
                {"support": description["support"], "trajectory": description["trajectory_region"]}
            ),
            "native_planes": True,
            "qualification_scope": (
                "finite-cohort refinement and every-read-node support; material "
                "boundary is separate"
            ),
            "boundary_convention": "fixed declared native planes; no placement wrapping",
            "transport_description": description,
            "qualified_ranges": record["plan"]["ranges"],
            "qualified_settings": [
                {k: s[k] for k in ("effective_rigidity_tm", "field_scale")}
                for s in record["plan"]["settings"]
            ],
            "material_boundary_evidence": record["plan"]["material_boundary_evidence"],
            "effective_coupling_range": record["plan"].get("effective_coupling_range"),
        },
        "dataset": {
            "identity_sha256": cohorts["dataset_identity_sha256"],
            "access_and_reuse": request["access_and_reuse"],
            "splits": cohorts["splits"],
            "generation_record_path": str(Path(generation_dir).resolve() / "generation.json"),
        },
        "candidates": candidates,
        "selection": request["selection"],
        "timing": request["timing"],
        "resources": request["resources"],
        "reproduction": {
            "adapter_path": str(Path(neural.__file__).resolve()),
            "adapter_sha256": file_hash(neural.__file__),
            "software": request["software"],
        },
    }
    from element_surrogate.physics_review import review

    p["physics"]["physics_review_required"] = True
    if "physics_review" in request:
        p["physics"]["modeling_review"] = review(p, request["physics_review"], Path.cwd())
    validate_protocol(p)
    path = output_dir / "protocol.json"
    write_json(path, p, exclusive=True)
    return {
        "protocol_path": str(path),
        "protocol_sha256": protocol_identity(p),
        "cohort_rows": {s: r["rows"] for s, r in cohorts["splits"].items()},
        "test_arrays_opened": False,
        "requires_user_decisions": ["description_target", "region_ranges", "compute_budget"],
    }
