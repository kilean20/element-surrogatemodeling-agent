"""File-based scientific workflow entry points; arrays remain inside tools.

This is the deterministic core used by a client/skill, not a second agent loop.
User decisions remain external attestations. A saved reference is loaded and
verified, never silently fitted for a new particle rigidity or magnet setting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from element_surrogate.benchmark.protocol import file_hash, read_json, write_json
from element_surrogate.contracts import StaticMagneticCodec
from element_surrogate.data import generate, validate_plan
from element_surrogate.fields import load_field_file
from element_surrogate.readings import reconstruct
from element_surrogate.reference import ReferenceOrbit
from element_surrogate.region import Region
from element_surrogate.transport import Transport, frame_from_dict, qualify_transport


def region_from_dict(value):
    """The common Region type, without inventing a second domain representation."""
    return Region.from_request(value)


def load_transport(path):
    """Load exact source/reference identities and declared native plane convention."""
    path = Path(path).resolve()
    spec = read_json(path)
    if (
        spec.get("schema") != "element_surrogate.transport.v1"
        or spec.get("native_planes") is not True
    ):
        raise ValueError("native-plane transport specification required")
    # Geometry errors must fail before hashing/loading a potentially multi-GB field.
    support = region_from_dict(spec["support"])
    trajectory_region = region_from_dict(spec["trajectory_region"])
    source = spec["field"]
    field_path = (path.parent / source["path"]).resolve()
    if file_hash(field_path) != source["sha256"]:
        raise ValueError("field source changed")
    grid = load_field_file(field_path, source.get("format")).slot(source.get("slot"))
    method = spec.get("reconstruction", "total-degree-two")
    reference = spec["reference"]
    reference_path = (path.parent / reference["path"]).resolve()
    if file_hash(reference_path) != reference["sha256"]:
        raise ValueError("saved reference changed")
    orbit = ReferenceOrbit.load(
        reference_path,
        reconstruct(grid, method),
        expected_rigidity_tm=reference["rigidity_tm"],
        expected_field_scale=reference["field_scale"],
    )
    if spec["boundary_convention"] == "traced_reference":
        entry, exit = orbit.frame_at(0), orbit.frame_at(orbit.length_m)
    elif spec["boundary_convention"] == "declared_native":
        entry, exit = frame_from_dict(spec["entry_frame"]), frame_from_dict(spec["exit_frame"])
        # The saved reference must actually cross those same planes. Its output
        # can have a residual in their geometric frame; do not subtract it silently.
        original_entry = frame_from_dict(orbit.metadata["entry_frame"])
        original_exit = frame_from_dict(orbit.metadata["nominal_exit_frame"])
        for a, b in ((entry, original_entry), (exit, original_exit)):
            if any(
                not np.array_equal(getattr(a, k), getattr(b, k))
                for k in ("origin_m", "tangent", "horizontal", "vertical")
            ):
                raise ValueError("declared planes differ from the saved reference tracing planes")
    else:
        raise ValueError("choose traced_reference or declared_native boundary convention")
    codec = StaticMagneticCodec(**spec["codec"])
    transport = Transport(
        grid,
        entry,
        exit,
        orbit.length_m,
        reference["sha256"],
        codec,
        support,
        trajectory_region,
        spec["integration"],
        method,
        spec.get("actual_p0_kg_m_s"),
        spec.get("mass_kg"),
    )
    return transport, spec


def qualify_from_files(transport_path, request_path, output_path) -> dict:
    transport, _ = load_transport(transport_path)
    request = read_json(request_path)
    result = qualify_transport(transport, **request)
    if Path(output_path).exists():
        if read_json(output_path) != result:
            raise ValueError("qualification already exists with different evidence")
    else:
        write_json(output_path, result, exclusive=True)
    return {
        "qualification_path": str(Path(output_path).resolve()),
        "accepted": result["accepted"],
        "scope_sha256": result["scope_sha256"],
        "probe_count_per_setting": result["probe_count_per_setting"],
        "settings": [
            {
                "setting_id": r["setting_id"],
                "accepted": r["accepted"],
                "common_survivors": r["common_survivors"],
                "status_changes": r["status_changes"],
            }
            for r in result["reports"]
        ],
    }


def generate_from_files(
    transport_path,
    qualification_path,
    plan_path,
    output_dir,
    approval_paths,
    *,
    max_batches=None,
    execution_check=None,
) -> dict:
    transport, _ = load_transport(transport_path)
    qualification, plan = read_json(qualification_path), read_json(plan_path)
    return generate(
        transport,
        qualification,
        plan,
        output_dir,
        approval_paths,
        max_batches=max_batches,
        execution_check=execution_check,
    )


def validate_generation_files(transport_path, qualification_path, plan_path) -> dict:
    transport, _ = load_transport(transport_path)
    qualification, plan = read_json(qualification_path), read_json(plan_path)
    validate_plan(transport, qualification, plan)
    return {
        "valid": True,
        "dimension": transport.codec.dimension,
        "total_requested_survivors": sum(s["surviving_rows"] for s in plan["settings"]),
        "settings": len(plan["settings"]),
        "batch_size": plan["batch_size"],
        "material_boundary_evidence": plan["material_boundary_evidence"],
    }
