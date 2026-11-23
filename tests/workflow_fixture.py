"""Manufactured vacuum magnet; generated artifacts never ship."""

from pathlib import Path

import numpy as np

from element_surrogate.benchmark.protocol import protocol_identity, write_json
from element_surrogate.contracts import StaticMagneticCodec
from element_surrogate.data import SCHEMA
from element_surrogate.field_readiness import CATEGORIES, REVIEW_SCHEMA, assess, collect_evidence
from element_surrogate.fields import FieldMap
from element_surrogate.lorentz import straight_frame
from element_surrogate.region import Region
from element_surrogate.transport import Transport, qualify_transport


def manufactured_transport(dimension=4, *, longitudinal="path_m", gradient=3.0):
    a = np.linspace(-0.06, 0.06, 13)
    z = np.linspace(-0.1, 0.1, 21)
    x, y, _ = np.meshgrid(a, a, z, indexing="ij")
    b = np.stack([gradient * y, gradient * x, 0 * x], axis=-1)
    grid = FieldMap(a, a, z, b)
    refinements = [
        {"method": "fixed-rk4", "step_m": step, "plane_tolerance_m": 1e-12, "max_path_m": 0.3}
        for step in (0.02, 0.01, 0.005)
    ]
    transport = Transport(
        grid,
        straight_frame(-0.08),
        straight_frame(0.08),
        0.16,
        "a" * 64,
        StaticMagneticCodec(dimension, longitudinal),
        Region("circle", 0.04),
        Region("circle", 0.02),
        refinements[-1],
        actual_p0_kg_m_s=3e-19,
        mass_kg=1.5e-27,
    )
    ranges = [[-0.002, 0.002], [-0.003, 0.003], [-0.002, 0.002], [-0.003, 0.003]]
    if dimension == 6:
        ranges.append([-0.0001, 0.0001] if longitudinal != "time_s" else [-1e-10, 1e-10])
    if dimension > 4:
        ranges.append([-0.03, 0.03])
    settings = [
        {
            "id": name,
            "effective_rigidity_tm": rigidity,
            "field_scale": 1.0,
            "split": split,
            "surviving_rows": 32,
        }
        for name, rigidity, split in (
            ("fit_a", 1.8, "train"),
            ("fit_b", 2.4, "train"),
            ("validation", 2.0, "validation"),
            ("test", 2.2, "test"),
        )
    ]
    return transport, refinements, ranges, settings


def qualified_plan(dimension=4, *, continuous_coupling=False, **kwargs):
    transport, refinements, ranges, settings = manufactured_transport(dimension, **kwargs)
    limits = [1e-8] * dimension
    if dimension == 6 and transport.codec.longitudinal == "time_s":
        limits[4] = 1e-16
    values = [s["field_scale"] / s["effective_rigidity_tm"] for s in settings]
    interval = [min(values), max(values)] if continuous_coupling else None
    qualification = qualify_transport(
        transport,
        ranges,
        settings,
        refinements,
        random_probes=0,
        seed=17,
        component_limits=limits,
        path_limit_m=1e-9,
        effective_coupling_range=interval,
    )
    assert qualification["accepted"]
    assessment = readiness_for_test(transport)
    plan = {
        "schema": SCHEMA,
        "transport": transport.description(),
        "qualification_sha256": protocol_identity(qualification),
        "field_readiness": assessment,
        "field_readiness_sha256": protocol_identity(assessment),
        "ranges": ranges,
        "settings": settings,
        "seed": 102,
        "batch_size": 17,
        "max_proposals_per_setting": 100,
        "max_wall_seconds": 120.0,
        "effective_coupling_range": interval,
        "material_boundary_evidence": (
            "manufactured vacuum field; region is a test restriction, not a mechanical aperture"
        ),
    }
    return transport, qualification, plan


def readiness_for_test(transport):
    """Explicit manufactured interpretation, not a physical/user approval."""
    evidence = collect_evidence(transport, shells=2)
    review = {
        "schema": REVIEW_SCHEMA,
        "evidence_sha256": protocol_identity(evidence),
        "categories": {
            name: {
                "status": "reviewed",
                "evidence_refs": ["source_summary", "maxwell", "grid_integrity", "field_ends"],
                "rationale": (
                    "Manufactured analytic vacuum field in declared SI Cartesian coordinates. "
                    "Finite native planes define this test; "
                    "no field-free or mechanical-aperture claim."
                ),
            }
            for name in CATEGORIES
        },
        "issue_resolutions": {
            key: {
                "status": "explained_observation"
                if issue.get("issue_type") == "observation"
                else "accepted_numerical_limitation",
                "evidence_refs": ["source_summary"],
                "rationale": (
                    "Manufactured test construction: identical longitudinal planes are intentional "
                    "translational invariance; undefined zero-field/tail statistics "
                    "are not evidence of physical calibration."
                ),
            }
            for key, issue in evidence["issues"].items()
        },
    }
    return assess(evidence, review)


def approvals(directory, plan, gates=("description_target", "region_ranges", "compute_budget")):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for gate in gates:
        path = directory / f"approval_{gate}.json"
        write_json(
            path,
            {
                "schema": "element_surrogate.approval.v1",
                "protocol_sha256": protocol_identity(plan),
                "gate": gate,
                "decision": "approved",
                "actor": "manufactured-test-user",
                "user_statement": (
                    "Synthetic fixture decision; not an actual human or institutional approval."
                ),
            },
        )
        paths.append(str(path))
    return paths
