"""The synthetic corpus: clean fields, fields with planted defects, and a runner that applies
every flagging check to every case.

Each case is a field with exact vacuum values (``synthetic``) and the description a user
would give of it. Defects are planted either in the field (masked nodes, a duplicated plane,
a noisy outer shell, a wrong storage order, gauss read as tesla, a truncated map) or in the
description (a wrong current label, a reversed polarity, a wrong magnet kind, no aperture),
or in a second export of the same field (opposite polarity, another excitation).
The runner reports, for every case, which checks flagged it: each planted defect must be
flagged by the check meant for it, and any finding on a clean field is a false alarm.

The numbers here (source positions, strengths, grids, tolerances) are test inputs, not
properties of any magnet.

Run ``python -m element_surrogate.corpus`` for the report as JSON.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from element_surrogate import readiness
from element_surrogate.analysis import field_frame, harmonics
from element_surrogate.bend import NominalBend
from element_surrogate.description import compare_description
from element_surrogate.fields import FieldMap
from element_surrogate.region import Region
from element_surrogate.synthetic import (
    bend_sources,
    multipole_sources,
    plant_duplicated_plane,
    plant_gauss_as_tesla,
    plant_noisy_shell,
    plant_storage_mismatch,
    plant_truncation,
    plant_zeroed_nodes,
    point_source_field,
)

TOLERANCE = 1.0e-3  # the stated uncertainty of the clean descriptions
BEND_TOLERANCE = 2.0e-2  # a band centre is not the pole arc; see test_analysis


@dataclass(frozen=True)
class Case:
    name: str
    field: FieldMap
    description: dict[str, Any]
    planted: str | None = None
    expected: tuple[str, ...] = ()
    where: dict[str, Any] = dataclasses.field(default_factory=dict)
    other: FieldMap | None = None  # a second export of the same field, if there is one
    expected_observations: tuple[str, ...] = ()


Check = Callable[[Case], list[dict[str, Any]]]

CHECKS: dict[str, Check] = {
    "node_defects": lambda case: readiness.node_defects(case.field)["findings"],
    "harmonic_nodes": lambda case: readiness.harmonic_nodes(case.field)["findings"],
    "storage_order": lambda case: readiness.storage_order(case.field)["findings"],
    "field_ends": lambda case: readiness.field_ends(
        case.field, tolerance=case.description.get("tolerance")
    )["findings"],
    "description": lambda case: compare_description(case.field, case.description)["findings"],
    "compare_exports": lambda case: (
        []
        if case.other is None
        else readiness.compare_exports(
            case.field, case.other, tolerance=case.description.get("tolerance")
        )["findings"]
    ),
}


def clean_quadrupole(nodes_xy: int = 17, nodes_z: int = 481) -> FieldMap:
    """The clean quadrupole; other node counts give another export of the same field."""
    axis = np.linspace(-0.02, 0.02, nodes_xy)
    z = np.linspace(-0.6, 0.6, nodes_z)
    return point_source_field(multipole_sources(2, 0.04, 0.2, 2.0e-4, 81), axis, axis, z)


def clean_bend() -> FieldMap:
    bend = NominalBend(2.0, 0.3, 0.0)
    sources = bend_sources(bend, 0.02, 0.04, 1.0e-3, 160, 13)
    x = np.linspace(-0.12, 0.12, 61)
    y = np.linspace(-0.008, 0.008, 9)
    z = np.linspace(-0.6, 0.6, 241)
    return point_source_field(sources, x, y, z)


def build_corpus() -> list[Case]:
    """Two clean magnets and one case per planted defect."""
    quad = clean_quadrupole()
    region = Region.covering(quad)
    fit = harmonics(quad, region, region.radius_m)
    gradient = fit["integrated_t_m"][1]["normal"] / region.radius_m
    quad_text = {
        "kind": "quadrupole",
        "integrated_strength": gradient,
        "aperture": {"radius_m": 0.02},
        "tolerance": TOLERANCE,
    }
    bend = clean_bend()
    proposal = field_frame(bend)["proposed_bend"]
    bend_text = {
        "kind": "dipole",
        "bend": {"radius_m": 2.0, "angle_rad": 0.3, "bends_toward": "-x"},
        "integrated_strength": proposal["integral_along_centre_t_m"],
        "aperture": {"half_width_m": 0.04, "half_height_m": 0.008},
        "tolerance": BEND_TOLERANCE,
    }
    noise = 1.0e-3 * float(np.abs(quad.b_t).max())
    planted = [
        (plant_zeroed_nodes(quad, [(0, 0, 240), (1, 0, 240), (8, 8, 120)]), ()),
        (plant_duplicated_plane(quad, 2, 280), ("harmonic_nodes",)),
        (plant_noisy_shell(quad, 0.015, noise, seed=1), ("harmonic_nodes",)),
        (plant_storage_mismatch(quad, nesting=(2, 1, 0)), ("storage_order",)),
        (plant_gauss_as_tesla(quad), ("description",)),
        (plant_truncation(quad, (-0.12, 0.12)), ("field_ends",)),
    ]
    export = clean_quadrupole(9, 241)  # the same field on a coarser grid
    cases = [
        Case("clean quadrupole", quad, quad_text, other=export),
        Case("clean bend", bend, bend_text),
    ]
    cases += [
        Case(
            p.kind,
            p.field,
            quad_text,
            p.kind,
            expected,
            p.where,
            expected_observations=("node_defects",) if p.kind == "zeroed_nodes" else (),
        )
        for p, expected in planted
    ]
    without_strength = {k: v for k, v in quad_text.items() if k != "integrated_strength"}
    described = [
        (
            "wrong_current_label",
            {**without_strength, "current_a": 125.0, "strength_per_ampere": gradient / 100.0},
        ),
        ("reversed_polarity", {**quad_text, "integrated_strength": -gradient}),
        ("wrong_kind", {**quad_text, "kind": "sextupole", "integrated_strength": None}),
        ("aperture_not_described", {k: v for k, v in quad_text.items() if k != "aperture"}),
    ]
    cases += [Case(name, quad, text, name, ("description",)) for name, text in described]
    exports = [
        ("export_polarity_reversed", -1.0),
        ("export_at_another_excitation", 0.95),
    ]
    cases += [
        Case(
            name,
            quad,
            quad_text,
            name,
            ("compare_exports",),
            {"scale": s},
            export.with_values(s * export.b_t),
        )
        for name, s in exports
    ]
    return cases


def run_corpus(cases: list[Case] | None = None) -> dict[str, Any]:
    """Apply every check to every case. A check that cannot run on a case reports its error
    instead of findings."""
    cases = build_corpus() if cases is None else cases
    rows: list[dict[str, Any]] = []
    for case in cases:
        checks: dict[str, Any] = {}
        for name, check in CHECKS.items():
            try:
                checks[name] = {"findings": [f["kind"] for f in check(case)]}
            except ValueError as error:
                checks[name] = {"error": str(error)}
        flagged_by = [name for name, result in checks.items() if result.get("findings")]
        observations = readiness.node_defects(case.field)["observations"]
        observed_by = ["node_defects"] if observations else []
        rows.append(
            {
                "case": case.name,
                "planted": case.planted,
                "where": case.where,
                "expected": list(case.expected),
                "flagged_by": flagged_by,
                "observed_by": observed_by,
                "observations": observations,
                "expected_observations": list(case.expected_observations),
                "caught_by_expected": bool(
                    set(case.expected) & set(flagged_by)
                    or set(case.expected_observations) & set(observed_by)
                ),
                "checks": checks,
            }
        )
    planted = [r for r in rows if r["planted"]]
    return {
        "checks": list(CHECKS),
        "cases": rows,
        "planted": len(planted),
        "missed": [r["case"] for r in planted if not r["caught_by_expected"]],
        "false_alarms": {
            r["case"]: r["flagged_by"] for r in rows if not r["planted"] and r["flagged_by"]
        },
        "errors": {
            r["case"]: [n for n, c in r["checks"].items() if "error" in c]
            for r in rows
            if any("error" in c for c in r["checks"].values())
        },
        "tracking_support": run_tracking_support_corpus(),
        "exact_vacuum_nodes": run_exact_vacuum_node_controls(),
    }


def run_exact_vacuum_node_controls():
    axis = np.linspace(-0.01, 0.01, 9)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    zero = np.zeros_like(x)
    rows = []
    for name, values in (
        ("zero field", np.stack([zero, zero, zero], -1)),
        ("uniform transverse field", np.stack([zero + 1, zero, zero], -1)),
        ("longitudinally invariant quadrupole", np.stack([3 * y, 3 * x, zero], -1)),
    ):
        grid = FieldMap(axis, axis, axis, values)
        nodes = readiness.node_defects(grid)
        harmonic = readiness.harmonic_nodes(grid)
        residual = readiness.cell_residuals(grid)
        rows.append(
            {
                "case": name,
                "node_findings": nodes["findings"],
                "harmonic_findings": harmonic["findings"],
                "contextual_observations": len(nodes["observations"]),
                "max_curl_t_per_m": float(np.max(residual.curl_t_per_m)),
                "max_abs_divergence_t_per_m": float(np.max(np.abs(residual.divergence_t_per_m))),
            }
        )
    return rows


def run_tracking_support_corpus():
    """Clean and planted outside-stencil queries in an exact vacuum zero field."""
    from element_surrogate.domain import probe_domain
    from element_surrogate.lorentz import straight_frame

    axis = np.linspace(-0.04, 0.04, 9)
    z = np.linspace(-0.1, 0.1, 21)
    grid = FieldMap(axis, axis, z, np.zeros((9, 9, 21, 3)))
    rows = []
    for name, x, expected in (("supported centre", 0.0, False), ("outside stencil", 0.024, True)):
        report = probe_domain(
            grid,
            np.array([[x, 0, 0, 0]]),
            2.0,
            straight_frame(-0.08),
            straight_frame(0.08),
            Region("circle", 0.027),
            Region("circle", 0.026),
            {"method": "fixed-rk4", "step_m": 0.01},
            perturbation_t=0.01,
        )
        flagged = bool(report["findings"])
        rows.append(
            {
                "case": name,
                "planted": expected,
                "flagged": flagged,
                "correct": flagged == expected,
                "perturbation_unchanged": report["probe_results"][0][
                    "perturbation_exactly_unchanged"
                ],
            }
        )
    return {
        "cases": rows,
        "missed": [r["case"] for r in rows if r["planted"] and not r["flagged"]],
        "false_alarms": [r["case"] for r in rows if not r["planted"] and r["flagged"]],
    }


if __name__ == "__main__":
    print(json.dumps(run_corpus(), indent=2))
