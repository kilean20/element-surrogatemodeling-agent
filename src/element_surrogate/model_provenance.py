"""Geometry/source-field facts for model cards, independent of host skills."""

import math

import numpy as np


def _finite(value):
    try:
        return (
            not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        )
    except OverflowError:
        return False


def input_findings(model):
    """Semantic checks after the local request schema has checked shapes."""
    facts = model.get("physical_provenance") or {}
    design = facts.get("design_length") or {}
    longitudinal = facts.get("longitudinal_span") or {}
    front_length = model.get("frontmatter", {}).get("design_length_m")
    front_span = model.get("frontmatter", {}).get("model_longitudinal_span_m")
    findings = []
    for path, value, positive in [
        ("model.frontmatter.design_length_m", front_length, True),
        ("model.frontmatter.model_longitudinal_span_m", front_span, False),
        (
            "model.frontmatter.model_length_m",
            model.get("frontmatter", {}).get("model_length_m"),
            False,
        ),
        ("model.physical_provenance.longitudinal_span.input_m", longitudinal.get("input_m"), False),
        (
            "model.physical_provenance.longitudinal_span.output_m",
            longitudinal.get("output_m"),
            False,
        ),
        ("model.physical_provenance.design_length.value_m", design.get("value_m"), True),
        *[
            (
                f"model.physical_provenance.field_calculation_currents.{i}.current_a",
                row["current_a"],
                False,
            )
            for i, row in enumerate(facts.get("field_calculation_currents") or [])
        ],
    ]:
        if value is not None and (not _finite(value) or (positive and value <= 0)):
            findings.append(
                {
                    "path": path,
                    "message": "finite numeric value required"
                    + (" and length must be positive" if positive else ""),
                }
            )
    if (
        front_length is not None
        and design.get("value_m") is not None
        and front_length != design["value_m"]
    ):
        findings.append(
            {
                "path": "model.physical_provenance.design_length",
                "message": "design length conflicts with model.frontmatter.design_length_m",
            }
        )
    for length_key in ("model_length_m", "model_longitudinal_span_m"):
        front_span = model.get("frontmatter", {}).get(length_key)
        if front_span is None:
            continue
        if not longitudinal:
            findings.append(
                {
                    "path": "model.frontmatter." + length_key,
                    "message": "requires referenced physical_provenance.longitudinal_span stations",
                }
            )
        elif all(
            _finite(value)
            for value in (front_span, longitudinal.get("input_m"), longitudinal.get("output_m"))
        ):
            actual = abs(longitudinal["output_m"] - longitudinal["input_m"])
            if not math.isclose(front_span, actual, rel_tol=1e-12, abs_tol=1e-15):
                findings.append(
                    {
                        "path": "model.frontmatter." + length_key,
                        "message": "conflicts with the supplied longitudinal input/output stations",
                    }
                )
    return findings


def boundary_geometry(transport):
    """Distance between declared frame origins, without assuming a curved path."""
    frames = [getattr(transport, name, None) for name in ("entry", "exit")]
    if any(frame is None for frame in frames):
        return None
    positions = np.asarray([frame.origin_m for frame in frames], dtype=float)
    if positions.shape != (2, 3) or not np.isfinite(positions).all():
        raise ValueError("finite native input/output frame origins required")
    distance = float(np.linalg.norm(positions[1] - positions[0]))
    if not np.isfinite(distance):
        raise ValueError("finite native input/output distance required")
    return {
        "input_origin_m": positions[0].tolist(),
        "output_origin_m": positions[1].tolist(),
        "input_output_separation_m": distance,
        "definition": (
            "Euclidean distance between declared native input/output frame origins; "
            "in a bend this is a chord, not a longitudinal path coordinate or traced arc length"
        ),
    }


def collect(transport, model):
    """Inspect only one-dimensional axes; never materialize field-vector arrays."""
    failures = input_findings(model)
    if failures:
        raise ValueError("; ".join(f["path"] + ": " + f["message"] for f in failures))
    facts = model.get("physical_provenance") or {}
    design = facts.get("design_length") or {
        "value_m": model.get("frontmatter", {}).get("design_length_m"),
        "definition": None,
        "source_reference": None,
    }
    grid = transport.grid
    boundary = boundary_geometry(transport)
    longitudinal = facts.get("longitudinal_span")
    if longitudinal:
        longitudinal = {
            **longitudinal,
            "value_m": abs(longitudinal["output_m"] - longitudinal["input_m"]),
        }
        if not _finite(longitudinal["value_m"]):
            raise ValueError("finite longitudinal input/output span required")
    supplied_length = model.get("frontmatter", {}).get("model_length_m")
    if supplied_length is not None and (
        longitudinal is None
        or not math.isclose(supplied_length, longitudinal["value_m"], rel_tol=1e-12, abs_tol=1e-15)
    ):
        raise ValueError(
            "model.frontmatter.model_length_m conflicts with longitudinal input/output stations"
        )
    spacing, extent, nodes = {}, {}, []
    for name in "xyz":
        axis = np.asarray(getattr(grid, name + "_m"))
        steps = np.diff(axis)
        nodes.append(int(axis.size))
        extent[name] = [float(axis[0]), float(axis[-1])]
        spacing[name] = {"min": float(steps.min()), "max": float(steps.max())}
    missing = []
    if boundary is None:
        missing.append("native_input_output_boundary_origins")
    if longitudinal is None:
        missing.append("longitudinal_span")
    if design.get("value_m") is None:
        missing.append("design_length")
    for key in ("definition", "source_reference"):
        if not design.get(key):
            missing.append("design_length." + key)
    if not facts.get("solver_mesh"):
        missing.append("solver_mesh")
    if not facts.get("field_calculation_currents"):
        missing.append("field_calculation_currents")
    return {
        "schema": "element_surrogate.model_physical_provenance.v1",
        "design_length": design,
        "native_boundary_geometry": boundary,
        "model_length_m": longitudinal["value_m"] if longitudinal else None,
        "longitudinal_span": longitudinal,
        "traced_reference_length_m": float(transport.reference_path_m),
        "field_source": grid.source.as_dict() if grid.source is not None else None,
        "export_grid": {
            "coordinate_frame": "source Cartesian grid axes, converted to SI",
            "nodes_xyz": nodes,
            "extent_m": extent,
            "spacing_m": spacing,
        },
        "solver_mesh": facts.get("solver_mesh"),
        "field_calculation_currents": facts.get("field_calculation_currents"),
        "missing_metadata": missing,
        "fact_scope": (
            "Axis facts are computed; design/solver/current facts are caller-supplied "
            "with references. No calibration or source-solve convergence is inferred."
        ),
    }


def _text(value):
    return str(value).replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def _number(value):
    return f"{value:.9g}"


def render(report):
    design = report["design_length"]
    if design.get("value_m") is None:
        length = "Not recorded"
    else:
        length = (
            _number(design["value_m"])
            + " m; "
            + _text(design.get("definition") or "definition not recorded")
        )
        length += "; source: " + _text(
            design.get("source_reference") or "not recorded (supplied frontmatter)"
        )
    grid = report["export_grid"]
    spacing = []
    for axis in "xyz":
        low, high = (_number(grid["spacing_m"][axis][k] * 1000) for k in ("min", "max"))
        spacing.append(low if low == high else low + " .. " + high)
    extent = "; ".join(
        axis + ": [" + ", ".join(_number(v) for v in grid["extent_m"][axis]) + "]" for axis in "xyz"
    )
    mesh = report["solver_mesh"]
    mesh_text = (
        (_text(mesh["description"]) + "; source: " + _text(mesh["source_reference"]))
        if mesh
        else "Not recorded"
    )
    currents = report["field_calculation_currents"]
    current_text = "Not recorded"
    if currents:
        current_text = "<br>".join(
            f"{row['current_a']:+.9g} A"
            + (" — " + _text(row["description"]) if row.get("description") else "")
            + "; source: "
            + _text(row["source_reference"])
            for row in currents
        )
    source = report["field_source"]
    source_text = (
        (
            _text(source["path"])
            + "; slot: "
            + _text(source["slot"])
            + "; SHA-256: "
            + _text(source["sha256"])
        )
        if source
        else "Constructed field; no source-file identity recorded"
    )
    rows = [
        (
            "Model longitudinal input-to-output length",
            (
                _number(report["longitudinal_span"]["value_m"])
                + " m; abs(output - input) for input="
                + _number(report["longitudinal_span"]["input_m"])
                + " m, output="
                + _number(report["longitudinal_span"]["output_m"])
                + " m; "
                + _text(report["longitudinal_span"]["definition"])
                + "; source: "
                + _text(report["longitudinal_span"]["source_reference"])
            )
            if report["longitudinal_span"]
            else (
                "Not recorded; supply the longitudinal coordinate and boundary stations; "
                "do not substitute design/traced length"
            ),
        ),
        (
            "Straight-line input/output origin separation (m)",
            _number(report["native_boundary_geometry"]["input_output_separation_m"])
            + "; "
            + _text(report["native_boundary_geometry"]["definition"])
            if report["native_boundary_geometry"]
            else "Not recorded; native boundary origins unavailable",
        ),
        ("Design length", length),
        ("Traced-reference path length (m)", _number(report["traced_reference_length_m"])),
        ("Field source used for modeling", source_text),
        ("Export grid nodes x / y / z", " / ".join(map(str, grid["nodes_xyz"]))),
        ("Export spacing x / y / z (mm; min .. max where different)", " / ".join(spacing)),
        ("Field extent in source-grid coordinates (m)", extent),
        ("Internal field-solver mesh", mesh_text),
        ("Field-calculation input current(s)", current_text),
    ]
    return "\n".join(
        [
            "### Design geometry and source-field calculation",
            "",
            "| Property | Value and provenance |",
            "|---|---|",
            *[f"| {key} | {value} |" for key, value in rows],
            "",
            "Export spacing/extent are computed from the actual source-grid axes; exact SI "
            "ranges are saved in model_physical_provenance.json. Export spacing does not "
            "establish the internal solver mesh. Field extent does not establish physical "
            "aperture or qualified trajectory support. Design length, traced-reference travel "
            "and field-integral effective length have separate definitions. Field-calculation "
            "current is recorded only from supplied source facts; runtime field scale/effective "
            "rigidity does not establish it.",
        ]
    )
