"""Validity-versus-size reporting from observed qualification records."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from element_surrogate.benchmark.protocol import read_json, write_json


def build_domain_report(qualification_paths, output_dir) -> dict:
    """Report maximal tested regions meeting the declared numerical limits.

    Report both dimensions for rectangles. A recommendation is conditional on
    the provided support/material evidence and remains the user's decision.
    Maxwell residuals/material geometry are separate readiness evidence; they
    are not inferred from trajectory refinement here.
    """
    records = [read_json(p) for p in qualification_paths]
    if not records:
        raise ValueError("qualification records required")
    reference = records[0]
    rows = []
    for path, record in zip(qualification_paths, records, strict=True):
        if record.get("schema") != "element_surrogate.transport_qualification.v1":
            raise ValueError("unsupported qualification report")
        for key in (
            "component_limits",
            "path_limit_m",
            "refinements",
            "implementation_sha256",
            "probe_states_sha256",
            "probe_count_per_setting",
        ):
            if record[key] != reference[key]:
                raise ValueError(
                    "compare sizes using identical numerical criteria and implementation"
                )
        a = dict(record["scope"]["transport"])
        b = dict(reference["scope"]["transport"])
        region = a.pop("trajectory_region")
        reference_region = b.pop("trajectory_region")
        fixed = {k: v for k, v in region.items() if k not in {"radius_m", "half_height_m"}}
        fixed_reference = {
            k: v for k, v in reference_region.items() if k not in {"radius_m", "half_height_m"}
        }
        if fixed != fixed_reference:
            raise ValueError(
                "compare regions with the same shape, axis/path and longitudinal bounds"
            )
        if record["scope"].get("effective_coupling_range") != reference["scope"].get(
            "effective_coupling_range"
        ):
            raise ValueError("size study changes the declared coupling range")
        if (
            a != b
            or record["scope"]["ranges"] != reference["scope"]["ranges"]
            or record["scope"]["settings"] != reference["scope"]["settings"]
        ):
            raise ValueError("size study changes the field/reference/ranges/settings/support")
        changes = [r["changes"][-1] for r in record["reports"]]
        ratios = [
            max(
                v / limit
                for v, limit in zip(
                    c["max_component_change"], record["component_limits"], strict=True
                )
            )
            for c in changes
            if c["max_component_change"] is not None
        ]
        rows.append(
            {
                "radius_m": region["radius_m"],
                "half_height_m": region["half_height_m"],
                "shape": region["shape"],
                "accepted": record["accepted"],
                "minimum_common_survivors": min(r["common_survivors"] for r in record["reports"]),
                "max_last_component_change_over_limit": max(ratios) if ratios else None,
                "path": str(Path(path).resolve()),
            }
        )
    shapes = {r["shape"] for r in rows}
    if len(shapes) != 1:
        raise ValueError("compare the same region shape")
    rows.sort(key=lambda r: (r["radius_m"], r["half_height_m"] or r["radius_m"]))
    axis_key = (
        "half_height_m"
        if len({r["radius_m"] for r in rows}) == 1 and rows[0]["shape"] == "rectangle"
        else "radius_m"
    )
    axis_label = (
        "Vertical half-height (mm)"
        if axis_key == "half_height_m"
        else "Radius / horizontal half-width (mm)"
    )
    acceptable = [r for r in rows if r["accepted"]]

    def dimensions_of(row):
        return (row["radius_m"], row["half_height_m"] or row["radius_m"])

    def strictly_contains(a, b):
        da, db = dimensions_of(a), dimensions_of(b)
        return da != db and all(x >= y for x, y in zip(da, db, strict=True))

    maximal = [r for r in acceptable if not any(strictly_contains(a, r) for a in acceptable)]
    footprints = {dimensions_of(r) for r in maximal}
    recommendation = maximal[0] if len(footprints) == 1 else None
    result: dict[str, Any] = {
        "candidates": rows,
        "recommendation": recommendation,
        "maximal_passing_candidates": maximal,
        "criterion": (
            "maximal by inclusion among tested regions passing finite-cohort refinement criteria"
        ),
        "decision": (
            "use supplied user/delegated authority; incomparable passing regions require "
            "a declared objective or choice, not an invented area ranking; "
            "longitudinal variation follows the user's supplied decision"
        ),
        "limits": (
            "numerical qualification only; no mechanical aperture, field "
            "calibration or continuum-path guarantee inferred"
        ),
    }
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "domain.json", result, exclusive=True)
    # Each candidate gets a distinct column, so incomparable rectangles and equal
    # horizontal widths do not overlap. Numeric annotations expose small values
    # without a fabricated logarithmic floor for zeros or undefined ratios.
    width, height = max(800, 160 * len(rows)), 420
    maximum = max(1, max((r["max_last_component_change_over_limit"] or 0) for r in rows))
    left, right, top, bottom = 85, width - 35, 65, 290

    def y_of(value):
        return bottom - (bottom - top) * value / maximum

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="50" y="27" font-size="18">Numerical validity versus '
        "tested trajectory-region size</text>",
        f'<path d="M {left} {top} V {bottom} H {right}" fill="none" stroke="#222"/>',
    ]
    for value in sorted({0, 1, maximum}):
        y = y_of(value)
        elements += [
            f'<path d="M {left} {y} H {right}" stroke="#bbb" stroke-dasharray="5 4"/>',
            f'<text x="75" y="{y + 4}" text-anchor="end" font-size="12">{value:.3g}</text>',
        ]
    elements.append(
        f'<text x="{right}" y="{y_of(1) - 8}" text-anchor="end" font-size="12">'
        "declared component-change limit = 1</text>"
    )
    for i, row in enumerate(rows):
        x = left + (right - left) * (i + 0.5) / len(rows)
        value = row["max_last_component_change_over_limit"]
        color = "#16724b" if row["accepted"] else "#b13a37"
        if value is None:
            elements.append(
                f'<text x="{x}" y="{bottom - 12}" text-anchor="middle" '
                f'font-size="13" fill="{color}">not computed</text>'
            )
        else:
            y = y_of(value)
            elements += [
                f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"/>',
                f'<text x="{x}" y="{max(top + 12, y - 12)}" text-anchor="middle" '
                f'font-size="13" fill="{color}">{value:.4g}</text>',
            ]
        vertical_mm = (row["half_height_m"] or row["radius_m"]) * 1000
        label = (
            f"r={row['radius_m'] * 1000:g}"
            if row["shape"] == "circle"
            else f"w={row['radius_m'] * 1000:g}, h={vertical_mm:g}"
        )
        elements.append(
            f'<text x="{x}" y="320" text-anchor="middle" font-size="13">{label} mm</text>'
        )
    elements += [
        f'<text x="{width / 2}" y="350" text-anchor="middle" font-size="14">'
        f"{axis_label}; rectangle w/h are half sizes</text>",
        '<text x="22" y="185" transform="rotate(-90 22 185)" text-anchor="middle" '
        'font-size="13">Last component change / limit</text>',
        '<text x="50" y="380" font-size="12">Green: declared refinement/status '
        "criteria pass. Red: failed or inconclusive.</text>",
        '<text x="50" y="402" font-size="12">Numerical support only; field/material '
        "validity is separate. Missing values are not zero.</text></svg>",
    ]
    (output / "validity_vs_size.svg").write_text("\n".join(elements), encoding="utf-8")
    lines = [
        "# Numerical domain study",
        "",
        result["criterion"] + ". " + result["decision"] + ".",
        "",
        result["limits"] + ".",
        "",
        (
            "| Horizontal size (mm) | Vertical half-height (mm) | Accepted | "
            "Minimum common survivors | Last component change / limit |"
        ),
        "|---:|---:|---|---:|---:|",
    ]
    for r in rows:
        vertical = "N/A" if r["half_height_m"] is None else f"{r['half_height_m'] * 1000:g}"
        lines.append(
            f"| {r['radius_m'] * 1000:g} | {vertical} | {r['accepted']} | "
            f"{r['minimum_common_survivors']} | {r['max_last_component_change_over_limit']} |"
        )
    lines.extend(
        ["", "![Validity versus size](validity_vs_size.svg)", "", html.escape(result["limits"])]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "report_path": str(output / "REPORT.md"),
        "plot_path": str(output / "validity_vs_size.svg"),
        "recommendation": recommendation,
        "maximal_passing_candidates": maximal,
        "selection_objective_needed": len(footprints) > 1,
        "authority": (
            "use the user's already supplied decision or delegation; no tool creates approval"
        ),
    }
