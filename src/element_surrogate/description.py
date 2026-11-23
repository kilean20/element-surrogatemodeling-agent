"""The user's description of a magnet, and how the field data agree with it.

A description is a dictionary; every key is optional, and anything not described is
reported as such, not assumed:

* ``kind``: ``dipole``, ``corrector``, ``quadrupole``, ``sextupole``, ``octupole``,
  ``solenoid`` or another word; the first five imply the main harmonic.
* ``main_harmonic``: the main harmonic ``n`` (1 dipole, 2 quadrupole, ...), if ``kind``
  does not imply it.
* ``integrated_strength``: the integral along the magnet of the main normal harmonic, in SI
  units: ``integral of b_n dz`` with ``b_n = C_n / R^(n - 1)``, so tesla-metres for a dipole,
  tesla for a quadrupole (the integrated gradient), tesla per metre for a sextupole. Its
  sign is the polarity: a positive quadrupole value focuses positive particles moving
  along +z in x. For a bending magnet, the integral of the field along the band centre.
* ``current_a`` with ``strength_per_ampere``: the same integral as current times transfer
  function, when that is how the user knows it.
* ``bend``: the nominal path, ``radius_m``, ``angle_rad`` and optionally ``bends_toward``.
* ``aperture``: whatever the user knows of the pipe or the bore.
* ``tolerance``: the relative uncertainty of the stated numbers; mismatches are judged
  against it, and without it they are reported but not flagged.

Units are not trusted to labels: a ratio of measured to stated values equal to a unit
conversion (gauss and tesla, centimetres and metres, B and H) is named in the finding.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

from element_surrogate.analysis import field_frame, harmonics
from element_surrogate.fields import FieldMap
from element_surrogate.readiness import finding
from element_surrogate.region import Region

KNOWN_KEYS = (
    "kind",
    "main_harmonic",
    "integrated_strength",
    "current_a",
    "strength_per_ampere",
    "bend",
    "aperture",
    "tolerance",
)
MAIN_HARMONIC = {"dipole": 1, "corrector": 1, "quadrupole": 2, "sextupole": 3, "octupole": 4}
MU0 = 4.0e-7 * math.pi

# Scale factors a mislabelled unit introduces into a field value or a length.
_FIELD_SCALES = {
    "gauss read as tesla": 1.0e4,
    "tesla read as gauss": 1.0e-4,
    "millitesla read as tesla": 1.0e3,
    "tesla read as millitesla": 1.0e-3,
    "H in A/m read as B in tesla": 1.0 / MU0,
}
_LENGTH_SCALES = {
    "centimetres read as metres": 1.0e2,
    "metres read as centimetres": 1.0e-2,
    "millimetres read as metres": 1.0e3,
    "metres read as millimetres": 1.0e-3,
}


def unit_explanations(ratio: float, n: int, tolerance: float) -> list[str]:
    """Unit mislabels that would turn the stated integral into the measured one: field
    scales times length scales to the power ``2 - n``, the dimension of the integral of
    ``b_n`` in length."""
    explanations = []
    fields = {"": 1.0, **_FIELD_SCALES}
    lengths = {"": 1.0, **_LENGTH_SCALES}
    for (f_name, f), (l_name, s) in itertools.product(fields.items(), lengths.items()):
        if not (f_name or l_name):
            continue
        factor = f * s ** (2 - n)
        if factor != 1.0 and abs(math.log(abs(ratio) / factor)) <= math.log1p(tolerance):
            explanations.append(" and ".join(name for name in (f_name, l_name) if name))
    return explanations


def _stated_strength(description: dict[str, Any]) -> tuple[float | None, str | None]:
    if description.get("integrated_strength") is not None:
        return float(description["integrated_strength"]), "integrated_strength"
    if (
        description.get("current_a") is not None
        and description.get("strength_per_ampere") is not None
    ):
        value = float(description["current_a"]) * float(description["strength_per_ampere"])
        return value, "current_a x strength_per_ampere"
    return None, None


def _compare(
    name: str, measured: float, stated: float, tolerance: float | None, n: int | None
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    ratio = measured / stated if stated != 0.0 else math.inf
    row: dict[str, Any] = {"stated": stated, "measured": measured, "measured_over_stated": ratio}
    if tolerance is None or abs(ratio - 1.0) <= tolerance:
        return row, None
    reasons = []
    if abs(-ratio - 1.0) <= tolerance:
        reasons.append("polarity reversed")
    if n is not None and math.isfinite(ratio) and ratio != 0.0:
        reasons += unit_explanations(ratio, n, tolerance)
    row["possible_causes"] = reasons
    summary = f"measured {name} is {ratio:.6g} x the stated value"
    if reasons:
        summary += f" (matches: {'; '.join(reasons)})"
    elif name == "integrated strength":
        summary += " (a wrong current label or transfer function would also show this)"
    return row, finding(
        "description",
        f"{name.replace(' ', '_')}_mismatch",
        summary,
        f"|measured / stated - 1| above the stated tolerance {tolerance:g}",
        quantity=name,
    )


def compare_description(
    field: FieldMap,
    description: dict[str, Any],
    fit_region: Region | None = None,
    order: int = 6,
) -> dict[str, Any]:
    """Compare the field data with the user's description (see the module docstring).

    Multipoles are fitted in ``fit_region`` (default: the largest circle about the z axis
    that the grid covers); a bending magnet's strength and geometry come from
    ``analysis.field_frame``. Returns each comparison, what was not described, and
    findings. Without a stated ``tolerance``, numbers are compared but not flagged.
    """
    unknown = sorted(set(description) - set(KNOWN_KEYS))
    if unknown:
        raise ValueError(f"unknown description keys {unknown}; known: {list(KNOWN_KEYS)}")
    tolerance = description.get("tolerance")
    if tolerance is not None and not tolerance > 0.0:
        raise ValueError("tolerance must be positive")
    kind = description.get("kind")
    stated_n = description.get("main_harmonic", MAIN_HARMONIC.get(str(kind)))
    findings: list[dict[str, Any]] = []
    comparisons: dict[str, Any] = {}
    not_described = [
        key for key in ("kind", "aperture", "tolerance") if description.get(key) is None
    ]
    stated, strength_source = _stated_strength(description)
    if stated is None:
        not_described.append("integrated_strength")

    if description.get("aperture") is None:
        findings.append(
            finding(
                "description",
                "aperture_not_described",
                "no pipe or bore aperture is described: the region where training particles "
                "may go must come from the field's own vacuum-Maxwell analysis",
                "the description has no aperture",
            )
        )

    bend = description.get("bend")
    measured_n = None
    proposal = None
    if bend is not None or kind == "dipole":
        frame = field_frame(field)
        proposal = frame["proposed_bend"]
        comparisons["frame"] = {
            "ranked_by_residual": frame["ranked_by_residual"],
            "proposed_bend": proposal,
        }
        if bend is not None and proposal is None:
            findings.append(
                finding(
                    "description",
                    "bend_not_found",
                    "a bend is described but the field band is straight in the grid",
                    "the straight reading fits the band centre best",
                )
            )
        elif bend is not None and proposal is not None:
            for key in ("radius_m", "angle_rad"):
                row, flag = _compare(
                    key.replace("_", " "), proposal[key], float(bend[key]), tolerance, None
                )
                comparisons[key] = row
                findings += [flag] if flag else []
            if bend.get("bends_toward", proposal["bends_toward"]) != proposal["bends_toward"]:
                findings.append(
                    finding(
                        "description",
                        "bend_direction_mismatch",
                        f"the band bends toward {proposal['bends_toward']}, the description says "
                        f"{bend['bends_toward']}",
                        "direction of the fitted curvature",
                    )
                )
    if proposal is not None:
        measured_n = 1
        if stated is not None:
            row, flag = _compare(
                "integrated strength", proposal["integral_along_centre_t_m"], stated, tolerance, 1
            )
            comparisons["integrated_strength"] = {**row, "n": 1, "stated_as": strength_source}
            findings += [flag] if flag else []
    elif kind != "solenoid":
        region = fit_region if fit_region is not None else Region.covering(field)
        fit = harmonics(field, region, region.radius_m, order)
        measured_n = fit["main_harmonic_n"]
        if stated is not None:
            n = stated_n if stated_n is not None else measured_n
            c_n = fit["integrated_t_m"][n - 1]["normal"]
            measured = c_n / region.radius_m ** (n - 1)
            row, flag = _compare("integrated strength", measured, stated, tolerance, n)
            comparisons["integrated_strength"] = {**row, "n": n, "stated_as": strength_source}
            findings += [flag] if flag else []

    if stated_n is not None and measured_n is not None:
        comparisons["main_harmonic"] = {"stated": stated_n, "measured": measured_n}
        if stated_n != measured_n:
            findings.append(
                finding(
                    "description",
                    "main_harmonic_mismatch",
                    f"the strongest harmonic is n = {measured_n}, the description implies "
                    f"n = {stated_n}",
                    "largest integrated harmonic in the fit",
                )
            )

    return {
        "description": description,
        "comparisons": comparisons,
        "not_described": not_described,
        "findings": findings,
        "reading": (
            "Each finding names what disagrees and, where a unit mislabel or a reversed polarity "
            "would explain it, which. Ask the user which is right before using the data."
        ),
    }
