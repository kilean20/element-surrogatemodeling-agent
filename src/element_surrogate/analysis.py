"""Field analysis: multipoles, profiles, lengths and mirror symmetry of a straight-axis
map, and the frame of a map's strong-field band (straight or curved in the file's grid).

Multipoles use the 2D expansion ``By + i Bx = sum_n C_n ((x + i y) / R)^(n - 1)``, fitted
by least squares, plane by plane along z, to the nodes inside a fit region. ``n = 1`` is
the dipole, ``n = 2`` the quadrupole; ``Re C_n`` is the normal and ``Im C_n`` the skew
part. In a fringe field this 2D form is only the leading part of the full 3D expansion,
so per-plane coefficients there are descriptive; the z-integrated coefficients are the
ones a beam sees. The fit's own reliability is measured, not assumed: each result also
reports how much the coefficients change when the fit order is doubled.

The main harmonic is taken from the transverse field, so a field whose main component
is longitudinal (a solenoid) is outside this analysis. Every radius and length is an
argument: nothing here assumes a particular magnet.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from element_surrogate.fields import FieldMap
from element_surrogate.region import Region


def _trapezoid(values: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Integral over the first axis."""
    dz = np.diff(z).reshape((-1,) + (1,) * (values.ndim - 1))
    result: np.ndarray = np.sum(0.5 * (values[1:] + values[:-1]) * dz, axis=0)
    return result


def _fit_planes(
    field: FieldMap, region: Region, reference_radius_m: float, order: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Per-plane ``C_n`` in tesla at the reference radius, shape ``(planes, order + 1)``, the
    relative RMS fit residual per plane, the z of the planes used, and the nodes per plane.

    The basis is built in units of the fit region's radius, which keeps the least-squares
    problem well conditioned at any order, and converted to the reference radius after.
    """
    if order < 0:
        raise ValueError("order must be at least 0")
    if not reference_radius_m > 0.0:
        raise ValueError("reference_radius_m must be positive")
    inside = region.transverse_mask(field)
    planes = region.z_mask(field.z_m)
    if planes.sum() < 2:
        raise ValueError("the fit region holds fewer than two planes along z")
    if inside.sum() <= order + 1:
        raise ValueError(
            f"only {int(inside.sum())} nodes per plane in the fit region for order {order}"
        )
    X, Y = np.meshgrid(field.x_m, field.y_m, indexing="ij")
    w = (X + 1j * Y)[inside] / region.radius_m
    basis = np.stack([w**k for k in range(order + 1)], axis=1)
    target = (field.b_t[..., 1] + 1j * field.b_t[..., 0])[inside][:, planes]  # (nodes, planes)
    coefficients, *_ = np.linalg.lstsq(basis, target, rcond=None)  # (order+1, planes)
    fitted = basis @ coefficients
    residual = np.sqrt(np.mean(np.abs(fitted - target) ** 2, axis=0)) / (
        np.sqrt(np.mean(np.abs(target) ** 2, axis=0)) + 1e-300
    )
    to_reference = (reference_radius_m / region.radius_m) ** np.arange(order + 1)
    return coefficients.T * to_reference, residual, field.z_m[planes], int(inside.sum())


def _relative(integrated: np.ndarray) -> tuple[int, np.ndarray]:
    main = int(np.argmax(np.abs(integrated)))
    return main, integrated / abs(integrated[main])


def harmonics(
    field: FieldMap, fit_region: Region, reference_radius_m: float, order: int = 6
) -> dict[str, Any]:
    """Integrated normal and skew multipoles and the main harmonic's profile along z.

    ``C_n`` is in tesla-metres at ``reference_radius_m``, with harmonics relative to the
    largest one. The fit uses the nodes inside ``fit_region`` (choose it inside the region
    where the field passed the readiness checks) and the planes inside its z range.

    The profile is the main harmonic along z, projected onto the direction of its integral.
    ``map_profile_length_m`` is the integral divided by the peak: the length of a hard-edge
    field with the map's peak and the same integral. ``edge_fraction`` is the profile at
    the first and last planes relative to the peak; a map that ends before its field has
    decayed shows a large value there.
    """
    per_plane, residual, z, nodes = _fit_planes(field, fit_region, reference_radius_m, order)
    integrated = _trapezoid(per_plane, z)
    main, relative = _relative(integrated)

    # Compare with a fit of 2 * order + 1. A symmetric magnet has harmonics only every
    # 2, 4 or 6 orders (dipole, quadrupole, sextupole), so two more orders can hold
    # nothing new; doubling reaches the next allowed harmonic for these classes.
    check_order = min(2 * order + 1, nodes - 2)
    sensitivity: dict[str, Any] = {"fitted_order": check_order}
    if check_order > order:
        higher, _, _, _ = _fit_planes(field, fit_region, reference_radius_m, check_order)
        _, relative_higher = _relative(_trapezoid(higher, z))
        change = np.abs(relative_higher[: order + 1] - relative)
        sensitivity.update(
            max_change_relative_to_main=float(change.max()),
            change_by_n=[float(c) for c in change],
        )
    else:
        sensitivity.update(max_change_relative_to_main=None, change_by_n=None)

    values = per_plane[:, main]
    total = _trapezoid(values, z)
    if abs(total) == 0.0:
        raise ValueError("the main harmonic integrates to zero; a profile length is undefined")
    profile = np.real(values / (total / abs(total)))
    peak_index = int(np.argmax(np.abs(profile)))
    peak = float(profile[peak_index])
    integral = float(_trapezoid(profile, z))

    def listing(coefficients: np.ndarray) -> list[dict[str, Any]]:
        return [
            {"n": n + 1, "normal": float(c.real), "skew": float(c.imag)}
            for n, c in enumerate(coefficients)
        ]

    return {
        "expansion": "By + i Bx = sum_n C_n ((x + i y) / R)^(n - 1); Re C_n normal, Im C_n skew",
        "fit_region": fit_region.as_dict(),
        "reference_radius_m": reference_radius_m,
        "order": order,
        "fit_nodes_per_plane": nodes,
        "fit_planes": int(z.size),
        "main_harmonic_n": main + 1,
        "integrated_t_m": listing(integrated),
        "integrated_relative_to_main": listing(relative),
        "fit_residual_relative": {
            "median_plane": float(np.median(residual)),
            "worst_plane": float(np.max(residual)),
        },
        "order_sensitivity": {
            **sensitivity,
            "reading": (
                "how much each relative harmonic moves when the fit order is doubled (plus one); "
                "a change comparable to a harmonic means that harmonic is not determined by this "
                "fit: raise the order or shrink the fit region"
            ),
        },
        "profile": {
            "integral_t_m": integral,
            "peak_t": peak,
            "peak_z_m": float(z[peak_index]),
            "map_profile_length_m": integral / peak,
            "centroid_z_m": float(_trapezoid(profile * z, z) / integral),
            "edge_fraction": [float(profile[0] / peak), float(profile[-1] / peak)],
            "span_m": float(z[-1] - z[0]),
        },
    }


def _half_maximum_band(profile: np.ndarray, x: np.ndarray) -> tuple[float, float] | None:
    """The two half-maximum crossings around a profile's maximum, linearly interpolated;
    None when the band reaches the end of the profile."""
    top = int(np.argmax(profile))
    half = 0.5 * profile[top]
    below = profile < half
    left = np.flatnonzero(below[:top])
    right = np.flatnonzero(below[top:])
    if left.size == 0 or right.size == 0:
        return None
    i0, i1 = int(left[-1]), top + int(right[0])

    def cross(a: int, b: int) -> float:
        return float(x[a] + (half - profile[a]) * (x[b] - x[a]) / (profile[b] - profile[a]))

    return cross(i0, i0 + 1), cross(i1 - 1, i1)


# Readings of a band centre x_c(z) with signed curvature k (k > 0 bends toward -x): each is
# 0 at z = 0 and -k z^2 / 2 to leading order, and they differ from order z^4 on.
def _circle(k: float, z: np.ndarray) -> np.ndarray:
    result: np.ndarray = (np.sqrt(1.0 - (k * z) ** 2) - 1.0) / k
    return result


def _secant(k: float, z: np.ndarray) -> np.ndarray:
    result: np.ndarray = -(1.0 / np.cos(k * z) - 1.0) / k
    return result


_FRAME_READINGS: dict[str, tuple[Any, str]] = {
    "straight": (
        None,
        "the band runs straight in the grid: a straight magnet, or a curved one written in "
        "coordinates that follow it",
    ),
    "circle": (
        _circle,
        "a straight Cartesian grid with the magnet curving in it (the TRACK mag3d box)",
    ),
    "secant": (_secant, "a rectangular magnet written in its own sector coordinates"),
}


def _fit_reading(shape: Any, z: np.ndarray, xc: np.ndarray, k_guess: float) -> dict[str, Any]:
    """Least-squares fit of x_c = a + b z + shape(k, z): a and b linear, k by golden-section
    search about ``k_guess`` (the quadratic fit's curvature)."""
    basis = np.stack([np.ones_like(z), z], axis=1)

    def solve(k: float) -> tuple[float, np.ndarray, np.ndarray]:
        bent = shape(k, z) if (shape is not None and k != 0.0) else np.zeros_like(z)
        coef, *_ = np.linalg.lstsq(basis, xc - bent, rcond=None)
        residual = xc - bent - basis @ coef
        return float(np.sqrt(np.mean(residual**2))), coef, residual

    k = 0.0
    if shape is not None and k_guess != 0.0:
        limit = 0.999 / float(np.max(np.abs(z)))
        lo, hi = sorted((0.5 * k_guess, float(np.sign(k_guess)) * min(1.5 * abs(k_guess), limit)))
        golden = (np.sqrt(5.0) - 1.0) / 2.0
        for _ in range(100):
            m1, m2 = hi - golden * (hi - lo), lo + golden * (hi - lo)
            if solve(m1)[0] <= solve(m2)[0]:
                hi = m2
            else:
                lo = m1
        k = 0.5 * (lo + hi)
    rms, coef, residual = solve(k)
    return {
        "curvature_per_m": k,
        "radius_m": (1.0 / abs(k)) if k != 0.0 else None,
        "bends_toward": None if k == 0.0 else ("-x" if k > 0 else "+x"),
        "offset_m": float(coef[0]),
        "tilt_rad": float(np.arctan(coef[1])),
        "rms_residual_m": rms,
        "max_residual_m": float(np.max(np.abs(residual))),
    }


def _proposed_bend(
    circle: dict[str, Any], x: np.ndarray, z: np.ndarray, values: np.ndarray, body_t: float
) -> dict[str, Any]:
    """The nominal path that the circle reading implies: its radius, and the angle at which a
    hard-edge field equal to ``body_t`` gives the integral of the field along the centre."""
    k = circle["curvature_per_m"]
    inside = np.flatnonzero(np.abs(k * z) < 1.0)
    zi = z[inside]
    centre = circle["offset_m"] + np.tan(circle["tilt_rad"]) * zi + _circle(k, zi)
    on_centre = np.array(
        [np.interp(c, x, values[:, i]) for c, i in zip(centre, inside, strict=True)]
    )
    ds = np.hypot(np.diff(zi), np.diff(centre))
    integral = float(np.sum(0.5 * (on_centre[1:] + on_centre[:-1]) * ds))
    radius = circle["radius_m"]
    angle = integral / (body_t * radius)
    arm = (float(np.max(np.abs(z))) - radius * np.sin(0.5 * angle)) / np.cos(0.5 * angle)
    return {
        "radius_m": radius,
        "angle_rad": angle,
        "arm_m": max(arm, 0.0),
        "bends_toward": circle["bends_toward"],
        "integral_along_centre_t_m": integral,
        "body_field_t": body_t,
    }


def field_frame(
    field: FieldMap, component: str | None = None, core_fraction: float = 0.5
) -> dict[str, Any]:
    """Decide, from the field values alone, how the strong-field band lies in the file's grid.

    In the grid plane nearest y = 0, each z plane's profile of the main transverse component
    is cut at half its maximum; the band's centre ``x_c(z)`` is the midpoint of the two
    crossings. The body is the planes whose maximum is at least half the map's; near its
    ends the pole ends bend the band, so the centre is fitted over the central
    ``core_fraction`` of the body's z extent, in the planes whose band lies inside the grid
    (a band wider than the grid has no centre there), with three readings: straight, circle and
    secant (see ``_FRAME_READINGS``). The circle and the secant agree to order z^2, so
    only residuals well apart tell them apart. The band is a property of the field values,
    not a coordinate system or an orbit.

    When a curved reading fits best, a nominal path is proposed from the circle: its radius,
    and the angle at which a hard-edge field equal to the body field (the band maximum in the
    middle body plane) gives the same integral along the band centre,
    ``angle = integral / (B_body * radius)``. Its arms reach the grid's z ends. It places
    regions and boundaries; the reference orbit is found later, by tracking, between planes
    that lie inside the field (shorter arms: the end planes are tilted by half the angle).
    """
    if not 0.0 < core_fraction <= 1.0:
        raise ValueError("core_fraction must be in (0, 1]")
    names = {"Bx": 0, "By": 1}
    j = int(np.argmin(np.abs(field.y_m)))
    plane = field.b_t[:, j, :, :]
    if component is None:
        c = int(np.argmax([np.max(np.abs(plane[..., k])) for k in (0, 1)]))
    elif component in names:
        c = names[component]
    else:
        raise ValueError("component must be 'Bx' or 'By'")
    sign = 1.0 if plane[..., c].max() >= -plane[..., c].min() else -1.0
    values = sign * plane[..., c]
    peak = float(values.max())
    if not peak > 0.0:
        raise ValueError("the chosen component is zero in the plane nearest y = 0")

    x, z = field.x_m, field.z_m
    body = [k for k in range(z.size) if values[:, k].max() >= 0.5 * peak]
    rows, clipped = [], 0
    for k in body:
        band = _half_maximum_band(values[:, k], x)
        if band is None:  # the band runs off the grid in x: no centre in this plane
            clipped += 1
            continue
        rows.append((z[k], 0.5 * (band[0] + band[1]), band[1] - band[0]))
    if not rows:
        raise ValueError("no body plane has a band inside the grid")
    zb, xc, width = (np.array(column) for column in zip(*rows, strict=True))
    z_body = z[body]
    mid = 0.5 * (z_body[0] + z_body[-1])
    core = np.abs(zb - mid) <= 0.5 * core_fraction * (z_body[-1] - z_body[0]) * (1 + 1e-12)
    if core.sum() < 5:
        raise ValueError(f"only {int(core.sum())} planes in the fitted core; it needs five")
    k_guess = float(-2.0 * np.polyfit(zb[core], xc[core], 2)[0])
    readings = {}
    for name, (shape, meaning) in _FRAME_READINGS.items():
        fit = _fit_reading(shape, zb[core], xc[core], k_guess)
        readings[name] = {**fit, "meaning": meaning}
    ranked = sorted(readings, key=lambda name: readings[name]["rms_residual_m"])
    # The body field is the maximum in the body's middle plane, clipped band or not.
    body_field = float(values[:, body[int(np.argmin(np.abs(z_body - mid)))]].max())
    nearest = int(np.argmin(np.abs(zb - mid)))
    proposal = None
    if ranked[0] != "straight":
        proposal = _proposed_bend(readings["circle"], x, z, values, body_field)
        proposal["body_field_t"] *= sign
        proposal["integral_along_centre_t_m"] *= sign
    return {
        "component": ("Bx", "By")[c],
        "plane_y_m": float(field.y_m[j]),
        "body_planes": len(body),
        "body_planes_with_band_off_the_grid": clipped,
        "fitted_planes": int(core.sum()),
        "core_fraction": core_fraction,
        "band": {
            "body_z_range_m": [float(z_body[0]), float(z_body[-1])],
            "centre_z_range_m": [float(zb[0]), float(zb[-1])],
            "fitted_z_range_m": [float(zb[core].min()), float(zb[core].max())],
            "mean_width_m": float(width.mean()),
            "centre_at_ends_m": [float(xc[0]), float(xc[-1])],
            "centre_nearest_middle": {"z_m": float(zb[nearest]), "x_m": float(xc[nearest])},
        },
        "grid_x_step_m": float(np.median(np.diff(x))),
        "readings": readings,
        "ranked_by_residual": ranked,
        "proposed_bend": proposal,
        "reading": (
            "Compare each reading's rms_residual_m with grid_x_step_m, the scale of the "
            "crossing interpolation. The proposed bend is a nominal path for regions and "
            "boundaries, not the reference orbit; confirm the frame with the user."
        ),
    }


def _eager_mirror_symmetry(field: FieldMap, region: Region | None = None) -> dict[str, Any]:
    """How close each field component is to even or odd under each mirror x -> -x, y -> -y,
    z -> -z of the grid, over the nodes of ``region`` (default: the whole grid).

    For component ``B_c`` and mirror ``M``: ``even_defect = rms(B_c - M B_c) / rms(B_c)`` and
    ``odd_defect = rms(B_c + M B_c) / rms(B_c)``. A value near 0 means the component has
    that parity; a real magnet may have neither, and an asymmetry can be real. A mirror is
    evaluated only when the grid and the region are both symmetric under it.
    """
    mask = region.node_mask(field) if region is not None else np.ones(field.shape, dtype=bool)
    out: dict[str, Any] = {"region": region.as_dict() if region is not None else None}
    for axis, coord in enumerate(("x", "y", "z")):
        values = (field.x_m, field.y_m, field.z_m)[axis]
        grid_ok = np.allclose(values, -values[::-1], rtol=0.0, atol=1e-9 * np.ptp(values))
        region_ok = region is None or region.mirror_symmetric(axis)
        if not (grid_ok and region_ok):
            out[f"{coord}_mirror"] = None
            continue
        per_component: dict[str, dict[str, float] | None] = {}
        for c, name in enumerate(("Bx", "By", "Bz")):
            b = field.b_t[..., c][mask]
            m = np.flip(field.b_t[..., c], axis=axis)[mask]
            norm = float(np.sqrt(np.mean(b * b)))
            if norm == 0.0:
                per_component[name] = None
                continue
            per_component[name] = {
                "even_defect": float(np.sqrt(np.mean((b - m) ** 2)) / norm),
                "odd_defect": float(np.sqrt(np.mean((b + m) ** 2)) / norm),
            }
        out[f"{coord}_mirror"] = per_component
    out["reading"] = (
        "near 0 even_defect or odd_defect: the component has that parity; neither near 0: no "
        "mirror symmetry along that axis, which may be a real property of the magnet; measure "
        "inside the region that passed the readiness checks, so defects elsewhere do not count"
    )
    return out


def mirror_symmetry(field: FieldMap, region: Region | None = None) -> dict[str, Any]:
    """Exact all-node parity reductions without whole-grid masks/selected copies."""
    from element_surrogate.diagnostic_blocks import blocks, node_window

    out: dict[str, Any] = {"region": region.as_dict() if region is not None else None}
    for axis, coord in enumerate("xyz"):
        values = (field.x_m, field.y_m, field.z_m)[axis]
        grid_ok = np.allclose(values, -values[::-1], rtol=0.0, atol=1e-9 * np.ptp(values))
        if not grid_ok or (region is not None and not region.mirror_symmetric(axis)):
            out[f"{coord}_mirror"] = None
            continue
        squares = np.zeros((3, 3))
        count = 0
        mirrored = np.flip(field.b_t, axis=axis)
        for core in blocks(field.shape, 65536):
            view = node_window(field, core)
            mask = np.ones(view.shape, bool) if region is None else region.node_mask(view)
            if not mask.any():
                continue
            b, m = field.b_t[core][mask], mirrored[core][mask]
            squares[0] += np.sum(b * b, axis=0)
            squares[1] += np.sum((b - m) ** 2, axis=0)
            squares[2] += np.sum((b + m) ** 2, axis=0)
            count += int(mask.sum())
        per_component = {}
        for c, name in enumerate(("Bx", "By", "Bz")):
            if squares[0, c] == 0 and count:
                per_component[name] = None
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    per_component[name] = {
                        "even_defect": float(np.sqrt(squares[1, c] / squares[0, c])),
                        "odd_defect": float(np.sqrt(squares[2, c] / squares[0, c])),
                    }
        out[f"{coord}_mirror"] = per_component
    out["reading"] = (
        "near 0 even_defect or odd_defect: the component has that parity; neither near 0: no "
        "mirror symmetry along that axis, which may be a real property of the magnet; measure "
        "inside the region that passed the readiness checks, so defects elsewhere do not count"
    )
    return out
