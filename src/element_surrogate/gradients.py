"""Generalized gradients of a straight-axis map (Venturini and Dragt).

Inside a current-free cylinder of radius R about the z axis, ``B = grad(psi)`` with

    psi = sum_{m, l} a_ml rho^(m + 2l) [C_m,c^[2l](z) cos(m phi) + C_m,s^[2l](z) sin(m phi)],
    a_ml = (-1)^l m! / (2^(2l) l! (m + l)!),

where ``C^[n]`` is the n-th z derivative. The gradients follow from ``B_rho`` on the
cylinder alone: with ``b_m(z)`` its m-th angular harmonic,

    C_m^[n](z) = IFFT[ FFT(b_m) (i k)^n k^(m - 1) / (2^m m! I'_m(k R)) ].

Surface data therefore fix the whole interior field, and the map's interior nodes are a
test of it: ``generalized_gradients`` rebuilds the field at every node inside the cylinder
and reports how far the map is from it.

In the multipole convention of ``analysis`` (``By + i Bx = sum C_n ((x + i y) / R)^(n - 1)``)
the normal part of ``b_n = C_n / R^(n - 1)`` is ``n C_n,s^[0]`` and the skew part
``n C_n,c^[0]``. Their z integrals are the integrals of ``B_rho``'s harmonics on the
cylinder divided by ``R^(n - 1)``; the kernel shapes their profiles.

Beyond the map's ends each harmonic is continued with an exponential tail whose decay
length is measured at that end from the strongest harmonic. The m = 0 (solenoidal) part
needs the net axial flux, which ``B_rho`` does not give: its k = 0 mode is dropped.
Radii, orders and sampling are arguments; nothing here assumes a particular magnet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage, special

from element_surrogate.fields import FieldMap

INTERPOLATIONS = ("linear", "cubic")


@dataclass(frozen=True)
class Gradients:
    """``cos[m, n, k]`` and ``sin[m, n, k]`` are ``C_m,c^[n]`` and ``C_m,s^[n]`` at the map's
    planes ``z_m[k]``, in T / m^(m + n - 1)."""

    z_m: np.ndarray
    radius_m: float
    cos: np.ndarray
    sin: np.ndarray
    tails: dict[str, Any]

    @property
    def m_max(self) -> int:
        return int(self.cos.shape[0] - 1)

    @property
    def n_max(self) -> int:
        return int(self.cos.shape[1] - 1)


def _integral(values: np.ndarray, z: np.ndarray) -> float:
    """Trapezoid-rule integral over z (written out: numpy's helper changed name in 2.0)."""
    return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(z)))


def _uniform(axis: np.ndarray) -> bool:
    steps = np.diff(axis)
    return bool(np.allclose(steps, steps[0], rtol=1e-9, atol=0.0))


def surface_harmonics(
    field: FieldMap, radius_m: float, m_max: int, n_phi: int, interpolation: str = "linear"
) -> tuple[np.ndarray, np.ndarray]:
    """``b_m,c(z)`` and ``b_m,s(z)``, shape ``(m_max + 1, nz)``, with
    ``B_rho = sum_m b_m,c cos(m phi) + b_m,s sin(m phi)`` on the circle of ``radius_m``,
    sampled at ``n_phi`` angles by interpolating Bx and By in each plane."""
    if interpolation not in INTERPOLATIONS:
        raise ValueError(f"interpolation must be one of {INTERPOLATIONS}")
    if n_phi < 2 * m_max + 2:
        raise ValueError(f"n_phi = {n_phi} cannot resolve harmonics up to m = {m_max}")
    x, y = field.x_m, field.y_m
    margin = 1 if interpolation == "linear" else 2
    room = min(-x[margin - 1], x[-margin], -y[margin - 1], y[-margin])
    if not 0.0 < radius_m < room:
        raise ValueError(f"radius_m must be positive and below {room:.6g} m for {interpolation}")
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    px, py = radius_m * np.cos(phi), radius_m * np.sin(phi)
    if interpolation == "cubic":
        if not (_uniform(x) and _uniform(y)):
            raise ValueError("cubic sampling needs a uniform grid in x and y")
        fx = (px - x[0]) / (x[1] - x[0])
        fy = (py - y[0]) / (y[1] - y[0])
        bx, by = (
            np.stack(
                [
                    ndimage.map_coordinates(field.b_t[:, :, k, c], [fx, fy], order=3)
                    for k in range(field.z_m.size)
                ],
                axis=1,
            )
            for c in (0, 1)
        )
    else:
        ix = np.searchsorted(x, px) - 1
        iy = np.searchsorted(y, py) - 1
        tx = ((px - x[ix]) / (x[ix + 1] - x[ix]))[:, None]
        ty = ((py - y[iy]) / (y[iy + 1] - y[iy]))[:, None]

        def bilinear(c: int) -> np.ndarray:
            b = field.b_t[..., c]
            result: np.ndarray = (
                b[ix, iy] * (1 - tx) * (1 - ty)
                + b[ix + 1, iy] * tx * (1 - ty)
                + b[ix, iy + 1] * (1 - tx) * ty
                + b[ix + 1, iy + 1] * tx * ty
            )
            return result

        bx, by = bilinear(0), bilinear(1)
    brho = bx * np.cos(phi)[:, None] + by * np.sin(phi)[:, None]
    spectrum = np.fft.fft(brho, axis=0) / n_phi
    bc = np.zeros((m_max + 1, field.z_m.size))
    bs = np.zeros_like(bc)
    bc[0] = spectrum[0].real
    bc[1:] = 2.0 * spectrum[1 : m_max + 1].real
    bs[1:] = -2.0 * spectrum[1 : m_max + 1].imag
    return bc, bs


def _tail(profile: np.ndarray, z: np.ndarray, end: int) -> dict[str, Any]:
    """The exponential continuation beyond one end (``end`` 0 or -1) of the strongest
    harmonic: its decay length from the last two planes, or none when it is not decaying."""
    step = 1 if end == 0 else -1
    b_end, b_prev = abs(profile[end]), abs(profile[end + step])
    if 0.0 < b_end < b_prev:
        decay = float(abs(z[end + step] - z[end]) / math.log(b_prev / b_end))
        return {"decaying": True, "decay_length_m": decay}
    return {"decaying": False, "decay_length_m": 0.0}


def fit_gradients(
    field: FieldMap,
    radius_m: float,
    m_max: int = 12,
    l_max: int = 6,
    n_phi: int | None = None,
    interpolation: str = "linear",
    decay_length_m: float | None = None,
) -> Gradients:
    """Generalized gradients ``C_m^[n]`` for ``m <= m_max`` and ``n <= 2 l_max + 1`` at the
    map's planes. ``decay_length_m`` fixes the tail decay length at both ends instead of
    measuring it (to reproduce a computation that fixed it)."""
    z = field.z_m
    if not _uniform(z):
        raise ValueError("the Fourier kernel needs a uniform grid in z")
    if m_max < 1 or l_max < 0:
        raise ValueError("need m_max >= 1 and l_max >= 0")
    n_phi = max(128, 4 * (m_max + 1)) if n_phi is None else n_phi
    bc, bs = surface_harmonics(field, radius_m, m_max, n_phi, interpolation)
    dz = float(z[1] - z[0])
    strength = np.sum(np.abs(bc) + np.abs(bs), axis=1)
    strongest = int(np.argmax(strength))
    main = (
        bc[strongest]
        if np.abs(bc[strongest]).sum() >= np.abs(bs[strongest]).sum()
        else bs[strongest]
    )
    tails: dict[str, Any] = {"strongest_m": strongest}
    lengths: list[float] = []
    for name, end in (("low_z", 0), ("high_z", -1)):
        tail: dict[str, Any] = (
            _tail(main, z, end)
            if decay_length_m is None
            else {"decaying": None, "decay_length_m": float(decay_length_m)}
        )
        tails[name] = tail
        lengths.append(float(tail["decay_length_m"]))
    planes = [int(math.ceil(37.0 * length / dz)) for length in lengths]  # e^-37 ~ 1e-16
    data = z.size + sum(planes)
    n_fft = 1 << int(math.ceil(math.log2(2 * data)))
    left = (n_fft - data) // 2
    start = left + planes[0]

    def extended(profile: np.ndarray) -> np.ndarray:
        buffer = np.zeros(n_fft)
        buffer[start : start + z.size] = profile
        for count, length, end, sign in (
            (planes[0], lengths[0], 0, -1),
            (planes[1], lengths[1], -1, 1),
        ):
            if count:
                steps = np.arange(1, count + 1)
                values = profile[end] * np.exp(-dz * steps / length)
                if sign < 0:
                    buffer[start - count : start] = values[::-1]
                else:
                    buffer[start + z.size : start + z.size + count] = values
        return buffer

    k = 2.0 * np.pi * np.fft.fftfreq(n_fft, d=dz)
    ak, kr = np.abs(k), np.abs(k) * radius_m
    n_max = 2 * l_max + 1
    cos = np.zeros((m_max + 1, n_max + 1, z.size))
    sin = np.zeros_like(cos)
    for m in range(m_max + 1):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            # I'_m(kR) exponentially scaled; the kernel is even in k.
            scaled = (
                special.ive(1, kr)
                if m == 0
                else 0.5 * (special.ive(m - 1, kr) + special.ive(m + 1, kr))
            )
            base = ak ** (m - 1) * np.exp(-kr) / scaled
        if m == 1:
            base[k == 0.0] = 2.0
        elif m >= 2:
            base[k == 0.0] = 2.0**m * math.factorial(m - 1) / radius_m ** (m - 1)
        base = np.where(np.isfinite(base), base, 0.0)
        norm = 1.0 / (2.0**m * math.factorial(m))
        spectra = [np.fft.fft(extended(profile[m])) for profile in (bc, bs)]
        for n in range(n_max + 1):
            kernel = norm * base * (1j * k) ** n
            if m == 0:
                kernel = np.where(k == 0.0, 0.0, kernel)
            for out, spectrum in zip((cos, sin), spectra, strict=True):
                out[m, n] = np.fft.ifft(spectrum * kernel).real[start : start + z.size]
    return Gradients(z.copy(), float(radius_m), cos, sin, tails)


def reconstruct(
    gradients: Gradients, x: np.ndarray, y: np.ndarray, plane: int, l_max: int | None = None
) -> np.ndarray:
    """B, shape ``(points, 3)``, at transverse points ``(x, y)`` in plane ``plane``, from the
    scalar potential truncated at ``l <= l_max`` (default: all derivatives available)."""
    l_top = (gradients.n_max - 1) // 2 if l_max is None else l_max
    if 2 * l_top + 1 > gradients.n_max:
        raise ValueError(f"l_max = {l_top} needs derivatives up to {2 * l_top + 1}")
    rho, phi = np.hypot(x, y), np.arctan2(y, x)
    b_rho, b_phi, b_z = (np.zeros_like(rho) for _ in range(3))
    for m in range(gradients.m_max + 1):
        c_m, s_m = np.cos(m * phi), np.sin(m * phi)
        for ell in range(l_top + 1):
            a = (
                (-1) ** ell
                * math.factorial(m)
                / (4**ell * math.factorial(ell) * math.factorial(m + ell))
            )
            power = m + 2 * ell
            cc, cs = gradients.cos[m, 2 * ell, plane], gradients.sin[m, 2 * ell, plane]
            angular = cc * c_m + cs * s_m
            if power > 0:
                lower = rho ** (power - 1)
                b_rho += a * power * lower * angular
                b_phi += a * m * lower * (cs * c_m - cc * s_m)
            dc, ds = gradients.cos[m, 2 * ell + 1, plane], gradients.sin[m, 2 * ell + 1, plane]
            b_z += a * rho**power * (dc * c_m + ds * s_m)
    return np.stack(
        (b_rho * np.cos(phi) - b_phi * np.sin(phi), b_rho * np.sin(phi) + b_phi * np.cos(phi), b_z),
        axis=-1,
    )


def _integrated(gradients: Gradients) -> list[dict[str, Any]]:
    """Integrated normal and skew ``b_n`` in T*m^(2 - n), and at the cylinder radius
    (``b_n R^(n - 1)``, T*m for every n), where harmonics can be compared."""
    z, radius = gradients.z_m, gradients.radius_m
    rows = []
    for m in range(1, gradients.m_max + 1):
        normal = m * float(_integral(gradients.sin[m, 0], z))
        skew = m * float(_integral(gradients.cos[m, 0], z))
        at_radius = radius ** (m - 1)
        rows.append(
            {
                "n": m,
                "normal": normal,
                "skew": skew,
                "normal_at_radius_t_m": normal * at_radius,
                "skew_at_radius_t_m": skew * at_radius,
            }
        )
    return rows


def _size(row: dict[str, Any]) -> float:
    return math.hypot(row["normal_at_radius_t_m"], row["skew_at_radius_t_m"])


def _interior_misfit(field: FieldMap, gradients: Gradients, l_max: int) -> dict[str, Any]:
    """The map against the field rebuilt from the surface data, at the nodes inside the
    cylinder, in shells of rho; and how much the rebuilt field moves from l_max - 1 to l_max."""
    X, Y = np.meshgrid(field.x_m, field.y_m, indexing="ij")
    rho = np.hypot(X, Y)
    inside = rho < gradients.radius_m * (1 - 1e-12)
    xs, ys = X[inside], Y[inside]
    shells = [0.5, 0.75, 1.0]
    sums = {s: [0.0, 0.0, 0, 0.0] for s in shells}  # misfit^2, field^2, count, max misfit
    change = 0.0
    for plane in range(field.z_m.size):
        rebuilt = reconstruct(gradients, xs, ys, plane, l_max)
        if l_max > 0:
            lower = reconstruct(gradients, xs, ys, plane, l_max - 1)
            change = max(change, float(np.max(np.linalg.norm(rebuilt - lower, axis=-1))))
        data = field.b_t[:, :, plane][inside]
        misfit = np.linalg.norm(rebuilt - data, axis=-1)
        magnitude = np.linalg.norm(data, axis=-1)
        r = rho[inside] / gradients.radius_m
        for s in shells:
            sel = r <= s
            sums[s][0] += float(np.sum(misfit[sel] ** 2))
            sums[s][1] += float(np.sum(magnitude[sel] ** 2))
            sums[s][2] += int(sel.sum())
            sums[s][3] = max(sums[s][3], float(misfit[sel].max(initial=0.0)))
    return {
        "shells": [
            {
                "rho_over_radius_up_to": s,
                "nodes": sums[s][2],
                "rms_misfit_t": math.sqrt(sums[s][0] / sums[s][2]) if sums[s][2] else None,
                "rms_misfit_relative": math.sqrt(sums[s][0] / sums[s][1]) if sums[s][1] else None,
                "max_misfit_t": sums[s][3],
            }
            for s in shells
        ],
        "max_change_from_l_max_minus_1_t": change,
    }


def generalized_gradients(
    field: FieldMap,
    radius_m: float,
    m_max: int = 12,
    l_max: int = 6,
    n_phi: int | None = None,
    interpolation: str = "linear",
) -> dict[str, Any]:
    """Generalized gradients from ``B_rho`` on the cylinder of ``radius_m``: integrated
    multipoles, the tails used, how much the integrals move with the other sampling, and
    the map's interior against the field the surface data imply."""
    gradients = fit_gradients(field, radius_m, m_max, l_max, n_phi, interpolation)
    integrated = _integrated(gradients)
    main = max(integrated, key=_size)
    scale = _size(main)
    for row in integrated:
        row["relative_to_main"] = _size(row) / scale
    other = "cubic" if interpolation == "linear" else "linear"
    sampling: float | None
    try:
        alternative = _integrated(fit_gradients(field, radius_m, m_max, l_max, n_phi, other))
        sampling = max(
            math.hypot(
                a["normal_at_radius_t_m"] - b["normal_at_radius_t_m"],
                a["skew_at_radius_t_m"] - b["skew_at_radius_t_m"],
            )
            / scale
            for a, b in zip(integrated, alternative, strict=True)
        )
    except ValueError:  # the other sampling does not fit this grid
        sampling = None
    n = main["n"]
    skew_main = abs(main["skew"]) > abs(main["normal"])
    profile = n * (gradients.cos[n, 0] if skew_main else gradients.sin[n, 0])
    total = float(_integral(np.abs(profile), field.z_m))
    tail_share: dict[str, float] = {}
    for name, end in (("low_z", 0), ("high_z", -1)):
        length = float(gradients.tails[name]["decay_length_m"])
        tail_share[name] = abs(float(profile[end])) * length / total if total > 0 else 0.0
    return {
        "method": (
            "Venturini-Dragt generalized gradients from B_rho sampled on a cylinder; Fourier "
            "kernel k^(m-1) / I'_m(kR) per angular harmonic; exponential tails beyond the ends"
        ),
        "radius_m": radius_m,
        "m_max": m_max,
        "l_max": l_max,
        "n_phi": max(128, 4 * (m_max + 1)) if n_phi is None else n_phi,
        "interpolation": interpolation,
        "tails": {**gradients.tails, "share_of_main_integral": tail_share},
        "main_harmonic_n": main["n"],
        "integrated_b_n": integrated,
        "units": "integrated b_n in T*m^(2 - n): T*m dipole, T quadrupole, T/m sextupole",
        "sampling_sensitivity_relative_to_main": sampling,
        "interior_misfit": _interior_misfit(field, gradients, l_max),
        "reading": (
            "interior_misfit compares the map with the vacuum field its own cylinder surface "
            "implies: near the noise floor where the data are a vacuum field, larger where the "
            "interior holds wrong values or the series is not converged (see "
            "max_change_from_l_max_minus_1_t). sampling_sensitivity bounds the error that "
            "interpolating onto the cylinder adds to the integrals."
        ),
    }
