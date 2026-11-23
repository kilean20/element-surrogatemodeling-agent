"""Tests for multipoles, profiles and mirror symmetry, on fields with known answers."""

import numpy as np
import pytest

from element_surrogate.analysis import field_frame, harmonics, mirror_symmetry
from element_surrogate.bend import NominalBend
from element_surrogate.fields import FieldMap
from element_surrogate.region import Region
from element_surrogate.synthetic import bend_sources, multipole_sources, point_source_field

AXIS = np.linspace(-0.02, 0.02, 21)
Z = np.linspace(-0.3, 0.3, 121)
FIT = Region("circle", 0.015)


@pytest.fixture(scope="module")
def quad():
    return point_source_field(multipole_sources(2, 0.05, 0.2, 1e-3, 41), AXIS, AXIS, Z)


@pytest.fixture(scope="module")
def dipole():
    return point_source_field(multipole_sources(1, 0.05, 0.2, 1e-3, 41), AXIS, AXIS, Z)


def _harmonic(result, n):
    return next(h for h in result["integrated_relative_to_main"] if h["n"] == n)


def _two_d_field(coefficients, reference_radius_m):
    """An exact z-independent 2D vacuum field By + i Bx = sum_n C_n (w / R)^(n - 1)."""
    axis = np.linspace(-0.02, 0.02, 41)
    X, Y = np.meshgrid(axis, axis, indexing="ij")
    w = (X + 1j * Y) / reference_radius_m
    f = sum(c * w ** (n - 1) for n, c in coefficients.items())
    b = np.stack((f.imag, f.real, np.zeros_like(f.real)), axis=-1)[:, :, None, :].repeat(3, axis=2)
    return FieldMap(axis, axis, np.array([-0.01, 0.0, 0.01]), b)


def test_quadrupole_sources_are_a_normal_quadrupole(quad):
    result = harmonics(quad, FIT, reference_radius_m=0.01)
    assert result["main_harmonic_n"] == 2
    assert abs(_harmonic(result, 2)["normal"]) == pytest.approx(1.0)
    assert abs(_harmonic(result, 2)["skew"]) < 1e-9
    # Four-fold symmetry allows n = 6 but forbids n = 3 and n = 4.
    assert abs(_harmonic(result, 3)["normal"]) < 1e-9
    assert abs(_harmonic(result, 4)["normal"]) < 1e-9
    assert abs(_harmonic(result, 6)["normal"]) > 1e-6


def test_dipole_sources_are_a_normal_dipole(dipole):
    result = harmonics(dipole, FIT, reference_radius_m=0.01)
    assert result["main_harmonic_n"] == 1
    assert abs(_harmonic(result, 1)["skew"]) < 1e-9
    assert abs(_harmonic(result, 2)["normal"]) < 1e-9  # up-down symmetric: no quadrupole


@pytest.mark.parametrize("order", [4, 8, 12])
def test_fit_recovers_known_coefficients_at_any_order(order):
    """The basis in units of the fit radius keeps the fit well conditioned at high order."""
    reference = 0.01
    true = {n: (1e-3 if n == 2 else 1e-6) * (1 + 0.5j) for n in range(1, order + 2)}
    result = harmonics(_two_d_field(true, reference), Region("circle", 0.02), reference, order)
    span = 0.02
    for h in result["integrated_t_m"]:
        got = complex(h["normal"], h["skew"]) / span
        assert abs(got - true[h["n"]]) < 1e-8 * abs(true[h["n"]])


def test_order_sensitivity_flags_an_unresolved_fit():
    """Fitting fewer orders than the field holds moves the coefficients, and the report says so."""
    reference = 0.01
    true = {n: (1e-3 if n == 2 else 1e-6) for n in range(1, 12)}
    field = _two_d_field(true, reference)
    low = harmonics(field, Region("circle", 0.02), reference, order=6)
    high = harmonics(field, Region("circle", 0.02), reference, order=10)
    assert low["order_sensitivity"]["max_change_relative_to_main"] > 1e-4
    assert high["order_sensitivity"]["max_change_relative_to_main"] < 1e-9


def test_profile_of_a_complete_map(quad):
    profile = harmonics(quad, FIT, reference_radius_m=0.01)["profile"]
    assert profile["centroid_z_m"] == pytest.approx(0.0, abs=1e-9)
    assert 0.15 < profile["map_profile_length_m"] < 0.35
    assert max(abs(f) for f in profile["edge_fraction"]) < 0.05


def test_truncated_map_shows_at_its_edges(quad):
    """The same field cut inside the magnet, through the region's z range."""
    cut = Region("circle", 0.015, z_min_m=-0.08, z_max_m=0.08)
    profile = harmonics(quad, cut, reference_radius_m=0.01)["profile"]
    assert min(abs(f) for f in profile["edge_fraction"]) > 0.5


def test_square_and_circle_fits_agree_once_the_orders_are_resolved(quad):
    circle = harmonics(quad, Region("circle", 0.012), 0.01, order=10)
    square = harmonics(quad, Region("square", 0.012), 0.01, order=10)
    assert _harmonic(square, 6)["normal"] == pytest.approx(_harmonic(circle, 6)["normal"], rel=1e-3)


def test_order_sensitivity_reaches_the_next_allowed_harmonic(quad):
    """A quadrupole's next harmonic after n = 6 is n = 10, four orders up; the check must see it."""
    unresolved = harmonics(quad, Region("square", 0.012), 0.01, order=6)
    change = unresolved["order_sensitivity"]["change_by_n"][5]  # n = 6
    assert change > 1e-3 * abs(_harmonic(unresolved, 6)["normal"])


def test_quadrupole_parities(quad):
    sym = mirror_symmetry(quad)
    x, y, z = sym["x_mirror"], sym["y_mirror"], sym["z_mirror"]
    assert x["By"]["odd_defect"] < 1e-9 and y["By"]["even_defect"] < 1e-9
    assert x["Bx"]["even_defect"] < 1e-9 and y["Bx"]["odd_defect"] < 1e-9
    assert z["By"]["even_defect"] < 1e-9 and z["Bz"]["odd_defect"] < 1e-9


def test_a_real_asymmetry_is_measured_not_hidden(quad):
    """A small skew dipole breaks the up-down parity of Bx by a measurable amount."""
    b = quad.b_t.copy()
    b[..., 0] += 1e-2 * np.abs(quad.b_t[..., 0]).max()
    sym = mirror_symmetry(quad.with_values(b))
    assert sym["y_mirror"]["Bx"]["odd_defect"] > 1e-3
    assert sym["x_mirror"]["Bx"]["even_defect"] < 1e-9  # the other parity survives


def test_parity_inside_a_region_ignores_defects_outside_it(quad):
    b = quad.b_t.copy()
    b[-1, -1, :, :] = 0.0  # a masked corner column
    defective = quad.with_values(b)
    whole = mirror_symmetry(defective)["x_mirror"]["By"]["odd_defect"]
    inside = mirror_symmetry(defective, Region("circle", 0.015))["x_mirror"]["By"]["odd_defect"]
    assert whole > 1e-3 and inside < 1e-9


def test_mirrors_that_the_grid_or_region_breaks_are_skipped(quad):
    keep = quad.x_m >= -0.01
    shifted = FieldMap(quad.x_m[keep], quad.y_m, quad.z_m, quad.b_t[keep])
    assert mirror_symmetry(shifted)["x_mirror"] is None
    lopsided = Region("circle", 0.01, z_min_m=-0.1, z_max_m=0.2)
    assert mirror_symmetry(quad, lopsided)["z_mirror"] is None


def test_fit_region_must_hold_enough_nodes(quad):
    with pytest.raises(ValueError, match="nodes per plane"):
        harmonics(quad, Region("circle", 1e-4), 0.01)


def _bend_field(toward="-x"):
    bend = NominalBend(2.0, 0.3, 0.0, bends_toward_negative_x=(toward == "-x"))
    sources = bend_sources(bend, 0.02, 0.04, 1.0e-3, 80, 9)
    x = np.linspace(-0.16, 0.16, 81)
    return point_source_field(sources, x, np.linspace(-0.01, 0.01, 5), np.linspace(-0.5, 0.5, 101))


@pytest.mark.parametrize("toward", ["-x", "+x"])
def test_frame_of_a_band_that_curves_in_the_grid(toward):
    """Poles along a 2 m, 0.3 rad arc in a Cartesian grid: the band centre is a circle. It is
    not the pole arc itself, so radius and angle are found to about a percent."""
    result = field_frame(_bend_field(toward))
    assert result["component"] == "By"
    assert result["ranked_by_residual"][-1] == "straight"
    circle = result["readings"]["circle"]
    assert circle["radius_m"] == pytest.approx(2.0, rel=0.01)
    assert circle["rms_residual_m"] < 0.01 * result["readings"]["straight"]["rms_residual_m"]
    bend = result["proposed_bend"]
    assert bend["bends_toward"] == toward
    assert bend["angle_rad"] == pytest.approx(0.3, rel=0.02)
    reach = bend["radius_m"] * np.sin(bend["angle_rad"] / 2) + bend["arm_m"] * np.cos(
        bend["angle_rad"] / 2
    )
    assert reach == pytest.approx(0.5)  # the nominal path reaches the grid's z ends


def test_frame_of_a_straight_band(dipole):
    wide = point_source_field(
        multipole_sources(1, 0.03, 0.2, 1e-3, 41),
        np.linspace(-0.1, 0.1, 41),
        np.linspace(-0.01, 0.01, 5),
        Z,
    )
    result = field_frame(wide)
    assert result["readings"]["straight"]["rms_residual_m"] < 1e-12
    assert result["proposed_bend"] is None


def test_frame_needs_a_band_inside_the_grid(dipole):
    with pytest.raises(ValueError, match="band"):
        field_frame(dipole)  # the dipole's band is wider than this grid
    with pytest.raises(ValueError, match="core_fraction"):
        field_frame(dipole, core_fraction=0.0)
    with pytest.raises(ValueError, match="component"):
        field_frame(dipole, component="Bz")


def test_frame_when_the_band_runs_off_the_grid():
    """A grid that ends inside the band near the middle: those planes have no centre, the fit
    uses the others, and the body field still comes from the middle plane."""
    full = _bend_field()
    # nodes up to x = 48 mm: the band's edge passes them for |z| below about 45 mm
    cut = full.window(slice(0, int(np.searchsorted(full.x_m, 0.0481))), slice(None), slice(None))
    result = field_frame(cut)
    assert result["body_planes_with_band_off_the_grid"] > 0
    assert result["band"]["centre_nearest_middle"]["z_m"] != 0.0
    assert result["readings"]["circle"]["radius_m"] == pytest.approx(2.0, rel=0.02)
    whole = field_frame(full)["proposed_bend"]
    assert result["proposed_bend"]["body_field_t"] == pytest.approx(whole["body_field_t"])
