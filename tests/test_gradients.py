"""Tests for generalized gradients, on point-source fields whose interior is an exact vacuum
field determined by the cylinder surface."""

import numpy as np
import pytest

from element_surrogate.analysis import harmonics
from element_surrogate.fields import FieldMap
from element_surrogate.gradients import fit_gradients, generalized_gradients, reconstruct
from element_surrogate.region import Region
from element_surrogate.synthetic import multipole_sources, point_source_field

AXIS = np.linspace(-0.02, 0.02, 21)
Z = np.linspace(-0.4, 0.4, 161)
RADIUS = 0.015


@pytest.fixture(scope="module")
def quad():
    return point_source_field(multipole_sources(2, 0.05, 0.2, 1e-3, 41), AXIS, AXIS, Z)


@pytest.fixture(scope="module")
def report(quad):
    return generalized_gradients(quad, RADIUS, interpolation="cubic")


def test_integrated_gradient_matches_the_multipole_fit(quad, report):
    fit = harmonics(quad, Region("circle", RADIUS), RADIUS)
    gradient = fit["integrated_t_m"][1]["normal"] / RADIUS
    assert report["main_harmonic_n"] == 2
    assert report["integrated_b_n"][1]["normal"] == pytest.approx(gradient, rel=1e-4)
    assert report["integrated_b_n"][1]["relative_to_main"] == 1.0
    # a quadrupole of four rows has only the allowed harmonics 2, 6, 10
    allowed = {2, 6, 10}
    assert all(
        r["relative_to_main"] < 1e-6 for r in report["integrated_b_n"] if r["n"] not in allowed
    )


def test_interior_is_rebuilt_from_the_surface(report):
    shells = report["interior_misfit"]["shells"]
    assert shells[0]["rms_misfit_relative"] < 2e-5  # near the axis: interpolation error only
    assert all(s["nodes"] > 0 for s in shells)
    assert report["interior_misfit"]["max_change_from_l_max_minus_1_t"] < 1e-8
    assert report["sampling_sensitivity_relative_to_main"] < 1e-3
    assert report["tails"]["low_z"]["decaying"] and report["tails"]["high_z"]["decaying"]


def test_a_wrong_interior_node_stands_out(quad):
    b = quad.b_t.copy()
    b[10, 12, 80] += np.array([0.0, 0.01, 0.0])  # inside the cylinder, off its surface
    report = generalized_gradients(quad.with_values(b), RADIUS, interpolation="cubic")
    assert report["interior_misfit"]["shells"][0]["max_misfit_t"] == pytest.approx(0.01, rel=0.01)


def test_reconstruction_on_the_axis_is_the_gradient_field(quad):
    gradients = fit_gradients(quad, RADIUS, interpolation="cubic")
    x = np.array([0.002, 0.0])
    y = np.array([0.0, 0.002])
    field = reconstruct(gradients, x, y, 80)
    exact = quad.b_t[[11, 10], [10, 11], 80]
    np.testing.assert_allclose(field, exact, atol=1e-6)


def test_a_fixed_tail_reproduces_a_computation_that_fixed_it(quad):
    fixed = fit_gradients(quad, RADIUS, decay_length_m=0.05)
    assert fixed.tails["low_z"] == {"decaying": None, "decay_length_m": 0.05}
    measured = fit_gradients(quad, RADIUS)
    assert measured.tails["low_z"]["decay_length_m"] > 0
    # the tail model matters only near the ends: the kernel is nonlocal over a few radii
    body = np.abs(Z) < 0.3
    peak = np.abs(measured.sin[2, 0]).max()
    np.testing.assert_allclose(fixed.sin[2, 0][body], measured.sin[2, 0][body], atol=1e-6 * peak)


def test_arguments_are_checked(quad):
    with pytest.raises(ValueError, match="radius_m"):
        fit_gradients(quad, 0.02)  # the circle must stay inside the grid
    with pytest.raises(ValueError, match="n_phi"):
        fit_gradients(quad, RADIUS, m_max=12, n_phi=16)
    uneven = FieldMap(AXIS, AXIS, Z**3 / Z.max() ** 2 + Z, quad.b_t)
    with pytest.raises(ValueError, match="uniform grid in z"):
        fit_gradients(uneven, RADIUS)
    with pytest.raises(ValueError, match="interpolation"):
        fit_gradients(quad, RADIUS, interpolation="spline")
