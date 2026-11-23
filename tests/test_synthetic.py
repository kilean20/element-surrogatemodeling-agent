"""Tests for the synthetic fields themselves."""

import numpy as np
import pytest

from element_surrogate.synthetic import multipole_sources, point_source_field


def _jacobian(sources, point, h=1e-6):
    """Central-difference Jacobian dB_i/dx_j of the point-source field at one point."""
    jac = np.zeros((3, 3))
    for j in range(3):
        offset = np.zeros(3)
        offset[j] = h
        axes_p = [np.array([point[k] + offset[k]]) for k in range(3)]
        axes_m = [np.array([point[k] - offset[k]]) for k in range(3)]
        bp = _field_at(sources, axes_p)
        bm = _field_at(sources, axes_m)
        jac[:, j] = (bp - bm) / (2 * h)
    return jac


def _field_at(sources, axes):
    # A one-node grid needs two nodes per axis; evaluate on a tiny grid and take the first node.
    grid = [np.array([a[0], a[0] + 1e-3]) for a in axes]
    return point_source_field(sources, *grid).b_t[0, 0, 0]


def test_point_source_field_is_divergence_and_curl_free():
    sources = multipole_sources(2, 0.05, 0.2, 1e-3, 9)
    rng = np.random.default_rng(3)
    for point in rng.uniform([-0.02, -0.02, -0.15], [0.02, 0.02, 0.15], size=(5, 3)):
        jac = _jacobian(sources, point)
        scale = np.abs(jac).max()
        assert abs(np.trace(jac)) < 1e-6 * scale
        assert np.abs(jac - jac.T).max() < 1e-6 * scale


def test_quadrupole_sources_give_a_quadrupole_at_the_centre():
    axis = np.linspace(-0.004, 0.004, 3)
    z = np.array([-0.001, 0.001])
    field = point_source_field(multipole_sources(2, 0.05, 0.2, 1e-3, 41), axis, axis, z)
    b = field.b_t[:, :, 0]
    assert abs(b[1, 1]).max() < 1e-12  # zero on the axis
    gx = (b[2, 1, 1] - b[0, 1, 1]) / 0.008  # dBy/dx
    gy = (b[1, 2, 0] - b[1, 0, 0]) / 0.008  # dBx/dy
    assert gx == pytest.approx(gy, rel=1e-3)


def test_dipole_sources_give_a_uniform_field_at_the_centre():
    axis = np.linspace(-0.002, 0.002, 3)
    z = np.array([-0.001, 0.001])
    field = point_source_field(multipole_sources(1, 0.05, 0.2, 1e-3, 41), axis, axis, z)
    centre = field.b_t[1, 1, 0]
    assert abs(centre[1]) > 1e3 * max(abs(centre[0]), abs(centre[2]))


def test_source_inside_the_grid_is_rejected():
    axis = np.linspace(-0.1, 0.1, 3)
    with pytest.raises(ValueError, match="inside the grid"):
        point_source_field([((0.0, 0.05, 0.0), 1.0)], axis, axis, axis)


def test_multipole_sources_validate_their_inputs():
    with pytest.raises(ValueError):
        multipole_sources(0, 0.05, 0.2, 1e-3, 5)
