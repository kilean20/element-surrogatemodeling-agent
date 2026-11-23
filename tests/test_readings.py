"""Tests for the triquadratic mesh reading."""

import numpy as np
import pytest

from element_surrogate.fields import FieldMap
from element_surrogate.readings import QuadraticReading


def _field(values, x, y, z):
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return FieldMap(x, y, z, np.stack(values(X, Y, Z), axis=-1))


def _quadratic(X, Y, Z):
    """Degree at most two along each axis, with cross terms: read exactly by the stencil."""
    return (1 + X * Y - Z**2, X**2 * Y + 2 * Z, X * Y * Z + Y**2)


@pytest.mark.parametrize("centre", ["nearest", "lower"])
def test_fields_quadratic_along_each_axis_are_read_exactly(centre):
    x = np.array([-0.02, -0.011, 0.0, 0.007, 0.02])  # non-uniform
    y = np.linspace(-0.01, 0.01, 4)
    z = np.linspace(-0.1, 0.1, 6)
    reading = QuadraticReading(_field(_quadratic, x, y, z), centre=centre)
    rng = np.random.default_rng(3)
    points = np.column_stack(
        [rng.uniform(-0.02, 0.02, 50), rng.uniform(-0.01, 0.01, 50), rng.uniform(-0.1, 0.1, 50)]
    )
    expected = np.stack(_quadratic(points[:, 0], points[:, 1], points[:, 2]), axis=-1)
    np.testing.assert_allclose(reading.sample(points), expected, rtol=0, atol=1e-13)


def test_the_nearest_node_reading_jumps_half_way_and_the_tie_picks_a_side():
    axis = np.linspace(0.0, 4.0, 5)
    cubic = _field(lambda X, Y, Z: (X**3, 0 * X, 0 * X), axis, axis, axis)
    half_way = np.array([[1.5, 2.0, 2.0]])
    upper = QuadraticReading(cubic, tie="upper").sample(half_way)[0, 0]
    lower = QuadraticReading(cubic, tie="lower").sample(half_way)[0, 0]
    assert upper != lower  # a cubic is not read exactly: the two stencils differ
    below = QuadraticReading(cubic).sample(half_way - [[1e-9, 0, 0]])[0, 0]
    assert below == pytest.approx(lower, abs=1e-6)


def test_arguments_are_checked():
    axis = np.linspace(0.0, 1.0, 3)
    field = _field(lambda X, Y, Z: (X, Y, Z), axis, axis, axis)
    with pytest.raises(ValueError, match="centre"):
        QuadraticReading(field, centre="middle")
    with pytest.raises(ValueError, match="outside"):
        QuadraticReading(field).sample(np.array([[2.0, 0.0, 0.0]]))
