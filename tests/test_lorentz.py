"""Tests for the Lorentz-force tracker and reference orbits, against exact solutions."""

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from element_surrogate.bend import NominalBend, PathFrame
from element_surrogate.fields import FieldMap
from element_surrogate.lorentz import (
    Status,
    UniformField,
    step_convergence,
    straight_frame,
    to_lab,
    to_plane,
    track,
)
from element_surrogate.reference import fit_reference, trace_reference
from element_surrogate.region import Region

ENTRY, EXIT = straight_frame(0.0), straight_frame(1.0)
STATES = np.array(
    [[0.0, 0.0, 0.0, 0.0], [0.002, 0.003, -0.001, 0.002], [-0.003, -0.001, 0.002, -0.004]]
)


def _exact_helix(b, rigidity, r0, t0):
    """Position and direction at path length s in a uniform field: t rotates about b by
    -|b| s / rigidity."""
    magnitude = np.linalg.norm(b)
    k = b / magnitude
    kappa = -magnitude / rigidity
    along = (t0 @ k) * k
    across = t0 - along
    side = np.cross(k, across)

    def at(s):
        c, w = math.cos(kappa * s), math.sin(kappa * s)
        r = r0 + along * s + across * w / kappa + side * (1 - c) / kappa
        return r, along + across * c + side * w

    return at


@pytest.mark.parametrize(
    "b", [(0.0, 0.2, 0.0), (0.15, 0.0, 0.0), (0.05, -0.1, 0.3), (0.1, 0.1, -0.2)]
)
@pytest.mark.parametrize("rigidity", [1.0, -0.7])
def test_uniform_fields_give_exact_helices(b, rigidity):
    field = UniformField(b)
    result = track(field, STATES, rigidity, ENTRY, EXIT, step_m=0.005)
    assert result.reached_exit.all()
    r0, t0 = to_lab(ENTRY, STATES)
    for i in range(len(STATES)):
        at = _exact_helix(np.array(b), rigidity, r0[i], t0[i])
        s_exit = brentq(lambda s, at=at: at(s)[0][2] - 1.0, 0.0, 2.0)
        r, t = at(s_exit)
        exact = to_plane(EXIT, r[None], t[None])[0]
        np.testing.assert_allclose(result.states[i], exact, rtol=0, atol=1e-11)
        assert result.path_length_m[i] == pytest.approx(s_exit, abs=1e-11)
    assert result.max_direction_norm_error.max() < 1e-12


def test_a_drift_is_a_straight_line():
    result = track(UniformField((0.0, 0.0, 0.0)), STATES, 1.0, ENTRY, EXIT, step_m=0.1)
    expected = STATES.copy()
    expected[:, 0] += STATES[:, 1]
    expected[:, 2] += STATES[:, 3]
    np.testing.assert_allclose(result.states, expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        result.path_length_m, np.sqrt(1 + STATES[:, 1] ** 2 + STATES[:, 3] ** 2), rtol=1e-14
    )


def test_scaling_field_and_rigidity_together_changes_nothing():
    base = track(UniformField((0.03, 0.2, 0.0)), STATES, 0.8, ENTRY, EXIT, step_m=0.01)
    scaled = track(
        UniformField((0.03, 0.2, 0.0)), STATES, 2.4, ENTRY, EXIT, step_m=0.01, field_scale=3.0
    )
    np.testing.assert_allclose(scaled.states, base.states, rtol=1e-12, atol=1e-15)


def test_every_stop_is_reported():
    axis = np.linspace(-0.01, 0.01, 5)
    small = FieldMap(axis, axis, np.linspace(-0.1, 0.5, 7), np.zeros((5, 5, 7, 3)))
    assert track(small, STATES[:1], 1.0, ENTRY, EXIT, step_m=0.01).status[0] == Status.OUTSIDE_FIELD
    tight = Region("circle", 0.001)
    wide = STATES[1:2] * [1, 10, 1, 1]
    lost = track(UniformField((0, 0, 0)), wide, 1.0, ENTRY, EXIT, step_m=0.01, region=tight)
    assert lost.status[0] == Status.OUTSIDE_REGION
    limited = track(
        UniformField((0, 0, 0)), STATES[:1], 1.0, ENTRY, EXIT, step_m=0.01, max_path_m=0.5
    )
    assert limited.status[0] == Status.PATH_LIMIT
    bad = small.with_values(np.full((5, 5, 7, 3), np.nan))
    assert (
        track(bad, STATES[:1], 1.0, ENTRY, straight_frame(0.4), step_m=0.01).status[0]
        == Status.NON_FINITE
    )
    backwards = PathFrame(ENTRY.origin_m, -ENTRY.tangent, -ENTRY.horizontal, ENTRY.vertical)
    turned = track(
        UniformField((0, 0, 0)), STATES[:1], 1.0, ENTRY, backwards, step_m=0.01, max_path_m=1.0
    )
    assert turned.status[0] == Status.WRONG_EXIT_DIRECTION
    assert set(turned.summary()) == {s.name.lower() for s in Status}


def test_arguments_are_checked():
    with pytest.raises(ValueError, match="shape"):
        track(UniformField((0, 0, 0)), np.zeros(4), 1.0, ENTRY, EXIT, step_m=0.01)
    with pytest.raises(ValueError, match="rigidity"):
        track(UniformField((0, 0, 0)), STATES, 0.0, ENTRY, EXIT, step_m=0.01)
    with pytest.raises(ValueError, match="observer"):
        track(UniformField((0, 0, 0)), STATES, 1.0, ENTRY, EXIT, step_m=0.01, observer=print)
    with pytest.raises(ValueError, match="largest first"):
        step_convergence(UniformField((0, 0, 0)), STATES, 1.0, ENTRY, EXIT, [0.01, 0.02])


def test_step_convergence_reports_the_order():
    # steps small enough to be in the asymptotic range of RK4, large enough to stay above
    # rounding
    report = step_convergence(
        UniformField((0.0, 0.3, 0.1)), STATES, 1.0, ENTRY, EXIT, [0.05, 0.025, 0.0125]
    )
    assert report["reached_at_every_step"] == 3
    orders = report["steps"][1]["observed_order"]
    assert orders[0] == pytest.approx(4.0, abs=0.5)  # RK4 in a smooth field
    assert orders[1] == pytest.approx(4.0, abs=0.5)


@pytest.mark.parametrize("toward", [True, False])
def test_a_reference_fit_in_a_uniform_field_is_exact(toward):
    """With no arms, the uniform field that bends a particle round the nominal arc has
    radius R: scale 1. With arms, the field fills them too, and the fitted circle is the one
    through the entry and exit points with the same turning angle."""
    rigidity, b = 0.5, 0.1
    field = UniformField((0.0, b if toward else -b, 0.0))
    for arm in (0.0, 0.2):
        bend = NominalBend(5.0, 0.3, arm, bends_toward_negative_x=toward)
        orbit = fit_reference(
            field,
            rigidity,
            bend.entry_frame(),
            bend.exit_frame(),
            scale_bracket=(0.5, 2.0),
            step_m=0.005,
            position_tolerance_m=1e-9,
            slope_tolerance_rad=1e-9,
        )
        chord = 2 * (5.0 * math.sin(0.15) + arm * math.cos(0.15))
        radius = chord / (2 * math.sin(0.15))
        assert orbit.field_scale == pytest.approx(rigidity / (b * radius), rel=1e-9)
        assert orbit.length_m == pytest.approx(radius * 0.3, rel=1e-9)
        departure = orbit.summary(bend)["departure_from_nominal_path"]
        assert departure["max_abs_horizontal_m"] < 1e-9 if arm == 0.0 else 1e-3


def test_reference_frames_and_reporting_on_them():
    bend = NominalBend(5.0, 0.3, 0.0)
    orbit = trace_reference(
        UniformField((0.0, 0.1, 0.0)), 0.5, bend.entry_frame(), bend.exit_frame(), step_m=0.005
    )
    for s in (0.0, 0.5 * orbit.length_m, orbit.length_m):
        f = orbit.frame_at(s)
        basis = np.stack([f.horizontal, f.vertical, f.tangent])
        np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(np.cross(f.horizontal, f.vertical), f.tangent, atol=1e-9)
    states = track(
        UniformField((0.0, 0.1, 0.0)),
        STATES,
        0.5,
        orbit.frame_at(0.0),
        orbit.frame_at(orbit.length_m),
        step_m=0.005,
    ).states
    np.testing.assert_allclose(states[0], 0.0, atol=1e-9)  # the design particle is the reference
    with pytest.raises(ValueError, match="outside the orbit"):
        orbit.frame_at(orbit.length_m + 1.0)


def test_a_fit_without_closure_is_refused():
    bend = NominalBend(5.0, 0.3, 0.0)
    tilted = UniformField((0.002, 0.1, 0.0))  # a horizontal field component bends vertically
    with pytest.raises(ValueError, match="does not close"):
        fit_reference(
            tilted,
            0.5,
            bend.entry_frame(),
            bend.exit_frame(),
            scale_bracket=(0.5, 2.0),
            step_m=0.005,
            position_tolerance_m=1e-9,
            slope_tolerance_rad=1e-9,
        )


def test_points_inside_fields_and_regions():
    axis = np.linspace(-0.01, 0.01, 3)
    grid = FieldMap(axis, axis, axis, np.zeros((3, 3, 3, 3)))
    assert grid.contains(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.02]])).tolist() == [True, False]
    bend = NominalBend(1.0, 0.4, 0.1)
    region = Region("square", 0.01, path=bend)
    on_path = bend.frame_at(0.5).origin_m
    off = on_path + 0.02 * bend.frame_at(0.5).horizontal
    assert region.contains(np.stack([on_path, off])).tolist() == [True, False]


def test_leaving_the_support_before_the_exit_is_reported():
    """A particle that leaves the grid sideways before the exit plane stops there; one that
    stays inside lands on the plane at the grid's last plane."""
    axis = np.linspace(-0.01, 0.01, 5)
    grid = FieldMap(axis, axis, np.linspace(0.0, 1.0, 11), np.zeros((5, 5, 11, 3)))
    states = np.array([[0.0, 0.0, 0.0, 0.0], [0.009, 0.002, 0.0, 0.0]])
    edge = track(grid, states, 1.0, ENTRY, EXIT, step_m=0.05)
    assert edge.status.tolist() == [Status.REACHED_EXIT, Status.OUTSIDE_FIELD]
