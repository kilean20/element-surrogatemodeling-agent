"""Tests for the nominal bend path: geometry with exact answers."""

import math

import numpy as np
import pytest

from element_surrogate.bend import NominalBend

# Test geometry only; not any real magnet.
RADIUS, ANGLE, ARM = 2.0, math.radians(30.0), 0.3


@pytest.fixture(params=[True, False], ids=["toward -x", "toward +x"])
def bend(request):
    return NominalBend(RADIUS, ANGLE, ARM, bends_toward_negative_x=request.param)


def test_lengths(bend):
    assert bend.arc_length_m == pytest.approx(RADIUS * ANGLE)
    assert bend.length_m == pytest.approx(RADIUS * ANGLE + 2 * ARM)
    same = NominalBend.from_path_length(RADIUS, ANGLE, bend.length_m, bend.bends_toward_negative_x)
    assert same.arm_m == pytest.approx(ARM)


def test_midpoint_is_the_origin_and_the_chord_is_z(bend):
    mid = bend.frame_at(bend.length_m / 2)
    np.testing.assert_allclose(mid.origin_m, 0.0, atol=1e-12)
    np.testing.assert_allclose(mid.tangent, [0.0, 0.0, 1.0], atol=1e-12)


def test_entry_and_exit_directions_make_half_the_angle_with_z(bend):
    for frame in (bend.entry_frame(), bend.exit_frame()):
        assert math.acos(frame.tangent[2]) == pytest.approx(ANGLE / 2)
    # The bend turns the beam toward -x (or +x).
    turn = bend.exit_frame().tangent[0] - bend.entry_frame().tangent[0]
    assert (turn < 0) == bend.bends_toward_negative_x


def test_frames_are_orthonormal_right_handed_and_vertical_is_up(bend):
    for s in np.linspace(0.0, bend.length_m, 23):
        f = bend.frame_at(s)
        basis = np.stack([f.horizontal, f.vertical, f.tangent])
        np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.cross(f.horizontal, f.vertical), f.tangent, atol=1e-12)
        np.testing.assert_allclose(f.vertical, [0.0, 1.0, 0.0], atol=1e-12)


def test_path_is_continuous_and_has_the_right_length(bend):
    s = np.linspace(0.0, bend.length_m, 4001)
    points = np.array([bend.frame_at(v).origin_m for v in s])
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert steps.sum() == pytest.approx(bend.length_m, rel=1e-6)
    assert steps.max() < 2 * (s[1] - s[0])


def test_path_is_mirror_symmetric_about_its_midpoint(bend):
    for s in np.linspace(0.0, bend.length_m / 2, 7):
        a, b = bend.frame_at(s).origin_m, bend.frame_at(bend.length_m - s).origin_m
        np.testing.assert_allclose([a[0], a[1], -a[2]], b, atol=1e-12)


def test_path_coordinates_recover_offsets_everywhere(bend):
    rng = np.random.default_rng(5)
    for s in np.r_[0.0, ARM * 0.5, ARM, rng.uniform(0, bend.length_m, 10), bend.length_m]:
        f = bend.frame_at(s)
        for h, v in ((0.0, 0.0), (0.02, -0.01), (-0.03, 0.015)):
            point = f.origin_m + h * f.horizontal + v * f.vertical
            np.testing.assert_allclose(bend.path_coordinates(point), [s, h, v], atol=1e-12)


def test_points_before_and_after_the_path(bend):
    entry, exit_ = bend.entry_frame(), bend.exit_frame()
    before = entry.origin_m - 0.1 * entry.tangent + 0.01 * entry.horizontal
    after = exit_.origin_m + 0.2 * exit_.tangent
    np.testing.assert_allclose(bend.path_coordinates(before), [-0.1, 0.01, 0.0], atol=1e-12)
    np.testing.assert_allclose(
        bend.path_coordinates(after), [bend.length_m + 0.2, 0.0, 0.0], atol=1e-12
    )


def test_horizontal_axis_points_away_from_the_centre_only_for_a_bend_toward_minus_x(bend):
    mid = bend.frame_at(bend.length_m / 2)
    centre_x = -RADIUS if bend.bends_toward_negative_x else RADIUS
    away = mid.horizontal @ (mid.origin_m - np.array([centre_x, 0.0, 0.0])) > 0
    assert away == bend.bends_toward_negative_x


def test_invalid_geometry_is_refused():
    with pytest.raises(ValueError):
        NominalBend(-1.0, ANGLE, ARM)
    with pytest.raises(ValueError):
        NominalBend(RADIUS, math.pi, ARM)
    with pytest.raises(ValueError):
        NominalBend.from_path_length(RADIUS, ANGLE, RADIUS * ANGLE - 0.1)
    with pytest.raises(ValueError):
        NominalBend(RADIUS, ANGLE, ARM).frame_at(-0.1)
