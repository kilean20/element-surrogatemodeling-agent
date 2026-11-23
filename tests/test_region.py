"""Tests for regions: one definition of 'where' shared by every check and fit."""

import numpy as np
import pytest

from element_surrogate.bend import NominalBend
from element_surrogate.fields import FieldMap
from element_surrogate.region import Region, region_for


def _grid(x, y=None, z=None):
    y = x if y is None else y
    z = np.linspace(-0.1, 0.1, 5) if z is None else z
    return FieldMap(x, y, z, np.zeros((x.size, y.size, z.size, 3)))


def test_covering_region_is_the_largest_inside_the_grid():
    field = _grid(np.linspace(-0.02, 0.03, 6), np.linspace(-0.025, 0.02, 10))
    assert Region.covering(field).radius_m == pytest.approx(0.02)
    assert Region.covering(field, "square").shape == "square"


def test_grid_that_does_not_straddle_the_axis_is_refused():
    field = _grid(np.linspace(0.001, 0.02, 5))
    with pytest.raises(ValueError, match="both sides of the z axis"):
        Region.covering(field)
    region = region_for(field, radius_m=0.01)  # an explicit region is still allowed
    assert region.radius_m == 0.01


def test_radial_measure_defines_the_shape():
    circle, square = Region("circle", 1.0), Region("square", 1.0)
    assert circle.radial(np.array(0.6), np.array(0.8)) == pytest.approx(1.0)
    assert square.radial(np.array(0.6), np.array(0.8)) == pytest.approx(0.8)


def test_cell_mask_requires_the_whole_cross_section_and_z_faces_inside():
    axis = np.linspace(-0.02, 0.02, 5)  # cells 1 cm wide
    field = _grid(axis, z=np.linspace(-0.1, 0.1, 5))
    region = Region("square", 0.01, z_min_m=-0.05, z_max_m=0.1)
    mask = region.cell_mask(field)
    assert mask.shape == (4, 4, 4)
    assert mask[1:3, 1:3, 1:].all() and not mask[0].any() and not mask[:, :, 0].any()


def test_node_mask_and_z_symmetry():
    field = _grid(np.linspace(-0.02, 0.02, 5))
    region = Region("circle", 0.015, -0.05, 0.05)
    assert region.node_mask(field).sum() == 9 * 3
    assert all(region.mirror_symmetric(axis) for axis in range(3))
    assert not Region("circle", 0.015, -0.05, 0.1).mirror_symmetric(2)


def test_invalid_regions_are_refused():
    with pytest.raises(ValueError):
        Region("hexagon", 0.01)
    with pytest.raises(ValueError):
        Region("circle", 0.0)
    with pytest.raises(ValueError):
        Region("circle", 0.01, z_min_m=0.1, z_max_m=0.0)


def test_rectangle_measure_scales_the_vertical_offset():
    rectangle = Region("rectangle", 0.04, half_height_m=0.01)
    assert rectangle.radial(np.array(0.03), np.array(0.01)) == pytest.approx(0.04)
    assert rectangle.radial(np.array(0.02), np.array(0.0)) == pytest.approx(0.02)
    field = _grid(np.linspace(-0.05, 0.05, 11), np.linspace(-0.02, 0.02, 9))
    mask = rectangle.node_mask(field)[:, :, 0]
    assert mask.sum() == 9 * 5  # |x| <= 0.04 (9 nodes), |y| <= 0.01 (5 nodes)
    with pytest.raises(ValueError, match="only for a rectangle"):
        Region("circle", 0.01, half_height_m=0.01)
    with pytest.raises(ValueError, match="only for a rectangle"):
        Region("rectangle", 0.01)
    with pytest.raises(ValueError, match="sized from the grid"):
        Region.covering(field, "rectangle")


BEND = {"radius_m": 1.0, "angle_rad": 0.4, "arm_m": 0.1}


@pytest.mark.parametrize("toward", ["-x", "+x"])
def test_region_about_a_bend_uses_path_coordinates(toward):
    field = _grid(
        np.linspace(-0.12, 0.12, 25), np.linspace(-0.01, 0.01, 5), np.linspace(-0.3, 0.3, 31)
    )
    region = region_for(field, "square", 0.015, bend={**BEND, "bends_toward": toward})
    mask = region.node_mask(field)
    X, Y, Z = np.meshgrid(field.x_m, field.y_m, field.z_m, indexing="ij")
    s_h_v = region.path.path_coordinates(np.stack([X, Y, Z], axis=-1))
    expected = np.maximum(np.abs(s_h_v[..., 1]), np.abs(s_h_v[..., 2])) <= 0.015 * (1 + 1e-12)
    np.testing.assert_array_equal(mask, expected)
    # the region follows the arc: at the arc's ends its centre is off the z axis
    end = np.argmin(np.abs(field.z_m - np.sin(0.2)))
    centre_x = field.x_m[mask[:, 2, end]].mean()
    assert centre_x == pytest.approx((np.cos(0.2) - 1.0) * (1 if toward == "-x" else -1), abs=0.01)


def test_cells_about_a_bend_need_all_corners_inside():
    field = _grid(
        np.linspace(-0.12, 0.12, 25), np.linspace(-0.01, 0.01, 5), np.linspace(-0.3, 0.3, 31)
    )
    region = region_for(field, "circle", 0.02, z_min_m=-0.1, z_max_m=0.1, bend=BEND)
    nodes = region.node_mask(field)
    n, m, k = (size - 1 for size in nodes.shape)
    corners = [
        nodes[a : a + n, b : b + m, c : c + k] for a in (0, 1) for b in (0, 1) for c in (0, 1)
    ]
    np.testing.assert_array_equal(region.cell_mask(field), np.logical_and.reduce(corners))
    assert region.cell_mask(field).any()
    assert nodes[region.node_window(field)].sum() == nodes.sum()


def test_mirrors_of_a_region_about_a_bend():
    path = NominalBend(BEND["radius_m"], BEND["angle_rad"], BEND["arm_m"])
    region = Region("circle", 0.01, -0.1, 0.1, path=path)
    assert not region.mirror_symmetric(0)
    assert region.mirror_symmetric(1) and region.mirror_symmetric(2)
    assert not Region("circle", 0.01, -0.1, 0.2, path=path).mirror_symmetric(2)
    with pytest.raises(ValueError, match="not about a bend"):
        region.z_mask(np.zeros(3))
    with pytest.raises(ValueError, match="not about a bend"):
        region.transverse_mask(_grid(np.linspace(-0.1, 0.1, 5)))


def test_bend_arguments_are_checked():
    field = _grid(np.linspace(-0.1, 0.1, 5))
    with pytest.raises(ValueError, match="unknown bend keys"):
        region_for(field, radius_m=0.01, bend={**BEND, "length": 1.0})
    with pytest.raises(ValueError, match="bends_toward"):
        region_for(field, radius_m=0.01, bend={**BEND, "bends_toward": "up"})
    with pytest.raises(ValueError, match="give radius_m"):
        region_for(field, bend=BEND)
