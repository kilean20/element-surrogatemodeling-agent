"""Tests for the readiness checks, on synthetic fields whose answers are known exactly."""

import numpy as np
import pytest

from element_surrogate.bend import NominalBend
from element_surrogate.fields import FieldMap
from element_surrogate.readiness import (
    cell_residuals,
    coarsen,
    compare_exports,
    field_ends,
    harmonic_nodes,
    maxwell_report,
    node_defects,
    storage_order,
)
from element_surrogate.region import Region
from element_surrogate.synthetic import (
    bend_sources,
    multipole_sources,
    plant_duplicated_plane,
    plant_noisy_shell,
    plant_storage_mismatch,
    plant_truncation,
    plant_zeroed_nodes,
    point_source_field,
)


def _quad(nodes_xy=21, nodes_z=81):
    axis = np.linspace(-0.02, 0.02, nodes_xy)
    z = np.linspace(-0.2, 0.2, nodes_z)
    return point_source_field(multipole_sources(2, 0.05, 0.2, 1.0e-3, 41), axis, axis, z)


@pytest.fixture(scope="module")
def quad():
    return _quad()


def _orders(report, statistic):
    rows = [s["coarsened_by_2"] for s in report["shells"] if s["coarsened_by_2"]["coarse_cells"]]
    return [(r["divergence"][statistic], r["curl"][statistic]) for r in rows]


def test_clean_field_converges_at_second_order_everywhere(quad):
    """On an exact vacuum field the residual is truncation error: pointwise order 2."""
    orders = _orders(maxwell_report(quad, shells=4), "pointwise_order")
    assert len(orders) >= 3
    for div, curl in orders:
        assert 1.9 < div < 2.1
        assert 1.9 < curl < 2.1


def test_same_volume_order_is_also_reported_and_reads_lower_near_the_axis(quad):
    orders = _orders(maxwell_report(quad, shells=4), "same_volume_order")
    assert all(0.8 < div <= 2.1 and 0.8 < curl <= 2.1 for div, curl in orders)


def test_clean_residual_is_small_against_the_derivatives(quad):
    report = maxwell_report(quad, shells=1)
    assert report["whole_region"]["curl_t_per_m"]["p99_relative"] < 1e-2


def test_refined_field_has_four_times_smaller_residual():
    fine, coarse = _quad(41, 161), _quad(21, 81)
    p99 = [
        maxwell_report(f, shells=1)["whole_region"]["curl_t_per_m"]["p99"] for f in (fine, coarse)
    ]
    assert 3.0 < p99[1] / p99[0] < 5.0


def test_noise_in_the_data_does_not_converge(quad):
    planted = plant_noisy_shell(quad, inner_radius_m=0.012, amplitude_t=5e-4, seed=7)
    clean = maxwell_report(quad, shells=4)["shells"]
    noisy = maxwell_report(planted.field, shells=4)["shells"]
    # Untouched inside: the same absolute residual (relative values share one region-wide scale).
    assert noisy[1]["curl_t_per_m"]["p99"] == clean[1]["curl_t_per_m"]["p99"]
    assert noisy[-1]["curl_t_per_m"]["p99"] > 5 * clean[-1]["curl_t_per_m"]["p99"]
    outer = noisy[-1]["coarsened_by_2"]["divergence"]
    assert outer["pointwise_order"] < 1.0  # theory for independent node errors: about 0.5
    assert outer["same_volume_order"] < 0.0  # theory: about -1


def test_jump_between_neighbouring_nodes_has_its_own_signature(quad):
    """A step in Bz across a plane is a curl error that neither converges nor behaves like noise."""
    b = quad.b_t.copy()
    b[quad.x_m > 0.015, :, :, 2] += 0.05 * np.abs(quad.b_t).max()
    outer = maxwell_report(quad.with_values(b), shells=4)["shells"][-1]["coarsened_by_2"]
    assert abs(outer["curl"]["pointwise_order"]) < 0.3
    assert outer["curl"]["same_volume_order"] < -0.5
    assert outer["divergence"]["pointwise_order"] > 1.9  # a Bz step does not touch the divergence


def test_storage_order_is_not_misled_by_a_local_defect(small_asymmetric):
    """A non-physical outer shell must not make a wrong reading look better."""
    noisy = plant_noisy_shell(small_asymmetric, inner_radius_m=0.015, amplitude_t=1e-3, seed=3)
    assert storage_order(noisy.field)["as_read_rank"] == 1


def test_region_limits_the_report(quad):
    region = Region("square", 0.01, z_min_m=-0.1, z_max_m=0.1)
    report = maxwell_report(quad, region, shells=2)
    assert report["region"]["shape"] == "square"
    whole = maxwell_report(quad, shells=1)["whole_region"]["curl_t_per_m"]["cells"]
    assert report["whole_region"]["curl_t_per_m"]["cells"] < whole


def test_grid_off_the_axis_needs_an_explicit_region():
    axis = np.linspace(0.001, 0.02, 9)
    field = point_source_field(multipole_sources(2, 0.05, 0.1, 1e-3, 5), axis, axis, axis)
    with pytest.raises(ValueError, match="both sides of the z axis"):
        maxwell_report(field)


def test_non_uniform_grid_uses_actual_cell_widths():
    """A linear field has exact finite-volume derivatives on any grid."""
    x = np.array([-0.02, -0.011, -0.004, 0.0, 0.007, 0.02])
    y = np.array([-0.02, -0.003, 0.0, 0.009, 0.02])
    z = np.array([-0.1, -0.02, 0.05, 0.1])
    X, Y, _ = np.meshgrid(x, y, z, indexing="ij")
    g = 2.5  # T/m: an ideal quadrupole field, curl- and divergence-free
    b = np.stack((g * Y, g * X, np.zeros_like(X)), axis=-1)
    res = cell_residuals(FieldMap(x, y, z, b))
    assert np.max(np.abs(res.divergence_t_per_m)) < 1e-12
    assert np.max(res.curl_t_per_m) < 1e-12
    np.testing.assert_allclose(res.derivative_norm_t_per_m, np.sqrt(2) * g)


def test_clean_field_floor_is_truncation(quad):
    result = harmonic_nodes(quad)
    assert result["floor"]["regime"] == "truncation of a smooth field"
    assert result["floor"]["median_order"] > 1.5


def test_noisy_field_floor_is_node_level_error(quad):
    clean_floor = harmonic_nodes(quad)["floor"]["median_implied_error_t"]
    sigma = 20 * clean_floor  # node noise well above this field's truncation floor
    noisy = quad.with_values(quad.b_t + np.random.default_rng(1).normal(0, sigma, quad.b_t.shape))
    result = harmonic_nodes(noisy)
    assert result["floor"]["regime"] == "node-level error in the data"
    assert result["floor"]["median_order"] < -1.5
    # For independent errors of s per component the median implied error is about 1.66 s.
    assert result["floor"]["median_implied_error_t"] == pytest.approx(1.66 * sigma, rel=0.1)


def test_wrong_node_stands_far_above_the_floor(quad):
    node = (10, 7, 40)
    planted = plant_zeroed_nodes(quad, [node])
    result = harmonic_nodes(planted.field)
    worst = result["worst_nodes"][0]
    assert tuple(worst["node"]) == node
    assert worst["order"] < -1.5
    assert worst["implied_error_t"] == pytest.approx(np.linalg.norm(quad.b_t[node]), rel=0.05)
    assert worst["ratio_to_floor"] > 1e3
    first = result["clusters_of_top_nodes"]["clusters"][0]  # clusters come worst first
    low, high = first["node_range"]
    assert all(lo <= n <= hi for lo, n, hi in zip(low, node, high, strict=True))


def test_masked_corners_form_clusters_at_their_edge(quad):
    """Corners outside a circle replaced by a near-zero value, as a masked export does."""
    r = np.hypot(quad.x_m[:, None], quad.y_m[None, :])
    masked = r > 0.022
    b = quad.b_t.copy()
    b[masked] = 5e-5  # tiny, not zero, so the zero-node scan cannot see it
    field = quad.with_values(b)
    assert node_defects(field)["zero_nodes_next_to_nonzero"]["count"] == 0
    result = harmonic_nodes(field, top=200, examples=200)
    for entry in result["worst_nodes"]:
        i, j, _ = entry["node"]
        assert masked[max(i - 1, 0) : i + 2, max(j - 1, 0) : j + 2].any(), entry
        assert entry["ratio_to_floor"] > 100
    assert 1 <= len(result["clusters_of_top_nodes"]["clusters"]) <= 8  # the four corners


def test_harmonic_nodes_respects_the_region(quad):
    inner = harmonic_nodes(quad, Region("circle", 0.01))
    assert inner["evaluated_nodes"] < harmonic_nodes(quad)["evaluated_nodes"]
    for entry in inner["worst_nodes"]:
        assert np.hypot(*entry["position_m"][:2]) <= 0.01 + 1e-12


def test_zero_nodes_and_duplicated_planes_are_found(quad):
    nodes = [(3, 4, 10), (17, 17, 60)]
    defects = node_defects(plant_zeroed_nodes(quad, nodes).field)
    assert defects["zero_nodes_next_to_nonzero"]["count"] == 2
    assert sorted(map(tuple, defects["zero_nodes_next_to_nonzero"]["examples"])) == sorted(nodes)
    duplicated = node_defects(plant_duplicated_plane(quad, axis=2, index=30).field)
    assert duplicated["duplicated_planes"] == [{"axis": "z", "planes": [30, 31]}]


def test_clean_field_has_no_node_defects_and_zero_planes_are_not_duplicates(quad):
    defects = node_defects(quad)
    assert defects["zero_nodes_next_to_nonzero"]["count"] == 0
    assert defects["non_finite_nodes"]["count"] == 0
    assert defects["duplicated_planes"] == []
    b = quad.b_t.copy()
    b[:, :, :2] = 0.0
    assert node_defects(quad.with_values(b))["duplicated_planes"] == []


@pytest.fixture(scope="module")
def small_asymmetric():
    """A small grid with unequal axis lengths and a skew dipole added, so no symmetry makes
    two storage interpretations equivalent."""
    x = np.linspace(-0.02, 0.02, 11)
    y = np.linspace(-0.02, 0.02, 9)
    z = np.linspace(-0.15, 0.15, 31)
    sources = multipole_sources(2, 0.05, 0.2, 1e-3, 21) + [((0.06, 0.0, 0.01), 2e-4)]
    return point_source_field(sources, x, y, z)


def test_storage_order_confirms_a_correct_reading(small_asymmetric):
    result = storage_order(small_asymmetric)
    assert result["as_read_rank"] == 1
    assert result["candidates_tested"] == 144


@pytest.mark.parametrize(
    "nesting, columns, signs",
    [
        ((2, 1, 0), (0, 1, 2), (1, 1, 1)),
        ((0, 1, 2), (1, 0, 2), (1, 1, 1)),
        ((0, 1, 2), (0, 1, 2), (1, -1, 1)),
    ],
)
def test_storage_order_finds_a_misread_file(small_asymmetric, nesting, columns, signs):
    planted = plant_storage_mismatch(small_asymmetric, nesting, columns, signs)
    result = storage_order(planted.field)
    assert result["best"]["axes_slowest_to_fastest"] == planted.where["axes_slowest_to_fastest"]
    assert result["best"]["columns_hold"] == planted.where["columns_hold"]
    assert result["as_read_over_best"] > 100


def test_coarsen_keeps_every_other_node_and_the_source(quad):
    coarse = coarsen(quad)
    assert coarse.shape == (11, 11, 41)
    np.testing.assert_array_equal(coarse.b_t, quad.b_t[::2, ::2, ::2])


def test_only_a_wrong_node_is_a_finding_where_noise_varies_across_the_map(quad):
    """Node errors ten times larger on one side of the map than on the other, as in real
    exports: a single global floor would flag the noisy side, the local fence does not."""
    rng = np.random.default_rng(3)
    sigma = 1.0e-5 * float(np.abs(quad.b_t).max())
    scale = np.linspace(1.0, 10.0, quad.shape[0])[:, None, None, None]
    noisy = quad.with_values(quad.b_t + scale * rng.normal(0.0, sigma, quad.b_t.shape))
    clean = harmonic_nodes(noisy)
    assert clean["floor"]["median_order"] < 0  # the floor is node-level noise
    assert clean["findings"] == []
    b = noisy.b_t.copy()
    b[10, 12, 40] += 60.0 * scale[10, 0, 0, 0] * sigma
    report = harmonic_nodes(noisy.with_values(b))
    # one finding: the wrong node, with the neighbours its error spills into
    [only] = report["findings"]
    low, high = only["where"]["node_range"]
    assert all(lo <= n <= hi for lo, n, hi in zip(low, (10, 12, 40), high, strict=True))
    assert max(h - lo for lo, h in zip(low, high, strict=True)) <= 2


def test_node_defects_and_storage_order_report_findings(quad):
    defects = node_defects(plant_duplicated_plane(quad, 2, 30).field)["observations"]
    assert [f["kind"] for f in defects] == ["duplicated_plane"]
    assert node_defects(quad)["findings"] == []
    swapped = storage_order(plant_storage_mismatch(quad, columns=(1, 0, 2)).field)
    assert [f["kind"] for f in swapped["findings"]] == ["storage_order"]
    assert storage_order(quad)["findings"] == []


def test_field_ends_flag_a_map_cut_inside_its_fringe(quad):
    long = point_source_field(
        multipole_sources(2, 0.05, 0.2, 1.0e-3, 41),
        np.linspace(-0.02, 0.02, 9),
        np.linspace(-0.02, 0.02, 9),
        np.linspace(-0.8, 0.8, 161),
    )
    whole = field_ends(long, tolerance=1e-3)
    assert whole["findings"] == []
    assert whole["ends"]["high_z"]["decaying"]
    assert whole["ends"]["high_z"]["missing_integral_fraction"] < 1e-3
    cut = plant_truncation(long, (-0.12, 0.12)).field
    assert [f["where"]["end"] for f in field_ends(cut, tolerance=1e-3)["findings"]] == [
        "low_z",
        "high_z",
    ]
    assert field_ends(cut)["findings"] == []  # no tolerance, nothing flagged


def test_checks_run_in_a_rectangle_about_a_bend():
    bend = NominalBend(1.0, 0.4, 0.0)
    sources = bend_sources(bend, 0.02, 0.03, 1.0e-3, 60, 7)
    field = point_source_field(
        sources,
        np.linspace(-0.1, 0.06, 41),
        np.linspace(-0.012, 0.012, 7),
        np.linspace(-0.4, 0.4, 81),
    )
    region = Region("rectangle", 0.02, half_height_m=0.008, path=bend)
    report = maxwell_report(field, region, shells=4)
    shells = sum(row["curl_t_per_m"]["cells"] for row in report["shells"])
    assert shells == report["whole_region"]["curl_t_per_m"]["cells"] > 0  # shells partition it
    assert report["region"]["axis"] == "a bend's nominal path"
    assert storage_order(field, region)["findings"] == []


def test_stencil_comparison_separates_truncation_from_node_noise(quad):
    smooth = maxwell_report(quad, shells=3)
    rng = np.random.default_rng(5)
    sigma = 1.0e-3 * float(np.abs(quad.b_t).max())
    noisy = maxwell_report(
        quad.with_values(quad.b_t + rng.normal(0, sigma, quad.b_t.shape)), shells=3
    )
    for row in smooth["shells"]:
        assert row["higher_order_stencil"]["curl_p99_fourth_over_second"] < 0.2
    for row in noisy["shells"]:
        assert row["higher_order_stencil"]["curl_p99_fourth_over_second"] == pytest.approx(
            np.sqrt(130) / 12 / (np.sqrt(2) / 2), rel=0.05
        )


def test_exports_on_the_same_grid_compare_node_by_node(quad):
    same = compare_exports(quad, quad)
    assert same["same_grid"] and same["interpolation_error_rms_t"] is None
    assert same["difference_with_polarity_matched"]["vector_rms_t"] == 0.0
    assert same["scale"] == pytest.approx(1.0) and same["findings"] == []
    flipped = compare_exports(quad, quad.with_values(-0.9 * quad.b_t), tolerance=1e-3)
    assert flipped["polarity"] == -1.0
    assert [f["kind"] for f in flipped["findings"]] == ["opposite_polarity", "scale_differs"]


def test_exports_on_other_grids_are_read_by_interpolation(quad):
    coarse = point_source_field(
        multipole_sources(2, 0.05, 0.2, 1.0e-3, 41),
        np.linspace(-0.02, 0.02, 11),
        np.linspace(-0.02, 0.02, 11),
        np.linspace(-0.2, 0.2, 41),
    )
    result = compare_exports(quad, coarse, tolerance=1e-3)
    difference = result["difference_with_polarity_matched"]["vector_rms_t"]
    estimate = result["interpolation_error_rms_t"]
    assert not result["same_grid"]
    assert difference == pytest.approx(estimate, rel=0.5)  # the difference is the reading's error
    assert result["findings"] == []  # within the tolerance plus the interpolation error
    shells = result["maxwell_shells"]
    assert len(shells["field"]) == len(shells["other"]) == 4
