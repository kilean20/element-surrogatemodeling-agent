"""Scientific equivalence, tile boundaries and scratch/precision controls."""

import numpy as np
import pytest

from element_surrogate import readiness as r
from element_surrogate.bend import NominalBend
from element_surrogate.diagnostic_blocks import Workspace
from element_surrogate.fields import FieldMap
from element_surrogate.region import Region
from element_surrogate.synthetic import multipole_sources, point_source_field


def same(actual, expected, path=""):
    if isinstance(expected, dict):
        assert set(actual) - {"execution"} == set(expected) - {"execution"}, path
        for key in expected:
            if key != "execution":
                same(actual[key], expected[key], path + "." + key)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            same(a, b, path + f"[{i}]")
    elif isinstance(expected, (float, np.floating)):
        np.testing.assert_allclose(
            actual, expected, rtol=2e-11, atol=1e-16, equal_nan=True, err_msg=path
        )
    else:
        assert actual == expected, (path, actual, expected)


def grid(nonuniform=False, damaged=False):
    x = np.linspace(-0.02, 0.02, 9)
    y = np.linspace(-0.018, 0.018, 11)
    z = np.linspace(-0.12, 0.12, 17)
    if nonuniform:
        x = x + 0.03 * x * x
        y = y + 0.04 * y * y
        z = z + 0.02 * z * z
    field = point_source_field(multipole_sources(2, 0.05, 0.2, 1e-3, 21), x, y, z)
    if damaged:
        b = field.b_t.copy()
        b[4, 5, 8, 0] += 0.02
        b[4, 5, 9, 1] -= 0.01
        field = field.with_values(b)
    return field


@pytest.mark.parametrize("nonuniform", [False, True])
@pytest.mark.parametrize("damaged", [False, True])
@pytest.mark.parametrize("budget", [256 * 1024, 2 * 1024**2])
def test_all_diagnostics_match_eager_across_different_tiles(tmp_path, nonuniform, damaged, budget):
    field = grid(nonuniform, damaged)
    region = Region("rectangle", 0.016, half_height_m=0.015, z_min_m=-0.11, z_max_m=0.11)
    options = {"working_memory_bytes": budget, "scratch_directory": tmp_path, "backend": "bounded"}
    same(r.maxwell_report(field, region, 3, **options), r._eager_maxwell_report(field, region, 3))
    same(
        r.harmonic_nodes(field, region, 40, 8, **options),
        r._eager_harmonic_nodes(field, region, 40, 8),
    )
    same(r.field_ends(field, region, 1e-3, **options), r._eager_field_ends(field, region, 1e-3))
    same(r.node_defects(field, **options), r._eager_node_defects(field))
    assert not list(tmp_path.iterdir())


def test_global_quantiles_are_exact_not_averaged_batch_percentiles(tmp_path):
    batches = [np.arange(103, dtype=float), np.array([-1e10, 2e10]), np.full(217, 42.0)]
    original = [b.copy() for b in batches]
    with Workspace(256 * 1024, tmp_path) as work:
        values = work.scalar_series()
        for b in batches:
            values.append(b)
        q = [0, 0.25, 0.5, 0.75, 0.99, 0.999, 1]
        np.testing.assert_array_equal(values.quantiles(q), np.quantile(np.concatenate(batches), q))
    for a, b in zip(batches, original, strict=True):
        np.testing.assert_array_equal(a, b)
    assert not list(tmp_path.iterdir())


def test_integrity_edges_zero_duplicates_and_nonfinite_match(tmp_path):
    field = grid()
    b = field.b_t.copy()
    b[0, 0, 0] = 0
    b[4, 5, 7] = np.nan
    b[-1] = b[-2]
    field = field.with_values(b)
    same(
        r.node_defects(
            field, backend="bounded", working_memory_bytes=256 * 1024, scratch_directory=tmp_path
        ),
        r._eager_node_defects(field),
    )
    assert not list(tmp_path.iterdir())


def test_global_boundaries_and_single_evaluated_node_are_not_tile_edges(tmp_path):
    a = np.linspace(-1, 1, 5)
    b = np.ones((5, 5, 5, 3))
    field = FieldMap(a, a, a, b)
    result = r.harmonic_nodes(
        field, backend="bounded", working_memory_bytes=256 * 1024, scratch_directory=tmp_path
    )
    assert result["evaluated_nodes"] == 1
    assert len(result["worst_nodes"]) == 1
    assert result["worst_nodes"][0]["node"] == [2, 2, 2]
    assert result["flagged_nodes"] == 0
    assert not list(tmp_path.iterdir())


def test_scratch_cleanup_on_failure(tmp_path):
    with pytest.raises(RuntimeError, match="deliberate"):
        with Workspace(256 * 1024, tmp_path) as work:
            values = work.mapped((16, 16, 16))
            values.fill(1.0)
            work.scalar_series().append(np.arange(100.0))
            raise RuntimeError("deliberate")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("direction", [True, False])
def test_curved_region_node_windows_and_diagnostics(tmp_path, direction):
    field = grid(damaged=True)
    path = NominalBend(0.5, 0.3, 0.03, direction)
    region = Region("rectangle", 0.014, -0.08, 0.08, 0.013, path)
    mask = region.node_mask(field)
    expected = []
    for axis in range(3):
        hit = np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))
        expected.append(slice(int(hit[0]), int(hit[-1]) + 1))
    assert region.node_window(field) == tuple(expected)
    options = {
        "backend": "bounded",
        "working_memory_bytes": 256 * 1024,
        "scratch_directory": tmp_path,
    }
    same(r.maxwell_report(field, region, 3, **options), r._eager_maxwell_report(field, region, 3))
    same(
        r.harmonic_nodes(field, region, 40, 8, **options),
        r._eager_harmonic_nodes(field, region, 40, 8),
    )
    same(r.field_ends(field, region, 1e-3, **options), r._eager_field_ends(field, region, 1e-3))


def test_storage_order_exact_all_nestings_without_whole_renest(tmp_path, monkeypatch):
    field = grid()
    region = Region("rectangle", 0.011, -0.06, 0.06, 0.01)
    expected = r._eager_storage_order(field, region)

    def forbidden(*args):
        raise AssertionError("whole-source renest used")

    monkeypatch.setattr(r, "_renest", forbidden)
    same(
        r.storage_order(
            field,
            region,
            backend="bounded",
            working_memory_bytes=256 * 1024,
            scratch_directory=tmp_path,
        ),
        expected,
    )
    assert not list(tmp_path.iterdir())


def test_disk_exhaustion_is_explicit_and_cleans_scratch(tmp_path, monkeypatch):
    import shutil

    original = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: original._replace(free=1024))
    with pytest.raises(OSError, match="insufficient disk scratch"):
        r.harmonic_nodes(
            grid(), backend="bounded", working_memory_bytes=256 * 1024, scratch_directory=tmp_path
        )
    assert not list(tmp_path.iterdir())


def test_field_summary_ignores_nonfinite_components_without_full_grid_copy():
    field = grid()
    b = field.b_t.copy()
    b[0, 0, 0] = [np.nan, np.inf, -np.inf]
    field = field.with_values(b)
    finite = np.isfinite(b)
    expected = {
        "all_finite": bool(finite.all()),
        "max_abs_b_t": float(np.linalg.norm(np.where(finite, b, 0.0), axis=-1).max()),
        "max_abs_component_t": [
            float(np.abs(b[..., c][finite[..., c]]).max(initial=0.0)) for c in range(3)
        ],
        "all_zero": bool(not np.any(b[finite])),
    }
    same(field.field_summary(), expected)


@pytest.mark.parametrize("curved", [False, True])
def test_blocked_mirror_matches_all_node_eager_reduction(curved):
    from element_surrogate.analysis import _eager_mirror_symmetry, mirror_symmetry

    field = grid(damaged=True)
    region = Region("circle", 0.014, path=NominalBend(0.5, 0.3, 0.03) if curved else None)
    same(mirror_symmetry(field, region), _eager_mirror_symmetry(field, region))


def test_components_merge_across_batches_and_keep_global_extents(tmp_path):
    from element_surrogate.bounded_readiness import _DiskComponents

    positions = np.array([[2, 2, 2], [2, 2, 3], [3, 3, 4], [6, 6, 6], [6, 6, 7]])
    with Workspace(256 * 1024, tmp_path) as work:
        components = _DiskComponents(work, (10, 10, 10))
        try:
            for position in positions[::-1]:
                components.append([position], [2.0], [0.1])
            count, groups = components.summarize(10)
            assert count == 2
            assert sorted(g[1] for g in groups) == [2, 3]
            three = next(g for g in groups if g[1] == 3)
            assert three[2:8] == (2, 2, 2, 3, 3, 4)
        finally:
            components.close()


def test_zero_field_is_computed_observation_with_undefined_relative_tail(tmp_path):
    a = np.linspace(-0.1, 0.1, 7)
    field = FieldMap(a, a, a, np.zeros((7, 7, 7, 3)))
    result = r.field_ends(field, backend="bounded", scratch_directory=tmp_path)
    same(result, r._eager_field_ends(field))
    assert result["peak_t"] == 0.0
    assert result["centroid_z_m"] is None
    assert result["ends"]["low_z"]["missing_integral_fraction"] is None
    assert result["observations"][0]["kind"] == "zero_profile"
    assert "diagnostic_error" not in result
