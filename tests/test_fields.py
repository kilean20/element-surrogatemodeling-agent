"""Tests for field maps, field files, the TRACK table reader and format detection."""

import numpy as np
import pytest

from element_surrogate.fields import (
    FieldMap,
    detect_format,
    load_field_file,
    read_track_table,
)
from element_surrogate.synthetic import multipole_sources, point_source_field
from tests.track_format import write_track_table


@pytest.fixture
def quad():
    axis = np.linspace(-0.01, 0.01, 5)
    z = np.linspace(-0.05, 0.05, 7)
    return point_source_field(multipole_sources(2, 0.03, 0.06, 1.0e-4, 5), axis, axis, z)


def test_round_trip_converts_cm_and_gauss_to_si(tmp_path, quad):
    path = write_track_table(tmp_path / "quad.table", [quad])

    table = read_track_table(path)

    slot = table.slot(None)
    np.testing.assert_allclose(slot.x_m, quad.x_m, rtol=0, atol=1e-15)
    np.testing.assert_allclose(slot.z_m, quad.z_m, rtol=0, atol=1e-15)
    np.testing.assert_allclose(slot.b_t, quad.b_t, rtol=1e-13, atol=0)
    assert table.integrity["rows_per_node"] == 1
    assert table.integrity["duplicate_terminal_row"] is False
    assert table.summary()["slots"][0]["grid"]["nodes"] == [5, 5, 7]


def test_each_slot_carries_its_source(tmp_path, quad):
    path = write_track_table(tmp_path / "quad.table", [quad])
    source = read_track_table(path).slots[0].source
    assert source.format == "track_table"
    assert source.slot == "slot0"
    assert source.source_units == {"coordinates": "cm", "field": "gauss"}
    assert source.assumed_storage["axes_slowest_to_fastest"] == ["x", "y", "z"]
    assert len(source.sha256) == 64


def test_interleaved_zero_rows_are_reported(tmp_path, quad):
    path = write_track_table(tmp_path / "quad.table", [quad], interleave_zero_rows=True)
    table = read_track_table(path)
    assert table.integrity["interleaved_zero_rows"] is True
    np.testing.assert_allclose(table.slots[0].b_t, quad.b_t, rtol=1e-13, atol=0)


def test_duplicated_final_row_is_reported_and_dropped(tmp_path, quad):
    path = write_track_table(tmp_path / "quad.table", [quad], duplicate_last_row=True)
    table = read_track_table(path)
    assert table.integrity["duplicate_terminal_row"] is True
    assert table.integrity["data_rows_found"] == table.integrity["nodes"] + 1
    np.testing.assert_allclose(table.slots[0].b_t, quad.b_t, rtol=1e-13, atol=0)


def test_several_slots_require_a_choice(tmp_path, quad):
    empty = FieldMap(quad.x_m, quad.y_m, quad.z_m, np.zeros_like(quad.b_t))
    path = write_track_table(tmp_path / "coils.table", [empty, quad, quad])
    table = read_track_table(path)
    assert table.integrity["field_slots"] == 3
    assert table.integrity["all_zero_slots"] == ["slot0"]
    with pytest.raises(ValueError, match="choose one"):
        table.slot(None)
    with pytest.raises(ValueError, match="does not exist"):
        table.slot(3)
    np.testing.assert_allclose(table.slot(2).b_t, quad.b_t, rtol=1e-13, atol=0)


def test_row_count_that_does_not_match_the_header_raises(tmp_path, quad):
    path = write_track_table(tmp_path / "quad.table", [quad])
    text = path.read_text(encoding="ascii").splitlines()
    path.write_text("\n".join(text[:-2]) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="data rows"):
        read_track_table(path)


def test_nonzero_first_rows_are_not_mistaken_for_electric_rows(tmp_path, quad):
    path = write_track_table(tmp_path / "quad.table", [quad, quad])
    lines = path.read_text(encoding="ascii").splitlines()
    doubled = lines[:5] + [row for row in lines[5:] for _ in range(2)]
    path.write_text("\n".join(doubled) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="not zero"):
        read_track_table(path)


def test_documented_header_is_interpreted_and_checked(tmp_path, quad):
    single = read_track_table(write_track_table(tmp_path / "b.table", [quad]))
    assert single.integrity["declared_field_type"] == {"code": 1, "meaning": "B"}
    assert single.integrity["declared_type_agrees_with_rows"] is True
    assert single.integrity["symmetry_flags"] == [0, 0, 0]
    assert single.integrity["symmetry_note"] is None
    both = read_track_table(
        write_track_table(tmp_path / "eb.table", [quad], interleave_zero_rows=True)
    )
    assert both.integrity["declared_field_type"]["meaning"] == "E and B"
    assert both.integrity["declared_type_agrees_with_rows"] is True


def test_header_that_disagrees_with_the_rows_is_reported(tmp_path, quad):
    path = write_track_table(tmp_path / "q.table", [quad], field_type_line="2")
    assert read_track_table(path).integrity["declared_type_agrees_with_rows"] is False


def test_electric_only_table_is_refused(tmp_path, quad):
    path = write_track_table(tmp_path / "e.table", [quad], field_type_line="0")
    with pytest.raises(ValueError, match="electric field only"):
        read_track_table(path)


def test_symmetry_flags_and_other_first_lines(tmp_path, quad):
    flagged = write_track_table(tmp_path / "s.table", [quad], symmetry_line="1 0 0")
    integrity = read_track_table(flagged).integrity
    assert integrity["symmetry_flags"] == [1, 0, 0]
    assert "does not unfold" in integrity["symmetry_note"]
    multi = write_track_table(tmp_path / "m.table", [quad, quad, quad], field_type_line="111")
    integrity = read_track_table(multi).integrity
    assert integrity["declared_field_type"] is None  # undocumented form, kept verbatim
    assert integrity["declared_type_agrees_with_rows"] is None


def test_format_is_detected_and_loaded(tmp_path, quad):
    path = write_track_table(tmp_path / "quad.table", [quad])
    assert detect_format(path) == "track_table"
    np.testing.assert_allclose(load_field_file(path).slot(None).b_t, quad.b_t, rtol=1e-13)


def test_unknown_format_asks_for_one(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not a field file\n", encoding="ascii")
    with pytest.raises(ValueError, match="give format"):
        load_field_file(path)
    with pytest.raises(ValueError, match="unknown format"):
        load_field_file(path, format="opera")
    with pytest.raises(FileNotFoundError):
        detect_format(tmp_path / "absent.table")


def test_with_values_keeps_grid_and_source(tmp_path, quad):
    field = read_track_table(write_track_table(tmp_path / "quad.table", [quad])).slots[0]
    changed = field.with_values(np.zeros_like(field.b_t))
    assert changed.source == field.source
    np.testing.assert_array_equal(changed.z_m, field.z_m)


def test_field_map_rejects_bad_shapes():
    axis = np.linspace(0.0, 1.0, 3)
    with pytest.raises(ValueError):
        FieldMap(axis, axis, axis, np.zeros((3, 3, 2, 3)))
    with pytest.raises(ValueError):
        FieldMap(axis[::-1], axis, axis, np.zeros((3, 3, 3, 3)))


def test_trilinear_sample_reproduces_nodes_and_linear_fields(quad):
    X, Y, Z = np.meshgrid(quad.x_m, quad.y_m, quad.z_m, indexing="ij")
    nodes = np.stack([X, Y, Z], axis=-1)
    np.testing.assert_allclose(quad.sample(nodes), quad.b_t, rtol=0, atol=1e-15)
    linear = quad.with_values(np.stack([X + 2 * Y, 3 * Z, X * 0 + 1], axis=-1))
    points = np.array([[0.0013, -0.004, 0.017], [0.01, 0.01, -0.05]])
    expected = np.stack([points[:, 0] + 2 * points[:, 1], 3 * points[:, 2], np.ones(2)], axis=-1)
    np.testing.assert_allclose(linear.sample(points), expected, rtol=0, atol=1e-15)
    with pytest.raises(ValueError, match="outside the grid"):
        quad.sample(np.array([0.0, 0.0, 1.0]))
