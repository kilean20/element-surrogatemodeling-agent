"""Tests for the CST text and HDF5 readers: units, placement by coordinates, the recorded
storage order, and refusals."""

import numpy as np
import pytest

from element_surrogate.fields import detect_format, load_field_file, read_cst_hdf5, read_cst_text
from element_surrogate.readiness import storage_order
from element_surrogate.synthetic import multipole_sources, point_source_field
from tests.cst_format import write_cst_hdf5, write_cst_text


@pytest.fixture(scope="module")
def quad():
    """Different node counts on the three axes, so a misplaced value cannot go unnoticed."""
    x = np.linspace(-0.012, 0.012, 7)
    y = np.linspace(-0.01, 0.01, 6)
    z = np.linspace(-0.06, 0.06, 9)
    return point_source_field(multipole_sources(2, 0.03, 0.06, 1.0e-4, 7), x, y, z)


def _same(field, expected, rtol=1e-13):
    np.testing.assert_allclose(field.x_m, expected.x_m, rtol=0, atol=1e-15)
    np.testing.assert_allclose(field.y_m, expected.y_m, rtol=0, atol=1e-15)
    np.testing.assert_allclose(field.z_m, expected.z_m, rtol=0, atol=1e-15)
    np.testing.assert_allclose(field.b_t, expected.b_t, rtol=rtol, atol=0)


@pytest.mark.parametrize("nesting", ["xyz", "zyx", "yxz"])
def test_text_rows_are_placed_by_their_coordinates(tmp_path, quad, nesting):
    path = write_cst_text(tmp_path / "b.txt", quad, nesting=nesting)
    table = read_cst_text(path)
    _same(table.slot(None), quad)
    assert table.integrity["row_nesting_slowest_to_fastest"] == list(nesting)
    assert table.slots[0].source.assumed_storage["axes_slowest_to_fastest"] == list(nesting)
    assert table.slots[0].source.source_units == {"coordinates": "mm", "field": "V.s/m^2"}


def test_shuffled_and_repeated_rows(tmp_path, quad):
    path = write_cst_text(tmp_path / "b.txt", quad, shuffle_seed=3, repeat_rows=(0, 5))
    table = read_cst_text(path)
    _same(table.slot(None), quad)
    assert table.integrity["row_nesting_slowest_to_fastest"] is None
    assert table.integrity["duplicated_rows_dropped"] == 2


def test_text_component_names_and_units(tmp_path, quad):
    path = write_cst_text(tmp_path / "b.txt", quad, field_columns=("Bx", "By", "Bz"))
    _same(read_cst_text(path).slot(None), quad)
    gauss = write_cst_text(tmp_path / "g.txt", quad.with_values(quad.b_t * 1.0e4), field_unit="G")
    _same(read_cst_text(gauss).slot(None), quad, rtol=1e-12)


def test_complex_export_needs_zero_imaginary_parts(tmp_path, quad):
    zero = write_cst_text(tmp_path / "c.txt", quad, imaginary=np.zeros(3))
    table = read_cst_text(zero)
    _same(table.slot(None), quad)
    assert table.integrity["imaginary_parts"] == "present, all zero"
    nonzero = write_cst_text(tmp_path / "d.txt", quad, imaginary=np.full(3, 1.0e-9))
    with pytest.raises(ValueError, match="imaginary"):
        read_cst_text(nonzero)


def test_text_refusals(tmp_path, quad):
    with pytest.raises(ValueError, match="H in A/m"):
        read_cst_text(write_cst_text(tmp_path / "h.txt", quad, field_unit="A/m"))
    with pytest.raises(ValueError, match="unknown magnetic-field unit"):
        read_cst_text(write_cst_text(tmp_path / "u.txt", quad, field_unit="Oe"))
    missing = write_cst_text(tmp_path / "m.txt", quad)
    lines = missing.read_text(encoding="ascii").splitlines()
    missing.write_text("\n".join(lines[:-1]) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="Cartesian product"):
        read_cst_text(missing)
    conflict = write_cst_text(tmp_path / "k.txt", quad)
    lines = conflict.read_text(encoding="ascii").splitlines()
    first = lines[2].split()
    lines.append(" ".join(first[:3] + ["1", "2", "3"]))
    conflict.write_text("\n".join(lines) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="twice with different values"):
        read_cst_text(conflict)


def test_hdf5_round_trip_and_recorded_storage(tmp_path, quad):
    table = read_cst_hdf5(write_cst_hdf5(tmp_path / "b.h5", quad))
    _same(table.slot(None), quad, rtol=1e-6)  # float32 export
    source = table.slots[0].source
    assert source.assumed_storage["axes_slowest_to_fastest"] == ["z", "y", "x"]
    assert source.format == "cst_hdf5"


def test_hdf5_complex_parts(tmp_path, quad):
    zero = read_cst_hdf5(write_cst_hdf5(tmp_path / "c.h5", quad, complex_parts=True))
    _same(zero.slot(None), quad)
    assert zero.integrity["imaginary_parts"] == "present, all zero"
    with pytest.raises(ValueError, match="imaginary"):
        read_cst_hdf5(write_cst_hdf5(tmp_path / "d.h5", quad, complex_parts=True, imaginary=1.0))


def test_hdf5_refusals(tmp_path, quad):
    with pytest.raises(ValueError, match="H-Field"):
        read_cst_hdf5(write_cst_hdf5(tmp_path / "h.h5", quad, dataset="H-Field"))
    with pytest.raises(ValueError, match="unknown magnetic-field unit"):
        read_cst_hdf5(write_cst_hdf5(tmp_path / "u.h5", quad, field_unit="Oe"))


def test_storage_test_starts_from_the_recorded_order(tmp_path, quad):
    """The labels of the storage test refer to the order the reader recorded: an HDF5 export
    read as stored ranks first under its own (z, y, x) nesting."""
    field = read_cst_hdf5(write_cst_hdf5(tmp_path / "b.h5", quad)).slot(None)
    result = storage_order(field)
    assert result["as_read_rank"] == 1
    assert result["best"]["axes_slowest_to_fastest"] == ["z", "y", "x"]
    assert result["findings"] == []


def test_formats_are_detected(tmp_path, quad):
    text = write_cst_text(tmp_path / "b.txt", quad)
    h5 = write_cst_hdf5(tmp_path / "b.h5", quad)
    assert detect_format(text) == "cst_text"
    assert detect_format(h5) == "cst_hdf5"
    _same(load_field_file(h5).slot(None), quad, rtol=1e-6)
