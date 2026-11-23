"""Complete native-resolution CST conversion; no eager reader or source mutation."""

import numpy as np
import pytest

from element_surrogate.benchmark.protocol import file_hash
from element_surrogate.field_store import convert_cst_hdf5, read_mapped_field
from element_surrogate.fields import FieldMap, read_cst_hdf5
from tests.cst_format import write_cst_hdf5


@pytest.mark.parametrize("complex_parts", [False, True])
@pytest.mark.parametrize("block_nodes", [1, 17, 256])
def test_streamed_conversion_preserves_all_values_and_axes(tmp_path, complex_parts, block_nodes):
    a = np.linspace(-0.01, 0.01, 7)
    b = np.linspace(-0.02, 0.02, 9)
    c = np.linspace(-0.03, 0.03, 11)
    field = FieldMap(a, b, c, np.random.default_rng(11).normal(size=(7, 9, 11, 3)))
    source = write_cst_hdf5(
        tmp_path / "source.h5", field, field_unit="G", complex_parts=complex_parts
    )
    before = file_hash(source)
    expected = read_cst_hdf5(source).slot(None)
    result = convert_cst_hdf5(source, tmp_path / "mapped", block_nodes=block_nodes)
    actual = read_mapped_field(result["path"]).slot(None)
    for key in ("x_m", "y_m", "z_m", "b_t"):
        np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))
    assert file_hash(source) == before
    assert not list(tmp_path.glob(".element-conversion-*"))


def test_bad_complex_field_has_no_completed_store_and_no_scratch_leak(tmp_path):
    a = np.linspace(-1, 1, 5)
    field = FieldMap(a, a, a, np.ones((5, 5, 5, 3)))
    source = write_cst_hdf5(tmp_path / "source.h5", field, complex_parts=True, imaginary=1e-9)
    with pytest.raises(ValueError, match="imaginary"):
        convert_cst_hdf5(source, tmp_path / "mapped", block_nodes=11)
    assert not (tmp_path / "mapped").exists()
    assert not list(tmp_path.glob(".element-conversion-*"))
