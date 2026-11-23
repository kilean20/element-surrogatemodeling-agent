"""Byte-preserving perturbation, raw lost outputs and sentinel controls."""

import numpy as np
import pytest

from element_surrogate.region import Region
from element_surrogate.track_adapter import TrackRun, parse_loss_positions
from element_surrogate.track_domain import compare_runs, perturb_track_table


def test_loss_positions_and_malformed_rows():
    np.testing.assert_allclose(
        parse_loss_positions("header\n1 2 3 4 5 6 7 12.5\n1 2 3 4 5 6 7 2D1\n"), [0.125, 0.2]
    )
    assert parse_loss_positions("header\n").size == 0
    with pytest.raises(ValueError, match="malformed"):
        parse_loss_positions("header\n1 2\n")


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("duplicate", [False, True])
def test_unchanged_rows_are_byte_identical_and_zero_field_addition_works(
    tmp_path, interleaved, duplicate
):
    from element_surrogate.fields import read_track_table

    field = np.zeros((5, 5, 7, 3))
    source = tmp_path / "source.table"
    # Build the small documented table explicitly, preserving CRLF as a control.
    header = ["2" if interleaved else "1", "0 0 0", "-3 3 5", "-3 3 5", "-10 10 7"]
    rows = []
    for _ in range(5 * 5 * 7):
        if interleaved:
            rows.append("0.0 0.0 0.0")
        rows.append("0.00000000 0.00000000 0.00000000")
    if duplicate:
        rows.append(rows[-1])
    source.write_bytes(("\r\n".join(header + rows) + "\r\n").encode())
    output = tmp_path / "perturbed.table"
    support = Region("circle", 0.022)
    receipt = perturb_track_table(source, output, support, addition_t=[0.1, 0.2, 0.3])
    before = read_track_table(source).slot(None)
    after = read_track_table(output).slot(None)
    inside = support.node_mask(before)
    np.testing.assert_array_equal(after.b_t[inside], before.b_t[inside])
    np.testing.assert_allclose(
        after.b_t[~inside], np.broadcast_to([0.1, 0.2, 0.3], after.b_t[~inside].shape)
    )
    original_lines = source.read_bytes().splitlines(keepends=True)
    changed_lines = output.read_bytes().splitlines(keepends=True)
    assert len(original_lines) == len(changed_lines)
    np.testing.assert_array_equal(original_lines[:5], changed_lines[:5])
    for index, (a, b) in enumerate(zip(original_lines[5:], changed_lines[5:], strict=True)):
        node = min(index // (2 if interleaved else 1), 175 - 1)
        electric = interleaved and index % 2 == 0 and index < 350
        if electric or inside.reshape(-1)[node]:
            assert a == b
    assert receipt["changed_magnetic_rows"] > 0
    assert np.count_nonzero(field) == 0


def test_nonresponding_sentinel_is_not_positive_sensitivity():
    states = np.zeros((2, 4))
    a = TrackRun(states, np.array([0, 1]), None, "", states.copy())
    same = compare_runs(a, a)
    assert same["retained_probe_isolation"] and not same["perturbation_responded"]
    changed = states.copy()
    changed[0, 1] = 0.02
    b = TrackRun(changed, np.array([0, 1]), None, "", changed)
    assert not compare_runs(a, b)["retained_probe_isolation"]
    assert compare_runs(a, b)["perturbation_responded"]
