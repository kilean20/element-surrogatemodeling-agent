"""Write synthetic fields in the TRACK Cartesian table layout, for reader tests."""

from pathlib import Path

import numpy as np

from element_surrogate.fields import CM_TO_M, GAUSS_TO_T, FieldMap


def write_track_table(
    path: Path,
    slots: list[FieldMap],
    interleave_zero_rows: bool = False,
    duplicate_last_row: bool = False,
    field_type_line: str | None = None,
    symmetry_line: str = "0 0 0",
) -> Path:
    """Write ``slots`` (sharing one grid) as a TRACK table. The field-type line defaults to
    the documented value for the rows written: 2 (E and B) when zero electric rows are
    interleaved, else 1 (B)."""
    grid = slots[0]
    default_type = "2" if interleave_zero_rows else "1"
    lines = [field_type_line if field_type_line is not None else default_type, symmetry_line]
    for axis in (grid.x_m, grid.y_m, grid.z_m):
        lines.append(f"{axis[0] / CM_TO_M:.10g} {axis[-1] / CM_TO_M:.10g} {axis.size}")
    values = np.concatenate([s.b_t / GAUSS_TO_T for s in slots], axis=-1).reshape(
        -1, 3 * len(slots)
    )
    zeros = " ".join(["0"] * values.shape[1])
    rows = []
    for row in values:
        if interleave_zero_rows:
            rows.append(zeros)
        rows.append(" ".join(f"{v:.15e}" for v in row))
    if duplicate_last_row:
        rows.append(rows[-1])
    path.write_text("\n".join(lines + rows) + "\n", encoding="ascii")
    return path
