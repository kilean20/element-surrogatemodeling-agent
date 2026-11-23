"""Identities and explicit static-magnetic coordinate conventions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import cast

import numpy as np


def array_digest(*arrays) -> str:
    value = hashlib.sha256()
    for data in arrays:
        a = np.asarray(data)
        value.update(str((a.shape, a.dtype.str)).encode())
        if a.dtype.hasobject:
            raise ValueError("object arrays do not have a portable numerical identity")
        # Buffered C-order iteration bounds temporary storage even for a strided
        # map whose entire first-axis slab is much larger than the buffer.
        chunks = np.nditer(
            a,
            flags=["external_loop", "buffered", "zerosize_ok"],
            op_flags=["readonly"],
            order="C",
            buffersize=max(1, (1 << 20) // a.dtype.itemsize),
        )
        for chunk in chunks:
            value.update(cast(np.ndarray, chunk).tobytes())
    return value.hexdigest()


def field_identity(field) -> dict:
    """Identify both raw values and their reconstruction, not just a file name.

    Source hashes identify supplied file bytes; synthetic/modified maps hash values.
    A source identity is not a claim of field calibration or physical correctness.
    """
    grid = getattr(field, "field", field)
    if hasattr(grid, "b_t") and hasattr(grid, "x_m"):
        return {
            "values_sha256": array_digest(grid.b_t),
            "grid_sha256": array_digest(grid.x_m, grid.y_m, grid.z_m),
            "source": {k: v for k, v in grid.source.as_dict().items() if k != "path"}
            if grid.source
            else None,
            "reconstruction": field.describe()
            if hasattr(field, "describe")
            else {"reading": "trilinear-v1"},
            "units": {"position": "m", "field": "T"},
        }
    if hasattr(field, "b_t"):
        return {
            "uniform_b_t": list(field.b_t),
            "reconstruction": {"reading": "analytic-uniform-v1"},
        }
    raise ValueError("field needs a reproducible identity before saving a reference")


def transform_vectors(orthogonal, position, direction, magnetic_field):
    """Passive basis change: r/u polar, B axial; signed charge stays unchanged."""
    o = np.asarray(orthogonal, dtype=float)
    if (
        o.shape != (3, 3)
        or not np.isfinite(o).all()
        or not np.allclose(o @ o.T, np.eye(3), atol=1e-12, rtol=0)
    ):
        raise ValueError("orthogonal 3x3 basis change required")
    return (
        np.asarray(position) @ o.T,
        np.asarray(direction) @ o.T,
        np.linalg.det(o) * (np.asarray(magnetic_field) @ o.T),
    )


def coordinate_metadata(codec):
    """Names, conventional units and definitions of the native static map."""
    result = [
        {
            "name": "x",
            "unit": "m",
            "role": "position",
            "definition": "horizontal offset on the declared native plane",
        },
        {
            "name": "xp",
            "unit": "rad",
            "role": "slope",
            "definition": (
                "direction ratio (t dot horizontal)/(t dot plane normal); "
                "conventional rad units, not arctan"
            ),
        },
        {
            "name": "y",
            "unit": "m",
            "role": "position",
            "definition": "vertical offset on the declared native plane",
        },
        {
            "name": "yp",
            "unit": "rad",
            "role": "slope",
            "definition": (
                "direction ratio (t dot vertical)/(t dot plane normal); "
                "conventional rad units, not arctan"
            ),
        },
    ]
    if codec.dimension == 6:
        result.append(
            {
                "name": codec.longitudinal,
                "unit": "s" if codec.longitudinal == "time_s" else "m",
                "role": "arrival",
                "definition": (
                    "relative arrival coordinate advanced against the fixed traced reference"
                ),
            }
        )
    if codec.dimension > 4:
        result.append(
            {
                "name": "delta_p",
                "unit": "1",
                "role": "energy",
                "definition": (
                    "actual momentum deviation p/p0 - 1; conserved in a static magnetic field"
                ),
            }
        )
    return result


@dataclass(frozen=True)
class StaticMagneticCodec:
    """4D slopes, 5D slopes+delta_p/p0, or 6D slopes+time/ct/path+delta_p/p0.

    Positions are metres, direction ratios dimensionless. Rest mass and actual
    reference momentum are required for time; an effective rigidity encoding a
    field-amplitude error must never substitute for the particle's actual speed.
    This codec does not assert canonical symplecticity for slope/time coordinates.
    """

    dimension: int
    longitudinal: str = "time_s"

    def __post_init__(self):
        if self.dimension not in (4, 5, 6) or self.longitudinal not in ("time_s", "ct_m", "path_m"):
            raise ValueError("dimension 4/5/6 and named longitudinal coordinate required")

    def decode(self, states):
        a = np.asarray(states, dtype=float)
        if a.ndim != 2 or a.shape[1] != self.dimension or not np.isfinite(a).all():
            raise ValueError("finite states with codec dimension required")
        delta = a[:, -1] if self.dimension > 4 else np.zeros(len(a))
        if np.any(delta <= -1):
            raise ValueError("delta_p/p0 must exceed -1")
        return a[:, :4], delta

    def encode_exit(
        self,
        inputs,
        transverse,
        path_length_m,
        reference_path_m,
        *,
        actual_p0_kg_m_s=None,
        mass_kg=None,
    ):
        """Preserve momentum deviation exactly and advance a relative arrival coordinate."""
        a = np.asarray(inputs, dtype=float)
        _, delta = self.decode(a)
        result = a.copy()
        transverse = np.asarray(transverse, dtype=float)
        if transverse.shape != (len(a), 4):
            raise ValueError("matching transverse exits required")
        result[:, :4] = transverse
        if self.dimension == 6:
            length = np.broadcast_to(np.asarray(path_length_m, dtype=float), (len(a),))
            if (
                not np.isfinite(length).all()
                or np.any(length < 0)
                or not np.isfinite(reference_path_m)
                or reference_path_m < 0
            ):
                raise ValueError("nonnegative finite path lengths required")
            if self.longitudinal == "path_m":
                result[:, 4] += length - reference_path_m
            else:
                if (
                    actual_p0_kg_m_s is None
                    or mass_kg is None
                    or not np.isfinite([actual_p0_kg_m_s, mass_kg]).all()
                    or min(actual_p0_kg_m_s, mass_kg) <= 0
                ):
                    raise ValueError("time requires actual positive momentum and rest mass")
                c = 299792458.0
                p = actual_p0_kg_m_s * (1 + delta)
                inv_v = np.hypot(p, mass_kg * c) / (p * c)
                inv_v0 = np.hypot(actual_p0_kg_m_s, mass_kg * c) / (actual_p0_kg_m_s * c)
                dt = length * inv_v - reference_path_m * inv_v0
                result[:, 4] += dt * (c if self.longitudinal == "ct_m" else 1.0)
        return result
