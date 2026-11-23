"""Training-only numerical scales and explicit slope/momentum coordinate math.

Numerical normalization never supplies a vector-potential gauge or changes
physical loss tolerances. All state values here use native metres and radians.
"""

from __future__ import annotations

import numpy as np


def rms_scales(values):
    """Stable column RMS, with zero preserved as a degenerate observation."""
    data = np.asarray(values, dtype=float)
    if data.ndim != 2 or not len(data) or not np.isfinite(data).all():
        raise ValueError("nonempty finite training matrix required")
    maximum = np.max(np.abs(data), axis=0)
    safe = np.where(maximum > 0, maximum, 1.0)
    return maximum * np.sqrt(np.mean((data / safe) ** 2, axis=0))


def output_scales(inputs, targets, dimension):
    """Common raw-target RMS BEFORE base subtraction; training rows only.

    Six-dimensional arrival uses the increment. Momentum pass-through is not
    learned. Exact zero components fall back to matching training-input RMS,
    then one native unit, with the degeneracy explicitly recorded.
    """
    x, y = np.asarray(inputs, float), np.asarray(targets, float)
    if dimension not in (4, 5, 6) or x.ndim != 2 or x.shape[1] != dimension + 2:
        raise ValueError("4D/5D/6D native inputs and controls required")
    if y.shape != (len(x), dimension) or not np.isfinite(x).all():
        raise ValueError("matching finite training inputs and targets required")
    learned = dimension if dimension == 4 else dimension - 1
    target = y[:, :learned].copy()
    if dimension == 6:
        target[:, 4] -= x[:, 4]
    scale = rms_scales(target)
    input_rms = rms_scales(x[:, :learned])
    zero = scale == 0
    fallback = zero & (input_rms > 0)
    scale[fallback] = input_rms[fallback]
    unit = scale == 0
    scale[unit] = 1.0
    return scale, {
        "method": "train_rms",
        "rows": len(x),
        "scales_native": scale.tolist(),
        "target": "raw output before any base subtraction; arrival increment for 6D",
        "training_input_fallback_components": np.flatnonzero(fallback).tolist(),
        "degenerate_native_unit_fallback_components": np.flatnonzero(unit).tolist(),
        "validation_or_test_used": False,
        "loss_targets_used": False,
        "scope": "numerical conditioning; not canonical coordinate conversion",
    }


def slopes_to_mechanical(slopes, delta=0.0, longitudinal_metric=1.0):
    """Return transverse mechanical p/p0 from dx/ds,dy/ds.

    Metric is the positive tangent coefficient in dr/ds. It is one for a
    straight local plane; a curved reference requires its declared signed
    geometry. Momentum deviation refers to actual p0, not effective rigidity.
    """
    slopes = np.asarray(slopes, float)
    if slopes.ndim < 1 or slopes.shape[-1] != 2:
        raise ValueError("last slope axis must have two transverse components")
    shape = slopes.shape[:-1]
    deviation = np.broadcast_to(np.asarray(delta, float), shape)
    metric = np.broadcast_to(np.asarray(longitudinal_metric, float), shape)
    if (
        not np.isfinite(slopes).all()
        or not np.isfinite(deviation).all()
        or not np.isfinite(metric).all()
        or np.any(deviation <= -1)
        or np.any(metric <= 0)
    ):
        raise ValueError("finite forward-chart slopes, metric>0 and delta>-1 required")
    norm = np.sqrt(metric**2 + np.sum(slopes**2, axis=-1))
    return slopes * ((1 + deviation) / norm)[..., None]


def mechanical_to_slopes(momentum, delta=0.0, longitudinal_metric=1.0):
    """Invert the forward mechanical momentum chart without clamping."""
    p = np.asarray(momentum, float)
    if p.ndim < 1 or p.shape[-1] != 2:
        raise ValueError("last momentum axis must have two components")
    deviation = np.broadcast_to(np.asarray(delta, float), p.shape[:-1])
    metric = np.broadcast_to(np.asarray(longitudinal_metric, float), p.shape[:-1])
    radicand = (1 + deviation) ** 2 - np.sum(p * p, axis=-1)
    if (
        not np.isfinite(p).all()
        or not np.isfinite(deviation).all()
        or not np.isfinite(metric).all()
        or np.any(deviation <= -1)
        or np.any(metric <= 0)
        or np.any(radicand <= 0)
    ):
        raise ValueError("real positive forward momentum chart required")
    return p * (metric / np.sqrt(radicand))[..., None]


def scaled_canonical_form(scales):
    """Pull back J for (x,Px,y,Py)=diag(scales)*z_normalized.

    Gauge/coordinate eligibility is a prerequisite supplied separately. A
    numerical scaling need not itself be a canonical transformation.
    """
    scales = np.asarray(scales, float)
    if scales.shape != (4,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("four positive canonical coordinate scales required")
    form = np.zeros((4, 4))
    form[0, 1], form[2, 3] = scales[0] * scales[1], scales[2] * scales[3]
    form[1, 0], form[3, 2] = -form[0, 1], -form[2, 3]
    return form


class _RMS:
    """Mergeable column sums scaled by their largest observed magnitude."""

    def __init__(self):
        self.maximum = None
        self.squares = None
        self.rows = 0

    def update(self, values):
        values = np.asarray(values, dtype=float)
        maximum = np.max(np.abs(values), axis=0)
        if self.maximum is None:
            self.maximum = np.zeros_like(maximum)
            self.squares = np.zeros_like(maximum)
        updated = np.maximum(self.maximum, maximum)
        safe = np.where(updated > 0, updated, 1.0)
        self.squares = self.squares * (self.maximum / safe) ** 2 + np.sum(
            (values / safe) ** 2, axis=0
        )
        self.maximum = updated
        self.rows += len(values)

    def rms(self):
        if not self.rows:
            raise ValueError("nonempty training cohort required")
        return self.maximum * np.sqrt(self.squares / self.rows)


def propose_from_cohort(
    cohort_path,
    dimension,
    output_path,
    *,
    protocol_path=None,
    batch_rows=65536,
    working_memory_bytes=268435456,
):
    """Propose train-only magnitude scales from an explicitly supplied cohort.

    Descriptors must declare train. A frozen protocol additionally binds units,
    rows and identity; a legacy NPZ without one relies on the caller's contract.
    """
    from element_surrogate.benchmark.protocol import file_hash, write_json
    from element_surrogate.cohort_io import native_batches

    if dimension not in (4, 5, 6):
        raise ValueError("native 4D/5D/6D dimension required")
    identity = file_hash(cohort_path)
    protocol_digest = file_hash(protocol_path) if protocol_path else None
    target_rms, input_rms, q_rms, p_rms = (_RMS() for _ in range(4))
    learned = dimension if dimension == 4 else dimension - 1
    for x, y in native_batches(
        cohort_path,
        dimension,
        batch_rows=batch_rows,
        working_memory_bytes=working_memory_bytes,
        protocol_path=protocol_path,
    ):
        target = y[:, :learned].astype(float, copy=True)
        if dimension == 6:
            target[:, 4] -= x[:, 4]
        if not np.isfinite(target).all():
            raise ValueError("finite arrival increment required")
        target_rms.update(target)
        input_rms.update(x[:, :learned])
        delta = 0.0 if dimension == 4 else x[:, dimension - 1]
        p_rms.update(slopes_to_mechanical(x[:, [1, 3]], delta))
        q_rms.update(x[:, [0, 2]])
    scales, inputs = target_rms.rms(), input_rms.rms()
    fallback = (scales == 0) & (inputs > 0)
    scales[fallback] = inputs[fallback]
    unit = scales == 0
    scales[unit] = 1.0
    report = {
        "method": "train_rms",
        "rows": target_rms.rows,
        "scales_native": scales.tolist(),
        "target": "raw output before any base subtraction; arrival increment for 6D",
        "training_input_fallback_components": np.flatnonzero(fallback).tolist(),
        "degenerate_native_unit_fallback_components": np.flatnonzero(unit).tolist(),
        "validation_or_test_used": False,
        "loss_targets_used": False,
        "scope": "numerical conditioning; not canonical coordinate conversion",
    }
    q = float(np.hypot(*q_rms.rms()) / np.sqrt(2))
    p = float(np.hypot(*p_rms.rms()) / np.sqrt(2))
    if file_hash(cohort_path) != identity:
        raise ValueError("cohort changed during scale calculation")
    if protocol_path and file_hash(protocol_path) != protocol_digest:
        raise ValueError("frozen protocol changed during scale calculation")
    result = {
        "schema": "element_surrogate.training_scaling_proposal.v1",
        "cohort_sha256": identity,
        "training_scope": "frozen protocol training identity"
        if protocol_path
        else "declared train descriptor or caller-designated native NPZ; not authenticated split",
        "protocol_sha256": protocol_digest,
        "execution": {
            "batch_rows": batch_rows,
            "working_memory_bytes": working_memory_bytes,
            "method": "one shard plus batched scaled-sum statistics; no implicit row cap",
            "scope": "working arrays; process RSS and runtime baseline are separate",
        },
        "dimension": dimension,
        "native_contract_required": "plane offsets in m; direction ratios in rad; delta_p/p0",
        "output_normalization": report,
        "common_canonical_magnitude_proposal": {
            "q_scale_m": q if q > 0 else None,
            "p_over_p0_scale": p if p > 0 else None,
            "canonical_eligibility_assessed": False,
            "assumption": "unit longitudinal metric at native planes; actual p0 normalization",
            "reason": "pooled train RMS gives a common scale for both transverse pairs",
            "limits": (
                "Mechanical momentum is canonical only under a separately established "
                "gauge/boundary contract; zero magnitude needs a user scale."
            ),
        },
        "decision": "proposed default; user may override with declared native units and rationale",
        "loss_scales": "separate physical error targets, not inferred here",
    }
    write_json(output_path, result, exclusive=True)
    return result
