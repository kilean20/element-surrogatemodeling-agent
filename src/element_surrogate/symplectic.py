"""Transverse magnetic two-form diagnostics at fixed momentum and arrival.

From P=p+qA, dx∧dPx+dy∧dPy = dx∧dpx+dy∧dpy-qBn dx∧dy.
Normalize momenta by the particle momentum, use right-handed plane frames,
and pull back this form by the native map. No vector-potential gauge is needed
for this diagnostic. It is not a full 5D/6D canonical-form assertion.
"""

import numpy as np

from element_surrogate.readings import reconstruct
from element_surrogate.transport import verify_transport_source


def to_mechanical(states, q_scale, p_scale):
    values = np.asarray(states, float)
    slopes = values[:, [1, 3]]
    momenta = slopes / np.sqrt(1 + np.sum(slopes * slopes, axis=1))[:, None]
    return np.column_stack(
        [
            values[:, 0] / q_scale,
            momenta[:, 0] / p_scale,
            values[:, 2] / q_scale,
            momenta[:, 1] / p_scale,
        ]
    )


def from_mechanical(values, template, q_scale, p_scale):
    result = np.asarray(template, float).copy()
    p = np.asarray(values)[:, [1, 3]] * p_scale
    square = 1 - np.sum(p * p, axis=1)
    if np.any(square <= 0):
        raise ValueError("mechanical momentum perturbation has no forward direction")
    result[:, [0, 2]] = np.asarray(values)[:, [0, 2]] * q_scale
    result[:, [1, 3]] = p / np.sqrt(square)[:, None]
    return result


def form(normal_t, coupling, q_scale, p_scale):
    b = np.asarray(normal_t)
    result = np.zeros((len(b), 4, 4))
    result[:, 0, 1] = result[:, 2, 3] = 1
    result[:, 1, 0] = result[:, 3, 2] = -1
    result[:, 0, 2] = -coupling * b * q_scale / p_scale
    result[:, 2, 0] = -result[:, 0, 2]
    return result


def diagnose(
    map_function,
    boundary_normal,
    states,
    rigidity_tm,
    field_scale,
    *,
    q_scale_m,
    p_scale,
    relative_steps,
):
    states = np.asarray(states, float)
    steps = np.asarray(relative_steps, float)
    if (
        states.ndim != 2
        or states.shape[1] not in (4, 5, 6)
        or not len(states)
        or not np.isfinite(states).all()
    ):
        raise ValueError("finite native 4D/5D/6D probe states required")
    if (
        not np.isfinite([rigidity_tm, field_scale, q_scale_m, p_scale]).all()
        or rigidity_tm == 0
        or q_scale_m <= 0
        or p_scale <= 0
    ):
        raise ValueError("nonzero signed rigidity and positive declared canonical scales required")
    if (
        steps.ndim != 1
        or len(steps) < 3
        or np.any(steps <= 0)
        or not np.isfinite(steps).all()
        or np.any(np.diff(steps) >= 0)
    ):
        raise ValueError("at least three decreasing finite-difference resolutions required")
    delta = states[:, -1] if states.shape[1] > 4 else np.zeros(len(states))
    if np.any(delta <= -1):
        raise ValueError("delta_p/p0 must exceed -1")
    coupling = field_scale / (rigidity_tm * (1 + delta))
    inputs = to_mechanical(states, q_scale_m, p_scale)
    raw_outputs = np.asarray(map_function(states))
    outputs = np.asarray(raw_outputs, float)
    if outputs.shape != states.shape or not np.isfinite(outputs).all():
        raise ValueError("every diagnostic probe must have a complete finite exit")
    if states.shape[1] > 4 and not np.array_equal(
        raw_outputs[:, -1], states[:, -1].astype(raw_outputs.dtype)
    ):
        raise ValueError("static-map momentum deviation is not conserved")
    before = form(boundary_normal(states, False), coupling, q_scale_m, p_scale)
    after = form(boundary_normal(outputs, True), coupling, q_scale_m, p_scale)
    reports = []
    for step in steps:
        columns = []
        for j in range(4):
            d = np.zeros_like(inputs)
            d[:, j] = step
            plus = map_function(from_mechanical(inputs + d, states, q_scale_m, p_scale))
            minus = map_function(from_mechanical(inputs - d, states, q_scale_m, p_scale))
            if not np.isfinite(plus).all() or not np.isfinite(minus).all():
                raise ValueError("a finite-difference diagnostic probe was lost or unsupported")
            columns.append(
                (to_mechanical(plus, q_scale_m, p_scale) - to_mechanical(minus, q_scale_m, p_scale))
                / (2 * step)
            )
        jacobian = np.stack(columns, axis=-1)
        defect = np.swapaxes(jacobian, -1, -2) @ after @ jacobian - before
        residual = np.linalg.norm(defect, axis=(1, 2)) / np.linalg.norm(before, axis=(1, 2))
        reports.append(
            {
                "relative_step": float(step),
                "max_relative_two_form_defect": float(residual.max()),
                "p99_relative_two_form_defect": float(np.quantile(residual, 0.99)),
                "per_probe_relative_defect": residual.tolist(),
            }
        )
    return {
        "schema": "element_surrogate.transverse_two_form.v1",
        "probe_count": len(states),
        "q_scale_m": q_scale_m,
        "p_scale": p_scale,
        "refinements": reports,
        "boundary_normal_field_included": True,
        "scope": (
            "4D transverse derivatives at each fixed momentum/arrival; no"
            " full 5D/6D symplectic claim"
        ),
        "reading": (
            "A small stable defect supports this finite-cohort structural"
            " diagnostic. Compare derivative refinement, integration/reco"
            "nstruction changes and fitting errors separately."
        ),
    }


def for_transport(transport, states, rigidity_tm, field_scale, **options):
    verify_transport_source(transport)
    reconstructed = reconstruct(transport.grid, transport.reconstruction)

    def run(values):
        output, status, _ = transport.run(values, rigidity_tm, field_scale)
        if not status.reached_exit.all():
            raise ValueError("all symplecticity probes must reach the exit with supported stencils")
        return output

    def normal(values, at_exit):
        frame = transport.exit if at_exit else transport.entry
        positions = (
            frame.origin_m
            + values[:, 0, None] * frame.horizontal
            + values[:, 2, None] * frame.vertical
        )
        return reconstructed.sample(positions) @ frame.tangent

    result = diagnose(run, normal, states, rigidity_tm, field_scale, **options)
    verify_transport_source(transport)
    return {
        **result,
        "transport": transport.description(),
        "rigidity_tm": rigidity_tm,
        "field_scale": field_scale,
    }


def from_files(path, request_path, output_path) -> dict:
    from element_surrogate.benchmark.protocol import file_hash, read_json, write_json
    from element_surrogate.workflow import load_transport

    transport, _ = load_transport(path)
    record = for_transport(transport, **read_json(request_path))
    write_json(output_path, record, exclusive=True)
    return {
        "report_path": str(output_path),
        "sha256": file_hash(output_path),
        "probe_count": record["probe_count"],
        "scope": record["scope"],
        "refinements": [
            {k: v for k, v in r.items() if not k.startswith("per_probe")}
            for r in record["refinements"]
        ],
    }
