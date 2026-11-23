"""Independent manufactured-field, frame, persistence and timing controls."""

import json
import zipfile

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from element_surrogate.adaptive import adaptive_convergence, track_adaptive
from element_surrogate.bend import PathFrame
from element_surrogate.contracts import StaticMagneticCodec, array_digest, transform_vectors
from element_surrogate.distribution import audit_archive
from element_surrogate.fields import FieldMap
from element_surrogate.lorentz import UniformField, step_convergence, straight_frame
from element_surrogate.readings import TotalDegreeTwo, reconstruct
from element_surrogate.reference import ReferenceOrbit, trace_reference
from element_surrogate.regression import RegressionMetrics
from element_surrogate.runs import RunStore, Tool, record_call, replay


def grid(function):
    axes = [np.linspace(-1, 1, n) for n in (7, 9, 11)]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    return FieldMap(*axes, function(x, y, z))


def test_total_degree_two_reproduces_manufactured_polynomial_and_edges():
    def f(x, y, z):
        return np.stack(
            [1 + 2 * x + 3 * y * z + x * x - y * y, z * z - 2 * x * y, y + 4 * x * z], axis=-1
        )

    field = grid(f)
    p = np.r_[
        np.random.default_rng(8).uniform(-1, 1, (300, 3)), [[-1, -1, -1], [1, 1, 1], [0, 0, 0]]
    ]
    np.testing.assert_allclose(reconstruct(field).sample(p), f(*p.T), rtol=0, atol=8e-15)
    nonuniform = FieldMap(np.array([-1.0, -0.5, 1.0]), field.y_m, field.z_m, field.b_t[:3])
    with pytest.raises(ValueError, match="uniform"):
        reconstruct(nonuniform)
    with pytest.raises(ValueError):
        reconstruct(field).sample([[1.01, 0, 0]])


def test_degree_two_is_distinct_from_tensor_product():
    field = grid(lambda x, y, z: np.stack([x * x * y * y, 0 * x, 0 * x], axis=-1))
    p = np.array([[0.17, 0.13, 0.23]])
    exact = p[0, 0] ** 2 * p[0, 1] ** 2
    assert reconstruct(field, "tensor-quadratic").sample(p)[0, 0] == pytest.approx(exact)
    assert abs(TotalDegreeTwo(field).sample(p)[0, 0] - exact) > 1e-6


@pytest.mark.parametrize("brho", [0.6, 1.0, -1.2])
def test_adaptive_signed_helix_refines_against_independent_exact_solution(brho):
    # B vertical: analytic circular orbit for zero transverse entrance state.
    field = UniformField((0, 0.3, 0))
    angle = np.arcsin(0.3 / brho)
    expected = np.array([(np.cos(angle) - 1) * brho / 0.3, -np.tan(angle), 0, 0])
    coarse = track_adaptive(
        field,
        np.zeros((1, 4)),
        brho,
        straight_frame(0),
        straight_frame(1),
        position_tolerance_m=1e-5,
        direction_tolerance=1e-5,
    )
    fine = track_adaptive(
        field,
        np.zeros((1, 4)),
        brho,
        straight_frame(0),
        straight_frame(1),
        position_tolerance_m=1e-11,
        direction_tolerance=1e-11,
    )
    assert fine.reached_exit.all()
    assert np.max(np.abs(fine.states[0] - expected)) < 2e-10
    assert np.max(np.abs(fine.states[0] - expected)) < np.max(np.abs(coarse.states[0] - expected))
    assert fine.steps[0] > coarse.steps[0]


def test_nonuniform_refinement_never_reports_equal_ratio_order():
    args = (UniformField((0, 0.4, 0)), np.zeros((1, 4)), 1.0, straight_frame(0), straight_frame(1))
    report = step_convergence(*args, [0.1, 0.09, 0.01])
    assert report["steps"][-1]["observed_order"] == [None] * 4
    with pytest.raises(ValueError):
        step_convergence(*args, [0.1, 0.1, 0.01])
    report = adaptive_convergence(*args, [[1e-6, 1e-6], [1e-9, 1e-9]])
    assert report["common_survivors"] == 1 and report["changed_status_particles"] == 0


def test_reference_persistence_refuses_changed_field_scale_and_reconstruction(tmp_path):
    f = grid(lambda x, y, z: np.stack([0 * x, 0.1 + 0 * x, 0 * x], axis=-1))
    field = TotalDegreeTwo(f)
    orbit = trace_reference(
        field,
        1.0,
        straight_frame(-0.4),
        straight_frame(0.4),
        step_m=None,
        integration_options={"position_tolerance_m": 1e-10, "direction_tolerance": 1e-10},
    )
    path = tmp_path / "reference.npz"
    orbit.save(path)
    loaded = ReferenceOrbit.load(path, field, expected_rigidity_tm=1.0, expected_field_scale=1.0)
    np.testing.assert_array_equal(loaded.lab_states, orbit.lab_states)
    for wrong, scale in [
        (reconstruct(f, "trilinear"), 1.0),
        (field, 2.0),
        (TotalDegreeTwo(f.with_values(f.b_t * 1.001)), 1.0),
    ]:
        with pytest.raises(ValueError, match="mismatch"):
            ReferenceOrbit.load(path, wrong, expected_rigidity_tm=1.0, expected_field_scale=scale)


def test_frame_handedness_and_axial_field_transform():
    o = np.diag([-1.0, 1.0, 1.0])
    _, u, b = transform_vectors(o, [0, 0, 0], [0.2, 0.3, 0.4], [0.5, 0.6, 0.7])
    np.testing.assert_allclose(np.cross(u, b), np.cross([0.2, 0.3, 0.4], [0.5, 0.6, 0.7]) @ o.T)
    with pytest.raises(ValueError, match="right handed"):
        PathFrame(np.zeros(3), [0, 0, 1], [-1, 0, 0], [0, 1, 0])


def test_time_uses_actual_momentum_and_delta_passes_through():
    codec = StaticMagneticCodec(6)
    inputs = np.array([[0, 0, 0, 0, 0, 0.2]])
    out = codec.encode_exit(
        inputs, inputs[:, :4], [2.0], 2.0, actual_p0_kg_m_s=2e-19, mass_kg=1e-27
    )
    assert out[0, 5] == inputs[0, 5] and out[0, 4] < 0
    with pytest.raises(ValueError, match="actual"):
        codec.encode_exit(inputs, inputs[:, :4], [2.0], 2.0)
    ct = StaticMagneticCodec(6, "ct_m").encode_exit(
        inputs, inputs[:, :4], [2.0], 2.0, actual_p0_kg_m_s=2e-19, mass_kg=1e-27
    )
    assert ct[0, 4] == pytest.approx(out[0, 4] * 299792458.0)


def test_durable_run_record_precedes_tool_and_survives_interrupt(tmp_path):
    store = RunStore(tmp_path)

    def interrupted():
        [run_id] = store.list_runs()
        assert store.load(run_id)["status"] == "running"
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        record_call(store, Tool("interrupted", interrupted, "1"), {})
    [run_id] = store.list_runs()
    assert json.loads(store.path_for(run_id).read_text())["status"] == "running"


def test_pooled_metrics_and_distribution_allowlist(tmp_path):
    metric = RegressionMetrics(("um",))
    metric.update([[0.0], [0.0]], [[0.0], [0.0]])
    metric.update([[10.0], [10.0]], [[0.0], [0.0]])
    assert metric.summary()["rmse"][0] == pytest.approx(np.sqrt(50.0))
    path = tmp_path / "release.whl"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("package/core.py", "pass")
        z.writestr("package/field.txt", "0 0 0 1 1 1")
    report = audit_archive(
        path, allowed_members={"package/core.py"}, max_archive_bytes=10000, max_member_bytes=1000
    )
    assert not report["passed"] and report["forbidden_members"] == ["package/field.txt"]


def test_sharp_manufactured_field_refines_across_grid_switches():
    # Numerical manufactured control; no claim this one-component fringe is vacuum Maxwellian.
    axes = [np.linspace(-0.2, 0.2, 5), np.linspace(-0.1, 0.1, 5), np.linspace(-0.1, 1.1, 61)]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    b = np.stack([0 * x, 0.8 * np.exp(-(((z - 0.47) / 0.06) ** 2)), 0 * x], axis=-1)
    field = TotalDegreeTwo(FieldMap(*axes, b))

    def rhs(s, v):
        return np.r_[v[3:], np.cross(v[3:], field.sample(v[None, :3])[0])]

    # Independent solver, tightly capped below the reconstruction variation scale.
    def exit_event(s, v):
        return v[2] - 1.0

    exit_event.terminal = True
    sol = solve_ivp(
        rhs,
        (0, 1.1),
        [0, 0, 0, 0, 0, 1],
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
        max_step=0.001,
        events=exit_event,
    )
    target = sol.y_events[0][0]
    expected = np.array([target[0], target[3] / target[5], target[1], target[4] / target[5]])
    report = []
    for tol in (1e-6, 1e-9):
        r = track_adaptive(
            field,
            np.zeros((1, 4)),
            1.0,
            straight_frame(0),
            straight_frame(1),
            position_tolerance_m=tol,
            direction_tolerance=tol,
        )
        assert r.reached_exit.all()
        report.append(np.max(np.abs(r.states[0] - expected)))
    assert report[1] < report[0] and report[1] < 2e-8


def test_tilted_plane_drift_round_trip():
    angle = 0.3
    exit = PathFrame(
        [0, 0, 1], [np.sin(angle), 0, np.cos(angle)], [np.cos(angle), 0, -np.sin(angle)], [0, 1, 0]
    )
    states = np.array([[0.01, 0.02, -0.03, 0.04]])
    from element_surrogate.lorentz import to_lab, to_plane

    r, t = to_lab(straight_frame(0), states)
    path = ((exit.origin_m - r[0]) @ exit.tangent) / (t[0] @ exit.tangent)
    exact = to_plane(exit, r + path * t, t)
    tracked = track_adaptive(
        UniformField((0, 0, 0)),
        states,
        1.0,
        straight_frame(0),
        exit,
        position_tolerance_m=1e-11,
        direction_tolerance=1e-11,
    )
    np.testing.assert_allclose(tracked.states, exact, atol=3e-15, rtol=0)


def test_replay_refuses_changed_tool_version(tmp_path):
    store = RunStore(tmp_path)
    tool = Tool("constant", lambda: {"value": 1}, "1")
    run = record_call(store, tool, {})["run_id"]
    changed = Tool("constant", tool.func, "2")
    report = replay(store, {"constant": changed}, run)
    assert not report.reproduced and report.changed_implementation


def test_default_tracking_reference_records_degree_two_not_raw_grid_reader():
    field = grid(lambda x, y, z: np.stack([0 * x, 0 * x, 0 * x], axis=-1))
    ref = trace_reference(field, 1.0, straight_frame(-0.2), straight_frame(0.2), step_m=0.05)
    assert ref.metadata["field"]["reconstruction"]["reading"] == "total-degree-two-v1"


def test_strided_identity_is_equal_to_contiguous_copy():
    values = np.arange(96.0, dtype=float).reshape(2, 3, 4, 4).transpose(1, 0, 2, 3)
    assert array_digest(values) == array_digest(np.ascontiguousarray(values))
    with pytest.raises(ValueError, match="object"):
        array_digest(np.array([object()], dtype=object))
