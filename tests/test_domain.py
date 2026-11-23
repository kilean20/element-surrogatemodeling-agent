"""Stencil support is an executable invariant, not a geometric clearance guess."""

import numpy as np
import pytest

from element_surrogate.domain import SupportedField, perturb_outside, probe_domain
from element_surrogate.fields import FieldMap
from element_surrogate.lorentz import Status, straight_frame, track
from element_surrogate.readings import reconstruct
from element_surrogate.region import Region


def grid():
    a = np.linspace(-0.04, 0.04, 9)
    z = np.linspace(-0.1, 0.1, 21)
    return FieldMap(a, a, z, np.zeros((9, 9, 21, 3)))


@pytest.mark.parametrize(
    "method,count", [("total-degree-two", 19), ("tensor-quadratic", 27), ("trilinear", 8)]
)
def test_node_selectors_are_actual_sample_reads(method, count):
    f = grid()
    indices = []

    class Spy:
        def __getitem__(self, key):
            indices.append(np.stack(np.broadcast_arrays(*key), -1))
            return f.b_t[key]

    import copy

    observed = copy.copy(f)
    object.__setattr__(observed, "b_t", Spy())
    points = np.array([[0.015, 0.011, -0.075], [0.04, -0.04, 0.1]])
    reader = reconstruct(observed, method)
    selected = reader.stencil_nodes(points)
    reader.sample(points)
    assert selected.shape == (2, count, 3)
    actual = np.stack(indices, axis=1)
    for row in range(2):
        assert set(map(tuple, actual[row])) == set(map(tuple, selected[row]))


def test_inside_particle_can_use_outside_nodes_and_zero_field_perturbation_responds():
    f = grid()
    support = Region("circle", 0.027)
    state = np.array([[0.024, 0, 0, 0]])
    assert support.contains(np.array([[0.024, 0, 0]])).all()
    reading = SupportedField(f, support, enforce=False)
    reading.sample(np.array([[0.024, 0, 0]]))
    assert reading.outside_queries == 1
    strict = SupportedField(f, support)
    assert not strict.contains(np.array([[0.024, 0, 0]])).all()
    with pytest.raises(ValueError, match="stencil"):
        strict.sample(np.array([[0.024, 0, 0]]))
    report = probe_domain(
        f,
        state,
        2.0,
        straight_frame(-0.08),
        straight_frame(0.08),
        support,
        Region("circle", 0.026),
        {"method": "fixed-rk4", "step_m": 0.01},
        perturbation_t=0.01,
    )
    assert report["survivors"] == 1
    assert report["survivors_reading_outside_support"] == 1
    assert not report["probe_results"][0]["perturbation_exactly_unchanged"]
    assert np.count_nonzero(f.b_t) == 0


def test_central_trajectory_is_exactly_isolated_and_strict_support_rejects_outer_one():
    f = grid()
    support = Region("circle", 0.027)
    report = probe_domain(
        f,
        np.zeros((1, 4)),
        2.0,
        straight_frame(-0.08),
        straight_frame(0.08),
        support,
        Region("circle", 0.026),
        {"method": "fixed-rk4", "step_m": 0.01},
        perturbation_t=0.01,
    )
    assert report["all_retained_probes_isolated"]
    assert report["probe_results"][0]["perturbation_exactly_unchanged"]
    result = track(
        SupportedField(f, support),
        np.array([[0.024, 0, 0, 0]]),
        2.0,
        straight_frame(-0.08),
        straight_frame(0.08),
        step_m=0.01,
    )
    assert result.status[0] == Status.OUTSIDE_FIELD


def test_tensor_corner_support_differs_from_total_degree_two():
    f = grid()
    # At the centre node, tensor reads (dx,dy,dz) corners, total degree two does not.
    support = Region("circle", 0.01 * np.sqrt(2) + 1e-10, z_min_m=-0.001, z_max_m=0.001)
    # Longitudinal stencil margin is still necessary even on a transverse axis.
    assert not SupportedField(f, support).contains(np.zeros((1, 3))).all()
    assert not SupportedField(f, support, "tensor-quadratic").contains(np.zeros((1, 3))).all()


def test_no_survivors_is_inconclusive():
    f = grid()
    report = probe_domain(
        f,
        np.array([[0.03, 0, 0, 0]]),
        2.0,
        straight_frame(-0.08),
        straight_frame(0.08),
        Region("circle", 0.027),
        Region("circle", 0.02),
        {"method": "fixed-rk4", "step_m": 0.01},
    )
    assert report["survivors"] == 0
    assert not report["all_retained_probes_isolated"]


def test_overlay_changes_only_outside_nodes_and_invalid_inputs_refused():
    f = grid()
    support = Region("circle", 0.027)
    view = perturb_outside(f, support, 0.1)
    p = np.array([[0.0, 0.0, 0.0], [0.04, 0, 0]])
    original = reconstruct(f).sample(p)
    changed = reconstruct(view).sample(p)
    np.testing.assert_array_equal(changed[0], original[0])
    assert np.any(changed[1] != original[1])
    with pytest.raises(ValueError):
        perturb_outside(f, support, 0)


def test_public_support_corpus_has_no_misses_or_false_alarms():
    from element_surrogate.corpus import run_tracking_support_corpus

    report = run_tracking_support_corpus()
    assert report["missed"] == [] and report["false_alarms"] == []
    assert all(r["correct"] for r in report["cases"])
