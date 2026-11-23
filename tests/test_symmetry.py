"""Case-linked equivariance evidence, with a physically valid asymmetric control."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from element_surrogate.symmetry import assess, validate
from tests.workflow_fixture import manufactured_transport

GROUP = [[1] * 6, [-1, -1, -1, -1, 1, 1]]


def measured(transport):
    _, _, ranges, _ = manufactured_transport()
    return assess(
        transport,
        ranges,
        [{"id": "control", "effective_rigidity_tm": 2.0, "field_scale": 1.0}],
        GROUP,
        [1e-10] * 4,
        random_probes=2,
        seed=17,
        rationale=(
            "Manufactured local vacuum map; test central reflection on the declared finite cohort"
        ),
    )


def test_symmetric_map_record_is_measured_and_scope_bound():
    transport, *_ = manufactured_transport()
    record = measured(transport)
    assert record["supported"] and record["reports"][0]["common_survivors"] > 0
    validate(record, GROUP, 4, expected_scope=record["scope"], require_current=True)
    wrong = copy.deepcopy(record["scope"])
    wrong["ranges"][0][1] *= 0.5
    with pytest.raises(ValueError, match="scope differs"):
        validate(record, GROUP, 4, expected_scope=wrong)


def test_uniform_dipole_offset_is_real_asymmetry_and_ineligible():
    transport, *_ = manufactured_transport()
    b = transport.grid.b_t.copy()
    b[..., 0] += 0.02
    asymmetric = replace(transport, grid=transport.grid.with_values(b))
    record = measured(asymmetric)
    assert not record["supported"]
    assert np.max(record["reports"][0]["max_component_difference"]) > 1e-10
    with pytest.raises(ValueError, match="does not support"):
        validate(record, GROUP, 4)


def test_dispositions_cannot_hide_failed_symmetry_measurements():
    transport, *_ = manufactured_transport()
    record = measured(transport)
    record["reports"][0]["max_component_difference"][0] = 0.1
    with pytest.raises(ValueError, match="unsupported measured"):
        validate(record, GROUP, 4)
