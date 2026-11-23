import numpy as np
import pytest

from element_surrogate import symplectic
from element_surrogate.lorentz import UniformField, straight_frame, track


@pytest.mark.parametrize("brho", [1.5, -1.5])
@pytest.mark.parametrize("b", [(0, 0, 0), (0, 0, 0.4), (0.1, 0.2, 0.4)])
def test_magnetic_two_form_exact_uniform_control(brho, b):
    states = np.array([[0.001, 0.02, -0.002, -0.01], [-0.001, -0.01, 0.001, 0.02]])

    def mapping(values):
        result = track(
            UniformField(b), values, brho, straight_frame(0), straight_frame(0.2), step_m=0.002
        )
        assert result.reached_exit.all()
        return result.states

    result = symplectic.diagnose(
        mapping,
        lambda s, e: np.full(len(s), b[2]),
        states,
        brho,
        1.0,
        q_scale_m=0.01,
        p_scale=0.03,
        relative_steps=[1e-3, 5e-4, 2.5e-4],
    )
    assert result["refinements"][-1]["max_relative_two_form_defect"] < 2e-7
    if b[2] and b[0]:
        wrong = symplectic.diagnose(
            mapping,
            lambda s, e: np.zeros(len(s)),
            states,
            brho,
            1.0,
            q_scale_m=0.01,
            p_scale=0.03,
            relative_steps=[1e-3, 5e-4, 2.5e-4],
        )
        assert wrong["refinements"][-1]["max_relative_two_form_defect"] > 1e-5


def test_non_symplectic_scaling_and_invalid_resolution_detected():
    states = np.array([[0.001, 0.02, 0.002, 0.01]])

    def wrong(s):
        out = s.copy()
        out[:, 0] *= 1.1
        return out

    result = symplectic.diagnose(
        wrong,
        lambda s, e: np.zeros(len(s)),
        states,
        1.0,
        1.0,
        q_scale_m=0.01,
        p_scale=0.03,
        relative_steps=[1e-3, 5e-4, 2.5e-4],
    )
    assert result["refinements"][-1]["max_relative_two_form_defect"] > 0.01
    with pytest.raises(ValueError, match="three decreasing"):
        symplectic.diagnose(
            wrong,
            lambda s, e: np.zeros(len(s)),
            states,
            1.0,
            1.0,
            q_scale_m=0.01,
            p_scale=0.03,
            relative_steps=[1e-4, 1e-3],
        )
