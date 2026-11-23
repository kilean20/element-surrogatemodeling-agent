"""Scientific metadata: missing facts, different length meanings and source axes."""

from types import SimpleNamespace

import numpy as np
import pytest

from element_surrogate.card_metadata import inspect_request
from element_surrogate.model_provenance import collect, render


class AxesOnly:
    x_m = np.array([-0.03, -0.01, 0.02])
    y_m = np.array([-0.007, 0.0, 0.007])
    z_m = np.array([-0.6, 0.0, 0.6])
    source = None

    @property
    def b_t(self):
        raise AssertionError("Card metadata must not load the field-vector payload")


def transport():
    return SimpleNamespace(grid=AxesOnly(), reference_path_m=1.243)


def facts():
    return {
        "physical_provenance": {
            "design_length": {
                "value_m": 1.25,
                "definition": "nominal native path",
                "source_reference": "manufactured layout",
            },
            "solver_mesh": {
                "description": "adaptive unstructured elements; local max 0.7 mm",
                "source_reference": "manufactured solver record",
            },
            "field_calculation_currents": [
                {
                    "current_a": 7.25,
                    "description": "coil H only",
                    "source_reference": "manufactured H solve",
                },
                {
                    "current_a": -2.75,
                    "description": "coil V only",
                    "source_reference": "manufactured V solve",
                },
                {
                    "current_a": 0,
                    "description": "deenergized control",
                    "source_reference": "manufactured control",
                },
            ],
        }
    }


def test_actual_nonuniform_axes_and_extent_without_payload_access():
    report = collect(transport(), facts())
    assert report["export_grid"]["nodes_xyz"] == [3, 3, 3]
    assert report["export_grid"]["spacing_m"]["x"] == pytest.approx({"min": 0.02, "max": 0.03})
    assert report["export_grid"]["extent_m"]["y"] == [-0.007, 0.007]
    assert "20 .. 30" in render(report)


def test_design_is_neither_traced_length_nor_grid_extent():
    report = collect(transport(), facts())
    assert report["design_length"]["value_m"] == 1.25
    assert report["traced_reference_length_m"] == 1.243
    assert report["export_grid"]["extent_m"]["z"] == [-0.6, 0.6]


def test_unknowns_remain_visible_without_guessing_from_reference_or_scale():
    report = collect(transport(), {})
    assert report["design_length"]["value_m"] is None
    assert report["solver_mesh"] is None
    assert report["field_calculation_currents"] is None
    text = render(report)
    assert "| Design length | Not recorded |" in text
    assert "| Internal field-solver mesh | Not recorded |" in text
    assert "Field-calculation input current(s) | Not recorded" in text
    assert inspect_request({"model": {"physical_provenance": None}})["valid"]


def test_signed_zero_and_independent_coil_descriptions_survive():
    model = facts()
    assert inspect_request({"model": model})["valid"]
    text = render(collect(transport(), model))
    assert "+7.25 A — coil H only" in text
    assert "-2.75 A — coil V only" in text
    assert "+0 A — deenergized control" in text


def test_legacy_frontmatter_length_retained_with_unknown_definition_source():
    report = collect(transport(), {"frontmatter": {"design_length_m": 1.25}})
    assert report["design_length"]["value_m"] == 1.25
    assert "design_length.definition" in report["missing_metadata"]
    assert "definition not recorded" in render(report)


def test_conflicting_lengths_fail_preflight_instead_of_silent_override():
    model = facts()
    model["frontmatter"] = {"design_length_m": 1.4}
    result = inspect_request({"model": model})
    assert not result["valid"]
    assert any("conflicts" in row["message"] for row in result["findings"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 10**400])
def test_nonfinite_current_fails_before_field_loading(value):
    model = facts()
    model["physical_provenance"]["field_calculation_currents"][0]["current_a"] = value
    assert not inspect_request({"model": model})["valid"]


@pytest.mark.parametrize("value", [-1, 0, True, "1 m", float("nan"), float("inf"), 10**400])
def test_invalid_design_length_is_not_a_valid_fact(value):
    model = facts()
    model["physical_provenance"]["design_length"]["value_m"] = value
    assert not inspect_request({"model": model})["valid"]


def test_known_fact_needs_a_referenced_source():
    model = facts()
    model["physical_provenance"]["solver_mesh"]["source_reference"] = " "
    assert not inspect_request({"model": model})["valid"]
