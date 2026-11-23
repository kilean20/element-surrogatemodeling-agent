"""Boundary distance, longitudinal span, design and traced lengths differ."""

from types import SimpleNamespace

import numpy as np
import pytest

from element_surrogate.card_metadata import inspect_request
from element_surrogate.model_provenance import collect, render
from tests.test_model_provenance import AxesOnly


def transport():
    return SimpleNamespace(
        grid=AxesOnly(),
        reference_path_m=7.0,
        entry=SimpleNamespace(origin_m=np.array([0.0, 0.0, 0.0])),
        exit=SimpleNamespace(origin_m=np.array([3.0, 4.0, 0.0])),
    )


def test_bend_chord_is_not_relabelled_as_design_or_traced_path():
    model = {
        "physical_provenance": {
            "design_length": {
                "value_m": 6.0,
                "definition": "manufactured design path",
                "source_reference": "synthetic",
            },
            "longitudinal_span": {
                "input_m": -2.0,
                "output_m": 6.0,
                "definition": "manufactured unwrapped design station",
                "source_reference": "synthetic",
            },
        }
    }
    assert inspect_request({"model": model})["valid"]
    report = collect(transport(), model)
    assert report["native_boundary_geometry"]["input_output_separation_m"] == 5.0
    assert report["model_length_m"] == 8.0
    assert report["design_length"]["value_m"] == 6.0
    assert report["traced_reference_length_m"] == 7.0
    assert report["longitudinal_span"]["value_m"] == 8.0
    assert "chord" in render(report)
    assert "abs(output - input)" in render(report)


def test_unknown_longitudinal_convention_is_not_guessed_from_other_lengths():
    report = collect(transport(), {})
    assert report["native_boundary_geometry"]["input_output_separation_m"] == 5.0
    assert report["model_length_m"] is None
    assert report["longitudinal_span"] is None
    assert report["design_length"]["value_m"] is None
    assert "Model longitudinal input-to-output length | Not recorded" in render(report)


def test_conflicting_caller_length_fails():
    with pytest.raises(ValueError, match="conflicts.*longitudinal"):
        collect(
            transport(),
            {
                "frontmatter": {"model_length_m": 7.0},
                "physical_provenance": {
                    "longitudinal_span": {
                        "input_m": 0.0,
                        "output_m": 8.0,
                        "definition": "s",
                        "source_reference": "synthetic",
                    }
                },
            },
        )


def test_unreferenced_longitudinal_frontmatter_is_not_a_verified_length():
    result = inspect_request({"model": {"frontmatter": {"model_longitudinal_span_m": 6.0}}})
    assert not result["valid"]
    assert any("requires referenced" in row["message"] for row in result["findings"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1 m"])
def test_bad_longitudinal_station_fails_metadata_preflight(value):
    model = {
        "physical_provenance": {
            "longitudinal_span": {
                "input_m": value,
                "output_m": 1.0,
                "definition": "s",
                "source_reference": "synthetic",
            }
        }
    }
    assert not inspect_request({"model": model})["valid"]
