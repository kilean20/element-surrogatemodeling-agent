"""Tests for the comparison of field data with the user's description."""

import numpy as np
import pytest

from element_surrogate.analysis import harmonics
from element_surrogate.bend import NominalBend
from element_surrogate.description import compare_description, unit_explanations
from element_surrogate.region import Region
from element_surrogate.synthetic import (
    bend_sources,
    multipole_sources,
    plant_gauss_as_tesla,
    point_source_field,
)

APERTURE = {"radius_m": 0.02}


@pytest.fixture(scope="module")
def quad():
    axis = np.linspace(-0.02, 0.02, 9)
    return point_source_field(
        multipole_sources(2, 0.04, 0.2, 2.0e-4, 41), axis, axis, np.linspace(-0.6, 0.6, 121)
    )


@pytest.fixture(scope="module")
def gradient(quad):
    region = Region.covering(quad)
    return harmonics(quad, region, region.radius_m)["integrated_t_m"][1]["normal"] / 0.02


def _kinds(result):
    return [f["kind"] for f in result["findings"]]


def test_a_matching_description_has_no_findings(quad, gradient):
    text = {"kind": "quadrupole", "integrated_strength": gradient, "aperture": APERTURE}
    result = compare_description(quad, {**text, "tolerance": 1e-6})
    assert result["findings"] == []
    assert result["comparisons"]["integrated_strength"]["measured_over_stated"] == pytest.approx(1)
    assert result["comparisons"]["main_harmonic"] == {"stated": 2, "measured": 2}
    assert result["not_described"] == []


def test_unit_mislabels_are_named():
    assert "gauss read as tesla" in unit_explanations(1.0e4, 2, 1e-3)
    assert unit_explanations(1.0e2, 2, 1e-3) == []  # an integrated gradient has no length unit
    assert "centimetres read as metres" in unit_explanations(1.0e2, 1, 1e-3)
    assert unit_explanations(1.3, 2, 1e-3) == []


def test_gauss_read_as_tesla(quad, gradient):
    field = plant_gauss_as_tesla(quad).field
    text = {"kind": "quadrupole", "integrated_strength": gradient, "aperture": APERTURE}
    [mismatch] = compare_description(field, {**text, "tolerance": 1e-3})["findings"]
    assert mismatch["kind"] == "integrated_strength_mismatch"
    assert "gauss read as tesla" in mismatch["summary"]


def test_current_label_polarity_and_kind(quad, gradient):
    base = {"kind": "quadrupole", "aperture": APERTURE, "tolerance": 1e-3}
    labelled = {**base, "current_a": 125.0, "strength_per_ampere": gradient / 100.0}
    [mismatch] = compare_description(quad, labelled)["findings"]
    assert "current label" in mismatch["summary"]
    reversed_ = compare_description(quad, {**base, "integrated_strength": -gradient})
    assert "polarity reversed" in reversed_["findings"][0]["summary"]
    assert _kinds(compare_description(quad, {**base, "kind": "sextupole"})) == [
        "main_harmonic_mismatch"
    ]


def test_missing_items(quad, gradient):
    result = compare_description(quad, {"integrated_strength": 2 * gradient})
    assert _kinds(result) == ["aperture_not_described"]  # no tolerance: numbers not judged
    assert result["comparisons"]["integrated_strength"]["measured_over_stated"] == pytest.approx(
        0.5
    )
    assert set(result["not_described"]) == {"kind", "aperture", "tolerance"}
    with pytest.raises(ValueError, match="unknown description keys"):
        compare_description(quad, {"colour": "red"})


@pytest.fixture(scope="module")
def bend_field():
    bend = NominalBend(2.0, 0.3, 0.0)
    sources = bend_sources(bend, 0.02, 0.04, 1.0e-3, 80, 9)
    return point_source_field(
        sources,
        np.linspace(-0.16, 0.16, 81),
        np.linspace(-0.01, 0.01, 5),
        np.linspace(-0.5, 0.5, 101),
    )


def test_a_described_bend_is_compared_with_the_field_frame(bend_field):
    text = {"kind": "dipole", "aperture": {"half_gap_m": 0.01}, "tolerance": 0.02}
    good = compare_description(bend_field, {**text, "bend": {"radius_m": 2.0, "angle_rad": 0.3}})
    assert good["findings"] == []
    assert good["comparisons"]["main_harmonic"] == {"stated": 1, "measured": 1}
    wrong = {"radius_m": 3.0, "angle_rad": 0.3, "bends_toward": "+x"}
    assert _kinds(compare_description(bend_field, {**text, "bend": wrong})) == [
        "radius_m_mismatch",
        "bend_direction_mismatch",
    ]


def test_a_bend_described_for_a_straight_field(quad):
    text = {"bend": {"radius_m": 2.0, "angle_rad": 0.3}, "aperture": APERTURE}
    wide = point_source_field(
        multipole_sources(1, 0.03, 0.2, 1e-3, 41),
        np.linspace(-0.1, 0.1, 41),
        np.linspace(-0.01, 0.01, 5),
        np.linspace(-0.6, 0.6, 121),
    )
    assert "bend_not_found" in _kinds(compare_description(wide, text))
