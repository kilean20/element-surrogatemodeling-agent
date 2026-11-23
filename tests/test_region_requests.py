"""Unambiguous input serialization, legacy compatibility and early refusal."""

import copy

import pytest

from element_surrogate.benchmark.protocol import protocol_identity, write_json
from element_surrogate.bend import NominalBend
from element_surrogate.region import Region
from element_surrogate.workflow import load_transport, region_from_dict


@pytest.mark.parametrize("direction", [True, False])
def test_entrance_and_midpoint_origins_resolve_same_path_region(direction):
    path = NominalBend(0.8, 0.4, 0.1, direction)
    legacy = {
        "shape": "rectangle",
        "radius_m": 0.02,
        "half_height_m": 0.01,
        "z_min_m": -0.5 * path.length_m,
        "z_max_m": 0.5 * path.length_m,
        "bend": path.to_request(),
    }
    region = region_from_dict(legacy)
    entrance = region.to_request(origin="path_entrance")
    assert entrance["longitudinal_bounds"]["min_m"] == 0.0
    assert entrance["longitudinal_bounds"]["max_m"] == path.length_m
    for request in (entrance, region.to_request()):
        assert region_from_dict(request).as_dict() == region.as_dict()
    for s in (0.0, 0.5 * path.length_m, path.length_m):
        assert region.contains(path.frame_at(s).origin_m)


def test_straight_origin_is_laboratory_not_shifted_to_field_entrance():
    region = Region("circle", 0.01, -0.2, 0.3)
    assert Region.from_request(region.to_request()).as_dict() == region.as_dict()
    assert region.to_request()["longitudinal_bounds"] == {
        "origin": "laboratory",
        "min_m": -0.2,
        "max_m": 0.3,
    }


def test_preview_distinguishes_full_path_from_body_only_region():
    from element_surrogate.tools import describe_region_request

    path = NominalBend(0.8, 0.4, 0.1)
    full = Region("circle", 0.01, -0.5 * path.length_m, 0.5 * path.length_m, path=path)
    body = Region("circle", 0.01, -0.05, 0.05, path=path)
    full_preview = describe_region_request(full.to_request())
    body_preview = describe_region_request(body.to_request())
    assert all(row["axis_point_in_region"] for row in full_preview["nominal_landmarks"].values())
    assert body_preview["nominal_landmarks"]["midpoint"]["axis_point_in_region"]
    assert not body_preview["nominal_landmarks"]["entrance"]["axis_point_in_region"]
    assert not body_preview["nominal_landmarks"]["exit"]["axis_point_in_region"]


@pytest.mark.parametrize(
    "bad",
    [
        {"coordinate_basis": "nominal_path"},
        {"longitudinal_bounds": {"origin": "path_entrance", "min_m": 0, "max_m": 1}},
        {"z_min_m": 0, "longitudinal_bounds": {"origin": "laboratory", "min_m": 0, "max_m": 1}},
        {"longitudinal_bounds": {"origin": "laboratory", "min_m": 0}},
        {"longitudinal_bounds": {"origin": "laboratory", "min_m": 0, "max_m": float("inf")}},
        {"longitudinal_bounds": {"origin": "laboratory", "min_m": 1, "max_m": 0}},
    ],
)
def test_wrong_basis_origins_mixed_or_nonfinite_bounds_refused(bad):
    with pytest.raises(ValueError):
        region_from_dict({"shape": "circle", "radius_m": 0.01, **bad})


def test_description_cannot_be_confused_with_accepted_request():
    path = NominalBend(0.8, 0.4, 0.1)
    assert set(path.to_request()) == {"radius_m", "angle_rad", "arm_m", "bends_toward"}
    with pytest.raises(ValueError, match="unknown bend keys"):
        region_from_dict({"shape": "circle", "radius_m": 0.01, "bend": path.as_dict()})


def test_malformed_region_fails_before_any_source_hash_or_load(tmp_path, monkeypatch):
    spec = {
        "schema": "element_surrogate.transport.v1",
        "native_planes": True,
        "support": {"shape": "circle", "radius_m": 0.01, "coordinate_basis": "nominal_path"},
    }
    write_json(tmp_path / "transport.json", spec)

    def forbidden(*args, **kwargs):
        raise AssertionError("field accessed before validation")

    monkeypatch.setattr("element_surrogate.workflow.file_hash", forbidden)
    monkeypatch.setattr("element_surrogate.workflow.load_field_file", forbidden)
    with pytest.raises(ValueError, match="coordinate_basis"):
        load_transport(tmp_path / "transport.json")


def test_resource_error_cannot_be_accepted_as_numerical_field_limit(monkeypatch):
    from element_surrogate.field_readiness import assess, collect_evidence
    from tests.workflow_fixture import manufactured_transport, readiness_for_test

    transport, *_ = manufactured_transport()
    good = readiness_for_test(transport)

    def out_of_memory(*args, **kwargs):
        raise MemoryError("deliberate control")

    monkeypatch.setattr("element_surrogate.field_readiness.maxwell_report", out_of_memory)
    evidence = collect_evidence(transport, shells=2)
    assert "MemoryError" in evidence["diagnostics"]["maxwell"]["diagnostic_error"]
    review = copy.deepcopy(good["review"])
    review["evidence_sha256"] = protocol_identity(evidence)
    review["issue_resolutions"] = {
        key: {
            "status": "accepted_numerical_limitation",
            "rationale": "Test an attempted computation bypass",
            "evidence_refs": ["maxwell"],
        }
        for key in evidence["issues"]
    }
    assert not assess(evidence, review)["generation_eligible"]
