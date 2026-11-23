"""Tests for the MCP server: the tools it serves and the records its calls leave."""

import asyncio
import inspect

import numpy as np
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from element_surrogate.bend import NominalBend
from element_surrogate.fields import FieldMap
from element_surrogate.mcp_server import PROJECT_ENV, build_server, project_dir
from element_surrogate.runs import RunStore
from element_surrogate.synthetic import (
    bend_sources,
    multipole_sources,
    plant_duplicated_plane,
    point_source_field,
)
from element_surrogate.tools import REGISTRY
from tests.cst_format import write_cst_hdf5
from tests.track_format import write_track_table


@pytest.fixture
def store(tmp_path):
    return RunStore(tmp_path / "project")


def _call(server, name, arguments):
    result = asyncio.run(server.call_tool(name, arguments))
    assert not result.is_error, result.content
    return result.structured_content


def test_every_registered_tool_is_served_from_its_single_definition(store):
    tools = {tool.name: tool for tool in asyncio.run(build_server(store).list_tools())}
    assert set(REGISTRY) | {"list_runs", "replay_run"} == set(tools)
    for name, registered in REGISTRY.items():
        assert tools[name].description == inspect.getdoc(registered.func)
        parameters = set(inspect.signature(registered.func).parameters)
        assert set(tools[name].input_schema["properties"]) == parameters
        assert tools[name].annotations.read_only_hint is registered.read_only
    assert set(tools["analyze_harmonics"].input_schema["required"]) == {
        "path",
        "fit_radius_m",
        "reference_radius_m",
    }


def test_call_through_server_is_recorded_and_replays(store, tmp_path):
    field = tmp_path / "field.txt"
    field.write_text("0 0 0 0.0 0.0 1.0\n", encoding="utf-8")
    server = build_server(store)

    summary = _call(server, "inspect_file", {"path": str(field)})
    assert summary["size_bytes"] == field.stat().st_size

    listed = _call(server, "list_runs", {})
    assert listed["run_ids"] == [summary["run_id"]]

    report = _call(server, "replay_run", {"run_id": summary["run_id"]})
    assert report["reproduced"] is True


def test_field_tools_through_server(store, tmp_path):
    axis = np.linspace(-0.01, 0.01, 9)
    z = np.linspace(-0.1, 0.1, 21)
    quad = point_source_field(multipole_sources(2, 0.04, 0.1, 1e-3, 11), axis, axis, z)
    empty = FieldMap(axis, axis, z, np.zeros_like(quad.b_t))
    defective = plant_duplicated_plane(quad, axis=2, index=5).field
    path = str(write_track_table(tmp_path / "coils.table", [empty, defective]))
    server = build_server(store)

    summary = _call(server, "read_field_file", {"path": path})
    assert summary["format"] == "track_table"
    assert summary["integrity"]["all_zero_slots"] == ["slot0"]

    # Two slots and none chosen: the tool says so instead of guessing.
    with pytest.raises(ToolError, match="choose one"):
        asyncio.run(server.call_tool("check_maxwell", {"path": path}))

    maxwell = _call(server, "check_maxwell", {"path": path, "slot": 1, "shells": 2})
    assert maxwell["source"]["slot"] == "slot1"
    assert maxwell["region"]["shape"] == "circle"
    assert len(maxwell["shells"]) == 2

    order = _call(server, "check_storage_order", {"path": path, "slot": 1})
    assert order["as_read_rank"] >= 1

    defects = _call(server, "scan_node_defects", {"path": path, "slot": 1})
    assert defects["duplicated_planes"] == [{"axis": "z", "planes": [5, 6]}]

    report = _call(server, "replay_run", {"run_id": maxwell["run_id"]})
    assert report["reproduced"] is True

    harmonics = _call(
        server,
        "analyze_harmonics",
        {"path": path, "slot": 1, "fit_radius_m": 0.008, "reference_radius_m": 0.005},
    )
    assert harmonics["main_harmonic_n"] == 2
    assert harmonics["region"]["radius_m"] == 0.008

    symmetry = _call(server, "check_mirror_symmetry", {"path": path, "slot": 1})
    assert symmetry["x_mirror"]["By"]["odd_defect"] < 1e-9

    nodes = _call(server, "check_harmonic_nodes", {"path": path, "slot": 1, "examples": 3})
    assert len(nodes["worst_nodes"]) == 3
    assert nodes["floor"]["median_implied_error_t"] > 0


def test_bend_and_description_tools_through_server(store, tmp_path):
    bend = NominalBend(1.0, 0.4, 0.0)
    field = point_source_field(
        bend_sources(bend, 0.02, 0.03, 1.0e-3, 60, 7),
        np.linspace(-0.12, 0.08, 51),
        np.linspace(-0.01, 0.01, 5),
        np.linspace(-0.4, 0.4, 81),
    )
    path = str(write_cst_hdf5(tmp_path / "bend.h5", field))
    server = build_server(store)

    frame = _call(server, "analyze_field_frame", {"path": path})
    proposed = frame["proposed_bend"]
    assert frame["source"]["format"] == "cst_hdf5"
    assert proposed["bends_toward"] == "-x"

    bend_arg = {k: proposed[k] for k in ("radius_m", "angle_rad", "arm_m", "bends_toward")}
    region = {"shape": "rectangle", "radius_m": 0.02, "half_height_m": 0.006, "bend": bend_arg}
    maxwell = _call(server, "check_maxwell", {"path": path, "shells": 2, **region})
    assert maxwell["region"]["axis"] == "a bend's nominal path"

    ends = _call(server, "check_field_ends", {"path": path, "tolerance": 0.5})
    assert ends["findings"] == []

    description = {"kind": "dipole", "bend": {"radius_m": 1.0, "angle_rad": 0.4}, "tolerance": 0.05}
    compared = _call(server, "compare_with_description", {"path": path, "description": description})
    assert [f["kind"] for f in compared["findings"]] == ["aperture_not_described"]
    report = _call(server, "replay_run", {"run_id": compared["run_id"]})
    assert report["reproduced"] is True


def test_gradient_and_export_tools_through_server(store, tmp_path):
    sources = multipole_sources(2, 0.04, 0.1, 1e-3, 21)
    z = np.linspace(-0.2, 0.2, 81)
    fine = point_source_field(
        sources, np.linspace(-0.01, 0.01, 11), np.linspace(-0.01, 0.01, 11), z
    )
    coarse = point_source_field(
        sources, np.linspace(-0.01, 0.01, 6), np.linspace(-0.01, 0.01, 6), z
    )
    first = str(write_track_table(tmp_path / "fine.table", [fine]))
    second = str(write_cst_hdf5(tmp_path / "coarse.h5", coarse))
    server = build_server(store)

    gradients = _call(server, "analyze_generalized_gradients", {"path": first, "radius_m": 0.006})
    assert gradients["main_harmonic_n"] == 2

    compared = _call(server, "compare_field_exports", {"path": first, "other_path": second})
    assert compared["polarity"] == 1.0 and not compared["same_grid"]
    assert compared["other_source"]["format"] == "cst_hdf5"
    report = _call(server, "replay_run", {"run_id": compared["run_id"]})
    assert report["reproduced"] is True


def test_tracking_tools_through_server(store, tmp_path):
    bend = NominalBend(1.0, 0.4, 0.0)
    bent = point_source_field(
        bend_sources(bend, 0.02, 0.03, 1.0e-3, 60, 7),
        np.linspace(-0.12, 0.08, 51),
        np.linspace(-0.01, 0.01, 5),
        np.linspace(-0.4, 0.4, 81),
    )
    path = str(write_cst_hdf5(tmp_path / "bend.h5", bent))
    server = build_server(store)
    proposed = _call(server, "analyze_field_frame", {"path": path})["proposed_bend"]
    arguments = {
        "path": path,
        "rigidity_tm": proposed["body_field_t"] * proposed["radius_m"],
        # the proposed arms reach the grid's ends; tracking planes must lie inside the field
        "bend": {
            **{k: proposed[k] for k in ("radius_m", "angle_rad", "bends_toward")},
            "arm_m": proposed["arm_m"] - 0.05,
        },
        "scale_bracket": [0.85, 1.2],  # around the hard-edge estimate; wider leaves this grid
        "reconstruction": "trilinear",  # pin this historical compatibility control
        "step_m": 0.002,
        "position_tolerance_m": 1e-6,
        "slope_tolerance_rad": 1e-6,
    }
    orbit = _call(server, "fit_reference_orbit", arguments)
    assert 0.85 < orbit["field_scale"] < 1.2
    assert abs(orbit["exit_state_on_nominal_plane"]["xp_rad"]) < 1e-9

    quad = point_source_field(
        multipole_sources(2, 0.04, 0.1, 1e-3, 21),
        np.linspace(-0.01, 0.01, 11),
        np.linspace(-0.01, 0.01, 11),
        np.linspace(-0.2, 0.2, 81),
    )
    table = str(write_track_table(tmp_path / "quad.table", [quad]))
    ranges = {"x_m": 0.003, "xp_rad": 0.002, "y_m": 0.003, "yp_rad": 0.002}
    report = _call(
        server,
        "check_tracking_convergence",
        {
            "path": table,
            "rigidity_tm": 0.5,
            "steps_m": [0.02, 0.01, 0.005],
            "half_ranges": ranges,
            "particles": 5,
        },
    )
    assert report["reached_at_every_step"] == 5
    assert len(report["steps"]) == 2


def test_project_dir_comes_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(PROJECT_ENV, str(tmp_path))
    assert project_dir() == tmp_path
    monkeypatch.delenv(PROJECT_ENV)
    assert project_dir().resolve() == type(tmp_path).cwd().resolve()
