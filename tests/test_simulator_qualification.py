"""A zero exit code with wrong physics must not qualify a simulator build."""

from pathlib import Path

import numpy as np
import pytest

from element_surrogate import simulator_qualification as sq
from element_surrogate.benchmark.protocol import file_hash, write_json
from element_surrogate.fields import FieldMap
from element_surrogate.track_adapter import TrackRun
from tests.track_format import write_track_table


def request(tmp_path):
    axis = np.linspace(-0.02, 0.02, 5)
    grid = FieldMap(axis, axis, np.linspace(-0.15, 0.15, 7), np.zeros((5, 5, 7, 3)))
    field = write_track_table(tmp_path / "field.dat", [grid])
    exe = tmp_path / "synthetic.exe"
    exe.write_bytes(b"manufactured executable fixture")
    states = np.array([[0.001, 0.002, -0.001, -0.003], [0.0, 0.0, 0.0, 0.0]])
    expected = states.copy()
    expected[:, [0, 2]] += 0.3 * states[:, [1, 3]]
    ref = tmp_path / "reference.npz"
    np.savez(
        ref, states=states, expected=expected, rigidity_tm=[1.0, 1.3], reached_exit=np.ones(2, bool)
    )
    r = {
        "schema": "element_surrogate.track_compatibility.v1",
        "runtime": {"executable": str(exe), "libraries": []},
        "beam": {"mass_number": 40, "charge": 18, "energy_ev_per_u": 30e6},
        "momentum_constants": {
            "speed_of_light_m_per_s": 299792458.0,
            "atomic_mass_unit_ev": 931494028.0,
        },
        "field": {"path": str(field), "sha256": file_hash(field), "filename": "map.#7"},
        "reference": {
            "path": str(ref),
            "sha256": file_hash(ref),
            "provenance": "manufactured exact straight drift; not real approval",
        },
        "steps": [10, 20, 40],
        "straight_lab_planes_z_m": [-0.15, 0.15],
        "state_units": ["m", "rad", "m", "rad"],
        "element": {
            "file_number": 7,
            "field_scale": 0.0,
            "length_m": 0.3,
            "aperture_shape": "circle",
            "aperture_m": 0.015,
        },
        "compatibility_limits_native": [1e-7] * 4,
        "refinement_limits_native": [1e-7] * 4,
    }
    path = tmp_path / "request.json"
    write_json(path, r)
    return path, r, expected


@pytest.mark.parametrize("failure", [None, "wrong_output", "unstable_steps", "status"])
def test_actual_reference_and_refinement_not_successful_exit_qualify(
    tmp_path, monkeypatch, failure
):
    path, _, expected = request(tmp_path)

    def fake(runtime, element, field, filename, beam, states, bg, *, workdir):
        workdir = Path(workdir)
        workdir.mkdir()
        result = expected.copy()
        flags = np.zeros(len(states), int)
        if failure == "wrong_output":
            result[0, 0] += 1e-3
        if failure == "unstable_steps":
            result[0, 0] += 1e-3 / element.steps
        if failure == "status":
            flags[0] = 1
        return TrackRun(result, flags, workdir, "synthetic zero exit")

    monkeypatch.setattr(sq, "run_element", fake)
    report = sq.check(path, tmp_path / "output")
    assert report["passed"] == (failure is None)
    assert report["grid_shape"] == [5, 5, 7]
    assert not report["automatic_generation_or_release_approval"]
    assert Path(report["receipt_path"]).is_file()


def test_changed_field_and_unmatched_frames_refuse_before_any_run(tmp_path, monkeypatch):
    path, r, _ = request(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("must refuse before executing")

    monkeypatch.setattr(sq, "run_element", forbidden)
    r["straight_lab_planes_z_m"] = [-0.16, 0.15]
    write_json(path, r)
    with pytest.raises(ValueError, match="planes"):
        sq.check(path, tmp_path / "output")
    r["straight_lab_planes_z_m"] = [-0.15, 0.15]
    r["field"]["sha256"] = "0" * 64
    write_json(path, r)
    with pytest.raises(ValueError, match="bytes changed"):
        sq.check(path, tmp_path / "output")
