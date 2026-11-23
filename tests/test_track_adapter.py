"""Tests for the TRACK adapter's files, staging and parsing, without running TRACK.

The ion, element and numbers are test inputs, not any facility's."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from element_surrogate import track_adapter as ta

BEAM = ta.TrackBeam(mass_number=40, charge=18, energy_ev_per_u=50e6)
CONSTANTS = ta.MomentumConstants(speed_of_light_m_per_s=2.99792458e8, atomic_mass_unit_ev=931.494e6)
STATES = np.array([[0.001, 0.002, -0.003, 0.0005], [0.0, 0.0, 0.0, 0.0], [0.02, -0.001, 0.01, 0.0]])
ELEMENT = ta.Sol3d(7, 1.0, 0.3, "circle", 0.02, 150)


def test_the_aperture_shape_sets_the_sign():
    circle = ta.Sol3d(7, 2.0 / 3.0, 0.3, "circle", 0.02, 150)
    assert circle.line() == "7\tsol3d\t0.6666666666666666\t30.0\t-2.0\t150"
    assert ta.Sol3d(7, 1.0, 0.3, "rectangle", 0.02, 150).line().split("\t")[4] == "2.0"
    with pytest.raises(ValueError, match="aperture_shape"):
        ta.Sol3d(7, 1.0, 0.3, "square", 0.02, 150)


def test_momentum_constants_round_trip():
    bg = CONSTANTS.beta_gamma(np.array([1.0, 2.5]), 18, 40)
    np.testing.assert_allclose(CONSTANTS.rigidity(bg, 18, 40), [1.0, 2.5], rtol=1e-15)
    other = ta.MomentumConstants(2.99792e8, 931.5e6)
    assert other.beta_gamma(1.0, 18, 40) / CONSTANTS.beta_gamma(1.0, 18, 40) == pytest.approx(
        (2.99792e8 / 2.99792458e8) * (931.494e6 / 931.5e6)
    )


def test_particle_file_is_written_in_track_units_and_parsed_back():
    bg = CONSTANTS.beta_gamma(np.full(3, 1.2), 18, 40)
    text = ta.read_dis(STATES, bg, BEAM)
    lines = text.splitlines()
    assert lines[:3] == ["50000.0\t1", "2", "18"]
    assert lines[3].split("\t")[:2] == [repr(0.1), repr(0.002)]  # cm and rad
    states, out_bg, flags = ta.parse_read_dis_out(text)
    np.testing.assert_allclose(states, STATES, rtol=1e-15, atol=1e-18)
    np.testing.assert_allclose(out_bg, bg, rtol=1e-15)
    assert flags.tolist() == [0, 0, 0]


def test_control_file_names_the_ion_and_reads_particles():
    text = ta.track_dat(BEAM, 3)
    for setting in (
        "Win = 50000000.0",
        "atp = 40",
        "qq = 18",
        "npat = 2",
        "iread_dis = 2",
        "iwrite_dis = 2",
    ):
        assert setting in text


def _fake_track(lost: set[int]):
    """A stand-in for the executable: it copies read_dis.dat to read_dis.out, flagging the
    particles in ``lost``."""

    def run(command, cwd, **kwargs):
        cwd = Path(cwd)
        assert (cwd / "FIELDS" / "map.#7").is_file() and (cwd / "fi_in.dat").is_file()
        assert (cwd / "sclinac.dat").read_text().endswith("0\tstop\n")
        lines = (cwd / "read_dis.dat").read_text().splitlines()
        out = lines[:3]
        for i, line in enumerate(lines[3:]):
            fields = line.split("\t")
            fields[6] = "1" if i in lost else "0"
            out.append("\t".join(fields))
        (cwd / "read_dis.out").write_text("\n".join(out) + "\n")
        return subprocess.CompletedProcess(command, 0, "done", "")

    return run


def test_a_run_is_staged_in_an_empty_directory_and_parsed(tmp_path, monkeypatch):
    executable = tmp_path / "track.exe"
    library = tmp_path / "runtime.dll"
    field = tmp_path / "field.table"
    for path in (executable, library, field):
        path.write_bytes(b"x")
    runtime = ta.TrackRuntime(executable, (library,))
    bg = CONSTANTS.beta_gamma(np.full(3, 1.2), 18, 40)
    monkeypatch.setattr(ta.subprocess, "run", _fake_track({2}))
    run = ta.run_element(
        runtime, ELEMENT, field, "map.#7", BEAM, STATES, bg, workdir=tmp_path / "run"
    )
    assert run.transported.tolist() == [True, True, False]
    np.testing.assert_allclose(run.states[:2], STATES[:2], rtol=1e-15, atol=1e-18)
    assert np.isnan(run.states[2]).all()
    assert run.summary() == {"particles": 3, "transported": 2, "flags": {0: 2, 1: 1}}
    with pytest.raises(FileExistsError):
        ta.run_element(
            runtime, ELEMENT, field, "map.#7", BEAM, STATES, bg, workdir=tmp_path / "run"
        )
    with pytest.raises(FileNotFoundError, match="missing TRACK files"):
        ta.run_element(
            ta.TrackRuntime(tmp_path / "absent.exe"), ELEMENT, field, "m", BEAM, STATES, bg
        )
    with pytest.raises(ValueError, match="file name"):
        ta.run_element(runtime, ELEMENT, field, "a/b", BEAM, STATES, bg)
