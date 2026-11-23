"""Running TRACK on one field-map element (simulator profile ``track``).

TRACK is an executable that reads its inputs from, and writes its outputs to, the current
directory. This adapter stages everything in an isolated, empty directory, so concurrent
runs and the user's files are never touched:

* the executable and the runtime libraries it needs (given by the caller: nothing here
  knows where a TRACK installation lives);
* the field file, under the name TRACK expects for the element's file number (given by the
  caller, since installations name tables differently);
* ``track.dat``, the design ion and the control flags (particles read from a file, one
  ion species, zero beam current, so particles move independently);
* ``sclinac.dat``, one element and ``stop``;
* ``read_dis.dat``, the particles; and an empty ``fi_in.dat``, which TRACK opens always.

TRACK writes every particle back, in input order, to ``read_dis.out`` with a status flag
(0: transported). Positions there are in centimetres and slopes in radians; this adapter
converts both ways and works in metres and radians.

A particle's momentum enters TRACK as ``beta * gamma``. Converting a rigidity to it needs a
speed of light and an atomic mass unit, and those must be the constants of the comparison
being made: the ones a data generator used, or the simulator's own when its design particle
is meant (knowledge entry ``track``). They are therefore arguments, never defaults.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

APERTURE_SHAPES = ("circle", "rectangle")


def _number(value: float) -> str:
    """A number as TRACK reads it: the shortest repr of a Python float (a NumPy float's repr
    names its type and is not a number)."""
    return repr(float(value))


@dataclass(frozen=True)
class TrackRuntime:
    """A TRACK installation: the executable and the libraries staged beside it."""

    executable: Path
    libraries: tuple[Path, ...] = ()
    timeout_s: float = 600.0

    def missing(self) -> list[str]:
        return [str(p) for p in (self.executable, *self.libraries) if not Path(p).is_file()]


@dataclass(frozen=True)
class TrackBeam:
    """The design ion TRACK is told about: mass number, charge state and kinetic energy
    per nucleon in eV."""

    mass_number: float
    charge: int
    energy_ev_per_u: float


@dataclass(frozen=True)
class MomentumConstants:
    """The constants that turn a rigidity into ``beta * gamma``."""

    speed_of_light_m_per_s: float
    atomic_mass_unit_ev: float

    def beta_gamma(self, rigidity_tm: np.ndarray, charge: int, mass_number: float) -> np.ndarray:
        """``beta gamma = c q B rho / (A m_u)``."""
        result: np.ndarray = (
            self.speed_of_light_m_per_s
            * charge
            * np.asarray(rigidity_tm, dtype=np.float64)
            / (mass_number * self.atomic_mass_unit_ev)
        )
        return result

    def rigidity(self, beta_gamma: np.ndarray, charge: int, mass_number: float) -> np.ndarray:
        result: np.ndarray = (
            np.asarray(beta_gamma, dtype=np.float64)
            * mass_number
            * self.atomic_mass_unit_ev
            / (self.speed_of_light_m_per_s * charge)
        )
        return result


@dataclass(frozen=True)
class Sol3d:
    """A TRACK ``sol3d`` element: a 3D field table scaled by ``field_scale``, ``length_m``
    long, integrated in ``steps`` steps, with an aperture. TRACK reads a negative aperture as
    a circular bore of that radius and a positive one as a rectangular half-aperture; give the
    shape here and the sign is written for you."""

    file_number: int
    field_scale: float
    length_m: float
    aperture_shape: str
    aperture_m: float
    steps: int

    def __post_init__(self) -> None:
        if self.aperture_shape not in APERTURE_SHAPES:
            raise ValueError(f"aperture_shape must be one of {APERTURE_SHAPES}")
        if not (self.length_m > 0 and self.aperture_m > 0 and self.steps > 0):
            raise ValueError("length, aperture and steps must be positive")
        if not math.isfinite(self.field_scale):
            raise ValueError("field_scale must be finite")

    def line(self) -> str:
        aperture_cm = 100.0 * self.aperture_m * (-1.0 if self.aperture_shape == "circle" else 1.0)
        fields = (_number(self.field_scale), _number(100.0 * self.length_m), _number(aperture_cm))
        return f"{self.file_number}\tsol3d\t" + "\t".join(fields) + f"\t{self.steps}"


def track_dat(beam: TrackBeam, particles: int, integration_steps: int = 200) -> str:
    """The control file: the design ion, particles read from ``read_dis.dat`` (``iread_dis =
    2``), zero beam current, and outputs that include ``read_dis.out`` (``iwrite_dis = 2``).
    The emittance lines are required by the format and unused when particles are read."""
    if particles < 1:
        raise ValueError("at least one particle")
    a = f"{beam.mass_number:g}"
    q = f"{beam.charge}"
    return f"""&TRAN

table_dir='FIELDS/'
rfq3d_dir='FIELDS/'
Win = {_number(beam.energy_ev_per_u)}
atp = {a}
freqb = 40250000.0
part = 0.995

nqtot = 1
qq = {q}
amass = {a}
npat = {particles - 1}
current = 0.0
Qavr = {q}
Qdesign = {q}
Adesign = {a}

epsnx = 1.43144531, alfax = 1.0, betax = 1.0
epsny = 1.43144531, alfay = 1.0, betay = 1.0
epsnz = 0.27962365259957617, alfaz = 0.007828926108780682, betaz = 6.611459502850964

db_b = 0.0001, df_4D = 180.0

phmax = 180.0, dwwmax = 1.0, Dwwacc = 100.0

x00 = 0.0
xp00 = 0.0
y00 = 0.0
yp00 = 0.0
ph00 = 0.0
Dww00 = 0.0

&END

&INDEX

NRZ = 1
igraph = 1
iflag_dis = 6
iflag2d = 0
iflag_qq = 1
iint = {integration_steps}
nstep_cav = {integration_steps}
iflag_cav = 1
iflag_ell = 0
isrf = 100
iflag_bc = 1
iread_dis = 2
iflag_log = 1
iwrite_dis = 2
nrz_min = 1
iwrite_step = 1
&END
"""


def read_dis(states: np.ndarray, beta_gamma: np.ndarray, beam: TrackBeam) -> str:
    """The particle file: reference energy in keV/u, one species, ``N - 1``, the charge, then
    one row per particle ``x_cm xp_rad y_cm yp_rad phase_deg beta_gamma flag``."""
    s = np.asarray(states, dtype=np.float64)
    bg = np.broadcast_to(np.asarray(beta_gamma, dtype=np.float64), (s.shape[0],))
    lines = [f"{_number(beam.energy_ev_per_u * 1e-3)}\t1\n{s.shape[0] - 1}\n{beam.charge}\n"]
    for (x, xp, y, yp), g in zip(s, bg, strict=True):
        row = (_number(100.0 * x), _number(xp), _number(100.0 * y), _number(yp), "0.0", _number(g))
        lines.append("\t".join(row) + "\t0\t\n")
    return "".join(lines)


def parse_read_dis_out(text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """States ``(x, x', y, y')`` in metres and radians, ``beta * gamma`` and status flags of
    every particle, in input order."""
    rows = []
    for line in text.splitlines()[3:]:
        fields = line.split()
        if len(fields) >= 7:
            rows.append([float(v.replace("D", "E")) for v in fields[:7]])
    data = np.asarray(rows, dtype=np.float64).reshape(-1, 7)
    states = np.stack([data[:, 0] / 100.0, data[:, 1], data[:, 2] / 100.0, data[:, 3]], axis=1)
    return states, data[:, 5], data[:, 6].astype(np.int64)


@dataclass
class TrackRun:
    """What one run returned: exit states (NaN where TRACK did not transport the particle),
    TRACK's status flag per particle, and the run directory if it was kept."""

    states: np.ndarray
    flags: np.ndarray
    workdir: Path | None
    stdout: str
    raw_states: np.ndarray | None = None
    loss_positions_m: np.ndarray | None = None

    @property
    def transported(self) -> np.ndarray:
        result: np.ndarray = self.flags == 0
        return result

    def summary(self) -> dict[str, Any]:
        values, counts = np.unique(self.flags, return_counts=True)
        return {
            "particles": int(self.flags.size),
            "transported": int(self.transported.sum()),
            "flags": {int(v): int(c) for v, c in zip(values, counts, strict=True)},
        }


def run_element(
    runtime: TrackRuntime,
    element: Sol3d,
    field_file: Path,
    field_file_name: str,
    beam: TrackBeam,
    states: np.ndarray,
    beta_gamma: np.ndarray,
    *,
    workdir: Path | None = None,
    keep: bool = False,
) -> TrackRun:
    """Track ``states`` (``(N, 4)``, metres and radians, at the element's entrance) with
    momenta ``beta_gamma`` through one element. ``workdir``, if given, must be empty."""
    s = np.asarray(states, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != 4 or not np.isfinite(s).all() or s.shape[0] < 1:
        raise ValueError("states must be finite, shape (N, 4), N >= 1")
    bg = np.broadcast_to(np.asarray(beta_gamma, dtype=np.float64), (s.shape[0],)).copy()
    if not (np.isfinite(bg).all() and np.all(bg > 0)):
        raise ValueError("beta_gamma must be finite and positive")
    missing = runtime.missing() + ([] if Path(field_file).is_file() else [str(field_file)])
    if missing:
        raise FileNotFoundError(f"missing TRACK files: {missing}")
    if "/" in field_file_name or "\\" in field_file_name:
        raise ValueError("field_file_name is a file name, not a path")

    temporary = None
    if workdir is None:
        if keep:
            run_dir = Path(tempfile.mkdtemp(prefix="track_run_"))
        else:
            temporary = tempfile.TemporaryDirectory(prefix="track_run_")
            run_dir = Path(temporary.name)
    else:
        run_dir = Path(workdir)
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"the TRACK work directory must be empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
    try:
        (run_dir / "FIELDS").mkdir()
        executable = run_dir / Path(runtime.executable).name
        shutil.copy2(runtime.executable, executable)
        for library in runtime.libraries:
            shutil.copy2(library, run_dir / Path(library).name)
        shutil.copy2(field_file, run_dir / "FIELDS" / field_file_name)
        (run_dir / "track.dat").write_text(
            track_dat(beam, s.shape[0], element.steps), encoding="ascii"
        )
        (run_dir / "sclinac.dat").write_text(f"{element.line()}\n0\tstop\n", encoding="ascii")
        (run_dir / "read_dis.dat").write_text(read_dis(s, bg, beam), encoding="ascii")
        (run_dir / "fi_in.dat").write_text("", encoding="ascii")
        process = subprocess.run(
            [str(executable)],
            cwd=run_dir,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=runtime.timeout_s,
            check=False,
        )
        stdout = process.stdout + process.stderr
        if process.returncode != 0:
            raise RuntimeError(f"TRACK exited with code {process.returncode}: {stdout[-2000:]}")
        output = run_dir / "read_dis.out"
        if not output.is_file():
            raise FileNotFoundError(f"TRACK wrote no read_dis.out in {run_dir}")
        out_states, out_bg, flags = parse_read_dis_out(output.read_text(errors="replace"))
        if flags.size != s.shape[0]:
            raise RuntimeError(f"TRACK returned {flags.size} particles for {s.shape[0]}")
        # Without acceleration each particle keeps its momentum: a check of row order.
        if not np.allclose(out_bg, bg, rtol=1e-12, atol=0.0):
            raise RuntimeError("read_dis.out does not list the particles in input order")
        if not np.isfinite(out_states[flags == 0]).all():
            raise RuntimeError("TRACK returned nonfinite transported coordinates")
        raw_states = out_states.copy()
        loss_path = run_dir / "lost.out"
        loss_positions = (
            parse_loss_positions(loss_path.read_text(errors="replace"))
            if loss_path.exists()
            else None
        )
        out_states[flags != 0] = np.nan
        return TrackRun(
            out_states,
            flags,
            run_dir if (keep or workdir is not None) else None,
            stdout,
            raw_states,
            loss_positions,
        )
    finally:
        if temporary is not None:
            temporary.cleanup()


def parse_loss_positions(text: str) -> np.ndarray:
    """TRACK lost.out column 8, cm from lattice entrance, after one header line.

    No particle-ID association or full loss phase space is inferred. Malformed
    nonempty rows fail explicitly instead of being misreported as no losses.
    """
    positions = []
    for line in text.splitlines()[1:]:
        if not line.strip():
            continue
        columns = line.split()
        if len(columns) < 8:
            raise ValueError("malformed TRACK lost.out row")
        value = float(columns[7].replace("D", "E")) / 100
        if not math.isfinite(value):
            raise ValueError("nonfinite TRACK loss position")
        positions.append(value)
    return np.asarray(positions, dtype=float)
