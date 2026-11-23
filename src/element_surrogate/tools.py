"""The tool registry: the single definition of every tool.

Each tool is a plain, deterministic function that works without an agent and returns a
JSON-serializable summary whose numeric keys carry their units; tools never return raw
field or particle arrays. Registering a function with ``@tool`` makes it a recordable,
replayable call and an MCP tool: the server takes the tool's schema from the function's
signature and its description from the docstring.

Field tools share their arguments: ``path`` and ``format`` (detected when omitted) name
the file, ``slot`` picks the field when the file holds several, and ``shape`` (circle,
square or rectangle), ``radius_m``, ``half_height_m`` (rectangles), ``z_min_m``, ``z_max_m``
and ``bend`` give the region: by default the largest circle about the z axis that the grid
covers; with ``bend`` (``radius_m``, ``angle_rad``, ``arm_m``, optionally ``bends_toward``)
a region about that nominal path, limited along it by path length from its midpoint.
Checks that can decide something on their own return ``findings``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np

from element_surrogate import knowledge
from element_surrogate.adaptive import adaptive_convergence
from element_surrogate.analysis import field_frame, harmonics, mirror_symmetry
from element_surrogate.benchmark import protocol as benchmark_protocol
from element_surrogate.benchmark import report as benchmark_report
from element_surrogate.benchmark import study as benchmark_study
from element_surrogate.description import compare_description
from element_surrogate.diagnostic_blocks import DEFAULT_WORKING_BYTES
from element_surrogate.fields import FieldMap, load_field_file
from element_surrogate.gradients import generalized_gradients
from element_surrogate.lorentz import step_convergence, straight_frame
from element_surrogate.readiness import (
    compare_exports,
    field_ends,
    harmonic_nodes,
    maxwell_report,
    node_defects,
    storage_order,
)
from element_surrogate.readings import reconstruct
from element_surrogate.reference import fit_reference
from element_surrogate.region import Region, nominal_bend, region_for
from element_surrogate.runs import Tool, sha256_file

_CHUNK = 1 << 20
_TEXT_PROBE_BYTES = 1 << 16

REGISTRY: dict[str, Tool] = {}


def tool(
    version: str, input_file_args: tuple[str, ...] = ("path",), read_only: bool = True
) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., dict[str, Any]]]:
    """Register a function as a tool. Bump ``version`` when its arguments or results change."""

    def register(func: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        if func.__name__ in REGISTRY:
            raise ValueError(f"tool {func.__name__} is registered twice")

        @wraps(func)
        def guarded(*args, **kwargs):
            from element_surrogate.jobs import managed_context_required

            managed_context_required(func.__name__)
            return func(*args, **kwargs)

        REGISTRY[func.__name__] = Tool(func.__name__, guarded, version, input_file_args, read_only)
        return guarded

    return register


def _field(path: str, format: str | None, slot: int | None) -> FieldMap:
    return load_field_file(path, format).slot(slot)


def _context(field: FieldMap, region: Region | None) -> dict[str, Any]:
    return {
        "source": field.source.as_dict() if field.source is not None else {"label": field.label},
        "region": region.as_dict() if region is not None else None,
    }


@tool("1")
def inspect_file(path: str) -> dict[str, Any]:
    """Summarize a file without interpreting it: size in bytes, SHA-256, whether it looks like
    text (no NUL byte in its first 64 KiB), and its line count if it is text."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"not a file: {path}")
    digest = hashlib.sha256()
    size = 0
    newlines = 0
    with file_path.open("rb") as handle:
        head = handle.read(_TEXT_PROBE_BYTES)
        is_text = b"\x00" not in head
        block = head
        while block:
            digest.update(block)
            size += len(block)
            newlines += block.count(b"\n")
            block = handle.read(_CHUNK)
    line_count = None
    if is_text:
        ends_with_newline = size == 0 or _last_byte(file_path) == b"\n"
        line_count = newlines if ends_with_newline else newlines + 1
    return {
        "path": str(file_path.resolve()),
        "size_bytes": size,
        "sha256": digest.hexdigest(),
        "is_text": is_text,
        "line_count": line_count,
    }


def _last_byte(path: Path) -> bytes:
    with path.open("rb") as handle:
        handle.seek(-1, 2)
        return handle.read(1)


@tool("2")
def read_field_file(path: str, format: str | None = None) -> dict[str, Any]:
    """Read a magnetic field file and summarize it in SI units: format, header, the units it
    was converted from, the storage order the reader assumed, integrity findings
    (interleaved zero rows, a duplicated final row, all-zero slots) and each slot's grid and
    field range. format is detected when omitted."""
    return load_field_file(path, format).summary()


@tool("4")
def check_storage_order(
    path: str,
    format: str | None = None,
    slot: int | None = None,
    shape: str = "circle",
    radius_m: float | None = None,
    z_min_m: float | None = None,
    z_max_m: float | None = None,
    half_height_m: float | None = None,
    bend: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Test the reader's storage assumption with Maxwell's equations: re-read the numbers under
    every nesting of the grid axes, every assignment of the columns to (Bx, By, Bz) and every
    relative sign, and rank them by relative vacuum-Maxwell residual. Run this first on a new
    file: every other check relies on the storage order being right. A reading clearly better
    than the one read is a finding."""
    field = _field(path, format, slot)
    region = region_for(field, shape, radius_m, z_min_m, z_max_m, half_height_m, bend)
    return {**_context(field, region), **storage_order(field, region)}


@tool("5")
def check_maxwell(
    path: str,
    format: str | None = None,
    slot: int | None = None,
    shells: int = 10,
    shape: str = "circle",
    radius_m: float | None = None,
    z_min_m: float | None = None,
    z_max_m: float | None = None,
    half_height_m: float | None = None,
    bend: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Divergence and curl by finite volumes on the map's own grid, in shells of the region
    (T/m and relative to the RMS field derivative). Each shell also compares the grid
    coarsened by two: pointwise order about 2 is truncation of a smooth field, about 0.5
    independent node errors in the data, about 0 an error that does not depend on spacing.
    It also compares fourth- and second-order stencils at the same nodes: well below 1 is
    truncation, about 1.34 independent node errors."""
    field = _field(path, format, slot)
    region = region_for(field, shape, radius_m, z_min_m, z_max_m, half_height_m, bend)
    report = maxwell_report(field, region, shells)
    return {**_context(field, region), **{k: v for k, v in report.items() if k != "region"}}


@tool("5")
def scan_node_defects(
    path: str, format: str | None = None, slot: int | None = None
) -> dict[str, Any]:
    """Find nodes no export of a real field produces: non-finite values, zero vectors next to a
    non-zero node, and grid planes identical to their neighbour. Each is a finding."""
    field = _field(path, format, slot)
    return {**_context(field, None), **node_defects(field)}


@tool("5")
def check_harmonic_nodes(
    path: str,
    format: str | None = None,
    slot: int | None = None,
    top: int = 100,
    examples: int = 10,
    fence_iqr: float = 3.0,
    shape: str = "circle",
    radius_m: float | None = None,
    z_min_m: float | None = None,
    z_max_m: float | None = None,
    half_height_m: float | None = None,
    bend: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure a node-consistency residual and find nodes far above its median. Each node's
    implied correction (T) measures disagreement with its discrete vacuum neighbourhood;
    it is not calibrated absolute field uncertainty. Harmonic scaling/polarity errors can
    pass this test. The top nodes are grouped into touching clusters with their ratio to the
    floor: a cluster far above it can indicate defects or a material surface. Nodes
    with node-level order that stand out from the nodes around them beyond the far-out fence
    of the map's own distribution (fence_iqr interquartile ranges; 3 is Tukey's) are
    findings."""
    field = _field(path, format, slot)
    region = region_for(field, shape, radius_m, z_min_m, z_max_m, half_height_m, bend)
    report = harmonic_nodes(field, region, top, examples, fence_iqr)
    return {**_context(field, region), **{k: v for k, v in report.items() if k != "region"}}


@tool("3")
def analyze_harmonics(
    path: str,
    fit_radius_m: float,
    reference_radius_m: float,
    format: str | None = None,
    slot: int | None = None,
    order: int = 6,
    fit_shape: str = "circle",
    z_min_m: float | None = None,
    z_max_m: float | None = None,
    fit_half_height_m: float | None = None,
) -> dict[str, Any]:
    """Integrated normal and skew multipoles (T*m at the reference radius, and relative to the
    main harmonic), how much they move when two more orders are fitted, and the main
    harmonic's profile along z: integral, peak, centroid, map-profile length and edge
    fractions. Choose fit_radius_m (circle radius, or square or rectangle half-width) inside
    the region that passed the readiness checks; nothing is assumed. About the z axis only."""
    field = _field(path, format, slot)
    region = Region(fit_shape, fit_radius_m, z_min_m, z_max_m, fit_half_height_m)
    report = harmonics(field, region, reference_radius_m, order)
    return {**_context(field, region), **{k: v for k, v in report.items() if k != "fit_region"}}


@tool("4")
def check_mirror_symmetry(
    path: str,
    format: str | None = None,
    slot: int | None = None,
    shape: str = "circle",
    radius_m: float | None = None,
    z_min_m: float | None = None,
    z_max_m: float | None = None,
    half_height_m: float | None = None,
    bend: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """How close each field component is to even or odd under the x, y and z mirrors, over the
    region. A missing parity can be a real property of the magnet: report it, do not impose a
    symmetry."""
    field = _field(path, format, slot)
    region = region_for(field, shape, radius_m, z_min_m, z_max_m, half_height_m, bend)
    report = mirror_symmetry(field, region)
    return {**_context(field, region), **{k: v for k, v in report.items() if k != "region"}}


@tool("1")
def analyze_field_frame(
    path: str,
    format: str | None = None,
    slot: int | None = None,
    component: str | None = None,
    core_fraction: float = 0.5,
) -> dict[str, Any]:
    """How the strong-field band lies in the file's grid, decided from the field values: the
    band centre in the plane nearest y = 0, fitted as straight, as a circle (a straight box
    with the magnet curving in it) and as a secant (a grid in sector coordinates), over the
    central core_fraction of the body. With a curved band, a nominal bend (radius, angle from
    the field integral, arms to the grid ends) to use for regions; not the reference orbit.
    For tracking, shorten the arms so the tilted end planes lie inside the field."""
    field = _field(path, format, slot)
    return {**_context(field, None), **field_frame(field, component, core_fraction)}


@tool("2")
def check_field_ends(
    path: str,
    format: str | None = None,
    slot: int | None = None,
    tolerance: float | None = None,
    shape: str = "circle",
    radius_m: float | None = None,
    z_min_m: float | None = None,
    z_max_m: float | None = None,
    half_height_m: float | None = None,
    bend: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Whether the map holds its whole field along z: the field at the end planes relative to
    the peak, and the integral missed beyond each end (the larger of exponential and power-law
    extrapolation). With tolerance (the accuracy needed of integrals), ends beyond it are
    findings; without it nothing is flagged. The region defaults to whole planes."""
    field = _field(path, format, slot)
    region = None
    if radius_m is not None or bend is not None or half_height_m is not None:
        region = region_for(field, shape, radius_m, z_min_m, z_max_m, half_height_m, bend)
    report = field_ends(field, region, tolerance)
    return {**_context(field, region), **{k: v for k, v in report.items() if k != "region"}}


@tool("1")
def compare_with_description(
    path: str,
    description: dict[str, Any],
    format: str | None = None,
    slot: int | None = None,
    fit_radius_m: float | None = None,
    order: int = 6,
) -> dict[str, Any]:
    """Compare the field with the user's description. Keys, all optional: kind, main_harmonic,
    integrated_strength (SI: T*m dipole, T quadrupole, T/m sextupole; signed, positive focuses
    positive particles in x), or current_a with strength_per_ampere, bend (radius_m,
    angle_rad, bends_toward), aperture, tolerance (relative). Mismatches beyond tolerance are
    findings and name any unit mislabel or reversed polarity that would explain them; a
    missing aperture is a finding; what was not described is listed."""
    field = _field(path, format, slot)
    region = Region("circle", fit_radius_m) if fit_radius_m is not None else None
    return {**_context(field, region), **compare_description(field, description, region, order)}


@tool("1")
def analyze_generalized_gradients(
    path: str,
    radius_m: float,
    format: str | None = None,
    slot: int | None = None,
    m_max: int = 12,
    l_max: int = 6,
    n_phi: int | None = None,
    interpolation: str = "linear",
) -> dict[str, Any]:
    """Generalized gradients (Venturini-Dragt) of a straight-axis map from B_rho on the
    cylinder of radius_m (inside the region that passed the readiness checks): integrated
    normal and skew b_n (SI) and at the radius, the exponential tails measured at each end,
    how much the integrals move with the other sampling (linear or cubic), and how far the
    map's interior nodes are from the vacuum field its cylinder surface implies."""
    field = _field(path, format, slot)
    report = generalized_gradients(field, radius_m, m_max, l_max, n_phi, interpolation)
    return {**_context(field, None), **report}


@tool("1", input_file_args=("path", "other_path"))
def compare_field_exports(
    path: str,
    other_path: str,
    format: str | None = None,
    other_format: str | None = None,
    slot: int | None = None,
    other_slot: int | None = None,
    tolerance: float | None = None,
    shells: int = 4,
    shape: str = "circle",
    radius_m: float | None = None,
    z_min_m: float | None = None,
    z_max_m: float | None = None,
    half_height_m: float | None = None,
    bend: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare two exports of one field solve at the first map's nodes in the region: the
    best scale between them and its sign (relative polarity), the difference with polarity
    matched and with the best scale, the interpolation error when the grids differ, each map's
    node-level floor, and both maps' Maxwell residuals by shell. Residuals both maps share
    come from the solve; residuals in one only, from its export. Opposite polarity is a
    finding; with tolerance (relative), so are a scale or a difference beyond it."""
    field = _field(path, format, slot)
    other = _field(other_path, other_format, other_slot)
    region = region_for(field, shape, radius_m, z_min_m, z_max_m, half_height_m, bend)
    report = compare_exports(field, other, region, tolerance, shells)
    return {
        "source": field.source.as_dict() if field.source is not None else None,
        "other_source": other.source.as_dict() if other.source is not None else None,
        **report,
    }


@tool("3", read_only=False)
def fit_reference_orbit(
    path: str,
    rigidity_tm: float,
    bend: dict[str, Any],
    scale_bracket: list[float],
    step_m: float | None,
    position_tolerance_m: float,
    slope_tolerance_rad: float,
    format: str | None = None,
    slot: int | None = None,
    reconstruction: str = "total-degree-two",
    output_path: str | None = None,
    integration_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit the field scale, within scale_bracket, at which the design particle (rigidity_tm,
    signed) leaves the bend's nominal exit plane with zero horizontal slope, as a simulator
    that fits its bend for a symmetric exit does; check that the other exit coordinates close
    within the tolerances (refused otherwise). Returns the scale, the orbit's length, its exit
    state and its largest departure from the nominal path. The orbit is not refitted for
    particles of other rigidity: they represent a strength error."""
    field = _field(path, format, slot)
    nominal = nominal_bend(bend)
    if len(scale_bracket) != 2:
        raise ValueError("scale_bracket is [low, high]")
    orbit = fit_reference(
        reconstruct(field, reconstruction),
        rigidity_tm,
        nominal.entry_frame(),
        nominal.exit_frame(),
        scale_bracket=(float(scale_bracket[0]), float(scale_bracket[1])),
        step_m=step_m,
        integration_options=integration_options,
        position_tolerance_m=position_tolerance_m,
        slope_tolerance_rad=slope_tolerance_rad,
    )
    saved = None
    if output_path is not None:
        orbit.save(output_path)
        saved = {"path": str(Path(output_path).resolve()), "sha256": sha256_file(output_path)}
    return {
        **_context(field, None),
        "bend": nominal.as_dict(),
        "reconstruction": reconstruction,
        "saved_reference": saved,
        "reference_identity": orbit.metadata,
        **orbit.summary(nominal),
    }


@tool("2")
def check_tracking_convergence(
    path: str,
    rigidity_tm: float,
    steps_m: list[float],
    half_ranges: dict[str, float],
    particles: int = 50,
    seed: int = 0,
    field_scale: float = 1.0,
    bend: dict[str, Any] | None = None,
    z_entry_m: float | None = None,
    z_exit_m: float | None = None,
    format: str | None = None,
    slot: int | None = None,
    reconstruction: str = "total-degree-two",
) -> dict[str, Any]:
    """Track a seeded sample of particles (uniform within half_ranges: x_m, xp_rad, y_m,
    yp_rad) with each step in steps_m (largest first) through field_scale times the map, and
    report how much the exit states change between steps and the observed order (about 4 for
    RK4 in a smooth field; lower where a trilinear reading's cell faces are crossed). Planes:
    the bend's nominal entry and exit, or z_entry_m and z_exit_m (default: the grid's ends)."""
    field = _field(path, format, slot)
    keys = ("x_m", "xp_rad", "y_m", "yp_rad")
    if set(half_ranges) != set(keys):
        raise ValueError(f"half_ranges needs exactly {keys}")
    if particles < 1:
        raise ValueError("particles must be at least 1")
    rng = np.random.default_rng(seed)
    states = rng.uniform(-1.0, 1.0, (particles, 4)) * np.array([half_ranges[k] for k in keys])
    if bend is not None:
        nominal = nominal_bend(bend)
        entry, exit_ = nominal.entry_frame(), nominal.exit_frame()
    else:
        z0 = float(field.z_m[0]) if z_entry_m is None else z_entry_m
        z1 = float(field.z_m[-1]) if z_exit_m is None else z_exit_m
        entry, exit_ = straight_frame(z0), straight_frame(z1)
    report = step_convergence(
        reconstruct(field, reconstruction),
        states,
        rigidity_tm,
        entry,
        exit_,
        list(steps_m),
        field_scale=field_scale,
    )
    return {
        **_context(field, None),
        "reconstruction": reconstruction,
        "sample": {"particles": particles, "seed": seed, "half_ranges": half_ranges},
        **report,
    }


@tool("2", input_file_args=())
def search_knowledge(
    query: str,
    kind: str | None = None,
    include_proposed: bool = False,
    project_dir: str | None = None,
) -> dict[str, Any]:
    """Search the shipped knowledge base: methods (definitions, choices and pitfalls of a
    computation), simulator and magnet-class profiles, lessons and references. Returns active
    entries ranked by the query's words; include_proposed also returns entries awaiting the
    user's approval, which must not be relied on as approved knowledge."""
    from element_surrogate.project import entries as project_entries

    entries = project_entries(project_dir)
    return {"query": query, "results": knowledge.search(entries, query, kind, include_proposed)}


@tool("2", input_file_args=())
def read_knowledge(entry_id: str, project_dir: str | None = None) -> dict[str, Any]:
    """Read one knowledge entry in full: its status, body, the tools and tests that implement
    it, and its evidence, with any source whose support has not been verified listed
    separately."""
    from element_surrogate.project import entries as project_entries

    return knowledge.find(project_entries(project_dir), entry_id).as_dict()


@tool("1")
def check_adaptive_tracking_convergence(
    path: str,
    rigidity_tm: float,
    tolerances_m_direction: list[list[float]],
    half_ranges: dict[str, float],
    z_entry_m: float,
    z_exit_m: float,
    particles: int = 50,
    seed: int = 0,
    field_scale: float = 1.0,
    max_step_m: float | None = None,
    format: str | None = None,
    slot: int | None = None,
    reconstruction: str = "total-degree-two",
    bend: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Refine adaptive position/direction tolerances on a seeded state cohort.

    Uses total-degree-two by default; reports all statuses, common-cohort changes,
    path changes and actual step statistics. This tests one declared field/rigidity/
    scale/frame configuration, not all magnets or the full operating envelope.
    """
    field = _field(path, format, slot)
    keys = ("x_m", "xp_rad", "y_m", "yp_rad")
    if (
        set(half_ranges) != set(keys)
        or particles < 1
        or any(not np.isfinite(v) or v < 0 for v in half_ranges.values())
    ):
        raise ValueError("nonnegative finite half ranges and positive particle count required")
    states = np.random.default_rng(seed).uniform(-1, 1, (particles, 4)) * [
        half_ranges[k] for k in keys
    ]
    if bend is None:
        entry, exit_ = straight_frame(z_entry_m), straight_frame(z_exit_m)
    else:
        nominal = nominal_bend(bend)
        entry, exit_ = nominal.entry_frame(), nominal.exit_frame()
    return {
        **_context(field, None),
        "reconstruction": reconstruction,
        "sample": {"particles": particles, "seed": seed, "half_ranges": half_ranges},
        **adaptive_convergence(
            reconstruct(field, reconstruction),
            states,
            rigidity_tm,
            entry,
            exit_,
            tolerances_m_direction,
            field_scale=field_scale,
            max_step_m=max_step_m,
        ),
    }


@tool("1", input_file_args=("path",))
def validate_benchmark_protocol(path: str) -> dict[str, Any]:
    """Check an explicit 4D/5D/6D scientific study contract and return its identity.
    This validates declarations; physical field qualification remains separate."""
    p = benchmark_protocol.load_protocol(path)
    return {
        "protocol_sha256": benchmark_protocol.protocol_identity(p),
        "coordinate_names": [c["name"] for c in p["coordinates"]],
        "coordinate_units": [c["unit"] for c in p["coordinates"]],
        "configurations": len(p["candidates"]),
        "planned_trials": sum(len(c["seeds"]) for c in p["candidates"]),
    }


@tool("1", input_file_args=("path",), read_only=False)
def create_benchmark_study(path: str, output_dir: str, approval_paths: list[str]) -> dict[str, Any]:
    """Freeze a study after protocol-bound user approvals for target, range and budget.
    Approval files are user attestations supplied by the client, not authenticated
    human identities. This tool never creates approvals or reads the test cohort."""
    return benchmark_study.create_study(path, output_dir, approval_paths)


@tool("2", input_file_args=("path",), read_only=False)
def start_benchmark_study(path: str) -> dict[str, Any]:
    """Start a bounded background study using the declared trusted local adapter.
    Workers have parent-enforced deadlines. This is not a sandbox for untrusted
    code. Repeated starts do not launch duplicate trials; inspect with status."""
    return benchmark_study.start_study(path)


@tool("1", input_file_args=("path",))
def benchmark_study_status(path: str) -> dict[str, Any]:
    """Read study/trial status and artifact paths without returning particle arrays."""
    return benchmark_study.study_status(path)


@tool("1", input_file_args=("path",), read_only=False)
def lock_benchmark_selection(path: str) -> dict[str, Any]:
    """Recheck complete validation results, group seeds and lock the selected checkpoint.
    Hyperparameter configurations are compared separately. No test rows are read."""
    return benchmark_study.lock_selection(path)


@tool("1", input_file_args=("path",), read_only=False)
def evaluate_benchmark_test(path: str, approval_paths: list[str]) -> dict[str, Any]:
    """Score only the locked model on test after protocol-bound release approval.
    Existing results are checked without refitting or changing the selected model."""
    result = benchmark_study.evaluate_test(path, approval_paths)
    metrics = result["metrics"]
    return {
        "test_result_path": result["test_result_path"],
        "rows": metrics["rows"],
        "coordinate_names": metrics["coordinate_names"],
        "coordinate_units": metrics["coordinate_units"],
        "rmse": metrics["rmse"],
        "p99_absolute": metrics["p99_absolute"],
        "max_absolute": metrics["max_absolute"],
    }


@tool("1", input_file_args=("path",), read_only=False)
def build_benchmark_tradeoff_report(
    path: str, output_dir: str, check: bool = False
) -> dict[str, Any]:
    """Report validation tradeoffs before selection, including infeasible ceilings.
    Creates no selection lock and reads no test cohort or test-result file.
    Shows all complete configurations and physical validation errors/tails;
    it does not certify speed-regime coverage or deployment compliance."""
    return benchmark_report.build_tradeoff_report(path, output_dir, check=check)


@tool("1", input_file_args=("path",), read_only=False)
def build_benchmark_report(path: str, output_dir: str, check: bool = False) -> dict[str, Any]:
    """Build or verify a selected-model report, accuracy/cost plots and result-card draft.
    Numeric identities and units are checked. Full Genesis card schema validation
    and external publication are separate; this tool performs no upload."""
    return benchmark_report.build_report(path, output_dir, check=check)


@tool("1", input_file_args=("path",), read_only=False)
def prepare_benchmark_metric_crosscheck(
    path: str, output_dir: str, maximum_rows: int
) -> dict[str, Any]:
    """Export an ID-aligned, explicitly bounded metric-check cohort and receipt.
    The row cap must come from the chosen evaluator. No external evaluator runs;
    its global MAE/RMSE check cannot substitute for setting/tail/physics controls."""
    return benchmark_report.prepare_metric_crosscheck(path, output_dir, maximum_rows)


@tool("1", input_file_args=("path",), read_only=False)
def score_benchmark_documentation(path: str, output_path: str) -> dict[str, Any]:
    """Apply the pinned upstream documentation rubric to a filled evidence record.
    This internal self-assessment does not confer institutional or MLCommons approval."""
    return benchmark_report.score_documentation(path, output_path)


@tool("2", read_only=False)
def prepare_mapped_field_store(
    path: str, output_dir: str, block_nodes: int, format: str | None = None, slot: int | None = None
) -> dict[str, Any]:
    """Create an exact float64 memory-mapped field cache without changing the source.
    CST HDF5 conversion streams bounded blocks directly. Other formats retain
    their original reader's ingestion memory cost; convert them externally if
    that cost exceeds resources. Subsequent mapped reads avoid a whole-field
    copy. This preserves resolution and does not refine a solver mesh."""
    from element_surrogate.field_store import convert_cst_hdf5, save_mapped_field
    from element_surrogate.fields import detect_format

    chosen = format or detect_format(path)
    if chosen == "cst_hdf5":
        if slot not in (None, 0):
            raise ValueError("CST HDF5 has one field slot")
        return convert_cst_hdf5(path, output_dir, block_nodes=block_nodes)

    return save_mapped_field(_field(path, chosen, slot), output_dir, block_nodes=block_nodes)


@tool("1", read_only=False)
def trace_reference_orbit(
    path: str,
    output_path: str,
    rigidity_tm: float,
    field_scale: float,
    entry_frame: dict[str, Any],
    exit_frame: dict[str, Any],
    integration: dict[str, Any],
    format: str | None = None,
    slot: int | None = None,
    reconstruction: str = "total-degree-two",
) -> dict[str, Any]:
    """Trace and save a fixed-strength reference on explicit native planes.
    No symmetric-exit fit or rigidity-dependent refit is performed. Integration
    and plane tolerances are declared; reference convergence must be checked for
    the intended map before using the saved orbit as a label convention."""
    from element_surrogate.reference import trace_reference
    from element_surrogate.workflow import frame_from_dict

    field = reconstruct(_field(path, format, slot), reconstruction)
    options = dict(integration)
    method = options.pop("method")
    max_path = options.pop("max_path_m", None)
    plane_tolerance = options.pop("plane_tolerance_m", None)
    if method == "fixed-rk4":
        step = options.pop("step_m")
        if options:
            raise ValueError("unsupported fixed reference integration options")
        adaptive = None
    elif method == "adaptive-rk4":
        step, adaptive = None, options
    else:
        raise ValueError("explicit fixed-rk4 or adaptive-rk4 method required")
    target = Path(output_path)
    if target.exists():
        raise FileExistsError("reference output already exists; never overwrite a saved convention")
    orbit = trace_reference(
        field,
        rigidity_tm,
        frame_from_dict(entry_frame),
        frame_from_dict(exit_frame),
        step_m=step,
        field_scale=field_scale,
        max_path_m=max_path,
        integration_options=adaptive,
        plane_tolerance_m=plane_tolerance,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    orbit.save(target)
    return {
        "saved_reference": {"path": str(target.resolve()), "sha256": sha256_file(target)},
        "reference_identity": orbit.metadata,
        **orbit.summary(),
    }


@tool("1", input_file_args=("path", "request_path"), read_only=False)
def qualify_element_transport(path: str, request_path: str, output_path: str) -> dict[str, Any]:
    """Qualify a fixed native map on explicit ranges/settings and refinement probes.
    Require three resolutions and supplied component/path limits. Every field
    read stays in the declared stencil support. Report finite-cohort evidence,
    not a material aperture or a certified absolute field error."""
    from element_surrogate.workflow import qualify_from_files

    return qualify_from_files(path, request_path, output_path)


@tool("2", input_file_args=("path", "qualification_path", "plan_path"))
def validate_training_data_plan(
    path: str, qualification_path: str, plan_path: str
) -> dict[str, Any]:
    """Check bound field-readiness review, transport qualification and generation budgets."""
    from element_surrogate.workflow import validate_generation_files

    return validate_generation_files(path, qualification_path, plan_path)


@tool("3", input_file_args=("path", "qualification_path", "plan_path"), read_only=False)
def start_training_data_generation(
    path: str, qualification_path: str, plan_path: str, output_dir: str, approval_paths: list[str]
) -> dict[str, Any]:
    """Start a background generation job after protocol-bound user attestations.
    A supervisor enforces the wall budget. Immutable batches retain all proposal
    IDs/losses; failed/interrupted jobs need inspection before a new launch."""
    from element_surrogate.generation_job import start_generation

    return start_generation(path, qualification_path, plan_path, output_dir, approval_paths)


@tool("1", input_file_args=())
def training_data_status(output_dir: str) -> dict[str, Any]:
    """Read job state and verify committed shard hashes; return no particle arrays."""
    from element_surrogate.generation_job import generation_status

    return generation_status(output_dir)


@tool("1", input_file_args=(), read_only=False)
def prepare_training_benchmark_cohorts(
    generation_dir: str, output_dir: str, maximum_rows_per_split: int
) -> dict[str, Any]:
    """Prepare fixed bounded cohort descriptors without opening test particle arrays.
    The existing locked-study release gate controls later test scoring."""
    from element_surrogate.data import export_benchmark_cohorts

    return export_benchmark_cohorts(
        generation_dir, output_dir, maximum_rows_per_split=maximum_rows_per_split
    )


@tool("2", read_only=False)
def prepare_neural_surrogate_study(
    path: str, generation_dir: str, output_dir: str
) -> dict[str, Any]:
    """Freeze explicit neural configurations/targets/timing against generated labels.
    PyTorch is an optional backend; sizes, seeds and budgets come from the request.
    Conserved momentum and arrival translation are exact. A reflection requires
    declared evidence. This writes no user approval and starts no fitting."""
    from element_surrogate.modeling import prepare_generated_study

    return prepare_generated_study(generation_dir, output_dir, benchmark_protocol.read_json(path))


@tool("1", input_file_args=(), read_only=False)
def build_tracking_domain_report(qualification_paths: list[str], output_dir: str) -> dict[str, Any]:
    """Plot numerical refinement versus candidate region size and propose the largest pass.
    The user chooses; physical geometry and Maxwell readiness remain separate evidence."""
    from element_surrogate.domain_report import build_domain_report

    return build_domain_report(qualification_paths, output_dir)


@tool("1", read_only=False)
def export_native_surrogate(
    path: str, output_dir: str, approval_paths: list[str]
) -> dict[str, Any]:
    """Prepare a local native PyTorch/ONNX map after locked test and release attestation.
    Verify full-map dynamic-batch equivalence on validation. Optional export
    dependencies must be installed by the caller. No upload/registry action or
    full Genesis/governance approval is performed."""
    from element_surrogate.export import export_selected

    return export_selected(path, output_dir, approval_paths)


@tool("1", read_only=False)
def check_track_domain_isolation(path: str, output_dir: str) -> dict[str, Any]:
    """Run a declared optional TRACK perturbation/sentinel/loss-timing study.
    The request supplies runtime, beam, probe files, steps and perturbation. This
    is finite black-box compatibility evidence, separate from direct sampler support."""
    from element_surrogate.track_adapter import Sol3d, TrackBeam, TrackRuntime
    from element_surrogate.track_domain import black_box_probe
    from element_surrogate.workflow import region_from_dict

    request = benchmark_protocol.read_json(path)
    base = Path(path).resolve().parent
    runtime = TrackRuntime(
        base / request["runtime"]["executable"],
        tuple(base / p for p in request["runtime"].get("libraries", [])),
        request["runtime"]["timeout_s"],
    )
    cases = []
    for case in [*request["cases"], request["sentinel"]]:
        with np.load(base / case["probes_path"], allow_pickle=False) as arrays:
            states, beta_gamma = arrays["states"], arrays["beta_gamma"]
        cases.append(
            {"element": Sol3d(**case["element"]), "states": states, "beta_gamma": beta_gamma}
        )
    result = black_box_probe(
        runtime,
        TrackBeam(**request["beam"]),
        base / request["field_path"],
        request["field_file_name"],
        region_from_dict(request["support"]),
        cases[:-1],
        cases[-1],
        output_dir,
        **request["perturbation"],
    )
    report_path = Path(output_dir) / "report.json"
    benchmark_protocol.write_json(report_path, result, exclusive=True)
    return {
        "report_path": str(report_path.resolve()),
        "accepted": result["accepted"],
        "retained_probes_isolated": result["retained_probes_isolated"],
        "sensitivity_exercised": result["sensitivity_exercised"],
        "interior_losses_observed": result["interior_losses_observed"],
        "scope": result["scope"],
    }


@tool("2", input_file_args=(), read_only=False)
def record_user_decision(
    output_dir: str,
    actor: str,
    user_statement: str,
    scope: str,
    source_reference: str,
    supersedes: list[str] | None = None,
) -> dict[str, Any]:
    """Preserve a client-supplied user's verbatim decision in a private case ledger.
    This does not authenticate the speaker, create gate approvals or promote knowledge."""
    from element_surrogate.decisions import record_decision

    return record_decision(
        output_dir,
        actor=actor,
        user_statement=user_statement,
        scope=scope,
        source_reference=source_reference,
        supersedes=supersedes or [],
    )


@tool("2", input_file_args=())
def read_user_decisions(output_dir: str) -> dict[str, Any]:
    """Read the private case's client-attested decisions; never mix users or projects."""
    from element_surrogate.decisions import list_decisions

    return list_decisions(output_dir)


@tool("4", input_file_args=("path",), read_only=False)
def collect_field_readiness(
    path: str,
    output_path: str,
    shells: int,
    end_tolerance: float | None = None,
    working_memory_bytes: int = DEFAULT_WORKING_BYTES,
    scratch_directory: str | None = None,
) -> dict[str, Any]:
    """Compute raw-grid readiness evidence bound to a transport's field and support.

    Bounded working arrays; exact global statistics use owned disk scratch, defaulting
    to output_path's directory. The array budget is not a fixed process RSS or address
    space cap. Resource/computation errors block generation and require rerunning.
    Return findings, categories and actionable nonblocking warnings. Inspect evidence
    and existing answers before useful questions; unknown geometry/source history is
    not itself an execution failure. Never infer physical validity or user approval.
    """
    from element_surrogate.field_readiness import collect_from_files

    return collect_from_files(
        path,
        output_path,
        shells=shells,
        end_tolerance=end_tolerance,
        working_memory_bytes=working_memory_bytes,
        scratch_directory=scratch_directory,
    )


@tool("1", input_file_args=())
def describe_region_request(region: dict[str, Any]) -> dict[str, Any]:
    """Validate/normalize region coordinates without opening any field.

    coordinate_basis is laboratory or nominal_path. Explicit longitudinal_bounds
    declares origin laboratory, path_entrance or path_midpoint, with min_m/max_m.
    Legacy z_min_m/z_max_m remain laboratory z without a bend and midpoint-relative
    path length with a bend. Use accepted_request to serialize again; descriptive
    summaries contain derived keys. Nominal-path offsets are not traced-reference
    phase-space coordinates. Regions are numerical bounds, not material apertures.
    """
    resolved = Region.from_request(region)
    landmarks = None
    if resolved.path is not None:
        length = resolved.path.length_m
        landmarks = {
            name: {
                "path_length_from_entrance_m": s,
                "path_length_from_midpoint_m": s - 0.5 * length,
                "axis_point_in_region": bool(resolved.contains(resolved.path.frame_at(s).origin_m)),
            }
            for name, s in (("entrance", 0.0), ("midpoint", 0.5 * length), ("exit", length))
        }
    return {
        "accepted_request": resolved.to_request(),
        "resolved": resolved.as_dict(),
        "nominal_landmarks": landmarks,
        "scope": (
            "numerical node/trajectory region; not a physical aperture or a traced reference frame"
        ),
    }


@tool("3", input_file_args=("path", "evidence_path", "review_path"), read_only=False)
def assess_field_readiness(
    path: str, evidence_path: str, review_path: str, output_path: str
) -> dict[str, Any]:
    """Bind explicit scientific interpretation to computed field evidence.

    Unresolved findings make generation ineligible; returned warnings guide analysis
    without raising an error merely for an unresolved assessment. Unknown facts may
    support an agent-reviewed numerical scope. This creates no user approval
    and does not certify physical truth, material clearance or human identity.
    """
    from element_surrogate.field_readiness import assess_from_files

    return assess_from_files(path, evidence_path, review_path, output_path)


@tool("1", input_file_args=("path", "request_path"), read_only=False)
def assess_map_symmetry(path: str, request_path: str, output_path: str) -> dict[str, Any]:
    """Measure paired native-map reflections for explicit ranges, settings and limits.

    The saved field/reference/region-bound record supports a stated numerical
    approximation, not a proof of exact physical symmetry or user approval.
    """
    from element_surrogate.symmetry import from_files

    return from_files(path, request_path, output_path)


@tool("1", input_file_args=(), read_only=False)
def initialize_surrogate_project(directory: str, name: str) -> dict[str, Any]:
    """Initialize empty private case state; inherit no developer memory or approvals."""
    from element_surrogate.project import initialize

    return initialize(directory, name)


@tool("1", input_file_args=())
def restore_surrogate_project(directory: str, max_runs: int = 20) -> dict[str, Any]:
    """Restore private decisions, knowledge summaries and bounded run metadata."""
    from element_surrogate.project import restore

    return restore(directory, max_runs)


@tool("1", input_file_args=(), read_only=False)
def propose_project_knowledge(
    directory: str, entry_id: str, frontmatter: dict[str, Any], body: str
) -> dict[str, Any]:
    """Add a proposed private knowledge entry; never invent an approver."""
    from element_surrogate.project import add_knowledge

    return add_knowledge(directory, entry_id, frontmatter, body)


@tool("1", input_file_args=(), read_only=False)
def record_project_knowledge_decision(
    directory: str,
    entry_id: str,
    expected_sha256: str,
    decision: str,
    actor: str,
    user_statement: str,
    source_reference: str,
) -> dict[str, Any]:
    """Record the client's explicit user disposition for private knowledge only.

    This does not authenticate identity, create scientific workflow gate approvals
    or change the shipped/shared knowledge base.
    """
    from element_surrogate.project import apply_knowledge_decision

    return apply_knowledge_decision(
        directory, entry_id, expected_sha256, decision, actor, user_statement, source_reference
    )


@tool("1", input_file_args=())
def inspect_project_learning_context(directory: str) -> dict[str, Any]:
    """Inspect offline knowledge/decision/guidance identity and local delegation."""
    from element_surrogate.learning import context

    return context(directory)


@tool("1", input_file_args=("policy_path",), read_only=False)
def record_project_learning_policy(
    directory: str,
    policy_path: str,
    expected_sha256: str,
    actor: str,
    user_statement: str,
    source_reference: str,
) -> dict[str, Any]:
    """Record actual user delegation or revocation for local learning only.

    No sharing, shared-KB change or scientific gate authority is created.
    The client must obtain the user's real decision; this is an attestation.
    """
    from element_surrogate.learning import record_policy

    return record_policy(
        directory, policy_path, expected_sha256, actor, user_statement, source_reference
    )


@tool("1", input_file_args=("validation_path",), read_only=False)
def activate_validated_project_knowledge(
    directory: str,
    entry_id: str,
    expected_sha256: str,
    validation_path: str,
    expected_policy_sha256: str,
    rationale: str,
) -> dict[str, Any]:
    """Use recorded local delegation and validation; otherwise retain a proposal.

    Hash checks bind reviewed evidence/context; they do not prove scientific truth.
    """
    from element_surrogate.learning import activate

    return activate(
        directory, entry_id, expected_sha256, validation_path, expected_policy_sha256, rationale
    )


@tool("1", input_file_args=(), read_only=False)
def stage_project_client_update(directory: str, update_id: str) -> dict[str, Any]:
    """Stage candidate installed client assets; never change active project science.

    Invoke from the separate candidate runtime. Keep the current runtime available.
    """
    from element_surrogate.updates import stage

    return stage(directory, update_id)


@tool("1", input_file_args=())
def inspect_project_client_update(directory: str, update_id: str) -> dict[str, Any]:
    """Read update/recovery state and a blank compatibility-review template."""
    from element_surrogate.updates import inspect_update

    return inspect_update(directory, update_id)


@tool("1", input_file_args=(), read_only=False)
def resolve_project_client_update(
    directory: str,
    update_id: str,
    expected_plan_sha256: str,
    relative_path: str,
    choice: str,
    rationale: str,
    merged_path: str | None = None,
) -> dict[str, Any]:
    """Record an explicit staged-file resolution without replacing active files."""
    from element_surrogate.updates import resolve

    return resolve(
        directory, update_id, expected_plan_sha256, relative_path, choice, rationale, merged_path
    )


@tool("1", input_file_args=("review_path",), read_only=False)
def apply_project_client_update(
    directory: str,
    update_id: str,
    expected_plan_sha256: str,
    review_path: str,
) -> dict[str, Any]:
    """Apply only a current, reconciled client plan with concrete compatibility evidence.

    The idle-checkpoint/check results are trusted-client attestations, not proofs.
    Does not install packages, change running processes or write private knowledge.
    """
    from element_surrogate.updates import apply

    return apply(directory, update_id, expected_plan_sha256, review_path)


@tool("1", input_file_args=(), read_only=False)
def rollback_project_client_update(directory: str, update_id: str) -> dict[str, Any]:
    """Recover a journaled client update; preserve later user edits and knowledge."""
    from element_surrogate.updates import rollback

    return rollback(directory, update_id)


@tool("1", input_file_args=("path",), read_only=False)
def inspect_benchmark_recovery(path: str, output_path: str) -> dict[str, Any]:
    """Verify idle coordinator/workers and save current artifact identities for recovery."""
    from element_surrogate.recovery import inspect_study

    return inspect_study(path, output_path)


@tool("1", input_file_args=("path", "inspection_path"), read_only=False)
def resume_benchmark_study(path: str, inspection_path: str, rationale: str) -> dict[str, Any]:
    """Preserve partial fits as interrupted; resume untouched trials under original deadline."""
    from element_surrogate.recovery import resume_study

    return resume_study(path, inspection_path, rationale)


@tool("1", input_file_args=(), read_only=False)
def inspect_generation_recovery(directory: str, output_path: str) -> dict[str, Any]:
    """Verify idle supervisor/child and committed shards; produce no gate approval."""
    from element_surrogate.recovery import inspect_generation

    return inspect_generation(directory, output_path)


@tool("1", input_file_args=("inspection_path",), read_only=False)
def resume_training_data_generation(
    directory: str, inspection_path: str, rationale: str
) -> dict[str, Any]:
    """Resume verified generation without resetting the original wall deadline or approvals."""
    from element_surrogate.recovery import resume_generation

    return resume_generation(directory, inspection_path, rationale)


@tool("1", input_file_args=())
def read_genesis_guidance(
    asset: str | None = None, start_line: int = 1, lines: int = 100
) -> dict[str, Any]:
    """Read the bundled pinned Genesis instructions/schema index or a bounded excerpt."""
    from element_surrogate.cards import guidance

    return guidance(asset, start_line, lines)


@tool("1", input_file_args=(), read_only=False)
def introspect_card_dataset(dataset_dir: str, output_path: str) -> dict[str, Any]:
    """Run the packaged Genesis introspector before proposing dataset capability flags.

    The result inventories actual files/metadata; the client asks for human facts
    and capability confirmation. It does not create rights or release approval.
    """
    from element_surrogate.cards import introspect_dataset

    return introspect_dataset(dataset_dir, output_path)


@tool("1", input_file_args=("study_path", "transport_path", "request_path"), read_only=False)
def prepare_genesis_release_cards(
    study_path: str,
    transport_path: str,
    generation_dir: str,
    field_dataset_dir: str,
    request_path: str,
    output_dir: str,
) -> dict[str, Any]:
    """Prepare full model and separate field/particle cards with installed Genesis assets.

    study_path is the study.json file returned by create_neural_study, not its directory.
    Run actual upstream evaluation/LinkML validation. Pending human metadata remains
    a draft; the tool creates neither release approval nor external publication.
    """
    from element_surrogate.cards import prepare

    return prepare(
        study_path, transport_path, generation_dir, field_dataset_dir, request_path, output_dir
    )


@tool("1", input_file_args=("request_path",), read_only=False)
def validate_card_metadata(request_path: str, output_path: str) -> dict[str, Any]:
    """Preflight explicit metadata/rights before simulation-dependent card creation.

    Missing owner facts remain pending. A capability Yes cannot inherit rights
    from a different artifact. This does not run upstream schema validation.
    """
    from element_surrogate.benchmark.protocol import read_json, write_json
    from element_surrogate.card_metadata import inspect_request

    result = inspect_request(read_json(request_path))
    write_json(output_path, result, exclusive=True)
    return {**result, "review_path": output_path}


@tool("1", input_file_args=("native_contract_path", "request_path"), read_only=False)
def export_public_surrogate_interface(
    native_contract_path: str, request_path: str, output_path: str
) -> dict[str, Any]:
    """Bind public units and explicit effective controls to a native map.

    Only compatible units change. No reference orbit, actual momentum or
    effective-length placement transform is introduced. See request-example.
    """
    from element_surrogate.public_interface import export_interface

    return export_interface(native_contract_path, request_path, output_path)


@tool("1", input_file_args=("path", "request_path"), read_only=False)
def assess_native_theory_base(path: str, request_path: str, output_path: str) -> dict[str, Any]:
    """Bind an ideal drift or paraxial quadrupole base to aligned native planes.

    These are approximate native-span baselines for a neural residual, not
    placement/effective-length wrappers or a certification of the measured field.
    """
    from element_surrogate.theory import from_files

    return from_files(path, request_path, output_path)


@tool("1", input_file_args=("path", "request_path"), read_only=False)
def analyze_transverse_symplecticity(
    path: str, request_path: str, output_path: str
) -> dict[str, Any]:
    """Refine transverse magnetic two-form derivatives, including boundary normal B.

    Use native slope/mechanical-momentum transformations. Fixed-momentum 4D
    slices of 5D/6D maps do not establish full longitudinal symplecticity.
    """
    from element_surrogate.symplectic import from_files

    return from_files(path, request_path, output_path)


@tool("1", input_file_args=("path", "request_path"), read_only=False)
def assess_canonical_shear_coordinates(
    path: str, request_path: str, output_path: str
) -> dict[str, Any]:
    """Assess a conservative zero-normal-field condition for the 4D shear backend.

    Nonzero normal field requires another canonical/gauge construction. Rejection
    of this sufficient condition does not reject the magnetic field or tracker.
    """
    from element_surrogate.canonical import from_files

    return from_files(path, request_path, output_path)


@tool("2", input_file_args=("cohort_path", "protocol_path"), read_only=False)
def propose_training_coordinate_scales(
    cohort_path: str,
    dimension: int,
    output_path: str,
    protocol_path: str | None = None,
    batch_rows: int = 65536,
    working_memory_bytes: int = 268435456,
) -> dict[str, Any]:
    """Propose numerical scales from the protocol's training cohort only.

    Requires native m/rad plane slopes, optional delta_p/p0 and two controls.
    Saves raw-target RMS output scales and common transverse q/mechanical-p
    magnitude proposals. Does not establish canonical eligibility or physical
    loss targets. Accepts numeric NPZ or generated train JSON shard descriptors.
    Bind protocol_path to verify the frozen training identity. Reads every
    declared row with bounded shard/batch storage; never silently subsamples.
    """
    from element_surrogate.scaling import propose_from_cohort

    return propose_from_cohort(
        cohort_path,
        dimension,
        output_path,
        protocol_path=protocol_path,
        batch_rows=batch_rows,
        working_memory_bytes=working_memory_bytes,
    )


@tool("1", input_file_args=("request_path",), read_only=False)
def check_track_build_compatibility(request_path: str, output_dir: str) -> dict[str, Any]:
    """Compare an optional actual TRACK sol3d build/grid with supplied reference probes.

    The track_compatibility.v1 request gives runtime, field/reference hashes,
    straight laboratory planes, m/rad states, momentum constants, three or more
    step counts and native per-component limits. Saves all runs and a receipt;
    a successful executable exit alone cannot pass compatibility. No universal
    grid limit, mandatory TRACK dependency, physical-field or 6D timing claim.
    """
    from element_surrogate.simulator_qualification import check

    return check(request_path, output_dir)


@tool("1", input_file_args=("request_path",), read_only=False)
def configure_scientific_campaign(project_dir: str, request_path: str) -> dict[str, Any]:
    """Bind explicit supplied campaign resources/authority; never invent an approval.

    Limits distinguish numerical threads, working arrays, disk scratch, summed
    active job-wall seconds and optional calendar time. No short duration default.
    Configuration is immutable once recorded. Ordinary shell code is outside it.
    """
    from element_surrogate.jobs import configure

    return configure(project_dir, request_path)


@tool("1", input_file_args=())
def scientific_job_catalog() -> dict[str, Any]:
    """Describe allowed managed scientific operations and their actual arguments."""
    from element_surrogate.jobs import catalog

    return catalog()


@tool("1", input_file_args=("campaign_path", "request_path"), read_only=False)
def bind_scientific_executor(campaign_path: str, request_path: str) -> dict[str, Any]:
    """Inspectably bind a legacy owner or hand off a quiescent executor.

    Requires executor_binding.v1 with expected_ledger_sha256 and rationale.
    Legacy adoption additionally requires its live recorded legacy_launch_owner
    (pid, created_unix) in this namespace. Charges/policy are never reset.
    """
    from element_surrogate.executor_binding import bind_executor

    return bind_executor(campaign_path, request_path)


@tool("1", input_file_args=("campaign_path",), read_only=False)
def inspect_scientific_diagnostic_retry(
    campaign_path: str, job_id: str, output_path: str
) -> dict[str, Any]:
    """Inspect a terminal failed diagnostic before a fresh allocation.

    Verifies inactive owners, released guard and original payload hashes. Returns
    a request template and current implementation digest. This is not a refund
    or approval; stateful training/generation retain their own recovery.
    """
    from element_surrogate.diagnostic_retry import inspect_retry

    return inspect_retry(campaign_path, job_id, output_path)


@tool("1", input_file_args=("campaign_path", "inspection_path", "request_path"), read_only=False)
def retry_scientific_diagnostic(
    campaign_path: str, job_id: str, inspection_path: str, request_path: str
) -> dict[str, Any]:
    """Create one linked diagnostic successor within the original campaign.

    Supply diagnostic_retry.v1 from the inspection template: positive new wall
    allocation, resources, fresh_output_root, exact implementation acknowledgment
    and rationale. Prior charges remain; stale inputs/owners, forks and exhausted
    original total/calendar limits are refused. No user authorization is invented.
    """
    from element_surrogate.diagnostic_retry import retry

    return retry(campaign_path, job_id, inspection_path, request_path)


@tool("1", input_file_args=("campaign_path", "request_path"), read_only=False)
def start_scientific_job(campaign_path: str, request_path: str) -> dict[str, Any]:
    """Submit a durable job and return promptly; RPC lifetime is not job lifetime.

    A job declares operation/arguments, threads/working-memory/scratch and a positive
    max_wall_seconds with explicit active/calendar semantics. Duplicate scientific
    work is not launched. No field resolution or numerical threshold is changed.
    """
    from element_surrogate.jobs import submit

    return submit(campaign_path, request_path)


@tool("1", input_file_args=("campaign_path",))
def scientific_campaign_status(campaign_path: str) -> dict[str, Any]:
    """Report all managed jobs, complete wall charges and active reservations.

    This is not a fixed RSS limit or an enforcement claim over arbitrary shell code.
    """
    from element_surrogate.jobs import campaign_status

    return campaign_status(campaign_path)


@tool("1", input_file_args=("campaign_path",))
def scientific_job_status(campaign_path: str, job_id: str) -> dict[str, Any]:
    """Read durable state, ownership/cost and result paths without opening arrays."""
    from element_surrogate.jobs import job_status

    return job_status(campaign_path, job_id)


@tool("1", input_file_args=("campaign_path",), read_only=False)
def cancel_scientific_job(campaign_path: str, job_id: str, rationale: str) -> dict[str, Any]:
    """Request cancellation; retain reservations until owned workers really exit."""
    from element_surrogate.jobs import cancel

    return cancel(campaign_path, job_id, rationale)


@tool("1", input_file_args=("campaign_path",), read_only=False)
def inspect_scientific_job_recovery(
    campaign_path: str, job_id: str, output_path: str
) -> dict[str, Any]:
    """Verify idle ownership and input identity before any explicit bounded recovery."""
    from element_surrogate.jobs import inspect_recovery

    return inspect_recovery(campaign_path, job_id, output_path)


@tool("1", input_file_args=("campaign_path", "inspection_path"), read_only=False)
def resume_scientific_job(
    campaign_path: str, job_id: str, inspection_path: str, rationale: str
) -> dict[str, Any]:
    """Resume verified unfinished work within its original time/identity contract.

    This does not reset spent time, create a new approval or retry partial model
    fits. Benchmark interruptions use the separate preserved-study recovery path.
    """
    from element_surrogate.jobs import resume

    return resume(campaign_path, job_id, inspection_path, rationale)


@tool("1", input_file_args=("campaign_path", "inspection_path", "request_path"), read_only=False)
def continue_training_data_phase(
    campaign_path: str, job_id: str, inspection_path: str, request_path: str
) -> dict[str, Any]:
    """Continue a verified paused pilot into a linked generation execution phase.

    Request schema generation_continuation.v1 supplies max_batches (null for all
    remaining work), positive max_wall_seconds, rationale and optional resources.
    Original inputs/IDs/plan allowance, all charges and calendar bounds remain.
    This is not fault recovery, a refit or permission to change scientific inputs.
    """
    from element_surrogate.jobs import continue_generation_phase

    return continue_generation_phase(campaign_path, job_id, inspection_path, request_path)


@tool(
    "2",
    input_file_args=(
        "path",
        "qualification_path",
        "readiness_path",
        "request_path",
        "population_policy_path",
    ),
    read_only=False,
)
def build_training_data_request(
    path: str,
    qualification_path: str,
    readiness_path: str,
    request_path: str,
    output_path: str,
    population_policy_path: str | None = None,
) -> dict[str, Any]:
    """Build a generation contract from the actual transport and exact qualification.

    The supplied generation_request.v1 names qualified setting IDs, split/survivor
    counts, seed, batch/proposal/wall budgets and material evidence. Transport
    payload and ranges are computed. Returns the exact subject hash for separately
    supplied user attestations; never invents approvals or a reference trajectory.
    An optional population_policy.v1 checks the requested retained-survivor floor.
    """
    from element_surrogate.generation_planning import build_request

    return build_request(
        path, qualification_path, readiness_path, request_path, output_path, population_policy_path
    )


@tool("2", input_file_args=(), read_only=False)
def summarize_generation_capacity(directory: str, output_path: str) -> dict[str, Any]:
    """Verify committed batches and summarize their measured tracking/publication cost.

    Conditional per-setting projections need at least two survivor-bearing batches.
    Untimed batches and untested settings stay explicit. Does not claim million-row
    acceptance or sample sufficiency from an extrapolation.
    Flags the default 12-elapsed-hour resource question from known projected work;
    uncalibrated settings/startup/queue/transfer are additional or unknown costs.
    """
    from element_surrogate.generation_planning import capacity_report

    return capacity_report(directory, output_path)


@tool("1", input_file_args=("plan_path", "policy_path"), read_only=False)
def assess_population_requirement(
    plan_path: str, policy_path: str, output_path: str, generation_dir: str | None = None
) -> dict[str, Any]:
    """Check planned/verified actual survivor counts; count alone never proves adequacy.

    The optional generation directory is verified against committed shards and the
    exact plan identity. Pilot/partial records cannot claim full population completion.
    """
    from element_surrogate.population import assess

    return assess(plan_path, policy_path, output_path, generation_dir)


@tool("1", input_file_args=("request_path",), read_only=False)
def estimate_workload_time(request_path: str, output_path: str) -> dict[str, Any]:
    """Project remaining generation/study elapsed time from source-bound observations.

    Declare each stage, same hardware/numerical context and included costs. Two
    positive repeated observations are required; missing/zero-yield phases remain
    inconclusive. Reports when the configured 12-hour resource question is needed.
    """
    from element_surrogate.workload import estimate

    return estimate(request_path, output_path)


@tool("2", input_file_args=("request_path",), read_only=False)
def prepare_improvement_feedback(
    directory: str, feedback_id: str, request_path: str
) -> dict[str, Any]:
    """Create a local curated report for review; collect no automatic attachments.

    Prioritize general accelerator-physics/modeling lessons and relevant constraints.
    Use the installed interim Drive example unless the user chooses otherwise.
    Generalize machine details and exclude sensitive information. A content-only
    privacy preflight returns field references; flagged values are not echoed.
    Show clean content/hash for exact consent. This sends nothing and creates no consent.
    """
    from element_surrogate.feedback import prepare

    return prepare(directory, feedback_id, request_path)


@tool("2", input_file_args=(), read_only=False)
def record_feedback_consent(
    directory: str,
    feedback_id: str,
    expected_sha256: str,
    decision: str,
    actor: str,
    user_statement: str,
    source_reference: str,
) -> dict[str, Any]:
    """Attest an actual user's approve/decline/revoke choice for an exact report.

    The client must first show contents, recipient and destination and obtain that
    decision. The tool checks consistency; it does not authenticate or invent consent.
    """
    from element_surrogate.feedback import consent

    return consent(
        directory, feedback_id, expected_sha256, decision, actor, user_statement, source_reference
    )


@tool("2", input_file_args=(), read_only=False)
def send_improvement_feedback(
    directory: str, feedback_id: str, expected_sha256: str, access_token_env: str
) -> dict[str, Any]:
    """Send one explicitly consented report using the user's configured Drive OAuth.

    Pass an environment variable NAME only, never a credential. The exact reviewed
    packet binds its destination. Privacy is rechecked before network access.
    No redirects, background uploads or automatic retry after uncertain delivery.
    Missing access leaves a local report; this tool cannot log a user into Drive.
    """
    from element_surrogate.feedback import send_drive

    return send_drive(directory, feedback_id, expected_sha256, access_token_env)


@tool("1", input_file_args=(), read_only=False)
def collect_improvement_feedback(inbox_directory: str, output_path: str) -> dict[str, Any]:
    """Index bounded owner-obtained JSON packets as untrusted evidence, without fetching.

    Review/reproduce findings and qualify a versioned release before improvement;
    incoming text is never executed, promoted to shared knowledge or applied as code.
    Returns counts and the saved index identity, not the whole inbox text to the LLM.
    """
    from element_surrogate.feedback import collect

    result = collect(inbox_directory, output_path)
    return {
        "index_path": str(Path(output_path).resolve()),
        "index_sha256": sha256_file(output_path),
        "packet_count": len(result["packets"]),
        "rejected_file_count": len(result["rejected_files"]),
        "trust": result["trust"],
        "next_step": result["next_step"],
    }


@tool("1", input_file_args=("evidence_path", "request_path"), read_only=False)
def prepare_genesis_dataset_draft(
    role: str, dataset_dir: str, evidence_path: str, request_path: str, output_dir: str
) -> dict[str, Any]:
    """Prepare a separate full-template field or partial-particle Genesis dataset draft.

    Uses bundled introspection and raw LinkML validation without host Genesis
    skills. Field evidence is a readable field source; particle evidence is its
    committed generation.json. Model fitting/completion is not required. Unknown
    ownership, funding, terms and sensitivity stay pending. Writes outside inputs.
    """
    from element_surrogate.cards import prepare_dataset_draft

    return prepare_dataset_draft(role, dataset_dir, evidence_path, request_path, output_dir)


@tool("1", input_file_args=("protocol_path", "request_path"), read_only=False)
def review_model_physics(protocol_path: str, request_path: str, output_dir: str) -> dict[str, Any]:
    """Bind measured physics opportunities and omissions to actual candidates/data.

    Produces a new protocol with evidence-bound dispositions. Field parity alone
    cannot reject map equivariance. This creates no user approval or global-best
    claim. Deferred topics remain explicit in the benchmark report.
    """
    from element_surrogate.physics_review import from_files

    return from_files(protocol_path, request_path, output_dir)


@tool("1", input_file_args=("request_path",), read_only=False)
def prepare_supplied_neural_surrogate_study(request_path: str, output_dir: str) -> dict[str, Any]:
    """Validate existing native label cohorts and prepare the common neural study.

    Reads declared train/validation arrays with bounded existing cohort guards,
    verifies units/hashes/IDs and records split-use history. No test targets,
    label generation, physics qualification or user approvals are produced.
    """
    from element_surrogate.supplied_data import prepare

    return prepare(request_path, output_dir)


@tool("1", input_file_args=("transport_path", "protocol_path", "request_path"), read_only=False)
def assess_supplied_label_reference(
    transport_path: str, protocol_path: str, request_path: str, output_path: str
) -> dict[str, Any]:
    """Compare existing train/validation labels with a declared tracker sample.

    Records source/coordinate/control identities, statuses and component errors.
    This diagnostic performs no fitting or training-label generation. A sampled
    pass does not certify every label or physical field accuracy.
    """
    from element_surrogate.supplied_data import check_reference_files

    return check_reference_files(transport_path, protocol_path, request_path, output_path)


@tool("1", input_file_args=("study_path", "transport_path", "request_path"), read_only=False)
def prepare_supplied_genesis_release_cards(
    study_path: str,
    transport_path: str,
    label_dataset_dir: str,
    field_dataset_dir: str,
    request_path: str,
    output_dir: str,
) -> dict[str, Any]:
    """Create model/field/particle cards for a native study using supplied labels.

    Reuses pinned Genesis procedures. Requires the provided-label source contract,
    actual selected artifact and recorded test history. Does not invent a local
    generation ledger, fresh test, owner, license or physical qualification.
    """
    from element_surrogate.cards import prepare

    return prepare(
        study_path,
        transport_path,
        label_dataset_dir,
        field_dataset_dir,
        request_path,
        output_dir,
        supplied_labels=True,
    )


@tool("2", input_file_args=("request_path",), read_only=False)
def write_scientific_uncertainty_review(request_path: str, output_path: str) -> dict[str, Any]:
    """Save reasoned analysis and a scoped path forward, without adding a gate.

    Use for material unresolved facts, not a mandatory checklist. Preserve evidence,
    alternatives, limitations, existing authority and outputs that can continue.
    Unknown geometry/source facts may allow numerical research. The client judges
    physics; this tool creates no answer, approval or qualification and overrides
    no numerical/resource/consent check. Return actionable nonblocking warnings for
    unresolved/partial scope; reuse answered/delegated facts. Keep visible records.
    """
    from element_surrogate.uncertainty import review

    return review(request_path, output_path)
