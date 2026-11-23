"""Optional finite-probe compatibility checks for an actual TRACK sol3d build."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from element_surrogate.benchmark.protocol import file_hash, read_json, write_json
from element_surrogate.fields import read_track_table
from element_surrogate.track_adapter import (
    MomentumConstants,
    Sol3d,
    TrackBeam,
    TrackRuntime,
    run_element,
)


def check(request_path, output_dir):
    """Keep build/grid identity and numerical compatibility separate from physics.

    Reference outputs must be supplied independently in the declared straight
    laboratory planes. This tool neither creates independent truth nor certifies
    a physical aperture or field solve. No grid-capacity limit is assumed.
    """
    request = read_json(request_path)
    if request.get("schema") != "element_surrogate.track_compatibility.v1":
        raise ValueError("track_compatibility.v1 request required")
    base = Path(request_path).resolve().parent

    def path(value):
        return (base / value).resolve()

    runtime = TrackRuntime(
        path(request["runtime"]["executable"]),
        tuple(path(p) for p in request["runtime"].get("libraries", [])),
        request["runtime"].get("timeout_s", 600.0),
    )
    beam = TrackBeam(**request["beam"])
    constants = MomentumConstants(**request["momentum_constants"])
    steps = request["steps"]
    if len(steps) < 3 or any(type(n) is not int or n < 1 for n in steps):
        raise ValueError("at least three positive integer refinement step counts required")
    if any(a >= b for a, b in zip(steps[:-1], steps[1:], strict=True)):
        raise ValueError("strictly increasing TRACK step counts required")
    element = Sol3d(**request["element"], steps=steps[-1])
    planes = request["straight_lab_planes_z_m"]
    if not np.allclose(planes, [-element.length_m / 2, element.length_m / 2], rtol=0, atol=1e-12):
        raise ValueError("this sol3d tool requires centered straight laboratory reference planes")
    field = path(request["field"]["path"])
    probes = path(request["reference"]["path"])
    for p, expected in (
        (field, request["field"]["sha256"]),
        (probes, request["reference"]["sha256"]),
    ):
        if file_hash(p) != expected:
            raise ValueError("field or independent reference bytes changed")
    if request.get("state_units") != ["m", "rad", "m", "rad"]:
        raise ValueError("native m/rad plane direction-ratio units required")
    if not request["reference"].get("provenance"):
        raise ValueError("independent reference provenance required; caller attestation only")
    limits = np.asarray(request["compatibility_limits_native"], dtype=float)
    refine = np.asarray(request["refinement_limits_native"], dtype=float)
    for value in (limits, refine):
        if value.shape != (4,) or not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError("four positive finite native component limits required")
    with np.load(probes, allow_pickle=False) as data:
        states = data["states"]
        expected = data["expected"]
        reached = data["reached_exit"]
        rigidity = np.broadcast_to(data["rigidity_tm"], (len(states),)).copy()
    if (
        states.ndim != 2
        or states.shape[1] != 4
        or expected.shape != states.shape
        or reached.shape != (len(states),)
        or reached.dtype.kind != "b"
        or not len(states)
        or not np.isfinite(states).all()
        or not np.isfinite(expected[reached]).all()
    ):
        raise ValueError(
            "finite reference states, matching expected outputs and boolean exits required"
        )
    grid = read_track_table(field).slots[0]
    identities = {
        str(p): file_hash(p)
        for p in (
            Path(request_path).resolve(),
            field,
            probes,
            runtime.executable,
            *runtime.libraries,
        )
    }
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    runs = []
    for n in steps:
        result = run_element(
            runtime,
            replace(element, steps=n),
            field,
            request["field"]["filename"],
            beam,
            states,
            constants.beta_gamma(rigidity, beam.charge, beam.mass_number),
            workdir=output / f"steps_{n}",
        )
        np.savez(output / f"steps_{n}.npz", states=result.states, flags=result.flags)
        runs.append(result)
    status_changes = sum(
        int(np.count_nonzero(a.transported != b.transported))
        for a, b in zip(runs[:-1], runs[1:], strict=True)
    )
    common = reached & np.logical_and.reduce([r.transported for r in runs])

    def difference(a, b, threshold):
        error = np.abs(a[common] - b[common])
        maximum = np.max(error, axis=0) if common.any() else None
        return {
            "max_absolute_native": None if maximum is None else maximum.tolist(),
            "within_limits": maximum is not None and bool(np.all(maximum <= threshold)),
        }

    adjacent = [
        difference(a.states, b.states, refine) for a, b in zip(runs[:-1], runs[1:], strict=True)
    ]
    agreement = difference(runs[-1].states, expected, limits)
    status_mismatch = int(np.count_nonzero(runs[-1].transported != reached))
    stable_inputs = all(file_hash(p) == digest for p, digest in identities.items())
    report = {
        "schema": "element_surrogate.track_compatibility_receipt.v1",
        "identities": identities,
        "grid_shape": list(grid.shape),
        "grid_extent_m": [[float(a[0]), float(a[-1])] for a in (grid.x_m, grid.y_m, grid.z_m)],
        "steps": steps,
        "rows": len(states),
        "common_exits": int(common.sum()),
        "state_units": request["state_units"],
        "straight_lab_planes_z_m": planes,
        "reference_provenance": request["reference"]["provenance"],
        "reference_is_caller_supplied_not_authenticated": True,
        "compatibility_limits_native": limits.tolist(),
        "refinement_limits_native": refine.tolist(),
        "adjacent_refinement": adjacent,
        "compatibility": agreement,
        "status_changes": status_changes,
        "reference_exit_status_mismatches": status_mismatch,
        "inputs_unchanged": stable_inputs,
        "passed": bool(
            common.any()
            and not status_changes
            and not status_mismatch
            and stable_inputs
            and all(a["within_limits"] for a in adjacent[-2:])
            and agreement["within_limits"]
        ),
        "scope": (
            "finite transverse sol3d compatibility; no physical-field or universal "
            "build-capacity certificate"
        ),
        "automatic_generation_or_release_approval": False,
    }
    write_json(output / "receipt.json", report, exclusive=True)
    return {**report, "receipt_path": str(output / "receipt.json")}
