"""Optional black-box TRACK isolation control, separate from direct sampler proof."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from element_surrogate.fields import read_track_table
from element_surrogate.runs import sha256_file
from element_surrogate.track_adapter import run_element


def perturb_track_table(source, output, support, *, factor=None, addition_t=None, slot=None):
    """Perturb only outside magnetic rows; preserve all other bytes exactly.

    A caller-supplied additive vector also exercises initially zero values.
    Multiplication is retained for replaying existing black-box experiments.
    Interleaved zero electric rows and a duplicate terminal row are preserved.
    """
    if (factor is None) == (addition_t is None):
        raise ValueError("give exactly one perturbation factor or additive vector")
    if factor is not None and (not np.isfinite(factor) or factor == 1):
        raise ValueError("finite nontrivial factor required")
    if addition_t is not None:
        addition_t = np.asarray(addition_t, float)
        if addition_t.shape != (3,) or not np.isfinite(addition_t).all() or not np.any(addition_t):
            raise ValueError("finite nonzero additive magnetic vector in tesla required")
    original = read_track_table(source)
    grid = original.slot(slot)
    selected = 0 if slot is None else slot
    outside = ~support.node_mask(grid)
    rows_per_node = original.integrity["rows_per_node"]
    data_index = 0
    changed = 0
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open("rb") as reader, output.open("xb") as writer:
        for _ in range(5):
            writer.write(reader.readline())
        for raw in reader:
            numeric, separator, comment = raw.partition(b"#")
            if not numeric.strip():
                writer.write(raw)
                continue
            node = min(data_index // rows_per_node, int(np.prod(grid.shape)) - 1)
            magnetic = (
                rows_per_node == 1
                or data_index % rows_per_node == 1
                or data_index >= int(np.prod(grid.shape)) * rows_per_node
            )
            index = np.unravel_index(node, grid.shape)
            if magnetic and outside[index]:
                values = np.fromstring(numeric.decode("ascii"), sep=" ")
                first = 3 * selected
                if factor is not None:
                    values[first : first + 3] *= factor
                else:
                    values[first : first + 3] += addition_t * 1e4
                ending = b"\r\n" if raw.endswith(b"\r\n") else b"\n" if raw.endswith(b"\n") else b""
                modified = "\t".join(format(float(v), ".17g") for v in values).encode("ascii")
                writer.write(
                    modified + (b" #" + comment.rstrip(b"\r\n") if separator else b"") + ending
                )
                changed += 1
            else:
                writer.write(raw)
            data_index += 1
    if sha256_file(source) != original.sha256:
        raise ValueError("source field changed while preparing perturbation")
    return {
        "source_sha256": original.sha256,
        "perturbed_sha256": sha256_file(output),
        "outside_nodes": int(outside.sum()),
        "changed_magnetic_rows": changed,
        "factor": factor,
        "addition_t": None if addition_t is None else addition_t.tolist(),
        "scope": "only supplied outside magnetic rows changed; all other rows byte-identical",
    }


def compare_runs(baseline, perturbed):
    if baseline.raw_states is None or perturbed.raw_states is None:
        raise ValueError("raw simulator outputs required for a black-box comparison")
    live = baseline.transported
    changed_flags = baseline.flags != perturbed.flags
    row_equal = np.all(
        (baseline.raw_states == perturbed.raw_states)
        | (np.isnan(baseline.raw_states) & np.isnan(perturbed.raw_states)),
        axis=1,
    )
    return {
        "baseline_survivors": int(live.sum()),
        "changed_flags": int(changed_flags.sum()),
        "changed_raw_rows": int((~row_equal).sum()),
        "retained_probe_isolation": bool(live.any())
        and not np.any(changed_flags[live])
        and bool(row_equal[live].all()),
        "perturbation_responded": bool(changed_flags.any() or not row_equal.all()),
    }


def black_box_probe(
    runtime,
    beam,
    field_file,
    field_file_name,
    support,
    cases,
    sentinel,
    output_dir,
    *,
    factor=None,
    addition_t=None,
):
    """Cases declare element/states/beta_gamma; sentinel must exercise outside nodes.

    The library supplies no simulator-specific perturbation strength, particle
    count, radius or step. Results apply only to the probes/configuration tested.
    A nonresponding sentinel is inconclusive, never a passed isolation gate.
    """
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    perturbed = directory / "perturbed.table"
    receipt = perturb_track_table(
        field_file, perturbed, support, factor=factor, addition_t=addition_t
    )
    reports = []
    interior = 0
    for i, case in enumerate([*cases, sentinel]):
        runs = [
            run_element(
                runtime,
                case["element"],
                source,
                field_file_name,
                beam,
                case["states"],
                case["beta_gamma"],
                workdir=directory / f"case_{i}_{label}",
            )
            for label, source in (("baseline", field_file), ("perturbed", perturbed))
        ]
        result = compare_runs(*runs)
        losses = runs[0].loss_positions_m
        inside = (
            0
            if losses is None
            else int(np.count_nonzero((losses > 0) & (losses < case["element"].length_m)))
        )
        if i < len(cases):
            interior += inside
        reports.append(
            {**result, "interior_losses": inside, "loss_file_present": losses is not None}
        )
    return {
        "cases": reports[:-1],
        "sentinel": reports[-1],
        "perturbation": receipt,
        "retained_probes_isolated": bool(cases)
        and all(r["retained_probe_isolation"] for r in reports[:-1]),
        "sensitivity_exercised": reports[-1]["perturbation_responded"],
        "interior_losses_observed": interior,
        "accepted": bool(cases)
        and all(r["retained_probe_isolation"] for r in reports[:-1])
        and reports[-1]["perturbation_responded"]
        and interior > 0,
        "scope": (
            "finite black-box probe, not direct knowledge of TRACK "
            "interpolation or a material aperture"
        ),
    }
