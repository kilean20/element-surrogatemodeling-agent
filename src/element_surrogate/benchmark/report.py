"""Reports and one validated adapter to the pinned benchmark-builder renderer."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import file_hash, read_json, write_json
from .scoring import load_cohort
from .study import (
    _guard,
    _read_study,
    _validated_trials,
    configuration_summaries,
    evaluate_test,
    latency_eligibility,
    lock_selection,
    pareto_frontier,
)

VENDOR = Path(__file__).parents[1] / "_vendor/benchmark_builder"


def verify_vendor():
    record = read_json(VENDOR / "manifest.json")
    for name, expected in record["sha256"].items():
        if file_hash(VENDOR / name) != expected:
            raise ValueError("vendored benchmark-builder source changed")
    return record


def _selected_metrics(study_path):
    directory, record, p = _read_study(study_path)
    if not (directory / "selection.json").exists():
        raise ValueError("selection lock required for a selected-model report")
    lock_selection(study_path)
    lock = read_json(directory / "selection.json")
    test_path = directory / "test/result.json"
    if test_path.exists():
        test = read_json(test_path)
        approvals = [a["path"] for a in test["release_approvals"]]
        for receipt in test["release_approvals"]:
            if file_hash(receipt["path"]) != receipt["sha256"]:
                raise ValueError("release approval changed")
        metrics = evaluate_test(study_path, approvals)["metrics"]
        source = test_path
        predictions = directory / "test/test.npz"
    else:
        source = Path(lock["selected"]["trial_path"])
        metrics = read_json(source)["metrics"]
        predictions = source.parent / "validation.npz"
    return directory, record, p, lock, metrics, source, predictions


def _write_checked(path, text, check):
    path = Path(path)
    if check:
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            raise ValueError(f"generated report changed or missing: {path.name}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise ValueError("choose a new output directory for changed evidence")
        path.write_text(text, encoding="utf-8")


def render_card(study_path, output_dir, *, check=False) -> dict[str, Any]:
    """Create/check a result-block draft; full Genesis card validation is separate."""
    vendor = verify_vendor()
    directory, record, p, lock, metrics, source, _ = _selected_metrics(study_path)
    if (
        metrics["protocol_sha256"] != record["protocol_sha256"]
        or metrics["coordinate_units"] != [c["unit"] for c in p["coordinates"]]
        or metrics["coordinate_names"] != [c["name"] for c in p["coordinates"]]
        or metrics["checkpoint_sha256"] != lock["selected"]["checkpoint_sha256"]
    ):
        raise ValueError("metric units/coordinates/model/protocol mismatch")
    groups = {}
    for j, c in enumerate(p["coordinates"]):
        groups[f"{c['name']} ({c['unit']})"] = {
            key: metrics[key][j]
            for key in (
                "rmse",
                "mae",
                "p95_absolute",
                "p99_absolute",
                "p999_absolute",
                "max_absolute",
                "setting_mean_rmse",
            )
        }
    values = [v for row in groups.values() for v in row.values()]
    if not np.isfinite(values).all():
        raise ValueError("nonfinite card metrics")
    adapter = {
        "coverage": metrics["coverage"],
        "groups": {"all": groups},
        "provenance": {
            "metric_source_sha256": file_hash(source),
            "selection_sha256": file_hash(directory / "selection.json"),
            "protocol_sha256": record["protocol_sha256"],
            "quantile_method": metrics["quantile_method"],
        },
    }
    output = Path(output_dir).resolve()
    model_id = lock["selected"]["candidate_id"]
    metric_path = output / "card_results" / metrics["split"] / model_id / "metrics.json"
    if check:
        if read_json(metric_path) != adapter:
            raise ValueError("card adapter changed")
    else:
        if metric_path.exists() and read_json(metric_path) != adapter:
            raise ValueError("choose a new card output directory for changed evidence")
        write_json(metric_path, adapter)
    card = output / "RESULTS_CARD.md"
    existed = card.exists()
    if not check and not card.exists():
        card.write_text(
            "# Selected model result-block draft\n\nThis draft contains benchmark results. "
            "Full model-card metadata, Genesis template/schema review and release approval "
            "are separate requirements.\n\n## Evaluation results\n",
            encoding="utf-8",
        )
    command = [
        sys.executable,
        "-I",
        str(VENDOR / "scripts/render_card_results.py"),
        str(card),
        "--benchmark",
        p["name"],
        "--model",
        model_id,
        "--results",
        str(output / "card_results"),
        "--splits",
        metrics["split"],
    ]
    # Existing drafts are checked, never silently rewritten after a manual edit.
    completed = subprocess.run(
        [*command, "--check"] if check or existed else command,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode:
        raise ValueError(completed.stdout + completed.stderr)
    return {
        "card_path": str(card),
        "adapter_path": str(metric_path),
        "card_sha256": file_hash(card),
        "adapter_sha256": file_hash(metric_path),
        "upstream_commit": vendor["commit"],
        "full_card_validated": False,
    }


def _scatter(rows, title, xlabel, ylabel, *, focus=None):
    rows = sorted(rows, key=lambda r: (r[1], r[0]))
    focus_index = next((i for i, r in enumerate(rows) if r[0] == focus), None)
    legend_count = min(len(rows), 11 if focus_index is not None else 12)
    named = set(np.linspace(0, len(rows) - 1, legend_count, dtype=int).tolist())
    if focus_index is not None:
        named.add(focus_index)
    named = sorted(named)
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    named_colors = {index: colors[j % len(colors)] for j, index in enumerate(named)}
    width, height, left, top, plot_width, plot_height = 840, 380, 80, 45, 450, 260
    xs = np.array([r[1] for r in rows], dtype=float)
    ys = np.array([r[2] for r in rows], dtype=float)
    xmax, ymax = max(float(xs.max()), 1e-30) * 1.1, max(float(ys.max()), 1e-30) * 1.1
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<g font-family="sans-serif" font-size="13"><text x="80" y="25">{escape(title)}</text>',
        f'<path d="M {left} {top} v {plot_height} h {plot_width}" fill="none" stroke="black"/>',
    ]
    for i in range(6):
        x, y = left + plot_width * i / 5, top + plot_height * (1 - i / 5)
        lines += [
            f'<text x="{x:.1f}" y="325" text-anchor="middle">{xmax * i / 5:.3g}</text>',
            f'<text x="72" y="{y + 4:.1f}" text-anchor="end">{ymax * i / 5:.3g}</text>',
        ]
    for i, (label, x, y) in enumerate(rows):
        color = named_colors.get(i, "#888888")
        selected = i == focus_index
        px, py = left + plot_width * x / xmax, top + plot_height * (1 - y / ymax)
        lines += [
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{5.5 if selected else 4}" '
            f'fill="{color}" stroke="{"black" if selected else "none"}" stroke-width="2" '
            f'data-candidate-id="{escape(label)}" data-selected="{str(selected).lower()}">'
            f"<title>{escape(label)}: time {x:.6g}; score {y:.6g}</title></circle>",
        ]
    for j, index in enumerate(named):
        label = rows[index][0]
        lines.append(
            f'<text x="550" y="{65 + 18 * j}" fill="{named_colors[index]}" '
            f'data-candidate-label="{escape(label)}">{escape(label)}</text>'
        )
    lines += [
        '<text x="550" y="286" font-size="11">All points shown; see full table.</text>',
        '<text x="550" y="304" font-size="11">Outline: selected configuration.</text>',
        f'<text x="305" y="360" text-anchor="middle">{escape(xlabel)}</text>',
        '<text transform="translate(18 180) rotate(-90)" text-anchor="middle">'
        f"{escape(ylabel)}</text>",
        "</g></svg>\n",
    ]
    return "\n".join(lines)


def _components(p, metrics):
    width = max(720, len(p["coordinates"]) * 125)
    values = np.array([metrics[k] for k in ("rmse", "p99_absolute", "max_absolute")])
    values /= np.array([c["error_scale"] for c in p["coordinates"]])
    maximum = max(float(values.max()), 1e-30) * 1.12
    colors = ("#0072B2", "#009E73", "#D55E00")
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="400" viewBox="0 0 {width} 400">',
        '<rect width="100%" height="100%" fill="white"/>'
        '<g font-family="sans-serif" font-size="13">',
        '<text x="60" y="25">Selected model errors divided by each coordinate error scale</text>',
        f'<path d="M 60 50 v 240 h {width - 80}" fill="none" stroke="black"/>',
    ]
    for i in range(6):
        lines.append(
            f'<text x="52" y="{294 - i * 48}" text-anchor="end">{maximum * i / 5:.3g}</text>'
        )
    for j, coordinate in enumerate(p["coordinates"]):
        center = 60 + (width - 80) * (j + 0.5) / len(p["coordinates"])
        for k in range(3):
            h = 240 * values[k, j] / maximum
            lines.append(
                f'<rect x="{center - 30 + k * 20:.2f}" y="{290 - h:.2f}" '
                f'width="17" height="{h:.2f}" fill="{colors[k]}"/>'
            )
        lines.append(
            f'<text x="{center:.1f}" y="312" text-anchor="middle">'
            f"{escape(coordinate['name'])}</text>"
        )
        lines.append(
            f'<text x="{center:.1f}" y="332" text-anchor="middle">'
            f"scale {coordinate['error_scale']:g} {escape(coordinate['unit'])}</text>"
        )
    for k, name in enumerate(("RMSE", "P99 absolute", "Maximum absolute")):
        lines.append(f'<text x="{80 + 210 * k}" y="375" fill="{colors[k]}">{name}</text>')
    return "\n".join([*lines, "</g></svg>\n"])


def _validation_comparison(p, configurations, trials):
    """Small validated summaries; never pool seed quantiles or read test results."""
    rows = []
    for configuration in configurations:
        sources = [r for r in trials if r["candidate_id"] == configuration["candidate_id"]]
        metrics = []
        for source in sources:
            if file_hash(source["trial_path"]) != source["trial_sha256"]:
                raise ValueError("comparison trial changed after validation")
            metrics.append(read_json(source["trial_path"])["metrics"])
        row = dict(configuration)
        for name in ("rmse", "setting_mean_rmse", "p99_absolute"):
            row["mean_seed_" + name] = np.mean([m[name] for m in metrics], axis=0).tolist()
        row["worst_seed_p99_absolute"] = np.max(
            [m["p99_absolute"] for m in metrics], axis=0
        ).tolist()
        row["source_trials"] = [
            {k: source[k] for k in ("seed", "trial_path", "trial_sha256", "checkpoint_sha256")}
            for source in sources
        ]
        rows.append(row)
    return {
        "schema": "element_surrogate.validation_comparison.v1",
        "split": "validation",
        "coordinate_names": [c["name"] for c in p["coordinates"]],
        "coordinate_units": [c["unit"] for c in p["coordinates"]],
        "aggregation": "Arithmetic mean of seed-level statistics; worst seed P99 is their maximum. "
        "These are not pooled quantiles or confidence bounds.",
        "configurations": rows,
    }


def _comparison_lines(p, comparison):
    lines = [
        "## Validation component comparisons",
        "",
        comparison["aggregation"],
        "Global RMSE and setting-mean RMSE are distinct. The primary score retains "
        "its declared per-seed scaling and aggregation; max and mean need not commute.",
        "",
        "| Configuration | Coordinate | Unit | Mean seed RMSE | Mean seed setting-mean RMSE | "
        "Mean seed P99 | Worst seed P99 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in comparison["configurations"]:
        for j, coordinate in enumerate(p["coordinates"]):
            values = " | ".join(
                f"{row[name][j]:.6g}"
                for name in (
                    "mean_seed_rmse",
                    "mean_seed_setting_mean_rmse",
                    "mean_seed_p99_absolute",
                    "worst_seed_p99_absolute",
                )
            )
            lines.append(
                f"| {row['candidate_id']} | {coordinate['name']} | {coordinate['unit']} | "
                f"{values} |"
            )
    return lines


def _write_comparison(output, comparison, check):
    path = output / "validation_comparison.json"
    _write_checked(
        path, json.dumps(comparison, sort_keys=True, indent=2, allow_nan=False) + "\n", check
    )
    return path


def build_tradeoff_report(study_path, output_dir, *, check=False) -> dict[str, Any]:
    """Validate and report observations without a feasible selection or test access."""
    directory, record, p = _read_study(study_path)
    output = Path(output_dir).resolve()
    with _guard(directory):
        trials, exclusions = _validated_trials(directory, record, p)
        configurations = configuration_summaries(p, trials)
        comparison = _validation_comparison(p, configurations, trials)
        comparison_path = _write_comparison(output, comparison, check)
        scatter = _scatter(
            [(r["candidate_id"], r["median_us_per_row"], r["mean_score"]) for r in configurations],
            f"Validation tradeoffs ({len(configurations)} complete configurations)",
            "Median microseconds per row",
            "Mean primary score (lower is better)",
        )
        _write_checked(output / "accuracy_cost.svg", scatter, check)
        lines = [
            f"# {p['name']}: validation-only tradeoff exploration",
            "",
            "This report selects no final model, creates no selection lock and reads no test "
            "cohort or test-result file. All complete configurations are shown regardless "
            "of deployment feasibility. Existing goals, locks and trial records are unchanged.",
            "",
            "Speed-regime/family coverage and optimization convergence are not inferred "
            "from this plot. The client must assess them using the study design "
            "and timing variation.",
            "",
            "| Configuration | Architecture | Seeds | Mean primary score | Median us/row |",
            "|---|---|---:|---:|---:|",
        ]
        lines.extend(
            f"| {r['candidate_id']} | {r['architecture']} | {r['seed_count']} | "
            f"{r['mean_score']:.6g} | {r['median_us_per_row']:.6g} |"
            for r in configurations
        )
        lines += [
            "",
            "Observed accuracy/configuration-cost Pareto frontier: "
            + ", ".join(r["candidate_id"] for r in pareto_frontier(configurations)),
            "",
            "![Validation accuracy/cost](accuracy_cost.svg)",
            "",
        ]
        checks = latency_eligibility(p, configurations, trials)
        if checks is not None:
            eligible = [r["candidate_id"] for r in checks if r["eligible"]]
            lines += [
                "Declared latency rule: `" + json.dumps(p["selection"], sort_keys=True) + "`.",
                "",
                "Eligible on configuration and representative-checkpoint adapter cost: "
                + (", ".join(eligible) or "none; no feasible model is asserted"),
                "",
            ]
        lines += _comparison_lines(p, comparison)
        lines += [
            "",
            f"Timing: {p['timing']['hardware']}; device {p['timing']['device']}; "
            f"batch {p['timing']['batch_size']}; {p['timing']['threads']} threads; "
            f"{p['timing']['repetitions']} repetitions. Scope: {p['timing']['scope']}.",
            "Included: " + p["timing"]["included_operations"],
            "Excluded: " + p["timing"]["excluded_operations"],
            "",
            "Adapter costs are not a full exported/checked API latency guarantee. "
            "Review actual deployment timing before claiming compliance.",
            "",
            f"Protocol SHA256: `{record['protocol_sha256']}`. "
            f"Comparison SHA256: `{file_hash(comparison_path)}`.",
            f"Failed/incomplete trial exclusions: {len(exclusions)}; "
            "the declared incomplete policy applies.",
            "Supplied-label prior-use history: "
            + str(
                p["dataset"]
                .get("supplied_label_provenance", {})
                .get("split_history", "not supplied")
            ),
            "Qualification scope: " + p["physics"]["qualification_scope"],
            "",
        ]
        _write_checked(output / "REPORT.md", "\n".join(lines), check)
    return {
        "report_path": str(output / "REPORT.md"),
        "report_sha256": file_hash(output / "REPORT.md"),
        "comparison_path": str(comparison_path),
        "plot_paths": [str(output / "accuracy_cost.svg")],
        "split": "validation",
        "configurations": len(configurations),
        "excluded_trials": len(exclusions),
        "test_opened": False,
        "selection_lock_created": False,
        "selection_lock_present": (directory / "selection.json").exists(),
    }


def build_report(study_path, output_dir, *, check=False) -> dict[str, Any]:
    directory, record, p, lock, metrics, source, _ = _selected_metrics(study_path)
    output = Path(output_dir).resolve()
    card = render_card(study_path, output, check=check)
    configurations = lock["configurations"]
    comparison = _validation_comparison(p, configurations, lock["trials"])
    comparison_path = _write_comparison(output, comparison, check)
    # All points stay in the artifact; cap only legend names, not cost coverage.
    scatter = _scatter(
        [(r["candidate_id"], r["median_us_per_row"], r["mean_score"]) for r in configurations],
        f"Validation accuracy and inference cost ({len(configurations)} complete configurations)",
        "Median microseconds per row",
        "Mean primary score (lower is better)",
        focus=lock["selected"]["candidate_id"],
    )
    _write_checked(output / "accuracy_cost.svg", scatter, check)
    _write_checked(output / "component_errors.svg", _components(p, metrics), check)
    selected = next(c for c in p["candidates"] if c["id"] == lock["selected"]["candidate_id"])
    from element_surrogate.physics_review import summarize as physics_summary

    physics_review = physics_summary(p)
    lines = [
        f"# {p['name']}: bounded model study",
        "",
        f"Selected `{selected['id']}`, seed {lock['selected']['seed']}.",
        (
            "The checkpoint before iterative optimization won validation selection; "
            "see adapter provenance for fitted or pretrained components."
            if lock["selected"].get("best_step") == 0
            else f"Selected optimizer step: {lock['selected'].get('best_step', 'unrecorded')}."
        ),
        "",
        "## Physics opportunities and study scope",
        "",
        "Physics review: " + physics_review["status"] + ". " + physics_review["scope"],
        "Unresolved topics: " + (", ".join(physics_review["unresolved_topics"]) or "none recorded"),
        "",
        "Supplied-label prior-use history: "
        + str(
            p["dataset"]
            .get("supplied_label_provenance", {})
            .get("split_history", "not a supplied-data import")
        ),
        "",
        "## Task and model",
        "",
        p["coordinate_convention"],
        "",
        selected["forward_description"],
        "",
        "Candidate rationale: " + selected["rationale"],
        "",
        "## Validation selection",
        "",
        "Configurations are selected on validation. Stochastic seeds are averaged within "
        "each configuration; hyperparameter configurations are compared separately. "
        f"The declared relative accuracy tie is {p['selection']['relative_accuracy_tie']:g}; "
        "eligible configurations are ranked by median measured cost.",
        "",
        "| Configuration | Architecture | Seeds | Mean score | Seed SD | Median us/row |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in configurations:
        standard_deviation = "N/A" if r["std_score"] is None else f"{r['std_score']:.6g}"
        lines.append(
            f"| {r['candidate_id']} | {r['architecture']} | {r['seed_count']} | "
            f"{r['mean_score']:.6g} | {standard_deviation} | {r['median_us_per_row']:.6g} |"
        )
    if "policy" in p["selection"]:
        rule = p["selection"]
        lines += ["", "Selection policy: `" + rule["policy"] + "`.", ""]
        if rule["policy"] == "accuracy_with_latency_budget":
            relation = (
                "strictly below"
                if rule.get("latency_bound", "inclusive") == "exclusive"
                else "at or below"
            )
            lines += [
                f"The inference ceiling is **{rule['max_median_us_per_row']:g} us/row** "
                f"at the declared workload. Only configurations {relation} it enter "
                "the accuracy comparison and relative tie. Both the configuration "
                "median of seed-level medians and its best-validation feasible checkpoint's "
                "own median must pass. This is not maximum, tail or full exported-API latency.",
                "",
                "Selected checkpoint adapter median: "
                f"{lock['selected']['cost_us_per_row']:.6g} us/row. "
                f"Its validation primary score is {lock['selected']['primary_score']:.6g}.",
                "All required seeds remain in configuration statistics; the selected "
                "checkpoint is the most accurate measured latency-feasible seed.",
                "",
                "Budget rationale: " + rule["budget_rationale"],
                "",
                "Latency-eligible configurations: "
                + ", ".join(lock["latency_eligible_candidate_ids"]),
            ]
        lines += [
            "",
            "Accuracy/cost Pareto frontier (all complete configurations): "
            + ", ".join(r["candidate_id"] for r in lock["pareto_frontier"])
            + ". A point remains when no other configuration is at least as good "
            "in both score and cost and strictly better in one. This observed "
            "frontier does not establish converged or matched-latency family coverage.",
            "",
        ]
    lines += [
        "",
        "![Validation accuracy/cost](accuracy_cost.svg)",
        "",
        *_comparison_lines(p, comparison),
        "",
        "## Selected model errors",
        "",
        f"Split: **{metrics['split']}**, {metrics['rows']} rows, "
        f"{metrics['setting_count']} setting groups. "
        f"Absolute quantile method: `{metrics['quantile_method']}`.",
        f"Holdout unit: **{p['dataset'].get('split_unit', 'setting')}**. "
        + (
            "Settings may be shared; this does not test unseen-setting generalization."
            if p["dataset"].get("split_unit") == "particle"
            else "Setting groups are disjoint across train/validation/test."
        ),
        "",
        "| Coordinate | Unit | RMSE | MAE | P99 absolute | Maximum absolute |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for j, c in enumerate(p["coordinates"]):
        row = " | ".join(
            f"{metrics[k][j]:.6g}" for k in ("rmse", "mae", "p99_absolute", "max_absolute")
        )
        lines.append(f"| {c['name']} | {c['unit']} | {row} |")
    lines += [
        "",
        "![Component errors and tails](component_errors.svg)",
        "",
        "## Timing scope",
        "",
        f"{p['timing']['hardware']}; device `{p['timing']['device']}`; "
        f"batch {p['timing']['batch_size']}; "
        f"{p['timing']['repetitions']} repeats; dtype `{p['timing']['input_dtype']}`; "
        f"{p['timing']['threads']} requested threads. Scope: `{p['timing']['scope']}`.",
        "",
        "Included: " + p["timing"]["included_operations"],
        "",
        "Excluded: " + p["timing"]["excluded_operations"],
        "",
        "Raw repeats and P10/P90 timing variation are retained in each trial record. "
        "Timings describe this adapter, environment and batch; "
        "hardware-independent rankings are not established.",
        "",
        "## Qualification and provenance",
        "",
        p["physics"]["qualification_scope"],
        "",
        p["physics"]["boundary_convention"],
        "",
        "Benchmark scores measure the declared labels. "
        "They do not certify absolute field accuracy, a physical aperture, "
        "exact symplecticity or a global error bound.",
        "",
        "Independent standard-metric recomputation is not asserted by this report; "
        "an external evaluator "
        "and its explicit cohort receipt are a separate control.",
        "",
        f"Protocol SHA256: `{record['protocol_sha256']}`. "
        f"Selection SHA256: `{file_hash(directory / 'selection.json')}`.",
        f"Metric-source SHA256: `{file_hash(source)}`. "
        f"Selected checkpoint SHA256: `{lock['selected']['checkpoint_sha256']}`.",
        "",
        "Data access/reuse: " + p["dataset"]["access_and_reuse"],
        "",
        f"Excluded failed/incomplete trials: {len(lock['excluded'])}; "
        "reasons remain in the selection record.",
        "",
        "[Generated result-block draft](RESULTS_CARD.md). "
        "Full Genesis metadata/schema validation and "
        "external publication approval remain separate from this local report.",
        "",
    ]
    _write_checked(output / "REPORT.md", "\n".join(lines), check)
    return {
        "report_path": str(output / "REPORT.md"),
        "report_sha256": file_hash(output / "REPORT.md"),
        "comparison_path": str(comparison_path),
        "plot_paths": [
            str(output / name) for name in ("accuracy_cost.svg", "component_errors.svg")
        ],
        "split": metrics["split"],
        "rows": metrics["rows"],
        **card,
    }


def prepare_metric_crosscheck(study_path, output_dir, maximum_rows) -> dict[str, Any]:
    """Export a named bounded cohort, never silently call it a full evaluation."""
    if type(maximum_rows) is not int or maximum_rows < 1:
        raise ValueError("explicit positive external-evaluator row cap required")
    _, record, p, _, metrics, _, predictions = _selected_metrics(study_path)
    truth = load_cohort(p, record["base"], metrics["split"])
    with np.load(predictions, allow_pickle=False) as data:
        order = np.argsort(data["sample_id"].astype(str))
        predicted = data["prediction"][order]
    order = np.argsort(truth["sample_id"])
    selected = np.linspace(0, len(order) - 1, min(len(order), maximum_rows), dtype=int)
    indices = order[selected]
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    path = output / "joined.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                *[f"true_{j}" for j in range(len(p["coordinates"]))],
                *[f"pred_{j}" for j in range(len(p["coordinates"]))],
            ]
        )
        for k, idx in enumerate(indices):
            writer.writerow(
                [truth["sample_id"][idx], *truth["targets"][idx], *predicted[selected[k]]]
            )
    receipt = {
        "split": metrics["split"],
        "protocol_sha256": record["protocol_sha256"],
        "source_rows": len(order),
        "exported_rows": len(indices),
        "external_maximum_rows": maximum_rows,
        "cohort": "evenly spaced sorted sample IDs",
        "joined_sha256": file_hash(path),
        "coordinate_units": metrics["coordinate_units"],
        "external_evaluator_run": False,
        "scope": "positional-alignment input for global component MAE/RMSE; "
        "setting weights/tails not checked",
    }
    write_json(output / "cohort.json", receipt)
    return {"csv_path": str(path), "receipt_path": str(output / "cohort.json"), **receipt}


def score_documentation(rubric_path, output_path) -> dict[str, Any]:
    verify_vendor()
    target = Path(output_path).resolve()
    if target.exists():
        raise ValueError("choose a fresh self-assessment report path")
    target.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(VENDOR / "scripts/score_benchmark.py"),
            str(Path(rubric_path).resolve()),
            "--out",
            str(target),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode:
        raise ValueError(completed.stdout + completed.stderr)
    text = (
        "Internal documentation self-assessment. "
        "This is not institutional or MLCommons endorsement.\n\n"
        + target.read_text(encoding="utf-8")
    )
    target.write_text(text, encoding="utf-8")
    return {
        "report_path": str(target),
        "report_sha256": file_hash(target),
        "rubric_sha256": file_hash(rubric_path),
        "official_endorsement": False,
    }
