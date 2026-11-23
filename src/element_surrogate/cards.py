"""Self-contained Genesis card preparation with actual upstream validators.

Metadata is explicit case input. It cannot establish human identity, rights,
funding, sensitivity or institutional release approval. No external publication.
"""

import os
import re
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import psutil
import yaml

from element_surrogate.benchmark.protocol import file_hash, read_json, write_json
from element_surrogate.benchmark.report import _selected_metrics
from element_surrogate.card_frontmatter import compose_metrics, metric_coverage
from element_surrogate.card_metadata import (
    classify_upstream_data_validation,
    model_rights_text,
    require_valid_request,
    review_model_structure,
)
from element_surrogate.data import verify_generation
from element_surrogate.model_inputs import collect as collect_model_inputs
from element_surrogate.model_inputs import frontmatter as input_frontmatter
from element_surrogate.model_inputs import render as render_model_inputs
from element_surrogate.model_provenance import collect as collect_model_provenance
from element_surrogate.model_provenance import render as render_model_provenance
from element_surrogate.workflow import load_transport

VENDOR = Path(__file__).parent / "_vendor/genesis"


def verify_assets():
    manifest = read_json(VENDOR / "manifest.json")
    for name, digest in manifest["sha256"].items():
        if file_hash(VENDOR / name) != digest:
            raise ValueError("bundled Genesis asset changed: " + name)
    return manifest


def guidance(asset=None, start_line=1, lines=100) -> dict:
    """Read bounded portions of pinned instructions/schema without host skills."""
    manifest = verify_assets()
    if asset is None:
        return {
            "commit": manifest["commit"],
            "assets": sorted(manifest["sha256"]),
            "license_note": manifest["license_note"],
        }
    if asset not in manifest["sha256"]:
        raise ValueError("choose an asset from the bundled index")
    if (
        type(start_line) is not int
        or type(lines) is not int
        or start_line < 1
        or not 1 <= lines <= 200
    ):
        raise ValueError("positive starting line and 1..200 lines required")
    text = (VENDOR / asset).read_text(encoding="utf-8").splitlines()
    return {
        "asset": asset,
        "sha256": manifest["sha256"][asset],
        "total_lines": len(text),
        "text": "\n".join(text[start_line - 1 : start_line - 1 + lines]),
    }


def _command(args, output, timeout=120):
    """Run a bounded card helper with durable output and owned-child cleanup.

    File-backed output prevents an inherited pipe from holding communicate()
    open after timeout. Known descendants are stopped; this is trusted helper
    lifecycle handling, not a sandbox against an intentionally escaping process.
    Scientific job allowances are managed elsewhere and are unchanged.
    """
    command = [sys.executable, *args]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUTF8": "1"},
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except OSError as error:
            log.write("Helper launch failed: " + type(error).__name__ + ": " + str(error))
            raise
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _stop_helper(process)
            log.flush()
            raise subprocess.TimeoutExpired(
                command, timeout, output=output.read_text(encoding="utf-8", errors="replace")
            ) from None
    return subprocess.CompletedProcess(
        command, returncode, stdout=output.read_text(encoding="utf-8", errors="replace"), stderr=""
    )


def _stop_helper(process):
    """Stop only this still-owned Popen process and its observed descendants."""
    descendants = []
    if process.poll() is None:
        try:
            descendants = psutil.Process(process.pid).children(recursive=True)
        except psutil.Error:
            pass
        try:
            process.kill()
        except ProcessLookupError:
            pass
    for child in reversed(descendants):
        try:
            child.kill()
        except psutil.Error:
            pass
    if descendants:
        psutil.wait_procs(descendants, timeout=2)
    process.wait(timeout=5)


def _normalize(value):
    from element_surrogate.card_metadata import normalize_heading

    return normalize_heading(value)


def _filename(kind, supplied):
    name = (
        supplied.get("frontmatter", {})
        .get("discoverability", {})
        .get("identification", {})
        .get("name", kind)
    )
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not slug:
        raise ValueError("dataset name needs a usable filename")
    return "genesis_datacard_" + slug + ".md"


def _dataset_body(front, notes):
    """Capability-gated narrative from the same metadata, per the body-fill guide."""
    sections = [
        ("Dataset Description", None, ["discoverability.dataset_description.dataset_summary"]),
        (
            "Domain and Purpose",
            None,
            [
                "discoverability.dataset_description.science_domain",
                "discoverability.dataset_description.purpose",
            ],
        ),
        ("Security / Marking Considerations", None, ["discoverability.sensitivity"]),
        ("Context and Provenance", None, ["discoverability.identification"]),
        (
            "Resources used, including funding and facilities, to create the dataset",
            None,
            [
                "discoverability.sponsor_organizations",
                "discoverability.research_organizations",
                "discoverability.facilities",
            ],
        ),
        ("Developed by", None, ["discoverability.authors"]),
        ("Contributed by", None, ["discoverability.contributors", "discoverability.contact"]),
        (
            "Dataset generation, collection, and procedures",
            None,
            [
                "discoverability.dataset_description.collection_methodology",
                "interoperability.provenance",
            ],
        ),
        ("Sharing & Access", "accessibility", ["accessibility"]),
        ("Files & Structure", "interoperability", ["interoperability.data_structure"]),
        ("Related Resources", "interoperability", ["interoperability.related_resources"]),
        (
            "Understanding the Data",
            "interoperability",
            ["interoperability.semantic_layer", "interoperability.data_structure.features"],
        ),
        ("Semantic / Schema Information", "interoperability", ["interoperability.domain_metadata"]),
        ("Citation", "reusability", ["reusability.citation"]),
        ("License and Usage Rights", "reusability", ["reusability.license"]),
        ("Maintenance & Updates", "reusability", ["reusability.stewardship"]),
        (
            "Data Characteristics",
            "interoperability",
            ["discoverability.dataset_description.data_characteristics"],
        ),
        (
            "Data Quality & Limitations",
            "reusability",
            ["reusability.data_quality", "discoverability.dataset_description.limitations"],
        ),
        ("Integrity & Versioning", "reusability", ["reusability.integrity"]),
        (
            "Access and Permissions",
            "governed_use",
            ["governed_use.compliance", "governed_use.non_sensitivity_governance_metadata"],
        ),
        ("Access conditions", "governed_use", ["governed_use.use_governance"]),
        ("Review Provenance", "governed_use", ["governed_use.review_provenance_companion"]),
        ("AI / Machine Learning Considerations", "ai_usability", ["ai_usability.ai_usage"]),
    ]
    lines = [
        notes,
        "",
        "### Machine Usability Snapshot",
        "",
        "| Intended capability | DataCard support |",
        "| --- | --- |",
    ]
    lines += [
        f"| {k.removeprefix('supports_')} | {v} |"
        for k, v in front.items()
        if k.startswith("supports_")
    ]
    for title, capability, paths in sections:
        if capability and front.get("supports_" + capability) != "Yes":
            continue
        values = {}
        for path in paths:
            value = front
            for key in path.split("."):
                value = value.get(key) if isinstance(value, dict) else None
            if value is not None:
                values[path] = value
        lines += ["", "## " + title, ""]
        if not values:
            lines += ["Not provided; see the pending metadata and validation findings."]
        else:
            for path, value in values.items():
                lines += [f"**{path.rsplit('.', 1)[-1].replace('_', ' ')}**", ""]
                lines += (
                    [value]
                    if isinstance(value, str)
                    else ["```yaml", yaml.safe_dump(value, sort_keys=False).strip(), "```"]
                )
                lines += [""]
    return "\n".join(lines)


def _bundle(study_path, output):
    directory, record, p, lock, metrics, source, _ = _selected_metrics(study_path)
    if metrics["split"] != "test":
        raise ValueError("held-out evaluation of the locked model required for release cards")
    selected = lock["selected"]
    trial = read_json(selected["trial_path"])
    rows = []
    for j, coordinate in enumerate(p["coordinates"]):
        for metric in (
            "rmse",
            "mae",
            "p95_absolute",
            "p99_absolute",
            "p999_absolute",
            "max_absolute",
            "setting_mean_rmse",
        ):
            rows.append(
                {
                    "benchmark": p["name"],
                    "metric": f"{coordinate['name']}/{metric}[{coordinate['unit']}]",
                    "value": metrics[metric][j],
                    "scale": "raw",
                    "is_partial": False,
                    "higher_is_better": False,
                    "num_samples": metrics["coverage"]["n_scored"],
                    "num_samples_total": metrics["coverage"]["n_truth"],
                    "num_runs": 1,
                    "seeds": [selected["seed"]],
                    "artifact_path": str(source),
                }
            )
    rows.append(
        {
            "benchmark": p["name"],
            "metric": "full_prediction_median_us_per_row[us/row]",
            "value": trial["timing"]["median_us_per_row"],
            "scale": "raw",
            "is_partial": False,
            "higher_is_better": False,
            "artifact_path": selected["trial_path"],
        }
    )
    bundle = {
        "schema_version": "1.0",
        "run": {"harness": "particle-map-regression", "artifact_root": str(directory)},
        "model": {"id": selected["candidate_id"], "id_provenance": "harness"},
        "compute": {
            "device": p["timing"]["device"],
            "batch_size": p["timing"]["batch_size"],
            "hardware_note": p["timing"]["hardware"],
        },
        "results": rows,
    }
    schema = read_json(VENDOR / "card-eval-updater/references/eval-bundle.schema.json")
    schema["properties"]["run"]["properties"]["harness"]["enum"].append("particle-map-regression")
    import jsonschema

    jsonschema.Draft202012Validator(schema).validate(bundle)
    write_json(output / "eval-bundle.schema.json", schema)
    write_json(output / "eval_bundle.json", bundle)
    return p, record, lock, metrics, trial


def _data_card(kind, dataset, supplied, output, provenance):
    """Introspect first; only explicit metadata supplies governance/authorship."""
    review = require_valid_request({"datasets": {kind: supplied}})["roles"][kind]
    dataset = Path(dataset).resolve()
    if not dataset.is_dir():
        raise ValueError("dataset directory required")
    script = VENDOR / "datacard-generator/scripts/introspect.py"
    result = _command([str(script), str(dataset)], output / f"{kind}_introspection.json")
    if result.returncode:
        raise ValueError("Genesis introspection failed")
    introspection = read_json(output / f"{kind}_introspection.json")
    front = deepcopy(supplied.get("frontmatter", {}))
    # These are proposals until a client-attested confirmation is supplied.
    defaults = {
        "discoverability": "Yes",
        "accessibility": "Yes",
        "interoperability": "Yes",
        "reusability": "No",
        "governed_use": "No",
        "ai_usability": "Yes",
    }
    for cap, value in defaults.items():
        front.setdefault("supports_" + cap, value)
    name = _filename(kind, supplied)
    discover = front.setdefault("discoverability", {})
    card_meta = discover.setdefault("datacard", {})
    card_meta.update(
        template_version="1.2",
        datacard_version="1.2",
        filename=name,
        language="en",
        creation_method="Hybrid",
        created_date=datetime.now(UTC).date().isoformat(),
    )
    from element_surrogate import __version__
    from element_surrogate.card_metadata import software_identity

    card_meta.setdefault(
        "created_by",
        [
            {
                "contribution_date": datetime.now(UTC).date().isoformat(),
                "creator": {"software": software_identity("element-surrogate", __version__)},
            }
        ],
    )
    discover.setdefault("release_status", "Draft")
    discover.setdefault("product_type", "Data")
    content = (
        "---\n"
        + yaml.safe_dump(front, sort_keys=False)
        + "---\n\n"
        + f"# {kind.title()} dataset\n\n"
        + _dataset_body(front, supplied.get("body", "Metadata draft; owner facts remain pending."))
        + "\n\n## Computed provenance\n\n"
        + provenance
        + f"\n\nIntrospection: `{kind}_introspection.json`. "
        + "Exact artifact identities: `traceability.json`.\n"
    )
    (output / name).write_text(content, encoding="utf-8")
    yaml_path = output / f"{kind}_frontmatter.yaml"
    yaml_path.write_text(yaml.safe_dump(front), encoding="utf-8")
    validation = _command(
        [
            "-m",
            "linkml.validator.cli",
            "-s",
            str(VENDOR / "datacard-generator/scripts/genesis_datacard.yaml"),
            "-C",
            "GenesisDatacardClass",
            str(yaml_path),
        ],
        output / f"{kind}_validation.log",
    )
    confirmation = supplied.get("capability_confirmation", {})
    confirmed = all(
        isinstance(confirmation.get(k), str) and confirmation[k].strip()
        for k in ("actor", "user_statement", "source_reference")
    )
    return {
        "card_path": str(output / name),
        "schema_pass": validation.returncode == 0,
        "upstream_validation": classify_upstream_data_validation(
            front,
            (output / f"{kind}_validation.log").read_text(encoding="utf-8"),
            validation.returncode == 0,
        ),
        "validator_exit_code": validation.returncode,
        "capabilities_confirmed": confirmed,
        "confirmation_provenance": "client attestation; not authenticated identity",
        "metadata_missing": review["metadata_missing"],
        "rights_confirmed": review["rights_confirmed"],
        "introspection": introspection,
    }


def introspect_dataset(dataset_dir, output_path) -> dict:
    verify_assets()
    directory, output = Path(dataset_dir).resolve(), Path(output_path).resolve()
    if not directory.is_dir() or output.exists():
        raise ValueError("existing dataset directory and unused introspection output required")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = _command(
        [str(VENDOR / "datacard-generator/scripts/introspect.py"), str(directory)], output
    )
    if result.returncode:
        raise ValueError("Genesis introspection failed; inspect the saved output")
    return {
        "introspection_path": str(output),
        "sha256": file_hash(output),
        "introspection": read_json(output),
        "capability_confirmation_required": True,
    }


def prepare_dataset_draft(role, dataset_dir, evidence_path, request_path, output_dir) -> dict:
    """Document a field or committed partial particle dataset before model fitting.

    This uses the pinned full Genesis dataset template and raw validator. A
    lifecycle draft is not a release, and missing human facts remain missing.
    """
    verify_assets()
    if role not in {"field", "particle"}:
        raise ValueError("separate field or particle dataset role required")
    try:
        metadata.version("linkml")
    except metadata.PackageNotFoundError as exc:
        raise ValueError("install element-surrogate[cards] in this environment") from exc
    request = read_json(request_path)
    supplied = request.get("datasets", {}).get(role)
    if supplied is None:
        raise ValueError(
            "explicit metadata for this dataset role required; unknown facts may remain pending"
        )
    review = require_valid_request({"datasets": {role: supplied}})
    dataset, evidence = Path(dataset_dir).resolve(), Path(evidence_path).resolve()
    if not evidence.is_relative_to(dataset):
        raise ValueError("dataset evidence must belong to the described dataset directory")
    source_evidence_sha256 = file_hash(evidence)
    if role == "particle":
        if evidence != dataset / "generation.json":
            raise ValueError("particle draft requires its committed generation.json")
        generation = verify_generation(dataset)
        computed = {
            "lifecycle_status": generation["status"],
            "plan_sha256": generation["plan_sha256"],
            "transport": generation["plan"]["transport"],
            "qualification_sha256": generation["plan"]["qualification_sha256"],
            "field_readiness_sha256": generation["plan"]["field_readiness_sha256"],
            "committed_survivors": sum(generation["survivors"].values()),
            "committed_proposals": sum(generation["attempts"].values()),
            "settings": generation["plan"]["settings"],
            "survivors_by_setting": generation["survivors"],
            "shards": generation["shards"],
            "training_performed_by_this_tool": False,
        }
    else:
        from element_surrogate.fields import load_field_file

        field = load_field_file(evidence)
        computed = {
            "lifecycle_status": "source dataset; physical readiness is separate evidence",
            "field_sources": [grid.source.as_dict() for grid in field.slots],
            "grid_shapes": [list(grid.shape) for grid in field.slots],
            "internal_units": {"coordinates": "m", "field": "T"},
            "training_performed_by_this_tool": False,
        }
    output = Path(output_dir).resolve()
    if output.is_relative_to(dataset):
        raise ValueError(
            "write draft outside the input dataset; introspection must not ingest its own output"
        )
    output.mkdir(parents=True, exist_ok=False)
    trace = {
        "schema": "element_surrogate.dataset_draft.v1",
        "role": role,
        "dataset_directory": str(dataset),
        "evidence_path": str(evidence),
        "evidence_sha256": source_evidence_sha256,
        "computed": computed,
        "metadata_request_sha256": file_hash(request_path),
        "genesis_assets": verify_assets(),
        "publication_performed": False,
    }
    write_json(output / "traceability.json", trace)
    write_json(output / "metadata_review.json", review)
    result = _data_card(
        role,
        dataset,
        supplied,
        output,
        "Lifecycle draft with verified source/committed bytes in traceability.json. "
        f"Current lifecycle: {computed['lifecycle_status']}. "
        "Partial data are not complete training data; no trained model or release is asserted.",
    )
    result.update(
        role=role,
        lifecycle_status=computed["lifecycle_status"],
        traceability_path=str(output / "traceability.json"),
        publication_performed=False,
    )
    if file_hash(evidence) != source_evidence_sha256:
        raise ValueError(
            "dataset evidence changed during draft preparation; preserve this failed draft "
            "and retry with a stable snapshot"
        )
    write_json(output / "result.json", result)
    return result


def prepare(
    study_path,
    transport_path,
    generation_dir,
    field_dataset_dir,
    request_path,
    output_dir,
    *,
    supplied_labels=False,
) -> dict:
    """Create full-template model and two separately validated dataset cards."""
    manifest = verify_assets()
    for dependency in ("linkml", "jsonschema"):
        try:
            metadata.version(dependency)
        except metadata.PackageNotFoundError as exc:
            raise ValueError("install element-surrogate[cards] in this environment") from exc
    request = read_json(request_path)
    metadata_review = require_valid_request(request)
    transport, _ = load_transport(transport_path)
    if supplied_labels:
        from element_surrogate.supplied_data import card_source

        _, _, source_protocol, _, _, _, _ = _selected_metrics(study_path)
        generation = card_source(source_protocol, transport)
    else:
        generation = verify_generation(generation_dir)
        if (
            generation["status"] != "complete"
            or generation["plan"]["transport"] != transport.description()
        ):
            raise ValueError("complete generation for exactly this transport required")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "metadata_review.json", metadata_review)
    p, record, lock, metrics, trial = _bundle(study_path, output)
    if supplied_labels:
        dataset_root = Path(generation_dir).resolve()
        if any(
            not Path(v["path"]).resolve().is_relative_to(dataset_root)
            for v in p["dataset"]["splits"].values()
        ):
            raise ValueError("supplied dataset directory must contain the actual cohort files")
    supplied = request.get("model", {})
    physical_provenance = collect_model_provenance(transport, supplied)
    from element_surrogate.model_documentation import selected_description

    model_summary = selected_description(
        trial, Path(lock["selected"]["trial_path"]).parent / "model.bin"
    )
    write_json(output / "selected_model_summary.json", model_summary)
    input_contract = collect_model_inputs(
        transport,
        generation,
        p,
        supplied,
        protocol_sha256=record["protocol_sha256"],
        selected=lock["selected"],
    )
    write_json(output / "model_physical_provenance.json", physical_provenance)
    write_json(output / "model_input_contract.json", input_contract)
    # Bind the actual source kind; an import is never a fabricated generation.
    from element_surrogate.physics_review import summarize

    physics_review = summarize(p)
    if supplied_labels:
        generation_hash = generation["source_record_sha256"]
    else:
        generation_hash = file_hash(Path(generation_dir) / "generation.json")
        if (
            Path(p["dataset"].get("generation_record_path", "")).resolve()
            != (Path(generation_dir) / "generation.json").resolve()
            or p["dataset"]["identity_sha256"] != generation["plan_sha256"]
        ):
            raise ValueError("benchmark dataset is not bound to this generation")
    write_json(
        output / "traceability.json",
        {
            "genesis_commit": manifest["commit"],
            "schema_sha256": file_hash(VENDOR / "datacard-generator/scripts/genesis_datacard.yaml"),
            "transport": transport.description(),
            (
                "supplied_label_declaration_sha256" if supplied_labels else "generation_sha256"
            ): generation_hash,
            "label_origin": "supplied" if supplied_labels else "locally_generated",
            "supplied_label_provenance": p["dataset"].get("supplied_label_provenance"),
            "protocol_sha256": record["protocol_sha256"],
            "selection": lock,
            "request_sha256": file_hash(request_path),
            "model_physical_provenance": physical_provenance,
            "selected_model_summary": model_summary,
            "model_input_contract": input_contract,
            "physics_modeling_review": physics_review,
            "capability_metadata": request.get("datasets", {}),
            "validator_versions": {k: metadata.version(k) for k in ("linkml", "jsonschema")},
            "regression_adapter": (
                "Explicit regression harness enum extension; all generated blocks and "
                "the pinned validator are unchanged. Regression interpretation is "
                "separate narrative."
            ),
        },
    )
    data = request.get("datasets", {})
    field_name = _filename("field", data.get("field", {}))
    particle_name = _filename("particle", data.get("particle", {}))
    if field_name == particle_name:
        raise ValueError("field and particle dataset identities must be distinct")
    if transport.grid.source is None or not Path(
        transport.grid.source.path
    ).resolve().is_relative_to(Path(field_dataset_dir).resolve()):
        raise ValueError("field dataset directory must contain the transport's actual source")
    field = _data_card(
        "field",
        field_dataset_dir,
        data.get("field", {}),
        output,
        (
            "Static field source for the declared Cartesian reconstructio"
            "n. Its numerical support is not a mechanical aperture. "
        )
        + f"Transport/source: `{transport_path}`. "
        + f"Particle card: [particle]({particle_name}).",
    )
    particle = _data_card(
        "particle",
        generation_dir,
        data.get("particle", {}),
        output,
        (
            "Supplied particle-map labels, not generated by this study. "
            + "Source-declared provenance and prior split use: "
            + __import__("json").dumps(p["dataset"]["supplied_label_provenance"])
            + ". Array hashes and native interpretation are bound to the protocol. "
            + f"Source [field]({field_name})."
            if supplied_labels
            else (
                "Survivor-conditioned single-element labels; committed "
                "proposal/status and survivor shards. "
            )
            + f"Generation: `{Path(generation_dir) / 'generation.json'}`. "
            + f"Source [field]({field_name}). "
            + (
                "Split/ranges/reference, measured integration qualification and "
                "field-readiness interpretations are in traceability.json and the "
                "immutable campaign plan."
            )
        ),
    )
    supplied = request.get("model", {})
    from element_surrogate.card_metadata import model_sections

    narratives, section_coverage = model_sections(supplied)
    auto = {
        "model short description": model_summary["short_description"],
        "model description": "Architecture and full configuration: "
        + f"`{lock['selected']['candidate_id']}`.\n\n```json\n"
        + __import__("json").dumps(trial["candidate"], indent=2)
        + "\n```",
        "inputs and outputs": "```json\n"
        + __import__("json").dumps(
            {k: p[k] for k in ("inputs", "input_units", "coordinates", "reference") if k in p},
            indent=2,
        )
        + "\n```",
        "training data": f"[Particle dataset]({particle_name}), "
        + f"derived from [field dataset]({field_name}). "
        + "Survivor conditioning excludes losses from regression accuracy.",
        "training procedure": (
            "Fixed protocol, common split/loss/metrics and full-predictio"
            "n timing; selection used validation only. Candidate seeds ar"
            "e replicates. The locked model alone received the saved held"
            "-out evaluation."
        ),
        "training configuration": (
            "Exact candidate configuration, seed, selected step and proto"
            "col/selection identities are in traceability.json. No effect"
            "ive-length placement wrapper is added."
        ),
        "inference configuration": (
            "Use the exported native-plane coordinate/reference contract,"
            " supported numerical domain and declared control reduction. "
            "Integrator convergence does not prove field correctness."
        ),
        "evaluation procedure": (
            "Errors use each component's native unit; saved empirical qua"
            "ntiles/maxima are cohort-specific. Timing includes the proto"
            "col-declared full prediction at its measured batch/hardware."
            " This is deterministic particle-map regression; token decoding does not apply. "
            "Recorded seeds identify training initialization, not stochastic inference."
        ),
        "uncertainty quantification": (
            "Empirical error quantiles/maxima describe the scored cohort, not uniform error "
            "bounds, experimental uncertainty or calibrated per-particle confidence. "
            "The automated block concerns dispersion estimates for this selected evaluation; "
            "numerical qualification and source-field validity remain separate evidence."
        ),
        "pre-training information": (
            "No pretrained foundation model is required. The study record"
            " identifies any theoretical or fitted residual base."
        ),
        "finetuned from model (optional)": (
            "See declared candidate configuration; do not infer a pretrained model."
        ),
    }
    headings = read_json(VENDOR / "model_card_headings.json")["headings"]
    seen, body, pending = set(), [], []
    for heading in headings:
        key = _normalize(heading)
        if key in seen:
            continue
        seen.add(key)
        content = (
            model_rights_text(supplied)
            if key == "model license"
            else narratives.get(key, auto.get(key))
        )
        if not isinstance(content, str) or not content.strip():
            content = "Pending case-owner documentation/review."
            pending.append(key)
        if key == "training data":
            content += "\n\n" + render_model_provenance(physical_provenance)
        if supplied_labels and key in ("training data", "training procedure"):
            content += "\n\nSupplied-label origin and prior split use: "
            content += __import__("json").dumps(
                p["dataset"]["supplied_label_provenance"], sort_keys=True
            )
            content += (
                ". No training labels were generated by this study. Previously used "
                "or unknown-history test rows are not a new untouched test."
            )
        if key == "model description":
            content += "\n\n" + model_summary["selection_interpretation"]
            content += "\n\nPhysics-opportunity review: " + physics_review["status"] + ". "
            if physics_review["unresolved_topics"]:
                content += (
                    "Unresolved topics: " + ", ".join(physics_review["unresolved_topics"]) + ". "
                )
            content += physics_review["scope"]
            content += "\n\nSelected artifact details:\n```json\n"
            content += __import__("json").dumps(model_summary, indent=2) + "\n```"
        if key == "inputs and outputs":
            content = render_model_inputs(input_contract) + "\n\n" + content
        body += ["## " + heading.strip().rstrip("."), "", content, ""]
    front = deepcopy(supplied.get("frontmatter", {}))
    if physical_provenance["design_length"].get("value_m") is not None:
        front["design_length_m"] = physical_provenance["design_length"]["value_m"]
    front["model_length_m"] = physical_provenance["model_length_m"]
    if physical_provenance["native_boundary_geometry"]:
        geometry = physical_provenance["native_boundary_geometry"]
        front["input_output_separation_m"] = geometry["input_output_separation_m"]
    if physical_provenance["longitudinal_span"]:
        front["model_length_definition"] = physical_provenance["longitudinal_span"]["definition"]
        front["model_longitudinal_span_m"] = physical_provenance["longitudinal_span"]["value_m"]
    front.update(input_frontmatter(input_contract))
    supplied_metrics = front.pop("metrics", [])
    front.setdefault("language", ["en"])
    front.setdefault("tags", ["type:model", "science:accelerator-physics"])
    name = supplied.get("name", lock["selected"]["candidate_id"])
    front.setdefault("name", name)
    front = {
        **{
            key: front[key] for key in ("name", "design_length_m", "model_length_m") if key in front
        },
        **front,
    }
    card = output / "MODEL_CARD.md"
    card.write_text(
        "---\n"
        + yaml.safe_dump(front, sort_keys=False)
        + "---\n\n# "
        + name
        + "\n\n"
        + "\n".join(body),
        encoding="utf-8",
    )
    update = _command(
        [
            "-m",
            "element_surrogate.genesis_adapter",
            "update",
            str(card),
            str(output / "eval_bundle.json"),
            "-o",
            str(card),
        ],
        output / "model_update.log",
    )
    if update.returncode:
        raise ValueError("Genesis model-card updater failed; inspect model_update.log")
    raw_card = card.read_text(encoding="utf-8")
    (output / "MODEL_CARD.upstream.md").write_text(raw_card, encoding="utf-8")
    bundle = read_json(output / "eval_bundle.json")
    card.write_text(compose_metrics(raw_card, bundle, supplied_metrics), encoding="utf-8")
    metrics_review = metric_coverage(card.read_text(encoding="utf-8"), bundle)
    write_json(output / "model_metrics_validation.json", metrics_review)
    validation = _command(
        [
            "-m",
            "element_surrogate.genesis_adapter",
            "validate",
            str(card),
            str(output / "eval_bundle.json"),
            "--json",
        ],
        output / "model_validation.log",
    )
    structure = review_model_structure(card.read_text(encoding="utf-8"), headings)
    write_json(output / "model_structure_validation.json", structure)
    technical_contract_ready = physical_provenance["model_length_m"] is not None
    result = {
        "model_card": str(card),
        "evaluation_validation_pass": validation.returncode == 0 and metrics_review["pass"],
        "model_metrics_validation": metrics_review,
        "raw_pinned_validation_log": str(output / "model_validation.log"),
        "pinned_metric_parser_limit": (
            "readable benchmark names with spaces may produce false METRICS_MISMATCH warnings; "
            "raw findings retained and exact YAML coverage checked separately"
        ),
        "model_template_structure": structure,
        "metadata_review": metadata_review,
        "model_physical_provenance": physical_provenance,
        "selected_model_summary": model_summary,
        "model_input_contract": input_contract,
        "model_input_output_documentation_ready": technical_contract_ready,
        "model_contract_missing_metadata": []
        if technical_contract_ready
        else ["referenced native longitudinal input/output stations"],
        "model_pending_sections": pending,
        "model_section_coverage": section_coverage,
        "field": field,
        "particle": particle,
        "institutional_release_approved": False,
        "editorial_and_owner_review_required": True,
        "external_publication_performed": False,
        "full_documentation_ready": validation.returncode == 0
        and metrics_review["pass"]
        and not pending
        and structure["template_structure_pass"]
        and metadata_review["release_metadata_ready"]
        and technical_contract_ready
        and all(r["schema_pass"] and r["capabilities_confirmed"] for r in (field, particle)),
    }
    write_json(output / "card_validation.json", result)
    return {
        **{k: v for k, v in result.items() if k not in ("field", "particle")},
        "data_cards": [
            {k: v for k, v in r.items() if k != "introspection"} for r in (field, particle)
        ],
        "validation_path": str(output / "card_validation.json"),
    }
