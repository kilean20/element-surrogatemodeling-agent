"""Full physical label -> selected neural map -> three Genesis cards control."""

from pathlib import Path

import pytest

pytest.importorskip("linkml")
pytest.importorskip("torch")

from element_surrogate.benchmark.protocol import read_json, write_json
from element_surrogate.benchmark.study import create_study, evaluate_test, lock_selection, run_study
from element_surrogate.cards import VENDOR, _normalize, prepare
from element_surrogate.data import generate
from element_surrogate.modeling import prepare_generated_study
from element_surrogate.workflow import load_transport
from tests.test_cards import core_metadata
from tests.test_physical_workflow import study_request
from tests.test_workflow_tools import file_case
from tests.workflow_fixture import approvals


@pytest.mark.parametrize("provided", [False, True])
def test_complete_cards_for_exact_generation_and_pending_metadata(tmp_path, provided):
    from functools import partial

    prepare_cards = partial(prepare, supplied_labels=provided)
    label_dir = tmp_path / "data"
    path, _, plan = file_case(tmp_path)
    transport, _ = load_transport(path)
    generate(
        transport,
        read_json(tmp_path / "qualification.json"),
        plan,
        tmp_path / "data",
        approvals(tmp_path / "gen-approvals", plan),
    )
    request = study_request(4)
    request["candidates"] = request["candidates"][:1]
    request["candidates"][0]["max_steps"] = 2
    request["candidates"][0]["seeds"] = [3]
    protocol_path = prepare_generated_study(tmp_path / "data", tmp_path / "benchmark", request)[
        "protocol_path"
    ]
    protocol = read_json(protocol_path)
    if provided:
        import numpy as np

        from element_surrogate.benchmark.protocol import file_hash
        from element_surrogate.benchmark.scoring import load_cohort
        from element_surrogate.supplied_data import prepare as import_labels

        label_dir = tmp_path / "supplied_labels"
        label_dir.mkdir()
        for split, item in protocol["dataset"]["splits"].items():
            arrays = load_cohort(protocol, Path(protocol_path).parent, split)
            target = label_dir / (split + ".npz")
            np.savez_compressed(target, **arrays)
            item["path"], item["sha256"] = str(target), file_hash(target)
        protocol["dataset"].pop("generation_record_path")
        declaration = {
            "schema": "element_surrogate.supplied_neural_study.v1",
            "protocol": protocol,
            "source_contract": plan,
            "physics_review": {"topics": protocol["physics"]["modeling_review"]["topics"]},
            "label_provenance": {
                "description": "Manufactured supplied-label snapshot",
                "qualification_status": "source_declared",
                "split_history": {
                    "train": "previously_used",
                    "validation": "previously_used",
                    "test": "previously_used",
                },
            },
        }
        write_json(tmp_path / "provided_request.json", declaration)
        protocol_path = import_labels(
            tmp_path / "provided_request.json", tmp_path / "provided_protocol"
        )["protocol_path"]
        protocol = read_json(protocol_path)
    study = create_study(
        protocol_path, tmp_path / "study", approvals(tmp_path / "study-approvals", protocol)
    )["study_path"]
    run_study(study)
    lock_selection(study)
    evaluate_test(study, approvals(tmp_path / "release-approval", protocol, ("release",)))
    headings = read_json(VENDOR / "model_card_headings.json")["headings"]
    metadata = {
        "model": {
            "name": "Synthetic native map",
            "frontmatter": {"metrics": ["training/loss"]},
            "rights_confirmation": core_metadata("model")["rights_confirmation"],
            "physical_provenance": {
                "longitudinal_span": {
                    "input_m": float(transport.entry.origin_m[2]),
                    "output_m": float(transport.exit.origin_m[2]),
                    "definition": "Manufactured straight native boundary stations along source z",
                    "source_reference": "synthetic layout specification",
                },
                "design_length": {
                    "value_m": 1.25,
                    "definition": "Manufactured nominal native-plane path",
                    "source_reference": "synthetic layout specification",
                },
                "solver_mesh": {
                    "description": "Manufactured solver metadata: local maximum element 0.7 mm",
                    "source_reference": "synthetic solver log",
                },
                "field_calculation_currents": [
                    {
                        "current_a": 7.25,
                        "description": "independent horizontal-coil calculation",
                        "source_reference": "synthetic current record H",
                    },
                    {
                        "current_a": -2.75,
                        "description": "independent vertical-coil calculation",
                        "source_reference": "synthetic current record V",
                    },
                ],
            },
            "sections": {
                _normalize(h): "Synthetic fixture documentation; test only." for h in headings
            },
        },
        "datasets": {"field": core_metadata("field"), "particle": core_metadata("particle")},
    }
    for kind in ("field", "particle"):
        metadata["datasets"][kind]["frontmatter"]["discoverability"]["identification"]["name"] = (
            "Synthetic " + kind
        )
    metadata["model"]["sections"]["Model Description. "] = "Supplied title-case scientific facts."
    metadata["model"]["sections"].pop("model description")
    write_json(tmp_path / "metadata.json", metadata)
    result = prepare_cards(
        study,
        path,
        label_dir,
        tmp_path / "map",
        tmp_path / "metadata.json",
        tmp_path / "cards",
    )
    assert result["evaluation_validation_pass"]
    from element_surrogate.card_frontmatter import split_card

    front, _ = split_card(Path(result["model_card"]).read_text(encoding="utf-8"))
    assert front["metrics"].count("training/loss") == 1
    assert front["design_length_m"] == 1.25
    assert front["model_length_m"] == pytest.approx(
        __import__("numpy").linalg.norm(transport.exit.origin_m - transport.entry.origin_m)
    )
    assert front["beam_parameter_dimensions"] == 4
    assert front["model_input_dimensions"] == 6
    assert [row["unit"] for row in front["beam_parameters"]] == ["m", "rad", "m", "rad"]
    assert [row["unit"] for row in front["magnet_parameters"]] == ["T*m", "1"]
    assert [row["rough_valid_range"] for row in front["beam_parameters"]] == plan["ranges"]
    assert read_json(tmp_path / "cards/model_input_contract.json")["interface"] == "native"
    card_text = Path(result["model_card"]).read_text(encoding="utf-8")
    assert "Supplied title-case scientific facts." in card_text
    assert "Selected artifact details" in card_text
    if provided:
        assert (
            "Previously used or unknown-history test rows are not a new untouched test" in card_text
        )
        assert read_json(tmp_path / "cards/traceability.json")["label_origin"] == "supplied"
        assert read_json(tmp_path / "cards/model_input_contract.json")["generation_record"] is None
    assert result["model_section_coverage"]["unused_sections"] == []
    assert "Design geometry and source-field calculation" in card_text
    assert "independent horizontal-coil calculation" in card_text
    assert "-2.75 A" in card_text
    assert "Internal field-solver mesh" in card_text
    physical = read_json(tmp_path / "cards/model_physical_provenance.json")
    assert physical["export_grid"]["extent_m"]["z"] == [
        float(transport.grid.z_m[0]),
        float(transport.grid.z_m[-1]),
    ]
    assert read_json(tmp_path / "cards/traceability.json")["model_physical_provenance"] == physical
    assert len(front["metrics"]) == len(set(front["metrics"]))
    assert result["model_metrics_validation"]["pass"]
    assert result["full_documentation_ready"], result
    assert result["model_input_output_documentation_ready"]
    assert not result["institutional_release_approved"]
    assert all(card["schema_pass"] for card in result["data_cards"])
    # Document the actual exported public API, not assumed example units.
    from element_surrogate.export import export_selected
    from element_surrogate.public_interface import export_interface

    export_selected(
        study,
        tmp_path / "native_export",
        approvals(tmp_path / "export-approval", protocol, ("release",)),
    )
    public_request = {
        "state_units": ["mm", "mrad", "mm", "mrad"],
        "fixed_controls": {"field_scale": plan["settings"][0]["field_scale"]},
        "control_bounds": {
            row["name"]: row["rough_valid_range"] for row in front["magnet_parameters"]
        },
    }
    write_json(tmp_path / "public_request.json", public_request)
    export_interface(
        tmp_path / "native_export/contract.json",
        tmp_path / "public_request.json",
        tmp_path / "public_interface.json",
    )
    metadata["model"]["public_interface"] = {
        "native_contract_path": str(tmp_path / "native_export/contract.json"),
        "interface_path": str(tmp_path / "public_interface.json"),
    }
    write_json(tmp_path / "metadata.json", metadata)
    public_cards = prepare_cards(
        study,
        path,
        label_dir,
        tmp_path / "map",
        tmp_path / "metadata.json",
        tmp_path / "public_cards",
    )
    public_front, _ = split_card(Path(public_cards["model_card"]).read_text(encoding="utf-8"))
    assert public_cards["evaluation_validation_pass"]
    assert public_front["model_input_dimensions"] == 5
    assert [row["unit"] for row in public_front["beam_parameters"]] == ["mm", "mrad", "mm", "mrad"]
    assert public_front["beam_parameters"][0]["rough_valid_range"] == pytest.approx(
        [value * 1000 for value in plan["ranges"][0]]
    )
    assert len(public_front["magnet_parameters"]) == 1
    assert public_front["magnet_parameters"][0]["unit"] == "T*m"
    assert public_cards["model_input_contract"]["interface"] == "public"
    metadata["model"].pop("public_interface")
    metadata["datasets"]["field"]["frontmatter"]["discoverability"].pop("sensitivity")
    metadata["model"]["sections"].pop("developed by")
    metadata["model"].pop("physical_provenance")
    write_json(tmp_path / "metadata.json", metadata)
    draft = prepare_cards(
        study,
        path,
        label_dir,
        tmp_path / "map",
        tmp_path / "metadata.json",
        tmp_path / "draft",
    )
    assert not draft["full_documentation_ready"]
    assert not draft["model_input_output_documentation_ready"]
    assert draft["model_contract_missing_metadata"]
    draft_text = Path(draft["model_card"]).read_text(encoding="utf-8")
    assert "| Design length | Not recorded |" in draft_text
    assert "| Field-calculation input current(s) | Not recorded |" in draft_text
    assert "Export spacing x / y / z" in draft_text
    assert "developed by" in draft["model_pending_sections"]
