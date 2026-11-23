"""Actual bundled Genesis assets/validators, with synthetic metadata only."""

from copy import deepcopy

import pytest

pytest.importorskip("linkml")
pytest.importorskip("jsonschema")

from element_surrogate import cards
from element_surrogate.benchmark.protocol import read_json, write_json


def core_metadata(role="field"):
    # Schema-only fixture: it deliberately enables discovery alone. Real cases
    # must confirm capabilities from their own introspection and owner review.
    return {
        "rights_confirmation": {
            "status": "confirmed",
            "subject": role,
            "terms": {"name": "Restricted synthetic fixture terms"},
            "downstream_use_permitted": False,
            "actor": "Synthetic fixture owner",
            "user_statement": "Test only: restricted terms supplied for this artifact.",
            "source_reference": "manufactured test, not real institutional authority",
        },
        "frontmatter": {
            **{
                "supports_" + cap: "No"
                for cap in (
                    "accessibility",
                    "interoperability",
                    "reusability",
                    "governed_use",
                    "ai_usability",
                )
            },
            "supports_discoverability": "Yes",
            "discoverability": {
                "identification": {
                    "name": "Synthetic fixture",
                    "project": "schema-test",
                    "version": "1.0",
                    "primary_id": {"type": "local", "value": "synthetic-schema-fixture"},
                },
                "datacard": {
                    "id": {"type": "local", "value": "schema-fixture"},
                    "created_by": [
                        {
                            "contribution_date": "2026-11-01",
                            "creator": {
                                "software": {
                                    "name": "Synthetic test fixture",
                                    "version": "1.0",
                                    "identifier": {"type": "local", "value": "schema-test-tool"},
                                    "relationship": "used_to_create",
                                }
                            },
                        }
                    ],
                    "change_log": [
                        {
                            "change_date": "2026-11-01",
                            "datacard_version": "1.2",
                            "summary": "Synthetic schema test",
                        }
                    ],
                },
                "dataset_description": {
                    "science_domain": "Physics",
                    "dataset_summary": "Manufactured test fixture only",
                    "keywords": ["synthetic"],
                },
                "contact": {
                    "person": {
                        "given_name": "Synthetic",
                        "family_name": "Fixture",
                        "email": "fixture@example.org",
                    }
                },
                "authors": [{"organization": {"name": "Synthetic fixture authors"}}],
                "sponsor_organizations": [{"name": "Synthetic fixture sponsor"}],
                "research_organizations": [{"name": "Synthetic fixture research"}],
                "sensitivity": {
                    "overall_sensitivity": "Public",
                    "source_marking_string": "Synthetic test only",
                    "source_marking_scheme": "None",
                    "classified_status": "No",
                    "cui_status": "No",
                    "ucni_status": "No",
                },
                "workflow": {"state": "Review", "is_intermediate": "No"},
                "dataset_publisher": {"name": "Synthetic fixture publisher"},
            },
        },
        "body": "Synthetic metadata fixture; no institutional facts or approval.",
        "capability_confirmation": {
            "actor": "synthetic test",
            "user_statement": "Test fixture only",
            "source_reference": "test fixture",
        },
    }


def test_bundled_assets_and_guidance_no_host_dependency():
    record = cards.verify_assets()
    assert record["commit"] == "b00054a6375c18463de751f2eef8c4d4bf8643ee"
    assert "datacard-generator/SKILL.md" in cards.guidance()["assets"]
    assert "introspect" in cards.guidance("datacard-generator/SKILL.md", 1, 80)["text"]
    with pytest.raises(ValueError, match="bundled index"):
        cards.guidance("../../outside")


def test_dataset_introspection_precedes_capability_confirmation(tmp_path):
    dataset = tmp_path / "data"
    dataset.mkdir()
    write_json(dataset / "sample.json", {"synthetic": True})
    record = cards.introspect_dataset(dataset, tmp_path / "introspection.json")
    assert record["capability_confirmation_required"]
    assert read_json(record["introspection_path"]) == record["introspection"]


def test_real_linkml_schema_pass_and_missing_facts_fail(tmp_path):
    dataset, output = tmp_path / "dataset", tmp_path / "cards"
    dataset.mkdir()
    output.mkdir()
    write_json(dataset / "sample.json", {"synthetic": True})
    good = cards._data_card("field", dataset, core_metadata(), output, "Synthetic provenance")
    assert good["schema_pass"], (output / "field_validation.log").read_text(encoding="utf-8")
    assert good["capabilities_confirmed"]
    incomplete = deepcopy(core_metadata("particle"))
    incomplete["frontmatter"]["discoverability"].pop("sensitivity")
    incomplete.pop("capability_confirmation")
    bad = cards._data_card("particle", dataset, incomplete, output, "Synthetic provenance")
    assert not bad["schema_pass"] and not bad["capabilities_confirmed"]
    assert "sensitivity" in bad["metadata_missing"]


def test_model_renderer_extracts_saved_metrics_and_validator_detects_edit(tmp_path):
    # Use the generic algebraic study here to isolate Genesis rendering. The
    # separate physical-pipeline acceptance binds actual generation/transport.
    from element_surrogate.benchmark.study import (
        create_study,
        evaluate_test,
        lock_selection,
        run_study,
    )
    from tests.benchmark_fixture import approval, make_case

    path, protocol, approvals = make_case(tmp_path / "case", configurations=1)
    study = create_study(path, tmp_path / "study", approvals)["study_path"]
    run_study(study)
    lock_selection(study)
    evaluate_test(study, [approval(tmp_path / "case", protocol, "release")])
    output = tmp_path / "cards"
    output.mkdir()
    _, _, _, metrics, _ = cards._bundle(study, output)
    bundle = read_json(output / "eval_bundle.json")
    assert bundle["results"][0]["value"] == metrics["rmse"][0]
    card = output / "MODEL_CARD.md"
    card.write_text(
        "---\nmetrics: []\n---\n\n# Synthetic fixture\n\n"
        + "\n".join(
            "## " + h + "\n\nFixture only\n"
            for h in (
                "Evaluation data",
                "Evaluation Procedure",
                "Uncertainty Quantification",
                "Evaluation results",
            )
        ),
        encoding="utf-8",
    )
    command = ["-m", "element_surrogate.genesis_adapter"]
    assert (
        cards._command(
            [*command, "update", str(card), str(output / "eval_bundle.json"), "-o", str(card)],
            output / "update.log",
        ).returncode
        == 0
    )
    assert (
        cards._command(
            [*command, "validate", str(card), str(output / "eval_bundle.json")],
            output / "validate.log",
        ).returncode
        == 0
    )
    # Check the raw pinned validator too, without the local adapter's imports.
    raw = cards._command(
        [
            str(cards.VENDOR / "card-eval-updater/scripts/validate_card_eval.py"),
            str(card),
            str(output / "eval_bundle.json"),
            "--json",
        ],
        output / "stock_validate.log",
    )
    assert raw.returncode == 0, raw.stdout + raw.stderr
    card.write_text(
        card.read_text(encoding="utf-8").replace(
            "### Automated benchmark results", "### Altered results", 1
        ),
        encoding="utf-8",
    )
    assert (
        cards._command(
            [*command, "validate", str(card), str(output / "eval_bundle.json")],
            output / "altered.log",
        ).returncode
        != 0
    )
