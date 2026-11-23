"""Installed examples and preflight remain useful without source checkout."""

import json
import subprocess
import sys

from element_surrogate.card_metadata import inspect_request


def test_cli_examples_are_valid_pending_drafts_and_utf8_files(tmp_path):
    output = tmp_path / "metadata.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "element_surrogate.cli",
            "request-example",
            "cards",
            "--output",
            str(output),
        ],
        capture_output=True,
        check=True,
    )
    value = json.loads(result.stdout)
    assert json.loads(output.read_text(encoding="utf-8")) == value
    review = inspect_request(value)
    assert review["valid"] and not review["release_metadata_ready"]
    assert not output.read_bytes().startswith(b"\xef\xbb\xbf")


def test_upstream_draft_classification_keeps_raw_failure():
    from element_surrogate.card_metadata import classify_upstream_data_validation

    log = "[ERROR] 'dataset_publisher' is a required property in /discoverability"
    draft = {"discoverability": {"release_status": "Draft"}}
    result = classify_upstream_data_validation(draft, log, False)
    assert result["known_draft_publisher_precondition_only"]
    assert not result["upstream_schema_pass"] and not result["upstream_schema_modified"]
    extra = log + "\n[ERROR] Other missing fact"
    assert not classify_upstream_data_validation(draft, extra, False)[
        "known_draft_publisher_precondition_only"
    ]
    draft["discoverability"]["release_status"] = "Approved"
    assert not classify_upstream_data_validation(draft, log, False)[
        "known_draft_publisher_precondition_only"
    ]


def test_blank_confirmation_cannot_count_as_rights():
    from tests.test_card_metadata_contract import supplied

    metadata = supplied()
    metadata["rights_confirmation"]["actor"] = "  "
    assert not inspect_request({"datasets": {"particle": metadata}})["valid"]
