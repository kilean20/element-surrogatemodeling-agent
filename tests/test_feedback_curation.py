"""General scientific feedback stays useful; obvious private text never uploads."""

import json
from pathlib import Path

import pytest

from element_surrogate import feedback
from element_surrogate.benchmark.protocol import file_hash, read_json, write_json
from element_surrogate.feedback_privacy import review_content
from tests.test_population_runtime_feedback import packet


def content(text):
    return {
        "category": "physics",
        "summary": text,
        "user_feedback": "none supplied",
        "findings": "Manufactured observation; not a private machine case",
        "evidence": ["Synthetic minimal reproduction"],
        "suggested_change": "Review a reusable physics/workflow improvement",
        "limitations": "No physical machine claim; human semantic review still required",
    }


@pytest.mark.parametrize(
    "text",
    [
        "Distinguish canonical momenta from slopes before checking J.T @ S @ J = S.",
        "A single excitation field and unknown aperture permit "
        "a qualified numerical research scope.",
        "Use B2/Brho in m^-2; field shape changes need independent evidence.",
        "Compare train-only scaling, shared loss and inference cost for symmetry ablations.",
        "Memory-limited CPU-only execution and unavailable network need a resumable workflow.",
        "Public reference: https://doi.org/10.1234/example and "
        "https://docs.python.org/3/library/os.html",
        "Connection details are unavailable; access_token=[REDACTED].",
        'Public example: {"password": "[REDACTED]"}',
    ],
)
def test_general_physics_and_coarse_constraints_are_not_falsely_blocked(text):
    result = review_content(content(text))
    assert result["pattern_check_passed"] and result["human_review_required"]
    assert "not a confidentiality guarantee" in result["scope"]


@pytest.mark.parametrize(
    "private_text",
    [
        "Contact scientist@example.invalid",
        r"Loaded C:\Users\example\private\field.h5",
        "Loaded /home/example/private/field.h5",
        "Connect to 192.0.2.10",
        "http://localhost:8999/?token=manufactured-secret",
        "https://example.invalid/?access_token=manufactured-secret",
        "https://user:manufactured-secret@example.invalid/upload",
        "hostname: example-device",
        "access_token=manufactured-secret",
        "-----BEGIN PRIVATE KEY----- invented",
        "sk-" + "x" * 32,
        "Bearer " + "x" * 32,
        '{"password": "manufactured-secret"}',
        "access_token=[REDACTED]still-a-value",
    ],
)
def test_private_patterns_are_flagged_without_echoing_values(private_text):
    result = review_content(content(private_text))
    assert not result["pattern_check_passed"] and result["findings"]
    assert private_text not in json.dumps(result)
    assert all(row["field"] == "content.summary" for row in result["findings"])


def test_long_plain_text_does_not_trigger_unbounded_email_prefix_backtracking():
    result = review_content(content("x" * (96 * 1024)))
    assert result["pattern_check_passed"]


def private_packet(tmp_path):
    project, _ = packet(tmp_path)
    request = read_json(tmp_path / "request.json")
    request["content"] = content("hostname: example-device")
    write_json(tmp_path / "private_request.json", request)
    result = feedback.prepare(project, "private-draft", tmp_path / "private_request.json")
    return project, result


def test_flagged_draft_is_local_and_non_echoing_while_decline_and_revoke_work(
    tmp_path, monkeypatch
):
    project, result = private_packet(tmp_path)
    assert result["review_payload"] is None and not result["uploaded"]
    assert result["sharing_status"] == "local_only_needs_generalization"
    assert result["warnings"] and not result["warnings"][0]["blocking"]
    assert "example-device" not in json.dumps(result)
    monkeypatch.setattr(feedback, "_http", lambda *a, **k: pytest.fail("must not connect"))
    args = (project, "private-draft", result["packet_sha256"])
    with pytest.raises(ValueError, match="generalization"):
        feedback.consent(*args, "approve", "fixture", "synthetic approval", "manufactured test")
    for decision in ("decline", "revoke"):
        assert (
            feedback.consent(*args, decision, "fixture", "synthetic choice", "test")["decision"]
            == decision
        )


def test_sender_rechecks_legacy_approved_content_before_any_network(tmp_path, monkeypatch):
    project, result = private_packet(tmp_path)
    saved = read_json(result["packet_path"])
    # Manufactured legacy client attestation: never an actual-user approval.
    write_json(
        Path(result["packet_path"]).with_name("private-draft.state.json"),
        {
            "consent": {
                "decision": "approve",
                "packet_sha256": result["packet_sha256"],
                **{k: saved[k] for k in ("recipient", "destination", "purpose")},
            }
        },
    )
    monkeypatch.setattr(feedback, "_http", lambda *a, **k: pytest.fail("must not connect"))
    monkeypatch.setenv("ESA_TEST_TOKEN", "manufactured-token")
    with pytest.raises(ValueError, match="generalization"):
        feedback.send_drive(project, "private-draft", result["packet_sha256"], "ESA_TEST_TOKEN")


def test_remote_filename_does_not_disclose_local_feedback_or_machine_label(tmp_path, monkeypatch):
    project, _ = packet(tmp_path)
    result = feedback.prepare(project, "private-machine-label", tmp_path / "request.json")
    feedback.consent(
        project,
        "private-machine-label",
        result["packet_sha256"],
        "approve",
        "fixture",
        "Synthetic exact-packet choice",
        "manufactured test",
    )
    monkeypatch.setenv("ESA_TEST_TOKEN", "manufactured-token")
    captured = []

    def http(method, url, token, body=None, content_type=None):
        if method == "GET":
            return {
                "id": "synthetic_folder",
                "mimeType": "application/vnd.google-apps.folder",
                "capabilities": {"canAddChildren": True},
            }
        captured.append(body)
        return {"id": "synthetic_remote"}

    monkeypatch.setattr(feedback, "_http", http)
    receipt = feedback.send_drive(
        project, "private-machine-label", result["packet_sha256"], "ESA_TEST_TOKEN"
    )
    assert receipt["uploaded"]
    assert b"private-machine-label" not in captured[0]
    assert ("esa-feedback-" + result["packet_sha256"][:24]).encode() in captured[0]


def test_installed_example_has_interim_destination_but_creates_no_approval(tmp_path, monkeypatch):
    import element_surrogate

    example = Path(element_surrogate.__file__).parent / "examples/feedback_request.json"
    request = read_json(example)
    assert request["destination"] == {
        "provider": "google_drive",
        "folder_id": "1YAAzHcKzOFOmIJxirR2ioAxZPObDM0mI",
    }
    assert "Kilean Hwang" in request["recipient"]
    project, _ = packet(tmp_path)
    request["content"] = content("Canonical versus slope charts need explicit definitions.")
    write_json(tmp_path / "interim.json", request)
    result = feedback.prepare(project, "interim", tmp_path / "interim.json")
    assert not result["uploaded"] and result["sharing_status"] == "local_only_no_consent"
    monkeypatch.setattr(feedback, "_http", lambda *a, **k: pytest.fail("must not connect"))
    with pytest.raises(ValueError, match="approval"):
        feedback.send_drive(project, "interim", result["packet_sha256"], "ESA_TEST_TOKEN")
    assert file_hash(result["packet_path"]) == result["packet_sha256"]
