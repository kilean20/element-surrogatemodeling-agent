"""Offline learning authority, evidence and unchanged private provenance."""

from pathlib import Path

import pytest

from element_surrogate import knowledge, learning, project
from element_surrogate.benchmark.protocol import file_hash, write_json
from tests.test_project import front


def policy(root, mode="delegated", kinds=None, replacements=False):
    p = root / "policy-request.json"
    write_json(
        p,
        {
            "schema": learning.POLICY_SCHEMA,
            "mode": mode,
            "allowed_kinds": kinds if kinds is not None else ["lesson", "method"],
            "scope": "Manufactured validated project lessons only",
            "allow_replacements": replacements,
        },
    )
    return learning.record_policy(
        root,
        p,
        file_hash(p),
        "synthetic user",
        "Delegate the scoped fixture only",
        "manufactured transcript",
    )


def lesson(root, identifier="local-fixture", replaces=None):
    metadata = front(identifier)
    if replaces:
        metadata["replaces"] = replaces
    return project.add_knowledge(root, identifier, metadata, "Manufactured local evidence")


def validation(root, entry, **changes):
    p = root / "validation.json"
    evidence = root / "probe.json"
    write_json(evidence, {"manufactured": True, "zero_field_drift_control_passed": True})
    data = {
        "schema": learning.VALIDATION_SCHEMA,
        "entry_sha256": entry["sha256"],
        "context_sha256": learning.context(root)["context_sha256"],
        "outcome": "supported",
        "scope_assessment": "Within the explicitly delegated fixture scope",
        "rationale": "Manufactured local control and derivation agree",
        "existing_guidance_comparison": "A scoped regression under the existing method",
        "conflicts": [],
        "evidence": [
            {
                "path": evidence.name,
                "sha256": file_hash(evidence),
                "supports": "Manufactured control only",
            }
        ],
    }
    data.update(changes)
    write_json(p, data)
    return p


def activate(root, entry, review, grant):
    return learning.activate(
        root,
        Path(entry["knowledge_path"]).stem,
        entry["sha256"],
        review,
        grant["policy_sha256"],
        "Use the validated scoped local lesson",
    )


def test_delegated_offline_loop_and_correct_approver(tmp_path, monkeypatch):
    import socket

    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("No network"))
    project.initialize(tmp_path, "offline")
    grant = policy(tmp_path)
    item = lesson(tmp_path)
    review = validation(tmp_path, item)
    result = activate(tmp_path, item, review, grant)
    assert result["status"] == "active" and result["approved_by"] == "agent-approved"
    restored = project.restore(tmp_path)
    assert restored["local_learning"]["mode"] == "delegated"
    entry = knowledge.find(project.entries(tmp_path), "local-fixture")
    assert entry.approved_by == "agent-approved"
    assert knowledge.search(project.entries(tmp_path), "private")
    assert not result["sharing_authorized"] and not list(
        (tmp_path / "records").rglob("*.state.json")
    )
    assert len(list((tmp_path / "records/learning_policy_history").glob("*.json"))) == 1


def test_no_delegation_is_an_advisory_not_a_global_halt(tmp_path):
    project.initialize(tmp_path, "manual")
    item = lesson(tmp_path)
    review = validation(tmp_path, item)
    result = learning.activate(
        tmp_path, "local-fixture", item["sha256"], review, "absent", "Fixture"
    )
    assert (
        result["reason"] == "no_active_local_delegation" and result["unrelated_work_may_continue"]
    )
    assert file_hash(item["knowledge_path"]) == item["sha256"]


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"outcome": "inconclusive"}, "validation_not_supported"),
        ({"outcome": "failed"}, "validation_not_supported"),
        (
            {"conflicts": ["user-confirmed reference convention"]},
            "scientific_or_user_decision_conflict",
        ),
        ({"entry_sha256": "0" * 64}, "validation_names_different_entry"),
        ({"context_sha256": "0" * 64}, "reviewed_context_changed"),
    ],
)
def test_unsupported_or_conflicting_claims_remain_proposed(tmp_path, change, reason):
    project.initialize(tmp_path, "test")
    grant = policy(tmp_path)
    item = lesson(tmp_path)
    result = activate(tmp_path, item, validation(tmp_path, item, **change), grant)
    assert result["reason"] == reason
    assert file_hash(item["knowledge_path"]) == item["sha256"]


def test_revocation_does_not_erase_previously_accepted_knowledge(tmp_path):
    project.initialize(tmp_path, "test")
    grant = policy(tmp_path)
    item = lesson(tmp_path)
    result = activate(tmp_path, item, validation(tmp_path, item), grant)
    accepted = result["sha256"]
    new = lesson(tmp_path, "another-fixture")
    review = validation(tmp_path, new)
    policy(tmp_path, mode="manual")
    assert activate(tmp_path, new, review, grant)["reason"] == "no_active_local_delegation"
    assert file_hash(item["knowledge_path"]) == accepted


@pytest.mark.parametrize("what", ["evidence", "entry", "context", "policy"])
def test_changed_inputs_cannot_reuse_validation(tmp_path, what):
    project.initialize(tmp_path, "test")
    grant = policy(tmp_path)
    item = lesson(tmp_path)
    review = validation(tmp_path, item)
    if what == "evidence":
        write_json(tmp_path / "probe.json", {"manufactured": "different"})
    elif what == "entry":
        p = Path(item["knowledge_path"])
        p.write_text(p.read_text() + "\nChanged claim\n")
    elif what == "context":
        lesson(tmp_path, "new-context")
    else:
        policy(tmp_path, kinds=["lesson"])
    result = activate(tmp_path, item, review, grant)
    assert result["status"] == "proposed_review_required"


def test_local_delegation_cannot_overwrite_user_accepted_entry(tmp_path):
    project.initialize(tmp_path, "test")
    old = lesson(tmp_path, "user-accepted")
    project.apply_knowledge_decision(
        tmp_path,
        "user-accepted",
        old["sha256"],
        "active",
        "synthetic user",
        "Accept this fixture",
        "test",
    )
    original = file_hash(old["knowledge_path"])
    grant = policy(tmp_path, replacements=True)
    new = lesson(tmp_path, "revised-fixture", "user-accepted")
    result = activate(tmp_path, new, validation(tmp_path, new), grant)
    assert result["reason"] == "replacement_of_user_accepted_knowledge_needs_user_decision"
    assert file_hash(old["knowledge_path"]) == original


def test_scoped_override_preserves_shared_original(tmp_path):
    project.initialize(tmp_path, "test")
    grant = policy(tmp_path, replacements=True)
    shared = knowledge.find(knowledge.load_entries(), "lorentz-force-tracking")
    original = file_hash(shared.path)
    new = lesson(tmp_path, "local-method-control", shared.id)
    assert activate(tmp_path, new, validation(tmp_path, new), grant)["status"] == "active"
    assert file_hash(shared.path) == original
    assert knowledge.find(project.entries(tmp_path), shared.id).status == "superseded"
    assert knowledge.find(knowledge.load_entries(), shared.id).status == "active"


def test_existing_explicit_user_acceptance_still_works_without_policy(tmp_path):
    project.initialize(tmp_path, "test")
    item = lesson(tmp_path)
    project.apply_knowledge_decision(
        tmp_path,
        "local-fixture",
        item["sha256"],
        "active",
        "synthetic user",
        "Accept this exact fixture",
        "test",
    )
    assert knowledge.find(project.entries(tmp_path), "local-fixture").approved_by == "user"


def test_rejected_scope_and_missing_evidence(tmp_path):
    project.initialize(tmp_path, "test")
    grant = policy(tmp_path, kinds=["reference"])
    item = lesson(tmp_path)
    assert (
        activate(tmp_path, item, validation(tmp_path, item), grant)["reason"]
        == "kind_outside_local_delegation"
    )
    grant = policy(tmp_path)
    with pytest.raises(ValueError, match="concrete"):
        activate(tmp_path, item, validation(tmp_path, item, evidence=[]), grant)


def test_policies_and_entries_are_project_bound(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    project.initialize(a, "a")
    project.initialize(b, "b")
    grant = policy(a)
    item = lesson(b)
    result = activate(b, item, validation(b, item), grant)
    assert result["reason"] == "no_active_local_delegation"
    assert learning.policy_summary(b)["mode"] == "manual"
