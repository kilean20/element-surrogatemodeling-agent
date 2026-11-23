import pytest

from element_surrogate import knowledge, project
from element_surrogate.benchmark.protocol import read_json, write_json
from element_surrogate.decisions import list_decisions, record_decision


def front(identifier):
    return {
        "id": identifier,
        "title": "Private test lesson",
        "kind": "lesson",
        "status": "proposed",
        "date": "2026-11-01",
        "summary": "Test only",
        "keywords": ["private"],
        "evidence": [{"source": "test receipt", "supports": "fixture only", "verified": True}],
    }


def test_private_project_lifecycle_and_isolation(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    first = project.initialize(a, "a")
    assert project.initialize(a, "a") == first
    project.initialize(b, "b")
    assert project.restore(a)["private_knowledge"] == []
    proposed = project.add_knowledge(a, "private-test", front("private-test"), "Private content")
    assert not knowledge.search(project.entries(a), "private")
    assert knowledge.search(project.entries(a), "private", include_proposed=True)
    with pytest.raises(ValueError, match="changed since review"):
        project.apply_knowledge_decision(
            a, "private-test", "a" * 64, "active", "test user", "Approve", "test"
        )
    approved = project.apply_knowledge_decision(
        a,
        "private-test",
        proposed["sha256"],
        "active",
        "test user",
        "Approve fixture only",
        "test transcript",
    )
    assert not approved["shared_knowledge_modified"]
    assert knowledge.search(project.entries(a), "private")
    assert not knowledge.search(project.entries(b), "private")
    assert len(project.restore(a)["private_knowledge"]) == 1
    project.apply_knowledge_decision(
        a,
        "private-test",
        approved["sha256"],
        "retired",
        "test user",
        "Retire fixture",
        "test transcript",
    )
    assert not knowledge.search(project.entries(a), "private")
    assert knowledge.find(project.entries(a), "private-test").status == "retired"


def test_invalid_proposal_is_not_published(tmp_path):
    project.initialize(tmp_path, "test")
    invalid = front("private-test")
    invalid["evidence"] = []
    with pytest.raises(ValueError, match="needs evidence"):
        project.add_knowledge(tmp_path, "private-test", invalid, "Test")
    assert project.restore(tmp_path)["private_knowledge"] == []


def test_approved_private_replacement_updates_search_and_restored_view(tmp_path):
    project.initialize(tmp_path, "private-chain")
    for identifier, replaces in [("private-first", None), ("private-second", "private-first")]:
        metadata = front(identifier)
        if replaces:
            metadata["replaces"] = replaces
        proposed = project.add_knowledge(
            tmp_path, identifier, metadata, "Private replacement fixture"
        )
        project.apply_knowledge_decision(
            tmp_path,
            identifier,
            proposed["sha256"],
            "active",
            "synthetic user",
            "Approve this fixture only",
            "test transcript",
        )
    results = knowledge.search(project.entries(tmp_path), "private")
    assert [row["id"] for row in results] == ["private-second"]
    restored = {row["id"]: row["status"] for row in project.restore(tmp_path)["private_knowledge"]}
    assert restored == {"private-first": "superseded", "private-second": "active"}


def test_replaced_shared_lesson_remains_readable_and_is_not_automatically_reactivated(tmp_path):
    project.initialize(tmp_path, "test")
    old = knowledge.find(knowledge.load_entries(), "lorentz-force-tracking")
    metadata = front("replacement-test")
    metadata["replaces"] = old.id
    proposed = project.add_knowledge(tmp_path, "replacement-test", metadata, "Fixture replacement")
    active = project.apply_knowledge_decision(
        tmp_path,
        "replacement-test",
        proposed["sha256"],
        "active",
        "test",
        "Approve test replacement",
        "test",
    )
    assert knowledge.find(project.entries(tmp_path), old.id).status == "superseded"
    project.apply_knowledge_decision(
        tmp_path,
        "replacement-test",
        active["sha256"],
        "retired",
        "test",
        "Retire fixture replacement",
        "test",
    )
    assert knowledge.find(project.entries(tmp_path), old.id).status == "superseded"
    assert knowledge.find(knowledge.load_entries(), old.id).status == "active"


def test_explicit_decision_supersession(tmp_path):
    def record(statement, supersedes=()):
        return record_decision(
            tmp_path,
            actor="test",
            user_statement=statement,
            scope="limit",
            source_reference="test transcript",
            supersedes=supersedes,
        )

    a = record("first")
    b = record("second")
    assert list_decisions(tmp_path)["multiple_active_scopes"]
    c = record("replace both", [a["decision_id"], b["decision_id"]])
    assert [v["id"] for v in list_decisions(tmp_path)["active_events"]] == [c["decision_id"]]
    damaged = read_json(a["decision_path"])
    damaged["supersedes"] = [c["decision_id"]]
    write_json(a["decision_path"], damaged)
    with pytest.raises(ValueError, match="cycle"):
        list_decisions(tmp_path)


def test_dangling_decision_refused(tmp_path):
    event = record_decision(
        tmp_path, actor="test", user_statement="first", scope="test", source_reference="test"
    )
    damaged = read_json(event["decision_path"])
    damaged["supersedes"] = ["missing"]
    write_json(event["decision_path"], damaged)
    with pytest.raises(ValueError, match="unknown"):
        list_decisions(tmp_path)
