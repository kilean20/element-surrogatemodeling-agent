"""Managed updates preserve local science and require current compatibility evidence."""

import json
import shutil
import tomllib
from pathlib import Path

import pytest

from element_surrogate import cli, knowledge, project, updates
from element_surrogate.benchmark.protocol import file_hash, read_json, write_json
from tests.test_project import front

REL = ".agents/skills/field-surrogate-workflow/references/coordinates_and_scaling.md"
SRC = "field-surrogate-workflow/references/coordinates_and_scaling.md"


@pytest.fixture
def case(tmp_path, monkeypatch):
    source = tmp_path / "incoming-skills"
    shutil.copytree(Path(__file__).resolve().parents[1] / "skills", source)
    monkeypatch.setattr(cli, "skills_directory", lambda: source)
    root = tmp_path / "project"
    cli.install_project(root, "fixture", "both")
    (root / "inputs/private-source.txt").write_text("Manufactured private input")
    (root / "reports/accepted-notes.md").write_text("Preserve manufactured project decisions")
    return root, source


def change_source(source):
    p = source / SRC
    p.write_text(
        p.read_text(encoding="utf-8") + "\nManufactured vendor clarification.\n", encoding="utf-8"
    )


def review(root, staged, **overrides):
    plan = read_json(staged["plan_path"])
    evidence = root / "checks/probe.json"
    write_json(evidence, {"manufactured": True, "interface_control": "passed"})
    data = {
        "schema": "element_surrogate.client_update_validation.v1",
        "plan_sha256": staged["plan_sha256"],
        "context_sha256": plan["context_sha256"],
        "checkpoint": "idle",
        "rationale": "Manufactured update control only",
        "checks": [
            {
                "name": name,
                "outcome": "passed",
                "rationale": "Unit-test fixture only",
                "evidence": [{"path": "checks/probe.json", "sha256": file_hash(evidence)}],
            }
            for name in [
                "guidance_and_local_knowledge",
                "runtime_and_interfaces",
                "affected_project_behavior",
            ]
        ],
    }
    data.update(overrides)
    p = root / "checks/update-review.json"
    write_json(p, data)
    return p


def apply(root, staged, **changes):
    return updates.apply(
        root, "update-fixture", staged["plan_sha256"], review(root, staged, **changes)
    )


def test_staged_update_keeps_private_knowledge_and_can_rollback(case):
    root, source = case
    item = project.add_knowledge(root, "accepted-local", front("accepted-local"), "Local finding")
    project.apply_knowledge_decision(
        root, "accepted-local", item["sha256"], "active", "test", "Accept fixture", "test"
    )
    private = {
        p.relative_to(root): file_hash(p)
        for p in root.rglob("*")
        if p.is_file()
        and p.relative_to(root).as_posix().startswith(("records/knowledge/", "inputs/", "reports/"))
    }
    before = file_hash(root / REL)
    change_source(source)
    staged = updates.stage(root, "update-fixture")
    assert staged["status"] == "staged" and file_hash(root / REL) == before
    assert not staged["conflicts"]
    assert apply(root, staged)["status"] == "applied"
    assert file_hash(root / REL) != before
    assert all(file_hash(root / name) == digest for name, digest in private.items())
    assert updates.rollback(root, "update-fixture")["status"] == "rolled_back"
    assert file_hash(root / REL) == before
    assert all(file_hash(root / name) == digest for name, digest in private.items())


def test_upstream_unchanged_local_edit_is_kept(case):
    root, source = case
    p = root / REL
    p.write_text(p.read_text(encoding="utf-8") + "\nLocal addition.\n", encoding="utf-8")
    before = file_hash(p)
    # Update a different vendor file.
    other = source / "surrogate-review/ATTRIBUTION.md"
    other.write_text(other.read_text() + "\nManufactured vendor note.\n")
    staged = updates.stage(root, "update-fixture")
    assert not staged["conflicts"]
    assert apply(root, staged)["status"] == "applied"
    assert file_hash(p) == before


def test_both_edited_conflict_requires_staged_resolution(case):
    root, source = case
    p = root / REL
    p.write_text(p.read_text(encoding="utf-8") + "\nOwner local change.\n", encoding="utf-8")
    before = file_hash(p)
    change_source(source)
    staged = updates.stage(root, "update-fixture")
    assert REL in staged["conflicts"]
    assert apply(root, staged)["reason"] == "unresolved_local_file_conflicts"
    merged = root / "reports/merged-guide.md"
    merged.write_text(
        p.read_text(encoding="utf-8") + "\nReconciled vendor clarification.\n", encoding="utf-8"
    )
    resolved = updates.resolve(
        root,
        "update-fixture",
        staged["plan_sha256"],
        REL,
        "merged",
        "Preserve both fixture changes",
        merged,
    )
    assert file_hash(p) == before
    assert (
        updates.apply(root, "update-fixture", staged["plan_sha256"], review(root, staged))["reason"]
        == "plan_changed"
    )
    staged.update(resolved)
    assert apply(root, staged)["status"] == "applied"
    assert file_hash(p) == file_hash(merged)


@pytest.mark.parametrize("what", ["private_knowledge", "active_file", "candidate", "manifest"])
def test_changed_context_cannot_reuse_update_review(case, what):
    root, source = case
    change_source(source)
    staged = updates.stage(root, "update-fixture")
    evidence = review(root, staged)
    if what == "private_knowledge":
        project.add_knowledge(root, "new-finding", front("new-finding"), "Later evidence")
    elif what == "active_file":
        (root / REL).write_text("Later owner edit")
    elif what == "candidate":
        change_source(source)
    else:
        p = root / "records/client_installation.json"
        data = read_json(p)
        data["version"] = "later-owner-change"
        write_json(p, data)
    assert (
        updates.apply(root, "update-fixture", staged["plan_sha256"], evidence)["status"]
        == "update_review_required"
    )


@pytest.mark.parametrize(
    "changes", [{"checkpoint": "running"}, {"checks": []}, {"context_sha256": "0" * 64}]
)
def test_no_adoption_without_matching_compatibility_review(case, changes):
    root, source = case
    before = file_hash(root / REL)
    change_source(source)
    staged = updates.stage(root, "update-fixture")
    assert apply(root, staged, **changes)["status"] == "update_review_required"
    assert file_hash(root / REL) == before


def test_changed_compatibility_evidence_is_not_accepted(case):
    root, source = case
    change_source(source)
    staged = updates.stage(root, "update-fixture")
    path = review(root, staged)
    write_json(root / "checks/probe.json", {"changed": True})
    assert (
        updates.apply(root, "update-fixture", staged["plan_sha256"], path)["reason"]
        == "compatibility_evidence_changed"
    )


def test_later_user_knowledge_or_edits_are_not_erased_by_rollback(case):
    root, source = case
    change_source(source)
    staged = updates.stage(root, "update-fixture")
    assert apply(root, staged)["status"] == "applied"
    p = root / REL
    p.write_text("Later deliberate local edit")
    assert updates.rollback(root, "update-fixture")["status"] == "update_review_required"
    assert p.read_text() == "Later deliberate local edit"


def test_interrupted_apply_can_recover_original_files(case, monkeypatch):
    root, source = case
    before = file_hash(root / REL)
    change_source(source)
    staged = updates.stage(root, "update-fixture")
    real = updates._write_selected

    def interrupted(*args):
        real(*args)
        raise SystemExit("Manufactured interruption after one complete file")

    monkeypatch.setattr(updates, "_write_selected", interrupted)
    with pytest.raises(SystemExit):
        apply(root, staged)
    monkeypatch.setattr(updates, "_write_selected", real)
    assert updates.rollback(root, "update-fixture")["status"] == "rolled_back"
    assert file_hash(root / REL) == before


def test_ordinary_write_failure_restores_original_files(case, monkeypatch):
    root, source = case
    before = file_hash(root / REL)
    change_source(source)
    staged = updates.stage(root, "update-fixture")
    real, failed = updates._write_selected, []

    def once(*args):
        real(*args)
        if not failed:
            failed.append(True)
            raise OSError("Manufactured write failure")

    monkeypatch.setattr(updates, "_write_selected", once)
    result = apply(root, staged)
    assert result["activation_failed"] and result["status"] == "rolled_back"
    assert file_hash(root / REL) == before


def test_future_shared_id_collision_prevents_adoption_not_old_work(case, monkeypatch):
    from dataclasses import replace

    root, source = case
    project.add_knowledge(root, "future-shared-id", front("future-shared-id"), "Private fact")
    old_loader = knowledge.load_entries
    fake = replace(old_loader()[0], id="future-shared-id")
    monkeypatch.setattr(
        knowledge,
        "load_entries",
        lambda directory=None: (
            [*old_loader(), fake] if directory is None else old_loader(directory)
        ),
    )
    staged = updates.stage(root, "update-fixture")
    assert staged["knowledge_id_collisions"] == ["future-shared-id"]
    assert apply(root, staged)["reason"] == "shared_private_knowledge_id_collision"
    assert (root / "records/knowledge/future-shared-id.md").exists()


def test_initializer_does_not_silently_upgrade_existing_project(case):
    root, source = case
    p = root / "records/client_installation.json"
    data = read_json(p)
    data["version"] = "prior-fixture-version"
    write_json(p, data)
    before = file_hash(root / REL)
    change_source(source)
    result = cli.install_project(root, "fixture", "both")
    assert result["status"] == "update_review_required" and not result["active_files_changed"]
    assert file_hash(root / REL) == before


def test_runtime_config_change_preserves_unrelated_values(case, monkeypatch):
    root, _ = case
    codex = root / ".codex/config.toml"
    codex.write_text('model="owner-selection"\n' + codex.read_text(), encoding="utf-8")
    monkeypatch.setattr(cli.sys, "executable", "/candidate/runtime/python")
    candidate = cli.client_files(root, "both", runtime_update=True)
    config = tomllib.loads(candidate[".codex/config.toml"].decode())
    assert config["model"] == "owner-selection"
    assert config["mcp_servers"]["element_surrogate"]["command"] == "/candidate/runtime/python"
    assert (
        json.loads(candidate[".mcp.json"])["mcpServers"]["element-surrogate"]["command"]
        == "/candidate/runtime/python"
    )


def test_mutation_scope_rejects_private_paths(case):
    root, _ = case
    for bad in [
        "records/knowledge/private.md",
        "../outside",
        "/absolute",
        ".agents/skills/../../../records/private",
    ]:
        with pytest.raises(ValueError):
            updates._target(root, bad)


@pytest.mark.parametrize("kind", ["before", "manifest", "candidate"])
@pytest.mark.parametrize("damage", ["corrupt", "missing"])
def test_all_adoption_and_recovery_inputs_checked_before_mutation(case, kind, damage):
    root, source = case
    change_source(source)
    staged = updates.stage(root, "update-fixture")
    plan = read_json(staged["plan_path"])
    item = next(c for c in plan["files"] if c["path"] == REL)
    digest = plan["previous_manifest_sha256"] if kind == "manifest" else item[kind + "_sha256"]
    blob = Path(staged["plan_path"]).parent / "blobs" / digest
    if damage == "corrupt":
        blob.write_bytes(b"manufactured corrupt bytes")
    else:
        blob.unlink()
    before = {c["path"]: file_hash(root / c["path"]) for c in plan["files"]}
    manifest = file_hash(root / "records/client_installation.json")
    result = apply(root, staged)
    assert result["reason"] == "update_assets_invalid"
    assert all(file_hash(root / name) == digest for name, digest in before.items())
    assert file_hash(root / "records/client_installation.json") == manifest
    assert not (Path(staged["plan_path"]).parent / "journal.json").exists()


@pytest.mark.parametrize("kind", ["before", "manifest"])
def test_rollback_preflights_all_bytes_without_partial_restore(case, kind):
    root, source = case
    change_source(source)
    staged = updates.stage(root, "update-fixture")
    assert apply(root, staged)["status"] == "applied"
    folder = Path(staged["plan_path"]).parent
    journal = read_json(folder / "journal.json")
    item = next(c for c in journal["files"] if c["path"] == REL)
    digest = journal["old_manifest_sha256"] if kind == "manifest" else item["before_sha256"]
    (folder / "blobs" / digest).write_bytes(b"manufactured recovery corruption")
    before = {c["path"]: file_hash(root / c["path"]) for c in journal["files"]}
    manifest = file_hash(root / "records/client_installation.json")
    assert updates.rollback(root, "update-fixture")["reason"] == "recovery_assets_invalid"
    assert all(file_hash(root / name) == digest for name, digest in before.items())
    assert file_hash(root / "records/client_installation.json") == manifest


def test_update_inspection_creates_no_approval_or_active_change(case):
    root, source = case
    change_source(source)
    staged = updates.stage(root, "update-fixture")
    before = file_hash(root / REL)
    result = updates.inspect_update(root, "update-fixture")
    assert result["plan_sha256"] == staged["plan_sha256"]
    assert result["review_template"]["checkpoint"] == "not_reviewed"
    assert all(c["outcome"] == "not_reviewed" for c in result["review_template"]["checks"])
    assert not result["approval_created"] and file_hash(root / REL) == before


def test_managed_directory_is_not_overwritten(case):
    root, source = case
    path = root / REL
    path.unlink()
    path.mkdir()
    with pytest.raises(ValueError, match="directory"):
        updates.stage(root, "update-fixture")
    assert path.is_dir()


def test_partial_rollback_write_failure_is_journaled_and_retryable(case, monkeypatch):
    root, source = case
    other = "surrogate-review/ATTRIBUTION.md"
    other_target = ".agents/skills/" + other
    originals = {name: file_hash(root / name) for name in (REL, other_target)}
    change_source(source)
    p = source / other
    p.write_text(p.read_text(encoding="utf-8") + "\nSecond vendor update.\n", encoding="utf-8")
    staged = updates.stage(root, "update-fixture")
    assert apply(root, staged)["status"] == "applied"
    real_write = updates._write_bytes
    restored = []

    def fail_after_first_changed_restore(folder, name, content):
        changed = name in originals and file_hash(folder / name) != originals[name]
        real_write(folder, name, content)
        if changed and not restored:
            restored.append(name)
            raise OSError("Manufactured storage interruption during rollback")

    monkeypatch.setattr(updates, "_write_bytes", fail_after_first_changed_restore)
    result = updates.rollback(root, "update-fixture")
    assert result["reason"] == "recovery_write_failed_retry_from_journal"
    journal = read_json(root / "records/updates/update-fixture/journal.json")
    assert journal["phase"] == "recovery_failed"
    assert len(restored) == 1
    assert sum(file_hash(root / name) == digest for name, digest in originals.items()) == 1
    monkeypatch.setattr(updates, "_write_bytes", real_write)
    assert updates.rollback(root, "update-fixture")["status"] == "rolled_back"
    assert all(file_hash(root / name) == digest for name, digest in originals.items())


def test_runtime_identity_tracks_non_python_package_resources(tmp_path, monkeypatch):
    from element_surrogate import environment

    package = tmp_path / "fake-package"
    package.mkdir()
    resource = package / "rules.json"
    resource.write_text('{"revision": 1}', encoding="utf-8")
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "SKILL.md").write_text("Manufactured skill", encoding="utf-8")
    monkeypatch.setattr(updates, "__file__", str(package / "updates.py"))
    monkeypatch.setattr(cli, "skills_directory", lambda: skills)
    monkeypatch.setattr(environment, "dependency_snapshot", lambda profile: {"versions": {}})
    before = updates.runtime_identity()
    resource.write_text('{"revision": 2}', encoding="utf-8")
    after = updates.runtime_identity()
    assert before["package_sha256"] != after["package_sha256"]
    assert before["skills_sha256"] == after["skills_sha256"]
