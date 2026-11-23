"""User-visible new cases, unchanged legacy cases and non-destructive setup."""

import json

import pytest

from element_surrogate.cli import install_project
from element_surrogate.project import initialize, restore
from element_surrogate.project_paths import state_directory
from element_surrogate.runs import RunStore, Tool, record_call


def test_new_case_has_visible_records_and_intuitive_index(tmp_path):
    result = install_project(tmp_path, "new-field-study", "both")
    assert result["layout"]["records"] == str(tmp_path / "records")
    assert (tmp_path / "records/project.json").is_file()
    assert not (tmp_path / ".element_surrogate").exists()
    for name in ("inputs", "data", "studies", "models", "reports"):
        assert (tmp_path / name).is_dir()
        assert f"]({name}/)" in (tmp_path / "README.md").read_text()
    for name in (".agents/skills", ".claude/skills"):
        assert len(list((tmp_path / name).glob("*/SKILL.md"))) == 5


def test_user_readme_and_unrelated_inputs_are_preserved(tmp_path):
    (tmp_path / "README.md").write_text("My approved project notes")
    (tmp_path / "inputs").mkdir()
    field = tmp_path / "inputs/original.txt"
    field.write_text("unmodified supplied source")
    first = install_project(tmp_path, "same", "codex")
    second = install_project(tmp_path, "same", "codex")
    assert first["project"]["id"] == second["project"]["id"]
    assert (tmp_path / "README.md").read_text() == "My approved project notes"
    assert field.read_text() == "unmodified supplied source"


def test_legacy_project_identity_records_and_budget_stay_in_place(tmp_path):
    initialize(tmp_path, "old")
    (tmp_path / "records").rename(tmp_path / ".element_surrogate")
    legacy = tmp_path / ".element_surrogate"
    before = json.loads((legacy / "project.json").read_text())
    budget = legacy / "campaign_ledger.json"
    budget.write_text('{"jobs":{"previous":{"spent_seconds":12.5}}}')
    digest = budget.read_bytes()
    restored = restore(tmp_path)
    setup = install_project(tmp_path, "old", "claude")
    assert restored["project"] == before == setup["project"]
    assert budget.read_bytes() == digest
    assert not (tmp_path / "records").exists()
    assert (legacy / "client_installation.json").is_file()


@pytest.mark.parametrize("legacy", [False, True])
def test_run_logging_and_restore_use_the_same_state_root(tmp_path, legacy):
    initialize(tmp_path, "test")
    if legacy:
        (tmp_path / "records").rename(tmp_path / ".element_surrogate")
    store = RunStore(tmp_path)
    result = record_call(store, Tool("echo", lambda: {"ok": True}, "1"), {})
    assert store.path_for(result["run_id"]).is_file()
    assert restore(tmp_path)["recent_runs"][0]["run_id"] == result["run_id"]
    assert store.root.parent == state_directory(tmp_path)


def test_nonempty_dual_roots_fail_without_changing_either(tmp_path):
    initialize(tmp_path, "original")
    legacy = tmp_path / ".element_surrogate"
    legacy.mkdir()
    (legacy / "other.json").write_text('{"keep":true}')
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    for operation in (
        lambda: initialize(tmp_path, "original"),
        lambda: restore(tmp_path),
        lambda: RunStore(tmp_path),
        lambda: install_project(tmp_path, "original", "both"),
    ):
        with pytest.raises(ValueError, match="both records/"):
            operation()
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_empty_legacy_root_does_not_block_visible_setup(tmp_path):
    (tmp_path / ".element_surrogate").mkdir()
    initialize(tmp_path, "visible")
    assert state_directory(tmp_path) == tmp_path / "records"


def test_unrelated_user_records_do_not_block_legacy_project(tmp_path):
    initialize(tmp_path, "old")
    (tmp_path / "records").rename(tmp_path / ".element_surrogate")
    (tmp_path / "records").mkdir()
    notes = tmp_path / "records/user-notes.md"
    notes.write_text("ordinary user records, not a second agent identity")
    assert restore(tmp_path)["project"]["name"] == "old"
    assert state_directory(tmp_path).name == ".element_surrogate"
    assert notes.read_text() == "ordinary user records, not a second agent identity"


def test_conflicting_layout_file_is_not_replaced(tmp_path):
    (tmp_path / "data").write_text("user-owned file")
    with pytest.raises(ValueError, match="conflicts with existing file"):
        initialize(tmp_path, "not-created")
    assert (tmp_path / "data").read_text() == "user-owned file"
    assert not (tmp_path / "records").exists()


def test_legacy_campaign_worker_and_accounting_keep_original_root(tmp_path):
    from element_surrogate import jobs
    from tests.test_scientific_jobs import campaign, request, wait

    initialize(tmp_path, "manufactured resource test")
    (tmp_path / "records").rename(tmp_path / ".element_surrogate")
    path, _ = campaign(tmp_path)
    assert path == str(tmp_path / ".element_surrogate/campaign.json")
    source = tmp_path / "inputs/tiny.txt"
    source.write_text("synthetic legacy compatibility")
    row = wait(path, jobs.submit_request(path, request(source)))
    assert row["status"] == "complete", row
    before = (tmp_path / ".element_surrogate/campaign_ledger.json").read_bytes()
    assert jobs.find_campaign(tmp_path / "studies") == path
    assert row["charged_job_wall_seconds"] > 0
    install_project(tmp_path, "manufactured resource test", "both")
    assert (tmp_path / ".element_surrogate/campaign_ledger.json").read_bytes() == before
    assert not (tmp_path / "records").exists()


@pytest.mark.parametrize("legacy", [False, True])
def test_nested_field_metadata_named_records_does_not_shadow_campaign(tmp_path, legacy):
    from element_surrogate import jobs
    from tests.test_scientific_jobs import campaign

    initialize(tmp_path, "manufactured resource test")
    if legacy:
        (tmp_path / "records").rename(tmp_path / ".element_surrogate")
    path, _ = campaign(tmp_path)
    nested = tmp_path / "inputs/fields"
    nested.mkdir()
    (nested / "records").write_text("ordinary input metadata, not an agent project")
    source = nested / "field.txt"
    source.write_text("synthetic field header")
    assert jobs.find_campaign(source) == path
