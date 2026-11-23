"""Setup/package controls use manufactured files only; no network or facility data."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup(tmp_path):
    module = load_script("esa_setup_test", ROOT / "setup_esa.py")
    kit = tmp_path / "renamed workspace with spaces"
    (kit / "src/element_surrogate").mkdir(parents=True)
    (kit / "skills/example").mkdir(parents=True)
    for name in ("__init__.py", "cli.py", "mcp_server.py", "lorentz.py", "cards.py"):
        (kit / "src/element_surrogate" / name).write_bytes(b"# manufactured\n")
    (kit / "skills/example/SKILL.md").write_bytes(b"# Manufactured skill\n")
    (kit / "setup_esa.py").write_bytes(b"# Manufactured bootstrap\n")
    (kit / "pyproject.toml").write_text(
        '[project]\nname="element-surrogate"\nversion="0.2.0rc29"\n'
        "[tool.setuptools.data-files]\n"
        '"share/element-surrogate/skills/example"=["skills/example/SKILL.md"]\n',
        encoding="utf-8",
    )
    module.KIT = kit
    return module


def args(module, **changes):
    defaults = dict(
        action="plan",
        project=None,
        env=None,
        client="both",
        profile="full",
        torch="cpu",
        reuse_env=False,
        name=None,
        probe_timeout=60,
    )
    defaults.update(changes)
    return types.SimpleNamespace(**defaults)


def paths(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


def test_plan_is_read_only_and_uses_renamed_space_path(setup):
    before = paths(setup.KIT)
    result = setup.execute(args(setup))
    assert result["project"] == str(setup.KIT)
    assert result["environment"] == str(setup.KIT / ".venv")
    assert result["profile"] == "full"
    assert result["action"] == "install_and_verify"
    assert paths(setup.KIT) == before
    assert not (setup.KIT / "records").exists()


def test_source_snapshot_tracks_software_but_not_user_data(setup):
    original = setup.release()
    data = setup.KIT / "inputs/fields/user-field.h5"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"manufactured user case, not software")
    assert setup.release()["source_sha256"] == original["source_sha256"]
    (setup.KIT / "src/element_surrogate/lorentz.py").write_text("# local edit\n")
    assert setup.release()["source_sha256"] != original["source_sha256"]
    assert len(original["files"]) == 6
    assert not (setup.KIT / "esa").exists()


@pytest.mark.parametrize("name", ["../outside.txt", "/outside.txt", "esa\\outside.txt"])
def test_source_path_escape_is_rejected(setup, name):
    with pytest.raises(setup.SetupError, match="Unsafe release"):
        setup.member(setup.KIT, name)


def test_incomplete_source_not_installed(setup):
    (setup.KIT / "src/element_surrogate/lorentz.py").unlink()
    with pytest.raises(setup.SetupError, match="incomplete"):
        setup.execute(args(setup, action="install"))
    assert not (setup.KIT / ".venv").exists()
    assert not (setup.KIT / "records").exists()


def test_existing_environment_requires_deliberate_reuse(setup):
    (setup.KIT / ".venv").mkdir()
    with pytest.raises(setup.SetupError, match="reuse-env"):
        setup.plan(args(setup))
    assert paths(setup.KIT / ".venv") == []


def test_existing_environment_is_not_global_python(setup, monkeypatch):
    env = setup.KIT / ".venv"
    python = setup.env_python(env)
    python.parent.mkdir(parents=True)
    python.write_text("manufactured", encoding="utf-8")
    monkeypatch.setattr(
        setup,
        "runtime_identity",
        lambda _: {"prefix": str(env), "base_prefix": str(env), "esa_version": None},
    )
    with pytest.raises(setup.SetupError, match="global Python"):
        setup.plan(args(setup, reuse_env=True))


def test_owned_interrupted_environment_can_resume(setup):
    env = setup.KIT / ".venv"
    env.mkdir()
    setup.write_json(
        setup.KIT / "records/setup/environment.json",
        {
            "environment": str(env),
            "created_by_setup": True,
            "package_version": "0.2.0rc29",
        },
    )
    result = setup.plan(args(setup))
    assert result["environment_owned_by_setup"]
    assert result["action"] == "install_and_verify"


def test_old_case_blocks_upgrade_before_environment_changes(setup):
    path = setup.KIT / "records/client_installation.json"
    setup.write_json(path, {"version": "0.2.0rc27", "files": {}})
    before = path.read_bytes()
    with pytest.raises(setup.SetupError, match="staged update"):
        setup.execute(args(setup, action="install"))
    assert path.read_bytes() == before
    assert not (setup.KIT / ".venv").exists()


@pytest.mark.parametrize(
    "client,relative,content",
    [
        ("codex", ".codex/config.toml", '[mcp_servers.element_surrogate]\ncommand="owner"\n'),
        ("claude", ".mcp.json", '{"mcpServers":{"element-surrogate":{"command":"owner"}}}'),
    ],
)
def test_unmanaged_client_connection_preserved(setup, client, relative, content):
    path = setup.KIT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    with pytest.raises(setup.SetupError, match="Unmanaged"):
        setup.execute(args(setup, action="install", client=client))
    assert path.read_text(encoding="utf-8") == content
    assert not (setup.KIT / ".venv").exists()


def test_unrelated_client_configuration_does_not_block(setup):
    path = setup.KIT / ".codex/config.toml"
    path.parent.mkdir()
    path.write_text('model = "user-choice"\n', encoding="utf-8")
    result = setup.plan(args(setup))
    assert result["client"] == "both"
    assert path.read_text(encoding="utf-8") == 'model = "user-choice"\n'


def test_lock_preserves_other_owner_and_releases_own_on_failure(setup):
    with pytest.raises(RuntimeError, match="manufactured failure"):
        with setup.installation_lock(setup.KIT):
            assert (setup.KIT / "records/setup/install.lock").exists()
            with pytest.raises(setup.SetupError, match="setup owns"):
                with setup.installation_lock(setup.KIT):
                    pass
            raise RuntimeError("manufactured failure")
    assert not (setup.KIT / "records/setup/install.lock").exists()


def test_environment_creation_failure_has_resumable_evidence(setup, monkeypatch):
    def fail(*_, **__):
        raise setup.SetupError("manufactured install failure")

    monkeypatch.setattr(setup, "run", fail)
    with pytest.raises(setup.SetupError, match="manufactured"):
        setup.execute(args(setup, action="install"))
    latest = setup.read_json(setup.KIT / "records/setup/latest.json")
    record = setup.read_json(Path(latest["attempt"]) / "result.json")
    assert latest["status"] == record["status"] == "failed"
    assert record["error"] == "manufactured install failure"
    assert setup.read_json(setup.KIT / "records/setup/environment.json")["created_by_setup"]
    assert not (setup.KIT / "records/setup/install.lock").exists()
    assert not (setup.KIT / "records/project.json").exists()


def test_same_version_rerun_verifies_without_pip_or_budget_reset(setup, monkeypatch):
    setup.write_json(
        setup.KIT / "records/client_installation.json", {"version": "0.2.0rc29", "files": {}}
    )
    decisions = setup.KIT / "records/decisions.json"
    decisions.write_text('{"budget_spent":123,"accepted_units":"mm/mrad"}', encoding="utf-8")
    before = decisions.read_bytes()
    monkeypatch.setattr(setup, "run", lambda *a, **k: pytest.fail("must not reinstall"))
    monkeypatch.setattr(setup, "verify", lambda *a: {"runtime_ready": True})
    result = setup.execute(args(setup, action="install"))
    assert result["status"] == "passed"
    assert decisions.read_bytes() == before
    assert not (setup.KIT / ".venv").exists()


def test_modified_managed_skill_is_not_replaced(setup):
    path = setup.KIT / ".agents/skills/workflow/SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text("user-owned changes", encoding="utf-8")
    setup.write_json(
        setup.KIT / "records/client_installation.json",
        {
            "version": "0.2.0rc29",
            "files": {path.relative_to(setup.KIT).as_posix(): "original-hash"},
        },
    )
    with pytest.raises(setup.SetupError, match="local work preserved"):
        setup.check_client_files(setup.KIT, "codex", setup.KIT / ".venv/python", "0.2.0rc29")
    assert path.read_text() == "user-owned changes"


def test_installed_payload_change_is_detected(setup, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "installed"))
    skill = Path(sys.prefix) / "share/element-surrogate/skills/example/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_bytes(b"# Manufactured skill\n")
    package = tmp_path / "installed/element_surrogate"
    package.mkdir(parents=True)
    for name in ("__init__.py", "cli.py", "mcp_server.py", "lorentz.py", "cards.py"):
        (package / name).write_bytes(b"# manufactured\n")
    monkeypatch.setitem(
        sys.modules,
        "element_surrogate",
        types.SimpleNamespace(__file__=str(package / "__init__.py")),
    )
    assert setup.installed_payload() == 6
    (package / "lorentz.py").write_text("# modified numerical implementation\n", encoding="utf-8")
    with pytest.raises(setup.SetupError, match="Installed payload differs"):
        setup.installed_payload()


def test_bad_probe_timeout_is_not_a_scientific_deadline(setup):
    with pytest.raises(setup.SetupError, match="Probe timeout"):
        setup.plan(args(setup, probe_timeout=0))
    assert setup.plan(args(setup, probe_timeout=600))["probe_timeout_s"] == 600


@pytest.mark.parametrize("pip_present", [True, False])
def test_pip_repair_stays_in_selected_private_environment(
    setup, monkeypatch, tmp_path, pip_present
):
    python = tmp_path / "private/Scripts/python.exe"
    monkeypatch.setattr(
        setup.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode=0 if pip_present else 1),
    )
    calls = []
    monkeypatch.setattr(setup, "run", lambda command, log: calls.append(command))
    pip = [str(python), "-I", "-m", "pip", "install"]
    setup.ensure_packaging(python, pip, tmp_path)
    assert all(command[0] == str(python) for command in calls)
    assert any("ensurepip" in command for command in calls) is (not pip_present)
    assert calls[-1] == [*pip, "--upgrade", "pip"]


def test_install_uses_local_source_extras_and_records_identity(setup, monkeypatch):
    calls = []

    def fake_run(command, log, **kwargs):
        calls.append(command)
        return {"status": "initialized"}

    monkeypatch.setattr(setup, "run", fake_run)
    monkeypatch.setattr(setup, "ensure_packaging", lambda *a: None)
    monkeypatch.setattr(setup, "verify", lambda *a: {"runtime_ready": True})
    result = setup.execute(args(setup, action="install"))
    package_calls = [c for c in calls if c[-1] == str(setup.KIT) + "[full]"]
    assert len(package_calls) == 1
    assert package_calls[0][1:4] == ["-I", "-m", "pip"]
    snapshot = setup.read_json(Path(result["attempt"]) / "source_snapshot.json")
    assert snapshot["source_root"] == str(setup.KIT)
    assert snapshot["source_sha256"] == setup.release()["source_sha256"]
    assert not list(setup.KIT.rglob("*.zip"))


def test_source_changed_after_plan_leaves_failed_attempt(setup, monkeypatch):
    original = setup.plan

    def changed(args):
        details = original(args)
        (setup.KIT / "src/element_surrogate/lorentz.py").write_text("# concurrent edit\n")
        return details

    monkeypatch.setattr(setup, "plan", changed)
    monkeypatch.setattr(setup, "run", lambda *a, **k: pytest.fail("must not install"))
    with pytest.raises(setup.SetupError, match="changed after planning"):
        setup.execute(args(setup, action="install"))
    latest = setup.read_json(setup.KIT / "records/setup/latest.json")
    record = setup.read_json(Path(latest["attempt"]) / "result.json")
    assert record["status"] == "failed"
    assert not (setup.KIT / "records/setup/install.lock").exists()


def test_root_instructions_and_no_builder_requirement():
    for name in ("START_HERE.md", "SETUP.md", "SHIPPING.md", "QUALIFICATION.md"):
        assert (ROOT / name).is_file()
    assert not (ROOT / "workspace").is_dir()
    assert not (ROOT / "scripts/build_workspace.py").exists()
    assert "local source" in (ROOT / "SETUP.md").read_text(encoding="utf-8")
    guidance = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "legitimate user artifacts" in guidance
    assert "source_folder" in (ROOT / "setup_esa.py").read_text(encoding="utf-8")


def test_uninstalled_skill_readme_is_documentation_not_missing_payload(setup):
    (setup.KIT / "skills/README.md").write_text("Source documentation only\n")
    snapshot = setup.release()
    assert "skills/README.md" not in snapshot["files"]
    assert "skills/example/SKILL.md" in snapshot["files"]


def test_declared_skill_payload_cannot_silently_disappear(setup):
    (setup.KIT / "skills/example/SKILL.md").unlink()
    with pytest.raises(setup.SetupError, match="Declared skill source missing"):
        setup.release()
