import json
import tomllib

import pytest

from element_surrogate.cli import doctor, install_project, self_test
from element_surrogate.project import restore


def test_project_setup_both_clients_preserves_other_settings(tmp_path):
    (tmp_path / ".codex").mkdir()
    (tmp_path / ".codex/config.toml").write_text('model = "owner-choice"\n', encoding="utf-8")
    (tmp_path / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"owner": {"command": "owner-command"}}}), encoding="utf-8"
    )
    result = install_project(tmp_path, "independent-test", "both")
    assert result["managed_files"] > 10
    config = tomllib.loads((tmp_path / ".codex/config.toml").read_text(encoding="utf-8"))
    assert config["model"] == "owner-choice"
    assert config["mcp_servers"]["element_surrogate"]["env"]["ELEMENT_SURROGATE_PROJECT"] == str(
        tmp_path
    )
    assert "owner" in json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]
    for target in (".agents/skills", ".claude/skills"):
        assert len(list((tmp_path / target).glob("*/SKILL.md"))) == 5
        assert (
            tmp_path / target / "field-surrogate-workflow/references/field_readiness.md"
        ).is_file()
        assert (tmp_path / target / "surrogate-benchmark/references/inference_budgets.md").is_file()
    assert (
        install_project(tmp_path, "independent-test", "both")["managed_files"]
        == result["managed_files"]
    )
    assert restore(tmp_path)["decisions"]["events"] == []
    skill = tmp_path / ".agents/skills/field-surrogate-workflow/SKILL.md"
    skill.write_text("owner-edited", encoding="utf-8")
    with pytest.raises(ValueError, match="preserved"):
        install_project(tmp_path, "independent-test", "both")
    assert skill.read_text() == "owner-edited"


def test_conflicting_server_not_overwritten(tmp_path):
    path = tmp_path / ".mcp.json"
    original = json.dumps({"mcpServers": {"element-surrogate": {"command": "different"}}})
    path.write_text(original)
    with pytest.raises(ValueError, match="differs"):
        install_project(tmp_path, "test", "claude")
    assert path.read_text() == original
    assert not (tmp_path / "records/project.json").exists()


def test_doctor_and_self_test(tmp_path):
    result = doctor("analysis")
    assert result["ready"], result
    assert self_test(tmp_path)["passed"]
