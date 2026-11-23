"""Doctor must detect dependency inconsistency even when major imports work."""

from email.message import Message
from importlib import metadata

import pytest

from element_surrogate.environment import dependency_snapshot


@pytest.fixture
def distributions(monkeypatch):
    from element_surrogate import environment

    data = {
        "element-surrogate": ("1.0", ["engine>=1", "optional>=1; extra == 'neural'"]),
        "engine": ("1.0", ["helper>=2"]),
        "helper": ("1.0", []),
    }

    def distribution(name):
        if name not in data:
            raise metadata.PackageNotFoundError(name)
        version, requires = data[name]
        info = Message()
        info["Version"] = version
        for requirement in requires:
            info["Requires-Dist"] = requirement
        return type("Distribution", (), {"metadata": info})()

    monkeypatch.setattr(environment.metadata, "distribution", distribution)
    return data


def test_transitive_mismatch_is_visible_without_import_failure(distributions):
    result = dependency_snapshot()
    assert result["incompatible"] == [
        {"required_by": "engine", "requirement": "helper>=2", "installed": "1.0"}
    ]
    assert not result["missing"]
    assert "optional" not in result["versions"]


def test_selected_extra_is_checked_but_not_fabricated(distributions):
    assert dependency_snapshot("neural")["missing"] == ["optional"]


def test_transitive_extra_and_cycle_are_handled(distributions):
    distributions["engine"] = ("1.0", ["helper[feature]>=1"])
    distributions["helper"] = ("2.0", ["engine>=1", "missing>=1; extra == 'feature'"])
    assert dependency_snapshot()["missing"] == ["missing"]


def test_unrelated_installed_package_is_not_part_of_identity(distributions):
    distributions["unrelated"] = ("1.0", ["not-installed>=1"])
    assert "unrelated" not in dependency_snapshot()["versions"]
