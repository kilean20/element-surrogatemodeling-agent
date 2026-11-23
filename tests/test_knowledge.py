"""Tests for the knowledge base: its format, the shipped entries and their rules, and search."""

import re
from pathlib import Path

import pytest

from element_surrogate import knowledge
from element_surrogate.tools import REGISTRY

REPO = Path(__file__).resolve().parent.parent
# Identifiers that must never appear in shipped knowledge: local paths, facility names,
# facility file names and process-variable patterns.
FORBIDDEN = re.compile(r"[A-Za-z]:\\|TRACKsurrogate|FRIB|eh_EMS|FLD3D\.#|_D\d{4}|_T\d{8}")


@pytest.fixture(scope="module")
def shipped():
    return knowledge.load_entries()


def test_shipped_entries_load_and_validate(shipped):
    assert {e.id for e in shipped} >= {
        "lorentz-force-tracking",
        "reference-orbit-in-bending-magnets",
        "strength-error-and-effective-rigidity",
        "track",
    }


def test_shipped_entries_hold_no_facility_identifiers(shipped):
    for entry in shipped:
        text = Path(entry.path).read_text(encoding="utf-8")
        assert not FORBIDDEN.search(text), (entry.id, FORBIDDEN.search(text).group(0))


def test_shipped_evidence_is_public_a_derivation_or_a_package_test(shipped):
    for entry in shipped:
        for item in entry.evidence:
            source = item.source
            if source.startswith("tests/"):
                assert (REPO / source).is_file(), (entry.id, source)
            else:
                assert "http" in source or source == "derivation in this entry" or "," in source, (
                    entry.id,
                    source,
                )


def test_listed_tests_and_tools_exist(shipped):
    for entry in shipped:
        for test in entry.tests:
            assert (REPO / test).is_file(), (entry.id, test)
        for name in entry.tools:
            assert name in REGISTRY, (entry.id, name)


def test_shipped_entries_are_approved_and_found(shipped):
    for entry in shipped:
        if entry.id in {"coordinate-scales", "reflection-features"}:
            assert entry.status == "proposed" and entry.approved_by is None
        else:
            assert entry.status == "active" and entry.approved_by == "user"
    assert "coordinate-scales" not in {
        r["id"] for r in knowledge.search(shipped, "coordinate-scales")
    }
    assert "coordinate-scales" in {
        r["id"] for r in knowledge.search(shipped, "coordinate-scales", include_proposed=True)
    }
    found = knowledge.search(shipped, "reference orbit dipole")
    assert found[0]["id"] == "reference-orbit-in-bending-magnets"


def test_search_filters_by_kind(shipped):
    found = knowledge.search(shipped, "TRACK", kind="simulator_profile")
    assert [r["id"] for r in found] == ["track"]


def test_full_entry_lists_unverified_evidence(shipped):
    entry = knowledge.find(shipped, "track").as_dict()
    assert entry["body"].startswith("## Field tables")
    assert any("PAC 2005" in s for s in entry["unverified_evidence"])
    with pytest.raises(ValueError, match="no knowledge entry"):
        knowledge.find(shipped, "absent")


def _write(directory, name, front):
    path = directory / f"{name}.md"
    path.write_text(f"---\n{front}\n---\nBody.\n", encoding="utf-8")
    return path


VALID = """id: sample
title: Sample
kind: lesson
status: proposed
date: 2026-10-28
summary: A sample.
keywords: [sample]
evidence:
  - {source: derivation in this entry, supports: everything, verified: true}"""


def test_a_valid_entry_parses(tmp_path):
    entry = knowledge.parse_entry(_write(tmp_path, "sample", VALID))
    assert entry.kind == "lesson" and entry.date == "2026-10-28"


@pytest.mark.parametrize(
    "change, message",
    [
        (("id: sample", "id: other"), "must equal the file name"),
        (("kind: lesson", "kind: rumour"), "kind must be one of"),
        (("status: proposed", "status: active"), "needs approved_by"),
        (("status: proposed", "status: proposed\napproved_by: user"), "no approver yet"),
        (("verified: true", "trusted: true"), "needs source, supports and verified"),
        (("title: Sample\n", ""), "missing fields"),
    ],
)
def test_invalid_entries_are_refused(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        knowledge.parse_entry(_write(tmp_path, "sample", VALID.replace(*change)))


def test_an_active_entry_needs_an_approver(tmp_path):
    front = VALID.replace("status: proposed", "status: active\napproved_by: user")
    assert knowledge.parse_entry(_write(tmp_path, "sample", front)).approved_by == "user"


def test_proposed_entries_are_not_returned_as_approved_knowledge(tmp_path):
    _write(tmp_path, "sample", VALID)
    active = VALID.replace("id: sample", "id: approved").replace(
        "status: proposed", "status: active\napproved_by: user"
    )
    _write(tmp_path, "approved", active)
    entries = knowledge.load_entries(tmp_path)
    assert [r["id"] for r in knowledge.search(entries, "sample")] == ["approved"]
    found = knowledge.search(entries, "sample", include_proposed=True)
    assert {r["id"]: r["status"] for r in found} == {"approved": "active", "sample": "proposed"}


def test_duplicated_ids_are_refused(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _write(tmp_path / "a", "sample", VALID)
    _write(tmp_path / "b", "sample", VALID)
    with pytest.raises(ValueError, match="duplicated entry ids"):
        knowledge.load_entries(tmp_path)
