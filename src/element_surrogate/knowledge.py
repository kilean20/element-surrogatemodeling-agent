"""The knowledge base: versioned entries that the agent searches and the user approves.

Each entry is one Markdown file with YAML front matter. Shipped entries live in the
package (``element_surrogate/kb/``) and cite only public sources and the package's own
tests; a facility's own entries (case records and lessons from its magnets) live in its
project folder. A *method* entry holds the definitions, choices and pitfalls of a
computation, while the computation itself is a tested tool that the entry names.

Entries start as ``proposed`` and become ``active`` when the user approves them (or, if
the user has chosen so, when the agent approves them, marked ``agent-approved``). Search
returns active entries unless proposed ones are asked for; a superseded or retired entry
is returned only by its id.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

KINDS = (
    "method",
    "magnet_class_profile",
    "simulator_profile",
    "case_record",
    "lesson",
    "reference",
)
STATUSES = ("proposed", "active", "superseded", "retired")
APPROVERS = ("user", "agent-approved")
_REQUIRED = ("id", "title", "kind", "status", "date", "summary", "evidence", "keywords")


@dataclass(frozen=True)
class Evidence:
    source: str
    supports: str
    verified: bool

    def as_dict(self) -> dict[str, Any]:
        return {"source": self.source, "supports": self.supports, "verified": self.verified}


@dataclass(frozen=True)
class Entry:
    id: str
    title: str
    kind: str
    status: str
    date: str
    summary: str
    keywords: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    tools: tuple[str, ...]
    tests: tuple[str, ...]
    replaces: str | None
    approved_by: str | None
    body: str
    path: str

    def as_summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "status": self.status,
            "date": self.date,
            "summary": self.summary,
            "approved_by": self.approved_by,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.as_summary(),
            "keywords": list(self.keywords),
            "evidence": [e.as_dict() for e in self.evidence],
            "unverified_evidence": [e.source for e in self.evidence if not e.verified],
            "tools": list(self.tools),
            "tests": list(self.tests),
            "replaces": self.replaces,
            "body": self.body,
        }


def _split(text: str, path: Path) -> tuple[dict[str, Any], str]:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.S)
    if match is None:
        raise ValueError(f"{path}: no YAML front matter")
    front = yaml.safe_load(match.group(1))
    if not isinstance(front, dict):
        raise ValueError(f"{path}: front matter is not a mapping")
    return front, match.group(2)


def parse_entry(path: str | Path) -> Entry:
    """Read and validate one entry file."""
    file_path = Path(path)
    front, body = _split(file_path.read_text(encoding="utf-8"), file_path)
    missing = [k for k in _REQUIRED if k not in front]
    if missing:
        raise ValueError(f"{file_path}: missing fields {missing}")
    if front["id"] != file_path.stem:
        raise ValueError(f"{file_path}: id {front['id']!r} must equal the file name")
    if front["kind"] not in KINDS:
        raise ValueError(f"{file_path}: kind must be one of {KINDS}")
    if front["status"] not in STATUSES:
        raise ValueError(f"{file_path}: status must be one of {STATUSES}")
    approved_by = front.get("approved_by")
    if front["status"] == "proposed" and approved_by is not None:
        raise ValueError(f"{file_path}: a proposed entry has no approver yet")
    if front["status"] == "active" and approved_by not in APPROVERS:
        raise ValueError(f"{file_path}: an active entry needs approved_by in {APPROVERS}")
    date = front["date"]
    if isinstance(date, dt.date):
        date = date.isoformat()
    evidence = []
    for item in front["evidence"] or []:
        if not isinstance(item, dict) or not {"source", "supports", "verified"} <= set(item):
            raise ValueError(f"{file_path}: each evidence item needs source, supports and verified")
        evidence.append(
            Evidence(str(item["source"]), str(item["supports"]), bool(item["verified"]))
        )
    if not evidence:
        raise ValueError(f"{file_path}: an entry needs evidence")
    return Entry(
        id=str(front["id"]),
        title=str(front["title"]),
        kind=front["kind"],
        status=front["status"],
        date=str(date),
        summary=" ".join(str(front["summary"]).split()),
        keywords=tuple(str(k) for k in front["keywords"]),
        evidence=tuple(evidence),
        tools=tuple(str(t) for t in front.get("tools") or ()),
        tests=tuple(str(t) for t in front.get("tests") or ()),
        replaces=front.get("replaces"),
        approved_by=approved_by,
        body=body.strip(),
        path=str(file_path),
    )


def shipped_directory() -> Path:
    return Path(str(resources.files("element_surrogate") / "kb"))


def load_entries(directory: str | Path | None = None) -> list[Entry]:
    """All entries under ``directory`` (default: the shipped knowledge base), sorted by id.
    Raises on an invalid entry or a duplicated id."""
    root = Path(directory) if directory is not None else shipped_directory()
    entries = [parse_entry(p) for p in sorted(root.rglob("*.md")) if p.name != "README.md"]
    ids = [e.id for e in entries]
    duplicated = {i for i in ids if ids.count(i) > 1}
    if duplicated:
        raise ValueError(f"duplicated entry ids: {sorted(duplicated)}")
    return sorted(entries, key=lambda e: e.id)


def search(
    entries: list[Entry], query: str, kind: str | None = None, include_proposed: bool = False
) -> list[dict[str, Any]]:
    """Entries ranked by how many query words they contain (title and keywords count
    double). Active entries only, unless ``include_proposed``."""
    words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 1]
    allowed = {"active", "proposed"} if include_proposed else {"active"}
    ranked = []
    for entry in entries:
        if entry.status not in allowed or (kind is not None and entry.kind != kind):
            continue
        heading = (entry.title + " " + " ".join(entry.keywords)).lower()
        text = (entry.summary + " " + entry.body).lower()
        score = sum(2 * heading.count(w) + text.count(w) for w in words)
        if score > 0:
            ranked.append((score, entry))
    ranked.sort(key=lambda pair: (-pair[0], pair[1].id))
    return [{**entry.as_summary(), "score": score} for score, entry in ranked]


def find(entries: list[Entry], entry_id: str) -> Entry:
    for entry in entries:
        if entry.id == entry_id:
            return entry
    raise ValueError(f"no knowledge entry {entry_id!r}")
