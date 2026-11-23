"""Private client-attested decisions; never shipped case memory.

These events preserve the user's words and their scope. They are not approval
files, authenticated transcripts or automatic promotion into the shared KB.
"""

import uuid
from datetime import UTC, datetime
from pathlib import Path

from element_surrogate.benchmark.protocol import read_json, write_json


def record_decision(
    directory, *, actor, user_statement, scope, source_reference, supersedes=()
) -> dict:
    for value in (actor, user_statement, scope, source_reference):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("actor, verbatim statement, scope and source reference required")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    prior = list_decisions(directory)["events"]
    known = {r["id"] for r in prior}
    if (
        not isinstance(supersedes, (list, tuple))
        or len(set(supersedes)) != len(supersedes)
        or not set(supersedes) <= known
    ):
        raise ValueError("superseded IDs must be distinct existing private decisions")
    identifier = uuid.uuid4().hex
    event = {
        "schema": "element_surrogate.decision.v1",
        "id": identifier,
        "actor": actor,
        "user_statement": user_statement,
        "scope": scope,
        "source_reference": source_reference,
        "provenance": "client attestation; verify against user transcript",
        "not_an_approval_record": True,
        "created_utc": datetime.now(UTC).isoformat(),
        "supersedes": list(supersedes),
    }
    path = directory / f"{identifier}.json"
    write_json(path, event, exclusive=True)
    return {"decision_path": str(path), "decision_id": identifier, "scope": scope}


def list_decisions(directory) -> dict:
    path = Path(directory)
    events = []
    for member in sorted(path.glob("*.json")):
        event = read_json(member)
        if event.get("schema") != "element_surrogate.decision.v1" or member.stem != event["id"]:
            raise ValueError("invalid private decision event")
        events.append(event)
    retired = {identifier for event in events for identifier in event.get("supersedes", [])}
    known = {event["id"]: event for event in events}
    if not retired <= set(known):
        raise ValueError("private decision supersedes an unknown event")

    def visit(identifier, ancestors):
        if identifier in ancestors:
            raise ValueError("cycle in private decision supersession")
        for previous in known[identifier].get("supersedes", []):
            visit(previous, ancestors | {identifier})

    for identifier in known:
        visit(identifier, set())
    active = [event for event in events if event["id"] not in retired]
    scopes: dict[str, list[str]] = {}
    for event in active:
        scopes.setdefault(event["scope"], []).append(event["id"])
    return {
        "events": events,
        "active_events": sorted(active, key=lambda r: (r.get("created_utc", ""), r["id"])),
        "multiple_active_scopes": {k: v for k, v in scopes.items() if len(v) > 1},
        "scope": "private client-attested case decisions; not authenticated approvals",
    }
