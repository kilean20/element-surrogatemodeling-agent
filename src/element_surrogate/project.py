"""Explicit per-user project state, without importing developer memory."""

from __future__ import annotations

import re
import tempfile
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import yaml

from element_surrogate import knowledge
from element_surrogate.benchmark.protocol import file_hash, read_json, write_json
from element_surrogate.decisions import list_decisions
from element_surrogate.file_io import artifact_guard, publish_file
from element_surrogate.project_paths import mutation_guard, state_directory
from element_surrogate.runs import RunStore

SCHEMA = "element_surrogate.project.v1"


def _validate_text(parent, identifier, text):
    with tempfile.TemporaryDirectory(dir=parent) as sandbox:
        candidate = Path(sandbox) / (identifier + ".md")
        candidate.write_text(text, encoding="utf-8")
        knowledge.parse_entry(candidate)


def initialize(directory, name) -> dict:
    root = Path(directory).resolve()
    if not isinstance(name, str) or not name.strip():
        raise ValueError("project name required")
    state = state_directory(root)
    layout = {name: str(root / name) for name in ("inputs", "data", "studies", "models", "reports")}
    for value in layout.values():
        path = Path(value)
        if path.exists() and not path.is_dir():
            raise ValueError("project layout directory conflicts with existing file: " + str(path))
    state.mkdir(parents=True, exist_ok=True)
    path = state / "project.json"
    if path.exists():
        record = read_json(path)
        if record.get("schema") != SCHEMA or record["name"] != name:
            raise ValueError("existing project identity differs")
    else:
        record = {
            "schema": SCHEMA,
            "id": uuid.uuid4().hex,
            "name": name,
            "created_utc": datetime.now(UTC).isoformat(),
            "memory_policy": "Private state only; no inherited developer/user ledger",
        }
        write_json(path, record, exclusive=True)
    for member in ("decisions", "knowledge"):
        (state / member).mkdir(exist_ok=True)
    for value in layout.values():
        Path(value).mkdir(parents=True, exist_ok=True)
    layout["records"] = str(state)
    readme = root / "README.md"
    if not readme.exists():
        readme.write_text(
            "# " + name.strip() + "\n\n"
            "Single-element surrogate modeling project. Start the chosen client here and "
            "invoke field-surrogate-workflow. Restore this project's records when resuming.\n\n"
            "| Location | Contents |\n|---|---|\n"
            "| [inputs/](inputs/) | Original field files and supplied geometry/descriptions; "
            "keep the source-field data card with the fields. |\n"
            "| [data/](data/) | Generated particle labels, split manifests "
            "and particle-data card. |\n"
            "| [studies/](studies/) | Readiness, tracking qualification and benchmark protocols, "
            "evidence and candidate checkpoints; create subfolders when used. |\n"
            "| [models/](models/) | Selected native model, exports and model card. |\n"
            "| [reports/](reports/) | Human-readable reports and figures. |\n"
            "| [" + state.name + "/](" + state.name + "/) | Durable project decisions, "
            "knowledge, tool provenance and job accounting/recovery; preserve these records. |\n\n"
            "The current state is in the project records and actual reports; this index "
            "does not certify field readiness, completed training or model accuracy. "
            "Generated records do not create user approvals. "
            "Use inputs/fields/, optional inputs/geometry/, studies/field_readiness/, "
            "studies/tracking_validation/ and studies/benchmarks/ as needed. "
            "Keep one authoritative copy of each artifact.\n",
            encoding="utf-8",
        )
    return {
        "project_path": str(root),
        "project": record,
        "layout": layout,
        "created_shared_approval": False,
    }


def state_paths(directory):
    root = Path(directory).resolve()
    path = state_directory(root) / "project.json"
    record = read_json(path)
    if record.get("schema") != SCHEMA:
        raise ValueError("initialized project required")
    return root, record


def entries(directory=None):
    shipped = knowledge.load_entries()
    if directory is None:
        return shipped
    root, _ = state_paths(directory)
    private = knowledge.load_entries(state_directory(root) / "knowledge")
    shared_ids = {e.id for e in shipped}
    if shared_ids & {e.id for e in private}:
        raise ValueError("private knowledge IDs must be distinct; use replaces for an override")
    replaced = {
        e.replaces for e in private if e.status != "proposed" and e.approved_by and e.replaces
    }
    return [replace(e, status="superseded") if e.id in replaced else e for e in shipped + private]


def add_knowledge(directory, entry_id, frontmatter, body) -> dict:
    root, _ = state_paths(directory)
    with mutation_guard(root):
        return _add_knowledge(root, entry_id, frontmatter, body)


def _add_knowledge(directory, entry_id, frontmatter, body) -> dict:
    root, project = state_paths(directory)
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", entry_id):
        raise ValueError("safe lowercase knowledge ID required")
    if not isinstance(body, str) or not body.strip():
        raise ValueError("knowledge body required")
    if (
        not isinstance(frontmatter, dict)
        or frontmatter.get("id") != entry_id
        or frontmatter.get("status") != "proposed"
        or "approved_by" in frontmatter
    ):
        raise ValueError("new project knowledge must be proposed, without an invented approver")
    target = state_directory(root) / "knowledge" / (entry_id + ".md")
    if target.exists():
        raise ValueError("knowledge already exists; preserve original provenance")
    if entry_id in {entry.id for entry in knowledge.load_entries()}:
        raise ValueError(
            "private knowledge needs a distinct ID; use replaces for a scoped override"
        )
    text = "---\n" + yaml.safe_dump(frontmatter, sort_keys=False) + "---\n" + body + "\n"
    _validate_text(target.parent, entry_id, text)
    with tempfile.TemporaryDirectory(dir=target.parent) as sandbox:
        temporary = Path(sandbox) / "entry"
        temporary.write_text(text, encoding="utf-8")
        publish_file(temporary, target, exclusive=True)
    return {
        "knowledge_path": str(target),
        "sha256": file_hash(target),
        "project_id": project["id"],
        "status": "proposed",
    }


def apply_knowledge_decision(
    directory, entry_id, expected_sha256, decision, actor, user_statement, source_reference
) -> dict:
    root, _ = state_paths(directory)
    with mutation_guard(root):
        return _apply_knowledge_decision(
            root, entry_id, expected_sha256, decision, actor, user_statement, source_reference
        )


def _apply_knowledge_decision(
    directory, entry_id, expected_sha256, decision, actor, user_statement, source_reference
) -> dict:
    root, project = state_paths(directory)
    if decision not in ("active", "retired"):
        raise ValueError("explicit active/retired user disposition required")
    if not all(isinstance(v, str) and v.strip() for v in (actor, user_statement, source_reference)):
        raise ValueError("client-attested actor, user statement and source reference required")
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", entry_id):
        raise ValueError("safe knowledge ID required")
    path = state_directory(root) / "knowledge" / (entry_id + ".md")
    with artifact_guard(path, create=True):
        if file_hash(path) != expected_sha256:
            raise ValueError("project knowledge changed since review")
        text = path.read_text(encoding="utf-8")
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.S)
        if not match:
            raise ValueError("knowledge frontmatter required")
        front = yaml.safe_load(match[1])
        if front["id"] != entry_id:
            raise ValueError("knowledge identity differs")
        front["status"] = decision
        if decision == "active":
            front["approved_by"] = "user"
        # Retiring an approved replacement must not silently reactivate the
        # superseded lesson. Its historical approval remains explicit.
        front["project_review"] = {
            "project_id": project["id"],
            "actor": actor,
            "user_statement": user_statement,
            "source_reference": source_reference,
            "provenance": "client attestation, not authenticated identity or shared-KB promotion",
        }
        front["project_reviews"] = front.get("project_reviews", []) + [front["project_review"]]
        updated = "---\n" + yaml.safe_dump(front, sort_keys=False) + "---\n" + match[2]
        _validate_text(path.parent, entry_id, updated)
        # Publication under the existing lock uses os.replace, avoiding nested guards.
        import os

        with tempfile.TemporaryDirectory(dir=path.parent) as sandbox:
            candidate = Path(sandbox) / "entry"
            candidate.write_text(updated, encoding="utf-8")
            os.replace(candidate, path)
    return {
        "knowledge_path": str(path),
        "sha256": file_hash(path),
        "status": decision,
        "shared_knowledge_modified": False,
        "approval_scope": "private project knowledge only",
    }


def restore(directory, max_runs=20) -> dict:
    root, record = state_paths(directory)
    if type(max_runs) is not int or not 0 <= max_runs <= 100:
        raise ValueError("max_runs must be an integer in [0,100]")
    decisions = list_decisions(state_directory(root) / "decisions")
    store = RunStore(root)
    runs = []
    for identifier in store.list_runs()[-max_runs:] if max_runs else []:
        row = store.load(identifier)
        runs.append(
            {
                "run_id": identifier,
                "tool": row["tool"],
                "status": row["status"],
                "error": row.get("error"),
                "inputs": row.get("inputs", []),
                "artifact_arguments": {
                    k: v
                    for k, v in row.get("arguments", {}).items()
                    if k == "path" or k.endswith(("_path", "_dir"))
                },
            }
        )
    private = [
        e for e in entries(root) if Path(e.path).is_relative_to(state_directory(root) / "knowledge")
    ]
    from element_surrogate.learning import policy_summary

    return {
        "project_path": str(root),
        "project": record,
        "decisions": decisions,
        "private_knowledge": [e.as_summary() for e in private],
        "local_learning": policy_summary(root),
        "recent_runs": runs,
        "limits": (
            "Restore explicit case state; do not infer approval or resolv"
            "e ambiguous instructions silently"
        ),
    }
