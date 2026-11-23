"""Offline project learning under recorded, revocable user delegation.

These are trusted-client consistency records, not identity authentication or a
proof that a scientific judgment is true. No sharing or workflow gate authority
is created. Inconclusive learning leaves a proposal and unrelated work usable.
"""

from __future__ import annotations

import re
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml

from element_surrogate import __version__, knowledge
from element_surrogate.benchmark.protocol import file_hash, read_json, write_json
from element_surrogate.file_io import artifact_guard
from element_surrogate.project import state_paths
from element_surrogate.project_paths import mutation_guard, state_directory
from element_surrogate.runs import sha256_json

POLICY_SCHEMA = "element_surrogate.local_learning_policy.v1"
VALIDATION_SCHEMA = "element_surrogate.learning_validation.v1"


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + ": nonempty text required")


def _inside(root, path):
    path = Path(path)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("learning evidence must stay inside this project")
    return path


def policy_summary(directory):
    root, project = state_paths(directory)
    path = state_directory(root) / "learning_policy.json"
    if not path.exists():
        return {"mode": "manual", "policy_sha256": None, "sharing_authorized": False}
    record = read_json(path)
    if record.get("project_id") != project["id"]:
        return {
            "mode": "manual",
            "policy_sha256": file_hash(path),
            "advisory": "delegation_belongs_to_another_project",
            "sharing_authorized": False,
        }
    return {
        "mode": record["policy"]["mode"],
        "policy_sha256": file_hash(path),
        "allowed_kinds": record["policy"]["allowed_kinds"],
        "scope": record["policy"]["scope"],
        "allow_replacements": record["policy"]["allow_replacements"],
        "sharing_authorized": False,
    }


def record_policy(directory, policy_path, expected_sha256, actor, user_statement, source_reference):
    root, project = state_paths(directory)
    for name, value in [
        ("actor", actor),
        ("user_statement", user_statement),
        ("source_reference", source_reference),
    ]:
        _text(value, name)
    source = _inside(root, policy_path)
    with mutation_guard(root):
        if file_hash(source) != expected_sha256:
            raise ValueError("learning policy changed since review")
        policy = read_json(source)
        if policy.get("schema") != POLICY_SCHEMA or policy.get("mode") not in {
            "manual",
            "delegated",
        }:
            raise ValueError("supported local learning policy and manual/delegated mode required")
        kinds = policy.get("allowed_kinds")
        if (
            not isinstance(kinds, list)
            or len(kinds) != len(set(kinds))
            or any(k not in knowledge.KINDS for k in kinds)
            or (policy["mode"] == "delegated" and not kinds)
        ):
            raise ValueError("explicit allowed knowledge kinds required")
        _text(policy.get("scope"), "scope")
        if type(policy.get("allow_replacements")) is not bool:
            raise ValueError("explicit allow_replacements boolean required")
        record = {
            "schema": "element_surrogate.local_learning_authority.v1",
            "project_id": project["id"],
            "policy": policy,
            "reviewed_policy_sha256": expected_sha256,
            "actor": actor,
            "user_statement": user_statement,
            "source_reference": source_reference,
            "recorded_utc": datetime.now(UTC).isoformat(),
            "authority": "client-attested local delegation; not authenticated identity",
            "sharing_or_scientific_gate_authority": False,
        }
        history = state_directory(root) / "learning_policy_history" / (uuid.uuid4().hex + ".json")
        write_json(history, record, exclusive=True)
        write_json(state_directory(root) / "learning_policy.json", record)
    return {
        **policy_summary(root),
        "history_path": str(history),
        "existing_accepted_knowledge_changed": False,
    }


def context(directory):
    """Fingerprint explicit knowledge, decisions and active managed guidance."""
    root, project = state_paths(directory)
    state = state_directory(root)
    files = {}
    for folder in (state / "knowledge", state / "decisions"):
        for path in folder.rglob("*"):
            if path.is_file() and not path.name.endswith(".guard"):
                files[path.relative_to(root).as_posix()] = file_hash(path)
    shared = {entry.id: file_hash(entry.path) for entry in knowledge.load_entries()}
    managed = {}
    manifest = state / "client_installation.json"
    if manifest.exists():
        for name in read_json(manifest)["files"]:
            path = _inside(root, name)
            managed[name] = file_hash(path) if path.is_file() else None
    value = {
        "project_id": project["id"],
        "project_sha256": file_hash(state / "project.json"),
        "policy_sha256": file_hash(state / "learning_policy.json")
        if (state / "learning_policy.json").exists()
        else None,
        "runtime_version": __version__,
        "shared_knowledge": shared,
        "project_records": files,
        "managed_guidance": managed,
    }
    return {
        "context_sha256": sha256_json(value),
        "baseline": value,
        "local_learning": policy_summary(root),
        "scope": (
            "Explicit records and installed assets; semantic agreement is not proven by hashes"
        ),
    }


def activate(
    directory, entry_id, expected_sha256, validation_path, expected_policy_sha256, rationale
):
    root, _ = state_paths(directory)
    _text(rationale, "rationale")
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", entry_id):
        raise ValueError("safe lowercase knowledge ID required")
    path = state_directory(root) / "knowledge" / (entry_id + ".md")

    def pending(reason):
        return {
            "status": "proposed_review_required",
            "reason": reason,
            "knowledge_path": str(path),
            "unrelated_work_may_continue": True,
            "shared_knowledge_modified": False,
            "sharing_authorized": False,
        }

    with mutation_guard(root), artifact_guard(path, create=True):
        if file_hash(path) != expected_sha256:
            return pending("entry_changed_since_validation")
        entry = knowledge.parse_entry(path)
        if entry.status != "proposed":
            return {
                "status": entry.status,
                "approved_by": entry.approved_by,
                "changed": False,
                "sharing_authorized": False,
            }
        policy = policy_summary(root)
        if policy["mode"] != "delegated":
            return pending("no_active_local_delegation")
        if policy["policy_sha256"] != expected_policy_sha256:
            return pending("local_delegation_changed")
        if entry.kind not in policy["allowed_kinds"]:
            return pending("kind_outside_local_delegation")
        validation_file = _inside(root, validation_path)
        validation = read_json(validation_file)
        if validation.get("schema") != VALIDATION_SCHEMA:
            raise ValueError("learning validation schema required")
        if validation.get("entry_sha256") != expected_sha256:
            return pending("validation_names_different_entry")
        if validation.get("context_sha256") != context(root)["context_sha256"]:
            return pending("reviewed_context_changed")
        if validation.get("outcome") != "supported":
            return pending("validation_not_supported")
        for key in ("scope_assessment", "rationale", "existing_guidance_comparison"):
            _text(validation.get(key), key)
        if not isinstance(validation.get("conflicts"), list):
            raise ValueError("explicit conflict review required")
        if validation["conflicts"]:
            return pending("scientific_or_user_decision_conflict")
        evidence = validation.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("concrete local validation evidence required")
        for item in evidence:
            _text(item.get("supports"), "evidence supports")
            target = _inside(root, item["path"])
            if file_hash(target) != item.get("sha256"):
                return pending("validation_evidence_changed")
        if entry.replaces:
            if not policy["allow_replacements"]:
                return pending("replacement_outside_local_delegation")
            from element_surrogate.project import entries

            try:
                old = knowledge.find(entries(root), entry.replaces)
            except ValueError:
                return pending("replacement_baseline_unavailable")
            if Path(old.path).is_relative_to(state_directory(root)) and old.approved_by == "user":
                return pending("replacement_of_user_accepted_knowledge_needs_user_decision")
        text = path.read_text(encoding="utf-8")
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.S)
        front = yaml.safe_load(match[1])
        front.update(status="active", approved_by="agent-approved")
        front["evidence"].append(
            {
                "source": validation_file.relative_to(root).as_posix(),
                "supports": validation["rationale"],
                "verified": True,
            }
        )
        front["local_learning_review"] = {
            "policy_sha256": expected_policy_sha256,
            "validation_sha256": file_hash(validation_file),
            "context_sha256": validation["context_sha256"],
            "rationale": rationale,
            "recorded_utc": datetime.now(UTC).isoformat(),
            "provenance": (
                "agent judgment under client-attested user delegation; not proof of truth"
            ),
        }
        updated = "---\n" + yaml.safe_dump(front, sort_keys=False) + "---\n" + match[2]
        from element_surrogate.project import _validate_text

        _validate_text(path.parent, entry_id, updated)
        import os

        with tempfile.TemporaryDirectory(dir=path.parent) as scratch:
            candidate = Path(scratch) / "entry"
            candidate.write_text(updated, encoding="utf-8")
            os.replace(candidate, path)
    return {
        "status": "active",
        "approved_by": "agent-approved",
        "sha256": file_hash(path),
        "knowledge_path": str(path),
        "shared_knowledge_modified": False,
        "sharing_authorized": False,
        "validation_scope": "trusted-client local evidence review",
    }
