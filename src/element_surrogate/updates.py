"""Staged, recoverable updates of managed client assets, never project science.

Run from a separate candidate environment. These consistency guards are not a
security sandbox, semantic merge engine, package installer or proof of physics.
The prior environment must remain available; no running process is upgraded.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import tomllib
from copy import deepcopy
from pathlib import Path, PurePosixPath

from element_surrogate import __version__, knowledge
from element_surrogate.benchmark.protocol import file_hash, read_json, write_json
from element_surrogate.file_io import publish_file
from element_surrogate.learning import context
from element_surrogate.project import state_paths
from element_surrogate.project_paths import mutation_guard, state_directory
from element_surrogate.runs import sha256_json


def _digest(blob):
    return hashlib.sha256(blob).hexdigest()


def _target(root, name):
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "\\" in name
        or not (
            name in {".codex/config.toml", ".mcp.json"}
            or name.startswith((".agents/skills/", ".claude/skills/"))
        )
    ):
        raise ValueError("only managed client skills/configuration may be updated")
    path = root.joinpath(*relative.parts)
    if path.resolve() != path or not path.is_relative_to(root):
        raise ValueError("symlinked client paths require explicit manual reconciliation")
    if path.exists() and not path.is_file():
        raise ValueError("a managed client file cannot replace an existing directory")
    return path


def _atomic(path, blob):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as folder:
        temp = Path(folder) / "complete"
        with temp.open("xb") as output:
            output.write(blob)
            output.flush()
            os.fsync(output.fileno())
        publish_file(temp, path)


def _store(folder, blob):
    digest = _digest(blob)
    path = folder / digest
    if path.exists():
        if file_hash(path) != digest:
            raise ValueError("stored update blob is corrupt")
    else:
        _atomic(path, blob)
    return digest


def _blob(folder, digest):
    if digest is None:
        return None
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("valid stored update digest required")
    path = folder / digest
    raw = path.read_bytes()
    if _digest(raw) != digest:
        raise ValueError("stored update blob changed")
    return raw


def replace_codex_server(before, expected):
    """Replace our generated tables only; verify all other TOML values survive."""
    original = tomllib.loads(before)
    wanted = deepcopy(original)
    wanted.setdefault("mcp_servers", {})["element_surrogate"] = expected
    lines, kept, skipping = before.splitlines(keepends=True), [], False
    for line in lines:
        header = re.match(r"^\s*\[([^\]]+)\]\s*(?:#.*)?$", line)
        if header:
            name = header[1].strip()
            skipping = name == "mcp_servers.element_surrogate" or name.startswith(
                "mcp_servers.element_surrogate."
            )
        if not skipping:
            kept.append(line)
    block = "\n[mcp_servers.element_surrogate]\n" + "\n".join(
        key + " = " + json.dumps(value, ensure_ascii=False)
        for key, value in expected.items()
        if key != "env"
    )
    block += (
        "\n[mcp_servers.element_surrogate.env]\n"
        + "\n".join(
            key + " = " + json.dumps(value, ensure_ascii=False)
            for key, value in expected["env"].items()
        )
        + "\n"
    )
    result = "".join(kept) + block
    try:
        parsed = tomllib.loads(result)
    except tomllib.TOMLDecodeError as error:
        raise ValueError("manual Codex configuration reconciliation required") from error
    if parsed != wanted:
        raise ValueError("manual Codex configuration reconciliation required")
    return result


def runtime_identity():
    from element_surrogate.cli import skills_directory
    from element_surrogate.environment import dependency_snapshot

    package = Path(__file__).parent
    skills = skills_directory()
    return {
        "version": __version__,
        "python": sys.executable,
        "python_version": sys.version,
        "dependency_versions": dependency_snapshot("full")["versions"],
        "skills_sha256": sha256_json(
            {
                p.relative_to(skills).as_posix(): file_hash(p)
                for p in skills.rglob("*")
                if p.is_file()
            }
        ),
        "package_sha256": sha256_json(
            {
                p.relative_to(package).as_posix(): file_hash(p)
                for p in package.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
            }
        ),
    }


def knowledge_identity():
    return {e.id: file_hash(e.path) for e in knowledge.load_entries()}


def installation_manifest(root, files, previous=None):
    """Preserve vendor before-images separately from deliberate local variants."""
    from element_surrogate.cli import skills_directory

    previous = previous or {}
    bases = dict(previous.get("baselines", {}))
    folder = state_directory(root) / "client_baselines"
    for name, expected in files.items():
        path = _target(root, name)
        if path.is_file() and file_hash(path) == expected:
            bases[name] = _store(folder, path.read_bytes())
    skills = skills_directory()
    return {
        "schema": "element_surrogate.client_installation.v1",
        "version": __version__,
        "files": files,
        "baselines": bases,
        "runtime": runtime_identity(),
        "knowledge_baseline": knowledge_identity(),
        "retained_local_only": previous.get("retained_local_only", []),
        "source_skills_sha256": sha256_json(
            {
                p.relative_to(skills).as_posix(): file_hash(p)
                for p in skills.rglob("*")
                if p.is_file()
            }
        ),
    }


def _location(directory, update_id):
    root, _ = state_paths(directory)
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", update_id):
        raise ValueError("safe lowercase update ID required")
    return root, state_directory(root) / "updates" / update_id


def _pending(reason, **extra):
    return {
        "status": "update_review_required",
        "reason": reason,
        "unrelated_work_may_continue_with_pinned_runtime": True,
        **extra,
    }


def stage(directory, update_id):
    from element_surrogate.cli import client_files

    root, folder = _location(directory, update_id)
    with mutation_guard(root):
        if folder.exists():
            raise ValueError("update ID already exists; preserve previous evidence")
        manifest_path = state_directory(root) / "client_installation.json"
        previous = read_json(manifest_path)
        names = previous["files"]
        codex = any(n.startswith(".agents/") or n == ".codex/config.toml" for n in names)
        claude = any(n.startswith(".claude/") or n == ".mcp.json" for n in names)
        if not (codex or claude):
            return _pending("no_managed_client_installation")
        client = "both" if codex and claude else "codex" if codex else "claude"
        try:
            incoming = client_files(root, client, runtime_update=True)
        except ValueError as error:
            return _pending("manual_configuration_reconciliation", detail=str(error))
        blobs = folder / "blobs"
        blobs.mkdir(parents=True)
        changes = []
        for name in sorted(set(names) | set(incoming)):
            path = _target(root, name)
            before = _store(blobs, path.read_bytes()) if path.is_file() else None
            candidate = _store(blobs, incoming[name]) if name in incoming else None
            base = names.get(name)
            choice = "candidate"
            if before == candidate or before == base:
                selected = candidate
            elif candidate == base or (
                candidate is None and name in previous.get("retained_local_only", [])
            ):
                selected, choice = before, "keep_local"
            else:
                selected, choice = before, "conflict"
            changes.append(
                {
                    "path": name,
                    "base_sha256": base,
                    "before_sha256": before,
                    "candidate_sha256": candidate,
                    "selected_sha256": selected,
                    "resolution": choice,
                    "rationale": None,
                }
            )
        private = knowledge.load_entries(state_directory(root) / "knowledge")
        new_kb = knowledge_identity()
        collisions = sorted({e.id for e in private} & new_kb.keys())
        old_kb = previous.get("knowledge_baseline")
        changed_kb = (
            sorted(k for k in set(old_kb) | set(new_kb) if old_kb.get(k) != new_kb.get(k))
            if old_kb is not None
            else None
        )
        plan = {
            "schema": "element_surrogate.client_update.v1",
            "update_id": update_id,
            "project_id": state_paths(root)[1]["id"],
            "client": client,
            "previous_manifest_sha256": file_hash(manifest_path),
            "context_sha256": context(root)["context_sha256"],
            "candidate_runtime": runtime_identity(),
            "candidate_knowledge": new_kb,
            "changed_shared_knowledge": changed_kb,
            "knowledge_baseline_incomplete": old_kb is None,
            "knowledge_id_collisions": collisions,
            "private_knowledge": [e.as_summary() for e in private],
            "files": changes,
            "scope": (
                "managed client assets; no private knowledge/data/model "
                "or live-environment mutation"
            ),
        }
        write_json(folder / "plan.json", plan, exclusive=True)
        _store(blobs, manifest_path.read_bytes())
    return {
        "status": "staged",
        "plan_path": str(folder / "plan.json"),
        "plan_sha256": file_hash(folder / "plan.json"),
        "conflicts": [c["path"] for c in changes if c["resolution"] == "conflict"],
        "knowledge_id_collisions": collisions,
        "active_files_changed": False,
        "knowledge_baseline_incomplete": old_kb is None,
    }


def resolve(
    directory, update_id, expected_plan_sha256, relative_path, choice, rationale, merged_path=None
):
    root, folder = _location(directory, update_id)
    if choice not in {"keep_local", "candidate", "merged"} or not rationale.strip():
        raise ValueError("explicit resolution and rationale required")
    with mutation_guard(root):
        path = folder / "plan.json"
        if file_hash(path) != expected_plan_sha256 or (folder / "journal.json").exists():
            raise ValueError("plan changed or activation already began")
        plan = read_json(path)
        item = next((c for c in plan["files"] if c["path"] == relative_path), None)
        if item is None:
            raise ValueError("resolution must name a staged file")
        if choice == "merged":
            source = Path(merged_path or "")
            source = (source if source.is_absolute() else root / source).resolve()
            if not source.is_relative_to(root) or not source.is_file():
                raise ValueError("merged file must be inside this project")
            selected = _store(folder / "blobs", source.read_bytes())
        else:
            selected = item["before_sha256" if choice == "keep_local" else "candidate_sha256"]
        item.update(selected_sha256=selected, resolution=choice, rationale=rationale)
        write_json(path, plan)
    return {"plan_path": str(path), "plan_sha256": file_hash(path), "active_files_changed": False}


def inspect_update(directory, update_id):
    """Read staging/recovery state and a blank review template; no adoption."""
    _, folder = _location(directory, update_id)
    plan_path = folder / "plan.json"
    plan = read_json(plan_path)
    journal_path = folder / "journal.json"
    return {
        "status": read_json(journal_path)["phase"] if journal_path.exists() else "staged",
        "plan_path": str(plan_path),
        "plan_sha256": file_hash(plan_path),
        "journal_path": str(journal_path) if journal_path.exists() else None,
        "unresolved_files": [c["path"] for c in plan["files"] if c["resolution"] == "conflict"],
        "knowledge_id_collisions": plan["knowledge_id_collisions"],
        "changed_shared_knowledge": plan["changed_shared_knowledge"],
        "review_template": {
            "schema": "element_surrogate.client_update_validation.v1",
            "plan_sha256": file_hash(plan_path),
            "context_sha256": plan["context_sha256"],
            "checkpoint": "not_reviewed",
            "rationale": "",
            "checks": [
                {"name": name, "outcome": "not_reviewed", "rationale": "", "evidence": []}
                for name in (
                    "guidance_and_local_knowledge",
                    "runtime_and_interfaces",
                    "affected_project_behavior",
                )
            ],
        },
        "active_files_changed": False,
        "approval_created": False,
    }


def _verify_plan(root, folder, expected):
    path = folder / "plan.json"
    if file_hash(path) != expected:
        return None, "plan_changed"
    plan = read_json(path)
    if plan["project_id"] != state_paths(root)[1]["id"]:
        return None, "different_project"
    if (
        file_hash(state_directory(root) / "client_installation.json")
        != plan["previous_manifest_sha256"]
    ):
        return None, "installation_manifest_changed"
    if plan["candidate_runtime"] != runtime_identity():
        return None, "candidate_runtime_changed"
    if plan["context_sha256"] != context(root)["context_sha256"]:
        return None, "project_context_changed"
    if plan["knowledge_id_collisions"]:
        return None, "shared_private_knowledge_id_collision"
    if any(c["resolution"] == "conflict" for c in plan["files"]):
        return None, "unresolved_local_file_conflicts"
    for item in plan["files"]:
        p = _target(root, item["path"])
        if (file_hash(p) if p.is_file() else None) != item["before_sha256"]:
            return None, "active_file_changed_since_stage"
        _blob(folder / "blobs", item["before_sha256"])
        _blob(folder / "blobs", item["selected_sha256"])
        _blob(folder / "blobs", item["candidate_sha256"])
    _blob(folder / "blobs", plan["previous_manifest_sha256"])
    return plan, None


def _validate_review(root, folder, plan, expected, review_path):
    path = Path(review_path)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("update review must be inside the project")
    review = read_json(path)
    if (
        review.get("schema") != "element_surrogate.client_update_validation.v1"
        or review.get("plan_sha256") != expected
        or review.get("context_sha256") != plan["context_sha256"]
    ):
        return "review_does_not_match_plan_and_context"
    if review.get("checkpoint") != "idle" or not review.get("rationale", "").strip():
        return "reviewed_idle_checkpoint_required"
    checks = review.get("checks", [])
    required = {
        "guidance_and_local_knowledge",
        "runtime_and_interfaces",
        "affected_project_behavior",
    }
    if {c.get("name") for c in checks} != required or len(checks) != len(required):
        return "change_directed_compatibility_checks_required"
    for check in checks:
        if (
            check.get("outcome") not in {"passed", "not_applicable"}
            or not check.get("rationale", "").strip()
        ):
            return "compatibility_check_not_supported"
        if not isinstance(check.get("evidence"), list) or not check["evidence"]:
            return "concrete_compatibility_evidence_required"
        for item in check["evidence"]:
            source = Path(item["path"])
            source = (source if source.is_absolute() else root / source).resolve()
            if not source.is_relative_to(root) or file_hash(source) != item.get("sha256"):
                return "compatibility_evidence_changed"
    # Check links against the selected overlay, rather than only the incoming vendor copy.
    selected = {item["path"]: item["selected_sha256"] for item in plan["files"]}
    for name, digest in selected.items():
        if digest is None or not name.endswith(".md"):
            continue
        text = _blob(folder / "blobs", digest).decode("utf-8")
        for link in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", text):
            if ":" in link or link.startswith("#"):
                continue
            target = (root / name).parent.joinpath(link).resolve()
            if not target.is_relative_to(root):
                return "managed_reference_leaves_project"
            relative = target.relative_to(root).as_posix()
            if selected[relative] is None if relative in selected else not target.exists():
                return "selected_client_reference_missing"
    return None


def _write_selected(root, folder, item, key):
    content = _blob(folder / "blobs", item[key])
    _write_bytes(root, item["path"], content)


def _write_bytes(root, name, content):
    path = _target(root, name)
    if content is None:
        path.unlink(missing_ok=True)
    else:
        _atomic(path, content)


def _restore(root, folder, journal):
    # Read and validate every recovery byte before writing any active file. The
    # stored copies remain immutable inputs; validated buffers prevent a later
    # blob read from leaving a known partially restored overlay.
    try:
        original_files = {
            item["path"]: _blob(folder / "blobs", item["before_sha256"])
            for item in journal["files"]
        }
        original_manifest = _blob(folder / "blobs", journal["old_manifest_sha256"])
        if original_manifest is None:
            raise ValueError("old installation manifest is required")
    except (OSError, ValueError) as error:
        return _pending("recovery_assets_invalid", error_type=type(error).__name__)
    for item in journal["files"]:
        path = _target(root, item["path"])
        actual = file_hash(path) if path.is_file() else None
        if actual not in {item["before_sha256"], item["selected_sha256"]}:
            return _pending("later_user_edit_preserved", path=item["path"])
    manifest = state_directory(root) / "client_installation.json"
    if file_hash(manifest) not in {journal["old_manifest_sha256"], journal["new_manifest_sha256"]}:
        return _pending("later_installation_manifest_preserved")
    try:
        for item in reversed(journal["files"]):
            _write_bytes(root, item["path"], original_files[item["path"]])
        _atomic(manifest, original_manifest)
    except OSError as error:
        journal.update(phase="recovery_failed", recovery_error_type=type(error).__name__)
        write_json(folder / "journal.json", journal)
        return _pending("recovery_write_failed_retry_from_journal", error_type=type(error).__name__)
    journal["phase"] = "rolled_back"
    write_json(folder / "journal.json", journal)
    return {
        "status": "rolled_back",
        "private_knowledge_changed": False,
        "running_runtime_changed": False,
    }


def apply(directory, update_id, expected_plan_sha256, review_path):
    root, folder = _location(directory, update_id)
    with mutation_guard(root):
        if (folder / "journal.json").exists():
            return _pending("activation_already_recorded_inspect_or_recover")
        try:
            plan, reason = _verify_plan(root, folder, expected_plan_sha256)
        except (OSError, ValueError) as error:
            return _pending("update_assets_invalid", error_type=type(error).__name__)
        if reason:
            return _pending(reason)
        reason = _validate_review(root, folder, plan, expected_plan_sha256, review_path)
        if reason:
            return _pending(reason)
        manifest_path = state_directory(root) / "client_installation.json"
        previous = read_json(manifest_path)
        blobs = folder / "blobs"
        vendor_files = {
            c["path"]: c["candidate_sha256"]
            for c in plan["files"]
            if c["candidate_sha256"] is not None
        }
        retained = [
            c["path"]
            for c in plan["files"]
            if c["candidate_sha256"] is None and c["selected_sha256"] is not None
        ]
        for name in retained:
            vendor_files[name] = previous["files"][name]
        # Build metadata before mutation; baseline blobs always describe vendor bytes.
        from element_surrogate.cli import skills_directory

        skills = skills_directory()
        new_manifest = {
            "schema": previous["schema"],
            "version": __version__,
            "files": vendor_files,
            "runtime": runtime_identity(),
            "knowledge_baseline": knowledge_identity(),
            "retained_local_only": retained,
            "source_skills_sha256": sha256_json(
                {
                    p.relative_to(skills).as_posix(): file_hash(p)
                    for p in skills.rglob("*")
                    if p.is_file()
                }
            ),
            "baselines": dict(previous.get("baselines", {})),
        }
        for item in plan["files"]:
            if item["candidate_sha256"] is not None:
                blob = _blob(blobs, item["candidate_sha256"])
                new_manifest["baselines"][item["path"]] = _store(
                    state_directory(root) / "client_baselines", blob
                )
        blob = (json.dumps(new_manifest, sort_keys=True, indent=2) + "\n").encode()
        new_hash = _store(blobs, blob)
        journal = {
            "phase": "applying",
            "plan_sha256": expected_plan_sha256,
            "old_manifest_sha256": file_hash(manifest_path),
            "new_manifest_sha256": new_hash,
            "files": sorted(
                plan["files"], key=lambda c: c["path"] in {".mcp.json", ".codex/config.toml"}
            ),
        }
        write_json(folder / "journal.json", journal, exclusive=True)
        try:
            for item in journal["files"]:
                if item["selected_sha256"] != item["before_sha256"]:
                    _write_selected(root, folder, item, "selected_sha256")
            _atomic(manifest_path, blob)
            journal.update(
                phase="applied",
                after_context_sha256=context(root)["context_sha256"],
                review_sha256=file_hash(
                    Path(review_path) if Path(review_path).is_absolute() else root / review_path
                ),
            )
            write_json(folder / "journal.json", journal)
        except Exception as error:
            journal.update(phase="failed", error_type=type(error).__name__)
            write_json(folder / "journal.json", journal)
            result = _restore(root, folder, journal)
            return {**result, "activation_failed": True, "error_type": type(error).__name__}
    return {
        "status": "applied",
        "private_knowledge_changed": False,
        "running_runtime_changed": False,
        "next_step": (
            "Reconnect the client at the reviewed checkpoint; "
            "retain the prior environment for recovery"
        ),
        "qualification": "recorded compatibility judgments and checks, not a semantic guarantee",
    }


def rollback(directory, update_id):
    root, folder = _location(directory, update_id)
    with mutation_guard(root):
        journal = read_json(folder / "journal.json")
        if journal["phase"] == "rolled_back":
            return {"status": "rolled_back", "changed": False}
        if (
            journal["phase"] == "applied"
            and journal["after_context_sha256"] != context(root)["context_sha256"]
        ):
            return _pending("project_changed_since_adoption_stage_a_fresh_reverse_update")
        return _restore(root, folder, journal)
