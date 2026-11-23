"""Install project-scoped tools and skills from this environment's distribution."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from importlib import import_module, metadata
from pathlib import Path

from element_surrogate import __version__
from element_surrogate.benchmark.protocol import file_hash, read_json, write_json
from element_surrogate.project import initialize
from element_surrogate.project_paths import mutation_guard, state_directory
from element_surrogate.runs import sha256_json


def skills_directory():
    installed = Path(sys.prefix) / "share/element-surrogate/skills"
    if installed.is_dir():
        return installed
    source = Path(__file__).resolve().parents[2]
    if (source / "pyproject.toml").is_file() and (source / "skills").is_dir():
        return source / "skills"
    raise ValueError("bundled skills missing; reinstall the distribution")


def doctor(profile="analysis"):
    profiles = {
        "analysis": [],
        "neural": ["torch", "onnx", "onnxruntime"],
        "cards": ["linkml", "jsonschema"],
        "full": ["torch", "onnx", "onnxruntime", "linkml", "jsonschema"],
    }
    if profile not in profiles:
        raise ValueError("unknown runtime profile")
    versions, failures = {}, {}
    for package in ["numpy", "scipy", "h5py", "mcp", "pyyaml", "psutil", *profiles[profile]]:
        try:
            import_module("yaml" if package == "pyyaml" else package)
            versions[package] = metadata.version(package)
        except (ImportError, OSError, metadata.PackageNotFoundError) as error:
            failures[package] = str(error)
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    for raw in metadata.requires("element-surrogate") or []:
        requirement = Requirement(raw)
        if requirement.marker and not requirement.marker.evaluate({"extra": profile}):
            continue
        name = canonicalize_name(requirement.name)
        if name in versions and versions[name] not in requirement.specifier:
            failures[name] = "installed version does not satisfy " + str(requirement)
    from element_surrogate.environment import dependency_snapshot

    dependencies = dependency_snapshot(profile)
    if dependencies["missing"] or dependencies["incompatible"]:
        failures["dependency_consistency"] = "Inspect dependency_issues for unmet requirements"
    try:
        from element_surrogate.benchmark.report import verify_vendor
        from element_surrogate.cards import verify_assets

        verify_assets()
        verify_vendor()
        skills = skills_directory()
        names = sorted(p.parent.name for p in skills.glob("*/SKILL.md"))
        if len(names) != 5:
            raise ValueError("five workflow skills expected")
        for path in skills.rglob("*.md"):
            import re

            for link in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", path.read_text(encoding="utf-8")):
                if ":" not in link and not (path.parent / link).exists():
                    raise ValueError("missing packaged skill reference: " + link)
    except (ValueError, OSError) as error:
        failures["assets"] = str(error)
        names = []
    return {
        "version": __version__,
        "profile": profile,
        "python": sys.version.split()[0],
        "versions": versions,
        "skills": names,
        "failures": failures,
        "dependency_issues": dependencies,
        "ready": not failures,
        "scope": (
            "Runtime imports and asset integrity; not real-magnet/autonomous-client qualification"
        ),
    }


def client_files(directory, client, *, runtime_update=False):
    """Construct the candidate overlay without writing active project files."""
    root = Path(directory).resolve()
    proposed = {}
    entry = {
        "command": sys.executable,
        "args": ["-m", "element_surrogate.mcp_server"],
        "env": {"ELEMENT_SURROGATE_PROJECT": str(root)},
    }
    if client in ("codex", "both"):
        config = root / ".codex/config.toml"
        before = config.read_text(encoding="utf-8") if config.exists() else ""
        current = tomllib.loads(before).get("mcp_servers", {}).get("element_surrogate")
        expected = {**entry, "cwd": str(root)}
        if current is not None and current != expected:
            if not runtime_update:
                raise ValueError(
                    "existing element_surrogate MCP configuration differs; review before replacing"
                )
            from element_surrogate.updates import replace_codex_server

            before = replace_codex_server(before, expected)
            current = expected
        # JSON quoted strings/arrays are valid TOML basic strings/arrays here.
        block = "\n[mcp_servers.element_surrogate]\n" + "\n".join(
            k + " = " + json.dumps(v, ensure_ascii=False) for k, v in expected.items() if k != "env"
        )
        block += (
            "\n[mcp_servers.element_surrogate.env]\nELEMENT_SURROGATE_PROJECT = "
            + json.dumps(str(root))
            + "\n"
        )
        proposed[".codex/config.toml"] = (
            before if current is not None else before + block
        ).encode()
    if client in ("claude", "both"):
        config = root / ".mcp.json"
        before = read_json(config) if config.exists() else {}
        current = before.setdefault("mcpServers", {}).get("element-surrogate")
        expected = {"type": "stdio", **entry}
        if current is not None and current != expected and not runtime_update:
            raise ValueError(
                "existing element-surrogate Claude configuration differs; review before replacing"
            )
        before["mcpServers"]["element-surrogate"] = expected
        proposed[".mcp.json"] = (json.dumps(before, indent=2) + "\n").encode()
    sources = skills_directory()
    for target in (
        [".agents/skills"]
        if client == "codex"
        else [".claude/skills"]
        if client == "claude"
        else [".agents/skills", ".claude/skills"]
    ):
        for member in sources.rglob("*"):
            if member.is_file():
                proposed[(Path(target) / member.relative_to(sources)).as_posix()] = (
                    member.read_bytes()
                )
    return proposed


def install_project(directory, name, client):
    root = Path(directory).resolve()
    if (state_directory(root) / "project.json").exists():
        with mutation_guard(root):
            return _install_project(root, name, client)
    return _install_project(root, name, client)


def _install_project(directory, name, client):
    """Preflight all managed files; preserve unrelated config and user edits."""
    if client not in ("codex", "claude", "both"):
        raise ValueError("client must be codex, claude or both")
    root = Path(directory).resolve()
    state = state_directory(root)
    manifest_path = state / "client_installation.json"
    previous = read_json(manifest_path) if manifest_path.exists() else {"files": {}}
    if manifest_path.exists():
        sources = skills_directory()
        source_hash = sha256_json(
            {
                p.relative_to(sources).as_posix(): file_hash(p)
                for p in sources.rglob("*")
                if p.is_file()
            }
        )
        if (
            previous.get("version") != __version__
            or previous.get("source_skills_sha256") != source_hash
        ):
            return {
                "status": "update_review_required",
                "project_path": str(root),
                "active_files_changed": False,
                "next_step": (
                    "Stage a client update from a separate candidate environment; "
                    "preserve local knowledge and validate before adoption"
                ),
            }
    proposed = client_files(root, client)
    for relative, content in proposed.items():
        path = root / relative
        if path.exists() and path.read_bytes() != content:
            old = previous["files"].get(relative)
            # Config changes adding this server are permitted when no managed
            # server existed. Unmanaged skill files are never overwritten.
            config_addition = relative in (".codex/config.toml", ".mcp.json") and old is None
            if not config_addition and (old is None or file_hash(path) != old):
                raise ValueError("user-edited/unmanaged file preserved: " + relative)
    project = initialize(root, name)
    hashes = dict(previous["files"])
    for relative, content in proposed.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        hashes[relative] = file_hash(path)
    from element_surrogate.updates import installation_manifest

    write_json(manifest_path, installation_manifest(root, hashes, previous))
    return {
        **project,
        "client": client,
        "managed_files": len(proposed),
        "next_step": (
            "Open this project in the chosen client, trust its project MC"
            "P configuration, and invoke field-surrogate-workflow. No glo"
            "bal client configuration was changed."
        ),
    }


def self_test(directory):
    """Small exact-vacuum numerical smoke control; no training or model weights."""
    import numpy as np

    from element_surrogate.fields import FieldMap
    from element_surrogate.lorentz import straight_frame, track

    axis, z = np.linspace(-0.04, 0.04, 9), np.linspace(-0.1, 0.1, 21)
    field = FieldMap(axis, axis, z, np.zeros((9, 9, 21, 3)))
    states = np.array([[0.001, 0.002, -0.001, -0.003]])
    result = track(field, states, 2.0, straight_frame(-0.08), straight_frame(0.08), step_m=0.01)
    expected = states.copy()
    expected[:, 0] += 0.16 * states[:, 1]
    expected[:, 2] += 0.16 * states[:, 3]
    error = float(np.max(np.abs(result.states - expected)))
    record = {
        "schema": "element_surrogate.self_test.v1",
        "version": __version__,
        "exact_zero_field_drift_max_absolute_si": error,
        "passed": error < 1e-12,
        "scope": (
            "Numerical/runtime smoke test; no surrogate quality or real-field qualification claim"
        ),
    }
    output = Path(directory).resolve() / "self_test.json"
    write_json(output, record, exclusive=True)
    return {**record, "report_path": str(output)}


def _update_command(args):
    from element_surrogate import updates

    if args.action == "stage":
        return updates.stage(args.project, args.update_id)
    if args.action == "inspect":
        return updates.inspect_update(args.project, args.update_id)
    if args.action == "rollback":
        return updates.rollback(args.project, args.update_id)
    if not args.plan_sha256:
        raise ValueError("--plan-sha256 from the latest staged plan is required")
    if args.action == "apply":
        if not args.review:
            raise ValueError("--review with concrete compatibility evidence is required")
        return updates.apply(args.project, args.update_id, args.plan_sha256, args.review)
    if not all((args.path, args.choice, args.rationale)):
        raise ValueError("resolve requires --path, --choice and --rationale")
    return updates.resolve(
        args.project,
        args.update_id,
        args.plan_sha256,
        args.path,
        args.choice,
        args.rationale,
        args.merged,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor")
    d.add_argument("--profile", choices=["analysis", "neural", "cards", "full"], default="analysis")
    i = sub.add_parser("init")
    i.add_argument("--project", required=True)
    i.add_argument("--name", required=True)
    i.add_argument("--client", choices=["codex", "claude", "both"], required=True)
    s = sub.add_parser("self-test")
    s.add_argument("--project", required=True)
    update = sub.add_parser(
        "update", help="stage/review/recover client assets from a candidate runtime"
    )
    update.add_argument("action", choices=["stage", "inspect", "resolve", "apply", "rollback"])
    update.add_argument("--project", required=True)
    update.add_argument("--id", dest="update_id", required=True)
    update.add_argument("--plan-sha256")
    update.add_argument("--review")
    update.add_argument("--path")
    update.add_argument("--choice", choices=["keep_local", "candidate", "merged"])
    update.add_argument("--rationale")
    update.add_argument("--merged")
    example = sub.add_parser("request-example", help="print an installed JSON request example")
    example.add_argument(
        "kind",
        choices=[
            "cards",
            "public-4d",
            "public-5d",
            "public-6d",
            "campaign",
            "scientific-job",
            "generation",
            "track-compatibility",
            "physics-review",
            "supplied-study",
            "supplied-label-check",
            "uncertainty-review",
            "learning-policy",
            "learning-validation",
            "update-review",
        ],
    )
    example.add_argument(
        "--output", help="write a new UTF-8 JSON file without shell encoding ambiguity"
    )
    sub.add_parser("request-schema", help="print the local card metadata JSON Schema")
    args = p.parse_args()
    try:
        if args.command == "doctor":
            result = doctor(args.profile)
        elif args.command == "init":
            result = install_project(args.project, args.name, args.client)
        elif args.command == "update":
            result = _update_command(args)
        elif args.command == "request-example":
            result = json.loads(
                (Path(__file__).parent / "examples" / (args.kind + ".json")).read_text(
                    encoding="utf-8"
                )
            )
        elif args.command == "request-schema":
            result = json.loads(
                (Path(__file__).parent / "schemas/card_request.json").read_text(encoding="utf-8")
            )
        else:
            result = self_test(args.project)
    except (ValueError, OSError) as error:
        print(json.dumps({"error": str(error)}))
        raise SystemExit(1) from error
    if args.command == "request-example" and args.output:
        write_json(args.output, result, exclusive=True)
    print(json.dumps(result, indent=2))
    if (
        args.command == "doctor"
        and not result["ready"]
        or args.command == "self-test"
        and not result["passed"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
