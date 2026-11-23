"""AI-assisted project setup using standard Python; no global client changes or scientific jobs."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

if sys.version_info < (3, 11):  # noqa: UP036 - bootstrap runs before installation
    raise SystemExit("Use Python 3.11 or newer. Reference setup checks use Python 3.11.")

KIT = Path(__file__).resolve().parent


class SetupError(ValueError):
    """A concrete setup finding; preserve existing work."""


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def member(root, relative):
    rel = Path(relative)
    target = (root / rel).resolve()
    if (
        rel.is_absolute()
        or ".." in rel.parts
        or "\\" in str(relative)
        or not target.is_relative_to(root.resolve())
    ):
        raise SetupError("Unsafe release member: " + str(relative))
    return target


def release():
    """Snapshot the local software; no prebuilt wheel or generated manifest is required."""
    import tomllib

    path = KIT / "pyproject.toml"
    if not path.is_file():
        raise SetupError("Open the complete ESA source folder containing pyproject.toml")
    project = tomllib.loads(path.read_text(encoding="utf-8"))["project"]
    if project["name"].lower().replace("_", "-") != "element-surrogate":
        raise SetupError("This is not the ESA source project")
    required = ("__init__.py", "cli.py", "mcp_server.py", "lorentz.py", "cards.py")
    package = KIT / "src/element_surrogate"
    if not all((package / name).is_file() for name in required):
        raise SetupError("The local ESA runtime source is incomplete")
    # Source documentation under skills/ is not necessarily installed data.
    # Check the declared skill payload, rather than inventing another file list.
    skill_data = tomllib.loads(path.read_text(encoding="utf-8"))["tool"]["setuptools"].get(
        "data-files", {}
    )
    candidates = list(package.rglob("*"))
    skill_sources = []
    for destination, patterns in skill_data.items():
        if not destination.startswith("share/element-surrogate/skills/"):
            continue
        for pattern in patterns:
            found = sorted(KIT.glob(pattern))
            if not found:
                raise SetupError("Declared skill source missing: " + pattern)
            skill_sources.extend(found)
    if not skill_sources:
        raise SetupError("No installable workflow skills declared")
    candidates.extend(skill_sources)
    files = {}
    for source in sorted(set(candidates)):
        if (
            not source.is_file()
            or "__pycache__" in source.parts
            or source.suffix in (".pyc", ".pyo")
        ):
            continue
        relative = source.relative_to(KIT).as_posix()
        checked = member(KIT, relative)
        if source.is_symlink() or checked != source.absolute():
            raise SetupError("Software source links need a regular source copy: " + relative)
        if source in skill_sources and not relative.startswith("skills/"):
            raise SetupError("Declared skill source must be under skills/: " + relative)
        files[relative] = file_hash(source)
    identity = {
        "package_version": project["version"],
        "pyproject_sha256": file_hash(path),
        "setup_helper_sha256": file_hash(KIT / "setup_esa.py"),
        "files": files,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {
        "schema": "element_surrogate.source_folder.v1",
        **identity,
        "source_root": str(KIT),
        "source_sha256": digest,
    }


def env_python(environment):
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def runtime_identity(python):
    code = (
        "import sys,json,importlib.metadata as m;"
        "print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,"
        "'python':sys.version.split()[0],"
        "'esa_version':next((d.version for d in m.distributions() if "
        "d.metadata['Name'].lower().replace('_','-')=='element-surrogate'),None)}))"
    )
    result = subprocess.run([str(python), "-I", "-c", code], capture_output=True, text=True)
    if result.returncode:
        raise SetupError("Cannot inspect selected Python: " + str(python))
    return json.loads(result.stdout)


def installation_record(project):
    paths = [project / x / "client_installation.json" for x in ("records", ".element_surrogate")]
    existing = [p for p in paths if p.exists()]
    if len(existing) > 1:
        raise SetupError("Two installation records require inspection; neither was changed")
    return read_json(existing[0]) if existing else None


def plan(args):
    manifest = release()
    if args.probe_timeout <= 0:
        raise SetupError("Probe timeout must be positive")
    project = Path(args.project or KIT).expanduser().resolve()
    environment = (
        Path(args.env).expanduser().resolve() if args.env else (project / ".venv").resolve()
    )
    if environment == project or project.is_relative_to(environment):
        raise SetupError("An environment must not contain the project")
    if not args.env and not environment.is_relative_to(project):
        raise SetupError("The default .venv points outside this project; choose it explicitly")
    installed = installation_record(project)
    if installed and installed["version"] != manifest["package_version"]:
        raise SetupError(
            "Existing ESA version differs. Use staged update; no runtime or decisions changed."
        )
    owner_path = project / "records/setup/environment.json"
    owner = read_json(owner_path) if owner_path.exists() else {}
    owned = (
        owner.get("created_by_setup") is True
        and owner.get("environment") == str(environment)
        and owner.get("package_version") == manifest["package_version"]
    )
    identity = None
    if environment.exists():
        if not (args.reuse_env or owned or installed):
            raise SetupError(
                "Existing environment preserved. Inspect it, then use --reuse-env deliberately."
            )
        python = env_python(environment)
        if python.is_file():
            identity = runtime_identity(python)
            if Path(identity["prefix"]).resolve() != environment:
                raise SetupError("Python does not belong to the selected environment")
            if (
                identity["prefix"] == identity["base_prefix"]
                and not (environment / "conda-meta").is_dir()
            ):
                raise SetupError("Use a private environment instead of global Python")
            if identity["esa_version"] not in (None, manifest["package_version"]):
                raise SetupError(
                    "Environment contains another ESA version; choose a new environment"
                )
        elif not owned:
            raise SetupError("Existing directory is not a usable environment; it was preserved")
    if installed is None:
        import tomllib

        targets = (
            [".agents/skills"]
            if args.client == "codex"
            else [".claude/skills"]
            if args.client == "claude"
            else [".agents/skills", ".claude/skills"]
        )
        for target in targets:
            if (project / target / "field-surrogate-workflow").exists():
                raise SetupError("Unmanaged ESA skills need inspection before setup")
        codex = project / ".codex/config.toml"
        if args.client in ("codex", "both") and codex.exists():
            if "element_surrogate" in tomllib.loads(codex.read_text(encoding="utf-8")).get(
                "mcp_servers", {}
            ):
                raise SetupError("Unmanaged ESA Codex connection needs inspection")
        claude = project / ".mcp.json"
        if args.client in ("claude", "both") and claude.exists():
            if "element-surrogate" in read_json(claude).get("mcpServers", {}):
                raise SetupError("Unmanaged ESA Claude connection needs inspection")
    return {
        "schema": "element_surrogate.setup_plan.v1",
        "package_version": manifest["package_version"],
        "project": str(project),
        "environment": str(environment),
        "python": str(env_python(environment)),
        "bootstrap_python": sys.executable,
        "platform": platform.system(),
        "client": args.client,
        "profile": args.profile,
        "torch": args.torch,
        "probe_timeout_s": args.probe_timeout,
        "source_root": manifest["source_root"],
        "source_sha256": manifest["source_sha256"],
        "runtime_source_members": len(manifest["files"]),
        "existing_environment": identity,
        "environment_owned_by_setup": owned,
        "existing_case": installed is not None,
        "action": "verify_existing" if installed else "install_and_verify",
        "scope": "Project environment/client files; no global settings or scientific generation",
        "notes": [
            "Numerical libraries are installed separately from this source folder.",
            "Updates pip only inside the selected private environment.",
            "CPU fallback preserves already compatible PyTorch.",
            "Actual client trust/reload and live-session tool discovery remain to be checked.",
            "Setup checks do not qualify fields or surrogates.",
        ],
    }


def run(command, log, *, as_json=False):
    print(
        "Running " + " ".join(str(p) for p in command[:4]) + "; log: " + str(log),
        file=sys.stderr,
        flush=True,
    )
    with Path(log).open("w", encoding="utf-8") as stream:
        stream.write(json.dumps([str(p) for p in command]) + "\n")
        stream.flush()
        if as_json:
            result = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
            stream.write(result.stdout + "\n" + result.stderr)
        else:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SetupError("Command failed; inspect " + str(log))
    if as_json:
        try:
            return json.loads(result.stdout)
        except ValueError as error:
            raise SetupError("Expected JSON; inspect " + str(log)) from error
    return None


def check_client_files(project, client, python, version):
    import tomllib

    installed = installation_record(project)
    if not installed or installed["version"] != version:
        raise SetupError("Project installation record missing or version differs")
    changed = [
        n
        for n, h in installed["files"].items()
        if not (project / n).is_file() or file_hash(project / n) != h
    ]
    if changed:
        raise SetupError("Managed files differ; local work preserved: " + ", ".join(changed[:5]))
    for selected in ("codex", "claude") if client == "both" else (client,):
        if selected == "codex":
            entry = tomllib.loads((project / ".codex/config.toml").read_text(encoding="utf-8"))[
                "mcp_servers"
            ]["element_surrogate"]
            skills = project / ".agents/skills"
        else:
            entry = read_json(project / ".mcp.json")["mcpServers"]["element-surrogate"]
            skills = project / ".claude/skills"
        if Path(entry["command"]).resolve() != python.resolve() or entry["args"] != [
            "-m",
            "element_surrogate.mcp_server",
        ]:
            raise SetupError("MCP does not select this exact runtime")
        if Path(entry["env"]["ELEMENT_SURROGATE_PROJECT"]).resolve() != project:
            raise SetupError("MCP selects another project")
        if len(list(skills.glob("*/SKILL.md"))) < 5:
            raise SetupError("Workflow skills are missing")
    return {"client": client, "managed_file_hashes_verified": len(installed["files"])}


def installed_payload():
    import element_surrogate

    snapshot = release()
    package = Path(element_surrogate.__file__).resolve().parent
    checked = 0
    for name, expected in snapshot["files"].items():
        if name.startswith("src/element_surrogate/"):
            target = package / name.removeprefix("src/element_surrogate/")
        elif name.startswith("skills/"):
            target = (
                Path(sys.prefix) / "share/element-surrogate/skills" / name.removeprefix("skills/")
            )
        else:
            raise SetupError("Unknown runtime source path: " + name)
        if not target.is_file() or file_hash(target) != expected:
            raise SetupError("Installed payload differs from the local source snapshot: " + name)
        checked += 1
    return checked


async def probe(project, expected_version):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from element_surrogate import __version__

    payload_members = installed_payload()
    if __version__ != expected_version:
        raise SetupError("Installed runtime differs from source version")
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "element_surrogate.mcp_server"],
        env={**os.environ, "ELEMENT_SURROGATE_PROJECT": str(project)},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            initialized = await session.initialize()
            info = initialized.model_dump(by_alias=True)
            if (
                info.get("serverInfo", info.get("server_info", {})).get("version")
                != expected_version
            ):
                raise SetupError("Live MCP server reports another version")
            tools = await session.list_tools()
            names = {item.name for item in tools.tools}
            if not {"inspect_file", "search_knowledge", "read_knowledge"} <= names:
                raise SetupError("Core ESA tools missing")
            path = project / "records/client_installation.json"
            if not path.exists():
                path = project / ".element_surrogate/client_installation.json"
            result = await session.call_tool("inspect_file", {"path": str(path)})
            if result.is_error or result.structured_content["size_bytes"] != path.stat().st_size:
                raise SetupError("MCP inspect_file did not reproduce the local file")
    return {
        "server_version": expected_version,
        "installed_payload_members_verified": payload_members,
        "tools_discovered": len(names),
        "actual_tool_call": "inspect_file",
        "tool_call_passed": True,
        "scope": "Direct local protocol test; not attestation of the LLM client's current catalog",
    }


def verify(details, attempt):
    if release()["source_sha256"] != details["source_sha256"]:
        raise SetupError(
            "Software changed during setup; inspect the saved snapshot before retrying"
        )
    project, python = Path(details["project"]), Path(details["python"])
    runtime = runtime_identity(python)
    if runtime["esa_version"] != details["package_version"]:
        raise SetupError("Installed package metadata differs from release")
    files = check_client_files(project, details["client"], python, details["package_version"])
    doctor = run(
        [
            str(python),
            "-I",
            "-m",
            "element_surrogate.cli",
            "doctor",
            "--profile",
            details["profile"],
        ],
        attempt / "doctor.log",
        as_json=True,
    )
    if not doctor["ready"]:
        raise SetupError("Runtime doctor did not pass")
    smoke = run(
        [str(python), "-I", "-m", "element_surrogate.cli", "self-test", "--project", str(attempt)],
        attempt / "numerical_smoke.log",
        as_json=True,
    )
    if not smoke["passed"]:
        raise SetupError("Numerical smoke did not pass")
    connection = run(
        [
            str(python),
            "-I",
            str(Path(__file__).resolve()),
            "_probe",
            "--project",
            str(project),
            "--expected-version",
            details["package_version"],
            "--probe-timeout",
            str(details["probe_timeout_s"]),
        ],
        attempt / "mcp_probe.log",
        as_json=True,
    )
    packages = run(
        [str(python), "-I", "-m", "pip", "list", "--format", "json"],
        attempt / "packages.log",
        as_json=True,
    )
    write_json(
        attempt / "environment.json", {"runtime": runtime, "packages": packages, "doctor": doctor}
    )
    if release()["source_sha256"] != details["source_sha256"]:
        raise SetupError("Software changed during verification; retry after reviewing local edits")
    return {
        "runtime_ready": True,
        "client_files": files,
        "numerical_smoke_passed": True,
        "mcp_probe": connection,
        "current_ai_session_connection": "pending live client trust/reload and tool call",
        "scientific_qualification": "not performed by setup",
    }


@contextmanager
def installation_lock(project):
    path = project / "records/setup/install.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "host": platform.node()}, stream)
    except FileExistsError as error:
        raise SetupError(
            "Another or interrupted setup owns " + str(path) + "; inspect before retrying"
        ) from error
    try:
        yield
    finally:
        path.unlink()


def ensure_packaging(python, pip, attempt):
    available = subprocess.run(
        [str(python), "-I", "-m", "pip", "--version"], capture_output=True, text=True
    )
    if available.returncode:
        run([str(python), "-I", "-m", "ensurepip", "--upgrade"], attempt / "seed_pip.log")
    run([*pip, "--upgrade", "pip"], attempt / "packaging_tools.log")


def execute(args):
    details = plan(args)
    if args.action == "plan":
        return details
    if args.action == "verify" and not details["existing_case"]:
        raise SetupError("Project not initialized; inspect plan and run install")
    project = Path(details["project"])
    with installation_lock(project):
        attempt = project / "records/setup/attempts" / uuid.uuid4().hex[:12]
        attempt.mkdir(parents=True, exist_ok=False)
        record = {
            "schema": "element_surrogate.setup_attempt.v1",
            "action": args.action,
            "plan": details,
            "status": "running",
            "started_unix_s": time.time(),
        }
        write_json(attempt / "result.json", record)
        try:
            snapshot = release()
            if snapshot["source_sha256"] != details["source_sha256"]:
                raise SetupError("Software changed after planning; no install was started")
            write_json(attempt / "source_snapshot.json", snapshot)
            if args.action == "install" and not details["existing_case"]:
                environment, python = Path(details["environment"]), Path(details["python"])
                if not environment.exists() or not python.is_file():
                    write_json(
                        project / "records/setup/environment.json",
                        {
                            "environment": str(environment),
                            "created_by_setup": True,
                            "package_version": details["package_version"],
                        },
                    )
                    run(
                        [sys.executable, "-m", "venv", "--copies", str(environment)],
                        attempt / "create_environment.log",
                    )
                pip = [
                    str(python),
                    "-I",
                    "-m",
                    "pip",
                    "--cache-dir",
                    str(project / ".cache/pip"),
                    "install",
                ]
                ensure_packaging(python, pip, attempt)
                if args.profile in ("neural", "full"):
                    if args.torch == "cpu":
                        run(
                            [
                                *pip,
                                "torch>=2.6,<3",
                                "--index-url",
                                "https://download.pytorch.org/whl/cpu",
                            ],
                            attempt / "torch.log",
                        )
                    else:
                        run(
                            [
                                str(python),
                                "-I",
                                "-c",
                                (
                                    "import torch; from packaging.specifiers import SpecifierSet; "
                                    'assert torch.__version__ in SpecifierSet(">=2.6,<3"), '
                                    '"Prepare compatible PyTorch first"; print(torch.__version__)'
                                ),
                            ],
                            attempt / "existing_torch.log",
                        )
                source = details["source_root"] + (
                    "" if args.profile == "analysis" else "[" + args.profile + "]"
                )
                run([*pip, source], attempt / "install_package.log")
                initialized = run(
                    [
                        str(python),
                        "-I",
                        "-m",
                        "element_surrogate.cli",
                        "init",
                        "--project",
                        str(project),
                        "--name",
                        args.name or project.name,
                        "--client",
                        args.client,
                    ],
                    attempt / "initialize.log",
                    as_json=True,
                )
                if initialized.get("status") == "update_review_required":
                    raise SetupError(
                        "Project requires staged update; existing client files preserved"
                    )
            record["checks"] = verify(details, attempt)
            record["status"] = "passed"
        except BaseException as error:
            record.update(status="failed", error=str(error), error_type=type(error).__name__)
            write_json(attempt / "result.json", record)
            write_json(
                project / "records/setup/latest.json", {"attempt": str(attempt), "status": "failed"}
            )
            raise
        record["finished_unix_s"] = time.time()
        write_json(attempt / "result.json", record)
        write_json(
            project / "records/setup/latest.json", {"attempt": str(attempt), "status": "passed"}
        )
        return {
            **record["checks"],
            "version": details["package_version"],
            "project": str(project),
            "attempt": str(attempt),
            "status": "passed",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "install", "verify", "_probe"))
    parser.add_argument("--project", help="Default: this source folder")
    parser.add_argument("--client", choices=("codex", "claude", "both"))
    parser.add_argument(
        "--profile", choices=("analysis", "neural", "cards", "full"), default="full"
    )
    parser.add_argument("--torch", choices=("cpu", "preserve"), default="cpu")
    parser.add_argument("--env", help="Explicit private environment")
    parser.add_argument("--reuse-env", action="store_true")
    parser.add_argument("--name", help="Default: project folder name")
    parser.add_argument("--expected-version", help=argparse.SUPPRESS)
    parser.add_argument(
        "--probe-timeout",
        type=float,
        default=60,
        help="Local MCP diagnostic bound in seconds; not an installation/science deadline",
    )
    args = parser.parse_args()
    if args.action != "_probe" and not args.client:
        parser.error("--client codex, claude or both is required")
    try:
        if args.action == "_probe":
            result = asyncio.run(
                asyncio.wait_for(
                    probe(Path(args.project).resolve(), args.expected_version),
                    timeout=args.probe_timeout,
                )
            )
        else:
            result = execute(args)
        print(json.dumps(result, indent=2))
    except Exception as error:
        print(
            json.dumps(
                {"status": "failed", "error": str(error), "error_type": type(error).__name__},
                indent=2,
            )
        )
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
