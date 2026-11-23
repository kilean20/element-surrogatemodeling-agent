"""Inspect actual release archives without extracting them or loading assets."""

import tarfile
import zipfile
from pathlib import Path


def audit_archive(path, *, allowed_members, max_archive_bytes: int, max_member_bytes: int):
    """Enforce a source/text-only release and explicit reviewed size budgets."""
    path = Path(path)
    if min(max_archive_bytes, max_member_bytes) < 1:
        raise ValueError("positive release size budgets required")
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            members = [(x.filename, x.file_size, False) for x in z.infolist() if not x.is_dir()]
    else:
        with tarfile.open(path) as t:
            members = [
                (x.name, x.size, x.issym() or x.islnk()) for x in t.getmembers() if not x.isdir()
            ]
    forbidden = []
    text_suffixes = {
        ".py",
        ".md",
        ".txt",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".cfg",
        ".ini",
        ".rst",
    }
    names = {
        "LICENSE",
        "NOTICE",
        "METADATA",
        "WHEEL",
        "RECORD",
        "PKG-INFO",
        "Makefile",
        "MANIFEST.in",
        "uv.lock",
    }
    for name, size, link in members:
        parts = Path(name).parts
        if (
            link
            or name not in allowed_members
            or ".." in parts
            or Path(name).is_absolute()
            or size > max_member_bytes
            or any(p in (".venv", "__pycache__", ".cache", "node_modules") for p in parts)
            or (Path(name).suffix.lower() not in text_suffixes and Path(name).name not in names)
        ):
            forbidden.append(name)
    return {
        "archive_bytes": path.stat().st_size,
        "members": len(members),
        "forbidden_members": forbidden,
        "passed": not forbidden and path.stat().st_size <= max_archive_bytes,
    }
