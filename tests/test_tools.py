"""Tests for the tool registry."""

import hashlib

import pytest

from element_surrogate.tools import REGISTRY, inspect_file


def test_text_file_summary(tmp_path):
    data = b"x y z bx by bz\n0 0 0 0.0 0.0 1.0\n"
    path = tmp_path / "field.txt"
    path.write_bytes(data)

    summary = inspect_file(str(path))

    assert summary["size_bytes"] == len(data)
    assert summary["sha256"] == hashlib.sha256(data).hexdigest()
    assert summary["is_text"] is True
    assert summary["line_count"] == 2


def test_last_line_without_newline_is_counted(tmp_path):
    path = tmp_path / "field.txt"
    path.write_bytes(b"a\nb")
    assert inspect_file(str(path))["line_count"] == 2


def test_empty_file(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_bytes(b"")
    summary = inspect_file(str(path))
    assert summary["size_bytes"] == 0
    assert summary["line_count"] == 0


def test_binary_file_has_no_line_count(tmp_path):
    path = tmp_path / "field.bin"
    path.write_bytes(b"\x89HDF\r\n\x1a\n\x00\x00")
    summary = inspect_file(str(path))
    assert summary["is_text"] is False
    assert summary["line_count"] is None


def test_file_larger_than_one_read_block(tmp_path):
    data = b"0123456789\n" * 200_000  # about 2.2 MB, several read blocks
    path = tmp_path / "big.txt"
    path.write_bytes(data)
    summary = inspect_file(str(path))
    assert summary["sha256"] == hashlib.sha256(data).hexdigest()
    assert summary["line_count"] == 200_000


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_file(str(tmp_path / "absent.txt"))


def test_registry_marks_inputs_and_read_only():
    tool = REGISTRY["inspect_file"]
    assert tool.read_only
    assert tool.input_paths({"path": "a.txt"}) == ["a.txt"]
