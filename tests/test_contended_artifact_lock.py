"""A live cooperating writer is not an implicit ten-second job deadline."""

import errno
import subprocess
import sys
import time
import types

import pytest

from element_surrogate.file_io import _windows_guard, artifact_guard


@pytest.mark.parametrize("create", [False, True])
def test_live_contention_beyond_crt_retry_cutoff_releases_normally(tmp_path, create):
    target = tmp_path / "artifact.json"
    ready = tmp_path / "attempting"
    code = (
        "from pathlib import Path;from element_surrogate.file_io import artifact_guard;"
        f"Path({str(ready)!r}).write_text('attempting');"
        f"\nwith artifact_guard({str(target)!r},create={create!r}):print('acquired',flush=True)\n"
    )
    with artifact_guard(target, create=True):
        child = subprocess.Popen(
            [sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        try:
            end = time.monotonic() + 30
            while not ready.exists():
                if child.poll() is not None:
                    pytest.fail(child.communicate()[1])
                if time.monotonic() > end:
                    pytest.fail("child did not reach lock control")
                time.sleep(0.05)
            time.sleep(12)
            assert child.poll() is None, "a contending cooperating process must wait"
        except BaseException:
            child.kill()
            child.communicate()
            raise
    stdout, stderr = child.communicate(timeout=30)
    assert child.returncode == 0, stderr
    assert stdout.strip() == "acquired"


def test_windows_permission_errors_are_not_retried_as_contention(monkeypatch):
    calls = []
    api = types.SimpleNamespace(
        lock=lambda *args: calls.append(args) or 0,
        unlock=lambda *_: pytest.fail("unlock without a lock"),
        overlap=object,
        os_handle=lambda fd: fd,
        byref=lambda obj: obj,
        error=lambda: PermissionError(errno.EACCES, "access denied"),
    )
    monkeypatch.setattr("element_surrogate.file_io._windows_api", lambda: api)
    with pytest.raises(PermissionError):
        with _windows_guard(types.SimpleNamespace(fileno=lambda: 42), create=True):
            pytest.fail("permission failure cannot enter the protected operation")
    assert len(calls) == 1


def test_windows_lock_releases_if_the_protected_body_fails(monkeypatch):
    calls = []
    api = types.SimpleNamespace(
        lock=lambda *args: calls.append(("lock", args)) or 1,
        unlock=lambda *args: calls.append(("unlock", args)) or 1,
        overlap=object,
        os_handle=lambda fd: fd,
        byref=lambda obj: obj,
        error=lambda: OSError("unexpected API failure"),
    )
    monkeypatch.setattr("element_surrogate.file_io._windows_api", lambda: api)
    with pytest.raises(ValueError, match="body"):
        with _windows_guard(types.SimpleNamespace(fileno=lambda: 42), create=False):
            raise ValueError("body failed")
    assert [k for k, _ in calls] == ["lock", "unlock"]
    assert calls[0][1][1] == 0  # Shared read; the real cross-process test checks exclusion.
