"""Coordinate short artifact reads and atomic publication across processes."""

from __future__ import annotations

import errno
import os
import stat
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace


@lru_cache(maxsize=1)
def _windows_api():
    """Use native synchronous locks: contention waits, permission errors fail."""
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class Overlapped(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_size_t),
            ("InternalHigh", ctypes.c_size_t),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    lock, unlock = kernel.LockFileEx, kernel.UnlockFileEx
    lock.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(Overlapped),
    ]
    unlock.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(Overlapped),
    ]
    lock.restype = unlock.restype = wintypes.BOOL
    return SimpleNamespace(
        lock=lock,
        unlock=unlock,
        overlap=Overlapped,
        os_handle=msvcrt.get_osfhandle,
        byref=ctypes.byref,
        error=lambda: ctypes.WinError(ctypes.get_last_error()),
    )


@contextmanager
def _windows_guard(handle, *, create):
    """Lock one sidecar byte with the same blocking semantics as POSIX flock.

    Ordinary Python files are synchronous. With FAIL_IMMEDIATELY absent, the
    kernel waits for a live conflicting lock; an actual API/permission error
    returns and is propagated. Zeroed OVERLAPPED specifies byte zero. The OS
    also releases ownership when the handle closes or the process exits.
    """
    api = _windows_api()
    offset, raw = api.overlap(), api.os_handle(handle.fileno())
    if not api.lock(raw, 2 if create else 0, 0, 1, 0, api.byref(offset)):
        raise api.error()
    try:
        yield
    finally:
        if not api.unlock(raw, 0, 1, 0, api.byref(offset)):
            raise api.error()


def open_guard(path):
    """Exclusively initialize a marker or open it without creation flags.

    Ordinary concurrent O_CREAT opens can alias guest inodes on emulated
    filesystems. This marker is separate from the complete-file data target.
    Existing sidecar bytes are preserved. Range locks may extend past EOF.
    """
    path = Path(path)
    try:
        return path.open("x+b")
    except FileExistsError:
        return path.open("r+b")


@contextmanager
def artifact_guard(path, *, create=False):
    """Guard cooperating JSON readers/writers without holding a data handle.

    Writers create an empty sidecar before first publication. Readers of
    existing artifacts only open that sidecar read-only. Legacy/immutable files
    without a sidecar remain readable without creating files beside user inputs.
    The OS releases the lock if a process exits; the sidecar itself persists.
    """
    guard = Path(str(path) + ".guard")
    if not create and not guard.exists():
        yield
        return
    if create:
        handle = open_guard(guard)
    else:
        handle = guard.open("rb")
    with handle:
        # Windows byte locks may extend beyond EOF; no initialization write
        # races with a reader that has already locked the sidecar.
        handle.seek(0)
        if os.name == "nt":
            with _windows_guard(handle, create=create):
                yield
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX if create else fcntl.LOCK_SH)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def read_utf8(path: str | Path) -> str:
    with artifact_guard(path):
        return Path(path).read_text(encoding="utf-8")


def publish_file(temporary, target, *, exclusive=False):
    """Publish complete files under the cooperating artifact-lock protocol.

    Prefer a hard link for exclusive publication. On shared filesystems that
    reject hard links, reject any existing destination and rename the complete
    staging file while holding the same exclusive guard. Cooperating publishers
    cannot overwrite each other; readers never see an incomplete file. External
    writers ignoring the guard protocol are unsupported by this fallback.
    Never reserve an empty target or copy into it: a crash leaves the target
    absent or complete.
    """
    with artifact_guard(target, create=True):
        if exclusive:
            try:
                os.link(temporary, target)
            except OSError as error:
                if error.errno not in {errno.EPERM, errno.ENOSYS, errno.EOPNOTSUPP}:
                    raise
                if not stat.S_ISREG(os.lstat(temporary).st_mode):
                    raise
                if os.path.lexists(target):
                    raise FileExistsError(
                        errno.EEXIST, "artifact already exists", str(target)
                    ) from None
                os.replace(temporary, target)
        else:
            os.replace(temporary, target)
