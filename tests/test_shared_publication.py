"""Contention, error and killed-writer behavior on real filesystems."""

import errno
import json
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

from element_surrogate import file_io


@pytest.mark.parametrize("number", [errno.EPERM, errno.EOPNOTSUPP, errno.ENOSYS])
def test_unsupported_link_publishes_complete_file(tmp_path, number):
    source, target = tmp_path / "staged", tmp_path / "artifact"
    data = b"complete\n" * 16384
    source.write_bytes(data)
    with patch.object(file_io.os, "link", side_effect=OSError(number, "unsupported")):
        file_io.publish_file(source, target, exclusive=True)
    assert target.read_bytes() == data
    assert not source.exists()


def test_existing_record_is_not_replaced(tmp_path):
    source, target = tmp_path / "staged", tmp_path / "artifact"
    source.write_bytes(b"new")
    target.write_bytes(b"original")
    with patch.object(file_io.os, "link", side_effect=OSError(errno.EPERM, "unsupported")):
        with patch.object(file_io.os, "replace") as replace:
            with pytest.raises(FileExistsError):
                file_io.publish_file(source, target, exclusive=True)
            replace.assert_not_called()
    assert target.read_bytes() == b"original"
    assert source.read_bytes() == b"new"


def test_lexists_rejects_dangling_target(tmp_path):
    source = tmp_path / "staged"
    source.write_bytes(b"new")
    with patch.object(file_io.os, "link", side_effect=OSError(errno.EPERM, "unsupported")):
        with patch.object(file_io.os.path, "lexists", return_value=True):
            with patch.object(file_io.os, "replace") as replace:
                with pytest.raises(FileExistsError):
                    file_io.publish_file(source, tmp_path / "dangling", exclusive=True)
                replace.assert_not_called()


@pytest.mark.parametrize("number", [errno.EACCES, errno.EIO, errno.ENOSPC, errno.EXDEV])
def test_other_io_errors_fail_closed(tmp_path, number):
    source, target = tmp_path / "staged", tmp_path / "artifact"
    source.write_bytes(b"new")
    with patch.object(file_io.os, "link", side_effect=OSError(number, "failure")):
        with patch.object(file_io.os, "replace") as replace:
            with pytest.raises(OSError) as caught:
                file_io.publish_file(source, target, exclusive=True)
            assert caught.value.errno == number
            replace.assert_not_called()
    assert not target.exists()


def test_directory_does_not_gain_publication_support(tmp_path):
    source = tmp_path / "directory"
    source.mkdir()
    with patch.object(file_io.os, "link", side_effect=OSError(errno.EPERM, "directory")):
        with patch.object(file_io.os, "replace") as replace:
            with pytest.raises(OSError):
                file_io.publish_file(source, tmp_path / "artifact", exclusive=True)
            replace.assert_not_called()


PREAMBLE = """
from pathlib import Path
import sys,time,json,errno
from element_surrogate import file_io
root=Path(sys.argv[1])
def unsupported(*a,**k): raise OSError(errno.EPERM,'unsupported')
file_io.os.link=unsupported
"""


def worker(code, root, *args):
    return subprocess.Popen(
        [sys.executable, "-c", PREAMBLE + code, str(root), *map(str, args)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def wait_for(path, process):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            pytest.fail("early worker exit: " + repr(process.communicate()))
        time.sleep(0.015)
    pytest.fail("worker did not create " + str(path))


def cleanup(processes):
    for process in processes:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


@pytest.mark.parametrize("cold_round", range(8))
def test_competing_publishers_exactly_one_winner(tmp_path, cold_round):
    code = """
i=sys.argv[2]; source=root/('staged_'+i); source.write_bytes(i.encode()*262144)
(root/('ready_'+i)).touch()
while not (root/'go').exists():time.sleep(0.01)
try:
 file_io.publish_file(source,root/'artifact',exclusive=True);status='won'
except FileExistsError:status='exists'
print(json.dumps({'identity':i,'status':status}))
"""
    processes = [worker(code, tmp_path, i) for i in "ABC"]
    try:
        for i, process in zip("ABC", processes, strict=True):
            wait_for(tmp_path / ("ready_" + i), process)
        (tmp_path / "go").touch()
        outcomes = []
        for process in processes:
            out, err = process.communicate(timeout=15)
            assert process.returncode == 0, err
            outcomes.append(json.loads(out))
        winners = [r for r in outcomes if r["status"] == "won"]
        assert len(winners) == 1
        assert sum(r["status"] == "exists" for r in outcomes) == 2
        assert (tmp_path / "artifact").read_bytes() == winners[0]["identity"].encode() * 262144
    finally:
        cleanup(processes)


def test_guard_initialization_preserves_existing_sidecar_and_absent_data(tmp_path):
    target = tmp_path / "artifact"
    guard = tmp_path / "artifact.guard"
    with file_io.artifact_guard(target, create=True):
        assert guard.exists() and not target.exists()
    guard.write_bytes(b"legacy coordination bytes")
    for create in (False, True, False):
        with file_io.artifact_guard(target, create=create):
            assert not target.exists()
        assert guard.read_bytes() == b"legacy coordination bytes"


def test_empty_and_legacy_study_guards_refuse_a_live_duplicate(tmp_path):
    from element_surrogate.benchmark.study import _guard

    for legacy in (False, True):
        directory = tmp_path / str(legacy)
        directory.mkdir()
        marker = directory / "study.guard"
        if legacy:
            marker.write_bytes(b"0")
        with _guard(directory):
            with pytest.raises(ValueError, match="another process"):
                with _guard(directory):
                    pytest.fail("a live study guard was acquired twice")
        with _guard(directory):
            pass
        assert marker.read_bytes() == (b"0" if legacy else b"")


@pytest.mark.parametrize("after", [False, True])
def test_killed_writer_leaves_absent_or_complete_record(tmp_path, after):
    code = """
after=sys.argv[2]=='True';source=root/'staged';source.write_bytes(b'complete'*65536)
replace=file_io.os.replace
def paused(source,target):
 if after:replace(source,target)
 (root/'paused').touch();time.sleep(60)
file_io.os.replace=paused
file_io.publish_file(source,root/'artifact',exclusive=True)
"""
    process = worker(code, tmp_path, after)
    try:
        wait_for(tmp_path / "paused", process)
        process.kill()
        process.communicate(timeout=5)
        target = tmp_path / "artifact"
        assert target.exists() is after
        if after:
            assert target.read_bytes() == b"complete" * 65536
        source = tmp_path / "recovery"
        source.write_bytes(b"recovery")
        with patch.object(file_io.os, "link", side_effect=OSError(errno.EPERM, "unsupported")):
            if after:
                with pytest.raises(FileExistsError):
                    file_io.publish_file(source, target, exclusive=True)
            else:
                file_io.publish_file(source, target, exclusive=True)
                assert target.read_bytes() == b"recovery"
    finally:
        cleanup([process])


def test_reader_waits_for_complete_first_publication(tmp_path):
    code = """
source=root/'staged';source.write_text('complete'*65536);replace=file_io.os.replace
def paused(source,target):
 (root/'paused').touch()
 while not (root/'release').exists():time.sleep(0.01)
 replace(source,target)
file_io.os.replace=paused
file_io.publish_file(source,root/'artifact',exclusive=True)
"""
    publisher = worker(code, tmp_path)
    reader = None
    try:
        wait_for(tmp_path / "paused", publisher)
        reader = worker(
            "(root/'reader_started').touch();print(len(file_io.read_utf8(root/'artifact')))",
            tmp_path,
        )
        wait_for(tmp_path / "reader_started", reader)
        time.sleep(0.15)
        assert reader.poll() is None, "reader did not wait for publisher guard"
        (tmp_path / "release").touch()
        out, err = publisher.communicate(timeout=10)
        assert publisher.returncode == 0, err
        out, err = reader.communicate(timeout=10)
        assert reader.returncode == 0, err
        assert int(out.strip()) == len("complete") * 65536
    finally:
        cleanup([publisher, reader])
