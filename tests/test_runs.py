"""Tests for run records and replay."""

import json

import pytest

from element_surrogate.runs import (
    RECORD_SCHEMA,
    RunStore,
    Tool,
    record_call,
    replay,
    sha256_file,
    sha256_json,
)
from element_surrogate.tools import REGISTRY


@pytest.fixture
def field_file(tmp_path):
    path = tmp_path / "field.txt"
    path.write_text("0 0 0 0.0 0.0 1.0\n", encoding="utf-8")
    return path


@pytest.fixture
def store(tmp_path):
    return RunStore(tmp_path / "project")


def test_record_holds_hashes_and_timing(store, field_file):
    result = record_call(store, REGISTRY["inspect_file"], {"path": str(field_file)})

    record = store.load(result["run_id"])
    assert record["schema"] == RECORD_SCHEMA
    assert record["status"] == "ok"
    assert record["tool"] == "inspect_file"
    assert record["inputs"] == [{"path": str(field_file), "sha256": sha256_file(field_file)}]
    summary = {k: v for k, v in result.items() if k != "run_id"}
    assert record["result"] == summary
    assert record["result_sha256"] == sha256_json(summary)
    assert record["wall_time_s"] >= 0.0
    assert record["cpu_time_s"] >= 0.0


def test_replay_reproduces_identical_hash(store, field_file):
    run_id = record_call(store, REGISTRY["inspect_file"], {"path": str(field_file)})["run_id"]

    report = replay(store, REGISTRY, run_id)

    assert report.reproduced
    assert report.changed_inputs == []
    assert report.replayed_result_sha256 == report.recorded_result_sha256


def test_replay_reports_changed_input_instead_of_running(store, field_file):
    run_id = record_call(store, REGISTRY["inspect_file"], {"path": str(field_file)})["run_id"]
    field_file.write_text("0 0 0 0.0 0.0 2.0\n", encoding="utf-8")

    report = replay(store, REGISTRY, run_id)

    assert not report.reproduced
    assert report.changed_inputs == [str(field_file)]
    assert report.replayed_result_sha256 == ""


def test_failed_call_is_recorded_and_reraised(store, tmp_path):
    def broken(path):
        raise RuntimeError("boom")

    tool = Tool(name="broken", func=broken, version="1")
    with pytest.raises(RuntimeError):
        record_call(store, tool, {"path": "unused"})

    (run_id,) = store.list_runs()
    record = store.load(run_id)
    assert record["status"] == "error"
    assert "boom" in record["error"]
    with pytest.raises(ValueError):
        replay(store, {"broken": tool}, run_id)


def test_records_are_listed_in_order_and_are_valid_json(store, field_file):
    ids = [
        record_call(store, REGISTRY["inspect_file"], {"path": str(field_file)})["run_id"]
        for _ in range(3)
    ]
    assert store.list_runs() == sorted(ids)
    for run_id in ids:
        json.loads(store.path_for(run_id).read_text(encoding="utf-8"))


@pytest.mark.parametrize("bad", ["", "../escape", "a/b", "a\\b"])
def test_run_ids_cannot_leave_the_store(store, bad):
    with pytest.raises(ValueError):
        store.path_for(bad)
