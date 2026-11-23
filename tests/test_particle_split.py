"""Particle splits may share physical settings, never particle identities."""

import numpy as np
import pytest

from element_surrogate.benchmark.protocol import (
    file_hash,
    validate_protocol,
    write_json,
)
from element_surrogate.benchmark.report import build_report
from element_surrogate.benchmark.study import (
    create_study,
    evaluate_test,
    lock_selection,
    run_study,
)
from tests.benchmark_fixture import approval, make_case


def shared_case(tmp_path):
    path, p, _ = make_case(tmp_path / "case", configurations=1)
    for split in ("train", "validation", "test"):
        f = path.parent / (split + ".npz")
        with np.load(f, allow_pickle=False) as data:
            content = {k: data[k] for k in data.files}
        content["setting_id"] = np.full(len(content["sample_id"]), "same-physical-setting")
        np.savez_compressed(f, **content)
        p["dataset"]["splits"][split].update(sha256=file_hash(f), groups=["same-physical-setting"])
    return path, p


def decisions(path, p):
    write_json(path, p)
    return [
        approval(path.parent, p, g)
        for g in ("description_target", "region_ranges", "compute_budget")
    ]


def test_shared_settings_require_explicit_particle_split(tmp_path):
    _, p = shared_case(tmp_path)
    with pytest.raises(ValueError, match="unique and disjoint"):
        validate_protocol(p)
    p["dataset"]["split_unit"] = "particle"
    assert validate_protocol(p) is p
    p["dataset"]["split_unit"] = "renamed-folds"
    with pytest.raises(ValueError, match="split_unit"):
        validate_protocol(p)


def test_particle_split_preserves_train_validation_id_guard(tmp_path):
    path, p = shared_case(tmp_path)
    p["dataset"]["split_unit"] = "particle"
    f = path.parent / "validation.npz"
    with np.load(f, allow_pickle=False) as d:
        content = {k: d[k] for k in d.files}
    with np.load(path.parent / "train.npz", allow_pickle=False) as d:
        content["sample_id"][0] = d["sample_id"][0]
    np.savez_compressed(f, **content)
    p["dataset"]["splits"]["validation"]["sha256"] = file_hash(f)
    with pytest.raises(ValueError, match="sample ID overlap"):
        create_study(path, tmp_path / "study", decisions(path, p))


def test_particle_report_and_guarded_test(tmp_path):
    path, p = shared_case(tmp_path)
    p["dataset"]["split_unit"] = "particle"
    study = create_study(path, tmp_path / "study", decisions(path, p))["study_path"]
    assert run_study(study)["status"] == "completed"
    lock_selection(study)
    evaluate_test(study, [approval(path.parent, p, "release")])
    report = build_report(study, tmp_path / "report")
    from pathlib import Path

    text = Path(report["report_path"]).read_text()
    assert "Holdout unit: **particle**" in text
    assert "does not test unseen-setting" in text


def test_particle_split_preserves_final_test_id_guard(tmp_path):
    path, p = shared_case(tmp_path)
    p["dataset"]["split_unit"] = "particle"
    target = path.parent / "test.npz"
    with np.load(target, allow_pickle=False) as data:
        content = {k: data[k] for k in data.files}
    with np.load(path.parent / "train.npz", allow_pickle=False) as data:
        content["sample_id"][0] = data["sample_id"][0]
    np.savez_compressed(target, **content)
    p["dataset"]["splits"]["test"]["sha256"] = file_hash(target)
    study = create_study(path, tmp_path / "study", decisions(path, p))["study_path"]
    assert run_study(study)["status"] == "completed"
    lock_selection(study)
    with pytest.raises(ValueError, match="test sample ID overlap"):
        evaluate_test(study, [approval(path.parent, p, "release")])
