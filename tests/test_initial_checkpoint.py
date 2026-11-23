"""An accurate train-only base must not be forced through a worse checkpoint."""

import numpy as np

from element_surrogate.benchmark.protocol import file_hash, read_json, write_json
from element_surrogate.benchmark.study import create_study, lock_selection, run_study
from tests.benchmark_fixture import approval, make_case


def test_initial_checkpoint_is_eligible_and_selection_revalidates_it(tmp_path):
    path, protocol, _ = make_case(tmp_path / "inputs", configurations=1)
    for item in protocol["dataset"]["splits"].values():
        source = path.parent / item["path"]
        with np.load(source, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        arrays["targets"] = arrays["inputs"][:, :4].copy()
        np.savez(source, **arrays)
        item["sha256"] = file_hash(source)
    protocol["candidates"][0]["max_steps"] = 1
    write_json(path, protocol)
    decisions = [
        approval(tmp_path, protocol, gate)
        for gate in ("description_target", "region_ranges", "compute_budget")
    ]
    study = create_study(path, tmp_path / "study", decisions)
    run_study(study["study_path"])
    result = lock_selection(study["study_path"])
    trial = read_json(result["selected"]["trial_path"])
    assert trial["best_step"] == 0
    assert trial["metrics"]["primary_score"] == 0
