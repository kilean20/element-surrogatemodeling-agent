"""Scientific and installed-asset failures reproduced by the independent audit."""

import re
import tomllib
from pathlib import Path

import numpy as np
import pytest

from element_surrogate import neural
from element_surrogate.fields import FieldMap
from element_surrogate.readiness import cell_residuals, node_defects
from element_surrogate.runs import RunStore, record_call
from element_surrogate.tools import REGISTRY


def test_exact_vacuum_zeros_and_invariant_planes_are_contextual_observations():
    axis = np.linspace(-0.01, 0.01, 7)
    z = np.linspace(-0.05, 0.05, 9)
    x, y, zz = np.meshgrid(axis, axis, z, indexing="ij")
    grid = FieldMap(axis, axis, z, np.stack((3 * y, 3 * x, np.zeros_like(zz)), axis=-1))
    residuals = cell_residuals(grid)
    assert np.max(np.abs(residuals.divergence_t_per_m)) < 1e-12
    assert np.max(residuals.curl_t_per_m) < 1e-12
    report = node_defects(grid)
    assert report["zero_nodes_next_to_nonzero"]["count"] == 9
    assert len(report["duplicated_planes"]) == 8
    assert report["findings"] == []
    assert {row["kind"] for row in report["observations"]} == {"zero_vectors", "duplicated_plane"}
    assert all("context" in row["criterion"] for row in report["observations"])


def test_failed_input_hash_has_a_durable_error_record(tmp_path):
    store = RunStore(tmp_path)
    missing = tmp_path / "absent.table"
    with pytest.raises(FileNotFoundError):
        record_call(store, REGISTRY["inspect_file"], {"path": str(missing)})
    assert len(store.list_runs()) == 1
    record = store.load(store.list_runs()[0])
    assert record["status"] == "error"
    assert record["error_stage"] == "input_validation"
    assert record["arguments"]["path"] == str(missing)


def test_a_formatted_but_unbound_symmetry_hash_is_refused():
    config = {
        "schema": "element_surrogate.static_magnetic_mlp.v1",
        "dimension": 4,
        "hidden_widths": [8],
        "base": "train_affine",
        "dtype": "float64",
        "device": "cpu",
        "threads": 1,
        "batch_size": 8,
        "inference_batch_size": 8,
        "learning_rate": 0.001,
        "loss_scales": [1, 1, 1, 1],
        "reflection_group": [[1] * 6, [-1, -1, 1, 1, 1, 1]],
        "symmetry_evidence_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="bound symmetry"):
        neural._configuration(config)


def test_explicit_skill_data_files_close_every_local_reference():
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    shipped = {
        path.relative_to(root).as_posix()
        for patterns in config["tool"]["setuptools"]["data-files"].values()
        for pattern in patterns
        for path in root.glob(pattern)
    }
    missing = []
    for name in sorted(shipped):
        path = root / name
        if path.suffix != ".md":
            continue
        for link in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if link.startswith(("http:", "https:", "mailto:", "#")):
                continue
            target = (path.parent / link.split("#")[0]).resolve()
            assert target.is_relative_to(root)
            if target.relative_to(root).as_posix() not in shipped:
                missing.append((name, link))
    assert not missing, missing
