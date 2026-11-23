"""Observed final-review issues, using small manufactured sources only."""

import numpy as np
import pytest

from element_surrogate import jobs
from element_surrogate.benchmark.protocol import file_hash, read_json
from element_surrogate.field_store import convert_cst_hdf5
from element_surrogate.fields import FieldMap, read_cst_hdf5
from element_surrogate.neural import forward_description
from tests.cst_format import write_cst_hdf5
from tests.test_neural import configuration
from tests.test_scientific_jobs import campaign, request, wait


def quad_source(path):
    axis = np.linspace(-0.01, 0.01, 7)
    z = np.linspace(-0.02, 0.02, 9)
    xg, yg, zg = np.meshgrid(axis, axis, z, indexing="ij")
    field = FieldMap(axis, axis, z, np.stack([3 * yg, 3 * xg, np.zeros_like(zg)], axis=-1))
    return write_cst_hdf5(path, field, field_unit="T")


@pytest.mark.parametrize("dataset", ["Mesh line x", "Mesh line y", "Mesh line z", "B-Field"])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("reader", ["eager", "streamed"])
def test_missing_units_fail_before_any_dataset_payload_read(
    tmp_path, monkeypatch, dataset, empty, reader
):
    import h5py

    source = quad_source(tmp_path / "source.h5")
    with h5py.File(source, "a") as h5:
        if empty:
            h5[dataset].attrs["unit"] = "  "
        else:
            del h5[dataset].attrs["unit"]
    before = file_hash(source)

    def payload(*args, **kwargs):
        raise AssertionError("Unit preflight read a dataset payload")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", payload)
    with pytest.raises(ValueError, match="requires a nonempty 'unit'") as problem:
        if reader == "eager":
            read_cst_hdf5(source)
        else:
            convert_cst_hdf5(source, tmp_path / "mapped", block_nodes=11)
    assert dataset in str(problem.value)
    assert "no units are assumed" in str(problem.value)
    assert not (tmp_path / "mapped").exists()
    assert file_hash(source) == before


@pytest.mark.parametrize("dimension", [4, 5, 6])
def test_architecture_prose_only_claims_coordinates_actually_modeled(dimension):
    description = forward_description(configuration(dimension)).lower()
    assert f"native {dimension}d" in description
    assert "optimizer-trainable parameters" in description
    assert "optimizer-trained" not in description
    assert ("momentum deviation passes through" in description) == (dimension > 4)
    assert ("arrival is input plus a learned increment" in description) == (dimension == 6)
    assert ("exclude absolute arrival" in description) == (dimension == 6)


@pytest.mark.parametrize("stated", [0.12, 1.2, -0.12])
def test_managed_description_matches_direct_scientific_result(tmp_path, stated):
    from element_surrogate.tools import compare_with_description

    source = quad_source(tmp_path / "source.h5")
    arguments = {
        "path": str(source),
        "order": 3,
        "fit_radius_m": 0.008,
        "description": {
            "kind": "quadrupole",
            "integrated_strength": stated,
            "aperture": {"radius_m": 0.008},
            "tolerance": 1e-5,
        },
    }
    expected = compare_with_description(**arguments)
    path, _ = campaign(tmp_path)
    spec = request(source)
    spec.update(operation="compare_with_description", arguments=arguments)
    row = wait(path, jobs.submit_request(path, spec))
    assert row["status"] == "complete", row
    actual = read_json(row["result_path"])
    assert isinstance(actual.pop("run_id"), str)  # Managed provenance is additional evidence.
    assert actual == expected
    assert bool(actual["findings"]) == (stated != 0.12)


def test_new_managed_helpers_are_single_registered_operations():
    from element_surrogate.tools import REGISTRY

    wanted = {
        "compare_with_description",
        "propose_training_coordinate_scales",
        "check_track_domain_isolation",
        "build_benchmark_report",
        "prepare_benchmark_metric_crosscheck",
    }
    assert wanted <= jobs.TOOL_OPERATIONS
    assert wanted <= REGISTRY.keys()
    assert wanted <= jobs.catalog()["operations"].keys()
