"""Manufactured controls for portable cohorts, card inputs and actual model prose."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from element_surrogate.benchmark.protocol import file_hash, write_json
from element_surrogate.card_metadata import inspect_request, model_sections, software_identity
from element_surrogate.contracts import StaticMagneticCodec, coordinate_metadata
from element_surrogate.scaling import output_scales, propose_from_cohort


def descriptor(tmp_path, dimension=4):
    rng = np.random.default_rng(203)
    x = rng.normal(0, 0.01, (107, dimension + 2))
    x[:, -2:] = [2, 0.3]
    if dimension > 4:
        x[:, dimension - 1] = 0.1
    y = x[:, :dimension] * 1.3
    if dimension == 6:
        y[:, 4] = x[:, 4] + 0.002
    members = []
    for i, sl in enumerate((slice(0, 31), slice(31, 69), slice(69, None))):
        p = tmp_path / f"shard{i}.npz"
        np.savez(
            p,
            inputs=x[sl],
            targets=y[sl],
            sample_id=np.arange(len(x))[sl],
            setting_id=np.full(len(x[sl]), "synthetic"),
        )
        members.append({"path": p.name, "rows": len(x[sl]), "sha256": file_hash(p)})
    coordinates = coordinate_metadata(StaticMagneticCodec(dimension=dimension))
    names, units = [c["name"] for c in coordinates], [c["unit"] for c in coordinates]
    d = {
        "schema": "element_surrogate.cohort_shards.v1",
        "rows": len(x),
        "split": "train",
        "members": members,
        "metadata": {
            "coordinate_names": names,
            "coordinate_units": units,
            "input_names": names + ["effective_rigidity_tm", "field_scale"],
            "input_units": units + ["T*m", "1"],
        },
    }
    write_json(tmp_path / "train.json", d)
    return tmp_path / "train.json", d, x, y


@pytest.mark.parametrize("dimension", [4, 5, 6])
def test_streaming_scales_use_all_rows_and_match_full_array(tmp_path, dimension):
    path, d, x, y = descriptor(tmp_path, dimension)
    result = propose_from_cohort(path, dimension, tmp_path / "scales.json", batch_rows=7)
    np.testing.assert_allclose(
        result["output_normalization"]["scales_native"],
        output_scales(x, y, dimension)[0],
        rtol=1e-14,
    )
    assert result["output_normalization"]["rows"] == 107
    assert not result["output_normalization"]["validation_or_test_used"]
    d["members"].reverse()
    write_json(path, d)
    other = propose_from_cohort(path, dimension, tmp_path / "other.json", batch_rows=19)
    np.testing.assert_allclose(
        other["output_normalization"]["scales_native"],
        result["output_normalization"]["scales_native"],
        rtol=1e-14,
    )
    np.savez(tmp_path / "legacy.npz", inputs=x, targets=y)
    legacy = propose_from_cohort(tmp_path / "legacy.npz", dimension, tmp_path / "legacy.json")
    np.testing.assert_allclose(
        legacy["output_normalization"]["scales_native"],
        result["output_normalization"]["scales_native"],
        rtol=1e-14,
    )
    assert "not authenticated" in legacy["training_scope"]
    renamed = tmp_path / "numeric-cohort.bin"
    renamed.write_bytes((tmp_path / "legacy.npz").read_bytes())
    named = propose_from_cohort(renamed, dimension, tmp_path / "named.json")
    np.testing.assert_allclose(
        named["output_normalization"]["scales_native"],
        legacy["output_normalization"]["scales_native"],
        rtol=1e-14,
    )


@pytest.mark.parametrize("defect", ["hash", "units", "dimension", "split", "rows", "duplicate"])
def test_bad_shards_refuse_without_publishing_scales(tmp_path, defect):
    p, d, _, _ = descriptor(tmp_path)
    if defect == "hash":
        d["members"][0]["sha256"] = "0" * 64
    elif defect == "units":
        d["metadata"]["coordinate_units"][0] = "mm"
    elif defect == "dimension":
        d["metadata"]["coordinate_names"].pop()
    elif defect == "split":
        d["split"] = "test"
    elif defect == "rows":
        d["rows"] += 1
    elif defect == "duplicate":
        d["members"].append(deepcopy(d["members"][0]))
    write_json(p, d)
    with pytest.raises(ValueError):
        propose_from_cohort(p, 4, tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


def test_large_shard_is_refused_before_decompression_not_subsampled(tmp_path):
    p, _, _, _ = descriptor(tmp_path)
    with pytest.raises(ValueError, match="allowance"):
        propose_from_cohort(p, 4, tmp_path / "out.json", working_memory_bytes=3000)
    assert not (tmp_path / "out.json").exists()


def test_frozen_protocol_binding_rejects_a_different_split_or_identity(tmp_path):
    from tests.benchmark_fixture import make_case

    protocol_path, protocol, _ = make_case(tmp_path / "protocol")
    path, d, _, _ = descriptor(tmp_path)
    d["metadata"]["dataset_identity_sha256"] = protocol["dataset"]["identity_sha256"]
    write_json(path, d)
    for coordinate, unit, name in zip(
        protocol["coordinates"],
        d["metadata"]["coordinate_units"],
        d["metadata"]["coordinate_names"],
        strict=True,
    ):
        coordinate["name"], coordinate["unit"] = name, unit
    protocol["inputs"] = d["metadata"]["input_names"]
    protocol["input_units"] = d["metadata"]["input_units"]
    protocol["resources"]["max_rows_per_split"] = d["rows"]
    protocol["dataset"]["splits"]["train"].update(
        path=str(path),
        sha256=file_hash(path),
        rows=d["rows"],
        groups=["synthetic"],
    )
    write_json(protocol_path, protocol)
    result = propose_from_cohort(path, 4, tmp_path / "bound.json", protocol_path=protocol_path)
    assert result["training_scope"] == "frozen protocol training identity"
    d["rows"] += 1
    write_json(path, d)
    with pytest.raises(ValueError):
        propose_from_cohort(path, 4, tmp_path / "changed.json", protocol_path=protocol_path)


def test_scale_sums_do_not_overflow_and_object_arrays_are_never_unpickled(tmp_path):
    x = np.zeros((3, 6))
    x[:, -2:] = [1.0, 1.0]
    y = np.zeros((3, 4))
    y[0, 0] = 1e300
    np.savez(tmp_path / "extreme.npz", inputs=x, targets=y)
    r = propose_from_cohort(tmp_path / "extreme.npz", 4, tmp_path / "extreme.json", batch_rows=1)
    assert r["output_normalization"]["scales_native"][0] == pytest.approx(1e300 / np.sqrt(3))
    assert r["output_normalization"]["degenerate_native_unit_fallback_components"] == [1, 2, 3]
    np.savez(tmp_path / "objects.npz", inputs=np.array([{}], dtype=object), targets=y)
    with pytest.raises(ValueError, match="Object arrays"):
        propose_from_cohort(tmp_path / "objects.npz", 4, tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()


def test_card_headings_are_normalized_and_unknown_or_colliding_keys_refuse():
    normalized, coverage = model_sections({"sections": {" Model Description. ": "Exact facts"}})
    assert normalized == {"model description": "Exact facts"}
    assert coverage["narrative_sections_applied"] == ["model description"]
    assert coverage["unused_sections"] == []
    for sections in (
        {"Made up section": "x"},
        {"Model description": "a", "model description": "b"},
    ):
        review = inspect_request({"model": {"sections": sections}})
        assert not review["valid"] and any(
            f["code"] == "model_sections" for f in review["findings"]
        )


def test_explicit_wrong_npz_units_are_not_ignored(tmp_path):
    _, d, x, y = descriptor(tmp_path)
    metadata = d["metadata"]
    metadata["coordinate_units"][0] = "mm"
    np.savez(tmp_path / "wrong.npz", inputs=x, targets=y, **metadata)
    with pytest.raises(ValueError, match="m/rad"):
        propose_from_cohort(tmp_path / "wrong.npz", 4, tmp_path / "no.json")


def test_prerelease_identity_is_faithful_without_a_fake_numeric_release():
    identity = software_identity("manufactured-tool", "3.1.0rc7")
    assert "version" not in identity
    assert identity["identifier"]["value"] == "manufactured-tool@3.1.0rc7"
    assert software_identity("manufactured-tool", "3.1.0")["version"] == "3.1.0"


def test_zero_neural_correction_and_fitted_base_are_documented_from_weights(tmp_path):
    pytest.importorskip("torch")
    from element_surrogate.model_documentation import selected_description
    from element_surrogate.neural import build_adapter
    from tests.test_scaling import config

    _, _, x, y = descriptor(tmp_path)
    c = config(4, "train_polynomial")
    c["base_degree"] = 2
    c["max_base_features"] = 21
    model = build_adapter(x, y, c, 11)
    path = tmp_path / "model.bin"
    model.save(path)
    trial = {
        "candidate": {"configuration": c},
        "checkpoint_sha256": file_hash(path),
        "steps": 50,
        "best_step": 0,
    }
    result = selected_description(trial, path)
    assert result["neural_output_layer_exactly_zero"]
    assert result["train_fitted_base_coefficients"] == 84
    assert "polynomial" in result["short_description"]
    assert result["completed_optimizer_updates"] == 50 and not result["convergence_established"]
    assert "does not establish neural improvement" in result["selection_interpretation"]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        selected_description(trial, path)


def test_software_schema_findings_remain_separate_from_missing_human_facts():
    front = {
        "discoverability": {
            "datacard": {
                "created_by": [
                    {"creator": {"software": {"name": "manufactured", "version": "3.1.0rc7"}}}
                ]
            }
        }
    }
    result = inspect_request({"datasets": {"field": {"frontmatter": front}}})
    assert result["valid"]  # valid local draft metadata, not full upstream conformance
    role = result["roles"]["field"]
    assert len(role["metadata_missing"]) == 6
    assert role["software_schema_findings"][0]["category"] == "software_schema_compatibility"
    assert not result["release_metadata_ready"]


def test_actual_pinned_validator_accepts_exact_prerelease_identifier(tmp_path):
    pytest.importorskip("linkml")
    from element_surrogate import cards
    from tests.test_cards import core_metadata

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    write_json(dataset / "synthetic.json", {"kind": "manufactured"})
    output = tmp_path / "cards"
    output.mkdir()
    metadata = core_metadata()
    metadata["frontmatter"]["discoverability"]["datacard"].pop("created_by")
    result = cards._data_card("field", dataset, metadata, output, "Manufactured provenance")
    assert result["schema_pass"], Path(output / "field_validation.log").read_text()
    import yaml

    front = yaml.safe_load((output / "field_frontmatter.yaml").read_text())
    creator = front["discoverability"]["datacard"]["created_by"][0]["creator"]["software"]
    from element_surrogate import __version__

    assert creator["identifier"]["value"].endswith("@" + __version__)
    assert "version" not in creator
