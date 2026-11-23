"""Physics, conservation, source binding, bounded generation and resume controls."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from element_surrogate.benchmark.protocol import protocol_identity, read_json
from element_surrogate.benchmark.scoring import _npz
from element_surrogate.data import export_benchmark_cohorts, generate, verify_generation
from element_surrogate.field_store import read_mapped_field, save_mapped_field
from element_surrogate.fields import load_field_file
from element_surrogate.transport import qualify_transport
from tests.workflow_fixture import (
    approvals,
    manufactured_transport,
    qualified_plan,
    readiness_for_test,
)


@pytest.mark.parametrize("dimension", [4, 5, 6])
def test_qualified_physical_labels_conservation_and_signed_rigidity(dimension):
    transport, levels, ranges, settings = manufactured_transport(dimension)
    settings.append({"id": "negative_charge", "effective_rigidity_tm": -2.0, "field_scale": 1.0})
    result = qualify_transport(
        transport,
        ranges,
        settings,
        levels,
        random_probes=0,
        seed=1,
        component_limits=[1e-8] * dimension,
        path_limit_m=1e-9,
    )
    assert result["accepted"]
    states = np.array([np.mean(ranges, axis=1), np.array(ranges)[:, 1]])
    positive, status, audit = transport.run(states, 2.0, 1.0)
    negative, _, _ = transport.run(states, -2.0, 1.0)
    assert status.reached_exit.all() and audit["all_sampled_stencils_supported"]
    assert not np.array_equal(positive[1, :4], negative[1, :4])
    if dimension > 4:
        np.testing.assert_array_equal(positive[:, -1], states[:, -1])
    if dimension == 6:
        assert positive[1, 4] > states[1, 4]


def test_actual_momentum_timing_not_effective_rigidity():
    transport, _, ranges, _ = manufactured_transport(6, longitudinal="time_s", gradient=0.0)
    state = np.array([np.mean(ranges, axis=1)])
    state[0, 5] = 0.02
    a, _, _ = transport.run(state, 2.0, 1.0)
    b, _, _ = transport.run(state, 7.0, 1.0)
    np.testing.assert_array_equal(a, b)
    assert a[0, 4] < state[0, 4]
    c, _, _ = replace(transport, actual_p0_kg_m_s=6e-19).run(state, 2.0, 1.0)
    assert c[0, 4] != a[0, 4]


def test_coarsening_changed_controls_and_wrong_production_level_refused():
    transport, levels, ranges, settings = manufactured_transport()
    options = {"random_probes": 0, "seed": 1, "component_limits": [1e-8] * 4, "path_limit_m": 1e-9}
    with pytest.raises(ValueError, match="decreasing"):
        qualify_transport(transport, ranges, settings, levels[::-1], **options)
    with pytest.raises(ValueError, match="finest"):
        qualify_transport(
            replace(transport, integration=levels[0]), ranges, settings, levels, **options
        )
    changed = copy.deepcopy(levels)
    changed[1]["max_path_m"] *= 2
    with pytest.raises(ValueError, match="non-refinement"):
        qualify_transport(transport, ranges, settings, changed, **options)


def test_generation_refuses_absent_approval_and_mutated_qualification(tmp_path):
    transport, qualification, plan = qualified_plan()
    with pytest.raises(ValueError, match="approval"):
        generate(transport, qualification, plan, tmp_path / "data", [])
    paths = approvals(tmp_path, plan)
    wrong = copy.deepcopy(plan)
    wrong["ranges"][0][1] *= 2
    with pytest.raises(ValueError, match="qualification"):
        generate(transport, qualification, wrong, tmp_path / "data", paths)
    wrong = copy.deepcopy(qualification)
    wrong["reports"][0]["probe_count"] = 123
    with pytest.raises(ValueError, match="receipt"):
        generate(transport, wrong, plan, tmp_path / "data", paths)


def test_generation_resume_exact_shards_and_no_test_array_access_on_export(tmp_path, monkeypatch):
    transport, qualification, plan = qualified_plan(5)
    paths = approvals(tmp_path, plan)
    first = generate(transport, qualification, plan, tmp_path / "resumed", paths, max_batches=1)
    assert first["status"] == "paused_at_batch_boundary"
    done = generate(transport, qualification, plan, tmp_path / "resumed", paths)
    assert done["status"] == "complete" and done["surviving_rows"] == 128
    other = generate(transport, qualification, plan, tmp_path / "whole", paths)
    assert other["surviving_rows"] == 128
    a, b = [verify_generation(tmp_path / n) for n in ("resumed", "whole")]
    for x, y in zip(a["shards"], b["shards"], strict=True):
        # NPZ timestamps can differ; numerical identity and exact IDs must not.
        for kind in ("proposals", "survivors"):
            with (
                np.load(tmp_path / "resumed" / x["files"][kind]["name"]) as left,
                np.load(tmp_path / "whole" / y["files"][kind]["name"]) as right,
            ):
                for key in left.files:
                    np.testing.assert_array_equal(left[key], right[key])
    original = np.load

    def forbid_test_arrays(path, *args, **kwargs):
        if "setting_000003" in str(path):
            raise AssertionError("sealed test arrays opened")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", forbid_test_arrays)
    exported = export_benchmark_cohorts(
        tmp_path / "resumed", tmp_path / "cohorts", maximum_rows_per_split=25
    )
    assert not exported["test_arrays_opened"]
    p = {"resources": {"max_npz_uncompressed_bytes": 1 << 24}}
    train = _npz(exported["splits"]["train"]["path"], p)
    assert len(train["sample_id"]) == 25
    assert len(set(train["setting_id"])) == 2
    with pytest.raises(AssertionError, match="sealed"):
        _npz(exported["splits"]["test"]["path"], p)


def test_committed_shard_mutation_and_changed_resume_decision_refused(tmp_path):
    transport, qualification, plan = qualified_plan()
    paths = approvals(tmp_path, plan)
    generate(transport, qualification, plan, tmp_path / "data", paths, max_batches=1)
    data = read_json(tmp_path / "data/generation.json")
    member = tmp_path / "data" / data["shards"][0]["files"]["survivors"]["name"]
    member.write_bytes(member.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="shard changed"):
        generate(transport, qualification, plan, tmp_path / "data", paths)


def test_proposal_budget_exhaustion_is_not_a_complete_dataset(tmp_path):
    transport, qualification, plan = qualified_plan()
    # All proposals start outside the declared numerical trajectory region.
    transport = replace(
        transport, trajectory_region=type(transport.trajectory_region)("circle", 0.0001)
    )
    ranges = plan["ranges"]
    # The central ray is retained, so finite-cohort qualification remains explicit.
    levels = [{**transport.integration, "step_m": s} for s in (0.02, 0.01, 0.005)]
    qualification = qualify_transport(
        transport,
        ranges,
        plan["settings"],
        levels,
        random_probes=0,
        seed=1,
        component_limits=[1e-8] * 4,
        path_limit_m=1e-9,
    )
    assert qualification["accepted"]
    plan.update(
        transport=transport.description(),
        qualification_sha256=protocol_identity(qualification),
        max_proposals_per_setting=32,
    )
    # A changed trajectory restriction needs a new scientific disposition too.
    with pytest.raises(ValueError, match="scope changed"):
        generate(transport, qualification, plan, tmp_path / "data", approvals(tmp_path, plan))
    assessment = readiness_for_test(transport)
    plan.update(field_readiness=assessment, field_readiness_sha256=protocol_identity(assessment))
    result = generate(transport, qualification, plan, tmp_path / "data", approvals(tmp_path, plan))
    assert result["status"] == "proposal_budget_exhausted"
    assert result["surviving_rows"] < 128


def test_mapped_store_exact_values_read_only_and_member_guard(tmp_path):
    transport, _, _, _ = manufactured_transport()
    result = save_mapped_field(transport.grid, tmp_path / "field", block_nodes=3)
    loaded = load_field_file(result["path"]).slot(None)
    np.testing.assert_array_equal(loaded.b_t, transport.grid.b_t)
    assert not loaded.b_t.flags.writeable
    assert isinstance(loaded.b_t.base, np.memmap)
    del loaded
    member = tmp_path / "field/b_t.npy"
    member.write_bytes(member.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="member changed"):
        read_mapped_field(result["path"])


def test_nonfinite_encoded_exit_cannot_be_counted_as_a_live_label():
    from element_surrogate.contracts import StaticMagneticCodec
    from element_surrogate.lorentz import Status

    class InvalidArrivalCodec(StaticMagneticCodec):
        def encode_exit(self, *args, **kwargs):
            result = super().encode_exit(*args, **kwargs)
            result[:, 4] = np.inf
            return result

    transport, _, ranges, _ = manufactured_transport(6)
    transport = replace(transport, codec=InvalidArrivalCodec(6, "path_m"))
    output, result, _ = transport.run(np.array([np.mean(ranges, axis=1)]), 2.0, 1.0)
    assert result.status[0] == Status.NON_FINITE
    assert not result.reached_exit.any() and np.isnan(output).all()


@pytest.mark.parametrize("max_batches", [None, 1])
def test_source_change_invalidates_the_campaign_and_cannot_be_resumed(tmp_path, max_batches):
    transport, qualification, plan = qualified_plan()
    original = transport.run
    saved = transport.grid.b_t.copy()

    def changed_after_tracking(*args, **kwargs):
        result = original(*args, **kwargs)
        transport.grid.b_t[0, 0, 0, 0] += 1e-6
        return result

    transport.run = changed_after_tracking
    paths = approvals(tmp_path, plan)
    with pytest.raises(ValueError, match="transport"):
        generate(transport, qualification, plan, tmp_path / "data", paths, max_batches=max_batches)
    assert read_json(tmp_path / "data/generation.json")["status"] == "invalidated"
    transport.grid.b_t[:] = saved
    transport.run = original
    with pytest.raises(ValueError, match="invalidated"):
        generate(transport, qualification, plan, tmp_path / "data", paths)
