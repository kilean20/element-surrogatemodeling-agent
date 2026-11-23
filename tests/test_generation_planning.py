"""Computed contracts, observed costs and early separate dataset documentation."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from element_surrogate.benchmark.protocol import read_json, write_json
from element_surrogate.data import generate
from element_surrogate.domain_report import build_domain_report
from element_surrogate.generation_planning import build_request, capacity_report
from element_surrogate.region import Region
from element_surrogate.transport import qualify_transport
from tests.workflow_fixture import approvals, manufactured_transport, qualified_plan


def test_request_builder_uses_computed_payload_and_refuses_setting_guess(tmp_path):
    from tests.test_workflow_tools import file_case

    path, _, plan = file_case(tmp_path)
    # file_case deliberately leaves split/count decisions out; supply them here.
    _, _, _, settings = manufactured_transport()
    request = {
        k: plan[k]
        for k in (
            "seed",
            "batch_size",
            "max_proposals_per_setting",
            "max_wall_seconds",
            "material_boundary_evidence",
        )
    }
    request.update(
        schema="element_surrogate.generation_request.v1",
        settings=[{k: s[k] for k in ("id", "split", "surviving_rows")} for s in settings],
    )
    write_json(tmp_path / "request.json", request)
    write_json(tmp_path / "readiness.json", plan["field_readiness"])
    result = build_request(
        path,
        tmp_path / "qualification.json",
        tmp_path / "readiness.json",
        tmp_path / "request.json",
        tmp_path / "computed.json",
    )
    actual = read_json(result["plan_path"])
    assert actual["transport"] == plan["transport"]
    assert actual["ranges"] == plan["ranges"]
    assert actual["settings"] == settings and not result["authority_created"]
    assert result["total_requested_survivors"] == 128
    request["settings"][0]["effective_rigidity_tm"] = 99
    write_json(tmp_path / "bad.json", request)
    with pytest.raises(ValueError, match="qualified IDs"):
        build_request(
            path,
            tmp_path / "qualification.json",
            tmp_path / "readiness.json",
            tmp_path / "bad.json",
            tmp_path / "bad_output.json",
        )
    assert not (tmp_path / "bad_output.json").exists()


def test_batch_costs_bound_to_commits_and_untested_settings_not_extrapolated(tmp_path):
    transport, qualification, plan = qualified_plan()
    decisions = approvals(tmp_path / "decisions", plan)
    output = tmp_path / "particles"
    generate(transport, qualification, plan, output, decisions, max_batches=2)
    result = capacity_report(output, tmp_path / "capacity.json")
    assert result["settings"][0]["measured_batches"] == 2
    assert result["settings"][0]["remaining_batch_time_projection_seconds"] is not None
    assert result["settings"][1]["remaining_batch_time_projection_seconds"] is None
    assert "requires actual" in result["million_row_acceptance"]
    metrics = sorted((output / "batch_metrics").glob("*.json"))
    metrics[0].unlink()
    result = capacity_report(output, tmp_path / "missing.json")
    assert len(result["committed_batches_without_timing"]) == 1
    assert result["settings"][0]["remaining_batch_time_projection_seconds"] is None
    edited = read_json(metrics[1])
    edited["shard_sha256"] = "0" * 64
    write_json(metrics[1], edited)
    with pytest.raises(ValueError, match="bound to the committed"):
        capacity_report(output, tmp_path / "changed.json")


@pytest.mark.parametrize(
    "dimensions",
    [
        ((0.014, 0.007), (0.014, 0.012)),
        ((0.007, 0.014), (0.012, 0.014)),
        ((0.008, 0.006), (0.012, 0.010)),
    ],
)
def test_nested_rectangles_allow_one_or_both_dimensions_to_grow(tmp_path, dimensions):
    transport, refinements, ranges, settings = manufactured_transport()
    paths = []
    for i, (width, height) in enumerate(dimensions):
        candidate = replace(
            transport, trajectory_region=Region("rectangle", width, half_height_m=height)
        )
        report = qualify_transport(
            candidate,
            ranges,
            settings,
            refinements,
            random_probes=0,
            seed=7,
            component_limits=[1e-8] * 4,
            path_limit_m=1e-9,
        )
        path = tmp_path / f"q{i}.json"
        write_json(path, report)
        paths.append(path)
    result = build_domain_report(paths, tmp_path / "domain")
    assert result["recommendation"]["radius_m"] == dimensions[-1][0]
    assert result["recommendation"]["half_height_m"] == dimensions[-1][1]
    plot = Path(result["plot_path"]).read_text(encoding="utf-8")
    assert (
        "Vertical half-height" if dimensions[0][0] == dimensions[1][0] else "horizontal half-width"
    ) in plot
    a, b = [read_json(p) for p in paths]
    b["scope"]["transport"]["trajectory_region"]["z_min_m"] = -0.09
    write_json(paths[1], b)
    with pytest.raises(ValueError, match="longitudinal bounds"):
        build_domain_report(paths, tmp_path / "bad_bounds")
    b = deepcopy(a)
    b["scope"]["transport"]["trajectory_region"].update(
        radius_m=dimensions[0][0] * 1.1, half_height_m=dimensions[0][1] * 0.9
    )
    write_json(paths[1], b)
    mixed = build_domain_report(paths, tmp_path / "non_nested")
    assert mixed["recommendation"] is None
    assert len(mixed["maximal_passing_candidates"]) == 2
    assert mixed["selection_objective_needed"]


def test_partial_particle_and_field_drafts_without_completed_model(tmp_path):
    pytest.importorskip("linkml")
    from element_surrogate.cards import prepare_dataset_draft
    from element_surrogate.field_store import save_mapped_field
    from tests.test_cards import core_metadata

    transport, qualification, plan = qualified_plan()
    output = tmp_path / "particles"
    generate(
        transport,
        qualification,
        plan,
        output,
        approvals(tmp_path / "decisions", plan),
        max_batches=1,
    )
    request = tmp_path / "metadata.json"
    particle_meta = core_metadata("particle")
    particle_meta["frontmatter"]["discoverability"]["identification"]["name"] = (
        "Manufactured partial particles"
    )
    write_json(request, {"datasets": {"particle": particle_meta}})
    result = prepare_dataset_draft(
        "particle", output, output / "generation.json", request, tmp_path / "particle_card"
    )
    assert result["schema_pass"] and result["lifecycle_status"] == "paused_at_batch_boundary"
    trace = read_json(result["traceability_path"])
    assert trace["computed"]["committed_survivors"] == 17
    assert not trace["publication_performed"]
    field = save_mapped_field(transport.grid, tmp_path / "field", block_nodes=21)
    write_json(request, {"datasets": {"field": core_metadata("field")}})
    result = prepare_dataset_draft(
        "field", tmp_path / "field", field["path"], request, tmp_path / "field_card"
    )
    assert result["schema_pass"]
    assert read_json(result["traceability_path"])["computed"]["grid_shapes"] == [
        list(transport.grid.shape)
    ]


def test_draft_rejects_evidence_mutation_during_introspection(tmp_path, monkeypatch):
    pytest.importorskip("linkml")
    from element_surrogate import cards
    from element_surrogate.field_store import save_mapped_field
    from tests.test_cards import core_metadata

    transport = manufactured_transport()[0]
    field = save_mapped_field(transport.grid, tmp_path / "field", block_nodes=21)
    evidence = Path(field["path"])
    metadata = tmp_path / "metadata.json"
    write_json(metadata, {"datasets": {"field": core_metadata("field")}})

    def mutation(*args):
        evidence.write_bytes(evidence.read_bytes() + b"\n")
        return {"schema_pass": True}

    monkeypatch.setattr(cards, "_data_card", mutation)
    with pytest.raises(ValueError, match="evidence changed"):
        cards.prepare_dataset_draft(
            "field", tmp_path / "field", evidence, metadata, tmp_path / "draft"
        )
    assert not (tmp_path / "draft/result.json").exists()
