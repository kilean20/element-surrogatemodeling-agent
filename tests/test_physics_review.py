"""Physics opportunities cannot be silently equated with completed comparisons."""

from copy import deepcopy

import pytest

from element_surrogate.benchmark.protocol import file_hash, read_json, write_json
from element_surrogate.physics_review import TOPICS, require_before_fit, review, summarize
from tests.benchmark_fixture import make_case


def declared_review():
    return {
        "topics": {
            k: {
                "disposition": "deferred",
                "candidate_ids": [],
                "rationale": "This topic remains unmeasured in the manufactured study.",
                "evidence": [],
            }
            for k in TOPICS
        }
    }


def test_absent_review_and_deferred_topics_are_not_complete(tmp_path):
    path, *_ = make_case(tmp_path)
    p = read_json(path)
    assert summarize(p)["status"] == "absent"
    p["physics"]["physics_review_required"] = True
    with pytest.raises(ValueError, match="before fitting"):
        require_before_fit(p)
    record = review(p, declared_review(), tmp_path)
    p["physics"]["modeling_review"] = record
    result = summarize(p)
    assert result["status"] == "incomplete"
    assert set(result["unresolved_topics"]) == set(TOPICS)
    assert require_before_fit(p)["status"] == "incomplete"
    changed = deepcopy(p)
    changed["candidates"][0]["rationale"] += " changed"
    with pytest.raises(ValueError, match="differs"):
        summarize(changed)


def test_field_parity_cannot_be_used_as_a_rejected_map_test(tmp_path):
    protocol_path, *_ = make_case(tmp_path)
    p = read_json(protocol_path)
    path = tmp_path / "parity.json"
    write_json(path, {"x_mirror": {"Bx": {"even_defect": 1e-5}}})
    request = declared_review()
    request["topics"]["symmetry"].update(
        disposition="measured_not_supported",
        evidence=[{"path": str(path), "sha256": file_hash(path)}],
    )
    with pytest.raises(ValueError, match="parity alone"):
        review(p, request, tmp_path)
    request["topics"]["symmetry"]["disposition"] = "established_contract"
    with pytest.raises(ValueError, match="measured comparison"):
        review(p, request, tmp_path)


def test_missing_candidate_and_unmeasured_claim_are_refused(tmp_path):
    protocol_path, *_ = make_case(tmp_path)
    p = read_json(protocol_path)
    request = declared_review()
    request["topics"]["symmetry"].update(
        disposition="compared", candidate_ids=[c["id"] for c in p["candidates"]]
    )
    with pytest.raises(ValueError, match="structured and unconstrained"):
        review(p, request, tmp_path)
    request = declared_review()
    request["topics"].pop("limiting_cases")
    with pytest.raises(ValueError, match="every physics topic"):
        review(p, request, tmp_path)


def test_dispositions_and_evidence_remain_bound(tmp_path):
    protocol_path, *_ = make_case(tmp_path)
    p = read_json(protocol_path)
    path = tmp_path / "declared_contract.json"
    write_json(path, {"schema": "manufactured-control-contract", "actual_momentum_separate": True})
    request = declared_review()
    request["topics"]["controls"].update(
        disposition="established_contract",
        evidence=[{"path": str(path), "sha256": file_hash(path)}],
    )
    p["physics"]["modeling_review"] = review(p, request, tmp_path)
    assert "controls" not in summarize(p)["unresolved_topics"]
    p["physics"]["modeling_review"]["coverage_status"] = "recorded"
    with pytest.raises(ValueError, match="coverage or dispositions"):
        summarize(p)
    p["physics"]["modeling_review"] = review(p, request, tmp_path)
    write_json(path, {"changed": True})
    with pytest.raises(ValueError, match="SHA256"):
        summarize(p)


def test_real_measured_symmetry_comparison_and_rejection(tmp_path):
    from dataclasses import replace

    from element_surrogate import symmetry
    from tests.test_equivariant_features import setup
    from tests.workflow_fixture import manufactured_transport

    path, *_ = make_case(tmp_path)
    p = read_json(path)
    config, _, _ = setup()
    t, _, ranges, _ = manufactured_transport()
    p["physics"]["transport_description"] = t.description()
    p["candidates"][0]["configuration"] = {
        **config,
        "model_family": "mlp",
        "reflection_group": [[1] * 6],
    }
    p["candidates"][0]["configuration"].pop("symmetry_assessment")
    p["candidates"][0]["configuration"].pop("symmetry_evidence_sha256")
    p["candidates"][1]["configuration"] = config
    request = declared_review()
    request["topics"]["symmetry"].update(
        disposition="compared", candidate_ids=[c["id"] for c in p["candidates"]]
    )
    p["physics"]["modeling_review"] = review(p, request, tmp_path)
    assert "symmetry" not in summarize(p)["unresolved_topics"]

    b = t.grid.b_t.copy()
    b[..., 0] += 0.02
    t = replace(t, grid=t.grid.with_values(b))
    measured = symmetry.assess(
        t,
        ranges,
        [{"id": "control", "effective_rigidity_tm": 2.0, "field_scale": 1.0}],
        config["reflection_group"],
        [1e-8] * 4,
        random_probes=0,
        seed=41,
        rationale="Manufactured real dipole breaks the proposed reflection",
    )
    assert not measured["supported"]
    evidence = tmp_path / "rejected_map_symmetry.json"
    write_json(evidence, measured)
    p["physics"]["transport_description"] = t.description()
    request = declared_review()
    request["topics"]["symmetry"].update(
        disposition="measured_not_supported",
        evidence=[{"path": str(evidence), "sha256": file_hash(evidence)}],
    )
    p["physics"]["modeling_review"] = review(p, request, tmp_path)
    assert "symmetry" not in summarize(p)["unresolved_topics"]
    measured["reports"][0]["status_changes"] = -1
    write_json(evidence, measured)
    request["topics"]["symmetry"]["evidence"][0]["sha256"] = file_hash(evidence)
    with pytest.raises(ValueError, match="survivor/status"):
        review(p, request, tmp_path)
