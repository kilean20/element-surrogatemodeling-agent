"""All three structured/general families use one physical accuracy/cost protocol."""

from copy import deepcopy

import pytest

pytest.importorskip("torch")

from element_surrogate import canonical, theory
from element_surrogate.benchmark.protocol import read_json
from element_surrogate.benchmark.study import create_study, lock_selection, run_study
from element_surrogate.data import generate
from element_surrogate.modeling import prepare_generated_study
from tests.test_physical_workflow import study_request
from tests.workflow_fixture import approvals, qualified_plan


def test_structured_candidates_share_metrics_loss_and_full_cost(tmp_path):
    transport, qualification, plan = qualified_plan(4, continuous_coupling=True, gradient=0.0)
    generate(
        transport,
        qualification,
        plan,
        tmp_path / "data",
        approvals(tmp_path / "generation-approvals", plan),
    )
    request = study_request(4)
    plain = request["candidates"][0]
    plain["max_steps"] = 3
    plain["configuration"]["hidden_widths"] = [8]
    residual = deepcopy(plain)
    residual["id"] = "native_drift_residual"
    residual["architecture"] = "native drift residual MLP"
    residual["configuration"]["base"] = "theory_drift"
    residual["configuration"]["theory_contract"] = theory.assess(
        transport, "drift", "Manufactured drift base control"
    )
    shear = deepcopy(plain)
    shear["id"] = "canonical_shear"
    shear["architecture"] = "conditional canonical potential shears"
    shear["configuration"].update(
        model_family="canonical_shear",
        shear_pairs=2,
        canonical_contract=canonical.assess(
            transport,
            plan["ranges"],
            plan["settings"],
            q_scale_m=0.01,
            p_scale=0.03,
            rationale="Manufactured canonical control",
            effective_coupling_range=plan["effective_coupling_range"],
        ),
    )
    request["candidates"] = [plain, residual, shear]
    request["resources"]["max_study_wall_seconds"] = 60 + sum(
        c["max_wall_seconds"] * len(c["seeds"]) for c in request["candidates"]
    )
    prepared = prepare_generated_study(tmp_path / "data", tmp_path / "protocol", request)
    protocol = read_json(prepared["protocol_path"])
    study = create_study(
        prepared["protocol_path"],
        tmp_path / "study",
        approvals(tmp_path / "study-approvals", protocol),
    )["study_path"]
    result = run_study(study)
    assert all(r["status"] == "complete" for r in result["trials"])
    selected = lock_selection(study)
    assert selected["selected"]["candidate_id"] in {
        "mlp",
        "native_drift_residual",
        "canonical_shear",
    }
    for trial in result["trials"]:
        record = read_json(trial["trial_path"])
        assert record["metrics"]["coordinate_units"] == ["m", "rad", "m", "rad"]
        assert record["timing"]["contract"] == protocol["timing"]
        assert record["timing"]["median_us_per_row"] > 0
    assert not (tmp_path / "study/test").exists()
