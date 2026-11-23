"""Optional end-to-end field -> qualified labels -> neural study -> native export."""

import numpy as np
import pytest

pytest.importorskip("torch")

from element_surrogate.benchmark.protocol import read_json
from element_surrogate.benchmark.report import build_report
from element_surrogate.benchmark.study import create_study, evaluate_test, lock_selection, run_study
from element_surrogate.data import generate, verify_generation
from element_surrogate.export import ElementSurrogate, export_selected
from element_surrogate.modeling import prepare_generated_study
from tests.workflow_fixture import approvals, qualified_plan

# Includes interpreter/library startup in these correctness fixtures only.
# Scientific job budgets and dedicated timeout tests are unchanged.
WORKER_WALL_SECONDS = 120


def study_request(dimension):
    scales = [1e-4] * dimension
    if dimension == 6:
        scales[4] = 1e-6
    if dimension > 4:
        scales[-1] = 1e-4
    candidates = []
    for name, base in (("mlp", "none"), ("affine_residual", "train_affine")):
        candidates.append(
            {
                "id": name,
                "architecture": name,
                "kind": "stochastic",
                "seeds": [3, 7],
                "configuration": {
                    "schema": "element_surrogate.static_magnetic_mlp.v1",
                    "dimension": dimension,
                    "hidden_widths": [16, 16],
                    "base": base,
                    "dtype": "float64",
                    "device": "cpu",
                    "threads": 1,
                    "batch_size": 32,
                    "inference_batch_size": 32,
                    "learning_rate": 0.005,
                    "loss_scales": [0.003, 0.01, 0.003, 0.01]
                    + ([1e-5] if dimension == 6 else [])
                    + ([0.03] if dimension > 4 else []),
                },
                "rationale": (
                    "Manufactured vacuum quadrupole: compare a general neural map "
                    "with a train-only affine base for dominant linear transport."
                ),
                "forward_description": (
                    "Native tanh network with optional train-only affine base, "
                    "conserved momentum and arrival-translation behavior."
                ),
                "max_steps": 120,
                "max_wall_seconds": WORKER_WALL_SECONDS,
            }
        )
    return {
        "name": "manufactured-physical-map",
        "physics_review": {
            "topics": {
                k: {
                    "disposition": "deferred",
                    "candidate_ids": [],
                    "rationale": (
                        "Manufactured numerical pipeline control, not an architecture-search"
                        " qualification."
                    ),
                    "evidence": [],
                }
                for k in __import__("element_surrogate.physics_review", fromlist=["TOPICS"]).TOPICS
            }
        },
        "maximum_rows_per_split": 64,
        "error_scales": scales,
        "candidates": candidates,
        "access_and_reuse": "generated synthetic acceptance data only",
        "software": "Python/NumPy and optional PyTorch; worker versions and source hashes recorded",
        "selection": {
            "primary": "worst_scaled_setting_mean_rmse",
            "relative_accuracy_tie": 0.01,
            "incomplete_policy": "refuse",
        },
        "resources": {
            "max_rows_per_split": 64,
            "max_npz_uncompressed_bytes": 1 << 24,
            "max_checkpoint_bytes": 1 << 20,
            "max_study_wall_seconds": 60
            + sum(c["max_wall_seconds"] * len(c["seeds"]) for c in candidates),
            "max_test_wall_seconds": WORKER_WALL_SECONDS,
        },
        "timing": {
            "batch_size": 16,
            "repetitions": 3,
            "warmup_calls": 1,
            "threads": 1,
            "input_dtype": "float64",
            "scope": "adapter_predict_call",
            "hardware": "synthetic acceptance host",
            "device": "cpu",
            "synchronization": "synchronous_cpu",
            "golden_atol": [1e-12] * dimension,
            "included_operations": (
                "complete native neural map, normalization, base and conserved coordinates"
            ),
            "excluded_operations": "loading, fitting, input file reads",
        },
    }


@pytest.mark.parametrize(
    "dimension,dtype", [(4, "float64"), (5, "float64"), (6, "float64"), (6, "float32")]
)
def test_physical_pipeline_withheld_test_and_export(tmp_path, dimension, dtype, monkeypatch):
    transport, qualification, plan = qualified_plan(dimension, continuous_coupling=True)
    generate(
        transport,
        qualification,
        plan,
        tmp_path / "data",
        approvals(tmp_path / "generation_decisions", plan),
    )
    request = study_request(dimension)
    for candidate in request["candidates"]:
        candidate["configuration"]["dtype"] = dtype
    request["timing"]["input_dtype"] = dtype
    tolerance = 1e-12 if dtype == "float64" else 1e-7
    request["timing"]["golden_atol"] = [tolerance] * dimension
    prepared = prepare_generated_study(tmp_path / "data", tmp_path / "benchmark", request)
    protocol = read_json(prepared["protocol_path"])
    decisions = approvals(tmp_path / "study_decisions", protocol)
    created = create_study(prepared["protocol_path"], tmp_path / "study", decisions)
    # Move the numeric test files away: training and selection must still work.
    record = verify_generation(tmp_path / "data")
    moved = []
    for shard in record["shards"]:
        if shard["split"] == "test":
            source = tmp_path / "data" / shard["files"]["survivors"]["name"]
            target = source.with_suffix(".sealed")
            source.rename(target)
            moved.append((source, target))
    run_study(created["study_path"])
    lock = lock_selection(created["study_path"])
    assert lock["selected"]["primary_score"] < 1.0, "declared synthetic accuracy target unmet"
    with pytest.raises(ValueError, match="approval"):
        evaluate_test(created["study_path"], [])
    for source, target in moved:
        target.rename(source)
    release = approvals(tmp_path / "release_decision", protocol, ("release",))
    test = evaluate_test(created["study_path"], release)
    assert test["metrics"]["primary_score"] < 1.0
    if dimension > 4:
        assert test["metrics"]["max_absolute"][-1] <= np.finfo(dtype).eps * 0.03
    report = build_report(created["study_path"], tmp_path / "report")
    assert report
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    exported = export_selected(created["study_path"], tmp_path / "release", release)
    assert exported["native_planes"]
    native = ElementSurrogate(tmp_path / "release", backend="torch")
    onnx = ElementSurrogate(tmp_path / "release", backend="onnx")
    assert onnx.session.get_session_options().intra_op_num_threads == request["timing"]["threads"]
    assert onnx.session.get_session_options().inter_op_num_threads == 1
    from element_surrogate.benchmark.protocol import write_json
    from element_surrogate.public_interface import PublicElementSurrogate, export_interface

    units = [
        {"m": "mm", "rad": "mrad", "s": "ns", "1": "1"}[coordinate["unit"]]
        for coordinate in onnx.contract["coordinates"]
    ]
    settings = onnx.contract["physics"]["qualified_settings"]
    control = settings[0]
    interface_request = {
        "state_units": units,
        "fixed_controls": control,
        "control_bounds": {key: [value, value] for key, value in control.items()},
    }
    write_json(tmp_path / "public_request.json", interface_request)
    export_interface(
        tmp_path / "release/contract.json",
        tmp_path / "public_request.json",
        tmp_path / "interface.json",
    )
    public = PublicElementSurrogate(tmp_path / "release", tmp_path / "interface.json")
    public_states = np.zeros((2, dimension))
    public_states[:, 0] = 0.1
    native_inputs = np.column_stack(
        [
            public_states * public.factors,
            np.full(2, control["effective_rigidity_tm"]),
            np.full(2, control["field_scale"]),
        ]
    )
    np.testing.assert_allclose(
        public.predict(public_states, numerical_support_established=True),
        onnx.predict(native_inputs, numerical_support_established=True) / public.factors,
        rtol=0,
        atol=0,
    )
    import json

    metadata = onnx.session.get_modelmeta().custom_metadata_map
    assert json.loads(metadata["native_io_json"])["native_planes"]
    assert json.loads(metadata["native_io_json"])["input_dtype"] == dtype
    if dimension == 4:
        import element_surrogate.export as export_module

        original_loader = export_module.load_adapter

        def changed_backend(*args, **kwargs):
            adapter = original_loader(*args, **kwargs)
            original_predict = adapter.predict

            def shifted(inputs):
                values = original_predict(inputs).copy()
                values[:, 0] += 10 * tolerance
                return values

            adapter.predict = shifted
            return adapter

        with monkeypatch.context() as context:
            context.setattr(export_module, "load_adapter", changed_backend)
            with pytest.raises(ValueError, match="selected validation"):
                export_module.export_selected(
                    created["study_path"], tmp_path / "rejected_export", release
                )
    inputs = np.zeros((2, dimension + 2))
    inputs[:, -2:] = [2.2, 1.0]
    with pytest.raises(ValueError, match="numerical support"):
        native.predict(inputs)
    expected = native.predict(inputs, numerical_support_established=True)
    np.testing.assert_allclose(
        onnx.predict(inputs, numerical_support_established=True), expected, atol=tolerance, rtol=0
    )
    checked, status = onnx.predict_checked(inputs, transport)
    np.testing.assert_array_equal(status, [0, 0])
    np.testing.assert_allclose(checked, expected, atol=tolerance, rtol=0)
    unseen = inputs.copy()
    unseen[:, -2] = 2.15
    checked, status = onnx.predict_checked(unseen, transport)
    assert np.all(status == 0) and np.isfinite(checked).all()
    inputs[0, 0] = 100
    _, status = onnx.predict_checked(inputs, transport)
    assert status[0] == -1
