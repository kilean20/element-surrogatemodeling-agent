"""Verified local native-map export and a common inference API.

PyTorch/ONNX/ONNX Runtime are optional caller-provided dependencies. Export uses
validation only for numerical equivalence and never changes model selection.
This prepares a local artifact; it performs no upload or registry publication.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from element_surrogate.benchmark.protocol import (
    canonical,
    check_approvals,
    file_hash,
    read_json,
    write_json,
)
from element_surrogate.benchmark.scoring import load_cohort
from element_surrogate.benchmark.study import _read_study, lock_selection
from element_surrogate.native_scope import export_physics, mask, validate
from element_surrogate.neural import load_adapter
from element_surrogate.resources import numerical_threads


def export_selected(study_path, output_dir, approval_paths) -> dict:
    """Copy the locked checkpoint and check dynamic-batch ONNX against PyTorch."""
    import onnx
    import onnxruntime as ort
    import torch

    directory, record, p = _read_study(study_path)
    check_approvals(p, approval_paths, ("release",))
    lock_selection(study_path)
    if not (directory / "test/result.json").is_file():
        raise ValueError("complete the locked held-out evaluation before preparing a release")
    selection = read_json(directory / "selection.json")
    selected = selection["selected"]
    candidate = next(c for c in p["candidates"] if c["id"] == selected["candidate_id"])
    checkpoint = directory / selected["checkpoint_path"]
    if file_hash(checkpoint) != selected["checkpoint_sha256"]:
        raise ValueError("selected checkpoint changed")
    physics = export_physics(p)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = candidate["configuration"]
    model = load_adapter(checkpoint, config)
    # Portable export is CPU at the same declared arithmetic precision.
    model.module.cpu().eval()
    model.device = torch.device("cpu")
    validation = load_cohort(p, record["base"], "validation")
    input_dtype = p["timing"]["input_dtype"]
    examples = validation["inputs"].astype(input_dtype)
    if not mask(validation["inputs"], physics, len(p["coordinates"]))[1].all():
        raise ValueError("validation cohort lies outside the declared native API scope")
    golden = model.predict(examples)
    tolerances = np.asarray(p["timing"]["golden_atol"])
    with np.load(
        Path(selected["trial_path"]).parent / "validation.npz", allow_pickle=False
    ) as saved:
        order = np.argsort(saved["sample_id"].astype(str))
        if not np.array_equal(saved["sample_id"].astype(str)[order], validation["sample_id"]):
            raise ValueError("selected validation IDs differ before portable export")
        selected_prediction = saved["prediction"][order]
    portable_change = np.max(np.abs(golden - selected_prediction), axis=0)
    if not np.isfinite(golden).all() or np.any(portable_change > tolerances):
        raise ValueError("portable CPU backend differs from selected validation predictions")
    destination = output / "model.pt"
    runtime_config = {**config, "device": "cpu"}
    model.config = runtime_config
    model.save(destination)
    path = output / "model.onnx"
    options = {
        "opset_version": 17,
        "input_names": ["inputs"],
        "output_names": ["outputs"],
        "dynamic_axes": {"inputs": {0: "batch"}, "outputs": {0: "batch"}},
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        options["dynamo"] = False

    class PortableInputMap(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, inputs):
            return self.inner(inputs.to(dtype=model.dtype))

    torch.onnx.export(
        PortableInputMap(model.module),
        torch.as_tensor(examples[:1], dtype=getattr(torch, input_dtype)),
        str(path),
        **options,
    )
    graph = onnx.load(str(path))
    onnx.helper.set_model_props(
        graph,
        {
            "element_surrogate_schema": "element_surrogate.native_model.v1",
            "protocol_sha256": record["protocol_sha256"],
            "native_io_json": canonical(
                {
                    "inputs": p["inputs"],
                    "input_units": p["input_units"],
                    "input_dtype": input_dtype,
                    "coordinates": p["coordinates"],
                    "output_dtype": config["dtype"],
                    "native_planes": True,
                }
            ),
            "physics_identities_json": canonical(
                {
                    k: p["physics"][k]
                    for k in (
                        "field_sha256",
                        "reference_sha256",
                        "reconstruction_sha256",
                        "integration_qualification_sha256",
                        "region_sha256",
                    )
                }
            ),
        },
    )
    onnx.checker.check_model(graph)
    onnx.save(graph, str(path))
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = numerical_threads(p["timing"]["threads"])
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path), sess_options=session_options, providers=["CPUExecutionProvider"]
    )
    changes = []
    for size in sorted({1, min(7, len(examples)), len(examples)}):
        actual = session.run(["outputs"], {"inputs": examples[:size]})[0]
        difference = np.max(np.abs(actual - golden[:size]), axis=0)
        if (
            actual.shape != golden[:size].shape
            or not np.isfinite(actual).all()
            or np.any(difference > tolerances)
        ):
            raise ValueError(
                "exported full native map differs from PyTorch; no complete release record written"
            )
        changes.append({"batch_size": size, "max_component_change": difference.tolist()})
    contract = {
        "schema": "element_surrogate.native_model.v1",
        "protocol_sha256": record["protocol_sha256"],
        "selection_sha256": file_hash(directory / "selection.json"),
        "selected": selected,
        "configuration": runtime_config,
        "training_configuration": config,
        "inference_input_dtype": input_dtype,
        "trainable_parameters": sum(v.numel() for v in model.module.parameters()),
        "inputs": p["inputs"],
        "input_units": p["input_units"],
        "coordinates": p["coordinates"],
        "physics": physics,
        "dataset_identity_sha256": p["dataset"]["identity_sha256"],
        "candidate_rationale": candidate["rationale"],
        "forward_description": candidate["forward_description"],
        "files": {name: file_hash(output / name) for name in ("model.pt", "model.onnx")},
        "export_check": {
            "source": "validation only",
            "changes": changes,
            "portable_cpu_change_from_selected_validation": portable_change.tolist(),
            "golden_atol": tolerances.tolist(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "provider": "CPUExecutionProvider",
        },
        "qualification_limit": (
            "survivor-conditioned map; physical field accuracy/material aperture not certified"
        ),
        "publication_status": "local artifact; institutional/governance/card review separate",
    }
    write_json(output / "contract.json", contract, exclusive=True)
    return {
        "contract_path": str(output / "contract.json"),
        "dimension": len(p["coordinates"]),
        "native_planes": True,
        "export_checks": changes,
        "external_publication": False,
    }


class ElementSurrogate:
    """The same native map in PyTorch or ONNX; placement belongs to composites."""

    def __init__(self, directory, *, backend="onnx"):
        self.directory = Path(directory).resolve()
        self.contract = read_json(self.directory / "contract.json")
        if (
            self.contract.get("schema") != "element_surrogate.native_model.v1"
            or self.contract["physics"]["native_planes"] is not True
        ):
            raise ValueError("native element contract required")
        validate(self.contract["physics"], len(self.contract["coordinates"]))
        for name, digest in self.contract["files"].items():
            member = (self.directory / name).resolve()
            if not member.is_relative_to(self.directory) or file_hash(member) != digest:
                raise ValueError("model release member changed")
        self.backend = backend
        if backend == "torch":
            self.adapter = load_adapter(self.directory / "model.pt", self.contract["configuration"])
        elif backend == "onnx":
            import onnxruntime as ort

            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = numerical_threads(
                self.contract["configuration"]["threads"]
            )
            session_options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(
                str(self.directory / "model.onnx"),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
        else:
            raise ValueError("choose torch or onnx backend")

    def _scope_mask(self, inputs):
        return mask(inputs, self.contract["physics"], len(self.contract["coordinates"]))

    def predict(self, inputs, *, numerical_support_established=False):
        """Fast inference requires independent survivor/support evidence."""
        if numerical_support_established is not True:
            raise ValueError("establish numerical support independently or use predict_checked")
        a, inside = self._scope_mask(inputs)
        if not inside.all():
            raise ValueError("proposal outside declared qualified box/settings")
        if not len(a):
            return np.empty((0, len(self.contract["coordinates"])))
        if self.backend == "torch":
            return self.adapter.predict(
                a.astype(
                    self.contract.get(
                        "inference_input_dtype", self.contract["configuration"]["dtype"]
                    ),
                    copy=False,
                )
            )
        dtype = self.contract["configuration"]["dtype"]
        size = self.contract["configuration"]["inference_batch_size"]
        output = np.empty((len(a), len(self.contract["coordinates"])), dtype=dtype)
        for start in range(0, len(a), size):
            output[start : start + size] = self.session.run(
                ["outputs"],
                {
                    "inputs": a[start : start + size].astype(
                        self.contract.get("inference_input_dtype", dtype), copy=False
                    )
                },
            )[0]
        return output

    def predict_checked(self, inputs, transport):
        """Check full tracked support, then infer; this costs full tracking."""
        a, inside = self._scope_mask(inputs)
        if transport.description() != self.contract["physics"]["transport_description"]:
            raise ValueError("domain checker transport differs from the released map")
        prediction = np.full((len(a), transport.codec.dimension), np.nan)
        status = np.full(len(a), -1, dtype=int)  # -1 means outside the qualified declaration.
        for rigidity, scale in np.unique(a[inside, -2:], axis=0):
            ids = np.flatnonzero(inside & (a[:, -2] == rigidity) & (a[:, -1] == scale))
            if not len(ids):
                continue
            _, tracked, _ = transport.run(
                a[ids, : transport.codec.dimension],
                rigidity,
                scale,
            )
            status[ids] = tracked.status
            live = ids[tracked.reached_exit]
            prediction[live] = self.predict(a[live], numerical_support_established=True)
        return prediction, status
