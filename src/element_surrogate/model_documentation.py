"""Describe the selected function without changing fitting or model selection."""

import json
from pathlib import Path

from element_surrogate.benchmark.protocol import file_hash


def selected_description(trial, checkpoint):
    candidate = trial["candidate"]
    config = candidate.get("configuration", {})
    result = {
        "short_description": "Selected native-plane particle map for one static magnetic element.",
        "checkpoint_sha256": trial["checkpoint_sha256"],
        "selected_optimizer_step": trial.get("best_step"),
        "completed_optimizer_updates": trial.get("steps"),
        "convergence_established": False,
        "artifact_inspection": "unavailable for this adapter",
        "architecture_description": candidate.get(
            "forward_description", "See adapter configuration."
        ),
    }
    if file_hash(checkpoint) != result["checkpoint_sha256"]:
        raise ValueError("selected checkpoint changed before documentation")
    supported = config.get("schema") == "element_surrogate.static_magnetic_mlp.v1"
    if supported and config.get("model_family") != "canonical_shear":
        try:
            import torch
        except ImportError:
            result["artifact_inspection"] = "install neural extra to inspect native checkpoint"
        else:
            saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
            if json.loads(saved["configuration_json"]) != config:
                raise ValueError("selected checkpoint configuration differs from trial")
            state = saved["state_dict"]
            network = {
                k: v for k, v in state.items() if k.startswith(("network.", "breaking_network."))
            }
            if not network or not all(torch.isfinite(v).all() for v in state.values()):
                raise ValueError("finite native model state required for artifact documentation")
            final = []
            for prefix in ("network", "breaking_network"):
                subset = {k: v for k, v in network.items() if k.startswith(prefix + ".")}
                if subset:
                    layer = max(int(k.split(".")[1]) for k in subset)
                    final.extend(v for k, v in subset.items() if k.startswith(f"{prefix}.{layer}."))
            zero = all(not torch.count_nonzero(v) for v in final)
            fitted = config["base"] in {"train_affine", "train_polynomial"}
            result.update(
                artifact_inspection="native checkpoint configuration and finite weights",
                network_parameter_count=sum(v.numel() for v in network.values()),
                train_fitted_base_coefficients=(
                    int(state["base_mask"].count_nonzero())
                    if "base_mask" in state
                    else state["affine"].numel()
                )
                if fitted
                else 0,
                neural_output_layer_exactly_zero=zero,
                base=config["base"],
                base_degree=config.get("base_degree"),
            )
            if zero:
                result["short_description"] = (
                    "Selected native-plane map with a fitted "
                    + (
                        f"degree-{config['base_degree']} polynomial"
                        if config["base"] == "train_polynomial"
                        else "affine"
                    )
                    + " base and an exactly zero neural correction."
                    if fitted
                    else "Selected native-plane map with an exactly zero neural correction."
                )
            else:
                result["short_description"] = (
                    "Selected native-plane map with a neural correction; see the declared base."
                )
    step = result["selected_optimizer_step"]
    result["selection_interpretation"] = (
        "Validation selected the checkpoint before iterative optimization. Fitted bases may "
        "already use the full training cohort. This does not establish neural improvement."
        if step == 0
        else f"Validation selected optimizer step {step}; this alone is not convergence."
    )
    return result
