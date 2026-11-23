"""Manufactured quadrupole field and trainable equivariant MLP, generated locally.

Run: python -m element_surrogate.quadrupole_example --output ./quadrupole-demo
The defaults are small demonstration budgets, never production prescriptions.
"""

import argparse
from pathlib import Path

import numpy as np


def run(output_dir, *, steps=200, seed=314159):
    import torch

    from element_surrogate.benchmark.protocol import file_hash, protocol_identity, write_json
    from element_surrogate.contracts import StaticMagneticCodec
    from element_surrogate.field_store import save_mapped_field
    from element_surrogate.fields import FieldMap, load_field_file
    from element_surrogate.lorentz import straight_frame
    from element_surrogate.neural import build_adapter, forward_description, load_adapter
    from element_surrogate.region import Region
    from element_surrogate.symmetry import assess
    from element_surrogate.transport import Transport, qualify_transport

    if type(steps) is not int or steps < 1 or type(seed) is not int or seed < 0:
        raise ValueError("positive demonstration update count and nonnegative seed required")
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    axis, z_axis = np.linspace(-0.03, 0.03, 13), np.linspace(-0.1, 0.1, 21)
    x, y, z = np.meshgrid(axis, axis, z_axis, indexing="ij")
    gradient, variation = 3.0, 40.0
    # Phi = G*x*y*(1+a*z²-a*(x²+y²)/6), B=grad(Phi), Laplacian(Phi)=0.
    b = np.stack(
        [
            gradient * y * (1 + variation * z * z - variation * (3 * x * x + y * y) / 6),
            gradient * x * (1 + variation * z * z - variation * (x * x + 3 * y * y) / 6),
            2 * gradient * variation * x * y * z,
        ],
        axis=-1,
    )
    field = save_mapped_field(FieldMap(axis, axis, z_axis, b), out / "field", block_nodes=64)
    grid = load_field_file(field["path"]).slots[0]
    refinements = [
        {"method": "fixed-rk4", "step_m": h, "plane_tolerance_m": 1e-12, "max_path_m": 0.3}
        for h in (0.02, 0.01, 0.005)
    ]
    transport = Transport(
        grid,
        straight_frame(-0.08),
        straight_frame(0.08),
        0.16,
        "a" * 64,
        StaticMagneticCodec(4),
        Region("circle", 0.025),
        Region("circle", 0.018),
        refinements[-1],
    )
    ranges = [[-0.008, 0.008], [-0.01, 0.01], [-0.008, 0.008], [-0.01, 0.01]]
    settings = [
        {"id": str(i), "effective_rigidity_tm": 2.0, "field_scale": v}
        for i, v in enumerate((-0.6, 0.0, 0.6))
    ]
    interval = [-0.3, 0.3]
    qualification = qualify_transport(
        transport,
        ranges,
        settings,
        refinements,
        random_probes=2,
        seed=seed,
        component_limits=[1e-6] * 4,
        path_limit_m=1e-9,
        effective_coupling_range=interval,
    )
    write_json(out / "integration.json", qualification, exclusive=True)
    if not qualification["accepted"]:
        raise ValueError(
            "manufactured demonstration integration did not meet its explicit finite-cohort limits"
        )
    group = [[1] * 6, [-1, -1, 1, 1, 1, 1], [1, 1, -1, -1, 1, 1], [-1, -1, -1, -1, 1, 1]]
    symmetry = assess(
        transport,
        ranges,
        settings,
        group,
        [1e-10] * 4,
        random_probes=2,
        seed=seed,
        rationale=(
            "Manufactured harmonic scalar-potential quadrupole, centered native "
            "planes and symmetric numerical region"
        ),
        effective_coupling_range=interval,
    )
    write_json(out / "symmetry.json", symmetry, exclusive=True)
    if not symmetry["supported"]:
        raise ValueError("manufactured map failed the declared reflection comparison")
    rng = np.random.default_rng(seed)
    states = rng.uniform(np.array(ranges)[:, 0], np.array(ranges)[:, 1], (192, 4))
    controls = rng.uniform(-0.6, 0.6, 192)
    # One batched transport per control is unnecessary here: a few fixed controls
    # keep this worked example cheap and distinct from architecture selection.
    controls = np.repeat([-0.6, 0.0, 0.6], 64)
    outputs = np.empty_like(states)
    for value in np.unique(controls):
        mask = controls == value
        outputs[mask], result, _ = transport.run(states[mask], 2.0, float(value))
        if not result.reached_exit.all():
            raise ValueError("manufactured example contains unsupported labels")
    inputs = np.column_stack([states, np.full(len(states), 2.0), controls])
    train = np.arange(len(states)) % 3 != 0
    config = {
        "schema": "element_surrogate.static_magnetic_mlp.v1",
        "dimension": 4,
        "model_family": "reflection_features",
        "hidden_widths": [16, 16],
        "base": "none",
        "input_normalization": "zero_centered_rms",
        "output_normalization": "train_rms",
        "dtype": "float64",
        "device": "cpu",
        "threads": 1,
        "batch_size": 32,
        "inference_batch_size": 64,
        "learning_rate": 0.003,
        "loss_scales": [1e-4] * 4,
        "reflection_group": group,
        "symmetry_assessment": symmetry,
        "symmetry_evidence_sha256": protocol_identity(symmetry),
    }
    model = build_adapter(inputs[train], outputs[train], config, seed)
    for _ in range(steps):
        model.step()
    checkpoint = out / "equivariant_mlp.pt"
    model.save(checkpoint)
    restored = load_adapter(checkpoint, config)
    np.testing.assert_array_equal(restored.predict(inputs), model.predict(inputs))
    changes = []
    for signs in group:
        changes.append(
            np.max(
                np.abs(
                    restored.predict(inputs * np.array(signs))
                    - restored.predict(inputs) * np.array(signs[:4])
                ),
                axis=0,
            ).tolist()
        )
    native_path = out / "equivariant_mlp.onnx"
    torch.onnx.export(
        model.module.eval(),
        (torch.as_tensor(inputs[:2], dtype=torch.float64),),
        str(native_path),
        dynamo=False,
        opset_version=17,
        input_names=["inputs"],
        output_names=["outputs"],
        dynamic_axes={"inputs": {0: "batch"}, "outputs": {0: "batch"}},
    )
    import onnxruntime as ort

    session = ort.InferenceSession(str(native_path), providers=["CPUExecutionProvider"])
    portable = session.run(None, {"inputs": inputs})[0]
    np.testing.assert_allclose(portable, restored.predict(inputs), atol=1e-12, rtol=1e-12)
    np.savez_compressed(
        out / "local_labels.npz", inputs=inputs, targets=outputs, training_mask=train
    )
    result = {
        "schema": "element_surrogate.manufactured_quadrupole_example.v1",
        "field": field,
        "scalar_potential": "G*x*y*(1+a*z^2-a*(x^2+y^2)/6)",
        "gradient_t_per_m": gradient,
        "variation_per_m2": variation,
        "native_length_m": 0.16,
        "configuration": config,
        "description": forward_description(config),
        "updates": steps,
        "train_rows": int(train.sum()),
        "validation_rows": int((~train).sum()),
        "validation_rmse_native": np.sqrt(
            np.mean((restored.predict(inputs[~train]) - outputs[~train]) ** 2, axis=0)
        ).tolist(),
        "reflection_max_changes_native": changes,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": file_hash(checkpoint),
        "onnx_path": str(native_path),
        "onnx_max_change": float(np.max(np.abs(portable - restored.predict(inputs)))),
        "scope": (
            "Manufactured software example, not a facility magnet, "
            "model-selection benchmark, canonical-symplectic construction or "
            "production accuracy prescription."
        ),
    }
    write_json(out / "EXAMPLE.json", result, exclusive=True)
    (out / "README.md").write_text(
        (
            "# Manufactured quadrupole and equivariant MLP\n\nThe general software"
            " generated this harmonic scalar-potential field, numerical labels "
            "and trained MLP locally. EXAMPLE.json records its equations, units,"
            " finite integration and symmetry checks, actual fit error and "
            "export identity. This is an instructional example. It is not a "
            "recommended magnet model, field mesh, architecture or compute "
            "budget. See the installed physics feature guidance for real-case "
            "eligibility, asymmetric corrections and fair controls.\n"
        ),
        encoding="utf-8",
    )
    return result


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=314159)
    args = parser.parse_args()
    result = run(args.output, steps=args.steps, seed=args.seed)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "checkpoint_path",
                    "onnx_path",
                    "updates",
                    "train_rows",
                    "validation_rows",
                    "validation_rmse_native",
                    "scope",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
