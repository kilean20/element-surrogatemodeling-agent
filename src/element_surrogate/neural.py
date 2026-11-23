"""Optional PyTorch neural adapters for a qualified static-magnetic map.

No network size or optimizer budget is a universal default. The benchmark owns
selection. Exact momentum conservation and arrival-translation equivariance
follow from a static magnetic field. Optional reflection projection requires a
declared, evidenced symmetry group; it is never inferred from magnet names.
PyTorch is imported only when this optional backend is used.
"""

from __future__ import annotations

import json
import math
from itertools import combinations_with_replacement

import numpy as np

from element_surrogate.resources import configure_torch_threads


def _configuration(config, inputs=None):
    allowed = {
        "schema",
        "dimension",
        "hidden_widths",
        "base",
        "dtype",
        "device",
        "threads",
        "batch_size",
        "inference_batch_size",
        "learning_rate",
        "loss_scales",
        "output_normalization",
        "output_scales",
        "reflection_group",
        "symmetry_evidence_sha256",
        "symmetry_assessment",
        "base_degree",
        "max_base_features",
        "theory_contract",
        "model_family",
        "input_normalization",
        "symmetry_breaking",
        "breaking_hidden_widths",
        "residual_gate",
        "canonical_contract",
        "shear_pairs",
    }
    if set(config) - allowed:
        raise ValueError("unknown neural configuration; unsupported options are never ignored")
    if config.get("schema") != "element_surrogate.static_magnetic_mlp.v1":
        raise ValueError("unsupported neural configuration")
    d = config["dimension"]
    if (
        type(d) is not int
        or d not in (4, 5, 6)
        or not config["hidden_widths"]
        or any(type(w) is not int or w < 1 for w in config["hidden_widths"])
    ):
        raise ValueError("dimension 4/5/6 and positive hidden widths required")
    if config["base"] not in (
        "none",
        "train_affine",
        "train_polynomial",
        "theory_drift",
        "theory_quad",
    ) or config["dtype"] not in (
        "float32",
        "float64",
    ):
        raise ValueError("explicit base and floating-point dtype required")
    gate = config.get("residual_gate")
    if gate not in (None, "coupling") or (
        gate is not None and not config["base"].startswith("theory_")
    ):
        raise ValueError("coupling residual gate requires a qualified native theoretical base")
    if config["base"].startswith("theory_"):
        from element_surrogate.theory import validate

        kind = "drift" if config["base"] == "theory_drift" else "paraxial_quadrupole"
        validate(config.get("theory_contract"), kind, d)
    elif "theory_contract" in config:
        raise ValueError("theory contract applies only to a theoretical base")
    if config["base"] == "train_polynomial":
        _base_powers((d if d != 6 else d - 1) + 1, config)
    elif {"base_degree", "max_base_features"} & set(config):
        raise ValueError("polynomial degree/feature budget applies only to train_polynomial")
    for key in ("batch_size", "inference_batch_size", "threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError("positive batch size and thread count required")
    if not np.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        raise ValueError("positive learning rate required")
    if config.get("input_normalization", "mean_std") not in ("mean_std", "zero_centered_rms"):
        raise ValueError("unsupported input normalization")
    mode = config.get("output_normalization", "loss_scales")
    if mode not in ("loss_scales", "train_rms", "explicit"):
        raise ValueError("output_normalization must be loss_scales, train_rms or explicit")
    if mode == "explicit":
        out = np.asarray(config.get("output_scales"), float)
        learned = d if d == 4 else d - 1
        if out.shape != (learned,) or not np.isfinite(out).all() or np.any(out <= 0):
            raise ValueError("explicit positive native output_scales required")
    elif "output_scales" in config:
        raise ValueError("output_scales applies only to explicit normalization")
    if config.get("model_family") == "canonical_shear" and mode != "loss_scales":
        raise ValueError("canonical shear uses its explicit canonical scales")
    scales = np.asarray(config["loss_scales"], float)
    if scales.shape != (d,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("positive dimensional loss scales required")
    if inputs is not None and (
        inputs.ndim != 2 or inputs.shape[1] != d + 2 or not np.isfinite(inputs).all()
    ):
        raise ValueError("native state plus effective rigidity and scale inputs required")
    if inputs is not None and (
        np.any(inputs[:, -2] == 0) or (d > 4 and np.any(inputs[:, d - 1] <= -1))
    ):
        raise ValueError("nonzero signed effective rigidity and delta_p/p0 > -1 required")
    signs = config.get("reflection_group", [[1] * (d + 2)])
    group = {tuple(s) for s in signs}
    if (
        len(group) != len(signs)
        or tuple([1] * (d + 2)) not in group
        or any(len(s) != d + 2 or any(v not in (-1, 1) for v in s) for s in signs)
    ):
        raise ValueError("a finite sign group including identity required")
    for a in signs:
        if any(v != 1 for v in a[4:]):
            raise ValueError("reflection group may act only on transverse coordinates")
        if any(tuple(x * y for x, y in zip(a, b, strict=True)) not in group for b in signs):
            raise ValueError("reflection group must be closed under composition")
    if len(group) > 1:
        digest = config.get("symmetry_evidence_sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("reflection projection requires a symmetry-evidence SHA256")
        from element_surrogate.benchmark.protocol import protocol_identity
        from element_surrogate.symmetry import validate, validate_inputs, validate_measurement

        evidence = config.get("symmetry_assessment")
        validator = (
            validate_measurement
            if config.get("model_family") == "reflection_features"
            and config.get("symmetry_breaking")
            else validate
        )
        validator(evidence, signs, d)
        if protocol_identity(evidence) != digest:
            raise ValueError("bound symmetry evidence identity differs")
        if inputs is not None:
            validate_inputs(evidence, inputs)
    if inputs is not None and config["base"].startswith("theory_"):
        from element_surrogate.theory import validate_inputs as validate_theory_inputs

        validate_theory_inputs(config["theory_contract"], inputs, d)
    family = config.get("model_family", "mlp")
    if family != "reflection_features" and {"symmetry_breaking", "breaking_hidden_widths"} & set(
        config
    ):
        raise ValueError("symmetry-breaking branch options require reflection_features")
    if family == "canonical_shear":
        if "input_normalization" in config:
            raise ValueError(
                "canonical input scales belong in canonical_contract, not input_normalization"
            )
        from element_surrogate.canonical import validate

        validate(config)
        if inputs is not None:
            from element_surrogate.symmetry import validate_inputs

            validate_inputs(config["canonical_contract"], inputs)
    elif family == "reflection_features":
        from element_surrogate.equivariant import layout

        layout(config)
        if {"canonical_contract", "shear_pairs"} & set(config):
            raise ValueError("canonical options do not apply to reflection features")
    elif family != "mlp" or {
        "canonical_contract",
        "shear_pairs",
        "symmetry_breaking",
        "breaking_hidden_widths",
    } & set(config):
        raise ValueError("unsupported neural family or misplaced canonical options")
    return d


def _base_powers(inputs, config):
    degree, maximum = config.get("base_degree"), config.get("max_base_features")
    if type(degree) is not int or degree < 1 or type(maximum) is not int or maximum < 1:
        raise ValueError("explicit positive polynomial degree and feature budget required")
    count = math.comb(inputs + degree, inputs)
    if count > maximum:
        raise ValueError("polynomial basis exceeds declared feature budget")
    powers = []
    for order in range(1, degree + 1):
        for indices in combinations_with_replacement(range(inputs), order):
            powers.append(np.bincount(indices, minlength=inputs))
    powers.append(np.zeros(inputs, dtype=int))
    return np.asarray(powers)


def _native(module, inputs):
    import torch

    # Absolute arrival does not enter a static field. Learn its increment.
    coupling = inputs[:, -1:] / inputs[:, -2:-1]
    work = torch.cat((inputs[:, module.active_indices], coupling), dim=1)
    total = None
    for index in range(module.group_order):
        signs, output_signs = module.signs[index], module.output_signs[index]
        transformed = work * signs
        normalized = (transformed - module.input_mean) / module.input_scale
        correction = module.network(normalized)
        if hasattr(module, "coupling_gate_scale"):
            correction = correction * coupling / module.coupling_gate_scale
        if module.base_kind == "train_polynomial":
            basis = torch.pow(normalized[:, None, :], module.base_powers[None, :, :]).prod(dim=-1)
            affine = basis @ module.affine
        else:
            affine = normalized @ module.affine[:-1] + module.affine[-1]
        scale = (
            module.output_scale
            if hasattr(module, "output_scale")
            else module.loss_scales[: module.learned_dimension]
        )
        output = (affine + correction) * scale
        if module.base_kind.startswith("theory_"):
            from element_surrogate.theory import predict_torch

            reflected = inputs * module.native_signs[index]
            output = output + predict_torch(reflected, module.theory_contract, module.dimension)
        output = output * output_signs
        total = output if total is None else total + output
    result = total / module.group_order
    if module.dimension == 6:
        result = torch.cat((result[:, :4], result[:, 4:5] + inputs[:, 4:5], inputs[:, 5:6]), dim=1)
    elif module.dimension == 5:
        result = torch.cat((result[:, :4], inputs[:, 4:5]), dim=1)
    return result


class Adapter:
    """Complete native map; optimizer and mini-batch state stay inside the worker."""

    def predict(self, inputs):
        import torch

        array = np.asarray(inputs)
        if (
            array.ndim != 2
            or array.shape[1] != self.module.dimension + 2
            or not np.isfinite(array).all()
        ):
            raise ValueError("finite complete native inference inputs required")
        if np.any(array[:, -2] == 0):
            raise ValueError("nonzero effective rigidity required")
        output = np.empty((len(array), self.module.dimension), dtype=self.config["dtype"])
        with torch.no_grad():
            size = self.config["inference_batch_size"]
            for start in range(0, len(array), size):
                tensor = torch.as_tensor(
                    array[start : start + size], dtype=self.dtype, device=self.device
                )
                output[start : start + size] = self.module(tensor).cpu().numpy()
        return output

    def step(self):
        import torch

        indices = self.rng.integers(
            0, len(self.x), size=min(self.config["batch_size"], len(self.x))
        )
        x = torch.as_tensor(self.x[indices], dtype=self.dtype, device=self.device)
        y = torch.as_tensor(self.y[indices], dtype=self.dtype, device=self.device)
        self.optimizer.zero_grad(set_to_none=True)
        error = (self.module(x) - y) / self.module.loss_scales
        loss = error.square().mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite neural training loss")
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        import torch

        torch.save(
            {
                "configuration_json": json.dumps(self.config, sort_keys=True),
                "state_dict": self.module.state_dict(),
                "normalization_json": json.dumps(
                    getattr(self, "normalization_report", {}), sort_keys=True
                ),
            },
            path,
        )

    def synchronize(self):
        import torch

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


def _module(config, mean, scale, affine, output_scale=None, coupling_gate_scale=None):
    import torch
    from torch import nn

    d = _configuration(config)
    learned = d if d == 4 else d - 1
    dtype = getattr(torch, config["dtype"])
    active = [i for i in range(d) if not (d == 6 and i == 4)]

    class NativeMap(nn.Module):
        def __init__(self):
            super().__init__()
            self.dimension = d
            self.base_kind = config["base"]
            self.theory_contract = config.get("theory_contract")
            self.learned_dimension = learned
            self.active_indices = active
            layers = []
            widths = [len(active) + 1, *config["hidden_widths"], learned]
            for index, (before, after) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
                layers.append(nn.Linear(before, after))
                if index < len(widths) - 2:
                    layers.append(nn.Tanh())
            self.network = nn.Sequential(*layers).to(dtype=dtype)
            if config.get("residual_gate") == "coupling":
                gate_value = float(coupling_gate_scale) if coupling_gate_scale is not None else None
                if gate_value is None or not np.isfinite(gate_value) or gate_value <= 0:
                    raise ValueError("positive training-only coupling gate scale required")
                self.register_buffer(
                    "coupling_gate_scale", torch.as_tensor(gate_value, dtype=dtype)
                )
            if self.base_kind == "train_polynomial":
                self.register_buffer(
                    "base_powers",
                    torch.as_tensor(_base_powers(len(active) + 1, config), dtype=dtype),
                )
            for name, value in (
                ("input_mean", mean),
                ("input_scale", scale),
                ("affine", affine),
                ("loss_scales", config["loss_scales"]),
            ):
                self.register_buffer(name, torch.as_tensor(value, dtype=dtype))
            if config.get("output_normalization") in ("train_rms", "explicit"):
                out = np.asarray(output_scale, dtype=float)
                if out.shape != (learned,) or not np.isfinite(out).all() or np.any(out <= 0):
                    raise ValueError("saved positive train-only output scales required")
                self.register_buffer("output_scale", torch.as_tensor(out, dtype=dtype))
            group = config.get("reflection_group", [[1] * (d + 2)])
            self.group_order = len(group)
            self.register_buffer(
                "native_signs", torch.as_tensor(group, dtype=dtype), persistent=False
            )
            self.register_buffer(
                "signs", torch.as_tensor([[*[s[i] for i in active], 1] for s in group], dtype=dtype)
            )
            self.register_buffer(
                "output_signs", torch.as_tensor([s[:learned] for s in group], dtype=dtype)
            )

        def forward(self, inputs):
            return _native(self, inputs)

    return NativeMap()


def build_adapter(x, y, configuration, seed):
    import torch

    x, y = np.asarray(x), np.asarray(y)
    if configuration.get("model_family") == "reflection_features":
        from element_surrogate.equivariant import build

        return build(x, y, configuration, seed)
    if configuration.get("model_family") == "canonical_shear":
        from element_surrogate.canonical import build

        return build(x, y, configuration, seed)
    d = _configuration(configuration, x)
    if y.shape != (len(x), d) or not len(x) or not np.isfinite(y).all():
        raise ValueError("complete finite training labels required")
    if d > 4 and not np.array_equal(x[:, d - 1], y[:, d - 1].astype(x.dtype)):
        raise ValueError("static magnetic labels must conserve delta_p/p0 exactly")
    config = configuration
    configure_torch_threads(torch, config["threads"])
    active = [i for i in range(d) if not (d == 6 and i == 4)]
    work = np.column_stack([x[:, active], x[:, -1] / x[:, -2]])
    if not np.isfinite(work).all():
        raise ValueError("nonfinite reduced coupling")
    mean = work.mean(axis=0)
    scale = work.std(axis=0)
    if config.get("input_normalization") == "zero_centered_rms":
        mean = np.zeros(work.shape[1])
        scale = np.sqrt(np.mean(work**2, axis=0))
    scale[scale == 0] = 1
    target = y.copy()
    if d == 6:
        target[:, 4] -= x[:, 4]
    if d > 4:
        target[:, -1] = 0
    learned = d if d == 4 else d - 1
    normalization = {"method": "legacy_loss_scales"}
    output_scale = np.asarray(config["loss_scales"])[:learned]
    if config.get("output_normalization") == "train_rms":
        from element_surrogate.scaling import output_scales

        output_scale, normalization = output_scales(x, y, d)
    elif config.get("output_normalization") == "explicit":
        output_scale = np.asarray(config["output_scales"])
        normalization = {
            "method": "explicit",
            "scales_native": output_scale.tolist(),
            "loss_targets_used": False,
        }
    if config["base"].startswith("theory_"):
        from element_surrogate.theory import predict_torch

        base = predict_torch(
            torch.as_tensor(x, dtype=getattr(torch, config["dtype"])), config["theory_contract"], d
        ).numpy()
        target[:, : base.shape[1]] -= base
    normalized = (work - mean) / scale
    learned = d if d == 4 else d - 1
    affine = np.zeros((len(active) + 2, learned))
    if config["base"] in {"train_affine", "train_polynomial"}:
        if config["base"] == "train_polynomial":
            powers = _base_powers(normalized.shape[1], config)
            design = np.prod(normalized[:, None, :] ** powers[None, :, :], axis=-1)
        else:
            design = np.column_stack([normalized, np.ones(len(x))])
        affine = np.linalg.lstsq(
            design,
            target[:, :learned] / output_scale,
            rcond=None,
        )[0]
    adapter = Adapter()
    adapter.config = config
    adapter.dtype = getattr(torch, config["dtype"])
    adapter.device = torch.device(config["device"])
    # Do not change the caller's random state while initializing a candidate.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        gate_scale = float(np.sqrt(np.mean(work[:, -1] ** 2))) or 1.0
        adapter.module = _module(config, mean, scale, affine, output_scale, gate_scale).to(
            adapter.device
        )
        torch.nn.init.zeros_(adapter.module.network[-1].weight)
        torch.nn.init.zeros_(adapter.module.network[-1].bias)
    adapter.normalization_report = normalization
    adapter.x, adapter.y = x, y
    adapter.rng = np.random.default_rng(seed)
    adapter.optimizer = torch.optim.Adam(adapter.module.parameters(), lr=config["learning_rate"])
    return adapter


def forward_description(configuration):
    d = _configuration(configuration)
    if configuration.get("model_family") == "reflection_features":
        from element_surrogate.equivariant import description

        return description(configuration)
    if configuration.get("model_family") == "canonical_shear":
        return (
            f"Native 4D potential-shear map: {configuration['shear_pairs']} alternating pairs, "
            f"potential width {configuration['hidden_widths'][0]}, "
            "conditioned on amplitude/rigidity. "
            "Exact slope/mechanical-momentum conversions and common canonical scales; "
            "symplectic where forward momentum is real under recorded zero-normal-field condition. "
            "No additive residual or reflection averaging. "
            "All conversions included in prediction cost."
            f" {10 * configuration['hidden_widths'][0] * configuration['shear_pairs']} "
            "optimized parameters."
        )
    base = (
        "a train-only least-squares affine base plus "
        if configuration["base"] == "train_affine"
        else ""
    )
    if configuration["base"] == "train_polynomial":
        base = (
            f"a train-only total-degree-{configuration['base_degree']} "
            "polynomial least-squares base plus "
        )
    if configuration["base"].startswith("theory_"):
        base = configuration["theory_contract"]["approximation"] + "; plus "
    group = len(configuration.get("reflection_group", [[1] * (d + 2)]))
    inputs = (d if d != 6 else d - 1) + 1
    learned = d if d == 4 else d - 1
    widths = [inputs, *configuration["hidden_widths"], learned]
    optimized = sum((a + 1) * b for a, b in zip(widths[:-1], widths[1:], strict=True))
    features = (
        math.comb(inputs + configuration["base_degree"], inputs)
        if configuration["base"] == "train_polynomial"
        else inputs + 1
    )
    base_coefficients = (
        features * learned if configuration["base"] in {"train_affine", "train_polynomial"} else 0
    )
    coordinate_behavior = "Four transverse outputs; no energy or arrival coordinate is modeled. "
    if d > 4:
        coordinate_behavior = "Momentum deviation passes through in inference precision. "
    if d == 6:
        coordinate_behavior += "Arrival is input plus a learned increment. "
    arrival_features = "Exclude absolute arrival from neural features. " if d == 6 else ""
    return (
        f"Native {d}D static-magnetic map. "
        "Replace amplitude and effective rigidity by their ratio; "
        f"{arrival_features}Train-only "
        f"{configuration.get('input_normalization', 'mean_std')} input normalization; "
        f"output parameterization {configuration.get('output_normalization', 'loss_scales')}; "
        f"{base}tanh MLP with hidden widths {configuration['hidden_widths']}; "
        f"{optimized} optimizer-trainable parameters, "
        f"{base_coefficients} train-fitted base coefficients; "
        f"{group}-element declared reflection projection. "
        + (
            "Coupling-gated residual vanishes at zero field on the qualified native base. "
            if configuration.get("residual_gate")
            else ""
        )
        + f"{coordinate_behavior}"
        "No field-free, canonical-symplectic or effective-length claim is inferred."
    )


def load_adapter(path, configuration):
    import torch

    if configuration.get("model_family") == "reflection_features":
        from element_surrogate.equivariant import load

        return load(path, configuration)
    if configuration.get("model_family") == "canonical_shear":
        from element_surrogate.canonical import load

        return load(path, configuration)
    d = _configuration(configuration)
    configure_torch_threads(torch, configuration["threads"])
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["configuration_json"] != json.dumps(configuration, sort_keys=True):
        raise ValueError("checkpoint neural configuration differs from protocol")
    state = saved["state_dict"]
    adapter = Adapter()
    adapter.config = configuration
    adapter.dtype = getattr(torch, configuration["dtype"])
    adapter.device = torch.device(configuration["device"])
    adapter.module = _module(
        configuration,
        state["input_mean"],
        state["input_scale"],
        state["affine"],
        state.get("output_scale"),
        state.get("coupling_gate_scale"),
    )
    adapter.module.load_state_dict(state, strict=True)
    adapter.module.to(adapter.device)
    adapter.module.eval()
    adapter.normalization_report = json.loads(saved.get("normalization_json", "{}"))
    assert adapter.module.dimension == d
    return adapter
