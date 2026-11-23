"""Single-pass character features for supported transverse reflection groups.

The finite reflection assumption is supplied by the existing measured map
assessment. Feature equivariance is algebraic, not a field-symmetry proof and
not a symplectic construction. No facility geometry or model ships here.
"""

from __future__ import annotations

import json

import numpy as np


def layout(config):
    from element_surrogate.symmetry import validate_group

    d = config["dimension"]
    group = validate_group(config.get("reflection_group", []), d)
    if len(group) < 2:
        raise ValueError("reflection features require a measured nontrivial group")
    if config.get("input_normalization") != "zero_centered_rms":
        raise ValueError("reflection features require zero_centered_rms about the native center")
    breaking = config.get("symmetry_breaking", False)
    if type(breaking) is not bool:
        raise ValueError("symmetry_breaking must be an explicit boolean")
    widths = config.get("breaking_hidden_widths")
    if breaking and (
        not isinstance(widths, list)
        or not widths
        or any(type(w) is not int or w < 1 for w in widths)
    ):
        raise ValueError("explicit positive breaking_hidden_widths required")
    if not breaking and widths is not None:
        raise ValueError("breaking widths require the symmetry-breaking branch")
    active = [i for i in range(d) if not (d == 6 and i == 4)]
    n = len(active) + 1
    signs = np.column_stack([group[:, active], np.ones(len(group), dtype=int)])
    invariant = []
    for i in range(n):
        if np.all(signs[:, i] == 1):
            power = np.zeros(n, dtype=int)
            power[i] = 1
            invariant.append(power)
    for i in range(4):
        for j in range(i, 4):
            if np.any(signs[:, i] != 1) and np.array_equal(signs[:, i], signs[:, j]):
                power = np.zeros(n, dtype=int)
                power[i] += 1
                power[j] += 1
                invariant.append(power)
    gates = []
    learned = d if d == 4 else d - 1
    for j in range(learned):
        character = group[:, j]
        gates.append(
            [-1]
            if np.all(character == 1)
            else [i for i in range(4) if np.array_equal(signs[:, i], character)]
        )
    if not all(gates):
        raise ValueError("unsupported reflection output character")
    return np.asarray(invariant), gates, signs, group[:, :learned]


def base_layout(config):
    from element_surrogate.neural import _base_powers

    _, _, signs, outputs = layout(config)
    n = signs.shape[1]
    if config["base"] == "train_polynomial":
        powers = _base_powers(n, config)
    elif config["base"] == "train_affine":
        powers = np.vstack([np.eye(n, dtype=int), np.zeros(n, dtype=int)])
    else:
        powers = np.zeros((1, n), dtype=int)
    characters = np.prod(signs[:, None, :] ** powers[None, :, :], axis=-1)
    mask = np.all(characters[:, :, None] == outputs[:, None, :], axis=0)
    keep = mask.any(axis=1)
    if not keep.any():
        keep[0] = True  # A zero constant base avoids empty-tensor portable graphs.
    return powers[keep], mask[keep]


def _monomials(values, powers):
    return np.prod(values[:, None, :] ** powers[None, :, :], axis=-1)


def module(config, state):
    import torch
    from torch import nn

    invariant, gates, _, _ = layout(config)
    powers, mask = base_layout(config)
    d = config["dimension"]
    learned = d if d == 4 else d - 1
    dtype = getattr(torch, config["dtype"])

    class ReflectionFeatureMap(nn.Module):
        def __init__(self):
            super().__init__()
            self.dimension = d
            self.learned_dimension = learned
            self.active_indices = [i for i in range(d) if not (d == 6 and i == 4)]
            self.base_kind = config["base"]
            self.theory_contract = config.get("theory_contract")
            self.gates = gates
            widths = [len(invariant), *config["hidden_widths"], sum(map(len, gates))]
            layers = []
            for i, (a, b) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
                layers.append(nn.Linear(a, b))
                if i < len(widths) - 2:
                    layers.append(nn.Tanh())
            self.network = nn.Sequential(*layers).to(dtype=dtype)
            if config.get("residual_gate") == "coupling":
                value = state.get("coupling_gate_scale")
                value = float(value) if value is not None else None
                if value is None or not np.isfinite(value) or value <= 0:
                    raise ValueError("positive coupling residual gate scale required")
                self.register_buffer("coupling_gate_scale", torch.as_tensor(value, dtype=dtype))
            self.has_breaking = config.get("symmetry_breaking", False)
            if self.has_breaking:
                bw = [len(self.active_indices) + 1, *config["breaking_hidden_widths"], learned]
                branch = []
                for i, (a, b) in enumerate(zip(bw[:-1], bw[1:], strict=True)):
                    branch.append(nn.Linear(a, b))
                    if i < len(bw) - 2:
                        branch.append(nn.Tanh())
                self.breaking_network = nn.Sequential(*branch).to(dtype=dtype)
            defaults = {"invariant_powers": invariant, "base_powers": powers, "base_mask": mask}
            for name in (
                "input_mean",
                "input_scale",
                "invariant_mean",
                "invariant_scale",
                "affine",
                "loss_scales",
                "output_scale",
                "invariant_powers",
                "base_powers",
                "base_mask",
            ):
                value = state.get(name, defaults.get(name))
                if value is None:
                    raise ValueError("missing saved reflection-feature state: " + name)
                self.register_buffer(name, torch.as_tensor(value, dtype=dtype))

        def forward(self, inputs):
            work = torch.cat(
                (inputs[:, self.active_indices], inputs[:, -1:] / inputs[:, -2:-1]), dim=1
            )
            normalized = (work - self.input_mean) / self.input_scale
            features = torch.pow(normalized[:, None, :], self.invariant_powers[None]).prod(dim=-1)
            weights = self.network((features - self.invariant_mean) / self.invariant_scale)
            correction, offset = [], 0
            for indices in self.gates:
                terms = [
                    weights[:, offset + j] * (1 if i == -1 else normalized[:, i])
                    for j, i in enumerate(indices)
                ]
                correction.append(sum(terms))
                offset += len(indices)
            correction = torch.stack(correction, dim=1)
            if self.has_breaking:
                correction = correction + self.breaking_network(normalized)
            if hasattr(self, "coupling_gate_scale"):
                correction = (
                    correction * inputs[:, -1:] / inputs[:, -2:-1] / self.coupling_gate_scale
                )
            basis = torch.pow(normalized[:, None, :], self.base_powers[None]).prod(dim=-1)
            output = (basis @ (self.affine * self.base_mask) + correction) * self.output_scale
            if self.base_kind.startswith("theory_"):
                from element_surrogate.theory import predict_torch

                output = output + predict_torch(inputs, self.theory_contract, d)
            if d == 6:
                output = torch.cat(
                    (output[:, :4], output[:, 4:5] + inputs[:, 4:5], inputs[:, 5:6]), dim=1
                )
            elif d == 5:
                output = torch.cat((output[:, :4], inputs[:, 4:5]), dim=1)
            return output

    return ReflectionFeatureMap()


def build(x, y, config, seed):
    import torch

    from element_surrogate.neural import Adapter, _configuration
    from element_surrogate.resources import configure_torch_threads
    from element_surrogate.scaling import output_scales

    x, y = np.asarray(x), np.asarray(y)
    d = _configuration(config, x)
    if y.shape != (len(x), d) or not len(x) or not np.isfinite(y).all():
        raise ValueError("complete finite training labels required")
    if d > 4 and not np.array_equal(x[:, d - 1], y[:, d - 1].astype(x.dtype)):
        raise ValueError("static magnetic labels must conserve delta_p/p0 exactly")
    configure_torch_threads(torch, config["threads"])
    active = [i for i in range(d) if not (d == 6 and i == 4)]
    work = np.column_stack([x[:, active], x[:, -1] / x[:, -2]])
    scale = np.sqrt(np.mean(work**2, axis=0))
    scale[scale == 0] = 1
    normalized = work / scale
    invariant, _, _, _ = layout(config)
    features = _monomials(normalized, invariant)
    f_mean, f_scale = features.mean(axis=0), features.std(axis=0)
    f_scale[f_scale == 0] = 1
    learned = d if d == 4 else d - 1
    out = np.asarray(config["loss_scales"])[:learned]
    normalization = {"method": "legacy_loss_scales"}
    if config.get("output_normalization") == "train_rms":
        out, normalization = output_scales(x, y, d)
    elif config.get("output_normalization") == "explicit":
        out = np.asarray(config["output_scales"])
        normalization = {
            "method": "explicit",
            "scales_native": out.tolist(),
            "loss_targets_used": False,
        }
    target = y[:, :learned].copy()
    if d == 6:
        target[:, 4] -= x[:, 4]
    if config["base"].startswith("theory_"):
        from element_surrogate.theory import predict_torch

        target -= predict_torch(
            torch.as_tensor(x, dtype=getattr(torch, config["dtype"])), config["theory_contract"], d
        ).numpy()
    powers, mask = base_layout(config)
    affine = np.zeros((len(powers), learned))
    if config["base"] in ("train_affine", "train_polynomial"):
        design = _monomials(normalized, powers)
        for j in range(learned):
            if mask[:, j].any():
                affine[mask[:, j], j] = np.linalg.lstsq(
                    design[:, mask[:, j]], target[:, j] / out[j], rcond=None
                )[0]
    state = {
        "input_mean": np.zeros(work.shape[1]),
        "input_scale": scale,
        "invariant_mean": f_mean,
        "invariant_scale": f_scale,
        "affine": affine,
        "output_scale": out,
        "loss_scales": config["loss_scales"],
    }
    if config.get("residual_gate") == "coupling":
        state["coupling_gate_scale"] = float(scale[-1])
    if not all(np.isfinite(value).all() for value in state.values()):
        raise ValueError("nonfinite reflection features or normalization")
    adapter = Adapter()
    adapter.config, adapter.dtype, adapter.device = (
        config,
        getattr(torch, config["dtype"]),
        torch.device(config["device"]),
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        adapter.module = module(config, state).to(adapter.device)
        torch.nn.init.zeros_(adapter.module.network[-1].weight)
        torch.nn.init.zeros_(adapter.module.network[-1].bias)
        if config.get("symmetry_breaking"):
            torch.nn.init.zeros_(adapter.module.breaking_network[-1].weight)
            torch.nn.init.zeros_(adapter.module.breaking_network[-1].bias)
    adapter.normalization_report = {
        **normalization,
        "input_method": "zero_centered_rms",
        "invariant_statistics": "train only",
        "network_evaluations_per_prediction": 2 if config.get("symmetry_breaking") else 1,
        "complete_reflection_equivariance_enforced": not config.get("symmetry_breaking", False),
    }
    adapter.x, adapter.y, adapter.rng = x, y, np.random.default_rng(seed)
    adapter.optimizer = torch.optim.Adam(adapter.module.parameters(), lr=config["learning_rate"])
    return adapter


def load(path, config):
    import torch

    from element_surrogate.neural import Adapter, _configuration
    from element_surrogate.resources import configure_torch_threads

    _configuration(config)
    configure_torch_threads(torch, config["threads"])
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["configuration_json"] != json.dumps(config, sort_keys=True):
        raise ValueError("checkpoint neural configuration differs from protocol")
    state = saved["state_dict"]
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError("nonfinite reflection-feature checkpoint")
    if torch.count_nonzero(state["input_mean"]):
        raise ValueError("reflection center differs from the declared native origin")
    invariant, _, _, _ = layout(config)
    powers, mask = base_layout(config)
    for name, expected in (
        ("invariant_powers", invariant),
        ("base_powers", powers),
        ("base_mask", mask),
    ):
        if not np.array_equal(state[name].numpy(), expected):
            raise ValueError("saved reflection characters/basis differ from configuration")
    for name in ("input_scale", "invariant_scale", "output_scale"):
        if torch.any(state[name] <= 0):
            raise ValueError("positive saved reflection-feature scales required")
    adapter = Adapter()
    adapter.config, adapter.dtype, adapter.device = (
        config,
        getattr(torch, config["dtype"]),
        torch.device(config["device"]),
    )
    adapter.module = module(config, state).to(adapter.device)
    adapter.module.load_state_dict(state, strict=True)
    adapter.module.eval()
    adapter.normalization_report = json.loads(saved.get("normalization_json", "{}"))
    return adapter


def description(config):
    invariant, gates, _, _ = layout(config)
    _, mask = base_layout(config)
    widths = [len(invariant), *config["hidden_widths"], sum(map(len, gates))]
    parameters = sum((a + 1) * b for a, b in zip(widths[:-1], widths[1:], strict=True))
    if config.get("symmetry_breaking"):
        d = config["dimension"]
        bw = [
            (d if d != 6 else d - 1) + 1,
            *config["breaking_hidden_widths"],
            d if d == 4 else d - 1,
        ]
        parameters += sum((a + 1) * b for a, b in zip(bw[:-1], bw[1:], strict=True))
    fitted = int(mask.sum()) if config["base"] in ("train_affine", "train_polynomial") else 0
    return (
        f"Native {config['dimension']}D single-pass reflection-feature map; "
        f"{len(invariant)} dimensionless invariant features, train-only invariant normalization; "
        f"tanh widths {config['hidden_widths']}, {parameters} optimized parameters; "
        f"base {config['base']} with {fitted} permitted train-fitted coefficients; "
        f"residual gate {config.get('residual_gate', 'none')}; "
        "zero-centered native coordinate scales, character-matched output gates; "
        + (
            (
                "Equivariant main branch plus an unconstrained symmetry-breaking "
                "branch, two network evaluations; no complete-map reflection "
                "guarantee. "
            )
            if config.get("symmetry_breaking")
            else (
                "One network evaluation, no reflection averaging; complete map "
                "enforces the measured reflection assumption. "
            )
        )
        + "Momentum passthrough and "
        "arrival increments follow the static-map dimension. Measured reflection "
        "assumption, not a physical-symmetry proof or symplectic guarantee."
    )
