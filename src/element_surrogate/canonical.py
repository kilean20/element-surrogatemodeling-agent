"""Conditional potential shears in eligible 4D canonical plane coordinates.

Each q+=grad(T(p,c)), p-=grad(V(q,c)) is symplectic for fixed control c.
Slopes are converted to mechanical momentum; these are canonical only under
the recorded zero-normal-field condition. No clipping or reflection averaging.
"""

import numpy as np

from element_surrogate.resources import configure_torch_threads
from element_surrogate.runs import implementation_identity, sha256_json
from element_surrogate.transport import scope_description, verify_transport_source


def assess(
    transport, ranges, settings, *, q_scale_m, p_scale, rationale, effective_coupling_range=None
):
    verify_transport_source(transport)
    if (
        transport.codec.dimension != 4
        or not np.isfinite([q_scale_m, p_scale]).all()
        or min(q_scale_m, p_scale) <= 0
        or not isinstance(rationale, str)
        or not rationale.strip()
    ):
        raise ValueError("4D map, positive canonical scales and scientific rationale required")
    # Conservative sufficient condition, covering tilted planes too. This is
    # stricter than merely Bn=0 on each boundary. Linear component-wise grid
    # reconstruction preserves this condition throughout the sampled volume.
    maxima = [0.0, 0.0]
    iterator = np.nditer(
        transport.grid.b_t,
        flags=["external_loop", "buffered"],
        op_flags=["readonly"],
        order="C",
        buffersize=3 * (1 << 15),
    )
    for values in iterator:
        block = np.asarray(values).reshape(-1, 3)
        if not np.isfinite(block).all():
            raise ValueError("non-finite field cannot establish canonical coordinates")
        for j, frame in enumerate((transport.entry, transport.exit)):
            maxima[j] = max(maxima[j], float(np.max(np.abs(block @ frame.tangent))))
    eligible = all(v == 0 for v in maxima)
    return {
        "schema": "element_surrogate.canonical_plane.v1",
        "scope": scope_description(transport, ranges, settings, effective_coupling_range),
        "q_scale_m": q_scale_m,
        "p_scale": p_scale,
        "eligible": eligible,
        "maximum_normal_component_t": maxima,
        "rationale": rationale,
        "criterion": (
            "Exactly zero normal component for both native-plane normals "
            "at every supplied grid node; conservative sufficient conditi"
            "on"
        ),
        "implementation_sha256": sha256_json(implementation_identity()),
        "limits": (
            "Canonical 4D construction for the reconstructed field. Does "
            "not certify physical field accuracy. Nonzero boundary normal"
            " field needs a gauge/coordinate construction absent from thi"
            "s backend."
        ),
    }


def validate(config, expected_scope=None, current=False):
    record = config.get("canonical_contract")
    if (
        config.get("dimension") != 4
        or config.get("base") != "none"
        or len(config.get("hidden_widths", [])) != 1
    ):
        raise ValueError(
            "canonical shear requires a 4D map, one potential width and no additive residual base"
        )
    if type(config.get("shear_pairs")) is not int or config["shear_pairs"] < 1:
        raise ValueError("positive declared potential-shear pair count required")
    if len(config.get("reflection_group", [[1] * 6])) != 1:
        raise ValueError(
            "averaging symplectic maps is not symplectic; reflection projection is ineligible"
        )
    if (
        not isinstance(record, dict)
        or record.get("schema") != "element_surrogate.canonical_plane.v1"
        or record.get("eligible") is not True
        or record.get("maximum_normal_component_t") != [0.0, 0.0]
    ):
        raise ValueError("eligible zero-normal-field canonical-plane assessment required")
    if (
        not np.isfinite([record["q_scale_m"], record["p_scale"]]).all()
        or min(record["q_scale_m"], record["p_scale"]) <= 0
    ):
        raise ValueError("positive common canonical scales required")
    if expected_scope is not None and record["scope"] != expected_scope:
        raise ValueError("canonical-plane scope differs")
    if current and record["implementation_sha256"] != sha256_json(implementation_identity()):
        raise ValueError("canonical-plane implementation changed")


def module(config, coupling_mean, coupling_scale):
    import torch
    from torch import nn

    validate(config)
    dtype = getattr(torch, config["dtype"])
    width = config["hidden_widths"][0]

    class Potential(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(width, 2, dtype=dtype) / np.sqrt(2.0))
            self.control = nn.Parameter(torch.randn(width, dtype=dtype))
            self.bias = nn.Parameter(torch.zeros(width, dtype=dtype))
            self.amplitude = nn.Parameter(torch.zeros(width, dtype=dtype))

        def gradient(self, state, control):
            activation = torch.tanh(state @ self.weight.T + control * self.control + self.bias)
            return ((1 - activation * activation) * self.amplitude) @ self.weight

    class CanonicalMap(nn.Module):
        def __init__(self):
            super().__init__()
            self.dimension = 4
            self.register_buffer("loss_scales", torch.as_tensor(config["loss_scales"], dtype=dtype))
            self.pairs = nn.ModuleList(
                [nn.ModuleList([Potential(), Potential()]) for _ in range(config["shear_pairs"])]
            )
            self.register_buffer("coupling_mean", torch.as_tensor(coupling_mean, dtype=dtype))
            self.register_buffer("coupling_scale", torch.as_tensor(coupling_scale, dtype=dtype))
            self.q_scale = config["canonical_contract"]["q_scale_m"]
            self.p_scale = config["canonical_contract"]["p_scale"]

        def forward(self, inputs):
            q = torch.stack([inputs[:, 0], inputs[:, 2]], dim=1) / self.q_scale
            slope = torch.stack([inputs[:, 1], inputs[:, 3]], dim=1)
            p = slope / torch.sqrt(1 + torch.sum(slope * slope, dim=1, keepdim=True)) / self.p_scale
            c = (inputs[:, -1:] / inputs[:, -2:-1] - self.coupling_mean) / self.coupling_scale
            for up, down in self.pairs:
                q = q + up.gradient(p, c)
                p = p - down.gradient(q, c)
            mechanical = p * self.p_scale
            # A negative radicand is an invalid forward chart, not something
            # to clamp: clamping would destroy the claimed map structure.
            out_slope = mechanical / torch.sqrt(
                1 - torch.sum(mechanical * mechanical, dim=1, keepdim=True)
            )
            positions = q * self.q_scale
            return torch.stack(
                [positions[:, 0], out_slope[:, 0], positions[:, 1], out_slope[:, 1]], dim=1
            )

    return CanonicalMap()


def build(x, y, config, seed):
    import torch

    from element_surrogate.neural import Adapter, _configuration

    _configuration(config, x)
    if y.shape != (len(x), 4) or not np.isfinite(y).all() or not len(x):
        raise ValueError("complete finite 4D training labels required")
    coupling = x[:, -1] / x[:, -2]
    mean, scale = float(coupling.mean()), float(coupling.std()) or 1.0
    adapter = Adapter()
    adapter.config = config
    adapter.dtype = getattr(torch, config["dtype"])
    adapter.device = torch.device(config["device"])
    configure_torch_threads(torch, config["threads"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        adapter.module = module(config, mean, scale).to(adapter.device)
    adapter.x, adapter.y = x, y
    adapter.rng = np.random.default_rng(seed)
    adapter.optimizer = torch.optim.Adam(adapter.module.parameters(), lr=config["learning_rate"])
    return adapter


def load(path, config):
    import json

    import torch

    from element_surrogate.neural import Adapter, _configuration

    _configuration(config)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["configuration_json"] != json.dumps(config, sort_keys=True):
        raise ValueError("checkpoint neural configuration differs from protocol")
    state = saved["state_dict"]
    adapter = Adapter()
    adapter.config = config
    adapter.dtype = getattr(torch, config["dtype"])
    adapter.device = torch.device(config["device"])
    adapter.module = module(config, state["coupling_mean"], state["coupling_scale"])
    adapter.module.load_state_dict(state, strict=True)
    adapter.module.to(adapter.device).eval()
    return adapter


def from_files(path, request_path, output_path) -> dict:
    from element_surrogate.benchmark.protocol import file_hash, read_json, write_json
    from element_surrogate.workflow import load_transport

    transport, _ = load_transport(path)
    result = assess(transport, **read_json(request_path))
    write_json(output_path, result, exclusive=True)
    return {
        "assessment_path": str(output_path),
        "sha256": file_hash(output_path),
        "eligible": result["eligible"],
        "criterion": result["criterion"],
        "limits": result["limits"],
    }
