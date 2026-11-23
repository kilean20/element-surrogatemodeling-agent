"""Native-span analytical bases; approximations, never placement wrappers."""

import math

import numpy as np

from element_surrogate.benchmark.protocol import protocol_identity, read_json, write_json
from element_surrogate.runs import implementation_identity, sha256_json
from element_surrogate.transport import scope_description, verify_transport_source


def _maximum_argument(gradient, length, scope, dimension):
    deviations = [0.0] if dimension == 4 else np.asarray(scope["ranges"])[-1].tolist()
    return max(
        abs(gradient * s["field_scale"] / s["effective_rigidity_tm"] / (1 + delta)) * length**2
        for s in scope["settings"]
        for delta in deviations
    )


def assess(
    transport,
    kind,
    rationale,
    gradient_t_m=None,
    *,
    ranges=None,
    settings=None,
    effective_coupling_range=None,
):
    verify_transport_source(transport)
    if (
        kind not in ("drift", "paraxial_quadrupole")
        or not isinstance(rationale, str)
        or not rationale.strip()
    ):
        raise ValueError("declared analytical base and scientific rationale required")
    entry, exit = transport.entry, transport.exit
    delta = exit.origin_m - entry.origin_m
    length = float(delta @ entry.tangent)
    if (
        length <= 0
        or not np.allclose(delta, length * entry.tangent, atol=1e-12, rtol=0)
        or any(
            not np.allclose(getattr(entry, k), getattr(exit, k), atol=1e-12, rtol=0)
            for k in ("tangent", "horizontal", "vertical")
        )
    ):
        raise ValueError(
            "analytical base needs aligned straight native planes; no placement adjustment"
        )
    if kind == "paraxial_quadrupole" and (
        transport.codec.dimension == 6 or gradient_t_m is None or not np.isfinite(gradient_t_m)
    ):
        raise ValueError("quadrupole base requires a finite declared local gradient and 4D/5D map")
    scope = None
    maximum, scalings = 0.0, 0
    if kind == "paraxial_quadrupole":
        if ranges is None or not settings:
            raise ValueError("quadrupole matrix kernel needs declared ranges/settings")
        scope = scope_description(transport, ranges, settings, effective_coupling_range)
        maximum = _maximum_argument(gradient_t_m, length, scope, transport.codec.dimension)
        if not np.isfinite(maximum):
            raise ValueError("finite matrix argument bound required")
        scalings = max(0, math.ceil(math.log2(maximum) / 2)) if maximum > 1 else 0
    return {
        "schema": "element_surrogate.native_theory_base.v1",
        "kind": kind,
        "transport": transport.description(),
        "length_m": length,
        "gradient_t_m": float(gradient_t_m) if kind == "paraxial_quadrupole" else None,
        "rationale": rationale,
        "implementation_sha256": sha256_json(implementation_identity()),
        "approximation": "Ideal field-free straight drift"
        if kind == "drift"
        else (
            "Paraxial constant-gradient quadrupole over the whole native "
            "span; real fringes/nonlinearities belong in the residual"
        ),
        "placement_wrapping": False,
        "scope": scope,
        "maximum_abs_kappa_l2": maximum,
        "matrix_squarings": scalings,
        "matrix_kernel": (
            "degree-12 entire-function Taylor kernel at |kappa*(L/2**s)^2"
            "| <= 1, then matrix squaring"
        ),
    }


def validate(record, kind, dimension, expected_transport=None, current=False, expected_scope=None):
    if (
        not isinstance(record, dict)
        or record.get("schema") != "element_surrogate.native_theory_base.v1"
        or record.get("kind") != kind
    ):
        raise ValueError("bound analytical-base assessment required")
    length = record.get("length_m")
    if (
        not isinstance(length, (int, float))
        or not np.isfinite(length)
        or length <= 0
        or record.get("placement_wrapping") is not False
    ):
        raise ValueError("positive native span without placement wrapping required")
    if record["transport"]["codec"]["dimension"] != dimension:
        raise ValueError("theory dimension differs")
    entry, exit = record["transport"]["entry"], record["transport"]["exit"]
    span = np.asarray(exit["origin_m"]) - np.asarray(entry["origin_m"])
    if length != float(span @ np.asarray(entry["tangent"])):
        raise ValueError("analytical length differs from the native span")
    if kind == "paraxial_quadrupole" and (
        dimension == 6 or not np.isfinite(record["gradient_t_m"])
    ):
        raise ValueError("quadrupole base supports only 4D/5D")
    if kind == "paraxial_quadrupole":
        maximum = record.get("maximum_abs_kappa_l2")
        scope = record.get("scope")
        if (
            not isinstance(scope, dict)
            or scope.get("transport") != record["transport"]
            or maximum != _maximum_argument(record["gradient_t_m"], length, scope, dimension)
        ):
            raise ValueError("matrix argument bound differs from declared controls")
        if (
            not isinstance(maximum, (int, float))
            or not np.isfinite(maximum)
            or maximum < 0
            or record.get("matrix_squarings")
            != (max(0, math.ceil(math.log2(maximum) / 2)) if maximum > 1 else 0)
        ):
            raise ValueError("consistent bounded matrix kernel required")
        if expected_scope is not None and record.get("scope") != expected_scope:
            raise ValueError("theory-base range/control scope differs")
    if expected_transport is not None and record["transport"] != expected_transport:
        raise ValueError("theory-base transport scope differs")
    if current and record["implementation_sha256"] != sha256_json(implementation_identity()):
        raise ValueError("theory-base implementation changed")
    if not record.get("rationale"):
        raise ValueError("theory rationale required")


def validate_inputs(record, inputs, dimension):
    if record["kind"] != "paraxial_quadrupole":
        return
    delta = inputs[:, dimension - 1] if dimension > 4 else 0.0
    argument = (
        np.abs(record["gradient_t_m"] * inputs[:, -1] / inputs[:, -2] / (1 + delta))
        * record["length_m"] ** 2
    )
    limit = record["maximum_abs_kappa_l2"]
    if np.any(
        (argument > limit) & ~np.isclose(argument, limit, rtol=8 * np.finfo(float).eps, atol=0)
    ):
        raise ValueError("training controls exceed the analytical matrix argument bound")


def predict_torch(inputs, contract, dimension):
    """Native SI state; returns only learned outputs, arrival as an increment."""
    import torch

    length = contract["length_m"]
    x, xp, y, yp = [inputs[:, j : j + 1] for j in range(4)]
    if contract["kind"] == "drift":
        result = [x + length * xp, xp, y + length * yp, yp]
    else:
        delta = inputs[:, dimension - 1 : dimension] if dimension > 4 else 0.0
        k = contract["gradient_t_m"] * inputs[:, -1:] / inputs[:, -2:-1] / (1 + delta)

        def plane(q, p, kappa):
            scalings = contract["matrix_squarings"]
            reduced_length = length / (2**scalings)
            argument = -kappa * reduced_length**2
            cosine, sine = 1 / math.factorial(24), 1 / math.factorial(25)
            for j in range(11, -1, -1):
                cosine = 1 / math.factorial(2 * j) + argument * cosine
                sine = 1 / math.factorial(2 * j + 1) + argument * sine
            sine = sine * reduced_length
            # At |argument|<=1 the next terms are below 1/26! and 1/27!.
            # Multiplication/addition preserve double ONNX support and a
            # finite input derivative at zero strength.
            for _ in range(scalings):
                cosine, sine = cosine * cosine - kappa * sine * sine, 2 * cosine * sine
            return cosine * q + sine * p, -kappa * sine * q + cosine * p

        q, p = plane(x, xp, k)
        v, w = plane(y, yp, -k)
        result = [q, p, v, w]
    if dimension == 6:
        path = length * torch.sqrt(1 + xp * xp + yp * yp)
        desc = contract["transport"]
        reference = desc["reference_path_m"]
        if desc["codec"]["longitudinal"] == "path_m":
            arrival = path - reference
        else:
            c = 299792458.0
            p0, m = desc["actual_p0_kg_m_s"], desc["mass_kg"]
            ratio = (p0 / (m * c)) * (1 + inputs[:, 5:6])
            speed = c * ratio / torch.sqrt(1 + ratio * ratio)
            reference_ratio = p0 / (m * c)
            reference_speed = c * reference_ratio / np.sqrt(1 + reference_ratio**2)
            arrival = path / speed - reference / reference_speed
        result.append(arrival)
    return torch.cat(result, dim=1)


def from_files(path, request_path, output_path) -> dict:
    from element_surrogate.workflow import load_transport

    transport, _ = load_transport(path)
    record = assess(transport, **read_json(request_path))
    write_json(output_path, record, exclusive=True)
    return {
        "assessment_path": str(output_path),
        "sha256": protocol_identity(record),
        "kind": record["kind"],
        "approximation": record["approximation"],
    }
