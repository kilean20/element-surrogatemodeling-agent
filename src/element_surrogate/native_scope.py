"""Declared native API envelopes; no new tracking or physical qualification."""

from copy import deepcopy

import numpy as np

from element_surrogate.benchmark.protocol import file_hash, read_json

CONTROLS = ("effective_rigidity_tm", "field_scale")


def validate(physics, dimension):
    """Validate the scope already required by the native prediction API."""
    if not isinstance(physics, dict):
        raise ValueError("native API scope metadata must be an object")
    missing = {"qualified_ranges", "qualified_settings"} - set(physics)
    if missing:
        raise ValueError(
            "native API input/control scope is missing: "
            + ", ".join(sorted(missing))
            + "; re-export with an explicit bound source contract"
        )
    bounds = np.asarray(physics["qualified_ranges"])
    if (
        bounds.shape != (dimension, 2)
        or bounds.dtype.kind not in "fiu"
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] > bounds[:, 1])
    ):
        raise ValueError("finite ordered native scope ranges required for every coordinate")
    settings = physics["qualified_settings"]
    if not isinstance(settings, list) or not settings:
        raise ValueError("nonempty declared native control settings required")
    try:
        values = np.asarray([[row[key] for key in CONTROLS] for row in settings])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("native scope needs finite signed rigidity and field scale") from exc
    if (
        values.shape != (len(settings), 2)
        or values.dtype.kind not in "fiu"
        or not np.isfinite(values).all()
        or np.any(values[:, 0] == 0)
    ):
        raise ValueError("native scope needs finite nonzero signed rigidity and finite scale")
    interval = physics.get("effective_coupling_range")
    if interval is not None:
        interval = np.asarray(interval)
        if (
            interval.shape != (2,)
            or interval.dtype.kind not in "fiu"
            or not np.isfinite(interval).all()
            or interval[0] > interval[1]
        ):
            raise ValueError("finite ordered explicitly declared coupling interval required")
    return bounds, values, interval


def _settings_key(settings):
    # IDs/order carry provenance; prediction membership uses the two controls.
    return sorted((float(row[CONTROLS[0]]), float(row[CONTROLS[1]])) for row in settings)


def export_physics(protocol):
    """Resolve a frozen supplied declaration without fabricating a generation record.

    Legacy qualified_* names are retained for API compatibility. For supplied
    labels these are declared envelopes, not newly qualified physical domains.
    Missing intervals remain discrete settings; no extrema-based widening.
    """
    physics = deepcopy(protocol["physics"])
    dataset = protocol["dataset"]
    source = dataset.get("supplied_source_contract")
    if source is not None:
        if not isinstance(source, dict):
            raise ValueError("supplied native source contract must be an object")
        imported = dataset.get("supplied_import_request", {})
        if (
            not isinstance(imported, dict)
            or not imported.get("path")
            or file_hash(imported["path"]) != imported.get("sha256")
            or read_json(imported["path"]).get("source_contract") != source
        ):
            raise ValueError(
                "supplied native scope declaration changed or is not bound to its import"
            )
        if source.get("transport") != physics.get("transport_description"):
            raise ValueError("supplied native scope belongs to another transport")
        declared = {
            "qualified_ranges": source.get("ranges"),
            "qualified_settings": source.get("settings"),
            "effective_coupling_range": source.get("effective_coupling_range"),
        }
        validate(declared, len(protocol["coordinates"]))
        validate(
            {**declared, **{key: physics[key] for key in declared if key in physics}},
            len(protocol["coordinates"]),
        )
        for key, value in declared.items():
            if key in physics:
                a, b = physics[key], value
                same = (
                    _settings_key(a) == _settings_key(b) if key == "qualified_settings" else a == b
                )
                if not same:
                    raise ValueError("conflicting native and supplied scope: " + key)
            else:
                physics[key] = deepcopy(value)
        physics["input_scope_provenance"] = {
            "kind": "source_declared",
            "source_declaration": deepcopy(imported),
            "new_physical_qualification": False,
            "scope": "Declared entrance/control envelope, conditional on supported trajectories; "
            "not a physical aperture or uniform accuracy/survival guarantee.",
        }
    validate(physics, len(protocol["coordinates"]))
    return physics


def mask(inputs, physics, dimension):
    """The same native box/control membership for export and runtime prediction."""
    a = np.asarray(inputs, dtype=float)
    if a.ndim != 2 or a.shape[1] != dimension + 2 or not np.isfinite(a).all():
        raise ValueError("finite native state plus effective rigidity/scale inputs required")
    box = np.asarray(physics["qualified_ranges"])
    inside = np.all((a[:, :dimension] >= box[:, 0]) & (a[:, :dimension] <= box[:, 1]), axis=1)
    interval = physics.get("effective_coupling_range")
    if interval is not None:
        nonzero = a[:, -2] != 0
        coupling = np.divide(a[:, -1], a[:, -2], out=np.full(len(a), np.nan), where=nonzero)
        inside &= nonzero & (coupling >= interval[0]) & (coupling <= interval[1])
    else:
        inside &= np.any(
            [
                (a[:, -2] == row["effective_rigidity_tm"]) & (a[:, -1] == row["field_scale"])
                for row in physics["qualified_settings"]
            ],
            axis=0,
        )
    return a, inside
