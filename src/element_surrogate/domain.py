"""Numerical reconstruction support, distinct from a material aperture.

The centre of a particle can be inside a region while its reconstruction reads
outside nodes. Instrument the *same index selectors used by sampling*. A strict
support field rejects such queries before sampling, including rejected RK stages
and the final plane correction. This is a per-call invariant, not an inference
from a mesh spacing or a perturbation response.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from dataclasses import field as dcfield
from typing import Any

import numpy as np

from element_surrogate.adaptive import track_adaptive
from element_surrogate.fields import FieldMap
from element_surrogate.lorentz import track
from element_surrogate.readings import reconstruct
from element_surrogate.region import Region


def node_positions(grid: FieldMap, indices: np.ndarray) -> np.ndarray:
    """Laboratory positions of actual reconstruction nodes, in metres."""
    return np.stack([a[indices[..., k]] for k, a in enumerate((grid.x_m, grid.y_m, grid.z_m))], -1)


@dataclass
class SupportedField:
    """A grid reconstruction whose every accessed node must lie in ``support``.

    With ``enforce=False`` queries are observed but not rejected. Diagnostic
    counts include repeated accesses and rejected integration stages; they are
    not unique node counts or accepted-trajectory-only statistics.
    """

    field: FieldMap
    support: Region
    reconstruction: str = "total-degree-two"
    enforce: bool = True
    queries: int = dcfield(default=0, init=False)
    outside_queries: int = dcfield(default=0, init=False)
    outside_node_reads: int = dcfield(default=0, init=False)
    _reader: Any = dcfield(init=False, repr=False)

    def __post_init__(self):
        self._reader = reconstruct(self.field, self.reconstruction)

    def describe(self):
        return {
            **self._reader.describe(),
            "support": self.support.as_dict(),
            "support_policy": "every-read-node" if self.enforce else "observed-only",
        }

    def contains(self, points_m):
        points = np.asarray(points_m, dtype=float)
        result = self.field.contains(points) & np.isfinite(points).all(axis=-1)
        if self.enforce and np.any(result):
            nodes = self._reader.stencil_nodes(points[result])
            result = result.copy()
            result[result] = self.support.contains(node_positions(self.field, nodes)).all(axis=-1)
        return result

    def sample(self, points_m):
        points = np.asarray(points_m, dtype=float)
        nodes = self._reader.stencil_nodes(points)
        supported = self.support.contains(node_positions(self.field, nodes))
        self.queries += int(np.prod(points.shape[:-1]))
        self.outside_queries += int(np.count_nonzero(~supported.all(axis=-1)))
        self.outside_node_reads += int(np.count_nonzero(~supported))
        if self.enforce and not supported.all():
            raise ValueError("reconstruction stencil leaves the declared numerical support")
        return self._reader.sample(points)

    def summary(self):
        return {
            "queries": self.queries,
            "queries_with_outside_nodes": self.outside_queries,
            "outside_node_reads_with_repetition": self.outside_node_reads,
            "all_sampled_stencils_supported": self.outside_queries == 0,
            "scope": "all sampled queries, including rejected stages; not untested particles",
        }


class _OverlayValues:
    """On-read additive perturbation; never copies or modifies the supplied map."""

    def __init__(self, grid, support, amplitude_t):
        self.grid, self.support, self.amplitude_t = grid, support, amplitude_t

    def __getitem__(self, indices):
        value = self.grid.b_t[indices].copy()
        broadcast = np.broadcast_arrays(*indices)
        nodes = np.stack(broadcast, -1)
        outside = ~self.support.contains(node_positions(self.grid, nodes))
        # Three independent index patterns exercise zero-valued components too.
        delta = self.amplitude_t * np.stack([1 + (nodes[..., k] % 3) / 3 for k in range(3)], -1)
        value[outside] += delta[outside]
        return value


def perturb_outside(grid: FieldMap, support: Region, amplitude_t: float) -> FieldMap:
    """A probe-only overlay. Positive amplitude is supplied, never magnet-derived.

    This view is for reconstruction sampling only; do not save/hash it as an
    original field. It changes even an originally zero outside field. A negative
    control that does not respond is inconclusive, not an isolation certificate.
    """
    if not np.isfinite(amplitude_t) or amplitude_t <= 0:
        raise ValueError("positive finite perturbation amplitude in tesla required")
    overlay = copy.copy(grid)
    object.__setattr__(overlay, "b_t", _OverlayValues(grid, support, amplitude_t))
    object.__setattr__(overlay, "source", None)
    return overlay


def advance(field_map, states, rigidity_tm, entry, exit, integration, *, region=None):
    """One explicit backend selection shared by qualification and generation."""
    options = dict(integration)
    method = options.pop("method", None)
    if method == "adaptive-rk4":
        return track_adaptive(field_map, states, rigidity_tm, entry, exit, region=region, **options)
    if method == "fixed-rk4":
        return track(field_map, states, rigidity_tm, entry, exit, region=region, **options)
    raise ValueError("explicit integration method fixed-rk4 or adaptive-rk4 required")


def probe_domain(
    grid,
    states,
    rigidity_tm,
    entry,
    exit,
    support,
    trajectory_region,
    integration,
    *,
    reconstruction="total-degree-two",
    perturbation_t=None,
):
    """Audit supplied probes individually so lost probes do not mask survivors.

    Retained-particle isolation means all of that probe's sampled stencils are
    supported. Perturbation is optional corroboration and reports its empirical
    response separately. Nothing here infers a mechanical/material boundary.
    """
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] != 4 or not len(states) or not np.isfinite(states).all():
        raise ValueError("nonempty finite probe states of shape (N,4) required")
    rigidities = np.broadcast_to(np.asarray(rigidity_tm, dtype=float), (len(states),))
    rows = []
    perturbed = None if perturbation_t is None else perturb_outside(grid, support, perturbation_t)
    for state, rigidity in zip(states, rigidities, strict=True):
        observed = SupportedField(grid, support, reconstruction, enforce=False)
        result = advance(
            observed, state[None], rigidity, entry, exit, integration, region=trajectory_region
        )
        row = {
            "status": int(result.status[0]),
            "reached_exit": bool(result.reached_exit[0]),
            **observed.summary(),
        }
        if perturbed is not None:
            changed = advance(
                reconstruct(perturbed, reconstruction),
                state[None],
                rigidity,
                entry,
                exit,
                integration,
                region=trajectory_region,
            )
            row["perturbation_status_changed"] = bool(result.status[0] != changed.status[0])
            both = bool(result.reached_exit[0] and changed.reached_exit[0])
            row["perturbation_exactly_unchanged"] = bool(
                both
                and np.array_equal(result.states, changed.states)
                and np.array_equal(result.path_length_m, changed.path_length_m)
            )
            row["max_perturbation_change_m_rad"] = (
                np.abs(result.states[0] - changed.states[0]).tolist() if both else None
            )
        rows.append(row)
    survivors = [r for r in rows if r["reached_exit"]]
    outside = sum(not r["all_sampled_stencils_supported"] for r in survivors)
    return {
        "probe_count": len(rows),
        "survivors": len(survivors),
        "survivors_reading_outside_support": outside,
        "all_retained_probes_isolated": bool(survivors) and outside == 0,
        "support": support.as_dict(),
        "trajectory_region": trajectory_region.as_dict(),
        "reconstruction": reconstruct(grid, reconstruction).describe(),
        "integration": integration,
        "perturbation_amplitude_t": perturbation_t,
        "probe_results": rows,
        "findings": (
            [
                {
                    "kind": "outside_stencil_support",
                    "criterion": (
                        "at least one retained probe samples a node outside declared support"
                    ),
                    "count": outside,
                }
            ]
            if outside
            else []
        ),
        "scope": (
            "supplied probes and numerical configuration only; not a material"
            " aperture or absolute field accuracy certificate"
        ),
    }
