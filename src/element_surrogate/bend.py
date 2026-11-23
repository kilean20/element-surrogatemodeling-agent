"""The nominal path of a bending magnet, and coordinates along it.

A bending magnet's *nominal path* is the hard-edge design geometry: a straight entrance
arm, a circular arc of radius ``radius_m`` turning through ``angle_rad``, and a straight
exit arm of the same length, symmetric about the arc's midpoint. It places the field
box, the entry and exit planes and the regions that follow the beam. It is **not** the
reference orbit: the design particle, integrated through the real body and fringe
fields, departs from it inside the magnet (knowledge entry
``reference-orbit-in-bending-magnets``).

Frame, matching the TRACK ``mag3d`` field box: the origin is the arc's midpoint, z runs
along the chord (so the entrance and exit directions each make half the bend angle with
z), y is vertical, and the arc bends toward -x unless ``bends_toward_negative_x`` is
false. Along the path the transverse frame is (horizontal, vertical, tangent), right-
handed, with vertical = laboratory up and horizontal = up x tangent. The horizontal axis
therefore points away from the centre of curvature for a bend toward -x and toward it for
a bend toward +x. Simulators differ in this sign; confirm a simulator's convention before
comparing coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

_UP = np.array([0.0, 1.0, 0.0])


@dataclass(frozen=True)
class PathFrame:
    """A point on the path and its orthonormal transverse frame, in the field file's frame."""

    origin_m: np.ndarray
    tangent: np.ndarray
    horizontal: np.ndarray
    vertical: np.ndarray

    def __post_init__(self) -> None:
        for name in ("origin_m", "tangent", "horizontal", "vertical"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError("frame vectors must be finite triples")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        basis = np.stack([self.horizontal, self.vertical, self.tangent])
        if not np.allclose(basis @ basis.T, np.eye(3), atol=1e-10, rtol=0):
            raise ValueError("frame basis must be orthonormal")
        if np.linalg.det(basis) < 0:
            raise ValueError("internal frames must be right handed")


@dataclass(frozen=True)
class NominalBend:
    radius_m: float
    angle_rad: float
    arm_m: float
    bends_toward_negative_x: bool = True

    def __post_init__(self) -> None:
        values = (self.radius_m, self.angle_rad, self.arm_m)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("radius, angle and arm length must be finite")
        if self.radius_m <= 0 or not 0 < self.angle_rad < math.pi or self.arm_m < 0:
            raise ValueError("need radius > 0, 0 < angle < pi and arm length >= 0")

    @classmethod
    def from_path_length(
        cls,
        radius_m: float,
        angle_rad: float,
        path_length_m: float,
        bends_toward_negative_x: bool = True,
    ) -> NominalBend:
        """The bend whose nominal path (arm + arc + arm) has the given total length."""
        return cls(
            radius_m,
            angle_rad,
            (path_length_m - radius_m * angle_rad) / 2.0,
            bends_toward_negative_x,
        )

    @property
    def arc_length_m(self) -> float:
        return self.radius_m * self.angle_rad

    @property
    def length_m(self) -> float:
        return self.arc_length_m + 2.0 * self.arm_m

    @property
    def _sigma(self) -> float:
        """+1 for a bend toward -x, -1 for a bend toward +x."""
        return 1.0 if self.bends_toward_negative_x else -1.0

    @property
    def _centre(self) -> np.ndarray:
        return np.array([-self._sigma * self.radius_m, 0.0, 0.0])

    def _on_arc(self, phi: float) -> tuple[np.ndarray, np.ndarray]:
        """Position and tangent at arc angle ``phi`` (0 at the midpoint, rising along the path)."""
        sigma = self._sigma
        outward = np.array([sigma * math.cos(phi), 0.0, math.sin(phi)])
        tangent = np.array([-sigma * math.sin(phi), 0.0, math.cos(phi)])
        return self._centre + self.radius_m * outward, tangent

    def frame_at(self, s_m: float) -> PathFrame:
        """The nominal path's point and transverse frame at path length ``s_m`` from the entry."""
        if not 0.0 <= s_m <= self.length_m * (1 + 1e-12):
            raise ValueError("s_m lies outside the nominal path")
        half = self.angle_rad / 2.0
        arc_end = self.arm_m + self.arc_length_m
        if s_m < self.arm_m:
            point, tangent = self._on_arc(-half)
            point = point - (self.arm_m - s_m) * tangent
        elif s_m <= arc_end:
            point, tangent = self._on_arc(-half + (s_m - self.arm_m) / self.radius_m)
        else:
            point, tangent = self._on_arc(half)
            point = point + (s_m - arc_end) * tangent
        horizontal = np.cross(_UP, tangent)
        horizontal /= np.linalg.norm(horizontal)
        return PathFrame(point, tangent, horizontal, np.cross(tangent, horizontal))

    def entry_frame(self) -> PathFrame:
        return self.frame_at(0.0)

    def exit_frame(self) -> PathFrame:
        return self.frame_at(self.length_m)

    def path_coordinates(self, points_m: np.ndarray) -> np.ndarray:
        """``(s, horizontal, vertical)`` of each point relative to the nominal path, ``(..., 3)``.

        Space is divided by the planes normal to the path at the two ends of the arc: points
        between them are measured from the arc (``s`` from their angle about the centre of
        curvature), the others from the straight arm on their side. ``s`` runs from the entry
        plane (negative before it, beyond ``length_m`` after the exit). Valid for points closer
        to the path than its radius of curvature.
        """
        p = np.asarray(points_m, dtype=np.float64)
        sigma, half = self._sigma, self.angle_rad / 2.0
        rel = p - self._centre
        phi = np.arctan2(rel[..., 2], sigma * rel[..., 0])
        out = np.empty(p.shape)
        out[..., 0] = self.arm_m + (phi + half) * self.radius_m
        out[..., 1] = sigma * (np.hypot(rel[..., 0], rel[..., 2]) - self.radius_m)
        out[..., 2] = p[..., 1]
        for s_end, on_arm in (
            (self.arm_m, phi < -half),
            (self.arm_m + self.arc_length_m, phi > half),
        ):
            end = self.frame_at(s_end)
            d = p[on_arm] - end.origin_m
            out[on_arm, 0] = s_end + d @ end.tangent
            out[on_arm, 1] = d @ end.horizontal
            out[on_arm, 2] = d @ end.vertical
        return out

    def to_request(self) -> dict[str, Any]:
        """Accepted input keys; descriptive/derived fields are deliberately separate."""
        return {
            "radius_m": self.radius_m,
            "angle_rad": self.angle_rad,
            "arm_m": self.arm_m,
            "bends_toward": "-x" if self.bends_toward_negative_x else "+x",
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "radius_m": self.radius_m,
            "angle_rad": self.angle_rad,
            "arm_m": self.arm_m,
            "path_length_m": self.length_m,
            "bends_toward": "-x" if self.bends_toward_negative_x else "+x",
            "frame": "origin at the arc midpoint, z along the chord, y vertical",
        }
