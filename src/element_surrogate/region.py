"""Where a check or a fit applies.

A ``Region`` is a prism about an axis: the file's z axis, or a bending magnet's nominal
path (``path``, a ``bend.NominalBend``). Its cross-section is a circle of radius
``radius_m``, a square of half-width ``radius_m``, or a rectangle of half-width
``radius_m`` and half-height ``half_height_m``, in the transverse offsets from the axis
(x and y about the z axis; the path's horizontal and vertical offsets about a path). It
may be limited along the axis: in z about the z axis, and in path length measured from
the path's midpoint about a path.

Its *radial measure* is the distance that defines it: ``hypot(h, v)`` for a circle,
``max(|h|, |v|)`` for a square and ``max(|h|, |v| * radius_m / half_height_m)`` for a
rectangle, for horizontal and vertical offsets ``h`` and ``v``. Shells of a region are
level sets of that measure (scaled copies of its cross-section). A cell belongs to a
region when all eight of its corners do.

Every check and fit takes a region, so their numbers describe the same volume. A region
selects nodes and cells only: derivatives stay on the file's own grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from element_surrogate.bend import NominalBend
from element_surrogate.fields import FieldMap

SHAPES = ("circle", "square", "rectangle")
_TOL = 1.0 + 1e-12
_MEASURES = {
    "circle": "hypot(h, v)",
    "square": "max(|h|, |v|)",
    "rectangle": "max(|h|, |v| * radius_m / half_height_m)",
}


def _corners(values: np.ndarray) -> list[np.ndarray]:
    """The eight corner values of every cell of a node array (broadcast axes of size 1 stay)."""
    pick = [
        (slice(None), slice(None)) if n == 1 else (slice(None, -1), slice(1, None))
        for n in values.shape
    ]
    return [values[a, b, c] for a in pick[0] for b in pick[1] for c in pick[2]]


@dataclass(frozen=True)
class Region:
    shape: str
    radius_m: float
    z_min_m: float | None = None
    z_max_m: float | None = None
    half_height_m: float | None = None
    path: NominalBend | None = None

    def __post_init__(self) -> None:
        if self.shape not in SHAPES:
            raise ValueError(f"shape must be one of {SHAPES}, got {self.shape!r}")
        if not np.isfinite(self.radius_m) or not self.radius_m > 0.0:
            raise ValueError("radius_m must be positive")
        if (self.shape == "rectangle") != (self.half_height_m is not None):
            raise ValueError("half_height_m is given for a rectangle, and only for a rectangle")
        if self.half_height_m is not None and (
            not np.isfinite(self.half_height_m) or not self.half_height_m > 0.0
        ):
            raise ValueError("half_height_m must be positive")
        if any(v is not None and not np.isfinite(v) for v in (self.z_min_m, self.z_max_m)):
            raise ValueError("finite longitudinal bounds required")
        if self.z_min_m is not None and self.z_max_m is not None and self.z_min_m >= self.z_max_m:
            raise ValueError("z_min_m must be below z_max_m")

    @classmethod
    def covering(
        cls,
        field: FieldMap,
        shape: str = "circle",
        z_min_m: float | None = None,
        z_max_m: float | None = None,
    ) -> Region:
        """The largest circle or square about the z axis that the grid covers."""
        if shape not in ("circle", "square"):
            raise ValueError(
                "only a circle or a square can be sized from the grid; give the region"
            )
        x, y = field.x_m, field.y_m
        if not (x[0] < 0.0 < x[-1] and y[0] < 0.0 < y[-1]):
            raise ValueError(
                "the grid does not extend to both sides of the z axis in x and y; "
                "give the region explicitly"
            )
        return cls(shape, float(min(-x[0], x[-1], -y[0], y[-1])), z_min_m, z_max_m)

    def radial(self, h: np.ndarray, v: np.ndarray) -> np.ndarray:
        """The radial measure of transverse offsets ``h`` (horizontal) and ``v`` (vertical)."""
        if self.shape == "circle":
            result: np.ndarray = np.hypot(h, v)
        elif self.shape == "square":
            result = np.maximum(np.abs(h), np.abs(v))
        else:
            assert self.half_height_m is not None
            result = np.maximum(np.abs(h), np.abs(v) * (self.radius_m / self.half_height_m))
        return result

    def _longitudinal_ok(self, values: np.ndarray) -> np.ndarray:
        ok = np.ones(values.shape, dtype=bool)
        if self.z_min_m is not None:
            ok &= values >= self.z_min_m - 1e-12
        if self.z_max_m is not None:
            ok &= values <= self.z_max_m + 1e-12
        return ok

    def z_mask(self, z: np.ndarray) -> np.ndarray:
        """Planes inside the region's z range (a region about the z axis only)."""
        self._require_straight("a z range")
        return self._longitudinal_ok(z)

    def _require_straight(self, what: str) -> None:
        if self.path is not None:
            raise ValueError(f"{what} is defined for a region about the z axis, not about a bend")

    def contains(self, points_m: np.ndarray) -> np.ndarray:
        """Whether each point ``(..., 3)`` lies inside the region."""
        p = np.asarray(points_m, dtype=np.float64)
        if self.path is None:
            h, v, along = p[..., 0], p[..., 1], p[..., 2]
        else:
            s_h_v = self.path.path_coordinates(p)
            h, v, along = s_h_v[..., 1], s_h_v[..., 2], s_h_v[..., 0] - 0.5 * self.path.length_m
        inside: np.ndarray = (self.radial(h, v) <= self.radius_m * _TOL) & self._longitudinal_ok(
            along
        )
        return inside

    def node_measures(self, field: FieldMap) -> tuple[np.ndarray, np.ndarray]:
        """Each node's radial measure and whether it lies in the longitudinal range, as arrays
        that broadcast to ``(nx, ny, nz)``."""
        x, y, z = field.x_m, field.y_m, field.z_m
        if self.path is None:
            radial = self.radial(x[:, None, None], y[None, :, None])
            return radial, self._longitudinal_ok(z)[None, None, :]
        # The nominal path lies in the plane y = 0 with vertical = laboratory up, so the
        # vertical offset is y and the rest depends on (x, z) only.
        plane = np.stack(np.meshgrid(x, np.zeros(1), z, indexing="ij"), axis=-1)[:, 0, :, :]
        s_h_v = self.path.path_coordinates(plane)
        h = s_h_v[..., 1][:, None, :]
        longitudinal = (s_h_v[..., 0] - 0.5 * self.path.length_m)[:, None, :]
        return self.radial(h, y[None, :, None]), self._longitudinal_ok(longitudinal)

    def node_mask(self, field: FieldMap) -> np.ndarray:
        """Nodes inside the region, shape ``(nx, ny, nz)``."""
        radial, along = self.node_measures(field)
        mask: np.ndarray = np.broadcast_to((radial <= self.radius_m * _TOL) & along, field.shape)
        return mask

    def transverse_mask(self, field: FieldMap) -> np.ndarray:
        """Transverse nodes inside the region, shape ``(nx, ny)`` (a region about the z axis)."""
        self._require_straight("a transverse mask")
        mask: np.ndarray = (
            self.radial(field.x_m[:, None], field.y_m[None, :]) <= self.radius_m * _TOL
        )
        return mask

    def cell_radial(self, field: FieldMap) -> np.ndarray:
        """Largest radial measure over each cell's corners; broadcasts to the cell grid
        ``(nx-1, ny-1, nz-1)``."""
        radial, _ = self.node_measures(field)
        result: np.ndarray = np.maximum.reduce(_corners(radial))
        return result

    def cell_mask(self, field: FieldMap, outer_m: float | None = None) -> np.ndarray:
        """Cells whose eight corners lie inside the region (or inside the scaled copy of
        radius ``outer_m``, if given), shape ``(nx-1, ny-1, nz-1)``."""
        limit = self.radius_m if outer_m is None else outer_m
        radial, along = self.node_measures(field)
        inside = (np.maximum.reduce(_corners(radial)) <= limit * _TOL) & np.logical_and.reduce(
            _corners(along)
        )
        cells = tuple(n - 1 for n in field.shape)
        mask: np.ndarray = np.broadcast_to(inside, cells)
        return mask

    def node_window(self, field: FieldMap) -> tuple[slice, slice, slice]:
        """Index ranges of the smallest box of nodes holding every node of the region."""
        from element_surrogate.diagnostic_blocks import blocks, node_window

        hits = [np.zeros(n, dtype=bool) for n in field.shape]
        for core in blocks(field.shape, 32768):
            mask = self.node_mask(node_window(field, core))
            for axis in range(3):
                hits[axis][core[axis]] |= mask.any(axis=tuple(a for a in range(3) if a != axis))
        if not hits[0].any():
            raise ValueError("no node lies inside the region")
        ranges = []
        for axis in range(3):
            indices = np.flatnonzero(hits[axis])
            ranges.append(slice(int(indices[0]), int(indices[-1]) + 1))
        return ranges[0], ranges[1], ranges[2]

    def to_request(self, *, origin=None) -> dict[str, Any]:
        """Serialize accepted inputs, with an explicit longitudinal origin.

        Offsets about a nominal path are not coordinates about a traced reference.
        These bounds select an evaluated region; they do not establish an aperture.
        """
        basis = "laboratory" if self.path is None else "nominal_path"
        origin = origin or ("laboratory" if self.path is None else "path_midpoint")
        allowed = {"laboratory"} if self.path is None else {"path_entrance", "path_midpoint"}
        if origin not in allowed:
            raise ValueError(f"origin must be one of {sorted(allowed)} for {basis}")
        shift = (
            0.5 * self.path.length_m if self.path is not None and origin == "path_entrance" else 0.0
        )
        result = {
            "shape": self.shape,
            "radius_m": self.radius_m,
            "coordinate_basis": basis,
            "longitudinal_bounds": {
                "origin": origin,
                "min_m": None if self.z_min_m is None else self.z_min_m + shift,
                "max_m": None if self.z_max_m is None else self.z_max_m + shift,
            },
        }
        if self.half_height_m is not None:
            result["half_height_m"] = self.half_height_m
        if self.path is not None:
            result["bend"] = self.path.to_request()
        return result

    @classmethod
    def from_request(cls, value) -> Region:
        """Normalize legacy or explicit requests into the single Region representation."""
        if not isinstance(value, dict):
            raise ValueError("region request must be an object")
        allowed = {
            "shape",
            "radius_m",
            "z_min_m",
            "z_max_m",
            "half_height_m",
            "bend",
            "coordinate_basis",
            "longitudinal_bounds",
        }
        if set(value) - allowed:
            raise ValueError(
                f"unknown region specification keys {sorted(set(value) - allowed)}; "
                "use to_request(), not as_dict()"
            )
        path = nominal_bend(value["bend"]) if value.get("bend") is not None else None
        basis = "laboratory" if path is None else "nominal_path"
        if value.get("coordinate_basis", basis) != basis:
            raise ValueError("coordinate_basis must agree with the presence of a nominal bend")
        lower, upper = value.get("z_min_m"), value.get("z_max_m")
        if "longitudinal_bounds" in value:
            if "z_min_m" in value or "z_max_m" in value:
                raise ValueError(
                    "choose explicit longitudinal_bounds or legacy z_min_m/z_max_m, never both"
                )
            bounds = value["longitudinal_bounds"]
            if not isinstance(bounds, dict) or set(bounds) != {"origin", "min_m", "max_m"}:
                raise ValueError(
                    "longitudinal_bounds requires origin, min_m and max_m (null means unbounded)"
                )
            origins = {"laboratory"} if path is None else {"path_midpoint", "path_entrance"}
            if bounds["origin"] not in origins:
                raise ValueError(f"longitudinal origin must be one of {sorted(origins)}")
            shift = (
                0.5 * path.length_m
                if path is not None and bounds["origin"] == "path_entrance"
                else 0.0
            )
            try:
                lower = None if bounds["min_m"] is None else float(bounds["min_m"]) - shift
                upper = None if bounds["max_m"] is None else float(bounds["max_m"]) - shift
            except (TypeError, ValueError) as exc:
                raise ValueError("longitudinal bounds must be numeric or null") from exc
        if "shape" not in value or "radius_m" not in value:
            raise ValueError("explicit region shape and radius_m required")
        return cls(
            value["shape"], value["radius_m"], lower, upper, value.get("half_height_m"), path
        )

    def mirror_symmetric(self, axis: int) -> bool:
        """Whether the region is unchanged by the mirror of the file's axis ``axis`` (0, 1, 2
        for x, y, z). A bend's nominal path is symmetric about y = 0 and z = 0, not x = 0."""
        if axis == 0:
            return self.path is None
        if axis == 1:
            return True
        if self.z_min_m is None and self.z_max_m is None:
            return True
        if self.z_min_m is None or self.z_max_m is None:
            return False
        return bool(np.isclose(self.z_min_m, -self.z_max_m, rtol=1e-9, atol=1e-12))

    def as_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "radius_m": self.radius_m,
            "half_height_m": self.half_height_m,
            "z_min_m": self.z_min_m,
            "z_max_m": self.z_max_m,
            "axis": "the file's z axis" if self.path is None else "a bend's nominal path",
            "path": None if self.path is None else self.path.as_dict(),
            "longitudinal_coordinate": (
                "z" if self.path is None else "path length from the path's midpoint"
            ),
            "radial_measure": _MEASURES[self.shape],
        }


def nominal_bend(bend: dict[str, Any]) -> NominalBend:
    """A nominal bend from tool arguments: ``radius_m``, ``angle_rad``, ``arm_m`` and optionally
    ``bends_toward`` ("-x", the default, or "+x")."""
    unknown = set(bend) - {"radius_m", "angle_rad", "arm_m", "bends_toward"}
    if unknown:
        raise ValueError(f"unknown bend keys {sorted(unknown)}")
    toward = bend.get("bends_toward", "-x")
    if toward not in ("-x", "+x"):
        raise ValueError("bends_toward must be '-x' or '+x'")
    return NominalBend(
        float(bend["radius_m"]), float(bend["angle_rad"]), float(bend["arm_m"]), toward == "-x"
    )


def region_for(
    field: FieldMap,
    shape: str = "circle",
    radius_m: float | None = None,
    z_min_m: float | None = None,
    z_max_m: float | None = None,
    half_height_m: float | None = None,
    bend: dict[str, Any] | None = None,
) -> Region:
    """A region from tool arguments. Without ``radius_m``, the largest circle or square about
    the z axis that the grid covers. ``bend`` names a nominal path: ``radius_m``,
    ``angle_rad``, ``arm_m`` and optionally ``bends_toward`` ("-x", the default, or "+x")."""
    path = None if bend is None else nominal_bend(bend)
    if radius_m is None:
        if path is not None or half_height_m is not None:
            raise ValueError("give radius_m for a rectangle or a region about a bend")
        return Region.covering(field, shape, z_min_m, z_max_m)
    return Region(shape, float(radius_m), z_min_m, z_max_m, half_height_m, path)
