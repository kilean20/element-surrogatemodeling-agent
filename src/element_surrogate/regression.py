"""Pooled regression metrics with units; batch RMSEs/quantiles are never averaged."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RegressionMetrics:
    """Exact empirical tails for bounded benchmark cohorts; stores errors locally.

    Large campaigns should stream errors to a project-local array or explicitly
    use an approximate quantile algorithm, never average batch quantiles.
    """

    coordinate_units: tuple[str, ...]
    _errors: list = field(default_factory=list)

    def update(self, prediction, truth):
        a, b = np.asarray(prediction, dtype=float), np.asarray(truth, dtype=float)
        if (
            a.shape != b.shape
            or a.ndim != 2
            or a.shape[1] != len(self.coordinate_units)
            or not np.isfinite(a).all()
            or not np.isfinite(b).all()
        ):
            raise ValueError("matching finite predictions/labels and declared units required")
        self._errors.append(a - b)

    def summary(self, *, quantile_method="linear"):
        errors = (
            np.concatenate(self._errors)
            if self._errors
            else np.empty((0, len(self.coordinate_units)))
        )
        if not len(errors):
            raise ValueError("no evaluated rows")
        return {
            "rows": len(errors),
            "coordinate_units": list(self.coordinate_units),
            "rmse": np.sqrt(np.mean(errors**2, axis=0)).tolist(),
            "mae": np.mean(np.abs(errors), axis=0).tolist(),
            "p99_absolute": np.quantile(
                np.abs(errors), 0.99, axis=0, method=quantile_method
            ).tolist(),
            "max_absolute": np.max(np.abs(errors), axis=0).tolist(),
            "quantile_method": quantile_method,
        }
