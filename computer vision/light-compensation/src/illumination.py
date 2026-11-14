"""03 Environmental illumination estimator.

The scalar used everywhere (`level`) is the median of the grid-cell medians.
A person or object occluding a few cells cannot move it, which keeps a walk-in
from looking like a light switch. `dropped_cell_fraction` separates a global
light change (almost every cell darkens) from a local occlusion (a few do).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IlluminationStats:
    mean: float
    median: float
    p10: float
    p90: float
    level: float  # median of cell medians: the robust scalar
    cells: np.ndarray  # float32, row-major, one median per grid cell


def _bounds(n: int, parts: int) -> list[int]:
    return [round(i * n / parts) for i in range(parts + 1)]


def estimate(luma: np.ndarray, rows: int, cols: int) -> IlluminationStats:
    """Statistics of a float 0..1 luma plane. Requires at least rows x cols pixels."""
    h, w = luma.shape
    if h < rows or w < cols:
        raise ValueError(f"plane {luma.shape} smaller than grid {rows}x{cols}")
    rb, cb = _bounds(h, rows), _bounds(w, cols)
    cells = np.empty(rows * cols, dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            cells[r * cols + c] = np.median(luma[rb[r] : rb[r + 1], cb[c] : cb[c + 1]])
    p10, med, p90 = np.percentile(luma, [10, 50, 90])
    return IlluminationStats(
        mean=float(luma.mean()),
        median=float(med),
        p10=float(p10),
        p90=float(p90),
        level=float(np.median(cells)),
        cells=cells,
    )


def dropped_cell_fraction(
    cells: np.ndarray, ref_cells: np.ndarray, drop_ratio: float, floor: float = 0.02
) -> float:
    """Fraction of cells whose brightness fell below `drop_ratio` x their reference."""
    if cells.shape != ref_cells.shape:
        raise ValueError(f"cell grid mismatch: {cells.shape} vs {ref_cells.shape}")
    valid = ref_cells >= floor  # a near-black reference cell has nothing left to lose
    if not valid.any():
        return 0.0
    ratio = cells[valid] / ref_cells[valid]
    return float(np.mean(ratio < drop_ratio))


class LevelFilter:
    """Asymmetric smoother: a rising level passes through instantly, a falling level is
    low-passed. Rising fast lets the controller release gain the moment lights return
    (no over-bright flash); falling slowly keeps transient shadows from triggering."""

    def __init__(self, fall_tau_s: float) -> None:
        self._tau = fall_tau_s
        self._value: float | None = None
        self._t: float | None = None

    @property
    def value(self) -> float | None:
        return self._value

    def reset(self) -> None:
        self._value = None
        self._t = None

    def update(self, level: float, t: float) -> float:
        if self._value is None or self._t is None:
            self._value, self._t = level, t
            return level
        dt = max(t - self._t, 0.0)
        self._t = t
        if level >= self._value or self._tau <= 0:
            self._value = level
        else:
            a = 1.0 - math.exp(-dt / self._tau)
            self._value += (level - self._value) * a
        return self._value
