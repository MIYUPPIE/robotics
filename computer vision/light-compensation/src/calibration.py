"""04 Reference environment manager: calibration, persistence, drift.

L_ref = median(L_1..L_n) over the calibration window. The reference is written
atomically to JSON and reloaded on the next start so the system remembers what
"normal" looked like. A reference recorded under different camera geometry is
rejected rather than silently trusted.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import Settings
from .illumination import IlluminationStats
from .linear_light import decode, encode

REFERENCE_VERSION = 1


class CalibrationError(RuntimeError):
    """Calibration cannot produce a trustworthy reference (too dark, too few frames)."""


class ReferenceFileError(ValueError):
    """Stored reference is missing fields, corrupt, or from another camera setup."""


@dataclass(frozen=True)
class Reference:
    level: float
    cells: tuple[float, ...]
    grid_rows: int
    grid_cols: int
    n_frames: int
    spread: float
    created_at: str
    camera_index: int
    width: int
    height: int
    exposure: dict[str, Any] = field(default_factory=dict)
    version: int = REFERENCE_VERSION

    @property
    def cells_array(self) -> np.ndarray:
        return np.asarray(self.cells, dtype=np.float32)


class Calibrator:
    """Collects illumination stats over a window, then finalizes a Reference."""

    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._levels: list[float] = []
        self._cells: list[np.ndarray] = []
        self._t0: float | None = None
        self._t_last: float = 0.0
        self._sum: np.ndarray | None = None
        self._sum_n = 0

    @property
    def n_frames(self) -> int:
        return len(self._levels)

    def progress(self) -> float:
        if self._t0 is None:
            return 0.0
        return min((self._t_last - self._t0) / self._s.calibration_seconds, 1.0)

    def add(self, stats: IlluminationStats, t: float, frame: np.ndarray | None = None) -> None:
        if self._t0 is None:
            self._t0 = t
        self._t_last = t
        self._levels.append(stats.level)
        self._cells.append(stats.cells)
        if frame is not None:
            lin = decode(frame)
            if self._sum is None or self._sum.shape != lin.shape:
                self._sum = lin.astype(np.float32)
                self._sum_n = 1
            else:
                self._sum += lin
                self._sum_n += 1

    def image(self) -> np.ndarray | None:
        """Mean of the calibration frames in linear light, re-encoded: a clean picture of the
        normal room (averaging ~150 frames removes almost all sensor noise)."""
        if self._sum is None:
            return None
        return encode(self._sum / self._sum_n)

    @property
    def ready(self) -> bool:
        return (
            self._t0 is not None
            and self._t_last - self._t0 >= self._s.calibration_seconds
            and self.n_frames >= self._s.min_calibration_frames
        )

    def finalize(self, exposure: dict[str, Any] | None = None) -> Reference:
        if self.n_frames < self._s.min_calibration_frames:
            raise CalibrationError(
                f"only {self.n_frames} frames, need {self._s.min_calibration_frames}"
            )
        levels = np.asarray(self._levels, dtype=np.float64)
        level = float(np.median(levels))
        if level < self._s.min_reference_level:
            raise CalibrationError(
                f"room too dark to calibrate (level {level:.3f} < {self._s.min_reference_level})"
            )
        q1, q3 = np.percentile(levels, [25, 75])
        spread = float((q3 - q1) / level)
        cells = np.median(np.stack(self._cells), axis=0)
        return Reference(
            level=level,
            cells=tuple(float(c) for c in cells),
            grid_rows=self._s.grid_rows,
            grid_cols=self._s.grid_cols,
            n_frames=self.n_frames,
            spread=spread,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            camera_index=self._s.camera_index,
            width=self._s.width,
            height=self._s.height,
            exposure=dict(exposure or {}),
        )


def save_reference(ref: Reference, path: Path | str) -> None:
    """Atomic write: a crash mid-save can never leave a half-written reference."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(ref), indent=2)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=p.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(payload)
        os.replace(tmp, p)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_reference(path: Path | str, settings: Settings) -> Reference | None:
    """Return the stored reference, None if there is no file, or raise ReferenceFileError
    if it exists but cannot be trusted for the current camera/grid configuration."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text())
        if not isinstance(raw, dict):
            raise ReferenceFileError("reference file is not a JSON object")
        if raw.get("version") != REFERENCE_VERSION:
            raise ReferenceFileError(f"unsupported reference version {raw.get('version')!r}")
        raw["cells"] = tuple(float(c) for c in raw["cells"])
        ref = Reference(**raw)
    except ReferenceFileError:
        raise
    except (json.JSONDecodeError, TypeError, KeyError, ValueError) as exc:
        raise ReferenceFileError(f"corrupt reference file {p}: {exc}") from exc

    if not (0.0 < ref.level <= 1.0) or len(ref.cells) != ref.grid_rows * ref.grid_cols:
        raise ReferenceFileError("reference values are inconsistent")
    try:
        age_h = (datetime.now(UTC) - datetime.fromisoformat(ref.created_at)).total_seconds() / 3600
    except ValueError as exc:
        raise ReferenceFileError(f"bad created_at: {ref.created_at!r}") from exc
    if age_h > settings.max_reference_age_h:
        raise ReferenceFileError(
            f"reference is {age_h:.0f} h old (max {settings.max_reference_age_h:g})"
        )
    mismatches = [
        name
        for name, stored, now in (
            ("camera_index", ref.camera_index, settings.camera_index),
            ("width", ref.width, settings.width),
            ("height", ref.height, settings.height),
            ("grid_rows", ref.grid_rows, settings.grid_rows),
            ("grid_cols", ref.grid_cols, settings.grid_cols),
        )
        if stored != now
    ]
    if mismatches:
        raise ReferenceFileError(f"reference was recorded with different {', '.join(mismatches)}")
    return ref


def drift_reference(
    ref: Reference, stats: IlluminationStats, rate_per_s: float, dt: float
) -> Reference:
    """Move the reference a small step toward the current stats (EMA). Callers gate this
    to the NORMAL state inside the drift band, so a real dimming is never absorbed."""
    a = min(max(rate_per_s * dt, 0.0), 1.0)
    if a == 0.0:
        return ref
    cells = tuple(float(c + (n - c) * a) for c, n in zip(ref.cells, stats.cells, strict=True))
    return replace(ref, level=float(ref.level + (stats.level - ref.level) * a), cells=cells)


def reference_image_path(reference_json: Path | str) -> Path:
    return Path(reference_json).with_suffix(".png")


def save_reference_image(img: np.ndarray, reference_json: Path | str) -> Path:
    path = reference_image_path(reference_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.png")
    if not cv2.imwrite(str(tmp), img):
        raise OSError(f"could not write {tmp}")
    os.replace(tmp, path)
    return path


def load_reference_image(reference_json: Path | str, settings: Settings) -> np.ndarray | None:
    """The stored normal-room image, or None if absent. Raises ReferenceFileError if it exists
    but does not match the configured frame size (wrong camera setup)."""
    path = reference_image_path(reference_json)
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ReferenceFileError(f"cannot decode reference image {path}")
    if img.shape[:2] != (settings.height, settings.width):
        raise ReferenceFileError(
            f"reference image is {img.shape[1]}x{img.shape[0]}, camera is "
            f"{settings.width}x{settings.height}"
        )
    return img
