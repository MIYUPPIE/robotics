"""01 Acquisition layer: webcam capture and exposure locking (V4L2 via OpenCV).

Why lock exposure: a webcam's own auto-exposure brightens a dark scene by itself,
which would hide the very illumination change we are trying to measure. We let
auto-exposure converge in the normal room, read the converged exposure, then
switch to manual at that value. The locked values are stored in the reference so
the next start re-applies them. No `v4l2-ctl` or sudo is needed; OpenCV's V4L2
backend exposes CAP_PROP_AUTO_EXPOSURE / EXPOSURE / GAIN directly.

The camera is restored to its original auto-exposure mode on release().
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import cv2
import numpy as np

from .config import Settings

V4L2_MANUAL = 1.0
V4L2_AUTO = 3.0
_SETTLE_FRAMES = 5


class CameraError(RuntimeError):
    pass


class Capture(Protocol):
    def isOpened(self) -> bool: ...  # noqa: N802 (OpenCV naming)
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def get(self, prop: int) -> float: ...
    def set(self, prop: int, value: float) -> bool: ...
    def release(self) -> None: ...


@dataclass(frozen=True)
class ExposureState:
    locked: bool
    mode_before: float
    exposure: float
    gain: float | None
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Camera:
    def __init__(
        self,
        settings: Settings,
        capture_factory: Callable[[int], Capture] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._s = settings
        self._factory = capture_factory or (lambda idx: cv2.VideoCapture(idx))
        self._sleep = sleep
        self._cap: Capture | None = None
        self._restore_mode: float | None = None

    def open(self) -> None:
        cap = self._factory(self._s.camera_index)
        if not cap.isOpened():
            cap.release()
            raise CameraError(f"cannot open camera index {self._s.camera_index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._s.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._s.height)
        cap.set(cv2.CAP_PROP_FPS, self._s.fps)
        self._cap = cap

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            raise CameraError("camera not open")
        ok, frame = self._cap.read()
        return frame if ok and frame is not None else None

    def actual_size(self) -> tuple[int, int]:
        if self._cap is None:
            raise CameraError("camera not open")
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

    def lock_exposure(self, saved: dict[str, Any] | None = None) -> ExposureState:
        """Lock exposure/gain. With `saved` (from a stored reference) re-apply those values;
        otherwise let auto-exposure converge first and lock what it settled on."""
        cap = self._require()
        if not self._s.lock_exposure:
            return ExposureState(False, cap.get(cv2.CAP_PROP_AUTO_EXPOSURE), -1.0, None, "disabled")

        mode_before = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        if saved and saved.get("locked") and float(saved.get("exposure", -1)) > 0:
            exposure = float(saved["exposure"])
            gain = saved.get("gain")
        else:
            self._warmup()
            exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
            g = cap.get(cv2.CAP_PROP_GAIN)
            gain = g if g >= 0 else None
        if exposure <= 0:
            return ExposureState(False, mode_before, exposure, gain, "driver reports no exposure")

        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, V4L2_MANUAL)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        if gain is not None:
            cap.set(cv2.CAP_PROP_GAIN, gain)
        for _ in range(_SETTLE_FRAMES):
            cap.read()

        if cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) != V4L2_MANUAL:
            return ExposureState(False, mode_before, exposure, gain, "camera refused manual mode")
        self._restore_mode = mode_before
        note = "locked" if gain is not None else "locked (gain control not supported)"
        return ExposureState(True, mode_before, exposure, gain, note)

    def _warmup(self) -> None:
        cap = self._require()
        deadline = time.monotonic() + self._s.exposure_warmup_s
        while time.monotonic() < deadline:
            ok, _ = cap.read()
            if not ok:
                self._sleep(0.01)

    def _require(self) -> Capture:
        if self._cap is None:
            raise CameraError("camera not open")
        return self._cap

    def release(self) -> None:
        if self._cap is None:
            return
        if self._restore_mode is not None:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, self._restore_mode)
            self._restore_mode = None
        self._cap.release()
        self._cap = None
