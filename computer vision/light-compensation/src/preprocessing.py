"""02 Image preprocessing: BGR -> LAB, luminance-plane extraction, downscale.

Brightness is measured on the CIELAB L channel. The statistics plane is built small-first
(area-downscale the BGR frame, then convert), which is ~10x cheaper than converting the full
frame and is used identically for calibration, detection and the output-error metric. The
full-resolution LAB image is computed lazily, only by the gamma enhancer that needs it.
"""

from __future__ import annotations

import cv2
import numpy as np


def validate_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError("frame must be a uint8 numpy array")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must be HxWx3 BGR, got shape {frame.shape}")
    if frame.shape[0] < 2 or frame.shape[1] < 2:
        raise ValueError(f"frame too small: {frame.shape}")


def downscale_bgr(frame_bgr: np.ndarray, analysis_width: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if w <= analysis_width:
        return frame_bgr
    new_h = max(2, round(h * analysis_width / w))
    return cv2.resize(frame_bgr, (analysis_width, new_h), interpolation=cv2.INTER_AREA)


def luma_small_of(frame_bgr: np.ndarray, analysis_width: int) -> np.ndarray:
    """float32 0..1 L-channel plane, area-downscaled to `analysis_width` columns."""
    small = downscale_bgr(frame_bgr, analysis_width)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    return lab[:, :, 0].astype(np.float32) / 255.0


def downscale_luma(luma_u8: np.ndarray, analysis_width: int) -> np.ndarray:
    """uint8 L plane -> float32 0..1 plane, area-downscaled to `analysis_width` columns."""
    h, w = luma_u8.shape
    if w > analysis_width:
        new_h = max(2, round(h * analysis_width / w))
        luma_u8 = cv2.resize(luma_u8, (analysis_width, new_h), interpolation=cv2.INTER_AREA)
    return luma_u8.astype(np.float32) / 255.0


class Preprocessed:
    def __init__(self, frame_bgr: np.ndarray, analysis_width: int) -> None:
        self._frame = frame_bgr
        self.luma_small = luma_small_of(frame_bgr, analysis_width)
        self._lab: np.ndarray | None = None

    @property
    def lab(self) -> np.ndarray:
        """Full-resolution uint8 LAB (OpenCV scaling: L in 0..255). Computed on first use."""
        if self._lab is None:
            self._lab = cv2.cvtColor(self._frame, cv2.COLOR_BGR2LAB)
        return self._lab


def preprocess(frame_bgr: np.ndarray, analysis_width: int) -> Preprocessed:
    validate_frame(frame_bgr)
    return Preprocessed(frame_bgr, analysis_width)
