"""02 Image preprocessing: BGR -> LAB, luminance-plane extraction, downscale.

Brightness is measured on the CIELAB L channel and the enhancer corrects that
same channel, so the estimate and the correction agree by construction.
"""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np


class Preprocessed(NamedTuple):
    lab: np.ndarray  # full-resolution uint8 LAB (OpenCV scaling: L in 0..255)
    luma_small: np.ndarray  # float32 0..1, downscaled L plane used for statistics


def validate_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError("frame must be a uint8 numpy array")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must be HxWx3 BGR, got shape {frame.shape}")
    if frame.shape[0] < 2 or frame.shape[1] < 2:
        raise ValueError(f"frame too small: {frame.shape}")


def downscale_luma(luma_u8: np.ndarray, analysis_width: int) -> np.ndarray:
    """uint8 L plane -> float32 0..1 plane, area-downscaled to `analysis_width` columns."""
    h, w = luma_u8.shape
    if w > analysis_width:
        new_h = max(2, round(h * analysis_width / w))
        luma_u8 = cv2.resize(luma_u8, (analysis_width, new_h), interpolation=cv2.INTER_AREA)
    return luma_u8.astype(np.float32) / 255.0


def preprocess(frame_bgr: np.ndarray, analysis_width: int) -> Preprocessed:
    validate_frame(frame_bgr)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    return Preprocessed(lab=lab, luma_small=downscale_luma(lab[:, :, 0], analysis_width))
