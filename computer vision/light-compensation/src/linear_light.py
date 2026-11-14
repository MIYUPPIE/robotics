"""Linear-light helpers. Light adds and scales in linear intensity, not in the gamma-encoded
8-bit values a webcam delivers, so relighting (multiplying by a gain field) must happen
here. Decode/encode use a 2.2 power curve (close to sRGB, and exact for the synthetic eval)."""

from __future__ import annotations

import cv2
import numpy as np

GAMMA = 2.2
_DECODE = ((np.arange(256, dtype=np.float32) / 255.0) ** GAMMA).astype(np.float32)
_DECODE_CV = np.ascontiguousarray(_DECODE.reshape(256, 1))


def decode(img_u8: np.ndarray) -> np.ndarray:
    """uint8 encoded -> float32 linear 0..1."""
    return np.asarray(cv2.LUT(img_u8, _DECODE_CV), dtype=np.float32)


def encode(lin: np.ndarray) -> np.ndarray:
    """float linear -> uint8 encoded, exact (used for calibration, not per-frame). Values
    outside 0..1 saturate."""
    enc = 255.0 * np.power(np.clip(lin, 0.0, 1.0), 1.0 / GAMMA)
    return np.clip(enc + 0.5, 0, 255).astype(np.uint8)


# sqrt-domain index -> encoded value. Index = round(255 * sqrt(x)); finer than 8 bits in the
# shadows (where relit footage lives) and within half an output level elsewhere.
_SQRT_TO_ENCODED = np.ascontiguousarray(
    np.round(255.0 * (np.arange(256, dtype=np.float32) / 255.0) ** (2.0 / GAMMA)).astype(np.uint8)
).reshape(256, 1)


def encode_fast(lin: np.ndarray) -> np.ndarray:
    """Like `encode` but ~2x faster, using OpenCV SIMD sqrt + an 8-bit LUT. Requires lin >= 0
    (values above 1 saturate)."""
    idx = cv2.convertScaleAbs(cv2.sqrt(lin), alpha=255.0)
    return cv2.LUT(idx, _SQRT_TO_ENCODED)
