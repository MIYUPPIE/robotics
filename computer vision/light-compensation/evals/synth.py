"""Deterministic synthetic scenes and a physically-motivated darkening model.

Darkening happens in linear light (sRGB-ish decode), then sensor noise (read +
shot) is added, then the result is re-encoded and quantised to 8 bit, which is
what a real webcam does when the room lights drop with exposure locked.
"""

from __future__ import annotations

import cv2
import numpy as np

H, W = 240, 320
_DECODE = ((np.arange(256, dtype=np.float32) / 255.0) ** 2.2).astype(np.float32)
_ENCODE = np.round(255 * (np.arange(4096, dtype=np.float32) / 4095.0) ** (1 / 2.2)).astype(np.uint8)


def make_scene(seed: int = 7, h: int = H, w: int = W) -> np.ndarray:
    """Textured BGR scene (gradient wall, coloured blocks, fine texture), mean L* ~ 0.5."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 110 + 50 * (xx / w) + 30 * (yy / h)
    img = np.stack([base * 0.9, base, base * 1.05], axis=-1)
    for _ in range(14):
        x0, y0 = int(rng.integers(0, w - 40)), int(rng.integers(0, h - 30))
        bw, bh = int(rng.integers(25, 80)), int(rng.integers(20, 60))
        color = rng.integers(40, 230, size=3).astype(np.float32)
        img[y0 : y0 + bh, x0 : x0 + bw] = color
    img += rng.normal(0, 4, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def darken(
    scene: np.ndarray,
    k: float,
    rng: np.random.Generator,
    read_noise: float = 0.003,
    shot: float = 0.004,
) -> np.ndarray:
    """Scene as seen with the light scaled by k (0..1] in linear light, plus sensor noise."""
    lin = _DECODE[scene] * k
    noise = rng.standard_normal(lin.shape, dtype=np.float32)
    lin = lin + noise * (read_noise + np.sqrt(np.maximum(lin, 0) * shot))
    return _ENCODE[np.clip(lin * 4095.0 + 0.5, 0, 4095).astype(np.int16)]


def add_occluder(
    frame: np.ndarray, x: int, width_frac: float = 0.45, value: int = 25
) -> np.ndarray:
    """A dark vertical band (a person) covering `width_frac` of the width, full height."""
    out = frame.copy()
    h, w = out.shape[:2]
    bw = int(w * width_frac)
    out[:, max(x, 0) : min(x + bw, w)] = value
    return out


def to_level(frame: np.ndarray) -> float:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    return float(np.median(lab[:, :, 0]) / 255.0)
