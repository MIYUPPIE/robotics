"""07 Image enhancement engine: denoise -> adaptive gamma -> CLAHE, on LAB L only.

Colour preservation: the a/b chroma channels are never brightened, only (lightly)
blurred when the gain is high, so hue is kept while luminance is restored.

Gamma is solved so the frame's median luma maps to gain x level:
    out = in ** gamma,  gamma = ln(min(G * L, ceiling)) / ln(L),  gamma in [floor, 1]
A pointwise monotonic map preserves the median ordering, so the median lands on
the target (before CLAHE, which redistributes local contrast and can shift the
median a lot on flat regions). A single closed-loop trim gamma after CLAHE pulls
the output level back onto the target, so brightness retention never depends on
what CLAHE happened to do to this particular frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import Settings
from .illumination import estimate
from .preprocessing import Preprocessed, downscale_luma

_IDENTITY_GAIN = 1.0 + 1e-3


@dataclass(frozen=True)
class EnhanceResult:
    frame: np.ndarray
    applied: bool
    gamma: float
    clahe_clip: float
    trim_gamma: float = 1.0


def solve_gamma(gain: float, level: float, floor: float, ceiling: float) -> float:
    level = min(max(level, 0.01), 0.97)
    target = min(gain * level, ceiling)
    if target <= level:
        return 1.0
    return min(max(math.log(target) / math.log(level), floor), 1.0)


def gamma_lut(gamma: float) -> np.ndarray:
    x = np.arange(256, dtype=np.float32) / 255.0
    return np.clip(np.round(255.0 * x**gamma), 0, 255).astype(np.uint8)


class Enhancer:
    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._clahe: dict[float, cv2.CLAHE] = {}

    def _get_clahe(self, clip: float) -> cv2.CLAHE:
        key = round(clip, 1)
        if key not in self._clahe:
            tile = (self._s.clahe_tile, self._s.clahe_tile)
            self._clahe[key] = cv2.createCLAHE(clipLimit=key, tileGridSize=tile)
        return self._clahe[key]

    def clahe_clip_for(self, gain: float) -> float:
        s = self._s
        frac = math.log(max(gain, 1.0)) / math.log(s.max_gain)
        frac = min(max(frac, 0.0), 1.0)
        return s.clahe_clip_min + (s.clahe_clip_max - s.clahe_clip_min) * frac

    def enhance(
        self, frame_bgr: np.ndarray, pre: Preprocessed, gain: float, level: float
    ) -> EnhanceResult:
        """Return the compensated frame. At gain ~1 the input frame is returned untouched."""
        s = self._s
        if gain <= _IDENTITY_GAIN:
            return EnhanceResult(frame_bgr, False, 1.0, 0.0)

        target = min(gain * min(max(level, 0.01), 0.97), s.target_ceiling)

        lum = pre.lab[:, :, 0]
        a, b = pre.lab[:, :, 1], pre.lab[:, :, 2]
        if gain >= s.denoise_gain_min:
            lum = cv2.bilateralFilter(lum, 5, s.denoise_sigma_color * gain, s.denoise_sigma_space)
        if gain >= s.chroma_blur_gain_min:
            a = cv2.GaussianBlur(a, (5, 5), 0)
            b = cv2.GaussianBlur(b, (5, 5), 0)

        gamma = solve_gamma(gain, level, s.gamma_floor, s.target_ceiling)
        lum = cv2.LUT(lum, gamma_lut(gamma))
        clip = self.clahe_clip_for(gain)
        trim = 1.0
        if clip > 1.0:
            lum = self._get_clahe(clip).apply(lum)
            lum, trim = self._trim(lum, target)

        out = cv2.cvtColor(cv2.merge([lum, a, b]), cv2.COLOR_LAB2BGR)
        return EnhanceResult(out, True, gamma, clip, trim)

    def _trim(self, lum: np.ndarray, target: float) -> tuple[np.ndarray, float]:
        """One corrective gamma so the post-CLAHE grid level equals `target`."""
        s = self._s
        got = estimate(downscale_luma(lum, s.analysis_width), s.grid_rows, s.grid_cols).level
        got = min(max(got, 0.01), 0.97)
        if abs(got - target) / target <= s.trim_tolerance:
            return lum, 1.0
        g = min(max(math.log(target) / math.log(got), s.trim_gamma_min), s.trim_gamma_max)
        return cv2.LUT(lum, gamma_lut(g)), g


def output_level_of(frame_bgr: np.ndarray, analysis_width: int) -> np.ndarray:
    """Luma plane (float 0..1, downscaled) of an already-enhanced frame, for the error metric."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    return downscale_luma(lab[:, :, 0], analysis_width)
