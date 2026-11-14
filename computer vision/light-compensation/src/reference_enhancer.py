"""Reference-guided relighting: make the live frame look like the stored normal-light scene.

The camera is fixed, so calibration stores a clean mean image of the normal room. For each
live frame we estimate, per colour channel, a smooth gain field M(x, c) such that
M * live ~= reference in the static background, then multiply the live frame by M in LINEAR
light. This restores global brightness, uneven lighting (a lamp that went off on one side) and
colour cast (a warm lamp replaced by a cool one), while keeping all live detail, because only
the low-frequency illumination ratio is changed.

Robustness: the field is estimated on a coarse grid. A person or moved object makes a few cells
disagree with their neighbours; those outlier cells are replaced by the neighbourhood median, and
every cell is clamped near the global ratio, so occluders do not distort the correction.

Library choice: OpenCV (resize/medianBlur/bilateralFilter) and NumPy only. No new dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import Settings
from .linear_light import decode, encode_fast

_CUR_FLOOR = 2e-5  # linear; below this the sensor is just noise
_REF_FLOOR = 1e-4  # a near-black reference cell carries no usable ratio
_TEMPORAL_FULL_GAIN = 3.0
_SPIKE_GAIN = 4.0  # median pre-filter kicks in at this linear gain


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


@dataclass(frozen=True)
class FieldEstimate:
    log_field: np.ndarray  # (rows, cols, 3) float32, natural log of the gain
    deviation: float  # robust magnitude of the required correction (0 = same as reference)


class FieldEstimator:
    def __init__(self, settings: Settings, reference_bgr: np.ndarray) -> None:
        self._s = settings
        self._size = (settings.field_cols, settings.field_rows)
        self._ref_cells = self._cells(reference_bgr)
        self._ref_valid = self._ref_cells >= _REF_FLOOR
        gy, gx = np.mgrid[
            -1 : 1 : complex(settings.field_rows), -1 : 1 : complex(settings.field_cols)
        ]
        gx, gy = gx.ravel(), gy.ravel()
        self._design = np.stack([np.ones_like(gx), gx, gy, gx * gx, gx * gy, gy * gy], axis=1)

    def _cells(self, frame_bgr: np.ndarray) -> np.ndarray:
        s = self._s
        h, w = frame_bgr.shape[:2]
        small = frame_bgr
        if w > s.analysis_width:
            small = cv2.resize(
                frame_bgr,
                (s.analysis_width, max(2, round(h * s.analysis_width / w))),
                interpolation=cv2.INTER_AREA,
            )
        return cv2.resize(decode(small), self._size, interpolation=cv2.INTER_AREA)

    def estimate(self, frame_bgr: np.ndarray) -> FieldEstimate:
        s = self._s
        lo, hi = math.log(s.field_min_gain), math.log(s.field_max_gain)
        cur = np.maximum(self._cells(frame_bgr), _CUR_FLOOR)
        ref = np.maximum(self._ref_cells, _CUR_FLOOR)
        raw = np.clip(np.log(ref / cur), lo, hi).astype(np.float32)  # (rows, cols, 3)

        flat = raw.reshape(-1, 3)
        valid = self._ref_valid.reshape(-1, 3)
        glob = np.array(
            [np.median(flat[valid[:, c], c]) if valid[:, c].any() else 0.0 for c in range(3)]
        )
        fit = np.empty_like(flat)
        for c in range(3):
            fit[:, c] = self._robust_fit(flat[:, c], valid[:, c].astype(np.float64), glob[c])
        fit = np.clip(fit, glob - s.field_dev_max, glob + s.field_dev_max)
        # Structure the quadratic cannot capture (a spot lamp) survives if it spans neighbouring
        # cells (3x3 median of the residual drops isolated noisy cells); a residual larger than
        # the outlier threshold is an occluder and takes the fit.
        resid = cv2.medianBlur((flat - fit).reshape(raw.shape).astype(np.float32), 3)
        resid = resid.reshape(-1, 3)
        resid = np.where((np.abs(resid) > s.field_outlier_thr) | ~valid, 0.0, resid)
        field = (fit + resid).reshape(raw.shape)
        field = np.clip(field, lo, hi).astype(np.float32)

        deviation = float(np.median(np.abs(field).mean(axis=2)))
        return FieldEstimate(field, deviation)

    def _robust_fit(self, y: np.ndarray, w0: np.ndarray, fallback: float) -> np.ndarray:
        """Quadratic surface through the cell values by iteratively reweighted least squares
        (Tukey biweight): a person covering under half the cells is rejected as an outlier,
        a smooth lighting gradient is captured."""
        x = self._design
        # Start from the robust constant (median) so a large occluder cannot drag the first fit.
        r0 = y - fallback
        scale0 = max(1.4826 * float(np.median(np.abs(r0[w0 > 0]))), 0.05)
        u0 = np.clip(np.abs(r0) / (4.685 * scale0), 0.0, 1.0)
        w = w0 * (1.0 - u0 * u0) ** 2
        beta = None
        for _ in range(4):
            if (w > 1e-6).sum() < 8:
                break
            sw = np.sqrt(w)
            beta, *_ = np.linalg.lstsq(x * sw[:, None], y * sw, rcond=None)
            r = y - x @ beta
            scale = max(1.4826 * float(np.median(np.abs(r[w0 > 0]))), 0.02)
            u = np.clip(np.abs(r) / (4.685 * scale), 0.0, 1.0)
            w = w0 * (1.0 - u * u) ** 2
        if beta is None:
            return np.full_like(y, fallback)
        return x @ beta


class FieldController:
    """Gates and smooths the log-field in time.

    Gate: no correction while the scene is within `field_gate_lo` of the reference, full
    correction beyond `field_gate_hi`, smooth in between (no visible switch-on step).
    Smoothing: rising correction eases in over `field_tau_up_s`; falling correction (light came
    back) is applied at once, so the output never flashes over-bright.
    """

    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((self._s.field_rows, self._s.field_cols, 3), np.float32)
        self._t: float | None = None

    def gate(self, deviation: float) -> float:
        s = self._s
        return smoothstep((deviation - s.field_gate_lo) / (s.field_gate_hi - s.field_gate_lo))

    def update(self, est: FieldEstimate, t: float) -> np.ndarray:
        s = self._s
        target = (est.log_field * self.gate(est.deviation)).astype(np.float32)
        dt = 0.0 if self._t is None else max(t - self._t, 0.0)
        self._t = t
        a = 1.0 if s.field_tau_up_s <= 0 else 1.0 - math.exp(-dt / s.field_tau_up_s)
        rising = target > self._state
        self._state = np.where(rising, self._state + (target - self._state) * a, target)
        return self._state

    @property
    def state(self) -> np.ndarray:
        return self._state


@dataclass(frozen=True)
class RelightResult:
    frame: np.ndarray
    applied: bool
    gain: float  # geometric-mean linear gain of the applied field


class ReferenceEnhancer:
    def __init__(self, settings: Settings, reference_bgr: np.ndarray) -> None:
        self._s = settings
        self.estimator = FieldEstimator(settings, reference_bgr)
        self.controller = FieldController(settings)
        self._ref_lin = decode(reference_bgr)
        self._prev: np.ndarray | None = None
        self._prev_raw: np.ndarray | None = None
        self.noise_levels = settings.noise_sigma_init  # sensor noise, in 8-bit levels

    def reset(self) -> None:
        self.controller.reset()
        self._prev = None
        self._prev_raw = None

    def _update_noise(self, frame_bgr: np.ndarray) -> None:
        """Track sensor noise from frame-to-frame differences (median is robust to motion)."""
        prev = self._prev_raw
        self._prev_raw = frame_bgr
        if prev is None or prev.shape != frame_bgr.shape:
            return
        d = np.abs(frame_bgr[::4, ::4].astype(np.int16) - prev[::4, ::4].astype(np.int16))
        sigma = 1.4826 * float(np.median(d)) / math.sqrt(2.0)
        sigma = min(max(sigma, 0.5), 12.0)
        self.noise_levels += (sigma - self.noise_levels) * 0.05

    def process(self, frame_bgr: np.ndarray, t: float) -> tuple[RelightResult, FieldEstimate]:
        self._update_noise(frame_bgr)
        est = self.estimator.estimate(frame_bgr)
        field = self.controller.update(est, t)
        return self.apply(frame_bgr, field), est

    def apply(self, frame_bgr: np.ndarray, log_field: np.ndarray) -> RelightResult:
        s = self._s
        if float(np.abs(log_field).max()) < 1e-3:
            self._prev = None
            return RelightResult(frame_bgr, False, 1.0)
        gain = math.exp(float(log_field.mean()))
        h, w = frame_bgr.shape[:2]

        boost = max(gain, 1.0)
        work = frame_bgr
        if boost >= _SPIKE_GAIN:
            work = cv2.medianBlur(work, 3)  # single-pixel spikes would be amplified into specks
        if boost >= s.denoise_gain_min:
            sigma = min(12.0 + 6.0 * math.log2(boost), 60.0)
            work = cv2.bilateralFilter(work, 5, sigma, s.denoise_sigma_space)
        work = self._temporal(work, boost)

        m_full = cv2.resize(
            np.exp(log_field).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
        )
        relit = cv2.multiply(decode(work), m_full)
        relit = self._shrink_to_reference(relit, work, m_full)
        return RelightResult(encode_fast(relit), True, gain)

    def _shrink_to_reference(
        self, relit: np.ndarray, work: np.ndarray, m_full: np.ndarray
    ) -> np.ndarray:
        """Background-model denoising. Where the relit pixel differs from the stored normal
        scene by less than the amplified sensor noise (kappa x sigma), the difference cannot be
        told from noise, so the stored value is used; larger differences (people, moved objects,
        real changes) pass through, reduced by the threshold (soft shrinkage).
        Soft shrink identity: ref + soft(d, tau) == relit - clip(relit - ref, -tau, tau)."""
        kappa = self._s.background_shrink_kappa
        if kappa <= 0:
            return relit
        # sigma_lin(x) = 2.2 * x^1.2 * sigma_levels / 255 for encoded x in 0..1 (delta method)
        xs = np.arange(256, dtype=np.float32) / 255.0
        sig = (2.2 * xs**1.2 * (self.noise_levels / 255.0)).astype(np.float32)
        sig = np.maximum(sig, 1e-5) * np.float32(kappa)
        tau = cv2.multiply(np.asarray(cv2.LUT(work, sig.reshape(256, 1)), dtype=np.float32), m_full)
        d = cv2.subtract(relit, self._ref_lin)
        d = cv2.max(cv2.min(d, tau), -tau)
        return cv2.subtract(relit, d)

    def _temporal(self, frame: np.ndarray, boost: float) -> np.ndarray:
        """Recursive temporal denoise, weight rising with the gain; moving pixels bypass it."""
        s = self._s
        cur = frame.astype(np.float32)
        prev = self._prev
        wgt = s.temporal_max_weight * min(
            math.log(max(boost, 1.0)) / math.log(_TEMPORAL_FULL_GAIN), 1.0
        )
        if prev is None or prev.shape != cur.shape or wgt <= 0:
            self._prev = cur
            return frame
        diff = cv2.absdiff(cur, prev)
        moving = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) > s.temporal_motion_threshold / 2.0
        out = cv2.addWeighted(prev, wgt, cur, 1.0 - wgt, 0.0)
        if moving.any():
            np.copyto(out, cur, where=moving[:, :, None])
        self._prev = out
        return cv2.convertScaleAbs(out)
