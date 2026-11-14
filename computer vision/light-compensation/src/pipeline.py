"""Pure frame-in / frame-out pipeline gluing modules 02-07 together.

No camera, no disk, no clock: the caller supplies frames and timestamps, so the
whole system is deterministic and testable. `main.py` is the thin shell around it.

Measurement is always taken on the RAW frame, never the enhanced one, so the
controller has no feedback loop through its own output.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from .calibration import (
    CalibrationError,
    Calibrator,
    Reference,
    drift_reference,
)
from .config import Settings
from .controller import CompensationController
from .enhancer import Enhancer, EnhanceResult, output_level_of
from .illumination import IlluminationStats, LevelFilter, dropped_cell_fraction, estimate
from .light_detector import LightDetector, LightState, Transition
from .preprocessing import preprocess
from .reference_enhancer import ReferenceEnhancer


class Phase(StrEnum):
    CALIBRATING = "CALIBRATING"
    RUNNING = "RUNNING"


@dataclass(frozen=True)
class FrameResult:
    frame: np.ndarray
    phase: Phase
    state: LightState
    level: float  # instantaneous robust level of the raw frame
    level_filtered: float
    ref_level: float  # 0.0 while calibrating
    ratio: float  # level_filtered / ref_level (1.0 while calibrating)
    gain: float
    gamma: float
    clahe_clip: float
    dropped_fraction: float
    output_level: float  # robust level of the enhanced frame (NaN if not measured)
    brightness_error_pct: float  # |out - ref| / ref * 100 (NaN while calibrating)
    latency_ms: float
    transition: Transition | None = None
    calibrated_now: bool = False
    calibration_progress: float = 0.0
    message: str = ""


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        reference: Reference | None = None,
        reference_image: np.ndarray | None = None,
    ) -> None:
        self._s = settings
        self._enhancer = Enhancer(settings)
        self._detector = LightDetector(settings)
        self._controller = CompensationController(settings)
        self._filter = LevelFilter(settings.level_fall_tau_s)
        self._calibrator = Calibrator(settings)
        self._reference: Reference | None = reference
        self._ref_image: np.ndarray | None = reference_image
        self._relighter: ReferenceEnhancer | None = None
        self._build_relighter()
        self._exposure: dict[str, Any] = {}
        self._last_t: float | None = None
        self._frames_run = 0
        self._last_measure = (float("nan"), float("nan"))
        self._t_first: float | None = None
        self._mismatch_warned = reference is None  # only check stored references

    @property
    def reference(self) -> Reference | None:
        return self._reference

    @property
    def reference_image(self) -> np.ndarray | None:
        return self._ref_image

    @property
    def mode(self) -> str:
        return "reference" if self._relighter is not None else "gamma"

    def _build_relighter(self) -> None:
        img = self._ref_image
        if self._s.enhance_mode == "reference" and img is not None:
            self._relighter = ReferenceEnhancer(self._s, img)
        else:
            self._relighter = None

    @property
    def phase(self) -> Phase:
        return Phase.RUNNING if self._reference is not None else Phase.CALIBRATING

    def set_exposure_info(self, exposure: dict[str, Any]) -> None:
        """Stored into the reference at the next calibration."""
        self._exposure = dict(exposure)

    def recalibrate(self) -> None:
        self._reference = None
        self._ref_image = None
        self._relighter = None
        self._calibrator = Calibrator(self._s)
        self._detector.reset()
        self._controller.reset()
        self._filter.reset()
        self._last_t = None
        self._frames_run = 0
        self._last_measure = (float("nan"), float("nan"))
        self._t_first = None
        self._mismatch_warned = True  # fresh calibration: nothing stale to warn about

    def process(self, frame_bgr: np.ndarray, t: float) -> FrameResult:
        t_start = time.perf_counter()
        s = self._s
        pre = preprocess(frame_bgr, s.analysis_width)
        stats = estimate(pre.luma_small, s.grid_rows, s.grid_cols)
        dt = 0.0 if self._last_t is None else max(t - self._last_t, 0.0)
        self._last_t = t

        if self._reference is None:
            return self._calibrating(frame_bgr, stats, t, t_start)

        ref = self._reference
        level_f = self._filter.update(stats.level, t)
        ratio = level_f / ref.level
        dropped = dropped_cell_fraction(stats.cells, ref.cells_array, s.cell_drop_ratio)
        uniform = dropped >= s.uniform_cell_fraction
        transition = self._detector.update(ratio, uniform, t)
        state = self._detector.state

        if s.drift_rate_per_s > 0 and state is LightState.NORMAL and abs(ratio - 1) <= s.drift_band:
            self._reference = ref = drift_reference(ref, stats, s.drift_rate_per_s, dt)

        message = self._check_startup(state, t)
        ctl = self._controller.update(level_f, ref.level, state, t)
        if self._relighter is not None:
            rr, _ = self._relighter.process(frame_bgr, t)
            result = EnhanceResult(rr.frame, rr.applied, 1.0, 0.0)
            gain = rr.gain
        else:
            result = self._enhancer.enhance(frame_bgr, pre, ctl.gain, stats.level)
            gain = ctl.gain

        out_level, err = self._last_measure
        if s.measure_output and self._frames_run % s.measure_every == 0:
            out_stats = estimate(
                output_level_of(result.frame, s.analysis_width), s.grid_rows, s.grid_cols
            )
            out_level = out_stats.level
            err = abs(out_level - ref.level) / ref.level * 100.0
            self._last_measure = (out_level, err)
        self._frames_run += 1

        return FrameResult(
            frame=result.frame,
            phase=Phase.RUNNING,
            state=state,
            level=stats.level,
            level_filtered=level_f,
            ref_level=ref.level,
            ratio=ctl.ratio,
            gain=gain,
            gamma=result.gamma,
            clahe_clip=result.clahe_clip,
            dropped_fraction=dropped,
            output_level=out_level,
            brightness_error_pct=err,
            latency_ms=(time.perf_counter() - t_start) * 1000.0,
            transition=transition,
            message=message,
        )

    def _check_startup(self, state: LightState, t: float) -> str:
        """A stored reference is only valid for the lighting it was recorded in. If the very
        first seconds already read DIMMED/DARK, either the lights really are off or the
        reference is stale (other day, room, angle). Say so, once, so it is not a mystery."""
        s = self._s
        if self._t_first is None:
            self._t_first = t
        if state is LightState.NORMAL or self._mismatch_warned:
            return ""
        if t - self._t_first <= s.startup_check_s + s.confirm_seconds + 1e-9:
            self._mismatch_warned = True
            return (
                f"Started {state.value} against a stored reference. If the lights were on, "
                "press r to recalibrate"
            )
        return ""

    def _calibrating(
        self, frame: np.ndarray, stats: IlluminationStats, t: float, t_start: float
    ) -> FrameResult:
        self._calibrator.add(stats, t, frame)
        message = ""
        calibrated_now = False
        if self._calibrator.ready:
            try:
                self._reference = self._calibrator.finalize(self._exposure)
                self._ref_image = self._calibrator.image()
                self._build_relighter()
                calibrated_now = True
                self._filter.reset()
                message = "calibration complete"
                if self._reference.spread > self._s.max_reference_spread:
                    message += f" (unstable lighting, spread {self._reference.spread:.2f})"
            except CalibrationError as exc:
                self._calibrator = Calibrator(self._s)
                message = f"calibration failed: {exc}"
        return FrameResult(
            frame=frame,
            phase=self.phase,
            state=LightState.NORMAL,
            level=stats.level,
            level_filtered=stats.level,
            ref_level=self._reference.level if self._reference else 0.0,
            ratio=1.0,
            gain=1.0,
            gamma=1.0,
            clahe_clip=0.0,
            dropped_fraction=0.0,
            output_level=float("nan"),
            brightness_error_pct=float("nan"),
            latency_ms=(time.perf_counter() - t_start) * 1000.0,
            calibrated_now=calibrated_now,
            calibration_progress=1.0 if calibrated_now else self._calibrator.progress(),
            message=message,
        )
