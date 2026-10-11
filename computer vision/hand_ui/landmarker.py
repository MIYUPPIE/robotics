"""MediaPipe HandLandmarker wrapper -> CursorSample.

The thin CV shell around hand_ui/dwell_fsm.py's pure state machine.
Jitter-reduction (OneEuroFilter) lives here, not in the pure core, since
it's about de-noising a sensor reading, not about selection logic -- see
docs/ARCHITECTURE.md#deterministic-core-vs-thin-cv-shell.

Runs MediaPipe Tasks HandLandmarker in synchronous VIDEO mode rather than
async LIVE_STREAM -- simpler per-frame model, adequate for ~1.75s dwell
timing (see docs/ARCHITECTURE.md).

IMPORTANT: `bgr_frame` passed to `process()` is expected to already be
horizontally mirrored by the caller (`cv2.flip(frame, 1)` -- see
kiosk_app/main.py::run()), so the on-screen cursor lines up with the
mirrored camera view the voter sees of themselves. This module does not
mirror coordinates itself; it trusts the input frame's orientation.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from contracts.events import CursorSample

INDEX_FINGERTIP_LANDMARK = 8

# Starting points, not final values -- tune on real hardware. min_cutoff is
# the smoothing strength (Hz) while the cursor is nearly still (lower =
# more jitter suppression while holding steady for a dwell-select); beta
# controls how fast smoothing relaxes as pointing speed (px/s) increases
# (higher = less lag during a fast, deliberate point).
DEFAULT_MIN_CUTOFF = 1.0
DEFAULT_BETA = 0.007


def next_video_timestamp_ms(caller_timestamp_ms: int, last_timestamp_ms: int) -> int:
    """Ratchets a caller-supplied timestamp so it's always strictly greater
    than the last one actually sent to a MediaPipe VIDEO-mode task -- see
    biometric/face_mesh.py::next_video_timestamp_ms for the full rationale
    (duplicated here rather than imported: hand_ui and biometric are
    intentionally independent concerns, and this is a 1-line pure function)."""
    return max(caller_timestamp_ms, last_timestamp_ms + 1)


class OneEuroFilter:
    """1-Euro filter (Casiez, Roussel & Vogel, 2012) for a single scalar
    signal.

    Adaptively smooths out jitter when the signal is nearly stationary --
    exactly the situation while a voter tries to hold the cursor steady for
    a dwell-select -- while staying responsive when it's moving quickly.
    A fixed-alpha exponential moving average can't do both: any alpha low
    enough to kill jitter also makes fast pointing feel laggy, and any
    alpha high enough to feel responsive lets jitter straight through.
    """

    def __init__(
        self,
        min_cutoff: float = DEFAULT_MIN_CUTOFF,
        beta: float = DEFAULT_BETA,
        d_cutoff: float = 1.0,
    ):
        if min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("min_cutoff and d_cutoff must be positive")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev = 0.0
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, t: float) -> float:
        """`t` must be strictly increasing across calls (seconds)."""
        if self._t_prev is None or self._x_prev is None:
            self._x_prev, self._dx_prev, self._t_prev = x, 0.0, t
            return x

        dt = max(t - self._t_prev, 1e-6)
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self._x_prev

        self._x_prev, self._dx_prev, self._t_prev = x_hat, dx_hat, t
        return x_hat

    def reset(self) -> None:
        """Clears state so the next `filter()` call is treated as the first
        sample again -- call this whenever the hand is lost, so no stale
        position/velocity leaks into a newly re-acquired hand."""
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None


class HandCursorTracker:
    def __init__(
        self,
        model_path: str | Path,
        screen_width: int,
        screen_height: int,
        min_cutoff: float = DEFAULT_MIN_CUTOFF,
        beta: float = DEFAULT_BETA,
        num_hands: int = 1,
    ):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._mp_image_cls = mp.Image
        self._mp_image_format = mp.ImageFormat.SRGB
        self.screen_width = screen_width
        self.screen_height = screen_height
        self._filter_x = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self._filter_y = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self._last_timestamp_ms = -1

    def process(self, bgr_frame: np.ndarray, timestamp_ms: int) -> CursorSample:
        timestamp_ms = next_video_timestamp_ms(timestamp_ms, self._last_timestamp_ms)
        self._last_timestamp_ms = timestamp_ms

        rgb_frame = np.ascontiguousarray(bgr_frame[:, :, ::-1])
        mp_image = self._mp_image_cls(image_format=self._mp_image_format, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        t = timestamp_ms / 1000.0
        if not result.hand_landmarks:
            self._filter_x.reset()
            self._filter_y.reset()
            return CursorSample(x=0.0, y=0.0, t=t, present=False)

        landmark = result.hand_landmarks[0][INDEX_FINGERTIP_LANDMARK]
        raw_x = landmark.x * self.screen_width
        raw_y = landmark.y * self.screen_height

        smoothed_x = self._filter_x.filter(raw_x, t)
        smoothed_y = self._filter_y.filter(raw_y, t)

        return CursorSample(x=smoothed_x, y=smoothed_y, t=t, present=True)

    def close(self) -> None:
        self._landmarker.close()
