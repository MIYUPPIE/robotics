"""Eye-Aspect-Ratio (EAR) blink-based liveness detection.

Pure functions/classes over face-mesh landmark coordinates -- no MediaPipe or
camera dependency, so this is fully unit-testable with synthetic landmarks
(see biometric/tests/test_liveness.py). The threshold/consecutive-frame logic
follows Soukupová & Čech's classic real-time EAR blink detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from biometric.thresholds import LIVENESS_CONSEC_FRAMES, LIVENESS_EAR_THRESHOLD
from contracts.schemas import LivenessResult

# MediaPipe FaceLandmarker 478-point mesh indices for each eye's 6 EAR
# landmarks, in the (p1..p6) order the classic EAR formula expects:
# p1/p4 = outer/inner corner, p2/p3 = upper lid, p6/p5 = lower lid.
LEFT_EYE_EAR_INDICES: tuple[int, int, int, int, int, int] = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_EAR_INDICES: tuple[int, int, int, int, int, int] = (362, 385, 387, 263, 373, 380)


def eye_aspect_ratio(eye_points: np.ndarray) -> float:
    """eye_points: shape (6, 2), rows in (p1..p6) order.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    eye_points = np.asarray(eye_points, dtype=np.float64)
    if eye_points.shape != (6, 2):
        raise ValueError(f"eye_points must have shape (6, 2), got {eye_points.shape}")
    p1, p2, p3, p4, p5, p6 = eye_points
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal == 0:
        return 0.0
    return float(vertical / (2.0 * horizontal))


def extract_eye_points(landmarks: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    """landmarks: shape (N, 2+) full face mesh; returns shape (6, 2) for the given indices."""
    return np.asarray(landmarks)[list(indices), :2]


def mean_ear(landmarks: np.ndarray) -> float:
    """Average EAR across both eyes for one frame's full face-mesh landmarks."""
    left = extract_eye_points(landmarks, LEFT_EYE_EAR_INDICES)
    right = extract_eye_points(landmarks, RIGHT_EYE_EAR_INDICES)
    return (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0


@dataclass
class BlinkDetector:
    """Stateful across a stream of frames within one capture window.

    Feed it one full-face-mesh landmark array per frame via `update()`, then
    read `result()`. A full blink = EAR below threshold for >= consec_frames
    consecutive frames, followed by EAR rising back above threshold.
    """

    ear_threshold: float = LIVENESS_EAR_THRESHOLD
    consec_frames: int = LIVENESS_CONSEC_FRAMES
    _below_threshold_run: int = field(default=0, repr=False)
    _blink_confirmed: bool = field(default=False, repr=False)
    _frames_observed: int = field(default=0, repr=False)

    def update(self, landmarks: np.ndarray) -> None:
        self._frames_observed += 1
        ear = mean_ear(landmarks)
        if ear < self.ear_threshold:
            self._below_threshold_run += 1
        else:
            if self._below_threshold_run >= self.consec_frames:
                self._blink_confirmed = True
            self._below_threshold_run = 0

    def result(self) -> LivenessResult:
        # A closure run still in progress at the end of the window that
        # already cleared the consecutive-frame bar counts as a completed
        # blink too (voter's eyes may still be closed on the last frame).
        blink = self._blink_confirmed or self._below_threshold_run >= self.consec_frames
        return LivenessResult(
            is_live=blink,
            blink_detected=blink,
            frames_observed=self._frames_observed,
            reason="" if blink else "no blink observed within the capture window",
        )


def detect_blink(
    landmark_stream: list[np.ndarray],
    ear_threshold: float = LIVENESS_EAR_THRESHOLD,
    consec_frames: int = LIVENESS_CONSEC_FRAMES,
) -> LivenessResult:
    """Convenience wrapper: run a whole pre-collected landmark stream through BlinkDetector."""
    detector = BlinkDetector(ear_threshold=ear_threshold, consec_frames=consec_frames)
    for landmarks in landmark_stream:
        detector.update(landmarks)
    return detector.result()
