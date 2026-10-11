import numpy as np
import pytest

from biometric.liveness import (
    LEFT_EYE_EAR_INDICES,
    RIGHT_EYE_EAR_INDICES,
    BlinkDetector,
    detect_blink,
    eye_aspect_ratio,
    mean_ear,
)


def _eye_points(ear: float, width: float = 1.0) -> np.ndarray:
    """Synthetic 6-point eye shape with a known, exact EAR value."""
    h = ear * width / 2.0
    return np.array(
        [
            (0.0, 0.0),  # p1: outer corner
            (0.25 * width, h),  # p2: upper lid
            (0.75 * width, h),  # p3: upper lid
            (width, 0.0),  # p4: inner corner
            (0.75 * width, -h),  # p5: lower lid
            (0.25 * width, -h),  # p6: lower lid
        ]
    )


def _face_landmarks(left_ear: float, right_ear: float, n_points: int = 478) -> np.ndarray:
    landmarks = np.zeros((n_points, 2), dtype=np.float64)
    for idx, pt in zip(LEFT_EYE_EAR_INDICES, _eye_points(left_ear), strict=True):
        landmarks[idx] = pt
    for idx, pt in zip(RIGHT_EYE_EAR_INDICES, _eye_points(right_ear), strict=True):
        landmarks[idx] = pt
    return landmarks


class TestEyeAspectRatio:
    def test_matches_requested_value(self):
        for target in (0.10, 0.21, 0.30, 0.45):
            assert eye_aspect_ratio(_eye_points(target)) == pytest.approx(target, abs=1e-9)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            eye_aspect_ratio(np.zeros((4, 2)))

    def test_zero_width_is_safe(self):
        degenerate = np.zeros((6, 2))
        assert eye_aspect_ratio(degenerate) == 0.0


class TestMeanEar:
    def test_averages_both_eyes(self):
        landmarks = _face_landmarks(left_ear=0.20, right_ear=0.30)
        assert mean_ear(landmarks) == pytest.approx(0.25, abs=1e-9)


class TestBlinkDetector:
    def test_full_blink_cycle_is_detected(self):
        detector = BlinkDetector(ear_threshold=0.21, consec_frames=2)
        stream = (
            [_face_landmarks(0.30, 0.30)] * 3  # open
            + [_face_landmarks(0.10, 0.10)] * 3  # closed (below threshold)
            + [_face_landmarks(0.30, 0.30)] * 3  # open again
        )
        for landmarks in stream:
            detector.update(landmarks)
        result = detector.result()
        assert result.is_live
        assert result.blink_detected
        assert result.frames_observed == 9

    def test_no_blink_when_eyes_stay_open(self):
        result = detect_blink([_face_landmarks(0.30, 0.30) for _ in range(10)])
        assert not result.is_live
        assert not result.blink_detected
        assert "no blink" in result.reason

    def test_single_frame_closure_below_consec_frames_is_not_a_blink(self):
        detector = BlinkDetector(ear_threshold=0.21, consec_frames=3)
        stream = (
            [_face_landmarks(0.30, 0.30)] * 2
            + [_face_landmarks(0.10, 0.10)] * 2  # only 2 frames, need 3
            + [_face_landmarks(0.30, 0.30)] * 2
        )
        for landmarks in stream:
            detector.update(landmarks)
        assert not detector.result().is_live

    def test_static_photo_replay_never_blinks(self):
        """A static photo (constant open-eye EAR, zero variance) must never
        register as live -- this is the liveness check's whole purpose."""
        result = detect_blink([_face_landmarks(0.32, 0.32) for _ in range(30)])
        assert not result.is_live

    def test_closure_still_in_progress_at_window_end_counts_as_blink(self):
        detector = BlinkDetector(ear_threshold=0.21, consec_frames=2)
        for landmarks in [_face_landmarks(0.30, 0.30)] * 2 + [_face_landmarks(0.10, 0.10)] * 3:
            detector.update(landmarks)
        assert detector.result().is_live
