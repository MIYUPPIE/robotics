import numpy as np
import pytest

from biometric.capture import (
    CapturedFrame,
    average_embedding,
    best_frames,
    burst_capture,
    capture_liveness,
)
from biometric.embedding import DetectedFace
from biometric.liveness import LEFT_EYE_EAR_INDICES, RIGHT_EYE_EAR_INDICES


class FakeEmbedder:
    """Duck-types biometric.capture.Embedder -- returns one canned response per call."""

    def __init__(self, responses):
        self._responses = iter(responses)

    def detect(self, frame):
        return next(self._responses)


class FakeClock:
    """Deterministic monotonic-clock stand-in -- advances by `step` each call."""

    def __init__(self, start: float = 0.0, step: float = 0.01):
        self.t = start
        self.step = step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


def _face(bbox=(10.0, 10.0, 100.0, 100.0), confidence=0.9, seed=0) -> DetectedFace:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return DetectedFace(bbox=bbox, detection_confidence=confidence, embedding=v / np.linalg.norm(v))


def _frame(shape=(480, 640, 3)) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint8)


class TestBurstCapture:
    def test_collects_n_qualifying_frames(self):
        good = _face()
        embedder = FakeEmbedder([[good]] * 5)
        frames = iter([_frame()] * 5)
        result = burst_capture(
            lambda: next(frames, None), embedder, n=5, timeout_s=10.0, clock=FakeClock(step=0.001)
        )
        assert len(result) == 5

    def test_skips_frames_with_zero_or_multiple_faces(self):
        good = _face()
        embedder = FakeEmbedder([[], [good, good], [good]])
        frames = iter([_frame()] * 3)
        result = burst_capture(
            lambda: next(frames, None), embedder, n=1, timeout_s=10.0, clock=FakeClock(step=0.001)
        )
        assert len(result) == 1

    def test_skips_low_confidence_faces(self):
        low = _face(confidence=0.1)
        high = _face(confidence=0.9)
        embedder = FakeEmbedder([[low], [high]])
        frames = iter([_frame()] * 2)
        result = burst_capture(
            lambda: next(frames, None), embedder, n=1, timeout_s=10.0, clock=FakeClock(step=0.001)
        )
        assert len(result) == 1
        assert result[0].face.detection_confidence == 0.9

    def test_skips_clipped_bounding_box(self):
        clipped = _face(bbox=(0.0, 10.0, 100.0, 100.0))  # touches left edge
        unclipped = _face(bbox=(10.0, 10.0, 100.0, 100.0))
        embedder = FakeEmbedder([[clipped], [unclipped]])
        frames = iter([_frame()] * 2)
        result = burst_capture(
            lambda: next(frames, None), embedder, n=1, timeout_s=10.0, clock=FakeClock(step=0.001)
        )
        assert len(result) == 1

    def test_returns_partial_result_on_timeout(self):
        good = _face()
        embedder = FakeEmbedder([[good]] * 10)
        frames = iter([_frame()] * 10)
        # step chosen so the loop condition fails after exactly one iteration.
        result = burst_capture(
            lambda: next(frames, None),
            embedder,
            n=5,
            timeout_s=1.0,
            clock=FakeClock(start=0.0, step=0.6),
        )
        assert 0 < len(result) < 5

    def test_none_frame_is_skipped_without_crashing(self):
        good = _face()
        embedder = FakeEmbedder([[good]])
        frames = iter([None, _frame()])
        result = burst_capture(
            lambda: next(frames, None), embedder, n=1, timeout_s=10.0, clock=FakeClock(step=0.001)
        )
        assert len(result) == 1


class TestBestFrames:
    def test_ranks_by_area_times_confidence(self):
        small = CapturedFrame(
            frame=_frame(), face=_face(bbox=(0, 0, 10, 10), confidence=0.9, seed=1)
        )
        large = CapturedFrame(
            frame=_frame(), face=_face(bbox=(0, 0, 100, 100), confidence=0.9, seed=2)
        )
        ranked = best_frames([small, large], k=2)
        assert ranked[0] is large

    def test_returns_at_most_k(self):
        frames = [CapturedFrame(frame=_frame(), face=_face(seed=i)) for i in range(5)]
        assert len(best_frames(frames, k=3)) == 3


class TestAverageEmbedding:
    def test_averages_and_renormalizes(self):
        f1 = CapturedFrame(frame=_frame(), face=_face(seed=1))
        f2 = CapturedFrame(frame=_frame(), face=_face(seed=2))
        avg = average_embedding([f1, f2])
        assert avg.shape == (512,)
        assert np.linalg.norm(avg) == pytest.approx(1.0, abs=1e-5)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            average_embedding([])


def _eye_points(ear: float, width: float = 1.0) -> np.ndarray:
    h = ear * width / 2.0
    return np.array(
        [
            (0.0, 0.0),
            (0.25 * width, h),
            (0.75 * width, h),
            (width, 0.0),
            (0.75 * width, -h),
            (0.25 * width, -h),
        ]
    )


def _face_landmarks(left_ear: float, right_ear: float, n_points: int = 478) -> np.ndarray:
    landmarks = np.zeros((n_points, 2))
    for idx, pt in zip(LEFT_EYE_EAR_INDICES, _eye_points(left_ear), strict=True):
        landmarks[idx] = pt
    for idx, pt in zip(RIGHT_EYE_EAR_INDICES, _eye_points(right_ear), strict=True):
        landmarks[idx] = pt
    return landmarks


class FakeFaceMesh:
    """Duck-types biometric.capture.FaceMesh -- returns one canned landmark
    array (or None) per call, cycling if it runs out."""

    def __init__(self, responses: list):
        self._responses = responses
        self._i = 0

    def process(self, frame, timestamp_ms):
        response = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return response


class TestCaptureLiveness:
    def test_detects_a_blink_within_the_window(self):
        stream = (
            [_face_landmarks(0.30, 0.30)] * 3
            + [_face_landmarks(0.10, 0.10)] * 3
            + [_face_landmarks(0.30, 0.30)] * 3
        )
        face_mesh = FakeFaceMesh(stream)
        frames = iter([_frame()] * 20)
        clock = FakeClock(step=0.05)

        result = capture_liveness(lambda: next(frames, None), face_mesh, window_s=5.0, clock=clock)
        assert result.is_live

    def test_static_stream_is_not_live(self):
        stream = [_face_landmarks(0.30, 0.30)] * 50
        face_mesh = FakeFaceMesh(stream)
        frames = iter([_frame()] * 50)
        # Small step so many frames fit before the window elapses.
        clock = FakeClock(step=0.05)

        result = capture_liveness(lambda: next(frames, None), face_mesh, window_s=1.0, clock=clock)
        assert not result.is_live

    def test_none_frame_and_none_landmarks_are_skipped_without_crashing(self):
        face_mesh = FakeFaceMesh([None, _face_landmarks(0.30, 0.30)])
        frames = iter([None, _frame(), _frame()])
        clock = FakeClock(step=0.05)

        result = capture_liveness(lambda: next(frames, None), face_mesh, window_s=1.0, clock=clock)
        assert result.frames_observed >= 0  # must not raise regardless of outcome
