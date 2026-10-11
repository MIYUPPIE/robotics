"""Camera burst-capture orchestration.

`burst_capture` is the thin CV shell, but it takes `read_frame` and `embedder`
as injected dependencies (duck-typed: `embedder.detect(frame) -> list[DetectedFace]`),
so its orchestration logic (qualifying-frame filtering, timeout handling) is
gate-tested with fakes -- no real camera or InsightFace model needed. Only a
real camera + real FaceEmbedder in biometric/evals/ exercises the full path.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from biometric.embedding import DetectedFace
from biometric.liveness import BlinkDetector
from biometric.thresholds import (
    BURST_CAPTURE_FRAMES,
    BURST_CAPTURE_TIMEOUT_S,
    BURST_MIN_DETECTOR_CONFIDENCE,
    LIVENESS_BLINK_WINDOW_S,
)
from contracts.schemas import LivenessResult


class Embedder(Protocol):
    def detect(self, bgr_frame: np.ndarray) -> list[DetectedFace]: ...


class FaceMesh(Protocol):
    def process(self, bgr_frame: np.ndarray, timestamp_ms: int) -> np.ndarray | None: ...


@dataclass(frozen=True)
class CapturedFrame:
    frame: np.ndarray  # BGR
    face: DetectedFace


def _is_unclipped(
    bbox: tuple[float, float, float, float],
    frame_shape: tuple[int, ...],
    margin_px: float = 4.0,
) -> bool:
    x0, y0, x1, y1 = bbox
    h, w = frame_shape[0], frame_shape[1]
    return x0 > margin_px and y0 > margin_px and x1 < (w - margin_px) and y1 < (h - margin_px)


def burst_capture(
    read_frame: Callable[[], np.ndarray | None],
    embedder: Embedder,
    n: int = BURST_CAPTURE_FRAMES,
    timeout_s: float = BURST_CAPTURE_TIMEOUT_S,
    min_confidence: float = BURST_MIN_DETECTOR_CONFIDENCE,
    clock: Callable[[], float] = time.monotonic,
) -> list[CapturedFrame]:
    """Grab frames via `read_frame()` until `n` qualifying single-face frames
    are collected or `timeout_s` elapses.

    A qualifying frame has exactly one detected face, an unclipped bounding
    box, and detector confidence >= min_confidence.
    """
    captured: list[CapturedFrame] = []
    deadline = clock() + timeout_s
    while len(captured) < n and clock() < deadline:
        frame = read_frame()
        if frame is None:
            continue
        faces = embedder.detect(frame)
        if len(faces) != 1:
            continue
        face = faces[0]
        if face.detection_confidence < min_confidence:
            continue
        if not _is_unclipped(face.bbox, frame.shape):
            continue
        captured.append(CapturedFrame(frame=frame, face=face))
    return captured


def best_frames(captured: list[CapturedFrame], k: int) -> list[CapturedFrame]:
    """Rank by bbox area x detector confidence, return the top k."""

    def score(cf: CapturedFrame) -> float:
        x0, y0, x1, y1 = cf.face.bbox
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        return area * cf.face.detection_confidence

    return sorted(captured, key=score, reverse=True)[:k]


def average_embedding(frames: list[CapturedFrame]) -> np.ndarray:
    """Mean of the frames' embeddings, re-normalized to unit length."""
    if not frames:
        raise ValueError("no frames to average")
    stacked = np.stack([cf.face.embedding for cf in frames]).astype(np.float32)
    mean = stacked.mean(axis=0)
    norm = np.linalg.norm(mean)
    return (mean / norm).astype(np.float32) if norm > 0 else mean


def capture_liveness(
    read_frame: Callable[[], np.ndarray | None],
    face_mesh: FaceMesh,
    window_s: float = LIVENESS_BLINK_WINDOW_S,
    clock: Callable[[], float] = time.monotonic,
) -> LivenessResult:
    """Blocking liveness collection loop for a single supervised attempt
    (used by registration_app, which runs this synchronously inside a button
    handler). kiosk_app/main.py does NOT use this -- its real-time render
    loop needs a non-blocking, incremental per-frame version, implemented
    directly against biometric.liveness.BlinkDetector there instead.

    `face_mesh` is duck-typed (`.process(frame, timestamp_ms) -> landmarks |
    None`), so this is gate-tested with a fake -- no real MediaPipe model.
    """
    detector = BlinkDetector()
    start = clock()
    while clock() - start < window_s:
        frame = read_frame()
        if frame is None:
            continue
        timestamp_ms = int((clock() - start) * 1000)
        landmarks = face_mesh.process(frame, timestamp_ms)
        if landmarks is None:
            continue
        detector.update(landmarks)
        if detector.result().is_live:
            break
    return detector.result()
