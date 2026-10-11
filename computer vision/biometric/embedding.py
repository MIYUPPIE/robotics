"""InsightFace (ArcFace, buffalo_l) face detection + embedding extraction.

The only file in this project that imports `insightface`/`onnxruntime`. The
provider-selection *logic* (`select_provider`) is a pure function, gate-tested
without either dependency being importable; `FaceEmbedder` itself needs the
real packages and model weights, so it's exercised only by
biometric/evals/face_match_eval.py, never by the gate-test suite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contracts.schemas import EMBEDDING_DIM

MODEL_VERSION = "insightface-buffalo_l"


@dataclass(frozen=True)
class DetectedFace:
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixel coordinates
    detection_confidence: float
    embedding: np.ndarray  # shape (EMBEDDING_DIM,), L2-normalized float32


def select_provider(available_providers: list[str], preferred: str | None = None) -> list[str]:
    """Pure onnxruntime execution-provider priority: preferred > CUDA > CPU."""
    if preferred and preferred in available_providers:
        return [preferred]
    if "CUDAExecutionProvider" in available_providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class FaceEmbedder:
    """Thin wrapper around insightface.app.FaceAnalysis (buffalo_l)."""

    MODEL_VERSION = MODEL_VERSION

    def __init__(self, provider: str | None = None, det_size: tuple[int, int] = (640, 640)):
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        providers = select_provider(ort.get_available_providers(), provider)
        self._app = FaceAnalysis(name="buffalo_l", providers=providers)
        self._app.prepare(ctx_id=0, det_size=det_size)

    def detect(self, bgr_frame: np.ndarray) -> list[DetectedFace]:
        """Detect every face in a BGR frame and return its embedding + bbox."""
        faces = self._app.get(bgr_frame)
        results: list[DetectedFace] = []
        for face in faces:
            embedding = np.asarray(face.normed_embedding, dtype=np.float32)
            if embedding.shape != (EMBEDDING_DIM,):
                continue
            x0, y0, x1, y1 = face.bbox
            results.append(
                DetectedFace(
                    bbox=(float(x0), float(y0), float(x1), float(y1)),
                    detection_confidence=float(face.det_score),
                    embedding=embedding,
                )
            )
        return results
