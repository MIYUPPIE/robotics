"""MediaPipe FaceLandmarker wrapper -> landmark arrays for biometric/liveness.py.

Thin CV shell: the only file that runs MediaPipe's face-mesh model.
biometric/liveness.py stays pure (fed plain (N, 2) landmark arrays), so its
blink logic is unit-tested with zero MediaPipe dependency; this module is
what connects it to a real camera at runtime -- see
docs/ARCHITECTURE.md#deterministic-core-vs-thin-cv-shell.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def next_video_timestamp_ms(caller_timestamp_ms: int, last_timestamp_ms: int) -> int:
    """Ratchets a caller-supplied timestamp so it's always strictly greater
    than the last one actually sent to a MediaPipe VIDEO-mode task.

    MediaPipe's `detect_for_video` requires strictly increasing timestamps
    for the *entire lifetime* of the landmarker object -- but callers that
    reuse one persistent FaceMeshTracker/HandCursorTracker across multiple,
    independently-timed capture attempts (e.g. registration_app re-running
    biometric.capture.capture_liveness on every button click, each of which
    restarts its own clock near zero) would otherwise violate that and crash
    with "Input timestamp must be monotonically increasing." This makes that
    impossible regardless of what the caller passes in.
    """
    return max(caller_timestamp_ms, last_timestamp_ms + 1)


class FaceMeshTracker:
    def __init__(self, model_path: str | Path, num_faces: int = 1):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=num_faces,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._mp_image_cls = mp.Image
        self._mp_image_format = mp.ImageFormat.SRGB
        self._last_timestamp_ms = -1

    def process(self, bgr_frame: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        """Returns an (N, 2) array of pixel-space (x, y) landmark coordinates
        for the first detected face, or None if no face was detected."""
        timestamp_ms = next_video_timestamp_ms(timestamp_ms, self._last_timestamp_ms)
        self._last_timestamp_ms = timestamp_ms

        rgb_frame = np.ascontiguousarray(bgr_frame[:, :, ::-1])
        mp_image = self._mp_image_cls(image_format=self._mp_image_format, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            return None
        h, w = bgr_frame.shape[:2]
        return np.array([(lm.x * w, lm.y * h) for lm in result.face_landmarks[0]])

    def close(self) -> None:
        self._landmarker.close()
