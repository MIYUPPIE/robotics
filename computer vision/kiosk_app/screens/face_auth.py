"""FACE_AUTH / FACE_AUTH_FAIL / ALREADY_VOTED screens.

No hand-tracking regions here -- the voter just looks at the camera. The
actual capture/liveness/identify orchestration lives in
kiosk_app/main.py::KioskApp._run_face_auth (it needs the biometric embedder,
which main.py owns).
"""

from __future__ import annotations

import cv2
import numpy as np


def render(frame: np.ndarray, instructions: str) -> None:
    h, w = frame.shape[:2]
    cv2.putText(
        frame,
        instructions,
        (w // 2 - min(460, len(instructions) * 10), h - 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
