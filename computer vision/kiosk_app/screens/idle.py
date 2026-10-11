"""IDLE_ATTRACT screen: attract loop, waiting for a voter to step up."""

from __future__ import annotations

import cv2
import numpy as np

TITLE = "Nigeria Voter Kiosk (Prototype)"
MESSAGE = "Step forward to begin"


def render(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    cv2.putText(
        frame,
        TITLE,
        (w // 2 - 380, h // 2 - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        MESSAGE,
        (w // 2 - 220, h // 2 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )
