"""THANK_YOU screen: brief confirmation display after a successful submit."""

from __future__ import annotations

import cv2
import numpy as np


def render(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    cv2.putText(
        frame,
        "Thank you for voting!",
        (w // 2 - 320, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 220, 0),
        3,
        cv2.LINE_AA,
    )
