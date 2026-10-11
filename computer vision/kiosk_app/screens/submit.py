"""SUBMITTING / SUBMIT_FAILED screens."""

from __future__ import annotations

import cv2
import numpy as np

from contracts.events import Region
from hand_ui import renderer

RETRY_REGION_ID = "RETRY_SUBMIT"


def render_submitting(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    cv2.putText(
        frame,
        "Submitting your ballot...",
        (w // 2 - 320, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def build_retry_region(screen_width: int, screen_height: int) -> Region:
    y0 = screen_height - 200
    return Region(RETRY_REGION_ID, screen_width // 2 - 200, y0, screen_width // 2 + 200, y0 + 100)


def render_submit_failed(frame: np.ndarray, retry_region: Region, hover_fraction: float) -> None:
    h, w = frame.shape[:2]
    cv2.putText(
        frame,
        "Something went wrong submitting your ballot.",
        (w // 2 - 460, h // 2 - 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 220),
        2,
        cv2.LINE_AA,
    )
    renderer.draw_region(frame, retry_region, "RETRY", fill_color=(0, 0, 140))
    cx = (retry_region.x0 + retry_region.x1) / 2
    cy = (retry_region.y0 + retry_region.y1) / 2
    renderer.draw_progress_ring(frame, cx, cy, hover_fraction)
