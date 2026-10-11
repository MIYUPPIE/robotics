"""LANGUAGE_SELECT screen: one dwell region per supported language."""

from __future__ import annotations

import cv2
import numpy as np

from contracts.events import Region
from contracts.schemas import Language
from hand_ui import renderer
from hand_ui.regions import grid_layout
from kiosk_app import config

LANGUAGE_ORDER: list[Language] = [Language.ENGLISH, Language.YORUBA, Language.IGBO, Language.HAUSA]

LANGUAGE_LABELS: dict[Language, str] = {
    Language.ENGLISH: "English",
    Language.YORUBA: "Yoruba",
    Language.IGBO: "Igbo",
    Language.HAUSA: "Hausa",
}


def build_regions(
    screen_width: int = config.SCREEN_WIDTH, screen_height: int = config.SCREEN_HEIGHT
) -> list[Region]:
    cell_w, cell_h = 380, 220
    gap = (40, 40)
    total_w = cell_w * len(LANGUAGE_ORDER) + gap[0] * (len(LANGUAGE_ORDER) - 1)
    origin = ((screen_width - total_w) / 2, (screen_height - cell_h) / 2)
    return grid_layout(
        [lang.value for lang in LANGUAGE_ORDER],
        origin=origin,
        cell_size=(cell_w, cell_h),
        columns=len(LANGUAGE_ORDER),
        gap=gap,
    )


def render(frame: np.ndarray, regions: list[Region], hover_fraction: dict[str, float]) -> None:
    h, w = frame.shape[:2]
    cv2.putText(
        frame,
        "Choose your language",
        (w // 2 - 260, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for region in regions:
        language = Language(region.region_id)
        renderer.draw_region(frame, region, LANGUAGE_LABELS[language], fill_color=(70, 70, 70))
    for region_id, fraction in hover_fraction.items():
        progress_region = next((r for r in regions if r.region_id == region_id), None)
        if progress_region is None:
            continue
        cx = (progress_region.x0 + progress_region.x1) / 2
        cy = (progress_region.y0 + progress_region.y1) / 2
        renderer.draw_progress_ring(frame, cx, cy, fraction)
