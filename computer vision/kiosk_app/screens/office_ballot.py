"""OFFICE_BALLOT screen: two-step select-then-confirm party picker.

Step 1 (SELECT): dwelling on a party card provisionally picks it.
Step 2 (CONFIRM): dwelling CONFIRM locks it in; dwelling CHANGE returns to
step 1 to pick a different party. No single dwell can cast a vote -- see
docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import cv2
import numpy as np

from contracts.events import Region
from hand_ui import renderer
from hand_ui.regions import grid_layout
from kiosk_app import config

CONFIRM_REGION_ID = "CONFIRM"
CHANGE_REGION_ID = "CHANGE"


def build_party_regions(
    party_acronyms: list[str],
    screen_width: int = config.SCREEN_WIDTH,
    screen_height: int = config.SCREEN_HEIGHT,
    columns: int = config.PARTY_GRID_COLUMNS,
) -> list[Region]:
    cell_w, cell_h = 300, 180
    gap = (24, 24)
    rows = (len(party_acronyms) + columns - 1) // columns
    total_w = cell_w * columns + gap[0] * (columns - 1)
    total_h = cell_h * rows + gap[1] * (rows - 1)
    origin = ((screen_width - total_w) / 2, (screen_height - total_h) / 2 - 40)
    return grid_layout(
        party_acronyms, origin=origin, cell_size=(cell_w, cell_h), columns=columns, gap=gap
    )


def build_confirm_regions(
    screen_width: int = config.SCREEN_WIDTH, screen_height: int = config.SCREEN_HEIGHT
) -> list[Region]:
    cell_w, cell_h = 320, 120
    gap = 60
    total_w = cell_w * 2 + gap
    origin_x = (screen_width - total_w) / 2
    y0 = screen_height - cell_h - 60
    return [
        Region(CONFIRM_REGION_ID, origin_x, y0, origin_x + cell_w, y0 + cell_h),
        Region(
            CHANGE_REGION_ID,
            origin_x + cell_w + gap,
            y0,
            origin_x + cell_w + gap + cell_w,
            y0 + cell_h,
        ),
    ]


def render_select_step(
    frame: np.ndarray,
    regions: list[Region],
    party_colors: dict[str, tuple[int, int, int]],
    party_logos: dict[str, np.ndarray],
    hover_fraction: dict[str, float],
    office_label: str,
) -> None:
    cv2.putText(
        frame,
        f"Choose a party for: {office_label}",
        (40, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for region in regions:
        color = party_colors.get(region.region_id, (100, 100, 100))
        logo = party_logos.get(region.region_id)
        renderer.draw_region(frame, region, region.region_id, fill_color=color, logo=logo)
    _draw_progress(frame, regions, hover_fraction)


# Preview box for the confirm step's logo -- fixed placement, not hit-tested.
_PREVIEW_BOX = Region("PREVIEW", 40, 70, 340, 260)


def render_confirm_step(
    frame: np.ndarray,
    regions: list[Region],
    provisional_party: str,
    party_logos: dict[str, np.ndarray],
    hover_fraction: dict[str, float],
) -> None:
    cv2.putText(
        frame,
        f"You selected: {provisional_party}",
        (40, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    logo = party_logos.get(provisional_party)
    if logo is not None:
        renderer.draw_logo(frame, _PREVIEW_BOX, logo)
    for region in regions:
        color = (0, 140, 0) if region.region_id == CONFIRM_REGION_ID else (0, 0, 140)
        renderer.draw_region(frame, region, region.region_id, fill_color=color)
    _draw_progress(frame, regions, hover_fraction)


def _draw_progress(
    frame: np.ndarray, regions: list[Region], hover_fraction: dict[str, float]
) -> None:
    for region_id, fraction in hover_fraction.items():
        region = next((r for r in regions if r.region_id == region_id), None)
        if region is None:
            continue
        cx, cy = (region.x0 + region.x1) / 2, (region.y0 + region.y1) / 2
        renderer.draw_progress_ring(frame, cx, cy, fraction)
