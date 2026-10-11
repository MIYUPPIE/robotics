"""REVIEW screen (DRE-voting-machine style): lists all 5 office->party
selections with a per-row EDIT jump, plus SUBMIT (two-step confirm) and
START_OVER.
"""

from __future__ import annotations

import cv2
import numpy as np

from contracts.events import Region
from contracts.schemas import Office
from hand_ui import renderer

SUBMIT_REGION_ID = "SUBMIT"
SUBMIT_CONFIRM_REGION_ID = "SUBMIT_CONFIRM"
SUBMIT_CANCEL_REGION_ID = "SUBMIT_CANCEL"
START_OVER_REGION_ID = "START_OVER"

_EDIT_PREFIX = "EDIT_"


def edit_region_id(office: Office) -> str:
    return f"{_EDIT_PREFIX}{office.value}"


def office_from_edit_region(region_id: str) -> Office:
    return Office(region_id.removeprefix(_EDIT_PREFIX))


def is_edit_region(region_id: str) -> bool:
    return region_id.startswith(_EDIT_PREFIX)


def build_regions(offices: list[Office], screen_width: int, screen_height: int) -> list[Region]:
    row_h = 100
    top = 120
    regions = [
        Region(
            edit_region_id(office),
            100,
            top + i * row_h,
            screen_width - 100,
            top + i * row_h + row_h - 10,
        )
        for i, office in enumerate(offices)
    ]
    button_y = top + len(offices) * row_h + 40
    regions.append(Region(SUBMIT_REGION_ID, 100, button_y, 500, button_y + 100))
    regions.append(
        Region(
            START_OVER_REGION_ID, screen_width - 500, button_y, screen_width - 100, button_y + 100
        )
    )
    return regions


def build_submit_confirm_regions(screen_width: int, screen_height: int) -> list[Region]:
    y0 = screen_height - 200
    return [
        Region(SUBMIT_CONFIRM_REGION_ID, 100, y0, 500, y0 + 100),
        Region(SUBMIT_CANCEL_REGION_ID, screen_width - 500, y0, screen_width - 100, y0 + 100),
    ]


def render(
    frame: np.ndarray,
    offices: list[Office],
    selections: dict[Office, str],
    regions: list[Region],
    hover_fraction: dict[str, float],
) -> None:
    cv2.putText(
        frame,
        "Review your ballot",
        (100, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for office in offices:
        region = next(r for r in regions if r.region_id == edit_region_id(office))
        party = selections.get(office, "-")
        renderer.draw_region(
            frame, region, f"{office.value}: {party}  (edit)", fill_color=(60, 60, 60)
        )
    for special_id, color in ((SUBMIT_REGION_ID, (0, 140, 0)), (START_OVER_REGION_ID, (0, 0, 140))):
        special_region = next((r for r in regions if r.region_id == special_id), None)
        if special_region is not None:
            renderer.draw_region(frame, special_region, special_id, fill_color=color)
    for region_id, fraction in hover_fraction.items():
        progress_region = next((r for r in regions if r.region_id == region_id), None)
        if progress_region is None:
            continue
        cx = (progress_region.x0 + progress_region.x1) / 2
        cy = (progress_region.y0 + progress_region.y1) / 2
        renderer.draw_progress_ring(frame, cx, cy, fraction)
