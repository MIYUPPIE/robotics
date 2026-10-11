"""OpenCV rendering: region boxes, cursor, radial dwell-progress ring.

Visual output only -- not gate-tested. The logic it renders (what's hovered,
how far along the dwell is) is fully covered by hand_ui/tests/test_dwell_fsm.py.
"""

from __future__ import annotations

import cv2
import numpy as np

from contracts.events import Region
from hand_ui.regions import fit_within_region

CURSOR_RADIUS_PX = 14
CURSOR_FILL_COLOR = (255, 255, 255)
CURSOR_OUTLINE_COLOR = (0, 0, 0)
PROGRESS_RING_COLOR = (0, 200, 255)
PROGRESS_RING_THICKNESS = 6
PROGRESS_RING_RADIUS_PX = 40
REGION_BORDER_COLOR = (200, 200, 200)
REGION_BORDER_THICKNESS = 2
REGION_HIGHLIGHT_THICKNESS = 5
REGION_LABEL_COLOR = (255, 255, 255)
LOGO_CARD_BACKGROUND = (255, 255, 255)
LOGO_CARD_LABEL_COLOR = (20, 20, 20)
LOGO_PADDING_FRAC = 0.14


def draw_region(
    frame: np.ndarray,
    region: Region,
    label: str,
    fill_color: tuple[int, int, int],
    highlighted: bool = False,
    logo: np.ndarray | None = None,
) -> None:
    """Draws a ballot card. With no `logo`, this is a filled+labeled color
    card (the original look, still used for parties with no logo asset -- see
    kiosk_app/party_logos.py). With a `logo` (BGRA, from
    kiosk_app.party_logos.load_party_logos), the card gets a white background
    so the real logo artwork reads clearly, `fill_color` becomes the border
    accent instead of the fill, and `label` (the party acronym) is captioned
    below the logo rather than overlaid on top of it.
    """
    pt1 = (int(region.x0), int(region.y0))
    pt2 = (int(region.x1), int(region.y1))
    cv2.rectangle(frame, pt1, pt2, LOGO_CARD_BACKGROUND if logo is not None else fill_color, -1)
    thickness = REGION_HIGHLIGHT_THICKNESS if highlighted else REGION_BORDER_THICKNESS
    border_color = fill_color if logo is not None else REGION_BORDER_COLOR
    cv2.rectangle(frame, pt1, pt2, border_color, thickness)
    if logo is not None:
        logo_region = Region(region.region_id, region.x0, region.y0, region.x1, region.y1 - 26)
        draw_logo(frame, logo_region, logo, padding_frac=LOGO_PADDING_FRAC)
        text_origin = (pt1[0] + 10, pt2[1] - 10)
        text_color = LOGO_CARD_LABEL_COLOR
    else:
        text_origin = (pt1[0] + 10, pt1[1] + 30)
        text_color = REGION_LABEL_COLOR
    cv2.putText(
        frame, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA
    )


def draw_logo(
    frame: np.ndarray, region: Region, logo_bgra: np.ndarray, padding_frac: float = 0.12
) -> None:
    """Alpha-composites `logo_bgra` (H, W, 4 uint8), aspect-fit and centered,
    onto `frame` inside `region`. Geometry comes from
    hand_ui.regions.fit_within_region; this function does the actual pixel
    blending, so it's visual-only and not gate-tested (see module docstring).
    """
    image_h, image_w = logo_bgra.shape[:2]
    x, y, w, h = fit_within_region(image_w, image_h, region, padding_frac)
    _blit_bgra(frame, logo_bgra, x, y, w, h)


def _blit_bgra(frame: np.ndarray, logo_bgra: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    frame_h, frame_w = frame.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, frame_w), min(y + h, frame_h)
    if x1 <= x0 or y1 <= y0:
        return
    resized = cv2.resize(logo_bgra, (w, h), interpolation=cv2.INTER_AREA)
    crop = resized[y0 - y : y1 - y, x0 - x : x1 - x]
    alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
    background = frame[y0:y1, x0:x1].astype(np.float32)
    blended = crop[:, :, :3].astype(np.float32) * alpha + background * (1.0 - alpha)
    frame[y0:y1, x0:x1] = blended.astype(np.uint8)


def draw_cursor(frame: np.ndarray, x: float, y: float) -> None:
    center = (int(x), int(y))
    cv2.circle(frame, center, CURSOR_RADIUS_PX, CURSOR_FILL_COLOR, -1, cv2.LINE_AA)
    cv2.circle(frame, center, CURSOR_RADIUS_PX, CURSOR_OUTLINE_COLOR, 2, cv2.LINE_AA)


def draw_progress_ring(
    frame: np.ndarray, x: float, y: float, fraction: float, radius: int = PROGRESS_RING_RADIUS_PX
) -> None:
    fraction = max(0.0, min(1.0, fraction))
    end_angle = int(360 * fraction)
    cv2.ellipse(
        frame,
        (int(x), int(y)),
        (radius, radius),
        -90,
        0,
        end_angle,
        PROGRESS_RING_COLOR,
        PROGRESS_RING_THICKNESS,
        cv2.LINE_AA,
    )
