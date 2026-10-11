"""Region layout + hit-testing helpers, used by kiosk_app screens to build the
party-selection grid and by hand_ui/renderer.py to draw it. Pure -- no
MediaPipe/OpenCV dependency.
"""

from __future__ import annotations

from contracts.events import Region


def hit_test(regions: list[Region], x: float, y: float) -> str | None:
    for region in regions:
        if region.contains(x, y):
            return region.region_id
    return None


def fit_within_region(
    image_w: int, image_h: int, region: Region, padding_frac: float = 0.12
) -> tuple[int, int, int, int]:
    """Aspect-fit + center placement for drawing an `image_w` x `image_h` image
    (e.g. a party logo) inside `region`, inset by `padding_frac` of the
    region's shorter side on every edge. Returns integer (x, y, w, h) in the
    same coordinate space as `region`. Pure geometry -- no image data touched,
    see hand_ui/renderer.py for the actual pixel compositing.
    """
    if image_w <= 0 or image_h <= 0:
        raise ValueError("image_w and image_h must be positive")
    region_w = region.x1 - region.x0
    region_h = region.y1 - region.y0
    pad = padding_frac * min(region_w, region_h)
    avail_w = max(region_w - 2 * pad, 1.0)
    avail_h = max(region_h - 2 * pad, 1.0)
    scale = min(avail_w / image_w, avail_h / image_h)
    w = max(int(round(image_w * scale)), 1)
    h = max(int(round(image_h * scale)), 1)
    x = int(round(region.x0 + (region_w - w) / 2))
    y = int(round(region.y0 + (region_h - h) / 2))
    return x, y, w, h


def grid_layout(
    region_ids: list[str],
    origin: tuple[float, float],
    cell_size: tuple[float, float],
    columns: int,
    gap: tuple[float, float] = (0.0, 0.0),
) -> list[Region]:
    """Lay out `region_ids` left-to-right, top-to-bottom in a grid of
    `columns` columns, each cell `cell_size` big, starting at `origin`, with
    `gap` spacing between cells.
    """
    if columns <= 0:
        raise ValueError("columns must be positive")
    ox, oy = origin
    cw, ch = cell_size
    gx, gy = gap
    regions = []
    for i, region_id in enumerate(region_ids):
        col = i % columns
        row = i // columns
        x0 = ox + col * (cw + gx)
        y0 = oy + row * (ch + gy)
        regions.append(Region(region_id=region_id, x0=x0, y0=y0, x1=x0 + cw, y1=y0 + ch))
    return regions
