"""08b Output and monitoring: side-by-side view, HUD, rolling intensity graph.

Pure drawing code (numpy + cv2 primitives, no window), so it is testable headless.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .light_detector import LightState
from .pipeline import FrameResult, Phase

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_STATE_COLOR = {  # BGR
    LightState.NORMAL: (80, 200, 80),
    LightState.DIMMED: (0, 200, 255),
    LightState.DARK: (60, 60, 255),
    LightState.RECOVERY: (255, 180, 60),
}
GRAPH_HEIGHT = 120


class History:
    """Rolling window of (t, ratio, gain) for the intensity graph."""

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self._data: deque[tuple[float, float, float]] = deque()

    def add(self, t: float, ratio: float, gain: float) -> None:
        self._data.append((t, ratio, gain))
        while self._data and t - self._data[0][0] > self._seconds:
            self._data.popleft()

    def __len__(self) -> int:
        return len(self._data)

    def points(self) -> list[tuple[float, float, float]]:
        return list(self._data)


def _panel(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, alpha: float = 0.55) -> None:
    """Darken a rectangle so HUD text stays readable on any background."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 > x0 and y1 > y0:
        roi = img[y0:y1, x0:x1]
        img[y0:y1, x0:x1] = (roi * (1.0 - alpha)).astype(np.uint8)


def _text(img: np.ndarray, s: str, org: tuple[int, int], color=(255, 255, 255), scale=0.5) -> None:
    (tw, th), _ = cv2.getTextSize(s, _FONT, scale, 1)
    _panel(img, org[0] - 4, org[1] - th - 4, org[0] + tw + 4, org[1] + 5)
    cv2.putText(img, s, org, _FONT, scale, color, 1, cv2.LINE_AA)


def draw_graph(history: History, width: int, seconds: float, max_gain: float) -> np.ndarray:
    """Ratio (level/ref, green, 0..1.5) and gain (orange, 1..max_gain) over time."""
    img = np.full((GRAPH_HEIGHT, width, 3), 24, dtype=np.uint8)
    pts = history.points()
    y_ref = int(GRAPH_HEIGHT * (1 - 1.0 / 1.5))
    cv2.line(img, (0, y_ref), (width - 1, y_ref), (90, 90, 90), 1)
    if len(pts) >= 2:
        t_end = pts[-1][0]
        xs = [int(width - 1 - (t_end - p[0]) / seconds * (width - 1)) for p in pts]
        ratio_y = [int((GRAPH_HEIGHT - 1) * (1 - min(max(p[1], 0.0), 1.5) / 1.5)) for p in pts]
        gain_y = [
            int((GRAPH_HEIGHT - 1) * (1 - (min(max(p[2], 1.0), max_gain) - 1) / (max_gain - 1)))
            for p in pts
        ]
        cv2.polylines(
            img, [np.array(list(zip(xs, ratio_y, strict=True)), np.int32)], False, (80, 220, 80), 1
        )
        cv2.polylines(
            img, [np.array(list(zip(xs, gain_y, strict=True)), np.int32)], False, (0, 160, 255), 1
        )
    _text(img, "ratio L/Lref (green)  gain (orange)", (6, 14), scale=0.4)
    return img


def compose(
    original: np.ndarray,
    r: FrameResult,
    history: History,
    fps: float,
    seconds: float,
    max_gain: float,
) -> np.ndarray:
    """Original | compensated, with HUD, plus the intensity graph underneath."""
    left, right = original.copy(), r.frame.copy()
    _text(left, "ORIGINAL", (8, 20))
    _text(right, "COMPENSATED" if r.phase is Phase.RUNNING else "CALIBRATING", (8, 20))

    if r.phase is Phase.CALIBRATING:
        w = left.shape[1]
        cv2.rectangle(left, (8, 30), (w - 8, 42), (200, 200, 200), 1)
        cv2.rectangle(
            left, (8, 30), (8 + int((w - 16) * r.calibration_progress), 42), (80, 200, 80), -1
        )
        _text(left, "Learning normal illumination - keep the lights on", (8, 62), scale=0.45)
    else:
        color = _STATE_COLOR[r.state]
        lines = [
            (f"STATE {r.state.value}", color),
            (
                f"Lref {r.ref_level:.3f}  L {r.level_filtered:.3f}  ratio {r.ratio:.2f}",
                (255, 255, 255),
            ),
            (f"gain x{r.gain:.2f}  gamma {r.gamma:.2f}  clahe {r.clahe_clip:.1f}", (255, 255, 255)),
        ]
        if r.brightness_error_pct == r.brightness_error_pct:  # not NaN
            lines.append((f"output error {r.brightness_error_pct:.1f}%", (255, 255, 255)))
        for i, (txt, col) in enumerate(lines):
            _text(left, txt, (8, 42 + 22 * i), col, scale=0.48)
    _text(right, f"{fps:.0f} fps  {r.latency_ms:.1f} ms", (8, 40), scale=0.48)
    if r.message:
        _text(right, r.message[:60], (8, right.shape[0] - 10), (0, 200, 255), scale=0.45)

    top = np.hstack([left, right])
    graph = draw_graph(history, top.shape[1], seconds, max_gain)
    return np.vstack([top, graph])
