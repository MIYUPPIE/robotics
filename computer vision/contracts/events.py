"""Cursor + dwell-selection event types shared between hand_ui and kiosk_app.

CursorSample/Region are the input side of the pure dwell state machine
(hand_ui/dwell_fsm.py); the DwellEvent variants are its output side. Keeping
these here (rather than inside hand_ui) lets kiosk_app depend on the *shape*
of these events without depending on hand_ui's MediaPipe-touching internals.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CursorSample:
    """One frame's worth of pointing-cursor state, already in screen coordinates."""

    x: float
    y: float
    t: float  # monotonic seconds
    present: bool  # False when no hand/fingertip is currently tracked


@dataclass(frozen=True)
class Region:
    """An axis-aligned hit-test region (a ballot card, a button, ...)."""

    region_id: str
    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


@dataclass(frozen=True)
class HoverStart:
    region_id: str
    t: float


@dataclass(frozen=True)
class HoverProgress:
    region_id: str
    fraction: float  # 0..1, elapsed / dwell_seconds
    t: float


@dataclass(frozen=True)
class HoverCancel:
    region_id: str
    t: float


@dataclass(frozen=True)
class DwellComplete:
    region_id: str
    t: float


DwellEvent = HoverStart | HoverProgress | HoverCancel | DwellComplete
