"""Attributes a shelf-count drop to whoever grabbed in that zone shortly before.

A ShelfCountEvent with a negative delta means a product left a zone. If a
GrabEvent happened in that same zone within THEFT_FUSION_WINDOW_FRAMES
beforehand, the drop is attributed to that person. If not, it's still
surfaced as an unattributed TheftEvent (track_id=None) for human review --
inventory shrinkage with no attributable grab is a real signal (miscounted
stock, an unseen customer, a shelf restock the wrong way) that shouldn't be
silently dropped just because fusion couldn't tie it to a tracked person.
"""
from __future__ import annotations

from dataclasses import dataclass

from config import THEFT_FUSION_WINDOW_FRAMES
from pose_events import GrabEvent
from shelf_counter import ShelfCountEvent


@dataclass(frozen=True)
class TheftEvent:
    zone_id: str
    class_name: str
    quantity: int
    frame: int
    track_id: int | None  # None => unattributed, needs human review


class TheftFusion:
    def __init__(self, fusion_window_frames: int = THEFT_FUSION_WINDOW_FRAMES):
        self.fusion_window_frames = fusion_window_frames
        self._recent_grabs: list[GrabEvent] = []

    def record_grab(self, event: GrabEvent) -> None:
        self._recent_grabs.append(event)

    def record_shelf_count(self, event: ShelfCountEvent) -> TheftEvent | None:
        self._expire_stale_grabs(event.frame)
        if event.delta >= 0:
            return None  # restock or no change, not a removal

        return TheftEvent(
            zone_id=event.zone_id,
            class_name=event.class_name,
            quantity=-event.delta,
            frame=event.frame,
            track_id=self._match_grab(event),
        )

    def _match_grab(self, event: ShelfCountEvent) -> int | None:
        for grab in reversed(self._recent_grabs):
            within_window = event.frame - grab.frame <= self.fusion_window_frames
            if grab.zone_id == event.zone_id and within_window:
                return grab.track_id
        return None

    def _expire_stale_grabs(self, current_frame: int) -> None:
        self._recent_grabs = [
            grab
            for grab in self._recent_grabs
            if current_frame - grab.frame <= self.fusion_window_frames
        ]
