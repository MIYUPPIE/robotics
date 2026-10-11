"""Weapon and fire/smoke detection -- safety-critical, alerts immediately.

Unlike the theft path (which needs shelf-count + wrist-zone fusion before it
means anything), any weapon or fire/smoke detection above threshold is
actionable on its own, so this has no fusion step: the caller sends an alert
for every SafetyEvent returned. Runs full-frame at a reduced cadence
(SAFETY_DETECTION_FRAME_STRIDE) since, unlike pose events, there's no
per-person crop to narrow the search first.

`weapon_model` / `fire_model` are duck typed the same way as ShelfCounter's
`model` -- see shelf_counter.py's docstring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import (
    FIRE_SMOKE_CONFIDENCE_MIN,
    SAFETY_DETECTION_FRAME_STRIDE,
    WEAPON_CONFIDENCE_MIN,
)


@dataclass(frozen=True)
class SafetyEvent:
    alert_type: str  # "weapon" or "fire"
    class_name: str
    confidence: float
    frame: int


class SafetyDetector:
    def __init__(
        self,
        weapon_model: Any,
        fire_model: Any,
        frame_stride: int = SAFETY_DETECTION_FRAME_STRIDE,
    ):
        self.weapon_model = weapon_model
        self.fire_model = fire_model
        self.frame_stride = frame_stride

    def update(self, frame: Any, frame_index: int) -> list[SafetyEvent]:
        if frame_index % self.frame_stride != 0:
            return []
        events = self._detect(
            self.weapon_model, "weapon", WEAPON_CONFIDENCE_MIN, frame, frame_index
        )
        events += self._detect(
            self.fire_model, "fire", FIRE_SMOKE_CONFIDENCE_MIN, frame, frame_index
        )
        return events

    @staticmethod
    def _detect(
        model: Any, alert_type: str, confidence_min: float, frame: Any, frame_index: int
    ) -> list[SafetyEvent]:
        result = model.predict(frame, conf=confidence_min, verbose=False)[0]
        return [
            SafetyEvent(alert_type, result.names[int(box.cls)], float(box.conf), frame_index)
            for box in result.boxes
        ]
