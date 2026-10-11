"""Per-zone shelf product counting via frame-to-frame class-count deltas.

Requires a fine-tuned detector for the store's own SKU/category classes --
COCO-pretrained weights don't recognize shelf products. `model` is duck
typed: anything with a `.predict(crop, verbose=False) -> [Results]` method
where `Results` exposes `.boxes` (each with `.conf`, `.cls`) and `.names`
(a {class_index: class_name} mapping) works, so tests use a small fake
instead of a real ultralytics model.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from config import SHELF_COUNT_CONFIDENCE_MIN, SHELF_DISPLAY_CONFIDENCE_MIN


@dataclass(frozen=True)
class ShelfCountEvent:
    zone_id: str
    class_name: str
    previous_count: int
    current_count: int
    frame: int

    @property
    def delta(self) -> int:
        return self.current_count - self.previous_count


@dataclass(frozen=True)
class ProductDetection:
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]  # full-frame pixel coords, not crop-local


class ShelfCounter:
    def __init__(self, model: Any, shelf_zones: Mapping[str, tuple[int, int, int, int]]):
        self.model = model
        self.shelf_zones = shelf_zones
        # Absent entry = zone never observed yet, distinct from an observed
        # empty zone ({}) -- the first observation only establishes a
        # baseline, it never emits events (there's nothing to diff against).
        self._last_counts: dict[str, dict[str, int]] = {}
        self._last_detections: dict[str, list[ProductDetection]] = {
            zone_id: [] for zone_id in shelf_zones
        }

    def update(self, frame: Any, frame_index: int) -> list[ShelfCountEvent]:
        events: list[ShelfCountEvent] = []
        for zone_id, (x1, y1, x2, y2) in self.shelf_zones.items():
            crop = frame[y1:y2, x1:x2]
            detections = self._detect_products(crop, offset=(x1, y1))
            self._last_detections[zone_id] = detections

            # Counting only trusts the stricter SHELF_COUNT_CONFIDENCE_MIN bar;
            # detections is already filtered down to the much looser
            # SHELF_DISPLAY_CONFIDENCE_MIN, purely so near-misses stay visible.
            counts: dict[str, int] = {}
            for detection in detections:
                if detection.confidence < SHELF_COUNT_CONFIDENCE_MIN:
                    continue
                counts[detection.class_name] = counts.get(detection.class_name, 0) + 1

            previous = self._last_counts.get(zone_id)
            if previous is not None:
                for class_name in set(previous) | set(counts):
                    previous_count = previous.get(class_name, 0)
                    current_count = counts.get(class_name, 0)
                    if current_count != previous_count:
                        events.append(
                            ShelfCountEvent(
                                zone_id, class_name, previous_count, current_count, frame_index
                            )
                        )
            self._last_counts[zone_id] = counts
        return events

    def last_detections(self, zone_id: str) -> list[ProductDetection]:
        """Raw per-zone detections from the most recent update(), for display.

        Unlike update()'s return value (only class-count *changes*), this is
        every current detection every frame -- what to draw on screen to show
        what the model actually sees right now, not just what changed.
        """
        return self._last_detections.get(zone_id, [])

    def _detect_products(self, crop: Any, offset: tuple[int, int]) -> list[ProductDetection]:
        result = self.model.predict(crop, verbose=False)[0]
        offset_x, offset_y = offset
        detections: list[ProductDetection] = []
        for box in result.boxes:
            confidence = float(box.conf)
            if confidence < SHELF_DISPLAY_CONFIDENCE_MIN:
                continue
            class_name = result.names[int(box.cls)]
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            box_full_frame = (
                int(x1 + offset_x),
                int(y1 + offset_y),
                int(x2 + offset_x),
                int(y2 + offset_y),
            )
            detections.append(ProductDetection(class_name, confidence, box_full_frame))
        return detections
