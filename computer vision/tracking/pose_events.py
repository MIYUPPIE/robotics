"""Grab and fall event detection from per-track pose keypoints.

Both detectors are stateful per tracker ID: a grab only fires on the frame a
wrist *enters* a shelf zone (not on every frame it stays inside), and a fall
only fires once a torso stays more horizontal than vertical for
FALL_CONFIRMATION_FRAMES straight -- a single horizontal frame is as likely
to be someone tying a shoelace as an actual fall.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from config import (
    FALL_CONFIRMATION_FRAMES,
    FALL_TORSO_ASPECT_RATIO,
    POSE_KEYPOINT_CONFIDENCE_MIN,
)

# COCO-17 keypoint indices, ultralytics pose output order.
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12


@dataclass(frozen=True)
class GrabEvent:
    track_id: int
    zone_id: str
    frame: int


@dataclass(frozen=True)
class FallEvent:
    track_id: int
    frame: int


def point_in_box(point: tuple[float, float], box: tuple[float, float, float, float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


class GrabDetector:
    """Fires a GrabEvent the frame a tracked person's wrist enters a shelf zone."""

    def __init__(self, shelf_zones: Mapping[str, tuple[float, float, float, float]]):
        self.shelf_zones = shelf_zones
        self._wrist_in_zone: dict[tuple[int, str], bool] = {}

    def update(
        self,
        track_id: int,
        frame_index: int,
        keypoints_xy: np.ndarray,
        keypoints_conf: np.ndarray,
        box_offset: tuple[float, float] = (0.0, 0.0),
    ) -> list[GrabEvent]:
        """Advance one frame for this track and return any newly fired grabs.

        keypoints_xy/keypoints_conf are in the coordinate space of whatever
        crop they were computed on; box_offset shifts them into full-frame
        (and therefore shelf-zone) coordinates.
        """
        events: list[GrabEvent] = []
        offset_x, offset_y = box_offset

        wrist_points: list[tuple[float, float]] = []
        for wrist_index in (LEFT_WRIST, RIGHT_WRIST):
            if keypoints_conf[wrist_index] < POSE_KEYPOINT_CONFIDENCE_MIN:
                continue
            wrist_x, wrist_y = keypoints_xy[wrist_index]
            wrist_points.append((float(wrist_x) + offset_x, float(wrist_y) + offset_y))

        # One aggregate "is any wrist inside this zone" boolean per zone per
        # frame -- checking each wrist against the shared per-zone state
        # independently would let one wrist's "outside" overwrite the
        # other's "inside" and cause spurious re-fires every frame.
        for zone_id, zone_box in self.shelf_zones.items():
            key = (track_id, zone_id)
            was_inside = self._wrist_in_zone.get(key, False)
            is_inside = any(point_in_box(point, zone_box) for point in wrist_points)
            if is_inside and not was_inside:
                events.append(GrabEvent(track_id, zone_id, frame_index))
            self._wrist_in_zone[key] = is_inside
        return events

    def clear_track(self, track_id: int) -> None:
        """Drop state for a track that StrongSort has fully retired."""
        stale_keys = [key for key in self._wrist_in_zone if key[0] == track_id]
        for key in stale_keys:
            del self._wrist_in_zone[key]

    def active_zones(self, track_id: int) -> list[str]:
        """Zone IDs this track currently has a wrist inside, per the last update()."""
        return [
            zone_id
            for (tid, zone_id), inside in self._wrist_in_zone.items()
            if tid == track_id and inside
        ]


class FallDetector:
    """Fires one FallEvent after a torso stays horizontal for the confirmation window."""

    def __init__(self, confirmation_frames: int = FALL_CONFIRMATION_FRAMES):
        self.confirmation_frames = confirmation_frames
        self._consecutive_horizontal: dict[int, int] = {}
        self._already_fired: set[int] = set()

    def update(
        self,
        track_id: int,
        frame_index: int,
        keypoints_xy: np.ndarray,
        keypoints_conf: np.ndarray,
    ) -> FallEvent | None:
        if not self._torso_visible(keypoints_conf):
            self._consecutive_horizontal[track_id] = 0
            return None

        if not self._torso_is_horizontal(keypoints_xy):
            self._consecutive_horizontal[track_id] = 0
            self._already_fired.discard(track_id)
            return None

        count = self._consecutive_horizontal.get(track_id, 0) + 1
        self._consecutive_horizontal[track_id] = count
        if count >= self.confirmation_frames and track_id not in self._already_fired:
            self._already_fired.add(track_id)
            return FallEvent(track_id, frame_index)
        return None

    def clear_track(self, track_id: int) -> None:
        self._consecutive_horizontal.pop(track_id, None)
        self._already_fired.discard(track_id)

    def is_down(self, track_id: int) -> bool:
        """Whether this track is currently in a confirmed-fall state."""
        return track_id in self._already_fired

    @staticmethod
    def _torso_visible(keypoints_conf: np.ndarray) -> bool:
        return bool(
            keypoints_conf[LEFT_SHOULDER] >= POSE_KEYPOINT_CONFIDENCE_MIN
            and keypoints_conf[LEFT_HIP] >= POSE_KEYPOINT_CONFIDENCE_MIN
        )

    @staticmethod
    def _torso_is_horizontal(keypoints_xy: np.ndarray) -> bool:
        shoulder_x, shoulder_y = keypoints_xy[LEFT_SHOULDER]
        hip_x, hip_y = keypoints_xy[LEFT_HIP]
        torso_dx = abs(float(shoulder_x) - float(hip_x))
        torso_dy = abs(float(shoulder_y) - float(hip_y))
        return torso_dx > torso_dy * FALL_TORSO_ASPECT_RATIO
