import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pose_events import (
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_WRIST,
    FallDetector,
    GrabDetector,
    point_in_box,
)

ZONE = {"zone_1": (0.0, 0.0, 100.0, 100.0)}
FULL_CONF = np.ones(17, dtype=np.float32)


OUTSIDE_ANY_ZONE = (-1000.0, -1000.0)


def _keypoints(overrides: dict[int, tuple[float, float]]) -> np.ndarray:
    # Unset keypoints default to well outside the test zone rather than
    # (0, 0), which would coincide with ZONE's own corner and register as a
    # spurious "wrist inside the zone" from whichever keypoint the caller
    # didn't set.
    kpts = np.full((17, 2), OUTSIDE_ANY_ZONE, dtype=np.float32)
    for index, xy in overrides.items():
        kpts[index] = xy
    return kpts


def test_point_in_box_inclusive_bounds():
    assert point_in_box((0.0, 0.0), (0.0, 0.0, 10.0, 10.0))
    assert point_in_box((10.0, 10.0), (0.0, 0.0, 10.0, 10.0))
    assert not point_in_box((10.1, 5.0), (0.0, 0.0, 10.0, 10.0))


def test_grab_fires_once_on_zone_entry_not_every_frame_inside():
    detector = GrabDetector(ZONE)
    inside = _keypoints({RIGHT_WRIST: (50.0, 50.0)})

    first = detector.update(1, frame_index=0, keypoints_xy=inside, keypoints_conf=FULL_CONF)
    second = detector.update(1, frame_index=1, keypoints_xy=inside, keypoints_conf=FULL_CONF)

    assert len(first) == 1 and first[0].zone_id == "zone_1"
    assert second == []


def test_grab_fires_again_after_leaving_and_reentering():
    detector = GrabDetector(ZONE)
    inside = _keypoints({RIGHT_WRIST: (50.0, 50.0)})
    outside = _keypoints({RIGHT_WRIST: (500.0, 500.0)})

    detector.update(1, 0, inside, FULL_CONF)
    detector.update(1, 1, outside, FULL_CONF)
    third = detector.update(1, 2, inside, FULL_CONF)

    assert len(third) == 1


def test_grab_detects_left_wrist_too():
    detector = GrabDetector(ZONE)
    inside = _keypoints({LEFT_WRIST: (50.0, 50.0)})

    events = detector.update(1, 0, inside, FULL_CONF)

    assert len(events) == 1


def test_grab_ignores_low_confidence_wrist():
    detector = GrabDetector(ZONE)
    inside = _keypoints({RIGHT_WRIST: (50.0, 50.0)})
    low_conf = FULL_CONF.copy()
    low_conf[RIGHT_WRIST] = 0.1

    events = detector.update(1, 0, inside, low_conf)

    assert events == []


def test_grab_applies_box_offset_into_full_frame_coordinates():
    detector = GrabDetector(ZONE)
    # Wrist at (5, 5) in crop space; crop is offset to (0, 0) in the frame, so
    # this lands inside zone_1 (0,0,100,100).
    local = _keypoints({RIGHT_WRIST: (5.0, 5.0)})

    events = detector.update(1, 0, local, FULL_CONF, box_offset=(0.0, 0.0))

    assert len(events) == 1


def test_grab_tracks_are_independent_per_person():
    detector = GrabDetector(ZONE)
    inside = _keypoints({RIGHT_WRIST: (50.0, 50.0)})

    events_1 = detector.update(1, 0, inside, FULL_CONF)
    events_2 = detector.update(2, 0, inside, FULL_CONF)

    assert len(events_1) == 1
    assert len(events_2) == 1


def test_clear_track_forgets_grab_state():
    detector = GrabDetector(ZONE)
    inside = _keypoints({RIGHT_WRIST: (50.0, 50.0)})
    detector.update(1, 0, inside, FULL_CONF)

    detector.clear_track(1)
    events = detector.update(1, 1, inside, FULL_CONF)

    assert len(events) == 1  # treated as a fresh entry, not a still-inside frame


def test_active_zones_reports_zone_while_wrist_stays_inside():
    detector = GrabDetector(ZONE)
    inside = _keypoints({RIGHT_WRIST: (50.0, 50.0)})

    detector.update(1, 0, inside, FULL_CONF)

    assert detector.active_zones(1) == ["zone_1"]


def test_active_zones_is_empty_before_any_update():
    detector = GrabDetector(ZONE)

    assert detector.active_zones(1) == []


def test_active_zones_clears_once_wrist_leaves():
    detector = GrabDetector(ZONE)
    inside = _keypoints({RIGHT_WRIST: (50.0, 50.0)})
    outside = _keypoints({RIGHT_WRIST: (500.0, 500.0)})
    detector.update(1, 0, inside, FULL_CONF)

    detector.update(1, 1, outside, FULL_CONF)

    assert detector.active_zones(1) == []


def test_active_zones_does_not_leak_between_tracks():
    detector = GrabDetector(ZONE)
    inside = _keypoints({RIGHT_WRIST: (50.0, 50.0)})
    detector.update(1, 0, inside, FULL_CONF)

    assert detector.active_zones(2) == []


def _torso(shoulder: tuple[float, float], hip: tuple[float, float]) -> np.ndarray:
    return _keypoints({LEFT_SHOULDER: shoulder, LEFT_HIP: hip})


def test_fall_does_not_fire_before_confirmation_window():
    detector = FallDetector(confirmation_frames=5)
    horizontal = _torso(shoulder=(0.0, 50.0), hip=(100.0, 55.0))

    for frame in range(4):
        event = detector.update(1, frame, horizontal, FULL_CONF)
        assert event is None


def test_fall_fires_once_confirmation_window_elapses():
    detector = FallDetector(confirmation_frames=3)
    horizontal = _torso(shoulder=(0.0, 50.0), hip=(100.0, 55.0))

    events = [detector.update(1, frame, horizontal, FULL_CONF) for frame in range(3)]

    assert events[0] is None and events[1] is None
    assert events[2] is not None and events[2].track_id == 1


def test_fall_does_not_refire_every_frame_while_still_down():
    detector = FallDetector(confirmation_frames=3)
    horizontal = _torso(shoulder=(0.0, 50.0), hip=(100.0, 55.0))

    for frame in range(3):
        detector.update(1, frame, horizontal, FULL_CONF)
    again = detector.update(1, 3, horizontal, FULL_CONF)

    assert again is None


def test_fall_streak_resets_when_torso_goes_vertical_again():
    detector = FallDetector(confirmation_frames=3)
    horizontal = _torso(shoulder=(0.0, 50.0), hip=(100.0, 55.0))
    vertical = _torso(shoulder=(50.0, 0.0), hip=(55.0, 100.0))

    detector.update(1, 0, horizontal, FULL_CONF)
    detector.update(1, 1, horizontal, FULL_CONF)
    detector.update(1, 2, vertical, FULL_CONF)  # resets the streak
    event = detector.update(1, 3, horizontal, FULL_CONF)

    assert event is None  # only 1 consecutive horizontal frame so far


def test_fall_can_refire_after_standing_back_up_and_falling_again():
    detector = FallDetector(confirmation_frames=2)
    horizontal = _torso(shoulder=(0.0, 50.0), hip=(100.0, 55.0))
    vertical = _torso(shoulder=(50.0, 0.0), hip=(55.0, 100.0))

    detector.update(1, 0, horizontal, FULL_CONF)
    first_fall = detector.update(1, 1, horizontal, FULL_CONF)
    detector.update(1, 2, vertical, FULL_CONF)
    detector.update(1, 3, horizontal, FULL_CONF)
    second_fall = detector.update(1, 4, horizontal, FULL_CONF)

    assert first_fall is not None
    assert second_fall is not None


def test_fall_ignores_frames_with_low_confidence_torso_keypoints():
    detector = FallDetector(confirmation_frames=2)
    horizontal = _torso(shoulder=(0.0, 50.0), hip=(100.0, 55.0))
    low_conf = FULL_CONF.copy()
    low_conf[LEFT_HIP] = 0.1

    detector.update(1, 0, horizontal, low_conf)
    event = detector.update(1, 1, horizontal, low_conf)

    assert event is None


def test_clear_track_forgets_fall_state():
    detector = FallDetector(confirmation_frames=2)
    horizontal = _torso(shoulder=(0.0, 50.0), hip=(100.0, 55.0))
    detector.update(1, 0, horizontal, FULL_CONF)
    detector.update(1, 1, horizontal, FULL_CONF)  # fires

    detector.clear_track(1)
    detector.update(1, 2, horizontal, FULL_CONF)
    event = detector.update(1, 3, horizontal, FULL_CONF)

    assert event is not None  # confirmation streak restarted cleanly


def test_is_down_false_before_confirmation():
    detector = FallDetector(confirmation_frames=3)
    horizontal = _torso(shoulder=(0.0, 50.0), hip=(100.0, 55.0))

    detector.update(1, 0, horizontal, FULL_CONF)

    assert detector.is_down(1) is False


def test_is_down_true_once_confirmed_and_stays_true_while_still_down():
    detector = FallDetector(confirmation_frames=2)
    horizontal = _torso(shoulder=(0.0, 50.0), hip=(100.0, 55.0))
    detector.update(1, 0, horizontal, FULL_CONF)
    detector.update(1, 1, horizontal, FULL_CONF)  # fires

    detector.update(1, 2, horizontal, FULL_CONF)  # still down, no new event

    assert detector.is_down(1) is True


def test_is_down_clears_once_torso_goes_vertical_again():
    detector = FallDetector(confirmation_frames=2)
    horizontal = _torso(shoulder=(0.0, 50.0), hip=(100.0, 55.0))
    vertical = _torso(shoulder=(50.0, 0.0), hip=(55.0, 100.0))
    detector.update(1, 0, horizontal, FULL_CONF)
    detector.update(1, 1, horizontal, FULL_CONF)  # fires

    detector.update(1, 2, vertical, FULL_CONF)  # stood back up

    assert detector.is_down(1) is False
