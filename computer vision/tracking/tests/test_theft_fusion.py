import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pose_events import GrabEvent
from shelf_counter import ShelfCountEvent
from theft_fusion import TheftFusion


def _drop(zone_id: str, frame: int, previous: int = 2, current: int = 1) -> ShelfCountEvent:
    return ShelfCountEvent(zone_id, "soda_can", previous, current, frame)


def test_drop_with_no_prior_grab_is_unattributed():
    fusion = TheftFusion(fusion_window_frames=90)

    event = fusion.record_shelf_count(_drop("zone_1", frame=10))

    assert event is not None
    assert event.track_id is None
    assert event.quantity == 1


def test_drop_shortly_after_a_grab_is_attributed_to_that_track():
    fusion = TheftFusion(fusion_window_frames=90)
    fusion.record_grab(GrabEvent(track_id=7, zone_id="zone_1", frame=5))

    event = fusion.record_shelf_count(_drop("zone_1", frame=10))

    assert event.track_id == 7


def test_drop_outside_the_fusion_window_is_unattributed():
    fusion = TheftFusion(fusion_window_frames=10)
    fusion.record_grab(GrabEvent(track_id=7, zone_id="zone_1", frame=0))

    event = fusion.record_shelf_count(_drop("zone_1", frame=50))

    assert event.track_id is None


def test_grab_in_a_different_zone_does_not_match():
    fusion = TheftFusion(fusion_window_frames=90)
    fusion.record_grab(GrabEvent(track_id=7, zone_id="zone_2", frame=5))

    event = fusion.record_shelf_count(_drop("zone_1", frame=10))

    assert event.track_id is None


def test_positive_delta_is_not_a_theft_event():
    fusion = TheftFusion(fusion_window_frames=90)
    fusion.record_grab(GrabEvent(track_id=7, zone_id="zone_1", frame=5))

    event = fusion.record_shelf_count(_drop("zone_1", frame=10, previous=1, current=2))

    assert event is None


def test_zero_delta_is_not_a_theft_event():
    fusion = TheftFusion(fusion_window_frames=90)

    event = fusion.record_shelf_count(_drop("zone_1", frame=10, previous=2, current=2))

    assert event is None


def test_most_recent_matching_grab_is_used_when_several_are_in_window():
    fusion = TheftFusion(fusion_window_frames=90)
    fusion.record_grab(GrabEvent(track_id=1, zone_id="zone_1", frame=1))
    fusion.record_grab(GrabEvent(track_id=2, zone_id="zone_1", frame=8))

    event = fusion.record_shelf_count(_drop("zone_1", frame=10))

    assert event.track_id == 2


def test_multi_unit_drop_reports_correct_quantity():
    fusion = TheftFusion(fusion_window_frames=90)
    fusion.record_grab(GrabEvent(track_id=7, zone_id="zone_1", frame=5))

    event = fusion.record_shelf_count(_drop("zone_1", frame=10, previous=5, current=2))

    assert event.quantity == 3
