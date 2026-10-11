import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shelf_counter import ShelfCounter

ZONES = {"zone_1": (0, 0, 10, 10), "zone_2": (10, 0, 20, 10)}


class FakeBox:
    def __init__(
        self,
        cls: int,
        conf: float,
        xyxy: tuple[float, float, float, float] = (1, 2, 3, 4),
    ):
        self.cls = cls
        self.conf = conf
        self.xyxy = [xyxy]


class FakeResult:
    def __init__(self, boxes: list[FakeBox], names: dict[int, str]):
        self.boxes = boxes
        self.names = names


class FakeModel:
    """Returns a pre-scripted sequence of results, one per .predict() call."""

    def __init__(self, results_by_call: list[FakeResult]):
        self._results = iter(results_by_call)

    def predict(self, crop, verbose=False):
        return [next(self._results)]


class FakeFrame:
    """Minimal stand-in for a numpy frame -- only needs to support slicing."""

    def __getitem__(self, key):
        return self


NAMES = {0: "soda_can", 1: "chips"}


def test_first_frame_with_products_emits_no_baseline_events():
    model = FakeModel(
        [
            FakeResult([FakeBox(0, 0.9), FakeBox(0, 0.9)], NAMES),
            FakeResult([], NAMES),
        ]
    )
    counter = ShelfCounter(model, {"zone_1": (0, 0, 10, 10)})

    events = counter.update(FakeFrame(), frame_index=0)

    assert events == []  # first count establishes the baseline, not a delta


def test_count_drop_emits_negative_delta_event():
    model = FakeModel(
        [
            FakeResult([FakeBox(0, 0.9), FakeBox(0, 0.9)], NAMES),  # frame 0: 2 sodas
            FakeResult([FakeBox(0, 0.9)], NAMES),  # frame 1: 1 soda
        ]
    )
    counter = ShelfCounter(model, {"zone_1": (0, 0, 10, 10)})
    counter.update(FakeFrame(), frame_index=0)

    events = counter.update(FakeFrame(), frame_index=1)

    assert len(events) == 1
    assert events[0].class_name == "soda_can"
    assert events[0].previous_count == 2
    assert events[0].current_count == 1
    assert events[0].delta == -1


def test_count_increase_emits_positive_delta_event():
    model = FakeModel(
        [
            FakeResult([FakeBox(0, 0.9)], NAMES),
            FakeResult([FakeBox(0, 0.9), FakeBox(0, 0.9)], NAMES),
        ]
    )
    counter = ShelfCounter(model, {"zone_1": (0, 0, 10, 10)})
    counter.update(FakeFrame(), frame_index=0)

    events = counter.update(FakeFrame(), frame_index=1)

    assert events[0].delta == 1


def test_unchanged_count_emits_no_event():
    model = FakeModel(
        [
            FakeResult([FakeBox(0, 0.9)], NAMES),
            FakeResult([FakeBox(0, 0.9)], NAMES),
        ]
    )
    counter = ShelfCounter(model, {"zone_1": (0, 0, 10, 10)})
    counter.update(FakeFrame(), frame_index=0)

    events = counter.update(FakeFrame(), frame_index=1)

    assert events == []


def test_low_confidence_boxes_are_excluded_from_the_count():
    model = FakeModel(
        [
            FakeResult([FakeBox(0, 0.9), FakeBox(0, 0.1)], NAMES),  # only 1 counts
            FakeResult([FakeBox(0, 0.9)], NAMES),
        ]
    )
    counter = ShelfCounter(model, {"zone_1": (0, 0, 10, 10)})
    counter.update(FakeFrame(), frame_index=0)

    events = counter.update(FakeFrame(), frame_index=1)

    assert events == []  # 1 -> 1, no change once the low-conf box is dropped


def test_a_class_disappearing_entirely_still_emits_a_drop_to_zero():
    model = FakeModel(
        [
            FakeResult([FakeBox(0, 0.9)], NAMES),
            FakeResult([], NAMES),
        ]
    )
    counter = ShelfCounter(model, {"zone_1": (0, 0, 10, 10)})
    counter.update(FakeFrame(), frame_index=0)

    events = counter.update(FakeFrame(), frame_index=1)

    assert len(events) == 1
    assert events[0].current_count == 0
    assert events[0].delta == -1


def test_last_detections_is_empty_before_any_update():
    model = FakeModel([])
    counter = ShelfCounter(model, {"zone_1": (0, 0, 10, 10)})

    assert counter.last_detections("zone_1") == []


def test_last_detections_reflects_current_frame_regardless_of_count_change():
    model = FakeModel(
        [
            FakeResult([FakeBox(0, 0.9)], NAMES),
            FakeResult([FakeBox(0, 0.9)], NAMES),  # unchanged count, no event
        ]
    )
    counter = ShelfCounter(model, {"zone_1": (0, 0, 10, 10)})
    counter.update(FakeFrame(), frame_index=0)

    events = counter.update(FakeFrame(), frame_index=1)

    assert events == []  # no count-change event...
    assert len(counter.last_detections("zone_1")) == 1  # ...but still visible for display


def test_last_detections_excludes_boxes_below_the_display_threshold():
    # 0.05 is below SHELF_DISPLAY_CONFIDENCE_MIN (0.1) -- too weak to even display.
    model = FakeModel([FakeResult([FakeBox(0, 0.9), FakeBox(0, 0.05)], NAMES)])
    counter = ShelfCounter(model, {"zone_1": (0, 0, 10, 10)})

    counter.update(FakeFrame(), frame_index=0)

    assert len(counter.last_detections("zone_1")) == 1


def test_last_detections_includes_a_near_miss_below_the_counting_threshold():
    # 0.2 clears SHELF_DISPLAY_CONFIDENCE_MIN (0.1) but not
    # SHELF_COUNT_CONFIDENCE_MIN (0.4) -- visible, but never counted.
    model = FakeModel(
        [
            FakeResult([FakeBox(0, 0.2)], NAMES),
            FakeResult([FakeBox(0, 0.2)], NAMES),
        ]
    )
    counter = ShelfCounter(model, {"zone_1": (0, 0, 10, 10)})
    counter.update(FakeFrame(), frame_index=0)

    events = counter.update(FakeFrame(), frame_index=1)

    assert events == []  # never crosses the counting bar, so never an event
    assert len(counter.last_detections("zone_1")) == 1  # but still visible


def test_last_detections_box_is_offset_into_full_frame_coordinates():
    model = FakeModel([FakeResult([FakeBox(0, 0.9, xyxy=(1, 2, 3, 4))], NAMES)])
    counter = ShelfCounter(model, {"zone_1": (100, 200, 300, 400)})

    counter.update(FakeFrame(), frame_index=0)

    (detection,) = counter.last_detections("zone_1")
    assert detection.box == (101, 202, 103, 204)
    assert detection.class_name == "soda_can"
    assert detection.confidence == 0.9


def test_zones_are_counted_independently():
    model = FakeModel(
        [
            FakeResult([FakeBox(0, 0.9)], NAMES),  # zone_1 frame 0
            FakeResult([FakeBox(1, 0.9)], NAMES),  # zone_2 frame 0
            FakeResult([], NAMES),  # zone_1 frame 1: soda gone
            FakeResult([FakeBox(1, 0.9)], NAMES),  # zone_2 frame 1: unchanged
        ]
    )
    counter = ShelfCounter(model, ZONES)
    counter.update(FakeFrame(), frame_index=0)

    events = counter.update(FakeFrame(), frame_index=1)

    assert len(events) == 1
    assert events[0].zone_id == "zone_1"
