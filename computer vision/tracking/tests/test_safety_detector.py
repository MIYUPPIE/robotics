import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safety_detector import SafetyDetector


class FakeBox:
    def __init__(self, cls: int, conf: float):
        self.cls = cls
        self.conf = conf


class FakeResult:
    def __init__(self, boxes: list[FakeBox], names: dict[int, str]):
        self.boxes = boxes
        self.names = names


class FakeModel:
    def __init__(self, result: FakeResult):
        self._result = result
        self.calls = 0

    def predict(self, frame, conf=0.0, verbose=False):
        self.calls += 1
        return [self._result]


NO_DETECTIONS = FakeResult([], {})


def test_no_detection_on_frames_outside_the_stride_cadence():
    weapon_model = FakeModel(FakeResult([FakeBox(0, 0.9)], {0: "gun"}))
    fire_model = FakeModel(NO_DETECTIONS)
    detector = SafetyDetector(weapon_model, fire_model, frame_stride=5)

    events = detector.update(frame=object(), frame_index=3)

    assert events == []
    assert weapon_model.calls == 0
    assert fire_model.calls == 0


def test_weapon_detection_fires_on_a_strided_frame():
    weapon_model = FakeModel(FakeResult([FakeBox(0, 0.9)], {0: "gun"}))
    fire_model = FakeModel(NO_DETECTIONS)
    detector = SafetyDetector(weapon_model, fire_model, frame_stride=5)

    events = detector.update(frame=object(), frame_index=5)

    assert len(events) == 1
    assert events[0].alert_type == "weapon"
    assert events[0].class_name == "gun"
    assert events[0].confidence == 0.9


def test_fire_detection_fires_on_a_strided_frame():
    weapon_model = FakeModel(NO_DETECTIONS)
    fire_model = FakeModel(FakeResult([FakeBox(0, 0.8)], {0: "smoke"}))
    detector = SafetyDetector(weapon_model, fire_model, frame_stride=5)

    events = detector.update(frame=object(), frame_index=10)

    assert len(events) == 1
    assert events[0].alert_type == "fire"
    assert events[0].class_name == "smoke"


def test_both_weapon_and_fire_detections_combine_in_one_frame():
    weapon_model = FakeModel(FakeResult([FakeBox(0, 0.9)], {0: "knife"}))
    fire_model = FakeModel(FakeResult([FakeBox(0, 0.7)], {0: "fire"}))
    detector = SafetyDetector(weapon_model, fire_model, frame_stride=1)

    events = detector.update(frame=object(), frame_index=0)

    alert_types = {event.alert_type for event in events}
    assert alert_types == {"weapon", "fire"}


def test_frame_zero_is_on_the_cadence():
    weapon_model = FakeModel(FakeResult([FakeBox(0, 0.9)], {0: "gun"}))
    fire_model = FakeModel(NO_DETECTIONS)
    detector = SafetyDetector(weapon_model, fire_model, frame_stride=5)

    events = detector.update(frame=object(), frame_index=0)

    assert len(events) == 1
