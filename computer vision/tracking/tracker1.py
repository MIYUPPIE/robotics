from pathlib import Path

import config
import cv2
import torch
from alerting import AlertPublisher, AlertType
from boxmot.trackers.registry import create_tracker
from identity_gallery import IdentityGallery
from pos_feed import PosFeed
from pose_events import FallDetector, GrabDetector
from safety_detector import SafetyDetector
from shelf_counter import ShelfCounter
from theft_fusion import TheftFusion
from ultralytics import YOLO, YOLOE

DEVICE = "0" if torch.cuda.is_available() else "cpu"

ZONE_COLOR = (255, 255, 0)
PERSON_COLOR = (0, 255, 0)
GRAB_COLOR = (0, 165, 255)
FALL_COLOR = (0, 0, 255)
PRODUCT_COLOR = (255, 0, 255)
PRODUCT_NEAR_MISS_COLOR = (140, 140, 140)


def _draw_shelf_zones(frame, shelf_zones) -> None:
    """Outline every configured shelf zone so it's visible where to test a grab."""
    for zone_id, (x1, y1, x2, y2) in shelf_zones.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), ZONE_COLOR, 1)
        cv2.putText(frame, zone_id, (x1 + 4, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ZONE_COLOR, 1)


def _draw_shelf_detections(frame, shelf_counter: ShelfCounter, shelf_zones) -> None:
    """Box + label every product the shelf model currently sees, every frame.

    Independent of ShelfCountEvent (which only fires on a count *change*) --
    this shows live detector output so it's visible whether the model sees
    an item at all, not just whether its count moved. Magenta = confident
    enough to actually count (>= SHELF_COUNT_CONFIDENCE_MIN); gray = seen but
    below that bar -- a near-miss that would otherwise be invisible.
    """
    for zone_id in shelf_zones:
        for detection in shelf_counter.last_detections(zone_id):
            counts = detection.confidence >= config.SHELF_COUNT_CONFIDENCE_MIN
            color = PRODUCT_COLOR if counts else PRODUCT_NEAR_MISS_COLOR
            x1, y1, x2, y2 = detection.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{detection.class_name} {detection.confidence:.2f}",
                (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )


def _draw_person(
    frame,
    box: tuple[int, int, int, int],
    person_id: int,
    grab_detector: GrabDetector,
    fall_detector: FallDetector,
) -> None:
    """Box color/label reflect this track's live grab/fall state, not just its ID."""
    x1i, y1i, x2i, y2i = box
    active_zones = grab_detector.active_zones(person_id)
    is_down = fall_detector.is_down(person_id)

    if is_down:
        color, suffix = FALL_COLOR, " FALL"
    elif active_zones:
        color, suffix = GRAB_COLOR, f" GRAB:{','.join(active_zones)}"
    else:
        color, suffix = PERSON_COLOR, ""

    cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, 2)
    cv2.putText(
        frame,
        f"ID {person_id}{suffix}",
        (x1i, max(y1i - 10, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def _load_optional(weights_path: Path) -> YOLO | None:
    """Load a fine-tuned detector if its weights exist, else skip it.

    shelf/weapon/fire weights are fine-tuned on the store's own data and
    don't ship with this prototype -- the rest of the pipeline (person
    tracking, grab/fall events, alerting) still runs without them.
    """
    if not weights_path.exists():
        print(f"[tracker1] {weights_path} not found -- skipping that detector")
        return None
    return YOLO(str(weights_path))


def _load_shelf_model(product_names: list[str]) -> YOLO | YOLOE | None:
    """Fine-tuned detector if one exists (see train_shelf_model.py), else
    open-vocabulary YOLOE prompted with config.TRACKED_PRODUCTS.

    A fine-tuned models/shelf_products.pt wins automatically the moment it
    exists -- faster and more accurate than zero-shot, no config change
    needed to switch over. Skips shelf counting (and therefore theft fusion)
    entirely if neither exists and TRACKED_PRODUCTS is empty.
    """
    if config.FINE_TUNED_SHELF_WEIGHTS.exists():
        print(f"[tracker1] using fine-tuned shelf detector: {config.FINE_TUNED_SHELF_WEIGHTS}")
        return YOLO(str(config.FINE_TUNED_SHELF_WEIGHTS))

    if not product_names:
        print("[tracker1] no fine-tuned shelf detector and config.TRACKED_PRODUCTS is empty"
              " -- skipping shelf counting")
        return None
    print("[tracker1] no fine-tuned shelf detector found -- using zero-shot YOLOE")
    model = YOLOE(config.SHELF_PRODUCT_WEIGHTS)
    model.set_classes(product_names, model.get_text_pe(product_names))
    return model


def _process_pose(
    pose_model: YOLO,
    crop,
    offset: tuple[int, int],
    person_id: int,
    frame_index: int,
    grab_detector: GrabDetector,
    fall_detector: FallDetector,
    theft_fusion: TheftFusion,
    alerts: AlertPublisher,
) -> None:
    pose_result = pose_model.predict(crop, verbose=False)[0]
    if pose_result.keypoints is None or len(pose_result.keypoints) == 0:
        return
    keypoints_xy = pose_result.keypoints.xy[0].cpu().numpy()
    keypoints_conf = pose_result.keypoints.conf[0].cpu().numpy()

    grab_events = grab_detector.update(person_id, frame_index, keypoints_xy, keypoints_conf, offset)
    for grab_event in grab_events:
        theft_fusion.record_grab(grab_event)

    fall_event = fall_detector.update(person_id, frame_index, keypoints_xy, keypoints_conf)
    if fall_event is not None:
        alerts.send_alert(AlertType.FALL, track_id=fall_event.track_id)


def main() -> None:
    person_detector = YOLO(config.PERSON_DETECTOR_WEIGHTS)  # COCO pretrained, person class = 0
    pose_model = YOLO(config.POSE_WEIGHTS)  # COCO pretrained, no fine-tuning needed
    shelf_model = _load_shelf_model(config.TRACKED_PRODUCTS)
    weapon_model = _load_optional(config.WEAPON_WEIGHTS)
    fire_model = _load_optional(config.FIRE_SMOKE_WEIGHTS)

    tracker = create_tracker(
        "strongsort",
        reid_weights=config.REID_WEIGHTS,
        device=DEVICE,
        half=False,
    )
    # StrongSort deletes a track (and its ID) once it goes unseen past max_age, so a
    # person who walks off-screen and back gets a brand new ID with no memory of who
    # they were. This gallery re-links that new ID back to the original one whenever
    # the appearance embedding is a close enough match.
    identity_gallery = IdentityGallery(
        similarity_threshold=config.REID_SIMILARITY_THRESHOLD,
        max_retired_frames=config.REID_GALLERY_MEMORY_FRAMES,
    )
    grab_detector = GrabDetector(config.SHELF_ZONES)
    fall_detector = FallDetector()
    shelf_counter = ShelfCounter(shelf_model, config.SHELF_ZONES) if shelf_model else None
    safety_detector = (
        SafetyDetector(weapon_model, fire_model) if weapon_model and fire_model else None
    )
    theft_fusion = TheftFusion()
    pos_feed = PosFeed(config.POS_FEED_PATH, aliases=config.PRODUCT_ALIASES)
    alerts = AlertPublisher()
    alerts.connect()

    cap = cv2.VideoCapture(0)
    frame_index = 0
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            results = person_detector.predict(frame, classes=[0], verbose=False)[0]  # person only
            dets = results.boxes.data.cpu().numpy()  # [x1,y1,x2,y2,conf,cls]

            tracks = tracker.update(dets, frame)  # [x1,y1,x2,y2,track_id,conf,cls,det_ind]

            if len(tracks):
                embeddings = tracker.model.get_features(tracks.xyxy, frame)
                persistent_ids = identity_gallery.update(tracks.id, embeddings)
            else:
                identity_gallery.update([], [])
                persistent_ids = []

            _draw_shelf_zones(frame, config.SHELF_ZONES)

            for (x1, y1, x2, y2, *_), person_id in zip(tracks, persistent_ids, strict=True):
                x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                crop = frame[y1i:y2i, x1i:x2i]
                if crop.size:
                    _process_pose(
                        pose_model,
                        crop,
                        (x1i, y1i),
                        person_id,
                        frame_index,
                        grab_detector,
                        fall_detector,
                        theft_fusion,
                        alerts,
                    )

                _draw_person(frame, (x1i, y1i, x2i, y2i), person_id, grab_detector, fall_detector)

            if shelf_counter is not None:
                for count_event in shelf_counter.update(frame, frame_index):
                    theft_event = theft_fusion.record_shelf_count(count_event)
                    if theft_event is not None:
                        pos_feed.record_removal(theft_event.class_name, theft_event.quantity)
                        alerts.send_alert(
                            AlertType.THEFT,
                            zone=theft_event.zone_id,
                            class_name=theft_event.class_name,
                            quantity=theft_event.quantity,
                            track_id=theft_event.track_id,
                        )
                _draw_shelf_detections(frame, shelf_counter, config.SHELF_ZONES)

            if safety_detector is not None:
                # Fire/weapon bypass fusion entirely -- these are safety-critical,
                # not inventory-logic-dependent like the theft path above.
                for safety_event in safety_detector.update(frame, frame_index):
                    alerts.send_alert(
                        AlertType(safety_event.alert_type),
                        class_name=safety_event.class_name,
                        confidence=safety_event.confidence,
                    )

            cv2.imshow("tracker", frame)
            frame_index += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
