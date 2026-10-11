"""All tunable thresholds and placeholder deployment config for this prototype.

Every magic number used by more than one call site lives here, not inline --
mirrors the sibling voter-kiosk-cv project's `biometric/thresholds.py`
convention one directory up.
"""
from __future__ import annotations

from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"

# --- Model weights ---
# COCO-pretrained -- ultralytics downloads these automatically on first use
# if the file isn't found locally.
PERSON_DETECTOR_WEIGHTS = "yolo26n.pt"
POSE_WEIGHTS = "yolo26n-pose.pt"
REID_WEIGHTS = Path("osnet_x0_25_msmt17.pt")

# Open-vocabulary detector (YOLOE) for shelf products -- add or remove a
# tracked product by editing TRACKED_PRODUCTS below, no labeled dataset or
# fine-tuning required. This is the fast path to get something tracking
# today, not the ceiling: tracker1.py automatically prefers a real fine-tuned
# detector at FINE_TUNED_SHELF_WEIGHTS the moment one exists (see
# capture_dataset.py / auto_label.py / build_dataset.py / train_shelf_model.py
# for how to produce one), falling back to this zero-shot path only when it
# doesn't. Auto-downloads on first use (~50MB detector + a one-time ~600MB
# text-encoder download for set_classes()).
SHELF_PRODUCT_WEIGHTS = "yoloe-11s-seg.pt"
FINE_TUNED_SHELF_WEIGHTS = MODELS_DIR / "shelf_products.pt"

# The store's tracked SKUs/categories, as plain-English text prompts fed to
# YOLOE.set_classes(). This *is* how you add a new product: append a
# descriptive name here (more specific descriptions match better than bare
# nouns, e.g. "red soda can" over "can"). Empty list disables shelf counting
# and theft fusion entirely, same as a missing fine-tuned weights file used to.
TRACKED_PRODUCTS: list[str] = [
    "soda can",
    "water bottle",
    "chips bag",
    "plastic bottle",  # generic catch-all -- broad terms often match better
    "beverage bottle",  # than an over-specific one for a zero-shot matcher
]

# Fine-tuned on this store's own data -- do NOT exist yet, need a labeled
# dataset first. tracker1.py checks for these on disk at startup and skips
# the corresponding detector if missing, so the rest of the pipeline (person
# tracking, grab/fall events, alerting) still runs without them.
WEAPON_WEIGHTS = MODELS_DIR / "weapons.pt"
FIRE_SMOKE_WEIGHTS = MODELS_DIR / "fire_smoke.pt"

# --- Person re-identification (StrongSort appearance matching) ---
# StrongSort's own matching cascade uses max_cos_dist=0.4 (i.e. similarity
# >= 0.6) to accept an appearance match; reuse that same bar for gallery
# re-identification.
REID_SIMILARITY_THRESHOLD = 0.6
# How long (in frames) a departed person's appearance is remembered before a
# reappearance is treated as a stranger instead of being re-linked. ~30s at 30fps.
REID_GALLERY_MEMORY_FRAMES = 900

# --- Pose keypoint gating ---
# Minimum ultralytics pose keypoint confidence to trust its position at all
# (wrist-in-zone checks, torso-angle checks). Below this the keypoint is
# probably occluded or mis-estimated.
POSE_KEYPOINT_CONFIDENCE_MIN = 0.5

# --- Fall detection ---
# Torso counts as "horizontal" once dx exceeds dy by more than this ratio.
FALL_TORSO_ASPECT_RATIO = 1.0
# Consecutive frames the torso must stay horizontal before a fall alert
# fires. ~1.5s at 30fps -- long enough that tying a shoelace doesn't trigger it.
FALL_CONFIRMATION_FRAMES = 45

# --- Shelf product counting ---
# Counting/theft-fusion only trusts a detection at or above this bar.
SHELF_COUNT_CONFIDENCE_MIN = 0.4
# The on-screen overlay shows anything at or above this much lower bar --
# separate from the counting threshold on purpose, so a near-miss (detected
# but not confident enough to count) is still visible instead of invisible,
# which is exactly what made the "why isn't this being detected" question
# unanswerable before this existed.
SHELF_DISPLAY_CONFIDENCE_MIN = 0.1

# --- Safety detection (weapon / fire+smoke) ---
# Runs full-frame rather than on a person crop, so it's checked at a
# reduced cadence to stay cheap: every Nth frame, not every frame.
SAFETY_DETECTION_FRAME_STRIDE = 5
WEAPON_CONFIDENCE_MIN = 0.5
FIRE_SMOKE_CONFIDENCE_MIN = 0.4

# --- Theft fusion ---
# Max gap (in frames) between a wrist-grab event and a shelf count drop in
# the same zone for the two to be attributed to the same person. ~3s at 30fps.
THEFT_FUSION_WINDOW_FRAMES = 90

# --- Shelf zones (pixel space, per camera) ---
# PLACEHOLDERS. Split left/right across the full 640x480 frame so testing at
# a webcam isn't finicky about exactly where in frame you hold an item --
# calibrate these to the real camera's actual shelf regions before trusting
# grab or theft events for anything other than testing.
SHELF_ZONES: dict[str, tuple[int, int, int, int]] = {
    "zone_1": (0, 0, 320, 480),
    "zone_2": (320, 0, 640, 480),
}

# --- MQTT alerting ---
# PLACEHOLDER broker address -- point this at whatever broker the ESP8266
# zone nodes actually subscribe to before relying on live alerts.
MQTT_BROKER_HOST = "localhost"
MQTT_BROKER_PORT = 1883
MQTT_CLIENT_ID = "store-cv-alerting"

# --- Product grouping (POS-facing) ---
# Maps a raw detected class name to the name it should count/report as on
# the POS feed. This is how visually-distinct variants "classify as one":
# each flavor/size still gets its own TRACKED_PRODUCTS prompt (or its own
# training_data/images/<name>/ folder) so the detector stays precise, but
# they all roll up to one POS line item here. A class left out of this dict
# reports under its own name unchanged -- aliasing is opt-in, not required.
PRODUCT_ALIASES: dict[str, str] = {
    # "red soda can": "soda_can",
    # "blue soda can": "soda_can",
}

# --- POS feed ---
# Where the running "items removed from the shelf" list gets written, as
# JSON, every time a theft/removal event fires. Generic and integration-
# agnostic on purpose -- point a real POS's file-import or a small watcher
# script at this path, or swap PosFeed._write() in pos_feed.py for a direct
# API call once you know which POS platform you're integrating with.
POS_FEED_PATH = Path(__file__).resolve().parent / "pos_feed.json"
