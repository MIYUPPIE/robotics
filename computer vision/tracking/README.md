# In-store CV tracking prototype

Person detection + persistent re-identification, pose-based grab/fall
detection, shelf shrinkage detection, weapon/fire safety detection, and
MQTT alerting to zone ESP8266 nodes. All built on the YOLO26 model family
(detect + pose share one framework) plus StrongSORT++ for tracking.

This is a **separate prototype from the voter-kiosk-cv application** that
otherwise occupies this git repo (`computer vision/`) -- it shares no code,
dependencies, or purpose with the kiosk. It's self-contained on purpose (own
`requirements.txt`, own `pytest.ini`, no imports from `contracts/`,
`storage/`, or any other kiosk module) so it can be lifted into its own repo
later without untangling anything. Consider giving it one before it grows
much further.

## On-screen indicators (`tracker1.py`)

- Cyan outlines -- the configured `config.SHELF_ZONES`, always drawn, labeled
  by zone ID.
- Green box -- a tracked person, no active grab or fall.
- Orange box, label `ID N GRAB:zone_1` -- that person currently has a wrist
  inside `zone_1`. Clears the instant the wrist leaves.
- Red box, label `ID N FALL` -- that person's torso has been horizontal for
  `FALL_CONFIRMATION_FRAMES` straight. Clears once they stand back up.
- Magenta box, label `<class> <confidence>` -- the shelf model detected that
  product at or above `SHELF_COUNT_CONFIDENCE_MIN` (it counts).
- Gray box, same label -- detected, but below the counting bar (a near-miss:
  visible so you can tell "not confident enough" apart from "not seen at
  all," but it won't trigger a count change or a theft event).

These reflect live detector state every frame (via `GrabDetector.active_zones()`
/ `FallDetector.is_down()`), not a one-shot flash -- so if you don't see a
color change, either your hand didn't register as inside a drawn zone, or the
pose model didn't get a confident wrist/torso keypoint that frame (occlusion,
low light, motion blur). Successful alerts are also logged to the terminal
(`published alert: ...`), independent of the broker being connected, so you
can watch there too.

## What works today (no training required)

- **Person detection + tracking** -- `tracker1.py`, COCO-pretrained
  `yolo26n.pt`, `person` class only.
- **Persistent re-identification** -- `identity_gallery.py`. StrongSort
  drops a track's ID once it's been unseen past `max_age`; `IdentityGallery`
  re-links a freshly minted tracker ID back to the same person via cosine
  similarity on the ReID embedding.
- **Grab + fall detection** -- `pose_events.py`, COCO-pretrained
  `yolo26n-pose.pt` run on each tracked person's crop.
  - `GrabDetector` fires once per zone-entry (not every frame a wrist stays
    inside a shelf zone).
  - `FallDetector` requires the torso to stay more horizontal than vertical
    for `FALL_CONFIRMATION_FRAMES` straight before firing, so a single
    horizontal frame (tying a shoelace) doesn't trigger it.
- **Theft fusion** -- `theft_fusion.py`. Attributes a shelf count drop to
  whoever grabbed in that zone shortly before; an unmatched drop still
  surfaces as an unattributed event for human review instead of being
  dropped.
- **Shelf product counting** (`shelf_counter.py` + `tracker1._load_shelf_model`)
  -- open-vocabulary detection via YOLOE, prompted with `config.TRACKED_PRODUCTS`.
  No labeled dataset or fine-tuning needed; see "Adding products" below.
- **Product grouping** (`product_grouping.py`) -- rolls up visually-distinct
  SKU variants into one reported name via `config.PRODUCT_ALIASES`, without
  weakening detection precision. See "Grouping variants" below.
- **POS feed** (`pos_feed.py`) -- writes every shelf removal out as a
  running, POS-ready item list. See "POS feed" below.
- **MQTT alerting** -- `alerting.py`. Routes by `AlertType`: weapon/fire to
  `store/alerts/critical/{zone}`, fall to `store/alerts/staff`, theft to
  `store/alerts/review/{zone}`.

### How shelf counting tracks "previous" product counts

`ShelfCounter.update()` runs the shelf model on each configured zone's crop
every frame, tallies detections per class name (dropping anything below
`SHELF_COUNT_CONFIDENCE_MIN`), and compares that tally to the *previous
frame's* tally for that same zone (`self._last_counts[zone_id]`, a plain
dict updated every call). Any class whose count changed between the two
emits a `ShelfCountEvent` with the before/after counts; `TheftFusion` only
acts on ones where the count went down. A zone's very first observation only
records the baseline -- it never emits an event, since there's nothing to
diff against yet (otherwise every restart would report every item already
on the shelf as a "sudden increase").

### Adding products to track

Two paths. `tracker1.py` always prefers the fine-tuned one the moment it
exists (`_load_shelf_model` in tracker1.py checks `config.FINE_TUNED_SHELF_WEIGHTS`
first), falling back to zero-shot automatically -- so starting with the fast
path and upgrading to a custom one later needs zero code changes.

#### Fast path: zero-shot, no training (works today)

Edit `config.TRACKED_PRODUCTS` -- a plain list of text descriptions fed to
YOLOE's `set_classes()`:

```python
TRACKED_PRODUCTS: list[str] = [
    "soda can",
    "water bottle",
    "chips bag",
    "energy drink can",  # just add a line
]
```

No photos, no labeling, no training run. More specific descriptions ("red
soda can" vs. bare "can") usually match better, since YOLOE is matching
against a text embedding, not a fixed class ID -- but zero-shot accuracy on
an unusual product (odd label design, bad lighting, an off-angle bottle) can
still be inconsistent. That's what the fine-tuned path below is for.

#### Custom path: your own product, image labeling + transfer learning

For a specific product zero-shot doesn't recognize reliably, fine-tune a
real detector on photos of exactly that item. Four scripts, one pipeline:

```bash
# 1. Capture ~40 varied photos of the product (rotate it, vary background/lighting)
python capture_dataset.py "my custom product" --count 40

# 2. Bootstrap-label them using the zero-shot detector already wired up above
#    (review/fix results in a tool like labelImg if any look wrong -- always
#    label as class 0 there, see auto_label.py's docstring for why)
python auto_label.py "my custom product"

# 3. Assemble the YOLO training layout + dataset.yaml from every labeled product
python build_dataset.py

# 4. Fine-tune (default 60 epochs; takes a while, uses your GPU if you have one)
python train_shelf_model.py
```

Step 4 copies the result to `models/shelf_products.pt` automatically --
restart `tracker1.py` and it's live, no config edits needed. Repeat steps
1-2 for each additional custom product before step 3, and re-run 3-4 to
retrain on the combined set.

Other ways to get a custom detector, if you'd rather not use this pipeline:
- **Manual labeling from scratch** in a tool like [labelImg](https://github.com/HumanSignal/labelImg)
  or [CVAT](https://github.com/cvat-ai/cvat) instead of auto_label.py's
  bootstrap step -- more accurate on a hard product, much slower per photo.
- **A hosted labeling/training service** (Roboflow, Google AutoML Vision,
  etc.) -- easier UI, but photos leave your machine and most have a cost
  past a free tier.
- **More zero-shot prompt tuning first** -- sometimes a better-worded
  `TRACKED_PRODUCTS` entry (or a generic fallback term, see the ones already
  in that list) fixes it without any of the above.

### Grouping variants for reporting

Detection stays precise on purpose -- every flavor/size of a product is its
own `TRACKED_PRODUCTS` prompt (zero-shot path) or its own
`training_data/images/<name>/` folder and trained class (custom path),
because collapsing them at detection time would make the model worse, not
simpler. If several of those should still count as *one* line item for
reporting -- "classify as one" -- add the mapping to `config.PRODUCT_ALIASES`:

```python
PRODUCT_ALIASES: dict[str, str] = {
    "red soda can": "soda_can",
    "blue soda can": "soda_can",
    "diet soda can": "soda_can",
}
```

Any class left out reports under its own name, unchanged -- aliasing is
opt-in per class, not required. This only affects the POS feed below;
`ShelfCounter`/`TheftFusion` still diff and attribute per exact detected
class, since precision matters more there than for a human-facing report.

### POS feed

Every shelf removal (theft-fusion event, attributed or not) is written to
`config.POS_FEED_PATH` (`pos_feed.json` by default) as a running,
grouped-by-canonical-name item list:

```json
[
  { "product_name": "soda_can", "quantity": 3 },
  { "product_name": "chips_bag", "quantity": 1 }
]
```

This is intentionally generic rather than tied to one POS platform -- point
a real POS's file-import, or a small watcher script, at that path, or
replace `PosFeed._write()` in `pos_feed.py` with a direct API call once a
specific platform (Square, Shopify, Clover, an in-house system, ...) is
chosen. `PosFeed.clear()` starts a fresh list -- call it once a transaction
(e.g. a checkout) closes, wherever that decision gets made.

## What still needs fine-tuned weights (not built here)

**Weapon detection** and **fire/smoke detection** (`safety_detector.py`)
still need their own fine-tuned detectors and labeled datasets -- the same
"prompt YOLOE by name instead" trick used for shelf products above would
work here too, but weapon/fire false positives drive safety-critical alerts
(`store/alerts/critical/{zone}`), so that's a deliberate call worth making
on purpose rather than defaulting to zero-shot silently. Ask if you want it
wired up the same way.

Neither `models/weapons.pt` nor `models/fire_smoke.pt` exists yet.
`tracker1.py` checks for each on disk at startup and skips the corresponding
detector if it's missing, so the rest of the pipeline (tracking, grab/fall
events, shelf counting, alerting) still runs without them. Drop trained
weights in `models/` once you have them and no code changes are needed.

## Placeholders that need real values before trusting live alerts

- `config.SHELF_ZONES` -- pixel-space shelf zone boxes. The two zones
  shipped here are stand-ins; calibrate them to the real camera's field of
  view, or grab/theft events will fire against the wrong shelves.
- `config.MQTT_BROKER_HOST` / `MQTT_BROKER_PORT` -- point these at whatever
  broker the physical ESP8266 zone nodes actually subscribe to.

## Setup

This prototype has its own dependencies in a dedicated conda env named
`tracking-cv`, separate from the voter-kiosk-cv env (`ultralytics`/`torch`/
`boxmot` are heavy and unrelated to that project). Used conda rather than
`venv` here because `python3-venv` isn't installed system-wide and installing
it needs `sudo`; conda doesn't:

```bash
conda create -n tracking-cv python=3.11 -y
conda activate tracking-cv
pip install -r requirements.txt
```

Optional, for alerts to actually reach a broker instead of just being
logged: `sudo apt install mosquitto` (needs sudo, one-time; `run.sh` starts
it for you afterward without needing sudo again).

## Running the tests

```bash
cd tracking
pytest
```

Gate tests only -- deterministic, no camera/GPU/broker required. Every
model-shaped dependency (`ShelfCounter.model`, `SafetyDetector`'s two
models, `AlertPublisher.client`) is duck-typed and injected, so tests use
small fakes instead of real ultralytics/MQTT.

## Running it live (all at once)

```bash
./run.sh
```

One command: starts a local mosquitto broker if it's installed and not
already running, then launches `tracker1.py` -- person tracking, grab/fall
detection, and alerting together in one process, reading from the camera at
index 0. `yolo26n.pt` and `yolo26n-pose.pt` download automatically on first
use if not already present locally; StrongSort's ReID weights
(`osnet_x0_25_msmt17.pt`) do the same. Press `q` in the video window to quit.

Verified end-to-end on this machine: camera capture, YOLO26 person
detection, StrongSort tracking on GPU (CUDA), and pose-based grab/fall
detection all ran cleanly for a live 20s smoke test; the only "error" in the
log is the expected one -- an MQTT connection-refused warning when no
broker is running, which is caught and just downgrades alerts to
log-only rather than crashing.

If running `tracker1.py` directly instead of `./run.sh` (skipping the
broker autostart), also export `QT_QPA_PLATFORM=xcb` first if on a native
Wayland session -- pip's `opencv-python` doesn't bundle a Wayland Qt
platform plugin, so `cv2.imshow` can't find one without it.
