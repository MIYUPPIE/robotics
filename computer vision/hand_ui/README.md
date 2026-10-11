# hand_ui

MediaPipe hand tracking + the pure dwell-to-select state machine that drives every
gesture interaction in `kiosk_app`.

- `dwell_fsm.py` -- `DwellSelector`, the deterministic core. Pure function of
  `(cursor position, timestamp, region geometry)`, zero MediaPipe dependency. This is
  what makes "what counts as a valid selection" fully unit-testable -- see
  [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md#deterministic-core-vs-thin-cv-shell).
- `regions.py` -- pure region layout (`grid_layout`, for the party ballot grid) and
  hit-testing.
- `landmarker.py` -- the thin CV shell: MediaPipe `HandLandmarker` (synchronous VIDEO
  mode) → index-fingertip `CursorSample`. Expects an already-mirrored input frame (see
  [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md#camera-mirroring-is-applied-once-at-capture-everywhere));
  smooths the raw landmark with `OneEuroFilter`, an adaptive filter that suppresses
  jitter while the cursor is nearly still (exactly when a voter is trying to hold it
  steady for a dwell-select) without adding lag while it's moving.
- `renderer.py` -- OpenCV drawing (cursor, radial dwell-progress ring, region boxes).
  Visual output only, not gate-tested.

## Tests vs. evals

Gate tests (`tests/test_dwell_fsm.py`, `tests/test_regions.py`, `tests/test_landmarker.py`)
cover every enter/exit/complete/hand-lost/debounce case with hand-picked synthetic
cursor sequences, the timestamp-monotonicity ratchet, and the `OneEuroFilter`'s
jitter-damping/lag-reduction behavior -- all pure, <2s, no camera.

`evals/gesture_dwell_eval.py` replays broader randomized cursor paths (or real
recorded sessions if present in `evals/data/`, see `evals/data/CONSENT.md`) through
the real `DwellSelector`, measuring correct-intent selection rate and false-trigger
rate. Run: `python -m hand_ui.evals.gesture_dwell_eval`.
