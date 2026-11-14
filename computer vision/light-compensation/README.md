# light-compensation

Automatic, environment-aware webcam light compensation (Python + OpenCV). It learns
what "normal" room brightness looks like, detects when the lights drop, and restores
the camera output to about the same perceived brightness. Spec: [document.md](document.md).

> It preserves the *appearance of the feed*, not the room's light. In a very dark room
> the sensor may not have the detail to recover. Brightness is image-based luminance,
> not lux.

## Run

Uses the repo's existing `voter-kiosk-cv` conda env (opencv, numpy, matplotlib). No
new dependencies, no sudo, no `v4l2-ctl` (exposure is locked through OpenCV's V4L2 backend).

```bash
cd light-compensation
conda run -n voter-kiosk-cv python main.py            # window: original | compensated | graph
conda run -n voter-kiosk-cv python main.py --recalibrate
conda run -n voter-kiosk-cv python main.py --no-display --max-seconds 30 --snapshot /tmp/s.png
```

Keys: `q`/ESC quit, `r` recalibrate, `s` snapshot. First run: keep the lights on for
`calibration_seconds` (5 s). The reference is saved to `data/reference.json` and reused on
the next start, together with the locked exposure value, so restarting in a dark room
still compensates against the original normal.

If your shell sources ROS, run pytest/python with `PYTHONPATH=` (the root Makefile does).

## How it works

```
frame -> 02 preprocessing (LAB L plane) -> 03 estimator (median of 3x3 cell medians)
      -> 05 detector (NORMAL/DIMMED/DARK/RECOVERY, hysteresis + debounce + uniformity)
      -> 06 controller (G = clip(R^alpha,1,Gmax), smoothed, release-capped)
      -> 07 enhancer (denoise, gamma on L, CLAHE, trim to target) -> 08 monitor + CSV log
```

Measurement is on the raw frame only, so there is no feedback loop through the output.

| Module | File | Notes |
|---|---|---|
| 01 acquisition | `src/camera.py` | auto-exposure converges, then exposure is locked; restored to auto on exit |
| 02 preprocessing | `src/preprocessing.py` | BGR to LAB, downscaled luma plane |
| 03 illumination | `src/illumination.py` | median/percentiles/grid; person-robust `level`; `LevelFilter` |
| 04 reference | `src/calibration.py` | `L_ref = median(L_1..L_n)`, atomic JSON, rejects reference from other camera geometry, optional slow drift (off by default) |
| 05 detector | `src/light_detector.py` | pure FSM; darkening needs most cells to drop, so occlusion is not a light switch |
| 06 controller | `src/controller.py` | gain falls instantly when light returns (no over-bright flash) |
| 07 enhancer | `src/enhancer.py` | colour kept (a/b untouched), gamma floor, CLAHE scaled by gain |
| 08 output | `src/monitor.py`, `src/logger.py` | HUD + graph; `logs/light_history.csv`, `logs/events.csv` |
| glue | `src/pipeline.py` | pure frame-in/frame-out, no camera, no disk |

All tunables live in `src/config.py` (defaults) and `config/settings.json` (overrides).
Unknown keys or inconsistent values are rejected at startup.

## Verify

```bash
make test               # gate tests (repo root); light-compensation adds ~130, ~3 s
make lightcomp-eval     # closed-loop synthetic eval, ~1 min, 21 checks (also in `make eval`)
python tools/plot_history.py                      # logs/history.png from the CSV
```

Eval pass bars: steady brightness error < 15 % (k = 0.7..0.2 of normal light), detection
< 1.5 s, flicker < 0.012, recovery < 4 s without over-bright flash, zero false triggers
(person walking, screen brightness), graceful at extreme dark, < 30 ms/frame at 640x480.
The eval is synthetic (linear-light darkening plus sensor noise). Measured live on this
machine's webcam: 30 fps, ~6 ms/frame.

## Known limits (honest)

- Uneven lighting (spec test 7): only global light changes trigger compensation. Local
  dark regions get CLAHE only once a global state fires. Spatial compensation is v0.2.
- Webcam gain is not exposed by this driver; exposure is locked, gain is not.
- Gamma and CLAHE amplify sensor noise; denoise is mild. Very dark scenes look noisy by nature.
- Thresholds were tuned on synthetic data. Re-tune on real recordings by lowering the
  room light with `lock_exposure` on.
- Deep-learning enhancement (Zero-DCE++) is milestone M7, not built.
