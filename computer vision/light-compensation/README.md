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

Keys: `q`/ESC quit, `r` recalibrate, `s` snapshot.

**A stored reference is only valid for the lighting it was recorded in.** If the app
starts already DIMMED/DARK on a stored reference it prints and shows
"press r to recalibrate". References older than `max_reference_age_h` (24 h) are discarded.
For experiments with many lights, start with `--recalibrate` under the lighting you call normal. First run: keep the lights on for
`calibration_seconds` (5 s). The reference is saved to `data/reference.json` and reused on
the next start, together with the locked exposure value, so restarting in a dark room
still compensates against the original normal.

If your shell sources ROS, run pytest/python with `PYTHONPATH=` (the root Makefile does).

## How it works

Goal: the output should look like the same scene before the light changed.

**Reference-guided relighting (default).** The camera is fixed, so calibration stores a clean
image of the normal room (mean of ~150 frames in linear light, `data/reference.png`). Each live
frame is compared with it on a coarse 16x12 grid, per colour channel. A robust quadratic surface
(Tukey-weighted least squares, started from the median) is fitted through the grid's log-ratios:
a lamp that went off on one side, a lower overall level or a colour cast become a smooth gain
field, while a person (an outlier under half the cells) is rejected. The live frame is multiplied
by that field in **linear light**, so brightness, uneven lighting and colour all return to the
reference, and all live detail is kept.

Noise is the real limit in the dark, so relighting is followed by: median spike filter, bilateral
filter, motion-guarded temporal filter, then **background-model shrinkage**: where the relit pixel
differs from the stored scene by less than the amplified sensor noise (2 sigma, sigma tracked from
frame differences), the stored value is used; larger differences (people, moved objects) pass
through. Honest consequence: in very dark rooms fine static detail comes from the stored image,
not the live sensor. Set `background_shrink_kappa` to 0 to turn that off.

The correction is gated (none within 3 % of the reference, full beyond 12 %, smooth between), eases
in over 0.3 s and is released instantly when light returns, so there is no over-bright flash.

`enhance_mode: "gamma"` keeps the earlier gamma+CLAHE path (also the automatic fallback when no
reference image exists).

```
frame -> 02 preprocessing -> 03 estimator (level, grid) -> 05 detector (events/HUD state)
      -> 07b reference relighter (field fit, gate, denoise, shrink)   [default]
      -> 07  gamma + CLAHE enhancer                                   [fallback]
      -> 08 monitor + CSV log
```

| Module | File | Notes |
|---|---|---|
| 01 acquisition | `src/camera.py` | exposure lock, restored on exit; `LatestFrameReader` thread keeps only the newest frame |
| 02 preprocessing | `src/preprocessing.py` | small-first L plane, lazy full LAB |
| 03 illumination | `src/illumination.py` | robust level (median of cell medians), `LevelFilter` |
| 04 reference | `src/calibration.py` | `L_ref` median, mean reference image, atomic JSON + PNG, rejects other camera geometry, 24 h expiry |
| 05 detector | `src/light_detector.py` | pure FSM with hysteresis and debounce, drives events and the HUD |
| 06 controller | `src/controller.py` | scalar gain for the gamma path |
| 07 enhancer | `src/enhancer.py` | gamma + CLAHE path |
| 07b relighter | `src/reference_enhancer.py`, `src/linear_light.py` | field estimator/controller, relight, shrink |
| 08 output | `src/monitor.py`, `src/logger.py` | HUD + graph; `logs/light_history.csv`, `logs/events.csv` |
| glue | `src/pipeline.py` | pure frame-in/frame-out |

Libraries: OpenCV (resize, medianBlur, bilateralFilter, LUT, SIMD arithmetic) and NumPy. No new
runtime dependency. The eval uses scikit-image SSIM (already in the env). Considered and not used:
Zero-DCE++ (generic low-light enhancement, no notion of "same as before", needs PyTorch which is
not in the env) and Retinex/quotient-image methods (halo around a person; the robust grid fit avoids
that).

## Verify

```bash
make test               # gate tests (repo root); light-compensation adds ~130, ~3 s
make lightcomp-eval     # closed-loop synthetic eval, ~1.5 min, 24 checks (also in `make eval`)
python light-compensation/evals/fidelity_eval.py  # "same as before the light?" dE/SSIM, 17 checks
python tools/plot_history.py                      # logs/history.png from the CSV
```

**Fidelity eval** (compensated frame vs the clean normal-light scene, CIELAB dE, ~2.3 is barely
visible): uniform dimming to 5 % light dE 0.97 to 1.7 (gamma path: 10 to 15), SSIM 0.87 to 0.97;
left side of the room dark dE 1.7 (12.4); warm lamp swapped for cool light dE 0.97 (14.6);
dark room plus a person covering 40 % of the width dE 9.3 (16.0, the person is noisy).

Control eval pass bars: steady brightness error < 15 % (k = 0.7..0.2 of normal light), detection
< 1.5 s, flicker < 0.012, recovery < 4 s without over-bright flash, zero false triggers
(person walking, screen brightness), graceful at extreme dark, < 33 ms/frame at 640x480 (30 fps budget).
The eval is synthetic (linear-light darkening plus sensor noise). Measured live on this
machine's webcam: ~6 ms/frame at normal light; relighting is ~30 ms/frame at 640x480 while active, so the camera runs on a reader thread (always the newest frame).

## Known limits (honest)

- The reference is a picture of one room at one camera pose. Move the camera or rearrange the
  room and press `r`. The relighter assumes the lighting changes smoothly (lamps), not that
  objects moved.
- Webcam gain is not exposed by this driver; exposure is locked, gain is not.
- Gamma and CLAHE amplify sensor noise. Mitigated by bilateral + chroma blur and a motion-guarded
  temporal filter (about 46 % less frame-to-frame noise in the eval), but very dark scenes still
  look noisy. Linear gain cap is 100 (`field_max_gain`).
- Thresholds were tuned on synthetic data. Re-tune on real recordings by lowering the
  room light with `lock_exposure` on.
- Deep-learning enhancement (Zero-DCE++) is milestone M7, not built.
