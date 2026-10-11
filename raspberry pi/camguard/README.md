# camguard — buzz GPIO17 when the camera is covered

Sounds an active buzzer on **GPIO17** whenever the ethernet PTZ camera
([../CAMERA.md](../CAMERA.md)) gets covered or blinded.

## How it works
A covered/obscured lens loses spatial detail (blur) and goes uniform. camguard pulls
low-res grayscale frames from the camera substream via ffmpeg (160x120 @ 4fps) and, per
frame, measures:
- **detail** = mean absolute neighbor difference (gradient energy / sharpness)
- **std** = intensity spread

`covered = detail < 16 AND std < 55` (calibrated, see below), debounced so it only
fires after the state holds (`on_delay` 1.5s) and only clears after it holds (`off_delay` 2.0s).
While covered, a background thread beeps the buzzer (0.3s on / 0.3s off).

The buzzer is **active-high**: GPIO17 HIGH = sound, LOW = silent (gpiochip0, line 17, via lgpio).

## Calibration (this camera/scene, 2026-06-29)
| metric | uncovered | covered |
|---|---|---|
| detail | ~25.5 | 1.2–11.7 |
| std    | ~70   | 9–44 |

Thresholds in `calib.json`. Re-measure any time:
```sh
ssh -4 okhub@192.168.1.180
cd ~/camguard
python3 camguard.py calibrate 8     # prints detail/std/brightness stats for current scene
```
Then edit `~/camguard/calib.json` and `systemctl --user restart camguard`.

## Service control (on the Pi)
```sh
export XDG_RUNTIME_DIR=/run/user/1000
systemctl --user status  camguard       # state
journalctl --user -u camguard -f        # live detection log (covered=…, detail=…, std=…)
systemctl --user restart camguard
systemctl --user stop    camguard       # disable the alarm
systemctl --user disable camguard       # stop auto-start on boot
```
Enabled with linger, so it auto-starts on boot independent of the desktop/VNC session.

## Tests
```sh
cd ~/camguard && python3 test_camguard.py     # 9 gate tests: metrics + debounce, no GPIO/network
```

## Manual buzzer check
```sh
python3 camguard.py buzztest    # 3 short beeps
```

## Files
- `camguard.py` — detector + buzzer (modes: `run`, `calibrate`, `buzztest`)
- `test_camguard.py` — deterministic gate tests
- `calib.json` — thresholds + delays
- `camguard.service` — systemd --user unit (installed at `~/.config/systemd/user/`)
