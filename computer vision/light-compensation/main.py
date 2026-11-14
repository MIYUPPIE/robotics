#!/usr/bin/env python
"""Automatic environment-aware webcam light compensation.

Run (project env):  conda run -n voter-kiosk-cv python main.py
Keys:  q/ESC quit   r recalibrate   s save snapshot to logs/

Thin shell only: camera, clock, window, disk. All decisions live in src/pipeline.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
from src.calibration import (  # noqa: E402
    ReferenceFileError,
    load_reference,
    load_reference_image,
    save_reference,
    save_reference_image,
)
from src.camera import Camera, CameraError, ExposureState, LatestFrameReader  # noqa: E402
from src.config import SettingsError, load_settings  # noqa: E402
from src.logger import HistoryLogger  # noqa: E402
from src.monitor import History, compose  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--settings", help="path to a settings JSON (default: config/settings.json)")
    p.add_argument("--camera", type=int, help="camera index override")
    p.add_argument("--recalibrate", action="store_true", help="ignore the stored reference")
    p.add_argument("--no-display", action="store_true", help="headless: no window")
    p.add_argument("--max-seconds", type=float, help="exit after this many seconds")
    p.add_argument("--snapshot", help="write the last composite frame to this PNG on exit")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = load_settings(args.settings)
        if args.camera is not None:
            settings = replace(settings, camera_index=args.camera)
            settings.validate()
    except SettingsError as exc:
        print(f"settings error: {exc}", file=sys.stderr)
        return 2

    ref_path = settings.resolve(settings.reference_file)
    reference = None
    ref_image = None
    if not args.recalibrate:
        try:
            reference = load_reference(ref_path, settings)
            if reference is not None:
                ref_image = load_reference_image(ref_path, settings)
                if ref_image is None and settings.enhance_mode == "reference":
                    raise ReferenceFileError(
                        "no stored reference image (needed for reference mode)"
                    )
        except ReferenceFileError as exc:
            print(f"stored reference rejected ({exc}); recalibrating", file=sys.stderr)
            reference, ref_image = None, None

    camera = Camera(settings)
    try:
        camera.open()
    except CameraError as exc:
        print(f"camera error: {exc}", file=sys.stderr)
        return 3

    exposure: ExposureState = camera.lock_exposure(reference.exposure if reference else None)
    print(f"exposure: {exposure.note} (exposure={exposure.exposure}, gain={exposure.gain})")
    if not exposure.locked and settings.lock_exposure:
        print("WARNING: exposure not locked; webcam auto-exposure may mask the light change")

    pipeline = Pipeline(settings, reference, ref_image)
    print(f"mode: {pipeline.mode}")
    reader = LatestFrameReader(camera)
    reader.start()
    pipeline.set_exposure_info(exposure.to_dict())
    history = History(settings.graph_seconds)
    show = not args.no_display
    last_composite = None
    fps, last_t, t0 = 0.0, None, time.monotonic()
    dropped_reads = 0
    banner, banner_until = "", 0.0

    with HistoryLogger(
        settings.resolve(settings.log_file),
        settings.resolve(settings.events_file),
        settings.log_interval_s,
    ) as log:
        log.event(
            0.0, "start", f"reference={'loaded' if reference else 'none'} exposure={exposure.note}"
        )
        try:
            while True:
                frame = reader.get()
                t = time.monotonic() - t0
                if frame is None:
                    dropped_reads += 1
                    if dropped_reads > 5 or reader.failures > 100:
                        print("camera stopped delivering frames", file=sys.stderr)
                        return 4
                    continue
                dropped_reads = 0
                if last_t is not None and t > last_t:
                    inst = 1.0 / (t - last_t)
                    fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
                last_t = t

                r = pipeline.process(frame, t)
                if r.message:
                    banner, banner_until = r.message, t + 8.0
                    print(f"[{t:7.2f}s] {r.message}")
                if banner and t < banner_until:
                    r = replace(r, message=banner)
                if r.calibrated_now and pipeline.reference is not None:
                    save_reference(pipeline.reference, ref_path)
                    if pipeline.reference_image is not None:
                        save_reference_image(pipeline.reference_image, ref_path)
                    print(f"reference saved: level={r.ref_level:.3f} mode={pipeline.mode}")
                if r.transition is not None:
                    print(f"[{t:7.2f}s] {r.transition.old.value} -> {r.transition.new.value}")
                log.log(t, r, fps)
                history.add(t, r.ratio, r.gain)

                if show or args.snapshot:
                    last_composite = compose(
                        frame, r, history, fps, settings.graph_seconds, settings.max_gain
                    )
                if show and last_composite is not None:
                    cv2.imshow(settings.window_title, last_composite)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("r"):
                        pipeline.recalibrate()
                        log.event(t, "recalibrate", "requested by user")
                    if key == ord("s") and last_composite is not None:
                        out = settings.resolve(f"logs/snapshot_{int(time.time())}.png")
                        cv2.imwrite(str(out), last_composite)
                        print(f"snapshot: {out}")
                if args.max_seconds is not None and t >= args.max_seconds:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            if args.snapshot and last_composite is not None:
                cv2.imwrite(args.snapshot, last_composite)
            reader.stop()
            camera.release()
            if show:
                cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
