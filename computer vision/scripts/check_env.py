#!/usr/bin/env python
"""Pre-flight environment check: camera, onnxruntime execution provider,
model files, audio cache, admin PIN config. Run before `make run-kiosk`.

Usage: python scripts/check_env.py
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).parent.parent


def check_camera() -> bool:
    try:
        import cv2
    except ImportError:
        print("FAIL: opencv-python not installed")
        return False
    cap = cv2.VideoCapture(0)
    ok = cap.isOpened()
    cap.release()
    print(f"{'OK' if ok else 'FAIL'}: camera (index 0)")
    return ok


def check_onnx_provider() -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        print("FAIL: onnxruntime not installed")
        return False
    providers = ort.get_available_providers()
    gpu = "CUDAExecutionProvider" in providers
    print(
        f"OK: onnxruntime providers={providers} "
        f"(GPU {'available' if gpu else 'not available -- CPU fallback'})"
    )
    return True


def check_model_files() -> bool:
    models_dir = ROOT / "models" / "mediapipe"
    required = ["hand_landmarker.task", "face_landmarker.task"]
    ok = True
    for name in required:
        path = models_dir / name
        exists = path.exists()
        print(f"{'OK' if exists else 'FAIL'}: {path}")
        ok = ok and exists
    if not ok:
        print("  -> run `bash scripts/download_models.sh`")
    return ok


def check_audio_cache() -> bool:
    from voice.cache import verify_complete

    missing = verify_complete()
    ok = not missing
    print(f"{'OK' if ok else 'FAIL'}: audio cache ({len(missing)} missing)")
    if not ok:
        print("  -> run `make prerender-audio` (needs YARNGPT_API_KEY, see .env.example)")
    return ok


def check_admin_pin() -> bool:
    ok = bool(os.environ.get("ADMIN_PIN_HASH"))
    label = "set" if ok else "not set (admin_app will refuse to start)"
    print(f"{'OK' if ok else 'WARN'}: ADMIN_PIN_HASH {label}")
    return True  # warning only -- doesn't block the voting kiosk itself


def main() -> int:
    checks = [
        check_camera(),
        check_onnx_provider(),
        check_model_files(),
        check_audio_cache(),
        check_admin_pin(),
    ]
    if all(checks):
        print("\nAll checks passed.")
        return 0
    print("\nSome checks failed -- see above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
