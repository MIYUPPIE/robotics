"""Webcam capture helper for building a custom shelf-product training set.

Usage:
    python capture_dataset.py "soda can" --count 40

Captures photos ~0.5s apart into training_data/images/<product_name>/.
Move the item between shots -- rotate it, change the background, vary
distance and lighting. Variety across the raw photos matters far more than
sheer photo count for a small custom dataset; 30-50 varied photos beats
200 near-identical ones.

Next steps after capturing: auto_label.py, then build_dataset.py, then
train_shelf_model.py -- see README.md.
"""
from __future__ import annotations

import argparse
import time

import cv2
from training_common import TRAINING_DATA_DIR

DEFAULT_CAPTURE_COUNT = 40
CAPTURE_INTERVAL_SECONDS = 0.5


def capture(product_name: str, count: int) -> int:
    """Captures `count` photos from the webcam into training_data/images/<product_name>/.

    Returns how many were actually saved (camera dropout can end this early).
    """
    out_dir = TRAINING_DATA_DIR / "images" / product_name
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*.jpg")))

    cap = cv2.VideoCapture(0)
    saved = 0
    try:
        print(f"Capturing {count} photos of '{product_name}' -- move/rotate the item between shots")
        for i in range(count):
            ok, frame = cap.read()
            if not ok:
                print("camera read failed, stopping early")
                break
            path = out_dir / f"{product_name}_{existing + i:04d}.jpg"
            cv2.imwrite(str(path), frame)
            saved += 1
            print(f"  [{i + 1}/{count}] saved {path.name}")
            time.sleep(CAPTURE_INTERVAL_SECONDS)
    finally:
        cap.release()
    print(f"done -- {saved} photos in {out_dir}")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("product_name", help='e.g. "soda can" -- becomes the folder/class name')
    parser.add_argument("--count", type=int, default=DEFAULT_CAPTURE_COUNT)
    args = parser.parse_args()
    capture(args.product_name, args.count)
