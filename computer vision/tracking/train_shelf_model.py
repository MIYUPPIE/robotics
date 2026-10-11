"""Fine-tunes a YOLO26 detector on the custom dataset assembled by
build_dataset.py, then copies the result to config.FINE_TUNED_SHELF_WEIGHTS
(models/shelf_products.pt). tracker1.py picks it up automatically over the
zero-shot fallback the moment it exists there -- no other code changes needed.

Usage:
    python train_shelf_model.py [--epochs 60] [--imgsz 640]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from build_dataset import build_dataset
from config import FINE_TUNED_SHELF_WEIGHTS
from ultralytics import YOLO

DEFAULT_EPOCHS = 60
DEFAULT_IMAGE_SIZE = 640


def train(epochs: int = DEFAULT_EPOCHS, imgsz: int = DEFAULT_IMAGE_SIZE) -> Path:
    dataset_yaml = build_dataset()

    model = YOLO("yolo26n.pt")  # start from COCO pretrained, fine-tune from there
    results = model.train(data=str(dataset_yaml), epochs=epochs, imgsz=imgsz)

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    FINE_TUNED_SHELF_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_weights, FINE_TUNED_SHELF_WEIGHTS)
    print(f"[train_shelf_model] fine-tuned weights ready at {FINE_TUNED_SHELF_WEIGHTS}")
    return FINE_TUNED_SHELF_WEIGHTS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE)
    args = parser.parse_args()
    train(epochs=args.epochs, imgsz=args.imgsz)
