"""Shared paths and helpers for the capture -> label -> train custom-product
pipeline (capture_dataset.py, auto_label.py, build_dataset.py,
train_shelf_model.py).
"""
from __future__ import annotations

from pathlib import Path

TRAINING_DATA_DIR = Path(__file__).resolve().parent / "training_data"
IMAGES_DIR = TRAINING_DATA_DIR / "images"


def discover_products(images_dir: Path = IMAGES_DIR) -> list[str]:
    """Product names come from training_data/images/ subfolder names, sorted
    alphabetically for a stable class ordering shared by every pipeline step.
    """
    if not images_dir.exists():
        return []
    return sorted(p.name for p in images_dir.iterdir() if p.is_dir())


def remap_class_index(label_line: str, class_index: int) -> str:
    """Replaces a YOLO-format label line's leading class index, keeping the
    box coordinates unchanged.

    auto_label.py and manual labeling in a tool like labelImg both always
    write class 0 (each product lives in its own folder, so within that
    folder there's only ever one class). build_dataset.py uses this to remap
    0 -> that product's actual global class index when assembling the final
    dataset -- so re-labeling never breaks if products are added later and
    the alphabetical index assignment shifts.
    """
    _, rest = label_line.strip().split(" ", 1)
    return f"{class_index} {rest}"


def yolo_label_line(
    class_index: int,
    box_xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> str:
    """Converts a pixel-space box into a normalized YOLO-format label line."""
    x1, y1, x2, y2 = box_xyxy
    x_center = (x1 + x2) / 2 / image_width
    y_center = (y1 + y2) / 2 / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return f"{class_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
