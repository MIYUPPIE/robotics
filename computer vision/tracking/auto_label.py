"""Bootstrap-labels captured training photos with the zero-shot YOLOE
detector already wired up for live shelf counting -- turns
training_data/images/<product>/*.jpg into YOLO-format .txt labels
automatically, so a separate manual labeling tool isn't needed for the
common case (one dominant product filling most of the frame).

The product/folder name and the zero-shot *detection prompt* are two
different things and can need different values: the folder name should be
the specific product ("rigg perfume" -- exactly what you want the trained
model to recognize), but CLIP-style zero-shot matching (what YOLOE uses
under set_classes()) doesn't know brand names or unusual product names, only
generic visual descriptions. If the folder name isn't a generic enough
description to match visually, pass --prompt with one that is:

    python auto_label.py "rigg perfume" --prompt "perfume bottle"

Zero-shot boxes are a starting point, not ground truth -- review the result
before training on it. Open training_data/images/<product>/ in a tool like
labelImg (`pip install labelImg`) to fix any wrong or missing boxes; always
label as class 0 there too, build_dataset.py remaps it to the right global
index later.

Usage:
    python auto_label.py "soda can"
    python auto_label.py "rigg perfume" --prompt "perfume bottle"
    python auto_label.py --all   # every product folder under training_data/images/
"""
from __future__ import annotations

import argparse

import cv2
from config import SHELF_PRODUCT_WEIGHTS
from training_common import IMAGES_DIR, discover_products, yolo_label_line
from ultralytics import YOLOE

# Each product's own folder is single-class; see training_common.remap_class_index.
SINGLE_CLASS_INDEX = 0

# Deliberately much looser than the live runtime's SHELF_COUNT_CONFIDENCE_MIN
# (0.4) or SHELF_DISPLAY_CONFIDENCE_MIN (0.1) -- a human reviews every
# bootstrap label before training, so a weak candidate box is still more
# useful than none. Runtime detection stays strict; this doesn't touch it.
BOOTSTRAP_CONFIDENCE_MIN = 0.05


def auto_label_product(
    product_name: str, model: YOLOE, prompt: str | None = None
) -> tuple[int, int]:
    """Labels every captured photo for one product. Returns (labeled, skipped).

    `prompt` is the zero-shot text description fed to the detector -- falls
    back to `product_name` (spaces for underscores) if not given, which
    works fine for already-generic names ("soda can") but not for brand
    names or unusual product names (see module docstring).
    """
    image_dir = IMAGES_DIR / product_name
    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"no photos found in {image_dir} -- run capture_dataset.py first")

    detection_prompt = prompt or product_name.replace("_", " ")
    model.set_classes([detection_prompt], model.get_text_pe([detection_prompt]))

    labeled, skipped = 0, 0
    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"  [auto_label] {image_path.name}: unreadable, skipping")
            skipped += 1
            continue
        height, width = frame.shape[:2]
        result = model.predict(frame, conf=BOOTSTRAP_CONFIDENCE_MIN, verbose=False)[0]
        if len(result.boxes) == 0:
            skipped += 1
            continue
        best_box = max(result.boxes, key=lambda box: float(box.conf))
        x1, y1, x2, y2 = (float(v) for v in best_box.xyxy[0])
        label_line = yolo_label_line(SINGLE_CLASS_INDEX, (x1, y1, x2, y2), width, height)
        image_path.with_suffix(".txt").write_text(label_line + "\n")
        labeled += 1
    return labeled, skipped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "product_name", nargs="?", help='e.g. "soda can" -- matches a capture_dataset.py folder'
    )
    parser.add_argument("--all", action="store_true", help="label every product folder found")
    parser.add_argument(
        "--prompt",
        help='zero-shot detection prompt, if the product name itself won\'t match '
        '(e.g. --prompt "perfume bottle" for a folder named "rigg perfume"). '
        "Ignored with --all -- each product falls back to its own name.",
    )
    args = parser.parse_args()

    if not args.all and not args.product_name:
        parser.error("pass a product name or --all")
    if args.all and args.prompt:
        parser.error("--prompt only applies to a single product, not --all")

    products = discover_products() if args.all else [args.product_name]
    model = YOLOE(SHELF_PRODUCT_WEIGHTS)
    for name in products:
        labeled, skipped = auto_label_product(name, model, prompt=args.prompt)
        print(
            f"{name}: labeled {labeled}, skipped {skipped}"
            " (no confident detection, label manually)"
        )
        if labeled == 0:
            print(
                f"  -> zero detections across all photos. Try --prompt with a more"
                f" generic visual description instead of '{name}'."
            )
