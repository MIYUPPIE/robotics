"""Assembles training_data/images/<product>/*.jpg + matching .txt labels
into the directory layout ultralytics YOLO training expects, and writes
dataset.yaml.

Run after capture_dataset.py + auto_label.py (and any manual label
corrections) for every product you want in the fine-tuned model. Class
indices are assigned alphabetically, but *only* across products that
actually have at least one labeled image -- an unlabeled product (still
sitting at class 0 in its own folder, see training_common.remap_class_index)
is skipped without consuming a global index, since ultralytics requires
class indices to be a contiguous 0..N-1 range and a skipped product's index
would otherwise leave a gap the next product falls into.
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

import yaml
from training_common import TRAINING_DATA_DIR, discover_products, remap_class_index

VAL_SPLIT = 0.2
RANDOM_SEED = 42


def build_dataset(training_data_dir: Path = TRAINING_DATA_DIR) -> Path:
    images_dir = training_data_dir / "images"
    dataset_dir = training_data_dir / "dataset"

    products = discover_products(images_dir)
    if not products:
        raise FileNotFoundError(
            f"no product folders in {images_dir} -- run capture_dataset.py first"
        )

    labeled_by_product: dict[str, list[Path]] = {}
    for product in products:
        product_dir = images_dir / product
        labeled_images = sorted(
            p for p in product_dir.glob("*.jpg") if p.with_suffix(".txt").exists()
        )
        if labeled_images:
            labeled_by_product[product] = labeled_images
        else:
            print(f"[build_dataset] {product}: no labeled images, skipping")

    if not labeled_by_product:
        raise FileNotFoundError(
            "no product has any labeled images -- run auto_label.py (or label manually) first"
        )

    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(RANDOM_SEED)
    total_labeled = 0
    names: dict[int, str] = {}

    for class_index, product in enumerate(sorted(labeled_by_product)):
        labeled_images = labeled_by_product[product]
        names[class_index] = product

        rng.shuffle(labeled_images)
        split_point = max(1, int(len(labeled_images) * (1 - VAL_SPLIT)))
        val_images = labeled_images[split_point:] or labeled_images[:1]
        splits = {"train": labeled_images[:split_point], "val": val_images}

        for split, images in splits.items():
            for image_path in images:
                dest_stem = f"{product}_{image_path.stem}"
                shutil.copy(image_path, dataset_dir / "images" / split / f"{dest_stem}.jpg")

                label_line = image_path.with_suffix(".txt").read_text()
                remapped = remap_class_index(label_line, class_index)
                (dataset_dir / "labels" / split / f"{dest_stem}.txt").write_text(remapped + "\n")

        total_labeled += len(labeled_images)

    dataset_yaml = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    yaml_path = dataset_dir / "dataset.yaml"
    yaml_path.write_text(yaml.dump(dataset_yaml, sort_keys=False))
    print(f"[build_dataset] {total_labeled} images, {len(names)} products -> {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    build_dataset()
