import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build_dataset import build_dataset


def _make_labeled_photo(product_dir: Path, stem: str, label: str = "0 0.5 0.5 0.4 0.4") -> None:
    product_dir.mkdir(parents=True, exist_ok=True)
    (product_dir / f"{stem}.jpg").write_bytes(b"not a real jpg, just a placeholder")
    (product_dir / f"{stem}.txt").write_text(label + "\n")


def test_raises_when_no_product_folders_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_dataset(tmp_path)


def test_raises_when_products_exist_but_none_are_labeled(tmp_path):
    (tmp_path / "images" / "soda_can").mkdir(parents=True)
    (tmp_path / "images" / "soda_can" / "soda_can_0.jpg").write_bytes(b"placeholder")

    with pytest.raises(FileNotFoundError):
        build_dataset(tmp_path)


def test_builds_dataset_yaml_with_alphabetical_class_indices(tmp_path):
    images_dir = tmp_path / "images"
    for i in range(3):
        _make_labeled_photo(images_dir / "soda_can", f"soda_can_{i}")
    for i in range(3):
        _make_labeled_photo(images_dir / "chips_bag", f"chips_bag_{i}")

    yaml_path = build_dataset(tmp_path)

    dataset_yaml = yaml.safe_load(yaml_path.read_text())
    assert dataset_yaml["names"] == {0: "chips_bag", 1: "soda_can"}  # alphabetical


def test_copies_images_into_train_and_val_splits(tmp_path):
    images_dir = tmp_path / "images"
    for i in range(5):
        _make_labeled_photo(images_dir / "soda_can", f"soda_can_{i}")

    build_dataset(tmp_path)

    dataset_dir = tmp_path / "dataset"
    train_images = list((dataset_dir / "images" / "train").glob("*.jpg"))
    val_images = list((dataset_dir / "images" / "val").glob("*.jpg"))
    assert len(train_images) + len(val_images) == 5
    assert len(val_images) >= 1  # every product contributes at least one val image


def test_single_labeled_image_still_gets_a_val_copy(tmp_path):
    images_dir = tmp_path / "images"
    _make_labeled_photo(images_dir / "soda_can", "soda_can_0")

    build_dataset(tmp_path)

    dataset_dir = tmp_path / "dataset"
    assert len(list((dataset_dir / "images" / "val").glob("*.jpg"))) == 1


def test_label_class_index_is_remapped_to_the_global_index(tmp_path):
    images_dir = tmp_path / "images"
    _make_labeled_photo(images_dir / "chips_bag", "chips_bag_0", label="0 0.1 0.2 0.3 0.4")
    _make_labeled_photo(images_dir / "soda_can", "soda_can_0", label="0 0.5 0.5 0.2 0.2")

    build_dataset(tmp_path)

    dataset_dir = tmp_path / "dataset"
    all_labels = list((dataset_dir / "labels" / "train").glob("*.txt")) + list(
        (dataset_dir / "labels" / "val").glob("*.txt")
    )
    contents = {p.name: p.read_text().strip() for p in all_labels}
    soda_label = next(v for k, v in contents.items() if "soda_can" in k)
    chips_label = next(v for k, v in contents.items() if "chips_bag" in k)
    # alphabetical order: chips_bag=0, soda_can=1
    assert chips_label.startswith("0 ")
    assert soda_label.startswith("1 ")


def test_unlabeled_product_is_skipped_but_labeled_siblings_still_processed(tmp_path):
    images_dir = tmp_path / "images"
    _make_labeled_photo(images_dir / "soda_can", "soda_can_0")
    (images_dir / "chips_bag").mkdir(parents=True)
    (images_dir / "chips_bag" / "chips_bag_0.jpg").write_bytes(b"placeholder, no label")

    yaml_path = build_dataset(tmp_path)

    dataset_yaml = yaml.safe_load(yaml_path.read_text())
    # chips_bag is unlabeled and alphabetically first, but must NOT leave a
    # gap at index 0 -- ultralytics requires contiguous 0..N-1 indices.
    assert dataset_yaml["names"] == {0: "soda_can"}


def test_class_indices_stay_contiguous_with_an_unlabeled_product_in_the_middle(tmp_path):
    images_dir = tmp_path / "images"
    _make_labeled_photo(images_dir / "apple_juice", "apple_juice_0")
    (images_dir / "chips_bag").mkdir(parents=True)  # alphabetically between the two, unlabeled
    (images_dir / "chips_bag" / "chips_bag_0.jpg").write_bytes(b"placeholder, no label")
    _make_labeled_photo(images_dir / "soda_can", "soda_can_0")

    yaml_path = build_dataset(tmp_path)

    dataset_yaml = yaml.safe_load(yaml_path.read_text())
    assert dataset_yaml["names"] == {0: "apple_juice", 1: "soda_can"}
    assert set(dataset_yaml["names"]) == {0, 1}  # contiguous, no gap left by chips_bag
