import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training_common import discover_products, remap_class_index, yolo_label_line


def test_discover_products_returns_empty_list_when_images_dir_missing(tmp_path):
    assert discover_products(tmp_path / "does_not_exist") == []


def test_discover_products_lists_subfolder_names_alphabetically(tmp_path):
    (tmp_path / "soda_can").mkdir()
    (tmp_path / "chips_bag").mkdir()
    (tmp_path / "water_bottle").mkdir()

    assert discover_products(tmp_path) == ["chips_bag", "soda_can", "water_bottle"]


def test_discover_products_ignores_non_directory_entries(tmp_path):
    (tmp_path / "soda_can").mkdir()
    (tmp_path / "notes.txt").write_text("not a product folder")

    assert discover_products(tmp_path) == ["soda_can"]


def test_remap_class_index_replaces_leading_index_only():
    line = "0 0.512345 0.612345 0.234567 0.198765"

    assert remap_class_index(line, 3) == "3 0.512345 0.612345 0.234567 0.198765"


def test_remap_class_index_strips_trailing_newline():
    line = "0 0.5 0.5 0.2 0.2\n"

    assert remap_class_index(line, 1) == "1 0.5 0.5 0.2 0.2"


def test_remap_class_index_works_from_any_original_index():
    line = "7 0.1 0.2 0.3 0.4"

    assert remap_class_index(line, 0) == "0 0.1 0.2 0.3 0.4"


def test_yolo_label_line_box_filling_the_whole_image_normalizes_to_full_extent():
    line = yolo_label_line(0, (0, 0, 100, 100), image_width=100, image_height=100)

    class_index, x_center, y_center, width, height = line.split()
    assert class_index == "0"
    assert float(x_center) == 0.5
    assert float(y_center) == 0.5
    assert float(width) == 1.0
    assert float(height) == 1.0


def test_yolo_label_line_box_in_a_corner_normalizes_correctly():
    line = yolo_label_line(0, (0, 0, 50, 50), image_width=200, image_height=100)

    _, x_center, y_center, width, height = line.split()
    assert float(x_center) == 25 / 200
    assert float(y_center) == 25 / 100
    assert float(width) == 50 / 200
    assert float(height) == 50 / 100


def test_yolo_label_line_class_index_is_written_as_given():
    line = yolo_label_line(3, (10, 10, 20, 20), image_width=100, image_height=100)

    assert line.startswith("3 ")


def test_yolo_label_line_output_has_exactly_five_fields():
    line = yolo_label_line(0, (1, 2, 3, 4), image_width=10, image_height=10)

    assert len(line.split()) == 5
