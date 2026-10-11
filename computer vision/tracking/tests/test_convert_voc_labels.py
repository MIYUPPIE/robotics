import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from convert_voc_labels import convert_product, voc_xml_to_yolo_lines

VOC_TEMPLATE = """<annotation>
    <size>
        <width>{width}</width>
        <height>{height}</height>
        <depth>3</depth>
    </size>
    {objects}
</annotation>
"""

OBJECT_TEMPLATE = """<object>
        <name>{name}</name>
        <bndbox>
            <xmin>{xmin}</xmin>
            <ymin>{ymin}</ymin>
            <xmax>{xmax}</xmax>
            <ymax>{ymax}</ymax>
        </bndbox>
    </object>"""


def _write_voc_xml(path: Path, width: int, height: int, boxes: list[tuple]) -> None:
    objects = "\n".join(
        OBJECT_TEMPLATE.format(name="whatever", xmin=x1, ymin=y1, xmax=x2, ymax=y2)
        for x1, y1, x2, y2 in boxes
    )
    path.write_text(VOC_TEMPLATE.format(width=width, height=height, objects=objects))


def test_single_box_converts_to_one_normalized_yolo_line(tmp_path):
    xml_path = tmp_path / "photo.xml"
    _write_voc_xml(xml_path, width=640, height=480, boxes=[(137, 235, 267, 411)])

    lines = voc_xml_to_yolo_lines(xml_path)

    assert len(lines) == 1
    class_index, xc, yc, w, h = lines[0].split()
    assert class_index == "0"
    assert float(xc) == pytest.approx((137 + 267) / 2 / 640)
    assert float(yc) == pytest.approx((235 + 411) / 2 / 480)
    assert float(w) == pytest.approx((267 - 137) / 640)
    assert float(h) == pytest.approx((411 - 235) / 480)


def test_multiple_boxes_all_get_class_zero(tmp_path):
    xml_path = tmp_path / "photo.xml"
    _write_voc_xml(xml_path, width=100, height=100, boxes=[(0, 0, 10, 10), (20, 20, 30, 30)])

    lines = voc_xml_to_yolo_lines(xml_path)

    assert len(lines) == 2
    assert all(line.startswith("0 ") for line in lines)


def test_xml_with_no_objects_produces_no_lines(tmp_path):
    xml_path = tmp_path / "photo.xml"
    _write_voc_xml(xml_path, width=100, height=100, boxes=[])

    assert voc_xml_to_yolo_lines(xml_path) == []


def test_convert_product_writes_matching_txt_files(tmp_path):
    images_dir = tmp_path / "images"
    product_dir = images_dir / "widget"
    product_dir.mkdir(parents=True)
    _write_voc_xml(product_dir / "widget_0.xml", 640, 480, [(10, 10, 50, 50)])
    _write_voc_xml(product_dir / "widget_1.xml", 640, 480, [(20, 20, 60, 60)])

    converted, no_objects = convert_product("widget", images_dir)

    assert converted == 2
    assert no_objects == 0
    assert (product_dir / "widget_0.txt").exists()
    assert (product_dir / "widget_1.txt").exists()


def test_convert_product_counts_boxless_xml_separately(tmp_path):
    images_dir = tmp_path / "images"
    product_dir = images_dir / "widget"
    product_dir.mkdir(parents=True)
    _write_voc_xml(product_dir / "widget_0.xml", 640, 480, [(10, 10, 50, 50)])
    _write_voc_xml(product_dir / "widget_1.xml", 640, 480, [])  # saved with no box drawn

    converted, no_objects = convert_product("widget", images_dir)

    assert converted == 1
    assert no_objects == 1
    assert not (product_dir / "widget_1.txt").exists()


def test_convert_product_raises_when_no_xml_files_exist(tmp_path):
    images_dir = tmp_path / "images"
    (images_dir / "widget").mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        convert_product("widget", images_dir)
