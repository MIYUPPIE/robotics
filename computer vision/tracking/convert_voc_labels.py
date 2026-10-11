"""Converts Pascal VOC XML labels (labelImg's default save format) into the
YOLO-format .txt labels this pipeline expects.

labelImg defaults to Pascal VOC on first launch -- the format toggle has to
be switched to YOLO explicitly (see auto_label.py's module docstring). If
that got missed, this converts the existing .xml work instead of requiring
a full relabel.

Usage:
    python convert_voc_labels.py "31-IN-1 ELECTROc SCREWDRIVER SET"
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from training_common import IMAGES_DIR, yolo_label_line

# Matches auto_label.py's convention: one class per product folder, remapped
# to the real global index later by build_dataset.py.
SINGLE_CLASS_INDEX = 0


def _require_text(parent: ET.Element, tag: str, xml_path: Path) -> str:
    child = parent.find(tag)
    if child is None or child.text is None:
        raise ValueError(f"{xml_path}: missing <{tag}> -- not a valid Pascal VOC label")
    return child.text


def voc_xml_to_yolo_lines(xml_path: Path) -> list[str]:
    """Parses one Pascal VOC XML label into YOLO-format label lines, one per
    <object> box. Every box gets SINGLE_CLASS_INDEX -- multiple differently
    -named objects in one XML would all collapse to the same class, so this
    is only correct for a single-object-type folder (this pipeline's
    convention throughout).
    """
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"{xml_path}: missing <size> -- not a valid Pascal VOC label")
    width = int(_require_text(size, "width", xml_path))
    height = int(_require_text(size, "height", xml_path))

    lines = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"{xml_path}: <object> missing <bndbox>")
        xyxy = (
            float(_require_text(box, "xmin", xml_path)),
            float(_require_text(box, "ymin", xml_path)),
            float(_require_text(box, "xmax", xml_path)),
            float(_require_text(box, "ymax", xml_path)),
        )
        lines.append(yolo_label_line(SINGLE_CLASS_INDEX, xyxy, width, height))
    return lines


def convert_product(product_name: str, images_dir: Path = IMAGES_DIR) -> tuple[int, int]:
    """Converts every .xml label in a product's folder to a matching .txt.

    Returns (converted, no_objects) counts.
    """
    image_dir = images_dir / product_name
    xml_paths = sorted(image_dir.glob("*.xml"))
    if not xml_paths:
        raise FileNotFoundError(f"no .xml labels found in {image_dir}")

    converted, no_objects = 0, 0
    for xml_path in xml_paths:
        lines = voc_xml_to_yolo_lines(xml_path)
        if not lines:
            no_objects += 1
            continue
        xml_path.with_suffix(".txt").write_text("\n".join(lines) + "\n")
        converted += 1
    return converted, no_objects


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("product_name", help='matches a capture_dataset.py folder, e.g. "soda can"')
    args = parser.parse_args()

    converted, no_objects = convert_product(args.product_name)
    print(f"{args.product_name}: converted {converted}, {no_objects} had no boxes")
