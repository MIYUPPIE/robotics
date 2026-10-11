import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pos_feed import PosFeed, PosLineItem


def _read(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def test_starts_with_an_empty_list_written_to_disk(tmp_path):
    output_path = tmp_path / "pos_feed.json"

    PosFeed(output_path)

    assert _read(output_path) == []


def test_record_removal_adds_a_line_item(tmp_path):
    feed = PosFeed(tmp_path / "pos_feed.json")

    feed.record_removal("soda_can", 1)

    assert feed.line_items() == [PosLineItem("soda_can", 1)]


def test_repeated_removals_of_the_same_product_accumulate_quantity(tmp_path):
    feed = PosFeed(tmp_path / "pos_feed.json")

    feed.record_removal("soda_can", 1)
    feed.record_removal("soda_can", 2)

    assert feed.line_items() == [PosLineItem("soda_can", 3)]


def test_aliased_variants_accumulate_under_the_canonical_name(tmp_path):
    aliases = {"red soda can": "soda_can", "blue soda can": "soda_can"}
    feed = PosFeed(tmp_path / "pos_feed.json", aliases=aliases)

    feed.record_removal("red soda can", 1)
    feed.record_removal("blue soda can", 2)

    assert feed.line_items() == [PosLineItem("soda_can", 3)]


def test_different_products_stay_separate_line_items(tmp_path):
    feed = PosFeed(tmp_path / "pos_feed.json")

    feed.record_removal("soda_can", 1)
    feed.record_removal("chips_bag", 2)

    assert feed.line_items() == [
        PosLineItem("chips_bag", 2),
        PosLineItem("soda_can", 1),
    ]


def test_clear_resets_to_an_empty_list(tmp_path):
    output_path = tmp_path / "pos_feed.json"
    feed = PosFeed(output_path)
    feed.record_removal("soda_can", 1)

    feed.clear()

    assert feed.line_items() == []
    assert _read(output_path) == []


def test_disk_content_matches_line_items_after_every_write(tmp_path):
    output_path = tmp_path / "pos_feed.json"
    feed = PosFeed(output_path)

    feed.record_removal("soda_can", 2)

    assert _read(output_path) == [{"product_name": "soda_can", "quantity": 2}]
