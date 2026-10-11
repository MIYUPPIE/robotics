import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from product_grouping import canonical_name, group_counts

ALIASES = {"red soda can": "soda_can", "blue soda can": "soda_can"}


def test_unmapped_class_passes_through_unchanged():
    assert canonical_name("chips_bag", ALIASES) == "chips_bag"


def test_mapped_class_returns_its_alias():
    assert canonical_name("red soda can", ALIASES) == "soda_can"


def test_empty_aliases_never_renames_anything():
    assert canonical_name("red soda can", {}) == "red soda can"


def test_group_counts_merges_aliased_variants():
    counts = {"red soda can": 2, "blue soda can": 1}

    grouped = group_counts(counts, ALIASES)

    assert grouped == {"soda_can": 3}


def test_group_counts_keeps_unmapped_classes_separate():
    counts = {"red soda can": 2, "chips_bag": 4}

    grouped = group_counts(counts, ALIASES)

    assert grouped == {"soda_can": 2, "chips_bag": 4}


def test_group_counts_with_no_aliasing_is_a_passthrough():
    counts = {"soda_can": 1, "chips_bag": 1}

    assert group_counts(counts, {}) == counts
