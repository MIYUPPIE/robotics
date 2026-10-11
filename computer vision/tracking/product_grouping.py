"""Groups visually-distinct SKU variants under one canonical product name.

Detection stays precise -- "red soda can" and "blue soda can" are tracked
(and, for a fine-tuned detector, trained) as separate classes, since lumping
them together at detection time would make the model worse, not simpler.
Grouping only happens here, at the reporting layer, via config.PRODUCT_ALIASES.
"""
from __future__ import annotations

from collections.abc import Mapping

import config


def canonical_name(
    detected_class_name: str, aliases: Mapping[str, str] = config.PRODUCT_ALIASES
) -> str:
    """Maps a raw detected class name to its POS-facing grouped name.

    Unmapped classes pass through unchanged -- aliasing is opt-in.
    """
    return aliases.get(detected_class_name, detected_class_name)


def group_counts(
    counts: Mapping[str, int], aliases: Mapping[str, str] = config.PRODUCT_ALIASES
) -> dict[str, int]:
    """Re-aggregates a {detected_class_name: count} mapping by canonical name."""
    grouped: dict[str, int] = {}
    for class_name, count in counts.items():
        name = canonical_name(class_name, aliases)
        grouped[name] = grouped.get(name, 0) + count
    return grouped
