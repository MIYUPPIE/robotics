"""Writes shelf-removal events out as a POS-ready item list (JSON).

Generic and integration-agnostic on purpose: every removal event -- theft or
otherwise -- accumulates into a running list of {product_name, quantity},
grouped by product_grouping.canonical_name(), written to disk on every
change. Point a real POS's file-import or a small watcher script at
config.POS_FEED_PATH, or replace `_write()` below with a direct API call
once you know which POS platform this is integrating with.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from product_grouping import canonical_name


@dataclass(frozen=True)
class PosLineItem:
    product_name: str
    quantity: int


class PosFeed:
    def __init__(self, output_path: Path, aliases: dict[str, str] | None = None):
        self.output_path = output_path
        self.aliases: dict[str, str] = aliases if aliases is not None else {}
        self._quantities: dict[str, int] = {}
        self._write()

    def record_removal(self, class_name: str, quantity: int) -> None:
        name = canonical_name(class_name, self.aliases)
        self._quantities[name] = self._quantities.get(name, 0) + quantity
        self._write()

    def clear(self) -> None:
        """Start a fresh list -- call once a transaction (e.g. checkout) closes."""
        self._quantities = {}
        self._write()

    def line_items(self) -> list[PosLineItem]:
        return [
            PosLineItem(name, quantity)
            for name, quantity in sorted(self._quantities.items())
        ]

    def _write(self) -> None:
        payload: list[dict[str, Any]] = [
            {"product_name": item.product_name, "quantity": item.quantity}
            for item in self.line_items()
        ]
        self.output_path.write_text(json.dumps(payload, indent=2))
