"""Loads party logo image assets (data/party_logos/) into the BGRA arrays
hand_ui/renderer.py composites onto the ballot. Thin I/O shell -- the pure
placement geometry lives in hand_ui/regions.py::fit_within_region.

Not every party has a logo asset (see data/party_logos/ATTRIBUTION.md and
docs/DISCLAIMER.md#party-logos); parties with none are simply absent from the
returned dict, and callers fall back to the color+acronym card (see
hand_ui/renderer.py::draw_region).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_party_logos(
    data_dir: Path, party_records: list[dict[str, str]]
) -> dict[str, np.ndarray]:
    """`party_records` is the parsed contents of data/parties_nigeria.json.
    Returns {acronym: BGRA ndarray} for every entry with a `logo` field that
    resolves to a real, readable image file under `data_dir`.
    """
    logos: dict[str, np.ndarray] = {}
    for party in party_records:
        rel_path = party.get("logo")
        if not rel_path:
            continue
        image = cv2.imread(str(data_dir / rel_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        if image.ndim == 2:  # grayscale, no alpha
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        elif image.shape[2] == 3:  # BGR, no alpha -- treat as fully opaque
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        logos[party["acronym"]] = image
    return logos
