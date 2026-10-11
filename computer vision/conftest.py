"""Fixtures shared by every module's gate-test suite.

Placed at the repo root (not inside a `tests/` package) so pytest's conftest
discovery applies it to contracts/tests, storage/tests, biometric/tests,
hand_ui/tests, kiosk_app/tests, voice/tests, etc. -- anything nested under a
`tests/` subdirectory only reaches tests inside that same subdirectory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from contracts.schemas import EMBEDDING_DIM


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """A fresh, not-yet-created on-disk DB path for tests that need file I/O."""
    return tmp_path / "voter_kiosk_test.db"


def _normalized_embedding(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def make_embedding():
    """Factory fixture: make_embedding(seed) -> deterministic L2-normalized
    512-d float32 vector, standing in for a real InsightFace embedding."""
    return _normalized_embedding
