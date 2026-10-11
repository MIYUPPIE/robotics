from datetime import date, datetime

import numpy as np
import pytest

from contracts.schemas import EMBEDDING_DIM, Gender, Language, VoterRecord
from storage import db


@pytest.fixture
def conn():
    connection = db.connect_memory()
    yield connection
    connection.close()


def _normalized_embedding(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def make_voter():
    def _make(voter_id: str = "PVR-LAG-20260101-AAAAAA", seed: int = 0, **overrides):
        defaults = dict(
            voter_id=voter_id,
            surname="Okafor",
            first_name="Ada",
            middle_name=None,
            dob=date(2000, 1, 1),
            gender=Gender.FEMALE,
            state_code="LAG",
            lga="Ikeja",
            phone=None,
            language_pref=Language.ENGLISH,
            face_embedding=_normalized_embedding(seed),
            embedding_model_version="insightface-buffalo_l",
            registered_at=datetime(2026, 1, 1, 12, 0, 0),
            has_voted=False,
        )
        defaults.update(overrides)
        # mypy can't verify per-field types through a dict-unpack factory pattern.
        return VoterRecord(**defaults)  # type: ignore[arg-type]

    return _make
