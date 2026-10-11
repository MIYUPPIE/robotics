from datetime import date, datetime

import numpy as np
import pytest

from contracts.schemas import EMBEDDING_DIM, Gender, Language, Office, VoterRecord
from storage import db
from storage.repositories import vote_repository, voter_repository


def _embedding(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _voter(i: int) -> VoterRecord:
    return VoterRecord(
        voter_id=f"PVR-LAG-20260101-SEED{i:02d}",
        surname=f"Seed{i}",
        first_name="Voter",
        middle_name=None,
        dob=date(2000, 1, 1),
        gender=Gender.OTHER,
        state_code="LAG",
        lga="Ikeja",
        phone=None,
        language_pref=Language.ENGLISH,
        face_embedding=_embedding(i),
        embedding_model_version="test",
        registered_at=datetime(2026, 1, 1),
    )


@pytest.fixture
def seeded_conn():
    """3 registered voters, 2 of whom have completed a full 5-office session
    with known party choices -- used to hand-verify tally/export output."""
    conn = db.connect_memory()

    voters = [_voter(i) for i in range(3)]
    for v in voters:
        voter_repository.insert(conn, v)

    vote_repository.submit_vote_session(
        conn,
        voters[0].voter_id,
        {
            Office.PRESIDENT: "APC",
            Office.GOVERNOR: "PDP",
            Office.SENATE: "APC",
            Office.HOR: "LP",
            Office.SHA: "APC",
        },
        datetime(2026, 8, 1, 9, 0, 0),
    )
    vote_repository.submit_vote_session(
        conn,
        voters[1].voter_id,
        {
            Office.PRESIDENT: "PDP",
            Office.GOVERNOR: "PDP",
            Office.SENATE: "LP",
            Office.HOR: "LP",
            Office.SHA: "APC",
        },
        datetime(2026, 8, 1, 9, 15, 0),
    )
    # voters[2] registered but has not voted.

    yield conn
    conn.close()
