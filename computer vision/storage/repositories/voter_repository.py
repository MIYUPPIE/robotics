"""CRUD for the `voters` table.

`all_embeddings` is the 1:N matching gallery consumed by biometric/matcher.py --
it never leaves this module carrying anything but (voter_id, embedding) pairs.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime

import numpy as np

from contracts.schemas import Gender, Language, VoterRecord


def insert(conn: sqlite3.Connection, voter: VoterRecord) -> None:
    conn.execute(
        """
        INSERT INTO voters (
            voter_id, surname, first_name, middle_name, dob, gender, state_code,
            lga, phone, language_pref, face_embedding, embedding_model_version,
            registered_at, has_voted
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            voter.voter_id,
            voter.surname,
            voter.first_name,
            voter.middle_name,
            voter.dob.isoformat(),
            voter.gender.value,
            voter.state_code,
            voter.lga,
            voter.phone,
            voter.language_pref.value,
            voter.face_embedding.astype(np.float32).tobytes(),
            voter.embedding_model_version,
            voter.registered_at.isoformat(),
            int(voter.has_voted),
        ),
    )


def _row_to_voter(row: sqlite3.Row) -> VoterRecord:
    embedding = np.frombuffer(row["face_embedding"], dtype=np.float32).copy()
    return VoterRecord(
        voter_id=row["voter_id"],
        surname=row["surname"],
        first_name=row["first_name"],
        middle_name=row["middle_name"],
        dob=date.fromisoformat(row["dob"]),
        gender=Gender(row["gender"]),
        state_code=row["state_code"],
        lga=row["lga"],
        phone=row["phone"],
        language_pref=Language(row["language_pref"]),
        face_embedding=embedding,
        embedding_model_version=row["embedding_model_version"],
        registered_at=datetime.fromisoformat(row["registered_at"]),
        has_voted=bool(row["has_voted"]),
    )


def get(conn: sqlite3.Connection, voter_id: str) -> VoterRecord | None:
    row = conn.execute("SELECT * FROM voters WHERE voter_id = ?", (voter_id,)).fetchone()
    return _row_to_voter(row) if row is not None else None


def exists(conn: sqlite3.Connection, voter_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM voters WHERE voter_id = ?", (voter_id,)).fetchone()
    return row is not None


def all_embeddings(conn: sqlite3.Connection) -> list[tuple[str, np.ndarray]]:
    """(voter_id, embedding) for every registered voter -- the 1:N matching gallery."""
    rows = conn.execute("SELECT voter_id, face_embedding FROM voters").fetchall()
    return [
        (row["voter_id"], np.frombuffer(row["face_embedding"], dtype=np.float32).copy())
        for row in rows
    ]


def count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM voters").fetchone()
    return int(row["n"])


def mark_has_voted(conn: sqlite3.Connection, voter_id: str) -> None:
    conn.execute("UPDATE voters SET has_voted = 1 WHERE voter_id = ?", (voter_id,))


def has_voted(conn: sqlite3.Connection, voter_id: str) -> bool:
    row = conn.execute("SELECT has_voted FROM voters WHERE voter_id = ?", (voter_id,)).fetchone()
    if row is None:
        raise KeyError(f"unknown voter_id {voter_id!r}")
    return bool(row["has_voted"])
