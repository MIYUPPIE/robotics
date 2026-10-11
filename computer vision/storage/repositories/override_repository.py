"""Audit trail for registration-officer overrides of a flagged duplicate-face match."""

from __future__ import annotations

import sqlite3
from datetime import datetime


def log_override(
    conn: sqlite3.Connection,
    new_voter_id: str,
    matched_voter_id: str,
    similarity: float,
    officer_note: str,
    overridden_at: datetime,
) -> None:
    conn.execute(
        """
        INSERT INTO registration_overrides
            (new_voter_id, matched_voter_id, similarity, officer_note, overridden_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (new_voter_id, matched_voter_id, similarity, officer_note, overridden_at.isoformat()),
    )


def count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM registration_overrides").fetchone()
    return int(row["n"])
