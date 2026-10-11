"""Read access to the voted_log audit table.

Writes only ever happen inside vote_repository.submit_vote_session, as part
of the atomic ballot-submission transaction -- this module is read-only.
"""

from __future__ import annotations

import sqlite3


def has_voted(conn: sqlite3.Connection, voter_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM voted_log WHERE voter_id = ?", (voter_id,)).fetchone()
    return row is not None


def count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM voted_log").fetchone()
    return int(row["n"])
