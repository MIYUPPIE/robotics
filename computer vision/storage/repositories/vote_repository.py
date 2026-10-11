"""Anonymous vote storage + the one atomic ballot-submission transaction.

This module is the entire write path for `votes`/`voted_log`. Nothing outside
`submit_vote_session` may write to either table -- that invariant is what makes
"abandoning a session writes nothing" true (see docs/ARCHITECTURE.md).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from contracts.schemas import Office
from storage.repositories import voter_repository


class VoteSubmissionError(RuntimeError):
    """Raised when submit_vote_session fails. The transaction is rolled back
    before this is raised, so the caller can assume nothing was written."""


def _bucket_15min(ts: datetime) -> datetime:
    floored_minute = (ts.minute // 15) * 15
    return ts.replace(minute=floored_minute, second=0, microsecond=0)


def submit_vote_session(
    conn: sqlite3.Connection,
    voter_id: str,
    selections: dict[Office, str],
    now: datetime,
) -> None:
    """Atomically cast every office selection and mark the voter as having voted.

    Writes, in a single transaction: one `votes` row per office selection
    (fully anonymous -- no voter_id anywhere in that row, `cast_at` rounded to
    the nearest 15 minutes), one exact-timestamped `voted_log` row, and flips
    `voters.has_voted`. All-or-nothing: any failure rolls back every part.
    """
    if not selections:
        raise VoteSubmissionError("no selections to submit")

    bucketed_cast_at = _bucket_15min(now).isoformat()
    exact_cast_at = now.isoformat()

    try:
        conn.execute("BEGIN IMMEDIATE")
        for office, party_code in selections.items():
            conn.execute(
                "INSERT INTO votes (office, party_code, cast_at) VALUES (?, ?, ?)",
                (office.value, party_code, bucketed_cast_at),
            )
        conn.execute(
            "INSERT INTO voted_log (voter_id, cast_at) VALUES (?, ?)",
            (voter_id, exact_cast_at),
        )
        voter_repository.mark_has_voted(conn, voter_id)
        conn.execute("COMMIT")
    except sqlite3.Error as exc:
        conn.execute("ROLLBACK")
        raise VoteSubmissionError(f"vote submission failed and was rolled back: {exc}") from exc


def tally_by_office(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    """{office: {party_code: count}} for every office/party combination cast so far."""
    rows = conn.execute(
        "SELECT office, party_code, COUNT(*) AS n FROM votes GROUP BY office, party_code"
    ).fetchall()
    result: dict[str, dict[str, int]] = {}
    for row in rows:
        result.setdefault(row["office"], {})[row["party_code"]] = int(row["n"])
    return result


def total_votes(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM votes").fetchone()
    return int(row["n"])
