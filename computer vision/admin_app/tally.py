"""Aggregate tally queries against the anonymous `votes` table.

Never queries `voters` rows beyond a COUNT(*) -- no name, embedding, or any
other PII flows through this module (see admin_app/export.py for the
enforced-by-test guarantee on the export path).
"""

from __future__ import annotations

import sqlite3

from storage.repositories import vote_repository, voted_log_repository, voter_repository


def per_office_tally(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    return vote_repository.tally_by_office(conn)


def turnout(conn: sqlite3.Connection) -> tuple[int, int, float]:
    """Returns (voted, registered, turnout_percent)."""
    registered = voter_repository.count(conn)
    voted = voted_log_repository.count(conn)
    percent = (voted / registered * 100.0) if registered else 0.0
    return voted, registered, percent


def integrity_check(conn: sqlite3.Connection, offices_per_session: int) -> bool:
    """The core write-path invariant from docs/ARCHITECTURE.md#ballot-secrecy:
    total votes == offices_per_session x number of completed sessions."""
    total_votes = vote_repository.total_votes(conn)
    total_voted = voted_log_repository.count(conn)
    return total_votes == offices_per_session * total_voted
