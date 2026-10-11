"""Anonymized CSV export -- only aggregate tallies, never voter/embedding data.

admin_app/tests/test_export.py enforces this by string-searching the output
for `embedding` and for known seeded voter_ids and asserting absence.
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from admin_app.tally import per_office_tally

CSV_COLUMNS = ["office", "party_code", "party_name", "vote_count"]


def export_tallies_csv(
    conn: sqlite3.Connection, path: str | Path, party_names: dict[str, str]
) -> Path:
    """Writes office/party/vote-count aggregates only. Queries only `votes`
    (via per_office_tally -> vote_repository, never `voters`)."""
    path = Path(path)
    tally = per_office_tally(conn)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for office in sorted(tally):
            for party_code, count in sorted(tally[office].items()):
                party_name = party_names.get(party_code, party_code)
                writer.writerow([office, party_code, party_name, count])
    return path
