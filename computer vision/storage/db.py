"""SQLite connection factory: WAL mode, foreign keys on, explicit transaction
control (autocommit at the driver level, BEGIN/COMMIT/ROLLBACK issued by
callers that need atomicity -- see storage/repositories/vote_repository.py).
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
REPO_ROOT = Path(__file__).parent.parent


def default_db_path() -> Path:
    """Resolves the one shared kiosk database path: $VOTER_KIOSK_DB_PATH if
    set (see .env.example), otherwise an absolute path anchored at the repo
    root.

    Deliberately never a bare relative filename: registration_app,
    kiosk_app, and admin_app are three independently-launched processes,
    and if each resolved "voter_kiosk.db" against its own current working
    directory, launching any one of them from a different terminal/directory
    would silently point it at a different, empty database -- indistinguishable
    from "the system doesn't recognize a previously registered voter."
    """
    override = os.environ.get("VOTER_KIOSK_DB_PATH")
    if override:
        return Path(override)
    return REPO_ROOT / "voter_kiosk.db"


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text())


def connect(db_path: str | Path, *, busy_timeout_ms: int = 5000) -> sqlite3.Connection:
    """Open (creating if needed) the on-disk kiosk database."""
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute(f"PRAGMA busy_timeout = {busy_timeout_ms};")
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def connect_memory() -> sqlite3.Connection:
    """In-memory database with the same schema -- used by the gate-test suite."""
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn
