"""DEMO-ONLY: seeds a few synthetic registered voters (fixture embeddings,
NOT real faces) so the kiosk/admin/registration flows can be smoke-tested
without needing real people first.

Guarded: refuses to seed into a database that already has non-demo
registrations unless --force is passed. Every seeded voter's surname starts
with "DemoVoter" so it's always obvious which rows are synthetic.

Usage: python scripts/seed_demo_data.py [--count 5] [--db-path PATH] [--force]
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import date, datetime

import numpy as np

from biometric.voter_id import generate_voter_id
from contracts.schemas import EMBEDDING_DIM, Gender, Language, VoterRecord
from storage import db
from storage.repositories import voter_repository

DEMO_MARKER_SURNAME_PREFIX = "DemoVoter"


def is_demo_surname(surname: str) -> bool:
    return surname.startswith(DEMO_MARKER_SURNAME_PREFIX)


def non_demo_surnames(surnames: list[str]) -> list[str]:
    return [s for s in surnames if not is_demo_surname(s)]


def _embedding(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def seed(conn: sqlite3.Connection, count: int) -> list[VoterRecord]:
    created: list[VoterRecord] = []
    for i in range(count):
        voter_id = generate_voter_id(
            "LAG", date.today(), exists=lambda vid: voter_repository.exists(conn, vid)
        )
        voter = VoterRecord(
            voter_id=voter_id,
            surname=f"{DEMO_MARKER_SURNAME_PREFIX}{i}",
            first_name="Synthetic",
            middle_name=None,
            dob=date(1990, 1, 1),
            gender=Gender.OTHER,
            state_code="LAG",
            lga="Ikeja",
            phone=None,
            language_pref=Language.ENGLISH,
            face_embedding=_embedding(i),
            embedding_model_version="synthetic-demo-seed",
            registered_at=datetime.now(),
        )
        voter_repository.insert(conn, voter)
        created.append(voter)
    return created


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument(
        "--db-path",
        default=None,
        help="defaults to the shared kiosk DB (see storage.db.default_db_path)",
    )
    parser.add_argument(
        "--force", action="store_true", help="seed even if the DB already has non-demo voters"
    )
    args = parser.parse_args()

    db_path = args.db_path or db.default_db_path()
    print(f"using database: {db_path}")
    conn = db.connect(db_path)
    if voter_repository.count(conn):
        rows = conn.execute("SELECT surname FROM voters").fetchall()
        blockers = non_demo_surnames([r["surname"] for r in rows])
        if blockers and not args.force:
            print(
                f"Refusing to seed: database already has {len(blockers)} non-demo voter(s). "
                "Pass --force to seed anyway."
            )
            conn.close()
            return 1

    created = seed(conn, args.count)
    for voter in created:
        print(f"seeded: {voter.voter_id} ({voter.surname} {voter.first_name})")
    conn.close()
    print(f"\n{len(created)} demo voters seeded into {db_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
