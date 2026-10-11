from pathlib import Path

import numpy as np

from storage import db
from storage.repositories import voter_repository


class TestSchemaRoundTrip:
    def test_insert_and_get_preserves_all_fields(self, conn, make_voter):
        voter = make_voter()
        voter_repository.insert(conn, voter)

        fetched = voter_repository.get(conn, voter.voter_id)

        assert fetched is not None
        assert fetched.voter_id == voter.voter_id
        assert fetched.surname == voter.surname
        assert fetched.first_name == voter.first_name
        assert fetched.dob == voter.dob
        assert fetched.gender == voter.gender
        assert fetched.state_code == voter.state_code
        assert fetched.lga == voter.lga
        assert fetched.language_pref == voter.language_pref
        assert fetched.embedding_model_version == voter.embedding_model_version
        assert fetched.registered_at == voter.registered_at
        assert fetched.has_voted == voter.has_voted
        np.testing.assert_array_almost_equal(fetched.face_embedding, voter.face_embedding)

    def test_get_missing_voter_returns_none(self, conn):
        assert voter_repository.get(conn, "PVR-NOPE-00000000-ZZZZZZ") is None

    def test_tables_exist(self, conn):
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert {"voters", "votes", "voted_log", "registration_overrides"} <= tables


class TestDefaultDbPath:
    def test_defaults_to_an_absolute_path_anchored_at_repo_root(self, monkeypatch):
        monkeypatch.delenv("VOTER_KIOSK_DB_PATH", raising=False)
        path = db.default_db_path()
        assert path.is_absolute()
        assert path.name == "voter_kiosk.db"
        assert path.parent == db.REPO_ROOT

    def test_is_independent_of_current_working_directory(self, monkeypatch, tmp_path):
        """The whole point: registration_app, kiosk_app, and admin_app are
        three independently-launched processes that may be started from
        different shells/directories. They must still agree on one file."""
        monkeypatch.delenv("VOTER_KIOSK_DB_PATH", raising=False)
        monkeypatch.chdir(tmp_path)
        assert db.default_db_path() == db.REPO_ROOT / "voter_kiosk.db"

    def test_env_override_takes_precedence(self, monkeypatch, tmp_path):
        override = tmp_path / "custom.db"
        monkeypatch.setenv("VOTER_KIOSK_DB_PATH", str(override))
        assert db.default_db_path() == override

    def test_env_override_result_is_a_path_object(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VOTER_KIOSK_DB_PATH", str(tmp_path / "x.db"))
        assert isinstance(db.default_db_path(), Path)
