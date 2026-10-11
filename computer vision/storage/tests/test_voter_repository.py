import sqlite3

import pytest

from storage.repositories import voter_repository


class TestVoterRepository:
    def test_exists_true_after_insert(self, conn, make_voter):
        voter = make_voter()
        voter_repository.insert(conn, voter)
        assert voter_repository.exists(conn, voter.voter_id)

    def test_exists_false_for_unknown_id(self, conn):
        assert not voter_repository.exists(conn, "PVR-NOPE-00000000-ZZZZZZ")

    def test_duplicate_voter_id_rejected(self, conn, make_voter):
        voter_repository.insert(conn, make_voter(voter_id="PVR-LAG-20260101-DUPE01"))
        with pytest.raises(sqlite3.IntegrityError):
            voter_repository.insert(conn, make_voter(voter_id="PVR-LAG-20260101-DUPE01", seed=1))

    def test_count(self, conn, make_voter):
        assert voter_repository.count(conn) == 0
        voter_repository.insert(conn, make_voter(voter_id="PVR-LAG-20260101-AAAAAA", seed=0))
        voter_repository.insert(conn, make_voter(voter_id="PVR-LAG-20260101-BBBBBB", seed=1))
        assert voter_repository.count(conn) == 2

    def test_all_embeddings_returns_every_voter(self, conn, make_voter):
        voter_repository.insert(conn, make_voter(voter_id="PVR-LAG-20260101-AAAAAA", seed=0))
        voter_repository.insert(conn, make_voter(voter_id="PVR-LAG-20260101-BBBBBB", seed=1))

        gallery = voter_repository.all_embeddings(conn)

        assert {voter_id for voter_id, _ in gallery} == {
            "PVR-LAG-20260101-AAAAAA",
            "PVR-LAG-20260101-BBBBBB",
        }
        for _, embedding in gallery:
            assert embedding.shape == (512,)

    def test_mark_has_voted_flips_flag(self, conn, make_voter):
        voter = make_voter()
        voter_repository.insert(conn, voter)
        assert voter_repository.has_voted(conn, voter.voter_id) is False

        voter_repository.mark_has_voted(conn, voter.voter_id)

        assert voter_repository.has_voted(conn, voter.voter_id) is True

    def test_has_voted_unknown_voter_raises(self, conn):
        with pytest.raises(KeyError):
            voter_repository.has_voted(conn, "PVR-NOPE-00000000-ZZZZZZ")
