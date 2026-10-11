import sqlite3
from datetime import datetime

import pytest

from contracts.schemas import Office
from storage.repositories import vote_repository, voted_log_repository, voter_repository


class _FlakyConnProxy:
    """Wraps a real sqlite3.Connection, forwarding everything except that it
    raises after a chosen number of `INSERT INTO votes` statements -- used to
    prove submit_vote_session rolls back cleanly on a mid-transaction failure.
    """

    def __init__(self, real_conn: sqlite3.Connection, fail_after_n_vote_inserts: int):
        self._real = real_conn
        self._vote_inserts = 0
        self._fail_after = fail_after_n_vote_inserts

    def execute(self, sql: str, params=()):
        if sql.strip().upper().startswith("INSERT INTO VOTES"):
            self._vote_inserts += 1
            if self._vote_inserts > self._fail_after:
                raise sqlite3.OperationalError("simulated failure")
        return self._real.execute(sql, params)

    def __getattr__(self, name):
        return getattr(self._real, name)


class TestSubmitVoteSessionHappyPath:
    def test_writes_one_vote_row_per_office_plus_one_voted_log_row(self, conn, make_voter):
        voter = make_voter()
        voter_repository.insert(conn, voter)
        selections = {
            Office.PRESIDENT: "APC",
            Office.GOVERNOR: "PDP",
            Office.SENATE: "LP",
            Office.HOR: "NNPP",
            Office.SHA: "APGA",
        }

        vote_repository.submit_vote_session(
            conn, voter.voter_id, selections, datetime(2026, 8, 1, 10, 3, 12)
        )

        assert vote_repository.total_votes(conn) == 5
        assert voted_log_repository.count(conn) == 1
        assert voted_log_repository.has_voted(conn, voter.voter_id)
        assert voter_repository.has_voted(conn, voter.voter_id)

    def test_cast_at_is_bucketed_to_nearest_15_minutes(self, conn, make_voter):
        voter = make_voter()
        voter_repository.insert(conn, voter)
        vote_repository.submit_vote_session(
            conn, voter.voter_id, {Office.PRESIDENT: "APC"}, datetime(2026, 8, 1, 10, 7, 42)
        )
        row = conn.execute("SELECT cast_at FROM votes").fetchone()
        assert row["cast_at"] == "2026-08-01T10:00:00"

    def test_no_voter_reference_in_votes_table(self, conn, make_voter):
        voter = make_voter()
        voter_repository.insert(conn, voter)
        vote_repository.submit_vote_session(
            conn, voter.voter_id, {Office.PRESIDENT: "APC"}, datetime(2026, 8, 1, 10, 0, 0)
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(votes)").fetchall()}
        assert "voter_id" not in columns

    def test_rejects_empty_selections(self, conn, make_voter):
        voter = make_voter()
        voter_repository.insert(conn, voter)
        with pytest.raises(vote_repository.VoteSubmissionError):
            vote_repository.submit_vote_session(conn, voter.voter_id, {}, datetime.now())


class TestSubmitVoteSessionAtomicity:
    def test_mid_transaction_failure_leaves_zero_rows_everywhere(self, conn, make_voter):
        voter = make_voter()
        voter_repository.insert(conn, voter)
        selections = {
            Office.PRESIDENT: "APC",
            Office.GOVERNOR: "PDP",
            Office.SENATE: "LP",
        }
        flaky = _FlakyConnProxy(conn, fail_after_n_vote_inserts=2)

        with pytest.raises(vote_repository.VoteSubmissionError):
            vote_repository.submit_vote_session(flaky, voter.voter_id, selections, datetime.now())

        assert vote_repository.total_votes(conn) == 0
        assert voted_log_repository.count(conn) == 0
        assert voter_repository.has_voted(conn, voter.voter_id) is False

    def test_second_full_session_for_same_voter_is_rejected_and_atomic(self, conn, make_voter):
        voter = make_voter()
        voter_repository.insert(conn, voter)
        first = {Office.PRESIDENT: "APC"}
        second = {Office.PRESIDENT: "PDP", Office.GOVERNOR: "LP"}

        vote_repository.submit_vote_session(
            conn, voter.voter_id, first, datetime(2026, 8, 1, 9, 0, 0)
        )
        assert vote_repository.total_votes(conn) == 1

        with pytest.raises(vote_repository.VoteSubmissionError):
            vote_repository.submit_vote_session(
                conn, voter.voter_id, second, datetime(2026, 8, 1, 9, 5, 0)
            )

        # The second attempt must not have added any of its votes either.
        assert vote_repository.total_votes(conn) == 1


class TestTally:
    def test_tally_by_office_aggregates_correctly(self, conn, make_voter):
        v1 = make_voter(voter_id="PVR-LAG-20260101-AAAAAA", seed=0)
        v2 = make_voter(voter_id="PVR-LAG-20260101-BBBBBB", seed=1)
        voter_repository.insert(conn, v1)
        voter_repository.insert(conn, v2)

        vote_repository.submit_vote_session(
            conn, v1.voter_id, {Office.PRESIDENT: "APC", Office.GOVERNOR: "PDP"}, datetime.now()
        )
        vote_repository.submit_vote_session(
            conn, v2.voter_id, {Office.PRESIDENT: "APC", Office.GOVERNOR: "LP"}, datetime.now()
        )

        tally = vote_repository.tally_by_office(conn)

        assert tally["PRESIDENT"]["APC"] == 2
        assert tally["GOVERNOR"]["PDP"] == 1
        assert tally["GOVERNOR"]["LP"] == 1

    def test_integrity_invariant_votes_equals_5x_voted_log(self, conn, make_voter):
        offices = list(Office)
        for i in range(3):
            v = make_voter(voter_id=f"PVR-LAG-20260101-VOTER{i:02d}", seed=i)
            voter_repository.insert(conn, v)
            selections = {office: "APC" for office in offices}
            vote_repository.submit_vote_session(conn, v.voter_id, selections, datetime.now())

        assert vote_repository.total_votes(conn) == 5 * voted_log_repository.count(conn)
