import pytest

from admin_app.tally import integrity_check, per_office_tally, turnout
from storage import db


class TestPerOfficeTally:
    def test_matches_hand_computed_expectation(self, seeded_conn):
        tally = per_office_tally(seeded_conn)
        assert tally["PRESIDENT"] == {"APC": 1, "PDP": 1}
        assert tally["GOVERNOR"] == {"PDP": 2}
        assert tally["SENATE"] == {"APC": 1, "LP": 1}
        assert tally["HOR"] == {"LP": 2}
        assert tally["SHA"] == {"APC": 2}

    def test_office_with_zero_votes_is_absent(self, seeded_conn):
        tally = per_office_tally(seeded_conn)
        assert "NOT_A_REAL_OFFICE" not in tally


class TestTurnout:
    def test_counts_match_seeded_fixture(self, seeded_conn):
        voted, registered, percent = turnout(seeded_conn)
        assert registered == 3
        assert voted == 2
        assert percent == pytest.approx(200 / 3, abs=0.01)

    def test_zero_registered_is_zero_percent_not_a_crash(self):
        empty_conn = db.connect_memory()
        voted, registered, percent = turnout(empty_conn)
        assert registered == 0
        assert voted == 0
        assert percent == 0.0
        empty_conn.close()


class TestIntegrityCheck:
    def test_holds_for_a_correctly_written_fixture(self, seeded_conn):
        assert integrity_check(seeded_conn, offices_per_session=5)

    def test_fails_if_an_extra_vote_row_is_tampered_in(self, seeded_conn):
        seeded_conn.execute(
            "INSERT INTO votes (office, party_code, cast_at) VALUES (?, ?, ?)",
            ("PRESIDENT", "APC", "2026-08-01T09:00:00"),
        )
        assert not integrity_check(seeded_conn, offices_per_session=5)
