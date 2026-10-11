from contracts.schemas import OFFICE_ORDER, Office
from kiosk_app.session import VotingSession


class TestVotingSession:
    def test_current_office_starts_at_first_in_order(self):
        session = VotingSession()
        assert session.current_office() == OFFICE_ORDER[0]

    def test_is_last_office_only_true_at_final_index(self):
        session = VotingSession()
        assert not session.is_last_office()
        session.current_office_index = len(OFFICE_ORDER) - 1
        assert session.is_last_office()

    def test_all_offices_selected_requires_every_office(self):
        session = VotingSession()
        for office in OFFICE_ORDER[:-1]:
            session.selections[office] = "APC"
            assert not session.all_offices_selected()
        session.selections[OFFICE_ORDER[-1]] = "APC"
        assert session.all_offices_selected()

    def test_reset_clears_everything(self):
        session = VotingSession()
        session.voter_id = "PVR-LAG-1"
        session.selections[Office.PRESIDENT] = "APC"
        session.current_office_index = 3
        session.return_to_review = True
        session.liveness_retry_count = 2

        session.reset()

        assert session.voter_id is None
        assert session.selections == {}
        assert session.current_office_index == 0
        assert session.return_to_review is False
        assert session.liveness_retry_count == 0
