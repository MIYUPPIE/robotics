import pytest

from biometric.thresholds import MAX_LIVENESS_RETRIES
from contracts.schemas import Language, MatchResult, Office
from kiosk_app.state_machine import KioskState, KioskStateMachine

CONFIDENT_MATCH = MatchResult(
    matched=True, voter_id="PVR-LAG-1", similarity=0.9, margin=0.2, ambiguous=False
)
AMBIGUOUS_MATCH = MatchResult(
    matched=True, voter_id="PVR-LAG-1", similarity=0.9, margin=0.01, ambiguous=True
)
NO_MATCH = MatchResult(matched=False, voter_id=None, similarity=0.1, margin=0.0, ambiguous=False)

ALL_OFFICES = list(Office)


@pytest.fixture
def sm(spy_submit, clock):
    return KioskStateMachine(submit_vote_session=spy_submit, clock=clock)


def _advance_to_office_ballot(sm: KioskStateMachine) -> None:
    sm.presence_detected()
    sm.language_selected(Language.ENGLISH)
    sm.face_auth_result(CONFIDENT_MATCH, has_voted=False)


def _select_all_offices(sm: KioskStateMachine, party_code: str = "APC") -> None:
    for _ in range(len(ALL_OFFICES)):
        sm.office_selection_confirmed(party_code)


class TestInitialState:
    def test_starts_idle(self, sm):
        assert sm.state == KioskState.IDLE_ATTRACT
        assert sm.session.voter_id is None


class TestIdleToLanguageSelect:
    def test_presence_detected_moves_to_language_select(self, sm):
        sm.presence_detected()
        assert sm.state == KioskState.LANGUAGE_SELECT

    def test_presence_detected_ignored_outside_idle(self, sm):
        sm.presence_detected()
        sm.presence_detected()  # already in LANGUAGE_SELECT, should no-op
        assert sm.state == KioskState.LANGUAGE_SELECT


class TestLanguageSelection:
    def test_sets_language_and_moves_to_face_auth(self, sm):
        sm.presence_detected()
        sm.language_selected(Language.YORUBA)
        assert sm.state == KioskState.FACE_AUTH
        assert sm.session.language == Language.YORUBA

    def test_ignored_outside_language_select_state(self, sm):
        sm.language_selected(Language.YORUBA)  # still IDLE_ATTRACT
        assert sm.state == KioskState.IDLE_ATTRACT
        assert sm.session.language is None


class TestFaceAuthLiveness:
    def test_stays_in_face_auth_until_retries_exhausted(self, sm):
        sm.presence_detected()
        sm.language_selected(Language.ENGLISH)
        for _ in range(MAX_LIVENESS_RETRIES):
            sm.face_auth_liveness_failed()
            assert sm.state == KioskState.FACE_AUTH
        sm.face_auth_liveness_failed()  # one more than the max
        assert sm.state == KioskState.FACE_AUTH_FAIL

    def test_ignored_outside_face_auth(self, sm):
        sm.face_auth_liveness_failed()
        assert sm.state == KioskState.IDLE_ATTRACT


class TestFaceAuthResult:
    def test_confident_unvoted_match_proceeds_to_first_office(self, sm):
        sm.presence_detected()
        sm.language_selected(Language.ENGLISH)
        sm.face_auth_result(CONFIDENT_MATCH, has_voted=False)
        assert sm.state == KioskState.OFFICE_BALLOT
        assert sm.session.voter_id == "PVR-LAG-1"
        assert sm.session.current_office() == Office.PRESIDENT

    def test_no_match_goes_to_face_auth_fail(self, sm):
        sm.presence_detected()
        sm.language_selected(Language.ENGLISH)
        sm.face_auth_result(NO_MATCH, has_voted=False)
        assert sm.state == KioskState.FACE_AUTH_FAIL

    def test_ambiguous_match_never_guesses_goes_to_face_auth_fail(self, sm):
        sm.presence_detected()
        sm.language_selected(Language.ENGLISH)
        sm.face_auth_result(AMBIGUOUS_MATCH, has_voted=False)
        assert sm.state == KioskState.FACE_AUTH_FAIL
        assert sm.session.voter_id is None

    def test_confident_match_already_voted_goes_to_already_voted(self, sm):
        sm.presence_detected()
        sm.language_selected(Language.ENGLISH)
        sm.face_auth_result(CONFIDENT_MATCH, has_voted=True)
        assert sm.state == KioskState.ALREADY_VOTED


class TestOfficeBallotSequence:
    def test_confirming_advances_through_all_five_offices_to_review(self, sm):
        _advance_to_office_ballot(sm)
        for i, office in enumerate(ALL_OFFICES[:-1]):
            assert sm.session.current_office() == office
            sm.office_selection_confirmed("APC")
            assert sm.state == KioskState.OFFICE_BALLOT
            assert sm.session.current_office_index == i + 1
        sm.office_selection_confirmed("APC")  # last office
        assert sm.state == KioskState.REVIEW

    def test_records_selection_for_correct_office(self, sm):
        _advance_to_office_ballot(sm)
        sm.office_selection_confirmed("PDP")
        assert sm.session.selections[Office.PRESIDENT] == "PDP"

    def test_ignored_outside_office_ballot_state(self, sm):
        sm.office_selection_confirmed("APC")  # still IDLE_ATTRACT
        assert sm.session.selections == {}

    def test_edit_jump_returns_to_review_after_one_confirm(self, sm):
        _advance_to_office_ballot(sm)
        _select_all_offices(sm)
        assert sm.state == KioskState.REVIEW

        sm.review_edit_requested(Office.GOVERNOR)
        assert sm.state == KioskState.OFFICE_BALLOT
        assert sm.session.current_office() == Office.GOVERNOR

        sm.office_selection_confirmed("LP")
        assert sm.state == KioskState.REVIEW
        assert sm.session.selections[Office.GOVERNOR] == "LP"
        # Every other office's earlier selection is untouched.
        assert sm.session.selections[Office.PRESIDENT] == "APC"


class TestReview:
    def test_start_over_discards_session_and_writes_nothing(self, sm, spy_submit):
        _advance_to_office_ballot(sm)
        _select_all_offices(sm)
        sm.review_start_over_confirmed()
        assert sm.state == KioskState.IDLE_ATTRACT
        assert sm.session.voter_id is None
        assert sm.session.selections == {}
        assert spy_submit.calls == []

    def test_start_over_ignored_outside_review(self, sm):
        sm.review_start_over_confirmed()
        assert sm.state == KioskState.IDLE_ATTRACT  # no-op, already idle


class TestSubmit:
    def test_full_session_writes_exactly_once_with_all_selections(self, sm, spy_submit):
        _advance_to_office_ballot(sm)
        _select_all_offices(sm, "APC")
        sm.submit_confirmed()

        assert sm.state == KioskState.THANK_YOU
        assert len(spy_submit.calls) == 1
        voter_id, selections, _ = spy_submit.calls[0]
        assert voter_id == "PVR-LAG-1"
        assert set(selections.keys()) == set(ALL_OFFICES)

    def test_refuses_to_submit_incomplete_ballot(self, sm, spy_submit):
        _advance_to_office_ballot(sm)
        sm.office_selection_confirmed("APC")  # only 1 of 5 -- still OFFICE_BALLOT, not REVIEW
        # Force into REVIEW artificially is not possible via the public API,
        # which is exactly the point: submit_confirmed only runs from REVIEW.
        sm.submit_confirmed()
        assert spy_submit.calls == []

    def test_submission_failure_goes_to_submit_failed_without_writing_thank_you(
        self, sm, spy_submit
    ):
        spy_submit.should_fail = True
        _advance_to_office_ballot(sm)
        _select_all_offices(sm)
        sm.submit_confirmed()
        assert sm.state == KioskState.SUBMIT_FAILED
        assert len(spy_submit.calls) == 1

    def test_retry_after_failure_resubmits_and_can_succeed(self, sm, spy_submit):
        spy_submit.should_fail = True
        _advance_to_office_ballot(sm)
        _select_all_offices(sm)
        sm.submit_confirmed()
        assert sm.state == KioskState.SUBMIT_FAILED

        spy_submit.should_fail = False
        sm.retry_submit()
        assert sm.state == KioskState.THANK_YOU
        assert len(spy_submit.calls) == 2

    def test_unexpected_non_vote_submission_error_still_degrades_to_submit_failed(self, clock):
        """A voting kiosk must never crash into an undefined state just
        because something unanticipated went wrong during submit -- even an
        exception type nobody planned for must land on the same
        officer-recoverable SUBMIT_FAILED screen, not propagate and take
        the whole kiosk down."""

        def explode(voter_id, selections, now):
            raise RuntimeError("something nobody anticipated")

        sm = KioskStateMachine(submit_vote_session=explode, clock=clock)
        _advance_to_office_ballot(sm)
        _select_all_offices(sm)

        sm.submit_confirmed()  # must not raise

        assert sm.state == KioskState.SUBMIT_FAILED

    def test_retry_ignored_outside_submit_failed(self, sm, spy_submit):
        sm.retry_submit()
        assert spy_submit.calls == []


class TestOfficerReset:
    def test_resets_from_deep_in_a_session_without_writing(self, sm, spy_submit):
        _advance_to_office_ballot(sm)
        sm.office_selection_confirmed("APC")
        sm.officer_reset()
        assert sm.state == KioskState.IDLE_ATTRACT
        assert sm.session.selections == {}
        assert spy_submit.calls == []

    def test_resets_from_review(self, sm):
        _advance_to_office_ballot(sm)
        _select_all_offices(sm)
        sm.officer_reset()
        assert sm.state == KioskState.IDLE_ATTRACT


class TestTimeouts:
    @pytest.mark.parametrize(
        "setup_state,timeout_attr",
        [
            (KioskState.LANGUAGE_SELECT, "LANGUAGE_SELECT_TIMEOUT_S"),
            (KioskState.FACE_AUTH, "FACE_AUTH_TIMEOUT_S"),
            (KioskState.FACE_AUTH_FAIL, "FACE_AUTH_FAIL_DISPLAY_S"),
            (KioskState.ALREADY_VOTED, "ALREADY_VOTED_DISPLAY_S"),
            (KioskState.OFFICE_BALLOT, "OFFICE_BALLOT_TIMEOUT_S"),
            (KioskState.REVIEW, "REVIEW_TIMEOUT_S"),
            (KioskState.THANK_YOU, "THANK_YOU_DISPLAY_S"),
        ],
    )
    def test_each_timed_state_returns_to_idle_after_its_configured_timeout(
        self, sm, clock, setup_state, timeout_attr
    ):
        from kiosk_app import config

        timeout_s = getattr(config, timeout_attr)
        sm.state = setup_state  # directly set for isolated timeout testing
        sm._state_entered_at = clock.now

        clock.advance(timeout_s - 1)
        sm.tick()
        assert sm.state == setup_state, "must not time out early"

        clock.advance(2)
        sm.tick()
        assert sm.state == KioskState.IDLE_ATTRACT

    def test_timeout_clears_in_progress_session_writing_nothing(self, sm, clock, spy_submit):
        from kiosk_app import config

        _advance_to_office_ballot(sm)
        sm.office_selection_confirmed("APC")
        clock.advance(config.OFFICE_BALLOT_TIMEOUT_S + 1)
        sm.tick()
        assert sm.state == KioskState.IDLE_ATTRACT
        assert sm.session.selections == {}
        assert spy_submit.calls == []

    def test_submitting_and_submit_failed_never_auto_timeout(self, sm, clock, spy_submit):
        spy_submit.should_fail = True
        _advance_to_office_ballot(sm)
        _select_all_offices(sm)
        sm.submit_confirmed()
        assert sm.state == KioskState.SUBMIT_FAILED

        clock.advance(10_000)  # absurdly long -- must never silently discard a failed ballot
        sm.tick()
        assert sm.state == KioskState.SUBMIT_FAILED

    def test_idle_never_times_out(self, sm, clock):
        clock.advance(10_000)
        sm.tick()
        assert sm.state == KioskState.IDLE_ATTRACT
