"""The kiosk voting-session state machine -- pure, deterministic core.

No camera, MediaPipe, audio, or direct database dependency: transitions are
driven by explicit method calls (language_selected, face_auth_result, ...)
and an injected clock for timeouts. kiosk_app/main.py is the thin shell that
wires real camera/biometric/hand_ui/voice/storage calls to these methods --
see docs/ARCHITECTURE.md#deterministic-core-vs-thin-cv-shell.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from enum import Enum, auto

from biometric.thresholds import MAX_LIVENESS_RETRIES
from contracts.schemas import OFFICE_ORDER, Language, MatchResult, Office
from kiosk_app import config
from kiosk_app.session import VotingSession
from storage.repositories.vote_repository import VoteSubmissionError

SubmitFn = Callable[[str, dict[Office, str], datetime], None]
ClockFn = Callable[[], datetime]


class KioskState(Enum):
    IDLE_ATTRACT = auto()
    LANGUAGE_SELECT = auto()
    FACE_AUTH = auto()
    FACE_AUTH_FAIL = auto()
    ALREADY_VOTED = auto()
    OFFICE_BALLOT = auto()
    REVIEW = auto()
    SUBMITTING = auto()
    SUBMIT_FAILED = auto()
    THANK_YOU = auto()


# States that auto-return to IDLE_ATTRACT after a fixed timeout. SUBMITTING
# and SUBMIT_FAILED are deliberately absent -- see the abandon-safety
# invariant in docs/ARCHITECTURE.md (never silently discard a ballot that's
# mid-submit or that failed to submit).
_TIMEOUTS: dict[KioskState, float] = {
    KioskState.LANGUAGE_SELECT: config.LANGUAGE_SELECT_TIMEOUT_S,
    KioskState.FACE_AUTH: config.FACE_AUTH_TIMEOUT_S,
    KioskState.FACE_AUTH_FAIL: config.FACE_AUTH_FAIL_DISPLAY_S,
    KioskState.ALREADY_VOTED: config.ALREADY_VOTED_DISPLAY_S,
    KioskState.OFFICE_BALLOT: config.OFFICE_BALLOT_TIMEOUT_S,
    KioskState.REVIEW: config.REVIEW_TIMEOUT_S,
    KioskState.THANK_YOU: config.THANK_YOU_DISPLAY_S,
}


class KioskStateMachine:
    def __init__(self, submit_vote_session: SubmitFn, clock: ClockFn):
        """`submit_vote_session` should be
        storage.repositories.vote_repository.submit_vote_session bound to a
        live connection: (voter_id, selections, now) -> None, raising
        VoteSubmissionError on failure. Injected so this class never touches
        sqlite3/a connection directly.
        """
        self._submit_vote_session = submit_vote_session
        self._clock = clock
        self.session = VotingSession()
        self.state = KioskState.IDLE_ATTRACT
        self._state_entered_at = self._clock()

    def _enter(self, state: KioskState) -> None:
        self.state = state
        self._state_entered_at = self._clock()

    def _reset_to_idle(self) -> None:
        self.session.reset()
        self._enter(KioskState.IDLE_ATTRACT)

    # --- transitions triggered by real-world events ---

    def presence_detected(self) -> None:
        if self.state == KioskState.IDLE_ATTRACT:
            self._enter(KioskState.LANGUAGE_SELECT)

    def language_selected(self, language: Language) -> None:
        if self.state != KioskState.LANGUAGE_SELECT:
            return
        self.session.language = language
        self._enter(KioskState.FACE_AUTH)

    def face_auth_liveness_failed(self) -> None:
        """Call each time a liveness check fails during FACE_AUTH. After
        MAX_LIVENESS_RETRIES failures, gives up and shows FACE_AUTH_FAIL;
        otherwise stays in FACE_AUTH so the caller can retry capture."""
        if self.state != KioskState.FACE_AUTH:
            return
        self.session.liveness_retry_count += 1
        if self.session.liveness_retry_count > MAX_LIVENESS_RETRIES:
            self._enter(KioskState.FACE_AUTH_FAIL)

    def face_auth_result(self, result: MatchResult, has_voted: bool) -> None:
        """`result.confident` covers both "no match" and "ambiguous match"
        (never guesses, see contracts.schemas.MatchResult) -- both land on
        FACE_AUTH_FAIL; the screen layer picks the exact spoken message."""
        if self.state != KioskState.FACE_AUTH:
            return
        if not result.confident:
            self._enter(KioskState.FACE_AUTH_FAIL)
            return
        if has_voted:
            self._enter(KioskState.ALREADY_VOTED)
            return
        self.session.voter_id = result.voter_id
        self.session.current_office_index = 0
        self._enter(KioskState.OFFICE_BALLOT)

    def office_selection_confirmed(self, party_code: str) -> None:
        """Call once the office_ballot screen's two-step select-then-confirm
        dwell flow completes for the current office."""
        if self.state != KioskState.OFFICE_BALLOT:
            return
        office = self.session.current_office()
        self.session.selections[office] = party_code

        if self.session.return_to_review:
            self.session.return_to_review = False
            self._enter(KioskState.REVIEW)
            return

        if self.session.is_last_office():
            self._enter(KioskState.REVIEW)
        else:
            self.session.current_office_index += 1
            self._enter(KioskState.OFFICE_BALLOT)

    def review_edit_requested(self, office: Office) -> None:
        if self.state != KioskState.REVIEW:
            return
        self.session.current_office_index = OFFICE_ORDER.index(office)
        self.session.return_to_review = True
        self._enter(KioskState.OFFICE_BALLOT)

    def review_start_over_confirmed(self) -> None:
        if self.state != KioskState.REVIEW:
            return
        self._reset_to_idle()

    def submit_confirmed(self) -> None:
        """Call once REVIEW's two-step select-then-confirm dwell flow
        completes on SUBMIT. No-ops if not every office has a selection yet
        (should be unreachable via the UI, but never trust that alone)."""
        if self.state != KioskState.REVIEW:
            return
        if not self.session.all_offices_selected():
            return
        self._enter(KioskState.SUBMITTING)
        self._do_submit()

    def retry_submit(self) -> None:
        if self.state != KioskState.SUBMIT_FAILED:
            return
        self._enter(KioskState.SUBMITTING)
        self._do_submit()

    def _do_submit(self) -> None:
        assert self.session.voter_id is not None, "cannot submit without an authenticated voter"
        try:
            self._submit_vote_session(
                self.session.voter_id, dict(self.session.selections), self._clock()
            )
        except VoteSubmissionError as exc:
            print(f"[kiosk] vote submission failed: {exc}")
            self._enter(KioskState.SUBMIT_FAILED)
            return
        except Exception as exc:
            # Last-resort safety net: submit_vote_session is expected to only
            # ever raise VoteSubmissionError, but a voting kiosk must never
            # crash into an undefined state on an unanticipated error --
            # degrade to the same officer-recoverable SUBMIT_FAILED screen
            # instead, and print loudly since this path "should never happen."
            print(f"[kiosk] UNEXPECTED error during vote submission: {exc!r}")
            self._enter(KioskState.SUBMIT_FAILED)
            return
        self._enter(KioskState.THANK_YOU)

    def officer_reset(self) -> None:
        """ESC override: force back to idle without writing anything, from
        any state. The keyboard is physically inaccessible to the voter in
        the gesture-only kiosk UI -- reserved for poll officials."""
        self._reset_to_idle()

    def tick(self) -> None:
        """Call once per frame; handles state-timeout expiry."""
        timeout = _TIMEOUTS.get(self.state)
        if timeout is None:
            return
        elapsed = (self._clock() - self._state_entered_at).total_seconds()
        if elapsed >= timeout:
            self._reset_to_idle()
