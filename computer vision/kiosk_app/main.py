"""Kiosk entrypoint: the camera loop that wires real
camera/biometric/hand_ui/voice/storage to the pure state machine.

`KioskApp` takes its heavier dependencies (db connection, embedder, hand
tracker, submit function, clock) as constructor parameters so
kiosk_app/evals/full_session_eval.py can construct one with fakes and drive
`process_frame` directly -- no real camera/display/audio device needed. This
is the same dependency-injection pattern used throughout the project (e.g.
biometric/capture.py::burst_capture).
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from biometric.capture import Embedder as FaceEmbedderLike
from biometric.capture import FaceMesh as FaceMeshLike
from biometric.embedding import FaceEmbedder
from biometric.face_mesh import FaceMeshTracker
from biometric.liveness import BlinkDetector
from biometric.matcher import identify
from biometric.thresholds import LIVENESS_BLINK_WINDOW_VOTING_S
from contracts.events import CursorSample, DwellComplete, DwellEvent
from contracts.schemas import Language, Office
from hand_ui import renderer as hand_renderer
from hand_ui.dwell_fsm import DwellSelector
from hand_ui.landmarker import HandCursorTracker
from kiosk_app import config
from kiosk_app.party_logos import load_party_logos
from kiosk_app.screens import face_auth as face_auth_screen
from kiosk_app.screens import idle as idle_screen
from kiosk_app.screens import language_select as language_screen
from kiosk_app.screens import office_ballot as office_ballot_screen
from kiosk_app.screens import review as review_screen
from kiosk_app.screens import submit as submit_screen
from kiosk_app.screens import thank_you as thank_you_screen
from kiosk_app.state_machine import KioskState, KioskStateMachine
from storage.repositories import voted_log_repository, voter_repository
from storage.repositories.vote_repository import submit_vote_session as _submit_vote_session
from voice import cache as voice_cache
from voice.playback import play_phrase, play_sequence

DATA_DIR = Path(__file__).parent.parent / "data"


class HandTrackerLike(Protocol):
    def process(self, bgr_frame: np.ndarray, timestamp_ms: int) -> CursorSample: ...


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def _load_party_data() -> tuple[list[str], dict[str, tuple[int, int, int]], dict[str, np.ndarray]]:
    parties = json.loads((DATA_DIR / "parties_nigeria.json").read_text())
    acronyms = [p["acronym"] for p in parties]
    colors = {p["acronym"]: _hex_to_bgr(p["color"]) for p in parties}
    logos = load_party_logos(DATA_DIR, parties)
    return acronyms, colors, logos


class KioskApp:
    def __init__(
        self,
        db_conn: sqlite3.Connection,
        submit_fn: Callable[[str, dict[Office, str], datetime], None] | None = None,
        clock: Callable[[], datetime] = datetime.now,
        embedder: FaceEmbedderLike | None = None,
        hand_tracker: HandTrackerLike | None = None,
        face_mesh_tracker: FaceMeshLike | None = None,
        play_audio: bool = True,
        screen_width: int = config.SCREEN_WIDTH,
        screen_height: int = config.SCREEN_HEIGHT,
    ):
        self.db_conn = db_conn
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.embedder = embedder
        self.hand_tracker = hand_tracker
        self.face_mesh_tracker = face_mesh_tracker
        self.play_audio = play_audio
        self._clock = clock

        bound_submit = submit_fn or (
            lambda voter_id, selections, now: _submit_vote_session(
                db_conn, voter_id, selections, now
            )
        )
        self.state_machine = KioskStateMachine(submit_vote_session=bound_submit, clock=clock)

        self.party_acronyms, self.party_colors, self.party_logos = _load_party_data()
        self._dwell_selector: DwellSelector | None = None
        self._select_step = True  # OFFICE_BALLOT: True = picking a party, False = confirming it
        self._provisional_party: str | None = None
        self._last_state: KioskState | None = None
        self._liveness_detector = BlinkDetector()
        self._liveness_window_started_at = clock()

    # --- audio (never a live network call -- see docs/ARCHITECTURE.md ADR-3) ---

    def _speak(self, phrase_id: str) -> None:
        language = self.state_machine.session.language
        if not self.play_audio or language is None:
            return
        try:
            play_phrase(phrase_id, language)
        except voice_cache.MissingAudioError:
            pass  # audio cache incomplete -- degrade to silent rather than crash

    def _speak_sequence(self, phrase_ids: list[str]) -> None:
        language = self.state_machine.session.language
        if not self.play_audio or language is None:
            return
        try:
            play_sequence(phrase_ids, language)
        except voice_cache.MissingAudioError:
            pass

    # --- dwell-selector lifecycle ---

    def _rebuild_dwell_selector_for_current_state(self) -> None:
        sm = self.state_machine
        if sm.state == KioskState.LANGUAGE_SELECT:
            regions = language_screen.build_regions(self.screen_width, self.screen_height)
        elif sm.state == KioskState.OFFICE_BALLOT:
            self._select_step = True
            self._provisional_party = None
            regions = office_ballot_screen.build_party_regions(
                self.party_acronyms, self.screen_width, self.screen_height
            )
        elif sm.state == KioskState.REVIEW:
            regions = review_screen.build_regions(
                list(Office), self.screen_width, self.screen_height
            )
        elif sm.state == KioskState.SUBMIT_FAILED:
            regions = [submit_screen.build_retry_region(self.screen_width, self.screen_height)]
        else:
            regions = []
        self._dwell_selector = DwellSelector(regions) if regions else None

    def _handle_dwell_events(self, events: list[DwellEvent]) -> dict[str, float]:
        hover_fraction: dict[str, float] = {}
        for event in events:
            if hasattr(event, "fraction"):
                hover_fraction[event.region_id] = event.fraction
            if isinstance(event, DwellComplete):
                self._handle_dwell_complete(event.region_id)
        return hover_fraction

    def _handle_dwell_complete(self, region_id: str) -> None:
        sm = self.state_machine
        if sm.state == KioskState.LANGUAGE_SELECT:
            sm.language_selected(Language(region_id))
            self._speak("LANGUAGE_CONFIRMED")
        elif sm.state == KioskState.OFFICE_BALLOT:
            self._handle_office_ballot_dwell(region_id)
        elif sm.state == KioskState.REVIEW:
            self._handle_review_dwell(region_id)
        elif sm.state == KioskState.SUBMIT_FAILED:
            if region_id == submit_screen.RETRY_REGION_ID:
                sm.retry_submit()

    def _handle_office_ballot_dwell(self, region_id: str) -> None:
        sm = self.state_machine
        if self._select_step:
            if region_id in self.party_acronyms:
                self._provisional_party = region_id
                self._select_step = False
                office = sm.session.current_office()
                confirm_regions = office_ballot_screen.build_confirm_regions(
                    self.screen_width, self.screen_height
                )
                self._dwell_selector = DwellSelector(confirm_regions)
                self._speak_sequence(
                    [
                        "SELECTION_MADE_PREFIX",
                        f"PARTY_NAME_{region_id}",
                        "OFFICE_FOR_PREFIX",
                        f"OFFICE_NAME_{office.value}",
                        "SELECTION_CONFIRM_PROMPT",
                    ]
                )
        else:
            if region_id == office_ballot_screen.CONFIRM_REGION_ID and self._provisional_party:
                sm.office_selection_confirmed(self._provisional_party)
                self._speak("SELECTION_CONFIRMED")
                self._rebuild_dwell_selector_for_current_state()
            elif region_id == office_ballot_screen.CHANGE_REGION_ID:
                self._select_step = True
                self._provisional_party = None
                regions = office_ballot_screen.build_party_regions(
                    self.party_acronyms, self.screen_width, self.screen_height
                )
                self._dwell_selector = DwellSelector(regions)

    def _handle_review_dwell(self, region_id: str) -> None:
        sm = self.state_machine
        if review_screen.is_edit_region(region_id):
            sm.review_edit_requested(review_screen.office_from_edit_region(region_id))
        elif region_id == review_screen.SUBMIT_REGION_ID:
            confirm_regions = review_screen.build_submit_confirm_regions(
                self.screen_width, self.screen_height
            )
            self._dwell_selector = DwellSelector(confirm_regions)
        elif region_id == review_screen.SUBMIT_CONFIRM_REGION_ID:
            sm.submit_confirmed()
            if sm.state == KioskState.THANK_YOU:
                self._speak("SUBMIT_SUCCESS")
                self._speak("THANK_YOU")
            elif sm.state == KioskState.SUBMIT_FAILED:
                self._speak("SUBMIT_FAILED_RETRY")
        elif region_id == review_screen.SUBMIT_CANCEL_REGION_ID:
            self._rebuild_dwell_selector_for_current_state()
        elif region_id == review_screen.START_OVER_REGION_ID:
            sm.review_start_over_confirmed()

    # --- per-frame orchestration ---

    def process_frame(self, frame: np.ndarray, timestamp_ms: int) -> np.ndarray:
        """One frame of the main loop: run inference relevant to the current
        state, feed the dwell selector, render the current screen, and tick
        timeouts. Mutates and returns `frame`."""
        sm = self.state_machine

        if sm.state != self._last_state:
            print(
                f"[kiosk] {self._last_state.name if self._last_state else 'START'} "
                f"-> {sm.state.name}"
            )
            self._rebuild_dwell_selector_for_current_state()
            if sm.state == KioskState.FACE_AUTH:
                self._liveness_detector = BlinkDetector()
                self._liveness_window_started_at = self._clock()
            self._last_state = sm.state

        if sm.state == KioskState.IDLE_ATTRACT:
            idle_screen.render(frame)
            if self.hand_tracker is not None:
                cursor = self.hand_tracker.process(frame, timestamp_ms)
                if cursor.present:
                    sm.presence_detected()
        elif sm.state == KioskState.FACE_AUTH:
            self._run_face_auth(frame, timestamp_ms)
        elif sm.state in (
            KioskState.LANGUAGE_SELECT,
            KioskState.OFFICE_BALLOT,
            KioskState.REVIEW,
            KioskState.SUBMIT_FAILED,
        ):
            self._run_hand_driven_screen(frame, timestamp_ms)
        elif sm.state == KioskState.FACE_AUTH_FAIL:
            face_auth_screen.render(frame, "Not recognized -- please see an official")
        elif sm.state == KioskState.ALREADY_VOTED:
            face_auth_screen.render(frame, "Our records show you have already voted")
        elif sm.state == KioskState.SUBMITTING:
            submit_screen.render_submitting(frame)
        elif sm.state == KioskState.THANK_YOU:
            thank_you_screen.render(frame)

        sm.tick()
        return frame

    def _run_hand_driven_screen(self, frame: np.ndarray, timestamp_ms: int) -> None:
        sm = self.state_machine
        if self.hand_tracker is not None:
            cursor = self.hand_tracker.process(frame, timestamp_ms)
        else:
            cursor = CursorSample(0.0, 0.0, timestamp_ms / 1000.0, present=False)

        hover_fraction: dict[str, float] = {}
        if self._dwell_selector is not None:
            events = self._dwell_selector.update(cursor)
            hover_fraction = self._handle_dwell_events(events)

        if cursor.present:
            hand_renderer.draw_cursor(frame, cursor.x, cursor.y)

        # `sm.state` may have just changed (a dwell event handled above can
        # advance the state machine) -- read it fresh for rendering.
        if sm.state == KioskState.LANGUAGE_SELECT:
            regions = language_screen.build_regions(self.screen_width, self.screen_height)
            language_screen.render(frame, regions, hover_fraction)
        elif sm.state == KioskState.OFFICE_BALLOT:
            office = sm.session.current_office()
            if self._select_step:
                regions = office_ballot_screen.build_party_regions(
                    self.party_acronyms, self.screen_width, self.screen_height
                )
                office_ballot_screen.render_select_step(
                    frame,
                    regions,
                    self.party_colors,
                    self.party_logos,
                    hover_fraction,
                    office.value,
                )
            else:
                regions = office_ballot_screen.build_confirm_regions(
                    self.screen_width, self.screen_height
                )
                office_ballot_screen.render_confirm_step(
                    frame,
                    regions,
                    self._provisional_party or "",
                    self.party_logos,
                    hover_fraction,
                )
        elif sm.state == KioskState.REVIEW:
            regions = review_screen.build_regions(
                list(Office), self.screen_width, self.screen_height
            )
            review_screen.render(
                frame, list(Office), sm.session.selections, regions, hover_fraction
            )
        elif sm.state == KioskState.SUBMIT_FAILED:
            retry_region = submit_screen.build_retry_region(self.screen_width, self.screen_height)
            submit_screen.render_submit_failed(
                frame, retry_region, hover_fraction.get(submit_screen.RETRY_REGION_ID, 0.0)
            )

    def _run_face_auth(self, frame: np.ndarray, timestamp_ms: int) -> None:
        """Liveness-gated identification: only runs the (slower) embedder +
        1:N identify once a blink has been observed in the current
        LIVENESS_BLINK_WINDOW_VOTING_S window. A window with no blink counts
        as a failed liveness attempt (see state_machine.face_auth_liveness_failed
        / biometric/thresholds.MAX_LIVENESS_RETRIES)."""
        sm = self.state_machine
        face_auth_screen.render(frame, "Please look at the camera and blink")
        if self.embedder is None or self.face_mesh_tracker is None:
            return  # no real models wired -- render-only (e.g. a smoke test)

        landmarks = self.face_mesh_tracker.process(frame, timestamp_ms)
        if landmarks is not None:
            self._liveness_detector.update(landmarks)

        if not self._liveness_detector.result().is_live:
            elapsed = (self._clock() - self._liveness_window_started_at).total_seconds()
            if elapsed >= LIVENESS_BLINK_WINDOW_VOTING_S:
                sm.face_auth_liveness_failed()
                self._liveness_detector = BlinkDetector()
                self._liveness_window_started_at = self._clock()
                if sm.state == KioskState.FACE_AUTH_FAIL:
                    self._speak("FACE_AUTH_FAIL")
            return

        faces = self.embedder.detect(frame)
        if len(faces) != 1:
            return

        gallery = voter_repository.all_embeddings(self.db_conn)
        result = identify(faces[0].embedding, gallery)

        if not result.confident:
            sm.face_auth_result(result, has_voted=False)
            self._speak("FACE_AUTH_AMBIGUOUS" if result.matched else "FACE_AUTH_FAIL")
            return

        assert result.voter_id is not None
        already_voted = voted_log_repository.has_voted(self.db_conn, result.voter_id)
        sm.face_auth_result(result, has_voted=already_voted)
        if sm.state == KioskState.ALREADY_VOTED:
            self._speak("ALREADY_VOTED")
        elif sm.state == KioskState.OFFICE_BALLOT:
            self._speak("FACE_AUTH_SUCCESS")


def run() -> None:  # pragma: no cover -- real-hardware entrypoint, not gate-tested
    import traceback

    from storage import db

    missing = voice_cache.verify_complete()
    if missing:
        raise SystemExit(
            f"Audio cache incomplete ({len(missing)} missing) -- run `make prerender-audio` first."
        )

    db_path = db.default_db_path()
    print(f"[kiosk] using database: {db_path}")
    conn = db.connect(db_path)
    embedder = FaceEmbedder()
    models_dir = Path(__file__).parent.parent / "models" / "mediapipe"
    hand_tracker = HandCursorTracker(
        models_dir / "hand_landmarker.task", config.SCREEN_WIDTH, config.SCREEN_HEIGHT
    )
    face_mesh_tracker = FaceMeshTracker(models_dir / "face_landmarker.task")

    app = KioskApp(
        conn, embedder=embedder, hand_tracker=hand_tracker, face_mesh_tracker=face_mesh_tracker
    )

    cap = cv2.VideoCapture(0)
    window_name = "Voter Kiosk"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    start = time.monotonic()
    consecutive_read_failures = 0
    warned_about_camera = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                consecutive_read_failures += 1
                # ~2s of unbroken failures at a typical 30fps camera -- print
                # once so a stalled/disconnected camera is visible in the
                # terminal instead of just a frozen window with no clue why.
                if consecutive_read_failures == 60 and not warned_about_camera:
                    print(
                        "[kiosk] WARNING: camera.read() has failed 60 times in a row -- "
                        "the camera may be disconnected, busy, or stalled. The window "
                        "will appear frozen until this recovers."
                    )
                    warned_about_camera = True
                continue
            if consecutive_read_failures:
                print(f"[kiosk] camera recovered after {consecutive_read_failures} failed read(s)")
            consecutive_read_failures = 0
            warned_about_camera = False

            frame = cv2.resize(frame, (config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            # Mirror once, here, so the whole pipeline (hand tracking, face
            # auth, and what's drawn on screen) is consistently in "mirror"
            # orientation -- see hand_ui/landmarker.py's module docstring.
            frame = cv2.flip(frame, 1)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            try:
                app.process_frame(frame, timestamp_ms)
            except Exception:
                sm = app.state_machine
                print(
                    f"[kiosk] CRASH while processing a frame in state={sm.state.name} "
                    f"voter_id={sm.session.voter_id!r} selections={sm.session.selections!r}"
                )
                traceback.print_exc()
                raise
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC: officer override, writes nothing
                app.state_machine.officer_reset()
            elif key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_tracker.close()
        face_mesh_tracker.close()
        conn.close()


if __name__ == "__main__":  # pragma: no cover
    run()
