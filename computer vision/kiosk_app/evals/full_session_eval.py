"""Full-session dry-run eval.

Scripts N complete voting sessions -- a fake face-mesh blink stream, a fake
biometric identify result, and a scripted hand-tracker cursor path computed
from the SAME region-layout functions kiosk_app/main.py itself uses -- through
the real KioskApp orchestration and a real in-memory database. Asserts
exactly 5N votes and N voted_log rows land (the concrete, executable form of
the docs/ARCHITECTURE.md#ballot-secrecy write-path invariant), and that a
replayed session for an already-voted voter is blocked and adds no votes.

This exercises kiosk_app/main.py's real orchestration end to end without a
real camera/display/audio device: only the "sensors" (hand tracker, face
mesh, face embedder) are faked. The state machine, dwell selector, screen
region math, and storage layer are all the real production code.

Usage: python -m kiosk_app.evals.full_session_eval
"""

from __future__ import annotations

import sys
from datetime import date, datetime

import numpy as np

from biometric.embedding import DetectedFace
from biometric.liveness import LEFT_EYE_EAR_INDICES, RIGHT_EYE_EAR_INDICES
from biometric.voter_id import generate_voter_id
from contracts.events import CursorSample, Region
from contracts.schemas import EMBEDDING_DIM, Gender, Language, Office, VoterRecord
from hand_ui.dwell_fsm import DEFAULT_DWELL_SECONDS
from kiosk_app import config
from kiosk_app.main import KioskApp, _load_party_data
from kiosk_app.screens import language_select as language_screen
from kiosk_app.screens import office_ballot as office_ballot_screen
from kiosk_app.screens import review as review_screen
from kiosk_app.state_machine import KioskState
from storage import db
from storage.repositories import vote_repository, voted_log_repository, voter_repository

N_SESSIONS = 3
FRAME_DT_S = 1.0 / 30.0
MAX_FACE_AUTH_FRAMES = 200
_BREAK_STATES = {
    KioskState.THANK_YOU,
    KioskState.SUBMIT_FAILED,
    KioskState.ALREADY_VOTED,
    KioskState.FACE_AUTH_FAIL,
}


# --- fakes for the two "sensors" KioskApp otherwise gets from real hardware ---


class FakeEmbedder:
    def __init__(self, embedding: np.ndarray):
        self.embedding = embedding

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        return [
            DetectedFace(
                bbox=(10.0, 10.0, 100.0, 100.0), detection_confidence=0.99, embedding=self.embedding
            )
        ]


class ScriptedHandTracker:
    def __init__(self, samples: list[CursorSample]):
        self._samples = iter(samples)

    def process(self, frame: np.ndarray, timestamp_ms: int) -> CursorSample:
        return next(self._samples, CursorSample(0.0, 0.0, timestamp_ms / 1000.0, present=False))


def _eye_points(ear: float, width: float = 1.0) -> np.ndarray:
    h = ear * width / 2.0
    return np.array(
        [
            (0.0, 0.0),
            (0.25 * width, h),
            (0.75 * width, h),
            (width, 0.0),
            (0.75 * width, -h),
            (0.25 * width, -h),
        ]
    )


def _face_landmarks(left_ear: float, right_ear: float, n_points: int = 478) -> np.ndarray:
    landmarks = np.zeros((n_points, 2))
    for idx, pt in zip(LEFT_EYE_EAR_INDICES, _eye_points(left_ear), strict=True):
        landmarks[idx] = pt
    for idx, pt in zip(RIGHT_EYE_EAR_INDICES, _eye_points(right_ear), strict=True):
        landmarks[idx] = pt
    return landmarks


class FakeFaceMesh:
    """A quick, always-resolving blink stream so the eval can focus on
    ballot/session-write correctness rather than liveness timing."""

    def __init__(self):
        self._script = [
            _face_landmarks(0.30, 0.30),
            _face_landmarks(0.30, 0.30),
            _face_landmarks(0.10, 0.10),
            _face_landmarks(0.10, 0.10),
            _face_landmarks(0.30, 0.30),
            _face_landmarks(0.30, 0.30),
        ]
        self._i = 0

    def process(self, frame: np.ndarray, timestamp_ms: int) -> np.ndarray:
        landmarks = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return landmarks


# --- script builders, sharing the real screens' region-layout functions ---


def _dwell_block(
    region: Region, start_t: float, dwell_seconds: float = DEFAULT_DWELL_SECONDS
) -> tuple[list[CursorSample], float]:
    cx, cy = (region.x0 + region.x1) / 2, (region.y0 + region.y1) / 2
    n = int(dwell_seconds / FRAME_DT_S) + 3
    samples = [CursorSample(cx, cy, start_t + i * FRAME_DT_S, present=True) for i in range(n)]
    return samples, start_t + n * FRAME_DT_S


def _language_block(start_t: float) -> tuple[list[CursorSample], float]:
    lang_regions = language_screen.build_regions(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
    en_region = next(r for r in lang_regions if r.region_id == Language.ENGLISH.value)
    return _dwell_block(en_region, start_t)


def _office_blocks(
    party_acronyms: list[str], target_party: str, start_t: float
) -> tuple[list[CursorSample], float]:
    party_regions = office_ballot_screen.build_party_regions(
        party_acronyms, config.SCREEN_WIDTH, config.SCREEN_HEIGHT
    )
    target_region = next(r for r in party_regions if r.region_id == target_party)
    select_samples, t = _dwell_block(target_region, start_t)

    confirm_regions = office_ballot_screen.build_confirm_regions(
        config.SCREEN_WIDTH, config.SCREEN_HEIGHT
    )
    confirm_region = next(
        r for r in confirm_regions if r.region_id == office_ballot_screen.CONFIRM_REGION_ID
    )
    confirm_samples, t = _dwell_block(confirm_region, t)
    return select_samples + confirm_samples, t


def _review_submit_blocks(start_t: float) -> tuple[list[CursorSample], float]:
    review_regions = review_screen.build_regions(
        list(Office), config.SCREEN_WIDTH, config.SCREEN_HEIGHT
    )
    submit_region = next(r for r in review_regions if r.region_id == review_screen.SUBMIT_REGION_ID)
    submit_samples, t = _dwell_block(submit_region, start_t)

    submit_confirm_regions = review_screen.build_submit_confirm_regions(
        config.SCREEN_WIDTH, config.SCREEN_HEIGHT
    )
    submit_confirm_region = next(
        r for r in submit_confirm_regions if r.region_id == review_screen.SUBMIT_CONFIRM_REGION_ID
    )
    confirm_samples, t = _dwell_block(submit_confirm_region, t)
    return submit_samples + confirm_samples, t


def _feed(app: KioskApp, frame: np.ndarray, samples: list[CursorSample]) -> None:
    for sample in samples:
        app.process_frame(frame, int(sample.t * 1000))


def _make_embedding(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def run_one_session(
    conn, voter: VoterRecord, target_party: str, party_acronyms: list[str]
) -> KioskState:
    embedder = FakeEmbedder(voter.face_embedding)
    app = KioskApp(
        conn,
        embedder=embedder,
        hand_tracker=None,
        face_mesh_tracker=FakeFaceMesh(),
        play_audio=False,
    )
    frame = np.zeros((config.SCREEN_HEIGHT, config.SCREEN_WIDTH, 3), dtype=np.uint8)
    t = 0.0

    # IDLE_ATTRACT -> LANGUAGE_SELECT
    app.hand_tracker = ScriptedHandTracker([CursorSample(500.0, 500.0, t, present=True)])
    app.process_frame(frame, int(t * 1000))
    t += FRAME_DT_S

    # LANGUAGE_SELECT -> FACE_AUTH
    block, t = _language_block(t)
    app.hand_tracker = ScriptedHandTracker(block)
    _feed(app, frame, block)

    # FACE_AUTH -> OFFICE_BALLOT / FACE_AUTH_FAIL / ALREADY_VOTED (no cursor needed)
    app.hand_tracker = None
    for _ in range(MAX_FACE_AUTH_FRAMES):
        if app.state_machine.state != KioskState.FACE_AUTH:
            break
        app.process_frame(frame, int(t * 1000))
        t += FRAME_DT_S
    if app.state_machine.state in _BREAK_STATES:
        return app.state_machine.state

    # OFFICE_BALLOT x5 -> REVIEW
    for _office in list(Office):
        block, t = _office_blocks(party_acronyms, target_party, t)
        app.hand_tracker = ScriptedHandTracker(block)
        _feed(app, frame, block)
        if app.state_machine.state in _BREAK_STATES:
            return app.state_machine.state

    # REVIEW -> SUBMITTING -> THANK_YOU / SUBMIT_FAILED
    block, t = _review_submit_blocks(t)
    app.hand_tracker = ScriptedHandTracker(block)
    _feed(app, frame, block)

    return app.state_machine.state


def _make_voter(index: int, conn) -> VoterRecord:
    voter_id = generate_voter_id(
        "LAG", date(2026, 8, 1), exists=lambda vid: voter_repository.exists(conn, vid)
    )
    return VoterRecord(
        voter_id=voter_id,
        surname=f"Test{index}",
        first_name="Voter",
        middle_name=None,
        dob=date(2000, 1, 1),
        gender=Gender.OTHER,
        state_code="LAG",
        lga="Ikeja",
        phone=None,
        language_pref=Language.ENGLISH,
        face_embedding=_make_embedding(seed=index),
        embedding_model_version="synthetic-eval",
        registered_at=datetime(2026, 1, 1),
    )


def main() -> int:
    conn = db.connect_memory()
    party_acronyms, _, _ = _load_party_data()
    target_parties = ["APC", "PDP", "LP"]

    voters: list[VoterRecord] = []
    all_ok = True
    for i in range(N_SESSIONS):
        voter = _make_voter(i, conn)
        voter_repository.insert(conn, voter)
        voters.append(voter)

        final_state = run_one_session(
            conn, voter, target_parties[i % len(target_parties)], party_acronyms
        )
        print(f"session {i}: voter={voter.voter_id} final_state={final_state.name}")
        if final_state != KioskState.THANK_YOU:
            print(f"FAIL: session {i} did not reach THANK_YOU", file=sys.stderr)
            all_ok = False

    total_votes = vote_repository.total_votes(conn)
    total_voted = voted_log_repository.count(conn)
    expected_votes = N_SESSIONS * len(list(Office))
    tally_ok = total_votes == expected_votes and total_voted == N_SESSIONS
    print(
        f"\nTally check: {total_votes} votes (expected {expected_votes}), "
        f"{total_voted} voted_log rows (expected {N_SESSIONS}) "
        f"({'PASS' if tally_ok else 'FAIL'})"
    )

    replay_state = run_one_session(conn, voters[0], "PDP", party_acronyms)
    replay_blocked = replay_state == KioskState.ALREADY_VOTED
    print(
        f"Double-vote block check: replaying voter 0 -> {replay_state.name} "
        f"({'PASS' if replay_blocked else 'FAIL'})"
    )
    replay_did_not_add_votes = vote_repository.total_votes(conn) == expected_votes
    print(
        f"Replay wrote no extra votes: {replay_did_not_add_votes} "
        f"({'PASS' if replay_did_not_add_votes else 'FAIL'})"
    )

    passed = all_ok and tally_ok and replay_blocked and replay_did_not_add_votes
    conn.close()
    print(f"\nOverall: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
