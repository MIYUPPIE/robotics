from datetime import datetime

import numpy as np
import pytest

from biometric.capture import CapturedFrame
from biometric.embedding import DetectedFace
from contracts.schemas import LivenessResult
from registration_app.capture_flow import enroll
from registration_app.tests.conftest import make_embedding
from storage.repositories import override_repository, voter_repository

LIVE = LivenessResult(is_live=True, blink_detected=True, frames_observed=10)
NOT_LIVE = LivenessResult(
    is_live=False, blink_detected=False, frames_observed=10, reason="no blink"
)


def _captured_frames(embedding: np.ndarray, n: int = 3) -> list[CapturedFrame]:
    return [
        CapturedFrame(
            frame=np.zeros((10, 10, 3), dtype=np.uint8),
            face=DetectedFace(
                bbox=(0, 0, 100, 100), detection_confidence=0.95, embedding=embedding
            ),
        )
        for _ in range(n)
    ]


class TestEnrollHappyPath:
    def test_enrolls_a_new_voter(self, conn, valid_form):
        frames = _captured_frames(make_embedding(1))
        outcome = enroll(conn, valid_form, frames, LIVE, now=datetime(2026, 8, 1))

        assert outcome.success
        assert outcome.voter is not None
        assert voter_repository.exists(conn, outcome.voter.voter_id)

    def test_voter_id_starts_with_state_code(self, conn, valid_form):
        frames = _captured_frames(make_embedding(1))
        outcome = enroll(conn, valid_form, frames, LIVE, now=datetime(2026, 8, 1))
        assert outcome.voter.voter_id.startswith("PVR-LAG-")

    def test_stored_embedding_matches_captured(self, conn, valid_form):
        embedding = make_embedding(1)
        frames = _captured_frames(embedding)
        outcome = enroll(conn, valid_form, frames, LIVE, now=datetime(2026, 8, 1))
        stored = voter_repository.get(conn, outcome.voter.voter_id)
        np.testing.assert_array_almost_equal(stored.face_embedding, embedding, decimal=5)


class TestEnrollLivenessGate:
    def test_rejects_when_not_live(self, conn, valid_form):
        frames = _captured_frames(make_embedding(1))
        outcome = enroll(conn, valid_form, frames, NOT_LIVE, now=datetime(2026, 8, 1))
        assert not outcome.success
        assert "liveness" in outcome.reason
        assert voter_repository.count(conn) == 0

    def test_rejects_when_no_frames_captured(self, conn, valid_form):
        outcome = enroll(conn, valid_form, [], LIVE, now=datetime(2026, 8, 1))
        assert not outcome.success
        assert voter_repository.count(conn) == 0


class TestEnrollDuplicateCheck:
    def test_flags_duplicate_and_does_not_insert(self, conn, valid_form):
        embedding = make_embedding(1)
        enroll(conn, valid_form, _captured_frames(embedding), LIVE, now=datetime(2026, 8, 1))
        assert voter_repository.count(conn) == 1

        outcome = enroll(
            conn, valid_form, _captured_frames(embedding), LIVE, now=datetime(2026, 8, 1)
        )

        assert not outcome.success
        assert outcome.duplicate_voter_id is not None
        assert outcome.duplicate_similarity == pytest.approx(1.0, abs=1e-4)
        assert voter_repository.count(conn) == 1  # nothing new inserted

    def test_distinct_faces_are_not_flagged(self, conn, valid_form):
        enroll(
            conn, valid_form, _captured_frames(make_embedding(1)), LIVE, now=datetime(2026, 8, 1)
        )
        outcome = enroll(
            conn, valid_form, _captured_frames(make_embedding(2)), LIVE, now=datetime(2026, 8, 1)
        )
        assert outcome.success
        assert voter_repository.count(conn) == 2

    def test_override_enrolls_and_logs_audit_row(self, conn, valid_form):
        embedding = make_embedding(1)
        enroll(conn, valid_form, _captured_frames(embedding), LIVE, now=datetime(2026, 8, 1))

        outcome = enroll(
            conn,
            valid_form,
            _captured_frames(embedding),
            LIVE,
            now=datetime(2026, 8, 1),
            override_duplicate=True,
            officer_note="confirmed different person by officer",
        )

        assert outcome.success
        assert voter_repository.count(conn) == 2
        assert override_repository.count(conn) == 1
