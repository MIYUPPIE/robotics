"""Registration capture flow: burst-capture + liveness + duplicate-check +
enroll, composed from biometric/*'s dependency-injected pieces so this
orchestration is gate-tested with fakes -- no real camera or InsightFace
model needed (see registration_app/tests/test_capture_flow.py).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime

from biometric.capture import CapturedFrame, average_embedding, best_frames
from biometric.embedding import MODEL_VERSION
from biometric.matcher import find_duplicates
from biometric.thresholds import EMBEDDING_FRAMES_TO_AVERAGE
from biometric.voter_id import generate_voter_id
from contracts.schemas import LivenessResult, VoterRecord
from registration_app.form import RegistrationFormData
from storage.repositories import override_repository, voter_repository


@dataclass(frozen=True)
class EnrollmentOutcome:
    success: bool
    voter: VoterRecord | None = None
    duplicate_voter_id: str | None = None
    duplicate_similarity: float = 0.0
    reason: str = ""


def enroll(
    conn: sqlite3.Connection,
    form: RegistrationFormData,
    captured_frames: list[CapturedFrame],
    liveness_result: LivenessResult,
    now: datetime,
    override_duplicate: bool = False,
    officer_note: str = "",
) -> EnrollmentOutcome:
    """The full registration decision path, given already-captured biometric
    inputs. Pure orchestration over biometric/storage functions -- no camera
    or model calls happen here, so it's fully gate-testable with fakes.
    """
    if not liveness_result.is_live:
        return EnrollmentOutcome(success=False, reason="liveness check failed")
    if not captured_frames:
        return EnrollmentOutcome(success=False, reason="no qualifying face frames captured")

    best = best_frames(captured_frames, EMBEDDING_FRAMES_TO_AVERAGE)
    embedding = average_embedding(best)

    gallery = voter_repository.all_embeddings(conn)
    duplicate = find_duplicates(embedding, gallery)
    if duplicate.is_duplicate and not override_duplicate:
        return EnrollmentOutcome(
            success=False,
            duplicate_voter_id=duplicate.matched_voter_id,
            duplicate_similarity=duplicate.similarity,
            reason="duplicate face match",
        )

    voter_id = generate_voter_id(
        form.state_code, now.date(), exists=lambda vid: voter_repository.exists(conn, vid)
    )
    assert form.dob is not None and form.gender is not None and form.language_pref is not None

    voter = VoterRecord(
        voter_id=voter_id,
        surname=form.surname.strip(),
        first_name=form.first_name.strip(),
        middle_name=(form.middle_name.strip() or None),
        dob=form.dob,
        gender=form.gender,
        state_code=form.state_code,
        lga=form.lga.strip(),
        phone=(form.phone.strip() or None),
        language_pref=form.language_pref,
        face_embedding=embedding,
        embedding_model_version=MODEL_VERSION,
        registered_at=now,
    )
    voter_repository.insert(conn, voter)

    if duplicate.is_duplicate and override_duplicate:
        assert duplicate.matched_voter_id is not None
        override_repository.log_override(
            conn, voter_id, duplicate.matched_voter_id, duplicate.similarity, officer_note, now
        )

    return EnrollmentOutcome(success=True, voter=voter)
