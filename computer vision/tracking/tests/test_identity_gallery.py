import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from identity_gallery import IdentityGallery

SAME_PERSON_A = np.array([1.0, 0.0, 0.0], dtype=np.float32)
SAME_PERSON_A_DRIFTED = np.array([0.9, 0.43589, 0.0], dtype=np.float32)  # cos sim = 0.9
DIFFERENT_PERSON_B = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def test_new_tracker_id_gets_a_fresh_persistent_id():
    gallery = IdentityGallery()

    (persistent_id,) = gallery.update([1], [SAME_PERSON_A])

    assert persistent_id == 0


def test_same_tracker_id_stays_stable_across_frames():
    gallery = IdentityGallery()

    ids_frame_1 = gallery.update([1], [SAME_PERSON_A])
    ids_frame_2 = gallery.update([1], [SAME_PERSON_A])
    ids_frame_3 = gallery.update([1], [SAME_PERSON_A])

    assert ids_frame_1 == ids_frame_2 == ids_frame_3


def test_two_simultaneous_people_get_distinct_persistent_ids():
    gallery = IdentityGallery()

    persistent_ids = gallery.update([1, 2], [SAME_PERSON_A, DIFFERENT_PERSON_B])

    assert persistent_ids[0] != persistent_ids[1]


def test_reappearing_person_with_a_new_tracker_id_reclaims_old_identity():
    gallery = IdentityGallery()

    (original_id,) = gallery.update([1], [SAME_PERSON_A])
    # Person leaves frame: tracker_id 1 is absent from this frame's update.
    gallery.update([], [])
    # StrongSort deleted the old track and minted tracker_id 7 for the same person.
    (reclaimed_id,) = gallery.update([7], [SAME_PERSON_A_DRIFTED])

    assert reclaimed_id == original_id


def test_different_person_after_departure_gets_a_new_identity():
    gallery = IdentityGallery()

    (original_id,) = gallery.update([1], [SAME_PERSON_A])
    gallery.update([], [])
    (new_id,) = gallery.update([2], [DIFFERENT_PERSON_B])

    assert new_id != original_id


def test_retired_identity_beyond_the_expiry_window_is_not_reclaimed():
    gallery = IdentityGallery(max_retired_frames=5)

    (original_id,) = gallery.update([1], [SAME_PERSON_A])
    for _ in range(10):
        gallery.update([], [])
    (new_id,) = gallery.update([7], [SAME_PERSON_A])

    assert new_id != original_id


def test_similarity_threshold_rejects_a_weak_match():
    gallery = IdentityGallery(similarity_threshold=0.99)

    (original_id,) = gallery.update([1], [SAME_PERSON_A])
    gallery.update([], [])
    # Similar but not near-identical -- should fail a strict 0.99 threshold.
    (new_id,) = gallery.update([7], [SAME_PERSON_A_DRIFTED])

    assert new_id != original_id


def test_embedding_magnitude_does_not_affect_matching():
    gallery = IdentityGallery()

    (original_id,) = gallery.update([1], [SAME_PERSON_A * 100.0])
    gallery.update([], [])
    (reclaimed_id,) = gallery.update([7], [SAME_PERSON_A * 0.01])

    assert reclaimed_id == original_id


def test_active_persistent_embedding_tracks_appearance_drift_via_ema():
    gallery = IdentityGallery(ema_alpha=0.5)

    gallery.update([1], [SAME_PERSON_A])
    gallery.update([1], [DIFFERENT_PERSON_B])

    stored = gallery._embeddings[gallery._active[1]]
    # Blend of A and B should sit strictly between the two, favouring neither exactly.
    assert 0.0 < _cosine(stored, SAME_PERSON_A) < 1.0
    assert 0.0 < _cosine(stored, DIFFERENT_PERSON_B) < 1.0


def test_zero_vector_embedding_does_not_crash():
    gallery = IdentityGallery()

    (persistent_id,) = gallery.update([1], [np.zeros(3, dtype=np.float32)])

    assert persistent_id == 0


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))
