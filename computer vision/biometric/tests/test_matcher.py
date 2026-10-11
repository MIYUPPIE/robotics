import numpy as np
import pytest

from biometric.matcher import cosine_similarity, find_duplicates, identify


class TestCosineSimilarity:
    def test_identical_vectors_similarity_one(self, make_embedding):
        v = make_embedding(0)
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors_similarity_minus_one(self, make_embedding):
        v = make_embedding(0)
        assert cosine_similarity(v, -v) == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector_is_safe(self, make_embedding):
        v = make_embedding(0)
        assert cosine_similarity(v, np.zeros_like(v)) == 0.0


class TestFindDuplicates:
    def test_empty_gallery_is_never_a_duplicate(self, make_embedding):
        result = find_duplicates(make_embedding(0), gallery=[])
        assert not result.is_duplicate
        assert result.matched_voter_id is None

    def test_exact_match_is_a_duplicate(self, make_embedding):
        existing = make_embedding(1)
        gallery = [("PVR-A", existing), ("PVR-B", make_embedding(2))]
        result = find_duplicates(existing.copy(), gallery)
        assert result.is_duplicate
        assert result.matched_voter_id == "PVR-A"
        assert result.similarity == pytest.approx(1.0, abs=1e-6)

    def test_distinct_random_embedding_is_not_a_duplicate(self, make_embedding):
        gallery = [("PVR-A", make_embedding(1)), ("PVR-B", make_embedding(2))]
        # High-dimensional random unit vectors are nearly orthogonal, well
        # below the duplicate threshold.
        result = find_duplicates(make_embedding(999), gallery)
        assert not result.is_duplicate


class TestIdentify:
    def test_no_gallery_never_matches(self, make_embedding):
        result = identify(make_embedding(0), gallery=[])
        assert not result.matched
        assert not result.ambiguous
        assert result.voter_id is None

    def test_confident_match_when_clearly_closest(self, make_embedding):
        target = make_embedding(1)
        gallery = [
            ("PVR-A", target),
            ("PVR-B", make_embedding(2)),
            ("PVR-C", make_embedding(3)),
        ]
        result = identify(target.copy(), gallery)
        assert result.matched
        assert result.confident
        assert not result.ambiguous
        assert result.voter_id == "PVR-A"

    def test_unrelated_query_does_not_match(self, make_embedding):
        gallery = [("PVR-A", make_embedding(1)), ("PVR-B", make_embedding(2))]
        result = identify(make_embedding(999), gallery)
        assert not result.matched
        assert not result.confident

    def test_near_twins_are_ambiguous_never_guesses(self, make_embedding):
        a = make_embedding(1)
        # A near-twin: mostly `a` with a small amount of noise mixed in, then
        # re-normalized -- cosine similarity to `a` stays very high.
        noise = make_embedding(2)
        b = a * 0.98 + noise * 0.02
        b = b / np.linalg.norm(b)

        gallery = [("PVR-A", a), ("PVR-B", b)]
        result = identify(a.copy(), gallery)

        assert result.matched
        assert result.ambiguous
        assert not result.confident

    def test_margin_reported_for_a_clear_single_entry_gallery(self, make_embedding):
        target = make_embedding(1)
        result = identify(target.copy(), gallery=[("PVR-A", target)])
        assert result.matched
        assert not result.ambiguous
        assert result.confident
