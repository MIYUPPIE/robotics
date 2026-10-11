from datetime import date, datetime

import numpy as np
import pytest

from contracts.schemas import (
    EMBEDDING_DIM,
    Gender,
    Language,
    MatchResult,
    Office,
    VoterRecord,
    is_age_eligible,
)


def _embedding() -> np.ndarray:
    v = np.ones(EMBEDDING_DIM, dtype=np.float32)
    return v / np.linalg.norm(v)


def _voter(**overrides) -> VoterRecord:
    defaults = dict(
        voter_id="PVR-LAG-20260101-ABCDEF",
        surname="Okafor",
        first_name="Ada",
        middle_name=None,
        dob=date(2000, 1, 1),
        gender=Gender.FEMALE,
        state_code="LAG",
        lga="Ikeja",
        phone=None,
        language_pref=Language.ENGLISH,
        face_embedding=_embedding(),
        embedding_model_version="insightface-buffalo_l",
        registered_at=datetime(2026, 1, 1, 12, 0, 0),
    )
    defaults.update(overrides)
    # mypy can't verify per-field types through a dict-unpack factory pattern.
    return VoterRecord(**defaults)  # type: ignore[arg-type]


class TestVoterRecord:
    def test_full_name_joins_present_parts(self):
        voter = _voter(middle_name="Chioma")
        assert voter.full_name == "Okafor Ada Chioma"

    def test_full_name_skips_missing_middle_name(self):
        voter = _voter(middle_name=None)
        assert voter.full_name == "Okafor Ada"

    def test_rejects_wrong_embedding_shape(self):
        with pytest.raises(ValueError):
            _voter(face_embedding=np.ones(10, dtype=np.float32))


class TestMatchResult:
    def test_confident_requires_matched_and_not_ambiguous(self):
        assert MatchResult(
            matched=True, voter_id="v1", similarity=0.9, margin=0.2, ambiguous=False
        ).confident

    def test_not_confident_when_ambiguous(self):
        assert not MatchResult(
            matched=True, voter_id="v1", similarity=0.9, margin=0.01, ambiguous=True
        ).confident

    def test_not_confident_when_unmatched(self):
        assert not MatchResult(
            matched=False, voter_id=None, similarity=0.1, margin=0.0, ambiguous=False
        ).confident


class TestAgeEligibility:
    @pytest.mark.parametrize(
        "dob,as_of,expected",
        [
            (date(2008, 8, 1), date(2026, 8, 1), True),  # exactly 18 today
            (date(2008, 8, 2), date(2026, 8, 1), False),  # turns 18 tomorrow
            (date(2008, 7, 31), date(2026, 8, 1), True),  # turned 18 yesterday
            (date(2000, 1, 1), date(2026, 8, 1), True),
            (date(2020, 1, 1), date(2026, 8, 1), False),
        ],
    )
    def test_boundary_cases(self, dob, as_of, expected):
        assert is_age_eligible(dob, as_of) is expected


def test_office_order_matches_all_offices():
    from contracts.schemas import OFFICE_ORDER

    assert set(OFFICE_ORDER) == set(Office)
    assert len(OFFICE_ORDER) == len(set(OFFICE_ORDER))
