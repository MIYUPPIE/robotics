"""Shared data shapes: voter/vote/office/party/language domain types.

Every concern imports these rather than inventing a locally-shaped equivalent --
this is the "typed Python module API" boundary described in docs/ARCHITECTURE.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

import numpy as np

EMBEDDING_DIM = 512


class Language(StrEnum):
    ENGLISH = "en"
    YORUBA = "yo"
    IGBO = "ig"
    HAUSA = "ha"


class Office(StrEnum):
    PRESIDENT = "PRESIDENT"
    GOVERNOR = "GOVERNOR"
    SENATE = "SENATE"
    HOR = "HOR"
    SHA = "SHA"


# Ballot presentation order -- must match data/offices.json "order" field.
OFFICE_ORDER: tuple[Office, ...] = (
    Office.PRESIDENT,
    Office.GOVERNOR,
    Office.SENATE,
    Office.HOR,
    Office.SHA,
)


class Gender(StrEnum):
    MALE = "M"
    FEMALE = "F"
    OTHER = "OTHER"


@dataclass(frozen=True)
class Party:
    acronym: str
    name: str
    color: str  # hex UI color; used for the ballot card border/accent, and as the
    # full card fill when `logo_path` is None (see hand_ui/renderer.py::draw_region)
    logo_path: str | None = None  # relative to data/, e.g. "party_logos/APC.png" --
    # None for parties with no verified logo asset, see data/party_logos/ATTRIBUTION.md
    # and docs/DISCLAIMER.md#party-logos for sourcing/licensing


@dataclass(frozen=True)
class OfficeInfo:
    code: Office
    name_en: str
    level: str  # "federal" | "state"
    order: int


@dataclass(frozen=True)
class StateInfo:
    code: str
    name: str
    zone: str


@dataclass
class VoterRecord:
    voter_id: str
    surname: str
    first_name: str
    middle_name: str | None
    dob: date
    gender: Gender
    state_code: str
    lga: str
    phone: str | None
    language_pref: Language
    face_embedding: np.ndarray  # shape (EMBEDDING_DIM,), float32, L2-normalized
    embedding_model_version: str
    registered_at: datetime
    has_voted: bool = False

    def __post_init__(self) -> None:
        if self.face_embedding.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"face_embedding must have shape ({EMBEDDING_DIM},), "
                f"got {self.face_embedding.shape}"
            )

    @property
    def full_name(self) -> str:
        parts = [self.surname, self.first_name, self.middle_name]
        return " ".join(p for p in parts if p)


@dataclass(frozen=True)
class MatchResult:
    """Outcome of a 1:N face identification attempt at voting time."""

    matched: bool
    voter_id: str | None
    similarity: float
    margin: float
    ambiguous: bool

    @property
    def confident(self) -> bool:
        """True only when there is exactly one clear match -- never guesses."""
        return self.matched and not self.ambiguous


@dataclass(frozen=True)
class DuplicateCheckResult:
    """Outcome of a 1:N duplicate check at registration time."""

    is_duplicate: bool
    matched_voter_id: str | None
    similarity: float


@dataclass(frozen=True)
class LivenessResult:
    is_live: bool
    blink_detected: bool
    frames_observed: int
    reason: str = ""


@dataclass(frozen=True)
class VoteSelection:
    office: Office
    party_code: str


def is_age_eligible(dob: date, as_of: date, min_age: int = 18) -> bool:
    """Pure age-eligibility check (Nigeria's voting age is 18)."""
    age = as_of.year - dob.year - ((as_of.month, as_of.day) < (dob.month, dob.day))
    return age >= min_age
