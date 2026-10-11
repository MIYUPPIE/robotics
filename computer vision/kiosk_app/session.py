"""In-memory voting session state.

Nothing here is ever persisted until
storage.repositories.vote_repository.submit_vote_session() commits it in one
atomic transaction (see docs/ARCHITECTURE.md#ballot-secrecy) -- a VotingSession
that's discarded (timeout, officer reset) simply vanishes, having written
nothing to the database.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from contracts.schemas import OFFICE_ORDER, Language, Office


@dataclass
class VotingSession:
    language: Language | None = None
    voter_id: str | None = None
    selections: dict[Office, str] = field(default_factory=dict)
    current_office_index: int = 0
    return_to_review: bool = False
    liveness_retry_count: int = 0

    def current_office(self) -> Office:
        return OFFICE_ORDER[self.current_office_index]

    def is_last_office(self) -> bool:
        return self.current_office_index == len(OFFICE_ORDER) - 1

    def all_offices_selected(self) -> bool:
        return all(office in self.selections for office in OFFICE_ORDER)

    def reset(self) -> None:
        self.language = None
        self.voter_id = None
        self.selections = {}
        self.current_office_index = 0
        self.return_to_review = False
        self.liveness_retry_count = 0
