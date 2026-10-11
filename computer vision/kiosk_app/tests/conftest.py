from datetime import datetime, timedelta

import pytest

from storage.repositories.vote_repository import VoteSubmissionError


class FakeClock:
    """A controllable clock: `.advance(seconds)` moves it forward; calling it
    returns the current `datetime`."""

    def __init__(self, start: datetime | None = None):
        self.now = start or datetime(2026, 8, 1, 9, 0, 0)

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)

    def __call__(self) -> datetime:
        return self.now


class SpySubmit:
    """Records every call; raises VoteSubmissionError while `.should_fail` is True."""

    def __init__(self):
        self.calls: list[tuple[str, dict, datetime]] = []
        self.should_fail = False

    def __call__(self, voter_id, selections, now) -> None:
        self.calls.append((voter_id, dict(selections), now))
        if self.should_fail:
            raise VoteSubmissionError("simulated failure")


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def spy_submit():
    return SpySubmit()
