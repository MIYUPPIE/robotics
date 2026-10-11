"""Voter ID generation.

Deliberately NOT shaped like a real 19-digit INEC VIN -- see
docs/NIGERIA_ELECTION_MODEL.md -- so this prototype's ID can never be mistaken
for, or misused as, a real voter document.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from datetime import date

# Excludes 0/O/1/I/L to avoid visual ambiguity on a printed slip.
_ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"
_SUFFIX_LEN = 6
_MAX_ATTEMPTS = 20


def _random_suffix() -> str:
    return "".join(secrets.choice(_ALPHABET) for _ in range(_SUFFIX_LEN))


def generate_voter_id(state_code: str, on: date, exists: Callable[[str], bool]) -> str:
    """`PVR-{STATE}-{YYYYMMDD}-{6-char suffix}`, retried on collision.

    `exists` should check the candidate ID against storage (e.g.
    storage.repositories.voter_repository.exists, bound to a live connection).
    Collision odds are astronomically low (32**6 ~= 1.07B combinations per
    state per day) but this is checked for correctness, never assumed.
    """
    date_part = on.strftime("%Y%m%d")
    for _ in range(_MAX_ATTEMPTS):
        candidate = f"PVR-{state_code}-{date_part}-{_random_suffix()}"
        if not exists(candidate):
            return candidate
    raise RuntimeError(f"could not generate a unique voter ID after {_MAX_ATTEMPTS} attempts")
