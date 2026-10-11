"""1:N face-embedding matching -- pure numpy, no camera/ML-framework dependency.

Two use cases, both built on the same cosine-similarity core:
- `find_duplicates`: registration-time 1:N duplicate-registration check.
- `identify`: voting-time 1:N voter identification. Never guesses -- if the
  top-2 matches are too close together, the result is `ambiguous` rather than
  silently picking the higher one (see docs/ARCHITECTURE.md).
"""

from __future__ import annotations

import numpy as np

from biometric.thresholds import (
    DUPLICATE_REJECT_COSINE_THRESHOLD,
    VOTING_IDENTIFY_COSINE_THRESHOLD,
    VOTING_MATCH_MARGIN,
)
from contracts.schemas import DuplicateCheckResult, MatchResult

Gallery = list[tuple[str, np.ndarray]]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _best_two(
    query: np.ndarray, gallery: Gallery
) -> tuple[tuple[str, float] | None, tuple[str, float] | None]:
    """(best, runner_up) as (voter_id, similarity) pairs, sorted descending."""
    if not gallery:
        return None, None
    scored = sorted(
        ((voter_id, cosine_similarity(query, emb)) for voter_id, emb in gallery),
        key=lambda pair: pair[1],
        reverse=True,
    )
    best = scored[0]
    runner_up = scored[1] if len(scored) > 1 else None
    return best, runner_up


def find_duplicates(
    query: np.ndarray,
    gallery: Gallery,
    threshold: float = DUPLICATE_REJECT_COSINE_THRESHOLD,
) -> DuplicateCheckResult:
    """Is `query` a close match to anyone already in `gallery`?"""
    best, _ = _best_two(query, gallery)
    if best is None or best[1] < threshold:
        return DuplicateCheckResult(
            is_duplicate=False, matched_voter_id=None, similarity=best[1] if best else 0.0
        )
    return DuplicateCheckResult(is_duplicate=True, matched_voter_id=best[0], similarity=best[1])


def identify(
    query: np.ndarray,
    gallery: Gallery,
    threshold: float = VOTING_IDENTIFY_COSINE_THRESHOLD,
    margin: float = VOTING_MATCH_MARGIN,
) -> MatchResult:
    """1:N identification with a mandatory confidence margin over the runner-up."""
    best, runner_up = _best_two(query, gallery)
    if best is None or best[1] < threshold:
        return MatchResult(
            matched=False,
            voter_id=None,
            similarity=best[1] if best else 0.0,
            margin=0.0,
            ambiguous=False,
        )

    runner_up_similarity = runner_up[1] if runner_up is not None else -1.0
    gap = best[1] - runner_up_similarity
    ambiguous = runner_up is not None and gap < margin
    return MatchResult(
        matched=True,
        voter_id=best[0],
        similarity=best[1],
        margin=gap,
        ambiguous=ambiguous,
    )
