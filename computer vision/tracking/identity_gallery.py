"""Persistent re-identification on top of a short-horizon tracker.

StrongSort (and every DeepSORT-style tracker) only keeps a track's appearance
history alive for `max_age` frames after its last update, then deletes the
track outright. A person who walks out of frame and back in gets a brand new
track ID with no memory of who they were -- the tracker's own ReID embeddings
only assist matching *within* a track's lifetime, not across a deletion.

IdentityGallery sits on top of that: it remembers the appearance embedding of
every tracker ID that has gone quiet, and re-links a freshly minted tracker ID
back to its original persistent identity if the appearance is a close enough
match.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= eps:
        return vector
    return vector / norm


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_normalize(a), _normalize(b)))


class IdentityGallery:
    """Maps volatile tracker IDs onto stable persistent identity IDs."""

    def __init__(
        self,
        similarity_threshold: float = 0.6,
        ema_alpha: float = 0.9,
        max_retired_frames: int = 900,
    ):
        self.similarity_threshold = similarity_threshold
        self.ema_alpha = ema_alpha
        self.max_retired_frames = max_retired_frames

        self._active: dict[int, int] = {}  # tracker_id -> persistent_id
        self._embeddings: dict[int, np.ndarray] = {}  # persistent_id -> embedding
        self._retired: set[int] = set()  # persistent_ids eligible for re-match
        self._last_seen_frame: dict[int, int] = {}  # persistent_id -> frame index
        self._next_persistent_id = 0
        self._frame = 0

    def update(self, tracker_ids: Sequence[int], embeddings: Sequence[np.ndarray]) -> list[int]:
        """Advance one frame and return the persistent ID for each tracker ID.

        Args:
            tracker_ids: raw tracker IDs present in this frame's output.
            embeddings: appearance embedding for each tracker ID, same order.

        Returns:
            Persistent identity IDs, one per input tracker ID, same order.
        """
        self._frame += 1
        self._expire_stale()

        active_this_frame = {int(tid) for tid in tracker_ids}
        self._retire_missing(active_this_frame)

        return [
            self._resolve(int(tid), emb)
            for tid, emb in zip(tracker_ids, embeddings, strict=True)
        ]

    def _resolve(self, tracker_id: int, embedding: np.ndarray) -> int:
        embedding = _normalize(embedding)

        if tracker_id in self._active:
            persistent_id = self._active[tracker_id]
            previous = self._embeddings[persistent_id]
            self._embeddings[persistent_id] = _normalize(
                self.ema_alpha * previous + (1.0 - self.ema_alpha) * embedding
            )
            self._last_seen_frame[persistent_id] = self._frame
            return persistent_id

        matched_id = self._match_retired(embedding)
        if matched_id is None:
            persistent_id = self._next_persistent_id
            self._next_persistent_id += 1
            self._embeddings[persistent_id] = embedding
        else:
            persistent_id = matched_id
            self._retired.discard(persistent_id)

        self._active[tracker_id] = persistent_id
        self._last_seen_frame[persistent_id] = self._frame
        return persistent_id

    def _match_retired(self, embedding: np.ndarray) -> int | None:
        best_id, best_similarity = None, self.similarity_threshold
        for persistent_id in self._retired:
            similarity = _cosine_similarity(embedding, self._embeddings[persistent_id])
            if similarity >= best_similarity:
                best_similarity, best_id = similarity, persistent_id
        return best_id

    def _retire_missing(self, active_this_frame: set[int]) -> None:
        gone = [tid for tid in self._active if tid not in active_this_frame]
        for tracker_id in gone:
            persistent_id = self._active.pop(tracker_id)
            self._retired.add(persistent_id)

    def _expire_stale(self) -> None:
        stale = [
            persistent_id
            for persistent_id in self._retired
            if self._frame - self._last_seen_frame[persistent_id] > self.max_retired_frames
        ]
        for persistent_id in stale:
            self._retired.discard(persistent_id)
            self._embeddings.pop(persistent_id, None)
            self._last_seen_frame.pop(persistent_id, None)
