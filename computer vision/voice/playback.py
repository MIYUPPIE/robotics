"""Audio playback -- reads only from the local cache (voice/cache.py), never
the network. Non-blocking by default so the kiosk's camera/state-machine
loop never stalls on speech.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

from contracts.schemas import Language
from voice.cache import get as get_cached_path


def _load(phrase_id: str, language: Language) -> tuple[np.ndarray, int]:
    path = get_cached_path(phrase_id, language)
    data, sample_rate = sf.read(str(path), dtype="float32")
    return data, sample_rate


def build_sequence_audio(
    phrase_ids: list[str], language: Language, gap_ms: float = 150.0
) -> tuple[np.ndarray, int]:
    """Concatenate cached phrase clips (with a short silence gap) into one buffer.

    Pure file-I/O, no audio device involved -- gate-tested against fixture
    WAV files (voice/tests/test_playback.py).
    """
    if not phrase_ids:
        raise ValueError("phrase_ids must not be empty")

    clips: list[np.ndarray] = []
    sample_rate: int | None = None
    for phrase_id in phrase_ids:
        data, sr = _load(phrase_id, language)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(
                f"sample rate mismatch: {phrase_id} is {sr}Hz, expected {sample_rate}Hz"
            )
        clips.append(data)

    assert sample_rate is not None
    gap = np.zeros(int(sample_rate * gap_ms / 1000.0), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, clip in enumerate(clips):
        parts.append(clip)
        if i != len(clips) - 1:
            parts.append(gap)
    return np.concatenate(parts), sample_rate


def _play(data: np.ndarray, sample_rate: int, blocking: bool) -> None:
    import sounddevice as sd

    sd.play(data, sample_rate)
    if blocking:
        sd.wait()


def play_phrase(phrase_id: str, language: Language, blocking: bool = False) -> None:
    data, sample_rate = _load(phrase_id, language)
    _play(data, sample_rate, blocking)


def play_sequence(phrase_ids: list[str], language: Language, blocking: bool = False) -> None:
    data, sample_rate = build_sequence_audio(phrase_ids, language)
    _play(data, sample_rate, blocking)
