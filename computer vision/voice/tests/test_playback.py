import numpy as np
import pytest
import soundfile as sf

from contracts.schemas import Language
from voice import cache, playback


@pytest.fixture
def fake_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "AUDIO_CACHE_DIR", tmp_path)
    return tmp_path


def _write_tone(path, seconds: float, sample_rate: int = 16000, freq: float = 440.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    tone = 0.1 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(str(path), tone, sample_rate)
    return tone, sample_rate


class TestBuildSequenceAudio:
    def test_concatenates_with_gap(self, fake_cache_dir):
        tone_a, sr = _write_tone(cache.cache_path("A", Language.ENGLISH), seconds=0.1)
        tone_b, _ = _write_tone(cache.cache_path("B", Language.ENGLISH), seconds=0.1)

        audio, sample_rate = playback.build_sequence_audio(["A", "B"], Language.ENGLISH, gap_ms=100)

        assert sample_rate == sr
        gap_samples = int(sr * 100 / 1000.0)
        expected_len = len(tone_a) + gap_samples + len(tone_b)
        assert len(audio) == expected_len

    def test_no_gap_after_last_clip(self, fake_cache_dir):
        _write_tone(cache.cache_path("A", Language.ENGLISH), seconds=0.05)
        audio, _ = playback.build_sequence_audio(["A"], Language.ENGLISH, gap_ms=500)
        # A single clip should be exactly its own length -- no trailing gap.
        _, sr = _write_tone(cache.cache_path("A", Language.ENGLISH), seconds=0.05)
        assert len(audio) == pytest.approx(int(sr * 0.05), abs=1)

    def test_empty_sequence_raises(self, fake_cache_dir):
        with pytest.raises(ValueError):
            playback.build_sequence_audio([], Language.ENGLISH)

    def test_missing_phrase_raises_missing_audio_error(self, fake_cache_dir):
        with pytest.raises(cache.MissingAudioError):
            playback.build_sequence_audio(["NOT_CACHED"], Language.ENGLISH)

    def test_sample_rate_mismatch_raises(self, fake_cache_dir):
        _write_tone(cache.cache_path("A", Language.ENGLISH), seconds=0.05, sample_rate=16000)
        _write_tone(cache.cache_path("B", Language.ENGLISH), seconds=0.05, sample_rate=22050)

        with pytest.raises(ValueError, match="sample rate mismatch"):
            playback.build_sequence_audio(["A", "B"], Language.ENGLISH)
