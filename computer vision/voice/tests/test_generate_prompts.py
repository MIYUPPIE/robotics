import io
import json

import numpy as np
import pytest
import requests
import soundfile as sf

from contracts.schemas import Language
from voice import cache, generate_prompts
from voice.client import YarnGptError


def _tone_bytes(seconds: float = 0.3, sample_rate: int = 16000, amplitude: float = 0.2) -> bytes:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    tone = (amplitude * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, tone, sample_rate, format="WAV")
    return buf.getvalue()


def _silence_bytes(seconds: float = 0.3, sample_rate: int = 16000) -> bytes:
    silence = np.zeros(int(sample_rate * seconds), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, silence, sample_rate, format="WAV")
    return buf.getvalue()


class TestValidateAudioBytes:
    def test_accepts_normal_tone(self):
        generate_prompts._validate_audio_bytes(_tone_bytes(), "SOME_PHRASE")

    def test_rejects_too_short(self):
        with pytest.raises(YarnGptError, match="too short"):
            generate_prompts._validate_audio_bytes(_tone_bytes(seconds=0.01), "SOME_PHRASE")

    def test_rejects_silence(self):
        with pytest.raises(YarnGptError, match="silent"):
            generate_prompts._validate_audio_bytes(_silence_bytes(), "SOME_PHRASE")


class _FakeClient:
    def __init__(self):
        self.calls: list[tuple[str, Language, str]] = []

    def synthesize(self, text: str, language: Language, speaker: str) -> bytes:
        self.calls.append((text, language, speaker))
        return _tone_bytes()


@pytest.fixture
def fake_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "AUDIO_CACHE_DIR", tmp_path)
    return tmp_path


class TestGenerateAll:
    def test_renders_every_requested_phrase_and_writes_manifest(self, fake_cache_dir):
        client = _FakeClient()
        manifest = generate_prompts.generate_all(
            client, languages=[Language.ENGLISH], phrase_ids=["WELCOME_ATTRACT", "THANK_YOU"]
        )

        assert cache.cache_path("WELCOME_ATTRACT", Language.ENGLISH).exists()
        assert cache.cache_path("THANK_YOU", Language.ENGLISH).exists()
        assert "WELCOME_ATTRACT/en" in manifest
        assert "THANK_YOU/en" in manifest
        assert len(client.calls) == 2

    def test_skips_existing_files_by_default(self, fake_cache_dir):
        client = _FakeClient()
        generate_prompts.generate_all(
            client, languages=[Language.ENGLISH], phrase_ids=["WELCOME_ATTRACT"]
        )
        assert len(client.calls) == 1

        generate_prompts.generate_all(
            client, languages=[Language.ENGLISH], phrase_ids=["WELCOME_ATTRACT"]
        )
        assert len(client.calls) == 1  # still 1 -- second run was a no-op skip

    def test_force_reregenerates_existing_files(self, fake_cache_dir):
        client = _FakeClient()
        generate_prompts.generate_all(
            client, languages=[Language.ENGLISH], phrase_ids=["WELCOME_ATTRACT"]
        )
        generate_prompts.generate_all(
            client, languages=[Language.ENGLISH], phrase_ids=["WELCOME_ATTRACT"], force=True
        )
        assert len(client.calls) == 2

    def test_manifest_records_speaker_and_text_hash(self, fake_cache_dir):
        client = _FakeClient()
        manifest = generate_prompts.generate_all(
            client, languages=[Language.ENGLISH], phrase_ids=["WELCOME_ATTRACT"]
        )
        entry = manifest["WELCOME_ATTRACT/en"]
        assert entry["speaker"] == "emma"
        assert entry["text_hash"]
        assert entry["generated_at"]

    def test_unknown_phrase_id_is_skipped_with_warning(self, fake_cache_dir, capsys):
        client = _FakeClient()
        manifest = generate_prompts.generate_all(
            client, languages=[Language.ENGLISH], phrase_ids=["NOT_A_REAL_PHRASE"]
        )
        assert manifest == {}
        assert len(client.calls) == 0

    def test_manifest_persisted_to_disk(self, fake_cache_dir):
        client = _FakeClient()
        generate_prompts.generate_all(
            client, languages=[Language.ENGLISH], phrase_ids=["WELCOME_ATTRACT"]
        )
        manifest_on_disk = json.loads((fake_cache_dir / "manifest.json").read_text())
        assert "WELCOME_ATTRACT/en" in manifest_on_disk


class _FlakyClient:
    """Fails `fail_times` times with the given exception, then succeeds."""

    def __init__(self, fail_times: int, exc_factory):
        self.fail_times = fail_times
        self.exc_factory = exc_factory
        self.calls = 0

    def synthesize(self, text, language, speaker) -> bytes:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc_factory()
        return _tone_bytes()


class _AlwaysFailClient:
    def __init__(self, exc_factory):
        self.exc_factory = exc_factory
        self.calls = 0

    def synthesize(self, text, language, speaker) -> bytes:
        self.calls += 1
        raise self.exc_factory()


def _no_real_sleep():
    """Records sleep durations instead of actually sleeping, so retry tests stay fast."""
    recorded: list[float] = []

    def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)

    return fake_sleep, recorded


class TestSynthesizeWithRetry:
    def test_succeeds_immediately_without_retry(self):
        client = _FakeClient()
        sleep, recorded = _no_real_sleep()
        audio = generate_prompts._synthesize_with_retry(
            client, "hi", Language.ENGLISH, "emma", "X/en", sleep=sleep
        )
        assert audio == _tone_bytes()
        assert len(client.calls) == 1
        assert recorded == []

    def test_retries_network_error_then_succeeds(self):
        client = _FlakyClient(
            fail_times=2, exc_factory=lambda: requests.exceptions.ReadTimeout("timeout")
        )
        sleep, recorded = _no_real_sleep()
        audio = generate_prompts._synthesize_with_retry(
            client, "hi", Language.ENGLISH, "emma", "X/en", sleep=sleep
        )
        assert audio == _tone_bytes()
        assert client.calls == 3
        assert len(recorded) == 2  # slept between attempts 1->2 and 2->3

    def test_retries_retryable_status_code_then_succeeds(self):
        client = _FlakyClient(
            fail_times=1, exc_factory=lambda: YarnGptError("server busy", status_code=503)
        )
        sleep, recorded = _no_real_sleep()
        audio = generate_prompts._synthesize_with_retry(
            client, "hi", Language.ENGLISH, "emma", "X/en", sleep=sleep
        )
        assert audio == _tone_bytes()
        assert client.calls == 2
        assert len(recorded) == 1

    def test_does_not_retry_non_retryable_status_code(self):
        client = _AlwaysFailClient(exc_factory=lambda: YarnGptError("bad api key", status_code=401))
        sleep, recorded = _no_real_sleep()
        with pytest.raises(YarnGptError, match="bad api key"):
            generate_prompts._synthesize_with_retry(
                client, "hi", Language.ENGLISH, "emma", "X/en", sleep=sleep
            )
        assert client.calls == 1  # failed fast, no retry burned
        assert recorded == []

    def test_gives_up_after_max_attempts(self):
        client = _AlwaysFailClient(
            exc_factory=lambda: requests.exceptions.ConnectionError("no route")
        )
        sleep, recorded = _no_real_sleep()
        with pytest.raises(YarnGptError, match="failed after 3 attempts"):
            generate_prompts._synthesize_with_retry(
                client, "hi", Language.ENGLISH, "emma", "X/en", max_attempts=3, sleep=sleep
            )
        assert client.calls == 3
        assert len(recorded) == 2  # slept between 1->2 and 2->3, not after the last attempt

    def test_backoff_is_exponential(self):
        client = _AlwaysFailClient(exc_factory=lambda: requests.exceptions.Timeout("slow"))
        sleep, recorded = _no_real_sleep()
        with pytest.raises(YarnGptError):
            generate_prompts._synthesize_with_retry(
                client, "hi", Language.ENGLISH, "emma", "X/en", max_attempts=4, sleep=sleep
            )
        assert recorded == [2.0, 4.0, 8.0]


class TestGenerateAllPartialFailure:
    def test_one_failing_phrase_does_not_block_the_rest_of_the_batch(self, fake_cache_dir):
        good = _tone_bytes()

        thank_you_text = "Thank you for voting. Please step away for the next voter."

        class _MixedClient:
            def __init__(self):
                self.calls: list[str] = []

            def synthesize(self, text, language, speaker) -> bytes:
                self.calls.append(text)
                if text == thank_you_text:
                    raise YarnGptError("bad key", status_code=401)
                return good

        client = _MixedClient()
        sleep, _ = _no_real_sleep()

        with pytest.raises(YarnGptError, match="1 phrase\\(s\\) failed"):
            generate_prompts.generate_all(
                client,
                languages=[Language.ENGLISH],
                phrase_ids=["WELCOME_ATTRACT", "THANK_YOU"],
                sleep=sleep,
            )

        # The phrase that succeeded is still cached, despite the overall raise.
        assert cache.cache_path("WELCOME_ATTRACT", Language.ENGLISH).exists()
        assert not cache.cache_path("THANK_YOU", Language.ENGLISH).exists()

    def test_a_failed_run_is_resumable(self, fake_cache_dir):
        attempt_count = {"n": 0}

        class _FailOnceClient:
            def synthesize(self, text, language, speaker) -> bytes:
                attempt_count["n"] += 1
                if attempt_count["n"] == 1:
                    raise YarnGptError("bad key", status_code=401)
                return _tone_bytes()

        client = _FailOnceClient()
        sleep, _ = _no_real_sleep()

        with pytest.raises(YarnGptError):
            generate_prompts.generate_all(
                client, languages=[Language.ENGLISH], phrase_ids=["WELCOME_ATTRACT"], sleep=sleep
            )
        assert not cache.cache_path("WELCOME_ATTRACT", Language.ENGLISH).exists()

        # Re-running (as an operator would after fixing whatever broke)
        # retries the failed phrase and succeeds this time.
        manifest = generate_prompts.generate_all(
            client, languages=[Language.ENGLISH], phrase_ids=["WELCOME_ATTRACT"], sleep=sleep
        )
        assert "WELCOME_ATTRACT/en" in manifest
        assert cache.cache_path("WELCOME_ATTRACT", Language.ENGLISH).exists()
