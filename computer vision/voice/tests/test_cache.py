import pytest

from contracts.schemas import Language
from voice import cache


@pytest.fixture
def fake_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "AUDIO_CACHE_DIR", tmp_path)
    return tmp_path


class TestCachePath:
    def test_layout_is_language_then_phrase_id(self, fake_cache_dir):
        path = cache.cache_path("WELCOME_ATTRACT", Language.YORUBA)
        assert path == fake_cache_dir / "yo" / "WELCOME_ATTRACT.wav"


class TestGet:
    def test_raises_when_missing(self, fake_cache_dir):
        with pytest.raises(cache.MissingAudioError):
            cache.get("WELCOME_ATTRACT", Language.ENGLISH)

    def test_returns_path_when_present(self, fake_cache_dir):
        target = cache.cache_path("WELCOME_ATTRACT", Language.ENGLISH)
        target.parent.mkdir(parents=True)
        target.write_bytes(b"fake wav bytes")

        assert cache.get("WELCOME_ATTRACT", Language.ENGLISH) == target


class TestVerifyComplete:
    def test_reports_every_missing_combination(self, fake_cache_dir):
        missing = cache.verify_complete(
            languages=[Language.ENGLISH, Language.YORUBA], phrase_ids={"A", "B"}
        )
        assert set(missing) == {"A/en", "B/en", "A/yo", "B/yo"}

    def test_empty_when_all_present(self, fake_cache_dir):
        for lang in (Language.ENGLISH, Language.YORUBA):
            for phrase_id in ("A", "B"):
                path = cache.cache_path(phrase_id, lang)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")

        missing = cache.verify_complete(
            languages=[Language.ENGLISH, Language.YORUBA], phrase_ids={"A", "B"}
        )
        assert missing == []

    def test_partial_coverage_reports_only_the_gap(self, fake_cache_dir):
        present = cache.cache_path("A", Language.ENGLISH)
        present.parent.mkdir(parents=True, exist_ok=True)
        present.write_bytes(b"x")

        missing = cache.verify_complete(languages=[Language.ENGLISH], phrase_ids={"A", "B"})
        assert missing == ["B/en"]


class TestManifest:
    def test_load_manifest_empty_when_no_file(self, fake_cache_dir):
        assert cache.load_manifest() == {}

    def test_load_manifest_parses_entries(self, fake_cache_dir):
        import json

        fake_cache_dir.mkdir(parents=True, exist_ok=True)
        (fake_cache_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "WELCOME_ATTRACT/en": {
                        "phrase_id": "WELCOME_ATTRACT",
                        "language": "en",
                        "speaker": "emma",
                        "generated_at": "2026-08-01T00:00:00+00:00",
                        "text_hash": "abc123",
                    }
                }
            )
        )
        manifest = cache.load_manifest()
        assert manifest["WELCOME_ATTRACT/en"].speaker == "emma"
