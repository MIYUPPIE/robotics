"""Local audio-file cache -- the only thing kiosk_app/hand_ui/registration_app/
admin_app touch for audio. Never calls the network (see voice/client.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from contracts.schemas import Language
from voice.phrase_registry import canonical_phrase_ids

AUDIO_CACHE_DIR = Path(__file__).parent / "audio_cache"


class MissingAudioError(RuntimeError):
    pass


def cache_path(phrase_id: str, language: Language) -> Path:
    return AUDIO_CACHE_DIR / language.value / f"{phrase_id}.wav"


def get(phrase_id: str, language: Language) -> Path:
    path = cache_path(phrase_id, language)
    if not path.exists():
        raise MissingAudioError(
            f"no cached audio for phrase_id={phrase_id!r} language={language.value!r} "
            f"at {path}. Run `make prerender-audio` first."
        )
    return path


def verify_complete(
    languages: list[Language] | None = None,
    phrase_ids: set[str] | None = None,
) -> list[str]:
    """Returns "phrase_id/language" for every missing cached file. Empty = complete."""
    languages = languages or list(Language)
    ids = phrase_ids if phrase_ids is not None else canonical_phrase_ids()
    missing = []
    for phrase_id in ids:
        for language in languages:
            if not cache_path(phrase_id, language).exists():
                missing.append(f"{phrase_id}/{language.value}")
    return missing


@dataclass(frozen=True)
class ManifestEntry:
    phrase_id: str
    language: str
    speaker: str
    generated_at: str
    text_hash: str


def load_manifest() -> dict[str, ManifestEntry]:
    manifest_path = AUDIO_CACHE_DIR / "manifest.json"
    if not manifest_path.exists():
        return {}
    raw = json.loads(manifest_path.read_text())
    return {key: ManifestEntry(**value) for key, value in raw.items()}


def _main() -> int:
    import sys

    if "--verify-complete" in sys.argv:
        missing = verify_complete()
        if missing:
            print(f"Missing {len(missing)} cached audio file(s):")
            for m in sorted(missing):
                print(f"  - {m}")
            return 1
        print("Audio cache is complete.")
        return 0
    print("Usage: python -m voice.cache --verify-complete")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
