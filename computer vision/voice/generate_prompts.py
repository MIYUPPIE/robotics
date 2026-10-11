"""Pre-render every phrase in the registry to a local cache, for every
supported language. A one-off, operator-run script -- never imported by
kiosk_app/hand_ui/registration_app/admin_app (enforced by
voice/tests/test_no_live_tts_in_runtime.py).

Usage:
    python -m voice.generate_prompts [--force] [--language en] [--phrase-id WELCOME_ATTRACT]

Requires YARNGPT_API_KEY (see .env.example).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import requests

from contracts.schemas import Language
from voice import cache
from voice.client import YarnGptClient, YarnGptError
from voice.phrase_registry import DEFAULT_SPEAKERS, load_phrases

MIN_DURATION_S = 0.05
SILENCE_RMS_THRESHOLD = 1e-4

# Network hiccups and 429/5xx are worth retrying; other 4xx (bad key, bad
# request) will just fail identically again, so fail fast on those instead of
# burning through retries and rate-limit budget.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 3
RETRY_BACKOFF_BASE_S = 2.0


def _synthesize_with_retry(
    client: YarnGptClient,
    text: str,
    language: Language,
    speaker: str,
    manifest_key: str,
    max_attempts: int = MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    """Retries transient failures (network errors, 429/5xx) with exponential
    backoff; re-raises immediately on a non-retryable error (e.g. 401/400)."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return client.synthesize(text, language, speaker)
        except requests.exceptions.RequestException as exc:
            last_exc = exc
        except YarnGptError as exc:
            if exc.status_code is not None and exc.status_code not in RETRYABLE_STATUS_CODES:
                raise
            last_exc = exc

        if attempt < max_attempts:
            wait_s = RETRY_BACKOFF_BASE_S * (2 ** (attempt - 1))
            print(
                f"  retry {attempt}/{max_attempts} for {manifest_key} in {wait_s:.0f}s "
                f"({last_exc})",
                file=sys.stderr,
            )
            sleep(wait_s)

    assert last_exc is not None
    raise YarnGptError(f"{manifest_key}: failed after {max_attempts} attempts: {last_exc}")


def _manifest_path() -> Path:
    # Read dynamically off `cache.AUDIO_CACHE_DIR` (not a frozen module-level
    # constant) so tests can monkeypatch it and this still points at the
    # patched location.
    return cache.AUDIO_CACHE_DIR / "manifest.json"


def _validate_audio_bytes(raw: bytes, phrase_id: str) -> None:
    import numpy as np
    import soundfile as sf

    data, sample_rate = sf.read(io.BytesIO(raw), dtype="float32")
    duration = len(data) / sample_rate if sample_rate else 0.0
    if duration < MIN_DURATION_S:
        raise YarnGptError(f"{phrase_id}: rendered audio too short ({duration:.3f}s)")
    rms = float(np.sqrt(np.mean(np.square(data)))) if len(data) else 0.0
    if rms < SILENCE_RMS_THRESHOLD:
        raise YarnGptError(f"{phrase_id}: rendered audio appears silent (rms={rms:.6f})")


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def generate_all(
    client: YarnGptClient,
    languages: list[Language] | None = None,
    phrase_ids: list[str] | None = None,
    force: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Renders every requested phrase, in order, to the local cache.

    A phrase that fails (after retries) is logged and skipped -- it does NOT
    abort the rest of the batch. The manifest is written with everything that
    *did* succeed before this function raises, so a failed run is resumable:
    re-running the same command skips already-cached phrases and only
    retries what previously failed.
    """
    languages = languages or list(Language)
    manifest_path = _manifest_path()
    manifest: dict = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    failures: list[str] = []

    for language in languages:
        phrases = load_phrases(language)
        speaker = DEFAULT_SPEAKERS[language]
        ids_to_render = phrase_ids if phrase_ids is not None else list(phrases.keys())

        for phrase_id in ids_to_render:
            if phrase_id not in phrases:
                print(
                    f"WARNING: {phrase_id} missing from {language.value}.json, skipping",
                    file=sys.stderr,
                )
                continue
            phrase = phrases[phrase_id]
            out_path = cache.cache_path(phrase_id, language)
            manifest_key = f"{phrase_id}/{language.value}"

            if out_path.exists() and not force:
                print(f"skip (exists): {manifest_key}")
                continue

            print(f"rendering: {manifest_key} ...")
            try:
                audio_bytes = _synthesize_with_retry(
                    client, phrase.text, language, speaker, manifest_key, sleep=sleep
                )
                _validate_audio_bytes(audio_bytes, phrase_id)
            except (YarnGptError, requests.exceptions.RequestException) as exc:
                print(f"  FAILED: {manifest_key}: {exc}", file=sys.stderr)
                failures.append(manifest_key)
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(audio_bytes)

            manifest[manifest_key] = {
                "phrase_id": phrase_id,
                "language": language.value,
                "speaker": speaker,
                "generated_at": datetime.now(UTC).isoformat(),
                "text_hash": _text_hash(phrase.text),
            }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    if failures:
        raise YarnGptError(
            f"{len(failures)} phrase(s) failed after retries: {', '.join(failures)}"
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-render even if already cached")
    parser.add_argument(
        "--language", action="append", help="limit to one or more language codes (en/yo/ig/ha)"
    )
    parser.add_argument("--phrase-id", action="append", help="limit to one or more phrase_ids")
    args = parser.parse_args()

    languages = [Language(code) for code in args.language] if args.language else None

    try:
        client = YarnGptClient()
        generate_all(client, languages=languages, phrase_ids=args.phrase_id, force=args.force)
    except YarnGptError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
