"""Audio-cache quality eval.

Structural checks (non-silence/duration/sample-rate) run on every cached WAV
across all 4 languages. An English-only Whisper ASR-transcript-proxy check
adds a rough correctness signal -- Yoruba/Igbo/Hausa are excluded from that
check because Whisper's coverage there is too weak to be a reliable
automated signal (see docs/TRANSLATION_REVIEW_CHECKLIST.md for the human
review process those three languages actually need).

Usage: python -m voice.evals.audio_quality_eval
Requires the audio cache to already be populated: `make prerender-audio`.
"""

from __future__ import annotations

import re
import sys

import numpy as np
import soundfile as sf

from contracts.schemas import Language
from voice.cache import AUDIO_CACHE_DIR, cache_path
from voice.phrase_registry import canonical_phrase_ids, load_phrases

MIN_DURATION_S = 0.05
MAX_DURATION_S = 30.0
SILENCE_RMS_THRESHOLD = 1e-4
EXPECTED_SAMPLE_RATE_MIN = 8000
ASR_WORD_OVERLAP_PASS_BAR = 0.70


def _structural_checks() -> tuple[int, list[str]]:
    checked = 0
    failures: list[str] = []
    for language in Language:
        for phrase_id in sorted(canonical_phrase_ids()):
            path = cache_path(phrase_id, language)
            if not path.exists():
                failures.append(
                    f"{phrase_id}/{language.value}: MISSING (run `make prerender-audio`)"
                )
                continue
            checked += 1
            data, sample_rate = sf.read(str(path), dtype="float32")
            duration = len(data) / sample_rate if sample_rate else 0.0
            if duration < MIN_DURATION_S or duration > MAX_DURATION_S:
                failures.append(
                    f"{phrase_id}/{language.value}: suspicious duration {duration:.2f}s"
                )
            if sample_rate < EXPECTED_SAMPLE_RATE_MIN:
                failures.append(f"{phrase_id}/{language.value}: low sample rate {sample_rate}Hz")
            rms = float(np.sqrt(np.mean(np.square(data)))) if len(data) else 0.0
            if rms < SILENCE_RMS_THRESHOLD:
                failures.append(f"{phrase_id}/{language.value}: appears silent (rms={rms:.6f})")
    return checked, failures


def _normalize_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _asr_proxy_check_english() -> list[str]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print(
            "faster-whisper not installed (eval-only extra: `pip install '.[eval]'`) "
            "-- skipping the English ASR-transcript-proxy check.",
            file=sys.stderr,
        )
        return []

    model = WhisperModel("base", device="cpu", compute_type="int8")
    phrases = load_phrases(Language.ENGLISH)
    failures = []
    for phrase_id, phrase in phrases.items():
        path = cache_path(phrase_id, Language.ENGLISH)
        if not path.exists():
            continue
        segments, _ = model.transcribe(str(path), language="en")
        transcript = " ".join(seg.text for seg in segments)
        expected_words = _normalize_words(phrase.text)
        if not expected_words:
            continue
        got_words = _normalize_words(transcript)
        overlap = len(expected_words & got_words) / len(expected_words)
        if overlap < ASR_WORD_OVERLAP_PASS_BAR:
            failures.append(
                f"{phrase_id}/en: ASR overlap {overlap:.0%} < {ASR_WORD_OVERLAP_PASS_BAR:.0%} "
                f"(expected {phrase.text!r}, got {transcript!r})"
            )
    return failures


def main() -> int:
    if not AUDIO_CACHE_DIR.exists():
        print(
            f"No audio cache at {AUDIO_CACHE_DIR} -- run `make prerender-audio` first.",
            file=sys.stderr,
        )
        return 1

    checked, structural_failures = _structural_checks()
    print(f"Structural checks: {checked} files checked, {len(structural_failures)} failure(s).")
    for f in structural_failures:
        print(f"  FAIL: {f}")

    asr_failures = _asr_proxy_check_english()
    print(f"\nEnglish ASR-transcript-proxy: {len(asr_failures)} failure(s).")
    for f in asr_failures:
        print(f"  FAIL: {f}")
    print(
        "\nYoruba/Igbo/Hausa are NOT ASR-checked (Whisper coverage too weak to be a "
        "reliable signal) -- see docs/TRANSLATION_REVIEW_CHECKLIST.md for the human "
        "review process those languages actually need."
    )

    passed = not structural_failures and not asr_failures
    print(f"\nOverall: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
