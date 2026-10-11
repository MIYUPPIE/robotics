# voice

YarnGPT2-backed Nigerian-language voice prompts (English/Yoruba/Igbo/Hausa) --
**pre-rendered only**. See [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) (ADR-3).

- `phrase_registry.py` -- loads `data/phrases/{lang}.json`; fragment vs static phrases
  distinguished by naming convention (`PARTY_NAME_*`, `OFFICE_NAME_*` are fragments).
- `client.py` -- `YarnGptClient`, the hosted-API wrapper. **Only** imported by
  `generate_prompts.py` -- never by any runtime-facing module.
- `cache.py` -- local audio-file lookup (`get`, `cache_path`) + completeness check
  (`verify_complete`). This is what `kiosk_app`/`hand_ui`/`registration_app`/
  `admin_app` are allowed to touch.
- `playback.py` -- non-blocking playback (`play_phrase`, `play_sequence`) via
  `soundfile`+`sounddevice`. `play_sequence` concatenates fragment clips locally
  (e.g. "You selected" + "APC" + "for" + "President") -- no network call.
- `generate_prompts.py` -- the one-off pre-render script:
  `python -m voice.generate_prompts` (needs `YARNGPT_API_KEY`, see `.env.example`).

## The runtime rule

The kiosk **never** calls YarnGPT live. `voice/tests/test_no_live_tts_in_runtime.py`
AST-scans `kiosk_app/`, `hand_ui/`, `registration_app/`, and `admin_app/` and fails
the build if any of them import `voice.client` or `requests` -- this is what makes
"no network call during a voting session" an enforced invariant, not just a
convention.

## Tests vs. evals

Gate tests (`tests/`, <2s) cover the phrase-completeness invariant, cache
lookup/miss/manifest logic, sequence-audio concatenation (against fixture WAVs),
the YarnGPT client's request-building (mocked HTTP), and the pre-render script's
skip/force/validation logic (fake client, no network).

`evals/audio_quality_eval.py` checks the *real* generated cache: non-silence /
duration / sample-rate on every file, plus an English-only Whisper
ASR-transcript-proxy check (Yoruba/Igbo/Hausa aren't reliably ASR-checkable -- see
[../docs/TRANSLATION_REVIEW_CHECKLIST.md](../docs/TRANSLATION_REVIEW_CHECKLIST.md)).
Run after `make prerender-audio`.
