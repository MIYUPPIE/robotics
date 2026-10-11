# Translation review checklist

Every phrase in `data/phrases/yo.json` (Yoruba), `data/phrases/ig.json` (Igbo), and
`data/phrases/ha.json` (Hausa) is machine-drafted and marked
`"review_status": "machine_draft_unreviewed"`. They are good enough to exercise the
full pipeline (rendering, caching, playback, dwell-linked cues) end to end, but they
have **not** been checked by a native/fluent speaker and must not be treated as
production-quality translations.

## Before any real deployment

For each phrase in each of the three files:

1. **Meaning check** -- does the translation say what `text_en` says, with no drift,
   omissions, or mistranslated idioms?
2. **Register check** -- is the tone appropriate for a public kiosk instruction (clear,
   respectful, plain) rather than overly formal/literary or overly casual?
3. **Regional-variant check** -- Yoruba, Igbo, and Hausa each have regional dialect
   variation; confirm the phrasing used is broadly understood, not a narrow dialect
   choice that would confuse speakers from other regions.
4. **TTS-fit check** -- after `voice/generate_prompts.py` renders the phrase, *listen*
   to the actual YarnGPT output (not just read the source text) -- TTS systems can
   mispronounce loanwords, misplace tone/stress, or mishandle diacritics. Flag any
   phrase where the rendered audio doesn't sound right even if the source text is
   correct.
5. **Fragment-concatenation check** -- for phrases composed at runtime from fragments
   (e.g. `SELECTION_MADE_PREFIX` + `PARTY_NAME_{acronym}` + `OFFICE_FOR_PREFIX` +
   `OFFICE_NAME_{code}` + `SELECTION_CONFIRM_PROMPT`, see `voice/phrase_registry.py`),
   listen to at least one full concatenated example per office and confirm the joins
   sound natural -- grammatical agreement across concatenated fragments in Yoruba/Igbo/
  Hausa is not guaranteed just because each fragment is individually correct.

## After review

- Update the reviewed phrase's `review_status` to `"native_reviewed"` and add a
  `reviewed_by` / `reviewed_at` field.
- Re-run `make prerender-audio` only for changed phrases (`generate_prompts.py`
  skips existing files unless `--force`, so this is cheap).
- Re-run `voice/evals/audio_quality_eval.py` to confirm the re-rendered audio still
  passes the automated non-silence/duration checks.
- `voice/tests/test_phrase_registry.py` includes a completeness check that every
  `phrase_id` in the registry exists in all 4 language files -- a review pass should
  never leave a language file with fewer phrases than English.

## Known automated-eval limitation

`voice/evals/audio_quality_eval.py` uses Whisper-based ASR as a rough correctness
proxy, but only for **English** -- Whisper's Yoruba/Igbo/Hausa transcription quality is
too weak to be a reliable automated signal. For those three languages, the eval only
checks non-silence/duration/sample-rate (that *something* was rendered), never
correctness. Human review per this checklist is the only real quality gate for
Yoruba/Igbo/Hausa content.
