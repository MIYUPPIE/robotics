# CLAUDE.md -- Voter Kiosk CV

Project-specific notes on top of the global CLAUDE.md rules. Read
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before making structural changes -- it
documents the four deliberate deviations from the default services-first/network-
contract pattern (in-process module contracts, one shared conda env, YarnGPT-is-
not-an-LLM-call, flat top-level concern directories) and why each one is correct
for this specific single-process, single-machine kiosk app.

## Non-negotiables specific to this repo

- **Never call `voice.client` (the YarnGPT hosted API) from `kiosk_app/`,
  `hand_ui/`, `registration_app/`, or `admin_app/`.** Only `voice/generate_prompts.py`
  may. This is enforced by `voice/tests/test_no_live_tts_in_runtime.py` -- if you add
  a new runtime-facing module that needs a spoken phrase, use `voice.cache`/
  `voice.playback`, never `voice.client`.
- **Never add a voter reference to the `votes` table.** Ballot secrecy depends on
  `votes` never being joinable back to `voters` -- see
  [docs/ARCHITECTURE.md#ballot-secrecy](docs/ARCHITECTURE.md#ballot-secrecy). All
  writes to `votes`/`voted_log` must go through
  `storage.repositories.vote_repository.submit_vote_session`, never ad hoc INSERTs.
- **Keep the pure/thin-shell split.** `hand_ui/dwell_fsm.py` and
  `kiosk_app/state_machine.py` must stay free of camera/MediaPipe/database imports.
  If a change to selection or ballot logic seems to require touching a camera API,
  the change belongs in the thin shell (`hand_ui/landmarker.py`, `kiosk_app/main.py`)
  calling into the pure core, not the other way around.
- **Every new biometric/gesture/audio threshold goes in
  `biometric/thresholds.py`** (or `hand_ui/dwell_fsm.py`'s `DEFAULT_DWELL_SECONDS` /
  `kiosk_app/config.py` for UX timings) -- never hardcode a magic number inline.
- **This is a prototype, not real election infrastructure.** Don't let a task
  description imply otherwise -- see [docs/DISCLAIMER.md](docs/DISCLAIMER.md). Any
  README/UI copy should stay honest about that.

## Running things

See the root [README.md](README.md) Quickstart. Short version: `make setup` (conda
env `voter-kiosk-cv`), `make test` (gate tests, <2s, no camera/network/model), `make
eval` (real-model quality evals), `make prerender-audio` (needs `YARNGPT_API_KEY`)
before `make run-kiosk` will produce sound.

## When adding a new concern directory

Follow the existing pattern: `<name>/`, `<name>/tests/`, `<name>/README.md`, and
(if it has real-model/real-camera quality characteristics worth measuring)
`<name>/evals/`. Import only from `contracts/`, `data/`, `storage/`, and other
concerns' public modules -- never reach into another concern's private
(underscore-prefixed) names.
