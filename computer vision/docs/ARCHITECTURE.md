# Architecture

## System overview

One SQLite datastore (`storage/`), shared by three independent front-end processes:

```
registration_app (Tkinter)  ──┐
kiosk_app (fullscreen OpenCV) ─┼──► storage/ (SQLite: voters, votes, voted_log)
admin_app (Tkinter, PIN)      ──┘
```

Each front-end is its own OS process, started independently (`python -m
registration_app.app`, `python -m kiosk_app.main`, `python -m admin_app.app`). They
never talk to each other directly -- only through the shared SQLite file, via the
`storage/repositories/` layer. `contracts/` defines the typed data shapes (dataclasses)
that flow between every module, so nobody invents a slightly-different shape locally.

## ADR-1: In-process module contracts, not network services

CLAUDE.md's default architecture mandate is services-first with **network** contracts
between services (HTTP/gRPC), so that independent Claude sessions can work in parallel
without collision.

`kiosk_app`, however, is a single real-time process: one camera feed, decoded and
processed at the display frame rate, driving one on-screen state machine. Introducing a
network hop between, say, `hand_ui` (cursor tracking) and `kiosk_app` (the screen
reacting to that cursor) would add latency and a new failure mode (What if the hand
tracking "service" is down? What if the network call to identify a face times out
mid-frame?) for zero benefit -- there is no plausible world where `hand_ui` and
`kiosk_app` run on different machines for this project.

**Decision**: keep the *goal* of services-first (clean boundaries, one concern per
directory, each independently testable, no directory reaching into another directory's
internals, each with its own `tests/`/`evals/`/`README.md`) but implement the boundary
as a **typed Python module API** -- `contracts/schemas.py` dataclasses and `Protocol`
classes, imported by value -- rather than a network call. `registration_app`,
`kiosk_app`, and `admin_app` remain separate **processes** (so they genuinely are
independently deployable/runnable, and two people could work on two of them in
parallel), but the CV/ML/biometric/voice concerns they all depend on
(`biometric/`, `voice/`, `hand_ui/`, `storage/`) are libraries, not services.

## ADR-2: One shared Python environment

Every concern shares heavyweight, hard-to-install binary dependencies -- OpenCV,
MediaPipe, onnxruntime -- and (for `kiosk_app`) they all run inside the same OS process
and the same real-time loop. Splitting into one virtualenv per concern would mean N
redundant multi-gigabyte installs (model weights, CUDA libs) with no actual isolation
benefit, since `kiosk_app` imports `biometric`, `voice`, and `hand_ui` directly in the
same interpreter.

**Decision**: one dependency-managed project (`pyproject.toml`), one dedicated conda
environment, `voter-kiosk-cv` (Python 3.11) -- explicitly **not** the sibling `tf_gpu`
env used by unrelated projects on this machine, because this project's dependency set
(exact `mediapipe`/`onnxruntime`/`insightface` versions) is its own artifact and
shouldn't drift with an unrelated project's upgrades. Logical isolation between
concerns is enforced by **import discipline**, checked by a gate test
(`voice/tests/test_no_live_tts_in_runtime.py`) that AST-scans `kiosk_app/`, `hand_ui/`,
`registration_app/`, and `admin_app/` for imports of `voice.client` or `requests`, and
fails the build if it finds one.

## ADR-3: YarnGPT is a media call, not an "LLM access" call

CLAUDE.md requires routing "LLM access" through local Claude Code rather than a hosted
LLM API. That rule targets *reasoning* calls (text generation, classification,
agentic tool use). YarnGPT is a text-to-speech **audio synthesis** API -- there is no
reasoning happening, just a Nigerian-accented voice rendering fixed text -- so that rule
doesn't apply here.

More importantly: the kiosk **never calls YarnGPT live**, at all, during a voting or
registration session. See "Voice pre-rendering pipeline" below -- every phrase is
rendered once, ahead of time, to a local file, by a script that is never imported by
any runtime-facing module. This is deliberate for reliability (a kiosk shouldn't go
mute because a network call timed out mid-instruction) and for architectural hygiene
(exactly one file, `voice/client.py`, knows the YarnGPT HTTP contract; everything else
only ever reads local files).

## ADR-4: Flat top-level concern directories

`biometric/`, `voice/`, `hand_ui/`, `kiosk_app/`, `registration_app/`, `admin_app/`,
`storage/`, `contracts/`, `data/` sit at repo root rather than nested under `services/`
-- CLAUDE.md explicitly allows "or equivalent top-level directory," and a `services/`
prefix would only add a redundant path segment given there's no other top-level
category (no `libs/`, no `apps/` split) this project needs.

## Ballot secrecy

The single hardest correctness property in this system: **nobody, including someone
with full read access to the database, should be able to determine how a specific
voter voted.**

Schema-level design (`storage/schema.sql`):

- `voters` -- identity + face embedding + a `has_voted` boolean. No vote content.
- `votes` -- `(vote_id, office, party_code, cast_at)`. **No voter reference of any
  kind** -- not a foreign key, not a session UUID, nothing that could be joined back to
  `voters`. `cast_at` is rounded to the nearest 15-minute bucket specifically so it
  can't be correlated against `voted_log.cast_at` (which needs to be exact for the
  double-vote audit trail) to infer "the person who checked in at 10:03:12 must be the
  vote that landed at 10:03:41."
- `voted_log` -- `(voter_id, cast_at)`. Exists only to block a second full voting
  session for the same voter. Contains no vote content, so it's safe for its timestamp
  to be exact.

Write-path invariant, enforced in `storage/repositories/vote_repository.py::
submit_vote_session`: the 5 `votes` rows (one per office), the 1 `voted_log` row, and
the `voters.has_voted = 1` update happen in a **single SQLite transaction**. If any
part fails, the whole transaction rolls back -- there is no code path that can produce
"the voter is marked as having voted but no votes were recorded" or "votes were
recorded but the voter can vote again." A gate test forces a mid-transaction failure
(monkeypatched cursor raising on the 3rd insert) and asserts every table is back to
its pre-call state.

Everything upstream of `submit_vote_session` (language selection, face auth, all 5
office ballots, the review screen) only ever mutates an in-memory `VotingSession`
(`kiosk_app/session.py`) -- nothing touches the database until the voter explicitly
confirms SUBMIT on the review screen. Abandoning the kiosk at any point before that
(walking away, a timeout) leaves the database completely untouched.

**Residual risk, stated plainly**: this is a software-engineering control (no join
path exists), not a cryptographic one. A sufficiently motivated attacker with
disk-level access and, say, security camera footage timestamped to the second could in
principle attempt statistical correlation using arrival order plus the (bucketed)
`votes.cast_at` -- the 15-minute bucketing raises the cost of that attack substantially
for a single-kiosk demo but does not make it information-theoretically impossible.
Real e-voting systems solve this with techniques (mixnets, batched/shuffled commits,
homomorphic tallying) that are out of scope here; this is called out explicitly in
`docs/NIGERIA_ELECTION_MODEL.md` rather than glossed over.

## Deterministic core vs. thin CV shell

The two pieces of interactive logic most prone to subtle bugs -- the hand-gesture
dwell-selection state machine and the kiosk screen state machine -- are written as
**pure functions/classes with zero camera or MediaPipe dependency**:

- `hand_ui/dwell_fsm.py::DwellSelector.update(CursorSample) -> list[DwellEvent]` -- pure
  function of `(cursor position, timestamp, region geometry)`. Fed real MediaPipe
  landmarks at runtime, but fed synthetic `CursorSample` sequences in tests
  (`hand_ui/tests/test_dwell_fsm.py`), so every enter/exit/complete/hand-lost/debounce
  case is covered in <2s with no camera in the loop.
- `kiosk_app/state_machine.py::KioskStateMachine` -- pure transition logic with an
  injectable clock and a mocked vote-repository in tests
  (`kiosk_app/tests/test_state_machine.py`), covering every state transition
  including every timeout path, without a camera, a database, or MediaPipe.

The "thin shell" (`hand_ui/landmarker.py`, `kiosk_app/main.py`) only ever does I/O:
read a camera frame, run MediaPipe/InsightFace inference, convert the result into the
plain-dataclass shape the pure core expects, and render the pure core's output back to
the screen. This split is what makes the highest-stakes logic in the whole system
(what counts as a valid vote, what counts as a valid selection) fully unit-testable.

## Model residency

`biometric` (InsightFace `buffalo_l` + MediaPipe FaceLandmarker) and `hand_ui`
(MediaPipe HandLandmarker) models are all loaded **once**, at `kiosk_app` startup --
not lazily per-state -- to avoid a multi-second model-load stall interrupting a live
voter interaction. `kiosk_app/main.py`'s frame loop decides, based on the current
`KioskStateMachine` state, which model(s) to actually *run inference with* on a given
frame (face models only during `FACE_AUTH`; the hand model only during
language-select/ballot/review states) -- a cheap `if` in the shell, not a reload.

## Camera mirroring is applied once, at capture, everywhere

`kiosk_app/main.py::run()` and `registration_app/webcam_widget.py` both call
`cv2.flip(frame, 1)` immediately after reading a camera frame, before any other
processing touches it. This has to be true in **both** places, consistently, for two
different reasons:

- **Hand tracking.** The on-screen cursor is meant to feel like a mirror (move your
  hand right, the dot moves right on screen, same convention as every selfie camera
  and video-call app). `hand_ui/landmarker.py::HandCursorTracker` does *not* invert
  coordinates itself -- it trusts the frame it's given is already mirrored. An earlier
  version mirrored the cursor math instead of the frame, which meant the cursor moved
  correctly relative to the voter's *intent* but visually pointed at the wrong side of
  whatever camera background was visible on screen -- a real bug found during
  real-hardware testing, not a hypothetical one.
- **Biometric consistency.** `kiosk_app`'s face-auth step and `registration_app`'s
  enrollment step both compute an InsightFace embedding from a live camera frame. If
  one mirrored and the other didn't, a genuine voter's stored (registration-time)
  embedding would be compared against a differently-oriented (voting-time) embedding --
  ArcFace embeddings are *not* guaranteed flip-invariant, and this is not a risk worth
  taking in a biometric identification path. Both capture points mirror identically so
  every embedding in the system is computed in the same orientation.

`biometric/face_mesh.py::FaceMeshTracker` (used only for liveness/blink detection) does
not need to care about this either way -- EAR is a ratio of relative distances between
eye landmarks, which mirroring doesn't change.
