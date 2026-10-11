# kiosk_app

The voting kiosk: a pure state machine driving a fullscreen, gesture-only OpenCV UI.

- `state_machine.py` -- `KioskStateMachine`, the deterministic core. Every screen
  transition (`IDLE_ATTRACT → LANGUAGE_SELECT → FACE_AUTH → OFFICE_BALLOT x5 →
  REVIEW → SUBMITTING → THANK_YOU`, plus every failure/timeout path) is a plain
  method call with an injected clock and an injected `submit_vote_session` -- zero
  camera/database/MediaPipe dependency. See
  [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md#deterministic-core-vs-thin-cv-shell).
- `session.py` -- `VotingSession`, in-memory only. Never touches the database (see
  [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md#ballot-secrecy)) -- abandoning a
  session at any point before the REVIEW screen's SUBMIT writes nothing.
- `config.py` -- per-state timeouts, screen dimensions.
- `screens/` -- one module per screen: region layout (where the dwell-selectable
  boxes are) + `render()` (OpenCV drawing). `office_ballot.py` implements the
  mandatory two-step select-then-confirm party picker; `review.py` implements the
  DRE-voting-machine-style review screen with per-office edit + a two-step SUBMIT.
- `main.py` -- `KioskApp`, the thin shell: wires a real camera frame, `biometric`
  (face identify), `hand_ui` (dwell-driven cursor), and `voice` (pre-rendered audio)
  to the state machine. Dependency-injected (db connection, embedder, hand tracker,
  clock) so `evals/full_session_eval.py` can drive it with fakes.

## Tests vs. evals

Gate tests (`tests/test_state_machine.py`, `tests/test_session.py`) cover every
transition and every timeout path with an injected fake clock and a spy submit
function -- no camera, no real database, <2s.

`evals/full_session_eval.py` scripts full voter sessions (fake face-identify result
+ a cursor path computed from the real screen region-layout functions) through the
real `KioskApp` and a real in-memory database, and asserts exactly 5N votes / N
voted_log rows land, plus that a replayed already-voted session is blocked and adds
no extra votes. Run: `python -m kiosk_app.evals.full_session_eval`.
