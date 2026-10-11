# contracts

Typed data shapes shared by every other concern in this project -- see
[docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) (ADR-1) for why this is a plain Python
module boundary rather than a network API.

- `schemas.py` -- domain types: `Language`, `Office`, `Party`, `VoterRecord`,
  `MatchResult`, `DuplicateCheckResult`, `LivenessResult`, `VoteSelection`,
  `is_age_eligible`.
- `events.py` -- the hand-tracking cursor/dwell event types (`CursorSample`, `Region`,
  `DwellEvent` variants) that flow from `hand_ui` into `kiosk_app`.

Nothing here imports from any other concern directory -- this is the leaf of the
dependency graph. If you find yourself wanting to import `biometric` or `kiosk_app`
from here, the type you're adding probably belongs somewhere else.

## Tests

`pytest contracts/` -- pure dataclass/validation tests, no I/O, <1s.
