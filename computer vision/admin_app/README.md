# admin_app

PIN-gated Tkinter results dashboard. Run: `python -m admin_app.app` (needs
`ADMIN_PIN_HASH` set, see `.env.example` for how to generate one).

- `auth.py` -- `PinGate` / `hash_pin` / `verify_pin`, stdlib-only (`pbkdf2_hmac` +
  constant-time `hmac.compare_digest`), locks out after repeated failed attempts.
- `tally.py` -- `per_office_tally`, `turnout`, and `integrity_check` (the executable
  form of the `COUNT(votes) == offices_per_session x COUNT(voted_log)` invariant
  from [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md#ballot-secrecy)).
- `export.py` -- `export_tallies_csv`: office/party/vote-count aggregates **only**,
  queries only the anonymous `votes` table, never `voters`.
- `app.py` -- tabbed per-office tally view (2s auto-refresh), turnout display, CSV
  export button.

## Tests

`pytest admin_app/` -- PIN hashing/verification/lockout, tally math against a
hand-computed seeded fixture DB, and `test_export.py::test_contains_no_pii` (string-
searches exported CSV content for `embedding`/known seeded `voter_id`s/surnames and
asserts absence). No Tkinter, no real DB file, <2s.
