# storage

SQLite schema + repository layer shared by `kiosk_app`, `registration_app`, and
`admin_app`. See [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md#ballot-secrecy)
for the reasoning behind the `voters` / `votes` / `voted_log` split.

- `schema.sql` -- the full DDL (single source of truth, applied by `db.py` on connect).
- `db.py` -- `connect(path)` (on-disk, WAL mode) / `connect_memory()` (tests).
- `repositories/voter_repository.py` -- voter CRUD + `all_embeddings()` (the 1:N
  matching gallery for `biometric/matcher.py`).
- `repositories/vote_repository.py` -- **the only code allowed to write to `votes` or
  `voted_log`**. `submit_vote_session()` is a single atomic transaction: N anonymous
  vote rows + 1 voted_log row + a `has_voted` flip, or nothing at all.
- `repositories/voted_log_repository.py` -- read-only queries against `voted_log`.
- `repositories/override_repository.py` -- audit log for registration-officer
  duplicate-match overrides.

## Tests

`pytest storage/` -- schema round-trip, CRUD correctness, and the atomicity guarantee
(`test_vote_repository.py::TestSubmitVoteSessionAtomicity`, which forces a
mid-transaction failure via a wrapping proxy and asserts every table is untouched).
All in-memory, <1s total.
