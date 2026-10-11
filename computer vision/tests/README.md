# tests

Cross-cutting integration tests that span more than one concern directory (e.g. a
full registration→voting→tally dry run). Everything that tests a single concern in
isolation belongs in that concern's own `tests/` directory instead -- this directory
is only for tests that would otherwise have no natural home.

Shared fixtures usable by every concern's test suite live in the repo-root
`conftest.py`, not here -- see that file's docstring for why it isn't nested inside
this package.
