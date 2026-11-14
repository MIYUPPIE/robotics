"""pytest wrapper so `make eval` (pytest -m eval) runs the compensation eval."""

import pytest

from evals.compensation_eval import run


@pytest.mark.eval
def test_compensation_meets_all_bars():
    failed = [f"{c.name}: {c.value} (bar {c.bar})" for c in run() if not c.passed]
    assert not failed, "\n".join(failed)
