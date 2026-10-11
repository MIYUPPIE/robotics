import re
from datetime import date

import pytest

from biometric.voter_id import generate_voter_id

ID_PATTERN = re.compile(r"^PVR-[A-Z]{3}-\d{8}-[2-9A-HJ-NP-Z]{6}$")


class TestGenerateVoterId:
    def test_matches_expected_format(self):
        voter_id = generate_voter_id("LAG", date(2026, 8, 1), exists=lambda _id: False)
        assert ID_PATTERN.match(voter_id), voter_id

    def test_embeds_state_and_date(self):
        voter_id = generate_voter_id("LAG", date(2026, 8, 1), exists=lambda _id: False)
        assert voter_id.startswith("PVR-LAG-20260801-")

    def test_excludes_visually_ambiguous_characters(self):
        for _ in range(50):
            voter_id = generate_voter_id("LAG", date(2026, 8, 1), exists=lambda _id: False)
            suffix = voter_id.split("-")[-1]
            assert not set(suffix) & set("01OIL")

    def test_retries_on_collision(self):
        calls = {"n": 0}

        def exists(candidate: str) -> bool:
            calls["n"] += 1
            return calls["n"] <= 3  # first 3 candidates are "taken"

        voter_id = generate_voter_id("LAG", date(2026, 8, 1), exists=exists)
        assert calls["n"] == 4
        assert ID_PATTERN.match(voter_id)

    def test_gives_up_after_max_attempts(self):
        with pytest.raises(RuntimeError):
            generate_voter_id("LAG", date(2026, 8, 1), exists=lambda _id: True)

    def test_is_not_shaped_like_a_real_inec_vin(self):
        """A real INEC VIN is 19 raw digits -- this must never look like one."""
        voter_id = generate_voter_id("LAG", date(2026, 8, 1), exists=lambda _id: False)
        digits_only = voter_id.replace("-", "")
        assert not (digits_only.isdigit() and len(digits_only) == 19)
