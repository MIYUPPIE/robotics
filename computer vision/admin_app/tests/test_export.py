import csv

from admin_app.export import export_tallies_csv

PARTY_NAMES = {
    "APC": "All Progressives Congress",
    "PDP": "Peoples Democratic Party",
    "LP": "Labour Party",
}


class TestExportTalliesCsv:
    def test_writes_expected_header(self, seeded_conn, tmp_path):
        path = export_tallies_csv(seeded_conn, tmp_path / "tallies.csv", PARTY_NAMES)
        with path.open() as f:
            header = next(csv.reader(f))
        assert header == ["office", "party_code", "party_name", "vote_count"]

    def test_row_counts_match_tally(self, seeded_conn, tmp_path):
        path = export_tallies_csv(seeded_conn, tmp_path / "tallies.csv", PARTY_NAMES)
        with path.open() as f:
            rows = list(csv.DictReader(f))
        president_apc = next(
            r for r in rows if r["office"] == "PRESIDENT" and r["party_code"] == "APC"
        )
        assert president_apc["vote_count"] == "1"
        assert president_apc["party_name"] == "All Progressives Congress"

    def test_contains_no_pii(self, seeded_conn, tmp_path):
        """The export must never leak voter identity or embeddings, even
        indirectly -- this is the enforced form of the ballot-secrecy
        guarantee for the one artifact meant to leave the machine."""
        path = export_tallies_csv(seeded_conn, tmp_path / "tallies.csv", PARTY_NAMES)
        content = path.read_text()

        assert "embedding" not in content.lower()
        assert "PVR-LAG-20260101-SEED00" not in content
        assert "PVR-LAG-20260101-SEED01" not in content
        assert "PVR-LAG-20260101-SEED02" not in content
        assert "Seed0" not in content  # surname
        assert "voter_id" not in content.lower()

    def test_unvoted_registered_voter_never_appears(self, seeded_conn, tmp_path):
        path = export_tallies_csv(seeded_conn, tmp_path / "tallies.csv", PARTY_NAMES)
        content = path.read_text()
        assert "SEED02" not in content  # registered but never voted
