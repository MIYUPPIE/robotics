from scripts.seed_demo_data import (
    DEMO_MARKER_SURNAME_PREFIX,
    is_demo_surname,
    non_demo_surnames,
    seed,
)
from storage import db
from storage.repositories import voter_repository


class TestIsDemoSurname:
    def test_recognizes_demo_prefix(self):
        assert is_demo_surname(f"{DEMO_MARKER_SURNAME_PREFIX}0")

    def test_rejects_real_surname(self):
        assert not is_demo_surname("Okafor")


class TestNonDemoSurnames:
    def test_filters_out_demo_entries(self):
        result = non_demo_surnames(
            [f"{DEMO_MARKER_SURNAME_PREFIX}0", "Okafor", f"{DEMO_MARKER_SURNAME_PREFIX}1"]
        )
        assert result == ["Okafor"]

    def test_empty_when_all_demo(self):
        assert non_demo_surnames([f"{DEMO_MARKER_SURNAME_PREFIX}0"]) == []


class TestSeed:
    def test_creates_requested_count(self):
        conn = db.connect_memory()
        created = seed(conn, count=4)
        assert len(created) == 4
        assert voter_repository.count(conn) == 4
        conn.close()

    def test_every_seeded_surname_is_demo_marked(self):
        conn = db.connect_memory()
        created = seed(conn, count=3)
        assert all(is_demo_surname(v.surname) for v in created)
        conn.close()

    def test_seeded_voters_have_distinct_embeddings(self):
        conn = db.connect_memory()
        seed(conn, count=3)
        gallery = voter_repository.all_embeddings(conn)
        similarities = [float(a[1] @ b[1]) for i, a in enumerate(gallery) for b in gallery[i + 1 :]]
        assert all(s < 0.9 for s in similarities)
        conn.close()
