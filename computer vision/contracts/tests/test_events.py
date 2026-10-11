from contracts.events import CursorSample, Region


class TestRegion:
    def test_contains_inside_point(self):
        r = Region("PARTY_APC", x0=0, y0=0, x1=100, y1=100)
        assert r.contains(50, 50)

    def test_contains_boundary_is_inclusive(self):
        r = Region("PARTY_APC", x0=0, y0=0, x1=100, y1=100)
        assert r.contains(0, 0)
        assert r.contains(100, 100)

    def test_excludes_outside_point(self):
        r = Region("PARTY_APC", x0=0, y0=0, x1=100, y1=100)
        assert not r.contains(101, 50)
        assert not r.contains(50, -1)


def test_cursor_sample_absent_hand():
    sample = CursorSample(x=0.0, y=0.0, t=1.0, present=False)
    assert sample.present is False
