import pytest

from contracts.events import Region
from hand_ui.regions import fit_within_region, grid_layout, hit_test


class TestHitTest:
    def test_finds_containing_region(self):
        regions = [Region("A", 0, 0, 100, 100), Region("B", 200, 0, 300, 100)]
        assert hit_test(regions, 50, 50) == "A"
        assert hit_test(regions, 250, 50) == "B"

    def test_returns_none_outside_all_regions(self):
        regions = [Region("A", 0, 0, 100, 100)]
        assert hit_test(regions, 500, 500) is None

    def test_empty_region_list(self):
        assert hit_test([], 0, 0) is None


class TestGridLayout:
    def test_places_first_region_at_origin(self):
        regions = grid_layout(["A"], origin=(10, 20), cell_size=(100, 80), columns=3)
        assert regions[0] == Region("A", x0=10, y0=20, x1=110, y1=100)

    def test_wraps_to_next_row_after_columns(self):
        regions = grid_layout(["A", "B", "C", "D"], origin=(0, 0), cell_size=(100, 100), columns=3)
        by_id = {r.region_id: r for r in regions}
        assert by_id["C"].x0 == 200  # 3rd column, same row
        assert by_id["D"].y0 == 100  # wraps to row 2
        assert by_id["D"].x0 == 0

    def test_applies_gap_between_cells(self):
        regions = grid_layout(
            ["A", "B"], origin=(0, 0), cell_size=(100, 100), columns=2, gap=(10, 10)
        )
        by_id = {r.region_id: r for r in regions}
        assert by_id["B"].x0 == 110  # 100 (cell) + 10 (gap)

    def test_produces_non_overlapping_regions_for_a_15_party_grid(self):
        party_ids = [f"PARTY_{i}" for i in range(15)]
        regions = grid_layout(
            party_ids, origin=(0, 0), cell_size=(300, 200), columns=5, gap=(20, 20)
        )
        assert len(regions) == 15
        # No two regions should overlap.
        for i, a in enumerate(regions):
            for b in regions[i + 1 :]:
                overlaps_x = a.x0 < b.x1 and b.x0 < a.x1
                overlaps_y = a.y0 < b.y1 and b.y0 < a.y1
                assert not (overlaps_x and overlaps_y), f"{a.region_id} overlaps {b.region_id}"

    def test_rejects_nonpositive_columns(self):
        with pytest.raises(ValueError):
            grid_layout(["A"], origin=(0, 0), cell_size=(10, 10), columns=0)


class TestFitWithinRegion:
    def test_centers_a_square_image_in_a_square_region(self):
        region = Region("A", 0, 0, 100, 100)
        x, y, w, h = fit_within_region(50, 50, region, padding_frac=0.0)
        assert (x, y, w, h) == (0, 0, 100, 100)

    def test_width_constrained_image_is_letterboxed_vertically(self):
        # A wide logo (e.g. YPP's) in a squarer card: width fills first, height
        # shrinks to match aspect ratio, and stays vertically centered.
        region = Region("A", 0, 0, 300, 180)
        x, y, w, h = fit_within_region(700, 190, region, padding_frac=0.0)
        assert w == 300
        assert h == pytest.approx(300 * 190 / 700, abs=1)
        assert x == 0
        assert y == pytest.approx((180 - h) / 2, abs=1)

    def test_height_constrained_image_is_pillarboxed_horizontally(self):
        region = Region("A", 0, 0, 180, 300)
        x, y, w, h = fit_within_region(190, 700, region, padding_frac=0.0)
        assert h == 300
        assert w == pytest.approx(300 * 190 / 700, abs=1)
        assert y == 0
        assert x == pytest.approx((180 - w) / 2, abs=1)

    def test_padding_shrinks_the_available_box(self):
        region = Region("A", 0, 0, 100, 100)
        x, y, w, h = fit_within_region(50, 50, region, padding_frac=0.1)
        # 0.1 * min(100, 100) = 10px padding on every edge -> 80x80 box.
        assert (x, y, w, h) == (10, 10, 80, 80)

    def test_offset_region_places_image_relative_to_its_origin(self):
        region = Region("A", 200, 300, 300, 400)
        x, y, w, h = fit_within_region(50, 50, region, padding_frac=0.0)
        assert (x, y, w, h) == (200, 300, 100, 100)

    def test_result_fits_entirely_inside_the_region(self):
        region = Region("A", 10, 10, 310, 190)
        for image_w, image_h in [(300, 270), (700, 190), (190, 700), (1, 1), (320, 320)]:
            x, y, w, h = fit_within_region(image_w, image_h, region)
            assert x >= region.x0 and y >= region.y0
            assert x + w <= region.x1 + 1  # rounding tolerance
            assert y + h <= region.y1 + 1

    def test_rejects_nonpositive_image_dimensions(self):
        region = Region("A", 0, 0, 100, 100)
        with pytest.raises(ValueError):
            fit_within_region(0, 50, region)
        with pytest.raises(ValueError):
            fit_within_region(50, -1, region)
