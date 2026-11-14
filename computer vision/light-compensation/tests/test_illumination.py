import numpy as np
import pytest
from src.illumination import LevelFilter, dropped_cell_fraction, estimate


def test_uniform_plane():
    s = estimate(np.full((30, 30), 0.4, np.float32), 3, 3)
    assert s.level == pytest.approx(0.4) and s.median == pytest.approx(0.4)
    assert s.mean == pytest.approx(0.4) and s.cells.shape == (9,)


def test_percentiles_ordered():
    rng = np.random.default_rng(0)
    s = estimate(rng.random((60, 60)).astype(np.float32), 3, 3)
    assert s.p10 < s.median < s.p90


def test_occluder_covering_few_cells_does_not_move_level():
    plane = np.full((90, 90), 0.6, np.float32)
    base = estimate(plane, 3, 3).level
    plane[:, 30:60] = 0.02  # one full column of cells (3 of 9) blacked out
    assert estimate(plane, 3, 3).level == pytest.approx(base)


def test_non_divisible_grid():
    s = estimate(np.full((31, 47), 0.5, np.float32), 3, 3)
    assert s.level == pytest.approx(0.5)


def test_plane_smaller_than_grid():
    with pytest.raises(ValueError):
        estimate(np.zeros((2, 2), np.float32), 3, 3)


def test_dropped_fraction():
    ref = np.full(9, 0.5, np.float32)
    cur = ref.copy()
    cur[:3] = 0.1
    assert dropped_cell_fraction(cur, ref, 0.9) == pytest.approx(3 / 9)
    assert dropped_cell_fraction(ref, ref, 0.9) == 0.0
    assert dropped_cell_fraction(ref * 0.1, ref, 0.9) == 1.0


def test_dropped_fraction_ignores_near_black_reference_cells():
    ref = np.zeros(9, np.float32)
    assert dropped_cell_fraction(np.zeros(9, np.float32), ref, 0.9) == 0.0
    ref = np.array([0.0, 0.0, 0.5, 0.5], np.float32)
    cur = np.array([0.0, 0.0, 0.1, 0.5], np.float32)
    assert dropped_cell_fraction(cur, ref, 0.9) == pytest.approx(0.5)  # only 2 valid cells


def test_dropped_fraction_shape_mismatch():
    with pytest.raises(ValueError):
        dropped_cell_fraction(np.zeros(4), np.zeros(9), 0.9)


def test_level_filter_rise_instant_fall_smoothed():
    f = LevelFilter(fall_tau_s=0.5)
    assert f.update(0.5, 0.0) == 0.5
    assert f.update(0.8, 0.1) == 0.8  # rises pass straight through
    v = f.update(0.2, 0.2)  # falls are low-passed
    assert 0.2 < v < 0.8
    for i in range(3, 80):
        v = f.update(0.2, i * 0.1)
    assert v == pytest.approx(0.2, abs=1e-3)


def test_level_filter_zero_tau_and_reset():
    f = LevelFilter(0.0)
    f.update(0.8, 0.0)
    assert f.update(0.1, 0.1) == 0.1
    f.reset()
    assert f.value is None
