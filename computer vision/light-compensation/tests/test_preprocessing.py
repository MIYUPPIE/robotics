import numpy as np
import pytest
from src.preprocessing import downscale_luma, preprocess, validate_frame


def test_preprocess_shapes_and_range():
    frame = np.full((240, 320, 3), 128, np.uint8)
    pre = preprocess(frame, 160)
    assert pre.lab.shape == (240, 320, 3)
    assert pre.luma_small.shape == (120, 160)
    assert pre.luma_small.dtype == np.float32
    assert 0.0 <= pre.luma_small.min() and pre.luma_small.max() <= 1.0


def test_uniform_gray_luma_value():
    pre = preprocess(np.full((60, 80, 3), 255, np.uint8), 40)
    assert pre.luma_small.mean() == pytest.approx(1.0, abs=0.01)
    pre = preprocess(np.zeros((60, 80, 3), np.uint8), 40)
    assert pre.luma_small.mean() == pytest.approx(0.0, abs=0.01)


def test_small_frame_not_upscaled():
    assert downscale_luma(np.zeros((20, 30), np.uint8), 160).shape == (20, 30)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((10, 10), np.uint8),
        np.zeros((10, 10, 4), np.uint8),
        np.zeros((10, 10, 3), np.float32),
        np.zeros((1, 10, 3), np.uint8),
        "frame",
    ],
)
def test_validate_rejects_bad_frames(bad):
    with pytest.raises(ValueError):
        validate_frame(bad)
