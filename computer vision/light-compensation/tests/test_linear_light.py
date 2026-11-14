import numpy as np
import pytest
from src.linear_light import GAMMA, decode, encode, encode_fast


def test_decode_endpoints_and_dtype():
    img = np.array([[[0, 128, 255]]], np.uint8)
    lin = decode(img)
    assert lin.dtype == np.float32 and lin[0, 0, 0] == 0 and lin[0, 0, 2] == 1.0
    assert lin[0, 0, 1] == pytest.approx((128 / 255) ** GAMMA, rel=1e-6)


def test_roundtrip_within_one_level():
    img = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, axis=2)
    for enc in (encode, encode_fast):
        back = enc(decode(img)).astype(int)
        assert np.abs(back - img.astype(int)).max() <= 1, enc.__name__


def test_saturation_and_zero():
    x = np.array([[[0.0, 1.0, 5.0]]], np.float32)
    for enc in (encode, encode_fast):
        out = enc(x)
        assert out[0, 0, 0] == 0 and out[0, 0, 1] == 255 and out[0, 0, 2] == 255


def test_encode_negative_and_garbage_clip_to_zero():
    assert encode(np.array([[[-1.0, -1e-9, 0.0]]], np.float32)).max() == 0


def test_fast_matches_reference_encoder_and_is_monotonic():
    x = np.linspace(0, 1.2, 5000, dtype=np.float32).reshape(1, -1, 1).repeat(3, axis=2)
    a, b = encode(x).astype(int), encode_fast(x).astype(int)
    assert np.abs(a - b).max() <= 4
    assert (np.diff(b[0, :, 0]) >= 0).all()


def test_light_scaling_is_linear_after_decode():
    img = np.full((4, 4, 3), 100, np.uint8)
    assert np.allclose(decode(img) * 2, decode(img) + decode(img))
