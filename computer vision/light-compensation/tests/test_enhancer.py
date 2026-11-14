import cv2
import numpy as np
import pytest
from evals.synth import darken, make_scene, to_level
from src.config import Settings
from src.enhancer import Enhancer, gamma_lut, solve_gamma
from src.illumination import estimate
from src.preprocessing import preprocess

S = Settings()


def enhance(frame, gain, level=None, settings=S):
    pre = preprocess(frame, settings.analysis_width)
    lv = estimate(pre.luma_small, 3, 3).level if level is None else level
    return Enhancer(settings).enhance(frame, pre, gain, lv)


def test_identity_at_unity_gain_returns_same_pixels():
    f = make_scene()
    r = enhance(f, 1.0)
    assert not r.applied and np.array_equal(r.frame, f) and r.gamma == 1.0


def test_gamma_hits_target_median_without_clahe():
    s = Settings(clahe_clip_max=1.0, denoise_gain_min=99, chroma_blur_gain_min=99)
    dark = darken(make_scene(), 0.2, np.random.default_rng(0))
    level = to_level(dark)
    r = enhance(dark, 1.5, settings=s)
    assert to_level(r.frame) == pytest.approx(min(level * 1.5, s.target_ceiling), abs=0.02)


def test_enhancement_brightens_and_is_valid_image():
    dark = darken(make_scene(), 0.15, np.random.default_rng(1))
    r = enhance(dark, 2.0)
    assert r.applied and r.frame.shape == dark.shape and r.frame.dtype == np.uint8
    assert to_level(r.frame) > to_level(dark) * 1.4


def test_hue_preserved():
    """CIELAB hue angle (chroma channels untouched) must survive the brightening."""
    dark = darken(make_scene(), 0.3, np.random.default_rng(2), read_noise=0.0, shot=0.0)
    pre = preprocess(dark, S.analysis_width)
    r = Enhancer(Settings(denoise_gain_min=99, chroma_blur_gain_min=99)).enhance(
        dark, pre, 1.8, 0.3
    )

    def hue_deg(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        a, b = lab[:, :, 1] - 128, lab[:, :, 2] - 128
        m = np.hypot(a, b) > 12
        return np.degrees(np.median(np.arctan2(b[m], a[m])))

    assert abs(hue_deg(r.frame) - hue_deg(dark)) < 5


def test_gamma_floor_limits_extreme_boost():
    assert solve_gamma(50.0, 0.05, 0.35, 0.92) == pytest.approx(0.35)


def test_solve_gamma_edge_cases():
    assert solve_gamma(1.0, 0.5, 0.35, 0.92) == 1.0
    assert solve_gamma(2.0, 0.0, 0.35, 0.92) >= 0.35  # level clamped, no log(0)
    assert solve_gamma(2.0, 1.0, 0.35, 0.92) <= 1.0
    g = solve_gamma(2.0, 0.3, 0.35, 0.92)
    assert 0.3**g == pytest.approx(0.6, abs=1e-6)


def test_gamma_lut_monotonic_and_bounds():
    lut = gamma_lut(0.5)
    assert lut[0] == 0 and lut[255] == 255 and np.all(np.diff(lut.astype(int)) >= 0)


def test_clahe_clip_scales_with_gain():
    e = Enhancer(S)
    assert e.clahe_clip_for(1.0) == S.clahe_clip_min
    assert e.clahe_clip_for(S.max_gain) == pytest.approx(S.clahe_clip_max)
    assert e.clahe_clip_for(1.5) < e.clahe_clip_for(4.0)
    assert e.clahe_clip_for(1000) == pytest.approx(S.clahe_clip_max)


def test_denoise_reduces_noise_at_high_gain():
    flat = np.full((120, 160, 3), 40, np.uint8)
    noisy = np.clip(flat + np.random.default_rng(3).normal(0, 6, flat.shape), 0, 255).astype(
        np.uint8
    )
    on = enhance(noisy, 2.5, settings=Settings(clahe_clip_max=1.0))
    off = enhance(
        noisy,
        2.5,
        settings=Settings(clahe_clip_max=1.0, denoise_gain_min=99, chroma_blur_gain_min=99),
    )
    assert on.frame.astype(float).std() < off.frame.astype(float).std()


def test_input_not_mutated():
    f = darken(make_scene(), 0.2, np.random.default_rng(4))
    before = f.copy()
    enhance(f, 2.0)
    assert np.array_equal(f, before)
