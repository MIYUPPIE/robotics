import math
from dataclasses import replace

import cv2
import numpy as np
import pytest
from evals.synth import add_occluder, darken, make_scene
from src.config import Settings
from src.linear_light import decode
from src.reference_enhancer import (
    FieldController,
    FieldEstimate,
    FieldEstimator,
    ReferenceEnhancer,
    smoothstep,
)

H, W = 120, 160
S = replace(Settings(), analysis_width=80, lock_exposure=False)
SCENE = make_scene(h=H, w=W)


def ref_image():
    """Clean reference like calibration produces: average of many noisy frames."""
    rng = np.random.default_rng(0)
    acc = np.mean([decode(darken(SCENE, 1.0, rng)) for _ in range(20)], axis=0)
    return np.clip(np.round(255 * acc ** (1 / 2.2)), 0, 255).astype(np.uint8)


REF = ref_image()


def delta_e(a, b):
    la = cv2.cvtColor(a.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    lb = cv2.cvtColor(b.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    return float(np.sqrt(((la - lb) ** 2).sum(axis=2)).mean())


def run(field, frames=30, seed=1, settings=S, occluder=None):
    enh = ReferenceEnhancer(settings, REF)
    rng = np.random.default_rng(seed)
    out = None
    for i in range(frames):
        f = darken(SCENE, field, rng)
        if occluder is not None:
            f = add_occluder(f, occluder, 0.25, 6)
        out, est = enh.process(f, i / 15)
    return enh, out, est


def uniform(k):
    return np.full((H, W, 3), k, np.float32)


def test_smoothstep():
    assert smoothstep(-1) == 0 and smoothstep(2) == 1 and smoothstep(0.5) == pytest.approx(0.5)


def test_same_light_is_identity():
    enh, out, est = run(uniform(1.0))
    assert not out.applied and est.deviation < S.field_gate_lo
    # frame object passes through untouched
    f = darken(SCENE, uniform(1.0), np.random.default_rng(9))
    res, _ = enh.process(f, 99.0)
    assert res.frame is f


@pytest.mark.parametrize("k", [0.5, 0.2, 0.1])
def test_uniform_dimming_restored_to_reference(k):
    _, out, _ = run(uniform(k))
    assert out.applied
    assert delta_e(out.frame, SCENE) < 4.0
    dark = darken(SCENE, uniform(k), np.random.default_rng(3))
    assert delta_e(out.frame, SCENE) < 0.5 * delta_e(dark, SCENE)


def test_uneven_lighting_restored():
    xs = np.linspace(0, 1, W, dtype=np.float32)[None, :, None]
    field = np.ones((H, W, 3), np.float32) * (0.1 + 0.5 * xs)
    _, out, _ = run(field)
    left, right = out.frame[:, : W // 4], out.frame[:, -W // 4 :]
    sl, sr = SCENE[:, : W // 4], SCENE[:, -W // 4 :]
    assert delta_e(left, sl) < 6 and delta_e(right, sr) < 6


def test_colour_cast_corrected():
    tint = np.ones((H, W, 3), np.float32) * 0.5 * np.array([1.3, 1.0, 0.65], np.float32)  # BGR
    _, out, _ = run(tint)
    bgr_out = out.frame.reshape(-1, 3).mean(axis=0)
    bgr_ref = SCENE.reshape(-1, 3).mean(axis=0)
    assert np.abs(bgr_out - bgr_ref).max() < 8


def test_person_does_not_distort_background_field():
    est0 = FieldEstimator(S, REF).estimate(darken(SCENE, uniform(0.15), np.random.default_rng(2)))
    f = add_occluder(darken(SCENE, uniform(0.15), np.random.default_rng(2)), 100, 0.3, 4)
    est1 = FieldEstimator(S, REF).estimate(f)
    # columns left of the person: field must match the no-person estimate
    cols = slice(0, int(S.field_cols * 0.45))
    assert np.abs(est1.log_field[:, cols] - est0.log_field[:, cols]).max() < 0.2
    assert abs(est1.deviation - est0.deviation) < 0.1


def test_field_clipped_to_gain_limits_and_finite():
    black = np.zeros((H, W, 3), np.uint8)
    est = FieldEstimator(S, REF).estimate(black)
    assert np.isfinite(est.log_field).all()
    assert est.log_field.max() <= math.log(S.field_max_gain) + 1e-6


def test_black_reference_cells_take_fit_not_garbage():
    ref = REF.copy()
    ref[:40, :40] = 0
    est = FieldEstimator(S, ref).estimate(darken(SCENE, uniform(0.3), np.random.default_rng(4)))
    assert np.isfinite(est.log_field).all()
    spread = est.log_field.reshape(-1, 3).std(axis=0).max()
    assert spread < 0.35


def test_controller_gate_and_instant_release():
    c = FieldController(S)
    big = FieldEstimate(np.full((S.field_rows, S.field_cols, 3), 2.0, np.float32), 2.0)
    none = FieldEstimate(np.zeros((S.field_rows, S.field_cols, 3), np.float32), 0.0)
    assert c.gate(0.0) == 0 and c.gate(10.0) == 1 and 0 < c.gate(0.07) < 1
    f1 = c.update(big, 0.0).copy()
    f2 = c.update(big, 0.05).copy()
    assert 0 <= f1.max() < f2.max() < 2.0  # eases in
    f3 = c.update(none, 0.10)
    assert np.abs(f3).max() == 0  # light returned: released at once
    c.reset()
    assert np.abs(c.state).max() == 0


def test_small_deviation_below_gate_applies_nothing():
    c = FieldController(S)
    est = FieldEstimate(np.full((S.field_rows, S.field_cols, 3), 0.02, np.float32), 0.02)
    assert np.abs(c.update(est, 0.0)).max() == 0


def test_noise_estimate_tracks_actual_noise():
    enh = ReferenceEnhancer(S, REF)
    rng = np.random.default_rng(0)
    base = np.full((H, W, 3), 60, np.float32)
    for sigma in (1.0, 6.0):
        enh.noise_levels = 3.0
        for _ in range(80):
            f = np.clip(base + rng.normal(0, sigma, base.shape), 0, 255).astype(np.uint8)
            enh._update_noise(f)
        assert enh.noise_levels == pytest.approx(sigma, rel=0.35)


def test_static_noise_is_collapsed_toward_reference():
    on = run(uniform(0.08))[1].frame
    off = run(uniform(0.08), settings=replace(S, background_shrink_kappa=0.0))[1].frame
    assert delta_e(on, SCENE) < delta_e(off, SCENE)


def test_real_change_survives_shrinkage():
    """A bright object that was not in the reference must still show up."""
    enh = ReferenceEnhancer(S, REF)
    rng = np.random.default_rng(5)
    scene2 = SCENE.copy()
    scene2[40:80, 60:100] = (230, 40, 40)  # blue block appears
    out = None
    for i in range(30):
        out, _ = enh.process(darken(scene2, uniform(0.2), rng), i / 15)
    block = out.frame[45:75, 65:95].reshape(-1, 3).mean(axis=0)
    assert block[0] > 150 and block[0] > block[2] + 80  # still clearly blue, not erased


def test_shapes_and_dtypes():
    _, out, _ = run(uniform(0.2), frames=10)
    assert out.frame.shape == (H, W, 3) and out.frame.dtype == np.uint8


def test_reset_clears_temporal_state():
    enh, _, _ = run(uniform(0.2), frames=10)
    enh.reset()
    assert enh._prev is None and enh._prev_raw is None
