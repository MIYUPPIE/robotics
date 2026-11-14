import pytest
from src.config import Settings
from src.controller import CompensationController
from src.light_detector import LightState

S = Settings(alpha=1.0, max_gain=8.0, gain_tau_up_s=0.4, gain_tau_down_s=0.3)
N, D, K = LightState.NORMAL, LightState.DIMMED, LightState.DARK


def run(c, level, ref, state, seconds, t0=0.0, fps=30):
    t, out = t0, None
    for _ in range(int(seconds * fps)):
        out = c.update(level, ref, state, t)
        t += 1 / fps
    return t, out


def test_normal_state_never_compensates():
    c = CompensationController(S)
    _, out = run(c, 0.45, 0.5, N, 2)  # 10% dip but still NORMAL
    assert out.gain == 1.0 and out.target_gain == 1.0
    assert out.compensation_ratio == pytest.approx(0.5 / 0.45)


def test_gain_converges_to_ratio_in_dark():
    c = CompensationController(S)
    _, out = run(c, 0.25, 0.5, K, 5)
    assert out.gain == pytest.approx(2.0, abs=1e-3)
    assert out.ratio == pytest.approx(0.5)


def test_clipped_at_max_gain_and_epsilon_protects_division():
    c = CompensationController(S)
    _, out = run(c, 0.0, 0.5, K, 10)
    assert out.gain == pytest.approx(S.max_gain, abs=1e-3)
    assert out.compensation_ratio == pytest.approx(0.5 / S.epsilon)


def test_alpha_scales_correction():
    c = CompensationController(Settings(alpha=0.5))
    _, out = run(c, 0.125, 0.5, K, 8)
    assert out.gain == pytest.approx(2.0, abs=1e-3)  # sqrt(4)


def test_gain_never_below_one_when_brighter_than_reference():
    c = CompensationController(S)
    _, out = run(c, 0.9, 0.5, D, 3)
    assert out.gain == 1.0


def test_smoothing_ramps_not_jumps():
    c = CompensationController(S)
    c.update(0.25, 0.5, K, 0.0)
    out = c.update(0.25, 0.5, K, 1 / 30)
    assert 1.0 < out.gain < 2.0
    assert out.gain < out.target_gain


def test_release_cap_drops_gain_immediately_when_light_returns():
    c = CompensationController(S)
    t, out = run(c, 0.25, 0.5, K, 5)
    assert out.gain == pytest.approx(2.0, abs=1e-3)
    out = c.update(0.5, 0.5, K, t)  # lights back on, state not updated yet
    assert out.gain == pytest.approx(1.0)


def test_state_gating_release_is_smooth():
    c = CompensationController(S)
    t, _ = run(c, 0.46, 0.5, D, 3)
    g0 = c.gain
    assert g0 > 1.05
    out = c.update(0.46, 0.5, N, t + 1 / 30)  # DIMMED -> NORMAL, ratio unchanged
    assert 1.0 < out.gain < g0


def test_zero_tau_is_instant_and_reset():
    c = CompensationController(Settings(gain_tau_up_s=0.0))
    assert c.update(0.25, 0.5, K, 0.0).gain == pytest.approx(2.0)
    c.reset()
    assert c.gain == 1.0
