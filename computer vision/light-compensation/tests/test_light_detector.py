from src.config import Settings
from src.light_detector import LightDetector, LightState

S = Settings(confirm_seconds=0.5, min_recovery_s=1.0, recovery_timeout_s=10.0)
N, D, K, R = LightState.NORMAL, LightState.DIMMED, LightState.DARK, LightState.RECOVERY


def feed(det, ratio, seconds, t0, uniform=True, fps=10):
    t = t0
    transitions = []
    for _ in range(int(seconds * fps)):
        tr = det.update(ratio, uniform, t)
        if tr:
            transitions.append(tr)
        t += 1 / fps
    return t, transitions


def test_starts_normal_and_stays_with_steady_light():
    d = LightDetector(S)
    _, trs = feed(d, 1.0, 5, 0)
    assert d.state is N and not trs


def test_lights_off_goes_dark_after_confirm():
    d = LightDetector(S)
    t, _ = feed(d, 1.0, 1, 0)
    t, trs = feed(d, 0.2, 2, t)
    assert d.state is K
    assert [(x.old, x.new) for x in trs] == [(N, K)]
    assert trs[0].t - 1.0 >= S.confirm_seconds - 1e-9  # debounced


def test_brief_dip_shorter_than_confirm_is_ignored():
    d = LightDetector(S)
    t, _ = feed(d, 1.0, 1, 0)
    t, _ = feed(d, 0.2, 0.3, t)
    t, trs = feed(d, 1.0, 2, t)
    assert d.state is N and not trs


def test_moderate_dim_goes_dimmed_not_dark():
    d = LightDetector(S)
    t, _ = feed(d, 1.0, 1, 0)
    feed(d, 0.8, 2, t)
    assert d.state is D


def test_hysteresis_no_chatter_inside_band():
    d = LightDetector(S)
    t, _ = feed(d, 1.0, 1, 0)
    t, _ = feed(d, 0.85, 2, t)
    assert d.state is D
    # ratio hovering between dim_enter (0.88) and dim_exit (0.93) must not flip state
    trs_all = []
    for _ in range(10):
        t, a = feed(d, 0.90, 1, t)
        t, b = feed(d, 0.87, 1, t)
        trs_all += a + b
    assert d.state is D and not trs_all


def test_dimmed_returns_to_normal_above_exit():
    d = LightDetector(S)
    t, _ = feed(d, 0.85, 3, 0)
    assert d.state is D
    feed(d, 0.97, 2, t)
    assert d.state is N


def test_non_uniform_drop_never_leaves_normal():
    d = LightDetector(S)
    t, _ = feed(d, 1.0, 1, 0)
    _, trs = feed(d, 0.3, 5, t, uniform=False)
    assert d.state is N and not trs


def test_full_cycle_dark_recovery_normal():
    d = LightDetector(S)
    t, _ = feed(d, 0.2, 2, 0)
    assert d.state is K
    t, trs = feed(d, 1.0, 0.8, t)
    assert d.state is R
    t, trs2 = feed(d, 1.0, 2.5, t)
    assert d.state is N
    assert [x.new for x in trs + trs2] == [R, N]


def test_recovery_held_for_min_recovery_s():
    d = LightDetector(S)
    t, _ = feed(d, 0.2, 2, 0)
    t, _ = feed(d, 1.0, 0.8, t)
    assert d.state is R
    entered = d._state_since
    t, trs = feed(d, 1.0, 5, t)
    nrm = [x for x in trs if x.new is N][0]
    assert nrm.t - entered >= S.min_recovery_s - 1e-9


def test_recovery_back_to_dark_if_light_drops_again():
    d = LightDetector(S)
    t, _ = feed(d, 0.2, 2, 0)
    t, _ = feed(d, 0.9, 0.8, t)
    assert d.state is R
    feed(d, 0.2, 2, t)
    assert d.state is K


def test_recovery_times_out_to_dimmed_at_new_steady_level():
    d = LightDetector(S)
    t, _ = feed(d, 0.2, 2, 0)
    feed(d, 0.86, 14, t)  # recovered to 0.86: above recovery 0.84, below dim_exit 0.93
    assert d.state is D


def test_dark_not_left_until_recovery_threshold():
    d = LightDetector(S)
    t, _ = feed(d, 0.2, 2, 0)
    feed(d, 0.80, 3, t)  # between dark_enter and recovery_threshold
    assert d.state is K


def test_reset():
    d = LightDetector(S)
    feed(d, 0.2, 2, 0)
    d.reset()
    assert d.state is N
