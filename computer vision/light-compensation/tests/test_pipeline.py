import json
from dataclasses import replace

import numpy as np
import pytest
from evals.synth import add_occluder, darken, make_scene
from src.calibration import load_reference, save_reference
from src.config import Settings
from src.light_detector import LightState
from src.pipeline import Phase, Pipeline

FPS = 15
S = Settings(calibration_seconds=1.0, min_calibration_frames=10, analysis_width=40)
SCENE = make_scene(h=72, w=96)


class Sim:
    def __init__(self, settings=S, reference=None, seed=0):
        self.p = Pipeline(settings, reference)
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.last = None
        self.events = []

    def run(self, k, seconds, occluder=None):
        for _ in range(int(seconds * FPS)):
            f = darken(SCENE, k, self.rng)
            if occluder is not None:
                f = add_occluder(f, occluder, 0.4)
            self.last = self.p.process(f, self.t)
            if self.last.transition:
                self.events.append((self.t, self.last.transition))
            self.t += 1 / FPS
        return self.last


def calibrated(**kw):
    sim = Sim(**kw)
    sim.run(1.0, 1.5)
    assert sim.p.phase is Phase.RUNNING
    return sim


def test_calibrates_then_runs_and_output_is_untouched_while_calibrating():
    sim = Sim()
    r = sim.run(1.0, 0.3)
    assert r.phase is Phase.CALIBRATING and r.ref_level == 0.0 and 0 < r.calibration_progress < 1
    assert r.frame is not None and np.isnan(r.brightness_error_pct)
    sim.run(1.0, 1.2)
    assert sim.p.reference is not None and 0.2 < sim.p.reference.level < 0.9


def test_calibrated_now_fires_exactly_once():
    sim = Sim()
    flags = []
    for _ in range(int(2 * FPS)):
        r = sim.p.process(darken(SCENE, 1.0, sim.rng), sim.t)
        flags.append(r.calibrated_now)
        sim.t += 1 / FPS
    assert sum(flags) == 1


def test_steady_normal_light_leaves_frames_identical():
    sim = calibrated()
    r = sim.run(1.0, 2.0)
    assert r.state is LightState.NORMAL and r.gain == 1.0
    assert sim.events == []


def test_lights_off_detected_and_brightness_retained():
    sim = calibrated()
    r = sim.run(0.3, 4.0)
    assert r.state in (LightState.DIMMED, LightState.DARK)
    assert r.gain > 1.3
    assert r.brightness_error_pct < 15.0
    assert sim.events and sim.events[0][1].old is LightState.NORMAL


def test_lights_back_on_returns_to_normal_without_overshoot():
    sim = calibrated()
    sim.run(0.2, 4.0)
    errs = []
    for _ in range(int(4 * FPS)):
        r = sim.p.process(darken(SCENE, 1.0, sim.rng), sim.t)
        errs.append(r.brightness_error_pct)
        sim.t += 1 / FPS
    assert r.state is LightState.NORMAL and r.gain == pytest.approx(1.0, abs=0.01)
    assert max(errs) < 15.0  # gain released immediately, no over-bright flash


def test_person_walking_through_does_not_trigger():
    sim = calibrated()
    for x in range(-30, 100, 6):  # occluder sweeps across
        sim.run(1.0, 0.25, occluder=x)
    sim.run(1.0, 1.0)
    assert sim.events == []
    assert sim.last.state is LightState.NORMAL


def test_small_screen_brightness_change_does_not_trigger():
    sim = calibrated()
    sim.run(0.93, 3.0)  # ~7% light change
    assert sim.events == []


def test_recalibrate_restarts_learning():
    sim = calibrated()
    sim.p.recalibrate()
    assert sim.p.phase is Phase.CALIBRATING
    sim.run(0.5, 1.5)
    assert sim.p.phase is Phase.RUNNING
    assert sim.p.reference.level < 0.6  # learned the dimmer room as the new normal


def test_too_dark_calibration_reports_and_retries():
    sim = Sim()
    msgs = []
    for _ in range(int(1.5 * FPS)):
        r = sim.p.process(darken(SCENE, 0.002, sim.rng), sim.t)
        sim.t += 1 / FPS
        if r.message:
            msgs.append(r.message)
    assert sim.p.phase is Phase.CALIBRATING
    assert any("too dark" in m for m in msgs)
    sim.run(1.0, 1.5)  # lights come on: the retry succeeds
    assert sim.p.phase is Phase.RUNNING


def test_stored_reference_survives_restart(tmp_path):
    sim = calibrated()
    path = tmp_path / "ref.json"
    save_reference(sim.p.reference, path)
    ref = load_reference(path, S)
    sim2 = Sim(reference=ref)
    assert sim2.p.phase is Phase.RUNNING  # no recalibration needed
    r = sim2.run(0.3, 4.0)  # room is already dark at startup
    assert r.gain > 1.3 and r.brightness_error_pct < 15.0


def test_exposure_info_saved_in_reference():
    sim = Sim()
    sim.p.set_exposure_info({"locked": True, "exposure": 157.0})
    sim.run(1.0, 1.5)
    assert sim.p.reference.exposure["exposure"] == 157.0


def test_drift_off_by_default_and_tracks_when_enabled():
    sim = calibrated()
    ref0 = sim.p.reference.level
    sim.run(0.97, 5.0)
    assert sim.p.reference.level == ref0
    s = replace(S, drift_rate_per_s=0.5)
    sim = calibrated(settings=s)
    ref0 = sim.p.reference.level
    sim.run(0.97, 5.0)
    assert sim.p.reference.level != ref0


def test_drift_never_absorbs_real_dimming():
    s = replace(S, drift_rate_per_s=0.5)
    sim = calibrated(settings=s)
    ref0 = sim.p.reference.level
    sim.run(0.3, 5.0)
    assert sim.p.reference.level == pytest.approx(ref0, rel=0.05)
    assert sim.last.state is not LightState.NORMAL


def test_measure_output_off_gives_nan():
    sim = calibrated(settings=replace(S, measure_output=False))
    assert np.isnan(sim.run(1.0, 0.2).output_level)


def test_bad_frame_raises():
    with pytest.raises(ValueError):
        Pipeline(S).process(np.zeros((4, 4), np.uint8), 0.0)


def test_reference_json_is_plain_json(tmp_path):
    sim = calibrated()
    save_reference(sim.p.reference, tmp_path / "r.json")
    assert json.loads((tmp_path / "r.json").read_text())["version"] == 1


def test_stale_reference_warns_once_at_startup():
    ref = calibrated().p.reference
    sim = Sim(reference=ref, seed=3)
    msgs = []
    for _ in range(int(6 * FPS)):
        r = sim.p.process(darken(SCENE, 0.2, sim.rng), sim.t)  # already dark at start
        sim.t += 1 / FPS
        if r.message:
            msgs.append(r.message)
    assert len(msgs) == 1 and "press r" in msgs[0] and "DARK" in msgs[0]


def test_no_startup_warning_when_lights_are_normal_or_fresh_calibration():
    ref = calibrated().p.reference
    sim = Sim(reference=ref)
    msgs = [r.message for r in [sim.run(1.0, 2.0)] if r.message]
    assert msgs == []
    sim = calibrated()  # fresh calibration, then lights off within seconds
    msgs = []
    for _ in range(int(3 * FPS)):
        r = sim.p.process(darken(SCENE, 0.2, sim.rng), sim.t)
        sim.t += 1 / FPS
        msgs.append(r.message)
    assert not any(msgs)


def test_late_dimming_on_stored_reference_is_not_a_startup_warning():
    ref = calibrated().p.reference
    sim = Sim(reference=ref)
    sim.run(1.0, 6.0)
    msgs = []
    for _ in range(int(3 * FPS)):
        r = sim.p.process(darken(SCENE, 0.2, sim.rng), sim.t)
        sim.t += 1 / FPS
        msgs.append(r.message)
    assert not any(msgs)


def test_reference_mode_is_default_after_calibration_and_gamma_when_requested():
    sim = calibrated()
    assert sim.p.mode == "reference" and sim.p.reference_image is not None
    gsim = calibrated(settings=replace(S, enhance_mode="gamma"))
    assert gsim.p.mode == "gamma"
    r = gsim.run(0.3, 3.0)
    assert r.gain > 1.3 and r.gamma < 1.0  # legacy path reports gamma


def test_stored_reference_without_image_falls_back_to_gamma_mode():
    ref = calibrated().p.reference
    sim = Sim(reference=ref)  # no image supplied
    assert sim.p.mode == "gamma"
    assert sim.run(0.3, 4.0).gain > 1.3


def test_stored_reference_with_image_uses_relighting():
    src = calibrated()
    p = Pipeline(S, src.p.reference, src.p.reference_image)
    assert p.mode == "reference"


def test_recalibrate_drops_relighter():
    sim = calibrated()
    sim.p.recalibrate()
    assert sim.p.mode == "gamma" and sim.p.reference_image is None
    sim.run(0.5, 1.5)
    assert sim.p.mode == "reference"


def test_relit_output_matches_reference_scene_colours():
    sim = calibrated()
    r = sim.run(0.15, 4.0)
    assert r.brightness_error_pct < 10.0
    ref = sim.p.reference_image.astype(float).reshape(-1, 3).mean(axis=0)
    out = r.frame.astype(float).reshape(-1, 3).mean(axis=0)
    assert np.abs(ref - out).max() < 10  # per-channel means within 10 levels
