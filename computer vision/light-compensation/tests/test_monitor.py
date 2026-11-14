import numpy as np
from src.light_detector import LightState
from src.monitor import GRAPH_HEIGHT, History, compose, draw_graph
from src.pipeline import FrameResult, Phase


def res(phase=Phase.RUNNING, **kw):
    base = dict(
        frame=np.zeros((60, 80, 3), np.uint8),
        phase=phase,
        state=LightState.DARK,
        level=0.2,
        level_filtered=0.2,
        ref_level=0.5,
        ratio=0.4,
        gain=2.5,
        gamma=0.5,
        clahe_clip=2.0,
        dropped_fraction=1.0,
        output_level=0.5,
        brightness_error_pct=1.0,
        latency_ms=4.0,
        calibration_progress=0.5,
        message="hello",
    )
    base.update(kw)
    return FrameResult(**base)


def test_history_window_trims_old_points():
    h = History(2.0)
    for i in range(100):
        h.add(i * 0.1, 1.0, 1.0)
    assert len(h) <= 22 and h.points()[0][0] >= 9.9 - 2.0 - 1e-9


def test_graph_shape_with_and_without_data():
    h = History(10)
    assert draw_graph(h, 200, 10, 8.0).shape == (GRAPH_HEIGHT, 200, 3)
    for i in range(50):
        h.add(i * 0.1, 0.3 + 0.01 * i, 1 + 0.1 * i)
    g = draw_graph(h, 200, 10, 8.0)
    assert g.shape == (GRAPH_HEIGHT, 200, 3) and g.std() > 0


def test_graph_survives_out_of_range_values():
    h = History(10)
    for i in range(5):
        h.add(i, -3.0, 99.0)
    draw_graph(h, 100, 10, 8.0)


def test_compose_layout_running_and_calibrating():
    orig = np.full((60, 80, 3), 90, np.uint8)
    h = History(10)
    out = compose(orig, res(), h, 29.7, 10, 8.0)
    assert out.shape == (60 + GRAPH_HEIGHT, 160, 3)
    out = compose(orig, res(phase=Phase.CALIBRATING), h, 30.0, 10, 8.0)
    assert out.shape == (60 + GRAPH_HEIGHT, 160, 3)


def test_compose_does_not_mutate_inputs():
    orig = np.full((60, 80, 3), 90, np.uint8)
    r = res()
    compose(orig, r, History(10), 30.0, 10, 8.0)
    assert (orig == 90).all() and (r.frame == 0).all()


def test_compose_nan_error_ok():
    compose(
        np.zeros((60, 80, 3), np.uint8),
        res(brightness_error_pct=float("nan")),
        History(10),
        30,
        10,
        8.0,
    )
