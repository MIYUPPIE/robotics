import csv

import numpy as np
from src.light_detector import LightState, Transition
from src.logger import EVENT_COLUMNS, HISTORY_COLUMNS, HistoryLogger
from src.pipeline import FrameResult, Phase


def result(**kw):
    base = dict(
        frame=np.zeros((2, 2, 3), np.uint8),
        phase=Phase.RUNNING,
        state=LightState.NORMAL,
        level=0.5,
        level_filtered=0.5,
        ref_level=0.5,
        ratio=1.0,
        gain=1.0,
        gamma=1.0,
        clahe_clip=0.0,
        dropped_fraction=0.0,
        output_level=0.5,
        brightness_error_pct=0.0,
        latency_ms=3.0,
    )
    base.update(kw)
    return FrameResult(**base)


def rows(p):
    with p.open() as fh:
        return list(csv.DictReader(fh))


def test_header_rows_and_rate_limit(tmp_path):
    h, e = tmp_path / "h.csv", tmp_path / "e.csv"
    with HistoryLogger(h, e, interval_s=0.2) as log:
        for i in range(30):  # 1 s at 30 fps
            log.log(i / 30, result(), 30.0)
    data = rows(h)
    assert list(data[0].keys()) == HISTORY_COLUMNS
    assert 4 <= len(data) <= 7
    assert data[0]["state"] == "NORMAL"


def test_nan_written_as_empty_cell(tmp_path):
    h, e = tmp_path / "h.csv", tmp_path / "e.csv"
    with HistoryLogger(h, e, 0.0) as log:
        log.log(0.0, result(output_level=float("nan"), brightness_error_pct=float("nan")), 30.0)
    r = rows(h)[0]
    assert r["output_level"] == "" and r["brightness_error_pct"] == ""


def test_transition_forces_row_and_event(tmp_path):
    h, e = tmp_path / "h.csv", tmp_path / "e.csv"
    tr = Transition(LightState.NORMAL, LightState.DARK, 1.0, 0.3)
    with HistoryLogger(h, e, interval_s=100.0) as log:
        log.log(0.0, result(), 30.0)
        log.log(0.01, result(state=LightState.DARK, transition=tr), 30.0)
    assert len(rows(h)) == 2  # second row bypassed the rate limit
    ev = rows(e)
    assert list(ev[0].keys()) == EVENT_COLUMNS
    assert ev[0]["event"] == "state_change" and "NORMAL->DARK" in ev[0]["detail"]


def test_calibration_and_message_events(tmp_path):
    h, e = tmp_path / "h.csv", tmp_path / "e.csv"
    with HistoryLogger(h, e, 0.0) as log:
        log.log(0.0, result(calibrated_now=True, message="calibration complete"), 30.0)
        log.log(0.1, result(message="calibration failed: x"), 30.0)
        log.event(0.2, "start", "x")
    kinds = [r["event"] for r in rows(e)]
    assert kinds == ["calibrated", "message", "start"]


def test_append_does_not_duplicate_header(tmp_path):
    h, e = tmp_path / "h.csv", tmp_path / "e.csv"
    for _ in range(2):
        with HistoryLogger(h, e, 0.0) as log:
            log.log(0.0, result(), 30.0)
    assert len(rows(h)) == 2
    assert h.read_text().count("session,timestamp") == 1


def test_close_is_idempotent(tmp_path):
    log = HistoryLogger(tmp_path / "h.csv", tmp_path / "e.csv", 0.0)
    log.close()
    log.close()
