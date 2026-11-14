import json

import numpy as np
import pytest
from src.calibration import (
    CalibrationError,
    Calibrator,
    ReferenceFileError,
    drift_reference,
    load_reference,
    save_reference,
)
from src.config import Settings
from src.illumination import IlluminationStats

S = Settings(calibration_seconds=1.0, min_calibration_frames=10)


def stats(level, cells=None):
    cells = np.full(9, level, np.float32) if cells is None else cells
    return IlluminationStats(level, level, level, level, level, cells)


def run_calibration(levels, s=S, fps=30):
    c = Calibrator(s)
    for i, lv in enumerate(levels):
        c.add(stats(lv), i / fps)
    return c


def test_not_ready_before_window_or_frames():
    c = run_calibration([0.5] * 5)
    assert not c.ready
    c = Calibrator(S)
    c.add(stats(0.5), 0.0)
    c.add(stats(0.5), 5.0)  # window elapsed but too few frames
    assert not c.ready


def test_reference_is_median_not_mean():
    levels = [0.5] * 40 + [0.05] * 4  # person/shadow outliers
    c = run_calibration(levels)
    assert c.ready
    assert c.finalize().level == pytest.approx(0.5)


def test_progress_and_spread():
    c = run_calibration([0.4] * 15)
    assert 0 < c.progress() <= 1
    assert c.finalize().spread == pytest.approx(0.0)
    noisy = run_calibration(list(np.linspace(0.3, 0.7, 40)))
    assert noisy.finalize().spread > 0.3


def test_too_dark_to_calibrate():
    with pytest.raises(CalibrationError, match="too dark"):
        run_calibration([0.02] * 40).finalize()


def test_too_few_frames():
    with pytest.raises(CalibrationError, match="frames"):
        run_calibration([0.5] * 3).finalize()


def test_save_load_roundtrip_and_exposure(tmp_path):
    ref = run_calibration([0.5] * 40).finalize({"locked": True, "exposure": 157.0})
    p = tmp_path / "sub" / "ref.json"
    save_reference(ref, p)
    loaded = load_reference(p, S)
    assert loaded == ref
    assert loaded.exposure["exposure"] == 157.0
    assert not list(p.parent.glob("*.tmp"))


def test_missing_file_returns_none(tmp_path):
    assert load_reference(tmp_path / "none.json", S) is None


@pytest.mark.parametrize(
    "content", ["{bad", "[]", json.dumps({"version": 99}), json.dumps({"version": 1})]
)
def test_corrupt_or_wrong_version(tmp_path, content):
    p = tmp_path / "r.json"
    p.write_text(content)
    with pytest.raises(ReferenceFileError):
        load_reference(p, S)


def test_inconsistent_values(tmp_path):
    ref = run_calibration([0.5] * 40).finalize()
    p = tmp_path / "r.json"
    save_reference(ref, p)
    raw = json.loads(p.read_text())
    raw["cells"] = [0.5]
    p.write_text(json.dumps(raw))
    with pytest.raises(ReferenceFileError, match="inconsistent"):
        load_reference(p, S)


@pytest.mark.parametrize("override", [{"camera_index": 2}, {"width": 1280}, {"grid_rows": 4}])
def test_geometry_mismatch_rejected(tmp_path, override):
    ref = run_calibration([0.5] * 40).finalize()
    p = tmp_path / "r.json"
    save_reference(ref, p)
    from dataclasses import replace

    with pytest.raises(ReferenceFileError, match="different"):
        load_reference(p, replace(S, **override))


def test_drift_moves_toward_stats():
    ref = run_calibration([0.5] * 40).finalize()
    d = drift_reference(ref, stats(0.52), rate_per_s=0.5, dt=0.1)
    assert 0.5 < d.level < 0.52
    assert drift_reference(ref, stats(0.52), 0.0, 0.1) is ref
    assert drift_reference(ref, stats(0.52), 0.5, 0.0) is ref


def test_old_reference_rejected(tmp_path):
    from dataclasses import replace
    from datetime import UTC, datetime, timedelta

    ref = run_calibration([0.5] * 40).finalize()
    old = (datetime.now(UTC) - timedelta(hours=30)).isoformat(timespec="seconds")
    p = tmp_path / "r.json"
    save_reference(replace(ref, created_at=old), p)
    with pytest.raises(ReferenceFileError, match="old"):
        load_reference(p, S)
    fresh = replace(S, max_reference_age_h=48.0)
    assert load_reference(p, fresh) is not None


def test_garbage_timestamp_rejected(tmp_path):
    from dataclasses import replace

    ref = run_calibration([0.5] * 40).finalize()
    p = tmp_path / "r.json"
    save_reference(replace(ref, created_at="yesterday"), p)
    with pytest.raises(ReferenceFileError, match="created_at"):
        load_reference(p, S)


def test_calibrator_image_is_clean_mean_of_frames():
    import cv2
    from evals.synth import darken, make_scene
    from src.calibration import Calibrator

    scene = make_scene(h=48, w=64)
    rng = np.random.default_rng(0)
    c = Calibrator(S)
    for i in range(30):
        f = darken(scene, 1.0, rng)
        c.add(stats(0.5), i / 30, f)
    img = c.image()
    assert img.shape == scene.shape and img.dtype == np.uint8
    noisy = darken(scene, 1.0, np.random.default_rng(9))
    err_avg = np.abs(img.astype(int) - scene.astype(int)).mean()
    err_one = np.abs(noisy.astype(int) - scene.astype(int)).mean()
    assert err_avg < 0.6 * err_one  # averaging removed most of the sensor noise
    assert cv2 is not None


def test_calibrator_without_frames_has_no_image():
    assert run_calibration([0.5] * 5).image() is None


def test_reference_image_roundtrip_and_size_check(tmp_path):
    from dataclasses import replace

    from src.calibration import (
        load_reference_image,
        reference_image_path,
        save_reference_image,
    )

    img = np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    ref_json = tmp_path / "reference.json"
    assert load_reference_image(ref_json, S) is None
    save_reference_image(img, ref_json)
    assert reference_image_path(ref_json).exists()
    assert not list(tmp_path.glob("*.tmp*"))
    assert np.array_equal(load_reference_image(ref_json, replace(S, width=64, height=48)), img)
    with pytest.raises(ReferenceFileError, match="camera is"):
        load_reference_image(ref_json, S)  # S is 640x480


def test_corrupt_reference_image(tmp_path):
    from src.calibration import reference_image_path

    p = tmp_path / "reference.json"
    reference_image_path(p).write_bytes(b"not a png")
    from src.calibration import load_reference_image

    with pytest.raises(ReferenceFileError, match="decode"):
        load_reference_image(p, S)
