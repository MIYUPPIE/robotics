import json

import main as app
import numpy as np


def test_parse_args_defaults():
    a = app.parse_args([])
    assert not a.recalibrate and not a.no_display and a.max_seconds is None


def test_bad_settings_file_exits_2(tmp_path, capsys):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"nonsense": 1}))
    assert app.main(["--settings", str(p), "--no-display"]) == 2
    assert "unknown" in capsys.readouterr().err


def test_missing_settings_file_exits_2(tmp_path):
    assert app.main(["--settings", str(tmp_path / "x.json")]) == 2


def test_unopenable_camera_exits_3(capsys):
    assert app.main(["--camera", "99", "--no-display"]) == 3
    assert "camera error" in capsys.readouterr().err


def test_reference_without_image_is_rejected_in_reference_mode(tmp_path, capsys, monkeypatch):
    """A stored reference.json with no reference.png must force recalibration, not crash."""
    import main as m
    from src.calibration import Calibrator, save_reference
    from src.config import Settings
    from src.illumination import IlluminationStats

    s = Settings(calibration_seconds=0.1, min_calibration_frames=1, camera_index=99)
    cal = Calibrator(s)
    cells = np.full(9, 0.5, np.float32)
    cal.add(IlluminationStats(0.5, 0.5, 0.5, 0.5, 0.5, cells), 0.0)
    cal.add(IlluminationStats(0.5, 0.5, 0.5, 0.5, 0.5, cells), 0.2)
    ref = cal.finalize()
    ref_path = tmp_path / "reference.json"
    save_reference(ref, ref_path)
    cfg = tmp_path / "s.json"
    cfg.write_text(json.dumps({"reference_file": str(ref_path), "camera_index": 99}))
    assert m.main(["--settings", str(cfg), "--no-display"]) == 3  # camera 99 fails afterwards
    assert "no stored reference image" in capsys.readouterr().err
