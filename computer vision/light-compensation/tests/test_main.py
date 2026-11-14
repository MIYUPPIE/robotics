import json

import main as app


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
