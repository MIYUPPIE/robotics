import json

import pytest
from src.config import Settings, SettingsError, load_settings, settings_from_dict


def test_defaults_are_valid():
    Settings().validate()


def test_shipped_settings_json_loads():
    s = load_settings()
    assert s.dark_enter == 0.72 and s.recovery_threshold == 0.84


def test_override_subset(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"alpha": 0.8, "lock_exposure": False, "width": 1280}))
    s = load_settings(p)
    assert s.alpha == 0.8 and s.lock_exposure is False and s.width == 1280
    assert s.max_gain == Settings().max_gain


def test_unknown_key_rejected():
    with pytest.raises(SettingsError, match="unknown"):
        settings_from_dict({"dark_enterr": 0.5})


@pytest.mark.parametrize(
    "bad",
    [
        {"width": "640"},
        {"width": 6.4},
        {"alpha": True},
        {"lock_exposure": 1},
        {"log_file": 3},
        {"dark_enter": 0.9},  # >= dim_enter
        {"recovery_threshold": 0.5},  # <= dark_enter
        {"dim_exit": 1.2},
        {"max_gain": 1.0},
        {"gamma_floor": 0.0},
        {"clahe_clip_min": 3.0},
        {"grid_rows": 0},
    ],
)
def test_invalid_values_rejected(bad):
    with pytest.raises(SettingsError):
        settings_from_dict(bad)


def test_explicit_missing_file_is_error(tmp_path):
    with pytest.raises(SettingsError, match="not found"):
        load_settings(tmp_path / "nope.json")


def test_bad_json_and_non_object(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{oops")
    with pytest.raises(SettingsError, match="valid JSON"):
        load_settings(p)
    p.write_text("[1]")
    with pytest.raises(SettingsError, match="object"):
        load_settings(p)


def test_resolve_paths(tmp_path):
    s = Settings()
    assert s.resolve(str(tmp_path / "x")) == tmp_path / "x"
    assert s.resolve("logs/a.csv").name == "a.csv"
