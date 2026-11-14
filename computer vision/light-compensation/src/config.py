"""Single source of truth for every tunable constant in light-compensation.

`Settings` holds defaults; `config/settings.json` overrides any subset. Unknown
keys and out-of-range values raise `SettingsError` so a typo never silently
falls back to a default. Nothing else in the package hardcodes a threshold.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.json"


class SettingsError(ValueError):
    """Raised for unknown keys, wrong types or inconsistent values."""


@dataclass(frozen=True)
class Settings:
    # --- 01 acquisition ---
    camera_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    lock_exposure: bool = True
    exposure_warmup_s: float = 2.0  # auto-exposure converges before we lock it

    # --- 02/03 preprocessing + estimation ---
    analysis_width: int = 160  # luma plane is downscaled to this width for statistics
    grid_rows: int = 3
    grid_cols: int = 3

    # --- 04 reference / calibration ---
    calibration_seconds: float = 5.0
    min_calibration_frames: int = 30
    min_reference_level: float = 0.08  # refuse to calibrate in a room this dark
    max_reference_spread: float = 0.12  # IQR/median of calibration levels; warn above
    drift_rate_per_s: float = 0.0  # 0 disables slow reference drift (default: off)
    drift_band: float = 0.05  # only drift while |ratio - 1| is inside this band
    reference_file: str = "data/reference.json"
    max_reference_age_h: float = 24.0  # older stored references are discarded
    startup_check_s: float = 3.0  # DIMMED/DARK this soon after start on a stored reference -> warn

    # --- 05 light-state detector (ratio = current level / reference level) ---
    dim_enter: float = 0.88
    dim_exit: float = 0.93
    dark_enter: float = 0.72
    recovery_threshold: float = 0.84
    confirm_seconds: float = 0.5  # a transition must hold this long before it fires
    min_recovery_s: float = 1.0  # RECOVERY lasts at least this long
    recovery_timeout_s: float = 10.0  # stuck in RECOVERY below dim_exit -> DIMMED
    cell_drop_ratio: float = 0.90  # a grid cell "dropped" if cell/ref_cell is below this
    uniform_cell_fraction: float = 0.60  # fraction of dropped cells = global light change
    level_fall_tau_s: float = 0.40  # falling level is smoothed; rising level is instant

    # --- 06 controller (G = clip(R^alpha, 1, Gmax), smoothed) ---
    alpha: float = 1.0
    max_gain: float = 16.0
    gain_tau_up_s: float = 0.40
    gain_tau_down_s: float = 0.30
    epsilon: float = 1e-3

    # --- 07 enhancer ---
    gamma_floor: float = 0.20
    target_ceiling: float = 0.92  # never ask gamma to push the median above this
    clahe_clip_min: float = 1.0
    clahe_clip_max: float = 2.5
    clahe_tile: int = 8
    trim_tolerance: float = 0.02  # post-CLAHE level within this of target: no trim
    trim_gamma_min: float = 0.33
    trim_gamma_max: float = 3.0
    temporal_max_weight: float = 0.8  # weight of previous frame in temporal denoise at max gain
    temporal_motion_threshold: float = 20.0  # L difference (0..255) treated as motion
    denoise_gain_min: float = 1.4  # denoise only when the gain is at least this
    denoise_sigma_color: float = 25.0
    denoise_sigma_space: float = 5.0
    chroma_blur_gain_min: float = 2.0
    measure_output: bool = True  # measure output brightness for the error metric
    measure_every: int = 3  # ...on every Nth frame (the last value is carried between)

    # --- 07b reference-guided relighting (default mode) ---
    enhance_mode: str = "reference"  # "reference" (needs stored image) or "gamma"
    field_cols: int = 16
    field_rows: int = 12
    field_outlier_thr: float = 0.35  # log units: cell vs neighbourhood median (~42 %)
    field_dev_max: float = 0.9  # log units: max spread of the field around its global ratio
    field_min_gain: float = 0.5
    field_max_gain: float = 100.0  # linear-light gain cap
    field_gate_lo: float = 0.03  # robust deviation below this: no correction
    field_gate_hi: float = 0.12  # ...at or above this: full correction
    field_tau_up_s: float = 0.30
    background_shrink_kappa: float = 2.0  # 0 disables; noise multiples treated as "unchanged"
    noise_sigma_init: float = 3.0  # starting sensor-noise estimate, 8-bit levels

    # --- 08 output / monitoring ---
    log_file: str = "logs/light_history.csv"
    events_file: str = "logs/events.csv"
    log_interval_s: float = 0.2
    graph_seconds: float = 30.0
    window_title: str = "Light Compensation"

    def validate(self) -> None:
        def req(cond: bool, msg: str) -> None:
            if not cond:
                raise SettingsError(msg)

        req(self.width > 0 and self.height > 0 and self.fps > 0, "width/height/fps must be > 0")
        req(self.analysis_width >= 16, "analysis_width must be >= 16")
        req(self.grid_rows >= 1 and self.grid_cols >= 1, "grid must be at least 1x1")
        req(self.calibration_seconds > 0, "calibration_seconds must be > 0")
        req(self.min_calibration_frames >= 1, "min_calibration_frames must be >= 1")
        req(0 < self.min_reference_level < 1, "min_reference_level must be in (0, 1)")
        req(self.drift_rate_per_s >= 0, "drift_rate_per_s must be >= 0")
        req(0 < self.drift_band < 0.5, "drift_band must be in (0, 0.5)")
        req(
            0 < self.dark_enter < self.dim_enter < self.dim_exit <= 1.0,
            "need 0 < dark_enter < dim_enter < dim_exit <= 1",
        )
        req(
            self.dark_enter < self.recovery_threshold <= self.dim_exit,
            "need dark_enter < recovery_threshold <= dim_exit",
        )
        req(self.confirm_seconds >= 0, "confirm_seconds must be >= 0")
        req(self.min_recovery_s >= 0, "min_recovery_s must be >= 0")
        req(self.recovery_timeout_s > self.min_recovery_s, "recovery_timeout_s > min_recovery_s")
        req(0 < self.cell_drop_ratio < 1, "cell_drop_ratio must be in (0, 1)")
        req(0 < self.uniform_cell_fraction <= 1, "uniform_cell_fraction must be in (0, 1]")
        req(self.level_fall_tau_s >= 0, "level_fall_tau_s must be >= 0")
        req(0 < self.alpha <= 2.0, "alpha must be in (0, 2]")
        req(self.max_gain > 1.0, "max_gain must be > 1")
        req(self.gain_tau_up_s >= 0 and self.gain_tau_down_s >= 0, "gain taus must be >= 0")
        req(self.epsilon > 0, "epsilon must be > 0")
        req(0 < self.gamma_floor <= 1.0, "gamma_floor must be in (0, 1]")
        req(0.5 < self.target_ceiling < 1.0, "target_ceiling must be in (0.5, 1)")
        req(1.0 <= self.clahe_clip_min <= self.clahe_clip_max, "need 1 <= clahe_clip_min <= max")
        req(self.max_reference_age_h > 0, "max_reference_age_h must be > 0")
        req(self.startup_check_s >= 0, "startup_check_s must be >= 0")
        req(0 <= self.temporal_max_weight < 1, "temporal_max_weight must be in [0, 1)")
        req(self.temporal_motion_threshold > 0, "temporal_motion_threshold must be > 0")
        req(self.enhance_mode in ("reference", "gamma"), "enhance_mode: reference or gamma")
        req(self.field_cols >= 2 and self.field_rows >= 2, "field grid must be >= 2x2")
        req(self.field_outlier_thr > 0 and self.field_dev_max > 0, "field thresholds must be > 0")
        req(0 < self.field_min_gain <= 1.0 < self.field_max_gain, "need min_gain<=1<max_gain")
        req(0 <= self.field_gate_lo < self.field_gate_hi, "need 0 <= gate_lo < gate_hi")
        req(self.background_shrink_kappa >= 0, "background_shrink_kappa must be >= 0")
        req(0 < self.noise_sigma_init <= 12, "noise_sigma_init must be in (0, 12]")
        req(self.field_tau_up_s >= 0, "field_tau_up_s must be >= 0")
        req(self.measure_every >= 1, "measure_every must be >= 1")
        req(self.clahe_tile >= 2, "clahe_tile must be >= 2")
        req(0 <= self.trim_tolerance < 0.5, "trim_tolerance must be in [0, 0.5)")
        req(
            0 < self.trim_gamma_min <= 1.0 <= self.trim_gamma_max, "need min <= 1 <= max trim gamma"
        )
        req(self.log_interval_s >= 0, "log_interval_s must be >= 0")
        req(self.graph_seconds > 0, "graph_seconds must be > 0")

    def resolve(self, relative: str) -> Path:
        """Resolve a settings path against the project root (absolute paths pass through)."""
        p = Path(relative)
        return p if p.is_absolute() else PROJECT_ROOT / p


def _coerce(name: str, default: Any, value: Any) -> Any:
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise SettingsError(f"{name} must be a boolean, got {value!r}")
        return value
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise SettingsError(f"{name} must be an integer, got {value!r}")
        return value
    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SettingsError(f"{name} must be a number, got {value!r}")
        return float(value)
    if isinstance(default, str):
        if not isinstance(value, str):
            raise SettingsError(f"{name} must be a string, got {value!r}")
        return value
    raise SettingsError(f"unsupported setting type for {name}")  # pragma: no cover


def settings_from_dict(overrides: dict[str, Any]) -> Settings:
    defaults = Settings()
    known = {f.name for f in fields(Settings)}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise SettingsError(f"unknown setting(s): {', '.join(unknown)}")
    coerced = {k: _coerce(k, getattr(defaults, k), v) for k, v in overrides.items()}
    settings = Settings(**coerced)
    settings.validate()
    return settings


def load_settings(path: Path | str | None = None) -> Settings:
    """Load settings from JSON. A missing *default* file yields defaults; a missing
    explicitly-requested file is an error."""
    explicit = path is not None
    p = DEFAULT_SETTINGS_PATH if path is None else Path(path)
    if not p.exists():
        if explicit:
            raise SettingsError(f"settings file not found: {p}")
        return settings_from_dict({})
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise SettingsError(f"{p} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise SettingsError(f"{p} must contain a JSON object")
    return settings_from_dict(raw)
