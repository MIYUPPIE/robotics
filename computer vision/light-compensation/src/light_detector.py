"""05 Adaptive light-state detector: NORMAL / DIMMED / DARK / RECOVERY.

Pure state machine (no camera, no OpenCV). Input per tick: ratio = level / L_ref,
whether the drop is spatially uniform, and a timestamp.

Hysteresis: the exit thresholds sit above the entry thresholds.
Debounce:   a transition fires only after its condition has held `confirm_seconds`.
Occlusion:  entering a darker state requires `uniform` (most grid cells darkened),
            so a person walking through the frame is not a light switch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .config import Settings


class LightState(StrEnum):
    NORMAL = "NORMAL"
    DIMMED = "DIMMED"
    DARK = "DARK"
    RECOVERY = "RECOVERY"


@dataclass(frozen=True)
class Transition:
    old: LightState
    new: LightState
    t: float
    ratio: float


class LightDetector:
    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self.reset()

    def reset(self) -> None:
        self.state = LightState.NORMAL
        self._state_since: float | None = None
        self._pending: LightState | None = None
        self._pending_since: float = 0.0

    def _candidate(self, ratio: float, uniform: bool, in_state_s: float) -> LightState:
        s, st = self._s, self.state
        if st is LightState.NORMAL:
            if uniform and ratio < s.dark_enter:
                return LightState.DARK
            if uniform and ratio < s.dim_enter:
                return LightState.DIMMED
        elif st is LightState.DIMMED:
            if uniform and ratio < s.dark_enter:
                return LightState.DARK
            if ratio >= s.dim_exit:
                return LightState.NORMAL
        elif st is LightState.DARK:
            if ratio >= s.recovery_threshold:
                return LightState.RECOVERY
        else:  # RECOVERY
            if uniform and ratio < s.dark_enter:
                return LightState.DARK
            if ratio >= s.dim_exit and in_state_s >= s.min_recovery_s:
                return LightState.NORMAL
            if ratio < s.dim_exit and in_state_s >= s.recovery_timeout_s:
                return LightState.DIMMED
        return st

    def update(self, ratio: float, uniform: bool, t: float) -> Transition | None:
        if self._state_since is None:
            self._state_since = t
        cand = self._candidate(ratio, uniform, t - self._state_since)
        if cand is self.state:
            self._pending = None
            return None
        if cand is not self._pending:
            self._pending, self._pending_since = cand, t
        if t - self._pending_since >= self._s.confirm_seconds:
            tr = Transition(old=self.state, new=cand, t=t, ratio=ratio)
            self.state, self._state_since, self._pending = cand, t, None
            return tr
        return None
