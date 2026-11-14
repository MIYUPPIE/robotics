"""06 Compensation controller.

    R_t = L_ref / max(L_t, eps)
    G_t = clip(R_t ** alpha, 1, G_max)          (forced to 1 while NORMAL)
    G_smooth = EMA of G_t (time-aware: tau_up when rising, tau_down when falling)

Release cap: G_smooth never exceeds the *ungated* target. When light physically
returns, R falls at once, so gain falls at once and the output never flashes
over-bright. Falling gain from state gating alone (leaving DIMMED) stays smooth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import Settings
from .light_detector import LightState


@dataclass(frozen=True)
class ControlOutput:
    ratio: float  # level / L_ref
    compensation_ratio: float  # R_t
    target_gain: float  # gated, clipped
    gain: float  # smoothed, release-capped


def _alpha(dt: float, tau: float) -> float:
    return 1.0 if tau <= 0 else 1.0 - math.exp(-max(dt, 0.0) / tau)


class CompensationController:
    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self.reset()

    def reset(self) -> None:
        self._gain = 1.0
        self._t: float | None = None

    @property
    def gain(self) -> float:
        return self._gain

    def update(self, level: float, ref_level: float, state: LightState, t: float) -> ControlOutput:
        s = self._s
        r = ref_level / max(level, s.epsilon)
        ungated = min(max(r**s.alpha, 1.0), s.max_gain)
        target = 1.0 if state is LightState.NORMAL else ungated
        dt = 0.0 if self._t is None else t - self._t
        self._t = t
        tau = s.gain_tau_up_s if target > self._gain else s.gain_tau_down_s
        self._gain += (target - self._gain) * _alpha(dt, tau)
        self._gain = min(self._gain, ungated)
        return ControlOutput(
            ratio=level / ref_level if ref_level > 0 else 0.0,
            compensation_ratio=r,
            target_gain=target,
            gain=self._gain,
        )
