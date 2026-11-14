#!/usr/bin/env python
"""Closed-loop quality eval for the light-compensation pipeline (synthetic, deterministic).

The scene is darkened in linear light with read+shot noise (evals/synth.py), as a
webcam with locked exposure would see it when room lights drop. This measures the
controller and enhancer end to end. It is NOT a measurement of a real webcam: tune
thresholds on real recordings before trusting absolute numbers.

Pass bars (from document.md section 6 + the project brief):
  steady brightness error      < 15 %   for k in MODERATE_K (moderately dark)
  detection latency            < 1.5 s  for every dimming step
  output flicker (level std)   < 0.012  during steady darkness
  recovery to NORMAL           < 4 s    and no over-bright flash (error < 15 %)
  false triggers               == 0     person sweeps + small screen-brightness changes
  extreme dark                 graceful: finite values, gain <= max_gain
  median frame latency         < 30 ms  at 640x480

Usage: python evals/compensation_eval.py        (from light-compensation/)
       make eval                                (via the pytest wrapper)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from src.config import Settings  # noqa: E402
from src.light_detector import LightState  # noqa: E402
from src.pipeline import FrameResult, Pipeline  # noqa: E402

from evals.synth import add_occluder, darken, make_scene  # noqa: E402

FPS = 30
SETTINGS = replace(Settings(), calibration_seconds=3.0, lock_exposure=False)
MODERATE_K = (0.7, 0.5, 0.35, 0.2)
EXTREME_K = (0.1, 0.05, 0.02)
SEEDS = (0, 1, 2)

ERROR_BAR_PCT = 15.0
DETECT_BAR_S = 1.5
FLICKER_BAR = 0.012
RECOVERY_BAR_S = 4.0
LATENCY_BAR_MS = 30.0


@dataclass
class Check:
    name: str
    value: str
    bar: str
    passed: bool


class Rig:
    """A pipeline calibrated on the normal scene, driven frame by frame."""

    def __init__(
        self, seed: int, h: int = 240, w: int = 320, settings: Settings | None = None
    ) -> None:
        settings = settings or SETTINGS
        self.rng = np.random.default_rng(seed)
        self.scene = make_scene(seed=7, h=h, w=w)
        self.p = Pipeline(settings)
        self.t = 0.0
        self.log: list[tuple[float, FrameResult]] = []
        self.step(1.0, settings.calibration_seconds + 0.5)
        assert self.p.reference is not None
        self.t_mark = self.t
        self.log.clear()

    def step(self, k: float, seconds: float, occluder_x=None, occluder_value=25) -> None:
        for _ in range(round(seconds * FPS)):
            f = darken(self.scene, k, self.rng)
            if occluder_x is not None:
                f = add_occluder(f, occluder_x(self.t), 0.4, occluder_value)
            r = self.p.process(f, self.t)
            self.log.append((self.t, r))
            self.t += 1 / FPS

    def transitions(self) -> list[tuple[float, str]]:
        return [(t, f"{r.transition.old}->{r.transition.new}") for t, r in self.log if r.transition]


def dimming_step(k: float, seed: int) -> dict[str, float]:
    rig = Rig(seed)
    t0 = rig.t
    rig.step(k, 6.0)
    trs = rig.transitions()
    detect = trs[0][0] - t0 if trs else float("inf")
    tail = [r for t, r in rig.log if t >= t0 + 3.0]
    lv = np.array([r.output_level for r in tail])
    err = np.mean([r.brightness_error_pct for r in tail])
    return {
        "detect_s": detect,
        "error_pct": float(err),
        "flicker": float(lv.std()),
        "gain": tail[-1].gain,
    }


def recovery(seed: int) -> dict[str, float]:
    rig = Rig(seed)
    rig.step(0.2, 5.0)
    n0 = len(rig.log)
    t0 = rig.t
    rig.step(1.0, 6.0)
    seg = [(t, r) for t, r in rig.log[n0:]]
    to_normal = next((t - t0 for t, r in seg if r.state is LightState.NORMAL), float("inf"))
    worst = max(r.brightness_error_pct for _, r in seg)
    return {"to_normal_s": to_normal, "max_error_pct": float(worst)}


def false_triggers(seed: int) -> dict[str, int]:
    sweep = lambda t0: lambda t: int(-130 + (t - t0) * 120)  # noqa: E731
    rig = Rig(seed)
    rig.step(1.0, 1.0)
    rig.step(1.0, 4.0, occluder_x=sweep(rig.t))
    person_normal = len(rig.transitions())

    rig = Rig(seed)
    rig.step(0.3, 5.0)
    base = len(rig.transitions())
    rig.step(0.3, 4.0, occluder_x=sweep(rig.t), occluder_value=8)
    person_dark = len(rig.transitions()) - base

    rig = Rig(seed)
    rig.step(0.95, 3.0)
    rig.step(1.04, 3.0)
    rig.step(0.93, 3.0)
    screen = len(rig.transitions())
    return {"person_normal": person_normal, "person_dark": person_dark, "screen": screen}


def extreme(k: float, seed: int) -> dict[str, float]:
    rig = Rig(seed)
    rig.step(k, 5.0)
    r = rig.log[-1][1]
    ok = bool(
        np.isfinite(r.gain)
        and np.isfinite(r.output_level)
        and r.gain <= SETTINGS.max_gain + 1e-6
        and r.frame.dtype == np.uint8
    )
    return {"ok": float(ok), "error_pct": r.brightness_error_pct, "gain": r.gain}


def frame_latency_ms() -> float:
    rig = Rig(0, h=480, w=640)
    rig.log.clear()
    rig.step(0.25, 3.0)
    return float(np.median([r.latency_ms for _, r in rig.log[30:]]))


def run() -> list[Check]:
    checks: list[Check] = []

    def add(name: str, value: str, bar: str, passed: bool) -> None:
        checks.append(Check(name, value, bar, bool(passed)))

    for k in MODERATE_K:
        rs = [dimming_step(k, s) for s in SEEDS]
        err = max(r["error_pct"] for r in rs)
        det = max(r["detect_s"] for r in rs)
        flk = max(r["flicker"] for r in rs)
        add(f"brightness error k={k}", f"{err:.1f}%", f"< {ERROR_BAR_PCT}%", err < ERROR_BAR_PCT)
        add(f"detection latency k={k}", f"{det:.2f}s", f"< {DETECT_BAR_S}s", det < DETECT_BAR_S)
        add(f"output flicker k={k}", f"{flk:.4f}", f"< {FLICKER_BAR}", flk < FLICKER_BAR)

    rec = [recovery(s) for s in SEEDS]
    t_norm = max(r["to_normal_s"] for r in rec)
    flash = max(r["max_error_pct"] for r in rec)
    add("recovery to NORMAL", f"{t_norm:.2f}s", f"< {RECOVERY_BAR_S}s", t_norm < RECOVERY_BAR_S)
    add(
        "no over-bright flash on recovery",
        f"{flash:.1f}%",
        f"< {ERROR_BAR_PCT}%",
        flash < ERROR_BAR_PCT,
    )

    ft = [false_triggers(s) for s in SEEDS]
    for key, label in (
        ("person_normal", "false triggers: person, normal light"),
        ("person_dark", "false triggers: person, dark room"),
        ("screen", "false triggers: screen brightness"),
    ):
        n = sum(f[key] for f in ft)
        add(label, str(n), "== 0", n == 0)

    for k in EXTREME_K:
        e = extreme(k, 0)
        add(
            f"extreme dark k={k} graceful",
            f"gain {e['gain']:.1f}, err {e['error_pct']:.0f}%",
            "finite, gain<=max",
            bool(e["ok"]),
        )

    lat = frame_latency_ms()
    add(
        "median frame latency 640x480",
        f"{lat:.1f}ms",
        f"< {LATENCY_BAR_MS}ms",
        lat < LATENCY_BAR_MS,
    )
    return checks


def main() -> int:
    cv2.setRNGSeed(0)
    checks = run()
    width = max(len(c.name) for c in checks)
    for c in checks:
        print(f"[{'PASS' if c.passed else 'FAIL'}] {c.name:<{width}}  {c.value:>18}  (bar {c.bar})")
    failed = [c for c in checks if not c.passed]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
