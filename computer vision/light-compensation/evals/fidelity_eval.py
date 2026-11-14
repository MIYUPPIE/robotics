#!/usr/bin/env python
"""Fidelity eval: how close is the compensated frame to the scene BEFORE the lights changed?

Truth is the noise-free scene at normal light. Metrics on the compensated frame:
  SSIM (grayscale, scikit-image)   structure/detail, higher is better, 1 = identical
  dE   (mean CIELAB distance)      colour + brightness error, lower is better, ~2.3 = barely visible
Compares the gamma+CLAHE baseline against the reference-guided relighter on:
  uniform dimming, one side of the room dark (uneven), a warm lamp swapped for a cool one (colour),
  and a person walking through while dark.

Usage: python evals/fidelity_eval.py
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
from skimage.metrics import structural_similarity  # noqa: E402
from src.config import Settings  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402

from evals.synth import add_occluder, darken, make_scene  # noqa: E402

FPS = 30
BASE = replace(Settings(), calibration_seconds=3.0, lock_exposure=False)
H, W = 240, 320


def _lab(img_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_u8.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def delta_e(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((_lab(a) - _lab(b)) ** 2).sum(axis=2)).mean())


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    ga, gb = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (a, b))
    return float(structural_similarity(ga, gb, data_range=255))


@dataclass
class Scenario:
    name: str
    field: np.ndarray  # (H, W, 3) or scalar illumination scale in linear light, BGR order
    occluder: bool = False


def scenarios() -> list[Scenario]:
    xs = np.linspace(0, 1, W, dtype=np.float32)[None, :, None]
    ones = np.ones((H, W, 3), np.float32)
    out = [Scenario(f"uniform k={k}", ones * k) for k in (0.5, 0.2, 0.1, 0.05)]
    out.append(Scenario("uneven: left side dark (0.08..0.6)", ones * (0.08 + 0.52 * xs)))
    out.append(
        Scenario(
            "colour: warm lamp -> cool light", ones * 0.45 * np.array([1.25, 1.0, 0.7], np.float32)
        )
    )
    out.append(Scenario("dark + person walking (k=0.1)", ones * 0.1, occluder=True))
    return out


def run_scenario(sc: Scenario, mode: str, seed: int = 0) -> tuple[float, float]:
    s = replace(BASE, enhance_mode=mode)
    rng = np.random.default_rng(seed)
    scene = make_scene(seed=7, h=H, w=W)
    p = Pipeline(s)
    t = 0.0
    for _ in range(round((s.calibration_seconds + 0.5) * FPS)):
        p.process(darken(scene, 1.0, rng), t)
        t += 1 / FPS
    assert p.reference is not None
    res = None
    frames = []
    for i in range(round(6.0 * FPS)):
        truth = scene
        if sc.occluder:  # a person (value 60 at normal light) is physically in the room
            truth = add_occluder(scene, int(-130 + i * 2), 0.4, 60)
        f = darken(truth, sc.field, rng)
        res = p.process(f, t)
        t += 1 / FPS
        if i >= round(4.5 * FPS):
            frames.append((res.frame, truth))
    assert res is not None
    de = np.mean([delta_e(o, tr) for o, tr in frames])
    ss = np.mean([ssim(o, tr) for o, tr in frames])
    return float(de), float(ss)


DE_BAR = 3.0  # CIELAB: ~2.3 is the just-noticeable difference
SSIM_BAR = 0.80
UNEVEN_COLOUR_DE_BAR = 3.5


def run() -> list[tuple[str, str, str, bool]]:
    """(name, value, bar, passed) rows."""
    rows: list[tuple[str, str, str, bool]] = []
    for sc in scenarios():
        g = run_scenario(sc, "gamma")
        r = run_scenario(sc, "reference")
        if sc.occluder:
            rows.append(
                (
                    f"{sc.name}: no worse than gamma",
                    f"dE {r[0]:.1f} vs {g[0]:.1f}",
                    "<= gamma",
                    r[0] <= g[0],
                )
            )
            continue
        bar = UNEVEN_COLOUR_DE_BAR if sc.name.startswith(("uneven", "colour")) else DE_BAR
        rows.append((f"{sc.name}: dE", f"{r[0]:.2f}", f"< {bar}", r[0] < bar))
        rows.append(
            (
                f"{sc.name}: beats gamma baseline",
                f"{r[0]:.2f} vs {g[0]:.2f}",
                "< 0.5 x gamma",
                r[0] < 0.5 * g[0],
            )
        )
        if sc.name.startswith("uniform"):
            rows.append((f"{sc.name}: SSIM", f"{r[1]:.3f}", f">= {SSIM_BAR}", r[1] >= SSIM_BAR))
    return rows


def main() -> int:
    scene = make_scene(seed=7, h=H, w=W)
    n = darken(scene, 1.0, np.random.default_rng(1))
    floor = f"dE {delta_e(n, scene):.2f}  SSIM {ssim(n, scene):.3f}"
    print(f"normal-light frame vs clean truth (noise floor): {floor}\n")
    print(f"{'scenario':<40} {'gamma dE':>9} {'ref dE':>8} {'gamma SSIM':>11} {'ref SSIM':>9}")
    for sc in scenarios():
        g = run_scenario(sc, "gamma")
        r = run_scenario(sc, "reference")
        print(f"{sc.name:<40} {g[0]:>9.2f} {r[0]:>8.2f} {g[1]:>11.3f} {r[1]:>9.3f}")
    print()
    rows = run()
    for name, value, bar, ok in rows:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<62} {value:>16}  (bar {bar})")
    failed = [r for r in rows if not r[3]]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
