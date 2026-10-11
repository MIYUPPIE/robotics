"""Gesture dwell-selection accuracy eval.

Real mode: if evals/data/sessions/*.json exist (recorded cursor-sample
streams from a real HandLandmarker run, each labeled with the region the
voter actually intended to select, or null for an intentional pass-through --
see evals/data/CONSENT.md), replays them through the real DwellSelector and
reports correct-intent selection rate + false-trigger rate against a pass bar.

Synthetic mode (no recorded sessions): runs the same accuracy computation
over procedurally generated cursor paths (intentional dwells + fast
pass-throughs with jitter) -- a broader randomized self-test of the
DwellSelector pipeline than the hand-picked unit tests in
hand_ui/tests/test_dwell_fsm.py, but still NOT a measurement of real
camera/MediaPipe tracking accuracy.

Usage: python -m hand_ui.evals.gesture_dwell_eval
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from contracts.events import CursorSample, DwellComplete, Region
from hand_ui.dwell_fsm import DEFAULT_DWELL_SECONDS, DwellSelector

DATA_DIR = Path(__file__).parent / "data"
SESSIONS_DIR = DATA_DIR / "sessions"

CORRECT_INTENT_PASS_BAR = 0.95
FALSE_TRIGGER_PASS_BAR = 0.02

Session = tuple[list[Region], list[CursorSample], str | None]


def _replay(regions: list[Region], samples: list[CursorSample], dwell_seconds: float) -> str | None:
    """Returns the region_id of the first DwellComplete fired, or None."""
    selector = DwellSelector(regions, dwell_seconds=dwell_seconds)
    for sample in samples:
        for event in selector.update(sample):
            if isinstance(event, DwellComplete):
                return event.region_id
    return None


def _score_sessions(
    sessions: list[Session], dwell_seconds: float = DEFAULT_DWELL_SECONDS
) -> tuple[float, float, int]:
    intentional = [s for s in sessions if s[2] is not None]
    pass_through = [s for s in sessions if s[2] is None]

    correct = sum(
        1
        for regions, samples, intended in intentional
        if _replay(regions, samples, dwell_seconds) == intended
    )
    false_triggers = sum(
        1
        for regions, samples, _ in pass_through
        if _replay(regions, samples, dwell_seconds) is not None
    )

    correct_rate = correct / len(intentional) if intentional else 1.0
    false_trigger_rate = false_triggers / len(pass_through) if pass_through else 0.0
    return correct_rate, false_trigger_rate, len(sessions)


def _load_real_sessions() -> list[Session]:
    sessions: list[Session] = []
    for path in sorted(SESSIONS_DIR.glob("*.json")):
        payload = json.loads(path.read_text())
        regions = [Region(**r) for r in payload["regions"]]
        samples = [CursorSample(**s) for s in payload["samples"]]
        sessions.append((regions, samples, payload.get("intended_region_id")))
    return sessions


def _synthetic_intentional_dwell_session(rng: np.random.Generator, dwell_seconds: float) -> Session:
    target = Region("TARGET", x0=100, y0=100, x1=300, y1=300)
    other = Region("OTHER", x0=400, y0=100, x1=600, y1=300)
    cx, cy = 200.0, 200.0  # steady in the middle of the target region
    n_frames = int(dwell_seconds * 30) + 10  # ~30fps worth of frames, plus margin
    samples = [
        CursorSample(
            x=cx + rng.normal(scale=2.0), y=cy + rng.normal(scale=2.0), t=i / 30.0, present=True
        )
        for i in range(n_frames)
    ]
    return [target, other], samples, "TARGET"


def _synthetic_pass_through_session(rng: np.random.Generator, dwell_seconds: float) -> Session:
    target = Region("TARGET", x0=100, y0=100, x1=300, y1=300)
    other = Region("OTHER", x0=400, y0=100, x1=600, y1=300)
    # Sweep straight through the target region in well under dwell_seconds.
    n_frames = 6
    duration = dwell_seconds * 0.3
    xs = np.linspace(50, 350, n_frames)
    samples = [
        CursorSample(x=float(x), y=200.0, t=i * (duration / n_frames), present=True)
        for i, x in enumerate(xs)
    ]
    return [target, other], samples, None


def _synthetic_plumbing_check() -> bool:
    print(
        "=" * 78 + "\nNo evals/data/sessions/ found -- running a SYNTHETIC plumbing check.\n"
        "This exercises DwellSelector over randomized procedural cursor paths; it\n"
        "is NOT a measurement of real camera/MediaPipe hand-tracking accuracy.\n"
        "See evals/data/CONSENT.md for how to supply real recorded sessions.\n" + "=" * 78 + "\n"
    )
    rng = np.random.default_rng(11)
    dwell_seconds = DEFAULT_DWELL_SECONDS
    sessions = [_synthetic_intentional_dwell_session(rng, dwell_seconds) for _ in range(40)]
    sessions += [_synthetic_pass_through_session(rng, dwell_seconds) for _ in range(40)]

    correct_rate, false_trigger_rate, n = _score_sessions(sessions, dwell_seconds)
    print(
        f"[SYNTHETIC] correct-intent selection={correct_rate:.1%} "
        f"(bar >= {CORRECT_INTENT_PASS_BAR:.0%}), "
        f"false-trigger rate={false_trigger_rate:.1%} (bar <= {FALSE_TRIGGER_PASS_BAR:.0%}) "
        f"over {n} synthetic sessions"
    )
    print("\nSynthetic plumbing check complete (informational only, always exits 0).")
    return True


def _real_eval() -> bool:
    sessions = _load_real_sessions()
    correct_rate, false_trigger_rate, n = _score_sessions(sessions)
    passed = (
        correct_rate >= CORRECT_INTENT_PASS_BAR and false_trigger_rate <= FALSE_TRIGGER_PASS_BAR
    )
    status = "PASS" if passed else "FAIL"
    print(
        f"[{status}] Real eval over {n} recorded sessions: "
        f"correct-intent selection={correct_rate:.1%} (bar >= {CORRECT_INTENT_PASS_BAR:.0%}), "
        f"false-trigger rate={false_trigger_rate:.1%} (bar <= {FALSE_TRIGGER_PASS_BAR:.0%})"
    )
    return passed


def main() -> int:
    has_real_sessions = SESSIONS_DIR.exists() and any(SESSIONS_DIR.glob("*.json"))
    passed = _real_eval() if has_real_sessions else _synthetic_plumbing_check()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
