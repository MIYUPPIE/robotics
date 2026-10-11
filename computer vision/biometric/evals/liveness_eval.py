"""Liveness (blink) detection eval: pass-rate on recorded or synthetic sessions.

Real mode: if evals/data/liveness_sessions/*.json exist (see
evals/data/CONSENT.md), runs detect_blink over each recorded session and
reports blink recall + static-photo-replay rejection against a pass bar.

Synthetic mode (no recorded sessions): runs the identical accuracy
computation over programmatically generated blink / static-replay sequences --
a self-test of the eval harness, not a measurement of real-world accuracy.

Usage: python -m biometric.evals.liveness_eval
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from biometric.liveness import LEFT_EYE_EAR_INDICES, RIGHT_EYE_EAR_INDICES, detect_blink

DATA_DIR = Path(__file__).parent / "data"
SESSIONS_DIR = DATA_DIR / "liveness_sessions"

BLINK_DETECTION_PASS_BAR = 0.90  # >=90% of real blink clips must be detected
STATIC_REPLAY_REJECTION_BAR = 1.00  # 100% of static-photo replays must be rejected


def _eye_points(ear: float, width: float = 1.0) -> np.ndarray:
    h = ear * width / 2.0
    return np.array(
        [
            (0.0, 0.0),
            (0.25 * width, h),
            (0.75 * width, h),
            (width, 0.0),
            (0.75 * width, -h),
            (0.25 * width, -h),
        ]
    )


def _synthetic_face_landmarks(left_ear: float, right_ear: float, n_points: int = 478) -> np.ndarray:
    landmarks = np.zeros((n_points, 2))
    for idx, pt in zip(LEFT_EYE_EAR_INDICES, _eye_points(left_ear), strict=True):
        landmarks[idx] = pt
    for idx, pt in zip(RIGHT_EYE_EAR_INDICES, _eye_points(right_ear), strict=True):
        landmarks[idx] = pt
    return landmarks


def _synthetic_blink_session(rng: np.random.Generator) -> list[np.ndarray]:
    open_ear = 0.30 + rng.uniform(-0.03, 0.03)
    closed_ear = 0.10 + rng.uniform(-0.02, 0.02)
    return (
        [_synthetic_face_landmarks(open_ear, open_ear) for _ in range(int(rng.integers(3, 8)))]
        + [
            _synthetic_face_landmarks(closed_ear, closed_ear)
            for _ in range(int(rng.integers(2, 4)))
        ]
        + [_synthetic_face_landmarks(open_ear, open_ear) for _ in range(int(rng.integers(3, 8)))]
    )


def _synthetic_static_replay_session(rng: np.random.Generator) -> list[np.ndarray]:
    ear = 0.28 + rng.uniform(-0.02, 0.02)  # a static photo: constant, no closure
    return [_synthetic_face_landmarks(ear, ear) for _ in range(15)]


def _run_sessions(sessions: list[tuple[list[np.ndarray], bool]]) -> tuple[float, float]:
    """sessions: (landmark_stream, should_blink) pairs.

    Returns (blink_recall, replay_rejection_rate).
    """
    blink_sessions = [s for s in sessions if s[1]]
    replay_sessions = [s for s in sessions if not s[1]]

    blink_hits = sum(1 for stream, _ in blink_sessions if detect_blink(stream).is_live)
    replay_rejections = sum(1 for stream, _ in replay_sessions if not detect_blink(stream).is_live)

    blink_recall = blink_hits / len(blink_sessions) if blink_sessions else 1.0
    replay_rejection_rate = replay_rejections / len(replay_sessions) if replay_sessions else 1.0
    return blink_recall, replay_rejection_rate


def _real_eval() -> bool:
    sessions: list[tuple[list[np.ndarray], bool]] = []
    for path in sorted(SESSIONS_DIR.glob("*.json")):
        payload = json.loads(path.read_text())
        stream = [np.array(frame) for frame in payload["landmarks"]]
        sessions.append((stream, bool(payload["should_blink"])))

    blink_recall, replay_rejection_rate = _run_sessions(sessions)
    passed = (
        blink_recall >= BLINK_DETECTION_PASS_BAR
        and replay_rejection_rate >= STATIC_REPLAY_REJECTION_BAR
    )
    status = "PASS" if passed else "FAIL"
    print(
        f"[{status}] Real eval over {len(sessions)} recorded sessions: "
        f"blink recall={blink_recall:.1%} (bar >= {BLINK_DETECTION_PASS_BAR:.0%}), "
        f"static-replay rejection={replay_rejection_rate:.1%} "
        f"(bar >= {STATIC_REPLAY_REJECTION_BAR:.0%})"
    )
    return passed


def _synthetic_plumbing_check() -> bool:
    print(
        "=" * 78 + "\nNo evals/data/liveness_sessions/ found -- running a SYNTHETIC plumbing\n"
        "check. This verifies the blink-detection harness itself is correct; it\n"
        "is NOT a measurement of real-world liveness accuracy. See\n"
        "evals/data/CONSENT.md for how to supply real recorded sessions.\n" + "=" * 78 + "\n"
    )
    rng = np.random.default_rng(7)
    sessions: list[tuple[list[np.ndarray], bool]] = [
        (_synthetic_blink_session(rng), True) for _ in range(30)
    ]
    sessions += [(_synthetic_static_replay_session(rng), False) for _ in range(30)]

    blink_recall, replay_rejection_rate = _run_sessions(sessions)
    print(
        f"[SYNTHETIC] blink recall={blink_recall:.1%}, "
        f"static-replay rejection={replay_rejection_rate:.1%} "
        f"(over {len(sessions)} synthetic sessions)"
    )
    print("\nSynthetic plumbing check complete (informational only, always exits 0).")
    return True


def main() -> int:
    has_real_sessions = SESSIONS_DIR.exists() and any(SESSIONS_DIR.glob("*.json"))
    passed = _real_eval() if has_real_sessions else _synthetic_plumbing_check()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
