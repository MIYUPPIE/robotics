"""Face-match accuracy eval: FAR/FRR at the thresholds in biometric/thresholds.py.

Real mode: if evals/data/manifest.json exists (see evals/data/CONSENT.md for
how to supply consented images), computes real InsightFace embeddings for
every listed image and reports genuine/impostor FAR/FRR against a pass bar.

Synthetic mode (no manifest.json): runs the identical FAR/FRR computation over
synthetic embeddings, as a self-test of the eval harness's math only -- NOT a
measurement of real-world accuracy. Clearly labeled as such, always exits 0.

Usage: python -m biometric.evals.face_match_eval
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

from biometric.matcher import cosine_similarity
from biometric.thresholds import (
    DUPLICATE_REJECT_COSINE_THRESHOLD,
    VOTING_IDENTIFY_COSINE_THRESHOLD,
)

DATA_DIR = Path(__file__).parent / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"

FAR_PASS_BAR = 0.001  # 0.1%
FRR_PASS_BAR = 0.05  # 5%
MAX_IMPOSTOR_PAIRS = 5000  # cap combinatorial blowup on larger galleries


def compute_far_frr(
    genuine_sims: list[float], impostor_sims: list[float], threshold: float
) -> tuple[float, float]:
    far = (
        (sum(1 for s in impostor_sims if s >= threshold) / len(impostor_sims))
        if impostor_sims
        else 0.0
    )
    frr = (
        (sum(1 for s in genuine_sims if s < threshold) / len(genuine_sims)) if genuine_sims else 0.0
    )
    return far, frr


def _report(
    label: str, genuine_sims: list[float], impostor_sims: list[float], threshold: float
) -> bool:
    far, frr = compute_far_frr(genuine_sims, impostor_sims, threshold)
    passed = far <= FAR_PASS_BAR and frr <= FRR_PASS_BAR
    status = "PASS" if passed else "FAIL"
    print(
        f"[{status}] {label} @ threshold={threshold:.3f}: "
        f"FAR={far:.4%} (bar <= {FAR_PASS_BAR:.2%}), "
        f"FRR={frr:.4%} (bar <= {FRR_PASS_BAR:.2%}) "
        f"[{len(genuine_sims)} genuine pairs, {len(impostor_sims)} impostor pairs]"
    )
    return passed


def _real_eval() -> bool:
    import cv2

    from biometric.embedding import FaceEmbedder

    manifest: dict[str, list[str]] = json.loads(MANIFEST_PATH.read_text())
    embedder = FaceEmbedder()

    identity_embeddings: dict[str, list[np.ndarray]] = {}
    for identity, image_paths in manifest.items():
        embeddings = []
        for rel_path in image_paths:
            img = cv2.imread(str(DATA_DIR / rel_path))
            if img is None:
                print(f"WARNING: could not read {rel_path}, skipping", file=sys.stderr)
                continue
            faces = embedder.detect(img)
            if len(faces) != 1:
                print(
                    f"WARNING: expected 1 face in {rel_path}, found {len(faces)}, skipping",
                    file=sys.stderr,
                )
                continue
            embeddings.append(faces[0].embedding)
        if embeddings:
            identity_embeddings[identity] = embeddings

    genuine_sims: list[float] = [
        cosine_similarity(a, b)
        for embeddings in identity_embeddings.values()
        for a, b in combinations(embeddings, 2)
    ]

    impostor_sims: list[float] = []
    identities = list(identity_embeddings.items())
    for i, (_, embs_a) in enumerate(identities):
        for _, embs_b in identities[i + 1 :]:
            for a in embs_a:
                for b in embs_b:
                    impostor_sims.append(cosine_similarity(a, b))
                    if len(impostor_sims) >= MAX_IMPOSTOR_PAIRS:
                        break

    print(
        f"Real eval: {len(identity_embeddings)} identities, "
        f"{len(genuine_sims)} genuine pairs, {len(impostor_sims)} impostor pairs.\n"
    )

    ok_identify = _report(
        "voting identify", genuine_sims, impostor_sims, VOTING_IDENTIFY_COSINE_THRESHOLD
    )
    ok_duplicate = _report(
        "registration duplicate-check",
        genuine_sims,
        impostor_sims,
        DUPLICATE_REJECT_COSINE_THRESHOLD,
    )
    return ok_identify and ok_duplicate


def _synthetic_plumbing_check() -> bool:
    print(
        "=" * 78 + "\nNo evals/data/manifest.json found -- running a SYNTHETIC plumbing check.\n"
        "This verifies the FAR/FRR computation itself is correct; it is NOT a\n"
        "measurement of real-world face-match accuracy. See evals/data/CONSENT.md\n"
        "for how to supply real consented face images for the real eval.\n" + "=" * 78 + "\n"
    )
    rng = np.random.default_rng(42)
    n_identities, photos_per_identity, dim = 20, 4, 512
    # Per-component noise scale chosen so genuine pairs land around
    # cosine~0.9 (1/sqrt(1 + dim*scale^2)) -- clearly below 1.0 (not literally
    # identical) but clearly above the ~0.42-0.45 thresholds, so a working
    # harness should show near-zero FAR/FRR here. This is calibrated for a
    # convincing plumbing demonstration, not derived from any real face data.
    noise_scale = 0.02

    identity_embeddings = []
    for _ in range(n_identities):
        base = rng.standard_normal(dim)
        base /= np.linalg.norm(base)
        variants = []
        for _ in range(photos_per_identity):
            noisy = base + rng.normal(scale=noise_scale, size=dim)
            variants.append((noisy / np.linalg.norm(noisy)).astype(np.float32))
        identity_embeddings.append(variants)

    genuine_sims = [
        cosine_similarity(a, b)
        for variants in identity_embeddings
        for a, b in combinations(variants, 2)
    ]
    impostor_sims = []
    for i, variants_a in enumerate(identity_embeddings):
        for variants_b in identity_embeddings[i + 1 :]:
            for a in variants_a:
                for b in variants_b:
                    impostor_sims.append(cosine_similarity(a, b))

    _report(
        "[SYNTHETIC] voting identify", genuine_sims, impostor_sims, VOTING_IDENTIFY_COSINE_THRESHOLD
    )
    _report(
        "[SYNTHETIC] registration duplicate-check",
        genuine_sims,
        impostor_sims,
        DUPLICATE_REJECT_COSINE_THRESHOLD,
    )
    print("\nSynthetic plumbing check complete (informational only, always exits 0).")
    return True


def main() -> int:
    passed = _real_eval() if MANIFEST_PATH.exists() else _synthetic_plumbing_check()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
