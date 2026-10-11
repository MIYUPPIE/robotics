# biometric

Face embedding, liveness (blink) detection, 1:N matching, and voter ID generation.

- `thresholds.py` -- single source of truth for every tunable constant.
- `liveness.py` -- EAR blink detection, pure over face-mesh landmark arrays.
- `embedding.py` -- InsightFace (`buffalo_l`) wrapper; the only file that imports
  `insightface`/`onnxruntime`.
- `face_mesh.py` -- MediaPipe FaceLandmarker wrapper, producing the plain landmark
  arrays `liveness.py` consumes; the only file that imports MediaPipe's face model.
- `matcher.py` -- pure cosine-similarity 1:N matching (`find_duplicates`, `identify`).
  `identify` never guesses: a match too close to the runner-up comes back `ambiguous`.
- `capture.py` -- camera burst-capture (`burst_capture`) and liveness collection
  (`capture_liveness`) orchestration, dependency-injected (`read_frame`, `embedder`,
  `face_mesh`, `clock`) so the logic is gate-tested without a camera or real models.
- `voter_id.py` -- ID generation, deliberately not INEC-VIN-shaped.

## Tests vs. evals

Gate tests (`tests/`, <2s, no camera/network/real model) cover every pure function
and the dependency-injected orchestration logic in `capture.py`, using synthetic
landmarks/embeddings/fakes.

Evals (`evals/`, real models, pass-threshold gated) measure actual accuracy:
`face_match_eval.py` (FAR/FRR) and `liveness_eval.py` (blink recall / spoof
rejection). Both run a clearly-labeled **synthetic plumbing check** instead when no
real consented data is present in `evals/data/` -- see `evals/data/CONSENT.md` before
adding any real biometric data there.

Run: `pytest biometric/` (gate) / `python -m biometric.evals.face_match_eval` and
`python -m biometric.evals.liveness_eval` (evals).
