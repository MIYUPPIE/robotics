"""Single source of truth for every tunable biometric constant.

These are starting points, not final values -- see biometric/evals/face_match_eval.py
and biometric/evals/liveness_eval.py for the empirical tuning process. Import from
here rather than hardcoding a threshold anywhere else in the codebase.
"""

# --- Liveness (EAR blink detection, biometric/liveness.py) ---
# 0.20-0.25 is the classic range from Soukupová & Čech's real-time blink
# detector; 0.21 favors fewer false "no blink" rejections over stricter closure.
LIVENESS_EAR_THRESHOLD = 0.21
LIVENESS_CONSEC_FRAMES = 2
LIVENESS_BLINK_WINDOW_S = 5.0  # registration: more time, officer-supervised
LIVENESS_BLINK_WINDOW_VOTING_S = 3.0  # voting: shorter, unsupervised
MAX_LIVENESS_RETRIES = 3

# --- Face matching (biometric/matcher.py) ---
# ArcFace (InsightFace buffalo_l) L2-normalized embeddings; cosine similarity
# via normalized dot product. 0.42-0.45 is a conservative starting separation
# point pending the labeled eval set -- see biometric/evals/face_match_eval.py.
DUPLICATE_REJECT_COSINE_THRESHOLD = 0.45  # registration: favor fewer false accusations
VOTING_IDENTIFY_COSINE_THRESHOLD = 0.42  # voting: 1:N identify
VOTING_MATCH_MARGIN = 0.05  # best - runner_up must clear this or result is AMBIGUOUS

# --- Burst capture (biometric/capture.py) ---
BURST_CAPTURE_FRAMES = 5
BURST_CAPTURE_TIMEOUT_S = 3.0
BURST_MIN_DETECTOR_CONFIDENCE = 0.5
EMBEDDING_FRAMES_TO_AVERAGE = 3
