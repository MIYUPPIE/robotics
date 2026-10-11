"""Kiosk timing/display config. Import biometric.thresholds directly for
biometric-specific tunables (liveness retries, match thresholds) rather than
duplicating them here.
"""

from __future__ import annotations

# Per-state timeouts (seconds) -- exceeding these returns to IDLE_ATTRACT
# without writing anything (see docs/ARCHITECTURE.md's abandon-safety
# invariant). SUBMITTING/SUBMIT_FAILED intentionally have no timeout here.
LANGUAGE_SELECT_TIMEOUT_S = 30.0
FACE_AUTH_TIMEOUT_S = 45.0
FACE_AUTH_FAIL_DISPLAY_S = 6.0
ALREADY_VOTED_DISPLAY_S = 6.0
OFFICE_BALLOT_TIMEOUT_S = 60.0
REVIEW_TIMEOUT_S = 60.0
THANK_YOU_DISPLAY_S = 8.0

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

PARTY_GRID_COLUMNS = 5
