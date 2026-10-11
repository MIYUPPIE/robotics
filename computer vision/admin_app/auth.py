"""PIN gate for the admin dashboard.

stdlib-only (pbkdf2_hmac + hmac.compare_digest) -- a local soft deterrent
appropriate for a single-machine kiosk, not a full access-control system. No
extra dependency (bcrypt/argon2) needed for this threat model, per
CLAUDE.md's "vanilla by default."
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os

PBKDF2_ITERATIONS = 200_000


def hash_pin(pin: str, salt: bytes | None = None) -> str:
    """Returns "<b64 salt>:<hex digest>" -- store this in ADMIN_PIN_HASH.
    Never store the raw PIN."""
    salt = salt if salt is not None else os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"{base64.b64encode(salt).decode()}:{digest.hex()}"


def verify_pin(pin: str, stored_hash: str) -> bool:
    try:
        salt_b64, hex_digest = stored_hash.split(":", 1)
        salt = base64.b64decode(salt_b64)
        expected = bytes.fromhex(hex_digest)
    except (ValueError, TypeError):
        return False
    actual = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return hmac.compare_digest(actual, expected)


class PinGate:
    """Tracks failed attempts; locks out after `max_attempts`."""

    def __init__(self, stored_hash: str, max_attempts: int = 3):
        self.stored_hash = stored_hash
        self.max_attempts = max_attempts
        self.attempts = 0
        self.locked = False

    def try_pin(self, pin: str) -> bool:
        if self.locked:
            return False
        if verify_pin(pin, self.stored_hash):
            return True
        self.attempts += 1
        if self.attempts >= self.max_attempts:
            self.locked = True
        return False
