"""YarnGPT2 hosted API client (https://yarngpt.ai/api-docs).

The ONLY file allowed to talk to the network TTS API. Imported exclusively by
voice/generate_prompts.py -- never by kiosk_app, hand_ui, registration_app, or
admin_app (enforced by voice/tests/test_no_live_tts_in_runtime.py, see
docs/ARCHITECTURE.md ADR-3).

Verified against a live call on 2026-08-01: POST https://yarngpt.ai/api/v1/tts,
body {"text": ..., "voice": "<TitleCase speaker name>"}, no separate language
field (the voice name alone determines language/accent). Response is raw
audio bytes (observed as WAV; the API also supports mp3/opus/flac via an
optional response_format field, not currently sent).
"""

from __future__ import annotations

import os

import requests

from contracts.schemas import Language

DEFAULT_BASE_URL = "https://yarngpt.ai/api/v1"
DEFAULT_TIMEOUT_S = 60.0


class YarnGptError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class YarnGptClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_S,
    ):
        self.api_key = api_key or os.environ.get("YARNGPT_API_KEY")
        if not self.api_key:
            raise YarnGptError(
                "YARNGPT_API_KEY is not set. Put it in .env (see .env.example) "
                "before running `python -m voice.generate_prompts`."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def synthesize(self, text: str, language: Language, speaker: str) -> bytes:
        """Returns raw audio bytes for `text` spoken by `speaker`.

        `language` isn't sent to the API -- the voice name alone determines
        the accent/language. Kept as a parameter for call-site clarity (see
        voice/phrase_registry.py::DEFAULT_SPEAKERS) and so this signature
        doesn't need to change if the API adds explicit language support.
        """
        voice = speaker.capitalize() if speaker else "Idera"
        response = requests.post(
            f"{self.base_url}/tts",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"text": text, "voice": voice},
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise YarnGptError(
                f"YarnGPT API returned {response.status_code} for phrase {text!r}: "
                f"{response.text[:500]}",
                status_code=response.status_code,
            )
        return response.content
