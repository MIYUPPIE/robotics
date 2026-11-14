"""Public contract for the yoruba_subtitle service."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Subtitle:
    yoruba: str
    english: str
    stt_seconds: float
    translate_seconds: float
