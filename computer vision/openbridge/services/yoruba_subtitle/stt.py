"""Yoruba speech-to-text via local faster-whisper. Model loads lazily."""
from config import WHISPER_MODEL, WHISPER_LANGUAGE

_model = None


def transcribe(audio_path: str) -> str:
    global _model
    from faster_whisper import WhisperModel  # lazy: keeps gate tests model-free
    if _model is None:
        _model = WhisperModel(WHISPER_MODEL, device="auto", compute_type="auto")
    segments, _ = _model.transcribe(audio_path, language=WHISPER_LANGUAGE, vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()
