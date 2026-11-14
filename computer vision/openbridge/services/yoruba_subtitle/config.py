"""All tunables live here, per repo rules."""
import os

WHISPER_MODEL = os.environ.get("OPENBRIDGE_WHISPER_MODEL", "large-v3")
WHISPER_LANGUAGE = "yo"
CLAUDE_BIN = os.environ.get("OPENBRIDGE_CLAUDE_BIN", "claude")
# Best available model by default; override only on instruction.
CLAUDE_MODEL = os.environ.get("OPENBRIDGE_CLAUDE_MODEL", "claude-fable-5-2")
TRANSLATE_TIMEOUT_S = 120
MAX_INPUT_CHARS = 2000

# Live mode
SAMPLE_RATE = 16000
FRAME_MS = 30
SPEECH_RMS = 0.015          # frame RMS above this counts as speech
END_SILENCE_S = 0.7         # silence that closes an utterance
MIN_UTTERANCE_S = 0.4       # shorter blips are dropped
MAX_UTTERANCE_S = 15.0      # force a cut so subtitles never stall
PREROLL_S = 0.2             # audio kept before speech onset
