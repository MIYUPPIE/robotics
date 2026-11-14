"""All tunables live here, per repo rules."""
import os

WHISPER_MODEL = os.environ.get("OPENBRIDGE_WHISPER_MODEL", "large-v3")
WHISPER_LANGUAGE = "yo"
CLAUDE_BIN = os.environ.get("OPENBRIDGE_CLAUDE_BIN", "claude")
# Best available model by default; override only on instruction.
CLAUDE_MODEL = os.environ.get("OPENBRIDGE_CLAUDE_MODEL", "claude-fable-5-2")
TRANSLATE_TIMEOUT_S = 120
MAX_INPUT_CHARS = 2000
