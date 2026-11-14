"""Yoruba -> English via local Claude Code (no hosted API)."""
import subprocess
from typing import Callable

from config import CLAUDE_BIN, CLAUDE_MODEL, MAX_INPUT_CHARS, TRANSLATE_TIMEOUT_S

PROMPT = (
    "Translate the following Yoruba text into natural English. "
    "Output ONLY the English translation, no notes, no quotes. "
    "If the text is empty or not Yoruba, output exactly: [unintelligible]\n\n"
    "Yoruba:\n"
)

Runner = Callable[[list[str], str], str]


class TranslationError(RuntimeError):
    pass


def _run_claude(cmd: list[str], stdin: str) -> str:
    p = subprocess.run(cmd, input=stdin, capture_output=True, text=True,
                       timeout=TRANSLATE_TIMEOUT_S)
    if p.returncode != 0:
        raise TranslationError(f"claude exited {p.returncode}: {(p.stdout.strip() or p.stderr.strip())[:300]}")
    return p.stdout


def build_command() -> list[str]:
    return [CLAUDE_BIN, "-p", "--model", CLAUDE_MODEL]


def clean(output: str) -> str:
    return output.strip().strip('"“”').strip()


def translate(yoruba: str, runner: Runner = _run_claude) -> str:
    yoruba = yoruba.strip()
    if not yoruba:
        return "[unintelligible]"
    if len(yoruba) > MAX_INPUT_CHARS:
        raise ValueError(f"input exceeds {MAX_INPUT_CHARS} chars")
    out = clean(runner(build_command(), PROMPT + yoruba))
    if not out:
        raise TranslationError("empty translation")
    return out
