# yoruba_subtitle

Workstream A slice from `document.md` section 18: Yoruba audio -> Yoruba text (local
faster-whisper `large-v3`) -> English subtitle (local Claude Code, `claude -p`, no hosted API).

Outcome moved: first measurable Yoruba->English path. Metric: eval pass rate (>= 80%) and
per-stage latency in `Subtitle.stt_seconds` / `translate_seconds`.

## Run
    python3 -m venv .venv  (or conda create -p .venv python=3.12) && .venv/bin/pip install faster-whisper pytest
    env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest   # gate, <1s
    env -u PYTHONPATH .venv/bin/python evals/eval_translate.py                       # paid eval
    .venv/bin/python pipeline.py clip.wav

Env: `OPENBRIDGE_CLAUDE_BIN`, `OPENBRIDGE_CLAUDE_MODEL` (default `claude-fable-5-2`),
`OPENBRIDGE_WHISPER_MODEL`. Audio evals: add `evals/clips/x.wav` + `x.txt` (reference English).

## Known gaps
- STT path is untested against real Yoruba audio (no clip yet; ~3GB model download on first run).
- Translation quality is only spot-checked on 5 sentences; needs a community-validated set.
