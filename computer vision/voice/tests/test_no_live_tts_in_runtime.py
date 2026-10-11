"""Enforces docs/ARCHITECTURE.md ADR-3: no runtime-facing module ever talks
to the network TTS API. If this fails, something under
kiosk_app/hand_ui/registration_app/admin_app imports voice.client or
requests directly -- use voice.cache/voice.playback (the local cached-audio
path) instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FORBIDDEN_MODULES = ("voice.client", "requests")
SCANNED_DIRS = ("kiosk_app", "hand_ui", "registration_app", "admin_app")


def _imported_modules(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _violations() -> list[str]:
    violations = []
    for dir_name in SCANNED_DIRS:
        base = REPO_ROOT / dir_name
        if not base.exists():
            continue
        for py_file in base.rglob("*.py"):
            imported = _imported_modules(py_file)
            for forbidden in FORBIDDEN_MODULES:
                hit = forbidden in imported or any(
                    m == forbidden or m.startswith(forbidden + ".") for m in imported
                )
                if hit:
                    violations.append(f"{py_file.relative_to(REPO_ROOT)} imports {forbidden!r}")
    return violations


def test_no_runtime_module_imports_live_tts_dependencies():
    violations = _violations()
    assert not violations, "Live-TTS import found in a runtime-facing module:\n" + "\n".join(
        violations
    )


def test_scanner_actually_detects_a_violation():
    """A test for the test: make sure `_violations` isn't vacuously passing
    because the AST scan is broken (e.g. always returning an empty set)."""
    fake_file = REPO_ROOT / "voice" / "tests" / "_fixture_violation.py.txt"
    fake_file.write_text("import requests\n")
    try:
        fake_module_set = _imported_modules(fake_file)
        assert "requests" in fake_module_set
    finally:
        fake_file.unlink()
