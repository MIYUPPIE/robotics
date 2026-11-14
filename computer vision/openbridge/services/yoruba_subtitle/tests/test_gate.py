"""Gate tests: deterministic, no model, no network."""
import pytest
import pipeline
import translate as tr


def fake_runner(reply):
    calls = []
    def run(cmd, stdin):
        calls.append((cmd, stdin))
        return reply
    run.calls = calls
    return run


def test_translate_passes_yoruba_in_prompt_and_cleans_output():
    r = fake_runner('  "Please, can you help me?"\n')
    assert tr.translate("Ẹ jọ̀ọ́, ṣé ẹ lè ràn mí lọ́wọ́?", runner=r) == "Please, can you help me?"
    cmd, stdin = r.calls[0]
    assert "Ẹ jọ̀ọ́" in stdin and cmd[1] == "-p" and "--model" in cmd


def test_empty_input_never_calls_model():
    r = fake_runner("x")
    assert tr.translate("   ", runner=r) == "[unintelligible]"
    assert r.calls == []


def test_oversize_input_rejected():
    with pytest.raises(ValueError):
        tr.translate("a" * 5000, runner=fake_runner("x"))


def test_empty_model_output_raises():
    with pytest.raises(tr.TranslationError):
        tr.translate("bawo", runner=fake_runner("  \n"))


def test_nonzero_exit_raises(monkeypatch):
    class P: returncode, stdout, stderr = 1, "", "boom"
    monkeypatch.setattr(tr.subprocess, "run", lambda *a, **k: P)
    with pytest.raises(tr.TranslationError, match="boom"):
        tr._run_claude(["claude"], "x")


def test_pipeline_wires_stages_and_times_them():
    s = pipeline.run("c.wav", transcribe=lambda p: "bawo ni", translate=lambda y: f"EN({y})")
    assert (s.yoruba, s.english) == ("bawo ni", "EN(bawo ni)")
    assert s.stt_seconds >= 0 and s.translate_seconds >= 0


def test_stt_module_imports_without_loading_model():
    import stt
    assert stt._model is None


def test_eval_scoring_ignores_accents():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "ev", pathlib.Path(__file__).resolve().parents[1] / "evals" / "eval_translate.py")
    ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    assert ev.score("My name is Adé", [["name"], ["ade"]])
    assert not ev.score("My name is Bola", [["name"], ["ade"]])
