import queue
import numpy as np
import config as C
from segmenter import Segmenter
import live

N = C.SAMPLE_RATE * C.FRAME_MS // 1000
def tone(): return (0.1 * np.sin(np.linspace(0, 200, N))).astype(np.float32)
def quiet(): return np.zeros(N, dtype=np.float32)


def run(frames):
    s, out = Segmenter(), []
    for f in frames:
        u = s.feed(f)
        if u is not None: out.append(u)
    return out


def test_speech_then_silence_yields_one_utterance_with_preroll():
    fr = [quiet()] * 20 + [tone()] * 30 + [quiet()] * 40
    out = run(fr)
    assert len(out) == 1
    assert len(out[0]) >= 30 * N  # speech frames plus preroll and trailing silence


def test_pure_silence_yields_nothing():
    assert run([quiet()] * 200) == []


def test_short_blip_dropped():
    assert run([tone()] * 3 + [quiet()] * 40) == []


def test_two_utterances_split_by_silence():
    fr = ([tone()] * 30 + [quiet()] * 40) * 2
    assert len(run(fr)) == 2


def test_max_length_forces_cut():
    fr = [tone()] * int(C.MAX_UTTERANCE_S / (C.FRAME_MS / 1000) + 5)
    assert len(run(fr)) >= 1


def test_worker_pipes_and_survives_translation_failure():
    q, lines = queue.Queue(), []
    def bad(_): raise RuntimeError("down")
    q.put(np.ones(10, dtype=np.float32)); q.put(None)
    live.worker(q, transcribe=lambda a: "bawo ni", translate=bad, out=lines.append)
    assert "bawo ni" in lines[0] and "translation failed" in lines[0]


def test_worker_skips_empty_transcript():
    q, lines = queue.Queue(), []
    q.put(np.ones(10, dtype=np.float32)); q.put(None)
    live.worker(q, transcribe=lambda a: "", translate=lambda y: "x", out=lines.append)
    assert lines == []
