"""Audio file -> Subtitle. Run: python pipeline.py clip.wav"""
import sys
import time
from typing import Callable

import stt
import translate as tr
from contract import Subtitle


def run(audio_path: str,
        transcribe: Callable[[str], str] = stt.transcribe,
        translate: Callable[[str], str] = tr.translate) -> Subtitle:
    t0 = time.perf_counter()
    yo = transcribe(audio_path)
    t1 = time.perf_counter()
    en = translate(yo)
    t2 = time.perf_counter()
    return Subtitle(yo, en, round(t1 - t0, 3), round(t2 - t1, 3))


if __name__ == "__main__":
    print(run(sys.argv[1]))
