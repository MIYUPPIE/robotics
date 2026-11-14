"""Real-time Yoruba -> English subtitles from the microphone. Ctrl+C to stop.
Run: python live.py [--device N] [--list-devices]"""
import argparse
import queue
import threading
import time

import numpy as np

import config as C
import stt
import translate as tr
from segmenter import Segmenter


def worker(q: "queue.Queue", transcribe=stt.transcribe, translate=tr.translate, out=print):
    while True:
        audio = q.get()
        if audio is None:
            return
        t0 = time.perf_counter()
        yo = transcribe(audio)
        if not yo:
            continue
        t1 = time.perf_counter()
        try:
            en = translate(yo)
        except Exception as e:  # keep the live loop alive
            en = f"[translation failed: {e}]"
        out(f"YO  {yo}\nEN  {en}   (stt {t1 - t0:.1f}s, tr {time.perf_counter() - t1:.1f}s)\n")


def main():
    import sounddevice as sd
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int)
    ap.add_argument("--list-devices", action="store_true")
    a = ap.parse_args()
    if a.list_devices:
        print(sd.query_devices()); return
    q: queue.Queue = queue.Queue()
    threading.Thread(target=worker, args=(q,), daemon=True).start()
    seg = Segmenter()
    n = C.SAMPLE_RATE * C.FRAME_MS // 1000
    print("Loading model...", flush=True)
    stt.transcribe(np.zeros(C.SAMPLE_RATE, dtype=np.float32))  # warm up
    print("Listening. Speak Yoruba.\n", flush=True)
    with sd.InputStream(samplerate=C.SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=n, device=a.device) as s:
        try:
            while True:
                frame, _ = s.read(n)
                utt = seg.feed(frame[:, 0])
                if utt is not None:
                    q.put(utt)
        except KeyboardInterrupt:
            q.put(None)


if __name__ == "__main__":
    main()
