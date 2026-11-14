"""Pure energy-VAD utterance segmenter. No audio device, no model imports."""
from collections import deque

import numpy as np

import config as C


class Segmenter:
    def __init__(self, sr=C.SAMPLE_RATE, frame_ms=C.FRAME_MS):
        self.frame_s = frame_ms / 1000
        self.pre = deque(maxlen=max(1, round(C.PREROLL_S / self.frame_s)))
        self.buf: list[np.ndarray] = []
        self.silence = 0.0
        self.active = False

    def _close(self):
        dur = len(self.buf) * self.frame_s - self.silence
        out = np.concatenate(self.buf) if self.buf and dur >= C.MIN_UTTERANCE_S else None
        self.buf, self.silence, self.active = [], 0.0, False
        return out

    def feed(self, frame: np.ndarray):
        """Feed one frame; returns a finished utterance (float32 array) or None."""
        loud = float(np.sqrt(np.mean(frame ** 2))) >= C.SPEECH_RMS
        if not self.active:
            if not loud:
                self.pre.append(frame)
                return None
            self.active, self.buf = True, list(self.pre)
            self.pre.clear()
        self.buf.append(frame)
        self.silence = 0.0 if loud else self.silence + self.frame_s
        if self.silence >= C.END_SILENCE_S or len(self.buf) * self.frame_s >= C.MAX_UTTERANCE_S:
            return self._close()
        return None
