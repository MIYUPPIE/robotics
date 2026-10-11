"""Embedded webcam preview widget for the Tkinter registration app.

Standard Tkinter+OpenCV pattern: a Label widget updated via `.after()`,
converting each BGR frame to a PhotoImage. Not gate-tested (needs a display
and a camera) -- see registration_app/capture_flow.py for the tested
orchestration logic the "Capture" button triggers.

Frames are mirrored (`cv2.flip(frame, 1)`) before being stored/displayed,
matching kiosk_app/main.py::run(). This isn't just cosmetic: biometric
face embeddings must be computed on the same orientation at registration
time and at voting time, or a genuine voter's stored embedding may not
reliably match themselves later -- see hand_ui/landmarker.py's module
docstring for the full mirroring rationale.
"""

from __future__ import annotations

import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk


class WebcamWidget(tk.Label):
    def __init__(self, master: tk.Widget, camera_index: int = 0, fps: int = 30):
        super().__init__(master)
        self._cap = cv2.VideoCapture(camera_index)
        self._interval_ms = int(1000 / fps)
        self._last_frame: np.ndarray | None = None
        self._running = False
        self._photo: ImageTk.PhotoImage | None = None  # keeps a ref so Tk doesn't GC it

    def start(self) -> None:
        self._running = True
        self._tick()

    def stop(self) -> None:
        self._running = False
        self._cap.release()

    def current_frame(self) -> np.ndarray | None:
        return self._last_frame

    def _tick(self) -> None:
        if not self._running:
            return
        ok, frame = self._cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            self._last_frame = frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self.configure(image=self._photo)
        self.after(self._interval_ms, self._tick)
