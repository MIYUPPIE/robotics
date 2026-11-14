"""08a CSV logging: light-intensity/compensation history and state-change events."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import IO, Any

from .pipeline import FrameResult

HISTORY_COLUMNS = [
    "session",
    "timestamp",
    "t_s",
    "frame",
    "phase",
    "state",
    "level",
    "level_filtered",
    "ref_level",
    "ratio",
    "gain",
    "gamma",
    "clahe_clip",
    "dropped_fraction",
    "output_level",
    "brightness_error_pct",
    "fps",
    "latency_ms",
]
EVENT_COLUMNS = ["session", "timestamp", "t_s", "event", "detail"]


def _open_csv(path: Path, columns: list[str]) -> tuple[IO[str], Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    fh = path.open("a", newline="")
    writer = csv.writer(fh)
    if new_file:
        writer.writerow(columns)
    return fh, writer


def _fmt(x: float) -> str:
    return "" if x != x else f"{x:.5f}"  # NaN -> empty cell


class HistoryLogger:
    """Rate-limited history rows plus an always-written event log (state changes, calibration)."""

    def __init__(self, history_path: Path, events_path: Path, interval_s: float) -> None:
        self._interval = interval_s
        self._session = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self._hist_fh, self._hist = _open_csv(history_path, HISTORY_COLUMNS)
        self._evt_fh, self._evt = _open_csv(events_path, EVENT_COLUMNS)
        self._last_row_t: float | None = None
        self._frame = 0
        self.rows_written = 0
        self.events_written = 0

    def _now(self) -> str:
        return datetime.now(UTC).isoformat(timespec="milliseconds")

    def event(self, t: float, event: str, detail: str = "") -> None:
        self._evt.writerow([self._session, self._now(), f"{t:.3f}", event, detail])
        self._evt_fh.flush()
        self.events_written += 1

    def log(self, t: float, r: FrameResult, fps: float) -> None:
        self._frame += 1
        if r.transition is not None:
            tr = r.transition
            self.event(t, "state_change", f"{tr.old.value}->{tr.new.value} ratio={tr.ratio:.3f}")
        if r.calibrated_now:
            self.event(t, "calibrated", f"level={r.ref_level:.4f}")
        if r.message and not r.calibrated_now:
            self.event(t, "message", r.message)
        if (
            self._last_row_t is not None
            and t - self._last_row_t < self._interval
            and r.transition is None
        ):
            return
        self._last_row_t = t
        self._hist.writerow(
            [
                self._session,
                self._now(),
                f"{t:.3f}",
                self._frame,
                r.phase.value,
                r.state.value,
                _fmt(r.level),
                _fmt(r.level_filtered),
                _fmt(r.ref_level),
                _fmt(r.ratio),
                _fmt(r.gain),
                _fmt(r.gamma),
                _fmt(r.clahe_clip),
                _fmt(r.dropped_fraction),
                _fmt(r.output_level),
                _fmt(r.brightness_error_pct),
                f"{fps:.1f}",
                f"{r.latency_ms:.2f}",
            ]
        )
        self.rows_written += 1
        if self.rows_written % 25 == 0:
            self._hist_fh.flush()

    def close(self) -> None:
        for fh in (self._hist_fh, self._evt_fh):
            if not fh.closed:
                fh.flush()
                fh.close()

    def __enter__(self) -> HistoryLogger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
