"""Pure dwell-to-select state machine -- the deterministic core of the
gesture ballot UX.

Zero MediaPipe/camera dependency: fed `CursorSample` values (already in
screen coordinates), emits `DwellEvent` values. This is what makes "what
counts as a valid selection" fully unit-testable -- see
docs/ARCHITECTURE.md#deterministic-core-vs-thin-cv-shell.
"""

from __future__ import annotations

from dataclasses import dataclass

from contracts.events import (
    CursorSample,
    DwellComplete,
    DwellEvent,
    HoverCancel,
    HoverProgress,
    HoverStart,
    Region,
)

DEFAULT_DWELL_SECONDS = 1.75


@dataclass
class _RegionState:
    hover_start_t: float | None = None
    completed_and_awaiting_exit: bool = False


class DwellSelector:
    """Tracks dwell progress for a fixed set of regions across a stream of
    per-frame `CursorSample` updates.

    - Entering a region starts its timer (`HoverStart`) and emits
      `HoverProgress` on every subsequent update while still hovering.
    - Leaving a region before it completes cancels it (`HoverCancel`) and
      resets its timer to zero.
    - `present=False` (hand lost) is treated exactly like leaving -- no
      partial credit survives a lost hand.
    - Reaching `dwell_seconds` fires `DwellComplete` exactly once. The region
      is then debounced: it cannot fire again until the cursor actually
      exits and re-enters it (prevents a still finger from re-voting).
    """

    def __init__(self, regions: list[Region], dwell_seconds: float = DEFAULT_DWELL_SECONDS):
        if dwell_seconds <= 0:
            raise ValueError("dwell_seconds must be positive")
        self._regions: dict[str, Region] = {r.region_id: r for r in regions}
        self._dwell_seconds = dwell_seconds
        self._state: dict[str, _RegionState] = {
            region_id: _RegionState() for region_id in self._regions
        }
        self._active_region_id: str | None = None

    def _region_at(self, x: float, y: float) -> str | None:
        for region in self._regions.values():
            if region.contains(x, y):
                return region.region_id
        return None

    def update(self, sample: CursorSample) -> list[DwellEvent]:
        events: list[DwellEvent] = []
        hit_region_id = self._region_at(sample.x, sample.y) if sample.present else None

        if self._active_region_id is not None and self._active_region_id != hit_region_id:
            self._deactivate(self._active_region_id, sample.t, events)

        if hit_region_id is None:
            return events

        state = self._state[hit_region_id]

        if state.completed_and_awaiting_exit:
            # Debounced: already fired DwellComplete, not yet exited.
            self._active_region_id = hit_region_id
            return events

        if state.hover_start_t is None:
            state.hover_start_t = sample.t
            self._active_region_id = hit_region_id
            events.append(HoverStart(hit_region_id, sample.t))
            return events

        elapsed = sample.t - state.hover_start_t
        fraction = min(1.0, elapsed / self._dwell_seconds)
        events.append(HoverProgress(hit_region_id, fraction, sample.t))

        if elapsed >= self._dwell_seconds:
            events.append(DwellComplete(hit_region_id, sample.t))
            state.completed_and_awaiting_exit = True

        self._active_region_id = hit_region_id
        return events

    def _deactivate(self, region_id: str, t: float, events: list[DwellEvent]) -> None:
        state = self._state[region_id]
        if state.hover_start_t is not None and not state.completed_and_awaiting_exit:
            events.append(HoverCancel(region_id, t))
        state.hover_start_t = None
        state.completed_and_awaiting_exit = False
        self._active_region_id = None

    def reset(self) -> None:
        """Clears all dwell progress and debounce state (e.g. on screen transition)."""
        self._state = {region_id: _RegionState() for region_id in self._regions}
        self._active_region_id = None
