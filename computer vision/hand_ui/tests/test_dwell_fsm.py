import pytest

from contracts.events import (
    CursorSample,
    DwellComplete,
    HoverCancel,
    HoverProgress,
    HoverStart,
    Region,
)
from hand_ui.dwell_fsm import DwellSelector

REGION_A = Region("A", x0=0, y0=0, x1=100, y1=100)
REGION_B = Region("B", x0=200, y0=0, x1=300, y1=100)
OUTSIDE = (500.0, 500.0)
INSIDE_A = (50.0, 50.0)
INSIDE_B = (250.0, 50.0)


def _sample(x, y, t, present=True) -> CursorSample:
    return CursorSample(x=x, y=y, t=t, present=present)


class TestHoverLifecycle:
    def test_entering_a_region_emits_hover_start(self):
        selector = DwellSelector([REGION_A], dwell_seconds=1.0)
        events = selector.update(_sample(*INSIDE_A, t=0.0))
        assert events == [HoverStart("A", 0.0)]

    def test_continued_hover_emits_progress_with_increasing_fraction(self):
        selector = DwellSelector([REGION_A], dwell_seconds=1.0)
        selector.update(_sample(*INSIDE_A, t=0.0))
        events = selector.update(_sample(*INSIDE_A, t=0.5))
        assert events == [HoverProgress("A", 0.5, 0.5)]

    def test_dwell_complete_fires_exactly_once_at_threshold(self):
        selector = DwellSelector([REGION_A], dwell_seconds=1.0)
        selector.update(_sample(*INSIDE_A, t=0.0))
        events = selector.update(_sample(*INSIDE_A, t=1.0))
        assert HoverProgress("A", 1.0, 1.0) in events
        assert DwellComplete("A", 1.0) in events

    def test_no_events_when_never_entering_any_region(self):
        selector = DwellSelector([REGION_A], dwell_seconds=1.0)
        assert selector.update(_sample(*OUTSIDE, t=0.0)) == []
        assert selector.update(_sample(*OUTSIDE, t=1.0)) == []


class TestCancellation:
    def test_leaving_before_completion_emits_hover_cancel(self):
        selector = DwellSelector([REGION_A], dwell_seconds=1.0)
        selector.update(_sample(*INSIDE_A, t=0.0))
        selector.update(_sample(*INSIDE_A, t=0.4))
        events = selector.update(_sample(*OUTSIDE, t=0.5))
        assert events == [HoverCancel("A", 0.5)]

    def test_hand_lost_cancels_progress_immediately(self):
        selector = DwellSelector([REGION_A], dwell_seconds=1.0)
        selector.update(_sample(*INSIDE_A, t=0.0))
        events = selector.update(_sample(0.0, 0.0, t=0.3, present=False))
        assert events == [HoverCancel("A", 0.3)]

    def test_reentry_after_cancel_restarts_from_zero(self):
        selector = DwellSelector([REGION_A], dwell_seconds=1.0)
        selector.update(_sample(*INSIDE_A, t=0.0))
        selector.update(_sample(*OUTSIDE, t=0.4))  # cancel
        events = selector.update(_sample(*INSIDE_A, t=0.4))
        assert events == [HoverStart("A", 0.4)]
        # And it takes a full dwell_seconds from *this* start to complete
        # (well past the 1.4 threshold, with margin against float error).
        events = selector.update(_sample(*INSIDE_A, t=1.5))
        assert DwellComplete("A", 1.5) in events

    def test_hand_lost_with_no_prior_hover_emits_nothing(self):
        selector = DwellSelector([REGION_A], dwell_seconds=1.0)
        assert selector.update(_sample(0.0, 0.0, t=0.0, present=False)) == []


class TestDebounce:
    def test_no_repeat_dwell_complete_while_still_hovering(self):
        selector = DwellSelector([REGION_A], dwell_seconds=1.0)
        selector.update(_sample(*INSIDE_A, t=0.0))
        events_at_complete = selector.update(_sample(*INSIDE_A, t=1.0))
        assert DwellComplete("A", 1.0) in events_at_complete

        events_after = selector.update(_sample(*INSIDE_A, t=1.5))
        assert events_after == []

        events_much_later = selector.update(_sample(*INSIDE_A, t=3.0))
        assert events_much_later == []

    def test_exiting_a_completed_region_does_not_emit_hover_cancel(self):
        selector = DwellSelector([REGION_A], dwell_seconds=1.0)
        selector.update(_sample(*INSIDE_A, t=0.0))
        selector.update(_sample(*INSIDE_A, t=1.0))  # completes
        events = selector.update(_sample(*OUTSIDE, t=1.2))
        assert events == []

    def test_dwell_complete_can_fire_again_after_exit_and_reentry(self):
        selector = DwellSelector([REGION_A], dwell_seconds=1.0)
        selector.update(_sample(*INSIDE_A, t=0.0))
        selector.update(_sample(*INSIDE_A, t=1.0))  # first completion
        selector.update(_sample(*OUTSIDE, t=1.2))  # exit, clears debounce

        selector.update(_sample(*INSIDE_A, t=1.2))  # re-enter
        events = selector.update(_sample(*INSIDE_A, t=2.2))  # dwell again
        assert DwellComplete("A", 2.2) in events


class TestMultipleRegions:
    def test_switching_regions_cancels_the_first_and_starts_the_second(self):
        selector = DwellSelector([REGION_A, REGION_B], dwell_seconds=1.0)
        selector.update(_sample(*INSIDE_A, t=0.0))
        events = selector.update(_sample(*INSIDE_B, t=0.3))
        assert events == [HoverCancel("A", 0.3), HoverStart("B", 0.3)]

    def test_independent_regions_track_independently(self):
        selector = DwellSelector([REGION_A, REGION_B], dwell_seconds=1.0)
        selector.update(_sample(*INSIDE_A, t=0.0))
        selector.update(_sample(*INSIDE_A, t=1.0))  # A completes
        selector.update(_sample(*INSIDE_B, t=1.1))  # move to B (cancels nothing, A was done)
        events = selector.update(_sample(*INSIDE_B, t=2.1))
        assert DwellComplete("B", 2.1) in events


class TestReset:
    def test_reset_clears_in_progress_dwell(self):
        selector = DwellSelector([REGION_A], dwell_seconds=1.0)
        selector.update(_sample(*INSIDE_A, t=0.0))
        selector.reset()
        events = selector.update(_sample(*INSIDE_A, t=0.1))
        assert events == [HoverStart("A", 0.1)]

    def test_reset_clears_debounce(self):
        selector = DwellSelector([REGION_A], dwell_seconds=1.0)
        selector.update(_sample(*INSIDE_A, t=0.0))
        selector.update(_sample(*INSIDE_A, t=1.0))  # completes
        selector.reset()
        events = selector.update(_sample(*INSIDE_A, t=1.1))
        assert events == [HoverStart("A", 1.1)]


class TestConstruction:
    def test_nonpositive_dwell_seconds_rejected(self):
        with pytest.raises(ValueError):
            DwellSelector([REGION_A], dwell_seconds=0)
        with pytest.raises(ValueError):
            DwellSelector([REGION_A], dwell_seconds=-1.0)
