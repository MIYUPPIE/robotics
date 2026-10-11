from itertools import pairwise

from biometric.face_mesh import next_video_timestamp_ms


class TestNextVideoTimestampMs:
    def test_increasing_caller_timestamp_passes_through(self):
        assert next_video_timestamp_ms(100, 50) == 100

    def test_equal_caller_timestamp_is_ratcheted_forward(self):
        assert next_video_timestamp_ms(50, 50) == 51

    def test_a_reset_caller_timestamp_is_ratcheted_forward(self):
        """The exact real-world bug: a caller (e.g. registration_app calling
        biometric.capture.capture_liveness fresh on every button click)
        restarts its own clock near zero on a second call, while the
        landmarker instance already saw much larger timestamps from the
        first call. This must never go backwards or repeat."""
        assert next_video_timestamp_ms(3, 3042) == 3043

    def test_first_call_with_fresh_tracker_state(self):
        assert next_video_timestamp_ms(0, -1) == 0

    def test_strictly_increasing_across_a_sequence(self):
        last = -1
        caller_values = [0, 1, 2, 0, 1, 2, 5000, 0, 1]  # simulates repeated resets
        seen = []
        for caller_ts in caller_values:
            ts = next_video_timestamp_ms(caller_ts, last)
            seen.append(ts)
            last = ts
        assert seen == sorted(set(seen))  # strictly increasing, no duplicates
        assert all(b > a for a, b in pairwise(seen))
