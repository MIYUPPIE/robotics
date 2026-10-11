from itertools import pairwise

import numpy as np
import pytest

from hand_ui.landmarker import OneEuroFilter, next_video_timestamp_ms


class TestNextVideoTimestampMs:
    def test_increasing_caller_timestamp_passes_through(self):
        assert next_video_timestamp_ms(100, 50) == 100

    def test_equal_caller_timestamp_is_ratcheted_forward(self):
        assert next_video_timestamp_ms(50, 50) == 51

    def test_a_reset_caller_timestamp_is_ratcheted_forward(self):
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
        assert seen == sorted(set(seen))
        assert all(b > a for a, b in pairwise(seen))


class TestOneEuroFilterBasics:
    def test_first_sample_returned_unchanged(self):
        f = OneEuroFilter()
        assert f.filter(5.0, 0.0) == 5.0

    def test_constant_signal_stays_constant_after_first_sample(self):
        f = OneEuroFilter()
        f.filter(10.0, 0.0)
        for i in range(1, 10):
            assert f.filter(10.0, i * 0.033) == pytest.approx(10.0)

    def test_reset_makes_next_sample_pass_through_unchanged(self):
        f = OneEuroFilter()
        f.filter(0.0, 0.0)
        f.filter(100.0, 0.033)  # smoothed -- won't equal 100 exactly
        f.reset()
        assert f.filter(50.0, 0.066) == 50.0

    def test_rejects_nonpositive_min_cutoff(self):
        with pytest.raises(ValueError):
            OneEuroFilter(min_cutoff=0.0)
        with pytest.raises(ValueError):
            OneEuroFilter(min_cutoff=-1.0)

    def test_rejects_nonpositive_d_cutoff(self):
        with pytest.raises(ValueError):
            OneEuroFilter(d_cutoff=0.0)


class TestOneEuroFilterDampensJitter:
    def test_output_variance_is_much_smaller_than_input_variance(self):
        rng = np.random.default_rng(0)
        center = 500.0
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        dt = 1.0 / 30.0
        outputs = []
        t = 0.0
        for _ in range(90):  # 3 seconds at 30fps -- a voter "holding still"
            noisy = center + rng.normal(scale=8.0)  # +/- a few px of jitter
            outputs.append(f.filter(noisy, t))
            t += dt

        steady = outputs[20:]  # skip initial convergence
        assert np.std(steady) < 8.0 * 0.5  # meaningfully damped vs raw noise
        assert abs(np.mean(steady) - center) < 5.0

    def test_lower_min_cutoff_smooths_a_stationary_signal_more(self):
        rng = np.random.default_rng(1)
        dt = 1.0 / 30.0
        noise = [rng.normal(scale=10.0) for _ in range(60)]

        loose = OneEuroFilter(min_cutoff=5.0, beta=0.0)
        tight = OneEuroFilter(min_cutoff=0.3, beta=0.0)
        loose_out, tight_out = [], []
        t = 0.0
        for n in noise:
            loose_out.append(loose.filter(200.0 + n, t))
            tight_out.append(tight.filter(200.0 + n, t))
            t += dt

        assert np.std(tight_out[10:]) < np.std(loose_out[10:])


class TestOneEuroFilterTracksMovement:
    def test_adaptive_beta_reduces_lag_versus_fixed_beta_zero(self):
        """A fast, deliberate movement (large derivative) should be tracked
        more closely by a filter with beta > 0 than one with beta == 0 --
        that's the entire point of the adaptive cutoff."""
        dt = 1.0 / 30.0
        no_beta = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        with_beta = OneEuroFilter(min_cutoff=1.0, beta=0.02)

        t = 0.0
        no_beta.filter(0.0, t)
        with_beta.filter(0.0, t)

        target = 0.0
        no_beta_out = with_beta_out = 0.0
        for _ in range(10):
            t += dt
            target += 1500.0 * dt  # a fast, deliberate point: ~1500 px/s
            no_beta_out = no_beta.filter(target, t)
            with_beta_out = with_beta.filter(target, t)

        lag_no_beta = target - no_beta_out
        lag_with_beta = target - with_beta_out
        assert lag_with_beta < lag_no_beta
