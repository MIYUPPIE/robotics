#!/usr/bin/env python3
"""Gate tests for camguard detection logic. Deterministic, no GPIO/network.
Run: python3 test_camguard.py   (also works under pytest)"""
import numpy as np
from camguard import compute_metrics, is_covered, CoverDebouncer, DEFAULT_CFG

CFG = dict(DEFAULT_CFG)


def test_uniform_frame_has_no_detail():
    f = np.full((120, 160), 100, dtype=np.uint8)
    m = compute_metrics(f)
    assert m["detail"] == 0.0
    assert m["std"] == 0.0
    assert m["brightness"] == 100.0


def test_gradient_frame_has_detail():
    row = np.linspace(0, 255, 160).astype(np.uint8)
    f = np.tile(row, (120, 1))
    m = compute_metrics(f)
    assert m["detail"] > 0.0
    assert m["std"] > 0.0


def test_noise_frame_has_high_detail():
    rng = np.random.default_rng(0)
    f = rng.integers(0, 256, size=(120, 160), dtype=np.uint8)
    m = compute_metrics(f)
    assert m["detail"] > 50.0   # random noise -> large neighbor differences
    assert m["std"] > 50.0


def test_is_covered_true_for_flat_frame():
    f = np.full((120, 160), 30, dtype=np.uint8)
    assert is_covered(compute_metrics(f), CFG) is True


def test_is_covered_false_for_detailed_frame():
    rng = np.random.default_rng(1)
    f = rng.integers(0, 256, size=(120, 160), dtype=np.uint8)
    assert is_covered(compute_metrics(f), CFG) is False


def test_is_covered_needs_both_low_detail_and_low_std():
    # high std but flat-ish: half black half white -> low detail (one edge) but high std
    f = np.zeros((120, 160), dtype=np.uint8)
    f[:, 80:] = 255
    m = compute_metrics(f)
    # one sharp edge -> detail small but std large -> NOT covered
    assert m["std"] > CFG["std_thresh"]
    assert is_covered(m, CFG) is False


def test_debounce_requires_on_delay():
    d = CoverDebouncer(on_delay=2.0, off_delay=1.5)
    assert d.update(True, 0.0) is False
    assert d.update(True, 1.9) is False    # not long enough
    assert d.update(True, 2.0) is True     # crossed on_delay


def test_debounce_requires_off_delay():
    d = CoverDebouncer(on_delay=2.0, off_delay=1.5)
    d.update(True, 0.0); d.update(True, 2.0)
    assert d.state is True
    assert d.update(False, 2.5) is True    # just went clear at 2.5, not long enough
    assert d.update(False, 4.0) is False   # crossed off_delay (2.5 + 1.5)


def test_debounce_ignores_flicker():
    d = CoverDebouncer(on_delay=2.0, off_delay=1.5)
    t = 0.0
    for _ in range(20):              # rapid toggling, never stable 2s
        assert d.update(True, t) is False; t += 0.3
        assert d.update(False, t) is False; t += 0.3


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
