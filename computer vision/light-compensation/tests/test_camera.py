import cv2
import numpy as np
import pytest
from src.camera import V4L2_AUTO, V4L2_MANUAL, Camera, CameraError
from src.config import Settings


class FakeCap:
    """Minimal V4L2-like capture: manual mode sticks only if `accepts_manual`."""

    def __init__(self, opened=True, exposure=157.0, gain=-1.0, accepts_manual=True):
        self.opened = opened
        self.props = {
            cv2.CAP_PROP_AUTO_EXPOSURE: V4L2_AUTO,
            cv2.CAP_PROP_EXPOSURE: exposure,
            cv2.CAP_PROP_GAIN: gain,
        }
        self.accepts_manual = accepts_manual
        self.released = False
        self.reads = 0

    def isOpened(self):
        return self.opened

    def read(self):
        self.reads += 1
        return True, np.zeros((480, 640, 3), np.uint8)

    def get(self, prop):
        return self.props.get(prop, 0.0)

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_AUTO_EXPOSURE and value == V4L2_MANUAL and not self.accepts_manual:
            return False
        self.props[prop] = value
        return True

    def release(self):
        self.released = True


def make(cap, **kw):
    cam = Camera(
        Settings(exposure_warmup_s=0.0, **kw), capture_factory=lambda i: cap, sleep=lambda s: None
    )
    cam.open()
    return cam


def test_open_failure():
    cam = Camera(Settings(), capture_factory=lambda i: FakeCap(opened=False))
    with pytest.raises(CameraError):
        cam.open()


def test_read_before_open():
    with pytest.raises(CameraError):
        Camera(Settings()).read()


def test_open_sets_geometry_and_reads():
    cap = FakeCap()
    cam = make(cap, width=800, height=600)
    assert cap.props[cv2.CAP_PROP_FRAME_WIDTH] == 800
    assert cam.read().shape == (480, 640, 3)


def test_lock_after_warmup_records_converged_exposure():
    cap = FakeCap(exposure=157.0)
    st = make(cap).lock_exposure()
    assert st.locked and st.exposure == 157.0 and st.gain is None
    assert "gain control not supported" in st.note
    assert cap.props[cv2.CAP_PROP_AUTO_EXPOSURE] == V4L2_MANUAL


def test_lock_with_gain_support():
    cap = FakeCap(gain=32.0)
    st = make(cap).lock_exposure()
    assert st.locked and st.gain == 32.0 and st.note == "locked"


def test_saved_values_are_reapplied_without_warmup():
    cap = FakeCap(exposure=999.0)
    st = make(cap).lock_exposure({"locked": True, "exposure": 120.0, "gain": 10.0})
    assert st.exposure == 120.0 and cap.props[cv2.CAP_PROP_EXPOSURE] == 120.0
    assert cap.props[cv2.CAP_PROP_GAIN] == 10.0


def test_unusable_saved_state_falls_back_to_warmup():
    cap = FakeCap(exposure=157.0)
    st = make(cap).lock_exposure({"locked": False, "exposure": -1.0})
    assert st.locked and st.exposure == 157.0


def test_driver_without_exposure_is_reported_not_locked():
    st = make(FakeCap(exposure=-1.0)).lock_exposure()
    assert not st.locked and "no exposure" in st.note


def test_camera_refusing_manual_mode_is_reported():
    st = make(FakeCap(accepts_manual=False)).lock_exposure()
    assert not st.locked and "refused" in st.note


def test_lock_disabled_by_setting():
    cap = FakeCap()
    st = make(cap, lock_exposure=False).lock_exposure()
    assert not st.locked and st.note == "disabled"
    assert cap.props[cv2.CAP_PROP_AUTO_EXPOSURE] == V4L2_AUTO


def test_release_restores_original_auto_mode():
    cap = FakeCap()
    cam = make(cap)
    cam.lock_exposure()
    assert cap.props[cv2.CAP_PROP_AUTO_EXPOSURE] == V4L2_MANUAL
    cam.release()
    assert cap.props[cv2.CAP_PROP_AUTO_EXPOSURE] == V4L2_AUTO and cap.released
    cam.release()  # idempotent
