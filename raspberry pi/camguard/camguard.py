#!/usr/bin/env python3
"""camguard - sound the GPIO17 buzzer when the camera lens is covered.

A covered/obscured lens makes the image lose spatial detail (blur) and go uniform.
We pull low-res grayscale frames from the camera via ffmpeg, measure detail + std,
debounce the covered/uncovered decision, and drive an active-high buzzer on GPIO17.

Modes:
  camguard.py run                 # the service: detect + buzz
  camguard.py calibrate [secs]    # print metric stats for the current scene
  camguard.py buzztest            # 3 short beeps, to confirm wiring
"""
import sys, os, time, json, argparse, subprocess, threading
import numpy as np

# ---------- config ----------
RTSP_URL = "rtsp://192.168.0.104:554/user=admin&password=&channel=1&stream=1.sdp?real_stream"
ANALYZE_W, ANALYZE_H = 160, 120
FPS = 4
GPIO_CHIP = 0          # Pi 4 BCM2711 40-pin header = gpiochip0
BUZZER_LINE = 17       # BCM GPIO17, active-high (HIGH = sound)
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib.json")

DEFAULT_CFG = {
    "detail_thresh": 6.0,   # covered when gradient detail drops below this
    "std_thresh": 12.0,     # ...and intensity spread is below this
    "on_delay": 2.0,        # seconds covered must persist before buzzing
    "off_delay": 1.5,       # seconds clear must persist before silencing
}


# ---------- pure, unit-tested logic ----------
def compute_metrics(gray):
    """gray: 2D uint8 ndarray. Returns brightness, std, detail (gradient energy)."""
    g = gray.astype(np.float32)
    dx = np.abs(np.diff(g, axis=1)).mean() if g.shape[1] > 1 else 0.0
    dy = np.abs(np.diff(g, axis=0)).mean() if g.shape[0] > 1 else 0.0
    return {
        "brightness": float(g.mean()),
        "std": float(g.std()),
        "detail": float(dx + dy),
    }


def is_covered(metrics, cfg):
    """Covered = lens obscured: low spatial detail AND low intensity spread.
    Both conditions guard against false positives from a merely dark scene."""
    return metrics["detail"] < cfg["detail_thresh"] and metrics["std"] < cfg["std_thresh"]


class CoverDebouncer:
    """Flip confirmed state only after a raw state persists long enough."""
    def __init__(self, on_delay, off_delay):
        self.on_delay = on_delay
        self.off_delay = off_delay
        self.state = False
        self._raw = False
        self._since = 0.0

    def update(self, raw_covered, now):
        if raw_covered != self._raw:
            self._raw = raw_covered
            self._since = now
        if raw_covered and not self.state and now - self._since >= self.on_delay:
            self.state = True
        elif not raw_covered and self.state and now - self._since >= self.off_delay:
            self.state = False
        return self.state


# ---------- I/O: config, frames, buzzer ----------
def load_cfg():
    cfg = dict(DEFAULT_CFG)
    try:
        with open(CONFIG_PATH) as f:
            cfg.update(json.load(f))
    except FileNotFoundError:
        pass
    return cfg


def frame_reader(url, w, h, fps):
    """Yield h x w uint8 grayscale frames from ffmpeg; respawn ffmpeg on failure."""
    frame_bytes = w * h
    while True:
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error",
            "-rtsp_transport", "tcp", "-i", url,
            "-an", "-vf", f"fps={fps},scale={w}:{h},format=gray",
            "-f", "rawvideo", "-",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break  # stream ended / hiccup -> respawn ffmpeg
                yield np.frombuffer(buf, dtype=np.uint8).reshape(h, w)
        finally:
            proc.kill()
            proc.wait()
        time.sleep(2)  # backoff before reconnect


class Buzzer:
    def __init__(self, chip=GPIO_CHIP, line=BUZZER_LINE, active_high=True):
        import lgpio
        self.lgpio = lgpio
        self.line = line
        self.active_high = active_high
        self.h = lgpio.gpiochip_open(chip)
        lgpio.gpio_claim_output(self.h, line, 0 if active_high else 1)

    def set(self, on):
        level = (1 if on else 0) if self.active_high else (0 if on else 1)
        self.lgpio.gpio_write(self.h, self.line, level)

    def close(self):
        try:
            self.set(False)
            self.lgpio.gpiochip_close(self.h)
        except Exception:
            pass


def buzzer_alarm_loop(buzzer, covered_flag, stop_event, on=0.3, off=0.3):
    """Intermittent beep while covered_flag is set; silent otherwise."""
    while not stop_event.is_set():
        if covered_flag["v"]:
            buzzer.set(True);  time.sleep(on)
            buzzer.set(False); time.sleep(off)
        else:
            buzzer.set(False)
            time.sleep(0.1)
    buzzer.set(False)


# ---------- modes ----------
def mode_run():
    cfg = load_cfg()
    print(f"[camguard] starting; cfg={cfg}", flush=True)
    buzzer = Buzzer()
    covered_flag = {"v": False}
    stop_event = threading.Event()
    t = threading.Thread(target=buzzer_alarm_loop, args=(buzzer, covered_flag, stop_event), daemon=True)
    t.start()
    deb = CoverDebouncer(cfg["on_delay"], cfg["off_delay"])
    last_log = 0.0
    try:
        for frame in frame_reader(RTSP_URL, ANALYZE_W, ANALYZE_H, FPS):
            m = compute_metrics(frame)
            now = time.monotonic()
            covered = deb.update(is_covered(m, cfg), now)
            covered_flag["v"] = covered
            if covered != covered_flag.get("logged") or now - last_log > 30:
                print(f"[camguard] covered={covered} detail={m['detail']:.1f} "
                      f"std={m['std']:.1f} bright={m['brightness']:.0f}", flush=True)
                covered_flag["logged"] = covered
                last_log = now
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        time.sleep(0.2)
        buzzer.close()


def mode_calibrate(secs):
    print(f"[calibrate] sampling {secs}s of the current scene ...", flush=True)
    vals = {"detail": [], "std": [], "brightness": []}
    end = time.monotonic() + secs
    for frame in frame_reader(RTSP_URL, ANALYZE_W, ANALYZE_H, FPS):
        m = compute_metrics(frame)
        for k in vals:
            vals[k].append(m[k])
        if time.monotonic() > end:
            break
    for k, arr in vals.items():
        a = np.array(arr)
        print(f"  {k:>10}: n={len(a)} min={a.min():.1f} mean={a.mean():.1f} "
              f"max={a.max():.1f} p10={np.percentile(a,10):.1f} p90={np.percentile(a,90):.1f}")


def mode_buzztest():
    b = Buzzer()
    for _ in range(3):
        b.set(True);  time.sleep(0.3)
        b.set(False); time.sleep(0.3)
    b.close()
    print("[buzztest] done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    cal = sub.add_parser("calibrate"); cal.add_argument("secs", nargs="?", type=float, default=5.0)
    sub.add_parser("buzztest")
    args = ap.parse_args()
    if args.cmd == "run":
        mode_run()
    elif args.cmd == "calibrate":
        mode_calibrate(args.secs)
    elif args.cmd == "buzztest":
        mode_buzztest()
