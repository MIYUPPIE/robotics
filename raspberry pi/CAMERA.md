# Ethernet PTZ camera on the okhub Pi

A Wi-Fi/IP PTZ camera is wired by ethernet straight into the Pi's `eth0` (point-to-point,
no router on that link). This is how to see it and how it was set up.

## Watch it
The Pi is headless (no monitor); the desktop runs over VNC. From this laptop (on **NewageG**):

```sh
xtigervncviewer 192.168.1.180:5900     # login: okhub + system password
```

The camera opens automatically in a normal VLC window (systemd `camview.service`,
auto-starts on boot). The window has a titlebar: minimize, resize, move, close all work.
Keyboard: `f` = fullscreen toggle, `Esc` = leave fullscreen, `q` = quit.

If you close it, reopen without SSH via the **Camera Feed** icon on the Pi desktop
(or `systemctl --user start camview` over SSH). The service only auto-restarts on a
network drop, not when you close it deliberately.

## The camera
- IP `192.168.0.104`, MAC `00:12:43:84:b1:49` — XMeye/Sofia (Hi35xx) firmware, statically configured.
- RTSP: user `admin`, **no password**. Quote the URL (literal `&`):
  - main 2304x2592 H.265: `rtsp://192.168.0.104:554/user=admin&password=&channel=1&stream=0.sdp?real_stream`
  - sub  640x720  H.265: same URL with `stream=1` (this is what the live view uses)

## How it was wired up on the Pi
1. `eth0` had no IP (direct link, no DHCP). Static IP via NetworkManager:
   - profile `cam-eth0` = `192.168.0.10/24`, autoconnect, priority 999.
   - `netplan-eth0` autoconnect disabled so DHCP can't claim eth0. Survives reboot.
2. Viewer `~/view-camera.sh` on the Pi: cvlc fullscreen on labwc/Wayland, waits for the
   `wayland-0` socket, ensures the eth0 IP. `./view-camera.sh main` for the full-res stream.
3. `camview.service` (systemd --user, enabled, linger on) runs the viewer on boot.

## Control (over SSH; set the runtime dir first)
```sh
ssh -4 okhub@192.168.1.180
export XDG_RUNTIME_DIR=/run/user/1000
systemctl --user restart camview     # restart the view
systemctl --user stop camview        # stop it
journalctl --user -u camview -n 50   # logs
```

## Notes
- Camera clock is wrong (OSD showed 2026-06-03). Cosmetic; fixing needs dvrip (port 34567)
  or ONVIF SetSystemDateAndTime.
- Launch GUI apps on the Pi's Wayland session over SSH with `systemd-run --user`, not
  `nohup &` — plain backgrounding drops the SSH session.
