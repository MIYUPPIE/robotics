#!/bin/sh
# view-camera.sh - show the ethernet-connected PTZ camera on the Pi desktop.
# Window appears on labwc (Wayland) and is visible over VNC (xtigervncviewer <pi-ip>:5900).
#
# Camera: 192.168.0.104, direct on eth0 (192.168.0.0/24), XMeye/Sofia firmware,
#         RTSP user=admin, no password. main=6MP (stream 0), sub=640x720 (stream 1).
#
# Usage: ./view-camera.sh         # substream (default, light)
#        ./view-camera.sh main    # full 2304x2592 stream
#
# Opens a normal decorated VLC window (labwc titlebar): minimize / resize / close all work.
# Double-click or press 'f' to toggle fullscreen, Esc to leave fullscreen, 'q' to quit.

export XDG_RUNTIME_DIR=/run/user/1000
export WAYLAND_DISPLAY=wayland-0
export QT_QPA_PLATFORM=wayland
export GDK_BACKEND=wayland

if [ "$1" = "main" ]; then S=0; else S=1; fi
URL="rtsp://192.168.0.104:554/user=admin&password=&channel=1&stream=${S}.sdp?real_stream"

# wait for the labwc Wayland socket (handles boot/service race before desktop is up)
i=0
while [ ! -S "$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY" ] && [ "$i" -lt 30 ]; do
    i=$((i + 1)); sleep 1
done

# ensure eth0 is on the camera's subnet (no-op if the NetworkManager profile already did it)
ip -4 addr show eth0 2>/dev/null | grep -q "192.168.0." || sudo ip addr add 192.168.0.10/24 dev eth0

pkill -f 'vlc.*192.168.0.104' 2>/dev/null || true
sleep 0.5
# Full Qt interface = a real labwc-decorated window (titlebar with minimize/close),
# working keyboard shortcuts, and a control bar. Not fullscreen, so it can be managed.
exec vlc --no-qt-privacy-ask --no-qt-error-dialogs \
     --no-video-title-show --no-qt-system-tray \
     --network-caching=300 --rtsp-tcp --avcodec-hw=any "$URL"
