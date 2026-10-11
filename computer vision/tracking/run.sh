#!/usr/bin/env bash
# Single entry point: starts a local MQTT broker (if mosquitto is installed
# and not already running) and launches the full tracking pipeline.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if command -v mosquitto >/dev/null 2>&1; then
    if ! pgrep -x mosquitto >/dev/null 2>&1; then
        echo "[run.sh] starting local mosquitto broker on 1883"
        mosquitto -d
    fi
else
    echo "[run.sh] mosquitto not installed -- alerts will be logged only, not delivered"
    echo "[run.sh] install with: sudo apt install mosquitto"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tracking-cv

# pip's opencv-python doesn't bundle a Wayland Qt platform plugin, so
# cv2.imshow can't find one to render on under a native Wayland session
# (XDG_SESSION_TYPE=wayland). Force the XWayland-compatible xcb backend
# instead -- confirmed working on this machine's Wayland session.
export QT_QPA_PLATFORM=xcb

python tracker1.py
