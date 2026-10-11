#!/usr/bin/env bash
# Downloads MediaPipe Tasks model files (.task) into models/mediapipe/.
#
# InsightFace's buffalo_l pack is fetched automatically on first use by the
# `insightface` package itself (cached under ~/.insightface) -- nothing to
# download for that here.
set -euo pipefail

MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models/mediapipe"
mkdir -p "$MODELS_DIR"

# Official MediaPipe Tasks model URLs (see
# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker and
# .../face_landmarker for the current canonical links if these ever move).
HAND_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
FACE_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

download() {
  local url="$1" dest="$2"
  if [ -f "$dest" ]; then
    echo "already present: $dest"
    return
  fi
  echo "downloading: $url -> $dest"
  curl -fL "$url" -o "$dest"
}

download "$HAND_URL" "$MODELS_DIR/hand_landmarker.task"
download "$FACE_URL" "$MODELS_DIR/face_landmarker.task"

echo "Model files ready in $MODELS_DIR"
