#!/bin/bash
# Hand Meme — one-click launcher (macOS).
# Double-click this file in Finder. First run installs Python deps and downloads
# MediaPipe models (~12MB). Subsequent runs start instantly.

set -e
cd "$(dirname "$0")"

echo "================================================="
echo "          HAND MEME — one-click launcher          "
echo "================================================="
echo

find_python() {
  PY=""
  # Common system locations first (matches what the python.org installer drops in).
  for cmd in \
      /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
      /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
      /Library/Frameworks/Python.framework/Versions/3.10/bin/python3 \
      python3.12 python3.11 python3.10 python3.9 python3 python; do
    if command -v "$cmd" >/dev/null 2>&1 && \
       "$cmd" -c "import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)" >/dev/null 2>&1; then
      PY="$cmd"
      return 0
    fi
  done
  return 1
}

# --- 1) Find a usable Python 3.9+, install the bundled one if missing ---
echo "[1/4] Looking for Python 3.9 or newer..."
if find_python; then
  echo "    Found: $($PY --version)"
else
  echo "    Not found."
  echo
  if [ -f "python-installer.pkg" ]; then
    echo "    Found a bundled installer (python-installer.pkg). Launching it now."
    echo "    A macOS installer window will open — click through it:"
    echo "      Continue → Continue → Continue → Agree → Install → (Mac password) → Close"
    echo
    echo "    This script is waiting; it'll continue once you close the installer."
    echo
    open -W "python-installer.pkg"
  else
    echo "    No bundled installer found. Opening the Python.org download page in"
    echo "    your browser — download and run the .pkg, then double-click start.command"
    echo "    again."
    echo
    open "https://www.python.org/downloads/macos/"
    read -n 1 -s -r -p "Press any key to close..."
    exit 0
  fi
  echo "    Installer closed. Re-checking for Python..."
  if find_python; then
    echo "    Found: $($PY --version)"
  else
    echo
    echo "ERROR: Python still not detected. Try running this file again, or install"
    echo "       Python manually from https://www.python.org/downloads/"
    read -n 1 -s -r -p "Press any key to close..."
    exit 1
  fi
fi

# --- 2) Create venv if missing ---
echo "[2/4] Setting up local virtual environment (.venv/)..."
if [ ! -d ".venv" ]; then
  "$PY" -m venv .venv
  echo "    Created .venv/"
else
  echo "    Already exists."
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# --- 3) Install dependencies if missing ---
echo "[3/4] Checking Python dependencies..."
if ! python -c "import cv2, mediapipe, numpy" >/dev/null 2>&1; then
  echo "    First-time install of opencv-python + mediapipe + numpy (~150MB, takes a few minutes)..."
  python -m pip install --upgrade pip >/dev/null
  python -m pip install opencv-python mediapipe numpy
else
  echo "    All present."
fi

# --- 4) Download MediaPipe models if missing ---
echo "[4/4] Checking MediaPipe model files..."
download_if_missing() {
  local file="$1"
  local url="$2"
  if [ ! -f "$file" ]; then
    echo "    Downloading $file ..."
    curl -fsSL -o "$file" "$url"
  fi
}
download_if_missing "gesture_recognizer.task" \
  "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
download_if_missing "face_landmarker.task" \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
echo "    Models ready."

echo
echo "All set. Launching Hand Meme — press 'q' or close the window to quit."
echo
echo "NOTE: macOS will prompt for camera permission the first time. After granting"
echo "      permission, fully quit your terminal (Cmd+Q) and double-click this file again."
echo

python hand_meme.py

echo
read -n 1 -s -r -p "App closed. Press any key to exit..."
